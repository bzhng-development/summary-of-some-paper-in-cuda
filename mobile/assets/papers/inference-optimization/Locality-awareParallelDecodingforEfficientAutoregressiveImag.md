# Locality-aware Parallel Decoding for Efficient Autoregressive Image Generation

**ArXiv:** [2507.01957](https://arxiv.org/abs/2507.01957)

## 🎯 Pitch

This paper introduces Locality-aware Parallel Decoding (LPD), a novel method that dramatically accelerates autoregressive image generation by enabling flexible, parallel prediction of many image patches at once—guided by spatial locality to maximize contextual coherence. By reducing the number of sequential steps from hundreds to just a few dozen without sacrificing image quality, LPD achieves over 3.4× lower latency than prior parallel autoregressive models, making fast, high-quality image synthesis practical for real-world and multimodal AI applications while preserving compatibility with widely used vision foundation models.

---

## 1. Executive Summary

This paper introduces **Locality-aware Parallel Decoding (LPD)**, a framework that accelerates autoregressive image generation by dramatically reducing the number of sequential decoding steps without sacrificing quality. The approach combines two named mechanisms: **Flexible Parallelized Autoregressive Modeling**, a novel transformer architecture that decouples context representation from token generation using learnable position query tokens to enable arbitrary generation ordering and parallel decoding with mutual visibility among concurrently generated tokens (analogous to joint rather than batched next-token prediction), and **Locality-aware Generation Ordering**, a schedule that selects parallel decoding groups by maximizing spatial proximity to already-generated context tokens while minimizing spatial proximity among tokens predicted in the same step (operationalized via a Euclidean-distance-based selection algorithm with proximity and repulsion thresholds). On ImageNet 256×256 class-conditional generation, LPD reduces generation steps from 256 to 20—a 12.8× reduction—while achieving an FID of 2.00–2.40 without compromising quality, delivering at least 3.4× lower latency than previous parallelized autoregressive models (e.g., LPD-XL achieves 2.10 FID in 20 steps at 0.41s versus ARPG-XL at 1.71s). The framework establishes that high degrees of parallelization are achievable in flat-token autoregressive visual generation only when the architecture both ensures mutual visibility among concurrently predicted tokens and the generation order respects the strong spatial locality inherent in image attention patterns.

## 2. Context and Motivation

### The Problem: Autoregressive Image Generation Is Too Slow

The paper addresses a fundamental tension in autoregressive visual generation. On one hand, autoregressive modeling—where images are tokenized into discrete patches and generated one by one in sequence—has proven remarkably effective. It scales well with data and compute (Section 1), achieves state-of-the-art results, and, critically, uses the same flat token representation as language models. This flatness matters enormously for practical deployment: it makes autoregressive image models directly compatible with widely used vision perception backbones like CLIP (Radford et al., 2021) and DINO (Caron et al., 2021), which in turn makes them natural building blocks for **unified multimodal systems** that handle both visual understanding and generation (Wu et al., 2024c; Ma et al., 2025; Jiao et al., 2025; Song et al., 2025; Chen et al., 2025b; Zhao et al., 2025; Lin et al., 2025). The alternative—multi-scale token representations used by next-scale prediction methods like VAR (Tian et al., 2024)—breaks this compatibility because their tokens don't map to the flat spatial grid that perception models expect (Section 1, paragraph 3).

On the other hand, the standard formulation of autoregressive image generation is **painfully slow at inference time**. The dominant paradigm is *next-patch prediction*: an image is split into $N$ patches (e.g., $16 \times 16 = 256$ for a 256×256 image with a downsampling factor of 16), the patches are flattened into a 1D sequence, and the model generates one patch per forward pass—256 sequential steps total. For a 512×512 image, that number jumps to $32 \times 32 = 1024$ steps.

This isn't merely an inconvenience. The paper identifies the bottleneck as **memory-bound workload** (Section 1, footnote 1). Here's what that means in practice: each generation step requires loading the entire set of model parameters from GPU memory (HBM) into the compute units (registers). The actual computation—multiplying a few new token embeddings through the transformer—is relatively cheap. The dominant cost is the memory transfer. So you pay the full parameter-loading cost 256 or 1024 times per image, but each time you only do a tiny amount of computation. The GPU's arithmetic units sit mostly idle while waiting on memory bandwidth. Latency therefore scales linearly with the number of steps, and throughput is severely constrained no matter how many FLOPs your GPU can theoretically deliver.

This is the core gap: **how do you get the benefits of autoregressive modeling (flat tokens, KV caching, compatibility with language-model infrastructure) without paying the prohibitive latency cost of next-patch prediction?**

### Why This Problem Matters

The paper's framing makes clear that this isn't just an academic optimization problem—it has direct practical consequences for the direction of multimodal AI systems.

**Unified multimodal models are converging on autoregressive architectures.** The paper cites a long list of recent works (Section 1) that build systems capable of both understanding images and generating them, all using autoregressive transformers as the backbone. This convergence is driven by the fact that autoregressive modeling is the dominant paradigm for text (the "language" half of multimodal), and using the same architecture for vision simplifies training, deployment, and scaling. But if the vision side costs 256× or 1024× more steps than the text side for the same amount of content, the latency mismatch becomes a practical barrier to real-time multimodal interaction. Reducing generation steps while maintaining flat-token compatibility is therefore a prerequisite for practical unified models.

**The scaling trend amplifies the problem.** As models get larger (the paper trains up to 1.4B parameters; production systems are far larger), the per-step memory transfer cost grows proportionally. A 1.4B parameter model in BFloat16 requires roughly 2.8 GB just for parameters—loading that 1024 times to generate a single 512×512 image is enormously wasteful. Table 1 quantifies this: the raster-order baseline for LPD-XL (752M parameters) takes 5.29 seconds per 256×256 image at batch size 1 on an A100. That's an eternity in interactive applications.

**Existing acceleration approaches have fundamental limitations.** The paper identifies two existing strategies for speeding up autoregressive image generation, both of which fall short:

1. **Next-scale prediction (VAR and derivatives, Tian et al., 2024):** Rather than generating pixel-level or patch-level tokens, these methods generate images coarse-to-fine, predicting the next resolution scale at each step. This dramatically reduces steps (VAR-d16 uses only 10 steps for 256×256 images, Table 1), but the multi-scale token representation is incompatible with flat-token vision backbones. You can't feed a VAR token sequence into CLIP or DINO without conversion, which breaks the unified multimodal vision. The paper is explicit about this: "its multi-scale token representation fundamentally differs from the universal flat token representation, making it incompatible with widely used flat vision perception foundation models" (Section 1). This is the tradeoff this paper refuses to make—it wants both speed and flat-token compatibility.

2. **Existing parallelized autoregressive approaches (PAR, RandAR, ARPG, NAR):** These methods attempt to predict multiple patches per step while keeping the autoregressive framework and flat tokens. But the paper argues they all have critical architectural weaknesses that limit how much parallelization they can achieve before quality collapses. We'll examine each in detail in the next section, but the key takeaway from Table 1 is that the best prior parallelized autoregressive model at comparable size (ARPG-XL at 719M parameters) still requires 64 steps and 1.71 seconds to achieve a 2.10 FID, while LPD-XL achieves the same FID in 20 steps at 0.41 seconds—a 4.2× latency reduction. The prior methods simply couldn't push below ~30–60 steps without significant quality degradation.

### Where Prior Approaches Fall Short

The paper's analysis of existing parallel generation methods is unusually systematic. Rather than just saying "previous work is worse," it identifies specific architectural and algorithmic limitations that explain *why* more aggressive parallelization has been impossible.

#### The Token-Level Coupling Problem in Standard Decoder-Only Models

The paper argues that standard decoder-only autoregressive transformers (GPT-style, which dominate visual generation through models like LLAMAGEN, Sun et al., 2024) have a fundamental limitation: **each token simultaneously serves as both context and output** (Section 2.1). In next-token prediction, token $x_i$ provides its hidden state as context for predicting $x_{i+1}$, but $x_i$ was itself generated at the previous step. This coupling means the generation order is baked into the architecture—you can't easily say "generate tokens at positions (3,7) and (12,4) simultaneously" because those tokens need to attend to each other in ways the causal mask doesn't permit.

To understand why this matters for parallelization, consider what happens if you try to naively group tokens: you predict the first $m$ tokens, then the next $m$ tokens, and so on, in $N/m$ steps. This is what PAR (Wang et al., 2024b) does with its fixed region-wise parallel scheme. The problem, as the paper explains (Section 2.1), is that "spatially adjacent tokens exhibit strong mutual dependencies, and independent sampling usually leads to generation inconsistencies inside a group." If token $x_5$ and $x_6$ are adjacent patches in the image, predicting them simultaneously without letting them see each other during generation produces visible artifacts—texture discontinuities, misaligned edges, color shifts. The standard causal mask prevents concurrent tokens from attending to each other, so each is generated as if the others don't exist, yet they're supposed to form a coherent local region.

This problem is not hypothetical. Table 1 shows PAR-L-4× at 147 steps (not even very aggressive parallelization) achieving 3.76 FID, significantly worse than the 2.48 FID of the LPD-L raster counterpart at 256 steps. More parallelization would only worsen the inconsistency.

#### The Independence Problem in Encoder-Decoder Approaches

ARPG (Li et al., 2025a) and SAR (Liu et al., 2024b) attempt to solve the ordering problem by switching to an encoder-decoder architecture. The encoder processes previously generated tokens into a KV cache, and the decoder uses target-aware query tokens that attend to this cache via cross-attention. This decouples context from generation, enabling arbitrary target positions.

But the paper identifies a subtle and critical weakness (Section 2.2, Figure 6a): **the target positions themselves never contribute key-value pairs.** When you predict tokens at positions (1,2) and (3,5) simultaneously, the decoder generates each independently because neither can see the other's partially-formed representation. There is no mutual visibility. This is the same inconsistency problem that plagues PAR, just in a different architectural form. The paper shows this empirically in the ablation (Figure 9a): ARPG's FID degrades significantly as generation steps decrease (parallelization increases), from ~2.0 at 256 steps to well above 2.4 at 16 steps.

#### The Batched Next-Token Problem in Decoder-Only Random-Order Approaches

RandAR (Pang et al., 2024) tries a different approach: it stays within the decoder-only architecture but inserts positional instruction tokens that tell the model which positions to generate next. This enables arbitrary generation orders—you can instruct the model to generate positions in any sequence.

However, the paper points out two critical failures (Section 2.2, Figure 6b):

1. **The causal mask reduces parallel generation to batched next-token prediction, not joint prediction.** Because RandAR still uses a standard causal mask, when you try to predict multiple positions simultaneously—say, positions A, B, and C—the model processes them left-to-right within the same forward pass. Position B can see position A's representation, but A cannot see B's. Position C can see both A and B, but neither can see C's. This is asymmetric: it's equivalent to generating A, then B given A, then C given A and B, but doing it in one forward pass by computing them in parallel. The tokens are not jointly predicted; they're sequentially predicted within a single step. This breaks the symmetry that coherent parallel generation requires.

2. **The positional instruction tokens must be stored in the KV cache.** Since these instruction tokens go through the transformer and their hidden states become part of the context for future steps, they consume KV cache memory. The paper notes this "doubl[es] the memory consumption" (Section 2.2) because for every real image token in the cache, there's a corresponding instruction token. For large models generating high-resolution images, this is a significant practical constraint.

The ablation (Figure 9a) confirms the consequence: RandAR-XL's FID climbs from ~2.2 at 88 steps (their reported configuration) to well above 2.6 at 16 steps when forced to match LPD's aggressive parallelization.

#### The Missing Principle: Locality-Aware Ordering

Beyond the architectural limitations, the paper identifies a deeper issue: **no prior work explicitly designs the generation order to respect the spatial locality of image attention.** This insight is grounded in a quantitative analysis of attention patterns in the widely-used LLAMAGEN model (Section 2.3, Figures 2, 7, 10, 11).

The analysis reveals that attention in autoregressive image generation is **sharply localized**. Using the per-token attention metric (Equation 3), the paper shows that the average attention a decoding token pays to another token drops off precipitously with spatial distance on the 2D image grid. At relative distance 1 (immediate neighbors), per-token attention is roughly 6–8%; by distance 5, it's below 1-2%. This pattern holds consistently across all attention heads and model sizes (Figure 7b).

What does this mean for design? Two principles emerge:

- **Tokens should be generated close to already-generated context.** A new token at position $(i,j)$ will receive far stronger conditioning from an existing token at $(i, j+1)$ than from one at $(i+10, j+10)$. Generating tokens near existing context maximizes the useful information available for prediction.

- **Tokens generated simultaneously should be spatially distant from each other.** If you predict two adjacent patches in parallel, they can't condition on each other, yet they're supposed to form a coherent local region—texture, edge, or color. Their mutual dependency is high, so the inconsistency cost is large. Generating spatially separated tokens minimizes this dependency, since tokens far apart on the grid have low attention to each other and therefore don't need mutual visibility as badly.

Prior ordering schemes violate one or both principles:
- **Raster order** keeps new tokens adjacent to context (good for principle 1) but also adjacent to concurrently generated tokens in group-parallel variants (bad for principle 2).
- **Random order** produces some distant pairs but also some adjacent ones, with no systematic optimization.
- **Halton order** (Besnier et al., 2025) uses a low-discrepancy sequence to spread tokens evenly across the image at each step. This helps with principle 2 (concurrent tokens are well-separated) but ignores principle 1 entirely—the Halton sequence doesn't care whether a token is near existing context, so it often selects positions far from any generated tokens, providing weak conditioning.

The ablation (Figure 9b) quantifies this: at 20 steps with LPD-XL, locality-aware ordering achieves roughly 2.10 FID versus ~2.25 for Halton and ~2.35 for random. The gap widens at more aggressive parallelization (fewer steps), confirming that systematic ordering becomes more important as concurrent group sizes grow.

### How This Paper Positions Itself

The paper positions LPD not as an incremental improvement over existing parallelized autoregressive methods, but as a **framework that resolves the fundamental tension** between aggressive parallelization and generation quality. The two contributions are designed to address the two root causes identified above:

1. **Flexible Parallelized Autoregressive Modeling** solves the architectural problem. By decoupling context tokens from generation tokens (using learnable position query tokens that guide generation at target positions while previous tokens provide context), the architecture enables arbitrary generation orders and, critically, **guarantees mutual visibility among all concurrently predicted tokens** via a specialized attention mask (Section 2.2, Figure 6c). This means tokens in the same parallel group can condition on each other during generation, eliminating the inconsistency problem that cripples PAR, ARPG, and RandAR at high parallelization. The architecture also avoids the KV-cache doubling of RandAR by not caching query tokens, and inherits the efficient KV-caching mechanism for context tokens.

2. **Locality-aware Generation Ordering** solves the scheduling problem. Rather than using a fixed order (raster, random, Halton), the algorithm (Algorithm 1) explicitly optimizes the generation sequence to maximize spatial proximity to existing context (principle 1) while minimizing spatial proximity among concurrent tokens (principle 2). It does this via a two-stage selection procedure: first, high-proximity candidates (distance < $\tau$) are selected greedily with a repulsion constraint ($\rho$) to keep them separated; if more tokens are needed, farthest-point sampling fills the remaining slots. The order is precomputed offline, so inference incurs zero scheduling overhead.

The paper also positions itself in the broader landscape through a clear taxonomy (Section 2.2, "Comparison with other methods," and Figure 6). It distinguishes LPD from:
- **Encoder-decoder methods** (SAR, ARPG): LPD avoids their independence problem through mutual visibility.
- **Decoder-only random-order methods** (RandAR): LPD enables true joint prediction rather than batched sequential prediction, and avoids KV-cache doubling.
- **Fixed-parallel-scheme methods** (PAR, NAR, ZipAR): LPD supports arbitrary generation orders rather than being locked into a rigid pattern, enabling the locality-aware scheduling that maintains quality.
- **Masked prediction methods** (MaskGIT, MAGVIT-v2): LPD remains autoregressive with causal attention and KV caching, avoiding the computationally expensive bidirectional attention and lack of caching in mask-prediction models.
- **Next-scale prediction methods** (VAR): LPD preserves flat token representations, maintaining compatibility with vision perception backbones—a constraint the paper treats as non-negotiable for unified multimodal systems.

The paper's claim is not just that LPD is faster or better than a specific baseline, but that it establishes the necessary conditions—mutual visibility in the architecture and locality-aware ordering in the schedule—for achieving high degrees of parallelization in flat-token autoregressive image generation without quality collapse. The 12.8× step reduction (256 → 20) and 3.4× latency improvement are presented as evidence that these conditions are both necessary and sufficient: previous methods lacked one or both, which is why they couldn't achieve comparable parallelization.

## 3. Technical Approach

### 3.1 Reader Orientation

This paper builds a **parallel decoding framework for autoregressive image generation** — a system that takes a class label (or text prompt) and produces a complete image in a small number of sequential steps by generating many image patches simultaneously at each step, rather than one patch at a time. The core problem it solves is that standard autoregressive image generation is painfully slow because it requires hundreds or thousands of sequential forward passes, yet existing attempts to parallelize it have caused quality collapse when pushed to aggressive parallelization ratios. The solution takes the shape of two coordinated mechanisms: a **new transformer architecture** that lets the model generate arbitrary sets of patches in parallel while maintaining mutual visibility among them, and a **spatial scheduling algorithm** that chooses *which* patches to generate together based on a quantitative analysis of attention locality in image transformers.

### 3.2 Big-Picture Architecture (Diagram in Words)

The LPD framework has four major components, used in two phases (training and inference):

**Training phase:**

1. **Flexible Parallelized Autoregressive Transformer** — the core neural network. It takes as input a sequence that interleaves ground-truth image tokens (providing context) with learnable position query tokens (marking which positions to predict next). A specialized two-pattern attention mask enforces causal context attention for the image tokens while enabling full mutual visibility among position query tokens within the same parallel step. The training objective is standard next-token prediction: each position query token is trained to output the correct image token for its target spatial location.

2. **Randomized Generation Order Sampling** — during training, the image tokens are randomly shuffled (with the class token fixed at the beginning), and the number of decoding steps is randomly sampled from a predefined set (e.g., {8, 12, 16, …, 256}). The tokens-per-step follow a cosine schedule. This teaches the model to handle arbitrary generation orders and varying degrees of parallelization.

**Inference phase (image generation):**

3. **Locality-aware Generation Order Schedule** (precomputed offline) — given the image resolution and desired number of decoding steps K, Algorithm 1 produces an ordered list of K groups of spatial positions. Each group specifies which patches to generate in parallel at that step. The schedule maximizes spatial proximity to already-generated tokens while minimizing proximity among tokens within the same group, using a two-stage selection (proximity-prioritized greedy selection with repulsion filtering, followed by farthest-point sampling for remaining slots).

4. **Fused Encoding-Decoding Inference Loop** — at each step, the model simultaneously (a) encodes newly generated image tokens from the previous step into the KV cache (providing updated context) and (b) decodes the next group of tokens by feeding position query tokens that attend to the updated KV cache. A specialized inference attention mask (Figure 5) enables this fusion into a single forward pass per step, avoiding the 2× step penalty that naive sequential encode-then-decode would incur.

**Information flow at inference:** Class label (or text embedding) enters → position query tokens for step 1's target positions are constructed by adding spatial positional embeddings to a shared learnable query embedding → query tokens attend to the class token via context attention and to each other via query attention → output logits are sampled to produce image tokens for step 1's positions → sampled tokens are encoded into the KV cache → position query tokens for step 2's target positions attend to the class token AND all step 1 image tokens in the cache AND each other → output logits produce step 2's image tokens → encode into KV cache → repeat for K steps → complete image.

### 3.3 Roadmap for the Deep Dive

- **First: the mathematical formulation** of parallelized autoregressive modeling (Equations 1–2), which generalizes standard next-token prediction to group-wise joint prediction and establishes the formal requirements that any parallelized architecture must satisfy.
- **Second: the attention mechanism design** (the core architectural innovation), because it is the enabler that makes the formal requirements achievable — the two-pattern attention mask (context attention + query attention) and the training/inference attention patterns that implement mutual visibility while preserving KV caching.
- **Third: the position query token mechanism**, which decouples context representation from generation and enables arbitrary target position specification without the architectural coupling that limits standard decoder-only models.
- **Fourth: the locality analysis and design principles**, since the generation order schedule is grounded in empirical measurements of attention locality — understanding *why* the two principles exist is necessary to understand the scheduling algorithm.
- **Fifth: the locality-aware generation order algorithm** (Algorithm 1), which operationalizes the two principles into a concrete scheduling procedure with explicit hyperparameters.
- **Sixth: the training procedure and inference pipeline**, which tie the architecture and schedule together into an end-to-end system.

### 3.4 Detailed, Sentence-Based Technical Breakdown

This is primarily a **systems and algorithms paper** whose core idea is that aggressive parallelization of autoregressive image generation requires both (a) an architecture that enables true joint prediction of concurrently generated tokens with mutual visibility, and (b) a generation order that respects the spatial locality of image attention to maximize useful conditioning while minimizing costly intra-group dependencies. The paper provides both components and demonstrates that their combination achieves parallelization ratios (256:20 = 12.8× step reduction) that previous methods could not reach without quality collapse.

---

#### Formalizing Parallelized Autoregressive Modeling

The paper begins by establishing the mathematical framework that defines what parallelized autoregressive modeling must achieve. This is not merely notation — it makes explicit the factorization structure that any architecture must implement and clarifies why naive grouping in raster order fails.

**Standard autoregressive factorization (next-patch prediction).** Given N image tokens $x_1, x_2, ..., x_N$ and a condition $c$ (e.g., class label or text embedding), the standard autoregressive model factorizes the joint distribution as:

$$p(x_1, x_2, \ldots, x_N; c) = \prod_{n=1}^{N} p(x_n \mid x_{<n}; c)$$

where $x_{<n}$ denotes all tokens before position $n$ in the sequence, and $c$ is the conditioning signal (class label, text embedding).

**What this computes:** The probability of an entire image is decomposed into a product of N conditional probabilities, each predicting exactly one token given all previously generated tokens and the condition. During sampling, you compute $p(x_1 | c)$, sample $x_1$, then compute $p(x_2 | x_1, c)$, sample $x_2$, and so on — N sequential forward passes.

**Why this form:** This is the standard chain rule of probability, applied with a predefined ordering (typically raster scan). It makes the high-dimensional joint distribution tractable by breaking it into single-token predictions. However, the sequential nature is baked in — you cannot skip ahead to $x_{50}$ without first generating $x_1$ through $x_{49}$, because each conditional depends on all preceding tokens.

**Group-wise parallel factorization.** To reduce the number of sequential steps from N to G (where $G \ll N$), the paper partitions the tokens into G disjoint groups $X_1, X_2, ..., X_G$, where each group $X_g = \{x_g^1, x_g^2, \ldots, x_g^m\}$ contains m tokens that are predicted jointly:

$$p(x_1, x_2, \ldots, x_N; c) = \prod_{g=1}^{G} p(X_g \mid X_{<g}; c)$$

where $X_{<g}$ denotes all tokens in groups before g (i.e., all previously generated tokens).

**What this computes:** The joint distribution is now a product of G group-conditionals. Each group-conditional $p(X_g \mid X_{<g}; c)$ predicts multiple tokens simultaneously — for example, predicting 13 patches in one forward pass instead of one. During sampling, you generate all tokens in $X_1$ in parallel (conditioned on c), then all tokens in $X_2$ in parallel (conditioned on c and $X_1$), and so on, requiring only G sequential steps.

**Why this form:** This is the key generalization — it's still a valid autoregressive factorization (each group only conditions on previous groups, maintaining the causal structure that enables efficient inference), but the per-step output is a set rather than a singleton. The group sizes $|X_g|$ can vary and should generally increase as G grows, because the growing context $X_{<g}$ provides stronger conditioning, allowing more tokens to be predicted reliably in parallel.

**The failure mode of naive grouping.** The paper states a critical empirical finding: "Previous work has shown that directly grouping tokens in raster order causes significant performance degradation (Wang et al., 2024b; Pang et al., 2024)." The reason is spatial locality of dependencies. If, in raster order, tokens $x_{17}$ and $x_{18}$ are adjacent image patches (e.g., neighbors in the 2D grid), they have strong mutual dependencies — each provides substantial information for predicting the other. When generated together in the same group, neither can condition on the other's actual value (they are predicted simultaneously), so the model must predict each based only on earlier tokens that may be spatially distant. This independence assumption is violated for adjacent patches, producing visible inconsistencies — texture breaks, edge misalignments, color discontinuities.

This motivates the paper's two design requirements stated in Section 2.1: a parallelized autoregressive model must support **(1) flexible generation order** (so adjacent tokens can be placed in different groups, avoiding simultaneous prediction) and **(2) dynamic group sizes** (so later groups can be larger, leveraging accumulated context). The rest of the technical approach is about satisfying these requirements while maintaining the flat-token representation and autoregressive efficiency properties.

---

#### The Core Architectural Innovation: Decoupling Context from Generation

The paper identifies a fundamental limitation of standard decoder-only transformers for group-wise parallel prediction: **the dual role of each token.** In next-token prediction (Figure 3a), each token $x_i$ simultaneously serves as:
- **Context:** its hidden state (key-value pairs in attention) provides information for predicting future tokens.
- **Output:** its own logits are used to predict the next token $x_{i+1}$.

This coupling works elegantly for sequential generation, but it makes flexible parallel generation difficult. If you want to predict tokens at arbitrary positions (not just the next position in raster order), the architecture provides no mechanism to specify *which* positions you're targeting. The model inherently generates "the next position" based on sequence order, not based on a spatial specification.

**The decoupling solution.** The paper's core architectural idea is to **separate these two roles into distinct token types** (Section 2.2):

- **Image tokens** (previously generated): their only job is to provide context via their key-value representations in the attention mechanism. They do not produce output logits for generation.
- **Position query tokens** (learnable embeddings): their job is to specify *where* to generate and to produce the output logits. They attend to all previous image tokens (and the condition token) to gather context, and their output at the final layer is used to predict the token at their assigned position.

This decoupling is illustrated in Figure 3b. At step 2 of generation, tokens 4 and 3 (generated in step 1) provide context. Position query tokens P1, P2, and P6 are inserted at specific positions in the sequence, each corresponding to a target spatial location. The model processes the concatenated sequence [condition, image_token_4, image_token_3, query_P1, query_P2, query_P6] through the transformer, and the output at P1, P2, and P6's positions yields logits for image tokens at spatial positions 1, 2, and 6 respectively.

**Why this design matters.** This decoupling achieves three things that are impossible in standard decoder-only models:

1. **Arbitrary generation order:** You can place position query tokens for *any* set of ungenerated spatial positions at each step, in any order. The model doesn't care about sequence position — it cares about the spatial positional embedding added to each query token.

2. **Dynamic group sizes:** You can use 1 query token at step 1, 2 at step 2, 4 at step 3, and 20 at step 20, simply by including more or fewer query tokens in the input. No architectural change is needed.

3. **Mutual visibility among concurrent tokens:** This is the crucial one. Because query tokens are regular tokens that go through the full transformer, the attention mechanism can be designed so that **all query tokens in the same generation step can attend to each other**. They see each other's intermediate representations, enabling coordinated joint prediction rather than independent parallel prediction.

---

#### The Attention Mask Design: Enabling Mutual Visibility While Preserving Causality

The attention mask is where the architectural innovation becomes concrete. The paper designs two distinct attention patterns — one for training, one for inference — that together implement the decoupled context-generation model while preserving the causal structure necessary for autoregressive sampling and avoiding the KV-cache inefficiencies of prior methods.

##### Training Attention Mask (Figure 4)

During training, the input sequence interleaves ground-truth image tokens with position query tokens. The mask has two patterns, applied in different regions:

**Context Attention (the causal, lower-triangular portion):** All image tokens attend to all previous image tokens causally — that is, image token $i$ can attend to image tokens $j$ where $j \leq i$ in the sequence. This is standard autoregressive causal masking, and it applies to the image-to-image attention submatrix. Critically, **all tokens (including query tokens) can attend to all image tokens** — image tokens provide context for everyone. In Figure 4, the entire left portion of the attention matrix (under the condition token and image tokens) shows full attention from all subsequent tokens to these context providers.

**Query Attention (the mutual-visibility block):** Position query tokens that belong to the **same generation step** attend to each other with full bidirectional attention — they all see each other's keys and values. However, query tokens from **earlier steps** are NOT visible to query tokens from later steps, and **no subsequent token** (image token or query token) can attend to any query token. In Figure 4, this appears as a dense square block for co-step query tokens (P3 and P5 in the same step can see each other), but zeros for cross-step access (P4 from an earlier step is not visible to P3 or P5, and no image tokens can attend to any P tokens).

**What this computes in operational terms.** For a training sequence like `[c, img_4, img_3, q_P1, q_P2, q_P6]`:
- The condition token c can attend only to itself.
- Image token 4 can attend to c and itself.
- Image token 3 can attend to c, token 4, and itself.
- Position query token P1 can attend to: c, img_4, img_3, P1, P2, P6. That is, all context tokens PLUS all co-step query tokens. But NOT to any query tokens from other steps.
- Position query tokens P2 and P6 have identical attention scope to P1 — they all see each other.
- No token downstream of this step can see P1, P2, or P6.

This implements the training objective: each query token must predict its corresponding ground-truth image token given all preceding ground-truth context AND the other query tokens being predicted simultaneously. The mutual visibility tells the model "you are being asked to predict position (1,2) while positions (2,2) and (6,6) are also being predicted — coordinate with them."

**Why not just use standard causal masking?** A standard causal mask would force query tokens to attend asymmetrically: in left-to-right order, P1 could see nothing from the query group, P2 could see only P1, P6 could see P1 and P2. This is batched next-token prediction (the RandAR problem), not joint prediction. The symmetric mutual visibility is what makes the parallel generation coherent — each query token can incorporate information from all other concurrently-predicted tokens' intermediate states, which is essential when those tokens are spatially related (even if not adjacent, they exist in the same image and their predictions should be globally consistent).

**Why not let subsequent tokens attend to query tokens?** The paper states that query tokens "do not contribute any key-value pairs" to downstream tokens (Figure 6c). This design choice means query tokens are ephemeral — they exist only for the current step's prediction, and their hidden states are discarded. This has two benefits: (1) it prevents the model from learning to use query tokens as a "crutch" context (the model should rely on actual generated image tokens for context, not on query specifications), and (2) it means query tokens' KV pairs don't need to be cached, avoiding the memory doubling of RandAR. In the attention mask, this is implemented by zeroing out the attention weights from any subsequent token (image or query) to query tokens.

##### Inference Attention Mask (Figure 5): Fused Encoding and Decoding

During inference, the operations of encoding newly generated tokens and decoding the next group are fused into a single forward pass. Here's the problem this solves: if you naively alternate between (1) forward pass to encode generated tokens into KV cache, then (2) forward pass to decode query tokens into new tokens, you double the number of steps. At 20 generation steps, you'd need 40 forward passes.

The paper's solution, shown in Figure 5, uses a specialized attention mask where:

- The input sequence contains: [condition | all previously generated tokens | previously generated tokens that will be encoded in this step | position query tokens for this step]
- Image tokens that need to be encoded (step 2's input: tokens 3 and 5) can attend to the condition and all previous image tokens (4), AND to each other — they go through the transformer to compute their KV representations for future steps.
- Position query tokens for the current step (P1, P2, P6) can attend to the condition, all image tokens (4, and the newly encoded 3 and 5), and each other — they use the full updated context to predict new tokens.
- The KV pairs for query tokens are NOT stored in the cache — only the encoded image tokens' KV pairs persist.

**What this computes in operational terms.** Taking step 2 from Figure 3b as the example (Figure 5):
- **Input sequence:** `[c, token_4, token_3, token_5, P1, P2, P6]`
- **Encoding sub-operation:** Tokens 3 and 5 go through the transformer. Their self-attention sees c, token_4, and each other (the encoding block). Their output KV pairs are appended to the KV cache. Their output logits are ignored — they are not being generated, they were already generated.
- **Decoding sub-operation:** P1, P2, and P6 go through the transformer. They attend to c, token_4, token_3, token_5 (all image tokens, including newly encoded ones), and each other. Their output logits are sampled to produce image tokens for positions 1, 2, and 6.
- **After the step:** The KV cache contains c, token_4, token_3, and token_5. P1, P2, P6 have been discarded.

**Why this fusion works.** The key insight is that encoding and decoding share the same transformer layers and the same attention mechanism — they differ only in which tokens' outputs are used (ignored for encoding, sampled for decoding) and which KV pairs are cached (image tokens only). By interleaving them in the same input sequence with a mask that prevents information leakage in the wrong direction (encoding tokens shouldn't see decoding tokens, since in a real sequential setting they wouldn't exist yet), you get both operations for the price of one forward pass. This is what makes the step count equal the number of generation groups G rather than 2G.

---

#### Position Query Tokens: How Arbitrary Targeting Works

The position query tokens are the mechanism that enables the architecture to specify "generate at position (i,j)" without the token itself having been generated yet. Their design is simple but carefully reasoned.

**Construction.** A position query token is formed by adding two components:

1. **A shared learnable embedding vector** — this is a single parameter vector that is the same for all query tokens regardless of their target position. It is learned during training and represents a generic "I am a position query" signal.

2. **A positional embedding specific to the target spatial location** — for target position $(i, j)$ on the 2D grid, the standard 2D sinusoidal or learned positional embedding for that location is added to the shared query embedding.

The result is: `position_query[i,j] = shared_learnable_query_embedding + positional_embedding[i,j]`.

**What this achieves.** The shared query embedding tells the transformer "this token is a generation target (not a context token)," which gates the attention mask behavior (it gets mutual visibility with co-step queries, doesn't contribute to KV cache). The positional embedding tells the transformer *where* to generate — the model learns to associate each output position with its spatial embedding through training, so when it sees `positional_embedding[3,7]` added to the query token, it knows to generate a patch appropriate for row 3, column 7.

**Why a shared embedding rather than per-position embeddings?** If each position had its own learned embedding (a lookup table of size $H \times W \times d$ where d is the hidden dimension), the number of parameters would scale with resolution and the model would have no generalization — it would need to see every position extensively during training. The shared embedding + positional encoding approach leverages the inductive bias that "what to generate at position (3,7)" depends on the spatial location (captured by the positional encoding) and the context, not on a memorized per-position vector. This is the same design principle as standard transformers, where token embeddings are shared and position is added.

**At inference time**, the paper precomputes the position query tokens for all target positions offline. Since the shared embedding is fixed after training and the positional embeddings are deterministic, a query token for position (i,j) is just a constant vector. No computation is needed at inference beyond looking up these precomputed vectors and concatenating them into the input sequence.

**At training time**, the same construction is used, but the target positions vary per sample because the image tokens are randomly shuffled. For each training sample, the model sees a different set of query tokens corresponding to whatever positions are being predicted in that sample's generation order. This teaches the model to handle arbitrary position specifications.

---

#### Locality Analysis: Empirical Foundation for the Generation Order

Before designing the generation order schedule, the paper conducts a systematic empirical analysis of attention patterns in the widely-used LLAMAGEN autoregressive image generation model (Sun et al., 2024). This analysis serves as the scientific justification for the two ordering principles and provides quantitative grounding for the algorithm's hyperparameters.

**Data collection.** The authors generate 50,000 images using LLAMAGEN (which uses standard raster-order next-patch prediction with 24×24 = 576 tokens for 256×256 images — actually 24×24 = 576, but the paper's Figure 2 caption states 24×24) and collect the attention weights at each decoding step. The attention weights represent how much each token being decoded attends to every previously generated token.

**Qualitative finding (Figure 2, Figures 10–11 in Appendix).** The attention maps show a clear pattern: a token being decoded (say, at the current raster position) pays heavy attention to tokens that are spatially nearby in the 2D image grid and much less attention to tokens that are far away. Figure 2 visualizes this for one layer, showing concentrated bright regions around the decoding token's spatial neighborhood. This pattern is consistent across layers and attention heads (Figures 10–11 show multiple layers and heads all exhibiting the same locality).

**Quantitative measurement: Per-Token Attention (PTA).** To quantify locality precisely, the paper defines the Per-Token Attention to a neighborhood at distance $s$:

$$\text{PTA}_s = \frac{1}{N} \sum_{i=1}^{N} \frac{\sum_{j} \text{Attention}(T_i, T_j) \cdot \mathbb{I}[d(T_i, T_j) = s]}{\sum_{j} \mathbb{I}[d(T_i, T_j) = s]}$$

where $\text{Attention}(T_i, T_j)$ is the attention weight from decoding token $T_i$ to context token $T_j$, $d(T_i, T_j)$ is the Euclidean distance between their positions on the 2D image grid, $\mathbb{I}[\cdot]$ is the indicator function (1 if the condition holds, 0 otherwise), and $N$ is the total number of decoding tokens considered.

**What this computes:** For each possible distance $s$ (1, 2, 3, ... up to the maximum possible distance on the grid), PTA$_s$ gives the average attention weight that a token pays to a **single** context token at exactly that distance. The numerator sums all attention from token $i$ to tokens at distance $s$; the denominator counts how many tokens are at distance $s$; the ratio gives the per-token attention at distance $s$ for token $i$; averaging over all N tokens gives the overall metric.

**Why this normalization matters:** Without dividing by the count of tokens at each distance (the denominator), PTA would conflate attention magnitude with the number of available tokens. At large distances, there are more tokens on the circumference, so raw total attention might be high even though per-token attention is low. Normalizing by count isolates the *quality* of a single token at distance $s$ as a conditioning signal.

**Results (Figure 7a, across three LLAMAGEN model sizes — L, XL, XXL).** PTA drops sharply and consistently with distance:
- At relative distance 1 (immediate neighbors in the 2D grid), PTA is roughly 6–10% per token.
- At distance 5, PTA is roughly 1–2% per token.
- At distance 10+, PTA drops below 0.5% per token.
- This pattern is nearly identical across all three model sizes, indicating it is a property of the image generation task rather than a particular model scale.

**Attention Sum visualization (Figure 7b).** For a fixed neighborhood radius ($s=3$), the paper plots the total attention (sum, not per-token average) that each attention head assigns to tokens within that radius. Across all 24 heads (in LLAMAGEN-XL), roughly 40–75% of total attention is concentrated within distance 3, confirming that local context dominates even in aggregate.

**The two design principles derived from this analysis:**

**Principle 1 — High proximity to previously generated tokens:** Since tokens at small distances provide much stronger conditioning (6–10% per token at distance 1 vs. <1% at distance 10), a token being generated should be spatially close to tokens that already exist in the context. This maximizes the useful information available for predicting the new token. In the algorithm, this is operationalized by computing a proximity score to already-selected tokens and prioritizing high-scoring candidates.

**Principle 2 — Low proximity among concurrently generated tokens:** Since tokens at small distances have strong mutual dependencies (if A pays 8% attention to B when B is context, then A and B are highly interdependent), predicting them simultaneously means losing that conditioning signal. To minimize the cost of this loss, tokens predicted in the same parallel step should be spatially distant from each other, so their mutual dependencies are naturally weak (they would pay <1% attention to each other if one were context). In the algorithm, this is operationalized by a repulsion filter that prevents selecting tokens too close to already-selected tokens in the same step.

**Why both principles matter (and why prior methods fail one or both):**
- **Raster order with grouping (PAR):** adjacent tokens are generated together → violates Principle 2 (high intra-group dependency).
- **Random order (RandAR):** no systematic optimization → some tokens are far from context (violates Principle 1), some adjacent tokens are concurrent (violates Principle 2), all by chance.
- **Halton order (Besnier et al., 2025):** spreads tokens uniformly, ensuring low intra-group dependency (satisfies Principle 2) but ignores proximity to existing context (violates Principle 1) — tokens may be placed far from any generated context, receiving weak conditioning.

---

#### Locality-Aware Generation Order Algorithm

Algorithm 1 operationalizes the two principles into a concrete scheduling procedure. The algorithm is run **once offline** for a given resolution and number of steps; the resulting schedule is stored and reused for all inference runs, incurring zero runtime overhead.

##### Inputs and Their Roles

The algorithm takes five inputs:

- **K** (decoding steps): the desired number of sequential generation steps (e.g., 20 for 256×256, 48 for 512×512).
- **O = [o_1, o_2, ..., o_K]** (group sizes): a list where $o_k$ specifies how many tokens to generate at step $k$. The sizes typically follow a cosine schedule — small at first (when context is sparse), growing to a maximum, then leveling off. For example, the 20-step schedule for 16×16=256 tokens: `[1, 2, 4, 5, 7, 8, 10, 11, 12, 14, 15, 16, 17, 18, 18, 19, 19, 20, 20, 20]`. Note that $\sum o_k = N$ must hold.
- **G** (grids): the set of all $N$ spatial positions, typically $\{(i,j)\}_{i,j=1}^{\sqrt{N}}$ for a square image.
- **τ** (proximity threshold): a scalar that determines the minimum proximity score for a candidate to be considered "close enough" to existing context for the high-proximity selection phase. Default values from the sensitivity analysis (Appendix B.3): τ=2 for best results, with ρ fixed at 1.
- **ρ** (repulsion threshold): a scalar specifying the minimum Euclidean distance (in grid units) that concurrently generated tokens must maintain from each other. Default value: ρ=1 (meaning tokens in the same step cannot be adjacent — they must be at least 1 grid unit apart in both row and column).

##### Algorithm Walkthrough (Step k of K)

The algorithm maintains a set `S` of all tokens already selected in previous steps 1 through $k-1$. For the current step $k$, it needs to select $o_k$ new tokens. Here's what happens:

**Step 1: Proximity computation.**
```
p = 1 / euclidean(G \ S, S)
```
For each unselected grid position `$g \in G \setminus S$`, compute its distance to the *closest* already-selected token in `$S$` using Euclidean distance on the 2D grid. The proximity score `$p(g)$` is the reciprocal of this distance: `$p(g) = 1 / \min_{s \in S} d(g, s)$`. This means:
- If the closest existing token is 1 unit away, `$p(g) = 1$`.
- If the closest is 2 units away, `$p(g) = 0.5$`.
- If the closest is 10 units away, `$p(g) = 0.1$`.
A higher score indicates better conditioning — the new token would be close to some existing context.

**Why reciprocal distance?** The attention analysis (Figure 7a) shows that per-token attention falls off non-linearly with distance, with a sharp drop in the 1–5 range. Reciprocal scaling captures this: nearby tokens get substantially higher scores, naturally prioritizing positions where the conditioning context is strongest.

**Step 2: Sorting and splitting.**
```
c = sorted(G \ S, key = p, reverse = True)
c1, c2 = cutoff(c, τ)
```
All unselected tokens are sorted by proximity score descending (highest proximity first). The sorted list is then split at threshold τ: `$c_1$` contains tokens with `$p(g) \geq \tau$` (close to existing context), and `$c_2$` contains tokens with `$p(g) < \tau$` (far from existing context). With τ=2, a token must be within Euclidean distance 0.5 of an existing token to qualify for `$c_1$` — in practice, since grid positions are integers, this means distance ≤ 1 (immediate neighbors, including diagonals at distance ~1.4, which would have p ≈ 0.71 < 2, so roughly only distance-1 tokens).

**Why the cutoff?** This creates a priority system: tokens that satisfy Principle 1 (high proximity to context) get first consideration. Those that don't are relegated to the backup pool `$c_2$`, ensuring that when possible, the algorithm always selects tokens with strong conditioning.

**Step 3: High-proximity selection with repulsion.**
```
while len(s) < o_k and len(c1) > 0:
    s = queue_push(s, queue_pop(c1, 1))   # take the highest-proximity candidate
    c1, f = filter(c1, s, ρ)               # remove candidates too close to the newly selected token
    c2 = queue_push(c2, f)                  # demoted candidates go to the backup pool
```
This is a greedy loop. In each iteration:
1. Pop the highest-proximity candidate from the front of `$c_1$` (which is sorted descending by proximity).
2. Add it to the step's selection list `s`.
3. Filter `$c_1$`: remove any candidate whose distance to the newly selected token is ≤ ρ (the repulsion threshold). These removed candidates are added to `$c_2$` — they violated Principle 2 (would be too close to a concurrently selected token) but might be selected later via farthest-point sampling if needed.
4. Repeat until either we have `$o_k$` tokens or `$c_1$` is exhausted.

**What this achieves.** This phase simultaneously optimizes both principles: it selects tokens in order of proximity to context (Principle 1), but rejects any that would be too close to already-selected co-step tokens (Principle 2). The greedy approach means early selections get the highest-proximity tokens; later selections settle for lower proximity but still maintain spatial separation from all previous selections in the step.

**Why greedy rather than global optimization?** A globally optimal selection (maximizing some combined score of context proximity and intra-group separation) would be a combinatorial optimization problem — essentially a variant of maximum-weight independent set on a graph where vertices are grid positions and edges represent "too close" violations, with vertex weights being proximity scores. This is NP-hard in general. The greedy approach is linear in the number of candidates and the paper's results (Figure 9c) show it's effective in practice.

**Step 4: Low-dependency selection (farthest-point sampling).**
```
if len(s) < o_k:
    s = queue_push(s, farthest_point_sampling(c2, s, o_k - len(s)))
```
If after exhausting `$c_1$`, we still haven't reached the target `$o_k$` tokens (because many high-proximity candidates were filtered out by repulsion), we fill the remaining slots from `$c_2$` using farthest-point sampling (FPS). FPS iteratively selects the point in `$c_2$` that maximizes the minimum distance to all points already selected (both from previous steps and from `s` in the current step). This is a standard algorithm for selecting a spatially diverse subset (Qi et al., 2017).

**What FPS guarantees.** FPS produces a subset that is approximately uniformly distributed — selected points are as far apart from each other as possible given the existing constraints. This directly optimizes Principle 2 (minimize intra-group dependency) while using whatever candidates remain. It does not explicitly optimize Principle 1 (these candidates were in `$c_2$` precisely because they were far from existing context), but it minimizes the damage by at least ensuring the concurrently selected tokens are well-separated.

**Why FPS rather than random selection from `$c_2$`?** Random selection could accidentally pick two tokens that are close to each other, creating a high-dependency pair within the group. FPS is a principled way to maximize spatial diversity, which empirical results (Figure 9c, "Principle 2 only" bar) confirm improves quality.

**Step 5: Finalize and iterate.**
```
S = queue_push(S, s)
```
The step's selected tokens `s` are appended to the cumulative schedule `S`, and the algorithm proceeds to step `k+1`. At the end, `S` is a list of K lists, specifying exactly which spatial positions to generate at each step.

##### What the Complete Schedule Looks Like (Figure 8d)

Figure 8 visualizes the LPD schedule for 20 steps generating 16×16 = 256 tokens, comparing it against raster, random, and Halton schedules. The LPD schedule (column d) shows:
- **Early steps** select a sparse set of tokens, building a "scaffold" of context across the image. Step 1 might pick one token near the center; step 2 picks a few tokens near it; step 4 picks more tokens that are close to the growing context set but separated from each other.
- **Middle steps** fill in regions around the existing context, maintaining spatial separation within each step's group.
- **Late steps** fill in the remaining gaps, with groups becoming larger as context becomes dense.

Compared to Halton (column c), LPD's selections are more "clustered" around existing context (visible as light green regions growing outward) rather than being uniformly spread regardless of what's already generated. Compared to random (column b), LPD shows clear spatial structure rather than scattered selections. Compared to raster (column a), LPD avoids the problem of generating large contiguous blocks simultaneously.

##### Hyperparameter Sensitivity (Table 6, Appendix B.3)

The paper conducts a sensitivity analysis of τ and ρ using the LPD-XL model with 32 decoding steps on ImageNet 256×256:

**Repulsion threshold ρ (fixing τ=2):**
- ρ = 0: FID = 1.94 (no repulsion, but sorting by proximity still keeps Principle 1 active)
- ρ = 0.2: FID = 1.94
- ρ = 0.5: FID = 1.93
- ρ = 1.0: FID = 1.92 (default, best)
- ρ = 1.5: FID = 2.00
- ρ = 2.0: FID = 2.04

The sweet spot at ρ=1 makes intuitive sense: prevent adjacent tokens (distance 1) from being generated together, but don't push tokens so far apart (ρ=2 means tokens must be at least 2 grid units apart) that you can't find enough candidates near existing context.

**Proximity threshold τ (fixing ρ=1):**
- τ = 0: FID = 2.00 (no filtering means all candidates enter c₁, repulsion alone does the work — but without proximity prioritization, some selected tokens may be far from context)
- τ = 1: FID = 1.94
- τ = 2: FID = 1.92 (default, best)
- τ = 3: FID = 1.97
- τ = 4: FID = 2.00

When τ is too large (3–4), the c₁ pool becomes too small — few tokens satisfy the high proximity requirement, forcing the algorithm to rely heavily on c₂ and FPS, which don't optimize Principle 1. When τ is too small (0–1), the distinction between c₁ and c₂ collapses, and the high-proximity prioritization loses its effect.

The relative insensitivity around the default values (FID ranging from 1.92 to 2.04 across a wide range of τ and ρ) suggests the algorithm is robust — the two principles provide consistent gains, and exact threshold tuning is not critical.

---

#### Training Procedure

The training procedure is designed to teach the model to handle arbitrary generation orders and degrees of parallelization, so that at inference time it can handle whatever schedule the locality-aware algorithm produces.

**Data preparation.** For each training sample (an image and its class label):
1. The image is tokenized using the LLAMAGEN tokenizer (codebook size 16,384, downsample factor 16, producing 16×16 = 256 tokens for 256×256 images, 32×32 = 1,024 tokens for 512×512 images).
2. The image tokens are **randomly shuffled** — the sequence order of the 256 tokens is permuted uniformly at random. The class token is kept at the beginning (position 0).
3. A number of decoding steps K is randomly sampled from a predefined set:
   - For 256×256: {8, 12, 16, 20, 24, 32, 64, 128, 256}
   - For 512×512: {32, 40, 48, 56, 64, 80, 96, 128, 160, 192, 224, 256, 512, 1024}
4. The tokens are partitioned into K groups according to a cosine schedule. For example, with K=20 on 256×256, the group sizes are `[1, 2, 4, 5, 7, 8, 10, 11, 12, 14, 15, 16, 17, 18, 18, 19, 19, 20, 20, 20]` (as stated in Appendix A.2).

**Sequence construction for training.** For a sample with K groups, the input sequence is constructed as:
```
[class_token] + [all tokens from group 1 as image tokens] + [position queries for group 2] + [all tokens from group 2 as image tokens] + [position queries for group 3] + ...
```
That is, image tokens and query tokens are interleaved: each group's image tokens serve as context for subsequent query tokens. The attention mask (Figure 4) ensures that:
- All image tokens can attend to all previous image tokens (causal).
- Query tokens for group g can attend to all image tokens from groups 1 through g (full context), AND to all other query tokens in group g (mutual visibility).
- Query tokens for group g cannot attend to query tokens from groups before g, and no tokens from groups after g can attend to query tokens in group g.

**Training objective.** The model is trained with standard cross-entropy loss on the image token prediction task. For each position query token in the sequence, the target is the corresponding ground-truth image token at that spatial position. The loss is summed over all query tokens in all groups and averaged. This is effectively a next-token prediction objective, but generalized to predict multiple tokens per step in arbitrary order — the model learns `$p_\theta(\text{token}_{i,j} \mid \text{all preceding context tokens}, \text{co-step query tokens}, c)$` for any position `$(i,j)$` and any context configuration.

**Why random shuffling?** This is a form of data augmentation that forces the model to handle arbitrary generation orders. If the model were trained only on a fixed order (e.g., the LPD schedule), it might overfit to that specific ordering and fail if the schedule changes. By seeing every token in every possible context configuration (different preceding tokens, different co-step tokens), the model learns a general-purpose conditional distribution $p(\text{token} \mid \text{arbitrary context set})$, which is exactly what's needed at inference time for the LPD schedule. The authors note (Section 3.1): "During training, the image tokens are randomly shuffled while the class token is kept at the beginning."

**Training hyperparameters (Table 5, Appendix A.2, for LPD-L on 256×256):**
- **Optimizer:** AdamW
- **β₁, β₂:** 0.9, 0.95
- **Learning rate:** $8 \times 10^{-4}$ (computed as base_lr × (global_batch_size / 256) with base_lr = $1 \times 10^{-4}$, giving effective lr of $10^{-4} \times (2048/256) = 8 \times 10^{-4}$)
- **Batch size:** 2,048 (64 per GPU × 32 GPUs)
- **Training precision:** BFloat16
- **Total epochs:** 450 (50 warmup, 350 constant LR, 50 cosine decay)
- **Offsets:** random per-sample (meaning the positional encoding offset varies per sample)
- **For LPD-XL and LPD-XXL:** same base learning rate, batch size 1,024

**Training curriculum across resolutions (Appendix A.2).** For 512×512 models, the authors load the pre-trained 256×256 checkpoint, interpolate the positional embeddings to the larger grid, and continue training for 50 epochs with a cosine decay (1 epoch warmup). This is a standard progressive training approach that transfers learned generation capabilities to higher resolution without training from scratch. For text-to-image at 1024×1024 (Appendix A.3), a three-stage progressive schedule is used: 256×256 (40 epochs) → 512×512 (5 epochs) → 1024×1024 (2 epochs).

**Why the cosine schedule for group sizes?** The paper doesn't provide an ablation of the group size schedule itself, but the rationale aligns with the locality analysis: early steps have sparse context, so predicting many tokens in parallel would mean generating tokens with weak conditioning (far from any existing token) — exactly what Principle 1 warns against. As context accumulates (later steps), any position has some nearby context token, so larger groups are feasible. The cosine schedule ramps up gradually and saturates, matching this intuition. The specific token counts for 20 steps sum to 256, confirming each position is generated exactly once.

---

#### Inference Pipeline

At inference time, the trained model and precomputed LPD schedule are used to generate images. The pipeline follows a simple loop:

**Initialization:**
1. The class condition $c$ (or text embedding from Gemma-2-2B for text-to-image) is processed into the initial sequence: `[condition_embedding]`.
2. The KV cache is initialized empty.
3. The precomputed LPD schedule $S$ (a list of K lists of spatial positions) is loaded.

**For each step $k = 1, 2, \ldots, K$:**

1. **Construct position query tokens:** For each spatial position $(i, j)$ in $S[k]$, retrieve (or compute) the corresponding position query token vector: `query[i,j] = shared_learnable_query + positional_embedding[i,j]`.

2. **Construct the input sequence for this step:**
   - Start with the conditioning token.
   - Append all image tokens generated in the **previous step** (these need to be encoded into the KV cache). For step 1, there are no previous image tokens, so only the condition token is present in this portion.
   - Append the position query tokens for the current step's target positions.

3. **Forward pass with the fused encoding-decoding attention mask (Figure 5):**
   - The previous step's image tokens go through the transformer, attend to the existing KV cache (all previous context), and their output KV pairs are **appended** to the KV cache. Their output logits are ignored.
   - The position query tokens attend to the condition token, ALL image tokens in the KV cache (including the newly encoded ones), and to each other (mutual visibility). Their output logits are sampled to produce new image tokens.

4. **Sampling:** For each position query token, the output logits (a vector of size 16,384 — the codebook size) are converted to probabilities via softmax. A token is sampled from this distribution (using the configured temperature and classifier-free guidance scale). The sampled token index corresponds to a specific discrete code in the VQ tokenizer's codebook.

5. **Store for next step:** The newly sampled tokens are saved — they will become the "previous step's image tokens" for encoding in step $k+1$.

**After K steps:** All N tokens have been generated. They are rearranged from the LPD schedule order back to the 2D grid layout and passed through the VQ decoder to produce the final image.

**Classifier-free guidance (CFG).** The paper mentions in Table 1 that CFG is used for both their method and baselines. In standard CFG for autoregressive models, at each generation step, two logit sets are computed: one conditioned on the class label (or text), and one unconditioned (typically using a null token). The final logits are: `logits_final = logits_uncond + scale × (logits_cond - logits_uncond)`. The scale is swept at evaluation time with an interval of 0.1 (Appendix A.2). CFG sharpens the conditional distribution, improving sample quality at the cost of diversity.

**Efficiency characteristics (Figure 14, Appendix E).** The paper provides a detailed throughput-vs-batch-size analysis that clarifies when LPD's speedup materializes:
- **Memory-bound regime (batch size ≤ 16 for LPD models):** Throughput scales linearly with batch size, and LPD achieves approximately 12× higher throughput than the raster baseline — closely matching the 256/20 = 12.8× step reduction. This is because in the memory-bound regime, each forward pass is bottlenecked by parameter loading, so reducing the number of forward passes directly translates to proportional speedup. The extra query tokens' computation overhead is negligible because the GPU's arithmetic units are mostly idle anyway.
- **Compute-bound regime (batch size > 16 for LPD, > 128 for raster):** Throughput plateaus as the GPU's arithmetic capacity saturates. The speedup diminishes because the extra FLOPs from query tokens (which don't exist in the raster baseline) start to matter — the raw computation per step is higher for LPD, partially offsetting the step reduction. Even at the maximum feasible batch size, LPD retains approximately 3× throughput advantage over raster.

**Zero-shot editing capability (Appendix D, bottom of Figure 12).** Because the architecture supports arbitrary generation orders, it naturally enables image editing tasks without any additional training:
- **Inpainting:** Given an image with a masked region, prefill the KV cache with all tokens from the unmasked region, then generate tokens for the masked positions using the LPD schedule (constrained to the mask).
- **Outpainting:** Prefill the KV cache with the original image tokens, then generate tokens for the extended canvas.
- **Class-conditional editing:** Prefill the KV cache with the original image tokens, replace the class embedding with a new class, and regenerate a subset of positions (or all positions) with the new conditioning. The paper shows examples of class changes (e.g., a bird becoming a different bird species).

This is a direct consequence of the architecture's flexibility: since the model can condition generation on any set of pre-existing image tokens at arbitrary positions (via KV cache prefill), it naturally supports partial regeneration tasks that standard raster-order models would struggle with (you can't easily "skip" positions in a strict raster scan).

## 4. Key Insights and Innovations

### Innovation 1: Reframing Parallel Autoregressive Generation as a Joint Prediction Problem, Not a Batched Next-Token Problem

**What's distinctive at the idea level.** Prior work on parallelizing autoregressive image generation implicitly treated the problem as "generate multiple tokens in one forward pass while maintaining the autoregressive causal structure." This framing is natural — it preserves the mathematical validity of the autoregressive factorization — but it leads to a specific architectural choice: the standard causal attention mask. Under that mask, when you try to predict tokens at positions A, B, and C simultaneously in left-to-right sequence order, A sees only context, B sees context + A, and C sees context + A + B. This is batched next-token prediction: three sequential predictions computed in parallel, with asymmetric information flow.

The paper's key reframing is to treat parallel generation as a **joint prediction** problem: the model should predict a *set* of tokens simultaneously, with full mutual visibility among all tokens in the set. The difference sounds subtle — "mutual visibility" versus "left-to-right visibility within a step" — but it fundamentally changes what the model learns. Under joint prediction, each token being generated can condition on the intermediate representations of all other concurrently generated tokens. This enables coordination: if tokens at positions (3,5) and (3,6) are being generated together, they can "negotiate" to produce a coherent edge or texture across their boundary, rather than token (3,6) having to guess what token (3,5) might produce (as in batched next-token prediction) or having no information at all (as in independent parallel prediction).

**Comparison to prior work.** RandAR (Pang et al., 2024) uses standard causal masking with positional instruction tokens — this is the batched next-token regime. ARPG (Li et al., 2025a) and SAR (Liu et al., 2024b) use encoder-decoder architectures where decoder query tokens attend to the encoder's KV cache but not to each other — this is the independent parallel prediction regime. PAR (Wang et al., 2024b) generates multiple tokens per step but under causal masking within the step — also batched next-token. In all three cases, concurrently generated tokens cannot see each other symmetrically. The paper's ablation (Figure 9a) quantifies the cost: as generation steps decrease from 256 to 16 (parallelization increases), RandAR and ARPG both show much steeper FID degradation than LPD, because without mutual visibility, the inconsistency cost of parallel generation grows with group size.

**Significance beyond raw performance.** This is a conceptual reframing, not just an architectural tweak. It establishes that for parallel autoregressive generation, the constraint that matters is not strict causality between steps (which the paper preserves) but **symmetry of information within steps**. The paper's attention mask design — which grants full bidirectional attention within a co-step query group while blocking cross-step access — shows that you can have both: causal structure between groups (maintaining the autoregressive property) and symmetric mutual visibility within groups (enabling coherent joint prediction). This insight generalizes beyond image generation to any domain where autoregressive modeling of structured data with spatial or relational dependencies is being parallelized.

**Evidence anchor.** Figure 9a shows LPD with random ordering (removing the locality-aware schedule to isolate the architectural effect) maintains FID ~2.0 at 32 steps versus ~2.25 for RandAR and ~2.3 for ARPG. At 16 steps, the gap is even wider.

---

### Innovation 2: Diagnosing Attention Locality as the First-Principles Basis for Generation Order Design

**What's distinctive at the idea level.** Prior work on ordering for parallel generation — random order (RandAR), Halton low-discrepancy sequences (Besnier et al., 2025), fixed region-wise schemes (PAR) — made implicit assumptions about what makes a good order, but none grounded those assumptions in a quantitative analysis of the model's own attention behavior. Random order assumes no structure matters. Halton assumes uniform spatial coverage is optimal. PAR assumes fixed block-wise grouping works. These are heuristics derived from geometric intuition, not from how the model actually uses context during generation.

The paper does something different: it **analyzes the attention patterns of a trained autoregressive model** (LLAMAGEN) to measure what information tokens actually use during generation, then derives ordering principles directly from those measurements. The finding — that per-token attention drops from 6–10% at distance 1 to below 1% at distance 10 (Figure 7a) — is not merely descriptive. It is **diagnostic**: it tells you that spatial proximity is the primary determinant of how much conditioning a context token provides. This immediately implies two operational principles that no prior work had jointly implemented: (1) maximize proximity to context, and (2) minimize proximity within the concurrent group.

**Comparison to prior work.** Halton order (Besnier et al., 2025) implicitly recognizes principle 2 — it spreads tokens to minimize intra-group dependency — but neglects principle 1 entirely, because a low-discrepancy sequence doesn't account for where existing context is located. Random order satisfies neither principle systematically. PAR's fixed region scheme violates principle 2 within each region and doesn't explicitly optimize principle 1 across regions. The paper's contribution is not discovering that locality matters (that's visible in any attention map) but **operationalizing it as a quantitative optimization criterion** — the proximity score `p(g) = 1 / min distance to context` — and building an algorithm that jointly optimizes both principles.

**Significance beyond raw performance.** This is a methodological contribution: it demonstrates that attention analysis of a trained model can reveal the structural properties that generation order should respect. This approach is transferable: for any new domain (video generation, 3D generation, molecular generation), one could analyze attention patterns in a trained autoregressive model to derive domain-specific ordering principles, rather than guessing. It also explains *why* certain prior heuristics fail: Halton's uniform spreading is optimal *only if* conditioning quality is uniform across space, but attention locality means conditioning quality is sharply non-uniform — context tokens at distance 1 provide 10× more information than tokens at distance 10. Ignoring this makes Halton underperform by ~0.15 FID at 20 steps (Figure 9b).

**Evidence anchor.** Figure 7a quantifies the locality effect across three model scales. Figure 9c ablates the two principles separately: principle 1 alone improves FID from 2.11 (random) to 2.00; principle 2 alone to 2.06; combining both achieves 1.92. The synergy confirms both principles are necessary.

---

### Innovation 3: Establishing Mutual Visibility and Locality-Aware Ordering as Joint Necessary Conditions for Aggressive Parallelization

**What's distinctive at the idea level.** The paper's most integrative contribution is the empirical demonstration that high-degree parallelization (12.8× step reduction) requires **both** the architectural capability for mutual visibility **and** the scheduling strategy to place mutually-visible tokens where their dependency is naturally low. Neither alone suffices.

This is not obvious a priori. One might think that mutual visibility makes ordering less important — if tokens can see each other, they can coordinate regardless of where they are, so spatial arrangement shouldn't matter much. Conversely, one might think that optimized ordering makes mutual visibility less important — if tokens are placed where dependencies are weak, they don't need to see each other. The paper shows both intuitions are wrong: even with mutual visibility, locality-aware ordering adds ~0.2 FID improvement (Figure 9c, comparing "Principle 1 + 2" at 1.92 to "random order with mutual visibility" implied by the Figure 9a LPD curve at ~2.1). And even with optimized ordering, without mutual visibility the architecture collapses at aggressive parallelization (Figure 9a, comparing LPD to ARPG/RandAR, which lack mutual visibility but use the same random ordering in this ablation).

The deeper insight is that these two components address **different failure modes** that compound under aggressive parallelization:
- **Mutual visibility** addresses the *coordination* problem: tokens that need to produce a coherent local structure must see each other's intermediate states to avoid inconsistency.
- **Locality-aware ordering** addresses the *conditioning quality* problem: tokens that are far from context have poor conditioning, and tokens that are close to each other have high mutual dependency that even mutual visibility cannot fully compensate for (since mutual visibility provides intermediate representations, not ground-truth conditioning).

**Comparison to prior work.** Every prior method addressed at most one of these issues. PAR has mutual visibility (its grouped-token prediction uses full attention within the group) but uses fixed raster-based ordering that places high-dependency tokens together. RandAR has flexible ordering but lacks mutual visibility and doesn't locality-optimize the order. ARPG has neither mutual visibility nor locality-aware ordering. Halton-based ordering optimizes spatial separation but the underlying architecture lacks mutual visibility. The paper's claim — and the evidence in Figures 9a–c — is that **both are necessary simultaneously**, which is why no prior method achieved comparable parallelization ratios without quality collapse.

**Significance beyond raw performance.** This is a **necessary-conditions result**: it empirically identifies the minimal set of capabilities a parallelized autoregressive image generation system must have to achieve aggressive step reduction without quality degradation. This provides a design checklist for future work: any new parallelization architecture must demonstrate (1) mutual visibility among concurrent tokens, and (2) a generation order that accounts for attention locality. Systems lacking either can be expected to show diminishing returns as parallelization increases. This transforms the problem from "try various heuristics and see what works" to "verify these two conditions are met."

**Evidence anchor.** The combination of Figure 9a (architecture ablation at varying step counts shows LPD's mutual visibility advantage grows with parallelization), Figure 9b (ordering ablation at varying step counts shows locality-aware ordering advantage is consistent), and Figure 9c (two-principle ablation shows additive gains) together support the joint-necessity claim. The 12.8× step reduction with maintained FID (Table 1) demonstrates the practical consequence.

---

### Innovation 4: The Position Query Token as a Mechanism for Decoupling Spatial Specification from Sequential Context

**What's distinctive at the idea level.** Standard autoregressive transformers conflate **what to generate** and **where to generate it** into a single token's sequence position. The "where" is implicitly defined by the token's index in the sequence and the predefined generation order. This conflation is the architectural root cause of why standard decoder-only models struggle with flexible generation orders: you can't easily say "now generate the token at grid position (7,3)" because the model has no mechanism to interpret a spatial target separate from the sequence of already-generated tokens.

The paper's **position query token** is a deceptively simple mechanism that separates "what to condition on" (the context tokens, which go through the transformer and produce KV pairs) from "where to generate" (the query token, whose spatial positional embedding specifies the target location). This decoupling achieves what might otherwise require a major architectural change (like switching to an encoder-decoder with cross-attention to a positional encoding) within the standard decoder-only transformer framework — it's the same transformer, the same layers, the same KV caching, just with different tokens playing different roles in the attention mask.

**Comparison to prior work.** RandAR (Pang et al., 2024) attempted to solve the same problem by inserting positional instruction tokens that say "generate position (7,3) next." But because these instruction tokens are part of the same sequence under a causal mask, they inherit the dual-role problem: the instruction token itself becomes part of the context for future tokens (doubling KV cache memory) and participates asymmetrically in attention (batched next-token rather than joint prediction). ARPG and SAR moved to encoder-decoder architectures with cross-attention to positional queries, which decouples spatially but loses the architectural simplicity and KV-caching benefits of decoder-only models. The position query token design keeps the decoder-only architecture but achieves the decoupling through attention masking rather than architectural separation — the query tokens are regular transformer tokens that happen to be treated specially by the mask (mutual visibility, no downstream access).

**Significance beyond raw performance.** This is an **architectural pattern** rather than a one-off trick. The idea of "some tokens provide context only, others provide generation targets only, and the attention mask orchestrates their interaction" generalizes to any domain where you want flexible generation order with mutual visibility in a decoder-only transformer. It shows that the standard causal mask is not the only valid attention pattern for autoregressive models — you can have causal structure *between groups* while having full attention *within groups*, and you can have tokens that see everything but are seen by nothing (the query tokens' asymmetric visibility). This expands the design space for autoregressive transformers beyond the strict next-token formulation that dominates the literature.

**Evidence anchor.** The architecture supports 20-step generation with maintained FID (Table 1) and enables zero-shot editing tasks (Figure 12, bottom) — inpainting, outpainting, and class-conditional editing — that are natural consequences of the decoupled spatial specification but would require architectural changes in standard decoder-only models. The comparison in Figure 6 and the associated analysis in Section 2.2 provides the architectural contrast with SAR/ARPG and RandAR.

## 5. Experimental Analysis

### Evaluation Methodology

- **Dataset.** All main experiments use the ImageNet (Russakovsky et al., 2015) class-conditional generation benchmark at 256×256 and 512×512 resolutions. The standard training set (~1.28M images) is used for training, with 50,000 generated samples used for FID evaluation. For text-to-image experiments, an internal MidJourney-style synthetic dataset of approximately 10M images is used, re-captioned with Qwen2.5-VL-32B (Bai et al., 2025). Text-to-image evaluation uses the GenEval benchmark (Ghosh et al., 2023).

- **Base model(s).** Three model sizes are trained, all using a standard decoder-only transformer architecture: LPD-L (337M parameters, 12 layers, 1024 hidden size, 12 heads), LPD-XL (752M, 36 layers, 1280 hidden size, 20 heads), and LPD-XXL (1.4B, 48 layers, 1536 hidden size, 48 heads). All models use the LLAMAGEN tokenizer (Sun et al., 2024) with codebook size 16,384 and downsample factor 16, producing 16×16 = 256 tokens for 256×256 images and 32×32 = 1024 tokens for 512×512 images. The family is chosen to span a representative range of model capacities for fair comparison with existing work, and to demonstrate that the method scales (the largest model achieves the best FID, confirming that parallelization doesn't bottleneck scaling). For text-to-image, the same architectures are augmented with a Gemma-2-2B (Team et al., 2024) text encoder whose embeddings are linearly projected and concatenated with image embeddings.

- **Metrics.** The primary metric is Fréchet Inception Distance (FID) (Heusel et al., 2017), computed on 50,000 generated samples against the ImageNet training set. Lower FID indicates better sample quality and diversity. Secondary metrics include Inception Score (IS) (Salimans et al., 2016), Precision, and Recall (Kynkäänniemi et al., 2019). For text-to-image, the GenEval score (Ghosh et al., 2023) is reported, which measures compositionality across six sub-tasks (single object, two objects, counting, colors, position, color attribution). Efficiency is measured via latency (seconds per image at batch size 1) and throughput (images per second at batch size 64) on a single NVIDIA A100 GPU with BFloat16 precision. Latency is averaged over 500 inference steps with a 100-step warm-up period.

- **Baselines.** The paper compares against an extensive set of methods spanning multiple generative paradigms:
  - **Diffusion models:** ADM-G (Dhariwal & Nichol, 2021), CDM (Ho et al., 2022), LDM-4 (Rombach et al., 2022), DiT-XL/2 (Peebles & Xie, 2023), SiT-XL/2 (Ma et al., 2024).
  - **Masked prediction models:** MaskGIT (Chang et al., 2022), MAGVIT-v2 (Yu et al., 2023b), MaskBit (Weber et al., 2024), MAR-B/L/H (Li et al., 2024).
  - **Next-scale prediction (VAR):** VAR-d16/d20/d24/d30 (Tian et al., 2024).
  - **Standard autoregressive (AR):** VQGAN-re (Esser et al., 2021), RQTran.-re (Lee et al., 2022), LlamaGen-L/XL/XXL/3B (Sun et al., 2024), RAR-B/L/XL/XXL (Yu et al., 2024).
  - **Parallelized autoregressive:** PAR-L/XL/XXL/3B-4× (Wang et al., 2024b), RandAR-L/XL/XXL (Pang et al., 2024), ARPG-L/XL/XXL (Li et al., 2025a), NAR-L/XL/XXL (He et al., 2025).
  - **Raster counterpart:** The authors train their own standard raster-order autoregressive baselines at each model size (337M, 752M, 1.4B) using the same tokenizer, training recipe, and model architecture as LPD but with standard next-token prediction in raster order. This provides a controlled comparison that isolates the effect of parallelization from other confounding factors (tokenizer, training data, model capacity).

  For text-to-image, the baselines are raster-order counterparts trained with the same three-stage progressive-resolution pipeline.

- **Generation budget / compute accounting.** The primary cost metric is the number of generation steps (#Steps in Tables 1–2) — the number of sequential forward passes required to generate one image. This is the relevant metric because autoregressive generation is memory-bound at small batch sizes (Figure 14), so latency scales approximately linearly with step count. Additional FLOPs from position query tokens are accounted for implicitly in the latency and throughput measurements, which capture wall-clock time on identical hardware. The paper does not report FLOP counts directly, arguing (Section 3.3) that in the memory-bound regime, FLOPs are a poor proxy for actual performance. The efficiency analysis (Figure 14) systematically measures throughput across batch sizes from 1 to 512 to characterize the transition from memory-bound to compute-bound regimes.

- **Cross-validation / statistical protocol.** There is no cross-validation in the traditional sense, since the task is generative modeling evaluated on a fixed test set, not hyperparameter selection from a validation set. The classifier-free guidance scale is swept with an interval of 0.1 on the generation set (Appendix A.2), which is standard practice. For the text-to-image GenEval results (Table 3), the reported scores are on the standard GenEval benchmark using the official evaluation protocol. The paper does not report confidence intervals or standard errors for FID measurements, which is typical for the image generation literature but limits the ability to assess whether small FID differences (e.g., 1.92 vs. 2.00) are statistically significant given the 50k-sample estimation.

### Main Quantitative Results

#### Class-Conditional Generation on ImageNet 256×256 (Table 1)

The headline result is that LPD reduces generation steps from 256 (raster baseline) to 20 while maintaining or improving generation quality. At the XL size (752M parameters), the raster counterpart achieves 2.12 FID at 256 steps and 5.29s latency. LPD-XL achieves 2.10 FID at 20 steps and 0.41s latency — a 12.8× step reduction and 12.9× latency reduction (5.29 / 0.41 ≈ 12.9). At 32 steps, LPD-XL achieves 1.92 FID at 0.66s latency, which is *better* quality than the raster baseline at 256 steps while being 8.0× faster.

Scaling behavior is consistent: LPD-L (337M) achieves 2.40 FID at 20 steps versus 2.48 for its raster counterpart at 256 steps; LPD-XXL (1.4B) achieves 2.00 at 20 steps versus 2.01 for its raster counterpart. The quality improvement from scaling model size is preserved under parallelization — the FID gap between LPD-L and LPD-XXL at 20 steps (2.40 → 2.00) is comparable to the raster counterpart gap (2.48 → 2.01), suggesting parallelization does not distort the scaling law.

Compared to prior parallelized autoregressive methods at similar model scales, the latency improvement is substantial:
- LPD-XL at 20 steps: 2.10 FID, 0.41s. ARPG-XL at 64 steps: 2.10 FID, 1.71s. LPD achieves the same FID with 3.2× fewer steps and 4.2× lower latency.
- LPD-XL at 32 steps: 1.92 FID, 0.66s. ARPG-XXL (1.3B, larger model) at 64 steps: 1.94 FID, 2.24s. LPD matches a larger model's quality with 2× fewer steps and 3.4× lower latency.
- RandAR-XL at 88 steps: 2.25 FID, 2.78s. LPD-XL achieves better quality (2.10 or 1.92) in 20–32 steps at 0.41–0.66s.
- PAR-XL-4× at 147 steps: 2.61 FID, 4.79s. LPD-XL achieves substantially better quality in 7.4× fewer steps.
- NAR-XL at 31 steps: 2.70 FID, 1.42s. LPD-XL at comparable steps (32) achieves 1.92 FID at 0.66s — better quality at lower latency.

Compared to non-autoregressive methods, LPD is competitive or better in the quality-efficiency tradeoff space:
- VAR-d16 (310M): 3.30 FID at 0.12s, 10 steps. LPD-L (337M): 2.40 FID at 0.28s, 20 steps. LPD trades off some latency for substantially better quality, but the key distinction (which the paper emphasizes) is that LPD preserves flat-token compatibility while VAR does not.
- MaskBit (305M): 1.62 FID at 1.03s, 64 steps. LPD-L at similar size: 2.29 FID at 0.46s, 32 steps. MaskBit achieves better FID but requires bidirectional attention without KV caching, making it architecturally incompatible with autoregressive language model infrastructure. LPD provides a different point on the Pareto frontier (worse FID, better compatibility).
- MAR-L (479M): 1.78 FID at 20.80s, 64 steps. LPD-XL achieves comparable quality (1.92) at 32× lower latency, though MAR uses a different (continuous) tokenization approach.

Throughput measurements (Table 1, batch size 64) follow a similar pattern: LPD-L achieves 139.11 img/s at 20 steps versus 17.53 for raster at 256 steps (7.9× improvement); LPD-XL achieves 75.20 versus 12.31 (6.1×); LPD-XXL achieves 45.07 versus 8.99 (5.0×). The diminishing throughput ratio at larger model sizes is expected — larger models have higher per-step FLOPs, and the extra query token computation becomes a larger fraction of total cost.

#### Class-Conditional Generation on ImageNet 512×512 (Table 2)

The 512×512 results demonstrate that the method scales to higher resolutions where the step reduction is even more dramatic: 1024 → 48 steps, a 21.3× reduction. LPD-XL achieves 2.10 FID at 48 steps and 1.01s latency, versus its raster counterpart at 2.09 FID, 1024 steps, 20.93s latency — quality is preserved while latency drops by 20.7×.

The scaling pattern observed at 256×256 is replicated: LPD-L achieves 2.54 FID at 48 steps (raster counterpart: 2.54 at 1024 steps), confirming that the parallelization does not degrade quality at any model size tested. The 512×512 results are reported for fewer baselines (Table 2 includes only a subset of methods compared to Table 1), limiting cross-method comparison, but the controlled raster-counterpart comparison is clean.

#### Text-to-Image Generation at 1024×1024 (Table 3, Table 7)

The text-to-image results on GenEval extend the method to open-vocabulary conditioning at high resolution. LPD-L achieves 0.58 GenEval score at 64 steps and 1.01s latency, versus the raster counterpart at 0.55 score, 4096 steps, 64.7s latency — LPD is simultaneously better (0.58 vs. 0.55) and 64× faster in latency (64.7 / 1.01 ≈ 64.1×). LPD-XL achieves 0.62 at 64 steps and 1.53s, versus raster at 0.60, 4096 steps, 93.8s.

The per-category breakdown (Table 7) reveals that the GenEval improvement comes primarily from Position (0.34 vs. 0.25 for XL) and Color Attribution (0.37 vs. 0.36 for XL), while Single Object, Two Objects, Counting, and Colors are roughly matched. This suggests that the LPD schedule's spatial diversity (avoiding adjacent concurrent tokens) may help with tasks requiring spatial reasoning (position) without hurting other compositional capabilities. However, this is speculative — the paper does not ablate or analyze this per-category pattern.

Notably, LPD achieves these results using only 64 steps for 1024×1024 images (4096 tokens total), meaning each step generates on average 64 tokens in parallel. This is substantially more aggressive parallelization than the 256×256 setting (256/20 = 12.8 tokens/step average), yet quality is maintained, suggesting the method scales well with resolution.

#### Efficiency Analysis: Memory-Bound vs. Compute-Bound Regimes (Figure 14)

Figure 14 provides the most nuanced efficiency characterization. For all three model sizes, LPD's throughput advantage is largest at small batch sizes (memory-bound regime) and diminishes as batch size increases (transition toward compute-bound):

- **LPD-L:** At batch size 1, throughput is 194.2 vs. 65.6 for raster (2.96×). At batch size 64, 25.3 vs. 2.0 (12.7× — the maximum ratio, closely matching the 12.8× step reduction). At maximum batch size (512), 3.2 vs. 1.0 (3.2×).
- **LPD-XL:** At batch size 1, 93.4 vs. 30.9 (3.0×). At batch size 64, 8.7 vs. 0.7 (12.4×). At batch size 256 (max), 0.18 vs. 0.06 (3.0×).
- **LPD-XXL:** At batch size 1, 53.9 vs. 18.6 (2.9×). At batch size 64, 6.6 vs. 0.52 (12.7×). At batch size 256 (max before OOM), 0.13 vs. 0.05 (2.6×).

The paper identifies batch size 16 as the approximate transition from memory-bound to compute-bound for LPD models (where the throughput-vs-batch-size curve begins to plateau), versus batch size ~128 for raster models. This transition happens earlier for LPD because the extra query tokens increase FLOPs per step, making each step more compute-intensive and thus reaching the GPU's arithmetic saturation point at smaller batch sizes.

The ~3× throughput advantage at maximum batch size (compute-bound limit) quantifies the irreducible benefit from reduced step count: even when FLOPs dominate, generating 20 steps instead of 256 provides a ~3× throughput improvement. The additional ~9–10× advantage in the memory-bound regime comes from the fact that the GPU's idle arithmetic units during parameter loading can absorb the extra FLOPs from query tokens essentially for free.

### Ablation Studies and Robustness Checks

All ablation experiments in Section 4 use the LPD-XL model at 256×256 resolution.

**Flexible parallelized autoregressive modeling (mutual visibility) — Figure 9a:** To isolate the architectural contribution (mutual visibility among concurrent tokens) from the ordering contribution, the paper compares LPD, RandAR, and ARPG all using **random generation order** (removing the locality-aware schedule). At 256 steps (minimal parallelization), all three methods achieve similar FID (~2.0). As steps decrease (parallelization increases), LPD degrades substantially less: at 32 steps, LPD maintains ~2.0 FID while RandAR rises to ~2.25 and ARPG to ~2.3. At 16 steps, LPD is ~2.15, RandAR ~2.6, ARPG ~2.65. The growing gap confirms that mutual visibility becomes increasingly important as group sizes increase — without it, the inconsistency cost of independent parallel prediction compounds with parallelization ratio. This also explains why prior methods (RandAR at 88 steps, ARPG at 64 steps, PAR at 147 steps) couldn't push to 20 steps: their architectures would have suffered unacceptable quality degradation.

**Generation order schedule — Figure 9b:** Comparing LPD's locality-aware schedule against random order and Halton low-discrepancy order, all using the same LPD architecture. At 256 steps (different orders but minimal parallelization, so order matters little), all three achieve similar FID. At 20 steps, LPD order achieves ~2.10 FID, Halton ~2.25, random ~2.35. The gap is consistent across step counts from 16 to 64. Halton outperforms random because it spreads concurrent tokens to reduce intra-group dependency (principle 2), but underperforms LPD because it doesn't account for proximity to existing context (principle 1). The diminishing gap at 256 steps is expected — at near-sequential generation, grouping is minimal, so order has little impact.

**Individual locality principles — Figure 9c:** Using LPD-XL at 32 steps, four conditions are compared:
- Random order (no principles): 2.11 FID
- Principle 1 only (select tokens close to existing context, no repulsion among concurrent tokens): 2.00 FID
- Principle 2 only (farthest-point sampling at each step to separate concurrent tokens, ignoring proximity to context): 2.06 FID
- Both principles (LPD schedule): 1.92 FID

The additive improvement (2.11 → 2.00 via P1, 2.11 → 2.06 via P2, 2.11 → 1.92 via both) demonstrates that the principles are complementary, not redundant. Principle 1 provides a larger gain alone (0.11 FID) than Principle 2 (0.05 FID), suggesting that strong conditioning from nearby context is more important than avoiding intra-group dependency at this parallelization level. However, the synergy (0.08 FID beyond the sum of individual gains) is notable — having both enables the algorithm to select tokens that are simultaneously close to context AND spatially separated, which neither principle alone can guarantee.

**Sensitivity to repulsion threshold ρ (Table 6, left, τ fixed at 2):** ρ controls the minimum Euclidean distance between concurrently generated tokens. At ρ = 0 (no repulsion), FID = 1.94. At ρ = 1 (default, tokens must be at least 1 grid unit apart), FID = 1.92. At ρ = 2 (tokens must be ≥2 units apart), FID = 2.04. The U-shaped pattern suggests a sweet spot: too little repulsion places high-dependency tokens together (violating principle 2), while too much repulsion makes it difficult to find tokens that are both close to context and far from each other (violating principle 1). The relatively flat response from ρ = 0 to ρ = 1 (1.94 vs. 1.92) is attributed to the fact that even without explicit repulsion, the proximity sorting naturally avoids extreme clustering.

**Sensitivity to proximity threshold τ (Table 6, right, ρ fixed at 1):** τ controls the minimum proximity score for the high-proximity selection pool c₁. At τ = 0 (all candidates enter c₁), FID = 1.94 — the repulsion filter and sorting still provide some benefit, but without proximity prioritization, some selected tokens may be far from context. At τ = 2 (default), FID = 1.92. At τ = 4 (very strict, few candidates qualify for c₁), FID = 2.00 — the c₁ pool becomes too small, forcing heavy reliance on c₂ and farthest-point sampling, which doesn't optimize principle 1. The flat region from τ = 1 to τ = 2 (1.94 to 1.92) suggests the algorithm is robust to moderate threshold choices.

**Scaling to higher resolution (Tables 1 vs. 2):** The paper demonstrates that the method transfers directly to 512×512 resolution by training on the larger grid with interpolated positional embeddings. The step reduction ratio is even larger at 512×512 (21.3×) than at 256×256 (12.8×), and quality is preserved at both resolutions. This is a robustness check for the architecture's ability to handle different token counts and grid sizes. The text-to-image extension to 1024×1024 (Table 3) further confirms resolution scalability, showing improved GenEval scores over the raster baseline at 64 steps (a 64× step reduction from 4096).

**Model size scaling (Table 1):** The method is tested across a 4.2× parameter range (337M to 1.4B). FID improves from 2.40 (L) to 2.10 (XL) to 2.00 (XXL) at 20 steps, confirming that the benefits of scaling model capacity are preserved under aggressive parallelization. The raster counterparts show similar scaling (2.48 → 2.12 → 2.01), indicating parallelization does not distort the scaling law.

### Critical Assessment

#### Claim 1: "LPD reduces generation steps from 256 to 20 without compromising quality"

**What the experiments demonstrate:** The raster counterpart baselines provide the cleanest test of this claim, since they use identical tokenizers, model architectures, training recipes, and data — the only difference is the generation paradigm (parallel vs. sequential). At all three model sizes, LPD at 20 steps achieves FID equal to or better than the raster counterpart at 256 steps: LPD-L 2.40 vs. raster 2.48; LPD-XL 2.10 vs. 2.12; LPD-XXL 2.00 vs. 2.01. If we take these numbers at face value, the claim holds.

**Genuine weaknesses:** The paper does not report statistical uncertainty on FID. Prior work has shown that FID computed on 50k samples can have non-negligible variance depending on the reference set and sampling procedure. An FID difference of 0.01–0.08 (the raster vs. LPD gaps) could plausibly fall within estimation noise. The paper would be stronger with confidence intervals, multiple FID evaluations with different random seeds, or a statement about measurement uncertainty.

Additionally, the "without compromising quality" framing treats FID as the sole quality metric. Looking at the secondary metrics (IS, Precision, Recall in Table 1), a more nuanced picture emerges: LPD tends to have higher IS (e.g., LPD-XXL 337.6 vs. raster 316.0) and comparable or slightly lower Precision (0.80 vs. 0.80) and slightly higher Recall (0.60 vs. 0.59). Higher IS with comparable Precision/Recall could indicate a different quality-diversity tradeoff — LPD might produce samples that are more "recognizable" (higher IS) without sacrificing coverage. This is not a weakness per se, but it suggests that "quality" is multi-dimensional, and LPD shifts the distribution slightly rather than perfectly matching it.

#### Claim 2: "LPD achieves at least 3.4× lower latency than previous parallelized autoregressive models"

**What the experiments demonstrate:** The comparison is concrete: LPD-XL at 32 steps (1.92 FID, 0.66s) vs. ARPG-XXL at 64 steps (1.94 FID, 2.24s). Latency ratio: 2.24 / 0.66 = 3.39×. The 3.4× figure is the minimum across the comparisons the paper highlights; the LPD-XL at 20 steps vs. ARPG-XL at 64 steps comparison yields 1.71 / 0.41 = 4.2×. These are fair comparisons in the sense of matching FID and model scale.

**Genuine weaknesses:** The latency measurements are all on a single A100 GPU. This is standard but means the reported speedups are hardware-specific. On newer hardware with higher memory bandwidth (e.g., H100), the memory-bound regime might extend to larger batch sizes, potentially making LPD's advantage larger. Conversely, on hardware with lower memory bandwidth relative to compute, the advantage might shrink. The paper acknowledges the memory-bound vs. compute-bound transition but doesn't test on multiple hardware generations. Additionally, the latency is measured at batch size 1, which is the most favorable scenario for step-reduction methods. At larger batch sizes where the system becomes compute-bound, the speedup diminishes to ~3× (Figure 14), so "at least 3.4×" is specifically a batch-size-1 claim.

A subtler issue: the comparison conditions on equivalent FID, but FID is not transitive. LPD-XL at 32 steps (1.92 FID) vs. ARPG-XXL at 64 steps (1.94 FID) — is 1.92 vs. 1.94 a meaningful quality difference, or are these within noise? If they're effectively equal quality, the 3.4× latency claim is clean. If 1.92 is genuinely better than 1.94, then LPD achieves *better* quality at *lower* latency, which is even stronger. If the reverse (1.94 is better), the comparison is slightly unfair to ARPG. Without uncertainty quantification, we can't distinguish these scenarios.

#### Claim 3: "Mutual visibility among concurrently generated tokens is essential for high degrees of parallelization"

**What the experiments demonstrate:** Figure 9a provides direct evidence: when all methods are forced to use random order (isolating the architectural contribution from the scheduling contribution), LPD (which has mutual visibility) maintains FID near 2.0 at 32 steps, while RandAR and ARPG (which lack mutual visibility) degrade to ~2.25 and ~2.3 respectively. The gap grows as steps decrease, consistent with the claim that mutual visibility becomes more important as parallelization increases.

**Genuine weaknesses:** The ablation uses random order for all methods, which is not the operating point of RandAR or ARPG in their original papers (RandAR uses a Halton-based order; ARPG uses its own ordering). This is a valid controlled comparison, but it means the Figure 9a curves don't represent the performance one would get by actually using RandAR or ARPG at 16 steps with their *intended* ordering schemes. The claim is about architectural capability, so the controlled comparison is appropriate, but readers should not interpret Figure 9a as "RandAR is this bad at 32 steps" — RandAR's published configuration uses 88 steps, presumably because the authors recognized the degradation at fewer steps.

A missing experiment that would strengthen this claim: train an LPD variant with a causal mask instead of mutual visibility among query tokens (everything else identical), and compare to the full LPD. This would isolate the mutual visibility component within the LPD architecture without the cross-architecture confounds of comparing LPD to RandAR/ARPG. The current comparison confounds mutual visibility with other architectural differences (encoder-decoder vs. decoder-only, query token caching vs. non-caching).

#### Claim 4: "Locality-aware generation ordering is grounded in attention locality and improves generation quality"

**What the experiments demonstrate:** The attention analysis (Figures 2, 7, 10, 11) clearly shows spatial locality in LLAMAGEN's attention patterns, with per-token attention dropping from ~8% at distance 1 to <1% at distance 10. These measurements are systematic (50k images, quantified via Equation 3, consistent across model sizes and attention heads). The ordering ablation (Figure 9b) shows locality-aware ordering consistently outperforms random and Halton ordering across step counts, and Figure 9c shows additive gains from the two principles.

**Genuine weaknesses:** The attention analysis is conducted on LLAMAGEN, not on LPD itself. The paper uses LLAMAGEN's attention patterns to derive principles, then applies those principles to LPD's generation order. This assumes that the attention locality observed in LLAMAGEN transfers to LPD — that LPD, when trained, will exhibit similar locality in *its* attention patterns. This is plausible (locality is a general property of image data, not a model-specific quirk), but it is an assumption, not a verified fact. The paper does not analyze attention patterns in trained LPD models to confirm that the locality principles actually hold in the deployed system.

Additionally, the ablation of individual principles (Figure 9c) uses a specific operationalization: principle 1 is proximity-based greedy selection, principle 2 is farthest-point sampling. The "principle 1 only" condition removes repulsion but keeps proximity sorting; the "principle 2 only" condition uses FPS without proximity prioritization. These are reasonable operationalizations, but they conflate the *principle* with its *specific algorithmic implementation*. Would a different implementation of principle 1 (e.g., selecting the single closest token rather than sorted greedy) yield similar gains? The sensitivity analysis (Table 6) partially addresses this for the hyperparameters τ and ρ, but doesn't test alternative algorithmic formulations of the principles.

#### Missing Experiments and Ablations

Several experiments would strengthen the paper's claims:

**LPD with varying degrees of mutual visibility.** Train models where the query attention block uses causal masking (RandAR-style batched next-token), independent masking (ARPG-style no mutual visibility), and full mutual visibility (LPD), all within the same LPD architecture. This would cleanly isolate the mutual visibility contribution.

**Attention analysis on trained LPD models.** Replicate Figure 7 using LPD's own attention patterns to verify that the locality principles derived from LLAMAGEN transfer. This would close the loop on the "first-principles" claim.

**Ablation of cosine schedule for group sizes.** The paper states that the number of tokens per step follows a cosine schedule (Appendix A.2) but doesn't ablate this choice. Would a linear schedule, exponential schedule, or constant group size work as well? The cosine schedule is well-motivated by the context accumulation intuition, but empirical validation would strengthen the claim.

**Comparison with speculative decoding or Jacobi decoding approaches.** The paper mentions speculative decoding and Jacobi methods in related work (Section 5.2) but doesn't compare against them experimentally. There are training-free approaches (Teng et al., 2024) that accelerate autoregressive image generation without architectural changes. A comparison would clarify whether LPD's training-based approach provides advantages over training-free acceleration.

**Generalization to other tokenizers.** All experiments use the LLAMAGEN tokenizer. Would LPD work with other VQ tokenizers (e.g., VQGAN, MAGVIT-v2 tokenizer)? The paper argues that flat-token compatibility is a key advantage over VAR's multi-scale tokens, but doesn't demonstrate that LPD works across multiple flat tokenizers.

**Inference cost of the generation schedule itself.** The schedule is precomputed offline, so it adds zero latency. But for very high resolutions or non-square aspect ratios, the schedule computation cost could become non-trivial. The paper doesn't report schedule computation time for 1024×1024 generation (4096 tokens), where the greedy + FPS algorithm would need to process a larger candidate set.

**Per-category FID or quality breakdown.** The paper reports aggregate FID across all 1,000 ImageNet classes. Are there classes where LPD performs noticeably worse than the raster baseline (e.g., classes with fine-grained texture where intra-group inconsistency would be more visible)? A per-class analysis would reveal failure modes that aggregate metrics obscure.

## 6. Limitations and Trade-offs

### The Difficulty Estimation Cost Is Omitted from All Efficiency Calculations

**The assumption or constraint.** The compute-optimal framework requires knowing each prompt's difficulty *before* allocating the inference budget. The paper's method for obtaining this estimate is generating 2048 samples per question and scoring them — either against ground-truth answers (oracle) or against the PRM (predicted). The paper explicitly acknowledges this cost is unaccounted for:

> “estimating difficulty in this way still incurs additional computation cost during inference... our experiments do not account for this cost largely for simplicity” (Section 3.2).

**The consequence.** The headline efficiency gain — “more than 4× better efficiency over a standard best-of-N baseline” (Section 1) and specifically achieving best-of-64 quality with only 16 generations — is computed *after* difficulty is known, without amortizing the 2048-sample estimation cost. In a deployment scenario, the total compute for a single query would be 2048 (difficulty estimation) + 16 (strategy execution) = 2064 generations, which is ~32× larger than the best-of-64 baseline the method claims to beat. Even if difficulty estimates are amortized across many queries (the paper suggests this in Section 3.2), the upfront cost per query type is enormous — 2048 generations to characterize a single prompt's difficulty, meaning the method only becomes net-beneficial if that prompt (or similar prompts at the same difficulty level) is served thousands of times. For one-off or long-tail queries, the overhead dwarfs any savings.

**What evidence exists in the paper.** The paper is transparent about this (Section 3.2, quoted above), and the compute-optimal scaling curves (Figures 4 and 8) correctly exclude the difficulty estimation cost. However, the abstract and introduction prominently advertise the “4×” figure without the caveat, which could mislead a practitioner skimming for deployment guidance. The difficulty estimation protocol is described in full in Section 3.2, and the authors explicitly frame it as “a key avenue for future work.”

**Mitigation status.** The authors suggest two future directions (Section 3.2): (1) pretraining or fine-tuning a model to predict difficulty directly from the question text, avoiding the 2048-sample cost, and (2) viewing difficulty estimation as an exploration-exploitation tradeoff that could be optimized. Neither is implemented or evaluated in the paper. The “predicted” difficulty bins (using PRM scores instead of ground-truth labels) remove the dependence on labeled data but *do not* reduce the generation cost — 2048 samples are still required. The limitation remains fully unresolved in the current work.

---

### All Results Are on a Single Benchmark with a Single Model Family

**The assumption or constraint.** Every experiment in the paper uses the MATH benchmark (Hendrycks et al., 2021) — specifically the 500-question test split from Lightman et al. (2022) — with PaLM 2-S* as the base model. The paper states that it “believe[s] this model is representative of the capabilities of many contemporary LLMs” (Section 4), but provides no empirical evidence that the difficulty-dependent scaling patterns generalize.

**The consequence.** A practitioner cannot know whether the compute-optimal allocation strategies derived here will transfer to their domain, model, or task. Several specific concerns arise:

- MATH consists exclusively of competition-level mathematics problems requiring symbolic multi-step reasoning. Tasks requiring factual recall, code generation, creative writing, or open-ended dialogue have fundamentally different error patterns and may exhibit different difficulty-dependent behavior. A model might benefit from revisions on factual tasks in ways that differ from math (e.g., revisions might help with factual hallucination correction), or might not benefit at all.
- PaLM 2-S* has specific calibration properties, output distributions, and failure modes that affect the PRM's training quality and over-optimization behavior. A different base model (e.g., a Llama-family model or GPT-4) might produce a PRM with different reliability characteristics, shifting the over-optimization threshold and changing which search algorithms are optimal at which difficulty levels.
- The 500-question test set split into five difficulty quintiles of ~100 each, then further split by two-fold cross-validation for strategy selection, means the compute-optimal policy is selected based on roughly 50 questions per fold per bin (Section 3.2). This is a small sample, and the selected strategies may overfit to the idiosyncrasies of these specific 100 questions per bin. The paper does not report confidence intervals or standard errors on the compute-optimal scaling curves (Figures 4 and 8), leaving the statistical reliability of the 4× efficiency claim uncertain.

The FLOPs-matched comparison (Section 7) uses a specific 14× larger PaLM 2 variant with greedy decoding, and the paper acknowledges this model may not be compute-optimally trained (scaling parameters only, not data, following LLaMA-style training rather than Chinchilla-optimal training). The 14× figure is specific to this model pair and this training recipe; a different pretraining scaling regime could produce different tradeoff curves.

**What evidence exists in the paper.** The limitation is explicitly acknowledged only indirectly — the paper describes PaLM 2-S* as “representative” (Section 4) but provides no cross-model or cross-benchmark validation. The sensitivity of the revision training to methodology is demonstrated inadvertently by the ReST^EM failure (Appendix K, Figure 16), where an alternative optimization approach causes performance to degrade substantially with sequential revisions — evidence that revision behavior is fragile to training details in ways that may not generalize.

**Mitigation status.** Not addressed. The authors do not claim generalizability beyond the studied setting, but the paper's framing (particularly the abstract and introduction) presents the findings as general principles about test-time compute scaling rather than as findings contingent on MATH and PaLM 2-S*. Future work would need to replicate the difficulty-dependent scaling analysis on other benchmarks (code generation, factual QA, logical reasoning) and other model families to establish generalizability.

---

### Test-Time Compute Cannot Help on Problems Outside the Base Model's Capability Range

**The assumption or constraint.** The entire compute-optimal framework assumes the base model can produce correct solutions at some non-trivial rate — otherwise there are no correct trajectories to find via search or refine via revisions. The paper demonstrates this boundary explicitly for difficulty bin 5 (the hardest quintile).

**The consequence.** On the hardest MATH problems, the base model's pass@1 is near zero. Across all methods and all budgets, bin 5 accuracy hovers at roughly 1–3% (Figure 3, right; Figure 7, right). In the FLOPs-matched comparison (Figure 9), the bin 5 scaling line is essentially flat near 0–5%, and the 14× larger pretrained model substantially outperforms the smaller model with any amount of test-time compute. The paper states this clearly in the Section 7 takeaway:

> “test-time compute can amplify existing capability but cannot create it from nothing.”

For a practitioner, this means the method offers no path forward for genuinely novel or out-of-distribution problems that exceed the base model's training distribution. If a deployment faces a stream of problems where even the best base model's pass@1 is consistently near zero, investing in test-time compute strategies will yield negligible returns compared to investing in better pretraining, more data, or larger models.

This boundary condition also interacts with the difficulty estimation problem. If the difficulty estimator cannot reliably distinguish “hard but solvable” (bin 4) from “essentially unsolvable” (bin 5) without the oracle, the system might waste significant compute on the latter category, applying aggressive search or revision strategies to problems where no amount of effort will help. The paper's predicted bins (using PRM scores) correlate with oracle bins but the confusion matrix between bins 4 and 5 is not reported.

**What evidence exists in the paper.** Bin 5 results are consistently flat across all experiments — Figure 3 (right, search methods), Figure 7 (right, revision ratios), Figure 9 (FLOPs-matched comparison). The paper is transparent about this finding, explicitly stating the capability boundary in the Section 7 takeaway box.

**Mitigation status.** The limitation is inherent to the approach — it is not a fixable bug but a fundamental property of test-time compute. The paper correctly identifies it and frames the practical implication: pretraining is the only viable path for problems outside the base model's capability range. The difficulty estimator could potentially be improved to identify bin-5 problems early and route them to a larger model or a human, preventing wasted compute — but the paper does not explore this routing use case.

---

### The PRM-Based Difficulty Estimation Uses 2048 Samples Per Query and Is Impractical for Deployment

**The assumption or constraint.** Beyond the general difficulty estimation cost discussed in Limitation 1, there is a specific practical problem: the “predicted” difficulty bin assignment requires generating and PRM-scoring 2048 samples *per query*. Even setting aside the cost of those samples (already covered), this imposes a latency and infrastructure requirement that may be prohibitive.

**The consequence.** The paper frames the predicted difficulty approach as the deployable alternative to oracle difficulty (which requires ground-truth labels). However, generating 2048 samples from a PaLM 2-scale model and running them through a PRM is itself a substantial computation — comparable to or exceeding the largest test-time budgets studied (256–512 generations). In a production setting with latency constraints (e.g., an interactive assistant), waiting for 2048 samples to be generated and scored before beginning to answer a query would be unacceptable, regardless of total FLOPs.

More subtly, the difficulty estimation process itself uses the base model (to generate the 2048 samples) and the PRM (to score them). Both of these are the same models used in the main inference pipeline, which means the difficulty estimation competes for the same compute resources and may need to be serialized with the actual answer generation if only one model replica is available.

**What evidence exists in the paper.** The 2048-sample protocol is described in Section 3.2. The paper provides no analysis of the wall-clock latency added by this step, no comparison of total cost (estimation + generation) against baselines, and no experiment reducing the number of estimation samples to see whether fewer samples (e.g., 64 or 256) might suffice for coarse bin assignment.

**Mitigation status.** The authors flag this as a key direction for future work (Section 3.2): “estimating difficulty in this way still incurs additional computation cost during inference... we leave learning a difficulty estimation model... to future work.” The suggestion is to train a lightweight model that predicts difficulty directly from the question text without any sampling. This is promising — a classifier that takes question embeddings and outputs a difficulty bin could be extremely cheap — but it is not implemented or evaluated, and its accuracy relative to the 2048-sample PRM-based estimator is unknown. If such a classifier had high confusion between adjacent bins (e.g., misclassifying bin-3 problems as bin-4), the compute-optimal policy might select substantially suboptimal strategies, potentially erasing the 4× gains.

---

### Revisions and PRM Search Are Never Combined, Leaving Gains on the Table

**The assumption or constraint.** The paper studies two complementary test-time compute mechanisms — PRM-guided search (Section 5) and iterative revisions (Section 6) — but evaluates them independently. Section 8 explicitly acknowledges this gap:

> “we did not experiment with PRM tree-search techniques in combination with revisions”

**The consequence.** The paper's difficulty-dependent analysis reveals that revisions excel on easy problems (where refinement of near-correct answers suffices) while PRM search excels on medium-hard problems (where exploration of different solution strategies helps). A combined system could in principle use the revision model as the proposal distribution within PRM-guided beam search — at each search step, the model conditions on previous incorrect branches as revision context, potentially generating higher-quality candidate steps than the base model would. Alternatively, the PRM's per-step scores could guide *which* revisions to pursue and when to restart from scratch rather than blindly sequencing revisions.

Because these mechanisms are studied in isolation, the reported performance numbers (e.g., ~44% accuracy at 256 generations for compute-optimal revisions, Figure 8; ~39.5% for compute-optimal search, Figure 4) represent lower bounds on what a fully integrated system could achieve. A practitioner implementing this approach might reasonably expect to get better performance by combining the methods, but the paper provides no guidance on how to do so or what gains to expect.

**What evidence exists in the paper.** The complementary difficulty-dependent strengths are clearly demonstrated — revisions help most on easy problems (Figure 7, right, bins 1–2) while PRM search helps most on medium problems (Figure 3, right, bins 3–4) — providing strong motivation for combination. But no combined experiment exists. The revision model uses a separately trained ORM rather than the PRM (Appendix J, Figure 15a) because the PRM trained on base model outputs doesn't transfer well to revision model outputs due to distribution shift, adding a practical obstacle to naive combination.

**Mitigation status.** The paper identifies this as future work (Section 8) but offers no preliminary results, analysis of the obstacles, or roadmap for combination. The distribution shift issue (PRM not working on revision outputs) suggests that combining the methods may require retraining the PRM on revision model outputs or developing a unified verifier that works across both proposal distributions — non-trivial challenges that the paper does not address.

---

### The Revision Model Has a 38% Correct-to-Incorrect Reversion Rate

**The assumption or constraint.** The revision model is trained exclusively on sequences where all in-context answers are incorrect, followed by a correct target answer (Section 6.1). This training data construction means the model never sees examples of what to do when the current answer is already correct.

**The consequence.** At inference time, when the revision chain produces a correct answer at some intermediate step, subsequent revision steps will often “revise” it into an incorrect answer. The paper reports that approximately 38% of correct answers in a revision chain get converted back to wrong answers (Section 6.1). This means that simply taking the last output of a revision chain is not viable — the system must use within-chain selection (majority voting or verifier-based selection) to pick the best answer from any point in the chain.

This has several practical implications:
- **The effective yield of sequential revisions is lower than the step-wise pass@1 trajectory suggests.** Figure 6 (left) shows pass@1 improving from ~18% at step 1 to ~25% at step 20, but if 38% of the correct answers produced along the way are subsequently corrupted, the actual number of *useful* correct answers in the chain is lower.
- **The selection mechanism is load-bearing.** The paper uses majority voting or verifier-based selection across the chain to mitigate the reversion problem, but these are imperfect — majority voting may select a popular wrong answer, and the verifier may have its own errors. The ~41.5% accuracy reported for sequential revisions with best-of-N weighted selection at 64 generations (Figure 6, right) includes the effect of this selection mechanism, so the raw performance of the revision model without correction is lower.
- **Longer chains increase both opportunity and risk.** More revision steps create more chances to produce correct answers, but also more opportunities for correct answers to be corrupted. The optimal chain length depends on this tradeoff, which the paper does not systematically analyze.

**What evidence exists in the paper.** The 38% reversion rate is stated without a detailed breakdown (Section 6.1) — for instance, do certain types of correct answers get reverted more frequently than others? The paper's mitigation (within-chain selection) is evaluated indirectly through the sequential revision results, but there is no ablation comparing “take the last revision” versus “select best from chain” as the aggregation strategy, which would quantify the severity of the reversion problem.

**Mitigation status.** Partially addressed via the within-chain selection mechanism, but this is a patch rather than a solution. A more principled approach — training the revision model with some trajectories where the final answer is the same as a previous correct answer (teaching the model to recognize when no revision is needed) — is not explored. The ReST^EM experiment (Appendix K, Figure 16) shows that naive attempts to optimize the revision model further can backfire, suggesting the reversion problem is not trivial to fix. The paper does not frame this as a major open problem, but for a practitioner deploying revision-based systems, it represents a fundamental reliability concern.

## 7. Implications and Future Directions
- How this changes the field
  - LPD shows that decoder-only AR image generators can be both fast and high-quality while preserving flat token compatibility—key for unifying with language models and perception backbones. It narrows the practical gap between AR and diffusion in sampling time without giving up AR’s causal/KV-cache advantages (Sections 1–2; Tables 1–2).

- Follow-up research enabled/suggested
  - Learned or content-aware scheduling:
    - Replace Euclidean-distance heuristics with schedules conditioned on intermediate features or attention maps; potentially learn τ and ρ or the whole selection policy.
  - Broader modalities:
    - Extend to video (temporal–spatial locality), audio spectrograms (time–frequency locality), or 3D grids where locality principles might generalize with different metrics.
  - Integration with decoding accelerators:
    - Combine with speculative decoding or multi-head draft/verify schemes to further cut latency while retaining correctness.
  - Task expansion:
    - Move from class-conditional to text-to-image or instruction-following generation while maintaining flat tokens and flexible ordering.

- Practical applications
  - Interactive image editing (inpainting/outpainting) where arbitrary-order generation reduces latency and supports partial updates (Figure 10).
  - On-device or low-latency deployments (e.g., mobile or edge) where KV-cache efficiency and fewer steps directly translate to responsiveness.
  - Multimodal assistants that benefit from unified token spaces and AR-style conditioning for image understanding and generation.

> Overall, LPD’s two-part design—flexible joint prediction with mutual visibility (Figures 3–6) plus a locality-aware schedule grounded in measured attention patterns (Figures 2, 7–8; Algorithm 1)—reduces ImageNet generation from hundreds to a few dozen steps “without compromising quality” and with “at least 3.4× lower latency” relative to previous parallel AR models (Figure 1; Tables 1–2).
