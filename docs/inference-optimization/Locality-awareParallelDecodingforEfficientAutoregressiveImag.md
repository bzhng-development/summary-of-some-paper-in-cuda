# Locality-aware Parallel Decoding for Efficient Autoregressive Image Generation

**ArXiv:** [2507.01957](https://arxiv.org/abs/2507.01957)

## 🎯 Pitch

This paper introduces Locality-aware Parallel Decoding (LPD), a novel method that dramatically accelerates autoregressive image generation by enabling flexible, parallel prediction of many image patches at once—guided by spatial locality to maximize contextual coherence. By reducing the number of sequential steps from hundreds to just a few dozen without sacrificing image quality, LPD achieves over 3.4× lower latency than prior parallel autoregressive models, making fast, high-quality image synthesis practical for real-world and multimodal AI applications while preserving compatibility with widely used vision foundation models.

---

## 1. Executive Summary
Locality-aware Parallel Decoding (LPD) makes autoregressive image generation much faster without sacrificing quality by (a) changing how tokens are predicted so many image patches can be generated together, and (b) choosing which patches to generate next based on spatial locality. On ImageNet, LPD cuts steps from 256 to 20 at 256×256 and from 1024 to 48 at 512×512, while matching or improving quality and achieving at least 3.4× lower latency than previous parallel autoregressive models (Figure 1; Tables 1–2).

## 2. Context and Motivation
- Problem/gap
  - Standard autoregressive (AR) image generators predict the “next patch” one at a time in a fixed raster order. Each step requires loading model parameters and running attention over the growing context, creating a memory-bound bottleneck with high latency (Section 1; footnote on memory-bound workloads).
  - An alternative, “next-scale” diffusion-like AR prediction reduces steps but uses multi-scale representations incompatible with flat, patch-token representations that underpin widely used vision backbones (e.g., CLIP, DINO), limiting interoperability needed for unified multimodal models (Section 1).

- Why it matters
  - Practical: Fast sampling is crucial for interactive image generation, on-device deployment, and multimodal assistants.
  - Theoretical/systemic: Preserving flat token representations keeps AR image models aligned with language-model-style tokenization and with perception backbones—important for unified multimodal systems (Section 1).

- Prior approaches and shortcomings
  - Next-patch AR (e.g., VQGAN/VQ-VAEs; “LlamaGen”) produce high quality but require many sequential steps (hundreds to thousands) and are memory-bound (Section 1; Table 1).
  - Next-scale AR (e.g., Next-Scale Prediction) reduces steps but breaks compatibility with flat tokens (Section 1).
  - Masked non-autoregressive models (e.g., MaskGIT) predict multiple patches per step but rely on bidirectional attention (no KV caching), making them less efficient at inference (Section 1).
  - Parallelized AR attempts:
    - Encoder–decoder queries (SAR, ARPG) let multiple targets be predicted together but the targets are independent within a step (no mutual influence), harming consistency (Figure 6a).
    - Decoder-only with positional instruction tokens (RandAR) allows arbitrary order, but a standard causal mask in training degenerates joint prediction into batched next-token prediction and requires storing instruction tokens in the cache, doubling memory (Figure 6b).
    - Fixed parallel orders (PAR, NAR) limit flexibility and parallelization degree (Section 2.2, paragraph “Comparison with other methods”).

- Positioning
  - LPD preserves flat tokens and KV caching while enabling arbitrary generation order and true joint prediction in each step. It further maximizes quality under high parallelism with a locality-aware schedule grounded in measured attention locality (Figures 2, 7; Algorithm 1).

## 3. Technical Approach
LPD comprises two parts that work together:

1) Flexible parallelized autoregressive modeling (Sections 2.1–2.2; Figures 3–5)
- What “flexible parallelization” means here
  - Instead of predicting one next token conditioned on all previous tokens, LPD groups many target positions and predicts all of them in a single step while still respecting causality in how context is formed.

- Key idea: decouple “context” from “generation”
  - In standard decoder-only AR, each token both provides context (via its hidden state/keys–values) and is predicted as an output logit. This coupling fixes input–output structure and generation order.
  - LPD separates roles using two token types:
    - Image tokens (context): previously generated tokens that provide keys–values for attention (and are stored in the KV cache).
    - Position query tokens (generation): learnable query vectors that are position-specific. Each is the sum of a shared learnable embedding and a positional embedding of the target location (Section 2.2).
  - By feeding any set of position query tokens, the model can predict arbitrary positions in parallel.

- Training formulation (Figure 4)
  - Interleave ground-truth image tokens and position query tokens.
  - Specialized attention mask with two patterns:
    - Context Attention: subsequent tokens can attend to earlier image tokens causally (teacher forcing preserved).
    - Query Attention: all position queries in the same step can attend to each other (mutual visibility) and to prior context, but subsequent tokens cannot attend to queries. This creates true joint prediction within a step rather than batched next-token prediction.

- Inference formulation (Figure 5)
  - Alternating encode and decode:
    - Encoding: newly sampled image tokens are passed to update the KV cache (so future steps can attend to them).
    - Decoding: a set of position query tokens attends to the cached context to output logits for all target positions simultaneously.
  - Fusion to avoid doubling steps: A specialized inference mask lets a single forward pass both (a) encode the newly generated image tokens into the cache and (b) decode the next group of queries. Importantly, queries are not stored in the cache—only actual image tokens—so memory stays low.

- Why this attention/masking matters
  - Mutual visibility among queries ensures jointly generated tokens in the same step can “coordinate,” improving within-step consistency at high parallelism. This is the main architectural difference from SAR/ARPG (no mutual influence among targets) and from RandAR (causal mask makes it batched next-token; Figure 6).

2) Locality-aware generation order schedule (Section 2.3; Figures 2, 7–8; Algorithm 1)
- Empirical observation
  - Decoding attention has strong spatial locality: tokens attend mostly to nearby tokens. This is shown qualitatively (Figure 2) and quantitatively by Per-Token Attention (PTA) as a function of distance (Equation 3; Figure 7a) and by attention-sum concentration within local neighborhoods across heads (Figure 7b).
  - Definition: `Per-Token Attention (PTA)` averages the attention weight a decoding token assigns to tokens exactly at Euclidean distance `s` on the 2D grid (Equation 3).

- Two scheduling principles derived from locality (Section 2.3)
  - Principle 1 (high proximity to context): choose next targets close to already generated tokens to maximize conditioning strength.
  - Principle 2 (low proximity within group): ensure concurrently generated targets are far apart to minimize mutual dependency.

- The schedule (Algorithm 1; Figure 8)
  - Precompute a `K`-step plan that partitions all grid positions into groups with increasing sizes `O = [o1, …, oK]` (cosine schedule) so later steps can decode more tokens in parallel thanks to richer context.
  - At each step:
    - Rank unselected positions by proximity to the selected set. Split into `c1` (above threshold τ) and `c2` (remaining).
    - From `c1`, pick positions greedily while ensuring a repulsion radius ρ from already picked positions (Principle 2). Near-by candidates filtered by ρ are moved to `c2`.
    - If fewer than `ok` positions are selected, fill the remainder via farthest point sampling from `c2` to keep targets well separated (Principle 2).
  - The schedule is precomputed offline and reused at inference with zero overhead.

- Why these design choices
  - Decoupling context and generation with specialized masks enables arbitrary ordering and true joint prediction—something standard decoder-only causal masking cannot.
  - The locality-guided order exploits how vision transformers actually allocate attention during decoding, improving sample quality under aggressive parallelization.

## 4. Key Insights and Innovations
- Flexible parallelized AR with mutual visibility (fundamental)
  - What’s new: Separate tokens for context and generation plus a training mask that allows position queries in the same step to see each other (Figure 4) and an inference mask that fuses encode–decode (Figure 5).
  - Why it matters: It turns “batched next-token prediction” into true joint prediction per step, preserving consistency at high parallelism and enabling arbitrary orders. Competing encoder–decoder methods do not allow targets to influence one another within a step (Figure 6a), and decoder-only causal training degenerates parallelism (Figure 6b).

- KV-cache efficiency with query-exclusion (incremental but important)
  - Only generated image tokens enter the cache; position queries do not. This avoids doubling memory versus instruction-token approaches like RandAR (Figure 6b vs. 6c) and keeps inference scalable.

- Locality-aware order scheduler that unifies “stay near context” and “stay far from each other” (fundamental)
  - What’s new: A two-principle schedule based on measured attention locality (Figures 2, 7; Equation 3), balancing contextual support and low intra-group dependency (Algorithm 1; Figure 8).
  - Why it matters: It enables much larger group sizes (fewer steps) without quality collapse. Halton or random orders miss at least one of the two principles (Figure 9b–c).

- Strong empirical reductions in steps and latency with matched quality (fundamental, system-level)
  - On ImageNet, LPD achieves the same or better FID with 12.8–21.3× fewer steps and ≥3.4× lower latency than prior parallel AR methods (Figure 1; Tables 1–2).

## 5. Experimental Analysis
- Evaluation setup (Section 3.1)
  - Datasets: ImageNet class-conditional generation at 256×256 and 512×512 (50k samples for metrics).
  - Tokenizer: LLAMAGEN tokenizer with codebook size 16384 and downsample factor 16 (Models paragraph).
  - Models: Decoder-only transformers of sizes `LPD-L` (337M), `LPD-XL` (752M), `LPD-XXL` (1.4B) (Table 3; note model param count in Table 1 slightly differs due to rounding).
  - Training: 256×256 for 450 epochs; 512×512 continues 50 epochs from the 256×256 checkpoint with positional embedding interpolation. Decoding steps sampled from predefined sets; group sizes follow a cosine schedule (Section 3.1; Appendix A.2).
  - Metrics: FID (primary), Inception Score, Precision, Recall (Section 3.1).
  - Efficiency: Measured on a single NVIDIA A100, bfloat16, latency at batch=1, throughput at batch=64; 100-step warmup; averaged over 500 runs (Section 3.1).

- Main quantitative results on 256×256 (Table 1; Figure 1)
  - Step reduction and quality
    - Raster counterparts (256 steps) vs LPD (20 steps): e.g., `Raster XL` FID 2.12 vs `LPD-XL (20)` FID 2.10—LPD matches quality with 12.8× fewer steps.
  - Against prior parallel AR:
    - `ARPG-XL (64)` FID 2.10, latency 1.71 s vs `LPD-XL (20)` FID 2.10, latency 0.41 s → ~4.2× lower latency with 3.2× fewer steps.
    - `ARPG-XXL (64)` FID 1.94, latency 2.24 s vs `LPD-XL (32)` FID 1.92, latency 0.66 s → similar quality with ~3.4× lower latency.
    - `RandAR-XXL (88)` FID 2.15, latency 3.58 s vs `LPD-XXL (20)` FID 2.00, latency 0.55 s → better quality and ~6.5× lower latency.
  - Throughput (batch=64):
    - `LPD-XL (20)` 75.20 img/s vs `ARPG-XL (64)` 36.53 img/s.
    - `LPD-L (20)` 139.11 img/s vs `RandAR-L (88)` 28.59 img/s (Table 1).

- Main quantitative results on 512×512 (Table 2)
  - Step reduction: 1024 → 48 (21.3× fewer steps).
  - Quality parity: `Raster-XL (1024)` FID 2.09 vs `LPD-XL (48)` FID 2.10 (virtually identical).
  - Latency: `LPD-XL (48)` 1.01 s (batch=1); throughput 18.18 img/s (batch=64).

- Ablations (Figure 9)
  - Architecture ablation—mutual visibility matters (Figure 9a):
    - With random orders for all methods to isolate architecture effects, LPD’s FID degrades far less as steps decrease. At 32 steps, LPD “almost maintain[s] the performance with 256 steps” while ARPG and RandAR degrade significantly.
  - Order scheduler ablation (Figure 9b):
    - LPD’s locality-aware schedule consistently beats Random and Halton across step counts.
  - Principles ablation (Figure 9c):
    - Random: FID 2.11.
    - Principle 1 only (close to context): FID 2.00.
    - Principle 2 only (spread within group): FID 2.06.
    - Both (LPD): FID 1.92. The two principles are complementary.
  
- Qualitative capabilities (Figure 10; Section 3.3)
  - Zero-shot inpainting/outpainting and class-conditional edits by prefilling the KV cache with known tokens and regenerating masked regions in arbitrary order. The flexible order is the enabler; no finetuning is required.

- Do experiments support the claims?
  - Yes, across sizes and resolutions, LPD achieves large step and latency reductions with maintained quality relative to strong raster AR baselines and beats prior parallel AR approaches on both quality and efficiency (Tables 1–2; Figure 1). Ablations credibly isolate the impact of architectural mutual visibility and the locality-aware schedule (Figure 9).

## 6. Limitations and Trade-offs
- Assumptions the approach relies on
  - Strong spatial locality during decoding. The schedule’s two principles hinge on attention concentrating locally (Figures 2, 7). If a task or dataset requires long-range dependencies early (e.g., strong global symmetries), locality-based grouping might be suboptimal.
  - Flat discrete tokenization. The method assumes a VQ-style tokenizer (LLAMAGEN, codebook 16384, downsample 16). Performance inherits its representational limits (Section 3.1).

- Scenarios not addressed
  - Text-to-image and fully open-ended multimodal prompts are not evaluated; the study is strictly class-conditional ImageNet (Tables 1–2; Section 3.1).
  - Non-natural images or structured graphics where Euclidean proximity poorly reflects dependency might violate the scheduler’s assumptions.

- Computational and implementation considerations
  - While latency improves, large models (up to 1.4B) still require substantial GPU memory. Implementing specialized training/inference masks and the fused pass adds engineering complexity (Figures 4–5).
  - The schedule uses fixed Euclidean geometry and static thresholds (τ, ρ). It does not adapt to image content, which could limit optimality on diverse scenes (Algorithm 1; Figure 8).

- Open questions
  - How does LPD perform with different tokenizers/codebooks or with hybrid semantic tokenizations?
  - Can the schedule be made content-aware or learned online to adapt to specific images?
  - Interaction with other acceleration techniques (e.g., speculative decoding) is unexplored.

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
