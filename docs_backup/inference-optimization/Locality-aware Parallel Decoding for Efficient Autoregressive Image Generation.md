# Locality-aware Parallel Decoding for Efficient Autoregressive Image Generation

**ArXiv:** [2507.01957](https://arxiv.org/abs/2507.01957)
**Authors:** Zhuoyang Zhang, Luke J. Huang, Chengyue Wu, Shang Yang, Kelly Peng, Yao Lu, Song Han
**Institutions:** MIT, NVIDIA or MIT (not explicitly stated but authors likely affiliated with MIT and possibly NVIDIA)

## 🎯 Pitch

Locality-aware Parallel Decoding (LPD) revolutionizes autoregressive image generation by enabling simultaneous multi-token prediction through a novel attention mask and locality-aware scheduling, reducing generation steps significantly. This innovation dramatically lowers latency while maintaining quality, crucial for interactive applications and enhancing compatibility with unified multimodal systems.

---

## 1. Executive Summary (2-3 sentences)
This paper introduces Locality-aware Parallel Decoding (`LPD`), a way to make autoregressive image generators much faster without changing their token representation. It enables the model to predict many image tokens at once—guided by a locality-aware schedule—while keeping the quality and the practical efficiencies (like KV caching) of standard autoregressive decoding.

## 2. Context and Motivation
- Problem addressed
  - Autoregressive (AR) image generators typically produce one token at a time in a fixed “raster” order (left-to-right, top-to-bottom). This is slow and memory-bound: each step requires loading model weights even though only one token is produced, so latency increases with the number of steps (Introduction; Figure 3a; footnote 1).
  - An alternative “next-scale” approach generates at multiple resolutions and reduces steps, but it requires multi-scale token representations that are incompatible with the flat, universal tokenizations used by perception backbones like CLIP and DINO (Section 1).

- Why it matters
  - Practical: Lower latency and higher throughput for AR image generation are crucial for interactive applications and deployment.
  - Strategic: Many unified multimodal systems rely on flat image tokens compatible with language-style modeling (Section 1). Speeding up AR with flat tokens improves interoperability with those systems.

- Prior approaches and their limits
  - Non-autoregressive mask-prediction (e.g., MaskGIT) allows multi-token prediction but needs bidirectional attention, which is less efficient than AR and forgoes KV caching during inference (Section 1).
  - Parallelized AR with fixed or ad-hoc orders (e.g., `PAR`, `NAR`) achieves limited parallelization and quality because grouping adjacent tokens causes inconsistencies (Sections 1–2.1).
  - Encoder–decoder AR with target-aware queries (e.g., `SAR`, `ARPG`) generates tokens in parallel but those tokens are independent within a step since queries don’t provide key–value pairs (Figure 6a).
  - Decoder-only AR with positional instruction tokens (e.g., `RandAR`) degenerates to batched next-token prediction under a standard causal mask and doubles KV memory by caching instruction tokens (Figure 6b).

- Positioning of this work
  - `LPD` stays fully autoregressive with flat tokens, but:
    - It decouples “context” from “generation” using learnable position query tokens and a specialized attention mask to jointly predict multiple target positions (Figures 3b, 4, 5).
    - It exploits measured spatial locality in attention to schedule which positions to generate together, maximizing context and minimizing interference (Figures 2, 7; Algorithm 1).

## 3. Technical Approach
LPD has two pillars: a flexible AR modeling architecture that supports joint prediction at arbitrary positions, and a locality-aware schedule for deciding which positions to decode in each parallel step.

A) Rethinking factorization: from next-token to next-group
- Standard AR factorization (Equation 1):
  - Plain English: model the probability of all tokens by multiplying the conditional probability of each token given all prior tokens in a fixed order.
  - Notation: `p(x1,…,xN; c) = ∏_{n=1}^N p(xn | x<n; c)` (Equation 1).
- LPD factorization with groups (Equation 2):
  - Plain English: split tokens into groups, and predict one group at a time conditioned on earlier groups; the order and size of groups are flexible.
  - Notation: `p(x1,…,xN; c) = ∏_{g=1}^G p(Xg | X<g; c)` (Equation 2).

B) Flexible parallelized AR modeling (Figures 3–5)
- Core idea: Decouple context representation from token generation using separate token types.
  - Previously generated “image tokens” provide context (as usual).
  - A set of learnable “position query tokens” asks the model to generate tokens at specific target coordinates in parallel (Figure 3b). Each query is formed by adding the positional embedding of the target location to a shared learnable embedding (Section 2.2).
- Training with a specialized attention mask (Figure 4):
  - Interleave ground-truth image tokens and position query tokens (teacher-forcing).
  - Two attention patterns:
    1) Context Attention: later tokens (queries and future image tokens) can attend to earlier image tokens causally.
    2) Query Attention: position queries within the same step can attend to each other (mutual visibility), but no later token can attend to these queries (so queries don’t leak as context).
  - Why this matters: Joint visibility among same-step queries lets the model resolve conflicts and produce consistent tokens across the group; without it, parallel decoding behaves like independent or batched single-token prediction (Section 2.2; Figure 6 comparisons).
- Inference with a fused encode–decode step (Figure 5):
  - Definitions:
    - `KV cache`: Stored key–value tensors from prior tokens so subsequent steps can skip recomputing attention for earlier positions.
  - Two sub-operations are fused in one forward pass:
    - Encoding: feed the newly generated image tokens to update the KV cache (they become context for future steps).
    - Decoding: feed the position query tokens to obtain logits for all target positions in parallel, but do not store KV for the queries.
  - Benefits: No doubling of steps (encoding and decoding in one pass); memory stays low because only generated image tokens are cached (Figure 6c).

C) How LPD differs from prior parallel AR designs (Figure 6)
- Encoder–decoder with queries (SAR/ARPG): queries don’t contribute KV, so same-step tokens are independent (Figure 6a).
- Decoder-only with instruction tokens (RandAR): the standard causal mask makes it batched next-token prediction; also must cache instruction tokens, doubling memory (Figure 6b).
- LPD: guarantees joint visibility among parallel targets, caches only image tokens; achieves true joint prediction with low memory (Figure 6c).

D) Locality-aware generation order schedule (Figures 2, 7, 8; Algorithm 1)
- Evidence of local attention:
  - Qualitative: attention maps show strong spatial locality—decoding tokens attend most to nearby spatial tokens (Figure 2).
  - Quantitative: “Per-Token Attention” (`PTA`) drops sharply with distance (Figure 7a), and this pattern is consistent across heads (Figure 7b).
  - Definition (Equation 3): `PTA_s` is the average attention mass a token gives to other tokens exactly at Euclidean distance `s` on the 2D grid.
- Scheduling principles (Section 2.3):
  1) High proximity to existing context: target positions should be spatially close to already generated tokens (for strong conditioning).
  2) Low proximity within the group: concurrently generated positions should be far apart (to reduce mutual dependency and conflicts).
- Algorithmic steps (Algorithm 1; Figure 8):
  - Predefine the number of steps `K` and group sizes `O = [o1,…,oK]`, usually increasing (cosine schedule).
  - At each step:
    - Compute a proximity score for unselected positions relative to the selected set (`1 / euclidean distance`).
    - Split into high-proximity (`c1`) and low-proximity (`c2`) candidates using a threshold `τ`.
    - From `c1`, greedily select positions while enforcing a “repulsion” distance `ρ` between picks; nearby candidates are filtered to `c2`.
    - If fewer than `ok` positions selected, fill the remainder via farthest point sampling from `c2` (maximizing separation).
  - The entire schedule is precomputable, adding no runtime overhead. A reference PyTorch implementation is provided (Appendix C).

E) Model and training setup (Section 3.1; Appendix A)
- Tokenizer: LlamaGen tokenizer with codebook size 16,384 and downsample factor 16 (Section 3.1).
- Architecture: standard decoder-only transformer at three scales: `LPD-L` (337M), `LPD-XL` (752M), and `LPD-XXL` (1.4B) parameters (Table 1; Section 3.1). Note: Appendix Table 3 lists smaller parameter counts for “LPD-L” (111M); the main results table consistently uses 337M for `LPD-L`.
- Training:
  - ImageNet class-conditional at 256×256 for 450 epochs, then continue at 512×512 for 50 epochs with positional embedding interpolation (Section 3.1).
  - During training, randomly shuffle token sequences (class token first).
  - Train on a range of decoding steps; group sizes per step follow a cosine schedule (Section 3.1; Appendix A.2).
  - Optimization: AdamW, BF16 precision; classifier-free guidance (CFG) scale swept for evaluation (Section 3.1; Table captions).

## 4. Key Insights and Innovations
- Decoupled context and generation via position query tokens with a specialized attention mask
  - What’s new: Separate tokens for context (image tokens) and for target queries; mutual visibility among parallel queries; queries do not become context (Figures 3–5).
  - Why it matters: Enables true joint prediction of multiple positions while preserving AR efficiencies (KV caching) and avoiding memory bloat (Figure 6c).

- Fused encode–decode inference step that caches only image tokens
  - What’s new: One forward pass both updates KV with new context and decodes multiple target positions; queries aren’t cached (Figure 5).
  - Why it matters: Cuts step count in half relative to a naive two-pass approach and avoids doubling KV size seen in instruction-token methods (Figure 6b vs 6c).

- Locality-aware generation schedule grounded in measured attention behavior
  - What’s new: A principled order that selects positions near existing context but far from each other (Figures 7–8; Algorithm 1).
  - Why it matters: Makes large parallel groups viable without quality loss by exploiting the fact that attention is strongly local in AR image decoders (Figure 7).

- Empirical demonstration of high parallelization with flat tokens and AR modeling
  - What’s new: Reduces steps from 256→20 at 256×256 and from 1024→48 at 512×512 with comparable or better FID than raster-order AR and competitive parallel AR baselines (Tables 1–2).
  - Why it matters: Shows AR with flat tokens can be both fast and high-quality, preserving compatibility with perception backbones (Section 1).

## 5. Experimental Analysis
- Evaluation methodology (Section 3.1)
  - Datasets: ImageNet class-conditional at 256×256 and 512×512.
  - Metrics: FID (50k samples), Inception Score (IS), Precision/Recall. CFG scales swept with 0.1 granularity.
  - Efficiency: Single A100 GPU, BF16; latency with batch size 1, throughput with batch size 64; 500 measured runs per configuration after a 100-step warm-up.

- Main quantitative results
  - Step reductions with maintained or improved quality
    - 256×256: Steps reduced 256→20 (12.8× fewer) with stable or better FID relative to raster-order counterparts (Table 1).
      - `LPD-XL` at 20 steps: FID 2.10, latency 0.41 s, throughput 75.20 img/s.
      - Raster Counterpart-XL (256 steps): FID 2.12, latency 5.29 s, throughput 12.31 img/s.
      - Parallel baselines:
        - `ARPG-XL` (64 steps): FID 2.10, latency 1.71 s → LPD-XL is 4.2× lower latency at matched FID (Table 1; also highlighted in Figure 1).
        - `RandAR-XL` (88 steps): FID 2.25, latency 2.78 s → LPD-XL is better quality and 6.8× lower latency.
      - Increasing to 32 steps, `LPD-XL` reaches FID 1.92 with latency 0.66 s, matching or beating larger baselines (e.g., `ARPG-XXL` FID 1.94, 2.24 s; Table 1).
    - 512×512: Steps reduced 1024→48 (21.3× fewer) with comparable quality (Table 2).
      - `LPD-XL` at 48 steps: FID 2.10, latency 1.01 s.
      - Raster Counterpart-XL at 1024 steps: FID 2.09, latency 20.93 s.
  - System-level perspective
    - Figure 1 visualizes that `LPD` achieves “at least 3.4× lower latency” than prior parallel AR at similar quality on ImageNet 256×256.

- Ablations and diagnostics (Section 4; Figure 9)
  - Architecture-level benefit (Figure 9a):
    - With random generation order for all methods (to isolate architecture), LPD’s joint-query design degrades far less when steps shrink. At 32 steps, `LPD-XL` nearly matches its 256-step FID, while `ARPG` and `RandAR` degrade significantly.
  - Scheduling benefit (Figure 9b):
    - LPD’s locality-aware order outperforms both Random and Halton orders across steps.
    - Insight: Halton spreads tokens uniformly (good for low intra-group dependency), but ignores proximity to existing context; LPD does both.
  - Principles ablation (Figure 9c):
    - Random order: FID 2.11.
    - Principle 1 only (close to context): 2.00.
    - Principle 2 only (far within group): 2.06.
    - Both principles (LPD): 1.92.
    - Conclusion: Both proximity-to-context and intra-group repulsion are necessary and synergistic.
  - Zero-shot editing (Figure 10):
    - Inpainting/outpainting: prefill KV with non-masked tokens, then generate masked regions (random order).
    - Class-conditional edits: replace the class embedding and regenerate a region.
    - These tasks show the utility of arbitrary-order generation enabled by LPD’s architecture.

- Do the experiments support the claims?
  - Yes, on class-conditional ImageNet:
    - Large, consistent step reductions at both 256 and 512 resolutions with maintained or improved FID relative to raster AR (Tables 1–2).
    - Strong speedups vs. parallel AR baselines at matched FID (Figure 1; Table 1).
    - Ablations distinguish the contributions of the architecture and the schedule (Figure 9).

## 6. Limitations and Trade-offs
- Assumptions about spatial locality
  - The schedule assumes attention is strongly local (Figures 2, 7). If a task or tokenizer induces more global dependencies (e.g., structured text rendering, extreme global symmetries), gains might diminish or ordering might need adaptation.

- Task scope and conditioning
  - Experiments focus on class-conditional ImageNet. The paper does not report text-to-image evaluations, so robustness to diverse prompts or open-domain semantics remains to be demonstrated (Tables 1–2).

- Hyperparameters and scheduling details
  - The schedule uses thresholds `τ` (proximity) and `ρ` (repulsion), plus a stepwise group-size schedule (cosine). These add knobs to tune; the paper shows effectiveness but not an automated selection strategy (Section 2.3; Algorithm 1).

- Compute and training scale
  - Training lasts 450 epochs at 256×256, then 50 more at 512×512 (Section 3.1). Although the method improves inference latency, training still requires substantial resources typical of state-of-the-art AR generators.

- Tokenizer coupling
  - Results rely on the LlamaGen VQ tokenizer (codebook 16,384; downsample 16). Different tokenizers may influence locality patterns and generation dynamics; portability was not tested here (Section 3.1).

- Minor reporting inconsistency
  - Appendix Table 3 lists `LPD-L` as 111M parameters, while the main results table uses 337M for `LPD-L` (Table 1). The results consistently reference the larger figure; clarification would aid reproducibility.

## 7. Implications and Future Directions
- How this changes the landscape
  - Demonstrates that flat-token autoregressive image generation can be highly parallel and low-latency without sacrificing AR’s strengths (KV caching, causal context) or compatibility with multimodal systems (Section 1). This bridges the speed gap often cited in favor of diffusion or multiscale AR while maintaining interoperability.

- Follow-up research enabled or suggested
  - Adaptive scheduling: Learn the schedule parameters (`τ`, `ρ`, group sizes) or decide per-image which positions to decode next, potentially conditioned on uncertainty estimates.
  - Beyond 2D images: Extend to video (spatiotemporal grids) or 3D representations where locality is richer and scheduling becomes multi-dimensional.
  - Integration with decoding accelerators: Combine with speculative decoding or multi-head decoding (e.g., Medusa-like approaches) for further reductions in wall-clock time.
  - Tokenizer co-design: Explore tokenizers that accentuate locality without hurting semantics, or hybrid schemes that adapt codebooks to LPD’s scheduling.
  - Broader conditioning: Validate on text-to-image and image-to-image tasks beyond classes; integrate with large multimodal models to test the promised interoperability.

- Practical applications
  - Faster AR image synthesis for interactive design, on-device generation, or streaming creative tools.
  - Efficient, arbitrary-order inpainting/outpainting and region-specific edits at inference time (Figure 10), useful in graphics, photo editing, and content creation pipelines.

> Headline result: “We reduce the generation steps from 256 to 20 (256×256) and 1024 to 48 (512×512) without compromising quality … achieving at least 3.4× lower latency than previous parallelized autoregressive models.” (Abstract; Figures 1, 9; Tables 1–2)

> Mechanism in a sentence: A specialized attention mask lets many position query tokens see each other and the existing context in a single fused encode–decode step, while a locality-aware schedule picks target positions that are close to context but far from each other (Figures 4–5, 7–8; Algorithm 1).
