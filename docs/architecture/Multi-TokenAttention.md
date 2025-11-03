# Multi-Token Attention

**ArXiv:** [2504.00927](https://arxiv.org/abs/2504.00927)

## 🎯 Pitch

This paper introduces Multi-Token Attention (MTA), a novel attention mechanism that overcomes a key limitation of standard transformers by allowing each attention weight to depend on multiple query tokens, key tokens, and attention heads simultaneously. By convolving attention scores across these dimensions, MTA enables large language models to focus more precisely on relevant context—especially when retrieving information spread across long sequences—yielding significant gains in language modeling and long-context retrieval tasks. This innovation is impactful because it addresses a fundamental bottleneck in how current models reason over and integrate complex cues, directly boosting their ability to handle challenging, realistic scenarios where salient information is distributed throughout lengthy texts.

---

## 1. Executive Summary (2-3 sentences)
This paper introduces Multi-Token Attention (MTA), a drop-in replacement for standard multi-head attention that lets each attention weight depend on multiple query tokens, multiple key tokens, and multiple heads simultaneously. By convolving attention scores along the query, key, and head dimensions (before and/or after the softmax), MTA focuses more precisely on relevant context, which yields consistent gains in language modeling and long-context retrieval (Tables 2–4; Figures 3, 5).

## 2. Context and Motivation
- Problem addressed
  - Standard attention computes each weight from a single query–key similarity (Equation 1). This “single-token attention” bottleneck means the model must compress all the cues needed to recognize a relevant passage into one query vector and one key vector at a time (Section 3, Figure 1 left).
  - Many real queries require matching combinations of cues spread across multiple tokens. Example: finding a sentence that mentions both “Alice” and “rabbit” requires combining evidence from multiple query tokens and multiple key tokens (Section 3).
- Why this matters
  - Practically: LLMs often fail to retrieve “needles” hidden in long contexts, especially in the middle of documents (Section 4.4; references to Liu et al. 2024/2025). Stronger, more precise attention can improve reading comprehension, code search, document QA, and long-context tasks.
  - Conceptually: The paper exposes a structural limitation of single-vector similarity and offers a principled, learnable way to aggregate attention evidence across tokens and heads.
- Prior approaches and gaps
  - “Sharpening” attention (e.g., sparsemax, adaptive/scalable softmax temperatures) focuses weights but still bases each weight on a single query–key pair (Related Work; Section 5).
  - “Talking-heads” attention mixes heads linearly before/after softmax, but does not combine evidence across neighboring queries and keys (baseline in Tables 2–4).
  - DIFF Transformer compares two attention maps to cancel noise, but again does not let a single weight condition on multiple query–key pairs (baseline in Tables 2–4).
  - Other convolutional augmentations typically operate on hidden states or compress keys/values, not on the attention maps across queries, keys, and heads simultaneously (Section 5).
- Positioning
  - MTA is a minimal modification to attention that directly mixes attention maps over local neighborhoods of queries, keys, and heads via learnable convolutions (Figures 1–2). It complements head mixing ideas (Talking-heads, DIFF) by adding multi-token mixing along the query–key plane.

## 3. Technical Approach
At a glance: replace “each weight depends on one (q, k) pair” with “each weight can depend on a learned combination of several nearby queries and keys, and on other heads,” implemented via convolutions on the attention logits/weights.

Step 1: Baseline attention (for reference)
- Standard multi-head attention for one head uses query `Q`, key `K`, value `V` and attention logits `Â = QKᵀ/√d`, followed by a causal mask and softmax (Equation 1; Section 2). Output is `AV`, concatenated across heads and projected.

Step 2: Key–query convolution (mixing across tokens within a head)
- Idea in plain words: Before softmax, treat the attention logit matrix for a head as an “image” whose axes are queries (rows) and keys (columns). Apply a small 2D convolution to each local patch so that each logit at position (i, j) can incorporate nearby rows (past queries) and nearby columns (nearby keys).
- Pre-softmax formulation (Equation 3): 
  - `A = Softmax(Conv2d_θ(Â))`, where `Conv2d_θ` has kernel sizes `(c_q, c_k)` along queries and keys, respectively.
- Causality handling (Equation 4 explains intended masking; Equation 5 shows the practical implementation):
  - To avoid future leakage, the convolution only uses past queries and non-future keys. Since precise masking inside the convolution kernel is cumbersome, the implementation applies a “zero mask” before convolution and the standard causal “-∞ mask” after convolution: `A = Softmax(Mask_{-∞}(Conv2d_θ(Mask_0(Â))))`.
- Post-softmax option (Equation 6):
  - Alternatively, convolve attention probabilities after softmax: `A = Mask_0(Conv2d_θ(Softmax(Mask_{-∞}(Â))))`. This changes the interaction from multiplicative (pre-softmax) to additive (post-softmax).
- Intuition with the “Alice + rabbit” example (Section 3):
  - A head can first light up positions for “Alice” and a second query for “rabbit,” and the convolution can upweight locations where both patterns co-occur within the same sentence (Figure 2 concept).

Step 3: Head mixing convolution (mixing across heads)
- For each group of `c_h` consecutive heads, perform a 1D convolution (effectively a small learned linear mixing) on their logits or weights (Equation 7; Section 3.2). Example for `c_h = 2`:
  - `A¹_new = w₁₁A¹ + w₁₂A²`, `A²_new = w₂₁A¹ + w₂₂A²`.
- Pre- and post-softmax variants are supported (Section 3.2).
- Why this matters:
  - Different heads may specialize (e.g., “Alice” vs. “rabbit”); mixing lets a head’s signal amplify or contrast another’s, enabling composite cues.

Step 4: Combine all dimensions
- A single MTA module can apply:
  - 3D convolution over queries, keys, and heads pre-softmax; optionally a second 3D convolution post-softmax (Section 3.3; Figures 1–2).
  - Or a pre-softmax key–query convolution followed by post-softmax head mixing (also supported; Section 3.3).
- Normalization and gating
  - After the softmax-side convolution(s), MTA applies group normalization with scalar gating (Figure 1 right; Section 3.3).
  - “Scalar gating” is a learned sigmoid gate per head group used to modulate the contribution of the MTA output against the residual stream, improving gradient flow and letting the model “turn on/off” heads as needed (Section 3.3, ablations in Table 5).

Design choices and why
- Local kernels `(c_q, c_k)` constrain mixing to nearby tokens, matching the intuition that relevant cues often lie within small neighborhoods (e.g., inside a sentence). This keeps parameter growth small (Appendix F, Table 8).
- Pre-softmax mixing enables multiplicative sharpening of evidence; post-softmax mixing enables additive smoothing/combination. Ablations in Table 5 show both orders work, with small differences (~0.01–0.04 perplexity).
- Identity initialization of kernels stabilizes training and preserves pre-trained behavior when retrofitting (Ablations; “Kernel initialization” in Section 4.6; Appendix I).

Mechanics at the equation level
- Explicit pre-softmax mixing for a weight at (i, j) (Equation 4):
  - It sums weighted dot-products from a local patch of past queries `q_{i−i'}` and nearby keys `k_{j−j'}`, controlled by kernel parameters `θ_{i', j'}`, then softmaxes. This is how multiple queries and keys jointly shape a single weight.
- Head mixing as rank increase (Appendix B):
  - The paper shows post-softmax head mixing can be rewritten as equivalent to a higher-rank value projection; pre-softmax mixing corresponds to a higher-rank key/query projection. This provides a theoretical rationale for increased expressiveness without fully increasing dimensionality.

Implementation used in large-scale experiments
- For the 880M model: pretrain on SlimPajama for 105B tokens. Apply key–query convolution every 4th layer; head convolution in all layers; use fixed kernels `c_q=6`, `c_k=11`, and head group size `c_h=16` (Section 4.2).
- Identity or near-identity kernel init aids convergence (Section 4.6, Kernel initialization).

## 4. Key Insights and Innovations
- Multi-token-conditioned attention weights (fundamental)
  - Novelty: Each attention weight can depend on multiple nearby queries and keys via 2D convolution, not just a single (q, k) dot product (Section 3.1; Equations 3–6).
  - Significance: Enables looking up “composite patterns” (e.g., multiple entities together) without forcing all cues into one vector; demonstrated by a toy task and long-context benchmarks.
- Cross-head evidence mixing with small groups (incremental but important)
  - Novelty: Learned convolution (linear mixing) across head groups both pre- and post-softmax (Section 3.2; Equation 7).
  - Significance: Practical way to amplify or cancel signals from specialized heads, related to Talking-heads but now combined with multi-token mixing.
- Dual-location mixing (pre and post softmax) plus gated group normalization (engineering innovation)
  - Novelty: Apply mixing in the logit domain for multiplicative sharpening and in the probability domain for additive fusion; stabilize with group normalization and scalar gating (Figure 1; Section 3.3; Table 5 ablations).
  - Significance: Yields robust improvements with minimal parameter overhead, and training remains stable.
- Learned “pattern” kernels that implement useful behaviors (insight)
  - Observation: Kernels learn interpretable structures like diagonal “sequence-matching” filters that boost attention when sequences align (Figure 4), “priming,” and “edge detection” of contiguous spans (Appendix H, Figures 9–14).
  - Significance: Offers a mechanistic explanation of how MTA finds needles and composite cues.

## 5. Experimental Analysis
Evaluation setup and baselines
- Pretraining: 880M-parameter decoder-only models trained on SlimPajama for 105B tokens using LLaMA-like configs (Appendix D; Table 7). Key–query convolution every 4th layer; head convolution on all layers; `c_q=6`, `c_k=11`, `c_h=16`.
- Baselines:
  - Transformer (standard attention).
  - DIFF Transformer (differential amplifiers across attention maps).
  - Talking-heads attention (linear head mixing pre/post softmax).
- Metrics: Validation perplexity across seven domains, and zero-shot accuracy on standard benchmarks. Long-context tasks include LAMBADA, Needle-In-A-Haystack, and BabiLong (Section 4).

Main quantitative results
- Language modeling perplexity (Table 2)
  - Pretraining (avg. over two runs): 
    - Transformer: 11.25; DIFF: 11.15; Talking-heads: 11.04; MTA: 10.91.
    - Quote: “MTA 4.54 13.09 19.63 14.00 4.12 9.76 11.18 10.91 (0.01).”
  - Long-context finetuning to 4K context:
    - Transformer: 11.02; DIFF: 10.89; Talking-heads: 10.88; MTA: 10.65.
    - MTA remains best after context extension.
- Zero-shot benchmarks (Table 3; average of two runs)
  - Average score: Transformer 43.7, DIFF 43.9, Talking-heads 44.4, MTA 44.9.
  - Notable individual tasks: BoolQ 62.1 (MTA), PIQA 71.7 (MTA), WinoGrande 57.2 (MTA).
- LAMBADA (Table 4; lower is better)
  - Standard: Transformer 17.6 vs. MTA 13.2.
  - OpenAI version: Transformer 9.5 vs. MTA 8.4.
- Needle-In-A-Haystack (Figure 3; see detailed depth curves in Appendix G, Figure 8)
  - Across 2K and 4K contexts and multiple needles (2–10), MTA achieves the highest retrieval accuracy at all depths; advantage grows for deeper insertions.
- BabiLong QA1–5 (Figure 5 left; Appendix Figure 7)
  - With increasing distraction (0K → 4K tokens), MTA consistently outperforms baselines. Gains are largest at 4K distraction where focused retrieval is hardest.
- Toy task (Table 1)
  - Task: Identify a block of letters containing L=2 query letters and output either the full block, its first token, or last token.
  - Result: MTA ~0% error across variants (e.g., “0.1 ± 0.0%”), while standard Transformer struggles (e.g., up to 58.4 ± 46.8% error).
  - Interpretation: Demonstrates the core limitation of single-token conditioning and MTA’s ability to aggregate evidence from multiple queries/keys.

Ablations and robustness checks
- How many MTA layers? (Figure 5 right)
  - Even 2 layers with key–query convolution beat baselines; 6 layers strike a good balance of performance and complexity.
- Head kernel size `c_h` (Figure 6 left)
  - Larger mixing across heads steadily improves perplexity → cross-head communication is valuable.
- Component ablations (Table 5)
  - Group normalization with scalar gating is important; removing it or using less suitable scaling hurts perplexity (e.g., “layer-norm scaling” degrades to 11.41).
  - Pre-vs-post-softmax orders differ only slightly (≤0.04 perplexity).
  - Kernel sizes `(c_q, c_k)` from 4×9 to 8×13 perform similarly, indicating robustness to exact neighborhood width.
- Kernel initialization (Section 4.6)
  - Identity init converges best; zero or constant init degrade average validation perplexity by 0.02–0.08.
- Scaling laws (Figure 6 right)
  - Across 300M–1B models, MTA yields consistent % perplexity gains over Transformer/DIFF/Talking-heads at comparable parameter scales.
- Kernel visualization (Section 4.5; Figure 4; Appendix H)
  - Learned diagonal kernels highlight sequences that match; others implement priming or edge detection. Supports a mechanistic story for improved retrieval.
- Complexity and runtime (Appendix F)
  - Parameters: negligible increase (Table 8) — e.g., 880M baseline vs. 876,583,320 with MTA (≈ +29.6K).
  - Runtime: current PyTorch implementation is not kernel-optimized; higher memory and lower FLOPS vs. baselines (Table 9), mainly because FlashAttention-like kernels are not used (Limitation A).
- Retrofitting into pretrained models (Appendix I; Table 10)
  - Identity-initialized MTA modules can be inserted into trained models and finetuned to exceed baseline perplexity without full retraining. Shown for in-house 1.4B and Llama 3.2 (1B/3B) and Llama 3.1 (8B).

Do the experiments support the claims?
- Yes, in three ways:
  - Functionality: the toy task (Table 1) directly tests the “combine multiple cues” ability and shows categorical success for MTA.
  - General LM quality: consistent perplexity and benchmark gains over strong baselines (Tables 2–3).
  - Long-context retrieval: superior LAMBADA, Needle-in-a-Haystack (across depths), and BabiLong under heavy distraction (Table 4; Figure 3; Figure 5 left; Appendix Figure 8).  
  - Component and scale ablations strengthen the causal story.

## 6. Limitations and Trade-offs
- Computational efficiency
  - Not compatible (yet) with optimized attention kernels like FlashAttention; results in higher memory and slower training (Appendix A; Table 9).
- Locality of convolution
  - Mixing is limited to fixed neighborhoods of size `(c_q, c_k)`. Extremely long-distance cross-sentence cues beyond the kernel may still be hard to integrate within a single layer, though multiple layers and head mixing mitigate this (Section 3.3; Section 4.6 on #layers).
- Causal masking approximation
  - Practical implementation masks twice (`Mask_0` before and `Mask_{-∞}` after; Equation 5), which “masks out a little more than necessary.” This is safe but may discard some borderline interactions (Section 3.1).
- Hyperparameter choices
  - Performance depends on kernel sizes and placement (how many layers use key–query convolution), though ablations show broad robustness (Figure 5 right; Table 5).
- Scope of evaluations
  - Main detailed experiments are at 880M parameters with 2K→4K contexts. The paper shows scaling trends (Figure 6 right) and retrofit results (Appendix I), but full exploration at frontier scales and very long contexts (>32K) is not presented.
- Interpretability vs. complexity
  - While some learned kernels are interpretable (Figure 4, Appendix H), the overall behavior emerges from interacting convolutions, masks, and head mixing, which can still be complex to reason about.

## 7. Implications and Future Directions
- How this changes the landscape
  - MTA reframes attention from “a weight = one similarity” to “a weight = learned combination of many similarities.” This unlocks multi-cue, multi-token retrieval directly inside attention, which is especially impactful for long-context tasks.
- Follow-up research enabled/suggested
  - Kernel-optimized implementations (e.g., integrating with FlashAttention-like kernels) to make MTA practical at large scale and inference-time efficient (Appendix A).
  - Dynamic or content-adaptive kernels that change with input, not just fixed learned filters per head/layer.
  - Curriculum or architectural strategies to decide where to place MTA layers (Figure 5 right shows diminishing returns after several layers).
  - Combining MTA with retrieval augmentation, memory compression, or adaptive softmax temperature to further improve long-context reasoning.
  - Theoretical analysis of expressivity: leveraging the “increased effective rank” perspective (Appendix B) to design principled head mixing schemes.
- Practical applications
  - Long-document QA, legal/financial contract analysis, code navigation and multi-needle search, multi-entity relation extraction, and any task where the answer hinges on co-occurrence or composition of multiple cues.
  - Retrofitting existing models: Appendix I shows identity-initialized MTA can be added post hoc and finetuned to improve perplexity on established LLMs (Table 10), which suggests a practical upgrade path for deployed systems.

> Core takeaway: Figures 1–2 and Equations (3)–(7) define a simple but powerful way to make attention weights depend on multiple queries, keys, and heads. This multi-token conditioning consistently improves perplexity (Table 2), standard benchmarks (Table 3), and long-context retrieval (Table 4, Figure 3, Figure 5), with minimal parameter overhead and clear opportunities for runtime optimization.
