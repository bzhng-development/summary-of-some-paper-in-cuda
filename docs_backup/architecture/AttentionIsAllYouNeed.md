# Attention Is All You Need

**ArXiv:** [1706.03762](https://arxiv.org/abs/1706.03762)

## 🎯 Pitch

This paper introduces the Transformer, a groundbreaking sequence transduction model that eliminates recurrence and convolution, instead using self-attention mechanisms to connect inputs and outputs. By enabling dramatically more parallelization and reducing maximum path lengths for long-range dependencies, the Transformer achieves superior translation quality at a fraction of the training cost, catalyzing rapid advances in machine translation and reshaping the foundations of deep learning architectures for sequential data.

---

## 1. Executive Summary
This paper introduces the `Transformer`, a sequence-to-sequence model that removes recurrence and convolution entirely and uses only attention to connect inputs and outputs. It matters because it achieves higher translation quality than prior state-of-the-art systems while enabling much more parallel training and substantially lower training cost (Table 2).

## 2. Context and Motivation
- Problem or gap
  - Sequence transduction (e.g., translating a sentence from one language to another) had been dominated by architectures that process tokens sequentially with recurrent neural networks (RNNs) or by deep stacks of convolutions (Section 1, Section 2).
  - These approaches have two key limitations:
    - They require many sequential steps, which slows training and makes poor use of modern parallel hardware (Section 1).
    - They make it harder to model long-range dependencies; signals often must pass through many steps/layers (Section 4; Table 1 “Maximum Path Length”).

- Why it’s important
  - Reducing sequential computation increases training speed and scalability without sacrificing quality. This matters both for faster research iteration and for deploying high-quality systems (Section 1).
  - Better handling of long-range dependencies improves translation accuracy and generalization, which has direct impact on machine translation quality and other sequence tasks (Sections 1, 4).

- Prior approaches and shortcomings
  - RNN encoder–decoders with attention: strong quality but inherently sequential and hard to parallelize (Section 1).
  - CNN-based models (e.g., ConvS2S, ByteNet): more parallelizable, but relating distant positions still requires multiple layers; path length grows with distance (Section 2; Table 1).
  - Memory networks and other attention-centric variants still typically rely on recurrence or convolution (Section 2).

- How this work positions itself
  - The `Transformer` is the first transduction architecture to compute all representations using only attention—no recurrence and no convolution—while keeping path length between any two positions constant (O(1)) and enabling high parallelism (Sections 1, 3; Table 1).

## 3. Technical Approach
The Transformer is an encoder–decoder model built entirely from attention and position-wise feed-forward layers (Figure 1; Section 3).

- Overall layout (Figure 1; Section 3.1)
  - Encoder: N=6 identical layers. Each layer has:
    - `Multi-Head Self-Attention` sub-layer.
    - `Position-wise Feed-Forward Network (FFN)` sub-layer.
    - Residual connection and `LayerNorm` around each sub-layer; all sub-layers and embeddings use dimension `d_model=512`.
  - Decoder: N=6 identical layers. Each adds a third sub-layer:
    - Masked `Multi-Head Self-Attention` (prevents seeing future tokens).
    - Encoder–decoder attention (decoder queries attend over encoder outputs).
    - FFN.
    - Residual connections and `LayerNorm` as in the encoder.

- What the core building blocks do
  - Self-attention (Section 3.2):
    - For a sequence, each position computes a weighted combination of representations at all positions (including itself). This lets the model directly relate any two positions in one step.
  - Scaled Dot-Product Attention (Figure 2 left; Section 3.2.1; Equation (1)):
    - Given query matrix `Q`, key matrix `K`, and value matrix `V`, compute attention weights with a scaled dot product: `softmax(QK^T / sqrt(d_k)) V`.
    - Why the scaling by `sqrt(d_k)`: Without it, dot products grow with dimension, pushing the softmax into small-gradient regions; scaling keeps gradients well-behaved for larger `d_k` (Section 3.2.1).
  - Multi-Head Attention (Figure 2 right; Section 3.2.2):
    - Project `Q`, `K`, and `V` into `h` lower-dimensional subspaces, compute attention in each head in parallel, then concatenate and project back.
    - Default in base model: `h=8` heads; each head uses `d_k = d_v = d_model / h = 64`.
    - Why: Each head can focus on different relations (e.g., syntax, coreference), and splitting reduces the averaging effect of a single head, improving expressiveness (Section 3.2.2).
  - Masked attention in the decoder (Section 3.2.3):
    - During training and inference, the decoder’s self-attention masks out positions to the right, ensuring position `i` cannot attend to future tokens (preserves auto-regressive generation).
  - Position-wise Feed-Forward Network (Section 3.3; Equation (2)):
    - For each position independently: `FFN(x) = max(0, xW1 + b1) W2 + b2`, with inner dimension `d_ff = 2048`. This adds nonlinearity and depth apart from attention mixing.
  - Positional encoding (Section 3.5):
    - Since there’s no recurrence/convolution to encode order, add a positional signal to input embeddings.
    - Uses sinusoidal encodings with wavelengths forming a geometric progression:
      - Even dims: `PE(pos, 2i) = sin(pos / 10000^(2i/d_model))`
      - Odd dims: `PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))`
    - Rationale: Enables the model to infer relative positions and extrapolate to sequence lengths beyond training (Section 3.5). Learned positional embeddings performed similarly (Table 3 row E).
  - Embedding and softmax weight sharing (Section 3.4):
    - Share parameters between input embeddings (encoder/decoder) and the final pre-softmax projection; multiply embeddings by `sqrt(d_model)` for scale.
  - Residual connections and LayerNorm (Section 3.1):
    - Each sub-layer output is `LayerNorm(x + Sublayer(x))`. Residuals ease optimization; LayerNorm stabilizes training.

- Why this design over alternatives (Section 4; Table 1)
  - Parallelism: Self-attention computes all positions simultaneously (minimum sequential ops O(1)), while RNNs require O(n) sequential steps.
  - Path length: With self-attention, any two positions connect in O(1) layers; convolution requires multiple layers to connect distant positions (O(log_k n) for dilated conv; Table 1).
  - Computation: Per-layer self-attention cost is `O(n^2 d)`, which is often faster than RNNs when sequence length `n` is smaller than representation dimension `d` (common with subword tokens; Section 4).
  - Caveat and mitigation: Attention’s “averaging” can reduce effective resolution; `Multi-Head` design counters this by specializing heads (Section 2; Figure 2).

- Training setup (Section 5)
  - Data (Section 5.1):
    - WMT 2014 English–German (~4.5M sentence pairs), subword tokenization via byte-pair encoding (shared vocab ~37k).
    - WMT 2014 English–French (36M pairs), 32k wordpiece vocabulary.
    - Batches group sentences by length; each batch ≈25k source and 25k target tokens.
  - Hardware and schedule (Section 5.2):
    - 8× NVIDIA P100 GPUs. Base: 0.4 s/step, 100k steps (~12 hours). Big: 1.0 s/step, 300k steps (~3.5 days).
  - Optimization (Section 5.3; Equation (3)):
    - Adam with β1=0.9, β2=0.98, ε=1e-9. Learning rate `= d_model^(-0.5) * min(step^-0.5, step * warmup^-1.5)`, with `warmup_steps=4000`. Intuition: Linear warmup to a peak LR, then decay as 1/sqrt(step).
  - Regularization (Section 5.4):
    - Dropout on sub-layer outputs and on input+positional sums (base: `P_drop=0.1`).
    - Label smoothing `ε_ls=0.1` (encourages calibrated probabilities and improves BLEU despite worse perplexity).

## 4. Key Insights and Innovations
- Attention-only transduction with O(1) dependency path length
  - What’s new: Completely removes recurrence and convolution, using attention for all token–token interactions (Sections 1, 3).
  - Why it matters: Enables massive parallelism and eliminates long path-length bottlenecks for long-range dependencies (Section 4; Table 1).

- Scaled dot-product attention with multi-head decomposition
  - What’s new: Introduces the `1/sqrt(d_k)` scaling in the attention logits to stabilize gradients for large key/query dimensions (Section 3.2.1) and uses multiple attention heads in parallel (Section 3.2.2).
  - Why it matters: Stable training at scale and the ability to attend to different relational patterns simultaneously (Figures 2–5 show diverse, interpretable head behaviors).

- Sinusoidal positional encodings
  - What’s new: Adds fixed, continuous positional signals that support relative-position reasoning and possible length extrapolation without learned parameters (Section 3.5).
  - Why it matters: Removes the need for recurrence/convolution to encode order while keeping performance on par with learned positional embeddings (Table 3 row E).

- Simple, effective training recipes for stability and quality
  - What’s new: A combination of residual connections + LayerNorm (Section 3.1), label smoothing (Section 5.4), shared embeddings (Section 3.4), and a warmup–decay learning rate schedule (Equation (3)).
  - Why it matters: These choices collectively produce strong single-model performance with comparatively modest training time (Table 2).

Fundamental vs. incremental:
- Fundamental: Attention-only architecture; multi-head scaled attention; positional encodings.
- Incremental but important: Weight sharing, LR schedule, and regularization blend that make the model practical and performant.

## 5. Experimental Analysis
- Evaluation methodology
  - Tasks and datasets (Section 5.1):
    - WMT14 English→German (En–De) and English→French (En–Fr).
  - Metrics:
    - `BLEU`: an n-gram overlap score for machine translation quality; higher is better.
  - Baselines and comparisons (Table 2):
    - Strong prior systems, including GNMT with reinforcement learning, convolutional sequence-to-sequence (ConvS2S), ByteNet, and ensembles.
  - Inference setup (Section 6.1):
    - Beam search with beam size 4 and length penalty `α=0.6`.
    - Maximum output length = input length + 50, with early termination.
    - Checkpoint averaging (base: last 5, big: last 20) for final single model.

- Main results (Table 2)
  - English→German:
    - `Transformer (big)`: 28.4 BLEU, outperforming all prior single and ensemble models by >2 BLEU.
    - `Transformer (base)`: 27.3 BLEU, already exceeding prior state-of-the-art ensembles.
  - English→French:
    - `Transformer (big)`: 41.8 BLEU, single-model state-of-the-art; trains in 3.5 days on 8 GPUs.
  - Training cost (FLOPs; Table 2 notes and footnote 5):
    - The paper estimates FLOPs from time × number of GPUs × per-GPU sustained FLOPS; this provides a coarse but comparable cost indicator.
  - Quote of headline claim:
    > "Our model achieves 28.4 BLEU on the WMT 2014 English-to-German... On the WMT 2014 English-to-French... 41.8" (Abstract; corroborated by Table 2).

- Ablations and design justifications (Table 3; Section 6.2)
  - Number of heads (`h`) and per-head dims (`d_k`, `d_v`) (rows A):
    - Single-head attention is ≈0.9 BLEU worse than best multi-head settings.
    - Too many heads also hurts (e.g., `h=32` drops to 25.4 BLEU on dev).
    - Interpretation: Multiple specialized heads are helpful, but each head must retain enough dimensionality to be expressive.
  - Key size (`d_k`) sensitivity (rows B):
    - Reducing `d_k` decreases BLEU, indicating compatibility computation is non-trivial and benefits from larger `d_k`.
  - Model size and FFN width (`d_ff`) (rows C):
    - Larger models and wider FFN layers yield consistent gains (e.g., `d_ff=4096` improves dev BLEU to 26.2).
  - Dropout (rows D):
    - Removing dropout degrades BLEU; a moderate rate helps avoid overfitting.
  - Positional encoding (row E):
    - Learned positional embeddings perform nearly identically to sinusoidal, supporting the choice of fixed sinusoids for simplicity and extrapolation.
  - Overall, these ablations support that the headline quality comes from the full recipe (architecture + training choices), not a single trick.

- Additional task: English constituency parsing (Section 6.3; Table 4)
  - Setup: 4-layer Transformer (`d_model=1024`) on WSJ-only (≈40k sentences) and in a semi-supervised setting (≈17M sentences).
  - Results:
    - WSJ-only: F1 = 91.3, competitive with strong discriminative parsers and close to RNNG (generative).
    - Semi-supervised: F1 = 92.7, better than prior semi-supervised results reported in Table 4.
  - Takeaway: The architecture generalizes beyond translation without task-specific engineering.

- Do the experiments support the claims?
  - Yes, for machine translation:
    - Consistent gains over strong baselines and ensembles (Table 2) with significantly lower estimated training FLOPs.
    - Extensive ablations (Table 3) show the importance of core components (multi-head, adequate `d_k`, FFN width, dropout), supporting that the quality stems from the proposed design.
  - Generalization beyond MT is suggestive:
    - Parsing results are strong with minimal tuning (Table 4), indicating the architecture is not MT-specific.

## 6. Limitations and Trade-offs
- Quadratic attention cost in sequence length (Table 1; Section 4)
  - Per-layer self-attention complexity is `O(n^2 d)`. For very long sequences, memory and compute can be heavy.
  - The paper notes possible mitigation via restricted/local attention of neighborhood size `r` (would raise maximum path length to `O(n/r)`), but does not implement it here (Section 4).

- “Averaging” in attention can blur signals (Section 2)
  - Attention computes weighted sums of values; this can reduce effective resolution. The multi-head design mitigates but does not eliminate this effect.

- Training cost is still non-trivial
  - Although far lower than prior SOTA (Table 2), the big model still trains for 3.5 days on 8×P100 GPUs (Section 5.2).

- Evaluation scope
  - Main evidence is on two MT benchmarks and one parsing task. Other modalities (speech, vision) or extremely long-context tasks are not evaluated (Conclusion and future work).

- Auto-regressive decoding latency
  - Generation remains sequential at inference time (masked decoder) even though training is highly parallel (Section 3.1). The paper flags “making generation less sequential” as future work (Conclusion).

- Tokenization assumptions
  - Uses subword tokenization (BPE/wordpiece; Section 5.1). Performance might depend on the chosen segmentation and vocabulary size, which are not ablated here.

## 7. Implications and Future Directions
- Field impact
  - Establishes attention-only architectures as a powerful and scalable alternative to RNNs/CNNs for sequence modeling (Sections 1, 6), with immediate influence on subsequent models across NLP.
  - The constant path length and high parallelism reframe how we design sequence models for speed and quality (Table 1).

- What it enables next
  - Efficient handling of large datasets with parallel training.
  - Richer modeling of long-range dependencies through multi-head mechanisms and deeper stacks.

- Practical applications
  - Machine translation systems with faster training and improved accuracy (Table 2).
  - Broadly applicable to tasks needing pairwise token interactions: summarization, question answering, parsing (Section 6.3), and beyond.

- Explicit future directions mentioned (Conclusion)
  - Local/restricted attention to handle very large inputs efficiently (Section 4).
  - Extend to other modalities (images, audio, video).
  - Reduce sequentiality in generation (e.g., non-autoregressive decoding).
  - The released codebase (`tensor2tensor`) facilitates replication and exploration.

> Key numbers to remember (Table 2; Section 6.1):
> - En–De BLEU: 28.4 (big), single-model SOTA at the time.
> - En–Fr BLEU: 41.8 (big), single-model SOTA.
> - Training: base ≈12 hours, big ≈3.5 days on 8×P100 GPUs.
> - Complexity advantages: O(1) sequential operations per layer and O(1) maximum path length between any two positions (Table 1).
