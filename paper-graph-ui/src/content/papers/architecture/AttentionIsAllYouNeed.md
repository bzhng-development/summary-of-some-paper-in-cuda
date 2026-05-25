# Attention Is All You Need

**ArXiv:** [1706.03762](https://arxiv.org/abs/1706.03762)

## 🎯 Pitch

This paper introduces the Transformer, a novel neural network architecture that relies solely on attention mechanisms, discarding both recurrence and convolutions entirely. By enabling full parallelization and dramatically shortening the paths through which signals can flow, the Transformer achieves state-of-the-art results in machine translation while training much faster and scaling efficiently to long sequences—redefining what’s possible in sequence modeling and laying the groundwork for breakthroughs across NLP and beyond.

---

## 1. Executive Summary
This paper introduces the Transformer, a sequence-to-sequence neural architecture that eliminates recurrence and convolutions, using only attention mechanisms to model dependencies. It matters because it achieves state-of-the-art translation quality while training faster and scaling better through parallelism, and it generalizes beyond translation to other structured prediction tasks (e.g., constituency parsing).

## 2. Context and Motivation
- Problem addressed
  - Sequence transduction (mapping an input sequence to an output sequence) in tasks like machine translation typically relied on recurrent neural networks (RNNs) or convolutional neural networks (CNNs). Even when attention was used, it was coupled to RNN/CNN backbones.
  - These backbones impose limitations:
    - RNNs require sequential processing across time steps, limiting parallelization during training and making long-range dependencies hard to learn (Introduction; Section 2).
    - CNNs reduce sequential dependence but require multiple layers to connect distant positions, increasing path length and computation (Section 2; Table 1).
- Why it’s important
  - Practical: Training efficiency and the ability to parallelize across sequence positions are critical for large datasets and long sequences.
  - Theoretical: Shorter computational paths between input and output positions help learn long-range dependencies (Section 4).
- Prior approaches and shortcomings
  - RNN encoder-decoder with attention achieved strong results but remained sequential (Introduction; Section 2).
  - CNN-based seq2seq models (ByteNet, ConvS2S) parallelize better but still require depth to span long distances and may be more expensive per layer (Section 2; Table 1).
- Positioning
  - The Transformer removes recurrence and convolution entirely, relying on self-attention and encoder–decoder attention to compute representations. It aims to:
    - Minimize sequential operations to O(1) per layer.
    - Minimize path length between any two positions to O(1).
    - Maintain or improve translation quality (Figure 1; Sections 3–4; Table 1).

## 3. Technical Approach
The Transformer follows an encoder–decoder design (Figure 1; Section 3.1).

- Architectural overview
  - Encoder: Stack of N=6 identical layers; each has:
    - Multi-head self-attention sub-layer.
    - Position-wise feed-forward sub-layer.
    - Residual connections around each sub-layer followed by layer normalization (“Add & Norm”; Section 3.1).
  - Decoder: Also N=6 layers; each layer has:
    - Masked multi-head self-attention (prevents attending to future positions).
    - Encoder–decoder multi-head attention (attends over encoder outputs).
    - Position-wise feed-forward.
    - Residual connections and layer normalization (Section 3.1).

- What “self-attention” and “encoder–decoder attention” do
  - Self-attention: Each position in a sequence attends to all positions in the same sequence to mix information globally (Section 3.2.3).
  - Encoder–decoder attention: Decoder positions attend to all encoder positions, enabling the decoder to consult the encoded input (Section 3.2.3).

- How attention is computed (Scaled Dot-Product Attention; Section 3.2.1; Figure 2; Equation (1))
  - Each attention layer takes:
    - Queries (`Q`), Keys (`K`), Values (`V`).
    - Attention weights are computed by scaled dot products: `softmax(Q K^T / sqrt(dk))`.
    - Output is the weighted sum of `V`: `Attention(Q,K,V) = softmax(QK^T / sqrt(dk)) V` (Eq. 1).
  - Why the scaling by `sqrt(dk)` matters:
    - Without scaling, large `dk` increases dot-product magnitude, pushing softmax into regions of very small gradients. Scaling stabilizes training for larger key dimensions (Section 3.2.1; footnote).

- Multi-Head Attention (Section 3.2.2; Figure 2)
  - Instead of one attention, use `h` parallel “heads.” For each head:
    - Linearly project inputs to `dk`, `dk`, `dv`; compute attention; produce a `dv`-dimensional output.
  - Concatenate all heads and project back to `dmodel`:
    - `MultiHead(Q,K,V) = Concat(head1, …, headh) WO`, where `headi = Attention(QWQi, KWKi, VWVi)`.
  - Why multiple heads:
    - Different heads learn to focus on different types of relationships or positions (e.g., syntax vs. coreference), improving effective resolution compared to a single averaged attention (Section 3.2.2; attention visualizations in Appendix Figures 3–5).
  - Default sizes:
    - `h=8`, `dmodel=512`, `dk=dv=dmodel/h=64` (Section 3.2.2).

- Masking in the decoder (Section 3.2.3)
  - To ensure auto-regressive decoding, the decoder’s self-attention masks out all future positions (sets logits to −∞ before softmax) so position `i` cannot attend to positions > `i` (Figure 2).

- Position-wise feed-forward networks (FFN; Section 3.3; Equation (2))
  - Applied identically to each position: `FFN(x) = max(0, xW1 + b1) W2 + b2`.
  - Dimensions: input/output `dmodel=512`, inner `dff=2048`.

- Embeddings and softmax (Section 3.4)
  - Learned token embeddings for input and output are shared with each other and with the pre-softmax linear layer’s weights; embeddings are scaled by `sqrt(dmodel)`.

- Positional encoding (Section 3.5)
  - Since there is no recurrence or convolution, add positional information to embeddings.
  - Fixed sinusoidal encodings:
    - `PE(pos, 2i) = sin(pos/10000^{2i/dmodel})`
    - `PE(pos, 2i+1) = cos(pos/10000^{2i/dmodel})`
  - Motivation:
    - Enables the model to reason about relative positions; may allow extrapolation to longer sequences than seen in training (Section 3.5).

- Why this design over RNN/CNN backbones (Section 4; Table 1)
  - Per-layer complexity and path length comparison:
    - Self-attention: Complexity `O(n^2 · d)`, sequential operations `O(1)`, maximum path length `O(1)`.
    - Recurrent: `O(n · d^2)`, sequential `O(n)`, path length `O(n)`.
    - Convolutional (kernel `k`): `O(k · n · d^2)`, sequential `O(1)`, path length `O(log_k n)` with dilations.
  - Interpretation:
    - Self-attention minimizes sequential bottlenecks and shortens paths for long-range dependency learning.

- Training setup (Section 5)
  - Data and batching (Section 5.1):
    - WMT14 En–De: ~4.5M sentence pairs; byte-pair encoding (shared 37k vocabulary).
    - WMT14 En–Fr: 36M pairs; 32k word-piece vocabulary.
    - Batches: ~25k source tokens + ~25k target tokens per batch; sentences batched by similar lengths.
  - Hardware and schedule (Section 5.2):
    - 8× NVIDIA P100 GPUs; base model ~0.4 s/step for 100k steps (~12 h); big model ~1.0 s/step for 300k steps (~3.5 days).
  - Optimizer and learning-rate schedule (Section 5.3; Equation (3)):
    - Adam with β1=0.9, β2=0.98, ε=1e−9.
    - Learning rate: `lrate = dmodel^{-0.5} · min(step_num^{-0.5}, step_num · warmup_steps^{-1.5})` with 4000 warmup steps. Intuition: linearly increase then decay as inverse square root.
  - Regularization (Section 5.4):
    - Dropout on sub-layer outputs and on embedding+positional sums (base `Pdrop=0.1`; higher for some big models).
    - Label smoothing ε_ls=0.1 (improves BLEU even if perplexity worsens).

- Inference (Section 6.1)
  - Beam search with beam size 4, length penalty α=0.6.
  - Max output length = input length + 50; early stopping.
  - Checkpoint averaging: last 5 (base) or last 20 (big) checkpoints.

## 4. Key Insights and Innovations
- A. “Attention-only” sequence transduction architecture (Figure 1; Sections 3–4)
  - What’s new: Removes both recurrence and convolution; uses only attention to connect positions.
  - Why it matters:
    - Enables per-layer `O(1)` sequential operations and `O(1)` path length between any positions (Table 1), improving parallelism and ability to model long-range dependencies.

- B. Scaled dot-product attention with multi-head decomposition (Figure 2; Sections 3.2.1–3.2.2)
  - What’s new:
    - The scaling by `1/sqrt(dk)` for stable gradients at larger key dimensions.
    - Splitting attention into multiple low-dimensional heads to capture diverse relations, then recombining.
  - Why it matters:
    - Achieves both computational efficiency and representational richness. Ablations (Table 3, rows A–B) show single-head attention or too-small `dk` degrade BLEU.

- C. Sinusoidal positional encodings (Section 3.5)
  - What’s new: Fixed, continuous encodings with frequencies spanning a geometric progression.
  - Why it matters:
    - Injects order without recurrence; hypothesized to support extrapolation to longer sequences. Empirically similar to learned positional embeddings (Table 3, row E), making it a simple, parameter-free choice.

- D. Practical training recipe that reliably scales (Sections 5–6)
  - What’s new:
    - Warmup-then-decay learning-rate schedule (Eq. 3), label smoothing, and residual-dropout “Add & Norm” structure.
  - Why it matters:
    - These choices are essential to stabilize training and extract the benefits of the architecture, evidenced by substantial BLEU gains and fast wall-clock training (Table 2; Section 5.2).

Fundamental vs. incremental:
- Fundamental: (A) attention-only architecture; (B) multi-head with scaled dot-product attention.
- Incremental/pragmatic: (C) sinusoidal positional encoding choice; (D) training recipe details.

## 5. Experimental Analysis
- Evaluation methodology
  - Datasets (Section 5.1):
    - WMT14 English→German (~4.5M pairs; BPE 37k).
    - WMT14 English→French (36M pairs; word-piece 32k).
  - Metrics:
    - BLEU for translation quality; higher is better (Table 2).
    - For parsing: labeled F1 on WSJ Section 23 (Table 4).
  - Baselines (Table 2):
    - Representative strong systems: ByteNet, GNMT+RL (RNN-based), ConvS2S (CNN-based), MoE, and their ensembles.

- Main quantitative results
  - Translation quality and cost (Table 2):
    - > “Transformer (big)” achieves BLEU 28.4 on En→De and 41.8 on En→Fr.
    - Improvements:
      - En→De: +2.0+ BLEU over best prior reported results including ensembles (best ensemble listed 26.36; ConvS2S Ensemble).
      - En→Fr: New single-model state-of-the-art at 41.8 BLEU.
    - Training cost (FLOPs, lower is better at fixed quality):
      - Transformer (base): 3.3×10^18 FLOPs.
      - Transformer (big): 2.3×10^19 FLOPs.
      - Competing ensembles can be ≥10× more expensive (e.g., ConvS2S Ensemble 1.2×10^21 FLOPs).
  - Training speed (Section 5.2):
    - Base: ~12 hours on 8×P100 GPUs for 100k steps.
    - Big: ~3.5 days for 300k steps.

- Ablation studies (Table 3; Section 6.2)
  - Number of attention heads (rows A):
    - Single-head is ~0.9 BLEU worse than the best setting; too many heads also hurt.
  - Key/value dimension `dk` (rows B):
    - Smaller `dk` reduces quality—compatibility computation needs sufficient capacity.
  - Model size (rows C):
    - Larger `dmodel`/`dff` improves BLEU; e.g., `dff=4096` increases BLEU to 26.2 on dev set.
  - Dropout (rows D):
    - Dropout is helpful; removing it lowers BLEU.
  - Positional encodings (row E):
    - Learned vs. sinusoidal are nearly identical on dev BLEU; sinusoidal is chosen for simplicity/generalization reasoning.
  - Takeaway:
    - Multi-head structure and adequate attention dimensions are crucial; depth/width help; regularization matters.

- Generalization to constituency parsing (Section 6.3; Table 4)
  - Setup:
    - 4-layer Transformer (`dmodel=1024`), trained on WSJ only (~40k sentences) or semi-supervised with large additional corpora (~17M sentences).
  - Results (WSJ Section 23 F1):
    - WSJ only: 91.3 (beats BerkeleyParser at 90.4; close to strong discriminative neural models).
    - Semi-supervised: 92.7 (competitive with best discriminative systems; below RNNG generative at 93.3).
  - Inference hyperparameters:
    - Max output length increased to input length + 300; beam size 21; α=0.3.

- Qualitative analyses (Appendix; Figures 3–5)
  - Attention heads learn interpretable behaviors:
    - Long-distance dependencies (e.g., the verb “making” attending to “more difficult”).
    - Pronoun resolution (sharp attention from “its” to antecedent).
    - Different heads specializing in different structural cues.

- Do the experiments support the claims?
  - Yes, in three ways:
    - Quantitative SOTA BLEU with lower or comparable compute (Table 2).
    - Ablations isolate contributions of multi-head, dimensions, and regularization (Table 3).
    - Evidence of broader applicability (parsing results in Table 4) and interpretability (Appendix).

- Conditions and trade-offs
  - Training on substantial compute (8×P100), but less than many prior SOTA systems (Table 2).
  - Efficiency advantage is strongest when sequence length `n` is not too large relative to `dmodel`; complexity remains `O(n^2)` in self-attention (Table 1).

## 6. Limitations and Trade-offs
- Quadratic attention cost with sequence length (Table 1)
  - Per-layer complexity `O(n^2 · d)` and memory scale quadratically with `n`, which can be prohibitive for very long sequences (Section 4).
  - The paper suggests restricting attention to a neighborhood of size `r` to reduce compute to `O(r · n · d)` with a trade-off in path length `O(n/r)` (Table 1; Section 4), but does not implement this variant here.

- Order modeling relies on positional encodings (Section 3.5)
  - Without recurrence, order is injected externally. While sinusoidal encodings work well, robustness to very different positional regimes is not deeply probed; learned positional embeddings perform similarly on the tested setup (Table 3, row E).

- Decoding remains sequential
  - Although training parallelizes over sequence positions, generation is auto-regressive in the decoder and still proceeds step-by-step (Section 3.1; Section 6.1). The conclusion explicitly notes future work on “making generation less sequential.”

- Training recipe sensitivity
  - The architecture benefits from specific choices: warmup schedule (Eq. 3), label smoothing, dropout placements, and checkpoint averaging (Sections 5–6). Deviating from these may degrade performance (Table 3).

- Scope of evaluation
  - Primary evidence is on machine translation (two large benchmarks) and one parsing task. Other modalities (images, audio, video) are proposed for future work (Conclusion).

- Resolution vs. averaging in attention
  - The paper notes a potential “reduced effective resolution due to averaging attention-weighted positions,” mitigated by multi-head attention (Section 2). This trade-off is not quantitatively isolated beyond ablations.

## 7. Implications and Future Directions
- Field impact
  - By demonstrating an attention-only architecture that matches or exceeds RNN/CNN systems while training faster, the work shifts the default approach to sequence modeling toward self-attention (Sections 3–4; Table 2). The complexity/path-length analysis (Table 1) provides a clear conceptual reason for this shift.

- What this enables
  - Scalable sequence models with global receptive fields and highly parallel training.
  - Easier interpretability via attention maps (Appendix Figures 3–5).
  - Straightforward extension to tasks where global context matters and long dependencies are common (Section 6.3).

- Practical applications
  - High-throughput machine translation systems that train efficiently and deliver high-quality outputs (Section 6.1).
  - Structured prediction tasks like parsing without specialized architectures (Section 6.3).
  - Any domain where sequence elements interact across long ranges (e.g., document-level tasks).

- Research directions proposed in the paper
  - Apply the Transformer to other modalities such as images, audio, and video.
  - Explore local/restricted attention for efficient handling of very large inputs/outputs (Section 7; Section 4).
  - Reduce sequentiality in generation (Conclusion).
  - Public code release invites replication and extension (“tensorflow/tensor2tensor,” Conclusion).

- Broader open questions
  - How to handle very long sequences efficiently without losing the modeling advantages of global attention.
  - How positional information and relative-position mechanisms affect generalization and extrapolation.
  - Balancing interpretability with capacity as models and heads scale.

Overall, the paper’s core contributions—a fully attention-based architecture with multi-head scaled dot-product attention and positional encodings—are both conceptually clean and empirically validated. The results in Table 2 and the ablations in Table 3 substantiate that these design choices are not merely cosmetic; they drive state-of-the-art performance with strong computational benefits.
