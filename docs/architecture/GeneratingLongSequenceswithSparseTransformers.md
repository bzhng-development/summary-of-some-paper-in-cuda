# Generating Long Sequences with Sparse Transformers

**ArXiv:** [1904.10509](https://arxiv.org/abs/1904.10509)

## 🎯 Pitch

This paper introduces Sparse Transformers, a novel approach that factorizes the attention mechanism to reduce the time and memory complexity of self-attention from quadratic to nearly linear in sequence length. By enabling efficient and scalable modeling of long-range dependencies, this innovation unlocks the potential to generate and model images, text, and audio with unprecedented sequence lengths, achieving state-of-the-art results in multiple domains and paving the way for practical large-context generative models.

---

## 1. Executive Summary
This paper introduces Sparse Transformers, a way to restructure self‑attention so that time and memory grow roughly as O(n√n) instead of O(n²) with sequence length n. By factorizing attention into inexpensive patterns that still preserve global connectivity, plus several training and systems improvements, the model handles sequences tens of thousands of tokens long and achieves state‑of‑the‑art density modeling on images and text, with strong audio generation.

## 2. Context and Motivation
- Problem addressed
  - Standard self‑attention compares every position to every other, so both compute and memory scale quadratically with sequence length. This makes long‑context modeling (e.g., long text, raw audio, large images) prohibitively expensive.
- Why it matters
  - Many real‑world signals have long‑range dependencies: language consistency across pages, musical motifs across minutes, and global structures in images. Handling long contexts without prohibitive cost is central for high‑quality generation and compression.
- Prior approaches and gaps
  - CNNs (e.g., PixelCNN) extend receptive fields by stacking layers or using dilations, but still require great depth to capture long range (Intro; Related Work).
  - WaveNet used dilated convolutions to reach distant context, but layer count grows with context length.
  - Transformer architectures provide direct global access but at O(n²) cost, limiting context length (Intro).
  - Image Transformer used blocks of local attention and other works introduced memory/state reuse (Related Work), but these are domain‑specific or add complexity.
- Positioning
  - This work aims for a simple, domain‑agnostic modification to Transformers: fixed, hand‑designed sparse attention patterns that reduce cost yet preserve the ability to propagate information globally in a constant number of layers/steps (Section 4, Fig. 3).

## 3. Technical Approach
The method starts from a decoder‑only Transformer used as an autoregressive model (Section 3). The joint probability of a sequence x = {x₁,…,xₙ} is factorized into conditionals (Eq. 1), and each token distribution is predicted from past context.

Step-by-step design

1) Attention formalism (Section 4.2, Eqs. 2–4)
- For each position i, standard self‑attention computes a weighted average over keys/values from all earlier positions j ≤ i.
- Notation:
  - S = {S₁,…,Sₙ} indicates, for each output i, which input indices j are attended.
  - For position i, a(xᵢ, Sᵢ) = softmax((W_q xᵢ)(K_{Sᵢ})ᵀ / √d) V_{Sᵢ}, where K_{Sᵢ} and V_{Sᵢ} collect the keys/values for the selected indices (Eqs. 3–4). d is the key/query inner dimension.

2) Factorized self‑attention to reduce complexity (Section 4.2)
- Core idea: replace full attention with p separate attention “heads” that each look at only a sparse subset A^{(m)} of the past, with |A^{(m)}_i| proportional to √n (for p=2). Compute these attention steps sequentially so information can flow globally through short paths.
- Validity constraint: even though each head is sparse, any earlier position j can influence any later position i through a short path of at most p+1 attentional hops (Section 4.2). This preserves Transformer‑like global connectivity while reducing total operations from O(n²) to O(n√n) for p=2.

3) Two concrete sparse patterns (Section 4.3; Fig. 3)
- Strided pattern (good for data with spatial/periodic structure like images):
  - Choose stride l ≈ √n. Head 1 attends to the last l local positions: A^{(1)}_i = {max(0, i − l), …, i}.
  - Head 2 attends to every l‑th prior position (a stride): A^{(2)}_i = {j : (i − j) mod l = 0}.
  - Intuition: one head handles local detail; the other passes summary information across the sequence at stride intervals.
- Fixed pattern (good for non‑periodic data like text):
  - Partition the sequence into blocks of size l. Head 1 attends within the current block (A^{(1)}_i = {j : floor(j/l) = floor(i/l)}).
  - Head 2 attends to a small set of “summary” positions (a trailing slice of c positions) from every previous block: A^{(2)}_i = {j : j mod l ∈ {t,…,l}}, where t = l − c and c is a hyperparameter (Section 4.3). Setting c in {8,16,32} with l in {128,256} worked well. Larger c increases compute by factor c but adds capacity.

Why these patterns? A qualitative analysis of dense‑attention models (Fig. 2) shows that:
- Many layers learn local, convolution‑like patterns (Fig. 2a).
- Some layers naturally factorize attention into rows and columns (Fig. 2b).
- Others show sparse or global, data‑dependent patterns (Figs. 2c–d).
This motivates fixed sparse patterns that preserve global information flow while capitalizing on locality and structure.

4) Integrating sparse patterns into the Transformer stack (Section 5)
- Attention interfaces (Eqs. 5–8; Fig. 4):
  - Replace dense attention with `attend(X, A)` using the specified sparse connectivity.
  - Three integration choices:
    1) Interleaved heads across residual blocks: use a single pattern per block, alternating patterns across layers (Eq. 6).
    2) Merged head: construct one attention that attends to the union of multiple sparse sets (Eq. 7).
    3) Multi‑head: compute several attentions in parallel; A can be the distinct sparse patterns or their unions (Eq. 8).
- Depth‑friendly residual block and initialization (Section 5.2; Eqs. 9–14; Fig. 4):
  - Use a pre‑activation residual block: LayerNorm → attention → dropout; residual add; LayerNorm → feedforward → dropout; residual add. The residual block returns the sum of attention and feedforward branches (Eqs. 12–14).
  - Initialization scales W₂ (feedforward output) and W_p (post‑attention projection) by √(1/(2N)) for N layers, to stabilize very deep stacks (Section 5.2).
- Embeddings that expose structure (Section 5.3; Eq. 15):
  - Treat all modalities (images, text, audio) as byte sequences.
  - Add learned position embeddings tailored to the data:
    - Images: `d_data = 3` positional channels for row, column, channel index.
    - Text/audio: `d_attn = 2` embeddings representing row/column in a virtual 2‑D layout aligned with the stride of the sparse attention.
  - Final token representation = token embedding + sum of chosen positional embeddings (Eq. 15).
- Memory and speed optimizations
  - Recompute attention and feedforward activations in the backward pass (gradient checkpointing) to cut memory (Section 5.4; Fig. 4). Dropout is applied only at residual outputs to simplify recomputation.
  - Custom block‑sparse CUDA kernels compute only the needed attention submatrices, fuse softmax, and skip the upper triangle entirely for autoregressive masking (Section 5.5). This avoids mask biases and halves operations.
  - Mixed‑precision training with dynamic loss scaling; cast queries/keys to float32 at sampling time to avoid overflow (Section 5.6).
- Training setup (Section 6)
  - Adam, 5k‑step linear warmup, gradient clipping at 1.0, weight decay 0.01, cosine learning‑rate decay, and typically 8 V100 GPUs.

How the O(n√n) arises
- With stride l ≈ √n, each position attends to O(l) local items and O(n/l) strided or summary items, giving O(l + n/l). Minimizing over l yields l ≈ √n and cost O(√n + √n) = O(√n) per token, hence O(n√n) overall. The path‑validity constraint ensures any dependency can propagate through at most p+1 sparse layers (Section 4.2).

## 4. Key Insights and Innovations
- Factorized attention that preserves global connectivity
  - Novelty: explicit, hand‑crafted sparse patterns (strided or fixed) that guarantee any position can influence any later one via short paths (Section 4.2–4.3; Fig. 3). Different from prior local‑only windows, which lose global access unless many layers are stacked.
  - Significance: reduces complexity to O(n√n) without sacrificing long‑range reasoning. On Enwik8 and CIFAR‑10, these sparse patterns not only run faster but also converge to better losses than dense attention (Table 2).
- Depth‑scalable architecture with specialized initialization
  - Novelty: pre‑activation residual structure plus √(1/(2N)) scaling of key projection matrices (Eqs. 9–14; Section 5.2), enabling hundreds of layers.
  - Significance: allows very deep models on long contexts, a known difficulty for Transformers.
- Systems innovations for long sequences
  - Recompute attention/FFN during backprop to fit long sequences (Section 5.4; Fig. 4).
  - Purpose‑built block‑sparse kernels and fused softmax; compute only lower‑triangular blocks (Section 5.5).
  - Significance: practical training on sequence lengths up to 16,384 with dense attention and much longer with sparse attention; million‑token sequences become feasible (Section 7.4; Table 4).
- Unified byte‑level modeling across modalities
  - Novelty: same self‑attention architecture models images, text, and audio using byte tokens and modality‑appropriate position embeddings (Section 5.3).
  - Significance: sets new state‑of‑the‑art for Enwik8 and ImageNet‑64 image density, and produces globally coherent audio and images (Table 1; Fig. 5).

## 5. Experimental Analysis
Evaluation setup
- Datasets and metrics (Section 7; Table 1)
  - CIFAR‑10 (image density; bits per dim = bits per byte).
  - Enwik8 (first 10⁸ bytes of Wikipedia; bits per byte).
  - ImageNet 64×64 (image density; bits per dim).
  - Classical music audio (µ‑law at 12 kHz; bits per byte).
- Baselines (Table 1)
  - Images: PixelCNN, PixelCNN++, Image Transformer, PixelSNAIL, Glow, SPN.
  - Text: Deeper Self‑Attention, Transformer‑XL.
- Model configurations per task
  - CIFAR‑10: strided Sparse Transformer, 128 layers, d=256, 2 attention heads, half‑size feedforward and query‑key projections; sequence length 3,072 (Section 7.1).
  - Enwik8: fixed Sparse Transformer, 30 layers, d=512, 8 heads, stride 128 with c=32; context 12,288 (Section 7.2).
  - ImageNet 64×64: strided, 48 layers, 16 heads, d=512; stride 128; trained 7 days on 64 V100s (Section 7.3).
  - Audio: strided; sequence lengths 65,536 to 1,048,576; model size adjusted to fit 16 GB GPUs (Section 7.4; Table 4).

Main quantitative results
- Table 1 (state of the art summary):
  - CIFAR‑10: 2.80 bits/dim (59M params), improving over PixelSNAIL’s 2.85.
  - Enwik8: 0.99 bits/byte (95M params), matching a much larger Transformer‑XL (277M) and beating a similar‑size Transformer‑XL (88M) at 1.03.
  - ImageNet 64×64: 3.44 bits/dim (152M), best among listed baselines (e.g., SPN at 3.52).
  - Audio (classical, 5s at 12 kHz): 1.97 bits/byte (152M) for 65,536‑length sequences.
- Speed vs loss trade‑off (Table 2)
  - Enwik8 (12,288 context): fixed sparse beats dense on both quality and speed.
    - “Dense Attention 1.00 bpb; 1.31 time/iter”
    - “Sparse (Fixed) 0.99 bpb; 0.55 time/iter”
    - “Sparse (Strided) 1.13 bpb; 0.35 time/iter” (strided underperforms on text)
  - CIFAR‑10 (3,072 context): strided sparse is both faster and better.
    - “Dense 2.82; 0.54”
    - “Sparse (Fixed) 2.85; 0.47”
    - “Sparse (Strided) 2.80; 0.38”
- Long‑context utilization (Table 3)
  - Quality improves monotonically as more context is provided at test time, up to 12,160 tokens of 12,288 trained: e.g., 0.9952 bpb at 6,144 context vs 0.9908 bpb at 12,160. This indicates the model truly leverages long‑range dependencies.
- Million‑length sequences (Table 4)
  - Demonstration that sparse attention scales to 1,048,576 tokens with a small model (3M params), though quality degrades (2.99 bpb). At 65,536 tokens with 152M params, quality is strong (1.97 bpb).

Qualitative results
- Fig. 2: Dense models naturally learn sparse/local patterns and even row/column factorizations, motivating fixed sparse structures.
- Fig. 5: Unconditional ImageNet‑64 samples show global coherence without explicit multi‑scale design.

Ablations and design sensitivity
- Strided vs fixed patterns:
  - Strided excels when data align with strides (images, some audio); fails on non‑periodic text (Table 2; Section 4.3 and 7.2).
  - Fixed patterns with c summary positions restore global information routing for text at a moderate compute increase (Section 4.3).
- Embedding choices:
  - Data‑aligned positional embeddings (image row/col/channel) vs attention‑aligned embeddings (text/audio) are crucial for performance (Section 5.3).
- Depth and initialization:
  - The pre‑activation residual block plus scaled projections enable training 100+ layers (Sections 5.2 and 7.1).

Do the experiments support the claims?
- The paper claims reduced complexity with maintained global connectivity and superior or comparable performance across modalities. Quantitatively, sparse patterns outperform dense attention on both speed and loss for CIFAR‑10 and Enwik8 (Table 2) and achieve SOTA numbers on ImageNet‑64 and Enwik8 (Table 1). Monotonic gains with longer contexts (Table 3) substantiate effective long‑range modeling. The audio study (Table 4) shows feasibility up to million‑token contexts but also highlights capacity‑length trade‑offs.

## 6. Limitations and Trade-offs
- Fixed, hand‑designed sparsity
  - The sparse patterns are predetermined, not learned or data‑adaptive. This can misalign with data that lack clear periodic or block structure (e.g., strided pattern on text performs poorly; Table 2).
- Compute is O(n√n), not linear
  - While better than O(n²), the complexity is still super‑linear; truly massive contexts still demand either small models (Table 4) or many resources.
- Capacity vs length trade‑off
  - To fit very long sequences into fixed memory, model size must shrink (Table 4), hurting quality. This indicates that sparse attention alone does not remove the need for substantial capacity.
- Additional engineering complexity
  - Gains rely on custom block‑sparse kernels, gradient recomputation, and careful initialization/training schedules (Sections 5.4–5.6). Portability and implementation effort can be nontrivial.
- Potential expressivity constraints
  - The “validity” constraint ensures short attention paths, but the exact function class differs from dense attention. Some highly data‑dependent global patterns observed in Fig. 2c–d may be harder to match with fixed patterns.

## 7. Implications and Future Directions
- Field impact
  - Demonstrates that thoughtfully factorized attention can deliver both scalability and strong accuracy across domains without elaborate multi‑scale or domain‑specific architectures. This reframes long‑sequence modeling as a sparsity‑design problem rather than an unavoidable O(n²) cost.
- Follow‑up research directions
  - Learnable or dynamic sparsity: replacing fixed patterns with patterns learned per layer or per input while preserving global connectivity and O(n√n) (or better) complexity.
  - Hybrid schemes: combining sparse attention with memory tokens, recurrence, or compressed representations to approach O(n) while keeping expressivity.
  - Better summaries: improving the “c summary positions per block” mechanism, possibly with adaptive c or content‑based summaries.
  - Multi‑modal scaling: extending to higher‑resolution images, longer audio sequences, and long‑form document modeling with richer positional schemes.
- Practical applications
  - Long‑form text generation and compression; whole‑track music modeling; high‑resolution image generation; any setting where long contexts are essential but quadratic cost is prohibitive.

References to paper content used above
- Equations: Eq. 1 (autoregressive factorization), Eqs. 2–4 (attention), Eqs. 5–8 (integration of sparse patterns), Eqs. 9–14 (residual block/initialization), Eq. 15 (embedding).
- Figures: Fig. 2 (learned attention patterns), Fig. 3 (strided vs fixed sparse designs), Fig. 4 (residual block and recomputation), Fig. 5 (samples).
- Tables: Table 1 (SOTA summary), Table 2 (speed/quality vs dense), Table 3 (benefit of longer context), Table 4 (sequence length vs model size).
