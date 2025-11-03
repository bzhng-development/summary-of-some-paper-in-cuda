# AN IMAGE IS WORTH 16X16 WORDS: TRANSFORMERS FOR IMAGE RECOGNITION AT SCALE

**ArXiv:** [2010.11929](https://arxiv.org/abs/2010.11929)

## 🎯 Pitch

This paper introduces Vision Transformer (ViT), a landmark approach that applies standard Transformer architectures directly to sequences of image patches for image classification, abandoning the reliance on convolutions and vision-specific design. Demonstrating that, with sufficient pre-training data, ViT matches or surpasses the performance of state-of-the-art convolutional neural networks while requiring less computational effort, this work not only reshapes the landscape of large-scale visual recognition but also facilitates the direct transfer of scalable and efficient NLP transformer techniques to computer vision.

---

## 1. Executive Summary
This paper introduces Vision Transformer (`ViT`), a pure Transformer model that operates on sequences of image patches instead of pixels, and shows that—with sufficient pre-training data—such models match or outperform state-of-the-art convolutional neural networks (CNNs) for image classification. The key significance is that by largely removing vision-specific inductive biases (like convolutions), `ViT` scales efficiently and achieves top results when pre-trained on large datasets (e.g., ImageNet-21k, JFT-300M) and then fine-tuned, while using less compute than competitive CNN baselines.

## 2. Context and Motivation
- Problem/Gap
  - Transformers dominate NLP but are underused in vision due to the quadratic cost of naïve self-attention over pixels and a belief that strong vision-specific inductive biases (e.g., convolution, locality, translation equivariance) are necessary.
  - Prior vision models using attention either hybridize with CNNs or employ specialized attention patterns that complicate scaling (Section 2).
- Importance
  - Practical: Unlocks the scalability and tooling of standard Transformers for vision; potentially better compute-efficiency and easier model scaling (Section 1).
  - Theoretical: Tests how far one can go with minimal inductive bias by “learning” spatial structure directly from data.
- Prior approaches and shortcomings
  - Local and sparse attention variants (e.g., local attention, axial/sparse attention) reduce complexity but require complex engineering, limiting hardware efficiency (Section 2).
  - Hybrid models add attention atop CNN features but retain CNN structure (Section 2).
  - Pixel-level generative Transformers (e.g., iGPT) showed promising representations but lagged behind supervised CNNs on classification (72% top-1 on ImageNet; Section 2).
- Positioning
  - The paper uses a standard Transformer encoder “almost out of the box,” treating an image as a sequence of fixed-size patches, and demonstrates that large-scale pre-training compensates for weaker inductive bias (Section 1; Figure 1).

## 3. Technical Approach
Step-by-step overview of ViT (Section 3; Figure 1; Equations 1–4):
1. Convert image to tokens
   - Split an image of size `H×W×C` into non-overlapping `P×P` patches (typically `16×16` or `14×14`; Table 1 lists model variants like `ViT-B/16`, `ViT-L/16`, `ViT-H/14`).
   - Flatten each patch to a vector and linearly project to `D` dimensions to create a patch embedding (Eq. 1: `E ∈ R^{(P^2·C)×D}`).
   - Why patches? Reduces sequence length from pixels to patches (sequence length `N = HW/P^2`), making global self-attention tractable.
2. Add positional information and a class token
   - Prepend a learned `class` token whose final state is used as the image representation for classification (Eq. 4).
   - Add learnable 1D position embeddings to retain spatial order (Section 3.1). Ablations show 1D vs 2D positional embeddings make little difference (Appendix D.4; Table 8).
3. Standard Transformer encoder
   - Apply `L` layers of multi-head self-attention (`MSA`) and 2-layer MLP blocks, each preceded by LayerNorm and followed by residual connections (Eqs. 2–3; Figure 1).
   - Nonlinearity is `GELU` in MLP. Other than the initial patching, the encoder is the same as in NLP Transformers.
4. Classification head
   - For pre-training: MLP with one hidden layer on top of the `class` token.
   - For fine-tuning: replace with a single linear layer (Section 3.1).
5. Fine-tuning at higher resolution
   - Often fine-tune on larger images (e.g., train at 224px, fine-tune at 384–518px). Keep `P` constant; sequence length increases. Interpolate the learned position embeddings in 2D to the larger patch grid (Section 3.2).
6. Hybrid alternative (optional)
   - Instead of raw patches, take a CNN feature map and treat each spatial location as a “patch” token; feed to the same Transformer (Section 3.1).
7. Training setup (Appendix B; Table 3)
   - Pre-train with Adam (β1=0.9, β2=0.999), large batch size (4096), high weight decay (0.1), linear LR warmup and decay; training resolution 224.
   - Fine-tune with SGD (momentum 0.9), batch size 512, cosine LR decay; typical fine-tune resolution 384. For the best ImageNet results, use higher resolutions and Polyak averaging (Section 4.1; Table 4; Table 2 notes).
8. Self-supervised variant (Section 4.6; Appendix B.1.2)
   - Masked patch prediction (BERT-style): mask 50% of patch embeddings and predict 3-bit mean color of corrupted patches. Provides gains over training from scratch but trails supervised pre-training.

Design rationale:
- Minimal vision-specific biases: Only the patching step and position embeddings inject 2D structure. The model “learns” spatial relations via attention when sufficient data is available (Section 3.1; Section 1).
- Reuse NLP Transformer infrastructure: Simpler to scale and optimize than specialized attention for images (Section 3).

## 4. Key Insights and Innovations
- Treat patches as tokens in a standard Transformer
  - Novelty: Use an off-the-shelf Transformer encoder on patch sequences with minimal changes (Figure 1; Eqs. 1–4).
  - Significance: Enables direct transfer of NLP scaling practices to vision. Delivers SOTA classification performance when pre-trained at scale (Table 2).
- Data scale trumps inductive bias for classification
  - Finding: With enough pre-training data (ImageNet-21k or JFT-300M), `ViT` matches or beats strong CNNs; on small data (ImageNet-1k) it underperforms (Figure 3 and 4; Table 5).
  - Significance: Challenges the necessity of CNN inductive biases for high performance when large pre-training corpora are available.
- Superior compute–performance trade-off at scale
  - Finding: For transfer performance averaged across datasets, `ViT` attains the same accuracy as ResNets with roughly 2–4× less pre-training compute (Figure 5; Table 6).
  - Significance: Important for training cost and carbon footprint; suggests Transformers are a better scaling substrate for vision classification.
- Simple fine-tuning to higher resolutions via positional interpolation
  - Mechanism: 2D interpolation of position embeddings lets a model trained at 224px be fine-tuned at 384–518px without architectural changes (Section 3.2).
  - Significance: Unlocks additional accuracy with minimal complexity; used in the best ImageNet results (Table 2 notes).
- Interpretable attention behaviors
  - Observation: Low layers mix local and global context; attention spans grow with depth; attention focuses on semantically relevant regions (Figures 6–7; Appendix D.7–D.8).
  - Significance: Offers insight into how a minimal-bias model learns spatial structure.

## 5. Experimental Analysis
Evaluation design (Section 4.1; Table 3–4):
- Pre-training datasets
  - ImageNet (1.3M images, 1k classes), ImageNet-21k (14M images, 21k classes), JFT-300M (303M images, 18k classes).
  - Duplicates removed w.r.t. downstream test sets (Section 4.1).
- Downstream tasks
  - ImageNet (original and ReaL labels), CIFAR-10/100, Oxford-IIIT Pets, Oxford Flowers-102, and VTAB-1k (19 low-data tasks spanning Natural, Specialized, Structured; Section 4.1; Figure 2; Appendix D.10).
- Metrics and protocols
  - Fine-tuning accuracy is primary. Few-shot linear evaluation used for fast comparisons and scaling analyses (Section 4.1).

Main results and comparisons:
- State-of-the-art transfer with less pre-training compute (Table 2)
  - Quote:
    > ImageNet top-1: `ViT-H/14 (JFT)` 88.55 ± 0.04; `ViT-L/16 (JFT)` 87.76 ± 0.03; `BiT-L (ResNet152x4, JFT)` 87.54 ± 0.02; `Noisy Student (EffNet-L2)` 88.4/88.5
    > VTAB mean (19 tasks): `ViT-H/14 (JFT)` 77.63 ± 0.23 vs `BiT-L` 76.29 ± 1.70
    > TPUv3-core-days: `ViT-H/14` 2.5k; `ViT-L/16` 0.68k; `BiT-L` 9.9k; `Noisy Student` 12.3k
  - Takeaway: On the same large dataset (JFT), `ViT-L/16` outperforms `BiT-L` across all listed datasets with substantially less pre-training compute. `ViT-H/14` improves further, especially on harder suites like VTAB (Figure 2).
- Data requirements and scaling behavior
  - Pre-training dataset size matters (Figure 3; Table 5).
    - On ImageNet-1k pre-training, large `ViT-L` underperforms `ViT-B` and CNNs; with ImageNet-21k they become comparable; with JFT-300M `ViT` surpasses CNNs on transfer.
    - Quote (Table 5, fine-tuned ImageNet top-1): `ViT-B/16` 77.91 (ImageNet pretrain) → 83.97 (IN-21k) → 84.15 (JFT); `ViT-L/16` 76.53 → 85.15 → 87.12/88.04 (14 epochs/`H/14`).
  - Few-shot scaling on ImageNet vs pre-training size (Figure 4)
    - `ViT` underperforms on 9M examples but overtakes as pre-training grows to 90M and 300M; CNNs plateau earlier.
- Compute–accuracy trade-offs (Figure 5; Table 6)
  - Averaged over 5 datasets, `ViT` reaches a given accuracy with ~2–4× less exaFLOPs than ResNets. Hybrids slightly help at small budgets but provide no advantage at large scale.
  - Detailed per-model numbers (Table 6) align with the trend.
- Implementation practicality (Appendix D.5)
  - On TPUv3, `ViT` achieves similar peak inference speed to comparable ResNets across input sizes and supports substantially larger per-core batch sizes (better memory efficiency).
- Ablations and analyses
  - Positional embeddings (Appendix D.4; Table 8): Removing positional information hurts; 1D, 2D, and relative encodings perform similarly at patch-level tokenization.
  - Class token vs global average pooling (Appendix D.3; Figure 9): Both can work; they require different learning rates during pre-training.
  - Transformer shape scaling (Appendix D.2; Figure 8): Depth and sequence length (smaller patches) yield the largest gains for a given parameter budget; width gives smaller returns.
  - ResNet optimizer choice (Appendix D.1; Table 7): Adam pre-training slightly outperforms SGD in this transfer setting, improving fairness when comparing to `ViT` (which uses Adam).
  - Attention behavior (Figures 6–7; Appendix D.7–D.8): Early heads mix local/global contexts; attention distance increases with depth; attention highlights semantically relevant regions.
  - Self-supervised `ViT` (Section 4.6; Appendix B.1.2): Masked patch prediction for `ViT-B/16` achieves 79.9% top-1 on ImageNet—about +2% over training from scratch yet ~4% behind supervised pre-training at similar scale.
  - Alternative axial attention (Appendix D.6; Figure 13): Can improve accuracy but increases compute; naïve implementations are slow on TPUs.
- Robustness and negatives
  - With small pre-training datasets, `ViT` overfits more than ResNets (Figure 4).
  - Gains depend on sufficient pre-training epochs and data scale; larger models realize benefits only with large datasets (Figure 3; Table 6).

Assessment of evidence:
- The evaluations convincingly show that standard Transformers, when trained at scale, are competitive or superior for classification, and they quantify compute advantages carefully (Figure 5; Table 6).
- Results on public data (ImageNet-21k) remain strong (Table 2, “Ours-I21k”), though slightly below JFT-300M pre-training, supporting generality beyond proprietary data.
- The analysis includes ablations and training details (Tables 3–4) that increase reproducibility and clarify what drives performance.

## 6. Limitations and Trade-offs
- Data scale dependency
  - `ViT` needs large pre-training datasets to shine; on ImageNet-1k pre-training, it underperforms similarly sized CNNs (Figure 3; Table 5).
- Task coverage
  - Focused on image classification. Object detection/segmentation are not addressed; only discussed as future directions (Conclusion).
- Inductive bias trade-off
  - Minimal built-in spatial bias means more data is required to learn locality/translation properties; overfitting risk is higher at small scales (Figure 4).
- Compute and hardware
  - Although compute-efficient at scale (Figure 5), top models still require significant resources (e.g., `ViT-H/14` at 2.5k TPUv3-core-days; Table 2). Access to JFT-300M is limited; public-data results are slightly lower.
- Positional interpolation assumptions
  - Fine-tuning at higher resolution relies on 2D interpolation of positional embeddings (Section 3.2). This works empirically but is an additional heuristic step.
- Self-supervision gap
  - The masked patch objective lags behind supervised pre-training by ~4% on ImageNet for `ViT-B/16` (Section 4.6), indicating room for better self-supervised objectives.

## 7. Implications and Future Directions
- Field impact
  - Establishes that “plain” Transformers are a strong foundation for vision classification when trained at scale, reducing the need for specialized vision architectures. This unifies modeling across language and vision and simplifies scaling and engineering (Sections 1, 4.4).
- Research avenues
  - Self-supervised pre-training: Explore contrastive or modern masked-image modeling variants that could close the gap to supervised pre-training (Section 4.6).
  - Beyond classification: Extend the patch-token Transformer approach to detection, segmentation, and multi-modal tasks; findings and related work (e.g., DETR) suggest promise (Conclusion; Section 2).
  - Efficient scaling: Investigate better patch sizes, depth/width trade-offs, and attention sparsity that preserve the hardware simplicity advantage (Appendix D.2; D.6).
  - Inductive bias hybridization: The hybrid results suggest small benefits at low compute; principled ways to inject just-enough spatial bias may reduce data needs without sacrificing scalability (Figure 5).
- Practical applications
  - Industrial-scale classification systems where pre-training data is abundant (e.g., retail visual search, content moderation).
  - Low-data transfer: VTAB results and few-shot analyses indicate strong transfer properties after large-scale pre-training (Table 2; Figure 4), beneficial for domains with limited labeled data.
  - Deployment: Memory efficiency and large per-core batch sizes (Appendix D.5) make `ViT` attractive for large-scale training and inference on modern accelerators.

Overall, this paper shows that a minimal-bias Transformer, fed with image patches, becomes a top-tier vision classifier when trained on sufficiently large image corpora, with favorable compute–accuracy trade-offs and simple fine-tuning techniques. The work shifts the default vision backbone choice toward Transformers and opens a broad agenda for scaling, self-supervision, and extension to richer vision tasks.
