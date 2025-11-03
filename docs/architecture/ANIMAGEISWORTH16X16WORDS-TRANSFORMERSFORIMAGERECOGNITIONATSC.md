# AN IMAGE IS WORTH 16X16 WORDS: TRANSFORMERS FOR IMAGE RECOGNITION AT SCALE

**ArXiv:** [2010.11929](https://arxiv.org/abs/2010.11929)

## 🎯 Pitch

This paper introduces Vision Transformer (ViT), a groundbreaking model that applies the standard Transformer architecture—originally designed for NLP—directly to sequences of image patches, thereby eliminating the need for convolutional layers in image classification. By leveraging large-scale pre-training, ViT achieves state-of-the-art performance on benchmark vision tasks with substantially less computational cost than leading convolutional networks, signaling a paradigm shift that simplifies model design and unifies vision and language architectures under the same scalable framework.

---

## 1. Executive Summary
This paper introduces Vision Transformer (`ViT`), a “pure” Transformer model that classifies images by treating an image as a sequence of fixed-size patches (e.g., 16×16 pixels) and applying the standard Transformer encoder used in NLP. When pre-trained on large datasets (ImageNet-21k with 14M images or JFT-300M with 303M images) and then fine-tuned, ViT achieves state-of-the-art or competitive results on many image classification benchmarks while requiring substantially less pre-training compute than strong convolutional baselines (Tables 1–2, Sections 3–4).

## 2. Context and Motivation
- The specific gap addressed
  - Transformer architectures dominate NLP because they scale well with data and compute, and benefit enormously from large-scale pre-training. In vision, convolutional networks (CNNs) remain dominant; prior uses of attention in vision either augment CNNs or replace only parts of them, often with specialized attention patterns that are hard to scale efficiently (Section 1; Related Work, Section 2).
  - A naïve application of global attention to pixels is computationally infeasible (quadratic in pixels), so prior work used local or sparse attention, or CNN+attention hybrids. Even “stand-alone” attention models struggled to scale on hardware due to custom patterns (Section 1; Section 2).

- Why this matters
  - A single, simple architecture that scales, reuses the mature NLP stack, and works across modalities is attractive for research and industry. If a “pure” Transformer can replace much of the hand-crafted inductive bias in CNNs (locality, translation equivariance), it simplifies model design and can benefit from the same scaling rules that drove NLP progress (Section 1).

- Prior approaches and their shortcomings
  - Local/sparse/axial attention variants reduce cost but require custom kernels or complex engineering; they have not been scaled “effectively on modern hardware accelerators” (Section 1; Related Work, Section 2).
  - CNN+attention hybrids deliver gains but keep convolutional structure and its biases (Section 2).
  - iGPT treats images as sequences of pixels and trains a large generative Transformer. It achieves 72% top-1 on ImageNet with linear probes—well below SOTA classifiers—and incurs heavy generative pre-training cost (Section 2).

- How this work positions itself
  - ViT applies a standard Transformer encoder directly to a sequence of image patches, with minimal image-specific components. It shows that, at scale, this minimal inductive bias suffices and can outperform ResNet-based SOTA on multiple benchmarks, at lower pre-training cost (Sections 1, 3, 4; Figure 5; Table 2).

## 3. Technical Approach
Core idea: Represent an image as a sequence of tokens, each token being a flattened P×P patch (e.g., P=16), and feed this sequence to a standard Transformer encoder with almost no architectural changes (Section 3.1; Figure 1).

Step-by-step mechanics (Section 3.1; Equations 1–4; Figure 1):
1. Patchification and embedding
   - Input image `x ∈ R^{H×W×C}` is partitioned into `N = (H·W)/P^2` non-overlapping patches of size `P×P`.
   - Each patch is flattened to a vector of length `P^2·C` and linearly projected to a `D`-dimensional embedding via a trainable matrix `E ∈ R^{(P^2·C)×D}` (Eq. 1).
   - Definition: `patch embedding` is this linear projection that maps a flattened image patch to a fixed-length vector.

2. Add a learnable class token and positional information
   - A learned `classification token` (`x_class`) is prepended to the patch sequence; the Transformer’s final representation of this token is used as the image representation for classification (Eq. 4). This mirrors BERT’s `[CLS]`.
   - Learnable `position embeddings` (`E_pos ∈ R^{(N+1)×D}`) are added to preserve order (Eq. 1). The paper finds no advantage from more elaborate 2D-aware position embeddings (Appendix D.4; Table 8).

3. Transformer encoder layers
   - The model uses `L` standard Transformer encoder blocks. Each block has:
     - Pre-LayerNorm, Multi-Head Self-Attention (MSA), residual connection (Eq. 2; Appendix A).
     - Pre-LayerNorm, 2-layer `MLP` with GELU nonlinearity, residual connection (Eq. 3).
   - The latent size `D` is constant across layers. The final class token representation is LayerNorm’ed to yield the image representation `y` (Eq. 4).

4. Classification head
   - During pre-training: an MLP head (one hidden layer) on top of `y` (Section 3.1).
   - During fine-tuning: a single linear layer (Section 3.1). Appendix D.3 shows global average pooling works similarly if one retunes the learning rate (Figure 9), but the class token is used throughout for simplicity.

5. Minimal inductive bias
   - CNNs encode locality and translation equivariance in every layer. ViT removes most of this: attention is global by default and learns spatial relations from data. The only 2D bias is from patch extraction and the way positional embeddings are resized for different input resolutions (Section 3.1 “Inductive bias”; Section 3.2).

6. Fine-tuning at higher resolution (Section 3.2)
   - It is often beneficial to fine-tune on higher-resolution images while keeping the same patch size, which increases the sequence length.
   - Key trick: interpolate the learned 2D grid of position embeddings to the new patch grid (bilinear-like 2D interpolation), then fine-tune. This simple interpolation bridges pre-training and fine-tuning resolutions.

7. Variants and sizes (Table 1; model naming)
   - `ViT-Base (B)`: 12 layers, D=768, 12 heads, 86M params.
   - `ViT-Large (L)`: 24 layers, D=1024, 16 heads, 307M params.
   - `ViT-Huge (H)`: 32 layers, D=1280, 16 heads, 632M params.
   - Naming: `ViT-L/16` means `Large` with `P=16` pixels per patch. Smaller `P` yields longer sequences and higher cost, but improves accuracy.

8. Hybrid models (Section 3.1)
   - Replace raw patches with tokens from a CNN feature map (e.g., ResNet stage output), optionally with “patch size” 1×1 (simply flatten spatial positions). These hybrids test whether early convolutional processing helps.

Training and evaluation setup (Section 4.1; Table 3; Table 4):
- Pre-training datasets: ImageNet-1k (1.3M), ImageNet-21k (14M), JFT-300M (303M). De-duplicated against downstream test sets.
- Downstream benchmarks: ImageNet (original labels and ReaL), CIFAR-10/100, Oxford Pets, Oxford Flowers, VTAB-1k (19 tasks across Natural, Specialized, Structured).
- Optimization:
  - Pre-train with Adam, high weight decay (0.1), large batches (4096), linear LR warmup/decay (Table 3). Training resolution 224.
  - Fine-tune with SGD + momentum, cosine LR decay, batch size 512, typically at 384 resolution; for best ImageNet numbers, fine-tune at ~512 resolution and use Polyak averaging (Section 4.1; Table 4).
- Compute reporting: TPUv3-core-days for headline SOTA (Table 2) and exaFLOPs for the controlled compute study (Figure 5; Table 6).

How and why this design
- Simplicity and scalability: using a standard Transformer allows reuse of optimized NLP training stacks “almost out of the box” and tests whether inductive bias is truly necessary if one scales data/model/compute (Section 3).
- Patch tokens instead of pixel tokens: reduces sequence length by P^2, making global attention feasible on standard resolutions (Section 3.1; compare to pixel-level iGPT in Section 2).
- Interpolated position embeddings enable decoupling pre-training resolution from fine-tuning resolution with almost no engineering (Section 3.2).

## 4. Key Insights and Innovations
- Treating images as sequences of patches is sufficient for high-accuracy classification at scale
  - Novelty: Use a standard Transformer encoder with minimal modifications—just patchification, class token, and position embeddings (Figure 1; Equations 1–4).
  - Significance: When pre-trained on large datasets, ViT matches or exceeds the best CNNs on ImageNet, CIFAR-100, Flowers, Pets, and the VTAB suite (Table 2). This overturns the view that strong convolutional inductive biases are strictly necessary for SOTA classification.

- Scaling trumps inductive bias
  - Evidence: With ImageNet-only pre-training, ViT trails strong CNNs; with ImageNet-21k or JFT-300M pre-training, larger ViTs overtake CNNs (Figure 3; Table 5).
  - Quote:
    > Figure 3 shows ViT-L underperforms on ImageNet pre-training but surpasses ResNet-based BiT when pre-trained on ImageNet-21k/JFT-300M; ViT-H/14 reaches 88.55% top-1 on ImageNet (Table 2).

- Better performance per unit compute than CNNs at large scale
  - Controlled study (Figure 5; Table 6): Across multiple model sizes and datasets, ViT achieves the same transfer accuracy using roughly 2–4× less pre-training compute than ResNets.
  - Practicality: Headline SOTA on ImageNet is achieved with fewer TPUv3-core-days than prior SOTA (Table 2).

- Simple resolution transfer via 2D interpolation of position embeddings
  - Innovation: A minimal technique to adapt pre-trained position embeddings when fine-tuning at higher resolution (Section 3.2).
  - Impact: Allows decoupled pre-training/fine-tuning resolutions, which improves performance (Section 4.1; Table 2 setup).

- Empirical analysis reveals ViT learns both local and global interactions early
  - Attention distance analysis (Figure 7 right; Figure 11) shows some heads attend globally from shallow layers while others stay local—emulating a mixture of CNN-like and non-local behaviors, learned from data rather than imposed by architecture.
  - Attention maps highlight semantically relevant regions (Figure 6; Appendix D.8).

These are fundamental innovations (rethinking image modeling around standard Transformers) rather than incremental tweaks to CNNs or specialized attention blocks.

## 5. Experimental Analysis
Evaluation methodology (Section 4.1)
- Datasets and tasks
  - Pre-training: ImageNet-1k (1.3M), ImageNet-21k (14M), JFT-300M (303M).
  - Transfer: ImageNet (original and ReaL labels), CIFAR-10/100, Oxford Pets, Oxford Flowers, VTAB-1k (19 tasks across Natural, Specialized, Structured categories).
- Metrics
  - Top-1 accuracy for classification tasks; VTAB uses the official average. Few-shot evaluation uses a closed-form regularized least-squares classifier on frozen features (Section 4.1 “Metrics”).
- Baselines
  - BiT (ResNet-based, supervised pre-training) and Noisy Student (EfficientNet-L2 with semi-supervised learning) for SOTA comparison (Table 2).
  - Controlled compute study includes a range of ResNets, ViTs, and hybrid CNN+ViT models trained for comparable epochs with measured exaFLOPs (Figure 5; Table 6).

Headline results (Table 2; Sections 4.2, 4.4)
- SOTA-level performance with less pre-training compute:
  - Quote:
    > On ImageNet, `ViT-H/14` (pre-trained on JFT) achieves 88.55% top-1 and 90.72% on ReaL, surpassing BiT-L (87.54%) while using 2.5k vs 9.9k TPUv3-core-days for pre-training. `ViT-L/16` (JFT) obtains 94.55% on CIFAR-100, and `ViT-H/14` attains 77.63% on VTAB-19 (Table 2).
  - `ViT-L/16` pre-trained on public ImageNet-21k achieves strong results too (e.g., 85.30% ImageNet top-1), with only 0.23k TPUv3-core-days (Table 2).

- Controlled performance vs compute (Figure 5; Table 6):
  - Averaged across five datasets, ViTs reach target accuracy with 2–4× less pre-training compute than ResNets.
  - Hybrids slightly outperform ViT at small scales but the advantage vanishes as models grow (Figure 5).

Data scale requirements (Section 4.3; Figure 3; Figure 4; Table 5)
- With small pre-training sets (ImageNet-1k), larger ViTs underperform smaller ViTs and BiT; with ImageNet-21k, the gap closes; with JFT-300M, larger ViTs significantly outperform (Figure 3; Table 5).
- Quote:
  > Few-shot ImageNet accuracy vs JFT subset size (Figure 4): ResNets are ahead at 9M samples but plateau, while ViTs overtake them on 90M+ samples.

Ablations and analyses
- Positional encoding (Appendix D.4; Table 8): No clear gains from 2D or relative positional embeddings over simple 1D learned embeddings when operating on patch tokens.
- Class token vs global average pooling (Appendix D.3; Figure 9): Both work, but require different learning rates; the paper chooses the class token for consistency with NLP practice.
- Architectural scaling (Appendix D.2; Figure 8): Depth scales more effectively than width; reducing patch size (longer sequences) consistently improves accuracy but increases compute.
- Self-supervision (Section 4.6; Appendix B.1.2): A masked patch prediction objective yields 79.9% ImageNet top-1 with `ViT-B/16`, about +2% vs training from scratch but ~4% behind supervised pre-training on large datasets.
- Attention behavior (Section 4.5; Figures 6–7; Appendix D.7–D.8): Some heads attend globally even in early layers; attention maps align with salient object regions.
- Compute/runtime (Appendix D.5; Figure 12): ViTs have comparable inference speed to ResNets and are more memory-efficient (support larger batch sizes per core). The theoretical quadratic scaling with sequence length becomes noticeable only for the largest models at highest resolutions.
- Alternative attention (Appendix D.6; Figure 13): Axial attention variants can improve accuracy but, with non-optimized implementations, are slow on TPUs; the main paper focuses on standard global attention due to simplicity and speed.

Are the experiments convincing?
- Strengths
  - Extensive controlled compute study across architectures (Figure 5; Table 6) addresses fairness concerns.
  - Consistent pre-training/fine-tuning procedures and reporting of training details (Tables 3–4) aid reproducibility.
  - Broad evaluation on many benchmarks, including low-data VTAB-1k and ImageNet-ReaL, plus per-task VTAB breakdown (Figure 2; Appendix Table 9).
- Caveats
  - The strongest results use JFT-300M, an internal dataset. Still, `ViT-L/16` trained on public ImageNet-21k shows robust gains (Table 2).
  - Some hyperparameter choices favor Transformers (e.g., Adam pre-training, high weight decay), though Appendix D.1 shows Adam pre-training also helps ResNets in this setup.

## 6. Limitations and Trade-offs
- Data hunger when inductive bias is low
  - ViT underperforms CNNs when pre-trained on small datasets (ImageNet-1k) even with regularization (Figure 3; Table 5). The model benefits greatly from large-scale pre-training (ImageNet-21k/JFT).

- Compute/memory vs sequence length
  - Global attention scales quadratically with sequence length; smaller patch size (e.g., `P=16` vs `P=32`) increases cost but boosts accuracy (Table 5; Appendix D.2). While wall-clock measurements (Figure 12) are favorable, very high resolutions can become expensive.

- Dependence on labeled pre-training (as evaluated)
  - The strongest results rely on supervised pre-training on massive labeled datasets; preliminary masked-patch self-supervision lags supervised pre-training by ~4% on ImageNet (Section 4.6).

- Minimal inductive bias can be a double-edged sword
  - Without built-in translation equivariance or locality, ViT must learn such patterns from data. This can impair performance in low-data regimes or domains with strong geometric priors unless compensated by pre-training or hybrids (Sections 3.1, 4.3; Figure 5 hybrids at small scales).

- Dataset accessibility and environmental cost
  - JFT-300M is not publicly available; even with better compute-efficiency than CNNs, headline models require thousands of TPUv3-core-days (Table 2). This may limit replicability for smaller labs.

- Scope of tasks
  - The study focuses on classification; downstream tasks like detection/segmentation are not addressed here, though related work (e.g., DETR) suggests feasibility (Conclusion, Section 5).

## 7. Implications and Future Directions
- How this changes the field
  - Demonstrates that a standard Transformer, minimally adapted to images via patch tokens, can outperform top CNNs when scaled. This catalyzes a shift toward unified Transformer-based vision models that can benefit from the same scaling laws and tooling as NLP (Sections 1, 4).

- Research directions enabled or suggested
  - Data efficiency
    - Develop stronger self-supervised or semi-supervised objectives for ViT (e.g., contrastive or masked-image modeling) to close the gap with supervised pre-training (Section 4.6).
    - Explore curriculum, augmentation, or distillation to reduce data requirements.
  - Architectural refinements
    - Hierarchical or multi-scale patching to better handle high resolutions at lower cost; adaptive tokenization.
    - Hybridization that helps small-scale regimes while preserving ViT’s large-scale strengths (Figure 5).
    - Efficient attention variants with hardware-friendly implementations (Appendix D.6).
  - Beyond classification
    - Extend ViT to detection, segmentation, and video understanding; leverage global attention for long-range reasoning (Conclusion, Section 5).
  - Understanding and safety
    - Deeper analysis of attention patterns, robustness, and fairness across domains (Appendix D.5–D.8).

- Practical applications
  - Any setting where large-scale pre-training is feasible and transfer to diverse downstream image tasks is desired: cloud vision services, medical/satellite imagery with fine-tuning on limited labels (VTAB Specialized), and embedded systems that benefit from ViT’s memory efficiency (Figure 12).

In short, by reducing image modeling to patch-level token sequences processed by a standard Transformer, ViT shows that scalability and data can overcome the lack of hard-coded vision inductive biases. The study pairs a simple method with rigorous scaling and analysis, reshaping the roadmap for vision architectures.
