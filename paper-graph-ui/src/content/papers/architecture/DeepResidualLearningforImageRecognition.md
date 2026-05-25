# Deep Residual Learning for Image Recognition

**ArXiv:** [1512.03385](https://arxiv.org/abs/1512.03385)

## 🎯 Pitch

This paper introduced the concept of residual learning, transforming how deep neural networks are built and optimized by letting layers explicitly learn residual functions via identity shortcut connections. This simple yet powerful idea enables extremely deep networks (ResNets) to be trained reliably, leading to groundbreaking accuracy on major benchmarks such as ImageNet and COCO. By overcoming fundamental optimization barriers, ResNets set new performance standards, profoundly influencing the evolution and capabilities of modern computer vision systems.

---

## 1. Executive Summary
This paper introduces residual learning, a simple architectural idea that lets very deep convolutional networks train reliably by having layers learn changes relative to their inputs rather than full transformations. Implemented via identity “shortcut” connections, residual networks (ResNets) reach unprecedented depth (up to 152 layers on ImageNet and 1202 layers on CIFAR-10) and set state-of-the-art accuracy across classification, detection, and localization tasks.

## 2. Context and Motivation
- The gap addressed
  - Stacking more layers had been the main path to better image recognition, but beyond a certain depth, “plain” deep networks became harder to optimize and started to perform worse as they got deeper. This “degradation problem” means deeper models show higher training error than shallower ones even though, in principle, they have larger solution spaces.
  - Evidence: On CIFAR-10, a 56-layer plain network has higher training and test error than a 20-layer plain network (Fig. 1). On ImageNet, a 34-layer plain net trains worse than an 18-layer plain net (Fig. 4 left; Table 2).

- Why it matters
  - Depth is a key driver of representational power in vision. Overcoming the optimization barrier unlocks further accuracy gains and new capabilities in classification, detection, and segmentation.
  - Practically, the paper’s models achieve state-of-the-art results on ImageNet and large improvements on COCO detection, implying broad downstream impact in applications that rely on robust visual perception.

- Prior approaches and their limits
  - Vanishing/exploding gradients were already mitigated by improved initialization and batch normalization, enabling dozens of layers to start converging. Yet, degradation persisted: deeper plain nets trained worse (Sec. 1; Fig. 4 left).
  - Shortcut-like ideas existed (e.g., “highway networks” with learnable gates), but they introduced gating parameters and had not shown accuracy gains when scaled past ~100 layers (Sec. 2).

- Positioning
  - The paper reframes deep learning as residual learning: each block learns a residual function relative to its input and adds it back via parameter-free identity connections. This addresses optimization (not only generalization) and enables substantially deeper, more accurate networks.

## 3. Technical Approach
Residual learning in one sentence: instead of making a stack of layers directly approximate a desired function `H(x)`, make it approximate the residual `F(x) = H(x) − x`, and output `y = F(x) + x`.

- Core building block (Fig. 2; Eqns. 1–2)
  - Equation: `y = F(x, {W_i}) + x` (Eqn. 1).
    - `x`: block input; `y`: block output.
    - `F`: the residual function (typically two or three convolutional layers with BatchNorm and ReLU).
    - The “shortcut” implements identity mapping and is added element-wise to `F`.
  - If the input/output dimensionalities differ, use a projection on the shortcut:
    - `y = F(x, {W_i}) + W_s x` (Eqn. 2), where `W_s` is a 1×1 convolution (“projection shortcut”).
  - Design details:
    - Nonlinearity placement: a ReLU is applied after the addition (σ(y) in Fig. 2).
    - For matched dimensions, the shortcut is identity—no extra parameters or FLOPs.

- Why this helps (intuition formalized)
  - If the optimal mapping is (close to) identity, it is difficult for a stack of nonlinear layers to learn identity; it is easy to learn a residual `F(x) ≈ 0`. The optimizer can push the residual weights toward zero to approach identity (Sec. 3.1).
  - Even when the optimal function is not identity, learning a perturbation relative to identity can be easier (“preconditioning” intuition). Empirical evidence: residual responses have smaller magnitudes than plain responses (Fig. 7), consistent with “small residuals around identity.”

- Plain vs. residual architectures (Fig. 3; Table 1)
  - Plain 34-layer net (Fig. 3 middle): stacks 3×3 conv layers, doubles channels when spatial size halves, uses global average pooling + 1000-way FC layer.
  - Residual 34-layer net (Fig. 3 right): same as plain, plus identity shortcuts after every two 3×3 convs. When spatial resolution changes, shortcuts either:
    - Option A: identity with zero-padding to increase channels (no parameters),
    - Option B: projection (1×1 conv) only when increasing dimensions,
    - Option C: projection for all shortcuts.
  - FLOPs: 34-layer plain/residual nets are 3.6B FLOPs—about 18% of VGG-19 (Fig. 3 left; 19.6B).

- Deeper “bottleneck” design (Fig. 5; Table 1)
  - To scale depth efficiently, each residual block becomes three layers: 1×1 (reduce channels) → 3×3 (compute) → 1×1 (restore channels). This keeps computation controlled while increasing depth (Sec. “Deeper Bottleneck Architectures”).
  - Identity shortcuts are particularly important here: replacing identity with projection would double complexity because the shortcut touches high-dimensional ends (Sec. 4, “Deeper Bottleneck Architectures”).

- Training and evaluation setup (Sec. 3.4)
  - ImageNet training:
    - Data: 1.28M images, standard augmentations—scale jittering (shorter side ∈ [256, 480]), random 224×224 crops/horizontal flips, per-pixel mean subtraction; color augmentation from [21].
    - Optimization: SGD, batch size 256, initial LR 0.1 decayed by 10× on plateaus, 60×10^4 iterations, momentum 0.9, weight decay 1e-4; BatchNorm used throughout; no dropout.
    - Testing: 10-crop for comparison studies; best models use fully-convolutional multi-scale testing (shorter side ∈ {224, 256, 384, 480, 640}).
  - CIFAR-10 training:
    - Architecture: 6n 3×3 conv layers over feature maps of sizes 32, 16, 8 (2n layers each), followed by global average pooling and 10-way classifier; identity shortcuts after every pair of 3×3 layers (Sec. 4.2).
    - Depths tried: 20, 32, 44, 56, 110, and 1202 layers.
    - Optimization: SGD with batch 128, LR schedule starting at 0.1 (warmup 0.01 for 110-layer), decays at 32k and 48k iterations; standard CIFAR augmentation (4-pixel padding + random crops/flips).

- Detection and localization integration (Appendix A–C)
  - Detection (Faster R-CNN backbone replacement):
    - Use ResNet as shared convolutional “feature extractor” up to a total stride of 16 (conv1–conv4_x), analogous to VGG-16’s conv stack.
    - RoI pooling before conv5_1; per-RoI “conv5_x and up” serve as VGG’s fully connected heads (Appendix A).
    - For training with limited GPU memory, BatchNorm statistics are precomputed on ImageNet and then fixed during fine-tuning.
  - Localization (per-class RPN and R-CNN; Appendix C):
    - Per-class binary classification and per-class bounding box regression heads (1000 classes) over translation-invariant anchor boxes.
    - Dense (fully convolutional) multi-scale testing; further improves with a per-class R-CNN stage that refines top proposals.

## 4. Key Insights and Innovations
- Residual formulation with identity shortcuts (fundamental)
  - What’s new: Recasting `H(x)` as `F(x) + x` and implementing the addition via parameter-free identity connections (Eqn. 1; Fig. 2).
  - Why it matters: It targets the optimization barrier directly and makes very deep networks trainable without extra parameters on the shortcuts. Evidence: deeper ResNets train to lower training error than shallower ones (Fig. 4 right), reversing the degradation seen in plain nets (Fig. 4 left).

- Minimal, general building block (practical and conceptual)
  - No gates, no extra parameters for most shortcuts; only use projections when dimensions change (Eqn. 2).
  - Unlike highway networks (Sec. 2), identity shortcuts are always “open,” guaranteeing information flow and ensuring each block learns a residual.

- Efficient bottleneck blocks (engineering enabling extreme depth)
  - The 1×1–3×3–1×1 “bottleneck” makes 50/101/152-layer models feasible with manageable FLOPs, still lower than VGG-16/19 while being far deeper (Table 1).

- Empirical diagnosis of residual behavior (insightful analysis)
  - Measured response magnitudes show residual functions tend to be small (Fig. 7), consistent with the premise that blocks learn modest adjustments to their inputs.

- Strong cross-task generalization (impact)
  - Simply swapping VGG-16 with ResNet-101 in Faster R-CNN yields large gains on PASCAL VOC and COCO baselines (Table 7, Table 8). The improvements extend to ImageNet DET and LOC (Table 12–14).

## 5. Experimental Analysis
- Evaluation methodology
  - Datasets and metrics:
    - ImageNet Classification (1000 classes): top-1 and top-5 error on validation and test (Sec. 4.1).
    - CIFAR-10: classification error on the test set with standard augmentation (Table 6).
    - PASCAL VOC 2007/2012 Detection: mAP@0.5 IoU (“VOC metric”; Tables 7, 10, 11).
    - MS COCO Detection: mAP@0.5 and COCO’s primary metric mAP@[0.5, 0.95] (Table 8, Table 9).
    - ImageNet Localization: top-5 localization error given ground-truth class and under predicted class (Tables 13–14).
  - Baselines:
    - Internally controlled: “plain” networks of the same depth/parameters vs. ResNets (Fig. 4; Table 2–3).
    - External: VGG-16/19, GoogLeNet/BN-Inception, PReLU-net (Tables 3–5).
    - For detection: Faster R-CNN with VGG-16 vs. ResNet backbones under identical training setup (Appendix A; Tables 7–8).

- Main quantitative results
  - Degradation vs. residual learning (ImageNet, controlled setting):
    - Plain-34 performs worse than plain-18 (Table 2 top-1: 28.54% vs. 27.94%), while ResNet-34 outperforms ResNet-18 (25.03% vs. 27.88%).
    - Training curves (Fig. 4): deeper plain nets have higher training error throughout; deeper ResNets have lower training error and lower validation error.
  - Shortcut design ablation (Table 3):
    - ResNet-34 options: A (all identity, zero-pad on channel increases) top-1 25.03%; B (projection only when increasing dims) 24.52%; C (projection everywhere) 24.19%. All are much better than plain-34 (28.54%). Projections help slightly; identity alone already solves degradation.
  - Depth scaling with bottlenecks (Table 3–4):
    - ResNet-50/101/152 top-1 (10-crop val): 22.85%, 21.75%, 21.43% and top-5: 6.71%, 6.05%, 5.71%.
    - Single-model multi-scale val top-5 (Table 4): ResNet-152 4.49%, beating BN-Inception (5.81%) and PReLU-net (5.71%).
    - Ensemble (Table 5): 3.57% top-5 test error—1st place in ILSVRC 2015 classification.
  - CIFAR-10 depth study (Fig. 6; Table 6):
    - Plain nets worsen with depth (training/testing curves in Fig. 6 left; 110-layer plain net >60% error not even plotted).
    - ResNets improve with depth (Fig. 6 middle). ResNet-110: 6.43% error (best of five runs 6.43; mean 6.61±0.16). A 1202-layer ResNet trains to <0.1% training error but has 7.93% test error, indicating overfitting (Fig. 6 right; Table 6).
    - Residual response magnitudes are smaller than in plain nets (Fig. 7), consistent with learning small perturbations.
  - Detection (baseline swaps; Tables 7–8):
    - PASCAL VOC mAP@0.5: VGG-16 → ResNet-101 yields 73.2→76.4 (VOC07 test) and 70.4→73.8 (VOC12 test).
    - COCO validation: mAP@0.5 41.5→48.4; mAP@[0.5,0.95] 21.2→27.2. The +6.0 absolute gain in COCO’s primary metric is a 28% relative improvement.
  - Detection with additional techniques (Table 9):
    - Starting from ResNet-101 baseline: box refinement (+~2 mAP@0.5), global context (+~1 mAP@0.5), multi-scale testing (+~2.7 mAP@0.5). Final single-model on test-dev: 55.7 mAP@0.5, 34.9 mAP@[0.5,0.95]; ensemble of 3 models: 59.0 and 37.4.
    - PASCAL with these improvements: 85.6% (VOC07) and 83.8% (VOC12) mAP@0.5 (Tables 10–11).
  - ImageNet Detection & Localization (Tables 12–14):
    - Detection (DET): single-model 58.8% mAP; ensemble 62.1% (vs. 43.9% GoogLeNet ILSVRC’14).
    - Localization (LOC): with ground-truth class, center-crop error drops from 33.1% (VGG-16) to 13.3% using ResNet-101 RPN; dense testing: 11.7%. With predicted classes and RPN+R-CNN: 10.6% val; ensemble: 9.0% test (vs. 25.3% for VGG ILSVRC’14; Table 14).

- Do the experiments support the claims?
  - Yes, on optimization: controlled comparisons isolate the effect of residual connections by keeping depth/width/parameters constant (Table 2–3; Fig. 4). Residual nets train to lower training error and achieve higher validation accuracy.
  - Yes, on scalability: depth scaling from 34→152 layers consistently improves accuracy (Tables 3–4).
  - Yes, on generalization across tasks: swapping backbones in a fixed Faster R-CNN pipeline yields large, clean gains (Tables 7–8), and further improvements stack on top (Table 9).
  - Ablations: shortcut options A/B/C, residual response magnitudes (Fig. 7), and very-deep limits (1202 layers) provide robustness checks and expose overfitting behavior on small data.

- Qualitative assessment of trade-offs and conditions
  - Projection shortcuts give small additional gains but add parameters (Table 3).
  - Extremely deep models can overfit on small datasets (CIFAR-10, 1202-layer; Fig. 6 right).
  - For detection, fixing BN statistics during fine-tuning is a practical memory choice (Appendix A) and may slightly constrain adaptation, but results remain strong.

## 6. Limitations and Trade-offs
- Assumptions and design choices
  - The benefit depends on identity shortcuts providing a good reference mapping; when dimensions change, projections are needed (Eqn. 2), which add parameters and cost.
  - Residual blocks place ReLU after the addition (Fig. 2). Alternative placements are not explored in this paper.

- Scope and edge cases
  - While depth scales well on ImageNet, on small datasets extremely deep, high-capacity models can overfit (ResNet-1202 on CIFAR-10; Table 6; Fig. 6 right).
  - The paper focuses on feedforward convolutional recognition; no experiments on sequence modeling or non-vision domains, though the principle is general.

- Computational considerations
  - Despite lower FLOPs than VGG, very deep ResNets (e.g., 152 layers, 11.3B FLOPs; Table 1) are still computationally demanding for training and inference without specialized hardware.
  - Detection/segmentation pipelines add further cost (multi-scale testing, ensembling; Table 9).

- Open questions
  - The precise theoretical reason plain nets degrade (e.g., “exponentially low convergence rates” are conjectured in Sec. 4.1) is not proved.
  - How residual learning interacts with other regularization (e.g., dropout/maxout) is only briefly noted as future work in the context of CIFAR-10 overfitting (Sec. 4.2).

## 7. Implications and Future Directions
- Field impact
  - Residual learning changes the default design philosophy: instead of hand-crafting deeper stacks and hoping optimization succeeds, always provide identity paths and make layers learn residuals. This unlocks very deep, accurate networks and becomes a foundation for state-of-the-art vision systems (classification, detection, localization; Tables 5, 9, 12–14).

- Research avenues enabled
  - Theory: analyze why residual parameterization eases optimization—links to conditioning, implicit preconditioning, and function space bias (supported empirically by Fig. 7).
  - Architecture: explore residual layouts (e.g., where to place normalization/activation), block types, widths, and different shortcut forms; study when projections should be preferred.
  - Regularization and small-data regimes: combine residual learning with stronger regularizers to prevent overfitting in ultra-deep settings (as suggested by the 1202-layer CIFAR experiment).
  - Cross-domain applications: adapt residual connections to tasks beyond vision—detection and localization results suggest the principle is broadly useful; similar gains may occur in speech, NLP, and reinforcement learning.

- Practical applications
  - Immediate drop-in backbone improvements for detection/segmentation pipelines (Faster R-CNN, semantic segmentation), with documented large gains (Tables 7–11).
  - Production systems can benefit from the identity-shortcut design’s parameter efficiency (no extra parameters for most shortcuts) and the bottleneck blocks’ computational efficiency, achieving higher accuracy at comparable or lower FLOPs than prior very-deep nets (Table 1).

> Signature result: Table 5 reports an ensemble of residual nets achieving 3.57% top-5 error on ImageNet test; Table 9 shows 59.0%/37.4% mAP (mAP@0.5 / mAP@[0.5,0.95]) on COCO test-dev with a ResNet-101 ensemble; Table 14 reports 9.0% top-5 localization error—each a first-place result in ILSVRC/COCO 2015.

Overall, the paper’s contribution is both conceptual and practical: a small architectural change—identity shortcuts enabling residual learning—solves a core optimization bottleneck, scales depth dramatically, and delivers large, consistent accuracy gains across multiple, rigorous benchmarks.
