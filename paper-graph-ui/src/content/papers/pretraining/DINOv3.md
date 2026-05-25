# DINOv3

**ArXiv:** [2508.10104](https://arxiv.org/abs/2508.10104)

## 🎯 Pitch

DINOv3 introduces a new self-supervised vision foundation model that successfully scales Vision Transformers up to 7 billion parameters while preserving high-quality dense features—a breakthrough made possible by its innovative 'Gram anchoring' regularizer and scalable, metadata-free training pipeline. This enables a single frozen encoder to achieve state-of-the-art performance on dense computer vision tasks (like segmentation, depth, and tracking) while remaining competitive on global tasks, and crucially, it generalizes across diverse domains—including satellite imagery—without requiring fine-tuning, paving the way for robust, multi-task vision systems with dramatically lower deployment complexity.

---

## 1. Executive Summary
DINOv3 is a self-supervised vision foundation model and training recipe that scales a Vision Transformer (ViT) to 7B parameters while preserving high-quality dense features—solving a long-standing problem where large SSL models trained for long schedules lose local spatial consistency. It introduces a simple but powerful “Gram anchoring” regularizer, plus a robust data and training pipeline, to produce a frozen encoder that achieves state-of-the-art results on dense tasks (e.g., segmentation, depth, tracking) and competitive performance on global tasks, and it transfers across domains (including satellite imagery) without task-specific fine-tuning (Secs. 3–6; Figs. 2–4, 8–11; Tabs. 3–12, 17–19).

## 2. Context and Motivation
- The problem addressed
  - Scaling self-supervised learning (SSL) in vision to large models and long training schedules typically improves global recognition but degrades dense features (local patch-level consistency), which are crucial for segmentation, depth, correspondence, and tracking (Sec. 1; Fig. 5b–c, Fig. 6).
  - Standard training practice also depends on pre-specified cosine schedules and curated metadata; both are difficult to set or obtain at scale without labels (Sec. 1; Sec. 3.1–3.2).

- Why it matters
  - Dense features are the backbone of many high-value applications—autonomous driving, robotics, 3D vision, video understanding, medical and geospatial imaging—where pixel- or patch-level outputs are needed and annotations are scarce (Sec. 1; Sec. 2 “Dense Transformer Features”).
  - A frozen encoder that works off-the-shelf across tasks and domains can dramatically reduce compute and complexity in deployment, enabling multi-task systems with a single backbone (Sec. 1; Sec. 6.3).

- What existed before and where it fell short
  - Weakly supervised (image–text) models such as CLIP families excel in global classification and zero-shot transfer but typically produce noisy or lower-quality dense features (Tab. 3, Tab. 5; Fig. 13).
  - Self-supervised models like DINOv2 offered strong dense features but did not scale cleanly: longer training and larger models led to patch-level degradation (Sec. 1; Fig. 5a–c, Fig. 6; also discussed by Fan et al., 2025).
  - “Agglomerative” dense models distilling from supervised segmenters (e.g., SAM) achieve strong dense outputs but rely on annotation-heavy teachers and still lag on some domains/tasks (Fig. 2; Tab. 3).

- How this work positions itself
  - DINOv3 is an SSL-only pipeline that (1) scales model capacity and data with careful curation and constant hyperparameters; (2) introduces Gram anchoring to stabilize dense features during long training; (3) adds high-resolution post-training and efficient single-teacher/multi-student distillation; (4) extends to text alignment and a satellite domain variant—all while keeping the backbone frozen for most evaluations (Sec. 3–5; Sec. 6–8).

## 3. Technical Approach
This section explains the end-to-end recipe: data, model/losses, the Gram anchoring mechanism, post-training, and distillation.

- Data creation and sampling (Sec. 3.1; Tab. 1)
  - Three data parts from a 17B-image pool:
    1) A clustering-curated balanced subset `LVD-1689M` using hierarchical k-means over DINOv2 embeddings (Vo et al., 2024).
    2) A retrieval-curated subset “close” to seed datasets (as in DINOv2).
    3) Raw public CV datasets (ImageNet-1k/22k, Mapillary SLS).
  - Sampling strategy: in each iteration, either a homogeneous ImageNet-1k batch (10% of iterations) or a heterogeneous batch mixing the other components—balancing “high-quality focused” and “broad diverse” data (Sec. 3.1).
  - Ablation (Tab. 1): no single curation approach wins everywhere; the mixed pipeline gives the best average performance (e.g., IN1k-Linear 87.2, Paris Retrieval 85.9).

- Model and SSL losses (Sec. 3.2; Tab. 2; Eq. 1)
  - Architecture: a custom ViT-7B with 40 blocks, 4096-d embeddings, 32×128 heads, a patch size of 16, and 4 `register` tokens (extra learned tokens that absorb global interactions and mitigate patch outliers; Darcet et al., 2024). Uses rotary positional embeddings (`RoPE`) with “box jitter” to improve robustness across scales and aspect ratios (Tab. 2).
  - Multi-crop SSL with two global and eight local crops; batch size 4096 across 256 GPUs; constant learning rate/weight decay/momentum (no cosine schedule) with warmup only, enabling indefinite training (Sec. 3.2).
  - Pretraining objective (Eq. 1): `L_Pre = L_DINO + L_iBOT + 0.1 * L_DKoleo`
    - `L_DINO`: global discrimination via self-distillation with Sinkhorn-Knopp centering (like SwAV).
    - `L_iBOT`: patch-level latent reconstruction objective.
    - `L_DKoleo`: spreads features uniformly (Sablayrolles et al., 2018), implemented across GPUs in small batches.
    - Dedicated heads and layer norms for global and local branches stabilize training (Sec. 3.2).

- The observed failure mode: dense feature drift (Sec. 4.1; Fig. 5–6)
  - With scale and long training, classification keeps improving but dense performance declines (Fig. 5b–c).
  - Cosine similarity between CLS and patch tokens steadily increases, causing features to become less localized; similarity maps get noisier (Fig. 5a, Fig. 6).
  - Register tokens fix high-norm patch outliers (Appendix A.1; Fig. 20) but do not prevent the “similarity drift” that harms dense tasks.

- Gram anchoring: stabilize local structure without freezing features (Sec. 4.2; Eq. 2–3; Fig. 7–10)
  - Key idea: constrain the student’s pairwise patch similarities (its `Gram matrix`) to match those from an earlier “Gram teacher” checkpoint that still has clean dense structure.
    - Define `Gram matrix` as all pairwise dot-products between L2-normalized patch features within a crop.
    - Loss (Eq. 2): `L_Gram = || X_S X_S^T – X_G X_G^T ||_F^2`, where `X_S` are student patch features and `X_G` are the teacher’s.
  - Training schedule:
    - Run standard SSL for ~1M iterations; then start a refinement phase with `L_Ref = w_D L_DINO + L_iBOT + w_DKL L_DKoleo + w_Gram L_Gram` (Eq. 3).
    - Update the Gram teacher every 10k iterations to the current EMA teacher for a few steps (Sec. 4.2; Fig. 7).
  - Effects:
    - Immediate dense gains within ~10k iterations (Fig. 8), while global metrics remain stable or slightly improve.
    - `L_iBOT` decreases faster under Gram anchoring, suggesting synergy between local reconstruction and similarity-structure constraints (Fig. 7a–c).

- High-resolution Gram and high-resolution adaptation (Sec. 4.3; Sec. 5.1; Figs. 9–11)
  - Compute teacher features at 2× input resolution, then bicubic downsample features to the student resolution; use these smoothed high-res features for `L_Gram` (“`LHRef`” in Fig. 8–10).
    - Qualitatively preserves fine structure in downsampled cosine maps (Fig. 9a).
    - Quantitatively adds +2 mIoU on ADE20k over the non-HR Gram (Tab. in Fig. 9b).
  - Resolution adaptation (10k iterations) with mixed global/local crop sizes and Gram anchoring makes features robust up to very high resolutions (4K+)—improving dense tasks at high res while keeping global performance stable (Fig. 11; Fig. 4).

- Efficient multi-student distillation (Sec. 5.2; Fig. 12; Fig. 16; Tab. 14–15)
  - Distill the 7B teacher into a family of smaller ViTs (S, S+, B, L, H+) and ConvNeXts (T, S, B, L).
  - A single-teacher/multi-student pipeline shares expensive teacher inference across all GPUs, then trains each student in its own synchronized group; group sizes are tuned so all students step in lockstep (Fig. 12).
  - Results: large students (e.g., ViT-H+) match the 7B teacher closely (Fig. 16b); ConvNeXt students inherit robust dense and OOD performance even though they are convolutional (Tab. 15).

- Optional text alignment (Sec. 5.3; Tab. 16)
  - A lightweight `LiT`-style head trains a text encoder from scratch to match a frozen DINOv3 image encoder; concatenates mean-pooled patch embeddings with the CLS token to align both global and local signals.
  - Delivers competitive zero-shot classification/retrieval and strong open-vocabulary segmentation vs similarly sized models (Tab. 16).

## 4. Key Insights and Innovations
- Gram anchoring for dense-feature stability (fundamental)
  - What’s new: regularize the student’s patch-similarity structure (Gram matrix) toward an earlier “clean” iteration of itself—preserving local consistency while allowing global features to continue improving (Sec. 4.2; Eq. 2–3).
  - Why it matters: it “repairs” dense features after long training at scale—immediate mIoU gains and visibly cleaner similarity maps (Figs. 8–10)—solving a central scalability barrier for SSL.

- High-resolution Gram + resolution adaptation (fundamental + practical)
  - What’s new: compute teacher features at 2× resolution and distill their smoothed structure into the student; then run a short high-resolution mixed-crop training with Gram anchoring (Sec. 4.3; Sec. 5.1).
  - Why it matters: dense features remain crisp and semantically coherent up to very high resolutions (Figs. 4, 11), which is critical for dense tasks and large-scene imagery.

- Constant-schedule large-scale SSL with robust ViT-7B design (incremental but enabling)
  - What’s new: constant LR/weight-decay/momentum after warmup, axial RoPE with “box jitter,” registers for outlier control, and a scalable multi-crop setup (Sec. 3.2; Tab. 2; Appendix A).
  - Why it matters: simplifies long, indefinite training and supports scaling to 7B parameters without dense-feature collapse (once Gram anchoring is applied).

- Efficient multi-student distillation across architectures (practical innovation)
  - What’s new: a system to distill many students in parallel by sharing teacher inference, minimizing idle time with synchronized groups (Sec. 5.2; Fig. 12).
  - Why it matters: turns an expensive 7B teacher into a practical model family (ViT and ConvNeXt) that preserves the teacher’s dense and OOD strengths (Fig. 16; Tabs. 14–15).

- Domain-general SSL that transfers to satellites (demonstration of generality)
  - What’s new: the same recipe directly applies to a 493M-image satellite dataset, yielding state-of-the-art canopy height mapping and strong results on GEO-Bench and high-res remote sensing datasets (Sec. 8; Tabs. 17–19; Fig. 18–19).
  - Why it matters: shows a label-free path for high-impact scientific and industrial domains with scarce metadata.

## 5. Experimental Analysis
- Evaluation design: frozen encoders + light heads wherever possible (Sec. 6)
  - Dense linear probing: semantic segmentation (ADE20k/Cityscapes/VOC; mIoU) and depth (NYUv2/KITTI; RMSE) with a single linear layer on patch features (Sec. 6.1.2; Tab. 3).
  - Non-parametric dense tasks: 3D correspondences (NAVI/SPair; recall), unsupervised object discovery (TokenCut; CorLoc), video segmentation tracking (DAVIS/YouTube-VOS/MOSE; J&F) (Secs. 6.1.3–6.1.5; Tab. 4, Fig. 14, Tab. 5).
  - Video classification: attentive probe (4-layer transformer) on extracted per-frame patch features (UCF101/SSv2/Kinetics-400; top-1) (Sec. 6.1.6; Tab. 6).
  - Global probes: ImageNet and OOD variants (V2, ReaL, Renditions, Sketch, A, C (mCE), ObjectNet), fine-grained (Places205, iNat18/21, 12 small datasets) (Sec. 6.2; Tabs. 7–8, 22).
  - Instance retrieval: Oxford/Paris (mAP), Met (GAP), AmsterTime (mAP) (Sec. 6.2.2; Tabs. 9, 23).
  - Strong decoders on top of frozen backbone: detection (Plain-DETR on COCO/COCO-O), segmentation (ViT-Adapter + Mask2Former on ADE20k/COCO-Stuff/VOC/Cityscapes), depth (DPT in DAv2 pipeline), 3D (swap DINOv3 into VGGT) (Sec. 6.3; Tabs. 10–13, 24).

- Main results (selected highlights)
  - Dense linear probes (Tab. 3):
    > ADE20k mIoU: DINOv3 55.9 vs DINOv2 49.5 (+6.4), AM-RADIO 53.0, PEspatial 49.3  
    > Cityscapes mIoU: DINOv3 81.1 vs DINOv2 75.6 (+5.5)  
    > NYUv2 RMSE↓: DINOv3 0.309 vs DINOv2 0.372 (better), PEspatial 0.362  
    > KITTI RMSE↓: DINOv3 2.346 vs DINOv2 2.624 (better)

  - 3D correspondences (Tab. 4):
    > NAVI recall: DINOv3 64.4 vs DINOv2 60.1; SPair: 58.7 vs 56.1

  - Unsupervised object discovery (Fig. 14):
    > VOC07/12/COCO CorLoc: DINOv3 66.1 / 69.5 / 55.1 (best among compared models)

  - Video segmentation tracking (Tab. 5):
    > DAVIS J&F (L): DINOv3 83.3 vs DINOv2 76.6; YouTube-VOS (L): 80.7 vs 74.6; MOSE (L): 55.6 vs 48.5

  - Video classification (Tab. 6):
    > SSv2 Single: DINOv3 70.1 vs DINOv2 67.4; K400 Single: 87.8 vs 84.4; near SigLIP 2 / PEcore

  - Global linear probes (Tab. 7):
    > IN-1k val: DINOv3 88.4; OOD: ReaL 90.4, Rendition 91.1, Sketch 71.3, A 86.9, C mCE 19.6 (best), ObjectNet 79.0  
    Performance is close to or better than comparable weakly supervised models on several OOD sets.

  - Fine-grained classification (Tab. 8; Tab. 22):
    > iNat21: DINOv3 89.8 (best); Fine-S average: 93.0, competitive with PEcore 94.5

  - Instance retrieval (Tabs. 9, 23):
    > Oxford-H / Paris-H mAP: 60.7 / 87.1 (best)  
    > Met GAP: 55.4 (+10.8 over DINOv2)  
    > AmsterTime mAP: 56.5 (+7.6 over DINOv2)

  - Detection with frozen backbone (Tab. 10):
    > COCO mAP: 66.1 (TTA), state of the art among listed systems despite training only ~100M decoder parameters (backbone frozen)  
    > COCO-O mAP 66.4, ER 36.8 (highest ER listed)

  - Segmentation with frozen backbone (Tab. 11; Tab. 24):
    > ADE20k mIoU 63.0 (ties ONE-PEACE) at 896px; improves previous SOTA on COCO-Stuff/VOC among models listed, with frozen backbone

  - Depth (Tab. 12):
    > New SOTA on NYUv2, KITTI, ETH3D, ScanNet using frozen DINOv3 in a DPT head within DAv2 pipeline (e.g., NYUv2 ARel 4.3, δ1 98.0)  
    > On DIODE, ARel (25.6) lags behind DPT (18.2), though δ1 (82.2) is higher than some baselines—mixed result on that dataset

  - 3D with VGGT (Tab. 13):
    > Improves over VGGT with DINOv2 across camera pose estimation (Re10K/CO3Dv2), DTU multi-view depth (lower overall error), and ScanNet-1500 matching (higher AUC)

  - Model family (Fig. 16; Tab. 14–15):
    > ViT-H+ nearly matches the 7B teacher while using ~1/10th the parameters (Fig. 16b)  
    > ConvNeXt students: large gains on OOD and dense tasks vs supervised ConvNeXts, especially at higher input resolutions (Tab. 15)

  - Text alignment (Tab. 16):
    > DINOv3-based `dino.txt` improves open-vocabulary segmentation notably (ADE20k 24.7, Cityscapes 36.9), competitive zero-shot classification/retrieval for its size

  - Geospatial transfer (Tabs. 17–19; Fig. 18–19):
    > Canopy height mapping SOTA MAE: Sat 7B = 2.2 (val), 3.2 (test); Open-Canopy MAE 2.02 (best) (Tab. 17)  
    > GEO-Bench: web 7B model tops mean scores across classification and segmentation even though it uses only RGB (Tab. 18)  
    > LoveDA/iSAID/DIOR: web 7B sets new or near-best marks with frozen backbone and standard decoders (Tab. 19)

- Ablations, diagnostics, and robustness checks
  - Data curation ablation (Tab. 1): mixed curation is best overall.
  - Gram teacher ablation (Fig. 9b): early teachers (~100k–200k iters) work best; too-late teachers (1M) underperform, consistent with dense degradation.
  - Loss dynamics (Fig. 7): Gram anchoring primarily accelerates the patch-local objective (iBOT), minimal interference with global DINO.
  - Outlier analysis (Appendix A; Fig. 20): registers remove high-norm patch outliers better than attention/value-bias tricks; feature-dimension outliers are mostly neutralized by final layer norm.
  - Layer-wise utility (Appendix B.2; Fig. 21): depth/tracking/3D often peak around layer ~32, while classification/segmentation generally improve toward the last layer.

- Do the experiments support the claims?
  - Yes on dense tasks: consistent gains across linear probes, non-parametric tasks, and strong decoders indicate genuinely higher-quality local features (Fig. 2; Tabs. 3–6, 10–13).
  - Yes on global tasks: DINOv3 is competitive with leading weakly supervised models without any labels, and excels on some robustness metrics (Tab. 7–9).
  - Domain generality is credible: the satellite model and even the web model perform strongly in geospatial settings (Sec. 8; Tabs. 17–19; Fig. 18–19).

## 6. Limitations and Trade-offs
- Training cost and complexity
  - Training a 7B SSL ViT with 1M+ iterations is compute-intensive; the total project footprint is ~9M GPU hours (~2600 tCO2eq under the stated assumptions) even if a single 7B run is ~18 tCO2eq (Sec. 9; Tab. 20).
  - Although the backbone is frozen for most downstream tasks, the initial pretraining remains resource-heavy.

- Dependence on early checkpoints for Gram anchoring
  - Gram anchoring requires a suitable “early teacher” checkpoint (best around 100–200k iterations; Fig. 9b). If those are missing or if pretraining starts already “too late,” dense repair is less effective.

- Mixed results on OCR-heavy classification and specific datasets
  - On OCR-centric benchmarks, DINOv3 trails weakly supervised models trained on image–text pairs (Tab. 25). On DIODE depth (ARel), DINOv3+DPT underperforms DPT, though other metrics are strong (Tab. 12).

- Architectural choices constrain granularity
  - Patch size 16 means native token grids can still be relatively coarse at low input resolution. The work partly offsets this via high-resolution adaptation and decoders (Sec. 5.1; Sec. 6.3), but patch size is a design trade-off.

- Assumptions and scope
  - The training relies on large-scale web imagery (Instagram-derived pool with moderation) and curation heuristics; while transparent and ablated (Sec. 3.1; Tab. 1), this may not cover rare or sensitive domains.
  - Constant-schedule optimization is a design choice; while it simplifies indefinite training, it may not always be optimal for every setting or hardware profile.

## 7. Implications and Future Directions
- Field-level impact
  - Demonstrates that SSL can match or surpass weakly/supervised pipelines on dense vision while remaining highly competitive on global tasks, with a single frozen backbone (Fig. 2; Tabs. 3–7, 10–11).
  - Provides a general recipe—data curation + robust architecture + Gram anchoring + HR polishing + efficient distillation—that can become a standard for training large-scale SSL vision encoders.

- Research directions
  - Beyond patch-size limits: explore native high-resolution tokenization or multi-scale tokenization that integrates with Gram anchoring (cf. Fig. 4, 11).
  - Adaptive or learned Gram teachers: instead of fixed iteration snapshots, learn when and how to update the Gram teacher or combine multiple teachers.
  - Cross-modal SSL (not weakly supervised): use Gram-like constraints to stabilize local correspondence across modalities (e.g., video/audio), or to guide generative pretraining without text labels.
  - Robustness and fairness: the fairness table (Tab. 26) shows DINOv3 reduces some regional gaps vs DINOv2, but further work could equalize performance across income regions and cultural distributions.

- Practical applications
  - Off-the-shelf dense backbone for robotics, AR/VR, autonomous systems, and 3D mapping without fine-tuning (Tabs. 3–6, 10, 13).
  - Scientific imaging (medical, astronomical, geospatial): high-resolution dense embeddings with no labels (Sec. 8; Tabs. 17–19; Fig. 18–19).
  - Multi-task inference at the edge: distillations (ViT-S/B/L/H+, ConvNeXt T–L) retain dense/global performance under tight FLOPs/latency budgets (Fig. 16; Tabs. 14–15).
  - Open-vocabulary workflows: DINOv3-based `dino.txt` provides strong dense alignment for zero-shot segmentation and competitive global zero-shot (Tab. 16).

In short, DINOv3 contributes a principled, scalable way to keep self-supervised dense features “clean” while pushing model and data scale, enabling a single frozen encoder to underpin a broad suite of vision tasks across domains. The central mechanism—Gram anchoring—addresses a core scalability failure mode and should be broadly useful wherever local feature structure matters.
