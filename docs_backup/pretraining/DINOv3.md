# DINOv3

**ArXiv:** [2508.10104](https://arxiv.org/abs/2508.10104)
**Authors:** Oriane Siméoni, Huy V. Vo, Maximilian Seitzer, Federico Baldassarre, Maxime Oquab, Cijo Jose, Vasil Khalidov, Marc Szafraniec, Seungeun Yi, Michaël Ramamonjisoa, Francisco Massa, Daniel Haziza, Luca Wehrstedt, Jianyuan Wang, Timothée Darcet, Théo Moutakanni, Leonel Sentana, Claire Roberts, Andrea Vedaldi, Jamie Tolan, John Brandt, Camille Couprie, Julien Mairal, Hervé Jégou, Patrick Labatut, Piotr Bojanowski
**Institutions:** Meta AI Research, WRI, Inria (France)

## 🎯 Pitch

DINOv3 pioneers a self-supervised learning approach that scales a Vision Transformer to 7 billion parameters, achieving unprecedented global and dense feature recognition without labeled data. By introducing Gram anchoring to stabilize dense feature maps, it enables a single model to outperform specialized systems across tasks like detection and segmentation, reshaping the foundation of scalable visual encoders.

---

## 1. Executive Summary
DINOv3 is a self-supervised “vision foundation model” that scales training to a 7B-parameter Vision Transformer and, for the first time, simultaneously achieves state-of-the-art global recognition and best-in-class dense (pixel-wise) features without any labeled pretraining (Secs. 1, 3–6). Its core innovation, Gram anchoring, prevents the well-known degradation of dense feature maps during long training and unlocks scaling benefits, enabling frozen backbones to outperform specialized systems on detection, segmentation, depth, tracking, and 3D correspondence (Secs. 4, 5, 6).

## 2. Context and Motivation
- Problem addressed
  - Self-supervised learning (SSL) can train on “raw” image collections at web scale, but two practical issues block progress at large model sizes and long training:
    1) How to curate useful unlabeled data; and
    2) How to prevent dense features (patch-level representations) from becoming noisy and semantically unstable as training continues (Sec. 1, Fig. 5–6).
  - In large models and long schedules (beyond ViT-L), dense features degrade: patch similarities get noisy and less localized, hurting tasks like segmentation and depth, even while global classification keeps improving (Sec. 4.1; Fig. 5a–c, Fig. 6).

- Why it matters
  - SSL promises “generalist” visual encoders robust across domains (natural, medical, aerial), decoupling representation learning from expensive human annotations (Sec. 1, Fig. 1). High-quality dense features are essential for classical vision (detection, segmentation, depth), 3D geometry, video, robotics, and downstream multimodal systems (Sec. 2, “Dense Transformer Features”).

- Prior approaches and gaps
  - Weakly supervised image–text training (e.g., CLIP derivatives) excels at global tasks but often produces inferior dense features unless aided by heavy supervision or distillation from mask-based models (AM-RADIO, PEspatial; Sec. 2, “Agglomerative methods”).
  - Prior SSL at scale (e.g., DINOv2) matched CLIP-like models on many global benchmarks but hit a ceiling on dense feature quality when moving to larger models/long schedules (Secs. 1, 3.2, 4.1).
  - Remedies like register tokens help with high-norm patch outliers (Darcet et al., 2024), but do not stop the long-run drift of patch similarity structure (App. A.1 vs. Sec. 4.1).

- Positioning
  - DINOv3 modernizes the SSL pipeline along four axes: scalable unlabeled data curation, a stable large-model training recipe, a new Gram anchoring phase that preserves patch-level structure, and post-hoc polishing (high-resolution adaptation, single-teacher/multi-student distillation, and text alignment). It aims to be a single frozen backbone that’s strong on both global and dense tasks (Secs. 3–5).

## 3. Technical Approach
Step-by-step overview (with “how it works” and why these choices were made).

1) Data preparation at scale (Sec. 3.1)
- Source pool: ~17B public Instagram images after platform-level moderation.
- Curation strategy combines two complementary automatic pipelines:
  - Clustering-balanced sampling (hierarchical k-means over DINOv2 embeddings; 5 levels up to 200M leaf clusters) yields LVD-1689M, a balanced coverage of web concepts (Sec. 3.1 “Data Collection and Curation”).
  - Retrieval-based curation: pull images similar to “seed” datasets to bias toward useful downstream concepts (as in DINOv2).
  - Add a small supervised “spice”: mix in public datasets (ImageNet-1k/22k, Mapillary SLS) for practical coverage.
- Sampling during training: alternate homogeneous ImageNet1k-only batches (10% of steps) with heterogeneous batches from the curated pool—motivated by better optimization from homogeneous high-quality bursts (Sec. 3.1 “Data Sampling”).
- Why: The two curation methods excel on different benchmarks; combining them wins across the board (Tab. 1).

2) Base SSL training recipe (Sec. 3.2)
- Objective composition (`L_Pre` in Eq. (1)):
  - `L_DINO`: global, discriminative alignment across augmented views (DINO-family).
  - `L_iBOT`: local, patch-level latent reconstruction to preserve spatial detail.
  - `L_Koleo`: “spreading” regularizer that encourages features in a batch to be well-distributed (helps stability).
  - Both DINO and iBOT use Sinkhorn-Knopp centering (from SwAV) instead of DINO centering; this stabilizes training and tweaks balance of global/local signals (Sec. 3.2 “Learning Objective”).
  - Dedicated heads and LayerNorms for global vs local crops allow mild specialization without forking the backbone (Sec. 3.2).
- Architecture changes (Tab. 2):
  - Scale to a 7B-parameter ViT with 40 blocks, patch size 16 (longer spatial field), 4096-dim tokens, 32 heads, SwiGLU FFNs. Crucially, keep 4 `register tokens` to soak up communications and reduce patch-norm outliers (Sec. 3.2; App. A.1).
  - Use `axial RoPE` (rotary positional encoding) with “box jittering” (randomly scale the coordinate box) to improve robustness to resolution, scale, and aspect ratio (Sec. 3.2).
- Optimization:
  - Constant hyperparameter schedules (LR, weight decay, teacher-EMA) after warmup; no cosine decays. This eliminates the need to predetermine training length and supports “train-as-long-as-useful” (Sec. 3.2 “Optimization”).
  - Multi-crop with 2 global (256²) and 8 local (112²) crops; total sequence length ~3.7M tokens per batch (256 GPUs; Sec. 3.2).

3) Diagnosing and fixing dense-feature degradation (Sec. 4)
- Observed phenomenon (Sec. 4.1):
  - As training continues, classification keeps rising, but dense-task performance declines (Fig. 5b–c). Cosine similarity maps around a reference patch become noisy and overly global, indicating loss of patch-level locality/consistency (Fig. 6).
  - Cosine similarity between CLS and patch tokens increases late in training (Fig. 5a), correlating with worse segmentation.
- Gram anchoring (Sec. 4.2; Eq. (2), (3)):
  - Key idea: preserve the structure of pairwise patch similarities (not the raw features) by matching the student’s `Gram matrix` of normalized patch features (`X_S X_S^T`) to that of a “Gram teacher” taken from an early, more local-consistent checkpoint (`X_G X_G^T`).
  - Why Gram? The Gram matrix compresses “who is similar to whom” among patches; forcing this structure to persist curbs drift toward globally entangled, less local representations while allowing features themselves to keep improving.
  - Implementation: After ~1M iterations of base SSL, start a “refinement” phase with `L_Ref = w_D L_DINO + L_iBOT + w_DK L_Koleo + w_Gram L_Gram` (Eq. (3)). Update the Gram teacher every 10k steps by copying the main EMA teacher (Fig. 7 tracks loss behavior).
  - Immediate effect: Dense metrics jump within ~10k steps; ADE20k and VOC steadily improve, while global benchmarks remain stable or slightly improve (Fig. 8).

4) High-resolution Gram anchoring (“HR Gram”) (Sec. 4.3)
- Observation: Teachers produce cleaner local structure at higher input resolution; downsampling those features preserves the improved patch-consistency at the original size (Fig. 9a).
- Method: Run the teacher at 2× resolution, bicubic downsample the teacher’s patch features back to student resolution, and compute Gram loss against this smoothed target (Sec. 4.3). This yields additional gains (e.g., +2 mIoU on ADE vs standard Gram; Fig. 8; Tab. 9b).

5) Post-training polish (Sec. 5)
- High-resolution adaptation (Sec. 5.1):
  - Short mixed-resolution training (~10k iters), sampling global crops from {512, 768} and local crops from {112,168,224,336}; include Gram anchoring using the 7B teacher.
  - Outcome: Much better dense performance at high resolutions; stable feature maps even above 4k inputs (Fig. 4). Quantitatively: improved ADE20k with resolution; better DAVIS J&F at larger sizes (Fig. 11).
- Distillation to a model family (Sec. 5.2; Fig. 16a; Tab. 14–15):
  - Single fixed teacher (the 7B model) trains multiple students (ViT S/B/L/H+, plus ConvNeXt T/S/B/L).
  - Efficient multi-student distillation (Fig. 12): share teacher inference once per iteration across all GPUs; then partition GPUs into student groups for parallel student updates. This reduces total compute and yields a whole family in one run.
  - No Gram needed for students: the fixed teacher already has strong patch consistency.
- Text alignment with `dino.txt` (Sec. 5.3; Tab. 16):
  - Freeze the visual encoder; train a text encoder via LiT-style contrastive alignment. Concatenate pooled patch embeddings with CLS before matching to text to encourage alignment of both global and local semantics.

## 4. Key Insights and Innovations
1) Gram anchoring to preserve patch similarity structure (Sec. 4; Eq. (2), (3))
- Novelty: Regularize the student to retain the teacher’s intra-image patch-pair similarity matrix (Gram), rather than matching raw patch features. This directly targets the symptom of long-run dense degradation—noisy, over-global patch similarities (Fig. 6).
- Why it’s significant:
  - Immediately reverses dense metrics’ decline late in training (Fig. 8).
  - Enables training “as long as helpful” for global metrics without sacrificing dense performance—a key unlock for scaling SSL.
  - HR Gram (Sec. 4.3) further lifts dense performance by transferring the teacher’s higher-res smoothness (Fig. 9).

2) A stable large-model SSL training regime (Sec. 3.2)
- What’s new: Constant schedules (post-warmup), axial RoPE with jitter for resolution robustness, specific head/LN separation for global/local crops, and continued use of registers (4) to mitigate patch-norm outliers (App. A.1).
- Impact: ViT-7B training becomes predictable, obviating the need to guess an “optimization horizon” (Sec. 3.2), and providing a base for the Gram refinement and HR adaptation phases.

3) End-to-end scaling and polishing pipeline (Secs. 3–5)
- Integrated sequence: Curated unlabeled data mix → 7B SSL with mixed global/local losses → Gram refinement → HR adaptation → multi-student distillation → optional text alignment.
- Significance: Delivers a suite of frozen backbones that beat specialized pipelines on dense tasks and rival or surpass weakly supervised models on global tasks (Figs. 2, 11; Tabs. 3–16).

4) Practical innovations: multi-student distillation and ConvNeXt students (Sec. 5.2)
- Multi-student distillation shares expensive teacher inference across all nodes, making it efficient to produce a whole family at once (Fig. 12).
- Cross-architecture distillation to ConvNeXt yields efficient backbones that clearly outperform supervised ConvNeXt across OOD classification and dense tasks (Tab. 15).

## 5. Experimental Analysis
- Evaluation scope (Sec. 6)
  - Dense features (frozen): linear segmentation and depth; 3D keypoint matching; unsupervised object discovery; video segmentation tracking; video classification via attentive probe.
  - Global features (frozen): linear classification on IN-1k and OOD variants; instance retrieval.
  - As backbones for decoders (mostly frozen backbones): object detection (Plain-DETR), semantic segmentation (Mask2Former + ViT-Adapter), depth estimation (DPT within Depth-Anything v2), 3D geometry (VGGT).
  - Baselines: strongest SSL (DINOv2, Franca, Web-DINO), weakly supervised (SigLIP 2, PE), agglomerative (AM-RADIO, PEspatial), supervised giants (ViT-22B references), video JEPA for video classification.
  - Metrics: mIoU (segmentation), RMSE/ARel/δ1 (depth), recall (correspondence), J&F (tracking), top-1 acc (classification), mAP/GAP (retrieval), COCO mAP and COCO-O ER (detection).

- Main dense results (frozen backbones):
  - Linear segmentation (Tab. 3):
    > ADE20k mIoU: DINOv3 55.9 vs DINOv2 49.5; vs AM-RADIO 53.0; vs PEspatial 49.3  
    > Cityscapes: 81.1 (best; +2.5 over AM-RADIO)  
    > VOC: 86.6 (best)
  - Linear depth (Tab. 3):
    > NYUv2 RMSE: 0.309 (best; vs DINOv2 0.372; vs AM-RADIO 0.340)  
    > KITTI RMSE: 2.346 (best; vs DINOv2 2.624)
  - 3D keypoint matching (Tab. 4):
    > NAVI (geometric) recall: 64.4 (best; +4.3 over DINOv2)  
    > SPair (semantic) recall: 58.7 (best; +2.6 over DINOv2)
  - Unsupervised object discovery (Fig. 14):
    > VOC07 CorLoc: 66.1 (best; +5.0 over DINO S/16; +10.5 over AM-RADIO)  
    > COCO-20k CorLoc: 55.1 (best)
  - Video segmentation tracking (Tab. 5):
    > DAVIS J&F, Large: 83.3 (best; +6.7 vs DINOv2)  
    > Gains scale with resolution (S→M→L), unlike PEspatial which degrades at high res.

- Main global results (frozen backbones):
  - Classification (Tab. 7):
    > IN-1k Val 88.4; V2 81.4; ReaL 90.4—competitive with SigLIP2/PE; best mCE on ImageNet-C (19.6).  
    > On hard OOD (ImageNet-A, ObjectNet), DINOv3 trails PE slightly but beats SigLIP2 on ObjectNet (79.0 vs 78.6).
  - Fine-grained (Tab. 8):
    > iNat2021: 89.8 (best among all compared; +2.8 over PEcore).
  - Instance retrieval (Tabs. 9, 23):
    > Oxford-Hard mAP 60.7 (best), Paris-Hard mAP 87.1 (best), Met GAP 55.4 (best; +10.8 vs DINOv2), AmsterTime 56.5 mAP (best; +7.6 vs DINOv2).

- As frozen backbones in downstream systems:
  - Object detection (Tab. 10; Plain-DETR on top of frozen 7B):
    > COCO mAP: 66.1 TTA (best among listed; surpasses EVA-02 Co-DETR and PEspatial-DETA);  
    > COCO-O ER: 36.8 (best robustness).
  - Semantic segmentation (Tab. 11; ViT-Adapter + Mask2Former, frozen backbone):
    > ADE20k: 63.0 mIoU TTA (ties or beats ONE-PEACE/InternImage-H/BEIT3), with far fewer trainable params (decoder-only).  
    > Also best or on-par on COCO-Stuff/VOC and competitive on Cityscapes (Tab. 24).
  - Relative monocular depth (Tab. 12; DPT head; frozen backbone):
    > New SOTA across NYU, KITTI, ETH3D, ScanNet; DIODE competitive.  
    > Example: NYUv2 ARel 4.3 (best), δ1 98.0; KITTI ARel 7.3, δ1 96.7.
  - 3D geometry (VGGT) (Tab. 13; swap DINOv2 → DINOv3 ViT-L):
    > Re10K/CO3Dv2 camera pose AUC@30: 86.3/89.6 (best);  
    > DTU multi-view depth “Overall”: 0.368 (best);  
    > ScanNet-1500 matching AUC@10: 56.1 (best).

- Distilled model families (Sec. 7; Fig. 16; Tab. 14–15)
  - ViT-S/B/L/H+: Clear Pareto across compute. Dense tasks: large margins vs DINOv2/SigLIP2/PE of similar size (e.g., ViT-L ADE20k 54.9 mIoU; ObjectNet 74.8; Oxford-Hard 63.1; Tab. 14).
  - ViT-H+ is very close to the 7B teacher on multiple tasks (Fig. 16b).
  - ConvNeXt T/S/B/L: strong OOD and dense wins over supervised ConvNeXt-22k counterparts (Tab. 15).

- Text alignment (Tab. 16)
  - DINOv3 dino.txt (on ViT-L) improves over DINOv2 dino.txt, competitive on global zero-shot (behind PE/SigLIP2) and notably stronger on open-vocabulary segmentation (ADE20k 24.7 mIoU, Cityscapes 36.9, both best in table).

- Geospatial transfer (Sec. 8; Tabs. 17–19; Fig. 18–19)
  - Train a 7B satellite DINOv3 on 493M Maxar RGB tiles; distill to ViT-L.
  - Canopy height mapping: new SOTA on SatLidar Val/Test and Open-Canopy (Tab. 17), with a frozen DPT head.  
  - GEO-Bench (classification/segmentation): RGB-only DINOv3 (satellite or web) surpasses prior multi-band EO foundation models on most tasks (Tab. 18).  
  - High-res benchmarks: new SOTA on LoveDA and DIOR; near-SOTA on iSAID (Tab. 19).

- Ablations and diagnostics
  - Data curation: mixed curation > clustering-only/retrieval-only/raw (Tab. 1).
  - Gram anchoring: rapid dense gains; iBOT loss drops faster; DINO losses unaffected (Fig. 7–9).
  - High-resolution adaptation: improvements scale with resolution for dense tasks; stable features even at 4k+ (Secs. 5.1; Figs. 4, 11).
  - Outliers: registers mitigate high-norm patch tokens; channel-dimension outliers handled by final LayerNorm/batch-norm at inference (App. A).
  - Layer-wise probes: geometry/temporal consistency often peaks slightly before the last layer (App. B.2, Fig. 21).

- Overall assessment
  - The experiments are broad, deep, and ablated. The central claims—state-of-the-art dense features from a purely self-supervised model and a frozen backbone competitive with specialized or weakly supervised systems—are compellingly supported by quantitative and qualitative evidence across many tasks and domains (Secs. 6–8; Figs. 2–4, 13; Tabs. 3–19).

## 6. Limitations and Trade-offs
- Compute and data scale
  - Training a 7B SSL model remains expensive (Tab. 20: ~47 MWh per full pretraining; total project ~2,600 tCO2eq with 9M GPU hours estimate). Despite efficient distillation, the frontier backbone is heavy at inference (Fig. 16a).
- Heavy reliance on curated web-scale data
  - Although self-supervised, strong performance still hinges on sophisticated data curation (Sec. 3.1), including a non-trivial mixture with ImageNet1k batches (10%).
- Dense-vs-global objective balance still needs intervention
  - The base SSL losses alone (DINO + iBOT + Koleo) allow global metrics to dominate at scale; Gram anchoring is an extra phase (and a new hyperparameter regime) to restore local structure (Sec. 4). Choosing teacher checkpoints and HR factors introduces design latitude (Fig. 9b).
- Text-heavy/OCR tasks lag weakly supervised models
  - On OCR-oriented classification (street signs, logos, products), DINOv3 improves over DINOv2 but still trails PEcore by a wide margin (Tab. 25), reflecting the advantage of image–text supervision for glyph semantics.
- Fairness and geographic performance gaps remain
  - While improved over DINOv2, performance still varies across income buckets and regions (e.g., lower in low-income/Africa vs high-income/Europe; Tab. 26). The training set composition and cultural/visual biases likely contribute and warrant dedicated mitigation.
- Feature outliers at the channel dimension
  - At large scales, certain feature dimensions can dominate activations across layers and grow during training (App. A.2). The final LayerNorm/batch-norm usage is recommended; using intermediate layers may require extra normalization care.

## 7. Implications and Future Directions
- How this changes the landscape
  - DINOv3 demonstrates that purely image-only self-supervision can yield a single frozen backbone that is SOTA or competitive across dense and global tasks—without labels or paired text during pretraining. This narrows the historical advantage of weakly supervised image–text training for general-purpose encoders and surpasses them on dense vision (Figs. 1–2, Tabs. 3–5, 10–13).
  - Gram anchoring reframes SSL stability: track and regularize the geometry of patch similarities, not just feature vectors—an idea likely useful beyond vision, e.g., in video SSL, cross-modal SSL, or even LLM token-level structure preservation.

- What it enables
  - Practical: Edge deployment of smaller distilled students (ViT-S/B/L/H+, ConvNeXt T/S/B/L) that retain strong dense features; frozen-backbone decoders for detection/segmentation/depth with reduced compute and memory budgets (Secs. 5.2, 6.3).
  - Scientific: High-resolution, high-quality dense maps directly from ViTs (Figs. 3–4, 17), facilitating new pipelines in 3D vision, mapping, and robotics without heavy supervised pretraining. Remote sensing results suggest task-dependent benefits of domain-specific vs. domain-agnostic SSL (Sec. 8; Tabs. 17–19).

- Research directions
  - Automating Gram teacher selection and schedule: adaptively select “best local-consistency” checkpoints; explore multi-scale/temporal Gram for video or 3D.
  - Unifying objectives: integrate Gram-style structure preservation earlier or jointly, reducing the need for separate refinement phases.
  - Multimodal dense alignment: extend dino.txt to patch-level cross-modal grounding more explicitly; combine with segmentation-friendly supervision without inducing bias toward mask tasks.
  - Data governance and fairness: controlled studies on data mixture effects on fairness metrics (Tab. 26) and OCR-heavy skills (Tab. 25); explore targeted augmentations or lightweight multimodal adapters to close gaps.
  - Efficiency: investigate low-rank adapters or token-pruning compatible with high-res dense maps; push multi-student distillation further (e.g., cross-task teacher ensembles) while keeping compute bounded.

In short, DINOv3 offers a clear recipe—data curation + stable large-model SSL + Gram anchoring + HR adaption + efficient distillation—that turns SSL into a practical, scalable path to general-purpose, dense-capable visual backbones. It provides strong evidence that high-quality dense features and global robustness can be learned at scale without manual labels, setting a new baseline for future foundation models in vision.
