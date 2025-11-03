# SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic Understanding, Localization, and Dense Features

**ArXiv:** [2502.14786](https://arxiv.org/abs/2502.14786)
**Authors:** Michael Tschannen, Alexey Gritsenko, Xiao Wang, Muhammad Ferjad Naeem, Ibrahim Alabdulmohsin, Nikhil Parthasarathy, Talfan Evans, Lucas Beyer, Ye Xia, Basil Mustafa, Olivier Hénaff, Jeremiah Harmsen, Andreas Steiner, Xiaohua Zhai
**Institutions:** Google Research / DeepMind

## 🎯 Pitch

SigLIP 2 revolutionizes vision-language encoding by unifying decoder-based pretraining, self-distillation, and active data curation to create a backward-compatible model that excels in zero-shot classification, localization, and multilingual tasks. This enhancement is crucial for real-world applications like OCR and robotics, offering a fairer, more inclusive approach to global and spatial understanding across multiple languages and contexts.

---

## 1. Executive Summary
SigLIP 2 is a family of multilingual vision–language encoders that unifies several previously separate techniques—captioning-based pretraining with a decoder, self-distillation with masked prediction, and active data curation—on top of the original SigLIP training objective. The result is a drop-in, backward-compatible replacement for SigLIP that substantially improves zero-shot classification and retrieval, while also fixing long-standing weaknesses of CLIP-style models in localization and dense prediction; it also enhances multilinguality and fairness (Sec. 1; Fig. 1; Sec. 3).

## 2. Context and Motivation
- Problem/gap addressed
  - CLIP-style image–text encoders are strong for global semantics (e.g., zero-shot classification, retrieval) but lag on tasks that require spatial grounding and dense features such as segmentation, depth estimation, or referring expressions. They also often underperform in multilingual settings and exhibit societal biases (Sec. 1; Sec. 3.3–3.4; Table 2; Table 5).
  - Prior open-weight releases largely “track” the original CLIP recipe and do not incorporate the full breadth of recent improvements in a single model (Sec. 1).
- Why it matters
  - Real-world applications increasingly need both global understanding and spatial grounding (e.g., OCR, UI and document understanding, robotics). Multilingual capability and fairness are essential for equitable deployment across cultures and languages (Sec. 1; Sec. 3.5).
- Prior approaches and their limitations
  - Contrastive training (CLIP/ALIGN) enabled zero-shot performance but offered limited localization/dense features (Sec. 1; Sec. 4).
  - Individual improvements existed:
    - Re-captioning datasets to improve text quality (e.g., TIPS) (Sec. 1; Sec. 4).
    - Image-only self-supervised losses (SILC, DINO-style) to enrich local features (Sec. 1; Sec. 4).
    - Auxiliary decoder tasks (captioning, localization) to improve grounding/OCR (LocCa) (Sec. 1; Sec. 4).
    - Open-source models exist but typically lack the combination of all three (Sec. 1).
- How this work positions itself
  - SigLIP 2 combines all three strands into a single, scalable recipe while keeping the original SigLIP architecture for backward compatibility. It introduces variable-aspect, variable-resolution “NaFlex” variants and curates training data online for stronger small models (Sec. 1; Sec. 2; Sec. 2.4; Sec. 2.5).

## 3. Technical Approach
This is a staged recipe that starts with SigLIP’s sigmoid loss and incrementally adds decoder-based pretraining, self-distillation with masked prediction, resolution/aspect adaptations, and active data curation (Sec. 2; Fig. 1).

Key definitions used below:
- `Sigmoid loss` vs. contrastive: SigLIP turns each image–text pair in the batch into independent binary classification problems (match vs. non-match) rather than a single softmax across the batch (Sec. 2.2). This reduces interactions across negatives and can stabilize training at scale.
- `Self-distillation (teacher–student)`: a moving-average “teacher” network provides targets for a “student” to match, improving feature quality without labels (Sec. 2.3).
- `Masked prediction`: predict features for masked image patches by matching the teacher’s features, encouraging local, spatially aware representations (Sec. 2.3).
- `MAP head` (multi-head attention pooling): an attention-based pooling layer used instead of a [CLS] token to aggregate patch features to an image-level representation (Sec. 2.1).
- `NaFlex`: a variant that preserves images’ native aspect ratios and supports multiple sequence lengths with one checkpoint, combining NaViT-like aspect handling with FlexiViT-like variable token lengths (Sec. 2.4.2).
- `ACID` (Active data curation via implicit distillation): selects the most “learnable” examples online using a teacher to maximize the utility of each batch, improving small models without explicit soft-target losses (Sec. 2.5).

Step-by-step recipe

1) Architecture, data, and optimizer (Sec. 2.1)
- Architecture
  - Both image and text towers are ViTs with learned positional embeddings; pooled by a MAP head. Vision `g` pairs with an So400m-sized text encoder; others use same size for image/text (Sec. 2.1).
  - Tokenization: multilingual Gemma tokenizer (256k vocab, lowercased), text length 64 (Sec. 2.1).
- Data
  - WebLI: 10B images and 12B alt-texts in 109 languages. Mixture: 90% English, 10% non-English to balance English and multilingual performance (Sec. 2.1).
  - Debiasing filters from prior work are applied to reduce representation and association biases (Sec. 2.1).
- Optimization and scale
  - Adam, lr 1e-3, decoupled weight decay 1e-4, grad clip 1.0, cosine schedule with 20k warmup for a total of 40B examples; batch size 32k; up to 2048 TPUv5e chips, trained with fully sharded data parallel (FSDP) (Sec. 2.1).

2) Stage A — Sigmoid image–text loss + decoder-based pretraining (Sec. 2.2; Fig. 1)
- Keep SigLIP’s sigmoid loss for image–text alignment (Sec. 2.2).
- Add a transformer `decoder` with cross-attention to unpooled vision features for auxiliary tasks:
  - `Captioning`: optionally with “parallel prediction” 50% of the time (predict all tokens from masks without causal mask), which speeds learning of global semantics (Sec. 2.2).
  - `Referring expression comprehension`: predict bounding boxes for localized phrases (Sec. 2.2).
  - `Grounded (dense) captioning`: generate region-specific captions given bounding boxes (Sec. 2.2).
- Where do the region–text pairs come from?
  - Automatically: n-grams from alt-text plus open-vocabulary detection; additionally use a fixed object category set from WebLI (Sec. 2.2).
- Practicalities
  - Decoder has half as many layers as the text encoder and uses cross-attention to the image tokens (Sec. 2.2).
  - Decoder loss is “chunked” to reduce memory given the 256k vocab (Sec. 2.2).
  - Image patch size 16, base resolution 256 (thus 16×16=256 tokens) during the main pretraining (Sec. 2.2).
  - Decoder is used only for pretraining; it is not part of the released checkpoints (Sec. 2.2).

3) Stage B — Self-distillation and masked prediction in the last 20% of training (Sec. 2.3; Fig. 1)
- Timing: added at 80% of training; teacher is initialized as an EMA copy of the student at that point (Sec. 2.3).
- Two auxiliary consistency losses (applied on additional augmented views to avoid hurting alignment):
  - `Local-to-global`: student sees eight cropped “local” views; teacher sees one “global” full view. An MLP “projection head” defines the space for matching pooled features (Sec. 2.3).
  - `Masked prediction`: replace 50% of student patch embeddings with a mask token and match the teacher’s features at masked locations, operating at the per-patch level (Sec. 2.3).
- Loss weights
  - Base weights: 1.0 for local-to-global, 0.25 for masked prediction, then reweighted by model size to balance global vs. dense-task quality: B:×0.25, L:×0.5, So400m:×1.0, g:×0.5 (Sec. 2.3).

4) Stage C — Resolution and aspect-ratio adaptation (Sec. 2.4)
- Fixed-resolution variants (Sec. 2.4.1)
  - At 95% training, resize positional embeddings to the target sequence length and continue training at that resolution with all losses.
  - When changing patch size (e.g., 16→14), use pseudoinverse (PI) resizing of the patch embedding (Sec. 2.4.1).
- NaFlex variants (variable aspect and resolution; one checkpoint supports multiple token lengths) (Sec. 2.4.2)
  - At 90% training, switch to aspect-preserving preprocessing that minimally distorts width/height to be multiples of the patch size, capping the token count at the target sequence length (Sec. 2.4.2).
  - Positional embeddings are bilinearly resized to non-square patch grids; attention uses masks to ignore padded tokens when the resized image produces fewer tokens than the target (Sec. 2.4.2).
  - Train by uniformly sampling sequence lengths per mini-batch from {128, 256, 576, 784, 1024}; stretch the remaining schedule by 3.75× so each length sees sufficient examples; halve batch size and double steps at the largest sequence length to avoid OOM (Sec. 2.4.2).
  - To keep complexity manageable, this stage does not include self-distillation and masked prediction (Sec. 2.4.2).

5) Stage D — Active data curation for small models (B/32, B/16) (Sec. 2.5)
- Fine-tune for 4B additional examples with the sigmoid loss at lr 1e-5 and no weight decay; perform online data selection with `ACID` (Sec. 2.5).
- How ACID works
  - Score a large “super-batch” (e.g., 64k examples) each step with a strong teacher and the current student by “learnability,” then select the top-scoring subset to form the actual training batch (Sec. 2.5).
  - Filtering ratio: 0.5 for B/16; 0.75 for B/32 (Sec. 2.5).
  - Teacher: a single So400m SigLIP 2 model fine-tuned for 1B examples on a high-quality curated dataset, blending diverse pretraining knowledge with curated quality. This recovers benefits of two-teacher ACED without explicit soft-target distillation (Sec. 2.5).

What is released
- Open-weight checkpoints at four sizes: `ViT-B` (86M), `ViT-L` (303M), `So400m` (400M), and `g` (1B), including fixed-resolution and NaFlex variants; backward compatible with SigLIP (Sec. 1).

## 4. Key Insights and Innovations
- Unified training recipe that blends three complementary ideas (Fig. 1; Sec. 2)
  - Novelty: Prior open models typically adopted one or two of these ideas; SigLIP 2 integrates `decoder-based pretraining (LocCa) + self-distillation with masked prediction (SILC/TIPS) + active data curation (ACID)` on top of SigLIP’s sigmoid loss. This yields global alignment, spatial grounding, and stronger local features concurrently.
  - Significance: Translates into consistent gains in zero-shot classification/retrieval and large jumps on localization/dense tasks (Table 1; Table 2; Table 5).
- NaFlex: native aspect ratio + variable sequence length with one checkpoint (Sec. 2.4.2; Fig. 3; Appendix Table 7)
  - What’s new: Combines NaViT-style aspect handling with FlexiViT-style multi-length training and masked attention to support multiple resolutions in a single model.
  - Why it matters: Reduces aspect distortion harm at lower resolutions and improves OCR/document/screen retrieval tasks; simplifies deployment by avoiding a zoo of resolution-specific checkpoints (Fig. 3).
- Backward compatibility with SigLIP architecture (Sec. 1; Sec. 2.1)
  - What’s new: Users can swap weights and tokenizer and immediately benefit from improvements without changing downstream pipelines.
  - Why it matters: Greatly lowers adoption barriers and accelerates impact.
- Fairness and multilinguality baked into pretraining (Sec. 2.1; Sec. 3.5)
  - What’s new: A deliberate language mix (90% English, 10% non-English) and de-biasing filters reduce representation bias and improve multilingual retrieval nearly to the level of fully multilingual SigLIP (mSigLIP) while far surpassing it on English-centric tasks (Fig. 2; Sec. 3.1; Sec. 3.5).
  - Why it matters: More culturally inclusive encoders with better global utility.

## 5. Experimental Analysis
Evaluation design
- Tasks and datasets span:
  - Zero-shot classification and retrieval: ImageNet-1k, ImageNet-v2, ReaL, ObjectNet; COCO/Flickr retrieval; multilingual Crossmodal-3600 (XM3600) (Table 1; Fig. 2).
  - VLM transfer: freeze the vision encoder, train a Gemma 2 (2B) LLM on 50M multimodal examples (Stage 1 from PaliGemma 2), then fine-tune per-task—benchmarks include VQA, OCR, referring expressions, captioning, counting, science, screens, etc. (Sec. 3.2; Fig. 4; Appendix Table 6).
  - Dense prediction probing: segmentation, depth, surface normals with linear/DPT heads (Table 2).
  - Open-vocabulary segmentation and detection: Cat-Seg on COCO-Stuff → ADE/Pascal/VOC; OWL-ViT fine-tuning on COCO/LVIS (Table 3; Table 4).
  - Localization: referring expression comprehension (three RefCOCO variants) using a freshly trained 6-layer decoder head on frozen encoders (Table 5; Sec. 3.4.1).
  - Cultural diversity and fairness: Dollar Street, GeoDE, GLDv2; representation bias and disparity metrics (Sec. 3.5; Fig. 5; Fig. 6; Appendix Tables 8–9).
- Baselines include CLIP/OpenCLIP, MetaCLIP, EVA-CLIP, DFN, mSigLIP, and AIMv2 (Table 1; Fig. 4; Table 3).

Main quantitative results and comparisons
- Zero-shot classification and retrieval (Table 1)
  - Improvements across sizes and resolutions. For example:
    - B/16 at 256 tokens: 
      > “SigLIP 2 79.1 vs. SigLIP 76.7 on ImageNet; COCO R@1 T→I 53.2 vs. 47.4; I→T 69.7 vs. 65.1” (Table 1).
    - L/16 at 256 tokens:
      > “SigLIP 2 82.5 vs. SigLIP 80.5 on ImageNet; COCO R@1 T→I 54.7 vs. 51.2; I→T 71.5 vs. 69.6” (Table 1).
    - So400m/14 at 384 tokens:
      > “SigLIP 2 84.1 vs. SigLIP 83.2 on ImageNet; COCO R@1 T→I 55.8 vs. 52.0; I→T 71.7 vs. 70.2” (Table 1).
  - Multilingual retrieval on XM3600:
    > SigLIP 2 “almost matches” mSigLIP per-language recall while outperforming SigLIP by a large margin (Fig. 2; Table 1, XM3600).
- NaFlex vs. standard (Fig. 3; Appendix Table 7)
  - NaFlex tends to outperform at short sequence lengths and on aspect-sensitive OCR/screen/document benchmarks (TextCaps, HierText, SciCap, Screen2Words), while the fixed-resolution variants can lead on natural-image benchmarks at small model sizes—likely because NaFlex excludes the self-distillation stage (Sec. 2.4.2; Fig. 3).
- VLM integration (Fig. 4; Appendix Table 6)
  - With Gemma 2 (2B) and a frozen vision encoder, SigLIP 2 improves across a broad set of tasks and resolutions.
  - Example highlights from Fig. 4 (So400m/14 at 384px):
    > “RefCOCO (testA): 78.2 vs. 76.6 (SigLIP 2 vs. SigLIP); TextVQA (val): 74.0 vs. 69.7; DocVQA (val): 65.9 vs. 62.7; COCOcap: 143.8 vs. 142.2” (Fig. 4; Appendix Table 6).
- Dense prediction probing (Table 2)
  - Significant gains:
    > “PASCAL mIoU at 384px: SigLIP 2 So/14 = 78.1 vs. SigLIP So/14 = 73.8; ADE20k mIoU 45.4 vs. 40.8; NYUv2 depth RMSE 0.466 vs. 0.563 (lower better)” (Table 2).
- Open-vocabulary segmentation/detection (Table 3; Table 4)
  - Open-vocab segmentation (Cat-Seg) at L/16:
    > “ADE-150 mIoU: 38.8 (SigLIP 2) vs. 37.5 (SigLIP) vs. 36.2 (OpenCLIP G/14)” (Table 3).
  - OWL-ViT detection:
    > “LVIS rare AP: B/16 → 32.7 vs. 31.0; So/14 → 42.3 vs. 40.9 (SigLIP 2 vs. SigLIP)” (Table 4).
- Localization: referring expression comprehension (Table 5)
  - Large jumps across all sizes/resolutions; e.g., B/16 at 256 tokens:
    > “RefCOCO val: 83.76 vs. 64.05; RefCOCO+ val: 74.26 vs. 55.77; RefCOCOg val: 77.25 vs. 59.06 (SigLIP 2 vs. SigLIP)” (Table 5).
  - SigLIP 2 trails only LocCa, which is English-only; SigLIP 2 trains multilingual and still nearly matches (Table 5).
- Cultural diversity and fairness (Sec. 3.5; Fig. 5; Fig. 6; Appendix Table 8–9)
  - Geodiverse performance:
    > “L/16 256px GeoDE (region) 10-shot: 44.4 vs. 36.2; Dollar Street 0-shot: 55.2 vs. 52.1 (SigLIP 2 vs. SigLIP)” (Appendix Table 8).
  - Representation bias (lower is better):
    > “L/16 256px: 7.3% (SigLIP 2) vs. 35.5% (SigLIP)” (Fig. 6; Appendix Table 9).

Do the experiments support the claims?
- Breadth and consistency: The paper evaluates across global, multilingual, spatial, dense, and downstream VLM tasks with consistent gains, especially where CLIP-style models were weak (Table 1–5; Fig. 2–4).
- Component-wise attribution: While the staged recipe is well motivated (Sec. 2), detailed ablations isolating each component’s contribution are limited in the main text. The outsized improvements on localization strongly align with decoder-based pretraining (Sec. 2.2; Table 5), and dense prediction gains align with self-distillation/masked prediction (Sec. 2.3; Table 2).
- Trade-offs revealed: NaFlex improves aspect-sensitive tasks but without self-distillation can trail fixed-resolution models on natural images, especially at small sizes (Fig. 3; Sec. 2.4.2). This nuance strengthens the paper’s credibility.

## 6. Limitations and Trade-offs
- Compute and data scale
  - Training uses 40B examples, batch size 32k, and up to 2048 TPUv5e chips (Sec. 2.1). Reproducing training from scratch is resource-intensive; however, open checkpoints mitigate this for users.
- Decoder not released
  - The decoder used during pretraining is not part of the release (Sec. 2.2). Users get the encoder benefits but cannot directly reuse the pretraining decoder heads without re-implementing them for new training.
- NaFlex excludes self-distillation
  - To manage complexity, NaFlex training omits self-distillation and masked prediction (Sec. 2.4.2). This likely explains some performance gaps on natural-image benchmarks at lower token counts (Fig. 3).
- Language mix and per-language ceiling
  - The 90/10 English/non-English mixture (Sec. 2.1) balances performance, but Fig. 2 shows SigLIP 2 still slightly trails fully multilingual mSigLIP on some languages—suggesting trade-offs between English-centric and broad multilingual optimization (Fig. 2; Table 1, XM3600).
- Limited ablations in the main text
  - The paper provides strong end-to-end gains but offers few controlled ablations quantifying each component’s isolated contribution (e.g., decoder-only vs. self-distillation-only vs. both).
- Fairness: partial progress
  - Representation bias drops dramatically (Fig. 6; Appendix Table 9), but disparities across income levels and geographic regions show only minor or mixed improvements (Sec. 3.5; Appendix Table 9). The precise impact of each debiasing filter is not ablated.

## 7. Implications and Future Directions
- Field impact
  - SigLIP 2 sets a new default for open image–text encoders: go beyond contrastive alignment to include decoder pretraining and self-distillation—yielding both global semantics and spatial/dense features. This directly benefits VLMs, open-vocabulary perception, and OCR/document pipelines (Sec. 3.2–3.4; Fig. 4; Table 2–5).
- Practical applications
  - Drop-in upgrade for CLIP/SigLIP-based systems with better zero-shot and retrieval (Table 1).
  - Substantially better localization (referring expressions), OCR-heavy tasks, and dense perception—relevant to AR/VR, robotics, UI automation, and enterprise document understanding (Fig. 3; Table 2; Table 5).
  - Multilingual deployments with improved fairness characteristics (Sec. 3.5; Fig. 5–6).
- Recommended follow-ups
  - Combine NaFlex with self-distillation/masked prediction to close the remaining gap on natural-image tasks while keeping aspect benefits (Sec. 2.4.2; Fig. 3).
  - Detailed ablations to quantify each component’s contribution and to optimize the weighting schedule across model sizes (Sec. 2.3).
  - Explore broader or adaptive language mixtures and debiasing strategies to further reduce disparities across income/region while maintaining strong English benchmarks (Sec. 3.5; Appendix Table 9).
  - Evaluate decoder reuse in downstream fine-tuning (e.g., initializing a detection/REC decoder from pretraining) to probe how much extra head capacity is needed after SigLIP 2 encoding.
  - Efficiency research: distillation and curation strategies that bring SigLIP 2-like performance to even smaller or edge-deployable models (Sec. 2.5).

Overall, SigLIP 2 is a carefully engineered, unified recipe that produces an open, multilingual, fairer, and spatially aware vision–language encoder. Its methodological choices map cleanly to the observed gains: decoder tasks improve grounding; self-distillation and masking strengthen dense features; active data curation unlocks small-model performance; and NaFlex simplifies deployment while boosting aspect-sensitive workloads (Fig. 1–6; Tables 1–5; Appendix Tables 6–9).
