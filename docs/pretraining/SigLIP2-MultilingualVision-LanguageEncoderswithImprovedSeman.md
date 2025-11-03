# SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic Understanding, Localization, and Dense Features

**ArXiv:** [2502.14786](https://arxiv.org/abs/2502.14786)

## 🎯 Pitch

SigLIP 2 introduces a powerful new family of multilingual vision-language encoders that, for the first time, unifies sigmoid-based image-text alignment, decoder-based localization/captioning pretraining, and self-supervised local feature learning into a single open-source model suite. By doing so, SigLIP 2 raises the bar for zero-shot classification, retrieval, localization, and dense prediction tasks, while adding multi-resolution, aspect-ratio–preserving inference and dramatically reducing representation bias—directly addressing the needs of real-world applications like search, OCR, document analysis, and global multilingual use. This breakthrough empowers the next generation of vision-language models, especially as front-ends for multimodal LLMs, and sets a new standard for both performance and fairness across languages and domains.

---

## 1. Executive Summary
SigLIP 2 is a family of open, multilingual vision–language encoders that unifies three previously separate ideas—sigmoid-based image–text alignment, decoder-based caption/localization pretraining, and self-supervised local feature learning—into one training recipe. Across four model sizes, it improves zero-shot classification, retrieval, localization, and dense prediction, while adding single-checkpoint multi-resolution and aspect-ratio–preserving inference (NaFlex) and significantly reducing measured representation bias (Sec. 2; Tables 1–5; Figs. 3–6).

## 2. Context and Motivation
- Problem addressed
  - CLIP-style encoders are strong at semantic alignment (e.g., “a dog” ↔ a dog image) but lag on localization (where in the image?), dense prediction (pixel-level tasks), and multilingual robustness; open models often follow older CLIP-style recipes and miss recent advances (Sec. 1).
  - Standard fixed-square resizing distorts aspect ratios, hurting OCR, documents, UI screenshots, and other aspect-sensitive inputs (Sec. 2.4.2; Fig. 3).
  - Smaller models underperform at low compute budgets; multilingual training and fairness require care in data mixture and filtering (Sec. 2.1; Sec. 3.5).

- Why it matters
  - Real-world uses (search, retrieval, OCR, document understanding, mobile UIs, robotics) need both global semantics and local/dense understanding, often across many languages and aspect ratios.
  - These encoders are the front-end to multimodal LLMs; better encoders directly raise the ceiling of VLM performance (Sec. 3.2; Fig. 4).

- Prior approaches and gaps
  - CLIP/ALIGN contrastive training excels at alignment but lacks strong local/dense features (Sec. 1).
  - Additions from separate lines of work:
    - Re-captioning with stronger captions (e.g., CoCa, TIPS) and captioner-based pretraining with decoders (e.g., LocCa) improve OCR/localization (Sec. 1; Sec. 2.2).
    - Self-supervision (DINO-v2, SILC) improves dense features (Sec. 2.3).
    - Open-weight encoders exist (OpenCLIP, EVA-CLIP, MetaCLIP, DFN) but generally hew close to the CLIP recipe and are largely English-centric (Table 1; Sec. 1).
  - Missing unification: no single open model combined all of these improvements with multilingual support, aspect-ratio preservation, and small-model optimization.

- How this work positions itself
  - Builds on SigLIP (sigmoid loss) and integrates:
    - A decoder for captioning, grounded captioning, and referring expressions (LocCa-style),
    - Self-distillation and masked prediction for local/dense features (SILC/TIPS),
    - A NaFlex variant for native aspect ratio and variable token counts,
    - Multilingual tokenizer and data with de-biasing,
    - Active data curation to lift small models (Sec. 2).
  - Releases open checkpoints at four sizes: `ViT-B (86M), L (303M), So400m (400M), g (1B)` and remains backward-compatible with SigLIP architecture (Sec. 1; Sec. 2.1).

## 3. Technical Approach
This section unpacks “what is trained,” “how it is trained,” and “why those choices were made.”

- Architecture (Sec. 2.1)
  - Vision and text towers both use Vision Transformers (`ViT`) with attention pooling via a `MAP head` (multi-head attention pooling; it replaces a CLS token for global pooling).
  - Text length is 64 tokens with a multilingual Gemma tokenizer (256k vocab); text is lower-cased before tokenization.
  - For the largest model `g/16`, the vision encoder is paired with an So400m-sized text encoder (Sec. 2.1).

- Data and optimization (Sec. 2.1)
  - Training corpus: WebLI (10B images, 12B alt-texts) spanning 109 languages.
  - Mixture: 90% English, 10% non-English (recommended balance from prior multilingual work).
  - De-biasing: filters from [2] applied to mitigate first-order (representation imbalance) and second-order (attribute associations) biases.
  - Compute: Adam (lr 1e-3), decoupled weight decay 1e-4, gradient clip 1; batch size 32k, cosine schedule with 20k warmup, total 40B examples; trained on up to 2048 TPUv5e with fully sharded data parallelism (Sec. 2.1).

- Training step 1: Global alignment + decoder pretraining (Sec. 2.2)
  - Sigmoid loss (“SigLIP”): Instead of CLIP’s softmax over the batch, it treats each image–text pair in the batch as a binary classification (match vs non-match) with a logistic regression per pair. This avoids normalization across the entire batch and has empirically strong alignment properties (Fig. 1, “Sigmoid loss (100%)”).
  - Decoder-based pretraining (“LocCa”):
    - Attach a transformer decoder with cross-attention to the un-pooled vision tokens.
    - Train it on three tasks, all in-batch:
      - Image captioning,
      - Grounded/dense captioning (predict region-specific captions given box coordinates),
      - Referring expression comprehension (predict bounding boxes for region-descriptive captions).
    - Region–caption pairs are auto-annotated via open-vocabulary detection with n-grams and a fixed object-category set (Sec. 2.2).
    - 50% of captioning uses “parallel prediction” (non-autoregressive tokens predicted from mask tokens without causal masking) to reduce decoding compute (Sec. 2.2).
    - A chunked decoder loss reduces memory with the large 256k vocabulary.
  - Losses: Sigmoid loss and decoder loss are combined with equal weight during this stage (Fig. 1; Sec. 2.2).
  - Why: Caption/grounding tasks improve OCR/localization; sigmoid loss keeps strong global alignment.

- Training step 2 (last 20%): Local feature learning via self-distillation and masked prediction (Sec. 2.3)
  - Self-distillation (local-to-global consistency):
    - A teacher network is an exponential moving average (`EMA`) of the student parameters.
    - Student sees 8 local crops (“partial views”); teacher sees 1 global view.
    - Student matches teacher features via a separate MLP head in a high-dimensional feature space (“consistency loss”), encouraging local patches to align with the global semantics (Sec. 2.3).
  - Masked prediction:
    - Replace 50% of student’s patch embeddings with a learned mask token and train to match teacher’s per-patch features at masked locations (same loss family as above) (Sec. 2.3).
  - Scheduling and weighting:
    - Added at 80% of training to avoid early distortions of image–text alignment (data augmentations only applied to the extra views, not the alignment view) (Sec. 2.3).
    - Loss weights: 1.0 (consistency) and 0.25 (masked); then globally re-weighted by model size (B/L/So/g use multipliers 0.25/0.5/1.0/0.5) to balance global vs dense-task quality (Sec. 2.3).
  - Why: This late-stage, view-separated local learning strengthens dense features while protecting global image–text alignment.

- Adapting to different resolutions
  - Fixed-resolution checkpoints (Sec. 2.4.1)
    - At 95% training, resume from the 256-token model and resize positional embeddings to the target sequence length; sometimes also resize the patch embedding (16→14) using FlexiViT’s pseudoinverse (PI) strategy.
    - Continue training with all losses. The common “small LR, no weight decay” fine-tuning did not work consistently across sizes/resolutions (Sec. 2.4.1).
  - NaFlex: native aspect ratio, variable sequence length (Sec. 2.4.2)
    - Preprocess each image to multiples of the patch size with minimal distortion, preserving aspect ratio (like NaViT). Limit total tokens to a target `sequence length`.
    - Encode with non-square positional embeddings by bilinear-resizing the learned 2D grid to match the resized patch grid; mask out any padding tokens (Sec. 2.4.2).
    - Train from 90% completion of the default model by switching to aspect-preserving resizing and uniformly sampling sequence lengths from {128, 256, 576, 784, 1024}; stretch the last-10% schedule by 3.75× so each length gets enough updates; halve batch size and double steps for the largest length to fit memory.
    - To keep complexity manageable, NaFlex does not include the self-distillation/masked-prediction heads during this adaptation (Sec. 2.4.2).

- Boosting small models via active data curation (Sec. 2.5)
  - For `B/16` and `B/32`, continue training for 4B examples with only the sigmoid image–text loss, lr 1e-5, no weight decay.
  - Use `ACID` (Active CuratIon via Distillation): At each step, compute a “learnability” score for many candidates using the frozen teacher and current learner; select a top fraction to form the actual batch (filtering ratio 0.5; for B/32, 0.75) from a larger super-batch (Sec. 2.5).
  - Teacher choice: fine-tune a strong diversified teacher (SigLIP 2 So400m) for 1B examples on a curated dataset so “implicit distillation through data” approximates ACED’s benefits without explicit soft-label loss—saving compute (Sec. 2.5).
  - Why: Curating harder/useful examples for the learner meaningfully raises small-model quality at constant compute.

## 4. Key Insights and Innovations
- Unifying three complementary training signals in one recipe (Fig. 1; Sec. 2)
  - What’s new: A single encoder is trained with (a) sigmoid alignment, (b) a decoder for caption/grounding/referring expressions, and (c) late-stage self-distillation + masked prediction.
  - Why it matters: Each piece targets a different weakness—global alignment, localization/OCR, and dense semantics—yielding broad gains with one set of weights (Tables 1–5).

- Late-stage, view-separated local feature learning (Sec. 2.3)
  - Different from prior: Applies consistency/masking only in the final 20%, and only on additional augmented views, to avoid harming image–text alignment while still improving dense features (a known tension).
  - Impact: Large boosts on segmentation/depth/normals and localization tasks without sacrificing zero-shot retrieval/classification (Table 2; Table 5).

- NaFlex: one checkpoint, many resolutions, native aspect ratio (Sec. 2.4.2; Fig. 3; Table 7)
  - Different from prior: Combines FlexiViT-style variable token counts with NaViT-style aspect preservation in an image–text encoder, using resized learned positional embeddings and attention masking.
  - Impact: Especially strong on OCR/document/screen benchmarks at low token budgets, where square-resize distortion is most harmful (Fig. 3; Table 7).

- Multilingual encoder with de-biasing that retains English strength (Sec. 2.1; Sec. 3.1; Fig. 2; Tables 1, 8–9)
  - Different from prior: Uses a multilingual tokenizer and 90/10 EN/non-EN WebLI mixture with explicit de-biasing filters.
  - Impact: Large multilingual retrieval gains over SigLIP and near parity with multilingual SigLIP (mSigLIP) while improving English tasks; representation bias drops dramatically (Fig. 2; Table 1; Table 9).

- Active data curation for small models (Sec. 2.5; Table 1)
  - Different from prior: Reuses ACID with a specifically fine-tuned teacher to capture ACED-like benefits without explicit distillation loss.
  - Impact: B-sized models get outsized improvements, closing much of the gap at lower inference cost (Table 1, B/16 and B/32 rows).

Overall, the unification and scheduling choices are the fundamental innovations; NaFlex and active curation are impactful engineering contributions that broaden applicability and improve the cost–quality trade-off.

## 5. Experimental Analysis
- Evaluation setup
  - Zero-shot classification and retrieval (Sec. 3.1; Table 1)
    - Datasets: ImageNet-1k (val), ImageNet-v2, ImageNet ReaL, ObjectNet; COCO and Flickr retrieval; multilingual Crossmodal-3600 (XM3600) retrieval.
    - Metric: Accuracy for classification; `recall@1` for retrieval (percentage of queries where the correct match is ranked first).
  - NaFlex vs standard (Sec. 3.1.1; Fig. 3; Table 7)
    - Compare single NaFlex checkpoint vs separate fixed-resolution checkpoints across sequence lengths.
    - Add OCR/document/screen retrieval: TextCaps, HierText, SciCap, Screen2Words.
  - VLM transfer (Sec. 3.2; Fig. 4; Table 6)
    - Pair encoders with Gemma 2 (2B) LLM; Stage 1: 50M examples of a rich multi-task mix with frozen vision encoder; Stage 3: dataset-specific fine-tuning.
    - Resolutions: 224/256 tokens and 384px settings.
  - Dense prediction probing (Sec. 3.3.1; Table 2)
    - Tasks: semantic segmentation (PASCAL, ADE20k), monocular depth (NYUv2, NAVI), surface normals (NYUv2, NAVI).
    - Protocol: linear head or DPT decoder on frozen features; use MAP-pooled embedding in place of a CLS token as in [38].
  - Open-vocabulary segmentation (Sec. 3.3.2; Table 3)
    - Framework: Cat-Seg; train on COCO-Stuff-164k; test on ADE20k (847/150 classes), Pascal Context (459/59), Pascal VOC (20/21). Metric: mIoU.
  - Referring expression comprehension (Sec. 3.4.1; Table 5)
    - Attach a 6-layer cross-attention decoder to frozen vision tokens; train on all RefCOCO variants; metric: Acc@0.5 (IoU threshold).
  - Open-vocabulary detection (Sec. 3.4.2; Table 4)
    - OWL-ViT fine-tuning for COCO and LVIS; metrics: AP (and rare-category AP on LVIS).
  - Cultural diversity and fairness (Sec. 3.5; Fig. 5–6; Tables 8–9)
    - Cultural: Dollar Street, GeoDE, GLDv2; report 0-shot and 10-shot accuracies (geographical diversity, geolocalization).
    - Fairness: “Representation bias” (tendency to associate random objects with one gender) and “disparity” (max difference in accuracy across Dollar Street income groups).

- Main results (highlights with grounded references)
  - Zero-shot and retrieval (Table 1)
    - `B/16, 256 tokens`: ImageNet val 79.1% vs 76.7% (SigLIP), ReaL 85.4 vs 83.1; COCO R@1 T→I 53.2 vs 47.4; XM3600 R@1 T→I 40.7 vs 22.5.
    - `L/16, 256`: ImageNet val 82.5 vs 80.5; ObjectNet 78.8 vs 76.8; XM3600 T→I 46.5 vs 30.9.
    - Across sizes and resolutions, SigLIP 2 outperforms OpenCLIP, MetaCLIP, EVA-CLIP, and DFN in most settings despite being multilingual.
  - Multilingual retrieval (Fig. 2; Table 1)
    - Per-language XM3600: SigLIP 2 nearly matches multilingual SigLIP (mSigLIP) while greatly exceeding English-centric SigLIP, e.g., average XM3600 R@1 T→I improves from 22.5 (SigLIP B/16-256) to 40.7 (SigLIP 2), approaching mSigLIP’s 50.0 at So/16-256 (Fig. 2; Table 1).
  - NaFlex vs standard (Fig. 3; Table 7)
    - On OCR/document/screen retrieval at small sequence lengths, NaFlex is stronger (e.g., B/16 TextCaps R@1 at 256 tokens: 19.7/17.1 I→T/T→I for NaFlex vs 17.1/14.2 standard; Table 7).
    - On natural-image classification/retrieval, standard B-sized checkpoints often edge out NaFlex (likely due to the extra self-distillation stage; Fig. 3). NaFlex “interpolates fairly well” between trained lengths but “does not extrapolate well” to unseen lengths (Fig. 3 caption).
  - VLM transfer (Fig. 4; Table 6)
    - With Gemma 2 (2B), SigLIP 2 beats SigLIP and AIMv2 across model sizes and resolutions.
    - Examples at So400m/14, 384px: TextVQA val +4.3 points (69.7→74.0), ST-VQA +2.3 (75.0→77.3), DocVQA +3.2 (62.7→65.9), SciCap +2.1 (177.2→179.3); RefCOCO testA +1.6 (76.6→78.2). Many tasks show consistent but smaller gains (Fig. 4; Table 6).
  - Dense prediction probing (Table 2)
    - `So/14, 384px`: PASCAL mIoU 78.1 vs 73.8 (SigLIP); ADE20k mIoU 45.4 vs 40.8; NYUv2 depth RMSE 0.466 vs 0.563 (lower is better); normals RMSE 23.0 vs 24.1. Even `So/14, 224px` shows strong gains.
  - Open-vocabulary segmentation (Table 3)
    - `L/16`: On ADE20k-150, 38.8 mIoU vs 37.5 (SigLIP) and 36.2 (OpenCLIP G/14). Gains also on Pascal Context and VOC.
  - Referring expression comprehension (Table 5)
    - Huge jumps at all sizes: `B/16, 256 tokens` RefCOCO val 83.76 vs 64.05; `L/16, 256` 86.04 vs 67.33; `So/14, 256` 86.42 vs 64.68. Only LocCa (English-only training) slightly exceeds SigLIP 2 in some splits.
  - Open-vocabulary detection (Table 4)
    - OWL-ViT fine-tuning: `B/16` COCO AP 42.8 vs 42.2; LVIS AP 34.4 vs 33.0; rare categories 32.7 vs 31.0. `So/14` COCO 45.2 vs 44.3; LVIS 40.5 vs 39.5; rare 42.3 vs 40.9.
  - Cultural diversity and fairness (Fig. 5–6; Tables 8–9)
    - 10-shot geolocalization on GeoDE (region): `L/16, 256` 44.4% vs 36.2% (SigLIP). 0-shot Dollar Street: 55.2% vs 52.1%; GLDv2: 64.5% vs 56.7% (Table 8; Fig. 5).
    - Representation bias drops sharply: `L/16, 256` 7.3% vs 35.5% (lower is better), with larger models generally fairer (Table 9; Fig. 6).
    - Income/region disparity reductions are modest (Table 9 and Sec. 3.5).

- Do the experiments support the claims?
  - Breadth and consistency: Strong, spanning global alignment (Table 1), multilingual (Fig. 2), local/dense (Tables 2–3, 5), detection (Table 4), VLM transfer (Fig. 4), and fairness (Fig. 6; Table 9).
  - Component attribution: While the full recipe clearly outperforms baselines, there are limited explicit ablations isolating the contribution of each new loss (e.g., turning off decoder or self-distillation). The timing (late-stage) and view separation are design choices substantiated by results on dense/local tasks without alignment regressions.

- Robustness and trade-offs visible in results
  - NaFlex trades a bit of performance on natural images at B-size (vs standard) but wins on OCR/screen with fewer tokens; it interpolates well but not beyond trained lengths (Fig. 3; Table 7).
  - Referring expressions: SigLIP 2 is very strong but English-only LocCa is slightly ahead in some splits (Table 5), suggesting multilingual training may slightly dilute maximal English-only specialization on this task.

## 6. Limitations and Trade-offs
- Compute and reproducibility
  - Pretraining on 40B examples with up to 2048 TPUv5e and large batch FSDP is resource-intensive (Sec. 2.1). Small-model improvements add 4B more examples (Sec. 2.5).
- Component attribution
  - The paper does not provide fine-grained ablations for each added loss and scheduling choice across all tasks, making it hard to quantify per-component gains beyond qualitative rationale (Sec. 2).
- NaFlex caveats (Sec. 3.1.1; Fig. 3)
  - No self-distillation/masked prediction during NaFlex adaptation (engineering simplification) may leave performance on natural images slightly behind standard B-size checkpoints.
  - Does not extrapolate well to untrained sequence lengths.
- Decoder not shipped for inference
  - The decoder is only used during pretraining; tasks that fully exploit its cross-attention (e.g., referring expressions) may further improve if the pretrained decoder were retained (Table 5 discussion).
- Multilingual scope and cultural fairness
  - Despite improvements, SigLIP 2 slightly trails mSigLIP on some multilingual retrieval languages (Fig. 2). Disparity by income/region shows only modest reductions (Table 9).
- Data mixture choices
  - The 90/10 EN/non-EN split is a compromise; performance on truly low-resource languages or scripts not well represented in WebLI is not deeply examined.

## 7. Implications and Future Directions
- Field impact
  - SigLIP 2 provides a new “default” open encoder for multimodal systems: stronger global alignment, better dense/local features, multilingual coverage, fairness improvements, and practical NaFlex inference.
  - For VLM builders, Fig. 4/Table 6 indicate that simply swapping the encoder lifts many downstream tasks without changing LLM or training recipes.

- Practical applications
  - Cross-lingual retrieval/search; OCR-heavy tasks (TextVQA, documents, UI); open-vocabulary detection/segmentation for robotics and AR; geolocalization and diverse-object recognition; fairer and more culturally inclusive systems (Tables 1–4, 6–9; Figs. 3–6).
  - NaFlex checkpoints are especially attractive in production settings where images vary widely in aspect ratio and compute budgets fluctuate.

- Research directions
  - Joint training of NaFlex with self-distillation/masking to close the small natural-image gap while retaining OCR/screen gains (Sec. 2.4.2; Fig. 3).
  - Detailed ablations to quantify the contribution and optimal scheduling/weighting of decoder vs self-distillation vs masked prediction, and their interactions.
  - Multilingual data mixtures beyond 90/10, targeted augmentation for low-resource scripts, and fairness evaluations across additional sensitive attributes and benchmarks (Sec. 3.5).
  - Retaining or lightweightly adapting the pretrained decoder for localization tasks at inference time (Table 5 suggests headroom vs LocCa).
  - Extending active data curation beyond small models, or online curation that adapts to task-specific domains.

> Bottom line (Sec. 5): SigLIP 2 demonstrates that carefully combining sigmoid alignment, decoder-based grounding, and late-stage local feature learning—plus pragmatic advances in aspect-ratio handling, multilingual tokenization, and data curation—produces a single family of encoders that advance the state of the art across classification, retrieval, VLM transfer, localization, dense prediction, and fairness (Tables 1–6; Figs. 2–6).
