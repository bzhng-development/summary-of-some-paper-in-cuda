# Segment Anything

**ArXiv:** [2304.02643](https://arxiv.org/abs/2304.02643)

## 🎯 Pitch

Segment Anything introduces the revolutionary Segment Anything Model (SAM), a foundation model for image segmentation that can generate high-quality object masks from a wide variety of prompts—including points, boxes, masks, and even text—without task-specific training. By leveraging a model-in-the-loop data engine, the authors created SA-1B, the largest segmentation dataset ever (1.1 billion masks over 11 million images), enabling SAM to perform impressive zero-shot transfer across numerous vision tasks. This work establishes a new paradigm for segmentation, unlocking practical, generalized, and reusable segmentation capabilities that significantly lower the barrier for computer vision applications in diverse, real-world domains.

---

## 1. Executive Summary
Segment Anything introduces a new way to do image segmentation called promptable segmentation and a model, `SAM`, that returns high‑quality masks for virtually any prompt (points, boxes, masks, and even text). Using a model‑in‑the‑loop “data engine,” it builds the largest segmentation dataset to date, `SA‑1B` (11M images, 1.1B masks), enabling strong zero‑shot transfer to many tasks (Section 1; Fig. 1).

## 2. Context and Motivation
- Problem addressed
  - Traditional segmentation models are trained for fixed label sets and tasks (semantic, instance, panoptic) and typically require task‑specific training and data. There was no “foundation” segmentation model that could generalize to new data distributions and tasks via lightweight prompting (Section 1).
  - There is no web‑scale source of segmentation masks to train such a general model; existing datasets are two to three orders of magnitude smaller (Fig. 6).
- Why it matters
  - A single, reusable segmentation component that can be prompted for many goals (e.g., segment an object from a single point or box; generate proposals; support edge detection) reduces the need for task‑specific models and labels, improving practicality for new domains where annotated data is scarce (Sections 1–2).
- Prior approaches and gaps
  - Interactive segmentation methods can refine a mask from clicks but are not designed to be composed inside larger systems, nor to always output a valid mask after any prompt—including ambiguous ones (Section 2).
  - Multi‑task segmentation systems (e.g., joint semantic/instance/panoptic) still assume the test task matches the training task set (Section 2, “Related tasks”).
  - Vision‑language pretraining (e.g., CLIP) shows zero‑shot capabilities for classification, but segmentation problems need detailed per‑pixel masks and lack abundant training masks (Section 1).
- Positioning
  - This paper proposes promptable segmentation as the pretraining task and uses it to build a general, composable segmentation component. It contributes a model (`SAM`) and the data engine that collected `SA‑1B`, enabling zero‑shot transfer across tasks and image domains (Fig. 1; Sections 2–5).

## 3. Technical Approach
The work consists of three tightly connected parts: a task, a model, and a data engine.

- Task: `promptable segmentation` (Section 2; Fig. 1a)
  - Goal: Given a prompt that specifies “what to segment,” return a valid segmentation mask.
  - Prompts can be sparse (points, boxes, text) or dense (masks). A “valid” mask means that even if the prompt is ambiguous (e.g., a point on a shirt could refer to the shirt or the person), the output should be a reasonable mask for at least one valid object (Fig. 3).
  - Why: This task encourages generality and composability; downstream tasks become “prompt engineering” problems (Section 2, “Zero‑shot transfer”).

- Model: `SAM` (Section 3; Fig. 4)
  - High‑level flow
    1. An `image encoder` computes a one‑time, high‑resolution embedding of the image.
    2. A `prompt encoder` embeds the user prompt (points/boxes/text/masks).
    3. A fast `mask decoder` fuses image and prompt embeddings to produce one or more masks and a confidence score for each.
  - Key design choices and how they work
    - Image encoder: A Vision Transformer (ViT, MAE‑pretrained) adapted for high‑res inputs (1024×1024), producing a 16× downscaled embedding (64×64×C). A 1×1 conv then 3×3 conv reduce channels to 256 with layer norms (Appendix §A “Image encoder”).
      - Rationale: Heavy computation is done once; subsequent prompts reuse the embedding, enabling amortized real‑time interaction (~50 ms per prompt on CPU in browser; Section 3, “Efficiency”).
    - Prompt encoder: 
      - Points/boxes: add a positional encoding of coordinates to a learned embedding that marks point type (foreground/background) or box corner (top‑left/bottom‑right) (Appendix §A “Prompt encoder”).
      - Dense mask prompts: downsample and embed via small convs and add element‑wise to the image embedding.
      - Text: use CLIP’s text encoder to produce a text embedding (Section 3 “Prompt encoder”).
    - Mask decoder (lightweight Transformer; Fig. 14)
      - Two decoder layers with:
        - Token self‑attention (over prompt tokens),
        - Token‑to‑image cross‑attention,
        - Per‑token MLP,
        - Image‑to‑token cross‑attention (updates image embedding with prompt information).
      - After decoding: upsample image embedding by 4× with transposed convs; an MLP maps a learned `output token` to a dynamic linear classifier that predicts the per‑pixel foreground probability by dot‑product with the upsampled embedding (Appendix §A “Lightweight mask decoder”).
      - Positional encodings are added wherever the image embedding attends; the original prompt tokens are re‑added at attention layers to keep strong geometric grounding (Appendix §A).
    - Ambiguity handling: predict multiple masks per prompt (default 3), using different `output tokens`. During training, compute loss against each candidate and backprop only the minimum (“minimum loss” or “multiple choice” training; Section 3 “Resolving ambiguity”). A small head predicts an `IoU` score (estimated overlap with the true mask) to rank the candidates.
        - Definition: `IoU` (Intersection‑over‑Union) measures overlap between prediction and ground truth: area of intersection divided by area of union. `mIoU` is the mean across examples.
    - Training objective and schedule (Section 3 “Losses and training”; Appendix §A)
      - Mask loss: focal loss + dice loss (20:1 weight) on the predicted mask logits.
      - IoU head: mean squared error between predicted IoU and true IoU of the chosen mask.
      - Interactive simulation: train in 11 “rounds” per mask—start with a point or box, then sample next points from the error region between prediction and ground truth, and feed back the previous mask logits as an additional prompt (Section 3; Appendix §A “Training algorithm”).
      - Initialization: MAE‑pretrained ViT; AdamW optimizer with learning‑rate warmup and steps; large batch and distributed training (Appendix §A “Training recipe”).

- Data engine to build `SA‑1B` (Section 4; Fig. 1c)
  - Motivation: masks aren’t available on the web at scale. The data engine iteratively improves the model while using it to accelerate or automate labeling.
  - Three stages (Sections 4; B for details):
    1. Assisted‑manual: professional annotators click foreground/background points; SAM runs in‑browser at ~50 ms/prompt using precomputed embeddings. Over iterations, SAM is retrained on collected masks; annotation time dropped from 34 s to 14 s per mask; 4.3M masks on 120k images were collected (Section 4 “Assisted‑manual stage”).
    2. Semi‑automatic: a generic box detector (trained on stage‑1 masks as “object”) pre‑fills confident masks; annotators focus on missing/difficult objects; 5.9M additional masks on 180k images were collected (Section 4 “Semi‑automatic stage”).
    3. Fully automatic: SAM is prompted with a 32×32 grid of points (also overlapping zoomed crops) and returns multiple masks per point; stable, confident masks are kept; duplicates are filtered with `NMS` (non‑maximum suppression removes overlapping duplicates). This produced 1.1B masks across 11M images (Sections 4 “Fully automatic stage”; B “Cropping, Filtering, Postprocessing”).
       - “Stable mask” heuristic: keep a mask only if thresholding its probability map at 0.5−δ and 0.5+δ yields very similar binary masks (IoU ≥ 95.0; Appendix §B).
       - Confidence filter: keep masks whose predicted IoU ≥ 88.0 (Appendix §B).
       - NMS threshold: 0.7 within and across crops (Appendix §B).

## 4. Key Insights and Innovations
- Promptable segmentation as a pretraining target (Section 2)
  - Novelty: train the model to respond to arbitrary prompts with a valid mask, rather than to a fixed task definition or label set.
  - Significance: makes the model composable—downstream tasks are solved by prompting rather than retraining (e.g., “box → instance mask,” “single point → object,” “grid of points → segment everything”).
- Ambiguity‑aware multi‑mask prediction (Section 3 “Resolving ambiguity”; Fig. 3)
  - Novelty: instead of forcing a single answer to an ambiguous prompt, predict a small set of plausible masks and rank them with an IoU head.
  - Significance: boosts single‑point performance and realism; with an “oracle” choice among the three, SAM surpasses prior interactive methods across all 23 datasets (Fig. 9a).
- Lightweight, bidirectional decoder with amortized computation (Section 3; Fig. 4, Fig. 14)
  - Novelty: cross‑attention in both directions (token→image and image→token) to fuse prompts and image features; heavy encoding computed once; prompt processing is fast (~50 ms on CPU).
  - Significance: supports real‑time interactive use and scalable, automated mask generation in browsers and pipelines.
- Scalable data engine and SA‑1B (Sections 4–5; Figs. 2, 5–7)
  - Novelty: three‑stage, model‑in‑the‑loop pipeline that ends in fully automatic, ambiguity‑aware mask generation with quality filters and NMS.
  - Significance: `SA‑1B` offers 1.1B high‑quality masks on 11M images—about 400× more masks than previous largest dataset (Open Images; Fig. 6 legend). Human verification shows 94% of automatic masks have ≥90% IoU with professionally corrected versions (Section 5 “Mask quality”).
- Text‑prompting without text supervision via CLIP alignment (Section 7.5)
  - Novelty: during training, replace text with CLIP image embeddings of the masked region; at inference, feed CLIP text embeddings instead (both are aligned by CLIP).
  - Significance: enables early “text‑to‑mask” capability without assembling text annotations (Fig. 12).

## 5. Experimental Analysis
- Evaluation setup (Sections 7; D)
  - Datasets
    - A suite of 23 segmentation datasets spanning many domains (egocentric, underwater, X‑ray, aerial, simulation, etc.; Fig. 8; Table 7).
    - SA‑1B analysis includes geographic distribution and representation (Fig. 7; Table 1).
  - Metrics (defined where needed)
    - `mIoU`: mean IoU across objects.
    - `AP`: average precision for instance segmentation (COCO, LVIS).
    - `AR@1000`: average recall using 1000 proposals per image (LVIS; object proposals).
    - Edge detection: ODS, OIS, AP, R50 on BSDS500 (Section 7.2; Table 3). ODS/OIS are dataset-/image‑level F‑score optima; `R50` is recall at 50% precision.
  - Baselines
    - Interactive segmentation: RITM, FocalClick, SimpleClick (Section 7.1; D.1).
    - Instance segmentation and proposals: ViTDet‑H with Cascade Mask R‑CNN (Sections 7.3–7.4).
  - Human ratings protocol
    - Professional annotators rate single masks on a 1–10 quality scale using multi‑view panels, with guidance and QA (Appendix §E, Figs. 19–20). Rating distributions shown in Fig. 11 and Fig. 18; statistical significance in Table 8.

- Main results
  - Single‑point promptable segmentation across 23 datasets (Section 7.1)
    - Quantitative (Fig. 9a):
      - SAM beats RITM on 16/23 datasets with the model’s top‑ranked mask. With an oracle selecting the best of SAM’s three masks, it outperforms RITM on all 23 datasets.
    - Human quality ratings (Fig. 9b; Table 8):
      - On seven diverse datasets, SAM’s masks receive substantially higher ratings than RITM. The differences are statistically significant:
        > Table 8 shows p‑values ≤ 7e‑26 across datasets and 99% confidence intervals for mean score differences that exclude zero (e.g., on LVIS v0.5, improvement CI99 is (1.40, 1.84)).
    - More points (Figs. 9c–9d):
      - With 1–9 points, SAM dominates at 1 point and remains competitive as points increase; gaps shrink as the task becomes easier. Under random point sampling, SAM’s advantage grows further (Fig. 9d).
  - Zero‑shot edge detection (Section 7.2; Fig. 10; Table 3)
    - Pipeline: regular point grid → many masks → Sobel gradients of mask probability maps → edge NMS (D.2).
    - Performance on BSDS500:
      > Table 3: SAM ODS .768, OIS .786, AP .794, R50 .928.
      - Not SOTA vs. trained edge detectors (e.g., EDTER ODS .840), but far ahead of classical zero‑shot edges (e.g., Canny ODS .600).
  - Zero‑shot object proposals (Section 7.3; Table 4)
    - Setup: generate masks with a dense point grid and NMS; evaluate AR@1000 on LVIS v1 (D.3).
    - Results:
      > Table 4: SAM overall AR@1000 59.3 vs. ViTDet‑H 63.0; for medium objects 81.6 vs. 80.8 (SAM higher); for rare/common categories 65.8/63.9 vs. 58.3/63.3 (SAM higher); it trails on small objects (45.5 vs. 51.7) and frequent categories (59.1 vs. 63.1).
      - Multi‑mask ability matters: a single‑output ablation drops overall AR to 54.9.
  - Zero‑shot instance segmentation via boxes (Section 7.4; Table 5; Fig. 16–11)
    - Setup: use ViTDet boxes as prompts; SAM predicts the mask (D.4).
    - AP:
      > Table 5: On COCO, SAM 46.5 AP vs. ViTDet‑H 51.0; on LVIS v1, SAM 44.7 vs. 46.6.
    - Human ratings on LVIS boxes (Fig. 11):
      > SAM mean 8.1 ± 0.07 vs. ViTDet‑H 7.9 ± 0.08; LVIS GT 8.6 ± 0.06; COCO GT 7.6 ± 0.12.
      - Interpretation: SAM’s masks are often visually sharper and more faithful (Fig. 16), while ViTDet benefits from training on dataset‑specific mask biases (e.g., LVIS polygon rules).
  - Text‑to‑mask (Section 7.5; Fig. 12; D.5)
    - Training trick: Replace text with CLIP image embeddings of the target region; at inference, use CLIP text embeddings (both are aligned).
    - Qualitative results show success on simple phrases (“a wheel”) and more nuanced ones (“beaver tooth grille”); when text alone is ambiguous, adding a point disambiguates (Fig. 12).
  - Ablations and scaling (Section 7.6; Fig. 13)
    - Data engine stages: using only fully automatic masks is within ~0.5 mIoU of using all stages (Fig. 13 left), simplifying training.
    - Data volume: training with ~1M images (~10% of SA‑1B, ≈100M masks) reaches performance comparable to full 11M (Fig. 13 middle); 0.1M images degrades notably.
    - Model scale: ViT‑H improves substantially over ViT‑B; gains from ViT‑L to ViT‑H are marginal (Fig. 13 right).
  - Responsible AI analysis (Section 6; Table 1; Table 2; Fig. 7; C)
    - Dataset geography and income (Table 1; Fig. 7):
      > SA‑1B has higher proportions from Europe and Asia & Oceania and more middle‑income countries vs. COCO/Open Images; all regions have ≥28M masks.
    - People segmentation fairness (Table 2):
      > mIoU with 1 point is similar across perceived gender presentation (feminine 54.4, masculine 55.7) and across skin tones (range ~51.5–56.7); with 3 points, all groups are ~90–92 mIoU.
    - Clothing segmentation shows a gap by perceived gender presentation at 1 point (masculine higher), which narrows at 3 points (Appendix Table 6).

- Do the experiments support the claims?
  - The 23‑dataset study with both automatic metrics and human ratings (Figs. 9a–b) strongly supports the core claim that SAM returns valid, high‑quality masks from minimal prompts and transfers zero‑shot across domains. The proposal and edge experiments demonstrate composability beyond the training task. Human studies explain AP gaps by dataset annotation biases (Fig. 11 with Table 5).

## 6. Limitations and Trade-offs
- Assumptions and scope (Section 8 “Limitations”)
  - `Promptable segmentation` promises a valid mask for a prompt; it does not enforce semantic consistency across an image (e.g., panoptic labeling) nor provide closed‑vocabulary semantics unless combined with other modules.
  - Designed for generality and speed of prompting, not for maximizing IoU after many clicks; specialized interactive segmenters can surpass SAM in that regime (Section 8; Figs. 9c–d).
- Quality limitations (Section 8)
  - Misses fine structures and can hallucinate small disconnected components; boundaries are not as crisp as “zoom‑in” methods that spend computation per region (e.g., FocalClick; see also Fig. 16 examples).
- Computational constraints
  - The image encoder is heavy (ViT‑H); while prompt processing is ~50 ms on CPU (Section 3 “Efficiency”), end‑to‑end performance depends on the one‑time image embedding.
- Text prompting
  - The text‑to‑mask pathway is preliminary; it relies on CLIP alignment and shows qualitative promise but lacks rigorous quantitative evaluation (Section 7.5).
- Dataset biases
  - Despite geographic improvements over prior datasets, regions like Africa and low‑income countries remain underrepresented relative to Europe and North America (Table 1). Clothing segmentation shows a gender‑presentation gap at 1 point (Appendix Table 6).
- Automatic mask generation heuristics
  - Fully automatic stage depends on stability thresholds, IoU prediction accuracy, and NMS settings (Appendix §B). While human audits are strong (94% ≥90% IoU; Section 5), heuristic choices may still bias which masks are kept.

## 7. Implications and Future Directions
- How this changes the landscape
  - Establishes segmentation as a `promptable` capability akin to language prompting and CLIP zero‑shot classification—one reusable model becomes a versatile component across tasks (Section 2 Discussion; Section 8 “Compositionality”).
  - SA‑1B provides a new pretraining substrate for segmentation research at scale (Section 5). This can catalyze new foundation models that go beyond SAM.
- Follow‑up research directions
  - Richer prompts and interfaces: combine text, gaze, gestures, and multiple context masks to reduce ambiguity; better multi‑mask ranking and uncertainty estimation.
  - Toward semantic/panoptic outputs via prompting: design prompt sets or composition strategies that induce consistent scene‑level labeling without retraining.
  - Improved fine‑structure and boundary quality: integrate efficient “zoom‑in” refinement while keeping amortized cost low (Section 8).
  - Stronger text‑to‑mask: explicit language grounding, multi‑modal pretraining with paired text–mask supervision, or improved use of CLIP‑like alignment (Section 7.5).
  - Fairness and coverage: targeted data enrichment for underrepresented regions/attributes; task‑specific audits when composing SAM into larger systems (Section 6).
- Practical applications
  - Rapid dataset labeling and bootstrapping segmentation in new domains (Section 4; fully automatic stage).
  - Plug‑and‑play component in perception stacks: instance segmentation from boxes, edge maps, and object proposals (Sections 7.2–7.4).
  - Human‑in‑the‑loop tools in browsers with real‑time feedback; accessibility tools where users indicate objects by pointing, speaking, or looking (Section 3 “Efficiency”, Section 8 “Compositionality”).

In short, Segment Anything reframes segmentation as a general prompting problem and demonstrates that a single, ambiguity‑aware model trained on SA‑1B can transfer zero‑shot to a wide range of tasks and distributions, with strong human‑perceived quality. The method’s scalability and composability are its primary strengths; crispness on fine structures, text grounding, and certain fairness/coverage aspects remain open areas for improvement.
