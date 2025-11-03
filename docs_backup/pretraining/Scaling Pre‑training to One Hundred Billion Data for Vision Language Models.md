# Scaling Pre‑training to One Hundred Billion Data for Vision Language Models

**ArXiv:** [2502.07617](https://arxiv.org/abs/2502.07617)
**Authors:** Xiao Wang, Ibrahim Alabdulmohsin, Daniel Salz, Zhe Li, Keran Rong, Xiaohua Zhai
**Institutions:** Google DeepMind

## 🎯 Pitch

This paper reveals a significant shift in vision-language model benefits at an unprecedented scale of 100 billion image-text pairs from the web, demonstrating that while classic benchmarks saturate, there is a marked improvement in cultural inclusivity and multilingual coverage. This finding is crucial for developing equitable AI systems, emphasizing the need to prioritize data diversity and low-resource language support over traditional Western-centric metrics.

---

## 1. Executive Summary
This paper studies what changes when you pre-train vision–language models (VLMs) on an unprecedented scale of 100 billion image–text pairs drawn from the raw web. The main finding is counterintuitive: classic “Western‑centric” benchmarks (e.g., ImageNet zero‑shot and COCO retrieval) largely saturate at this scale, but diversity-focused capabilities—cultural coverage, low‑resource languages, and cross‑group parity—improve markedly (Sections 4.1–4.4; Tables 2–5; Figure 1).

## 2. Context and Motivation
- Problem addressed
  - Whether pushing VLM pre‑training data from ≈10B to 100B unique image–text pairs yields meaningful benefits, and in what dimensions those benefits appear.
- Why this matters
  - VLMs underpin retrieval, captioning, and multimodal assistants. If additional web‑scale data only yields diminishing returns on standard benchmarks, resources could be better spent elsewhere. Conversely, if scaling uniquely improves inclusivity (e.g., cultural and linguistic coverage), it is vital for building equitable systems (Abstract; Figure 1; Sections 1 and 7).
- Prior landscape and gaps
  - Dataset growth: from curated millions (Conceptual Captions) to web‑scale billions (LAION‑5B, WebLI‑10B) using filters such as CLIP to “improve quality” (Section 2; citations [59, 15, 60]).
  - Scaling laws: error often follows a power law with data size, implying diminishing but non‑zero returns; however, effects at 100B for VLMs were unknown (Section 1; scaling law references).
  - Inclusivity concerns: filtering and English‑centric pipelines can suppress cultural diversity and multilingual coverage (Section 2; e.g., [53]).
- Positioning of this work
  - Introduces `WebLI‑100B` (100B unique pairs) with minimal filtering and evaluates models in a compute‑matched regime across 1B, 10B, and 100B scales (Sections 3.1–3.2).
  - Broadens evaluation beyond traditional metrics to cultural diversity, multilinguality, and fairness (Section 3.3), showing where the 100B scale pays off (Sections 4.2–4.4).

## 3. Technical Approach
This is an empirical scale‑up study. The core question—what changes at 100B data?—is isolated by holding training compute roughly fixed across data scales and by evaluating many capability axes.

- Data construction
  - `WebLI‑100B`: 100B image–text pairs scraped from the web; minimally filtered (remove harmful images and PII) to preserve breadth of languages and cultures; use both `alt-text` and page `title` as paired text; remove near‑duplicates with overlap to >90 evaluation tasks to avoid leakage (Section 3.1, Raw Datasets).
  - Subsets: `1B` and `10B` are uniform random samples of `WebLI‑100B` (1% and 10%) (Section 3.1).
  - Language attribution: use the page’s `content-language` HTML meta tag rather than noisy on‑the‑fly language detection (Section 3.1).
  - Quality‑filtered sets for analysis: three 5B‑pair English datasets—(i) “CLIP‑filtered” using a `CLIP-L/14` alignment score, (ii) “Classifier‑filtered” using a VLM trained to predict alignment, and (iii) “Baseline (en)” by sampling English pairs without filtering (Section 3.1; Figure 4; Appendix D).
  - Language rebalancing (for a specific study): upsample 7 low‑resource languages in the `Crossmodal-3600` benchmark (bn, fil, hi, iw/he, mi, sw, te) to 1% each of training examples, with the remaining 93% drawn from the original mix (Section 3.1 Language‑rebalanced Datasets; Section 5.2; Appendix F lists language shares).

- Models and training
  - VLM type: `SigLIP` contrastive models with ViT backbones (`ViT-B/16`, `ViT-L/16`, `ViT-H/14`) for both image and text encoders (Section 3.2).
    - Contrastive learning aligns matched image–text pairs in a shared embedding space and pushes apart mismatched pairs; `SigLIP` uses a sigmoid loss rather than softmax/InfoNCE (reference [78]).
  - Compute‑matched protocol:
    - Fix the total number of seen examples to 100B for every condition. Therefore, `1B` data runs 100 epochs, `10B` runs 10 epochs, and `100B` runs 1 epoch (Section 3.2: “All models are trained on a maximum of 100 billion examples”).
    - Batch size 32K; inverse square‑root learning rate schedule with 200M warmup and cooldown examples; LR 0.001; weight decay 1e‑4 (Section 3.2).
    - Inputs: images resized to 224×224; texts tokenized with multilingual `mt5` tokenizer up to 64 tokens (Section 3.2).
    - Periodic checkpoints when models have seen 3, 7, 10, 17, 26, 33, 49, 66, and 100B examples (Section 3.2).
  - Transfer to generative VLMs:
    - Initialize `PaliGemma` (a compact, instruction‑tunable VLM) with these vision encoders; pretrain on 50M seen examples following its Stage‑1 recipe at 224×224. Evaluate two scenarios: vision frozen vs unfrozen during PaliGemma pretraining/finetuning (Section 3.3 “Transfer to Generative Models”; Table 6; Appendix C).

- Evaluations and metrics (Section 3.3)
  - Western‑centric tasks:
    - Zero‑shot classification: ImageNet, CIFAR‑100, Oxford‑IIIT Pets (Table 2).
    - 10‑shot classification: CUB‑Birds, Caltech‑101, Cars196, Colorectal Histology, DTD (Table 2).
    - Zero‑shot retrieval: COCO Captions and Flickr30k (image→text and text→image) (Table 2).
  - Cultural diversity:
    - Zero‑shot: Dollar Street (DS), GeoDE, Google Landmarks v2 (GLDv2).
    - 10‑shot geolocalization: DS and GeoDE (Table 3).
    - “Geolocalization” here means predicting an image’s country/region category with few labeled examples per class.
  - Multilinguality:
    - `Crossmodal‑3600`: 3600 images with human captions in 36 languages; measure zero‑shot retrieval (image→text and text→image) per language. Report averages for low‑resource vs high‑resource groups (Section 3.3; Figure 3; Appendix B).
  - Fairness:
    - Representation bias (RB): tendency to associate random images with label “Male” vs “Female” (values >50% mean a male preference). Values reported as the percentage of times “Male” wins (Table 4).
    - Association bias (AB): for pairs of occupation labels (e.g., “secretary” vs “manager”), measure how often a gendered image steers the model to specific occupations using `FairFace` images (Figure 2; Section 4.4).
    - Performance disparity: maximum gap across subgroups—by income level on DS (four income bins) and by region on GeoDE (Africa, Americas, East Asia, Europe, SE Asia, West Asia) (Table 5).
  - Statistics and scaling fits:
    - Use Wilcoxon signed‑rank tests to compare conditions (Sections 4.1 and 4.2).
    - Fit power‑law scaling laws for error vs data size and report exponents and asymptotic error limits (Tables 2–3), following [2].

- Qualitative analysis
  - Attention‑map visualizations show where models focus in images across scales (Tables 1 and 7). These illustrate better localization of culturally specific elements at 100B.

## 4. Key Insights and Innovations
1) Scaling to 100B yields little on classic benchmarks but large gains in inclusivity axes
- What is new: Prior work emphasized better classic scores with better filtering and moderate scale-ups. This study isolates dataset size (1B→10B→100B) under fixed compute and shows a split outcome across task families (Sections 4.1–4.3).
- Evidence:
  - Western benchmarks saturate. For example, `ViT-L/16` ImageNet zero‑shot error changes 29.7%→28.5% going 10B→100B, a modest 1.2‑point drop; COCO I2T@1 error increases 47.2%→45.3% (Table 2). A signed‑rank test yields p=0.9 (Section 4.1).
  - Cultural diversity improves considerably. Example: Dollar Street 10‑shot error for `ViT-L/16` improves 64.1%→58.3% (−5.8 points), and for `ViT-H/14` 59.1%→53.7% (−5.4) when scaling 10B→100B; improvements are statistically significant (p=0.002) (Table 3; Section 4.2).
  - Low‑resource languages benefit more. Figure 3 shows larger error reductions for low‑resource languages than for high‑resource ones across all model sizes (Section 4.3).

2) Filtering improves classic scores but harms cultural and fairness metrics
- What is new: A systematic, side‑by‑side comparison of CLIP‑based filtering vs raw English sampling vs a classifier‑based filter, all at the same 5B size (Section 5.1; Figure 4; Appendix D).
- Evidence:
  - Western tasks: CLIP filter shows consistent error reductions (e.g., ImageNet 0‑shot error at 30B seen examples: 23.9% CLIP vs 24.3% baseline‑en; Table in Appendix D).
  - Cultural diversity and fairness: all filtered sets perform worse. Figure 4 (middle/right) shows higher error on cultural tasks and fairness aggregates across training trajectories; Table 10 details per‑benchmark degradations (Appendix D).

3) Language rebalancing is a cheap, targeted fix for low‑resource languages
- What is new: Upsampling seven low‑resource languages to 1% each yields substantial retrieval error drops for those languages, with minor regressions on high‑resource languages; overall multilingual average improves (Section 5.2; Figure 5; Table 11).
- Evidence:
  - Low‑resource average error decreases markedly after rebalancing across data scales (Figure 5, top‑left). For example, at `ViT-L/16` and 100B seen examples, low‑resource average goes from 75.01% to 70.10% (Table 11, “Average Multilingual: Low‑Resource Lang”).

4) Bias persistence vs parity improvements at 100B
- What is new: More unfiltered data does not fix gender label/occupation associations, but it narrows cross‑group performance gaps (Section 4.4).
- Evidence:
  - Persistent representation bias: models prefer “Male” over “Female” ≈85% of the time; this remains high at 100B (Table 4).
  - Association bias: heatmaps in Figure 2 show occupation–gender skews remain across scales (e.g., “nurse”, “secretary” favored for female images).
  - Performance disparity shrinks: e.g., GeoDE regional disparity decreases for all sizes when trained with 100B data (Table 5, lower half: disparities fall from 4.7→4.4 for `ViT-B`, 3.2→2.8 for `ViT-L`, 3.6→2.7 for `ViT-H`).

These are not merely incremental metric bumps; they reframe what “scaling helps” means for VLMs: less gain on headline Western benchmarks, more gain on long‑tail inclusion and cross‑group parity.

## 5. Experimental Analysis
- Evaluation design and metrics
  - All classification and retrieval numbers are reported as error percentages (lower is better) unless otherwise noted; representation bias is a disparity measure (Section headers of Tables 2–3; footnote in Appendix B).
  - Compute control: every configuration sees 100B examples total, enabling a fair comparison of “more unique data once” vs “less unique data many times” (Section 3.2).

- Main quantitative results
  - Western‑centric saturation (Section 4.1; Table 2)
    - Quote: 
      > Wilcoxon’s signed rank test gives a p‑value of 0.9, indicating differences are not significant.
    - Examples (10B→100B):
      - `ViT-L/16` ImageNet 0‑shot error: 29.7%→28.5% (−1.2).
      - `ViT-H/14` COCO T2I@1 error: 60.3%→59.3% (−1.0).
    - Scaling law fits report similar asymptotic limits across scales (Tables 2, 95% CIs not significantly different; p=0.09).

  - Cultural diversity gains (Section 4.2; Table 3)
    - Quote:
      > Scaling training data from 10B to 100B yields substantial gains on Dollar Street 10‑shot, where ViT‑L and ViT‑H see absolute improvements of 5.8% and 5.4% respectively.
    - Additional examples:
      - `ViT-H/14` GeoDE‑Country 10‑shot error: 50.2%→47.6% (−2.6).
      - `ViT-L/16` GLDv2 0‑shot error: 46.4%→45.7% (−0.7), a smaller but consistent improvement.

  - Multilinguality (Section 4.3; Figure 3; Appendix B)
    - Figure 3 shows larger decreases in error for low‑resource languages at 100B than for high‑resource. The gap widens with model size (bars annotated with improvements; e.g., Δ≈2–3 points for low‑resource vs ≈1 point for high‑resource at `ViT-H`).

  - Fairness (Section 4.4; Tables 4–5; Figure 2)
    - Representation bias: 
      > Values ≈85%—preference for “Male” remains—do not improve at 100B (Table 4).
    - Association bias:
      > Heatmaps (Figure 2) for five occupation pairs across three model sizes and three data scales show persistent gender‑occupation stereotypes.
    - Performance disparity improvement:
      - GeoDE regional disparity reduces across all model sizes at 100B (Table 5). 
      - Dollar Street income disparity slightly improves for `ViT-B` (32.5→29.0), stays similar for `ViT-H` (32.2→32.1), and increases slightly for `ViT-L` (29.7→30.4).

  - Transfer to generative models (Section 4.5; Table 6; Appendix C)
    - Aggregated results for `ViT-L/16` encoders in `PaliGemma` (frozen vs unfrozen):
      - Frozen averages: 73.6 (1B), 72.7 (10B), 73.9 (100B).
      - Unfrozen averages: 75.1 (1B), 73.7 (10B), 75.3 (100B).
    - Quote:
      > When taking noise level into consideration, no consistent performance gains across downstream tasks are observed as pretraining data scale increases (Table 6).

  - Quality filtering (Section 5.1; Figure 4; Appendix D)
    - Western metrics: CLIP‑filtered outperforms baseline (e.g., average Western 0‑shot classification error is lower across seen‑example checkpoints; Appendix D).
    - Cultural/fairness metrics: CLIP and Classifier filters hurt performance (Figure 4 middle/right).

  - Language rebalancing (Section 5.2; Figure 5; Table 11)
    - After upsampling 7 low‑resource languages to 1% each, low‑resource retrieval error drops notably (Figure 5 top‑left; e.g., `ViT-L/16`, 100B seen: 75.01%→70.10%).
    - Side‑effects: small degradations on high‑resource and Western metrics but overall multilingual average still improves (Figure 5; Table 11).

  - Qualitative attention maps (Section 5.3; Tables 1 and 7)
    - At 100B, attention focuses more precisely on culturally salient regions (e.g., the dome shape of an igloo; detailed patterns in “Igorot Dance”), illustrating learned representations beyond Western concepts.

- Do the experiments support the claims?
  - Yes for the central thesis: tables and statistical tests directly show saturation on classic benchmarks (Section 4.1) and significant gains in cultural diversity and low‑resource languages (Section 4.2; Figure 3).
  - The filtering and rebalancing analyses are ablations that clarify mechanisms: filtering skews the data distribution (hurting diversity), while rebalancing directly benefits targeted languages (Sections 5.1–5.2).
  - Bias findings are mixed: parity improves (Table 5), but inherent gender biases persist (Table 4; Figure 2), aligning with the nuance in the conclusions (Section 4.4; Section 7).

## 6. Limitations and Trade-offs
- Assumptions and design choices
  - Minimal filtering is intentional to preserve diversity, but it also retains noise (Section 6 “Discussion: Data Filtering”). This likely explains why classic benchmarks do not improve.
  - Language identification uses the `content-language` meta tag; it can be missing or inaccurate, potentially affecting multilingual statistics (Section 3.1).
- Unaddressed scenarios / scope limits
  - Inclusivity is broader than the chosen metrics; only 36 languages in `Crossmodal‑3600`, and fairness is limited to gender and regional/income disparity (Section 6 “Limitations”).
  - Cultural metrics use specific datasets (Dollar Street, GeoDE, GLDv2); other forms of cultural knowledge (e.g., festivals, artifacts beyond landmarks/household items) are not separately benchmarked.
- Computational and data constraints
  - While compute is “matched” across scales in terms of seen examples, training on 100B unique pairs (1 epoch) requires enormous data infrastructure and may behave differently than training on smaller data for many epochs (Section 3.2). This matters for practitioners who may not afford data collection or streaming at this scale.
- Trade‑offs evidenced by experiments
  - Filtering vs inclusivity: CLIP filtering helps Western benchmarks but reduces cultural diversity and fairness metrics (Figure 4; Appendix D). 
  - Rebalancing vs Western performance: upsampling low‑resource languages helps those languages but slightly hurts Western and some high‑resource metrics (Figure 5; Table 11).
- Open questions
  - Can we design filtering that preserves or even enhances cultural diversity and multilinguality? The paper explicitly calls for new filtering strategies with this goal (Section 6).
  - How do results change at higher image resolutions or longer text inputs than 224×224 and 64 tokens?

## 7. Implications and Future Directions
- How this work changes the field’s perspective
  - It reframes “benefit from more raw data” for VLMs: at 100B, the primary returns are not on saturated Western‑centric leaderboards but on inclusivity—cultural coverage, low‑resource languages, and reduced regional performance disparity (Figure 1; Sections 4.2–4.4; Table 5).
  - It cautions against default CLIP‑based filtering if the goal includes inclusivity; filtering can systematically strip away long‑tail cultural content (Section 5.1; Figure 4).
- Follow‑up research directions
  - Diversity‑preserving filtering and data governance:
    - Learn data selection that explicitly optimizes for cultural and linguistic coverage alongside quality.
    - Develop auditing tools that quantify coverage before training (expanding beyond the current fairness metrics).
  - Mixture design and curriculum:
    - Automated language and culture‑aware mixing ratios that balance Western performance with inclusivity.
    - Adaptive curriculum over data scales so that the model sees rare concepts enough times without overwhelming repeats of common ones.
  - Model/architecture innovations:
    - Explore whether generative pre‑training or hybrid losses at 100B can translate inclusivity gains into stronger downstream generative performance (since PaliGemma transfer gains were inconsistent; Table 6).
    - Investigate higher‑resolution pre‑training and longer text contexts for better OCR and fine‑grained cultural cues.
- Practical applications and use cases
  - Global search and retrieval systems that must work across languages and cultures.
  - Public‑sector and NGO tools where equitable performance (reduced subgroup disparity) matters—e.g., geolocalization for disaster response or cultural heritage identification (Table 5, GeoDE disparity improvements).
  - Multimodal assistants intended for worldwide audiences; the results argue for 100B‑scale pretraining if inclusivity is a requirement (Section 7).

> Bottom line (Section 7): while 100B raw web data may not move the needle on classic Western benchmarks, it is “vital for building truly inclusive multimodal systems.” The study’s careful compute‑matched design and broad evaluation provide strong evidence for this claim, and its analyses of filtering and rebalancing show actionable levers to further improve inclusivity.
