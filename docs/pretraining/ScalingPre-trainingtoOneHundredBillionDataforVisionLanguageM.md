# Scaling Pre-training to One Hundred Billion Data for Vision Language Models

**ArXiv:** [2502.07617](https://arxiv.org/abs/2502.07617)

## 🎯 Pitch

This paper empirically explores the impact of scaling vision-language model pre-training to an unprecedented 100 billion image–text pairs with the new WebLI-100B dataset. While performance on traditional Western-centric benchmarks plateaus at this scale, the study reveals substantial new gains in cultural diversity, support for low-resource languages, and demographic fairness—making a compelling case that massive, diverse data is crucial for building truly inclusive multimodal AI systems.

---

## 1. Executive Summary (2-3 sentences)
This paper investigates what happens when pre‑training data for vision‑language models (VLMs) is scaled by an order of magnitude—from 10 billion to 100 billion image–text pairs—using a new dataset, `WebLI‑100B`. The central finding is counterintuitive but important: at this scale, “standard” Western‑centric benchmarks (e.g., ImageNet zero‑shot, COCO retrieval) largely saturate, but inclusivity‑related capabilities—cultural diversity, multilinguality (especially low‑resource languages), and performance parity across subgroups—improve markedly (Sections 1, 4; Figure 1; Tables 2–5).

## 2. Context and Motivation
- Problem/gap addressed
  - The largest reported web‑scale image–text datasets have plateaued around 10 billion pairs. It is unclear whether pushing to 100 billion unique examples yields meaningful benefits, and if so, where those benefits appear (Section 1).
  - Prior work on scaling laws mostly focused on accuracy improvements in established Western‑centric benchmarks; less is known about inclusivity‑related outcomes such as cultural diversity and low‑resource multilingual performance (Sections 1–2).

- Why it matters
  - Real‑world impact: Inclusive multimodal systems need breadth—coverage of low‑frequency, long‑tail cultural and linguistic concepts. These are under‑represented on the web and are often pruned by common quality filters (Section 1; Section 5.1).
  - Theoretical significance: Classic power‑law scaling suggests diminishing returns but continued gains with more data; this paper examines whether the “returns” shift from headline benchmarks to inclusivity metrics at massive data scale (Section 1; scaling laws fit in Tables 2–3).

- Prior approaches and shortcomings
  - Datasets like Conceptual Captions, LAION‑5B, and WebLI (~10B) enabled strong VLMs (CLIP, ALIGN, SigLIP), typically with quality filtering (often English‑centric) to improve benchmark performance (Section 2).
  - These filters can disproportionately remove long‑tail, culturally diverse examples (Section 2; Section 5.1), and many benchmarks reflect Western images/languages, masking inclusivity gaps (Section 2).

- This paper’s position
  - Builds `WebLI‑100B`, a 100‑billion‑pair multilingual, minimally filtered web dataset (only safety/PII filters), and studies the effect of scaling on both traditional and inclusivity‑oriented evaluations under compute‑matched training (Sections 3.1–3.3).
  - Goes beyond accuracy: also measures multilingual retrieval across 36 languages (Crossmodal‑3600), cultural diversity (Dollar Street, GeoDE, GLDv2), fairness (bias and disparity), and transfer to a generative VLM (`PaliGemma`) (Section 3.3).

## 3. Technical Approach
This is an empirical scaling study with a carefully controlled pre‑training and evaluation pipeline.

- Data construction and splits (Section 3.1)
  - `WebLI‑100B`: 100B image–text pairs scraped from the web, using image alt‑text and page titles as text. Only essential filters applied: remove harmful images and PII; near‑duplicates removed against >90 common evaluation tasks to prevent leakage.
  - `1B` and `10B` subsets: random 1% and 10% samples of the 100B set.
  - Quality‑filtered sets (Section 5.1): From raw web data, create three 5B‑pair English datasets:
    - “CLIP‑filtered” using `CLIP‑L/14` as a filter (keep high image–text alignment).
    - “Classifier‑filtered” using a custom VLM classifier trained to detect aligned pairs.
    - “Baseline (en)” as an unfiltered English subset for comparison.
  - Language‑rebalanced sets (Section 3.1; Section 5.2): Upsample selected low‑resource languages used in Crossmodal‑3600—Bengali, Filipino, Hindi, Hebrew, Māori, Swahili, Telugu—so that each comprises 1% of training batches (collectively 7%); the remaining 93% sampling follows the original distribution. This isolates the effect of targeted language balancing.

- Models and training (Section 3.2)
  - Contrastive pre‑training using `SigLIP` (Sigmoid loss variant of CLIP) with `ViT‑B/16`, `ViT‑L/16`, and `ViT‑H/14` backbones for both image and text encoders.
  - Training details:
    - Batch size 32k; inverse square‑root learning‑rate schedule with 200M warmup and cooldown examples.
    - LR = 0.001; weight decay = 1e‑4.
    - Images at 224×224 resolution; text tokenized with multilingual `mt5` tokenizer, max 64 tokens.
  - Compute‑matched regime: every model is trained until it has “seen” 100B examples total—e.g., 1B‑data models see 100 epochs; 10B‑data models see 10 epochs; 100B‑data models see 1 epoch (Section 3.2). Checkpoints evaluated after seeing 3, 7, 10, 17, 26, 33, 49, 66, and 100B examples.

- Evaluations (Section 3.3)
  - Western‑centric tasks:
    - Zero‑shot classification: ImageNet, CIFAR‑100, Oxford‑IIIT Pet.
    - 10‑shot classification (few labeled examples per class; a lightweight classifier is trained on frozen features): Birds (CUB), Caltech‑101, Cars196, Colorectal Histology, DTD.
    - Image–text retrieval on COCO Captions and Flickr30k in both directions; metric: `Recall@1` (what fraction of queries retrieve the correct item at rank 1).
  - Cultural diversity:
    - Zero‑shot classification on Dollar Street (mapped to ImageNet labels), GeoDE, GLDv2.
    - 10‑shot geolocalization on Dollar Street and GeoDE (predict country/region from image with few labeled examples).
  - Multilinguality:
    - Crossmodal‑3600: zero‑shot retrieval in 36 languages; report language‑wise results and aggregates for low‑ vs high‑resource languages.
  - Fairness (Section 3.3; Section 4.4):
    - Representation bias (RB): how often the model prefers “Male” over “Female” for random images (first‑order bias).
    - Association bias (AB): how often gender correlates with occupation words (e.g., “nurse” vs “doctor”) using FairFace images (second‑order bias); reported as preference probabilities (Figure 2).
    - Performance disparity: max accuracy gap across socioeconomic (Dollar Street income buckets) and geographic (GeoDE regions) subgroups (Table 5).
  - Transfer to generative models (Section 3.3; Section 4.5):
    - Initialize `PaliGemma`’s vision tower with each contrastively‑trained encoder; run 50M pre‑training examples (stage‑1, 224×224) under two settings: frozen vs unfrozen vision tower; then fine‑tune on a broad suite of captioning, VQA (incl. OCR, counting), multilingual, and remote sensing tasks (Table 6; Appendix C).

- Scaling‑law fitting (Sections 4.1, 4.2; Tables 2–3)
  - For each benchmark and model size, fit a power law `f(x) = α x^{-c} + ε` where `x` is data size and `f(x)` is error. Report the exponent `c` and the asymptotic “limit” (`ε`) to assess whether additional compute would change the observed trends.

## 4. Key Insights and Innovations
- A1. At 100B scale, traditional benchmarks saturate, but inclusivity improves
  - What’s new: The paper separates “Where do we see gains?” into two buckets and shows they diverge at 100B.
  - Evidence:
    - Western‑centric: “Scaling from 10B to 100B shows limited benefits” (Table 2). Example: `ViT‑L` ImageNet zero‑shot error drops modestly 29.7% → 28.5% (−1.2 pts); many COCO/Flickr retrieval numbers stagnate or even worsen slightly at `Recall@1` (higher error) for larger backbones.
    - Inclusivity: Cultural and multilingual metrics improve more at 100B (Figure 1 right; Table 3; Figure 3). Example: Dollar Street 10‑shot (`ViT‑L`) error 64.1% → 58.3% (−5.8 pts), i.e., accuracy +5.8 points (Table 3); low‑resource language retrieval improves more than high‑resource (Figure 3).
  - Why it matters: It reframes the objective of extreme scaling—less about squeezing extra points on COCO/ImageNet, more about reaching the long tail of cultural/linguistic phenomena.

- A2. Common quality filters trade off diversity for benchmark performance
  - What’s new: A controlled comparison of CLIP‑style filtering vs no filtering at fixed dataset size (5B) shows clear trade‑offs (Section 5.1; Figure 4; Appendix D).
  - Evidence:
    - Western‑centric averages improve with CLIP filter (e.g., at 30B seen examples, “Average Western‑centric” error: 23.21% baseline vs 22.14% CLIP; Appendix D).
    - Cultural diversity degrades (e.g., at 30B seen examples, “Average Cultural Diversity” error: 49.49% baseline vs 54.96% CLIP; Appendix D).
  - Why significant: Many pipelines default to CLIP‑filtering; the study warns this can erase rare cultural contexts even when starting from 100B raw examples.

- A3. Language rebalancing helps low‑resource languages with minimal collateral damage
  - What’s new: Simple upsampling of seven low‑resource languages to 1% each substantially improves their zero‑shot retrieval while only slightly affecting high‑resource languages (Section 5.2; Figure 5; Table 11).
  - Evidence (for `ViT‑L`, 100B seen examples):
    - Low‑resource average error: 75.01% → 70.10% (−4.91 pts).
    - High‑resource average error: 45.43% → 45.75% (+0.32 pts).
    - Cultural diversity average improves slightly (44.01% → 43.29%), while Western‑centric average degrades slightly (26.87% → 27.55%) (Table 11).
  - Why significant: It shows a lightweight, data‑mixing knob that directly targets inclusivity without requiring more total data.

- A4. Data scale reduces performance disparity across groups, but not intrinsic gender biases
  - Evidence (Table 5): With 100B seen examples, regional disparity on GeoDE shrinks (e.g., `ViT‑L`: 3.2 → 2.8), and Dollar Street income disparity is stable or slightly improved for some backbones.
  - However, representation bias remains high (Table 4): models prefer “Male” over “Female” ~85% of the time, and association bias heatmaps (Figure 2) show persistent gender–occupation stereotypes; scaling alone does not fix these.

## 5. Experimental Analysis
- Evaluation design recap (Section 3.3)
  - Broad coverage: Zero‑shot and few‑shot classification, bidirectional retrieval, multilingual retrieval, cultural geolocalization, fairness (bias and subgroup disparity), and transfer to generative tasks.
  - Compute‑matched training ensures observed differences are attributable to data scale and mix, not more optimizer steps or larger batches.

- Main quantitative results and comparisons
  - Western‑centric saturation (Table 2; Section 4.1)
    > “Increasing the dataset size from 10B to 100B… does not improve performance substantially,” supported by Wilcoxon’s test with p = 0.9 and scaling‑law limits that are statistically indistinguishable (p = 0.09).
    - Examples (error ↓ is better):
      - ImageNet zero‑shot: `ViT‑B`: 39.35 → 39.04 (−0.31); `ViT‑L`: 29.70 → 28.49 (−1.21); `ViT‑H`: 25.60 → 24.90 (−0.70).
      - COCO I2T Recall@1 error (lower is better): `ViT‑L`: 47.18 → 45.28 (small gain), but Flickr I2T: 15.50 → 16.60 (worse).
    - Scaling exponents vary (−0.1 to −1.3 across rows) with near‑identical asymptotic limits for 10B vs 100B, reinforcing saturation.

  - Cultural diversity gains (Table 3; Section 4.2)
    > “Scaling … yields substantial gains on Dollar Street 10‑shot… `ViT‑L` and `ViT‑H` see absolute improvements of 5.8% and 5.4%,” with p = 0.002 (Wilcoxon).
    - Dollar Street 10‑shot error: `ViT‑L`: 64.1 → 58.3; `ViT‑H`: 59.1 → 53.7.
    - GeoDE 10‑shot region error: `ViT‑H`: 47.6 → 44.7 (−2.9); country: 50.2 → 47.6 (−2.6).
    - GLDv2 zero‑shot error improves substantially with larger models at 100B vs 10B (e.g., `ViT‑H`: 40.1 → 38.8).

  - Multilinguality (Figure 3; Appendix B; Section 4.3)
    > “Low‑resource languages benefit more from the 100B scale than the high‑resource ones,” and the gap widens with model size.
    - Examples (`ViT‑L`, Image‑to‑Text error, lower is better): Telugu 76.67 → 69.69 (−6.98), Bengali 66.36 → 63.75 (−2.61), Hebrew 39.44 → 35.72 (−3.72).
    - High‑resource languages see smaller changes; sometimes flat or mixed across directions (Appendix B, Table 8).

  - Fairness (Section 4.4; Tables 4–5; Figure 2)
    - Representation bias: 
      > “Models… have a significantly higher preference to associate… ‘Male’ over ‘Female.’ In fact, this occurs nearly 85% of the time. Training on 100B examples does not mitigate this effect.”  
      Table 4 shows: `ViT‑L` RB 88.2% (1B) → 85.5% (100B); `ViT‑H` 86.8% → 86.6%.
    - Association bias: heatmaps (Figure 2) display persistent gender–occupation associations across all scales; no systematic reduction from 10B → 100B.
    - Disparity (max subgroup accuracy gap): improves overall with scale on GeoDE (e.g., `ViT‑L` 3.2 → 2.8; Table 5), and is stable/slightly improved on Dollar Street for some backbones (e.g., `ViT‑B` 32.5 → 29.0).

  - Transfer to generative models (`PaliGemma`; Section 4.5; Table 6)
    > “We do not observe consistent performance gains across downstream tasks as we scale the pre‑training dataset.”
    - Aggregated (unfrozen vision): Semantics 77.1 → 77.2; OCR 69.5 → 70.0; Multilinguality 66.9 → 67.0; Remote Sensing 92.0 → 91.8. Similar small shifts with frozen vision. Gains are within noise.

  - Qualitative attention maps (Section 5.3; Table 1; Appendix A)
    > “Models trained on larger data tend to have more focused attention on semantically relevant regions,” e.g., sharper focus on igloo dome structure and bison rather than background.

  - Quality filtering ablation (Section 5.1; Figure 4; Appendix D)
    - Western‑centric averages: improve under CLIP filtering (e.g., at 30B seen, 23.21% → 22.14% error).
    - Cultural diversity: worsens under CLIP filtering (e.g., at 30B seen, 49.49% → 54.96% error).
    - Fairness averages: marginal changes; sometimes worse under filtering (Appendix D).

  - Language rebalancing ablation (Section 5.2; Figure 5; Table 11)
    - Low‑resource average error drops by ~5 points at 100B seen; high‑resource slightly up (~0.3 points); cultural metrics slightly better; Western tasks slightly worse.

- Convincingness and robustness
  - Strong: compute‑matched training, multiple backbones (B/L/H), multiple “seen‑examples” checkpoints, significance tests (Wilcoxon), and scaling‑law fits all support the central claim that inclusivity gains dominate at 100B while traditional benchmarks saturate (Sections 4.1–4.3).
  - Balanced: transfer experiments and bias metrics show where scaling alone is insufficient (Sections 4.4–4.5).
  - Coverage: extensive appendices detail per‑language results and filter/rebalance ablations (Appendices B–F).

- Conditions and trade‑offs
  - Improvements concentrate in diverse/long‑tail tasks; small or negative changes on familiar benchmarks and for high‑resource languages (Tables 2–3; Figure 3).
  - Filtering trades Western‑centric gains for diversity losses (Figure 4).
  - Rebalancing trades high‑resource performance slightly for low‑resource improvement (Figure 5; Table 11).

## 6. Limitations and Trade-offs
- Assumptions and scope
  - Raw web data is assumed beneficial for long‑tail coverage; only essential safety and PII filters were applied (Section 3.1). This keeps diversity but retains noise/misalignment that may depress classic benchmarks.
  - Evaluation of “inclusivity” is via specific proxies (Dollar Street, GeoDE, GLDv2, Crossmodal‑3600). Inclusivity is broader than these metrics (Section 6, Limitations).

- What is not addressed
  - Public release of `WebLI‑100B` is not indicated; reproducibility at this scale is challenging.
  - Bias mitigation beyond data scale/mix (e.g., debiasing objectives, counterfactual augmentation) is not explored; gender biases persist (Section 4.4).
  - Only 224×224 resolution and SigLIP are studied; other architectures/losses or higher resolutions might interact differently with 100B scale.

- Computational and data constraints
  - Training to 100B “seen examples” is extremely compute‑intensive; while compute‑matched comparisons are fair, many labs cannot reproduce them.
  - Data labeling is weak (alt‑text/page titles), which is noisy; filters that help benchmarks tend to erase long‑tail data (Section 5.1).

- Open questions
  - Can we design “diversity‑preserving” filters that retain long‑tail cultural/linguistic content while improving alignment?
  - How would multi‑stage curricula (e.g., start diverse, then refine with filtered data) affect both inclusivity and benchmark performance?
  - What architectural or objective choices (e.g., generative pre‑training at scale) better exploit 100B raw data for both inclusivity and standard benchmarks?

## 7. Implications and Future Directions
- How this changes the landscape
  - Reorients the rationale for extreme data scaling: at 100B scale, the most salient returns are inclusivity‑centric—cultural diversity, low‑resource multilinguality, and reduced subgroup disparity—rather than headline benchmark gains (Sections 4, 7).
  - Calls into question heavy reliance on Western‑centric leaderboards to judge progress at massive scale.

- Follow‑up research enabled/suggested
  - Diversity‑preserving data curation:
    - New filter objectives that explicitly protect long‑tail cultural/linguistic content (Section 6 Discussion).
    - Language‑aware or region‑aware sampling schedules beyond uniform upsampling.
  - Objective and architecture innovations:
    - Combine contrastive and generative objectives at 100B scale; investigate whether generative pre‑training can translate inclusivity gains into broader task improvements (Section 4.5).
    - Study resolution scaling and tokenization choices for multilingual alt‑text at scale.
  - Better benchmarks:
    - Develop broader, more representative test suites for culture, language, and fairness, since current gains may be underestimated by Western‑centric metrics (Section 6, Limitations).
  - Bias mitigation beyond data volume:
    - Integrate debiasing techniques (balanced batches, adversarial losses, counterfactual prompts) since gender representation/association biases persisted even at 100B (Section 4.4).

- Practical applications
  - Building inclusive multimodal assistants and retrieval systems for global users; better recognition/geolocalization of non‑Western concepts and landmarks; improved zero‑shot performance in low‑resource languages (Sections 4.2–4.3).
  - Caution for industry pipelines: CLIP‑style filtering improves near‑term benchmarks but can harm inclusivity (Section 5.1). For global products, prefer larger, less‑filtered corpora and language‑balanced sampling.

> Overall conclusion (Section 7): scaling noisy, raw web data to 100B examples is “vital for building truly inclusive multimodal systems,” even if it delivers only “tiny improvements” on established Western‑centric benchmarks.
