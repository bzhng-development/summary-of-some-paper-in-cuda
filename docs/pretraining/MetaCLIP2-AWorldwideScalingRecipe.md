# Meta CLIP 2: A Worldwide Scaling Recipe

**ArXiv:** [2507.22062](https://arxiv.org/abs/2507.22062)

## 🎯 Pitch

Meta CLIP 2 delivers the first fully open and scalable method to train CLIP vision–language models directly on worldwide, multilingual web data—solving the long-standing challenge where adding non-English data degraded English performance (the 'curse of multilinguality'). By introducing per-language metadata, scalable curation, and carefully matched model capacity, it achieves state-of-the-art results across both English and multilingual tasks using standard CLIP architectures. This breakthrough paves the way for future foundation models that natively understand a truly global, culturally rich web, and ensures continued progress as English data becomes saturated.

---

## 1. Executive Summary
Meta CLIP 2 introduces the first fully public, from-scratch recipe to train CLIP-style image–text encoders on worldwide (multilingual) web data while maintaining or improving English performance. By building per‑language metadata and curation, scaling the number of seen training pairs, and using sufficient model capacity, it breaks the long‑standing “curse of multilinguality” and achieves new state‑of‑the‑art results on multilingual benchmarks (e.g., XM3600 and CVQA), all with minimal architectural changes to standard CLIP (Fig. 1, Sec. 3; Table 1).

## 2. Context and Motivation
- Problem addressed
  - CLIP-style models are typically trained on English web image–text pairs. Extending training to non‑English data has faced two unsolved issues (Sec. 1):
    - No open, scalable curation method for multilingual web data.
    - English performance degrades when training with multilingual data (“curse of multilinguality”).
- Why this matters
  - Web content is increasingly multilingual: roughly half of the web is non‑English (Sec. 1 citing Wikipedia 2025). Foundation models (e.g., CLIP encoders used in many multimodal LLMs) need to represent global concepts, languages, and cultures. English data may soon be exhausted, making multilingual data crucial for continued scaling (Sec. 1; Villalobos et al., 2022).
- Prior approaches and their gaps
  - Distillation or filtering with external/closed systems:
    - LAION (used by OpenCLIP) filters with a black‑box CLIP teacher (Sec. 2.1).
    - DFN uses a private filter model trained on private data (Sec. 2.1).
    - mSigLIP/SigLIP 2 rely on WebLI, built with private pipelines (Sec. 2.3).
  - Translation‑based pipelines (translate captions to/from English) inject translation artifacts and bias (Sec. 2.3).
  - Empirically, multilingual models often underperform English‑only counterparts on English benchmarks (e.g., mSigLIP trails SigLIP on ImageNet; Sec. 2.3).
- Positioning
  - Meta CLIP 2 provides a public, reproducible, and model‑agnostic worldwide curation and training recipe that:
    - Requires no private data, teacher models, or machine translation (Sec. 1).
    - Maximizes overlap with standard CLIP architecture to make findings broadly applicable (Sec. 1).
    - Identifies minimal changes that jointly remove the English‑multilingual trade‑off (Fig. 1; Sec. 3–4).

## 3. Technical Approach
This section explains “how it works,” from data construction to training.

- Background: how English Meta CLIP curation works (Sec. 3.1)
  - Build `metadata M`: a list of high‑quality “visual concepts” (e.g., WordNet synsets, Wikipedia unigrams/bigrams, page titles).
  - For each web image–alt‑text pair, do substring matching between the alt‑text and `M` to find which concepts appear.
    - “Alt‑text” is the caption-like text associated with an image on the web.
  - Count how often each metadata entry is matched across the data pool.
  - Balance head vs. tail concepts using a threshold `t`:
    - “Tail” concepts are rare (count < `t`); “head” concepts are frequent (count ≥ `t`).
    - Tail matches get kept with probability 1; head matches are downsampled with probability `t / count`.
  - Sample pairs proportionally to these probabilities, transforming the raw long‑tailed internet distribution toward a balanced training set.

- Step 1: Worldwide metadata (Sec. 3.2)
  - Build separate `M_lang` per language rather than merging everything:
    - 31 languages from Multilingual WordNet; Wikipedia unigrams/bigrams for 329 languages; Wikipedia titles ranked by click‑through (Sec. 3.2).
    - For languages without spaces (e.g., Chinese, Japanese, Thai), apply open‑source language‑specific tokenizers to extract meaningful word units for unigrams/bigrams (Appendix A.1, Table 5).
  - Why per‑language?
    - A string can mean different things in different languages (“mit” in English vs. German).
    - Ablations show per‑language metadata improves results over a merged pool (Table 2, Steps 3–5).

- Step 2: Worldwide curation with language‑specific balancing (Sec. 3.3; Algorithm 1)
  - Assign each alt‑text a language using language identification (`LID`; Grave et al., 2018).
  - Map LID languages to metadata languages; merge metadata where needed to cover all LID labels (Sec. 3.3).
  - Substring‑match each alt‑text only against its language’s metadata: `matched_entry_ids = substr_match(text, M[text.lang])`.
  - Compute counts per metadata entry per language: `entry_counts[lang]`.
  - Derive a head–tail threshold `t_lang` separately for each language by preserving the same tail‑match proportion across languages:
    - Compute the English tail proportion `p` implied by `t_en` (20k in OpenAI CLIP, 170k in Meta CLIP) using `t_to_p(t_en, entry_counts["en"])` (Algorithm 1).
    - For each language, find `t_lang` such that the cumulative proportion of matches below `t_lang` equals `p` (`p_to_t` in Algorithm 1).
  - Compute sampling probabilities: for each language, set `entry_probs[lang] = t_lang / max(entry_count, t_lang)`.
  - Sample an image–text pair if any of its matched entries is sampled; tail entries always pass (Stage 3 in Algorithm 1).
  - Intuition
    - English `t_en` encodes a desirable head–tail mix (6% tail matches in prior work; Sec. 3.3). Keeping the same tail proportion per language avoids over‑representing frequent concepts from high‑resource languages and drowning out rare concepts in low‑resource languages (Table 2 shows merged or single‑threshold approaches hurt performance).

- Step 3: Training framework tailored to worldwide scaling (Sec. 3.4)
  - Multilingual tokenizer
    - Swap the English tokenizer for a multilingual one; ablations compare mT5, Gemma, XLM‑R, and XLM‑V (Table 3). XLM‑V yields the best multilingual results without hurting English.
  - Scale “seen pairs” (total number of training examples processed) proportionally to the added non‑English data to keep English exposure constant:
    - After LID, English is 44% of training data. Increase global batch size by 2.3× so the model sees the same number of English examples while adding the non‑English ones (Sec. 3.4).
    - Keep all other hyperparameters from standard CLIP (learning rate, warm‑up) unchanged (Appendix B; Table 6 shows batch from 32,768 to 75,366; seen pairs from 12.8B to 29B).
  - Minimal viable model capacity
    - The larger `ViT‑H/14` is needed to break the multilingual curse; `ViT‑L/14` remains capacity‑limited even with scaled pairs (Fig. 1 left; Table 1).

- Practical engineering to make worldwide curation feasible (Appendix A.2)
  - Use the Aho‑Corasick multi‑pattern matching algorithm to speed substring matching by ~2000× over brute force, enabling million‑scale metadata per language.
  - Lazy loading of per‑language automatons and memory‑mapped counts keep CPU/RAM use tractable.
  - Safety filtering (remove NSFW), face/PII removal, and benchmark de‑dup via 64‑bit hashes limit leakage (Appendix A.2).

## 4. Key Insights and Innovations
- Per‑language metadata and per‑language head–tail balancing (fundamental)
  - What’s new: Separate metadata and thresholds `t_lang` per language, computed to preserve a fixed tail proportion derived from English (Algorithm 1; Sec. 3.3).
  - Why it matters: Avoids biasing toward high‑resource languages and prevents tail concepts from being washed out. Ablations show that not isolating languages or using a single threshold substantially hurts English and multilingual performance (Table 2, Steps 3–4 vs. Step 5).
- Scale seen pairs rather than “squeezing” English (fundamental)
  - What’s new: Increase global batch/seen pairs to match the larger data pool so English exposure remains unchanged while adding non‑English data (Sec. 3.4).
  - Why it matters: With constant seen pairs, English performance drops; with 2.3× seen pairs, `ViT‑H/14` trained on worldwide data surpasses English‑only on ImageNet (81.3% vs. 80.5%) and sets new multilingual SoTA (Table 1).
- Identify the capacity threshold that breaks the multilingual curse (insightful empirical finding)
  - What’s new: A systematic comparison shows `ViT‑L/14` still suffers the curse even with scaled pairs, whereas `ViT‑H/14` with scaled pairs breaks it (Fig. 1, Table 1).
  - Why it matters: Clarifies that multilingual degradation is not inevitable; it stems from insufficient capacity and training exposure.
- “No‑filter” worldwide scaling with open data curation (practical and impactful)
  - What’s new: Remove the last language filter in the pipeline (don’t drop non‑English alt‑texts), relying instead on principled balancing; train directly on native alt‑texts (Sec. 1).
  - Why it matters: Improves cultural and geographic coverage and reduces translation/teacher biases. This yields large gains on diversity‑focused benchmarks (Table 4; Fig. 3).

## 5. Experimental Analysis
- Evaluation setup (Sec. 4.1–4.2)
  - Data
    - Large web image–alt‑text corpus curated with the worldwide algorithm. After LID, English is ~44% (Sec. 4.1).
    - Training subsets used for analysis: English‑only (1.0× seen pairs), Non‑English (1.3×), Worldwide with constant seen pairs (1.0×), Worldwide with scaled seen pairs (2.3×) (Sec. 4.2.1; Table 1).
  - Architectures and training
    - Aligns with OpenAI/Meta CLIP; primarily `ViT‑L/14` and `ViT‑H/14` at 224 px. Only necessary changes: multilingual tokenizer, scaled batch/seen pairs (Sec. 3.4; Table 6).
  - Benchmarks and metrics
    - English: ImageNet (top‑1), SLIP‑26 average, DataComp‑37 average.
    - Multilingual: Babel‑ImageNet (avg. across 280 languages), XM3600 retrieval Recall@1 (T→I and I→T), CVQA (English and Local answer accuracy).
    - Additional retrieval: Flickr30k‑200, XTD‑10, XTD‑200 (Recall@1).
    - Cultural diversity: Dollar Street, GeoDE, GLDv2 (Table 4).
    - Embedding quality: alignment and uniformity on a 5k holdout (Fig. 4).
  - Baselines
    - OpenCLIP on LAION‑5B (XLM‑CLIP), mSigLIP and SigLIP 2 (WebLI), and English Meta CLIP (Table 1).

- Main results (Table 1, Fig. 1)
  - Breaking the multilingual curse requires both capacity and scaled seen pairs:
    - `ViT‑H/14`, Worldwide 2.3× seen pairs
      - “English helps multilingual and vice versa” (Fig. 1 right) and surpasses English‑only on ImageNet:
      - ImageNet: 81.3% vs. 80.5% English‑only (Table 1).
      - Multilingual SoTA with minimal changes:
        - Babel‑ImageNet: 50.2% avg.
        - XM3600 I→T: 64.3; T→I: 51.5
        - CVQA Local: 57.4; English: 61.5
        - Also strong on Flickr30k‑200 and XTD‑200 (Table 1).
    - `ViT‑H/14`, Worldwide 1.0× seen pairs
      - ImageNet lower (79.5%) and multilingual lower than 2.3×, showing scaling seen pairs is crucial.
    - `ViT‑L/14`, Worldwide 2.3×
      - Still worse on ImageNet than its English counterpart (78.8% vs. 79.5%; Table 1), showing capacity limits (Fig. 1 left).
  - Comparison with mSigLIP/SigLIP 2
    - With fewer seen pairs (72% of SigLIP series) and lower resolution (224 vs. 256), Meta CLIP 2 outperforms on most English aggregation metrics and clearly on multilingual ones:
      - Quote from Table 1:
        > “Meta CLIP 2 H/14 (Worldwide 2.3×): IN 81.3; SLIP‑26 74.5; DC‑37 69.6; Babel‑IN 50.2; XM3600 I→T 64.3; CVQA Local 57.4.”
        > “mSigLIP SO400M (WebLI 40B): IN 80.6; SLIP‑26 69.1; DC‑37 65.5; Babel‑IN 46.4; XM3600 I→T 62.8; CVQA Local 49.8.”
        > “SigLIP 2 SO400M (WebLI 40B): IN 83.2; SLIP‑26 73.7; DC‑37 69.4; Babel‑IN 40.8; XM3600 I→T 59.7; CVQA Local 49.0.”
    - SigLIP 2 prioritizes English (90% English data) and is worse than mSigLIP on some multilingual metrics and than Meta CLIP 2 on most English aggregates except raw ImageNet (Table 1 notes).
- Ablation: metadata and per‑language thresholds (Table 2)
  - Removing English‑only filtering but keeping English metadata degrades ImageNet slightly (Step 2: 66.9 vs. 67.5 with ViT‑B/32).
  - Merging all metadata without language isolation further hurts English while only partially enabling multilingual (Step 3).
  - Using language isolation but a single English threshold (`t_lang = t_en`) continues to depress English (Step 4).
  - Computing per‑language `t_lang` recovers both English and multilingual accuracy (Step 5: IN 64.7, Babel‑IN 31.5, XM3600 I→T 38.1, CVQA Local 46.6), demonstrating the need for language‑specific balancing.
- Ablation: tokenizer choice (Table 3)
  - XLM‑V (900k token vocab) achieves the best multilingual results while matching the best English result in the table:
    - Quote (Table 3):
      > “XLM‑V: IN 64.7; Babel‑IN 32.7; XM3600 T→I 40.0 / I→T 51.4; CVQA EN 50.4 / LOCAL 47.4.”
- Cultural diversity (Table 4; Fig. 3)
  - Simply switching from 13B English to 13B worldwide training (same seen pairs) improves geo‑diverse tasks:
    - GLDv2 jumps from 52.8 (English 13B) to 65.8 (Worldwide 13B), and to 69.0 with Worldwide 29B; Dollar Street and GeoDE also improve (Table 4).
  - Few‑shot geo‑localization on Dollar Street, GeoDE, XM3600 shows consistent gains with worldwide training and more seen pairs (Fig. 3).
- Embedding quality (Fig. 4)
  - On a 5k holdout, Meta CLIP 2 variants exhibit favorable alignment and uniformity (lower is better), indicating compact, semantically aligned embeddings without collapsing diversity.
  - Caveat noted: possible leakage for baselines cannot be controlled, but holdout was not used for training Meta CLIP 2 (Sec. 4.2.4).

- Overall assessment
  - The experiments are broad, use strong baselines, and include targeted ablations that isolate the effect of the three core design choices (metadata isolation, per‑language balancing, scaled seen pairs + capacity). The consistent pattern—capacity and scaled seen pairs are necessary and sufficient to break the curse—appears well supported (Fig. 1; Table 1–3).

## 6. Limitations and Trade-offs
- Assumptions and design choices
  - Fixed “tail proportion” invariance: The per‑language threshold `t_lang` is derived by matching the English tail proportion (Sec. 3.3). This assumes the optimal head–tail mix carries over to every language. Languages with very different web distributions may benefit from different tail proportions.
  - Reliance on LID accuracy: Misclassification of an alt‑text’s language routes it to the wrong metadata and threshold, potentially harming curation quality (Sec. 3.3).
- Data coverage and quality
  - Metadata quality varies with language resources. Low‑resource languages with sparse Wikipedia/WordNet coverage may be underrepresented in metadata, even though the algorithm does not filter them out (Sec. 3.2).
  - Native alt‑texts avoid translation bias but inherit web biases and noise; safety filtering and PII removal mitigate but do not eliminate this (Appendix A.2).
- Compute and scaling costs
  - To keep English exposure constant, batch size and seen pairs increase by ~2.3× (Table 6). Training `ViT‑H/14` with 29B seen pairs is compute‑intensive (Appendix B).
  - Worldwide curation is heavier than English‑only; the engineering (Aho‑Corasick, lazy loading, mmap) is necessary to keep it tractable (Appendix A.2).
- Model constraints
  - The curse persists for `ViT‑L/14` and for worldwide training without scaling seen pairs (Fig. 1; Table 1). Thus, benefits are contingent on both capacity and training exposure.

## 7. Implications and Future Directions
- How this changes the landscape
  - Demonstrates that multilingual scaling does not inherently hurt English performance; it requires balanced curation, sufficient seen pairs, and adequate capacity. This reframes multilingual CLIP training as an engineering and data‑recipe problem rather than a fundamental incompatibility (Fig. 1; Sec. 5).
  - Provides an open, reproducible recipe (metadata + curation + training) that avoids private data, teacher models, or translation, helping the community move beyond English‑centric CLIP (Sec. 1; Fig. 2).
- Practical applications
  - Stronger, more culturally aware vision encoders for:
    - Multimodal LLMs (plug‑in encoders).
    - Cross‑lingual retrieval (XM3600, XTD‑200 gains; Table 1).
    - Geo‑aware recognition and localization (Dollar Street, GLDv2, GeoDE; Table 4, Fig. 3).
    - Data curation for other paradigms (e.g., SSL like Web‑DINO) and image generation (Appendix; Sec. 1, bullets 5–6).
- Follow‑up research
  - Adaptive tail proportions: Learn or tune per‑language head–tail targets instead of fixing them from English.
  - Language‑aware schedules: Curriculum or sampling strategies that adapt over training or per language/domain.
  - Better LID and segmentation for mixed‑language alt‑texts, code‑switching, and dialects.
  - Richer metadata sources for low‑resource languages (beyond Wikipedia/WordNet), including community‑curated lexicons.
  - Model scaling studies: Find the next capacity “inflection point” and efficiency techniques (e.g., Mixture‑of‑Experts, parameter sharing) for worldwide training.
  - Broader, less Western‑centric benchmarks: The paper notes current multilingual/geo benchmarks still inherit biases and gaps (Appendix C). Building more representative evaluations will better reveal the benefits of worldwide training.

> Key takeaway: Meta CLIP 2 shows that with principled per‑language curation, scaled exposure, and sufficient capacity, one model can learn from worldwide multimodal web data to improve both English and multilingual performance—replacing the long‑assumed trade‑off with mutual gains (Fig. 1; Table 1–4).
