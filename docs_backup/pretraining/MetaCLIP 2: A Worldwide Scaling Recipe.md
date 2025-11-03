# MetaCLIP 2: A Worldwide Scaling Recipe

**ArXiv:** [2507.22062](https://arxiv.org/abs/2507.22062)
**Authors:** Yung‑Sung Chuang, Yang Li, Dong Wang, Ching‑Feng Yeh, Kehan Lyu, Ramya Raghavendra, James Glass, Lifei Huang, Jason Weston, Luke Zettlemoyer, Xinlei Chen, Zhuang Liu, Saining Xie, Wen‑tau Yih, Shang‑Wen Li, Hu Xu
**Institutions:** Meta, MIT, Princeton, NYU

## 🎯 Pitch

Meta CLIP 2 revolutionizes CLIP-style models by introducing a language-aware curation algorithm and scalable training framework, enabling efficient multilingual learning on web data across 300+ languages without translation or private data. This innovation not only breaks the curse of multilingual performance trade-offs but also enhances English accuracy, setting new benchmarks in multilingual vision-language tasks, thus democratizing access to culturally diverse, zero-shot classification and retrieval capabilities.

---

## 1. Executive Summary
Meta CLIP 2 provides the first end‑to‑end, publicly reproducible recipe for training CLIP-style image–text models from scratch on worldwide web data (English + 300+ non‑English languages). It introduces a language-aware curation algorithm, scaled metadata, and a training framework that together overcome the “curse of multilinguality” (English performance drop when adding other languages), yielding stronger English accuracy and new state-of-the-art on multilingual benchmarks without translation or private data.

## 2. Context and Motivation
- Problem gap
  - Modern CLIP models are mostly trained on English web pairs, leaving roughly half of the web’s non‑English content unused. As the paper notes, “about 44% of alt-texts are in English” in their crawl (Sec. 4.1), mirroring web statistics where ~50% of content is non-English (Intro).
  - Attempts at multilingual CLIP have struggled with two challenges (Intro):
    1) No open, scalable curation method for non‑English web data.
    2) “Curse of multilinguality”: multilingual training lowers English performance versus an English-only model (e.g., mSigLIP is −1.5% on ImageNet vs SigLIP; Challenge #2).

- Why this matters
  - Practical impact: Zero-shot classification, retrieval, and vision encoders for multimodal LLMs increasingly need culturally and linguistically diverse coverage. English-only data is nearing saturation (Intro; Villalobos et al., 2022).
  - Scientific impact: Demonstrates that multilinguality need not trade off with English performance if data curation, model capacity, and training are scaled coherently (Fig. 1 and Sec. 3.4).

- Prior approaches and shortcomings
  - Distillation/outsourcing: LAION filters via an existing CLIP; DFN uses a filter trained on private data; WebLI (powering mSigLIP/SigLIP 2) is built on Google proprietary pipelines (Sec. 2.1–2.3). These create opaque data distributions and often inherit teacher/translation biases.
  - Translation methods: Translate captions into English or vice versa, introducing synthetic bias and diverging from native-language supervision (Sec. 2.3).
  - Result: Existing multilingual CLIP models either underperform on English or prioritize English at the cost of multilingual quality (Sec. 2.3; Wang et al., 2025).

- Positioning
  - Meta CLIP 2 scales the “curation from scratch” philosophy to the worldwide web (Fig. 2), avoiding external teachers and translations. It maximizes overlap with vanilla CLIP architecture/settings to isolate what truly matters for multilingual scaling (Intro; Sec. 3).

## 3. Technical Approach
Meta CLIP 2 is a three-part recipe (Sec. 3; Fig. 2):

1) Worldwide metadata (Sec. 3.2)
- Goal: Create a high-quality, human-knowledge-derived concept list for many languages—analogous to the English metadata used in Meta CLIP.
- Sources (same four as English Meta CLIP, but multilingual):
  - Multilingual WordNet: all synsets from 31 languages.
  - Wikipedia unigrams and bigrams: processed from May 2024 dumps across 329 languages; for languages without spaces (e.g., Chinese, Thai), community tokenizers split words while preserving semantics (Appendix A.1, Table 5).
  - Wikipedia titles: collected from 40 snapshot dates; per-language titles ranked by click-through traffic.
- Design choices:
  - Maintain separate metadata per language to handle polysemy and language-specific forms (e.g., “mit” in English vs German). Ablation shows per-language metadata performs better than merged metadata (Sec. 4.2.2; Table 2, Steps 3–5).

2) Language-aware curation algorithm (Sec. 3.3; Algorithm 1)
- Problem: English Meta CLIP balances “head” vs “tail” concepts using a threshold `t` on per-concept match counts. Applying one global threshold to all languages would distort distributions because data volume differs by language.
- Mechanics (Algorithm 1; three stages):
  - Stage 1: Language identification (LID) assigns a language to each alt-text. Then, for that language’s metadata, substring match to find which metadata entries (concepts) appear in the alt-text; increment per-entry `entry_counts[lang]`.
    - Matching uses per-language metadata; if a language is unmapped, it falls into “other.”
  - Stage 2: Compute per-language thresholds `t_lang` so that the fraction of tail matches is invariant across languages.
    - Core idea: OpenAI CLIP used `t_en=20k`; Meta CLIP used `t_en=170k` to scale to 2.5B pairs while keeping ~6% of matches from tail concepts. Meta CLIP 2 preserves the same “tail match proportion” `p` across languages:
      - Derive `p = t_to_p(t_en, entry_counts["en"])` from English (Algorithm 1).
      - For each language, invert `p` to find `t_lang = p_to_t(p, entry_counts[lang])` such that the same tail proportion holds.
  - Stage 3: Sampling for balance.
    - Convert counts to probabilities. Tail entries (< `t_lang`) are kept with prob 1; head entries are downsampled with probability `t_lang / entry_counts[lang][entry_id]`. Each pair is accepted if any matched entry is sampled.
- Intuition:
  - Think of concepts as bins with uneven frequencies. The algorithm scales down head bins and preserves tail bins so the curated dataset covers long-tail concepts per language similarly to English, preventing head concepts in high-resource languages from drowning out the rest.
- Implementation scale-ups:
  - Efficient Aho–Corasick automata for substring matching, lazy per-language automaton loading, and memory-mapped counts to avoid OOM (Appendix A.2).

3) Training framework for worldwide data (Sec. 3.4)
- Multilingual tokenizer
  - Swap the English tokenizer for a multilingual one. Ablation (Table 3) shows the `XLM‑V` vocabulary performs best across English and multilingual benchmarks with ViT‑B/32.
- Scale “seen pairs” proportionally
  - Problem: If you keep the same total number of training examples (“seen pairs”) as English-only CLIP, English pairs get under-sampled once you add non-English data, hurting English accuracy.
  - Solution: Increase the global batch size so the total seen pairs scales with data growth, preserving the number of English pairs seen during training. In their data, English is ~44%, so they use a 2.3× increase (Sec. 3.4; Table 6: batch 75,366 vs 32,768; seen pairs 29B vs 12.8B).
- Minimal viable model capacity
  - Observation: Even with scaled data and seen pairs, ViT‑L/14 can still suffer the curse of multilinguality. ViT‑H/14 provides enough capacity to benefit both English and non-English simultaneously (Fig. 1; Sec. 3.4).

Putting it together:
- Data pipeline: Collect public web image–alt-text pairs; run LID; language-specific substring match to per-language metadata; compute `t_lang` via tail-proportion invariance; sample head/tail-balanced pairs (Algorithm 1). Remove NSFW and faces; deduplicate against ImageNet with 64-bit hash on projected embeddings (Appendix A.2).
- Training: Standard CLIP architecture and settings where possible (QuickGELU; same LR and warmup as Meta CLIP; Table 6), multilingual tokenizer, larger batch to achieve 2.3× seen pairs. Train ViT‑H/14 at 224px resolution.

## 4. Key Insights and Innovations
- Worldwide, language-aware curation from scratch (fundamental innovation)
  - What’s new: The first open, scalable curation algorithm that handles hundreds of languages without translation or external teacher models (Sec. 3.3; Algorithm 1). It balances concepts per language by keeping the tail-match proportion constant across languages.
  - Why it matters: Produces a controllable, transparent, long‑tail‑covering dataset per language that is not tied to a proprietary pipeline or a teacher’s biases (Sec. 2.1–2.3).

- Tail‑proportion invariance to compute per-language thresholds (conceptual insight)
  - What’s new: The threshold `t_lang` is not guessed or globally shared; it is derived so that each language exhibits the same fraction of tail matches as English (Algorithm 1: `t_to_p` and `p_to_t`).
  - Why it matters: Prevents head concepts in high-resource languages from dominating curation; ablations (Table 2, Steps 4→5) show per-language thresholds improve performance in both English and multilingual tasks.

- Scaling seen pairs rather than reweighting losses (practical training insight)
  - What’s new: Instead of inventing new loss weightings, Meta CLIP 2 preserves the number of English examples seen during training by proportionally increasing batch size/seen pairs (Sec. 3.4; Table 6).
  - Why it matters: Simple, architecture-agnostic, and empirically crucial. Without scaling seen pairs, the curse persists (Table 1: “Worldwide (1.0×)” underperforms).

- Capacity threshold for multilingual synergy (empirical insight)
  - What’s new: Identifies ViT‑H/14 as an inflection point where English accuracy improves when adding non-English data (Fig. 1; Table 1).
  - Why it matters: Explains why past multilingual attempts (often with smaller backbones) suffered English regressions. Capacity enables mutual benefit rather than trade-off.

## 5. Experimental Analysis
- Evaluation setup (Sec. 4.1–4.2)
  - Models:
    - CLIP variants at ViT‑L/14 and ViT‑H/14, 224px resolution.
    - Training on English-only, non-English-only, or worldwide curated data.
    - Seen pairs varied: 1.0× (≈13B), 1.3× (≈17B for non‑English), 2.3× (≈29B for worldwide).
  - Benchmarks and metrics:
    - English: ImageNet (IN val), SLIP‑26 average, DataComp‑37 average (Table 1).
    - Multilingual: Babel‑ImageNet (zero‑shot accuracy across 280 languages), XM3600 retrieval (Recall@1 for T→I and I→T), CVQA (English and local-language accuracies), Flickr30k‑200, XTD‑10/XTD‑200 retrieval (Table 1).
    - Cultural diversity/geo: Dollar Street, GLDv2, GeoDE zero‑shot and few‑shot (Table 4; Fig. 3).
    - Representation quality: Alignment and uniformity metrics on 5k holdout pairs (Fig. 4).
  - Baselines:
    - mSigLIP and SigLIP 2 (trained on Google’s WebLI, 40B seen pairs, 256px), and OpenCLIP LAION‑5B (Table 1). The paper flags these as “SoTA-aiming systems with confounding factors.”

- Main quantitative results (Table 1; Fig. 1)
  - Breaking the curse requires both capacity and scaled seen pairs:
    - ViT‑H/14, worldwide, 2.3× seen pairs:
      - ImageNet: 81.3% vs 80.5% for English-only Meta CLIP H/14 (gain +0.8%).
      - Babel‑ImageNet: 50.2% avg.
      - XM3600 retrieval: 51.5% (T→I) / 64.3% (I→T).
      - CVQA: 61.5% (EN) / 57.4% (LOCAL).
      - Flickr30k‑200: 50.9% / 53.2% (T→I / I→T).
      - XTD‑200: 48.9% / 51.0% (T→I / I→T).
    - Same model, worldwide but 1.0× seen pairs: lower across the board (e.g., IN 79.5%; XM3600 I→T 56.0%), demonstrating the necessity of scaling seen pairs (Table 1).
    - ViT‑L/14, worldwide 2.3×: still below English-only on ImageNet (78.8% vs 79.5%; Table 1), showing capacity limits.
  - Relative to mSigLIP / SigLIP 2:
    - Despite fewer seen pairs (29B vs 40B) and lower resolution (224 vs 256), Meta CLIP 2 H/14 worldwide 2.3× surpasses mSigLIP on IN, SLIP‑26, and DC‑37 and sets new highs on multilingual tasks (Table 1):
      - Babel‑ImageNet: +3.8% over mSigLIP SO400M.
      - XM3600 I→T: +1.5% over mSigLIP SO400M.
      - CVQA (LOCAL): +7.6% vs mSigLIP SO400M; CVQA (EN): +3.0%.
      - Flickr30k‑200 and XTD‑200: substantial gains (Table 1).

- Ablations and robustness
  - Metadata and curation design (Table 2, ViT‑B/32):
    - Stepwise transition from English-only to worldwide shows:
      - Simply removing the English filter (Step 2) hurts English (+0 multilingual coverage, IN drops from 67.5%→66.9%).
      - Merging all metadata without separation degrades further (Step 3: IN 62.1%) even as multilingual metrics appear.
      - Language isolation with a single `t_en` per language does not fix it (Step 4: IN 61.1%).
      - Per-language thresholds `t_lang` regain accuracy (Step 5: IN 64.7%; XM3600 improves; CVQA improves).
  - Tokenizer choice (Table 3, ViT‑B/32):
    - `XLM‑V` (900k vocab) yields the best combined English + multilingual performance (e.g., Babel‑IN 32.7%; XM3600 I→T 51.4%; CVQA LOCAL 47.4%).
  - Cultural diversity and geo‑localization (Table 4; Fig. 3):
    - Switching from English-only to worldwide data (same 13B seen pairs) boosts Dollar Street Top‑1 from 37.2%→37.2% (stable) but GLDv2 52.8%→65.8% and GeoDE 93.4%→94.3%. Scaling to 29B yields further gains (GLDv2 69.0%). Few-shot geo‑localization curves show consistent improvements with worldwide data (Fig. 3).
  - Embedding alignment/uniformity (Fig. 4):
    - Meta CLIP 2 models occupy favorable regions (lower is better on both axes), suggesting balanced semantic alignment and spread versus mSigLIP/SigLIP 2.

- Do the results support the claims?
  - Yes, they directly test the hypothesized levers:
    - Curation design (Table 2), tokenizer (Table 3), seen-pairs scaling and capacity (Table 1), and cultural coverage (Table 4, Fig. 3). The “curse” only breaks when both data curation is language-aware and training scales seen pairs with sufficient capacity (ViT‑H/14; Fig. 1 and Table 1).

## 6. Limitations and Trade-offs
- Dependence on LID and substring matching (Sec. 3.3; Appendix A.2)
  - If LID misclassifies an alt-text’s language, the wrong metadata is used for matching.
  - Substring matching, even with Aho–Corasick, can miss morphological variants or multiword concepts not captured in the metadata or tokenization rules; ambiguous strings may also cause false matches.
- Metadata coverage and quality (Sec. 3.2; Appendix A.1)
  - Metadata comes from WordNet (31 languages) and Wikipedia (329 languages). Languages with sparse Wikipedia coverage or missing WordNet may be underrepresented.
  - Tokenization for “scriptio continua” languages uses community tools (Table 5); quality depends on those tools.
- Compute and memory
  - Worldwide curation requires multi-language automata and large memory-mapped count arrays; training uses larger global batches to reach 29B seen pairs (Table 6). This raises compute cost versus English-only training.
- Capacity requirement
  - The paper identifies ViT‑H/14 as a practical minimum to break the curse under their setup (Fig. 1; Table 1). Smaller backbones (ViT‑L/14; ViT‑B/32) still show trade-offs.
- Benchmark limitations and bias
  - Appendix C discusses that widely used benchmarks are skewed toward Western content; even “diverse” datasets may inherit collection biases. Thus, absolute numbers may not reflect all regions/cultures equally.
- No architectural innovations by design
  - To isolate data/training effects, they keep CLIP architecture close to standard (Sec. 3). While this clarifies causality, specialized architectures might further improve certain tasks (e.g., dense features, localization).

## 7. Implications and Future Directions
- Field impact
  - Shifts the default from English-centric CLIP to worldwide CLIP trained from scratch with open, controllable curation. Demonstrates that multilingual training can improve English performance if the data/training recipe is right (Fig. 1; Table 1).
  - Provides a reusable, language-aware data pipeline that other modalities and tasks can adopt (Appendix A.2).

- What it enables
  - Multilingual MLLMs with native-language visual supervision (no translation), potentially improving instruction following and region-specific understanding (Intro; Sec. 2.2).
  - Stronger geo‑sensitive recognition and retrieval due to broader cultural coverage (Sec. 4.2.3; Table 4; Fig. 3).
  - Foundation for non-English-heavy domains (e.g., local e-commerce, non-Western social media, governmental archives).

- Practical applications
  - Zero-shot classification and retrieval across 300+ languages with one model (Table 1).
  - Downstream use as a vision encoder for multimodal LLMs, SSL pretraining on curated data (e.g., Web‑DINO on Meta CLIP data; Sec. 2.2), and image generation model conditioning (Intro).

- Research directions
  - Smarter matching beyond substrings: morphologically aware matching, multilingual entity linking, or learned matchers to reduce false positives/negatives while retaining transparency (build on Sec. 3.3).
  - Dynamic or data-driven tail proportions: learn or adapt `p` per language/domain instead of fixing it at ~6% (Algorithm 1).
  - Curriculum or mixture-of-experts to better utilize capacity on smaller backbones, easing compute demands while preserving multilingual gains (Sec. 3.4 observations).
  - Broader, less biased evaluations: community benchmarks that cover underrepresented regions, languages, and cultural practices (Appendix C).
  - Synergy with SSL: hybrid objectives to capture both semantic alignment and fine visual detail (Sec. 2.2 suggests complementarity).

> Key takeaway (Fig. 1 and Table 1): With language-aware curation, scaled seen pairs, and sufficient capacity (ViT‑H/14), adding non‑English web data ceases to hurt and instead helps English performance, while simultaneously delivering state-of-the-art multilingual results—all without private data, translation, or distillation.
