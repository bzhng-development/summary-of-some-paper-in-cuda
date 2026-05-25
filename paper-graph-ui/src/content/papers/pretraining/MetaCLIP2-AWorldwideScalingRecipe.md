# Meta CLIP 2: A Worldwide Scaling Recipe

**ArXiv:** [2507.22062](https://arxiv.org/abs/2507.22062)

## 🎯 Pitch

Meta CLIP 2 delivers the first fully open and scalable method to train CLIP vision–language models directly on worldwide, multilingual web data—solving the long-standing challenge where adding non-English data degraded English performance (the 'curse of multilinguality'). By introducing per-language metadata, scalable curation, and carefully matched model capacity, it achieves state-of-the-art results across both English and multilingual tasks using standard CLIP architectures. This breakthrough paves the way for future foundation models that natively understand a truly global, culturally rich web, and ensures continued progress as English data becomes saturated.

---

## 1. Executive Summary

Meta CLIP 2 introduces the first end-to-end recipe for training CLIP from scratch on native worldwide image-text pairs—without relying on machine translation, distillation, or private data—and studies how to overcome the **curse of multilinguality** (the phenomenon where multilingual CLIP underperforms its English-only counterpart, e.g., mSigLIP lagging SigLIP by 1.5% on ImageNet) through jointly scaled metadata, curation, and model capacity. The recipe extends English Meta CLIP’s curation algorithm with per-language substring matching and balancing (computing language-specific head/tail thresholds `t_lang` to preserve a consistent 6% tail-match proportion across 329 languages), then couples this with proportional scaling of seen training pairs (2.3× increase in global batch size to match the added non-English data volume), establishing that a ViT-H/14 with sufficient expressivity breaks the curse entirely—non-English data helps English CLIP rise from 80.5% to 81.3% on ImageNet, while English data simultaneously pushes multilingual benchmarks to new state-of-the-art (XM3600 image-to-text retrieval at 64.3%, Babel-ImageNet at 50.2%, CVQA at 57.4%), an inflection point that fails to materialize at ViT-L/14 scale.

## 2. Context and Motivation

### The Core Problem: CLIP Cannot Digest the Non-English World

CLIP (Radford et al., 2021) has become a foundational building block of modern vision and multimodal AI — it powers zero-shot image classification, cross-modal retrieval, and serves as the vision encoder in multimodal large language models like LLaMA 3, Gemini, and Qwen-VL. However, almost all widely-used CLIP variants share a critical architectural assumption: **they only learn from English text**. This is not a minor limitation. As the paper notes (Section 1), approximately **50.9% of web content is non-English** — a figure that is growing as Internet access expands globally. By design, English-only CLIP throws away half the world's image-text pairs before training even begins.

This matters for three concrete reasons that the paper develops throughout its motivation:

**First, English Internet data is finite and may be exhausted soon.** The paper explicitly cites Villalobos et al. (2022)'s projection that high-quality English text data could be depleted within this decade. When that happens, the only path to further scaling is to incorporate non-English data. If we lack the methods to do so effectively, the entire trajectory of CLIP-based models hits a data wall.

**Second, English-only CLIP encodes cultural blindness.** Training exclusively on English alt-texts means the model never sees concepts described in their native cultural context — a dish described in Vietnamese, a landmark captioned in Arabic, a garment labeled in Hindi. This produces models that perform poorly on region-specific recognition tasks and fail to serve non-English-speaking populations. The paper frames this as both a performance problem and an equity problem: a "foundation" model that fails for most of the world's population is foundationally incomplete.

**Third, the practical cost of maintaining separate models is high.** If multilingual CLIP underperforms English CLIP on English tasks (the curse of multilinguality), organizations must deploy *two separate models* — one optimized for English, one for multilingual — doubling inference infrastructure, maintenance burden, and model serving complexity. The paper's goal of a single model that is simultaneously best-in-class on both English and multilingual benchmarks is motivated as much by deployment pragmatics as by scientific curiosity.

The paper frames this entire situation as the **worldwide scaling challenge**: the set of unsolved problems preventing CLIP from learning natively from the global web, as opposed to the curated English-only slice that current methods assume.

### Prior Approaches and Their Shortcomings

The paper identifies three families of prior attempts to build multilingual vision-language models, each with fundamental limitations that Meta CLIP 2 is designed to address.

#### Approach 1: Distillation from English CLIP (No Native Multilingual Learning)

The earliest multilingual CLIP efforts sidestep the problem of curating non-English data entirely. M-CLIP (Carlsson et al., 2022) and mCLIP (Chen et al., 2023a) take an existing English CLIP model as a frozen vision encoder and train only a multilingual text encoder — typically on low-quality, automatically-paired multilingual image-text data. The vision backbone never sees non-English captions paired with its own training images; it merely inherits whatever visual representations the English CLIP learned.

Succinctly: these approaches teach the model to *translate* its English visual knowledge into other languages, rather than to *learn* visual concepts from non-English descriptions in the first place. The paper identifies three specific failure modes:

- **Teacher bias**: The visual representations are permanently constrained by what English CLIP considered important. A concept that is visually salient in a non-English culture but rarely described in English web content (e.g., a specific regional food or traditional garment) may be poorly represented or missing entirely, and no amount of multilingual text training can recover it — the vision encoder is frozen.

- **Translation artifacts**: The multilingual text data used is often machine-translated or low-quality, introducing linguistic errors that the model learns to associate with visual concepts. A Vietnamese caption that mistranslates a dish name teaches the model an incorrect association.

- **No scaling path**: Since the vision encoder never improves from non-English data, adding more languages or more multilingual pairs only trains the text tower to better map onto a fixed visual representation space. The fundamental capacity ceiling is set by the English teacher.

The paper frames distillation as a pragmatic stopgap, not a solution to worldwide scaling. It works when no native multilingual training pipeline exists, but it cannot deliver the mutual benefits (English improving non-English performance and vice versa) that native training enables.

#### Approach 2: Machine Translation as a Proxy (The Synthetic Data Trap)

A second class of approaches avoids native multilingual curation by translating everything into English (or translating English into other languages). Representative works include Santos et al. (2023), Nguyen et al. (2024), and Pouget et al. (2024). The pipeline is straightforward: take non-English alt-texts, machine-translate them to English, and then train a standard English CLIP on the translated pairs.

The paper identifies two deep problems with this strategy:

**Loss of native speaker signal.** When a Vietnamese speaker writes an alt-text describing a bowl of phở, the specific word choices, modifiers, and cultural connotations are products of their lived experience. Machine translation flattens this into a generic English description, discarding the very cultural information that makes the original caption valuable. The paper explicitly advocates for **native-language supervision** — the model learning directly from text written by native speakers — as a goal that translation-based approaches fundamentally cannot achieve.

**Translation errors compound in the vision-language mapping.** Machine translation systems make mistakes, particularly on rare words, culturally specific terms, and non-standard grammar — precisely the kinds of text that are most valuable for learning diverse visual concepts. If "bánh chưng" (a traditional Vietnamese rice cake) is mistranslated as "rice cake," the model loses the ability to distinguish it from Japanese mochi, Korean tteok, or any other rice-based food. These errors propagate into the visual representations.

More subtly, translation-based approaches implicitly assume that conceptual categories are language-universal — that the English word for something captures its visual essence equally well for all cultures. This is false. A concept that requires a compound phrase in English might be a single word in its native language, or vice versa. Translating everything into English imposes an Anglophone conceptual grid on fundamentally non-Anglophone visual categories.

#### Approach 3: Private Data Pipelines on Undisclosed Data (The Reproducibility Crisis)

The strongest published multilingual CLIP results come from Google's SigLIP family (Zhai et al., 2023; Tschannen et al., 2025), trained on the WebLI dataset (Chen et al., 2023b). WebLI is derived from Google Image Search (Juan et al., 2019), making it both private and built on a proprietary data processing pipeline whose details are not publicly disclosed. The paper identifies this as a fundamental barrier to scientific progress:

> "mSigLIP... is built from Google Image Search and thus private data, making the scalable recipe undisclosed."

This is not merely a complaint about closed-source models. The paper argues that without a transparent, reproducible curation algorithm, the community cannot study *why* multilingual CLIP behaves as it does. When mSigLIP outperforms other approaches, is it because of better curation, more data, better filtering, different loss functions (SigLIP uses sigmoid loss instead of contrastive loss), different architectures, or some interaction of these factors? The opacity of the pipeline makes controlled experimentation impossible.

Moreover, even WebLI-based approaches exhibit the **curse of multilinguality**. mSigLIP is 1.5% worse than its English-only counterpart SigLIP on ImageNet, despite having access to vastly more training data. The authors of SigLIP 2 (Tschannen et al., 2025) responded to this by making their training data **90% English** — essentially retreating from multilingual training to preserve English performance, at the cost of even worse multilingual results. This is the paper's Exhibit A for why the curse is a real and unsolved problem, not merely an artifact of insufficient data.

Wang et al. (2025) provide further evidence of this difficulty: when scaling WebLI from 10B to 100B raw pairs, they observed *mixed* results on English benchmarks. Simply having more multilingual data is not sufficient — the curation and training methodology must be specifically designed for the worldwide setting.

#### The Deeper Issue: No Curation Method Exists for Non-English Data

The paper argues that all three prior approaches share a hidden common failure: **none of them have solved the fundamental data curation problem for non-English text**. To understand why this matters, we need to understand what curation does in the English CLIP setting.

**What Meta CLIP's curation algorithm does (and why it's essential).** Meta CLIP (Xu et al., 2024) formalized OpenAI CLIP's implicit curation guidance into an explicit algorithm. The core idea: raw Internet image-text pairs have a wildly unbalanced distribution of visual concepts — a few common concepts (like "car" or "dog") appear millions of times, while the vast majority of concepts appear rarely. Training directly on this raw distribution produces models that overfit to head concepts and fail on tail ones. Meta CLIP's curation uses metadata — a list of ~500k high-quality visual concepts extracted from English WordNet and English Wikipedia — to count how often each concept appears in the training data, then **downsamples pairs containing head concepts** to create a balanced distribution where tail concepts receive proportionally more training signal.

This algorithm has several critical properties:
- It is **model-free**: no black-box CLIP filter decides what to keep (unlike LAION or DFN).
- It is **transparent**: the metadata and balancing logic are fully inspectable and controllable.
- It **scales**: Meta CLIP demonstrated that raising the head/tail threshold from 20k (OpenAI's 400M-pair setting) to 170k allowed scaling to 2.5B pairs while maintaining the same 6% tail-match proportion.

**Why this breaks for non-English data.** Every component of this curation pipeline assumes English:

1. **Metadata**: The concept vocabulary comes from English WordNet (synsets defined in English) and English Wikipedia (titles, unigrams, bigrams extracted from English text). There is no equivalent metadata for Vietnamese, Swahili, or Quechua. You cannot run substring matching with English metadata on a Vietnamese alt-text and expect to find the right concepts — "chó" won't match "dog," and even if it did, the cultural framing of the concept is lost.

2. **Substring matching**: The matching algorithm searches for English concept strings within English alt-texts. Non-English text doesn't contain English concept strings (mostly). Running the same algorithm on worldwide data without language-aware metadata simply fails to match anything, treating all non-English pairs as having no recognized concepts — which either discards them entirely or includes them without any balancing.

3. **Threshold tuning**: The head/tail threshold `t` that balances the distribution depends on the size of the data pool. English data is 44% of the worldwide pool; other languages range from large (dozens of languages with millions of pairs) to tiny (hundreds of languages with minimal representation). A single global threshold `t` would be far too high for small languages (making almost everything appear as head data and getting aggressively downsampled) or far too low for large languages (failing to downsample head concepts adequately). The paper explicitly demonstrates this failure mode in Table 2: using a single `t_en` across all languages yields an accuracy drop on ImageNet compared to language-specific thresholds.

**What happens when you skip curation entirely.** The paper's ablation in Table 2 provides direct evidence: if you simply remove the English language filter from Meta CLIP's pipeline and let English metadata try to match non-English alt-texts, ImageNet accuracy drops from 67.5% to 66.9%. If you go further and merge all metadata into one set (no language isolation), accuracy plummets to 62.1%. Uncurated multilingual training is not just ineffective — it is actively harmful to English performance because the model wastes capacity trying to learn from noisy, poorly-matched concept signals.

This is the gap Meta CLIP 2 fills: **a principled, transparent, model-free curation algorithm that extends the English Meta CLIP approach to 300+ languages**, with per-language metadata, per-language substring matching, and a mathematically derived method for computing language-specific head/tail thresholds that preserve a consistent tail-concept proportion across all languages. The curation algorithm is the first innovation; the training framework (scaling seen pairs, studying minimal viable model capacity) is the second; together they constitute the first complete recipe for worldwide CLIP training from scratch.

### The Curse of Multilinguality: A Problem Inherited from LLMs

The paper explicitly frames the English performance degradation observed in multilingual CLIP as an instance of the **curse of multilinguality**, a phenomenon well-documented in text-only large language models. When an LLM is trained on multiple languages with a fixed model capacity, performance on any single language (including English) typically degrades compared to a monolingual model of the same size — the model's representational capacity is diluted across languages, and cross-lingual interference can hurt high-resource language performance.

For CLIP, this manifests with a concrete, measurable penalty: mSigLIP loses 1.5% on ImageNet versus its English-only SigLIP counterpart, despite having more total training data. SigLIP 2's response — making data 90% English — trades away multilingual performance to recover English performance, confirming that the curse is a fundamental capacity allocation problem, not merely a data quantity problem.

The paper's key hypothesis, tested through the ViT-L/14 vs. ViT-H/14 scaling experiment in Figure 1, is that **the curse of multilinguality in CLIP is not inevitable** — it is a consequence of insufficient model capacity relative to the data diversity introduced by multilingual training. When the model is too small (ViT-L/14, even with properly scaled training), the additional languages compete for limited representational capacity, crowding out English-specific features. When the model is large enough (ViT-H/14 with 2.3× scaled seen pairs), the additional languages provide complementary visual signal that actually improves English performance — English accuracy rises, rather than falls, from adding non-English data.

This is the paper's central empirical claim and its most important theoretical contribution: the curse of multilinguality has an **inflection point** in model scale. Below that point, multilingual data hurts; above it, multilingual data helps — and helps *both* English and non-English performance simultaneously.

### How This Paper Positions Itself

Meta CLIP 2 positions itself not as a state-of-the-art system paper (which would combine every available technique to maximize benchmark numbers) but as a **recipe paper** — a systematic study of the minimal, necessary changes required to extend CLIP to worldwide data, with all other variables held constant to ensure findings are generalizable. The paper makes this distinction explicit:

> "The overlap makes our findings generalizable to CLIP and its variants, compared to system works aiming at state-of-the-art performance with combination of all available techniques."

This is a deliberate methodological choice. System papers like SigLIP and SigLIP 2 change many things simultaneously: loss function (contrastive → sigmoid), data source (public web → Google Image Search), architecture (ViT → SO400M), resolution (224 → 256), and training recipe. When performance changes, it is impossible to attribute the change to any specific factor. Meta CLIP 2, by contrast, changes only what is necessary for worldwide capability (metadata, curation algorithm, tokenizer, seen pairs) while keeping everything else — architecture, loss function, optimizer, learning rate schedule, data source type — identical to OpenAI/Meta CLIP settings.

This design philosophy serves three purposes:

1. **Scientific clarity**: The paper can attribute observed effects (e.g., breaking the curse of multilinguality at ViT-H/14) to specific recipe changes, because confounds are minimized.
2. **Reproducibility**: By open-sourcing the metadata, curation code, and training recipe, the paper enables other researchers to replicate and extend the findings on their own data and model architectures.
3. **Community adoption**: By showing that the recipe works with standard CLIP architecture and training settings, the paper maximizes the likelihood that practitioners will adopt the approach — they don't need to switch to a new loss function or data pipeline, just extend their existing one.

The paper also positions itself relative to the **exhaustion of English data** as a temporal forcing function:

> "Achieving such worldwide scaling is highly desirable, especially when English Internet data is exhausted soon."

This is not presented as a speculative future concern but as a near-term practical reality that gives the work urgency. If English data is depleting, the community *must* learn to use non-English data effectively — not as a nice-to-have, but as the only path to continued scaling. Meta CLIP 2 is presented as the first principled answer to that imperative.

### The Necessary and Sufficient Conditions for Success

The paper's ablation structure (Tables 1 and 2, Figures 1 and 3) is designed to isolate which changes are *necessary* to break the curse of multilinguality and which are merely *helpful*. The findings can be summarized as a chain of necessity:

- **Language-specific metadata and curation are necessary but not sufficient.** Table 2 shows that even with optimal per-language curation (row 5), ViT-B/32 still suffers from the curse — multilingual benchmarks improve but English drops. Curation solves the data quality problem but not the capacity problem.

- **Scaling seen pairs proportionally is necessary.** Table 1 shows that training ViT-H/14 on worldwide data with English-scale seen pairs (Worldwide 1.0×) yields 79.5% ImageNet accuracy — worse than English-only (80.4%). The model sees fewer English examples because non-English pairs consume training slots. Scaling seen pairs to 2.3× (Worldwide 2.3×) restores and exceeds English-only performance (81.3%). This is because the 2.3× factor preserves the absolute number of English examples seen during training, ensuring English learning is not diluted.

- **Sufficient model capacity (ViT-H/14) is necessary.** Figure 1 shows that ViT-L/14 with full recipe (2.3× seen pairs, language-specific curation) still drops from 79.5% (English-only) to 78.8% (worldwide) on ImageNet. The curse persists at L scale. Only at H scale does the relationship flip: worldwide data helps English. The paper interprets this as evidence that ViT-H/14 crosses a representational capacity threshold where the model has enough parameters to learn complementary features from multilingual data rather than suffering from representational interference.

- **All three together are jointly sufficient.** Worldwide data + 2.3× seen pairs + ViT-H/14 → 81.3% ImageNet (above 80.4% English-only) + state-of-the-art multilingual benchmarks. Remove any one component and either English performance degrades (remove ViT-H/14 or scaled pairs) or multilingual performance collapses (remove curation).

This decomposition is the paper's primary intellectual contribution beyond the recipe itself: it provides a **theory of what causes the curse of multilinguality in CLIP** (insufficient curation + capacity dilution + reduced English exposure) and an **empirically validated set of interventions** that eliminate it. The finding that the three factors interact — that ViT-L/14 fails even with good curation and scaled pairs, while ViT-H/14 succeeds only with both — underscores why prior work struggled: addressing any subset of the causes was insufficient to break the curse.

## 3. Technical Approach

### 3.1 Reader Orientation (Approachable Technical Breakdown)

Meta CLIP 2 is a **data curation and training recipe** — not a new model architecture or loss function — that extends the existing English-only Meta CLIP pipeline to handle image-text pairs from over 300 languages. The system takes raw, noisy image-text pairs scraped from the worldwide web, curates them into a balanced training distribution where rare visual concepts from any language get sufficient learning signal, and then trains a standard CLIP model (ViT-H/14) with scaled-up throughput so that adding non-English data **helps** English performance rather than hurting it. The core problem it solves is the **curse of multilinguality** — the consistent finding that multilingual CLIP underperforms English-only CLIP on English benchmarks — and the shape of the solution is a three-part intervention: construct per-language metadata to enable concept-aware curation in any language, compute language-specific head/tail thresholds to preserve tail-concept diversity across heterogeneous data pools, and proportionally scale seen training pairs so that English examples are not diluted when non-English data joins the batch.

### 3.2 Big-Picture Architecture (Diagram in Words)

The Meta CLIP 2 pipeline has four major components operating sequentially, with the first three replacing their English-only counterparts from the original Meta CLIP and the fourth being a training-scale adjustment:

1. **Worldwide Metadata Construction (Section 3.2):** Given Wikipedia dumps in 329 languages and multilingual WordNet covering 31 languages, this component extracts cleaned unigrams, bigrams, and page titles per language, merges them with WordNet synsets, and produces a dictionary `M` where each key is a language code (from the language identification system's output space) and each value is a deduplicated list of high-quality visual concept strings in that language. This is the "vocabulary of what to look for" during curation — one vocabulary per language, built from the same four source types (WordNet synsets, Wikipedia unigrams, Wikipedia bigrams, Wikipedia titles) that English Meta CLIP used.

2. **Worldwide Curation Algorithm (Section 3.3, Algorithm 1):** Given raw image-text pairs `D` and the worldwide metadata `M`, this component (a) runs language identification on every alt-text, (b) performs per-language substring matching using the corresponding metadata vocabulary (e.g., Vietnamese alt-texts are matched against the Vietnamese metadata list), (c) aggregates global match counts per concept per language, (d) computes language-specific head/tail thresholds `t_lang` via a two-step procedure that enforces a consistent 6% tail-match proportion across all languages, (e) assigns a sampling probability to each pair based on its matched concepts' head/tail status, and (f) probabilistically selects pairs to produce the curated dataset `D*`. The output is a balanced training set where tail concepts from any language are preserved.

3. **Multilingual Tokenizer Replacement:** The English-only GPT-2 tokenizer used by OpenAI/Meta CLIP is swapped for a multilingual tokenizer (XLM-V with 900k vocabulary entries is selected after ablation) that can encode text from all training languages without excessive fragmentation. This is the only architectural change to the model itself.

4. **Scaled Training Framework (Section 3.4):** The curated worldwide dataset `D*` is fed into standard CLIP contrastive training with two adjustments: (a) the global batch size is multiplied by 2.3× (from 32,768 to 75,366), proportionally scaling the number of seen pairs from 12.8B to 29B so that the absolute number of English examples seen during training remains unchanged despite English pairs constituting only ~44% of the worldwide data, and (b) the model capacity is increased to ViT-H/14, which the paper identifies as the minimum viable scale for breaking the curse of multilinguality.

Information flows linearly: raw web pairs → language identification → per-language substring matching → global count aggregation → language-specific threshold computation → probabilistic pair sampling → tokenization with multilingual tokenizer → contrastive training with scaled batch size → trained CLIP model. There is no feedback loop, no iterative refinement, and no dependence on external models or translations at any stage.

### 3.3 Roadmap for the Deep Dive

- **First, the foundation: English Meta CLIP revisited (Section 3.1).** I will explain the original English-only curation algorithm in concrete detail — what metadata is, how substring matching works, what the head/tail threshold does, and why balancing matters. This is essential because Meta CLIP 2's worldwide algorithm is a direct generalization, and understanding the English base case makes the extensions self-evident rather than mysterious.

- **Second, worldwide metadata construction (Section 3.2).** I will walk through how the four metadata sources (WordNet, Wikipedia unigrams, bigrams, titles) are scaled from English-only to 300+ languages, which languages are covered by which sources, how text is cleaned and tokenized (including special handling for languages without space-delimited words), and how the per-language metadata dictionary `M` is structured and keyed. This establishes *what* the curation algorithm will match against.

- **Third, the worldwide curation algorithm (Section 3.3, Algorithm 1).** This is the longest subsection because Algorithm 1 encapsulates the core intellectual contribution. I will step through the three stages (matching, threshold computation, sampling) with explicit discussion of the pseudocode, explaining the `t_to_p` and `p_to_t` functions, the invariance assumption (6% tail proportion), the language-specific threshold derivation, and why a single global threshold fails. I will also cover the engineering optimizations (Aho-Corasick matching, lazy metadata loading, memory-mapped probability files) that make this feasible at billion-pair scale.

- **Fourth, the training framework modifications (Section 3.4).** I will explain the two training-side interventions: scaling seen pairs proportionally to data growth (the 2.3× factor, how it's implemented as a batch size increase, and why it's necessary) and studying minimal viable model capacity (why ViT-L/14 fails even with good curation, why ViT-H/14 succeeds, and what this implies about representational capacity thresholds).

- **Fifth, design choice justifications across the pipeline.** I will synthesize the "why" behind key decisions — per-language vs. merged metadata, language-specific vs. global thresholds, 2.3× vs. other scaling factors, XLM-V vs. other tokenizers — referencing the ablation results in Tables 2 and 3 to ground each choice in empirical evidence.

### 3.4 Detailed, Sentence-Based Technical Breakdown

This is primarily a **data curation and training methodology paper** whose core idea is that the curse of multilinguality in CLIP arises from three jointly insufficient conditions — (1) lack of language-aware concept curation for non-English data, (2) dilution of English training examples when non-English data joins the batch, and (3) inadequate model capacity to learn complementary features from multilingual data — and that addressing all three simultaneously eliminates the curse and enables mutual benefits between English and non-English training data.

---

#### 3.4.1 English Meta CLIP Algorithm Revisited (The Foundation)

Before we can understand the worldwide extensions, we must understand exactly what the English-only Meta CLIP algorithm does, because Meta CLIP 2's worldwide algorithm is a direct generalization of every step. The original Meta CLIP (Xu et al., 2024) formalized the data curation process that OpenAI CLIP (Radford et al., 2021) described only in high-level prose, making it reproducible and scalable. The algorithm takes raw image-text pairs from the web — a distribution that is wildly imbalanced, with a few common concepts appearing millions of times and the vast majority of concepts appearing rarely — and transforms them into a **balanced training distribution** where tail (rare) concepts receive proportional training signal.

**Metadata construction.** The algorithm first constructs a list `M` of approximately 500,000 high-quality visual concept strings, drawn from four sources written by human experts: (1) all English WordNet synsets (WordNet is a lexical database that organizes English words into sets of synonyms called synsets, each representing a distinct concept — for example, the synset {dog, domestic dog, Canis familiaris} represents the concept of a dog as an animal), (2) English Wikipedia unigrams (single words extracted from Wikipedia text, frequency-filtered), (3) English Wikipedia bigrams (two-word phrases), and (4) English Wikipedia page titles. These sources are combined and deduplicated to produce a single flat list of concept strings. The key property: these are concepts that **human experts have deemed visually meaningful**, not concepts automatically extracted by a model. The metadata serves as a vocabulary of "what to look for" in alt-texts.

**Substring matching.** For each image-text pair in the raw data pool `D`, the algorithm performs substring matching: it checks whether each concept string in `M` appears as a contiguous substring within the alt-text. For example, the alt-text "a golden retriever playing in the park" would match the metadata entries "golden retriever" (if it's in the bigram list), "retriever," "playing," "park," and possibly "golden" (if they exist as entries). Each matched entry's ID is recorded. The algorithm does not use semantic matching, embedding similarity, or any learned model — it is pure deterministic string search, making it fast, transparent, and model-free.

**Global counting.** After processing all pairs, the algorithm aggregates match counts: for each metadata entry `e`, it counts how many image-text pairs in the entire data pool `D` contain `e` as a substring in their alt-text. This produces `entry_count[e]` — the raw frequency of each visual concept in the training data. In the raw web, a few entries (like "car," "dog," "person") have counts in the millions, while most entries (like "axolotl," "parhelion," "sousaphone") have counts in the tens or hundreds. Training directly on this distribution would mean the model sees "car" millions of times for every one time it sees "axolotl," leading to catastrophic overfitting to head concepts and near-zero learning on tail concepts.

**Balancing via threshold.** To correct this imbalance, the algorithm introduces a threshold `t`. Any metadata entry with `entry_count[e] < t` is defined as a **tail entry** (or tail concept); any entry with `entry_count[e] ≥ t` is a **head entry**. The algorithm then computes a sampling probability for each entry:

- For tail entries: `entry_prob[e] = 1.0` (always keep pairs containing this concept).
- For head entries: `entry_prob[e] = t / entry_count[e]` (downsample proportionally to how far above the threshold they are).

For each image-text pair, the algorithm looks at all the metadata entries matched in its alt-text, takes the maximum `entry_prob` among them (or, in Meta CLIP's actual implementation, samples based on a per-entry independent sampling scheme), and decides whether to include the pair in the curated dataset `D*`. The effect: pairs containing only common concepts are aggressively downsampled, while pairs containing at least one rare concept are kept with high probability.

**The threshold `t` and the 6% tail proportion invariant.** The threshold `t` is the single most important hyperparameter. OpenAI CLIP used `t = 20,000` for a raw data pool of approximately 400 million pairs. Meta CLIP scaled this to `t = 170,000` for a pool of approximately 2.5 billion pairs. The paper notes that both settings produce an important invariant: approximately **6% of all substring matches** in the curated dataset come from tail entries. In other words, `t` is chosen so that the curated training data's concept distribution has 6% of its mass on rare concepts — enough to ensure they get learned, but not so much that head concepts are starved of training signal.

Formally, the tail proportion `p` for a given threshold `t` and a set of entry counts is:

$$p = \frac{\sum_{e: \text{entry\_count}[e] < t} \text{entry\_count}[e]}{\sum_{e} \text{entry\_count}[e]}$$

where the numerator sums match counts for all entries with counts below `t` (tail entries), and the denominator sums match counts for all entries head and tail.

**What it computes:** the fraction of total concept-occurrences in the raw data that belong to concepts appearing fewer than `t` times. This is a distributional statistic — it tells you what proportion of all training signal would come from rare concepts at this threshold.

**Why this form:** the ratio isolates the mass of the long tail of the concept frequency distribution. If `t` is too low, `p` is near zero (almost everything is head data, tail concepts are ignored). If `t` is too high, `p` approaches 1 (everything is treated as tail, no downsampling occurs, and head concepts dominate training). The 6% value was empirically discovered by OpenAI and validated by Meta CLIP as providing good head-tail balance across different data scales.

When scaling to a larger data pool, `t` must increase to maintain the same `p`, because a larger pool naturally has higher absolute counts for the same concepts. Meta CLIP's tuning from `t = 20k` to `t = 170k` preserved the 6% tail proportion across a 6.25× increase in data volume.

---

#### 3.4.2 Worldwide Metadata Construction

The first worldwide extension: **replace the English-only metadata list with a per-language metadata dictionary**. The paper's key design choice is to maintain **independent metadata per language** rather than merging all languages into a single list. The justification is both practical and linguistic: the same string "mit" is a word in English (a conjunction) and German (preposition meaning "with"), and merging them would conflate unrelated concepts. Per-language isolation preserves semantic integrity and, as shown in Table 2, yields better performance than merged metadata (62.1% ImageNet for merged vs. 64.7% for per-language isolated, both on ViT-B/32).

The metadata sources mirror English Meta CLIP's four sources but are extended across languages:

**Source 1: Multilingual WordNet.** The paper includes all synsets from **31 languages** available in the multilingual WordNet project. WordNet is not available for all 300+ languages — it exists primarily for higher-resource languages — so this source covers a subset. Each synset in each language contributes concept strings in that language (e.g., the synset for "dog" contributes "perro" in Spanish, "chien" in French, "собака" in Russian, each as separate entries keyed to their respective language codes).

**Sources 2 and 3: Wikipedia Unigrams and Bigrams.** The paper processes Wikipedia dumps from **May 2024** covering **329 languages**. The processing pipeline for each language:

1. **Text extraction:** WikiExtractor (Attardi, 2015) is used to strip Wikipedia markup and extract plain text from the dumps. This removes formatting, links, templates, and metadata, leaving only the article body text.

2. **Tokenization:** For most languages (those using space-separated writing systems), the text is split on spaces and punctuation to obtain individual words. For languages that do not use spaces between words — a phenomenon known as *scriptio continua*, common in several Asian languages — the paper uses **open-source tokenizers developed by local communities** to properly split text while preserving semantic units. Table 5 in Appendix A lists the specific tokenizers: Tibetan (for wiki codes `bo`, `dz`), Japanese (for `ja`, `ryu`), Khmer (`km`), Lao (`lo`), Myanmar (`my`), Thai (`th`), and Chinese (for `zh`, `zh_classical`, `zh_yue`). Without these tokenizers, Chinese text would be split into individual characters rather than words, destroying the semantic content of unigrams and bigrams.

3. **Frequency counting:** After tokenization, the algorithm counts occurrences of each unigram (single token) and bigram (adjacent token pair) across all articles for that language. These frequency-counted n-grams become the metadata entries for that language.

4. **Filtering:** While not explicitly detailed, the paper implies standard frequency-based filtering analogous to English Meta CLIP: extremely rare or extremely common n-grams are likely filtered to maintain metadata quality, though exact thresholds are not specified.

**Key linguistic nuance:** the tokenizers in Table 5 are **only used for Wikipedia dump processing** — they are not applied to alt-texts during substring matching. Alt-texts are matched as raw strings against the metadata entries, regardless of language. This means the metadata entries must be in the same surface form that appears in alt-texts.

**Source 4: Wikipedia Titles.** For each language, the paper collects page titles from **40 random dates of Wikipedia snapshots** and ranks them by **click-through traffic**. This captures which articles people actually look at, providing a signal of cultural relevance — a page about a popular local dish will have higher traffic than a page about an obscure historical footnote. The ranked titles become metadata entries for that language.

**Metadata dictionary structure.** The output is a dictionary `M` where:
- **Keys** are language codes from the language identification (LID) system's output space.
- **Values** are lists of concept strings for that language, merging entries from all four sources that map to that language code.

**Language mapping between LID and metadata sources.** The paper notes that the sets of languages covered by the LID system and the metadata sources are usually different — for instance, WordNet covers 31 languages, Wikipedia covers 329, and the LID system (fastText-based, from Grave et al., 2018) covers 157. To handle this, the algorithm first establishes a mapping from each LID language code to one or more metadata language codes. Metadata languages that map to the same LID language are **merged** into a single metadata list for that LID key. For metadata languages that cannot be mapped to any LID language (e.g., a Wikipedia dump in a language the LID system doesn't recognize), their metadata is collected under a special key `"other"`. This ensures no metadata is discarded due to LID coverage gaps, though in practice the `"other"` category likely serves a tiny fraction of pairs.

**This structured dictionary is what feeds into Algorithm 1's substring matching stage.**

---

#### 3.4.3 The Worldwide Curation Algorithm (Algorithm 1)

This is the core technical contribution: a three-stage algorithm that extends English Meta CLIP's curation to worldwide data by making every step language-aware. The pseudocode in Algorithm 1 is reproduced in the paper with Python/NumPy-style notation. I will walk through each stage, connecting the pseudocode to the mathematical logic and the empirical design choices.

---

##### Stage 1: Per-Language Substring Matching and Global Counting

**Input:** Raw image-text pairs `D`, where each pair `(image, text)` has already been processed by a language identification (LID) system (Grave et al., 2018) to assign `text.lang` — a language code. The worldwide metadata dictionary `M` maps each language code to its metadata list.

**Step-by-step execution:**

1. **Initialize entry counts:** For each language `lang` that appears as a key in `M`, create a zero-initialized array `entry_counts[lang]` of length equal to the number of metadata entries for that language. This array will accumulate, for each metadata entry in that language, the total number of image-text pairs whose alt-text contains that entry as a substring.

2. **Process each pair:** For each `(image, text)` in `D`:
   - Call `substr_match(text, M[text.lang])`. This function takes the alt-text string and the metadata list for the text's detected language, and returns a list of indices `text.matched_entry_ids` — the IDs of all metadata entries (in that language's list) that appear as substrings in the alt-text.
   - Increment `entry_counts[text.lang][entry_id]` by 1 for each matched `entry_id`. This is a global accumulation: every pair that mentions "perro" (Spanish for "dog") increments the count for the "perro" entry in the Spanish metadata.

**Key design choice: language-specific matching.** The alt-text is matched only against metadata for its detected language. A Vietnamese alt-text is matched against the Vietnamese metadata list (which contains Vietnamese words, phrases, and titles from Vietnamese Wikipedia and the Vietnamese WordNet synsets if available). It is not matched against English metadata or any other language's metadata. This is the "language isolation" that Table 2 shows is critical: when all metadata is merged into one set (row 3 of Table 2), ImageNet accuracy drops to 62.1% from 66.9% (row 2, where English metadata matches all alt-texts). The merged approach causes English concept strings to spuriously match non-English text (e.g., the English word "die" appearing as a substring in German text where it means "the") and non-English concept strings to fragment the English matching, creating noise in the concept counts.

**Engineering detail: Aho-Corasick algorithm.** The paper notes (Appendix A.2) that the substring matching uses the Aho-Corasick algorithm, which builds a prefix tree (trie) from all metadata entries for a language and then scans the alt-text in a single pass, identifying all matches simultaneously. The paper reports this is "about 2k times faster than Meta CLIP's brute-force implementation," which is essential for matching against million-scale metadata across billions of pairs. The algorithm pre-builds one Aho-Corasick automaton per language and loads it lazily — only when the first alt-text of that language is encountered in a data shard, preventing all 300+ language automata from being resident in memory simultaneously.

**Output of Stage 1:** `entry_counts`, a dictionary mapping each language to an array of match frequencies for each of its metadata entries. This is the worldwide analog of the English-only `entry_count` array from the original Meta CLIP.

---

##### Stage 2: Computing Language-Specific Thresholds `t_lang`

This stage addresses the central scaling challenge: the head/tail threshold `t` that works for English (170k for a 2.5B-pair English dataset) is not appropriate for every language, because different languages have vastly different numbers of image-text pairs and different concept frequency distributions.

**The problem with a single global threshold.** Suppose we use `t_en = 170k` for all languages. For a high-resource non-English language like Spanish, which might have hundreds of millions of pairs, this threshold may be reasonable — Spanish "head" concepts like "perro" might have counts in the millions and get downsampled, while tail concepts with counts below 170k are preserved. But for a low-resource language like Swahili, which might have only tens of thousands of pairs total, every single metadata entry will have `entry_count < 170k`. All entries become tail entries, no downsampling occurs, and the Swahili pairs are included in the training set with their raw, unbalanced distribution — exactly what curation is supposed to fix. Conversely, for an extremely high-resource language where even rare concepts have counts above 170k, everything becomes head data and gets aggressively downsampled, potentially starving the model of signal from that language entirely.

**The invariance assumption.** The paper leverages an empirical invariant discovered by the original Meta CLIP scaling study: **the optimal tail-match proportion `p` is approximately 6%, independent of data scale**. OpenAI CLIP achieved this with `t = 20k` on 400M pairs; Meta CLIP achieved it with `t = 170k` on 2.5B pairs. The paper's key move is to assume this invariant generalizes **across languages** — that is, for any language, the curated training data should have approximately 6% of its concept occurrences coming from tail concepts (concepts that are rare *within that language's data*).

This is not obviously true. Different languages might have different optimal head/tail balances depending on the diversity of their visual concept vocabularies, the quality of their alt-texts, or the coverage of their metadata. The paper does not ablate this assumption directly (e.g., testing whether 4% or 8% works better for specific languages), but the strong multilingual results in Table 1 provide indirect validation.

**Step 1: From `t_en` to global tail proportion `p`.** The function `t_to_p(t_en, entry_counts["en"])` computes:

$$p = \frac{\sum_{e: \text{entry\_counts}[\text{"en"}][e] < t_{\text{en}}} \text{entry\_counts}[\text{"en"}][e]}{\sum_{e} \text{entry\_counts}[\text{"en"}][e]}$$

where `entry_counts["en"]` is the array of match frequencies for all English metadata entries, the numerator sums the counts of all entries whose count is strictly less than `t_en`, and the denominator sums all entry counts (head and tail). The output `p` is a scalar between 0 and 1 — the fraction of total English concept matches that come from tail concepts.

**What it computes, operationally:** take the English match counts (how many times each English metadata entry appeared in English alt-texts), set the threshold `t_en` (inherited from Meta CLIP's tuning, e.g., 170k), identify which entries fall below this threshold, sum their counts, and divide by the sum of all counts. This yields the proportion of training signal that the English curation allocates to tail concepts.

**Why this form:** it extracts the distributional invariant — the tail mass proportion — that was implicitly chosen when `t_en` was tuned for English. Rather than manually tuning a separate threshold for each of 300+ languages (which would be computationally prohibitive and require language-specific held-out validation sets), the algorithm uses English (the highest-resource, best-understood language) to calibrate `p`, then transfers that proportion to all other languages.

**Step 2: From global tail proportion `p` to language-specific threshold `t_lang`.** For each language `lang` (including English, though English gets back approximately `t_en`), the function `p_to_t(p, entry_counts[lang])` does the inverse operation. The pseudocode shows:

```
sorted_count = np.sort(entry_counts[lang])
cumsum_count = np.cumsum(sorted_count)
cumsum_prob = cumsum_count / sorted_count.sum()
return sorted_count[(np.abs(cumsum_prob - p)).argmin()]
```

**What it computes, step by step:** (1) Sort all entry counts for language `lang` in ascending order. (2) Compute the cumulative sum of these sorted counts — the i-th element is the total matches for all entries with count ≤ the i-th smallest count. (3) Normalize by the total sum to get cumulative probabilities — the i-th element is the fraction of total matches coming from entries with count ≤ the i-th smallest count. (4) Find the index where this cumulative probability is closest to `p` (the target tail proportion). (5) Return the count value at that index as `t_lang`.

**In plain language:** "Find the count threshold such that entries with counts below it account for approximately `p` (6%) of all matches in this language." This is the language-specific `t_lang` that preserves the same tail proportion as the English curation.

**Why this form (sorting and cumulative sum):** the entry count distribution within a language is an empirical cumulative distribution function. The `p_to_t` function computes the **quantile** of this distribution corresponding to probability `p`. This is a non-parametric way to set the threshold — it doesn't assume any particular distributional shape, only that the sorted counts provide a meaningful ordering of concept rarity. The `argmin` over absolute differences handles the fact that with discrete counts, the exact probability `p` may not be achievable.

**Example to build intuition.** Consider a small language with 3 metadata entries and counts: A=5, B=100, C=1000. Total matches = 1105. Sorted: [5, 100, 1000]. Cumulative: [5, 105, 1105]. Cumulative probabilities: [0.0045, 0.095, 1.0]. If `p = 0.06` (6%), the closest cumulative probability is 0.095 (at count 100), so `t_lang ≈ 100`. Entries A (count 5 < 100) and B (count 100, which is exactly at threshold — the code uses `< t`, so B with count 100 equal to threshold would be a head entry) get tail treatment. The tail proportion is 5/1105 ≈ 0.45% (if only A is tail) or 105/1105 ≈ 9.5% (if both A and B are tail). The algorithm picks the threshold that gets closest to 6% — in this case, `t_lang = 100` gives 0.45%, `t_lang` slightly above 5 would give approximately 9.5% (if the next count after 5 is 100). The exact behavior depends on the `< t` strict inequality and the distribution.

**Output of Stage 2:** a dictionary `t` mapping each language code to its language-specific head/tail threshold `t_lang`.

---

##### Stage 3: Per-Language Balancing and Probabilistic Sampling

With per-language counts and per-language thresholds computed, the final stage assigns sampling probabilities and selects pairs for the curated dataset `D*`.

**Step 1: Convert counts to probabilities.** For each language:
```
entry_counts[lang][entry_counts[lang] < t[lang]] = t[lang]
entry_probs[lang] = t[lang] / entry_counts[lang]
```

**What this does, operationally:** For entries whose count is strictly below `t[lang]` (tail entries), their count is **clamped** to `t[lang]`. This means `entry_probs[lang][tail_entry] = t[lang] / t[lang] = 1.0` — tail entries always get sampling probability 1. For entries whose count is at least `t[lang]` (head entries), their probability is `t[lang] / entry_counts[lang][entry]`, which is ≤ 1 and inversely proportional to how common the entry is. An entry with count = 10 × `t[lang]` gets probability 0.1; an entry with count = 100 × `t[lang]` gets probability 0.01.

**Step 2: Sample pairs.** For each `(image, text)` in `D`:
```
for entry_id in text.matched_entry_ids:
    if random.random() < entry_probs[text.lang][entry_id]:
        D_star.append((image, text))
        break
```

**What this does, operationally:** For each image-text pair, iterate through the metadata entries matched in its alt-text (in the language-specific metadata for the text's detected language). For each matched entry, draw a uniform random number between 0 and 1. If that number is less than the entry's sampling probability, include the pair in `D*` and **break** (stop checking remaining matched entries for this pair). The `break` implements a logical OR across matched entries: the pair is included if *any* of its matched entries triggers inclusion. This means a pair containing both a head concept (low probability) and a tail concept (probability 1.0) will always be included because the tail concept's probability check will succeed.

**Why this form (the OR logic with break):** it ensures that pairs containing at least one rare concept are always preserved, regardless of what common concepts also appear. A pair whose alt-text is "a car parked in front of Angkor Wat" matches the head concept "car" and the tail concept "Angkor Wat" — the tail concept's probability 1.0 guarantees inclusion, so the cultural landmark is not lost. If the algorithm used the minimum probability or an AND logic, pairs with both head and tail concepts might be downsampled, defeating the purpose of tail concept preservation.

**Why the 2.3× scaling factor matters for Stage 3:** the `break` logic means the effective sampling rate for the curated dataset depends on the probability of at least one matched entry triggering inclusion. For head-concept-heavy languages where most pairs match only common entries (all with probability < 1), the sampling rate could be well below 100%. The 2.3× scaling of seen pairs (discussed in Section 3.4.4) ensures that even with this downsampling, the absolute number of curated pairs is sufficient to fill the larger global batch size. The paper does not report exact curation yields (what fraction of raw pairs survive to `D*`), but the 2.3× factor implies that the curated worldwide dataset is approximately 2.3× larger than the curated English dataset (29B seen pairs vs. 12.8B).

**Output of Stage 3:** `D*`, the curated dataset of balanced, diverse image-text pairs spanning all languages.

---

#### 3.4.4 Training Framework Modifications

Even with perfect worldwide curation, the paper finds (Table 1) that training a CLIP model on `D*` with the same number of seen pairs as English-only training (12.8B, or 1.0×) yields **worse English performance** (79.5% vs. 80.4% ImageNet for ViT-H/14). This is the curse of multilinguality manifesting even with curated data — because English pairs now constitute only ~44% of the dataset, the model sees fewer English examples in a fixed training budget. The training framework modifications address this by ensuring English exposure is not diluted.

**Scaling seen pairs proportionally to data growth.**

The paper's solution is simple in concept but requires careful justification: **multiply the number of seen training pairs by the factor needed to keep the absolute number of English examples constant**. Since English pairs are approximately 44% of the worldwide data, worldwide data is approximately 1/0.44 ≈ 2.27× larger than English-only data. The paper rounds this to **2.3×**, meaning the worldwide training sees 2.3× more pairs — approximately 29B pairs — over the course of training.

This is implemented by scaling the **global batch size** from 32,768 to 75,366 (also a 2.3× factor), while keeping all other hyperparameters identical: learning rate 4.0e-4, warmup 2k steps, QuickGELU activation, AdamW optimizer, and the same number of training steps (which, with larger batch size, yields proportionally more seen pairs). Table 6 documents these hyperparameters explicitly.

**Why this works (and why it's not just "train longer"):** the contrastive loss in CLIP operates over in-batch negatives — each image is contrasted against all texts in the batch, and vice versa. By increasing the batch size, the model sees more negative examples per positive pair, which improves the contrastive signal quality. Additionally, with a larger batch that samples proportionally from all languages, the model is more likely to encounter cross-lingual positive pairs (images captioned in different languages depicting similar concepts), which may facilitate cross-lingual transfer. The paper does not ablate whether the benefit comes from more English examples, better contrastive signal, or both, but the empirical result is clear: Worldwide 2.3× achieves 81.3% ImageNet (above English-only 80.4%), while Worldwide 1.0× achieves only 79.5% (below English-only).

**Why 2.3× specifically (and not other scaling factors):** the factor is derived from the data composition, not tuned as a hyperparameter. With English at 44%, a factor of 1/0.44 = 2.27× exactly preserves English example count. The choice to use 2.3× rather than 2.27× is likely a rounding to simplify batch size computation (32,768 × 2.3 = 75,366.4 → 75,366). The paper does not report ablations of, for example, 2.0× or 2.5× scaling, so we cannot assess how sensitive results are to this specific multiplier. This is a minor gap: if performance improves monotonically with seen pairs beyond the English-preserving point, the optimal factor might be larger than 2.3×.

**An important implicit assumption:** scaling seen pairs by 2.3× means the model trains for 2.3× more total computational cost (proportional to batch size increase). The paper's FLOPs comparison in Table 1 shows Meta CLIP 2 uses 72% of mSigLIP's seen pairs (29B vs. 40B), not 72% of total training compute (since per-pair compute depends on model size, resolution, and other factors). The efficiency claim is about *data* efficiency, not *compute* efficiency — Meta CLIP 2 achieves better results with fewer training examples, but the larger model (ViT-H/14) and higher batch size may mean similar or greater total FLOPs.

**Studying minimal viable model capacity.**

The paper's most striking empirical finding is in Figure 1 (left panel): training with the full Meta CLIP 2 recipe (worldwide curated data, 2.3× seen pairs) on **ViT-L/14** still exhibits the curse of multilinguality — ImageNet accuracy drops from 79.5% (English-only) to 78.8% (worldwide). But on **ViT-H/14**, the curse breaks: worldwide (81.3%) exceeds English-only (80.4%).

The paper interprets this as a **capacity threshold**: ViT-L/14 has approximately 300M parameters (standard ViT-L configuration), while ViT-H/14 has approximately 630M parameters (standard ViT-H configuration). The ~2× increase in parameters crosses a critical point where the model has enough representational capacity to learn **complementary** features from multilingual data rather than suffering from **representational interference**.

**What "representational interference" means concretely:** in a contrastive learning setup, the vision encoder must map images to a shared embedding space with text. When text comes from multiple languages, the text encoder must represent the same visual concept through different surface forms (e.g., "dog" in English, "perro" in Spanish, "собака" in Russian all point to similar visual features). If the text encoder's capacity is insufficient, these different surface forms compete for the same representational capacity — learning to map "perro" to the dog visual cluster slightly disrupts the "dog" mapping, and vice versa. With enough capacity, the text encoder can learn language-specific subspaces that all project to the same visual concept region without interference, and the vision encoder can learn richer features from the multilingual supervision (since different languages emphasize different aspects of the same concept — e.g., one language's word for a garment might encode its material, while another's encodes its cultural context).

**Why ViT-H/14 rather than some other scale:** the paper does not test intermediate scales (e.g., ViT-L with increased width, or ViT-g). The choice of ViT-H/14 is pragmatic — it's a standard, well-characterized architecture — but we cannot conclude from these experiments that ViT-H/14 is the *minimum* viable capacity. Some model size between L and H might also break the curse. The paper's claim is that H is *sufficient*, not that it's *necessary and minimal*.

**The interaction effect:** the three interventions — curation, scaled pairs, and model capacity — are **jointly necessary** for breaking the curse. Table 1 shows:
- ViT-H/14 + worldwide data + 1.0× seen pairs → 79.5% (curse persists, English examples diluted).
- ViT-L/14 + worldwide data + 2.3× seen pairs → 78.8% (curse persists, capacity insufficient).
- ViT-H/14 + worldwide data + 2.3× seen pairs → 81.3% (curse broken).

No subset of two interventions suffices. This is strong evidence that the curse of multilinguality in CLIP is a **compound problem** with multiple contributing causes, explaining why prior work — which typically addressed at most one of these factors — failed to resolve it.

---

#### 3.4.5 Multilingual Tokenizer Selection

The only architectural change to the model itself is replacing the English GPT-2 tokenizer (used by OpenAI CLIP and Meta CLIP) with a multilingual tokenizer capable of encoding text from all training languages. The paper ablates four tokenizers on ViT-B/32 with Worldwide 1.0× (Table 3):

| Tokenizer | Vocabulary Size | IN val | Babel-IN | XM3600 T→I / I→T | CVQA EN / LOCAL |
|-----------|----------------|--------|----------|-------------------|-----------------|
| mT5 (used by mSigLIP) | 250k | 64.7 | 31.5 | 38.1 / 50.0 | 50.3 / 46.6 |
| Gemma (used by SigLIP 2) | 256k | 63.7 | 26.1 | 36.1 / 47.8 | 48.3 / 44.0 |
| XLM-Roberta | 250k | 64.0 | 31.1 | 38.0 / 49.8 | 49.8 / 46.1 |
| XLM-V | 900k | 64.7 | 32.7 | 40.0 / 51.4 | 50.4 / 47.4 |

**XLM-V is selected** because it matches or exceeds all other tokenizers on English (ImageNet 64.7%, tied with mT5) while providing clear improvements on multilingual benchmarks: Babel-IN +1.2% over mT5, XM3600 text-to-image +1.9%, CVQA local +0.8%.

**Why vocabulary size matters:** XLM-V's 900k vocabulary is approximately 3.6× larger than the other candidates' ~250k. A larger vocabulary means fewer tokens are split into subword pieces — a Vietnamese word that would be split into 3-4 subword tokens by mT5 might be a single token in XLM-V. This matters for CLIP because the text encoder operates on a fixed sequence length (77 tokens for standard CLIP). If non-English text is fragmented into many subword tokens, the effective sequence length available for semantic content is reduced, and the model must learn to compose concepts from smaller pieces, which is harder. XLM-V's larger vocabulary preserves more words as whole tokens, giving the text encoder cleaner input representations.

**A design choice the paper does not discuss:** whether to train the tokenizer from scratch on the curated worldwide data or use a pre-trained tokenizer. All four candidates are pre-trained tokenizers (mT5 was trained on the mC4 corpus, XLM-Roberta on CC-100, Gemma on Gemini training data, XLM-V on a multilingual corpus). Using a pre-trained tokenizer means the subword splits are optimized for general text, not specifically for alt-text distributions. Alt-texts have different statistical properties than web articles — they're shorter, more noun-heavy, and contain more rare entity names. A tokenizer trained specifically on the curated alt-text data might achieve better compression and representation, but this is left unexplored.

---

#### 3.4.6 Synthesis of Design Choices and Their Justifications

**Per-language metadata isolation over merged metadata.** The paper ablated this directly in Table 2. Merging all language metadata into one set (row 3) yields 62.1% ImageNet; per-language isolation (row 4) yields 61.1%, both substantially worse than the English-only baseline (67.5%). Performance only recovers when language-specific thresholds are added (row 5: 64.7%). The initial drop from merging is due to cross-lingual substring matching noise (the "mit" problem and many others); the recovery comes from proper balancing per language. So isolation is necessary but not sufficient — it must be paired with per-language thresholding.

**Language-specific thresholds over a single global threshold.** If `t_en = 170k` is applied globally (row 4 of Table 2), the algorithm fails to balance non-English languages appropriately, falling to 61.1% ImageNet. The language-specific threshold computation (row 5) recovers to 64.7%. The mechanism: for small languages, `t_en` is too high (everything becomes tail, no downsampling); for large languages, `t_en` may be too low (everything becomes head, over-downsampling). Per-language thresholds fix both problems by calibrating to each language's data volume.

**2.3× seen pair scaling over other factors.** Derived from data composition (44% English → 2.27×), not from hyperparameter search. The paper does not ablate alternative scaling factors, so we must assume monotonicity: if 2.3× breaks the curse and 1.0× doesn't, any factor ≥ 2.27× should work, and the 2.3× choice is close to the minimum necessary. A larger factor might yield further improvements (since the model sees even more examples), but would also increase training cost proportionally.

**ViT-H/14 over smaller models.** Not a design choice in the usual sense — the paper *discovered* that H scale is necessary by trying L scale and observing that the curse persists. The result is presented as an empirical finding (the inflection point), not as a design principle that was known in advance. The implication for practitioners: when training multilingual CLIP, budget for at least ViT-H-scale capacity if you want worldwide data to help rather than hurt English performance.

**XLM-V tokenizer over alternatives.** Selected via ablation (Table 3) on the basis of best aggregate performance across English and multilingual benchmarks. The 900k vocabulary size is the salient differentiating feature, providing better token coverage for non-English languages and reducing subword fragmentation.

**6% tail proportion invariant over language-specific tuning.** This is the most consequential untested assumption. The paper transfers the English-calibrated tail proportion `p` to all other languages. It is plausible that different languages have different optimal balances — a language with very diverse, high-quality alt-texts might benefit from a higher tail proportion (preserving more rare concepts), while a language with noisy, low-quality alt-texts might benefit from a lower tail proportion (being more selective). The paper does not test this, and the strong results suggest the invariant is reasonable, but it represents a possible avenue for future improvement.

**Aho-Corasick over brute-force matching.** An engineering necessity, not a scientific choice. English Meta CLIP's brute-force substring matching, where each of 500k metadata entries is individually searched for in each alt-text, becomes completely infeasible when metadata grows to millions of entries across 300+ languages and the data pool grows to billions of pairs. The Aho-Corasick algorithm's linear-time matching (proportional to alt-text length, independent of the number of metadata entries) is the standard solution for multi-pattern string matching. The 2000× speedup reported makes worldwide curation computationally tractable.

## 4. Key Insights and Innovations

### Innovation 1: The Curse of Multilinguality in CLIP Is a Capacity Problem, Not a Data Problem

The paper's most fundamental conceptual move is to **re-diagnose** the curse of multilinguality — the consistent finding that multilingual models underperform their monolingual counterparts on the dominant language's benchmarks. In the text-only LLM literature, this phenomenon is typically understood as a competition for fixed representational capacity: adding languages dilutes the model's ability to encode any single language well, creating an unavoidable tension between multilinguality and per-language performance. SigLIP 2's decision to train with 90% English data is a direct operationalization of this belief — if multilingual data hurts English, minimize the multilingual data.

Meta CLIP 2's empirical contribution is to **falsify this framing for CLIP**, at least conditionally. Figure 1 and Table 1 demonstrate that the relationship between multilinguality and English performance is not a monotonic dilution curve but an **inflection point**: below a capacity threshold (ViT-L/14, ~300M parameters), adding non-English data indeed degrades English (79.5% → 78.8% on ImageNet); above that threshold (ViT-H/14, ~630M parameters), adding non-English data *improves* English (80.4% → 81.3%). The curse of multilinguality is not an inherent property of multilingual training — it is a **symptom of insufficient model capacity relative to data diversity**.

This is a fundamental reframing, not an incremental refinement. It converts "multilingual training hurts" from a law into a diagnostic: if you observe the curse, your model is too small. The implication is that scaling model capacity is not just a way to improve absolute performance but a **prerequisite for making multilingual data beneficial rather than harmful** to English. This inverts the standard narrative — rather than cautiously minimizing non-English data to protect English performance, you should scale capacity until non-English data becomes an asset.

The finding also provides a **unified explanation for prior contradictory results**. mSigLIP's 1.5% ImageNet drop relative to SigLIP (Zhai et al., 2023) and SigLIP 2's retreat to 90% English (Tschannen et al., 2025) occurred at model scales below this inflection point. The paper does not claim that mSigLIP's ViT-SO400M is capacity-insufficient in absolute terms, but that the specific combination of model scale, data volume, and multilingual diversity was on the wrong side of the threshold. This implies that previous negative results on multilingual CLIP should be re-examined through the lens of capacity scaling rather than taken as evidence against multilingual training in principle.

The tie to evidence is direct: Figure 1 (left) shows the ViT-L/14 worldwide run (78.8%) falling below the English-only run (79.5%), while the ViT-H/14 worldwide run with 2.3× pairs (81.3%) exceeds its English-only counterpart (80.4%). Table 1 confirms that this pattern holds across all English benchmarks (SLIP 26, DC 37), not just ImageNet.

### Innovation 2: Language-Specific Curation Thresholds via Distributional Invariance

Prior to Meta CLIP 2, no curation algorithm existed for non-English CLIP data at scale. The prior approaches either skipped curation entirely (accepting raw, unbalanced non-English data), relied on English CLIP teachers for filtering (importing teacher bias and failing for low-resource languages), or used private pipelines with undisclosed methodology (WebLI). The core intellectual gap was that the central mechanism of English CLIP curation — the head/tail threshold `t` that balances concept frequency — had no principled extension to languages with vastly different data volumes.

The paper's solution is conceptually elegant: **extract a distributional invariant from the well-understood English curation and transfer it across languages, rather than manually tuning per language**. The invariant is the tail-match proportion `p ≈ 6%` — the fraction of total concept occurrences that come from rare (tail) concepts — which was implicitly chosen when OpenAI and Meta CLIP tuned `t` for English at different data scales. By computing `p` from English via the `t_to_p` function and then inverting it for each language via `p_to_t`, the algorithm derives language-specific thresholds without requiring any language-specific validation data or manual tuning.

This is a **methodological innovation**, not a performance innovation. It provides the first principled answer to "what should the curation threshold be for a language with 100× less data than English?" without resorting to heuristics or expensive per-language hyperparameter search. The two-step procedure (`t_en → p → t_lang`) encodes a specific assumption — that the optimal tail proportion is invariant across languages — which is empirically testable (and could be wrong for some languages), but the explicit, mathematical form of the assumption makes it falsifiable and improvable, unlike the opaque thresholds in closed-source pipelines.

The significance extends beyond CLIP. Any data curation pipeline that uses threshold-based balancing (concept frequency, entity occurrence, quality score) faces the same problem when extended to multiple heterogeneous data sources: the threshold that works for the dominant source is wrong for the others. The invariance-assumption approach — identify a distributional invariant from the best-understood source and transfer it — is a generalizable pattern for multi-source data curation.

Evidence: Table 2 shows that language-specific thresholds (row 5: 64.7% ImageNet) outperform a single global threshold (row 4: 61.1%) by a substantial margin on ViT-B/32. The ablation isolates the threshold computation as the critical factor, since rows 4 and 5 differ only in whether `t` is shared or per-language — metadata isolation, substring matching, and sampling are identical.

### Innovation 3: The Three-Factor Interaction — Curation, Exposure, and Capacity Are Jointly Necessary

The paper's ablation structure (Table 1) reveals an insight that is easy to miss if you read only the final numbers: **breaking the curse of multilinguality requires all three interventions simultaneously — no subset of two suffices**. ViT-H/14 with scaled pairs but 1.0× seen pairs (Worldwide 1.0×) still shows the curse (79.5% vs. 80.4% English-only). ViT-L/14 with proper curation and scaled pairs still shows the curse (78.8% vs. 79.5% English-only). Only the full combination — ViT-H/14 + language-specific curation + 2.3× seen pairs — breaks through.

This interaction pattern is a **diagnostic contribution**: it explains *why prior work failed to solve the problem*. Each prior approach addressed at most one of the three factors:
- mSigLIP had private curation (potentially addressing data quality) and large models (ViT-SO400M), but trained with a fixed number of seen pairs — English examples were diluted.
- SigLIP 2 added capacity and data but explicitly deprioritized multilingual curation (90% English).
- M-CLIP and distillation approaches had no curation at all for non-English data and frozen vision encoders that couldn't benefit from multilingual supervision.
- Machine translation approaches addressed the data problem (making non-English text usable) but introduced translation artifacts and didn't address capacity or exposure.

The interaction implies that the curse of multilinguality is a **compound problem** with multiple independent causes, not a single bottleneck. This is methodologically significant: it means ablation studies that vary only one factor at a time (e.g., "does better curation help?" without also scaling capacity) will produce negative or misleading results. The field's prior failure to solve this problem may be attributable not to the difficulty of any individual factor but to the difficulty of identifying that all three must be addressed together.

The finding also has practical implications for anyone attempting multilingual CLIP training: you cannot add one piece of the recipe and expect improvement. Scaling the model without scaling seen pairs hurts English. Adding curation without scaling the model leaves the curse intact. The three factors are not additive — they are multiplicative, and the product crosses the threshold only when all are above their minimum levels.

Evidence: Table 1 provides the full matrix — Worldwide 1.0× on H/14 (curation yes, scaled pairs no, capacity yes → curse persists), Worldwide 2.3× on L/14 (curation yes, scaled pairs yes, capacity no → curse persists), Worldwide 2.3× on H/14 (all three yes → curse broken). The pattern is clean and internally replicated across English benchmarks (IN, SLIP 26, DC 37).

### Innovation 4: Curation from Scratch Is a Viable Alternative to Distillation for Multilingual CLIP

A less explicit but practically important contribution: Meta CLIP 2 demonstrates that **native multilingual curation from scratch can outperform distillation-based approaches**, even when the distillation approaches use larger models, more seen pairs, and higher resolution. Table 1 shows Meta CLIP 2 ViT-H/14 surpassing mSigLIP ViT-SO400M (a model with a more powerful architecture, 40B vs. 29B seen pairs, and 256px vs. 224px resolution) on nearly all benchmarks — both English (IN 81.3% vs. 80.6%, SLIP 26 74.5% vs. 69.1%) and multilingual (Babel-IN 50.2% vs. 46.4%, XM3600 I→T 64.3% vs. 62.8%, CVQA 57.4% vs. 49.8%).

This is significant because the prevailing narrative in the CLIP community has been that distillation — using a pretrained teacher model (usually English CLIP) to filter or guide training — is the most effective approach, especially when training data is noisy or heterogeneous. LAION, DFN, and many OpenCLIP variants all rely on teacher-based filtering. The implicit assumption is that curation without a teacher model is too crude for multilingual data, where concept matching is harder and language quality varies.

Meta CLIP 2 challenges this by showing that **deterministic, model-free curation with language-specific metadata can produce better results than teacher-based filtering**, at least when combined with proper capacity and exposure scaling. The curation algorithm has no access to any pretrained CLIP model — it operates purely on string matching against human-curated metadata. This means it avoids teacher bias (the curated distribution is shaped by human knowledge about visual concepts, not by what an existing CLIP model already knows), it works equally for low-resource languages where teacher models may have poor coverage, and it is fully transparent and reproducible.

The framing contribution is: **multilingual CLIP data does not need to be filtered by an English CLIP model to be useful**. Native-language curation with language-specific metadata is not just a principled alternative — it can be *better* than distillation, even when competing against systems that use proprietary data sources (Google Image Search) and more expensive training recipes. This is a counter-narrative to the distillation-dominated status quo and provides a path forward that does not depend on access to high-quality teacher models for every language.

Evidence: Table 1 provides the head-to-head comparison between Meta CLIP 2 and mSigLIP/SigLIP 2. The paper explicitly notes that mSigLIP and SigLIP 2 are "greyed out" as SoTA-aiming systems with confounding factors, making the comparison about the viability of curation-from-scratch rather than about claiming methodological superiority over those specific systems.

## 5. Experimental Analysis

### Evaluation Methodology

- **Dataset.** All experiments use the authors' proprietary collection of publicly available image-text pairs sourced from the Internet. After language identification, approximately 44% of alt-texts are in English, which the paper notes is "on par with the scale of English-only data from Meta CLIP." The exact number of raw pairs before curation is not disclosed, and the curated dataset size is characterized indirectly through the seen-pairs metric (29B pairs at 2.3× scaling). No existing public multilingual benchmark is used for training; evaluations span 10+ downstream zero-shot transfer benchmarks described below. The paper also uses a 5k holdout set of image-text pairs not used in training for the alignment/uniformity analysis (Section 4.2.4).

- **Base model(s).** All main experiments use standard ViT-based CLIP architectures, specifically ViT-B/32 (for the metadata/curation/tokenizer ablations in Section 4.2.2, chosen for efficiency), ViT-L/14 (~300M parameters, the largest model size used by OpenAI CLIP), and ViT-H/14 (~630M parameters). The ViT scaling study (L vs. H) is central to the paper's core claim about the capacity inflection point. All models use the standard CLIP dual-encoder architecture, QuickGELU activation, and contrastive loss — the paper explicitly avoids architectural innovations (e.g., SigLIP's sigmoid loss) to isolate the effect of the worldwide data recipe. The text encoder uses the XLM-V tokenizer (900k vocabulary) for worldwide runs; English-only runs presumably use the original GPT-2 tokenizer, though this is not explicitly re-stated for each ablation.

- **Metrics.** The paper reports zero-shot transfer accuracy (for classification tasks) and Recall@1 (for retrieval tasks) across a diverse set of benchmarks. For classification, accuracy is computed by matching image embeddings against text embeddings of class prompts and selecting the highest cosine similarity. For retrieval, Recall@1 measures the fraction of queries where the correct target is the top-ranked result. The alignment and uniformity metrics (Section 4.2.4, Figure 4) follow Wang and Isola (2020): alignment is the expected cosine distance between paired image and text embeddings (lower is better), and uniformity measures how evenly image embeddings are distributed on the unit hypersphere (lower is better). No confidence intervals, standard deviations, or statistical significance tests are reported for any result.

- **Baselines.** The paper compares against several published multilingual CLIP models:
  - **mSigLIP** (Zhai et al., 2023): trained on WebLI (12B pairs, undisclosed data from Google Image Search) with sigmoid loss, evaluated in ViT-B/16(256px) and ViT-SO400M(256px) configurations. The B/16 model uses 75.1% ImageNet as a reference; the SO400M model is the primary competitor at 80.6% ImageNet.
  - **SigLIP 2** (Tschannen et al., 2025): the successor to mSigLIP, trained with 90% English data on WebLI, evaluated in ViT-SO400M(256px) configuration. Achieves 83.2% ImageNet but underperforms mSigLIP on multilingual benchmarks.
  - **XLM-CLIP** (Ilharco et al., 2021): based on OpenCLIP ViT-H/14 trained on LAION-5B, evaluated as a reference for multilingual performance.
  - **Meta CLIP** (Xu et al., 2024): the English-only predecessor, evaluated in ViT-L/14 and ViT-H/14 at 224px resolution on English benchmarks only. Serves as the English-only reference point for measuring the curse of multilinguality.
  - **English-only subsets of Meta CLIP 2**: the paper also uses its own English (1.0×) and Non-English (1.3×) training runs on ViT-H/14 as ablation baselines to isolate the effect of worldwide data.

  The paper explicitly notes that mSigLIP, SigLIP 2, and XLM-CLIP are "greyed out" as "SoTA-aiming systems with confounding factors" — they differ from Meta CLIP 2 not just in data but in loss function, architecture, resolution, and data source, making direct attribution of performance differences impossible.

- **Generation budget / compute accounting.** The primary unit of compute reported is **seen pairs** — the total number of image-text pairs processed during training. This is computed as (global batch size) × (number of training steps). The baseline English-only training uses 12.8B seen pairs (reported as "13B" with rounding); the worldwide 1.0× condition uses 12.8B as well; worldwide 2.3× uses 29B. Global batch size is scaled proportionally (32,768 → 75,366) while keeping number of training steps constant. Comparisons against mSigLIP note that Meta CLIP 2 uses 72% of mSigLIP's seen pairs (29B vs. 40B). The paper does not report total FLOPs, GPU-hours, or wall-clock time, and does not account for the compute cost of the curation algorithm itself (substring matching across billions of pairs, which the paper notes runs on 800 parallel jobs with 40GB memory each for approximately 1 hour).

- **Cross-validation / statistical protocol.** The paper does not employ cross-validation, bootstrap confidence intervals, or statistical significance testing. The main results (Table 1) report single-run numbers with no error bars. The 5k holdout set for alignment/uniformity analysis (Figure 4) is a single fixed split. The paper acknowledges that "we have no control on whether these 5k pairs are leaked in other baselines" (Section 4.2.4), meaning the alignment/uniformity numbers for mSigLIP and SigLIP 2 may be contaminated if those models trained on overlapping data. The paper performs mitigation for benchmark contamination (deduplication against ImageNet evaluation sets using 64-bit hashes from random projection) but does not extend this deduplication to multilingual benchmarks or to the holdout set.

---

### Main Quantitative Results

#### The Main Ablation: Breaking the Curse of Multilinguality (Table 1)

The central result of the paper is the ablation matrix in Table 1, which tests six training configurations across ViT-L/14 and ViT-H/14 models on 14 benchmarks (3 English, 11 multilingual). The key comparisons are organized around demonstrating that all three recipe components — worldwide curation, scaled seen pairs, and sufficient model capacity — must be present simultaneously to break the curse of multilinguality.

**ViT-L/14 results (the curse persists).** At ViT-L/14 scale, the English-only baseline (trained on the English portion of the worldwide data with 1.0× seen pairs, 13B) achieves 79.5% ImageNet. The worldwide run with 2.3× seen pairs (29B, full worldwide data + language-specific curation) drops to **78.8%** — a 0.7 percentage point degradation. This is the curse of multilinguality: adding properly curated non-English data and proportionally scaling training examples still hurts English performance at L scale. The degradation is echoed on SLIP 26 (69.5% → 67.2%, a 2.3 point drop) and DC 37 (66.0% → 63.5%, a 2.5 point drop). Multilingual benchmarks for ViT-L/14 worldwide: Babel-IN 44.2%, XM3600 text-to-image 45.3%, image-to-text 58.2%, CVQA English 59.2%, CVQA Local 55.1%. These numbers are reported but have no English-only L/14 comparison (since the English-only L/14 model has no multilingual capability), so the paper cannot claim the multilingual performance is "good" or "bad" relative to any L-scale baseline.

**ViT-H/14 results (the curse breaks).** The ViT-H/14 English (1.0×) baseline achieves 80.4% ImageNet. Two worldwide configurations are tested at H scale:
- **Worldwide 1.0× (13B seen pairs):** 79.5% ImageNet — the curse persists. This configuration has proper curation and H-scale capacity but dilutes English examples because non-English pairs consume training slots without increasing total seen pairs.
- **Worldwide 2.3× (29B seen pairs):** **81.3% ImageNet** — the curse breaks. This is 0.9 points above the English-only H/14 baseline (80.4%) and 0.8 points above the English-only L/14 baseline (79.5% reported in table, but note the English L/14 number is 79.5% from the English data trained at L scale, while the H English is 80.4%). The cross-model comparison (worldwide H/14 at 81.3% vs. English-only H/14 at 80.4%) is the cleanest evidence of non-English data *helping* English, since both use ViT-H/14.

The pattern holds across all English benchmarks at H/14 2.3×: SLIP 26 rises from 72.6% (English-only) to 74.5% (+1.9 points), and DC 37 rises from 68.7% to 69.6% (+0.9 points). The multilingual benchmarks reach state-of-the-art levels: Babel-IN 50.2%, XM3600 T→I 51.5%, XM3600 I→T 64.3%, CVQA EN 61.5%, CVQA LOC 57.4%, Flickr30k-200 T→I 50.9%, I→T 53.2%, XTD-10 T→I 86.1%, I→T 87.5%, XTD-200 T→I 48.9%, I→T 51.0%.

The Non-English (1.3×) configuration — training ViT-H/14 on only the non-English portion of the curated data with 17B seen pairs — achieves 71.4% ImageNet, confirming that the model does learn visual concepts from non-English data alone, but at a significant penalty compared to bilingual training. Interestingly, Non-English (1.3×) achieves the highest score on some cultural diversity metrics: GLDv2 68.6% (Table 4) vs. Worldwide 2.3× at 69.0% (essentially tied) and far above English-only at 52.8%.

**Comparison to external baselines.** Meta CLIP 2 ViT-H/14 worldwide 2.3× outperforms mSigLIP ViT-SO400M on ImageNet (81.3% vs. 80.6%), SLIP 26 (74.5% vs. 69.1%), and DC 37 (69.6% vs. 65.5%). It surpasses SigLIP 2 ViT-SO400M on SLIP 26 (74.5% vs. 73.7%) and DC 37 (69.6% vs. 69.4%) but falls short on ImageNet (81.3% vs. 83.2%). The paper emphasizes that Meta CLIP 2 achieves these results with 72% of mSigLIP's seen pairs (29B vs. 40B) and lower resolution (224px vs. 256px), though the confounding factors (loss function, data source, architecture) make direct numerical comparisons unreliable for attribution. On multilingual benchmarks, the margins are more substantial: Meta CLIP 2 leads mSigLIP on Babel-IN (50.2% vs. 46.4%, +3.8 points), XM3600 I→T (64.3% vs. 62.8%, +1.5 points), CVQA LOC (57.4% vs. 49.8%, +7.6 points), Flickr30k-200 I→T (53.2% vs. 42.0%, +11.2 points), and XTD-200 I→T (51.0% vs. 45.2%, +5.8 points).

**The "curse of multilinguality" decomposition.** Table 1 cleanly isolates the three failure modes:
- **Curation without capacity or exposure scaling:** ViT-H/14 Worldwide 1.0× drops 0.9 points on ImageNet relative to English 1.0× (79.5% vs. 80.4%).
- **Capacity scaling without exposure scaling:** ViT-L/14 Worldwide 2.3× drops 0.7 points relative to its English baseline (78.8% vs. 79.5%).
- **Capacity and exposure scaling without curation:** Not directly tested (the worldwide runs all use curation), but Table 2 (discussed in ablations below) shows that removing curation (row 2: no English filter, all alt-texts matched against English metadata) drops ImageNet from 67.5% to 66.9% even before adding multilingual metadata.

The interaction is stark: only the full combination (H/14 + worldwide curation + 2.3×) breaks the curse.

#### Metadata, Curation, and Tokenizer Ablations (Tables 2 and 3)

These ablation studies are conducted on the smaller ViT-B/32 model for computational efficiency, using English 1.0× and Worldwide 1.0× configurations (since the goal is to isolate curation effects, not to break the curse, which requires H scale). All runs in Table 2 use the mT5 multilingual tokenizer.

**Table 2: Stepwise degradation and recovery from English CLIP to worldwide curation.**
- **Row 1: English CLIP baseline.** Trained on English-only data with English metadata. ImageNet: 67.5%. No multilingual capability.
- **Row 2: Remove English filter, keep English metadata.** All alt-texts (English and non-English) are matched against English metadata. ImageNet drops to 66.9% (-0.6 points). The paper attributes this to English metadata failing to properly match non-English alt-texts, creating noisy or missing concept associations. The non-English text is effectively uncurated.
- **Row 3: No language isolation — all metadata merged, all alt-texts matched against merged set.** ImageNet crashes to 62.1% (-5.4 points from baseline). This is the "mit" problem at scale: English and non-English concept strings are merged, so English metadata entries spuriously match non-English text (and vice versa), corrupting the global concept counts. Multilingual benchmarks begin to appear: Babel-IN 31.2%, XM3600 T→I 37.8%, I→T 49.7%, CVQA EN 49.8%, CVQA LOC 45.8%.
- **Row 4: Language isolation with global `t_en`.** Alt-texts are matched language-by-language against their corresponding metadata, but all languages use the same English-calibrated threshold `t_en` for head/tail balancing. ImageNet drops further to 61.1% (-0.9 points from row 3). Multilingual benchmarks are roughly flat or slightly worse: Babel-IN 31.5%, XM3600 T→I 37.9%, I→T 49.4%, CVQA EN 49.0%, CVQA LOC 46.5%. The degradation occurs because `t_en` is miscalibrated for non-English languages — too high for small languages (treating everything as tail, no downsampling, preserving noisy raw distributions) and potentially too low for very large non-English languages (over-downsampling head concepts).
- **Row 5: Language isolation with language-specific `t_lang` (full Meta CLIP 2 curation).** ImageNet recovers to 64.7% (+3.6 points from row 4, though still 2.8 points below the English-only baseline — the curse persists at B/32 scale as expected). Multilingual benchmarks improve across the board: Babel-IN 31.5% (flat), XM3600 T→I 38.1% (+0.2), I→T 50.0% (+0.6), CVQA EN 50.3% (+1.3), CVQA LOC 46.6% (+0.1). The improvement is modest but consistent, confirming that language-specific thresholds improve curation quality.

**Table 3: Tokenizer ablation.** Four multilingual tokenizers are compared on ViT-B/32 with Worldwide 1.0× and the full curation recipe:
- **mT5** (250k vocab, used by mSigLIP): IN 64.7%, Babel-IN 31.5%, XM3600 T→I 38.1%, I→T 50.0%, CVQA EN 50.3%, CVQA LOC 46.6%.
- **Gemma** (256k vocab, used by SigLIP 2): IN 63.7% (-1.0 vs. mT5), Babel-IN 26.1% (-5.4), XM3600 T→I 36.1% (-2.0), I→T 47.8% (-2.2), CVQA EN 48.3% (-2.0), CVQA LOC 44.0% (-2.6). Gemma substantially underperforms on multilingual tasks.
- **XLM-Roberta** (250k vocab): IN 64.0% (-0.7 vs. mT5), Babel-IN 31.1% (-0.4), XM3600 T→I 38.0% (-0.1), I→T 49.8% (-0.2), CVQA EN 49.8% (-0.5), CVQA LOC 46.1% (-0.5). Comparable to mT5 with a slight disadvantage on English.
- **XLM-V** (900k vocab): IN 64.7% (ties mT5), Babel-IN 32.7% (+1.2 over mT5), XM3600 T→I 40.0% (+1.9), I→T 51.4% (+1.4), CVQA EN 50.4% (+0.1), CVQA LOC 47.4% (+0.8). Best overall, with clear multilingual advantages attributable to the larger vocabulary providing better token coverage for non-English text.

XLM-V is selected for all main experiments.

#### Cultural Diversity Benchmarks (Table 4 and Figure 3)

Table 4 reports zero-shot classification accuracy on three geographically diverse benchmarks for ViT-H/14 configurations and external baselines:
- **Dollar Street** (Gaviria Rojas et al., 2022): images of household items across diverse socioeconomic and geographic settings. The worldwide 2.3× configuration achieves 37.9% Top-1, 64.0% Top-5, compared to 37.2%/63.3% for English-only (1.0×) — modest gains. mSigLIP achieves 36.0%/62.5% and SigLIP 2 achieves 36.7%/61.9%, both below Meta CLIP 2.
- **GLDv2** (Google Landmarks Dataset v2): landmark recognition requiring geographic knowledge. Worldwide 2.3× achieves **69.0%**, massively above English-only (52.8%, a +16.2 point gain) and above Non-English (68.6%). mSigLIP (45.3%) and SigLIP 2 (48.5%) trail substantially. This is the strongest evidence that non-English training data improves geographically diverse visual knowledge.
- **GeoDE** (Ramaswamy et al., 2023): object recognition across geographic regions. Results are clustered: English-only 93.4%, Non-English 91.7%, Worldwide 13B 94.3%, Worldwide 29B 93.4%, mSigLIP 94.5%, SigLIP 2 95.2%. The paper notes GeoDE may be "probably saturated" given the tight clustering near ceiling.

Figure 3 shows few-shot geo-localization accuracy (predicting the geographic region of an image) across 4 benchmarks at {5, 10, 25} shots. Worldwide 29B consistently outperforms or matches English-only and Non-English configurations. The gap between Worldwide 29B and English-only is largest on Dollar Street and XM3600, moderate on GeoDE (country), and small on GeoDE (region). The external baselines (mSigLIP, SigLIP 2) are included in the figure but not discussed in detail in the text — visual inspection suggests Meta CLIP 2 configurations generally match or lead them, though the dense overplotting makes precise comparison difficult from the figure alone.

#### Alignment and Uniformity Analysis (Figure 4)

Figure 4 plots alignment score (x-axis, lower is better) against uniformity score (y-axis, lower is better) for 9 models evaluated on a 5k holdout set of image-text pairs. The dotted line shows `uniform + align` (combined metric, lower is better) as a diagonal reference.
- Meta CLIP 2 models occupy a favorable region: L/14 English 1× achieves the best alignment (~1.45) but poor uniformity (~3.0), while H/14 Worldwide 2.3× achieves the best uniformity (~2.25) and moderate alignment (~1.65), landing near the lower-left edge of the distribution.
- mSigLIP and SigLIP 2 models cluster with higher alignment scores (worse, around 1.8 for alignment) and higher uniformity scores (worse, around 2.6-2.8 for uniformity).
- The paper cautions that the 5k holdout data may be leaked in the external baselines' training, so the comparison against mSigLIP/SigLIP 2 is not fully controlled.

---

### Ablation Studies and Robustness Checks

**Removing the English language filter without adding multilingual curation (Table 2, row 2):** Simply allowing non-English alt-texts to be matched against English metadata drops ImageNet from 67.5% to 66.9%, a 0.6-point degradation. This confirms that English metadata cannot effectively curate non-English text, and that the language filter in English Meta CLIP was doing important work in preventing noisy concept associations.

**Merging all language metadata into a single set vs. per-language isolation (Table 2, rows 3-4):** Merging metadata without language isolation causes a catastrophic 5.4-point ImageNet drop (67.5% → 62.1%), far worse than simply removing the English filter. Adding language isolation (matching each alt-text against its own language's metadata) but with a shared global threshold worsens the drop to 61.1%. This demonstrates that cross-lingual substring matching noise is a first-order problem — when metadata from hundreds of languages is merged, short frequent words in one language spuriously match as substrings in another, corrupting the concept frequency estimates that drive downstream balancing.

**Language-specific thresholds vs. shared global threshold (Table 2, rows 4-5):** Moving from a single global `t_en` to per-language `t_lang` recovers 3.6 points on ImageNet (61.1% → 64.7%) and provides small but consistent gains on all multilingual benchmarks. This validates the core curation innovation — the two-step `t_en → p → t_lang` procedure produces better-calibrated head/tail balancing per language than using a single threshold. The ablation isolates the threshold computation specifically: metadata isolation, substring matching, and sampling are identical between rows 4 and 5; only `t` differs.

**Tokenizer selection impact on multilingual performance (Table 3):** XLM-V (900k vocab) outperforms mT5 (250k) by 1.2 points on Babel-IN, 1.9 points on XM3600 T→I, and 1.4 points on XM3600 I→T, while matching mT5 on English ImageNet. The Gemma tokenizer performs noticeably worse than all alternatives on multilingual benchmarks (Babel-IN 26.1% vs. mT5's 31.5%), suggesting tokenizer choice is a high-stakes decision for multilingual CLIP — a poorly chosen tokenizer can erase multilingual gains even with good curation.

**Scaling seen pairs proportionally vs. keeping English-scale (Table 1, ViT-H/14 Worldwide 1.0× vs. 2.3×):** With identical curation and model capacity, scaling seen pairs from 12.8B to 29B (2.3×) raises ImageNet from 79.5% to 81.3% (+1.8 points), SLIP 26 from 71.1% to 74.5% (+3.4 points), and DC 37 from 67.2% to 69.6% (+2.4 points). Multilingual benchmarks also improve substantially: Babel-IN from 47.1% to 50.2% (+3.1 points), XM3600 I→T from 62.6% to 64.3% (+1.7 points), CVQA LOC from 56.0% to 57.4% (+1.4 points). This ablation demonstrates that seen pair dilution is a major component of the curse — even with perfect curation and H-scale capacity, failing to scale training examples proportionally leaves English performance below the English-only baseline.

**Model capacity: ViT-L/14 vs. ViT-H/14 (Table 1 and Figure 1):** The ViT-L/14 worldwide 2.3× run underperforms the ViT-L/14 English baseline by 0.7 points on ImageNet (78.8% vs. 79.5%), while the ViT-H/14 worldwide 2.3× run exceeds its English baseline by 0.9 points (81.3% vs. 80.4%). This 1.6-point relative swing is the empirical basis for the "inflection point" claim. No intermediate model sizes (e.g., ViT-L with wider dimensions, or ViT-g) are tested, so the exact location of the threshold is not established — only that H scale is sufficient and L scale is insufficient for the specific data volume used.

**Non-English-only training vs. bilingual training (Table 1, Non-English 1.3× vs. Worldwide 2.3×):** Training on non-English data alone (17B pairs) yields 71.4% ImageNet — far below both English-only (80.4%) and worldwide (81.3%). This confirms that (a) non-English data does contain visual concept signal (71.4% is well above random), (b) English data is essential for strong English performance, and (c) the combination is better than either alone — worldwide 2.3× beats both English-only and non-English-only, the key evidence for "mutual benefits."

**Cultural diversity gains from worldwide data (Table 4 and Figure 3):** Adding non-English data (moving from English 13B to Worldwide 13B, same seen pairs) improves GLDv2 by 13.0 points (52.8% → 65.8%) and Dollar Street by modest amounts (Top-1 37.2% → 37.2%, flat; Top-5 63.3% → 63.7%, +0.4). Scaling to 29B pairs further improves GLDv2 to 69.0% (+3.2 over Worldwide 13B) and Dollar Street Top-1 to 37.9%. The few-shot geo-localization results (Figure 3) confirm that worldwide training configurations consistently match or outperform English-only across shots and benchmarks, with the gap being largest on Dollar Street and XM3600.

**Training from non-English data alone on cultural diversity (Table 4, Non-English 1.3×):** The Non-English-only configuration achieves 68.6% on GLDv2 — essentially tied with Worldwide 29B (69.0%) and far above English-only (52.8%). This suggests that landmark recognition knowledge is primarily carried by non-English alt-texts, likely because landmarks outside the Anglosphere are described in local languages. Dollar Street shows the opposite pattern: English-only (37.2%) matches or exceeds Non-English (35.7%), suggesting household object concepts are well-covered in English data.

---

### Critical Assessment

#### Claim 1: "Meta CLIP 2 breaks the curse of multilinguality — worldwide training improves English performance."

**What was demonstrated:** Table 1 shows that ViT-H/14 trained on worldwide data with 2.3× seen pairs achieves 81.3% ImageNet, exceeding the English-only ViT-H/14 baseline of 80.4% by 0.9 percentage points. The same pattern holds for SLIP 26 (+1.9 points) and DC 37 (+0.9 points). The Non-English 1.3× run achieves only 71.4%, confirming that worldwide training is better than non-English-only training.

**What was not demonstrated / limitations:** The claim of "breaking the curse" is demonstrated for exactly one model scale (ViT-H/14) with exactly one data scale (29B pairs) and exactly one data composition (44% English, 56% non-English). The paper does not demonstrate that the curse stays broken if you further increase the non-English proportion (e.g., 30% English, 70% non-English), if you add more non-English languages beyond the 329 currently covered, or if you scale data further (e.g., 50B or 100B pairs). The inflection point is observed at H scale, but the paper does not test whether a ViT-g or ViT-G/14 model would show further gains from worldwide data or whether a model between L and H would also break the curse.

Additionally, the 2.3× seen-pair scaling factor increases total training compute proportionally. The comparison between English-only 1.0× (13B pairs) and worldwide 2.3× (29B pairs) is not FLOPs-matched — the worldwide model uses approximately 2.3× more training compute. The paper does not report what English-only ViT-H/14 performance would be if trained for 2.3× more steps on the same English data (29B English pairs). If English-only ViT-H/14 trained on 29B English pairs also reaches ~81.3% or higher, then the worldwide benefit might be attributable to increased training compute rather than to multilingual data specifically. The Worldwide 1.0× run (79.5%) shows that worldwide data at equal compute hurts English, but the converse — English data at 2.3× compute — is never tested. This missing baseline is a significant gap for attributing the 81.3% to multilingual data rather than to extended training.

#### Claim 2: "Language-specific curation thresholds are necessary for worldwide scaling and outperform a single global threshold."

**What was demonstrated:** Table 2, rows 4-5, shows that on ViT-B/32, language-specific `t_lang` recovers 3.6 ImageNet points compared to a single global `t_en` (64.7% vs. 61.1%), with small but consistent gains on multilingual benchmarks. The ablation directly isolates the threshold computation.

**What was not demonstrated / limitations:** The ablation is conducted only on ViT-B/32 with Worldwide 1.0×. The paper does not ablate threshold strategies at H/14 scale, so we cannot assess whether the per-language threshold benefit persists or diminishes when model capacity and seen pairs are sufficient to break the curse. It's possible that at H/14 2.3× scale, the curse-breaking capacity makes the exact threshold less critical — the model might be robust enough to learn well even with somewhat miscalibrated per-language thresholds. The paper also does not ablate the 6% tail proportion invariant itself: would 4%, 8%, or 10% yield better results for specific languages or overall? The invariance assumption is plausible but untested.

#### Claim 3: "The curse of multilinguality is a compound problem requiring all three interventions (curation, exposure scaling, capacity) simultaneously."

**What was demonstrated:** Table 1 provides a clean 2 × 2 test of the interaction between capacity (L vs. H) and exposure scaling (1.0× vs. 2.3×), with curation held constant at the worldwide recipe. Neither ViT-L/14 with 2.3× (curse persists) nor ViT-H/14 with 1.0× (curse persists) breaks the curse; only the combined cell (H/14 + 2.3×) does.

**What was not demonstrated / limitations:** The three-factor claim is slightly misleading because curation is not independently varied in the main ablation — all worldwide runs in Table 1 use language-specific curation. The "all three necessary" conclusion relies on combining Table 2 (which varies curation on B/32 and shows it matters) with Table 1 (which varies capacity and exposure on L/H and shows they matter). A true three-factor ablation would test Hi/Lo curation × Hi/Lo capacity × Hi/Lo exposure in a single 2 × 2 × 2 design on the same model scale. The current evidence is strongly consistent with three-factor necessity but does not experimentally isolate all possible interactions (e.g., does good curation reduce the capacity threshold, such that a model between L and H might succeed? Does good capacity reduce the exposure scaling requirement?).

#### Claim 4: "Meta CLIP 2 sets new state-of-the-art on multilingual benchmarks."

**What was demonstrated:** Table 1 shows Meta CLIP 2 ViT-H/14 Worldwide 2.3× surpassing mSigLIP ViT-SO400M on Babel-IN (50.2% vs. 46.4%), XM3600 I→T (64.3% vs. 62.8%), CVQA LOC (57.4% vs. 49.8%), Flickr30k-200 I→T (53.2% vs. 42.0%), XTD-10 I→T (87.5% vs. 88.8% — mSigLIP wins here), and XTD-200 I→T (51.0% vs. 45.2%). The margins are substantial on several benchmarks.

**What was not demonstrated / limitations:** The SoTA comparison is confounded by multiple factors that differ between Meta CLIP 2 and mSigLIP/SigLIP 2 — loss function (contrastive vs. sigmoid), training data source (public web vs. Google Image Search), model architecture (ViT-H/14 vs. ViT-SO400M), resolution (224px vs. 256px), and training recipe. The paper acknowledges this explicitly by "greying out" the external baselines. Attributing the multilingual benchmark advantage to the worldwide curation recipe specifically, rather than to some interaction of these confounds, is not supported by the experimental design. A clean comparison would require training mSigLIP's architecture and loss on Meta CLIP 2's data (or vice versa), which is not done.

Furthermore, the paper reports single-run numbers with no error estimates. On the 500-question CVQA benchmark, a 7.6-point improvement (57.4% vs. 49.8%) corresponds to roughly 38 more correct answers out of 500. Without confidence intervals, we cannot assess whether this difference is statistically reliable or within the range of training variance.

#### Missing experiments that would strengthen the paper:

- **English-only training at 29B pairs:** Train ViT-H/14 on English-only data for 2.3× more steps to match the worldwide compute budget. If English-only at 29B pairs also reaches ~81.3%, the worldwide benefit is a compute effect, not a multilingual data effect.
- **Ablation of the tail proportion invariant:** Test `p` values of 3%, 6%, 10%, 15% to determine sensitivity and optimality of the 6% assumption.
- **Intermediate capacity models:** Test ViT-L with increased width or a ViT-"g" configuration to find the actual inflection point rather than just observing that L fails and H succeeds.
- **Language resource stratification:** Report multilingual benchmark results broken down by language resource level (high/medium/low resource languages) to assess whether the recipe benefits all languages equally or primarily helps high-resource non-English languages.
- **Data quality ablation:** Compare Meta CLIP 2's curation-from-scratch against a distillation baseline (e.g., using an English CLIP teacher to filter non-English data) on the same base architecture and training budget to isolate the curation methodology effect.
- **Statistical reliability:** Report standard deviations across multiple training runs or bootstrap confidence intervals on the main benchmark numbers, particularly given the small per-bin sample sizes implied by some benchmarks (e.g., CVQA per-language subsets may be quite small).
- **Scaling curves:** Show ImageNet accuracy as a function of seen pairs for both English-only and worldwide training on ViT-H/14, to visualize whether the worldwide curve will continue to outpace the English-only curve at higher compute budgets or converge.

#### Genuine weaknesses:

- **The 2.3× factor conflates compute and data.** The worldwide model trains on more pairs, so it's unclear whether the gains come from multilingual data, longer training, or a larger effective dataset. The missing English-only 29B baseline is a serious gap.
- **Single-run reporting with no error bars.** All results are point estimates. Given the known sensitivity of CLIP training to random seed, data order, and batch composition, the absence of any variance information makes it impossible to assess whether 0.9-point differences are meaningful or noise.
- **No FLOPs-matched comparison.** The paper compares against mSigLIP on the basis of seen pairs (29B vs. 40B) but does not normalize for model size, resolution, or architecture. A FLOPs-matched comparison would provide a fairer picture of efficiency.
- **The holdout set for alignment/uniformity analysis may be contaminated.** The paper acknowledges this explicitly, but still includes the mSigLIP/SigLIP 2 comparison in Figure 4 without correcting for possible data leakage.
- **Curation cost is not amortized in any efficiency metric.** The 800-job, 1-hour substring matching pre-processing step is a non-trivial computational cost, especially for researchers without access to large compute clusters. The paper does not report this cost in a way that allows practitioners to assess total resource requirements.
- **Limited scope of cultural diversity evaluation.** The cultural diversity benchmarks (Dollar Street, GeoDE, GLDv2) are valuable but narrow — they cover household objects, landmarks, and a small set of object categories. There is no evaluation on culturally diverse tasks like recipe recognition, traditional clothing classification, or regional art identification, which would provide stronger evidence for the paper's claims about cultural diversity benefits.

## 6. Limitations and Trade-offs

### The 2.3× Seen-Pair Scaling Conflates Multilingual Data with Extended Training Compute

**The assumption or constraint.** The paper's central result — that worldwide training improves English performance over English-only training at ViT-H/14 scale (81.3% vs. 80.4% ImageNet, Table 1) — compares two training runs that differ not only in data composition but also in total training compute. The worldwide 2.3× configuration processes 29B image-text pairs, while the English-only 1.0× baseline processes 13B pairs. The 2.3× factor is specifically chosen to preserve the absolute number of English examples seen (since English constitutes ~44% of worldwide data), but it also means the worldwide model trains for 2.3× more steps at a proportionally larger batch size, consuming approximately 2.3× more total FLOPs. The paper does not report any English-only baseline trained with comparable compute (e.g., English-only at 29B pairs or 2.3× batch scaling).

**The consequence.** This missing baseline creates a fundamental attribution ambiguity: the 0.9-point ImageNet improvement could be driven by the multilingual data providing complementary visual signal (the paper's claimed mechanism), or it could be driven by the model simply seeing more English examples and training longer (a compute effect), or some combination of both. If English-only ViT-H/14 trained on 29B English pairs also achieves ~81.3% or higher, then the worldwide data contributes nothing to English performance beyond what additional English training would have provided — the curse of multilinguality would not be "broken" so much as "out-computed." The paper acknowledges the dilution mechanism explicitly when discussing the Worldwide 1.0× run (79.5%, below 80.4% English-only): "Training CLIP for worldwide distribution with the same number of seen pairs as English CLIP downsamples English training pairs and harms English performance." This reasoning correctly identifies dilution as a cause of the curse, but the converse — that *restoring* English exposure via extended training might account for all or most of the recovery — is never tested.

**What evidence exists in the paper.** Table 1 provides the key data. The Worldwide 1.0× run (13B pairs, same compute as English-only) achieves 79.5% ImageNet — below English-only at 80.4%, confirming that adding non-English data at equal compute hurts. The Worldwide 2.3× run (29B pairs, 2.3× compute) recovers to 81.3%. The gap between these two worldwide runs (79.5% → 81.3%, +1.8 points) could be decomposed into (a) restoration of full English exposure and (b) any additional benefit from non-English data. Without an English-only 29B-pair baseline, this decomposition is impossible. The paper does not report experiments that vary the English-only data volume to establish a scaling curve from which the expected English-only performance at 29B pairs could be estimated.

**Mitigation status.** Not addressed. The paper does not acknowledge this as a limitation. Section 3.4 frames the 2.3× factor as "proportional to the growth of data size from non-English pairs" and states the goal as ensuring "the amount of English seen pairs unchanged during the worldwide CLIP training," which is a design choice presented as correct rather than as a confound requiring a controlled comparison. No future work is suggested on this point.

---

### Difficulty Estimation for Worldwide Curation Requires 800 Parallel Jobs and Is Not Amortized in Any Efficiency Metric

**The assumption or constraint.** The worldwide curation algorithm (Algorithm 1, Section 3.3) requires substring matching of billions of alt-texts against per-language metadata lists containing millions of entries across 300+ languages. The paper reports (Appendix B) that this curation runs on 800 parallel jobs, each with 40GB CPU memory, and takes approximately 1 hour to complete substring matching and counting for all pairs. This is a substantial computational overhead that occurs *before training begins* and is not included in any of the paper's efficiency metrics. The seen-pairs comparison (29B for Meta CLIP 2 vs. 40B for mSigLIP) counts only training examples, not curation cost. The paper also does not report the total size of the raw data pool, the fraction of pairs that survive curation, or the total CPU-hours consumed by the Aho-Corasick matching, lazy metadata loading, and probabilistic sampling stages.

**The consequence.** A practitioner deciding whether to adopt Meta CLIP 2's recipe cannot accurately assess total computational cost. The 800-job, 1-hour pre-processing step requires access to a compute cluster with 32,000 total GB of CPU memory and the orchestration infrastructure to run 800 parallel jobs — resources that many academic labs and smaller industry teams do not have. Even for well-resourced organizations, the curation cost may be non-trivial compared to the training cost, especially if the raw data pool is periodically refreshed (requiring re-curation) or if the recipe is being applied to a new domain with different data sources. The paper's headline claim that Meta CLIP 2 uses "72% of mSigLIP's seen pairs" suggests better data efficiency, but if curation cost is included, the total computational efficiency picture might look quite different — particularly since mSigLIP's curation pipeline (proprietary Google Image Search-based) may have very different overhead characteristics that are not publicly documented for comparison.

**What evidence exists in the paper.** Appendix B explicitly provides the curation infrastructure details: "Our data curation algorithm is running in parallel with 800 jobs (each job has 40GB CPU memory) and it takes 1 hour to substring match and count for all alt-text pairs." Appendix A.2 describes the engineering optimizations (Aho-Corasick, lazy metadata loading, memory-mapped files) but does not report total CPU-hours, the scaling behavior with data volume, or whether the 1-hour figure includes all three stages of Algorithm 1 or only the matching and counting stages. Section 3.2 notes that metadata construction uses Wikipedia dumps from May 2024 across 329 languages, but does not report the cost of extracting, cleaning, tokenizing, and deduplicating this metadata.

**Mitigation status.** The paper acknowledges the engineering complexity implicitly by describing the optimizations in Appendix A.2, but does not frame the curation cost as a limitation. There is no discussion of how curation cost scales with data volume, whether it could be reduced (e.g., by sampling-based approximate counting, by caching metadata automata across data refreshes, or by using a smaller subset of languages for metadata matching), or whether the cost could be amortized over multiple training runs. The paper does not suggest future work on reducing curation overhead.

---

### The Recipe Is Validated on a Single Proprietary Dataset with a Single Model Family — No Evidence of Generalization to Other Data Sources, Model Architectures, or Training Paradigms

**The assumption or constraint.** All experiments use the authors' proprietary collection of publicly available image-text pairs with a specific language composition (~44% English after LID). The paper does not apply the Meta CLIP 2 curation recipe to any public benchmark dataset (e.g., LAION-5B, DataComp, COYO) or report results from training on such datasets. The model architecture is exclusively the standard ViT-based CLIP dual encoder with contrastive loss — no experiments test whether the recipe transfers to SigLIP-style sigmoid loss, to CNN-based vision encoders, to larger ViT variants (e.g., ViT-G), or to autoregressive vision-language models. The training paradigm is exclusively contrastive pretraining from scratch — no experiments test whether the curated worldwide data benefits fine-tuning, knowledge distillation, or SSL-based vision encoders like Web-DINO, despite the paper's claim that the recipe "benefits not only CLIP, but also efforts using CLIP data such as MLLM, SSL and image generation."

**The consequence.** The paper positions itself as providing a "recipe" — a term that implies reproducibility and transferability. But without validation on public data or alternative architectures, a practitioner cannot assess whether the recipe will work on their specific data source (with potentially different language distributions, alt-text quality, or image domains) or with their preferred model architecture. The paper's finding that ViT-H/14 is the inflection point for breaking the curse may be specific to the ViT architecture and the particular data volume — for a different architecture with different representational capacity per parameter, the inflection point might occur at a different scale. The claim about broader impacts on MLLM and SSL training data is entirely unvalidated: no MLLM or SSL model is trained on Meta CLIP 2-curated data and evaluated.

**What evidence exists in the paper.** All results in Tables 1-4 and Figures 1-4 use the authors' proprietary data and standard ViT CLIP architectures. The paper references Web-DINO (Fan et al., 2025) as showing "SSL has better scalability on Meta CLIP curated large-scale data," but this refers to the original English Meta CLIP curation, not Meta CLIP 2's worldwide curation — and Web-DINO is a separate study, not an experiment conducted in this paper. Section 1 claims broader impacts on "MLLM, SSL and image generation," but these are aspirational statements, not demonstrated results. The paper does not include any out-of-distribution evaluation (e.g., testing the trained model on a completely different image domain or text style) beyond the standard zero-shot transfer benchmarks.

**Mitigation status.** The paper does not acknowledge this as a limitation. It frames the deliberate maximization of overlap with OpenAI CLIP's architecture as a strength ("The overlap makes our findings generalizable to CLIP and its variants"), but this is about the recipe's design philosophy, not about empirical validation of generalizability. The paper open-sources the metadata, curation code, and training recipe (as stated in the header), which partially mitigates the concern by enabling other researchers to test generalization — but the paper itself provides no such evidence.

---

### The ViT-H/14 Inflection Point Claim Is Supported by Only Two Data Points (L vs. H) With No Intermediate Scales Tested

**The assumption or constraint.** The paper's most important conceptual claim — that there exists a capacity threshold above which worldwide data helps English rather than hurts it — is supported by exactly two model scales: ViT-L/14 (~300M parameters) and ViT-H/14 (~630M parameters). Figure 1 shows ViT-L/14 worldwide 2.3× dropping to 78.8% (below 79.5% English-only) while ViT-H/14 worldwide 2.3× rises to 81.3% (above 80.4% English-only). No intermediate scales are tested — for instance, ViT-L with increased width (which might reach ~450M parameters), ViT-g (~1B parameters), or a ViT-B scaled up to match ViT-L parameter count through increased width. The paper also does not test whether the same inflection occurs at smaller data volumes (e.g., would a ViT-L trained on a smaller worldwide dataset break the curse, suggesting the threshold depends on the data-to-parameter ratio rather than absolute parameter count?) or with different data compositions (e.g., a 50/50 English/non-English split at ViT-L scale).

**The consequence.** A practitioner reading this paper might conclude that ViT-H/14 is *necessary* to break the curse — that any model smaller than H scale cannot benefit from worldwide data. But the experimental design only demonstrates that L scale is *insufficient* under the specific data conditions tested; it does not establish that H scale is *minimal*. A model at 400M or 500M parameters might also break the curse, which would have significant practical implications for deployment scenarios where H-scale inference is too expensive or slow. More fundamentally, without multiple data points on the scaling curve, the paper cannot establish the *functional form* of the relationship between capacity and the worldwide/English performance gap. Is the transition sharp (the curse breaks abruptly at a specific parameter count) or gradual (the gap narrows continuously with scale, crossing zero at some point)? The paper's framing of an "inflection point" implies the former, but two data points can fit either a step function or a smooth curve.

**What evidence exists in the paper.** Figure 1 (left panel) and Table 1 provide the L vs. H comparison. The paper also tests ViT-B/32 (Table 2) as a much smaller model, but these experiments use Worldwide 1.0× (not 2.3×) and mT5 tokenizer, making them not directly comparable to the main L/H comparison. The B/32 worldwide result (64.7% ImageNet in Table 2 row 5) is below the B/32 English-only baseline (67.5% in row 1), confirming the curse persists at B scale — but this is 1.0× seen pairs with a different tokenizer, so it doesn't cleanly extend the scaling curve.

**Mitigation status.** The paper does not acknowledge the sparsity of the capacity scaling evidence. It treats the L → H jump as sufficient to establish the inflection point claim. Future work on mapping the exact scaling relationship is not suggested.

---

### The Difficulty Threshold Invariance Assumption (6% Tail Proportion Across All Languages) Is Untested — Yet It Controls the Core Curation Mechanism

**The assumption or constraint.** The worldwide curation algorithm's central mathematical move is to compute a single global tail proportion `p` from the English-calibrated threshold `t_en` (via `t_to_p`) and then transfer this proportion to every other language to derive language-specific thresholds `t_lang` (via `p_to_t`). This assumes that the optimal balance between head and tail concepts — approximately 6% of concept matches coming from rare concepts — is invariant not only across data scales (validated by Meta CLIP for English) but also across languages. The paper states: "With this assumption, we determine t in two steps" (Section 3.3). The 6% figure derives from OpenAI CLIP's tuning on English data and Meta CLIP's subsequent validation at English scales of 400M to 2.5B pairs; it has never been validated for any non-English language, let alone the 300+ languages in the worldwide metadata.

**The consequence.** If the optimal tail proportion varies by language — and there are plausible reasons it might — the worldwide curation is systematically miscalibrated for languages where 6% is suboptimal. For example, a language with very high-quality, professionally written alt-texts (e.g., a Wikipedia-dominated language) might have a longer tail of useful rare concepts that deserve more than 6% of training signal. Conversely, a language with noisy, spam-heavy alt-texts might have a tail full of garbage concepts (typos, fragments, non-visual text) that should be suppressed below 6%. The paper observes that some languages have vastly different data volumes and likely different alt-text quality distributions. Applying the English-optimal tail proportion to all of them is computationally convenient but may be leaving performance on the table for many languages. Table 2 shows that language-specific thresholds (row 5) outperform a single global threshold (row 4), but this only tests the *existence* of per-language thresholds, not their *optimality* — the derived `t_lang` values could still be suboptimal for specific languages if 6% is not the right target.

**What evidence exists in the paper.** None. The paper does not ablate the tail proportion `p`. It does not test alternative values (e.g., 3%, 10%, 15%) for any language or in aggregate. It does not report the language-specific `t_lang` values that the algorithm produces, which would allow readers to assess whether the derived thresholds are plausible for languages of different sizes. It does not break down multilingual benchmark performance by language resource level (high/medium/low), which might reveal systematic differences in curation quality driven by the invariant assumption.

**Mitigation status.** The paper does not acknowledge the invariance assumption as a potential limitation. It is presented as a natural extension of the Meta CLIP design philosophy: "the invariance assumption adopted in Meta CLIP algorithm design, the percentage of tail matches (i.e., 6%), and apply it across languages." The phrase "apply it across languages" assumes what needs to be tested — that the invariance holds cross-lingually. No future work is suggested on language-specific tail proportion tuning.

---

### All Multilingual Results Are Single-Run Point Estimates With No Confidence Intervals, Statistical Tests, or Multi-Seed Validation — Yet the Paper Claims State-of-the-Art on Multiple Benchmarks

**The assumption or constraint.** Every number in Tables 1-4 and Figures 1-4 is a single-run point estimate. The paper does not report standard deviations across training runs with different random seeds, bootstrap confidence intervals, or any form of statistical significance testing. The test sets for several multilingual benchmarks are small — Babel-ImageNet uses 280 languages, but the number of images per language is not reported and is likely in the low dozens for many languages given the total test set size constraints. CVQA (Mogrovejo et al., 2024) is a curated benchmark with culturally diverse questions, but per-language or per-region sample sizes are not reported in the paper. For retrieval benchmarks like XM3600, Recall@1 on small per-language query sets can be highly sensitive to a few ranking changes.

**The consequence.** A 0.9-point difference on ImageNet (81.3% vs. 80.4%) corresponds to approximately 450 correctly classified images out of 50,000. Without variance estimates, we cannot assess whether this difference is reliably attributable to the worldwide recipe or falls within the range of training noise (which for large-scale CLIP training can be substantial — different random seeds can produce variations of 0.5-1.5% on ImageNet depending on data order, batch composition, and initialization). This matters most acutely for the core "curse-breaking" claim: if the English-only ViT-H/14 run at 80.4% has a standard deviation of ±0.5%, and the worldwide run at 81.3% has a similar deviation, the difference may not be statistically significant at conventional thresholds. The paper's comparison to mSigLIP (80.6%) similarly hinges on a 0.7-point gap that may not be reliable. For multilingual benchmarks with smaller test sets, the variance is likely larger. The paper's SoTA claims on Babel-IN (+3.8%), CVQA (+7.6%), and other benchmarks are presented without any quantification of uncertainty.

**What evidence exists in the paper.** The paper reports all results as single-point numbers without error estimates. Section 4 describes the evaluation methodology in terms of which benchmarks and metrics are used, but does not mention multiple training runs, seed variation, or statistical testing. The 5k holdout set used for alignment/uniformity analysis (Figure 4, Section 4.2.4) is a single fixed split, and the paper explicitly acknowledges contamination concerns for external baselines on this set but does not address variance in its own measurements. The paper does not report the number of test samples per language for any multilingual benchmark.

**Mitigation status.** Not addressed. The paper does not acknowledge the absence of variance estimates as a limitation. Multi-seed training is expensive at ViT-H/14 scale, but even a single alternative evaluation (e.g., bootstrap resampling of test set predictions to estimate confidence intervals, which requires no additional training) is not performed. No future work on statistical reliability is suggested.

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
