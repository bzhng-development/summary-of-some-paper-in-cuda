# Dolma: an Open Corpus of Three Trillion Tokens for Language Model Pretraining Research

**ArXiv:** [2402.00159](https://arxiv.org/abs/2402.00159)
**Authors:** Luca Soldaini, Rodney Kinney, Akshita Bhagia, Dustin Schwenk, David Atkinson, Russell Authur, Ben Bogin, Khyathi Chandu, Jennifer Dumas, Yanai Elazar, Valentin Hofmann, Ananya Harsh Jha, Sachin Kumar, Li Lucy, Xinxi Lyu, Nathan Lambert, Ian Magnusson, Jacob Morrison, Niklas Muennighoff, Aakanksha Naik, Crystal Nam, Matthew E. Peters, Abhilasha Ravichander, Kyle Richardson, Zejiang Shen, Emma Strubell, Nishant Subramani, Oyvind Tafjord, Pete Walsh, Luke Zettlemoyer, Noah A. Smith, Hannaneh Hajishirzi, Iz Beltagy, Dirk Groeneveld, Jesse Dodge, Kyle Lo
**Institutions:** Allen Institute for AI (AI2)

## 🎯 Pitch

Dolma introduces a pioneering 3-trillion-token English corpus with a robust curation toolkit, advancing reproducible research in large language model (LLM) training. By openly documenting data curation processes and offering a scalable, diverse dataset, Dolma empowers researchers to systematically explore how data composition influences model behavior, ultimately making pretraining data a transparent, first-class research object.

---

## 1. Executive Summary (2–3 sentences)
This paper releases Dolma, an openly available corpus of roughly three trillion English tokens purpose‑built for training and studying large language models (LLMs). Beyond the data itself, it contributes a high‑performance curation toolkit and a series of controlled ablation experiments that show how concrete data decisions (e.g., filtering, deduplication, source mixing) affect downstream model performance and domain coverage (§§4–8, Table 1, Figs. 1–5).

## 2. Context and Motivation
- Problem addressed
  - Pretraining data for top LLMs is largely undisclosed, unreleased, or insufficiently documented, which blocks reproducible science on how data composition affects model behavior (§1; Appendix C surveys the opacity around PaLM 2, GPT‑4, Claude, Llama 2).
- Why this matters
  - Practical: model behavior is tightly linked to pretraining data distribution; users and developers need to know what data their systems learned from (e.g., performance advantages on in‑distribution tasks; risk of toxic or private content exposure) (§1, motivation bullets).
  - Scientific: without access to corpora and recipes, research on memorization, deduplication, contamination, toxicity, and attribution stalls (§1; related work §2).
- Prior open corpora and gaps
  - C4 (≈175B tokens) and The Pile (≈387B) are high‑quality but small for today’s compute budgets (§2).
  - ROOTS is large but largely multilingual, leaving relatively few English tokens for English‑only studies (§2).
  - RedPajama v2 and RefinedWeb reach massive scale but are web‑only, lacking source diversity (code, papers, social) that many modern LLMs rely on (§2).
  - RedPajama v1 is multi‑source but ≈1.2T tokens and has known cleanliness issues (§2).
- Positioning
  - Dolma targets both scale (2–3T tokens) and diversity (web, code, papers, books, social, encyclopedic) while being fully released and reproducible (Table 1; §3 goals).
  - It pairs the corpus with an open toolkit and ablation‑backed design choices to enable and accelerate data‑centric LLM research (§§4–8).

## 3. Technical Approach
This section explains how Dolma is built and why each step exists. A token is a subword unit produced by a tokenizer; “3T tokens” is measured with the LLaMA tokenizer (Table 1).

- Design goals (§3)
  - Match common LLM data recipes where known (to make results comparable).
  - Prefer evidence‑based choices via ablations when recipes diverge or are unknown (§4.2).
  - Scale to 2–3T tokens to support large‑model training.
  - Preserve openness and mitigate legal/ethical risks (e.g., avoid certain proprietary books; perform PII filtering) (§3).

- The Dolma Toolkit (§4.1)
  - Two core operations:
    - `filtering`: configurable language/quality/content filters applied at document or sub‑document level. It accepts a text unit (document, paragraph, sentence), a scoring method (e.g., classifier score, perplexity, regex), and a removal policy (delete or replace). The system parallelizes these operations, processing ≈122 CPU‑hours per TB; a 200 TB raw pool can be filtered in ≈5 days on a 192‑vCPU instance (§4.1).
    - `mixing`: cross‑file operations such as up/down‑sampling, deduplication, and test decontamination implemented in Rust. It includes a `Bloom filter` (a probabilistic set membership data structure) to detect near‑duplicates or seed decontamination with known test examples (§4.1).
  - Why it matters: unifies, speeds up, and open‑sources data curation that is often bespoke, slow, and closed (§4.1).

- Empirical recipe selection with ablations (§4.2)
  - Models: 1.2B parameter decoder‑only `OLMo` variants, trained up to 150B tokens to compare data decisions while holding architecture and training fixed (§4.2, Appendix D.1).
  - Evaluation: zero‑shot accuracy on eight common tasks (ARC‑E/C, BoolQ, HellaSwag, OpenBookQA, PIQA, SciQ, WinoGrande) and perplexity on a diverse, stratified evaluation suite (`Paloma`) that spans many domains (§4.2; Appendix D.2–D.3).
  - Rationale: control for confounders and directly attribute performance changes to data curation choices.

- Source‑by‑source curation (Table 1; §§5–8)
  1) `Dolma‑Web` (Common Crawl; 2.28T tokens)
     - Acquisition & language filtering (§5.1): start with 25 CC snapshots (2020‑05 to 2023‑06). Use `CCNet` to detect English with a `FastText` classifier (keep score ≥ 0.5) and to remove large volumes of repeated boilerplate paragraphs by grouping shards and deduping paragraphs within each group. This reduces ≈175 TB to 27.7 TB (84.2% filtered).
     - Quality filtering (§5.2): adopt a heuristic combo that outperforms alternatives in ablations (Fig. 1):
       - `Gopher All` rules (from Gopher) plus a single C4 rule `NoPunc` that drops paragraphs not ending in punctuation. These tag 15.23% and 22.73% of characters for removal, respectively.
       - Notably, CCNet’s model‑based “Wikipedia‑likeness” buckets (via `KenLM` perplexity) are largely orthogonal to these heuristics, so combining them captures different “quality” signals (§5.2).
     - Content filtering (§5.3): sentence‑level toxicity detection with two fast `FastText` classifiers trained on Jigsaw Toxic Comments to remove “hate” or “NSFW” sentences. Two thresholds were tried; the stricter “Low” threshold removes ≈29–35% of sentences and yields better accuracy, but the “High” threshold (τ=0.4) removes ≈6–7% and was chosen to preserve enough data after all filters compound (Fig. 2; §5.3).
       - PII removal/masking: regex for emails, IPs, phone numbers; replace spans with special tokens if ≤5 spans or drop the document otherwise. Only ≈0.02% masked and 0.001% removed, with negligible effect in ablations (§5.3; Appendix I).
     - Deduplication (§5.4): three stages using the toolkit’s Bloom filter—exact URL dedup (removes 53.2% of documents), exact document dedup (removes 14.9% of URL‑deduped), and exact paragraph dedup (removes 18.7% of paragraphs). Ordered to maximize efficiency and avoid disrupting later filtering (§5.4).
     - Combined effect: stacking quality filters, dedup, then content filters yields a clear, compounding performance gain (Fig. 3; §5.5).
  2) `Dolma‑Code` (GitHub via The Stack; 411B tokens; §6)
     - Start from an already deduplicated, permissively licensed code snapshot.
     - Apply RedPajama‑style heuristics (e.g., remove license preambles and non‑source large files) plus `StarCoder` rules (e.g., code‑to‑comment ratios, low‑star repos; §6.2). Ablations show the combined rules lower code perplexity and improve downstream task accuracy vs. RedPajama rules alone (Appendix O.7).
     - Detect secrets (`detect‑secrets`) and mask PII (§6.3).
  3) `Dolma‑Social` (Reddit; 89B tokens in Table 1 say 89B Llama tokens; text states 80B; §7)
     - Acquisition from Pushshift: 378M posts (2005–03/2023), including submissions and comments (§7.1).
     - Thread formatting ablation (§7.1; Fig. 4): three options—atomic messages, partial threads, full threads. Treating each comment/submission independently (“Atomic Content”) performs best; complex linearizations introduce formatting artifacts that hurt training.
     - Quality filters (§7.2): minimum lengths (comments ≥500 chars; submissions ≥400), minimum votes (≥3), remove deleted/removed/NSFW, and exclude a curated list of banned or NSFW subreddits (26,123) (§7.2).
     - Content filters and dedup (§§7.3–7.4): toxicity filtering, PII removal (entire document due to shortness), and document‑level deduplication.
  4) Other curated sources (§8)
     - `C4`: reprocessed through Dolma’s pipeline for extra cleanup (PII masking, further deduplication).
     - `peS2o` (Semantic Scholar) for academic papers: used as‑is (already cleaned for LM pretraining).
     - `Project Gutenberg` (public‑domain books): language ID and short‑page removal; dedup by title.
     - `Wikipedia/Wikibooks` (English + Simple): process with WikiExtractor; drop very short pages.

- Contamination control for evaluations
  - For perplexity evaluations on Paloma, the mixer’s Bloom filter marks any paragraph in Dolma that exactly matches Paloma content (≥13 Unicode‑words) and removes documents containing such paragraphs (§L). The impact on training data volume is negligible (≤0.02% documents removed).

## 4. Key Insights and Innovations
- A massive, multi‑source open corpus at 3T tokens (Table 1)
  - Innovation: brings both scale and source diversity (web, code, papers, books, social, encyclopedic) in one openly downloadable corpus. Prior public corpora either had fewer English tokens (C4, Pile, ROOTS) or were web‑only at very large scale (RedPajama v2, RefinedWeb) (§2).
  - Significance: enables training and, crucially, reproducible research on pretraining data practices at modern LLM scales.

- Open, high‑throughput curation toolkit that generalizes beyond this dataset (§4.1)
  - Innovation: unified `filtering` and `mixing` APIs, with scalable Bloom‑filter dedup and test decontamination and a performance profile suitable for hundreds of terabytes.
  - Significance: lets others reproduce and iterate on data recipes at web scale (e.g., “122 CPU‑hours/TB”; parallelizable; §4.1).

- Evidence‑backed filtering choices that beat common heuristics (§5.2; Figs. 1 & 3)
  - Innovation: ablations identify a specific heuristic combination—`Gopher All + C4 NoPunc`—that outperforms using all C4 rules or all Gopher rules alone on both perplexity and downstream accuracy (e.g., HellaSwag in Fig. 1). Stacking quality filters, dedup, then toxicity filtering compounds gains (Fig. 3).
  - Significance: moves the field from opaque or ad‑hoc rules to validated, reproducible recipes.

- Domain‑fit analysis at scale showing why multi‑source data matters (Fig. 5)
  - Innovation: controlled training of 1.2B models on 150B‑token samples from various open corpora shows `Dolma` and `Pile` achieve substantially lower perplexity across a stratified, domain‑diverse evaluation (`Paloma`) than single‑source web corpora (`C4`, English `mC4`, `RefinedWeb`) (Fig. 5).
  - Significance: quantifies the value of including non‑web sources for broad generalization.

- Practical guidance on contentious decisions (toxicity thresholds, Reddit formatting, code filters)
  - Examples: choosing the less aggressive toxicity threshold to preserve scale under compounding filters (§5.3; Fig. 2); showing atomic Reddit messages beat thread reconstructions (§7.1; Fig. 4); and combining RedPajama + StarCoder code heuristics improves metrics (§6.2; O.7).
  - Significance: researchers can adopt these settings with clear trade‑off rationales.

- Baseline model `OLMo‑1B` trained on Dolma with competitive results (§9; Table 2)
  - Purpose: sanity‑check data pipeline and provide a public reference model. `OLMo‑1B` averages 60.3 across the evaluation suite, outperforming TinyLlama (59.4) and far ahead of Pythia 1.1B (54.5); it’s close to StableLM2 1.6B on several tasks (Table 2).

## 5. Experimental Analysis
- Evaluation setup (§4.2; Appendix D)
  - Models: 1.2B parameter `OLMo` variants trained for 150B tokens for ablations; the final `OLMo‑1B` trained to 3.1T tokens (§9.1; Appendix D.4).
  - Downstream tasks: eight zero‑shot classification tasks—ARC‑E/C, BoolQ, HellaSwag, OpenBookQA, PIQA, SciQ, WinoGrande (Appendix D.3).
  - Perplexity: `Paloma` suite covers diverse domains (news, Wikipedia, scientific papers, social platforms, code; Appendix D.2). Lower is better.
  - Contamination checks: Bloom‑filter decontamination for Paloma; WIMBD‑based analysis flagged heavily contaminated academic/GLUE/SuperGLUE datasets (often via code repos), which were excluded from evaluation (§L, Fig. 11).

- Main quantitative findings (selected)
  - Scale and composition (Table 1)
    - >3T LLaMA tokens across ≈4.37B documents; ≈11 TB UTF‑8 after curation from ≈200 TB raw input. Major sources: Common Crawl (2.48T tokens), GitHub (411B), Reddit (89B), Semantic Scholar (70B), Gutenberg (6B), Wikipedia/Wikibooks (4.3B).
  - Web quality filters (Fig. 1; §5.2)
    - `C4 NoPunc + Gopher All` improves accuracy over baselines and over either rule set alone (e.g., see HellaSwag accuracy rising faster with training tokens in Fig. 1). Gains replicate across other tasks (Appendix O).
  - Toxicity thresholds (Fig. 2; §5.3)
    - More aggressive removal (“Low” threshold) boosts accuracy but drops much more data (~30% of sentences). Due to the compounded effect with quality and dedup filters (low correlation yet additive removal; Fig. 9 in Appendix J), Dolma v1.6 adopts the more permissive “High” threshold (≈6–7% sentence removal) to preserve scale while still seeing gains over no‑filtering.
  - Deduplication stages (§5.4)
    - URL dedup removes 53.2% of documents; exact document dedup removes an additional 14.9%; paragraph dedup removes 18.7% of paragraphs. Results in more token‑efficient training and further performance gains when stacked with quality and toxicity filters (Fig. 3; and ablations in O.2 and O.5).
  - Reddit formatting (Fig. 4; §7.1)
    - Treating comments/submissions as independent (“Atomic Content”) yields higher accuracy than building partial or full threads; likely because thread formatting injects unnatural patterns or repeated text that hurts language modeling.
  - Code heuristics (Appendix O.7)
    - Combining RedPajama + StarCoder filters yields lower code perplexity (e.g., HumanEval) and improved downstream task accuracy over using RedPajama rules alone.
  - Domain fit (Fig. 5; §9.2)
    - 1.2B models trained on 150B tokens from `Dolma` closely match `Pile` and outperform single‑source corpora (`C4`, English `mC4`, `RefinedWeb`) across Paloma’s diverse sources. The plot shows lower perplexity curves (left‑shifted down) for multi‑source datasets.
  - `OLMo‑1B` comparison (Table 2; §9.1)
    - On eight tasks, `OLMo‑1B` averages 60.3 vs. TinyLlama 59.4 and Pythia 54.5. Highlights:
      - HellaSwag: 62.5 (OLMo‑1B) vs. 58.7 (TinyLlama) and 44.7 (Pythia).
      - PIQA: 73.7 (OLMo‑1B) vs. 71.1 (TinyLlama) and 69.1 (Pythia).
      - ARC‑E: 58.1 (OLMo‑1B) vs. 53.2 (TinyLlama) and 50.2 (Pythia).
    - StableLM2‑1.6B averages higher (66.5), but it’s larger and trained differently; the `OLMo‑1B` results validate Dolma’s overall quality while keeping the model and training fully open.

- Robustness and diagnostics
  - Filter correlations are low (Fig. 9), which explains why their removal effects compound: each filter family targets different artifacts (e.g., PII vs. toxicity vs. randomness vs. duplication).
  - Dialectal bias check for toxicity: using country‑subreddit proxies, the FastText classifier’s toxic‑label rate differs <5% across locations for thresholds between 0.1–0.9 (Appendix H, Fig. 8), indicating limited dialectal skew from that specific detector.
  - Tokenizer analysis (Appendix F, Fig. 6): code has high “fertility” (≈2.45 tokens per word) due to whitespace tokens (`\n`, `\t`), suggesting potential value in adding code‑specific tokens if training code‑heavy models.

- Do the experiments support the claims?
  - Yes, for the paper’s stated scope. The ablations link concrete data choices to measurable changes in perplexity and accuracy (Figs. 1–4; O.2–O.10). Domain‑fit analysis (Fig. 5) justifies multi‑source composition. `OLMo‑1B` performance (Table 2) demonstrates that models trained on Dolma are competitive for their size/training budget.
  - Caveat: ablations use 1.2B models trained to 150B tokens (not full convergence), so some effects may differ at larger scales (§Limitations).

## 6. Limitations and Trade-offs
- English‑only focus (§Limitations)
  - Assumption: English is the target; tools and pipeline are tuned for English. Some non‑English may remain due to language ID errors, but models trained solely on Dolma are not expected to perform well on non‑English tasks.
- Representativeness (§Limitations)
  - Dolma cannot include sources that are legally restricted or practically unavailable (e.g., certain copyrighted books), so it is not a full surrogate for closed‑model training mixtures. Web‑scale sampling can over‑ or under‑represent communities and topics (§N.7).
- Ablation scale and model size (§Limitations)
  - The controlled studies use 1.2B models trained to 150B tokens—sufficient to compare data choices efficiently, but not the same regime as 7B–70B LLMs trained on multi‑trillion tokens. Some data decisions could have different effects at larger scales.
- Task coverage (§Limitations)
  - Evaluation targets base‑LM abilities in zero‑shot classification; it does not include instruction‑following, code execution, or safety‑specific stress tests. Effects of code on executable reasoning only appear after specialized fine‑tuning (Appendix M on GSM8K via PAL).
- Filtering trade‑offs (§5.3; Appendix J)
  - Aggressive toxicity removal (“Low” threshold) improves accuracy but drops ~30% of sentences; with quality and dedup filters, total removal becomes too high. Choosing the “High” threshold preserves scale but accepts more borderline content.
  - PII detection is regex‑based for speed; it targets email/IP/phone with high precision but misses other PII types (Appendix I). Model‑based PII detection at this scale was impractical.
- Legal/ethical uncertainty (§N.7)
  - Copyright and fair‑use interpretations vary by jurisdiction and are evolving. Dolma uses publicly available sources and provides a takedown request mechanism, but it cannot perfectly guarantee the absence of copyrighted material (§N.7 Ethical Considerations).
- Compute and storage demands
  - Although the toolkit is efficient (e.g., 122 CPU‑hours/TB), end‑to‑end processing still requires substantial compute and storage (≈200 TB raw → 11 TB processed). Training at 3T‑token scale also demands significant GPU resources (Appendix D.4).

## 7. Implications and Future Directions
- How this changes the landscape
  - Provides the community with a modern‑scale, multi‑source, fully open pretraining dataset plus a robust toolkit and documented, ablation‑validated recipes. This unlocks reproducible data‑centric LLM research at a scale that previously required proprietary datasets or closed infrastructure.
- What this enables
  - Systematic studies of:
    - Memorization, contamination, and attribution on the actual training data (Appendix L for initial contamination analysis).
    - Deduplication strategies beyond exact matching (e.g., semantic or fuzzy dedup at web scale using the mixing API).
    - Safety filtering and fairness (e.g., alternative toxicity/PII detectors; content governance frameworks mentioned in §Ethical Considerations).
    - Tokenizer design and domain fertility (Appendix F suggests code‑aware tokens).
    - Mixture optimization: programmatic searches over upsampling/downsampling of sources (Appendix M gives starter mixes and their effects).
- Practical applications
  - Training transparent base LLMs, code LLMs, or domain specialists (e.g., scientific) without relying on undisclosed data. The `OLMo‑1B` baseline (§9) is a first instance; the paper notes later Dolma versions (e.g., v1.7) already yield “significant performance improvement” holding the model constant (§Conclusion).
- Concrete next steps (suggested by the paper’s findings)
  - Expand beyond English while retaining the same transparency (noted as a goal in §3 and §Limitations).
  - Start from more CC snapshots to enable stricter toxicity thresholds without falling short on target token counts (§5.3).
  - Integrate faster or specialized PII detectors as they mature; extend to more PII types.
  - Optimize tokenizer vocabularies for code and other high‑fertility domains (Appendix F).
  - Standardize decontamination protocols (tooling is present; develop community norms around thresholding and matching strategies).

> Key takeaway: Dolma is not just a large open dataset; it is a documented, evidence‑backed recipe and an industrial‑grade toolkit that together make pretraining data a first‑class, reproducible research object. The paper shows, with figures and ablations (Figs. 1–5; O.2–O.10), how concrete data choices move downstream metrics—turning opaque “data quality” talk into testable, reusable practice.
