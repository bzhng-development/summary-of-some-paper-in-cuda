# Dolma : an Open Corpus of Three Trillion Tokens for Language Model Pretraining Research

**ArXiv:** [2402.00159](https://arxiv.org/abs/2402.00159)

## 🎯 Pitch

Dolma introduces a fully open, three-trillion-token English pretraining corpus alongside an open-source curation toolkit, faithfully reproducing and documenting the data mixtures and curation recipes behind state-of-the-art language models. By enabling large-scale, transparent, and reproducible experimentation on how concrete data curation choices (from quality and toxicity filtering to deduplication and source mixing) affect model behavior, Dolma removes a critical barrier to scientific progress, safety analysis, and practical understanding in language model research—a breakthrough for both open innovation and model accountability.

---

## 1. Executive Summary (2-3 sentences)
Dolma is a fully open, three‑trillion‑token English pretraining corpus plus a matching, open‑source curation toolkit that reproduces at scale the kinds of data mixtures used in state‑of‑the‑art language models. It matters because it turns opaque, irreproducible data pipelines into a documented, testable artifact and shows—through controlled ablations—how concrete curation choices (quality rules, toxicity thresholds, deduplication, PII handling, source mixes) affect downstream performance (§4.2; Figs. 1–3; Table 4).

## 2. Context and Motivation
- Problem addressed
  - Pretraining data for powerful language models is usually undisclosed or under‑documented, even when models are open (e.g., Llama 2, Mistral) or entirely closed (e.g., GPT‑4, PaLM 2) (§2; App. C). This prevents scientific understanding of how data composition influences capabilities, safety, and limitations.
- Why it matters
  - Practical impact: A model’s strengths and biases correlate with what it sees during pretraining (e.g., memorization risks, benchmark leakage, social biases) (§1; references therein).
  - Scientific impact: Without access to comparable corpora and repeatable recipes, researchers cannot systematically test data‑centric interventions (deduplication, filtering, decontamination) at scale.
- Prior approaches and their gaps
  - Smaller open corpora exist (C4: ~175B tokens; The Pile: ~387B; ROOTS: multilingual with ~30% English), which are either too small for current training regimes or not English‑focused (§2).
  - Larger open corpora exist but are web‑only or lightly curated (RefinedWeb/Falcon; RedPajama v2) (§2).
  - RedPajama v1 (~1.2T tokens) is close in spirit but misses data families (e.g., larger academic papers, social media) and has quality issues identified by subsequent audits (§2).
- Positioning of this work
  - Dolma aims for “compute‑relevant” scale (2–3T tokens), multi‑source diversity (web, code, papers, books, social, encyclopedias), and full transparency: the dataset, tooling, and ablation evidence are all released (§1, §3; Table 1; links on p.1).

## 3. Technical Approach
This section describes how Dolma is built and why each design choice was made, then how choices were validated.

- Conceptual structure: a “data operating system” plus a corpus
  - The Dolma Toolkit abstracts two families of operations (§4.1):
    - `Filtering`: Apply language ID, quality heuristics, content filters (toxicity, PII), or other scorers to remove or edit text at document or subdocument granularity. It parallelizes these operations at web scale and supports speed‑critical use cases (122 CPU hours/TB in internal tests replicating C4; §4.1).
    - `Mixing`: Cross‑file operations such as up/down‑sampling sources, deduplication, and decontamination. Implemented in Rust, with a `Bloom filter` (a compact probabilistic set membership structure) for linear‑time duplicate detection and for test set decontamination (§4.1).

- Data ablation framework to justify curation decisions (§4.2)
  - Train a controlled 1.2B‑parameter OLMo model up to 150B tokens (same optimizer/schedule; App. D.1) for each intervention and compare against a baseline.
  - Evaluate zero‑shot on eight standard tasks (ARC‑E/C, BoolQ, HellaSwag, OpenBookQA, PIQA, SciQ, WinoGrande) using standardized prompts and truncation (App. D.3).
  - Track perplexity (lower is better predictive fit) on Paloma, a stratified set of diverse sources (App. D.2).

- End‑to‑end pipelines by source (what runs and how)
  1) Web data (`Dolma‑Web`; §5)
     - Acquisition + language ID: Process 25 Common Crawl snapshots (2020‑05—2023‑06). Use CCNet with FastText language ID; keep English score ≥ 0.5; CCNet also removes frequent boilerplate paragraphs. CCNet reduces web bytes by 84.2% (175.1 TB → 27.7 TB; §5.1).
     - Quality filtering: Combine `Gopher` rules (e.g., n‑gram repetition, symbol ratio; §5.2 & Datasheet §N) and a single C4 rule (`NoPunc`: remove paragraphs not ending in punctuation). These remove 15.23% and 22.73% of characters, respectively.
     - Content filtering: Two lightweight FastText classifiers trained on Jigsaw Toxic Comments: one for “hate”, one for “NSFW”. Apply at sentence level with a chosen threshold (τ). Low τ (~0.0004) removes ~29–35% sentences and improves accuracy more; High τ (~0.4) removes ~5.5–7.3% sentences but preserves scale; v1 uses High τ to ensure a 3T‑token target (§5.3; Fig. 2; footnote 10).
     - PII handling: Fast regexes for emails, phone numbers, IPs. Mask up to 5 spans (0.02% docs) with special tokens; drop doc if ≥6 spans (0.001% docs). Ablations show negligible effect on performance given rarity (§5.3; App. I; Fig. 33).
     - Deduplication: Three stages using the Bloom filter: exact URL (removes 53.2% docs), exact document (removes 14.9% of remaining), and exact paragraph (removes 18.7% of remaining paragraphs) (§5.4).
     - Outcome synergy: Stacking quality + dedup + content filtering yields a “compounding” improvement over baseline (Fig. 3).

  2) Code data (`Dolma‑Code`; §6)
     - Source: `The Stack` (permissively licensed GitHub code, deduplicated; §6.1).
     - Filtering: Combine RedPajama v1’s code heuristics (e.g., max line length, data‑like files) and StarCoder rules (e.g., repo stars, comment ratios), which in ablations lowers perplexity on code and improves downstream tasks vs. RedPajama‑only rules (App. O.7; Figs. 56–61).
     - Content: Apply the same PII regex masking and run `detect-secrets` to drop files with secrets (§6.3).

  3) Social data (Reddit; §7)
     - Source: 378M posts/comments via Pushshift (2005–03/2023).
     - Formatting ablation: Treat Reddit elements as independent “atomic” documents vs. stitching partial or full threads. “Atomic” performs best; thread formatting introduces noise (§7.1; Fig. 4).
     - Quality: Keep longer items (comments ≥500 chars; submissions ≥400), minimum 3 upvotes, remove deleted/NSFW/banned‑subreddit content (§7.2).
     - Content/PII: Apply the same toxicity and PII rules (whole‑doc removal for PII due to short length; §7.3).

  4) Additional sources (curated reference material; §8)
     - `C4` reprocessed through the web pipeline (minus URL dedup) for extra cleanup (§8).
     - `peS2o` (open‑access academic papers from Semantic Scholar; used as‑is; §8).
     - `Project Gutenberg` public‑domain books (English only; §8).
     - `Wikipedia` and `Wikibooks` (English + Simple; cleaned via WikiExtractor and length filters; §8).

- Mixing strategies (what proportion of each source to sample)
  - The toolkit does not fix a single mixture; instead, it supports controlled comparisons. Four concrete mixes are analyzed (Table 4): `Naïve` (equal by source), `Web‑Only`, `Reference+` (upsample papers/books/wiki 2×), and `Gopher‑like` (heavily reference‑weighted).
  - Mixing results: web‑only matches others on generic web domains but underfits code and papers; reference upsampling reduces perplexity on academic text without large amounts of extra data (§M; Fig. 12).

- Definitions for uncommon terms used above
  - `Perplexity`: a measure of how well a model predicts text; lower means better predictive fit.
  - `Bloom filter`: a tiny, probabilistic data structure to test if an item is in a set, with no false negatives and controlled false positives, enabling fast deduplication (§4.1).
  - `Decontamination`: removing training text that duplicates test set content to avoid leakage (implemented by seeding a Bloom filter with test paragraphs; §4.1; App. L).

## 4. Key Insights and Innovations
1) A compute‑scale, multi‑source, reproducible open corpus (Table 1; §3)
   - What’s new: 3,059B LLaMA‑tokenized tokens spanning web (2,479B), code (411B), Reddit (89B), academic papers (70B), books (6B), and encyclopedias (4.3B) (Table 1).
   - Why it matters: Prior large open corpora typically lacked non‑web diversity or were lightly curated (§2). Dolma matches modern training regimes while allowing data‑centric science.

2) An open, high‑throughput curation toolkit with testable recipes (§4.1)
   - What’s new: A unified “filter then mix” API (Python + Rust) that supports fast curation (122 CPU‑hours/TB in tests), paragraph‑level dedup with a Bloom filter, and turnkey decontamination.
   - Why it matters: Makes it feasible for labs to reproduce, audit, and extend data pipelines—previously hidden behind proprietary infrastructure.

3) Evidence‑backed curation choices via controlled ablations (§4.2; Figs. 1–3)
   - Quality rules: The combination `Gopher All + C4 NoPunc` consistently outperforms either alone (e.g., on HellaSwag zero‑shot; Fig. 1).
   - Dedup at the paragraph level compounds with quality filters (Fig. 3), and filters target largely different texts (low correlations in Fig. 9), justifying the stack rather than either alone.
   - Toxicity thresholds: More aggressive filtering improves downstream accuracy but costs scale; v1 picks a conservative threshold to hit 3T tokens, while documenting the trade‑off (Fig. 2; §5.3).

4) Domain‑fit analysis that isolates the value of source diversity (§9.2; Fig. 5)
   - Finding: A 1.2B model trained 150B tokens on Dolma nearly matches the Paloma perplexity of a model trained on The Pile (diverse multi‑source) and clearly outperforms models trained on single‑source web corpora (C4, mC4, RefinedWeb) (Fig. 5).
   - Significance: Mixing curated non‑web sources improves generality beyond simply scaling web data.

5) Practical insights about code and tokenization (§M; Table 3; App. F)
   - Small fractions of code (5–15%) improve program‑aided reasoning (bAbI/WebNLG ICL; GSM8K+PAL FT; Table 3).
   - Tokenization “fertility” is >2× higher for code (2.45 vs. ~1.15–1.28 for text subsets) due to whitespace tokens, implying higher compute cost and motivation for code‑aware tokenizers (App. F; Fig. 6b–c).

## 5. Experimental Analysis
- Evaluation methodology (what is measured and how)
  - Ablations: 1.2B‑parameter OLMo variant trained to 150B tokens, controlled optimizer/schedule (App. D.1). Metrics include zero‑shot accuracy on eight classification‑style tasks (App. D.3) and perplexity on Paloma’s diverse domains (App. D.2). Decontamination removes training paragraphs ≥13 Unicode words that match Paloma (≤0.02% docs removed; App. L).
  - Full‑scale model: `OLMo‑1B` trained ~3.1T tokens on Dolma (batch×steps scaled; AdamW; App. D.4) and compared against peer 1B‑scale models (Table 2).

- Main quantitative results
  - Curation ablations (web pipeline)
    - Quality rules: `C4 NoPunc + Gopher All` yields the best zero‑shot curves; example on HellaSwag shows clear separation (Fig. 1). These two heuristics remove 22.73% and 15.23% characters, respectively (§5.2).
    - Toxicity filtering: Low threshold (removing ~29–35% sentences) outperforms High (~5.5–7.3%) but reduces scale; curves in Fig. 2 show consistent gains for Low on HellaSwag and similar patterns across tasks (App. O.6). v1 adopts High to preserve 3T tokens (§5.3).
    - Deduplication: Adding paragraph‑level dedup after URL/doc dedup and quality filtering further improves accuracy (Fig. 3). The three dedup stages remove, in sequence, 53.2% (URL), 14.9% (exact doc), and 18.7% of remaining paragraphs (§5.4).
    - Filter complementarity: Pearson correlations between what each filter removes are small in High/Medium/Low “quality” buckets (Fig. 9), indicating orthogonal signals rather than redundancy.

  - Source‑format ablations (Reddit)
    - Treating comments and submissions as independent documents (“Atomic”) beats thread‑stitching formats; see HellaSwag in Fig. 4 and matching trends across tasks (App. O.9). Reason: artificial formatting adds noise (§7.1).

  - Mixing ablations (Table 4; Fig. 12; Table 3)
    - Domain sensitivity: `Web‑Only` matches others on C4‑100 web domains but underperforms on code (HumanEval perplexity higher) and academic text (M2D2‑S2ORC) (Fig. 12).
    - Reference upsampling (`Reference+`) reduces perplexity on academic papers relative to equal mixes, while `Gopher‑like` does not improve further despite much more reference content—indicating diminishing returns (Fig. 12, right).
    - Code fraction: bAbI ICL exact match improves from 0.0 (0% code) → 8.8±0.9 (5%) → 10.1±2.8 (15%); WebNLG Rouge‑2: 16.8±1.1 → 19.3±1.1 → 22.0±1.3; GSM8K + PAL finetuning: 11.8±0.8 → 14.2±1.3 → 14.7±0.9 (Table 3).

  - Domain‑fit comparison across corpora (§9.2; Fig. 5)
    - Across training tokens (20B→150B), Dolma ≈ The Pile ≪ single‑source web corpora on Paloma perplexity (Fig. 5). This supports the claim that diverse sources yield broader coverage at fixed compute.

  - Full model benchmarking (Table 2)
    - Zero‑shot accuracy (higher is better). Averages over eight tasks:
      - `OLMo‑1B`: 60.3; `TinyLlama‑1.1B`: 59.4; `Pythia‑1.1B`: 54.5; `StableLM2‑1.6B`: 66.5 (Table 2). On individual tasks, `OLMo‑1B` leads TinyLlama on 4/8 (e.g., HellaSwag 62.5 vs. 58.7; PIQA 73.7 vs. 71.1).
    - Caveat: Peer models differ in training tokens and data composition (e.g., StableLM2 reportedly trained on 2T tokens, data unspecified). The comparison still shows Dolma’s competitiveness at the 1B scale.

- Robustness checks and safeguards
  - Test contamination analysis reveals many popular academic evaluation sets occur in public code repos; contaminated datasets (e.g., WSC, SICK, COPA) are excluded from the evaluation suite (App. L; Fig. 11).
  - Language ID audit on International Corpus of English indicates permissive English threshold (≥0.5) retains all English documents across regions (App. G; Fig. 7).
  - Toxicity classifier dialect check across location‑based subreddits shows <5% difference among regions across thresholds, with bimodal score distributions indicating threshold robustness (App. H; Fig. 8).

- Do the experiments support the claims?
  - Yes, for the stated scope. The ablations demonstrate consistent, directionally stable effects of quality rules, toxicity thresholds, and dedup at 1.2B/150B tokens, and link source diversity to domain fit. The full‑scale 1B training shows competitive downstream performance. External validity to much larger models is discussed as a limitation (§Limitations).

## 6. Limitations and Trade-offs
- Assumptions and scope bounds
  - English‑only corpus: Despite language ID filtering, a small fraction of non‑English persists; the dataset does not enable non‑English pretraining (§Limitations).
  - “Quality” and “toxicity” are operationalized via heuristics/classifiers trained on specific datasets; these encode sociotechnical assumptions (§5.2–§5.3; footnotes 4–5).
- Empirical scope
  - Ablations done at 1.2B parameters trained to 150B tokens; effects may differ at 7B–70B+ scales (§Limitations).
  - Evaluation suite focuses on decontaminated, base‑model tasks (no instruction tuning), leaving out many model capabilities (e.g., executable code generation; §Limitations).
- Data coverage and representativeness
  - Despite six source families, many materials used in practice (e.g., certain proprietary book corpora) cannot be redistributed; representativeness relative to commercial models is necessarily incomplete (§Limitations).
- Computational and operational trade‑offs
  - Toxicity threshold: stricter filtering improved accuracy but risked undercutting the 3T target; v1 uses a conservative threshold (§5.3).
  - Code tokenization “fertility” doubles token count for the same bytes, increasing compute cost unless tokenizer adaptations are made (App. F; Fig. 6).
- Legal and ethical uncertainties
  - Copyright/fair‑use doctrine for model training is unsettled and jurisdiction‑dependent (App. Ethical Considerations). PII regex covers only a few high‑precision types (emails, phones, IPs), so residual sensitive info can remain (§5.3; App. I).

## 7. Implications and Future Directions
- How this changes the landscape
  - Establishes a reproducible, documented baseline for pretraining corpora at modern scale. Researchers can now vary a single curation knob and attribute model differences to data, not to opaque pipelines.
  - Provides a platform for systematic data science of LMs: dedup strategies, toxicity/PII trade‑offs, language ID thresholds, mixture optimization, decontamination protocols.
- Enabled follow‑up research
  - Scaling laws for “data quality”: quantify marginal utility of each filter at larger model sizes; explore adaptive thresholds per domain.
  - Tokenizer‑data co‑design: reduce code fertility by adding whitespace‑prefixed tokens (App. F), or learn domain‑aware tokenizers that lower compute for code.
  - Mixture search: programmatic optimization of source weights for task portfolios, extending the `Reference+`/`Gopher‑like` explorations (Table 4; Fig. 12).
  - Safety and privacy: extend PII detection beyond regex (e.g., contextual PII models fast enough for TB‑scale), and refine toxicity filters with bias controls verified across dialects.
  - Multilingual Dolma: the toolkit generalizes to other languages; extending beyond English would enable broad comparative studies (§3, footnote 2).
- Practical applications
  - Training fully open base models with transparent data lineage.
  - Benchmark decontamination for new test suites (the Bloom‑filter recipe is turnkey; §4.1; App. L).
  - Auditing and improving existing corpora (e.g., reprocessing C4 with stricter filters reduced duplication and masked additional PII; §8).

> Key takeaway: With Dolma (Table 1) and the Dolma Toolkit (§4.1), data curation for LMs is no longer a black box. The paper shows—via concrete numbers in Figs. 1–3, 5, 12 and Tables 2–4—how specific, repeatable choices in filtering, deduplication, and mixing translate into downstream behavior.
