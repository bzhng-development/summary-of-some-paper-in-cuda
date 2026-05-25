# The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale

**ArXiv:** [2406.17557](https://arxiv.org/abs/2406.17557)

## 🎯 Pitch

FineWeb introduces a massive, fully open 15-trillion-token web-text dataset, curated through a rigorously documented and reproducible pipeline that systematically tests and optimizes each filtering and deduplication step for large language model pretraining. By also releasing FineWeb-Edu—a 1.3-trillion-token subset filtered for educational quality via LLM-driven classifiers, which dramatically boosts performance on knowledge and reasoning benchmarks—this work both democratizes access to top-tier data and sets a new transparency standard for the field, empowering the community to build, study, and advance LLMs with unprecedented resources and clarity.

---

## 1. Executive Summary
FineWeb is a fully open, 15-trillion-token web-text dataset curated from 96 Common Crawl snapshots with a documented, reproducible pipeline that measurably improves language-model pretraining quality. The paper also introduces FineWeb-Edu, a 1.3-trillion-token educational subset selected by an LLM-trained classifier, which yields large gains on knowledge and reasoning benchmarks (e.g., +4 points on MMLU and +11 on ARC with a 1.71B model; Section 4, Fig. 10–11).

## 2. Context and Motivation
- Problem addressed
  - High-quality pretraining data strongly determines LLM performance, but the best-performing open models rarely disclose their data or recipe. There is thus both a data gap (few large, high-quality, open corpora) and a knowledge gap (unclear which curation choices matter most).
- Importance
  - Practical: Training data quality drives capability, safety, and sample efficiency. Access to transparent, large-scale data and recipes lowers barriers for academic and open-source communities.
  - Scientific: Systematic ablations reveal what filtering and deduplication strategies actually improve downstream performance (Sections 3.2–3.6).
- Prior approaches and limitations
  - Public web corpora (C4, OSCAR, The Pile, RedPajama, RefinedWeb, Dolma) use different extraction, filtering, and deduplication heuristics, often with partial documentation and mixed results (Section 2). Few provide head-to-head ablations at scale tying design choices to model quality.
  - Closed recipes (e.g., MassiveText, GPT-3’s web selection pipeline) hint at strategies but are not reproducible (Section 2).
- Positioning
  - The work contributes (1) a large, open dataset (FineWeb), (2) a transparent pipeline with ablations at each step, (3) an educationally filtered subset (FineWeb-Edu) that delivers strong gains on reasoning/knowledge tasks, and (4) released code and all trained ablation models (Abstract; Sections 3–4; Appendix C).

## 3. Technical Approach
This is an empirical curation study: each pipeline choice is ablated by training small, controlled models on matched token budgets and evaluating on the same benchmarks (Section 3.1).

Key pipeline stages (Sections 3.2–3.7):
1) Text extraction from Common Crawl
   - Definitions:
     - `Common Crawl`: public monthly web crawls since 2007.
     - `WARC` files: raw web archives with HTML and metadata.
     - `WET` files: text extracted from WARC via a simple parser.
   - Choice and rationale:
     - Extract main content from WARC using `trafilatura` to reduce boilerplate/menu text (Section 3.2).
     - Fig. 1 shows that models trained on WARC+trafilatura outperform those trained on stock WET (same minimal filtering).
     - Mechanism: better text extraction reduces “unnatural language” (menus, navigation, junk), which otherwise harms pretraining fit (Section 2).

2) Base filtering (Section 3.3)
   - Steps:
     - URL blocklist to remove adult content (RefinedWeb-like).
     - Language identification with fastText; keep English with score ≥ 0.65.
     - MassiveText repetition/quality filters (original thresholds).
   - Effect:
     - Improves aggregate accuracy vs. unfiltered WARC (Fig. 2).
     - Produces ≈36T tokens after base filtering (Section 3.3).

3) Deduplication (Section 3.4; Appendix E)
   - Definitions:
     - `Deduplication`: removing repeated content across documents.
     - `n-gram`/`5-gram`: sequence of n consecutive tokens/words; here 5-grams represent documents’ shingles.
     - `MinHash`: a fuzzy hashing method that approximates Jaccard similarity between sets (here, sets of n-grams). Two documents are deemed duplicates if enough MinHash signatures match.
   - Parameters (Appendix E.1):
     - 5-grams; 112 hash functions split into 14 buckets of 8; documents match if they share 8 minhashes in any bucket.
     - This probabilistically catches ≥75% similar documents with high probability (e.g., ≈92% match probability at 0.8 similarity).
   - Design choice and mechanism:
     - The team tested “global” dedup (across all 96 snapshots) vs. “per-snapshot” dedup (each crawl dedups against itself only).
     - Global MinHash removed a huge fraction of older crawls (down to 10% retained for some) and left a dataset of ~4T tokens, but it barely improved performance and underperformed RefinedWeb (Fig. 3).
     - A diagnostic on crawl 2013-48 showed the globally “kept” 10% was lower quality than the “removed” 90% (Fig. 4), likely because cross-snapshot dedup retained templated/advertising-heavy residuals while discarding better duplicates.
     - Per-snapshot MinHash retains ~20T tokens and matches RefinedWeb performance (Fig. 5). Hypothesis: the main win comes from removing giant duplicate clusters that occur within snapshots; aggressively deduping residual small clusters across snapshots can upsample noise (Section 3.4).
     - Additional “lighter” global dedup variants (URL, line-level, 3-line-level) all degraded performance relative to per-snapshot dedup (Fig. 15; Appendix E.3).

4) Adding C4-style heuristic filters (Section 3.5)
   - C4 uses several simple rules. The team re-examined their impact on a base-filtered, per-snapshot-deduped crawl (2019-18).
   - Findings (Fig. 6):
     - The `terminal punctuation` rule (drop lines without sentence-ending punctuation) gives the biggest single boost but removes ~30% of tokens.
     - Other rules (e.g., curly bracket, word-length) yield small gains while removing 2.8–4.3% of tokens.
     - Applying “all but terminal punctuation” performs better than “terminal punctuation alone” while removing far less (≈7%).
   - Choice:
     - Apply all C4 filters tested except the terminal punctuation filter (Section 3.5).

5) Designing new, data-driven heuristic filters (Section 3.6; Fig. 8; Appendix E.4)
   - Method:
     - Compute >50 text statistics on one high-quality slice (per-snapshot dedup) and one low-quality slice (globally dedup) of the same crawl (2013-48).
     - For metrics whose distributions differed, set thresholds to preferentially remove regions overrepresented in the low-quality histogram (Fig. 8).
   - Three filters proved most effective in 28B-token ablations (Fig. 7):
     - Low punctuation density: drop docs where fraction of lines ending in punctuation ≤ 0.12 (removes 10.14% tokens vs. 30% if using C4’s line-level terminal punctuation).
     - High line duplication: drop if fraction of characters in duplicated lines ≥ 0.1 (removes 12.47% tokens; MassiveText’s original threshold was 0.2).
     - Too many very short lines: drop if fraction of lines shorter than 30 characters ≥ 0.67 (removes 3.73% tokens).
   - Combined, these remove ~22% tokens and add ≈+1% aggregate accuracy in 28B-token runs (Section 3.6; Fig. 7).

6) Final dataset and incremental gains (Section 3.7)
   - Pipeline: WARC+trafilatura → base filtering → per-snapshot MinHash → selected C4 filters → custom filters → PII anonymization (emails and public IPs).
   - Output sizes: FineWeb ≈15T tokens (Section 3.7).
   - Each step increases performance (Fig. 9).

7) Educational filtering (FineWeb-Edu; Section 4; Appendix F)
   - Goal: favor educational/step-by-step explanatory text, which recent closed models used to boost reasoning.
   - Approach:
     - Generate synthetic quality labels: Llama-3-70B-Instruct rates 460k pages on an additive 0–5 “educational value” rubric tailored to primary–middle school (prompt: Appendix F.1).
     - Train a linear regressor on `Snowflake-arctic-embed-m` embeddings to predict these scores over all FineWeb (410k train / 50k val; learning rate 3e-4; 20 epochs; best F1 on val) and classify docs as educational via a threshold (Section 4).
     - Choose threshold ≥3 by ablation (Appendix F.2, Fig. 17), achieving F1=82% on held-out annotations and the best aggregate performance trade-off.
     - Compute at scale: 
       > “Applying the classifier to the 15 trillion tokens of FineWeb required 6,000 H100 GPU hours.” (Section 4)
   - Output: FineWeb-Edu ≈1.3T tokens.

## 4. Key Insights and Innovations
- A. Per-snapshot (not global) deduplication is the sweet spot
  - What’s new:
    - Careful ablations show that global MinHash across snapshots can degrade quality by over-removing good content and upweighting templated residue in older crawls (Fig. 3–4).
  - Why it matters:
    - This overturns the common intuition that “more deduplication is always better.” It identifies a failure mode specific to multi-snapshot web corpora and provides a practical fix (per-snapshot MinHash; Fig. 5).

- B. A small set of tuned, interpretable heuristics beats heavy-handed rules
  - What’s new:
    - Three filters—punctuation-line ratio, duplicated-line character ratio, and short-line ratio—capture much of the benefit of C4-style cleaning while dropping far fewer tokens (Section 3.6; Fig. 7; Appendix E.4).
  - Why it matters:
    - These metrics are easy to compute, transparent, and reproducible. They improve quality without the 30% token loss of a full terminal-punctuation rule (Fig. 6).

- C. Educational-content filtering via LLM-generated annotations scales and pays off
  - What’s new:
    - Train a lightweight classifier on LLM-scored samples and apply it to trillions of tokens to select educational text (Section 4; Appendix F.1–F.2).
  - Why it matters:
    - The resulting FineWeb-Edu substantially boosts reasoning/knowledge performance: MMLU +4 points (≈+12% relative) and ARC +11 points (≈+24% relative) at 350B tokens (Section 4; Fig. 10–11).

- D. End-to-end transparency and tooling
  - What’s new:
    - Release of the full curation code (`datatrove`) and all ablation models; dataset under the permissive ODC-By license; PII anonymization policy (Section 3.7; Appendix C, A).
  - Why it matters:
    - Enables exact reproduction and further community-driven improvements—rare at this scale.

## 5. Experimental Analysis
- Evaluation design (Section 3.1)
  - Models: 1.71B parameters, Llama-like architecture, 2048 context, GPT-2 tokenizer; global batch ≈2M tokens.
  - Training budgets:
    - 28B tokens for many filtering ablations (near “Chinchilla-optimal” for this size; Section 3.1).
    - 350B tokens for deduplication tests and cross-dataset comparisons (Sections 3.4, 3.7; Fig. 3, 5, 9–11).
  - Replication: Two runs per condition, different seeds and data samples; results averaged to reduce variance (Section 3.1).
  - Benchmarks (capped at 1k examples for efficiency): CSQA, HellaSwag, OpenBookQA, PIQA, SIQA, WinoGrande, ARC, MMLU (Section 3.1).

- Main results and where they appear
  - Extraction matters (Section 3.2; Fig. 1):
    - WARC+trafilatura > WET. The performance curve rises higher across training tokens when using custom extraction.
  - Base filtering helps (Section 3.3; Fig. 2):
    - Adding the URL, language, and MassiveText filters yields a clear accuracy lift vs. unfiltered WARC text.
  - Global dedup is not a free win (Section 3.4; Fig. 3–4):
    - Despite reducing the dataset to ~4T tokens, global MinHash yields only a modest lift over no dedup and lags RefinedWeb across training; retained data from older snapshots looks worse than the removed portion in isolated tests (Fig. 4).
  - Per-snapshot dedup is strong (Section 3.4; Fig. 5):
    - Matches RefinedWeb in aggregate; the curve improves steadily with more data seen (up to 350B tokens).
  - C4 filters trade-off (Section 3.5; Fig. 6):
    - Terminal punctuation alone gives the biggest single gain but removes ~30% of tokens; “all but terminal punctuation” performs best overall while removing ~7%.
  - Custom heuristic filters (Section 3.6; Fig. 7):
    - Three compact rules together remove ~22% of tokens and add ≈+1% aggregate in 28B-token runs, surpassing C4 while being more data-preserving.
  - Final FineWeb stepwise gains (Section 3.7; Fig. 9):
    - Each stage—per-snapshot dedup → C4 filters → custom filters—adds incremental improvements over the base-filtered baseline.
  - Cross-dataset comparisons (Section 3.7; Fig. 10):
    - At 350B training tokens with a 1.71B model, FineWeb is competitive among public web datasets; FineWeb-Edu surpasses all listed alternatives on the paper’s aggregate metric by ≈2 percentage points.
  - Educational gains (Section 4; Fig. 10–11):
    - With the same model/budget, MMLU rises from ~33% to ~37% and ARC from ~46% to ~57%.
    - Fig. 11 emphasizes sample efficiency: FineWeb-Edu reaches ≈33.6% MMLU at just 38B tokens, whereas Matrix needs ~300B tokens to match that.
  - Domain fit (Section 4.2; Fig. 12; Appendix F.4):
    - FineWeb has lower perplexity on broad web-like sources (C4, mC4, Dolma v1.5, social media subsets).
    - FineWeb-Edu has lower perplexity on Wikipedia, arXiv, and programming (100 PLs) subsets.

- Ablations, robustness, and failure analyses
  - Extensive ablations at each pipeline stage (Sections 3.2–3.6), including negative results (e.g., URL/line-level dedup variants underperform; Fig. 15).
  - Threshold sweeps for FineWeb-Edu show score ≥3 is the best global trade-off (Appendix F.2, Fig. 17).
  - Bias analysis (Section 5; Appendix G):
    - Distributional skews (e.g., “man” and “christian” overrepresented; Fig. 19–21).
    - Association skews (e.g., several religion terms co-occur with online dating vocabulary; Fig. 23–28). FineWeb-Edu shifts associations toward history/health (Section 5).

- Do the experiments support the claims?
  - Yes, within scope. The pipeline’s contributions are validated by controlled ablations with replicated runs and consistent evaluation. Large gains for educational filtering are shown across multiple reasoning benchmarks and training budgets (Fig. 10–11).
  - Caveat: All training uses 1.71B models; generalization to much larger models is plausible but not demonstrated (Conclusion).

## 6. Limitations and Trade-offs
- Assumptions and scope
  - English-only corpus (Dataset Datasheet; Appendix A). Multilingual performance is out of scope.
  - The quality metric is benchmark-centric and pretraining-only; no instruction tuning or RLHF is included in the reported evaluations (Conclusion).
- Data composition and coverage
  - Exclusively Common Crawl web pages; other modalities and curated sources (e.g., books) are absent. The paper notes that augmenting with non-web data might improve results (Conclusion).
  - Heuristic filters likely reduce code prevalence (Dataset Datasheet, “Other Known Limitations”).
- Dedup design
  - Per-snapshot MinHash may miss near-duplicates spanning snapshots, and the chosen parameters (112 hashes, 5-grams) trade precision/compute (Appendix E.1). There is unavoidably some uncertainty around documents near the similarity threshold (Appendix E.1, Fig. 13).
- Educational filtering
  - The ground truth is produced by an LLM (Llama 3 70B), so classifier targets inherit its biases and rubric. The threshold choice (≥3) optimizes the paper’s evaluation mix; alternative use cases might favor different thresholds (Section 4; Appendix F.2).
  - Compute cost is nontrivial:
    > “Applying the classifier … required 6,000 H100 GPU hours.” (Section 4)
- External validity and scale
  - Most ablations are run at 28B or 350B training tokens on 1.71B models; extrapolation to much larger models or longer training remains an open question (Conclusion).
- Bias and safety
  - Despite URL filtering and removal of some PII (emails, public IPs), web-derived datasets retain toxic and biased content (Dataset Datasheet; Section 5, Appendix G). FineWeb shows overrepresentation of hegemonic terms; FineWeb-Edu mitigates some association patterns but is not a comprehensive safety solution.

## 7. Implications and Future Directions
- Field impact
  - Establishes an open, large-scale baseline: A 15T-token corpus with a documented, empirically validated curation recipe. This reduces reliance on proprietary data mixtures and enables reproducible, apples-to-apples data ablations.
  - Clarifies best practices:
    - Prefer per-snapshot deduplication over global dedup for multi-snapshot web corpora.
    - Use a small set of tuned, interpretable filters rather than blunt, high-drop rules.
    - Educational filtering via LLM-labeled classifiers is an effective and scalable lever for reasoning performance.
- Follow-up research
  - Data mixture design:
    - Combine FineWeb with books, Wikipedia dumps, code, and transcripts to test complementary gains (Conclusion).
    - Explore multilingual variants and domain-targeted subsets beyond education (e.g., programming, law, medicine).
  - Dedup and filtering:
    - Develop hybrid dedup approaches that remove cross-snapshot giant clusters while preserving diversity within small clusters.
    - Replace/augment heuristics with learned quality signals (beyond education), and examine causal links between metrics and downstream skills.
  - Scaling studies:
    - Re-run ablations at larger model sizes and longer training to test persistence or reversal of observed effects; integrate instruction-tuning/feedback alignment to reflect modern pipelines.
  - Safety and fairness:
    - Expand PII anonymization beyond regex; incorporate systematic toxicity filters and post-hoc decontamination. Extend the bias analyses (Appendix G) to interventions that meaningfully reduce harmful associations.
- Practical applications
  - Pretraining data for open LLMs up to hundreds of billions of parameters (“sufficiently large to train a Chinchilla-optimal >500B model”; Abstract).
  - Bootstrapping domain-specific corpora via LLM-labeled classifiers (the FineWeb-Edu recipe). Teams can build “FineWeb-<Domain>” slices by swapping the labeling rubric and embedding classifier.
  - Benchmarking data-curation choices: the released ablation models and `datatrove` codebase provide a testbed for the community to iterate on filters, dedup, and selection at scale.

Overall, the paper’s main contributions are empirical and practical: a transparent large-scale dataset with a vetted curation recipe, a clear negative result that changes how to deduplicate multi-snapshot web crawls (Fig. 3–5), and a scalable, effective educational filtering method that yields strong reasoning gains (Section 4; Fig. 10–11).
