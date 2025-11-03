# The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale

**ArXiv:** [2406.17557](https://arxiv.org/abs/2406.17557)
**Authors:** Guilherme Penedo, Hynek Kydlíček, Loubna Ben Allal, Anton Lozhkov, Margaret Mitchell, Colin Raffel, Leandro von Werra, Thomas Wolf
**Institutions:** 

## 🎯 Pitch

FineWeb revolutionizes large-scale pretraining by unveiling a transparent method to craft high-quality datasets from the public web, demonstrated with its 15 trillion token corpus and educationally-focused FineWeb-Edu subset. By meticulously filtering and deduplicating web data, FineWeb not only democratizes access to powerful natural language models but also sets a new standard for knowledge-intensive tasks, making open-source LLMs more equitable and robust.

---

## 1. Executive Summary
FineWeb is a fully documented, reproducible recipe for building large-scale pretraining corpora from the public web, plus two released datasets: FineWeb (15 trillion tokens from 96 Common Crawl snapshots) and FineWeb-Edu (a 1.3-trillion–token educational subset). The work shows, through controlled ablations, which filtering and deduplication choices actually improve downstream model performance and demonstrates that targeting educational content yields large gains on knowledge- and reasoning-heavy benchmarks (e.g., MMLU and ARC).

## 2. Context and Motivation
- Problem addressed
  - Pretraining datasets for strong open LLMs remain largely undocumented or closed, despite being decisive for model quality and safety. This creates a knowledge and capability gap for the broader community.
  - Even when the web is the source, “how” to turn raw web crawls into effective pretraining data—especially filtering and deduplication—has many choices that materially affect results (Section 2).

- Why it matters
  - Data governs what LLMs learn. Filters that remove “unnatural” text and deduplication that reduce copy-paste artifacts improve generalization and reduce memorization, privacy risk, and toxicity (Sections 2–3.4; [5], [53], [54]).
  - Publicly available, high-quality, large-scale datasets promote reproducibility and democratize LLM development.

- Prior approaches and limitations
  - Popular web-based datasets (e.g., C4, RefinedWeb, RedPajama, Dolma) use different extraction, filtering, and deduplication strategies; some are composite or partially labeled rather than fully curated (Section 2).
  - Known issues:
    - WET (Common Crawl’s default text files) retain boilerplate/menu text (Section 3.2).
    - Heavy-handed heuristics like C4’s “terminal punctuation” filter can remove too much useful data (Section 3.5; Fig. 6).
    - “Global” dedup across snapshots may remove the wrong data and degrade performance (Section 3.4; Figs. 3–5).

- Positioning
  - FineWeb contributes:
    - A transparent, step-by-step recipe built via empirical ablations, not assertions (Sections 3.1–3.7).
    - A new educationally filtered subset (FineWeb-Edu) that outperforms other public web-only datasets on reasoning/knowledge benchmarks (Section 4; Fig. 10).
    - Open release of datasets, code, and all ablation models for reproducibility (Abstract; Section 6; Appendix C).

## 3. Technical Approach
The paper builds FineWeb via a pipeline of extraction, filtering, and deduplication, then derives FineWeb-Edu using an educational-content classifier. Each decision is backed by controlled ablations.

1) Data source and text extraction (Section 3.2)
- Source: 96 Common Crawl snapshots (2013–2024).
- Formats:
  - `WARC` (raw HTML + metadata).
  - `WET` (Common Crawl’s default text-only extraction).
- Choice: extract text directly from `WARC` using `trafilatura` to remove boilerplate/navigation better than `WET`.
- Evidence:
  - On 28B tokens, models trained on `trafilatura`-extracted `WARC` outperform those trained on `WET` (Fig. 1; aggregate accuracy improves across training).

2) Base filtering (Section 3.3)
- Components:
  - URL blocklist for adult content (UT1).
  - Language filtering with `fastText` (keep English with score ≥ 0.65).
  - Quality/repetition heuristics from MassiveText (e.g., character repetition).
- Result:
  - Produces ~36T tokens from the 96 snapshots.
  - Improves performance versus unfiltered `WARC` text (Fig. 2).

3) Deduplication (Section 3.4; Appendix E)
- Why dedup?
  - The web has mirrors, aggregators, and templates that replicate content; dedup improves performance and reduces memorization risk (Section 2; [5], [53]).
- What is MinHash?
  - A fuzzy deduplication technique: it hashes fixed-length token sequences (`n`-grams) with many hash functions and considers two docs duplicates if they share enough identical hash “signatures.”
- Design chosen:
  - Represent each document by word 5-grams.
  - Compute 112 MinHashes, partitioned into 14 buckets of 8; consider docs duplicates if any bucket matches fully (8/8). This targets ≥75% similarity with high probability (Appendix E.1; Fig. 13).
  - Transitive clustering: if A ~ C and B ~ C, group A, B, C and keep one.
- Critical finding: global vs per-snapshot dedup
  - Global MinHash deduplication across all 96 snapshots leaves ~4T tokens but yields only modest gains and underperforms RefinedWeb on the aggregate (Fig. 3).
  - Diagnostic experiment on snapshot 2013-48:
    - The small fraction of data “kept” after global dedup is actually lower quality than the “removed” fraction (Fig. 4; visual inspection confirms more ads/keyword lists in kept data).
  - Final choice: deduplicate independently within each snapshot; yields ~20T tokens and matches RefinedWeb performance (Fig. 5).
  - Why not “lighter” global methods?
    - URL, line-level, and 3-line dedup variants further degraded performance (Appendix E.3; Fig. 15), despite removing 71.5–85% tokens (e.g., URL dedup leaves 5.6T; line-level leaves 4.4T).

  - Why small-scale tests can miss dedup effects:
    - Simulation (Appendix E.2; Fig. 14) shows that at small sample sizes, few duplicates appear even if the full corpus has many; hence dedup gains only show up at larger training scales (their dedup ablations use 350B tokens).

4) Additional heuristic filters: C4-inspired and FineWeb-specific (Sections 3.5–3.6)
- C4 filters revisited (Fig. 6):
  - Tested individually and in combination on snapshot 2019-18:
    - “Terminal punctuation” boosts HellaSwag most but removes ~30% tokens alone.
    - “Curly bracket” (~2.8% removed) and “word length” (~4.3% removed) give smaller gains.
    - Minor rules (e.g., “lorem ipsum,” “javascript,” “policy”) each remove <0.5%.
  - Decision: adopt all C4 filters except terminal punctuation, resulting in better HellaSwag gains with ~7% total removal (“All but terminal punct” beats the terminal-punct-only variant).

- FineWeb’s own filters (method in Section 3.6; details in Fig. 8 and Appendix E.4):
  - How thresholds were chosen:
    - Compute >50 document-level and repetition metrics.
    - Compare distributions between a “higher quality” set (individually deduped 2013-48) and a “lower quality” set (globally deduped 2013-48).
    - Pick cut points where the low-quality distribution is overrepresented (e.g., the fraction of lines ending with punctuation; Fig. 8).
  - Three retained filters (best ablation gains; Fig. 7):
    - Fraction of lines ending with punctuation ≤ 0.12 (removes 10.14% tokens).
    - Fraction of characters in duplicated lines ≥ 0.1 (12.47% removed; stricter than MassiveText’s 0.2).
    - Fraction of lines shorter than 30 characters ≥ 0.67 (3.73% removed).
  - Together, they remove ~22% and add ~1% aggregate accuracy at 28B tokens.

5) Final FineWeb dataset (Section 3.7)
- Pipeline: `WARC` extraction → base filters → per-snapshot MinHash dedup → selected C4 filters (no terminal-punct) → custom filters → PII anonymization (emails and public IPs).
- Size: 15T tokens (enough for Chinchilla-optimal training of 500B+ parameter models; Abstract; Section 3.1 note).
- Stepwise gains: each step raises the aggregate score (Fig. 9).

6) FineWeb-Edu: educational filtering at scale (Section 4; Appendix F)
- Idea: identify “educational” pages to favor knowledge/reasoning skills (inspired by recent closed models).
- How labels were created:
  - Use `Llama-3-70B-Instruct` to score 460k sampled pages (CC-MAIN-2024-10) from 0–5 using an additive rubric oriented to primary/middle-school content (Appendix F.1).
- How the classifier was built:
  - Train a linear regressor on top of `snowflake-arctic-embed-m` embeddings using 410k labeled examples (20 epochs, LR=3e-4), tuning to predict the 0–5 score; select checkpoint with highest F1 on a 50k held-out set (F1=82% after rounding to 0–5; Section 4).
  - Classify the full FineWeb corpus and keep docs with score ≥ 3 (best trade-off across ablations; Appendix F.2; Fig. 17).
- Cost: ~6,000 H100 GPU hours to run the scoring across 15T tokens.
- Result: FineWeb-Edu with 1.3T tokens.

7) Experimental setup for ablations (Section 3.1; Appendix D)
- Models: 1.71B parameters, Llama architecture, sequence length 2048, GPT-2 tokenizer; 2 runs per dataset variant to average out sampling/seed variance.
- Data scale: 28B tokens for most filter ablations; 350B tokens for dedup and end-to-end validations.
- Training: ~80,000 H100 GPU hours across >70 models; global batch ~2M tokens.
- Evaluation: lighteval on the following multiple-choice/QA benchmarks, truncated to 1k samples each for efficiency: CommonSenseQA, HellaSwag, OpenBookQA, PIQA, SIQA, WinoGrande, ARC, and MMLU (Sections 3.1 and 3).

## 4. Key Insights and Innovations
- A fully empirical, reproducible curation recipe (incremental but essential)
  - Instead of opaque “magic filters,” the paper performs step-by-step ablations and releases code/models (Sections 3.1–3.7; Fig. 9). This enables the community to see which curation choices matter and why.

- Counterintuitive finding: global deduplication can hurt (fundamental insight)
  - Deduplicating “across all snapshots” aggressively removes older data that turns out to be better than what remains; performance lags behind an individually deduped strategy (Section 3.4; Figs. 3–5).
  - The 2013-48 case study (Fig. 4) is particularly compelling: data “kept” by global dedup performs worse than the “removed” data; visual inspection shows the kept portion is lower quality.

- Systematic heuristic design (incremental but impactful)
  - The paper introduces a general method for creating filters: compute metric histograms on high- vs low-quality pools, pick thresholds where low-quality is overrepresented, and validate by ablation (Section 3.6; Fig. 8; Appendix E.4). The resulting three filters outperform simply adopting C4’s heavy “terminal punctuation” rule while discarding far less data.

- Educational filtering for reasoning (new capability for open datasets)
  - FineWeb-Edu, created with an LLM-labeled classifier focused on school-level content, substantially boosts performance on knowledge-reasoning tasks (Section 4; Figs. 10–11). This extends the idea behind closed datasets (Llama 3, Phi-3) into an open, documented pipeline.

## 5. Experimental Analysis
- Evaluation methodology
  - Benchmarks: CSQA, HellaSwag, OpenBookQA, PIQA, SIQA, WinoGrande, ARC, MMLU (Section 3.1).
  - Aggregation: overall accuracy averaged across tasks; plots show accuracy vs training tokens.
  - Robustness: two seeds per ablation; truncate tasks to 1k for frequent evaluation.

- Key quantitative results
  - Extraction matters
    - `WARC`+`trafilatura` beats `WET` across training on the 28B token ablation (Fig. 1).
  - Base filters help
    - Adding base filters on top of WARC extraction improves aggregate accuracy at 28B tokens (Fig. 2).
  - Dedup: per-snapshot wins
    - Global MinHash: leaves 4T tokens but only modestly improves over no dedup and lags RefinedWeb (Fig. 3).
    - Per-snapshot MinHash: 20T tokens and matches RefinedWeb (Fig. 5).
    - The “kept vs removed” test on 2013-48 shows the global approach keeps lower-quality data (Fig. 4).
    - Alternative lighter global methods (URL/line/3-line) remove 71.5–85% of tokens and reduce performance (Appendix E.3; Fig. 15).
  - C4 filters: use them selectively
    - On HellaSwag, “terminal punctuation” gives the single biggest boost but drops ~30% tokens; “all but terminal punctuation” performs better while dropping ~7% (Fig. 6).
  - FineWeb custom filters
    - Three targeted filters remove ~22% combined and add ~1% aggregate accuracy at 28B tokens (Fig. 7; Section 3.6; Appendix E.4 gives per-filter removals).
  - End-to-end gains
    - Each pipeline step raises aggregate performance; the final FineWeb model surpasses its base-filtered and per-snapshot-dedup predecessors (Fig. 9).
  - Comparison with public datasets (350B-token training)
    - FineWeb is competitive-to-strong; FineWeb-Edu is best on the aggregate across compared open corpora (C4, RefinedWeb, The Pile, SlimPajama, RedPajama2, Dolma 1.6/1.7, Matrix, CC-100, OSCAR), adding ~2% aggregate accuracy over the next best (Fig. 10).
  - Educational gains
    - FineWeb-Edu vs FineWeb at 350B tokens:
      - MMLU: 33% → 37% (+ ~12% relative; Section 4).
      - ARC: 46% → 57% (+ ~24% relative; Section 4).
    - Token efficiency on MMLU: FineWeb-Edu reaches 33.6% with only 38B tokens, roughly what Matrix needs ~300B tokens to match (Fig. 11).
  - Domain fit (Paloma; Fig. 12; Appendix F.4 sample numbers)
    - FineWeb has lower perplexity (better fit) on broad web sources and social media (e.g., C4, mC4, Falcon, Dolma CommonCrawl; Twitter AAE; Gab; 4chan).
    - FineWeb-Edu fits Wikipedia, academic, and programming-heavy sources better (e.g., RedPajama arXiv: 23.4 vs 32.3; RedPajama GitHub: 5.25 vs 5.61; M2D2 Math/Logic categories show advantages; Table excerpt in Appendix F.4).

- Do the experiments support the claims?
  - Yes. The ablation-by-stage design (Figs. 1–9) shows causal contributions of each step. The educational classifier’s effect is large and robust across multiple benchmarks (Figs. 10–11; Appendix F.2).

- Notable robustness/diagnostics
  - Deduplication effect vs sample size is carefully reasoned (Appendix E.2; Fig. 14).
  - Alternative dedup strategies were tried and rejected on evidence (Appendix E.3; Fig. 15).
  - Filter selection used metric distribution analysis, not ad-hoc intuition (Section 3.6; Fig. 8).

- Bias analysis (Section 5; Appendix G)
  - Distributional skews in FineWeb: words like “man,” “christian” overrepresented; associations between religion terms and dating-related words (Appendix G, Figs. 23–28).
  - FineWeb-Edu shifts associations toward history/health education (e.g., “man”→“king,” “woman”→“pregnancy”) and away from intimacy-related associations.

## 6. Limitations and Trade-offs
- Scope limitations
  - English-only and web-only (Common Crawl). This underrepresents code, books, and curated sources (Appendix A “Other Known Limitations”; Section 6).
  - PII anonymization covers emails and public IPs; other sensitive data may remain (Section 3.7; Appendix A).
  - No instruction-tuning or alignment in evaluation; benchmarks are academic multiple-choice (Conclusion).

- Methodological assumptions and risks
  - Educational classifier “ground truth” comes from LLM-generated labels (Llama-3-70B-Instruct), then a linear regressor on embeddings. This risks baking in the labeling model’s biases and rubric limitations (Section 4; Appendix F.1).
  - The additive rubric focuses on primary/middle-school suitability, potentially down-weighting advanced or specialized content (Section 4).
  - Dedup strategy assumes that removing large duplicate clusters helps most and that per-snapshot dedup is the right granularity; different mixtures or future crawls might change the balance (Section 3.4).

- Compute and scale constraints
  - Most ablations use a 1.71B model, 28B or 350B tokens; results might shift at larger scales or with different architectures (Section 3.1; Conclusion).
  - Educational scoring across 15T tokens cost ~6,000 H100 GPU hours (Section 4); reproducing that step is non-trivial.

- Trade-offs observed
  - Some filters (e.g., C4 terminal punctuation) yield higher single-benchmark gains (HellaSwag) but remove too much data (~30%), hurting overall data diversity (Section 3.5; Fig. 6).
  - FineWeb-Edu improves knowledge/reasoning tasks but can reduce fit to broad web domains (higher perplexity on CommonCrawl-like sources; Fig. 12; Table in Appendix F.4).

## 7. Implications and Future Directions
- How this changes the landscape
  - It provides the community with a high-quality, very large, fully documented web-only dataset and a proven recipe to reproduce or adapt it (Appendix C for resources).
  - The work normalizes rigorous, open, data-centric ablation as part of LLM pretraining research.

- Follow-up research enabled or suggested
  - Extend the recipe beyond the web: integrate books, code, speech transcripts, and Wikipedia with similarly rigorous ablations (Conclusion).
  - Improve dedup granularity: hybrid strategies that remove very large clusters globally but preserve rare content in older snapshots; learning-based dedup beyond MinHash.
  - Better educational/rubric models:
    - Human-in-the-loop or multi-rubric labeling to mitigate LLM annotator biases.
    - Different educational “profiles” (STEM-focused, language-focused, grade-specific) to produce task-targeted corpora.
  - Safety and fairness:
    - Expand PII anonymization and unsafe-content filters; quantify memorization reduction (building on [53], [54]).
    - Systematic decontamination studies for evaluation leaks.

- Practical applications
  - Training strong open models without private corpora; reproducing “education-filtering” benefits in public settings.
  - Domain-targeted data mixtures: combine FineWeb and FineWeb-Edu with code corpora or domain documents to build task-specialized models.
  - Benchmarking and data governance: use FineWeb as a reference dataset for replicable pretraining/evaluation pipelines.

> Most consequential takeaways are empirically grounded: use `WARC` extraction (Fig. 1), avoid heavy global dedup across snapshots (Figs. 3–5), adopt targeted C4-like filters while avoiding overly aggressive rules (Fig. 6), and consider educational filtering when aiming for knowledge/reasoning gains (Figs. 10–11).
