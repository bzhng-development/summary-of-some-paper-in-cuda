# RedPajama: an Open Dataset for Training Large Language Models

**ArXiv:** [2411.12372](https://arxiv.org/abs/2411.12372)

## 🎯 Pitch

RedPajama launches the first truly open, large-scale pretraining data ecosystem for large language models, addressing critical gaps in transparency, scale, and customizability. By not only replicating LLaMA-1's training set (RedPajama-V1) but also releasing RedPajama-V2—a massive, richly annotated web corpus exceeding 100 trillion tokens—the project empowers researchers and practitioners to audit, filter, and tailor training data for safer, higher-performing, and more adaptable AI. This landmark resource accelerates open-source LLM development and enables principled advances in data-driven model optimization.

---

## 1. Executive Summary (2–3 sentences)
This paper releases RedPajama, a fully open ecosystem of pretraining data for large language models (LLMs) that addresses transparency, scale, and curation. It has two pillars: RedPajama‑V1, a best‑effort open reproduction of the LLaMA‑1 training corpus with accompanying 3B/7B models; and RedPajama‑V2, a massive web‑only corpus (>100T tokens) where every document comes with machine‑readable “quality signals” to enable principled filtering. The work shows—via controlled ablations—that these signals can be turned into better-performing training subsets without re‑scraping the web.

## 2. Context and Motivation
- Problem the paper addresses
  - Modern high-performing LLMs depend on massive, high-quality pretraining data, but most state-of-the-art models report little about how that data is composed or filtered. The field lacks:
    - Transparency in data provenance and filtering choices.
    - Access to data at the scale used by SOTA models.
    - Reusable artifacts (metadata, heuristics, and code) to enable reproducible ablations on data composition.
  - See Introduction and the three “design principles” (Transparency, Scale, Versatility) and contributions C1–C4 on p. 2.

- Why this matters
  - Real-world impact: Without transparent, large-scale data, open models lag behind and practitioners cannot audit or adapt data for safety, performance, or domain focus.
  - Scientific significance: Studying data composition calls for controlled experiments over very large data pools; these are infeasible without accessible, well-instrumented corpora.

- Prior approaches and gaps
  - Web-only datasets such as `C4` [46], `RefinedWeb` [44], and `FineWeb` [43] show that carefully filtered Common Crawl can yield strong models, but they typically publish a single filtered end product, not the raw underlying data plus reusable quality annotations.
  - Composite datasets (e.g., `The Pile` [17], `ROOTS` [26,27], `Dolma` [52]) assemble multiple domains, but the filtering logic and per-document diagnostics are not universally standardized for downstream reuse at scale.
  - The “Gopher rules” [45] and other heuristics are influential, but researchers still need consistent document-level signals to rapidly explore variants without reprocessing the web.

- Positioning of this work
  - RedPajama‑V1: an open, best‑effort reproduction of the LLaMA‑1 corpus, plus trained models to assess fidelity (Section 3; Table 2).
  - RedPajama‑V2: a web-only dataset across five languages with 46 per-document quality signals spanning natural-language heuristics, repetition measures, content-based toxicity flags, ML-based domain similarity scores, and deduplication artifacts (Section 4.1.2; Tables 11–15). This enables users to rebuild their own C4/RefinedWeb/FineWeb-style filters from a single source.
  - The paper provides controlled ablations with 468M and 1.6B-parameter LMs to demonstrate how different filter recipes change downstream performance (Section 4.3; Tables 5–6, 18–23).

## 3. Technical Approach
At a high level, RedPajama has two complementary parts:

- RedPajama‑V1 (composite, LLaMA‑style reproduction; Section 3)
- RedPajama‑V2 (web-only, raw + quality signals; Section 4)

Key notions used below (defined briefly on first use):
- `Common Crawl`: a public, regular crawl of the web. The paper uses its text-extracted `.wet` files.
- `CCNet pipeline` [61]: a light-weight cleaning process for Common Crawl that deduplicates within a snapshot and computes document perplexity under a Wikipedia-trained 5‑gram model, assigning each doc to `head/middle/tail` buckets (low/medium/high perplexity with respect to Wikipedia; lower perplexity roughly indicates more Wikipedia-like text).
- `Gopher rules` [45]: a set of practical filters for web data (e.g., natural-language checks, repetition limits, simple toxicity signals).
- `MinHash`/`LSH`: fast techniques to detect near-duplicate documents via approximate Jaccard similarity.
- `Bloom filter`: a compact probabilistic data structure to flag exact duplicates with controlled false positives.
- `fastText classifier`: a light bag-of-ngrams text classifier, used here for domain similarity scores.
- `DSIR` [62]: “Data Selection via Importance Resampling,” where a log-likelihood ratio under two n-gram LMs gives an “importance weight” of a document to a target domain (e.g., Wikipedia).

A. RedPajama‑V1: Reproducing the LLaMA‑1 training data (Section 3)
1) Data composition (Table 2; Section 3.1)
   - Goal: recreate the seven-dataset recipe described for LLaMA‑1: English `CommonCrawl`, `C4`, `GitHub`, `Books` (PG19 + Books3), `ArXiv`, `Wikipedia` (20 languages), and `StackExchange`.
   - Key processing steps (with explicit choices where the original recipe was ambiguous; see Table 10):
     - Common Crawl: select five snapshots (2019‑30, 2020‑05, 2021‑04, 2022‑05, 2023‑06), process via CCNet, keep `head+middle`, train a `fastText` “Wikipedia reference” classifier to score pages mentioned by Wikipedia and filter CC docs below score 0.25 (Section 3.1).
     - C4: use `c4_en` from Hugging Face (footnote 2).
     - GitHub: restrict to MIT/BSD/Apache, then filter low-quality files by line length, alphanumeric proportion, and whitelisted extensions (Appendix C.1).
     - Wikipedia: use dumps from 2023‑03‑20 with boilerplate removed.
     - Books: keep `PG19`; de-duplicate via `SimHash`; initially included `Books3` but later removed due to copyright (Section 3.1).
     - ArXiv: use LaTeX sources from the public S3 bucket; strip preamble/comments/bibliography (per [29]).
     - StackExchange: keep 28 biggest sites, strip HTML, order answers by score, pair Q–A (Section 3.1).
   - Scale: ~1.2T tokens (Table 2).

2) Baseline models trained on V1 (“RedPajama‑INCITE”; Sections 3.2–3.2.2)
   - Compute platform: Summit supercomputer (4608 nodes; 6×V100 GPUs per node; IBM Power9 CPUs), requiring custom builds of PyTorch (Section 3.2.1).
   - Precision constraints: V100s lack `bf16`, so training used `fp16` + loss scaling, with lower learning rates than LLaMA (1.6e‑4 for 3B; 1.2e‑4 for 7B) for stability (Section 3.2.1).
   - Parallelism: pipeline (12‑way for 7B; 6‑way for 3B) + 2‑way tensor parallelism; 4M-token global batch (Section 3.2.1).
   - Training length: 3B for 800B tokens; 7B for ~1.0T tokens (Section 3.2.1).

B. RedPajama‑V2: A web‑only dataset with quality signals (Section 4)
1) Data acquisition (Section 4.1.1)
   - Source: `.wet` text files from 84 Common Crawl snapshots spanning 2014–Apr 2023.
   - Processing: CCNet; keep all `head/middle/tail` buckets; support five languages (English, German, French, Spanish, Italian).
   - Outcome: >100 billion documents; >100 trillion tokens (Section 4.2; Table 3).

2) Quality signals (Section 4.1.2; Tables 11–15; Figures 4–7)
   - Why: raw web text contains boilerplate, code fragments, high repetition, spam, and offensive content; quality signals let users filter flexibly for their target use case without rebuilding the corpus.
   - Categories and examples (with names used in released metadata):
     - Natural language heuristics (Table 12): e.g., fraction of all‑caps words (`rps_doc_frac_all_caps_words`), stop‑word fraction, line ends with punctuation, “lorem ipsum” ratio, unigram entropy, average word length, numerical character fraction, etc.
     - Repetition measures (Table 14): e.g., fraction of characters in the top 2/3/4‑grams (`rps_doc_frac_chars_top_{2,3,4}gram`), fraction in duplicated 5–10‑grams (`rps_doc_frac_chars_dupe_{5..10}grams`).
     - Content-based filters (Table 15): number of matches to the LDNOOBW profanity list (`rps_doc_ldnoobw_words`), domain blacklists (UT1 categorization, `rps_doc_ut1_blacklist`).
     - ML heuristics (Table 13): fastText scores for Wikipedia references (`rps_doc_ml_wikiref_score`) and “Palm-mix” (English Wikipedia + OpenWebText + RedPajama‑V1 books), plus DSIR importance weights vs. Wikipedia/OpenWebText/Books (`rps_doc_{wikipedia,openwebtext,books}_importance`).
     - Deduplication artifacts: exact duplicates via a Bloom filter (1% fp rate) and fuzzy duplicates via MinHash signatures at several Jaccard similarity levels; de‑dup runs sequentially in reverse time (newest to oldest) so earlier crawls lose more duplicates (Section 4.1.2; Figure 3).

3) Dataset structure and access (Appendix B)
   - Documents: JSONL with fields such as `url`, `raw_content`, `source_domain`, `language`, `perplexity`, and `bucket` (Appendix B.1.2).
   - Quality signals: JSONL per shard, storing each signal as a list of span‑scored tuples `[start, end, score]` aligned to `raw_content`. This shared schema (used also by Dolma [52]) supports both document‑level and line‑level signals (Appendix B.1.2).
   - Dedup/minhash: parquet files with duplicate ids and MinHash signatures.
   - File layout: predictable path patterns by snapshot, shard, language, and bucket (Appendix B.1.2).
   - Distribution: Hugging Face Hub and public HTTPS endpoints (Appendix B).

C. Controlled ablations using the signals (Section 4.3)
- Models (Section 4.3.1): LLaMA‑2 style decoder‑only models with 24 layers, 16 heads, MLP ratio 4, sequence length 2048.
  - 468M params (hidden 1024) trained on 100B tokens.
  - 1.6B params (hidden 2048) trained on 350B tokens.
- Training stack: OLMo framework with FSDP; up to five H100 nodes (Section 4.3.1).
- Data subsets tested: e.g., a single 2023‑14 crawl, nine crawls from 2021‑49 to 2023‑14, and a ~1T‑token filtered sample for 1.6B models, with dedup via MinHash LSH (128 hashes, 9 bands, 13 rows) (Section 4.3.1).
- Filter recipes: combinations of exact/fuzzy dedup; C4 line filters; “full” vs. “natlang‑only” vs. “repetition‑only” Gopher rules; ML‑based selection via fastText and DSIR; and custom composites (Table 5, Table 18–20).

## 4. Key Insights and Innovations
- A. Turning web data into a reusable “filtering laboratory” (fundamental)
  - Instead of releasing one “cleaned” dataset, RedPajama‑V2 exposes the raw web text plus 46 per‑document quality signals across five languages (Section 4.1.2; Tables 11–15). This design lets anyone instantiate diverse filtering regimes (C4/RefinedWeb/FineWeb‑like and beyond) without re‑scraping or recomputing heavy heuristics. Significant because it lowers the barrier to systematic data ablations at scale and supports rapid iteration.

- B. Unifying multiple filtering paradigms in one schema (incremental but impactful)
  - The signals cover line/document heuristics (C4, Gopher, Pretrainer’s Guide), ML filters (fastText domain similarity, DSIR), toxicity lists, and dedup metadata (Section 4.1.2). This consolidation bridges previously siloed practices and enables apples‑to‑apples comparisons (Table 5).

- C. Transparent LLaMA‑1 corpus reproduction plus compute report (incremental replication with insights)
  - RedPajama‑V1 documents the concrete choices and unavoidable uncertainties in reproducing LLaMA‑1 (Section 3.1; Table 10). Training on Summit under fp16 clarifies practical constraints (Section 3.2.1) and helps explain performance gaps at 7B (Section 3.2.2).

- D. Evidence that “quality signals + dedup” generalize across tasks (empirical)
  - In ablations with 468M and 1.6B models, combining fuzzy deduplication with “full” Gopher rules frequently yields the strongest or near‑strongest aggregate benchmark performance across many tasks, competitive with well‑known curated web datasets (Section 4.3.2; Tables 5–6, 18–23). This supports the claim that principled filtering of a very large, raw web pool can match curated alternatives.

## 5. Experimental Analysis
A. Evaluation methodology
- Models and training
  - 468M model on 100B tokens; 1.6B model on 350B tokens; AdamW; cosine LR schedule; 1% warmup (Section 4.3.1).
  - Parallelization via FSDP; up to five H100 nodes (Section 4.3.1).

- Datasets compared
  - RedPajama‑V2 variants (different filter recipes and crawl subsets).
  - Established baselines: `C4`, `RefinedWeb`, `FineWeb`, `Dolma v1.7 (CC)`, and `RedPajama‑V1` (Tables 5, 18–20).

- Benchmarks and metrics (Table 4)
  - A curated set chosen to be informative at small scale, covering:
    - Natural language inference: `ANLI`, `ARC-c`, `ARC-e`.
    - Coreference: `Winogrande`.
    - Sentence completion: `HellaSwag`, `LAMBADA`.
    - Knowledge/multi-choice QA: `MMLU` (5‑shot), `OpenBookQA`, `PIQA`, `PubMedQA`, `SciQ`, `CoQA`, `SocialIQA`, `TruthfulQA`.
  - Aggregation: “Avg,” “Normalized Avg,” and a “Rank‑Score” that sums normalized per‑task ranks to avoid scale confounds (Table 5 caption).

- Perplexity checks
  - Validated on `Pile` and `Paloma` held‑out sets for domain‑fit diagnostics (Tables 5–6).

B. Main quantitative findings (with specific numbers)
1) 468M model ablations on RPv2 (Table 5; detailed task scores in Tables 18–20)
   - The combination “2023‑14 crawl + exact + fuzzy dedup + full Gopher rules” attains:
     - Aggregate Avg 37.6, Normalized Avg 0.160, Rank‑Score 0.700.
     - For comparison, `RefinedWeb` achieves Avg 37.9, Norm Avg 0.165, Rank‑Score 0.650.
   - Quote:
     > In Table 5, “RPv2 (2023‑14) ✔ exact ✔ fuzzy (full) Gopher” reaches Avg 37.6, Norm Avg 0.160, Rank‑Score 0.700; `RefinedWeb` is 37.9 / 0.165 / 0.650.
   - Interpreting the trio of metrics: RPv2+Gopher+d edup is a close second on mean accuracy but ranks best overall across tasks (higher Rank‑Score), suggesting broader robustness.
   - The “C4 line filters” reduce validation perplexity but barely move aggregate accuracy (Table 5 row “RPv2 (9 Dumps) ✔ ✔ (line-filter) ...”: Avg 36.4 vs. 36.7 for full Gopher on nine dumps).
   - ML filters: fastText and DSIR perform similarly; using either on top of Gopher/repetition/natlang tends not to be decisive (Table 5; also Section 4.3.2).
   - Paloma perplexity is lowest for the unfiltered single-crawl RPv2 model (39.9 for “RPv2 (2023‑14) ✔ exact” vs. higher when filtered; Table 5), which the paper attributes to Paloma’s broad domain coverage and its inclusion of RPv1 (Section 4.3.2).

2) 1.6B model ablations on RPv2 (Table 6; per‑task in Tables 21–23)
   - With fuzzy dedup + full Gopher + WikipediaRef classifier (English), RPv2 reaches:
     - Agg Avg 50.0, Norm Avg 31.1, Rank‑Score 0.106; Pile/Paloma perplexities 13.6 / 20.8.
   - `RefinedWeb` remains a strong reference:
     - Agg Avg 52.0, Norm Avg 34.0, Rank‑Score 0.139; Pile/Paloma 10.7 / 17.7 (Table 6).
   - Variant with natlang filtering + Palm‑mix classifier tends to score slightly lower in aggregate (Agg Avg 47.9; Table 6).

3) V1-based model evaluations (Section 3.2.2; Tables 7–9)
   - 3B base model (trained 800B tokens) vs. similar scale baselines:
     - Quote (Table 7):
       > On HELM classic, RedPajama‑INCITE‑Base‑3B beats GPT‑Neo and Pythia‑2.8B by 3–5 points; on LM Evaluation Harness subset, by 2–7 points.
   - 7B base model:
     - It trails LLaMA‑7B by 4.1 points and Falcon‑7B by 1.0 point on HELM classic (Section 3.2.2).
     - The gap is most visible on tasks relying on log probabilities (LM harness), which are sensitive to precise training dynamics (Section 3.2.2; Table 9).
   - 7B instruct model:
     - Quote (Table 8):
       > On HELM, RedPajama‑INCITE‑7B‑Instruct achieves 0.492 average, outperforming LLaMA‑7B (0.472), Falcon‑7B (0.441), and MPT‑7B (0.444).
     - Few‑shot strength is attributed to instruction tuning on P3 and Natural Instructions (Section 3.2.2).

C. Ablations, robustness, and diagnostics
- Filtering levers matter, and their effects are complementary:
  - Deduplication consistently helps (fuzzy MinHash often outperforms exact only).
  - “Full” Gopher rules outperform using only natlang or repetition subsets (Table 5).
  - C4 line filters mainly reduce perplexity rather than task accuracy (Table 5, Section 4.3.2).
- Cross‑task stability:
  - Using rank‑based aggregation, the RPv2+Gopher+d edup recipe frequently sits “upper‑middle” or better across many tasks, whereas single‑recipe datasets can be strong on some tasks but weaker on others (Section 4.3.2; “per‑benchmark tables 18–20”).
- Temporal and dedup dynamics:
  - Figure 3 shows unique documents drop sharply for older crawls when deduped from newest to oldest, indicating substantial redundancy in early Common Crawl snapshots.

D. Do the experiments support the claims?
- Yes, for the core data curation claims:
  - The numbers in Table 5 and Table 6 demonstrate that RPv2—when filtered with broadly accepted rules (Gopher) and robust dedup—competes closely with curated datasets like RefinedWeb, validating RPv2 as a versatile raw source for high‑quality pretraining data.
- On RPv1 fidelity:
  - The 7B base model’s performance lag is plausibly explained by (1) fp16 training and lower learning rates (hardware constraint), and (2) unavoidable ambiguities in the original LLaMA data recipe (Table 10). The paper surfaces these limitations transparently rather than claiming perfect replication.

## 6. Limitations and Trade-offs
- Assumptions and scope
  - RPv2 assumes that “quality” can be approximated by heuristics (natlang, repetition, toxicity lists), ML similarity to target domains, and dedup artifacts. While standard, these proxies carry known biases (Section 4.1.2 notes that ML filters can underrepresent minorities [15]).
  - The evaluation focuses on English benchmarks; non‑English quality signals are sparser (e.g., only a Wikipedia similarity classifier) (Table 13).

- What is not addressed
  - No thorough decontamination analysis against evaluation benchmarks (explicitly acknowledged in Conclusion).
  - No systematic audit of personally identifiable information (PII) or sensitive content prevalence (Conclusion).
  - Safety and fairness analyses are not the focus, though signals to support such filtering (toxicity lists, blacklists) are provided.

- Computational constraints
  - RPv1 7B training used fp16 on V100s (no bf16), likely hurting optimization and stability compared to modern A100/H100 setups (Section 3.2.1–3.2.2).
  - Ablations are limited to 468M and 1.6B models to explore many filters quickly. While appropriate for comparative trends, absolute performance may not extrapolate linearly to 30B–70B regimes (Conclusion).

- Data and scalability trade-offs
  - RPv2’s “raw + signals” design optimizes versatility over immediate cleanliness: users must pick filters and bear the cost of data selection. The benefit is flexibility; the cost is effort and potential misconfiguration.
  - Deduplication choices (exact vs. MinHash thresholds, temporal order) change corpus size and diversity (Figure 3), which can affect both perplexity and benchmark accuracy.

- Open questions
  - Optimal multi‑domain mixtures using RPv2 signals remain to be charted (e.g., combining natlang + repetition + DSIR in a principled recipe across languages).
  - How well do small‑scale ablation insights transfer to training much larger models for longer?

## 7. Implications and Future Directions
- How this changes the landscape
  - RPv2 shifts the community from consuming fixed, prefiltered corpora to operating a common “data substrate” with standardized, per‑document diagnostics. This enables:
    - Faster iterations on filtering recipes.
    - Reproducible, side‑by‑side ablations of competing heuristics and ML selectors.
    - Better transparency in what “high quality” means for a given application.

- Research enabled or suggested
  - Data selection research at scale:
    - Learn weighted mixtures over signals (e.g., train a meta‑selector that predicts downstream utility using the provided signals).
    - Explore dynamic curriculum (age‑aware, domain‑aware) using `ccnet_perplexity`, `ccnet_language_score`, DSIR weights, and repetition measures (Tables 11–14).
    - Systematic comparisons of fastText vs. DSIR vs. embedding‑based selectors; the paper’s finding that fastText and DSIR behave similarly (Table 5) is a starting point.
  - Cross‑lingual filtering:
    - Extend ML similarity signals beyond Wikipedia for non‑English (Table 13 notes English‑only Palm‑mix).
    - Study whether natlang/repetition heuristics transfer uniformly across languages with different tokenization artifacts.
  - Benchmark decontamination and safety:
    - Use dedup ids and minhash clusters for stronger decontamination pipelines.
    - Build refined toxicity/PII filters and release them as additional signals aligned to RPv2’s schema.

- Practical applications
  - Enterprises and researchers can tailor corpora:
    - Safety‑first variants (tight natlang + toxicity filters + strict dedup).
    - Domain‑focused variants (DSIR/fastText targeting legal, biomedical, or coding domains by training domain‑specific n‑gram LMs or classifiers—hooking into RPv2’s signal schema).
    - Efficiency‑oriented variants (aggressive repetition control to mitigate degenerate generations).
  - Training and productization:
    - The paper cites production‑used models trained on RedPajama data (e.g., Snowflake Arctic, Salesforce XGen, AI2 OLMo; p. 1), indicating immediate utility.

- Concrete takeaways for practitioners
  - If you want a strong general web corpus without bespoke engineering, start from RPv2 with fuzzy dedup + full Gopher rules; in 468M experiments, this combination is near the top on averaged metrics and best on rank‑based aggregate (Table 5).
  - If low perplexity on heterogeneous validation sets is critical, be cautious: line‑level C4 filters reduce perplexity but may not lift task accuracy (Table 5 and Section 4.3.2).
  - For few‑shot instruction use cases, instruction‑tuning on top of RPv1‑trained models improves HELM performance substantially (Table 8).

Block quotes of notable results and references to help the reader verify:
- > “RPv2 (2023‑14) ✔ ✔ (full) [Gopher] … Avg 37.6, Norm Avg 0.160, Rank‑Score 0.700” (Table 5).
- > “RefinedWeb … Avg 37.9, Norm Avg 0.165, Rank‑Score 0.650” (Table 5).
- > “7B‑Instruct HELM‑AVG 0.492 vs. LLaMA‑7B 0.472” (Table 8).
- > “3B Base outperforms GPT‑Neo and Pythia‑2.8B by 3–5 points on HELM classic and by 2–7 points on a subset of LM‑Eval” (Section 3.2.2; Table 7).
- > “CCNet buckets, language scores, and perplexity distributions; ML similarity histograms; natural‑language and repetition signal histograms” (Figures 4–7).

In sum, the paper contributes a new way to do open LLM data work: make the raw web data accessible at scale and ship the metadata that turns it into a programmable laboratory for data curation. The ablations show that well‑known rules (dedup + Gopher) recovered from these signals produce competitive corpora, and the V1 replication—with full process notes—adds valuable transparency about the impact of compute and recipe ambiguities on model outcomes.
