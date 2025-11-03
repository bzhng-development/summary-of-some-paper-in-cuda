# RedPajama: an Open Dataset for Training Large Language Models

**ArXiv:** [2411.12372](https://arxiv.org/abs/2411.12372)
**Authors:** Maurice Weber, Daniel Fu, Quentin Anthony, Yonatan Oren, Shane Adams, Anton Alexandrov, Xiaozhong Lyu, Huu Nguyen, Xiaozhe Yao, Virginia Adams, Ben Athiwaratkun, Rahul Chalamala, Kezhen Chen, Max Ryabinin, Tri Dao, Percy Liang, Christopher Ré, Irina Rish, Ce Zhang
**Institutions:** Together Computer (and likely affiliated institutions of authors, including Stanford, UC Berkeley, etc.)

## 🎯 Pitch

RedPajama revolutionizes open large language model training by providing two extensive pretraining datasets: RedPajama-V1, a detailed replication of LLaMA-1's data blend, and RedPajama-V2, a vast, multilingual ‘web-only’ corpus with fine-grained quality annotations. This transparency in data sources not only tackles the traditional scarcity and opacity in LLM training data but also empowers researchers to refine and standardize open model development, fostering reproducibility and innovation across the AI community.

---

## 1. Executive Summary
RedPajama introduces two large, fully open pretraining datasets for large language models (LLMs): `RedPajama‑V1`, a best‑effort reproduction of the original LLaMA‑1 training mix, and `RedPajama‑V2`, a massive “web‑only, raw+signals” corpus spanning five languages with document‑level quality annotations. This work matters because it tackles three blocking issues for open LLMs—lack of transparency about training data, limited access to sufficiently large corpora, and scarce reusable artifacts for data curation—by releasing >100 trillion tokens plus detailed filtering signals, running ablations to show how to turn raw web into high‑quality data, and training open models to validate the recipes.

## 2. Context and Motivation
- Gap addressed
  - Most state‑of‑the‑art LLMs either do not disclose their pretraining data or do so at a very high level, which makes it difficult to reproduce results, study data‑quality trade‑offs, or build open models at scale. Section 1 identifies three data bottlenecks: transparency in curation, access to large, high‑quality data, and availability of reusable curation artifacts.
- Importance
  - Real‑world impact: Training data decisively shapes model behavior, safety, and capability; openness enables reproducibility and community iteration. Section 1 points to widespread model adoption that relies on unknown data mixtures.
  - Scientific significance: Without shared corpora and curation recipes, the field cannot systematically study how data composition and filtering affect downstream performance.
- Prior approaches and shortfalls
  - Curated composite corpora (e.g., The Pile) and cleaned web‑only corpora (e.g., C4, RefinedWeb, FineWeb) advanced open training but typically ship either a prefiltered dataset or limited quality metadata. Table 1 compares open datasets along transparency, versatility, and scale; few provide both raw web at massive scale and the per‑document quality signals needed to try many filtering strategies.
- Positioning
  - RedPajama provides:
    - `V1`: a transparent replication of the LLaMA‑1 recipe (Table 2 lists the seven slices and 1.2T tokens), plus trained `RedPajama‑INCITE` models used to sanity‑check the replication (Section 3).
    - `V2`: a gigantic Common Crawl–based corpus with minimal preprocessing but rich quality signals (46 measures per document) so users can construct many high‑quality subsets as “views” of the raw web (Sections 4.1–4.2; Table 3).

## 3. Technical Approach
This work has two complementary data products and a validation track.

- RedPajama‑V1: open reproduction of the LLaMA‑1 data mix (Section 3)
  - What’s inside (Table 2; total ~1.2T tokens):
    - `CommonCrawl` (processed with `CCNet`: a pipeline that deduplicates within snapshots and assigns a “head/middle/tail” quality bucket using a Wikipedia‑trained n‑gram language model)
    - `C4` (a cleaned Common Crawl subset)
    - `GitHub` (Apache/BSD/MIT licensed code; filtered by file heuristics listed in Appendix C.1)
    - `Books` (PG‑19; Books3 initially included then removed for copyright)
    - `ArXiv` (LaTeX source; cleaned to remove preambles, comments, bibliography)
    - `Wikipedia`
    - `StackExchange` (28 largest sites; HTML stripped; answers sorted by score)
  - Recreating LLaMA‑1 involved filling underspecified steps (Table 10 summarizes “uncertainties” and decisions):
    - Selected five English Common Crawl snapshots (2019–2023) and trained a `fastText` Wikipedia‑references classifier to filter (Section 3.1).
    - Applied language‑specific preprocessing consistent with cited sources (e.g., arXiv LaTeX cleaning following [29]).
  - Validation by training models (`RedPajama‑INCITE`, Section 3.2):
    - Trained 3B and 7B parameter decoder‑only models on the Summit supercomputer (Section 3.2.1).
    - Summit specifics constrained training: V100 GPUs don’t support `bf16`, so they used `fp16` with loss scaling; reduced learning rates; 12‑stage pipeline parallelism for 7B, 6‑stage for 3B, and 2‑way tensor parallel for both; 512 nodes (7B) and 256 nodes (3B); 4M token global batch (Section 3.2.1).
    - Tokens seen: 800B (3B) and ~1.0T (7B) (Section 3.2.1).

- RedPajama‑V2: raw web text + per‑document quality signals (Section 4)
  - Data acquisition and minimal processing (Section 4.1.1):
    - Includes text extracted from all 84 Common Crawl “WET” snapshots between 2014 and April 2023; passed through `CCNet` to produce >100B documents.
    - Languages: English, German, French, Spanish, Italian.
    - Unlike common practice, retains `head`, `middle`, and `tail` perplexity buckets to preserve breadth.
  - Quality signals: how they work and what they measure (Section 4.1.2; Appendix D)
    - Natural language heuristics (Table 12): e.g., fraction of all‑caps words, fraction of lines ending with ellipsis, stopword ratios, unigram entropy—aimed at catching boilerplate or non‑linguistic text.
    - Repetitiveness (Table 14): fractions of characters in duplicated n‑grams (5–10) and in the most frequent n‑grams (2–4)—repeat‑heavy pages often correlate with low informativeness.
    - Content‑based flags (Table 15): counts of blocklisted words (`LDNOOBW`) and domain blacklist categories (`UT1`) to help exclude NSFW or spammy sites.
    - ML heuristics (Table 13): 
      - `fastText` classifiers that score similarity to high‑quality domains (Wikipedia pages, Wikipedia references, OpenWebText, books).
      - `DSIR` importance weights (Data Selection via Importance Resampling): log‑likelihood ratios between bigram models of the target vs. source distributions—higher means more “target‑like.”
    - Deduplication signals:
      - Exact duplicates via `Bloom` filter (1% error rate), tracked by IDs; dedup proceeds snapshot‑by‑snapshot from newest to oldest (Section 4.1.2; footnote 6).
      - `MinHash` signatures for fuzzy deduplication at multiple Jaccard similarities (Appendix B.1.2 “Minhashes”).
    - Storage format binds signals to text spans: each signal is stored as triplets `[start, end, score]` pointing into the original text so both line‑level and document‑level features coexist (Appendix B.1.2 “Quality Signals Structure”).
  - Scale and composition (Section 4.2; Table 3):
    - 113.3B documents, 123.7T tokens estimated with the Mistral BPE tokenizer.
    - `head+middle` buckets: 32.8B docs, 50.7T tokens; after dedup: 20.8B docs, 30.4T tokens.
    - Typical length: tail ~850 tokens vs. head/middle ~1,500 tokens.
    - Per‑language counts detailed in Table 3.
  - “Versatility by design”: The dataset is intentionally not prefiltered; instead, it provides the metadata needed to instantiate many different high‑quality “views” (Section 4 and Appendix B).

- Experimental ablations: how to turn raw V2 into good training sets (Section 4.3)
  - Models and training setup (Section 4.3.1):
    - Decoder‑only LLaMA‑2–style models at 468M and 1.6B parameters, sequence length 2048; 24 layers, 16 attention heads, MLP ratio 4.0. Hidden size 1024 (468M) and 2048 (1.6B).
    - Tokens: 100B (468M) and 350B (1.6B). AdamW; peak LR 5e‑3 (468M) and 5e‑4 (1.6B); cosine decay; 1% warmup.
    - Trained with OLMo framework using FSDP on H100s (Section 4.3.1 “Hardware and Training Stack”).
  - Filtering recipes evaluated (Section 4.3.2; Tables 5–6):
    - “C4 rules,” “Gopher rules” (a widely used set of web filtering heuristics), exact/fuzzy deduplication, ML heuristics (`fastText` vs `DSIR`), custom rules mixing Wikipedia perplexity and classifiers.
    - Two data scopes: a single 2023‑14 snapshot and a set of nine snapshots (2021‑49 to 2023‑14).

## 4. Key Insights and Innovations
- Open release of raw web + rich quality signals at unprecedented scale
  - What’s new: `RedPajama‑V2` does not just ship a prefiltered corpus; it ships 46 per‑document signals and dedup artifacts so practitioners can derive many datasets (Section 4.1.2; Appendix D.2.1). This contrasts with datasets like C4 or RefinedWeb that provide cleaned text but limited annotations. Table 1 lists V2 as “Open Access + Open Code + Raw Data + Multilingual” with 270 TB scale.
  - Why it matters: It enables rapid, principled exploration of filtering strategies without re‑crawling or recomputing expensive diagnostics, supporting reproducible data science at web scale.

- Transparent, documented replication of LLaMA‑1 data with explicit uncertainty resolution
  - What’s new: Section 3.1 and Table 10 enumerate missing details in the original LLaMA recipe and document concrete choices (snapshots, classifier thresholds, code filtering rules). This level of documentation is rare for high‑profile datasets.
  - Why it matters: It provides a stronger baseline for reproducibility and educates the community about sensitive steps (e.g., classifier training for Wikipedia references).

- Empirical evidence that “signals + minimal processing” can match or approach curated web datasets
  - What’s new: With the 468M model, combining fuzzy deduplication and full `Gopher` rules on RPv2 yields competitive aggregate benchmark scores—second only to RefinedWeb on some aggregations and better rank‑consistency across tasks (Table 5 and Appendix Tables 18–20).
  - Why it matters: It supports the premise that large, weakly processed web corpora can be shaped into strong training data using transparent, reusable signals.

- Practical training validation at scale on FP16‑only hardware
  - What’s new: Section 3.2.1 details how to train multi‑billion‑parameter models on IBM Power9 + V100 with older software stacks, including pipeline/tensor parallel settings and lower LRs for FP16 stability.
  - Why it matters: It demonstrates workable recipes for environments where bf16 is unavailable, helping other teams working on constrained HPC systems.

## 5. Experimental Analysis
- Evaluation methodology (Section 4.3.1; Table 4)
  - Benchmarks: A carefully chosen set with good signal even for small models: ANLI, ARC‑c/e, Winogrande, HellaSwag, LAMBADA, CoQA, MMLU (and sub‑domains), OpenBookQA, PIQA, PubMedQA, SciQ, SocialIQA, TruthfulQA.
  - Aggregation: Three metrics—average accuracy, normalized average, and a rank‑based score—to avoid scale effects across tasks (Tables 5–6).
  - Perplexity probes: Validation perplexity on Paloma and The Pile (Table 5), following Dolma’s evaluation practice.

- Main quantitative results for RPv2 filtering (468M model; Table 5, with per‑task in Tables 18–20)
  - Best overall filters on RPv2:
    - Single snapshot (2023‑14): “Exact + Fuzzy dedup + Gopher (full)” achieves:
      > Aggregate BM‑Eval: Avg 37.6, Norm Avg 0.160, Rank‑Score 0.700; Pile ppl 24.9, Paloma ppl 34.5 (Table 5).
    - Nine snapshots: “Exact + Fuzzy + Gopher (full)” achieves:
      > Avg 36.7, Norm Avg 0.149, Rank‑Score 0.556; Pile ppl 43.8, Paloma ppl 63.9 (Table 5).
  - Comparisons against strong web datasets:
    - RefinedWeb: 
      > Avg 37.9 (best), Norm Avg 0.165 (best), Rank‑Score 0.650 (Table 5); but per‑task results show RefinedWeb lags RPv2+Gopher on specific tasks such as HellaSwag, LAMBADA, Winogrande, MMLU, and OpenBookQA (Appendix narrative below Table 5).
    - FineWeb and Dolma‑v1.7: RPv2 with Gopher generally equals or exceeds their aggregate scores (Table 5).
  - ML heuristics:
    - `fastText` vs. `DSIR` provide similar aggregate gains; neither clearly dominates across the board (Table 5).
  - C4 line‑level filters:
    - Reduce perplexity but have minor effect on the aggregate benchmark scores (Table 5).

- Results at 1.6B scale (Table 6, Tables 21–23)
  - RPv2 (full) with fuzzy dedup + Gopher (natlang only) + `Palm‑Mix` classifier:
    > Aggregate BM‑Eval Avg 47.9, Norm Avg 29.4, Rank‑Score 0.089; Pile ppl 22.2, Paloma ppl 30.7 (Table 6).
  - RefinedWeb remains ahead overall at this scale:
    > Avg 52.0, Norm Avg 34.0, Rank‑Score 0.139; Pile ppl 10.7, Paloma ppl 17.7 (Table 6).
  - Interpretation: With more training tokens (350B) and a larger model, curated datasets like RefinedWeb still lead, but RPv2 can be filtered to approach them on several tasks (Tables 21–23).

- Validation of V1 via RedPajama‑INCITE models (Section 3.2.2; Tables 7–9)
  - 3B model (800B tokens):
    > Outperforms GPT‑Neo and Pythia‑2.8B on HELM by 3–5 points and on LM harness subsets by 2–7 points (Section 3.2.2). Table 7 shows, for example, LAMBADA 0.654 vs Pythia‑2.8B 0.647 and HELM avg 0.406 vs 0.377.
  - 7B base model (~1T tokens):
    > Trails LLaMA‑7B by 4.1 points and Falcon‑7B by 1.0 point on HELM classic, particularly on “logprob”‑style tasks; direct generation tasks are comparable (Section 3.2.2; Table 8 and Table 9).
  - Plausible causes: FP16 training constraints and unavoidable dataset recipe mismatches (Section 3.2.2; Table 10).

- Robustness and ablations (Section 4.3.2; Appendix tables)
  - The “Gopher” family of rules consistently improves results relative to unfiltered or line‑only recipes (Table 5).
  - Repetition‑focused Gopher filters help, but “natlang” components contribute even more (Table 5).
  - Using more crawls increases domain coverage but can worsen perplexity on curated validation sets unless combined with strong filtering and deduplication (contrast single vs nine snapshots in Table 5).

- Do the experiments support the claims?
  - The dataset’s central claim—“quality signals enable building high‑quality subsets from raw web”—is supported by consistent gains from principled filters (Gopher, fuzzy dedup) and by competitive performance relative to curated datasets at 468M scale (Table 5).
  - At larger scale (1.6B), curated datasets still lead overall (Table 6), implying that filtering strategies for RPv2 can be further optimized.

## 6. Limitations and Trade-offs
- Assumptions and potential biases
  - ML heuristics rely on `bag‑of‑words` features (`fastText`, DSIR) that emphasize surface statistics; these can bias selection toward “Wikipedia‑like” text and underrepresent other valuable domains (Section 4.1.2 notes known bias risks; cites [15]).
  - `CCNet`’s head/middle/tail buckets depend on a Wikipedia‑trained language model; this assumes Wikipedia‑like fluency/structure is the correct proxy for “quality” (Section 3.1 and 4.1.1).
- Scope not covered
  - No full decontamination analysis against evaluation benchmarks; no systematic assessment of personally identifiable information (Conclusion, Section 5).
  - V2 focuses on web text only; domains like code, scientific papers, or books must be sourced separately if needed (Section 4).
- Computational constraints
  - Validation models for V1 were trained on hardware that forced `fp16` and reduced LRs; this likely limited 7B performance (Section 3.2.1–3.2.2).
  - Ablations used relatively small models (468M and 1.6B) to explore many filters; results may not extrapolate linearly to 30B+ scales (Section 5).
- Data realities
  - Raw RPv2 includes harmful/offensive content; users must apply filtering signals appropriately (Section 4.1.2; Appendix D.2.1).
  - Deduplication is performed pre‑`CCNet` exact (footnote 6) and with `MinHash` for fuzzy matches; different parameterizations yield different trade‑offs between recall and false positives.

## 7. Implications and Future Directions
- How this changes the landscape
  - Provides a new “substrate” for open LLM training: rather than prescribing one cleaned corpus, RPv2 makes it easy to iterate on filtering strategies at web scale. Table 1 positions RPv2 as unique on transparency + scale + versatility.
  - Already used to train several open models (Figure 1 highlights OpenELM, OLMo, Snowflake Arctic, and RedPajama‑INCITE), demonstrating practical value.
- Follow‑up research enabled
  - Data‑centric studies: optimize combinations of signals (natlang + repetition + ML) per task family; learn filters via differentiable or reinforcement learning approaches; compare DSIR vs. learned selectors beyond bag‑of‑words.
  - Bias and safety audits using the released signals; build fairness‑preserving filters that retain underrepresented language and domain varieties.
  - Multilingual extension: RPv2 currently covers five languages (Table 3); extending signals (e.g., non‑English `Palm‑mix` classifiers) could improve non‑English model performance.
  - Scaling laws for data curation: systematically study how dedup strength, snapshot mix, and quality thresholds interact with model size and training tokens.
- Practical applications
  - Enterprises and researchers can construct task‑specific corpora quickly (e.g., low‑toxicity customer support datasets, domain‑specific pretraining for finance or biomedicine) by combining content whitelists with ML/heuristic signals (Appendix D lists all features).
  - Reproducible baselines for open LLM training, including the ability to replicate filtering pipelines and share exact dataset “views” by referencing the released HTTP manifests (Appendix B).

> Availability and structure: Both datasets are downloadable via Hugging Face and public HTTPS endpoints; files are sharded JSONL with well‑documented schemas for documents, quality signals, duplicates, and minhashes (Appendix B.1.1–B.1.2).

In sum, RedPajama advances openness not just by releasing big data, but by releasing the metadata and ablation evidence needed to transform raw web into strong training sets. The experiments show clear benefits from principled filtering (especially Gopher rules plus fuzzy deduplication), with room to close remaining gaps to carefully curated datasets at larger model scales.
