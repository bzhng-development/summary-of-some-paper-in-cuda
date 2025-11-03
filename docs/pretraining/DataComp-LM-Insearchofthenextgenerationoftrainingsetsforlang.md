# DataComp-LM: In search of the next generation of training sets for language models

**ArXiv:** [2406.11794](https://arxiv.org/abs/2406.11794)

## 🎯 Pitch

DataComp-LM (DCLM) delivers the first large-scale, controlled testbed specifically designed to systematically evaluate and improve training datasets for language models. By providing a massive 240-trillion-token web corpus, standardized training and evaluation recipes across five compute scales, and open-source curation tools, DCLM enables reproducible, apples-to-apples comparisons of data-centric strategies—breaking the current bottleneck where progress is obscured by architecture and compute differences. This empowers the community to build better language models more efficiently, as evidenced by their strong open baseline that rivals state-of-the-art models with far less compute, fundamentally advancing transparent, data-first research in NLP.

---

## 1. Executive Summary (2–3 sentences)
DataComp-LM (DCLM) introduces the first large-scale, controlled benchmark to study how training data choices affect language model (LM) quality. It supplies a 240-trillion-token web pool, standardized training/evaluation across five compute scales, open tooling, and a strong baseline dataset (DCLM-BASELINE) curated with model-based filtering that trains a 7B model to 64% MMLU with substantially less compute than comparable open models (Table 8).

## 2. Context and Motivation
- Problem/gap addressed
  - There is no controlled, end-to-end benchmark to compare LM training datasets. Model performance differences are often confounded by architecture, compute, and hyperparameters rather than the data itself (Section 1).
  - Training set details for leading open-weight models (e.g., Llama, Mistral, Gemma) are opaque or unavailable, making it unclear what constitutes a state-of-the-art training set (Section 1).

- Why it matters
  - Training compute is skyrocketing; improving data quality delivers better models at lower cost (Figure 1).
  - A rigorous testbed allows the community to iterate efficiently on data curation strategies with reproducible comparisons and transparent trade-offs (Sections 1–3).

- Prior approaches and shortcomings
  - Open corpora (C4, The Pile, RefinedWeb, Dolma, FineWeb, RedPajama) exist but differ in extraction, filters, deduplication, and mixing; cross-paper comparisons mix in different models, training budgets, and recipes (Section 2).
  - Data-centric efforts for other modalities exist (DataComp for vision; DataPerf), but not at LM pretraining scale with trillion-token pools (Section 2).

- Positioning relative to existing work
  - DCLM provides: (1) a massive standardized pool (DCLM-POOL, 240T tokens), (2) a unified training stack and five compute scales, (3) a 53-task evaluation suite with low-variance metrics, (4) open-source data processing tooling, and (5) a strong, fully released baseline dataset and models (Sections 3.1–3.5; Appendix D; Figure 2).

## 3. Technical Approach
DCLM is both a benchmark and a methodology. It standardizes every step from raw crawl → curation → training → evaluation so that changes in model quality can be attributed to data decisions.

- The workflow (Figure 2)
  1) Choose a compute scale. Five “competition scales” specify model size, training tokens, pool size, and approximate compute (Table 1). A “Chinchilla multiplier” sets the data budget (e.g., `20× parameters × multiplier`) to normalize compute regimes.
     - Example: `7B-1x` trains a 6.9B-parameter model on 138B tokens; `7B-2x` trains the same size on 276B tokens (Table 1).
  2) Curate data in one of two tracks (Section 3.3).
     - Filtering track: Select a subset from the scale-specific pool (random splits of the 240T DCLM-POOL) using any filtering/dedup pipeline.
     - Mixing track: Combine filtered Common Crawl (CC) with external sources (e.g., Wikipedia, StackExchange).
  3) Train with a fixed recipe (Section 3.4; Appendix F).
     - Architecture: decoder-only Transformer implemented in `OpenLM`; layer norm, qk-LayerNorm for stability, SwiGLU MLPs; 2048 context (later extended to 8192 via continual learning).
     - Tokenizer: GPT-NeoX (50k vocabulary).
     - Optimization: Adam with z-loss; learning-rate/weight-decay schedules per scale (Table 10; Table 11).
  4) Evaluate on 53 tasks with standardized prompts and metrics (Section 3.5; Appendix G).
     - Primary metrics:
       - `MMLU (5-shot)`—standard LM capability benchmark.
       - `CORE centered accuracy`—a 22-task subset with low variance, normalized per task between random (0) and perfect (1).
       - `EXTENDED centered accuracy`—averaged across all 53 tasks.

- Building the baseline dataset (DCLM-BASELINE)
  - Step A—Text extraction (Section 4.2; Table 3).
    - HTML-to-text matters a lot. `resiliparse` and `trafilatura` both beat Common Crawl’s WET text by ≥2.5 CORE points at 1B scale; `resiliparse` is 8× faster (Table 16; Appendix K), so DCLM-POOL re-extracts all CC HTML using `resiliparse`.
  - Step B—Heuristic cleaning (Figure 4; Sections 4.1–4.2).
    - Reproduces RefinedWeb-style filters (e.g., language, page length, repetition, URL heuristics), which outperformed other popular datasets at 7B-1x (Table 2).
  - Step C—Deduplication (Section 4.3; Appendix L).
    - Two families are studied at scale:
      - MinHash + suffix arrays (near-duplicate doc-level + substring-level removal)—effective but costly.
      - Modified Bloom filter (“BFF”) at document and paragraph levels—comparable downstream performance but far better scalability to >10TB (Section 4.3). At 7B-2x, MinHash and BFF are within 0.2 CORE points.
  - Step D—Model-based quality filtering (Section 4.4; Tables 4–5).
    - Explored: PageRank scores, semantic dedup (SemDedup), embeddings classifiers, `AskLLM` judgments, perplexity filtering, top-k average logits, and `fastText` linear classifiers.
    - Best: a simple `fastText` classifier trained with bag-of-ngrams (unigram + bigram) using instruction-style positives (`OpenHermes-2.5` + high-karma `ELI5` answers) vs. negatives sampled from the RefinedWeb-like pool; keep top 10% by score (Table 5; Table 14).
  - Step E—Mixing with “high-quality” sources (Section 4.5; Table 6).
    - Mixing Wikipedia/Books/StackExchange/arXiv/GitHub improves weaker CC subsets (C4, RPJ-CC), but slightly hurts the strong filtered DCLM-BASELINE at 1B scale (−1.2 CORE). Conclusion: if Common Crawl is already well filtered, mixing is not necessarily beneficial.

- Decontamination tooling and checks (Section 3.1; Section 4.6; Appendix O)
  - The benchmark ships tools to measure overlap with evals (Lee et al.’s approach) rather than “pre-decontaminating” the pool, because the effect varies by task.
  - Removing detected overlaps for MMLU/HellaSwag in DCLM-BASELINE did not reduce performance at 7B-2x (Table 7).
  - Extended contamination analysis shows minimal differences between full and “not-dirty/clean” subsets for most tasks (Figures 11–12).

- Scaling up and additional training (Section 5; Appendices P–Q)
  - Combined DCLM-BASELINE (3.8T tokens) with StarCoder and ProofPile2 to 4.1T tokens; trained a 7B model for 2.5T tokens with two cool-downs and “model soup” (Table 28).
  - Extended context from 2k to 8k via continual pretraining with a variable-sequence curriculum and RoPE base frequency change (Table 30).
  - Instruction tuning on public data (OpenHermes-2.5 and a curated DCLM-IT mix) yields strong AlpacaEval2.0 results (Table 26) while retaining most base model performance.

Definitions of paper-specific or uncommon terms
- `DCLM-POOL`: A 240-trillion-token corpus re-extracted from Common Crawl HTML using `resiliparse` (Section 3.1).
- `DCLM-BASELINE`: A 3.8T-token dataset curated from DCLM-POOL via heuristic cleaning, Bloom filter deduplication, and fastText-based quality filtering (Figure 4).
- `fastText classifier`: A lightweight linear text classifier that uses bag-of-ngrams features (unigrams/bigrams) for efficiency and scalability (Section 4.4; Table 14).
- `resiliparse / trafilatura / WET files`: Three alternative HTML-to-text extraction methods (Section 4.2).
- `Bloom filter (BFF)`: A probabilistic set-membership data structure adapted to remove near-duplicate paragraphs and documents at scale (Sections 4.3; Appendix L).
- `CORE / EXTENDED centered accuracy`: Normalized accuracy metrics that aggregate across tasks; CORE is a low-variance 22-task subset (Section 3.5).

## 4. Key Insights and Innovations
1) A standardized, multi-scale benchmark where dataset rankings transfer across scales
   - What’s new: Five fixed compute scales, each with its own CC pool subset and fixed training hyperparameters (Table 1; Section 3.2).
   - Why it matters: Correlations of dataset rankings between smaller and larger scales are high (Pearson r = 0.838/0.956/0.982 at 400M/1B/3B vs. 7B-1x; Figure 3). This validates rapid iteration at low cost without losing validity at larger scales.

2) Simple, well-chosen model-based quality filters beat more complex alternatives
   - What’s new: A `fastText` bigram classifier trained on instruction-style positives (`OH-2.5 + ELI5`) outperforms perplexity, embedding classifiers, and LLM-judged `AskLLM` filtering (Table 4; Table 5; Table 14).
   - Why it matters: It’s simple, cheap, and effective. On 7B-1x, fastText (+OH-2.5+ELI5, top-10%) yields 41.0 CORE vs. 37.5 with GPT-3-like positive sets, and it scales trivially (Table 5).

3) HTML extraction and dedup details critically affect downstream quality and scalability
   - What’s new: Re-extracting CC using `resiliparse` or `trafilatura` yields sizable gains over WET text (Table 3). A modified Bloom-filter dedup performs comparably to MinHash + suffix arrays while scaling better beyond 10TB (Section 4.3; Tables 18–19).
   - Why it matters: Extraction and dedup are often treated as “plumbing,” but they shift CORE by multiple points and change feasibility at very large data scales (Appendix K, L).

4) Mixing “high-quality” sources is not automatically beneficial
   - What’s new: Mixing with Wikipedia/Books/etc. helps weaker CC subsets but slightly hurts the strongest CC-only filtered dataset (Table 6).
   - Why it matters: High-quality mixing is not a universal good; if Common Crawl is already carefully filtered, extra mixing can reduce performance.

5) A high-quality open dataset that closes the gap to closed-data models at a fraction of compute
   - What’s new: DCLM-BASELINE trains a 7B model to 63.7% MMLU with 2.6T tokens, competitive with Mistral-7B-v0.3 (62.7) and near Llama-3 8B (66.2), despite far less compute than Llama-3 8B (Table 8).
   - Why it matters: Demonstrates the leverage of data curation. The 7B-1x model also beats Llama-2 7B trained with 7× more compute when trained on the same 280B tokens (Section 1).

## 5. Experimental Analysis
- Evaluation design (Section 3.5; Appendix G)
  - 53 downstream tasks spanning commonsense, QA, reading comprehension, logic, math, and long-context; fixed prompting; two aggregate metrics (CORE, EXTENDED) plus MMLU 5-shot.
  - The paper also cross-checks with LightEval for MMLU and discusses its differences and sensitivities (Appendix G.2; Figure 5).

- Baselines and ablations
  - Existing datasets at 7B-1x: RefinedWeb > RedPajama ≈ Dolma > C4 on CORE/EXTENDED (Table 2). This justifies adopting RefinedWeb’s heuristic pipeline as the starting point (Section 4.1).
  - HTML extraction: `resiliparse`/`trafilatura` outperform WET by 2.5–3.8 CORE at 1B-1x (Table 3).
  - Deduplication: At 7B-1x and 7B-2x, Bloom-filter dedup (with `min_ngram_size=13`) matches MinHash+Suffix arrays within noise, with better scalability (Tables 18–19).
  - Model-based filtering comparison (Table 4):
    - `fastText` outperforms `AskLLM`, embeddings classifiers, perplexity, PageRank, and top-k logits; perplexity and top-k logits are competitive but lower.
  - `fastText` ablations (Table 5; Table 14):
    - Positive set matters most: instruction-style positives (`OH-2.5 + ELI5`) yield +3.5 CORE over conventional positives (Wikipedia, OWT2).
    - Threshold matters: top-10% beats top-15% and top-20%.
    - Bigrams > unigrams in features (+1 CORE; Table 14).
  - Mixing (Table 6; Appendix M):
    - Adds +0.8–2.2 CORE to weaker CC subsets; −1.2 CORE for DCLM-BASELINE.
  - Decontamination checks (Section 4.6; Appendix O):
    - Removing overlaps with MMLU/HellaSwag does not decrease performance at 7B-2x (Table 7).
    - DCLM-BASELINE contamination rates for MMLU are similar to Dolma and FineWeb-Edu (Table 25).
    - Differences between full and “clean/not-dirty” subsets are small for most tasks (Figures 11–12).
  - Robustness to hyperparameters and architectures:
    - Dataset rankings are stable across LR/WD settings (Table 12) and improvements stack with better hyperparameters (Table 13).
    - Rankings correlate across different architectures (Gemma-like, Mamba) and OpenLM (Figure 6).

- Main quantitative results
  - Compute-vs-quality frontier (Figure 1; Table 33, 8):
    - The DCLM-BASELINE-trained 7B model obtains 57.1 CORE, 63.7 MMLU, 45.4 EXTENDED at ~2.6T tokens (Table 8). This outperforms all open-data-trained 7B models listed and approaches closed-data models like Llama-3 8B (66.2 MMLU).
  - Scale transfer (Figure 3): Methods that win at 400M–3B scales also win at 7B, enabling low-cost iteration.
  - Long-context continual learning: extending context to 8192 improves multi-document QA substantially without sacrificing standard evals (Table 30).
  - Instruction tuning: DCLM-based models achieve competitive AlpacaEval2.0 scores (e.g., 16.6 LC win-rate for DCLM-IT) while preserving core capabilities (Table 26; Table 29).

- Do the experiments support the claims?
  - Yes, via comprehensive ablations isolating each pipeline decision:
    - HTML extraction, dedup method/parameters, classifier choice/positives/threshold, and mixing are each systematically varied with consistent evaluation. Results are stable across hyperparameters and architectures (Tables 12–13; Figure 6), and scale transfer is validated (Figure 3).

- Notable caveats and mixed outcomes
  - LightEval vs. LLM Foundry (Figure 5): small models get earlier non-random signal under LightEval, but scores compress at larger scales; authors rely on LLM Foundry for main comparisons (Appendix G.2).
  - Human judgments vs. actual gains: LLM-based `AskLLM` correlates more with human labels but underperforms fastText as a data filter (Appendix N; Figure 9).

> “fastText OH-2.5 + ELI5 (top 10%) achieves 41.0 CORE and 29.2 MMLU at 7B-1x, outperforming alternatives” (Table 5; Table 4).

> “resiliparse or trafilatura extraction improves CORE by ≥2.5 over WET at 1B-1x” (Table 3).

> “DCLM-BASELINE 7B reaches 63.7 MMLU with 2.6T tokens, competitive with Mistral-7B (62.7) and near Llama3-8B (66.2)” (Table 8).

## 6. Limitations and Trade-offs
- Scope and compute
  - While multi-scale, the largest model size explored is 7B; results may shift for >7B models (Section 6).
  - Although more compute-efficient than some baselines, full participation at larger scales still requires significant resources (Table 1; Appendix R).

- Data scope and domain coverage
  - The benchmark is centered on Common Crawl up to 2022; multilingual, code, and math are not the main focus in this first version (Section 6). The paper supplements code/math via StarCoder and ProofPile2 when scaling (Section 5).

- Security, safety, and privacy
  - DCLM-POOL includes raw web text; despite heuristic filters, it can contain PII, toxicity, and copyrighted material (Appendix U). Tooling is provided for decontamination analysis, but the pool itself is not decontaminated by default (Section 3.1).

- Generalization of findings
  - “Mixing hurts” is conditional: it holds for DCLM-BASELINE at 1B-1x but helps weaker CC subsets (Table 6). Conclusions may vary with different domains/goals.

- Dedup/membership definitions
  - Bloom-filter and MinHash dedup remove different kinds of duplicates; global MinHash still finds “fuzzy duplicates” after BFF (Appendix L.2.4; Table 23). The right dedup notion may depend on downstream objectives.

- Human quality judgments
  - Human annotator agreement was moderate (71%), and higher ROC-AUC vs. human labels did not translate to better downstream performance (Appendix N; Figure 9). What “quality” means for pretraining may differ from human intuitions.

## 7. Implications and Future Directions
- How this work changes the landscape
  - Establishes the first controlled, transparent, multi-scale benchmark for LM data curation with an unprecedented 240T-token pool and released tooling. This enables data-centric advances to be compared apples-to-apples and makes data design a first-class research axis alongside model and compute.

- Follow-up research enabled or suggested
  - Domain-aware curation: Extend DCLM to targeted domains (code, math, multilingual) with domain-specific metrics and pools (Section 6).
  - Better filtering models: Replace fastText with compact neural filters or hybrid methods (e.g., Color-Filter, data-selection via importance reweighting), while preserving efficiency.
  - Causal studies of contamination and duplication: More precise decontamination definitions per task; measure memorization vs. generalization trade-offs under different dedup regimes (Appendix L, O).
  - Fairness and safety: Integrate toxicity/PII filters, opt-out mechanisms, and fairness-aware curation into DCLM and measure trade-offs (Appendix S; Appendix U).

- Practical applications and use cases
  - Organizations can use DCLM’s pipeline to construct compute-efficient training sets tailored to their constraints, with confidence that small-scale experiments will likely transfer (Figure 3).
  - The released `DCLM-BASELINE` and models provide strong open baselines for general-purpose LMs, and the tooling (extraction, dedup, filtering, evaluation) can be adopted as production-quality data infrastructure (Appendix D; Section 5).

In short, DCLM reframes LM progress as a data problem with reproducible levers. The central practical lesson is that careful extraction, scalable deduplication, and a simple—but well-chosen—model-based filter can unlock large accuracy gains at far lower compute (Tables 3–5, 8; Figure 1).
