# DataComp‑LM: In search of the next generation of training sets for language models

**ArXiv:** [2406.11794](https://arxiv.org/abs/2406.11794)
**Authors:** Jeffrey Li, Alex Fang, Georgios Smyrnis, Maor Ivgi, Matt Jordan, Samir Gadre, Hritik Bansal, Etash Guha, Sedrick Keh, Kushal Arora, Saurabh Garg, Rui Xin, Niklas Muennighoff, Reinhard Heckel, Jean Mercat, Mayee Chen, Suchin Gururangan, Mitchell Wortsman, Alon Albalak, Yonatan Bitton, Marianna Nezhurina, Amro Abbas, Cheng‑Yu Hsieh, Dhruba Ghosh, Josh Gardner, Maciej Kilian, Hanlin Zhang, Rulin Shao, Sarah Pratt, Sunny Sanyal, Gabriel Ilharco, Giannis Daras, Kalyani Marathe, Aaron Gokaslan, Jieyu Zhang, Khyathi Chandu, Thao Nguyen, Igor Vasiljevic, Sham Kakade, Shuran Song, Sujay Sanghavi, Fartash Faghri, Sewoong Oh, Luke Zettlemoyer, Kyle Lo, Alaaeldin El‑Nouby, Hadi Pouransari, Alexander Toshev, Stephanie Wang, Dirk Groeneveld, Luca Soldaini, Pang Wei Koh, Jenia Jitsev, Thomas Kollar, Alexandros G. Dimakis, Yair Carmon, Achal Dave, Ludwig Schmidt, Vaishaal Shankar
**Institutions:** 

## 🎯 Pitch

DataComp-LM (DCLM) introduces a revolutionary, controlled benchmark for designing pre-training datasets that achieves top-tier performance using less compute. By isolating dataset effects and employing scalable, model-based filtering, DCLM sets a new standard in efficient language model training, enabling broader accessibility and innovation in LLM development.

---

## 1. Executive Summary
DataComp-LM (DCLM) introduces a controlled, reproducible benchmark to study how pre‑training datasets for large language models should be built. It supplies a 240‑trillion‑token web corpus (from Common Crawl), fixed training/evaluation recipes across five compute scales, and shows—through 416 experiments—that simple, scalable model‑based filtering yields a state‑of‑the‑art open dataset (DCLM‑BASELINE) that trains 7B models competitively with far less compute (Figure 1, Table 8).

## 2. Context and Motivation
- Problem and gap
  - Modern LLMs hinge on enormous training sets, but which curation strategies (filtering, deduplication, mixing sources) actually produce better models remains unclear because past comparisons confound data with model architecture, training hyperparameters, or compute budgets (Section 1).
  - Many strong open‑weights models (e.g., Llama, Mistral, Gemma) do not release their training datasets or provide only coarse descriptions, hindering progress on data‑centric methods (Section 1).

- Importance
  - Practical: Better datasets reduce training cost while improving downstream capability; this is crucial as training costs escalate.
  - Scientific: Isolating the effect of data disentangles one of the most important drivers in LLM performance from other variables.

- Prior approaches and shortcomings
  - Heuristic cleaning, language detection, basic quality filtering, deduplication (e.g., RefinedWeb, C4, RedPajama, Dolma) exist, but were evaluated with differing compute and training setups, making fair comparison hard (Sections 2, 4.1).
  - Mixing “high‑quality” sources (Wikipedia, books, code) is standard, but whether mixing helps once web data is very well filtered is under‑tested (Section 4.5).

- Positioning
  - DCLM provides a testbed where dataset interventions are the experimental variable and training/evaluation is held fixed. It spans five scales (from 412M to 7B parameters; Table 1), and two tracks (filtering from a public pool vs. mixing in external sources; Section 3.3). The benchmark is released with large pools, tools, recipes, and leaderboards (Section 3, Appendix D).

## 3. Technical Approach
DCLM is both a benchmark and a set of strong baselines that culminate in a new open dataset (DCLM‑BASELINE). The pipeline can be understood step‑by‑step.

1) Building the public data pool (DCLM‑POOL)
- Source: All Common Crawl WARCs prior to 2023 (to avoid recent synthetic web content), re‑extracted with `resiliparse` (a fast HTML text extractor) rather than using Common Crawl’s WET text (Section 3.1).
- Scale: ~200B documents, 370 TB gzipped, totaling 240T GPT‑NeoX tokens (Section 3.1).
- Decontamination tooling: DCLM ships code to measure or remove overlaps with evaluation sets; the pool itself is not pre‑decontaminated (Section 3.1, Section 4.6).

2) Competition scales and fixed training recipes
- Five compute scales (Table 1) define: number of parameters N, train tokens D (set as `20 × N × Chinchilla multiplier`), FLOPs (≈6ND), and pool subset sizes. Examples:
  - `400M‑1x`: 412M parameters, 8.2B tokens, 469B‑token pool.
  - `7B‑2x`: 6.9B parameters, 276B tokens, 15.7T‑token pool.
- Training recipe: standard decoder‑only Transformer in the OpenLM framework, with scale‑specific hyperparameters kept fixed to isolate dataset effects (Section 3.4; Appendix F for architecture details like qk‑LayerNorm and SwiGLU).

3) Two benchmark tracks (Section 3.3)
- Filtering track: Start from the scale‑specific subset of DCLM‑POOL and produce a dataset by filtering/processing only that pool.
- Mixing track: Combine pool data with any other sources (e.g., Wikipedia, Stack Exchange), within disclosure rules.

4) Evaluation suite and metrics (Section 3.5; Appendix G)
- 53 zero/few‑shot tasks spanning knowledge, QA, reasoning; implemented via LLM‑Foundry.
- Three metrics:
  - `MMLU 5‑shot accuracy` (popular capability measure).
  - `CORE centered accuracy`: average, over 22 low‑variance tasks, of per‑task accuracies linearly rescaled so that 0 = random guessing and 1 = perfect (stable even for small models).
  - `EXTENDED centered accuracy`: same centering but averaged across all 53 tasks.

5) Designing DCLM‑BASELINE through empirical choices (Section 4; Figure 4)
- Text extraction: Compare `resiliparse`, `trafilatura`, WET; tight extractors (`resiliparse` or `trafilatura`) substantially outperform WET downstream (Table 3).
- Deduplication: Evaluate MinHash + suffix arrays vs. a scalable `Bloom filter` approach (BFF) that handles near‑duplicates at document/paragraph levels; performance is comparable, BFF scales better >10 TB (Section 4.3; Tables 18–19; Appendix L).
- Model‑based filtering: Run many filters and find that a simple `fastText` bigram classifier trained with carefully chosen “positive” examples is best (Sections 4.4; Tables 4–5).
- Mixing: Test whether adding curated non‑CC sources helps; find improvements only when the CC baseline is weak—mixing can harm once CC data is strongly filtered (Table 6).
- Decontamination checks: Removing detected MMLU/HellaSwag overlaps from training does not reduce performance, suggesting contamination is not driving gains (Table 7; Appendix O).

How the key parts work

- `fastText` classifier (Section 4.4; Appendix J)
  - Train a linear classifier over word and bigram features to score documents by “quality.” Positives: instruction‑style datasets—OpenHermes‑2.5 and high‑karma ELI5 answers; negatives: random web pages from a RefinedWeb‑like pool.
  - Filter by keeping only the top x% scoring documents (best found at 10%, Table 5).
  - Intuition: instruction‑style positives are diverse, well‑structured, and edited for clarity; their statistical signature helps distinguish broadly useful text while avoiding overfitting to narrow domains like Wikipedia.

- `Bloom filter` near‑deduplication (Section 4.3; Appendix L)
  - A Bloom filter is a memory‑efficient probabilistic set that supports “have we seen this n‑gram?” with no false negatives and low false positives. DCLM extends AI2’s BFF to:
    - Tokenize and split each page into paragraphs; compute n‑grams (e.g., 13‑token n‑grams).
    - If a paragraph has a high fraction of already‑seen n‑grams (threshold ≈ 0.8), drop it; if the whole document exceeds threshold, drop the document.
  - Benefit: scales to multi‑terabyte corpora, approximating the effect of MinHash + suffix arrays at a fraction of the cost (Tables 18–19).

- `Centered accuracy` metric (Section 3.5; Appendix G)
  - For each task, raw accuracy is transformed so random guessing maps to 0 and perfect to 1; averaging then expresses cross‑task progress fairly even when task entropies differ and small models are noisy.

## 4. Key Insights and Innovations
1) A controlled, multi‑scale data benchmark where ranking is stable across scales
- Novelty: DCLM fixes models, hyperparameters, and compute; only data varies. It shows dataset rankings at small scales predict large‑scale performance (Figure 3): Pearson r = 0.838 (400M‑1x), 0.956 (1B‑1x), 0.982 (3B‑1x) vs. 7B‑1x.
- Significance: Enables low‑cost iteration on data curation with high confidence it will transfer.

2) Model‑based filtering beats heuristics, and the best filter is surprisingly simple
- Finding: `fastText` with bigrams, trained on OH‑2.5 + ELI5 positives, outperforms:
  - Perplexity pruning, top‑k logits scoring, PageRank, semantic deduplication, embedding classifiers, and LLM‑judged quality (“AskLLM”), at the 1B‑1x scale (Table 4).
- Design detail that matters: the positive set and the threshold (10% best) strongly affect results (Table 5).
- Significance: Points to a practical, reproducible, and inexpensive recipe for large‑scale filtering.

3) Text extraction quality has large downstream impact
- Result: After applying the same RefinedWeb‑like heuristics, `resiliparse` or `trafilatura` yields +2.5 to +3.8 CORE points over WET extraction at 1B‑1x (Table 3).
- Significance: Early pipeline choices (HTML extraction) are as important as later filters.

4) Deduplication that scales: Bloom filter near‑dedup matches MinHash + suffix arrays
- At 7B‑2x, BFF (min n‑gram 13) achieves CORE 45.3 and MMLU 44.3 vs. MinHash+SA’s 45.5 and 44.4 (Table 19), while being easier to scale beyond 10 TB (Section 4.3).
- Significance: A clear path to practical near‑dedup in trillion‑token settings.

5) “High‑quality data mixing” is not universally beneficial
- Mixing Wikipedia/Books/ArXiv/GitHub improves weak CC subsets but hurts a strong, filtered CC dataset: DCLM‑BASELINE’s CORE drops from 31.1 to 29.9 when mixed (Table 6).
- Significance: With sufficiently strong filtering, CC‑only can be best—challenging a common assumption.

6) Human quality judgments do not align with what makes better pretraining data
- ROC–AUC of various filters on human‑labeled “good/bad” pages does not correlate with gains on CORE, SQuAD, StrategyQA (Appendix N; Figure 9). The LLM‑grader AskLLM achieves higher agreement with annotators but worse downstream performance than fastText (Table 4; Appendix N).
- Significance: “Humanly good content” is not the same as “content that trains LLMs well.”

## 5. Experimental Analysis
- Evaluation setup
  - Datasets/metrics: 53 tasks; key metric is CORE centered accuracy (22 stable tasks), plus MMLU 5‑shot and EXTENDED centered accuracy (53 tasks) (Section 3.5; Appendix G).
  - Baselines: C4, RefinedWeb, RedPajama, Dolma‑V1, FineWeb‑Edu, OLMo, LLM360/Amber, MAP‑Neo, and closed‑data models for context (Figure 1; Tables 2, 8, 33).
  - Fixed training: OpenLM decoder‑only Transformers with scale‑specific hyperparameters (Section 3.4; Appendix F).

- Main quantitative results
  - Text extraction (1B‑1x, Table 3):
    > `resiliparse` CORE 24.1 vs. `trafilatura` 24.5 vs. WET 20.7.
  - Model‑based filtering methods (1B‑1x, Table 4):
    > `fastText OH‑2.5+ELI5` CORE 30.2 beats `perplexity` 29.0, `top‑k logits` 29.2, `AskLLM` 28.6, `BGE‑linear` 27.2, `PageRank` 26.1.
  - fastText ablations (7B‑1x, Table 5, 14):
    > Positives matter: `OH‑2.5+ELI5` CORE 41.0 vs. `Wikipedia` 35.7, `OpenWebText2` 34.7, `GPT‑3 Approx` 37.5.  
    > Threshold matters: top‑10% CORE 41.0 > top‑15% 39.8 > top‑20% 38.7.  
    > Bigrams help: unigrams‑only CORE 40.0 vs. bigrams+unigrams 41.0 (Table 14).
  - Dedup at scale (7B‑1x and 7B‑2x; Tables 18–19):
    > BFF (min n‑gram 13) ≈ MinHash+SA: at 7B‑2x MMLU 44.3/CORE 45.3 vs. 44.4/45.5.
  - Mixing (1B‑1x, Table 6):
    > Mixing helps weaker CC (C4 +2.2 CORE), hurts DCLM‑BASELINE (−1.2 CORE).
  - Decontamination (7B‑2x, Table 7):
    > Removing MMLU/HellaSwag overlaps: MMLU 51.8 → 52.7, HellaSwag 77.9 → 78.4 (no drop).  
    > Broader contamination analysis shows similar or better cleanliness vs. FineWeb‑Edu and Dolma‑V1.7 (Appendix O, Table 25; Figures 11–12).

- Final large‑scale model (Table 8; Section 5)
  - Train 7B on 2.6T tokens: 70% DCLM‑BASELINE (tighter filter in cool‑down), 30% math/code (StarCoder, ProofPile2). With fixed recipes, achieves:
    > CORE 57.1, MMLU 63.7, EXTENDED 45.4.
  - Comparisons:
    - Beats open‑data peers (e.g., MAP‑Neo‑7B: CORE 50.2, MMLU 57.1, EXTENDED 40.4).
    - Comparable to closed‑data `Mistral‑7B‑v0.3` (CORE 57.0, MMLU 62.7, EXTENDED 45.1) and near `Llama‑3 8B` (57.6/66.2/46.3), while using ≈6.6× less training tokens than Llama‑3 8B (2.6T vs. 15T; Section 1, Table 8).
  - Instruction tuning on public data preserves strong base performance and yields competitive AlpacaEval2.0 win‑rates (Appendix P, Table 26).

- Robustness and supporting evidence
  - Rankings persist across architectures (Gemma‑like and Mamba‑like) at 1B scale (Appendix I; Figure 6).
  - Rankings stable across hyperparameters; gains from better datasets stack with gains from better training settings (Appendix H; Tables 12–13).
  - LightEval vs LLM‑Foundry MMLU correlation study indicates evaluation conclusions are consistent, with small‑scale sensitivity differences (Appendix G.2; Figure 5).

- Do the experiments support the claims?
  - Yes. The paper isolates data effects by fixing training/evaluation across 416 runs, uses multiple scales, ablates each major pipeline stage (extraction, dedup, filtering, mixing), and cross‑checks with contamination and alternative eval setups. The final 7B result substantiates that the curated dataset competes with much more compute.

## 6. Limitations and Trade-offs
- Scope and compute
  - Models up to 7B parameters; training beyond 7B and broader compute sweeps remain future work (Section 6).
  - Run‑to‑run variance not exhaustively analyzed at the largest scales (Section 6).

- Tokenizer and domains
  - Primarily uses GPT‑NeoX tokenizer; other tokenizers may change multilingual/math behavior (Section 6).
  - Focus is language understanding; math/code abilities require additional domain‑specific pretraining (Section 6, Table 8).

- Data assumptions
  - DCLM‑POOL draws entirely from pre‑2023 Common Crawl; while very large, it may not represent all domains and languages and can include PII or toxic content (Appendix U Datasheet; Section 3.1). Heuristic and model‑based filters reduce but do not eliminate such content.

- Design dependence
  - The fastText approach depends on the choice of positive set and threshold (Table 5): shifting either can reduce gains.
  - Bloom filter hyperparameters (n‑gram size, thresholds, sharding) affect removal rates and document statistics (Appendix L).

- Evaluation breadth
  - While 53 tasks are broad, they may still under‑represent some specialized capabilities (e.g., complex tool use, long‑horizon reasoning). The paper partially addresses long‑context in Appendix Q.2 with continual learning to 8k context.

## 7. Implications and Future Directions
- How this work changes the field
  - Establishes a common, scalable yardstick for data curation—similar to how model benchmarks shaped architectures—making data a first‑class, testable design space.
  - Demonstrates that careful filtering of raw web text can rival or surpass mixed “high‑quality” corpora, reorienting effort toward scalable, data‑centric methods (Figures 1, 4; Tables 4–6, 8).

- Follow‑up research enabled/suggested
  - Better filtering networks: beyond fastText, explore lightweight neural filters optimized for recall/precision trade‑offs at terascale.
  - Targeted data selection: conditional filters for domains (math, code, safety, multilingual), curriculum schedules, and dynamic mixture optimization (Appendix B cites DoReMi, DoGE, MixMax lines of work).
  - Fairness/safety/multilingual tracks: extend DCLM with explicit bias, toxicity, and multilingual objectives (Section 6; Appendix S).
  - Efficient decontamination: task‑aware overlap removal at scale, and robust evaluation under paraphrase/rephrasing (Appendix O).

- Practical applications
  - Organizations can use the DCLM pipeline to build cost‑efficient training sets, particularly when compute is limited.
  - The open DCLM‑BASELINE dataset provides a strong default for training 7B‑scale models or as a base to mix with domain‑specific data (e.g., legal, biomedical, code).

---

Selected references to paper artifacts
- Pools/Workflow: Section 3; Figure 2.
- Scales: Table 1; Figure 3.
- Extraction: Section 4.2; Table 3; Appendix K.
- Dedup: Section 4.3; Tables 18–19; Appendix L.
- Filtering: Section 4.4; Tables 4–5, 14; Appendix J.
- Mixing: Section 4.5; Table 6; Appendix M.
- Decontamination: Section 4.6; Table 7; Appendix O.
- Final 7B model: Section 5; Table 8; Appendices Q, P.
- Stability across settings: Appendix H (Tables 12–13), Appendix I (Figure 6).
- Human judgment: Appendix N (Figure 9).
