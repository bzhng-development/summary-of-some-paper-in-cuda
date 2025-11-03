# The Common Pile v0.1: An 8TB Dataset of Public Domain and Openly Licensed Text

**ArXiv:** [2506.05209](https://arxiv.org/abs/2506.05209)
**Authors:** Nikhil Kandpal, Brian Lester, Colin Raffel, Sebastian Majstorovic, Stella Biderman, Baber Abbasi, Luca Soldaini, Enrico Shippole, A. Feder Cooper, Aviya Skowron, Shayne Longpre, Lintang Sutawika, Alon Albalak, Zhenlin Xu, Guilherme Penedo, Loubna Ben Allal, Elie Bakouch, John David Pressman, Honglu Fan, Dashiell Stander, Guangyu Song, Aaron Gokaslan, John Kirchenbauer, Tom Goldstein, Brian R. Bartoldson, Bhavya Kailkhura, Tyler Murray
**Institutions:** EleutherAI, University of Toronto, Hugging Face

## 🎯 Pitch

The paper introduces 'Common Pile v0.1,' an 8TB legally shareable corpus of public‑domain and openly licensed text, demonstrating that quality language model pretraining can be achieved without relying on copyrighted sources. This innovation addresses significant legal and ethical challenges in AI development, paving the way for transparent, reproducible, and compliant machine learning research and applications.

---

## 1. Executive Summary (2-3 sentences)
This paper introduces `Common Pile v0.1`, an 8TB corpus of public‑domain and openly licensed text spanning 30 sources, and validates its usefulness by training two 7B‑parameter language models (`Comma v0.1-1T` and `Comma v0.1-2T`). The models achieve performance competitive with similarly budgeted models trained on unlicensed web data, showing that high‑quality LLM pretraining is feasible without relying on copyrighted sources (Figures 3–4; Tables 10–11).

## 2. Context and Motivation
- Problem addressed
  - Can performant LLMs be trained using only public‑domain and openly licensed text, with fully shareable data and reproducible pipelines? This question arises because most modern pretraining relies on large amounts of unlicensed web data (Section 1).
- Why it matters
  - Legal and ethical stakes: web text is often copyrighted; compensating rights holders would cost billions and has triggered lawsuits and takedowns (Section 1, refs. [24, 40, 57, 96, 159, 191]). Consent concerns grew as websites began blocking AI crawlers in mid‑2023 (Section 1, ref. [107]).
  - Scientific stakes: open, shareable pretraining datasets enable research on learning dynamics, auditing, and memorization, which is limited when data cannot be redistributed (Section 1; refs. [18, 47, 50, 57, 76, 83, 128, 145]).
- Prior approaches and their gaps
  - Prior “open” corpora exist but are small, license‑ambiguous, or narrow:
    - `OLC`: similar scope but ~0.85 TB and includes sources like Hacker News which has no open license (Section 2.2).
    - `Common Corpus`: size comparable but less English text and includes OpenAlex, which is known to mislabel licensing (Section 2.2).
    - `KL3M`: strictly excludes CC BY‑SA, therefore mostly government text and smaller (3 TB) with less diversity (Section 2.2).
  - Existing high‑quality web datasets (e.g., FineWeb, OSCAR) are licensed at the collection level (e.g., ODC‑By) and include copyrighted documents, so the underlying text is not openly licensed (Section 2.1, “Use of collection licenses”).
- Positioning
  - The paper defines “openly licensed” per the Open Knowledge Foundation’s Open Definition 2.1; accepts CC BY, CC BY‑SA, CC0, Blue Oak–approved software licenses; excludes CC‑NC and CC‑ND (Section 2; Appendix C).
  - It sets strict provenance standards to avoid “license laundering” (re‑posting under incorrect terms) and avoids synthetic LLM‑generated data whose licensing is unsettled (Section 2.1).

## 3. Technical Approach
This section explains how the dataset is built, cleaned, mixed for training, and how models are trained and evaluated.

- Data sourcing under explicit licensing constraints (Sections 2, 2.1; Appendix B)
  - 30 sources spanning: research papers, government/legal text, wikis, public‑domain books, forums, openly licensed web pages, code, open educational resources, and CC BY YouTube transcripts (Figure 1; Appendix B).
  - Due‑diligence choices:
    - Require licensing provided by the rights holder; exclude sources with unreliable metadata (e.g., OpenAlex, YouTube “Commons” at large, some Kaggle sets) to mitigate license laundering (Section 2.1 “License due diligence”).
    - Treat collection‑level licenses as non‑sufficient for underlying documents (Section 2.1).
    - Avoid LLM‑generated synthetic datasets given unsettled licensing status (Section 2.1).

- Dataset construction for the “Comma” training mixture (Sections 4.1–4.2; Tables 5–7; Appendix J–K)
  - Preprocessing and filtering (Section 4.1; Table 5):
    - Language ID via `FastText` to retain English; quality filtering for web pages with a DataComp‑LM classifier using a low threshold for noise removal.
    - OCR error removal via a likelihood filter on a unigram model built from the Trillion Word Corpus; toxicity filtering using two `FastText` classifiers trained on Jigsaw Toxic Comment data.
    - PII redaction for emails, phone numbers, and IP addresses replaced with `<EMAIL_ADDRESS>`, `<PHONE_NUMBER>`, `<IP_ADDRESS>`.
    - Source‑specific regex cleanup (e.g., boilerplate, license headers).
  - Global deduplication (Section 4.1):
    - Cross‑source, document‑level near‑duplicate removal using a Bloom‑filter approach; duplicates are those sharing >90% of 20‑grams. A Bloom filter is a space‑efficient data structure that quickly tests set membership with low memory, suitable for de‑dup at web scale.
  - Code curation (Section 4.1):
    - Start from the openly licensed subset of `The Stack v2` (license detection by BigCode/Software Heritage).
    - Apply RedPajama V1 heuristics (e.g., max line length, character ratios), restrict to a language set (Python, C/C++, SQL, Java, PHP, Rust, JS/TS, Go, Ruby, Markdown, C#, Swift, shell).
    - Use language‑specific quality classifiers to keep well‑documented, educational code; HTML files extracted with Trafilatura and passed through the same text filters.
  - Open‑web text (`CCCC`) with license verification (Appendix G; Section B.10):
    - Scan 52 Common Crawl snapshots using CC regex as a first pass, then manually verify the top 1000 domains by volume; retain only 537 domains where the CC license covers all text, not just embedded media.
    - Extract main content and remove boilerplate with Resiliparse; apply exact and near‑duplicate removal and additional heuristics (C4 and Gopher rules).
  - Resulting raw vs. filtered sizes (Table 6):
    - From 7.56 TB raw text to 1.84 TB after filtering and deduplication across sources.

- Data mixing for pretraining (Section 4.2; Table 7)
  - Rationale: sources vary in quality and domain match; patents (USPTO) are huge but stylistically narrow, so size alone is a poor proxy for quality.
  - Procedure:
    - Train per‑source 1.7B models for 28B tokens (Section 4.2; 4.3) and use their performance to set heuristic mixing weights.
    - Target at most 6 passes over each source during a 1T‑token run; assume small sources are high quality and also repeat up to 6 times.
    - Attempted `MixMin` (an automatic mixture optimizer) but it did not beat the heuristics (Section 4.2).
  - The resulting 1T mixture (“`Comma dataset`”) is detailed in Table 7, including each source’s repetition count and token share (e.g., `peS2o` ≈27.4%, `StackExchange` ≈13.5%, `Stack v2` ≈13.0%, `CCCC` ≈8.7%).

- Model training and tokenizer (Section 4.4)
  - Tokenizer: 64k‑vocabulary BPE trained on a 600GB sample of the Comma dataset, using Llama‑3.2‑style splitting regex and byte‑level preprocessor (Section 4.4 “Tokenization”). BPE (byte‑pair encoding) merges frequent byte sequences to form subword tokens that compress well across domains.
  - Architecture and setup: Llama‑style 7B decoder‑only Transformer implemented in `lingua` (Section 4.4 “Training setup”).
    - `Comma v0.1-1T`: effective batch 512×4096 tokens, AdamW, weight decay 0.2; cosine schedule with warmup; two‑stage curriculum finishing with a “cool‑down” phase on a high‑quality subset (Table 8), linearly decaying LR to 0, then average 10 checkpoints (Section 4.4).
    - `Comma v0.1-2T`: same mixture repeated to reach 2T tokens; increase batch to 2048×4096; same cool‑down and averaging (Section 4.4).
    - Note: some sources are repeated up to 16× at 2T (Section 4.4 “Results”), which is known to risk diminishing returns.

- Evaluation protocols (Sections 4.3–4.4; Figures 2–4; Tables 9–11)
  - Controlled dataset quality study (1.7B models on 28B tokens): evaluate “early‑signal” tasks—`ARC`, `MMLU`, `HellaSwag`, `OpenBookQA`, `CommonSenseQA`, `PIQA`, `SIQA`—to compare datasets on equal footing (Section 4.3; Figure 2; Table 9).
  - Large‑scale model benchmarks (`Comma v0.1-1T` and `-2T`): `ARC-C/E`, `MMLU` (5‑shot), `BoolQ`, `HellaSwag`, `OBQA`, `CSQA`, `PIQA`, `SIQA`, plus code (`HumanEval`, `MBPP` pass@10), using OLMES evaluation formats (Section 4.4 “Evaluation”; Figures 3–4; Tables 10–11).

Definitions introduced only where uncommon:
- `License laundering`: redistribution of copyrighted work under an incorrect/unauthorized open license (Section 2.1).
- `PII` (personally identifiable information): data that can identify individuals; here, emails, phone numbers, IPs are redacted (Section 4.1).
- `Bloom filter`: probabilistic data structure used for fast membership tests, enabling memory‑efficient near‑deduplication (Section 4.1).

## 4. Key Insights and Innovations
- A legally shareable, multi‑domain, open‑license corpus at scale, with documented provenance
  - What’s new: `Common Pile v0.1` aggregates 8TB across 30 sources with per‑source licensing checks and manual verification for web content (Figure 1; Section 2.1; Appendix G).
  - Why it matters: It directly addresses reproducibility and legal barriers that prevent sharing of most pretraining datasets (Section 1, Section 2.1).
- Rigorous licensing stance and anti‑laundering practices integrated into curation
  - What’s new: Excludes CC‑NC/CC‑ND; filters out sources with unreliable license metadata; avoids synthetic LLM outputs; manually audits top web domains for CC coverage of textual content (Sections 2, 2.1; Appendix C, G).
  - Why it matters: This is a principled blueprint for curating “copyright‑clean” corpora at web scale.
- Evidence‑driven data mixing for pretraining with repetition caps
  - What’s new: Build a 1T mixture by training per‑source proxy models, then assign heuristic weights to up‑/down‑weight sources; cap repetitions to 6× at 1T (Section 4.2; Table 7).
  - Why it matters: Source size does not equal quality; mixing improves efficiency and performance (Section 4.2). The team also shows that off‑the‑shelf mixture optimization (MixMin) did not beat their informed heuristics in this setting.
- End‑to‑end validation: release of data, code, tokenizer, mixtures, and checkpoints
  - What’s new: Beyond the corpus, the paper releases the `Comma v0.1` training recipe, mixtures (Tables 7–8), and checkpoints, enabling full reproducibility (Section 5 Conclusion).
  - Why it matters: Transparent, open artifacts are rare in pretraining and enable community study of data effects.

## 5. Experimental Analysis
- Evaluation design
  - Dataset‑quality comparison (Section 4.3; Figure 2; Table 9)
    - Train identical 1.7B models for 28B tokens on different corpora: `Common Pile` (as the “Comma dataset”), `OLC`, `Common Corpus`, `KL3M`, `The Pile` (unlicensed blend), `OSCAR` and `FineWeb` (modern web curation).
    - Metrics: zero‑shot accuracy on `ARC`, `MMLU`, `HellaSwag`, `OBQA`, `CSQA`, `PIQA`, `SIQA` (Winogrande omitted to avoid data leakage from DPI content; Section 4.3).
  - Large‑scale model evaluation (Section 4.4; Figures 3–4; Tables 10–11)
    - Compare `Comma v0.1-1T` to 7B/1T models trained on unlicensed data (LLaMA‑1 7B, MPT‑7B, RPJ‑INCITE‑7B, StableLM‑7B, OpenLLaMA‑7B).
    - Compare `Comma v0.1-2T` to 7B/2T models (OLMo‑Twin, LLaMA‑2 7B, DeepSeekLLM).
    - Also show a higher‑budget reference point: `Qwen3 8B` trained on 36T tokens (Figure 3–4).

- Main quantitative findings
  - Dataset‑quality study (Figure 2; Table 9):
    - The `Comma` dataset beats all open‑license baselines (`OLC`, `Common Corpus`, `KL3M`) on every benchmark and surpasses `The Pile` on five of seven tasks.
    - Average accuracies (Table 9):
      - `Comma`: 40.8 average vs. `OLC` 37.3, `Common Corpus` 37.6, `KL3M` 36.2, `The Pile` 39.6.
      - `FineWeb` is highest overall at 43.7, but `Comma` leads on scientific knowledge: it is best on `MMLU` (29.5 vs. 29.1 FineWeb) and ties top on `ARC` (38.0, same as FineWeb).
    - Notable weakness: `HellaSwag`, `PIQA`, `CSQA` are lower than top web datasets—consistent with under‑representation of personal blogs, hobbies, and sports (Section 4.3 citing [188]).
    - DPI (task‑like) data ablation (Table 9): removing DPI has minor effect; the average drops from 40.8 to 40.0. A small drop on `HellaSwag` suggests DPI contains some relevant content.
  - `Comma v0.1-1T` vs compute‑matched 7B/1T models (Figure 3; Table 10):
    - Standout strengths: knowledge (`ARC‑C` 52.8 vs LLaMA 44.5; `MMLU` 42.4 vs MPT 30.2; vs LLaMA 34.8), and coding (`HumanEval` 36.5; `MBPP` 35.5), often the best among 1T peers.
    - Weaker on `HellaSwag` (62.6), where web‑heavy baselines reach mid‑70s.
    - Overall average in Table 10 is top among 1T compute‑matched baselines (54.7), narrowly ahead of OpenLLaMA and MPT averages (≈54–55).
  - `Comma v0.1-2T` vs 7B/2T models (Figure 4; Table 11):
    - Competitive with OLMo‑Twin and LLaMA‑2: strong on `MMLU` (49.8 vs LLaMA‑2 45.8), `ARC‑E` (71.8 vs 69.5), `SIQA` (52.3 vs 50.8), and notably strong on coding (`HumanEval` 44.2; `MBPP` 41.5).
    - Still weaker on `HellaSwag` (64.4) relative to LLaMA‑2 (76.2) and DeepSeekLLM (74.1).
- Ablations and robustness (Appendix O; Table 12)
  - Two additional 1T runs with different batch sizes and a three‑stage curriculum yield averages ~53.5–53.8 vs the main run’s 53.6 (pre‑averaging). Coding sometimes benefits further (Table 12), suggesting moderate robustness to hyperparameter choices and curricula.
- Do results support the claims?
  - Yes for the central claim: training on exclusively open‑license text can match older, compute‑matched unlicensed baselines on many tasks and excel at scientific knowledge and code (Figures 3–4; Tables 10–11).
  - The controlled 1.7B comparison isolates dataset effects and shows `Common Pile`’s relative data quality vs other open corpora (Figure 2; Table 9).
  - Caveat: `Comma v0.1-2T` repeats some sources up to 16× (Section 4.4 “Results”), so its 2T result is not a clean “best‑case” for scaling; authors note diminished returns are expected with heavy repetition.

> “Comma v0.1-1T outperforms budget‑matched baseline models on over half of the benchmarks tested… and is particularly strong at code‑related tasks” (Section 4.4; Figure 3; Table 10).

> “Comma v0.1-2T is competitive with OLMo, Llama 2, and DeepSeekLLM… with especially strong performance on MMLU, SIQA, ARC‑E, and the coding tasks” (Figure 4; Table 11).

## 6. Limitations and Trade-offs
- Residual licensing risk
  - Even with strict standards, license laundering and metadata drift are hard to eliminate completely; rights holders may later change license terms; public‑domain texts may quote copyrighted content (Section 2.1 “Caveats”).
- Coverage gaps that show up as performance trade‑offs
  - The corpus under‑represents casual, blog‑style, hobby, and sports content; this likely depresses commonsense benchmarks such as `HellaSwag` and `PIQA` (Section 4.3, with analysis informed by [188]).
- English‑centric focus
  - Primary emphasis on English (Section 4.1), reducing multilingual generality relative to some modern pretraining corpora.
- Data repetition at scale
  - The 2T run repeats certain sources up to 16× (Section 4.4 “Results”), which prior work shows can cause diminishing returns and memorization risks (cited in 4.2, 4.4).
- Heuristic mixing
  - Mixture weights are based on per‑source proxy training and heuristics; automated methods like `MixMin` did not help here (Section 4.2). There may be headroom in principled, dynamic mixture optimization.
- Benchmark scope
  - Evaluations focus on knowledge/reasoning and code; they do not cover safety, long‑context modeling, multilinguality, or instruction‑following/chat alignment. The paper’s claim is about pretraining viability, not end‑to‑end deployment readiness (Sections 4.3–4.4).
- Compute and resource demands
  - Training 1T–2T‑token 7B models is still costly; while datasets are open, reproducing full‑scale training requires significant compute and storage (implied by Sections 4.4 and the sizes in Table 7).

## 7. Implications and Future Directions
- How this changes the field
  - Demonstrates that enforceably open, auditable corpora can produce competitive general‑purpose LLMs at billions of parameters (Figures 3–4). This lowers legal barriers for organizations that require clean provenance and fosters transparent data science.
  - Provides a reference pipeline—including web verification, cross‑source dedup, and per‑source filtering thresholds (Table 5)—for building copyright‑compliant datasets at scale.
- What follow‑up research it enables
  - Data mixture optimization: systematic methods (e.g., online mixture learning, curriculum search) to replace heuristics (Section 4.2). The paper’s negative result for MixMin in this setting is a useful data point for future work.
  - Scaling without repetition: targeted collection to grow underrepresented domains (blogs, tutorials, sports, hobbies) and languages; measuring the returns of fresh data vs repeats (Sections 4.2, 4.4).
  - Attribution research: methods to connect model outputs to training points to satisfy attribution obligations when applicable (Section 2.1 “Caveats”, refs. [28, 129]).
  - Licensing toolchains: automated, reliable license detection across text and embedded assets (Appendix C); better standards for publishing licenses on the web.
- Practical applications
  - Organizations with strict compliance needs (government, healthcare, education) can pretrain or fine‑tune on the released data, mixtures, and checkpoints without the legal uncertainty of web‑scraped corpora (Section 5 Conclusion).
  - Domain‑focused LLMs for science, government, and education may benefit from `Common Pile`’s strengths (e.g., `peS2o`, PubMed, USGPO, Hansard); code assistants benefit from curated `Stack v2` and `PEPs`.

Overall, `Common Pile v0.1` plus the `Comma v0.1` models supply both a blueprint and concrete evidence that open‑license pretraining is viable at scale, with clear strengths in scientific knowledge and code. The most immediate frontier is broadening domain and language coverage while preserving license certainty, and improving data mixture strategies to further narrow gaps on commonsense benchmarks.
