# OpenCoder: The Open Cookbook for Top-Tier Code Large Language Models

**ArXiv:** [2411.04905](https://arxiv.org/abs/2411.04905)
**Authors:** Siming Huang, Tianhao Cheng, Jason Klein Liu, Jiaran Hao, Liuyihan Song, Yang Xu, J. Yang, J. H. Liu, Chenchen Zhang, Linzheng Chai, Ruifeng Yuan, Zhaoxiang Zhang, Jie Fu, Qian Liu, Ge Zhang, Zili Wang, Yuan Qi, Yinghui Xu, Wei Chu
**Institutions:** The University of Melbourne, Fudan University, Independent Researcher, INF, MAP, Institute of Automation, Chinese Academy of Sciences & Nanjing University

## 🎯 Pitch

OpenCoder introduces a fully open and transparent code large language model, complete with a reproducible training pipeline and an innovative dataset, RefineCode, comprising 960B tokens with extensive filtering rules. This model not only enhances reproducibility and performance in code generation and completion but also sets a new standard for openness in LLM development, enabling the scientific community to conduct controlled experiments and build upon its robust foundation.

---

## 1. Executive Summary
OpenCoder is a fully open, top-tier code large language model (LLM) suite (1.5B and 8B parameters) built with an end-to-end, reproducible “cookbook”: training data, data-processing pipeline, ablations, training protocols, SFT data, and intermediate checkpoints. Its core contribution is a high-quality, transparent pretraining corpus (RefineCode, 960B tokens) plus targeted annealing and two-stage instruction tuning that together yield competitive code performance while revealing which data and training choices matter most (e.g., file-level dedup, high-quality annealing data, two-stage SFT).

## 2. Context and Motivation
- Problem and gap
  - Code LLMs are crucial for software development tasks (generation, completion, reasoning, agents), but high-performing, fully open models with truly reproducible data pipelines are rare. Most strong models keep training data and processing private, limiting scientific study and reproducibility.
  - OpenCoder fills this by releasing not only model weights and inference code but also the full, reproducible pretraining corpus and pipeline, large-scale SFT datasets, ablations, and training details (Table 1).
- Why it matters
  - Real-world impact: Better code assistants and agents depend on code quality, data coverage, and up-to-date APIs. Transparent pipelines allow others to build, audit, and extend models safely and efficiently.
  - Scientific significance: Open access enables controlled experiments on data cleaning, deduplication, distribution shifts, and instruction tuning strategies—key to understanding how code LLMs learn.
- Prior approaches and shortcomings
  - Open models such as CodeLlama, StarCoder2, DeepSeek-Coder, and CodeGemma typically release model weights; some release partial data. But few provide a complete, reproducible training corpus plus full data-processing details and controlled ablations (Table 1).
  - Datasets like The Stack v2 are valuable but include substantial low-quality or redundant content; the paper finds that stronger filtering and dedup are necessary for top-tier training (Sections 2.1, 6.1; Figure 3).
- Positioning
  - OpenCoder positions itself as both a high-performing model and an “open cookbook” that enumerates key ingredients for strong code LLMs: aggressive file-level deduplication, code-optimized heuristic filtering, recall of code-related web text, high-quality annealing data, and a two-stage instruction-tuning strategy (Introduction; Sections 2–4; Table 1; Figure 1).

## 3. Technical Approach
This work is empirical and systems-oriented. It covers data curation (pretraining + annealing), model training, and post-training (instruction tuning), with extensive ablations.

- Pretraining dataset: RefineCode (Section 2.1)
  - What it is: A reproducible, high-quality, 960B-token corpus across 607 programming languages with >130 filtering rules and custom weights (Table 2; Appendix E; Appendix A).
  - How it is built (Figure 2a):
    1) Preprocessing
       - Exclude files >8 MB (mostly non-text); keep only programming-related extensions per GitHub Linguist; retain 607 languages (Section 2.1.1; Appendix E).
    2) Deduplication
       - Exact deduplication: SHA256 on file content; among identical files, keep the latest commit and highest-star repo (Section 2.1.1).
       - Fuzzy deduplication: MinHash with 2048 hash functions and LSH (16 bands × 128 rows) on 5-grams; again retain the latest/highest-star version; removes a further 6% of files (Section 2.1.1).
       - Definitions: `file-level deduplication` removes duplicate files irrespective of repository; `repository-level deduplication` removes duplicates only within each repo. The paper shows file-level is better (Section 6.1).
    3) Transformation (before filtering)
       - Copyright removal (common repetitive headers) and PII reduction (passwords, emails, IPs replaced with placeholders) via regex (Section 2.1.1).
    4) Heuristic filtering (Section 2.1.1; Appendix A)
       - Goal: remove low-information code (e.g., pure hex blobs, trivial snippets) while preserving diversity.
       - Three rule classes: Natural-language rules (e.g., file size, line counts), General code rules (e.g., variable and function stats, proportion of asserts), and Language-specific rules for 8 languages (e.g., frequency of Python `pass`, C `goto`) (Section 2.1.1; Figure 3; Appendix A.2).
       - Threshold setting process includes coarse-to-fine tuning and a perplexity-based inspection step to vet extremes (Appendix A.1).
    5) Data sampling
       - Downsample high-volume but less informative languages (HTML from 474GB→64GB; Java 449GB→200GB) to balance distributions (Section 2.1.1).
    6) Jupyter and text data handling
       - Convert notebooks into a structured format (markdown/code/results), discard redundant script format (Appendix C.3).
       - From GitHub text files, recall code-related content via filename heuristics and a FastText classifier (Appendix C.2).
  - Code-related web data recall (Figure 2b; Section 2.1.2)
    - Problem: The web has useful code-related text (explanations, Q&A) but is noisy.
    - Method: Start with 500k “code-like” seed pages from Common Crawl (via Autonomous Data Selection) to train a FastText classifier; recall pages from Common Crawl; discover domains with >10% code pages; manually annotate code-heavy URL patterns; iterate to grow seeds and improve recall.
    - Result: ~220GB recalled from Common Crawl; plus similar recall pass over FineWeb, SkyPile, and AutoMathText web parts to obtain in total ~330GB extra; combined with GitHub text classified as code-related, +178GB (Section 2.1.2; Table 2; Appendix C.1).
  - Why this pipeline: The PCA visualization (Figure 3) shows RefineCode’s embeddings are tighter with fewer outliers than The Stack v2; outliers often correspond to low-quality patterns (pure hex, comments-only, ultra-short code). This supports the need for aggressive filtering and dedup.

- Annealing data and strategy (Section 2.2; Table 3)
  - Definition: An “annealing phase” is a short training continuation with a rapidly decaying learning rate on very high-quality data to sharpen capabilities (MiniCPM-style).
  - Mixture (100B tokens): 84% original distribution from RefineCode (to avoid catastrophic forgetting), plus two high-quality additions:
    - `Algorithmic Corpus` (12.44B tokens): self-contained algorithmic code (e.g., “leetcode”, “def solution”) that aligns with interactive, function-level tasks.
    - `Synthetic Data`:
      - Verified Code Snippets (2.71B): generate functions plus test cases with a strong LLM; keep only those that pass tests; multilingual (Section 2.2).
      - Code Textbooks (0.91B): use Qwen2-72B-Instruct on the `hqcode` dataset to produce educational analyses that explain abstract code knowledge (Section 2.2).
  - Why: Section 6.2 (Figure 9) shows that removing the high-quality component (Algorithmic + Synthetic) substantially degrades annealing gains on HumanEval/MBPP.

- Model architecture and pretraining (Sections 3.1–3.2; Table 4)
  - Two sizes:
    - `OpenCoder-1.5B`: 24 layers, d=2240, 14 heads/14 KV heads, RoPE, seq len 4096.
    - `OpenCoder-8B`: 32 layers, d=4096, 32 heads/8 KV heads, RoPE with θ=500k, seq len 8192.
  - Training schedule: “WSD” (warmup–steady–decay) used in MiniCPM (Section 3.2). Peak LR 3e-4; decays to 1e-5 in annealing.
  - Compute:
    - 1.5B: 2T tokens (4 epochs) + 100B annealing; Megatron-LM, 256×H800, 109.5h ≈ 28k GPUh (Section 3.2).
    - 8B: 2.5T tokens (3.5 epochs) + 100B decay stage; 512×H100, 187.5h = 96k GPUh; first 130k steps at seq len 4096 (Section 3.2).

- Post-training: two-stage instruction tuning (Sections 4.1–4.4; Figure 5; Table 5)
  - Stage 1 (breadth): diverse, large-scale instruction data to build general problem-solving—including filtered Infinity-Instruct (1.0M), Large-scale Diverse-Instruct (2.3M, synthesized from web texts via prompt templates and validation), and RealUser-Instruct (0.7M; cleaned and augmented real chat logs) (Sections 4.1, 4.2; Table 5).
  - Stage 2 (depth in code): high-quality code-specific data—Educational-Instruct (110k, test-verified tasks), Package-Instruct (110k, synthesized from up-to-date Python library docs via PyDoc), Evol-Instruct (111k), and McEval-Instruct (36k) (Sections 4.1–4.2; Table 5).
  - Training details: Stage 1—1 epoch, batch 4096, LR 2e-5; Stage 2—3 epochs, batch 512, LR 5e-5; both use cosine LR with 100 warmup steps (Section 4.3).
  - Decontamination: remove entries overlapping with HumanEval/MBPP via entry points; 10-gram dedup against eval sets (Section 4.4).

- Design choices and rationale
  - File-level over repo-level dedup: preserves diversity while removing heavy duplication across forks and copies (Section 6.1; Table 9; Figure 8).
  - High-quality annealing: later-stage data quality beats quantity (Section 6.2; Figure 9).
  - Against star-based filtering: filtering GitHub files by stars≥5 lowers training loss but hurts performance due to reduced diversity (Section 6.3; Figures 10–11).
  - Two-stage SFT: breadth-first followed by depth-first improves both benchmarks and realistic prompts (Section 6.4; Table 10).

## 4. Key Insights and Innovations
- A fully reproducible, high-quality code corpus with code-specific heuristics and file-level dedup
  - What’s new: 960B-token RefineCode with >130 filters including language-specific rules for 8 major languages, early exact dedup, and MinHash-LSH fuzzy dedup at the file level (Sections 2.1, 6.1; Table 2; Figure 3; Appendix A).
  - Why it matters: Demonstrably better training efficiency and cleaner distributions than The Stack v2 (Figure 3; “RefineCode significantly improves training efficiency,” Section 2.1.3, referencing Figure 1).
- Demonstrated superiority of file-level deduplication on code
  - Evidence: On 485M Python files, repo-level keeps ~99.47B tokens vs file-level ~32.74B; yet the file-level dataset trains better (HumanEval/MBPP curves in Figure 8; Table 9). A large fraction of repo-level data is still exact duplicates (52B tokens) (Section 6.1).
  - Significance: Establishes a practical, CPU-efficient recipe—exact dedup then file-level fuzzy dedup—to build unbiased code corpora at scale.
- High-quality annealing data improves downstream performance
  - Evidence: Removing Algorithmic + Synthetic data from the annealing mix drops HumanEval/MBPP performance throughout annealing (Figure 9).
  - Contribution: A generalizable recipe—late-stage “quality over quantity” boosts—to sharpen code abilities.
- Two-stage instruction tuning for code
  - What’s different: A deliberate sequencing—Stage 1 broad diverse instructions; Stage 2 high-quality, code-specific tasks (including up-to-date library usage) (Sections 4.1–4.2).
  - Evidence: Stage1+Stage2 outperforms Stage1 alone and mixed training on HE/MBPP/BigCodeBench and a human-authored CodeArena set (Table 10).
- An open “cookbook” for top-tier code LLMs
  - Distinction: Open release not just of weights but of the entire data pipeline, SFT datasets, ablations, and training configurations (Table 1). This degree of transparency is rare at the 6B–8B scale.

## 5. Experimental Analysis
- Evaluation setup (Section 5; Tables 6–8; Figures 6–7)
  - Datasets and metrics:
    - `HumanEval`, `MBPP`, and their stricter variants `HumanEval+` and `MBPP+` (via EvalPlus). Metric: `Pass@1`—percentage of tasks solved by the first generated solution.
    - `BigCodeBench` (Completion and Instruct variants) tests code completion given function signatures and docs, including library-specific challenges.
    - `LiveCodeBench` evaluates complex algorithmic problem solving with contamination control (instruct models on 2305–2409 split).
    - `MultiPL-E` tests multi-language generation (Python, Java, C++, C#, TS, JS, PHP, Bash).
    - `McEval` (40 languages) for multilingual code generation; `MdEval` (18 languages) for multilingual code debugging.
  - Baselines include CodeLlama, CodeGemma, DS-Coder (DeepSeek), Yi-Coder, StarCoder2, and Qwen2.5-Coder. Some baselines are closed-data or non-reproducible; OpenCoder emphasizes reproducibility (Table 1 marks).

- Main quantitative results
  - Base models (Table 6):
    - `OpenCoder-8B-Base`: HumanEval 66.5 / HE+ 63.4; MBPP 79.9 / MBPP+ 70.4; BigCodeBench (3-shot/Full/Hard) 60.6 / 40.5 / 9.5.
      - Competitive among 6B+ base models; notably ahead of CodeLlama-7B and StarCoder2-7B on HumanEval/MBPP; behind Qwen2.5-Coder-7B on some metrics.
    - `OpenCoder-1.5B-Base`: HumanEval 54.3 / HE+ 49.4; MBPP 70.6 / 58.7, strong for its size class.
  - Instruct models (Table 7):
    - `OpenCoder-8B-Instruct`: HumanEval 83.5 / HE+ 78.7; MBPP 79.1 / MBPP+ 69.0; BigCodeBench Full 40.3 / Hard 16.9; LiveCodeBench Avg 23.2.
    - `OpenCoder-1.5B-Instruct`: HumanEval 72.5 / HE+ 67.7; MBPP 72.7 / 61.9; BigCodeBench Full 33.3 / Hard 11.5; LiveCodeBench 12.8.
  - Multilingual performance (Table 8; Figures 6–7):
    - `OpenCoder-8B-Instruct` averages 71.0 on MultiPL-E across 8 languages—competitive with CodeQwen1.5-7B-Chat (71.6) and Yi-Coder-9B (71.8), trailing Qwen2.5-Coder-7B-Instruct (76.5).
    - Figures 6 and 7 show strong multilingual generation and debugging compared to other open models of similar size, though not SOTA in all languages.

- Do experiments support the claims?
  - The core claims are about an open, reproducible pipeline and data-centric findings:
    - File-level dedup beats repo-level: convincingly supported by both dataset statistics (Table 9) and training curves (Figure 8).
    - High-quality annealing matters: Figure 9 shows clear, consistent gains.
    - GitHub star filtering reduces diversity and hurts performance despite lower loss: Figures 10–11 display the trade-off (performance vs. loss and distribution).
    - Two-stage SFT helps: Table 10 shows consistent improvements vs. Stage1-only or mixed training.
  - Performance positioning: While `OpenCoder-8B-Instruct` surpasses some open models, it does not always beat the strongest open-weight models (e.g., Qwen2.5-Coder-7B-Instruct often leads in Table 7–8). The paper’s broader value is the transparent, effective recipe and ablations rather than universal SOTA.

- Ablations, failure cases, robustness
  - Dedup granularity: File vs. repo vs. chunk-level—chunk-level dedup brings little or no benefit and can be wasteful (Appendix B; Table 13; Figure 12).
  - Star-based filtering: Better loss but worse generalization due to narrowed diversity (Section 6.3; Figures 10–11).
  - Two-stage SFT: Stage1+Stage2 > Stage1 or Mix on HE/MBPP/BigCodeBench and on the CodeArena human-authored prompts (Table 10).
  - Distribution shift control: Annealing keeps 84% original distribution to avoid forgetting (Section 2.2).

- Representative quotes of key claims/results
  > “OpenCoder surpasses all previous fully open models … at the 6B+ parameter scale” (Figure 1).

  > “Data deduplication at the file level … achieves higher performance than repository level” (Section 6.1; Table 9; Figure 8).

  > “Removing high-quality annealing data leads to substantial performance drops” (Section 6.2; Figure 9).

  > “Filtering by GitHub stars changes the data distribution, lowers loss but reduces performance” (Section 6.3; Figures 10–11).

  > “Two-stage instruction tuning outperforms single-stage and mixed strategies” (Section 6.4; Table 10).

## 6. Limitations and Trade-offs
- Data and pipeline assumptions
  - Heuristic filtering relies on many thresholds and language-specific rules (Section 2.1.1; Appendix A). Although carefully tuned, such rules may encode biases in what is considered “high quality” and could over-filter certain valuable edge cases.
  - The choice to downsample HTML and Java changes distribution intentionally; alternative tasks (e.g., web automation) might benefit from more HTML.
- Performance trade-offs
  - Not uniformly SOTA: On several benchmarks, Qwen2.5-Coder-7B-Instruct outperforms OpenCoder-8B-Instruct (Tables 7–8; Figures 6–7). The contribution is as much methodological transparency as raw performance.
  - Star-based filtering experiments show the classic generalization vs. optimization trade-off: lower loss does not imply better generalization (Section 6.3).
- Compute and scalability
  - Pretraining costs are substantial: 96,000 GPU hours for the 8B model (Section 3.2). While transparent, reproducing at full scale may be out of reach for many labs.
  - The annealing mixture ratio is acknowledged as “might not be ideal” due to compute budget constraints (Section 2.2).
- Coverage and recency
  - Package-Instruct focuses on Python libraries via PyDoc (Section 4.1). Other ecosystems (Java, JS/TS, C++) could require similar efforts to stay current.
- Contamination defenses
  - Decontamination uses entry-point removal and 10-gram overlap checks (Section 4.4). These are strong practices but can miss semantic near-duplicates without n-gram overlap.

## 7. Implications and Future Directions
- Field impact
  - OpenCoder elevates the standard for reproducibility in code LLMs by publishing a full, high-quality corpus with a rigorous pipeline and ablations (Table 1). This shifts the conversation from opaque “secret sauce” datasets to testable, improvable recipes.
  - The data-centric ablations (Sections 6.1–6.4) provide actionable guidance: prioritize file-level dedup, don’t rely on star filters, invest in high-quality annealing data, and tune in two stages.
- Follow-up research enabled
  - Data science for code LLMs: Researchers can reuse RefineCode and the pipeline to test alternative filters, dedup methods (e.g., embedding-based), or targeted recalls (e.g., security, embedded, scientific computing).
  - Curriculum learning: Extend the annealing idea with more structured curricula (e.g., staged difficulty in algorithmics, progressively complex repos, repository-level tasks).
  - Multilingual expansion: Port package-aware synthesis to other ecosystems (npm/JS, Maven/Java, Cargo/Rust, pip/R), keeping models current with API changes.
  - Contamination-resistant evaluation: Build stronger, auto-updating benchmarks (LiveCodeBench is a good direction; Section 5.2) and improve contamination detection beyond n-gram overlap.
- Practical applications
  - IDE copilots and code assistants with better function-level completion and reasoning (BigCodeBench, MBPP).
  - Teaching and onboarding tools: The “Code Textbooks” and “Educational-Instruct” components suggest ways to build explainable, pedagogical code LLMs.
  - Tool-augmented agents: Package-Instruct strengthens up-to-date library usage—a prerequisite for reliable tool-calling agents in data science or ML engineering.

Overall, OpenCoder contributes a strong, transparent baseline and a set of validated data/training practices that others can build upon. Even where it does not top every leaderboard, the released cookbook demystifies which choices most improve code LLMs and provides a reproducible foundation for the community to iterate.
