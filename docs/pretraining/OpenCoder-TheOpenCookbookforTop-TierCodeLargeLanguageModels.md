# OpenCoder: The Open Cookbook for Top-Tier Code Large Language Models

**ArXiv:** [2411.04905](https://arxiv.org/abs/2411.04905)

## 🎯 Pitch

OpenCoder sets a new standard for transparency and reproducibility in code large language models by releasing not only state-of-the-art model weights, but also the entire data processing pipeline, high-quality deduplicated datasets, synthetic instruction data, training protocols, and intermediate checkpoints. This 'open cookbook' empowers researchers and practitioners to fully understand, reproduce, and improve top-tier code LLMs, revealing which data curation and training techniques truly drive performance. By democratizing every aspect of model creation, OpenCoder accelerates innovation, closes the gap with proprietary models, and enables rigorous scientific study of code AI systems.

---

## 1. Executive Summary (2-3 sentences)
OpenCoder is a fully open code–focused large language model plus an “open cookbook” that releases everything needed to reproduce it: the cleaned and de-duplicated pretraining corpus (RefineCode, 960B tokens), processing pipeline, synthetic instruction datasets, training protocols, and intermediate checkpoints. At the 6B–9B scale, it sets a new bar for “fully open” code LLMs, and the paper isolates which data and training choices matter most—especially file‑level deduplication, targeted annealing with high‑quality data, and a two‑stage instruction tuning scheme (Sections 2–6; Figure 1; Tables 1, 3, 5–8).

## 2. Context and Motivation
- Problem/gap
  - Many high-performing code LLMs exist, but their training data and processes are largely opaque, limiting reproducibility and scientific study of what actually drives quality (Introduction; Table 1). Open-access models often release only weights, not datasets or pipelines; “fully open” models with reproducible datasets lag in performance (Figure 1).
- Importance
  - Practical: Better, transparent code LLMs affect software development, code reasoning, and agentic systems (Introduction).
  - Scientific: Understanding which data curation and training steps yield gains is essential to build stronger, more reliable, and auditable systems.
- Prior approaches and shortcomings
  - The Stack v2 (a major open code corpus) is broad but noisy for top-tier pretraining; it lacks fine-grained code-specific filtering and strong deduplication tuned for GitHub’s heavy reuse (Section 2.1; Figure 3; Appendix D).
  - Some pipelines rely on GitHub star filtering or repo-level deduplication, which can reduce diversity or leave near-duplicates (Section 6.1; Figure 8; Section 6.3; Figures 10–11).
- Positioning
  - OpenCoder offers both a competitive model (1.5B and 8B) and a fully reproducible pipeline. It identifies and tests key data curation decisions, then demonstrates their effect via controlled ablations (Sections 2, 3, 4, 6; Tables 6–10; Figures 8–12).

## 3. Technical Approach
This section unpacks the end-to-end recipe: data → pretraining → annealing → post-training (SFT) → evaluation, emphasizing how each module works.

- Pretraining data: RefineCode (Section 2; Figure 2; Table 2)
  - What it is
    - A 960B-token corpus across 607 languages combining “raw code” (primarily GitHub through Nov 2023 plus non-GitHub code from The Stack v2) and “code-related web data” recalled from large web corpora (Common Crawl, FineWeb, SkyPile) (Sections 2.1–2.1.3; Table 2; Figure 4).
  - Key pipeline modules (Figure 2a)
    1) Preprocessing: drop non-text or huge files (>8 MB) and keep only programming-related extensions guided by GitHub Linguist, ending with 607 language types (Section 2.1.1; Appendix E).
    2) Deduplication (file-first): 
       - Exact dedup: SHA-256 per file; retain the copy with highest GitHub stars and latest commit time (Section 2.1.1).
       - Fuzzy dedup: tokenize into 5-grams, compute 2048 MinHash signatures, then use LSH (16 bands × 128 rows) to cluster near-duplicates; again keep the highest-star, latest file; removes another ~6% of file volume (Section 2.1.1).
       - Definitions: `MinHash` approximates Jaccard similarity; `LSH` (Locality Sensitive Hashing) buckets similar items so near-duplicates collide.
    3) Transformation (fix rather than discard when widespread issues are small per-file but common):
       - Copyright removal: strip boilerplate notices at top comments that are repetitive and non-informative (Section 2.1.1).
       - PII reduction: detect and replace passwords, emails, IPs via regex with placeholders like `<password>` (Section 2.1.1).
    4) Heuristic filtering (Figure 2a; Figure 3; Appendix A.1–A.2):
       - Three rule groups:
         - Natural-language rules (e.g., file size and line counts).
         - General code rules (e.g., counts of variables, average function length, too many TODO/FIXME/assert lines).
         - Language-specific rules for eight major languages (Python, C, C++, C#, Java, JavaScript, Go, HTML), e.g., Python AST-parse check; too many `import` lines; excessive pure hex (Section 2.1.1; Figure 3; Appendix A.2).
       - Rule tuning method: compute “quality signals,” then coarse-to-fine thresholding; sanity-check with a PPL-based spot check using a strong LLM to examine very high/low PPL samples (Appendix A.1).
    5) Data sampling: preserve distribution except downsample large but less informative types (e.g., Java from 449GB to 200GB; HTML from 474GB to 64GB) to avoid dominance by markup or single-language skew (Section 2.1.1 “Data Sampling”).
  - Code-related web data recall (Figure 2b; Sections 2.1.2, C.1–C.3)
    - Build a “seed” of 500k manually labeled “code-like” samples from Common Crawl via an autonomous selection process; train a FastText classifier over BPE-tokenized text (Section 2.1.2).
      - Definition: `FastText` is a lightweight classifier that uses bag-of-subword features; chosen for speed and robustness to morphology.
    - Recall candidate pages, then refine by domain discovery: label base domains as “code-related” if >10% of their pages are code-related (e.g., stackoverflow.com), add URL pattern annotations, and iterate 3× adding false negatives (Section 2.1.2; Figure 2b).
    - Apply the same recall to FineWeb, SkyPile, and the web portion of AutoMathText, producing ~330GB of code-related web text; an additional 178GB of code-related GitHub text (README-like) is added via a trained classifier (Section 2.1.2; Table 2; Appendix C.2).
    - Manual lists of Chinese code/math domains and URL globs are provided for future Common Crawl iterations (Appendix C.1).

- Annealing data (Section 2.2; Table 3)
  - Definition: `annealing stage` is a short, post-pretraining phase using a rapidly decaying learning rate and very high-quality data to sharpen capabilities without forgetting useful general knowledge.
  - Mixture composition:
    - 84% continued sampling from RefineCode (“original distribution”) to prevent distribution shift.
    - `Algorithmic Corpus`: self-contained problem/solution style code pulled by keywords like “leetcode” or “def solution” (Section 2.2).
    - `Synthetic Data`:
      - High-quality, verified code snippets with auto-generated tests; only keep samples that pass tests; extended to multiple languages (Section 2.2).
      - Code “textbooks”: explanatory, concept-level narratives generated about code (seeded by the `hqcode` dataset) using Qwen2‑72B‑Instruct to encourage multi-perspective learning (Section 2.2).
    - Quantities: 83.94B tokens original-data, 12.44B Algorithmic Corpus, 2.71B verified snippets, 0.91B code textbooks (Table 3).

- Model architecture and training (Sections 3.1–3.2; Table 4)
  - Two sizes: `OpenCoder-1.5B` (24 layers, d=2240, 14 heads, 4096 context) and `OpenCoder-8B` (32 layers, d=4096, 32 heads with 8 KV heads, 8192 context).
  - `RoPE` positional encoding; θ=10,000 for 1.5B, θ=500,000 for 8B to support longer contexts (Table 4).
  - Optimizer schedule: Warmup–Steady–Decay (WSD). For both sizes, 2,000-step warmup over ~8B tokens; peak LR 3e‑4; exponential decay to 1e‑5 during annealing (Section 3.2).
  - Compute:
    - 1.5B: 2T tokens pretraining over 4 epochs + 100B annealing tokens; micro-batch 4, global batch 1024; 256× H800 GPUs for ~28k GPU hours (Section 3.2).
    - 8B: 2.5T tokens over 3.5 epochs + 100B decay tokens; micro-batch 1, tensor parallel=2, sequence 8192; global batch 1024 (2048 for first 130k steps at 4096 seq); 512× H100 GPUs for 96k GPU hours (Section 3.2).

- Post-training (instruction tuning) (Sections 4.1–4.4; Figure 5; Table 5)
  - Data sources (Stage 1 and Stage 2 shown in Table 5; workflows in Figure 5):
    - Open-source instruction corpora: Evol-Instruct, Infinity-Instruct (filtered for code), McEval-Instruct, plus real user code queries from WildChat and Code‑290k‑ShareGPT that are cleaned and, when needed, regenerated by a strong LLM (“RealUser‑Instruct”) (Section 4.1).
    - `Educational Instruction Synthesis`: choose high-quality code seeds via a scorer LLM; generate multiple tests; keep only passing items to ensure semantic and syntactic soundness (Section 4.1).
    - `Package-related Instruction Synthesis`: crawl up-to-date Python library docs (e.g., NumPy, pandas), retrieve API signatures and examples, and synthesize Q/A so the model learns current library usage (Section 4.1).
    - `Large-scale Diverse Instruction Synthesis`: from cleaned web text, prompt a large teacher LLM to generate diverse tasks and solutions with unit tests and iterative refinement; temperature=1.0 to encourage diversity (Section 4.1; Figure 5a).
  - Two-stage SFT strategy (Section 4.2; Table 5):
    - Stage 1 “broad capability”: diverse and real-user data (0.7M RealUser, 2.3M diverse-instruct, 1.0M Infinity = 4.0M examples).
    - Stage 2 “code‑specific sharpening”: curated, high-quality code tasks (36k McEval, 111k Evol, 110k Educational, 110k Package = 367k examples).
  - Training details (Section 4.3):
    - Stage 1: 1 epoch, batch size 4096, LR 2e‑5 with 100 warmup steps, cosine decay.
    - Stage 2: 3 epochs, batch size 512, LR 5e‑5 with 100 warmup, cosine decay.
  - Decontamination (Section 4.4):
    - Remove any SFT sample mentioning entry points of test sets (e.g., HumanEval, MBPP) and delete any item with a 10‑gram overlap with evaluation tests.

## 4. Key Insights and Innovations
- A. File-level deduplication is decisively better than repository-level deduplication for GitHub-scale code (Section 6.1; Table 9; Figure 8).
  - What’s new: Large-scale, controlled comparison shows that repo-level dedup leaves many character-identical files and much near-duplicate content; file-level exact+fuzzy dedup removes more redundancy.
  - Why it matters: On two standard code completion benchmarks during pretraining, file-level dedup yields markedly higher Pass@1 despite using far fewer tokens (Figure 8). It maximizes data diversity and quality per token, improving training efficiency.
- B. High-quality annealing data drives large gains beyond bulk pretraining (Section 6.2; Figure 9).
  - What’s new: Direct ablation—remove Algorithmic Corpus and synthetic verified snippets/textbooks—substantially reduces gains in the annealing phase.
  - Why it matters: Demonstrates that “quality over quantity” is crucial late in training; carefully curated, test-verified code and explanatory “textbooks” encode reusable patterns.
- C. Do not use GitHub stars as a filtering signal if you care about distributional diversity (Section 6.3; Figures 10–11).
  - What’s new: Two models trained on “original” vs “stars≥5” filtered data show that the star-filtered model has lower training loss yet worse downstream performance (HumanEval/MBPP curves in Figure 10; loss curves in Figure 11).
  - Why it matters: Stars reduce diversity and distort the data distribution (embedding plots in Figure 11), hurting generalization even if the optimization looks easier.
- D. Two-stage instruction tuning systematically outperforms single-stage or mixed training (Section 6.4; Table 10).
  - What’s new: Head-to-head comparison of Stage1 alone vs Stage1→Stage2 vs “Mix Training” shows the staged approach consistently wins on algorithmic benchmarks and a human-written CodeArena set.
  - Why it matters: It suggests a principled curriculum: first broaden, then specialize with clean, code-specific data—improving both benchmark scores and practical responses.

## 5. Experimental Analysis
- Evaluation methodology (Sections 5.1–5.2)
  - Benchmarks and metrics:
    - Code completion/generation: HumanEval and HumanEval+; MBPP and MBPP+ (Pass@1); BigCodeBench (completion and instruct variants; Full and Hard splits).
    - Real-time competitive coding: LiveCodeBench (split 2305–2409).
    - Multilingual code generation: MultiPL-E (eight languages; Pass@1).
    - Multilingual code generation and debugging: McEval (40 languages) and MdEval (18 languages).
  - Reproducible framework: OpenCodeEval is used for standardized runs (Section 5.1).
  - Baselines: Both “fully open” (e.g., StarCoder2-15B) and “open-weights only” (e.g., Qwen2.5‑Coder‑7B, Yi‑Coder‑9B) (Tables 6–8; Table 1).

- Main quantitative results
  - Base models (Table 6):
    - `OpenCoder-1.5B-Base`: HumanEval Pass@1 = 54.3 (HE) / 49.4 (HE+); MBPP Pass@1 = 70.6 (MBPP) / 58.7 (MBPP+); BigCodeBench Full=24.5, Hard=5.4.
      - Competitive with or better than other ≤3B bases, notably surpassing StarCoder2‑3B on HE/MBPP.
    - `OpenCoder-8B-Base`: HumanEval 66.5 (HE) / 63.4 (HE+); MBPP 79.9 / 70.4; BigCodeBench Full=40.5, Hard=9.5.
      - Strong at the 6B–9B scale; HE+/MBPP+ competitive with CodeQwen1.5‑7B‑Base and DS‑Coder‑6.7B‑Base (Table 6).
  - Instruct models (Tables 7–8; Figures 6–7):
    - `OpenCoder-8B-Instruct`:
      - HumanEval 83.5 (HE) / 78.7 (HE+); MBPP 79.1 / 69.0; BigCodeBench Full=40.3, Hard=16.9; LiveCodeBench Avg=23.2 (Table 7).
      - On MultiPL-E, average across 8 languages = 71.0; strong in JavaScript/TypeScript/PHP/Bash; slightly behind Qwen2.5‑Coder‑7B on average (76.5) but competitive with other open models (Table 8).
      - McEval (Figure 6) and MdEval (Figure 7) show it outperforms other open models of similar size across many languages in both generation and debugging.
    - `OpenCoder-1.5B-Instruct`:
      - HumanEval 72.5 / 67.7; MBPP 72.7 / 61.9; BigCodeBench Full=33.3, Hard=11.5; LiveCodeBench 12.8 (Table 7).
  - “Fully open” positioning
    - Figure 1 and Table 1 place OpenCoder‑8B ahead of previous fully open efforts at similar scale; some “open-weights only” models (e.g., Qwen2.5‑Coder‑7B‑Instruct) are still stronger on several benchmarks.

- Ablations and robustness checks
  - Deduplication level (Section 6.1; Table 9; Figure 8):
    - File‑level dedup keeps 2.4% of Python tokens vs 7.5% for repo‑level; despite far fewer tokens, downstream HE/MBPP curves favor file‑level. Post hoc file‑level dedup after repo‑level can still remove ~68B tokens, showing repo‑level under-dedups.
    - Appendix B (Table 13; Figure 12) shows chunk-level dedup provides little benefit and, when added to repo-level, still underperforms file-level.
  - Annealing data quality (Section 6.2; Figure 9):
    - Removing Algorithmic/Synthetic subsets substantially lowers HE/MBPP gains during annealing.
  - Star filtering (Section 6.3; Figures 10–11):
    - “Stars≥5” data yields lower loss but worse HE/MBPP performance; embedding plots show a narrowed distribution suggesting reduced diversity.
  - Two-stage SFT (Section 6.4; Table 10):
    - Stage1+Stage2 beats Stage1 alone and “Mix Training” on HE/HE+/MBPP/MBPP+ and on a ~400‑prompt CodeArena head-to-head judged vs GPT‑4:
      > Stage1+Stage2: HE 70.1, HE+ 64.0, MBPP 74.6, MBPP+ 64.8, BigCodeBench 31.5, CodeArena win-rate 6.9, vs Stage1: 52.4/48.1/68.7/57.4/22.1/5.3 (Table 10).

- Do the experiments support the claims?
  - Yes for the core data-curation insights: the controlled ablations for dedup level, annealing data quality, stars filtering, and SFT staging are carefully set up and show clear, consistent trends (Sections 6.1–6.4; Figures 8–11; Table 10).
  - For overall competitiveness: OpenCoder‑8B‑Instruct is not the strongest open‑weights model on every benchmark (e.g., Qwen2.5‑Coder‑7B‑Instruct is higher on HE+/MBPP+ and MultiPL-E average in Tables 7–8), but OpenCoder’s contribution is the fully open, reproducible pipeline with strong performance and extensive ablations (Table 1; Figure 1).

## 6. Limitations and Trade-offs
- Compute and scale
  - Training at 96k GPU hours (8B) and 28k (1.5B) is substantial (Section 3.2). The cookbook helps reproducibility, but reproducing the exact models requires significant resources.
- Teacher/model dependencies in synthesis
  - Some synthetic data rely on large teacher LLMs (e.g., Qwen2‑72B‑Instruct for code textbooks), which may limit strict end‑to‑end independence even if prompts are released (Section 2.2; Appendix G).
- Language coverage vs specificity
  - 607 languages are included (Appendix E), but language-specific heuristic rules are implemented for only eight major languages (Section 2.1.1), potentially leaving quality on the table for long-tail languages.
- Annealing mixture ratio
  - The annealing mixture keeps 84% original distribution “given limited compute,” and the paper notes this ratio “might not be ideal” (Section 2.2). Further tuning could improve outcomes.
- Benchmark focus
  - While diverse, many evaluations emphasize function-level code tasks; repository-level tasks and real-world tooling integration are less explored experimentally here (though the pipeline tries to mitigate outdated libraries via Package-Instruct; Section 4.1).
- Star filtering conclusion scope
  - The “avoid star filtering” conclusion is well-supported here (Section 6.3), but might be data‑source dependent; other quality signals (e.g., CI status) were not exhaustively tested.

## 7. Implications and Future Directions
- Field impact
  - OpenCoder shifts the emphasis from “mystery data + weights” to “full-stack reproducibility.” By releasing the RefineCode corpus, filtering rules, synthetic recipes, and checkpoints, it enables rigorous, apples-to-apples research on what makes code LLMs strong (Table 1; Appendices A–G).
- What it enables
  - Systematic studies on:
    - How fine-grained, language-specific filters generalize to more languages.
    - Annealing curriculum design (how much algorithmic vs educational vs verified snippets).
    - The interaction between dedup level and scaling laws for code data.
    - Contamination-robust evaluation and data governance using the released decontamination tooling (Section 4.4).
- Practical applications
  - High-quality base and instruct models for:
    - IDE completion and refactoring with stronger few-shot reasoning (Tables 6–7).
    - Multilingual code generation and debugging in international settings (Figures 6–7; Table 8).
    - Up-to-date API usage via Package-Instruct, improving tool‑augmented agent performance (Section 4.1).
- Future directions proposed
  - Continue updating models and datasets, incorporate community feedback, and iterate on the open cookbook (Conclusion & Future Work). Logical next steps include:
    - Scaling to larger sizes with the same transparent pipeline.
    - Extending language-specific rules and package-level instruction synthesis beyond Python.
    - More repository-level and end-to-end agent evaluations.

Overall, this paper’s main contribution is not only a capable model but a carefully validated methodology. The controlled ablations (Sections 6.1–6.4) and the public release of the entire process provide concrete guidance: use file-level (not repo-level) deduplication, keep data diverse (don’t filter by stars), invest in high-quality annealing data, and perform two-stage instruction tuning to balance breadth and code-specific depth.
