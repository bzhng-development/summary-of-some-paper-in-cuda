# LongCodeZip: Compress Long Context for Code Language Models

**ArXiv:** [2510.00446](https://arxiv.org/abs/2510.00446)

## 🎯 Pitch

LongCodeZip introduces the first training-free, model-agnostic compression framework tailored for code LLMs handling extremely long contexts. By combining conditional perplexity-based mutual information scoring for function selection with fine-grained, knapsack-optimized block pruning, it aggressively reduces code context size—up to 5.6× compression—without degrading, and sometimes even improving, performance across tasks like code completion, summarization, and repository-level QA. This breakthrough enables efficient and cost-effective scaling of code intelligence systems to real-world, large-scale software, overcoming bottlenecks of input length, latency, and information overload that previously limited practical use of code LLMs.

---

## 1. Executive Summary
LongCodeZip is a training-free, model-agnostic framework that compresses very long code contexts so code-focused large language models (LLMs) can reason efficiently without losing task-critical information. It does this via a two-stage process—first selecting the most relevant functions using a mutual-information-like score based on conditional perplexity, then pruning within those functions using perplexity-driven block chunking with budget-aware, knapsack-based selection—achieving up to 5.6× compression with no loss (and sometimes gains) in performance across code completion, summarization, and repository-level question answering (Abstract; Figs. 1–2; Sec. V; Tables II–IV).

## 2. Context and Motivation
- Problem addressed
  - LLMs for code must often consider tens of thousands of tokens (whole modules, multiple files, or entire repos). Long inputs inflate latency and cost due to attention’s quadratic complexity and API pricing, and models struggle to locate truly relevant content in the noise (Sec. I: “Three major challenges…”).
  - Even with 128k-token windows, real projects can exceed limits, forcing truncation and degrading outputs (Sec. I).
  - Code has structure and long-range dependencies (variables, classes, functions across files). When context windows are saturated or noisy, models produce code that doesn’t compile or violates constraints (Sec. I).

- Why it matters
  - Real-world tasks—repository-level QA (RepoQA), long-context code completion, and module summarization—are bottlenecked by context processing costs and by models “getting lost” in lengthy prompts (Sec. I; Datasets in Sec. IV-B).

- Prior approaches and their gaps
  - Retrieval-augmented generation (RAG): retrieves relevant chunks via embedding similarity (e.g., cosine similarity with UniXcoder or CodeBERT). Works when lexical overlap is high but misses implicit dependencies (e.g., configuration classes used in training loops) because similarity is surface-level (Sec. II; Fig. 1 right panel).
  - General text compression (LLMLingua, LongLLMLingua, LLMLingua-2): effective for natural language, but can break code syntax/structure or ignore code-specific dependencies (Sec. I; Related Work Sec. VIII-B).
  - Code-specific simplifiers (DietCode, SlimCode): largely heuristic, often function-level only, and not designed for long multi-function contexts; limited cross-language/task generalization (Sec. I; Related Work Sec. VIII-B).

- Positioning
  - LongCodeZip is a plug-and-play, code-aware compression framework tailored to long contexts. It combines:
    - Instruction-aware function selection using conditional perplexity as an approximate mutual information signal.
    - Intra-function pruning with perplexity-based block detection and knapsack selection under an adaptive, importance-weighted token budget (Sec. III; Fig. 2).
  - It targets long-context code tasks broadly (completion, summarization, QA), not just retrieval or a single language (Sec. IV–V).

## 3. Technical Approach
LongCodeZip compresses a long code context `c` with respect to a task instruction `q` under a token budget `B`, producing a compressed context `c'` that preserves relevance for the downstream model (Sec. III-A).

Key definitions (Sec. III-A; Eqs. 1–3):
- `Perplexity (PPL)`: a standard measure of how “surprised” a language model is by a sequence. Lower PPL means higher predicted likelihood. Conditional perplexity `PPL(q | c)` measures how well the model predicts tokens of `q` given context `c`.
- `Approximated Mutual Information (AMI)`: defined as `AMI(c, q) = PPL(q) − PPL(q | c)` (Eq. 1). If providing `c` lowers the perplexity of `q`, then `c` is helpful (higher AMI). This serves as an instruction-aware relevance score.

Overall pipeline (Fig. 2; Sec. III-B):
1) Coarse-grained compression: select the most relevant functions/classes.
2) Fine-grained compression: within those retained functions, detect semantic blocks and select only the most useful ones, respecting a token budget.

Step-by-step

A) Coarse-grained: relevant function selection (Sec. III-C)
- Function-level chunking: Split source along function/class boundaries. Rationale: functions are modular, syntactically valid units that keep semantics intact when moved/retained (Sec. III-C).
- Instruction-aware ranking with AMI: For each candidate function `cᵢ`, compute how much it reduces `PPL(q)` when included, i.e., `AMI(cᵢ, q)`. Rank functions by AMI descending (Sec. III-C; Eq. 1).
  - Why AMI vs embedding similarity? AMI captures dependency value—how much a chunk helps predict the desired output—even if surface words differ (Sec. II; Fig. 1 contrasts).
- Budget-constrained selection: Greedily keep top-ranked functions under a coarse budget `B_coarse = B / R_fine`, where `R_fine` is the planned fine-grained retention ratio (Sec. III-C). Non-selected regions may be replaced with placeholders to preserve global structure while cutting tokens.

B) Fine-grained: intra-function pruning (Sec. III-D)
- Block-level chunking by perplexity:
  - Treat each line as the smallest unit; compute line-wise perplexity and group consecutive lines into blocks.
  - Mark a block boundary when a line’s perplexity shows a sharp local increase—exceeding its neighbors by at least `α` standard deviations across the function’s lines (Sec. III-D, “Block-Level Chunking”).
  - Intuition: within a coherent subroutine, uncertainty (perplexity) falls as context accumulates; spikes signal semantic shifts (Fig. 4 “Perplexity Distribution”).
- Adaptive budget allocation across functions (Algorithm 1; Eqs. 4–6):
  - Compute a baseline retention ratio for large functions: `R_base = (B − Σ_{small} T_j) / Σ_{large} T_k` (Eq. 4), so short functions (e.g., <5 lines) can be kept in full.
  - Normalize function-level AMI to `[0,1]`, then bias each large function’s retention via `R_biased,i = R_base · (1 + β · (2·AMI_norm,i − 1))` (Eq. 5), where `β` controls how much more budget to give important functions.
  - Clamp rates to `[0,1]` and rescale so total retained tokens match the available budget for large functions `B_large` (Eq. 6).
  - Design rationale: preserve more detail where relevance is higher; avoid over-allocating to trivial functions.
- Dynamic block selection via 0/1 knapsack (Algorithm 2):
  - Treat each block as an item with “value” = (normalized) AMI and “weight” = token count.
  - Solve the 0/1 knapsack to maximize total value under the per-function budget, optionally forcing preservation of specific blocks (`P`) if user-specified.
  - Why knapsack? It balances value-density vs size, ensuring the most helpful set of blocks survives within the token limit.

Implementation notes (Sec. IV-E)
- Hyperparameters vary by task:
  - Code completion: `B=2k`, `R_fine=0.8`, `β=0.5`.
  - Summarization: `B=5k`, `R_fine=0.3`, `β=0.5`.
  - RepoQA: `B=2k`, `R_fine=1.0` (to preserve function structure).
- The compression model typically mirrors the generation model but can be smaller (Sec. V-C; Table VIII shows cross-model robustness, including 0.5B compressors).
- Cost: modest GPU memory/time for compression; much larger savings in generation due to smaller prompts (Sec. V-D; Table IX).

Concrete intuition (Fig. 1 and Fig. 4)
- If the instruction is “complete `train_model`,” the `Config` class with optimizer parameters may share few surface words but is crucial. AMI detects that including `Config` reduces uncertainty about the answer, so it ranks highly (Fig. 1, right).
- Within a long function, perplexity spikes mark shifts (e.g., new helper logic). Fine-grained pruning keeps blocks that best reduce uncertainty about the instruction (Fig. 4).

## 4. Key Insights and Innovations
- Instruction-aware relevance via conditional perplexity (AMI)
  - Novelty: Uses “how much this chunk reduces PPL of the instruction/target” rather than static embedding similarity (Sec. III-A/C; Eq. 1). This captures non-lexical dependencies (e.g., configuration classes) that RAG often misses (Fig. 1).
  - Significance: In ablations, replacing AMI with similarity drops ES by 7.89 points and EM by 7.2 at the same budget (Table VII, “w/ Similarity-based Ranking”).

- Perplexity-based block detection inside functions
  - Novelty: Identifies semantic boundaries by local perplexity spikes instead of line breaks or AST-only rules (Sec. III-D “Block-Level Chunking”; Fig. 4).
  - Significance: Outperforms simple line chunking by 1.57 ES points and 1.2 EM (Table VII, “w/ Line Chunking”), improving information density without heavy parsing.

- Adaptive, AMI-weighted budget allocation across functions
  - Novelty: Allocates more tokens to more relevant functions using a controllable importance parameter `β` (Algorithm 1; Eqs. 4–6).
  - Significance: Removing adaptivity reduces ES by 2.34 and EM by 3.0 (Table VII, “w/o Adaptive Budget Allocation”).

- Knapsack-based block selection for maximum relevance per token
  - Novelty: Formalizes intra-function pruning as a 0/1 knapsack, selecting the highest-value (AMI) blocks within budget (Algorithm 2).
  - Significance: Beats random intra-function selection by 2.48 ES and 3.40 EM (Table VII, “w/ Random Line Selection”).

These are more than incremental tweaks: together they establish a code-aware, instruction-aware compression pipeline that consistently beats both general text compressors and code-specific heuristics across diverse tasks and models (Tables II–IV).

## 5. Experimental Analysis
- Evaluation setup (Sec. IV)
  - Datasets (Table I):
    - Long Code Completion: 500 Python examples with contexts >5k tokens.
    - Long Module Summarization: 139 Python modules >2k tokens.
    - RepoQA: 600 repo-level QA examples across 6 languages (Python, Java, JS/TS, Rust, Go, C++).
  - Metrics:
    - Compression Ratio `= |C_original| / |C_compressed|` (Eq. 7).
    - Completion: Edit Similarity (ES) and Exact Match (EM).
    - Summarization: LLM-as-judge “CompScore” using GPT-4o-mini with order-robust prompting (Eq. 8; Sec. IV-D).
    - RepoQA: retrieved function accuracy by BLEU > 0.8 (Sec. IV-D).
  - Baselines (Sec. IV-C):
    - No Compression / No Context; Random Token/Line.
    - RAG: Sliding Window and Function Chunking with UniXcoder embeddings.
    - Text compressors: LLMLingua, LongLLMLingua, LLMLingua-2.
    - Code compressors: DietCode, SlimCode (Python reproduction via tree-sitter for fair comparison).
    - Advanced RAG for completion: A3-CodGen, cAST, RepoGenix, RLCoder (Table VI).
  - Models:
    - Open-source: DeepSeek-Coder-6.7B, Qwen2.5-Coder-7B, Seed-Coder-8B (instruct variants); also smaller compressors (0.5B/1.5B/3B; Table VIII).
    - Closed-source: GPT-4o, Claude-3.7-Sonnet (Table V).

- Main quantitative results
  - Long code completion (Table II):
    - With Qwen2.5-Coder-7B, LongCodeZip reaches ES 57.55 and EM 32.40 at 4.3× ratio, beating RAG(Function) ES 52.79 / EM 26.00 at 3.1×.
    - With Seed-Coder-8B, LongCodeZip attains ES 63.11 / EM 37.40 at 5.6×—very close to No Compression ES 64.04 / EM 40.20 while using ~1/5 tokens.
    - With DeepSeek-Coder-6.7B, LongCodeZip exceeds the no-compression ES (60.58 vs 57.14) and EM (35.40 vs 34.40) at 5.3× compression.
    - Quote:
      > Table II (Seed-8B): LongCodeZip 63.11 ES / 37.40 EM at 5.6× vs RAG(Function) 60.52 / 35.00 at 3.7×.
  - Module summarization (Table III):
    - DeepSeek-Coder-6.7B: LongCodeZip achieves CompScore 28.01 at 2.5×, beating next best 22.95 (RAG Sliding).
    - Qwen2.5-Coder-7B: LongCodeZip 56.47 at 1.7×, slightly above No Compression 56.00.
    - Seed-Coder-8B: LongCodeZip 55.07 at 3.5×, outperforming RAG and code/text compressors.
    - Quote:
      > Table III (Qwen-7B): LongCodeZip 56.47 CompScore at 1.7× vs RAG(Sliding) 53.50 at 1.7×.
  - RepoQA (Table IV):
    - DeepSeek-Coder-6.7B: LongCodeZip average accuracy 75.3 at 5.3×, vs 59.3 for LongLLMLingua at 3.0× and 38.3 No Compression.
    - Qwen2.5-Coder-7B: 87.2 at 4.5× (close to 86.0 No Compression, and >71.3 LongLLMLingua).
    - Seed-Coder-8B: 80.7 at 5.3×, surpassing No Compression 69.0.
    - Quote:
      > Table IV (Qwen-7B): LongCodeZip 92.0/78.0/87.0/85.0/86.0/95.0 per-language, average 87.2 at 4.5×.
  - Closed-source models (Table V):
    - GPT-4o: Completion ES 64.72 at 4.3× (very close to 65.13 No Compression); RepoQA 88.9 at 5.1× (higher than 87.8 baseline).
    - Claude-3.7-Sonnet: Completion ES 66.27 at 4.3× (matches 66.24 baseline); RepoQA 88.9–90.7 at 5.1× (meets/exceeds baseline).
    - Quote:
      > Table V (GPT-4o RepoQA): LongCodeZip avg 88.9 at 5.1× vs 87.8 No Compression.
  - Against advanced RAG (Table VI):
    - Seed-8B: LongCodeZip 63.11/37.40 at 5.6× vs RepoGenix 60.28/34.70 at 3.5×.
    - Claude-3.7-Sonnet: LongCodeZip 66.27/40.20 at 4.3× vs RLCoder 62.76/37.90 at 4.0×.
  - Ablations (Table VII):
    - Largest hit comes from replacing AMI ranking with similarity: −7.89 ES / −7.20 EM.
    - Removing fine-grained stage: −1.45 ES / −1.20 EM.
    - Replacing perplexity chunking with line chunking: −1.57 ES / −1.20 EM.
  - Efficiency (Table IX; Fig. 3):
    - Generation time: reduced from 15.70s (No Compression) to 6.59s with LongCodeZip; compression adds only 2.58s.
    - GPU memory: small extra during compression (Base + 0.69 GB), and similar generation memory to other compressors.
    - Performance vs remaining context (Fig. 3): LongCodeZip dominates other methods across all compression levels; gains are especially large at severe compression (<10% remaining).

- Robustness and transferability
  - Cross-model compressor/generator combinations (Table VIII): Even a 0.5B compressor performs on par with larger ones (Avg ES 60.13 vs 60.58 with 7B), indicating strong generalization and practicality for resource-constrained settings.

- Do the experiments support the claims?
  - Coverage: three tasks, multiple datasets/languages, open/closed models, advanced baselines, ablations, efficiency, and a qualitative case (Fig. 4).
  - Results consistently show equal-or-better downstream performance at substantially higher compression ratios than baselines, validating both effectiveness and efficiency claims (Sec. V; Tables II–VI, IX).

- Failure modes and threats
  - Failure cases: when context lacks relevant info or the instruction is ambiguous, selection can falter (Sec. VI-B “Case Study”).
  - Summarization metric uses LLM-as-judge; mitigated by order-robust prompting and a distinct referee model (Sec. VII “Threats to Validity”).

## 6. Limitations and Trade-offs
- Dependence on perplexity calibration
  - AMI and block segmentation hinge on the compression model’s perplexity estimates. If the model is poorly calibrated for a language or code style, relevance estimates may be noisy (Sec. III-A/D). Cross-model results mitigate but do not eliminate this risk (Table VIII).

- Hyperparameter sensitivity and task tuning
  - Thresholds like the boundary spike factor `α`, importance `β`, and budgets (`B`, `R_fine`) are tuned per task on a held-out set (Sec. IV-E). Different domains may require re-tuning.

- Two-stage overhead vs savings
  - Although compression overhead is small (2.58s; Table IX), it adds latency before generation. For cheap or short-generation tasks, the amortization may be less favorable (Sec. V-D; Fig. 3 shows diminishing returns at milder compression).

- Structural assumptions
  - Function-level chunking presumes code follows typical structure (functions/classes). Non-standard scripts, heavily macro’d code, or generated code could reduce boundary quality. Perplexity-based block detection uses line granularity; languages with significant semantics across single lines (e.g., minified JS) may be harder (Sec. III-D).

- Scope of tasks and metrics
  - Benchmarks focus on completion, summarization, and RepoQA. Other tasks (e.g., compilation success under edits, project-level refactoring planning) were not evaluated. Summarization relies on LLM-as-judge (Sec. IV-D; VII).

- Missing static analysis signals
  - The method is purely neural without explicit program analysis (data/control flow, symbol resolution). While AMI captures implicit usefulness, certain dependency types might benefit from static cues (Related Work Sec. VIII-B).

## 7. Implications and Future Directions
- Impact on the field
  - Demonstrates that instruction-aware, perplexity-driven compression can safely shrink long code contexts by 4–6× while preserving or improving quality (Tables II–IV). This reframes long-context handling: rather than only expanding windows or relying on RAG similarity, one can compress adaptively by “what helps predict the answer.”

- Practical applications
  - IDE integration for repository-level completion: faster, cheaper suggestions under token budgets.
  - Documentation and code review: generate module-level summaries from large files with reduced compute (Table III).
  - Repository QA and navigation assistants: more accurate function retrieval and reasoning across languages (Table IV).

- Research directions
  - Hybrid static–neural compression: fuse AMI with program analysis (symbol tables, dependency graphs) to preserve non-local constraints.
  - Learned importance models: train small compressors to predict AMI or boundary spikes, reducing passes needed for perplexity.
  - Multi-file and cross-repo dependency handling: budget allocation across files and packages; hierarchical budgets across project graphs.
  - Interactive/constraint-aware compression: let users pin regions or specify policies (e.g., “never drop import blocks,” aligning with Algorithm 2’s preserved-set `P`).
  - End-to-end pipelines: pair RAG retrieval to pre-filter a repository, then apply LongCodeZip for final compression (noted as complementary in Sec. V).
  - Evaluation broadening: include compile/run correctness, human-in-the-loop developer studies, and more languages/formats (e.g., notebooks, configs).

> Overall: LongCodeZip contributes a principled, code-aware compression framework using conditional perplexity as an effective proxy for instruction relevance (Eqs. 1–3), coupled with adaptive budgeting and knapsack selection (Algorithms 1–2). Evidence across datasets, models, and baselines shows strong performance at aggressive compression ratios with tangible latency and cost reductions (Tables II–V, IX; Fig. 3).
