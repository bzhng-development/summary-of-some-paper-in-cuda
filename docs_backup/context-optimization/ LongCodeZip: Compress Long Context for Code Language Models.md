# LongCodeZip: Compress Long Context for Code Language Models

**ArXiv:** [2510.00446](https://arxiv.org/abs/2510.00446)
**Authors:** Yuling Shi, Yichun Qian, Hongyu Zhang, Beijun Shen, Xiaodong Gu
**Institutions:** Shanghai Jiao Tong University, Stanford University, Chongqing University

## 1. Executive Summary (Reasoning + Summary)

Reasoning: To understand whether a compression method truly helps long-context code tasks, we look for three things: whether it preserves task-relevant code structure and dependencies, whether it reduces tokens without harming downstream performance, and whether it generalizes across models and tasks. The paper provides a clear mechanism and comprehensive evaluation covering all three aspects.

Summary:
- The paper introduces LongCodeZip, a training-free, model-agnostic, plug-and-play framework that compresses long code contexts for code LLMs through a two-stage process: function-level selection guided by instruction-aware `approximated mutual information (AMI)` and block-level pruning guided by `perplexity` and an adaptive knapsack optimization (Section III; Fig. 2).
- Across code completion, module summarization, and repository-level QA, LongCodeZip achieves up to a 5.6× compression ratio without degrading performance, while reducing generation latency and token costs (Abstract; Tables II–IV, IX).

## 2. Context and Motivation (Reasoning)

Reasoning: Long code prompts often exceed model context limits or become noisy, making relevant signals hard to find. General text compression methods, retrieval-only approaches, or rule-based code pruning fail to respect code’s structural semantics and dependencies.

- Problem addressed:
  - Code LLMs face high latency and API costs when handling long inputs, and often fail to identify relevant content amid lengthy, structured code contexts (Introduction; “Three major challenges…”).
  - Even with extended context windows (e.g., 128k), real repositories can exceed limits; truncated context harms correctness (Introduction; [21]).
- Importance:
  - Real-world tasks like repository-level QA and long-context code completion require cross-function and cross-file reasoning over tens of thousands of tokens; reducing tokens while preserving relevant dependencies directly lowers costs and improves reliability (Introduction; Fig. 1).
- Prior approaches and shortcomings:
  - General text compressors (LLMLingua, LongLLMLingua, LLMLingua-2) prune tokens or sentences but “often break code structure” and ignore code-specific dependencies (Abstract; Introduction).
  - Standard RAG selects by embedding similarity, missing implicit dependencies (e.g., config values used by training logic) as shown in Fig. 1 (“Challenge for RAG…”).
  - Code compressors (DietCode, SlimCode) either rely on attention heuristics or rule-based pruning; often limited to function-level or short examples and not designed for long contexts (Introduction; Related Work).
- Positioning:
  - LongCodeZip targets long-context code compression with code-aware mechanisms: instruction-aware AMI ranking at the function level and perplexity-based block detection at the intra-function level (Abstract; Sections III-C and III-D).
  - It is training-free and model-agnostic, designed to plug into existing LLM pipelines and work with both open- and closed-source models (Sections IV–V; Table V, VIII).

## 3. Technical Approach (Reasoning)

Reasoning: The approach must select context that actually helps the model generate the instruction-consistent output. Measuring how a context reduces the model’s uncertainty (perplexity) about the instruction provides a principled relevance signal beyond surface similarity. At the intra-function level, perplexity changes can mark semantic boundaries, and knapsack selection maximizes information density under a strict token budget.

- Problem formulation (Section III-A):
  - Goal: Given long code context `c` and instruction `q`, produce compressed `c′ ⊆ c` such that its length `|c′| ≤ B` (token budget `B`), maximizing downstream performance.
  - Core relevance measure: `AMI(c, q)` approximates how much context `c` helps predict the instruction `q`, defined by reductions in `perplexity`:

    > Equation (1): `AMI(c, q) = PPL(q) − PPL(q | c)`

    - `PPL(q|c)` is conditional perplexity: lower means the model is more confident about the instruction given the context (Equation (2)).
    - `PPL(q)` is perplexity of the instruction without context (Equation (3)).
    - Perplexity definition: A measure of token-level uncertainty; for a sequence, lower perplexity indicates tokens are more predictable for the model.

- Two-stage framework overview (Fig. 2; Section III-B):
  1) Coarse-grained compression: split the source by function or class boundaries, rank each chunk with `AMI`, and select under `B_coarse`.
  2) Fine-grained compression: segment selected functions into blocks by perplexity-based boundaries; allocate adaptive budgets per function based on importance; select blocks via a `0/1 knapsack` to maximize cumulative relevance (value) under budget (weight).

- Stage 1: Coarse-grained function selection (Section III-C):
  - Function-level chunking: preserve syntax and modularity by cutting at function/class boundaries (ensures coherent logic).
  - Instruction-aware ranking: compute `AMI` for each chunk relative to the instruction `q`. Intuition: if adding a chunk reduces the perplexity of the instruction tokens, it is likely relevant.
  - Budget-constrained selection: greedily include top-scored functions until hitting `B_coarse` = `B / R_fine` (configurable fine-grained ratio). Non-selected chunks are replaced with placeholders (e.g., comments) to preserve overall file structure.

- Stage 2: Fine-grained intra-function pruning (Section III-D):
  - Block-level chunking via perplexity-based boundaries:
    - Treat each line as atomic and compute line-wise perplexity (Equation (3)).
    - When a line’s perplexity sharply increases—exceeding local neighbors by a threshold proportional to the global standard deviation (`α` times std)—it marks a new block boundary. Intuition: a large increase suggests a semantic shift.
    - Result: semantically coherent blocks that better reflect code structure than naive line splits (see Fig. 4 for a real example).
  - Adaptive budget allocation (Algorithm 1; Equations (4)–(6)):
    - Small functions (`F_small`, e.g., <5 lines) are kept entirely.
    - Baseline retention ratio for large functions:

      > Equation (4): `R_base = (B − Σ_{j∈F_small} T_j) / Σ_{k∈F_large} T_k`

      where `T_j` is tokens in function `j`.
    - Importance-biased retention per function `i`:

      > Equation (5): `R_biased,i = R_base · (1 + β · (2 × AMI_norm,i − 1))`

      where `AMI_norm,i` is min-max normalized to [0,1], and `β` controls how much more budget to give to important functions.
    - Global rescaling to meet the large-function budget `B_large`:

      > Equation (6): `R_i = R_biased,i · (B_large / Σ_j R_biased,j · T_j)`
  - Dynamic block selection via `0/1 knapsack` (Algorithm 2):
    - Each block is an item with `value = AMI_norm` and `weight = token length`.
    - Solve knapsack to maximize total value under function’s budget `B_i`. Optionally pre-preserve a set `P` of blocks, subtracting their `T_j` from remaining budget.

- Why these design choices:
  - `AMI` over embedding similarity: captures how a chunk reduces uncertainty about the instruction, which includes implicit dependencies (Fig. 1 shows similarity can miss `Config` needed by `train_model`; Section II).
  - Perplexity-based block detection: respects semantic boundaries without parsing overhead; adaptable across languages; empirically aligns with code structure (Fig. 4).
  - Knapsack selection: formalizes “maximize relevance under strict tokens” and outperforms random selection (Table VII).
  - Adaptive per-function budgets (`β`): ensures more important functions retain richer context, improving information density (Equations (4)–(6); Algorithm 1).

## 4. Key Insights and Innovations (Reasoning)

Reasoning: The novelty lies not just in compressing tokens but in code-aware mechanisms that maintain semantic integrity and dependency relevance. Data shows these choices matter.

- Instruction-aware `AMI` ranking at the function level:
  - Innovation: Scoring chunks by “perplexity reduction” on the instruction rather than surface similarity.
  - Why it matters: Captures implicit dependencies and task relevance; in ablations, replacing AMI with similarity drops ES by 7.89 and EM by 7.20 for Qwen2.5-7B (Table VII).
  - Distinct from prior RAG baselines that rely on embeddings (Section II; Fig. 1).
- Perplexity-based block detection for code:
  - Innovation: Using local perplexity shifts to detect semantic boundaries within functions.
  - Why it matters: Creates meaningful blocks that preserve intra-function logic; outperforms simple line chunking in ES by 1.57 (Table VII) and is computationally lighter than line-by-line ranking (Section III-D; Fig. 4).
- Adaptive budget allocation aligned with function importance:
  - Innovation: Allocate more tokens to functions with higher normalized `AMI` via `β`-controlled bias; rescale to meet global budgets (Equations (4)–(6); Algorithm 1).
  - Why it matters: Improves ES by 2.34 and EM by 3.00 vs uniform allocation (Table VII), raising overall information density.
- Training-free, model-agnostic plug-and-play with cross-model generalization:
  - Innovation: No training or finetuning required; works across open and closed models; small compressors perform well.
  - Why it matters: Practical deployment across model ecosystems; e.g., using `Qwen2.5-Coder-0.5B` as compressor yields competitive ES (Table VIII); strong results on GPT-4o and Claude-3.7-Sonnet (Table V).

## 5. Experimental Analysis (Reasoning)

Reasoning: Robust validation requires diverse tasks, strong baselines, precise metrics, and efficiency measurements. The paper covers all and provides concrete comparative data.

- Evaluation setup (Section IV):
  - Tasks and datasets (Table I):
    - Long Code Completion: 500 Python examples with >5k token contexts [1].
    - Long Module Summarization: 139 examples from 43 repos with >2k tokens [21].
    - RepoQA: 600 tests across 6 languages; retrieve a target function (BLEU > 0.8) [13].
  - Metrics:
    - Compression `Ratio = |C_original| / |C_compressed|` (Equation (7)).
    - Code completion: `Exact Match (EM)`, `Edit Similarity (ES)` [1].
    - Summarization: `CompScore` via LLM-as-judge; order-robust scoring (Equation (8)).
    - RepoQA: retrieval accuracy using BLEU threshold 0.8 [13].
  - Baselines:
    - No Compression / No Context bounds.
    - Random pruning (token and line).
    - Retrieval-based: RAG (sliding window) vs RAG (function chunking).
    - Code compression: DietCode, SlimCode.
    - Text compression: LLMLingua, LongLLMLingua, LLMLingua-2.
  - Models:
    - Open-source: DeepSeek-Coder-6.7B, Qwen2.5-Coder-7B, Seed-Coder-8B.
    - Closed-source: GPT-4o, Claude-3.7-Sonnet (Table V).
  - Hyperparameters and budgets (Section IV-E): Task-tailored `B`, `R_fine`, `β` (e.g., completion uses `B=2k, R_fine=0.8, β=0.5`).

- Main quantitative results:
  - Code completion (Table II):
    - Qwen2.5-7B: LongCodeZip achieves `ES=57.55, EM=32.40` at `4.3×` compression vs. RAG (Function) `ES=52.79, EM=26.00` at `3.1×`.
    - Seed-8B: LongCodeZip `ES=63.11, EM=37.40` at `5.6×`, close to no-compression `ES=64.04, EM=40.20`.
    - DeepSeek-6.7B: LongCodeZip `ES=60.58, EM=35.40` at `5.3×`, exceeding no-compression `EM=34.40`.
  - Long module summarization (Table III):
    - DeepSeek-6.7B: LongCodeZip `CompScore=28.01` at `2.5×`, higher than SlimCode `20.24` and RAG (Sliding) `22.95`.
    - Qwen2.5-7B: LongCodeZip `CompScore=56.47` at `1.7×`, matching or exceeding no-compression `56.00`.
    - Seed-8B: LongCodeZip `CompScore=55.07` at `3.5×`, higher than RAG variants and SlimCode.
  - RepoQA (Table IV):
    - DeepSeek-6.7B: LongCodeZip `Avg=75.3` at `5.3×` vs LongLLMLingua `59.3` at `3.0×` and RAG (Sliding) `55.5`.
    - Qwen2.5-7B: LongCodeZip `Avg=87.2` at `4.5×`, near no-compression `86.0`.
    - Seed-8B: LongCodeZip `Avg=80.7` at `5.3×` vs LongLLMLingua `71.2`.
  - Closed models (Table V):
    - GPT-4o: LongCodeZip `ES=64.72` vs `65.13` no-compression, at `4.3×`; RepoQA `Avg=88.9`, exceeding `87.8` baseline.
    - Claude-3.7-Sonnet: LongCodeZip `ES=66.27` vs `66.24` baseline, at `4.3×`; RepoQA `Avg=88.9`–`90.7`, exceeding baseline.
  - Against advanced repository-level RAG (Table VI):
    - Seed-8B: LongCodeZip `ES=63.11, EM=37.40` at `5.6×`, outperforming A3-CodGen, cAST, RepoGenix, RLCoder under stricter compression.
    - Claude-3.7: LongCodeZip matches or exceeds those methods.

- Ablations and robustness (Table VII; Fig. 3; Section V-B, VI-A):
  - Coarse AMI ranking is critical: Replacing with similarity drops ES by 7.89 and random ranking by 17.79.
  - Fine-grained components add measurable gains: adaptive budgets (+2.34 ES), knapsack selection (+2.48 ES vs random).
  - Performance vs remaining context (Fig. 3): LongCodeZip dominates across compression levels, with especially large gains under severe compression (<10% remaining).
  - Statistical significance: Differences are “substantial and statistically significant (p < 0.001 via Wilcoxon signed-rank test on 10 repeated experiments)” (Section V-A).

- Efficiency (Table IX):
  - For Qwen2.5-7B, LongCodeZip reduces generation time from `15.70s` to `6.59s` with `4.3×` compression; compression overhead is `2.58s` GPU time and modest memory.
  - “This also translates to substantial cost savings when using expensive commercial LLM APIs” (Section V-D). Overhead can be reduced by using a 0.5B compressor (Table VIII) or quantization (Related Work [43]).

- Do experiments support claims?
  - Coverage across tasks, models (open and closed), and languages is strong (Tables II–V).
  - Multiple baselines establish breadth; ablations tie gains to specific components (Table VII).
  - Efficiency analysis demonstrates practical benefits (Table IX).
  - Summarization uses LLM-as-judge, which is common but has known caveats; the paper mitigates order bias (Equation (8), Section IV-D; Threats to Validity).

## 6. Limitations and Trade-offs (Reasoning)

Reasoning: Compression inevitably risks dropping useful context. The method’s reliance on perplexity and function boundaries introduces assumptions; we should identify conditions where it might falter and what costs it entails.

- Assumptions and sensitivities:
  - `Perplexity` as a signal depends on the compressor model’s tokenization and language modeling quality; inaccuracies can misplace boundaries or misjudge importance (Section III-D).
  - `Function-level chunking` assumes coherent modular boundaries; languages or code styles with atypical structure (macros, multi-file constructs, dynamic code) may weaken this assumption (Section III-C).
  - The thresholding parameter `α` for boundary detection influences block granularity; not fully specified for tuning across languages (Section III-D).
- Scenarios not well addressed:
  - Ambiguous instructions or contexts lacking relevant information: the method “may struggle to identify and preserve useful blocks” (Section VI-B; Case Study failure modes).
  - Tasks needing global, cross-file invariants or intricate project-wide dependencies may require deeper static analysis than perplexity-based grouping.
- Computational costs and constraints:
  - Compression requires additional GPU time (e.g., `2.58s` in Table IX), though much less than generation savings; small compressors (0.5B) mitigate overhead (Table VIII).
  - Knapsack selection per function introduces dynamic programming costs proportional to number of blocks and budget; may need optimization for massive repositories (Section III-D; Algorithm 2).
- Potential weaknesses or open questions:
  - Summarization evaluation relies on LLM-as-judge (Section IV-D), which can diverge from human expert ratings in nuanced documentation tasks (Threats to Validity).
  - Placeholders replacing unselected chunks preserve structure but may still affect model behavior if they inadvertently signal irrelevance too strongly (Section III-C).
  - The interaction between LongCodeZip and external retrieval (RAG) is promising but not fully explored; combining them may introduce new tuning complexity (Section V-A note).

## 7. Implications and Future Directions (Reasoning)

Reasoning: A practical, training-free compressor that maintains code semantics under tight budgets enables more scalable use of code LLMs. The methods introduced can seed a broader class of code-aware, information-theoretic compression techniques.

- Field impact:
  - Demonstrates that instruction-aware `AMI` and code-specific `perplexity` signals can substantially outperform similarity-based retrieval and general text compressors for code tasks (Tables II–IV).
  - Makes long-context code generation more affordable and faster without retraining models, broadening deployment in real repositories (Table IX; Section V-D).
- Follow-up research:
  - Hybrid systems: Integrate LongCodeZip with strong repository-level retrievers (A3-CodGen, cAST, RepoGenix, RLCoder) to first retrieve relevant files/functions, then compress them (Table VI suggests complementarity).
  - Enhanced segmentation: Combine `perplexity` with static analysis (ASTs, data-flow graphs) to further improve block boundaries, especially for languages with complex structures (Related Work; Section III-D).
  - Adaptive parameters: Auto-tune `β`, `R_fine`, and `α` by task type and model behavior; meta-learning or heuristic controllers could adjust budgets dynamically.
  - Caching and incremental compression: Maintain compressed context across multi-turn interactions, updating budgets and blocks as instructions evolve.
  - Beyond code: Explore AMI-driven compression for other structured modalities (e.g., configs, logs, build scripts) where dependencies are contextual but not purely lexical.
- Practical applications:
  - Repository-level assistants: Faster, cheaper code completion, QA, and summarization on large codebases (Tables II–IV).
  - IDE integration: On-device small compressors (0.5B) to pre-filter context before sending to cloud APIs, lowering token costs while maintaining performance (Table VIII; Section V-D).
  - CI pipelines: Compress module contexts for automated documentation and code review.

> “LongCodeZip achieves up to a 5.6× compression ratio without sacrificing task performance” (Abstract; confirmed by Seed-8B completion in Table II and multiple tasks), with “significantly reduced generation time and token costs” (Abstract; Table IX). The framework’s training-free, model-agnostic design and strong cross-model results (Table V, VIII) indicate it is practical and broadly applicable.