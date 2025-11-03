# DeepSeek-Coder: When the Large Language Model Meets Programming – The Rise of Code Intelligence

**ArXiv:** [2401.14196](https://arxiv.org/abs/2401.14196)
**Authors:** Daya Guo, Qihao Zhu, Dejian Yang, Zhenda Xie, Kai Dong, Wentao Zhang, Guanting Chen, Xiao Bi, Y. Wu, Y. K. Li, Fuli Luo, Yingfei Xiong, Wenfeng Liang
**Institutions:** 

## 🎯 Pitch

DeepSeek-Coder revolutionizes open-source code-specialized models by implementing repository-level training and a balanced Fill-in-the-Middle objective, setting new state-of-the-art results. This enables greater project-wide understanding and outperforms many leading closed models, all while providing unrestricted access for wider research and practical applications in IDEs and educational tools.

---

## 1. Executive Summary (2-3 sentences)
DeepSeek‑Coder introduces an open‑source family of code-specialized large language models (`1.3B`, `6.7B`, `33B`) trained from scratch on 2 trillion tokens with project‑level (repository‑level) code organization and a balanced Fill‑in‑the‑Middle (`FIM`) objective. Across code generation, cross‑file completion, code infilling, and program‑aided math reasoning benchmarks, the models set new state-of-the-art results for open models and in several cases surpass closed models such as Codex and GPT‑3.5, while offering a 16K token context window and a permissive license (Abstract; Fig. 1; Sec. 4; Table 3–8).

## 2. Context and Motivation
- Problem addressed:
  - Open code LLMs lag behind closed models in performance and capabilities (Introduction).
  - Most prior pretraining treats each source file independently, ignoring cross‑file dependencies needed for real project work (Sec. 2.2).  
- Why this matters:
  - Practical coding involves reading and modifying multiple files that depend on each other; IDE assistance, repository modernization, and automated refactoring all require repository-level understanding.
  - Wider research and commercial access needs strong open models under permissive licenses (Abstract).
- Prior approaches and gaps:
  - Open models like StarCoder and CodeLlama are trained mostly on file‑level data and typically use FIM, but do not organize data at repository level nor thoroughly analyze FIM trade‑offs (Introduction; Sec. 2.2; Sec. 3.1.2).
  - Closed systems (e.g., GPT‑3.5/4) achieve strong results but are proprietary, limiting reproducibility and control (Introduction).
- Positioning:
  - DeepSeek‑Coder contributes both model weights and a training recipe emphasizing repository‑level construction, balanced FIM, extended context, and extensive evaluation. It aims to narrow the gap with GPT‑4 while remaining fully open (Abstract; Sec. 4).

## 3. Technical Approach
This section explains the end‑to‑end pipeline: data, repository organization, deduplication and quality control, model design, training objectives, long context adaptation, and instruction tuning.

- Data construction (Sec. 2; Fig. 2; Table 1):
  - Composition: 87% source code across 87 languages, 10% English code‑related text (GitHub Markdown, StackExchange), 3% non‑code Chinese natural language (Sec. 2).
  - Rule‑based filtering: reuse StarCoder‑style rules to remove overly long lines, low alphabetic content, XML‑like files (except XSLT), HTML with low visible‑text ratio, and size‑bounded JSON/YAML (Sec. 2.1).
  - Quality screening and decontamination: compile checks and a quality model plus heuristics; n‑gram filtering removes overlaps with popular benchmarks such as HumanEval/MBPP/GSM8K/MATH (Sec. 2.4).

- Repository‑level dependency parsing and packing (Sec. 2.2; Algorithm 1):
  - Goal: preserve cross‑file dependencies by concatenating files in an order where a file’s prerequisites appear earlier.
  - How it works:
    - Extract simple static import/use/include edges via regular expressions per language (e.g., Python `import`, C# `using`, C `include`).
    - Build an adjacency list and in‑degree map over files in a repo (Algorithm 1, lines 1–16).
    - For each disconnected subgraph, perform a modified topological sort: iteratively pick the node with minimum in‑degree (not necessarily zero) to tolerate cycles, decrement in‑degrees of its outgoing neighbors, append to result (lines 18–32).
    - Concatenate the ordered file contents into one training sample, prefixing each file with a comment indicating its path to preserve location information (end of Sec. 2.2).
  - Why this choice: static regex-based extraction is simple and language‑portable; the “minimal in‑degree” variant resolves small cycles without discarding files, ensuring trainable sequences.

- Repository‑level deduplication (Sec. 2.3):
  - Instead of file‑level dedup (common in prior work), treat the concatenated repository string as the unit for near‑duplicate removal.  
  - Rationale: file‑level dedup can remove random files and destroy dependency structure; repo‑level keeps project integrity.

- Training objectives (Sec. 3.1):
  - Next Token Prediction (Sec. 3.1.1): standard autoregressive objective on packed sequences.
  - Fill‑in‑the‑Middle (`FIM`) (Sec. 3.1.2):
    - What is FIM: a pretraining task where each document is split into `prefix`, `middle`, `suffix`; the model is fed rearranged text and must generate the missing middle given both sides.
    - Modes:
      - `PSM` = Prefix–Suffix–Middle.
      - `SPM` = Suffix–Prefix–Middle.
      - A variant `MSP` (Masked Span Prediction) masks multiple spans (as in T5) for reconstruction.
    - Implementation:
      - Use three sentinel tokens `<|fim_start|>`, `<|fim_hole|>`, `<|fim_end|>`.
      - Example packed format (PSM): `<|fim_start|> f_pre <|fim_hole|> f_suf <|fim_end|> f_middle <|eos_token|>` (Sec. 3.1.2).
      - Apply at document level before sequence packing at a 50% rate in final models (Sec. 3.1.2).
    - Design choice: Ablation (Fig. 3) shows 100% FIM maximizes single‑line infilling but hurts normal completion; 50% PSM balances both and outperforms MSP at 50%.

- Tokenizer and architecture (Sec. 3.2–3.3; Table 2):
  - BPE tokenizer with 32k vocab (Sec. 3.2).
  - Decoder‑only Transformer with RoPE positional encoding; `33B` model uses `Grouped‑Query Attention (GQA)` with group size 8 to speed inference; `FlashAttention v2` for efficient attention computation (Sec. 3.3).
  - Key sizes (Table 2):  
    - `1.3B`: 24 layers, 2048 hidden, 16 heads.  
    - `6.7B`: 32 layers, 4096 hidden, 32 heads.  
    - `33B`: 62 layers, 7168 hidden, 56 heads, GQA(8).

- Optimization and infrastructure (Sec. 3.4–3.5):
  - AdamW (β1=0.9, β2=0.95). Three‑stage LR schedule with 2000 warm‑up steps; each stage’s LR scaled by √(1/10) vs the previous; final LR = 10% of initial (Sec. 3.4).
  - Training with HAI‑LLM framework using tensor parallelism, ZeRO data parallelism, and pipeline parallelism on A100/H800 clusters connected by NVLink/NVSwitch and InfiniBand (Sec. 3.5).

- Long context adaptation (Sec. 3.6):
  - Reconfigure RoPE with linear scaling factor 4 and base frequency 100000 (vs 10000).  
  - Extra 1000 steps of training at 16K sequence length; theoretically supports up to 64K tokens, but empirical reliability is best at 16K.

- Instruction tuning (Sec. 3.7):
  - Create `DeepSeek‑Coder‑Instruct` by fine‑tuning base models on high‑quality instructions in Alpaca format.  
  - Special delimiter `<|EOT|>` marks end of each conversational turn; cosine LR schedule with 100 warm‑up steps; LR `1e‑5`; batch of 4M tokens; total 2B tokens.

- Continued pretraining from a general LLM (Sec. 5; Table 9–10):
  - `DeepSeek‑Coder‑v1.5 7B`: initialize from DeepSeek‑LLM‑7B and continue pretraining on 2T tokens with a 4K context and only the next‑token objective.
  - Data mix emphasizes 70% source code, plus natural language and math text to improve broader reasoning (Table 9).

## 4. Key Insights and Innovations
- Repository‑level construction and ordering (Sec. 2.2–2.3):
  - Novelty: training samples are whole repositories, files ordered by inferred dependencies and deduplicated at repo granularity.
  - Why it matters: preserves cross‑file context that file‑level corpora discard; improves cross‑file completion (Table 7 shows higher exact match with retrieval vs other 7B‑scale models; removing repo‑level pretraining reduces EM across Java/TS/C#).

- Balanced `FIM` at 50% `PSM` rate with custom sentinels (Sec. 3.1.2; Fig. 3):
  - Novelty: systematic ablation that clarifies the trade‑off—100% FIM maximizes infilling but harms general completion; `PSM@50%` outperforms `MSP@50%`.
  - Impact: high single‑line infill accuracy without sacrificing standard code completion (Table 6 shows mean infilling accuracy 80.7% for `7B` and 81.2% for `33B`, strong vs CodeLlama).

- Repository‑aware deduplication (Sec. 2.3):
  - Novelty: deduplicate after concatenating the repo to avoid breaking structure.  
  - Impact: likely reduces overfitting to repeated boilerplate while keeping dependency graph intact; though not isolated as an ablation, it is integral to the cross‑file gains.

- Long‑context training with RoPE scaling to 16K tokens (Sec. 3.6):
  - Novelty: simple yet effective RoPE scaling and short continued training yield reliable 16K context processing for large repositories.
  - Impact: supports project‑wide tasks and “fill‑in‑middle” with long prefixes/suffixes.

- Open models with strong size‑efficiency (Abstract; Table 3–8):
  - Observation: `DeepSeek‑Coder‑Base 6.7B` matches or beats `CodeLlama‑Base 34B` on many tasks (e.g., HumanEval‑X average 44.7% vs 41.0% in Table 3), highlighting data/recipe quality beyond scale.

## 5. Experimental Analysis
- Evaluation setup and metrics (Sec. 4):
  - Benchmarks:
    - Code generation: HumanEval and MBPP; multilingual HumanEval‑X across Python/C++/Java/PHP/TS/C#/Bash/JS (Sec. 4.1; Table 3).
    - Practical data‑science tasks: DS‑1000 across seven libraries (Table 4).
    - LeetCode Contest: 180 recent problems (Jul 2023–Jan 2024) with 100 tests each; zero‑shot prompting; Chain‑of‑Thought (CoT) variant examined (Sec. 4.1; Table 5).
    - FIM code completion: Single‑Line Infilling for Python/Java/JS, metric = line exact match (Sec. 4.2; Table 6).
    - Cross‑file completion: CrossCodeEval with exact match (EM) and edit similarity (ES), with/without BM25 retrieval (Sec. 4.3; Table 7).
    - Program‑aided math reasoning (`PAL`): GSM8K, MATH, GSM‑Hard, SVAMP, TabMWP, ASDiv, MAWPS solved by alternating natural language and Python (Sec. 4.4; Table 8).
  - Baselines: CodeGeeX2, StarCoder, CodeLlama families; Codex (`code‑cushman‑001`); GPT‑3.5‑Turbo and GPT‑4‑Turbo for instruct comparisons (Sec. 4).
  - Decoding: for HumanEval/MBPP, greedy decoding with matched scripts to ensure fairness (Sec. 4.1).

- Main quantitative results (selected):
  - HumanEval‑X and MBPP (Table 3):
    > `DeepSeek‑Coder‑Base 33B`: average 50.3% on HumanEval‑X and 66.0% on MBPP.  
    > Beats `CodeLlama‑Base 34B` by +9.3 points (50.3 vs 41.0) on HumanEval‑X average and +10.8 on MBPP (66.0 vs 55.2).  
    > `DeepSeek‑Coder‑Instruct 33B`: HumanEval‑X average 69.2%, exceeding `GPT‑3.5‑Turbo` 64.9%, though still below `GPT‑4` 76.5%.
  - DS‑1000 (Table 4):
    > `DeepSeek‑Coder‑Base 33B` achieves 40.2% average across libraries, outperforming `CodeLlama‑Base 34B` (34.3%). Gains are broad: e.g., NumPy 49.6% vs 42.7%, PyTorch 36.8% vs 25.0%.
  - LeetCode Contest (Table 5):
    > `DeepSeek‑Coder‑Instruct 33B` 27.8% overall Pass@1 (Easy 57.8, Medium 22.0, Hard 9.1), the only open model outperforming `GPT‑3.5‑Turbo` (23.3%). With CoT prompting, 28.9%. Still behind `GPT‑4‑Turbo` 40.6% (41.8% with CoT).
    > The paper flags possible data contamination in the earliest contest months but uses recent problems to minimize it (note under Table 5).
  - FIM infilling (Table 6):
    > `DeepSeek‑Coder‑Base 7B` mean 80.7% and `33B` mean 81.2%; both exceed `CodeLlama‑Base 13B` mean 75.5% and `StarCoder 16B` mean 69.7%.
  - Cross‑file completion (Table 7):
    > Without retrieval, `DeepSeek‑Coder‑Base 6.7B` has the best EM across all four languages among 7B‑scale peers. With retrieval, EM improves further—e.g., Python 16.14% vs 13.06% (StarCoder) and 13.02% (CodeLlama).  
    > Removing repo‑level pretraining reduces EM in Java/TS/C# (e.g., C#: 16.23% → 14.48%), directly evidencing the benefit of repository‑level data.
  - Program‑aided math (Table 8):
    > `DeepSeek‑Coder‑Base 33B` average 65.8% vs `CodeLlama‑Base 34B` 62.0%, with strong results on GSM8K 60.7% and MAWPS 93.3%.
  - Continued pretraining (Table 10):
    > `DeepSeek‑Coder‑Base‑v1.5 6.9B` improves non‑code reasoning (MMLU 49.1 vs 36.6; HellaSwag 69.9 vs 53.8) while maintaining similar code scores (HumanEval 43.2 vs 44.7; MBPP 60.4 vs 60.6).  
    > The instruct version gains even more on math/NL (e.g., GSM8K 72.6% vs 62.8%).

- Ablations and training curves:
  - FIM rate ablation (Fig. 3):  
    > 100% FIM maximizes HumanEval‑FIM but depresses HumanEval/MBPP completion; 50% PSM strikes a balance and beats `MSP@50%`.
  - Repo‑pretraining ablation (Table 7 last row): measurable drops w/o repo pretraining, validating the design.
  - Learning progress (Fig. 7): performance rises smoothly with tokens for all model sizes across multiple metrics, indicating effective scaling to 2T tokens.

- Do the experiments support the claims?
  - Yes for open‑model SOTA: across diverse tasks, `DeepSeek‑Coder` consistently tops open baselines of similar or larger size.  
  - Yes for repository‑level benefit: ablations and cross‑file benchmark substantiate it.  
  - Yes for FIM balance: ablation demonstrates the claimed trade‑off.

## 6. Limitations and Trade-offs
- Static dependency parsing (Sec. 2.2):
  - Uses regex over `import/using/include`. This misses dynamic imports, reflective loading, build‑generated code, and language‑specific module systems; dependency order can be incomplete in such projects.
- FIM trade‑off (Fig. 3):
  - High FIM ratios improve infilling but harm standard completion; the chosen 50% PSM is a compromise, not optimal for either extreme.
- Long‑context reliability (Sec. 3.6):
  - Although RoPE scaling implies up to 64K tokens theoretically, robust behavior is only claimed up to 16K after a short continuation phase.
- Compute and data demands (Sec. 3.4–3.5):
  - Training requires multi‑node A100/H800 clusters and 2T tokens; reproducing the full recipe is resource‑intensive.
- Evaluation caveats:
  - Potential residual data contamination for the LeetCode benchmark is acknowledged (Table 5 note).  
  - Greedy decoding is standardized but may under‑estimate peak performance relative to sampling-based strategies.
- Scope:
  - Focuses on code generation/completion and program‑aided math; does not study advanced tool use (e.g., compilers, static analyzers) during inference or long‑horizon refactoring workflows.

## 7. Implications and Future Directions
- Field impact:
  - Demonstrates that carefully prepared repository‑level corpora and balanced FIM can push open models to (and beyond) Codex/GPT‑3.5 levels on many code tasks, reshaping the open‑source baseline.
  - Provides reproducible configurations (sizes from 1.3B to 33B) with permissive licensing, likely accelerating IDE integration, educational tools, and research on cross‑file reasoning.
- Research enabled:
  - Retrieval‑augmented repository assistance: Table 7 shows further gains with BM25; future work could combine repository‑aware pretraining with learned retrievers or code graph retrieval.
  - Stronger long‑context behavior: build on the RoPE‑scaled 16K training to establish dependable 32–64K code understanding for monorepos and project‑wide audits.
  - Better dependency modeling: replace regex with language‑aware parsers or build systems; incorporate build graphs and symbol resolution for more faithful ordering.
  - Objective design: extend FIM beyond single middle span (multi‑span, syntax‑aware holes), or mix with execution‑guided objectives.
  - General‑LLM synergy: results from `v1.5` (Table 10) suggest that continuing from a general LLM improves reasoning without hurting coding; larger‑scale versions could merge strengths of general and code‑focused pretraining.
- Practical applications:
  - Repository‑wide code completion and refactoring assistance in IDEs, where cross‑file context and 16K windows are essential.
  - Data‑science notebooks: DS‑1000 gains indicate improved library usage (NumPy/Pandas/PyTorch/TensorFlow).
  - Automated patching and fill‑in tasks (FIM) for template expansion, code migration, and snippet completion across languages.

Overall, DeepSeek‑Coder is a methodical advance: it operationalizes repository‑level pretraining and a carefully tuned FIM objective to deliver strong, accessible code LLMs. The empirical study is broad and ablation‑backed, and the open release is positioned to catalyze both research and deployment in real software engineering environments.
