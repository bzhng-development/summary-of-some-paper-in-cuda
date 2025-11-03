# DeepSeek-Coder: When the Large Language Model Meets Programming - The Rise of Code Intelligence

**ArXiv:** [2401.14196](https://arxiv.org/abs/2401.14196)

## 🎯 Pitch

DeepSeek-Coder introduces an open-source series of large language models for programming, uniquely trained on 2 trillion tokens of curated, repository-level code and equipped with advanced infilling capabilities and a 16K-token context window. By capturing inter-file dependencies and enabling powerful ‘fill-in-the-middle’ code completion, DeepSeek-Coder not only achieves or exceeds state-of-the-art performance among open models, but even surpasses closed models like GPT-3.5 on key code benchmarks, democratizing advanced code intelligence for research and industry alike.

---

## 1. Executive Summary (2-3 sentences)
DeepSeek-Coder introduces an open-source family of code-focused language models (`1.3B`, `6.7B`, `33B` parameters) trained from scratch on 2 trillion tokens and designed to work at the scale of whole software repositories. It combines repository-level pretraining data organization with a “fill-in-the-middle” infilling objective and a long 16K-token context window, achieving state-of-the-art performance among open models and, in several code-generation benchmarks, surpassing closed models like GPT‑3.5 (see Table 3 and Table 5).

## 2. Context and Motivation
- Problem addressed:
  - Open models for code lag behind closed models on code generation, completion, and cross-file understanding (Introduction §1). Most existing open models are trained on file-level code, ignoring inter-file dependencies common in real projects (§2.2).
  - Many code LLMs have short context windows, limiting their ability to reason over multi-file repositories (§3.6). Infilling capabilities (e.g., inserting code in the middle of a file) are also underdeveloped without specialized training (§3.1.2).
- Why it matters:
  - Practical coding assistance often requires understanding multi-file dependencies (imports/includes), library usage, and editing in the middle of existing code. Improving open models in these areas expands access to strong code intelligence for research and commercial use (Abstract; license note).
- Prior approaches and gaps:
  - File-level pretraining on large code corpora (e.g., StarCoder, CodeLlama) achieves good single-file generation but struggles with cross-file tasks (§2.2) and often uses shorter contexts.
  - Infilling has been explored (e.g., InCoder, CodeGen2.5) but trade-offs between infilling and left-to-right completion remain unclear (§3.1.2, Fig. 3).
- Positioning:
  - DeepSeek-Coder contributes both technical methods and an evaluation showing open models can close much of the gap to proprietary systems, including outperforming GPT‑3.5‑Turbo on several code tasks (Table 3, Table 5).

## 3. Technical Approach
The work combines a repository-aware data pipeline, dual pretraining objectives, a long-context architecture, and an instruction-tuning stage.

- Data pipeline (Section 2; Figure 2):
  - Composition: 87% source code, 10% English code-related text (GitHub Markdown, StackExchange), 3% unrelated Chinese language to strengthen Chinese understanding (§2; footnote to StackExchange).
  - Rule-based filtering (§2.1): Removes low-quality or data-heavy files using heuristics (e.g., overly long lines, low alphabetic character ratio; HTML visible text ≥20%; JSON/YAML length within [50, 5000] chars), cutting raw data to 32.8% of original size.
  - Dependency parsing and ordering (§2.2): Within each repository, files are ordered so that dependencies appear before dependents (e.g., `import` in Python, `include` in C). This is implemented via a customized topological sort (Algorithm 1). Unlike standard topological sort, it repeatedly picks the node with minimal in-degree to deal gracefully with cycles (e.g., mutual imports), then concatenates files in that order. A comment with the file path is inserted at each file’s start to preserve path context.
    - Definition: A (modified) `topological sort` orders nodes in a directed graph so edges go forward; using “minimal in-degree” selection offers a heuristic when cycles exist.
  - Repository-level deduplication (§2.3): Instead of deduplicating per file (which can break repo structure), they concatenate a repository into a single sample and apply near-duplicate removal to the whole sample. This retains architectural integrity across files.
  - Quality screening and decontamination (§2.4): Filters by compiler checks, a quality model, and n-gram matching against test sets (HumanEval, MBPP, GSM8K, MATH) to reduce leakage. For overlap ≥10-gram matches (or exact matches for 3–9 grams), samples are removed.
  - Scale: 798 GB, 603M files across 87 languages (Table 1).
- Objectives and training strategy (Section 3):
  - Objective 1 — `Next-token prediction` (§3.1.1): Standard left-to-right language modeling on packed sequences.
  - Objective 2 — `Fill-in-the-Middle (FIM)` (§3.1.2): Randomly split each document into `Prefix`, `Middle`, `Suffix` and train the model to generate the `Middle` given both sides.
    - Two arrangements studied: PSM (Prefix–Suffix–Middle) and SPM (Suffix–Prefix–Middle). DeepSeek adopts PSM with a 50% rate after ablations (Fig. 3).
    - Implementation: Special tokens `<|fim_start|>`, `<|fim_hole|>`, `<|fim_end|>` wrap `Prefix` and `Suffix`; the target is the `Middle` (§3.1.2).
    - Trade-off: 100% FIM maximizes infilling accuracy but hurts standard completion; 50% PSM balances both (Fig. 3).
  - Tokenizer (§3.2): 32K BPE vocabulary.
- Model architecture and optimization:
  - Models: `1.3B`, `6.7B`, `33B` decoder-only Transformers with RoPE positional encoding (§3.3).
  - Efficiency:
    - `GQA` (Grouped-Query Attention; group size 8) in the 33B model to reduce compute and memory (§3.3).
    - `FlashAttention v2` for faster attention computation (§3.3).
  - Hyperparameters (Table 2): e.g., 33B uses hidden size 7168, 62 layers, 56 heads; max LR 3.5e‑4; batch sizes scale with model.
  - Optimization (§3.4): AdamW (β1=0.9, β2=0.95), a three-stage LR schedule with 2000 warmup steps; each stage’s LR is reduced by a factor of √(1/10), and the final LR is 10% of the initial.
  - Training environment (§3.5): A100/H800 clusters with NVLink/NVSwitch, InfiniBand networking; HAI‑LLM framework with tensor, pipeline, and ZeRO data parallelism.
- Long context (§3.6):
  - Extending RoPE context via linear scaling (increase scaling factor from 1→4; base frequency 10,000→100,000), followed by 1000 additional steps at sequence length 16K and batch size 512. Theoretical capacity is up to 64K tokens, but outputs are empirically most reliable up to 16K.
  - Definitions: `RoPE` (Rotary Position Embedding) encodes token positions by rotating query/key vectors; `scaling` adjusts frequencies to support longer ranges.
- Instruction tuning (§3.7):
  - Builds `DeepSeek-Coder-Instruct` from the base models using high-quality instruction data in Alpaca format. Uses a special delimiter `<|EOT|>` for dialog turns, cosine LR schedule (warmup 100 steps), LR 1e‑5, trained on ~2B tokens with 4M-token batch sizes.
  - Figure 4 shows a multi-turn example (snake game with subsequent scoring HUD addition) running without errors, illustrating multi-turn capability.
- “Continue pretraining” variant (`v1.5`) (§5; Table 9 and 10):
  - Starts from a general language model (`DeepSeek-LLM-7B`), then continues with 2T tokens (70% code; mixture of code-related NL, math NL, and bilingual NL) using only next-token prediction at 4K context. `v1.5` significantly improves math and natural language tasks while slightly reducing code scores (Table 10).

## 4. Key Insights and Innovations
- Repository-level data construction and ordering (§2.2–§2.3; Algorithm 1):
  - What’s new: Training samples are entire repositories ordered by dependency, not isolated files. Deduplication happens at the repository level to keep structure intact.
  - Why it matters: Improves cross-file reasoning and code completion where understanding imports and inter-file calls is essential. Table 7 shows consistent gains over 7B peers and a drop when pretraining without repository-level structure.
- Balanced FIM training with evidence-based rate selection (§3.1.2; Fig. 3):
  - What’s new: A careful ablation on FIM modes (PSM vs MSP) and rates (0%, 50%, 100%) shows 50% PSM best balances infilling and left-to-right completion. MSP did not outperform PSM in their setup.
  - Why it matters: Many code assistants need both “insert here” and “continue typing” behaviors; the chosen policy improves practical utility across tasks.
- Long-context adaptation via RoPE scaling (§3.6):
  - What’s new: A lightweight method (adjust RoPE scales and a short additional training) yields robust behavior up to 16K tokens. Theoretical 64K is possible, but reliability is reported highest at 16K.
  - Why it matters: Enables repository-level prompts and broader context windows within the same model family, addressing a common limitation of code LLMs.
- Strong open-source baselines with instruction-tuned variants that compete with closed models (Tables 3 and 5):
  - What’s new: `DeepSeek-Coder-Instruct-33B` surpasses GPT‑3.5‑Turbo on multilingual HumanEval and MBPP (Table 3) and on the LeetCode Contest benchmark overall (Table 5).
  - Why it matters: Demonstrates that open models can reach or exceed closed-system quality on realistic code tasks, with permissive licensing for research and commercial use (Abstract).

## 5. Experimental Analysis
- Evaluation methodology and datasets:
  - Code generation:
    - Multilingual HumanEval and MBPP (Table 3; §4.1): HumanEval (zero-shot; 164 Python problems extended to 8 languages) and MBPP (few-shot; 500 Python tasks). Metric: `Pass@1` (first attempt success).
    - DS-1000 (Table 4; §4.1): 1,000 realistic data-science tasks across Matplotlib, NumPy, Pandas, SciPy, Scikit-Learn, PyTorch, TensorFlow. Metric: `Pass@1`.
    - LeetCode Contest benchmark (Table 5; §4.1): 180 recent problems (Jul 2023–Jan 2024) with 100 test cases per problem; prompts include the problem statement and a Python template. Evaluated with and without `CoT` (“write a step-by-step outline then write the code”).
  - Code completion:
    - FIM single-line infilling benchmark (Table 6; §4.2) across Python, Java, JavaScript; metric: exact match of the filled line.
    - Cross-file completion with CrossCodeEval (Table 7; §4.3): Languages: Python, Java, TypeScript, C#. Evaluation uses max seq length 2048, output 50 tokens, cross-file context 512 tokens via BM25 retrieval (official). Metrics: `Exact Match (EM)` and `Edit Similarity (ES)`.
      - Definition: `BM25` is a standard term-frequency based retrieval method for ranking documents by relevance.
  - Program-based math reasoning with PAL (§4.4; Table 8): Benchmarks include GSM8K, MATH, GSM-Hard, SVAMP, TabMWP, ASDiv, MAWPS. The prompt alternates NL reasoning and code execution to arrive at answers.
- Main quantitative results:
  - Multilingual HumanEval and MBPP (Table 3):
    - `DeepSeek-Coder-Base-33B`: 50.3% average Pass@1 on HumanEval across 8 languages; 66.0% on MBPP.
      - Improvement: On average, +9.3 pts vs CodeLlama-34B’s 41.0% (HumanEval avg) and +10.8 pts vs 55.2% (MBPP).
    - Instruction-tuned:
      - `DeepSeek-Coder-Instruct-33B`: HumanEval avg 69.2% vs GPT‑3.5‑Turbo 64.9%; MBPP 70.0% vs GPT‑3.5‑Turbo 70.8% (roughly parity).
  - DS-1000 (Table 4):
    - `DeepSeek-Coder-Base-33B`: 40.2% average, best among open baselines (CodeLlama-34B at 34.3%). Strong per-library scores (e.g., NumPy 49.6%, SciKit-Learn 40.0%, TensorFlow 46.7%).
  - LeetCode Contest (Table 5):
    - `DeepSeek-Coder-Instruct-33B`: 27.8% overall Pass@1; beats GPT‑3.5‑Turbo (23.3%) but below GPT‑4‑Turbo (40.6%).
    - CoT improves medium/hard subsets for 33B (25.3% medium, 11.4% hard vs 22.0% and 9.1% without CoT), modest overall gain to 28.9%.
    - Quote:
      > “DeepSeek-Coder-Instruct 33B is the only open-sourced model that outperforms OpenAI’s GPT‑3.5‑Turbo in this task” (Table 5).
    - Caveat: Potential data contamination is acknowledged for some months despite efforts to use recent contests (§4.1).
  - FIM single-line infilling (Table 6):
    - `DeepSeek-Coder-Base-7B/33B` average 80.7%/81.2% vs CodeLlama-13B at 75.5% and StarCoder at 69.7%. The 1.3B model already matches or beats larger baselines, pointing to strong pretraining data quality.
  - Cross-file completion (Table 7):
    - Without retrieval: `DeepSeek-Coder-Base-6.7B` leads peers in EM/ES across languages.
    - With retrieval: Further gains; e.g., Python EM 16.14% (vs CodeLlama-7B+retrieval 13.02%). Ablation “w/o Repo Pre-training” lowers EM across 3 of 4 languages, confirming repository-level construction benefits.
  - Program-aided math reasoning (Table 8):
    - `DeepSeek-Coder-Base-33B` averages 65.8% across 7 datasets; e.g., GSM8K 60.7%, MATH 29.1% (notably strong for a code model).
  - v1.5 continued-pretraining (Table 10):
    - Base 6.9B vs Base 6.7B: Math/NL scores rise substantially (e.g., GSM8K 62.4% vs 43.2%; MMLU 49.1% vs 36.6%), with minimal code drop (HumanEval 43.2% vs 44.7%).
- Ablations and training dynamics:
  - FIM rate ablation (Fig. 3):
    - 100% FIM → best single-line infilling but worst code completion; 50% PSM best trade-off. MSP did not outperform PSM in their experiments.
  - Repo-level pretraining ablation (Table 7): Removing repo-level construction reduces cross-file EM.
  - Training curves (Fig. 7): Show stable improvements over 2T tokens on HumanEval, MBPP, and infilling validation for all model sizes.
- Assessment of evidence:
  - The breadth of tasks (single-file, infilling, cross-file, DS workflows, competitive programming, and program-based math) provides converging evidence that the methods improve both practical and structured code abilities.
  - Where results are mixed or conditional:
    - Instruction-tuned models rival or beat GPT‑3.5 on some tasks but remain behind GPT‑4‑Turbo (Table 5).
    - Long-context is “theoretically” 64K but “reliable” at 16K (§3.6), indicating partial success.

## 6. Limitations and Trade-offs
- Data and contamination:
  - Despite decontamination (§2.4), the paper notes possible contamination in LeetCode months (§4.1). Real-world code corpora also inevitably include noisy or license-sensitive material; the paper reports permissive sources but noise remains (hence compiler/quality filters).
- Objective trade-offs:
  - FIM at 100% hurts left-to-right completion (Fig. 3). The chosen 50% PSM is a compromise; different applications might prefer different settings.
- Long-context:
  - Although RoPE scaling enables theoretical 64K contexts, the most reliable behavior is at 16K (§3.6). Extremely long repository prompts may still challenge the model.
- Cross-file retrieval setup:
  - CrossCodeEval uses BM25 with a cap of 512 tokens of retrieved context (§4.3). This evaluates the LLM under a specific retrieval regime; different retrieval pipelines might yield different absolute numbers.
- Compute and accessibility:
  - Training from scratch on 2T tokens with 33B parameters, plus long-context adaptation, requires significant compute (A100/H800 clusters; §3.5). This may limit replication or rapid iteration for smaller labs.
- General language vs. code specialization:
  - Base models are code-optimized. The v1.5 variant improves language/math but may slightly reduce some code metrics (Table 10), hinting at a specialization–generalization trade-off.

## 7. Implications and Future Directions
- Field impact:
  - Demonstrates that open, repository-aware pretraining with balanced FIM can close much of the gap to closed models on realistic code tasks, including competitive programming. This shifts expectations for what open models can deliver and provides methods (repo-level ordering, FIM balance, RoPE scaling) other groups can adopt.
- Practical applications:
  - IDE assistants for multi-file projects (cross-file completion improved in Table 7).
  - Code review and maintenance tools that need to insert code segments precisely (FIM results in Table 6).
  - Data-science copilots with stronger library usage (DS-1000 results in Table 4).
  - Education and programming pedagogy, where step-by-step reasoning plus coding (PAL) can be scaffolded (Table 8).
- Research directions:
  - Longer and more reliable contexts beyond 16K (extend §3.6, e.g., continued pretraining at long lengths, position interpolation variants).
  - Richer repository modeling: integrate static analysis, build graphs, or typed ASTs to deepen cross-file understanding beyond regex-based dependencies (§2.2).
  - Dynamic retrieval and memory: learn to select cross-file context better than BM25; evaluate with larger retrieval budgets.
  - Objective mixtures: explore adaptive schedules that vary FIM rates across training to optimize both infilling and continuation.
  - Robust decontamination and benchmark freshness: standardized, versioned, and continuously refreshed test suites (the paper’s LeetCode benchmark is a step here; §4.1).
  - Multilingual depth: Table 3 shows strong multilingual HumanEval; extending repo-level construction to non-Python ecosystems (e.g., Java/TypeScript build systems) could further raise cross-file scores.

> In sum, DeepSeek-Coder’s core technical package—repository-ordered pretraining, balanced FIM, and long-context RoPE scaling—translates to consistent empirical gains across a wide spectrum of code tasks (Tables 3–8), with strong open-source accessibility and a clear roadmap for building even more capable repository-scale code models.
