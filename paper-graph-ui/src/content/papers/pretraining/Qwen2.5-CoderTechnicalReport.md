# Qwen2.5-Coder Technical Report

**ArXiv:** [2409.12186](https://arxiv.org/abs/2409.12186)

## 🎯 Pitch

Qwen2.5-Coder introduces a new family of six code-specialized large language models (0.5B–32B parameters), blending massive-scale code, mathematical, and natural language pretraining with innovative repository-level context modeling and rigorous, execution-verified instruction tuning. This results in state-of-the-art open-source performance across coding, reasoning, editing, and Text-to-SQL tasks—narrowing the gap with top proprietary models and enabling more reliable, capable, and general-purpose AI coding assistants for research and real-world development.

---

## 1. Executive Summary (2-3 sentences)
Qwen2.5-Coder is a family of six code-focused large language models (0.5B–32B parameters) built on the Qwen2.5 architecture and trained on 5.2T tokens (plus 300B long-context tokens) with a carefully balanced mix of code, math, and general text. The models introduce repo-level training and verification-heavy instruction tuning that together deliver state-of-the-art open-source performance on a wide range of code generation, completion, reasoning, editing, Text-to-SQL, and long-context tasks (see Tables 5–12, 16–20; Figures 5–13).

## 2. Context and Motivation
- Problem addressed
  - Open-source code LLMs lag behind top proprietary systems in code generation reliability, multi-language breadth, and long-context/repository-level understanding. This paper tackles how to train open models that close this gap while preserving general and math skills.
- Why it matters
  - Real-world coding assistants must: (a) follow complex instructions, (b) work across many programming languages, (c) reason about multi-file repositories, and (d) remain useful on math and general language tasks to understand specifications and documentation.
- Prior approaches and shortcomings
  - Strong open baselines exist (StarCoder2, CodeLlama, DeepSeek-Coder; Tables 4, 15). However, they often:
    - Emphasize file-level training with limited repository-level context.
    - Underinvest in balanced training mixtures that keep general/mathematical skills.
    - Use instruction data that is not sufficiently verified for executability or multi-language balance.
    - Offer shorter context windows, limiting repository-scale tasks.
- Positioning of this work
  - Builds on Qwen2.5 with a code-specific recipe:
    - Long-context pretraining up to 128K tokens via repository-level objectives (Section 3.2.2; Figure 4).
    - A data mixture that deliberately includes math and general text (Table 3).
    - A large, verified instruction-tuning pipeline with execution-based filtering and preference optimization (Sections 4.1–4.2).
    - New special tokens and Fill-in-the-Middle (FIM) formats for both file-level and repo-level training (Tables 2; Figures 3–4).

## 3. Technical Approach
Step-by-step overview of the system and training pipeline (Figure 2):

- Core architecture (Table 1)
  - Six dense transformer models: `0.5B`, `1.5B`, `3B`, `7B`, `14B`, `32B`; 64 layers at 32B.
  - Same tokenizer vocabulary (151,646) with added code-specialized tokens (Table 2).
  - Context length: trained at 8,192 tokens (file-level), extended to 32,768 and extrapolated to 128K (repo-level, Section 3.2.2).
  - Long-context mechanisms:
    - `RoPE` (rotary position embeddings) base increased from 10,000 to 1,000,000 to reduce decay over long spans.
    - `YARN` (context window extension technique) to extrapolate to 128K tokens (Section 3.2.2; Peng et al., 2023).

- Special tokens and training formats (Tables 2; Figures 3–4)
  - `FIM` = Fill-in-the-Middle, an infilling task where the model predicts a missing span given both sides of context.
  - File-level FIM format (Figure 3) and repo-level FIM format (Figure 4) use special tokens like `<|fim_prefix|>`, `<|fim_suffix|>`, `<|fim_middle|>`, plus `<|repo_name|>` and `<|file_sep|>` to delimit repository structure.
  - Purpose: teach the model to fill missing code within a file or a repository, improving completion and cross-file coherence.

- Three-stage training (Figure 2)
  1) File-level pretraining (Section 3.2.1)
     - Objective: next-token prediction + file-level FIM.
     - Data: 5.2T tokens; max sequence 8,192.
     - Goal: learn basic code/statements, idioms, and infilling within single files.
  2) Repo-level pretraining (Section 3.2.2)
     - Objective: repo-level FIM across multiple files using long-context code (≈300B tokens).
     - Context: 32,768 tokens, extrapolated to 128K with YARN.
     - Goal: learn cross-file dependencies, imports, and project structure.
  3) Post-training (instruction tuning + preference optimization; Sections 4.1–4.2)
     - Instruction data creation
       - Language identification with a fine-tuned `CodeBERT` to filter/retain mainstream programming languages and drop most samples with no real code (Section 4.1).
       - Unsupervised “instruction-from-code” synthesis from GitHub snippets: generate the prompt from code, then generate the answer with a code LLM; filter with an LLM scorer (Section 4.1).
       - Multilingual multi-agent synthesis for underrepresented languages (Section 4.1): language-specific agents collaborate, maintain memory to avoid duplicates, and distill knowledge across languages.
       - Checklist-based scoring for instruction pairs (Q/A consistency, difficulty, code presence, correctness, clarity, comments, educational value) combined into a weighted score s (Section 4.1).
       - Multilingual sandbox for code verification (Section 4.1): static checks via parsing to AST, and automatic unit-test generation/execution across languages (Python, Java, C++, JS). Only self-contained snippets are executed.
     - Training policy
       - Coarse-to-fine SFT: millions of diverse but lower-quality instructions first, then millions of high-quality with rejection sampling (Section 4.2).
       - Mixed tuning: to preserve long-context ability, some instruction samples are converted to FIM-style tasks using `tree-sitter` AST extraction to mask code blocks (Section 4.2).
       - Offline `DPO` (Direct Preference Optimization; Section 4.2): pairwise preferences from (a) sandbox execution results for algorithmic tasks and (b) LLM-as-judge for complex snippets. Preference signals from both code and non-code data are combined.

- Data strategy (Section 3.1; Table 3; Figure 1)
  - Components: Source code from GitHub (92 languages); Text–Code grounding data from Common Crawl; Synthetic code data validated by execution; Math data (from Qwen2.5-Math); General text (from Qwen2.5) with code removed (Section 3.1.1).
  - Quality control: hierarchical filtering of web data using small models (e.g., fastText) at multiple stages; later-stage survivors receive higher quality scores (Section 3.1.1; Figure 1).
  - Mixture selection: empirical study at 7B exploring ratios of Code:Text:Math (Table 3). The final choice is `70:20:10`, which improved code metrics over more code-heavy mixtures while boosting math/general abilities.

- Decontamination (Section 5)
  - 10-gram overlap filtering against common test sets (HumanEval, MBPP, GSM8K, MATH) for both pretraining and post-training corpora.

Definitions of potentially unfamiliar terms
- `FIM` (Fill-in-the-Middle): predict the missing code segment given both left and right contexts; improves completion and editing.
- `Repo-level FIM`: same idea but the context spans multiple files within a repository; trains cross-file consistency and tool-use style reasoning.
- `RoPE`: a positional encoding that enables attention to incorporate relative positions; changing its base widens long-range sensitivity.
- `YARN`: a method to extend the usable context window of a trained model without retraining from scratch on full-length sequences.
- `DPO`: a preference-learning method that tends to be more stable/efficient than reinforcement learning from human feedback.
- `AST` (Abstract Syntax Tree): a tree representation of source code structure used for static checks and targeted masking.

## 4. Key Insights and Innovations
- Balanced data mixture improves code, math, and general ability simultaneously
  - What’s new: a large-scale mixture tuned to `70% code : 20% text : 10% math` beats 100% code on code metrics (Table 3). The 7B model with 70:20:10 improves average scores across code and non-code tasks, suggesting math/text provide complementary patterns that help code generation.
  - Why it matters: most code LLMs focus on code-only corpora; this shows a principled way to retain broader reasoning/reading skills without sacrificing coding performance.

- Repo-level pretraining with explicit repository structure and FIM
  - What’s new: training with explicit repo metadata (`<|repo_name|>`, `<|file_sep|>`) and repo-level FIM over 300B tokens (Section 3.2.2; Figure 4) plus long-context scaling to 128K.
  - Why it matters: strong gains on cross-file completion and repository-level tasks (Tables 8–10) and success on a 128K synthetic needle-in-code task (Figure 6).

- Verified, multilingual instruction pipeline at scale
  - What’s new: a multilingual sandbox that statically parses code, auto-generates unit tests, and executes them to filter data, combined with multi-agent instruction synthesis and a checklist-based scorer (Section 4.1).
  - Why it matters: higher-quality instruction data leads to markedly stronger results on instruction-following benchmarks across code generation (Table 16), editing (Table 19), Text-to-SQL (Figure 12), and multilingual settings (Table 17, McEval in Figure 7, MdEval in Figure 8).

- Coarse-to-fine SFT with mixed FIM and DPO alignment
  - What’s new: start with diverse low-quality SFT to broaden coverage, then refine with high-quality SFT and DPO that uses executable signals and LLM judgment (Section 4.2).
  - Why it matters: contributes to best-in-class open-source performance for code assistants (Table 16; Figure 11), with robust reasoning (Table 18) and repository-aware completion (Tables 8–10).

These are fundamental design choices (mixture, repo-level objectives, verified supervision, long-context scaling) rather than minor parameter tweaks.

## 5. Experimental Analysis
- Evaluation setup
  - Base models and instruct models are evaluated separately across six competence areas (Sections 6–7): code generation, completion, reasoning, math, general language, and long-context. Public baselines include StarCoder2, CodeLlama, DeepSeek-Coder (Tables 4, 15), and closed APIs for context.
  - Key datasets and metrics
    - Code generation: HumanEval/MBPP and + versions via EvalPlus (pass@1); BigCodeBench Complete (Full/Hard); BigCodeBench Instruct; LiveCodeBench (Pass@1) (Tables 5, 16).
    - Multilingual generation: MultiPL-E across 8 languages (Table 6 for base; Table 17 for instruct); broader McEval (Figure 7).
    - Code completion: HumanEval-FIM EM; CrossCodeEval and CrossCodeLongEval with Exact Match (EM) and Edit Similarity (ES); RepoEval (Tables 7–10).
    - Code reasoning: CRUXEval with chain-of-thought (CoT) for input/output execution tracing (Tables 11, 18).
    - Editing: Aider (Pass@1/Pass@2) and CodeEditorBench (win rate) (Table 19; Figure 11).
    - Text-to-SQL: Spider and BIRD with standardized prompts (Figure 12).
    - Math and general: MATH, GSM8K, MMLU-STEM, TheoremQA; MMLU/Base-Pro-Redux; ARC, TruthfulQA, WinoGrande, HellaSwag (Tables 12–14, 20).
    - Long-context: 128K “Needle in the Code” synthetic repo task (Figure 6).

- Main quantitative results (selected highlights; all numbers trace to the cited tables/figures)
  - Base models—code generation (Table 5)
    - `Qwen2.5-Coder-7B` surpasses larger open models: HumanEval 61.6 vs DS-Coder-33B 54.9; BigCodeBench (Full) 45.8 vs DS-Coder-33B 49.1 (close) and Hard 16.2 vs 20.3.
    - `Qwen2.5-Coder-32B` reaches top-tier open-source: HumanEval 65.9, MBPP 83.0, BigCodeBench (Full/Hard) 53.6/26.4.
  - Base models—multilingual generation (Table 6)
    - `Qwen2.5-Coder-7B` average 57.5 across 8 languages; `32B` reaches 63.9 with ≥60% in five languages.
  - Base models—code completion (Figure 5; Tables 7–10)
    - Humaneval-FIM Average EM: `32B` 88.3 vs DS-Coder-33B 86.2 (Table 7).
    - CrossCodeEval Average EM/ES: `32B` 57.1/86.8—SOTA; `7B` rivals >20B models (Table 8).
    - CrossCodeLongEval Average EM/ES: `32B` 36.9/66.4—SOTA; chunk completion EM 57.3 (Table 9).
    - RepoEval Average EM/ES: `32B` 51.6/78.5—SOTA; line completion EM 76.1 (Table 10).
  - Base models—code reasoning (Table 11)
    - `32B` CRUXEval Input-CoT/Output-CoT: 62.5/69.4; `14B` 60.6/66.4; `7B` 56.5/56.0.
  - Base models—math and general (Tables 12–14)
    - `32B` MATH 57.2, GSM8K 91.1, MMLU-STEM 75.1 (Table 12).
    - `32B` MMLU Base/Pro/Redux 79.1/50.4/77.5; strong general benchmarks (Table 13). ARC 70.5, HellaSwag 83.0 (Table 14).
  - Instruct models—code generation (Table 16)
    - `Qwen2.5-Coder-7B-Instruct`: HumanEval 88.4; MBPP 83.5; BigCodeBench Instruct Full/Hard 41.0/18.2; LiveCodeBench 18.2—consistently above peers of similar size.
    - `Qwen2.5-Coder-14B-Instruct`: HumanEval 89.6; MBPP 86.2; BigCodeBench 48.4/22.2; LiveCodeBench 23.4.
    - `Qwen2.5-Coder-32B-Instruct`: HumanEval 92.7, MBPP 90.2; BigCodeBench 49.6/27.0; LiveCodeBench 31.4. On LiveCodeBench it approaches but does not surpass GPT‑4o‑2024‑08‑06 (34.6) and remains far from o1-mini (60.0).
  - Instruct models—multilingual and debugging
    - MultiPL-E (8 languages): `32B-Instruct` average 79.4 vs DS‑Coder‑V2‑Instruct 79.9; `14B-Instruct` 79.6 > DS‑Coder‑33B‑Instruct 69.2 (Table 17).
    - McEval (40 languages): `32B-Instruct` leads open models across many languages (Figure 7).
    - MdEval (debugging): `32B-Instruct` comparable or better than larger models (Figure 8).
  - Instruct models—reasoning and editing
    - CRUXEval CoT: `7B-Instruct` 65.8/65.9; `32B-Instruct` 75.2/83.4—well above other open models (Table 18).
    - Aider: `7B-Instruct` Pass@1 55.6; `32B-Instruct` Pass@1/Pass@2 60.9/73.7—competitive with closed GPT‑4o‑2024‑08‑06 (56.8/74.4) and below Claude‑3.5‑20241022 (71.4/86.5) (Table 19).
    - CodeEditorBench: `32B-Instruct` overall win rate comparable to DS‑Coder‑V2‑Instruct (Figure 11).
  - Text-to-SQL and table understanding
    - BIRD/Spider exact match: `32B-Instruct` 58.4/85.1—best among open code models tested (Figure 12).
    - TableBench TCoT: `32B-Instruct` overall 45.1—best among compared open models (Figure 13).

- Ablations and diagnostics
  - Data mixture ablation (Table 3) shows 70:20:10 outperforms code-only and 85:10:5 mixtures on both code and non-code benchmarks.
  - Text–Code data filtering iterations improve 1.5B validation (HumanEval/MBPP) from ~41.6% to ~46.8% (Figure 1), evidencing the value of hierarchical filtering.
  - Long-context diagnostic (Figure 6) shows success on 128K “Needle in the Code.”

- Do the experiments support the claims?
  - Yes for open-source leadership across many coding tasks: consistent improvements against StarCoder2, CodeLlama, DeepSeek-Coder families at similar sizes (Tables 5–12, 16–19).
  - The long-context design correlates with repository-level completion gains (Tables 8–10) and the 128K diagnostic (Figure 6).
  - Where results are mixed: LiveCodeBench (Table 16) shows `32B-Instruct` is close to GPT‑4o but still behind the strongest closed models (o1 family); BigCodeBench Hard sometimes shows small gaps relative to DS-Coder variants.

> Example result: “Qwen2.5-Coder-32B-Instruct achieves HumanEval 92.7 and MBPP 90.2; BigCodeBench-Instruct 49.6 (Full) / 27.0 (Hard); LiveCodeBench 31.4” (Table 16).

## 6. Limitations and Trade-offs
- Assumptions and data choices
  - The 70:20:10 mixture is validated primarily at the 7B scale (Table 3); the transfer of this optimum to larger models is assumed rather than exhaustively ablated.
  - Hierarchical filtering leans on small classifiers (fastText-like; Section 3.1.1). While efficient, surface-level filters may discard nuanced but valuable content or introduce topical biases.
  - Synthetic instruction data and LLM-as-judge labels (Sections 4.1–4.2) risk propagating biases or stylistic preferences of the teacher models.
- Scope and edge cases not fully addressed
  - Long-context evaluation uses a synthetic “Needle in the Code” diagnostic (Figure 6). It validates capacity to recall across long sequences but does not fully replicate complex repository maintenance tasks (refactoring, dependency resolution, build systems).
  - Multilingual breadth is demonstrated across 8 languages (MultiPL-E) and broader sets (McEval/MdEval figures), but coverage across 92 GitHub languages is not uniformly reported and long-tail performance is less certain.
- Computational demands
  - Training on 5.2T + 300B tokens with long-context settings implies substantial compute and memory cost; inference at 128K contexts is also resource-heavy. The report does not detail compute budgets or latency trade-offs.
- Evaluation considerations
  - Decontamination uses 10-gram overlap (Section 5), which is strong but not infallible against paraphrased leakage.
  - LiveCodeBench performance, while strong for open-source, trails leading closed systems (Table 16), indicating headroom on competitive programming and OOD generalization.
- Open questions
  - How much each component (repo-level FIM vs. YARN vs. instruction sandbox vs. DPO) contributes individually is only partially illuminated by the provided ablations.

## 7. Implications and Future Directions
- How this work changes the landscape
  - Provides a reproducible recipe for high-capability open-source code LLMs:
    - Balanced data mixing that retains math/general strengths (Table 3).
    - Repo-aware training formats plus long-context scaling to 128K (Figures 3–4, 6).
    - Verified, multilingual instruction pipelines with executable filtering and preference optimization (Sections 4.1–4.2).
  - Demonstrates that well-engineered 7B–14B models can surpass prior 20B–33B models on many code tasks (Tables 5–12, 16–19), lowering the barrier to deployment.
- Suggested follow-up research
  - More granular ablations: quantify contributions of repo-level FIM, data mixture ratios at larger scales, and each post-training component.
  - Richer long-context evaluation: real repositories with builds, tests, and dependency graphs; agent-style tasks (bug localization, multi-file refactoring).
  - Safety and robustness: adversarial code prompts, security-sensitive completions, and license compliance in code generation.
  - Retrieval-augmented coding: couple repo-level models with retrieval or symbolic tools for large monorepos.
  - Data governance: open benchmarks and methodology for verifying multilingual code quality at scale.
- Practical applications
  - IDE assistants with strong autocomplete and cross-file understanding (Tables 7–10).
  - Automated code repair and collaborative editing tools (Table 19; Figure 11).
  - Data engineering assistants: SQL generation (Figure 12) and table QA (Figure 13).
  - Education: multi-language tutoring with verified exercises via the sandbox.
  - Agents for repository maintenance: summarize diffs, implement features, and run/interpret tests in long contexts.

> Scaling trend: Figure 14 shows monotonic gains across sizes on MBPP-3shot (base) and LiveCodeBench (instruct), reinforcing that the pipeline scales effectively.

Overall, the paper delivers a clear, mechanism-rich training recipe—balanced mixture, repo-level objectives, verified instruction tuning, and long-context scaling—that meaningfully advances open-source code LLM capabilities across generation, reasoning, editing, and tool-using tasks.
