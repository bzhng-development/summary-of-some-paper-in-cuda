# Qwen2.5‑Coder Technical Report

**ArXiv:** [2409.12186](https://arxiv.org/abs/2409.12186)
**Authors:** Binyuan Hui, Jian Yang, Zeyu Cui, Jiaxi Yang, Dayiheng Liu, Lei Zhang, Tianyu Liu, Jiajun Zhang, Bowen Yu, Kai Dang, An Yang, Rui Men, Fei Huang, Xingzhang Ren, Xuancheng Ren, Jingren Zhou, Junyang Lin
**Institutions:** 

## 🎯 Pitch

Qwen2.5-Coder introduces a family of open-source models that revolutionize code-focused language modeling by integrating large-scale, balanced data mixtures and innovative repository-level pretraining methods. With robust performance in code generation, multilingual understanding, and long-context reasoning, these models provide developers and enterprises with high accuracy, open-ended integration, and unmatched versatility across diverse coding tasks, challenging even top proprietary systems.

---

## 1. Executive Summary (2-3 sentences)
Qwen2.5-Coder is a family of six open-source code-focused large language models (0.5B–32B parameters) that combine massive code-centric pretraining (5.2T tokens) with a carefully engineered instruction-tuning pipeline. The models achieve state-of-the-art open-source performance across code generation, completion, reasoning, editing, text-to-SQL, and long-context tasks, while preserving strong math and general-language abilities (e.g., 32B-Instruct reaches 92.7% HumanEval and 49.6% BigCodeBench-Instruct full; Tables 16, 12–14).

## 2. Context and Motivation
- Problem addressed
  - Open-source code LLMs lag behind top proprietary systems (e.g., GPT-4o, Claude 3.5) in accuracy, breadth of coding tasks, reasoning, and long-context understanding (Introduction).
  - Existing open models often specialize in Python, struggle with repository-level tasks, and sacrifice general or math skills when optimized for code.

- Why it matters
  - Practical: Developers need reliable assistants for multi-language coding, debugging, repository-level comprehension, and database interaction; enterprises need permissive licensing for integration.
  - Scientific: Understanding data mixtures and training strategies that jointly improve code, math, and general language is a core scaling and alignment question.

- Prior approaches and gaps
  - StarCoder/StarCoder2, CodeLlama, DeepSeek-Coder and derivatives focus primarily on file-level code pretraining; long-context support and multilingual instruction data are limited or uneven.
  - Instruction data quality and verification pipelines vary; many rely heavily on LLM-as-a-judge without grounded executors.

- Positioning
  - Qwen2.5-Coder builds on the Qwen2.5 base architecture, introduces repo-level pretraining up to 128K tokens, and deploys a multilingual, execution-verified instruction pipeline (Sections 2–4). It aims to match or exceed closed systems on code while preserving math/general capabilities.

## 3. Technical Approach
Step-by-step overview of the system design and training pipeline.

- Architecture and tokenizer (Section 2; Tables 1–2)
  - Six dense sizes: `0.5B, 1.5B, 3B, 7B, 14B, 32B`. The 32B model uses 64 layers, hidden size 5120, 40 query heads / 8 KV heads, intermediate size 27,648 (Table 1).
  - Special tokens enable code-aware training:
    - FIM tokens `<|fim_prefix|>, <|fim_middle|>, <|fim_suffix|>, <|fim_pad|>` for infilling.
    - Repo tokens `<|repo_name|>, <|file_sep|>` to encode repository structure (Table 2).

- Data and mixture design (Section 3.1–3.2)
  - 5.2 trillion tokens across five types (Section 3.1): Source code (GitHub repos, PRs, commits, notebooks, Kaggle), text-code grounding (docs/tutorials/blogs from Common Crawl), synthetic code (validated by an executor), math data (from Qwen2.5-Math), and general text (from Qwen2.5; code segments removed to avoid leakage).
  - Hierarchical filtering for text-code grounding uses simple, surface-feature models (e.g., fastText) across 4 stages. Each stage increases quality; Figure 1 shows iterative gains on HumanEval/MBPP for a 1.5B model.
  - Mixture ratio ablation (Table 3): a 70:20:10 split of Code:Text:Math outperforms 100% code-only and 85:10:5. Rationale: non-code data helps only beyond a threshold; final training uses 70/20/10.

- Three-stage training (Figure 2; Sections 3.2.1–3.2.2, 4.2)
  1) File-level pretraining (Section 3.2.1)
     - Sequence length 8,192; objectives: next-token prediction + FIM.
     - FIM format encourages the model to infer missing middle code:
       - “`<|fim_prefix|>{code_pre}<|fim_suffix|>{code_suf}<|fim_middle|>{code_mid}<|endoftext|>`” (Figure 3).
  2) Repo-level pretraining (Section 3.2.2)
     - Extends context to 32,768 tokens, alters RoPE base frequency to 1,000,000, and applies YARN (context-window extension technique) to enable 128K token inputs.
       - RoPE: rotary positional encoding; changing the base shifts frequency scaling for long-range attention.
       - YARN: a method to extrapolate context length without retraining from scratch.
     - 300B additional long-context code tokens; extends FIM to repository format with explicit file boundaries:
       - “`<|repo_name|>{repo}` … `<|file_sep|>{file_path}` … and a FIM gap within a file`” (Figure 4).
  3) Post-training (Section 4)
     - Supervised Fine-Tuning (SFT) and Direct Preference Optimization (DPO) (Figure 2, step ③).
     - Instruction data recipe (Section 4.1):
       - Language identification with fine-tuned CodeBERT; samples with little/no code removed to preserve code generation quality.
       - Instruction synthesis from GitHub code: generate an instruction from a snippet (≤1024 tokens) using an LLM, generate responses with a code LLM, filter with an LLM scorer; combine with open datasets (e.g., McEval-Instruct).
       - Multilingual multi-agent generation across ~100 languages: agents specialize by language, collaborate via cross-lingual discussions and knowledge sharing, maintain memory to avoid duplicates, and use a new “synergy” metric to measure cross-language transfer.
       - Checklist-based scoring: weighted evaluation of Q/A consistency, relevance, difficulty, presence/correctness/clarity of code and comments, and educational value; aggregate with `s = Σ w_i s_i`.
       - Multilingual sandbox for verification:
         - Static checking via AST parsing to reject syntactically invalid code.
         - Auto-generates unit tests per language; executes snippets in isolated environments; analyzes results with detailed pass/fail reports (Section 4.1).
     - Training policies (Section 4.2):
       - Coarse-to-fine SFT: start with large diverse low-quality data; then refine with high-quality data using rejection sampling and SFT on the best candidate answer per prompt.
       - Mixed tuning with FIM during SFT to maintain long-context completion: extract infill targets using `tree-sitter` AST nodes to infill realistic logic blocks within the same file.
       - DPO for code: build preference pairs using (i) executor-based correctness for self-contained algorithmic code and (ii) LLM-as-a-judge for complex cases; combine code preferences with common data for offline DPO.
         - DPO: a preference-learning method that trains a model to assign higher likelihood to preferred answers without explicit reward models.

- Decontamination (Section 5)
  - Remove any 10-gram word-level overlaps with major benchmarks (HumanEval, MBPP, GSM8K, MATH, etc.) across pretraining and post-training corpora to reduce test leakage.

## 4. Key Insights and Innovations
- Balanced data mixture at scale (Table 3; Section 3.1.2)
  - Novelty: systematic large-scale mixture ablation demonstrating that 70% code + 20% general text + 10% math yields better code and reasoning than code-only training.
  - Significance: shows general and math text improve coding when present above a threshold; the 7:2:1 ratio outperforms even higher code proportions.

- Repo-level FIM with explicit repository structure (Section 3.2.2; Figure 4; Table 2)
  - Novelty: extends FIM from file to repository with special tokens for repo/file boundaries, paired with 300B long-context tokens and 128K support via RoPE/YARN.
  - Significance: enables repository-aware completion and reasoning—critical for real-world cross-file dependencies and long-context understanding.

- Multilingual, verified instruction pipeline (Section 4.1)
  - Novelty: multi-agent multilingual generation with cross-lingual knowledge transfer; checklist-based scoring; and a multilingual sandbox that statically parses and executes code with auto-generated unit tests.
  - Significance: improves instruction data quality, correctness, and language coverage—key for strong multi-language generation, debugging, and text-to-SQL.

- Mixed SFT with AST-driven FIM plus DPO using execution and judgment feedback (Section 4.2)
  - Novelty: keeps long-context infilling skill active during instruction tuning and aligns preferences with grounded executor results.
  - Significance: yields strong code completion and reasoning gains while retaining long-context capability.

These go beyond incremental tweaks by combining long-context repo training, multilingual verified data synthesis, and a validated data mixture recipe that jointly preserve code, math, and general skills.

## 5. Experimental Analysis
- Evaluation design (Sections 6–7)
  - Base models: six aspects—code generation, completion, code reasoning, math reasoning, general language, long-context (Section 6).
  - Instruct models: code generation, reasoning, editing, text-to-SQL, math/general language, table understanding (Section 7).
  - Decontaminated training data (Section 5). Public evaluation code is provided.
  - Baselines: StarCoder2, CodeLlama, DeepSeek-Coder (V1/V2 and Lite), CodeStral; closed APIs for reference (Tables 4, 15–16).

- Main quantitative results
  - Code generation (Base; Table 5)
    - 7B Base surpasses larger dense models:
      > “Qwen2.5-Coder-7B: HumanEval 61.6, HumanEval+ 53.0, MBPP 76.9, MBPP+ 62.9, BigCodeBench-Complete Full 45.8 / Hard 16.2.”
      > “DeepSeek-Coder-33B-Base: 54.9 / 47.6 / 74.2 / 60.7 / 49.1 / 20.3.”
  - Multi-language generation (Base; Table 6)
    - Average Pass@1 across 8 languages:
      > “Qwen2.5-Coder-32B: 63.9; 14B: 59.9; 7B: 57.5,” all ahead of comparable baselines.
  - Code completion (Base)
    - Humaneval-FIM single-line infill (Table 7):
      > “32B: avg 88.3 (Py 81.5, Java 91.0, JS 89.4),” beating DS-33B-Base at 86.2.
    - CrossCodeEval cross-file completion (Table 8; avg EM/ES):
      > “32B: 57.1 / 86.8 vs DS-33B-Base 48.8 / 83.7.”
    - CrossCodeLongEval long context (Table 9; avg EM/ES):
      > “32B: 36.9 / 66.4,” best among compared models.
    - RepoEval repo-level (Table 10; avg EM/ES):
      > “32B: 51.6 / 78.5 vs DS-33B-Base 43.7 / 74.3; 14B and 7B also lead same-size peers.”
  - Code reasoning (Base; CRUXEval with CoT; Table 11):
    > “32B: Input-CoT 62.5, Output-CoT 69.4; 14B: 60.6 / 66.4; 7B: 56.5 / 56.0.”
  - Math and general language (Base; Tables 12–14)
    - Math (Table 12):
      > “32B: MATH 57.2, GSM8K 91.1, MMLU-STEM 75.1, TheoremQA 43.1,” demonstrating preserved math ability.
    - General (Table 13–14):
      > “32B MMLU Base/Pro/Redux: 79.1 / 50.4 / 77.5; ARC-Challenge 70.5, TruthfulQA 54.2, WinoGrande 80.8, HellaSwag 83.0.”
  - Long-context “Needle in the Code” (Figure 6)
    - Synthetic repo-level retrieval test shows successful exact recall across context lengths up to 128K.
  - Instruct code generation (Table 16)
    - HumanEval / MBPP (EvalPlus):
      > “32B-Instruct: 92.7 / 87.2 (HE/HE+), 90.2 / 75.1 (MBPP/MBPP+); 7B-Instruct: 88.4 / 84.1 and 83.5 / 71.7.”
    - BigCodeBench-Instruct:
      > “32B-Instruct: Full 49.6, Hard 27.0,” matching or surpassing many closed APIs (GPT‑4o Full 50.1 / Hard 25.0).
    - LiveCodeBench (2407–2409):
      > “32B-Instruct: Pass@1 31.4; 7B-Instruct: 18.2,” both competitive among open models (Table 16).
  - Multi-language (Instruct; Table 17)
    - Avg across 8 languages:
      > “32B-Instruct: 79.4; 14B-Instruct: 79.6; 7B-Instruct: 76.5,” near DS‑Coder‑V2‑Instruct at 79.9.
  - Code reasoning (Instruct; CRUXEval; Table 18)
    > “32B-Instruct: Input-CoT 75.2, Output-CoT 83.4,” exceeding DS‑Coder‑V2‑Instruct (70.0 / 75.1).
  - Code editing
    - Aider benchmark (Table 19):
      > “32B-Instruct: Pass@1 60.9, Pass@2 73.7; 7B-Instruct: 55.6 / 68.4,” surpassing open baselines and approaching strong closed models.
    - CodeEditorBench (Figure 11): overall win rate for 32B-Instruct comparable to DS‑Coder‑V2‑Instruct (86.2% win), despite much smaller parameter count (Section 7.3).
  - Text-to-SQL (Figure 12)
    > “32B-Instruct: BIRD 58.4, Spider 85.1,” outperforming comparable open models (e.g., DS‑33B‑Instruct 45.6 / 73.6).
  - Table understanding (Figure 13)
    > “32B-Instruct: 45.1 overall on TableBench,” best among open baselines compared.
  - Math/general (Instruct; Table 20)
    > “32B-Instruct: MATH 76.4, GSM8K 93.0, MMLU 77.6, MMLU‑Pro 62.3, IFEval 79.9,” indicating retention of broad skills.

- Ablations and robustness checks
  - Data mixture ablation (Table 3) and staged filtering (Figure 1) substantiate the mixture and cleaning choices.
  - Long-context synthetic “needle” test demonstrates capacity; repository-level completion benchmarks cross-check practical utility (Tables 8–10).

- Do the experiments support the claims?
  - Yes, across diverse benchmarks and sizes, Qwen2.5-Coder consistently matches or exceeds same-size and often larger open models, especially notable at 7B and 14B where it beats 20B–33B baselines on several tasks (Tables 5, 8–10, 16–19). Gains are not limited to Python; multi-language and repo-level tasks corroborate the design’s breadth.

## 6. Limitations and Trade-offs
- Assumptions and scope
  - Heavy reliance on synthetic and LLM-scored instruction data (Section 4.1)—although partially grounded by executors and static checks—can encode biases of the generating/judging models.
  - LLM-as-a-judge is used for complex preference pairs during DPO (Section 4.2); judgment quality and consistency are imperfect and may favor certain styles.

- Unaddressed scenarios
  - Real-world build/toolchain variance (dependency installation, environment configuration) is only partially captured by unit-test sandboxes; large monorepos with complex build systems may remain challenging.
  - Security and safety aspects for code execution (e.g., side effects) are not discussed in depth; the sandbox focuses on static parsing and isolated execution.

- Compute and scalability constraints
  - Training compute details are not reported (e.g., GPU hours), making it hard to assess efficiency vs. baselines.
  - 128K context is supported, but inference cost grows superlinearly with length; not all benchmarks directly stress 128K contexts beyond the synthetic needle test.

- Ablation coverage
  - Mixture ablation (Table 3) and filtering stages (Figure 1) are provided, but broader studies (e.g., proportions per programming language, impact of sandbox coverage per language) are not included.
  - Limited failure-mode analysis; some long-context/function-completion EM scores remain low across all models (Table 9), suggesting unsolved challenges in generating long multi-line code exactly matching references.

- Data access and reproducibility
  - While evaluation code is public, the full pretraining dataset is constructed from many sources with multiple filtering stages; reproducing the exact corpus may be difficult.

## 7. Implications and Future Directions
- How this changes the landscape
  - Demonstrates that open-source, dense models at 7B–32B can match or surpass larger open baselines and approach closed models on several code tasks, especially when pairing repo-level pretraining with verified, multilingual instruction data.
  - Validates a concrete data recipe—70/20/10 Code:Text:Math—showing non-code data boosts coding ability at scale (Table 3).
  - Pushes long-context coding forward by combining RoPE base adjustment, YARN, repo-level FIM, and 300B long-context tokens.

- Follow-up research enabled or suggested
  - Data mixture science: characterize thresholds and interactions between code, text, and math across tasks and sizes beyond the 7:2:1 ratio.
  - Execution-grounded alignment: scale the multilingual sandbox with richer test synthesis, dynamic environments, and security-aware policies; compare executor-based RL vs. offline DPO.
  - Long-context agents: evaluate 128K repo reasoning with real build/test pipelines and retrieval-augmented coding agents; study cost-accuracy trade-offs with sparse attention variants or MoE.
  - Cross-lingual transfer: expand multi-agent instruction generation and measure “synergy” quantitatively across more languages and domains.

- Practical applications and use cases
  - IDE assistants and copilots: strong at completion (Tables 7–10), repair (Table 19), and multi-language generation/editing (Tables 6, 17).
  - Repository-level comprehension: navigating large codebases with long-context inputs and repo-aware FIM (Section 3.2.2; Tables 8–10).
  - Data and analytics teams: text-to-SQL for enterprise databases (Figure 12), and table QA (Figure 13).
  - Education and assessment: math and general-language capabilities remain competitive (Tables 12–14, 20), supporting broader assistant roles.

Overall, Sections 3–4 present a replicable framework—data mixture, repo-level pretraining, multilingual verified instruction, and execution-grounded alignment—that explains the strong empirical results in Sections 6–7. The key takeaway is that careful data/process engineering, not just parameter count, is central to building practical, general-purpose code LLMs.
