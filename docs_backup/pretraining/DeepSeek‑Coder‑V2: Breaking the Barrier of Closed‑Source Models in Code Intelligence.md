# DeepSeek‑Coder‑V2: Breaking the Barrier of Closed‑Source Models in Code Intelligence

**ArXiv:** [2406.11931](https://arxiv.org/abs/2406.11931)
**Authors:** DeepSeek‑AI, Qihao Zhu, Daya Guo, Zhihong Shao, Dejian Yang, Peiyi Wang, Runxin Xu, Y. Wu, Yukun Li, Huazuo Gao, Shirong Ma, Wangding Zeng, Xiao Bi, Zihui Gu, Hanwei Xu, Damai Dai, Kai Dong, Liyue Zhang, Yishi Piao, Zhibin Gou, Zhenda Xie, Zhewen Hao, and others
**Institutions:** 

## 🎯 Pitch

DeepSeek-Coder-V2 introduces a groundbreaking open-source MoE model, supporting 338 programming languages and a 128K-token context, that closes the performance gap with top closed-source models in coding and math. By leveraging a new, expansive corpus and targeted RL alignment, it revolutionizes code intelligence, enabling developers to tackle complex, multilingual, and long-context programming tasks with unprecedented accuracy and efficiency.

---

## 1. Executive Summary
DeepSeek-Coder-V2 is an open-source code-focused language model family (16B and 236B parameters in a Mixture-of-Experts design) that is continued-pretrained on 6 trillion additional tokens to close the performance gap with top closed-source models on coding and math. It supports 338 programming languages and 128K-token context, and reaches or surpasses leading proprietary systems on several code and math benchmarks while largely preserving general-language ability (Figure 1; Sections 1, 3.4, 4).

## 2. Context and Motivation
- Problem/gap addressed
  - Open-source code models (e.g., StarCoder, CodeLlama, DeepSeek-Coder) improve code intelligence but still trail closed-source systems like GPT-4-Turbo, Claude 3 Opus, and Gemini 1.5 Pro on coding and math tasks (Introduction).
  - Existing open-source models typically support fewer languages, shorter contexts, and lack robust instruction-following for real-world software engineering (Sections 1, 5 Conclusion).

- Why this matters
  - Practical impact: Developers need models that can understand large repositories, multiple languages, and perform accurate code generation/fixes with reproducible alignment to tests and compilers.
  - Scientific significance: Demonstrates that open-source MoE models, optimized with large-scale continued pretraining and targeted RL alignment, can rival or beat commercial systems on core code/math tasks (Figure 1; Tables 3–9).

- Prior approaches and shortcomings
  - Previous open-source code models trained from scratch on code-heavy corpora (DeepSeek-Coder) or on curated code datasets (StarCoder, CodeLlama) achieved strong but not top results and supported fewer languages and shorter contexts (Introduction; Section 4 baselines).
  - Instruction-following and robustness in long-context programming scenarios remained limited.

- Positioning
  - Builds on the general DeepSeek-V2 model by continued pretraining with a new code/math corpus and alignment targeted at coding correctness. Adds 338-language coverage and 128K context via YARN while keeping general abilities (Sections 1, 2, 3.4, 3.5).

## 3. Technical Approach
This section explains how the model is built, trained, and aligned.

- Model family and Mixture-of-Experts (MoE)
  - Two sizes: `DeepSeek-Coder-V2-Lite` (16B total, `2.4B active parameters`) and `DeepSeek-Coder-V2` (236B total, `21B active parameters`) (Table 2).
  - MoE overview: in MoE, only a subset of expert networks is activated for each token. `Active parameters` are the parameters used in a single forward pass; they are much fewer than total parameters, making inference more efficient.
  - Architecture follows DeepSeek-V2; they reverted from “exponential normalization” to conventional normalization to avoid gradient spikes during training (Section 3.2).

- Data pipeline and continued pretraining
  - Starting point: an intermediate DeepSeek-V2 checkpoint already trained on 4.2T tokens (Section 3.3).
  - Additional 6T tokens (for a total of 10.2T exposures) with composition: 60% source code, 10% math, 30% natural language (Sections 1, 2, 3.3).
  - Code data: 1,170B code-related tokens from GitHub and CommonCrawl across 338 languages; filtering removes long-line/noisy files and non-code heavy data (e.g., JSON/YAML length constraints); near-deduplication reduces duplicates (Section 2 Data Collection).
  - Web mining for code/math text uses a fastText classifier trained on curated seed domains and a BPE tokenizer to improve recall on non-space-delimited languages; iterative expansion labels domains as code/math if >10% pages are collected, then re-crawls to grow the set (Section 2).
  - Math data: 221B math tokens collected similarly (Section 2).
  - Ablation (Table 1) with a 1B model shows the new code corpus improves HumanEval from 30.5% to 36.0% with 1T tokens; extending to 2T tokens lifts HumanEval to 37.2% and MBPP to 54.0%.

- Training objectives and code-completion ability
  - Objective: next-token prediction for both sizes; the 16B model additionally trains with `Fill-In-the-Middle (FIM)` at 50% rate using `PSM (Prefix, Suffix, Middle)` format to enable infilling (Section 3.1). Example sequence:
    ```
    <|fim_begin|> prefix <|fim_hole|> suffix <|fim_end|> middle <|eos_token|>
    ```
  - The 236B model does not use FIM during pretraining (Section 3.1).

- Long-context extension
  - Uses `YARN` to extend rotary embeddings to 128K tokens; hyperparameters: scale s=40, α=1, β=32 (Section 3.4).
  - Two-stage long-context training: 1,000 steps at 32K (batch 1152), then 1,000 steps at 128K (batch 288). Long-context data is upsampled during this phase (Section 3.4).
  - `NIAH (Needle-In-A-Haystack)` test: retrieves a “needle” string from different depths across context windows up to 128K; results show strong retrieval across all lengths (Figure 2).

- Alignment: supervised fine-tuning (SFT) and reinforcement learning (RL)
  - SFT dataset mixes ~20k coding and ~30k math instructions (from DeepSeek-Coder and DeepSeek-Math) plus general instructions (from DeepSeek-V2), totaling ~300M tokens; cosine schedule with 100 warmup steps, lr 5e-6, 1M tokens batch, 1B total tokens (Section 3.5.1).
  - RL uses `GRPO (Group Relative Policy Optimization)`, which compares groups of responses to compute relative advantages and avoids a separate critic, reducing cost (Section 3.5.2).
  - Prompt set: ~40k coding/math prompts, each coding prompt carries test cases (Section 3.5.2).
  - Reward modeling for code: instead of using raw compiler 0/1 pass signals, they train a `reward model` on compiler-derived data to smooth noise and improve generalization. Figure 3 shows higher LeetCode Pass@1 when using reward-model feedback versus raw compiler signals.
  - For math, the reward uses ground-truth answers (Section 3.5.2).
  - FIM is also used during fine-tuning of the 16B model to preserve completion/infilling capability post-alignment (Section 1.5 and 3.5.1 note).

## 4. Key Insights and Innovations
- Scaling open-source MoE code models without losing efficiency
  - Contribution: The paper releases an open-source 236B-parameter MoE code model with only 21B active parameters and a 16B model with 2.4B active parameters, enabling practical inference while adding capacity (Table 2).
  - Why it matters: This is presented as the “first attempt” at an open-source hundred-billion-parameter code model (Section 1.1). It shows MoE can scale code intelligence while remaining usable.

- A larger, cleaner, and broader code+math corpus demonstrably helps
  - Contribution: A new corpus (1,170B code tokens across 338 languages + 221B math) and continued pretraining on 6T tokens improve coding and math substantially (Section 2).
  - Evidence: Table 1 ablation with a 1B model shows the new corpus and longer training raise HumanEval and MBPP materially (up to 37.2% HE and 54.0% MBPP with 2T tokens).

- Long-context code understanding up to 128K with YARN, validated by retrieval stress tests
  - Contribution: Extends context length from 16K to 128K and trains in two stages with upsampled long-context data (Section 3.4).
  - Evidence: `NIAH` heatmap shows strong retrieval across all tested depths and lengths up to 128K (Figure 2).
  - Significance: Enables repository-level reasoning and large-file tasks often needed in real software engineering.

- Reward-model-driven RL for code correctness using GRPO
  - Contribution: Instead of relying solely on noisy compiler pass/fail signals, a learned reward model provides finer-grained feedback for RL (Section 3.5.2).
  - Evidence: Figure 3 shows consistent gains in LeetCode Pass@1 when using the reward model vs. compiler signals.

- Infilling-first training for the small model to keep code-completion strong
  - Contribution: The 16B model uses FIM during both pretraining and alignment to optimize for completion/infilling workflows (Sections 3.1, 3.5.1).
  - Evidence: On single-line infilling, the 16B `V2-Lite-Base` matches much larger models with a mean 86.4% accuracy (Table 6).

## 5. Experimental Analysis
- Evaluation methodology
  - Benchmarks (code generation and reasoning)
    - HumanEval and MBPP+ with greedy decoding and standardized scripts; HumanEval expanded to multiple languages to test multilingual coding (Section 4.1; Table 3).
    - LiveCodeBench (new problems Dec 2023–June 2024) and USACO programming contest problems, to test competitive programming (Section 4.1; Table 4). To avoid contamination, only LiveCodeBench questions after the training cutoff are used.
  - Code completion
    - RepoBench v1.1 (December 2023 subset only) with 2k–16k contexts; exact-match on the first non-empty, non-comment line; prompts truncated to 15,800 tokens; greedy decoding with max 64 output tokens (Section 4.2.1; Table 5).
    - Single-line FIM infilling (Python/Java/JS) with line exact match (Section 4.2.2; Table 6).
  - Code fixing and understanding
    - Defects4J (single-method subset), SWE-bench, and Aider editing tasks (Section 4.3; Table 7).
    - CRUXEval forward/backward code reasoning with chain-of-thought (Section 4.4; Table 8).
  - Mathematics
    - GSM8K, MATH, AIME 2024, and Math Odyssey. Results mostly from greedy decoding; AIME also reports `maj@64` (majority vote over 64 samples) resulting in up to 5/30 solved; zero-shot chain-of-thought prompting with a “step-by-step and box the final answer” instruction (Section 4.5; Table 9 and footnote).
  - General language ability
    - Standard reasoning/knowledge benchmarks and open-ended evaluations (BBH, MMLU, ARC, TriviaQA, NQ, AGIEval; Chinese CLUEWSC, C-Eval, CMMLU; Arena-Hard, AlpacaEval 2.0, MT-Bench, AlignBench) (Section 4.6; Table 10).

- Main quantitative results (selected highlights, with direct citations)
  - Code generation across languages
    - > Table 3: `DeepSeek-Coder-V2-Instruct` achieves 90.2% on HumanEval (Python) and 76.2% on MBPP+; its average across languages is 75.3%, just behind GPT-4o’s 76.4%, and ahead of all other open-source models compared.
  - Competitive programming
    - > Table 4: On LiveCodeBench, `DeepSeek-Coder-V2-Instruct` reaches 84.1%, close to GPT-4-Turbo-0409 (84.1%) and GPT-4o (87.4%). On USACO, its overall score is 43.4%, tied with GPT-4o and just behind GPT-4-Turbo-0409 (45.7%).
  - Completion and infilling
    - > Table 5: On RepoBench, the 16B `V2-Lite-Base` (2.4B active params) is comparable to or better than earlier 7B–33B models in Python/Java exact match (e.g., 38.9% Python avg, 43.3% Java avg).
    - > Table 6: On single-line FIM tasks, `V2-Lite-Base` scores 86.4% mean, on par with 33B models.
  - Code fixing
    - > Table 7: `DeepSeek-Coder-V2-Instruct` achieves 21.0% on Defects4J, 12.7% on SWE-bench, and 73.7% on Aider (the highest Aider score among all listed models, including GPT-4 variants and Claude/Gemini).
  - Code understanding and reasoning
    - > Table 8: `DeepSeek-Coder-V2-Instruct` gets 70.0% on CRUXEval-I-COT and 75.1% on CRUXEval-O-COT, leading among open-source models compared but trailing GPT-4-class systems.
  - Mathematics
    - > Table 9: `DeepSeek-Coder-V2-Instruct` reaches 94.9% (GSM8K), 75.7% (MATH), solves 4/30 AIME 2024 with greedy (and up to 5/30 with maj@64), and 53.7% Math Odyssey—comparable to or better than closed-source baselines on some metrics (e.g., AIME 2024).
  - General language ability
    - > Table 10: Compared to DeepSeek-V2 Chat, `DeepSeek-Coder-V2 Instruct` maintains strong general performance (e.g., MMLU 79.2% vs 78.1%) and excels on reasoning-heavy Arena-Hard (65.00 vs 41.60).

- Do the experiments support the claims?
  - The broad benchmark suite spans code generation, completion, fixing, reasoning, long-context retrieval (NIAH), math, and general language. Results consistently place `DeepSeek-Coder-V2-Instruct` at or near the top among open-source models, and competitive with proprietary systems on several fronts (Figure 1; Tables 3–9).
  - The ablation in Table 1 credibly demonstrates that the new code corpus and longer training yield measurable gains even on a small model.
  - The reward-model RL result (Figure 3) justifies the RL design choice by quantifying Pass@1 gains over compiler-only feedback.

- Mixed or conditional results and trade-offs
  - SWE-bench remains challenging: `12.7%` for `V2-Instruct` is below GPT-4o (`26.7%`) (Table 7), suggesting limitations in complex, multi-file, multi-step changes.
  - On knowledge-heavy open-domain QA (TriviaQA, NQ), `V2 Instruct` is slightly behind `V2 Chat` (Table 10), likely due to less web data focus during continued pretraining (Section 4.6 commentary).
  - The 236B model does not use FIM, so its post-alignment infilling capability is not emphasized the way it is for the 16B model (Section 3.1).

## 6. Limitations and Trade-offs
- Instruction-following in complex software scenarios
  - The paper explicitly notes a remaining “significant gap in instruction-following” relative to state-of-the-art models, which affects challenging tasks like SWE-bench (Conclusion; Table 7).

- Reward-model dependence
  - RL relies on a learned reward model for code correctness. While Figure 3 shows gains over raw compiler signals, reward models can encode biases or misgeneralize in under-tested code regions, especially if test coverage is partial (Section 3.5.2).

- Data composition choices
  - Continued pretraining emphasizes code and math (60% and 10%) with only 30% general language—Table 10 shows slight regressions vs DeepSeek-V2 Chat on some knowledge-heavy QA benchmarks (TriviaQA, NQ), suggesting a trade-off (Section 4.6).

- FIM asymmetry across sizes
  - Only the 16B model is trained with FIM. The 236B model may not be as optimized for infilling workflows out-of-the-box (Section 3.1).

- Compute and scalability
  - Training exposure is 10.2T tokens and the larger model has 236B total parameters—even with 21B active, inference and training remain resource-intensive (Table 2; Section 3.3).

- Evaluation scope
  - While long-context retrieval is validated by NIAH (Figure 2), there is limited report of end-to-end, large-repository editing performance under 128K-token contexts beyond RepoBench and SWE-bench.

## 7. Implications and Future Directions
- Field impact
  - Demonstrates that open-source MoE models, when continued-pretrained on large, diverse code+math corpora and aligned with test-aware rewards, can closely track or surpass closed-source systems on many code/math metrics (Figure 1; Tables 3–9). This meaningfully narrows the open-source vs. closed-source gap for code intelligence.

- Practical applications
  - Multi-language coding assistants and IDE integration (338 languages), especially for infilling/completion (Table 6).
  - Long-context code understanding for repository-level navigation, refactoring, and documentation due to 128K context and validated long-context retrieval (Figure 2; Section 3.4).
  - Math tutoring or technical Q&A where precise reasoning is required (Table 9).

- Research directions enabled or suggested
  - Stronger instruction-following for multi-file, multi-step code changes: integrate richer task decomposition, tool-use, and environment interaction to lift SWE-bench and real-world issue resolution (Conclusion; Table 7).
  - Reward design: combine compiler/test feedback with static analysis, coverage-guided tests, and human preference models to reduce reward hacking and improve generalization (Section 3.5.2).
  - Broaden knowledge: rebalance or augment web/general data to recover open-domain QA without sacrificing code/math strengths (Table 10).
  - FIM for large models: explore FIM or structured infilling objectives for the 236B model to unify top-tier generation with completion/editing use cases (Sections 3.1, 4.2.2).
  - Long-context reasoning: beyond NIAH, create and evaluate on repository-scale tasks that require cross-file reasoning, dependency tracking, and stateful editing at 128K tokens.

Overall, DeepSeek-Coder-V2 provides a compelling, open, and practically usable advance in code intelligence: a large MoE model family with strong multilingual coding, long-context handling, and competitive math reasoning, backed by a careful data pipeline, explicit ablations (Table 1), and targeted RL alignment (Figure 3).
