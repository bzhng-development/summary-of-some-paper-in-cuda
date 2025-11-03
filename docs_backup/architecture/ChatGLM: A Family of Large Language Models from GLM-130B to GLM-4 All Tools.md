# ChatGLM: A Family of Large Language Models from GLM-130B to GLM-4 All Tools

**ArXiv:** [2406.12793](https://arxiv.org/abs/2406.12793)
**Authors:** Team GLM, Aohan Zeng, Bin Xu, Bowen Wang, Chenhui Zhang, Da Yin, Diego Rojas, Guanyu Feng, Hanlin Zhao, Hanyu Lai, Hao Yu, Hongning Wang, Jiadai Sun, Jiajie Zhang, Jiale Cheng, Jiayi Gui, Jie Tang, Jing Zhang, Juanzi Li, Lei Zhao, Lindong Wu, Lucen Zhong, Mingdao Liu, Minlie Huang, Peng Zhang, Qinkai Zheng, Rui Lu, Shuaiqi Duan, Shudan Zhang, Shulin Cao, Shuxun Yang, Weng Lam Tam, Wenyi Zhao, Xiao Liu, Xiao Xia, Xiaohan Zhang, Xiaotao Gu, Xin Lv, Xinghan Liu, Xinyi Liu, Xinyue Yang, Xixuan Song, Xunkai Zhang, Yifan An, Yifan Xu, Yilin Niu, Yuantao Yang, Yueyan Li, Yushi Bai, Yuxiao Dong, Zehan Qi, Zhaoyu Wang, Zhen Yang, Zhengxiao Du, Zhenyu Hou, Zihan Wang
**Institutions:** Tsinghua University / THUDM (Team GLM)

## 🎯 Pitch

ChatGLM-4 introduces a groundbreaking LLM that bridges high-quality Chinese-English bilingual alignment with the ability to handle 128K token contexts and autonomously use tools like Python, web browsing, and image generation. This innovation impacts enterprises and developers by providing a versatile solution for complex document processing and real-world tasks, addressing a critical gap in AI capabilities beyond English-centric models.

---

## 1. Executive Summary
This paper presents `ChatGLM`, an evolving family of large language models, and documents the fourth generation, `GLM-4`, including the `GLM-4 All Tools` agentic variant. It addresses two practical gaps—high-quality bilingual (Chinese/English) alignment and robust long‑context + tool‑use capability—by combining improved pretraining (≈10T tokens), targeted alignment (SFT + RLHF), architecture refinements, and an agent pipeline that plans, calls tools, and integrates feedback.

## 2. Context and Motivation
- Problem/gap addressed
  - Building LLMs that are:
    - Competitive with leading models on core reasoning, math, coding, and knowledge benchmarks.
    - Strongly aligned for Chinese and English, where many high-end models are English‑centric.
    - Practical for long‑context use (128K–1M tokens) and autonomous tool use (web browsing, Python, image generation, function calls).
  - Evidence of the gap appears throughout the paper’s evaluation focus on Chinese alignment, long‑context, and tool use (Sections 3.2–3.8; Tables 3–9).

- Why it matters
  - Real-world impact: Enterprises and end users increasingly need LLMs that can read long documents, follow complex bilingual instructions, and retrieve or compute via tools. Tool‑use agents (browser, code interpreter, function calls) are critical for accurate, up‑to‑date, and verifiable outputs (Figure 2, Figure 4).
  - Theoretical/technical significance: Demonstrates methods to scale context windows to 128K–1M tokens with maintained performance (LongBench‑Chat in Table 5), and alignment practices that raise instruction following and Chinese safety/quality (IFEval Table 3, AlignBench Table 4, SafetyBench Table 10).

- Prior approaches and their limitations
  - State-of-the-art instruction-tuned LLMs (GPT‑4 family, Claude, Gemini) excel in English. The paper highlights a relative lack of Chinese-first alignment and comprehensive agent integration that is openly documented.
  - Existing long-context recipes often extend context window limits but can degrade utility without specialized alignment. `LongAlign` is introduced as a remedy (Section 2, “The context length of our models…”; [1]).

- Positioning
  - The GLM line began with `GLM-130B` (blank‑infilling pretraining objective) and progressed through open `ChatGLM-6B/2/3` generations to `GLM‑4`. Figure 1 (timeline) and Figure 3 (evolution and MMLU gains) situate `GLM-4` among prior systems: it emphasizes bilingual alignment, long‑context competence, and integrated tool use.

## 3. Technical Approach
The paper’s approach spans data, architecture, alignment, and an agent system.

- Pretraining data pipeline (Section “Pre‑Training Data”)
  - Sources: multilingual web pages, Wikipedia, books, code, research papers; predominantly Chinese and English.
  - Three stages:
    - Deduplication (exact + fuzzy) to boost diversity.
    - Filtering to remove noisy/offensive/placeholder/code‑only web pages.
    - Tokenization: byte‑level BPE learned separately for Chinese and multilingual text; merged with `cl100k_base` tokens into a unified 150,000‑token vocabulary.
  - Reweighting: “increase the importance of high-quality and educational sources like books and Wikipedia.”
  - Scale: “around ten trillion tokens.”
  - Why these choices: A larger, cleaner, and bilingual-heavy corpus is needed to raise performance and alignment in both Chinese and English; a unified 150k vocab reduces tokenization inefficiencies across languages.

- Model architecture (Section “Architecture”)
  - Foundations: Transformer with several pragmatic changes:
    - `No Bias Except QKV`: drops biases elsewhere to speed training and (observationally) “slight improvement in length extrapolation.”
    - `RMSNorm` and `SwiGLU`: replace LayerNorm and ReLU for better performance.
    - `RoPE` extended to 2D: to accommodate GLM’s 2D positional encoding.
    - `GQA` (Group Query Attention): reduces KV cache at inference time. Because GQA decreases attention parameters, FFN parameters are increased (“set dffn to 10/3 of hidden size”) to keep total model capacity balanced.
  - Context extension strategy: from 2K (ChatGLM) → 32K (ChatGLM2/3) → 128K/1M (GLM‑4), achieved by
    - Position encoding extension techniques [31; 5].
    - Continual training on long text [47].
    - Long‑context alignment (`LongAlign`) to preserve instruction-following and QA quality at long lengths (Section 2 and [1]).

- Alignment pipeline (Section “Alignment”)
  - `SFT` (Supervised Fine‑Tuning): Uses “authentic human prompts and interactions” (not templates) to align base models with human intent.
  - `RLHF` (Reinforcement Learning from Human Feedback): Mitigates refusal, bilingual token mixing, safety concerns, and multi‑turn coherence problems.
  - Data curation: Mix of in-house annotation and proprietary third‑party data; raters score outputs on safety, factuality, relevance, helpfulness, and human preference.
  - Why SFT + RLHF: SFT gives initial alignment; RLHF further tunes behavior to human preferences and safety at deployment.

- Enhancement techniques developed around GLM (Section “ChatGLM Techniques”)
  - `LongAlign`: recipe for long‑context alignment, crucial for 128K context performance matching leading models on LongBench‑Chat (Table 5).
  - `Self‑Contrast`: feedback‑free preference data generation for alignment by self‑producing negatives.
  - `ChatGLM‑Math`: self‑critique to improve math reasoning without manual curation.
  - `AgentTuning`: instruction‑tuning on agent‑environment trajectories to improve tool/agent behaviors.
  - `APAR`: trains models to plan hierarchical outputs that can be generated in parallel at decoding time (inference speed for hierarchical structures).

- `GLM-4 All Tools` agent pipeline (Figure 4; Section “GLM‑4 All Tools”)
  - Components and flow:
    1) Analyze the user task and “Plan” a step‑by‑step solution.
    2) Decide whether tools are needed; if so, call tools in sequence: web browser, Python interpreter, text‑to‑image (CogView), or user‑defined functions/APIs.
    3) Use tool outputs as intermediate feedback; iterate (“Recursive Execute”).
    4) Maintain “Memory” for multi‑turn coherence and context.
  - Example: Figure 2 demonstrates browsing for population data and then using the Python interpreter to compute CAGR.
  - Why this design: Autonomy in tool choice + iterative planning/feedback enables complex, verifiable tasks that require external data or computation.

## 4. Key Insights and Innovations
- Long‑context alignment that preserves utility up to 128K (and an experimental 1M)
  - What’s new: A full stack—positional extension + continual long‑text training + `LongAlign`.
  - Significance: In Table 5, `GLM‑4 (0520)` reaches 87.3 (English) and 84.0 (Chinese) on LongBench‑Chat, matching GPT‑4 Turbo (English 85.0–87.2) and surpassing it on Chinese (82.1), and near Claude 3 Opus (87.7, 82.7).
  - This goes beyond just increasing context length; it demonstrates retained instruction-following and QA quality at long lengths.

- Bilingual alignment with strong Chinese capabilities
  - What’s new: Pretraining emphasizes Chinese/English with a unified 150k vocab, plus alignment data and procedures focused on Chinese.
  - Significance: On AlignBench (Table 4), `GLM‑4 (0520)` achieves the highest overall score 8.00, surpassing GPT‑4 Turbo (7.90) and Claude 3 Opus (7.53), with standout “Logic Reasoning” and “Language” categories.

- Practical, autonomous multi‑tool agent
  - What’s new: An alignment regime and application framework where the model autonomously plans and chooses tools (browser, Python, image model, and user-defined functions), integrating results across steps (Figure 4).
  - Significance: Table 9 reports near‑parity with ChatGPT‑4 on Python‑based math (e.g., GSM8K 91.59 vs 92.72) and strong browsing performance (“Information Seeking” 78.08 vs 67.12).

- Efficient inference‑oriented architecture choices
  - What’s new: Combining `GQA` (smaller KV cache), “No Bias Except QKV,” `RMSNorm` + `SwiGLU`, and 2D `RoPE`, while rebalancing capacity via larger FFN.
  - Significance: These are not just incremental tweaks; they target inference efficiency and length extrapolation without sacrificing overall capacity, underpinning the long‑context and agent use cases.

## 5. Experimental Analysis
- Evaluation design (Section 3)
  - Breadth:
    - Academic benchmarks: `MMLU`, `GSM8K`, `MATH`, `BBH`, `GPQA`, `HumanEval` (Table 2).
    - Instruction following: `IFEval`, in English and a Chinese translation (Table 3).
    - Alignment quality in Chinese: `AlignBench v1.1` (Table 4).
    - Long context: `LongBench‑Chat` with English/Chinese splits (Table 5).
    - Coding on real prompts: `NaturalCodeBench (NCB)`, bilingual Python/Java (Table 6).
    - Function calling: `Berkeley Function Calling Leaderboard` (Table 7).
    - Agentic tasks: `AgentBench` (7 environments) (Table 8).
    - Tool‑use system: internal evaluation vs ChatGPT‑4 for browsing and code‑interpreter tasks (Table 9).
    - Safety: `SafetyBench` (Chinese subset) (Table 10).
  - Setup details:
    - LongBench‑Chat uses GPT‑4 as an LLM‑as‑judge with few‑shot prompts and multiple runs averaged (Section 3.4).
    - IFEval scoring scripts are adjusted for Chinese (Section 3.2).

- Main quantitative results
  - Base capabilities (Table 2)
    - > “GLM‑4 (0520) MMLU 83.3, GSM8K 93.3, MATH 61.3, BBH 84.7, GPQA 39.9, HumanEval 78.5.”
    - Relative positioning:
      - Against GPT‑4 (0314): GLM‑4 trails on MMLU (83.3 vs 86.4) but exceeds on GSM8K (93.3 vs 92.0), BBH (84.7 vs 83.1), GPQA (39.9 vs 35.7), HumanEval (78.5 vs 67.0).
      - Against GPT‑4 Turbo (2024‑04‑09): GLM‑4 lags on most metrics (e.g., MATH 61.3 vs 73.4; HumanEval 78.5 vs 88.2).
    - Takeaway: competitive with earlier GPT‑4; somewhat behind latest GPT‑4 Turbo and Claude 3 Opus on some tasks, especially math.

  - Instruction following (Table 3)
    - > English strict instruction‑level: “GLM‑4 (0520) 85.0 vs GPT‑4 Turbo (2024‑04‑09) 85.9.”
    - > Chinese strict instruction‑level: “GLM‑4 (0520) 78.0 vs GPT‑4 Turbo (2024‑04‑09) 79.1.”
    - Takeaway: near‑parity with GPT‑4 Turbo; strong bilingual instruction adherence.

  - Chinese alignment quality (Table 4)
    - > Overall: “GLM‑4 (0520) 8.00 vs GPT‑4 Turbo (2024‑04‑09) 8.00, GPT‑4 Turbo (1106) 7.90, Claude 3 Opus 7.53.”
    - Category strengths: Logic (7.95) and Language (8.00) are competitive/best-in-class.
    - Takeaway: one of the strongest Chinese‑aligned models.

  - Long‑context performance (Table 5)
    - > English: “GLM‑4 (0520) 87.3 vs GPT‑4 Turbo (1106) 87.2 and Claude 3 Opus 87.7.”
    - > Chinese: “GLM‑4 (0520) 84.0 vs GPT‑4 Turbo (2024‑04‑09) 82.1 and Claude 3 Opus 82.7.”
    - Takeaway: matches top models in English and leads in Chinese, suggesting effective `LongAlign`.

  - Coding with real user prompts (NCB, Table 6)
    - > Overall: “GLM‑4 (0520) 47.1 vs GPT‑4 Turbo (2024‑04‑09) 53.8 and Claude 3 Opus 48.3.”
    - Takeaway: close to Claude 3 Opus; still behind GPT‑4 Turbo on practical coding.

  - Function calling (Table 7)
    - > Overall: “GLM‑4 (0520) 81.76; GPT‑4 Turbo (2024‑04‑09) 81.24; GPT‑4o (2024‑05‑13) 82.94.”
    - > Execution summary: “GLM‑4 (0520) 84.17; GLM‑4‑9B‑Chat 87.92.”
    - Takeaway: function‑call ability is on par with GPT‑4 Turbo; interestingly, the 9B chat variant performs very strongly on execution‑based scoring.

  - AgentBench (Table 8)
    - > Overall: “GLM‑4 (0520) 3.79 vs GPT‑4 Turbo (1106) 3.77; Claude 3 Opus 3.62.”
    - Strengths: Database, House‑Holding, Web Shopping; room to grow on Operating System, Knowledge Graph, and Lateral Thinking Puzzles.
    - Takeaway: credible agent performance against strong baselines.

  - All‑tools system (Table 9)
    - > Python Interpreter: “GSM8K 91.59 vs GPT‑4 92.72; MATH 63.60 vs 65.00.”
    - > Browser: “Information Seeking 78.08 vs GPT‑4 67.12.”
    - Takeaway: comparable Python‑math tool use; noticeably stronger browsing in the reported setup.

  - Safety (Table 10)
    - > Overall: “GLM‑4 (0520) 87.2 vs GPT‑4 (0613) 89.7; GPT‑4 Turbo (2024‑04‑09) 87.9; Claude 3 Opus 87.5.”
    - Weakest relative area: Physical Health.
    - Takeaway: competitive safety profile, slightly below GPT‑4 family overall.

- Convincingness of evidence
  - Strengths:
    - Broad, multi‑axis evaluation with clear baselines and numbers (Tables 2–10).
    - Chinese and long‑context performance are convincingly strong.
    - Function‑calling and agentic evaluations show practical capability.
  - Caveats:
    - Several evaluations use LLM‑as‑judge (e.g., LongBench‑Chat) which can introduce bias.
    - Tool‑use comparisons (Table 9) are “first‑hand test” style rather than a standardized public benchmark; still informative but not definitive.

- Ablations, failures, robustness
  - The paper surveys techniques (e.g., `LongAlign`, `Self‑Contrast`, `ChatGLM‑Math`) but does not include detailed ablations isolating each method’s contribution inside `GLM‑4`.
  - Failure modes are suggested by weaker math (Table 2 vs GPT‑4 Turbo), some agent environments (Table 8), and slightly lower safety on Physical Health (Table 10).

## 6. Limitations and Trade-offs
- Data and transparency
  - While the corpus size and processing steps are described, detailed composition (dataset identities, license status, and exact mixing ratios) and training compute are not disclosed. This limits reproducibility and fair apples‑to‑apples comparison.

- Bilingual focus and generalization
  - The system is “pre-trained on ten trillions of tokens mostly in Chinese and English” and “aligned primarily for Chinese and English usage” (Abstract). Performance in other languages is not deeply evaluated here; the paper notes only “a small set of corpus from 24 languages.”

- Math reasoning still trails top models
  - `MATH` and (to a lesser extent) `GSM8K` remain below the latest GPT‑4 Turbo (Table 2), indicating room for improved reasoning strategies or data.

- Evaluation dependencies
  - Some evaluations rely on LLM‑as‑judge and internal testing (e.g., LongBench‑Chat scoring via GPT‑4; Table 9 browsing/solver tests). These are informative but can be sensitive to the judge model and prompts.

- Model availability and scaling
  - The strongest `GLM‑4` models are accessible via APIs; the open releases are `GLM‑4‑9B` (128K) and `GLM‑4‑9B‑Chat‑1M` (experimental) rather than the largest variants. This is common in the field but may constrain external replication studies.

- Agentic complexity and safety
  - The `All Tools` system increases power and autonomy but also the attack surface (e.g., prompt injection during browsing). The paper covers safety metrics (Table 10) and data curation, but real‑world agent safety remains a challenging, ongoing area.

## 7. Implications and Future Directions
- How this work changes the landscape
  - Demonstrates that a bilingual‑first, long‑context, tool‑integrated LLM can reach near‑SOTA performance across core tasks while leading on Chinese alignment and competitive on agent tasks. This expands the set of viable, well‑documented alternatives to English‑centric flagships.
  - The open `GLM‑4‑9B` line—supporting 128K (and experimental 1M) contexts and tool use—lowers the barrier for research and applications that need long‑document processing and agent capabilities.

- Follow‑up research enabled or suggested
  - Long‑context methods: Beyond `LongAlign`, study retrieval‑augmented and memory mechanisms specialized for 128K–1M tokens; analyze cost/quality trade‑offs for hybrid chunking + long-context models.
  - Reasoning and math: Deepen `ChatGLM‑Math`-style self‑critique, combine with tool‑augmented proof checkers and formal solvers, and add targeted synthetic data to close the MATH gap to GPT‑4 Turbo.
  - Alignment without human labels: Scale `Self‑Contrast` and compare to DPO/DPO‑variants with rigorous ablations to quantify preference‑model data quality vs. downstream gains.
  - Agent safety and reliability: Systematic defenses against prompt injection and tool misuse; provenance and citations for browser outputs; sandboxing and resource governance for Python execution.
  - Multilingual extension: Expand high‑quality alignment data and evaluations for languages beyond Chinese/English to test the generality of the data pipeline and alignment recipe.

- Practical applications
  - Enterprise document understanding (contracts, RFPs, compliance) leveraging 128K contexts in Chinese and English.
  - Data‑driven analysis pipelines that blend browsing, computation (Python), and visualization (text‑to‑image) under a single agent loop (Figure 4).
  - Developer platforms that need robust function calling and agent orchestration (Table 7 and Table 8 suggest readiness).
  - Education and support tools in Chinese‑first contexts, where AlignBench shows strong safety and alignment (Table 4 and Table 10).

> In sum, the paper provides a comprehensive, practice‑oriented account of building `GLM‑4`—from data and architecture choices to alignment and agents—and backs it with broad evaluations. The models are particularly compelling for Chinese‑centric, long‑context, and tool‑integrated use cases, while math reasoning and full transparency remain the main opportunities for further improvement.
