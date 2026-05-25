# ChatGLM: A Family of Large Language Models from GLM-130B to GLM-4 All Tools

**ArXiv:** [2406.12793](https://arxiv.org/abs/2406.12793)

## 🎯 Pitch

ChatGLM introduces a cutting-edge family of bilingual (Chinese–English) large language models, culminating in the GLM-4 series and the agentic GLM-4 All Tools model. These models not only rival GPT-4 and other leading systems on reasoning, coding, and ultra-long context tasks, but also seamlessly integrate autonomous tool use—allowing them to understand user intent, plan multi-step tasks, and dynamically invoke web, code, and generative tools for real-world applications. This advances the frontiers of safe, accessible, and practical LLMs with deep Chinese capabilities, offering open models that democratize high-level AI and empower complex, end-to-end task automation across languages.

---

## 1. Executive Summary
ChatGLM presents an evolving family of large language models culminating in the `GLM-4` series and the `GLM-4 All Tools` agentic model. It tackles two problems at once: building bilingual (Chinese–English) foundation models that rival frontier systems on standard reasoning, coding, and long‑context tasks, and aligning them to autonomously plan and use external tools (browser, Python, image generation, and user APIs) to complete complex tasks.

## 2. Context and Motivation
- Problem or gap
  - Strong general‑purpose LLMs with robust Chinese ability and long‑context competence have lagged behind best‑in‑class English‑centric models. At the same time, real applications increasingly demand agentic behavior: understanding user intent, planning, and selectively calling tools. Section 1 lays out this dual goal: competitive base capability plus integrated tool use at scale (128K–1M tokens).
- Importance
  - Real-world: enterprise and consumer tasks often require up‑to‑date information retrieval, numerical computation, or image generation; an LLM that can autonomously decide when and how to invoke these tools provides end‑to‑end task completion (Figure 2 shows a browser‑plus‑Python workflow for computing population CAGR).
  - Theoretical/engineering: pushing stable training to “ten trillions” of tokens for bilingual data, extending context windows from 2K to 128K/1M, and aligning long‑context usage (Sections 2 and 3) are non‑trivial advances in data engineering, architecture, and alignment.
- Prior approaches and shortcomings
  - GPT‑3/3.5/4 popularized instruction tuning and RLHF but focused primarily on English and did not open‑source intermediate, smaller variants. Open models (OPT, BLOOM, LLaMA) provided foundations but lacked strong Chinese alignment, native tool‑use alignment, or long‑context performance up to 128K–1M with competitive alignment (Section 1; Figure 1 timeline).
- Positioning relative to existing work
  - ChatGLM evolves from `GLM-130B` (2022) with an alternative pretraining objective (autoregressive blank infilling), into `ChatGLM-6B/2/3`, and finally `GLM‑4` and `GLM‑4 All Tools` (Figure 1). It emphasizes:
    - High‑quality bilingual pretraining and alignment,
    - Long‑context alignment via `LongAlign` (Section 2, “ChatGLM Techniques”),
    - Agent/tool‑use alignment with function calling and multi‑tool planning (Figure 4; Section 2 “GLM‑4 All Tools”),
    - Open releases of capable 9B variants with 128K–1M context (Table 1, Section 1).

## 3. Technical Approach
This section distills the end‑to‑end system: pretraining data and tokenization; architecture; post‑training alignment; and the “All Tools” agent pipeline.

- Pretraining data pipeline (Section 2 “Pre‑Training Data”)
  - Data sources: multilingual web pages, Wikipedia, books, code, research papers; mostly Chinese and English with 24 additional languages.
  - Cleaning and selection:
    - Deduplication: both exact and fuzzy to improve diversity.
    - Filtering: remove noisy or unsafe pages (offensive text, placeholder text, raw code pages when undesired).
    - Tokenization: byte‑level BPE learned separately for Chinese and multilingual corpora, then merged with `cl100k_base` (OpenAI’s tiktoken) to form a 150k‑token vocabulary—this balances Chinese segmentation with compatibility for English and code.
  - Reweighting: higher weights for “educational, high‑quality sources” (books, Wikipedia) to improve reasoning/knowledge quality.
  - Scale: roughly “ten trillions” of tokens—important for emergent reasoning and long‑context generalization (Section 2).

- Core architecture (Section 2 “Architecture”)
  - Base is a Transformer, with design choices aimed at longer contexts and efficient inference:
    - `No Bias Except QKV`: remove most linear biases for speed and better length extrapolation; keep biases only in attention’s Query/Key/Value projections.
    - `RMSNorm` + `SwiGLU`: swaps LayerNorm/ReLU (or GeLU) for RMSNorm/SwiGLU to improve stability and quality.
    - `RoPE` (rotary position embeddings) extended to 2D to suit GLM’s 2D positional encoding.
    - `GQA` (Group Query Attention): reduces KV‑cache size at inference by sharing keys/values across multiple query heads. Since GQA has fewer parameters than vanilla MHA, FFN width is increased to keep overall model capacity, with `d_ffn = 10/3 × hidden_size`.
  - Long‑context scaling:
    - Position encoding extension methods (ALiBi/RoPE interpolation as in [31, 5]) plus continual training on long text [47].
    - `LongAlign` alignment (see below) to teach the model how to use long context effectively rather than just tolerate it. Context windows progress from 2K (ChatGLM) → 32K (ChatGLM2/3) → 128K and experimental 1M (GLM‑4; Section 2 and Table 1).

- Post‑training alignment (Section 2 “Alignment”)
  - `SFT` (supervised fine‑tuning): uses “authentic human prompts and interactions,” not synthetic templates, to improve instruction following and conversational helpfulness.
  - `RLHF`: reinforcement learning from human feedback mitigates refusal errors, bilingual mixing in responses, safety behaviors, and multi‑turn coherence.
  - Data sources: first generation (ChatGLM‑6B/130B) used developer‑crafted data; later generations combine in‑house annotations and proprietary third‑party data under strict quality control.

- Techniques introduced/used by ChatGLM (Section 2 “ChatGLM Techniques”; brief definitions)
  - `Emergent Abilities from Loss` [12]: reframes emergence using the pretraining loss threshold at which downstream performance rises above chance; implies that token budget and loss targets govern capability emergence.
  - `LongAlign` [1]: a procedure to align models for long contexts (up to 128K), combining training strategies and evaluation to reach parity with Claude 2 / GPT‑4 Turbo in long‑context chat (Table 5).
  - `ChatGLM‑Math` [48]: a “self‑critique” approach to select/curate math reasoning data without external models or heavy manual labeling; targeted at closing math gaps.
  - `ChatGLM‑RLHF` [17]: practical recipes for PPO/DPO‑style preference optimization at scale.
  - `Self‑Contrast` [24]: “feedback‑free” alignment that has the model generate its own negative samples to reduce costly human preference data.
  - `AgentTuning` [52]: instruction tuning with high‑quality agent–environment interaction trajectories to improve tool use and planning.
  - `APAR` [21]: “auto‑parallel auto‑regressive” generation; trains the model to plan hierarchical structures and generate some parts in parallel for responses with inherent structure.

- `GLM‑4 All Tools` agent pipeline (Section 2 “GLM‑4 All Tools”; Figure 4)
  - Goal: autonomously break down a user’s complex request, plan steps, and decide which tools (browser, Python, text‑to‑image, user APIs/functions) to call, using intermediate results to guide subsequent actions.
  - Mechanism:
    - Plan/analyze the request,
    - Decide on tool calls; issue structured function calls when needed,
    - Ingest tool feedback/results,
    - Iterate (recursive execute) until task completion,
    - Persist context/memory for multi‑step workflows.
  - Example (Figure 2): searching the web for population data and computing CAGR in Python within the same conversation.

## 4. Key Insights and Innovations
- Long‑context capability that is actually aligned for use (not just supported)
  - What’s new: `LongAlign` turns extended context windows (128K/1M) into usable capabilities for summarization, retrieval, and coding with long documents (Section 2; Table 5).
  - Why it matters: many commercial settings involve long reports or legal/technical documents; `GLM‑4 (0520)` matches GPT‑4 Turbo and Claude 3 Opus in English long‑context tasks and exceeds them in Chinese (Table 5).
- Agentic “All Tools” alignment with autonomous tool selection and planning
  - What’s new: a single model aligned to plan and call multiple tools in multi‑round workflows (Figure 4), including user‑defined APIs and knowledge bases. It integrates Python, web browsing, and text‑to‑image into a coherent loop with memory and feedback.
  - Why it matters: shifts LLMs from chatbots to problem‑solving agents; first‑hand tests show parity or better performance than GPT‑4 All Tools for web access and math via Python (Figure 2; Table 9).
- Bilingual (Chinese–English) alignment at scale with competitive general ability
  - What’s new: large‑scale bilingual pretraining (ten trillion tokens) and alignment “primarily for Chinese and English” produces strong general metrics on MMLU/GSM8K/MATH/BBH/GPQA/HumanEval, competitive with frontier models while outperforming on Chinese alignment (Section 3; Tables 2, 4).
  - Why it matters: models serving Chinese‑speaking users often face a tradeoff between Chinese ability and general reasoning; GLM‑4 reduces that gap (Table 4, “Overall” 8.00 vs GPT‑4 Turbo 7.90/8.00 and Claude 3 Opus 7.53).
- Practical architecture choices that balance speed, memory, and extrapolation
  - What’s new: `No Bias Except QKV`, `RMSNorm + SwiGLU`, 2D‑RoPE, and `GQA` with scaled‑up FFN width (Section 2).
  - Why it matters: these choices empirically improved length generalization and inference memory (via smaller KV cache) while maintaining model capacity.

## 5. Experimental Analysis
- Evaluation setup (Section 3)
  - Benchmarks span general knowledge (MMLU), math reasoning (GSM8K, MATH), multi‑step reasoning (BBH), graduate‑level science (GPQA), code generation (HumanEval), instruction following (IFEval), Chinese alignment (AlignBench), long‑context chat (LongBench‑Chat), real‑world coding (NaturalCodeBench/NCB), function calling (Berkeley Function Call Leaderboard), agent tasks (AgentBench), and safety (SafetyBench).
  - Deployment: GLM‑4 and GLM‑4‑Air evaluated with bfloat16 precision; long‑context chats are judged by GPT‑4 with few‑shot prompts and averaged over multiple runs (Table 5).
  - Open model: `GLM‑4‑9B` (128K and 1M context) trained with the same post‑training pipeline and released openly (Table 1 and Section 1).

- Headline quantitative results (all numbers taken directly from the cited tables)
  - General academic benchmarks (Table 2)
    - `GLM‑4 (0520)` achieves MMLU 83.3 (vs GPT‑4 86.4; GPT‑4 Turbo 86.7 [2024‑04‑09]), GSM8K 93.3 (close to GPT‑4 92.0; behind GPT‑4 Turbo 95.6), MATH 61.3 (between GPT‑4 52.9 and GPT‑4 Turbo 73.4), BBH 84.7 (below GPT‑4 Turbo 88.2), GPQA 39.9 (below GPT‑4 Turbo 49.3), HumanEval 78.5 (below GPT‑4 Turbo 88.2).
    - Takeaway: approaches GPT‑4 on average with strengths in GSM8K vs GPT‑4 (not Turbo) but trailing on GPQA/HumanEval vs latest GPT‑4 Turbo.
  - Instruction following (Table 3, IFEval)
    - In English strict instruction‑level accuracy, `GLM‑4 (0520)` 85.0 vs GPT‑4 Turbo (2024‑04‑09) 85.9 (99% of it); in Chinese strict instruction‑level, 78.0 vs 79.1 (98.6%).
    - Takeaway: near‑parity with GPT‑4 Turbo on instruction‑following in both English and Chinese.
  - Chinese alignment (Table 4, AlignBench‑v1.1)
    - `GLM‑4 (0520)` Overall 8.00, higher than GPT‑4 Turbo (1106) 7.90 and Claude 3 Opus 7.53; strongest in Chinese Logic and Language.
    - Caveat: slightly behind GPT‑4 Turbo (2024‑04‑09) on Math (8.32 vs GLM‑4 7.89).
  - Long context (Table 5, LongBench‑Chat)
    - English: `GLM‑4 (0520)` 87.3, on par with GPT‑4 Turbo (1106) 87.2 and Claude 3 Opus 87.7.
    - Chinese: `GLM‑4 (0520)` 84.0, above GPT‑4 Turbo (2024‑04‑09) 82.1 and Claude 3 Opus 82.7.
  - Coding on real user prompts (Table 6, NCB)
    - `GLM‑4 (0520)` Overall 47.1 vs GPT‑4 Turbo (2024‑04‑09) 53.8 and Claude 3 Opus 48.3.
    - Takeaway: close to Claude 3 Opus; room to close the gap to GPT‑4 Turbo.
  - Function calling (Table 7, Berkeley Leaderboard)
    - `GLM‑4 (0520)` Overall 81.76, comparable to GPT‑4 Turbo (2024‑04‑09) 81.24; the smaller `GLM‑4‑9B‑Chat` scores 81.00 and notably high Execution Summary 87.92.
    - Insight: function calling quality is not strictly monotonic with model size; alignment and data matter (Table 7 note).
  - Agent tasks (Table 8, AgentBench)
    - Overall score: `GLM‑4 (0520)` 3.79, slightly above GPT‑4 Turbo (1106) 3.77 and Claude 3 Opus 3.62; strongest on Database, House‑Holding, Web Shopping; weaker than GPT‑4 on OS/knowledge‑graph/lateral thinking.
  - All‑tools performance (Table 9)
    - Python‑assisted math: GSM8K 91.59 vs ChatGPT‑4 (Web) 92.72; MATH 63.60 vs 65.00; Math23K 88.50 vs 88.40.
    - Web information seeking: 78.08 vs ChatGPT‑4 (Web) 67.12.
  - Safety (Table 10, SafetyBench)
    - Overall: `GLM‑4 (0520)` 87.2, near Claude 3 Opus 87.5 and behind GPT‑4 (0613) 89.7; largest gap on Physical Health (hands‑on common‑sense risks).

- Progress across generations (Table 1; Figure 3)
  - `ChatGLM‑6B → ChatGLM2‑6B → ChatGLM3‑6B‑Base → GLM‑4‑9B` shows monotonic gains: e.g., MMLU 25.2 → 45.2 → 61.4 → 74.7; GSM8K 1.5 → 25.9 → 72.3 → 84.0. This supports that data/architecture/alignment changes accumulate into meaningful capability gains.

- Do experiments support the claims?
  - Claims of “close to GPT‑4/Claude on general benchmarks” are partly supported (Table 2 shows proximity on MMLU/GSM8K but gaps on GPQA/HumanEval vs latest GPT‑4 Turbo).
  - Claims of “matching GPT‑4 Turbo/Claude on long‑context” are supported by Table 5.
  - Claims of “strong Chinese alignment” are supported by Table 4.
  - Claims of “agentic tool use” have quantitative evidence (Tables 7–9) and system schematics (Figure 4). Note: LongBench‑Chat uses GPT‑4 as judge, which can bias results; the paper mitigates variance by averaging multiple runs (Table 5 notes).

- Missing or limited analyses
  - No detailed ablations isolating the effect sizes of `No Bias Except QKV`, `2D‑RoPE`, `GQA + wider FFN`, or `LongAlign` components.
  - Limited failure analysis for tool‑use edge cases (e.g., hallucinated tool parameters, unsafe browsing loops).
  - Contamination checks are discussed for HumanEval at a field level but not presented for each benchmark; NCB is used to reduce contamination risk (Section 3.5).

## 6. Limitations and Trade-offs
- Alignment focus and language balance
  - Alignment is “predominantly to Chinese and English” (Abstract; Section 3). Performance in other 24 languages is not evaluated here; cross‑lingual generalization remains unclear.
- Benchmarks with LLM-as-judge
  - LongBench‑Chat scoring uses GPT‑4 as a judge (Section 3.4), which can introduce bias and may favor certain style/formatting. The paper averages multiple runs, but external, human‑grounded evaluation would further increase confidence.
- Math and specialized science gaps vs latest GPT‑4 Turbo
  - `GPQA` and `MATH` still show gaps compared with the very latest GPT‑4 Turbo numbers (Table 2), though `ChatGLM‑Math` is aimed at closing this.
- Code and real‑world tasks
  - On NCB, `GLM‑4` trails GPT‑4 Turbo (Table 6). Bridging this likely requires additional high‑quality, real‑world programming supervision and tool‑use data.
- Tool‑use robustness and safety
  - While Table 9 shows promising averages, there is limited visibility into failure rates like incorrect API parameterization, browsing loops, or tool invocation cost/latency.
- Compute and scaling
  - Training “ten trillions” of tokens implies substantial compute and data curation infrastructure. The paper does not disclose exact compute budgets or carbon/latency trade‑offs; it introduces `GLM‑4‑Air` for lower latency/cost but does not quantify the savings.
- Open model vs closed models
  - The strongest model (`GLM‑4`) is API‑served; the open model is `GLM‑4‑9B`. The capability gap between the open and closed variants remains material (Table 2 vs `GLM‑4‑9B‑Chat` rows; Table 7 execution summary trends).

## 7. Implications and Future Directions
- How this work shifts the landscape
  - Demonstrates that a bilingual model family can approach frontier performance in general reasoning while surpassing them in Chinese alignment and matching them in long‑context tasks at 128K (Table 5). It also normalizes “All Tools” alignment—LLMs as planners and tool orchestrators—not just chatbots (Figure 4; Table 9).
- Follow‑up research enabled or suggested
  - Long‑context alignment science: `LongAlign` provides a blueprint; future work can quantify which components deliver the biggest gains, and extend to retrieval‑augmented generation and memory architectures.
  - Math and science reasoning: expand `ChatGLM‑Math` self‑critique to multimodal math, formal theorem proving, and verifier‑augmented training.
  - Code and real‑world programming: improve NCB performance via curated agent trajectories, execution‑based learning, and stronger function‑tool ecosystems (e.g., IDE integration).
  - Tool‑use reliability and safety: systematic stress tests for browsing, API calling, and recursive planning; formalize stop conditions, cost‑aware planning, and permissioning.
  - Evaluation methodology: reduce reliance on LLM‑as‑judge by adding human evaluation and automated metrics with ground truth where possible.
  - Multilingual expansion: extend alignment beyond Chinese/English to the 24 languages present in pretraining, measuring cross‑lingual parity.
- Practical applications
  - Enterprise assistants for research, data analysis, and operations that require web access, spreadsheet‑like computation (Python), and report summarization over 100K‑token documents.
  - Developer copilots for data engineering and analytics pipelines with function calling and execution summaries (Table 7).
  - Chinese‑first applications: education, public services, and customer support requiring culturally aligned and safety‑aware dialogue (Table 4, SafetyBench Table 10).

> In short, Sections 2–3 and Tables 2–10 show that `GLM‑4` delivers competitive base capabilities, strong Chinese alignment, long‑context proficiency up to 128K (with an experimental 1M variant), and a practically aligned agent that autonomously plans and calls tools. The remaining gaps—math at the frontier level, code on real prompts, and tool‑use robustness—are visible and tractable targets for the next iterations.
