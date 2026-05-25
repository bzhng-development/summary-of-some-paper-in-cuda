# QWEN2 TECHNICAL REPORT

**ArXiv:** [2407.10671](https://arxiv.org/abs/2407.10671)

## 🎯 Pitch

Qwen2 introduces a robust suite of open-weight large language models—ranging from lightweight (0.5B) to massive (72B) dense models and a 57B Mixture-of-Experts model—engineered for exceptional long-context processing, multilingual proficiency, and state-of-the-art coding and mathematics performance. By combining advanced architectural choices (like Grouped Query Attention and Dual Chunk Attention with YARN), a diverse 7-trillion-token training set, and scalable alignment strategies, Qwen2 dramatically elevates the performance bar for open models, rivaling proprietary systems on challenging benchmarks. Its open release empowers both research and real-world applications that demand high efficiency, reliability, and global language coverage, accelerating innovation across the AI ecosystem.

---

## 1. Executive Summary (2-3 sentences)
Qwen2 is a family of open‑weight large language models spanning 0.5B–72B parameters plus a 57B Mixture‑of‑Experts (MoE) model, redesigned for long context, multilingual use, and strong coding/mathematics performance. It advances open models’ efficiency and capability via architecture choices (e.g., Grouped Query Attention, Dual Chunk Attention with YARN), a 7‑trillion‑token multilingual/code/math‑heavy pretraining pipeline, and a scalable alignment strategy (SFT + DPO/RLHF), achieving state‑of‑the‑art results among open weights and approaching proprietary systems on many benchmarks (Sections 2–5).

## 2. Context and Motivation
- Problem/gap addressed
  - Open‑weight models have narrowed the gap with proprietary systems but still trail on reasoning, coding, long‑context handling, and multilingual coverage. There is also a deployment gap: models must be efficient (memory/latency) to serve long inputs and varied devices (Section 1).
  - Prior open releases (e.g., Llama‑3, Mixtral, Qwen1.5) improved core abilities but left room for: longer contexts with stable performance, stronger code/math reasoning, broader multilingual coverage, and instruction alignment quality (Sections 1, 3, 5).

- Why this matters
  - Real‑world: document analysis, software engineering, and multilingual applications need long-context comprehension, reliable coding/maths, and broad language support (Sections 3.2, 5.2.3, 5.2.4).
  - Theoretical/engineering: designing efficient attention for long inputs (e.g., memory‑efficient KV caching) and robust post‑training pipelines has broad implications for scalable LLM deployment (Sections 2.2, 4).

- Prior approaches and shortcomings
  - Dense Transformers with standard multi‑head attention are memory‑intensive at long contexts; long‑context extrapolation often degrades without specialized mechanisms (Section 2.2.1).
  - MoE models boosted efficiency but routing, expert granularity, and initialization affected stability and utilization (Section 2.2.2).
  - Alignment quality often demands expensive human annotation; many pipelines do not scale well or overfit to particular benchmark styles (Section 4.1).

- How Qwen2 positions itself
  - Architecture: memory‑lean attention (Grouped Query Attention) plus long‑context mechanisms (Dual Chunk Attention + YARN) for up to 131K tokens with minimal perplexity degradation (Sections 2.2.1, 3.2).
  - Data: 7T tokens with expanded multilingual, code, and math; careful filtering and distribution tuning (Section 3.1).
  - Alignment: scalable SFT and DPO/RLHF with automated data synthesis (rejection sampling, execution feedback, constitutional signals) to reduce human load (Sections 4.1–4.3).
  - Range: from 0.5B edge‑deployable models to 72B flagship and a 57B‑total/14B‑active MoE designed to match ~30B dense performance at lower per‑token compute (Sections 1, 2.2.3, Table 1, Table 3).

## 3. Technical Approach
Step‑by‑step overview of how Qwen2 is built and aligned:

- Tokenizer (Section 2.1)
  - Uses the Qwen byte‑level BPE tokenizer (151,643 regular + 3 control tokens), chosen for high compression and multilingual coverage. The embedding table has an “effective size” larger than the vocab count due to distributed training considerations.

- Dense model architecture (Sections 2.2.1, 2.2.3; Table 1)
  - Base: causal Transformer with standard components like Rotary Positional Embeddings (RoPE), SwiGLU activation, QKV bias, and RMSNorm with pre‑normalization for training stability.
  - Grouped Query Attention (GQA): replaces standard multi‑head attention. In GQA, multiple query heads share a smaller set of key/value heads, cutting the KV cache size and improving throughput at inference. Example from Table 1: `Qwen2-72B` has 64 query heads but only 8 KV heads.
    - Why it matters: KV cache dominates memory for long contexts; reducing KV heads lowers memory/latency without sacrificing much quality (Section 2.2.1).
  - Long‑context mechanisms:
    - Dual Chunk Attention (DCA): splits long sequences into manageable chunks; reproduces standard attention when input fits in one chunk, and otherwise tracks within‑ and cross‑chunk positional relations (Section 2.2.1).
    - YARN: rescales attention weights to improve length extrapolation (Section 2.2.1).
    - RoPE base frequency changed from 10,000 to 1,000,000 to better extrapolate position encodings to long inputs (Section 3.2).
  - Result: “substantially lower KV size per token relative to Qwen1.5,” reducing memory footprint, especially for long‑context inference (Section 2.2.3).

- Mixture‑of‑Experts (MoE) architecture (Section 2.2.2; Table 1)
  - Structure: replaces the FFN with n experts; a gating network `G(x)` outputs probabilities `p = softmax(G(x))`, and the model combines the top‑k experts per token: `y = sum_{i in top‑k(p)} p_i E_i(x)` (Equations (1)–(2)).
  - Fine‑grained experts: more, smaller experts (e.g., 64 routed experts with 8 activated per token, plus 8 shared experts; see Table 1 for `Qwen2‑57B‑A14B`) to increase the diversity of expert combinations and utilization (Section 2.2.2).
  - Expert routing: mix of shared and routing‑specific experts supports general knowledge plus specialization (Section 2.2.2).
  - Expert initialization (“upcycling with diversification”):
    - Start from a dense model’s FFN; replicate it enough times to allocate the desired number and size of experts, shuffle parameters along the intermediate dimension to diversify, then randomly reinitialize 50% of each expert’s parameters to encourage exploration (Section 2.2.2).

- Model configurations (Table 1)
  - Sizes: `0.5B`, `1.5B`, `7B`, `72B` dense; `57B‑A14B` MoE (57B total, 14B active parameters/token).
  - Trained tokens: `7T` for most dense models (except `0.5B` uses `12T`); MoE gets an additional `4.5T` on top of upcycling (Table 1; Sections 3.1, 2.2.3).
  - Embedding tying: enabled in the two smallest models, disabled for larger ones (Table 1).

- Pre‑training data and strategy (Section 3.1)
  - 7‑trillion‑token multilingual mix with substantially more code, math, and non‑English text than Qwen1.5. Data quality improved via heuristic and model‑based filtering; distributions tuned on smaller models (Section 3.1).
  - A 12T dataset was explored but did not beat 7T; the project prioritized higher‑quality 7T data for large models (Section 3.1).

- Long‑context pre‑training (Section 3.2)
  - Context window extended from 4,096 to 32,768 tokens in the final phase of pre‑training, with more long documents; combined with YARN + DCA for effective processing up to 131,072 tokens and “minimal perplexity degradation” in preliminary tests.

- Post‑training (alignment) pipeline (Sections 4.1–4.3)
  - Data types:
    - Demonstrations `D = {(x_i, y_i)}` and preferences `P = {(x_i, y_i^+, y_i^–)}` (Section 4.1).
  - How the data is built:
    - Collaborative annotation (Section 4.1.1): automatic instruction ontology extraction with `InsTag`, instruction selection for diversity/complexity, “self‑evolution” to increase difficulty, and human ranking to produce both good demonstrations and positive/negative pairs.
    - Automated synthesis (Section 4.1.2):
      - Rejection sampling for math: generate multiple reasoning paths; keep correct/“reasonable” ones as demonstrations, and pair correct vs incorrect for preferences.
      - Execution feedback for code and instruction following: generate code with tests; compile/execute to verify; also auto‑generate Python checkers for constraint following.
      - Data repurposing for writing/roleplay: derive instructions from curated texts or character profiles, pairing them with high‑quality outputs.
      - Constitutional feedback: use principles to synthesize aligned and misaligned responses for safety/value alignment.
  - SFT details (Section 4.2):
    - >500k instruction examples; train 2 epochs at 32,768 tokens; cosine‑like LR decay from `7e‑6` to `7e‑7`, weight decay `0.1`, gradient clip `1.0`.
  - RLHF details (Section 4.3):
    - Offline DPO on P, then online DPO using reward models to select best/worst among multiple sampled responses from the current policy.
    - `Online Merging Optimizer` (OMo/OMO; Section 4.3) mitigates “alignment tax” (the common drop in base capabilities after alignment) by merging the aligned updates in a way that preserves core skills.

## 4. Key Insights and Innovations
- Memory‑efficient long‑context attention that actually scales (Sections 2.2.1, 3.2; Figure 1; Table 12)
  - What’s new: integrating Dual Chunk Attention (chunked processing that matches full attention when short, but composes across chunks when long) with YARN (attention rescaling) and a higher RoPE base. This combination enables accurate retrieval and reasoning far beyond the 32K training window, up to 131K–256K in evaluation.
  - Why it matters: enables practical processing of long documents with lower memory via GQA (fewer KV heads) and with less quality loss than naive extrapolation.

- Fine‑grained MoE with diversified upcycling (Section 2.2.2; Table 3)
  - What’s new: more, smaller experts with both shared and routed specialists, plus an initialization that diversifies experts by shuffling and partially reinitializing. This supports richer expert combinations and better specialization.
  - Why it matters: `Qwen2‑57B‑A14B` performs like a ~30B dense model while activating only 14B parameters/token (Table 3), saving compute per token without sacrificing much quality.

- Scalable alignment data pipeline with automated synthesis (Section 4.1)
  - What’s new: systematic ontology‑guided instruction selection and “self‑evolution,” combined with automated rejection sampling (math), execution feedback (code and instruction‑following), data repurposing for creative tasks, and constitutional feedback for safety.
  - Why it matters: reduces reliance on costly human annotation while producing diverse, high‑signal data that improves both core skills and instruction following (Table 6–Table 9, Table 14).

- Data quality and distribution tuning over raw scale (Section 3.1)
  - Insight: a 12T dataset did not beat a cleaner 7T dataset on large models; gains came from quality and distribution (more code/math/multilingual), not just raw tokens. This is a practical lesson for scaling laws.

## 5. Experimental Analysis
- Evaluation methodology (Section 5)
  - Base models: standard few‑shot/zero‑shot accuracy on diverse suites covering general knowledge (MMLU, MMLU‑Pro, GPQA, TheoremQA, BBH, HellaSwag, Winogrande, ARC‑C, TruthfulQA), coding (HumanEval, MBPP, EvalPlus, MultiPL‑E), mathematics (GSM8K, MATH), Chinese (C‑Eval, CMMLU), and multilingual (M3Exam, IndoMMLU, ruMMLU, translated MMLU; plus understanding, reasoning, math, and translation tests) (Section 5.1.1).
  - Instruction‑tuned: same core skills plus alignment/instruction‑following (MT‑Bench, Arena‑Hard, MixEval, IFEval, AlignBench), coding with LiveCodeBench v1, and in‑house automatic evals in Chinese and English (Sections 5.2.1–5.2.2).
  - Long context: Needle‑in‑a‑Haystack (NIAH), NeedleBench (multi‑needle + reasoning), LV‑Eval (multi‑evidence QA) (Section 5.2.3).
  - Multilingual human evaluation across 10 languages (1–5 rating by professional annotators) (Section 5.2.4).
  - Safety: multilingual jailbreak‑style prompts across illegal, fraud, pornography, privacy; lower is better (Section 5.2.5).
  - Contamination: decontamination via n‑gram and LCS filters; re‑evaluate on strict non‑contaminated subsets to quantify impact (Section 5.2.6).

- Main quantitative results
  - Flagship base model (`Qwen2‑72B`, Table 2):
    - > “MMLU 84.2, MMLU‑Pro 55.6, GPQA 37.9, TheoremQA 43.1, BBH 82.4”
    - Coding: > “HumanEval 64.6, MBPP 76.9, EvalPlus 65.4, MultiPL‑E 59.6”
    - Math: > “GSM8K 89.5, MATH 51.1”
    - Chinese: > “C‑Eval 91.0, CMMLU 90.1”
    - Multilingual category averages: > “Exam 76.6, Understanding 80.7, Mathematics 76.0, Translation 37.8”
    - Relative to `Llama‑3‑70B`, Qwen2‑72B is higher on MMLU (84.2 vs 79.5) and coding (HumanEval 64.6 vs 48.2) but similar on HellaSwag and Winogrande (Table 2).
  - MoE base (`Qwen2‑57B‑A14B`, Table 3):
    - Matches or beats ~30B dense baselines on many tasks; especially strong on coding/math:
      > “HumanEval 53.0, MBPP 71.9, EvalPlus 57.2, MultiPL‑E 49.8; GSM8K 80.7, MATH 43.0”
    - General knowledge near `Yi‑1.5‑34B` (MMLU 76.5 vs 77.1) with much lower active parameters/token (14B vs 32B).
  - 7B base (`Qwen2‑7B`, Table 4):
    - Substantial gains over `Qwen1.5‑7B` and competitive with `Llama‑3‑8B`, especially in coding/math:
      > “HumanEval 51.2 (vs 33.5 in Llama‑3‑8B), MBPP 65.9, GSM8K 79.9, MATH 44.2”
    - Strong Chinese (C‑Eval 83.2; CMMLU 83.9) and multilingual understanding (Understanding 72.0).
  - Small base models (`Qwen2‑0.5B`, `Qwen2‑1.5B`, Table 5):
    - `Qwen2‑1.5B` outperforms `Gemma‑2B` and `Qwen1.5‑1.8B` on MMLU (56.5) and math (GSM8K 58.5); coding trails `Phi‑2` but is stronger than other small baselines. Both Qwen2 small models excel on Chinese benchmarks (C‑Eval/CMMLU).
  - Flagship instruction‑tuned (`Qwen2‑72B‑Instruct`, Table 6):
    - Core skills: > “MMLU 82.3, MMLU‑Pro 64.4, GPQA 42.4, TheoremQA 44.4”
    - Coding: > “HumanEval 86.0, MBPP 80.2, MultiPL‑E 69.2, LiveCodeBench v1 35.7”
    - Math: > “GSM8K 93.2, MATH 69.0”
    - Alignment: > “MT‑Bench 9.12, Arena‑Hard 48.1, MixEval 86.7, IFEval 77.6, AlignBench 8.27”
    - Beats `Llama‑3‑70B‑Instruct` on MMLU‑Pro (+8.2), HumanEval (+4.3), and MATH (+18.6), and is close on MBPP (–2.1). It trails `Mixtral‑8x22B‑Instruct` on GPQA (–7.3) but leads on most other fronts (Table 6).
  - MoE instruction‑tuned (`Qwen2‑57B‑A14B‑Instruct`, Table 7):
    - Competitive with `Yi‑1.5‑34B‑Chat` (~30B dense) and ahead of `Mixtral‑8x7B‑Instruct` on most coding/alignment metrics:
      > “HumanEval 79.9 vs 45.1 (Mixtral‑8x7B), LiveCodeBench 25.5 vs 12.3; MT‑Bench 8.55; MixEval 82.3”
  - 7B instruction‑tuned (`Qwen2‑7B‑Instruct`, Table 8):
    - Strong coding/math relative to peers:
      > “HumanEval 79.9 (vs 62.2 in Llama‑3‑8B‑Instruct), GSM8K 85.7, MATH 52.9”
    - Instruction following still lags `Llama‑3‑8B‑Instruct` on IFEval (54.7 vs 72.1) despite good MT‑Bench (8.41) (Table 8).
  - Small instruction‑tuned (Table 9):
    - `Qwen2‑0.5B‑Instruct` and `Qwen2‑1.5B‑Instruct` show large jumps vs Qwen1.5 on coding/math and IFEval (e.g., `1.5B` GSM8K 61.6; HumanEval 47.0; IFEval 29.0).
  - Long context (Figure 1; Table 12):
    - NeedleBench and LV‑Eval verify that YARN + DCA significantly improve performance beyond 32K:
      > For `Qwen2‑72B‑Instruct` on LV‑Eval, “+YARN+DCA” yields strong scores at long lengths (e.g., 128K/256K) compared to the vanilla model, and maintains high accuracy up to 32K (“does not change behavior within 32K tokens,” Table 12).
      > `Qwen2‑7B‑Instruct` shows degradation at 256K, but gains substantially from YARN+DCA (Table 12).
  - Multilingual human eval (Table 13):
    - `Qwen2‑72B‑Instruct` average 3.93/5 across 10 languages; substantially above `GPT‑3.5‑Turbo` (3.16), close to `GPT‑4‑Turbo` (3.98), and behind `Claude‑3‑Opus` (4.15).
  - Safety (Table 14; lower is better):
    - Qwen2 reduces harmful responses vs both `GPT‑4` and `Mixtral‑8x22B‑Instruct`:
      > “Illegal 0.00 (tie with GPT‑4), Fraud 2.41 vs GPT‑4 3.40, Pornography 22.91 vs GPT‑4 23.63, Privacy 2.47 vs GPT‑4 3.37.”
  - Contamination analysis (Table 15):
    - Strict filtering indicates high “contamination” percentages for some code/math sets (e.g., HumanEval 75%), but performance on non‑contaminated subsets changes very little for `Qwen2‑72B‑Instruct` (e.g., HumanEval +1.0), suggesting many are false positives (common snippets) rather than genuine leakage.

- Do the experiments support the claims?
  - Yes, across sizes and tasks, Qwen2 consistently improves over Qwen1.5 and is competitive with or better than leading open models; the 72B instruction‑tuned model approaches proprietary systems in coding/math and alignment metrics (Tables 2, 6, 13, 14).
  - Long‑context claims are supported by multiple tests and by the architectural design (Figure 1; Table 12; Sections 2.2.1, 3.2).

- Notable caveats and mixed results
  - On some “easier” English MC tasks, gains are small or slightly behind top baselines (e.g., HellaSwag/ARC‑C in Table 2).
  - `Qwen2‑7B‑Instruct` underperforms `Llama‑3‑8B‑Instruct` on IFEval (instruction‑constraint following) despite strong coding/math (Table 8).
  - `Qwen2‑57B‑A14B` trails `Yi‑1.5‑34B` on MMLU by ~0.6 points but wins on many coding/math tasks (Table 3).

## 6. Limitations and Trade-offs
- Assumptions and scope
  - The report emphasizes system‑level enhancements (attention, MoE, data/long‑context, scalable alignment), not theoretical optimality of any single mechanism. The approach assumes high‑quality filtering and balanced multilingual distributions are available (Section 3.1).
- Scenarios not fully addressed
  - Tool use, retrieval‑augmented generation, and multimodality are out of scope here (even though the broader Qwen family includes multimodal models).
  - Fine‑grained analysis of instruction following at small/medium sizes remains incomplete; `Qwen2‑7B‑Instruct`’s IFEval gap vs `Llama‑3‑8B‑Instruct` indicates room to improve constraint adherence (Table 8).
- Computational and data constraints
  - Training on 7T tokens with long‑context phases is compute‑intensive. While KV memory is reduced (GQA) and MoE saves per‑token compute, these models still require substantial hardware for training and high‑end GPUs for best inference speed (Sections 2.2.1–2.2.3, 3.2).
- English vs Chinese vs multilingual balance
  - In-house English evaluations show `Qwen2‑72B‑Instruct` slightly behind `Llama‑3‑70B‑Instruct` on some comprehension/coding averages (Table 11), suggesting remaining gaps in certain English domains despite strong multilingual and Chinese performance (Tables 10–11).
- Long‑context edge cases
  - At 256K tokens, accuracy drops for all models, particularly smaller ones (Table 12). Long‑range multi‑needle reasoning remains challenging at extreme lengths.
- Safety remains hard
  - Although Qwen2 improves over GPT‑4 on the tested categories, the pornography category still shows non‑trivial unsafe response rates (22.91%) (Table 14), reflecting the difficulty of perfectly safe generation across languages and prompts.
- Transparency on data sources
  - Section 3.1 describes quality improvements and distributions but does not enumerate exact data sources/weights; this can limit full reproducibility of the pre‑training corpus composition.

## 7. Implications and Future Directions
- How this changes the landscape
  - Qwen2 pushes open‑weight LLMs closer to proprietary models on alignment (MT‑Bench/Arena‑Hard), coding (HumanEval/LiveCodeBench), math (GSM8K/MATH), multilingual performance, and long‑context processing (Tables 6, 12–14). The publicly released weights (Section 1, links on Hugging Face/ModelScope/GitHub) lower the barrier for research and production deployment across sizes.
  - The MoE design demonstrates that fine‑grained experts with diversified upcycling can match ~30B dense performance with ~14B active parameters/token (Table 3), a practical blueprint for compute‑efficient scaling.

- Follow‑up research enabled/suggested
  - Long‑context: ablations on DCA vs YARN vs RoPE base; extending reliable accuracy to 256K–1M tokens and understanding failure modes for multi‑needle/multi‑hop reasoning (Section 5.2.3; Table 12).
  - Alignment: targeted improvement of instruction‑constraint following for smaller models (e.g., raise IFEval at 7–9B, Table 8) using richer automated checkers and harder constraint curricula.
  - Data: systematic studies on data mixture effects (quality vs quantity, language balance, code/math proportions) to further validate the 7T>12T finding (Section 3.1).
  - MoE: exploration of routing strategies, expert specialization diagnostics, and scaling laws for fine‑grained experts.

- Practical applications
  - Enterprise document processing and analytics with long contexts (up to 128K+ tokens) using the 7B/72B models with YARN+DCA (Figure 1; Table 12).
  - Software engineering copilots: strong coding results (HumanEval/LiveCodeBench) across the 7B/57B/72B instruction‑tuned models (Tables 6–8).
  - Multilingual assistants: competitive human ratings across 10 languages (Table 13), with particularly strong Chinese performance (Tables 2, 10).
  - Edge and on‑device AI: 0.5B/1.5B models tuned for portability while retaining useful coding/math and Chinese capabilities (Tables 5, 9).

> Overall, Qwen2 demonstrates that careful architectural choices (GQA, DCA+YARN), data quality/distribution, and scalable alignment pipelines can produce open‑weight models that are efficient, long‑context capable, multilingual, and highly competitive on coding and mathematics—closing a substantial portion of the gap to the best proprietary systems (Sections 2–5).
