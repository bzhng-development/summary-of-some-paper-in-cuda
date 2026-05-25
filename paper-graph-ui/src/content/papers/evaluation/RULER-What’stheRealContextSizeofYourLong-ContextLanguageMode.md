# RULER: What’s the Real Context Size of Your Long-Context Language Models?

**ArXiv:** [2404.06654](https://arxiv.org/abs/2404.06654)

## 🎯 Pitch

RULER introduces a flexible, synthetic benchmark that rigorously tests long-context language models (LLMs) beyond basic retrieval, encompassing multi-hop tracing, aggregation, and long-context question answering. By revealing how most models fail to sustain performance with increasing context length and complexity, RULER challenges current evaluation norms and provides a crucial tool to accurately measure—and accelerate—real progress in long-context AI.

---

## 1. Executive Summary
RULER is a configurable, synthetic benchmark for testing what long‑context language models can actually do with very long inputs. It goes beyond simple “find the needle” retrieval to evaluate multi‑hop tracing, aggregation, and long‑context question answering, revealing that many models advertised with 32K+ context windows cannot maintain reliable performance as length and task complexity grow (Table 3).

## 2. Context and Motivation
- Problem addressed
  - The field lacks a comprehensive, reliable way to measure whether long‑context language models (LLMs) truly use long inputs effectively. Most current evaluations rely on simple retrieval tests such as “needle‑in‑a‑haystack” (NIAH)—finding a single item buried in long distractors—which captures only a narrow ability (Abstract; Sec. 1).
- Why it matters
  - Real use cases (e.g., legal analysis, scientific literature review, multi‑document QA) require more than locating a single fact: models must trace references across a document, aggregate dispersed information, and answer questions precisely. Measuring only retrieval overestimates real capability and can mislead deployment decisions (Sec. 1; Table 1).
- Prior approaches and gaps
  - Realistic benchmarks (e.g., ZeroSCROLLS, L‑Eval, BAMBOO, LongBench; Table 1) use human or hybrid data but:
    - Often mix in “parametric knowledge” (knowledge stored in the model’s weights), which can hide whether the model actually uses the given context.
    - Have limited control over context length, task difficulty, and where key information appears.
  - Synthetic tests (NIAH, passkey/line/kv retrieval) offer control but mainly test simple retrieval (Table 1).
- How RULER positions itself
  - RULER is synthetic, so it controls sequence length and task difficulty while minimizing reliance on parametric knowledge. It extends beyond retrieval into:
    - Multi‑hop tracing (variable tracking),
    - Aggregation (counting/common word detection under controlled distributions),
    - Long‑context QA with distractors (Sec. 3, Table 2; Table 1).

## 3. Technical Approach
RULER is a suite of auto‑generated tasks with tunable length and difficulty. Each example is a long input with a concise query and a precise target answer; models are evaluated by recall‑based accuracy (Sec. 4: “append the input with an answer prefix … check the presence of the target output”).

Key design elements
- Synthetic construction to control difficulty and length
  - Inputs are generated so that:
    - The signal‑to‑noise ratio (how much relevant signal appears relative to distractors) and the number of target tokens are controllable proxies for task difficulty (Sec. 3).
    - Parametric knowledge is minimized by using synthetic words/numbers/UUIDs and generic texts (Table 2).
- Four task categories (Sec. 3; Table 2)
  1) Retrieval (Needle‑in‑a‑Haystack family)
     - Definitions
       - “Needle”: a key–value pair inserted somewhere in the long “haystack.”
       - Query: placed at the end, asks for the value(s) associated with specific key(s).
     - Variants test robustness and recall:
       - `S‑NIAH` (Single): one needle. Keys/values can be words, 7‑digit numbers, or 32‑digit `UUID`s; haystack can be repeated noise sentences or Paul Graham essays (Sec. 3.1; Table 2).
       - `MK‑NIAH` (Multi‑keys): many needles with different keys; only the queried key is relevant. Adding many distractor keys creates “hard distractors,” including an extreme setting where the entire haystack is filled with distractor needles (Sec. 3.1; Table 2).
       - `MV‑NIAH` (Multi‑values): multiple values share the same key; the model must return all values—tests completeness of retrieval (recall) (Sec. 3.1; Table 2).
       - `MQ‑NIAH` (Multi‑queries): multiple distinct keys to retrieve in one go—tests recall when many items must be fetched (Sec. 3.1; Table 2).
  2) Multi‑hop Tracing (`VT`: Variable Tracking)
     - Emulates following a chain of references, a minimal proxy for coreference resolution across long text.
     - Mechanism: initialize `X1 = V`; then insert statements like `X2 = X1; X3 = X2; …` at positions throughout the long input. The query asks for “all variables assigned value V.” Difficulty increases with the number of hops (chain length) and the number of parallel chains (Sec. 3.2; Table 2).
  3) Aggregation (`CWE` and `FWE`)
     - `CWE` (Common Words Extraction): the input is a long list of synthetic words; a fixed set of “common” words are injected more frequently. Task: output the top‑K most common words. Difficulty scales by increasing uncommon words with length (Sec. 3.3; Table 2).
     - `FWE` (Frequent Words Extraction): frequencies follow a Zeta distribution (a heavy‑tailed distribution closely related to Zipf’s law; parameter `α` controls how steeply frequencies drop). Task: output the top‑3 most frequent words. Lower `α` makes frequencies more similar, so counting is harder (Fig. 1; Sec. 3.3; Table 2).
  4) Long‑context QA (`QA`)
     - Start from short‑context QA datasets (SQuAD for single‑hop; HotpotQA for multi‑hop) and inject many distracting paragraphs from the same dataset; only a subset (“gold” paragraphs) contains the answer. Task: answer the question based solely on the provided documents (Sec. 3.4; Table 2).
- Experimental protocol (Sec. 4)
  - Models: 17 aligned long‑context models (Gemini‑1.5‑Pro, GPT‑4‑1106, and 15 open‑source), covering 7B–100B+ parameters and claimed windows 32K–1M (Appendix A).
  - Lengths and sampling: For each of 13 chosen task configurations, evaluate each model at 6 context lengths: 4K, 8K, 16K, 32K, 64K, 128K. For every (task, length), generate 500 examples (Sec. 4; Appendix B lists configurations).
  - Inference: vLLM with paged attention, bfloat16, greedy decoding on 8×A100 GPUs; prompts follow each model’s chat template; an answer prefix prevents refusal or extra text (Sec. 4; Appendix D).
  - Metrics and scoring:
    - Primary metric: accuracy by exact presence of required answer tokens (“recall‑based accuracy”).
    - “Effective context length”: the largest length where a model’s average score across RULER’s 13 tasks exceeds a fixed threshold—set to Llama‑2‑7B‑chat performance at 4K (85.6%, Table 3; justification in Sec. 4).
    - Weighted averages: two rankings aggregate performance across all lengths with linearly increasing weights (`wAvg (inc)`) or decreasing weights (`wAvg (dec)`), approximating usage skewed to long or short contexts (Sec. 4; Table 3).
  - Task selection: from an initial larger pool, tasks were clustered by correlation; redundant ones were removed, leaving 13 representative tasks spanning distinct behaviors (Appendix C; Fig. 5).

## 4. Key Insights and Innovations
1) A behaviorally rich, controllable long‑context benchmark
   - Novelty: moves beyond single‑item retrieval to test multi‑hop tracing and aggregation—capabilities central to real long‑document use, but largely missing from prior synthetic benchmarks (Sec. 3; Table 1).
   - Significance: reveals failure modes hidden by vanilla NIAH, where many models score perfectly (Tables 10–11) but then collapse on aggregation/QA at scale (Tables 15–16).
2) Systematic stress tests for retrieval robustness and recall
   - Novelty: the NIAH family explicitly varies (a) needle/haystack type; (b) number of distractor needles; (c) number of required outputs (MV/MQ). This isolates whether models can ignore hard distractors and return all relevant items (Sec. 3.1; Table 2).
   - Significance: shows substantial drops when facing distractors or multiple required items (Fig. 2), demonstrating retrieval‑only testing overestimates effective long‑context capabilities.
3) Aggregation tasks grounded in controlled frequency distributions
   - Novelty: `CWE` and `FWE` use uniform and Zeta‑distributed sampling to force genuine counting/aggregation across long sequences (Sec. 3.3; Fig. 1). Tuning `α` adjusts hardness by reducing separability between top‑frequency words.
   - Significance: exposes tendencies to rely on parametric priors (“the”, “a”) or to copy prompts rather than compute counts (Sec. 5; Fig. 3 middle‑right).
4) Two summary metrics that connect to deployment
   - “Effective context length” and length‑weighted averages provide interpretable summaries of practical capability at scale (Sec. 4; Table 3). This reframes “context window” claims (advertised token limits) into measured, task‑averaged effectiveness.
5) Empirical insights about scaling, training length, and architecture
   - Larger models are more robust at long context (Yi‑34B > Yi‑6B/9B; Fig. 4 middle‑right).
   - Training on longer windows helps but can be inconsistent, and extrapolating beyond trained length causes abrupt drops (LWM series; Fig. 4 left, middle‑left).
   - Non‑Transformer architectures tested (RWKV‑v5, Mamba‑2.8B) lag substantially behind a Transformer baseline on RULER (Fig. 4 right).

## 5. Experimental Analysis
- Evaluation setup (Sec. 4; Appendix B–D)
  - 17 aligned LLMs evaluated on 13 tasks at 6 lengths (4K–128K), 500 examples per (task, length).
  - Accuracy computed by matching the demanded outputs; a response prefix minimizes refusal/explanations.
- Main quantitative results (Table 3)
  - Overall ranking:
    - Weighted average across lengths:
      - Increasing‑with‑length weighting: 
        > “Gemini‑1.5‑Pro: 95.5 (1st), GPT‑4: 89.0 (2nd), GLM‑4‑9B: 88.0 (3rd), Llama‑3.1‑70B: 85.5 (4th).”
      - Decreasing‑with‑length weighting:
        > “Gemini‑1.5‑Pro: 96.1 (1st), GPT‑4: 94.1 (2nd), Llama‑3.1‑70B: 93.7 (3rd), Qwen‑2‑72B: 92.3 (4th), Command‑R‑plus: 92.1 (5th).”
  - Effective context length (threshold = Llama‑2‑7B‑chat 4K score, 85.6%):
    - > “Gemini‑1.5‑Pro: >128K (beyond tested maximum). GPT‑4: 64K. Llama‑3.1‑70B: 64K. GLM‑4‑9B: 64K. Qwen‑2‑72B: 32K. Command‑R‑plus: 32K. Yi‑34B‑200K: 32K. Mixtral‑8×22B: 32K. Llama‑3.1‑8B: 32K. Others fall to 16K or below” (Table 3).
    - Many models fail to stay above threshold at their own claimed length; e.g., DBRX (claims 32K) drops to 8K effective length; several “32K” models are <4K effective (Together‑7B‑32K, LongChat‑7B, LongAlpaca‑13B; Table 3).
  - Retrieval tasks alone can be misleading:
    - Passkey retrieval and vanilla NIAH show near‑perfect scores for most models even up to 64K–128K (Tables 10–11). For instance:
      > “Llama‑3.1‑8B: 100% across all lengths in passkey retrieval (Avg 100.0)” (Table 10),
      yet its average over RULER drops to 77.0 at 128K (Table 3).
- Detailed behavior analyses (Sec. 5; Figs. 2–3)
  - Changing “needle” type reduces robustness:
    - Yi‑34B performs well with word–number needles but degrades when keys/values are `UUID`s; sometimes returns incomplete 32‑digit UUIDs at >128K (Fig. 2, left).
  - Hard distractors lower precision:
    - In `MK‑NIAH`, adding many distractor keys steadily reduces accuracy; with a “FULL haystack” of distractors at 256K, Yi drops by ∼40 points (Fig. 2, middle‑left). Errors often come from retrieving a nearby (incorrect) value—coarse rather than precise matching.
  - Multi‑item recall is fragile:
    - Increasing the number of required queries in `MQ‑NIAH` from 1 to 8 drops Yi by ~15 points at long lengths (Fig. 2, right). In `MV‑NIAH`, Yi often duplicates some answers while missing others (Fig. 2, middle‑right).
  - Multi‑hop tracing degrades with scale and complexity:
    - In `VT`, more hops consistently reduce accuracy as length grows; with more parallel chains, degradation is pronounced beyond 128K (Fig. 3, left and middle‑left). Common mistakes include returning empty strings or variables from other chains.
  - Aggregation reveals counting failures and prompt copying:
    - In `CWE`, Yi frequently copies the in‑context example verbatim at long lengths (>80% of outputs at 128K), a behavior also seen in LWM and LongAlpaca but less in Mixtral (Sec. 5).
    - In `FWE`, lowering `α` (harder counting) markedly reduces accuracy for Yi (Fig. 3, middle‑right). Some other models ignore the context and output high‑frequency English stopwords (e.g., “the”, “a”)—a sign of relying on parametric priors (Sec. 5).
  - Long‑context QA approaches no‑context behavior:
    - As distractors increase, Yi’s QA accuracy trends toward its no‑context baseline (Fig. 3, right), indicating hallucination and diminished use of provided context at large lengths.
- Ablations on scale, training length, and architecture (Sec. 6; Fig. 4)
  - Training length: Longer training windows usually help but not monotonically; e.g., `LWM‑1M` can be worse than `LWM‑512K` at 256K, possibly due to sub‑optimal adjustment to RoPE base frequency (Fig. 4, left/middle‑left).
    - “RoPE” (rotary positional embeddings) is a positional encoding; changing its base frequency is a common method to extend context, but requires careful training.
  - Model size: Bigger models degrade less and start higher; Yi‑34B ≫ Yi‑6B/9B (Fig. 4, middle‑right).
  - Architecture: RWKV‑v5‑7B and Mamba‑2.8B fall sharply by 8K and lag far behind Llama‑2‑7B even at short lengths (Fig. 4, right).
- Do the experiments support the claims?
  - Yes. The breadth of tasks (13 configurations across 4 categories), multiple lengths (4K–128K), and model coverage (17 aligned LLMs) give a robust picture. The contrast between near‑perfect NIAH (Tables 10–11) and much lower RULER averages (Table 3) convincingly shows that vanilla retrieval is insufficient to gauge long‑context competence.
  - Failure cases and robustness checks are explicit (Figs. 2–3), and ablations probe key design factors (Fig. 4). One subjective element is the threshold choice for “effective length” (85.6% = Llama‑2‑7B‑chat at 4K), which affects categorization but not the relative trends (Sec. 4; Table 3).

## 6. Limitations and Trade-offs
- Synthetic focus and proxy validity
  - RULER deliberately uses synthetic tasks to control length/difficulty and reduce parametric knowledge. While this isolates behavior, the paper acknowledges the need to verify correlation with realistic long‑context tasks (Sec. 8: “Lack of correlation with realistic long‑context tasks”).
- Position control
  - RULER v1 does not vary the exact position (depth) of key information to test “lost‑in‑the‑middle” effects within the same length; adding position control is future work (Sec. 8).
- Short‑context coverage
  - Chosen tasks are tuned so most models perform reasonably at 4K; the benchmark focuses on degradation with length. Harder short‑length tasks exist but are not reported here (Sec. 8).
- Prompt and hyperparameter sensitivity
  - Only limited exploration of prompt robustness and certain hyperparameters (e.g., variable name length in `VT`, vocabulary size in `CWE/FWE`) was performed (Sec. 8).
- Threshold choice for “effective context length”
  - The 85.6% threshold (Llama‑2‑7B‑chat at 4K) is a reasonable but ultimately arbitrary reference; different thresholds would shift the absolute effective lengths though not the qualitative trends (Sec. 4; Table 3).
- Compute and scope
  - Even with synthetic generation, testing 500 examples per task×length×model across 13 tasks and 6 lengths is compute‑intensive; broader coverage (e.g., 256K for all models) was only done in focused analyses (Yi‑34B; Sec. 5; Figs. 2–3).

## 7. Implications and Future Directions
- Reframing context‑window claims
  - A model’s advertised context size is not the same as its effective context size. Practitioners should demand multi‑behavior, multi‑length evidence like RULER’s: retrieval robustness, multi‑hop tracing, aggregation, and QA under distractors (Table 3; Sec. 5).
- Benchmarking practice
  - Vanilla NIAH or passkey tests (Tables 10–11) are necessary but far from sufficient. Benchmarks should include:
    - Multiple needle types and hard distractors,
    - Multi‑item recall (MV/MQ),
    - Aggregation/counting under controlled distributions,
    - QA with distractor paragraphs from the same domain.
- Model development guidance
  - Training implications suggested by ablations (Sec. 6; Fig. 4):
    - Scale helps: larger models better sustain performance at long lengths.
    - Training windows matter, but simply increasing context length isn’t a silver bullet; proper adaptation of positional encoding (e.g., RoPE base frequency) is crucial.
    - Non‑Transformer alternatives need substantial improvements to compete on long‑context tasks tested here.
  - Mitigating failure modes:
    - Resist prompt copying at long lengths (e.g., curriculum with varied demonstrations, anti‑copy objectives).
    - Improve multi‑item recall and distractor rejection (e.g., training with synthetic distractors, contrastive objectives).
    - Strengthen aggregation/counting (e.g., explicit counting heads, tool‑augmented counting, or hybrid retrieval+aggregation pipelines).
    - Reduce reliance on parametric priors in aggregation/QA via training that enforces context use (e.g., context‑faithfulness objectives).
- Future research with RULER
  - Add position‑controlled placements to probe depth effects (“lost in the middle”).
  - Bridge to realistic long‑document tasks (e.g., aligning RULER performance with real summarization or legal/biomedical QA).
  - Explore instruction‑following and reasoning at long length (not heavily tested here).
  - Expand to multimodal long‑context settings (images + text over long sequences).

Block‑quoted highlights for quick reference
- Table 3 (overall long‑context performance, 13 tasks):
  > “Gemini‑1.5‑Pro: wAvg(inc)=95.5 (1st), wAvg(dec)=96.1 (1st); GPT‑4: 89.0 (2nd), 94.1 (2nd); Llama‑3.1‑70B: 85.5 (4th), 93.7 (3rd). Effective lengths: GPT‑4=64K; Llama‑3.1‑70B=64K; Qwen‑2‑72B=32K; GLM‑4‑9B=64K.”
- Retrieval is easy; RULER is not:
  > “Almost all models have perfect score [in passkey retrieval and vanilla NIAH] at their claimed length” (Tables 10–11), yet many “fail to maintain performance above the Llama‑2‑7B baseline at their claimed length” on RULER (Table 3).
- Failure patterns at scale (Yi‑34B; Figs. 2–3):
  > “Increased distractors and multiple queries/values reduce accuracy substantially; models copy prompt examples at 128K in `CWE`; QA accuracy trends toward no‑context baselines as distractors increase.”

In sum, RULER offers a rigorous, behavior‑focused way to measure long‑context competence. It shows that genuine long‑context understanding is more than surviving a single “needle” test—and that many models are not there yet, especially when asked to trace, aggregate, and answer precisely at 32K–128K tokens and beyond.
