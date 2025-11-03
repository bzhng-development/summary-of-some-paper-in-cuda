# RULER: What’s the Real Context Size of Your Long-Context Language Models?

**ArXiv:** [2404.06654](https://arxiv.org/abs/2404.06654)
**Authors:** Cheng‑Ping Hsieh, Simeng Sun, Samuel Kriman, Shantanu Acharya, Dima Rekesh, Fei Jia, Yang Zhang, Boris Ginsburg
**Institutions:** 

## 🎯 Pitch

RULER is a groundbreaking benchmark that challenges long-context language models (LLMs) by testing their abilities beyond mere retrieval, including multi-hop tracing, aggregation, and question answering across up to 128K tokens. This study not only reveals that many models fail to deliver on their advertised context lengths but also provides a critical lens on real-world applications, guiding developers in selecting and training more reliable LLMs for complex, long-context tasks.

---

## 1. Executive Summary (2-3 sentences)
This paper introduces RULER, a synthetic, configurable benchmark that evaluates long‑context language models (LLMs) on more than simple retrieval, adding multi‑hop tracing, aggregation, and long-context question answering. Using RULER to test 17 models across up to 128K tokens, the study shows that many models claiming very long context windows degrade sharply as length and task complexity grow; only about half maintain satisfactory performance at 32K, and almost none reach their advertised limits.

## 2. Context and Motivation
- Problem and gap
  - Long‑context LLMs are increasingly advertised with very large context windows (32K–1M tokens), but commonly used evaluations (e.g., passkey retrieval and “needle‑in‑a‑haystack,” or NIAH) primarily test whether a model can find an explicit snippet in a long distractor text.
  - This leaves open whether models actually understand and use long contexts for tasks that require chaining references, aggregating dispersed signals, or answering questions amid distractors.
  - The paper explicitly highlights this shortfall:
    > "Despite achieving nearly perfect accuracy in the vanilla NIAH test, almost all models exhibit large performance drops as the context length increases." (Abstract)
- Why it matters
  - Real-world uses (long documents, codebases, multi-document QA, many-shot contexts) require more than mere retrieval—they require tracing entities across references, combining evidence that is spread out, and ignoring plausible but irrelevant text.
  - Miscalibrated claims about “context size” can mislead practitioners about what lengths are usable in practice.
- Prior approaches and shortcomings
  - Synthetic retrieval tests like passkey retrieval and vanilla NIAH are widely used (e.g., Kamradt, 2023) but focus on exact-match search; they do not probe robustness to needle variants, multi-item recall, or tasks that rely on aggregation and reasoning.
  - Realistic long-context benchmarks exist (ZeroSCROLLS, LongBench, L-Eval; Table 1), but they often depend on parametric knowledge (knowledge stored in the model’s weights) and do not allow precise control over length and task difficulty.
- How this paper positions itself
  - RULER is a synthetic benchmark designed to:
    - Control sequence length and task complexity precisely.
    - Reduce interference from parametric knowledge by using synthetic inputs.
    - Cover four categories—retrieval, multi‑hop tracing, aggregation, and question answering—thereby testing behaviors “beyond searching from context” (Section 1; Table 1; Section 3).

## 3. Technical Approach
RULER is a configurable suite of synthetic tasks. Each task can be scaled in length and complexity by adjusting quantities like the number of distractors, hops, or distribution sharpness. Tasks are auto‑generated to avoid reliance on real‑world knowledge while still demanding nontrivial use of long context (Section 3; Table 2).

Step-by-step overview

1) Retrieval tasks: extended NIAH family (Section 3.1; Table 2)
   - Concept: Insert “needles” (key–value statements) into a long “haystack” (distractor text). A query at the end asks for specific values, requiring the model to scan and retrieve.
   - Four variants probe different retrieval stresses:
     - `S‑NIAH` (Single): One needle to retrieve; keys/values can be words, numbers, or UUIDs; haystack can be essays or repeated noise (Table 2; also Appendix B).
     - `MK‑NIAH` (Multi-keys): Multiple needles (hard distractors) appear, but the query asks for only one; includes an extreme version where the haystack is “full of irrelevant needles” (Section 3.1).
     - `MV‑NIAH` (Multi-values): The same key appears multiple times with different values; the model must list all values for that key.
     - `MQ‑NIAH` (Multi-queries): Several distinct keys are queried; the model must return all corresponding values without missing any.
   - Why these design choices: They test (a) robustness to needle and haystack types, (b) ability to ignore hard distractors, and (c) high recall when multiple items must be retrieved (Section 3.1).

2) Multi‑hop tracing: variable tracking (`VT`) (Section 3.2; Table 2)
   - Concept: A minimal proxy for coreference chain resolution. The input defines variables linked by assignments spread across the long context (e.g., `X2 = X1`, `X3 = X2`, …). The task asks for all variables pointing to a particular value.
   - Complexity control: Increase number of chains or hops (length of the chain).
   - Why it matters: Real long texts often require following entities and references over distance; this tests “tracing with skipped connections” beyond simple exact‑match retrieval.

3) Aggregation tasks (Section 3.3; Figure 1; Table 2)
   - Concept: The model must aggregate frequencies across the entire context to identify most common words.
   - Two flavors:
     - `CWE` (Common Words Extraction): Inputs contain a fixed set of “common” words and many “uncommon” words sampled uniformly; output the top‑K common words.
     - `FWE` (Frequent Words Extraction): Words are sampled from a heavy‑tailed `Zeta` distribution; the model must find the top‑3 by frequency. The parameter `α` controls how peaked the distribution is (Figure 1). Lower `α` makes top words less distinct, increasing difficulty.
   - Why it matters: Summarization-like skills require combining dispersed evidence, not locating a single snippet.

4) Question answering with distractors (Section 3.4; Table 2)
   - Concept: Standard QA datasets designed for short passages (SQuAD, HotpotQA) are “stretched” into long contexts by inserting the gold paragraph(s) among many distractors, then asking the original questions.
   - Why it matters: This is a realistic “find and reason over the relevant passage(s) in a long pile” setting.

Evaluation methodology (Section 4)
- Models: 17 aligned long‑context LLMs (15 open‑source plus GPT‑4 and Gemini 1.5), with claimed windows from 32K to 1M and sizes from 7B to 8×22B MoE (Appendix A).
- Setup: vLLM inference, BF16, greedy decoding on 8×A100 GPUs (Section 4).
- Lengths: 4K, 8K, 16K, 32K, 64K, 128K tokens; 500 examples per task per length; task prompts use model-specific chat templates and an “answer prefix” to elicit direct answers (Section 4; Appendix D).
- Metrics: “Recall-based accuracy” of producing the target outputs. For multi-item retrieval tasks, missing any required item counts as an error (Section 4).
- Task selection: 13 representative tasks chosen from 18 via a correlation analysis to reduce redundancy while keeping behavioral diversity (Appendix C; Figure 5).
- Aggregation across lengths: Two weighted averages simulate different usage distributions—`wAvg.(inc)` prioritizes longer inputs, `wAvg.(dec)` prioritizes shorter ones (Section 4).
- “Effective context length”: A pass/fail threshold defined as Llama‑2‑7B (chat) at 4K on RULER, 85.6% average over tasks (Table 3). A model’s effective length is the longest length at which its average exceeds this threshold (Section 4; Table 3).

## 4. Key Insights and Innovations
1) A broader, configurable lens on long‑context ability (fundamental)
   - Novelty: RULER spans four behavior categories—retrieval, multi‑hop tracing, aggregation, and QA—each with controllable difficulty (Section 3; Table 2; Figure 1). Prior long‑context evaluations mostly focused on retrieval (e.g., NIAH).
   - Significance: It disentangles “can the model find a string” from “can it track references or aggregate dispersed signals,” providing a more holistic and stress‑test‑able view of long‑context competence.

2) Measuring “effective context length” instead of “claimed length” (conceptual/diagnostic)
   - Novelty: A practical notion of effective context size tied to a fixed, transparent performance threshold (Section 4; Table 3).
   - Significance: It exposes a reality gap: many models fail far below their advertised windows. The study reports:
     > “While these models all claim context sizes of 32K tokens or greater, only half of them can maintain satisfactory performance at the length of 32K.” (Abstract; Table 3 supports the finding.)

3) Behavioral failure modes at scale (empirical)
   - Novelty: The paper systematically documents how errors change with length and complexity—e.g., incomplete multi‑item recall, non‑robustness to “needle” formats, copying the in‑context example verbatim, and reversion to parametric knowledge instead of using context (Section 5; Figures 2–3).
   - Significance: These patterns suggest that long‑context models can shift to brittle heuristics as inputs grow, a crucial insight for safety and reliability.

4) Architecture/training insights (empirical)
   - Novelty: Comparative analyses show that longer training windows are helpful but not monotonic (e.g., 1M-trained models can underperform at 256K), larger model size correlates with better long‑context use, and non‑Transformer families (RWKV, Mamba) lag in this setting (Section 6; Figure 4).
   - Significance: These results give practical guidance on how to develop and select long‑context models.

## 5. Experimental Analysis
- Evaluation design (Section 4; Appendix B–D)
  - 13 tasks across 4 categories; lengths 4K–128K; 500 examples per length; greedy decoding.
  - Models compared with consistent prompting using model-specific templates (Appendix D).
  - Metrics: accuracy on the required outputs; weighted averages `wAvg.(inc)` and `wAvg.(dec)` for comparisons across lengths.

- Main quantitative results (Table 3)
  - Overall averages across all 13 tasks:
    - `Gemini‑1.5‑Pro`: 96–97% across all lengths; effective length >128K; top in both `wAvg.(inc)`=95.5 and `wAvg.(dec)`=96.1.
    - `GPT‑4`: 96.6% at 4K → 81.2% at 128K; effective length 64K; `wAvg.(inc)`=89.0, `wAvg.(dec)`=94.1 (2nd in both).
    - Top open‑source at 128K:
      - `Llama3.1‑70B`: 100→78.9% on Retrieval-only (Table 13) and 96.5→66.6% on all tasks (Table 3); effective length 64K; `wAvg.(dec)`=93.7 (3rd overall).
      - `Qwen2‑72B`: 96.9% at 4K → 53.7% at 128K; effective 32K; `wAvg.(dec)`=92.3 (4th overall).
    - Many models claiming very large context windows falter earlier than claimed:
      - `GLM4‑9B (1M)`: effective 64K.
      - `GradientAI/Llama3‑70B (1M)`: effective 16K.
      - `LWM‑7B (1M)`: below the 4K threshold even at 4K (Table 3).
  - Takeaway: The gap between claimed and effective context is substantial. The study concludes:
    > “…almost all models fall below the threshold before reaching the claimed context lengths.” (Section 4; Table 3)

- Retrieval vs. RULER difficulty
  - Most models achieve near‑perfect scores on passkey retrieval and vanilla NIAH at long lengths (Appendix E; Tables 10–11). For example:
    - `GPT‑4` and `Gemini‑1.5` score 100% across all lengths.
    - Several open‑source models maintain 100% on passkey retrieval up to 128K (e.g., `GLM4‑9B`, `Yi‑34B`).
  - But on RULER’s broader tasks, performance drops with length and complexity (Table 3). This contrast demonstrates that “can find a single needle” is insufficient evidence of robust long‑context use.

- Detailed error analyses on `Yi‑34B‑200K` (Section 5; Figures 2–3)
  - Non‑robustness to needle type (Figure 2, left): Accuracy drops when switching from word–number to UUID needles, especially beyond 128K, sometimes failing to output complete 32‑digit UUIDs.
  - Susceptibility to distractors (MK‑NIAH; Figure 2, middle-left): As the number of distractor keys rises (up to the “FULL” setting), accuracy drops by ~40 points at 256K; errors often pick values near the target range, suggesting coarse rather than precise matching.
  - Incomplete multi‑item recall (MV‑NIAH and MQ‑NIAH; Figure 2, middle-right and right): With more queries or more values, accuracy drops by ~15 points at 256K (MQ‑NIAH), and outputs may duplicate some values while missing others.
  - Copying from context (Section 5): In `CWE` with one in‑context example, >80% of outputs at 128K are copies of the example rather than the actual answer; removal causes copying of the beginning of the input (likely due to attention sinks).
  - Unreliable multi‑hop tracing (Figure 3, left and middle-left): Performance drops with more hops and more chains, especially beyond 128K; frequent mistakes include returning empty strings or variables from other chains.
  - Aggregation difficulty (Figure 3, middle-right): In `FWE`, lowering `α` (less separation among top frequencies) degrades accuracy; models struggle to reliably rank near‑tied frequencies at long lengths.
  - QA reverts toward no‑context behavior (Figure 3, right): As distractors increase, prediction quality approaches a “no-context” baseline, with hallucinations and answers unrelated to the supplied documents.

- Architecture, size, and training length effects (Section 6; Figure 4)
  - Training window: Longer training context helps on average, but is not monotonic; e.g., `LWM‑1M` underperforms `LWM‑512K` at 256K, possibly due to insufficient training with a new RoPE base frequency and length extrapolation limits (Figure 4, left).
  - Model size: Within the `Yi` family trained to 200K, `Yi‑34B` outperforms `Yi‑9B` and `Yi‑6B` across lengths (Figure 4, middle-right), improving both short‑length absolute accuracy and degradation slope.
  - Architecture: Non‑Transformer models (`RWKV‑v5‑7B`, `Mamba‑2.8B`) lag considerably behind `Llama2‑7B` at 1–8K (Figure 4, right).

- Do results support the claims?
  - Yes. The quantitative gaps between easy retrieval tasks and RULER aggregates (Tables 10–11 vs. Table 3) clearly show that simple retrieval does not translate into robust long‑context competence.
  - The effective length analysis and weighted scores provide fairer rankings across use cases.
  - The rich error analysis (Figures 2–3) identifies concrete failure modes, not just performance drops.

## 6. Limitations and Trade-offs
- Synthetic proxy tasks vs. realism
  - Strength: Full control over length and difficulty; reduced interference from parametric knowledge (Section 3).
  - Limitation: Proxies for realistic tasks (e.g., `VT` for coreference, `CWE/FWE` for summarization) are not validated for correlation with real-world long‑context tasks (Section 8: “Lack of correlation with realistic long-context tasks”).
- Position control
  - Current RULER reports per-length scores but does not control or analyze depth-level position effects (e.g., “lost in the middle”); this is planned for future support (Section 8).
- Short‑context difficulty not explored
  - The chosen configurations aim for “most models perform decently at 4K” to study degradation with length; harder short‑context variants are possible but omitted (Section 8).
- Prompt robustness and hyperparameters
  - Limited exploration of prompt-format sensitivity and fixed hyperparameters (e.g., variable name lengths, vocabulary size in `CWE/FWE`) (Section 8).
- Compute and evaluation bounds
  - Main tables go up to 128K; some models claim 200K–1M. Deeper stress tests (e.g., >128K for all models) are not feasible within the evaluation budget (Section 4; the paper notes interest in “pressure testing” Gemini further).

## 7. Implications and Future Directions
- How this changes the landscape
  - RULER reframes “long‑context ability” from a single retrieval check to a spectrum of behaviors—retrieval robustness, multi‑hop tracing, aggregation, and QA with distractors—measured across controlled lengths and complexities. This encourages more honest, behavior-based reporting of “effective” vs. “claimed” context size.
- Practical guidance
  - When selecting a long‑context model for production:
    - Evaluate on tasks matching your use case (e.g., aggregation for summarization-like workloads, multi‑hop for entity tracking).
    - Expect performance declines with length and with increased distractors; consider chunking, retrieval-augmentation, or structured prompting for stability.
    - Prefer larger models and models trained (and validated) at the target lengths, but verify—training to very long windows does not guarantee monotonic gains (Section 6; Figure 4).
- Research directions enabled
  - Better position‑aware evaluations (lost‑in‑the‑middle analyses) integrated into RULER (Section 8).
  - Designing synthetic tasks with validated correlation to real long‑context reasoning (e.g., discourse, multi‑doc summarization).
  - Methods to reduce copying and reliance on parametric knowledge at long lengths (e.g., debiasing attention sinks, better decoding strategies, or training objectives that penalize shallow heuristics).
  - Architecture search and training curricula for robust multi‑item recall and aggregation over 100K+ tokens; probing why non‑Transformers lag in RULER (Figure 4, right).
- Potential applications
  - Model selection and regression testing for long‑document workflows (legal, scientific, code repositories).
  - Curriculum design for long‑context fine‑tuning that explicitly targets RULER behaviors (multi‑hop chains, distractor‑heavy settings, near‑tied frequency discrimination).
  - Automated evaluation dashboards that track “effective context length” and weighted performance for organizations adopting long‑context LLMs.

Key supporting citations from the paper
- Problem significance:
  > “While these models all claim context size of 32k tokens or greater, our results indicate that only half of them can effectively handle sequence length of 32K…” (Section 4; Table 3)
- Failure beyond retrieval:
  > “Despite achieving nearly perfect performance on the passkey retrieval and the vanilla NIAH test… all of them exhibit large degradation in RULER as sequence length increases…” (Section 4; Appendix E; Table 3)
- Behavioral failures:
  > “…increased reliance on parametric knowledge and the increased tendency to copy from context for non-retrieval tasks.” (Section 5; Figures 2–3)
- Architecture/training effects:
  > “…larger model sizes positively correlate with better long-context capabilities… non-Transformer architectures… still lag behind Transformer by large margins on RULER.” (Section 6; Figure 4)

Pointers to figures and tables for deeper reading
- Task design and examples: Table 2; Figure 1 (word frequency distributions).
- Model roster and main results: Table 3 (effective vs. claimed length; weighted rankings).
- Retrieval vs. RULER contrast: Tables 10–11 (near-perfect retrieval) vs. Table 3 (broad-task degradation).
- Failure analyses: Figures 2–3 (Yi‑34B across task variants and complexities).
- Architecture, size, training length: Figure 4 (LWM series, Yi sizes, RWKV/Mamba vs. Transformer).
- Task selection and independence: Appendix C; Figure 5 (correlation heatmap).
- Prompt formats and answer prefixes: Appendix D.
