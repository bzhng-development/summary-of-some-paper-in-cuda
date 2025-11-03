# Sleep-time Compute: Beyond Inference Scaling at Test-time

**ArXiv:** [2504.13171](https://arxiv.org/abs/2504.13171)

## 🎯 Pitch

This paper introduces 'sleep-time compute,' a novel paradigm where large language models anticipate and process reusable context offline—before queries arrive—reducing redundant computation and enabling rapid, low-cost responses at test-time. By precomputing a distilled, query-agnostic version of context, the technique slashes test-time inference costs by up to 5× without sacrificing accuracy and even achieves superior results when scaled, shifting the fundamental compute–accuracy trade-off. This approach addresses critical latency and cost bottlenecks in real-world, stateful LLM applications like document QA and software agents, unlocking more efficient and responsive AI systems.

---

## 1. Executive Summary
This paper introduces sleep-time compute: an offline, pre-query reasoning stage where a language model processes reusable context (like a document or codebase) to produce a distilled, query-agnostic “learned context” that can be reused when actual questions arrive. Across math and software-engineering benchmarks, sleep-time compute shifts the compute–accuracy Pareto frontier: it achieves similar or higher accuracy with substantially fewer test-time tokens (about 5× fewer on two math benchmarks; Figures 3–4) and further improves accuracy when its own offline budget is scaled (up to +13% and +18%; Figures 7–8).

## 2. Context and Motivation
- Problem addressed
  - Modern test-time scaling lets LLMs “think longer” during inference (e.g., longer chains of thought, multiple samples), but it increases latency and cost and repeatedly recomputes the same facts whenever related queries are asked about the same context (Section 1).
  - Typical evaluation assumes a stateless setting where the context and user query are given together at test time, which wastes idle time between user interactions and cannot amortize repeated work across related queries (Section 1; Figure 1).

- Why this matters
  - Many real applications are stateful: document QA, coding assistants, and conversational agents all operate over persistent, reusable contexts (Section 1).
  - Reducing test-time latency and cost without sacrificing accuracy unlocks responsive products and agentic workflows while preserving the benefits of test-time reasoning.

- Prior approaches and gaps
  - Sequential test-time scaling (longer reasoning per query) and parallel scaling (multiple samples, e.g., `pass@k`) both operate only at test time and redo similar context processing for each query (Section 2).
  - Speculative decoding reduces decoding latency but does not precompute reusable reasoning over contexts; it still relies on test-time token prediction (Section 2).
  - Traditional precomputation/caching exists outside LLMs; this work proposes an LLM-native way to preprocess and store reasoning in text form that is actually used as input later (Section 2).

- Positioning
  - Sleep-time compute moves a portion of the inference-time reasoning earlier, into periods when the model is idle, to create a “re-represented” context that accelerates and stabilizes later answers (Section 3; Figure 1).
  - It complements both sequential and parallel test-time scaling instead of replacing them; it constitutes a new axis—when the compute happens.

## 3. Technical Approach
Sleep-time compute reframes inference as two phases: an offline “sleep” phase and an online “test” phase (Section 3).

- Core formulation
  - Standard test-time compute: give the model both the user query and context, then reason: `T_B(q, c) → a`, where `T` denotes the test-time reasoning method under budget `B` (Section 3).
  - Sleep-time compute: before any query arrives, reason over the context alone to produce a better representation: `S(c) → c′`. Later, answer with a smaller budget: `T_b(q, c′) → a` with `b ≪ B` (Section 3).
  - If multiple queries `q1…qN` share the same `c`, the same `c′` can be reused, amortizing its cost (Section 3; Figure 1).

- What is `c′` in practice?
  - A query-agnostic, text-form “learned context” that contains predictions, intermediate calculations, definitions, subgoals, and other inferences likely to be useful for many future questions about `c` (Figure 1, “Learned Context” panel).
  - It is produced by prompting the model to “rethink” the context and store its inferences (Appendix K).

- Implementation details
  - Function-calling interface for offline processing (Appendix K):
    - `rethink_memory(new_memory, target_block_label, source_block_label)` replaces the current memory block with a refined version (Listings 1–2).
    - The model may call `rethink_memory` up to 10 times to iteratively refine `c → c′`; it ends with `finish_rethinking_memory`.
  - Prompts:
    - A sleep-time prompt that explicitly instructs the model to pre-compute useful quantities and anticipate possible question directions (Figure 18 for AIME; Figure 17 for the generic “Letta-Offline-Memory” agent).
    - Test-time prompts with adjustable “verbosity” control for non-reasoning models to vary budget (Figures 12–16).
  - Scaling sleep-time compute:
    - Non-reasoning models (e.g., `gpt-4o`, `gpt-4o-mini`): run `k` parallel generations of `c′` and concatenate them at test time, effectively ensembling offline (Section 5.2; Figure 7).
    - Reasoning models (e.g., `o1`, `o3-mini`): vary “reasoning effort” during sleep-time generation (Figure 8).
  - Parallel baselines:
    - `pass@k` at test time is used as a parallel-scaling baseline; it assumes access to an oracle verifier, which sleep-time compute does not rely on (Section 5.1; Figures 5–6).
  - Cost model for amortization:
    - To analyze end-to-end cost with multiple queries per context, a linear model upweights test-time tokens by a factor `t=10` (to reflect their higher latency/throughput cost), while sleep-time tokens are weighted by 1 (Section 5.3; Figure 9; footnote 4).

- Example to build intuition (Figure 1)
  - Context: text describing numbers of balls by type/color.
  - Sleep-time: the model computes intermediate totals (e.g., number of tennis balls, fraction marked) and stores them in `c′`.
  - Test-time: when asked any related question (e.g., “How many marked indigo tennis balls?” or “How many tennis balls?”), the model reads `c′`, uses a small budget to retrieve/compose numbers, and answers quickly.

## 4. Key Insights and Innovations
- A new inference-time dimension: when to compute
  - Innovation: shift part of reasoning from the online moment (test time) to earlier offline windows (“sleep time”), leveraging persistent contexts. This is conceptually different from both sequential and parallel test-time scaling because it builds a reusable, text-form representation (Section 3; Figure 1).
  - Significance: reduces online latency and cost while maintaining or improving accuracy; amortizes compute across related queries.

- Token-space representation learning over context
  - Innovation: treat “learned context” `c′` as a natural-language representation computed offline, not parameters or embeddings (Section 7, “Sleep-time compute as representation learning over tokens”).
  - Significance: keeps the approach model-agnostic and deployable with current LLM APIs and prompts (Appendix K).

- New stateful benchmarks and amortization setting
  - Innovation: construct `Stateful GSM-Symbolic` and `Stateful AIME` by splitting problems into `context` and `query` to emulate realistic stateful workflows (Section 4.1; Figure 2; Appendix J for AIME details). Introduce `Multi-Query GSM-Symbolic` with multiple questions per shared context (Appendix C; Figure 20; Table 1).
  - Significance: enables systematic evaluation of offline reuse and amortization.

- When sleep-time helps most: query predictability
  - Insight: the benefit correlates with how predictable the query is from the context. A log-probability measure of `P(query | context)` (estimated by `Llama2-70B base`) predicts larger gains when queries are more predictable (Section 5.4; Figure 10; Appendix E).
  - Significance: provides guidance on when to allocate offline compute.

## 5. Experimental Analysis
- Evaluation setup
  - Datasets (Section 4.1):
    - `Stateful GSM-Symbolic`: derived from GSM-Symbolic P1 (5k items) and P2 (2.5k items), where original problems are split into context and question (Figure 2).
    - `Stateful AIME`: 60 problems from AIME 2024 and 2025, split into context and final question (Appendix J; Figures 23–24).
    - `Multi-Query GSM-Symbolic`: multiple synthetic questions per context generated by `o3-mini` to study amortization (Appendix C; Figure 20; Table 1).
    - `SWE-Features`: 33 PR tasks from ComfyUI and Aider repos requiring multi-file edits and new features; context is related PRs; evaluation by F1 on modified-file sets (Section 6; Appendix D, G–H).
  - Models (Section 4.2):
    - GSM-Symbolic: `gpt-4o-mini`, `gpt-4o`.
    - AIME: `o1`, `o3-mini`, `Claude 3.7 Sonnet (Extended Thinking)`, `DeepSeek-R1`. For `R1`, test-time budget is controlled using “budget forcing” and an extension prompt (Muennighoff et al., 2025).
  - Baselines:
    - Standard test-time scaling (sequential budgets via verbosity or API “reasoning effort”).
    - Parallel scaling via `pass@k` (Figures 5–6).
    - Context-only ablation: only `c` is given and the model must guess a plausible question+answer (Appendix I; Figures 21–22).

- Main quantitative results
  - Pareto improvements on GSM-Symbolic (Figure 3):
    - Sleep-time compute achieves accuracy comparable to test-time-only while using roughly 5× fewer test-time tokens at low budgets; the shaded region marks the Pareto improvement.
    - At very high test-time budgets, test-time-only can slightly outperform sleep-time (likely because `c′` includes extra, potentially distracting information not specific to the query; discussion in Section 5.1).
  - Pareto improvements on AIME (Figure 4):
    - Similar 5× token reductions at comparable accuracy for `o3-mini`, `Claude 3.7 Sonnet`, and `R1`; `o1` shows smaller gains but still benefits at low budgets.
    - Quote: “Applying sleep-time compute allows models to reach similar levels of performance with much less compute at test-time” (Figure 4 caption).
  - Sleep-time vs. parallel `pass@k`:
    - GSM-Symbolic (Figure 5) and AIME (Figure 6): sleep-time compute generally Pareto-dominates `pass@k` for the same test-time token budgets, despite `pass@k` assuming an oracle verifier.
  - Scaling sleep-time compute:
    - GSM-Symbolic (Figure 7): increasing the number of parallel sleep-time generations (`k`) improves accuracy by up to 13% at similar test-time budgets. Gains saturate around `k=5`, with `k=10` sometimes worse (information overload).
    - AIME (Figure 8; Figure 26): increasing sleep-time reasoning effort further shifts the Pareto outward, improving accuracy by up to 18%.
  - Amortization across multiple queries (Figure 9):
    - Using the cost model with `t=10` (test-time tokens 10× more expensive), average cost per query drops by up to 2.5× when there are 10 queries per context compared to single-query baselines.
  - When does it help most? Predictability analysis (Section 5.4; Figure 10):
    - Binning GSM-Symbolic problems into 5 quantiles by `P(query | context)`, the accuracy gap (sleep-time minus test-time-only) grows monotonically with predictability.
  - Context-only ablation (Appendix I; Figures 21–22):
    - Sleep-time compute far outperforms the context-only baseline, showing that the task cannot be solved by merely guessing the question from the context.
  - SWE-Features case study (Section 6; Figure 11):
    - At low budgets, sleep-time compute yields higher F1 and requires about 1.5× fewer test-time tokens to match performance. At high budgets, test-time-only performs better, with higher precision and similar recall—possibly because sleep-time exploration leads to more files being edited, hurting precision.

- Representative numeric claims (from Abstract, Section 5, and Figures)
  - Quote: “reducing the test-time compute needed to achieve the same accuracy by ∼5× on Stateful GSM-Symbolic and Stateful AIME” (Abstract; Figures 3–4 show the gap).
  - Quote: “scaling sleep-time compute… increases accuracy by up to 13% on Stateful GSM-Symbolic and 18% on Stateful AIME” (Abstract; Figures 7–8).
  - Quote: “amortizing sleep-time compute… decreases the average cost per query by 2.5×” (Abstract; Figure 9).

- Do the experiments support the claims?
  - Yes, across different model families and tasks: both non-reasoning and reasoning models benefit (Figures 3–4); against both sequential and parallel test-time baselines (Figures 5–6); and the benefit is largest where the mechanism should help most—when queries are predictable from context (Figure 10).
  - Caveat: at very high online budgets, test-time-only sometimes catches up or wins (Figure 3 right, Figure 11), consistent with the idea that query-specific reasoning can surpass generic offline `c′` if enough test-time compute is available.

## 6. Limitations and Trade-offs
- Assumption of reusable, predictable context
  - Sleep-time compute is most effective when future queries are somewhat predictable from the context, enabling useful precomputation (Section 5.4; Figure 10). If queries are orthogonal or adversarial, the offline effort may add noise or even distract the model at test time (Section 5.1 discussion).

- Two-phase simplification
  - Experiments assume two phases (sleep/test). Real agentic systems have many rounds with changing context and variable idle windows; the method needs policies for when to recompute `c′` and how to update it incrementally (Section 7, “Extending…”).

- Prompt-length and consolidation issues
  - Concatenating many offline generations (`k=10`) can degrade performance (Figure 7), likely due to context-window crowding and contradictions among offline drafts.

- Cost accounting and infrastructure
  - Offline compute is not free; benefits materialize when:
    - Latency at test-time is expensive (hence `t=10` weighting in Figure 9).
    - The same context serves multiple queries (Multi-Query GSM-Symbolic).
  - Storage, retrieval, and versioning of `c′` across contexts and time are left as system-design concerns.

- Dataset construction choices
  - Multi-Query dataset uses synthetic question generation by `o3-mini` (Appendix C), which may bias the amortization findings toward predictable queries.
  - SWE-Features evaluation uses F1 on modified file sets rather than end-to-end functionality; it correlates with success but is an indirect measure (Section 6; Appendix D).

- Limited control of proprietary models
  - For some APIs (e.g., DeepSeek-R1), test-time budget is approximated via a “budget forcing” technique (Section 5.1), which may introduce variability.

## 7. Implications and Future Directions
- How this changes the landscape
  - Introduces “when” as a new axis for inference compute: not just “how much” or “how parallel,” but how to shift reusable reasoning offline to lower latency and cost without sacrificing accuracy.
  - Bridges LLM reasoning with classic ideas of precomputation/caching while keeping everything in natural-language token space, making it operational with current APIs.

- Practical applications
  - Document QA and analytics portals: preprocess documents into FAQs, derived facts, and structured summaries to speed up queries.
  - Coding agents: pre-map repository structures, summarize module interfaces, index common patterns, and anticipate debugging plans (Section 1; Section 6 case study).
  - Conversational assistants: proactively condense and reconcile user history into task-specific summaries during idle periods.

- Research opportunities
  - Allocation policy: learn when to deploy sleep-time compute vs. extra test-time compute based on a learned predictability estimator (Section 7).
  - Representation design: move from free-form text `c′` to more structured, modular artifacts (graphs, tables, program sketches) while keeping LLM-readability.
  - Continuous updating: handle evolving contexts with efficient incremental rethinking, de-duplication, and contradiction resolution.
  - Integration with retrieval and verifiers: combine `c′` with retrieval-augmented generation and lightweight verifiers to avoid drift and reduce distraction.
  - Synthetic data generation at scale: use sleep-time compute to cheaply produce families of related training examples or distilled knowledge that amortize across tasks (Section 7, “Synthetic data generation via sleep-time compute”).

> Overall, the paper demonstrates that sleep-time compute can “produce a Pareto improvement in the test-time compute vs. accuracy curve” and “reduce the test-time compute needed to achieve the same accuracy by ∼5×” (Abstract; Figures 3–4), especially when queries are predictable from persistent context (Figure 10), and that scaling the offline budget yields additional gains (up to +13% and +18%; Figures 7–8).
