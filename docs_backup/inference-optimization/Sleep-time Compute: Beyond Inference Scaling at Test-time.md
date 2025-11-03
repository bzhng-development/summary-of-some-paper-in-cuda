# Sleep-time Compute: Beyond Inference Scaling at Test-time

**ArXiv:** [2504.13171](https://arxiv.org/abs/2504.13171)
**Authors:** Kevin Lin, Charlie Snell, Yu Wang, Charles Packer, Sarah Wooders, Ion Stoica, Joseph E. Gonzalez
**Institutions:** Letta, UC Berkeley

## 🎯 Pitch

This paper introduces 'sleep-time compute,' an innovative approach allowing language models to harness idle periods by pre-computing inferences for persistent contexts, significantly reducing test-time computation by up to 5× while achieving up to 18% higher accuracy. This method is transformative for interactive systems such as coding assistants and conversational agents, as it decreases latency and costs, enhancing performance in real-world, stateful applications where shared context and predictability are key.

---

## 1. Executive Summary
This paper introduces sleep-time compute, a way for language models to pre-compute useful inferences about a persistent context before any user question arrives, then reuse those inferences to answer later questions with far less test-time computation. Across math reasoning benchmarks and a realistic software-engineering (SWE) agent setting, sleep-time compute shifts the accuracy–compute Pareto frontier: it achieves the same accuracy with roughly 5× fewer test-time tokens and, when scaled, raises peak accuracy by up to 13% on Stateful GSM-Symbolic and 18% on Stateful AIME (Abstract; Figures 3–4, 7–8, 11).

## 2. Context and Motivation
- The problem addressed
  - Modern language models often gain accuracy by spending more compute at inference time (“test-time compute”), e.g., longer chain-of-thought or multiple samples. This boosts quality but increases latency and cost (Introduction; Related Work).
  - Current test-time scaling implicitly assumes a stateless setting where the question and all relevant background “context” arrive together. In many applications (document QA, coding assistants, conversational agents), there is persistent context available between user turns that the model could analyze while idle (Introduction; Figure 1).

- Why it matters
  - High-latency, high-cost inference hampers interactive systems (minutes per answer, dollars per query; Introduction, footnote 1).
  - Many real-world deployments are stateful: multiple related questions are asked over the same context. Recomputing similar inferences for each question is wasteful (Introduction; Figure 1).

- Shortcomings of prior approaches
  - Sequential test-time scaling increases time per query; parallel scaling (e.g., pass@k) often assumes a strong verifier to pick the best candidate, which may be unrealistic for many tasks (Related Work; Figures 5–6).
  - Speculative decoding reduces decoding latency, but it predicts future tokens of the current answer; it does not reorganize the context for future, unknown questions (Related Work).

- Positioning relative to existing work
  - Sleep-time compute introduces an orthogonal axis to test-time scaling: pre-compute over the context between user interactions. It shares intuition with pre-computation/caching from systems research (Related Work) but operationalizes it for LLMs in natural language form, not hidden states or parameters (Discussion: “representation learning over tokens”).

## 3. Technical Approach
Sleep-time compute reframes inference in stateful applications by separating the persistent context from the eventual question and allocating compute to the idle periods between user turns.

- Core abstraction
  - Decompose an input into a persistent `context` (`c`, e.g., a codebase or document) and a later `query` (`q`, a user’s question about that context) (Section 3).
  - Standard test-time baseline: apply a test-time compute method `T` with budget `B` to answer `q` given `c`: `T_B(q, c) → a` (Section 3).
  - Sleep-time compute:
    - During idle time (no query yet), transform the context into a “re-represented” context `c'` by drawing inferences that might be useful later: `S(c) → c'` (Section 3).
    - At test time, answer with a smaller budget `b << B` using the enriched context: `T_b(q, c') → a` (Section 3).
    - If multiple questions `q1, q2, …` share the same context, reuse the same `c'` to amortize the cost (Section 3; Figure 1).

- What is `c'` in practice?
  - It is a natural-language summary/derivation of useful facts, intermediate results, and anticipated sub-computations that could serve many plausible questions (Figure 1; Appendix A prompts; Appendix K implementation details).
  - The paper implements `S(c)` with a function-calling loop that repeatedly “rethinks” memory: the model calls `rethink_memory(new_memory, source, target)` to update a “rethink memory block,” then calls `finish_rethinking_memory()` to stop (Appendix F and K). The loop is capped at 10 calls (Appendix K).

- Scaling levers
  - Scaling test-time compute (for baselines and sleep-time):
    - Sequential scaling: vary “verbosity” prompts for non-reasoning models (GPT-4o/mini), from short direct answers to longer, double-checked reasoning (Appendix A; Figure 3).
    - Reasoning models (o1, o3-mini, Claude 3.7 Sonnet, DeepSeek-R1): vary built-in reasoning effort or apply “budget forcing” for R1 (Section 5.1; Figure 4; Muennighoff et al., 2025).
    - Parallel scaling baseline: `pass@k` samples in parallel (Figures 5–6).
  - Scaling sleep-time compute:
    - For non-reasoning models: generate `k` parallel versions of `c'` (`c1'…ck'`) from the same `c` and concatenate them for test time (Section 5.2; Figure 7).
    - For reasoning models: increase the sleep-time reasoning effort when creating `c'` (Section 5.2; Figure 8).

- Experimental datasets and settings that make statefulness explicit
  - Stateful GSM-Symbolic (P1/P2): each GSM-Symbolic problem is split into `context` and `query` (Figure 2; Section 4.1).
  - Stateful AIME: AIME 2024+2025 questions split into context (all but last sentence) and question (last sentence), with manual fixes for edge cases (Appendix J).
  - Multi-Query GSM-Symbolic: for each context, generate multiple plausible questions/answers to study amortization across queries (Section 4.1; Appendix C; Figure 20).
  - SWE-Features: 33 pull requests (PRs) requiring multi-file feature additions in two repos; evaluate whether an agent edits the correct set of files (F1 over modified files) (Section 6; Appendix D, G, H).

- Cost modeling for amortization
  - Because low-latency test-time inference is typically more expensive, the paper models a linear cost where test-time tokens cost `t=10×` sleep-time tokens (Section 5.3; Figure 9).

- Why this design?
  - It exploits predictability: if many questions about a context share sub-computations, these can be pre-computed once, lowering latency and cost later (Introduction; Section 5.4; Figure 10).
  - It avoids relying on an oracle verifier (as in `pass@k`), so it’s applicable to tasks where verification is hard (Section 5.1; Figures 5–6).

## 4. Key Insights and Innovations
- Introduces sleep-time compute as an orthogonal axis to test-time scaling
  - What’s new: Pre-compute natural-language “representations” of a context between user turns, then answer later questions using less test-time compute (Section 3; Figure 1).
  - Why it matters: It cuts latency/cost while preserving or improving accuracy; it converts idle time into productive computation and amortizes across multiple queries.

- Demonstrates Pareto improvements and scaling behavior
  - Claim, grounded: 
    > “sleep-time compute can reduce the amount of test-time compute needed to achieve the same accuracy by ∼ 5× on Stateful GSM-Symbolic and Stateful AIME” (Abstract; Figures 3–4).
    - The method also “shifts the accuracy up by 13% on Stateful GSM-Symbolic and 18% on Stateful AIME” when scaling sleep-time compute (Abstract; Figures 7–8).
  - Significance: Not just faster/cheaper at equal accuracy, but also higher ceiling when more background compute is invested.

- Outperforms parallel sampling (`pass@k`) at equal token budgets
  - Evidence: Across GSM-Symbolic and AIME, sleep-time compute “generally pareto dominates pass@k” (Figures 5–6).
  - Significance: A more practical way to scale compute without assuming an oracle verifier.

- Quantifies when it helps most: predictable queries and shared context
  - Evidence:
    - Predictability: Gains are larger when the question is more predictable from the context (measured via the log-probability of the question given the context using Llama2-70B) (Section 5.4; Figure 10; examples in Appendix E).
    - Amortization: With more questions per context, average per-question cost drops; 
      > “decrease the average cost per query by 2.5×” with 10 questions per context (Abstract; Figure 9).

- Practical agent case study (SWE-Features)
  - Evidence: At low test-time budgets, sleep-time compute achieves higher F1 on predicting modified files; at high budgets, test-time-only can slightly outperform (Section 6; Figure 11).
  - Significance: Validates idea beyond synthetic math datasets and highlights trade-offs.

## 5. Experimental Analysis
- Evaluation methodology
  - Datasets
    - Stateful GSM-Symbolic: P1 (5000) and P2 (2500) problems, each split into `context` and `query` (Section 4.1; Figure 2).
    - Stateful AIME: 60 questions from AIME 2024/2025; context–query split with careful handling of figures and edge cases (Section 4.1; Appendix J).
    - Multi-Query GSM-Symbolic: multiple questions per context generated using `o3-mini` to study amortization (Section 4.1; Appendix C; Figure 20).
    - SWE-Features: 33 PR tasks across two repos; evaluate set overlap (F1) on modified files (Section 6; Appendix D, G).
  - Models
    - GSM-Symbolic: `gpt-4o-mini`, `gpt-4o` (Section 4.2).
    - AIME: `o1`, `o3-mini`, `Claude 3.7 Sonnet (Extended Thinking)`, `DeepSeek-R1` (Section 4.2).
  - Baselines
    - Standard test-time compute only (`T_B(q, c) → a`).
    - Parallel `pass@k` for comparison to parallel scaling (Figures 5–6).
    - Context-only ablation: model sees only `c` and must guess a likely question and answer; sleep-time compute substantially outperforms this, showing questions are not trivially predictable (Appendix I; Figures 21–22).
  - Setup details
    - For 4o/4o-mini, test-time budgets controlled via “verbosity” prompts (Appendix A), temperature 0 (Section 5.1).
    - For reasoning models, budgets controlled by API options or budget-forcing for R1 (Section 5.1; Figure 4).
    - Sleep-time compute scaling: multiple `c'` in parallel for non-reasoning models; higher “reasoning effort” at sleep-time for reasoning models (Section 5.2; Figures 7–8).
    - Cost model for amortization: test-time token = 10× sleep-time token (Section 5.3; Figure 9).

- Main quantitative findings
  - Pareto shift in GSM-Symbolic (Figure 3)
    - At low test-time budgets, sleep-time compute achieves accuracy comparable to higher-budget baselines while using far fewer tokens—roughly a 5× reduction at matched accuracy (Abstract; Figure 3 shaded regions).
    - At the highest budgets, test-time-only slightly outperforms in some settings, likely due to less distractor content (Section 5.1 commentary).
  - Pareto shift in AIME across reasoning models (Figure 4)
    - `o3-mini`, `Claude 3.7 Sonnet`, and `DeepSeek-R1` show clear improvements: same or better accuracy with substantially fewer test-time tokens. `o1` shows smaller gains (Figure 4; Section 5.1).
  - Parallel scaling comparison (Figures 5–6)
    - Sleep-time compute generally dominates `pass@k` at the same test-time budget, indicating a stronger avenue than naive parallel sampling without a perfect verifier (Figures 5–6).
  - Scaling sleep-time compute improves accuracy ceiling
    - GSM-Symbolic: adding more sleep-time generations (`k`) boosts accuracy up to a point; 5 parallel generations often beat 10, suggesting diminishing returns or prompt saturation (Figure 7). Reported maximum uplift: “up to 13%” (Abstract; Section 5.2).
    - AIME: increasing sleep-time reasoning effort pushes accuracy higher by “up to 18%” (Figure 8; Abstract).
  - Amortization across multiple queries (Figure 9)
    - Using the t=10 cost model, average cost per question drops as more questions share the same `c'`, reaching “up to 2.5×” savings at 10 questions per context (Section 5.3; Figure 9).
  - When does it help most? Predictability analysis (Figure 10)
    - Binning examples by estimated question predictability (Llama2-70B log-prob of question given context), the accuracy gap between sleep-time and test-time-only widens in higher-predictability bins (Section 5.4; Figure 10; Appendix E examples).
  - SWE-Features case study (Figure 11)
    - At low test-time budgets, sleep-time compute achieves better F1 on predicting modified files; at higher budgets, test-time-only overtakes, with a hypothesis that sleep-time encourages broader edits (lower precision) after exploring more files (Section 6; Figure 11).

- Are the experiments convincing?
  - Strengths
    - Diverse settings: synthetic math with controlled statefulness, amortization with multiple questions, and a real agent scenario (Sections 4–6).
    - Multiple model families (both reasoning and non-reasoning) and careful budget control (Section 5.1).
    - Robustness checks: context-only ablation (Appendix I) confirms that success is not due to trivial guessing; pass@k comparisons (Figures 5–6) show superiority over a common parallel baseline.
  - Caveats
    - The gains depend on the presence of reusable structure in the context and some predictability of forthcoming questions (Section 5.4; Figure 10).
    - At high budgets, additional prompt content can hurt, reversing the advantage (Figure 3; Figure 11 discussion).

## 6. Limitations and Trade-offs
- Dependence on query predictability and shared structure
  - Sleep-time compute works best when many queries over a context share sub-computations and when the next question is somewhat predictable (Section 5.4; Figure 10). When questions are unrelated, sleep-time work may be wasted.

- Potential prompt bloat and distraction
  - Concatenating multiple `c'` generations or highly verbose `c'` can crowd the test-time prompt and degrade performance at high budgets (Section 5.1; Figure 3; Section 6 commentary for SWE).

- Staleness and consistency
  - If the underlying context changes (e.g., codebase updates) after `c'` is computed, pre-computations may become stale; the paper does not address automatic invalidation or incremental updates.

- Compute and storage overheads
  - Sleep-time compute consumes resources in advance and requires storing `c'`; the paper models token costs (Section 5.3) but does not optimize scheduling or storage in production systems.

- Evaluation limitations
  - SWE-Features uses F1 over modified files rather than full functional correctness (Section 6; Appendix D), so it measures file targeting rather than end-to-end task success.
  - For DeepSeek-R1, budget control is indirect via “budget forcing” (Section 5.1), which might not perfectly match internal reasoning effort.

- Scope of methodology
  - The implemented `S(c)` is realized via natural-language “rethink memory” steps (Appendix F–K). Other forms (symbolic summaries, embeddings, tools) are not explored here.

## 7. Implications and Future Directions
- How this changes the landscape
  - Sleep-time compute adds a new planning axis: allocate compute when the model is idle to reduce future latency and cost. For stateful, multi-turn systems (document assistants, coding agents, customer support), this can be a default operating mode rather than an optimization.
  - It bridges systems ideas (caching, prefetching) with LLM prompting by learning useful natural-language representations of a context (Discussion: “representation learning over tokens”).

- Practical applications
  - Document QA: pre-extract entities, tables, cross-references, and likely aggregations; then answer many downstream questions fast.
  - Coding agents: pre-map architecture, dependencies, test surfaces, and common refactoring plans; then react quickly to fix or feature requests (Figure 11; Appendix H).
  - Conversational assistants: pre-summarize history, calendar, and tasks; anticipate likely follow-ups.

- Research opportunities
  - Adaptive scheduling and gating: automatically decide when to run `S(c)`, how much to invest, and which contexts merit more precomputation (Discussion).
  - Representation quality: explore structured or tool-augmented `c'` (e.g., tables, graphs, cached sub-results) and methods to control verbosity to avoid prompt bloat.
  - Consistency and freshness: incremental updates to `c'` when contexts change; correctness checks to reduce hallucinations in precomputed notes.
  - Learning to anticipate: train models to predict question distributions per context and optimize `S(c)` accordingly (Section 5.4).
  - Synthetic data generation: use sleep-time to generate high-quality data at lower cost than pure test-time compute pipelines (Discussion).
  - Verifier-light scaling: combine sleep-time compute with weak verifiers or self-checks that don’t require oracle access, improving over `pass@k` while keeping latency low.

In sum, this paper shows that reorganizing background reasoning into idle periods—sleep-time compute—can both lower the compute paid when a user asks a question and, with sufficient background effort, even raise the overall accuracy ceiling. The gains are largest when future questions are predictable and when multiple questions share a context, and they motivate a new class of systems that continuously “prepare” for the next user turn.
