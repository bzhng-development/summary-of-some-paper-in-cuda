# Inference Scaling for Long-Context Retrieval Augmented Generation

**ArXiv:** [2410.04343](https://arxiv.org/abs/2410.04343)

## 🎯 Pitch

This paper introduces DRAG and IterDRAG, two novel strategies for systematically scaling inference-time computation in retrieval-augmented generation (RAG) with long-context language models. By combining smarter use of demonstrations, iterative retrieval/generation, and a predictive allocation model, the authors show RAG can achieve nearly linear performance improvements—solving the plateau and unpredictability that cripple naive scaling. This breakthrough enables more effective, cost-efficient deployment of LLMs for knowledge-intensive tasks, ensuring that increases in computation consistently yield real-world gains.

---

## 1. Executive Summary
This paper shows how to turn the “long context” capability of modern language models into predictable, scalable gains for retrieval-augmented generation (RAG). It introduces two inference-time strategies—`DRAG` (Demonstration-based RAG) and `IterDRAG` (Iterative Demonstration-based RAG)—and a simple predictive model that tells you how to allocate a fixed inference budget across retrieved documents, in-context examples, and iterative steps to maximize quality.

## 2. Context and Motivation
- The problem/gap
  - Long-context language models can read millions of tokens, so a natural idea in RAG is to “stuff more knowledge” by retrieving more documents. In practice, this often plateaus or even hurts accuracy once you go past soft thresholds (e.g., top-10) because extra context introduces noise and distraction (Section 1; Related Work 2.3).
  - Long-context models also struggle to reliably locate relevant information in ultra-long sequences and frequently do not reach their best performance at the maximum context length (Section 1).
  - Two unanswered questions guide the paper (Section 1): 
    1) How much does RAG benefit when you scale inference compute if you allocate that compute well? 
    2) Can we predict the best way to spend a given inference budget?

- Why it matters
  - RAG underpins many knowledge-intensive applications (search assistants, enterprise Q&A, ops copilots). If gains from larger contexts are unpredictable or saturate, users waste compute. Predictable scaling and principled budget allocation translate directly into better latency/price-performance trade-offs.

- Where prior approaches fell short
  - Most prior “scaling” work increased only the quantity or length of retrieved documents (Section 1; Related Work 2.3). That helps recall but can hurt generation due to irrelevant content (Appendix A: Figure 7 shows Recall keeps improving with more docs while ranking metrics plateau early).
  - Chain-of-thought style prompting without interleaved retrieval often underperforms on knowledge-heavy, multi-hop questions (Appendix B, Table 6).

- How this paper positions itself
  - It treats inference compute as a budget that can be spent on three knobs—number of documents, number of demonstrations (“shots”), and number of iterative steps—and studies the entire configuration space rather than just “more documents.” It then models performance as a function of these knobs to predict optimal settings (Sections 3–5).

## 3. Technical Approach
This section explains the two strategies (`DRAG`, `IterDRAG`), how compute is measured, and how optimal allocations are modeled.

- Key definitions
  - `RAG` (Retrieval-Augmented Generation): before answering a query, retrieve relevant documents from a corpus and insert them into the model’s prompt.
  - `In-context learning (ICL)`: at test time, give the model a few task examples (“shots”) inside the prompt so it can imitate the pattern.
  - `Effective context length`: the total number of input tokens the model consumes across an entire answer, including all iterations if the method runs multiple rounds (Section 3.1). Output tokens and retrieval costs are excluded because answers in these datasets are short and ANN retrieval is comparatively cheap.
  - Budget `L_max`: an upper bound on effective context length you are allowed to spend for an answer (Section 4.1).
  - Inference parameters `θ`: three integers—`k` (number of retrieved documents per example), `m` (number of in-context examples), `n` (number of generation iterations). In `DRAG`, `n=1`. In `IterDRAG`, `n` can be >1 (Section 4.1).

- DRAG: demonstration-based RAG (Section 3.2; Figure 3 left)
  - Pipeline
    1) For each in-context example and for the test query, retrieve the top-`k` documents from a large corpus (Wikipedia from KILT; Appendix H).
    2) Build a long prompt that interleaves these document sets with example “Question → Answer” pairs, then the test documents and test question.
    3) Reverse the order of retrieved docs in each set so the highest-ranked documents sit closest to the question (Section 3.2).
    4) Do a single generation call to produce the answer.
  - Why this helps: instead of only enlarging the context, the examples “teach” the model how to use retrieved evidence inside long contexts—how to pick relevant snippets and apply them to a new question (Section 3.2).

- IterDRAG: iterative demonstration-based RAG (Section 3.3; Figure 3 right)
  - Motivation: on multi-hop questions, one-shot retrieval often misses intermediate facts. IterDRAG decomposes the question and interleaves retrieval and reasoning.
  - Pipeline
    1) Create demonstrations in a constrained “Self-Ask” style so each example shows a sequence like “Follow up: … → Intermediate answer: … → So the final answer is: …” (Section 3.3; Appendix H).
    2) At test time, start with initial retrieval and the demonstrations in the prompt. The model either emits a sub-question (“Follow up: …”) or an intermediate/final answer.
    3) When a sub-question appears, retrieve additional documents for it and append them; then the model produces the intermediate answer. Repeat up to 5 iterations (Section 3.3).
    4) Stop when the model outputs “So the final answer is: …”
  - Why this helps: targeted retrieval for simpler sub-queries raises the chance the right evidence enters the context and reduces distraction from unrelated documents (Appendix A, Table 5 shows large gains in ranking metrics over one-shot retrieval).

- Measuring and searching for the best use of compute
  - For a fixed budget `L_max`, the method searches over many combinations of `(k, m, n)` whose token count is ≤ `L_max` and selects the configuration with the best average metric (Equation 1 in Section 4.1).
  - Experimental budgets: `L_max` ∈ {16k, 32k, 128k, 1M, 5M} tokens (Section 4.1). Grid: `k` ∈ {0, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000}; `m` ∈ {0, 1, 2, 4, …, 256}; `n` up to 5 (Section 4.1).

- Modeling performance to predict optimal allocations (Section 5)
  - The paper introduces a simple “computation allocation model.” Informally, it says: if you take the log of the three knobs `(k, m, n)`, performance behaves almost linearly after a sigmoid-like transformation.
  - Equation 2 (Section 5.1): apply an inverse-sigmoid to the metric `P`, then approximate it by a linear function of `log(k)`, `log(m)`, and `log(n)`. Coefficients depend on the base model and on a task-specific “informativeness” vector `i = (i_doc, i_shot, 0)`. 
    - `i_doc` is measured as the performance gain from adding one document vs zero-shot QA on that task; `i_shot` is the gain from adding one example vs zero-shot (Section 5.1).
  - Estimation: ordinary least squares on observed runs to learn parameters `a, b, c`. Once fitted on some tasks or shorter budgets, it predicts the best `(k, m, n)` for new tasks or longer budgets (Sections 5.2, Table 3 and Table 4).

- System and implementation details (Appendix H)
  - Retriever: Gecko-1B embeddings over Wikipedia passages; documents truncated to 1024 tokens; top-`k` per step; documents listed nearest to the question in descending rank (Appendix H).
  - LLM: Gemini 1.5 Flash (1M token window) for efficiency (Section 4.1; Appendix H).
  - Constrained decoding for `IterDRAG` forces the Self-Ask output prefixes (“Follow up: …”, “Intermediate answer: …”, “So the final answer is: …”) to control the iteration (Appendix H).

## 4. Key Insights and Innovations
- Inference scaling laws for RAG (Sections 4.3; Figures 1 and 4)
  - Novelty: “Performance improves nearly linearly as you increase effective context length—if you allocate the budget well across documents, demonstrations, and iterations.”
  - Evidence: Red dots in Figures 1 and 4 mark the best configuration found at each budget, and the dashed lines fitting those points are close to linear growth on a log-scale x-axis. Gains are strongest up to ~1M tokens and then taper (Section 4.3).

- Two complementary scaling strategies
  - `DRAG` (Section 3.2): scales well at smaller budgets (16k–32k). It adds demonstrations that show how to use retrieved evidence; simpler to run (one model call).
  - `IterDRAG` (Section 3.3): scales better at larger budgets (≥128k) by interleaving retrieval and generation, building a reasoning chain that reduces the “compositionality gap.” Figure 2 and Table 1 highlight that `IterDRAG` overtakes `DRAG` beyond 128k tokens.

- A compute allocation model that predicts optimal settings (Section 5; Figures 6 and 12; Tables 2–4)
  - Significance: instead of brute force search every time, you can estimate how many documents, shots, and iterations to use given a budget and a new domain.
  - Results:
    - Fit quality for DRAG: R² = 0.903, MSE = 0.085 (Table 2, “Sigmoidal σ” column).
    - Domain generalization at 1M tokens achieves 96.6% of the oracle performance on average (Table 3).
    - Length extrapolation is accurate up to 1M tokens and degrades modestly at 5M (Table 4).

- A clearer picture of retrieval limits and the benefit of iterativity (Appendix A)
  - Finding: Recall steadily improves with more documents, but ranking quality (NDCG, MRR) plateaus near 100 documents (Figure 7). 
  - Iterative retrieval with sub-queries boosts Recall, NDCG, and MRR substantially; for 2WikiMultiHopQA, Recall rises from 0.722 to 0.935 and MRR from 0.336 to 0.528 (Table 5).

## 5. Experimental Analysis
- Setup (Section 4.1; Appendix H)
  - Tasks: multi-hop QA—Bamboogle, HotpotQA, MuSiQue, 2WikiMultiHopQA. Extra analyses include TriviaQA, Natural Questions, and StrategyQA (Appendix C).
  - Metrics: Exact Match (EM), token-level F1, and an “accuracy” that checks whether the ground-truth answer string appears in the prediction (Section 4.1).
  - Baselines: 
    - Zero-shot QA (no retrieval, no shots).
    - Many-shot QA (shots only).
    - Standard RAG (documents only).
  - Budget grid: `L_max` in {16k, 32k, 128k, 1M, 5M}, with broad sweeps over `k`, `m`, `n` (Section 4.1).
  - Model: Gemini 1.5 Flash with up to 1M-token window; iterative calls extend effective length beyond the window.

- Main quantitative results
  - DRAG and IterDRAG scale while baselines plateau (Table 1; Figures 1–2, 4).
    - At 128k tokens, `IterDRAG` clearly outperforms standard RAG on multi-hop tasks:
      > Table 1 (128k): On 2WikiMultiHopQA, `IterDRAG` reaches Acc 74.6 vs RAG 48.4 (+26.2 absolute).  
      > On Bamboogle, `IterDRAG` Acc 68.8 vs RAG 52.8 (+16.0).  
      > On MuSiQue, `IterDRAG` Acc 24.5 vs RAG 16.8 (+7.7).
    - At 1M tokens, DRAG is at its window limit while `IterDRAG` continues to scale by iterating:
      > Table 1 (1M): On 2WikiMultiHopQA, `IterDRAG` Acc 76.4; DRAG 53.3.  
      > On MuSiQue, `IterDRAG` Acc 30.5; DRAG 18.2.
    - At 5M effective tokens (achieved via iteration), `IterDRAG` still improves slightly:
      > Table 1 (5M): On HotpotQA, `IterDRAG` EM 51.7 (up from 48.7 at 1M); Acc 56.4.

  - Average accuracy comparison across methods (Figure 2):
    > DRAG and especially `IterDRAG` dominate zero-shot, many-shot, and standard RAG once the budget is allowed to scale (up to 5M effective tokens).

  - Linear-ish optimal scaling (Figures 1 and 4):
    > The red “optimal config” points aligned by dashed fits grow almost linearly with log effective length up to ~1M, then gains slow.

  - Parameter-specific scaling behavior (Section 4.4; Figure 5; Appendix C Figure 8)
    - Increasing `k` (documents) usually gives larger marginal gains than increasing `m` (shots) in DRAG (Figure 5b vs 5c).
    - In IterDRAG, adding even one shot helps more visibly by teaching decomposition (Section 4.4).
    - Both have soft thresholds: beyond ~100–500 documents, marginal gains fade or reverse due to noise (Figure 5b and retrieval analysis in Appendix A).

  - Retrieval quality and the case for iterativity (Appendix A)
    > Table 5: IterDRAG improves Recall by 21.7% on average and improves NDCG and MRR by ~30–40% over DRAG at k=50, m=2.

  - Chain-of-thought vs IterDRAG (Appendix B, Table 6)
    > IterDRAG substantially outperforms a CoT baseline: e.g., on 2WikiMultiHopQA, Acc 72.3 vs 36.7.

  - One-hop datasets and StrategyQA (Appendix C)
    > TriviaQA best Acc 69.0 at ~50 documents; Natural Questions peaks at ~20 documents (Figure 9).  
    > StrategyQA accuracy rises from 61.1 (zero-shot) to 79.0 (DRAG) and 83.4 (IterDRAG) (Table 7).

  - Predictive model validation (Section 5.2)
    - Good fit and ablations:
      > Table 2: Full model (“Sigmoidal σ”) achieves R² 0.903, MSE 0.085; removing the task-adaptive term `b ⊙ i` hurts fit.
    - Domain generalization:
      > Table 3: Predicted configurations at 1M tokens achieve near-oracle results across four datasets (e.g., Bamboogle Acc 68.0 predicted vs 68.8 oracle).
    - Length extrapolation:
      > Table 4: From 128k→1M, predictions differ from oracle by only ~2.8% on average; predicting to 5M is harder (avg gap ~5.6%).

- Do the experiments support the claims?
  - Yes, across four multi-hop datasets and multiple budgets, the “near-linear optimal scaling” pattern repeats (Figures 1 and 4; Figure 11 per-dataset). The predictive model is validated with out-of-domain and out-of-length tests (Tables 3–4). Retrieval analyses explain why naively adding documents plateaus and why interleaving retrieval helps (Appendix A).
  - Failure analyses (Appendix G; Figure 14) are candid about residual errors—retrieval misses, flawed reasoning, hallucinations, and evaluation artifacts—clarifying where gains stop.

## 6. Limitations and Trade-offs
- Assumptions and scope
  - The compute budget is defined solely as input tokens; output tokens and retrieval cost are ignored (Section 3.1). This is reasonable for short answers and cheap ANN retrieval but may not hold for long-form generation or complex retrieval stacks.
  - The approach is evaluated with one long-context model (Gemini 1.5 Flash). Scaling behavior could vary with other LLMs or architectures.

- Where it may not work as well
  - Ultra-long single-pass contexts: DRAG shows diminishing returns beyond ~10⁵ tokens per call (Figures 1 and 11); the model may not reliably extract the right evidence from extremely long inputs.
  - Very noisy corpora or stale knowledge: retrieval errors remain a leading failure source; when correct evidence is absent or ranked poorly, iterative reasoning still fails (Appendix G).
  - Extremely large budgets: beyond ~1M effective tokens, gains diminish (Section 4.3).

- Computational trade-offs
  - IterDRAG trades extra latency (multiple LLM calls) for better evidence quality and reasoning. While effective context length can grow to millions of tokens via iteration, each step adds overhead, which may affect real-time applications.
  - Finding optimal configurations by grid search can be expensive; the allocation model mitigates this but requires some initial runs for fitting (Section 5.1).

- Modeling simplifications
  - The informativeness vector sets the iteration component to zero (`i_iter = 0`) because adding it did not help in experiments (Section 5.1). This may limit expressiveness in settings where the number of iterations is especially impactful.
  - The inverse-sigmoid linearization is empirical; although it fits well (Table 2), it is not derived from first principles and mildly underperforms at the longest budgets (Table 4, last column).

## 7. Implications and Future Directions
- How this work changes the field
  - It reframes “long-context RAG” as a compute allocation problem with measurable, near-linear gains under optimal configurations (Figures 1 and 4). This demystifies test-time scaling and provides a practical recipe for spending inference budgets.
  - It shows that interleaved retrieval and reasoning can reliably beat “stuff more documents” strategies on multi-hop tasks and remain scalable past 1M effective tokens (Table 1, 1M–5M rows).

- Follow-up research enabled or suggested
  - Adaptive per-query allocation: learn a controller that, given a query and early signals (retrieval scores, partial generations), decides `(k, m, n)` on the fly to meet a latency/quality target using the allocation model as a prior.
  - Better retrieval pipelines: since ranking quality saturates early (Appendix A), add re-ranking, filtering, or clustering to reduce distractors before they hit the prompt; study how such modules shift the scaling curves.
  - Beyond QA: apply DRAG/IterDRAG and the allocation model to tasks like code generation with API docs, enterprise support search, and scientific literature Q&A; evaluate for longer outputs and richer reasoning.
  - Long-context modeling: techniques that help the model reliably find and compare evidence inside very long contexts could push the “knee” past 1M tokens (Section 6 Discussion: Long-Context Modeling).

- Practical applications
  - Enterprise knowledge assistants: predictable quality gains when allocating budgets for complex tickets or investigations; `IterDRAG` helps decompose multi-hop issues.
  - Legal/medical research tools: iterative retrieval to gather and assemble distributed evidence; budget-aware settings prevent over-spending on easy queries.
  - Data pipelines for RAG systems: use the computation allocation model to precompute recommended `(k, m, n)` per domain and per budget; fall back to DRAG for smaller budgets and switch to IterDRAG for larger ones.

> Representative headline result: “When test-time compute is optimally allocated, RAG performance scales almost linearly with effective context length up to about 1M tokens, with `IterDRAG` delivering the strongest gains at larger budgets and achieving up to roughly 59% higher accuracy than standard RAG on multi-hop benchmarks” (Sections 4.2–4.3; Table 1; Figure 2).
