# Inference Scaling for Long‑Context Retrieval Augmented Generation

**ArXiv:** [2410.04343](https://arxiv.org/abs/2410.04343)
**Authors:** Zhenrui Yue, Honglei Zhuang, Aijun Bai, Kai Hui, Rolf Jagerman, Hansi Zeng, Zhen Qin, Dong Wang, Xuanhui Wang, Michael Bendersky
**Institutions:** University of Illinois Urbana‑Champaign, Google DeepMind, University of Massachusetts Amherst

## 🎯 Pitch

This paper introduces the `DRAG` and `IterDRAG` strategies to optimize test-time compute for retrieval-augmented generation (RAG), using a computation-allocation model to predict the best use of resources for maximum accuracy. It transforms scalable inference into a reliable tool, empowering applications to trade compute resources for enhanced accuracy, ultimately advancing long-context language model capabilities in knowledge-intensive tasks.

---

## 1. Executive Summary
This paper shows how to turn the extra test‑time compute offered by long‑context LLMs into predictable, near‑linear gains for retrieval‑augmented generation (RAG). It introduces two inference strategies—`DRAG` (demonstration‑based RAG) and `IterDRAG` (iterative demonstration‑based RAG)—and a simple computation‑allocation model that predicts how to spend a fixed inference budget (documents, examples, iterations) to maximize accuracy.

## 2. Context and Motivation
- Problem the paper tackles
  - Long‑context LLMs can ingest huge inputs (e.g., up to millions of tokens), but simply stuffing more retrieved text into prompts often plateaus or hurts RAG quality due to noise and distraction. Figure 1 (left) and related discussion in Section 1 highlight that standard RAG plateaus around 10^4 tokens; retrieving beyond soft thresholds (e.g., top‑10) can degrade answers.
  - Two practical questions remain unanswered:
    1) If we scale inference compute wisely, how much can RAG actually improve?
    2) Given a test‑time compute budget, can we predict the best way to spend it?

- Why this matters
  - Knowledge‑intensive applications (search assistants, enterprise question answering, analytics) depend on RAG quality. If test‑time compute can buy reliable gains, teams can trade money/latency for accuracy with confidence.
  - Theoretically, a scaling law at inference time (not just model size) clarifies the role of context and retrieval for LLMs.

- Shortcomings of prior approaches
  - Prior “inference scaling for RAG” mainly means “retrieve more/longer documents” (Related Work, Section 2.3), which:
    - Increases recall but also injects noise; performance often plateaus or drops (Section 1; Figure 5b; Appendix A Figure 7).
    - Doesn’t teach the model how to use the extra information.

- How this paper positions itself
  - It expands the scaling dimensions beyond “more documents” to include “more demonstrations” and “more generation steps,” and introduces:
    - `DRAG`: many‑shot in‑context RAG with demonstrations that themselves include retrieved documents (Section 3.2).
    - `IterDRAG`: iterative query decomposition with interleaved retrieval and answering (Self‑Ask style), letting the model fetch targeted evidence for each sub‑question (Section 3.3).
    - A computation‑allocation model that predicts the optimal mix of retrieval (`k` docs), examples (`m` shots), and iterations (`n`) for a given token budget, and explains the observed “inference scaling laws” (Sections 4–5).

## 3. Technical Approach
Key terms (defined once as used here):
- `RAG`: Retrieval‑Augmented Generation—retrieve external text passages and condition the LLM’s answer on them.
- `In‑context learning (ICL)`: teach the model the task by showing input–output examples in the prompt.
- `Effective context length`: the total number of input tokens the LLM processes across all inference steps before the final answer (Section 3.1). Output tokens and retrieval compute are excluded.
- `Self‑Ask`: a prompting format where the model alternates between “Follow up:” sub‑questions and “Intermediate answer:” steps before “So the final answer is:” (Section 3.3).

Step‑by‑step:

1) Measuring and budgeting test‑time compute
- The paper treats inference compute as a budget `L_max` of input tokens (“effective context length”). For multi‑step methods, it sums tokens across steps (Section 3.1).
- It then asks: given `L_max`, what configuration `θ` = (`k` documents per step, `m` in‑context examples, `n` iterations) maximizes average task performance? This is formalized in Equation (1):
  > Maximize the average metric P over dataset X by searching `θ` subject to `l(x_i; θ) ≤ L_max` for all examples.

2) `DRAG`: Demonstration‑based RAG (Section 3.2)
- What it is:
  - A one‑call method (`n=1`) that combines: many retrieved documents (`k`) + many demonstrations (`m`). Each demonstration contains its own retrieved documents, a question, and its answer.
- How it works:
  - For each demonstration and the test query, a retriever selects top‑`k` documents from a large corpus (Wikipedia, via Gecko‑1B embeddings; Implementation, Section H).
  - Documents are reversed so higher‑ranked items appear closer to the question (a known prompt ordering trick; Section 3.2).
  - The prompt includes many such “context–question–answer” demonstrations, followed by the test query and its retrieved context (Figure 15 and the prompt in Figure 16).
  - The LLM (Gemini 1.5 Flash) answers in a single pass. Increasing `k` and/or `m` scales compute up to the model’s context window (1M for Flash).

3) `IterDRAG`: Iterative demonstration‑based RAG (Section 3.3)
- What it is:
  - A multi‑call method (`n > 1`) that learns to decompose the user query into sub‑queries. For each sub‑query, it retrieves more documents and produces an intermediate answer, then combines everything to produce the final answer (Figure 3 right; prompt in Figure 17).
- How it works:
  - Training the in‑context format: they synthesize demonstrations that include sub‑queries and intermediate answers using constrained decoding to the Self‑Ask format (Section 3.3).
  - Inference loop (details in Section 3.3):
    1. Start with initial retrieved documents for the main question plus the demonstration set.
    2. The model emits either a “Follow up:” sub‑query, an “Intermediate answer:”, or a final answer.
    3. If a sub‑query appears, the system retrieves additional documents and appends them to the running context.
    4. Repeat up to 5 iterations, after which the model must produce the final answer.
  - Compute scales with the number of iterations plus the extra retrieved context per step, so IterDRAG can exceed a single context window via multiple calls.

4) Finding optimal performance for a budget (Section 4.1)
- For each `L_max` in {16k, 32k, 128k, 1M, 5M}, the paper grid‑searches `k`, `m`, and (for IterDRAG) `n` to find the best achievable accuracy on each dataset—this gives the “optimal performance” `P*(L_max)` used to characterize scaling (Table 1; Figure 4).

5) Modeling how to allocate compute (Section 5)
- Goal: Predict performance as a function of `θ` and identify the optimal `θ` under a budget without exhaustive search.
- Model (Equation (2)):
  - Transform the metric `P` by an inverse sigmoid `σ^-1` to account for mild saturation at very long contexts (>1M tokens; Section 4.3), then fit a linear model in log‑space:
    > `σ^-1(P(θ)) ≈ (a + b ⊙ i)^T log(θ) + c`
  - `θ = (k, m, n)`. `i = (i_doc, i_shot, 0)` measures task‑specific informativeness of documents vs. examples, computed from simple base configurations (Section 5.1).
  - `a, b, c` are learned per LLM (estimated by ordinary least squares, parameters reported in Appendix F Table 8). This separates model‑level behavior (`a, b, c`) from task‑level informativeness (`i`).
- Use: once fitted, the model predicts the best mix of `k, m, n` for a given `L_max` and task (Sections 5.2, Tables 3–4).

6) System components and data (Implementation, Section H)
- Retriever: Gecko‑1B embeddings on Wikipedia (KILT) with right‑truncation to 1024 tokens per document.
- LLM: Gemini 1.5 Flash (1M token window). For >1M effective tokens, IterDRAG uses multiple calls.
- Prompt construction: demonstrations then test documents then test query; reverse ordering of retrieved documents; constrained decoding for Self‑Ask in IterDRAG.

## 4. Key Insights and Innovations
1) A practical “inference scaling law” for RAG
- What’s new: When compute is optimally allocated across documents, demonstrations, and iterations, RAG accuracy improves almost linearly with the order of magnitude of effective context length (Sections 4.2–4.3).
- Evidence:
  - Figure 1 (right) and Figure 4: red dots (optimal configs) lie close to a straight line in log‑scale for DRAG and IterDRAG; standard RAG plateaus early (~10^4 tokens).
  - Table 1: as the budget grows from 16k → 1M → 5M tokens, DRAG and especially IterDRAG keep improving, while baselines saturate.

2) Two complementary scaling strategies—`DRAG` and `IterDRAG`
- What’s new: The paper shows that many‑shot ICL (DRAG) and iterative retrieval with query decomposition (IterDRAG) are both effective, but at different scales (Section 4.2).
  - DRAG dominates at smaller budgets (≤32k).
  - IterDRAG takes over at larger budgets (≥128k) by adding multi‑step retrieval and reasoning.
- Significance: It turns extra tokens into guided computation rather than just more noise (Figures 5, 8).

3) A simple, predictive computation‑allocation model
- What’s new: A log‑linear model (Equation (2)) with a task‑informativeness vector `i` that predicts RAG performance and optimal hyperparameters under a budget (Section 5).
- Why it matters:
  - High fit quality: `R^2 = 0.903`, `MSE = 0.085` for the full model with sigmoidal scaling (Table 2).
  - Generalizes across domains and lengths:
    - Domain generalization reaches 96.6% of oracle performance at 1M tokens (Table 3).
    - Length extrapolation is accurate up to 1M tokens (average 2.8% gap from 128k → 1M; Table 4).

4) Iterative retrieval improves evidence quality, not just quantity
- What’s new: IterDRAG’s interleaved retrieval boosts ranking quality (not only recall), addressing a major pain point of long‑context RAG (Appendix A Table 5).
- Evidence:
  > On 2WikiMultiHopQA with 50 docs and 2 shots, NDCG improves from 0.421 (DRAG) to 0.605 (IterDRAG), and MRR from 0.336 to 0.528.

## 5. Experimental Analysis
- Setup (Section 4.1 + Implementation H)
  - LLM: Gemini 1.5 Flash (1M token context).
  - Retriever: Gecko‑1B over Wikipedia (KILT).
  - Budgets `L_max`: 16k, 32k, 128k, 1M, 5M tokens.
  - Search space: `k ∈ {0,1,2,5,10,20,50,100,200,500,1000}`, `m ∈ {0, 1, 2, 4, 8, 16, 32, 64, 128, 256}`, `n ≤ 5`.
  - Datasets: Multi‑hop QA—Bamboogle, HotpotQA, MuSiQue, 2WikiMultiHopQA; plus one‑hop TriviaQA and Natural Questions, and binary StrategyQA (Sections 4.1 and Appendix C).
  - Metrics: `EM` (exact match), `F1`, and `Acc` (whether the ground‑truth string appears in the model’s prediction; Section 4.1).

- Baselines (Section 4.1):
  - `Zero-shot QA` (no retrieval, no demos),
  - `Many-shot QA` (demos only),
  - `RAG` (retrieval only).

- Main quantitative results
  - Scaling behavior
    - Figure 4: For both DRAG and IterDRAG, the optimal points (red dots) align with a near‑linear trend as effective context grows; standard RAG curves flatten early.
  - End‑to‑end accuracy (Table 1)
    - At 128k tokens (typical long context):
      - On 2WikiMultiHopQA, `Acc`: RAG 48.4 → DRAG 53.1 → IterDRAG 74.6.
      - On Bamboogle, `Acc`: RAG 52.8 → DRAG 54.4 → IterDRAG 68.8.
      - On MuSiQue, `Acc`: RAG 16.8 → DRAG 17.9 → IterDRAG 24.5.
    - At 1M tokens:
      - On 2WikiMultiHopQA, `Acc`: DRAG 53.3 → IterDRAG 76.4.
      - On MuSiQue, `Acc`: DRAG 18.2 → IterDRAG 30.5.
    - At 5M tokens (via multiple IterDRAG steps):
      - On 2WikiMultiHopQA, `Acc` rises to 76.9; on HotpotQA, `Acc` to 56.4.
  - Takeaway:
    - DRAG is strongest at 16k–32k; IterDRAG overtakes beyond 128k (Section 4.2).
    - The paper reports “up to 58.9% gains over standard RAG” when optimally scaling compute (Abstract and Figure 2 summary).

- Parameter‑specific insights (Section 4.4; Figure 5)
  - Documents vs. shots:
    - For DRAG, increasing documents `k` typically yields larger gains than increasing shots `m` (Figure 5b vs. 5c).
    - For IterDRAG, adding just one shot (`m: 0 → 1`) often helps more than adding one document, because demonstrations teach query decomposition and evidence use.
  - Saturation and soft thresholds:
    - Gains diminish or reverse beyond certain `k`/`m` levels due to noise (Figure 5a–c).

- Retrieval quality analysis (Appendix A)
  - More documents improve recall but not ranking quality; NDCG/MRR plateau around ~100 documents (Appendix A Figure 7).
  - IterDRAG’s interleaved retrieval improves all retrieval metrics relative to one‑shot DRAG (Appendix A Table 5), e.g. for 2WikiMultiHopQA recall 0.722 → 0.935; NDCG 0.421 → 0.605.

- Comparison to chain‑of‑thought (Appendix B, Table 6)
  - IterDRAG outperforms a Self‑Ask‑style CoT without interleaved retrieval:
    > On 2WikiMultiHopQA (k=5, m=4), `Acc`: CoT 36.7 vs. IterDRAG 72.3.

- Predictive model validation (Section 5.2)
  - Fit quality and ablations (Table 2):
    > Full model with sigmoidal scaling: `R^2 = 0.903`, `MSE = 0.085`; removing task‑informativeness term (`b ⊙ i`) reduces `R^2` to 0.866.
  - Domain generalization (Table 3):
    > At 1M tokens, predicted configs achieve 96.6% of oracle performance; e.g., on 2WikiMultiHopQA `Acc` 76.4 (oracle) vs. 74.9 (predicted).
  - Length extrapolation (Table 4):
    > From 128k → 1M tokens, predicted `Acc` is within 2.8% of oracle on average; 1M → 5M is harder (5.6% gap).

- Failure analysis and robustness (Appendix G)
  - Four error types:
    1) Inaccurate/outdated retrieval,
    2) Incorrect or missing reasoning,
    3) Hallucination/unfaithful reasoning,
    4) Evaluation issues/refusals.
  - IterDRAG reduces (1) and (2) by targeting sub‑queries with fresh retrieval (Appendix G narrative).

- Do experiments support the claims?
  - Yes, across multiple datasets and budgets:
    - Near‑linear optimal scaling appears consistently (Figures 1, 4, 11).
    - DRAG and IterDRAG outperform baselines at their respective scales (Table 1, Figure 2).
    - The compute‑allocation model predicts well and generalizes (Tables 2–4).
  - Caveat: improvements beyond ~1M tokens are smaller (Section 4.3), indicating remaining long‑context limits.

## 6. Limitations and Trade-offs
- Dependence on retrieval quality
  - Recall improves with more documents, but NDCG/MRR saturate and noise grows (Appendix A Figure 7). Without re‑ranking/filtering, adding documents can distract the model (Section 4.4).
- Long‑context modeling limits
  - Gains become sub‑linear or plateau beyond ~1M tokens (Section 4.3). DRAG’s per‑step context seems to peak around 10^5 tokens, while IterDRAG benefits by spreading compute across steps (Discussion: Long‑Context Modeling).
- Budget definition and real‑world costs
  - Compute budget excludes output tokens and retrieval cost (Section 3.1). In production, retrieval latency/compute and output‑length constraints may matter.
- Demonstration quality and format
  - IterDRAG requires demonstrations formatted with Self‑Ask; the paper uses constrained decoding to create them (Section 3.3). Quality of these demos affects performance.
- Model and data scope
  - Most experiments use Gemini 1.5 Flash and Gecko‑1B retriever; while Appendix D shows similar scaling with GTR‑XXL, broader cross‑model validation would strengthen generality.
- Evaluation scale and metrics
  - Datasets are sub‑sampled (1.2k per dataset; Section 4.1). “Accuracy” metric is liberal (checks if ground truth string appears in output; Section 4.1), which favors verbose outputs; exact match still remains modest on harder datasets (Table 1).

## 7. Implications and Future Directions
- What changes for the field
  - Test‑time compute becomes a reliable knob for RAG: with the right allocation policy, you can “buy” accuracy with tokens up to ~1M effective tokens (Figures 1, 4).
  - Iterative retrieval and many‑shot demonstrations are not just training‑time ideas—they are scalable inference strategies that unlock long‑context LLMs.

- Practical guidance (from results throughout Sections 4–5)
  - Small budgets (≤32k): prefer `DRAG` with moderate `k` and a handful of `m`.
  - Medium budgets (~128k): start transitioning to `IterDRAG` to benefit from interleaved retrieval.
  - Large budgets (≥1M): rely on `IterDRAG` with multiple steps; expect diminishing returns above ~1M.
  - Documents vs. shots: for single‑step DRAG, prioritize raising `k`; for IterDRAG, even 1–2 demonstrations can markedly help query decomposition (Figure 5).
  - Use the computation‑allocation model (Equation (2)) to predict `k, m, n` under a specified token budget and task (Tables 3–4).

- Follow‑on research
  - Retrieval refinement at scale: pair IterDRAG with learned re‑rankers or selective reading to curb noise (Appendix A and Discussion).
  - Stronger long‑context representations: improve models’ ability to locate relevant spans in 10^5–10^6‑token contexts (Discussion: Long‑Context Modeling).
  - Dynamic, per‑query allocation: learn policies that adapt `k, m, n` online using uncertainty or early‑exit signals, building on the allocation model.
  - Better evaluation: beyond EM/F1, add faithfulness and step‑level reasoning checks to diagnose hallucination and unfaithful chains (Appendix G).

- Downstream applications
  - Enterprise search and analytics, scientific assistants, educational tutors: anywhere RAG is used and latency/compute budgets vary, the approach offers a principled way to spend extra tokens for predictable gains.

> In short, the paper provides both an engineering recipe (DRAG/IterDRAG) and a planning tool (the computation‑allocation model) to convert long‑context budgets into steady, near‑linear improvements for RAG up to about one million effective tokens (Figures 1 and 4; Table 1), with validated predictions across domains and lengths (Tables 3–4).
