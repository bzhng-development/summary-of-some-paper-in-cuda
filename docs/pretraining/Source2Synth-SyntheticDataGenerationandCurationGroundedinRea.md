# Source2Synth: Synthetic Data Generation and Curation Grounded in Real Data Sources

**ArXiv:** [2409.08239](https://arxiv.org/abs/2409.08239)

## 🎯 Pitch

Source2Synth introduces a scalable pipeline for generating high-quality synthetic training data grounded in real-world sources (like documents or tables), combined with an automated curation step that filters out low-quality examples based on model-predicted answerability. By anchoring synthetic data in actual source material and rigorously curating it, Source2Synth dramatically boosts large language model performance in complex tasks such as multi-hop reasoning and tool-based question answering, demonstrating substantial gains over both instruction-tuned and human-annotated baselines. This approach paves the way for powerful, low-cost domain adaptation in settings where annotated data is scarce but raw databases or documents are plentiful.

---

## 1. Executive Summary
Source2Synth is a three-stage pipeline that automatically generates and curates synthetic training data grounded in real-world sources (documents or tables), then fine-tunes a large language model (LLM) on the curated set. The key result is that small, carefully curated synthetic datasets—checked for answerability by an intermediate model—substantially improve complex capabilities: multi-hop reasoning on HotpotQA and SQL-based table question answering on WikiSQL (e.g., 66.05% soft-EM on HotpotQA vs. 58.40% for a fine-tuned baseline; 34.50% EM on WikiSQL vs. 12.30% for the best prompting baseline; Tables 1–2).

## 2. Context and Motivation
- Problem addressed:
  - High-quality, task-specific labeled data is scarce, expensive, and time-consuming to obtain. This is especially true for complex tasks like multi-step reasoning and tool use (e.g., SQL for table QA) where generic instruction-tuning often produces few usable examples and many low-quality or contrived ones (Introduction; Related Work).
- Why it matters:
  - Unlocking multi-step reasoning and reliable tool use is central to deploying LLMs in real-world domains (e.g., legal, medical) where raw text and tables are abundant but annotated QA data is not (Introduction; Section 4).
- Shortcomings of prior approaches:
  - Human annotation: costly, slow, and can introduce inconsistencies or biases (Introduction).
  - General synthetic instruction-tuning: tends to yield few multi-hop samples and variable quality (Related Work; cites [Chen et al., 2024]).
  - Prior tool-use methods often operate on simple inputs and fail to robustly compose SQL grounded in a database schema (Related Work).
- Positioning:
  - Source2Synth combines real-world grounding (e.g., Wikipedia documents, real tables) with self-curation. It filters generated samples by answerability using a model fine-tuned on a subset of the data, and imputes missing/awkward parts to improve naturalness. It shows consistent improvements on two distinct, challenging tasks: multi-hop QA (MHQA) and tabular QA (TQA) (Sections 3–5).

## 3. Technical Approach
Source2Synth has three stages (Figure 1).

Stage 1 — Dataset Generation (Section 3.1)
- Data source selection:
  - MHQA: English Wikipedia articles (Section 3.1.1).
  - TQA: 4,000 unlabeled tables from WikiSQL training data (Section 3.1.1).
- Seed selection:
  - MHQA seed `E` is an entity sampled from a first article `D1`. A second article `D2` is chosen from related pages that also mention `E`. This `E` is the “hop” connecting sub-questions (Section 3.1.2; Figure 2).
  - TQA seed is an “interesting fact” about the table, induced by an LLM prompt (Figure 11), e.g., “The country with most arrivals in 2012” (Section 3.1.2; Figure 3).
- Constructing synthetic examples (Section 3.1.3):
  - MHQA (Figure 2):
    1) Generate `Q1` from `D1` so its answer is the entity `E`.
    2) Generate `Q2` from `D2` about `E`.
    3) Merge `Q1` and `Q2` into a single two-hop bridge question `Q` using a prompted rewrite (Figure 15).
    4) Package `Q`, the sub-questions, the reasoning chain, and the final answer.
    - Example (Figure 2): `D1=The Moon`, `E=Apollo 11`, `D2=Neil Armstrong`, `Q1=“What was the spaceflight that first landed humans on the Moon?”`, `Q2=“Who was the commander of Apollo 11?”`, merged `Q=“Who was the commander of the spaceflight that first landed humans on the Moon?”`, `A=Neil Armstrong`.
  - TQA (Figure 3):
    1) From a table and seed, zero-shot generate an SQL statement (prompt in Figure 12).
    2) Execute it with `sqlite3` to obtain the answer. If SQL is invalid/non-executable, discard (Section 3.1.3; Appendix D).
    3) Generate a natural-language question from the SQL (Figure 13).
    4) Package the table, SQL, question, and answer.

Why this design:
- Grounding in real sources constrains generations to be realistic, diverse, and factual.
- The seed anchors every example, making the reasoning/tool use concrete and checkable (Section 3.1.2–3.1.3).
- Executing SQL provides a direct correctness check in TQA (Section 3.1.3; Appendix D).

Stage 2 — Dataset Curation (Section 3.2)
- Split the synthetic set into two halves (slices):
  - Fine-tune an intermediate model `LLMSynth` on slice 0.
  - Use `LLMSynth` to curate slice 1 by filtering and imputation (Figure 1, purple path).
- Filtering by answerability (Section 3.2.1):
  - For each example in slice 1, let `LLMSynth` attempt to produce the correct answer up to `k=3` tries. If it never produces the correct answer, reject the example.
  - Quote: “If after k = 3 tries the model has not provided the correct answer, we discard the entry.” (Section 3.2.1)
  - Observed retention:
    - MHQA: remove ~13% (keep ~87%) (Section 3.2.2).
    - TQA: keep 27% (i.e., reject ~73%) (Section 3.2.2).
- Imputation (fill-in/repair) to improve naturalness and cohesion (Section 3.2.2; Appendix C.4):
  - MHQA only: delete `Q1` and ask `LLMSynth` to reconstruct `Q1` from context (`Q`, `Q2`, `E`, and `D1`). Keep only if the reassembled question yields the same final answer.
  - This reduces perplexity (a proxy for unnatural text), e.g., “Synthetic grounded data: 24.7 → 13.6; Synthetic ungrounded data: 15.51 → 8.33” (Table 9). Figure 6 shows a before/after example.
  - TQA uses filtering only (Section 3.2.2).

Why this design:
- Filtering targets the central failure mode of synthetic data—unanswerable or ill-posed examples—using a direct task success signal.
- Imputation repairs awkward phrasing produced by generation prompts, improving fluency while preserving answer consistency.

Stage 3 — Model Fine-tuning (Section 3.3)
- Train the final model `LLMCurated` on the curated synthetic dataset (with the chain-of-thought or SQL intermediate artifacts).
- Figure 4 shows how the fine-tuned model answers MHQA (left) and TQA (right) by producing the intermediate reasoning/SQL and the final answer.

## 4. Key Insights and Innovations
1) Real-data-grounded seeding to steer synthetic generation (Sections 3.1.1–3.1.3; Figures 2–3)
- What’s new: The method anchors each synthetic example to a concrete entity (`E`) in real documents or a fact derived from a real table. This contrasts with unguided instruction generation where examples can be contrived.
- Why it matters: Grounding sharply reduces hallucination and increases realism, improving downstream performance. Appendix C.3 shows that starting from ungrounded toy data yields notably worse accuracy than grounded data (e.g., a −6.82% soft-EM drop for Llama2 70B-Chat; Table 8).

2) Self-curation by answerability with an intermediate model (Section 3.2.1)
- What’s new: Use a model trained on half the synthetic set to decide whether the other half is answerable; reject after `k=3` failed attempts.
- Why it matters: This converts subjective quality control into an operational pass/fail criterion tied to the intended skill (reasoning or SQL execution). It enables heavy pruning where needed (keep only 27% for TQA; Section 3.2.2).

3) Imputation to improve linguistic naturalness without changing answers (Section 3.2.2; Appendix C.4)
- What’s new: Reconstruct sub-questions (`Q1`) in context and check answer consistency. Measured reductions in perplexity suggest more natural text (Table 9; example in Figure 6).
- Why it matters: Synthetic data quality is not just factuality; fluency/cohesion affect how well models learn from the data.

4) A single pipeline that generalizes across two hard capabilities: multi-hop reasoning and SQL tool use (Sections 3–5)
- What’s new: Demonstrated on MHQA (HotpotQA) and TQA (WikiSQL) with concrete, reproducible prompts and evaluation protocols (Figures 7–17; Sections 4–5).
- Why it matters: Shows that the same generation–curation principles transfer across qualitatively different tasks, supporting generality.

## 5. Experimental Analysis
Evaluation setup (Section 4)
- Datasets:
  - MHQA: HotpotQA (FullWiki) test set of 7,405 questions—half bridge, half comparison (Section 4.1). No data contamination with generated synthetic questions (Section 4.1).
  - TQA: WikiSQL test set (after removing non-executable SQLs; Appendix D). Training-time sources come only from the WikiSQL training tables, avoiding leakage (Section 4.2).
- Metrics:
  - `soft-EM` (MHQA): 1 if the predicted string contains the gold answer; 0 otherwise (Section 4.1).
  - `EM` and `soft-EM` (TQA): exact equality vs. contains (Section 4.2).
- Models and data quantities:
  - MHQA main model: `Llama2 70B-Chat`, fine-tuned on 1,250 synthetic bridge examples, with/without 500 HotpotQA training examples (“datamix”) (Section 4.1; Table 1). Prompts: zero-shot and 3-shot CoT (Figure 14).
  - TQA main model: `Starchat-beta` (16B code-focused LLM), fine-tuned on synthetic SQL-based samples (Section 4.2). Generation produced 10k SQL statements, then split into slices; filtering retained 27% (Sections 3.2.2, 4.2).

Baselines (Sections 4.1–4.2)
- MHQA:
  - Prompted instruction-tuned LLMs: `Llama2 70B-Chat`, `Claude 3.5 Sonnet`.
  - Fine-tuned on HotpotQA only (500 training examples).
  - Fine-tuned on synthetic only without curation (`LLMSynth`) and with datamix (`LLMSynth-datamix`).
- TQA:
  - Prompt-only variants of `Starchat-beta`: zero-shot table; one-shot no-context; one-shot with table; one-shot with table + SQL tool execution.
  - Fine-tuned on synthetic only without curation (`LLMSynth`) and with curation (`LLMCurated`).

Main results (Tables 1–2; Figure 5)
- MHQA (Table 1; soft-EM):
  - Prompt-only:
    - `Llama2 70B-Chat`: 40.45% (0-shot), 44.13% (3-shot).
    - `Claude 3.5 Sonnet`: 50.3% (0-shot), 53.4% (3-shot).
  - Supervised fine-tuning:
    - HotpotQA-only (500 ex): 53.22% (0-shot), 58.40% (3-shot).
    - Synthetic-only uncurated (`LLMSynth`): 52.31% (0-shot), 56.70% (3-shot).
    - Synthetic+HotpotQA uncurated (`LLMSynth-datamix`): 57.46% (0-shot), 62.73% (3-shot).
    - Synthetic-only curated (`LLMCurated`): 64.07% (0-shot), 64.68% (3-shot).
    - Synthetic+HotpotQA curated (`LLMCurated-datamix`): 65.23% (0-shot), 66.05% (3-shot).
  - Takeaways:
    - Curation consistently outperforms uncurated synthetic fine-tuning (e.g., 62.73% → 66.05% with datamix at 3-shot).
    - Curated synthetic-only training nearly matches curated datamix (64.68% vs. 66.05% at 3-shot), suggesting effectiveness even in low-data regimes.
  - Scaling with more synthetic data (Figure 5):
    - With 500 HotpotQA examples fixed, adding more synthetic data (500→750→1250) improves performance for both uncurated and curated models; curation always yields higher accuracy. During curation, 7–11% of examples are removed as synthetic set size grows.
- MHQA difficulty/type analysis (Appendix C.1, Table 4):
  - On HotpotQA “train” split labeled by difficulty, curated models improve across bridge and comparison types—even though training data for the main experiment includes only bridge-type synthetic questions. Example for hard questions: bridge 14.5% → 31.3%; comparison 66.6% → 83.1% (base → curated datamix).
  - The method can be extended to generate comparison-type questions by matching named-entity properties across documents (Section 5.1, “Extending Source2Synth…”). When trained on both bridge+comparison synthetic data, `LLMCurated` reaches 64.5% (0-shot; Table 3), comparable to earlier curated results.
- Smaller LLMs (Appendix C.2):
  - `Llama3 8B-instruct`: 57.8% → 71.13% (0-shot; Table 5) with curated synthetic bridge-only data.
  - `Llama4 17Bx16E`: 49.6% → 67.9% (0-shot; Table 6) with curated synthetic bridge+comparison data.
  - Indicates transfer to smaller, cheaper models.
- Grounding vs. ungrounded synthetic data (Appendix C.3):
  - Ungrounded synthetic training performs worse than grounded synthetic (e.g., `Llama2 70B-Chat`: 59.70% curated ungrounded vs. 66.05% curated grounded; Tables 8 and 1), supporting the core design choice to ground in real sources.
- TQA (Table 2):
  - Prompt-only `Starchat-beta` performs poorly unless allowed to generate and execute SQL (EM: 0.25–2.03% without SQL execution; 12.30% with one-shot + tool).
  - Fine-tuning with synthetic data:
    - Uncurated (`LLMSynth`): 23.86% EM, 34.21% soft-EM.
    - Curated (`LLMCurated`): 34.50% EM, 42.80% soft-EM.
  - Quote: “LLMCurated (synthetic data only) 34.50% EM; 42.80% soft-EM” vs. best prompt baseline “one-shot table+SQL tool QA 12.30% EM; 34.13% soft-EM” (Table 2). Curation adds ~10.6 EM points over uncurated fine-tuning.

Do the experiments support the claims?
- Yes, across:
  - Two distinct tasks (MHQA and SQL-based TQA).
  - Multiple model sizes.
  - Analyses of data quantity (scaling), question type/difficulty, and grounded vs. ungrounded generation.
  - Quality diagnostics for imputation (perplexity reductions; Table 9).
- Open details:
  - Exact synthetic data volumes for TQA vs. retention percentages are summarized at a high level; the paper notes aggressive filtering there (keep 27%) (Section 3.2.2).

## 6. Limitations and Trade-offs
Assumptions and scope (Section A)
- Two-hop MHQA only in main experiments:
  - Quote: “The MHQA number of hops is restricted to two in this paper.” They propose looping the generation to extend to more hops.
- Single-table TQA:
  - Quote: “We use a single table per query in TQA… Multi-table tool use is not supported.” No SQL joins across tables; no table retrieval.
- Task focus:
  - Quote: “Source2Synth is restricted to question-answering tasks.” It targets two skills: producing MHQA reasoning chains and producing SQL.
- Data source integrity:
  - Requires the source to be “clean enough.” The method partially mitigates inconsistencies via answerability filtering, and discards non-executable SQL, but the initial source parsing still matters (Section A; Appendix D).
Computational and pipeline trade-offs
- Generation + curation overhead:
  - Requires running multiple model passes (generation, fine-tuning for `LLMSynth`, k-try filtering, and imputation), executing SQL, and maintaining careful prompting (Figures 7–17).
- Rejection-heavy curation for TQA:
  - Keeping 27% implies substantial generation waste; the trade-off is higher final quality (Section 3.2.2).
- Prompt and model dependence:
  - The approach benefits from strong instruction-tuned LLMs to generate coherent seeds, questions, and SQL. Prompt changes can affect quality (Appendix C.5).

## 7. Implications and Future Directions
How it changes the landscape
- Demonstrates that synthetic data, when grounded and rigorously curated for answerability, can supply complex capabilities without manual labels—even for tasks often assumed to require human annotation (e.g., text-to-SQL; Sections 4–5).
- Establishes a scalable template: pick a real source → induce a task-specific seed → generate structured intermediate steps → filter by self-answerability → fine-tune.

Practical applications
- Domains rich in unstructured text or tables but poor in annotations:
  - Legal and medical QA (mentioned in Section 4), enterprise document QA, knowledge-base construction, analytics over internal tables via SQL.
- Tool learning beyond SQL:
  - Any tool whose outputs can be executed/checked (APIs, calculators, retrieval pipelines) can be plugged into the same generate–execute–filter loop.

Research directions
- Beyond two hops and single-table queries:
  - Multi-hop chains longer than two via iterative seeding (Section A), multi-table SQL with join reasoning and table retrieval (Section A), and integration with multi-hop retrievers (e.g., [Xiong et al., 2020] noted in Section A).
- Generalize the curation criterion:
  - Explore richer self-verification signals (confidence calibration, consistency checks across paraphrases, or cross-model agreement) beyond k-try answerability.
- Extend beyond QA:
  - The same grounding, seed design, and self-curation could be adapted to data-to-text generation, complex multi-tool workflows, code reasoning beyond SQL, or planning tasks.
- Data efficiency and diversity:
  - Improve acceptance rates (especially for TQA) through better seed induction, schema-aware SQL generation, or curriculum designs; investigate diversity controls to avoid repetitive patterns observed in ungrounded data (Appendix C.3).

In sum, Source2Synth operationalizes a practical recipe for turning raw corpora and tables into high-quality synthetic supervision. Its core ingredients—real-world grounding, answerability-based filtering, and imputation—produce measurable and consistent gains on challenging reasoning and tool-use tasks (Tables 1–2), making it a strong baseline for future work in self-generated training data.
