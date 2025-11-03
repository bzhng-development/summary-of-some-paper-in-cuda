# Source2Synth: Synthetic Data Generation and Curation Grounded in Real Data Sources

**ArXiv:** [2409.08239](https://arxiv.org/abs/2409.08239)
**Authors:** Alisia Lupidi, Carlos Gemmell, Nicola Cancedda, Jane Dwivedi‑Yu, Jason Weston, Jakob Foerster, Roberta Raileanu, Maria Lomeli
**Institutions:** 

## 🎯 Pitch

Source2Synth introduces a novel three-stage pipeline to generate high-quality synthetic training data for large language models (LLMs), focusing on multi-hop and tabular question answering. By grounding data generation in real-world sources and utilizing an innovative self-curation process, the approach significantly enhances model performance compared to current instruction-tuned methods. This innovation is crucial as it enables scalable, domain-adapted training without expensive manual annotations, offering significant impact for domains rich in unstructured data, such as legal and medical fields.

---

## 1. Executive Summary
Source2Synth is a three-stage pipeline to generate and curate synthetic training data for large language models (LLMs) using real-world sources (documents and tables), then fine-tune models on that data. It tackles two hard capabilities—multi-hop question answering (MHQA) and tabular question answering (TQA with SQL)—by grounding generation in actual articles or tables, filtering low-quality examples via answerability checks, and optionally “imputing” missing parts to improve naturalness. On HotpotQA (MHQA) and WikiSQL (TQA), the curated synthetic datasets yield sizable gains over instruction-tuned and fine-tuned baselines (e.g., 65.23% soft-EM vs. 53.22% on HotpotQA; 34.50% EM vs. 12.30% on WikiSQL), as reported in Table 1 and Table 2.

## 2. Context and Motivation
- Problem addressed:
  - High-quality, task-specific data for complex LLM skills (like multi-step reasoning and tool use) is scarce and expensive to annotate. Synthetic data is attractive, but often contrived, ungrounded, or low-quality, especially for multi-hop tasks and SQL-based table reasoning (Abstract; Section 1).
- Why it matters:
  - Many valuable applications (e.g., legal and medical QA) have abundant unstructured sources (documents, tables) but few labeled QA pairs or reasoning traces. Enabling LLMs to learn from grounded, curated synthetic data promises better factuality, diversity, and realism without human annotation (Section 1; Conclusion).
- Prior approaches and gaps:
  - Instruction-backtranslation and general synthetic instruction-tuning can create diverse data but struggle to produce high-quality multi-hop examples; only a small fraction are multi-hop and often low quality (Section 2; cites Chen et al., 2024; Wang et al., 2023b). 
  - Tool-use training often focuses on text and numbers, while relational databases (SQL) remain hard due to schema relevance (Section 2; Appendix B).
  - Some recent work uses web documents or code corpora to guide generation, but often needs back-translation or initial fine-tuning to identify task-specific seeds (Section 2).
- Positioning:
  - Source2Synth’s core idea is to ground synthetic generation in real sources and introduce a self-curation loop that filters examples by “answerability” using an intermediate model. It is demonstrated on two distinct capabilities—multi-hop reasoning with documents and SQL-based tool use with tables (Section 1; Figure 1).

## 3. Technical Approach
Source2Synth is a three-stage pipeline (Figure 1): Dataset Generation → Dataset Curation → Model Fine-tuning.

Key terms (defined where uncommon):
- `Seed` (paper-specific): a task-specific anchor extracted from a source (e.g., an entity in a document or a fact derived from a table) that conditions synthetic example construction (Section 3.1.2).
- `Rejection sampling` (here): discarding generated examples that fail a quality criterion—in this case, answerability by a model within a small number of attempts (Section 3.2.1).
- `Imputation` (here): removing part(s) of an example and asking a model to reconstruct them, keeping only examples whose reconstructed form still yields the original answer (Section 3.2.2).
- `Soft-EM` (metric): 1 if the generated output contains the reference answer as a substring; 0 otherwise (Section 4.1, Metrics).

Stage 1: Dataset Generation (Section 3.1)
- Data source selection (Section 3.1.1)
  - MHQA: English Wikipedia articles. Randomly select an initial article `D1`. Build a pool of related articles; select another article `D2` from that pool (Section 3.1.1).
  - TQA: Unlabeled tables from the WikiSQL training split (4,000 tables are mentioned as sources) (Section 3.1.1; Section 4.2).
- Seed creation (Section 3.1.2)
  - MHQA seed: an entity `E` sampled from `D1`. `E` must also appear in `D2` so it can act as the “hop” connecting sub-questions (Figure 2 shows an example where `E = Apollo 11`).
  - TQA seed: an “interesting statement” about the table, prompted from an LLM (e.g., “The country with most arrivals in 2012”; Figures 3 and 11).
- Construct synthetic examples (Section 3.1.3)
  - MHQA (Figures 2, 15–17):
    1) Prompt the model to generate `Q1` from `D1` so that the answer `A1` is the seed entity `E` (Figure 16).
    2) Prompt a second question `Q2` from `D2` whose main topic is `E` (Figure 17).
    3) Merge `Q1` and `Q2` into a single two-hop “bridge” question `Q` by substituting `Q1` into `Q2` where `E` appears (Figure 15). The example stores `Q`, the reasoning chain (decomposition into `Q1`/`Q2`), and the final answer `A`. Figure 2 illustrates the process end-to-end.
  - TQA (Figures 3, 11–13):
    1) Given the table and the seed, prompt an SQL query (zero-shot) that answers the seed’s question (Figure 12).
    2) Execute the SQL with `sqlite3` to get the answer; discard the example if the SQL is invalid or non-executable (Section 3.1.3; Figure 3; Appendix D notes that from 800 generated statements across 50 tables, 658 were executable).

Stage 2: Dataset Curation (Section 3.2)
- Split the synthetic dataset into two halves: `slice 0` and `slice 1` (Figure 1).
- Train an intermediate model `LLMSynth` on `slice 0`.
- Filter `slice 1` by answerability (rejection sampling; Section 3.2.1):
  - For each example, let `LLMSynth` attempt the question up to `k = 3` times; keep the example only if `LLMSynth` matches the reference answer (exact string for TQA; answer string containment for MHQA soft-EM) (Section 3.2.1).
- Imputation (MHQA only; Section 3.2.2; Figure 6; Table 9):
  - Remove `Q1`, and ask `LLMSynth` to reconstruct `Q1` given `Q`, `Q2`, `E`, and `D1`. Recompute the merged question. Keep the example only if the final answer remains unchanged. This step reduces linguistic unnaturalness, as quantified by lower perplexity after imputation (Table 9: 24.7 → 13.6 for grounded data).

Stage 3: Model Fine-tuning (Section 3.3)
- Train the final model `LLMCurated` on the curated synthetic data, supervising both the intermediate reasoning (e.g., `Q1`/`Q2` or SQL) and final answers (Section 3.3). Figure 4 shows sample outputs after fine-tuning:
  - MHQA: the model decomposes a multi-hop question into `Q1` and `Q2` and derives the answer.
  - TQA: the model inspects the table schema, writes SQL against `sql_table`, and returns the answer.

Design choices and why:
- Grounding in real sources (documents/tables) anchors examples in actual content, improving realism and factuality and reducing “contrived” patterns common in purely synthetic data (Sections 1, 2, 3.1).
- Answerability-based filtering removes ill-posed or overly ambiguous examples without human labels, leveraging an intermediate model as a quality filter (Section 3.2.1).
- Imputation addresses coherence/fluency issues in multi-hop question wording and ensures consistency of the answer after reconstruction (Section 3.2.2; Table 9; Figure 6).
- Task-specific seeds steer generation to multi-hop links (entities across documents) or table facts that naturally translate into SQL—improving relevance and difficulty (Sections 3.1.2–3.1.3; Figures 2–3, 11–13, 15–17).

## 4. Key Insights and Innovations
- Grounded synthetic generation with task-specific seeds (fundamental):
  - Novelty: Instead of generic instruction synthesis, the pipeline selects a `seed` from real sources (entity or table-derived fact) and forces the synthetic example to be anchored to that seed (Sections 3.1.2–3.1.3; Figures 2–3). This reduces hallucination and improves relevance.
  - Significance: Enables creation of high-quality multi-hop and SQL-augmented QA without human labels; observed to outperform instruction-tuned LLM prompting and even some fine-tuned baselines (Tables 1–2).
- Self-curation via answerability with an intermediate model (fundamental):
  - Novelty: Split the synthetic set, train `LLMSynth` on the first half, and use it to filter the second half by whether it can produce the correct answer in `k=3` tries (Section 3.2.1; Figure 1).
  - Significance: Improves final data quality without external supervision, yielding consistent performance gains over uncurated synthetic data across scales (Table 1; Figure 5).
- Imputation to improve naturalness and maintain answer consistency (incremental but impactful for MHQA):
  - Novelty: Reconstruct a sub-question (`Q1`) and keep the example only if the final answer is preserved (Section 3.2.2).
  - Significance: Reduces perplexity (Table 9) and removes awkward phrasings (Figure 6), which can improve learning signal for reasoning.
- Cross-capability applicability: documents and tables (incremental but practical):
  - The same pipeline template works for two different skills—multi-hop reasoning and SQL-based table QA—simply by adapting sources and seeds (Figures 2–3). This suggests broader extensibility to other domains with unstructured sources (Conclusion; Section 6).

## 5. Experimental Analysis
Evaluation setup and baselines
- MHQA (Section 4.1)
  - Dataset: HotpotQA FullWiki test set (7,405 examples; bridge and comparison), a benchmark requiring multi-document reasoning (Section 4.1).
  - Generation scope: Source2Synth generates only bridge-type synthetic questions from 50 random Wikipedia articles (1,250 examples). To balance types for one variant, fine-tuning adds 500 comparison questions from HotpotQA train (Section 4.1).
  - Metric: soft-EM (1 if the model’s output contains the gold answer, else 0) (Section 4.1).
  - Baselines:
    - Instruction-tuned LLMs: `Llama2 70B-Chat`, `Claude 3.5 Sonnet` (Table 1).
    - Fine-tuned baseline on 500 HotpotQA examples.
    - `LLMSynth` (uncurated synthetic) and `LLMSynth-datamix` (synthetic + 500 HotpotQA).
  - Contamination check: For each synthetic question, ensure its seed entity `E` is not used in the test-set questions; none overlapped (Section 4.1).
- TQA (Section 4.2)
  - Dataset: WikiSQL (24,241 tables; 80,654 NL–SQL–table triples). Evaluation is on the test set; sources for synthetic generation are from the train split to avoid contamination (Section 4.2).
  - Metric: EM and soft-EM (Section 4.2).
  - Baselines (all using `Starchat-beta` via prompting): zero-shot table QA; one-shot without table; one-shot with table; one-shot with table+SQL tool; and `LLMSynth` vs. `LLMCurated` (Table 2).
  - Implementation: Fine-tuning `Starchat-beta` with batch size 32, 100 steps, LR=1e-4 (Section 4.2). They generate 10,000 SQL statements and keep 8,000 examples per slice; invalid SQL discarded via `sqlite3` (Sections 3.1.3, 4.2; Appendix D).

Main quantitative results
- MHQA (HotpotQA, Table 1)
  - Instruction-tuned prompting baselines:
    - `Llama2 70B-Chat`: 40.45% (0-shot), 44.13% (3-shot).
    - `Claude 3.5 Sonnet`: 50.3% (0-shot), 53.4% (3-shot).
  - Fine-tuned on HPQA only (500 ex): 53.22% (0-shot), 58.40% (3-shot).
  - Synthetic (uncurated): 
    - `LLMSynth`: 52.31% (0-shot), 56.70% (3-shot).
    - `LLMSynth-datamix`: 57.46% (0-shot), 62.73% (3-shot).
  - Source2Synth curated:
    - `LLMCurated` (synthetic only): 64.07% (0-shot), 64.68% (3-shot).
    - `LLMCurated-datamix` (synthetic + HPQA): 65.23% (0-shot), 66.05% (3-shot).
  - Takeaways:
    - Curated synthetic data substantially outperforms instruction-tuned prompting and fine-tuning on 500 HPQA examples.
    - Adding HPQA comparison questions gives a small boost over synthetic-only (65.23% vs. 64.07% 0-shot; Section 5.1).
    - Three-shot prompting adds modest gains for non-curated models but little for curated ones, likely because the model was already trained on reasoning traces (Table 1).
  - Scaling (Figure 5):
    - As the number of synthetic examples increases (500 → 750 → 1250), performance improves; curation consistently adds gains on top of uncurated data. Reported filtering rates: 7%, 8%, and 11% removed for synthetic sizes 500, 750, and 1250 respectively (Section 5.1).
  - Difficulty/type analysis (Appendix C.1; Table 4):
    - On the HotpotQA train set labeled by difficulty, the curated synthetic model improves over the base LLM by +13.38% on average, and over the HPQA-finetuned model by +7.35%. Gains are largest on hard questions, e.g., +16.8% (hard bridge) and +16.5% (hard comparison).
  - Generating comparison questions too (Table 3):
    - If synthetic data includes both bridge and comparison questions, `LLMCurated` reaches 64.5% (0-shot), close to `LLMCurated-datamix` in Table 1 (65.23%), indicating the method can also synthesize comparison-style multi-hop.
  - Smaller LLMs (Appendix C.2; Tables 5–6):
    - `Llama3 8B-instruct`: 57.8% → 71.13% with curated synthetic bridge questions.
    - `Llama4-17Bx16E`: 49.6% → 67.9% with curated synthetic bridge+comparison questions.
- TQA (WikiSQL, Table 2)
  - Prompting baselines (EM / soft-EM):
    - One-shot no context: 0.25% / 16.22%.
    - Zero-shot table QA: 1.83% / 20.07%.
    - One-shot table QA: 2.03% / 31.06%.
    - One-shot table + SQL tool: 12.30% / 34.13%.
  - Synthetic fine-tuning:
    - `LLMSynth` (uncurated): 23.86% / 34.21%.
    - `LLMCurated`: 34.50% / 42.80%.
  - Takeaways:
    - Fine-tuning on curated synthetic SQL data yields large EM gains over the tool-augmented prompting baseline (+22.2 EM points; 12.30% → 34.50%) and boosts soft-EM by +8.67 points (34.13% → 42.80%).
    - Even uncurated synthetic fine-tuning (`LLMSynth`) already surpasses prompting baselines, but curation adds a substantial further improvement (Table 2).

Ablations, robustness, and checks
- Imputation improves linguistic naturalness:
  - Average perplexity drops after imputation (Table 9): from 24.7 → 13.6 (grounded) and 15.51 → 8.33 (ungrounded). Figure 6 shows an example where the reconstructed wording is simpler while preserving semantics.
- Grounding matters (Appendix C.3; Tables 7–8):
  - Training on synthetic data generated from “made-up” topics (no real sources) hurts performance compared to grounded generation. For example, with `Llama3 8B-instruct`, curated ungrounded synthetic yields 66.37% vs. 71.13% with grounded synthetic (−4.76 points; Table 5 vs. Table 7). With `Llama2 70B-Chat`, curated ungrounded reaches 59.70% vs. 65.23% with grounded (−5.53 points; Table 8 vs. Table 1).
- Data contamination:
  - MHQA: explicit seed-based overlap check found no duplicates with the HotpotQA test set (Section 4.1).
  - TQA: sources are from WikiSQL train; evaluation is on the test split, ensuring disjointness (Section 4.2).
- Prompt sensitivity:
  - Appendix C.5 shows prompt variants for `Llama2 70B-Chat` on HotpotQA; the standard zero-shot and three-shot CoT prompts outperform role or multi-shot templates in this setup (Table in Appendix C.5).

Do the experiments support the claims?
- Evidence is consistent, multi-faceted, and includes:
  - Multiple baselines (prompting, small fine-tunes, uncurated vs. curated synthetic).
  - Scaling analysis (Figure 5).
  - Type/difficulty breakdown (Table 4).
  - Cross-model generality (smaller LLMs; Tables 5–6).
  - Grounding ablation (Tables 7–8).
- The strongest gains align with the method’s two main levers: grounding and curation.

## 6. Limitations and Trade-offs
Assumptions and scope
- Two-hop constraint in MHQA:
  - The presented pipeline targets two-hop “bridge” questions (Figures 2, 15). Extending to more hops would require iterating the generation loop (Appendix A).
- Single-table constraint in TQA:
  - Examples involve one table per query; the pipeline does not handle multi-table joins or table retrieval across a set (Appendix A).
- Task scope:
  - The method is developed for QA tasks (documents and tables). Extending to other task formats/capabilities is future work (Appendix A; Section 6).

Quality and curation trade-offs
- Answerability as a proxy for quality:
  - Filtering keeps examples that the intermediate model can solve within 3 tries (Section 3.2.1). This favors solvable, well-posed instances but may bias the dataset toward the intermediate model’s strengths and away from harder cases.
- Imputation only for MHQA:
  - TQA curation uses filtering only; potential benefits of imputation (e.g., reconstructing SQL or NL paraphrases) are unexplored here (Section 3.2.2).

Data and implementation details
- Reliance on data cleanliness:
  - The approach assumes the source data is reasonably consistent; some issues are mitigated by filtering (e.g., dropping non-executable SQL), but noisy/incorrect sources can still percolate (Appendix A; Section 3.1.3; Appendix D).
- Minor clarity inconsistency:
  - The paper states both that TQA generated 10,000 SQL statements (8,000 per slice kept; Section 4.2) and that “we keep 27% of the original examples” for TQA after curation (Section 3.2.2). The exact final counts per slice could be clarified.
- Compute and resource considerations:
  - Training and curation involve large models (e.g., `Llama2 70B-Chat`), though results with smaller models mitigate this to some extent (Appendix C.2).

## 7. Implications and Future Directions
How this changes the landscape
- Provides a practical recipe to unlock advanced LLM capabilities without manual labels, by leveraging abundant unstructured sources and an automated, self-improving curation loop (Figure 1; Sections 1, 6).
- Demonstrates that synthetic data—when grounded and curated—can outperform instruction-tuned prompting and even small supervised fine-tunes on standard benchmarks (Tables 1–2).

Follow-up research opportunities
- More hops and richer reasoning:
  - Extend MHQA beyond two hops; integrate multi-hop retrieval techniques (e.g., Xiong et al., 2020) to replace simple rejection sampling in document selection (Appendix A).
- Multi-table and compositional tool use:
  - Support SQL joins, table retrieval, and cross-table aggregation; consider using table encoders (e.g., TAPAS) for retrieval or schema linking (Appendix A).
- Broader tasks and modalities:
  - Apply the Source2Synth template to other complex tasks (e.g., verification, planning, code reasoning, or multi-modal table+text scenarios), using appropriate seeds and intermediate structures (Section 6; Appendix B).
- Stronger curation signals:
  - Beyond answerability, incorporate verification (e.g., tool-grounded consistency checks), diversity/novelty metrics, or uncertainty-based sampling to retain challenging but learnable examples.
- Domain deployment:
  - Legal and medical QA are highlighted as promising areas with abundant unstructured sources but limited annotations (Section 6). Careful source vetting and domain-specific seeds could yield high-value synthetic corpora.

Practical applications
- Enterprises with proprietary document corpora or internal databases can synthesize high-quality training data to teach in-house LLMs multi-step reasoning or SQL tool use without exposing data to annotators.
- Education and analytics systems can generate curriculum-aligned multi-hop questions or table analytics questions with verifiable SQL traces.
- Research pipelines can rapidly prototype new reasoning tasks by swapping in domain-specific sources and seed templates.

> Representative results:
> - MHQA (HotpotQA): `LLMCurated-datamix` reaches 65.23% soft-EM (0-shot) vs. 53.22% for a model fine-tuned on 500 HPQA examples (Table 1).
> - TQA (WikiSQL): `LLMCurated` attains 34.50% EM and 42.80% soft-EM vs. 12.30% EM and 34.13% soft-EM for the best prompting baseline (one-shot with SQL tool) (Table 2).
> - Imputation reduces perplexity: 24.7 → 13.6 on grounded MHQA questions (Table 9), indicating more natural phrasing.

Overall, Source2Synth offers a simple yet powerful recipe—grounded generation plus self-curation—that consistently improves complex LLM capabilities with minimal human effort, and opens a path toward scalable, domain-adapted synthetic data creation.
