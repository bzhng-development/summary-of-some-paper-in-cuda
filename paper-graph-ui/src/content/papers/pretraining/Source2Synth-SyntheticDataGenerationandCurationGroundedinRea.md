# Source2Synth: Synthetic Data Generation and Curation Grounded in Real Data Sources

**ArXiv:** [2409.08239](https://arxiv.org/abs/2409.08239)

## 🎯 Pitch

Source2Synth introduces a scalable pipeline for generating high-quality synthetic training data grounded in real-world sources (like documents or tables), combined with an automated curation step that filters out low-quality examples based on model-predicted answerability. By anchoring synthetic data in actual source material and rigorously curating it, Source2Synth dramatically boosts large language model performance in complex tasks such as multi-hop reasoning and tool-based question answering, demonstrating substantial gains over both instruction-tuned and human-annotated baselines. This approach paves the way for powerful, low-cost domain adaptation in settings where annotated data is scarce but raw databases or documents are plentiful.

---

## 1. Executive Summary

This paper introduces **Source2Synth**, a method for generating and curating synthetic training data grounded in real-world data sources to unlock advanced LLM capabilities without human annotation. The method is evaluated on two tasks—multi-hop question answering on HotpotQA with Llama2 70B-Chat, and tabular question answering on WikiSQL with Starchat-beta—using two distinct grounding mechanisms: **MHQA seed** (an entity sampled from a Wikipedia article that links sub-questions across multiple documents) and **TQA seed** (a fact derived from a table that anchors SQL query generation). Source2Synth achieves a 25.51% improvement for TQA on WikiSQL and a 22.57% improvement for MHQA on HotpotQA over fine-tuned baselines, with the curation step (rejection sampling via an intermediate fine-tuned model that discards unanswerable examples) proving essential to these gains. The method also establishes that a model fine-tuned exclusively on synthetic data with only bridge-type questions generalizes to comparison-type questions, outperforming the base LLM by an absolute 16.5% on hard comparison questions.

## 2. Context and Motivation

### The Core Problem: High-Quality Task-Specific Training Data Is Scarce and Expensive

The fundamental problem this paper addresses is deceptively simple: **how do you teach an LLM to perform a complex, structured reasoning task when you have no human-annotated training data?** This matters because the remarkable capabilities of modern LLMs—from answering multi-hop questions to querying databases with SQL—are typically unlocked through supervised fine-tuning on carefully curated, task-specific datasets. When those datasets don't exist, the capabilities remain locked.

The paper identifies two specific advanced capabilities that exemplify this problem (Section 1):

1. **Multi-hop question answering (MHQA)**: Answering questions that require finding and reasoning over multiple supporting documents. For example, "Who was the commander of the spaceflight that first landed humans on the Moon?" requires first identifying Apollo 11, then identifying Neil Armstrong—a two-step reasoning chain across different Wikipedia articles.

2. **Tabular question answering (TQA)**: Answering natural language questions by querying a structured table with SQL. For example, "What country had the most tourist arrivals in 2012?" requires composing a valid SQL query (`SELECT Country FROM table WHERE Year=2012 ORDER BY Arrivals DESC LIMIT 1`), executing it, and returning the result.

For both tasks, creating training data through traditional means is prohibitively expensive. The HPQA and WikiSQL benchmarks the paper evaluates on required thousands of human-annotated examples (113,000 for HotpotQA, 80,654 for WikiSQL). Human annotation at this scale is slow, costly, and introduces its own error modes—inconsistent labeling, annotator bias, and fatigue (Gilardi et al., 2023; Sylolypavan et al., 2023). In specialized domains like medicine or law, the annotators themselves need domain expertise, multiplying the cost further.

The theoretical significance of this problem extends beyond cost. Even if money were no object, **human annotation fundamentally constrains what you can teach a model**: you can only annotate data for tasks you already know how to solve. Synthetic data generation, if done correctly, opens the possibility of generating training examples for tasks where no human-annotated data exists and where the generation process itself encodes transferable skills (like decomposition into sub-questions or SQL composition).

---

### Why Existing Synthetic Data Approaches Fall Short

The paper identifies several specific failure modes in prior synthetic data generation methods (Section 2, Related Work).

**General-purpose instruction generation produces low-quality multi-hop data.** Methods like Self-Instruct (Wang et al., 2023a) prompt an LLM to generate diverse task instructions from a seed set. While effective for general instruction tuning, these approaches struggle with complex, structured tasks:

> "If a general recipe is used (Wang et al., 2023a), only a small fraction of synthetic instruction-tuning samples are multi-hop, and they can exhibit poor quality (Chen et al., 2024)."

The problem is twofold. First, the generation process has no grounding mechanism—the LLM generates questions from its parametric knowledge, which can produce factually incorrect or logically inconsistent examples. Second, the generation is not tailored to the structure of multi-hop reasoning. The LLM doesn't naturally decompose questions into sub-questions with a linking entity; it just generates text that looks like a complex question. Training on such data teaches the model surface patterns, not the underlying decomposition skill.

**Knowledge-probing methods are limited to template-filling.** Approaches that generate data by masking entities in text and having the model predict them (Schick and Schütze, 2020; Petroni et al., 2019) produce examples that are structurally simple—essentially cloze tests. They don't capture the multi-step reasoning required for MHQA or the tool-use patterns required for TQA.

**Instruction back-translation requires initial fine-tuning.** Recent work like instruction back-translation (Li et al., 2024; Nguyen et al., 2024) generates instruction-tuning data by taking web documents, having a model generate instructions that those documents could answer, and then fine-tuning on the resulting pairs. While this grounds the data in real content, it requires an initial fine-tuned model to bootstrap the back-translation process. Source2Synth, in contrast, generates data from scratch using only prompting, with no pre-existing fine-tuned model needed (Section 2: "Source2Synth does not require a back-translation approach or initial fine-tuning to generate the task-specific seed").

**Tabular QA data typically requires human annotation.** For the specific case of teaching models to use SQL on tables, prior approaches predominantly rely on human-written (question, SQL, answer) triples (Li et al., 2023a). Generating synthetic SQL that is both syntactically valid and semantically meaningful for a given table is challenging because the space of valid SQL queries is combinatorially large and most randomly generated queries are either trivially simple or nonsensical.

**Verifier-based filtering alone is insufficient.** Some prior work uses model-based filtering to improve synthetic data quality (Schick and Schütze, 2021; Liu et al., 2022). While the paper adopts filtering in its curation stage, it argues that **filtering alone is not enough**—you need the generation process itself to be grounded in real data to produce diverse, realistic, and factually correct examples. Without grounding, even a good filter can only reject bad examples; it cannot create good ones.

---

### The Critical Insight Gap: Grounding as the Missing Ingredient

The central insight that motivates Source2Synth is that **the generation process must be anchored to something real**. When an LLM generates a multi-hop question from scratch ("What is the main ingredient in the flagship product of Ferrero?"), it relies entirely on its parametric knowledge. If that knowledge is wrong or incomplete, the generated training example teaches incorrect facts. And because the LLM has no mechanism to verify consistency across the sub-questions, it may produce examples where the linking entity (the "hop") doesn't actually connect the two sub-questions logically.

The paper's position is that real-world data sources solve both problems simultaneously (Section 3.1):

- **Factual correctness**: By extracting entities, facts, and relationships from actual Wikipedia articles or database tables, the generated examples are constrained to reflect real-world information. The answer to "Who was the commander of Apollo 11?" will be factually correct because it's derived from a real Wikipedia article about Neil Armstrong, not from the LLM's potentially hallucinated knowledge.

- **Structural consistency**: By using a task-specific "seed" (an entity in MHQA, a fact in TQA) that is explicitly extracted from the source and threaded through every step of generation, the method enforces that all components of the example—the sub-questions, the reasoning chain, the final answer—are logically connected. The entity "Apollo 11" is not just mentioned in the question; it's the answer to Q1, the topic of Q2, and the link that binds them into a coherent multi-hop question.

- **Realistic diversity**: Real-world sources contain idiosyncratic details and edge cases that synthetic generation from scratch tends to smooth over. By sampling from a diverse corpus (random Wikipedia articles, random tables), the generated data inherits the natural long-tail distribution of real content.

This grounding approach distinguishes Source2Synth from methods that use real data merely as a source of topics or keywords. In Source2Synth, the real data source is not just inspiration—it's the structural backbone that determines what questions can be asked and how they must be decomposed.

---

### Why a Curation Step Is Necessary (and Why It's Non-Trivial)

Even with grounding, the paper acknowledges that **not all generated examples will be high quality** (Section 3.2). The LLM may produce questions that are grammatically awkward, logically inconsistent despite the grounding, or simply too difficult for the model to answer after fine-tuning. The paper's curation method is a specific response to this challenge: it uses the model itself to assess answerability.

The key design choice—and what makes the curation non-trivial—is the **slice training protocol**. Rather than using a separate verifier model or a fixed heuristic, the paper:

1. Splits the synthetic dataset in half (Slice 0 and Slice 1).
2. Fine-tunes an intermediate model (`LLMSynth`) on Slice 0.
3. Uses `LLMSynth` to test each example in Slice 1: if the model cannot produce the correct answer in $k = 3$ attempts, the example is discarded.

This is a form of **rejection sampling where the rejection criterion is the student model's own competence**. The intuition is subtle: an example that even a partially-trained student model (trained on other synthetic data) cannot answer is likely either too hard, too noisy, or structurally flawed. Training on such examples would waste capacity and potentially degrade performance. The 13% rejection rate for MHQA and the 73% rejection rate for TQA (Section 3.2.1) underscore just how much low-quality data the generation process produces—and how essential curation is.

This slice-based self-curation connects conceptually to ideas in active learning and curriculum learning, where the model's own difficulty assessments guide training data selection. But the implementation is uniquely self-contained: no external verifier, no human judgments, no oracle access to ground truth beyond what the synthetic data itself provides.

---

### The Imputation Step: Fixing Structural Awkwardness

A more subtle problem the paper identifies with synthetic MHQA data is **unnaturalness in the merged multi-hop question** (Section 3.2.2). When the LLM merges Q1 ("What was the spaceflight that first landed humans on the Moon?") and Q2 ("Who was the commander of Apollo 11?") into a single two-hop question, the result can be grammatically awkward or contain redundant detail from the merging process.

The imputation step addresses this by having `LLMSynth` reconstruct Q1 from the other components (Q, Q2, the entity E, and the source document D1). If the reconstructed $Q_1'$ produces a multi-hop question $Q'$ whose answer still matches the original answer A, the example is kept. This effectively "smooths out" the synthetic data—the model, having been fine-tuned on Slice 0, generates a more natural decomposition than the original prompted generation. The perplexity measurements in Appendix C.4 quantify this: perplexity drops from 24.7 to 13.6 after imputation, confirming that the questions become more natural.

---

### Scope: Two Tasks, Two Data Types, One Unified Framework

The paper deliberately chooses two tasks that leverage fundamentally different types of source data—documents for MHQA, tables for TQA—to demonstrate the generality of the Source2Synth framework (Section 1). The common thread is the **seed-conditioned generation pipeline**: identify a grounding element from the source, use it to generate structured intermediate steps (sub-questions or SQL), compose them into a complete example, and filter for quality.

This generality is important because it suggests Source2Synth is not a bespoke solution for one benchmark, but a template that can be applied to any domain where:
1. A real data source exists (documents, tables, knowledge graphs, code repositories).
2. The task can be decomposed into a structured reasoning chain anchored to the source.
3. The correctness of the reasoning chain can be verified (either through answer matching, as in QA, or through execution, as in SQL).

The paper explicitly gestures toward domain-specific applications in medicine and law (Section 6), where unstructured data is abundant but annotated QA pairs are scarce—precisely the regime where Source2Synth's grounding approach would be most valuable.

## 3. Technical Approach

This is primarily a **data generation and curation paper** whose core idea is that synthetic training data for complex reasoning tasks must be grounded in real-world data sources and filtered through self-assessment of answerability to achieve high quality.

### 3.1 Reader orientation

Source2Synth is a three-stage pipeline that takes raw data sources (Wikipedia articles or database tables) and produces high-quality, curated synthetic training examples for fine-tuning LLMs on complex reasoning tasks. The system solves the problem of generating task-specific training data without human annotation by anchoring every synthetic example to a real-world "seed" extracted from the source, then using the model itself to filter out examples it cannot answer—ensuring the final training set contains only examples the model can actually learn from.

### 3.2 Big-picture architecture (diagram in words)

The system has three sequential stages, each with distinct sub-components:

1. **Dataset Generation** — Takes a data source (Wikipedia articles for MHQA, unlabeled tables for TQA) and produces synthetic examples. For each entry in the source, it extracts a task-specific seed, uses that seed to condition the generation of structured intermediate steps (sub-questions or SQL queries), and assembles a complete training example with reasoning chain and final answer.

2. **Dataset Curation** — Splits the synthetic dataset into two halves. The first half (Slice 0) fine-tunes an intermediate model called `LLMSynth`. This `LLMSynth` then serves as a quality filter for the second half (Slice 1): for each example, `LLMSynth` attempts to predict the answer $k = 3$ times; if it fails all three, the example is rejected. For MHQA only, an additional imputation step reconstructs sub-questions to fix grammatical unnaturalness.

3. **Model Fine-tuning** — The curated synthetic dataset (possibly augmented with a small amount of real data, like 500 HPQA examples) is used for supervised fine-tuning of the final model `LLMCurated`, which learns both the reasoning chain and the final answer.

Information flows as follows: raw data source → seed extraction → structured generation (sub-questions/SQL + answer) → synthetic dataset → split into Slice 0 and Slice 1 → Slice 0 fine-tunes `LLMSynth` → `LLMSynth` filters and/or imputes Slice 1 → curated dataset → final fine-tuning of `LLMCurated`.

### 3.3 Roadmap for the deep dive

- **First**, the Dataset Generation stage in full detail—data source selection criteria, what a seed is and why it matters, and how seeds condition the step-by-step construction for both MHQA and TQA separately, since the mechanisms differ fundamentally between the two tasks.
- **Second**, the Dataset Curation stage—the slice-training protocol, the filtering mechanism (rejection sampling with $k=3$ tries), and the imputation step specific to MHQA, including how imputation addresses structural unnaturalness and how correctness is verified post-imputation.
- **Third**, the Model Fine-tuning stage—what is trained, on what data, and what the output model `LLMCurated` produces at inference time, connecting the synthetic training format to the model's expected behavior.
- **Fourth**, the design choices and their justifications—why two slices instead of a separate verifier, why $k=3$, why imputation is needed only for MHQA, and why bridge questions only are generated despite evaluation including comparison questions.

### 3.4 Detailed, sentence-based technical breakdown

Source2Synth is not a single algorithm but a **generation-curation-finetuning pipeline** that produces synthetic training data for two distinct complex reasoning tasks. Its defining architectural choice is that every synthetic example is anchored to a real-world data source through a task-specific seed, which serves as the invariant spine that keeps the generated question, reasoning chain, and answer logically consistent. The curation stage then uses the model's own ability (after partial training) to assess answerability, discarding examples that are too noisy or difficult to learn from.

---

#### Dataset Generation: Data Source Selection

The first step in Source2Synth is selecting a real-world data source. The paper chooses sources that are publicly available, unstructured or semi-structured, and contain the kinds of entities and relationships needed for the target task (Section 3.1.1). No human annotations are required on these sources—the data exists as-is, and Source2Synth enriches it through self-augmentation.

**For MHQA**, the data source is English Wikipedia. The choice is motivated by two properties: (1) Wikipedia articles contain natural language text with explicit entity references and hyperlinks, making it straightforward to extract entities and identify related articles, and (2) the HotpotQA benchmark itself is built on Wikipedia, so using it as a source produces in-distribution examples without contaminating the test set (since the specific articles and entities sampled for generation are disjoint from those used in HPQA test questions). The paper explicitly verifies this non-contamination: for each synthetic question, they check whether its entity (seed) appears in any HPQA test-set question, and if so, whether the questions are identical—finding zero overlaps (Section 4.1, "Evaluation data contamination checks").

The procedure for selecting source documents (Section 3.1.1, MHQA):

1. **Randomly select an initial article** $D_1$ from among all available Wikipedia articles. This provides the primary context for the first sub-question.
2. **Collect a pool of $n \geq 2$ related articles** for $D_1$, where "related" means articles that are linked from $D_1$ or that share entities with $D_1$.
3. **Sample a second document** $D_2$ from this pool. The sampling is constrained: $D_2$ must contain the entity $E$ that will serve as the seed (the "hop" connecting the two sub-questions).

The paper fixes $n = 2$ (two-hop questions) to match the structure of HotpotQA, but notes in Appendix A that the method can be extended to more hops by looping the generation steps—feeding the output of one hop as input to the next.

**For TQA**, the data source is four thousand unlabeled tables from the WikiSQL training dataset (Section 3.1.1, TQA). These tables contain columns and rows of structured data but come with no associated questions, SQL queries, or answers. The paper deliberately uses tables from the WikiSQL training split to avoid test-set contamination, since the WikiSQL splits are mutually exclusive by design (Section 4.2, "Evaluation data contamination checks").

A table in this context is a relational table with named columns and typed rows—for example, a table might have columns `Year`, `Country`, and `Arrivals` with rows like `(2012, USA, 21.7 million)`. The table is stored in a format that can be queried with SQL via the `sqlite3` Python library.

---

#### Dataset Generation: The Seed Concept

The seed is the central innovation in Source2Synth's generation process—it is a task-specific anchor extracted from the source data that conditions every subsequent step of example construction, ensuring that the question, the reasoning chain, and the answer are all logically connected to the same ground-truth element from the source. Without a seed, the LLM would generate questions from unconstrained parametric knowledge, producing examples that may be factually inconsistent or structurally incoherent.

**MHQA seed** (Section 3.1.2, MHQA): The seed is an **entity $E$** sampled from the first document $D_1$. For example, if $D_1$ is the Wikipedia article "The Moon," the extracted entity might be $E = \text{"Apollo 11"}$. This entity serves as the conceptual "hop" that links the two sub-questions in the multi-hop question. Specifically:

- $Q_1$ is a question about $D_1$ whose answer is exactly $E$.
- $Q_2$ is a question about $D_2$ whose main topic is $E$ (i.e., $E$ appears in $Q_2$ or is the subject of $Q_2$).
- When $Q_1$ and $Q_2$ are merged into the final multi-hop question $Q$, the entity $E$ is the hidden link that the model must infer: answering $Q_1$ yields $E$, and $E$ is substituted into $Q_2$ to form the complete question.

The seed is not just a topic or keyword—it is **the answer to the first sub-question and the subject of the second sub-question**, making it the structural linchpin of the multi-hop reasoning chain. This design ensures that every generated MHQA example has a well-defined decomposition path: find $E$ from $D_1$, then use $E$ to answer from $D_2$.

**TQA seed** (Section 3.1.2, TQA): The seed is an **interesting fact** derived from the table by prompting an instruction-tuned language model. Unlike the MHQA seed, which is extracted deterministically (an entity from the article), the TQA seed is generated zero-shot. The prompt used is shown in Figure 11:

> "Please generate an interesting statement about this table. The statement is a fact about one of the columns in the following table. {table} An interesting statement as a result of this is:"

The LLM responds with a natural language fact—for example, for a table of tourist arrivals by country and year, it might generate `"The country with most arrivals in 2012"`. This fact is not itself a question or an answer; it is a **description of a meaningful query** that could be asked about the table. This fact then conditions the SQL generation: given the table and this fact, the LLM generates an SQL statement that would retrieve the information described by the fact.

The choice to generate the seed via prompting rather than extracting it deterministically is a deliberate tradeoff. Unlike entities in Wikipedia text, which are explicitly marked (through hyperlinks, infoboxes, or named entity recognition), "interesting facts" about a table depend on the table's schema and content in ways that are hard to enumerate programmatically. Prompting the LLM to generate the fact leverages its language understanding to identify what kinds of queries are semantically meaningful for a given table structure. The risk is that the generated fact may be nonsensical or impossible to answer with SQL—this is mitigated by the subsequent filtering step that discards invalid SQL statements.

A critical consistency property: in both MHQA and TQA, the seed is **generated once and then reused throughout the construction process**. For MHQA, the entity $E$ extracted from $D_1$ is the fixed answer to $Q_1$ and the fixed topic of $Q_2$. For TQA, the fact generated from the table is the fixed semantic target of the SQL query. This single-source-of-truth design prevents the kind of inconsistency that would arise if different parts of the example were generated independently without a shared anchor.

---

#### Dataset Construction: Multi-Hop Question Answering (MHQA)

The MHQA dataset construction (Section 3.1.3, MHQA) proceeds through a sequence of five prompted generation steps, each building on the previous ones. The full generation process for a single example is illustrated in Figure 2. All generation steps use an instruction-tuned language model (in the paper's experiments, Llama2 70B-Chat) with carefully designed prompts.

**Step 1: Generate $Q_1$ from $D_1$ with answer constraint $E$.**

The prompt (Figure 16) instructs the LLM to:

> "Identify one entity in the following text. Come up with a question so that the answer to this question is the entity chosen earlier. The question must be based on the following text. Write your results as 'Question:' and then the question and 'Entity:' and then the entity. Text: {document_one}"

The output is a question $Q_1$ whose answer $A_1$ is constrained to be the entity $E$ that the LLM itself identified from the text. For example, given $D_1$ text about Apollo 11 landing on the Moon, the LLM might output:

- `Question: "What was the spaceflight that first landed humans on the Moon?"`
- `Entity: "Apollo 11"`

The design choice here is subtle but important: the LLM is asked to identify the entity *and then* generate the question, rather than being given the entity and asked to generate a question. This ensures that $Q_1$ is natural and genuinely answerable from $D_1$—the LLM picks an entity it can see in the text and formulates a question around it, rather than being forced to construct a question around an arbitrary entity that might not fit naturally.

**Step 2: Generate $Q_2$ from $D_2$ containing entity $E$.**

The prompt (Figure 17) instructs the LLM to:

> "Come up with a question based on the following text that contains the word: {entity} Text: {document_two}"

The output is a question $Q_2$ whose main topic is the entity $E$, and whose answer $A_2$ is derived from $D_2$. For example, given $D_2$ text about Neil Armstrong commanding Apollo 11, the LLM might output:

- `Q2: "Who was the commander of Apollo 11?"`
- `A2: "Neil Armstrong"`

Note that at this stage, the entity $E$ is explicitly present in $Q_2$—the question literally contains the word "Apollo 11." The merging step (Step 3) will replace this explicit mention with a reference to $Q_1$, creating the multi-hop structure.

**Step 3: Merge $Q_1$ and $Q_2$ into the multi-hop question $Q$.**

The merge prompt (Figure 15) is the most complex prompt in the pipeline. It provides three few-shot examples demonstrating the merging procedure and then instructs the LLM to apply the same structure. The merging algorithm is taught via these examples:

1. Answer $Q_1$ to obtain $A_1$ (which is the entity $E$).
2. Check if $A_1$ appears verbatim in $Q_2$.
3. If it does, rewrite $Q_2$ by deleting $A_1$ and substituting the text of $Q_1$ in its place.
4. The rewritten $Q_2$ becomes the final multi-hop question $Q$.

The three provided examples illustrate this pattern:

- Example 1: `Q1 = "What was the spaceflight that first landed humans on the Moon?"`, `A1 = "Apollo 11"`, `Q2 = "Who was the commander of Apollo 11?"` → after merging: `Q = "Who was the commander of the spaceflight that first landed humans on the Moon?"`
- Example 2: `Q1 = "What is the flagship product of Ferrero?"`, `A1 = "Nutella"`, `Q2 = "What is the main ingredient in Nutella?"` → after merging: `Q = "What is the main ingredient in the flagship product of Ferrero?"`
- Example 3: `Q1 = "When was Jesus born?"`, `A1 = "1 BCE"`, `Q2 = "Who was the Roman Emperor in 1 BCE?"` → after merging: `Q = "Who was the Roman Emperor when Jesus was born?"`

The key structural property of the merged question $Q$ is that **it requires solving $Q_1$ first to obtain the entity $E$, then substituting that entity into $Q_2$ to answer**. The entity $E$ is no longer explicitly mentioned in $Q$; it must be inferred from $D_1$. This is precisely the multi-hop reasoning pattern that the model needs to learn.

**Step 4: Assemble the full training example.**

The complete training example consists of:

- The multi-hop question $Q$
- The final answer $A = A_2$
- The sub-questions $Q_1$ and $Q_2$
- The reasoning chain: decompose $Q$ into $Q_1$ and $Q_2$, answer $Q_1$ to get $A_1$, substitute $A_1$ into $Q_2$, answer $Q_2$ to get $A$
- The entity $E$ (the seed/hop)
- The source documents $D_1$ and $D_2$ (used as context during generation but not necessarily included in the training example; the training example focuses on the question and reasoning chain)

**Step 5: Answer $A$ is directly derivable from $D_2$.**

The answer $A$ to the final multi-hop question is the answer to $Q_2$, which is derived from $D_2$. This means that the correctness of each generated example can be verified by checking whether $A$ matches the information in $D_2$, providing a ground-truth signal for curation.

**Scale and filtering at generation time.** The paper generates 1250 synthetic bridge questions from a collection of 50 randomly selected Wikipedia articles (Section 4.1, "Model"). Not all generation attempts succeed—some may produce $Q_2$ that does not contain the entity $E$, or the merge may fail to produce a coherent question. These failures are discarded at generation time, before the curation stage.

A critical scope note: Source2Synth **only generates synthetic data for bridge-type questions** (Section 4.1). Bridge questions build on a logical or causal link and require deriving intermediate statements—exactly the structure that the seed-based decomposition naturally produces. Comparison questions (e.g., "Who is taller, X or Y?") have a different structure that cannot be directly produced by the entity-hop mechanism. To counterbalance this, the paper includes 500 comparison questions from the HPQA training dataset in the fine-tuning data mix (Section 4.1, "Model"). Despite this training-data asymmetry, the fine-tuned model generalizes to comparison questions at test time, as shown in Table 4.

---

#### Dataset Construction: Tabular Question Answering (TQA)

The TQA dataset construction (Section 3.1.3, TQA) follows a different sequence from MHQA because the underlying task structure is fundamentally different: instead of decomposing a question across documents, the model must compose a SQL query that extracts information from a structured table. The generation process is illustrated in Figure 3.

**Step 1: Generate the seed (interesting fact) from the table.**

As described above, an instruction-tuned LLM is prompted (Figure 11) to produce a natural language fact about the table. Example output: `"The country with most arrivals in 2012."`

**Step 2: Generate an SQL statement from the table and seed.**

The prompt (Figure 12) zero-shot instructs the LLM:

> "Please generate SQL statements for the following table: {table} Seed: {seed} An interesting SQL statement as a result of this is"

The LLM outputs an SQL query that retrieves the information described by the seed. For the seed about the country with the most arrivals in 2012, the generated SQL might be:

```sql
SELECT Country FROM sql_table WHERE Year = 2012 ORDER BY Arrivals DESC LIMIT 1
```

The table is referred to as `sql_table` in the prompt, so the generated SQL is written against this named table. The paper uses the `sqlite3` Python library as the SQL execution engine.

**Step 3: Execute the SQL to obtain the answer.**

The generated SQL statement is executed against the source table using `sqlite3`. If the statement is syntactically valid and returns a result, that result becomes the answer $A$. If the statement is invalid (syntax error, column not found, type mismatch), **the entire example is discarded** at generation time (Section 3.1.3, TQA: "If the generated statement is invalid, we discard it").

This execution-based verification is a powerful quality control: unlike MHQA, where answer correctness depends on the factual accuracy of Wikipedia, TQA answer correctness is guaranteed by the table data itself. The SQL either executes and returns a result, or it doesn't. If it does, the answer is definitionally correct with respect to that table.

**Step 4: Translate the SQL into a natural language question.**

The prompt (Figure 13) zero-shot instructs the LLM:

> "I want to convert an SQL statement into a question. Here is the original table: {table} SQL: {SQL} What is the question that this SQL statement would be the answer to?"

The LLM outputs a natural language question $Q$ that corresponds to the SQL query. For the example SQL above, the output might be: `"What country had the most tourist arrivals in 2012?"`

This translation step is important because the final model `LLMCurated` will receive natural language questions as input and must produce SQL as output (see Figure 4, right). By generating the question from the SQL (rather than generating both independently), the method ensures that $Q$ and the SQL query are semantically aligned—the question asks exactly what the SQL retrieves.

**Step 5: Assemble the full training example.**

The complete TQA training example consists of:

- The table (the data context)
- The natural language question $Q$
- The SQL query (the reasoning chain)
- The answer $A$ (the executed SQL result)
- The seed fact (used during generation but not necessarily in the training example)

**Scale and filtering at generation time.** The paper generates ten thousand SQL statements based on the source tables and keeps eight thousand examples per slice after discarding invalid or non-executable SQL (Section 4.2, "Model"). More specifically, out of 50 tables with 800 seed statements generated, 658 produced executable SQL statements (Appendix D: "Out of 50 tables, we generate 800 seed statements and keep 658 executable SQL statements"). This 82.25% executability rate at generation time is separate from the subsequent curation filtering.

---

#### Dataset Curation: The Slice Training Protocol

The curation stage is where Source2Synth transitions from "generate lots of data" to "generate *good* data." It is essential to the paper's results: without curation, performance is substantially lower across both tasks (compare `LLMSynth` vs. `LLMCurated` in Tables 1 and 2). The curation process is depicted in the purple section of Figure 1.

**Step 1: Split the synthetic dataset into two halves.**

The full synthetic dataset—whether MHQA or TQA examples—is divided into two sections, each containing half the number of synthetic examples (Section 3.2). These are denoted as Slice 0 and Slice 1. The split is presumably random, though the paper does not specify a stratification strategy.

For MHQA with 1250 total synthetic examples (the main experimental setting), this means Slice 0 and Slice 1 each contain approximately 625 examples. For TQA with 8000 examples per slice, each slice contains exactly 4000 examples.

The purpose of this split is to create a clean separation between the data used to train the curation model and the data that will be curated. If the same data were used for both, the curation model would simply memorize the examples and declare everything answerable, defeating the purpose of filtering.

**Step 2: Fine-tune `LLMSynth` on Slice 0.**

The first half of the synthetic dataset (Slice 0) is used to fine-tune an intermediate model called `LLMSynth`. This is a standard supervised fine-tuning run where the model learns to produce the reasoning chain and answer given the question (and, for TQA, the table). The fine-tuning uses the same hyperparameters and procedure as the final model fine-tuning (discussed in Section 3.4, "Model Fine-tuning").

`LLMSynth` is not a different architecture or a separate model family—it is simply the base LLM fine-tuned on Slice 0. For MHQA experiments, `LLMSynth = Llama2 70B-Chat` fine-tuned on ~625 synthetic examples. For TQA experiments, `LLMSynth = Starchat-beta` fine-tuned on 4000 synthetic examples.

The critical property that makes this work is that `LLMSynth` has been partially trained on the task—it has learned something about how to decompose multi-hop questions or write SQL—but it has not seen Slice 1. This partial competence is exactly what's needed for filtering: the model can assess whether an unseen example is answerable, but it's not so capable that every example passes the filter.

**Step 3: Use `LLMSynth` to curate Slice 1.**

`LLMSynth` is applied to every example in Slice 1. The curation consists of two sub-operations, applied differently depending on the task:

**Filtering** (applied to both MHQA and TQA): For each example in Slice 1, `LLMSynth` is given the question (and, for TQA, the table) and asked to produce the answer. It is allowed $k = 3$ independent attempts (with sampling, so each attempt may produce a different output). If **none** of the $k$ attempts produce the correct answer (matching the synthetically generated answer $A$), the example is **discarded** from the final curated dataset.

The $k = 3$ threshold is an empirical choice. Using $k = 1$ would be too strict—the model might fail once due to sampling variance even for a perfectly good example. Using $k$ much larger would make the filter too permissive, passing borderline examples that the model can occasionally answer but can't learn robustly from. Three attempts provides a reasonable balance: the model gets multiple chances, but consistently unanswerable examples are caught.

The rejection rates reported in Section 3.2 are striking:

- **MHQA**: approximately 13% of originally generated examples are removed during curation. This relatively low rejection rate suggests that the grounding mechanism (Wikipedia-based seed extraction) produces mostly coherent, answerable questions.
- **TQA**: only 27% of original examples are kept—meaning **73% are rejected**. This is a much higher rejection rate, reflecting the difficulty of the TQA task: generating valid, executable, semantically meaningful SQL from natural language seeds is significantly harder than generating multi-hop questions from Wikipedia entities. Many generated SQL statements may be syntactically valid (they passed the execution check at generation time) but semantically disconnected from the natural language question, or they may represent queries that are too complex for `LLMSynth` to learn from.

**Imputation** (applied only to MHQA): This is an additional curation step that addresses a specific quality issue in MHQA synthetic data: the merged multi-hop question $Q$ can be **structurally awkward** because the merging process mechanically substitutes $Q_1$ into $Q_2$, sometimes producing run-on sentences or redundant phrasing.

The imputation procedure works as follows (Section 3.2.2):

1. **Discard the original $Q_1$** from the synthetic example.
2. **Provide `LLMSynth`** with the remaining components: the multi-hop question $Q$, the second sub-question $Q_2$, the entity $E$, and the first source document $D_1$.
3. **Ask `LLMSynth` to reconstruct** $Q_1$—that is, to generate a sub-question $Q_1'$ such that when reasoned from $D_1$, it yields $E$ as the answer.
4. **Assemble a new multi-hop question** $Q'$ by merging $Q_1'$ with $Q_2$ using the same merging procedure as the original generation.
5. **Check consistency**: if the answer to the new multi-hop question $Q'$ (obtained by solving the reasoning chain $Q_1' \to E \to Q_2$) matches the original answer $A$, the example is kept with $Q'$ and $Q_1'$ replacing the originals. If the answers don't match, the example is discarded.

The intuition behind imputation is that `LLMSynth`, having been fine-tuned on Slice 0, has learned a more natural decomposition style than the zero-shot prompted generation. By reconstructing $Q_1$, it effectively "smoothes out" awkward phrasings introduced by the original merge. The consistency check ensures that this smoothing doesn't change the answer, preserving the factual correctness of the example.

The paper quantifies the improvement in naturalness using perplexity (Appendix C.4, Table 9). For grounded synthetic data (Wikipedia-based), the average perplexity of multi-hop questions drops from 24.7 before imputation to 13.6 after imputation—a 45% reduction, indicating substantially more natural phrasing. For comparison, ungrounded synthetic data (generated from made-up topics) starts with lower perplexity (15.51) because the questions are more formulaic, but still drops to 8.33 after imputation.

Figure 6 provides a concrete example of the improvement:

- **Before imputation**: $Q =$ "What pet did the poet and father of mathematician Ada Lovelace had when he was a student at Trinity out of resentment for rules forbidding pet dogs like his beloved Boatswain?"
- **After imputation**: $Q' =$ "What pet did the poet and father of mathematician Ada Lovelace had when he was a student at Trinity?"

The imputed version removes the redundant trailing clause ("out of resentment for rules forbidding pet dogs like his beloved Boatswain") that was carried over from $Q_1$ into the merged question, making $Q$ cleaner and more natural.

Imputation is **not applied to TQA** (Section 3.2.2: "TQA the curation process consists only of the filtering step"). This is because TQA examples don't have the same structural unnaturalness problem—the natural language questions are generated by translating SQL, which tends to produce clean, direct questions. There is no multi-part merging that introduces awkwardness.

**Why this slice-based self-curation works.** The core insight is that `LLMSynth` acts as a **proxy for learnability**. An example that even a partially-trained model cannot answer in three tries is likely one of the following:

- Too hard: the reasoning chain exceeds what the model architecture and training data can teach at this scale.
- Inconsistent: the question, reasoning chain, and answer don't form a logically sound triple, and the model's confusion reflects this incoherence.
- Noisy: the generation process introduced artifacts (awkward phrasing in MHQA, semantically misaligned SQL in TQA) that make the example difficult to parse.

By filtering these examples out, the curated dataset contains only examples that are at an appropriate difficulty level for the model's capacity—challenging enough to drive learning, but not so hard or noisy that they waste training signal. This is a form of **curriculum learning where the curriculum is defined by the student model's own competence after partial training**.

---

#### Model Fine-tuning

The final stage of Source2Synth is straightforward supervised fine-tuning of the target LLM on the curated synthetic dataset (Section 3.3). The model is trained to produce both the reasoning chain and the final answer, conditioned on the question.

For **MHQA**, the training format (illustrated in Figure 4, left) teaches the model to:

1. Decompose the multi-hop question $Q$ into sub-questions $Q_1$ and $Q_2$.
2. Answer $Q_1$ using knowledge of the entity $E$ (which is implicitly learned from the training examples, not provided as input).
3. Substitute the answer $A_1$ into $Q_2$.
4. Answer $Q_2$ to obtain the final answer $A$.

The model is trained on 1250 curated synthetic examples (bridge questions only) for the `LLMCurated` configuration, or 1250 curated synthetic examples plus 500 HPQA training examples (including comparison questions) for the `LLMCurated-datamix` configuration (Section 4.1, "Model").

For **TQA**, the training format (illustrated in Figure 4, right) teaches the model to:

1. Inspect the table schema and content.
2. Compose a SQL query that answers the natural language question.
3. Execute the SQL (conceptually—the model outputs the SQL, and the answer is obtained by executing it externally or the model may output both SQL and answer).

The model is fine-tuned on the curated synthetic TQA examples. The paper uses the Starchat-beta language model (16B parameters) as the base LLM, with fine-tuning hyperparameters: batch size 32, 100 training steps, learning rate $1 \times 10^{-4}$, and linear warm-up (Section 4.2, "Model").

The resulting model `LLMCurated` is the final output of the pipeline. It has been trained exclusively on synthetic data (or a mix with a small amount of real data) and demonstrates competence on the target task that significantly exceeds prompting-based baselines and models fine-tuned on uncurated synthetic data.

---

#### Design Choices and Their Justifications

**Why use a seed rather than generating freely?** Without a seed, the generation process would be unconstrained—the LLM would generate questions, reasoning chains, and answers independently, with no guarantee that they are logically connected. The seed anchors every component to a single ground-truth element from the source, enforcing consistency. This is particularly important for multi-hop reasoning, where the "hop" must be a real entity that genuinely links the two sub-questions.

**Why generate bridge questions only for MHQA?** Bridge questions map naturally onto the seed-based decomposition: the seed is the bridge. Comparison questions require a different structure (comparing the same attribute across two entities) that doesn't fit the single-entity-hop pattern. Rather than overcomplicate the generation process, the paper uses a small number of real comparison questions from HPQA to supplement the synthetic bridge data, and shows that the model generalizes well to comparison questions at test time anyway (Table 4: +16.5% absolute improvement on hard comparison questions).

**Why split the data into two slices?** If the same data were used for both training the curation model and filtering, the curation model would simply memorize the examples and pass everything through the filter. The two-slice protocol ensures that the curation model evaluates out-of-sample data, making the answerability check a genuine test of learnability rather than a memory check.

**Why $k = 3$ attempts?** This provides a reasonable balance between over-strictness ($k = 1$ would reject examples that the model can answer but fails once due to sampling variance) and over-permissiveness ($k$ large would pass borderline examples that the model occasionally guesses but hasn't genuinely learned). The paper doesn't report ablation studies on $k$, so this is an empirical choice whose sensitivity is unknown.

**Why imputation only for MHQA?** TQA questions are generated by translating SQL to natural language, a process that tends to produce clean, semantically precise questions without the structural awkwardness that comes from mechanically merging two questions in MHQA. The imputation step addresses a specific failure mode of the MHQA merge process that doesn't exist in TQA.

**Why the 73% rejection rate for TQA?** This high rejection rate reflects the inherent difficulty of the TQA task. Generating a SQL query that is (a) syntactically valid, (b) semantically aligned with a natural language fact, and (c) answerable by a partially-trained model is challenging. Many generated SQL queries may be valid but semantically mismatched to the seed fact, or too complex for `LLMSynth` to reproduce. The high rejection rate underscores the importance of curation—without it, the model would be trained on 73% low-quality examples that could degrade performance.

**Why use Starchat-beta for TQA rather than Llama2?** Starchat-beta is a 16B-parameter model pre-trained on a large code corpus including SQL statements and fine-tuned as a coding assistant. This makes it a more natural base model for the TQA task, which requires SQL generation. The paper leverages this domain-specific pretraining rather than starting from a general-purpose chat model.

## 4. Key Insights and Innovations

### Innovation 1: Grounding Synthetic Data Generation in Real-World Sources as a General-Purpose Quality Control Mechanism

The paper's most conceptually distinctive contribution is not any single algorithmic step but rather the **meta-principle that anchoring synthetic data generation to real-world data sources simultaneously solves multiple quality problems that prior work addressed with separate, often fragile, post-hoc fixes.** This is a reframing of the synthetic data problem: instead of generating freely and then filtering aggressively, build the constraint into the generation process itself by tying every component of the synthetic example to a single, verifiable ground-truth element extracted from the source.

Prior to Source2Synth, the dominant paradigm for synthetic instruction-tuning data—exemplified by Self-Instruct (Wang et al., 2023a) and Unnatural Instructions (Honovich et al., 2023)—was to prompt an LLM to generate diverse tasks from a seed set of examples or topics, then apply model-based or heuristic filters to remove low-quality outputs. This approach treats generation and quality control as separable stages: first produce a large volume of potentially noisy data, then clean it up. The problem, as the paper documents, is that general-purpose generation produces very few multi-hop examples that are structurally coherent: "only a small fraction of synthetic instruction-tuning samples are multi-hop, and they can exhibit poor quality" (Section 2, citing Chen et al., 2024). The generation process has no built-in mechanism to ensure that the sub-questions, reasoning chain, and final answer form a logically consistent triple, because it has no external reference point to constrain consistency.

Source2Synth inverts this relationship: **the data source is the constraint, and the generation process is structured around a task-specific seed extracted from that source.** The seed is not a loose topic or keyword for the LLM to riff on—it is a specific, verifiable element (an entity in Wikipedia text, a fact derived from a table) that must be the answer to one sub-question and the subject of another. This transforms the generation problem from "produce a question that sounds complex" to "produce a question whose answer chain is guaranteed to trace back to this specific piece of the source." The quality guarantee comes from the structure of the generation, not from after-the-fact filtering.

What makes this a fundamental rather than incremental advance is that it **redefines the role of the LLM in synthetic data generation.** In prior work, the LLM is both the source of content and the arbiter of quality—a self-referential loop that can amplify errors. In Source2Synth, the LLM's role is to articulate the reasoning chain that connects the seed to the answer, but the factual content is provided by the source. The LLM can still produce awkward phrasing or logically flawed merges, but it cannot invent facts or fabricate hops, because the seed—extracted deterministically from the source—constrains the space of valid outputs. This is why the MHQA rejection rate (13%) is so much lower than TQA (73%): Wikipedia articles provide clean, explicit entities that tightly constrain the generation, while tables require the LLM to generate both the seed fact and the SQL, leaving more room for misalignment.

This insight is not merely about benchmark performance—it is a **design principle for synthetic data pipelines in any domain where structured or semi-structured real data exists.** The paper explicitly gestures toward medical and legal applications (Section 6), where unstructured text corpora (clinical notes, legal opinions) contain entities and relationships that could serve as seeds for generating QA pairs. The principle generalizes: identify a grounding element in your source data, use it as the invariant spine of your synthetic example, and let the LLM fill in the expression, not the facts.

The evidence for this innovation's significance is not a single ablation but the **architecture of the entire paper**: the fact that the identical seed-conditioned generation pipeline works for two fundamentally different data types (documents and tables) and two fundamentally different reasoning structures (entity-hop decomposition and SQL composition) demonstrates that the grounding principle, not the specific instantiation, is the contribution. The contrast with the ungrounded baseline (Appendix C.3, Tables 7–8) drives this home: when Source2Synth is applied to made-up topics instead of Wikipedia, accuracy drops by ~7% absolute, even though the rest of the pipeline (merge, curation, fine-tuning) is identical. The grounding is what makes the difference.

---

### Innovation 2: Slice-Based Self-Curation as a Learnability Filter—The Model Judges Its Own Curriculum

The second conceptual innovation is the **slice training protocol for dataset curation**, which reframes quality filtering from "is this example correct?" to "can a partially-trained version of the target model learn from this example?" This is a diagnostic insight, not just a method: it reveals that correctness and learnability are distinct properties, and that learnability—not correctness—is what matters for training data quality.

Prior approaches to synthetic data filtering fall into two categories. The first uses **external verifiers or heuristics**: human annotators check a subset of examples (Liu et al., 2022), a separate trained classifier scores quality (Schick and Schütze, 2021), or rule-based checks filter out obviously invalid outputs (e.g., format errors, length thresholds). The second uses **the generating model itself as a verifier**: the same LLM that produced the data assesses whether it's correct, often through self-consistency checks or confidence scoring. Both approaches share a common assumption: that the filtering criterion should be some notion of objective correctness or well-formedness.

Source2Synth's curation protocol challenges this assumption. It uses `LLMSynth`—a model fine-tuned on Slice 0 of the synthetic data—to test each example in Slice 1 by attempting to answer it $k = 3$ times. If the model cannot produce the correct answer, the example is rejected. The criterion is **not whether the example is correct in some absolute sense** (it is—the answer was verified against the source during generation), but whether a model with partial task competence can reproduce that answer. This inverts the usual relationship between model and data: **the data is training the model, but the model is also curating the data.**

Why is this a fundamental shift rather than just a clever filtering trick? Because it introduces a **feedback loop between training progress and data selection that is normally absent in static dataset curation.** Standard approaches generate a dataset once, filter it once (using fixed criteria), and train on it. In Source2Synth, the curation criterion depends on the model's current capability after partial training on Slice 0. If Slice 0 were larger or the fine-tuning were more aggressive, `LLMSynth` would be more capable, and the filtering criterion would shift—more examples would pass because the model can answer them. The dataset is not fixed; it is a function of the training process itself.

This connects conceptually to **curriculum learning**, where training examples are presented in order of increasing difficulty, and to **active learning**, where the model selects which examples to learn from next. But Source2Synth's implementation is distinctively self-contained: it requires no human difficulty annotations, no external difficulty estimator, and no oracle access to ground truth beyond what the synthetic data itself provides. The "difficulty" of an example is operationally defined as "can `LLMSynth` answer it after training on Slice 0," which is a direct measure of learnability from similar data.

The value of this innovation is most clearly visible in the **TQA results**, where the rejection rate is 73% (Section 3.2.1). This means that only about one-quarter of the synthetically generated TQA examples pass the learnability filter. If the paper had used a standard correctness-based filter (e.g., checking that the SQL executes and returns a non-empty result), far more examples would have passed—they were already filtered for executability at generation time. The additional 73% rejection represents examples that are technically valid but too complex, noisy, or semantically misaligned for a partially-trained model to learn from. Training on these examples would waste capacity and potentially degrade performance, as the `LLMSynth` vs. `LLMCurated` comparison demonstrates (Table 2: 23.86% EM → 34.50% EM from adding curation).

A subtle implication: this curation protocol **provides a diagnostic signal about the generation process itself.** The 73% rejection rate for TQA vs. 13% for MHQA tells us that the TQA generation process produces a much higher fraction of examples that are not learnable from a reasonable amount of synthetic data. This is actionable: improving TQA data generation (perhaps through better seed generation or more constrained SQL prompts) could reduce this rejection rate and increase the yield of the pipeline. The curation step thus serves both as a quality filter and as a **measurement tool** for diagnosing weaknesses in the generation process.

The evidence for this innovation's significance is the **consistent gap between `LLMSynth` and `LLMCurated`** across both tasks and all data sizes (Figure 5, Tables 1–2). Curation isn't a minor refinement—it closes a substantial fraction of the gap between uncurated synthetic data and the full potential of the method. In MHQA with 1250 synthetic examples, curation adds approximately 7–8 percentage points of accuracy (Table 1: 57.46% → 65.23% for the datamix, 52.31% → 64.07% for synthetic-only). In TQA, the gain is even larger proportionally: 23.86% → 34.50% EM, a ~45% relative improvement.

---

### Innovation 3: Empirical Proof That Bridge-Only Synthetic Training Transfers to Comparison Questions—Capability Generalization Without Task-Specific Data

The third innovation is primarily a **diagnostic empirical finding** with significant practical implications: a model fine-tuned exclusively on synthetic bridge-type multi-hop questions generalizes to comparison-type questions without any comparison-specific training data, and does so particularly strongly on hard comparison questions (+16.5% absolute improvement over the base LLM, Table 4). This is not an engineering achievement but a **revealed property of the learned capability**—it tells us something important about what the model is actually learning from the synthetic data.

The standard assumption in task-specific fine-tuning is that the model learns the distribution of the training data. If you train on bridge questions, you might expect the model to become good at bridge questions but not at comparison questions, which have a fundamentally different structure (comparing the same attribute across two entities rather than chaining through a linking entity). The dominant approach to handling this would be to generate synthetic data for both question types, which is what the paper explores for Llama4 (Appendix C.2, Table 6). But the main results in Table 1 show that even without comparison-specific synthetic data, `LLMCurated-datamix` (which uses synthetic bridge questions plus only 500 real HPQA comparison questions) dramatically outperforms the base model on comparison questions across all difficulty levels.

What does this tell us? The model is not merely memorizing bridge-question patterns. It is learning something more general—likely **the skill of question decomposition and structured reasoning that transfers across question types.** The bridge questions teach the model to break a complex question into sub-questions, identify the entity that links them, and solve sequentially. Comparison questions require a different decomposition (identifying two entities, extracting a common attribute for each, comparing the values), but the underlying meta-skill—"this complex question can be broken into simpler sub-questions that I can answer from documents"—transfers. The training data teaches the decomposition *habit*, and the model's pre-existing language understanding handles the specifics of how to decompose different question structures.

This finding is significant beyond HotpotQA because it suggests that synthetic data for complex reasoning tasks doesn't need to exhaustively cover every possible question structure. **Teaching the meta-skill of structured decomposition may be sufficient**—the model can generalize the decomposition pattern to structurally different question types without explicit training on them. This has practical implications for synthetic data generation: you may not need to engineer generation pipelines for every task variant; generating high-quality examples for one representative variant may unlock the broader capability.

The paper does not make this theoretical claim explicitly—it presents the comparison-question results as an empirical observation. But the observation is striking enough to constitute a conceptual contribution: it changes how we should think about what synthetic data for reasoning tasks is actually teaching the model. The fine-tuning is not just adding knowledge (the model already knew the facts from pretraining); it's instilling a **behavioral pattern** of structured decomposition that generalizes across question types.

The evidence is in Table 4. On hard comparison questions, the base Llama2 70B-Chat achieves 66.6% accuracy. Fine-tuning on HPQA data only (which includes both bridge and comparison questions) raises this to 74.5%—an 7.9-point gain from seeing real comparison examples. Source2Synth's `LLMCurated` (synthetic bridge questions only, no real comparison data at all) achieves 79.1%—a 12.5-point gain over the base model, and 4.6 points *better* than the model trained on real comparison data. The synthetic bridge data teaches the decomposition skill more effectively than the real mixed-type data does. This is a strong signal that the synthetic data's quality and structured format (explicit sub-questions, entity hops, reasoning chains) is doing something that the real HPQA data's format may not capture as cleanly.

---

### Innovation 4: Exposing the Correctness-vs-Learnability Gap Through Differential Rejection Rates Across Tasks

The fourth and most subtle innovation is a **diagnostic finding** that emerges from comparing the curation behavior across MHQA and TQA: the 13% vs. 73% rejection rates reveal that **correctness and learnability are fundamentally different properties of synthetic data, and the gap between them varies dramatically depending on the generation mechanism.** This is not a method contribution but a **measurement insight** that provides a new lens for evaluating synthetic data pipelines.

In both tasks, the generated examples that enter curation have already passed a correctness check. For MHQA, the answers are verified against Wikipedia content. For TQA, the SQL statements have been executed and produced results—they are definitionally correct with respect to the table. Yet when `LLMSynth` attempts to answer these correct examples, it fails on 13% of MHQA examples and 73% of TQA examples. These failures are not about factual correctness; they are about whether the reasoning chain is learnable from the limited training signal provided by Slice 0.

What does this gap measure? It measures the **pedagogical quality** of the synthetic data—how well the example's structure, phrasing, and difficulty level match what the model can absorb from a small amount of similar data. An example can be perfectly correct but pedagogically poor: the SQL might be too complex for the model to learn from 4000 examples, the multi-hop question might be phrased in a way that the decomposition pattern isn't clear, or the reasoning chain might contain steps that the model cannot reliably reproduce.

The wide gap between TQA and MHQA rejection rates is itself a finding. It tells us that the MHQA generation pipeline (entity extraction from Wikipedia → question generation → merge) produces data with much higher pedagogical quality than the TQA pipeline (seed fact generation → SQL generation → question translation). The TQA pipeline's high rejection rate isn't a failure of Source2Synth—it's a **measurement of the generation process's signal-to-noise ratio** that would be invisible if you only looked at correctness metrics (SQL executability, answer match with execution result). The paper doesn't explicitly frame it this way, but the rejection rates serve as a diagnostic tool for comparing generation strategies, and the differential between tasks points to specific weaknesses in the TQA generation process (likely the zero-shot seed fact generation, which is the least constrained step in the pipeline).

This insight has methodological implications beyond this paper. It suggests that synthetic data research should report not just downstream task performance but also **curation rejection rates as a measure of generation quality.** A generation pipeline that produces 90% learnable examples (like MHQA) is fundamentally better-engineered than one that produces 27% (like TQA), even if both can achieve strong final performance after curation. The rejection rate quantifies how much of the generation effort is wasted, which matters for compute efficiency and scalability. It also provides a target for improvement: reducing the TQA rejection rate from 73% to, say, 50% would double the pipeline's yield without changing the downstream model at all.

The evidence is the rejection rate data itself (Section 3.2.1) combined with the performance gaps in Tables 1 and 2. But the innovation here is not the result—it's the **framing of rejection rate as a first-class evaluation metric for synthetic data pipelines**, distinct from correctness and downstream accuracy. This is a conceptual contribution that, if adopted, would change how synthetic data papers report and compare their generation methods.

## 5. Experimental Analysis

### Evaluation Methodology

- **Dataset.** The paper evaluates on two benchmarks. For MHQA, the HotpotQA (HPQA) benchmark (Yang et al., 2018) is used, which contains 113,000 multi-hop question-answer pairs based on Wikipedia, split into train, test, and validation sets. The paper uses the FullWiki setup on the 7,405-example test set, evenly split between bridge and comparison questions. For TQA, the WikiSQL benchmark (Zhong et al., 2017) is used, consisting of 80,654 hand-annotated natural language questions, SQL queries, and tables from 24,241 Wikipedia-derived tables, with a validation split of 7,857 examples after removing non-executable SQL tables (Appendix D).

- **Base model(s).** For MHQA experiments, the paper uses Llama2 70B-Chat as the base LLM, a 70-billion-parameter instruction-tuned chat model. For TQA experiments, the paper uses Starchat-beta (Li et al., 2023b), a 16-billion-parameter instruction-tuned LLM pre-trained on a large code corpus containing SQL statements and fine-tuned as a coding assistant. The Starchat-beta model was chosen specifically for the TQA task because its code-focused pretraining makes it a more natural base for SQL generation than a general-purpose chat model. For cross-model experiments in Appendix C.2, Llama3 8B-instruct and Llama4 17Bx16E (Dubey et al., 2024) are also fine-tuned using Source2Synth data generated by Llama2 70B-Chat, testing whether the synthetic data transfers across model families.

- **Metrics.** For MHQA, performance is measured using the **soft exact match (soft-EM)** metric, which uses string comparison and scores 1 if the generated output contains the golden answer and 0 otherwise. For TQA, two metrics are reported: **exact match (EM)**, which requires the generated answer to exactly equal the golden answer string, and **soft-EM**, which only requires the golden answer to be contained in the output. The EM metric is stricter and better reflects the precision needed for SQL-based tool use, while soft-EM captures partial correctness.

- **Baselines.** The paper compares against several distinct categories of baselines:

  *For MHQA (Table 1)*:
  - **Instruction-tuned LLMs (zero-shot and 3-shot CoT prompting):** Llama2 70B-Chat and Claude3.5 Sonnet (Anthropic, 2024) prompted with task instructions and optionally 3-shot chain-of-thought examples (Figure 14). These measure what off-the-shelf models achieve without any fine-tuning on HPQA data.
  - **Fine-tuned LLM (HPQA only):** Llama2 70B-Chat fine-tuned on 500 examples from the HPQA training split. This establishes the performance floor achievable with a small amount of real annotated data.
  - **LLMSynth (synthetic data only):** Llama2 70B-Chat fine-tuned on 1250 synthetic examples from Slice 0 only—no curation, no real HPQA data. This isolates the value of the generation step without curation.
  - **LLMSynth-datamix:** Same as above but with 500 HPQA examples added to the synthetic data. This establishes the pre-curation datamix baseline.
  - **LLMCurated:** Llama2 70B-Chat fine-tuned on the full Source2Synth pipeline output (1250 curated synthetic examples, bridge questions only). This isolates the synthetic-only performance ceiling.
  - **LLMCurated-datamix:** The full configuration: 1250 curated synthetic bridge questions plus 500 HPQA examples (including comparison questions). This is the paper's primary model.

  *For TQA (Table 2)*:
  - **Starchat-beta (one-shot no context QA):** Prompted with a one-shot question-answer example but no table context (Figure 8). This measures the base model's ability to answer from parametric knowledge alone.
  - **Starchat-beta (zero-shot table QA):** Given the table and the question in a zero-shot prompt with task instruction (Figure 7). Tests whether the model can ingest structured data without examples.
  - **Starchat-beta (one-shot table QA):** A one-shot example including the table for the example question, plus the actual question to answer (Figure 9). Tests whether a single example improves structured data ingestion.
  - **Starchat-beta (one-shot table+SQL tool QA):** The strongest prompting baseline: a one-shot example containing both the table and an SQL query, with instructions suggesting the model can leverage SQL (Figure 10). The predicted SQL is executed to obtain the answer.
  - **LLMSynth (synthetic data only):** Starchat-beta fine-tuned on uncurated Source2Synth synthetic data (Slice 0 only).

- **Generation budget / compute accounting.** The paper does not use a standardized compute budget metric (like FLOPs or number of generations) for comparing methods, which is a notable absence given that many contemporary test-time compute papers do. Instead, comparisons are based on the **amount of training data**: 1250 synthetic examples for MHQA, 8000 synthetic examples per slice for TQA. The paper explicitly states that Source2Synth "improves the dataset quality by discarding low-quality generations based on their answerability" (Section 1), framing the contribution as data efficiency rather than compute efficiency. The computation cost of generating the synthetic data itself (LLM inference for seed extraction, question generation, merging, SQL generation) is not accounted for or compared to alternatives. This is a recognized limitation—the paper provides no FLOPs comparison between generating synthetic data via Source2Synth vs. collecting human annotations vs. using alternative synthetic data methods.

- **Cross-validation / statistical protocol.** The paper implements a specific contamination-check protocol rather than formal cross-validation. For MHQA, each synthetic question's entity (seed) is checked against all questions in the HPQA test set; if the entity matches, the questions are compared for identity. The paper reports that zero synthetic examples overlap with the HPQA test set (Section 4.1). For TQA, since tables are sourced from the WikiSQL training split and evaluation uses the test split, there is no overlap by construction (Section 4.2). The paper does not report multiple training runs with different random seeds, confidence intervals on accuracy numbers, or statistical significance tests. The small test set sizes (500 questions for MHQA per the HPQA test split specification, 7,857 TQA validation examples) mean that accuracy differences of 1-2 percentage points are unlikely to be statistically reliable absent variance reporting, though the larger gains (5-10+ points) are likely robust.

### Main Quantitative Results

#### Multi-Hop Question Answering (MHQA) Results

The headline result for MHQA is that **Source2Synth's `LLMCurated-datamix` achieves 65.23% accuracy (zero-shot) and 66.05% (3-shot CoT) on the HotpotQA FullWiki test set, representing a 24.78 percentage point improvement over the base Llama2 70B-Chat zero-shot (40.45%) and a 22.57 percentage point improvement over the 3-shot prompted version (44.13%)—which is an approximately 50.7% relative improvement over the strongest base model configuration.** All fine-tuned models are reported in Table 1.

Breaking down the results from Table 1:

The **base instruction-tuned LLMs** establish the lower bound. Llama2 70B-Chat achieves 40.45% zero-shot and 44.13% with a 3-shot CoT prompt. Claude 3.5 Sonnet, a more capable model, reaches 50.3% zero-shot and 53.4% 3-shot. The 3-shot CoT prompt provides a consistent ~3-4 percentage point boost across models by in-context demonstrating the decomposition pattern.

**Fine-tuning on real HPQA data** (500 examples) lifts Llama2 70B-Chat to 53.22% zero-shot and 58.40% 3-shot—an improvement of 12.77 and 14.27 percentage points respectively. This is the baseline for what a small amount of real annotated data can achieve. Notably, this modest fine-tuning already surpasses Claude 3.5 Sonnet's zero-shot performance (53.22% vs. 50.3%), demonstrating that even limited task-specific training can outperform stronger base models.

**Uncurated synthetic data (`LLMSynth-datamix`)** with 1250 synthetic examples plus 500 HPQA examples reaches 57.46% zero-shot and 62.73% 3-shot—modestly outperforming HPQA-only fine-tuning (+4.24 and +4.33 points). This shows that even uncurated synthetic data provides some signal beyond what the real HPQA examples alone offer.

**The curation step is essential.** `LLMCurated-datamix` (curated synthetic + HPQA) achieves 65.23% zero-shot and 66.05% 3-shot—a gain of 7.77 and 3.32 points over the uncurated `LLMSynth-datamix`. The gap is larger in the zero-shot setting (7.77 points) than in the 3-shot setting (3.32 points), suggesting that curation primarily improves the model's standalone reasoning capability rather than its ability to follow in-context examples.

**Synthetic-only performance is competitive with mixed data.** `LLMCurated` (curated synthetic bridge questions, zero real HPQA data) achieves 64.07% zero-shot and 64.68% 3-shot—only 1.13 percentage points behind the `LLMCurated-datamix` in the zero-shot condition (Table 1: 65.23% vs. 64.07%). This is a striking result: a model trained exclusively on synthetic data nearly matches one trained on synthetic plus real data. The gap is slightly larger in the 3-shot setting (1.37 points: 66.05% vs. 64.68%).

The **scaling analysis** in Figure 5 shows how performance changes when adding more synthetic data to a fixed base of 500 HPQA examples. The x-axis sweeps across 500, 750, and 1250 synthetic examples for both `LLMSynth-datamix` and `LLMCurated-datamix`. Three patterns emerge:

1. **Curation consistently outperforms no curation at every data size.** At 500 synthetic examples: `LLMCurated-datamix` (~64% zero-shot) vs. `LLMSynth-datamix` (~56%). At 750: ~67% vs. ~58%. At 1250: ~68% vs. ~60%. The absolute gap widens slightly with more data (from ~8 points at 500 to ~8.5 points at 1250).

2. **Both curves are monotonically increasing**, meaning adding more synthetic data (even uncurated) continues to help, though with diminishing returns. `LLMCurated-datamix` goes from ~64% (500 synthetic) to ~65-66% (750) to ~68% (1250). The slope is positive but shallow, suggesting that even with curation, data beyond 750 examples provides modest additional gains.

3. **The 3-shot variants follow the same pattern** at slightly higher absolute levels, confirming that the curation benefit is not an artifact of a particular evaluation protocol.

The rejection rates during curation scale with data size: 7% for 500 examples, 8% for 750, and 11% for 1250 (Section 5.1, scaling performance text). More data means more examples cross the threshold into unanswerability, but proportionally the rejection rate remains modest (11% for 1250, consistent with the overall ~13% reported in Section 3.2.1).

---

**The difficulty- and type-stratified analysis** (Table 4) drills into what Source2Synth actually improves. The table subdivides performance by question type (bridge vs. comparison) and difficulty level (easy, medium, hard), using labels provided by the HPQA dataset. The base Llama2 70B-Chat, the HPQA-only fine-tuned model, and two Source2Synth configurations (`LLMCurated` and `LLMCurated-datamix`) are compared.

For **bridge questions** (the type Source2Synth generates):
- **Hard:** Base model 14.5% → HPQA fine-tuned 20.1% → `LLMCurated` 27.6% → `LLMCurated-datamix` 31.3%. Absolute gain over base: +13.1% (from synthetic-only) to +16.8% (with datamix).
- **Medium:** Base 27.2% → HPQA 29.8% → `LLMCurated` 32.3% → `LLMCurated-datamix` 35.6%. Gains are more modest (+5.1% to +8.4%).
- **Easy:** Base 30.1% → HPQA 34.3% → `LLMCurated` 36.2% → `LLMCurated-datamix` 39.7%. Gains are smaller still (+6.1% to +9.6%).

The pattern is clear: Source2Synth provides the largest absolute gains on hard bridge questions, where the base model's performance is worst. This suggests the synthetic data is teaching decomposition skills that the base model lacks precisely for the most challenging examples.

For **comparison questions** (NOT generated by Source2Synth):
- **Hard:** Base 66.6% → HPQA 74.5% → `LLMCurated` 79.1% → `LLMCurated-datamix` 83.1%. Absolute gain over base: +12.5% (synthetic-only, despite zero comparison training data) to +16.5% (datamix).
- **Medium:** Base 71.3% → HPQA 78.3% → `LLMCurated` 82.3% → `LLMCurated-datamix` 85.7%. Gains: +11.0% to +14.4%.
- **Easy:** Base 73.2% → HPQA 82.1% → `LLMCurated` 88.0% → `LLMCurated-datamix` 87.8%. Gains: +14.8% to +14.6%.

The most surprising result here is that `LLMCurated` (synthetic bridge questions only, zero comparison training data) outperforms the HPQA-only fine-tuned model on comparison questions across all three difficulty levels. On hard comparison questions, the synthetic-only model beats the real-data model by 4.6 percentage points (79.1% vs. 74.5%). The synthetic bridge data generalizes to comparison questions better than the real mixed-type HPQA data does.

---

**Fine-tuning smaller models with Source2Synth data generated by a larger model** (Appendix C.2) tests cross-model transfer. Using Llama2 70B-Chat as the data generator and Llama3 8B-instruct as the fine-tuning target (Table 5), `LLMCurated` reaches 71.13% zero-shot accuracy—a 23.06 percentage point absolute improvement over the base Llama3 8B-instruct (57.8%). The gap between `LLMSynth` (64.46%) and `LLMCurated` (71.13%) confirms that curation matters even when the data generator differs from the fine-tuning model. For a much stronger base model, Llama4 17Bx16E (Table 6), `LLMCurated` reaches 67.9%—a 36.90 percentage point improvement over the base (49.6%). The Llama4 experiment used both bridge and comparison synthetic questions (unlike the Llama3 experiment, which used only bridge questions), making it the closest ablation to "what if we generated both types?"

The cross-model results are important because they demonstrate that Source2Synth's data is not overfitted to the generating model's idiosyncrasies—it transfers effectively to both smaller and larger models from different families, suggesting the grounding mechanism produces data with genuine pedagogical value rather than artifacts of the generator's output distribution.

---

#### Tabular Question Answering (TQA) Results

The headline result for TQA is that **Source2Synth's `LLMCurated` achieves 34.50% exact match (EM) and 42.80% soft-EM on the WikiSQL validation set, representing a 25.51 percentage point improvement in soft-EM over the strongest prompting baseline (Starchat-beta one-shot table+SQL tool QA at 34.13% soft-EM) and a 22.20 percentage point improvement in EM over the same baseline (12.30% EM).** Table 2 reports all results.

The **prompting baselines** reveal a clear gradient of difficulty in the TQA task:

- **No table context at all** (one-shot no context QA): 0.25% EM, 16.22% soft-EM. The model cannot answer table-specific questions without the table—the near-zero EM confirms that the questions genuinely require the table data and are not answerable from parametric knowledge.
- **Zero-shot with table** (zero-shot table QA): 1.83% EM, 20.07% soft-EM. Providing the table barely improves EM, despite a modest soft-EM gain (+3.85 points). The model struggles to parse structured data without examples.
- **One-shot with table** (one-shot table QA): 2.03% EM, 31.06% soft-EM. EM remains near floor (~2%), but soft-EM jumps significantly (+11 points), suggesting the model is producing answers that are in the right semantic neighborhood but not precisely correct.
- **One-shot with table and SQL tool use** (one-shot table+SQL tool QA): 12.30% EM, 34.13% soft-EM. Enabling SQL as a tool with a one-shot example of proper SQL usage produces the first substantial EM score, though 12.30% is still low relative to fine-tuned performance. The gap between EM (12.30%) and soft-EM (34.13%) with this baseline is notable: when the model gets the SQL wrong, it often produces answers that are semantically related but factually incorrect.

The **fine-tuned models** dramatically outperform all prompting baselines:

- `LLMSynth` (uncurated synthetic data, Slice 0 only): 23.86% EM, 34.21% soft-EM. Already nearly doubles the EM of the strongest prompting baseline (23.86% vs. 12.30%) and approximately matches its soft-EM (34.21% vs. 34.13%). However, the close soft-EM between prompting and uncurated fine-tuning underscores a subtle point: the prompting baseline's soft-EM was already competitive, suggesting that many answers were close but imprecise, while fine-tuning on uncurated data sacrificed soft-EM coverage in exchange for EM precision gain.
- `LLMCurated` (curated synthetic data): 34.50% EM, 42.80% soft-EM. The curation step adds 10.64 percentage points of EM and 8.59 percentage points of soft-EM over the uncurated `LLMSynth`. This is a proportionally larger curation gain than in MHQA (where curation added ~7-8 points), consistent with the higher rejection rate (73% in TQA vs. 13% in MHQA)—more low-quality data was being filtered, so the curation impact is larger.

The gap between `LLMCurated`'s EM (34.50%) and soft-EM (42.80%) is 8.30 points, smaller than the prompting baseline's gap (21.83 points for one-shot table+SQL). This indicates that the fine-tuned model's errors, when they occur, are less often "close but wrong"—the model is either exactly right or wrong by a larger margin.

The paper does not provide difficulty-stratified analysis or scaling curves for TQA as it does for MHQA. The TQA workload is presented as a single data point rather than a scaling study, which is a notable asymmetry in the experimental depth between the two tasks.

---

**Comparing tasks: rejection rates as a diagnostic.** The 73% TQA rejection rate vs. 13% MHQA rejection rate (Section 3.2.1) provides an implicit comparison of the generation mechanisms' quality. The TQA pipeline produces far more unlearnable examples, which is consistent with its more complex generation chain: the seed fact is generated zero-shot from the table (the weakest link), the SQL is generated from that seed and the table (two steps removed from the source), and the question is generated by translating the SQL back to natural language. Each step introduces potential misalignment. In contrast, MHQA's generation chain is simpler: an entity is extracted deterministically from Wikipedia text, and questions are generated directly from that entity and the surrounding text, with fewer degrees of freedom for semantic drift.

### Ablation Studies and Robustness Checks

**The effect of curation (learnability filtering) vs. no curation:** This is the paper's central ablation, embodied in the `LLMSynth` vs. `LLMCurated` comparison. For MHQA with 1250 synthetic examples plus 500 HPQA examples (Table 1): `LLMSynth-datamix` = 57.46% vs. `LLMCurated-datamix` = 65.23% in the zero-shot setting. The 7.77-point gap confirms that filtering out ~11% of examples (the rejection rate at 1250) substantially improves model quality. For synthetic-only MHQA: `LLMSynth` = 52.31% vs. `LLMCurated` = 64.07%—a 11.76-point gap, even larger than the datamix gap. This suggests that curation is more important when there is no real data to regularize the training. For TQA (Table 2): `LLMSynth` = 23.86% EM vs. `LLMCurated` = 34.50% EM. The 10.64-point gap is the largest absolute curation gain in the paper, consistent with the 73% rejection rate.

**Scaling synthetic data quantity (Figure 5):** Sweeping the number of synthetic examples (500, 750, 1250) added to a fixed base of 500 HPQA examples while comparing curated vs. uncurated. Key findings: (1) Both curves are monotonically increasing—more synthetic data always helps, even uncurated. (2) The slope is steeper for curated data, meaning curation increases the marginal value of each additional synthetic example. (3) The 3-shot CoT variants track the zero-shot variants closely at a ~3-4 point offset, suggesting that in-context examples and fine-tuning provide complementary rather than redundant benefits.

**Grounded vs. ungrounded data generation (Appendix C.3, Tables 7–8):** Starting Source2Synth from a made-up topic list (50 hand-selected words/phrases like "Moon", "Ocean", "Roman Empire") instead of Wikipedia articles. When fine-tuning Llama3 8B-instruct (Table 7), `LLMCurated` with ungrounded data achieves 66.37% vs. 71.13% with grounded data—a 4.76-point loss (or roughly 7.17% of the grounded accuracy). When fine-tuning Llama2 70B-Chat (Table 8), `LLMCurated` with ungrounded data achieves 59.70% vs. 64.07% with grounded data—a 4.37-point loss (6.82% relative). The paper notes that ungrounded questions exhibit "repeating patterns in the structure of the questions" (e.g., formulaic "What/Who [Q1] and/or What [Q2]?" templates), suggesting reduced diversity and potentially overfitting to surface patterns rather than learning decomposition. The perplexity of ungrounded questions (15.51 before imputation, 8.33 after) is substantially lower than grounded questions (24.7 before, 13.6 after), confirming that ungrounded generation produces more predictable, templated phrasing.

**The imputation step (Appendix C.4, Table 9, Figure 6):** Imputation is specific to MHQA and is evaluated through perplexity measurements and a qualitative example. Average perplexity of grounded synthetic questions drops from 24.7 before imputation to 13.6 after—a 45% reduction. For ungrounded synthetic questions, the drop is from 15.51 to 8.33—a 46% reduction. The relative reduction is similar, but the absolute perplexity is consistently lower for ungrounded data (confirming its more formulaic nature). The qualitative example in Figure 6 shows how imputation removes redundant trailing clauses from merged questions, making them more concise and natural. The paper does not provide an ablation comparing `LLMCurated` performance with and without imputation—imputation and filtering are bundled together in the curation stage, so their individual contributions cannot be separated from the reported results. This is a notable missing ablation.

**Cross-architecture transfer of synthetic data (Appendix C.2, Tables 5–6):** The synthetic data generated by Llama2 70B-Chat transfers effectively to Llama3 8B-instruct (71.13% accuracy, outperforming its base by 23.06 points) and to Llama4 17Bx16E (67.9%, +36.90 points over base). The Llama3 result is particularly informative because it used only bridge questions in the synthetic data, yet the fine-tuned model was evaluated on the full HPQA test set (bridge + comparison), confirming that the cross-type generalization observed with Llama2 also holds for Llama3. The Llama4 experiment used both bridge and comparison synthetic questions and achieved strong results, but without an ablation testing bridge-only synthetic data on Llama4, we cannot tell whether generating comparison questions was necessary or whether the bridge-only cross-type generalization would have sufficed.

**Prompt engineering sensitivity (Appendix E, Table 10):** The paper tests different prompt templates for MHQA evaluation, including zero-shot, role-based ("You are a QA-robot…"), 1-shot, 5-shot, and role-based 1-shot. The zero-shot prompt achieves 40.45% accuracy with the base Llama2 70B-Chat—the highest among all tested prompts. Role-based prompts substantially degrade performance (22.34%), as does increasing the number of shots (1-shot: 26.65%, 5-shot: 21.83%). This is a counterintuitive result: more in-context examples hurt rather than help. The paper hypothesizes that the zero-shot prompt's simplicity avoids confusing the model, while the few-shot examples may introduce distributional mismatch with the actual test questions. This sensitivity underscores why the main results use both zero-shot and 3-shot evaluations—the 3-shot CoT prompt in Figure 14 is a specific, carefully crafted template that performs well, but it's not representative of arbitrary few-shot prompting.

**Question type generation scope (bridge-only vs. bridge+comparison, Tables 1, 3, 6):** For Llama2 70B-Chat, generating both bridge and comparison synthetic questions (Table 3) yields `LLMCurated` at 64.5% zero-shot—very close to the bridge-only curated synthetic+500 HPQA datamix (65.23%, Table 1) and essentially identical to bridge-only curated synthetic alone (64.07%, Table 1). This suggests that generating comparison questions provides no additional benefit over generating only bridge questions + including 500 real comparison questions from HPQA, at least for Llama2. For Llama4 17Bx16E (Table 6), the experiment uses both bridge and comparison synthetic data, achieving 67.9%—but without a bridge-only ablation for Llama4, we cannot determine whether the comparison questions contributed.

**Filtering threshold $k = 3$ (not ablated):** The paper does not report ablation studies on the number of answer attempts used for filtering. The choice of $k = 3$ is stated (Section 3.2.1) but never varied. This is a notable gap—the filtering sensitivity to $k$ would reveal whether the benefit is robust or whether a different threshold would yield substantially different curated datasets. Given the high rejection rate for TQA (73% with $k = 3$), the threshold choice may have a particularly large impact on that task.

**Slice split ratio 50/50 (not ablated):** The paper splits the synthetic data into exactly two equal halves for Slice 0 (training `LLMSynth`) and Slice 1 (data to be curated). No ablation tests other split ratios (e.g., 25/75, 75/25, 90/10). The 50/50 split means that half the synthetic data is "consumed" by training `LLMSynth` and is not available for the final `LLMCurated` training (unless Slice 0 is also used, which is not specified). The paper does not clarify whether the final `LLMCurated` is trained only on the curated Slice 1, or on both Slice 0 (uncurated) and curated Slice 1, or on Slice 0 re-curated with a cross-validation scheme. This is a potentially important detail for reproducing the data efficiency claims.

**Fine-tuning hyperparameter sensitivity (not ablated):** The paper reports specific fine-tuning hyperparameters (for TQA: batch size 32, 100 steps, learning rate 0.0001, linear warm-up; Section 4.2) but performs no sensitivity analysis. For MHQA, fine-tuning hyperparameters are not reported at all in the main text (they may be in Appendix H, which is not included in the provided content beyond a mention that Appendix H contains "Hyperparameters"). Without this information, reproducing the exact experimental conditions is not fully possible from the main paper.

**Model scale ablation (not performed):** The paper uses a single model scale per task (70B for MHQA, 16B for TQA) and does not ablate across model sizes for the main experiments. The cross-model transfers in Appendix C.2 test different student model sizes but not different teacher (data generator) model sizes. It remains unknown how Source2Synth's data quality depends on the generator model's scale and capability.

**Multiple table / multiple database scenarios (not tested):** As acknowledged in Appendix A, Source2Synth in its current form uses a single table per query for TQA. No experiments test multi-table scenarios, JOIN queries, or database-level operations. The TQA results are therefore specific to single-table question answering, which is a subset of real-world tabular QA use cases.

### Critical Assessment

**Claim 1 from the executive summary: "Source2Synth achieves a 25.51% improvement for TQA on WikiSQL and a 22.57% improvement for MHQA on HotpotQA over fine-tuned baselines."**

These numbers are drawn from Table 2 (soft-EM: 42.80% for `LLMCurated` vs. 17.29% for the HPQA-only fine-tuned baseline, which is 42.80 - 17.29 = 25.51 percentage points—but note the baseline is the HPQA fine-tuned model, not a prompted baseline as one might infer) and Table 1 (soft-EM zero-shot: 65.23% for `LLMCurated-datamix` vs. 40.45% for base Llama2 70B-Chat, a 24.78 point difference, or vs. the HPQA-only fine-tuned baseline at 53.22%, a 12.01 point difference). The 22.57% number appears to reference the improvement over the base Llama2 70B-Chat in the 3-shot CoT setting: 66.05% vs. 44.13% = 21.92 points (approximately 22.57% when expressed as a relative percentage of the base model's performance: (66.05 - 40.45) / 40.45 = 63.3% relative improvement, not 22.57%).

The exact derivation of "22.57%" is not clearly traceable to a single baseline comparison in Table 1 or the surrounding text. The closest numerical match is the absolute difference between `LLMCurated-datamix` 3-shot (66.05%) and the base Llama2 70B-Chat 3-shot (44.13%), which is 21.92 percentage points. The 22.57% figure may be calculated as (66.05 - 40.45) = 25.6 points divided by or compared to something else, or it may reference a different baseline comparison. This imprecision matters for a headline claim—the reader cannot verify exactly what is being compared without additional specification.

More importantly, the claim conflates two different comparisons: the TQA number compares against a specific baseline (the strongest prompting baseline), while the MHQA number appears to compare against the base model, not against the strongest alternative fine-tuning approach. A reader skimming the abstract might incorrectly conclude that Source2Synth achieved 25.51% over the best alternative MHQA method, when in fact the gap over the best alternative fine-tuned MHQA baseline (HPQA-only fine-tuning at 58.40% 3-shot) is 7.65 points (66.05 - 58.40). The headline numbers are maximally favorable to Source2Synth by comparing against different baselines for different tasks.

**Claim 2: "The curation step proves essential to these gains."**

Strongly supported by Table 1 (`LLMSynth-datamix` = 57.46% vs. `LLMCurated-datamix` = 65.23% zero-shot) and Table 2 (`LLMSynth` = 23.86% EM vs. `LLMCurated` = 34.50% EM). The curation step provides 7-11 percentage points of improvement consistently across tasks, model configurations, and data sizes.

However, the claim's strength is qualified by a missing ablation: the imputation step (MHQA only) is bundled with filtering in the curation stage. We cannot attribute the 7-11 point gain entirely to filtering, because the imputation contribution is unknown. The perplexity improvements (Table 9) and qualitative example (Figure 6) strongly suggest imputation improves question naturalness, but without a filtering-only vs. filtering+imputation ablation, the claim that "curation" (as a combined step) is essential cannot be decomposed into which sub-component is doing the work.

Furthermore, the TQA result is based on a very high rejection rate (73%), which means that the curated model is trained on only 27% of the originally generated data. The baseline `LLMSynth` is trained on Slice 0 (uncurated), which contains the same number of examples (4000 per slice) but with a different distribution (unfiltered). A more rigorous ablation would train `LLMSynth` on a randomly sampled 27% subset of Slice 0 to control for dataset size—if the performance gap persists, it's genuinely about learnability filtering; if it narrows, some of the "curation gain" is actually a data quality vs. data quantity tradeoff.

**Claim 3: "A model fine-tuned exclusively on synthetic data with only bridge-type questions generalizes to comparison-type questions."**

Strongly supported by Table 4. `LLMCurated` (synthetic bridge questions only, zero comparison data) outperforms the base Llama2 70B-Chat on comparison questions across all difficulty levels (79.1% vs. 66.6% hard, 82.3% vs. 71.3% medium, 88.0% vs. 73.2% easy), and outperforms the HPQA-only fine-tuned model on hard and medium comparison questions (79.1% vs. 74.5% hard, 82.3% vs. 78.3% medium). The generalization is real and strong.

However, the scope of this claim is limited in ways the paper acknowledges but does not foreground. The comparison questions in HPQA are structurally different from bridge questions but still involve the same underlying skill set: reading Wikipedia articles, extracting entities, and comparing attributes. The generalization may not extend to question types that require fundamentally different reasoning patterns (e.g., temporal reasoning chains, multi-document synthesis without a clear entity hop). The claim is empirically true for the tested distribution but may not generalize to arbitrary question type shifts.

Additionally, the synthetic data still includes the decomposition structure (Q1 → Q2 → answer) and the entity hop, which teaches the model a structured reasoning pattern. Whether a model trained on synthetic data *without* the explicit decomposition structure would generalize as well is not tested. The generalization may be attributable to the explicit reasoning chain format rather than the bridge-type content.

**Claim 4 (from Section 4 of prior sections): "Grounding synthetic data in real-world sources is the critical architectural principle that distinguishes Source2Synth from prior work."**

Supported with qualifications by the ungrounded data ablation (Appendix C.3). Replacing Wikipedia with a hand-picked topic list causes a 4-7 point accuracy drop across two model families (Tables 7-8). This is a real effect and demonstrates that grounding matters.

However, the ablation conflates two changes: (1) grounding in real data is removed, and (2) the diversity and specificity of the topics are changed. A topic list of 50 words like "Moon," "Ocean," "Roman Empire" is not only ungrounded but also less diverse and less specific than Wikipedia articles with their rich entity structures, factual content, and hyperlink graphs. The drop in performance could be due to reduced diversity rather than the absence of grounding per se. A fairer ablation would compare Wikipedia grounding to a diverse but synthetic knowledge base (e.g., LLM-generated Wikipedia-style articles) to isolate the grounding mechanism from the information richness.

The paper also does not test whether the grounding mechanism's value comes primarily from factual accuracy (avoiding hallucinated facts) or from structural consistency (the seed enforcing logical coherence across sub-questions). The ungrounded ablation tests both simultaneously; a design that tested structural consistency alone (e.g., synthetic documents with fabricated but internally consistent facts used with the same seed mechanism) would separate these effects.

**Claim 5 (from Section 4 of prior sections): "The slice-based self-curation protocol measures learnability rather than correctness, establishing a new paradigm for synthetic data filtering."**

This is a conceptual framing claim, not an empirical one, and the experiments support it indirectly rather than directly. The evidence is that filtering based on `LLMSynth`'s answerability (a learnability proxy) produces better downstream models than not filtering at all. But the paper never compares learnability-based filtering against correctness-based filtering (e.g., using a separate verifier model, rule-based checks, or human judgments of correctness). Without this comparison, we cannot say that *learnability-based* filtering is the key insight rather than just *any* filtering being better than no filtering.

The claim's strength depends on whether you accept the conceptual argument that "answerability by a partially-trained model" is a fundamentally different criterion than "correctness." The paper makes a reasonable case for this distinction (especially with the 73% TQA rejection rate for correct-but-unlearnable examples), but the empirical design doesn't directly test whether a correctness-based filter with the same rejection rate would perform worse. If a simple rule-based filter (e.g., "discard SQL queries longer than 100 tokens" or "discard MHQA questions with perplexity above a threshold") achieved similar curation gains, the learnability framing would be less compelling.

**Methodological weaknesses not yet discussed:**

- **No held-out test set for the curation model.** `LLMSynth` is trained on Slice 0 and evaluated (as a filter) on Slice 1. These slices come from the same synthetic data generation run and may share systematic biases (e.g., if a particular Wikipedia article generated 50 examples, they might be split across Slice 0 and Slice 1, making the curation model's answerability judgment artificially easy). A stricter protocol would generate Slice 0 and Slice 1 from completely independent source data (different Wikipedia articles, different tables).

- **Single data generation run.** All results are based on one synthetically generated dataset. The paper does not report variance across multiple generation runs with different random seeds. Given the stochasticity of LLM generation, the composition of the synthetic dataset (which examples pass, which fail, the distribution of difficulty) could vary substantially between runs.

- **HPQA test set size of 7,405 examples for Table 1, but Table 4 appears to use the HPQA train set (which has difficulty labels) as evaluation.** The paper states Table 4 evaluates "on the full HPQA train dataset (where questions are labelled with easy, medium and hard)." This means the difficulty-stratified analysis uses the training set for evaluation, not the test set. This is a significant concern: a model fine-tuned on some HPQA training data is being evaluated on the HPQA training set (or a subset thereof), creating a potential train-test overlap that could inflate reported accuracy on the stratified analysis relative to what would be observed on truly held-out data.

- **No reporting of the number of unique source entities/tables used.** The paper states 50 randomly selected Wikipedia articles for MHQA (generating 1250 examples, or ~25 per article) and 4000 tables for TQA. The per-source yield matters because if the generation oversamples a small number of sources, the synthetic data may lack the diversity needed for robust generalization. The paper reports using only 50 tables for the TQA experiment described in Appendix D (800 seed statements → 658 executable SQL), which produces a high density of examples per table and may contribute to the high rejection rate if `LLMSynth` overfits to table-specific patterns rather than learning general SQL composition.

- **Missing baseline: fine-tuning on uncurated data with the same filtering rate applied randomly.** The paper attributes curation gains to learnability-based filtering. A control experiment that randomly discards the same fraction of examples (e.g., randomly reject 73% of TQA data, train on the remaining 27%) would distinguish whether the gain comes from data reduction (fewer but higher-quality examples vs. many noisy examples) or from the specific learnability criterion. If random filtering achieved similar performance, the learnability mechanism would be less important than the paper claims.
- **The bridging of generation cost vs. human annotation cost is purely qualitative.** The paper motivates Source2Synth by the expense of human annotation but provides no cost comparison—no estimate of LLM inference cost to generate 1250 MHQA examples or 8000 TQA examples, no comparison to the cost of annotating equivalent real data, and no analysis of how generation cost scales with the amount of source data processed.

## 6. Limitations and Trade-offs

### The Difficulty Estimation Cost Is Unaccounted For in the Curation Protocol

**The assumption or constraint.** The paper's curation protocol—fine-tuning `LLMSynth` on Slice 0, then using it to filter Slice 1—requires generating a full synthetic dataset, splitting it, and performing an intermediate fine-tuning run before the final training can begin. The paper explicitly acknowledges that the difficulty estimation for the broader Source2Synth framework incurs cost, but for curation specifically, the cost of training `LLMSynth` is treated as infrastructure rather than as part of the data generation budget. Section 3.2 describes the procedure without quantifying the computational overhead: the slice training, the $k = 3$ inference passes over Slice 1, and the imputation step (for MHQA) all consume compute that is not reflected in the headline accuracy numbers.

**The consequence.** In practice, the curation stage approximately **doubles the total training compute** relative to using the synthetic data directly: the model must be fine-tuned twice—once for `LLMSynth` (on Slice 0) and once for `LLMCurated` (on curated Slice 1). For a 70B-parameter model (MHQA experiments), this is a substantial cost. Furthermore, Slice 0 is consumed entirely by training `LLMSynth` and does not directly contribute to the final model (the paper does not clarify whether Slice 0 is also used for `LLMCurated` training, or only the curated Slice 1). If Slice 0 is discarded, the effective data efficiency is halved—the final model trains on only half the generated examples. The paper's data efficiency claims (e.g., achieving strong performance with 1250 synthetic examples) would then need to be re-evaluated against a baseline that uses all 1250 examples without curation, since the curation protocol only delivers ~556 curated examples (1250 × 0.5 for the Slice 1 allocation × 0.89 after 11% rejection) to the final model.

**What evidence exists in the paper.** The paper provides no direct measurement of curation's computational cost. The rejection rates (13% for MHQA, 73% for TQA; Section 3.2.1) and the slice architecture (Section 3.2) are described, but the FLOPs or GPU-hours required for `LLMSynth` training and Slice 1 inference are not reported. The scaling analysis in Figure 5 sweeps the number of synthetic examples but holds the curation protocol constant—it does not ask whether the same total compute, if spent on generating more uncurated data rather than on curation, would perform better.

**Mitigation status.** The paper does not attempt to mitigate this cost. The slice-based protocol is presented as the method itself, not as an expensive step to be optimized. A natural mitigation—using cross-validation where Slice 0 and Slice 1 are swapped and both contribute to the final model—is not explored. The paper also does not discuss whether a smaller or quantized `LLMSynth` could reduce curation cost, or whether the curation step could be amortized across multiple data generation runs.

---

### The TQA Pipeline Produces a 73% Rejection Rate, Making It Impractically Inefficient as a Data Generation Method

**The assumption or constraint.** The paper presents Source2Synth as a unified framework for generating synthetic data grounded in real-world sources, but the MHQA and TQA instantiations exhibit **dramatically different yield rates**. For MHQA, approximately 87% of generated examples survive curation (13% rejection; Section 3.2.1). For TQA, only 27% survive (73% rejection; Section 3.2.1). The TQA pipeline is therefore generating `~3.7` examples for every 1 example that reaches the final training set—a `~73%` waste rate.

**The consequence.** This waste rate fundamentally undermines the scalability argument for Source2Synth in tabular domains. If generating 8000 examples per slice (Section 4.2, "Model") requires generating approximately `~29,600` raw SQL-question pairs to yield `8000` curated examples, the method is consuming an order of magnitude more generation compute than the final dataset size suggests. The paper's TQA baselines use 4000 tables from the WikiSQL training set (Section 3.1.1). Scaling to larger table corpora would amplify this inefficiency. A practitioner facing the choice between Source2Synth and paying human annotators would need to weigh the cost of generating and discarding ~3 invalid examples per valid example against annotation costs—and the paper provides no data to make that comparison.

More subtly, the 73% rejection rate introduces a **selection bias** whose properties are uncharacterized. The examples that survive curation are those that `LLMSynth` (trained on Slice 0) can answer. If `LLMSynth` has systematic weaknesses—for instance, it may answer simple aggregation queries well but struggle with joins or nested subqueries—the curated dataset will reflect those biases. The final model `LLMCurated` may then appear to perform well on the evaluation set because the evaluation set and the curated training set share the same bias, not because the model has learned general SQL composition.

**What evidence exists in the paper.** The 73% rejection rate is reported explicitly in Section 3.2.1 for TQA. The paper does not provide a breakdown of *why* examples are rejected—whether due to SQL complexity, seed fact ambiguity, question-SQL misalignment, or other factors. Appendix D mentions that out of 800 seed statements generated from 50 tables, only 658 produced executable SQL (an 82.25% rate), but this pre-curation filter is separate from the curation rejection. The paper does not characterize the distribution of surviving TQA examples (by query type, table size, or complexity) or compare it to the distribution of rejected examples.

**Mitigation status.** The paper partially addresses this by noting that the TQA curation "consists only of the filtering step" (Section 3.2.2) and acknowledges in Appendix A that multi-table tool use is not supported and that "more clever sampling techniques beyond rejection sampling" could improve the method. However, no concrete steps are taken to reduce the rejection rate or to analyze whether the `27%` yield is inherent to the TQA task or specific to the current prompt engineering and base model choices. The paper frames the 73% rejection rate as a property of the data quality rather than as a failure mode of the generation pipeline.

---

### The Method Has Only Been Demonstrated on Publicly Available Benchmarks with a Single Model Family Per Task

**The assumption or constraint.** All Source2Synth experiments are conducted on two specific benchmarks (HotpotQA and WikiSQL) using two specific base model families (Llama2 70B-Chat for MHQA, Starchat-beta for TQA). The paper states in Section 4 that it focuses on "publicly available data and evaluation benchmarks" for reproducibility, and acknowledges in Section 6 that the method has "potential in many important domain-specific applications, such as medical or legal QA" but does not evaluate it there. The cross-model experiments in Appendix C.2 test synthetic data transfer to Llama3 and Llama4, but only for MHQA and only using Llama2 70B-Chat as the generator—they do not test whether the TQA pipeline transfers across model families or whether the MHQA pipeline works with a non-Llama generator.

**The consequence.** The paper's central claim—that Source2Synth is a "scalable approach for synthetic data generation and curation that is grounded in real-world data sources" (Section 1)—is supported for exactly two (model, benchmark) pairs. A practitioner in the medical or legal domain (the paper's own motivating examples, Section 6) would need to assume that the pipeline transfers without degradation to different data distributions, different reasoning structures, and different base models. Several specific transfer risks exist:

- **Generator model dependence.** The MHQA pipeline uses few-shot prompts (Figures 15-17) that were designed for and tested with Llama2 70B-Chat. A weaker generator may produce lower-quality sub-questions or fail at the merge step; a stronger generator may produce questions that are answerable by the generator but unlearnable by a weaker student. The TQA pipeline uses zero-shot prompts (Figures 11-13) that may be more or less effective depending on the generator's SQL proficiency.

- **Domain-specific entity structures.** Wikipedia articles have clean entity links and consistent formatting that make seed extraction straightforward. Medical texts (clinical notes, research papers) or legal documents (contracts, case law) have different entity structures, may use domain-specific terminology that obscures the "hop" relationship, and may contain implicit rather than explicit links between entities. The seed extraction mechanism may fail or produce lower-quality seeds in these domains.

- **Benchmark-specific evaluation.** HotpotQA and WikiSQL are well-established but narrow benchmarks. HotpotQA questions are exclusively two-hop and drawn from Wikipedia. WikiSQL queries are single-table and relatively simple (no JOINs, no subqueries beyond basic aggregation). Performance on these benchmarks does not guarantee performance on more complex multi-hop reasoning (3+ hops, implicit hops, multi-document synthesis) or more complex SQL tasks (multi-table queries, schema understanding, complex aggregations).

**What evidence exists in the paper.** The paper provides two forms of cross-domain evidence, both limited. First, the two tasks themselves demonstrate the framework on different data types (documents vs. tables), but within each task there is no cross-domain evaluation. Second, the cross-model experiments (Appendix C.2, Tables 5-6) demonstrate that MHQA synthetic data generated by Llama2 70B-Chat transfers to Llama3 8B-instruct and Llama4 17Bx16E, with gains of 23.06 and 36.90 percentage points respectively. However, these are still within the Wikipedia domain and the MHQA task—the cross-model transfer does not test cross-domain generalization.

**Mitigation status.** The paper explicitly acknowledges this limitation in Section 6: "We believe our method is valuable in domains where unstructured data is available as a source—such as the legal and medical fields—even though this data is typically not readily available in the form of question-answer pairs." This is a statement of belief, not an empirical claim. The paper frames domain-specific application as future work. Appendix A notes that "Source2Synth can be extended to any domain that has such data-types as source even if it is not publicly available," but provides no validation of this extension.

---

### The Comparison-Question Generalization Result Is Built on an Evaluation That Overlaps with Training Data

**The assumption or constraint.** The paper's most striking result—that a model trained only on synthetic bridge questions generalizes to comparison questions, outperforming the HPQA-only fine-tuned model on hard comparison questions by 4.6 percentage points (Table 4)—relies on a difficulty-stratified evaluation that uses the **HPQA training set** for evaluation, not the HPQA test set. The table caption states: "We evaluate models on the full HPQA train dataset (where questions are labelled with easy, medium and hard)." This is distinct from the test set evaluation in Table 1, which uses the HPQA test set.

**The consequence.** The difficulty-stratified results in Table 4 are **potentially contaminated** for any model that was fine-tuned on HPQA training data. The `LLMCurated-datamix` model includes 500 examples from the HPQA training split in its fine-tuning data (Section 4.1, "Model"). If the difficulty labels are present on the HPQA training set, and the "full HPQA train dataset" used for evaluation includes the exact examples used for fine-tuning, the reported accuracies are inflated by memorization, not by genuine generalization. Even if the 500 fine-tuning examples are a subset of the training set and the evaluation covers all training examples, the model has seen some of the evaluation data during training.

This applies most directly to the HPQA-only fine-tuned baseline and the datamix models, but it also affects the interpretation of the synthetic-only `LLMCurated` result. If the HPQA training set questions share structural or topical similarities with the synthetic training data (both are drawn from English Wikipedia), the synthetic model may be benefiting from in-distribution evaluation rather than demonstrating cross-type generalization to genuinely novel comparison questions.

**What evidence exists in the paper.** The paper explicitly states the evaluation set in Table 4's caption: "We evaluate models on the full HPQA train dataset." The data contamination checks described in Section 4.1 apply only to the **synthetic data vs. HPQA test set** ("we check if its entity E (seed) is present in any of the questions in HPQA's test-set... we found that none of the synthetic data overlaps with the questions in HPQA test set"). No contamination check is described for the HPQA training set, which is the evaluation set for Table 4.

Furthermore, the HPQA training set is the source of the 500 real examples used in the datamix configurations. If these 500 examples are part of the "full HPQA train dataset," the `LLMCurated-datamix` row in Table 4 is training-on-evaluation for up to 500 of the evaluated questions. The paper does not specify whether the 500 examples were excluded from the evaluation set used for Table 4.

**Mitigation status.** The paper does not address this issue. The discrepancy between test-set evaluation (Table 1, Figure 5) and training-set evaluation (Table 4) is noted in the table caption but not discussed in the text as a limitation. A clean evaluation would use the HPQA test set with difficulty labels obtained from the dataset metadata (if available) or from a separate difficulty classifier, ensuring that no fine-tuned models have seen the evaluation examples. The paper acknowledges in Appendix A that the difficulty labels come from the HPQA train dataset, implicitly conceding the data source but not the contamination risk.

---

### The Imputation Step and the Filtering Step Are Bundled Together, Preventing Attribution of the Curation Gain

**The assumption or constraint.** The curation stage for MHQA applies two operations to Slice 1 in sequence: (1) filtering via rejection sampling (discard examples `LLMSynth` cannot answer in $k = 3$ tries) and (2) imputation (reconstruct `$Q_1'$` from the remaining components, then verify that the reconstructed multi-hop question `$Q'$` preserves the original answer). The paper reports the combined effect of these two steps as "curation" and never ablates them separately. Section 3.2 describes them as sequential: the data is first filtered, then the surviving examples undergo imputation.

**The consequence.** The attribution of the 7.77-point curation gain in MHQA (Table 1: `LLMSynth-datamix` = 57.46% vs. `LLMCurated-datamix` = 65.23%, zero-shot) cannot be decomposed into the contribution of filtering vs. imputation. It is possible that one of these steps accounts for most of the gain and the other is negligible. If imputation is the primary driver, then the paper's framing of curation as "learnability-based filtering" (the insight discussed in the key innovations section) would be incorrect—the gain would come from making the questions more natural, not from removing unlearnable examples. Conversely, if filtering is the primary driver, then the imputation complexity is unnecessary for MHQA, and the pipeline could be simplified.

The imputation step also introduces a **correctness risk**: `LLMSynth` reconstructs `$Q_1'$` given `$Q$`, `$Q_2$`, `$E$`, and `$D_1$`. This reconstruction is verified by checking that the new multi-hop question `$Q'$` (assembled from `$Q_1'$` and `$Q_2$`) yields the same answer `$A$` as the original. However, the verification only checks answer consistency—it does not verify that `$Q_1'$` is genuinely answerable from `$D_1$` with answer `$E$`. If `LLMSynth` generates a `$Q_1'$` that happens to preserve the final answer through a different reasoning path (or through coincidence), the example may pass verification but contain a structural inconsistency that degrades training quality.

**What evidence exists in the paper.** The only direct evidence about imputation's effect is the perplexity reduction in Table 9 (Appendix C.4): grounded synthetic question perplexity drops from 24.7 to 13.6 after imputation, and the qualitative example in Figure 6 shows a cleaner question. These measurements confirm that imputation improves question naturalness, but they do not measure its impact on downstream model performance. No ablation compares `LLMCurated` trained with filtering-only vs. filtering+imputation. For TQA, there is no imputation step (Section 3.2.2: "TQA the curation process consists only of the filtering step"), so the TQA results (Table 2) provide a cleaner signal of filtering's contribution, but the tasks are too different to transfer conclusions.

**Mitigation status.** The paper does not attempt to separate these effects or to discuss the lack of an ablation. Appendix C.4 quantifies perplexity changes but does not perform a downstream performance ablation. The imputation step is described as a method component (Section 3.2.2), and its effect is measured on question quality (perplexity), but its contribution to the core claims about "curation" improving model performance is unquantified.

---

### The Method Cannot Improve Performance on Problems Where the Base Model Has Near-Zero Capability

**The assumption or constraint.** Source2Synth uses the base LLM both as the data generator (for seed extraction, question generation, merging, and SQL composition) and as the curation model (via `LLMSynth`). All synthetic examples are ultimately derived from the base model's ability to understand the source data, decompose tasks, and produce correct reasoning chains. If the base model cannot perform these operations reliably—because the task domain is too complex, the source data is too noisy, or the reasoning chain exceeds the model's capacity—then Source2Synth cannot generate high-quality synthetic data regardless of how much curation is applied. The paper indirectly acknowledges this in Appendix A: "we did not experiment with PRM tree-search techniques in combination with revisions," referring to the possibility of combining different mechanisms to push past capability ceilings—but this is about test-time compute, not about the base model's data-generation capability.

**The consequence.** Source2Synth can **amplify existing capabilities but cannot create new ones from scratch.** If the base LLM's pass@1 on generating a correct sub-question from a Wikipedia article is near zero, or if it cannot produce executable SQL for a given table schema, then the synthetic data pipeline will produce mostly invalid examples that are filtered out, and the few that survive will be trivially simple or structurally flawed. This is distinct from the difficulty-dependent efficacy observed in the main results—the paper shows that Source2Synth works well on medium and hard HotpotQA questions (Table 4), but these are "hard" relative to the HotpotQA distribution, not "hard" in an absolute sense. The base Llama2 70B-Chat already achieves 14.5% on hard bridge questions (Table 4), meaning it has non-trivial capability that can be amplified. For tasks where the base performance is truly zero (e.g., 3-hop reasoning with implicit entity links, or multi-table SQL with complex JOIN conditions), the pipeline would likely fail entirely.

This limitation is particularly acute for the **domain-specific applications** the paper envisions (Section 6: medical and legal QA). If the base LLM has limited medical knowledge or legal reasoning capability, the synthetic data it generates will reflect those limitations—it cannot ground examples in medical facts it doesn't know or reason about legal concepts it hasn't internalized. The "grounding" to real-world sources provides factual correctness *to the extent the LLM can accurately extract and use information from those sources*. If the LLM misinterprets a medical term in a clinical note, the resulting synthetic example will be factually wrong despite being "grounded" in real data.

**What evidence exists in the paper.** The paper does not directly test this limitation, but it is visible in two indirect ways. First, the TQA rejection rate of 73% (Section 3.2.1) suggests that even for the relatively simple WikiSQL task (single-table queries), the Starchat-beta base model struggles to generate SQL that a partially-trained `LLMSynth` can reproduce. The gap between generation capability (producing executable SQL 82.25% of the time; Appendix D) and learnability (only 27% survive curation) indicates that the base model's SQL generation skill is sufficient for producing valid queries but insufficient for producing *pedagogically useful* queries that teach the task. On a more complex SQL task (multi-table, nested subqueries), this gap would almost certainly widen, potentially to the point of zero yield.

Second, the MHQA results in Table 4 show that even the best Source2Synth model (`LLMCurated-datamix`) achieves only 31.3% on hard bridge questions—a substantial improvement over the base model's 14.5% but still leaving ~69% of hard questions unanswered. The method improves within the base model's capability range but does not approach ceiling performance on the hardest subset.

**Mitigation status.** The paper does not explicitly address this limitation. The discussion of future work in Section 6 gestures toward extending the method to other domains and tasks but does not discuss the base-model-capability prerequisite. The finding in Appendix C.2 that Source2Synth data generated by a strong model (Llama2 70B-Chat) transfers to weaker models (Llama3 8B-instruct, Llama4 17Bx16E) is partially reassuring—it suggests that a more capable generator can produce data that teaches less capable students—but this still assumes the *generator* has the relevant capability. If no available model can reliably perform the task, Source2Synth provides no bootstrapping path. This is a fundamental scope constraint: the method is a data amplification technique, not a capability creation technique.

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
