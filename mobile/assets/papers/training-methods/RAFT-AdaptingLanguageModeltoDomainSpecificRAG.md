# RAFT: Adapting Language Model to Domain Specific RAG

**ArXiv:** [2403.10131](https://arxiv.org/abs/2403.10131)

## 🎯 Pitch

RAFT introduces Retrieval-Augmented Fine-Tuning—a novel training approach that adapts large language models for domain-specific retrieval-augmented generation (RAG) tasks by explicitly teaching the model to extract answers from relevant retrieved documents while ignoring distractors. This method not only yields substantial performance improvements over standard fine-tuning and off-the-shelf RAG, but also enhances robustness to retrieval imperfections, making it highly impactful for mission-critical applications like medical QA and enterprise knowledge assistants where accurate, context-sensitive reasoning over domain documents is essential.

---

## 1. Executive Summary

This paper introduces **RAFT** (Retrieval Augmented Fine Tuning), a training recipe that adapts a base LLM (LLaMA2-7B) to domain-specific open-book question-answering by fine-tuning on question-answer pairs where the model must learn to extract answers from a mixture of relevant ("golden") and irrelevant ("distractor") documents, citing verbatim evidence within chain-of-thought responses. RAFT consistently outperforms both domain-specific fine-tuning with and without RAG across PubMed, HotpotQA, and the Gorilla APIBench datasets, with gains as large as 35.25% on HotpotQA and 76.35% on Torch Hub over the base LLaMA2-7B-chat model. The method establishes that training with a fraction of examples where the golden document is deliberately withheld—and with varying numbers of distractor documents—improves robustness to imperfect retrieval at test time, with the optimal proportion of gold-context training examples ranging from 40% to 100% depending on the dataset.

## 2. Context and Motivation

### The Core Problem: How Do You Adapt an LLM to a Domain Where the Answers Are in a Known Set of Documents?

The paper tackles a specific, practically ubiquitous problem. Suppose you deploy a pretrained LLM in a specialized setting — a company's internal knowledge base, a medical document collection, a specific set of API documentation, or a legal corpus. In these settings, the LLM's broad pretraining knowledge is less relevant; what matters is that it answers questions **accurately using the specific documents at hand**. The question is: given that you have access to these documents both at training time (for adaptation) and at test time (via retrieval), what is the best way to combine fine-tuning and retrieval to maximize downstream accuracy?

This is not an abstract research question. The paper opens by noting that "adapting LLMs to the specialized domains (e.g., recent news, enterprise private documents, or program resources constructed after the training cutoff) is essential to many emerging applications" (Section 1, citing Vu et al., 2023; Lazaridou et al., 2022). These are settings where the knowledge required does not exist in the pretraining corpus — either because it was created after the training cutoff (API documentation for a newly released library) or because it is proprietary (internal company documents). The LLM fundamentally cannot answer these questions from its parametric memory alone; it must learn to *read* the provided documents.

The paper formalizes this as a **domain-specific open-book exam**. In a general open-book exam, the LLM can consult any external source, and performance depends heavily on the retriever's quality. In a domain-specific open-book exam, the domain is known in advance, and the test documents come from the same collection the model was adapted on. This shifts the problem from "retrieve the right document from the whole internet" to "accurately extract and reason over information from documents that look a lot like the ones you studied." The central question of the paper is: **how should the training recipe be designed to maximize performance in this exact scenario?**

### Why This Problem Matters: The Failure of Naïve Adaptation

The practical importance of this problem becomes clear when you look at how existing approaches fail. The paper provides a striking example of this failure in Table 1. Consider the LLaMA2-7B-chat model on the HotpotQA dataset — a multi-hop question-answering benchmark based on Wikipedia articles:

- **LLaMA2-7B (no RAG):** 0.54% accuracy. Effectively zero. The model simply cannot answer domain-specific multi-hop questions from parametric memory.
- **LLaMA2-7B + RAG:** 0.03% accuracy. Adding retrieved documents to the prompt makes performance *worse*, not better. The model cannot effectively read and reason over the provided context.
- **Domain-Specific Fine-tuning (DSF) without RAG:** 6.38% accuracy. Teaching the model about the domain through supervised fine-tuning helps modestly.
- **DSF + RAG:** 4.41% accuracy. Adding retrieval back to the fine-tuned model *decreases* performance, dropping from 6.38% to 4.41%.

This pattern — where adding RAG to a domain-fine-tuned model hurts performance — appears across several datasets. On TensorFlow Hub, DSF achieves 86.56% but DSF + RAG drops to 60.29%. These are not small regressions; they are catastrophic degradations, where providing the model with the very documents that contain the answer causes it to perform substantially *worse* than if it were given no documents at all.

The paper's diagnosis of this failure is specific and actionable. DSF teaches the model to answer questions using its internal knowledge (acquired during fine-tuning) but does not teach it to **read**. When RAG is subsequently applied at test time, the model encounters documents it was never trained to process — it has not learned to distinguish relevant from irrelevant information, to extract precise evidence, or to integrate retrieved content with its parametric knowledge. The model is, in effect, taking an open-book exam having only studied for a closed-book one.

This problem has real economic consequences. The dominant deployment paradigm for domain-specific LLMs increasingly involves RAG pipelines (retrieve-then-generate). If a fine-tuned model cannot effectively use the retrieved documents — or worse, is actively harmed by their presence — then the entire RAG investment (retriever infrastructure, vector databases, document chunking pipelines) yields negative returns. Fixing this misalignment between how models are trained and how they are deployed is the paper's central practical contribution.

### The Analogy: How to Study for an Open-Book Exam

The paper introduces a memorable analogy that frames the entire problem (Section 1, Figure 1). Consider three ways to prepare for an exam where you are allowed to bring your textbook:

**Scenario A — "Memorization" (standard domain-specific fine-tuning):** You study by memorizing the textbook. At test time, you answer questions from memory without opening the book. This is equivalent to DSF without RAG. The model learns domain-specific patterns and answer styles but does not learn to reference documents. Performance is limited by what fits in parametric memory, and difficult multi-hop questions are nearly impossible.

**Scenario B — "No Studying" (standard RAG with a base model):** You don't study at all. At test time, you frantically flip through the textbook trying to find relevant passages. This is equivalent to applying RAG to a base instruction-tuned model. The retriever provides documents, but the model has never practiced reading and reasoning over documents like these, so it struggles to extract correct answers and is easily distracted by irrelevant content.

**Scenario C — "Studying for the Open-Book Test" (RAFT):** You study by practicing with the textbook open, learning which sections are relevant to which types of questions, how to find evidence quickly, and how to ignore irrelevant chapters. At test time, you do exactly what you practiced. This is the training paradigm RAFT implements.

This analogy clarifies the conceptual gap: prior fine-tuning approaches treat the document collection as a source of knowledge to be internalized, while RAG approaches treat it as a resource to be queried on-the-fly. Neither prepares the model to *use documents at test time in the way they will actually be presented* — which is through a noisy retrieval process that mixes relevant passages with irrelevant distractors.

### Prior Approaches and Where They Fall Short

The paper situates itself against three categories of prior work, each with distinct shortcomings that RAFT addresses.

#### Retrieval-Augmented Generation (RAG) with Base Models

The most straightforward approach is to take a pretrained instruction-tuned LLM and prepend retrieved documents to the user's question. This is the "Llama2 + RAG" baseline in the paper's experiments. The approach is theoretically appealing because it separates concerns: the retriever handles knowledge access, and the LLM handles reasoning and generation.

**Where it falls short:** The model has never been fine-tuned in the presence of retrieved documents from this specific domain. It does not know the document structure, the typical patterns of how evidence is distributed across passages, or how to filter out irrelevant content that the retriever includes. The paper's results show this approach often performs dramatically worse than even the base model without retrieval (e.g., 0.54% → 0.03% on HotpotQA). The base model's reading comprehension capabilities, developed on general-domain data, do not transfer effectively to specialized document collections without further adaptation.

Moreover, standard RAG is entirely dependent on retriever quality. If the retriever fails to place the golden document in the top-k results, the LLM has no fallback — it cannot draw on any domain-specific knowledge acquired during fine-tuning because there was none.

#### Domain-Specific Fine-Tuning (DSF) Without RAG

This approach fine-tunes the LLM on question-answer pairs from the target domain, teaching the model both the domain's content and the expected answering style. After fine-tuning, the model is used in a closed-book setting — it answers questions from its updated parametric knowledge without access to external documents. This is the "DSF" baseline in Table 1.

**Where it falls short:** The model's capacity to store domain knowledge in its parameters is finite and imperfect. For knowledge-intensive tasks requiring precise recall of specific facts, numbers, or API signatures, parametric memory is unreliable. More critically, this approach completely fails to leverage the fact that the test-time setting is open-book — the relevant documents *will* be available, but the model was never trained to use them.

The paper also notes a subtler failure mode. DSF teaches the model to answer questions without ever consulting documents. When documents are later provided at test time (as in the DSF + RAG baseline), the model may ignore them because it was trained to answer from memory, or worse, may become confused by the presence of document context it was not trained to process. The performance regressions in Table 1 (DSF vs. DSF + RAG on HotpotQA: 6.38% → 4.41%) provide direct evidence for this hypothesis.

#### Domain-Specific Fine-Tuning With RAG

This baseline — DSF + RAG — fine-tunes the model on domain QA pairs and then applies retrieval at test time. It seems like it should combine the benefits of both worlds: domain adaptation plus open-book access.

**Where it falls short:** The training procedure is misaligned with deployment. During training, the model learns from question-answer pairs without document context. During testing, documents are introduced. The model has never practiced: (1) identifying which parts of a retrieved document are relevant to a question, (2) ignoring distractor documents that the retriever includes, (3) resolving conflicts between its parametric memory and the retrieved content, or (4) extracting and citing verbatim evidence to justify its answers.

The results in Table 1 are damning. DSF + RAG often underperforms both DSF alone (HotpotQA: 4.41% vs. 6.38%) and the base model with RAG (TensorFlow Hub: 60.29% vs. 43.06% for LLaMA2-7B + RAG — actually, this one is better, but see HuggingFace: 42.59% vs. 26.43%, which shows the pattern). The core issue is that fine-tuning has taught the model a skill (closed-book QA) that is *different from* the skill needed at test time (open-book reasoning with noisy retrieval). The training-to-deployment gap is the root cause.

#### Contemporary Fine-Tuning-for-RAG Approaches

The paper acknowledges recent work that also explores fine-tuning LLMs for improved RAG performance (Section 6, citing Lin et al., 2023a; Wang et al., 2023; Xu et al., 2023; Liu et al., 2024). These works "focus on constructing a combination of finetuning dataset for RAG and train a model to perform well on these tasks." However, the paper draws a crucial distinction:

> "in their settings, at test time, the domain or documents can be different than the training time; whereas our paper studies a slightly opposite scenario where we only care about testing the LLM on the same set of documents."

This is a key positioning move. Prior RAG fine-tuning work (e.g., RA-DIT by Lin et al., 2023a; InstructRetro by Wang et al., 2023) aims to improve a model's general ability to use retrieved documents across many domains — a form of meta-training for RAG. RAFT, by contrast, targets the narrower but practically important setting where the domain is fixed and known in advance. This domain-specific focus allows RAFT to make design choices that would not make sense in a general RAG fine-tuning setup, such as training the model to memorize answers for a fraction of questions by withholding the golden document (forcing the model to fall back on parametric knowledge when retrieval fails).

### How RAFT Positions Itself

RAFT is presented not as a fundamentally new architecture or objective function, but as a **training data construction recipe**. The innovation is in *how* the fine-tuning data is assembled, not in *what* loss function optimizes it (it uses standard supervised fine-tuning, next-token prediction). The recipe encodes several design principles, each motivated by a specific failure mode of prior approaches:

1. **Include documents in the training context.** Unlike DSF, which trains on Q → A pairs, RAFT trains on Q + D* + D₁ + ... + Dₖ → A*. This directly addresses the training-deployment mismatch: what the model sees during training matches what it sees at test time.

2. **Include distractor documents during training.** The training context includes not only the golden document(s) from which the answer is derived, but also k distractor documents that are irrelevant to the question. This teaches the model to *filter* — to identify which documents contain evidence and which are noise. The paper explicitly connects this to prior findings that "LLMs can be easily distracted by irrelevant context" (Shi et al., 2023a; Weston & Sukhbaatar, 2023; Liu et al., 2023), positioning distractor training as a robustness intervention.

3. **Withhold the golden document for a fraction of training examples.** For a proportion (1 − P) of training questions, the golden document is deliberately excluded; the model sees only distractors. This forces the model to answer from parametric memory when relevant documents are absent — exactly the skill needed when the retriever fails at test time. The paper emphasizes this design choice: "By removing the golden documents in some instances, we are compelling the model to memorize answers instead of deriving them from the context" (Section 3).

4. **Generate chain-of-thought answers with verbatim citations.** Answers are not just the final response but include a reasoning chain with explicit quotations from the source documents, delimited by `##begin_quote##` and `##end_quote##` tags. This teaches the model to ground its answers in specific evidence rather than hallucinating or relying on parametric priors. The paper shows (Section 3) that this CoT + citation format improves the model's accuracy, and the ablation in Table 2 quantifies this contribution: removing CoT drops performance by 9.66% on HotpotQA and 14.93% on HuggingFace.

**The overarching thesis** is that these four design decisions together solve the training-deployment misalignment that plagues prior approaches. The model learns to: read documents when they are present (design 1), ignore irrelevant ones (design 2), fall back on memory when documents are absent or unhelpful (design 3), and explicitly ground its reasoning in source material (design 4). This combination is what the paper argues is necessary for robust domain-specific open-book performance, and what no prior approach provides in its entirety.

## 3. Technical Approach

### 3.1 Reader Orientation

RAFT is a **training data construction recipe** — a specific procedure for assembling fine-tuning examples that teaches a pretrained LLM to answer questions by reading and reasoning over a mixture of relevant and irrelevant documents, citing verbatim evidence in chain-of-thought responses. The problem it solves is the misalignment between how domain-specific LLMs are typically trained (closed-book question-answering) and how they are deployed (open-book retrieval-augmented generation), and the shape of the solution is a principled mixing strategy: for `$P$`% of training examples, the model learns to extract answers from documents including the golden document plus distractors; for `$(1-P)$`% of examples, the model learns to answer from memory when the golden document is withheld, with both regimes using chain-of-thought reasoning with explicit source citations.

### 3.2 Big-Picture Architecture (Diagram in Words)

The RAFT system has four major components connected in a pipeline:

1. **Document Collection (`$D$`)** — the fixed set of domain-specific documents (e.g., Wikipedia articles, API documentation pages, biomedical abstracts) that constitutes the knowledge base for the target domain. These documents are the "open book" that will be available at test time via a retriever.

2. **Training Data Constructor** — the core RAFT innovation. For each question-answer pair in the domain-specific dataset, this component: (a) identifies the golden document(s) `$D^*$` from which the answer was derived, (b) samples `$k-1$` distractor documents `$D_i$` from the collection that are irrelevant to the question, (c) for a fraction `$P$` of examples, includes `$D^*$` in the context along with `$k-1$` distractors; for fraction `$(1-P)$`, includes only `$k$` distractors (no golden document), and (d) generates a chain-of-thought answer `$A^*$` that includes verbatim citations from the golden document.

3. **Fine-Tuning Engine** — standard supervised fine-tuning (SFT) on the constructed training data, using next-token prediction (causal language modeling loss) to train the base LLM to generate the CoT answer `$A^*$` conditioned on the question `$Q$` and the document set `$D_{context}$`. No architectural modifications; the base model (LLaMA2-7B-chat) is fine-tuned as-is.

4. **RAG Inference Pipeline** — at test time, a standard retriever (independent of RAFT's training) fetches the top-`$k$` documents for a given question from the domain document collection. These `$k$` documents (which may or may not include the golden document, and may include distractors) are prepended to the question, and the RAFT-trained model generates the answer, including reasoning and citations.

Information flows as follows: the domain document collection → training data constructor assembles `$Q$` + contextual documents → fine-tuning engine trains the LLM to produce `$A^*$` → at deployment, the retriever fetches documents from the same collection → RAFT-trained model processes the retrieved documents and generates an answer with citations.

### 3.3 Roadmap for the Deep Dive

- **First**, the training data construction procedure in full detail — the exact formulation of what the model sees as input and what it is trained to produce as output, including the mathematical specification of the data mixing strategy and the role of each component (golden documents, distractor documents, chain-of-thought with citations).
- **Second**, the two-regime mixing logic — why training with and without golden documents in different proportions matters, how the fraction `$P$` controls the tradeoff between memorization and reading, and how this directly addresses retriever failure modes.
- **Third**, the chain-of-thought answer generation — the specific format (citation delimiters, reasoning structure), how these answers are generated for training, and why explicit citation grounding matters for accuracy and robustness.
- **Fourth**, the distractor document strategy — how distractors are selected, how many are used during training versus testing, and the empirical investigation of how training-time distractor count affects robustness to varying test-time retrieval sizes.
- **Fifth**, the test-time deployment — how RAFT interfaces with any standard retriever, what the model receives at inference, and how the training recipe's design principles map to specific test-time behaviors.

### 3.4 Detailed, Sentence-Based Technical Breakdown

This is primarily a **training recipe paper** whose core idea is that the way fine-tuning data is constructed — specifically the presence or absence of golden documents, the inclusion of distractor documents, and the answer format — determines whether a domain-adapted LLM can effectively use retrieved documents at test time. The technical contribution is a principled data construction procedure, not a new model architecture or training objective.

---

#### Training Data Construction: The RAFT Recipe

The central technical mechanism of RAFT is the procedure for constructing each training example. The paper specifies this procedure precisely in Section 3, and it encodes all of the method's design principles.

**Input components per training example.** For each question `$q_i$` in the domain-specific dataset, the training data constructor assembles:

- **A question `$Q$`** — the natural language query that the model must answer (e.g., "What screenwriter with credits for 'Evolution', a film starring Nicolas Cage and Téa Leoni?").
- **A golden document `$D^*$`** (or documents `$D^*_1, D^*_2, \ldots$` for multi-hop datasets like HotpotQA) — the document(s) from the domain collection that contain the information necessary to answer the question. The paper explicitly notes that "the 'golden' document doesn't need to be a single document, but can be more than one document, as is the case in HotpotQA" (Section 3).
- **A set of `$k-1$` distractor documents `$D_1, D_2, \ldots, D_{k-1}$`** — documents sampled from the domain collection that do not contain answer-relevant information for this specific question. The paper uses `$k = 5$` total documents in the training context for most experiments (one golden plus four distractors, or five distractors when the golden document is withheld).
- **A chain-of-thought answer `$A^*$`** — the target output that the model is trained to generate, which includes a reasoning chain with verbatim citations from the golden document(s) delimited by `##begin_quote##` and `##end_quote##` tags, followed by the final answer delimited by `##Answer:`.

**The mixing strategy: two regimes defined by `$P$`.** The paper's key innovation is that not all training examples include the golden document. The construction procedure operates in two regimes controlled by a hyperparameter `$P$`, which represents the fraction of training examples where the golden document is present in the context:

> **For `$P$`% of the data:**
> `$Q + D^* + D_1 + D_2 + \ldots + D_{k-1} \rightarrow A^*$`

> **For `$(1-P)$`% of the data:**
> `$Q + D_1 + D_2 + \ldots + D_k \rightarrow A^*$`

In the second regime, the golden document `$D^*$` is deliberately excluded. The model receives only distractor documents and the question, yet is still trained to produce the correct answer `$A^*$`. The paper explains the rationale directly: "By removing the golden documents in some instances, we are compelling the model to memorize answers instead of deriving them from the context" (Section 3).

**What this means operationally.** For a training example in the first regime (golden present), the model sees the question followed by five documents — one of which contains the answer and four of which are irrelevant — and must learn to identify which document matters, extract the relevant evidence, and construct a reasoning chain grounded in that evidence. For a training example in the second regime (golden absent), the model sees the question followed by five documents, all irrelevant, and must fall back on parametric knowledge — the domain information it has internalized through seeing other examples during fine-tuning — to produce the correct answer. The model cannot extract the answer from the context because the context contains no answer-relevant information.

**The construction of `$A^*$` with chain-of-thought and citations.** The target answer `$A^*$` is not simply the final answer string. The paper specifies a structured format (visible in Figure 3 of the paper):

```
##Reason: {reasoning chain with citations} ##Answer: {final answer}
```

Within the reasoning chain, verbatim quotes from the golden document are enclosed in `##begin_quote##` and `##end_quote##` delimiters. For example, from the paper's Figure 4:

```
##Reason: The screenwriter with credits for the film "Evolution," starring Nicolas Cage and Téa Leoni, is David Weissman. This information is provided in the reference documents where it mentions David Weissman as a screenwriter with film credits including "The Family Man" (2000), "Evolution" (2001), and "When in Rome" (2010). Therefore, the screenwriter for "Evolution" is David Weissman. ##Answer: David Weissman
```

The paper does not train the model to produce the `##begin_quote##` / `##end_quote##` delimiters as an explicit separate objective — rather, these delimiters are part of the target text string, and the model learns to generate them through standard next-token prediction because they appear in the training targets. The presence of these explicit citation markers teaches the model to ground its reasoning in specific source text, which the paper argues improves both accuracy (by forcing evidence-based reasoning) and interpretability (by making the provenance of each claim transparent).

**How the CoT answers are generated for training.** The paper uses GPT-4-1106 to generate the chain-of-thought answers for the training data. Specifically, given a question, the golden document(s), and the ground-truth answer from the dataset, GPT-4-1106 is prompted to produce a reasoning chain that cites relevant evidence and arrives at the correct answer. The paper provides the prompt format in Figure 3 under the instruction:

> "Given the question, context and answer above, provide a logical reasoning for that answer. Please use the format of: ##Reason: {reason} ##Answer: {answer}."

This means the training data construction involves a teacher model (GPT-4-1106) that generates high-quality reasoning traces from the golden documents and correct answers, and the student model (LLaMA2-7B-chat) is then fine-tuned to reproduce those traces. The paper does not discuss whether using a stronger teacher model introduces a capability gap or whether the student model can fully internalize the reasoning patterns demonstrated by GPT-4.

**What the fine-tuning objective is.** The training uses standard supervised fine-tuning (causal language modeling / next-token prediction). Given the input sequence `$X = [Q, D_{context}]$` and the target sequence `$Y = A^*$`, the model is trained to minimize:

$$\mathcal{L} = -\sum_{t=1}^{|Y|} \log p_\theta(y_t | X, y_{<t})$$

where `$p_\theta$` is the model's predicted probability distribution over tokens, `$y_t$` is the `$t$`-th token of the target answer `$A^*$`, and `$y_{<t}$` represents all tokens in `$A^*$` before position `$t$`.

**What it computes:** the standard autoregressive language modeling loss — at each position in the target answer, the model predicts the next token given the question, the document context, and the previously generated tokens of the answer. The loss is the sum of the negative log-probabilities assigned to the correct tokens.

**Why this form:** the paper deliberately uses standard SFT rather than a specialized objective because the innovation is in the data construction, not the optimization. The model architecture and training procedure remain unchanged from standard fine-tuning, making RAFT a drop-in replacement for existing fine-tuning pipelines. Any specialized loss function (e.g., a contrastive loss that explicitly penalizes attending to distractor documents) would require architectural modifications and would not be as directly transferable across model families.

---

#### The Two-Regime Mixing Logic: Why `$P$` Matters

The most counterintuitive design decision in RAFT is that the training context does not always include the golden document. The paper dedicates Section 4.4 to investigating this choice, with the key finding that `$P = 100\%$` (always including the golden document) is **not** optimal. Let us understand why.

**The failure mode of `$P = 100\%$`.** If every training example includes the golden document in the context, the model learns a simple strategy: the answer is always extractable from the provided documents, so all I need to do is find the relevant passage and quote from it. This strategy works perfectly when the retriever succeeds at test time — the golden document is in the top-`$k$` retrieved results — but fails catastrophically when the retriever fails. If the golden document is not in the retrieved set, the model either: (a) extracts an answer from a distractor document, producing a confident but wrong response, or (b) fails to produce any answer because it has never been trained to operate when the context lacks the answer.

The paper's analogy captures this precisely: a student who only practices with the textbook open never learns what to do when the relevant chapter is missing from their exam copy. They haven't developed the backup skill of answering from memory.

**What `$P < 100\%$` teaches.** When a fraction `$(1-P)$` of training examples exclude the golden document, the model encounters situations where it must answer the question despite having no answer-relevant context. In these examples, the model is forced to draw on parametric knowledge — the domain information it has internalized from seeing other training examples, including those where the golden document was present. This creates a training signal that teaches the model a **fallback behavior**: try to extract the answer from the provided documents, but if no document contains the answer, answer from internal knowledge.

The paper's empirical investigation of `$P$` in Section 4.4 (Figure 5) reveals dataset-specific optimal values:

- **Natural Questions (NQ):** optimal `$P \approx 40\%$` — meaning 60% of training examples have **no golden document**. This is a surprisingly low figure, suggesting that for this dataset, the retriever frequently fails, and the model benefits substantially from being forced to memorize domain knowledge.
- **Trivia QA (TQA):** optimal `$P \approx 60\%$` — a more balanced mix.
- **HotpotQA:** optimal `$P \approx 100\%$` — the golden document is always included. The paper does not explicitly explain why HotpotQA differs, but a plausible interpretation (based on the multi-hop nature of HotpotQA) is that the questions are too complex to answer from parametric memory alone; the model must have the documents to perform the necessary multi-step reasoning, so forcing memorization is counterproductive.

The paper's conclusion from this investigation is stated in Section 4.4: "training your LLM without the correct corresponding context at times can be beneficial for the downstream task of answering questions related to the documents." The key phrase is "at times" — the proportion matters and is dataset-dependent.

**The interaction between `$P$` and test-time retrieval.** The two-regime mixing directly addresses a specific test-time scenario: what happens when the retriever fails to place the golden document in the top-`$k$` results? At test time, the model always receives `$k$` documents from the retriever, and the golden document may or may not be among them. The training procedure has prepared the model for both possibilities:

- If the golden document is present (test-time success case), the model has been trained on `$P$`% of examples to extract and cite evidence from the relevant document.
- If the golden document is absent (test-time failure case), the model has been trained on `$(1-P)$`% of examples to answer from parametric memory — the domain knowledge it internalized during fine-tuning.

This dual preparation is what RAFT means by "adapting to domain-specific RAG" — the model is robust to both retriever success and retriever failure, unlike prior approaches that fail in one regime or the other.

---

#### Chain-of-Thought Answer Generation with Verbatim Citations

The third major design element in RAFT is the format of the target answer `$A^*$`. Rather than training the model to output only the final answer string, RAFT trains the model to generate a complete reasoning chain that includes explicit citations from the source documents.

**The CoT + citation format.** The answer format (visible in the paper's Figure 3 and the qualitative example in Figure 4) follows a two-part structure:

```
##Reason: {natural language reasoning that references and quotes relevant passages}
##Answer: {concise final answer}
```

Within the `##Reason:` section, verbatim quotations from the golden document are enclosed in `##begin_quote##` and `##end_quote##` delimiters. The paper describes the answer as "a full reasoning chain and in-addition, clearly citing sources" (Section 3). The reasoning chain does not merely state the answer — it walks through the logical steps connecting the question to the cited evidence and arriving at the conclusion.

**Why CoT with citations helps.** The paper identifies two mechanisms through which the CoT format improves performance (Section 3 and the ablation in Table 2):

1. **Preventing overfitting to short answers.** The paper notes that "simply providing the answer to a question may not always be adequate. This approach can lead to a rapid decrease in loss, resulting in the model beginning to overfit" (Section 4.2). When the target is just a short answer string (e.g., "David Weissman" or "Delhi"), the fine-tuning loss converges quickly, but the model may not learn the underlying reasoning process — it merely learns to map question patterns to answer patterns. The longer CoT targets provide a richer training signal: the model must learn intermediate reasoning steps, evidence extraction, and logical deduction, not just surface-level pattern matching.

2. **Teaching evidence grounding.** The explicit citation delimiters force the model to connect its claims to specific passages in the source documents. This teaches a form of attribution — the model learns that answers should be supported by quotable evidence, which the paper argues improves accuracy by reducing hallucination and forcing the model to verify its reasoning against the provided text. The qualitative example in Figure 4 illustrates this: the DSF model (trained without CoT or citations) produces "The Family Man" (a film title) when asked for a screenwriter's name — it has extracted a related but wrong entity from the context. The RAFT model, trained with CoT and citations, correctly produces "David Weissman" and justifies it with a specific quotation from the golden document.

**How the CoT answers are generated for training data.** The paper uses GPT-4-1106 as the teacher model to generate the CoT answers. The training data construction follows these steps:

1. For each question in the training set, identify the golden document(s) and the ground-truth answer.
2. Present GPT-4-1106 with the question, the golden document(s), and the correct answer.
3. Prompt GPT-4-1106 with the instruction shown in Figure 3 to produce a reasoning chain that cites evidence and concludes with the answer.
4. Use this GPT-4-generated reasoning chain as the target `$A^*$` for fine-tuning LLaMA2-7B-chat.

The paper notes that the Gorilla APIBench dataset is an exception: "the Gorilla APIBench dataset, already includes reasoning in the answers" (Section 3), meaning the CoT answers for that dataset are pre-existing rather than GPT-4-generated.

**Quantitative evidence for CoT effectiveness.** Table 2 provides the ablation comparing RAFT with and without chain-of-thought. Key numbers:

| Dataset | RAFT w/o CoT | RAFT (with CoT) | Improvement |
|---|---|---|---|
| PubMed | 68.30 | 73.30 | +5.00 |
| HotpotQA | 25.62 | 35.28 | +9.66 |
| HuggingFace | 59.07 | 74.00 | +14.93 |
| Torch Hub | 86.56 | 84.95 | −1.61 |
| TensorFlow | 83.21 | 86.86 | +3.65 |

The CoT format provides substantial gains on most datasets, with the largest improvements on HotpotQA and HuggingFace — datasets that require multi-step reasoning or precise evidence extraction from technical documentation. The slight regression on Torch Hub suggests the CoT format may not help uniformly, though the paper does not discuss this case.

**An important subtlety about `P` and CoT.** When `$P < 100\%$` and a training example has no golden document in the context, the CoT answer `$A^*$` still includes citations — but these citations refer to a document that is **not present in the model's input context**. The model cannot verify the quoted text against its input. In these cases, the CoT serves a different purpose: it demonstrates the reasoning structure that the model should follow, and it teaches the model that answers can be justified with evidence (even if the evidence is not currently visible). The model must learn to produce similar reasoning from its parametric memory when the golden document is absent.

This creates an interesting training dynamic. In `$P$`% of examples, the model learns: "read the documents, find the relevant one, quote it, and reason to the answer." In `$(1-P)$`% of examples, the model learns: "the documents don't contain the answer, but I know the answer from my training, and I should still produce a reasoned response with citations (from the document I remember, not from what I see)." The model thus learns both reading comprehension and a form of memory-grounded reasoning that mimics the citation style even when operating without source documents.

---

#### The Distractor Document Strategy

The fourth major design element is the deliberate inclusion of distractor documents — documents that are irrelevant to the question — in the training context. The paper treats distractors not as an unfortunate artifact of imperfect retrieval but as a **training signal** that teaches the model to filter irrelevant information.

**Definition and selection of distractors.** Distractor documents are documents from the domain collection that do not contain information relevant to answering the specific question. The paper does not describe a sophisticated selection procedure — distractors are sampled from the document collection, with the only constraint being that they are not the golden document for that question. In the experiments, the number of distractor documents `$k-1$` is typically 4 (alongside 1 golden document, for a total of 5 training-context documents when the golden document is present, or 5 distractor documents only when `$P < 100\%$`).

**Why distractors during training matter.** The paper's Section 5 addresses this question directly. Without distractor documents during training, "finetuning with only the golden document frequently results in inferior performance compared to configurations that include a greater number of distractor documents" (Section 5.1). The mechanism is straightforward:

- **Training with only the golden document** teaches the model that every document in its context is relevant. The model never learns to distinguish useful documents from useless ones. At test time, when the retriever invariably includes irrelevant documents (because no retriever is perfect, and top-`$k$` retrieval trades precision for recall), the model treats all retrieved documents as equally relevant and may extract answers from distractors.
- **Training with distractors** teaches the model that some documents in its context are not helpful. The model must learn to evaluate each document's relevance to the question, attend primarily to the relevant one(s), and ignore the rest. This is exactly the skill needed at test time when the retriever produces a noisy document set.

The paper connects this to prior work showing that "LLMs can be easily distracted by irrelevant context" (citing Shi et al., 2023a; Weston & Sukhbaatar, 2023; Liu et al., 2023), positioning distractor training as a robustness intervention specifically targeting this known vulnerability.

**Empirical investigation of distractor count during training.** Figure 6 (Section 5.1) investigates how the number of distractor documents during training affects test-time performance. The key finding is that the optimal number of training distractors varies by dataset:

- **Natural Questions:** training with `$D^* + 3D$` (one golden + three distractors) yields the best performance.
- **HotpotQA:** training with `$D^* + 1D$` (one golden + one distractor) is optimal.
- **Training with only `$D^*$` (zero distractors)** consistently underperforms configurations with distractors across both datasets.

This dataset-specific variation suggests that the "right" amount of distractor noise during training depends on how much irrelevant content the retriever typically produces for that domain. A domain with a precise retriever may need fewer training distractors; a domain with a noisy retriever may need more.

**Generalization to varying test-time document counts.** The paper also investigates (Section 5.1, Figure 6) how models trained with different numbers of distractors generalize to test-time scenarios with varying numbers of retrieved documents (top-`$k$` where `$k$` ranges from 2 to 10). The results show that:

- Models trained with distractors maintain more stable performance across different test-time `$k$` values.
- Models trained only with the golden document ("Train D*") show performance that degrades more sharply as the number of test-time documents increases, confirming that distractor-free training makes the model brittle to additional noise at test time.

The paper's conclusion: "the inclusion of distractor documents during training indeed makes the model more resilient to fluctuations in the number of documents encountered during testing" (Section 5.1). This is an important practical finding because real-world RAG deployments vary `$k$` based on latency constraints, cost, and retrieval quality — a model that is robust to a range of `$k$` values is more deployable than one that requires a specific `$k$` to work well.

**The connection between distractors and the "lost in the middle" problem.** The paper's distractor strategy implicitly addresses the "lost in the middle" phenomenon (Liu et al., 2023), where LLMs fail to attend to information in the middle of long contexts. By training the model to identify and extract information from one relevant document amid several distractors, RAFT teaches a form of **selective attention** — the model learns to scan the document set, identify the document that contains query-relevant information, and focus its generation on that document. This skill transfers to test time, where the model must find the golden document (if present) among the retrieved set.

---

#### Test-Time Deployment: How RAFT Interfaces with Retrieval

RAFT is designed to be retriever-agnostic. The paper states explicitly: "RAFT is independent of the retriever used" (Section 3). This is both a strength (no retriever-specific constraints) and an important design property — the training recipe is separated from the retrieval infrastructure.

**What the model receives at test time.** At inference, the RAFT-trained model is presented with:

1. The question `$Q$`.
2. The top-`$k$` documents retrieved by whatever retriever is deployed in the RAG pipeline. The paper's experiments use a standard retriever (not described in detail, but implicitly a dense or sparse retriever matching the domain's document collection).

The model then generates an answer, which (due to its training) typically includes a reasoning chain with citations and a final answer. The paper does not prescribe a specific `$k$` for test time; the experiments in Section 5.1 investigate `$k$` values from 2 to 10, and the training with distractors ensures robustness across this range.

**Why training documents and test documents come from the same distribution.** This is a crucial constraint that distinguishes RAFT from general RAG fine-tuning approaches like RA-DIT (Lin et al., 2023a). RAFT assumes the domain is **fixed and known in advance** — the same document collection is used for training data construction (as the source of golden and distractor documents) and for test-time retrieval. This means the model learns domain-specific patterns: the writing style, the typical document structure, the kinds of distractors common in that domain, and the formatting conventions of citations and answers. The paper explicitly contrasts this with prior work: "in their settings, at test time, the domain or documents can be different than the training time; whereas our paper studies a slightly opposite scenario where we only care about testing the LLM on the same set of documents" (Section 6).

**The two test-time behaviors RAFT enables.** Because of the two-regime training with `$P < 100\%$`, the model has learned two complementary behaviors that it can deploy at test time:

- **When the retriever succeeds** (golden document is in the top-`$k$`): the model identifies the relevant document, extracts evidence, cites it, and reasons to the answer. This is the behavior learned from the `$P$`% of training examples with golden documents present.
- **When the retriever fails** (golden document is absent from the top-`$k$`): the model falls back on parametric knowledge — the domain information it internalized during fine-tuning — and still produces a reasoned answer (potentially with citations from memory). This is the behavior learned from the `$(1-P)$`% of training examples where the golden document was withheld.

A standard DSF model can only do the second behavior (and does it without citations or reasoning chains). A standard RAG model can only attempt the first behavior (but without having been trained on domain-specific documents, so it does it poorly). RAFT combines both, with the proportion `$P$` controlling the balance between them.

**What happens when the distractor count at test time differs from training.** The paper's Section 5.1 (Figure 6) addresses this directly. Training with a fixed number of distractors (e.g., `$D^* + 3D$`) generalizes to test-time scenarios with different document counts (e.g., top-2 through top-10). Performance is not maximized at the exact training-time document count but rather remains relatively stable across a range, especially compared to distractor-free training ("Train D*") which shows steep performance cliffs as test-time documents increase. This generalization property is important because real-world retrieval systems often return variable numbers of documents, and retraining for every possible `$k$` is impractical.

---

#### Design Choices and Their Justifications: A Summary

RAFT's four design principles — each motivated by a specific failure mode of prior approaches — are:

1. **Documents in training context** (Q + D → A): fixes the training-deployment misalignment where models fine-tuned without documents cannot effectively use them at test time.

2. **Distractor documents during training**: teaches the model to filter irrelevant content, addressing the known vulnerability of LLMs to distraction by irrelevant context and preparing the model for the noisy outputs of real-world retrievers.

3. **Withholding golden documents for a fraction `$(1-P)$` of examples**: teaches the model a fallback-to-memory behavior for when the retriever fails, addressing the brittleness of pure RAG approaches that assume perfect retrieval.

4. **Chain-of-thought answers with verbatim citations**: prevents overfitting to short answers, teaches evidence grounding and attribution, and provides a richer training signal that improves accuracy and interpretability.

These design choices are not independent — they combine to create a training distribution that mirrors the test-time distribution in all relevant aspects: the presence of noise (distractors), the possibility of retrieval failure (absent golden documents), and the expected output format (reasoned, cited answers). The paper's empirical results (Table 1) demonstrate that each component contributes to the overall performance improvement, and the ablation in Table 2 quantifies the specific contribution of the chain-of-thought format.

## 4. Key Insights and Innovations

### Innovation 1: The Training-Deployment Alignment Problem as the Root Cause of RAG Failures

The paper's most important conceptual contribution is not a new algorithm but a **diagnosis**. Prior work treated poor RAG performance in domain-specific settings as either a retrieval quality problem (the retriever isn't finding the right documents) or a model capability problem (the LLM isn't strong enough at reading comprehension). RAFT identifies a third, more fundamental cause: **the model was trained for a different task than the one it performs at deployment**.

This is a reframing of the problem that shifts the burden of explanation from components (retriever, model scale) to process (the training-to-inference misalignment). Consider the paper's central example in Table 1: domain-specific fine-tuning on HotpotQA achieves 6.38% accuracy, but adding RAG at test time drops performance to 4.41%. A retriever-quality explanation would predict that adding relevant documents should help; a model-capability explanation would predict that LLaMA2-7B is simply too weak to do multi-hop reasoning. Neither predicts the **regression** — that providing the answer-containing document makes the model perform *worse* than giving it nothing. The regression only makes sense if the model has actively learned something during fine-tuning that is incompatible with document-augmented inference: specifically, the model learned to answer from parametric memory and never practiced reading, so the introduction of document context at test time is a distributional shift that degrades rather than aids performance.

This diagnosis matters because it explains a pattern of contradictory results in the broader literature. Prior work found that LLMs can be "easily distracted by irrelevant context" (Shi et al., 2023a) and that self-correction or iterative refinement often fails on reasoning tasks (Huang et al., 2023). Other work found that fine-tuning for RAG can improve performance (Lin et al., 2023a; Wang et al., 2023). RAFT's framing reconciles these: the negative results occur when the training procedure does not match the test-time RAG setting, and the positive results occur when it does — but prior work had not made this alignment the explicit optimization target.

The "open-book exam" analogy in Figure 1 crystallizes this insight. The analogy is not merely illustrative; it encodes a precise claim about what prior approaches miss: studying for an open-book exam requires *practicing with the book open*, not memorizing the book (DSF) or showing up unprepared and trying to use the book for the first time (base model + RAG). This reframes the adaptation problem from "how do we inject domain knowledge?" to "how do we prepare the model for the exact inference conditions it will face?" The shift from knowledge injection to process alignment is the paper's fundamental conceptual move, and it is what makes the specific design choices in RAFT (documents in context, distractors, withheld golden documents) coherent as a unified recipe rather than an arbitrary collection of tricks.

The significance of this diagnosis is that it provides a **portable principle** beyond the specific RAFT recipe. Any domain adaptation effort — whether for legal documents, medical records, or code repositories — should begin by asking: does the training procedure mirror the test-time procedure in terms of document presence, noise characteristics, and failure modes? If not, the resulting model may exhibit the same regressions documented in Table 1. The paper's contribution here is a diagnostic framework validated by systematic empirical demonstration, not just a method that happens to work.

---

### Innovation 2: Retrieval Failure as a First-Class Training Objective

The second conceptual innovation is the deliberate inclusion of **retrieval failure during training** — the withholding of the golden document for a fraction `$(1-P)$` of examples. This inverts a deeply held assumption in the RAG fine-tuning literature: that training examples should always include the correct source document, because the point of RAG is to teach the model to use retrieved documents.

The dominant assumption in prior RAG fine-tuning work (e.g., RA-DIT by Lin et al., 2023a; InstructRetro by Wang et al., 2023) is that the model should be trained to maximize its ability to extract answers from provided documents. In such approaches, every training example includes relevant documents because the goal is to maximize reading comprehension performance. RAFT's counterintuitive move is to argue that this assumption is not just suboptimal but actively harmful in domain-specific settings — because it produces models that are brittle to the inevitable failures of real-world retrievers.

The empirical evidence for this claim is in Figure 5 (Section 4.4): `$P = 100\%$` (always including the golden document) is **never the uniquely best configuration** across the three datasets examined. For Natural Questions, the optimum is `$P \approx 40\%$`, meaning the golden document is absent from **60%** of training examples. This is a radical result. It says that for this dataset, the model performs better when it is forced to answer from memory most of the time during training — even though the test-time setting is explicitly open-book. The implication is that the model's parametric memory of the domain, built up across many fine-tuning examples, is more reliable than its ability to extract answers from documents, at least for the questions in this dataset. The withheld-golden-document examples teach the model that "the documents might not contain the answer, and when they don't, I should trust my training, not hallucinate from distractors."

What makes this intellectually distinctive is that it treats retrieval failure not as an unfortunate edge case to be tolerated but as a **central design parameter** of the training recipe. The quantity `$P$` is a knob that controls the model's inductive bias toward document-reliance versus memory-reliance. In the extreme `$P = 100\%$`, the model becomes a pure document reader — accurate when the retriever works, useless when it fails. In the extreme `$P = 0\%$`, the model becomes a pure memorizer — accurate on whatever it has internalized, ignoring documents entirely. The optimum `$P$` for a given domain depends on the inherent retrievability of questions in that domain: if most questions have easily retrievable answers, high `$P$` is appropriate; if retrieval is hard (documents are long, relevant passages are buried, queries are underspecified), low `$P$` forces the model to fall back on parametric knowledge more aggressively.

This framework reframes the RAG adaptation problem as a **mixture-of-strategies learning problem**, where the model must learn a meta-policy: "first check whether any document contains the answer; if yes, extract it; if no, answer from memory." The two-regime training data construction implements this meta-policy without explicit architectural overhead — the model learns the switch implicitly through the distribution of training examples. The finding that this implicit meta-learning outperforms both pure-reading and pure-memory training is the empirical validation of the framework.

The comparison to prior work is stark. Standard DSF (domain-specific fine-tuning without documents) is the `$P = 0\%$` extreme — the model never sees documents during training and thus cannot use them at test time (hence the DSF + RAG regressions in Table 1). Standard RAG fine-tuning with always-present relevant documents is the `$P = 100\%$` extreme — the model always expects the answer to be extractable and has no fallback. RAFT's key insight is that neither extreme is correct for realistic deployment; the optimum lies in between, and that optimum is dataset-specific and empirically discoverable.

This is a **fundamental shift** in how to think about training for RAG, not a minor refinement. Prior work asked: "how do we make the model better at using retrieved documents?" RAFT asks: "given that retrieval is imperfect and sometimes fails entirely, what mix of document-reliance and memory-reliance should the model learn?" The distinction is between optimizing for the best case (perfect retrieval) and optimizing for the expected case (noisy, imperfect retrieval with possible failures). RAFT is the first work to make this distinction explicit and to provide a training recipe that directly addresses it.

---

### Innovation 3: Distractor Documents as a Training Signal, Not Just a Test-Time Nuisance

Prior work on LLMs and irrelevant context treats distractors as a **vulnerability** — something models fail to handle correctly, and therefore something to be mitigated through better retrieval (to reduce the number of distractors) or better prompting (to help the model ignore them). The literature on "LLMs can be easily distracted by irrelevant context" (Shi et al., 2023a), the "lost in the middle" phenomenon (Liu et al., 2023), and System 2 Attention (Weston & Sukhbaatar, 2023) all share this framing: irrelevant context is a problem to be solved by minimizing its presence or engineering around it.

RAFT inverts this framing. Distractor documents are deliberately included during training not as an unavoidable nuisance but as a **positive training signal** — a necessary component of the curriculum that teaches the model to filter, select, and attend discriminatively. The paper shows in Figure 6 that training with only the golden document (zero distractors) produces models that perform worse at test time than models trained with distractors, even when controlling for all other factors. This is not because distractors make the training task easier — they demonstrably make it harder by introducing noise — but because they teach a skill that the test-time setting demands.

The conceptual move here is from **distractors as environment** (something the deployment setting contains and the model must tolerate) to **distractors as curriculum** (something deliberately introduced during training to teach a specific capability). This is analogous to the role of noise in robust optimization or adversarial training in computer vision: adding perturbations during training degrades training performance but improves generalization and robustness. RAFT applies this principle to the text domain, with the specific claim that the capability being taught is **discriminative reading** — the ability to scan a set of documents, identify which one(s) contain query-relevant information, and selectively attend to those while suppressing the rest.

What makes this a genuine innovation rather than an obvious trick is the empirical finding in Figure 6 that the optimal number of training distractors is dataset-specific (4 for Natural Questions, 2 for HotpotQA) and that training with distractors enables **generalization to unseen test-time document counts**. A model trained with 4 distractors performs well at test time with anywhere from 2 to 10 retrieved documents; a model trained with zero distractors degrades sharply as the number of test-time documents increases. This generalization property is non-obvious — one might expect the model to overfit to the specific training-time document count — and it suggests that distractors during training teach a general filtering capability rather than a count-specific strategy.

The significance of this insight extends beyond RAFT's specific recipe. It suggests that for any RAG deployment, the training data should be constructed to match not only the retriever's successes (relevant documents present) but also its noise characteristics (irrelevant documents present, in proportions that reflect the retriever's precision-recall tradeoff). A high-recall, low-precision retriever (which returns many documents to ensure the golden document is included) demands more distractor training than a high-precision, low-recall retriever. The distractor count becomes a design parameter that should be tuned to the specific retriever being deployed — a connection the paper hints at but does not fully explore.

---

### Innovation 4: Chain-of-Thought as a Regularizer Against Fine-Tuning Overfitting

The paper's ablation in Table 2 reveals a finding that, while not as prominently featured as the `$P$` and distractor innovations, represents an important conceptual contribution about the role of chain-of-thought in fine-tuning: **CoT with explicit citations functions as a regularizer against overfitting during domain adaptation**.

The standard view of chain-of-thought (Wei et al., 2022) is that it improves reasoning accuracy by encouraging the model to decompose complex problems into intermediate steps — essentially, it is a prompting technique that elicits better reasoning from a frozen model. RAFT's use of CoT is different: it embeds CoT in the fine-tuning targets, training the model to produce reasoning chains as part of its learned behavior. The paper's observation (Section 4.2) that "simply providing the answer to a question may not always be adequate. This approach can lead to a rapid decrease in loss, resulting in the model beginning to overfit" reframes CoT as a **loss-landscape intervention**, not just a reasoning aid.

The mechanism is straightforward but conceptually elegant. When fine-tuning targets are short answer strings (e.g., "Delhi" or "David Weissman"), the model can achieve low training loss by learning superficial pattern matches between question templates and answer tokens — it doesn't need to learn the underlying evidence-extraction reasoning. The loss decreases rapidly, the model appears to converge, but test-time performance is poor because the superficial patterns don't transfer to novel questions. When targets include a full reasoning chain with citations, the model must learn to attend to the document content, extract relevant evidence, and construct a coherent justification — a much richer task that requires deeper processing of the input. The loss decreases more slowly, but what the model learns transfers better.

The quantitative evidence in Table 2 supports this interpretation. The largest CoT gains appear on datasets requiring multi-step reasoning (HotpotQA: +9.66%) and precise evidence extraction from technical documents (HuggingFace: +14.93%). The losses are small or nonexistent on datasets where answers are short and factual (PubMed: +5.00%, Torch Hub: −1.61%). This pattern aligns with the regularization interpretation: CoT provides the most benefit where superficial pattern matching would otherwise dominate, and provides the least benefit where the answer format is already sufficiently informative to prevent overfitting.

The citation delimiters (`##begin_quote##` / `##end_quote##`) within the CoT serve an additional role that the paper does not frame as regularization but that functions similarly: they act as a **grounding constraint** that ties the model's reasoning to observable evidence in the input. A model that learns to copy-paste relevant passages (with citation markers) before reasoning about them is less likely to hallucinate or rely on parametric priors that conflict with the provided documents. This is a form of behavioral regularization — the output format itself constrains the model's generation toward evidence-grounded responses.

This framing of CoT-as-regularizer is a conceptual contribution distinct from the original CoT literature. Wei et al. (2022) showed that CoT improves frozen-model reasoning; RAFT shows that CoT improves fine-tuning generalization. The two findings are complementary but mechanistically distinct, and RAFT's contribution opens the door to thinking about output format design as a deliberate tool for controlling the generalization properties of fine-tuned models — not just for improving their reasoning in deployment.

## 5. Experimental Analysis

### Evaluation Methodology

- **Dataset.** The paper evaluates on five datasets spanning three domain categories: (1) **PubMed QA** (Jin et al., 2019) — a biomedical question-answering dataset focused on answering medical and biology questions based on provided documents, using binary yes/no questions; (2) **HotpotQA** (Yang et al., 2018) — a multi-hop open-domain QA dataset based on Wikipedia, requiring reasoning across multiple documents to answer questions about entities, events, and relationships; (3) **Gorilla APIBench** (Patil et al., 2023) — consisting of three sub-datasets: **HuggingFace Hub**, **Torch Hub**, and **TensorFlow Hub**, each measuring the ability to generate correct, functional API calls based on documentation. Additionally, for the hyperparameter investigation in Section 4.4, the paper uses **Natural Questions (NQ)** (Kwiatkowski et al., 2019) and **Trivia QA** (Joshi et al., 2017). The paper does not explicitly report training/validation/test split sizes for each dataset; it references the standard splits or the evaluation protocol from the original dataset papers. The domain specificity varies: PubMed and Gorilla APIBench represent genuinely specialized domains (biomedical literature, API documentation), while HotpotQA and NQ/Trivia QA represent open-domain Wikipedia-based QA used to study hyperparameter sensitivity.

- **Base model(s).** All experiments use **LLaMA2-7B-chat** (the instruction-tuned variant of Meta's LLaMA2-7B) as the base model for fine-tuning. The paper chooses this model because it is a widely used open-source instruction-tuned LLM representative of the capabilities available to practitioners deploying domain-specific RAG systems. For reference comparison (not as a base model for RAFT), the paper includes **GPT-3.5** (via API, with RAG) in Table 1 to provide an upper bound from a substantially larger and more capable model. The choice of a 7B parameter model is deliberate: it is small enough to fine-tune on modest hardware, and the paper aims to show that RAFT can dramatically improve a modest base model's domain-specific RAG performance without requiring larger model scales.

- **Metrics.** The paper reports **accuracy** as the primary metric — the fraction of test questions for which the model's generated answer matches the ground-truth answer. For PubMed QA, this is binary classification accuracy (yes/no). For HotpotQA, this is exact match or F1 against the ground-truth answer span (the paper does not specify which; the results are reported as percentages in Table 1). For the Gorilla APIBench datasets (HuggingFace, Torch Hub, TensorFlow Hub), accuracy means generating the correct, functional API call that matches the ground-truth API invocation. The paper does not report confidence intervals, standard deviations, or statistical significance tests for any of the reported numbers.

- **Baselines.** The paper compares RAFT against four baseline configurations, all using LLaMA2-7B-chat as the backbone (except the GPT-3.5 reference):
  - **LLaMA2-7B-chat + 0-shot prompting:** The base instruction-tuned model, given the question with a clearly written instruction but no reference documents. This represents the "closed-book" baseline where the model relies entirely on its parametric knowledge.
  - **LLaMA2-7B-chat + RAG (Llama2 + RAG):** The base model with retrieved documents prepended to the question at test time. This represents the standard RAG pipeline applied to a model that was not domain-adapted.
  - **Domain-Specific Fine-tuning with 0-shot prompting (DSF):** LLaMA2-7B-chat fine-tuned on domain QA pairs (Q → A, without documents in context), then evaluated without RAG. This represents the "memorization" approach — studying the textbook but not practicing with it open.
  - **Domain-Specific Fine-tuning with RAG (DSF + RAG):** The DSF model evaluated with retrieved documents added at test time. This represents the combination of domain adaptation and retrieval, but where the training and deployment conditions are misaligned (trained closed-book, tested open-book).
  - **GPT-3.5 + RAG:** A reference point using a larger, more capable model with retrieval, included in Table 1 to contextualize the performance levels achievable.

- **Generation budget / compute accounting.** The paper does not report generation budgets in terms of FLOPs, tokens generated, or wall-clock time. The "compute" for RAFT is measured implicitly through the training data construction cost (generating CoT answers with GPT-4-1106, sampling distractor documents) and the fine-tuning cost (standard SFT on the constructed dataset). For test-time inference, all methods use a single generation per question (greedy decoding or temperature-based sampling is not specified). The paper does not perform a FLOPs-matched comparison between RAFT and the baselines — the comparison is purely accuracy-based at a fixed model scale (7B parameters), with training cost treated as a one-time upfront investment not accounted for in the evaluation. This is a significant omission for practitioners weighing the cost-benefit of RAFT against simpler approaches.

- **Cross-validation / statistical protocol.** The paper does not describe a cross-validation procedure, statistical significance testing, or multiple-seed averaging. All results in Table 1, Table 2, and the figures appear to be single-run numbers. The hyperparameter investigation of `$P$` in Figure 5 (Section 4.4) sweeps `$P$` values across {0%, 20%, 40%, 60%, 80%, 100%} on three datasets (NQ, Trivia QA, HotpotQA), but does not report variance across runs. The test-time document count experiments in Figure 6 (Section 5.1) sweep training distractor configurations (D*, D*+1D, D*+2D, D*+3D) and test-time top-k values (2 through 10), also without reported error bars. This lack of statistical rigor makes it difficult to assess whether the reported differences — particularly for close comparisons like PubMed's 71.6 (DSF+RAG) vs. 73.30 (RAFT) — are statistically reliable or within noise.

### Main Quantitative Results

#### RAFT vs. Baselines Across Five Datasets (Table 1)

The central quantitative result is presented in Table 1, which compares RAFT (LLaMA2-7B fine-tuned with the RAFT recipe) against the four baselines and GPT-3.5+RAG across five datasets. The headline finding is that RAFT achieves the highest accuracy on every dataset, often by substantial margins.

**On PubMed QA (biomedical yes/no questions):**
- LLaMA2-7B (0-shot): 56.5%
- LLaMA2-7B + RAG: 58.8%
- DSF: 59.7%
- DSF + RAG: 71.6%
- **RAFT: 73.30%**
- GPT-3.5 + RAG: 71.60%

The RAFT gain over DSF+RAG is modest (+1.7 percentage points), and the paper notes this explicitly: "for PubMed QA, since it is a binary yes/no question, we don't observe significant gains when we compare our model with DSF + RAG" (Section 4.1). The binary nature of PubMed QA means that even DSF+RAG with imperfect document processing can achieve high accuracy through simple heuristics, leaving limited headroom for RAFT's improvements. Both RAFT and DSF+RAG slightly outperform GPT-3.5+RAG (73.30 vs. 71.60), which is notable given the model scale difference.

**On HotpotQA (multi-hop Wikipedia QA):**
- LLaMA2-7B (0-shot): 0.54%
- LLaMA2-7B + RAG: 0.03%
- DSF: 6.38%
- DSF + RAG: 4.41%
- **RAFT: 35.28%**
- GPT-3.5 + RAG: 41.5%

This is the dataset where RAFT's impact is most dramatic. The base model with RAG performs near zero (0.03%) — worse than the closed-book baseline (0.54%), confirming the distraction problem. DSF alone achieves 6.38%, and adding RAG drops performance to 4.41% — the training-deployment misalignment regression the paper diagnoses. RAFT achieves 35.28%, a **30.87 percentage point improvement over DSF** and a **30.87 point improvement over DSF+RAG**. This is a qualitative jump from near-useless to moderately functional, on a model (7B parameters) that prior approaches rendered effectively non-functional on this benchmark. RAFT does not match GPT-3.5+RAG (41.5%), but the gap (6.22 points) is dramatically smaller than the baseline gaps, suggesting RAFT recovers a substantial fraction of the larger model's capability through better training-data alignment.

**On HuggingFace Hub (API call generation):**
- LLaMA2-7B (0-shot): 0.22%
- LLaMA2-7B + RAG: 26.43%
- DSF: 61.06%
- DSF + RAG: 42.59%
- **RAFT: 74.00%**
- GPT-3.5 + RAG: 29.08%

This dataset reveals the most striking regression pattern. DSF alone achieves 61.06%, but adding RAG drops performance catastrophically to 42.59% — a loss of 18.47 percentage points from providing the model with the very documents that contain the answers. This is the strongest evidence for the paper's training-deployment misalignment thesis. RAFT, which trains with documents in context, achieves 74.00% — a **31.41 point improvement over DSF+RAG** and a **12.94 point improvement over DSF without RAG**. RAFT also dramatically outperforms GPT-3.5+RAG (74.00 vs. 29.08), confirming that domain-specific adaptation with properly aligned training is more effective than scaling to a larger general-purpose model with unadapted RAG.

**On Torch Hub (API call generation):**
- LLaMA2-7B (0-shot): 0%
- LLaMA2-7B + RAG: 8.60%
- DSF: 84.94%
- DSF + RAG: 82.80%
- **RAFT: 84.95%**
- GPT-3.5 + RAG: 60.21%

Here, DSF already achieves near-ceiling performance (84.94%), and RAFT matches it at 84.95% — a 0.01 point difference. The DSF+RAG regression is present but small (84.94% → 82.80%, a 2.14 point drop), suggesting that for Torch Hub, the model can largely answer from parametric memory and the documents add noise without destroying performance. RAFT does not harm performance (maintaining the DSF level) while adding the robustness benefits demonstrated on other datasets. The gap over GPT-3.5+RAG is large (84.95 vs. 60.21), reinforcing the domain-specific adaptation advantage.

**On TensorFlow Hub (API call generation):**
- LLaMA2-7B (0-shot): 0%
- LLaMA2-7B + RAG: 43.06%
- DSF: 86.56%
- DSF + RAG: 60.29%
- **RAFT: 86.86%**
- GPT-3.5 + RAG: 65.59%

TensorFlow Hub shows a severe DSF+RAG regression (86.56% → 60.29%, a 26.27 point drop), one of the largest in the table. RAFT not only recovers this loss but slightly exceeds DSF alone (86.86 vs. 86.56), achieving the highest accuracy in the table. The pattern — large DSF+RAG regression, RAFT recovery — mirrors HuggingFace and HotpotQA, suggesting that when document-augmented inference is harmful to a DSF model, RAFT's training alignment is most valuable.

#### The DSF+RAG Regression Phenomenon (Table 1, Cross-Dataset)

A pattern that emerges from reading across the rows of Table 1 is the consistent **degradation when adding RAG to a DSF model** on several datasets:

| Dataset | DSF | DSF + RAG | Change |
|---|---|---|---|
| PubMed | 59.7 | 71.6 | +11.9 |
| HotpotQA | 6.38 | 4.41 | −1.97 |
| HuggingFace | 61.06 | 42.59 | −18.47 |
| Torch Hub | 84.94 | 82.80 | −2.14 |
| TensorFlow | 86.56 | 60.29 | −26.27 |

PubMed is the only dataset where RAG helps a DSF model. On the other four, adding documents degrades performance, with HuggingFace and TensorFlow showing catastrophic drops (>18 points). This pattern is the paper's core empirical motivation: it demonstrates that standard domain adaptation (DSF) and standard RAG are not composable — combining them produces worse results than either alone on three of five datasets. RAFT's consistent improvement over DSF+RAG (and usually over DSF alone) is the paper's evidence that the training recipe solves this composability problem.

#### Chain-of-Thought Ablation (Table 2)

Table 2 isolates the contribution of the chain-of-thought format with verbatim citations. The comparison is between RAFT (full recipe, with CoT) and "RAFT w.o CoT" (same training data construction but with short answers as targets instead of reasoning chains with citations):

| Dataset | RAFT w.o CoT | RAFT (with CoT) | CoT Gain |
|---|---|---|---|
| PubMed | 68.30 | 73.30 | +5.00 |
| HotpotQA | 25.62 | 35.28 | +9.66 |
| HuggingFace | 59.07 | 74.00 | +14.93 |
| Torch Hub | 86.56 | 84.95 | −1.61 |
| TensorFlow | 83.21 | 86.86 | +3.65 |

The CoT format provides substantial gains on the datasets where RAFT's improvements over baselines are largest (HotpotQA: +9.66, HuggingFace: +14.93). The paper interprets this as CoT preventing overfitting: "simply providing the answer to a question may not always be adequate. This approach can lead to a rapid decrease in loss, resulting in the model beginning to overfit" (Section 4.2). The slight regression on Torch Hub (−1.61) is the only negative result and receives no specific discussion in the paper. The pattern suggests CoT is most valuable when the reasoning required is complex (multi-hop on HotpotQA, API selection with documentation on HuggingFace) and less important when answers are simple factual lookups or when DSF already achieves high accuracy.

#### Hyperparameter Sensitivity: The `$P$` Proportion (Figure 5, Section 4.4)

The investigation of `$P$` (the fraction of training examples that include the golden document) is a central empirical contribution because it validates the paper's claim that withholding the golden document for some training examples is beneficial. Figure 5 sweeps `$P$` from 0% to 100% on three datasets: Natural Questions (NQ), Trivia QA (TQA), and HotpotQA. The key findings:

- **Natural Questions:** The optimal `$P$` is approximately **40%**, with accuracy peaking around 0.44 at `$P = 40\%$` and declining to roughly 0.30 at `$P = 100\%$`. This means the model performs best when the golden document is absent from **60%** of training examples — a striking result that strongly supports the paper's claim about the importance of training for retrieval failure.
- **Trivia QA:** The optimal `$P$` is approximately **60%**, with accuracy peaking around 0.63 at `$P = 60\%$` and declining to roughly 0.56 at `$P = 100\%$`.
- **HotpotQA:** The optimal `$P$` is **100%**, with accuracy monotonically increasing with `$P$` from roughly 0.40 at `$P = 0\%$` to roughly 0.56 at `$P = 100\%$`. The paper does not discuss why HotpotQA bucks the trend, but a plausible interpretation (not stated in the paper) is that HotpotQA's multi-hop questions require synthesizing information across documents, making it impossible to answer from memory — the model must have the documents to perform the necessary reasoning, so withholding them only creates impossible training examples that add noise.

The paper's conclusion — "training your LLM without the correct corresponding context at times can be beneficial for the downstream task" (Section 4.4) — is supported for NQ and TQA but explicitly contradicted for HotpotQA, where the optimum is at the extreme of always including the golden document. This dataset-dependence is an important qualification that the paper acknowledges but does not explain mechanistically.

#### Distractor Document Robustness (Figure 6, Section 5.1)

Figure 6 investigates two related questions: (1) how does the number of distractor documents during training affect test-time performance, and (2) how well do models generalize to varying numbers of test-time documents? The experiments sweep training configurations (D* only, D*+1D, D*+2D, D*+3D) and test-time document counts (top-2 through top-10).

**Training distractor count (within each subplot of Figure 6):**
- For **Natural Questions**, training with D*+3D (one golden + three distractors) yields the highest accuracy across most test-time `$k$` values. Training with only D* consistently underperforms all distractor-inclusive configurations.
- For **HotpotQA**, training with D*+1D (one golden + one distractor) is optimal. Training with D*+3D performs worse than D*+1D at most test-time `$k$` values, suggesting that too many distractors during training can be detrimental on this dataset.

The consistent finding is that **training with zero distractors ("Train D*") is never the best configuration** — adding at least one distractor document during training improves test-time performance on both datasets.

**Generalization across test-time document counts (across the x-axis of each subplot):**
- Models trained with distractors maintain relatively stable accuracy as test-time `$k$` varies from 2 to 10. The curves are relatively flat.
- Models trained with only D* ("Train D*") show steeper performance degradation as `$k$` increases, particularly on Natural Questions where accuracy drops from roughly 0.30 at `$k = 2$` to roughly 0.24 at `$k = 10$`.
- On HotpotQA, the degradation is less severe but still present.

The paper interprets this as evidence that "the inclusion of distractor documents during training indeed makes the model more resilient to fluctuations in the number of documents encountered during testing" (Section 5.1). This is a robustness result with practical significance: it means RAFT-trained models do not require careful tuning of `$k$` at deployment time — they perform well across a range of retrieval depths.

#### Qualitative Evidence (Figure 4)

The paper provides one concrete qualitative example comparing RAFT and DSF on a HotpotQA question: "What screenwriter with credits for 'Evolution', a film starring Nicolas Cage and Téa Leoni?" The documents include two relevant passages: one establishing David Weissman as a screenwriter with credits including "Evolution," and another describing "The Family Man" as co-written by David Weissman and David Diamond, starring Nicolas Cage and Téa Leoni. The RAFT model correctly answers "David Weissman" with a reasoning chain citing the first passage. The DSF model incorrectly answers "The Family Man" — a film title — confusing the entity type being asked for (screenwriter vs. film). The paper presents this as evidence that DSF models "extract the wrong information from the context" while RAFT "manages to get the accurate results" (Section 4.3).

### Ablation Studies and Robustness Checks

- **Chain-of-Thought format ablation (Table 2):** Removing the chain-of-thought reasoning with verbatim citations from the training targets reduces performance across four of five datasets, with drops of 9.66 points on HotpotQA and 14.93 points on HuggingFace. The CoT format matters most on datasets requiring complex multi-step reasoning or precise evidence extraction. The one exception is Torch Hub, where CoT slightly decreases performance (−1.61 points) — a finding the paper does not analyze or explain.

- **Training-time golden document proportion `$P$` (Figure 5):** Sweeping `$P$` from 0% to 100% reveals that the optimal proportion is dataset-specific: ~40% for Natural Questions, ~60% for Trivia QA, and 100% for HotpotQA. The `$P = 100\%$` configuration (always including the golden document, equivalent to standard RAG fine-tuning) is never uniquely optimal and is substantially suboptimal for NQ (roughly 0.30 vs. 0.44 at `$P = 40\%$`). This validates that training for retrieval failure by withholding golden documents is beneficial for some domains.

- **Training distractor count (Figure 6):** Comparing training with D* only vs. D* + {1,2,3}D shows that training without distractors consistently underperforms training with at least one distractor document. The optimal number of training distractors is dataset-specific: D*+3D for Natural Questions, D*+1D for HotpotQA. This confirms that distractor exposure during training is necessary for robust test-time performance.

- **Test-time document count generalization (Figure 6):** Models trained with distractors maintain stable performance as test-time `$k$` varies from 2 to 10. Models trained without distractors degrade more sharply, particularly on Natural Questions. This demonstrates that distractor training produces robustness to variable retrieval depths, not just improved performance at a specific `$k$`.

- **Comparison of training data construction with and without documents (DSF vs. RAFT, Table 1):** While not presented as a formal ablation, the comparison between DSF+RAG and RAFT across datasets effectively ablates the presence of documents in training context. The consistent gap (PubMed: +1.7, HotpotQA: +30.87, HuggingFace: +31.41, Torch Hub: +2.15, TensorFlow: +26.57) confirms that training with documents in context is essential for models that will use RAG at test time.

- **Negative result: HotpotQA's `$P = 100\%$` optimum (Figure 5):** For HotpotQA, the optimal training configuration includes the golden document 100% of the time — the withholding strategy that helps on NQ and TQA provides no benefit and is actually harmful on HotpotQA. This is a negative result for the universality of the withholding strategy, showing it is not a one-size-fits-all improvement.

- **Negative result: Torch Hub CoT regression (Table 2):** On Torch Hub, adding chain-of-thought to RAFT decreases accuracy from 86.56 to 84.95 (−1.61). The paper does not discuss this result, but it suggests that for domains where the base fine-tuning already achieves high accuracy with short answers, the additional complexity of CoT targets may introduce noise rather than regularization.

### Critical Assessment

#### Does RAFT genuinely improve domain-specific RAG performance over baselines?

The central claim — that RAFT outperforms both DSF and DSF+RAG — is supported by Table 1, which shows RAFT achieving the highest accuracy on all five datasets. However, the magnitude of the improvement varies enormously: from a 0.01 point gain on Torch Hub (84.95 vs. 84.94 for DSF) to a 30.87 point gain on HotpotQA (35.28 vs. 4.41 for DSF+RAG). The claim that RAFT "consistently improves the model's performance" is technically true but masks a wide range of effect sizes, and on Torch Hub the claim is effectively a tie.

Several factors weaken the conclusiveness of this headline result. First, no statistical significance tests or confidence intervals are reported. For the smaller gaps — PubMed (+1.7 over DSF+RAG), Torch Hub (+0.01 over DSF) — it is impossible to determine whether these differences are meaningful or within noise from a single training run. Second, the paper does not report the size of the test sets, so the precision of the accuracy estimates is unknown. If the Torch Hub test set contains 100 examples, a 0.01 point difference corresponds to a single example, which is clearly not distinguishable from noise. Third, the paper does not report multiple training runs with different random seeds, which is standard practice for demonstrating that fine-tuning improvements are robust to initialization and data ordering.

The HotpotQA result (35.28% vs. 4.41% for DSF+RAG) is dramatic enough that it almost certainly represents a real effect, even without statistical testing. The HuggingFace result (74.00% vs. 42.59%) similarly represents a qualitative regime change. The PubMed result (73.30% vs. 71.60%) is marginal and would benefit from statistical validation. The paper does not address this variation in effect sizes or attempt to characterize which domains benefit most from RAFT beyond the brief note that PubMed's binary yes/no format limits headroom.

#### Does the paper demonstrate that training with distractor documents improves robustness?

The claim that distractor training improves robustness is supported by Figure 6, which shows that models trained with distractors maintain more stable performance across test-time document counts than models trained without distractors. This is a valid and well-designed experiment. The evidence is convincing for the two datasets shown (Natural Questions and HotpotQA), but the paper does not extend this analysis to the primary evaluation datasets (PubMed, HuggingFace, Torch Hub, TensorFlow), limiting the generality of the finding.

A more significant limitation is that the distractor selection procedure is not described in detail. The paper states that distractors are "sampled from the domain collection" but does not specify whether they are random samples, hard negatives, or topically related but non-answer-containing documents. The difficulty of the distractor task — and therefore the generalizability of the results — depends heavily on how similar distractors are to golden documents. If distractors are randomly sampled and clearly topically unrelated, the filtering task is easy. If distractors are topically related (e.g., different API functions from the same library that could plausibly answer the question), the filtering task is harder and the training signal is richer. Without this detail, it is unclear whether the reported robustness gains would transfer to deployment settings with realistic distractors.

Additionally, the paper does not ablate the number of distractors during training against the number during testing in a systematic grid. Figure 6 shows four training configurations (D*, D*+1D, D*+2D, D*+3D) and sweeps test-time `$k$` from 2 to 10, but does not report the full cross-product of training distractor count vs. test-time distractor count. This means the paper cannot distinguish between "training with any distractors helps" and "matching the training distractor distribution to the test-time distribution helps." The results suggest the former (since D*+1D generalizes well to test-time `$k = 10$`), but a more granular experiment would strengthen this conclusion.

#### Does the `$P$` hyperparameter investigation validate the withholding strategy?

The finding that `$P < 100\%$` is optimal for Natural Questions and Trivia QA (Figure 5) provides genuine evidence that training without the golden document for some fraction of examples improves downstream RAG performance. This is the most conceptually novel empirical result in the paper and is well-supported by the sweep experiment.

However, several caveats apply. First, the `$P$` sweep is conducted on Natural Questions, Trivia QA, and HotpotQA — but not on the five primary evaluation datasets from Table 1. The paper does not report what `$P$` value was used for the PubMed, HotpotQA (the Table 1 result, as distinct from the Figure 5 sweep), HuggingFace, Torch Hub, or TensorFlow experiments. It is possible that suboptimal `$P$` values were used for some of those datasets, and that the reported RAFT numbers could be improved. The paper states in Section 5.1 that "we consistently employ a training setup consisting of one golden document alongside four distractor documents" but does not specify `$P$` for the main results.

Second, the HotpotQA result — where `$P = 100\%$` is optimal — contradicts the paper's narrative that withholding golden documents is generally beneficial. The paper acknowledges this result but does not explain it. A plausible explanation (multi-hop questions require document access) is not offered, leaving the reader to speculate about when the withholding strategy applies and when it does not. The paper would be stronger with a characterization of *which types of questions* benefit from withholding and which do not, rather than treating it as a dataset-level hyperparameter to tune.

Third, the `$P$` experiment uses the same retriever at test time as at training time (or at least, the same document collection), but does not investigate how the optimal `$P$` changes with retriever quality. A domain with a near-perfect retriever might benefit from `$P \approx 100\%$` (always training with the golden document, since retrieval rarely fails), while a domain with a poor retriever might benefit from low `$P$`. The paper does not explore this interaction, which limits the practical guidance for practitioners selecting `$P$` for their own domains.

#### Are the baselines fair and comprehensive?

The baseline selection covers the main alternatives a practitioner would consider: base model, base model + RAG, domain fine-tuning, and domain fine-tuning + RAG. This is a reasonable set. The inclusion of GPT-3.5+RAG provides a useful reference point for absolute performance levels.

However, there are missing baselines that would strengthen the paper's claims:

- **RAFT without CoT and without citations:** Table 2 ablates CoT but does not separately ablate the citation delimiters (`##begin_quote##` / `##end_quote##`). It is possible that the long-form reasoning is what helps (by preventing overfitting), independent of the explicit citation format. An ablation with CoT-style reasoning but without the structured citation markers would separate these effects.
- **DSF trained with documents in context but no distractors:** This would isolate the effect of distractor training from the effect of simply having documents in the training context. The current comparison (DSF vs. RAFT) changes multiple variables simultaneously (documents in context, distractors, withheld golden documents, CoT format), making it difficult to attribute improvements to specific design choices.
- **RAFT with `$P = 100\%$`:** This would serve as a direct test of whether the withheld-golden-document strategy (the paper's most novel claim) actually improves over standard RAG fine-tuning with distractors. The `$P$` sweep in Figure 5 provides this comparison for NQ, TQA, and HotpotQA, but not for the five evaluation datasets in Table 1.
- **A larger base model:** All experiments use LLaMA2-7B. Showing that RAFT's improvements hold (or amplify, or diminish) at a different model scale (e.g., LLaMA2-13B) would address generalizability concerns. The GPT-3.5 comparison is a different model family and training procedure, making it a poor scaling baseline.

#### Does the paper demonstrate that RAFT's gains come from the claimed mechanisms?

The paper attributes RAFT's success to four design principles: documents in training context, distractor training, withheld golden documents, and CoT with citations. The experiments provide evidence for some but not all of these claims at the level of mechanistic demonstration.

**Documents in training context:** Supported by the DSF+RAG vs. RAFT comparison (Table 1). The gap is large and consistent, confirming that training with documents present is better than training without them when deployment uses RAG. This is the least surprising finding — it would be more surprising if training without documents produced better RAG performance.

**Distractor training:** Supported by Figure 6, which shows that training with distractors outperforms training without them. However, this experiment is on NQ and HotpotQA, not on the primary evaluation datasets. The paper does not report a "RAFT without distractors" baseline in Table 1 to confirm that distractors help on PubMed, HuggingFace, etc.

**Withheld golden documents:** Supported by the `$P$` sweep in Figure 5 for NQ and TQA, but contradicted for HotpotQA. The paper does not demonstrate this effect on the primary evaluation datasets. The mechanism is plausible — forcing memorization for some examples teaches fallback behavior — but the evidence is limited to two of three datasets in the hyperparameter study, and the negative HotpotQA result is unexplained.

**CoT with citations:** Supported by Table 2 across four of five datasets, with the Torch Hub regression as a caveat. The paper's interpretation (CoT prevents overfitting) is reasonable but not directly tested — an experiment showing training loss curves with and without CoT, or measuring generalization gap, would provide stronger evidence for the overfitting-prevention mechanism.

#### Single model family and limited test set reporting

All experiments use LLaMA2-7B-chat. The paper does not test whether RAFT's improvements transfer to other model families (e.g., Mistral, Falcon, Qwen) or to base (non-instruction-tuned) models. Given that the base model's instruction-following ability likely interacts with RAFT's training format (the model must learn to follow the document-reading and citation style), the results may not generalize to models with different instruction-tuning characteristics.

The paper does not report test set sizes for any dataset. Without knowing whether PubMed has 100, 1,000, or 10,000 test examples, the precision of the reported accuracies is unknown. For the Torch Hub result where RAFT and DSF differ by 0.01 points (84.95 vs. 84.94), this could reflect a single-example difference on a test set of 100 examples — indistinguishable from noise. Standard practice would be to report test set sizes and, ideally, confidence intervals or standard errors.

#### Missing experiments that would have strengthened the paper

Several experiments are conspicuous by their absence:

- **An experiment varying retriever quality:** RAFT is described as retriever-agnostic, but the interaction between retriever precision/recall and RAFT's training recipe is not explored. Would RAFT with low `$P$` be even more beneficial when the retriever is poor? Would the optimal number of training distractors scale with retriever noise? These questions are central to the paper's claim of robustness and are not addressed.

- **An experiment combining RAFT with different retrievers:** The paper states RAFT is "independent of the retriever used" but all experiments appear to use a single retriever (unspecified details). Testing RAFT with a sparse retriever (BM25), a dense retriever, and a hybrid retriever would validate this independence claim.

- **An experiment measuring training cost vs. inference cost tradeoffs:** RAFT has an upfront training cost (GPT-4 API calls for CoT generation, fine-tuning compute) that is not accounted for. A comparison against simply using GPT-4 for all inference, or against using a larger fine-tuned model with simpler training, would help practitioners decide whether RAFT's complexity is justified.

- **An experiment on out-of-domain generalization:** The paper emphasizes that RAFT is for "domain-specific" settings where test documents come from the same collection as training. But what if new documents are added to the domain after training? Does RAFT's memorization component (from withheld-golden-document training) help or hurt when the model encounters documents it hasn't seen during fine-tuning? This is a realistic deployment scenario not addressed.

- **An experiment on citation accuracy:** RAFT trains models to include citations, but the paper never evaluates whether those citations are correct (i.e., whether the quoted text actually supports the answer, and whether the quoted text exists in the cited document). A model could learn to produce plausible-looking citations that are factually wrong, and the accuracy metric would not catch this. For domains where trustworthiness matters (medical, legal), citation accuracy is as important as answer accuracy.

---

#### Summary of Strengths and Weaknesses

**Where the experiments are convincing:**
- The DSF+RAG regression phenomenon is clearly demonstrated across multiple datasets (HotpotQA, HuggingFace, TensorFlow) and provides strong motivation for RAFT.
- RAFT's superiority over DSF+RAG on HotpotQA and HuggingFace is large enough (30+ points) to be clearly meaningful.
- The `$P$` hyperparameter sweep shows that always including the golden document is not optimal for at least some datasets, which is a non-obvious and important finding.
- The distractor robustness experiment (Figure 6) provides clean evidence that training with distractors improves generalization to varying test-time conditions.

**Where the experiments are less convincing:**
- The tiny or nonexistent gaps on Torch Hub (+0.01) and the modest gap on PubMed (+1.7) suggest RAFT's benefits are highly dataset-dependent, but the paper does not characterize when RAFT helps more vs. less.
- No statistical testing or multiple-seed averaging is reported, making small differences uninterpretable.
- Test set sizes are not reported, preventing assessment of estimate precision.
- The distractor and withheld-document ablations are not reported on the primary evaluation datasets (Table 1), only on separate datasets (Figures 5, 6).
- Single model family (LLaMA2-7B) limits generalizability claims.
- The CoT and citation components are ablated together but not individually — the paper cannot distinguish whether it is the reasoning length, the citation format, or both that drive the improvements in Table 2.
- The HotpotQA `$P = 100\%$` optimum contradicts the paper's general narrative about withholding and is not explained.
- No experiments characterize the cost of the GPT-4 teacher for CoT generation, the sensitivity of results to teacher quality, or whether a weaker/cheaper teacher would suffice.

## 6. Limitations and Trade-offs

### 6.1 The Difficulty Estimation Cost Is Excluded from All Efficiency Calculations

**The assumption or constraint.** The paper's headline finding — that RAFT achieves dramatic improvements (e.g., 30+ percentage points on HotpotQA and HuggingFace) over baseline approaches — rests on a training recipe that requires generating chain-of-thought answers for every training example using GPT-4-1106 as a teacher model. The paper acknowledges this dependence explicitly in Section 3: "for all the datasets in our experiments, we generate the answers using the technique described above," referring to the GPT-4-1106 prompting step shown in Figure 3, where the model is given "the question, context and answer above" and instructed to "provide a logical reasoning for that answer." The paper does not account for the computational or financial cost of these GPT-4 API calls anywhere in the evaluation, nor does it report how many training examples exist per dataset so a practitioner could estimate the cost.

**The consequence.** A practitioner considering RAFT for their domain must pay an upfront cost that is entirely absent from the paper's analysis: generating CoT training data with GPT-4-1106 for every question-answer pair in their domain. For a domain with 10,000 training questions, this means 10,000 GPT-4 API calls — each producing a multi-paragraph reasoning chain with citations. At current API pricing, this could cost hundreds to thousands of dollars for a moderately sized domain, entirely before any model fine-tuning begins. This cost is not amortized across the paper's evaluations, making the reported accuracy gains an overestimate of RAFT's practical efficiency relative to DSF (which does not require a teacher model for training data construction). Additionally, the paper provides no evidence that a weaker or cheaper teacher model would suffice — if GPT-4-level reasoning is necessary to produce effective training targets, the method is effectively gated behind access to a frontier proprietary model, which limits reproducibility and raises concerns for practitioners working with sensitive domain data that cannot be sent to external APIs.

**What evidence exists in the paper.** The paper provides no cost analysis, no comparison of training data construction cost across methods, no experiment testing whether a weaker teacher model (e.g., GPT-3.5, or a self-generated CoT from LLaMA2-7B itself) produces comparable results, and no characterization of how many training examples exist per dataset. Section 3 states that "the Gorilla APIBench dataset, already includes reasoning in the answers" — meaning that for three of the five evaluation datasets, the CoT targets were pre-existing and GPT-4 was not used, but this is noted only in passing and is not acknowledged as a cost advantage for those specific benchmarks. For PubMed, HotpotQA, NQ, and Trivia QA, GPT-4-generated CoT targets are the default.

**Mitigation status.** The paper does not address this limitation. It does not propose cheaper alternatives, does not ablate teacher model quality, does not report training data construction costs, and does not frame the GPT-4 dependency as a limitation. The sentence about Gorilla APIBench pre-existing reasoning (Section 3) is the closest the paper comes to acknowledging the issue, but it is not presented as a cost concern.

---

### 6.2 RAFT's Benefits Are Highly Dataset-Dependent, With Near-Zero Gains on Some Benchmarks

**The assumption or constraint.** The paper presents RAFT as a general training recipe that "consistently improves the model's performance across PubMed, HotpotQA, and Gorilla datasets" (Abstract). However, the magnitude of improvement varies from a dramatic 30.87 percentage points on HotpotQA (35.28% for RAFT vs. 4.41% for DSF+RAG, Table 1) to essentially zero on Torch Hub (84.95% for RAFT vs. 84.94% for DSF, a 0.01 point difference that is indistinguishable from noise given the absence of reported test set sizes or confidence intervals). The paper does not provide a framework for predicting *when* RAFT will provide large gains versus when it will provide negligible benefit, beyond a brief note about PubMed's binary yes/no format limiting headroom (Section 4.1). The `$P$` hyperparameter sweep in Section 4.4 further demonstrates this dataset-dependence: the optimal `$P$` ranges from ~40% (Natural Questions) to 100% (HotpotQA), with qualitatively different optimal strategies across datasets.

**The consequence.** A practitioner cannot know from this paper whether RAFT is worth implementing for their specific domain. The method provides transformative gains on some tasks (HotpotQA, HuggingFace) and no measurable gain on others (Torch Hub). Without a characterization of *which task properties* predict RAFT's effectiveness — question complexity, document length, retrieval quality, base model capability, answer format — a practitioner must implement the full RAFT pipeline (including GPT-4 CoT generation, distractor sampling, `$P$` tuning) merely to discover whether it helps their use case. The paper's claim of consistent improvement is technically true (RAFT is never worse than baselines by a meaningful margin) but obscures the fact that for some domains, the method adds substantial complexity and cost for zero practical gain.

**What evidence exists in the paper.** The evidence for this limitation is in the data itself, though the paper does not frame it as a limitation. Table 1 shows the range of RAFT gains over the next-best baseline (excluding GPT-3.5): HotpotQA +30.87, HuggingFace +12.94, TensorFlow +0.30, PubMed +1.70, Torch Hub +0.01. The `$P$` sweep (Figure 5) shows dataset-specific optima ranging from 40% to 100%. The CoT ablation (Table 2) shows that CoT helps substantially on HotpotQA (+9.66) and HuggingFace (+14.93) but slightly hurts on Torch Hub (−1.61). The paper reports these numbers but does not synthesize them into a predictive framework for when RAFT matters. Section 4.1 contains the only attempt at explanation: "for PubMed QA, since it is a binary yes/no question, we don't observe significant gains" — a single factor (binary answer format) that does not explain the Torch Hub or TensorFlow patterns.

**Mitigation status.** Not addressed. The paper does not propose a diagnostic for determining when RAFT will be beneficial, does not analyze task features that correlate with RAFT's effectiveness, and does not provide heuristics for practitioners to estimate expected gains before implementation. The dataset-dependence is presented as an empirical finding to be observed, not as a limitation to be understood or mitigated.

---

### 6.3 Single Model Family and Unknown Base Model Dependence

**The assumption or constraint.** Every experiment in the paper uses LLaMA2-7B-chat as the base model. The paper does not test RAFT on any other model family (e.g., Mistral, Falcon, Qwen, Gemma), any other model scale (e.g., LLaMA2-13B, LLaMA2-70B), or any base (non-instruction-tuned) variant. The paper implicitly assumes that RAFT's benefits transfer across model architectures and scales, but provides no evidence for this. The choice of an instruction-tuned model is particularly significant because RAFT's training format — structured CoT with explicit citation delimiters, document-processing behavior — relies on the base model's ability to follow formatting instructions and produce structured outputs. A base model without instruction tuning might struggle to learn these patterns from fine-tuning data alone, or might require substantially more training data to do so.

**The consequence.** The paper's results may not generalize to other model families, scales, or training paradigms. A practitioner using Mistral-7B or a fine-tuned CodeLlama for a code documentation domain has no evidence that RAFT will provide similar gains over DSF+RAG. The interaction between RAFT's recipe and model capabilities is unexplored: does a stronger base model benefit more from RAFT (because it can better learn the complex CoT reasoning patterns) or less (because it already has strong reading comprehension from pretraining, making the training-data alignment less critical)? Does a weaker base model fail entirely to learn the RAFT format? These questions are central to the practical deployability of the method, and the paper provides no data to answer them.

**What evidence exists in the paper.** None. The paper does not acknowledge model-family dependence as a limitation, does not report experiments on other base models, and does not discuss how RAFT's design choices might interact with base model properties. The only non-LLaMA2 model in the paper is GPT-3.5, used as a reference point in Table 1 with RAG but never fine-tuned with RAFT (which would be impossible for a closed model). The paper's claims about generalizability are therefore entirely untested.

**Mitigation status.** Not addressed and not acknowledged. The paper treats LLaMA2-7B-chat as a representative model without arguing for its representativeness or discussing the limitations of single-model evaluation.

---

### 6.4 The GPT-4 Teacher Model Introduces an Uncontrolled Capability Gap

**The assumption or constraint.** RAFT's training data construction relies on GPT-4-1106 to generate the chain-of-thought answers with verbatim citations that serve as fine-tuning targets for LLaMA2-7B-chat. The paper assumes that the student model (LLaMA2-7B) can learn to reproduce the reasoning patterns demonstrated by the teacher model (GPT-4-1106) without introducing a quality gap or learning superficial patterns that do not transfer to novel questions. The paper provides no analysis of the quality of the GPT-4-generated CoT targets, no measurement of whether LLaMA2-7B successfully learns to produce similarly structured reasoning, and no experiment testing whether the observed improvements come from the *format* of the CoT (reasoning chain with citations) or from the *quality* of the GPT-4-generated content (which may encode reasoning strategies that LLaMA2-7B does not independently possess).

**The consequence.** Two distinct failure modes are possible, and the paper cannot distinguish between them. First, RAFT's gains might depend critically on GPT-4-quality reasoning traces, meaning that a practitioner without GPT-4 API access (due to cost, privacy, or availability constraints) cannot achieve comparable results with a weaker teacher or self-generated CoT. Second, LLaMA2-7B might be learning to reproduce the *surface form* of GPT-4's reasoning (citation delimiters, multi-sentence structure, formal tone) without internalizing the underlying reasoning strategies, producing outputs that look like GPT-4 reasoning but are not actually more accurate. The paper's evaluation (accuracy of final answers) would not distinguish these cases — a model could produce correct answers with superficially GPT-4-like reasoning without having actually learned better reasoning, or could produce correct answers for the wrong reasons while mimicking the CoT format.

The Torch Hub CoT ablation result (Table 2) provides circumstantial evidence for the surface-form hypothesis: adding CoT *decreases* accuracy on Torch Hub from 86.56 to 84.95, suggesting that the CoT format can introduce noise rather than improved reasoning, at least in some domains. This is a single data point, but it raises the question of whether CoT's benefits are format-driven (preventing overfitting through longer targets, as the paper argues) or content-driven (better reasoning from GPT-4), and the paper cannot separate these explanations.

**What evidence exists in the paper.** The paper provides the prompt used to generate CoT answers (Figure 3) but does not evaluate the quality of the GPT-4-generated CoT traces, does not report examples of poor or incorrect GPT-4 reasoning, does not compare against CoT generated by a weaker model or by LLaMA2-7B itself, and does not measure how faithfully the fine-tuned model reproduces GPT-4's reasoning style versus developing its own. The CoT ablation in Table 2 removes CoT entirely (comparing CoT vs. no CoT) rather than varying teacher quality, so it cannot address the teacher-dependence question.

**Mitigation status.** Not addressed. The paper treats GPT-4 as an off-the-shelf tool for training data generation without analyzing its role as a confound in the experimental design. The observation that Gorilla APIBench already includes reasoning in its answers (Section 3) implicitly acknowledges that teacher-generated CoT is not necessary when human-written reasoning exists, but the paper does not use this to motivate an analysis of teacher quality effects.

---

### 6.5 No Statistical Reporting Prevents Confidence Assessment

**The assumption or constraint.** The paper reports all results as single-point accuracy estimates without confidence intervals, standard deviations, statistical significance tests, or multiple-seed averaging. Test set sizes are not reported for any dataset. The hyperparameter sweeps in Figures 5 and 6 show curves without error bars. The paper implicitly assumes that the reported differences are reliable and replicable, but provides no evidence for this assumption.

**The consequence.** For the datasets where RAFT's advantage over baselines is large (HotpotQA: 35.28 vs. 4.41; HuggingFace: 74.00 vs. 42.59), the effect sizes are almost certainly real and statistically significant regardless of test set size — a 30-point gap on a test set of even 50 examples would be highly unlikely under the null hypothesis. However, for the datasets where differences are small, the results are uninterpretable:

- **Torch Hub:** RAFT (84.95) vs. DSF (84.94) is a 0.01 point difference. If the Torch Hub test set contains 100 examples, this corresponds to a single test example — completely indistinguishable from noise. Without knowing the test set size, it is impossible to determine whether RAFT matches DSF (as the paper implies) or whether the numbers reflect random variation around an identical true accuracy.
- **PubMed:** RAFT (73.30) vs. DSF+RAG (71.60) is a 1.70 point difference. On a test set of 500, this could be meaningful or noise depending on the variance.
- **TensorFlow:** RAFT (86.86) vs. DSF (86.56) is a 0.30 point difference. Same interpretability problem as Torch Hub.

The absence of statistical reporting means that the central claim of Table 1 — that RAFT "consistently outperforms" baselines — is not statistically supported for three of the five evaluation datasets, where the margins are small and the sample sizes unknown.

**What evidence exists in the paper.** The paper reports no statistical measures anywhere. No test set sizes, no confidence intervals, no standard deviations, no p-values, no mention of multiple random seeds. The experimental setup in Section 4 describes datasets by reference to their original papers but does not report the specific splits or sizes used.

**Mitigation status.** Not addressed. Standard practice in NLP evaluation is to report test set sizes and, ideally, some measure of statistical reliability (bootstrap confidence intervals, standard deviation across runs, or at minimum an acknowledgement of the limitation when such measures are absent). The paper does none of these.

---

### 6.6 No Characterization of Citation Accuracy Despite Citations Being a Key Design Element

**The assumption or constraint.** A central design element of RAFT is that the model is trained to produce chain-of-thought answers with explicit verbatim citations from the source documents, delimited by `##begin_quote##` and `##end_quote##` tags. The paper presents this as a mechanism for improving answer accuracy and preventing overfitting (Section 4.2, Table 2). The evaluation metric, however, is **answer accuracy only** — whether the final answer matches the ground truth. The paper never evaluates whether the citations the model produces are *correct*: whether the quoted text actually appears in the provided documents, whether it genuinely supports the answer, and whether the model attributes evidence to the right source. A model could achieve high answer accuracy while producing hallucinated citations, miscited evidence, or correctly-answered questions with fabricated justifications.

**The consequence.** For domain-specific deployments where trustworthiness matters — medical question-answering (PubMed), legal document analysis, enterprise knowledge bases — citation accuracy is as important as answer accuracy. A model that answers "yes" correctly on a PubMed QA question but cites irrelevant or fabricated text to justify its answer is dangerous: it creates unfounded user trust in the model's reasoning process and makes errors harder to detect through manual review. The paper's qualitative example in Figure 4 demonstrates that RAFT can produce accurate citations ("David Weissman as a screenwriter with film credits including...") but does not quantify how often this occurs versus hallucinated or incorrect citations. Without citation accuracy evaluation, the paper cannot claim that RAFT produces *trustworthy* reasoning — only that it produces answers in a format that looks trustworthy.

Furthermore, the paper cannot distinguish whether the CoT format's accuracy benefits (Table 2) come from genuine evidence grounding or from the regularizing effect of longer training targets. If the model learns to produce plausible-looking citations that are often wrong but the answer happens to be correct (perhaps because the model memorized the answer from other training examples), then the citation mechanism is effectively a form of "stylistic regularization" rather than genuine reading comprehension improvement. The paper's framework provides no evidence to distinguish these hypotheses.

**What evidence exists in the paper.** The paper provides one qualitative example (Figure 4) showing correct citations, but conducts no quantitative evaluation of citation accuracy, citation grounding, or evidence fidelity. The paper does not report how often citations are verifiable against the provided documents, how often they support the answer, or how often they are hallucinated. The paper does not acknowledge citation accuracy as an unevaluated dimension of the method.

**Mitigation status.** Not addressed and not acknowledged as a limitation. The paper treats the presence of citations in the output as a self-evident good, without verifying that the content of those citations is reliable. For a method that makes citation-based reasoning a central design principle (mentioned in the Abstract, Section 3, Section 4.2, and the qualitative example), the absence of any citation quality evaluation is a significant gap between the method's claimed benefits and the evidence provided.

## 7. Implications and Future Directions
- How this changes the landscape
  - RAFT reframes “how to adapt an LLM to a domain with RAG” as a training-data design problem: teach the model to read the right evidence and ignore distractors under the exact conditions it will face in deployment (Figure 2). This is a practical, system-level insight for building reliable domain assistants.
- Practical applications
  - Enterprise document QA, healthcare knowledge assistants (PubMedQA), and developer copilots grounded in library docs (APIBench) can benefit by fine-tuning with RAFT to increase accuracy and robustness when retrieval returns mixed-quality results (Table 1).
- Recommendations for practitioners (from the paper’s analyses)
  - Include distractors in training, not just gold documents (Figure 6).
  - Tune the fraction P% of gold-in-context examples; values below 100% can perform better (Figure 5).
  - Prefer outputs that include evidence-grounded reasoning and citations; monitor for task-specific exceptions (Table 2).
  - Use a training setup with roughly one gold document plus several distractors (the paper commonly uses 1+4; Section 5.1).
- Follow-up research
  - Joint optimization of retriever and RAFT-trained generator; adversarial distractor training to stress-test robustness.
  - Preference optimization or verification models that reward faithful quoting and penalize hallucinations.
  - Scaling studies (larger models, longer contexts) and evaluation on dynamic or cross-domain transfer settings.
  - Efficiency: compressing reasoning or training with rationales without incurring large token costs.
  - Formal measures of faithfulness using the quoted evidence spans; user studies on interpretability/value of citations.

Overall, RAFT’s main contribution is a simple but powerful shift in fine-tuning: align the training context with the deployment context (documents plus distractors) and explicitly train the model to ground its reasoning in quoted evidence. The empirical results and ablations (Table 1; Tables 2; Figures 4–6) make a strong case that this approach consistently improves domain-specific RAG performance and robustness.
