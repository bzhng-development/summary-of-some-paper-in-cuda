# RAFT: Adapting Language Model to Domain Specific RAG

**ArXiv:** [2403.10131](https://arxiv.org/abs/2403.10131)
**Authors:** Tianjun Zhang, Shishir G. Patil, Naman Jain, Sheng Shen, Matei Zaharia, Ion Stoica, Joseph E. Gonzalez
**Institutions:** UC Berkeley

## 🎯 Pitch

Retrieval-Augmented Fine-Tuning (RAFT) revolutionizes domain-specific language model training by teaching models to navigate and reason effectively with both relevant and distracting documents. This innovative approach significantly boosts accuracy and robustness in complex, retrieval-based tasks across domains like medical QA and API documentation, enhancing real-world applicability and reliability.

---

## 1. Executive Summary
Retrieval-Augmented Fine-Tuning (`RAFT`) is a training recipe that adapts a pre-trained language model to perform better in domain-specific “open-book” settings where it receives retrieved documents at inference. It teaches the model to quote and reason from the correct source while ignoring distractors, yielding large gains over standard supervised fine-tuning and over “RAG-at-inference-only” approaches across medical QA and API documentation tasks (see Table 1) and making the model more robust to noisy retrieval (Figures 5–6).

## 2. Context and Motivation
- Problem addressed
  - LLMs are increasingly used in specialized domains (e.g., enterprise document QA, API help, medical research). In these settings, the model answers questions using documents retrieved at inference—often called retrieval-augmented generation (`RAG`). However:
    - Standard in-context `RAG` does not exploit the fact that the target domain is known ahead of time and that retrieved documents will be available at test time.
    - Standard supervised fine-tuning on question–answer pairs (without documents) can align style or memorize answers but does not train the model to read and use domain documents in context, nor to handle retrieval imperfections (distractors, missing gold documents).
- Why it matters
  - In real deployments, retrieval is imperfect. Models must extract and cite the right spans from relevant documents while ignoring distracting but plausible text. Improving this “reading under noise” ability directly impacts accuracy and reliability for domain assistants (Section 1; Figure 1).
- Prior approaches and their shortcomings
  - In-context `RAG` only: append top-k retrieved documents at inference without adapting the model to this setting. This is “taking an open-book exam without studying” (Section 1; Figure 1b).
  - Supervised fine-tuning (`DSF`) without documents: improves style/knowledge but does not train the model to use retrieved context, nor to resist distractors (Section 3).
  - Some recent works fine-tune for `RAG` with general domains where test documents can differ from train documents. This paper focuses instead on the domain-specific setting where test-time documents are from the same domain/corpus used during fine-tuning (Related Works, Section 6).
- Positioning
  - `RAFT` fine-tunes the generator (LLM) itself—without re-training the retriever—to “study for an open-book exam” by learning to:
    - Cite verbatim the relevant span from a “golden” document,
    - Produce a chain-of-thought style explanation,
    - Ignore distractors, and
    - Remain capable when the retriever fails to return the golden document (Section 3; Figures 2–3).

Key terms (paper-specific)
- `RAG` (Retrieval-Augmented Generation): generation using documents retrieved at inference.
- `Golden document` (`D*`): document(s) that support the answer.
- `Distractor document` (`Di`): retrieved document(s) that are unrelated or misleading.
- `P%`: fraction of training examples that include the golden document in context.
- `Top-k`: number of retrieved documents appended to the prompt at inference.

## 3. Technical Approach
At a high level, `RAFT` is a supervised fine-tuning (SFT) recipe that modifies the training data and targets so that the model learns to read retrieved documents, cite the right span, and reason to the answer while ignoring distractors (Section 3; Figure 2).

Step-by-step
1. Prepare domain and QA pairs
   - Start from a domain corpus and a QA dataset for that domain (e.g., PubMed QA for biomedical, Gorilla APIBench for code/API docs). For each question `Q`, identify document(s) `D*` that support the answer (the “golden” document(s)). Also sample `k−1` distractor documents `Di` from the same domain (Section 3).

2. Construct training contexts with and without the golden document
   - For a fraction `P` of examples, include the golden document along with distractors.
   - For the remaining fraction `1 − P`, include only distractors—no golden document—so the model cannot rely on the context being correct every time (Section 3; Section 4.4).
   - The training data template is quoted verbatim in Section 3:
     > P % of data: Q + D∗ + D1 + D2 + . . . + Dk → A∗  
     > (1 − P) % of data: Q + D1 + D2 + . . . + Dk → A∗

3. Train the target to be a chain-of-thought answer with explicit citations
   - The output `A*` includes:
     - A chain-of-thought reasoning segment (to organize the evidence and logic), and
     - Verbatim citations from the relevant document, delimited by special markers `##begin_quote## ... ##end_quote##` (Figure 3).
   - Example (Figure 3) shows a reasoning format:
     > “##Reason: ... cites … ##begin_quote## ... ##end_quote## ... ##Answer: Delhi”
   - Chain-of-thought targets are generated before training (for the datasets requiring it), using `GPT-4-1106` (Section 4.2).

4. Fine-tune with standard SFT
   - Use the above inputs (Q + documents) and outputs (reasoning with citations + final answer) to fine-tune the base model (e.g., `LLaMA2-7B-chat`) with standard language modeling (Section 3). No retriever is trained.

5. Inference (test-time)
   - Use any retriever to fetch `top-k` documents, append them to the question, and generate an answer. `RAFT` is retriever-agnostic (Section 3), but the training design makes the generator robust to missing gold documents and distractors.

Design choices and why
- Include distractors during training: teaches the model to ignore irrelevant text, a well-known failure mode in long-context settings (Section 5; Figure 6; related works cite “Lost in the Middle”).
- Mix in examples without the golden document (`1 − P`): prepares the model for retrieval failures and discourages over-reliance on context being perfect every time (Section 4.4; Figure 5).
- Require explicit quoting in the reasoning: aligns the model’s reasoning with concrete spans from the source, making evidence use explicit (Figure 3).
- Use chain-of-thought targets: improves training stability and reduces overfitting to short answers, improving accuracy on several datasets (Section 4.2; Table 2).

Analogy
- The method operationalizes the “study for an open-book exam” analogy (Figure 1c): practice with mixed-quality notes (distractors), learn to find and quote the right passage, and still answer when the notes are incomplete.

## 4. Key Insights and Innovations
- Training with controlled retrieval imperfections (fundamental)
  - Innovation: Build SFT examples that intentionally include distractors and sometimes omit the golden document (`P% < 100`).
  - Why it matters: Models trained only with perfect context overfit to ideal retrieval; mixing in failures improves robustness at test time with realistic top-k retrieval (Section 4.4; Figure 5 and Section 5.1; Figure 6).
  - Evidence: Figure 5 shows better final accuracy when `P` is not 100%—optimal `P` varies by dataset (40–100%). Figure 6 shows that including distractors during training yields better performance across varying test-time top-k.

- Reasoning with explicit, quoted evidence (incremental but impactful)
  - Innovation: The target answer includes chain-of-thought plus verbatim quotes from the supporting document(s) using delimiters (Figure 3).
  - Why it matters: This scaffolds the model to “read” and justify, not just produce an answer token; it reduces overfitting to terse labels and improves accuracy on multi-hop and document-grounded tasks.
  - Evidence: Table 2 shows notable gains with chain-of-thought vs. without on HotpotQA (+9.66 points) and HuggingFace (+14.93 points). Slight regressions can occur (Torch Hub −1.61), suggesting task-dependent effects.

- Retriever-agnostic generator adaptation for domain-specific `RAG` (incremental)
  - Innovation: Focus on training only the generator for a known domain so it can use retrieved documents effectively at inference, regardless of retriever details (Section 3).
  - Why it matters: In many deployments, retrievers are modular/black-box; adapting the LLM alone is practical and still yields strong gains (Table 1).

- Robustness to variable top-k at test time (incremental)
  - Innovation: Analyze generalization when the number of retrieved documents changes at inference (2–10 docs), and show that training with distractors helps (Section 5; Figure 6).
  - Why it matters: In production, top-k is tuned for recall/latency; robustness to such changes reduces brittleness.

## 5. Experimental Analysis
Evaluation design
- Datasets (Section 4)
  - Open-domain QA over Wikipedia: `Natural Questions (NQ)`, `TriviaQA`, `HotpotQA` (multi-hop).
  - Domain-specific corpora:
    - `PubMedQA`: biomedical yes/no/maybe questions grounded in medical research.
    - `Gorilla APIBench`: three API documentation domains—`HuggingFace Hub`, `Torch Hub`, `TensorFlow Hub`—requiring generation of correct, executable API calls from docs.
- Baselines (Section 4; Table 1)
  - `LLaMA2-7B-chat` (0-shot).
  - `LLaMA2-7B-chat + RAG` (in-context documents, no fine-tuning for RAG).
  - `DSF`: domain-specific fine-tuning without documents.
  - `DSF + RAG`: DSF model with in-context documents at inference.
  - `GPT-3.5 + RAG`: reference larger model with in-context documents.
- Metrics
  - Reported as accuracy-style scores across tasks (Table 1; Table 2). PubMedQA is a binary QA; APIBench evaluates correctness of API calls; exact metric names are not expanded in the preprint text provided, but scores are directly comparable within each dataset.

Main quantitative results
- Table 1 highlights that `RAFT (LLaMA2-7B)` outperforms both `LLaMA2-7B` and `DSF` variants across all domain-specific datasets and on PubMedQA. Selected comparisons:
  - PubMedQA: `RAFT` 73.30 vs. `DSF + RAG` 71.6 and `GPT-3.5 + RAG` 71.60.
  - HotpotQA: `RAFT` 35.28 vs. `DSF` 6.38; `GPT-3.5 + RAG` is higher at 41.5 (so `RAFT` does not win here).
  - HuggingFace Hub: `RAFT` 74.00 vs. `DSF` 61.06 and `GPT-3.5 + RAG` 29.08.
  - Torch Hub: `RAFT` 84.95 vs. `DSF` 84.94 and `GPT-3.5 + RAG` 60.21.
  - TensorFlow Hub: `RAFT` 86.86 vs. `DSF` 86.56 and `GPT-3.5 + RAG` 65.59.
- Interpretation
  - The base `LLaMA2-7B` struggles in these evaluations, even with `RAG` (e.g., HotpotQA 0.03 with RAG). `DSF` corrects answer style and domain familiarity but is not consistently improved by adding RAG, indicating insufficient training to use context (Section 4.1).
  - `RAFT` adds the missing “read-and-cite” skill: across coding/API domains, gains over `GPT-3.5 + RAG` are striking (e.g., +44.92 points on HuggingFace; Table 1), and on PubMed it is slightly better than both `DSF + RAG` and `GPT-3.5 + RAG`.
  - On HotpotQA, `RAFT` is still below `GPT-3.5 + RAG` (35.28 vs. 41.5), signaling that multi-hop reasoning over Wikipedia remains challenging for a 7B model even with RAFT.

Ablations and robustness
- Chain-of-thought impact (Table 2; Section 4.2)
  - `RAFT` vs. `RAFT w.o CoT`:
    - HotpotQA: 35.28 vs. 25.62 (+9.66).
    - HuggingFace: 74.00 vs. 59.07 (+14.93).
    - PubMedQA: 73.30 vs. 68.30 (+5.00).
    - TensorFlow Hub: 86.86 vs. 83.21 (+3.65).
    - Torch Hub: 84.95 vs. 86.56 (−1.61).
  - Takeaway: reasoning targets help most on tasks requiring document synthesis; minor regressions can occur on tasks where terse pattern matching suffices.

- How many training examples should include the golden doc? (Section 4.4; Figure 5)
  - Quote from Figure 5 caption: 
    > “mixing some amount of data that the golden document is not put in the context is helpful for in-domain RAG.”
  - Optimal `P%` varies by dataset (reported ranges include 40%, 60%, 100%), suggesting a real trade-off between “reading from context” and “answering despite imperfect retrieval.”

- Training with distractors improves generalization to variable top-k (Section 5; Figure 6)
  - When training with only the golden document, performance is poorer and brittle as test-time top-k varies (2–10 documents).
  - Natural Questions: best with `D* + 3D` training; HotpotQA: best with `D* + 1D`.
  - The paper standardizes on “one golden + four distractors” in experiments (Section 5.1: “we consistently employ a training setup consisting of one golden document alongside four distractor documents”), and mirrors this at test time in one study (Section 4.4).

Qualitative evidence (Section 4.3; Figure 4)
- Example: HotpotQA question about a screenwriter. `DSF` answers with a film title (“The Family Man”)—a plausible but wrong entity type—whereas `RAFT` identifies the correct person (“David Weissman”) by using the provided documents.

Overall assessment
- The experiments strongly support `RAFT` for domain-specific `RAG`: large margins on API documentation and PubMedQA, and clear robustness benefits from the distractor/no-golden training design.
- Results are mixed on open-domain multi-hop (HotpotQA), where `GPT-3.5 + RAG` still leads, highlighting limitations of a 7B model and potential need for stronger reasoning or multi-hop training.

## 6. Limitations and Trade-offs
- Domain-specific focus
  - The method explicitly targets settings where the domain corpus at test time is known during training. Generalization to new domains is not studied (Section 2; Section 6).
- Dependence on curated training signals
  - Requires identifying golden documents per question and generating high-quality chain-of-thought with explicit quotes (Figure 3; Section 4.2). This introduces data preparation overhead and may depend on stronger LLMs (e.g., `GPT-4-1106`) for target generation.
- Retriever remains a black box
  - `RAFT` does not train or adapt the retriever; it assumes some retrieval pipeline exists. While generator robustness is improved, overall performance still depends on retrieval recall/precision at test time (Section 3).
- Sensitivity to hyperparameters
  - Performance depends on `P%` (golden-doc inclusion rate) and the number/type of distractors (Figures 5–6). The optimal settings vary across datasets, suggesting additional tuning per deployment.
- Computational cost
  - Training inputs include question plus multiple documents and long reasoning targets, which increases context length and training compute/memory relative to standard `DSF`.
- Multi-hop coverage
  - Although the method can accommodate multiple golden documents (Section 3), the strongest reported wins are in API and PubMed settings; HotpotQA remains challenging (Table 1), pointing to limits in multi-hop reasoning or the need for specialized multi-hop training curricula.

## 7. Implications and Future Directions
- Practical impact
  - For enterprise and vertical assistants (medical, legal, developer tools), `RAFT` provides a pragmatic, retriever-agnostic way to make LLMs effective “open-book” readers: cite relevant spans, ignore noise, and remain useful when retrieval fails.
- Methodological implications
  - Demonstrates that fine-tuning the generator with imperfect retrieval examples is crucial; training on ideal contexts (gold-only) can hurt robustness (Figures 5–6). This may generalize to other long-context tasks beyond `RAG`.
- Research directions
  - Joint training with retrievers: extend `RAFT` by co-training or distilling retrieval signals while preserving distractor robustness.
  - Multi-hop reasoning curricula: explicitly construct examples with multiple golden documents and structured reasoning supervision for datasets like HotpotQA.
  - Automatic `P%` and distractor scheduling: learn to adapt the golden/no-golden mix and distractor difficulty over the course of training.
  - Verification and calibration: build mechanisms that check quoted evidence consistency and calibrate confidence when golden docs are absent.
  - Long-context architectures: combine `RAFT` with models and prompting strategies that mitigate “lost in the middle” effects for very long inputs.
  - Domain transfer: study how `RAFT` models trained on one domain adapt to related corpora, and whether the quoting/reasoning habits help in zero-shot or few-shot transfer.

Block-quoted anchors to the paper
- Training data recipe (Section 3):
  > P % of data: Q + D∗ + D1 + D2 + . . . + Dk → A∗  
  > (1 − P) % of data: Q + D1 + D2 + . . . + Dk → A∗
- Reasoning with quotes format (Figure 3):
  > “##Reason: … ##begin_quote## … ##end_quote## … ##Answer: …”
- Result highlights (Table 1): `RAFT (LLaMA2-7B)` reaches 73.30 (PubMed), 35.28 (Hotpot), 74.00 (HuggingFace), 84.95 (Torch), 86.86 (TensorFlow).
- CoT ablation (Table 2): notable gains on Hotpot (+9.66) and HuggingFace (+14.93).
- On mixing no-golden examples (Figure 5):
  > “mixing some amount of data that the golden document is not put in the context is helpful for in-domain RAG.”
- Robustness to variable top-k (Figure 6): training with `D* + distractors` beats `gold-only` across different test-time document counts.
