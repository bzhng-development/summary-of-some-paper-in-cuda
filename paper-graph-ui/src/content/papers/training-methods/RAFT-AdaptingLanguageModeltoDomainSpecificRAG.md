# RAFT: Adapting Language Model to Domain Specific RAG

**ArXiv:** [2403.10131](https://arxiv.org/abs/2403.10131)

## 🎯 Pitch

RAFT introduces Retrieval-Augmented Fine-Tuning—a novel training approach that adapts large language models for domain-specific retrieval-augmented generation (RAG) tasks by explicitly teaching the model to extract answers from relevant retrieved documents while ignoring distractors. This method not only yields substantial performance improvements over standard fine-tuning and off-the-shelf RAG, but also enhances robustness to retrieval imperfections, making it highly impactful for mission-critical applications like medical QA and enterprise knowledge assistants where accurate, context-sensitive reasoning over domain documents is essential.

---

## 1. Executive Summary
RAFT (Retrieval-Augmented Fine-Tuning) is a training recipe that adapts a pretrained language model to work better with retrieval-augmented generation (RAG) in a fixed domain by fine-tuning it to read from relevant documents while ignoring distractors. It couples this with chain-of-thought style answers that include verbatim citations, yielding large improvements over standard fine-tuning and off-the-shelf RAG across multiple domains (Table 1), and robustness to variable numbers of retrieved documents (Figures 5–6).

## 2. Context and Motivation
- Problem addressed
  - In many applications (e.g., medical QA, enterprise knowledge bases, API documentation assistants), a language model will answer questions with access to a fixed set of domain documents. The open question is how best to adapt the model so it actually uses those documents effectively at test time, especially in the presence of retrieval errors and irrelevant context.
- Why it matters
  - Real systems depend on RAG to inject up-to-date or domain-specific knowledge at inference time. If the model cannot reliably locate and extract the right passage from retrieved text—and resist distraction—overall accuracy suffers even with a strong retriever. This is particularly significant in high-stakes or specialized settings (Section 1 and “Domain-Specific Open-Book Exam” in Section 2).
- Prior approaches and their gaps
  - In-context RAG only: the model is not trained to read domain documents; it just receives them at test time. The paper analogizes this to “taking an open-book exam without studying” (Figure 1b). This wastes the opportunity to prepare the model to read within the specific domain and to cope with retrieval imperfections.
  - Supervised fine-tuning (SFT) only: the model is trained on question-answer pairs, often without the documents in-context. This either encourages memorization of domain facts or learns answer style, but does not teach the model how to leverage retrieved documents at test time (Figure 1a).
- Positioning relative to existing work
  - Retrieval-augmented language models (e.g., RETRO, Atlas) integrate retrieval either during pretraining or joint fine-tuning, often for open-domain QA or language modeling. Several recent works fine-tune for RAG generally, where training and test domains can differ. RAFT instead focuses narrowly on the domain-specific open-book setting: the same documents are available during training and testing, and the goal is to train the model to (1) read the right passages and (2) ignore distractors in that exact environment (Section 3; Figures 1–2).

## 3. Technical Approach
At a glance, RAFT is standard supervised fine-tuning on prompts that include the question plus retrieved documents (both relevant and distracting). The model is trained to produce a chain-of-thought explanation with explicit citations to the relevant text and a final answer.

- Core setup and notation (Section 3; Figure 2)
  - `Q`: question.
  - `D*`: “golden” document(s)—the document(s) containing the necessary evidence to answer the question.
  - `Di`: “distractor” documents—retrieved text that is topically related but not answer-bearing.
  - `A*`: the target output, consisting of a chain-of-thought style explanation and the final answer, where the explanation quotes relevant spans from `D*`.
- Training mixture design (Section 3)
  - RAFT constructs two kinds of training examples:
    - With gold context (P% of examples):
      - Input: `Q + D* + D1 + ... + Dk` (a mixture of the gold document(s) and k distractors).
      - Output: `A*` (reasoning + answer).
    - Without gold context ((1−P)% of examples):
      - Input: `Q + D1 + ... + Dk` (only distractors).
      - Output: `A*`.
  - Why include examples without the gold document? To prevent over-reliance on contextual evidence and support scenarios where the retriever fails. This also encourages the model to retain some domain knowledge directly (Section 3; Figure 5 explores different P%).
- Output format and citation mechanism (Figure 3)
  - The model is trained to produce:
    - A reasoning section: `##Reason: ...`
    - A final answer: `##Answer: ...`
  - Within the reasoning, the target includes verbatim citations of the form:
    - `##begin_quote## [quoted evidence] ##end_quote##`
  - Example (Figure 3): to answer “The Oberoi family is part of a hotel company that has a head office in what city?”, the reasoning explicitly quotes the sentences that (1) link the family to The Oberoi Group and (2) state “The Oberoi Group is a hotel company with its head office in Delhi,” then concludes with `##Answer: Delhi`.
- Training objective
  - Standard SFT (token-level cross-entropy) is used on the full reasoning+answer output, so the model learns both the quoting behavior and the final answer format (Section 3).
- Inference-time use (Figure 2, right)
  - RAFT is retriever-agnostic: at test time, any retrieval system supplies top-k documents. The model receives `Q + retrieved docs` and is expected to (1) select relevant snippets, (2) ignore distractors, and (3) answer with explanation and citations.
- Design choices and their rationale
  - Train with distractors: Models are “easily distracted by irrelevant context” (supported by prior work and tested in Section 5). Including distractors during fine-tuning teaches the model to filter noise.
  - Mix of gold/no-gold training (vary P%): Retrieval is imperfect; training should reflect that. Figure 5 shows that always including gold context (P=100%) is not always optimal.
  - Chain-of-thought with quotes: Encourages explicit evidence use and transparency. Table 2 studies how including chain-of-thought affects performance.

Analogy to clarify: Instead of asking students to memorize facts (SFT only) or letting them bring the book on test day without practice (RAG only), RAFT is a study program where students practice answering questions by reading the textbook and ignoring irrelevant pages under exam-like conditions (Figure 1c).

## 4. Key Insights and Innovations
- Train for the target test condition (domain-specific “open book”) rather than for general QA
  - Novelty: The training data includes the same kind of retrieved context—mixtures of relevant and irrelevant documents—that the model will see at inference. This closes the gap between training and deployment (Figure 2).
  - Significance: Leads to large gains over both domain-specific SFT and off-the-shelf RAG across multiple domains (Table 1).
- Mixed-context curriculum: include some training examples without the gold document
  - Difference from prior work: Rather than always supplying gold evidence, RAFT intentionally withholds it for a fraction of training examples (Section 3; Figure 5).
  - Why it matters: Improves robustness to retrieval failures; Figure 5 shows that the optimal P% (share of examples with gold context) often lies below 100%.
- Evidence-grounded chain-of-thought with verbatim citations
  - Novelty: The reasoning chain contains explicit quoted spans, bracketed with `##begin_quote##` and `##end_quote##` to force grounding (Figure 3).
  - Significance: Improves accuracy and reduces overfitting to short answers on several datasets (Table 2), while making reasoning auditable.
- Robustness to variable top-k at test time by training with distractors
  - New capability: Training with distractors improves generalization when the number of retrieved documents changes at inference (Figure 6).
  - Why it’s important: Real RAG systems often vary top-k to balance recall and context length; RAFT maintains performance across such variations.

## 5. Experimental Analysis
- Datasets and tasks (Section 4)
  - Main evaluations (Table 1):
    - PubMedQA: biomedical yes/no QA based on biomedical abstracts.
    - HotpotQA: multi-hop QA over Wikipedia requiring reasoning across documents.
    - APIBench (Gorilla) subsets: HuggingFace Hub, Torch Hub, TensorFlow Hub—generate correct API calls from library documentation.
  - Additional analyses (Figures 5–6): Natural Questions (NQ) and TriviaQA to study the effect of gold-context proportion P% and distractor counts.
- Baselines (Section 4)
  - `LLaMA2-7B-chat` 0-shot (no documents).
  - `LLaMA2-7B-chat + RAG` (documents appended at inference).
  - `DSF` (domain-specific fine-tuning on Q/A pairs, no documents in-context).
  - `DSF + RAG` (DSF model with documents at inference).
  - `GPT-3.5 + RAG` as a stronger reference model.
- Main quantitative results (Table 1)
  - RAFT outperforms all baselines across domains:
    - PubMedQA: 
      > RAFT 73.30 vs DSF+RAG 71.6; LLaMA2+RAG 58.8; GPT‑3.5+RAG 71.60.
    - HotpotQA:
      > RAFT 35.28 vs DSF 6.38 and DSF+RAG 4.41; LLaMA2+RAG 0.03; GPT‑3.5+RAG 41.5 (RAFT trails GPT‑3.5 on this one metric but massively surpasses LLaMA2-based baselines).
    - HuggingFace Hub:
      > RAFT 74.00 vs DSF 61.06; DSF+RAG 42.59; LLaMA2+RAG 26.43; GPT‑3.5+RAG 29.08.
    - Torch Hub:
      > RAFT 84.95 vs DSF 84.94; DSF+RAG 82.80; LLaMA2+RAG 8.60; GPT‑3.5+RAG 60.21.
    - TensorFlow Hub:
      > RAFT 86.86 vs DSF 86.56; DSF+RAG 60.29; LLaMA2+RAG 43.06; GPT‑3.5+RAG 65.59.
  - Takeaway: On APIBench domains, RAFT dramatically improves over both adding RAG to a generic model and over domain-specific SFT; in PubMedQA it edges out GPT‑3.5+RAG. In HotpotQA, RAFT greatly improves over the LLaMA2 baselines but remains below GPT‑3.5+RAG.
- Does chain-of-thought help? (Table 2)
  - RAFT with CoT vs without CoT:
    - PubMedQA: 73.30 vs 68.30 (+5.00).
    - HotpotQA: 35.28 vs 25.62 (+9.66).
    - HuggingFace Hub: 74.00 vs 59.07 (+14.93).
    - TensorFlow Hub: 86.86 vs 83.21 (+3.65).
    - Torch Hub: 84.95 vs 86.56 (−1.61).
  - Assessment: CoT with citations usually helps substantially, except on Torch Hub where it slightly hurts. This suggests task-dependent effects; long reasoning might sometimes interfere with precise API string generation.
- Qualitative comparison (Figure 4)
  - HotpotQA example: the DSF model confuses a film title with the requested screenwriter; RAFT correctly identifies “David Weissman,” demonstrating better use of provided context.
- How much gold context to include during training? (Figure 5)
  - Varying P% (share of examples that include gold context) across NQ, TriviaQA, and HotpotQA shows that:
    - The optimum P% is not always 100%; different datasets peak around 40–100%.
    - Quote: “Results … suggest that mixing some amount of data [where] the golden document is not put in the context is helpful for in-domain RAG” (Figure 5 caption).
- Training with distractors improves robustness to test-time top‑k (Figure 6)
  - Models trained with gold-only underperform those trained with gold+distractors when evaluated with varying numbers of retrieved documents at test time.
  - Best training configuration differs by dataset (e.g., Natural Questions: `D* + 3` distractors; HotpotQA: `D* + 1` distractor).
  - The paper’s default recipe uses 1 gold + 4 distractors in training and a similar format at test (Section 5.1).
- Do the experiments support the claims?
  - Yes, on the central claim: RAFT improves in-domain, open-book performance and robustness to distractors. Evidence includes strong cross-domain gains (Table 1), clear ablations for CoT (Table 2), and sensitivity studies for P% and distractor counts (Figures 5–6).
  - Mixed result: CoT is not universally beneficial (Torch Hub).

## 6. Limitations and Trade-offs
- Domain specificity by design
  - Assumption: The same document collection is available for training and testing. RAFT does not aim to generalize to new domains or unseen corpora (Section 2, “Domain-Specific Open-Book Exam”). This limits applicability for open-domain RAG or rapidly changing corpora unless retraining is feasible.
- Reliance on curated training signals
  - Chain-of-thought with verbatim citations requires generating high-quality reasoning traces and finding the gold documents. The paper uses prompts to elicit these traces (Figure 3), and mentions GPT‑4‑1106 for CoT generation (Section 4.2). This introduces additional data preparation cost and potential style bias toward the prompt format.
- Compute and context length overhead
  - Training examples include multiple documents plus long reasoning targets, increasing token counts and compute. The paper does not report training cost or efficiency trade-offs.
- Retriever and metric details are abstracted
  - RAFT is retriever-agnostic (Section 3), but the study does not compare different retrievers or indexing strategies; results could vary with retrieval quality. Metrics for APIBench are reported as aggregate numbers (Table 1) without per-type breakdown or statistical significance.
- Chain-of-thought trade-off
  - While CoT with citations often helps (Table 2), it slightly hurts on Torch Hub, suggesting that, for some structured generation tasks (e.g., exact API syntax), verbose reasoning can interfere unless constrained.
- Model scale and generality
  - Main experiments fine-tune `LLaMA2-7B`; generality to larger/smaller models or other architectures is not shown. HotpotQA results remain below GPT‑3.5+RAG, indicating that model capacity still matters for complex multi-hop reasoning.

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
