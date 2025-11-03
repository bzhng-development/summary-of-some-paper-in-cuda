# Language Models are Few-Shot Learners

**ArXiv:** [2005.14165](https://arxiv.org/abs/2005.14165)

## 🎯 Pitch

This paper introduces GPT-3, a groundbreaking 175-billion–parameter language model that can perform a wide range of NLP tasks simply by conditioning on natural language instructions and a handful of examples—without task-specific fine-tuning. By demonstrating that scaling up model size alone enables strong 'in-context learning', GPT-3 achieves or surpasses state-of-the-art results in translation, question answering, and more, fundamentally reducing the need for large labeled datasets and bringing machines closer to human-like learning flexibility. This work signals a paradigm shift toward more general-purpose, adaptable AI systems with broad practical and scientific impact.

---

## 1. Executive Summary (2-3 sentences)
This paper introduces `GPT-3`, a 175-billion–parameter transformer language model trained to perform tasks by conditioning on natural-language instructions and a few examples in its input context—without any task-specific fine-tuning. It demonstrates that simply scaling model size yields strong “in-context learning” across translation, question answering, cloze completion, and on-the-fly reasoning, sometimes rivaling or surpassing fine-tuned systems (e.g., state-of-the-art results on LAMBADA and TriviaQA few-shot), thereby reducing dependence on large labeled datasets (see Sections 1–3; Figures 1.2, 3.1; Tables 3.2–3.4).

## 2. Context and Motivation
- Problem addressed:
  - Modern NLP systems rely on “pre-train then fine-tune” pipelines that require thousands of labeled examples per downstream task. Humans, by contrast, can learn new tasks from instructions or a few demonstrations (Section 1).
  - The field lacks robust, general-purpose models that can adapt to new tasks “on the fly” using only textual context—called `in-context learning` (task instructions and demonstrations placed directly in the model’s input sequence, no weight updates).
- Why this matters:
  - Practical: Labeled data are expensive or unavailable for many tasks; supporting instructions + few examples makes NLP systems more broadly applicable.
  - Scientific: Fine-tuning on narrow distributions risks exploiting spurious correlations and weak out-of-distribution generalization (Section 1; discussion citing [MPL19], [HLW+20]).
- Prior approaches and gaps:
  - Pretrained LMs (e.g., GPT-2, BERT/T5) excel when fine-tuned but perform weakly in zero-/few-shot regimes (Section 1; [RWC+19], [RSR+19]).
  - Early `in-context learning` results showed promise but lagged far behind fine-tuning on many tasks (Section 1).
- Positioning:
  - This work tests whether simply scaling an autoregressive LM (no new training objective) yields strong `in-context learning`. It trains models from 125M to 175B parameters and studies performance without any gradient updates on downstream tasks—only prompts plus zero/one/few demonstrations (Sections 2–3).

Key terms
- `in-context learning`: At inference time, the model reads a sequence containing a task description and a few input–output examples, then continues the sequence by producing the answer for a new input (no parameter updates) (Figure 1.1; Section 2).
- `zero-shot / one-shot / few-shot`: Number of demonstrations provided in the prompt K=0/1/10–100, respectively (Section 2; Figure 2.1).

## 3. Technical Approach
Step-by-step overview

1) Model architecture and scale
- Architecture: Autoregressive transformer similar to GPT-2, with alternating dense and locally-banded sparse attention layers (inspired by Sparse Transformer) to improve efficiency (Section 2.1).
- Largest model `GPT-3 175B`: 96 layers, model width 12,288, 96 attention heads; trained on 300B tokens (Table 2.1).
- Context window: 2,048 tokens for all models (Section 2.1).

2) Training data and quality control
- Corpus blend: Filtered Common Crawl + curated WebText2, two internet book corpora (Books1, Books2), and English Wikipedia (Table 2.2).
- Quality filtering:
  - Train a logistic regression classifier to identify high-quality Common Crawl documents by similarity to reference corpora; reweight sampling using a Pareto criterion to favor high-quality documents (Appendix A; Section 2.2).
  - Fuzzy deduplication within and across datasets using MinHash to reduce redundancy and overfitting (Appendix A).
- Sampling strategy: Up-weight curated sources so they are seen 2–3×, while Common Crawl/Books2 are seen <1× during the 300B-token run (Table 2.2).

3) Training process
- Optimization: Adam; cosine LR decay; gradient clipping; weight decay; linear warmup and batch-size ramp-up (Appendix B; Section 2.3).
- Compute: Scaling laws guide training; GPT-3 175B used ~3.6×10^3 petaflop/s-days (Figure 2.2). Validation loss continues to follow a power law with compute/size (Figure 3.1).

4) How inference-time `in-context learning` is implemented
- Prompting formats (Section 2.4; Figure 2.1):
  - `Zero-shot`: only a natural-language instruction.
  - `One-shot`: instruction + 1 demonstration pair.
  - `Few-shot`: instruction + K demonstrations (K set by prompt length limit, often 10–100).
- Scoring strategies:
  - Multiple-choice tasks: Compare per-token log-likelihood of each candidate completion after the prompt. Some tasks add a normalization factor dividing by the unconditional probability of the completion to reduce length and prior bias (Section 2.4).
  - Free-form generation (e.g., QA spans): Use beam search (beam=4, length penalty α=0.6) and evaluate with F1, BLEU, or exact match as appropriate (Section 2.4).
- Example of task reframing:
  - For LAMBADA (predict final word), the model does better when the task is formatted as a cloze “fill-in-the-blank” with demonstrations, enabling it to infer “output exactly one word” (Section 3.1.2; Figure 3.2).

Why this approach?
- Hypothesis: In-context learning ability is a byproduct of larger capacity and broader pattern acquisition during pretraining; scaling should unlock stronger prompt-based adaptation (Section 1; Figure 1.2).
- Design minimizes changes: No new objective, no intermediate fine-tuning; isolates the effect of scale + prompting (Sections 1–2).

## 4. Key Insights and Innovations
1) Scaling alone substantially improves `in-context learning`
- Evidence: Larger models make increasingly efficient use of context examples and instructions (Figure 1.2). Validation loss follows a smooth power law as compute/model size increase (Figure 3.1).
- Impact: Few-shot performance often approaches or matches fine-tuned systems on certain tasks without gradient updates (e.g., TriviaQA few-shot; Table 3.3).

2) Prompting as a general interface for many tasks
- Innovation: A single autoregressive LM, prompted appropriately, performs translation, QA, reading comprehension, cloze, arithmetic, and synthetic symbol manipulation—without any task-specific parameters (Sections 3.1–3.9).
- Significance: Reduces the need for bespoke architectures or fine-tuning pipelines for each task; enables rapid prototyping by prompt engineering alone.

3) Systematic contamination analysis at internet scale
- Procedure: N-gram overlap mining between training data and each test set to produce “clean” subsets; evaluate performance differences to estimate contamination effects (Section 4; Appendix C).
- Findings: While some datasets show nontrivial overlap (e.g., Winograd 45%, PIQA 29%), most score changes on clean subsets are small (Figure 4.2). Noted exceptions are flagged with asterisks in results tables (e.g., Winograd/PIQA; Tables 3.5–3.6).
- Contribution: Establishes a conservative methodology to quantify and report potential train-test leakage in web-scale pretraining regimes.

4) New evidence that LMs can perform nontrivial computation and pattern manipulation in context
- Arithmetic: `GPT-3 175B` solves 2-digit addition with 100% few-shot accuracy and 3-digit subtraction with 94.2%, with non-negligible performance even on 4–5-digit operations and 2-digit multiplication (Table 3.9).
- Symbol manipulation: Strong few-shot performance on tasks like removing random insertions in words (67.2%) and constrained anagramming (39.7%), while failing at strict reversal (0.44%) (Table 3.10; Figure 3.11).
- Significance: Suggests the model learns task procedures from demonstrations at test time, not just recalling training facts.

5) Human evaluation of long-form generation shows near-chance detectability
- In a news-generation Turing-style test with ~200-word articles, human judges identify GPT-3 outputs as machine-written only 52% of the time (chance=50%), compared to 86% for a deliberately bad control model (Table 3.11; Figure 3.13). Similar results hold for ~500-word articles (Table 3.12).
- Implication: Scaled LMs can produce genre-conforming text that humans find hard to detect without specialized tools.

## 5. Experimental Analysis
Evaluation setup
- Settings: `zero-shot`, `one-shot`, and `few-shot`; K tuned within the 2,048-token budget (Section 2.4).
- Metrics: Accuracy, F1, BLEU, perplexity, and exact match depending on the dataset (Sections 3.1–3.9).
- Baselines/Comparators: Published fine-tuned SOTAs (e.g., T5-11B with task-specific fine-tuning), earlier LMs (GPT-2), and specialized systems like retrieval-augmented QA (RAG) (Tables 3.1–3.8).

Main quantitative results (selected highlights with citations)
- Language modeling & cloze/completion:
  > PTB perplexity improves to 20.5 (zero-shot) vs prior 35.8 (Table 3.1).  
  > LAMBADA (few-shot): 86.4% accuracy and 1.92 perplexity; zero-shot 76.2% (Table 3.2; Figure 3.2).  
  > StoryCloze (few-shot): 87.7% (SOTA=91.8) (Table 3.2).  
  > HellaSwag (few-shot): 79.3% (SOTA 85.6; surpasses earlier fine-tuned 1.5B LM) (Table 3.2).

- Closed-book QA (no retrieval, no fine-tuning):
  > TriviaQA: 64.3% (zero), 68.0% (one), 71.2% (few) — few-shot matches/exceeds fine-tuned open-domain RAG (68.0%) and exceeds fine-tuned closed-book T5-11B+SSM (60.5%) (Table 3.3; Figure 3.3).  
  > WebQuestions: 41.5% (few) vs 44.7% for fine-tuned T5-11B+SSM (Table 3.3).  
  > Natural Questions: 29.9% (few) vs 36.6% for fine-tuned T5-11B+SSM (Table 3.3).

- Translation (few-shot, 64 examples in prompt; no back-translation or parallel data fine-tuning):
  > En→Fr 32.6 BLEU; Fr→En 39.2; En→De 29.7; De→En 40.6; En→Ro 21.0; Ro→En 39.5 (Table 3.4; Figure 3.4).  
  Performance into English is substantially better than out of English; results competitive with unsupervised NMT methods, but below fully supervised SOTAs.

- Winograd-style coreference:
  > Winograd (WSC273): ~88–90% across settings; flagged for contamination (Table 3.5).  
  > Winogrande (adversarial): 77.7% (few) vs 84.6% SOTA fine-tuned T5; strong scaling trend (Figure 3.5; Table 3.5).

- Commonsense reasoning:
  > PIQA: 82.8% (few) on test server; flagged for potential contamination; still above fine-tuned RoBERTa baseline (Table 3.6; Figure 3.6).  
  > ARC-Easy/Challenge: 70.1% / 51.5% (few); lower than SOTA multi-task fine-tuned systems (Table 3.6).  
  > OpenBookQA: 65.4% (few), similar to fine-tuned BERT-Large baselines but below SOTA (Table 3.6).

- Reading comprehension:
  > CoQA: 85.0 F1 (few) — within ~6 points of SOTA; strong scaling with model size and shots (Figure 3.7; Table 3.7).  
  > SQuAD2.0: 69.8 F1 (few); ~10 F1 gain from zero-shot; still far from SOTA (Table 3.7).  
  > DROP, QuAC, RACE: results lag far behind SOTA; suffer especially when tasks require precise span extraction or multi-sentence reasoning (Table 3.7).

- SuperGLUE (test server, K=32 per task):
  > Overall: 71.8 (few-shot) vs 69.0 for a fine-tuned BERT-Large; still below overall SOTA 89.0 (Table 3.8).  
  > Strong tasks: COPA 92.0 (near SOTA), ReCoRD F1 91.1 (near SOTA).  
  > Weak tasks: WiC 49.4% (near chance), CB F1 52.0, RTE 69.0 — sentence-pair reasoning remains hard (Table 3.8).  
  > Performance rises with both model size and number of in-context examples (Figure 3.8).

- NLI (adversarial):
  > ANLI Round 3 (dev): 40.2% (few) vs random ~33%; smaller models at chance — indicates only nascent capability (Figure 3.9; Appendix H).

- Synthetic reasoning:
  > Arithmetic (few-shot, GPT-3 175B): 2-digit add/sub ~100/98.9%; 3-digit add/sub ~80.4/94.2%; 4–5 digit tasks 9–27%; 2-digit multiplication 29.2% (Table 3.9).  
  > SAT analogies: 65.2% (few) — above historical human average of ~57% (Figure 3.12; Section 3.9.3).  
  > Word scrambling: strongest on “random insertion removal” and constrained anagrams; fails on full reversal (Table 3.10; Figure 3.11).

- Human detection of generated news:
  > Human accuracy ~52% for GPT-3 175B at ~200 and ~500 words; control condition 86–88% (Tables 3.11, 3.12; Figure 3.13).

Robustness and checks
- Contamination analysis:
  > “Clean” vs “all” performance deltas are generally small even when overlap rates are high, suggesting limited inflation from leakage; exceptions noted (Figure 4.2; Section 4; Table C.1).  
  > Some benchmarks (e.g., Wikipedia-derived LMs, CBT) were dropped due to pervasive overlap (Section 4).

- Scaling trends:
  > Smooth improvements with model size across most tasks and settings (Figures 3.1, 3.3–3.11; Appendix H), supporting the scaling hypothesis.

Do the experiments support the claims?
- Yes for the central claim: scaling unlocks strong few-shot performance across many tasks without fine-tuning (e.g., LAMBADA, TriviaQA, COPA, CoQA; Tables 3.2–3.8).
- Mixed for broad generality: tasks requiring sentence-pair comparisons or structured span selection remain challenging (WiC, ANLI, RACE, DROP; Tables 3.7–3.8).

## 6. Limitations and Trade-offs
Assumptions and scope
- Assumes that increased parameter count and diverse web-scale pretraining suffice to induce general in-context learning; no task-specific objectives or architectures are added (Sections 1–2).

Where it struggles
- Sentence-pair reasoning and fine-grained reading comprehension (WiC, ANLI, QuAC, RACE) remain weak even few-shot (Tables 3.7–3.8). The uni-directional autoregressive setup may be suboptimal for tasks that benefit from bidirectionality or explicit comparison (Section 5).

Compute and data costs
- Training `GPT-3 175B` required several thousand petaflop/s-days (Figure 2.2), raising accessibility and environmental concerns (Section 6.3). Pretraining consumes vastly more text than a human reads in a lifetime (Section 5).

Sample efficiency vs. pretraining efficiency
- Test-time sample efficiency (few examples suffice) contrasts with low pretraining sample efficiency (hundreds of billions of tokens) (Section 5).

Bias, fairness, and safety
- Model reflects internet-scale biases (gender, race, religion); co-occurrence and sentiment analyses show stereotyped associations (Section 6.2; Table 6.2; Figure 6.1).  
- Generated text can be hard to distinguish from human-written (Tables 3.11–3.12), raising misuse risks for misinformation, spam, and social engineering (Section 6.1).

Train–test contamination
- Although carefully analyzed, overlap remains a confounder on a few datasets (PIQA, Winograd, parts of LAMBADA), and filtering bugs affected long documents (Section 4).

Interpretability and controllability
- Decisions are not easily interpretable; outputs may be unfaithful or inconsistent over long contexts; occasional non-sequiturs and repetitions persist (Section 5).

## 7. Implications and Future Directions
How this work shifts the field
- Establishes `prompting + scale` as a credible alternative to task-specific fine-tuning for many benchmarks. It reframes the interface to language models: instruct and demonstrate within the input rather than modify weights (Sections 1–3; Figure 2.1).
- Validates the scaling-law view: increasing parameters/computation yields predictable gains not just in loss, but in versatile few-shot capability (Figure 3.1; Figure 1.2).

Enabling follow-up research
- Improving weak spots with architectural or training changes:
  - Bidirectional or encoder–decoder variants at GPT-3 scale may close gaps on sentence-pair and span-extraction tasks (Section 5).
  - Retrieval-augmented prompting could combine in-context learning with access to external knowledge (compare with RAG in Table 3.3).
- Making models more usable:
  - `Distillation` to smaller models for deployment (Section 5; [HVD15]).  
  - Better prompt design tools, automatic prompt search, or learned prompting.
  - Calibration and uncertainty estimation for few-shot predictions.
- Safety and ethics:
  - Bias mitigation techniques tailored to generative, prompt-driven settings (Section 6.2.4).  
  - Detection and watermarking methods for generated text (Section 3.9.4 discussion referencing GLTR/GROVER).
  - Release practices, governance, and monitoring for misuse scenarios (Section 6.1).

Practical applications
- Few-shot “universal” NLP assistant that can:
  - Translate between many languages from a handful of examples (Table 3.4).  
  - Answer knowledge questions without retrieval in constrained domains (Table 3.3).  
  - Draft and edit text (grammar correction, summarization-by-prompting, style emulation), with quality sufficient to require careful human review (Figure 3.17; Tables 3.11–3.12).

Overall, the paper shows that scale unlocks a new operating mode for language models—`in-context learning`—that meaningfully reduces reliance on labeled datasets and task-specific fine-tuning. The approach is not universally strong and comes with real costs and risks, but it reshapes both research agendas and application design around prompting as a primary control surface for general-purpose language models.
