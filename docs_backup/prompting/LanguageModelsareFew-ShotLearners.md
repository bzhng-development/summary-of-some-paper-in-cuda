# Language Models are Few-Shot Learners

**ArXiv:** [2005.14165](https://arxiv.org/abs/2005.14165)

## 🎯 Pitch

This paper introduces GPT-3, a 175-billion-parameter language model that can perform a wide range of tasks by simply seeing a few examples or natural language instructions—without any parameter updates or task-specific training. By dramatically increasing model size and leveraging a prompt-based, in-context learning framework, GPT-3 matches or exceeds state-of-the-art results in many NLP benchmarks and even demonstrates surprising abilities like arithmetic and analogies, suggesting a major shift toward more general, flexible, and human-like language understanding systems.

---

## 1. Executive Summary (2-3 sentences)
This paper introduces GPT‑3, a 175‑billion‑parameter autoregressive language model that performs a wide range of tasks by “in‑context learning” — using only natural‑language instructions and a few task demonstrations at inference time, without any gradient updates. By scaling model size and training data and standardizing a prompt‑based evaluation (zero‑, one‑, and few‑shot), GPT‑3 reaches strong or state‑of‑the‑art results on many benchmarks (e.g., LAMBADA, TriviaQA) and reveals new capabilities (on‑the‑fly arithmetic, analogies), suggesting a shift from task‑specific fine‑tuning to task‑agnostic prompting.

## 2. Context and Motivation
- Problem addressed
  - Modern NLP pipelines typically require task‑specific labeled datasets and fine‑tuning, which is costly, brittle out of distribution, and unlike how people learn from instructions or a handful of examples (Section 1).
  - Prior attempts at “meta‑learning” with language models (LMs) — performing new tasks via instructions/demonstrations at inference time — had limited success at practical scales (e.g., 4% on Natural Questions in earlier work cited in Section 1).

- Why it matters
  - Practical impact: Reduces dependence on large labeled datasets for every task (Section 1).
  - Scientific significance: Tests the hypothesis that increasing model capacity and data enables general in‑context learning, not just better language modeling (Figures 1.2 and 3.1).

- Prior approaches and their gaps
  - Fine‑tuning large pretrained models yields strong benchmark scores but can overfit to narrow task distributions and exploit dataset artifacts (Section 1; references [HLW+20], [MPL19]).
  - Early in‑context learning with smaller LMs showed promise but lagged far behind fine‑tuned systems (Section 1).

- Positioning of this work
  - Train and analyze a family of eight GPT‑3 models (125M → 175B parameters; Table 2.1), and systematically evaluate zero‑shot (instruction only), one‑shot (one example), and few‑shot (dozens of examples) prompting across >25 tasks (Section 3; Figure 2.1).

## 3. Technical Approach
Step‑by‑step, how GPT‑3 is built and evaluated:

- Model family and architecture
  - Autoregressive transformer with alternating dense and locally banded sparse attention (akin to Sparse Transformer; Section 2.1).
  - Eight sizes from `125M` to `175B` parameters; context window `nctx = 2048`; feedforward width is 4× `dmodel` (Table 2.1).
  - Training uses Adam, cosine LR decay, gradient clipping, and weight decay; mixed model parallelism to fit larger models (Section 2.3; Appendix B).

- Training data and preprocessing
  - A mixture emphasizing quality:
    - Filtered Common Crawl (~410B tokens post‑filter), WebText2, Books1/2, Wikipedia (Table 2.2).
    - Quality filter: a logistic‑regression classifier prefers web pages similar to curated corpora; fuzzy de‑duplication reduces redundancy (Appendix A).
    - Sampling weights favor high‑quality sources even if that means re‑exposing them multiple times (Table 2.2 “Epochs elapsed”), accepting mild overfitting for quality (Section 2.2).
  - Total training budget ~300B tokens for each model (Table 2.1).

- What “in‑context learning” means here
  - Definition: The model adapts at inference time to the task described in its input context (prompt) — a natural‑language instruction optionally followed by K example pairs (“demonstrations”) — and then completes or answers the next instance (Figures 1.1, 2.1).
  - No gradient updates are performed at evaluation; the “learning” occurs in a single forward pass conditioned on the prompt.

- Evaluation protocol (Section 2.4)
  - Few‑shot: randomly sample K training examples (typically 10–100, bounded by the 2048‑token context) and append a new test instance to complete.
  - One‑shot: same with K=1 plus an instruction.
  - Zero‑shot: instruction only.
  - Scoring conventions:
    - Multiple‑choice: compare the (length‑normalized) log‑likelihood of each option (Section 2.4).
    - For some datasets (ARC, OpenBookQA, RACE), normalize choice likelihood by its unconditional prior to reduce length/priors bias (Section 2.4).
    - Free‑form generation: beam search (width 4, length penalty α=0.6) with Exact Match, F1, or BLEU as appropriate (Section 2.4).
    - Task‑specific framings matter; e.g., formatting LAMBADA as fill‑in‑the‑blank enables one‑word completions (Section 3.1.2; Figure 3.2).

- Safety check: benchmark contamination analysis
  - Because pretraining data comes from the web, test sets may be present. A conservative n‑gram overlap filter marks “dirty” examples; performance is re‑computed on the “clean” subset (Section 4; Appendix C).
  - Most benchmarks show negligible change; a few (PIQA, Winograd, LAMBADA) are flagged (Figure 4.2).

Analogy to build intuition: Think of the prompt as a mini “instruction manual” plus a few worked examples that the model reads instantly before solving a new problem. Larger models “read and generalize” from these tiny manuals more effectively (Figure 1.2).

## 4. Key Insights and Innovations
- Scaling transforms prompting into a competitive alternative to fine‑tuning
  - Insight: Validation loss continues to follow a power‑law with compute/size (Figure 3.1), and — crucially — downstream few‑shot performance improves faster than zero‑shot as size grows (Figure 1.3; Figure 3.8 on SuperGLUE).
  - Significance: The largest model’s few‑shot scores approach or surpass fine‑tuned SOTA on several tasks without any gradient updates (e.g., LAMBADA and TriviaQA; Tables 3.2 and 3.3).

- Standardized, task‑agnostic prompting framework
  - Contribution: A uniform evaluation across zero‑/one‑/few‑shot settings with careful likelihood normalization and task‑specific prompt design (Section 2.4).
  - Why it matters: Shows that instruction wording, examples count `K`, and scoring choices materially affect results; provides replicable recipes for many tasks (Appendix G).

- Systematic contamination measurement
  - Method: Per‑dataset conservative overlap detection (up to 13‑grams; with special handling for short synthetic tasks) and re‑evaluation on clean subsets (Section 4; Appendix C).
  - Finding: Large potential overlaps do not necessarily inflate scores; when effects exist they are small (e.g., Winograd −2.6% absolute on the clean subset; Figure 4.2). PIQA and Winograd are explicitly annotated with asterisks (Tables 3.5, 3.6).

- Emergent test‑time skills beyond memorization
  - New capability: On‑the‑fly computation and pattern manipulation (e.g., 2–3 digit arithmetic, symbol insertion/anagrams) improves sharply with scale and number of demonstrations (Section 3.9; Figures 3.10, 3.11).
  - Evidence against rote memorization: Only 0.8% of 3‑digit addition test items were found verbatim in training data; common errors are procedural (e.g., missed carry), consistent with real computation (Section 3.9.1).

## 5. Experimental Analysis
- Evaluation setup
  - Datasets span language modeling/cloze, open‑domain and closed‑book QA, commonsense, reading comprehension, NLI, translation, and synthetic reasoning tasks (Sections 3.1–3.9).
  - Metrics: accuracy, F1, BLEU, perplexity; with detailed per‑task scoring rules (Section 2.4; Appendix G).
  - Baselines: Prior fine‑tuned SOTAs and strong pretrained models (e.g., T5‑11B, RoBERTa, ALUM); sometimes also human performance lines (Figures 3.2, 3.5, 3.6, 3.7).

- Headline quantitative results
  - Language modeling / completion
    - PTB (zero‑shot perplexity): `20.5` vs prior `35.8` (Table 3.1).
    - LAMBADA: few‑shot `86.4%` accuracy; zero‑shot `76.2%` (Table 3.2; Figure 3.2). Formatting as cloze is key to getting one‑word answers (Section 3.1.2).
    - HellaSwag: few‑shot `79.3%` vs SOTA `85.6%` (Table 3.2).
    - StoryCloze: few‑shot `87.7%` vs SOTA `91.8%` (Table 3.2).

  - Closed‑book QA (no retrieval, no fine‑tune)
    - TriviaQA (wiki split): zero‑shot `64.3%`, one‑shot `68.0%`, few‑shot `71.2%`, exceeding fine‑tuned T5‑11B closed‑book (`60.5%`) and matching a fine‑tuned open‑domain retriever‑generator in one‑shot (Table 3.3; Figure 3.3).
    - WebQuestions: few‑shot `41.5%` approaching fine‑tuned T5‑11B+SSM `44.7%` (Table 3.3).
    - Natural Questions: few‑shot `29.9%` below fine‑tuned T5‑11B+SSM `36.6%`, with large gains from zero→few‑shot suggesting distribution mismatch that prompts partially fix (Table 3.3).

  - Translation (unsupervised/few‑shot)
    - Few‑shot GPT‑3 outperforms prior unsupervised NMT into English (e.g., Ro→En `39.5` BLEU; De→En `40.6` BLEU using multi-bleu) but lags when translating into other languages (e.g., En→Ro `21.0` BLEU) (Table 3.4; Figure 3.4).
    - Directional asymmetry aligns with GPT‑3 being a stronger English LM and with byte‑level BPE subword choices (Section 3.3).

  - Winograd‑style coreference
    - Classic Winograd: ~`89%` across settings, with small contamination caveat (Table 3.5).
    - Adversarial Winogrande: few‑shot `77.7%`, close to fine‑tuned RoBERTa‑large (`79%`) but below overall SOTA `84.6%` (Table 3.5; Figure 3.5).

  - Commonsense QA
    - PIQA: few‑shot `82.8%` exceeding leaderboard baseline (but flagged for possible training overlap; Table 3.6; Figure 3.6).
    - ARC‑Challenge: few‑shot `51.5%` (well below SOTA `78.5%`), but approaching fine‑tuned RoBERTa baseline (Table 3.6).
    - OpenBookQA: few‑shot `65.4%`, below SOTA `87.2%` (Table 3.6).

  - Reading comprehension
    - CoQA: few‑shot `85.0` F1, within ~6 points of fine‑tuned SOTA (`90.7`) (Table 3.7; Figure 3.7).
    - SQuADv2: few‑shot `69.8` F1; zero→few‑shot gain ~10 points (Table 3.7).
    - QuAC and RACE: relatively weak (`44.3` F1 and ~`47%`/`58%` accuracy respectively; Table 3.7).

  - SuperGLUE (test set)
    - Overall few‑shot score `71.8`, competitive with fine‑tuned BERT‑Large (`69.0`) but behind SOTA (`89.0`) (Table 3.8).
    - Strong tasks: COPA (`92.0` acc), ReCoRD (acc `90.2`, F1 `91.1`).
    - Weak task: WiC near random (`49.4%`), and middling RTE/CB (Table 3.8; development‑set scaling in Figure 3.8).

  - NLI (ANLI)
    - Rounds 1–2 near chance for all model sizes; Round 3 few‑shot reaches `40.2%` on dev, roughly halfway from chance to SOTA (Figure 3.9; Appendix H).

  - Synthetic reasoning and qualitative probes
    - Arithmetic (few‑shot): 2‑digit add/sub = `100%`/`98.9%`; 3‑digit add/sub = `80.4%`/`94.2%`; 4‑digit ~`26%`; 5‑digit ~`10%`; 2‑digit multiplication `29.2%` (Table 3.9; Figure 3.10).
    - Scrambling: Symbol‑insertion removal `67.2%`; cycling letters `37.9%`; reversing words remains near zero (Table 3.10; Figure 3.11).
    - SAT analogies: few‑shot `65.2%` vs historical human average `57%` (Figure 3.12).
    - News generation: humans identify GPT‑3 articles at ~`52%` accuracy (chance is 50%), vs `86–88%` on a deliberately bad control model (Tables 3.11, 3.12; Figure 3.13).

- Do the experiments support the claims?
  - Yes, for the central claim that scaling improves in‑context learning efficiency and utility:
    - Smooth scaling trends across tasks and K (Figures 1.2, 3.1, 3.3, 3.4, 3.5, 3.8).
    - Strong few‑shot performance on diverse tasks with no gradient updates.
  - Robustness/diagnostics:
    - Contamination analysis often shows negligible impact, with explicit caveats where effects might exist (Figure 4.2; Section 4).
    - Mixed results highlight boundaries: NLI (ANLI), WiC, some reading comprehension datasets remain challenging (Sections 3.6–3.8).

- Trade‑offs and conditions
  - Task format sensitivity: prompt phrasing and K strongly affect outcomes (LAMBADA cloze formatting; Section 3.1.2).
  - Directional biases in translation; reliance on English‑centric training distribution (Section 3.3).
  - Comparison‑style tasks (WiC, ANLI) remain weak, possibly reflecting limitations of left‑to‑right decoding for sentence pair comparison (Section 5).

## 6. Limitations and Trade-offs
- Modeling and objective assumptions (Section 5)
  - Pure autoregressive, unidirectional objective may hinder tasks requiring bidirectional context or explicit comparison (WiC, RTE, ANLI).
  - Self‑supervised next‑token prediction equally weights all tokens; lacks grounding and goal‑directed objectives (Section 5).

- Data and distributional constraints
  - Training data is ~93% English; translation from English lags translation into English (Section 3.3).
  - Potential train–test contamination exists; while measured impacts are mostly small, a few benchmarks are affected (Section 4).

- Capability boundaries revealed by experiments
  - NLI and some reading comprehension remain far from SOTA (Tables 3.7, 3.8; Figure 3.9).
  - Symbolic operations scale with digits/complexity; performance drops on 4–5 digit arithmetic and multiplication (Table 3.9).

- Compute and efficiency
  - Training requires several thousand PF‑days for 175B (Figure 2.2; Appendix D). Inference is expensive and latency is high without distillation.

- Social and fairness considerations (Section 6)
  - Bias: Gendered occupation associations skew male; race‑related sentiment disparities (Figure 6.1; Section 6.2.1–6.2.2; Table 6.1).
  - Misuse risks: Difficulty of human detection of generated news (Tables 3.11–3.12) raises concerns for misinformation.
  - Energy usage and environmental impact (Section 6.3).

## 7. Implications and Future Directions
- How this work changes the landscape
  - Establishes prompting — not fine‑tuning — as a viable, sometimes superior way to adapt large LMs to new tasks (Figures 1.3, 3.8).
  - Normalizes the practice of zero/one/few‑shot evaluation with explicit prompts and scoring procedures (Section 2.4; Appendix G).
  - Validates scaling laws as a predictor not only of loss but also of emergent in‑context capabilities (Figure 3.1).

- Follow‑up research enabled/suggested
  - Architectural/Objective advances:
    - Bidirectional or encoder‑decoder models at GPT‑3 scale to improve sentence‑pair and span‑selection tasks (Section 5).
    - Augment next‑token prediction with targeted objectives (entity/span prediction, reasoning) or RL from human feedback (Section 5; references [ZSW+19a]).
    - Retrieval‑augmented prompting to combine parametric knowledge with external memory (Table 3.3 context; [LPP+20]).
  - Data and training:
    - More balanced multilingual corpora; subword vocabularies tuned for non‑English languages (Section 3.3).
    - Larger context windows and memory mechanisms to support long‑document reasoning.
  - Efficiency and deployment:
    - Distillation and sparsity to reduce inference cost while keeping few‑shot behaviors (Section 5).
    - Better methods for automatic prompt construction and selection; learning to prompt.

- Practical applications and use cases
  - Low‑label or label‑free adaptation: QA, summarization, translation, grammar correction, and domain‑specific text generation via prompts (Sections 3.1–3.7, 3.9.6).
  - Rapid prototyping: New tasks can be specified by plain‑language instructions and a handful of examples (Appendix G).
  - Cautionary deployment: Monitoring for bias and misuse is necessary given near‑indistinguishable generated news and measured demographic skews (Section 6; Tables 3.11, 6.1).

> In short, the paper demonstrates that “scale + prompting” is a powerful recipe: a single, task‑agnostic model trained once can adapt in seconds to many tasks through natural‑language instructions and a few examples, achieving competitive performance across a broad spectrum of benchmarks (Figures 1.2, 1.3; Section 3).
