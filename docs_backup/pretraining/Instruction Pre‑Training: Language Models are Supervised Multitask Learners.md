# Instruction Pre‑Training: Language Models are Supervised Multitask Learners

**ArXiv:** [2406.14491](https://arxiv.org/abs/2406.14491)
**Authors:** Daixuan Cheng, Yuxian Gu, Shaohan Huang, Junyu Bi, Minlie Huang, Furu Wei
**Institutions:** Microsoft Research, Tsinghua University

## 🎯 Pitch

The paper introduces Instruction Pre-Training (Instruct PT), a novel approach that injects synthesized instruction-response pairs into the pre-training phase of language models, significantly enhancing both base model quality and domain adaptation capabilities. This advancement allows models to achieve performance levels comparable to those ten times larger, delivering scalable, cost-effective solutions for specialized fields like biomedicine and finance, and transforming the landscape of multitask learning at scale.

---

## 1. Executive Summary
This paper introduces Instruction Pre-Training (Instruct PT): instead of pre-training a language model only on raw text, the raw text is augmented with many “instruction–response” pairs that are automatically synthesized from the text, and the model is then pre-trained on this richer signal using the usual next-token objective. The approach delivers better base models and stronger downstream gains after instruction tuning; in continual pre-training for domains like biomedicine and finance, an 8B model trained with Instruct PT matches or surpasses a 70B model (Table 3), while using a cost-effective, open-source 7B instruction synthesizer.

## 2. Context and Motivation
- Problem/gap
  - Standard “vanilla” pre-training optimizes next-token prediction on raw corpora without explicit supervision across tasks. This scales well but under-exposes models to diverse, structured tasks during pre-training.
  - Instruction tuning (post-training on curated instructions) improves generalization, but it is a separate, later stage and still relies on relatively modest amounts of supervised data.

- Importance
  - If supervised multitask signals could be injected directly into pre-training at web scale, models might learn task formats, instruction following, and grounding earlier, improving data efficiency and downstream adaptability. This matters for both general models and domain-specialized models in areas like medicine and finance.

- Prior approaches and shortcomings
  - Synthetic instruction generation has focused on post-training (e.g., Self-Instruct, FLAN-style collections), often relying on large or closed models to create data, which is costly and hard to scale (Section 1; Related Work).
  - Rule-based domain adaptation (e.g., converting domain texts into reading-comprehension style tasks) improves over pure domain text but yields limited diversity (Table 4).
  - There is little work on moving supervised multitask learning into the pre-training stage at scale while keeping costs manageable.

- Positioning
  - This paper proposes a scalable, supervised multitask pre-training framework: automatically synthesize instruction–response pairs tied to each raw text using a compact, fine-tuned open-source model (`Mistral-7B-v0.1`), then pre-train on the augmented data (Figures 1–3). It aims to combine the breadth of web data with the structure and diversity of instructions.

## 3. Technical Approach
The method has two main components: an instruction synthesizer and the pre-training procedure on instruction-augmented corpora.

- Key terms (paper-specific)
  - `Instruction–response pair`: a natural-language instruction (task prompt) and the expected answer, both grounded in a given raw text.
  - `Instruction synthesizer`: a language model tuned to generate diverse, text-grounded instruction–response pairs from a given raw text.
  - `Few-shot example`: a preamble of several instruction–response pairs presented before a new instruction, so the model can learn the pattern from examples.
  - `Continual pre-training` (a.k.a. domain-adaptive pre-training): continuing to train an already pre-trained model on additional, domain-specific data.

A. Building the instruction synthesizer (Section 2.1; Figures 2–3; Appendix A–B)
- Data reformulation
  - A large set of existing, context-based QA/NLI/RC datasets is reformatted so that for each context (the “raw text”), multiple tasks (the “instruction–response pairs”) are attached. The sources span encyclopedic, social media, fiction, academic tests, expert materials, etc., and cover free-form, multiple-choice, and chain-of-thought formats (Figure 7).
- Fine-tuning strategy
  - Base model: `Mistral-7B-v0.1`.
  - Training examples are constructed as “few-shot” sequences: concatenate several one-shot examples from the same dataset so the synthesizer learns consistent patterns per dataset (Figure 3, top-left).
  - Loss is computed only on the instruction–response tokens, not on the raw text (Figure 3, “Compute loss only on the instruction-response pairs”), encouraging the model to focus on generating useful tasks rather than copying context.
  - Templates explicitly separate parts and unify formats: `<CON>...</CON>` for context, `<QUE>...</ANS>...</END>` for instructions and answers; variants cover multiple-choice and chain-of-thought (Appendix B, Table 7).
- Inference (task synthesis at scale)
  - Multi-round inference creates few-shot style inputs automatically: in round m, the model receives the current raw text plus the text and synthesized pairs from earlier rounds (Figure 3, “Round 1…Round M”). This prompts the synthesizer to produce more pairs following the earlier patterns.
  - Through this, ~5 pairs are created per raw text on average, each ~52 tokens (Section 3.1), enabling large-scale augmentation (200M pairs synthesized; Abstract; Section 3.2).

B. Instruction Pre-Training (Section 2.2; Figure 3, bottom)
- Data assembly
  - For general pre-training from scratch, a 100B-token subset of RefinedWeb is used (200M texts). One-fifth of the texts (40M) are converted to instruction-augmented form via two synthesis rounds, yielding ~200M pairs, ~10B tokens (Section 3.2).
  - The small fine-tuning corpus used to train the synthesizer (0.2B tokens) is mixed in with higher sampling (repeats 4×) for task diversity (Section 3.2).
  - For continual pre-training, all domain corpora (PubMed abstracts; finance news) are converted via three rounds and mixed with a general instruction set (Section 3.3).
- Training recipe
  - Pre-train the target LM with the standard next-token objective on the concatenation of raw text and synthesized instruction–response pairs (loss on all tokens during LM pre-training, unlike the synthesizer’s tuning) (Section 2.2).
  - From-scratch models: Mistral-style architectures at 500M and 1.3B parameters (Section 3.2; Table 14). Efficient attention is enabled (xformers).
  - Continual pre-training: continue training `Llama3-8B` on the domain mixtures (Table 14; Section 3.3).

Why these design choices?
- Ground all synthesized tasks in the raw text to ensure correctness and coverage (Abstract; Section 2.1).
- Multi-round synthesis creates implicit few-shot structure without manual curation (Figure 3), improving prompting ability (supported by ablations; Table 4).
- Use an open-source 7B synthesizer for cost efficiency, enabling 200M+ pairs (Section 3.1), in contrast to prior work that depends on very large/closed models.

## 4. Key Insights and Innovations
1) Moving supervision into pre-training at scale
- What’s new: Rather than limiting instructions to the post-training phase, the method injects diverse, context-grounded tasks directly into pre-training data (Figures 1–3).
- Why it matters: It aligns pre-training with the later instruction-tuning objective, which speeds up and improves downstream instruction tuning (Figure 4), and it improves data efficiency (Table 2).

2) A compact, open-source instruction synthesizer that generalizes
- What’s new: A `Mistral-7B` model is few-shot tuned on a diverse “context + many tasks” reformulation of existing datasets and then used to synthesize tasks for arbitrary raw text (Figure 2; Appendix A).
- Evidence: On seen and unseen datasets, the synthesizer’s response accuracy and pair quality far exceed the base `Mistral-7B` (Table 5). Including synthesized pairs in prompts improves a separate LM’s performance on both seen and unseen tasks (Figure 5).
- Significance: Cost-effectiveness enables 200M pairs across 40–50 task categories (Abstract; Table 6; Figure 6), which prior closed/very-large-model approaches struggled to scale.

3) Multi-round synthesis to produce few-shot structure automatically
- What’s new: By conditioning later rounds on earlier synthesized pairs, each raw text yields an M-shot training example (Figure 3).
- Why it matters: Few-shot structure improves prompting ability; ablations show single-turn (1-shot) synthesis is worse than multi-round (Table 4).

4) Domain-adaptive continual pre-training with large gains
- What’s new: Converting all domain corpora (biomedicine/finance) into instruction-augmented texts and mixing with general instructions yields large improvements for `Llama3-8B`, approaching or surpassing `Llama3-70B` (Table 3).
- Significance: Demonstrates a powerful, compute-efficient path to domain specialization: better than vanilla continual pre-training and, in finance, exceeding a model an order of magnitude larger (74.7 vs 71.9 average; Table 3).

## 5. Experimental Analysis
A. Evaluation methodology
- Settings
  - From-scratch pre-training: 500M and 1.3B models trained on 100B tokens (RefinedWeb subset; Section 3.2; Table 14).
  - Continual pre-training: `Llama3-8B` on PubMed and finance corpora (3 synthesis rounds), mixed with general instructions (Section 3.3).
- Benchmarks and metrics
  - General tasks: ARC-e/c, BoolQ, SIQA, WinoGrande, PIQA, OBQA, HellaSwag, MMLU via lm-evaluation-harness; acc-norm for MC tasks (Appendix C).
  - Instruction tuning: MMLU zero/few-shot during fine-tuning on FLAN-style data (Figure 4).
  - Bio/Finance tasks: PubMedQA, ChemProt, RCT, MQP, UMSLE; ConvFinQA, Headline, FiQA SA, FPB, NER (Appendix C).
- Baselines
  - “Vanilla PT”: same token budget, raw corpora only.
  - “Mix PT”: raw corpora + the synthesizer’s fine-tuning data only (controls for simply mixing supervised data).
  - External baselines: GPT-2, Pythia-1B, BLOOM (Tables 2 and 15).

B. Main quantitative results
- General pre-training (Table 1)
  - 500M model: Instruct PT outperforms Vanilla PT on most tasks:
    - BoolQ: 62.0 vs 57.5 (+4.5)
    - SIQA: 47.2 vs 44.6 (+2.6)
    - WinoGrande: 54.8 vs 53.8 (+1.0)
    - OBQA: 30.8 vs 29.8 (+1.0)
    - PIQA dips slightly: 69.9 vs 71.1 (−1.2)
    - MMLU similar: 25.3 vs 25.4
  - 1.3B model: broader gains, including MMLU:
    - ARC-e/c: 60.5/30.9 vs 58.5/28.8
    - BoolQ: 62.2 vs 60.3
    - SIQA: 49.2 vs 47.9
    - PIQA: 73.6 vs 73.0
    - MMLU: 27.3 vs 25.7 (+1.6)
- Data efficiency (Tables 2 and 15)
  - With 100B tokens, Instruct PT 500M ≈ Pythia-1B at 300B tokens (average 46.6 vs 47.1).
  - Instruct PT 1.3B ≈ BLOOM-3B at 341B tokens (average 49.7 vs 50.1).
  - Implication: similar performance with 3× fewer tokens and ~2–3× fewer parameters.
- Synergy with instruction tuning (Figure 4)
  - During instruction tuning on FLAN data, MMLU zero/few-shot for Instruct PT rises faster and maintains a higher trajectory across steps than Vanilla PT (both zero- and few-shot curves).
  - Interpretation: pre-training already “speaks the language” of instructions, so fine-tuning converges quickly.
- Continual pre-training (Table 3)
  - Biomedicine (average):
    - `Llama3-8B` baseline: 53.6
    - Vanilla PT-8B: 58.4
    - Instruct PT-8B: 61.3 (near `Llama3-70B` at 63.9)
    - Notable task: PubMedQA 68.7 (Instruct PT-8B) vs 54.3 (Llama3-70B)
  - Finance (average):
    - `Llama3-8B`: 70.1
    - Vanilla PT-8B: 72.0
    - Instruct PT-8B: 74.7 (surpasses `Llama3-70B` at 71.9)
    - Largest single gain: ConvFinQA 74.6 vs 62.9 (Vanilla PT) and 59.1 (Llama3-70B)
  - Caveat: Finance NER shows variance across models; results there are less stable (Table 3 note).
- Ablations (Table 4)
  - Removing domain corpora (“w/o Corpora”) hurts domain scores (Bio: 61.3 → 58.6; Fin: 74.7 → 73.3).
  - Replacing the synthesizer with rule-based construction reduces diversity and performance (Bio: 61.3 → 58.8; Fin: 74.7 → 73.1).
  - Using only single-turn synthesis (“1-shot”) underperforms multi-round (Bio: 61.3 → 58.5; Fin: 74.7 → 73.1), supporting the benefit of few-shot structure.

C. Quality and diversity of synthesized data
- Synthesizer quality and helpfulness
  - Response accuracy and pair quality far exceed the base model on both seen and unseen datasets, zero- and few-shot (Table 5).
  - Adding synthesized pairs to prompts improves an LM’s performance compared to no pairs or random/context-mismatched pairs (Figure 5).
- Corpus analysis
  - Human/LLM assessment of instruction-augmented corpora shows high context relevance (85–99%) and solid response accuracy (70–86%), across General/Bio/Finance (Table 6; Appendix E Table 10).
  - Tasks cover 49–51 categories (Table 6 and Appendix E Table 10) distributed across 9 scenarios such as commonsense, coreference, NLI, summarization, math, etc. (Figure 6).
  - Domain coverage and overlap between raw text and synthesized tasks are high (coverage 86.8%, overlap 84.9%; Appendix F Table 11). Synthesized task domain proportions closely match raw corpora (Appendix F Table 12).
- Contamination check
  - The synthesized pairs add minimal additional overlaps with evaluation sets; e.g., MMLU sees +2 contaminated examples out of 14,042 (Table 9).

D. Do the experiments support the claims?
- Yes, with nuance. Gains over vanilla pre-training are consistent across many general tasks (Table 1), clear data-efficiency benefits are shown (Tables 2 and 15), instruction tuning is accelerated (Figure 4), and domain adaptation results are strong—often approaching or exceeding much larger models (Table 3). Some tasks show smaller or mixed changes (e.g., PIQA at 500M, finance NER variability), but the aggregate picture is positive and supported by ablations and quality analyses.

## 6. Limitations and Trade-offs
- Accuracy of synthetic supervision
  - Estimated response accuracy in the instruction-augmented corpora is ~70–86% depending on domain (Table 6; Appendix E Table 10), so noise and hallucinations remain. The paper suggests post-verification/iterative filtering as future work (Limitations).
- Scale vs. quality trade-offs
  - The study runs at 100B tokens for from-scratch models and short continual pre-training runs (Table 14), while state-of-the-art base LMs often train on trillions of tokens. Scaling laws and optimal synthetic/raw ratios are open questions (Limitations).
- Design choices that may not generalize universally
  - Templates and multi-round synthesis are tuned to English, text-only settings; cross-lingual or multimodal generalization is not explored.
  - The optimal mixture of raw vs. augmented data is only partially explored (from-scratch: 1/5 texts augmented; Section 3.2). Other ratios might work better for different scales or domains.
- Compute and engineering cost
  - Although the synthesizer is small (7B) and open-source, synthesizing 200M pairs and running multi-round inference is still non-trivial (≈1 day per 1B tokens of raw corpora on a single A100-80GB GPU; Appendix B).
- Evaluation coverage
  - While many benchmarks are covered, certain areas (e.g., code, multilinguality, long-context tasks) are only lightly represented (Figure 6 shows ~1% code).

## 7. Implications and Future Directions
- Landscape shift
  - The work reframes pre-training as “supervised multitask learning at scale,” not merely unsupervised next-token prediction. This blurs the line between pre-training and instruction tuning and shows tangible benefits in data efficiency, downstream performance, and domain adaptation.
- Practical applications
  - Organizations with large proprietary corpora can synthesize domain-specific instructions grounded in their text and run continual pre-training to achieve “70B-like” performance with smaller models (Table 3). This is attractive for finance, healthcare, legal, scientific literature, and other high-value verticals.
- Research avenues
  - Quality control for synthetic supervision: integrate verification, weak labeling, or judge models to filter/score synthesized pairs (Limitations; Related Work).
  - Scaling laws for synthetic data: study how performance scales with the proportion/quality of instruction-augmented text; when to augment vs. leave raw; and how many synthesis rounds are optimal.
  - Task mixture optimization: automatically balance task categories and difficulty (Figure 6; Appendix F) and personalize for anticipated downstream workloads.
  - Broader modalities and languages: extend the synthesizer to multilingual and multimodal contexts; explore retrieval-augmented synthesis to improve factuality.
- Engineering guidance
  - If compute-constrained, start with an open 7B synthesizer, multi-round synthesis for few-shot structure, and mix domain-augmented data with a curated general instruction set (as in Section 3.3). Monitor gains via domain validation sets and ablate “1-shot” vs. multi-round to ensure prompting ability (Table 4).

> Key takeaway: Pre-training on instruction-augmented corpora—generated cheaply with an open 7B synthesizer and grounded in the same raw texts—produces more general, more adaptable base models and substantially boosts domain specialization, often rivaling much larger LMs (Tables 1–3; Figure 4).
