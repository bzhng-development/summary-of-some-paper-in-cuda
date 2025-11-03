# PaLM 2 Technical Report

**ArXiv:** [2305.10403](https://arxiv.org/abs/2305.10403)
**Authors:** Rohan Anil, Rohan Anil et al.
**Institutions:** Google Research

What the paper is about
- Introduces PaLM 2, Google’s successor to PaLM, a family of Transformer language models trained with a mixture of objectives (UL2-style) and a more multilingual, diverse data mix.
- Emphasizes compute efficiency: smaller models at inference time with better or comparable quality to larger predecessors through optimized scaling, data, and objectives rather than parameter count alone.
- Reports extensive evaluations across languages, reasoning, coding, translation, natural language generation (NLG), and safety, and presents methods for toxicity control at inference time.
- Distinguishes clearly between pre-trained models (S, M, L), fine-tuned variants (e.g., instruction- and code-tuned), and user-facing products which add pre/post-processing.

Key contributions
- Compute-optimal scaling verified at larger scales: best training performance achieved when model parameters and tokens grow in roughly equal proportion (a replication/extension of Chinchilla-style scaling laws).
- Training recipe changes: UL2 mixed objectives rather than strictly causal LM; longer context windows; deliberate mixture emphasizing multilingual, code, math, parallel text; deduplication and PII filtering.
- Targeted safety instrumentation: small fraction of pre-training augmented with toxicity control tokens; multilingual “canaries” injected to quantify memorization; comprehensive responsible AI analyses.
- Inference-time efficiency: PaLM 2-L is much smaller than PaLM 540B yet performs better on many tasks and is faster/cheaper at serve time.

Model, training and data
- Architecture: Transformer; UL2-style mixture of denoising and causal objectives (tuned mixture to learn different aspects of language).
- Sizes: Small (S), Medium (M), Large (L). A code-specialized Small variant (PaLM 2‑S*) continues training on code-heavy multilingual data.
- Data: Larger and higher-quality than PaLM, with substantially higher non‑English proportion and hundreds of languages; includes multilingual parallel corpora (inherently improves translation); math, code, conversational data; document-level dedup; PII removal; longer context training.
- Special tokens: toxicity control tokens (from signals akin to Perspective API) appear in a small fraction of pre-training; multilingual canaries for memorization assessment.

Scaling law study (separate from PaLM 2 training itself)
- Procedure: Trained multiple models under isoFLOPS budgets (1e19–1e22 FLOPs), used FLOPs≈6ND heuristic, cosine LR decay, and quadratic fits per isoFLOP curve.
- Finding: Optimal data tokens (D) and parameters (N) both scale linearly with compute; the 1:1 scaling yields lowest loss. Downstream metrics correlate with but do not perfectly track loss (e.g., a 16B model slightly outperformed a compute-optimal ~9.5B model on some tasks), so practical choices also weigh throughput and latency.

Evaluation highlights

Human language proficiency exams
- Real-world advanced proficiency tests (C2 CEFR or equivalent) in Chinese, Japanese, French, Spanish, Italian. Under simulated conditions, PaLM 2 passes all evaluated languages and strongly outperforms PaLM on overall and writing components.

English QA and classification (1‑shot)
- On a broad suite (TriviaQA, NQ, WebQuestions, LAMBADA, HellaSwag, StoryCloze, Winograd/WinoGrande/WSC, SQuAD v2, RACE, PIQA, ARC, OpenBookQA, SuperGLUE tasks and ANLI), PaLM 2-L improves substantially over PaLM. Average across tasks: 76.9% (PaLM 2‑L) vs 70.4% (PaLM 540B). Robustness gains are notable on ANLI and RACE.

Multilingual QA (TyDi QA)
- Gold Passage and no-context (closed-book) settings across 9 languages. Consistent gains over PaLM in both settings, with especially large improvements in lower-resource and non‑Latin script languages (e.g., Telugu, Swahili, Indonesian, Arabic, Korean). No-context average F1 rises to 40.3 (PaLM 2‑L).

Multilingual toxicity classification
- Zero-/few-shot classification on Jigsaw Multilingual and English CivilComments shows sizable AUC-ROC gains over PaLM, particularly for Russian and Turkish. English also improves.

Reasoning (few-shot, often with chain-of-thought and self-consistency)
- BIG-bench Hard (23 tasks): PaLM 2 sees large improvements over PaLM in both direct and chain-of-thought prompting; average rises from 65.2 to 78.1 with CoT. Massive gains on tasks like temporal reasoning, multi-step arithmetic, Dyck languages.
- Commonsense and strategy reasoning: New SOTA or GPT‑4‑competitive results across XCOPA (strong gains on under-represented languages), StrategyQA, CSQA, ARC‑C, DROP, WinoGrande. Instruction-tuned variants show further gains.
- Math reasoning: On MATH, GSM8K, MGSM, PaLM 2 dramatically improves over PaLM; with chain-of-thought and self-consistency it is competitive with specialized Minerva on MATH and reaches strong results on GSM8K (up to 91–92% with SC) and MGSM (up to 87% with SC), surpassing prior SOTA in MGSM without SC.

Coding
- PaLM 2‑S* (small, code‑specialized) beats the much larger PaLM‑Coder‑540B on HumanEval, MBPP, and ARCADE, both pass@1 and pass@k; on ARCADE, pass@1 roughly doubles (16.2 vs 7.9). Strong multilingual coding via BabelCode: PaLM 2‑S* improves across most languages, including large jumps for low‑resource languages (e.g., Haskell, Julia), and even exceeds Python on Java/JS/TS.

Translation
- Evaluated on WMT21 with strong metrics: human MQM (lower is better) and BLEURT. PaLM 2 improves on PaLM and outperforms Google Translate in MQM on zh→en (3.0 vs 3.1) and en→de (0.9 vs 1.0), with better BLEURT too.
- Few-shot regional variants (FRMT) show consistent improvements over PaLM and over Google Translate for Portuguese (Brazil/Portugal) and Chinese (Mainland/Taiwan).
- Misgendering harms: Into English (automated), PaLM 2 maintains strong accuracy overall and improves worst-case disaggregated performance. Out of English (human rated across 13 languages) shows mixed results: improvements in some high-resource languages (e.g., Spanish, Polish, Portuguese), regressions in others (e.g., Hindi, Telugu, Arabic). Highlights the need for language-specific evaluation and mitigation.

Natural language generation (summarization and instruction-like generation)
- One-shot on XSum (en), WikiLingua (7 languages), and XLSum (11 languages): PaLM 2‑L shows large ROUGE‑2 gains over PaLM (+59% to +101% relative improvements). Filtering for 15‑gram contamination shows minimal inflation of scores.

Memorization and privacy
- Prompted training-data extraction evaluations show PaLM 2 memorizes less verbatim than PaLM on average across S/M/L. Memorization probability grows with repeated n‑grams; PaLM 2 may memorize highly repeated sequences more readily, likely due to dedup making repeats rarer and more salient.
- Tail-language analysis: canary extraction succeeds with fewer repetitions in lower-resource languages, but training-data extraction on natural text does not show a simple correlation with language size. Single-repetition sequences in low-resource languages often have less memorization than English; heavy repetition can increase memorization.

Responsible AI analyses

Pre-training dataset analysis
- Examines distributions of identity mentions, pronouns, grammatical person, and toxicity probabilities (English). Observes Western skews (e.g., “Americans”) and gender imbalances (male mentions exceed female), plus higher toxicity probabilities in documents referencing certain identities (e.g., “white people”, “transsexual”, sexuality terms). Emphasizes limitations of automated identity and toxicity signals and calls for disaggregated analyses.

Toxic language harms and dialog uses
- Dialog prompting (following alignment-style instruction) substantially reduces toxic degeneration compared to raw language modeling. However, when sampling many responses per prompt, toxicity still emerges; specialized dialog systems (e.g., LaMDA variants) remain more robust.
- Multilingual representational bias: dialog prompting helps, but cross-language and identity-term disparities persist (worst toxicity rates in English, German, Portuguese), with identity-group-specific variation (e.g., higher for “Black/White” in some languages, “Judaism/Islam” among religions). Shows need for application-specific safeguards and multilingual evaluation.

Toxicity classification
- PaLM 2 provides clear AUC-ROC improvements over PaLM across English and multilingual benchmarks, both 0‑shot and few-shot.

Translation misgendering
- Into English: stable, high accuracy; improved worst-case disaggregated metrics.
- Out of English: mixed; shows higher potential for harm in zero-shot translation for some languages; human evaluation used due to language-specific gender marking.

Generative QA harms (BBQ adapted to generation)
- With disambiguated contexts, PaLM 2 answers correctly 91.4% of the time, but about 3% of all disambiguated examples reinforce a social bias; in ambiguous contexts, the model often answers despite insufficient information and is more likely to produce biased answers. Qualitative analysis reveals rare but concerning cases where new biases are introduced unrelated to the prompt, underscoring risks from hallucination.

Inference-time control with toxicity control tokens
- Adding control tokens learned during pre-training modulates toxicity in open-ended generation: can reduce or increase toxicity probability without harming other capabilities.
- In conversational modeling and dialog uses, dialog prompting alone reduces toxic responses more than control tokens; combining both provides small additional gains in non-adversarial settings. Specialized downstream mitigations still outperform general-purpose controls.

Instruction tuning
- Flan-style instruction tuning further improves performance on BIG-bench Hard, MMLU, TyDi QA, and MGSM over untuned PaLM 2.

Developer guidance and limitations
- Models in this paper are pre-trained; production systems add further safety layers. Results may differ in products.
- Recommended: perform application-specific harm analysis, use dialog-prompting and other mitigations, and consider new measurement methods tailored to the use case. Toxicity control tokens are useful but not sufficient; specialized downstream mitigations and safety tooling are crucial.
- Measurement rubrics provided to assess Responsible AI evaluation quality (relevance, validity, inclusivity, reliability, actionability).

Takeaways and conclusions
- PaLM 2 achieves broad, state-of-the-art or near-SOTA performance with significantly smaller inference-time footprint than PaLM, through better scaling, data, and objectives.
- Strong multilingual, reasoning, coding, translation, and NLG capabilities, with notable gains in robustness (e.g., ANLI, BIG-bench Hard) and math reasoning with chain-of-thought and self-consistency.
- Multilingual translation and coding show substantial advances; region-aware translation quality approaches or surpasses production baselines in selected directions.
- Average verbatim memorization declines vs PaLM; tail-language risks require careful monitoring under high repetition.
- Responsible AI improvements include inference-time toxicity control and comprehensive evaluation, but risks remain, especially in multilingual contexts and generative QA with ambiguous prompts.
- Strategic insight: with a fixed budget, training a smaller model on more high-quality tokens (and with the right objective mix) can be more effective than simply scaling parameters, yielding both better quality and faster inference.