# PaLM 2 Technical Report

**ArXiv:** [2305.10403](https://arxiv.org/abs/2305.10403)

## 🎯 Pitch

PaLM 2 introduces a new state-of-the-art family of Transformer-based language models that achieve dramatically better performance in multilingual tasks, reasoning, coding, and translation—while being far more compute-efficient than its predecessor, PaLM. By validating and applying compute-optimal scaling laws at unprecedented scales, employing a diverse data mixture and multiple training objectives, and integrating practical safety and memorization controls, PaLM 2 delivers superior quality and safer, faster deployment in real-world applications. This leap not only cuts inference costs and broadens accessibility but also sets a new benchmark for scientifically grounded, responsible, and inclusive generative AI.

---

## 1. Executive Summary (2–3 sentences)
PaLM 2 is a family of Transformer language models trained with a carefully engineered data mixture and a mixture of training objectives, designed to be compute‑efficient while markedly improving multilingual, reasoning, coding, and translation performance. By validating compute‑optimal scaling at large scales and adding practical safety tooling (toxicity control tokens) and memorization measurement (multilingual canaries), PaLM 2 delivers higher quality than its predecessor PaLM despite being smaller at inference time, enabling faster and broader deployment (Abstract; Sections 1–3, 5).

## 2. Context and Motivation
- Problem addressed:
  - Recent large language models (LLMs) reached strong performance by scaling parameters, but were often compute‑suboptimal, largely monolingual, trained with a single objective, and lacked robust safety/controllability and memorization analysis. This limited multilingual capabilities, reasoning strength, and safe deployment (Introduction; Section 3; Section 5).
- Why it matters:
  - Practical significance: smaller models with better quality and faster inference reduce serving cost and enable new products (Introduction).
  - Scientific significance: validating scaling laws at new regimes clarifies how to allocate compute between data and parameters (Section 2).
- Prior approaches and gaps:
  - Earlier scaling laws (Kaplan et al., 2020) favored growing parameters faster than data; later “Chinchilla” (Hoffmann et al., 2022) suggested data and parameters should grow roughly 1:1. Many LLMs still trained compute‑suboptimally.
  - Most training mixtures were English‑heavy and used a single causal LM objective; safety controls were added post‑hoc rather than built into pre‑training (Sections 1, 3, 5).
- How this work positions itself:
  - Independently verifies compute‑optimal scaling at much larger compute budgets (10^19–10^22 FLOPs) and derives optimal model/data ratios (Section 2, Figures 4–5, Table 1).
  - Introduces a more multilingual, deduplicated dataset with parallel data, code, math, and conversation; and a UL2‑style mixture of objectives for better generalization (Section 3).
  - Adds inference‑time steerability through toxicity control tokens and evaluates memorization with multilingual “canaries” (Sections 4.7, 5.1).

## 3. Technical Approach
Step‑by‑step view of what PaLM 2 is and how it was built and evaluated.

- Model family and training paradigm
  - Architecture: Transformer; significantly longer context window than PaLM to support long dialogue, long‑range reading, and summarization (Section 3).
  - Objective: UL2‑style “mixture of objectives”—instead of only predicting the next token (causal LM), training alternates among different denoising/LM sub‑tasks. This teaches the model complementary skills such as filling in missing text and reading with bidirectional context (Section 1).
  - Sizes and efficiency: Largest PaLM 2 (`PaLM 2‑L`) is significantly smaller than PaLM 540B yet trained with more compute and a larger, higher‑quality dataset, emphasizing that better data/objectives can beat brute‑force parameter scaling for overall quality and inference speed (Introduction).

- Training data mixture and processing (Section 3; Appendix D.1)
  - Sources: Web documents, books, code, mathematics, conversational data.
  - Multilingual emphasis: Substantially higher share of non‑English data; includes parallel (bitext) pairs covering “hundreds of languages,” improving translation and cross‑lingual reasoning.
  - Quality controls: Deduplication (to reduce memorization), PII removal, and filtering for quality. Table 21 lists the top 50 languages in the “multilingual web documents” sub‑corpus (e.g., Spanish 11.5%, Chinese 10.2%, Russian 8.7%, Japanese 7.6%).
  - Special tokens: A small fraction of pre‑training data is tagged for toxicity levels using a fixed Perspective API signal, enabling inference‑time control; multilingual “canary” sequences injected to quantify memorization (Sections 3, 4.7, 5.1).

- Compute‑optimal scaling experiments (Section 2)
  - Key terms:
    - `FLOPs` = floating‑point operations; a proxy for training compute.
    - `IsoFLOP curves` = for a fixed compute budget, train various (parameters, tokens) pairs and fit a curve of validation loss across model sizes at that fixed budget; the minimum identifies the best parameter count for that compute.
  - Procedure: Train many models at four compute scales (10^19, 10^20, 10^21, 10^22 FLOPs). Use the heuristic FLOPs ≈ 6ND (N = parameters, D = training tokens) to allocate training tokens per model. Fit quadratic curves per compute band (Figure 4); extract optimal N and D (Figure 5; Table 1).
  - Finding: Optimal scaling follows ~1:1 growth in tokens and parameters, corroborating “Chinchilla” at larger scales (Figure 5; Table 1).

- Instruction tuning for some evaluations (Appendix A.2)
  - After pre‑training, some evaluations use `Flan` instruction tuning, a large mixture of 1,800+ tasks and prompts. This improves instruction following and reasoning (Table 16). When used, the paper marks it (e.g., Table 5).

- Safety and memorization mechanisms integrated into training and eval
  - `Control tokens` for toxicity: Pre‑training exposes the model to text tagged with low/med/high toxicity. At inference, prompting with the control token steers generation (Section 5.1; Table 14; Figure 10–11).
  - `Canaries` for memorization: Synthetic sequences created by shuffling/interleaving real documents in multiple languages and injecting them during training at controlled repetition counts. During eval, feed the prefix and check if the model reproduces the suffix verbatim to estimate memorization risk (Section 4.7; Table 13; Figure 9). Verbatim extraction is also measured on real training snippets (Figure 8).

- Task‑specific variants
  - `PaLM 2‑S*` for code: Continue training `PaLM 2‑S` on a code‑heavy, multilingual mix to boost coding tasks while keeping natural‑language quality (Section 4.4; Table 8; Figure 6).

## 4. Key Insights and Innovations
1) Validating compute‑optimal scaling at unprecedented scale (Section 2)
- What’s new: Independent derivation of scaling laws across 10^19–10^22 FLOPs shows that training tokens should grow roughly in proportion to model size (Figures 4–5; Table 1).
- Why it matters: Confirms that quality and efficiency come from allocating compute to both parameters and data, not just bigger models. This underwrites PaLM 2’s “smaller but better” strategy.

2) Data and objective curation beats naive parameter scaling (Sections 1, 3; multiple eval sections)
- What’s new: A more multilingual, deduplicated dataset plus a UL2‑style objective mix avoids English regression while boosting multilingual tasks; long context length enables long‑form tasks. PaLM 2‑L, though smaller than PaLM 540B, outperforms it across many benchmarks (e.g., Table 2, Table 3, Table 5, Table 9, Table 11).
- Why it matters: Shifts emphasis from sheer size to “data and objective design,” improving quality and inference latency together.

3) Built‑in, low‑overhead safety control via control tokens (Section 5.1)
- What’s new: Tag a small fraction of training data with toxicity levels; expose `control tokens` so users can steer output toxicity at inference time without extra fine‑tuning or classifiers in the loop (Table 14; Figure 10–11).
- Why it matters: Practical, general‑purpose steerability mechanism with minimal training overhead and no degradation on unrelated tasks (Section 5.1).

4) Multilingual memorization analysis with canaries (Section 4.7)
- What’s new: Inject language‑specific outlier sequences (“interleave” and “shuffle” canaries) with controlled repetitions to quantify how often the model memorizes them; also compare to real‑data extraction (Table 13; Figures 8–9).
- Why it matters: Nuanced understanding of privacy risk across languages: PaLM 2 memorizes less on average than PaLM, but repeated n‑grams and tail‑language repetitions increase risk (Figures 8–9).

5) A small, code‑specialized model that surpasses a much larger coder LM on several tasks (Section 4.4)
- What’s new: `PaLM 2‑S*` (small) beats PaLM‑Coder‑540B on HumanEval@1, MBPP@1, and ARCADE@1 by large margins (Table 8) and across 12 programming languages (Figure 6; Table 18).
- Why it matters: Demonstrates that domain‑focused continued pre‑training on a strong base can replace extreme parameter counts for developer‑facing code assistants.

## 5. Experimental Analysis
How PaLM 2 is evaluated and what the numbers show.

- General English QA and classification (Section 4.2; Table 2)
  - Setup: 1‑shot in‑context across 24+ datasets (open‑domain QA, cloze, Winograd, reading comprehension, commonsense, SuperGLUE, ANLI).
  - Headline: Average score improves from 70.4 (PaLM 540B) to 76.9 (PaLM 2‑L).
  - Notable gains:
    - ANLI: R1 52.6→73.1, R2 48.7→63.4, R3 52.3→67.1.
    - RACE‑H 52.1→62.3, RACE‑M 69.3→77.0.
    - ReCoRD 92.8→93.8; SQuAD v2 EM 78.7→80.5.
  - Interpretation: Broad gains in robust reasoning and reading comprehension.

- Multilingual QA (TyDi QA; Section 4.2; Table 3)
  - `Gold Passage` F1 average: 69.8 (PaLM) → 73.6 (PaLM 2‑L).
  - `No‑context` (closed‑book) F1 average: 31.5 → 40.3; largest gains in low‑resource/Non‑Latin languages (e.g., Swahili 39.7→50.3; Indonesian 35.5→46.4; Korean 35.0→46.9).

- Reasoning and BIG‑Bench Hard (Section 4.3; Tables 5–6)
  - Few‑shot with CoT/SC where marked; some results use the instruction‑tuned variant (Appendix A.2).
  - Selected results (Table 5):
    - `StrategyQA`: 81.6 (SOTA reported) vs 90.4 (`PaLM 2`).
    - `CSQA`: 91.2 SOTA vs 90.4 PaLM 2 (near‑SOTA).
    - `BB Hard` (23 tasks): PaLM 65.2 (CoT) → PaLM 2 78.1 (CoT).
  - Task‑level jumps in BB Hard (Table 6):
    > `temporal_sequences`: 39.6/78.8 (PaLM direct/CoT) → 96.4/100.0 (PaLM 2).  
    > `multistep_arithmetic_two`: 1.6/19.6 → 0.8/75.6.  
    > `dyck_languages`: 28.4/28.0 → 35.2/63.6.  
    > `logical_deduction`: 42.7/56.9 → 64.5/69.1.
  - Takeaway: Chain‑of‑thought (CoT) amplifies PaLM 2’s gains on multi‑step reasoning.

- Mathematical reasoning (Section 4.3; Table 7)
  - `MATH`: 48.8 (PaLM 2 with SC) vs 50.3 (Minerva SOTA); far above PaLM 8.8.
  - `GSM8K`: 91.0 (PaLM 2 with SC) vs 92.0 (GPT‑4 reported) and PaLM 74.4.
  - `MGSM` (multilingual GSM8K): 87.0 (PaLM 2 with SC), exceeding prior SOTA 72.0.
  - Implication: Strong quantitative reasoning in both English and multiple languages.

- Coding (Section 4.4; Table 8; Figure 6; Table 18)
  - `PaLM 2‑S*` vs PaLM‑Coder‑540B:  
    > HumanEval pass@1: 37.6 vs 35.9; pass@100: both 88.4.  
    > MBPP pass@1: 50.0 vs 47.0; pass@80: 86.6 vs 80.8.  
    > ARCADE pass@1: 16.2 vs 7.9; pass@30: 43.6 vs 33.6.
  - Multilingual HumanEval (BabelCode): higher pass@1 than PaLM(-Coder) on 10/12 languages; extreme gains on low‑resource languages (e.g., Haskell 8.7% vs 1.86%; Julia 16.8% vs 4.35%; Figure 6; Table 18).
  - Message: A small, code‑tuned PaLM 2 achieves or beats the much larger coder baseline, especially on notebook completion (ARCADE).

- Translation (Section 4.5; Tables 9–10)
  - WMT21 with human `MQM` (lower better):  
    > Chinese→English: PaLM 3.7, Google Translate 3.1, PaLM 2 3.0.  
    > English→German: 1.2, 1.0, 0.9.  
    > BLEURT also improves (Table 9).
  - Dialect‑aware FRMT (Few‑shot): PaLM 2 beats both PaLM and Google Translate across Brazilian/European Portuguese and Mainland/Taiwanese Chinese (Table 10).

- Natural Language Generation (NLG) (Section 4.6; Tables 11–12; Appendix A.5)
  - One‑shot summarization/headline generation across English and 10+ non‑English languages:
    > `XSum` (en ROUGE‑2): 14.5 → 23.2.  
    > `WikiLingua` (ar/ja/ko/ru/th/tr): 11.7 → 23.5.  
    > `XLSum` (11 languages): 12.7 → 21.3 average with PaLM 2‑L (Table 11).
  - Data contamination check: Filtering based on 15‑gram overlap changes scores minimally and positively (+0.3 to +0.6), arguing against memorization‑inflated metrics (Table 12).

- Responsible‑AI measurements (Sections 4.2, 4.6, 5; Appendix D)
  - Toxicity classification AUC‑ROC improves in English and multilingual (Jigsaw/Civil Comments; Table 4).
  - Open‑ended toxic degeneration: small improvement vs PaLM on RealToxicityPrompts; conversational LM shows slight regression vs PaLM (Appendix D.7, Table 30).
  - Dialog prompting dramatically lowers toxic responses relative to language‑modeling alone (Appendix D.3, Figure 30), but bias varies by language/identity terms (Figures 31–32).
  - Misgendering in translation: Into English is stable or slightly better worst‑case vs PaLM (Table 24). Out of English (human‑rated), mixed: improvements in Spanish/Polish/Portuguese but regressions in Telugu, Hindi, Arabic (Table 26; Figure 33).

- Inference‑time toxicity control (Section 5.1)
  - With control tokens on non‑toxic prompts, probability of toxic continuation drops from 0.075 to 0.033 (low‑toxicity setting), and can be increased when desired (0.203 high setting; Table 14; Figure 10).
  - In conversational LM (single sample), control tokens reduce toxic responses on standard dataset 30%→12% and adversarial 18%→7% (Section 5.1).
  - In dialog uses, dialog‑prompting itself was even more effective than control tokens; specialized systems (LaMDA) remain best (Figure 11; Section 5.1).

- Memorization (Section 4.7; Figures 8–9; Table 13)
  - Average verbatim extraction on English shared data is lower for PaLM 2 than PaLM across model sizes (Figure 8a).
  - But as n‑grams repeat more, PaLM 2 memorizes repeated sequences more readily (Figure 8b).
  - Tail languages: canaries need fewer repetitions to be memorized; real data extraction shows no strong correlation with language size, except higher risk when sequences are highly repeated (Figure 9).

- Scaling law downstream check (Appendix A.1; Table 15)
  - At fixed compute (10^22 FLOPs), 9.5B and 16.1B models perform similarly across 26 downstream tasks (avg 57.7 vs 58.3), while 9.5B has the lowest training loss. Training loss is not a perfect surrogate for downstream task quality.

Overall assessment: The breadth of benchmarks, the human evaluation for translation, contamination checks for generation, and the safety analyses together support the core claims—PaLM 2 is markedly more capable and efficient, with new safety/measurement tooling—while also surfacing nuanced mixed results in certain safety and multilingual settings.

## 6. Limitations and Trade-offs
- Assumptions and constraints
  - Compute and infrastructure: Training at up to 10^22 FLOPs with TPUv4 and the Pathways stack is resource‑intensive (Model Card; Sections 2–3), out of reach for most labs.
  - Some architectural specifics and exact parameter counts are not publicly disclosed (Model Card), limiting reproducibility.
- Safety metrics and tools
  - Toxicity labeling relies on a fixed Perspective API signal during both pre‑training tagging and evaluation; this proxy has known limitations across languages and sociolects (Section 5.1; Appendix D.3–D.7).
  - Control tokens help in language modeling and conversational LM, but dialog‑prompting and specialized safety systems still outperform them in dialog uses (Section 5.1; Figure 11).
- Mixed or conditional results
  - Conversational LM shows a slight toxicity regression vs PaLM (Appendix D.7, Table 30).
  - Misgendering when translating out of English regresses in some languages (e.g., Telugu, Hindi, Arabic; Table 26).
  - Some reasoning results depend on chain‑of‑thought and self‑consistency sampling; direct prompting is weaker on certain tasks (Table 6).
- Data and evaluation caveats
  - Some instruction‑tuning tasks overlap with training in the Flan mixture (Appendix A.2 and notes in Section 4.3), though test/dev splits are held out; real‑world generalization is still the key question.
  - While memorization is lower on average, highly repeated sequences—especially in tail languages—remain a risk (Figures 8–9).

## 7. Implications and Future Directions
- How this work shifts the field
  - Confirms compute‑optimal scaling at large regimes and demonstrates that data/objective design can deliver step‑change performance without maximal parameter counts (Sections 2–3). This encourages focusing on data quality, multilingual breadth, and objective mixtures—not just bigger models.
  - Establishes built‑in steerability (control tokens) and multilingual memorization audits (canaries) as practical components of pre‑training pipelines (Sections 4.7, 5.1).
- Follow‑up research enabled/suggested
  - Steerability beyond toxicity: extend control tokens to attributes like helpfulness, formality, or disclosure of uncertainty; study how such conditional training interacts with instruction tuning (Section 5.1, final paragraphs).
  - Safer multilingual systems: develop evaluation sets and mitigation methods that go beyond English and better capture diverse sociocultural contexts and harms (Appendix D.2–D.6).
  - Memorization under distribution shift: explore how data repetition, dedup granularity, and objective mixtures influence memorization across languages and domains (Section 4.7).
  - Bridging training loss and task metrics: more systematic ablations to understand when compute‑optimal training loss transfers to downstream optimality (Appendix A.1, Table 15).
- Practical applications
  - Multilingual assistants and tools: stronger translation (MQM‑verified; Table 9), multilingual QA (Table 3), and MGSM math reasoning (Table 7) enable wider global deployment.
  - Coding copilots: `PaLM 2‑S*` shows that small, specialized models can surpass much larger coder LMs in practical settings such as notebooks (Table 8; Figure 6).
  - Summarization and content generation: large improvements across many languages (Table 11), with low contamination sensitivity (Table 12).
  - Safer deployments: integrated toxicity control and robust safety evaluations give developers more levers and guidance (Section 5; Appendix D).

> Bottom line: PaLM 2 demonstrates that smarter scaling (1:1 data–parameter growth), better data (multilingual, deduplicated, parallel), and richer objectives can yield a smaller, faster model that substantially advances multilingual reasoning, coding, and translation—while beginning to address safety and memorization in ways that are measurable and controllable.
