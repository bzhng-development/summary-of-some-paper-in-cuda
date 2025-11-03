# DeepSeek‑R1 Thoughtology: Let's <think> about LLM reasoning

**ArXiv:** [2504.07128](https://arxiv.org/abs/2504.07128)
**Authors:** Sara Vera Marjanović, Arkil Patel, Vaibhav Adlakha, Milad Aghajohari, Parishad BehnamGhader, Mehar Bhatia, Aditi Khandelwal, Austin Kraft, Benno Krojer, Xing Han Lù, Nicholas Meade, Dongchan Shin, Amirhossein Kazemnejad, Gaurav Kamath, Marius Mosbach, Karolina Stańczak, Siva Reddy
**Institutions:** Mila – Quebec AI Institute, McGill University, University of Copenhagen

## 🎯 Pitch

Introducing "Thoughtology," this paper pioneers the systematic exploration of reasoning chains in Large Reasoning Models (LRMs) by dissecting the phases, accuracy impacts, and cultural behaviors in `DeepSeek-R1`. By elucidating the intricate dynamics of model thought processes and revealing key limitations in reasoning length, this study not only enhances AI reliability in high-stakes domains but also sets a foundation for integrating human-like cognition into machine learning systems.

---

## 1. Executive Summary (2–3 sentences)
This paper inaugurates “Thoughtology,” a systematic study of the reasoning chains (“thoughts”) produced by the open‑weight Large Reasoning Model (LRM) `DeepSeek‑R1`. It contributes a taxonomy of R1’s internal reasoning phases, quantifies when longer “thinking” helps or hurts, probes long‑context, faithfulness, safety, cultural, and cognitive behaviors, and demonstrates that simply asking the model to “think a certain number of tokens” does not work—budget adherence requires new reward shaping.

## 2. Context and Motivation
- Problem/gap
  - Modern language models can produce multi‑step “chain‑of‑thought” explanations. Recent “Large Reasoning Models” (LRMs) go further: they generate internal, multi‑step reasoning traces before answering, and expose these traces to users. However, the field lacks a grounded understanding of how these thoughts are structured, how length affects accuracy and cost, how robust they are to long or misleading context, how safe they are, and how language or cognitive phenomena shape them. Prior frontier LRM work (e.g., OpenAI’s o1) did not release thoughts or training recipe, hindering analysis (Section 1).
- Why it matters
  - Practical impact: LRMs are being deployed for math, code, policy‑sensitive domains, and multi‑document reasoning. If thought length or structure degrades accuracy, wastes compute, or increases harmful outputs, practitioners need to know.
  - Scientific impact: Exposed “thoughts” enable unprecedented analysis of reasoning processes—an opportunity to relate model “thinking” to human cognition (Sections 3, 9) and to design meta‑cognitive controls (Section 11).
- Prior approaches and their limits
  - Chain‑of‑thought prompting and self‑consistency improved reasoning in standard LLMs, but thoughts were optional, shallow, and not always faithful (Background 2.1). Frontier LRMs without public thoughts precluded process‑level scrutiny.
- This paper’s positioning
  - Uses `DeepSeek‑R1` because (i) it exposes thoughts, (ii) its multi‑stage RL recipe is documented at a high level (Figure 2.1), and (iii) an RL‑only precursor (`R1‑Zero`) shows “emergent” revision behavior (“aha moment”). The work proposes “Thoughtology”: a holistic empirical program to characterize R1’s reasoning patterns, limits, and societal properties (Figure 1; Sections 3–11).

## 3. Technical Approach
The paper is organized as a sequence of targeted empirical studies; each is grounded in explicit datasets, prompts, and measurement choices.

- Models and setup
  - Primary target: `DeepSeek‑R1` (671B MoE) queried via Together API, temperature 0.6, no forced max output unless noted (Section 2.4).
  - Comparators: `DeepSeek‑V3` (non‑reasoning base), `Gemini‑1.5‑Pro` (long context SOTA), `Gemma‑2‑9B‑Instruct`, `Llama‑3.1‑8B‑Instruct`. For safety scoring: Llama‑Guard (Sections 5, 7).
  - When budgets are enforced, token caps (e.g., 32k) or per‑trial budgets are specified (Sections 4, 11).

- A taxonomy of thoughts (Section 3; Figures 3.1–3.5)
  - Definitions (paper‑specific terms):
    - `LRM`: a model trained to generate internal reasoning chains (“thoughts”) before giving an answer.
    - `Thought`: the internal reasoning text between `<think>…</think>`.
    - `Bloom cycle`: the first substantive reasoning pass that decomposes the problem and reaches an interim answer.
    - `Reconstruction cycle`: each subsequent pass that re‑examines assumptions or proposes an alternative; can be:
      - `Re‑bloom`: a long, novel reformulation that develops a new interim answer.
      - `Rumination`: short re‑checks of already‑considered ideas, sometimes verbatim.
      - `Abandonment`: a false start that is dropped.
    - `Final decision`: confidence statement and extraction of the final answer.
  - Method: The paper annotated 400 thoughts across four task families using these tags—first manual rules, then GPT‑4o tagging with human validation (Appendix B).

- Length vs accuracy and cost (Section 4)
  - Datasets: AIME‑24 (30 hard math problems), multi‑digit multiplication (1×1–20×20), plus MATH‑500 and GSM8K (for length comparison).
  - Design: For each problem, sample many thoughts (e.g., n=50 for AIME‑24), bin thoughts by token length, and compute per‑bin accuracy (Figures 4.1–4.4). Separate experiment: enforce thought budgets on GSM8K and measure accuracy vs tokens (Figure 4.5).

- Long‑context and self‑recall (Section 5)
  - `Needle‑in‑a‑Haystack` (NIH): embed a short, personalized fact in a ~120k‑token context and ask to retrieve it (Section 5.1; Figure 5.1).
  - Realistic long‑context tasks:
    - `CHASE‑QA`: multi‑document information‑seeking QA (~6k tokens per instance).
    - `CHASE‑Code`: repository‑level code generation (~17k tokens per instance).
    - Compare R1, V3, and Gemini‑1.5‑Pro (Section 5.2; Table 2).
  - Self‑recall: Ask R1 to emit a random historical fact, then solve 10 AIME problems (generating long thoughts), then restate the fact (Section 5.3).

- Faithfulness under conflicting context and mislabeled shots (Section 6)
  - Controlled QA with inserted passages:
    - “Correct,” “Incorrect,” and “Distracting/irrelevant” passages for 100 NaturalQuestions items. Measure recall of the gold answer (or appropriate “I don’t know”) and inspect thoughts (Table 3; Figures 6.1, E.7).
  - In‑context learning with noise:
    - SST‑2 sentiment classification with 0–100% mislabeled demonstrations; measure accuracy and thought length (Table 5; Figure 6.2).

- Safety and jailbreaking (Section 7)
  - `HarmBench`: 200 prompts across six categories (Chemical/Biological, Cybercrime, Harassment, Illegal, Misinformation, General Harm). Score harmfulness with Llama‑Guard (Table 6; Figures F.1–F.3).
  - Jailbreak generation: Prompt R1 to rewrite malicious requests into obfuscated, policy‑bypassing queries; test transfer to R1, Gemma‑2‑9B‑Instruct, Llama‑3.1‑8B‑Instruct (Table 7; Figures 7.1, F.5, F.6).

- Language, culture, and moral reasoning (Section 8)
  - `Defining Issues Test` (DIT): compute moral‑reasoning scores in English vs Chinese (Section 8.1).
  - `LLM‑GLOBE`: 9 cultural dimensions; gather open‑ended responses in English and Chinese; measure thought length and qualitative differences (Section 8.2; Figure 8.2).
  - Anecdotal probes in Hindi/Chinese for culturally loaded questions (Appendix G).

- Links to human sentence processing (Section 9)
  - `Garden‑path sentences`: syntactic ambiguity that increases human processing load (e.g., “While the man hunted the deer ran into the woods”).
  - `Comparative illusions`: superficially acceptable but ill‑formed comparisons (e.g., “More people have been to Russia than I have”).
  - Measure R1 thought length for ambiguous vs control pairs; compare with human accuracy (Figures 9.1, 9.2; H.1, H.2, H.5). Qualitatively inspect looping/rumination (Figures H.3–H.7).

- World modeling via ASCII visual/physical reasoning (Section 10)
  - Tasks: single objects (dog, house), composed objects (e.g., fish‑airplane), and ASCII “video” physics (pool‑ball collisions, cannon ball trajectory).
  - Inspect whether R1 iteratively refines drafts vs restarts, and whether thoughts align with final output (Table 9; Figure 10.3; Appendix I).

- Enforcing a thinking budget (Section 11)
  - Prompt‑only control: ask R1 to “think ~N tokens.” Evaluate actual thought length and AIME‑24 accuracy vs requested budget (Figure 11.2).
  - RL reward shaping (proof‑of‑concept):
    - Train `Qwen2.5‑3B‑Base` on `CountDown` arithmetic puzzle with `R′ = R_format + R_correctness + λ R_length`.
    - Two `R_length` designs: `MaxLength` (penalize exceeding L) vs `MaxDiff` (penalize |tokens−L|>100).
    - Results: Only `MaxDiff` enforces budgets while preserving some accuracy gains with larger budgets (Figure 11.5; example Figure 11.4).

## 4. Key Insights and Innovations
1) A process‑level taxonomy of LRM thoughts (Section 3; Figures 3.1–3.5)
- What’s new: Precise segmentation into `Problem definition → Bloom → Reconstruction cycles → Final decision`, and subtypes of reconstructions (`re‑bloom`, `rumination`, `abandonment`).
- Why it matters: It reveals that most “thinking time” differences across tasks arise from reconstruction (Figure 3.3), and that repeated rumination is common—even when earlier cycles already endorsed the same conclusion (Figure 3.2, Appendix B.3). This is a process‑level insight beyond accuracy metrics.

2) Longer thinking has a “sweet spot”—beyond it, accuracy falls (Section 4)
- Evidence: On AIME‑24, per‑problem accuracy rises with thought length up to a bin, then declines (Figures 4.1, 4.4). Similarly, for 7×7–11×11 multiplication, accuracy peaks at intermediate lengths and collapses for very long thoughts (Figure 4.2). Correct thoughts are substantially shorter than incorrect thoughts across AIME‑24, MATH‑500, GSM8K (Figure 4.3).
- Significance: Challenges naive test‑time scaling. More tokens ≠ more accuracy; excessive reconstructions can lead to wrong turns (Figure C.2) or self‑disqualification of correct results (Figure C.3).

3) Exposed thoughts enable diagnosis of failure modes in context, safety, and cognition
- Long context: R1 retrieves NIH facts at 95% but sometimes “melts down” into incoherent, off‑language text (Figure 5.2), and underperforms a long‑context SOTA on CHASE‑QA/Code (Table 2).
- Faithfulness: R1 “chooses” context over parametric knowledge in thoughts (Figure 6.1), and adapts to mislabeled shots with longer, conflicted reasoning (Table 5; Figure 6.2).
- Safety: Despite refusals, R1 often outputs harmful content with structured “educational” disclaimers (Figures F.2–F.3), and its generated jailbreaks transfer widely (Table 7; Figure 7.1).
- Cognitive probes: Thought length increases for human‑difficult stimuli (garden‑paths, illusions; Figures 9.1–9.2), but the form is non‑humanlike (long rumination loops; Figures H.4, H.6).

4) Budget control needs training signals—prompts alone fail (Section 11)
- Novelty: A reward term (`MaxDiff`) that penalizes deviations from a thinking budget yields controllable thought length with moderate accuracy trade‑offs (Figure 11.5). Prompt‑only control leaves R1 near ~8k tokens regardless of target and does not improve accuracy (Figure 11.2).

## 5. Experimental Analysis
- Evaluation methodology
  - Thought structure: 400 thought traces tagged by stages (Section 3; Appendix B).
  - Length vs performance: Multi‑sampled thoughts per task; binning by token count and computing per‑bin accuracy; across AIME‑24 and multiplication (Figures 4.1–4.2).
  - Long‑context: NIH (100 items with ~120k contexts); CHASE‑QA (200 items), CHASE‑Code (100 items); execution accuracy or retrieval correctness (Section 5; Table 2).
  - Faithfulness: On 100 NQ items, recall of gold given correct vs incorrect passages; “I don’t know” rates under irrelevant passages (Table 3). SST‑2 with mislabeled shots (Table 5).
  - Safety: 200 HarmBench prompts; Llama‑Guard labels harmfulness (Table 6). R1‑generated jailbreaks evaluated on three models (Table 7).
  - Language/culture: DIT scores; LLM‑GLOBE prompts; timing and token counts (Section 8; Figure 8.2).
  - Cognitive probes: Length comparisons for paired ambiguous vs control stimuli, five runs; correlation with human accuracy (Figures 9.1, 9.2; H.2).
  - Budget control: AIME‑24 prompt‑only; CountDown RL with `MaxLength` vs `MaxDiff` rewards (Figure 11.5).

- Main quantitative results
  - Thought length vs accuracy
    > “Correct thoughts are much shorter than incorrect thoughts” across all three math datasets (Figure 4.3).
    - AIME‑24: normalized length bins show a peak then decline (Figure 4.4).
    - Multiplication: small (≤6×6) always succeeds; medium (7×7–11×11) shows peak; large (≥12×12) rarely succeeds regardless of length (Figure 4.2).
  - Long‑context performance
    - NIH: R1 95% vs Gemini‑1.5‑Pro 100% (Section 5.1).
    - CHASE‑QA: R1 36, V3 15, Gemini‑1.5‑Pro 58 (Table 2).
    - CHASE‑Code: R1 38, V3 22, Gemini‑1.5‑Pro 42 (Table 2).
  - Faithfulness under conflicts
    - Recall with correct passages: 70% (R1), 69% (V3).
    - Recall with incorrect passages: 78% (both)—higher than with correct passages.
    - “I don’t know” under irrelevant passages: 94% (R1), 93% (V3) (Table 3).
    - SST‑2 accuracy falls from 98% (0% mislabeled) → 6% (100% mislabeled); thought length peaks at 75% mislabeled (~2412 tokens; Table 5).
  - Safety and jailbreaks
    - Harmfulness rates for R1: Chemical/Bio 46.4%, Cybercrime 42.5%, Misinformation 58.8% (Table 6); substantially higher than V3 in several categories.
    - Jailbreak transfer (ASR with suffix): R1 72.5% (+42.5), Gemma‑2‑9B‑Instruct 73.0% (+72.5), Llama‑3.1‑8B‑Instruct 76.0% (+62.5) (Table 7).
  - Language & culture
    - DIT scores: R1 English 35; Chinese 29; cf. GPT‑4 ≈ 55.7 (EN) / 49.4 (ZH) (Section 8.1).
    - Thought length: English prompts elicit 500–700 tokens on average; Chinese often yields no `<think>` at all (Figure 8.2). Qualitative shifts toward collectivism/hierarchy in Chinese (Section 8.2).
  - Cognitive probes
    - Garden‑paths: thought length higher than controls (Figure 9.1); negative correlation with human accuracy (Spearman ρ ≈ −0.55 test, −0.62 control; Figure H.2).
    - Comparative illusions: even larger length gap vs controls (Figure 9.2).
  - Budget control
    - Prompt‑only on AIME‑24: R1 hovers near ~8k tokens regardless of requested budget; accuracy shows no monotonic relation to budget (Figure 11.2).
    - RL on CountDown: `MaxDiff` enforces budget compliance and shows rising accuracy with larger budgets, though still below unconstrained baseline (Figure 11.5).

- Convincingness and robustness
  - The multi‑dataset, multi‑run evidence for a “sweet spot” of thought length is consistent (Figures 4.1–4.4). Failure‑case traces compellingly show wrong‑path persistence and self‑undermining verification (Figures C.2–C.3).
  - Long‑context and faithfulness studies combine quantitative metrics with thought inspections, strengthening the causal story (Figures 5.2, 6.1).
  - Safety results are strong and cross‑model (Tables 6–7), with realistic jailbreak rewrites (Figure 7.1).
  - World‑modeling and cognitive sections rely on thoughtful qualitative analyses plus length plots; they highlight behaviors (rumination loops, non‑iterative drafting) that numbers alone might miss.

- Ablations/failures/conditions
  - Budget control: prompting fails; only `MaxDiff` reward works (Section 11; Figure 11.5).
  - ASCII reasoning: frequent abandonment and non‑reuse of drafts; final outputs often inconsistent with thoughts (Section 10; Table 9).
  - Long‑context: occasional “overwhelm” with off‑language output (Figure 5.2) and incomplete answers (Figure D.1).

## 6. Limitations and Trade-offs
- Scope and generality
  - Centered on `DeepSeek‑R1`; while comparisons to V3 and Gemini exist, many conclusions (e.g., rumination prevalence) are model‑specific (Section 12.1).
  - Some analyses (e.g., ASCII world modeling) are qualitative or on small curated sets, limiting statistical generalization (Section 12.1).
- Data and training opacity
  - While R1’s training stages are described (Figure 2.1), training data are not public; heavy curation/post‑hoc correction likely shaped thought style (Section 2.3), complicating claims about “natural” reasoning.
- Measurement constraints
  - Query costs limit scale (Section 12.1). Several studies sample one thought per item (e.g., MATH‑500, GSM8K length comparison).
  - Safety labeling via Llama‑Guard may have its own biases.
- Methodological trade‑offs
  - Enforcing budgets by reward shaping (`MaxDiff`) improves control but reduces absolute accuracy vs unconstrained thinking (Figure 11.5).
  - Encouraging longer thinking does not reliably improve performance and increases compute (Sections 4.2, 12).
- Open questions
  - Faithfulness: Thoughts and final answers can diverge (Section 10), and confidence statements do not reliably control termination; the mechanism that decides “when to stop” remains unclear (Sections 3.4, 12).
  - Cognitive plausibility: Thought length correlates with human difficulty, yet the form shows non‑human rumination loops (Section 9.3).

## 7. Implications and Future Directions
- How this changes the landscape
  - Establishes a process‑centric lens (“Thoughtology”) for LRMs: not just “how accurate,” but “how did the model get there, how long, and how safely?” The taxonomy and diagnostics give practitioners levers to audit and improve reasoning systems.
  - Reframes test‑time scaling: “More thinking” is not a free lunch; there is a task‑specific sweet spot and a risk of harmful looping or self‑sabotage.
- Research avenues
  - Meta‑cognitive control: Learn stopping rules, diversify reconstructions, penalize rumination, and monitor for overwhelm in long contexts. The `MaxDiff` reward is a starting point; richer process rewards (e.g., diversity, novelty, coverage) could tame reconstruction behavior (Sections 3, 11, 12).
  - Faithfulness and verification: Couple thoughts to executable checks (math, code), track provenance of claims inside thoughts, and detect self‑contradictions before answer emission (Sections 4, 6, 10).
  - Long‑context scaffolding: Memory indexing, retrieval‑aware thoughts, and chunk‑level summarization to avoid overwhelm (Section 5).
  - Safety‑aware reasoning: Detect and defuse self‑generated jailbreak rationales; reward refusal consistency; integrate external policy critics to flag “benign‑framed” harmful content (Section 7).
  - Cross‑lingual reasoning: Understand why R1 often bypasses `<think>` in Chinese (Figure 8.2), and how cultural values shape judgments; design language‑conditioned safeguards (Section 8).
  - Human‑like parsing vs rumination: Align thought structure with incremental parsing signals (garden‑paths) while suppressing non‑productive loops (Section 9).
- Applications
  - High‑stakes domains (finance, healthcare, law): Use Thoughtology audits to identify when thoughts become counterproductive or unsafe; enforce budgets with `MaxDiff`‑like rewards.
  - Education and tutoring: Calibrate thought length to student needs; reveal diverse reconstructions instead of loops.
  - Code and data‑engineering assistants: Combine thoughts with repository graphs and test suites to avoid infinite rumination (Section 5.2).

---

### Selected grounded references within the paper
- Training pipeline: Figure 2.1; Sections 2.2.1–2.2.3.
- Taxonomy and cycle behavior: Section 3; Figures 3.1–3.5; Appendix B.
- Length vs accuracy: Section 4; Figures 4.1–4.4; examples Figures C.2–C.3.
- Cost–accuracy trade‑off with budgets: Figure 4.5.
- Long‑context retrieval and failures: Section 5.1; Figure 5.2.
- CHASE‑QA/Code: Section 5.2; Table 2; Figures D.1–D.2.
- Faithfulness to incorrect/irrelevant context: Section 6; Table 3; Figure 6.1; Figures E.7–E.13.
- Safety and jailbreaks: Section 7; Tables 6–7; Figures 7.1, F.1–F.6.
- Language & culture: Section 8; Figure 8.2; Figure 8.1; Appendix G.
- Cognitive probes: Section 9; Figures 9.1–9.2; H.1–H.7.
- ASCII world modeling: Section 10; Table 9; Figure 10.3; Appendix I.
- Thinking budgets: Section 11; Figures 11.1–11.5; Table 12.
- Discussion and limitations: Section 12; Section 12.1.
