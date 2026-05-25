# Scaling Instruction-Finetuned Language Models

**ArXiv:** [2210.11416](https://arxiv.org/abs/2210.11416)

## 🎯 Pitch

This paper systematically demonstrates that instruction finetuning—training large language models on thousands of diverse, natural-language instruction tasks—scales remarkably well across both model size and the number of tasks, especially when augmented with a small amount of chain-of-thought data. By finetuning models like PaLM-540B and T5 on 1,836 such tasks, the authors achieve significant gains in zero-/few-shot generalization, state-of-the-art results on reasoning and knowledge benchmarks, and substantial improvements in practical usability and some safety metrics—all with minimal additional compute. This work matters because it shows that scalable instruction finetuning is a broadly applicable, compute-efficient strategy for making large language models more capable, safer, and easier to use by non-experts—effectively unlocking their reasoning and instruction-following potential for real-world tasks.

---

## 1. Executive Summary (2-3 sentences)
This paper shows that “instruction finetuning” — supervised training on thousands of tasks phrased as natural-language instructions — scales effectively to very large language models and to many task types, especially when a small amount of chain‑of‑thought (step‑by‑step rationale) data is included. By finetuning PaLM and T5-family models on 1,836 instruction tasks (including nine chain‑of‑thought datasets), the resulting models (“Flan-*”) achieve strong gains across reasoning and knowledge benchmarks with only a tiny fraction of pre‑training compute (e.g., 0.2% for PaLM‑540B; Table 2), set new state-of-the-art results on several evaluations (e.g., 75.2% on five‑shot MMLU with chain‑of‑thought + self‑consistency; Table 4), and improve open‑ended usability and some safety metrics (Appendix C, Figure 8).

## 2. Context and Motivation
- Problem addressed
  - Large pre-trained language models are often poor at directly following natural-language instructions without careful prompt engineering or many in-context examples. They can also struggle to produce structured, step‑by‑step reasoning on unseen tasks.
  - Prior instruction finetuning work improved instruction following but (a) used far fewer tasks or smaller models, and (b) tended to hurt chain‑of‑thought (CoT) reasoning when the finetuning data did not contain CoT rationales (Section 4.2, Figure 5).

- Why this matters
  - Practical impact: better zero/few-shot task performance reduces the need for hand‑crafted prompts and few‑shot exemplars, improving usability across domains and languages. Human raters preferred Flan‑PaLM answers 79% of the time on open‑ended questions (Figure 8).
  - Scientific significance: clarifies how instruction finetuning scales with both model size and number/type of tasks; shows that a tiny amount of CoT data can preserve and unlock reasoning abilities.

- Prior approaches and gaps
  - InstructGPT, FLAN, T0 and related work used instruction finetuning but with fewer tasks (tens to hundreds) and smaller models; chain‑of‑thought prompting helped only at very large scales and was not part of most finetuning mixtures.
  - This paper extends scope by: (i) scaling finetuning to 1,836 tasks across 473 datasets (Figure 2), (ii) including explicit CoT datasets (nine tasks with human‑written rationales), and (iii) studying scaling across PaLM 8B/62B/540B, T5 80M–11B, cont‑PaLM, and U‑PaLM (Table 2).

- Positioning
  - The work unifies and scales multi‑task instruction finetuning (“Flan”) and analyzes its effects on unseen tasks, reasoning, multilingual QA, and open‑ended generation, while also reporting safety/representational metrics (Appendix C). It shows compatibility with other continued pre‑training methods (UL2R) and releases Flan‑T5 checkpoints.

## 3. Technical Approach
Instruction finetuning basics
- “Instruction finetuning” means supervised finetuning where each training example is formatted as a natural-language instruction (“Do X…”), possibly with a few exemplars in the prompt, paired with a target answer. It teaches a model to map instructions to outputs directly.
- “Few‑shot exemplars” are a few input‑output pairs shown in the prompt; “zero‑shot” uses just the instruction.
- “Chain‑of‑thought (CoT)” is a human‑written or model‑generated step‑by‑step rationale preceding the final answer. “Self‑consistency (SC)” samples multiple CoT solutions and chooses the most common final answer.

Data and formatting (Section 2.1, Figures 2–3; Appendix F)
- Finetuning combines four mixtures to reach 1,836 tasks:
  - `Muffin` (80 tasks): FLAN-style tasks plus added dialog and program synthesis.
  - `T0‑SF` (193 tasks): T0 tasks excluding overlap with Muffin.
  - `NIv2` (1,554 tasks): Natural Instructions v2 (with 44 MMLU-related tasks removed to keep MMLU held‑out).
  - `CoT` (9 tasks): human‑written chain‑of‑thought datasets (e.g., arithmetic, multi‑hop reasoning, NLI).
- Instruction templates:
  - For each task, multiple templates are used to vary phrasing and delimiters (e.g., “Q:/A:”) and to create both few‑shot and zero‑shot formats (Figure 3).
  - CoT tasks have ≈10 instruction templates each; CoT rationales are included in the target, followed by the final answer.
- Language coverage: 60 languages present across tasks (Appendix F, Figure 18).

Models and training (Section 2.2; Table 2; Appendix E)
- Models finetuned: T5 (80M–11B, encoder‑decoder), PaLM (8B/62B/540B, decoder‑only), cont‑PaLM (PaLM‑62B with extra pre‑training), and U‑PaLM (540B with UL2 objective continuation).
- Optimization: Adafactor with constant learning rate; “packing” multiple examples per sequence; masking to prevent cross‑example attention.
- Compute efficiency: finetuning FLOPs are small vs pre‑training. Example: Flan‑PaLM‑540B uses 5.6e21 finetune FLOPs vs 2.5e24 pre‑train FLOPs (0.2% of pre‑training compute; Table 2).
- Checkpoint selection: a single checkpoint per model chosen by periodic evaluation on held‑out tasks; the same step count is used across ablations for a fair comparison (Section 2.2).
- Mixture sampling and caps (Appendix E, Table 23): per‑task sampling weights proportional to number of examples with caps to avoid dominance (e.g., 5k cap for NIv2 tasks; specific proportions used in scaling vs final runs).

Evaluation protocol (Section 2.3)
- Held‑out benchmarks, each with fixed few‑shot settings and metrics:
  - `MMLU` (57 tasks): multi‑discipline multiple-choice; five‑shot. Evaluate both direct answers and CoT prompting (exact match accuracy).
  - `BBH` (23 challenging BIG‑Bench tasks; Section 4): three‑shot; evaluate direct and CoT.
  - `TyDiQA` (8 languages): one‑shot extractive QA; direct prompting exact match.
  - `MGSM` (10 languages): math word problems; eight‑shot CoT prompting accuracy.
- Aggregate metric: a “normalized average” (Section 2.3) — macro‑average across six normalized scores (MMLU‑Direct, MMLU‑CoT, BBH‑Direct, BBH‑CoT, TyDiQA‑Direct, MGSM‑CoT). Each task score is normalized relative to a lower bound (e.g., random guessing), so values below zero mean worse than baseline.

How CoT and SC are used
- During finetuning, only nine tasks contain CoT rationales (Figure 3, “with chain‑of‑thought”). This small CoT subset is designed to teach the model the “format” and utility of reasoning without overwhelming non‑CoT tasks.
- During inference, CoT prompting can be invoked by adding a rationale instruction (e.g., “answer by reasoning step‑by‑step”) or the phrase “let’s think step‑by‑step” (Section 4.3).
- Self‑consistency (Table 4) samples multiple CoT solutions and votes on the final answer, improving robustness of reasoning.

## 4. Key Insights and Innovations
1) Scaling instruction finetuning works and is compute‑efficient
- What’s new: Systematic scaling curves show gains from both larger models and more finetuning tasks (Section 3, Figure 4; Table 3). For PaLM‑540B, finetuning improves the normalized average by +9.4 points over no finetuning while using only 0.2% of pre‑training compute (Table 2).
- Why it matters: Demonstrates a practical, low‑compute path to improve very large models after pre‑training.

2) A small amount of CoT data is critical to retain and improve reasoning
- Observation: Finetuning on non‑CoT instructions alone degrades CoT performance substantially; adding only nine CoT datasets recovers and improves reasoning while maintaining non‑CoT performance (Section 4.2, Figure 5).
- Significance: Establishes that instruction finetuning should include both CoT and non‑CoT formats to avoid forgetting and to generalize across evaluation paradigms.

3) CoT finetuning unlocks zero‑shot reasoning with simple triggers
- Finding: After finetuning with some CoT data, the models generate effective rationales in zero‑shot setups when cued with “let’s think step‑by‑step,” substantially improving accuracy on the 23 BBH tasks (Section 4.3, Figure 6; examples in Figure 7).
- Impact: Reduces reliance on carefully constructed few‑shot CoT prompts — a practical usability gain.

4) Generality across architectures and synergy with continued pre‑training
- Evidence: Instruction finetuning improves T5, PaLM, cont‑PaLM, and U‑PaLM (Table 5). Combining UL2 continued pre‑training (U‑PaLM) with Flan yields the strongest overall results in that table.
- Importance: Shows Flan is a general adaptation layer compatible with different architectures and objectives.

5) Usability and some safety improvements
- Usability: On 190 open‑ended prompts, human raters preferred Flan‑PaLM‑540B responses 79% of the time (Figure 8).
- Safety: Lower toxicity rates on RealToxicityPrompts (Table 6, Figure 12) and better zero/ten‑shot toxicity classification AUC on CivilComments (Table 7). Caveat: disparities across identity terms remain (Figures 13–14); translation misgendering shows mixed results (Table 10).

## 5. Experimental Analysis
Evaluation design and baselines (Sections 2.3, 3–6; Tables 1–7, Figures 4–8; Appendix C–D)
- Datasets and metrics are held‑out from finetuning (e.g., MMLU tasks removed from NIv2; Section 2.1). Fixed few‑shot counts follow prior work: five‑shot MMLU, three‑shot BBH, one‑shot TyDiQA, eight‑shot MGSM (Section 2.3).
- Baselines include: non‑finetuned PaLM at the same parameter scale, prior SOTA models (e.g., Chinchilla, Codex), and for safety some human baselines (RealToxicityPrompts; Figures 12–14).

Main quantitative results
- Scaling and overall gains (Section 3; Figure 4; Table 3)
  - Normalized average improvements vs no finetuning:
    - PaLM‑8B: +15.5 (from 6.4 to 21.9).
    - PaLM‑62B: +10.4 (from 28.4 to 38.8).
    - PaLM‑540B: +9.4 (from 49.1 to 58.5).
  - Task‑wise for PaLM‑540B (Table 3):
    - MMLU Direct: 73.2 vs 71.3 (non‑finetuned).
    - MMLU CoT: 68.1 vs 62.9.
    - BBH Direct: 58.8 vs 49.1.
    - BBH CoT: 65.6 vs 63.7.
    - TyDiQA (1‑shot EM): 67.4 vs 52.9.
    - MGSM CoT: 61.3 vs 45.9.

- New state‑of‑the‑art with CoT + Self‑Consistency (Section 4.1; Table 4)
  > Flan‑PaLM‑540B with CoT + SC achieves “75.2%” on five‑shot MMLU (Table 4), surpassing prior bests (PaLM 69.3%; Chinchilla 67.6%—see Table 1 and Table 4).
  - On MGSM, CoT + SC yields 72.0% (Table 4) vs PaLM‑540B 57.9% with CoT + SC.
  - On BBH algorithmic tasks, Flan‑PaLM improves but does not beat specialized code models on some subtasks (Table 4 notes Codex remains strong for BBH‑alg).

- CoT ablation and “balancing” conclusion (Section 4.2; Figure 5)
  - Finetuning with only non‑CoT hurts held‑out CoT tasks (green curve, Figure 5 left).
  - Joint CoT + non‑CoT finetuning improves both CoT and non‑CoT held‑out performance (blue curve higher on both left and right plots).

- Zero‑shot CoT unlocked (Section 4.3; Figure 6; examples in Figure 7)
  - On 23 BBH tasks, zero‑shot with “let’s think step‑by‑step” substantially improves Flan‑PaLM accuracy across 8B/62B/540B, while non‑finetuned PaLM benefits little (Figure 6). Qualitative examples (Figure 7) show (i) PaLM failing to answer or looping, (ii) Flan‑PaLM producing concise, correct rationales.

- Cross‑architecture generality (Table 5)
  - All families improve on the normalized average. Illustrative numbers:
    - `T5‑XXL 11B`: from −2.9 (LM‑adapted T5) to 23.7 (Flan‑T5‑XXL).
    - `T5‑XL 3B`: MMLU Direct 52.4% (Flan‑T5‑XL), exceeding GPT‑3 five‑shot 43.9% (Table 1).
    - `Flan‑U‑PaLM 540B`: normalized average 59.1 vs U‑PaLM’s 50.2 (Table 5), indicating complementarity with UL2R.

- Open‑ended usability (Section 6; Figure 8; Figure 9)
  > Over 190 prompts spanning creativity, planning, complex reasoning, explanation, and few‑shot settings, Flan‑PaLM outputs are preferred “79%” of the time (Figure 8).
  - Gains are largest on zero‑shot reasoning categories; CoT triggers (“let’s think step-by-step”) further increase preference by ≈10% on those categories.
  - Error analysis shows non‑finetuned PaLM often repeats the prompt, continues context rather than answering, or fails to stop (Figure 9). Instruction finetuning mitigates these behaviors.

- Safety and representational metrics (Appendix C)
  - Toxic degeneration (RealToxicityPrompts; Table 6, Figure 12): Flan‑PaLM reduces probability of toxic continuations relative to PaLM at all scales (e.g., PaLM‑540B: 0.80→0.52 on toxic prompts; 0.44→0.18 on non‑toxic prompts; Table 6).
  - Toxicity classification (CivilComments; Table 7): Large AUC jumps in zero‑shot, e.g., PaLM‑540B 71.4 → Flan‑PaLM‑540B 86.5; ten‑shot 82.1 → 87.1.
  - Gender/occupation coreference (Winogender; Appendix C.4, Figures 15–16): instruction finetuning improves zero‑shot and few‑shot performance; Flan‑T5‑XXL is particularly strong; however, “gotcha” cases remain harder than stereotypical ones.
  - Translation misgendering (Table 10): mixed impact; errors are lower overall for some sets/languages but worst‑case slices persist (e.g., “she” pronouns; very low‑resource languages). Bias disparities across identity groups remain (Figure 14).

Do the experiments support the claims?
- The work offers extensive ablations (task scaling in Table 3, CoT inclusion in Figure 5), cross‑model validations (Table 5), SOTA comparisons (Table 4), and human evaluations (Figure 8) that collectively support the core claims: instruction finetuning scales, CoT data is necessary for reasoning retention and zero‑shot CoT, and Flan improves usability and some safety metrics.
- Caveats are acknowledged: specialized models can outperform on algorithmic manipulation; TyDiQA finetuned ByT5 remains stronger than Flan‑PaLM on that dataset; safety improvements do not eliminate tail toxicity or bias disparities (Table 4 note; Appendix C).

## 6. Limitations and Trade-offs
- Dependence on curated task mixtures
  - Gains after ~282 tasks are smaller (Figure 4), suggesting diminishing returns if added tasks are not diverse or mostly teach formatting rather than new knowledge (Section 3). Curating and templating 1.8K tasks is labor‑intensive.
  - Only nine CoT datasets are used; they may bias the “style” of reasoning the model learns.

- Compute and infrastructure
  - Although finetuning compute is tiny relative to pre‑training, absolute costs are still large for the biggest model (e.g., ~512 v4 TPUs for ~37 hours for 540B; Section 2.2). This limits accessibility.

- Mixed or conditional benefits
  - Chain‑of‑thought does not always beat direct prompting (e.g., MMLU direct often higher than CoT; Table 5), and specialized algorithmic models can outperform Flan on BBH‑alg tasks (Table 4).
  - TyDiQA: Flan‑PaLM improves strongly but remains below ByT5 trained directly on TyDiQA (Table 4 note).

- Safety and bias
  - Despite reductions, tail toxicity remains high for some identity groups (Figure 14), and misgendering persists in translation (Table 10). Safety metrics rely on imperfect automatic tools (Appendix C.6).

- Generalization assumptions
  - The “normalized average” aggregates diverse metrics/tasks (Section 2.3); improvements in the aggregate may hide regressions on specific niches.
  - While MMLU tasks were held out of finetuning mixes, pre‑training contamination is always a concern; the paper references prior analyses finding little contamination (Section 2.3), but this remains hard to rule out universally.

## 7. Implications and Future Directions
- How this shifts the landscape
  - Instruction finetuning emerges as a default, compute‑efficient post‑training step for large language models across architectures. The results indicate that even a small injection of CoT data is crucial for reasoning and zero‑shot CoT, changing best practices for model adaptation.

- Follow‑up research enabled/suggested
  - Task selection and diversity: identify which new tasks deliver the biggest marginal gains beyond ~282 tasks; automate mixture construction and weighting.
  - Reasoning data: expand CoT coverage (domains, languages), explore synthetic rationale generation and bootstrapping (e.g., with self-consistency), and study how CoT “style” affects transfer.
  - Safety: design finetuning mixtures explicitly targeting long‑tail toxicity and bias, beyond what RealToxicityPrompts captures; evaluate multilingual safety.
  - Methods: combine Flan with other efficient adaptation strategies (e.g., RLHF, adapters, parameter‑efficient finetuning) and with continued pre‑training (UL2R showed complementarity in Table 5).
  - Zero‑shot usability: formalize prompt triggers and develop robust instruction sets that generalize across tasks and languages without exemplars.

- Practical applications
  - Rapid deployment of generalist assistants with improved instruction following, step‑by‑step reasoning, and open‑ended response quality (Figure 8).
  - Multilingual QA and math reasoning (TyDiQA and MGSM; Tables 3–4), especially in under‑represented languages where gains are large.
  - Safer default generations and stronger out‑of‑the‑box toxicity classification (Tables 6–7), with the caveat that fairness audits remain essential.

Overall takeaway
> Instruction finetuning at scale — with a balance of non‑CoT and a small amount of CoT data — is a general, low‑compute strategy that substantially improves large language models’ instruction following, reasoning, multilingual abilities, and usability, while offering some safety benefits (Sections 3–6; Tables 3–7; Figures 4–8).
