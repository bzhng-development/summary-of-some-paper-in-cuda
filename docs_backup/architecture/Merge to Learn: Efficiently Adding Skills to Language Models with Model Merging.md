# Merge to Learn: Efficiently Adding Skills to Language Models with Model Merging

**ArXiv:** [2410.12937](https://arxiv.org/abs/2410.12937)
**Authors:** Jacob Morrison, Noah A. Smith, Hannaneh Hajishirzi, Pang Wei Koh, Jesse Dodge, Pradeep Dasigi
**Institutions:** 

## 🎯 Pitch

This paper introduces Parallel Train Then Merge (PTM), a method to integrate new skills into instruction-tuned language models without retraining, preserving existing knowledge and using just a fraction of the compute. By enabling efficient skill expansion and robust safety improvements, PTM supports continuous model evolution, making it a significant advancement for cost-effective, targeted model enhancement in rapidly changing environments. 

---

## 1. Executive Summary
This paper shows a practical way to add new skills to an already instruction‑tuned language model without retraining it from scratch and without erasing what it already knows. The core idea—“parallel train then merge” (PTM)—trains a small skill‑specific model and then combines its weights with a general model using a “task vector”; across science, safety, and coding skills, PTM often matches retraining on the new task while preserving general abilities and using a fraction of the compute (e.g., ~4% of training steps vs. retraining in Table 2).

## 2. Context and Motivation
- Problem addressed
  - How to add new capabilities (e.g., science literature understanding, coding, safety refusals) to an existing instruction‑tuned LM while preserving its general skills and keeping costs low (Section 1, Section 2).
- Why this matters
  - Instruction data is evolving; repeatedly retraining general models is expensive and sometimes infeasible because the original training mix is unavailable for many public models (e.g., Llama 3, Mistral 7B, Gemma; Section 2.2).
  - Continued finetuning (CFT) on a new skill often causes catastrophic forgetting—loss of earlier capabilities (Section 1; quantified in Table 3 as 10–40% drops on “General”).
  - Safety is a special case: you want stronger refusals to harmful prompts without over‑refusing benign requests (“exaggerated refusals,” defined in Section 3.1).
- Prior approaches and their gaps
  - CFT: inexpensive but degrades general ability (Section 2.1; Table 3 shows −32.5% to −40.1% on “General” when optimized for new skills).
  - Retraining from scratch (RT) with a combined data mix: preserves general ability better but is expensive and sometimes impossible due to missing base mixes (Section 2.2).
  - Model merging literature exists (task vectors, linear interpolation, WiSE‑FT; Section 2.3; Related Work Section 5), but had not been systematically tested for instruction‑tuned LMs and multi‑skill addition.
- Positioning
  - The paper formulates and evaluates a PTM pipeline for instruction tuning, comparing it head‑to‑head with CFT and RT on cost, general‑skill retention, and new‑skill gains across three domains (Section 3).

## 3. Technical Approach
At a high level, PTM follows a simple, repeatable pipeline:

1) Train a skill‑specific model in isolation
- Start from a pretrained base `θ_pre` and fully finetune on new skill data `D` for two epochs (Section 3.2), producing `θ_D`.
- This isolates skill learning from the general instruction mix, avoiding interference during training.

2) Build a task vector
- A “task vector” is the parameter difference between the skill‑specific model and its starting point:
  - Equation (1): `τ_D = θ_D − θ_pre` (Section 2.3, “Task Arithmetic”).
- Intuition: `τ_D` encodes what changed to learn the skill.

3) Merge the skill into the general model
- Combine the task vector with an existing general instruction‑tuned model `θ_G`:
  - Equation (2): `θ_final = θ_G + ω · τ_D`, where `ω` is a scalar weight controlling how much of the new skill to add (Section 2.3).
- Model selection for `ω`
  - If held‑out validation exists, tune `ω` to balance general vs. new‑skill performance.
  - When no held‑out data is available, use a simple heuristic (Equation (7) in Section 4.2): `ω = |D| / |G|`, i.e., the ratio of skill‑data steps to general‑data steps. Figure 1 highlights that this often lands near a good trade‑off across domains.

Why task arithmetic over alternatives?
- The paper evaluates three PTM instantiations (Section 2.3):
  - Task arithmetic (above).
  - Linear interpolation: treat both general and skill vectors relative to `θ_pre`, then linearly mix them (Equations (3)–(4)).
  - WiSE‑FT: continue finetuning the general model on the skill (`θ_CFT` from `θ_G`), then merge back with a weight (Equations (5)–(6)).
- Empirically, task arithmetic best preserves general skills for a given skill gain (Figure 3), and it also has the lowest training cost because it trains a single skill model once and then explores multiple `ω` values at negligible extra cost.

Experimental design (Section 3)
- Base and infrastructure
  - Backbone: `Llama 2 7B` (Touvron et al., 2023).
  - Training hyperparameters (Appendix A.2): full finetuning, 2 epochs, batch size 128, max length 4096, learning rate 2e−5, warmup 3%.
  - Compute: Google TPU v3; training steps are the cost unit (Section 2).
- Data (Table 1; Section 3.1)
  - General training: modified `Tülu V2` mix (275k instances) with science, code, and refusals removed to create room for “new skill” gains.
  - Skill datasets:
    - `SciRIFF` (61k): multi‑task scientific literature understanding; has validation and test (Section 3.1 “Science”).
    - `Safety` (66k): internally built harmful prompts with GPT‑4 refusals; covers malicious uses, toxicity, misinformation (Section 3.1 “Safety”).
    - `CodeFeedback` single‑turn subset (156k) (Section 3.1 “Coding”).
- Evaluations (Sections 3.1; Tables 7–16 for per‑benchmark detail)
  - General: average over MMLU, GSM8K, BBH, TruthfulQA, AlpacaEval.
  - Science: SciRIFF’s nine validation/test tasks.
  - Safety: average of ToxiGen, HarmBench, and XSTest Unsafe, plus a separate “Exaggerated Refusals” metric from XSTest Safe.
  - Coding: HumanEval+ and MBPP+, pass@10 with temperature 0.8.

Cost accounting (Section 2)
- CFT: train multiple runs on various subsampled `D_i`; cost is sum of `|D_i|`.
- RT: for each mixing ratio, train on all `G` plus some `D_i`; cost `n · |G| + Σ_i |D_i|`—expensive because `|G|` is large.
- PTM: train once on all `D` to get a single task vector; selection over `ω` is essentially free; cost `|D|`.

## 4. Key Insights and Innovations
- Efficient “skills as vectors” works for instruction‑tuned LMs
  - Novelty: A systematic, multi‑domain evaluation of task‑vector merging for instruction tuning (Section 4). Prior merging/editing work focused on other settings (Section 5).
  - Significance: PTM matches or approaches retraining on new skills while preserving the general skill set and cutting compute by 50–95% (Section 4.1; Table 2).
- Simple weight heuristic generalizes
  - Insight: Setting `ω = |D| / |G|` balances new‑skill gains with general‑skill preservation when no validation data exists (Equation (7)).
  - Evidence: Across science, safety, and coding, the “heuristic” points in Figure 1 lie near the knee of the trade‑off curves.
- Safety gains without over‑refusal
  - Insight: A “safety vector” dramatically improves refusal of unsafe prompts while avoiding exaggerated refusals—i.e., refusing safe but superficially risky prompts (Section 4.2 “PTM Mitigates Exaggerated Refusals”).
  - Evidence:
    - Table 3 shows “Best PTM (Safety)” changes general by −0.13% and improves safety by +88.9%, while “Best CFT (Safety)” loses 40.1% on general.
    - With all three skills merged, “Exaggerated Refusals” improve to 93.2 vs. 16.0 (CFT) and 37.2 (RT) in Table 4.
- Diagnosis of multi‑skill interference
  - Insight: Merging multiple skills sometimes creates negative interactions, especially between coding and safety vectors for science tasks (Section 4.3).
  - Evidence: Pairwise merges in Table 5 show the “Safety and Coding” pair drops Science to 18.8 vs. 32.1 (“Science and Coding”) and 31.6 (“Science and Safety”).
- Why some PTM variants hurt general skills
  - Observation: WiSE‑FT and linear interpolation can reach strong new‑skill results but degrade general ability more than task arithmetic (Figure 3).
  - Mechanism: WiSE‑FT finetunes from `θ_G` using only `D`, shifting the distribution; mixing a matching amount of general data with `D` during the WiSE‑FT step restores general skill and improves science (Figure 2).

## 5. Experimental Analysis
- Setup recap (Sections 3.1–3.2; Table 1; Appendix A)
  - Models: Llama 2 7B variants; all finetuned for two epochs.
  - Skill domains: Science (SciRIFF), Safety (internal), Coding (CodeFeedback).
  - Metrics: Composite general score; skill‑specific aggregates; separate “Exaggerated Refusals” from XSTest Safe.
  - Baseline “Tülu Only” general score: 49.9 (Table 2).

- Main quantitative results
  - Science, compute vs. performance (Section 4.1; Table 2)
    - PTM reaches Science 38.2 and General 47.1 with only 479 steps.
    - Retraining (RT) reaches Science 37.8 and General 50.6 but costs 11,766 steps.
    - Continued finetuning (CFT) reaches Science 40.6 but General falls to 33.7 (1,005 steps).
    - Quote the trade‑off:
      > Table 2: “PTM shows equivalent science performance to the best RT model … while taking about 4% as many training steps.”
  - Cross‑domain deltas (Section 4.2; Table 3)
    - PTM preserves general ability compared to CFT in all domains.
      - Example: “Best PTM (Coding)” improves General by +1.43% and Coding by +33.3%, whereas “Best CFT (Coding)” drops General by −7.73% even though Coding is +51.6%.
    - Safety stands out:
      - “Best PTM (Safety)” changes General by −0.13% and improves Safety by +88.9%; “Best RT (Safety)” is +0.66% General and +89.6% Safety but ~24× the training steps (12,311 vs. 517).
    - Exaggerated refusals:
      - “Best PTM (Ex. Ref.)” improves exaggerated‑refusal compliance by +72.6 (percentage points; see Table 3 note) at modest general loss (−6.45%).
  - Multi‑skill merging (Section 4.3; Table 4)
    - Merge all three vectors (“PTM (All 3)”):
      - General improves to 51.1 (vs. 49.9 baseline).
      - Coding: 45.3 (up from 37.6).
      - Safety: 84.0 (up from 50.3).
      - Exaggerated Refusals: 93.2 (huge gain vs. CFT 16.0 and RT 37.2).
      - Science drops slightly to 26.6 (vs. 27.8).
      - Cost: zero extra training steps once the three skill vectors exist (“Additional Training Steps” column).
    - Interference diagnosis (Table 5):
      > Science collapses most when merging “Safety and Coding” (Science 18.8) compared to “Science and Safety” (31.6) and “Science and Coding” (32.1).
  - PTM variants (Section 4.4; Figure 3; Figure 2)
    - Figure 3: task arithmetic best preserves general ability; linear interpolation and WiSE‑FT show larger general drops for similar new‑skill gains.
    - Figure 2: WiSE‑FT trained on SciRIFF plus a matched amount of Tülu data retains more general skill and improves science vs. WiSE‑FT on SciRIFF alone.
  - Other merge algorithms (Appendix B; Table 6)
    - TIES and DARE, which aim to reduce interference, do not outperform simple weighted averaging in this setup and do not fix the science drop in the 3‑skill merge.

- Robustness, ablations, and diagnostics
  - Trade‑off curves: The paper plots full curves over ω (Figure 1; Figure 3; Figure 6), not just single points, showing PTM’s controllable trade‑off between general and skill‑specific performance.
  - Heuristic selection: The `ω = |D| / |G|` marker is consistently near a good operating point (Figure 1; Figure 5).
  - Safety nuance: Direct comparison of “General vs. Exaggerated Refusals” demonstrates PTM’s advantage at any given general score (Figure 4).

- Do the experiments support the claims?
  - Yes for efficiency and retention: Across domains, PTM achieves skill gains comparable to RT with far fewer training steps, while CFT’s general performance consistently drops (Tables 2–4).
  - Yes for safety behavior: PTM boosts refusals and reduces exaggerated refusals substantially (Tables 3–4; Tables 12 and 16).
  - Mixed for multi‑skill composition: Merging multiple skills is feasible but can create interference; the paper both exposes and diagnoses this with pairwise analyses (Table 5), which is candid and useful.

## 6. Limitations and Trade-offs
- Assumptions and scope
  - Focus on one backbone (`Llama 2 7B`) and supervised instruction tuning; no RLHF or post‑training alignment layers are studied (Limitations section).
  - Coding evaluation uses single‑turn data and pass@10; multi‑turn code refinement is outside scope (Section 3.1).
- When PTM may underperform
  - Multi‑skill interference can hurt some domains (science in 3‑skill merge), and common anti‑interference mergers (TIES/DARE) did not resolve it here (Table 6).
  - WiSE‑FT and linear interpolation variants can degrade general skills if the finetuning distribution differs from the general mix (Figure 3); adding matched general data helps but increases cost (Figure 2).
- Data and selection constraints
  - Many instruction datasets lack validation splits; the paper offers a heuristic for `ω`, but selection without held‑out data will still be approximate (Section 3.2; Section 4.2).
- Compute reporting
  - Cost is measured in “training steps” at fixed batch size; hardware, FLOPs, and wall‑clock time may vary across setups (Section 2).
- External validity
  - Results are on open datasets and one internal safety dataset; broader generalization to other families, sizes, or proprietary corpora remains to be demonstrated (Limitations; Section 3.1).

## 7. Implications and Future Directions
- Practical impact
  - “Skills as plug‑in vectors” makes continual capability growth feasible without retraining large general models or needing access to their original training mix.
  - Organizations can add or update safety behaviors rapidly while avoiding excessive over‑refusal, which is critical for user experience and compliance (Tables 3–4, 12, 16).
  - Produces a zero‑inference‑overhead solution: merging is weight‑space, so runtime cost is unchanged compared to the base model (Section 2.3).
- New workflows this enables
  - A “skill library” ecosystem: curate and share task vectors (science, safety facets, coding languages, domains), then compose them as needed using simple weighted addition.
  - On‑device or privacy‑sensitive settings: PTM avoids retraining on potentially sensitive original data; only the new skill data is needed.
- Research directions
  - Interference‑aware composition: develop principled methods to detect and mitigate cross‑skill clashes during merge time; the failure case in Table 5 is a concrete target.
  - Data‑aware WiSE‑FT: quantify “how much general data” is needed during CFT to preserve general ability (two explicit questions raised in Section 4.4).
  - Automatic `ω` selection: meta‑learning or Bayesian selection of mixture weights using small, representative probes when validations are absent.
  - Beyond supervised instruction tuning: test PTM with models that underwent RLHF, tool‑use training, or multi‑turn interactive fine‑tuning.
  - Evaluation breadth: expand general‑reasoning and multi‑turn safety/coding tests to probe whether task vectors transfer to more complex interaction patterns.

Overall, this work reframes model adaptation as an efficient, modular merge problem. The evidence across sections and tables shows PTM as a strong default for adding skills: it usually matches retraining on the target skill, better preserves what the model already knows than continued finetuning, and is especially effective and controllable for safety behaviors.
