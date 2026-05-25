# Merge to Learn: Efficiently Adding Skills to Language Models with Model Merging

**ArXiv:** [2410.12937](https://arxiv.org/abs/2410.12937)

## 🎯 Pitch

This paper introduces 'parallel train–then–merge' (PTM), a novel method for efficiently adding new skills to instruction-tuned language models by training new capabilities in isolation and merging them into the existing model in weight space. PTM achieves skill gains comparable to full retraining while preserving general abilities far better than standard finetuning, all at a fraction of the computational cost. This approach is especially impactful in practical settings where original training data isn't available, enabling fast, low-risk updates that expand model capabilities without forgetting prior knowledge.

---

## 1. Executive Summary
This paper shows how to add new capabilities to already instruction-tuned language models by training the new skill in isolation and then merging it into the existing model in weight space. The proposed “parallel train–then–merge” (PTM) approach frequently matches retraining from scratch on performance while cutting training cost by an order of magnitude and, crucially, preserves general abilities far better than continued finetuning—especially for safety where it also avoids over-refusing harmless prompts.

## 2. Context and Motivation
- Problem addressed
  - Instruction-tuned language models need periodic “patches” to pick up new skills (e.g., reading scientific papers, writing code, refusing unsafe requests). The common options are:
    - `continued finetuning (CFT)`: keep training the current model on the new dataset.
    - `retraining (RT)`: start from the base pretrained model and train again on a new mixture of old + new data.
    - Both approaches require multiple trials to find the right data ratio and can damage previously learned skills (catastrophic forgetting).
  - Moreover, many public models ship without their original instruction datasets, making RT impossible in practice (Section 2; examples include Mistral 7B and Llama 3).

- Importance
  - Practitioners need a low-cost, low-risk way to add skills without erasing old ones and without access to the original data. This has both practical value (shipping updates quickly) and scientific interest (understanding when weight-space linearity/merging works).

- Prior approaches and gaps
  - CFT is cheap but often forgets old skills.
  - RT can preserve balance but is computationally expensive and often infeasible without the original mixture (Section 2.2).
  - Weight-space methods—task vectors, linear interpolation, WiSE-FT—exist but had not been systematically studied for instruction tuning across diverse skills and safety trade-offs (Sections 2.3, 5 Related Work).

- Positioning
  - The paper systematically evaluates PTM for instruction-tuned LMs and compares it head-to-head with CFT and RT across three skills (Science, Safety, Coding), with careful cost accounting and analysis of refusal behavior (Sections 3–4).

## 3. Technical Approach
The goal is to use a new skill-specific dataset `D` to improve performance on a new evaluation `E_D` while preserving general performance on `E_G` and minimizing training cost (Section 2).

- Methods compared (Sections 2.1–2.3)
  - Continued Finetuning (CFT)
    - Train the existing instruction-tuned model `θ_G` on the new dataset `D`.
    - Requires trying several data sizes `D_i` to find a good balance; total cost is the sum of training steps across these trials, `Σ |D_i|` (Section 2.1).
    - Risk: catastrophic forgetting of general skills.

  - Retraining (RT)
    - Start from the pretrained model `θ_pre` and train on a new mix `G + D_i` with several `D_i` ratios; cost is `n·|G| + Σ|D_i|`, dominated by the general set `|G|` (Section 2.2).
    - Strong performance, but expensive and often infeasible without access to `G`.

  - Parallel Train–then–Merge (PTM) in weight space (Section 2.3)
    - Step 1: Train a “skill model” on `D` only.
      - For task arithmetic, finetune `θ_pre` on `D` to get `θ_D`.
      - Form the task vector `τ_D = θ_D − θ_pre` (Eq. 1).
    - Step 2: Merge with the general model in weight space.
      - Add the weighted task vector to the instruction-tuned general model:
        - `θ_final = θ_G + ω · τ_D` (Eq. 2).
      - `ω` is a scalar mixture weight that controls how much the new skill affects the final model.
    - Cost: only one training run on `D` (`|D|` steps), because the balance is tuned by `ω` instead of retraining multiple times (Section 2.3).
    - Two alternatives are also evaluated (Section 2.3):
      - Linear interpolation: create a “general task vector” `τ_G = θ_G − θ_pre` (Eq. 3), then `θ_final = θ_pre + ω·τ_D + (1−ω)·τ_G` (Eq. 4).
      - WiSE-FT: finetune `θ_G` on `D` to get `θ_CFT`, then `θ_final = θ_G + ω·(θ_CFT − θ_G)` (Eqs. 5–6). Does not require `θ_pre`.

- Selecting the mixture weight `ω`
  - When a validation set exists, choose `ω` by validation (Section 3.2).
  - Otherwise, use a simple heuristic (Section 4.2):
    - Train on all of `D`, and set `ω = |D| / |G|` (Eq. 7).
    - Figure 1 highlights this point on the trade-off curves; it consistently lies near the “good trade-off” region (preserves general skill while boosting the new skill).

- Extending to multiple skills
  - Train one task vector per skill and add them together:
    - `θ_final = θ_G + Σ_k ω_k · τ_{D_k}` (implicit in Sections 4.3 and Table 4).
  - No extra training if the single-skill task vectors already exist; only the final merge changes.

- Experimental setup (Section 3)
  - Base and training details (Section 3.2; Appendix A):
    - Base: Llama 2 7B.
    - Instruction-tuned general model `θ_G`: trained on a modified Tülu V2 mix (275k instances after removing science, code, and refusal subsets to create room for improvement).
    - All finetuning: 2 epochs, context length 4096, batch size 128, bfloat16, LR 2e-5 on TPU v3. Merging implemented with MergeKit.
  - Skill datasets and evaluations (Section 3.1; Table 1):
    - Science: train on SciRIFF (61k). Evaluate on 9 held-out science tasks (validation + test).
    - Safety: internal refusals dataset (66k). Evaluate via HarmBench, XSTest Unsafe, ToxiGen; separately track exaggerated refusals with XSTest Safe.
      - Exaggerated refusals definition: the model incorrectly refuses harmless prompts that superficially resemble unsafe ones. Higher XSTest Safe score means fewer exaggerated refusals (more correct compliance).
    - Coding: train on CodeFeedback single-turn (156k). Evaluate with HumanEval+ and MBPP+ (pass@10 with temperature 0.8).
    - General evaluations: MMLU, GSM8K, BigBench Hard (BBH), TruthfulQA, AlpacaEval.

## 4. Key Insights and Innovations
- Efficient skill addition without old-data access
  - PTM reaches near-RT performance at a fraction of the cost because it trains only on `D` once and tunes `ω` in merging (Section 4.1).
  - Evidence: Table 2 shows PTM matches RT on science (38.2 vs 37.8) while costing 479 vs 11,766 steps—about 4% of RT’s training steps.

- Strong preservation of general skills vs CFT
  - Across all three domains, PTM keeps general abilities near baseline, whereas CFT often collapses them (Table 3).
  - Example: for Safety, “Best CFT” reduces general by 40.1%, while “Best PTM” is −0.13% (Table 3).

- Safety advantage: mitigates exaggerated refusals
  - When the goal is to increase safety refusal rates, PTM sharply reduces exaggerated refusals compared to RT and CFT (Section 4.2).
  - Example with all three skills merged: PTM achieves XSTest Safe 93.2 versus CFT 16.0 and RT 37.2—a 56–77 point advantage (Table 4). This shows PTM preserves the ability to comply with benign prompts while being safer on genuinely unsafe prompts.

- Practical heuristic for `ω` and multi-skill merging
  - The simple `ω = |D|/|G|` rule reliably lands on a good trade-off without validation data (Figure 1).
  - Multi-skill merging is “free” once per-skill vectors exist (Table 4), though it reveals interference effects the paper then diagnoses (next point).

- Diagnosing interference via ablations enabled by cheap merging
  - Pairwise merges isolate interference between coding and safety vectors that harms science performance (Table 5 shows Science drops to 18.8 when Safety+Coding are merged, versus 32.1 and 31.6 for other pairs). This analysis is possible because PTM makes many variants cheap to test.

Overall, the fundamental innovation is not “merging” itself—known techniques are used—but a thorough, instruction-tuning–focused methodology that demonstrates when and how PTM offers the best compute–performance–safety trade-off, plus practical selection heuristics and interference diagnostics.

## 5. Experimental Analysis
- Evaluation design (Sections 3.1–3.2; Table 1)
  - General model trained on a modified Tülu V2 mix (275k). Skills trained on SciRIFF (61k), Safety (66k), and CodeFeedback (156k).
  - For each method:
    - CFT/RT: five data mix sizes for `D`.
    - PTM: five `ω` values (0.2…1.0). Selection by validation when available; otherwise the `|D|/|G|` heuristic.

- Main results and comparisons
  - Science single-skill addition (Section 4.1; Table 2)
    - General vs Science test performance and training steps:
      > Tülu-only: General 49.9, Science 27.9  
      > Best CFT: General 33.7, Science 40.6, Steps 1,005  
      > Best RT: General 50.6, Science 37.8, Steps 11,766  
      > Best PTM: General 47.1, Science 38.2, Steps 479
    - Interpretation: PTM nearly matches RT on science (+0.4 absolute) at ~4% of the training steps; it dramatically outperforms CFT on general skills (+13.4 absolute) while being within 2.4 points on science.

  - Cross-skill summary (Table 3; percentage change relative to baseline)
    - Science: PTM keeps general skill (+1.30%) much better than CFT (−32.5%), with a moderate science gain (+26.3%).
    - Safety: PTM maintains general (−0.13%) while nearly matching safety gains to RT (+88.9% vs +89.6%).
    - Coding: PTM even increases general (+1.43%) while improving coding (+33.3%), though RT/CFT gain more on coding-specific metrics.
    - Exaggerated refusals focus: optimizing refusal behavior can crater general skills for CFT (−85.1%) and RT (−39.9%); PTM degrades much less (−6.45%).

  - Multi-skill merging (Section 4.3; Table 4)
    - With Science+Safety+Coding merged using the heuristic:
      > PTM(All 3): General 51.1 (+1.2 vs baseline), Science 26.6 (−1.2), Coding 45.3 (+7.7), Safety 84.0 (+33.7), Exaggerated Refusals 93.2 (strong).
    - CFT/RT on all three skills achieve higher coding and safety than PTM but at the cost of generalized ability and exaggerated refusals:
      > CFT(All 3): General 40.3 (−9.6), Exaggerated Refusals 16.0 (very poor).  
      > RT(All 3): General 50.1 (≈baseline), Exaggerated Refusals 37.2 (far below PTM’s 93.2).
    - The Science drop with PTM(All 3) is investigated via pairwise merges (Table 5):
      > Safety+Coding yields Science 18.8 (large drop) whereas Science+Coding is 32.1 and Science+Safety is 31.6.  
      This isolates interference between Safety and Coding task vectors as the main source of the Science degradation.

  - Method variants: task arithmetic vs linear interpolation vs WiSE-FT (Section 4.4; Figure 3)
    - All three improve the target skill as `ω` increases, but:
      - Task arithmetic preserves general skill best across domains.
      - Linear interpolation and WiSE-FT frequently trade off large general-skill losses for specialized gains.
    - Why WiSE-FT loses more general skill: distribution shift. Figure 2 shows that mixing SciRIFF with a matched amount of general data during CFT markedly improves both science and general performance compared to pure SciRIFF CFT.

  - Additional merging algorithms (Appendix B; Table 6)
    - Interference-aware methods TIES and DARE, applied to the three-skill merge, do not fix the Science degradation and offer no clear advantage over standard weighted averaging.

  - Safety deep-dive: exaggerated refusals trade-offs (Appendix B; Figures 4–5)
    - Figures 4–5 plot general skill vs exaggerated refusals. For the same general performance, PTM achieves much higher XSTest Safe (fewer exaggerated refusals) than CFT/RT, and the heuristic `ω` point lies on a favorable segment of the trade-off curve.

- Do the experiments support the claims?
  - The study covers three distinct capability areas, multiple baselines (CFT/RT), several merging methods, and includes ablations diagnosing interference and distribution effects. Results are consistently presented as both overall summaries (Tables 2–4) and detailed per-benchmark scores (Tables 7–16), with trade-off curves (Figures 1–3, 6). This breadth supports the central claims: PTM is compute-efficient, preserves general skills, and is particularly advantageous for safety with reduced exaggerated refusals.

- Where results are mixed or conditional
  - Coding: PTM improves coding substantially but not as much as CFT/RT when those methods are tuned for coding (Table 3, Coding +51% for CFT/RT vs +33% for PTM).
  - Multi-skill: PTM shows excellent generalized ability and refusal behavior, but suffers Science interference when safety and coding vectors are combined (Table 5).

## 6. Limitations and Trade-offs
- Assumptions and conditions
  - Weight-space linearity: PTM relies on the idea that adding a task vector linearly produces useful superposition of behaviors. This may not hold uniformly across all tasks or at larger scales.
  - Access to `θ_pre`: Task arithmetic needs the pretrained checkpoint to compute task vectors (Eq. 1). If unavailable, WiSE-FT is an alternative but tends to erode general skills unless general data is mixed in (Figure 2, Section 4.4).

- Scope limitations
  - Single base architecture/size: Experiments use Llama 2 7B only. Behavior may differ for larger models or different architectures.
  - No RLHF or post-instruction tuning steps: The work focuses on instruction tuning, not downstream stages like RLHF (Limitations section). Interactions with later stages remain open.
  - Validation scarcity: Many instruction datasets lack validation splits; the paper proposes a heuristic (Eq. 7) but selection may still be suboptimal for some tasks.

- Interference and composition
  - Merging multiple skills can produce negative interference, as seen with Safety+Coding hurting Science (Table 5). Existing interference-reduction algorithms (TIES, DARE) did not help in this setting (Table 6).

- Metrics and cost accounting
  - Training cost is measured in training steps, not wall-clock time or energy. Step-equivalence across different data mixes or hardware is assumed (Section 2).

## 7. Implications and Future Directions
- How this changes the landscape
  - PTM reframes capability updates as a modular “skill library” problem: train once on each new dataset to get a task vector, then merge as needed. This is attractive for organizations that cannot (or prefer not to) retrain on the full general mixture or do not have access to it.
  - For safety, PTM offers a compelling path: markedly higher refusal robustness with far fewer exaggerated refusals than conventional fine-tuning or retraining (Section 4.2; Tables 3–4).

- Practical applications
  - Rapid skill deployment: Add new domain skills (e.g., medicine, law, coding languages) by training and merging corresponding task vectors.
  - Safety patching: Deploy “safety vectors” that increase refusal rates for harmful prompts without breaking benign interactions (high XSTest Safe).
  - On-device or air-gapped scenarios: Avoid full retraining pipelines and keep update costs low.

- Follow-up research enabled
  - Interference-aware merging for instruction tuning:
    - Develop better algorithms or constraints that preserve orthogonality or disentanglement across task vectors.
  - Data-free generality preservation:
    - Figure 2 suggests that judicious inclusion of general data during WiSE-FT helps. When `G` is unavailable, can synthetic or proxy general data stabilize general skills?
  - Automated `ω` selection:
    - Learn to predict good `ω` from data characteristics or early training signals; or optimize `ω` per layer or module rather than globally.
  - Scaling studies:
    - Test PTM across larger models and a wider range of skills; evaluate interactions with RLHF and other post-SFT processes.
  - Safety compositionality:
    - Explore multiple safety vectors (toxicity, bio, misinformation) and their interactions with domain skills.

In short, Sections 4.1–4.4 and Tables 2–5 collectively show that PTM is a practical, compute-efficient way to add skills to instruction-tuned LMs while preserving general ability, and it is especially strong for safety due to its substantially lower rate of exaggerated refusals. The methodology and heuristics (Eqs. 1–7, Figure 1) form a clear recipe practitioners can adopt today, while the interference findings outline a concrete research agenda for robust multi-skill composition.
