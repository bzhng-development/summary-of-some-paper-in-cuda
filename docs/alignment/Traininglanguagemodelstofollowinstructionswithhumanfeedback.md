# Training language models to follow instructions with human feedback

**ArXiv:** [2203.02155](https://arxiv.org/abs/2203.02155)

## 🎯 Pitch

This paper presents InstructGPT, a family of large language models aligned to follow user instructions by fine-tuning GPT-3 with human feedback using reinforcement learning (RLHF) on real-world prompts. InstructGPT models are rated as more helpful, truthful, and less toxic than much larger GPT-3 models, marking a major advance in making language models practically useful, safer, and more controllable across broad tasks—setting a new paradigm for aligning AI systems with human intent.

---

## 1. Executive Summary (2–3 sentences)
This paper shows how to align large language models with user intent by fine‑tuning them using human feedback (reinforcement learning from human feedback, RLHF). The resulting models, called `InstructGPT`, produce responses that human evaluators prefer over those from much larger GPT‑3 models, improve truthfulness and reduce toxic output on several measures, and—when combined with a simple training modification—avoid major regressions on standard NLP benchmarks (Figures 1, 3, 4, 6, 7; Section 4.1–4.2).

## 2. Context and Motivation
- Problem/gap:
  - Pretrained language models are optimized to predict the next token on internet text, not to “follow a user’s instructions helpfully and safely.” This mismatch leads to unhelpful, untruthful, or toxic outputs (Introduction; Section 1).
  - Prior RLHF work focused on narrow tasks like summarization (Section 2), leaving open whether instruction-following could be aligned across a broad, real‑world distribution of tasks.
- Real‑world importance:
  - Aligned models are easier to deploy in applications without extensive prompt engineering, reduce risk of harmful outputs, and better match user expectations (Section 1; Table 3 for safety metadata tracked).
- Prior approaches and their limits:
  - Prompting or few‑shot learning improves instruction following but is brittle and still misaligned (Figure 1 “GPT (prompted)”; Section 4.1).
  - Multi‑task fine‑tuning on public NLP datasets (e.g., FLAN, T0) boosts zero‑shot transfer but underperforms on real API prompts dominated by open‑ended generation and brainstorming (Figure 5; Table 1).
- Positioning:
  - The paper scales RLHF to a wide distribution of instruction‑like prompts sourced from real API users, and compares against strong baselines including few‑shot GPT‑3, SFT‑only models, and FLAN/T0 (Sections 3.2, 3.5; Figures 1, 3, 5).

## 3. Technical Approach
The method has three stages (Figure 2; Section 3.1). Terms used below:
- `SFT` (Supervised Fine‑Tuning): standard fine‑tuning on input–output demonstrations.
- `Reward model` (`RM`): a model that maps (prompt, completion) to a scalar score trained to predict human preferences.
- `RLHF` with `PPO`: optimize the policy (the language model) via Proximal Policy Optimization using the RM as the reward.
- `KL penalty`: a per‑token penalty that discourages the policy from drifting too far from the SFT model.
- `PPO-ptx`: PPO training mixed with gradients from the original pretraining distribution to mitigate benchmark regressions (Equation (2); Section 3.5).

Step‑by‑step:
1) Data and labeling pipeline (Sections 3.2–3.4; Appendix A, B)
   - Prompt sources:
     - Real prompts submitted to early InstructGPT models via the OpenAI API Playground (with user consent, deduplication, PII filtering; train/val/test split by user ID).
     - Labeler‑written prompts used to bootstrap the first SFT models.
   - Datasets (Table 6):
     - `SFT` train: ~13k prompts (11,295 labeler; 1,430 customer), plus validation.
     - `RM` train: ~33k prompts with 4–9 model responses per prompt ranked by labelers (generates up to K(K−1)/2 pairwise comparisons per prompt).
     - `PPO` train: ~31k customer prompts for RL rollouts; separate validation.
   - Labelers:
     - ~40 trained contractors selected via screening for sensitivity to harmful content, preference agreement, and ability to write nuanced demos (Section 3.4; Appendix B.1).
     - Inter‑annotator agreement: training labelers 72.6 ± 1.5%; held‑out labelers 77.3 ± 1.3% (Section 3.4).
   - Task diversity:
     - Prompts span 10 use‑case categories; majority are open‑ended generation/brainstorming (Table 1; examples in Tables 2 and Appendix A.2). Dataset is >96% English (Appendix A.4).

2) Stage 1 — Supervised Fine‑Tuning (`SFT`) (Section 3.5 “Supervised fine-tuning”)
   - Fine‑tune GPT‑3 on labeler demonstrations for 16 epochs (cosine LR decay; residual dropout 0.2). Although validation loss overfits after ~1 epoch, continuing training improves human‑aligned metrics (reward model score and human preferences).

3) Stage 2 — Reward Model (`RM`) training (Section 3.5 “Reward modeling”; Equation (1))
   - Input: prompt x and completion y; output: scalar reward rθ(x, y).
   - Labelers rank K samples per prompt; training uses all pairwise comparisons from these rankings but computed efficiently:
     - Instead of treating each pair as a separate example (which overfits because pairs share the same completions), all K completions for a prompt are processed in one batch, and the loss aggregates the pairwise Bradley‑Terry style comparisons:
       - Equation (1): minimize −E[log σ(rθ(x, y_w) − rθ(x, y_l))], where y_w is the preferred completion, y_l the less preferred one.
   - Practical choices:
     - Use a 6B RM (not 175B) for stability and compute efficiency; 175B RM training was less stable and too costly to use as a value function during RL (Appendix C.2).
     - Normalize RM scores so labeler demonstrations have mean 0 (Section 3.5).

4) Stage 3 — RLHF with PPO (Section 3.5 “Reinforcement learning (RL)”)
   - Environment: contextual bandit—sample a prompt, the policy generates a completion, the RM returns a scalar reward; the episode ends (Section 3.5).
   - Objective (Equation (2)):
     - Maximize E[rθ(x, y)] minus a `β` KL penalty from the SFT policy (to prevent over‑optimization of the RM and preserve language quality), plus an optional pretraining log‑likelihood term with weight `γ` (`PPO-ptx`) that mixes in gradients from the original pretraining distribution:
       - objective(φ) = E[rθ(x, y) − β log(πRL(y|x)/πSFT(y|x))] + γ Ex∼Dpretrain log πRL(x)
   - Training details (Appendix C.4):
     - Initialize the RL policy from the SFT model (with 10% pretraining mix in SFT init; Appendix C.3).
     - Train 256k episodes over ~31k unique, filtered prompts; batch size 512; PPO clip 0.2; sampling temperature 1.
     - Value function initialized from the same 6B RM.
     - Example hyperparameters: β = 0.02; γ tuned (≈27.8 worked well and mitigated regressions; Figure 33).

5) Baselines (Section 3.5)
   - `GPT‑3` (no instruction prefix).
   - `GPT‑3 (prompted)`: a hand‑crafted instruction‑following prefix found via an internal “prefix competition.”
   - `SFT`: supervised fine‑tuned on demonstrations.
   - `FLAN` and `T0`: 175B GPT‑3 fine‑tuned on public instruction datasets; checkpoints selected by highest RM score (Appendix C.5; Figure 13).

Why these design choices?
- Preference‑trained RM gives a task‑general reward signal aligned with labelers’ judgments (Figure 2; Section 3.1).
- KL penalty avoids pathological over‑optimization of the RM; mixing in pretraining (`PPO-ptx`) retains general NLP capabilities (Figures 28–29, 33–34).
- Training the RM and value function at 6B improves stability and reduces compute (Appendix C.2, C.4).

## 4. Key Insights and Innovations
1) Broad‑distribution RLHF for instruction following
   - Innovation: Scale RLHF beyond a single task to the diverse, real‑world prompt distribution from API users (Section 3.2; Table 1).
   - Significance: Even a 1.3B `InstructGPT` is preferred to a 175B GPT‑3 on these prompts (Figure 1), demonstrating that alignment beats brute model scaling for usability.

2) Efficient RM training from rankings
   - Innovation: Train the RM using all K ranked completions per prompt in a single batch to avoid overfitting from correlated pairwise comparisons and to save compute (Section 3.5; Equation (1)).
   - Significance: Higher validation accuracy and better use of human data enable reliable PPO training.

3) `PPO-ptx`: Mixing pretraining updates during RL
   - Innovation: Add a pretraining‑likelihood term during PPO to mitigate regressions (“alignment tax”) on standard NLP benchmarks (Equation (2); Figures 28–29, 33–34).
   - Significance: Recovers much of the lost performance (and even surpasses GPT‑3 on HellaSwag in some settings; Section 4.2) while maintaining preference gains.

4) Real‑world evaluation design
   - Innovation: Evaluate on a held‑out slice of real API prompts with human preferences, Likert quality ratings, and fine‑grained safety/quality metadata (Section 3.6; Table 3).
   - Significance: Shows generalization to held‑out labelers and highlights concrete behavioral improvements (Figures 3–4).

## 5. Experimental Analysis
Evaluation methodology (Section 3.6):
- Two regimes:
  1) Real API prompts (“InstructGPT distribution”) and prompts crafted for GPT‑3 (“GPT distribution”); test users held out by organization ID (Sections 3.2, 4.1; Figure 3).
  2) Public NLP benchmarks for safety (toxicity, bias, truthfulness) and capabilities (QA, reading comprehension, summarization, translation; Appendix D; Figures 28–29; Table 14).
- Metrics:
  - Human pairwise preference win‑rate vs a 175B SFT baseline (Figures 1, 3).
  - Per‑output Likert quality (1–7) and metadata (e.g., instruction following, hallucination, safety flags; Figure 4; Table 3).
  - Automatic task metrics (e.g., F1, accuracy, BLEU, ROUGE; Table 14; Appendix D).
  - TruthfulQA human evaluation (true; true+informative; Figure 6).
  - RealToxicityPrompts via Perspective API and human toxicity/continuity ratings (Figures 7, 39–41).
  - Bias via entropy over balanced choices (higher entropy = less bias; Winogender and CrowS‑Pairs; Figure 32; Table 14).

Main quantitative results:
- Preference and quality on real prompts
  - Strong preference for `InstructGPT`:
    - > “175B `InstructGPT` is preferred to 175B GPT‑3 85 ± 3% of the time, and 71 ± 4% vs few‑shot GPT‑3” (Section 1; Figure 1).
    - Even 1.3B `InstructGPT` beats 175B GPT‑3 (Figure 1).
    - Results hold on both “Instruct” and “GPT” prompt distributions and for held‑out labelers (Figure 3).
  - Concrete behavioral improvements (Figure 4):
    - Fewer hallucinations on closed‑domain tasks; better adherence to explicit constraints; higher rate of attempting the correct instruction; more appropriate as a customer assistant.
    - The paper quantifies hallucination decreases by “about half” (21% vs 41% on closed‑domain tasks; Section 1 bullets).

- Comparison to FLAN/T0 (public instruction datasets)
  - FLAN/T0 improve over default GPT‑3 but underperform SFT and are far behind `InstructGPT` on API prompts:
    - Likert scores show `InstructGPT` highest; FLAN/T0 near “GPT‑3 (prompted)” (Figure 5).
    - Head‑to‑head: 175B `InstructGPT` wins 78 ± 4% vs FLAN, 79 ± 4% vs T0 (Section 4.1).

- Truthfulness (TruthfulQA; Figure 6)
  - PPO models generate more truthful and informative answers; with an explicit “helpful instruction” prompt, RLHF models “err on the side of being truthful and uninformative rather than confidently saying a falsehood,” whereas GPT‑3 does not (Section 4.2).
  - Caveat: the 1.3B `PPO-ptx` slightly underperforms GPT‑3 at that size on some TruthfulQA settings (Figure 6).

- Toxicity (RealToxicityPrompts; Figures 7, 39–41; Table 14)
  - With a respectful instruction, RLHF models output less toxic text than GPT‑3 (both Perspective API and human ratings; Figure 7).
  - Without the instruction, the advantage largely disappears; with an explicitly biased instruction, RLHF models produce very toxic text (Figure 39).
  - Human study: SFT is least toxic but lowest continuity/preference—suggesting overly terse or degenerate responses (Figure 40).

- Bias (Winogender, CrowS‑Pairs; Figure 32; Table 14)
  - No significant reduction in measured social bias. Instructed models can show lower entropy (higher certainty) regardless of stereotype direction, complicating interpretation (Section 4.2).

- Public NLP benchmarks (Figures 28–29; 33–35; Table 14)
  - Plain PPO suffers an “alignment tax” (regressions on SQuADv2, DROP, HellaSwag, WMT’15 Fr→En).
  - `PPO-ptx` (γ > ~20) largely recovers performance (Figure 33), even surpassing GPT‑3 on HellaSwag in few‑shot at 175B (Figure 29), while preserving validation reward; increasing only the KL coefficient cannot fully fix regressions (Figure 34).
  - Training too long (512k episodes) can reintroduce regressions (Figure 35).

- Generalization and failure cases
  - Generalization: examples show instruction following in other languages and on code questions, despite such data being rare in fine‑tuning (Figure 8; Section 4.3).
  - Failure cases:
    - Accepts false premises; overly hedges; struggles with prompts containing multiple hard constraints (Figure 9; Section 4.3).

Ablations/robustness:
- Held‑out labelers: preferences generalize (Figure 3 top); RM predicts held‑out labeler preferences at 69.6 ± 0.9% vs 72.4 ± 0.4% within‑group (Section 4.1; Appendix E.2).
- Hyperparameter studies: best KL ≈ 0.01–0.02 (Figure 36); learning‑rate sensitivity and pretraining mix evaluated (Figure 38; Appendix E.8–E.11).

Assessment:
- The experiments are broad, use both human and automatic metrics, and include useful ablations. The strongest claim—human preference dominance on real prompts—is well supported (Figures 1, 3). Safety results are nuanced: improvements depend on giving the model a respectful instruction; bias remains unresolved (Figures 7, 32, 39–41). Capability regressions are documented and mitigated via `PPO-ptx` (Figures 28–29, 33–35).

## 6. Limitations and Trade-offs
- What is being aligned, and to whom? (Section 5.2)
  - Alignment is to the preferences of a specific labeler pool, under instructions authored by the research team, and on prompts coming from OpenAI API users. This is not a universal notion of “human values.”
  - Labelers are mostly English‑speaking and from the US or Southeast Asia; dataset is >96% English (Appendix A.4; B.3). Preferences and judgments may not represent broader populations or domain‑expert contexts.

- Safety trade‑offs (Sections 3.4, 5.3–5.4)
  - During training, helpfulness had priority over truthfulness/harmlessness; in evaluation, truthfulness/harmlessness were emphasized (Section 3.4; B.2). This can produce models that comply even with harmful instructions (Figure 39), or that hedge too much when labelers reward epistemic humility (Figure 9).

- Residual harms
  - Bias not substantially reduced on Winogender or CrowS‑Pairs (Figure 32); toxicity improvement hinges on the presence of a respectful prompt (Figure 7).

- Alignment tax and mitigation
  - Plain PPO harms some benchmark performance (Figures 28–29). `PPO-ptx` mitigates but may not fully eliminate regressions (Section 4.2).

- Data and compute constraints
  - Human data is costly; RM and RL training still require substantial compute (though far less than pretraining; Section 5.1, point 1).
  - RM uses 6B for stability; 175B RM training was unstable (Appendix C.2).

- Edge cases not addressed
  - Little multilingual supervision and no domain‑expert pipelines; models can follow prompts with false premises; complex constraint satisfaction remains fragile (Section 4.3).
  - Refusal behaviors and context‑sensitive safety policies are not the primary focus here (Section 5.4).

## 7. Implications and Future Directions
- How this changes the landscape (Section 5.1)
  - RLHF at scale provides a practical path to align models with user intent across diverse tasks, delivering large usability gains without scaling parameter count. The alignment tax can be made small with `PPO-ptx`, making adoption more attractive.

- What it enables
  - General instruction‑following assistants usable with minimal prompt engineering; safer defaults when paired with appropriate system prompts; potential to condition on different communities’ preferences (Section 5.2).

- Promising research directions (Sections 5.4, 5.2)
  - Safety:
    - Train refusal behaviors and context‑dependent safety policies; adversarial data collection to fix failure modes (e.g., false premises, jailbreaks).
    - Combine RLHF with pretraining data filtration or truthfulness tools (e.g., web‑augmented QA).
  - Reward modeling:
    - Use richer feedback (edits, critiques), better interfaces, and methods beyond pairwise comparison; study how instructions to labelers shape collected signals.
  - Preference pluralism:
    - Condition models on different groups’ preferences; develop accountable processes for whose values are encoded and how they are represented (Section 5.2).
  - Technique improvements:
    - Explore alternative policy‑optimization methods (expert iteration, constrained optimization), better KL regularization, and more principled pretraining‑mix strategies.
  - Evaluation:
    - Broaden multilingual and domain‑expert tests; develop measures for honesty beyond truthfulness and calibrate hedging behavior.

> Core takeaway: Fine‑tuning with human feedback substantially improves instruction following, truthfulness, and perceived helpfulness—even allowing a 1.3B model to beat a 175B one in human preference—while a simple training modification (`PPO-ptx`) keeps most general NLP capabilities intact (Figures 1, 3, 4, 6, 28–29, 33).
