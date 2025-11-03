# Training language models to follow instructions with human feedback

**ArXiv:** [2203.02155](https://arxiv.org/abs/2203.02155)

## 🎯 Pitch

This paper introduces InstructGPT, a family of large language models fine-tuned with reinforcement learning from human feedback (RLHF) to reliably follow user instructions across a wide range of natural-language tasks. By leveraging human preferences to guide model behavior, the authors show that even much smaller InstructGPT models are strongly preferred by human evaluators over much larger standard GPT-3 models, and also produce fewer hallucinations and toxic outputs—advancing the alignment of language models with actual user needs. This approach demonstrates a scalable path toward safer, more useful, and more trustworthy AI systems that can be robustly deployed in real-world settings.

---

## 1. Executive Summary
This paper shows how to make large language models reliably follow natural-language instructions by fine-tuning them with human feedback (RLHF). The resulting models, called `InstructGPT`, are strongly preferred by human evaluators over much larger vanilla GPT‑3 models, reduce hallucinations and toxicity, and retain most benchmark performance—especially after a simple fix that mixes in pretraining gradients during RL (Figure 1, Figures 28–29, Equation (2)).

## 2. Context and Motivation
- Problem addressed
  - Modern language models are trained to predict the next token on internet text, not to “follow a user’s instructions helpfully and safely.” This objective misalignment leads to untruthful, toxic, or unhelpful outputs (Section 1).
  - The paper targets alignment to user intent across a broad, real-world distribution of tasks (generation, brainstorming, QA, rewriting, chat, summarization, etc.; Table 1).

- Why it matters
  - Real-world deployments require models that are helpful, honest, and harmless (Section 1; Section 3.6). Misaligned outputs degrade user experience and may cause harm (e.g., toxicity), limiting adoption.

- Prior approaches and their gaps
  - Prompt engineering and few-shot prompting help but are brittle (Figure 1, “GPT (prompted)”).
  - Multi-task instruction fine-tuning on public datasets (e.g., FLAN, T0) improves generalization to benchmark tasks but underperforms on real API prompts dominated by open-ended generation (Figure 5; Section 4.1).
  - RLHF existed but was demonstrated on narrow tasks like summarization (related work, Section 2), not across the diverse, messy instructions found in production settings.

- Positioning
  - The paper adapts and scales RLHF to a broad instruction-following setting, showing: (1) large preference gains on held-out customer prompts, (2) improved truthfulness and (some) toxicity reduction, and (3) a simple technique to mitigate benchmark regressions (“alignment tax”) that appear during RLHF (Section 4).

## 3. Technical Approach
The method has three stages (Figure 2), all starting from GPT‑3 architectures (Section 3.5).

Terminology (defined on first use):
- `Supervised Fine-Tuning (SFT)`: standard fine-tuning on human-written demonstrations (labeled correct outputs).
- `Reward Model (RM)`: a model that scores a prompt–response pair so that higher scores reflect human preference.
- `Reinforcement Learning from Human Feedback (RLHF)`: uses an RM as a reward for policy optimization; here optimized with `Proximal Policy Optimization (PPO)`, a stable RL algorithm.
- `KL penalty`: a regularizer that discourages the RL policy from drifting too far from a reference model by penalizing divergence (Section 3.5).
- `Alignment tax`: capability regressions on public benchmarks that appear after aligning behavior to human preferences (Section 1 and 4.2).
- `PPO-ptx`: PPO with an added pretraining loss term (“ptx” for pretraining mix) that mitigates the alignment tax (Equation (2)).

Step-by-step pipeline
1) Step 1 — Supervised Fine-Tuning (SFT)
   - Data: 13k prompts with demonstrations from API prompts and labeler-written prompts (Table 6). Prompts are deduplicated, split by user ID, and PII-filtered (Section 3.2).
   - Task coverage: generation, brainstorming, QA, chat, rewrite, summarization, etc. (Table 1; Section 3.3).
   - Training: 16 epochs with cosine LR decay; model selection by RM score on validation set (Section 3.5).

2) Step 2 — Reward Model (RM) training
   - Data: 33k prompts where labelers ranked K model outputs per prompt (K=4–9). This yields up to K(K−1)/2 comparisons per prompt (Section 3.5).
   - Loss: Pairwise Bradley–Terry style logistic loss over all pairs within a prompt batch (Equation (1)). Training on the whole set of within-prompt pairs in one batch prevents overfitting that occurs if pairs are shuffled independently (Section 3.5).
   - Normalization: RM bias is set so demonstrations have mean reward 0 before RL (Section 3.5).
   - Practical choice: a single 6B RM used for all PPO policy sizes to ensure stability and reduce compute; 175B RMs were less stable (Appendix C.2).

3) Step 3 — RLHF via PPO
   - Environment: a bandit—sample a prompt, generate a response, get scalar reward from RM, terminate (Section 3.5).
   - Objective: maximize RM reward minus a per-token KL penalty from the SFT policy (β controls strength) (Section 3.5).
   - Alignment tax mitigation: mix in pretraining gradients during PPO, producing `PPO-ptx`. The combined objective (Equation (2)) is:
     - maximize E[rθ(x,y) − β·KL(πRL||πSFT)] + γ·Epretrain[log πRL(x)],
       where γ controls the strength of the pretraining gradient mix (Section 3.5).
   - Value function: initialized from the RM; PPO hyperparameters in Appendix C.4.

Human data and evaluation setup
- Labelers: ~40 trained contractors, selected for sensitivity to harmful content and quality on a screening test; instructions evolve with feedback (Sections 3.4, B.1–B.2).
- Evaluation: preference comparisons, Likert ratings, and metadata (truthfulness, constraint satisfaction, hallucination, toxicity signals) per Table 3 and Section 3.6.
- Datasets: three splits tied to API customers (train/val/test by user ID), with a separate test for prompts originally written for GPT models (Figure 3 left) vs prompts addressed to instruction models (Figure 3 right).

Why these design choices?
- Multi-output ranking per prompt (K>2) yields more training signal per labeling action and reduces RM overfitting when trained per-prompt (Section 3.5).
- KL penalty stabilizes RL against reward hacking by tethering to SFT behavior (Section 3.5).
- Pretraining mix (PPO-ptx) preserves broad language competence learned during pretraining, reducing benchmark regressions without hurting human preference scores (Sections 1, 4.2; Figures 28–29, 33–34).

## 4. Key Insights and Innovations
1) Broad-coverage RLHF works—and beats scale alone
   - Insight: Human-aligned behavior on real instructions can exceed raw capability gains from scaling.
   - Evidence: On held-out API prompts, outputs from a 1.3B `PPO-ptx` model are preferred over a 175B GPT‑3 baseline (Figure 1). At 175B, `PPO-ptx` wins 85 ± 3% head-to-head vs GPT‑3 and 71 ± 4% vs carefully prompted GPT‑3 (Section 1; Figure 1).

2) A simple RM training recipe that prevents overfitting
   - Innovation: Aggregate all pairwise comparisons from each ranked set in a single batch (Equation (1); Section 3.5). This avoids the overfitting seen when each pair is treated as an independent datapoint and reduces compute (one forward pass per completion, not per pair).

3) Eliminating the alignment tax with a pretraining mix
   - Problem: PPO improves alignment but initially harms benchmark performance (DROP, SQuADv2, translation) (Figures 28–29).
   - Solution: Add pretraining gradients during PPO (`PPO-ptx`, Equation (2)). Figure 33 shows a range of γ values that significantly recovers SQuADv2 and DROP while maintaining validation reward; Figure 34 shows that increasing only the KL penalty cannot fully fix regressions.

4) Preference generalization beyond training labelers
   - RMs trained on 4/5 labeler groups achieve 69.6 ± 0.9% accuracy at predicting held-out labelers’ preferences vs 72.4 ± 0.4% on training labelers (Appendix E.2), and held-out labelers’ preferences mirror the preference gains seen with training labelers (Figure 3).

Incremental vs fundamental
- Fundamental: showing that broad, real-world instruction following can be effectively aligned with RLHF and outperform “just scale” and multi-task instruction fine-tuning (Figures 1 and 5).
- Incremental but important: the RM batching trick and the pretraining mix are simple implementation choices with outsized practical impact (Section 3.5; Figures 28–29, 33–34).

## 5. Experimental Analysis
Evaluation methodology
- Data
  - Real prompts from the OpenAI API “Playground,” split by customers so test sets use unseen users (Section 3.2). SFT: ~13k prompts; RM: ~33k prompts; PPO: ~31k prompts (Table 6).
  - Distributions: prompts sent to GPT models (“GPT distribution”) vs to early InstructGPT models (“Instruct distribution”) (Figure 3).
- Human evaluations
  - Main metric: win rate against a strong 175B SFT baseline (Figure 1).
  - Additional: 1–7 Likert quality scores (Figure 31) and metadata labels such as “follows constraints,” “hallucinations,” “appropriateness” (Table 3; Figure 4).
  - Held-out labelers to test generalization (Figure 3 top).
- Automatic benchmarks
  - Truthfulness (TruthfulQA; Figure 6), toxicity (RealToxicityPrompts; Figure 7, Figure 39), bias (Winogender, CrowS-Pairs; Figure 32), and standard NLP tasks (DROP, QuAC, SQuADv2, HellaSwag, SST, RTE, WSC, WMT Fr→En, CNN/DM, Reddit TL;DR; Figures 28–29; Table 14).

Main quantitative results
- Preference on real prompts
  - > “Outputs from our 175B InstructGPT are preferred to 175B GPT‑3 in 85 ± 3% of comparisons; to few-shot GPT‑3 in 71 ± 4%” (Figure 1).
  - Even 1.3B `PPO-ptx` beats 175B GPT‑3 in head-to-head preference (Figure 1).
  - Results hold on both GPT and Instruct distributions and for held-out labelers (Figure 3).
- Concrete behavior improvements (metadata)
  - Fewer hallucinations on closed-domain tasks: ~21% vs ~41% for GPT‑3, collapsed across sizes (Figure 4).
  - Better at following explicit constraints and attempting the correct instruction; more appropriate for a customer assistant (Figure 4; definitions in Table 3).
- Truthfulness (TruthfulQA, human-evaluated)
  - `PPO` and `PPO-ptx` increase the rate of truthful and truthful+informative answers (Figure 6). With an instruction that encourages “I have no comment” when unsure, aligned models prefer truthful-but-cautious responses (Figure 6 right).
- Toxicity (RealToxicityPrompts)
  - With a respectful instruction prefix, aligned models generate less toxic outputs than GPT‑3 by both Perspective API and human ratings (Figure 7; Figure 40).
  - Without the instruction, toxicity is similar; when explicitly instructed to be biased, aligned models produce very toxic outputs—even at low input toxicity (Figure 39).
- Public NLP datasets
  - Plain PPO initially regresses on DROP, SQuADv2, and translation; `PPO-ptx` recovers most losses and even exceeds GPT‑3 on HellaSwag (Figures 28–29).
  - Sweeps show that increasing `γ` (pretraining mix) recovers benchmarks with small validation reward drop (Figure 33), whereas increasing only the KL penalty does not (Figure 34).
- Baselines FLAN/T0
  - On the real API prompt distribution, `InstructGPT` decisively outperforms FLAN and T0; Likert scores for FLAN/T0 are close to few-shot GPT‑3 and below SFT/RLHF (Figure 5). Head-to-head, 175B `InstructGPT` wins 78 ± 4% vs FLAN and 79 ± 4% vs T0 (Section 4.1).

Ablations and robustness checks
- RM generalization to new labelers (Appendix E.2).
- Effect of `γ` (pretraining mix coefficient) and KL coefficient (Figures 33–34).
- Training longer can reintroduce regressions on DROP and SQuADv2 (Figure 35).
- PPO learning-rate and batch-size sweeps (Appendix E.9–E.11).
- Qualitative generalization to code and non‑English is observed despite minimal supervision (Figure 8; Section 4.3).

Failure cases and caveats
- Hedging and false-premise compliance: the model sometimes goes along with incorrect premises and can over‑hedge instead of giving a straightforward answer (Figure 9; Section 4.3).
- Toxicity control depends on prompting: benefits are prominent when a respectful instruction is present (Figure 7; Figure 39).

Do the experiments support the claims?
- Yes, for alignment to user intent on real prompts: strong, replicated human preferences with consistent metadata gains (Figures 1, 3–4, 31).
- Yes, for truthfulness improvements and conditional toxicity reduction (Figures 6–7, 39).
- The alignment tax and its mitigation are convincingly analyzed with ablations (Figures 28–29, 33–34).

## 6. Limitations and Trade-offs
Assumptions and scope
- Alignment target is a specific reference group
  - Behavior is shaped by the hired labelers’ preferences, the instruction guidelines, and the customer prompt distribution (Section 5.2). Inter-labeler agreement is ~73% (Section 3.4), indicating genuine preference diversity that a single model can only average over.

- Helpfulness prioritized during training
  - During data collection, helpfulness was prioritized over harmlessness/truthfulness; final evaluations prioritized the latter (Section 3.4, B.2). A direct side effect: when instructed to be biased or harmful, the model often complies (Figure 39; Section 5.3).

- Toxicity and bias are only partially addressed
  - Toxicity reductions depend on respectful prompting and do not persist under adversarial instructions (Figure 39). Bias measurements (Winogender, CrowS-Pairs) show no significant improvements and sometimes lower entropy (higher certainty) under respectful prompting (Figure 32).

Computational and data constraints
- RLHF adds nontrivial training complexity (SFT, RM, PPO), though compute is a small fraction of GPT‑3 pretraining (Section 5.1, point 1). High-quality human data collection remains a bottleneck (Sections 3.4, 5.3).

Residual performance trade-offs
- Without pretraining mix, PPO regresses on some benchmarks (Figures 28–29). Even with `PPO-ptx`, some gaps remain (e.g., DROP, SQuADv2, WMT Fr→En; Figure 29).

Open questions
- How to align to pluralistic or group-conditional values (Section 5.2).
- How to train models to refuse harmful requests without over-refusal (Sections 5.3–5.4).
- How to systematically reduce hedging and false-premise acceptance (Figure 9; Section 4.3).

## 7. Implications and Future Directions
Field impact
- Demonstrates that broad RLHF can outperform sheer scale and curated multi-task instruction training on real user tasks (Figure 1 vs Figure 5). This helped establish instruction tuning with human feedback as a standard step in modern LLM development.

Practical applications
- Safer, more controllable assistants for ideation, rewriting, summarization, and QA with improved faithfulness and constraint-following (Figure 4; Section 4.1).
- Deployable systems that better match user expectations in diverse applications while maintaining benchmark competence via `PPO-ptx` (Figures 28–29).

Follow-up research directions
- Data collection
  - Adversarial data gathering to reduce failure modes (false-premise compliance, hedging) and stress-test safety (Section 5.4; Figure 9).
  - Group-conditional alignment and preference conditioning so different communities can instantiate different values (Section 5.2).
  - Richer feedback channels: edits and natural-language critiques, not just rankings (Section 5.4).

- Algorithms
  - Better objectives that explicitly trade off helpfulness, honesty, and harmlessness, including principled refusal policies configurable by application context (Sections 3.6, 5.4).
  - Alternatives to PPO and pairwise RMs (e.g., expert iteration, constrained optimization, critique-based training) (Section 5.4).
  - Further reducing alignment tax with improved mixes and filtered pretraining distributions (Section 5.4; Figures 33–34).

- Evaluation and governance
  - Broader, participatory alignment processes and accountability mechanisms; transparency about whose preferences are encoded (Section 5.2).
  - Application-specific safety audits—since toxicity/bias harms depend on deployment context (Section 3.6; Section 5.5).

> Core takeaway: Fine-tuning large language models with human preference signals across a real, multi-task instruction distribution yields dramatic gains in user-perceived quality (Figure 1), measurable improvements in truthfulness and instruction adherence (Figures 4 and 6), and—with a simple pretraining-mix fix—retains strong performance on public NLP benchmarks (Figures 28–29, 33).
