# Model Merging in Pre‑training of Large Language Models

**ArXiv:** [2505.12082](https://arxiv.org/abs/2505.12082)
**Authors:** Changxin Tian, Jiapeng Wang, Kunlong Chen, Ziqi Liu, Jiaxin Mao, Wayne Xin Zhao, Zhiqiang Zhang, Jun Zhou
**Institutions:** 

## 🎯 Pitch

This paper presents Pre-trained Model Averaging (PMA), a method to merge multiple pre-training checkpoints in large language models, simulating annealed performance without costly learning-rate adjustments. By leveraging checkpoint averaging during stable training phases, PMA allows for faster iterations and reduced compute costs, significantly impacting model development by improving performance and stability while offering theoretical insights into weight averaging.

---

## 1. Executive Summary (2–3 sentences)
This paper introduces Pre-trained Model Averaging (`PMA`), a simple but carefully studied way to merge multiple pre‑training checkpoints into a single set of weights for large language models (LLMs). Across dense and Mixture‑of‑Experts (MoE) models up to 100B+ parameters, merging checkpoints saved during the constant‑learning‑rate “stable” phase consistently improves downstream performance and, crucially, lets practitioners approximate the effects of later learning‑rate annealing without doing the costly annealing runs (Section 4.1; Figures 1–3).

## 2. Context and Motivation
- Problem addressed
  - Large‑scale LLM pre‑training is expensive and slow; practitioners often use a Warmup‑Stable‑Decay (`WSD`) learning‑rate schedule with a long constant‑LR “stable” phase followed by a cosine “decay/annealing” phase (Section 3). Validating whether a model will improve after annealing requires running the annealing, which is costly and delays iteration.
  - Prior “model merging” methods mostly target post‑training: combining models fine‑tuned on different tasks into one versatile model (Section 2; references [18, 46, 51, 57]). Pre‑training checkpoint merging has been studied far less and not at the scale or breadth evaluated here.

- Why this matters
  - Real‑world impact: If one can predict annealed performance by merging stable‑phase checkpoints, teams can shorten validation cycles and reduce compute costs during pre‑training. Stability improvements from merged initializations could also reduce failed runs and the need for restarts (Sections 4.4–4.5).
  - Theoretical significance: Understanding when and why weight averaging reduces loss clarifies the geometry of the loss landscape for LLMs (Section 4.6, Equations 6–15).

- Prior approaches and their gaps
  - Post‑training merging (e.g., DARE, Fisher merging, Task Arithmetic) shows that weights from different specialized models can be combined (Section 2). But these methods do not answer: Can sequential checkpoints along a single pre‑training trajectory be merged to improve generalization and simulate annealing?
  - Limited pre‑training merging work exists (LAWA, Early Weight Averaging), but large‑scale evidence and detailed ablations across model sizes, MoE architectures, and scheduler phases were lacking (Section 2).

- Positioning
  - This work systematically studies pre‑training merges across sizes and architectures—from dense 411M–70B and MoE with 0.7B/7B to 20B/200B activated/total parameters—under WSD schedules, and provides practical guidance (Sections 3–4; Appendix A).

Definitions used once and then assumed:
- `Checkpoint`: a saved copy of model weights at a specific training step or token count.
- `WSD schedule`: learning rate warmup, long constant stable phase, then decay (annealing) via cosine schedule (Section 3).
- `MoE`: Mixture‑of‑Experts, a model with many expert subnetworks; only a subset (“activated parameters”) runs per input.
- `Annealing`: gradually reducing the learning rate near the end of training.
- `GradNorm`: the norm of gradients computed during backprop; spikes indicate instability.

## 3. Technical Approach
The central idea is to average multiple checkpoints from a single training trajectory to form a merged model that performs better and can approximate later annealed performance—especially when the checkpoints come from the “stable” constant‑LR phase.

Step by step
1. Save checkpoints at fixed token intervals during pre‑training.
   - Let `M_i` be the weights of the `i`‑th checkpoint, and `T_i` be the cumulative tokens seen (Section 3).
   - Define the interval between checkpoints as `V = T_{i+1} – T_i` (Equation 2).

2. Merge N consecutive checkpoints with weights `w_i`:
   - General form: `M_avg = sum_{i=1..N} (w_i * M_i)` (Equation 1).
   - Variants for `w_i` (Section 3):
     - `SMA` (Simple Moving Average): equal weights, `w_i = 1/N`. Formula: `M_avg = (1/N) * sum M_i` (Equation 3).
     - `EMA` (Exponential Moving Average): emphasize later checkpoints; recursive `M_avg^(i) = α M_i + (1 – α) M_avg^(i–1)` (Equation 4).
     - `WMA` (Weighted Moving Average): linearly increasing weights (e.g., `w_i = i`, normalized) (Equation 5).

3. Where and when to merge
   - Primary regime: merge during the stable constant‑LR phase (Section 4.1; Figure 1). These checkpoints tend to explore a “flat” region of the loss landscape; averaging moves the solution toward better generalization.
   - Secondary regime: merging early in the cosine annealing phase matches or can exceed the eventual end‑of‑annealing performance (Section 4.1; Figure 2).
   - Practical shortcut: continue training with constant LR and periodically merge checkpoints—this can track the annealed model’s performance, enabling earlier validation and potentially skipping full annealing (Section 4.1; Figure 3).

4. Choosing hyperparameters
   - `Interval V` (tokens between consecutive checkpoints) and `number N` of checkpoints are key (Section 4.3; Figure 5).
   - Empirical guidance (Section 4.3): the optimal `V` increases with model size; using more checkpoints helps once training has completed.

5. Using merged weights for later stages (`PMA-init`)
   - Initialize `Continual Training (CT)` or `Supervised Fine‑Tuning (SFT)` with a merged checkpoint instead of the last single checkpoint (Section 4.4).
   - Benefit: smoother optimization dynamics (lower early loss, smoother GradNorm) with little or no downstream performance penalty, and occasional gains (Sections 4.4–4.5; Figure 6; Appendix B Table 1).

6. Why merging can help (mechanism)
   - Second‑order Taylor expansion of the loss around a local optimum (`θ*`) shows that averaging weights reduces loss if cross‑terms among checkpoint deviations are sufficiently “negatively correlated” under the Hessian (Equations 6–15, Section 4.6). Intuition: different checkpoints sample nearby directions; averaging cancels idiosyncratic deviations and lands closer to a better point.
   - A 2D weight‑plane visualization shows the merged point (red star) lies nearer to higher MMLU contours than the individual checkpoints (black dots) (Figure 8).

Design choices and rationale
- Use `SMA` by default for simplicity—differences among `SMA`, `WMA`, and `EMA` fade later in training; `WMA` can be slightly better early when weights change faster (Section 4.2; Figure 4).
- Save 10 checkpoints (`N≈10`) where feasible; this balanced performance and cost (Section 4.3; Figure 5).
- Pick `V` to match model scale: small models use short intervals; large models use longer intervals (Section 4.3).

## 4. Key Insights and Innovations
1. Turning constant‑LR checkpoints into an “annealing simulator”
   - Novelty: Use weight merging during the stable phase to approximate what cosine decay would eventually achieve, enabling faster validation loops (Section 4.1).
   - Evidence:  
     > “At the early annealing stage, the results of PMA were comparable to those at the end of the annealing process… in some cases… the merged models even surpassed those naturally annealed.” (Figure 2)  
     > “Pre‑training with a constant learning rate, combined with model merging, can effectively match the performance of an annealed model … without the need for learning rate annealing.” (Section 4.1; Figure 3)

2. Practical scaling rules for merging hyperparameters
   - Contribution: Clear guidance on checkpoint spacing `V` and count `N` as functions of model size (Section 4.3).
   - Evidence:  
     > “The optimal interval scales with model size… around 8B tokens for 1.3B/13B models, 4B for 0.7B/7B models, and approximately 80B tokens for 10B/100B models.” (Section 4.3; Figure 5)

3. Stability booster via `PMA-init`
   - Contribution: Initialize CT/SFT from merged weights to reduce training instabilities (Section 4.4–4.5).
   - Evidence:  
     > “A model initialized with PMA-init … demonstrated a notably more stable GradNorm metric … with reduced frequency of loss spikes.” (Figure 7, left)  
     > In a small MoE trained at an excessively high LR (`6e−3`), merging three pre‑collapse checkpoints stabilized and resumed training past the loss spike (Figure 7, right).

4. Theoretical explanation rooted in Hessian cross‑terms
   - Contribution: A transparent derivation (Equations 6–15) showing why averaging can yield lower loss than the average of individual losses when checkpoints explore complementary directions.
   - Significance: Moves beyond empirical observations to a mechanistic account applicable to large models (Section 4.6).

Incremental vs. fundamental
- Incremental: Trying `EMA/WMA/SMA` variants; documenting small early‑phase differences (Section 4.2).
- Fundamental: The idea that stable‑phase checkpoint averaging accurately tracks annealed performance in large‑scale LLM pre‑training, with generalizable scaling guidelines and a stability‑oriented initialization method.

## 5. Experimental Analysis
Evaluation design
- Models: Dense (411M → 70B) and MoE (activated/total 0.7B/7B → 20B/200B) trained from scratch on an internal multi‑trillion‑token corpus with the `WSD` learning‑rate schedule (Section 3; Appendix A).
- Checkpoints: Saved at fixed token intervals; merged using `SMA`, `WMA`, or `EMA` (Sections 3–4.2).
- Metrics and benchmarks: Weighted average over public benchmarks (ARC‑C, BBH, DROP, WinoGrande, HellaSwag, MMLU, C‑Eval, TriviaQA, Ape210K, GSM8K, MATH, MBPP, HumanEval, AGIEval, GPQA, MMLU‑Pro) reported as the “overall performance” metric unless noted (Section 3). Several figures also show per‑task scores (Figures 1, 3, 9).

Main quantitative results
- Gains from merging during the stable phase (MoE models, Figure 1):
  - HumanEval examples:  
    > “Seed‑MoE‑1.3B/13B improved from 31.1 to 36.6… Seed‑MoE‑10B/100B increased from 54.3 to 61.6.” (Section 4.1; Figure 1)
  - Improvements are broad across BBH, MMLU, GSM8K (Figure 1; all four MoE scales).

- Annealing equivalence/acceleration (Figure 2 and Figure 3):
  - Early in cosine decay, `Annealing + PMA` matches end‑of‑anneal performance (Figure 2).
  - With constant LR plus PMA, the merged model outperforms both constant‑LR and annealed models early, and remains comparable later:  
    > Four panels in Figure 3 (HumanEval, BBH, MMLU, GSM8K) show `Constant + PMA` quickly catching up to or surpassing `Annealing`.

- Merging strategy ablation (Section 4.2; Figure 4):
  - Early training: `WMA` (or `EMA` with larger α) can be slightly better, because later checkpoints matter more when weights change rapidly.
  - Later training: differences among `SMA`, `WMA`, `EMA` become negligible; default to `SMA` for simplicity.

- Hyperparameters `V` and `N` (Section 4.3; Figure 5):
  - Large `V` early in training hurts because it includes unstable checkpoints; the gap narrows as training stabilizes.
  - After training is complete, larger `N` consistently helps;  
    > “Overall performance for `N=3` was nearly 1 point lower than for `N=15`.” (Section 4.3)

- Downstream CT/SFT with `PMA-init` (Section 4.4; Figure 6; Appendix B Table 1):
  - CT: Lower early loss across LR schedules and similar final performance; early MMLU edge (Figure 6).
  - SFT: Mixed but sometimes positive; a notable case on Seed‑MoE‑15B/150B for 220M SFT tokens:  
    > With the same LR schedule (`2e−5 → 2e−6`), `PMA-init` outperforms baseline on several metrics (Appendix B, Table 1). For a different LR (`1e−5 → 2e−6`), `PMA-init` shows +2.7 on LiveBench and +4.5 on AMC‑2023 vs. baseline `2e−5 → 2e−6`.

- Stability and recovery (Section 4.5; Figure 7):
  - `PMA-init` reduces GradNorm spikes during SFT (Figure 7, left).
  - In a deliberately unstable small‑scale MoE run with extremely high LR (`6e−3`), merging the three checkpoints before collapse allowed stable resumption and continuation past the spike (Figure 7, right).

- Dense models also benefit (Appendix A; Figure 9):
  - Large dense example:  
    > “Seed‑Dense‑70B improved from 50.6 to 57.9 on HumanEval and from 85.9 to 91.3 on GSM8K.” (Appendix A, Figure 9)

Assessment of support
- Breadth: Results span MoE and dense models, multiple scales, and many benchmarks (Sections 4.1–4.5; Appendix A–B).
- Depth: Ablations on merging strategies (`SMA/WMA/EMA`), intervals (`V`), number of checkpoints (`N`), and scheduler phases strengthen the case (Figures 4–5).
- Caveat: Details of datasets and exact architectures are not public (Section 3; Limitations C), which limits external reproduction. Still, the internal consistency across multiple scales and tasks is strong.

Conditions and trade‑offs
- Early training is sensitive: too large `V` or too many checkpoints may include unstable weights and degrade performance; after training stabilizes, larger `N` yields better merged models (Section 4.3; Figure 5).
- SFT benefits are inconsistent across models; `PMA-init` helps optimization smoothness more reliably than final scores (Section 4.4; Appendix B Table 1).

## 6. Limitations and Trade-offs
- Assumptions and scope
  - Merging is along a single training trajectory (sequential checkpoints), not across independently trained models on different data distributions during pre‑training.
  - The Hessian‑based explanation assumes local positive definiteness around a minimum and complementary deviations—conditions that may not always hold (Section 4.6).

- Scenarios not fully addressed
  - Post‑training RL phases are not studied, though the authors note potential relevance given longer RL schedules (Limitations, Appendix C).
  - Domain‑specific effects and contamination checks are not discussed; the evaluation relies on many public benchmarks but the pre‑training corpus is unspecified (Section 3; Appendix C).

- Computational/data constraints
  - While merging itself is cheap, it requires storing and managing multiple checkpoints and careful choice of `V` and `N`. The study hints at token intervals like 80B for 10B/100B MoE models—implying long training spans between useful checkpoints (Section 4.3).
  - Learning‑rate exploration is limited:  
    > “We defaulted to the optimal learning rate derived from scaling laws… we did not further quantify the impact of learning rate on model merging in a more detailed manner.” (Appendix C)

- Performance caveats
  - Near the end of heavy annealing (very low LR), checkpoints are tightly clustered; merging yields smaller gains (Section 4.6, Figure 8).
  - SFT gains are not guaranteed across model sizes (Appendix B).

## 7. Implications and Future Directions
- How this changes practice
  - Incorporate `PMA` as a routine step during the stable phase of pre‑training to:
    - Predict annealed performance early and decide whether to proceed with full annealing.
    - Get consistent performance boosts across tasks and scales (Figures 1–3).
  - Use `PMA-init` as a low‑cost stabilizer for CT/SFT and as a recovery tool when loss spikes occur (Figures 6–7).

- Practical guidelines distilled from the paper
  - Default to `SMA` for mature checkpoints; consider `WMA`/higher‑α `EMA` in early, rapidly changing phases (Section 4.2).
  - Save about `N ≈ 10` checkpoints for merging once training stabilizes (Section 4.3).
  - Choose `V` by model size:  
    > “~4B tokens (0.7B/7B), ~8B (1.3B/13B), ~80B (10B/100B).” (Section 4.3)

- Research directions enabled
  - Adaptive selection of `V` and `N` based on curvature/instability estimates; dynamic weighting schemes guided by Hessian approximations or gradient variance.
  - Extending merging to RL post‑training phases and exploring interactions with long‑horizon schedules (Appendix C).
  - Theoretical work to relax assumptions in Equations (6–15), and to understand merging behavior in non‑convex regions with saddle points or multi‑basin dynamics.
  - Systems work: lightweight checkpointing and streaming merges, so merging can act as an online monitor of eventual annealed performance.

- Applications
  - Faster iteration in foundation model pre‑training programs where compute is a bottleneck.
  - Production training pipelines wanting resilience: use `PMA-init` to resume from instability without rolling back far or re‑tuning hyperparameters.
  - Multi‑stage training stacks (pre‑train → CT → SFT) seeking smoother transitions without sacrificing final quality.

Selected anchored quotes
- Stable‑phase gains:  
  > “Seed‑MoE‑1.3B/13B improved from 31.1 to 36.6 [HumanEval] … Seed‑MoE‑10B/100B increased from 54.3 to 61.6.” (Section 4.1; Figure 1)
- Annealing simulation:  
  > “This suggests that pre‑training with a constant learning rate, combined with model merging, can effectively match the performance of an annealed model …” (Section 4.1; Figure 3)
- Scaling rule for intervals:  
  > “~8B tokens for 1.3B/13B … 4B for 0.7B/7B … ~80B for 10B/100B models.” (Section 4.3)
- Dense model uplift:  
  > “Seed‑Dense‑70B improved from 50.6 to 57.9 [HumanEval] and from 85.9 to 91.3 [GSM8K].” (Appendix A; Figure 9)
- Stability and recovery:  
  > “A model initialized with PMA‑init … showed a notably more stable GradNorm …” and “… merging three checkpoints before the training collapse … stabilized [and] resumed training past the loss spike.” (Section 4.5; Figure 7)

In sum, this paper transforms checkpoint averaging from a post‑training curiosity into a practical and theoretically informed tool for pre‑training: a way to both squeeze out extra performance and to preview annealed outcomes early, with concrete recipes that scale to very large LLMs.
