# Reinforcement Learning for Reasoning in Large Language Models with One Training Example

**ArXiv:** [2504.20571](https://arxiv.org/abs/2504.20571)

## 🎯 Pitch

This paper delivers a paradigm shift for post-training large language models by demonstrating that reinforcement learning with verifiable rewards (RLVR) on just a single training example can match the reasoning gains typically seen with thousands of curated problems. By uncovering that one well-chosen problem can double mathematical benchmark scores and induce robust generalization, the work not only slashes the data and compute requirements for RL-based alignment, but also deepens our scientific understanding of what really catalyzes reasoning in LLMs—paving the way for far more data-efficient and targeted language model improvement.

---

## 1. Executive Summary
This paper shows that reinforcement learning with verifiable rewards (RLVR) can substantially improve a language model’s mathematical reasoning even when trained on just one problem (“1-shot RLVR”). On the 1.5B-parameter math model `Qwen2.5-Math-1.5B`, training on a single curated problem boosts MATH500 accuracy from 36.0% to 73.6% and the average of six math benchmarks from 17.6% to 35.7%—matching training on 1,209 examples (Figure 1; Table 8). The study also uncovers a phenomenon they call post-saturation generalization: test accuracy continues to improve long after the single training example is solved perfectly (Figure 2), and identifies which loss terms matter (policy gradient is primary; entropy helps; weight decay/KL matter less; Table 5).

## 2. Context and Motivation
- Problem addressed
  - How much data is actually needed for RLVR to improve reasoning in large language models (LLMs)? Prior work focused on better RL algorithms (e.g., PPO/GRPO) and large curated datasets, but the data efficiency of RLVR itself remained unclear (Section 1).
  - The paper asks explicitly: “To what extent can we reduce the training dataset for RLVR while maintaining comparable performance compared to using the full dataset?” (end of Section 1).

- Why it matters
  - Practical: If a single (or a few) high-value problem(s) can trigger strong reasoning across tasks, we can dramatically reduce RL data curation and training cost while speeding iteration.
  - Scientific: Understanding what drives reasoning gains in RLVR (policy gradients vs. format correction vs. exploration) clarifies the mechanism of “reasoning incentive,” informs data selection, and distinguishes effects from known phenomena like “grokking.”

- Prior approaches and gaps
  - RLVR has recently powered reasoning improvements (OpenAI o1, DeepSeek-R1, Kimi) using large datasets and robust RL pipelines (Section 1).
  - Algorithmic refinements to PPO/GRPO improve stability and sample efficiency [9–16], but the role of data quantity and quality—especially extreme reductions—has been underexplored.
  - Closest study: LIMR reduces RLVR set size by ~6x using a learning-impact metric, but does not probe the extreme case (e.g., 1 example) (Related Work A).
  - A concurrent work shows 4-example PPO can already help, but without the systematic analysis here (Related Work A).

- Positioning
  - This paper pushes the data-efficiency limit: it shows 1-shot RLVR can match (and 2-shot slightly exceed) performance from thousands of examples on math reasoning (Figure 1; Tables 8–9).
  - It analyzes underlying mechanisms (ablation of loss components, effect of exploration), data selection (a simple variance-based ranker), cross-category generalization, format correction vs. reasoning, label robustness, and prompt complexity (Sections 3–4; Appendix C).

## 3. Technical Approach
At a glance: The authors train via GRPO (a PPO-variant used in recent RLVR systems) on `Qwen2.5-Math-1.5B` and other models. The reward is verifiable: 1 if the final answer matches the gold answer, 0 otherwise. “1-shot RLVR” means the training dataset contains a single math problem duplicated to fill batches (Section 3.1).

Key concepts (defined where nonstandard):
- RLVR (reinforcement learning with verifiable reward): RL where the reward is computed by a rule-based checker (e.g., does the model’s final boxed answer equal the ground truth?).
- Format reward: a control condition where the model is rewarded only for producing a parseable final answer (e.g., including a “\boxed{...}”), regardless of correctness (Appendix C.2.3).
- GRPO: a PPO-style training scheme that samples multiple responses per prompt and uses group-normalized advantages (Sections 2 and B.1).
- Group-normalized advantage: within each group of sampled responses for a prompt (e.g., 8 completions), convert rewards into zero-mean, unit-variance scores A_i = (r_i − mean)/std (Eq. 6). This centers learning around “better than the group average.”

Step-by-step method
1. Models and tasks (Section 3.1)
   - Primary model: `Qwen2.5-Math-1.5B`; also `Qwen2.5-Math-7B`, `Llama-3.2-3B-Instruct`, and `DeepSeek-R1-Distill-Qwen-1.5B` (Table 4).
   - Benchmarks: 6 math sets—MATH500, AIME 2024, AMC 2023, Minerva Math, OlympiadBench, AIME 2025—and 2 non-math reasoning sets—ARC-Easy/Challenge (B.3; Table 1).
   - Evaluation: pass@1; for small sets (AIME/AMC) they repeat the test 8 times at temperature 0.6 and report avg@8 (B.5).

2. Data and selection (Sections 2; 3.1)
   - Instance pool: a 1,209-example subset of DeepScaleR (“DSR-sub”).
   - A simple “historical variance score” selects promising examples:
     - Do a short RL run over the full pool; track each example’s per-epoch training accuracy list L_i (Section 2).
     - Compute the variance v_i = var(L_i) (Eq. 1).
     - Rank examples by v_i (Eq. 2) to get π_1, π_2, …; π_1 and π_13 turn out to be strong “1-shot” training seeds (Figures 1–2; Table 3).
   - Important nuance: many examples (including medium/low variance) still work in 1-shot RLVR (Table 3).

3. RL algorithm and loss (Sections 2; B.1)
   - Policy gradient term (Eq. 4): encourages higher-probability for responses with higher relative reward within a group (via A_i).
   - KL penalty to a reference model (Eq. 5): keeps language quality from drifting too far.
   - Entropy term (Eq. 7–8): increases per-token entropy to promote exploration/diversity (coefficient α < 0). This is optional in principle, but included by default in their framework “verl” (Section 2).
   - Key ablation: the paper later shows policy gradient is the main driver of gains; entropy helps further; KL/weight decay add little (Table 5; Figure 12).

4. Training and evaluation details (Section 3.1; B.4–B.5)
   - Sampling: 8 responses per prompt per step; 8 gradient updates per rollout.
   - Default temperature 0.6 for training rollouts; max prompt length 1024; max response length 3072 (Qwen2.5-Math models have 4096 context).
   - Batch size 128 (they duplicate the single example to reach this size for 1-shot; footnote in Section 3.1).
   - Coefficients: KL β = 0.001; entropy α = −0.001 by default.
   - For DeepSeek-R1-Distill-Qwen-1.5B they use longer outputs (max 8192) and report results at 8k and 32k evaluation lengths (Appendix C.1.6).

5. The one example that “works” (π1, Table 2; Section 3.2.1)
   - π1 is a joint-variation wind-pressure algebra problem solvable by formula P = k A V^3.
   - The base model already performs the derivation reliably and only hesitates on the cube-root 2048^(1/3) ≈ 12.7–12.8 (Section 3.2.1): in 128 base samples, 57.8% output “12.7/12.70”, 6.3% “12.8”, 6.3% “13”.
   - This matters: the example is not hard; it seems to “unlock” or stabilize reasoning behaviors the base model almost has. (The paper also shows π13 (geometry; Table 21) works similarly well.)

## 4. Key Insights and Innovations
- One example is enough to match thousand-example RLVR on math benchmarks
  - With `Qwen2.5-Math-1.5B`, 1-shot RLVR using π13 achieves MATH500 73.6% and six-task average 35.7%—on par with training on 1,209 examples (73.6% and 35.9%) and close to using 7,500 MATH training problems (36.7% average) (Figure 1; Table 8).
  - Two examples (π1, π13) slightly exceed both (average 36.6%; Figure 1; Table 8–9).
  - This is a fundamental change in how we think about data needs in RLVR: the “spark” can come from a single, well-chosen problem.

- Post-saturation generalization (Section 3.2.2; Figure 2; Figure 3)
  - Even after training accuracy on the single example hits ~100% before step 100, test accuracy continues to climb for 1,000+ steps (e.g., π1 gains +3.4% average from step 100 to 1540; π13 +9.9% from step 500 to 2000).
  - Overfitting on the training example appears late (step ~1400–1800) and manifests as multilingual gibberish interleaved with the correct solution for the training problem (Figure 3, “Step 1860”); yet test performance remains high and coherent.
  - This phenomenon differs from “grokking”: improvements are primarily tied to policy gradients rather than to heavy regularization like weight decay (Section 4.1; Table 5; Figure 12).

- Mechanistic clarity via ablations: policy gradient is the engine; exploration amplifies it (Section 4.1; Table 5; Figure 5; Figure 12)
  - Adding only policy gradient to π1 yields nearly all gains (Table 5, Row 2).
  - KL and weight decay have minor effects (Rows 3–4).
  - Entropy (exploration) significantly improves later-stage generalization but must be tuned (α too negative can destabilize; Rows 5–6; Figure 5).

- Format correction is a big piece—but not the whole story (Appendix C.2.3; Table 14; Figures 13–15)
  - Rewarding only format (producing a parseable “\boxed{…}”) improves performance substantially (e.g., MATH500 ≈ 65%), but outcome-based RLVR with the same single example does better (≈ 73–75%), even at similar “boxed” rates (Table 14; Figures 13–14).
  - The correlation between “boxed” ratio and accuracy is strong (Figure 15), but some tasks keep improving even after the “boxed” rate plateaus, suggesting extra reasoning gains beyond format.

- Cross-category generalization and example diversity (Section 3.2.3; Table 3)
  - Training on one example from, say, Geometry improves Algebra, Number Theory, etc. The magnitude varies by example. Some examples mostly fix format; others add nontrivial accuracy beyond format (Table 3).
  - Not all examples work equally; bad labels or very hard problems can fail (π1207 wrong label; π1208 too hard).

## 5. Experimental Analysis
Evaluation setup and baselines
- Datasets and metrics (Section 3.1; B.3)
  - Six math benchmarks: MATH500 (500 curated MATH test problems), AIME24 (30), AMC23 (40), Minerva Math (272), OlympiadBench (675), AIME25 (30).
  - Non-math: ARC-Easy/Challenge multiple-choice science reasoning (Table 1).
  - Primary metric: pass@1; for small sets they average pass@1 across 8 repeats (“avg@8”) at T=0.6.

- Training regimes compared (Figures 1–2; Tables 8–12)
  - 1-shot / 2-shot / 4-shot RLVR using selected examples.
  - Full-set RLVR using 1.2k DSR-sub and 7.5k MATH.
  - Format-reward RLVR baseline (Appendix C.2.3; Table 14).
  - In-context learning (ICL) baselines using either the single example π1 or four official Qwen examples (Table 16).
  - RL algorithms: GRPO (default) and PPO (Table 4 and Table 11).
  - Models: `Qwen2.5-Math-1.5B/7B`, `Llama-3.2-3B-Instruct`, `DeepSeek-R1-Distill-Qwen-1.5B` (Tables 4, 10–12).

Headline results (Qwen2.5-Math-1.5B; Figures 1, 7; Tables 8–9)
- 1-shot (π13): MATH500 73.6%; six-benchmark average 35.7%—matches 1.2k DSR-sub (73.6/35.9) and nearly 7.5k MATH train (74.4/36.7) (Figure 1; Table 8).
- 2-shot (π1+π13): even higher MATH500 74.8% and average 36.6% (Figure 1; Table 8).
- Format reward (full 1.2k) yields 65.0% on MATH500 and 28.7% average (Table 8). Thus, roughly 7–8 points of the 1-shot gain beyond format correction are attributable to outcome-based RL.
- Non-math generalization (Table 1): ARC-Easy/Challenge also increases with 1-shot math RL; π13 gives 55.8/33.4 vs the base 48.0/30.2 and vs full-set RLVR 42.2/29.9—indicating cross-domain benefits.

Cross-example analysis (Table 3; Figure 6)
- Most single examples deliver 30+ point MATH500 gains over base, across categories (Algebra, Geometry, etc.).
- Two corner cases illustrate failure modes: π1207 (incorrect label) and π1208 (extremely hard) barely improve.
- Some examples mainly fix format; others provide additional reasoning gains beyond format (compare each example to the format baseline rows in Table 3 and Appendix C.2.3).

Other models/algorithms (Table 4; Figures 8–11)
- `Qwen2.5-Math-7B` + GRPO:
  - Base avg 22.4 → 1-shot (π1) 40.2; 4-shot 42.5; full 1.2k 42.8 (Table 4).
  - 16-shot handpicked set outperforms 16-shot random (Table 4), reinforcing data selection value.
- `Llama-3.2-3B-Instruct` + GRPO:
  - Gains are smaller but few-shot matches or beats full-set RLVR in some cases (Table 4; Figure 9).
- `Qwen2.5-Math-1.5B` + PPO:
  - 1-shot (π1) also works (Table 4 and Table 11).
- `DeepSeek-R1-Distill-Qwen-1.5B` + GRPO:
  - Few-shot improves but does not fully close the gap with full-set; however, 16-shot approaches full-set (Table 4; Table 12).

Ablations and diagnostics
- Loss ablations (Section 4.1; Table 5; Figure 12):
  - Policy gradient alone delivers most gains (Row 2).
  - Entropy helps more (Row 5 vs Row 4; Figure 5 shows increasing exploration via entropy or higher temperature improves late-stage generalization).
  - Removing policy gradient and using only entropy still gives a smaller but nontrivial boost—consistent with broad format improvements (Rows 9–10; Table 6/13).
- Format vs. outcome (Appendix C.2.3; Table 14; Figures 13–15):
  - Format-only RLVR improves a lot but underperforms outcome-based 1-shot with similar “\boxed{}” rates; outcome-based adds genuine reasoning benefit.
  - LLM-as-a-judge cross-check (QwQ-32B) agrees closely with rule-based evaluation; improvements are not just “put the answer in the box” hacks (Table 15).
- Label robustness (Section 4.2; Table 5 rows 11–13; Appendix C.2.4 Table 17):
  - For 1-shot, small label deviations (12.7 vs 12.8) barely hurt; but an overfittable wrong label (“4”) is worse than a totally wrong/unlearnable label (Row 12 vs Row 13).
  - For full-set RL, randomizing 60% of labels still preserves decent performance; at 90% wrong labels, performance drops below 1-shot RL (Table 17).
- Prompt complexity matters (Appendix C.2.5; Table 18):
  - Training on only the hard sub-step of π1 (“compute 2048^(1/3)”) underperforms full π1 and is near format-reward levels—suggesting richer reasoning chains offer better “exploration space.”
- Self-reflection increases (Section 3.2.4; Figure 4 right):
  - The frequency of “rethink/recheck/recalculate” words in test-time outputs rises as 1-shot training continues, correlating with longer responses and higher entropy (Figure 4 left/middle).

Are the experiments convincing?
- The core claim—that 1-shot RLVR can match full-set results in math reasoning on some models—has strong quantitative support across multiple datasets, models, and RL algorithms (Figures 1–2; Tables 4, 8–12).
- The mechanism claims are supported by ablations (Table 5; Figures 5 and 12) and format controls (Appendix C.2.3), plus behavioral measurements (self-reflection counts; response length; entropy; Figure 4).
- The paper is careful to separate format improvements from reasoning accuracy through multiple controls (format reward, entropy-only training, LLM-as-a-judge, and ICL baselines; Appendix C.2.3; Tables 14–16).

## 6. Limitations and Trade-offs
- Scope: Primarily math reasoning with verifiable final answers. Generalization to coding or non-verifiable tasks is not demonstrated, though the authors expect similar dynamics may hold (Sections 5 and D.4).
- Dependence on example choice:
  - Many single examples work, but not all (Table 3). Bad labels or too-hard problems can nullify gains (π1207, π1208).
  - A variance-based selector (Eq. 1–2) helps, but is not guaranteed optimal, and π1 is identified specifically for the 1.5B Qwen model (Sections 2 and 3.2.3; Appendix D.1).
- Training stability and hyperparameter sensitivity:
  - Entropy coefficient and sampling temperature need tuning (Table 5; Figure 5). Some models (e.g., `Llama-3.2-3B-Instruct`) show training instability or early degradation (Appendix C.1.5; Figure 9).
- Compute still substantial:
  - Even with 1 example, training involves millions of rollouts before overfitting manifests (Section 3.2.2). 1-shot reduces data curation but not necessarily wall-clock compute.
- Pass@k dynamics:
  - For full-set RLVR, pass@8 can degrade later in training (Appendix C.4, Table 20). Effects on sampling diversity vs. single-answer accuracy may require careful balancing.

## 7. Implications and Future Directions
- Data efficiency rethought
  - A single well-chosen problem can “ignite” broad reasoning in small-to-mid models in math. This shifts emphasis from massive RL datasets to intelligent selection and curriculum design.
  - Practical upshot: faster, cheaper iteration cycles for RLVR post-training, especially for new domains with scarce labeled problems.

- Mechanism-level insights
  - The main driver is the policy gradient on verifiable outcomes; exploration (entropy, temperature) is a key amplifier that sustains post-saturation gains; KL/weight decay are secondary (Section 4.1).
  - Post-saturation generalization suggests the single-example reward acts as an “implicit regularizer,” filtering explorations that break the solved example while allowing broader behavior search (Discussion D.3; Figure 16).

- Research directions (Discussion D.4)
  - Better data selection: move beyond simple variance ranking to predictive selectors for 1/few-shot RLVR (active selection, influence functions, capability profiling).
  - Exploration strategies: more principled mechanisms than entropy to encourage diverse, useful reasoning without destabilization.
  - Beyond math: replicate in coding or multimodal settings with rule-based or tool-based verifiers.
  - Robustness: deeper study of label noise tolerance, prompt complexity, and pass@k behavior.
  - Evaluation: build benchmarks/tools to separate format, reasoning, and self-reflection effects more cleanly (Appendix C.2.3; Figure 15; Table 15).

> Core takeaway (Figure 1; Tables 8–9): “1-shot RLVR with a single math example (π13) reaches MATH500 73.6% and average 35.7%, matching 1,209-example full-set RLVR; 2-shot (π1+π13) slightly exceeds both.”

> Mechanism takeaway (Table 5; Figure 5): “Policy gradient provides the bulk of the improvement; properly weighted entropy (exploration) further improves late-stage generalization; weight decay and KL have minor impact.”

> Scope and caution (Table 3; Appendix C.2.3): “Format fixing explains a large fraction of the gains but not all; some examples mainly fix format, while others add nontrivial reasoning accuracy beyond format.”
