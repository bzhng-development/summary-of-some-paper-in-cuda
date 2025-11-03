# REASONING GYM: Reasoning Environments for Reinforcement Learning with Verifiable Rewards

**ArXiv:** [2505.24760](https://arxiv.org/abs/2505.24760)
**Authors:** Zafir Stojanovski, Oliver Stanley, Joe Sharratt, Richard Jones, Abdulhakeem Adefioye, Jean Kaddour, Andreas Köpf
**Institutions:** 

## 🎯 Pitch

REASONING GYM (RG) introduces an open-source library of procedurally generated reasoning environments designed to enhance Reinforcement Learning with Verifiable Rewards (RLVR). By providing virtually unlimited and difficulty-controlled tasks across diverse domains, RG overcomes data scarcity, enhances both in-domain and cross-domain reasoning, and boosts transfer learning, setting a new standard for scalable and robust training of reasoning models.

---

## 1. Executive Summary
REASONING GYM (RG) is an open-source library of more than 100 procedurally generated reasoning environments, each paired with an automatic verifier, purpose-built for Reinforcement Learning with Verifiable Rewards (RLVR). It solves the data bottleneck for training and evaluating reasoning models by producing unlimited, difficulty-controlled tasks across many domains (algebra, algorithms, logic, games, geometry, graphs, and more), and shows that RLVR on RG improves both in-domain and cross-domain reasoning as well as transfer to external benchmarks (Sections 1–2, Table 6, Figures 2–6, Tables 1–5).

## 2. Context and Motivation
- Problem addressed
  - RLVR depends on tasks where outputs can be automatically checked to give a reliable reward signal. Existing reasoning datasets are mostly fixed-size, human-curated or scraped from the web, which creates scarcity, contamination, and memorization issues (Section 1). This becomes a severe bottleneck as reasoning models grow (data scaling limits, Section 1).
- Why this matters
  - Without abundant, controllable, and verifiable tasks, it is hard to:
    - Train RLVR models at scale without overfitting.
    - Probe specific reasoning skills or study difficulty effects.
    - Build adaptive curricula that increase challenge as models learn.
- Prior approaches and shortcomings
  - Fixed benchmarks (e.g., GSM8K, MATH, BIG-Bench) are valuable but saturate or are susceptible to memorization and noise (Section 6).
  - Some projects procedurally generate tasks or use text games but cover narrower domains or lack unified difficulty controls and verifiers across a broad reasoning spectrum (Section 6).
- Positioning
  - RG offers a single, extensible library of generators with:
    - Algorithmic verifiers for objective rewards (P1).
    - Large solution spaces to discourage reward hacking (P2).
    - Parametric difficulty for continuous curricula (P3).
  - It is designed explicitly for RLVR training and systematic evaluation (Section 2, Figure 2, “Following are the core design principles”).

## 3. Technical Approach
RG is a collection of “environments” rather than a fixed dataset. Each environment has three core parts (Sections 2 and A.2; Table 6):
1) A generator that produces a novel instance and ground-truth solution.
2) A verifier that deterministically checks a model’s answer and returns binary correctness (plus optional auxiliary format checks).
3) Tunable parameters that modulate difficulty and structure.

Key mechanisms
- Procedural generation
  - “Procedural” means every training example is constructed by code at sampling time (not drawn from a fixed file). Parameters control size, constraints, and transformations so instances are virtually never repeated. Examples include:
    - Algorithms: `spiral_matrix`, `rotate_matrix`, `count_primes`.
    - Games/puzzles: `rush_hour`, `rubiks_cube`, `mini_sudoku`, `sokoban`.
    - Logic: `knights_knaves`, `circuit_logic`.
    - Geometry and graphs: `advanced_geometry`, `shortest_path`.
  - Concrete task examples with inputs/outputs and metadata appear in Appendix A.2 (e.g., spiral traversal, BF program output, mini-sudoku).
- Verifiable rewards (RLVR)
  - A “verifier” is a program that checks the output (e.g., does the Sudoku fill obey rules? does the path reach the goal? did the algebraic expression simplify correctly?). This enables:
    - Objective rewards (+1/0) without human judgment (P1).
    - Unlimited training at scale (no annotators).
  - RG also offers auxiliary, automatically computed rewards such as formatting, which are combined with accuracy during training (Section 4, “training reward plots represent the total reward … accuracy + auxiliary (formatting)”).
- Difficulty control
  - Each environment exposes parameters that adjust complexity (P3), e.g., matrix size, graph size, number of constraints, polynomial degree, or number of empty cells in Sudoku. Easy vs. hard configurations used in zero-shot tests are enumerated in Appendix A.3; difficulty cliffs are then analyzed in Figure 3b.
- Large solution spaces (P2)
  - Tasks are designed so that many solution paths exist (e.g., different move sequences in `rush_hour` or `rubiks_cube`, multiple expressions that evaluate to the target in arithmetic puzzles), which reduces the chance that models exploit narrow reward hacks.

How RL training is performed in practice
- RL algorithm and setup
  - Most training uses GRPO (a PPO-style method designed for sampling multiple responses per prompt and applying a KL penalty to keep the policy close to a reference) with 8 samples per prompt (`n: 8`) and a KL term (`kl_coef: 0.001`) (Appendix A.6 config).
  - Rewards combine: accuracy (pass/fail by verifier) and small auxiliary signals (e.g., formatting) [Figure 5 caption; Section 4].
  - Training scale: roughly 1500 A6000 GPU hours on cloud GPUs via Runpod (Section 4).
- Evaluation protocol
  - Zero-shot: evaluate frontier LLMs on easy/hard RG configurations (Figures 2, 3, 7, 8).
  - Intra-domain RLVR: train on several tasks within a category and evaluate on a held-out task from the same category (Figure 4; Table 1).
  - Cross-domain RLVR: train on one category and evaluate on other categories never seen during training (Figure 5; Table 2).
  - External benchmarks: test transfer to GSM8K, MATH, Big-Bench Hard, and MMLU-Pro (Tables 3 and 4).
- Metrics
  - Accuracy (0–100%) for zero-shot and external benchmarks (Figures 2–3; Tables 3–4).
  - Acc@3 for intra- and cross-domain experiments: problem is counted solved if any of up to three samples is correct (Tables 1–2).
  - Reward curves during training (Figures 4–6).

Design choices, and why
- Single library with breadth over depth:
  - Broad skill coverage (Table 6) lets RLVR target compositional and transferable reasoning, not just a single niche.
- Verifiers everywhere:
  - Reliable, low-noise rewards enable large-scale RL without humans (P1).
- Parametric difficulty:
  - Supports curriculum learning; yields controlled “hardness” for rigorous evaluation of scaling and robustness (P3; Section 5; Figure 6; Table 5).

## 4. Key Insights and Innovations
1) A unified, verifiable, procedurally generated corpus for reasoning RL
   - What’s new: Over 100 generators across 12 categories, each with a programmatic verifier and tunable difficulty (Section 2; Table 6; Appendix A.2).
   - Why it matters: Removes the data bottleneck for RLVR by enabling “virtually infinite” training signals with objective rewards (Abstract; P1–P3).
2) Difficulty cliffs as a diagnostic for depth of reasoning
   - Novel finding: Frontier models’ performance drops sharply when moving from easy to hard configurations, especially in code (−71.9%), graphs (−33.8%), and geometry (−33.1%) for `o3-mini` (Figure 3b). This suggests shallow competence and template reliance rather than robust reasoning.
   - Significance: Establishes RG as a sensitive tool for measuring real progress beyond trivial regimes (Section 3.2).
3) RLVR training induces intra-domain and cross-domain transfer
   - Intra-domain: RL on a category improves held-out tasks within that category (Table 1), e.g., algebra +11.7% Acc@3 and algorithms +7.4%.
   - Cross-domain: Training on algorithms improves algebra by +29.1% and geometry by +22.3%; training on logic improves cognition by +13.3% and graphs by +9.1% (Table 2).
   - Significance: Evidence that skills learned via verifiable tasks transfer to other domains, not merely memorization of patterns (Section 4).
4) Curriculum RLVR improves final performance
   - Contribution: An adaptive curriculum that increases difficulty once the model sustains ≥70% success over 20 steps (Section 5).
   - Results: Curriculum outperforms non-curriculum across tasks/levels (Table 5), e.g., Mini Sudoku with 8–10 empty cells: +13.33%.
   - Significance: Shows how RG’s difficulty controls can be operationalized to accelerate learning and reach higher difficulty.
5) External validity: benefits on standard benchmarks
   - MATH: +9.7%; Big-Bench Hard: +7.66%; GSM8K: +0.5% (Table 3) after RG-Math RLVR.
   - MMLU-Pro: notable gains across many subjects for RG-Algorithmic and RG-Math (Table 4).
   - Significance: Training purely on RG’s synthetic, verifiable tasks improves real-world evaluation targets.

## 5. Experimental Analysis
- Datasets and tasks
  - RG categories and datasets are enumerated in Table 6; examples and metadata in Appendix A.2.
  - Easy vs. hard parameter settings are fully listed in Appendix A.3; per-dataset heatmaps appear in Figures 7 (easy) and 8 (hard).
- Models and baselines (Zero-shot)
  - Frontier models include reasoning-optimized `openai/o3-mini`, `deepseek/deepseek-r1`, `x-ai/grok-3-mini-beta`, and non-reasoning strong baselines like `meta-llama/llama-4-maverick`, `anthropic/claude-3.5-sonnet`, and `google/gemma-3-27b-it` (Figure 3a; Figure 8).
- Metrics
  - Accuracy (%). For intra-/cross-domain RL, Acc@3. For training, total reward = accuracy + auxiliary formatting reward (Sections 3–5).
- Main quantitative results
  - Zero-shot hierarchy (Figure 3a):
    - Average scores on hard configs: `o3-mini` 63.51%, `DeepSeek-R1` 59.52%, `Grok 3 Mini` 55.06% vs `Llama 4 Maverick` 41.50%, `Claude 3.5 Sonnet` 40.33%, `Gemma 3 27B` 20.26%.
    - Insight: Reasoning-optimized models consistently outperform general-purpose models.
  - Difficulty cliffs (Figure 3b):
    - For `o3-mini`, accuracy drops from easy→hard are largest in code (−71.93%), graphs (−33.80%), geometry (−33.13%), algorithms (−25.57%). `DeepSeek-R1` shows similar patterns (−61.82% code, −29.60% graphs, −11.83% geometry, −27.85% algorithms).
    - Interpretation: Capabilities are brittle under increased structure/scale complexity.
  - Intra-domain RLVR (Figure 4; Table 1):
    - Acc@3 gains across all categories; e.g., algebra 5.0 → 16.7 (+11.7), arithmetic 89.7 → 96.0 (+6.3), games 0.0 → 3.3 (+3.3).
    - Reward curves show rapid early improvement partly from learning format rewards, then continued accuracy gains (Figure 4).
  - Cross-domain RLVR (Figure 5; Table 2):
    - Training on algorithmic tasks boosts algebra (+29.1%) and geometry (+22.3%).
    - Training on logic improves cognition (+13.3%) and graphs (+9.1%).
    - Games-trained models, despite weak in-domain competence, transfer to algebra (+21.8%) and cognition (+13.1%).
  - External benchmarks (Tables 3–4):
    - On MATH (0-shot CoT), RG-Math improves 48.5 → 58.2 (+9.7); on Big-Bench Hard (3-shot CoT), 8.68 → 16.34 (+7.66); on GSM8K (8-shot CoT), 76.2 → 76.7 (+0.5).
    - On MMLU-Pro, RG-Math raises Math (+5.62), Physics (+5.70), CS (+4.40), Biology (+4.19), Psychology (+6.02). RG-Algorithmic also improves many subjects (Table 4).
  - Curriculum RL (Section 5; Figure 6; Table 5):
    - Consistent improvements vs. non-curriculum; e.g., Spell Backwards word length 4: 12.00 (baseline) → 30.00 (non-curric) → 70.67 (curric).
- Robustness, ablations, and failure cases
  - Difficulty-graded evaluation (Appendix A.3; Figures 7–8) functions as a robustness test across parameter ranges.
  - Failure modes cluster in long-horizon, spatial, or constraint-satisfaction puzzles (e.g., `rush_hour`, `rubiks_cube`, `rotten_oranges`) where even top models falter under hard settings (Figure 8 caption).
- Do the experiments support the claims?
  - Yes, for three reasons:
    - Breadth: 100+ generators across domains (Table 6) demonstrate generality.
    - Difficulty sensitivity: Clear, quantitative cliffs (Figure 3b) show RG can discriminate shallow vs deep competence.
    - Transfer: Intra- and cross-domain improvements plus external benchmark gains (Tables 1–4) support that RLVR on RG teaches transferable skills.

## 6. Limitations and Trade-offs
- Scope limits of procedural generation (Section 7)
  - Hard to cover tasks demanding rich world knowledge, creativity, or unstructured answers; verifiers require well-defined correctness.
- Verifier fidelity (Section 7)
  - Automatic checkers may miss subtleties valued by humans (e.g., elegance, explanation quality). RLHF or human-centric signals may still be needed for some objectives.
- Interaction modality (Section 7)
  - Current RG focuses on single-turn, text-only tasks; multi-turn agentic interactions and multimodal (vision-language) settings are not yet included.
- Training distribution (Section 7)
  - Experiments sample tasks i.i.d. across environments. Continual, non-stationary scenarios and catastrophic forgetting mitigation are left for future work.
- Compute and optimization practicalities
  - RLVR training is compute-intensive (~1500 A6000 hours, Section 4). GRPO hyperparameters (KL, sampling n) and auxiliary rewards require careful tuning to avoid spurious optimization (formatting rewards partly inflate training curves; Figures 4–6).
- Reward mis-specification risk
  - Although RG emphasizes “large solution spaces,” some environments could still admit reward hacking if verifiers are narrow. The paper mitigates this through design (P2) but does not provide a formal guarantee.

## 7. Implications and Future Directions
- How this changes the field
  - RG provides the missing infrastructure for scalable, controlled, and verifiable RL training of reasoning models. It makes it practical to study curricula, difficulty scaling, and transfer systematically (Sections 2–5). The observed difficulty cliffs (Figure 3b) also set an agenda for genuinely robust reasoning.
- Follow-up research enabled
  - Multi-turn and multimodal reasoning environments (Section 7).
  - Continual learning on non-stationary procedurally generated streams with replay, regularization, or model merging (Section 7).
  - Reward design studies: balancing accuracy with auxiliary signals and preventing spurious rewards.
  - Specialized curricula that adapt per-skill or per-environment, beyond simple thresholds (Section 5 indicates promise).
- Practical applications
  - RL training pipelines for math, algorithmic reasoning, and games that do not require human annotation and can scale indefinitely.
  - Curriculum-based fine-tuning to target weak skills in a model portfolio.
  - Evaluation harness for product models to monitor robustness against increased problem difficulty (Figures 7–8 show how to configure easy/hard regimes).
  - Pretraining or post-training for downstream benchmarks—evidenced by gains on MATH, Big-Bench Hard, and MMLU-Pro (Tables 3–4).

> Overall, RG is both a dataset generator and a scientific instrument: it furnishes verifiable, difficulty-controlled reasoning tasks (Table 6; Appendix A.3) and demonstrates, through extensive experiments, that RLVR on these tasks improves reasoning breadth and transfers beyond the training domains (Figures 4–6; Tables 1–4).
