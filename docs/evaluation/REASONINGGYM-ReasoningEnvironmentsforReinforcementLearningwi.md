# REASONING GYM: Reasoning Environments for Reinforcement Learning with Verifiable Rewards

**ArXiv:** [2505.24760](https://arxiv.org/abs/2505.24760)

## 🎯 Pitch

REASONING GYM (RG) introduces an expansive library of over 100 procedurally generated, automatically verifiable reasoning tasks spanning mathematics, algorithms, logic, games, and more, specifically tailored for reinforcement learning with verifiable rewards (RLVR). By delivering unlimited, difficulty-adjustable, and objectively gradable challenges across diverse domains, RG breaks through the data bottleneck holding back next-generation reasoning models—enabling not only robust intra- and cross-domain generalization but also measurable gains on established benchmarks. This open-source, curriculum-ready resource paves the way for continual improvement and reliable evaluation of reasoning-capable language models, overcoming the limitations and memorization risks of fixed datasets.

---

## 1. Executive Summary
REASONING GYM (RG) is a library of 100+ procedurally generated, automatically verifiable reasoning environments designed for training and evaluating language models with Reinforcement Learning with Verifiable Rewards (`RLVR`). It solves the data bottleneck for reasoning training by producing unlimited, difficulty-controlled tasks across math, algorithms, logic, games, geometry, graphs, and more, and shows that RLVR on RG yields both intra-domain and cross-domain generalization and improves external benchmarks (e.g., MATH, Big-Bench Hard; Tables 3–4).

## 2. Context and Motivation
- Problem addressed:
  - Training modern reasoning-focused LLMs relies on `RLVR`—a setup where models receive reward only when their final answers can be checked automatically. This requires large volumes of clean, verifiable problems. Existing sources are fixed benchmarks or scraped/curated examples, which are limited, potentially noisy, and prone to memorization.
  - RG targets this bottleneck by providing “infinite” procedurally generated tasks with verifiable rewards and adjustable difficulty (Section 2; Figure 1; Table 6).

- Why this matters:
  - State-of-the-art reasoning models (e.g., o3-mini, DeepSeek-R1) gain capabilities via RLVR. Without scalable, high-quality, automatically checkable tasks, further progress is constrained (Introduction; Sec. 1).
  - Fixed benchmarks saturate quickly and are vulnerable to contamination/memorization; RG aims for continual training and evaluation at increasing difficulty with clear, objective scoring (P1–P3 in Section 2).

- Prior approaches and gaps:
  - Fixed datasets: GSM8K, MATH, BIG-Bench, etc., are not endlessly scalable and can be memorized; noisy internet-scraped sets can contain errors (Section 1; Related Work, Section 6).
  - Some procedural testbeds exist (e.g., games, puzzle generators), but coverage across reasoning domains with unified verifiers and curriculum-ready difficulty controls remains limited (Section 6).

- Positioning relative to existing work:
  - RG unifies 100+ tasks across heterogeneous reasoning skills with:
    - Verifiers for automatic grading (P1).
    - Large solution spaces to reduce reward hacking (P2).
    - Parametric difficulty controls for curriculum learning (P3).
  - The library includes training infrastructure and configs to make RLVR training turnkey (Section 2; Appendix A.6).

## 3. Technical Approach
RG is a suite of procedural generators plus verifiers, organized by reasoning category. Each task is not a fixed question; it is a program that, given parameters, generates a new instance and a ground-truth verifier.

- Core design principles (Section 2):
  > (P1) Algorithmic Verifiability. Every task admits automatic verification and requires no human judgment.  
  > (P2) Large Solution Spaces. Tasks are designed with expansive solution spaces…mitigating reward hacking.  
  > (P3) Parametric Difficulty Control. Configurable parameters systematically control problem characteristics…

- Task taxonomy and coverage:
  - Categories include algebra, arithmetic, algorithms, cognition/ARC-like tasks, code understanding, games/puzzles, geometry, graphs, induction, logic (Table 6). Figure 1 shows examples such as Rubik’s Cube (games), Rush Hour (games), Figlet fonts (cognition), Binary matrix (cognition), Circuit logic (logic).

- How a task is structured (Appendix A.2 provides concrete examples):
  - Generator: samples a problem instance using difficulty/structural/stylistic parameters (Section 2; “Concretely…”).
    - Example “spiral_matrix”: randomly samples an `n×n` matrix and asks for its clockwise spiral traversal (A.2.2), with `n` in `[2,10]` by default.
    - Example “bf” (Brainf*ck): outputs a BF program and asks for its output; verifier runs an interpreter (A.2.5).
    - Example “mini_sudoku”: produces a 4×4 Sudoku with a unique solution; verifier checks grid constraints (A.2.7).
  - Verifier: deterministic program checks the model’s final answer matches required format and truth condition.
  - Metadata: each sample ships the parameters and the ground truth to support reproducibility (see Metadata blocks in A.2).

- Parametric difficulty:
  - Each dataset exposes difficulty knobs (Appendix A.3). “Easy” vs “Hard” configurations adjust ranges such as size/degree/problem structure.  
    - Example: “prime_factorization” increases max value from 1000 (easy) to 5000 (hard).  
    - “rubiks_cube” increases scramble steps from 3–10 (easy) to 25–50 (hard).  
    - “color_cube_rotation” increases number of rotations up to 50 (hard).

- RLVR training pipeline (Sections 4 and A.6):
  - Base model: `Qwen2.5‑3B‑Instruct` is trained via `GRPO` (a lightweight policy-gradient variant related to PPO that samples multiple candidate completions per prompt and uses KL regularization) on RG tasks. Rollouts use sampling (n=8; Appendix A.6).
  - Reward shaping:
    - Total reward = accuracy (1.0 if the verifier passes, else 0) + an auxiliary format reward (worth 0.2) to incentivize well-formed answers (Section 4: “training reward plots represent the total reward…”).
  - Training infrastructure:
    - Implemented with the `verl` library; typical runs on a 4×A6000 node; all configs provided (A.6).  
    - Reported experiments consume ~1500 A6000 GPU-hours (Section 4).

- Evaluation protocol:
  - Zero-shot evaluations compare frontier models across RG tasks on easy vs hard settings (Sections 3, A.4–A.5).  
  - Intra-domain and cross-domain RLVR transfer studies evaluate post-RL performance on held-out tasks; each run uses 50-problem test sets and multiple seeds (Sections 4.1–4.2).  
  - External generalization is measured on GSM8K, MATH, Big-Bench Hard, and MMLU‑Pro using the LMEval harness (Section 4.3, Tables 3–4).

- A helpful mental model:
  - Think of RG as a “reasoning game arcade.” Each cabinet (dataset) can generate endless new levels (instances) at adjustable difficulty. Each cabinet also has a built-in referee (verifier) that instantly tells you if the final move sequence is valid.

## 4. Key Insights and Innovations
1. Unified, verifiable, and procedurally generated reasoning suite across many domains (Section 2; Table 6).
   - What’s new: breadth plus verifiers plus difficulty control, enabling RLVR at scale with near-infinite non-repeating data. Prior work tends to be fixed datasets or narrow procedural sets.  
   - Why it matters: removes data scarcity and memorization pitfalls; creates a controllable landscape for curricula and transfer studies.

2. Difficulty as a first-class, tunable axis with documented easy/hard configs (Appendix A.3) and “difficulty cliff” analysis (Figure 3b).
   - What’s new: standardized knobs for problem size/structure across many domains, enabling rigorous study of capability drop-offs.
   - Why it matters: reveals that current reasoning models often rely on brittle templates; performance collapses when complexity increases.

3. Demonstration of transfer via RLVR:
   - Intra-domain transfer: RLVR within a domain improves held-out tasks from the same domain (Table 1).  
   - Cross-domain transfer: RLVR on algorithms boosts algebra and geometry; RLVR on logic boosts cognition and graphs (Table 2).  
   - Why it matters: evidence that RLVR on procedurally generated data teaches reusable reasoning skills, not just task-specific tricks.

4. External benchmark gains from RG-trained models (Tables 3–4).
   - What’s new: training on purely synthetic, verifiable tasks improves real-world benchmarks without additional curated data.  
   - Why it matters: validates practicality; shows RG can serve as a pretraining curriculum for downstream reasoning tasks.

These are fundamental contributions in infrastructure and methodology (not a new RL algorithm), enabling new research workflows and analyses.

## 5. Experimental Analysis
- Evaluation methodology:
  - Zero-shot ability of frontier LLMs on RG easy/hard (Sections 3, A.4–A.5): reasoning-optimized vs general-purpose models are compared on per-task accuracy (% of problems solved by the verifier).
  - RLVR training studies (Section 4):
    - Intra-domain: train on multiple tasks in a category; test on a held-out task from the same category (50 problems; 3 runs).  
    - Cross-domain: train on one category; test on different domains (50 problems; 3 runs).  
    - Rewards tracked include accuracy + format reward (Section 4; Figure 4–5 captions).  
  - External generalization: GSM8K (8-shot CoT), MATH (0-shot CoT), Big-Bench Hard (3-shot CoT) using the LM Evaluation Harness (Table 3) and MMLU‑Pro subsets (Table 4).

- Main quantitative findings:
  - Zero-shot performance gaps (Figure 3a):
    > Top reasoning models average: `o3-mini` 63.51%, `DeepSeek-R1` 59.52%, `Grok 3 Mini` 55.06%.  
    > Strong general-purpose models are lower: `Llama 4 Maverick` 41.50%, `Claude 3.5 Sonnet` 40.33%, `Gemma 3 27B` 20.26%.
  - Difficulty cliff (Figure 3b): moving from easy→hard yields large drops. For `o3-mini`, declines are −71.93% (code), −33.80% (graphs), −33.13% (geometry), −25.57% (algorithmic). `DeepSeek‑R1` shows −61.82% (code), −29.60% (graphs), −11.83% (geometry), −27.85% (algorithmic).
    - Interpretation: capability is shallow; template-based rather than robust reasoning.
  - Intra-domain RLVR (Table 1, Acc@3):
    > Algebra: 5.0 → 16.7 (+11.7).  
    > Algorithmic: 52.3 → 59.7 (+7.4).  
    > Arithmetic: 89.7 → 96.0 (+6.3).  
    > Cognition: 40.3 → 42.3 (+2.0).  
    > Games: 0.0 → 3.3 (+3.3).
  - Cross-domain RLVR (Table 2, Acc@3):
    > Training on Algorithmic improves Algebra to 52.89 (+29.1) and Geometry to 23.17 (+22.3).  
    > Training on Logic improves Cognition to 24.94 (+13.3) and Graphs to 28.86 (+9.1).  
    > Training on Games yields gains on Algebra (+21.8) and Cognition (+13.1) despite modest in-domain ability.
  - External benchmarks (Table 3):
    > `RG-Math` improves MATH 48.5 → 58.2 (+9.7) and Big-Bench Hard 8.68 → 16.34 (+7.66) with a small GSM8K uptick 76.2 → 76.7 (+0.5).
  - MMLU‑Pro subsets (Table 4):
    > `RG-Math` improves Math (+5.62), Physics (+5.70), Computer Science (+4.40), Biology (+4.19), etc.  
    > `RG-Algorithmic` also helps: Psychology (+4.26), CS (+2.93), Engineering (+3.20), Graph-centric topics (indirectly via logic) see gains.

- Curriculum learning vs non-curriculum (Section 5; Figure 6; Table 5):
  - Setup: difficulty escalates when moving average performance exceeds 70% for 20 steps; compared to uniform sampling across all levels.  
  - Results (Table 5):
    > Spell Backwards (word length 4): 30.00 → 70.67 (+40.67).  
    > Mini Sudoku (8–10 empty cells): 6.67 → 20.00 (+13.33).  
    > Count Primes (100–500): 4.00 → 30.67 (+26.67).  
  - Dynamics (Figure 6): difficulty bumps cause transient reward drops, indicating the model faces genuinely harder instances; curriculum reaches high levels faster and finishes stronger on the hardest levels.

- Do the experiments support the claims?
  - The zero-shot study convincingly shows the gap between reasoning-optimized and general-purpose models and quantifies the difficulty cliff (Figures 2–3; A.4–A.5).  
  - RLVR transfer studies use multiple seeds and fixed test sets; the gains are consistent within and across domains (Tables 1–2; Figures 4–5).  
  - External benchmark improvements (Table 3–4) demonstrate practical utility beyond RG tasks, especially in MATH/BBH and knowledge-heavy MMLU‑Pro categories.

- Notes on metrics and design:
  - “Acc@3” in Tables 1–2 indicates credit if a correct solution appears within the top-3 samples; this mirrors stochastic generation in RLVR and reflects practical inference-time sampling.  
  - Reported compute (~1500 A6000 hours; Section 4) and open configs (A.6) improve reproducibility and transparency.

- Failure modes and hard areas:
  - Text-encoded spatial reasoning and long-horizon puzzles (e.g., `rush_hour`, `rubiks_cube`, `rotten_oranges`) remain challenging even for top models on hard settings (Figure 8 caption).  
  - Code-heavy tasks exhibit the steepest difficulty cliffs (Figure 3b).

## 6. Limitations and Trade-offs
- Domain coverage and open-endedness (Section 7):
  - Procedural generators are best for problems with clear, checkable endpoints; creative writing or open-form proofs are out-of-scope.
  - Some domains need extensive world knowledge that is hard to encode with verifiers.

- Reward fidelity (Section 7):
  - Verifiers ensure correctness but may miss nuances of solution quality. For example, a solver might use brittle heuristics that pass checks without demonstrating robust reasoning (“reward hacking” risk is reduced by P2—large solution spaces—but not eliminated).

- Interaction modality (Section 7):
  - RG currently focuses on single-turn, text-only problems; multi-turn agent tasks and multimodal reasoning are not included yet.

- Training distribution and forgetting (Section 7):
  - Experiments sample uniformly within tasks; continual learning over non-stationary streams and techniques to mitigate catastrophic forgetting are left for future work.

- Compute and scale:
  - While training runs are light compared to frontier model post-training, many experiments and curricula still require non-trivial GPU time (~1500 A6000 hours; Section 4).

- Evaluation breadth:
  - External benchmark improvements are strong on MATH/BBH and several MMLU‑Pro areas but modest on GSM8K (+0.5; Table 3), suggesting some skills transfer better than others.

## 7. Implications and Future Directions
- How this changes the landscape:
  - RG provides a standardized, extensible source of verifiable, infinite, difficulty-controlled reasoning tasks. This decouples reasoning improvement from finite, contamination-prone datasets and enables systematic curricula and capability diagnostics (Figures 3b, 6).

- Follow-up research enabled:
  - Curriculum and self-play curricula:
    - The documented difficulty knobs (A.3) and curriculum scheduler (Section 5) invite automated curricula methods (e.g., regret-based environment design, active curricula).
  - Multi-turn/multimodal reasoning:
    - Extending RG with dialogue-based tasks or visual inputs would bridge to agent and VLM training (Section 7).
  - Robustness and reward design:
    - Using adversarial instance generation to probe reward hacking, and richer verifiers that capture more nuanced solution qualities.
  - Continual learning:
    - Investigations into replay, regularization, and model merging to avoid forgetting across large rotating task menus (Section 7).
  - Cross-domain transfer theory:
    - Why algorithmic training improves algebra/geometry (Table 2) is a fertile area for mechanistic interpretability and curriculum theory.

- Practical applications:
  - Training open-weight reasoning models with limited budgets (the Qwen2.5‑3B case) to achieve meaningful gains on math, logic, and knowledge exams (Tables 3–4).
  - Continuous evaluation pipelines for production reasoning systems: organizations can tune RG parameters to track capability growth, regressions, and brittleness under increasing difficulty.

In short, RG is primarily an infrastructure and empirical methodology contribution: it systematizes the creation and verification of diverse reasoning tasks and demonstrates that RLVR on such procedurally generated curricula yields measurable, transferable reasoning improvements—while exposing the fragility of current models when complexity rises.
