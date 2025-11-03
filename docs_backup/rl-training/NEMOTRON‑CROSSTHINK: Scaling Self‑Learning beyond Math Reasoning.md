# NEMOTRON‑CROSSTHINK: Scaling Self‑Learning beyond Math Reasoning

**ArXiv:** [2504.13941](https://arxiv.org/abs/2504.13941)
**Authors:** Syeda Nahida Akter, Shrimai Prabhumoye, Matvei Novikov, Seungju Han, Ying Lin, Evelina Bakhturi, Eric Nyberg, Yejin Choi, Mostofa Patwary, Mohammad Shoeybi, Bryan Catanzaro
**Institutions:** NVIDIA ADLR (Applied Deep Learning Research), Carnegie Mellon University (Eric Nyberg), University of Washington (Yejin Choi)

## 🎯 Pitch

NEMOTRON-CROSSTHINK revolutionizes reinforcement learning for large language models by expanding their reasoning capabilities beyond mathematics into diverse domains through multi-domain data curation and strategic blending. This approach not only yields significant accuracy improvements across benchmarks but also enhances efficiency by producing concise, verifiable answers, reshaping practical applications of LLMs in fields such as law, education, and enterprise knowledge work.

---

## 1. Executive Summary (2-3 sentences)
NEMOTRON-CROSSTHINK is a reinforcement-learning (RL) framework that scales “self-learning” for large language models (LLMs) beyond math into general-purpose reasoning by curating multi-domain data, constraining answer formats to enable verifiable rewards, and blending data sources strategically. It achieves consistent gains across both math and non-math benchmarks and produces more concise correct answers, demonstrating not just higher accuracy but more efficient reasoning (Figures 1, 3; Tables 4, 7; Appendix B).

## 2. Context and Motivation
- Problem addressed
  - RL has recently boosted LLM reasoning, but most progress concentrates on mathematics and coding, where correctness is easy to verify. Extending RL to broader reasoning domains is hard because many tasks lack clearly verifiable answers, data is heterogeneous (multiple-choice vs open-ended), and reward design becomes fragile (Section 1).
- Why it matters
  - Real-world reasoning spans law, history, economics, social sciences, and STEM. Improving RL beyond math would yield LLMs that generalize across domains and are more useful in real applications (Section 1).
- Shortcomings of prior approaches
  - Math-centric pipelines: Most RL efforts rely on math datasets where exact correctness is computable, overlooking non-math reasoning (Section 1; Related Work, Section 6).
  - Limited exploration of data mixing: Recent works pull data from multiple sources but do not analyze the relative utility of each source nor optimize data blending strategies (Section 1).
  - Format blindness: Mixing MCQ and open-ended questions without addressing answer-space variation complicates reward design and can introduce reward hacking (Sections 1–2).
- Positioning of this work
  - A systematic framework to: (1) curate multi-domain data (both synthetic from web and open-source QA), (2) apply templates that constrain answer spaces for verifiable rewards, (3) filter hard-to-verify samples, and (4) optimize multi-source data blends for RL (Figure 2; Sections 2–3). The work explicitly studies how each design choice alters performance and efficiency, including ablations on question/answer formats and difficulty-based filtering (Sections 4–5; Tables 5–7).

## 3. Technical Approach
The framework consists of five linked steps; Figure 2 provides the schematic.

- Step 1: Data curation and synthesis (Section 2: “Data Curation”)
  - Two sources: `D = D_syn ∪ D_os`.
    - Synthetic (`D_syn`) from Common Crawl (CC): generate QA spanning many domains.
    - Open-source (`D_os`): collect established datasets.
  - Two domain categories within each source:
    - General-purpose reasoning (`D_gpr`): 
      - Open-source: MMLU [Train], NaturalReasoning (NR).
      - Synthetic: Syn-QA (generated from CC guided by MMLU categories).
    - Mathematical reasoning (`D_mr`):
      - Open-source: MATH, NuminaMath.
      - Synthetic: PersonaSkill-MATH (via persona-based problem generation; Ge et al., 2024).
  - Scale: 588,645 total QA items. Breakdown (Table 1):
    - GPR: MMLU [Train] 99,842 (MCQ), Syn-QA 192,930 (MCQ), NaturalReasoning 100,000 (open-ended).
    - Math: NuminaMath 87,350, PersonaSkill-MATH 100,000, MATH 8,523 (all open-ended).

- Step 2: Templates to control answer space and elicit diverse reasoning (Section 2: “Applying Templates...”)
  - Motivation: MCQ and open-ended questions elicit different reasoning styles; unconstrained outputs hinder verifiable rewards.
  - Two question templates:
    - `T_MCQ`: multiple-choice.
    - `T_Open`: open-ended (for GPR, they also convert some MCQ to open-ended by removing options).
  - Construction:
    - `D_mcq = T_MCQ(D_gpr)`.
    - `D_open = T_Open(D_gpr)`.
    - Some MCQs become invalid if options are removed (e.g., “Which of the following...?”). Those are discarded to avoid ambiguous open-ended prompts (Section 2).

- Step 3: Filtering and formatting to ensure verifiable rewards (Section 2: “Data Filtering and Formatting”)
  - Simple, rule-based filters `H(D)` to keep only samples that can be reliably autograded:
    - For MCQ: ensure the correct answer `a*` is within the provided options `{a1...an}`; if not, discard.
    - For open-ended: restrict answers to ≤ 10 words to keep exact-match evaluation tractable.
    - For math: ensure answers exist (no empty `a*`).
  - This reduces noise and increases the fraction of samples compatible with a simple reward (Section 2, filtering rule set).

- Step 4: Data blending strategies (Section 2: “Data Blending”; Table 2; Appendix A Table 8)
  - Goal: Study whether and how mixing domains and formats helps.
  - Blends include:
    - Natural distribution `B_nd`: proportional to dataset sizes.
    - Domain-weighted: `B_mr↑` (2:1 math:GPR), `B_gpr↑` (2:1 GPR:math).
    - Format-weighted: `B_mcq↑` (2:1 MCQ:Open), `B_open↑` (2:1 Open:MCQ).
    - Usefulness-weighted: `B_score` (weights per dataset’s average downstream scores from Table 3).
  - Single-domain controls:
    - `B_only_mr`: only math datasets.
    - `B_only_gpr`: only GPR datasets.

- Step 5: Self-learning with RL using GRPO (Section 2: “Reinforcement Learning with GRPO” and “Rule Based Reward Modeling”; Equation (1))
  - GRPO (Group Relative Policy Optimization): an RL algorithm that avoids a separate critic; instead, it samples a group of outputs for each prompt and uses group statistics as a baseline.
    - Procedure per question `q`:
      - Sample a group of generations `{o_i}` from the old policy `π_θ_old`.
      - Compute a reward `r_i` for each generation.
      - Standardize advantages within the group: `Â_i,t = (r_i − mean) / std` across the group’s rewards.
      - Optimize a clipped policy objective (PPO-style) with a KL penalty toward a reference model to stabilize learning (Equation (1)).
    - Intuition: The model learns to prefer better outputs relative to other outputs for the same prompt, without needing a separate value network.
  - Reward design is purely rule-based to ensure scalability and reproducibility:
    - Total reward uses logical AND:
      > R = R_acc ∧ R_format
    - `R_acc`: exact correctness. For MCQ, match the correct option; for short-form open-ended, exact string match.
    - `R_format`: enforce output structure with `<think>...</think>` for scratchpad reasoning and `\boxed{...}` for the final answer (Section 2).
  - Training setup (Section 3: “Training Details”):
    - Base models: `Qwen2.5-7B` and `Qwen2.5-32B`.
    - Framework: veRL (open-source implementation of HybridFlow RLHF).
    - Hyperparameters: learning rate 1e-6; batch size 128; 8 rollouts per prompt; max context 5,000 tokens; temperature=1.0, top-p=1.0; KL coefficient 0.001. No extensive hyperparameter tuning is reported.

- Difficulty-based filtering for larger model (Section 5: “Difficulty Filtering”; Table 7)
  - For `Qwen-2.5-32B`, they create a “hard-only” filtered blend `B_f(gpr)↑` by retaining only those questions that the smaller `Qwen-2.5-7B` fails in zero-shot. This approximates difficulty without external labels and focuses training on harder samples.

- Token efficiency analysis (Ablations, Section 5; Figure 3; Appendix B)
  - Measure token counts for correct vs incorrect responses across blends to study whether models learn to be concise when possible.
  - Also analyze adaptive verbosity across math vs non-math tasks (Appendix B Table 9 and Figure 4).

Why these choices?
- Constraining answer formats and filtering lets them keep a simple, scalable reward while expanding beyond math into domains that lack deterministic grading.
- Blending domains and formats intentionally stresses cross-domain generalization and lets them quantify trade-offs.
- GRPO simplifies RL training by removing a critic and using group-relative advantages, making multi-sample rollouts per prompt feasible.

## 4. Key Insights and Innovations
- Multi-domain RL with verifiable rewards at scale
  - What’s new: A full pipeline that makes non-math RL practical by combining data curation, answer-space templating, and simple rule-based rewards (Figure 2; Sections 2–3).
  - Why it matters: It enables consistent generalization across domains without sophisticated reward models, addressing a core bottleneck beyond math.
  - Evidence: Multi-domain blends outperform math-only training on non-math benchmarks while remaining competitive on math (Table 4, Figure 1).

- Data blending reveals cross-domain transfer, not just data volume
  - What’s new: A systematic comparison of blends by source, format, and usefulness (Table 2; Appendix A Table 8).
  - Why it matters: Shows that “only math is not enough.” The best overall results come from favoring general-purpose reasoning data with math included (2:1 ratio), indicating transfer from non-math to math and vice versa (Table 4).
  - Evidence: `B_gpr↑` (2:1 GPR:math) achieves the highest average (58.12%) vs baseline (44.75%) and math-only (57.82%) on `Qwen2.5-7B` (Table 4).

- Simple template choices materially affect RL outcomes
  - Unified open-ended questions improve performance over mixed MCQ/Open by +1.21% on average (Table 5).
  - Short-form answers (option label only) beat long-form (label + description) by +1.20% on average, likely because strict string-matching rewards penalize benign paraphrases (Table 6).

- Efficiency as an emergent property of multi-domain RL
  - The best multi-domain blend (`B_gpr↑`) learns to be concise on general reasoning tasks and detailed on math—using 28% fewer tokens for correct answers overall than math-only training (Figure 3; Appendix B Table 9).
  - Incorrect answers are ~3.6× longer than correct ones across tasks, suggesting verbosity correlates with uncertainty rather than accuracy (Appendix B Figure 4).

- Difficulty-based filtering scales with model size
  - Training `Qwen-2.5-32B` on only “hard” examples yields a further +2.15% average improvement over the unfiltered multi-domain RL model (Table 7).

## 5. Experimental Analysis
- Evaluation methodology (Section 3: “Evaluation Metrics”)
  - Benchmarks:
    - Math: MATH-500, AMC23.
    - General reasoning: MMLU (test), MMLU-PRO, AGIEVAL, GPQA-DIAMOND, SUPERGPQA.
      - MMLU-PRO and GPQA-DIAMOND emphasize graduate-level, “Google-proof” reasoning; SUPERGPQA spans 285 graduate disciplines, including long-tail domains (Section 3).
  - Metric: Accuracy (exact match with the answer in `\boxed{...}`), averaged over 3 runs with greedy decoding.
  - Input/output formatting: For MCQ evaluation, the target includes both the correct option label and its text to align with training; for open-ended, strict exact-match is used.

- Main quantitative results
  - Headline improvements (Abstract; Figure 1): 
    > On non-math benchmarks, accuracies improve by +12.8% (MMLU-PRO), +11.3% (GPQA-DIAMOND), +15.1% (AGIEVAL), +3.8% (SUPERGPQA); on math, +30.1% (MATH-500) and +27.5% (AMC23).
  - Blending study on `Qwen2.5-7B` (Table 4):
    - Baseline average: 44.75%.
    - Natural distribution `B_nd`: 55.66% (+10.91 over baseline), showing pure data scale helps.
    - Best overall: `B_gpr↑` 58.12% (+13.37). It outperforms `ORZ-7B` (55.20%) by ~3 points on average and is stronger on non-math tasks (Figure 1; Table 4).
      > On MMLU-PRO: 57.82% vs baseline 45.00% (+12.82).  
      > On AGIEVAL: 63.71% vs 48.59% (+15.12).  
      > On GPQA-Diamond: 38.58% vs 31.82% (+6.76).
    - Math-only `B_only_mr`: 57.82% (second best average). It slightly edges `B_gpr↑` on math tasks but underperforms it on several non-math tasks (Table 4).
    - Format balances: `B_open↑` (57.49%) beats `B_mcq↑` (56.89%), suggesting open-ended exposure aids generalization—even on math tasks (Table 4).
    - Usefulness-weighted `B_score`: 56.95%. Helpful but inferior to domain-aware blends because raw averages overweight math datasets and underweight non-math contributors (Table 4).
  - Individual dataset RL (Table 3; 250 steps each for uniform comparison):
    - `NuminaMath` yields the best average (53.06%), boosting not just math but also several non-math tasks—a strong cross-domain signal.
    - `Syn-QA` improves over baseline on challenging non-math benchmarks like MMLU-PRO and AGIEVAL.
    - `MMLU [Train]` alone underperforms on most tasks, except SUPERGPQA where its breadth helps (Table 3 discussion).
  - Template ablations (Tables 5–6):
    - All-open-ended questions outperform mixed MCQ+open by +1.21% on average (Table 5). This likely reduces reward hacking via option guessing (e.g., 25% random accuracy in 4-choice MCQ).
    - Short answers (option label only) beat long answers (label + description) by +1.20% (Table 6), avoiding noisy penalties from exact string matching on paraphrased descriptions.
  - Token efficiency (Figure 3; Appendix B Table 9):
    - On MMLU, the average token count for correct answers drops from 351 (`B_only_mr`) to 229 (`B_gpr↑`). Across tasks, `B_gpr↑` uses 28% fewer tokens for correct answers than `B_only_mr` and far fewer than `ORZ` (Figure 3; Appendix B Table 9).
    - Models adapt verbosity to task type: `B_gpr↑` increases average length by 62% from GPR to math tasks, while `B_only_mr` increases only 14%, showing less adaptive behavior (Appendix B Table 9).
    - Incorrect answers are ~3.6× longer than correct ones, indicating verbosity correlates with uncertainty rather than correctness (Appendix B Figure 4).
  - Difficulty filtering at scale (Table 7; `Qwen2.5-32B`):
    - Base 32B: 54.33% average.
    - RL on unfiltered `B_gpr↑`: 65.84% (+11.51).
    - RL on filtered `B_f(gpr)↑`: 67.99% (+13.66).  
      > Notable jumps on: MMLU-PRO 69.43% (vs 68.83%), GPQA-DIAMOND 49.75% (vs 46.70%), AGIEVAL 75.82% (vs 73.90%), AMC23 75.00% (vs 67.50%).

- Do the experiments support the claims?
  - Yes, through a layered set of ablations:
    - Blends vs baselines (Table 4); single-domain controls (`B_only_mr`, `B_only_gpr`).
    - Per-dataset RL (Table 3) informing usefulness blending.
    - Question/answer template ablations (Tables 5–6).
    - Token efficiency analysis (Figure 3; Appendix B Table 9 and Figure 4).
    - Difficulty filtering at larger scale (Table 7).
  - Trade-offs are transparent: math-only excels narrowly on math, but multi-domain is superior overall and nearly matches math performance (Table 4; Figures 5–7 in Appendix C).

- Notable nuance and trade-offs
  - `B_only_mr` attains the best math-only average yet still loses global average to `B_gpr↑`. Multi-domain training offers better overall value, especially on non-math (Table 4).
  - `B_score` shows that “weight by average score” is too coarse; task-aware mixing matters (Table 4 and discussion).

## 6. Limitations and Trade-offs
- Reward design constraints (Section 2)
  - Exact-match `R_acc` favors short answers and penalizes benign paraphrases, which is why short-form answers help (Table 6).
  - Open-ended answers are restricted to ≤10 words during training data filtering to keep evaluation verifiable; this excludes tasks needing long-form generation or nuanced explanations.
  - The formatting requirement (`<think>...` and `\boxed{...}`) is rigid. Models must learn a specific output schema; downstream applications may need postprocessing or prompting to align formats.
- Data curation choices
  - Converting MCQs to open-ended by removing options can invalidate some items; the paper discards such cases, potentially narrowing diversity (Section 2).
  - Synthetic data from CC plus dataset mixes may introduce domain or stylistic biases that are not fully audited.
  - Using MMLU [Train] in RL may raise concerns about topical overlap with test sets if not carefully managed (the paper evaluates on MMLU test, but overlap risks are not quantified).
- Difficulty filtering proxy
  - “Hard” is defined as “missed by a 7B model,” which reflects that model’s weaknesses and may not generalize to other model families or objectives (Section 5; Table 7).
- Compute considerations
  - RL with 8 rollouts per prompt, large context windows (5,000 tokens), and hundreds of thousands of questions is compute-intensive (Section 3). GRPO reduces complexity by removing the critic, but the overall training remains expensive.
- Generality of findings
  - Experiments use Qwen2.5 models; results may vary for other base models or tokenizer conventions. Little hyperparameter tuning was performed (Section 3), so absolute numbers may shift with further optimization.

## 7. Implications and Future Directions
- How this changes the field
  - It shows a practical path to scale RL for reasoning beyond math by engineering the data and output space—not by relying on complex reward models. This reframes “RL for reasoning” as a data-and-format problem as much as an algorithm problem (Figure 2; Sections 2–3).
  - It highlights the value of multi-domain exposure for both accuracy and efficiency, with adaptive verbosity emerging as a desirable behavior (Figure 3; Appendix B).

- What it enables next
  - Smarter reward functions: Replace exact string matching with semantic matching or option-label extraction to support longer open-ended answers while preserving verifiability.
  - Automated blend optimization: Use bandit or bilevel optimization to learn dataset weights online, conditioned on target tasks, rather than static ratios like 2:1 (Table 2).
  - Better difficulty curricula: Learn difficulty estimates directly (e.g., via uncertainty, variance of sampled returns) rather than proxying with a smaller model’s accuracy (Table 7).
  - Broaden formats: Expand from short-form answers to structured, multi-point justifications using verifiers that check steps or constraints, enabling complex non-math tasks like legal analysis or scientific QA.
  - Cross-lingual and multimodal extensions: The templating-and-verifier approach could be adapted to multilingual corpora or vision-language reasoning with tool-verified rewards.

- Practical applications
  - Education and testing: Models trained with `B_gpr↑`-style blends perform strongly on challenging, exam-like benchmarks (MMLU-PRO, AGIEVAL, GPQA-Diamond; Table 4) and do so more efficiently—useful for tutoring systems and assessment.
  - Enterprise knowledge work: The consistent improvements on SUPERGPQA (long-tail professional domains; Section 3) suggest better generalization to diverse real-world tasks.
  - Cost-aware assistants: The 28% reduction in tokens for correct answers (Figure 3; Appendix B) directly lowers inference costs for production deployments without sacrificing accuracy.

Overall, NEMOTRON-CROSSTHINK demonstrates that careful data engineering—multi-domain curation, answer-space templating, verifiability-driven filtering, and principled blending—lets RL move beyond math into broader reasoning, with measurable gains in both accuracy and efficiency (Figures 1–3; Tables 4–7).
