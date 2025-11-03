# Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?

**ArXiv:** [2504.13837](https://arxiv.org/abs/2504.13837)
**Authors:** Yang Yue, Zhiqi Chen, Rui Lu, Andrew Zhao, Zhaokai Wang, Shiji Song, Gao Huang
**Institutions:** Tsinghua University, LeapLab, Shanghai Jiao Tong University

## 🎯 Pitch

This paper introduces a formal framework to evaluate 'empirical support' in Reinforcement Learning with Verifiable Rewards (RLVR), revealing that while RLVR boosts single-shot accuracy, it narrows the model's solution diversity, confining it to previously known outputs. This insight is crucial for future RL strategies, as it highlights the need for exploration beyond refining existing knowledge, ultimately enhancing the model's reasoning capacity and reliability across diverse tasks.

---

## 1. Executive Summary (2–3 sentences)
This paper investigates whether current Reinforcement Learning with Verifiable Rewards (`RLVR`) truly expands a model’s reasoning capacity or mostly concentrates probability on answers the base model already knows. It introduces a formal and empirical framework for “empirical support” (the set of correct solutions a model can realistically sample) and shows across many models and benchmarks that RLVR reliably improves single-shot accuracy but typically shrinks the set of accessible correct solutions, revealing an “invisible leash” to the base model’s distribution (Fig. 1, Table 1, Sec. 3.2; Appx. C).

## 2. Context and Motivation
- Problem addressed
  - Large reasoning models (e.g., `DeepSeek-R1`, `OpenAI-o1`) increasingly use RLVR—RL with automatically checkable rewards (e.g., math answer is correct/incorrect)—to improve reasoning. Yet there is an open question: does RLVR enable discovery of genuinely new reasoning paths, or does it mostly reweight and sharpen the base model’s existing correct outputs? (Sec. 1)
  - A community-observed pattern fuels the debate: RLVR models beat their bases at one-shot accuracy (pass@1) but sometimes fall behind when allowed many samples (pass@k for large k). This suggests possible loss of breadth even as precision increases (Sec. 1, 3.2).

- Why it matters
  - Practical: If RLVR narrows solution diversity, systems may miss correct but rare answers, hurting reliability at scale (e.g., high-sample search, diverse tasks).
  - Theoretical: Understanding whether on-policy RLVR can expand support clarifies what kinds of algorithms are needed to push reasoning beyond a model’s initialization (Appx. C).

- What existed before and gaps
  - Prior RLVR recipes (e.g., PPO/GRPO variants) and recent scaling studies show empirical gains, but do not disentangle precision from coverage. The field lacked a clear notion and measurement of “which correct solutions remain reachable” after RLVR.
  - Observations of improvement from spurious or proxy rewards, and of pass@k reversals, raised concerns about what RLVR is actually learning, but without a common framework to quantify support changes (Sec. 1).

- Positioning of this work
  - Provides: 
    - A formal definition of “empirical support” for correct completions under finite sampling (Sec. 2.1), 
    - A taxonomy and metrics for how RLVR changes access to correct solutions (Sec. 2.2),
    - Large-scale measurements across models (1.5B–14B) and tasks (math, logic, code, science, and a VLM) (Sec. 3.1–3.2),
    - Theory explaining why standard RLVR is “support-bounded” (Appx. C),
    - An entropy analysis showing local uncertainty can rise even as global diversity collapses (Sec. 4).

## 3. Technical Approach
- Core objects and setup (Sec. 2.1)
  - A prompt `x` maps to a distribution over completions `y`.
  - `q(y|x)`: base model distribution.  
  - `R(x,y) ∈ {0,1}`: verifiable reward (1 if completion is correct).
  - `π_θ(y|x)`: RLVR-trained policy. Standard RLVR maximizes expected reward with a divergence penalty to the base, e.g.:
    - maximize over θ: E_{x∼D, y∼π_θ}[ R(x,y) − β^{-1} log(π_θ(y|x)/q(y|x)) ] (Sec. 2.1), which encourages reward while discouraging moving too far from `q`.

- Why an “empirical support” notion is needed (Sec. 2.1)
  - In theory softmax makes every `y` have nonzero probability. In practice, tokens with extremely tiny probability are never sampled.  
  - Define an `ϵ`-thresholded “empirical support” for correct completions:
    - `supp_ϵ(p) := { y ∈ C | p(y|x) > ϵ }`, where `C` is the set of correct completions (Def. 2.1).  
    - Intuition: this is the subset of correct answers that are realistically discoverable under a finite sampling budget.
  - The paper estimates a principled `ϵ` from sampling bounds: if a correct `y*` is never seen in `k` samples, then with confidence `1−ζ`, `p(y*|x) ≤ −log ζ / k` (Appx. C.4). Example: with `k=8192, ζ=0.05`, `ϵ ≈ 3.66e−4`.

- How support changes are categorized (Def. 2.2; Fig. 1)
  - For a given `ϵ`, each correct completion falls into one of four regions:
    - `Preservation (P)`: correct and above-ϵ in both `q` and `π_θ`.
    - `Shrinkage (S)`: correct and above-ϵ in `q` but falls below-ϵ in `π_θ`.
    - `Expansion (E)`: correct and below-ϵ in `q` but rises above-ϵ in `π_θ`.
    - `Out of Support (O)`: below-ϵ in both.
  - Support metrics (Def. 2.3) quantify retention vs discovery:
    - `SRR = P/(P+S)` (retention),
    - `NDR = E/(P+E)` (share of accessible solutions that are genuine new discoveries),
    - `SDS` (harmonic mean of SRR and NDR),
    - `NSCR = (E−S)/(P+E+S)` (net expansion vs shrinkage).

- Theoretical results: why standard RLVR is “support-bounded” (Appx. C)
  - On-policy support preservation (Thm. C.1): with typical policy-gradient updates, `supp(π_θ) ⊆ supp(q)`. Intuition: if `q(y*|x)=0`, on-policy sampling never observes `y*`, so gradients cannot assign it mass.
  - Asymptotic pass@k ceiling (Cor. C.2): as `k→∞`, `pass@k(π_θ) ≤ pass@k(q)` because any correct completion reachable by `π_θ` must also be reachable by `q`.
  - Empirical support preservation under finite detectability (Thm. C.3): even with `ϵ`-thresholding, `supp_ϵ(π_θ) ⊆ supp_ϵ(q)` under finite rollout budgets; a gradient argument shows NSR (negative sample reinforcement) cannot substantially inflate probability of `ϵ`-invisible sequences in finite steps.
  - Variational/KL view (Prop. C.4): the RLVR-optimal `π*` is an exponential tilt of `q`:  
    - `π*(y|x) ∝ q(y|x) · exp(β R(x,y))`.  
    - In the KL-free limit (`β→∞`), `π*` renormalizes `q` over only correct responses (`C`) without inventing new modes (Cor. C.5).  
    - Bottom line: RLVR sharpens around reward-consistent regions already supported by `q`, rather than exploring new ones.

- Entropy lens to separate “local stochasticity” from “global diversity” (Sec. 4.1)
  - Token-level entropy: average uncertainty over next-token distributions along generated traces.  
  - Answer-level entropy: entropy over the set of final answers extracted from multiple samples (captures global diversity over end results).  
  - Computation: generate `k=32` samples per problem, compute token entropies by teacher-forcing the generated sequences, and compute answer entropy over the unique extracted answers (Sec. 4.1; Appx. B.4).

- Experimental design highlights (Sec. 3.1; Appx. B)
  - RL method emphasized in analysis: `ProRL-1.5B` (built on `DeepSeek-R1-Distill-Qwen-1.5B`), training with GRPO plus regularization and exploration stabilizers (decoupled clipping, dynamic sampling, KL reg, periodic reference resets).
  - Models evaluated: 1.5B–14B RLVR models (Nemotron, AceReason, Skywork-OR1, Phi4-Reason-Plus) and a VLM (`Kangheng-OVR-7B`) (Table 1; Appx. A).
  - Benchmarks:
    - Math: AIME2024/2025, AMC 2023, MATH500, Minerva, OlympiadBench (Sec. 3.1).
    - Non-math: SimpleQA, LiveBench (reasoning/coding/language subsets), SciBench, Reasoning Gym (Sec. 3.1; Appx. B.2).
  - Sampling budgets: math k∈{4096, 8192}; Reasoning Gym k up to 16384; others k∈{1024, 2048} (Sec. 3.1).
  - Inference: vLLM, temperature 0.6, top-p 0.95, max 32768 tokens (Appx. B.1).
  - Answer normalization: a careful, task-specific extraction pipeline is used to avoid format-induced bias, especially for base models on Reasoning Gym (Appx. B.3).

## 4. Key Insights and Innovations
- A concrete framework for “empirical support” and its dynamics (Sec. 2; Fig. 1)
  - Novelty: Moves beyond aggregate accuracy to measure which correct solutions remain practically reachable after RLVR. Definitions of preservation/expansion/shrinkage/out-of-support with `ϵ`-thresholding, plus four metrics (SRR, NDR, SDS, NSCR).
  - Significance: Enables precise diagnosis of whether RLVR discovers new correct modes or prunes existing ones.

- Strong, cross-domain evidence that current RLVR is support-constrained (Sec. 3.2; Table 1; Figs. 2–4)
  - Finding: retention is high, genuine discovery is rare, and shrinkage exceeds expansion across models and domains.
  - Significance: Explains the “pass@k reversal” pattern—RLVR improves pass@1 by sharpening around familiar correct modes but loses breadth for larger k.

- Theory that formalizes the “invisible leash” (Appx. C)
  - Theorems show on-policy RLVR cannot assign mass to truly unseen solutions; variational view explains exponential tilting around the base distribution.
  - Significance: Grounds the empirical findings in clear mathematical limits, clarifying why naive scaling of current RLVR recipes may not unlock fundamentally new reasoning.

- Entropy decoupling: local uncertainty up, global diversity down (Sec. 4; Table 3)
  - Finding: token-level entropy can increase (more stepwise uncertainty), while answer-level entropy reliably falls (fewer distinct end answers).
  - Significance: Demonstrates that “stochastic-looking” generation does not guarantee exploration of new final solutions—useful guidance for interpreting RLVR training dynamics.

## 5. Experimental Analysis
- Datasets, metrics, and setup (Sec. 3.1; Appx. B)
  - Evaluations span math reasoning, logic, factual QA, code, science, and a vision-language math subset. The main metric for coverage is pass@k; support dynamics are measured with `ϵ` derived from the sampling budget (Appx. C.4).
  - Perplexity analyses use external reasoning traces (DeepSeek-R1 and Claude Sonnet 4) to probe whether RLVR can assign mass to external solution styles (Table 2).

- Main quantitative results
  - Aggregate support dynamics (Table 1):
    - High retention, little discovery:
      > “Overall SRR ≈ 0.93–0.99; NDR ≤ 0.04 across all models and domains.”  
      For example, `PRORL-1.5B-V1 Overall`:  
      > `SRR=0.94, NDR=0.02, SDS=0.03, NSCR=−0.05; P=2400, E=36, S=163, O=805`.
      Large models behave similarly, e.g., `Nemotron-1-14B Overall`:
      > `SRR=0.99, NDR=0.00, SDS=0.01, NSCR=−0.01; P=2418, E=8, S=23, O=501`.
    - Shrinkage > expansion:
      > `PRORL-1.5B-V2 Overall`: `E=48` vs `S=175` (≈3.6× more shrinkage than expansion; Table 5).

  - Pass@k patterns and task-level curves (Sec. 3.2; Figs. 2–4; Tables 4–9):
    - Typical preservation/improved early-k convergence (Fig. 2): e.g., `graph_color`, `palindrome_generation`, `advanced_geometry`. RLVR reaches high pass@k quickly, indicating effective sharpening around already-known modes.
    - Occasional expansion (Fig. 3): `boxnet`, `dice`, `arc_1d` show genuine new access, but these are rare relative to shrinkages documented elsewhere.
    - Clear shrinkage cases (Fig. 4): tasks like `leg_counting`, `family_relationships`, `power_function` where base models continue to improve with larger k, but RLVR plateaus earlier.
    - Concrete high-k reversal: On AIME 2024 with k=8192 (Table 4),
      > Base: 93.3% vs ProRL-1.5B: 83.3%,  
      even though RLVR often improves lower-k performance.

  - Entropy and accuracy at medium budget (k=32; Table 3):
    - Accuracy gains with RLVR:
      > `DeepSeek-1.5B Avg@32 = 54.5%` → `ProRL-1.5B = 65.4%`.  
      > `Qwen2.5-32B = 43.0%` → `DAPO-32B = 61.3%`.
    - Token-level entropy: mixed trends
      > `DeepSeek-1.5B` 0.44 → `ProRL-1.5B` 0.52 (increase),  
      > but `DeepSeek-7B` 0.37 → `AceReason-7B` 0.23 and `Skywork-OR1-7B` 0.16 (decreases).
    - Answer-level entropy: consistent decreases
      > `DeepSeek-1.5B` 1.30 → `ProRL-1.5B` 0.66,  
      > `Qwen2.5-32B` 1.61 → `DAPO-32B` 0.61,  
      > similar reductions across 7B/14B cases.
    - Interpretation (Sec. 4.2): higher precision (avg@32) coexists with narrower final answer diversity, supporting the support-sharpening view.

  - Perplexity against external solution styles (Table 2):
    - Where base solves and RLVR fails (“shrinkage”), RLVR has higher perplexity on base’s reasoning traces, e.g., AIME2024:
      > Base trace: Base 1.36 vs ProRL 1.60.
    - On problems neither solves, RLVR assigns lower probability to external traces:
      > Claude Sonnet 4 traces on AIME2024: Base 8.76 vs ProRL 14.91 (higher perplexity is worse alignment to those traces).
    - Interpretation: RLVR is less compatible with solution modes outside the base model’s supported styles.

- Do the experiments support the claims?
  - Breadth of models and tasks, plus consistent SRR/NDR/NSCR patterns (Table 1) and pass@k curves (Figs. 2–4), convincingly show predominant preservation, limited expansion, and net shrinkage.
  - Entropy and perplexity analyses triangulate the mechanism: sharpening around rewarded modes already in the base distribution (Sec. 4; Table 2–3).
  - Implementation controls: careful answer extraction avoids format artifacts (Appx. B.3).

- Notable nuances and robustness
  - Expansion exists but is rare (e.g., `boxnet`, `arc_1d` in Fig. 3; small E counts in Table 1).
  - Math tasks show particularly low discovery (NDR ≈ 0–0.01, Table 1), while some non-math subsets display slightly higher, yet still small, NDR (≤0.04).
  - Even with strong SRR at larger scales, NSCR remains slightly negative, indicating persistent net shrinkage (Table 1).

## 6. Limitations and Trade-offs
- Assumptions that limit expansion
  - On-policy sampling: gradients only update from what is sampled; if `q(y*|x)=0`, `π_θ` cannot discover `y*` (Thm. C.1).
  - KL/tilting view: `π* ∝ q · exp(βR)` preserves the base’s support and relative structure within reward-consistent regions (Prop. C.4, Cor. C.5).

- Metric-level constraints
  - `pass@k` measures “solution retrievability under sampling” rather than deep conceptual novelty. The paper uses it as a practical proxy but acknowledges its limits as a capability measure (Sec. 1).

- Thresholding and detectability
  - Empirical support uses an `ϵ` derived from sampling budgets (Appx. C.4). While statistically principled, conclusions hinge on realistic-but-finite k; different budgets shift what counts as “reachable.”

- Style vs structure confounds in perplexity
  - Perplexity on external traces reflects both distributional support and stylistic alignment (Table 2). Thus, it is a supportive, not decisive, indicator of support mismatch.

- Compute and coverage
  - Very large k are costly; some tasks and models (e.g., code LCB v5/v6 limited to ≤7B) are constrained for efficiency (Appx. B.2). Unexplored regimes (bigger models, different RLVR variants, or domains) may behave differently.

- No ablation on exploration mechanisms
  - The goal is to characterize current practice, not to propose and test exploration-augmented fixes. Hence, the paper diagnoses limits but does not empirically validate new algorithms to overcome them.

## 7. Implications and Future Directions
- How this changes the landscape
  - The “invisible leash” concept reframes mainstream RLVR as a precision enhancer within the base model’s reach rather than a capability expander (Sec. 5). This clarifies why pass@1 rises yet high-k breadth can fall.
  - The entropy perspective cautions against equating noisy stepwise generation with meaningful exploration: token-level entropy can increase while answer-level entropy collapses (Sec. 4.2; Table 3).

- Algorithmic directions to “break the leash”
  - Incorporate explicit exploration beyond on-policy sharpening:
    - Off-policy or mixed-policy sampling to seed mass in low-density, underrepresented correct regions (Sec. 5; Abstract).
    - Diversity-promoting objectives (e.g., constraints on answer-level entropy or reweighting rare-but-correct trajectories).
    - Techniques that “reward the unlikely” or allocate targeted credit to long-tail tokens/paths (Appx. C discussion references; see also Sec. 5).
    - Hybrid training that interleaves RLVR with data seeding or curriculum on rare solution modes.
  - Develop better capability metrics:
    - Complement pass@k with measures that detect conceptual novelty and reasoning breadth, not just retrieval probability.

- Practical applications and guidelines
  - When deploying RLVR for safety-critical or search-heavy workflows (code, theorem proving, planning), maintain or restore exploration at inference time (e.g., multi-seed sampling, diverse decoding) to mitigate support shrinkage.
  - Monitor support dynamics during training:
    - Track SRR/NDR/NSCR alongside accuracy to detect undesirable pruning of valid solutions.
    - Use answer-level entropy as an early warning for mode collapse even if token-level entropy appears healthy (Table 3).

- Research opportunities
  - Theoretical: stronger guarantees for exploration under verifiable rewards, support-aware RL objectives, and bounds that couple reward, KL, and coverage.
  - Empirical: systematic studies of exploration augmentations (off-policy buffers, bootstrapped policies, stochastic value guidance) across domains; controlled comparisons separating stylistic vs structural support changes (follow-ups to Table 2).

In sum, the paper provides a principled framework, large-scale evidence, and theory showing that current RLVR predominantly sharpens within the base model’s effective region—improving precision while often shrinking the set of accessible correct solutions. To grow genuine reasoning capacity, future RLVR must move beyond on-policy distribution sharpening toward exploration- and diversity-aware training.
