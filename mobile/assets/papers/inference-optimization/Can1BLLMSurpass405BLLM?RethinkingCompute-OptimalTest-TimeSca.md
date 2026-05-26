# Can 1B LLM Surpass 405B LLM? Rethinking Compute-Optimal Test-Time Scaling

**ArXiv:** [2502.06703](https://arxiv.org/abs/2502.06703)

## 🎯 Pitch

This paper demonstrates that with carefully chosen, reward-aware test-time scaling (TTS) strategies, small language models (as small as 1B–3B parameters) can outperform much larger models—even those over 100 times bigger—on advanced mathematical reasoning tasks. By systematically analyzing how the optimal use of inference-time computation depends on the interplay between model size, verifier (PRM), and problem difficulty, the authors show that efficient, tailored TTS can dramatically boost reasoning accuracy while reducing computational cost by up to 1000×, reshaping the path to cost-effective and scalable AI systems.

---

## 1. Executive Summary

This paper empirically analyzes how to optimally allocate test-time computation across diverse policy models, Process Reward Models (PRMs), and problem difficulty levels, demonstrating that no single **compute-optimal TTS strategy** is universal — the optimal method (best-of-N sampling, beam search, or Diverse Verifier Tree Search) depends critically on the specific policy model, the specific PRM used as verifier, and the absolute difficulty of each prompt. Through comprehensive experiments on MATH-500 and the more challenging AIME24 benchmark using policy models from 0.5B to 72B parameters (Llama and Qwen2.5 families) and seven PRMs spanning 1.5B to 72B parameters, the paper establishes that a compute-optimal TTS strategy enables extremely small models to surpass much larger ones — a 1B model outperforms a 405B model on MATH-500, a 0.5B model beats GPT-4o on both benchmarks, and a 7B model surpasses o1 and DeepSeek-R1 — while finding that TTS gains are most pronounced for weaker policy models and simpler tasks, establishing that test-time compute can amplify existing capability but its effectiveness diminishes as base model reasoning strength increases.

## 2. Context and Motivation

### The Core Problem: Test-Time Scaling Strategies Are Poorly Understood Across Their Design Space

The fundamental premise this paper investigates is that **the optimal way to spend inference-time compute is not a universal constant** — it changes depending on *which* policy model generates solutions, *which* Process Reward Model (PRM) verifies those solutions, and *how difficult* the specific problem is. While prior work has established that test-time scaling (TTS) can improve LLM reasoning performance, the community lacks a systematic understanding of how these three factors interact to determine which TTS method works best, when, and why.

The paper frames this gap explicitly in Section 1:

> "There is limited systematic analysis of how policy models, PRMs, and problem difficulty influence these TTS strategies. This limitation prevents the community from fully understanding the effectiveness of this method and developing insights for compute-optimal TTS strategies."

This gap matters for several practical reasons that the paper addresses directly:

- **Deployment efficiency tradeoffs**: If the optimal TTS strategy depends on the policy model, then organizations deploying different-sized models (for cost, latency, or hardware reasons) need to adapt their inference strategies accordingly — not simply apply a one-size-fits-all approach like best-of-N sampling. A small on-device model might benefit more from guided search than a large cloud model, and this paper provides evidence for exactly that pattern.

- **PRM selection in practice**: Open-source PRMs are proliferating rapidly — Math-Shepherd, RLHFlow, Skywork, Qwen2.5-Math — each trained on different base models with different data generation pipelines. Practitioners face the question: which PRM should I use with my policy model? The paper demonstrates that the answer is non-obvious because PRMs exhibit strong model-specific biases and generalization failures across policy models, a finding with immediate practical implications for anyone building TTS pipelines.

- **Understanding the capability ceiling**: If TTS can make a 1B model outperform a 405B model, what are the limits? Does this substitution hold across all difficulty levels? Across all tasks? The paper investigates precisely these boundaries, providing guidance on *where* TTS can substitute for pretraining compute and *where* it cannot — essential knowledge for allocating finite resources between training larger models and scaling inference.

### Why This Problem Is Important

The significance of this work spans both theoretical understanding and practical deployment considerations.

**Theoretical significance.** Test-time compute scaling represents one of two major paradigms for improving LLM reasoning — the other being internal TTS through long chain-of-thought training (as exemplified by OpenAI o1 and DeepSeek-R1). Understanding the scaling properties of external TTS methods is therefore a fundamental question about the nature of LLM reasoning: can search and verification compensate for weaker base models, or are some reasoning capabilities only acquirable through training? This paper provides empirical evidence on both sides: TTS can dramatically improve weak models on problems they can already *sometimes* solve, but hits diminishing returns — and ultimately a ceiling — on problems fundamentally outside the model's capability range.

**Practical deployment significance.** The headline results — a 0.5B model beating GPT-4o, a 7B model surpassing o1 and DeepSeek-R1 — have direct economic implications. These models differ in parameter count by factors of 100× to 1000×, which translates to dramatically different hardware requirements, inference costs, and deployment feasibility. If a 7B model with compute-optimal TTS can match a 671B reasoning model on specific benchmarks, it opens the door to deploying competitive mathematical reasoning capabilities on consumer hardware. The FLOPs analysis in Table 4 quantifies exactly this: small policy models "reduce the total FLOPs by 100× ∼ 1000×" compared to their larger competitors.

**Methodology significance.** The paper introduces a **reward-aware** formulation of compute-optimal TTS (Equation 3) that generalizes the prior formulation from Snell et al. (2024) by explicitly conditioning the optimal strategy on the reward function ℛ, not just the policy model and prompt. This is more than a notational change — it reflects the empirical finding that swapping PRMs can invert which strategy is optimal, and that PRMs trained on different base models exhibit qualitatively different biases (length bias, step-level scoring bias) that fundamentally alter the search dynamics.

### Where Prior Approaches Fall Short

The paper identifies specific limitations in existing TTS research along several dimensions:

**1. Narrow scope of policy model evaluation.** Snell et al. (2024), which established the compute-optimal TTS framework, conducted experiments primarily with PaLM 2-S\* as the policy model and used a PRM trained on the same model's outputs — what this paper calls an "on-policy PRM." While methodologically clean for establishing proof-of-concept, this setup avoids the messy reality that training a dedicated PRM for every policy model is computationally expensive. In practice, practitioners use off-the-shelf PRMs with policy models from entirely different families. The paper notes in Section 3.1:

> "For practical applications of compute-optimal TTS, training a PRM for each policy model to prevent OOD issues is computationally expensive. Therefore, we investigate the compute-optimal TTS strategy in a more general setting, where the PRM might be trained on a different policy model than the one used for TTS."

This is the realistic deployment scenario that prior work largely ignored, and the paper demonstrates that it leads to qualitatively different — and sometimes opposite — conclusions about which TTS strategy is optimal.

**2. Insufficient exploration of PRM diversity.** Prior TTS studies typically used a single PRM (Snell et al., 2024; Wu et al., 2024; Beeching et al., 2024). The question of how different PRMs influence TTS performance was essentially unasked. This paper evaluates seven PRMs spanning different sizes (1.5B to 72B), different base model families (Mistral, Llama, Qwen2.5-Math), and different training data generation procedures. The results reveal that PRM choice is arguably *the* most impactful decision in a TTS pipeline — the wrong PRM can make beam search perform worse than majority voting, while the right PRM enables it to substantially outperform best-of-N.

**3. The difficulty grouping problem.** Snell et al. (2024) binned problems into difficulty quintiles based on the policy model's pass@1 rate. This paper identifies a critical flaw: when the policy model is very capable, the quantile approach collapses. As shown in Figure 3, Qwen2.5-72B-Instruct achieves pass@1 above 80% on 76.2% of MATH-500 problems. Using quantile-based difficulty bins with such a model would compress most problems into the "easy" bin, obscuring meaningful difficulty-dependent patterns. The paper proposes absolute thresholds (easy: 50–100%, medium: 10–50%, hard: 0–10%) as a more robust alternative that works across policy models of vastly different capabilities.

**4. Task simplicity.** Prior TTS evaluations focused primarily on MATH-500. While MATH-500 is a standard benchmark, the paper argues that "recent LLMs show significant progress in mathematical reasoning" (Section 4.1), making MATH-500 increasingly saturated for strong models. The inclusion of AIME24 — a significantly harder competition dataset — allows the paper to test whether TTS benefits extend to problems where even the strongest models have low base accuracy. The answer, as the results show, is nuanced: TTS provides modest gains on AIME24 compared to MATH-500, suggesting that the effectiveness of TTS diminishes as problems become genuinely harder for the base model.

**5. Missing comparison to long-CoT methods.** The TTS literature had not systematically compared external TTS (search/verification with frozen models) against internal TTS (training models to produce long chain-of-thought reasoning). With the emergence of reasoning models like o1 and DeepSeek-R1, this comparison becomes crucial for understanding where each paradigm excels. The paper introduces this comparison in Section 5.3, showing that compute-optimal TTS can match or exceed some long-CoT-trained models (rStar-Math, Eurus-2, SimpleRL, Satori) but falls short of distillation from strong reasoning models (DeepSeek-R1-Distill-Qwen-7B), particularly on harder tasks.

### How This Paper Positions Itself

The paper positions itself as a **comprehensive empirical re-evaluation** of the compute-optimal TTS framework under realistic, heterogeneous conditions. Rather than proposing a fundamentally new method, it extends the framework established by Snell et al. (2024) along three axes:

**First, it generalizes the problem formulation.** The introduction of the reward-aware compute-optimal strategy (Equation 3) explicitly acknowledges that the reward function ℛ is a first-class input to the optimization, not an implementation detail. This formalizes what the experiments demonstrate empirically: that the choice of PRM changes the output distribution of search-based TTS methods, and therefore must be part of the optimization decision.

**Second, it dramatically expands the experimental matrix.** Where Snell et al. (2024) studied one policy model with one PRM on one dataset, this paper studies:
- **10 policy models** across two families (Llama 3: 1B, 3B, 8B; Qwen2.5: 0.5B, 1.5B, 3B, 7B, 14B, 32B, 72B)
- **7 PRMs** across three families and sizes from 1.5B to 72B
- **3 TTS methods** (Best-of-N, Beam Search, DVTS)
- **2 benchmarks** (MATH-500, AIME24)
- **3 difficulty levels** with an improved absolute-threshold grouping

This combinatorial breadth is the paper's primary empirical contribution — it reveals patterns (the dependence of optimal strategy on policy model size, the PRM length bias, the difficulty-strategy interaction) that are invisible in any single-row study.

**Third, it establishes new performance boundaries.** The paper explicitly frames its contributions around demonstrating "the significant potential of smaller language models to outperform larger models through TTS" (Section 1). The results in Table 3 and Figure 1 show concrete thresholds: a 0.5B model surpassing GPT-4o, a 3B model exceeding a 405B model, a 7B model beating o1 and DeepSeek-R1. These are not theoretical possibilities — they are measured results on standard benchmarks. The FLOPs analysis (Table 4) further quantifies the efficiency: achieving this performance while reducing total computational cost by two to three orders of magnitude.

**Fourth, it connects external TTS to the long-CoT paradigm.** By comparing compute-optimal TTS against models trained with long chain-of-thought (rStar-Math, Eurus-2, SimpleRL, Satori, DeepSeek-R1-Distill), the paper bridges what had been largely separate research threads. The finding that TTS outperforms RL/SFT-based long-CoT methods on MATH-500 but falls significantly behind on AIME24 suggests a fundamental difference in how these approaches scale with problem difficulty — external TTS is stronger on problems within the base model's reach, while internal TTS (via training) extends the reach itself.

**Positioning relative to Beeching et al. (2024).** The paper acknowledges Beeching et al. (2024) as direct prior work that introduced DVTS and demonstrated its effectiveness on easy/medium problems with large budgets. However, the paper argues that prior work lacked evaluation "with either strong verifiers or policies with different sizes / capabilities" (Section 6). The current paper is positioned as filling exactly this gap — adding diversity across policy model scales, PRM qualities, and task difficulties, all of which the experiments show dramatically affect the conclusions about which TTS method is optimal.

**The paper's implicit thesis.** Throughout the writing, there's an undercurrent of practical guidance: *don't assume one TTS strategy fits all — test across your specific policy model, PRM, and task distribution, because the interactions matter enormously*. The reward-aware formulation, the absolute difficulty thresholds, the systematic comparison of PRM generalization, and the identification of PRM biases (length sensitivity, over-criticism, error neglect, scoring bias in Appendix C) all serve this practical orientation. The paper is less a theoretical advance and more an empirical field guide for practitioners who need to make TTS work in heterogeneous deployment scenarios.

## 3. Technical Approach

### 3.1 Reader Orientation (Approachable Technical Breakdown)

This paper does not propose a fundamentally new algorithm or model architecture — it is an **empirical analysis paper** that systematically evaluates how existing test-time scaling (TTS) methods behave across a wide combinatorial space of policy models, Process Reward Models (PRMs), and problem difficulty levels, with the practical goal of determining **which TTS strategy to use, when, and why**. The system being analyzed is a decision procedure: given a math question, a base language model that generates candidate solutions, a separate verifier model that scores those solutions' reasoning steps, and a fixed budget of inference computation, the system must allocate that budget across competing strategies (parallel sampling, guided step-by-step search, or a hybrid branching approach) to maximize the probability of producing the correct final answer.

### 3.2 Big-Picture Architecture (Diagram in Words)

The TTS pipeline has four major components that interact in a straightforward sequential flow:

1. **Policy Model (the proposer):** A pre-trained, frozen LLM (Llama 3 or Qwen2.5 families, spanning 0.5B to 72B parameters) that takes a math problem description as input and generates candidate solution steps autoregressively. It is *not* fine-tuned for TTS — it remains exactly as released in its Instruct variant.

2. **Process Reward Model (the verifier):** A separate, frozen model (one of seven open-source PRMs spanning 1.5B to 72B parameters, from Math-Shepherd, RLHFlow, Skywork, or Qwen2.5-Math families) that takes a partial solution trajectory (the problem statement plus all steps generated so far) and outputs a scalar score estimating the probability that the partial solution will eventually lead to a correct final answer. The PRM is never the same model as the policy model — this paper deliberately studies the cross-model (off-policy) setting.

3. **TTS Strategy Engine (the search/selection procedure):** Three possible strategies map the compute budget into a concrete generation-and-scoring protocol:
   - **Best-of-N (BoN):** Generate N complete solutions independently, score each, vote to select the final answer.
   - **Beam Search:** Generate solutions step-by-step, using the PRM at each step to prune unpromising partial solutions and expand only the most promising ones.
   - **Diverse Verifier Tree Search (DVTS):** Split the budget into independent subtrees, run beam search within each, and then vote across subtrees to select the final answer.

4. **Difficulty Estimator (the policy selector):** Given a problem, estimate its difficulty (easy: 50–100% pass@1, medium: 10–50% pass@1, hard: 0–10% pass@1) by measuring the base policy model's probability of solving it correctly. The difficulty level determines which subset of strategy hyperparameters (TTS method, budget allocation, which PRM to use) is deployed.

Information flows as follows: **Problem text** → **Difficulty estimator** bins it into easy/medium/hard → **Compute-optimal policy** selects a specific TTS method based on the (policy model, PRM, difficulty) combination → **Policy model** generates candidate solutions under that method's protocol → **PRM** scores every intermediate step of every candidate → **Scoring and voting procedure** aggregates step-level scores into a single answer selection → **Final answer**.

### 3.3 Roadmap for the Deep Dive

- **First, the Problem Formulation as an MDP (Section 2.1 of the paper):** Understanding the mathematical framework that unifies all TTS methods — why reasoning is modeled as sequential decision-making, what states, actions, and rewards mean in this context, and how this framing enables a common vocabulary across Best-of-N, Beam Search, and DVTS.

- **Second, the Three TTS Strategies (Section 2.2 of the paper):** The mechanics of Best-of-N, Beam Search, and DVTS — what each does step-by-step, how the PRM's scores feed into search decisions versus post-hoc selection decisions, and what hyperparameters control their behavior.

- **Third, the Reward-Aware Compute-Optimal Formulation (Section 3.1 of the paper):** The core analytical contribution — how the paper extends Snell et al. (2024)'s compute-optimal framework to explicitly include the reward function as an input, why this matters (because different PRMs induce different output distributions for search-based methods), and the formal distinction between sampling-based methods (where the reward does NOT affect generation, only selection) and search-based methods (where the reward actively guides generation step-by-step).

- **Fourth, the Difficulty Grouping Criterion (Section 3.2 of the paper):** Why absolute difficulty thresholds replace quantile-based binning, the empirical motivation (Figure 3 showing Qwen2.5-72B-Instruct's overwhelming concentration in high pass@1), and the practical mechanics of how problems are assigned to easy/medium/hard levels.

- **Fifth, the Experimental Configuration Space (Section 4.1 of the paper):** The concrete choices — which policy models, which PRMs, which scoring and voting methods, which compute budgets, which generation parameters — and why each choice was made, including the crucial distinction between on-policy and off-policy PRM usage.

### 3.4 Detailed, Sentence-Based Technical Breakdown

This is an **empirical analysis paper** whose core idea is that the optimal test-time compute allocation strategy is not universal — it depends on a three-way interaction between the policy model, the PRM, and the problem difficulty, and ignoring any of these dimensions leads to suboptimal, sometimes catastrophically wrong, strategy selection.

---

#### The MDP Formulation: Reasoning as Sequential Decision-Making

The paper formalizes mathematical reasoning as a Markov Decision Process (MDP) defined by the tuple `$(\mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma)$`. This framing is not novel — it follows standard RL vocabulary (Sutton and Barto, 2018) and has been used in prior TTS work — but it provides the shared mathematical language that makes all three TTS methods commensurable.

**States.** The state space `$\mathcal{S}$` consists of all possible partial solution prefixes. The initial state `$s_1$` is the problem text `$x$` — a string containing the math question description. After each reasoning step, the state updates by concatenation: if the current state is `$s_t$` and the model generates action `$a_t$`, the new state is `$s_{t+1} = [s_t, a_t]$`, where `$[\cdot, \cdot]$` denotes string concatenation. Thus, a state at time `$t$` is the problem text followed by all previously generated reasoning steps: `$s_t = [x, a_1, a_2, \ldots, a_{t-1}]$`.

**Actions.** The action space `$\mathcal{A}$` is the set of all possible text completions — specifically, one reasoning step at a time. An action `$a_t$` is a string generated by the policy model conditioned on the current state: `$a_t \sim \pi_\theta(\cdot \mid s_t)$`. The paper defines a "step" by the delimiter `\n\n` (two newlines), matching the convention used in PRM training data (Xiong et al., 2024; Zhang et al., 2025). This is an important implementation detail: the PRM's step-level scoring depends on where steps are split, and consistency between training and inference step boundaries is necessary for the PRM's scores to be meaningful.

**Transitions.** The transition function `$\mathcal{P}$` is deterministic: `$s_{t+1} = \mathcal{P}(\cdot \mid s_t, a_t) = [s_t, a_t]$`. There is no stochastic environment — all randomness comes from the policy model's sampling (controlled by temperature) and the selection decisions made by the TTS strategy.

**Rewards.** The reward function `$\mathcal{R}: \mathcal{S} \times \mathcal{A} \to \mathbb{R}$` is implemented by the PRM. At each step `$t$`, the PRM observes the state-action pair `$(s_t, a_t)$` (equivalently, the partial solution up to and including step `$t$`) and outputs a scalar `$r_t = \mathcal{R}(s_t, a_t)$`. This scalar represents the PRM's estimate of the probability that this step, given all previous steps, is on a trajectory leading to a correct final answer. The reward is used either during search (to decide which partial solutions to expand) or after generation (to score and rank complete solutions).

**Termination and trajectories.** The process continues until an `<EOS>` token is generated or a maximum number of steps is reached. A complete trajectory of length `$H$` is denoted `$\tau = \{a_1, a_2, \ldots, a_H\}$`. The paper describes this formally in Equation (1):

> Initial State: `$s_1 = x \sim \mathcal{X}$`
> Action: `$a_t \sim \pi_\theta(\cdot \mid s_t)$`
> State Transition: `$s_{t+1} = \mathcal{P}(\cdot \mid s_t, a_t) = [s_t, a_t]$`
> Reward: `$r_t = \mathcal{R}(s_t, a_t)$`

**Why this formulation matters.** The MDP framing creates a unified interface that makes all three TTS methods directly comparable. In Best-of-N, the policy generates complete trajectories without using rewards during generation — rewards are only used post-hoc for selection, making it a pure "propose-then-verify" approach. In Beam Search, rewards actively prune the search space at each step — rewards are used during generation to decide which partial trajectories survive, making it a guided search approach. DVTS combines both: rewards guide search within subtrees (like beam search) AND are used post-hoc across subtrees (like Best-of-N). The MDP formulation makes it clear exactly when and how the reward function `$\mathcal{R}$` enters the decision process, which is the conceptual foundation for why different TTS methods behave differently with different PRMs — if the PRM is used during search (not just after), its biases directly shape which candidate solutions are generated in the first place, not just which one is selected.

A critical algorithmic detail: the temperature differs between methods. For Chain-of-Thought (CoT) and Best-of-N, temperature is set to 0.7 — providing diversity needed for effective sampling-based selection. For search-based methods (Beam Search and DVTS), the policy model is sampled with temperature 0.7 during generation, but the PRM's evaluation of each step is a deterministic forward pass (the PRM is not a generative model; it outputs a scalar, not a distribution).

---

#### Best-of-N (BoN)

Best-of-N is the simplest TTS strategy and serves as the primary baseline throughout the paper. It does not use the PRM during generation — it only uses the PRM after all candidates are generated, to select the best answer.

**Generation.** Given a compute budget `$N$`, the policy model independently samples `$N$` complete solutions from the problem description. Each solution is a full trajectory `$\tau_i$` for `$i = 1, \ldots, N$`, generated autoregressively at temperature 0.7 with a maximum token limit of 8192 new tokens. The solutions are generated completely independently — there is no interaction between them and no feedback from the PRM during generation. This makes BoN embarrassingly parallel: all `$N$` generations can be produced simultaneously.

**Scoring.** After all `$N$` solutions are generated, the PRM scores each step of each solution. For a trajectory `$\tau$` of length `$H$`, the PRM produces a sequence of per-step rewards `$\{r_1, r_2, \ldots, r_H\}$` where `$r_t = \mathcal{R}(s_t, a_t)$`. The paper considers three methods for aggregating these per-step scores into a single trajectory-level score:

- **PRM-Min:** Takes the minimum reward across all steps: `$\text{score} = \min\{r_1, r_2, \ldots, r_H\}$`. The intuition is that a solution is only as good as its weakest step — if any step is scored poorly, the entire solution is suspect. This was the recommended method in Lightman et al. (2024).

- **PRM-Last:** Takes only the reward of the final step: `$\text{score} = r_H$`. The intuition is that the PRM's assessment at the end of the solution, when it has seen all reasoning context, is the most informative about whether the solution as a whole is correct. This corresponds to using the PRM essentially as an Outcome Reward Model (ORM) at aggregation time, but benefiting from the step-level training signal during PRM training.

- **PRM-Avg:** Takes the arithmetic mean of all per-step rewards: `$\text{score} = \frac{1}{H} \sum_{t=1}^H r_t$`. This treats all steps as equally informative about the solution's correctness.

The paper evaluates all three scoring methods across different PRMs in Table 2, but does not prescriptively recommend one. The choice of scoring method interacts with PRM-specific biases — a PRM that assigns spuriously high scores to certain step types (e.g., short steps with simple arithmetic) will have those biases propagate differently depending on the aggregation.

**Voting (answer selection).** Once each trajectory has a score, the system must select a single final answer. The paper considers three voting methods:

- **Majority Vote:** Ignore the PRM scores entirely and select the answer that appears most frequently among the `$N$` solutions. This is the simplest possible selection method, using only the policy model's consensus signal, and serves as the lower-bound baseline for PRM-guided methods.

- **PRM-Max:** Select the single solution with the highest score and output its answer. This trusts the PRM's absolute score calibration — if the PRM is well-calibrated, the highest-scoring solution should be correct.

- **PRM-Vote:** Group all solutions that produce the same final answer, sum the scores of all solutions in each group, and select the answer with the highest total summed score. This is the method recommended in prior work (Li et al., 2023; Snell et al., 2024) because it incorporates a consensus signal: even if no single solution scores exceptionally high, if many solutions agree on the same answer, their accumulated scores can outweigh a single high-scoring (but potentially incorrect) outlier.

**When BoN is optimal.** The paper finds that BoN tends to be the best method when (a) the policy model is large and capable — strong models don't need step-by-step guidance because they rarely take wrong steps — and (b) when the PRM is poorly matched to the policy model, because in that case, the PRM's guidance during search is actively harmful. For example, Figure 7 shows that for Qwen2.5-72B-Instruct on MATH-500, BoN is the best method across all difficulty levels. For smaller policy models (0.5B–3B), search-based methods outperform BoN because these models benefit from step-level verification.

---

#### Beam Search

Beam search introduces step-by-step PRM guidance: rather than generating complete solutions first and scoring later, beam search evaluates partial solutions at each step and prunes unpromising ones before committing more compute.

**Parameters.** Beam search is governed by two hyperparameters:
- **Beam width `$N$`:** The total compute budget, measured as the number of generations at each expansion step. At each step, exactly `$N$` candidate next-steps are generated.
- **Beam size `$M$`:** The branching factor — how many candidate next-steps are generated per surviving beam. The paper sets `$M = 4$` in all beam search experiments (Section 4.1).

The relationship between beam width and beam size determines how many beams survive at each step: after generating `$N$` candidate steps, the system keeps only the top `$N/M$` highest-scoring beams. With `$M = 4$`, this means `$N/4$` beams survive each pruning round.

**Step-by-step procedure.** Beam search operates as follows:

1. **First step generation:** The policy model samples `$N$` candidate first steps (`$a_1^1, a_1^2, \ldots, a_1^N$`) from the initial state `$s_1 = x$` (the problem text alone). Each candidate is a string of text representing the first reasoning step, terminated by the step delimiter `\n\n` or by reaching the per-step token limit of 2048.

2. **First step scoring:** The PRM evaluates each candidate first step in the context of the problem: `$r_1^i = \mathcal{R}(s_1, a_1^i)$` for `$i = 1, \ldots, N$`. Each candidate receives a scalar score.

3. **Beam pruning:** The `$N$` candidates are ranked by their PRM scores, and only the top `$N/M = N/4$` highest-scoring candidates survive. The rest are discarded permanently. This is the critical decision point — the PRM's quality at this step determines which partial solutions the entire remaining budget will be spent on.

4. **Step expansion:** For each surviving beam, the policy model generates `$M = 4$` candidate next steps, conditional on the beam's partial solution so far. Specifically, for a beam with current state `$s_t = [x, a_1, \ldots, a_{t-1}]$`, the policy model samples `$M$` candidates for the next step: `$a_t^{i,1}, a_t^{i,2}, a_t^{i,3}, a_t^{i,4} \sim \pi_\theta(\cdot \mid s_t)$`. Since there are `$N/M$` surviving beams and each generates `$M$` candidates, the total number of candidate next-steps is `$(N/M) \times M = N$` — exactly the budget.

5. **Iteration:** Steps 2–4 repeat for each subsequent reasoning step. At each iteration, the PRM scores all new candidates, the top `$N/M$` survive, and each survivor expands `$M$` children. This continues until either an `<EOS>` token is generated (indicating the solution is complete) or the maximum number of steps (40 rounds of expansion) is reached.

6. **Final selection:** At termination, all completed solutions are collected — up to `$N$` of them, since each surviving beam eventually produces a complete answer. Best-of-N weighted voting (PRM-Vote) is then applied to select the final answer.

**Token limits.** The paper imposes two token limits during beam search: a per-step limit of 2048 tokens (each generated step cannot exceed this) and a total response limit of 8192 tokens (the sum of all steps in a solution cannot exceed this). These limits prevent unbounded generation — a step that consistently generates very long reasoning will be truncated, and the total solution length is bounded.

**When beam search is optimal.** The paper finds that beam search tends to outperform BoN for small policy models (0.5B–7B parameters) because these models have weaker reasoning capabilities and are more likely to produce incorrect intermediate steps. The PRM's step-level pruning helps catch errors early before they propagate through the solution. However, beam search becomes harmful when (a) the policy model is strong enough that its steps are rarely wrong (the PRM adds more noise than signal) or (b) the PRM is poorly matched to the policy model's output distribution (the PRM systematically miscategorizes correct steps as wrong, or vice versa).

---

#### Diverse Verifier Tree Search (DVTS)

DVTS, introduced by Beeching et al. (2024), extends beam search by introducing parallelism across independent search trees. The key insight is that a single beam search tree can get trapped in local optima — if the PRM mistakenly scores a suboptimal but superficially plausible partial solution highly, all subsequent exploration is confined to that subtree. DVTS mitigates this risk by splitting the budget across multiple independent trees.

**Mechanics.** Given a total compute budget `$N$` and beam size `$M$`, DVTS divides the search into `$N/M$` independent subtrees. Each subtree is explored using beam search with a budget of `$M$` per step (so `$M$` candidates are generated at each expansion, and `$M/M = 1$` beam survives — effectively, it becomes a greedy best-first search within each subtree). The subtrees are completely independent — they start from different first steps and explore different regions of the solution space without any communication between them.

At the end, each subtree produces its best completed solution. The final answer is selected via majority voting or PRM-Vote across all subtrees. The diversity comes from the independence of the subtrees: even if one subtree gets trapped in a bad region due to an early PRM error, other subtrees explore different solution paths and can recover.

**Relationship to beam search.** DVTS is essentially a hybrid between BoN and beam search. Like BoN, it generates multiple independent solution paths (the subtrees). Like beam search, it uses PRM guidance within each path (each subtree uses greedy best-first selection). Beeching et al. (2024) showed that DVTS "outperforms beam search on easy and medium problems with a large computational budget `$N$`", and Chen et al. (2024) found that "increasing the number of parallel subtrees proves to be more effective than increasing the beam width under the same budget." These findings suggest that diversity across trees is more valuable than depth within a single tree, at least for problems where the PRM signal is somewhat reliable.

**When DVTS is optimal.** The paper's results in Figure 7, Figure 8, and Figure 9 show that DVTS performs well for medium-sized policy models (7B–32B parameters) on easy and medium problems. For these models, the PRM provides useful guidance (so beam search helps), but the risk of getting trapped in a single suboptimal subtree is real (so diversity across subtrees helps). For the largest model (72B), BoN without any search guidance outperforms DVTS because the model is strong enough to produce correct solutions without step-level verification. For the smallest models (0.5B–3B), beam search outperforms DVTS because these models need the deepest possible search within the most promising path — splitting the budget across subtrees dilutes the search depth.

---

#### The Reward-Aware Compute-Optimal Formulation

This is the paper's primary analytical contribution. The prior formulation from Snell et al. (2024) defined the compute-optimal TTS strategy purely in terms of the policy model and the problem:

$$\theta^*_{x, y^*(x)}(N) = \arg\max_\theta \left( \mathbb{E}_{y \sim \text{Target}(\theta, N, x)} \left[ \mathbb{1}_{y = y^*(x)} \right] \right)$$

where `$\theta$` represents the hyperparameters of the TTS strategy (which method, what beam width, what budget allocation), `$N$` is the total compute budget, `$x$` is the prompt, `$\text{Target}(\theta, N, x)$` is the distribution over outputs induced by the policy model under strategy `$\theta$` with budget `$N$`, `$y^*(x)$` is the ground-truth correct answer, and `$\mathbb{1}_{y = y^*(x)}$` is the indicator that the generated answer matches the correct answer.

**What it computes:** For a given problem and compute budget, this equation finds the TTS strategy hyperparameters `$\theta$` that maximize the expected accuracy — the probability that the selected final answer matches the ground truth. The expectation is taken over the randomness in the policy model's sampling and the PRM's selection process.

**Why this form is insufficient.** The critical omission is that `$\text{Target}(\theta, N, x)$` — the distribution over generated outputs — is assumed to depend only on the policy model and the strategy hyperparameters. But this ignores the PRM entirely. For search-based methods (Beam Search and DVTS), the PRM's scores determine which partial solutions survive at each step, which means the PRM directly shapes *which outputs are generated in the first place*. If you swap PRMs, you get different output distributions even with the same policy model and same strategy — a phenomenon the paper demonstrates empirically (e.g., Figure 12 showing that RLHFlow-PRM-Mistral-8B leads to short incorrect answers while RLHFlow-PRM-Deepseek-8B leads to longer correct answers on the same problem with the same policy model).

**The paper's extension.** The reward-aware formulation explicitly includes the reward function `$\mathcal{R}$` in the optimization:

$$\theta^*_{x, y^*(x), \mathcal{R}}(N) = \arg\max_\theta \left( \mathbb{E}_{y \sim \text{Target}(\theta, N, x, \mathcal{R})} \left[ \mathbb{1}_{y = y^*(x)} \right] \right)$$

where `$\mathcal{R}$` is the specific PRM used as the verifier.

**What it computes:** The same objective as before — maximize expected accuracy — but now the target distribution `$\text{Target}(\theta, N, x, \mathcal{R})$` explicitly depends on the PRM. The paper makes a crucial distinction: "For sampling-based scaling methods, `$\text{Target}(\theta, N, x, \mathcal{R}) = \text{Target}(\theta, N, x)$`." In other words, for Best-of-N, the PRM does NOT affect the generation of candidates — only the selection among them — so the reward function drops out of the generation distribution and only matters in the selection/voting step. For search-based methods (Beam Search, DVTS), the PRM actively shapes generation through pruning, so `$\mathcal{R}$` is a nontrivial input to the distribution.

**Why this form.** The paper argues that this is necessary for practical TTS because "training a PRM for each policy model to prevent OOD issues is computationally expensive" (Section 3.1). In realistic deployments, practitioners use off-the-shelf PRMs with policy models from different families, which creates a "more general setting, where the PRM might be trained on a different policy model than the one used for TTS." The reward-aware formulation accommodates this by making `$\mathcal{R}$` an explicit, independent input — the optimal strategy can be different for the same policy model when paired with different PRMs. The empirical results validate this: Figure 4 shows that for Llama-3.1-8B-Instruct, BoN outperforms search when using Math-Shepherd or RLHFlow PRMs, but search outperforms BoN when using Skywork or Qwen2.5-Math PRMs. The optimal strategy inverts depending on the PRM, which the original formulation (without `$\mathcal{R}$`) cannot capture.

---

#### PRMs: Scoring, Voting, and the Influence of Rewards

**How PRMs score steps.** The paper does not train any PRMs — all seven PRMs are used off-the-shelf as released by their respective authors. However, understanding their scoring mechanism is essential for understanding TTS behavior. A PRM takes as input a partial solution (the problem text plus all steps generated so far, up to and including the step to be scored) and outputs a scalar — typically a value between 0 and 1 — representing the model's confidence that the solution is on a correct trajectory. The PRM is not a generative model; it is a classifier or regression model that produces a single number per step.

The step delimiter is critical: all PRMs in this study are trained with steps separated by `\n\n` (two newlines). The paper follows this convention (Section 4.1): "The division of steps follows the format \n\n as in prior works (Xiong et al., 2024; Zhang et al., 2025)." If steps were split differently at inference time — say, by sentence boundaries or by a different delimiter — the PRM's scores would be meaningless because the model was not trained to score that granularity of text segments.

**The three scoring methods — formal definitions.** For a trajectory of length `$H$` with per-step rewards `$\{r_1, r_2, \ldots, r_H\}$`:

- **PRM-Min:** `$\text{score}_{\text{min}} = \min\{r_1, r_2, \ldots, r_H\}$`. This is the most conservative scoring method — a single low-scoring step tanks the entire solution's score. It is appropriate when the PRM's false-negative rate (scoring correct steps as incorrect) is low, because you trust that a genuinely low score signals a genuine error.

- **PRM-Last:** `$\text{score}_{\text{last}} = r_H$`. This uses only the final step's score, effectively treating the PRM as an Outcome Reward Model. It is appropriate when the PRM's earlier-step scores are noisy but its final-step score (with full solution context) is well-calibrated.

- **PRM-Avg:** `$\text{score}_{\text{avg}} = \frac{1}{H} \sum_{t=1}^H r_t$`. This is a middle ground — it incorporates all steps' signals but averages out individual step noise. It is appropriate when the PRM's per-step scores are somewhat noisy but not systematically biased.

**The three voting methods — formal definitions.** Given `$N$` solutions with scores `$\{s_1, s_2, \ldots, s_N\}$` and corresponding answers `$\{a_1, a_2, \ldots, a_N\}$`:

- **Majority Vote:** `$\text{answer} = \arg\max_a \sum_{i: a_i = a} 1$`. Count occurrences of each answer; pick the most frequent. PRM scores are ignored entirely.

- **PRM-Max:** `$\text{answer} = a_{i^*}$` where `$i^* = \arg\max_i s_i$`. Pick the single highest-scored solution's answer. This is appropriate when the PRM is well-calibrated in an absolute sense — its highest score reliably corresponds to a correct answer.

- **PRM-Vote:** `$\text{answer} = \arg\max_a \sum_{i: a_i = a} s_i$`. Group by answer, sum scores within each group, pick the answer with the highest total. This combines the PRM's confidence signal with the policy model's consensus signal — an answer that many solutions agree on (even with mediocre individual scores) can beat a single high-scoring outlier.

Table 2 shows that Skywork-PRM-7B works better with PRM-Vote than with PRM-Max (87.0 vs. 84.4 with PRM-Last aggregation), while Qwen2.5-Math-PRM-7B is insensitive to the voting method (all methods yield 87.4–87.8). The paper attributes this to differences in PRM training data: "the training data of Qwen2.5-Math PRMs are processed with LLM-as-a-judge, which removes the wrong intermediate steps labeled as positive steps in the training data and makes the outputted large reward values more likely to be correct." In other words, Qwen2.5-Math PRMs are better calibrated at the high end of their score range, so even PRM-Max (trusting the single highest score) works well. Skywork-PRM-7B's scores are less individually reliable, so the consensus-based PRM-Vote provides needed robustness.

**PRM biases identified in the paper.** Appendix C catalogues four systematic failure modes observed across PRMs:

1. **Over-Criticism:** The PRM assigns low scores to mathematically correct steps (Figure 13). For example, a correct simplification step `$\sqrt{2 \times 11 \times 11} = \sqrt{2} \times \sqrt{11 \times 11} = 11\sqrt{2}$` receives a score of 0.53 — barely above random. This causes beam search to prune correct partial solutions, directing compute toward superficially higher-scoring (but potentially incorrect) alternatives.

2. **Error Neglect:** The PRM assigns relatively high scores to steps containing clear mathematical errors (Figures 14, 15). An incorrect statement that `$\angle EDF = 90^\circ$` in a triangle where the right angle is actually at `$E$` receives a score of 0.74, failing to flag a fundamental misunderstanding that propagates through the entire solution.

3. **Error Localization Bias:** The PRM assigns its lowest scores to steps that are NOT where the actual error occurs (Figure 16). A solution where the critical reasoning error happens early (misapplying the Angle Bisector Theorem) receives a score of 0.20 on the first step, but later steps with compounding errors receive scores of 0.66 and 0.92. The PRM correctly identifies that something is wrong (the early low score) but distributes its criticism across steps in a way that doesn't help locate the actual mistake.

4. **Scoring Bias (Length Sensitivity):** The PRM's scores are biased by the length of the step (Figures 17, 18). Two mathematically equivalent steps solving the same problem — one with 31 tokens, the other with 283 tokens — receive scores of 0.51 and 0.12 respectively, despite both being correct. The longer step is penalized simply for being verbose, which is an artifact of the PRM's training data distribution. This bias is especially problematic for beam search: the PRM will systematically prefer shorter (but not necessarily better) steps, steering the search toward terse, potentially incomplete reasoning.

These biases are not specific to out-of-distribution data — they "persist across both OOD datasets (e.g., the AIME24 dataset, which was not used during PRM training) and in-distribution data (e.g., the MATH dataset used to train the model)" (Appendix C). They represent fundamental limitations of current PRM technology that affect all TTS methods, but especially search-based methods where the PRM's scores directly control which solutions are explored.

---

#### The Difficulty Grouping Criterion

The paper revises the difficulty estimation approach from Snell et al. (2024), which used quantile-based binning of pass@1 rates. The problem, as explained in Section 3.2, is that quantile binning fails when the policy model's pass@1 distribution is heavily skewed — a strong model solves most problems easily, so quantiles would lump together problems of genuinely different difficulty (e.g., a problem solved 95% of the time and one solved 80% of the time both end up in the top quintile).

**Empirical motivation (Figure 3).** The paper shows that Qwen2.5-72B-Instruct achieves pass@1 above 80% on 76.2% of MATH-500 problems. The mean pass@1 is 0.82. In a quantile-based binning, this would compress the vast majority of problems into one or two bins, making it impossible to study difficulty-dependent patterns. The absolute threshold approach instead partitions the pass@1 range directly, independent of the distribution shape.

**The three difficulty levels.** The paper defines difficulty based on the policy model's pass@1 accuracy on each problem — the fraction of the model's sampled solutions that are correct:

- **Easy:** Problems with pass@1 between 50% and 100%. On these problems, the base policy model solves the problem correctly more often than not. The model "knows" how to solve the problem; the challenge is selecting the correct solution from among its candidates.

- **Medium:** Problems with pass@1 between 10% and 50%. The model can solve the problem sometimes — it's within its capability range — but not reliably. Extra computation has room to improve success rate by searching for the correct solution approach among many attempts.

- **Hard:** Problems with pass@1 between 0% and 10%. The model almost never solves these correctly. Extra computation is unlikely to help because there are essentially no correct solutions in the model's output distribution to find via search or verification.

This three-way split is coarser than the five-quintile binning in Snell et al. (2024) but is argued to be more robust across policy models of vastly different capabilities. It captures the essential structure: easy problems benefit from selection accuracy, medium problems benefit from search diversity, and hard problems are beyond the model's reach regardless of strategy.

**How difficulty is used.** The paper does not develop an automated difficulty estimator — it computes pass@1 directly from the base model's samples and bins problems accordingly. This is a post-hoc analysis tool, not a deployable system component. The difficulty analysis in Figure 8 and Figure 9 uses these bins to show *how* the optimal TTS strategy shifts with difficulty: BoN is better for easy problems (small models), beam search is better for harder problems (small models), DVTS works well for easy/medium problems (medium models), and BoN works best across all difficulty levels for the largest model (72B). This analysis informs the compute-optimal policy selections in Tables 3 and 5, where the best strategy is chosen per (policy model, difficulty) pair based on empirical results.

---

#### Experimental Configuration: Policy Models, PRMs, and Hyperparameters

**Policy models.** The paper evaluates 10 models from two families, all used in their Instruct (instruction-tuned) variants:

- **Llama 3 family (Dubey et al., 2024):** Llama-3.2-1B-Instruct, Llama-3.2-3B-Instruct, Llama-3.1-8B-Instruct. These represent a distinct model lineage with specific prompt formatting requirements (Table 7: a detailed system prompt with instructions for simple vs. complex problems).

- **Qwen2.5 family (Yang et al., 2024b):** Qwen2.5-0.5B-Instruct, Qwen2.5-1.5B-Instruct, Qwen2.5-3B-Instruct, Qwen2.5-7B-Instruct, Qwen2.5-14B-Instruct, Qwen2.5-32B-Instruct, Qwen2.5-72B-Instruct. The system prompt for Qwen2.5 is minimal: "Please reason step by step, and put your final answer within \boxed{}." (Table 8).

- **Additional models for specific comparisons:** DeepSeek-R1-Distill-Qwen-1.5B and DeepSeek-R1-Distill-Qwen-7B (for the long-CoT comparison in Section 5.3 and the large-model comparison in Table 3).

The choice to cover 0.5B to 72B parameters — a 144× range — is deliberate. It allows the paper to identify monotonic trends: as model size increases, the optimal TTS strategy shifts from search-based methods (beam search) to sampling-based methods (BoN), and the benefit of TTS over CoT diminishes (Table 5: CoT→TTS gain drops from 154.6% for 1B to 9.5% for 72B).

**Process Reward Models.** The paper evaluates seven PRMs spanning different sizes, base model families, and training methodologies:

1. **Math-Shepherd-PRM-7B (Wang et al., 2024b):** Trained on Mistral-7B using PRM data generated from Mistral-7B fine-tuned on MetaMath. This is an older, more limited PRM that serves as a lower baseline.
2. **RLHFlow-PRM-Mistral-8B (Xiong et al., 2024):** Trained on data from Mistral-7B fine-tuned on MetaMath; base model is Llama-3.1-8B-Instruct.
3. **RLHFlow-PRM-Deepseek-8B (Xiong et al., 2024):** Trained on data from deepseek-math-7b-instruct; base model is Llama-3.1-8B-Instruct. The training data for RLHFlow-Deepseek has longer average responses (333.1 tokens) and steps (58.4 tokens) compared to RLHFlow-Mistral (236.9 and 46.6 tokens, respectively; Table 1), which the paper hypothesizes causes a length bias.
4. **Skywork-PRM-1.5B (Skywork o1 Team, 2024):** Trained on Qwen2.5-Math-1.5B-Instruct using data from fine-tuned Llama-2 and Qwen2-Math series.
5. **Skywork-PRM-7B (Skywork o1 Team, 2024):** Trained on Qwen2.5-Math-7B-Instruct.
6. **Qwen2.5-Math-PRM-7B (Zhang et al., 2025):** Trained on Qwen2.5-Math-7B-Instruct using data from Qwen2-Math and Qwen2.5-Math series. The paper identifies this as "the most capable PRM among those with 7B/8B parameters" based on ProcessBench evaluation.
7. **Qwen2.5-Math-PRM-72B (Zhang et al., 2025):** Trained on Qwen2.5-Math-72B-Instruct. The paper identifies this as "the strongest open-source PRM for mathematical tasks."

The diversity in PRM training data, base models, and sizes is the experimental lever that enables the paper's central finding: PRM choice dramatically affects TTS outcomes. For instance, in Figure 4, beam search with Qwen2.5-Math-PRM-72B for Llama-3.1-8B-Instruct achieves high and scaling performance, while beam search with Math-Shepherd-PRM-7B is worse than majority voting. The same policy model, same TTS method, same budget — only the PRM differs — and the outcome can flip from improvement to degradation.

**Compute budgets.** The paper uses four budget levels: `$N \in \{4, 16, 64, 256\}$` for most experiments (Section 4.1). Some experiments extend to `$N = 512$` (Table 3 for Llama-3.2-1B-Instruct). The budgets are powers of 4 to span a meaningful range from very frugal (4 generations) to substantial (256–512 generations).

**Generation hyperparameters.** The paper specifies (Section 4.1):
- CoT temperature: 0.0 (deterministic, used only for the baseline CoT accuracy in Table 5).
- TTS temperature: 0.7 for all methods (BoN, Beam Search, DVTS). A temperature of 0.7 provides diversity while maintaining reasonable quality — too high a temperature would produce too many low-quality samples, while too low would lack the diversity needed for search to explore meaningfully different solution paths.
- Token limits: For BoN and CoT, maximum 8192 new tokens. For search-based methods, maximum 2048 tokens per step and 8192 total. The per-step limit prevents individual steps from consuming the entire budget — a step that generates an extremely long chain of reasoning without producing the step delimiter will be cut off at 2048 tokens, forcing the search to move to the next step.
- Beam width: `$M = 4$` for all beam search and DVTS experiments. This is a fixed hyperparameter, not swept — the paper does not explore how different beam widths affect the optimal strategy, which is a limitation.

**The codebase.** All experiments use OpenR (Wang et al., 2024a), an open-source framework for LLM reasoning. The paper does not modify the framework — it uses it as-is, making the results replicable with standard tooling.

**Datasets.** Two benchmarks:
- **MATH-500 (Lightman et al., 2024):** 500 representative problems from the MATH dataset (Hendrycks et al., 2021). This is the standard evaluation set used in prior TTS work (Snell et al., 2024; Beeching et al., 2024), chosen for comparability.
- **AIME24 (AI-MO, 2024):** Problems from the 2024 American Invitational Mathematics Examination. AIME problems are significantly harder than MATH-500 — they are competition problems where even strong models have low baseline accuracy. The paper includes AIME24 because "recent LLMs show significant progress in mathematical reasoning" and MATH-500 is becoming saturated for strong models. AIME24 tests whether TTS benefits extend to genuinely difficult problems.

**Answer extraction.** Answers are parsed from the final `\boxed{}` expression in each solution. For models that occasionally fail to produce this format (noted for Llama-3.2-1B-Instruct in a footnote to Table 3), the paper uses Qwen2.5-32B-Instruct as an auxiliary extractor to parse answers from free-form text. This extraction step is necessary for automated evaluation — the MATH grading function requires a normalized answer format to compare against ground truth.

---

#### The Design Space: On-Policy vs. Off-Policy PRMs

A crucial conceptual distinction running through the paper is between on-policy and off-policy PRM usage.

**On-policy PRMs.** A PRM trained on the outputs of policy model A, used to verify outputs of the same policy model A. The PRM's training distribution matches the policy model's generation distribution, so the PRM is well-calibrated for the specific model's error patterns, reasoning style, and typical mistakes. This is the setting in Snell et al. (2024), where a single PaLM 2-S\* model served as both policy and PRM training source.

**Off-policy (OOD) PRMs.** A PRM trained on the outputs of policy model B, used to verify outputs of policy model A, where A ≠ B. The PRM encounters a generation distribution different from its training distribution — the policy model may make different types of errors, structure reasoning differently, or use different notation. This is the realistic deployment scenario because training a custom PRM for every policy model is expensive. The paper explicitly investigates this because "training a PRM for each policy model to prevent OOD issues is computationally expensive" (Section 3.1).

**Empirical consequences of off-policy PRMs.** The paper documents several consequences:

- **Reduced accuracy.** In Figure 4, for Llama-3.1-8B-Instruct, Math-Shepherd-PRM-7B (trained on Mistral-7B) and RLHFlow PRMs (trained on Mistral-7B and DeepSeek-Math outputs) perform poorly, with search-based methods often worse than majority voting. Skywork and Qwen2.5-Math PRMs, which are trained on Qwen2.5-Math models (closer to the Llama-3.1 policy model's distribution), perform much better.

- **Step-level scoring biases.** PRMs trained on one model's outputs develop preferences for that model's typical step length, reasoning style, and verbosity. When applied to a different policy model, these preferences become biases. For example, the RLHFlow PRM length bias (Table 1) — preferring the typical step length of its training data — means it systematically underrates or overrates steps from other policy models based on their length rather than their correctness.

- **Amplification through search.** In BoN, a noisy PRM only affects the final selection — the policy model still generates the same candidates regardless. In beam search, the noisy PRM affects which candidates are generated in the first place, because it prunes at each step. A PRM that systematically underrates correct steps causes the search to discard good partial solutions early, steering the entire exploration toward worse regions. This is why beam search with a poor PRM can perform *worse* than BoN with the same PRM — the PRM's noise is more damaging when it acts during generation than when it only acts during selection.

The paper's empirical recommendation is implicit: use PRMs that are trained on models similar to your policy model, ideally from the same model family, and be aware that even same-family PRMs may have OOD issues when applied to policy models of substantially different sizes (since larger models produce qualitatively different reasoning traces).

---

#### Summary of Design Choices and Their Justifications

- **Off-policy PRM evaluation over on-policy:** Reflects realistic deployment constraints where training a dedicated PRM per policy model is infeasible. The paper trades some experimental cleanliness for practical relevance.
- **Absolute difficulty thresholds over quantile-based:** Avoids the collapse of quantile bins for strong policy models (Figure 3). Ensures difficulty comparisons remain meaningful across the 144× range of model sizes.
- **Three TTS methods, not more:** Covers the spectrum from pure post-hoc selection (BoN) to guided search (Beam Search) to hybrid (DVTS), capturing the essential tradeoff between diverse parallel exploration and focused sequential search. Monte Carlo Tree Search and lookahead search are explicitly excluded because prior work found them inefficient due to multi-step sampling overhead (Snell et al., 2024).
- **Beam width `$M = 4$` fixed, not swept:** Focuses the analysis on strategy-level comparison rather than hyperparameter optimization within each strategy. Sweeping beam width would multiply the already-large experimental matrix, and prior work (Beeching et al., 2024) established that moderate beam widths (3–5) work well.
- **Multiple scoring and voting methods:** The paper evaluates all combinations (Table 2) to separate the effect of the PRM's raw step scores from the effect of how those scores are aggregated — a confounding variable that prior work often conflated.
- **Llama and Qwen2.5 families:** These are the two most widely used open-source model families as of the paper's writing, making the results immediately actionable for practitioners. Using both families also tests whether findings generalize across model architectures and training procedures.
- **MATH-500 + AIME24:** Combines a standard benchmark (for comparability with prior work) with a harder benchmark (to test generalization and identify capability ceilings). The gap between MATH-500 and AIME24 performance reveals that TTS benefits are task-dependent.

## 4. Key Insights and Innovations

### Innovation 1: The Compute-Optimal TTS Strategy Must Be Reward-Aware — the PRM Is a First-Class Optimization Variable

Prior work in compute-optimal test-time scaling, most notably Snell et al. (2024), formulated the optimal strategy as a function of the policy model, the prompt, and the compute budget: choose the TTS hyperparameters that maximize expected accuracy for a given problem. The PRM — the verifier that scores candidate solutions and guides search — was treated as a fixed component of the strategy engine, not as an independent variable that changes what "optimal" means. If you swapped PRMs, you were implicitly swapping strategies, but the formulation didn't acknowledge this. The strategy `$\theta$` was optimized for a specific, implicitly assumed PRM (usually one trained on-policy with the same base model).

This paper makes a simple but profound conceptual move: **the reward function `$\mathcal{R}$` is elevated to a first-class input of the compute-optimal objective** (Equation 3). The optimal strategy `$\theta^*_{x, y^*(x), \mathcal{R}}(N)$` now explicitly depends on which PRM is being used. The reason this matters — and why it's more than a cosmetic change to the notation — is that for search-based TTS methods (beam search and DVTS), the PRM's scores determine *what gets generated in the first place*, not just what gets selected afterward. If the PRM is biased, the search explores biased regions of solution space. Swap in a different PRM with different biases, and the set of candidate solutions changes fundamentally, even with the identical policy model, identical problem, and identical compute budget.

This is a **framing innovation**. It converts the PRM from an implementation detail into an optimization dimension. The field previously asked: "Given my PRM, what's the best TTS strategy?" This paper asks: "Given that I can choose among multiple PRMs, what's the best (strategy, PRM) pair?" The empirical evidence in Figure 4 makes the case: for Llama-3.1-8B-Instruct on MATH-500, the optimal TTS method inverts depending on the PRM — BoN is best with Math-Shepherd and RLHFlow PRMs, while beam search is best with Skywork and Qwen2.5-Math PRMs. The original Snell et al. (2024) formulation would have arrived at *different conclusions about the optimal strategy* depending on which PRM happened to be used in the experiments. The reward-aware formulation makes this dependence explicit and, crucially, makes it something to be optimized rather than something to be worked around.

The significance extends beyond the equation. It reframes the practical problem of deploying TTS: instead of training a custom on-policy PRM for each policy model (which is computationally prohibitive), practitioners should think of PRM selection as part of the strategy optimization — test multiple off-the-shelf PRMs with the policy model and select the (PRM, strategy) combination that empirically works best on their problem distribution. This is a different mental model than "train a dedicated verifier," and it's more actionable with the current open-source PRM ecosystem.

### Innovation 2: The Relationship Between Policy Model Scale and Optimal TTS Strategy Is Monotonic and Interpretable

The paper uncovers a clean, monotonic relationship that was not previously documented: **as the policy model gets larger (and thus more capable at reasoning), the optimal TTS method shifts from search-based strategies (beam search) to sampling-based strategies (Best-of-N)**. Figure 7 shows this across the entire Qwen2.5 family from 0.5B to 72B parameters. For the smallest models (0.5B, 1.5B, 3B), beam search and DVTS substantially outperform BoN — these models produce many incorrect intermediate steps and benefit from the PRM catching errors early. At 7B and 14B, DVTS (which combines search within subtrees and diversity across them) performs well. By 72B, BoN is the best method across all difficulty levels — the model is strong enough that step-level guidance adds more noise than signal.

This finding is **more than a scaling curve — it's a diagnostic framework**. It tells practitioners: if you're using a small policy model, invest in a good PRM and use guided search; if you're using a large policy model, a good PRM still helps for answer selection, but don't bother with step-by-step search — it'll likely hurt. Prior work (Snell et al., 2024) established difficulty-dependent strategy variation but used a single policy model, so the model-size dimension was invisible. Beeching et al. (2024) focused on DVTS improvements without systematically varying policy model size. The current paper's contribution is showing that model size is not just another variable — it's a *monotonic driver* of which strategy class is appropriate, with a clear interpretation grounded in the model's reasoning competence.

The inverse relationship between model size and TTS benefit (Table 5: the performance gain over CoT drops from 154.6% for 1B models to 9.5% for 72B models) reinforces this insight. It suggests a **diminishing returns curve for test-time compute**: the weaker the base model, the more headroom there is for inference-time strategies to improve it. This has significant practical implications for deployment economics — TTS is most valuable where models are cheapest (small models) and least valuable where models are already expensive (large models). If you're deciding between deploying a 7B model with sophisticated TTS or a 72B model with simple CoT, the answer depends on your latency budget, hardware constraints, and the difficulty distribution of your queries, and Figure 7 gives you the empirical data to make that call.

### Innovation 3: Absolute Difficulty Thresholds Replace Quantile-Based Binning — a Necessity for Cross-Model Comparisons

This is a **methodological innovation** that enables the paper's entire cross-model analysis. Snell et al. (2024) binned problems into difficulty quintiles based on the policy model's pass@1 rate — a natural choice when studying a single model, because it ensures equal-sized bins and captures relative difficulty within that model's capability profile. But when comparing across policy models of vastly different capabilities, quantile binning collapses. The paper's Figure 3 shows why: Qwen2.5-72B-Instruct has pass@1 above 80% on 76.2% of MATH-500 problems. Quintile binning would lump together problems with 80%, 90%, and 99% pass@1 into the "easiest" bin, making it impossible to distinguish genuinely trivial problems from merely easy ones. Worse, the *same* problem would fall into different difficulty bins depending on which policy model was being evaluated — a problem that's medium-hard for a 0.5B model might be trivially easy for a 72B model. Cross-model comparisons become incoherent under quantile binning because "easy" means different things for different models.

The paper's solution — absolute thresholds (easy: 50–100% pass@1, medium: 10–50%, hard: 0–10%) — is simple enough to seem obvious in retrospect, but it represents a genuine methodological advance because it **decouples difficulty from the policy model's specific capability distribution**. A problem is "hard" if *this specific policy model* almost never solves it, regardless of what other models can do. This makes difficulty a local property of the (model, problem) pair rather than a global property of the problem relative to some reference model, which is the right framing for compute-optimal TTS: you're optimizing for a specific model, so difficulty should be measured relative to that model.

The practical consequence is that difficulty-dependent strategy recommendations become stable across model sizes. The paper can observe that for small models, BoN works best on easy problems and beam search works best on harder problems (Figure 8), while for large models, BoN works best across all difficulty levels (Figure 9). These patterns would be obscured or inverted under quantile binning because the bin compositions would shift with model size. The absolute-threshold approach isn't perfect — it's coarser (three bins vs. five) and the thresholds (50%, 10%) are chosen pragmatically rather than derived — but it's robust where quantile binning isn't, and that's the key tradeoff for enabling systematic cross-model analysis.

### Innovation 4: Characterizing PRM Failure Modes as a Diagnostic Taxonomy, Not Just an Accuracy Number

Most prior work on PRMs evaluates them via aggregate metrics — ProcessBench accuracy, best-of-N voting improvement, or correlation with ground-truth correctness. This paper goes further by cataloguing **four specific, named failure modes** that recur across different PRMs, policy models, and even across in-distribution and out-of-distribution data (Appendix C). The taxonomy — Over-Criticism, Error Neglect, Error Localization Bias, and Scoring Bias (length sensitivity) — is accompanied by concrete, annotated examples (Figures 13–18) showing exactly how each failure mode manifests in actual solutions.

This contribution is **diagnostic rather than quantitative**. It doesn't propose fixes for these failure modes; it names and exemplifies them, providing a vocabulary and a set of diagnostic patterns that future PRM developers can use to inspect their own models. The significance is that it shifts PRM evaluation from "how accurate is this model?" to "what *kinds* of mistakes does this model make, and how do those mistakes interact with different TTS strategies?" For instance, a PRM with strong Over-Criticism (scoring correct steps as wrong) will be especially damaging for beam search because it prunes good partial solutions, while a PRM with strong Error Neglect (missing errors) will damage BoN because it fails to distinguish correct from incorrect among complete solutions. The failure mode taxonomy provides a language for reasoning about which PRM biases are most harmful for which TTS strategies.

The finding that these biases "persist across both OOD datasets ... and in-distribution data" (Appendix C) is itself significant. It means these aren't just distribution-shift artifacts — they're baked into the PRM's training and would affect performance even in the ideal on-policy setting. This reframes the PRM improvement problem: reducing bias is at least as important as improving aggregate accuracy, and the two may require different training interventions. A PRM with 90% step-level accuracy but systematic Over-Criticism might perform worse in beam search than a PRM with 85% accuracy but well-calibrated scores, because the search algorithm amplifies systematic biases even when average accuracy is decent.

### Innovation 5: The Empirical Finding That Small Models Can Surpass Frontier Reasoning Models Through Compute-Optimal TTS

While the paper's headline results — a 0.5B model beating GPT-4o, a 7B model surpassing o1 and DeepSeek-R1 — are eye-catching performance claims, the **intellectual contribution** is not the numbers themselves but the systematic mapping of *when and why* this substitution works, and more importantly, *when it breaks*. The paper shows that compute-optimal TTS enables models with 100×–1000× fewer parameters to match or exceed much larger models on MATH-500 and AIME24, while simultaneously demonstrating that this capability amplification has sharp boundaries.

The comparison to long-CoT methods in Section 5.3 is particularly revealing. Compute-optimal TTS with Qwen2.5-7B-Instruct outperforms models explicitly trained for long chain-of-thought reasoning (rStar-Math-7B, Eurus-2-7B-PRIME, Qwen2.5-7B-SimpleRL, Satori-Qwen-7B) on MATH-500 — a finding that suggests external TTS with a good PRM can match or exceed internal TTS through training, at least for problems the base model already understands. But on AIME24, TTS falls substantially behind DeepSeek-R1-Distill-Qwen-7B (33.3% vs. 63.3% for the policy model alone, or 36.7% vs. 63.3% with the 72B PRM). The gap reveals a fundamental limit: **TTS amplifies what the model already knows; distillation from a strong reasoning model teaches it new reasoning patterns it didn't have**. This is a more nuanced finding than "small models can beat large models" — it's a characterization of *where* the substitution works (problems the small model can sometimes solve) versus *where* it fundamentally cannot (problems requiring reasoning patterns absent from the base model's training).

The FLOPs analysis (Table 4) adds another layer. The small models achieve this performance while "reducing the total FLOPs by 100× ∼ 1000×" compared to the large models they surpass. This isn't just about accuracy parity — it's about efficiency at parity, which has direct implications for deployment economics. The finding that a 7B model with TTS can match a 671B reasoning model while using two orders of magnitude less total compute challenges the default assumption that reasoning requires massive models, at least for problems within the smaller model's capability envelope.

The result also contextualizes the current excitement around reasoning models. If a well-tuned TTS pipeline on a standard 7B Instruct model can beat o1-preview and match DeepSeek-R1 on specific benchmarks, it raises the question: how much of the reasoning models' advantage comes from better base reasoning capabilities acquired during training, and how much comes from effectively doing internal TTS (long CoT) that could be replicated externally? The paper doesn't definitively answer this, but it provides the empirical evidence that makes the question unavoidable.

## 5. Experimental Analysis

### Evaluation Methodology

- **Dataset.** Two competition-level mathematical benchmarks are used: MATH-500 (Lightman et al., 2024), consisting of 500 representative problems from the MATH dataset (Hendrycks et al., 2021) and following the split used in prior TTS work (Snell et al., 2024; Beeching et al., 2024); and AIME24 (AI-MO, 2024), problems from the 2024 American Invitational Mathematics Examination, included because "recent LLMs show significant progress in mathematical reasoning" and MATH-500 is becoming saturated for the strongest models.

- **Base model(s).** Policy models span two families: Llama 3 (Llama-3.2-1B-Instruct, Llama-3.2-3B-Instruct, Llama-3.1-8B-Instruct) and Qwen2.5 (0.5B, 1.5B, 3B, 7B, 14B, 32B, and 72B Instruct variants), with additional DeepSeek-R1-Distill-Qwen-1.5B and DeepSeek-R1-Distill-Qwen-7B for specific comparisons. This 144× parameter range (0.5B to 72B) is deliberately chosen to reveal how optimal TTS strategy shifts with model capability.

- **Metrics.** The primary metric is **accuracy** — the fraction of problems for which the selected final answer matches the ground-truth, with answers extracted from the `\boxed{}` expression in model outputs (using Qwen2.5-32B-Instruct as an auxiliary extractor for models that sometimes fail to produce this format). The paper also reports **Pass@k** — the fraction of problems for which at least one correct answer exists among k independent samples — as a measure of the policy model's raw generation capability independent of the selection mechanism.

- **Baselines.** Several baselines are compared:
  - **Chain-of-Thought (CoT):** A single deterministic generation at temperature 0.0, representing the model's greedy reasoning performance without any test-time scaling.
  - **Pass@k:** The oracle upper bound — what fraction of problems would be solved if a perfect verifier selected the correct answer among k samples.
  - **Majority Voting:** Select the most common final answer among N sampled solutions, ignoring PRM scores entirely.
  - **Large model CoT performance:** Frontier models (GPT-4o, o1-preview, o1-mini, o1, Llama-3.1-405B-Instruct, DeepSeek-R1) evaluated with a single CoT generation, serving as the comparison point for the claim that small models with TTS can surpass large models.
  - **Long-CoT trained models:** rStar-Math-7B, Eurus-2-7B-PRIME, Qwen2.5-7B-SimpleRL, Satori-Qwen-7B, and DeepSeek-R1-Distill-Qwen-7B, all evaluated with CoT, for the comparison between external TTS and internal TTS through training.

- **Generation budget / compute accounting.** Compute is measured in **generations** (number of complete candidate solutions produced). The paper uses four budget levels: `$N \in \{4, 16, 64, 256\}$` for most experiments, with some extending to `$N = 512$`. For beam search and DVTS, the budget `$N$` represents the number of generations per expansion step (beam width), with beam size `$M = 4$` fixed. The paper does not measure wall-clock time or actual FLOPs during TTS experiments — the FLOPs analysis in Table 4 uses standard approximations (`$6ND_{\text{pretrain}}$` for pretraining, `$2ND_{\text{inference}}$` for inference) to compare total computational cost between small models with TTS and large models with CoT.

- **Cross-validation / statistical protocol.** The paper does not employ cross-validation for strategy selection — unlike Snell et al. (2024), which used two-fold cross-validation within difficulty bins, this paper's compute-optimal strategy is determined post-hoc by selecting the best-performing (method, PRM, voting configuration) per (policy model, difficulty level) combination based on full test-set results. This means the reported compute-optimal performance may be optimistically biased relative to a deployment setting where the optimal strategy must be chosen without test-set access. The difficulty grouping itself is based on pass@1 rates computed from the base model's samples, which requires generating many solutions per problem but does not require ground-truth labels — however, the paper does not develop or evaluate an automated difficulty predictor deployable at inference time.

### Main Quantitative Results

#### PRM Generalization Across Policy Models and Tasks

The first major axis of investigation asks: **how does TTS performance change when the PRM is applied to a policy model different from the one it was trained on?** The results in Figure 4 (MATH-500) and Figure 5 (AIME24) paint a stark picture of limited generalization.

**On MATH-500 (Figure 4),** for Llama-3.1-8B-Instruct paired with different PRMs at a budget of 256 generations:
- With **Skywork-PRM-7B**: Beam search achieves approximately 86% accuracy, DVTS reaches approximately 87%, and BoN reaches approximately 82%. Search-based methods clearly outperform sampling.
- With **Qwen2.5-Math-PRM-72B**: Beam search achieves approximately 83%, DVTS reaches approximately 88%, and BoN reaches approximately 87%. All methods are strong, with DVTS slightly ahead.
- With **Math-Shepherd-PRM-7B**: Beam search achieves only approximately 68% — worse than majority voting at approximately 72%. DVTS similarly underperforms. BoN manages approximately 74%.
- With **RLHFlow-PRM-Mistral-8B**: Beam search reaches approximately 72%, roughly matching majority voting. BoN reaches approximately 74%.
- With **RLHFlow-PRM-Deepseek-8B**: Beam search reaches approximately 70%, again below majority voting at approximately 72%.

The pattern is clear: the choice of PRM can flip whether search-based methods help or hurt. In the worst cases (Math-Shepherd, RLHFlow), beam search degrades performance below simple majority voting — meaning the PRM's step-level guidance is actively steering the search toward worse solutions than random sampling would find. The same policy model, same TTS method, same budget — only the PRM differs — and the performance gap between the best PRM and the worst PRM is approximately 20 percentage points.

For Qwen2.5-7B-Instruct on MATH-500 (Figure 4), the pattern is similar but with an important nuance: the model's baseline is much higher (CoT at approximately 77%, majority voting at approximately 84%), so the room for improvement is smaller. With the best PRMs (Skywork-PRM-7B, Qwen2.5-Math-PRM-7B, Qwen2.5-Math-PRM-72B), search-based methods reach approximately 90–91%, a meaningful gain over majority voting. With weaker PRMs, search provides minimal or negative benefit.

**On AIME24 (Figure 5),** the story becomes more pessimistic. AIME24 is substantially harder — CoT performance for Llama-3.1-8B-Instruct is around 3%, and even Pass@k at 256 samples barely reaches 28%. TTS with any PRM provides only modest improvements over majority voting:
- For Llama-3.1-8B-Instruct: The best configurations (Skywork-PRM-7B, Qwen2.5-Math-PRM-72B with DVTS or BoN) reach approximately 26–28% at 256 generations, compared to majority voting at approximately 22%. The absolute gain is 4–6 percentage points — significant relative to the baseline but far smaller than the 10–20 point gains seen on MATH-500.
- For Qwen2.5-7B-Instruct: The best configurations reach approximately 33–37% at 256 generations, compared to majority voting at approximately 30%. The gain is even smaller in absolute terms despite the higher Pass@k ceiling (approximately 50% at 256 samples).

The paper draws the conclusion that "the generalization of PRMs is particularly challenging across different policy models and tasks, especially for more complex tasks" (Section 4.2). The gap between Pass@k and TTS performance on AIME24 — the policy model can sometimes generate correct solutions, but TTS fails to reliably select them — indicates that the PRM's verification accuracy degrades more severely on harder problems, not just on out-of-distribution policy models.

**The relationship between PRM quality and TTS performance (Figure 6).** The paper plots TTS performance against ProcessBench scores (Zhang et al., 2025), a benchmark measuring PRMs' ability to identify errors in reasoning steps. The fitted curve `$Y = 7.66 \log(X) + 44.31$` shows a positive but sublinear relationship: better PRMs (as measured by ProcessBench) produce better TTS results, but with diminishing returns. Qwen2.5-Math-PRM-72B (the largest and strongest PRM) achieves the highest ProcessBench score and the highest TTS performance, but the margin over Qwen2.5-Math-PRM-7B is modest despite the 10× parameter difference. The log fit suggests that doubling ProcessBench performance yields only about 5.3 percentage points of additional TTS accuracy on MATH-500 — a sobering calibration for PRM development efforts.

**PRM sensitivity to voting methods (Table 2).** For Qwen2.5-7B-Instruct on MATH-500:
- **Skywork-PRM-7B** shows notable sensitivity: PRM-Min-Max achieves 83.0%, while PRM-Last-Vote achieves 87.0% — a 4-point gap. The PRM-Vote method (accumulating scores by answer) consistently outperforms PRM-Max (selecting the single highest-scoring solution), suggesting that Skywork-PRM-7B's individual score calibration is unreliable but its relative score signal, aggregated across solutions, is informative.
- **Qwen2.5-Math-PRM-7B** is remarkably insensitive: all scoring and voting combinations achieve between 87.4% and 87.8%. The paper attributes this to the PRM's training data being "processed with LLM-as-a-judge, which removes the wrong intermediate steps labeled as positive steps in the training data and makes the outputted large reward values more likely to be correct." In other words, when the PRM gives a high score, it's almost certainly correct, so even the simple PRM-Max works well.

This sensitivity analysis has practical implications: if using a PRM with uncertain calibration (like Skywork-PRM-7B), PRM-Vote is essential for robust performance; if using a well-calibrated PRM (like Qwen2.5-Math-PRM-7B), even simple maximum selection suffices, reducing computational overhead from the voting step.

#### Optimal TTS Method Depends on Policy Model Size

Figure 7 presents TTS performance across Qwen2.5 policy models from 0.5B to 72B parameters on MATH-500, using Qwen2.5-Math-PRM-72B as the verifier (the strongest available PRM, which should minimize PRM quality as a confounding variable). The results reveal a monotonic shift in the optimal strategy:

- **0.5B:** Beam search (approximately 76%) substantially outperforms BoN (approximately 70%) and DVTS (approximately 72%) at 256 generations. Majority voting reaches only about 47%. The gap between beam search and BoN is roughly 6 points.
- **1.5B:** Beam search and DVTS both reach approximately 86%, BoN reaches approximately 82%. The advantage of search narrows to about 4 points.
- **3B:** DVTS reaches approximately 88%, beam search approximately 87%, BoN approximately 87%. The methods are nearly tied.
- **7B:** DVTS reaches approximately 91%, beam search approximately 91%, BoN approximately 90%. All methods are within 1 point.
- **14B:** DVTS (approximately 91%) slightly outperforms beam search (approximately 90.5%) and BoN (approximately 90.5%).
- **32B:** BoN (approximately 91%) slightly outperforms DVTS (approximately 90%) and beam search (approximately 90%).
- **72B:** BoN (approximately 92%) outperforms both DVTS (approximately 91.5%) and beam search (approximately 91%).

The paper interprets this pattern as reflecting the underlying reasoning capability: "For small policy models, search-based methods outperform BoN, while for large policy models, BoN is more effective than search-based methods. This difference occurs because larger models have stronger reasoning capabilities and do not need a verifier to perform step-by-step selection. In contrast, smaller models rely on a verifier to select each step, ensuring the correctness of each intermediate step."

This interpretation is supported by the Pass@k curves: for the 0.5B model, Pass@k at 256 is only about 85%, while beam search achieves 76% — a substantial gap between what the model *can* generate and what TTS actually achieves. For the 72B model, Pass@k is near 100%, and BoN achieves 92% — the gap is much narrower, and beam search cannot close it further. The PRM's imperfections prevent search from fully exploiting the small model's generation capability while providing no benefit over simple sampling for the large model.

A notable observation at the low end: even for the 0.5B model, all TTS methods dramatically outperform majority voting (47%), and beam search more than doubles the CoT baseline (31.6%, from Table 5). For the weakest models, TTS is transformative rather than incremental.

#### Difficulty-Dependent Optimal Strategy

Figure 8 (Llama family) and Figure 9 (full Qwen2.5 family, in Appendix B) break down TTS performance by the three absolute difficulty levels. The results reveal that the optimal strategy shifts not only with model size but also with problem difficulty within the same model.

**For Llama-3.2-1B-Instruct (Figure 8):**
- **Easy problems (Level 1):** All methods perform extremely well (94–100% at high budgets). BoN slightly outperforms search-based methods — at 256 generations, BoN achieves approximately 99%, beam search approximately 97%. On problems the model almost always solves, aggressive search adds noise.
- **Medium problems (Level 2):** Beam search dominates. At 256 generations, beam search reaches approximately 92%, while BoN reaches approximately 72% — a 20-point gap. The PRM's step-level guidance is crucial for problems the model can sometimes but not reliably solve.
- **Hard problems (Level 3):** Beam search again dominates, reaching approximately 55% at 256 generations versus approximately 38% for BoN. However, all methods remain far below saturation — the model fundamentally struggles with these problems even with extensive search.

**For Llama-3.2-3B-Instruct (Figure 8):**
- **Easy problems:** BoN again slightly outperforms search at high budgets (approximately 100% vs. 99%).
- **Medium problems:** DVTS and beam search both reach approximately 95%, BoN reaches approximately 88%. The search advantage persists but narrows compared to the 1B model.
- **Hard problems:** Beam search reaches approximately 72%, BoN reaches approximately 58%. The gap remains large.

**For Llama-3.1-8B-Instruct (Figure 8):**
- **Easy problems:** All methods are at ceiling (near 100%).
- **Medium problems:** DVTS and BoN both reach approximately 92%, beam search approximately 89%. The search advantage disappears, and BoN becomes competitive.
- **Hard problems:** DVTS achieves approximately 60%, beam search approximately 54%, BoN approximately 48%. DVTS (which combines search within subtrees and diversity across them) emerges as the best method.

**For Qwen2.5 models (Figure 9, Appendix B),** the pattern extends with a consistent trend: for models 0.5B through 7B, beam search or DVTS is optimal on medium and hard problems, while BoN is competitive or better on easy problems. For 14B and above, BoN is the best method across all difficulty levels, with search-based methods providing no advantage and occasionally underperforming.

The practical upshot is a **decision tree** for strategy selection: if using a model under 7B, use beam search for medium/hard problems and BoN for easy ones; if using a model over 32B, use BoN uniformly; for the 7B-32B range, DVTS on easy/medium and beam search on hard provides the best balance.

#### Small Models Surpassing Large Models Through TTS

Table 3 and Figure 1 present the headline comparisons: smaller models with compute-optimal TTS versus larger models with standard CoT. The compute-optimal strategy is determined per (policy model, difficulty level) by selecting the best (method, PRM, voting configuration) from the experimental results.

**On MATH-500:**
- **Llama-3.2-3B-Instruct (compute-optimal TTS):** Achieves 78.2%, surpassing Llama-3.1-405B-Instruct (CoT) at 71.4% — a 135× parameter reduction with a 6.8-percentage-point accuracy improvement.
- **Llama-3.2-1B-Instruct (compute-optimal TTS, `$N = 512$`):** Achieves 72.2%, surpassing Llama-3.1-405B-Instruct at 71.4% — a 405× parameter reduction still achieving slight superiority.
- **Qwen2.5-0.5B-Instruct (compute-optimal TTS):** Achieves 76.4%, surpassing GPT-4o (CoT) at 74.6%.
- **Qwen2.5-1.5B-Instruct (compute-optimal TTS):** Achieves 81.8%, surpassing GPT-4o.
- **DeepSeek-R1-Distill-Qwen-1.5B (compute-optimal TTS):** Achieves 91.6%, surpassing o1-preview (85.5%) and o1-mini (90.0%).
- **DeepSeek-R1-Distill-Qwen-7B (compute-optimal TTS):** Achieves 95.2%, surpassing o1 (94.8%) and approaching DeepSeek-R1 (97.3%, CoT).

**On AIME24:**
- **Llama-3.2-3B-Instruct (compute-optimal TTS):** Achieves 30.0%, surpassing Llama-3.1-405B-Instruct (23.3%) and GPT-4o (9.3%).
- **DeepSeek-R1-Distill-Qwen-1.5B (compute-optimal TTS):** Achieves 63.3%, surpassing o1-preview (44.6%) and matching o1-mini (63.6%).
- **DeepSeek-R1-Distill-Qwen-7B (compute-optimal TTS):** Achieves 83.3%, surpassing o1 (79.2%) and DeepSeek-R1 (79.8%).

**The FLOPs analysis (Table 4)** quantifies the efficiency. Llama-3.2-3B-Instruct with compute-optimal TTS requires `$1.62 \times 10^{23}$` total FLOPs (pretraining + inference) while Llama-3.1-405B-Instruct requires `$3.65 \times 10^{25}$` — a 225× reduction in total compute while achieving higher accuracy. DeepSeek-R1-Distill-Qwen-7B requires `$7.56 \times 10^{23}$` versus DeepSeek-R1's `$5.96 \times 10^{25}$` — a 79× reduction.

**Caveats on the 1B surpasses 405B claim.** The paper's title asks "Can 1B LLM Surpass 405B LLM?" The answer from Table 3 is: **yes, but conditionally**. Llama-3.2-1B-Instruct at `$N = 256$` achieves 66.2% on MATH-500, which is below Llama-3.1-405B-Instruct at 71.4%. Only by extending to `$N = 512$` does the 1B model achieve 72.2%, surpassing the 405B model. On AIME24, the 1B model at `$N = 256$` achieves 16.7%, matching Llama-3.1-405B-Instruct, but at `$N = 512$` drops to 10.0% — the larger budget actually hurts on the harder task (likely due to the PRM's reduced accuracy on AIME24 problems causing harmful pruning in search). So the 1B surpasses 405B claim is true only on MATH-500, only at high compute budgets, and only with the right (policy model, PRM) pairing. The 3B model, by contrast, robustly surpasses the 405B model on both benchmarks.

#### Comparison to Long-CoT Methods

Table 6 compares compute-optimal TTS against models explicitly trained for long chain-of-thought reasoning:

**On MATH-500:**
- Qwen2.5-7B-Instruct with Qwen2.5-Math-PRM-7B (TTS): 88.0%
- Qwen2.5-7B-Instruct with Qwen2.5-Math-PRM-72B (TTS): 91.0%
- rStar-Math-7B (CoT): 78.4%
- Eurus-2-7B-PRIME (CoT): 79.2%
- Qwen2.5-7B-SimpleRL-Zero (CoT): 77.2%
- Qwen2.5-7B-SimpleRL (CoT): 82.4%
- Satori-Qwen-7B (CoT): 83.6%
- DeepSeek-R1-Distill-Qwen-7B (CoT): 92.4%

TTS substantially outperforms all long-CoT methods that use direct RL or SFT on MCTS-generated data (rStar-Math, Eurus-2, SimpleRL, Satori), with gains of 4.4–12.6 percentage points. It falls only 1.4 points short of DeepSeek-R1-Distill-Qwen-7B (91.0% vs. 92.4%), which benefits from distillation of 800K high-quality reasoning samples from the 671B DeepSeek-R1.

**On AIME24:**
- Qwen2.5-7B-Instruct with Qwen2.5-Math-PRM-72B (TTS): 36.7%
- rStar-Math-7B (CoT): 26.7%
- Eurus-2-7B-PRIME (CoT): 26.7%
- Qwen2.5-7B-SimpleRL-Zero (CoT): 33.3%
- Qwen2.5-7B-SimpleRL (CoT): 26.7%
- Satori-Qwen-7B (CoT): 23.3%
- DeepSeek-R1-Distill-Qwen-7B (CoT): 63.3%

The pattern reverses dramatically. TTS still outperforms the direct RL/SFT methods (by 3.4–13.4 points), but falls catastrophically behind DeepSeek-R1-Distill-Qwen-7B (36.7% vs. 63.3% — a 26.6-point gap). The paper concludes that "TTS is more effective than methods applying direct RL or SFT on the data generated via MCTS but is less effective than distilling from strong reasoning models. Also, TTS is more effective on simpler tasks than on more complex tasks."

This is arguably the paper's most important negative result. It establishes a clear boundary: external TTS amplifies what the base model already knows, but distillation from a model that has learned fundamentally new reasoning patterns (DeepSeek-R1's RL-trained long CoT) transfers those patterns to the smaller model, enabling it to solve problems that were previously beyond its reach. TTS cannot create new reasoning capabilities — it can only better exploit existing ones.

#### TTS Efficiency Relative to CoT and Majority Voting

Table 5 quantifies how much compute-optimal TTS improves over baselines for each policy model on MATH-500:

- **Llama-3.2-1B-Instruct:** CoT 26.0% → TTS 66.2% (154.6% improvement). TTS required to match CoT: >256× more compute. The efficiency gain — how many times more majority voting compute TTS needs to match majority voting performance — is >256×, meaning TTS with a small budget matches majority voting with a budget >256× larger.
- **Llama-3.2-3B-Instruct:** CoT 41.4% → TTS 78.2% (88.9% improvement). Efficiency gain: 14.1× over majority voting.
- **Qwen2.5-0.5B-Instruct:** CoT 31.6% → TTS 76.4% (141.8% improvement). Efficiency gain: >64×.
- **Qwen2.5-1.5B-Instruct:** CoT 54.4% → TTS 85.6% (57.4% improvement). Efficiency gain: >256×.
- **Qwen2.5-7B-Instruct:** CoT 76.8% → TTS 91.0% (18.5% improvement). Efficiency gain: 35.9×.
- **Qwen2.5-72B-Instruct:** CoT 83.8% → TTS 91.8% (9.5% improvement). Efficiency gain: 12.9×.

The diminishing returns are stark: the improvement over CoT drops from 154.6% for the weakest model to 9.5% for the strongest, and the efficiency gain drops from >256× to 12.9×. For Qwen2.5-32B-Instruct, the efficiency gain is only 0.8× — meaning TTS is barely more efficient than simply using majority voting, and may not justify the added complexity. The paper notes that "as the number of parameters in the policy model increases, the improvement of TTS gradually decreases. This suggests that the effectiveness of TTS is directly related to the reasoning ability of the policy model."

### Ablation Studies and Robustness Checks

**PRM length bias analysis (Table 1, Figure 12):** The paper examines why RLHFlow-PRM-Deepseek-8B produces consistently longer outputs during beam search than RLHFlow-PRM-Mistral-8B (nearly 2× the number of inference tokens for the same budget). Table 1 shows that the training data for RLHFlow-PRM-Deepseek-8B has longer average responses (333.1 vs. 236.9 tokens) and longer average steps (58.4 vs. 46.6 tokens) than RLHFlow-PRM-Mistral-8B. This training data length bias transfers to the PRM's scoring preferences: the DeepSeek-trained PRM assigns higher scores to longer steps, causing beam search to favor longer partial solutions and thus consume more tokens per problem. The paper also observes that Skywork-PRM-7B searches are more token-efficient than Qwen2.5-Math-PRM-7B searches (fewer tokens for similar performance), suggesting PRM-specific efficiency differences beyond just accuracy.

**Voting method sensitivity (Table 2):** As described in the main results above, Skywork-PRM-7B shows substantial sensitivity to voting method (4-point gap between PRM-Min-Max and PRM-Last-Vote), while Qwen2.5-Math-PRM-7B is nearly invariant (0.4-point range across all configurations). The paper attributes this to Qwen2.5-Math's training data being filtered with LLM-as-a-judge to remove incorrect steps labeled as positive, resulting in high scores being more reliably associated with correctness.

**PRM failure mode taxonomy (Appendix C, Figures 13-18):** The paper identifies four specific PRM failure patterns with annotated examples:
- **Over-Criticism (Figure 13):** A mathematically correct simplification step receives a score of 0.53, and the final correct answer receives 0.46 — both far below what correctness would warrant.
- **Error Neglect (Figures 14, 15):** A solution with a fundamental geometric error (identifying ∠EDF = 90° instead of ∠DEF = 90°) receives per-step scores of 0.99, 0.90, 0.97, 0.99, and 0.99 — the PRM completely misses the error.
- **Error Localization Bias (Figure 16):** A solution where the critical error occurs early (misapplying the Angle Bisector Theorem) receives scores of 0.20, 0.66, and 0.92 — the low score flags a problem, but it's on the first step (where the error might not yet be clear) rather than on the step where the error propagates into an obviously wrong conclusion.
- **Scoring Bias / Length Sensitivity (Figures 17, 18):** Two correct solutions to the same problem — one with short reasoning steps, one with longer but mathematically equivalent steps — receive dramatically different scores. The same correct final answer receives 0.68 (short version) versus 0.12 (long version).

The paper notes that these biases "persist across both OOD datasets ... and in-distribution data," indicating they are fundamental properties of the PRM's training, not artifacts of distribution shift.

**Scoring method interaction with PRM quality (Table 2, implicit):** The paper evaluates PRM-Min, PRM-Last, and PRM-Avg across two PRMs. While the overall differences are small (<4 points), the interaction with PRM quality is notable: for the less well-calibrated Skywork-PRM-7B, PRM-Min-Max (using the minimum step score and selecting the maximum-scoring answer) performs worst at 83.0%, while PRM-Last-Vote (using only the final step's score and accumulating by answer) performs best at 87.0%. For the better-calibrated Qwen2.5-Math-PRM-7B, all methods are in the 87.4–87.8% range. This suggests that when the PRM is noisy, the final-step score with answer accumulation is most robust — using intermediate-step minimums amplifies noise, and trusting the absolute maximum score is unreliable without good calibration.

**Prompt sensitivity (Appendix A):** The paper uses different system prompts for Llama 3 and Qwen2.5 families (Tables 7 and 8), following official evaluation templates. The Llama 3 prompt is substantially more detailed, with explicit instructions for simple vs. complex problem handling. The paper does not ablate across prompt templates, so the sensitivity of TTS results to prompt engineering is unmeasured — a significant uncontrolled variable, especially for the smallest models where prompt quality can dramatically affect baseline performance.

### Critical Assessment

The experimental results broadly support the paper's central claims, but several important qualifications and gaps deserve scrutiny.

**Does the paper demonstrate that compute-optimal TTS strategy depends on the policy model, PRM, and problem difficulty?** Yes, strongly and convincingly. Figure 4 shows that for the same policy model (Llama-3.1-8B-Instruct) on the same dataset (MATH-500), the optimal TTS method inverts depending on the PRM — BoN is best with Math-Shepherd, beam search is best with Qwen2.5-Math. Figure 7 shows that across the Qwen2.5 family, the optimal method shifts monotonically from beam search (small models) to BoN (large models). Figure 8 shows that for the same model, optimal strategy varies by difficulty level — BoN for easy, beam search for hard. The three-way interaction is empirically robust. However, the paper does not attempt to predict the optimal strategy a priori — it determines it post-hoc from full test-set evaluation, which means the "compute-optimal" label is descriptive rather than prescriptive. In a deployment setting, without access to the test set, one would need a method to estimate which strategy will work best, and the paper provides no such estimator.

**Does the paper demonstrate that small models can outperform large models through TTS?** Yes, but with conditions that should be stated more prominently than the headline. The 1B surpasses 405B result (the paper's title question) requires: (a) MATH-500 rather than AIME24 (on AIME24, the 1B model at best matches the 405B model and at higher budgets does worse); (b) a budget of 512 generations — nearly 4× the generation budget used in most experiments; (c) a specific (policy model, PRM) pairing. The 3B surpasses 405B result is more robust, holding on both benchmarks. The model surpassing o1 and DeepSeek-R1 requires DeepSeek-R1-Distill-Qwen-7B — which is not a standard Instruct model but one specifically distilled from a reasoning model, giving it a substantially stronger base than the Qwen2.5-7B-Instruct that achieves only 91.0% on MATH-500 (well below o1's 94.8%). The paper should distinguish more clearly between surpassing frontier models with a standard Instruct policy model (impressive) versus with a distilled reasoning model (still interesting but far less surprising — the distillation already transferred substantial reasoning capability). The abstract's claim that "a 7B LLM beats o1 and DeepSeek-R1" is technically true for the distilled model but could mislead readers into thinking the 7B Instruct model achieves this, which it does not (91.0% vs. 94.8% for o1).

**Are the FLOPs comparisons fair?** Table 4 compares total FLOPs (pretraining + inference) between the small model with TTS and the large model with CoT. The analysis uses standard scaling approximations and demonstrates massive efficiency advantages (100–1000×). However, the comparison is asymmetric in several ways: (a) The large model is evaluated with a single CoT generation — no TTS, no majority voting, no search. A fairer comparison might give the large model a modest TTS budget (e.g., best-of-8 or best-of-16) to see whether the small model's advantage persists. (b) The FLOPs calculation for the large model's pretraining assumes scaling parameters only (not data), following the LLaMA paradigm — a Chinchilla-optimal larger model might achieve higher accuracy at the same pretraining FLOPs, potentially closing some of the gap. (c) The inference FLOPs for TTS only count the policy model's generations, not the PRM's forward passes — beam search at `$N = 256$` with a 72B PRM involves 256 PRM evaluations per step, each of which is itself a 72B-parameter forward pass, which could substantially increase the inference FLOPs. The paper does not account for this.

**Limitations in the experimental design:**
- **No automated difficulty estimator:** The difficulty bins are computed post-hoc from pass@1 rates, requiring hundreds of samples per problem. The paper does not develop or test a deployable difficulty predictor, making the compute-optimal strategy selection a post-hoc analysis tool rather than a practical system component. This is a significant gap between the empirical analysis and practical deployment.
- **Test set is used for strategy selection:** Unlike Snell et al. (2024), which used cross-validation within difficulty bins, this paper determines the optimal strategy by selecting the best configuration from the full test set. This means the reported compute-optimal performance is optimistically biased — in a true deployment setting, the optimal strategy must be chosen without test-set access, and the actual performance would likely be lower.
- **Single beam width:** All beam search and DVTS experiments use `$M = 4$`. The paper does not explore whether different beam widths would change the optimal strategy boundaries — for instance, whether a wider beam (`$M = 8$`) would make beam search competitive for larger models, or whether a narrower beam (`$M = 2$`) would improve token efficiency. The fixed beam width means the optimal strategy may be conditional on this unoptimized hyperparameter.
- **No combination of TTS methods:** The paper evaluates BoN, beam search, and DVTS as separate, competing strategies. It does not explore hybrid approaches — for instance, using a small amount of beam search to generate candidate solutions and then applying BoN voting across beams, or dynamically switching between methods mid-generation based on the PRM's confidence.
- **PRM selection is post-hoc:** The paper shows that the optimal PRM depends on the policy model, but does not provide a method for selecting the best PRM without exhaustive evaluation. A practitioner reading this paper would learn that PRM choice matters enormously but would not learn how to choose a PRM for their specific model without running the full experimental matrix themselves.
- **No ablation on the number of difficulty bins:** The paper switches from five quantile bins to three absolute-threshold bins, arguing the latter is more appropriate for cross-model comparison. But it does not test whether a different number of bins (two, four, five with absolute thresholds) would change the conclusions. The three-bin split (easy: 50–100%, medium: 10–50%, hard: 0–10%) uses arbitrary thresholds.

**Experiments that would have strengthened the paper:**
- **Automated difficulty prediction:** Training a small classifier to predict difficulty from the problem text alone, and comparing compute-optimal TTS using predicted vs. oracle difficulty bins. This would address the "how do we deploy this?" question that the paper leaves open.
- **TTS budget for large model baselines:** Giving the 405B model or GPT-4o a modest TTS budget (e.g., best-of-16) to see whether the small model's advantage persists under a fairer comparison where both sides benefit from test-time compute.
- **Beam width sweep:** Testing whether the optimal strategy boundaries shift with different beam widths, which would determine whether the paper's recommendations are robust to this hyperparameter.
- **PRM ensemble:** Since PRM quality is identified as the key bottleneck, testing whether ensembling multiple PRMs (e.g., majority vote across PRM scores) improves robustness and reduces the strategy sensitivity to PRM choice.
- **Latency analysis:** The paper measures compute in generations but never discusses wall-clock time. Beam search with step-by-step PRM evaluation is inherently serial and potentially much slower than parallel BoN, which matters for interactive applications. A latency comparison would add practical context to the efficiency claims.
- **Extending to non-mathematical domains:** The paper acknowledges this as future work, but even a single additional benchmark (e.g., code generation, where unit tests provide natural verification) would strengthen the claim that the findings generalize beyond math.

In summary, the experiments convincingly demonstrate the **existence** of the three-way interaction (policy model, PRM, difficulty) and establish the **possibility** of small models surpassing large ones through TTS. However, the paper's **prescriptive** value — telling a practitioner exactly how to configure TTS for a given model and task distribution — is limited by the post-hoc nature of the strategy selection, the absence of automated difficulty estimation, and the unexplored sensitivity to several fixed hyperparameters (beam width, difficulty thresholds, prompt template). The paper succeeds as an empirical survey that maps the TTS design space and identifies the key variables, but it is a starting point for practical deployment, not a complete recipe.

## 6. Limitations and Trade-offs

### 6.1 Compute-Optimal Strategy Selection Requires Oracle Access to Test-Set Performance

**The assumption or constraint.** The paper determines the "compute-optimal" TTS strategy for each (policy model, difficulty level) pair by selecting the best-performing (TTS method, PRM, scoring method, voting method) configuration based on full test-set evaluation. Unlike Snell et al. (2024), which used two-fold cross-validation within difficulty bins to simulate deployment conditions, this paper selects the optimal strategy post-hoc from the same data used for evaluation. The paper does not acknowledge this explicitly — it presents the compute-optimal results in Tables 3 and 5 as if the optimal strategy were known in advance, but the strategy was in fact chosen after seeing which configuration worked best on the test set.

**The consequence.** The reported compute-optimal performance in Tables 3 and 5 is **optimistically biased** relative to what a practitioner could achieve in deployment. In a real setting, one must choose a TTS strategy without access to the test set's ground-truth answers. The paper provides no method for predicting which (method, PRM, voting) combination will be optimal for a given (policy model, difficulty) pair without exhaustive evaluation. The actual performance of a deployable system — one that must commit to a strategy based only on estimated difficulty and prior knowledge — would likely be lower than the reported numbers. The gap could be substantial: Figure 4 shows that choosing the wrong PRM can drop performance by 15–20 percentage points relative to the optimal PRM, and choosing the wrong method can cost an additional 5–10 points. If the strategy selection is itself imperfect, the compounded error could erase much of the reported gain over baselines.

**What evidence exists in the paper.** The paper provides no ablation or sensitivity analysis measuring how much the reported compute-optimal performance degrades under realistic strategy selection. Section 3.2 describes how difficulty is estimated (via pass@1 rates from model samples), but there is no corresponding method for selecting which TTS strategy to deploy at a given difficulty level. The Appendix B figures (10 and 11) show the raw performance of every (policy model, PRM, method) combination, making it clear how much variation exists — but the paper never addresses the question of how to pick the best combination without seeing the test-set results first. The entire compute-optimal framework is presented as an empirical observation ("this configuration happens to work best") rather than an operational procedure ("here is how to find the best configuration").

**Mitigation status.** The paper does not attempt to address this. It does not train a meta-classifier to predict the optimal strategy from problem features, does not use cross-validation for strategy selection, and does not report confidence intervals that would reflect strategy selection uncertainty. The authors do not flag this as a limitation in Section 7. This is arguably the most significant gap between the paper's empirical analysis and its practical applicability.

---

### 6.2 Difficulty Estimation Is Not Operationalized — No Deployable Predictor Exists

**The assumption or constraint.** The paper's compute-optimal framework groups problems by difficulty (easy, medium, hard) using the policy model's pass@1 rate — the fraction of sampled solutions that are correct. Computing this requires generating many solutions per problem and checking them against ground-truth answers. The paper acknowledges this implicitly by using difficulty as an analysis tool rather than a system component, but it does not measure the cost of difficulty estimation or propose a deployable alternative. In Section 3.2, the paper justifies switching from quantile-based to absolute-threshold difficulty bins, but never addresses the more fundamental question: **how does one determine a problem's difficulty before deciding how to allocate the inference budget?**

**The consequence.** The compute-optimal framework described in the paper is a **post-hoc analysis methodology**, not a deployable system. A practitioner cannot use it as described because the first step — "determine whether this problem is easy, medium, or hard for the policy model" — requires generating a large number of solutions and checking them against the correct answer, which either requires ground-truth labels (not available at deployment) or a reliable verifier (which is itself the component being optimized). If difficulty estimation consumes a significant fraction of the inference budget, the reported efficiency gains (e.g., 256× over majority voting in Table 5) shrink or reverse when the estimation cost is amortized. Moreover, the paper's difficulty bins are defined using the *policy model's own* pass@1, which means difficulty is model-specific — a problem that is easy for Qwen2.5-72B may be hard for Qwen2.5-0.5B. A deployable system would need to estimate difficulty for each (model, problem) pair on the fly, compounding the estimation cost.

**What evidence exists in the paper.** The paper provides no measurement of the difficulty estimation cost, no proposal for an automated difficulty predictor, and no analysis of how sensitive the compute-optimal results are to errors in difficulty classification. Figure 3 shows the pass@1 distribution for one model, but this was computed using the full test set with ground-truth access — not a realistic deployment scenario. The paper notes that this is a limitation only in passing in Section 7: "Extending TTS to more tasks such as coding and chemistry tasks" and "Exploring more effective methods for compute-optimal TTS" — neither of which directly addresses the difficulty estimation gap.

**Mitigation status.** Not addressed. The paper does not propose or evaluate any method for predicting problem difficulty without ground-truth labels. This is a critical missing piece: without it, the "compute-optimal" strategy is not computable at deployment time. Previous work (Snell et al., 2024) at least proposed using the PRM's average score as a proxy for difficulty, enabling a (somewhat circular but operational) estimation procedure. This paper neither adopts nor critiques that approach — it simply avoids the question.

---

### 6.3 PRM Quality Is the Dominant Bottleneck, and the Paper Provides No Path to PRM-Aware Strategy Selection

**The assumption or constraint.** The paper demonstrates that PRM choice is arguably the most impactful decision in a TTS pipeline — the wrong PRM can make beam search perform worse than majority voting (e.g., Math-Shepherd-PRM-7B with Llama-3.1-8B-Instruct in Figure 4 drops beam search to ~68% vs. ~72% for majority voting at 256 generations). The paper acknowledges this explicitly in Section 4.2: "PRMs are hard to generalize across policy models and tasks." However, the paper treats PRM quality as an empirically observed property rather than a predictable one. A practitioner cannot know in advance which PRM will work well with their policy model without running the full evaluation themselves — and if they run the full evaluation, they have already consumed the test set (see Limitation 6.1).

**The consequence.** The practical guidance from the paper is limited to: "test multiple PRMs and see which one works best." This is expensive (requiring evaluation across a combinatorial space of PRMs, TTS methods, scoring methods, and voting methods) and may not generalize — the optimal PRM for MATH-500 may not be optimal for AIME24 (Figure 5 shows that all PRMs struggle on AIME24, and the relative ranking shifts). Moreover, the paper identifies four specific PRM failure modes (Over-Criticism, Error Neglect, Error Localization Bias, Scoring Bias in Appendix C) but provides no diagnostic for detecting which failure modes a given PRM exhibits without annotated examples. A PRM with strong Over-Criticism will be especially harmful for beam search (pruning correct partial solutions), while a PRM with Error Neglect will harm BoN (failing to distinguish correct from incorrect). Knowing which failure mode dominates for a given (policy model, PRM) pair would enable strategic method selection, but the paper provides no such diagnostic framework.

**What evidence exists in the paper.** Figure 6 shows a positive but sublinear relationship between ProcessBench scores and TTS performance, suggesting that PRM benchmarks provide some signal. However, the correlation is noisy — RLHFlow-PRM-Mistral-8B and RLHFlow-PRM-Deepseek-8B have similar ProcessBench scores but produce dramatically different TTS outcomes (the DeepSeek variant allows substantially longer, sometimes correct solutions; the Mistral variant traps the search in short, incorrect outputs, as shown in Figure 12). The ProcessBench score alone does not capture the behavioral differences that determine TTS success. Appendix C catalogues the failure modes qualitatively but does not quantify their prevalence across PRMs or correlate them with TTS performance degradation.

**Mitigation status.** Partial. The paper's Figure 6 provides a starting point for using PRM benchmarks to guide selection, but the relationship is too weak and too averaged across policy models to serve as a reliable predictor for a specific (policy model, PRM) pair. The failure mode taxonomy in Appendix C provides a vocabulary for diagnosis but not an automated diagnostic tool. The paper does not suggest future work on PRM selection methodology.

---

### 6.4 The Hardest Problems Remain Unsolved — TTS Cannot Create New Capability

**The assumption or constraint.** The paper explicitly acknowledges in Section 7 that "the improvement of TTS gradually decreases" as model size increases, but this understates the finding: for the hardest problems, TTS provides **near-zero benefit regardless of model size**. In Figure 8 (Llama models on MATH-500), difficulty Level 3 (hard: 0–10% pass@1) shows accuracy improving from near-zero at low budgets to 55% (1B), 72% (3B), and 60% (8B) at 256 generations — gains over the CoT baseline, but still far from saturation. On AIME24 (Figure 5), where most problems are hard for all models, TTS provides only 4–6 percentage points of improvement over majority voting even at 256 generations, and the gap between Pass@k (what the model *can* generate) and TTS performance (what the PRM actually selects) remains large. The paper's own words in Section 5.3: "TTS is more effective on simpler tasks than on more complex tasks."

**The consequence.** TTS amplifies existing capability but does not create it. If the base policy model's pass@1 is near zero on a problem class — meaning it almost never produces a correct solution even among hundreds of attempts — no amount of search, verification, or voting will produce a correct answer. The PRM can only select among generated candidates; it cannot generate correct solutions that the policy model cannot produce. This is a **fundamental capability ceiling** that no amount of inference-time computation can exceed. The FLOPs comparison in Table 4, which shows 100–1000× reductions in total compute, applies only to problems within the small model's reach. For problems outside that reach (which includes most of AIME24 for models under 7B), the large model with CoT may still outperform even with unlimited TTS on the small model.

The paper's headline claim — "Can 1B LLM Surpass 405B LLM?" — is answered with a conditional yes on MATH-500 but a clear no on AIME24 (66.2% vs. 71.4% for the 1B model at `$N = 256$`; at `$N = 512$`, the 1B model's AIME24 performance actually drops to 10.0%). A practitioner deploying TTS for a real-world task must know: **what fraction of my problem distribution falls into the "hard" bin where TTS provides minimal benefit?** If that fraction is high, the large model with simple CoT may be more effective than a small model with sophisticated TTS, regardless of the FLOPs comparison.

**What evidence exists in the paper.** Table 3 and the difficulty-stratified results in Figures 8 and 9 provide clear evidence of the capability ceiling. On AIME24, the best TTS configuration for Qwen2.5-7B-Instruct (with the 72B PRM) achieves only 36.7% — dramatically below DeepSeek-R1-Distill-Qwen-7B's 63.3% with CoT alone. The paper correctly interprets this as showing that distillation from a strong reasoning model transfers new capabilities that TTS cannot replicate. The failure of Llama-3.2-1B-Instruct on AIME24 at higher budgets (dropping from 16.7% at `$N = 256$` to 10.0% at `$N = 512$`) is a concrete example of the ceiling in action — more compute actually hurts because the PRM's verification accuracy degrades on out-of-distribution hard problems, and the search amplifies those errors.

**Mitigation status.** The paper is transparent about this limitation in its results but does not frame it as a fundamental constraint on the TTS paradigm. Section 5.3 concludes that TTS is "less effective than distilling from strong reasoning models," which is accurate but does not characterize the boundary — TTS and distillation address different problems (selection from existing capability vs. transfer of new capability). The paper could more clearly state: *if your base model cannot solve a problem even 1% of the time among thousands of attempts, TTS will not help; you need a better base model or training-time intervention.*

---

### 6.5 The PRM Generalization Crisis: Off-Policy PRMs Are Unreliable, and On-Policy PRMs Are Impractical

**The assumption or constraint.** The paper deliberately studies the off-policy setting — PRMs trained on one model's outputs, used to verify a different model's outputs — because "training a PRM for each policy model to prevent OOD issues is computationally expensive" (Section 3.1). The results in Figure 4 demonstrate that this off-policy setting is **extremely fragile**: PRM performance varies by 15–20 percentage points depending on the (policy model, PRM) pairing, with some PRMs performing worse than majority voting when applied to out-of-distribution policy models. The paper's solution is to treat the PRM as a free variable in the optimization — test multiple PRMs and pick the best one — but this just relocates the problem: selecting the best PRM requires evaluation on the target task, which itself requires ground-truth labels or a reliable meta-verifier.

**The consequence.** The practical situation this paper leaves practitioners in is: *use an off-policy PRM, but only if you test it first and verify it works with your model; otherwise, you might make performance worse.* This is not actionable guidance for deployment at scale. An organization deploying dozens of fine-tuned policy model variants for different tasks cannot realistically evaluate seven PRMs with three TTS methods, three scoring methods, and three voting methods per model per task — the combinatorial explosion is intractable. The paper provides no shortcut, no transfer learning insight (e.g., "PRMs trained on Qwen2.5-Math generalize well to Qwen2.5-Instruct models of different sizes"), and no diagnostic for predicting cross-model PRM generalization.

Even the strongest PRM (Qwen2.5-Math-PRM-72B) is not immune: it achieves excellent results when paired with Qwen2.5 policy models (matching its training distribution) but shows degraded performance with Llama policy models (Figure 4 shows Llama-3.1-8B-Instruct with this PRM reaches ~87% on MATH-500 vs. ~91% for Qwen2.5-7B-Instruct with the same PRM — and the Llama model starts from a lower baseline, so the PRM is less effective at bridging that gap). The paper's own strongest PRM is not universally strong across all policy models.

**What evidence exists in the paper.** Figure 4 and Figure 5 provide comprehensive evidence of PRM generalization failures. The four failure modes in Appendix C provide a qualitative diagnosis. Table 1 quantifies one specific bias (length sensitivity in RLHFlow PRMs). The relationship between ProcessBench scores and TTS performance (Figure 6) provides a partial aggregate signal but is insufficient for per-model PRM selection.

**Mitigation status.** The paper proposes no solution. The reward-aware compute-optimal formulation (Equation 3) acknowledges the problem by making the PRM an explicit input to the optimization, but this is a descriptive move, not a solution — you still need to evaluate multiple PRMs to find the best one. The paper identifies the problem vividly but offers no path to resolving it beyond "test everything and pick the best," which is exactly the computationally expensive approach that motivated studying off-policy PRMs in the first place. The paper's conclusion that "future work should focus on developing more adaptable and universal supervision mechanisms" (Section 7) is an accurate recognition that this is an unsolved problem, not a solved one.

---

### 6.6 Single-Domain Evaluation: All Results Are on Competition Math with String-Match Evaluation

**The assumption or constraint.** Every experiment in the paper uses mathematical reasoning benchmarks (MATH-500 and AIME24) where answers are discrete values (numbers, expressions) that can be compared to ground truth via string matching after `\boxed{}` extraction. The PRMs are all trained on mathematical reasoning data, and the TTS methods are evaluated exclusively on step-by-step mathematical derivations. The paper acknowledges this scope limitation only in passing in Section 7: "Extending TTS to more tasks such as coding and chemistry tasks."

**The consequence.** The paper's findings may not generalize beyond mathematical reasoning in several specific, consequential ways:

**1. Verification difficulty scaling.** Mathematical reasoning has a unique property: each step is (in principle) deterministic and verifiable — given the step, a trained PRM can assess whether the logical conclusion follows from the premises. In domains like creative writing, dialogue, or strategic planning, "correctness" of intermediate steps is ill-defined, and PRM training requires subjective or heuristic labels. The paper's entire TTS framework depends on step-level verifiability; if PRMs are inherently less accurate in other domains, the search-based methods that drive the paper's strongest results (beam search for small models on medium/hard problems) would degrade disproportionately.

**2. Answer extraction and voting.** The voting methods (Majority Vote, PRM-Vote) rely on being able to identify identical answers. In math, this is straightforward: `11\sqrt{2}` and `\sqrt{242}` may need normalization, but there are standard graders. In code generation, functional equivalence is undecidable; in summarization, semantic equivalence is subjective. The paper's best results on MATH-500 use PRM-Vote (accumulating scores for identical answers), but this mechanism breaks down when "identical answer" cannot be reliably determined.

**3. PRM training data availability.** All seven PRMs evaluated were trained on mathematical reasoning data. For coding, chemistry, or other STEM domains, comparable open-source PRMs may not exist, forcing practitioners to either train their own (expensive) or use mathematical PRMs out-of-domain (whose generalization the paper has already shown to be fragile even within math).

**What evidence exists in the paper.** None — the paper does not evaluate any non-math task. The absence is acknowledged in the limitations section but not explored. The paper's claim that TTS enables small models to surpass large ones is empirically supported **only for competition-level math problems**. Whether the same holds for code generation (where unit tests provide natural verification), scientific reasoning, or open-ended generation is entirely untested.

**Mitigation status.** The paper flags extension to other tasks as future work but provides no preliminary results, no analysis of what properties of math make TTS effective, and no guidance on which of the paper's findings might transfer. A practitioner working in code generation, for example, cannot determine from this paper whether to use beam search (beneficial for small math models) or BoN (beneficial for large math models) — the entire difficulty-strategy mapping may be domain-specific.

## 7. Implications and Future Directions
- How this changes the landscape
  - Moves the field from “TTS is good in general” to “TTS must be reward‑aware and tailored to the policy, the PRM, and problem difficulty.” It demonstrates that careful inference strategy selection can flip performance orderings across 100× parameter gaps (Figure 1; Table 3).
  - Establishes empirical rules: use search for small policies (with strong PRMs), favor BoN for large policies, and adapt the method by difficulty (Figures 7–9).

- Follow‑up research enabled
  - PRM research:
    - Training data curation to reduce over‑criticism/error‑neglect and length biases (Figures 13–18; Table 1).
    - Weak‑to‑strong supervision: the paper shows a 7B PRM effectively supervises a 72B policy (§7 Conclusion), motivating scalable verifiers rather than ever‑larger PRMs.
    - More robust scoring/voting designs that are less sensitive across policy families (Table 2).
  - TTS algorithms:
    - Adaptive budget allocation “mid‑generation” based on self‑predicted uncertainty (related to §6 Related Work; could be combined with reward‑aware search).
    - Difficulty predictors to route problems to BoN vs beam vs DVTS automatically, following the empirical mapping in Figures 8–9.
  - Beyond math:
    - Apply the reward‑aware framework to coding (unit tests as rewards), scientific question answering (symbolic checkers), or multimodal reasoning (vision value models; see §6 Related Work).

- Practical applications
  - Cost‑effective deployment: small on‑device or edge models augmented with PRM‑guided TTS for high‑accuracy math tutoring, homework checking, or exam prep.
  - Cloud inference optimization: dynamically choose BoN vs search based on prompt difficulty and model size to minimize latency/compute for a target accuracy.

---

Key citations to ground claims:
- Reward‑aware compute‑optimal formulation: §3.1, Eq. (3).
- Difficulty thresholds: §3.2, Figure 3.
- TTS methods: §2.2, Figure 2.
- Cross‑matrix results: Figures 4–11.
- Small vs large comparisons: Figure 1; Table 3.
- FLOPS comparisons: Table 4.
- Gains vs CoT and Majority: §5.2, Table 5.
- TTS vs long‑CoT training: §5.3, Table 6.
- PRM biases and failures: §4.4, Table 1–2, Figures 12–18.
