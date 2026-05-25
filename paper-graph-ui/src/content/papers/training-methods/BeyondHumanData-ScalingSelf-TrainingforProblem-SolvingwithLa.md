# Beyond Human Data: Scaling Self-Training for Problem-Solving with Language Models

**ArXiv:** [2312.06585](https://arxiv.org/abs/2312.06585)

## 🎯 Pitch

This paper introduces ReST_EM, an expectation-maximization-based self-training method that enables large language models to fine-tune themselves using their own generated solutions filtered by external correctness feedback—without relying solely on expensive human-generated data. By demonstrating that ReST_EM not only surpasses traditional human-data fine-tuning on challenging math and code benchmarks but also scales favorably with model size and reliably transfers to new tasks, the work paves the way for more efficient, scalable, and autonomous improvement of language models' reasoning and coding abilities.

---

## 1. Executive Summary
This paper introduces ReST_EM, an expectation–maximization (EM) based self‑training procedure for large language models that learns from the model’s own generated solutions filtered by an external correctness signal. Applied to math problem solving (MATH) and code generation (APPS), ReST_EM consistently outperforms fine‑tuning only on human solutions, scales well with model size, improves both single‑answer accuracy and multi‑sample metrics (pass@K, majority voting), and transfers to held‑out tasks like GSM8K and HumanEval (Figures 2–5, 9–10).

## 2. Context and Motivation
- Problem addressed
  - High‑quality human solutions for complex tasks (e.g., competition math, programming problems) are scarce and expensive to collect, limiting supervised fine‑tuning (SFT) (Introduction; Equation 1).
  - Many such tasks provide a cheap, automatic correctness check (a scalar reward): verifying a final numeric answer in math or running unit tests in code (Section 1; Section 5, “Training Datasets”).

- Why this matters
  - Reducing dependence on human data can lower cost and speed up iteration for improving reasoning and coding capabilities in large models.
  - A reliable external reward enables principled learning from synthetic data while avoiding subjective preference modeling.

- Prior approaches and shortcomings
  - Online RL with policy gradients: updates and sampling are interleaved, which is expensive for large models (Section 2).
  - Single‑iteration self‑training (e.g., Rejection Sampling Fine‑Tuning, RFT) shows gains on smaller models and simpler datasets but scales weakly with model size and struggles on harder tasks (Section 4; Figure 2–3 discussion).
  - STaR uses greedy decoding or “rationalization” (feeding the correct answer); this increases false positives in reasoning (Section 4).
  - Iterative Maximum Likelihood (IML) couples data collection and optimization per mini‑batch, which is costly and prone to overfitting/drift for large LMs (Section 4).
  - ReST (Gulcehre et al., 2023) mixes human data and continues fine‑tuning from the last iteration; this can reduce transfer (Figure 7).

- Positioning of this work
  - ReST_EM formalizes self‑training as EM for reinforcement learning (RL), decoupling sample collection from optimization, and uses only binary external rewards—no human solutions in the synthetic set (Section 3; Algorithm 1; “Differences with ReST”).

## 3. Technical Approach
Goal in plain terms: iteratively make the model better at producing correct solutions by (a) generating many candidate answers per problem, (b) keeping only those that can be verified as correct, and (c) fine‑tuning on these verified solutions; then repeat.

Key terminology (defined once):
- `Binary reward`: 1 if a generated solution is correct (passes the verifier), 0 otherwise.
- `E‑step` / `M‑step`: “Expectation” collects/weights data based on likely correctness; “Maximization” updates model parameters to fit that weighted data.
- `ELBO` (evidence lower bound): a tractable objective that lower‑bounds the intractable log‑likelihood of observing high reward (Equation 2).

Step‑by‑step method
1) Preliminaries and SFT baseline
   - A language model defines `p_θ(y|x)` over output sequence `y` given input `x`. SFT minimizes negative log‑likelihood of human outputs (Equation 1).

2) EM for RL: how correctness becomes a learning signal (Section 3; Equations 2–3)
   - Introduce a latent “optimality” variable `O` that flags correct outputs. The aim is to maximize `log p(O=1|x)`, i.e., probability that a sample is correct.
   - Because summing over all sequences is intractable, maximize its ELBO (Equation 2):
     - E‑step: set a variational distribution `q(y|x)` proportional to `p(O=1|x,y) p_θ(y|x)`. With a non‑decreasing function `f` and non‑negative rewards, using `f(r)=r` makes `q(y|x) ∝ r(x,y) p_θ(y|x)`.
     - M‑step: update `θ` to minimize `KL[q(y|x) || p_θ(y|x)]`, equivalent to maximizing a reward‑weighted log‑likelihood (Equation 3):
       - Maximize `E_x E_{y~p_{θ_t}(·|x)} [ r(x,y) log p_θ(y|x) ]`.
   - Intuition: treat “correct” samples as soft expert demonstrations weighted by how often they occur, then train the model to imitate them. Unlike online RL, data collection uses the fixed policy from the previous iteration, decoupling generation from training for scalability.

3) ReST_EM algorithm in practice (Algorithm 1; Section 3 “ReST_EM”)
   - Generate (E‑step):
     - For each problem `x`, sample multiple outputs from the current model using stochastic decoding; compute the binary reward by an automatic checker.
     - Keep only correct outputs; limit per‑problem count to avoid a flood of easy‑problem solutions.
   - Improve (M‑step):
     - Fine‑tune the model with reward‑weighted next‑token prediction on the collected solutions.
     - Crucial design: always fine‑tune from the base model each iteration (not from the previous iteration’s fine‑tuned weights) to reduce task‑specific drift and preserve transfer (Figure 7).
   - Iterate until validation reward stops improving.

4) Concrete training details (Section “Implementation Details”)
   - Sampling: top‑K sampling with `K=40`, temperature `0.7` during data collection.
   - Solutions per problem: 32 for MATH, 64 for APPS; cap to at most 10 correct solutions per problem to balance difficulty.
   - Fine‑tuning: use few‑shot prompts as input and the model’s correct solutions as targets; apply loss only on target tokens.
   - Evaluation decoding: greedy decoding for pass@1 (Figures 2–3); for pass@K, use temperature `1.0` with nucleus sampling `p=0.95` (Figure 5 caption).

5) How this differs from close variants (Section 3 “Differences with ReST”; Table 1)
   - No human solutions are mixed into the synthetic set during Generate steps.
   - Each Improve step resets to the base pretrained model rather than continuing from the last iteration’s finetuned weights (yields better transfer; Figure 7).
   - Binary rewards make threshold‑scheduling tricks unnecessary.

Simplified example (math problem):
- Iteration 1:
  - Generate 32 solutions; mark each as correct if the final answer equals the ground‑truth answer.
  - Keep up to 10 correct solutions; train the model to reproduce them.
- Iteration 2:
  - The improved model tends to generate more correct solutions; repeat the loop, gradually making the model more likely to produce correct reasoning paths and answers.

## 4. Key Insights and Innovations
- EM‑based self‑training scales with model size and decouples sampling from optimization
  - Novelty: formalizes self‑training as EM for RL in LLMs, yielding a reward‑weighted maximum‑likelihood objective that is efficient for very large models (Section 3; Equations 2–3).
  - Significance: avoids the high cost of online RL while maintaining a principled objective; enables multiple Generate/Improve iterations at scale (Figures 2–3).

- Always fine‑tuning from the base model preserves transfer
  - Different from ReST, which continues from the last fine‑tuned checkpoint; ReST_EM restarts from the base each iteration (Section 3; “Differences with ReST”).
  - Impact: similar task performance but substantially better transfer to held‑out tasks (Figure 7).

- Model‑generated data can be better than human‑generated data
  - Head‑to‑head comparisons show higher test accuracy when fine‑tuning on one model‑generated solution per question versus one human solution per question (Figure 6, left/right).
  - Distillation: smaller models trained on solutions generated by a larger teacher outperform SFT on human solutions and even self‑training with the smaller model’s own data (Figure 6, right).

- Gains extend beyond single‑answer accuracy
  - Pass@K and majority voting improve meaningfully, not just pass@1 (Figure 5; Section 5.2). This indicates the method strengthens both the quality and diversity of correct generations.

## 5. Experimental Analysis
- Evaluation setup
  - Datasets with automatic reward:
    - Math: Hendrycks MATH (7,500 training problems; answer checking against ground truth) (Section “Training Datasets”).
    - Code: APPS (Introductory) (2,342 training problems; reward from unit tests) (same section).
  - Transfer/held‑out:
    - Math: GSM8K and Hungarian High School finals (Figure 10).
    - Code: HumanEval (Figure 3, right; Figure 5, left).
    - General reasoning: Big‑Bench Hard (BBH) (Figure 9).
  - Models: PaLM 2‑S (Bison), PaLM 2‑S* (Codey), PaLM 2‑L (Unicorn) via Google Cloud (Section “Models”).
  - Metrics:
    - `pass@1`: accuracy of a single greedy decode (Figures 2–4).
    - `pass@K`: probability at least one of K samples is correct (Figure 5).
    - Majority voting: pick the most frequent answer across many samples (Section 5.2).

- Main results
  - ReST_EM surpasses SFT on human data
    - Math (MATH): multiple iterations yield higher test accuracy than SFT for both PaLM 2‑S* and PaLM 2‑L (Figure 2, left). Transfer to GSM8K also improves (Figure 2, right).
    - Code (APPS): large gains after the first iteration; additional iterations can regress (overfitting) (Figure 3).
  - Gains scale with model size (Section 5.1):
    > “On the MATH dataset, the test accuracy improvement with ReST_EM is 5.94% for PaLM 2‑S compared to 6.34% for the larger PaLM 2‑L. Similarly, on the APPS dataset, improvements are 5.6% for PaLM 2‑S* compared to 6.4% for PaLM 2‑L.”
  - Overfitting after too many iterations
    - Training accuracy keeps rising, but test accuracy saturates or drops (Figure 4). The drop is stronger on APPS, which has fewer training problems, making it more prone to overfitting (Section “Train-test performance gap”).
  - Multi‑sample metrics
    - Pass@K improves for all K; the largest gap is typically at K=1 (Figure 5).
    - Majority voting on MATH with 64 samples:
      > “PaLM 2‑L fine‑tuned with ReST_EM obtains 48.82, while the base model gets 44.02.” (Section 5.2)
  - Multiple iterations vs more data in one iteration
    - A single iteration with 3× more samples per problem underperforms multiple smaller iterations:
      > “Fine‑tuning with this dataset results in pass@1 of 40.3%, lower than 41.0% in the second and 41.9% in the third iteration.” (Section “Impact of multiple iterations”)
    - Implication: iterative bootstrapping improves the quality of the collected dataset, not just its size.
  - Human vs model solutions (apples‑to‑apples, 5K questions)
    > “ReST* (5K) [one model‑generated solution per question] outperforms SFT (5K) [one human solution per question].” (Figure 6, left; Section “Comparing model-generated data with human data”)
  - Distillation across models
    > “Distill* (2‑L) [one teacher solution per problem] surpasses SFT (Human). Distill (2‑L) [multiple teacher solutions] surpasses ReST_EM (2‑S).” (Figure 6, right)
  - ReST vs ReST_EM (resetting to base each iteration)
    - Similar APPS accuracy but much better transfer to HumanEval when resetting (Figure 7).
  - Data efficiency and difficulty profile
    - Even 1,000 MATH questions yield substantial gains; performance generally increases with more questions (Figure 8, left).
    - Improvements occur across all difficulty levels, with the largest gains on “medium” and “hard” questions (Figure 8, right).
  - Generalization breadth
    - On BBH, no degradation; MATH‑trained ReST_EM improves average performance with chain‑of‑thought prompting (Figure 9).
    - On the 2023 Hungarian HS finals exam, PaLM 2‑L (ReST_EM) is competitive:
      > “Surpasses the performance of all existing models except GPT‑4.” (Figure 10)

- Do the experiments support the claims?
  - The study uses diverse metrics (pass@1, pass@K, majority voting), multiple models, and multiple datasets, including held‑out exams and BBH.
  - Ablations isolate the effects of data quantity, number of iterations, human vs model data, and training restart strategy (Figures 6–8). This breadth convincingly supports the central claims while also showing the limits (overfitting after too many iterations on smaller datasets; Figure 4).

## 6. Limitations and Trade-offs
- Assumptions about the task
  - Requires an automatic or easily computed external reward (`r(x,y)`), ideally binary. Many real tasks lack a verifiable signal; applying ReST_EM would then need a learned or heuristic reward (Section 6).
- Data requirements
  - Still needs a moderate set of input prompts/problems (thousands in the reported experiments). New domains will need data collection (Section 6).
- Overfitting and iteration count
  - On smaller datasets (APPS), more than one iteration can reduce test accuracy even as training accuracy rises (Figure 4). Careful early stopping and validation are necessary.
- Compute considerations
  - Although cheaper than online RL, each iteration generates many samples per problem (32–64) and fine‑tunes a large model. The approach trades off human data for sampling compute.
- Reward granularity
  - Binary rewards treat all correct solutions equally and ignore useful near‑misses; this may slow learning compared to shaped/graded rewards. The authors note a remaining gap between pass@1 and pass@K (Section 6).
- E‑step quality
  - The collection relies on stochastic decoding without explicit search/planning. Stronger E‑steps (e.g., majority‑vote guided search) might further improve data quality but were not explored here.

## 7. Implications and Future Directions
- Field impact
  - Demonstrates that self‑training with external feedback can surpass human‑only SFT for complex reasoning/coding and that synthetic data can scale with model size (Figures 2–3). This shifts the focus from collecting more human solutions to building better verifiers and sampling procedures.
- Practical applications
  - Upgrading specialized skills where correctness is checkable: competitive math assistance, program synthesis and repair, data transformation pipelines with tests, grading/verification‑heavy domains (e.g., symbolic reasoning tasks).
  - Knowledge transfer: create high‑quality, verified synthetic corpora from larger models to train smaller models (Figure 6, right).
- Research avenues
  - Better E‑steps: incorporate search/planning, verifier‑guided sampling, or self‑consistency to reduce false positives and improve sample efficiency (Section 4; discussion about ExI/T and rationalization).
  - Richer rewards: move beyond binary signals where feasible; EM readily accommodates non‑binary rewards (Section 3 “Reward weighted regression” link).
  - Overfitting mitigation: active data selection, curriculum design by difficulty (Figure 8, right), and regularization to sustain gains across iterations.
  - Closing pass@1 vs pass@K gap: algorithmic improvements that capture the benefits of sampling at training time, potentially via entropy regularization or mixture‑of‑solutions distillation (Section 6).
  - Automating the pipeline: learning verifiers for broader tasks, including subjective domains via preference models, while preserving the EM formulation (Section 6).

Overall, ReST_EM provides a simple, theoretically grounded, and empirically validated recipe: generate many candidates, keep only what a verifier deems correct, and iterate—resetting to the base model each time. The method reduces reliance on costly human solutions, scales to large LLMs, strengthens both single‑shot and multi‑sample performance, and transfers robustly across related tasks.
