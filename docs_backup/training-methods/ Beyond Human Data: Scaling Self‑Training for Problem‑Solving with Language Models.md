# Beyond Human Data: Scaling Self‑Training for Problem‑Solving with Language Models

**ArXiv:** [2312.06585](https://arxiv.org/abs/2312.06585)
**Authors:** Avi Singh, John D. Co‑Reyes, Rishabh Agarwal, Ankesh Anand, Piyush Patil, Peter J. Liu, James Harrison, Jaehoon Lee, Kelvin Xu, Aaron Parisi, Abhishek Kumar, Alex Alemi, Alex Rizkowsky, Azade Nova, Ben Adlam, Bernd Bohnet, Hanie Sedghi, Igor Mordatch, Isabelle Simpson, Izzeddin Gur, Jasper Snoek, Jeffrey Pennington
**Institutions:** Google DeepMind, Google Research (possibly others among authors)

## 🎯 Pitch

ReST_EM introduces a scalable self-training method for language models, utilizing a binary feedback loop to drive learning and optimization as an expectation-maximization process. This innovation significantly reduces the reliance on expensive human-generated data, enabling models to enhance their reasoning and coding performance autonomously, which is a pivotal step toward more efficient and adaptable AI systems.

---

## 1. Executive Summary
This paper introduces ReST_EM, a simple but scalable self-training method that lets a language model improve itself using its own generated solutions and a binary “correct/incorrect” feedback signal. By casting self-training as an expectation–maximization (EM) procedure for reinforcement learning, the method decouples sample collection from optimization and achieves strong gains on math (MATH, GSM8K) and code (APPS, HumanEval) benchmarks, often outperforming fine-tuning on human solutions and improving as the base model gets larger.

## 2. Context and Motivation
- Problem addressed
  - High-quality, task-specific human data for fine-tuning large language models (LLMs) is expensive and limited, especially for complex reasoning (e.g., competition-level math) and coding.
  - Reinforcement learning (RL) approaches can use scalar rewards (e.g., “correct answer?” or “passed tests?”) but standard online RL is computationally heavy for very large models because it interleaves sampling and policy updates constantly.

- Why it matters
  - If LLMs can reliably learn from their own data when a simple correctness signal exists, we can substantially reduce reliance on human-written solutions while scaling to larger models and harder tasks. This has both practical implications (lower cost, faster iteration) and theoretical importance (clarifies when and how self-training can improve large models).

- Prior approaches and their gaps
  - ReST (Reinforced Self-Training) and related methods like STaR, RFT, IML, RAFT explored self-generated data, but:
    - Many used small models (≤7B) or reported limited scaling (Section 4, Table 1).
    - Some rely on one sample per problem or “rationalization” (giving the answer in the prompt), which can inflate false-positive reasoning (Section 4).
    - Online RL methods remain expensive and tightly coupled to continuous sampling (Section 2).
  - Gap: a scalable, theoretically grounded procedure that works well for large LLMs on hard reasoning/coding tasks using only a simple binary reward.

- Positioning
  - ReST_EM reframes self-training as EM for RL (Section 3). This framing:
    - Cleanly separates data collection (E-step) from optimization (M-step), easing scaling to large models.
    - Uses a binary, automatically computed reward (correctness) to filter samples.
    - Improves on ReST by fine-tuning from the same base model each iteration, reducing task-specific drift and improving transfer (Section 3, “Differences with ReST”; Figure 7).

## 3. Technical Approach
ReST_EM is an expectation–maximization view of self-training with a binary reward.

Key terms (defined on first use):
- Binary reward: a 0/1 signal indicating whether a generated solution is correct (e.g., exact answer match for math, or all unit tests pass for code; Section 5).
- EM (expectation–maximization): an iterative algorithm alternating between estimating a distribution over latent variables (E-step) and optimizing parameters given those estimates (M-step).
- ELBO (evidence lower bound): a tractable objective that lower-bounds the log-likelihood we want to maximize.
- Variational distribution `q(y|x)`: an auxiliary distribution over outputs used inside the ELBO, optimized to approximate the true posterior over outputs that lead to high reward.
- KL divergence: a measure of distance between probability distributions; here it keeps `q` close to the model distribution `pθ`.

Step-by-step methodology (Section 3, Algorithm 1)
1) Formalization as EM for RL
   - Define an “optimality” variable `O∈{0,1}` where `p(O=1|x,y) ∝ f(r(x,y))`, with `r` the scalar reward and `f` a non-decreasing function (Section 3).
   - The target is to maximize `log p(O=1|x) = log ∑_y pθ(y|x) p(O=1|x,y)`, which is intractable to compute directly because it sums over all sequences.
   - Introduce the ELBO (Equation 2):
     > L(pθ, q) = Eq(y|x)[log p(O=1|x,y)] − KL[q(y|x) || pθ(y|x)].
   - EM alternates:
     - E-step: maximize L over `q`, yielding `q*(y|x) ∝ p(O=1|x,y) pθ(y|x)`.
     - M-step: maximize L over `θ`, equivalently minimize `KL[q || pθ]`, which is a weighted maximum likelihood update.

2) Specialization to non-negative, binary rewards
   - With non-negative rewards and `f` as identity, `p(O=1|x,y) ∝ r(x,y)`. The M-step becomes a reward-weighted log-likelihood objective (Equation 3):
     > maximize E_x E_{y∼pθ_t(y|x)} [ r(x,y) · log pθ(y|x) ].
   - Intuition: learn to put higher probability mass on outputs that receive reward 1.

3) Practical algorithm (Algorithm 1; Sections 3 and 5)
   - Generate (E-step)
     - Sample many solutions per problem from the current model `pθ` using stochastic decoding (top-k=40, temperature=0.7; Section “Implementation Details”).
     - Score each with the binary reward: for MATH, exact answer matching; for APPS, pass/fail against unit tests (Section 5).
     - To prevent over-representation of “easy” problems, keep at most 10 correct samples per problem (Section 5, “Implementation Details”).
     - In their runs: 32 samples/problem for MATH; 64 for APPS (Section 5).
   - Improve (M-step)
     - Fine-tune the model with a reward-weighted next-token prediction loss:
       > J(θ) = E_{(x,y)∼Di} [ r(x,y) · log pθ(y|x) ] (Algorithm 1).
     - Crucial design choice: always fine-tune from the same base pretrained model each iteration (not from the previous iteration’s checkpoint). This minimizes drift and preserves transfer to hold-out tasks (Section 3, “Differences with ReST”; Figure 7).
     - Use a validation set to stop training “while reward improves on D_val” (Algorithm 1).

4) Why EM over alternatives?
   - Compared to online RL: EM decouples sampling from optimization (Equation 3 discussion), making it more scalable for very large models.
   - Compared to prior self-training:
     - No need for rationalization or oracle answers (unlike STaR; Section 4).
     - Multiple iterations are supported (unlike one-shot RFT; Section 4), which matters for harder math tasks.
     - Avoids accumulating drift by restarting fine-tuning from the base model at each iteration (difference from ReST; Figure 7).

5) Evaluation and decoding settings (Sections 5, 5.2)
   - For pass@K and majority voting analyses, they use temperature 1.0 and nucleus sampling with p=0.95 (Figure 5).
   - For standard pass@1 evaluation on MATH and GSM8K, they use greedy decoding following the PaLM 2 evaluation protocol (Figure 2 caption).

## 4. Key Insights and Innovations
- EM framing for self-training at LLM scale (fundamental innovation)
  - Viewing self-training as EM for RL (Equations 2–3) yields a simple, stable update: collect samples with the current model, weight by correctness, and maximize a weighted log-likelihood. This clean separation of E- and M-steps is what makes the procedure scale to large models without the complexity of online RL (Section 3).

- Fine-tuning from the base model each iteration to improve transfer (design innovation)
  - Unlike ReST, each iteration’s M-step starts from the same pretrained checkpoint. Figure 7 shows similar in-domain performance on APPS but substantially better transfer to HumanEval when following this choice, suggesting reduced overfitting/drift.

- Self-generated data can beat human data for problem-solving (empirical insight)
  - On MATH and APPS, training on filtered, model-generated solutions produces larger gains than fine-tuning on human-written solutions of similar quantity (Figures 2–3, 6). This overturns a common assumption that human data is always superior once correctness can be validated.

- Gains scale with model size (scaling insight)
  - Improvements are larger for bigger models: on MATH, ReST_EM increases pass@1 by 6.34% for PaLM 2-L vs. 5.94% for PaLM 2-S; on APPS, +6.4% for PaLM 2-L vs. +5.6% for PaLM 2-S* (Section 5.1). This is noteworthy because prior work (e.g., RFT on GSM8K) reported diminishing returns with larger LMs.

- Broad downstream benefits without regressions (practical insight)
  - The method improves pass@K and majority voting (Figure 5), transfers well to GSM8K and HumanEval (Figures 2–3), and shows no major degradation on Big-Bench Hard; the MATH-tuned model even improves BBH with chain-of-thought prompting (Figure 9).

## 5. Experimental Analysis
- Evaluation methodology
  - Datasets (Section 5)
    - Training: Hendrycks MATH (7,500 train problems), APPS Introductory (2,342 train problems).
    - Transfer/held-out: GSM8K (grade-school math word problems), HumanEval (Python coding), Big-Bench Hard (23 diverse reasoning tasks), and the 2023 Hungarian high-school math finals (Figure 10).
  - Metrics
    - pass@1 (single-sample accuracy); pass@K (probability at least one of K samples is correct); majority voting (sample multiple solutions and take the most consistent answer) (Section 5.2; Figure 5).
  - Models
    - PaLM 2 variants via public APIs: `PaLM 2-S` (Bison), `PaLM 2-S*` (Codey), `PaLM 2-L` (Unicorn) (Section 5).
  - Decoding/training settings
    - E-step sampling: top-k=40, temperature=0.7; cap at 10 correct solutions per problem (Section “Implementation Details”).
    - Fine-tuning: reward-weighted next-token loss on generated targets, using few-shot prompts as inputs and generated solutions as targets (Section “Implementation Details”).

- Main results with specifics
  - In-domain performance and scaling
    - MATH (Figure 2 left):
      > “ReST_EM increases pass@1 by 5.94% (PaLM 2-S) and 6.34% (PaLM 2-L) over the base.”
      Gains accrue over the first 1–2 iterations; further iterations yield diminishing returns (Figure 2).
    - APPS Introductory (Figure 3 left):
      > “Most gains come in the first iteration; second iteration can regress,” consistent with overfitting on a smaller dataset (confirmed by train–test gap in Figure 4 right).
  - Transfer performance
    - GSM8K (Figure 2 right):
      > “Transfer improves alongside MATH; improvements remain after 2–3 iterations.”
    - HumanEval (Figure 3 right):
      > “First iteration delivers most of the gain; additional iterations may regress.”
    - Big-Bench Hard (Figure 9):
      > “No major degradation on any task; MATH-tuned model improves average BBH with chain-of-thought prompting.”
    - Hungarian high-school exam (Figure 10):
      > “PaLM 2-L (ReST_EM) ranks second only to GPT-4, while several math-specialized models with strong GSM8K scores perform poorly on the exam.”
  - Diversity-aware performance
    - Pass@K (Figure 5; temperature=1.0, nucleus p=0.95):
      > “ReST_EM improves pass@K across HumanEval, APPS, and MATH for all K; the largest gap is typically at K=1.”
    - Majority voting (Section 5.2):
      > “With 64 samples per MATH question, PaLM 2-L (ReST_EM) achieves 48.82% vs. 44.02% for the base model.”
  - Overfitting and iteration count
    - Train–test gap (Figure 4):
      > “Training accuracy increases linearly with iterations, but test accuracy plateaus (MATH) or regresses (APPS).”
    - Single big E-step vs multiple iterations (Section 5.3):
      > “Collecting 3× more samples in one iteration yields 40.3% on MATH (PaLM 2-L), worse than 41.0% (iter 2) and 41.9% (iter 3).” Multiple iterations matter.
  - Human vs model-generated data (Figure 6 left; Section 5.3)
      > “On a matched 5K-question subset: SFT(5K) < ReST*(5K) < ReST_EM(5K), showing model-generated data beats human even when controlled to one solution per question; multiple solutions and iterations add more gains.”
  - Cross-model distillation (Figure 6 right; Section 5.3)
      > “PaLM 2-S fine-tuned on PaLM 2-L’s generated data outperforms SFT on human data—even with fewer questions—and beats self-generated ReST_EM(S). Using multiple L-generated solutions per problem boosts further.”

- Ablations and robustness checks
  - ReST vs ReST_EM (Figure 7):
    > “Similar APPS performance in-domain, but ReST_EM substantially better on HumanEval transfer, validating the ‘restart from base’ design.”
  - Dataset size (Figure 8 left):
    > “Even 1,000 MATH questions produce large gains; performance improves with more questions, with some variance at 4,000.”
  - Which problems improve most? (Figure 8 right):
    > “Largest gains on medium and hard MATH problems; improvements at all difficulty levels.”

- Overall assessment
  - The experiments are extensive across domains and scales, include careful ablations, and tie back to the EM design (multiple iterations, reward filtering, base-model resets). The evidence convincingly supports the claims that (1) self-generated, reward-filtered data can beat human data for problem solving; (2) gains scale with model size; (3) the approach improves pass@K/majority voting and transfers without hurting general ability.

## 6. Limitations and Trade-offs
- Assumptions and prerequisites
  - Requires a dataset of input problems (prompts). While smaller than full human solution corpora, these must still be curated for new tasks (Discussion; Section 6).
  - Requires an automatic or reliably computed scalar reward:
    - Binary correctness checks are straightforward for math answers and code unit tests, but not available for many open-ended tasks (Section 6).
- Overfitting and iteration count
  - Multiple iterations can overfit small training sets; APPS (∼2.3K problems) shows regression after the first iteration (Figures 3–4). Stopping criteria and per-problem sampling caps help but do not eliminate this risk.
- Reward granularity
  - Binary rewards may underutilize informative partial credit (e.g., partially correct reasoning). The EM derivation supports real-valued rewards, but experiments here are binary, which can limit learning signal richness (Section 3 Remark; Section 6).
- Compute trade-offs
  - While cheaper than online RL and much cheaper than pretraining, ReST_EM still requires:
    - Sampling many solutions per problem (32–64 here).
    - Running validators/test suites (especially heavy for code).
    - Repeating over multiple iterations and fine-tunes.
- Gap to best-of-K
  - Even with improvements, pass@1 lags far behind pass@K if K is large; closing that gap remains future work (Section 6).

## 7. Implications and Future Directions
- Field-level impact
  - Establishes a principled, scalable recipe for LLM self-improvement when an automatic reward exists. This reframes “beyond human data” as a practical path for high-stakes reasoning tasks like competition math and code synthesis.
  - The EM view clarifies why the method is stable and scalable: decoupled sampling and optimization with a simple, weighted maximum likelihood objective (Equations 2–3).

- Practical applications
  - Any domain with an automatic checker:
    - Programming tasks with unit tests (APPS-like settings, competitive programming).
    - Structured problem solving with verifiers (math, logic puzzles, theorem proving with proof checkers).
    - Data wrangling, transformation, and query generation where outputs can be validated automatically (e.g., SQL execution).
  - Model distillation pipelines:
    - Use a larger LLM to generate correctness-filtered training sets to upgrade smaller models efficiently (Figure 6 right).

- Research directions
  - Stronger E-steps:
    - Replace pure sampling with search/planning (Expert Iteration, MCTS, verifier-guided search) to collect higher-quality candidate solutions before filtering (Section 4).
  - Beyond binary rewards:
    - Incorporate graded, dense, or learned reward models to provide richer signals, while keeping the EM decomposition (RWR-style or learned `f(r)`, Section 4).
  - Drift control and generalization:
    - Further study of “reset to base” vs. “continue from last” across tasks; adaptive mixing of base and last-iteration checkpoints to balance specialization and transfer (Figure 7).
  - Automating the pipeline:
    - LLM-based reward generation, problem harvesting, and unit-test synthesis to extend this approach to more open-ended tasks (Section 6).
  - Closing the pass@1 vs pass@K gap:
    - Training objectives or decoding strategies that explicitly optimize for the most probable correct solution, not just encourage a pool of diverse candidates (Section 6).

In sum, ReST_EM demonstrates that with the right framing (EM for RL), simple correctness signals and self-generated data can deliver robust, scalable improvements on hard problem-solving tasks, often surpassing human-written data and transferring broadly without degrading general capabilities. The approach is practical today where automatic checking exists and points to a broader future in which LLMs bootstrap themselves beyond the limits of curated human corpora.
