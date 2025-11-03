# RLHF Workflow: From Reward Modeling to Online RLHF – A Comprehensive Practical Alignment Recipe of Iterative Preference Learning

**ArXiv:** [2405.07863](https://arxiv.org/abs/2405.07863)

## 🎯 Pitch

This paper delivers the first truly practical and fully reproducible end-to-end workflow for online, iterative Reinforcement Learning from Human Feedback (RLHF) that works without direct human annotation. By building a strong, proxy preference model from diverse open-source datasets, the authors enable efficient on-policy preference collection, driving iterative RLHF that overcomes the common out-of-distribution pitfalls of static offline pipelines. Their method advances open-source large language model alignment, consistently surpassing offline RLHF baselines in conversational tasks while preserving reasoning ability, thus making state-of-the-art alignment accessible and scalable to the broader community.

---

## 1. Executive Summary (2-3 sentences)
This work delivers a complete, reproducible recipe for online, iterative Reinforcement Learning from Human Feedback (`RLHF`) that does not require new human labels: it trains a proxy preference model from open-source datasets and uses it to drive on-policy preference collection and learning. Implemented on `LLaMA‑3‑8B`, the workflow consistently outperforms strong offline baselines (e.g., vanilla `DPO`) on conversational benchmarks while largely preserving reasoning performance (Tables 2–3).

## 2. Context and Motivation
- Problem the paper addresses
  - Most open-source RLHF pipelines are offline: they train on a fixed, pre-collected preference dataset and never query for new feedback during learning (Section 1.1, Eq. 6). This causes out-of-distribution (OOD) issues once the policy moves far from the data used to train the reward signal.
  - Online iterative RLHF, which periodically deploys the current policy to collect new preference data, has shown strong gains in closed or resource-rich settings (e.g., PPO-based pipelines). However, end-to-end, reproducible, open-source recipes for online RLHF—especially ones that avoid expensive human labeling—have been lacking (Section 1.2).

- Why it matters
  - Practical significance: Iterative, on-policy feedback mitigates distribution shift as the model’s behavior changes, addressing the common failure mode where a policy over-optimizes on stale or mismatched reward signals (Section 1.1; also Figure 13 cited from Bai et al. 2022a shows very large density ratios).
  - Theoretical significance: Under a KL-regularized objective, online preference collection can be sample-efficient when exploration is guided strategically (Theorem 1; Section 3.2).

- Prior approaches and shortcomings
  - PPO-style DRL pipelines: powerful but notoriously fragile, implementation-sensitive, and memory-hungry; they require loading actor, critic, reward, and reference models at once (Section 1.1). Hyperparameter tuning is difficult for LLMs.
  - Offline direct preference learning (e.g., `DPO`): stable and efficient but limited by the static dataset; suffers when the policy’s distribution diverges from the data used to train the reward/preference model (Section 1.1).
  
- Positioning
  - This work builds a practical online iterative RLHF recipe around direct preference learning rather than PPO, and crucially replaces cost-prohibitive human annotation with a proxy preference model trained on diverse open datasets (Sections 1.3 and 2; Figure 1, left-to-right flow).

## 3. Technical Approach
The workflow spans three components: reward/preference modeling, a theoretically motivated online data-collection framework (main agent + enhancer), and a practical instantiation using `DPO` with best‑of‑n/worst‑of‑n sampling.

- RLHF setup and objective
  - Policy `π(a|x)` generates a response `a` to prompt `x`. A fixed reference policy `π0` is the SFT-initialized model.
  - A preference oracle `P` (real human or proxy) returns which of two responses is preferred (Definition 1).
  - Preferences are modeled by the Bradley–Terry (`BT`) assumption: the chance response `a1` is preferred over `a2` is `σ(r*(x,a1) − r*(x,a2))`, where `r*` is a latent reward and `σ` is the logistic function (Definition 2; Eq. 1).
  - The alignment target is a KL‑regularized objective: maximize expected reward minus a KL penalty from `π0` (Eq. 2). The corresponding optimal policy has exponential tilting of `π0` by reward (Eq. 3).

- Why offline learning is insufficient
  - Offline data come from fixed behavior policies (Eq. 6). During alignment, policies quickly move far from `π0`—density ratios can exceed `exp(25)`, making learned rewards unreliable off-distribution (Section 1.1).
  
- Online iterative RLHF (theoretical framework; Section 1.2 and Section 3.2)
  - Each iteration t:
    1) Update policy pair `(π1_t, π2_t)` based on all data so far.
    2) Collect m new comparisons by sampling prompts, sampling from both policies, and querying preferences.
    3) Add them to the dataset and repeat.
  - Non-symmetric “main agent” and “enhancer” (Algorithm 1; Section 3.2):
    - Main agent `π1_t`: the exploitation policy, i.e., the best policy under the current maximum-likelihood reward estimate `r_MLE` (Eq. 7).
    - Enhancer `π2_t`: an exploration policy chosen to maximize uncertainty relative to `π1_t` while remaining within a KL budget (Eq. 8). Intuition: collect data that is informative where the model is currently unsure.
  - Theoretical guarantee (Theorem 1): with suitable batch size and exploration, after Õ(d_e) iterations (d_e is a problem complexity measure; linear case reduces to feature dimension d), one can find a policy whose KL‑regularized value `J(π)` is within `ϵ` of optimal.

- Reward and preference modeling as human-feedback approximation (Section 2)
  - Two variants are trained on open datasets:
    - `BT reward model` (`r_θ`): predicts a scalar reward; trained by logistic loss on pairwise preferences (maximum likelihood for Bradley–Terry; Section 2.1).
    - `Preference model`: reformulates each pair as a single classification instance (“Which response is better, A or B?”) and trains the LLM with next-token prediction on that label (Section 2.1; Figure 2).
  - Data mixtures:
    - `mix1`: HH‑RLHF + SHP + UltraFeedback + Summarization (Section 2).
    - `mix2`: a larger, more diverse mix adding safety, math, and code preference data (Table 5 lists components and stats).
  - RewardBench evaluation (Table 1): the `LLaMA‑3‑8B` preference model trained on `mix2` outperforms BT reward on reasoning and is strong across categories.

- Practical online RLHF implementation (Algorithm 2; Section 3.3)
  - Oracle optimizer: use `DPO` to approximate the `r_MLE`-optimal policy (avoids PPO’s complexity).
  - On each iteration:
    1) Train `π_t` with `DPO` on all accumulated preference data (historical + new), using the SFT model `π0` as the reference. Hyperparameters: 2 epochs per iteration, cosine LR schedule, LR peak `5e-7`, warm-up `0.03`, global batch size `128`, KL coefficient `η = 0.1` (Section 3.3).
    2) For each of `m` prompts, sample `n` responses from `π_t` at two temperatures (0.7 and 1.0; step 4), and rank them with the reward model `r`.
    3) Form one training pair per prompt using the best-ranked response vs the worst-ranked response (best-of-n/worst-of-n; step 5), then add all `m` pairs to the dataset.
  - Exploration in practice (Section 3.3):
    - Best‑of‑n introduces diversity without excessive KL drift; the KL between base sampling and best‑of‑n is bounded by `log n − (n−1)/n`, typically much smaller in practice.
    - The paper goes further: it jointly uses the best‑of‑8 as `π1_t` and the worst‑of‑8 as `π2_t`, maximizing their difference to collect highly informative pairs (Figure 4). Pairs with identical responses are dropped.
  - Prompting and data generation details:
    - 60k prompts selected from UltraFeedback, HelpSteer, OpenOrca, UltraInteract, Capybara, and DIBT‑10K (Section 3.3).
    - Three iterations; each uses 20k prompts; for each prompt, 16 responses are generated (`20k × 16` per iteration); generation via `vLLM`, max length 2048, temperatures 1.0/0.7, no top‑k/top‑p (Section 3.3).

- Handling verbosity bias (Section 2.2 and Section 4)
  - Length bias diagnosis: reward–length correlation is positive for both UltraRM‑13B and the BT reward (Figure 3; mean Pearson 0.19 vs 0.06 respectively).
  - Mitigation: add a simple length penalty during data filtering/ranking, using `r_e(x,a) = r̂(x,a) − λ|a|` (Eq. 9), where `|a|` is response length in characters. This yields a more concise model variant.

## 4. Key Insights and Innovations
- Low-cost online RLHF via proxy preferences
  - Innovation: replace human-in-the-loop feedback with a proxy preference model trained on diverse, open datasets (Section 1.3; Section 2). This makes online RLHF feasible for the open-source community.
  - Significance: preserves the on-policy exploration benefits of online RLHF without the labeling budget. Table 1 shows the proxy models are competent, especially on safety and reasoning with the `mix2` dataset.

- Main agent + enhancer framework with uncertainty-aware exploration
  - Innovation: a non-symmetric, two-policy design (Algorithm 1) that separates exploitation (best current policy under `r_MLE`) from exploration (policy chosen to maximize uncertainty under a KL constraint; Eq. 8).
  - Significance: Theorem 1 guarantees sample-efficient convergence in the KL-regularized objective when exploration is strategic (Section 3.2), grounding the design beyond heuristics.

- Practical instantiation that is stable, efficient, and easy to reproduce
  - Innovation: instantiate the enhancer using best‑of‑n/worst‑of‑n selection and temperature variation, with `DPO` as the oracle optimizer (Algorithm 2; Section 3.3). Avoids PPO’s instability and memory footprint.
  - Significance: a working recipe using public toolchains (`TRL`, `vLLM`) and modest hyperparameters, enabling others to reproduce and extend.

- Diagnosis and control of verbosity bias in iterative RLHF
  - Innovation: explicit analysis of reward–length correlation (Figure 3) and a simple, effective length-penalized ranking during data collection (Eq. 9).
  - Significance: improves length-controlled win-rates substantially (Table 4), and clarifies judge/benchmark biases (e.g., Chat Arena-Hard tends to reward verbosity; Section 4.2 and Table 4).

## 5. Experimental Analysis
- Evaluation methodology
  - Proxy evaluator quality (Section 2.2):
    - `RewardBench` measures reward/preference model accuracy across Chat, Chat‑Hard, Safety, Reasoning.
    - Table 1 shows the `LLaMA‑3‑8B` preference model trained on `mix2` achieves strong results, e.g., Chat‑Hard 89.7 and Reasoning 94.7.
  - Policy quality (Section 4):
    - Conversational benchmarks:
      - `AlpacaEval‑2` (win rate vs GPT‑4; also length-controlled LC version).
      - `MT-Bench` (average judge score 1–10 across two turns).
      - `Chat‑Arena‑Hard` (win rate on curated difficult prompts).
    - Academic benchmarks to probe alignment tax:
      - `GSM‑8K` (math), `MMLU` (knowledge), `HumanEval` and `MBPP` (coding), `TruthfulQA` (truthfulness), `ARC` (reasoning). Shot settings summarized in Table 6.

- Main quantitative results
  - Conversational improvements over offline baselines (Table 2):
    - Iterative RLHF (8B) vs their own DPO baseline (8B):
      - LC AlpacaEval‑2: 31.3 vs 22.5 (+8.8 points).
      - MT‑Bench: 8.46 vs 8.17 (+0.29).
      - Chat‑Arena‑Hard: 29.1 vs 22.4 (+6.7).
    - Iterative RLHF (8B) also surpasses `LLaMA‑3‑8B‑instruct` on LC AlpacaEval‑2 (31.3 vs 22.9) and Chat‑Arena‑Hard (29.1 vs 20.6), with a small MT‑Bench edge (8.46 vs 8.16).
    - It even outperforms much larger open-source aligned models on some metrics (e.g., `Tulu‑2‑DPO‑70B`: LC 21.2, Arena-Hard 15.0; `Mixtral‑8×7B‑it`: LC 23.7, Arena-Hard 23.4).
  - Academic performance and alignment tax (Table 3):
    - Iterative RLHF vs SFT baseline:
      - GSM‑8K 80.7 vs 74.2; MMLU 65.3 vs 64.7; TruthfulQA 60.4 vs 53.4; ARC 64.3 vs 61.4; minor changes on HumanEval/MBPP.
    - Takeaway: online DPO alignment does not degrade—and sometimes slightly boosts—reasoning/knowledge metrics for this setup.
  - Iteration-wise gains (Figure 8): steady improvements across MT‑Bench, AlpacaEval‑2 (both overall and LC), and Chat‑Arena‑Hard as iterations progress, consistent with the intended benefits of online data collection.

- Ablations and robustness checks
  - Length penalty during ranking (Table 4; Eq. 9):
    - `Ours (no penalty)`: LC 31.3, Arena‑Hard 29.1, avg response length 656 chars.
    - `Ours‑concise (λ = 0.001)`: LC 38.1 (+6.8), Arena‑Hard 22.1 (−7.0), avg length 382 chars; modest improvements on HumanEval and MBPP, and stable MMLU. Interpretation: length control improves LC AlpacaEval‑2 but can hurt on judges favoring verbosity (Arena‑Hard).
  - Reward model choice (Table 4 and Figure 3):
    - Using UltraRM‑13B for ranking yields longer responses (avg length 745) and lower academic scores; it performs better than the concise variant on Arena‑Hard but worse on LC AlpacaEval‑2, consistent with verbosity bias.
    - Length–reward correlation analysis (Figure 3) explains these shifts.
  - Preference vs reward model accuracy (Table 1):
    - The pairwise preference model excels on Reasoning and Safety; the BT reward trained on `mix2` is also strong. In practice, ranking n responses is simpler with a scalar reward, motivating its use for data filtering in Algorithm 2 (Section 3.3).

- Do the experiments support the claims?
  - Yes, on two fronts:
    - Efficacy of online iterative RLHF without human raters: clear, repeated gains over the offline DPO baseline on multiple conversational benchmarks (Table 2) with reasonable academic performance (Table 3).
    - Practicality and bias control: the workflow is executable with public tools/datasets; verbosity is measured (Figure 3) and mitigated (Table 4). Still, judge and dataset biases remain a caveat (Remark 1 in Section 4).

## 6. Limitations and Trade-offs
- Reliance on proxy labels
  - All online preferences come from learned reward/preference models, not humans (Section 1.3). This can encode dataset and judge biases (e.g., response length, stylistic preferences). Figure 3 and Table 4 document meaningful length bias.
  
- Exploration heuristic vs theory
  - The uncertainty term `Γ` guiding the enhancer has no closed form beyond simple linear cases (Section 3.2). The practical best‑of‑n/worst‑of‑n and temperature tricks (Section 3.3) are heuristics that approximate the spirit of uncertainty maximization without explicit quantification.

- Evaluation biases and external validity
  - Several benchmarks rely on LLM judges (e.g., GPT‑4 in AlpacaEval‑2 and MT‑Bench). The paper notes judge configuration sensitivity and that Arena‑Hard appears to reward verbosity (Section 4.2, Table 4; Remark 1).

- Compute and scaling considerations
  - While cheaper than PPO, the workflow still requires substantial on-policy generation (e.g., `20k × 16` responses per iteration; Section 3.3). Best‑of‑n selection increases inference cost linearly in `n`. Scaling to much larger models or many more iterations will raise costs.

- Scope and safety
  - The paper aligns for helpfulness and general quality; it does not present dedicated safety evaluations beyond RewardBench safety accuracy (Table 1) or multi-objective trade-offs. It also focuses on single‑preference scalarization for ranking (Section 5 suggests multi-head rewards as future work).

## 7. Implications and Future Directions
- How this changes the landscape
  - It turns online RLHF from a resource-intensive, PPO-centered practice into a reproducible, low-cost pipeline for the open-source community by leveraging proxy preference models and stable direct preference learning. Others can now iterate on exploration strategies, reward shaping, and data sourcing without human raters in the loop.

- Enabled research avenues
  - Better proxy signals:
    - Multi-head or aspect-specific reward models (helpfulness, safety, reasoning) and controlled aggregation (Section 5).
    - Improved pairwise preference modeling with richer rubrics (Section 2.1 mentions rubric-based formatting as a possible improvement).
  - Principled exploration:
    - Closer approximations to the uncertainty-guided enhancer (Eq. 8), e.g., biasing `DPO` losses to encourage optimism (Section 3.3 cites recent works).
    - Active selection of prompts and response spaces to maximize information gain.
  - Bias control and evaluation:
    - Systematic control of verbosity and other stylistic artifacts; development of length-controlled versions of more benchmarks (Section 4.2 notes Arena‑Hard’s verbosity sensitivity).
  - Human-in-the-loop upgrades:
    - Hybrid pipelines where proxy models bootstrap iterations and a small budget of targeted human labels corrects biases or anchors safety.
  
- Practical applications
  - Organizations can align domain-specific assistants (e.g., coding, math tutoring, customer support) by training task-focused proxy preference models and running the iterative recipe with on-policy data generation in their domain. The paper’s code, datasets, and hyperparameters (Sections 2–3; Appendix B) provide a starting point.

> Notable headline results:
> - “Ours (Iterative RLHF)” improves LC AlpacaEval‑2 from 22.5 (DPO baseline) to 31.3 and Chat‑Arena‑Hard from 22.4 to 29.1, with MT‑Bench rising from 8.17 to 8.46 (Table 2).
> - With a length penalty (λ = 0.001), the LC AlpacaEval‑2 win rate further increases to 38.1, while Arena‑Hard decreases to 22.1; average response length drops from 656 to 382 (Table 4).
> - Academic tasks remain competitive: GSM‑8K 80.7, MMLU 65.3, TruthfulQA 60.4, ARC 64.3 (Table 3).

Overall, this paper delivers both a theory-backed framework (Algorithm 1; Theorem 1) and a practically effective recipe (Algorithm 2) for open, iterative RLHF without human raters, along with careful analysis of an important emergent bias (verbosity) and a simple mitigation.
