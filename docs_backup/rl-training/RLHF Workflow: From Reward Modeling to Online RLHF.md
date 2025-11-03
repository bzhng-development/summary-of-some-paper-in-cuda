# RLHF Workflow: From Reward Modeling to Online RLHF

**ArXiv:** [2405.07863](https://arxiv.org/abs/2405.07863)
**Authors:** Hanze Dong, Wei Xiong, Bo Pang, Haoxiang Wang, Han Zhao, Yingbo Zhou, Nan Jiang, Doyen Sahoo, Caiming Xiong, Tong Zhang
**Institutions:** Siebel School of Computing and Data Science, University of Illinois (inferred)

## 🎯 Pitch

This paper introduces a reproducible workflow for online Reinforcement Learning from Human Feedback (RLHF) that replaces costly human labeling with a high-quality proxy preference model, enhancing the conversation quality of large language models without sacrificing reasoning or knowledge performance. By implementing a grounded exploration-exploitation scheme, it democratizes advanced RLHF techniques, allowing resource-constrained teams to achieve state-of-the-art results without extensive human input, thus significantly broadening access to cutting-edge AI alignment methods.

---

## 1. Executive Summary (2-3 sentences)
This paper delivers an end‑to‑end, reproducible workflow for online, iterative Reinforcement Learning from Human Feedback (RLHF) that replaces costly human raters with a high‑quality proxy preference model and uses a theoretically grounded exploration–exploitation scheme. Practically, the recipe improves an 8B-parameter LLM’s conversation quality over strong baselines and keeps (or improves) scores on reasoning and knowledge benchmarks, while exposing concrete tactics to curb reward-driven verbosity (Sections 2–4; Tables 1–4; Figures 1, 3–4, 8–9).

## 2. Context and Motivation
- Problem addressed
  - Most open-source RLHF is still “offline”: models are trained once on a fixed, pre-collected set of preferences and are not updated with new, on-policy data during training. This causes out-of-distribution (OOD) issues when the policy drifts far from the data distribution (Section 1.1).
  - Deep-RL–based RLHF (e.g., PPO) is resource‑intensive and fragile to hyperparameters and implementation details, making it hard to reproduce in the open-source community (Section 1.1).
  - Online learning with live human feedback is ideal but expensive. The paper fills this gap by building a proxy preference model from diverse open datasets and using it to power an online, iterative RLHF loop (Sections 1.3 and 2).

- Why it matters
  - Real-world impact: Online preference learning has been key to performance in leading systems (e.g., Claude, LLaMA‑2), but practical open recipes have been lacking. The paper’s workflow enables resource‑constrained teams to reproduce the benefits of online RLHF without a human-labelling pipeline (Sections 1.2 and 1.3; Figure 1).
  - Theoretical significance: The paper connects iterative preference learning to a KL‑regularized objective and provides a formal exploration–exploitation formulation with finite‑sample guarantees (Equations 2–3; Algorithm 1; Theorem 1 in Section 3.2).

- Prior approaches and their shortcomings
  - PPO-style RLHF: powerful, but unstable and expensive; requires multiple large models in memory (actor, critic, reward, reference), tough for open projects (Section 1.1).
  - Offline direct preference learning (e.g., `DPO`): simpler and more stable than PPO but still limited by fixed datasets; struggles when the optimized policy diverges (large density ratios are reported during RLHF, indicating substantial distribution shift; Section 1.1).
  - Prior online efforts existed mainly in proprietary or DRL‑heavy settings (Claude, LLaMA‑2); open-source iterative DPO with clear guidance remained under-explored (Section 1.2).

- Positioning
  - The paper sits at the intersection of theory and practice:
    - It adopts the KL‑regularized RLHF formulation with a Bradley–Terry (BT) preference model abstraction (Definitions 1–2; Equations 1–3).
    - It proposes a practical online direct-preference workflow with an exploration “enhancer” and a high-quality, open preference model to approximate human raters (Algorithms 1–2; Sections 2–3).
    - It provides a full, replicable recipe—data curation, reward/preference modeling, on-policy sampling, and iterative DPO training—plus ablations on reward length bias (Sections 2–4; Table 4; Figure 3).

## 3. Technical Approach
The workflow has three major parts: (A) build a strong proxy preference model; (B) run an online, iterative preference-learning loop with exploration; (C) apply practical engineering choices that make the loop efficient and stable.

A. Build the proxy preference model (Section 2)
- Terminology
  - `Preference oracle`: a function that, given a prompt `x` and two responses `a1, a2`, returns which one is preferred (Definition 1).
  - `Bradley–Terry (BT) model`: a canonical model for pairwise preferences where the probability that `a1` is preferred over `a2` depends on a latent scalar reward `r*(x,a)` via a sigmoid of the reward difference (Definition 2; Equation 1).
- Two model families are constructed on top of `LLaMA‑3‑8B-Instruct`:
  1) BT reward model: replaces the final layer with a scalar “reward head” and trains by maximum likelihood on pairwise comparisons (negative log-sigmoid of reward differences), i.e.,
     > `LRM(θ) = − E_{(x,aw,al)} log σ(rθ(x,aw) − rθ(x,al))` (Section 2.1).
     - Training config: 1 epoch, global batch 512, lr `2e-6`, cosine schedule, warmup `0.03` (Section 2.1).
  2) Pairwise preference model (“LLM-as-classifier”): formats `(x, a1, a2)` as an instruction with label `A` or `B`, fine-tuned to predict the preferred side (Section 2.1).
     - Inference computes `pA/(pA+pB)` as the preference probability.
     - Training config: packed blocks of length `3072`, global batch `128`, lr `5e-6`, cosine lr, warmup `0.03`, 1 epoch.
- Data mixtures (Table 5; Section B.1)
  - `mix1`: HH-RLHF + SHP + UltraFeedback + Summarization.
  - `mix2` (larger and more diverse): adds safety, math, and coding preferences (e.g., UltraInteract, CodeUltraFeedback). Filtering removes noisy samples and low-margin pairs (Section B.1).
- Why two model types?
  - BT rewards are efficient for ranking many candidate responses (`O(n)` scoring).
  - Pairwise models can better capture complex preferences in reasoning tasks (Table 1 shows stronger “Reasoning” accuracy).

B. Iterative online preference optimization (Section 3)
- Objective background
  - The RLHF target balances reward and staying close to the initial SFT policy `π0`:
    > `J(π) = E_{x ~ d0} [ E_{a ~ π} r*(x,a) − η D_KL(π(·|x) || π0(·|x)) ]` (Equation 2),
    with an intractable closed-form solution that reweights `π0` by the exponentiated reward (Equation 3).
- Core idea: Learn from on-policy preferences with exploration (Algorithm 1, Figure 1).
  - At each iteration `t`:
    1) Main agent policy `π1_t` is the exploitation choice—best under the current reward estimate (Equation 7).
    2) Enhancer policy `π2_t` is the exploration choice—chosen to maximize an uncertainty measure `Γ` relative to `π1_t`, while keeping a moderate KL divergence from `π1_t` (Equation 8).
    3) Collect `m` new preference data points by sampling `(a1, a2)` from `(π1_t, π2_t)` on prompts `x ~ d0`, then query preferences from the proxy model (Algorithm 1, Step 5).
  - Intuition: The enhancer perturbation seeks high-uncertainty response regions so each batch contributes new information; the KL constraint prevents degenerate exploration (Section 3.2).
  - Theorem (informal): With suitable `m` and hyperparameters, after `Õ(d_e)` iterations the KL‑regularized value of some iterate approaches the optimum to accuracy `ε`, where `d_e` is the complexity of the function class (Theorem 1; Section 3.2).

C. Practical recipe (Section 3.3; Algorithm 2; Figure 4)
- Replace the theoretical oracle with stable, low‑overhead direct preference optimization:
  - Train `π_t` using `DPO` on all data so far (`Doff ∪ D1:t−1`), with `π0` as the fixed reference (Algorithm 2, Step 3).
  - Why DPO? It optimizes a surrogate derived from the KL‑regularized objective (Equation 5), is stable, and avoids multi-model PPO overhead (Section 1.1).
- Exploration without computing uncertainty explicitly:
  - Use ensemble-style variation through sampling and selection:
    - For each prompt, sample multiple responses and rank them using the proxy reward. Construct a training pair from the best and the worst candidates—this stretches the “preference margin,” promoting informative learning signals (Algorithm 2, Steps 4–5; Figure 4).
    - Temperature mixing: sample half the candidates with temperature `1.0` and half with `0.7` to diversify candidates (Algorithm 2, Step 4).
    - Best-of‑`n` / worst-of‑`n`: in practice the paper uses this as `π1_t` (best) and `π2_t` (worst) induced by `π_MLE_t`, which encourages large, informative differences while keeping policies related (Section 3.3).
- Training and generation specifics:
  - DPO details: 2 epochs per iteration, global batch 128, cosine schedule with peak lr `5e-7`, warmup `0.03`, KL coefficient `η=0.1`; they warm‑start each iteration from the previous model while keeping `π0` as the reference (Section 3.3).
  - Data generation: 60k prompts total, 3 iterations with 20k prompts each; candidates are generated with vLLM, max generation length 2048, temperatures 1.0 and 0.7 (Section 3.3). Note: Appendix mentions “20K × 16 responses per iteration,” while Section 3.3 also describes best-of‑8/worst-of‑8—this indicates `n` between 8 and 16 depending on the iteration or run configuration.

Clarifying uncommon terms used:
- `DPO` (Direct Preference Optimization): an algorithm that trains a policy directly on pairwise preferences by maximizing the log odds that preferred responses are more likely than dispreferred ones under the policy relative to a fixed reference (`π0`), effectively baking in KL regularization (Equation 5, Section 1.1).
- `Rejection sampling` / `best-of‑n`: generate `n` samples and choose the one with the highest reward; here extended to also use the worst candidate to form a strong contrast (Section 3.3).
- `Preference model`: a classifier over pairs (`x, a1, a2`) that predicts which `a` is preferred; different from a scalar reward model (Section 2.1).

## 4. Key Insights and Innovations
1) A reproducible online iterative preference-learning workflow that avoids PPO (Sections 3.2–3.3; Algorithm 2; Figure 4)
- What’s new: Moves the “iterative” part of RLHF into a DPO-based pipeline with an explicit exploration mechanism via best/worst-of‑`n` sampling and temperature mixing, rather than relying on PPO.
- Why it matters: Delivers stability and lower compute/memory footprint while retaining the well-known benefits of online data collection (Section 1.2), making the approach usable by open-source teams.

2) A strong, open proxy for human feedback using diverse datasets and two modeling strategies (Section 2; Table 1; Table 5)
- What’s new: Trains both a scalar BT reward model and a pairwise preference model from a carefully filtered and diverse set of open preference datasets (“mix2”: safety, math, coding included; Table 5).
- Why it matters: The proxy is accurate on RewardBench and supports on-policy learning without human labellers. The pairwise preference model notably excels on reasoning (“Reasoning” accuracy 94.7 vs 86.4 for BT; Table 1).

3) Theoretical framing with an “enhancer” policy and finite‑sample guarantees (Section 3.2; Algorithm 1; Theorem 1)
- What’s new: The main agent exploits the current reward estimate, while the enhancer explores under a KL‑bounded uncertainty criterion, bringing classical exploration ideas into preference optimization.
- Why it matters: Gives conceptual clarity and a path to principled exploration beyond simple heuristics. Although uncertainty is approximated pragmatically, the framework justifies exploration choices.

4) Diagnosing and mitigating length bias in reward models and iterative RLHF (Figure 3; Table 4)
- What’s new: Direct analysis of reward–length correlations (Figure 3) and a simple length‑penalized reward `re(x,a)= r̂(x,a) − λ|a|` for filtering (Equation 9).
- Why it matters: RLHF often amplifies verbosity; the length penalty improves length‑controlled win rate on AlpacaEval‑2 (from 31.3 to 38.1) and shortens responses substantially (average length: 656 → 382 characters; Table 4), while maintaining or improving several academic metrics.

## 5. Experimental Analysis
- Evaluation design (Sections 2 and 4; Appendix B.2; Tables 1–4)
  - Preference model quality: RewardBench across “Chat,” “Chat-Hard,” “Safety,” and “Reasoning” (Table 1).
  - Conversational ability: AlpacaEval‑2 (length-control win rate), MT‑Bench (GPT‑judged score), Chat‑Arena‑Hard (win rate vs GPT‑4 judge) (Table 2; Appendix B.2).
  - Reasoning and knowledge: GSM‑8K (math), MMLU (general knowledge), HumanEval/MBPP (code), TruthfulQA (truthfulness), ARC (reasoning) (Table 3; Appendix B.2).
  - Iterative dynamics: progress over three iterations (Figure 8).
  - Reward bias: reward–length correlation heatmaps (Figure 3).
  - Ablations: length penalty and impact of different reward models (UltraRM‑13B vs theirs) (Table 4).

- Key quantitative findings
  - Proxy preference model strength (Table 1):
    > With `mix2`, the BT reward model reaches “Chat 99.4, Chat‑Hard 65.1, Safety 87.8, Reasoning 86.4,” and the pairwise preference model reaches “Chat 98.3, Chat‑Hard 65.8, Safety 89.7, Reasoning 94.7.”
    - Takeaway: the pairwise model is notably better on reasoning tasks, justifying its use when reasoning quality matters.
  - Online iterative RLHF vs baselines (Table 2):
    > The 8B model with iterative DPO scores “LC AlpacaEval‑2 31.3, MT‑Bench 8.46, Chat‑Arena‑Hard 29.1.”
    - Comparisons:
      - vs its own DPO baseline (offline): “22.5, 8.17, 22.4” ⇒ consistent gains across all three.
      - vs LLaMA‑3‑8B‑Instruct: “22.9, 8.16, 20.6” ⇒ substantial improvements.
      - vs several 7B–45B open models: competitive or superior on conversational metrics; smaller than LLaMA‑3‑70B‑Instruct (as expected) but much better than GPT‑3.5‑turbo in Chat‑Arena‑Hard (29.1 vs 18.9).
  - Academic benchmarks (Table 3):
    > Iterative DPO: “GSM‑8K 80.7, MMLU 65.3, HumanEval 64.6, TruthfulQA 60.4, ARC 64.3, MBPP 60.8.”
    - Outcome: no major alignment tax; in several cases, it improves over the SFT baseline (e.g., GSM‑8K 80.7 vs 74.2; TruthfulQA 60.4 vs 53.4).
  - Iteration-by-iteration gains (Figure 8):
    > Steady increases across iterations for MT‑Bench and both AlpacaEval‑2 variants; Chat‑Arena‑Hard also increases monotonically.
  - Length bias and mitigation (Figure 3; Table 4):
    > Reward–length correlation: UltraRM‑13B shows stronger positive correlation (mean ≈ 0.19) than the paper’s BT reward (mean ≈ 0.06) (Figure 3).
    > With length penalty `λ=0.001`, LC AlpacaEval‑2 improves from 31.3 → 38.1, and average response length drops from 656 → 382 characters (Table 4). Some trade-offs appear (Chat‑Arena‑Hard declines).

- Do the experiments support the claims?
  - Yes, for three reasons:
    - The online loop provides consistent, across‑the‑board improvements in conversational metrics relative to the same model trained offline with DPO (Table 2; Figure 8).
    - The proxy preference model is demonstrably competent, especially on reasoning, supporting its role as an effective labeler (Table 1).
    - The length‑bias analysis identifies a real failure mode and shows a practical mitigation that improves a length‑controlled metric without eroding academic performance (Figure 3; Table 4).

- Robustness and ablations
  - Reward model choice matters: training with UltraRM‑13B leads to longer outputs and generally worse academic scores than the paper’s BT reward; this aligns with UltraRM’s stronger length bias and weaker reasoning accuracy (Table 1 vs Table 4).
  - Length penalty helps where verbosity is penalized (LC AlpacaEval‑2), but can hurt benchmarks favoring detailed answers (Chat‑Arena‑Hard) (Table 4).
  - The paper also cautions that benchmark scores can be sensitive to evaluation configuration and warns against over-interpreting leaderboard wins (Remark 1, Section 4.2).

## 6. Limitations and Trade-offs
- Reliance on a proxy preference model
  - Assumption: proxy preferences approximate human preferences well enough to guide online learning (Section 1.3; Table 1).
  - Risk: biases in the proxy (e.g., verbosity, safety strictness, reasoning blind spots) can shape the final policy. The length-bias analysis (Figure 3) shows this is a real concern; mitigation requires careful tuning (Table 4).

- Exploration is heuristic in practice
  - The theoretical uncertainty‑guided enhancer (Equation 8) is approximated by best/worst‑of‑`n` sampling with temperature mixing (Section 3.3). This is effective and simple but not the same as optimizing a principled uncertainty measure.

- Data generation cost
  - Online loops require generating many candidates per prompt and ranking them, which is inference‑heavy even without PPO’s training overhead (Section 3.3). The paper mitigates this with vLLM and modest `n`, but scaling to larger models or more iterations increases cost.

- Some configuration ambiguity
  - The number of samples per prompt appears as `n=8` in Section 3.3, while Appendix mentions “20K × 16 responses per iteration.” This does not undermine results but suggests that exact `n` varied across runs; reproducibility is still strong given shared code and details.

- Benchmark generalization and “alignment tax”
  - Although academic metrics do not degrade overall (Table 3), the paper acknowledges risks of benchmark overfitting and configuration sensitivity for automatic judges (Remark 1, Section 4.2).

## 7. Implications and Future Directions
- How this changes the field
  - Provides a credible, open, and efficient alternative to PPO‑based RLHF for achieving the benefits of online, on‑policy preference learning (Algorithms 1–2; Section 3.3).
  - Demonstrates that a well‑trained proxy preference model, built from diverse open datasets, can meaningfully replace human raters for iterative alignment without collapsing reasoning performance (Sections 2–4; Tables 1–3).

- Enabled follow‑up research
  - Better proxies and multi‑objective reward heads: explore multi‑head rewards (e.g., helpfulness, honesty, safety, conciseness) and dynamic activation policies (Section 5).
  - Principled exploration: replace best/worst‑of‑`n` with explicit uncertainty estimation or optimistic objectives adapted from RL theory (Section 3.2; related works cited in Section 3.3).
  - Controlling verbosity beyond penalties: integrate length- or brevity-aware objectives into DPO variants, or post‑training re-ranking with calibrated judges (Table 4; Figure 3).
  - Reasoning‑aware aligners: couple the preference model with program-of-thought or verifier signals for math/code, given the strong gains of the pairwise preference model on reasoning (Table 1).

- Practical applications
  - Open-source assistants: iterative improvement of chatbots with minimal labeling budgets.
  - Domain customization: plug domain‑specific preference datasets (e.g., legal, medical, coding) into the proxy model, then run the online loop to adapt style and safety constraints.
  - Continual alignment: periodically re‑run the online iterations as prompts and usage evolve, giving a maintainable path to keep models aligned over time.

> Bottom line: By (i) training a strong open preference model, (ii) instantiating an online exploration–exploitation loop with DPO, and (iii) engineering a reproducible recipe that diagnoses and reduces reward‑length bias, this paper makes online RLHF both attainable and effective for the broader community (Figures 1, 3–4, 8; Tables 1–4).
