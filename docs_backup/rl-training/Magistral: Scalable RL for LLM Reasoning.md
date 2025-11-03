# Magistral: Scalable RL for LLM Reasoning

**ArXiv:** [2506.10910](https://arxiv.org/abs/2506.10910)
**Authors:** Mistral‑AI, Abhinav Rastogi, Albert Q. Jiang, Andy Lo, Gabrielle Berrada, Guillaume Lample, Jason Rute, Joep Barmentlo, Karmesh Yadav, Kartik Khandelwal, Khyathi Raghavi Chandu, Léonard Blier, Lucile Saulnier, Matthieu Dinot, Maxime Darrin, Neha Gupta, Roman Soletskyi, Sagar Vaze, Teven Le Scao, Yihan Wang, Adam Yang, Alexander H. Liu, +78 others
**Institutions:** Mistral AI

## 🎯 Pitch

Magistral revolutionizes reinforcement learning for reasoning language models by eschewing traditional teacher distillation in favor of pure RL with verifiable text rewards. This approach not only boosts mathematical and coding performances dramatically but also preserves multimodal capabilities, providing a scalable, repeatable training pipeline for organizations without access to closed data models and pushing the frontiers of AI reasoning advancements.

---

## 1. Executive Summary
Magistral introduces a full, from‑scratch reinforcement learning (RL) stack for training “reasoning” large language models (LLMs) using only verifiable text rewards—no distillation from prior reasoning models. Trained on top of Mistral’s base models, pure RL boosts the 7B‑class `Mistral Medium 3` checkpoint to strong mathematical and coding performance (e.g., AIME’24 pass@1: 26.8 → 73.6), while preserving or improving multimodal and instruction‑following skills (Sections 5, 7.2–7.3; Table 2, Figure 10).

## 2. Context and Motivation
- Problem/gap
  - State‑of‑the‑art reasoning LLMs often rely on distilling chains‑of‑thought (CoT) from an existing teacher, then fine‑tuning with RL. This blurs how much RL alone contributes and complicates reproducibility.
  - Scaling online RL for LLMs is operationally hard: generation is heterogeneous and long; staying sufficiently on‑policy is tricky; and models can drift in format/language (Sections 1–3).
- Importance
  - Practical: High‑quality reasoning without proprietary teachers unlocks training pipelines for organizations that cannot access closed traces.
  - Scientific: Is pure RL (with only verifiable rewards) sufficient to significantly improve reasoning? Can RL on text preserve non‑text capabilities (vision, tools)? The paper provides concrete evidence (Sections 5, 7.2–7.3).
- Prior approaches and shortcomings
  - RLVR (Reinforcement Learning from Verifiable Rewards) recipes exist, but large‑scale successes (e.g., DeepSeek‑R1) used substantial SFT on teacher traces before RL (Introduction; Table 2).
  - Existing infrastructures often synchronize actors/learners, hurting throughput or on‑policyness when sequence lengths vary widely (Section 3).
- Positioning
  - Magistral presents: (i) an RL algorithmic recipe centered on GRPO with stability tweaks (Section 2.1), (ii) a compact but effective reward design for math and code plus format and language control (Section 2.2), (iii) an asynchronous, GPU‑to‑GPU online RL system (Figure 3), and (iv) curated verifiable datasets (Section 4). It tests two regimes: pure RL on `Mistral Medium 3` (“Magistral Medium”) and “SFT→RL” on `Mistral Small 3` (“Magistral Small”) (Section 5).

## 3. Technical Approach
The method combines an RL objective, verifiable rewards, an asynchronous training system, and curated math/code data.

- Core learning objective: GRPO with stability changes (Section 2.1)
  - GRPO (“Group Relative Policy Optimization”) removes a critic by computing a per‑prompt baseline from multiple generations (“a group”) of the current policy. Each generation receives a reward; the advantage is the reward minus the group mean.
  - Key modifications:
    - Remove KL penalty to a reference policy: avoids maintaining a reference model and the penalty that hinders large exploration (Section 2.1, “Eliminating KL divergence”).
    - Normalize loss by total tokens across the group to avoid length bias (Section 2.1, “Loss normalization”).
    - Normalize advantages at minibatch level to stabilize updates (Section 2.1, “Advantage normalization”).
    - Relax the PPO clipping upper bound using `Clip‑Higher`: allow probability increases up to `1 + ε_high` (set ∼0.26–0.28) while keeping a standard lower bound (`1 − ε_low`). This preserves entropy and encourages rare but valuable steps (Section 2.1, “Relaxing the trust region’s upper bound”).
    - Drop “non‑diverse” groups where all generations are equally correct or incorrect (zero advantage), which would add noise and weak gradients (Section 2.1, “Eliminating non‑diverse groups”).
  - The final loss clips the likelihood ratio with asymmetric bounds and requires at least two generations in the group to have different rewards (final equation in Section 2.1).
  - Intuition: GRPO rewards what works within each prompt, avoiding a heavy critic; asymmetrically wider upward clips let the model strengthen initially low‑probability but promising reasoning steps.

- Reward shaping with strict format and verifiability (Section 2.2)
  - Format gate (Section 2.2.1):
    - Responses must contain one `<think> ... </think>` block; math answers must include a final `\boxed{...}`; code answers must include a fenced block with a language tag.
    - If format fails, reward = 0; if passes, give +0.1 and proceed to grading.
  - Correctness (Section 2.2.2):
    - Math: extract the last `\boxed{...}` and verify via rule‑based parsers and SymPy. If correct, +0.9 (so 1.0 total with formatting).
    - Code: extract the first code block; if C++, compile with a 10s timeout and run 20 randomly chosen tests (4s/test, 300MB memory). If all pass, +0.9 (again 1.0 with formatting).
  - Soft length penalty (Equation (1), Section 2.2.3):
    - Uses `l_max` and `l_cache`: no penalty until `l_max − l_cache`, then linearly down to −0.1 by `l_max`. Purpose: warn the model that the hard cutoff is near without being myopic.
  - Language consistency (Section 2.2.4):
    - 10% of English problems are translated into French, Spanish, Italian, German, Chinese, and Russian.
    - A fastText classifier checks the problem, thoughts, and answer (after removing LaTeX/code). If all three are in the same language, +0.1. This reduces code‑switching.
  - System prompt (Figure 2): enforces “think then answer,” both in the user’s language. Notably, “Be as casual and as long as you want” increases entropy and exploration during RL.

- Asynchronous online RL infrastructure (Section 3; Figure 3)
  - Three worker types: `Generators` (rollouts), `Verifiers` (rewarding), `Trainers` (updates).
  - Generators run continuously and never wait for trainers. Completed generations are verified and streamed to trainers. Trainers update and broadcast new weights GPU‑to‑GPU via NCCL; generators swap in new weights even mid‑generation (KV caches are not recomputed) (Section 3).
  - Batching:
    - A “batch” is a fixed number of completions (not tokens); “minibatches” split batches by sequences (then further split into fixed‑token “microbatches”).
    - A greedy collation algorithm reduces padding by 19% (Section 3).
  - Off‑policy considerations:
    - Because generators receive weight updates asynchronously, some early tokens in a long completion may be slightly off‑policy, but later tokens are on‑policy. Empirically, no KV recomputation is needed, and the GRPO objective handles this mild off‑policyness (Section 3).

- Data curation with verifiable tasks (Section 4)
  - Math (Section 4.1):
    - From ~699k raw items → 501k (format‑filtered) → 38k (difficulty‑filtered), see Table 1.
    - Two‑stage difficulty selection: (1) use `Mistral Large 2` to remove trivial/unsolvable problems; (2) train a small RL model and re‑grade the entire pool with 16 samples/problem, keeping problems in the “goldilocks” zone and rejecting likely mislabeled ground truths.
  - Code (Section 4.2):
    - Aggregate contest problems; keep those with adequate tests; run available solutions across tests; drop disagreeing tests, fix inconsistent ones by majority outcome, and synthesize tests when missing. Duplicate statements to require Python or C++ (total ~35k problems).

- Training regimes (Section 5)
  - Magistral Medium (pure RL; Section 5.2):
    - Start from `Mistral Medium 3 Instruct`. Staged curriculum ensures: (1) data difficulty keeps rising, (2) allowed completion lengths keep rising (`l_max − l_cache`: 16k → 24k → 32k), and (3) KV memory stays manageable by reducing concurrent requests and batch/minibatch sizes as lengths grow.
  - Magistral Small (SFT→RL; Section 5.3):
    - Create an SFT dataset from correct traces produced by Magistral Medium (plus diverse prompts from OpenThoughts and OpenR1 code subset; retain 10% general instruction data).
    - Fine‑tune `Mistral Small 3 Instruct` (24B) for 4 epochs; then RL with batch size 2048, `l_max − l_cache = 32k`, temperature 1.0, `ε_high = 0.3`.

- Notation that benefits the reader
  - `pass@1`: accuracy using a single sample. `maj@64`: majority vote accuracy over 64 samples (Table 2).
  - `KV cache`: cached key/value states used to speed autoregressive decoding.
  - `n_async`, `n_batch`, `n_minibatch`: concurrent generations, sequences per update, and sequences per optimization step, respectively (Section 6.3).

## 4. Key Insights and Innovations
- A. Pure RL can strongly improve a capable base model without any teacher traces
  - On `Mistral Medium 3`, RL alone boosts AIME’24 pass@1 from 26.8 to 73.6 and AIME’25 from 21.2 to 64.9; LiveCodeBench v5 from 29.1 to 59.4 (Table 2).
  - Significance: isolates the impact of RL and shows that verifiable rewards suffice to elicit large reasoning gains on a modern base checkpoint.

- B. A simple, asynchronous RL system keeps generators busy while staying sufficiently on‑policy
  - Continuous generation plus fast GPU weight broadcast (Figure 3) avoids the “slowest sequence bottleneck” typical in synchronized setups.
  - Practical effect: higher throughput without recomputing KV caches; off‑policyness is naturally limited because later tokens are generated with the latest policy (Section 3).

- C. Light‑touch but effective control over output format and language
  - Minimal “gate + reward” design (Section 2.2) enforces `<think>` structure and finalization (`\boxed{}` or codeblock), and adds a small language‑consistency bonus. This produces usable outputs and multilingual CoTs without task‑specific hacks (Section 2.2.4; Table 4).

- D. RL on text also improves or preserves non‑text capabilities
  - On vision‑reasoning benchmarks, the RL‑trained models do not regress and often improve (e.g., MMMU: +5% absolute, to 70.1%; MMMU‑Pro‑Vision: +12%, to 52.1%), despite no image supervision during RL (Figure 10; Section 7.2).

- E. Output length is a primary driver of reward in the learned policy
  - A PCA analysis of weight trajectories reveals a “length direction” in parameter space where both reward and average output length grow until hitting penalties/limits (Figure 8). Raw reward scales roughly logarithmically with output length (Figure 9). This clarifies why increasing `l_max` and encouraging exploration matter (Sections 5.2, 7.1).

## 5. Experimental Analysis
- Evaluation setup (Section 5.1)
  - Benchmarks: AIME’24/’25 (math), MATH‑500, LiveCodeBench v5/v6 and Aider Polyglot (code), GPQA (STEM), and text‑only subset of Humanity’s Last Exam.
  - Decoding: temperature 0.7 for math/GPQA; 0.95 for coding; max length 40k for AIME/LCB, 32k otherwise.
  - For AIME, report both `pass@1` and `maj@64` (average over many runs to reduce variance: 64 for AIME, 16 for LCB; Table 2 caption).

- Main results
  - Magistral Medium (RL‑only on `Mistral Medium 3`; Table 2)
    - Quote:
      > AIME’24 pass@1/maj@64: 26.8/43.4 → 73.6/90.0  
      > AIME’25 pass@1/maj@64: 21.2/30.0 → 64.9/83.3  
      > LiveCodeBench v5: 29.1 → 59.4; v6: 30.0 → 50.3  
      > GPQA: 59.6 → 70.8; MATH‑500: 91.0 → 94.3
    - Multilingual AIME’24: performance drops by 4.3–9.9 percentage points vs English (Table 4), showing successful multilingual reasoning but a mild cost when constrained to non‑English CoT.
    - Multimodal: clear gains on MMMU and MMMU‑Pro with only text RL (Figure 10).
  - Magistral Small (24B; three regimes, Table 3)
    - Quote:
      > SFT (on Medium traces): AIME’24 pass@1 = 65.4  
      > RL‑only (no SFT): 65.8  
      > SFT + RL (final Magistral Small): 70.7
    - Observation: For a smaller model, RL‑only is competitive with distillation and further improves when applied after SFT (Figure 5; Table 3).
  - Cross‑domain generalization (Table 5)
    - Training on only math significantly improved coding (LCB v5: 22.7 → 38.3) and training on only code improved math (AIME’24: 32.2 → 49.7). This suggests RL learns transferable reasoning strategies (Section 6.1).
  - Infrastructure ablations (Section 6.3; Figure 6)
    - Keeping `n_batch = n_minibatch` and avoiding too many minibatch updates per batch is important. Performance degrades when `n_minibatch` is reduced at fixed `n_batch`, especially with high `n_async/n_batch`.
  - Advantage normalization ablation (Section 6.4; Figure 7)
    - Little difference among minibatch, group, or no normalization for evaluation metrics and length growth. The paper uses minibatch normalization for simplicity.
  - “What didn’t work” tests (Section 7.4; Figures 11–12)
    - Partial (proportional) code rewards speed up training and discard less data but slightly reduce final LCB accuracy and slow length growth (Figure 11).
    - Entropy bonus (standard PPO trick) is unstable: on math data it lowers entropy; on mixed math+code it can cause entropy explosion. Tuning `ε_high` is a safer control for exploration/exploitation (Figure 12).

- Do the experiments support the claims?
  - The large, consistent boosts across math/coding/STEM benchmarks (Table 2) under a pure‑RL regime substantiate the central claim.
  - The multimodal “free lunch” (Figure 10) and tool/instruction parity (Table 6) support the assertion that text‑only RL is non‑destructive to other competencies.
  - Ablations credibly link stability to batching choices and exploration to clipping bounds, and they analyze counter‑attempts that underperformed.

## 6. Limitations and Trade-offs
- Verifiable‑reward scope
  - The pipeline relies on tasks with automatic checking (numeric answers, unit tests). Open‑ended reasoning without reliable verifiers is out of scope (Section 4).
- Sensitivity to length and memory
  - Gains are tied to longer chains‑of‑thought (Figures 8–9). This demands long contexts and large KV caches; the system reduces concurrency and batch sizes as lengths increase (Section 5.2). Training and inference costs rise accordingly.
- Asynchrony and off‑policyness
  - Mid‑generation weight swaps mean early tokens can be off‑policy. The approach works empirically (Section 3), but theory‑level guarantees are not given. Extreme `n_async/n_batch` ratios hurt stability (Section 6.3).
- No KL constraint
  - Eliminating KL regularization enables exploration but risks drifting from the base model’s style/safety if rewards are imperfect. The paper controls drift with format/language gates, but broader alignment aspects are not studied.
- Data and verifier biases
  - Math filtering uses model‑based difficulty estimates and consensus checks that may reject rare but correct approaches or lock in hidden biases (Section 4.1). Code tests are fixed or majority‑repaired; this can overfit to test distributions (Section 4.2).
- Multilingual trade‑off
  - Forcing CoT to match the user language yields small accuracy drops on translated AIME (Table 4). Real‑world non‑English performance may lag English, especially on niche languages beyond the six used in training.
- Mixed or task‑specific results
  - On some code benchmarks or settings, RL‑only can underperform SFT (Table 3: LCB v5/v6 for the 24B model). Also, entropy bonuses—common in RL—were hard to stabilize here (Section 7.4.2).

## 7. Implications and Future Directions
- How this changes the landscape
  - Demonstrates that a modern, capable base model can be upgraded into a high‑performing reasoning model with pure RLVR—no teacher traces—using a practical asynchronous system. This lowers the barrier for organizations to build reasoning models while retaining multimodal/tool skills (Tables 2, 6; Figure 10).
- Research avenues
  - Better verifiers and broader rewardable domains: move beyond numeric/code to symbolic proofs, program synthesis with spec checking, and tool‑augmented tasks.
  - Principled control of exploration without KL/entropy instability: theory for asymmetric clipping (`ε_high`) and off‑policy corrections in asynchronous pipelines (Sections 2.1, 3, 6.3, 7.4.2).
  - Length budgeting and efficiency: methods that reap the “length direction” benefits (Figures 8–9) with less compute, e.g., selective KV eviction, adaptive summarization, or learned “stop‑when‑confident” policies.
  - Data curricula: automated difficulty shaping with uncertainty/competence estimates that avoid discarding valuable hard items (Section 4.1).
  - Safety and multilingual robustness: couple verifiable rewards with safety constraints; extend the language‑consistency scheme to more languages and domains (Section 2.2.4; Table 4).
- Practical applications
  - High‑stakes math/coding assistants where verifiability is essential (education, competitive programming, formal methods).
  - Reasoning‑heavy multimodal tasks (scientific figures, charts) that benefit from improved text‑reasoning applied to images (Figure 10, Figures 14–16).
  - Tool‑integrated agents with function calling preserved post‑RL (Table 6).

In short, Magistral offers a replicable, verifiable‑reward RL recipe—algorithm, rewards, system, and data—that materially advances reasoning without sacrificing other skills. The open‑sourced 24B “Magistral Small” checkpoint (Figure 4; Section 5.3) further enables community follow‑up on curricula, infrastructure, and safety.
