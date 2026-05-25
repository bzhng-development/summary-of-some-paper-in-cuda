# UFT: Unifying Fine-Tuning of SFT and RLHF/DPO/UNA through a Generalized Implicit Reward Function

**ArXiv:** [2410.21438](https://arxiv.org/abs/2410.21438)

## 🎯 Pitch

This paper introduces Unified Fine-Tuning (UFT), a novel single-stage framework that merges supervised fine-tuning (SFT) and alignment approaches (such as RLHF, DPO, and UNA) using a shared, generalized implicit reward function. By eliminating the traditional multi-stage pipeline that often leads to degraded model capabilities ('alignment tax'), UFT significantly enhances both instruction-following and factual accuracy in large language models—streamlining alignment and enabling LLMs to retain and extend their acquired skills in a more robust, efficient, and practical way.

---

## 1. Executive Summary
UFT is a single-stage fine-tuning framework that unifies supervised fine-tuning (SFT) and alignment methods (RLHF, DPO, UNA) by training a language model with one common objective built on a generalized implicit reward function. It matters because the common practice—doing SFT first and alignment second—often causes “alignment tax,” a drop in previously gained capabilities; UFT reduces this degradation and improves instruction-following and factuality across multiple benchmarks (see Tables 1–6).

## 2. Context and Motivation
- Problem/gap
  - Large language models (LLMs) are pretrained to predict the next token, which does not ensure good question answering or safe behavior. SFT improves task usefulness by teaching the model to answer prompts; alignment methods (e.g., RLHF, DPO, KTO, UNA) teach the model to prefer safe/helpful outputs.
  - In practice, SFT and alignment are applied sequentially. This staged pipeline often degrades performance on some tasks learned earlier (alignment tax/forgetting), as discussed at the start of Section 1 and shown empirically in Tables 1–3 for Mistral and Tables 4–6 for Qwen (the SFT+alignment models underperform SFT in several cases).

- Why it matters
  - Real-world deployments need both utility (follow instructions) and safety (avoid harmful content). A single-stage method that preserves skills while aligning behavior reduces engineering complexity and the risk of catastrophic regressions.

- Prior approaches and their limitations
  - SFT: maximizes likelihood of reference answers via cross-entropy (Eq. 1; Section 2.1). Useful for instruction-following but does not encode safety preferences.
  - RLHF: trains a reward model (Eq. 2) and then applies reinforcement learning with a KL penalty to a reference model (Eq. 3). This is memory-intensive and can be unstable (Section 2.2).
  - DPO: replaces online RL with direct optimization of an implicit reward, but relies on pairwise comparisons and an intractable normalization constant `Z(x)` that cancels only in differences (Eq. 4; Section 2.2).
  - UNA: introduces a generalized implicit reward mapping that removes `Z(x)` and supports pairwise, binary, and score feedback (Eq. 5–6; Section 2.2), but UNA by itself is framed as an alignment method, not yet unifying SFT.

- Positioning
  - UFT extends UNA to SFT: it treats high-quality instruction-response pairs as score-based alignment data with reward `r=1`, so SFT data and alignment data can be mixed and trained with one loss (Section 2.3). This aims to avoid alignment tax while keeping the benefits of KL regularization to a reference model embedded in the implicit reward (Eq. 5).

## 3. Technical Approach
UFT unifies SFT and alignment into a single training objective by expressing both as reward learning with a generalized implicit reward. The core mechanism is a mapping between a policy and an implicit reward anchored to a reference model.

Step-by-step:
1. Baselines and building blocks
   - SFT objective (Eq. 1; Section 2.1): maximize `πθ(y|x)` for ground-truth pairs `(x, y)` by minimizing cross-entropy. If `y = (y1,...,yN)`, this reduces to a tokenwise sum of negative log-likelihoods.
   - RLHF (Section 2.2): 
     - Train an explicit reward model `rϕ(x,y)` from pairwise preferences using a Bradley–Terry objective (Eq. 2).
     - Optimize the policy with RL to maximize expected reward while constraining deviation from a reference `πref` using KL divergence (Eq. 3), typically with PPO.
     - Limitations: memory (policy + value + reward models) and optimization instability.
   - DPO (Section 2.2): define an implicit reward tied to the policy and reference,
     - `rθ(x,y) = β log(πθ(y|x)/πref(y|x)) + β log Z(x)` (Eq. 4).
     - Because `Z(x)` is unknown, DPO cancels it by taking differences between preferred and dispreferred responses; hence DPO requires pairwise data and cannot use scalar scores directly.
   - UNA (Section 2.2): removes the dependency on the intractable `Z(x)` by defining a generalized implicit reward
     - `rθ(x,y) = β log(πθ(y|x)/πref(y|x))` (Eq. 5).
     - Then learn by minimizing a discrepancy `g( rϕ(x,y), rθ(x,y) )` between explicit reward (any label signal) and implicit reward (Eq. 6).
     - UNA thus works with pairwise, binary, or scalar scores, and can be used offline (prelabeled feedback) or online (on-the-fly scoring).

2. UFT: unifying SFT with alignment via UNA (Section 2.3)
   - Key idea: Treat SFT data `(x,y)` as alignment data with a maximal positive score. Concretely, assign `r=1` to each `(x,y)` pair because responses in instruction-tuning sets are high quality.
   - Merge instruction-tuning data (now with `r=1`) and alignment data (which may be pairwise, binary, or score-based) into one combined dataset.
   - Train the policy using the UNA loss (Eq. 6), minimizing `g(r, rθ(x,y))` across all examples, where `r` comes from either the transformed SFT data (`r=1`) or alignment data (`r` in [0,1]).
   - Choice of `g` (for SFT compatibility): The paper instantiates `g` as mean squared error (MSE) after applying a Sigmoid to `rθ` so that higher scores correspond to larger `rθ` (Eq. 7). Intuition:
     - With SFT-style `r=1`, minimizing `[σ(rθ(x,y)) - 1]^2` pushes `σ(rθ)` toward 1, i.e., `rθ → +∞`.
     - Because `rθ = β log(πθ/πref)`, driving `rθ` up increases `πθ(y|x)` relative to `πref(y|x)`, thus increasing likelihood of the desired response.
     - This yields the same end-goal as SFT—maximizing `πθ(y|x)`—but updates are tied to a reference policy via the log-ratio, providing a form of KL-style regularization (Eqs. 5 and 7).

3. Why this design
   - DPO is limited to pairwise preferences due to `Z(x)` (Eq. 4). UNA (Eq. 5–6) supports scalar scores and binary labels, letting SFT data fit naturally as “score=1”.
   - The implicit reward’s dependence on `πref` (Eq. 5) ties updates to a stable anchor (typically the pretrained model), mitigating catastrophic drift that can occur in pure SFT or miscalibrated RL.

4. Training modes
   - Offline UFT: use precollected instruction and alignment data; the setup is illustrated in Figure 2(C).
   - Online UFT: generate rewards on-the-fly with a reward model or LLM-as-judge (Section 2.2), then apply the same loss (Eq. 6).

5. Practical setup in experiments (Section 3)
   - Base models: `Mistral-7B-v0.1` and `Qwen 32B` with LoRA adapters (`r=16`).
   - Data:
     - Instruction data: UltraChat (20k samples by default; also 16k–260k in ablations, Section 3.3).
     - Alignment data: HelpSteer2 (20k samples).
   - Hyperparameters (Appendices A–B):
     - SFT best LR around `1e-4` (Tables 10–11).
     - UFT best LR around `3e-5` with `β=0.01` (Tables 12–13).

Analogy (Figure 1): Pretraining is “reading many books” (broad exposure without explicit feedback). Fine-tuning with UFT is “traveling many miles,” producing responses and immediately receiving evaluative feedback to refine behavior.

## 4. Key Insights and Innovations
- Unifying objective for SFT and alignment (fundamental)
  - Novelty: Recasts SFT pairs as score-based alignment (`r=1`) and optimizes both SFT and alignment data with the same implicit-reward objective (Eq. 6–7). This is different from prior pipelines that alternated between unrelated objectives (cross-entropy for SFT and preference optimization for alignment).
  - Significance: Removes the need for a two-stage SFT→alignment pipeline, thereby reducing alignment tax (see Tables 1–3 and 4–6 comparisons).

- Generalized implicit reward as the bridge (conceptual/theoretical)
  - Using UNA’s mapping `rθ = β log(πθ/πref)` (Eq. 5) converts policy learning into reward learning without `Z(x)`, enabling diverse feedback types (pairwise, binary, scalar).
  - For SFT-like data, the heuristic derivation (Eq. 7) shows the same likelihood-maximization target as cross-entropy, but with updates expressed as log-ratio to a reference. This gives a theoretical connection that SFT and UNA share the same maximization goal for `πθ(y|x)` on instruction-tuning data (Section 2.3).

- Reduced length bias and better instruction-following/factuality (empirical)
  - In Alpaca-eval, UFT’s output lengths remain near ~1300 tokens versus 5000–6000 for SFT+{DPO,KTO,UNA} on Mistral (Table 3), yet UFT attains much higher win rates (e.g., 8.28 vs 1.05/0.64/1.34). This suggests UFT avoids “longer-is-better” artifacts while improving judged quality.
  - On key tasks: ifeval (instruction following) and truthful (factuality) see large gains when training on mixed data with UFT (Tables 1–2 and 7–9).

- Data-mixing insight (practical)
  - Varying the instruction/alignment ratio notably changes outcomes (Section 3.3). Increasing instruction data improves instruction-following metrics but can reduce truthful scores; this provides actionable guidance for dataset composition (Tables 7–9).

## 5. Experimental Analysis
- Evaluation methodology
  - Benchmarks (Section 3):
    - New HuggingFace Open LLM Leaderboard: `bbh`, `gpqa`, `mmlu-pro`, `musr`, `ifeval`, `math-hard` (Table 1, Table 4, Table 7).
    - Old HuggingFace Open LLM Leaderboard: `gsm8k`, `truthful`, `winograde`, `arc`, `hellaswag`, `mmlu` (Table 2, Table 5, Table 8).
    - Free-form generation: MT-Bench and Alpaca-eval (length-controlled win-rate) (Tables 3 and 6, and ablation Table 9).
  - Setups:
    - SFT-only vs UFT-only on instruction data (UltraChat 20k).
    - Sequential SFT then alignment (DPO, KTO, UNA) vs UFT on mixed data (UltraChat 20k + HelpSteer2 20k).
    - Ablations on varying instruction data size with fixed alignment (16k–260k + 20k).

- Main quantitative results
  - UFT vs SFT on instruction-tuning data
    - Mistral (Tables 1–2, 3):
      - New leaderboard average: UFT 30.09 vs SFT 29.87.
      - Old leaderboard average: UFT 64.25 vs SFT 63.17.
      - MT-Bench: UFT 6.55 vs SFT 6.33.
      - Alpaca-eval LC WR: UFT 7.27 vs SFT 8.07 (SFT slightly higher here).
    - Qwen 32B (Tables 4–6):
      - New leaderboard average: UFT 48.71 vs SFT 48.88 (virtually tied).
      - Old leaderboard average: UFT 78.19 vs SFT 77.91.
      - MT-Bench: UFT 7.95 vs SFT 7.85; Alpaca-eval LC WR: UFT 9.96 vs SFT 8.34.
    - Takeaway: On average, UFT equals or exceeds SFT on most downstream tasks, with similar free-form generation quality (Section 3.1).

  - UFT on mixed data vs sequential SFT+alignment
    - Mistral (Tables 1–3):
      - New leaderboard average: UFT 32.81 vs SFT+DPO 29.09, SFT+KTO 28.85, SFT+UNA 29.16.
      - Old leaderboard average: UFT 64.34 vs SFT+DPO 62.84, SFT+KTO 63.23, SFT+UNA 63.02.
      - MT-Bench: UFT 6.78 vs 4.81/4.76/5.24; Alpaca-eval LC WR: UFT 8.28 vs 1.05/0.64/1.34.
      - Large task-specific boosts:
        - ifeval: baseline Mistral 23.22 → UFT 46.03 (Table 1).
        - truthful: baseline 42.58 → UFT 54.05 (Table 2).
      - Length bias: Sequential methods produce much longer outputs (4945–6215 tokens) than UFT (1317) while scoring worse in Alpaca-eval (Table 3).
    - Qwen 32B (Tables 4–6):
      - New leaderboard average: UFT 52.39 vs SFT+DPO 49.24, SFT+KTO 50.27, SFT+UNA 51.13.
      - Old leaderboard average: UFT 80.29 vs SFT+DPO 77.48, SFT+KTO 77.71, SFT+UNA 79.58.
      - Strong gains on `truthful`: 66.7 (UFT) vs 62.74 (SFT+UNA) and lower for others (Table 5).
      - Generation quality: MT-Bench 8.67 (UFT) vs 8.57–8.64; Alpaca-eval LC WR 13.79 (UFT) vs 9.83–13.75 (Table 6).
    - Takeaway: Combining SFT and alignment by UFT is consistently better than doing them sequentially, especially on instruction following and factuality (Section 3.2).

  - Data distribution ablations (Section 3.3; Tables 7–9)
    - Fix alignment data at 20k (HelpSteer2) and vary instruction data from 16k to 260k.
    - For Mistral:
      - ifeval rises from 23.22 (base) to ~46 with 20k–32k instruction data (Table 7).
      - truthful peaks near 56.69 with 16k and trends down toward ~50 as instruction share grows (Table 8).
      - Free-form quality (Alpaca-eval LC WR) stays high (6.85–9.92) with moderate lengths (~1320–1380), indicating stable generative behavior (Table 9).
    - Takeaway: More instruction data does not always help alignment-sensitive metrics (e.g., truthful). A balanced mix matters.

- Do the experiments support the claims?
  - Evidence of reduced alignment tax is strong: in both Mistral and Qwen, UFT on mixed data clearly outperforms SFT+{DPO,KTO,UNA} across most tasks and averages (Tables 1–6), while avoiding pathological length inflation (Table 3).
  - The claim that UFT can replace SFT is supported by parity or improvements on instruction-only training across many tasks (Tables 1–2, 4–6), with a heuristic objective equivalence shown in Eq. 7.
  - Robustness checks:
    - Hyperparameter sweeps for SFT and UFT are reported (Appendix Tables 10–13), lending credibility to chosen settings.
    - Data-mix ablations (Tables 7–9) explore sensitivity to instruction/alignment ratios.

- Caveats
  - Results are on two base models (7B and 32B), two datasets (UltraChat and HelpSteer2), and a specific LoRA setup; generality beyond these conditions is not yet established (Section 6).
  - Statistical significance and variance across runs are not reported.

## 6. Limitations and Trade-offs
- Assumptions about labels
  - Treating all instruction-tuning responses as `r=1` (maximally good) assumes uniformly high quality. In practice, instruction datasets are noisy; this could bias the implicit reward scale (Section 2.3 and Eq. 7).

- Reference model dependence
  - The implicit reward is a log-ratio to `πref` (Eq. 5). Choice of reference (often the pretrained model) affects update dynamics; very weak or very strong references could under- or over-regularize.

- Objective choice (`g`, Sigmoid+MSE)
  - While Eq. 6 admits any discrepancy function `g`, the paper instantiates `g` as MSE after Sigmoid for SFT conversion (Eq. 7). Other choices (e.g., logistic, Huber) might change stability or calibration; no ablation on `g` is provided.

- Data sensitivity
  - Performance depends on the proportion of instruction vs alignment data (Tables 7–9). For example, higher instruction share can reduce `truthful` (Table 8), implying real-world tuning is required.

- Scope and scale
  - Experiments are English-only and use academic datasets (Section 6). Multi-lingual or domain-specific settings may behave differently.
  - Only LoRA is explored; full-parameter training or other adapters might interact differently with the implicit reward objective.

- Computational considerations
  - UFT avoids RLHF’s heavy PPO loop and separate reward/value models, but still requires evaluating both `πθ` and `πref` to compute log-ratios (Eq. 5). This is lighter than RLHF yet heavier than plain SFT.

## 7. Implications and Future Directions
- Field impact
  - By demonstrating that SFT and alignment can be trained together under one objective (Eq. 6–7), UFT reframes post-training as a single-stage “learn-from-feedback” process. This simplifies pipelines and reduces capability regressions commonly seen after alignment.

- Practical applications
  - Building instruction-following assistants that also maintain safety and factuality without multi-stage tuning.
  - Continual post-training: add new alignment data or new instruction data and keep optimizing the same objective.
  - Deployment-time adaptation: online UFT with LLM-as-judge or reward models (Section 2.2) for rapid feedback incorporation.

- Research directions
  - Reward modeling: richer score schemas (calibrated 0–1 scales, multi-dimensional rewards for helpfulness, harmlessness, faithfulness) plugged into Eq. 6.
  - Objective design: explore alternative `g` functions and schedules for `β`, and token- or step-level variants compatible with Eq. 5.
  - Data curation: principled methods to balance instruction and alignment proportions per target KPI (e.g., optimize mixture for `truthful` vs `ifeval`).
  - Broader settings: multilingual/post-training in specialized domains; full-parameter vs adapter-based tuning; larger model scales; online learning stability.
  - Theoretical analysis: tighter guarantees for the heuristic equivalence to SFT (Eq. 7), calibration of implicit reward to explicit reward scales, and convergence properties under mixed feedback.

> Core takeaway: Figure 2 shows UFT’s single pipeline; Eqs. 5–7 formalize the unifying loss; Tables 1–6 demonstrate better average performance and reduced alignment tax; Tables 7–9 reveal how instruction/alignment ratios shape outcomes. UFT offers a practical and theoretically grounded path to combine utility and safety training in one stage.
