# Beyond Reasoning Gains: Mitigating General Capabilities Forgetting in Large Reasoning Models

**ArXiv:** [2510.21978](https://arxiv.org/abs/2510.21978)
**Authors:** Hoang Phan, Xianjun Yang, Kevin Yao, Jingyu Zhang, Shengjie Bi, Xiaocheng Tang, Madian Khabsa, Lijuan Liu, Deren Lei
**Institutions:** (not specified in arXiv abstract)

## 1. Executive Summary (2-3 sentences)
Reasoning-oriented reinforcement learning often boosts math and logic benchmarks but quietly erodes a model’s broader skills (e.g., perception, OCR, robustness). This paper introduces `RECAP` (Replay-Enhanced CApability Preservation), a simple, end-to-end add-on for reinforcement learning with verifiable rewards (`RLVR`) that replays general-capability data and dynamically reweights training objectives based on short-horizon signals of progress and instability. In experiments on Qwen2.5-VL models (3B and 7B), `RECAP` preserves or improves general capabilities while matching or exceeding reasoning improvements (e.g., Table 1 and Table 2), addressing a pervasive form of post-training forgetting.

## 2. Context and Motivation
- The specific gap:
  - Contemporary reasoning post-training, especially `RLVR` (reinforcement learning with verifiable rewards—automatic reward signals like exact-match correctness or IoU), can degrade non-target skills such as perception, OCR, and robustness acquired during pretraining.
  - Evidence: Figure 1 compares several reasoning-tuned variants of Qwen2.5-VL models against their base versions on six non-reasoning benchmarks (A-OKVQA, AesBench, VStar, VisOnly, OCRBench, R-Bench-Dis). The reasoning-finetuned models “generally underperform” the base models on these non-reasoning tasks, while one model (MiMo-VL-7B-RL) remains close, presumably due to mixed-domain RL, though its full method is undisclosed (Introduction and Fig. 1).
  - Further evidence: Figure 2 shows that training only on math reasoning causes a 7% drop on a segmentation task (LISA IoU) after ~100 iterations, while `RECAP` not only prevents degradation but increases IoU by ~2% above the base model.

- Why this matters:
  - Real-world systems need both strong reasoning and preserved core abilities (perception, OCR, factual grounding, safety). Post-training that trades off these general competencies harms practical utility and reliability (Introduction, lines on hallucinations and jailbreak vulnerability; references to increased hallucinations and safety issues).
  - Theoretically, sustained post-training without explicit mechanisms can cause “catastrophic forgetting”—a classic phenomenon where adapting to a new objective degrades previously learned capabilities (Related Work; McCloskey & Cohen, 1989; French, 1999).

- Shortcomings of prior approaches:
  - `KL` regularization (penalizing deviation from a reference/base policy) stabilizes RLHF/RLVR but is computed on the current task data. It does not guarantee retention of off-task skills (Introduction; Background on RLHF/RLVR and KL).
  - Experience replay across heterogeneous domains is common, but deciding how to weight each objective (e.g., accuracy, format, IoU, next-token prediction) is nontrivial. Static or hand-tuned weights often misallocate focus because different rewards converge at different rates (Introduction; Section 4; Figure 4).
  - Some pipelines even reduce/remove the `KL` penalty to encourage exploration in reasoning RL (Related Work), exacerbating forgetting.

- Positioning:
  - The paper systematically documents capability regression in open-source reasoning VLMs (Fig. 1) and demonstrates this regression in controlled training (Fig. 2).
  - It proposes `RECAP`, which:
    - Replays general-capability data during reasoning RL (so the model keeps seeing what it would otherwise forget),
    - Dynamically reweights objectives using online, short-horizon statistics of convergence and instability (Section 4, Figure 3 and 4),
    - Drops into existing `RLVR` pipelines without training extra models or heavy tuning (Abstract; Section 4).

Reasoning behind this framing: The observed degradations (Fig. 1, Fig. 2) logically connect to catastrophic forgetting under narrow optimization. If different objectives converge/saturate at different times (Fig. 4), static mixtures are ill-suited. A scheduler that reallocates capacity from saturated signals to unstable/underperforming ones is a targeted fix.

## 3. Technical Approach
At a high level, the method augments an `RLVR` training loop with two components:
1) replay of general-capability data (perception, OCR, etc.), and
2) dynamic objective reweighting based on short-horizon convergence and instability signals.

Key concepts and terms (defined on first use):
- `RLVR` (Reinforcement Learning with Verifiable Rewards): RL training where rewards come from programmatic verifiers—e.g., exact-match correctness for math answers, format checks (ensuring outputs include required tags), or `IoU` (Intersection-over-Union) scores for bounding boxes. These rewards are objective, automatic, and bounded (Section 3).
- `GRPO` (Group Relative Policy Optimization): A PPO-style algorithm tailored for long-form reasoning without a learned value function. For each prompt, multiple rollouts are sampled, their rewards are group-normalized (subtract mean, divide by std), and policy gradients use a clipped importance ratio (Section 3; JGRPO definition).
- `IoU` (Intersection-over-Union): Standard metric for localization tasks, defined as area of overlap / area of union between predicted and ground-truth bounding boxes or masks (used as a verifiable reward in perception tasks).
- `SFT` (Supervised Fine-Tuning): Minimizing negative log-likelihood on instruction–response pairs (Section 3).
- `KL` (Kullback–Leibler divergence): A measure of how much the trained policy deviates from a reference/base policy (Section 3).

Step-by-step mechanism (Section 4; Figure 3):
- Inputs:
  - A set of heterogeneous objectives `k ∈ {1,…,K}` across domains (e.g., reasoning accuracy, formatting, `IoU`, next-token prediction). Some objectives come from `RLVR` (verifiable rewards via surrogates) and others from `SFT` (e.g., next-token prediction loss).
  - General-capability datasets (e.g., RefCOCO, OCR subsets) are mixed with reasoning datasets during training (Section 5.1; Table 3 in Appendix A.1 lists datasets and which objective they feed: accuracy, format, IoU, next-token prediction).

- Training loop:
  1) Replay general data alongside reasoning data. Data sampling is uniform across sources by default to isolate the effect of loss reweighting (Section 5.1).
  2) Collect per-objective loss/reward surrogates `L_k^(t)` at each iteration `t`. These may represent negative surrogate rewards (for RL) or standard losses (for SFT). The framework treats them uniformly as per-objective quantities to be weighted (Section 4, “Setting”).
  3) For each objective `k`, compute short-horizon statistics over a sliding window of length `2W` (Section 4):
     - Current-window mean: μ_k^(t) = average of `L_k` over the most recent `W` steps.
     - Previous-window mean: 𝛍̃_k^(t) = average of `L_k` over the preceding `W` steps.
     - Instability (coefficient-of-variation-like): σ_k^(t) = std of `L_k` in the current window.
  4) Form two signals (Section 4):
     - Convergence rate: c_k^(t) = μ_k^(t) / 𝛍̃_k^(t). Intuition: if c > 1, the objective is improving (loss decreasing); ≈ 1 indicates saturation.
     - Inverse signal-to-noise: i_k^(t) = σ_k^(t) / (μ_k^(t) + 𝛍̃_k^(t)). Intuition: higher values imply instability/high variance—more room for stabilization or learning.
  5) Combine signals and compute weights (Equation (1), Section 4):
     - Priority score: s_k^(t) = c_k^(t) + i_k^(t).
     - Reweighting coefficient: λ_k^(t) = K · softmax(s_k^(t)/T) across k, with temperature `T` (default T = 5). The prefactor ensures average weight remains 1 across objectives.
     - Interpretation:
       - Early in training, “easy” objectives (e.g., `format` rewards) quickly saturate: c → ~1, i → ~0. Their λ decrease automatically (Figure 4 shows format c and i dropping early).
       - Noisy/underperforming signals (e.g., `accuracy` on reasoning) have higher `i` and sometimes higher `c`, pulling more weight (Figure 4 right panels: accuracy keeps fluctuating).
  6) Optimize the weighted objective (Section 4):
     - Overall loss: L^(t)(θ) = (1/K) Σ_k λ_k^(t) L_k^(t).
     - For RL components, `GRPO` is used to compute policy gradients from verifiable rewards (Section 3).
     - For SFT components, standard cross-entropy is used (Section 3).

- Design choices and rationale:
  - Why replay? Forgetting arises because the model sees only the target domain; replay maintains exposure to general skills (motivated by Fig. 1, Fig. 2, and continual learning literature in Related Work).
  - Why dynamic reweighting instead of static weights? Different objectives have different learning curves (Figure 4). Static weights or fixed mixtures overemphasize already-solved objectives and underemphasize unstable ones. Online reweighting based on progress and instability lets the model refocus capacity as needs evolve (Section 4; Fig. 3–4).
  - Why the simple `s = c + i` (no extra learned model)? To remain end-to-end, magnitude-agnostic, and drop-in for existing RLVR pipelines without overhead or additional training (Abstract; Section 4). The authors note more complex combinations are possible but the simple sum works well empirically (end of Section 4).

- Practical training details (Section 5.1; Appendix A.1):
  - Base models: `Qwen2.5-VL-3B` and `Qwen2.5-VL-7B`.
  - Two setups:
    - RLVR-Only Setting (smaller scale; 3B): 8 GPUs, per-device batch size 2, 4 rollouts per prompt; train until domain data exhausted; evaluate on six reasoning-heavy benchmarks (SAT, ScienceQA, MathVista, ChartQA, InfoVQA, MMMU).
    - Hybrid Setting (larger scale; 7B): mix RLVR (reasoning/format/IoU rewards) and SFT (next-token prediction), 500 steps, 8 GPUs, per-device batch size 1, 2 gradient accumulation (effective batch size 16), 4 rollouts per prompt (64 rollouts per optimizer step); evaluate on a broad general + reasoning suite (LISA, MMMU-Pro, AI2D, MathVista, MathVision, MathVerse, MMBench, VizWiz, OCRBench v2).
  - Optimizer and system: AdamW, linear LR warmup/decay (max 1e-6), bfloat16 precision, FlashAttention kernels (Appendix A.1).
  - KL setting: Unless otherwise noted, `KL` toward the base policy is disabled to isolate replay+reweighting effects (Section 5.1).

Reasoning about how it works: The core difficulty is non-stationary, high-variance multi-objective RL (Figure 11). Directly trusting per-step losses would be noisy; short-horizon windowed statistics are a pragmatic compromise. By lowering weight on “solved” low-variance objectives (often formatting) and raising it on more volatile or lagging ones (often correctness), the model reallocates limited capacity where it is most needed.

## 4. Key Insights and Innovations
- Replay, but make it objective-aware (fundamental innovation):
  - Contribution: Integrate general-capability replay into `RLVR` and add an online, magnitude-agnostic reweighting scheduler that adapts to each objective’s learning dynamics (Section 4; Figure 3–4; Equation (1)).
  - Why it’s new: Prior practice often mixes data from multiple domains but relies on static sampling or hand-tuned weights. `RECAP` uses short-horizon progress and instability to automate this allocation without auxiliary models.
  - Why it matters: It mitigates forgetting (Fig. 2) and lets the training focus shift from saturated signals (formatting) to hard, high-variance signals (accuracy), improving both general capability retention and reasoning.

- Diagnosing objective dynamics during reasoning RL (insightful empirical finding):
  - Observation: Different reward types converge at different rates; formatting signals are easy and saturate quickly, while reasoning accuracy stays unstable longer (Figure 4).
  - Significance: Explains why static weighting misallocates effort and why training can appear to “optimize the appearance” of reasoning (format compliance) rather than correctness.

- Task-appropriate “thinking” rewards (incremental but important practical design):
  - Finding: Applying a chain-of-thought (“thinking”) format reward to perception tasks is counterproductive; models quickly reduce their rationales as unnecessary (Figure 6) and concise answers suffice.
  - Adjustment: In the hybrid setting, the authors reserve thinking rewards for tasks that benefit (math/logic) and use answer-only format for perception (Section 5.3). This avoids conflating format inflation with real reasoning.

- Conciseness without losing accuracy (new capability + practical gain):
  - Result: With replay + reweighting, the model’s reasoning traces become shorter and less variable (Figure 7), yielding 60% fewer words on average (67 → 27) without harming accuracy. This directly improves inference efficiency (fewer tokens) while preserving quality (Section 5.3; Appendix A.3 qualitative examples).

Reasoning about novelty: The combination of replay with principled, online objective reweighting tailored for RLVR’s multi-objective regime is the central innovation. The rest (e.g., using `GRPO`, removing `KL`, data choices) are methodological decisions that create a clean testbed for evaluating the scheduler’s value.

## 5. Experimental Analysis
- Evaluation methodology:
  - Datasets and setups (Section 5.1; Table 3 in Appendix A.1):
    - RLVR-Only (3B): SAT, ScienceQA, MathVista (mini), ChartQA, InfoVQA, MMMU.
    - Hybrid (7B): LISA (segmentation), MMMU-Pro (broad understanding), AI2D (diagram QA), MathVista, MathVision, MathVerse (multimodal math), MMBench (general multimodal comprehension), VizWiz (images from blind users), OCRBench v2 (OCR and reasoning).
  - Metrics:
    - Accuracy-like metrics for QA; `IoU` for localization/segmentation; others specific to each benchmark.
    - Internal training rewards: format, accuracy, `IoU`, next-token prediction accuracy (e.g., Figure 4, 5, 10).
  - Baselines (Section 5.1):
    - Reasoning-only (no replay), PropMix (sample by source size), Uniform (uniform sampling, no reweighting), Coreset (limited replay), LwF (uniform sampling + small `KL` β=0.01).
    - External open models for reference (not strictly comparable due to different pipelines).

- Main quantitative results:
  - RLVR-Only (3B; Table 1):
    - Compared to the base model, RL on mixed domains helps a lot (e.g., ScienceQA: 6.20 → 64.85 with Uniform).
    - `RECAP` beats Uniform and MoDoMoDo (a static mixture approach) across most benchmarks:
      - ScienceQA: Uniform 64.85, MoDoMoDo 65.74, RECAP 71.59 (+6.74 over Uniform).
      - SAT: Base 43.98 → RECAP 55.19.
      - InfoVQA: Base 32.02 → RECAP 60.78 (best among variants listed).
      - MMMU: Base 38.67 → RECAP 42.44 (best among 3B variants listed in the bottom block).
    - Quote (Table 1):
      > RECAP: SAT 55.19; ScienceQA 71.59; MathVista 33.2; ChartQA 70.40; InfoVQA 60.78; MMMU 42.44.

  - Hybrid (7B; Table 2):
    - `RECAP` achieves the strongest or runner-up scores across most of the nine datasets:
      - LISA (IoU-based segmentation): Base 65.13; Reasoning-only 57.58; RECAP 67.24 (best), showing both preservation and improvement over base (+2.11).
      - MMMU-Pro: Base 25.55; RECAP 34.15 (best among Qwen2.5-VL-7B variants).
      - AI2D: Base 67.62; RECAP 78.21 (second to Coreset 79.92 but higher than Uniform 76.43).
      - MathVision: Base 9.54; RECAP 25.11 (best).
      - MathVerse: Base 26.29; RECAP 40.83 (best).
      - MMBench: Base 71.82; RECAP 78.52 (best).
      - VizWiz: Base 50.82; RECAP 61.97 (second to Coreset 63.76).
      - OCRBench v2: Base 39.49; RECAP 39.72 (competitive; slightly above base and LwF; Uniform and PropMix lower).
    - Quote (Table 2):
      > RECAP (7B): LISA 67.24; MMMU-Pro 34.15; AI2D 78.21; MathVista 66.70; MathVision 25.11; MathVerse 40.83; MMBench 78.52; VizWiz 61.97; OCRBenchv2 39.72.

  - Forgetting vs preservation (Figure 2):
    - On LISA (out-of-domain segmentation), Reasoning-only training quickly drops below the base model (after ~100 iterations), while `RECAP` rises and ends ~2% above the base.

  - Why reweighting helps (Ablations; Section 5.3, Figure 5 & Figure 10):
    - Format reward: Uniform initially climbs faster (format is easy), but `RECAP` overtakes on accuracy as training proceeds (Figure 5). This demonstrates the scheduler’s intended behavior: down-weight saturated format signals and focus on correctness.
    - Final reward breakdowns (Appendix A.5, Figure 10):
      > “near-parity on thinking format and answer format, but consistent improvements on reasoning accuracy, IoU, and mean token accuracy (+2.01, +1.11, +1.40 points).”
    - Reward dynamics (Appendix A.6, Figure 11):
      - All rewards fluctuate significantly; total reward std peaks near 0.9 around step ~20, motivating window-based smoothing and dynamic reweighting.

  - Chain-of-thought calibration (Section 5.3; Figure 6–7; Appendix A.3):
    - Applying a uniform thinking reward across all domains leads perception tasks to truncate rationales (Figure 6).
    - With `RECAP`, the model produces much shorter but stable rationales for reasoning tasks (Figure 7: 67 → 27 words on average), improving inference efficiency without loss in accuracy.

- Do the experiments support the claims?
  - Yes, within the tested settings:
    - They measure forgetting explicitly (Fig. 1, Fig. 2) and show `RECAP` preserves or improves general abilities while sustaining reasoning gains (Tables 1–2).
    - The scheduler’s mechanism is tied to observable dynamics (Fig. 4, 5, 10, 11), not treated as a black box.
  - Caveats:
    - Only Qwen2.5-VL-3B/7B are tested; broader generality needs further validation.
    - Some external baselines are not fully comparable (different pipelines), but the paper focuses on controlled comparisons among finetunes from the same base.

- Failure cases and robustness checks:
  - The paper notes mixed outcomes on some datasets where a simpler Coreset baseline can be competitive (e.g., AI2D, VizWiz in Table 2). This suggests replay volume and domain match still matter.
  - They also disable `KL` by default; in some contexts, modest `KL` might help stabilize, but here `LwF` (with β=0.01) underperforms on reasoning vs Uniform/RECAP (Table 2), consistent with prior reports that KL can reduce plasticity for reasoning RL.

Reasoning on evidence strength: The combination of controlled ablations (Uniform vs RECAP with same data sampling), dynamic curves (Fig. 5), and broad evaluation (Table 2) makes a credible case for the scheduler’s benefit. The most compelling signal is that Reasoning-only harms LISA badly (Fig. 2, Table 2) and `RECAP` flips that to a gain.

## 6. Limitations and Trade-offs
- Assumptions and design choices:
  - Windowed statistics assume short-horizon means/variances are informative enough to guide weights. The choice of window size `W` and temperature `T` can influence behavior (Section 4; default T=5), but the paper does not sweep these hyperparameters extensively.
  - The simple combination `s = c + i` is heuristic. Although effective (Section 4), other combinations or adaptive balancing between “progress” and “instability” might do better in some regimes.

- Scenarios not fully addressed:
  - Safety, hallucination, and adversarial robustness: The introduction cites concerns that reasoning training can raise hallucinations and jailbreak vulnerability, but the main experiments do not include safety evaluations (Introduction).
  - Very long-horizon RL or larger models: Experiments run 500 steps (hybrid) with 4 rollouts/prompt on 8 GPUs; behavior under much longer runs, larger models, or different rollout counts remains to be tested (Section 5.1).
  - Cross-model generality: Results are on Qwen2.5-VL-3B/7B. The approach is model-agnostic, but empirical confirmation on other backbones (e.g., LLaVA, InternVL, Gemini-like) would strengthen claims.

- Computational and pipeline considerations:
  - RLVR is inherently compute-intensive. While `RECAP` itself adds minimal overhead (no extra model), replaying multiple domains and tracking windowed stats adds implementation complexity and slightly more bookkeeping.
  - Disabling `KL` by default improves plasticity but could increase drift risk in other settings. The scheduler complements but does not replace the stabilizing role of `KL` in all cases.

- Open questions:
  - The objective set matters: The paper shows that format signals can saturate early; if some domains lack clean verifiable rewards (common outside math/vision), how to define objectives that reflect “true” progress (Limitations section acknowledges extension beyond RLVR/SFT remains to be evaluated).
  - Dynamic reweighting vs. dynamic sampling: The paper reweights losses but keeps uniform sampling by default (Section 5.1). Jointly optimizing both could yield further gains but adds complexity.

Reasoning about impact of these limits: The method’s elegance (no extra models; simple signals) is also its constraint—some regimes may need more nuanced signals or task-specific detectors. Still, the paper’s strategy targets the observed pathology (format dominance, noisy accuracy) directly and pragmatically.

## 7. Implications and Future Directions
- How this changes the landscape:
  - Practical RLVR pipelines can adopt `RECAP` to prevent capability regression without building auxiliary weighting models or performing costly mixture searches. This lowers the risk of “reasoning-only” training that silently degrades general abilities (Fig. 1–2).
  - The work reframes multi-objective reasoning RL as a continual learning problem where dynamic focus is crucial. This perspective encourages reporting general capability metrics alongside reasoning scores.

- Follow-up research enabled/suggested:
  - Broader objective sets: Extend beyond RLVR to preference optimization (`DPO`, `IPO`, `ORPO`) and process rewards (`PRM`), as noted in Limitations (Limitations section). Verifying that `RECAP` remains magnitude-agnostic and stable in those regimes would be valuable.
  - Adaptive scheduler design: Explore alternative combinations of progress and instability (e.g., weighted sum, multiplicative, or risk-sensitive transforms), automatic temperature tuning, or window-length adaptation.
  - Joint sampling + reweighting: Move beyond uniform sampling; co-design samplers that anticipate which domains need exposure given the current λ and performance trajectories.
  - Safety and robustness: Combine with safety reward suites to test whether replay + reweighting mitigates safety alignment collapse reported for reasoning models.

- Practical applications:
  - Enterprise and product settings where models must reason (math, charts, logic) but also remain strong at OCR, perception, and general instruction following.
  - Systems with tight latency/compute budgets: `RECAP`’s tendency toward shorter rationales while retaining accuracy (Figure 7; Appendix A.3 qualitative examples) can reduce inference cost without sacrificing quality.

Reasoning on broader impact: The core idea—track which objectives are learning and which are unstable, then allocate attention accordingly—is general and likely to transfer. As RLVR and reasoning models proliferate, methods like `RECAP` can be the difference between “great on one leaderboard” and “consistently useful in the wild.”

----------------
References to the paper’s content used above:
- Capability regression in open models: Figure 1 (Introduction).
- Degradation under reasoning-only and preservation/improvement with `RECAP`: Figure 2 (Introduction).
- Method overview and signals: Section 4; Figure 3 (pipeline), Figure 4 (convergence/instability behavior), Equation (1) (weight computation).
- Background on SFT, RLHF, RLVR, GRPO: Section 3.
- RLVR-Only results: Table 1 (Section 5.2).
- Hybrid results: Table 2 (Section 5.2).
- Ablations on format vs accuracy and final reward mix: Section 5.3; Figure 5; Appendix A.5 (Figure 10).
- Thinking rewards calibration: Section 5.3; Figure 6 and Figure 7; Appendix A.3 qualitative examples.
- Reward dynamics and variance: Appendix A.6 (Figure 11).
- Datasets and implementation: Section 5.1; Appendix A.1 (Table 3).
