# Beyond Reasoning Gains: Mitigating General Capabilities Forgetting in Large Reasoning Models

**ArXiv:** [2510.21978](https://arxiv.org/abs/2510.21978)

## 🎯 Pitch

This paper addresses a critical limitation in reinforcement learning-based reasoning models: enhanced reasoning abilities often come at the cost of forgotten general skills like perception and robustness. The authors introduce RECAP, a lightweight replay and dynamic reweighting strategy that seamlessly mixes general knowledge data into training and adaptively shifts focus to underperforming objectives, thereby preserving—and even improving—broad capabilities while still achieving strong reasoning gains. This approach directly boosts real-world utility by ensuring advanced language and vision models retain the versatile skills needed for practical deployment and scientific robustness.

---

## 1. Executive Summary (2–3 sentences)
This paper tackles a common side-effect of reinforcement-learning-based “reasoning” finetuning: large vision–language models gain math/logic ability but forget general skills like perception, OCR, and robustness. It introduces RECAP, a simple replay-plus-dynamic-reweighting scheduler that mixes general-capability data into RL with verifiable rewards (RLVR) and adaptively shifts training focus away from saturated objectives (e.g., output formatting) toward underperforming or unstable ones (e.g., answer accuracy), preserving and sometimes improving general capabilities while also improving reasoning.

## 2. Context and Motivation
- Problem addressed
  - Contemporary reasoning finetuning (especially RLVR) improves benchmark reasoning scores but often causes “capability regression” on non-target skills the base model already had (perception, OCR, robustness, safety). Figure 1 contrasts base models vs reasoning-tuned variants across six non-reasoning benchmarks; most reasoning-tuned models underperform their bases on VisOnly, OCRBench, A-OKVQA, and R-Bench-Dis, among others.
  - In a controlled experiment, training Qwen2.5-VL-7B solely on math reasoning degrades LISA segmentation IoU by ~7 points within 100 RL steps, while RECAP not only prevents the drop but improves the base by ~2 points; see Figure 2 (blue vs red curves).

- Why it matters
  - Practical: Degradation in OCR or perception harms real applications (document understanding, accessibility, robotics) even if reasoning gets better.
  - Scientific: Reveals imbalance in current post-training where easy-to-optimize “format” signals dominate, masking true reasoning gains and eroding broad competence (Section 1 and Figure 4).

- Prior approaches and limitations
  - KL regularization to a base/reference model is standard in RLHF/RLVR to limit drift, but it is computed on the current task distribution and cannot guarantee retention of arbitrary non-target skills (Section 1).
  - Experience replay and multi-domain mixing exist, but fixed mixing/weights are hard to tune and may overemphasize objectives that converge quickly (e.g., format), starving harder, noisy objectives (e.g., answer accuracy) (Sections 1 and 4).
  - Recent multi-domain RL mixtures (e.g., MoDoMoDo, cited in Section 5.2) tune static ratios using proxy models and task-monitoring; this is compute-heavy and still relies on manual reward trade-offs.

- Positioning
  - The paper reframes reasoning finetuning as a continual/multi-objective learning problem and proposes RECAP: replay general-capability data and dynamically reweight all objectives online by short-horizon signals of progress (convergence) and instability (variance). It is plug-in, magnitude-agnostic, needs no extra models, and slots into standard RLVR (Section 4).

## 3. Technical Approach
RECAP = Replay-Enhanced CApability Preservation. It modifies training, not the model architecture.

- Background concepts (Section 3)
  - RLVR (Reinforcement Learning with Verifiable Rewards): replaces learned preference models with programmatic/verifiable signals (e.g., exact-match correctness, IoU for boxes, output format checks).
  - GRPO (Group Relative Policy Optimization): a PPO-like algorithm that normalizes rewards within groups of rollouts per prompt, avoiding an explicit value critic. For a prompt, sample G responses with a frozen rollout policy, compute sequence-level rewards, normalize to advantages, and optimize clipped importance-weighted likelihood with optional KL to a reference (Section 3, JGRPO formula).

- What RECAP changes (Section 4; Figure 3)
  1) Replay general-capability data during RLVR
     - Alongside the target reasoning domains (math, charts, geometry), sample non-reasoning/perception domains (object detection/segmentation, OCR). Each domain contributes its own verifiable or supervised objective:
       - Reasoning: `accuracy` (correct final answer), `format` (required tags like <think>, <answer>).
       - Perception: `IoU` (intersection-over-union for predicted regions), `ntp` (next-token prediction for OCR-style SFT).
     - This ensures the optimizer continues to “see” general tasks as it trains on reasoning (Figure 3, Step 1).

  2) Dynamically reweight objectives online
     - Problem: RL signals are non-stationary; some objectives converge quickly (format), others stay noisy (accuracy). Fixed weights cause over-optimization of easy signals and under-training of harder ones (Figure 4).
     - RECAP monitors each objective `k` using a sliding window of length 2W to estimate:
       - Current mean loss µ_k^(t) over the latest W steps and previous-window mean ˜µ_k^(t).
       - Instability σ_k^(t), the standard deviation in the current window (Section 4).
     - Two signals (Section 4):
       - Convergence rate: c_k^(t) = µ_k^(t) / ˜µ_k^(t). If < 1 the loss is improving; ≈1 means saturation.
       - Inverse signal-to-noise (instability): i_k^(t) = σ_k^(t) / (µ_k^(t) + ˜µ_k^(t)). Higher means more volatile/noisy.
     - Priority score and weights (Equation 1):
       - s_k^(t) = c_k^(t) + i_k^(t).
       - λ_k^(t) = K · softmax(s_k^(t) / T) so that average weight remains 1 across K objectives; temperature T (default 5) controls sharpness.
     - Final loss at step t:
       - L^(t)(θ) = (1/K) ∑_k λ_k^(t) L_k^(t), where each L_k^(t) can be an RL surrogate loss or a supervised loss.

  3) How it behaves (Figures 4–5)
     - Early in training, `format` improves quickly and stabilizes (c→1, i→0) so its weight decays; capacity shifts to `accuracy` and other high-variance objectives.
     - This prevents “format domination,” allowing the model to learn correctness rather than just template compliance.

- Design choices and rationale
  - Sliding-window estimates: per-step RL rewards are too noisy (Total-Reward STD peaks ~0.9 around step ~20; Figure 11). Windowed averages/variances are more reliable short-horizon signals.
  - Using c + i (rather than just one): Figure 4 shows `format` saturates fast (low i) while `accuracy` remains volatile; combining progress and instability differentiates which objectives deserve attention across training phases.
  - Magnitude-agnostic: ratios and softmax normalization avoid hand-normalizing heterogeneous losses/rewards.
  - No auxiliary models or test-time monitors: unlike static mixture search (e.g., MoDoMoDo), RECAP is end-to-end and adapts on the fly (Section 1 and 5.2 commentary).

- Where RL and SFT meet (Hybrid Setting, Section 5.1)
  - RECAP treats RLVR and SFT objectives uniformly as members of {L_k}. In the large-scale “Hybrid” setup, `ntp` (next-token prediction) for OCR-style SFT trains alongside RL surrogates for `accuracy`, `format`, and `IoU`.

## 4. Key Insights and Innovations
- Empirical diagnosis of general-capability forgetting in reasoning-tuned VLMs (Figure 1; Section 1)
  - Novelty: A systematic, side-by-side comparison across non-reasoning benchmarks shows consistent degradation after reasoning-only finetuning. This quantifies a widely suspected issue and motivates a continual-learning perspective.

- Dynamic objective reweighting driven by convergence and instability (Section 4; Equation 1; Figure 4)
  - Difference from prior work: Instead of fixed or hand-tuned weights (common in multi-task RL/RLVR), RECAP uses short-horizon statistics to adaptively rebalance emphasis. No per-objective gradient norms or additional critics are required—important at LLM scale.
  - Significance: Avoids over-optimizing easy “format” signals and reallocates capacity to correctness and perception, improving both reasoning accuracy and general skills (Figures 5 and 2).

- Replay of general-capability data inside RLVR pipelines (Figure 3; Sections 4–5)
  - Difference: Replay is common in continual learning, but integrating perception/OCR replay directly into on-policy RLVR for VLMs and coupling it with dynamic reweighting is new here.
  - Significance: Prevents drift away from pretraining abilities without relying solely on KL-to-reference penalties (which operate only on current-task distributions).

- Concise reasoning without accuracy loss (Figures 6–7)
  - Insight: Pushing “thinking rewards” across all domains is counterproductive for perception; the model naturally shortens rationales on such tasks. In the reasoning domain, RECAP reduces average chain-of-thought length by ~60% (67 → 27 words) while preserving accuracy (Figure 7), improving inference efficiency.

## 5. Experimental Analysis
- Setups (Section 5.1)
  - RLVR-Only (small): Qwen2.5-VL-3B trained on RLVR with multiple reasoning/vision datasets, evaluated on SAT, ScienceQA, MathVista (mini), ChartQA, InfoVQA, MMMU. 8 GPUs, per-device batch 2, 4 rollouts/prompt.
  - Hybrid RL+SFT (large): Qwen2.5-VL-7B trained 500 steps using ThinkLite-VL-70k (reasoning) plus perception replay (RefCOCO, LLaVA-OneVision OCR). 8 GPUs, effective batch 16, 4 rollouts/prompt. Evaluated on LISA, MMMU-Pro, AI2D, MathVista, MathVision, MathVerse, MMBench, VizWiz, OCRBench v2.
  - KL penalty is disabled by default to isolate replay vs regularization (Section 5.1). LwF baseline adds small KL (β=0.01).

- Baselines (Section 5.1)
  - Reasoning-only (no replay), PropMix (sample by dataset size), Uniform (equal sampling), Coreset (size-limited replay), LwF (uniform + KL), MoDoMoDo (strong static mixture baseline requiring proxy models and per-task performance monitoring).

- Main results
  - RLVR-Only (Table 1)
    - RECAP achieves the best or tied-best across six benchmarks, e.g.:
      > ScienceQA: Base 6.20 → RECAP 71.59 (vs MoDoMoDo 65.74; Uniform 64.85).  
      > SAT: RECAP 55.19 (best among listed).  
      > ChartQA: RECAP 70.40 (tied with MoDoMoDo 70.40).  
      > MMMU: RECAP 42.44 vs Uniform 39.44 vs Base 38.67.
    - Interpretation: Mixing diverse tasks in RLVR helps a lot (Uniform is already strong), but dynamic reweighting adds consistent gains.

  - Hybrid RL+SFT (Table 2)
    - RECAP is best or runner-up on most benchmarks and improves general capabilities over Base and Reasoning-only:
      > LISA (segmentation IoU): Base 65.13 → Reasoning-only 57.58 (forgetting) → RECAP 67.24 (best in block).  
      > MathVerse: Base 26.29 → RECAP 40.83 (best).  
      > MMBench: RECAP 78.52 (best).  
      > VizWiz: Base 50.82 → RECAP 61.97 (still below Coreset 63.76 but +11 points over Base).
    - AI2D: Coreset 79.92 is the best; RECAP is close at 78.21.
    - Conclusion: RECAP prevents perceptual forgetting and boosts reasoning metrics simultaneously.

- Evidence the method works as intended
  - Reward dynamics (Figure 4): `format` converges and stabilizes early (c≈1, i≈0 after ~50 steps), while `accuracy` stays unstable—justifying shifting weight away from format and toward accuracy.
  - Training curves (Figure 5): Uniform baseline climbs faster on format early but RECAP surpasses it on accuracy by ~step 40 and maintains the lead.
  - Forgetting vs preservation (Figure 2): On out-of-domain LISA, Reasoning-only plunges from ~65.1 IoU to ~57.6 by 500 steps; RECAP rises to ~67.2.
  - Token-efficiency (Figure 7): Reasoning chain length drops from ~67 to ~27 words with RECAP, reducing compute at inference without hurting correctness.

- Ablations and diagnostics
  - Uniform vs RECAP ablation (Section 5.3; Figure 5): same data sampling and pipeline—only the weighting differs. RECAP achieves higher accuracy despite slightly slower early format gains.
  - “Thinking reward everywhere” is suboptimal (Section 5.3; Figure 6): on segmentation data, rationales rapidly shrink toward zero—formatting for thinking is unnecessary and may distract.
  - Final metric parity vs improvements (Appendix A.5, Figure 10): formatting rewards end near parity with the baseline, while correctness-oriented metrics improve (e.g., Accuracy +2.01, IoU +1.11, mean token accuracy +1.40).

- Do the experiments support the claims?
  - Yes. Multiple datasets and two scales of setup confirm:
    - Forgetting is real under reasoning-only finetuning (Figure 1; Figure 2; Table 2 “Reasoning-only”).
    - Replay helps, and dynamic reweighting adds extra gains beyond Uniform, Coreset, and LwF (Tables 1–2).
    - The scheduler behaves as designed (Figures 4–5, 11).

## 6. Limitations and Trade-offs
- Assumptions and scope
  - Availability of verifiable rewards or supervised losses for general-capability data (IoU, ntp) is assumed; not all domains have clean verifiers.
  - The paper evaluates RLVR and SFT; it does not include large-scale preference-model or DPO-style objectives, though the method should extend (Limitations section).

- Design sensitivities
  - Hyperparameters W (window length) and T (softmax temperature) influence responsiveness vs stability; defaults (e.g., T=5) worked here but may require tuning across domains.
  - Decision to disable reference-KL by default isolates effects, but in some deployments KL may be necessary for safety/stability and could interact with RECAP’s weights (Section 5.1 notes; Table 2 shows LwF’s trade-offs).

- Not addressed scenarios
  - Long-horizon safety alignment, jailbreak robustness, and hallucination metrics are qualitatively discussed in motivation but not directly measured here.
  - The method adapts weights based on short-horizon loss statistics, not on external task metrics; this is a strength (no extra models) but may miss long-term interactions.

- Compute and data
  - Replay requires curating and loading heterogeneous datasets; training still runs on 8 GPUs with on-policy rollouts (Sections 5.1, Appendix A.1), which is moderate but non-trivial.

## 7. Implications and Future Directions
- How this changes the landscape
  - Reframing reasoning finetuning as a multi-objective continual-learning problem with on-policy RL is the key shift. The work shows that simple, principled scheduling can recover general capabilities without sacrificing (and even improving) reasoning accuracy, challenging the notion that capability regression is an unavoidable trade-off.

- Follow-up research enabled/suggested
  - Extend RECAP to preference-based alignment (e.g., ORPO/DPO/KTO) and process reward models; the magnitude-agnostic, windowed scheduler should transfer (Limitations section).
  - Joint scheduling for safety and factuality verifiers (e.g., hallucination detectors) to counter known regressions during reasoning RL.
  - Adaptive reward design per domain: e.g., decouple “thinking format” from perception tasks, as Figures 6–7 suggest.
  - Explore hierarchical or meta-learned schedulers that learn the combination of c and i, or incorporate additional signals (e.g., exploration diversity, Pass@k spread).

- Practical applications
  - Deployable training recipes for models that must both “see” and “think”: document VQA, assistive tech (VizWiz), scientific diagram understanding (AI2D, MathVista), and spatial or robustness-critical tasks (LISA, R-Bench-Dis).
  - Token-efficient reasoning at inference (Figure 7) reduces latency and cost without losing accuracy—valuable for production systems.

Overall, the paper contributes an easy-to-adopt training control knob—replay plus dynamic objective reweighting—that corrects a widespread brittleness in reasoning-focused post-training. The comprehensive experiments (Figures 1–7, 10–11; Tables 1–3) indicate the method’s practicality and robustness, with clear room to extend it to broader alignment objectives.
