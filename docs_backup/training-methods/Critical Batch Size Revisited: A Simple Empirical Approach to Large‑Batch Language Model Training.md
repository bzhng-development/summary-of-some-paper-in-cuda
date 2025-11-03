# Critical Batch Size Revisited: A Simple Empirical Approach to Large‑Batch Language Model Training

**ArXiv:** [2505.23971](https://arxiv.org/abs/2505.23971)
**Authors:** William Merrill, Shane Arora, Dirk Groeneveld, Hannaneh Hajishirzi
**Institutions:** Not specified in abstract

## 🎯 Pitch

This paper presents a direct and assumption-free method to measure the critical batch size (CBS) for training large language models, enabling the use of larger batches without degrading model performance. By implementing a 'batch size warmup' strategy informed by these measurements, the approach reduces gradient steps by 43% while maintaining or improving loss, offering a significant enhancement in computational efficiency and throughput in large-scale language model training.

---

## 1. Executive Summary
This paper introduces a direct, low-cost way to measure the “critical batch size” (CBS) for training large language models and shows how to use that measurement to safely train with much larger batches via “batch size warmup.” The method avoids the strong assumptions behind the popular gradient-noise proxy and, in experiments on open OLMo models, enables comparable or slightly better loss with 43% fewer gradient steps (Table 1; Section 4.3).

## 2. Context and Motivation
- Problem addressed
  - When training at scale, increasing batch size improves hardware throughput but can hurt “token efficiency” if the batch is too large: the optimizer’s updates become less informative per token, degrading the loss achieved for a fixed token budget.
  - Practitioners therefore need a reliable way to pick (and adapt) batch size: large enough for speedup, but not so large that loss degrades. The target concept is the critical batch size `B*`: the largest batch for which loss is not meaningfully worse than with smaller batches at equal token budget (Introduction; “Critical Batch Size Hypothesis”).

- Why it matters
  - Production LLM training is bottlenecked by throughput and efficiency. A principled method to push batch size safely increases data parallelism without sacrificing final model quality.
  - It also clarifies how batch size should evolve over training rather than treating it as a single static number.

- Prior approaches and their limitations
  - The dominant practical proxy is the gradient noise scale (GNS) from McCandlish et al. (2018), which suggests setting `B*` to `Bsimple = tr(Σ) / ||G||^2` (Section 2). Two strong assumptions underlie this:
    - Optimizer is SGD and learning rate should scale linearly with batch (Equation (2); “Assumption 1”).
    - The Hessian is well-conditioned (proportional to identity), allowing `Bnoise = tr(Σ H) / (G^T H G)` to be approximated by `Bsimple` (“Assumption 2”).
  - In modern LLM pretraining, Adam is used, not SGD, and the loss landscape is not close to isotropic. The paper evaluates the GNS empirically and finds it underestimates CBS by orders of magnitude and often has a different trend (Figure 3 vs. Figure 2), making it unreliable in practice.

- Positioning of this work
  - Provides a direct, empirical way to measure CBS with minimal assumptions (Section 3.1).
  - Studies how local CBS evolves over time and across model sizes (OLMo 1B and 7B; Section 3.3).
  - Uses these measurements to propose and validate a “batch size warmup” schedule that increases batch only when the current training state can support it (Section 4).

## 3. Technical Approach
The paper centers on two linked ideas: (1) a direct CBS measurement procedure called “branched training” and (2) a training policy (“batch size warmup”) that uses those measurements to scale batch size safely.

A. Measuring CBS via branched training (Section 3.1)
- Key concept: “Local CBS”
  - Instead of a single global `B*` for the entire run, measure the maximum safe batch size at a specific training checkpoint. This captures that larger batches may be safe later, once the optimizer state and parameters have evolved.
- Procedure at a checkpoint
  1. Start from a saved checkpoint trained with batch size `B` and learning-rate schedule `η(t)`.
  2. Create several branches that resume training from that checkpoint but with batch-size multipliers `k ∈ {…}` (e.g., `k=1,2,3,…`), so the new batch size is `k·B`.
  3. Adjust the base learning rate using an optimizer-specific scaling rule `f(k)`:
     - For SGD: `f(k) = k` (linear scaling).
     - For Adam: `f(k) = sqrt(k)` (square-root scaling justified by SDE analysis, Malladi et al., 2022).
     - The branch uses base LR `f(k)·η` on top of the existing schedule form (e.g., cosine).
  4. Train each branch for a small fixed token budget `Δ` to allow the optimizer state to adapt to the new batch and LR. Implementation choices:
     - `Δ = 2B` tokens (small compared to the full pretraining budget; Implementation Detail 1).
     - Smooth per-batch losses using an exponential moving average with α=0.5 (Implementation Detail 3).
  5. Record the smoothed end-of-window loss `L_k` for each branch. Define a tolerance `ε = 0.01` to treat very similar losses as equivalent (Implementation Detail 2).
  6. Let `k*` be the largest multiplier such that “no worse than smaller batches,” i.e., for all `k < k*`, `L_{k*} ≤ L_k + ε`.
  7. The local CBS and base LR at this checkpoint are then:
     - `B* = k* · B`
     - `η* = f(k*) · η`
- Assumption used: Local Recovery (Assumption 3). If a larger-batch branch’s loss fully “recovers” to match smaller-batch branches within `Δ` tokens, the future trajectories will remain similar beyond `Δ`.
- Why this works
  - Think of `B*` as the local “speed limit” on batch size. If a branch with bigger batch can catch up in loss quickly after the change (accounting for the optimizer state’s transient), it is deemed safe at that point in training.
- Visual cue (Figure 1)
  - Plots loss vs. batch size at specific checkpoints for OLMo 1B. The dotted red line marks the detected CBS; loss curves rise sharply beyond this point.

B. Comparing against gradient noise scale (Section 2; Figure 3)
- For context, the paper also computes `Bsimple = tr(Σ) / ||G||^2` using the estimator from McCandlish et al. (2018) with `Bbig=64` and `Bsmall=1`, averaging over 4096 batches and reporting 95% confidence intervals (Appendix B).
- This is used strictly as a diagnostic comparison to the direct CBS measured above, not as a training signal.

C. Using CBS to drive “batch size warmup” (Section 4.1)
- Idea
  - Start training with a small batch, then double the batch size only when the measured local CBS exceeds twice the current batch. Always scale the base LR using the Adam square-root rule when batch increases.
- Policy
  1. Initialize with `B_0` and base LR `η_0`.
  2. As training progresses, if at time `t` the local CBS `B*_t > 2·B_t`, then update
     - `B_{t+1} = 2·B_t`
     - `η_{t+1} = sqrt(2) · η_t`
- In practice for OLMo 1B (Section 4.2):
  - Start with batch 1024, then double when measured CBS crosses 2048 (≈168B tokens) and again when it crosses 4096 (≈503B tokens). Figure 4 (left) shows the resulting batch and LR schedules.

D. Experimental setup for measuring CBS trends (Section 3.2)
- Models and data: Open OLMo 1B and 7B models with Dolma pretraining data; sequence length 4096 tokens per “document.”
- Measurement grid:
  - OLMo 1B: checkpoints from initialization up to ≈943B tokens with multiplier grids detailed in Appendix A (Figure 5 shows all loss-vs-batch-size plots).
  - OLMo 7B: checkpoints up to ≈2000B tokens with multiplier grids in Appendix A (Figure 6 shows all plots).

## 4. Key Insights and Innovations
- Direct, assumption-light CBS measurement
  - Novelty: Measures CBS by observing actual loss recovery under larger batches, rather than inferring from gradient statistics. This removes reliance on SGD-only and well-conditioned Hessian assumptions (Section 2 vs. Section 3.1).
  - Significance: Provides a practical, trustworthy tool for practitioners to decide how big they can push batch size at a given point in training.

- The evolution of CBS over training is rapid-then-flat and roughly size-invariant
  - Finding: For both OLMo 1B and 7B, local CBS starts near zero at initialization, rises quickly in the first ~50B tokens, and then plateaus around 4096 documents (Figure 2).
  - Significance: Suggests (1) the early phase is the risky period for large batches and (2) small-scale pilot runs may be informative for larger models, simplifying planning.

- Gradient noise scale is not a reliable proxy for CBS in LLM pretraining
  - Observation: The GNS estimate underestimates CBS by orders of magnitude and, for 7B in particular, does not match the qualitative trend (Figure 3 vs. Figure 2).
  - Significance: Cautions against relying on GNS-driven batch decisions in Adam-based LLM training.

- Batch size warmup validated end-to-end
  - Contribution: A concrete schedule—doubling batch when measured CBS doubles and scaling LR with `sqrt(2)`—achieves slightly better final loss with 43% fewer gradient steps than a fixed small-batch control (Figure 4; Table 1).
  - Significance: Demonstrates that CBS measurements can be operationalized to realize real training-time savings without sacrificing model quality.

## 5. Experimental Analysis
- Evaluation methodology
  - Settings
    - Models: OLMo 1B for warmup study; OLMo 1B and 7B for CBS/GNS trend studies.
    - Data: Open Dolma-based pretraining corpora; sequence length 4096.
    - Optimizer and schedule: Adam with cosine schedule; LR scaled by `sqrt(k)` when batch is multiplied by `k`.
  - CBS/GNS measurement
    - Branched training windows of `Δ = 2B` tokens; smoothed loss with α=0.5; tolerance `ε=0.01` (Section 3.1).
    - GNS via the two-batch-size estimator (Appendix B), `Bbig=64`, `Bsmall=1`, 4096 batches, 95% CIs.
  - Downstream loss proxies
    - After pretraining to 608B tokens, apply “mid-training” LR annealing to zero for 50B tokens (Figure 4 left), then report training losses and out-of-distribution cross-entropy on C4 and The Pile, plus bits-per-byte (BPB) on QA/code/math benchmarks (Table 2; Appendix E lists tasks).

- Main findings
  1. CBS vs. training progress and model size (Figure 2)
     - “CBS over OLMo 1B pretraining”: starts near 0, increases steeply, then plateaus around 4096 documents.
     - “CBS over OLMo 7B pretraining”: same qualitative shape; indicates weak dependence on model size within the explored range.
  2. CBS vs. gradient noise scale (Figure 3)
     - For 1B: GNS trend has some resemblance but is much smaller in magnitude (y-axis differs by orders).
     - For 7B: GNS trend diverges visibly from CBS trend, further weakening confidence in GNS as a proxy.
  3. End-to-end training outcomes with warmup (Figure 4; Table 1)
     - Three schedules evaluated:
       - “Small-Batch Control”: batch=1024, base LR `sqrt(2)·0.0004`.
       - “Large-Batch Control”: batch=4096 from the start, base LR `2·sqrt(2)·0.0004`.
       - “Batch Size Warmup”: 1024 → 2048 at 168B tokens → 4096 at 503B tokens; LR scaled by `sqrt(2)` on each doubling.
     - Quantitative results (Table 1; averages over last 10B tokens):
       > Pretraining loss (PT): Warmup 2.5891 vs. Small 2.6057 vs. Large 2.5962  
       > Mid-training loss (MT): Warmup 2.5433 vs. Small 2.5486 vs. Large 2.5506  
       > Gradient steps saved: Warmup 43% vs. Large 75% vs. Small 0%
     - Interpretation:
       - Warmup matches or slightly improves loss relative to small-batch control while saving 43% of gradient steps.
       - The large-batch control, although using 75% fewer steps, ends with worse loss than warmup (and slightly worse than small-batch) after mid-training.
  4. Out-of-distribution proxies (Table 2)
     - After pretraining (PT) and after mid-training (MT), warmup is broadly competitive or slightly better:
       > Tasks BPB (lower is better): PT 1.0316, MT 1.0076 for warmup vs. PT 1.0112, MT 0.9999 for small-batch; PT 1.0571, MT 1.01927 for large-batch.  
       > C4 loss: warmup PT 2.8049, MT 2.7597 vs. small-batch PT 2.8196, MT 2.7622.  
       > Pile loss: warmup PT 2.1916, MT 2.1521 vs. small-batch PT 2.2073, MT 2.1471.
     - Takeaway: No evidence that warmup hurts downstream-like losses; differences are small, as expected for modest schedule changes.

- Do the experiments support the claims?
  - Yes, for the stated scope:
    - CBS measurement behaves sensibly across time and sizes (Figure 2) and exposes a clear threshold effect in loss-vs-batch (Figure 1).
    - GNS’s mismatch with CBS is empirically demonstrated (Figure 3).
    - Warmup achieves the intended compute savings with comparable or slightly improved loss (Figure 4; Table 1–2).
  - Caveats:
    - Improvements are modest (e.g., MT loss gap vs. small-batch is 0.0053), but the savings in gradient steps are substantial.
    - Results are shown on OLMo configurations and specific token budgets; broader generalization is plausible but not proven here.

- Ablations, robustness, and failure modes
  - Implementation choices for CBS detection—window `Δ=2B`, tolerance `ε=0.01`, EMA α=0.5—are fixed heuristics (Section 3.1).
  - The paper reports all checkpoint loss–vs–batch curves (Appendix A; Figures 5–6), which helps assess visual robustness of CBS selection across many points.
  - Thresholds for when to double batch in warmup were set manually from measured CBS curves (Section 4.1 “Implementation Details”), not learned online, leaving some room for automation.

## 6. Limitations and Trade-offs
- Reliance on the Local Recovery assumption
  - The method assumes that if a larger batch “catches up” in loss within the short window `Δ`, it will remain fine later (Section 3.1). While reasonable and supported by experiments, this is not guaranteed in adversarial settings or near sharp transitions.

- Heuristic hyperparameters in CBS detection
  - `Δ=2B`, `ε=0.01`, LR smoothing α=0.5 (Section 3.1). Different values could change `B*` slightly, especially early in training; statistical testing to set `ε` is suggested for future work.

- Additional compute overhead for measurement
  - Branched training adds extra tokens per checkpoint and per multiplier. The paper argues this is small compared to launching many full runs (Section 3), but the overhead scales with how finely one probes batch multipliers and how often one checks.

- Manual decision points in warmup schedule
  - In the demonstration, the two batch doublings are triggered using offline CBS measurements and manual thresholds (Section 4.1). A fully online, automated controller is not implemented here.

- Generalization beyond OLMo and setup specifics
  - The results are on OLMo 1B and 7B with Adam, 4096-token sequences, cosine schedule, and Dolma-based data. The qualitative findings likely transfer, but are not yet validated across very different architectures, tokenization regimes, or optimizer choices.

- Fixed-step LR scaling choice
  - Warmup uses the Adam square-root scaling (`η ∝ sqrt(B)`). Alternatives (e.g., additional momentum or schedule tweaks) were not explored; performance might vary with optimizer tuning.

## 7. Implications and Future Directions
- Practical impact
  - Provides a straightforward, reproducible recipe to push batch size safely during LLM pretraining. Teams can:
    - Run a few short branched measurements early in training to map out local CBS.
    - Start small and double batch only when local CBS supports it, scaling LR by `sqrt(2)` each time.
    - Expect significant reduction in gradient steps without loss degradation, improving throughput and cost efficiency.

- Methodological shift
  - Moves the community from proxy-based CBS estimation (GNS) to direct, behavior-based measurement that aligns with actual optimization dynamics under Adam.

- Planning and forecasting
  - Because the CBS evolution looks similar for 1B and 7B (Figure 2), small pilot runs may suffice to plan safe batch escalations for larger models, de-risking large-scale training schedules.

- Toward automated controllers
  - Future work can:
    - Measure CBS online (e.g., intermittent, lightweight probes) and automatically trigger batch increases using statistical tests rather than fixed `ε`.
    - Integrate CBS-aware warmup with other schedule elements (e.g., LR warmup/decay, weight decay scaling) or with adaptive gradient clipping.

- Theory and scaling laws
  - The paper’s Appendix D shows how a global “fixed best batch size” emerges as the average of the local CBS over training (Proposition 1). If the local CBS grows like `t^0.5`, the aggregate best fixed batch scales like `T^0.5` (Proposition 2), aligning with prior empirical scaling laws. This linkage invites deeper theoretical work to predict CBS trajectories from model/data properties.

- Broader applications
  - Any large-scale training where Adam is used and batch scheduling matters—e.g., multimodal pretraining, instruction tuning with long sequences, or mixture-of-experts training—can adopt CBS measurement and warmup to raise safe throughput.
  - The methodology can also benchmark optimizer variants: different optimizers could be compared by how quickly CBS grows and where it plateaus.

> Bottom line: Treat the critical batch size as a measurable, time-varying property of your current training state. Measure it directly with short branches, then let it govern when you scale batch and learning rate. This replaces guesswork with an empirical “speed limit,” delivering throughput gains without sacrificing model quality.
