# SCALING LAWS MEET MODEL ARCHITECTURE: TOWARD INFERENCE-EFFICIENT LLMS

**ArXiv:** [2510.18245](https://arxiv.org/abs/2510.18245)

## 🎯 Pitch

This paper introduces an architecture-aware scaling law that rigorously models how specific design choices—hidden size, MLP-to-attention ratio, and grouped-query attention—impact both the accuracy and inference efficiency of large language models. By augmenting classical scaling theory with these architectural knobs and proposing a practical framework for optimal model search, the authors enable the creation of LLMs that match or surpass standard baselines while delivering up to 42% higher inference throughput under fixed training budgets—directly addressing the growing real-world need for cost-effective model deployment at scale.

---

## 1. Executive Summary (2-3 sentences)
This paper introduces an architecture-aware “conditional scaling law” and a practical search procedure to design large language models (LLMs) that deliver high accuracy at much lower inference cost. By quantifying how three architectural knobs—hidden size, the MLP-to-attention parameter ratio, and grouped-query attention (GQA)—affect both training loss and runtime throughput, the method predicts and then validates model designs that match or beat standard baselines (e.g., LLaMA-3.2) while achieving up to 42% higher inference throughput under the same training budget (Figure 6).

## 2. Context and Motivation
- Problem addressed
  - Classical scaling laws tell us how to allocate parameters and training tokens to minimize pretraining loss (e.g., Chinchilla-style laws), but they ignore inference cost. As LLMs are deployed widely, inference (serving) dominates cost and energy, making accuracy-vs-inference trade-offs central (Section 1).
  - Prior attempts to include inference either assume knowing the lifetime number of generated tokens (impractical in deployments, as inference is repeated; discussion of Sardana et al., 2023 in Section 1) or only consider very limited architectural factors (e.g., “aspect ratio” = hidden size / number of layers) which do not capture runtime determinants like GQA or attention/MLP parameter split (Section 1; limitations of Bian et al., 2025).

- Importance
  - Real-world impact: Serving LLMs at scale is expensive in time, energy, and money. A model that is 40% faster at inference for the same accuracy can significantly cut deployment costs (Figure 6).
  - Theoretical significance: Understanding how architecture choices shape the loss surface and inference FLOPs grounds practical design in quantitative laws (Sections 3.2, 3.3, Appendix H).

- Where earlier approaches fall short
  - Compute-optimal training (Eq. (2), Section 2) solves for model size vs tokens but not for architectural choices that influence runtime throughput.
  - Prior architecture-aware studies emphasized depth/width trade-offs but did not model inference efficiency or the effect of GQA and parameter allocation between attention and MLP (Section 1).

- Positioning of this work
  - The paper fixes depth per scale (to avoid confounding the strong effect of depth on generalization noted by Petty et al., 2023) and studies three knobs that directly influence inference: hidden size (`dmodel`), the MLP-to-attention parameter ratio (`rmlp/attn`, abbreviated `r`), and GQA (Section 3.1). It then augments Chinchilla’s loss prediction (Eq. (1)) with a conditional calibration that captures how `dmodel` and `r` move the loss up or down relative to the Chinchilla optimum, and uses a small, local search for GQA (Sections 3.3–3.4).

## 3. Technical Approach
This section explains the methodology step-by-step, including design choices, equations, and the experimental pipeline.

- Architectural knobs (Section 3.1)
  - `hidden size (dmodel)`: the dimensionality of the model’s internal representations.
  - `MLP-to-attention ratio (r = rmlp/attn)`: how the non-embedding parameters are split between MLP layers and attention layers. Larger `r` means a bigger MLP relative to attention under a fixed parameter budget.
  - `GQA (grouped-query attention)`: a variant of attention where multiple query heads share fewer key/value (KV) heads. This reduces the KV cache size and some attention compute, improving throughput. The paper fixes the per-head dimension `dhead` (64 for ≤1B models, 128 for ≥3B; Section 3.1) and adjusts the number of heads `nhead` as `dmodel` and `r` vary.

- Why fix the number of layers?
  - Varying depth changes both accuracy and inference cost in ways that can dominate other effects; cutting layers degrades generalization after fine-tuning (Petty et al., 2023). The study instead holds `nlayers` fixed at each scale to isolate the effects of `dmodel`, `r`, and GQA (Section 3.1).

- How the paper measures inference efficiency (Sections 3.2, 4; Appendix E)
  - Throughput (tokens/second) is measured with vLLM on A100 40GB GPUs using 4096-token inputs and 1024-token outputs, averaged across 5 runs (Section 4).
  - Systematic ablations vary one knob at a time while holding others and total non-embedding parameters `Nnon-embed` fixed:
    - Hidden size sweep at fixed `r` and GQA (Figure 2 left; Figure 8).
    - MLP/attention ratio sweep at fixed `dmodel` and GQA (Figure 2 right; Figure 9).
    - GQA sweep at fixed `dmodel` and `r` (Appendix E, Figure 10; also Figure 13 for Qwen).

- Why these knobs affect throughput (Appendix H: Inference FLOPs Analysis)
  - The paper derives that total non-embedding inference FLOPs per generated token can be expressed as:
    - Total-FLOPs = 2·Pnon-emb + 2·nlayers·T·dq (Appendix H),
      where `Pnon-emb` is the non-embedding parameter count, `T` is the KV length (context length), and `dq = A·dh` is the total query dimension (A = number of query heads, dh = per-head dimension).
  - Implications:
    - Increasing `r` (more MLP, less attention) reduces attention dimensionality (i.e., smaller `dq`), thus lowering the 2·nlayers·T·dq term (Appendix H, bullet 1).
    - Increasing `dmodel` under a fixed budget often requires fewer attention heads (since `dhead` is fixed and parameters are constrained), reducing `dq` and hence FLOPs (Appendix H, bullet 2).
    - Increasing GQA reduces the number of KV heads, shrinking the KV cache and memory bandwidth costs, and improves throughput (Appendix E, Figure 10).

- How the paper models accuracy with architecture (Section 3.3)
  - Observation 1: Loss vs hidden size shows a U-shaped curve when plotting loss against normalized hidden size `dmodel/√Nnon-embed` at fixed `r` and GQA across scales (Figure 3). Too small or too large `dmodel` harms loss; there is an interior optimum.
  - Observation 2: Loss vs MLP-to-attention ratio also shows a U-shaped curve at fixed `dmodel` and GQA (Figure 4). Too little or too much attention degrades performance; there is an interior optimum of `r`.
  - Functional form for each U-shape: the paper fits `c0 + c1·log(x) + c2/x` separately for `x = dmodel/√Nnon-embed` and `x = r` to capture a convex-like U-shape with sublinear growth (Section 3.3).
  - Conditional scaling law (two-step calibration; Section 3.3):
    1) Use Chinchilla form (Eq. (1), Section 2) to define the optimal loss for given parameters and tokens: `Lopt(N, D) = min E + A/N^α + B/D^β`.
    2) Calibrate architectural variants relative to `Lopt` with either a multiplicative or additive correction:
       - Multiplicative (Eq. (3)):
         L(d/√N, r | N, D) = (a0 + a1·log(d/√N) + a2·√N/d) · (b0 + b1·log r + b2/r) · Lopt
       - Additive (Section 3.3): same terms added to `Lopt`. The paper finds multiplicative and additive perform similarly (Appendix G, Figure 15 right), while a joint, non-separable alternative performs worse (Appendix G, Figure 16).

- Searching for architectures that are fast and accurate (Section 3.4)
  - Formulate: maximize inference efficiency `IN(P)` (e.g., throughput) over architectures `P`, subject to a loss constraint `L(P|N,D) ≤ Lt` (Eq. (4)).
  - Procedure (Algorithm 1):
    - Fit or estimate `Lopt(N,D)` (via small-scale experiments and/or Chinchilla fitting).
    - Solve Eq. (4) over `dmodel` and `r` using the conditional law to find candidates meeting the loss target.
    - Perform a local search over GQA (enumerate feasible divisors of `nhead`) with early stopping, since loss vs GQA is noisy (Appendix F, Figure 14), but throughput monotonically improves with GQA (Appendix E, Figure 10).

- Experimental pipeline (Section 4)
  - Data: Dolma v1.7 corpus; samples from 15 sources with proportional sampling (Section 4).
  - Training: decoder-only LLaMA-3.2-style models; `Nnon-embed ∈ {80M, 145M, 297M, 1B, 3B}`; each model trained on `100 × Nnon-embed` tokens (e.g., 1B → 100B tokens) to ensure convergence (Section 4). Hyperparameters in Appendix D (Table 5).
  - Evaluation: zero-shot accuracy on 9 benchmarks via lm-evaluation-harness: ARC-Easy/Challenge, LAMBADA, HellaSwag, OpenBookQA, PIQA, SciQ, WinoGrande, CoQA (Section 4).
  - Fitting: Levenberg–Marquardt least-squares to fit the conditional law coefficients (Section 4).

## 4. Key Insights and Innovations
- Architecture-aware conditional scaling law (Section 3.3)
  - What’s new: A separable calibration that maps any (`dmodel`, `r`) choice to an expected pretraining loss relative to an `Lopt(N, D)` baseline (Eq. (3)), enabling predictions across scales without retraining every variant.
  - Why it matters: It provides a quantitative way to trade accuracy against throughput by picking `dmodel` and `r` systematically, rather than by ad-hoc grid search. Predictive quality is validated with low MSE and high rank correlation across scales (Figure 5).

- U-shaped loss behavior for both `dmodel/√N` and `r` (Figures 3 and 4)
  - What’s new: Clear, repeated observation that both hidden size (normalized by √N) and MLP/attention split have interior optima for loss across 80M–297M scales, not just monotonic trends.
  - Why it matters: It cautions against blindly shrinking attention or inflating MLP at scale; there is a sweet spot that preserves performance. This insight directly informs the conditional law’s functional form.

- Bridging architecture to inference FLOPs with an interpretable term (Appendix H)
  - What’s new: Decomposition shows the variable inference cost term is proportional to `T·dq`, where `dq` (query dimension) shrinks when you allocate fewer attention heads (via higher `dmodel` under budget or larger `r`).
  - Why it matters: Explains the empirical finding that larger hidden sizes and higher `r` improve throughput (Figure 2, Figure 8–9) and why GQA helps (Figure 10, 13), by tying changes to a specific FLOPs term and KV cache size.

- Practical search under a loss constraint with local GQA refinement (Section 3.4, Algorithm 1)
  - What’s new: A simple, hardware-aware procedure that uses the conditional law to shortlist (`dmodel`, `r`) candidates, then empirically maximizes throughput via small GQA sweeps.
  - Why it matters: Produces Pareto-superior models (“Surefire-*”) that beat standard baselines on throughput without sacrificing target loss (Table 1; Figure 6).

## 5. Experimental Analysis
- Evaluation setup (Section 4)
  - Training: >200 models spanning 80M–3B parameters, trained on 8B–100B tokens (Section 1; Section 4).
  - Throughput: vLLM on A100 40GB, 4096 input, 1024 output tokens, averages of 5 runs (Section 4).
  - Accuracy: Zero-shot on 9 benchmarks via lm-evaluation-harness (Section 4).

- Predictive accuracy of the conditional law (Figure 5)
  - Progressive scaling tests:
    - Task 1: Fit on 80M, evaluate on 145M → MSE ~ 0.0002, Spearman ~ 0.89.
    - Task 2: Fit on 80/145M, evaluate on 297M → MSE ~ 0.0001, Spearman ~ 0.79.
    - Task 3: Fit on 80/145/297M, evaluate on 1B → MSE ~ 0.0001, Spearman ~ 0.75.
  - Interpretation: Across jumps in model scale, the law predicts relative ranking and absolute loss closely, good enough to guide architecture choice.

- Main throughput and accuracy results at 1B and 3B (Table 1; Figure 6)
  - 1B scale:
    - “Panda-1B” (picked to minimize loss via the law) uses `dmodel=2560`, `fsize=4096`, `r≈1.07`, GQA=4; it achieves lower training loss (2.782 vs 2.803 for LLaMA-3.2-1B) and +2.1% average accuracy across 9 tasks (57.0 vs 54.9; Table 1; Section 5.1).
    - “Surefire-1B” (Pareto point under the LLaMA loss constraint) uses GQA=9 and `r≈3.6`, and consistently improves throughput over LLaMA-3.2-1B across batch sizes (Figure 6 center), while maintaining comparable accuracy (55.4 avg; Table 1).
  - 3B scale:
    - “Panda-3B” (loss-optimal) uses `dmodel=4096`, `fsize=4096`, `r≈1.0`, GQA=3; it achieves slightly lower loss (2.619 vs 2.625 for LLaMA-3.2-3B) and higher accuracy (62.5 vs 61.9; Table 1).
    - “Surefire-3B” (Pareto point) improves throughput across batch sizes vs LLaMA-3.2-3B (Figure 6 right) with similar accuracy (62.6 avg; Table 1).
  - Headline: Under identical training budgets, Pareto models achieve up to 42% greater throughput vs LLaMA-3.2 (Figure 6 caption and center/right panels).

- Generality across model families and scales (Appendix E)
  - The same throughput trends (larger `dmodel`, larger `r`, larger GQA → higher throughput) appear for LLaMA-style 1B/3B/8B and Qwen3 0.6B/1.7B/4B variants (Figures 8–13).

- Why GQA is searched locally, not modeled in loss
  - Loss vs GQA is highly variable and not monotonic (Appendix F, Figure 14), unlike the clean U-shapes for `dmodel` and `r`. By contrast, throughput vs GQA increases consistently (Appendix E, Figure 10). Hence, GQA is enumerated locally with early stopping (Section 3.4; Algorithm 1).

- Robustness and ablations (Appendix G)
  - Outliers: Including extreme `r` values (<0.5 or >5) harms fit quality (Task 3 Spearman drops from ~0.75 to ~0.33; Figure 15 center vs left).
  - Additive vs multiplicative: Similar fit quality (Figure 15 right), supporting the two-step “reference plus calibration” design.
  - Joint non-separable calibration degrades accuracy (MSE increases, Spearman drops to ~0.21; Figure 16), justifying the separable assumption.

- Fitting data matters at larger scales (Table 2; Figure 7)
  - Fitting the 3B conditional law using only 1B data produces better predictions for 3B (MSE ~ 0.0001, Spearman = 1.0; Figure 7 right) than fitting on pooled 80M–1B data (Figure 7 left). This indicates that the calibration coefficients can shift with scale.
  - The 3B configuration produced from 1B-only fit (“Panda-3B°”, `r≈1.23`) attains even lower loss (2.606) and the same average accuracy (62.5) as Panda-3B (Table 2).

- Do the experiments support the claims?
  - Yes, on three fronts:
    - Predictive: The conditional law consistently predicts loss across scales (Figure 5).
    - Mechanistic: FLOPs analysis explains throughput gains (Appendix H), and throughput trends are replicated across architectures and scales (Figures 2, 8–13).
    - Practical: The law-guided designs outperform LLaMA-3.2 baselines in both loss/accuracy and throughput (Table 1; Figure 6), with headline throughput gains up to 42%.

## 6. Limitations and Trade-offs
- Assumptions and scope
  - Depth is fixed per scale (Section 3.1). This isolates effects of `dmodel`, `r`, and GQA, but omits depth-related trade-offs (“aspect ratio”), which are known to affect generalization and efficiency.
  - The conditional law assumes separability between `dmodel/√N` and `r` in their effect on loss (Eq. (3)). While empirically strong, it may miss interactions; a joint model performed worse on these datasets (Appendix G, Figure 16), but future data could favor more complex forms.

- Data, training regime, and generalization
  - All models use Dolma v1.7 sampling and a single training recipe (Section 4; Appendix D). The authors note hyperparameters might deserve architecture-specific tuning (Section 7).
  - Models are trained on “100×N” tokens (e.g., 1B → 100B tokens), described as 5× Chinchilla-optimal to ensure convergence (Section 4). The conditional law’s coefficients may shift under different corpora, qualities, or token budgets.

- Inference measurements are hardware- and stack-dependent
  - Throughput is measured on A100 (40GB) with vLLM, 4k-in/1k-out (Section 4). Other accelerators, kernels (e.g., FlashAttention variants), batch schedulers, or KV-cache policies could change the throughput landscape and might alter the Pareto frontier.

- GQA treatment
  - GQA’s effect on loss is irregular (Appendix F), so it is searched locally rather than modeled. This keeps the method practical but means full loss predictability does not extend to GQA.

- Scale ceiling and model types
  - Experiments stop at 3B for dense models; 7B is not evaluated (Section 7). MoE architectures are not covered by the scaling law, though preliminary inference trends are reported (Appendix J).

## 7. Implications and Future Directions
- How this changes the landscape
  - Moves the community from “compute-optimal training only” towards “architecture-aware, inference-conscious scaling.” Teams with fixed training budgets can now pick faster architectures while meeting a target loss, rather than defaulting to off-the-shelf designs.

- Practical applications
  - Model distillation and deployment: pick `dmodel`, `r`, and GQA to hit latency/throughput SLAs without sacrificing accuracy, especially in high-volume inference services.
  - Capacity planning: use the law to forecast accuracy vs throughput for prospective architectures before training them.
  - Hardware co-design: the FLOPs decomposition (Appendix H) connects architecture to memory/computation bottlenecks, informing kernel and accelerator choices.

- Follow-up research enabled or suggested
  - Extend the conditional law to include depth and position it jointly with `dmodel` and `r`, potentially with a richer (but still stable) functional form.
  - Integrate data quality/curation variables and post-training steps (instruction tuning, RL) into the conditional framework to predict downstream performance more directly.
  - Derive an analytic, hardware-calibrated model of `IN(P)` (throughput/latency) to reduce the need for empirical GQA sweeps and make the search fully closed-form.
  - Generalize to Mixture-of-Experts: Appendix J shows analogous throughput trends for MoE (Figure 17), but a corresponding architecture-aware scaling law remains to be developed (Section 7).
  - Investigate test-time compute regimes (e.g., chain-of-thought, multiple samples): the method could be extended to co-optimize architecture with expected inference tokens per request, unifying with ideas like test-time compute scaling (cited in Section 1).

> Highlight result: “Under the same training budget, optimized architectures achieve up to 2.1% higher accuracy and 42% greater inference throughput compared to LLaMA-3.2” (Abstract; Table 1; Figure 6).

> Mechanistic anchor: “Total inference FLOPs = 2·Pnon-emb + 2·nlayers·T·dq,” so designs that reduce `dq` (fewer attention heads via larger `dmodel` or higher `r`) or shrink KV cache (higher GQA) are inherently more inference-efficient (Appendix H; Appendix E).
