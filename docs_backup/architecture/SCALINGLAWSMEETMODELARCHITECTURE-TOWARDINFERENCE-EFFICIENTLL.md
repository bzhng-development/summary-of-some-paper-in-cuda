# SCALING LAWS MEET MODEL ARCHITECTURE: TOWARD INFERENCE-EFFICIENT LLMS

**ArXiv:** [2510.18245](https://arxiv.org/abs/2510.18245)

## 🎯 Pitch

This paper introduces an architecture-aware conditional scaling law and a novel search framework that explicitly optimize large language model (LLM) architectures for both inference efficiency and accuracy. By extending classic scaling laws to account for key architectural factors—hidden size, MLP-to-attention ratio, and grouped-query attention—the method reliably predicts high-throughput, high-accuracy configurations, achieving up to 42% faster serving speeds without sacrificing performance. This breakthrough empowers practitioners to deploy LLMs that are not only smarter but also dramatically cheaper to operate at scale.

---

## 1. Executive Summary (2–3 sentences)
This paper introduces an architecture-aware scaling law and a practical search procedure that jointly optimize large language models (LLMs) for both accuracy and inference efficiency. By extending the Chinchilla scaling framework with architectural knobs—hidden size, the MLP-to-attention parameter ratio, and grouped-query attention (GQA)—the method predicts inference-friendly designs that, when trained, achieve up to 42% higher throughput at equal or better accuracy than strong open-source baselines (Abstract; Table 1; Figure 6).

## 2. Context and Motivation
- Problem addressed
  - Existing scaling laws reliably forecast pretraining loss vs. parameters and data, but they ignore inference costs, which dominate deployment expenses (Introduction §1; Related Work). The practical question is: how to make LLMs that are both accurate and cheap to serve?
- Importance
  - Real-world impact: inference is the dominant cost for widely deployed LLMs (Introduction §1; citations to Sardana et al., Park et al.). Optimizing accuracy alone can produce models that are too expensive to serve.
  - Theoretical significance: current laws (e.g., Chinchilla) do not expose how architecture choices (beyond size) influence the loss/efficiency frontier.
- Shortcomings of prior approaches
  - Including inference in scaling laws by summing training and deployment FLOPs forces one to estimate “lifetime generated tokens,” which is impractical (Introduction §1, critique of Sardana et al.).
  - Prior architecture-aware work mostly collapsed design into a single “aspect ratio” (hidden size vs. depth), which misses critical factors for inference cost and may degrade generalization when changing depth (Introduction §1).
- Positioning
  - This work fixes the number of layers and varies hidden size (`d_model`), the `mlp-to-attention ratio` (`r_mlp/attn`, the ratio of MLP parameters to attention parameters), and GQA (grouped-query attention, where multiple query heads share key/value projections). It builds a conditional extension of Chinchilla’s law that predicts loss as these architecture knobs change, and pairs it with a constrained search that maximizes throughput subject to an accuracy target (Sections 3.1–3.4).

## 3. Technical Approach
Step-by-step overview of what is built and how it works.

- What is being optimized
  - Goal: under fixed parameter count (`N_non-embed`) and training tokens (`D`), find an architecture that is fast at inference and meets an accuracy (loss) target (Eq. 4).
  - Architecture knobs considered: hidden size `d_model`, MLP intermediate size (captured via the ratio `r_mlp/attn`), and GQA. Depth (number of layers) is held fixed within each parameter scale to avoid confounding effects on accuracy and serving cost (Section 3.1).

- Why these knobs
  - In practice, modern open-weight dense models with similar parameter counts adopt very different `d_model`, `r_mlp/attn`, and GQA (Section 3.1), and these choices strongly influence both throughput and loss (Figures 2–4, 8–10, 14).

- How inference efficiency is analyzed and measured
  - Ablations at fixed `N_non-embed` show:
    - Increasing `d_model` improves throughput across batch sizes (Figure 2 left; Appendix E Figure 8 for 1B/3B/8B; Figure 11 for Qwen3 variants).
    - Increasing `r_mlp/attn` improves throughput (Figure 2 right; Appendix E Figure 9).
    - Increasing GQA improves throughput (Appendix E Figure 10; Figure 13 for Qwen3).
  - Mechanism (Appendix H): the per-token inference FLOPs are
    > Total-FLOPs = 2·P_non-emb + 2·n_layers·T·d_q  
    where `P_non-emb` is non-embedding parameter FLOPs, `T` is KV cache length, and `d_q = n_heads · d_head`. With fixed `N_non-embed` and `d_head`, increasing `d_model` typically reduces the number of heads `n_heads` (to keep the parameter budget and `r_mlp/attn`), which lowers `d_q` and therefore the costly attention term `2·n_layers·T·d_q`. Larger `d_model` and higher `r_mlp/attn` also reduce KV-cache size and I/O during decoding, further boosting throughput (Section 3.2; Appendix H).

- How accuracy (training loss) depends on architecture
  - Empirical finding: both hidden size and `r_mlp/attn` show U-shaped relationships with loss when other factors are held fixed.
    - Loss vs. normalized hidden size `d_model / √N_non-embed` is U-shaped with consistent minima across scales 80M, 145M, 297M (Figure 3).
    - Loss vs. `r_mlp/attn` at fixed `d_model` is also U-shaped (Figure 4).
    - Intuition: making `d_model` too large reduces the number of attention heads too much (hurting attention capacity), while making it too small under-allocates representation width; similarly for the ratio, over- or under-allocating parameters to attention relative to MLP harms modeling power (Section 3.3).
  - GQA’s effect on loss is noisy and not monotonic (Appendix F Figure 14), so the paper treats GQA differently in the search (below).

- The conditional architecture-aware scaling law
  - Start from the standard Chinchilla-style law that models loss as a function of parameters `N` and data `D`:
    > L(N, D) = E + A / N^α + B / D^β  (Eq. 1)
  - Define `L_opt(N,D)` as the best achievable loss at given `(N,D)` ignoring architecture (Eq. 1 or found via empirical sweep for sub-1B; Section 4, Fitting Scaling Laws).
  - Model architecture effects as separable calibrations relative to `L_opt`:
    - Multiplicative calibration (used primarily):
      > L(d/√N, r | N, D) = (a0 + a1·log(d/√N) + a2·√N/d) · (b0 + b1·log r + b2/r) · L_opt  (Eq. 3)
    - Additive variant is also tested and performs similarly in this study (Section 3.3; Appendix G).
  - Rationale: the `log(·)` and reciprocal terms capture the U-shaped loss curves seen in Figures 3 and 4. The separability assumption is validated empirically; joint non-separable variants perform worse (Appendix G, Figure 16).

- Architecture search under accuracy constraints
  - Objective: maximize measured inference throughput `IN(P)` over candidate architectures `P`, subject to a loss constraint `L(P | N,D) ≤ L_t` (Eq. 4).
  - Procedure (Algorithm 1, Section 3.4):
    1) Fit (or obtain) `L_opt(N,D)` and the calibration parameters (`a_i`, `b_i`) on smaller models.
    2) Solve Eq. (4) over `d_model` and `r_mlp/attn` using the fitted law.
    3) Locally search feasible GQA values (divisors of the number of heads), with early stopping once accuracy drops below a GQA=4 baseline. This irregular GQA–loss relationship explains the local enumeration (Appendix F Figure 14).
  - Throughput is measured (rather than analytically computed) because it depends strongly on hardware and decoding settings (Section 5.1).

- Experimental design essentials
  - Data: sampled from Dolma-v1.7 (15 sources) to preserve source distribution (Section 4, Training Setup).
  - Scales: >200 models from 80M to 3B parameters; tokens from 8B up to 100B; for most scaling-law fitting, models are trained on 100·N_non-embed tokens (~5× Chinchilla-optimal) to reach convergence (Section 4).
  - Inference: throughput measured with vLLM on A100 40GB with 4096 input and 1024 output tokens; average of 5 runs (Section 4).
  - Evaluation: zero-shot on 9 tasks using the lm-eval-harness (ARC-Easy/Challenge, LAMBADA, HellaSwag, OpenBookQA, PIQA, SciQ, WinoGrande, CoQA) (Section 4).

## 4. Key Insights and Innovations
- Conditional, architecture-aware scaling law that is simple and predictive
  - Novelty: extends Chinchilla to include architecture knobs as multiplicative or additive calibrations around `L_opt`, capturing U-shaped loss vs. `d/√N` and vs. `r_mlp/attn` (Section 3.3; Figures 3–4).
  - Evidence: strong fit quality when extrapolating to larger scales—e.g., fit on 80M, evaluate on 145M: “MSE: 0.0002, Spearman: 0.8909”; fit on 80/145M, evaluate on 297M: “MSE: 0.0001, Spearman: 0.7920”; fit on 80/145/297M, evaluate on 1B: “MSE: 0.0001, Spearman: 0.7451” (Figure 5).
- Clear architectural levers for inference speed at fixed parameter count
  - Insight: larger `d_model`, larger `r_mlp/attn`, and higher GQA all increase throughput across batch sizes (Figure 2; Appendix E Figures 8–10 and 11–13). Mechanistically justified by FLOPs and KV-cache I/O analysis (Appendix H).
- Practical search framework that respects accuracy constraints
  - Innovation: optimizes throughput subject to a loss target using the fitted conditional law and local GQA search (Algorithm 1; Eq. 4).
  - Outcome: produces Pareto-competitive designs with large measured serving gains while preserving accuracy (Table 1; Figure 6).
- Scaling the fit with size-aware data selection
  - Finding: when predicting the 3B regime, fitting the calibration using only 1B-scale data improves prediction (Figure 7 right: “MSE: 0.0001, Spearman: 1.0000”) versus mixing smaller scales (Figure 7 left: “MSE: 0.0001, Spearman: 0.5000”). This suggests re-fitting the calibration as models scale (Section 5.1, Ablation of fitting data; Table 2).

## 5. Experimental Analysis
- Evaluation methodology
  - Datasets and metrics: nine standard zero-shot benchmarks; performance reported as task accuracy and averaged across tasks (Section 4, LLM Evaluation Setup; Tables 6–7).
  - Baselines: LLaMA-3.2-1B and LLaMA-3.2-3B architectures trained under the same setup are primary points of comparison (Table 1).
  - Inference setup: vLLM on A100 40GB GPUs; throughput in tokens/s with 4096-in/1024-out contexts; five-run averages (Section 4, Inference Setup).

- Quantitative results
  - Scaling-law prediction accuracy
    > Figure 5 shows low prediction error and high rank correlation across extrapolation tasks (MSE ~1e-4; Spearman 0.75–0.89).
    > Outlier exclusion improves fit quality; including extreme `r_mlp/attn` (<0.5 or >5) degrades Spearman notably (Appendix G Figure 15 center vs. left).
    > Multiplicative and additive calibrations perform similarly, but joint non-separable calibration is worse (Appendix G Figure 16).
  - Throughput ablations (fixed `N_non-embed`)
    > Larger `d_model` increases tokens/s across batch sizes and model scales (Section 3.2; Figure 2 left; Appendix E Figure 8 and Figure 11).
    > Larger `r_mlp/attn` increases tokens/s (Figure 2 right; Appendix E Figure 9 and Figure 12).
    > Higher GQA increases tokens/s (Appendix E Figure 10 and Figure 13).
  - End-to-end models selected by the method
    - 1B scale:
      > Panda-1B (predicted loss-minimizing config: `d_model/√N ≈ 0.08`, `r ≈ 1.03`) improves average accuracy by +2.1 points over LLaMA-3.2-1B (57.0 vs. 54.9) with lower training loss (2.782 vs. 2.803) (Table 1; Figure 6 left).
      > Surefire-1B (Pareto-efficient, throughput-optimized under same loss target) achieves higher throughput than LLaMA-3.2-1B across batch sizes (Figure 6 center) and competitive accuracy (55.4) (Table 1).
    - 3B scale:
      > Panda-3B (predicted `d_model/√N ≈ 0.08`, `r ≈ 1.06`) outperforms LLaMA-3.2-3B in average accuracy (62.5 vs. 61.9) with similar or lower loss (2.619 vs. 2.625) (Table 1).
      > Surefire-3B (throughput-optimized under the same loss target) consistently increases tokens/s across batch sizes (Figure 6 right) and slightly improves averaged accuracy (62.6) (Table 1).
      > Fitting the calibration using 1B data yields Panda-3B° with even lower training loss (2.606) and the same mean accuracy (62.5) (Table 2; Figure 7 right).
  - Overall serving gains
    > The paper reports “up to 42% higher inference throughput compared to LLaMA-3.2” under the same training budget (Abstract; validated qualitatively in Figure 6 center/right).

- Do experiments support claims?
  - Yes, for the dense model regimes (≤3B) tested: the ablations repeatedly show the same throughput trends, the U-shaped accuracy trends are consistent across three small scales (Figures 3–4), the fitted law generalizes to larger scales with strong rank correlation (Figure 5), and architected models deliver better accuracy and throughput trade-offs (Table 1; Figure 6).
  - Caveats: throughput gains are measured under a specific serving stack (vLLM), hardware (A100-40GB), and prompt/response lengths (4096/1024). Results may differ under other inference engines or lengths (Section 4).

## 6. Limitations and Trade-offs
- Fixed depth and separability assumptions
  - Depth is fixed per parameter scale; the work does not explore the depth–width trade-off (“aspect ratio”) even though depth strongly affects generalization after fine-tuning (Introduction §1; Section 3.1).
  - The conditional scaling law assumes that hidden-size and ratio effects are separable (Eq. 3). Joint effects exist but were harder to fit and underperformed here (Appendix G Figure 16). This separability may not hold universally.
- Scope: dense models and pretraining loss
  - All scaling laws and most experiments target dense transformers; extensions to Mixture-of-Experts (MoE) are left open (Section 7). Appendix J shows MoE throughput trends but no scaling law for MoE.
  - The calibration is based on pretraining loss. Post-training (SFT, RL) might shift the architecture–accuracy frontier (Section 7).
- Fitting data and transfer across scales
  - Calibration coefficients can shift with model size; mixing very small with larger scales can harm extrapolation (Figure 7). In practice, one may need to re-fit at each new regime.
- Hardware and workload dependence
  - Throughput depends on GPU architecture, inference engine, batching, and context length (Section 4). Gains may vary outside the tested setup.
- Not a compute-allocation law
  - The work does not re-solve the “optimal trade between model size and data” (Chinchilla’s Eq. 2). It assumes `N` and `D` are fixed and optimizes architecture within that budget (Section 2).

## 7. Implications and Future Directions
- How this changes the landscape
  - Provides a practical recipe to design “inference-efficient yet accurate” LLMs at a given size and data budget, using a simple, well-fitting conditional scaling law and a small architecture search (Sections 3.3–3.4; Figure 5; Algorithm 1). This bridges a crucial gap between training-focused scaling and real deployment constraints.
- What it enables next
  - Extending the conditional calibration to:
    - Depth–width trade-offs and rotary/positional schemes, attention head dimensions, or activation functions.
    - MoE architectures: calibrations over number of experts, active experts, and router load balance (Appendix J hints similar throughput trends).
    - Quantization- and sparsity-aware scaling: adding terms that capture precision/sparsity effects on loss and throughput.
  - Size-aware fitting practice: re-fitting calibrations as one scales up—e.g., fit at 1/3 target size (Section 5.1; Figure 7)—to maintain predictive accuracy.
- Practical applications
  - Model providers can use the law to pre-screen architecture grids and train only a few candidates likely to hit a target latency/throughput at a desired accuracy.
  - Serving teams can codify Eq. (4) with their own hardware-specific throughput measurements to pick Pareto-optimal designs for different products (chat vs. batch summarization). For example, designs with `d_model/√N ≈ 0.074–0.082` and `r_mlp/attn ≈ 1–1.2` repeatedly sit near the predicted minima (Table 1–2; Section 5.1).

> Key takeaways to apply immediately:
> - At fixed parameters, increasing `d_model` and `r_mlp/attn` generally increases throughput by shrinking attention heads and KV I/O (Figure 2; Appendix H).
> - Accuracy is U-shaped in both `d_model/√N` and `r_mlp/attn`; the sweet spot in these experiments is around `d_model/√N ≈ 0.08` and `r ≈ 1` at the 1–3B scales (Section 5.1; Table 1–2).
> - Treat GQA as a local search knob for throughput; its effect on loss is inconsistent (Appendix F Figure 14), but higher values often help serving (Appendix E Figure 10).
