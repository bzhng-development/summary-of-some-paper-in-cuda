# Scaling Laws for Neural Language Models

**ArXiv:** [2001.08361](https://arxiv.org/abs/2001.08361)

## 🎯 Pitch

This paper empirically establishes precise power-law relationships between language model performance and three key factors: model size, dataset size, and training compute. By uncovering unified scaling equations that predictably govern performance across massive variations in each factor, the work provides a practical blueprint for compute-efficient training and resource allocation. These findings not only optimize the cost and effectiveness of scaling language models in practice, but also lay a quantitative foundation for future theoretical understanding of deep learning systems.

---

## 1. Executive Summary
This paper empirically maps how the test loss of autoregressive Transformer language models scales with three knobs—model size (`N` parameters), dataset size (`D` tokens), and training compute (`C`)—and shows each follows a smooth power law when not bottlenecked by the other two (Figure 1; Equations 1.1–1.3). It further derives a practical recipe for compute‑efficient training: train much larger models on comparatively modest data, stop well before convergence, and allocate most extra compute to increasing model size rather than training steps (Section 6; Figure 14; Equation 1.7).

## 2. Context and Motivation
- Problem addressed
  - Practitioners lack a predictive, quantitative guide for how language model performance will improve as they scale up parameters, data, and compute. Choices about model width/depth, batch size, and number of steps are typically ad hoc and hardware‑driven.
- Why it matters
  - Real‑world impact: Scaling decisions determine cost, time‑to‑train, and attainable accuracy for production models.
  - Theoretical significance: Discovering regularities (“scaling laws”) can be the basis of a future theory of deep learning performance, akin to “thermodynamic” laws that ignore microscopic details (Section 8).
- Prior work and gaps
  - Earlier studies observed that more data or bigger models often help, and some reported scaling patterns (e.g., [Hestness et al. 2017] cited in Section 7). But they did not provide:
    - A single, unified set of equations tying together `N`, `D`, `C`, steps `S`, and batch size `B`.
    - A compute‑optimal allocation strategy and its empirical validation.
    - Evidence that architectural “shape” (depth vs. width) matters little compared to scale (Figure 5; Figure 6).
- Positioning
  - The work focuses on autoregressive Transformer decoders (Section 2), spanning 6 orders of magnitude in model size and 8 in compute, to uncover general power laws (Figure 1, Figure 4, Figure 13–14). It also compares against LSTMs and recurrent/parameter‑reused Transformers (Figure 7; Figure 17).

## 3. Technical Approach
This is an empirical framework that defines precise variables, measures them across large ranges, and fits simple power‑law equations. Key terms (defined as used here):
- `Test loss L`: cross‑entropy (in nats per token) averaged over a 1024‑token context (Section 2.3).
- `Parameters N`: count of non‑embedding parameters in the Transformer (embeddings excluded to get cleaner scaling; Section 2.1; Figure 6 right). With typical hyperparameters, `N ≈ 12 · nlayer · d_model^2` (Equation 2.1).
- `Training compute C`: estimated non‑embedding FLOPs as `C ≈ 6 N B S` where `B` is tokens per batch and `S` the number of parameter updates (“steps”) (Section 2.1).
- `Dataset size D`: tokens available for training (Section 2.3).
- `Critical batch size Bcrit(L)`: the batch size at which parallelism is efficient; above it, bigger batches give diminishing returns. It grows as loss decreases (Section 5.1; Figure 10; Equation 5.3).
- `Smin(L)`: the minimal number of serial steps needed to reach a target loss if one trains at `B ≫ Bcrit` (Equation 5.4).
- `Cmin(L)`: the minimal compute to reach a loss if one trains at `B ≪ Bcrit` (Equation 5.5).

Step‑by‑step methodology
1. Architecture and parameterization
   - Decoder‑only Transformers with varied depth, width, attention heads, and feed‑forward size; models from ~3×10^5 to ~1.5×10^9 non‑embedding parameters (Section 3; Figure 6).
   - Show that when `N` is fixed, changing “shape” barely moves test loss (Figure 5), so “scale” is the main driver.
2. Compute and parameter counting
   - Estimate forward/backward FLOPs by counting multiplications and accumulations in each sublayer (Table 1); ignore context‑dependent terms when `d_model ≫ nctx/12` (Section 2.1).
3. Datasets and tokenization
   - WebText2: 22.9B tokens total, with 0.66B held out for testing; additional evaluations on Wikipedia, Books, Internet Books, and Common Crawl (Section 2.3).
4. Training set‑up
   - Mostly Adam, 2.5×10^5 steps, batch size of 512 sequences × 1024 tokens, cosine decay schedule after warmup (Section 2.2). Learning‑rate schedule choice is largely irrelevant if warmup and decay are present (Appendix D.6; Figure 22).
5. Identify single‑variable power laws
   - Train many models to near convergence (or early stop) while varying only one bottleneck (model size, data size, or compute). Fit power laws:
     - `L(N) = (Nc/N)^αN` when data/compute are not limiting (Equation 1.1; Figure 1, right).
     - `L(D) = (Dc/D)^αD` with early stopping when data is limiting (Equation 1.2; Figure 1, middle).
     - `L(Cmin) = (Cmin_c/Cmin)^αCmin` when compute is optimally allocated (Equation 1.3; Figure 13).
6. Joint scaling and overfitting model
   - Fit a two‑variable equation that simultaneously captures capacity limits (`N`) and data limits (`D`), including how overfitting appears when one is too small:  
     `L(N, D) = [ (Nc/N)^(αN/αD) + (Dc/D) ]^αD` (Equation 1.5).  
     Visualized in Figure 4 (left) and Figure 9 (left).
7. Learning‑curve model vs. time (steps)
   - After warmup, learning curves are well fit by an additive model of “capacity term” plus “optimization term”:  
     `L(N, Smin) = (Nc/N)^αN + (Sc/Smin)^αS` (Equation 1.6).  
     Fits shown in Figure 4 (right) and used to analyze performance at fixed steps or compute (Figure 11).
8. Critical batch size and step/compute normalization
   - Measure `Bcrit(L)` empirically (Figure 18), verify it depends primarily on the target loss, not model size, and fit `Bcrit(L) ≈ B* · L^(−1/αB)` with `αB ≈ 0.21` (Equation 5.3; Figure 10).
   - Use `Bcrit(L)` to standardize reported steps (`Smin`) and compute (`Cmin`) across runs that used sub‑optimal batch sizes (Equations 5.4–5.5), enabling apples‑to‑apples comparisons (Section 5).
9. Derive compute‑efficient allocations
   - Combine the learning‑curve model with the `Bcrit` law to analytically minimize loss for a fixed compute budget. This yields how optimal model size, batch size, and steps scale with `Cmin` (Equations 1.7–1.8; Section 6.2; Appendix B).

Intuition
- The additive forms in Equations 1.5 and 1.6 say: you pay two separate “penalties”—one for finite capacity (finite `N`) and one for finite data or training time. Making either very large removes its penalty, leaving the other to dominate.

## 4. Key Insights and Innovations
- Smooth power laws over huge ranges
  - Finding: Loss scales as a power of `N`, `D`, or `Cmin` across 6–8 orders of magnitude without visible kinks (Figure 1; Figure 13).
  - Why it’s significant: Predictability enables forward planning; doubling `N` reduces loss by a constant factor `2^(−αN) ≈ 0.95` (Section 1.2).
- A unifying equation for model‑ and data‑limited regimes
  - Innovation: The joint equation `L(N, D)` (Equation 1.5) captures both the “infinite data” capacity limit and the “finite data” overfitting limit with one functional form; it collapses overfitting behavior as a function of `(N^(αN/αD))/D` (Figure 9 right).
  - Significance: Gives a concrete rule for how fast to grow data with model size to avoid overfitting: `D ∝ N^(αN/αD) ≈ N^0.74` (Section 4.2; Equation 4.4).
- Compute‑optimal training: stop far from convergence and scale the model
  - Result: For a fixed compute budget, the optimal strategy is to spend most compute on increasing `N`, grow batch size moderately, and increase steps minimally:  
    `N ∝ Cmin^0.73`, `B ∝ Cmin^0.24`, `S ∝ Cmin^0.03` (Equation 1.7; Figure 14).  
    Loss improves as `L ∝ Cmin^(−0.050)` (Equation 1.3; Figure 13).
  - Impact: Contradicts common practice of training small models to full convergence; encourages large models trained for fewer steps (Figure 11; Figure 12).
- Architecture “shape” matters little relative to scale
  - Evidence: Holding `N` fixed, wide vs. deep vs. head counts change loss only a few percent (Figure 5), and using non‑embedding parameters aligns models of different depths on one curve (Figure 6 right).
  - Implication: For planning, treat “parameter count” as the key design variable; many hyperparameters are second order.
- Critical batch size grows as loss falls
  - Measurement: `Bcrit(L)` roughly doubles when loss drops by ~13% (Figure 10), and depends on performance rather than `N`. This matches the gradient noise scale theory (Section 5.1), guiding parallelization and wall‑clock optimization.

## 5. Experimental Analysis
- Set‑up
  - Models: Transformer decoders from ~3×10^5 to ~1.5×10^9 non‑embedding parameters (Figure 6).
  - Training: Mostly Adam, 2.5×10^5 steps, batch size 512×1024 tokens; largest models with Adafactor (Section 2.2).
  - Data: WebText2 (22.9B tokens; 0.66B test); additional held‑out evaluations on Wikipedia, Books, Internet Books, Common Crawl (Section 2.3).
  - Metrics: Test cross‑entropy loss in nats/token over 1024‑token contexts (Section 2.3).
- Main quantitative results
  - Single‑factor power laws (Section 1.2; Figure 1; Table 5):
    - `L(N) = (Nc/N)^αN`, with `αN ≈ 0.076`, `Nc ~ 8.8×10^13` (non‑embedding params).
    - `L(D) = (Dc/D)^αD`, with `αD ≈ 0.095`, `Dc ~ 5.4×10^13` (tokens).
    - `L(Cmin) = (Cmin_c/Cmin)^αCmin`, with `αCmin ≈ 0.050`, `Cmin_c ~ 3.1×10^8 PF‑days` (Figure 13).
  - Joint scaling and overfitting (Section 4; Figure 9; Table 2):
    - Fitted `αN = 0.076`, `αD = 0.103`, `Nc = 6.4×10^13`, `Dc = 1.8×10^13` in Equation 1.5, predicting overfitting grows with `(N^(αN/αD))/D`.
    - Data requirement to avoid overfitting within run‑to‑run noise (~0.02 loss):  
      `D ≳ (5×10^3) · N^0.74` (Equation 4.4).
  - Learning curves vs. steps (Section 5; Equation 1.6; Table 3):
    - `αS ≈ 0.76`, `Sc ≈ 2.1×10^3 steps`, giving accurate post‑warmup fits (Figure 4 right), and enabling performance predictions at fixed steps/compute (Figure 11).
  - Critical batch size (Section 5.1; Figure 10):
    - `Bcrit(L) ≈ B* · L^(−1/αB)`, with `αB ≈ 0.21`, `B* ≈ 2×10^8 tokens` (Equation 5.3). Verified by explicit batch scans (Figure 18).
  - Compute‑efficient scaling (Section 6; Figure 14):
    - Optimal `N(Cmin) ∝ Cmin^0.73` and `Smin(Cmin) ∝ Cmin^0.03`. Most extra compute should enlarge the model; serial steps barely grow.
  - Generalization across datasets (Section 3.2.2; Figure 8):
    - Loss on other corpora aligns as a near‑parallel power law in `N`, with a roughly constant offset relative to WebText2. Performance on new distributions improves proportionally to in‑distribution validation loss, independent of training phase.
  - Architecture shape ablations (Section 3.1; Figure 5; Figure 6):
    - For fixed `N`, changing depth/width/feed‑forward/head counts shifts loss by only a few percent. Counting non‑embedding parameters aligns models of different depths (Figure 6 right).
  - Comparisons to LSTMs and recurrent Transformers (Section 3.2.1; Figure 7; Figure 17):
    - LSTMs match Transformer loss on early tokens in the context but plateau after ~100 tokens, while Transformers keep improving across the full context (Figure 7 right). Recurrent (parameter‑reused) Transformers perform slightly better at fixed `N` but worse per FLOP (Figure 17).
  - Sample efficiency (Figures 2 and 19):
    - Larger models reach the same loss with fewer tokens and fewer steps; `Emin` and `Smin` shrink rapidly with `N` (Figure 19).
- Robustness and diagnostics
  - Early stopping: derived lower bound on early stopping step as a function of overfitting gap using Equation 5.7; matches trends (Figure 16 left).
  - Training vs. test loss: Overfitting is underestimated by train‑test gaps; the joint scaling model better reflects true overfitting relative to the infinite‑data limit (Figure 16 right).
  - LR schedules: Many schedules yield similar final loss if warmup and decay exist (Appendix D.6; Figure 22).
- Do the experiments support the claims?
  - Yes, within the explored ranges: straight log‑log lines over orders of magnitude (Figures 1, 4, 13–14) and accurate two‑variable fits (Figures 4, 9, 11) indicate consistent power laws. The compute‑optimal allocations predicted analytically (Section 6.2; Appendix B) closely match the empirical exponents (Figure 14).
  - Caveat: Small‑data regime (~2×10^7 tokens) deviates from the fit (Table 2 note; Figure 16), and compute estimates omit context‑dependent FLOPs (Section C Caveats).

> “As the computational budget C increases, it should be spent primarily on larger models, without dramatic increases in training time or dataset size” (Equation 1.7; Figure 14; Section 6).

## 6. Limitations and Trade-offs
- Assumptions baked into the formulas
  - Power‑law behavior persists across the studied ranges; eventual saturation is acknowledged but not empirically observed (Section 6.3; Figure 15).
  - Counting compute as `C ≈ 6NBS` ignores context‑length terms that may grow important when `nctx` is large relative to `d_model` (Section C Caveats; Section 2.1).
  - `Bcrit(L)` is extrapolated from observed losses; behavior at much lower loss may differ (Section C Caveats).
- Regimes not covered or weakly supported
  - Very small datasets: with ~2×10^7 tokens (only ~40 updates per epoch), fits degrade and overfitting dynamics differ (Section 4.2; Figure 16).
  - Extremely large contexts: not systematically varied; token‑position analysis suggests long‑range patterns are learned later (Appendix D.5; Figure 20–21).
  - Beyond explored scales: The derived contradiction between `L(Cmin)` and data growth implies current laws must bend before predicted intersection (Section 6.3; Figure 15).
- Practical trade‑offs
  - Compute‑efficient training uses larger models (higher inference cost) to reduce steps; this may be undesirable for deployment latency/footprint even if it saves training compute (Appendix B.4; Figure 12).
  - The constants `Nc`, `Dc`, `B*` depend on tokenization/vocabulary, so absolute numbers are not universal (Section 1.2).
- Open questions
  - No mechanistic theory explains why these specific exponents appear; the paper calls for a “statistical mechanics” of learning curves (Section 8).
  - How `Bcrit` evolves outside the measured loss range and across architectures/modalities remains unclear (Section C Caveats).

## 7. Implications and Future Directions
- How this changes the field
  - Planning: Teams can now predict returns on investment when scaling `N`, `D`, `C` and choose compute‑optimal allocations (Equations 1.3, 1.7–1.8; Figures 13–14).
  - Methodology: Emphasis shifts from meticulous depth/width tuning to hitting the right parameter count and compute allocation; “big models over big data” clarifies to “bigger models over moderately bigger data.”
  - Benchmarking: Results suggest smoother progress in test loss can hide qualitative capability jumps; tracking both loss and emergent behaviors is important (Section 8).
- Follow‑up research enabled/suggested
  - Theory:
    - Derive the additive forms in Equations 1.5 and 1.6 from optimization dynamics and data statistics; connect `αS` to Hessian spectra (Section 5.2 discussion).
    - Explain why exponents take these values and whether they generalize to other domains (images, audio, video) and objectives (Section 8).
  - Practice:
    - Validate and refine `Bcrit(L)` at lower losses; design schedulers that track instantaneous `Bcrit` (Section 5.1; Appendix D.3).
    - Explore model‑parallel architectures that scale `N` efficiently with minimal serial steps, e.g., pipeline and tensor model parallel methods (Section 8).
    - Investigate curriculum or optimizer changes that accelerate early training, since compute‑efficient runs stop early (Section 6.1).
    - Study regularization and augmentation to improve the small‑data regime (Section C Caveats; Figure 16).
- Practical applications
  - Capacity planning for large‑scale pretraining: estimate data requirements (`D ∝ N^0.74`; Equation 4.4), expected loss at given compute (`L ∝ Cmin^(−0.050)`; Figure 13), and optimal batch/step schedules (Figure 14).
  - Cross‑domain transfer: Since loss improvements on new distributions track training‑distribution loss with nearly constant offsets (Figure 8), lowering validation loss is a reliable proxy for better transfer.

> “Compute‑efficient training stops far short of convergence” (Figure 2; Section 6), and “bigger models are significantly more sample‑efficient” (Figures 2 and 19). Together, these change how we budget data, compute, and time when training modern language models.
