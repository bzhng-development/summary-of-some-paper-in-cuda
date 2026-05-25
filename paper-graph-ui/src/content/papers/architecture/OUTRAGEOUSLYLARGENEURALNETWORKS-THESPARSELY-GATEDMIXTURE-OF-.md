# OUTRAGEOUSLY LARGE NEURAL NETWORKS: THE SPARSELY-GATED MIXTURE-OF-EXPERTS LAYER

**ArXiv:** [1701.06538](https://arxiv.org/abs/1701.06538)

## 🎯 Pitch

This paper introduces the Sparsely-Gated Mixture-of-Experts (MoE) layer, a novel neural network component that selectively activates a small subset of specialized 'expert' sub-networks for each input, instead of the entire model. By combining scalable algorithmic innovations with practical systems engineering, it enables models with over 1000× more capacity without increasing computation proportionally—delivering state-of-the-art results on large-scale language modeling and machine translation tasks. This breakthrough makes it feasible to unlock unprecedented model expressiveness and accuracy at modest computational cost, paving the way for deep learning models to scale to new heights in data-hungry real-world applications.

---

## 1. Executive Summary
This paper introduces a scalable building block for deep networks—the `Sparsely‑Gated Mixture‑of‑Experts (MoE) layer`—that activates only a few specialized “experts” per input instead of the entire model. By combining sparse, differentiable gating with practical systems techniques, it scales model capacity by 1000×+ while keeping computation roughly constant, yielding state‑of‑the‑art results in language modeling and machine translation at substantially lower compute (Sections 1.2–3; Figures 2–3; Tables 1–5).

## 2. Context and Motivation
- Problem addressed
  - Modern neural models improve with more parameters, but activating every parameter on every example makes training and inference compute scale linearly with model size; with growing datasets this creates a “quadratic” cost pressure (Section 1.1).
  - “Conditional computation” aims to activate only parts of the network per example, but prior attempts struggled to deliver large real gains due to:
    - Hardware branching inefficiency on GPUs.
    - Shrinking effective batch size for conditionally active submodules.
    - Network bandwidth bottlenecks in distributed training.
    - Load imbalance across conditional branches (some experts get most traffic).
    - Evaluation on small datasets that do not justify very large capacity (Section 1.1, bullet list).
- Why this matters
  - Real‑world language tasks have massive training corpora (“100 billion words” in Section 5.2) and benefit from large capacity to memorize and generalize patterns. Achieving far larger capacity without proportional compute unlocks better accuracy and economical scaling.
- Prior approaches and their gaps
  - Mixture‑of‑Experts (MoE) has a long history (Section 1.3), and deep MoE variants existed (e.g., two stacked dense MoEs in Eigen et al. 2013), but lacked:
    - Sparse, trainable gating that works efficiently on GPUs.
    - A concrete recipe for balanced routing at scale.
    - A systems design that preserves throughput on multi‑GPU clusters.
- Positioning of this work
  - Provides a general, drop‑in `MoE` layer with sparse, differentiable gating (Section 2).
  - Solves engineering bottlenecks: batch size, bandwidth, and load balancing (Sections 3–4).
  - Demonstrates large‑scale wins on strong benchmarks and production‑scale translation (Section 5).

## 3. Technical Approach
At a high level, the paper inserts a `MoE` layer between stacked recurrent layers (e.g., LSTMs) and ensures that, for each input token, only a few experts run. This yields huge parameter counts (experts replicate parameters) without executing all of them.

- Core `MoE` formulation (Section 2; Eq. 1)
  - A `MoE` has `n` experts `E1 … En` (each a small feed‑forward network) and a gating network `G`.
  - For input `x`, compute expert outputs `Ei(x)` for a small subset and mix them with learned weights `G(x)i`:
    - `y = sum_i G(x)i * Ei(x)` (Eq. 1).
  - Only experts with nonzero `G(x)i` are executed—this is the compute saving.

- Gating network design (Section 2.1)
  - Concept: produce a sparse probability distribution over experts.
  - Two gating variants:
    1) `Softmax gating` (dense; Eq. 2): `Gσ(x) = Softmax(x · Wg)`. Not sparse, thus no compute saving.
    2) `Noisy Top‑K gating` (sparse; Eqs. 3–5): add learnable Gaussian noise per expert logit before softmax, then keep only the top‑`k` logits and set others to `−∞`. After softmax, only `k` experts get nonzero weights.
       - Pre‑activation: `H(x)i = (x·Wg)i + Normal(0,1) * Softplus((x·Wnoise)i)` (Eq. 4).
       - Sparsification: `KeepTopK(H(x), k)` zeroes all but the top `k` elements (Eq. 5), then apply `Softmax` (Eq. 3).
       - Why noise? It promotes exploration and helps load balancing (Appendix A discussion).
  - Training: end‑to‑end backpropagation through the top‑`k` softmax for the active experts; gradients flow to gating weights of selected experts and to the input of the gating network (Section 2.1).

- Balancing expert utilization (Section 4; Appendix A)
  - Challenge: without constraints, the gate collapses to a few experts, hurting both quality and system load balance (Section 4).
  - Two complementary regularizers:
    - `Limportance` (Eqs. 6–7): Define per‑expert “importance” as the batchwise sum of gate weights `Importance(X) = sum_{x in X} G(x)`. Penalize coefficient of variation (CV) across experts: `Limportance = wimportance * CV(Importance(X))^2`. This pushes the gate to distribute weight evenly.
    - `Lload` (Appendix A; Eqs. 8–11): Encourages equal counts of assigned examples (“load”) per expert. Because the actual count is discrete, they derive a smooth estimator `Load(X)i = sum_{x in X} P(x,i)` where `P(x,i)` is the probability that expert `i` is in top‑`k` for `x`, computed using the CDF `Φ` of the normal distribution and the difference to the `k`‑th largest competitor (Eqs. 8–9). Penalize `CV(Load(X))^2` with weight `wload` (Eq. 11).
  - Initialization trick: set `Wg` and `Wnoise` to zeros to start in a noise‑only, balanced regime (Appendix A).

- Hierarchical MoE (Appendix B)
  - When `n` is huge, use two levels: a primary gate selects a few groups; each group contains a secondary MoE that then selects experts. The composite output multiplies primary and secondary gates (Eq. 12). Importance/load metrics are adapted accordingly (Eqs. 13–14).
  - This reduces branching factor per decision and maps nicely to device partitions.

- Systems techniques to keep throughput high (Section 3)
  - Shrinking batch problem (Section 3.1): As experts multiply, each expert sees fewer examples per batch (≈ `k*b/n`), reducing arithmetic efficiency. Solutions:
    - `Mix data and model parallelism`: run multiple data‑parallel replicas synchronously, but keep a single shared copy of each expert across devices; route micro‑batches from all replicas to the expert’s host device, raising per‑expert batch size by factor `d` (number of replicas). If each device processes batch `b`, expert batch becomes `≈ k*b*d/n` (Section 3.1).
    - `Exploit convolutionality over time`: in sequence models, call the same MoE at every time step; execute it jointly over all time steps of the unrolled window to enlarge its batch (Section 3.1).
    - Note on recurrent MoE: applying MoE recursively inside an RNN breaks the above trick; they point to recomputation strategies to reduce memory and enlarge batch (Section 3.1).
  - Network bandwidth (Section 3.2): Keep the computation‑to‑communication ratio high by using experts with large hidden layers. With input/output size `d` and hidden size `h`, compute scales with `~2*d*h`, while communication scales with `~2*d`. The ratio equals `h`, so using thousands of hidden units makes network transfer negligible relative to compute (Section 3.2).
  - Memory/optimizer tweaks for very large models (Appendix D):
    - Do not store expert hidden activations; recompute on the backward pass.
    - Modify Adam: set `β1=0` to drop first moment; use a factored second‑moment approximation (store row/column averages for matrices) to cut auxiliary memory.

- Where the `MoE` layer sits in end‑to‑end models (Figures 1; Sections 5.1, 5.3, 5.4; Appendices C–E)
  - Language modeling: two stacked LSTMs with an MoE between them, called once per token position (Figure 1; Appendix C).
  - Machine translation (GNMT‑style): reduced‑depth encoder/decoder LSTMs with MoE layers inserted in both encoder and decoder; attention connects encoder and decoder (Section 5.3; Appendix E). For some MT runs they use a strictly balanced, batchwise gating variant with learned inference thresholds to ensure equal batch sizes per expert (Appendix F).

- Simple walk‑through example
  - Suppose `n=256` experts, `k=4`. For token representation `x`:
    1) Compute `H(x)` via Eq. 4 (linear transform plus noise with learnable scale).
    2) Mask to the `k=4` largest entries (Eq. 5), set others to `−∞`, apply softmax (Eq. 3) to get `G(x)`.
    3) Dispatch `x` only to the four chosen experts; get `E_i(x)` from each.
    4) Return weighted sum `y = Σ G(x)_i * E_i(x)` (Eq. 1).
  - Only 4/256 experts do work for this token—a 98.4% sparsity in execution.

## 4. Key Insights and Innovations
- Sparse, differentiable gating that scales to thousands of experts
  - Distinctive features: `Noisy Top‑K` gating (Eqs. 3–5) allows training with standard backprop through a sparse softmax, unlike boolean gates requiring REINFORCE. It keeps routing decisions differentiable for the selected experts and injects controlled noise to promote balanced exploration (Section 2.1; Appendix A).
  - Significance: Enables massive conditional computation without bespoke gradient estimators.

- Load‑balancing losses that make sparse MoE practical
  - `Limportance` and `Lload` explicitly balance both total attention weight and the number of routed examples per expert (Section 4; Appendix A). Experiments show quality collapses without them and becomes stable with them:
    > In Appendix A Table 6, removing both losses yields test perplexity 39.8 and extreme load imbalance (max load 17.8× mean). Adding either loss reduces perplexity to ≈35.6–35.7 and keeps max load ≈1.1–1.5× mean.
  - Significance: Prevents gate collapse, avoids out‑of‑memory skew, and keeps hardware fully utilized.

- A systems recipe that keeps compute efficiency high at enormous scale
  - Combining synchronous data parallelism with model‑parallel expert placement preserves large per‑expert batches (Section 3.1) and makes network transfers amortized and compute‑dominated by using large expert hidden layers (Section 3.2).
  - Significance: Measured throughput remains a substantial fraction of GPU peak:
    > Table 7 reports 0.74–1.56 TFLOPS/GPU for MoE models on K40s; the largest high‑compute model reaches 1.56 TFLOPS/GPU.

- Hierarchical MoE for extreme expert counts
  - Two‑level gating reduces branching pressure and maps well to device topology (Appendix B), enabling tens of thousands to over a hundred thousand experts (Appendix D; Figure 3).
  - Significance: Achieves up to 137B parameters in an `MoE` layer with retained efficiency:
    > Appendix D Table 8 shows `MoE‑131072‑h` with 137.6B parameters in the MoE component.

- Evidence of emergent specialization
  - Experts learn interpretable roles (e.g., handling phrases like “plays a central …” vs. “rapidly …”), illustrated in Appendix E Table 9 by sorting inputs with highest gate weights for each expert.
  - Significance: Supports the intended “divide‑and‑conquer” behavior.

## 5. Experimental Analysis
- Evaluation setup
  - Datasets and tasks
    - Language modeling: 1‑Billion‑Word Benchmark (829M words; Section 5.1) and a 100‑Billion‑Word Google News corpus (Section 5.2).
    - Machine translation: WMT’14 En→Fr and En→De (36M and 5M sentence pairs) and a production‑scale En→Fr dataset (Section 5.3). Multilingual MT on 12 language pairs (Section 5.4).
  - Metrics
    - `Perplexity`: standard intrinsic LM/MT metric (lower is better).
    - `BLEU`: translation quality (higher is better).
    - `ops/timestep`: forward multiply‑adds per token/time step excluding softmax/embedding (Appendices C–E).
    - `TFLOPS/GPU`: realized throughput (Sections 5.1–5.2; Tables 7–8).
  - Baselines
    - Strong LSTM language models from Jozefowicz et al. (Section 5.1; Figure 2‑right, top line).
    - GNMT and GNMT+RL translation systems (Tables 2–3).
  - Architectures
    - LM: 2 LSTMs with an MoE block between them; varied number of experts; typically `k=4` experts used per input; experts are 1‑hidden‑layer MLPs with thousands of hidden units (Appendix C).
    - MT: shallower GNMT with MoE in encoder and decoder (Appendix E). For multilingual, non‑hierarchical 512‑expert MoE with larger expert hidden size (Appendix E).

- Main quantitative findings
  - 1‑Billion‑Word LM, fixed compute budget (~8M ops/timestep)
    - Progressively adding capacity via more experts reduces perplexity:
      > Table 7: `LSTM‑2048‑512` (no MoE) = 44.7; `MoE‑32` = 39.7; `MoE‑256` = 35.7; `MoE‑1024‑h` = 34.6; `MoE‑4096‑h` = 34.1.
    - This is a ≈24% relative drop from 44.7 to 34.1 at roughly equal compute (Figure 2‑left).
  - 1‑Billion‑Word LM, increasing compute with high capacity (all ~4B MoE params)
    - > Table 1: `Low‑Budget` MoE: 34.1; `Medium`: 31.3; `High`: 28.0 test perplexity after 10 epochs.
    - The “High‑Budget MoE” at 142.7M ops/timestep (still modest) beats the previously best 10‑epoch perplexity 34.7 by a large margin and even undercuts the best published 100‑epoch result (30.6) when trained for only 10 epochs (Section 5.1; Table 1; Figure 2‑right).
  - 100‑Billion‑Word LM (single pass)
    - With ~8–10M ops/timestep kept approximately constant, increasing experts continues to help up to 65,536 experts (≈68B parameters):
      > Appendix D Table 8: `MoE‑65536‑h` test perplexity = 28.9 vs LSTM baseline 47.0; `MoE‑131072‑h` slightly degrades to 29.2.
    - Figure 3 shows the perplexity improvement curve after 10B and 100B words; improvements are larger with more data, supporting the “capacity helps more when data is large” thesis (Section 5.2).
    - Efficiency remains reasonable even at extreme scale: > “0.72 TFLOPS/GPU” for 65,536 experts; efficiency drops for 131,072 experts partly because batch size wasn’t scaled with GPU count (Appendix D, Table 8, and discussion).
  - Machine Translation, single‑pair
    - En→Fr (WMT’14):
      > Table 2: `MoE‑2048` achieves BLEU 40.35 (40.56 with longer training) vs GNMT 39.22 (GNMT+RL 39.92); perplexity 2.69 vs 2.79. Training used fewer/smaller GPUs and shorter time compared to GNMT.
    - En→De (WMT’14):
      > Table 3: `MoE‑2048` BLEU 26.03 vs GNMT 24.91; perplexity 4.64 vs 5.25.
    - Production En→Fr:
      > Table 4: Test BLEU 36.57 vs GNMT 35.56; perplexity 2.69 vs 2.87 after 1 day on 64 K40s (GNMT trained 6 days on 96 K80s).
  - Multilingual MT (12 directions)
    - > Table 5: The MoE model reduces dev perplexity by 19% vs the multilingual GNMT baseline (3.35 vs 4.14) and improves BLEU on 11/12 directions (e.g., De→En: 34.80 vs 31.17; Ja→En: 25.91 vs 21.62). It even beats single‑pair GNMT on 8/12 directions.
    - One regression (En→Ko: −1.79 BLEU) is attributed to overtraining on rare pairs due to oversampling (Section 5.4).

- Do the experiments support the claims?
  - Yes, across multiple settings the same pattern appears: holding compute roughly fixed while increasing expert count consistently improves quality (Figures 2–3; Tables 7–8).
  - The regularizers’ necessity is empirically established (Appendix A Table 6).
  - Throughput measurements indicate the systems recipe retains a substantial fraction of GPU peak (Tables 7–8), supporting the efficiency claim.

- Ablations/robustness
  - Explicit ablation of the two balancing losses shows their effect on perplexity and on load balance metrics (Appendix A Table 6).
  - Scaling study on data size (Figure 3) shows larger datasets unlock further gains from increased capacity.
  - Very extreme sparsity (131,072 experts) begins to hurt efficiency and slightly harms perplexity, illustrating a practical operating regime (Appendix D Table 8).

## 6. Limitations and Trade-offs
- Dependence on large data
  - The largest gains emerge on very large corpora (Figure 3). For small datasets, capacity may not be exploited and regularization/overfitting risks grow.
- Systems complexity and hardware assumptions
  - Requires multi‑GPU clusters with high‑bandwidth interconnects and careful synchronization to mix data/model parallelism (Section 3.1). Engineering the routing/all‑to‑all exchanges is non‑trivial.
- Communication‑compute balance
  - Efficiency assumes experts have large hidden layers so that compute dominates network transfer (Section 3.2). With small experts or slow networks, bandwidth can bottleneck.
- Gating discontinuities and gradient locality
  - Top‑K gating creates discontinuities (Section 2.1 acknowledges “theoretically scary discontinuities”), though no issues were observed. Gradients flow only to selected experts per example; training stability relies on noise and balancing losses.
- Potential under‑utilization at extreme sparsity
  - Very large expert counts (e.g., 131k) reduced efficiency and slightly worsened perplexity compared to 65k experts (Appendix D Table 8), indicating diminishing returns and practical ceilings.
- Scope of experts
  - Experts are simple feed‑forward MLPs; the work does not explore more expressive expert types or MoE inside recurrent weights (Section 3.1 notes this as future work).
- Inference considerations
  - A strictly balanced batchwise gate used in some MT experiments necessitated learning per‑expert thresholds for inference (Appendix F), which adds complexity if batch sizes differ between train/test.

## 7. Implications and Future Directions
- How this changes the field
  - Establishes a practical path to dramatic capacity increases with modest compute increases via `conditional computation`. The MoE layer becomes a reusable component that can be placed between standard layers (Figure 1), allowing researchers to trade parameters for quality without linear compute growth.
- Follow‑up research enabled or suggested
  - Recurrent and attention‑internal MoEs: replacing RNN or attention sub‑matrices with MoE blocks (Section 3.1).
  - Improved routing: exploring alternative sparse gates (e.g., structured sparsity, learned temperature, or entropy regularization) and better load‑balancing objectives beyond CV penalties (Sections 2.1, 4, Appendix A).
  - Expert architectures: using deeper or specialized experts (convolutional, recurrent, or task‑specific modules), or heterogeneous experts per layer.
  - Distributed training advances: optimizing the routing/all‑to‑all communication pattern; adaptive device placement for experts; integrating with pipeline and tensor parallelism.
  - Data curriculum and multilingual sharing: leveraging expert specialization for low‑resource transfer, domain adaptation, and continual learning (Section 5.4 suggests experts naturally specialize).
- Practical applications
  - Large‑scale language modeling for search, dictation, and text generation (Sections 5.1–5.2).
  - High‑quality, cost‑effective machine translation—including multilingual systems that cover many directions with one model (Sections 5.3–5.4).
  - Any setting where model capacity is the bottleneck but compute is constrained—e.g., serving large models with latency budgets by activating only a fraction of the network per request.

Overall, the paper delivers an end‑to‑end recipe—sparse differentiable gating, balancing losses, hierarchical structuring, and distributed systems tricks—that turns the long‑standing idea of conditional computation into measurable gains on important NLP tasks, at scales up to 137 billion parameters in a single layer (Appendix D Table 8).
