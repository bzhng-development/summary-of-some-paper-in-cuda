# OUTRAGEOUSLY LARGE NEURAL NETWORKS: THE SPARSELY-GATED MIXTURE-OF-EXPERTS LAYER

**ArXiv:** [1701.06538](https://arxiv.org/abs/1701.06538)

## 🎯 Pitch

This paper introduces the Sparsely-Gated Mixture-of-Experts (MoE) layer, enabling neural networks to scale to hundreds of billions of parameters by activating only a small, learned subset of 'experts' for each input—achieving conditional computation at unprecedented scale. By solving practical challenges in efficiency, batch size, and load balancing, the MoE layer massively boosts model capacity without a proportional increase in computation, achieving state-of-the-art results in language modeling and machine translation while keeping compute costs manageable—opening the door to much larger and more capable neural models.

---

## 1. Executive Summary
This paper introduces the Sparsely-Gated Mixture-of-Experts (`MoE`) layer—a way to grow a neural network’s parameter count by orders of magnitude while activating only a small, learned subset of those parameters for each input. It solves the core bottleneck that model capacity usually scales linearly with computation by achieving “conditional computation”: the model uses only a few experts per example, enabling up to 137B parameters (Figure 3 and Table 8) with competitive or lower compute than strong baselines, and it sets new results in language modeling and machine translation (Tables 1–4).

## 2. Context and Motivation
- Problem/gap:
  - Large neural models perform better but require proportionally more compute. When every parameter is active for every example, training cost scales with both model size and data size, creating a roughly quadratic burden (Section 1.1).
  - “Conditional computation” activates only parts of a model per example, but previous attempts struggled with GPU efficiency, small effective batch sizes for conditional parts, network bandwidth overheads, unbalanced expert usage, and limited data scales (Section 1.1, bullet list).
- Importance:
  - Real-world: tasks with huge corpora (e.g., language modeling, translation) need capacity to absorb rare events and long-tail knowledge.
  - Theoretical: demonstrates a practical realization of conditional computation at scale.
- Prior approaches and limitations:
  - Earlier conditional computation methods turned on/off network chunks but failed to show “massive improvements” in capacity or training time due to hardware and algorithmic issues (Section 1.1).
  - Classic Mixture-of-Experts (MoE) usually treats the mixture as the entire model (Section 1.3); Eigen et al. (2013) explored MoEs as components but without large-scale sparse gating or robust load balancing.
- This paper’s positioning:
  - Uses MoE as a reusable layer within deep sequence models (Figure 1), adds sparse gating with noise (Section 2.1), and provides engineering and algorithmic solutions for batching, bandwidth, and load balancing (Sections 3–4 and Appendix A). It evaluates on very large corpora where capacity is critical (Sections 5.1–5.4).

## 3. Technical Approach
The `MoE` layer is a set of `n` feed-forward “experts” and a small “gating network” that decides which experts to use for each input.

- Core computation (Section 2, Eq. 1):
  - For input `x`, each expert `E_i(x)` proposes an output. The gate produces a weight `G(x)_i` per expert. The layer’s output is a weighted sum y = Σ_i G(x)_i E_i(x).
  - Sparsity is enforced so that only the top `k` experts (e.g., k=2–4) have nonzero `G(x)_i`; the rest are skipped, saving compute.

- Gating network (Section 2.1):
  - Dense baseline: `Softmax` gating (Eq. 2) multiplies input by `W_g` and normalizes; this activates all experts.
  - Sparse, trainable gating with noise: “Noisy Top-K Gating” (Eqs. 3–5).
    - Before the Softmax, additive Gaussian noise—whose scale is learned via `W_noise` and `softplus`—is added to each pre-activation (Eq. 4).
    - Keep only the top `k` components; set the others to `-∞` so their post-softmax weights become 0 (Eq. 5).
    - Why noise? It helps distribute load across experts by making the selection less brittle and improves differentiability for load estimators (Appendix A).
  - Training: standard backpropagation through the gate and experts; with `k>1`, the top selected gates remain differentiable for those experts (Section 2.1).

- Addressing “shrinking batch” efficiency (Section 3.1):
  - Problem: With `b` examples per worker, if each example activates only `k` of `n` experts, each expert sees roughly `kb/n` examples—too small for efficient GPU matmuls.
  - Solution 1—Mix data and model parallelism: run `d` data-parallel replicas of all shared layers and the gating network but shard the experts across devices. Each expert receives the union of relevant examples from all `d` replicas, boosting its effective batch to about `kbd/n` (Section 3.1).
  - Solution 2—Exploit “convolutionality” in sequence models: apply the same `MoE` to every time step and batch all time steps together when possible, further increasing expert batch size (Section 3.1).
  - Note: For fully recurrent MoEs (e.g., replacing RNN matrices), they point to activation checkpointing/recomputation to raise batch size (Section 3.1; Gruslys et al., 2016).

- Keeping network communication efficient (Section 3.2):
  - Communication is dominated by sending expert inputs/outputs across devices. To maintain a high compute-to-communication ratio, experts are made “compute-heavy” with large hidden layers (thousands of units). The ratio roughly equals the hidden layer size, so larger hidden layers improve efficiency (Section 3.2).

- Load balancing so experts are actually used (Section 4 and Appendix A):
  - Without constraints, the gate can collapse to a few experts (self-reinforcement) (Section 4).
  - Loss 1—`L_importance` encourages equal total gate weight per expert across a batch:
    - Define `Importance(X)_i = Σ_x∈X G(x)_i` (Eq. 6).
    - Add `w_importance * (CV(Importance(X)))^2` (Eq. 7), where `CV` is the coefficient of variation (std/mean). Lower is more balanced.
  - Loss 2—`L_load` encourages equal expected number of assigned examples (Appendix A):
    - Define `P(x,i)`: the probability that expert `i` is selected for `x`, using the noise distribution and the k-th threshold among other experts (Eqs. 8–9).
    - `Load(X)_i = Σ_x P(x,i)` (Eq. 10).
    - Add `w_load * (CV(Load(X)))^2` (Eq. 11).
  - Initialization ensures early balance by setting `W_g` and `W_noise` to zero so experts start equally likely (Appendix A).

- Hierarchical `MoE` to scale to many experts (Appendix B):
  - A two-level gate reduces branching: a primary gate selects among groups, then each selected group has its own secondary gate (Eq. 12). Analogous “importance” and “load” metrics extend to the hierarchy (Eqs. 13–14).
  - Used to scale to tens or hundreds of thousands of experts (Sections 5.1–5.2).

- Alternative strictly-balanced gating (Appendix F, used in some MT runs):
  - During training, a batchwise mask `M_batchwise` selects exactly `m=k|X|/n` items for each expert across the batch (Eqs. 16–18), making per-expert batch sizes identical—useful for certain infrastructure.
  - At inference, since batches may be small, a learned per-expert threshold vector `T` approximates the training mask (Eq. 19) using a consistency loss between batchwise and threshold masks (Eq. 20).

- Implementation notes for very large models (Appendix D):
  - Activation recomputation: do not store expert hidden activations; recompute them on backward pass to save memory.
  - Optimizer memory: Adam with `β1=0` and a factored second-moment estimator for matrices (maintain row/column averages instead of full matrices) to reduce optimizer state.

## 4. Key Insights and Innovations
- Sparsely-gated `MoE` as a reusable layer inside deep sequence models (Figure 1, Section 2):
  - Novelty: prior MoE work often used the mixture as the whole model; here it is a drop-in layer used between LSTMs and applied per token (“convolutionally” over time).
  - Significance: lets the model select different experts by token position, capturing fine-grained syntactic/semantic specialization (Appendix E, Table 9 shows experts focusing on patterns like “plays a central …”, or “with rapidly growing …”).

- Noisy Top-K gating with differentiable load estimators (Section 2.1 and Appendix A):
  - Novelty: trains a sparse gate via backprop while injecting learned Gaussian noise to improve exploration and enable smooth load estimates `P(x,i)` for `L_load`.
  - Significance: solves expert collapse and enables balanced utilization without REINFORCE-style estimators (contrast with Bengio et al., 2015).

- Scalable batching via mixed data/model parallelism (Section 3.1):
  - Novelty: unifies synchronous data-parallel replicas for shared layers with model-parallel sharded experts; each expert aggregates examples from all replicas, fixing the “shrinking batch” problem.
  - Significance: keeps GPU matmuls large and efficient even with thousands of experts; stated goal becomes feasible: “train a trillion-parameter model on a trillion-word corpus” by adding hardware (Section 3.1).

- Compute/communication co-design of experts (Section 3.2):
  - Insight: design expert MLPs with large hidden layers so the computation-to-I/O ratio exceeds GPU/network ratios.
  - Significance: maintains TFLOPS efficiency even with 99.994% layer sparsity and 65k experts (Section 5.2).

- Hierarchical `MoE` (Appendix B; used in Sections 5.1–5.2):
  - Novelty: two-level gating reduces the selection fanout for extreme scales (e.g., 4k–131k experts).
  - Significance: enables layers with up to 137B parameters (Table 8) while keeping compute similar.

## 5. Experimental Analysis
- Evaluation setup:
  - Datasets:
    - Language modeling: 1B Word Benchmark (829M words; 793k vocab; Section 5.1), and a 100B-word Google News corpus (Section 5.2).
    - Machine translation (MT): WMT’14 En→Fr (36M sentence pairs) and En→De (5M; Tables 2–3), plus a production En→Fr dataset (Table 4).
    - Multilingual MT: 12 language pairs from Johnson et al. (2016) (Table 5).
  - Metrics: Perplexity for language modeling; BLEU for MT (Appendix E).
  - Architectures:
    - LM: two stacked LSTMs with `MoE` between them; residual connections and dropout (Appendix C.1). Experts are 2-layer MLPs (ReLU) with hidden size 1024 (1M params/expert).
    - MT: slimmed GNMT enc/dec with `MoE` after selected LSTM layers; each `MoE` has up to 2048 experts with ≈2M params/expert (Appendix E).
    - Multilingual: 512 experts (non-hierarchical) per MoE with larger hidden size 8192 (Appendix E).
  - Compute controls:
    - Many LM comparisons hold ops/timestep roughly constant at ≈8M to isolate the effect of capacity (Figure 2-left; Table 7).
    - Variants also test higher compute budgets (33.8M and 142.7M ops/timestep) while keeping MoE capacity around 4B parameters (Table 7).

- Main results and comparisons:
  - 1B Word LM (computation fixed ~8M ops/timestep):
    - Increasing experts from 4 to 4096 (using hierarchical MoE) reduces test perplexity from ≈45 to 34.1, a ≈24% drop at similar compute (Figure 2-left; Table 7: MoE-4096-h 34.1 vs strong baselines ~44–46).
  - 1B Word LM (high-capacity, varied compute):
    - With ~4B parameters in `MoE`, increasing compute improves further: `MoE-34M` achieves 31.3; `MoE-143M` achieves 28.0 test perplexity (Table 7 bottom), outperforming the best published non-MoE baseline 34.7 (10 epochs) with less compute (Table 1).
    - Table 1 highlights:
      > “Even the fastest of these models beats the best published result … despite requiring only 6% of the computation.”
  - Efficiency:
    - Observed TFLOPS/GPU ranges from 0.74–1.56 for MoE models (up to 46% of total FLOPs in experts), and 1.07–1.29 for comparable non-MoE baselines (Section 5.1 “Computational Efficiency”; Table 7).
  - 100B Word LM (extreme capacity; Table 8; Figure 3):
    - Perplexity continues to improve up to 65,536 experts (≈68B parameters): test perplexity 28.9 after 1 epoch vs 47.0 for a matched 4xLSTM baseline—39% lower (Table 8).
    - Scaling to 131,072 experts (≈138B params) slightly degrades to 29.2 (Table 8), possibly “too much sparsity” (Section 5.2). Efficiency drops to 0.30 TFLOPS/GPU because batch size was not scaled with GPUs for comparability (Table 8 “observed”).
  - Single-pair MT (Tables 2–3):
    - En→Fr: BLEU 40.56 (longer training; perplexity 2.63) vs GNMT 39.22 (2.79 perplexity), using ~60% fewer ops/timestep (85M vs 214M). Training on 64 K40s for 3–6 days (Table 2).
    - En→De: BLEU 26.03 vs GNMT 24.91 with similarly lower compute (Table 3).
    - Production En→Fr: +1.01 BLEU over GNMT with 1/6th training time (Table 4).
  - Multilingual MT (Table 5):
    - Dev perplexity improves by 19% (4.14 → 3.35).
    - `MoE-Multi` surpasses the multilingual GNMT baseline on 11/12 language pairs, with gains up to +5.84 BLEU (Korean→English). It even beats monolingual GNMT on 8/12 pairs. One regression (English→Korean −1.79 BLEU) is attributed to overtraining of rare pairs (Table 5 note).

- Ablations and diagnostics:
  - Load-balancing losses matter (Appendix A, Table 6):
    - Without `L_importance`/`L_load`, perplexity is 39.8 and the most loaded expert gets 17.8× the average load.
    - With either loss alone, perplexity ≈35.6–35.7 and max/mean load reduces to 1.15–1.47. With both and larger weights, imbalance drops to ~1.07 with similar perplexity.
  - Expert specialization (Appendix E, Table 9): sorted examples show distinct syntactic/semantic niches per expert (e.g., “plays a central/critical/vital …”).

- Overall assessment:
  - The experiments convincingly demonstrate that conditional computation via MoE can raise parameter capacity by 1000× with modest or even reduced compute (Abstract; Sections 5.1–5.3), while retaining GPU efficiency and balancing loads.
  - The results are robust across tasks (LM and MT), data scales (1B and 100B words), and settings (single and multilingual MT).

## 6. Limitations and Trade-offs
- Dependence on large-scale distributed infrastructure:
  - The approach assumes access to many GPUs and high-throughput networking; some results use 64–128 GPUs (Appendix D). Efficiency degrades if the batch size per expert or global batch is not scaled appropriately (Table 8, 131k-expert model at 0.30 TFLOPS/GPU).
- Sparsity edge cases:
  - Extremely high expert counts can over-sparsify gating, slightly hurting quality (Section 5.2: 131,072 experts performs worse than 65,536).
- Gating discontinuities and noise:
  - Top-k gating introduces non-smooth decisions; while training works empirically, theoretical smoothness is only approximate (Section 2.1). Noise magnitude must be learned and well-tuned to balance exploration and stability.
- Load-balancing hyperparameters:
  - `w_importance` and `w_load` require tuning (Appendix A, Table 6). Under-regularization creates severe imbalance and poor perplexity; over-regularization could constrain specialization.
- Inference-time gating differences:
  - The strictly balanced training mask (Appendix F) needs a learned threshold at inference, introducing a potential train–test mismatch.
- Scope of experts:
  - Experts are feed-forward MLPs in this work (Section 2). Replacing recurrent weight matrices with MoE would remove the “convolutional” batching advantage and needs additional memory strategies (Section 3.1).
- Communication overhead at inference:
  - While compute/communication is engineered for training efficiency (Section 3.2), end-to-end latency implications for real-time inference across many devices are not explored in detail.

## 7. Implications and Future Directions
- Field impact:
  - Demonstrates that conditional computation can finally “unlock” massive capacity with manageable compute, making models with tens to hundreds of billions of parameters trainable and useful. The paper’s engineering principles (sparse gating, load-balanced routing, mixed parallelism) have become foundational for scaling sequence models (Sections 2–4, 5.1–5.3).
- Research avenues:
  - Trillion-parameter regime: Section 3.1 articulates the path—add experts and devices while keeping expert batch, memory, and step time constant.
  - Recurrent or Transformer-style MoE everywhere: replace more weight matrices with MoE blocks and pair with activation checkpointing and factored optimizer states (Sections 3.1 and Appendix D).
  - Better routers: explore continuous relaxations, entropy regularization, or learned budgets to further stabilize and optimize routing beyond noisy top-k and CV-based losses (Sections 2.1 and 4; Appendix A).
  - Adaptive expert granularity: vary expert hidden sizes to match compute/comm constraints dynamically (Section 3.2 insight).
  - Cross-domain applications: vision, speech, and multimodal tasks with very large datasets could benefit from sparse capacity increases (Conclusion).
- Practical applications:
  - Large-scale language modeling (autocomplete, ASR language models), machine translation (Tables 2–5), and multilingual systems that share capacity while specializing per language-pair (Table 5)—all with improved quality at similar or lower compute.

> “We obtain greater than 1000x improvements in model capacity with only minor losses in computational efficiency…” (Abstract)

> “We present model architectures in which a MoE with up to 137 billion parameters is applied… [and] achieve significantly better results than state-of-the-art at lower computational cost.” (Abstract; Sections 5.1–5.3)

In sum, the paper provides both the conceptual mechanism (sparse, learned routing to experts) and the systems recipe (batching, bandwidth-aware expert design, and load-balancing losses) that make outrageously large neural networks practical.
