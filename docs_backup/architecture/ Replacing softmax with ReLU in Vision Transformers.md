# Replacing softmax with ReLU in Vision Transformers

**ArXiv:** [2309.08586](https://arxiv.org/abs/2309.08586)
**Authors:** Mitchell Wortsman, Jaehoon Lee, Justin Gilmer, Simon Kornblith
**Institutions:** Google DeepMind

## 🎯 Pitch

This paper revolutionizes Vision Transformers by replacing softmax attention with a simple point-wise ReLU divided by sequence length, dramatically simplifying computation while maintaining accuracy. By eliminating the sequence normalization step, it enhances parallelizability and optimization, paving the way for efficient and scalable ViT architectures without sacrificing performance.

---

## 1. Executive Summary
This paper shows that Vision Transformers (ViTs) can replace the attention softmax with a simple point‑wise ReLU activation if the attention scores are divided by the sequence length `L` (i.e., use `ReLU / L`). With this single change, ViTs trained on ImageNet‑21k achieve scaling behavior and accuracy that approach or match standard softmax attention, while enabling easier parallelization because normalization across the sequence is no longer required.

## 2. Context and Motivation
- Problem addressed
  - Standard attention uses a softmax to convert similarity scores between tokens into a probability distribution across the sequence. This softmax requires both an expensive exponentiation and a sum across the sequence dimension, which is a synchronization bottleneck on modern accelerators.
  - Prior attempts to remove softmax by using point‑wise activations (e.g., plain ReLU) often degrade accuracy.

- Why it matters
  - Removing the sequence‑wise normalization step would allow more parallel computation over sequence elements with fewer cross‑token “gather” operations, which can translate into better hardware utilization (Section “Introduction”; Figure 1 caption).
  - If accuracy can be preserved, this would simplify attention and open new implementation strategies, especially for large ViTs.

- Prior approaches and their gaps
  - Point‑wise activations without normalization: Replacing softmax with ReLU or squared‑ReLU has been explored, but prior work typically did not divide by sequence length and thus lost accuracy (Section 2).
  - Alternatives that still normalize across the sequence: Some methods remove softmax but keep a sequence‑wise normalization so weights sum to one, preserving the bottleneck (Section 2).
  - Linear attention: Methods that remove nonlinearities entirely to achieve linear complexity help with very long sequences, but in this paper’s setting they reduced accuracy (footnote 1 under Section 3 and the line “removing the activation entirely reduced accuracy” in Section 2).

- Positioning
  - The paper proposes a minimal change to standard attention—swap softmax for `ReLU / L`—that retains the usual O(L^2) attention behavior and accuracy while reducing synchronization costs and enabling new parallelization opportunities (Figure 1 caption; Section “ReLU‑attention”).

## 3. Technical Approach
Step‑by‑step view of how the modified attention works:

- Baseline attention (Equation 1)
  - For each query vector `q_i` and all key vectors `k_j`, compute scaled dot products `q_i^T k_j / sqrt(d)`, where `d` is the head dimension.
  - Apply a transformation `φ` across the `j` dimension to get attention weights `α_ij`.
  - Compute the output for position `i` as a weighted sum of value vectors: `o_i = Σ_j α_ij v_j`.

- Standard choice of `φ` and its cost
  - In standard Transformers, `φ` is softmax across the sequence positions `j`. Softmax requires:
    - Exponentiation on each score.
    - A sum across `j` to normalize to probability weights (they sum to 1).
  - The cross‑sequence sum makes parallelization harder because it forces synchronization across tokens (Introduction; references [24, 7]).

- Proposed change: point‑wise activation with sequence‑length scaling
  - Replace `φ` with a “point‑wise” function that operates independently on each score and does not sum across `j`.
  - Concrete proposal: `φ = L^{-1} * ReLU` (Section “ReLU‑attention”).
    - “Point‑wise” means apply `ReLU` to each score independently.
    - Scale the result by `1/L`, where `L` is the sequence length (number of tokens in the input).
  - General family tested: `φ = L^{-α} * h`, where:
    - `α` is a non‑negative exponent.
    - `h` is chosen from `{relu, relu^2, gelu, softplus, identity, relu6, sigmoid}` (Section “Scaled point‑wise attention”).
    - Figure 2 sweeps `α` from 0.0 to 2.0 for multiple choices of `h`.

- Why divide by `L`? The scale argument (Section “Sequence length scaling”)
  - With softmax, for any query `i`, the weights across positions sum to one; thus the average weight per position is `E_j[α_ij] = 1/L`.
  - If one naively drops softmax and uses plain `ReLU`, the average weight can become O(1) at initialization because the inputs to `ReLU` are O(1), which makes the output sum across positions scale like O(L). This changes the scale of the attention outputs `o_i` and can destabilize training unless other hyperparameters are retuned.
  - Multiplying by `L^{-1}` restores the expected O(1/L) scale of weights at initialization so the overall scale of `o_i` remains close to the softmax regime without hyperparameter changes. The paper notes this as an empirical justification with a brief analytical motivation (Section “Sequence length scaling”).
  - Note: squared‑ReLU (`relu^2`) is an exception in that it does not preserve O(1) magnitude, hence the benefit of careful scaling (footnote “With the exception of squared ReLU.” under Section 4).

- Practical ablations and design choices
  - `qk‑layernorm`: A variant in which queries and keys are each normalized by LayerNorm before computing dot products (Section 4, “Experimental setup”). This helps with stability at very large scale in prior work; the paper evaluates its effect for the proposed attention (Figure 3).
  - Gated attention unit: Add a gating projection whose output multiplies the attention result element‑wise before the final output projection (as in [15]). The paper tests whether gating removes the need for sequence‑length scaling (Figure 4).

- Why this approach over alternatives
  - It eliminates the sequence‑wise normalization (no sum across `j`) while preserving the scale of outputs, which prior ReLU‑only attempts lacked.
  - It avoids adding complex mechanisms (e.g., kernel tricks of linear attention) and stays very close to the standard attention computation, reducing the need for hyperparameter retuning (Sections 1 and 3).

- Implementation and training setup (Section 4)
  - Codebase: BigVision.
  - Datasets and schedules:
    - ImageNet‑21k pretraining for 30 epochs.
    - ImageNet‑1k training for 300 epochs.
    - Both runs have roughly 9e5 optimization steps.
  - Models: ViT variants S/32, S/16, S/8, plus larger B/32, B/16, L/16 in scaling plots (Figure 1).
  - Reporting:
    - For ImageNet‑21k‑trained models, ImageNet‑1k accuracy is computed by taking the top predicted class among those that exist in 1k, without fine‑tuning (Figure 1 caption).
    - Transfer: 10‑shot linear probes on eight datasets, averaged over three seeds (Figure 1 caption). The datasets are CUB‑200, Caltech‑101, Stanford Cars, CIFAR‑100, DTD, Colorectal Histology, Oxford‑IIIT Pets, and UC Merced (Section 4).

- Computational advantage
  - Because `ReLU / L` is applied independently per score and does not require normalizing over `j`, it “can be parallelized over the sequence length dimension with fewer gather operations than traditional attention” (Figure 1 caption). This is a hardware‑level benefit even though asymptotic complexity remains O(L^2) for full attention.

## 4. Key Insights and Innovations
- Sequence‑length‑aware scaling is the missing piece for softmax‑free attention in ViTs.
  - Novelty: Prior ReLU‑based attention did not divide by `L`. The paper’s `L^{-1}` factor (or more broadly `L^{-α}` with `α≈1`) preserves the O(1/L) average weight scale that softmax implicitly enforces (Section “Sequence length scaling”; Figure 2).
  - Significance: This avoids re‑tuning hyperparameters and stabilizes training, producing accuracy close to softmax.

- Point‑wise activations can match softmax scaling in ViTs when properly scaled.
  - Novelty: Show that `ReLU / L` can “approach or match” the compute–accuracy scaling of softmax across model sizes from small to large (Figure 1).
  - Significance: Retains the empirical benefits of softmax while simplifying the operation (no exponent, no sequence sum).

- The best α is consistently near 1 across models and datasets.
  - Novelty: A systematic sweep over `α ∈ [0, 2]` and over several `h` confirms that `α≈1` is optimal in practice (Figure 2).
  - Significance: Provides a simple rule‑of‑thumb for implementation: set `α=1` and choose a fast point‑wise `h` such as ReLU.

- Removing sequence‑wise normalization remains beneficial even with gating or without qk‑layernorm.
  - Novelty: Ablations show (i) qk‑layernorm is not critical at the tested scales (Figure 3), and (ii) adding a gated attention unit does not obviate the need for `L^{-α}` scaling; best results still occur near `α=1` (Figure 4).
  - Significance: The core idea—`L`‑scaled point‑wise attention—is robust to common architectural variations.

These are incremental in mechanism but fundamental in implication: they demonstrate that softmax is not uniquely necessary for effective ViT attention if one preserves the correct scaling.

## 5. Experimental Analysis
- Evaluation methodology
  - Datasets:
    - ImageNet‑21k (pretraining; 30 epochs).
    - ImageNet‑1k (300‑epoch training; and used for evaluation of 21k‑trained models by restricting predictions to 1k classes; Section 4).
    - Eight transfer datasets for 10‑shot linear probing (listed in Section 4).
  - Models: ViT S/32, S/16, S/8 for most ablations; scaling plots also include B/32, B/16, and L/16 (Figure 1).
  - Metrics:
    - ImageNet‑1k top‑1 accuracy (y‑axis in multiple figures).
    - Average 10‑shot linear probe accuracy across eight datasets (Figure 1 right).
  - Compute accounting: x‑axis in Figure 1 reports TPU core hours.

- Main quantitative findings
  - Softmax vs `ReLU / L` scaling (Figure 1):
    - On both ImageNet‑1k accuracy and average 10‑shot transfer accuracy, the curves for `ReLU / L` track the softmax curves closely across small to large ViTs. The two lines nearly overlap, indicating no significant loss in accuracy at a given compute budget.
    - Practical note: Because `ReLU / L` avoids sequence‑wise normalization, it can be parallelized over tokens with fewer gather operations (Figure 1 caption), offering potential runtime benefits not captured by the accuracy plots.
  - Effect of `α` and choice of `h` (Figure 2):
    - Across S/32, S/16, and S/8 models trained on ImageNet‑21k and ImageNet‑1k, the best accuracy typically occurs for `α≈1`.
    - No single activation `h` dominates at `α≈1` (ReLU, GELU, softplus, etc. perform similarly), so the paper uses ReLU for speed (Figure 2 caption).
  - Effect of `qk‑layernorm` (Figure 3):
    - Using or removing qk‑layernorm has only a small effect on accuracy for S/32, S/16, S/8 with `L^{-α}`‑scaled ReLU or squared‑ReLU attention.
    - This suggests the proposed scaling is not dependent on this normalization at the tested scales.
  - Effect of gating (Figure 4):
    - Adding a gated attention unit does not eliminate the need for `L^{-α}` scaling; best results still cluster near `α≈1`.
    - Gating increases compute by roughly 9.3% for the S/8 model with ReLU (Section “Effect of adding a gate”), with no clear accuracy advantage relative to simply using `α≈1` without the gate.

- Do the experiments support the claims?
  - The evidence is consistent and multi‑faceted: scaling plots (Figure 1), α‑sweeps across datasets and models (Figure 2), and ablations on qk‑layernorm and gating (Figures 3–4).
  - The work does not present explicit wall‑clock benchmarks, but it argues for hardware advantages qualitatively and via reduced synchronization requirements.

- Notable details and conditions
  - Training follows unmodified BigVision defaults (Section 4). This helps establish that `ReLU / L` works without retuning.
  - For ImageNet‑21k pretraining, ImageNet‑1k accuracy is computed by picking the top class among the overlapping 1k classes, without fine‑tuning (Figure 1 caption). This is a conservative evaluation protocol.

## 6. Limitations and Trade-offs
- Theoretical understanding is partial.
  - The paper provides an initialization‑scale argument for why dividing by `L` helps, but it does not offer a full theory of optimization dynamics or generalization under the new attention (Section 5 “Conclusion”: “we are unsure why the factor L^{-1} improves performance or if this term could be learned”).

- Applicability and scope
  - Experiments are focused on ViTs for image classification and 10‑shot linear transfer. There are no results for language modeling, detection/segmentation, or very long sequences.
  - The method preserves O(L^2) attention complexity; it is not a linear‑time attention method. Its advantage is fewer cross‑sequence synchronizations, not a change in asymptotic cost (footnote 1 and Figure 1 caption).

- Numerical and stability considerations
  - Weights no longer sum to one, and normalization is not enforced. While `L^{-1}` keeps the expected scale similar to softmax at initialization, the behavior later in training depends on activations and data. The paper reports good results but does not analyze worst‑case saturation or gradient issues.
  - Squared‑ReLU can change magnitude more aggressively; proper scaling is even more important (footnote in Section 4).

- Engineering trade‑offs
  - The paper qualitatively argues for speedups from reduced gathers but does not provide wall‑clock or throughput numbers; actual runtime gains may depend on implementation details and hardware.
  - Adding gating increases compute by 9.3% for S/8 with ReLU (Section “Effect of adding a gate”), with limited benefit, so simplicity may be preferable.

## 7. Implications and Future Directions
- How this changes the field
  - It challenges the assumption that softmax is essential for attention in ViTs. With a simple `L`‑aware scale, point‑wise activations can be competitive.
  - It encourages implementations and kernels that exploit per‑token parallelism without cross‑sequence normalization, potentially simplifying high‑performance attention kernels.

- Follow‑up research enabled or suggested
  - Learnable scaling: Make `α` (or the `1/L` factor) learnable, per head or per layer, and study stability and performance (Section 5 “Conclusion”).
  - Broader activations: Search for alternative `h` that may outperform ReLU while retaining speed, especially dynamic or data‑dependent activations.
  - Beyond vision: Test on language models, multimodal Transformers, and very long sequences to see whether the `L^{-α}` rule generalizes.
  - Theoretical analysis: Develop a deeper understanding of optimization and generalization when attention weights are not normalized to sum to one.
  - Systems work: Benchmark wall‑clock speed and memory on modern kernels (e.g., FlashAttention‑style implementations) to quantify real runtime benefits.

- Practical applications
  - Large‑scale pretraining of ViTs where hardware synchronization costs dominate; the method can reduce sequence‑axis gathers and may improve throughput.
  - Edge or low‑power deployments where eliminating exponentials and sequence normalization simplifies inference kernels.

Quoted touchpoints for quick reference:
- Equation 1 (Section 3): defines attention weights via `φ` applied to scaled dot products.
- Section “ReLU‑attention” and “Scaled point‑wise attention”: define `φ = L^{-1} ReLU` and the general `L^{-α} h` family.
- Section “Sequence length scaling”: motivation for dividing by `L` to preserve expected O(1/L) weight scale.
- Figure 1: `ReLU / L` matches softmax scaling on ImageNet‑1k accuracy and average 10‑shot transfer vs TPU core hours; notes parallelization advantage.
- Figure 2: best accuracy generally at `α ≈ 1` across models and datasets; no clear winner among activations at that point.
- Figure 3: small effect from removing `qk‑layernorm` at these scales.
- Figure 4: gating does not remove the need for `L^{-α}`; gating increases compute by ~9.3% for S/8 with ReLU.
