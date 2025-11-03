# Muon Outperforms Adam in Tail-End Associative Memory Learning

**ArXiv:** [2509.26030](https://arxiv.org/abs/2509.26030)

## 🎯 Pitch

This paper demystifies why the Muon optimizer surpasses Adam in training transformers by pinpointing its primary advantage to the associative memory components—specifically, the attention value/output matrices and feed-forward networks. Through targeted ablation, spectral analysis, heavy-tailed learning tasks, and theory, the authors show that Muon updates yield more balanced and isotropic representations, enabling more effective learning of rare 'tail' knowledge and reducing biases in large language models. This insight bridges a critical gap in optimizer understanding and paves the way for fairer, stronger, and more reliable AI systems.

---

## 1. Executive Summary (2–3 sentences)
This paper explains why the Muon optimizer trains transformers faster and better than Adam by showing that Muon’s advantage concentrates in the model’s “associative memory” parameters—the attention Value/Output matrices (`W_V`, `W_O`) and the Feed-Forward Network (FFN). Through targeted ablations, spectral analyses, a heavy-tailed knowledge task, and a tractable theory model, it demonstrates that Muon produces more isotropic (balanced) updates that learn rare “tail” facts far more effectively than Adam (Figures 1–3; Theorems 5.3–5.4).

## 2. Context and Motivation
- Problem addressed
  - Large Language Models (LLMs) are trained on heavy‑tailed corpora where some patterns/facts (“head”) appear very frequently while many others (“tail”) are rare. The widely used Adam optimizer performs well overall but often struggles to learn tail patterns uniformly.
  - Muon—a recent matrix optimizer that normalizes gradients using spectral structure—has shown consistent empirical speedups relative to Adam in LLM training, but the mechanism of its advantage has remained unclear.
- Importance
  - Practical: Better tail learning means fairer, more comprehensive knowledge recall and fewer blind spots in LLMs, which matters for safety, coverage, and product reliability.
  - Theoretical: Clarifying why a matrix‑norm optimizer outperforms a vector‑norm optimizer in transformers sharpens our understanding of optimization–architecture interplay.
- Prior approaches and gaps
  - Muon has been interpreted as steepest descent under the spectral norm, while Adam corresponds to steepest descent under a vector infinity norm (Appendix A). These norm-based views alone do not explain why Muon should be better for transformers (Section 1).
  - Past empirical studies aggregate parameters or focus on other architectures (e.g., MoE), obscuring where Muon helps most in dense transformers (Section 4.2; contrast with Liu et al., 2025).
- Positioning
  - This paper links Muon’s update rule to the outer‑product structure of transformer associative memories. It shows empirically and theoretically that Muon equalizes learning across singular directions and thus mitigates head‑tail imbalance specifically in the VO attention and FFN components (Sections 4–5).

## 3. Technical Approach
Step‑by‑step, the paper combines component‑wise optimizer ablations, spectral diagnostics, a heavy‑tailed QA task, and a one‑layer theory model.

- Background: What is Muon and how it updates matrices?
  - Muon keeps a momentum accumulator `B_t = μ B_{t−1} + G_t` for matrix gradient `G_t = ∇_W L(W_t)`. It factors `B_t` via SVD `B_t = U_t S_t V_t^T`, forms the orthogonal factor `O_t = U_t V_t^T`, and updates `W_{t+1} = W_t − η_t O_t` (Section 3). In practice, `O_t` can be approximated via a few Newton–Schulz iterations to avoid a full SVD.
  - Intuition: the SVD decomposes gradient information into orthogonal “directions.” By using `U V^T`, Muon normalizes away singular value magnitudes and pushes equally along all orthogonal directions—i.e., a “spectrally normalized” step.
- Associative memory view of transformers
  - VO attention and FFN can be modeled as linear associative memories that store facts as outer products: for key–value pairs `(e_s, e_o)`, the memory matrix is `W = Σ_i e_{o_i} e_{s_i}^T` (Section 3).
  - In attention, `W_O` (and symmetrically `W_V`) encodes associations (Bietti et al., 2023). In FFN, the output matrix `W_out` is a recognized knowledge store (Geva et al., 2020; Section 3).
  - Connection to Muon: if the gradient on such a memory is itself a sum of outer products, Muon’s `U V^T = Σ_i u_i v_i^T` assigns equal magnitude updates to the orthogonal singular directions corresponding to these “stored facts.”
- Component‑wise optimizer ablations (Section 4.1; Figure 1; Table 1)
  - Two protocols:
    - Independent Blocks: apply Muon to just one component (e.g., `W_QK` or `W_VO` or FFN `W_in/W_out/W_gate`) while keeping the rest on Adam.
    - Combined Configurations: apply Muon to combinations suggested by the first stage (e.g., VO+FFN) to see how much of “full Muon” can be recovered.
  - Models: NanoGPT‑style 160M parameter transformer with both ungated and gated FFN variants; trained on FineWeb (Section 4.1; Appendix B.1).
- Spectral diagnostics (Section 4.2; Figure 2)
  - Define isotropy metrics for a weight’s singular values `σ`: normalized SVD entropy, effective rank (`eRank`), Top‑k energy fraction, and Q75/25 eigenvalue ratio (Section 4.2).
  - Measure these over training for VO and FFN matrices under Muon vs Adam.
- Heavy‑tailed QA task (Section 4.3; Figure 3; Appendix B.2)
  - Build a synthetic biographical QA dataset with a power‑law distribution of per‑entity samples (Figure 3a). Metric is First Token Accuracy (FTA): whether the first generated token of the answer matches the ground truth.
  - Compare Muon, Adam, and SGD+Momentum; also hybrid configs where Muon is applied only to VO+FFN or only to QK.
- Theory model (Section 5; Theorems 5.3–5.4)
  - A one‑layer associative memory with softmax outputs. Assumptions:
    - Orthonormal embeddings for keys and values (`E^T E = Ē^T Ē = I`; Assumption 5.1), empirically supported by near‑90° angles between embedding directions in Llama‑3 FFNs (Figure 4a; Appendix B.3, C.6).
    - Two‑mass class imbalance: first `L` facts share probability mass `α`, the rest `K − L` share `1−α` (Assumption 5.2).
  - Compare one‑step and multi‑step updates of GD, Adam‑without‑EMA (reduces to elementwise SignGD), and Muon; analyze balance across facts via the minimum correct‑class probability once at least one class reaches ≥ `1−ε` (Eq. 5.1 and surrounding text).

Analogy for clarity: think of the memory matrix as a pinboard storing many sticky notes (facts). Adam’s sign step may press harder on regions where notes overlap in particular ways (depending on embedding supports), potentially ignoring some notes. Muon presses evenly across all principal directions of the gradient pinboard, so none of the well‑aligned notes crowd out the rest.

## 4. Key Insights and Innovations
- Associative memories are where Muon delivers its gains
  - Evidence: When Muon is applied only to VO+FFN, it almost matches full‑model Muon, whereas applying Muon only to QK yields much smaller gains (Figure 1c–d; Table 1). This isolates the locus of benefit to the memory‑like VO and FFN blocks (Observation 1, Section 4.1).
  - Significance: It points to a practical recipe—use Muon where the model stores and retrieves facts.
- Muon induces more isotropic singular spectra—consistently and stably
  - Across seeds and during training, Muon yields higher SVD entropy and effective rank, lower Top‑10 energy concentration, and lower Q75/25 ratios on VO and `W_out` than Adam (Figure 2a–d; Observation 2). Error bars are small for Muon but large for Adam, indicating stability.
  - Significance: Isotropic spectra reflect balanced use of directions/features, which the theory later connects to balanced learning across classes (Theorem 5.3).
- Muon learns tail knowledge faster and more uniformly than Adam
  - On a power‑law QA task, Muon rapidly reaches near‑perfect FTA on head groups and substantially improves tail groups compared with Adam and SGD (Figure 3b–d). For example, at 10k steps on the most extreme tail group (g=15), Muon reaches 0.976 ± 0.006 FTA vs Adam 0.264 ± 0.048 (Appendix C.4, Table 5).
  - Hybrid ablations show this tail advantage comes from applying Muon to VO+FFN, not QK (Figure 3e–f; Appendix C.5 for gated FFN).
- Theory: Muon is provably balanced across classes regardless of embeddings
  - One‑step and multi‑step analyses show Muon achieves near‑equal correct probabilities across classes once any class reaches `≥ 1−ε` (Theorems 5.3–5.4):
    > ρ^ε_Muon ≥ 1 − ε (1 + O((log K)/K)) (Theorems 5.3 & 5.4)
  - In contrast, GD becomes highly imbalanced under class imbalance, and Adam’s SignGD can be either balanced or very imbalanced depending on embedding overlap; in a constructed case its smallest singular value is ≤ 25% of the largest and the worst class accuracy scales poorly:
    > ρ^ε_SignGD = O(ε^−0.7 K^−0.3), with σ_min/σ_max ≤ 25% (Theorem 5.3)

Together, these are more than incremental improvements: they connect an optimizer’s matrix‑norm update to the architecture’s memory structure and to tail‑robust learning dynamics.

## 5. Experimental Analysis
- Setup and methodology
  - Models and data
    - 160M NanoGPT on FineWeb for optimizer ablations and spectra; both ungated and gated FFN (Section 4.1; Figure 1).
    - Synthetic heavy‑tailed QA task built from 200k+ biographical records; sample counts per class follow a power law (Figure 3a; Appendix B.2). Metric: FTA.
    - Scaling check on a 0.7B NanoGPT variant (Appendix C.2; Figures 5–7).
  - Baselines and configurations
    - Full Muon on all attention and FFN parameters vs All‑Adam (Figure 1).
    - Hybrid configurations: Muon only on QK; only on VO; only on FFN submatrices (`W_in`, `W_out`, `W_gate`); combinations such as VO+FFN (Figure 1; Table 1).
    - For QA: add SGD+Momentum and two hybrids (Muon on VO+FFN vs Muon on QK) (Figure 3b–f).
  - Controls and diagnostics
    - Spectral metrics for isotropy (Section 4.2; Figure 2; Appendix C.3).
    - Check for attention “MaxLogit explosion” to ensure QK findings aren’t confounded (Appendix C.1): no explosion observed with RMSNorm on Q and K in this 160M setting.
- Main quantitative results
  - Where does Muon help?
    > “Muon(VO Attn, FFN) & Adam(QK Attn)” nearly matches full Muon: 3.5858 (non‑gated) vs full Muon 3.5654; 3.5312 (gated) vs full Muon 3.5125 (Table 1; Figure 1c–d).  
    > “Muon(QK Attn) & Adam(VO, FFN)” much weaker: 3.8925 (non‑gated); 3.8518 (gated) (Table 1).
    - Among FFN pieces, `W_out` is especially influential (Table 1; Section 4.1).
  - Spectral isotropy
    > Muon shows higher normalized SVD entropy and `eRank`, and lower Top‑10 energy and Q75/25 ratios than Adam for VO and `W_out` throughout training and across seeds (Figure 2a–d). Adam’s curves fluctuate and have larger error bars (Observation 2).
  - Heavy‑tailed QA
    > At 10k steps, on the extreme tail (group 15), FTA: Muon 0.976 ± 0.006 vs Adam 0.264 ± 0.048; on a mid‑tail group (g=13), Muon reaches 1.000 ± 0.000 vs Adam 0.890 ± 0.042 (Appendix C.4, Table 5).  
    > Hybrid “Muon(VO,FFN)&Adam(QK)” largely tracks full Muon, while “Muon(QK)&Adam(VO,FFN)” lags on tail groups (Figure 3e–f; Appendix C.5 mirrors this for gated FFN).
  - Scaling to 0.7B
    > Patterns replicate: VO+FFN nearly recovers full Muon’s validation loss; QK‑only gives little benefit. Muon increases SVD entropy and `eRank` for VO and FFN at scale (Figures 5–7).
- Do the experiments support the claims?
  - The ablations (Figure 1, Table 1) convincingly localize Muon’s gains to associative‑memory blocks (VO and FFN). The spectral analyses (Figure 2) align with the balance hypothesis. The heavy‑tailed QA (Figure 3, Appendix C.4–C.5) directly tests the tail‑learning claim and provides large, robust margins over Adam on tail groups. The MaxLogit control (Appendix C.1) reduces concerns that QK results stem from pathological attention peaking. Orthogonality measurements (Figure 4a; Appendix C.6) support the theory model’s assumptions.
- Notable ablations and robustness checks
  - Independent vs combined configurations; VO vs O‑only vs V‑only; FFN sub‑matrices; gated vs ungated FFN; scaling to 0.7B; random‑seed stability in spectral metrics; verification of no logit explosion.

## 6. Limitations and Trade-offs
- Theoretical simplifications
  - Adam is analyzed without EMAs (β1=β2=0), reducing it to SignGD (Section 5). While standard in some analyses, it omits EMA effects that may matter in practice.
  - The model assumes orthonormal embeddings and a two‑mass imbalance (Assumptions 5.1–5.2). Although angles near 90° are observed in FFN embeddings (Figure 4a), real LLM embeddings are not exactly orthonormal nor two‑mass.
  - One‑layer linear associative memory abstraction does not capture full transformer nonlinearities or depth, though it reflects the additive nature of memory outputs (Section 5).
- Empirical scope
  - Main training results are on 160M and 0.7B models with 10k steps (Sections 4.1, C.2). Full‑scale pretraining and downstream task breadth are not covered here.
  - The knowledge task is synthetic (Appendix B.2). While it isolates heavy‑tail behavior, generalization to messy, real‑world knowledge distributions is not directly measured.
- Compute and implementation
  - Muon requires matrix orthogonalization per update (SVD or Newton–Schulz), which adds overhead compared with elementwise Adam. The paper does not report wall‑clock comparisons here; external works cited elsewhere have measured speedups, but those are outside this paper’s experiments.
- Component nuance
  - QK receives little benefit from Muon here (Figure 1), but the paper primarily rules out “logit explosion” as the culprit in this 160M setting (Appendix C.1). Other attention pathologies at larger scales or different normalizations remain possible.
- Hyperparameter parity
  - In combined configs (e.g., VO+FFN) the learning rate is kept the same as full Muon “without further tuning,” which “could likely be reduced” to close the small remaining gap (Section 4.1). This suggests some residual sensitivity.

## 7. Implications and Future Directions
- How this changes the landscape
  - It reframes Muon’s edge as architectural: matrix‑norm, SVD‑based updates match the outer‑product structure of transformer memories. This motivates component‑aware optimizer design rather than one‑size‑fits‑all.
  - For practitioners: a strong default is hybrid optimization—use Muon for VO and FFN, Adam for QK and embeddings—capturing most gains at likely lower overhead (Section 4.1, Figure 1c–d; Table 1).
- Follow‑up research it enables
  - Optimizer design
    - Extend Muon‑like ideas to higher‑order tensor memories (Section 6) and to other memory‑bearing components (e.g., MoE experts, KV caches).
    - Adaptive variants (e.g., Adamuon, PolarGrad) targeted to associative memory spectra; schedules that switch optimizers per component or per training phase.
  - Theory
    - Beyond one‑layer to multi‑layer nonlinear settings; incorporate EMA effects for Adam; relax orthogonality assumptions and analyze richer heavy‑tail distributions.
    - Study generalization: how isotropic spectra in memory weights relate to factual robustness and editing stability.
  - Evaluation
    - Real‑world heavy‑tail benchmarks (Wikipedia entities, long‑tail slot filling), beyond synthetic QA; measure recall fairness across entities and attributes.
- Practical applications
  - Pretraining/fine‑tuning regimes where tail coverage matters: enterprise knowledge bases, legal/medical recall, safety‑critical FAQ systems.
  - Knowledge editing and maintenance: isotropic memory spectra may yield more predictable edits in `W_out` and `W_O` (Section 3, related work).
  - Fairness audits: spectral isotropy metrics (Section 4.2) as training diagnostics for balanced learning across rare classes.

Overall, the paper provides a cohesive empirical‑theoretical narrative: Muon’s spectral normalization aligns with the outer‑product structure of transformer memories, producing isotropic updates that improve tail learning. The ablations (Figure 1), spectra (Figure 2), heavy‑tail QA (Figure 3; Appendix C.4–C.5), and theory (Theorems 5.3–5.4) collectively make the case and offer a concrete, actionable recipe for deploying Muon where it counts most.
