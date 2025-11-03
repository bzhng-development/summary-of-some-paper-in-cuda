# Muon Outperforms Adam in Tail-End Associative Memory Learning

**ArXiv:** [2509.26030](https://arxiv.org/abs/2509.26030)

## 🎯 Pitch

This paper reveals that the Muon optimizer’s core advantage over Adam stems from its alignment with the associative memory structure of transformer models. By leveraging matrix-norm–based updates, Muon enables more balanced and isotropic learning of both common and rare (tail) knowledge, leading to markedly improved performance on heavy-tailed, real-world data. These insights not only demystify Muon's empirical success but also highlight where and why optimizer choice is critical for robust, equitable knowledge acquisition in large language models.

---

## 1. Executive Summary
This paper explains why the Muon optimizer trains transformers faster and better than Adam by tying Muon’s update rule to how transformers store knowledge as “associative memories.” Through targeted ablations, spectrum analyses, a heavy‑tailed knowledge task, and a stylized theory, it shows that Muon’s matrix‑norm update aligns with the outer‑product structure of these memories, yielding more isotropic (evenly distributed) learning and markedly better performance on rare (“tail”) facts.

## 2. Context and Motivation
- Problem addressed
  - Many groups report that Muon trains Large Language Models (LLMs) faster than Adam, but the mechanism remains unclear. Prior interpretations view Muon as steepest descent under a matrix operator norm, yet this alone does not explain the consistent empirical advantage over Adam (Introduction; discussion around steepest descent in Appendix A).
- Why this matters
  - Practical: Choosing optimizers and where to apply them is crucial for training efficiency and generalization in large models.
  - Scientific: Understanding how an optimizer interacts with internal “memory” structures (like Feed‑Forward Networks and attention projections) clarifies what LLMs learn and how they learn it.
- Prior approaches and limitations
  - Adam is well‑studied (Related Works, §2), with convergence analyses and feature‑learning perspectives. But none explain why Muon, which normalizes matrix gradients by their spectral structure, beats Adam inside transformer layers.
  - Existing Muon analyses focus on optimization geometry (steepest descent in spectral norm, operator‑norm constraints) and convergence (Related Works, §2), but not on how Muon interacts with the specific memory‑storing parts of transformers.
- Positioning of this paper
  - The paper reframes the question through the lens of associative memory in transformers (Preliminaries, §3): Value–Output (VO) attention weights and Feed‑Forward Networks (FFNs) behave like linear associative memories that store facts as sums of outer products. The core thesis is that Muon’s update (orthogonal, spectrum‑normalized) perfectly matches this outer‑product structure, balancing learning across frequent and rare facts.

## 3. Technical Approach
The paper builds a multi‑part argument combining implementation details, empirical tests, and theory.

- What is Muon?
  - For a matrix parameter W with gradient G, Muon maintains a momentum `B_t = μ B_{t-1} + G_t` and updates using the orthogonal factor of B_t: compute SVD `B_t = U_t S_t V_t^T`, then set `O_t = U_t V_t^T` and update `W_{t+1} = W_t − η_t O_t` (Preliminaries, §3). In practice it approximates `O_t` with Newton–Schulz iterations to avoid full SVD (Practical note in §3).
  - Intuition: Muon normalizes away singular values and keeps only the orthogonal “directions,” so each singular direction contributes equally to the update. In steepest‑descent terms, it performs descent under the matrix operator (spectral) norm (Appendix A).
- Associative memory view of transformer components
  - The paper treats `WO`, `WV` (the output and value projections in attention) and FFN weights (`W_in`, `W_out`, optional `W_gate`) as “associative memory parameters.” A linear associative memory stores facts as `W = Σ_i e_o,i e_s,i^T` (outer products of “value” and “key” embeddings) so that `e_o = W e_s` (Preliminaries, §3; references to Geva et al., Bietti et al., etc.).
  - Key observation: If gradients for such memories decompose into outer products, Muon’s `O = U V^T = Σ_i u_i v_i^T` updates each orthogonal component equally, counteracting imbalances in the singular values `S` that often encode frequency skew (Main Results, §4.1 end).
- Two‑stage ablation design to locate where Muon helps most (Main Results, §4.1; Fig. 1; Table 1)
  1) Independent Blocks: apply Muon to one block at a time (Q, K, V, O, and FFN parts), keep others on Adam.  
  2) Combined Configurations: combine the most promising blocks (e.g., VO+FFN) under Muon, leave the rest on Adam, and compare to “full Muon.”
  - Models and data: 160M NanoGPT on FineWeb; both non‑gated and gated FFN variants (details in §4.1 and Appendix B.1).
- Spectral analysis to test the isotropy hypothesis (Main Results, §4.2; Fig. 2)
  - Define four spectrum metrics for each weight matrix W with singular values σ:
    - Normalized SVD entropy (uniformity of energy),  
    - Effective rank (entropy perplexity),  
    - Top‑k energy fraction (energy concentration),  
    - Q75/Q25 eigenvalue ratio (spread robust to outliers) (Definitions in §4.2).
  - Compare Muon vs Adam over training and across seeds to assess isotropy and stability.
- Heavy‑tailed knowledge task to test tail learning (Main Results, §4.3; Fig. 3; Appendix B.2, C.4, C.5)
  - Synthetic QA over ~32,768 “classes” (individuals), with a power‑law frequency of training samples per class (Fig. 3a; parameter m=15; 6 QA pairs per selection).
  - Metric: First Token Accuracy (FTA) on answers. Optimizers: Muon, Adam, and SGD+Momentum. Also, hybrids: Muon on VO+FFN but Adam on QK, and vice versa (Fig. 3e–f).
- Theory on a one‑layer linear associative memory (Case Study, §5; Theorems 5.3–5.4)
  - Setup: K triplets (subject–relation–object) with orthonormal key/value embeddings `E` and `Ē` (`E^T E = Ē^T Ē = I`, Assumption 5.1), class imbalance split across two frequency groups (Assumption 5.2). Initialize `W_0 = 0`.
  - Compare one‑step (and multi‑step) updates for three optimizers:
    - GD (`W_{t+1} = W_t − η ∇W L(W_t)`),
    - Adam without moving averages, which reduces to `SignGD` (element‑wise sign of gradient) for analysis (Preliminaries §5, “Adam → SignGD”),
    - Muon without momentum (`W_{t+1} = W_t − η U norm(Σ) V^T`).
  - Main questions: With at least one class achieving high probability (≥ 1 − ε), how low can the worst class probability be under each optimizer? How isotropic are the updates?

## 4. Key Insights and Innovations
- Insight 1 — Where Muon actually helps in transformers
  - Claim: Applying Muon to VO and FFN yields almost all the gain; applying it to QK gives little benefit.
  - Evidence: In 160M NanoGPT (non‑gated and gated FFN), the “Independent Blocks” experiment shows larger validation‑loss gains for `WV`/`WO` and FFN than for `WQ`/`WK` (Fig. 1a–b; Table 1). In “Combined Configurations,” Muon on VO+FFN nearly recovers the full‑Muon curve (Fig. 1c–d; Table 1).  
  - Quote:
    > Observation 1: Muon is most effective when applied to VO and FFN; in particular, applying Muon to only VO+FFN almost recovers the full‑Muon trajectory. (end of §4.1)
- Insight 2 — Muon induces more isotropic weight spectra, stably
  - Difference from prior work: The analysis focuses on associative‑memory parameters (VO, FFN) rather than aggregating all weights. Isotropy is tied to balanced learning of facts.
  - Evidence: Across training and random seeds, Muon shows higher normalized SVD entropy and effective rank, lower Top‑k energy and Q75/Q25 ratios than Adam for VO and `W_out` (Fig. 2a–d). Error bars are small for Muon and large for Adam, indicating stability (Main Results, §4.2).
  - Quote:
    > Observation 2: Muon consistently yields more isotropic weight matrices … throughout training and across random initializations. (§4.2)
- Insight 3 — Muon learns tail facts better on heavy‑tailed data
  - Result: Muon matches Adam on head classes and exceeds it on tail classes, reducing the head–tail gap and accelerating convergence (Fig. 3b–d; Appendix C.4–C.5 tables). The VO+FFN hybrid reproduces most of this effect; QK‑only does not (Fig. 3e–f).
  - Quote:
    > Observation 3: In heavy‑tailed, knowledge‑intensive tasks, Muon … substantially improving learning on tail classes. (§4.3)
- Insight 4 — Theory: Muon’s balanced learning is intrinsic to its update rule
  - Theorem 5.3 (one‑step) and Theorem 5.4 (multi‑step) show Muon achieves nearly equal correct‑class probabilities across items once any item is near‑correct, regardless of embeddings (orthonormal), whereas GD and SignGD (Adam without EMA) can be highly imbalanced.  
  - Mechanism: Muon’s update is the (almost) uniform orthogonal factor of the gradient (`U V^T`), so its singular values are nearly equal—matching the isotropy seen empirically (see the gradient form and SVD reasoning in §5.2 and Appendix D/E).

## 5. Experimental Analysis
- Evaluation methodology
  - Where to apply Muon (ablation on transformer components):
    - 160M NanoGPT on FineWeb; both non‑gated and gated FFN. Two stages: Independent Blocks and Combined Configurations (Main Results, §4.1; Fig. 1; Table 1).
  - Spectrum analysis:
    - Track four isotropy metrics over training steps and random seeds for VO and FFN parameters (Main Results, §4.2; Fig. 2). Repeated at 0.7B scale with consistent trends (Appendix C.2; Figs. 6–7).
  - Heavy‑tail knowledge task:
    - Synthetic QA dataset with power‑law class frequencies (Fig. 3a; Appendix B.2). Measure First Token Accuracy (FTA). Compare Muon, Adam, SGD+Momentum, and hybrids (Fig. 3b–f; Appendix C.4–C.5 tables).
  - Additional checks:
    - Logit‑explosion control for attention via RMSNorm applied to Q/K; no instability is observed (Appendix C.1, Table 2).
    - Orthogonality plausibility of embeddings in real models: average angles between FFN embeddings are near 90° in Llama‑3‑8B‑Instruct (§5.1; Fig. 4a; Appendix B.3, C.6).
- Main quantitative results
  - Ablations (160M, 10k steps; Table 1):
    - Full Muon lowers validation loss to 3.565 (non‑gated) vs Adam 3.924.
    - Muon only on VO+FFN reaches 3.586 (non‑gated) and 3.531 (gated), “nearly recovering” full Muon (Fig. 1c–d).
    - Muon on QK only: much weaker (3.893 non‑gated; 3.852 gated).
    - Within FFN, `W_out` benefits strongly; in ungated FFN, VO+`W_out` is close to full Muon (Fig. 1c).
  - Spectral isotropy (Fig. 2):
    - For VO and `W_out`, Muon shows higher SVD entropy and effective rank, lower Top‑10 energy and Q75/Q25 ratio across 10k steps and across seeds. Adam’s curves fluctuate more with seed choice.
  - Heavy‑tail learning (Fig. 3; tables in Appendix C.4–C.5):
    - At 10k steps (non‑gated FFN, tail group 15): Muon FTA 0.976 ± 0.006 vs Adam 0.264 ± 0.048; VO+FFN hybrid: 0.954 ± 0.021; QK‑only hybrid: 0.286 ± 0.039 (Table 5).
    - Head groups: all optimizers reach ~1.0 FTA by 10k steps (Fig. 3b–d).
    - Trends replicate with gated FFN (Appendix C.5, Tables 6–8).
  - Scaling to 0.7B:
    - Muon > Adam in validation loss; VO+FFN hybrid ~ full Muon; QK‑only weak (Fig. 5). Spectral isotropy advantages persist (Figs. 6–7).
- Do the experiments support the claims?
  - Yes, in four ways that interlock:
    1) Targeted ablations localize Muon’s value to associative‑memory parameters (VO, FFN) (Fig. 1; Table 1).
    2) Spectrum metrics show Muon equalizes singular values—consistent with the associative memory hypothesis (Fig. 2).
    3) Heavy‑tail QA shows Muon’s balanced learning translates to tail‑class gains (Fig. 3; tables).
    4) Theory matches both isotropy and tail balance (Theorems 5.3–5.4).
- Notes on robustness and edge cases
  - Adam’s instability across seeds is visible in spectrum metrics (Fig. 2). Muon is stable.
  - The “MaxLogit explosion” is not a confound in this setup (Appendix C.1).
  - Similar findings appear at larger scale (0.7B) and with gated FFNs (Appendix C.2, C.5).

## 6. Limitations and Trade-offs
- Theoretical assumptions simplify reality
  - Orthonormal embeddings (`E^T E = Ē^T Ē = I`) and one‑layer linear associative memory (§5.2, Assumption 5.1). This aligns with measured near‑orthogonality (Fig. 4a), but real models are deeper and nonlinear.
  - Two‑group class imbalance (Assumption 5.2). It captures the head–tail effect but is simpler than a true power law. The proof techniques can extend, but the paper shows the two‑group case explicitly.
  - Adam is analyzed in the “no‑EMA” limit as SignGD (§5.2). This isolates the element‑wise normalization mechanism but does not capture the full Adam dynamics with moving averages.
- Scope of empirical evidence
  - Main training curves use 10k steps on 160M and 0.7B NanoGPT; the work does not report end‑to‑end pretraining of very large LLMs nor task‑rich evaluations.
  - Heavy‑tail task is synthetic (biographical QA) by design (Appendix B.2). It convincingly isolates the tail‑learning effect, but broader generalization to diverse knowledge tasks remains to be shown.
- Computational considerations
  - Muon requires computing an orthogonal factor per matrix per step; the paper uses an efficient Newton–Schulz approximation (Preliminaries, §3), but real‑world throughput and memory trade‑offs versus Adam are not benchmarked here.
- Architectural nuance
  - QK weights also become more isotropic under Muon (noted in §4.2), yet this does not translate to clear validation‑loss gains in ablations (Fig. 1a–d). When and how QK benefits behaviorally remains an open question.

## 7. Implications and Future Directions
- How this changes the field’s understanding
  - It reframes optimizer choice as a question of aligning updates with model internals. For transformers, VO and FFN implement associative memories; Muon’s orthogonal updates match these memories’ outer‑product structure, yielding balanced, tail‑friendly learning.
  - Practically, one can deploy Muon selectively: using Muon for VO+FFN while keeping Adam on QK nearly recovers full‑Muon gains (Fig. 1c–d; Fig. 5), simplifying adoption and reducing overhead.
- Follow‑up research enabled
  - Larger‑scale pretraining studies with selective Muon on VO+FFN to quantify wall‑clock and energy savings while tracking tail‑knowledge performance.
  - Extending the theory beyond orthonormal embeddings, multiple layers, and beyond two‑group imbalance; analyzing Muon with momentum and Adam with EMA.
  - Investigating how isotropy interacts with other desiderata (e.g., sparsity, low‑rank structure, retrieval augmentation).
  - Generalizing the “outer‑product alignment” idea to higher‑order tensor memories (Conclusion, §6).
- Practical applications
  - Knowledge‑intensive LLMs in domains with heavy‑tailed data (rare entities, long‑tail relations): Muon on VO+FFN should improve recall of rare facts without sacrificing head performance (Fig. 3; Tables 3–5, 6–8).
  - Knowledge editing and safety: isotropic, balanced memories may yield more predictable edits in `W_out`/`WO` (ties to prior work cited in §2–§3).
  - Model scaling and efficiency: selective Muon can be a drop‑in optimizer policy for specific parameter groups to improve sample efficiency early in training (Fig. 1, Fig. 5).

> Core takeaway: Muon’s orthogonal, spectrum‑normalized updates align with the associative‑memory structure of transformers. This alignment equalizes learning across singular directions, keeps spectra isotropic and stable, and—on heavy‑tailed data—translates into superior tail learning while maintaining strong head performance (Observations 1–3; Theorems 5.3–5.4; Figs. 1–3).
