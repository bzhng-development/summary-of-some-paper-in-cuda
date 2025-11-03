# Auxiliary‑Loss‑Free Load Balancing Strategy for Mixture‑of‑Experts

**ArXiv:** [2408.15664](https://arxiv.org/abs/2408.15664)
**Authors:** Lean Wang, Huazuo Gao, Chenggang Zhao, Xu Sun, Damai Dai
**Institutions:** Not specified in metadata

## 🎯 Pitch

Loss-Free Balancing (LFB) revolutionizes Mixture-of-Experts (MoE) models by evenly distributing expert workloads without auxiliary losses, using a bias adjustment based on load statistics that maintains language model performance. This innovation resolves the imbalance dilemma, enhancing both throughput and expert specialization in large-scale, expert-parallel systems, marking a significant step forward in efficient model training.

---

## 1. Executive Summary
This paper introduces Loss-Free Balancing (LFB), a training-time strategy that keeps Mixture-of-Experts (MoE) models’ experts evenly utilized without adding an auxiliary balancing loss. Instead, it adjusts routing decisions using a bias per expert that is updated from recent load statistics, avoiding extra gradients that can impair language modeling. On 1B- and 3B-parameter MoE models trained on 100B and 200B tokens, LFB both improves perplexity and achieves markedly better load balance than conventional auxiliary-loss methods (Table 2).

## 2. Context and Motivation
- Problem addressed:
  - MoE models use many experts but route only a few per token (“Top-K routing”). If routing is imbalanced, a few experts get overloaded while others are underused, causing “routing collapse” (experts never selected) and wasted compute (Section 1; Figure 2; Section 2.2).
- Why it matters:
  - Practical significance: Imbalance limits throughput, especially with expert parallelism where experts reside on different devices; the slowest device dominates step time.
  - Modeling significance: Collapse prevents many experts from learning, reducing specialization and capacity gains expected from MoE.
- Prior approaches and shortcomings:
  - Auxiliary-loss balancing (Section 2.2; Eq. 2) encourages uniform usage by penalizing deviations. However, this adds “interference gradients” that conflict with the language modeling objective. Figure 2 shows the dilemma: small coefficients α keep performance but allow imbalance; large α balances routing but hurts perplexity.
  - Expert Choice (EC) routing perfectly balances loads by constraining per-step token counts per expert, but it violates the causal constraint in language modeling (future tokens influence current routing), i.e., a form of “future token leakage” (Section 5.2; Figure 6; Appendix D).
- Positioning:
  - LFB aims to break the auxiliary-loss trade-off by controlling balance without adding gradients and without causing leakage. It’s a “control-plane” change to routing decisions rather than a “loss-plane” regularizer.

## 3. Technical Approach
At a high level, LFB modifies how tokens pick experts without modifying the gradients used to train expert parameters.

- Background: MoE and Top-K routing (Section 2.1; Eq. 1)
  - For each token embedding `u_t`, a gating function `G` scores each expert `i` as `s_i,t = G(u_t^T e_i)` where `e_i` is an expert’s centroid vector.
  - The model chooses the Top-K experts per token. The output is a weighted sum over selected experts; weights are the gating scores `g_i,t = s_i,t` for selected experts, and 0 otherwise.
- Core idea: Per-expert bias added before selection (Section 3; Eq. 3; Figure 1)
  - Before Top-K selection, add a scalar bias `b_i` to each expert’s score.
  - Selection uses “biased scores” `s_i,t + b_i`, but the weights applied to the selected experts’ outputs remain the original `s_i,t` (not the biased values). This ensures biases steer only the routing decision and do not alter forward computation’s mixing weights or introduce gradients tied to bias values.
  - Mathematically (Eq. 3): a token routes to experts whose `s_i,t + b_i` fall in the Top-K for that token; the contribution remains `g_i,t = s_i,t` if selected, zero otherwise.
- Bias update rule (Algorithm 1):
  - After each batch:
    - Count how many tokens were routed to each expert: `c_i`.
    - Compute the average load `c̄` across experts and the deviation `e_i = c_i − c̄`.
    - Update each bias with a small step in the direction that discourages overused experts and encourages underused ones: `b_i ← b_i + u * sign(e_i)` where `u` is a learning-rate-like hyperparameter.
  - Crucially, the update uses only past-batch statistics to avoid any per-sequence peeking that would violate causality (Section 3, note following Algorithm 1).
- Why this approach:
  - Avoids “interference gradients”: Because the balance control never appears in the loss, it doesn’t backpropagate. This contrasts with auxiliary-loss schemes (Eq. 2) that necessarily inject gradients unrelated to language modeling into the MoE parameters (Sections 2.2 and 3).
  - Preserves causality: Bias updates depend on completed batches, not on the current sequence’s yet-unseen tokens, unlike EC (Section 5.2; Table 1).
- Design choices and ablations:
  - Update magnitude:
    - A fixed step with `sign(e_i)` works best—stable and effective. Using the raw magnitude `e_i` can slightly improve balance but does not improve perplexity (Table 3).
    - The bias update rate `u` controls convergence vs. oscillation. `u=0.001` is the best compromise in their 1B experiments (Figure 4).
  - Additive vs. multiplicative bias:
    - Additive bias (`s_i,t + b_i`) outperforms multiplicative (`s_i,t * b_i`) on perplexity with similar balance (Table 4).
  - Gating function:
    - Main experiments use a sigmoid gate (Section 4.1). Experiments with a softmax gate require a different update variant (`b_i ← b_i + u * e_i`), and LFB still improves balance with comparable perplexity to the best auxiliary-loss baseline (Appendix C; Table 6; Figure 8).
- Practical example:
  - Think of each expert having a “priority knob” (`b_i`). After each step, experts that were too busy get their knob turned down a notch; underused experts get turned up. The next batch’s Top-K picks then automatically shift toward underused experts. No changes to how expert outputs are combined, and no extra terms in the loss.

## 4. Key Insights and Innovations
- Auxiliary-loss-free balance control (fundamental innovation):
  - LFB steers routing by biasing scores pre-selection without modifying model gradients (Section 3; Eq. 3; Algorithm 1). This directly addresses the core dilemma documented in Figure 2: balance vs. performance.
- Causality-preserving and leakage-free (fundamental):
  - Updates depend only on past batches, avoiding the EC-style lookahead that leaks future-token information (Section 5.2; Figure 6; Appendix D). EC can theoretically leak more than `K * log2((1−R)/R)` bits per layer (Appendix D.1, Eq. 7); for a 9-layer MoE with `N=16` and `K=2` this exceeds 50 bits per token.
- Strong compatibility with expert parallelism (practical innovation):
  - As computation-batch size grows (e.g., from larger expert-parallel groups), LFB’s batch-level balance converges toward its near-perfect global balance, improving efficiency (Section 5.1; Figure 5).
- Simple, robust control law (incremental but practical):
  - A per-expert scalar with a one-line update (`b_i += u * sign(e_i)`) is easy to implement, tunes a single parameter `u`, and shows stable behavior across training (Figure 4; Table 3; Table 4).

## 5. Experimental Analysis
- Evaluation setup (Section 4.1; Appendices A–B):
  - Models: DeepSeekMoE-style MoE Transformers with 1B and 3B parameters (Table 5 lists hyperparameters: 64 routed experts, 6 activated per token, 2 shared experts; 9 MoE layers for 1B, 11 for 3B).
  - Data: Multilingual mixture (web, math, code, literature); BPE tokenizer with 32K vocab.
  - Training:
    - 1B on 100B tokens, cosine scheduler; 3B on 200B tokens, multistep scheduler (Appendix B).
    - Main experiments use a sigmoid gate. Softmax-gate results in Appendix C.
  - Baseline: Auxiliary-loss balancing with α=0.001 for sigmoid (chosen from Figure 2’s trade-off curve) and α=0.0003 for softmax (Appendix C).
  - Metrics:
    - Perplexity on a held-out validation set (~71M tokens; Appendix B).
    - Load balance measured by maximal violation `MaxVio` (Eq. 4): the relative gap between the most-loaded expert and the ideal equal load. Reported both globally (`MaxVio_global` over the whole validation set) and during training (`MaxVio_batch`, smoothed in Figure 3). A computation-batch variant is used for the expert-parallelism analysis (Figure 5).
- Main quantitative results (Table 2; Figure 3):
  - Global balance and perplexity:
    - 1B model:
      > Table 2: LFB perplexity 9.50 vs 9.56 (aux-loss), `MaxVio_global` 0.04 vs 0.72.
    - 3B model:
      > Table 2: LFB perplexity 7.92 vs 7.97 (aux-loss), `MaxVio_global` 0.04 vs 0.52.
  - Training-time balance:
    - Figure 3 shows `MaxVio_batch` throughout training: LFB rapidly reaches and maintains near-zero violations, whereas aux-loss baselines plateau at much higher values.
  - Interpretation:
    - The perplexity gains are modest but consistent; the balance gains are large. This aligns with the motivation that removing interference gradients should improve language modeling while direct control achieves better routing balance.
- Ablations and variants:
  - Update rate `u` (Figure 4):
    - `u=0.001` balances early stability and late-stage smoothness (best PPL 9.50).
    - `u=0.0001`: slow convergence early; `u=0.01`: oscillation later; both slightly worse perplexity (9.51).
  - Update rule magnitude (Table 3):
    - Using magnitude (`b_i += u * e_i`) can slightly improve balance (`MaxVio_global` down to 0.028) but worsens perplexity (9.53 vs 9.50). The sign version is preferred.
  - Multiplicative bias (Table 4):
    - Similar balance but worse perplexity than additive bias at comparable `u`.
  - Softmax gate (Appendix C; Figure 7, Table 6, Figure 8):
    - Baseline sensitivity: softmax gates are more sensitive to imbalance and yield higher perplexity than sigmoid under similar balance (Figure 7).
    - With LFB adapted (`b_i += u * e_i`, `u=1e-3`), softmax models achieve slightly lower perplexity and much better balance than aux-loss:
      > Table 6: LFB perplexity 9.599 vs 9.604; `MaxVio_global` 0.027 vs 0.937.
- Expert parallelism analysis (Section 5.1; Figure 5):
  - As computation-batch size increases, LFB’s `MaxVio_computation-batch` improves continuously and surpasses auxiliary-loss baselines whose balance stops improving much beyond moderate batch sizes. This suggests LFB is particularly attractive for large expert-parallel setups.
- EC future-token leakage evidence (Section 5.2; Appendix D; Figure 6; Figure 9):
  - Theoretical leakage capacity per layer is `K * log2((1−R)/R)` bits (Appendix D.1, Eq. 7).
  - Empirical signs of leakage:
    - Reducing the chunk within which Top-K is computed from 8192 to 512 tokens yields an “abnormal loss drop (~10%)” (Figure 9), consistent with the model exploiting future tokens within the chunk.
    - Shuffling tokens across chunks mitigates this drop (Figure 9), further implicating leakage.
  - This supports LFB’s stance of preserving causality while avoiding balance-driven leakage.

Overall assessment: The experiments convincingly show that LFB commands superior load balance and avoids the performance penalties of auxiliary-loss methods. While absolute perplexity gains are small, the consistency across model sizes and gates, plus major balance improvements, substantiate the core claims.

## 6. Limitations and Trade-offs
- Hyperparameter sensitivity:
  - The update rate `u` is crucial (Figure 4). Too small yields slow balancing; too large creates oscillations. Although only one scalar, it must be tuned per scale/gate.
- Scope of evaluation:
  - Validation uses a single internal multilingual corpus; results are reported in perplexity rather than downstream tasks (Section 4.1; Appendix B). Generalization to diverse benchmarks or instruction-tuned settings is not assessed.
- Magnitude of modeling gains:
  - Perplexity improvements are modest (e.g., 9.56 → 9.50, Table 2). The primary benefit is balance and potential throughput gains; end-task improvements remain to be demonstrated.
- Potential side effects on specialization:
  - By actively steering routing toward underused experts, LFB may at times counteract emergent specialization if certain experts are genuinely more relevant to the current batch. The paper does not study impacts on expert specialization quality.
- Implementation caveats:
  - Requires tracking per-expert token counts every step and maintaining a stateful bias vector across steps. This is lightweight but nonzero complexity in large-scale distributed training.
- The “no interference gradients” claim is architectural:
  - LFB indeed avoids adding new loss terms. However, it still changes which experts receive gradients by altering routing. In pathological cases, overly aggressive balancing could route tokens to less-suitable experts, potentially degrading learning if `u` is mis-set.

## 7. Implications and Future Directions
- Field impact:
  - LFB cleanly separates “how to route” (a control problem) from “what to learn” (the loss). This reframing could become a standard way to manage sparsity-induced resource skew in large MoEs without corrupting the learning signal.
- Practical applications:
  - Scaling expert-parallel LLMs where step time is bottlenecked by the most-loaded device. LFB’s near-uniform `MaxVio` improves utilization and likely throughput (Section 5.1; Figure 5).
  - Stable training of large sparse models (e.g., code or multilingual models) where routing collapse is common and auxiliary-loss tuning is brittle.
- Research directions:
  - Adaptive or learned controllers: Replace the fixed-step sign update with principled controllers (e.g., proportional-integral schemes or bandit-style updates) while retaining causality and no-gradient constraints.
  - Specialization-aware balancing: Incorporate measures that preserve or enhance expert specialization alongside balance (e.g., bias floors/ceilings per expert, temperature scheduling).
  - Broader evaluations: Assess downstream task performance, transfer, robustness to distribution shifts, and training throughput in real distributed systems.
  - Interplay with other MoE components: Study joint effects with capacity constraints, token dropping, noisy routing, or learned expert capacity caps.
  - Safety and leakage guarantees: Formalize causality guarantees and develop audits to detect leakage in more routing schemes beyond EC.

In sum, Loss-Free Balancing is a simple, causality-preserving, and gradient-free way to stabilize expert utilization in MoE training. It achieves strong load balance and modest but consistent perplexity gains over auxiliary-loss approaches, with especially promising implications for large-scale expert-parallel training.
