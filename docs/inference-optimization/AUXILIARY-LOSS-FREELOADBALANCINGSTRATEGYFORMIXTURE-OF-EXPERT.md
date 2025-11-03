# AUXILIARY-LOSS-FREE LOAD BALANCING STRATEGY FOR MIXTURE-OF-EXPERTS

**ArXiv:** [2408.15664](https://arxiv.org/abs/2408.15664)

## 🎯 Pitch

This paper introduces Loss-Free Balancing, a novel routing strategy for Mixture-of-Experts (MoE) models that achieves expert load balance without the need for auxiliary losses. By dynamically biasing expert routing scores based on recent usage, it avoids the detrimental interference gradients that auxiliary losses introduce—resulting in both superior model performance and more efficient training. This innovation breaks a longstanding trade-off in MoE training between load balance and perplexity, offering a safer, more scalable foundation for large-scale language model development.

---

## 1. Executive Summary
This paper introduces Loss-Free Balancing, a training-time routing strategy for Mixture-of-Experts (MoE) models that keeps expert loads balanced without using an auxiliary loss. It matters because auxiliary losses—widely used to prevent routing collapse—inject gradients that compete with the language modeling objective, degrading final model quality; the proposed method balances load while avoiding these interference gradients.

## 2. Context and Motivation
- Problem addressed
  - In MoE models, tokens are routed to a small subset of “experts.” If too many tokens go to a few experts, two problems occur: routing collapse (unused experts never learn) and inefficiency (some devices idle while others stall), as discussed in §2.2.
  - Standard practice enforces balance with an auxiliary loss that penalizes imbalance, e.g., Switch and GShard. The auxiliary loss introduces extra gradients that compete with the main training objective (§2.2, Eq. (2)), creating a trade-off: stronger balancing harms perplexity, weaker balancing risks collapse.

- Why it matters
  - Practical: Uneven loads slow distributed training and waste compute; severe imbalance can cause instability from overloaded experts (§2.2).
  - Scientific: Interference gradients from auxiliary losses can cap model quality, especially as models scale (§1, §2.2, Figure 2).

- Prior approaches and shortcomings
  - Auxiliary-loss balancing (Switch/GShard): Controls load but creates a “dilemma between load balance and model performance” (Figure 2). The paper shows larger α improves balance but worsens perplexity; smaller α improves perplexity but risks collapse (§2.2, Figure 2).
  - Expert Choice (EC) routing: Guarantees equal load by construction but breaks the causal constraint in language modeling, leading to “future token leakage” (information from future tokens influences earlier routing) (§5.2, Figure 6, Appendix D).

- Positioning
  - The proposed Loss-Free Balancing (LFB) directly controls routing without auxiliary loss, thereby avoiding interference gradients. It keeps causal integrity by updating routing biases using only previous-batch statistics (§3, Algorithm 1; note in §3 that current-batch info would leak future tokens).

## 3. Technical Approach
At a high level, LFB adds a small, expert-specific bias to the routing scores before choosing the top-K experts for each token. These biases are updated after each training batch to dampen experts that were overloaded and to boost those underused—without backpropagating through the bias updates.

Key pieces:

- Base MoE layer and routing
  - Each token representation `u_t` passes through an MoE layer with `N` experts. A gating function produces a per-expert score `s_{i,t}`; Top-K experts are selected; only those experts’ outputs are combined to produce the token’s layer output (§2.1, Eq. (1)).
  - In main experiments, the gating `G` is a sigmoid over a similarity between `u_t` and an expert-specific centroid `e_i` (§2.1, Eq. (1); Appendix C compares sigmoid vs softmax gates).

- Why auxiliary losses are problematic
  - The standard balancing loss (Eq. (2)) penalizes deviation from equal token fractions per expert. Its gradient “interferes” with the language modeling objective, producing the trade-off seen in Figure 2 (§2.2).

- Loss-Free Balancing (LFB): biasing the routing scores
  - Idea: Before Top-K selection, add an expert-specific bias `b_i` to each expert’s score and select the Top-K using the biased score. Crucially, the biases affect only the routing decision, not the contribution weights in the MoE output (§3, Eq. (3), note immediately below Eq. (3)).
  - Selection rule (Eq. (3)): A token routes to expert `i` if `s_{i,t} + b_i` is among the top-K across experts. Once selected, the weight applied to expert `i`’s output remains the original `s_{i,t}` (not `s_{i,t}+b_i`).

- Bias update rule (Algorithm 1)
  - After each batch:
    1) Count tokens assigned to each expert (`c_i`), compute the average across experts (`c̄`).
    2) Compute load violation `e_i = c_i - c̄`.
    3) Update the bias by a small step in the opposite direction of overload: `b_i ← b_i + u * sign(e_i)`, where `u` is a small learning-rate-like hyperparameter (§3, Algorithm 1).
  - Design choice: Use previous-batch loads only, to avoid future-token leakage that would arise if current-sequence loads informed routing of earlier tokens (§3, last paragraph).
  - Intuition: If an expert was overloaded, reduce its bias (making it less likely to be picked next batch); if underloaded, increase its bias.

- Alternative variants and their behavior
  - Magnitude-proportional updates: `b_i ← b_i + u * e_i`. Slightly better balance but no perplexity gain; sometimes worse (§4.3, Table 3).
  - Multiplicative biasing (scale scores by `b_i`): Similar balance, slightly worse perplexity (§4.3, Eq. (5), Table 4).
  - Update rate sensitivity: Very small `u` converges too slowly (poor early balance); very large `u` oscillates (poor late balance). Best observed near `u = 0.001` in 1B model (§4.3, Figure 4).

- Why this avoids interference gradients
  - The biases are updated outside the gradient graph using simple counts; no auxiliary loss is added to the training objective. Therefore, gradients flowing into model parameters come only from the language modeling loss (§3, §1).

- Architecture and training setup that ground the method
  - Backbone: DeepSeekMoE with shared experts plus routed experts (§4.1 and Appendix A, Eq. (6)). Main setup: 64 routed experts, 6 activated per token (K=6), plus 2 shared experts (Table 5).
  - Gate: sigmoid used in main experiments; softmax analyzed in Appendix C.
  - Data: multilingual corpus (web, math, code, literature). Tokenizer: BPE with 32k vocab (§4.1).
  - Scales: 1B params trained on 100B tokens; 3B on 200B tokens (§4.1).
  - Optim schedules: cosine for 1B; multistep for 3B (Appendix B).

## 4. Key Insights and Innovations
- Bias-before-selection, not loss-after-selection
  - Novelty: Instead of adding a differentiable penalty to the loss, LFB introduces a non-differentiable, expert-wise bias to the gating scores purely to shape routing decisions (§3, Eq. (3), Algorithm 1). This removes interference gradients that plagued auxiliary-loss methods.
  - Significance: In experiments, this simultaneously improves balance and perplexity (Table 2), breaking the usual trade-off (Figure 2).

- Causality-preserving balancing
  - Insight: Using only historical (previous-batch) loads preserves the causal structure of language modeling, avoiding future token leakage (§3). This contrasts with Expert Choice, which the paper argues leaks future information (§5.2, Figure 6; Appendix D).

- Extremely simple, robust update rule
  - Contribution: A tiny, sign-only update with one scalar hyperparameter `u` yields strong balance and performance (§4.3, Figure 4; Table 3). The method works with additive biases and does not need precise tuning.

- Compatibility with expert parallelism and increasing batch sizes
  - Insight: As the “computation-batch” (micro-batch × expert-parallel data-parallel size) grows, LFB’s batch-level balance approaches its excellent global balance. Figure 5 shows `MaxVio_computation-batch` improves steadily with larger computation batch sizes for LFB, while auxiliary-loss balance plateaus (§5.1). This is important for large-scale deployments.

- Clarification of Expert Choice risks with theory and practice
  - The paper quantifies a theoretical upper bound for information leakage per token in EC: for sparsity `R = K/N`, leakage exceeds `K * log2((1 - R)/R)` bits per token (Appendix D.1, Eq. (7)). Empirically, smaller chunk sizes for EC cause “abnormal loss drop,” which is mitigated by shuffling tokens across chunks—consistent with reduced leakage opportunities (Appendix D.2, Figure 9).

## 5. Experimental Analysis
- Evaluation methodology
  - Metrics:
    - Performance: validation perplexity (lower is better) (§4.1).
    - Load balance: `MaxVio = (max_i Load_i − ȳLoad) / ȳLoad` (Eq. (4)), where `Load_i` is tokens assigned to expert `i`, and `ȳLoad` is the ideal equal-load baseline.
      - `MaxVio_global`: measured over the entire validation set (reflects overall balance when batches are large).
      - `MaxVio_batch`: measured per training batch (reflects training-time efficiency) (§4.1).
  - Baseline: Auxiliary-loss method with coefficient `α = 0.001` chosen from the trade-off curve (Figure 2) as a reasonable middle ground (§4.1).
  - Models: 1B (100B tokens), 3B (200B tokens), both on DeepSeekMoE; sigmoid gate by default (§4.1). Additional softmax experiments in Appendix C.

- Main quantitative results
  - Overall performance and global balance (Table 2):
    - 1B: 
      - Perplexity: LFB 9.50 vs aux-loss 9.56.
      - `MaxVio_global`: LFB 0.04 vs aux-loss 0.72.
    - 3B:
      - Perplexity: LFB 7.92 vs aux-loss 7.97.
      - `MaxVio_global`: LFB 0.04 vs aux-loss 0.52.
    - Quote:
      > Table 2: Loss-Free Balancing achieves lower perplexity and better load balance on both 1B and 3B models.
  - Training-time balance over steps (Figure 3): `MaxVio_batch` is consistently far lower for LFB across training. The curves show rapid stabilization near 0 for LFB; the aux-loss method remains significantly imbalanced.
  - Gate type sensitivity (Appendix C):
    - Baseline sensitivity: Softmax gate exhibits higher perplexity at similar balance and is more sensitive to imbalance (Figure 7).
    - Under softmax, LFB still improves both metrics over aux-loss:
      > Table 6: softmax gate — Perplexity 9.599 (LFB) vs 9.604 (aux-loss); `MaxVio_global` 0.027 (LFB) vs 0.937 (aux-loss).
  - Update rate ablation (Figure 4):
    - `u = 0.001` offers the best balance/performance trade-off. `u = 0.0001` under-corrects early; `u = 0.01` over-corrects late.
    - Quote:
      > Figure 4: Low `u` shows poor early balance; high `u` deteriorates late balance. Validation PPL is best near `u = 0.001`.
  - Update rule variants (Table 3):
    - Magnitude-based updates (`u * e_i`) slightly reduce `MaxVio_global` but do not improve perplexity; best observed perplexity remains with `sign(e_i)` updates.
  - Multiplicative vs additive bias (Table 4):
    - Additive bias slightly outperforms multiplicative bias in perplexity while offering comparable balance.
  - Expert parallelism scaling (Figure 5):
    - As computation batch increases, LFB’s `MaxVio_computation-batch` continues to drop, unlike aux-loss which plateaus. This suggests LFB’s batch-level balance converges to its strong global balance in realistic parallel setups (§5.1).
  - Expert Choice leakage tests (Appendix D.2, Figure 9):
    - Smaller chunk size (512 tokens) shows an “abnormal loss drop,” consistent with easier leakage; shuffling across chunks mitigates this drop.

- Do the experiments support the claims?
  - Yes, for the paper’s primary claims: LFB achieves both improved balance and slightly better perplexity than aux-loss at 1B and 3B scales (Table 2), maintains better balance over training (Figure 3), scales favorably with expert parallelism (Figure 5), and avoids EC’s causality issues (Appendix D).
  - Caveats: Improvements in perplexity are modest but consistent; evaluation is on an internal corpus with perplexity as the primary metric; speedups are inferred from balance metrics rather than reported wall-clock timings.

## 6. Limitations and Trade-offs
- Assumptions and scope
  - The bias updates use previous-batch statistics; this assumes stationarity across adjacent batches. Rapid distribution shifts within a training step are not addressed (§3).
  - The approach focuses on Top-K gating with per-expert scores; other routing paradigms (e.g., token-choice, capacity-constrained Sinkhorn) are not evaluated.

- Scenarios not covered
  - Public benchmarks (e.g., downstream NLP tasks) are not reported; evaluation relies on validation perplexity on a private corpus (§4.1).
  - Inference-time behavior is not directly measured. While balanced routing should help throughput, the paper does not present end-to-end latency/throughput results.
  - Extremely large MoE scales beyond 3B parameters or different K/N ratios are not tested. The leakage analysis is focused on EC, not on other balancing strategies.

- Computational and implementation considerations
  - The bias update requires counting tokens per expert per batch (low overhead), but in highly sharded, multi-node systems, synchronizing these counts may add communication.
  - Hyperparameter `u` needs light tuning (§4.3). An overly aggressive or conservative choice harms balance at different stages (Figure 4).

- Open questions
  - Convergence properties: There is no formal proof of convergence or optimality of the bias update dynamics.
  - Interaction with different gates: While softmax experiments exist (Appendix C), they require a different update rule (`u * e_i`); general prescriptions for other gates are not fully developed.
  - Robustness across non-language modalities (vision, speech) or multi-task settings is untested.

## 7. Implications and Future Directions
- How this changes the landscape
  - Demonstrates that MoE load balancing does not require auxiliary losses and their interference gradients. This reframes routing control as an online control problem outside the gradient graph, potentially raising the ceiling on MoE model quality (Table 2, Figure 2).

- Follow-up research enabled/suggested
  - Adaptive controllers: Learn `u` per expert or per layer; use moving averages or PID-like control to reduce oscillations while improving responsiveness (§4.3 hints at dynamics).
  - Multi-batch memory: Track longer windows of load to stabilize updates; consider momentum or exponential moving averages.
  - Theoretical analysis: Study stability and convergence of bias dynamics; characterize optimal control under stochastic token loads.
  - Broader routing contexts: Apply LFB to other MoE variants (e.g., hierarchical experts, MoE attention) and to non-language domains.
  - Fairer comparisons: Benchmark on public corpora and report wall-clock speed/efficiency under expert parallelism to quantify practical gains suggested by Figure 5.

- Practical applications
  - Large-scale LLM training where expert parallelism is essential. LFB’s improved batch-level balance as computation batch grows (Figure 5) implies better utilization and potentially lower training cost.
  - Safer scaling than Expert Choice for causal language modeling, avoiding future-token leakage (Figure 6; Appendix D).

> Bottom line: Loss-Free Balancing offers a simple, implementable mechanism—per-expert biases updated from previous-batch loads—that consistently improves both load balance and perplexity over auxiliary-loss baselines at 1B–3B scales (Table 2), while preserving causality and aligning well with expert-parallel training (Figure 5).
