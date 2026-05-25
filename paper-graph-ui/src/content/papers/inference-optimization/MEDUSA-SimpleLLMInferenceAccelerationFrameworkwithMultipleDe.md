# MEDUSA: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads

**ArXiv:** [2401.10774](https://arxiv.org/abs/2401.10774)

## 🎯 Pitch

MEDUSA introduces a novel framework that accelerates large language model (LLM) inference by adding lightweight multiple decoding heads to predict and verify several future tokens in parallel, all within the original model—eliminating the need for a separate draft model. This technique dramatically reduces decoding steps and memory transfers, unlocking over 2.3–2.8× speedups without sacrificing output quality, making real-time, cost-effective LLM deployment more accessible on edge devices and small-scale servers.

---

## 1. Executive Summary
MEDUSA accelerates text generation in large language models by adding several lightweight “decoding heads” that predict multiple future tokens in parallel and verify them at once using a tree-structured attention mask. With two training recipes and a new “typical acceptance” rule, the method achieves 2.3–2.8× end‑to‑end speedups without hurting quality on multiple chat models (e.g., Vicuna‑7B/13B, Zephyr‑7B), as shown in Figure 3 and Table 1.

## 2. Context and Motivation
- Problem addressed
  - Autoregressive LLMs generate one token at a time; each step reloads model weights from high‑bandwidth memory (HBM) to on‑chip caches and updates the key–value (KV) cache. This sequential, memory‑bandwidth‑bound process limits throughput and underutilizes compute even though only one token is produced per forward pass (Introduction; also contextualized by the roofline analysis in Appendix G).
- Why it matters
  - Reducing decoding latency translates directly to cheaper serving, better user experience for interactive agents, and feasibility of on‑device or small‑GPU deployments.
- Prior work and shortcomings
  - Speculative decoding uses a small draft model to propose a few tokens which the large model then verifies (Leviathan et al., Chen et al.). While effective, it requires obtaining, maintaining, and serving an additional model. Draft pre‑training can be expensive, can drift in distribution from the target model, and complicates distributed serving (Section 1; Section 2.1.1).
- How this paper positions itself
  - Replace the separate draft model with a few extra heads attached to the original model’s final hidden state. These heads predict future tokens several steps ahead. A tree‑based attention mask allows the main model to verify multiple candidate continuations in one pass (Sections 2.1.1–2.1.2; Figure 1, Figure 2).

## 3. Technical Approach
This section explains what MEDUSA changes during decoding and how it is trained.

- Core idea
  - Add K small “MEDUSA heads” on top of the last hidden state `h_t` to predict tokens for positions `(t+2) … (t+K+1)` in parallel while the original LM head still predicts `(t+1)` (Section 2.1.1).
  - Build a small candidate tree from the top predictions of each head and verify all candidates at once by modifying the attention mask so that each candidate branch only attends to its own past (Section 2.1.2; Figure 2).
  - Accept a verified candidate prefix using either classical rejection sampling (distribution‑preserving) or a new typical‑acceptance rule (Section 2.3.1). The longest accepted prefix determines how many tokens you “jump” forward this step (Figure 1).

- MEDUSA heads: architecture and initialization (Section 2.1.1)
  - Each head is a single feed‑forward layer with residual, applied to `h_t`:
    - In plain language: a small MLP takes `h_t`, applies a nonlinearity (SiLU), adds back `h_t` (residual), and projects to vocabulary logits. Softmax turns logits into probabilities `p_t^(k)`.
    - Notation (Eq. in Section 2.1.1): `p_t^(k) = softmax(W2^(k) * (SiLU(W1^(k) * h_t) + h_t))`.
  - Initialization aligns the heads with the original LM at start: copy `W2^(k)` from the LM head; set `W1^(k) = 0`. This makes initial predictions match the base LM, avoiding distribution shift at the beginning of training.

- Candidate construction and “tree attention” (Section 2.1.2; Figure 2)
  - For each head `k` take top‑`s_k` tokens; form candidates by the Cartesian product across heads. Example: if `s1=2, s2=3`, you get `2×3=6` candidates for positions `(t+1, t+2)`.
  - Tree attention modifies the attention mask so that each token in the candidate tree only attends to tokens earlier on its own branch. Positions for positional encoding are adjusted accordingly. This lets the model score many candidate continuations in a single pass without increasing batch size, only sequence length within the step.
  - Total new tokens computed in one pass equals `sum_{k=1..K} prod_{i=1..k} s_i` (Section 2.1.2).

- Acceptance step (Section 2.3.1)
  - Option A: classical rejection sampling (as in speculative decoding) yields outputs distributed exactly like the base LM but adds overhead when temperature > 0 (Section 2.3.1).
  - Option B: typical acceptance. Define a per‑token acceptance threshold as `min(ε, δ * exp(-H))`, where `H` is the entropy of the base LM’s next‑token distribution conditioned on the proposed prefix. Intuition: accept tokens that are “typical,” i.e., not too improbable when the base LM’s distribution is either sharp (low entropy) or broad (high entropy). In practice, always greedily accept the first token and apply the threshold to subsequent tokens; pick the longest accepted prefix among candidates (Section 2.3.1).
    - Effect: tends to accept more tokens at higher temperatures, boosting speed while maintaining quality (Figure 5).

- Training strategies (Section 2.2)
  - MEDUSA‑1 (frozen backbone; Section 2.2.1)
    - Freeze the base LM; train only the heads with cross‑entropy on the true tokens for future positions. Weight later heads more lightly (`λ_k ≈ 0.8^k`) because further‑ahead predictions are harder (Eq. 1).
    - Can load the backbone in 4‑bit/8‑bit to fit on a single GPU, similar to QLoRA; only a few hours of training for 7B models (Section 2.2.1).
  - MEDUSA‑2 (joint training; Section 2.2.2)
    - Train the base LM and heads together with a combined loss `L = L_LM + λ0 * L_MEDUSA-1` (Eq. 2). Use a smaller LR for the backbone and larger for heads.
    - Warmup: start by training only heads (like MEDUSA‑1), then enable joint training, or gradually increase the weight of the base LM loss to prevent quality drift.
    - Often implemented with LoRA/QLoRA adapters on the backbone for parameter efficiency (Appendix B).
  - How many heads? Empirically up to 5 is enough; at inference you can ignore extra heads if not needed (Section 2.2.3).

- Self‑distillation for when training data is unavailable (Section 2.3.2)
  - Generate a synthetic dataset by prompting the model itself with seed prompts (ShareGPT, UltraChat) to produce multi‑turn conversations.
  - For MEDUSA‑2, ground the backbone not on hard tokens but on the original model’s probability distribution using KL divergence (`KL(p_original || p_student)`), implemented with LoRA so the “teacher” is the same network with adapters turned off—no extra GPU memory required (Section 2.3.2).

- Optimizing the tree shape (Section 2.3.3; Appendix C)
  - Dense Cartesian products may waste compute on low‑accuracy branches. Estimate per‑head top‑`i` accuracies on a calibration set and greedily grow a sparse tree to maximize expected accepted length. Figure 4 shows this reduces overhead while keeping acceptance high.

- Why this approach increases speed
  - Each decoding step does more useful work (scores many plausible future tokens) while only loading the model weights once, increasing arithmetic intensity. Roofline plots in Appendix G demonstrate how MEDUSA shifts key operators (attention matmuls, linear layers) toward higher FLOP/s and operational intensity; see Figures 18–20 and Tables 6–8.

## 4. Key Insights and Innovations
- Parallel multi‑step prediction without a draft model
  - Novelty: multiple single‑layer heads attached to the base LM predict several future tokens using the same hidden state `h_t` (Section 2.1.1; Figure 1). Unlike speculative decoding, no separate model needs to be trained or served.
  - Significance: eliminates engineering overhead and distribution shift between draft and target models; easy to drop into existing models and even quantized backbones (Section 2.2.1).

- Tree‑structured verification in one pass
  - Novelty: a simple attention mask that embeds a candidate tree, letting the model verify many continuations concurrently (Section 2.1.2; Figure 2).
  - Significance: boosts expected accepted length per step without increasing batch size, which is crucial when KV cache and memory bandwidth limit batching.

- Typical acceptance rule
  - Novelty: a threshold based on entropy and absolute probability to accept “typical” tokens, trading exact distributional matching for higher acceptance at sampling temperatures > 0 (Section 2.3.1).
  - Significance: empirically yields higher speedups while maintaining quality comparable to random sampling (Figure 5).

- Two fine‑tuning modes and self‑distillation
  - Novelty: a lightweight MEDUSA‑1 for plug‑in acceleration and a joint MEDUSA‑2 with warmup and differential LRs to preserve base LM quality (Section 2.2.2), plus a LoRA‑based self‑distillation recipe that needs no extra memory (Section 2.3.2).
  - Significance: makes the method applicable to both existing closed‑data models and new training runs.

- Hardware‑conscious design
  - Insight: decoding is memory‑bandwidth‑bound; MEDUSA increases arithmetic intensity by creating more compute per weight load. Appendix G uses a roofline model to quantify this, showing up to 41× higher operational intensity for attention matmuls at fixed batch when adding candidate tokens (Table 6; Figure 18).

## 5. Experimental Analysis
- Evaluation methodology
  - Benchmarks: MT‑Bench (multi‑turn conversations; main results) and AlpacaEval (Appendix F/Table 4).
  - Models: Vicuna‑7B/13B/33B and Zephyr‑7B (Sections 3.1–3.2). ShareGPT was used for training 7B/13B heads; 33B and Zephyr used self‑distillation (Section 3.2).
  - Setup: focus on batch size 1 (typical for local interactive use). Speed measured in tokens/s; also decompose into “acceleration rate” (tokens advanced per step) and “overhead” per step (Appendix B.1). Speedup = acceleration rate / overhead.

- Main quantitative results
  - End‑to‑end speedups (Figure 3a):
    - Vicuna‑7B: MEDUSA‑1 2.18×; MEDUSA‑2 2.83×.
    - Vicuna‑13B: MEDUSA‑1 2.33×; MEDUSA‑2 2.83×.
  - Category‑wise on MT‑Bench for Vicuna‑7B/MEDUSA‑2 (Figure 3b):
    - Speedups range from 2.58× to 3.62×; “Coding” 3.29× and “Extraction” 3.62× are highest.
  - Cross‑model comparison with acceptance/overhead/quality (Table 1):
    - Reported MEDUSA‑2 speedups: Vicuna‑7B 2.83×, Zephyr‑7B 2.66×, Vicuna‑13B 2.83×, Vicuna‑33B 2.35×.
    - Quality (MT‑Bench score differences vs original): changes are small, e.g., Vicuna‑7B +0.01, Zephyr‑7B −0.07, Vicuna‑13B −0.14, Vicuna‑33B +0.05.
    - For reference, speculative decoding speedups on the same stacks are 1.47×, 1.56×, and 1.60× for 7B, 13B, 33B respectively (Table 1; Appendix D).
  - AlpacaEval corroborates speedups (Table 4 in Appendix F):
    - e.g., Vicuna‑7B: base 37.07 tok/s → MEDUSA 106.76 tok/s; acceleration rate 3.23; speedup 2.88×.

- Ablations and analysis
  - Tree configuration (Figure 4a,b; Section 3.3.1):
    - A sparse, accuracy‑aware tree with 64 nodes beats randomly sampled dense trees with up to 256 nodes in acceleration rate, and maintains higher real tokens/s by reducing compute overhead.
    - As candidate tokens increase, acceleration rate rises sub‑logarithmically but speed eventually falls due to compute becoming the bottleneck (Figure 4b).
  - Typical acceptance thresholds (Figure 5; Section 3.3.2):
    - As the posterior threshold `ε` increases, quality slightly increases while acceleration rate drops. With temperature 0.7, typical acceptance approaches random sampling quality while keeping higher speed than rejection sampling.
  - Two‑stage fine‑tuning (Table 2; Section 3.3.3):
    - Direct joint training without warmup hurts quality (6.17 → 5.925 on MT‑Bench). MEDUSA‑1 preserves and MEDUSA‑2 preserves or slightly improves quality while delivering larger speedup (2.83× vs 2.18×).
  - Contribution breakdown (Table 3):
    - Heads alone (~1.5×) → +tree attention (~1.9×) → +optimized tree (~2.2×) → +MEDUSA‑2 training (~2.8×).

- Do the experiments support the claims?
  - The speedups are consistent across models and datasets; quality remains comparable by automatic judging (MT‑Bench with GPT‑4). Hardware modeling (Appendix G) explains when/why adding more candidates stops helping—consistent with empirical curves (Figures 4, 21–23). The comparison to speculative decoding in Table 1 and Appendix D supports the claimed advantage when no extra draft model is used.

- Notable conditions and trade‑offs
  - Self‑distillation for Vicuna‑33B yields slightly lower speedup (2.35×) likely due to data mismatch (Section 3.2).
  - Increasing candidate tokens beyond ~64 can reduce real‑world tokens/s due to compute overhead (Figure 4b; simulated in Figure 21).
  - Speedups diminish at large batch sizes or very long sequences (Figures 22–23 in Appendix G.3).

> “MEDUSA‑1 can achieve over 2.2× speedup without compromising generation quality, while MEDUSA‑2 further improves the speedup to 2.3–2.8×.” (Abstract; quantified in Figure 3 and Table 1)

## 6. Limitations and Trade-offs
- Acceptance vs overhead
  - More candidates increase the chance of accepting longer prefixes but also add compute. Beyond a point, extra candidates reduce tokens/s (Figure 4b; Appendix G.3, Figure 21).
- Distributional fidelity
  - Typical acceptance is not distribution‑preserving. If exact sampling equivalence to the base LM is required (e.g., for evaluation reproducibility), one must use rejection sampling at some speed cost (Section 2.3.1; Figure 5).
- Training data and alignment
  - MEDUSA‑2 with self‑distillation is sensitive: training the backbone on self‑generated hard labels can degrade quality; it requires distillation on probabilities (Section 2.3.2). Even then, data mismatch may reduce head accuracy and speed gains (Vicuna‑33B in Table 1).
- Scalability across serving regimes
  - Results focus on batch size 1. Although the technique generalizes, speedups shrink as batch size grows and linear layers become compute‑bound (Appendix G.3, Figure 22).
- Implementation complexity
  - Requires modifying the attention mask and position handling to embed tree attention; although conceptually simple, serving systems must implement this efficient masking and candidate assembly.
- Edge cases
  - Extremely long contexts increase attention cost and reduce net benefit (Appendix G.3, Figure 23). Domains where the next‑token distribution is nearly uniform may yield many rejected tokens unless thresholds are tuned.

## 7. Implications and Future Directions
- Field impact
  - Shifts the default acceleration strategy from “use a draft model” to “extend the LM with parallel heads,” removing a major deployment barrier. MEDUSA complements KV‑cache optimizations and quantization by increasing arithmetic intensity per step (Appendix G).
- Practical applications
  - Latency‑sensitive chat assistants, code completion, on‑device or small‑GPU inference, server throughput gains for API providers, and interactive agents that need real‑time feedback.
- Follow‑up research
  - Adaptive trees: learn per‑prompt head depths or dynamic candidate budgets, potentially using the calibration framework in Section 2.3.3.
  - Theory of acceptance: model acceptance length as a function of entropy/temperature to optimally set typical‑acceptance thresholds.
  - Combination with other accelerations: integrate with paged KV caches, multi‑query/grouped‑query attention, and speculative decoding (e.g., MEDUSA heads inside a small draft model).
  - Quality‑aware heads: train heads to optimize downstream task metrics (not just cross‑entropy), or distill from diverse sampling strategies.
  - Distributed/engine support: native kernels for tree attention and masked position updates; automatic search for sparse tree structures.

Overall, MEDUSA is a simple, system‑friendly way to parallelize multiple decoding steps inside a single LM forward pass. Its careful training recipes, acceptance rule, and hardware‑aware tree construction explain why it reaches 2.3–2.8× wall‑time gains while keeping output quality steady.
