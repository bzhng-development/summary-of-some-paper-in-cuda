# Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads

**ArXiv:** [2401.10774](https://arxiv.org/abs/2401.10774)
**Authors:** Tianle Cai, Yuhong Li, Zhengyang Geng, Hongwu Peng, Jason D. Lee, Deming Chen, Tri Dao
**Institutions:** Princeton University (Electrical and Computer Engineering; Center for Statistics & Machine Learning; Princeton Language and Intelligence (PLI)); potentially affiliations of all authors (not fully enumerated in abstract)

## 🎯 Pitch

MEDUSA introduces a groundbreaking approach to speeding up large language model inference by integrating lightweight decoding heads, enabling parallel token prediction and verification. This innovation significantly reduces latency and energy consumption in interactive applications, offering a substantial 2.3–2.8x speedup without compromising on output quality, thus making it a compelling choice for efficient AI deployments.

---

## 1. Executive Summary
MEDUSA accelerates inference in large language models (LLMs) by adding a handful of lightweight “decoding heads” that predict several future tokens in parallel from the last hidden state, then verifying those predictions at once using a tree-structured attention mask. With two training modes—drop-in heads on a frozen model (MEDUSA-1) and joint training with the backbone (MEDUSA-2)—the method achieves 2.3–2.8× wall‑time speedups on multiple Vicuna and Zephyr models without hurting output quality (Figure 3a, Table 1).

## 2. Context and Motivation
- Problem/gap:
  - Autoregressive LLMs generate one token at a time; each step repeatedly streams full model weights from high-bandwidth memory to on-chip cache. This makes inference “memory‑bandwidth‑bound,” not compute‑bound, so accelerators are underutilized (Introduction; A.1 “LLM Inference Acceleration”).
- Why it matters:
  - Latency limits interactive applications (chatbots, coding assistants), cost, and energy usage. Increasing “arithmetic intensity” (doing more computation per memory transfer) per decoding step can reduce latency and energy (Introduction).
- Prior approaches and shortcomings:
  - Speculative decoding uses a small “draft model” to propose several tokens that the large model then accepts/rejects (Leviathan et al., 2022; Chen et al., 2023). Challenges: training or obtaining a well-matched draft model, distribution shift between draft and target, and serving complexity when two models must be orchestrated in distributed systems (Introduction; A.1).
- This paper’s positioning:
  - Replace the separate draft model with a few single‑layer heads attached to the backbone’s final hidden state. Predict multiple future tokens in parallel and verify them with a custom attention mask—no extra model to serve, and far simpler to integrate (Abstract; Sections 2.1–2.3; Figure 1).

## 3. Technical Approach
At each decoding step MEDUSA performs three sub-steps—generate candidates, process them, and accept a continuation—mirroring speculative decoding but without a separate draft model (Section 2; Figure 1).

1) How the “MEDUSA heads” generate parallel predictions (Section 2.1.1)
- Idea: Attach K extra “decoding heads” on top of the backbone’s last hidden state `h_t`. The `k`‑th head predicts the token at position `t + k + 1` (the base LM head still predicts position `t + 1`).
- Architecture: each head is a single-layer feed-forward block with residual connection:
  - For head `k`, the predictive distribution is
    - `p_t^(k) = softmax(W2^(k) * (SiLU(W1^(k) * h_t) + h_t))`, with `W2^(k) ∈ R^(d×V)`, `W1^(k) ∈ R^(d×d)` (Section 2.1.1).
  - Initialization aligns the heads with the base LM: `W2^(k)` copied from the original LM head; `W1^(k)` initialized to zero. This means initially the heads mimic the main head’s distribution at `t+1`.
- Why this design:
  - Parameter‑efficient (just head layers, no new transformer blocks).
  - Model‑serving simplicity: still only one model to host; heads share all backbone computation and KV cache with the LM (Sections 2.1.1, 2.2).

2) How multiple candidate continuations are formed and processed at once (Section 2.1.2; Figure 2)
- Candidate construction:
  - For each head `k`, take its top-`s_k` tokens. The Cartesian product across heads gives up to `∏_{i=1..K} s_i` candidate branches of length `K` (plus the first token from the base head).
  - Example: if head1 top-2 and head2 top-3, there are `2 × 3 = 6` branches (Figure 2).
- Tree attention mask:
  - Treat the candidates as branches in a tree. A token can only attend to predecessors on its own branch, enforced by a custom attention mask and adjusted positional indices (Figure 2).
  - This allows the backbone to score all branch tokens in one forward pass without increasing batch size. Total tokens processed in parallel per step equals `Σ_{k=1..K} ∏_{i=1..k} s_i` (Section 2.1.2).
- Why this matters:
  - Many proposed continuations are evaluated simultaneously, increasing the chance of accepting multiple tokens each step, which boosts “acceleration rate” (average tokens accepted per step).

3) Accepting a continuation (Section 2.3.1)
- Two options:
  - Rejection sampling (RS): accept only if the large model would sample those tokens under the same sampling scheme; preserves the exact output distribution but suffers efficiency loss at higher temperatures (Section 2.3.1).
  - Typical acceptance (new): accept any candidate prefix whose per-token probability under the large model exceeds a threshold that adapts with the distribution’s entropy:
    - For token `x_{n+k}`, accept if `p_original(x_{n+k} | x_{1..n+k-1}) > min(ε, δ * exp(-H(p_original)))`, where `H(·)` is entropy, `ε` a hard threshold, `δ` a scaling factor (Section 2.3.1).
    - To guarantee progress, the first token (greedy from the base head) is always accepted; beyond that, pick the longest candidate prefix that passes the threshold (Section 2.3.1).
- Rationale:
  - RS ensures distribution fidelity but adds overhead and degrades as temperature rises. The typical acceptance rule keeps outputs “typical” for the large model, preserving quality while improving acceptance length and speed (Section 2.3.1; Figure 5).

4) Training strategies (Section 2.2)
- MEDUSA‑1 (frozen backbone; Section 2.2.1):
  - Loss: cross‑entropy between the `k`‑th head and ground truth token `y_{t+k+1}` with a decay weight `λ_k` (to downweight higher‑k heads that are harder to predict):
    - Equation (1): `L_MEDUSA-1 = Σ_{k=1..K} -λ_k * log p_t^(k)(y_{t+k+1})`, with `λ_k ≈ 0.8^k` in practice.
  - Practical note: can train with a quantized backbone (QLoRA-like) on a single GPU; e.g., 5 hours for Vicuna‑7B using 60k ShareGPT samples (Section 2.2.1).
- MEDUSA‑2 (joint training; Section 2.2.2):
  - Preserve next‑token quality while improving head accuracy via three tactics:
    - Combined loss: Equation (2) `L_MEDUSA-2 = L_LM + λ0 * L_MEDUSA-1`, adding the base LM’s next-token loss to stabilize its behavior.
    - Differential learning rates: larger LR for heads than for the backbone.
    - Heads warmup / two‑stage schedule: start by training heads (MEDUSA-1), then joint training with a warmup for the backbone or by gradually increasing `λ0` (Section 2.2.2).
  - Can be integrated into standard SFT so the released model “natively” supports MEDUSA (Section 2.2.2).
- Self‑distillation when no SFT data is available (Section 2.3.2):
  - Generate conversations from seed prompts using the target model itself; for joint training use a KL loss that matches the backbone logits to the original model (teacher) while training heads:
    - `L_LM-distill = KL(p_original^(0) || p^(0))` (Section 2.3.2).
  - Memory‑efficient trick: implement the backbone as a LoRA adapter; the teacher is the same network with the adapter turned off, so no second model copy is needed (Section 2.3.2).

5) Optimizing the candidate tree (Section 2.3.3; Appendix C)
- Goal: with a fixed token budget (number of tree nodes), choose which top‑i predictions per head to include.
- Method: on a calibration set, estimate the accuracy `a_k^(i)` of the i‑th top token at head `k`. Approximate a candidate prefix `[i1..ik]` accuracy as `∏_j a_j^(i_j)` and greedily add nodes with highest marginal contribution to the expected accepted length until the token budget is reached (Section 2.3.3).

6) Hardware characterization (Appendix G)
- Roofline analysis shows that standard decoding is memory‑bandwidth‑bound for attention and most linear layers. MEDUSA increases “operational intensity” (FLOPs per byte moved) by processing many candidate tokens at once, shifting parts of the workload toward compute‑bound regimes (Figures 18–20; Tables 6–8). However, too many candidates create compute bottlenecks, so there is an optimal range (Figures 4b, 21–23).

## 4. Key Insights and Innovations
- Extra decoding heads instead of a draft model
  - What’s new: A single backbone with a few one‑layer heads predicts multiple future tokens from the same hidden state (Section 2.1.1).
  - Why it matters: Avoids the engineering and alignment burden of serving/training a separate draft model; reduces distribution mismatch and infrastructure complexity (Introduction; Section 2.1.1).
- Tree attention for concurrent verification of many candidates
  - What’s new: A top‑down tree mask lets tokens attend only to their branch predecessors, enabling many candidate continuations to be scored in one pass without increasing batch size (Section 2.1.2; Figure 2).
  - Why it matters: Increases accepted tokens per step while keeping memory movement per step similar; boosts arithmetic intensity.
- Typical acceptance rule
  - What’s new: A distribution‑aware thresholding based on entropy chooses “typical” candidate prefixes instead of strict rejection sampling (Section 2.3.1). The rule is
    - `p_original(x) > min(ε, δ * exp(-H(p_original)))`, applied token‑wise within candidates after greedily accepting the first token.
  - Why it matters: Especially at higher temperatures, typical acceptance yields longer accepted prefixes and higher speed than RS while maintaining similar quality (Figure 5).
- Two training modes and self‑distillation
  - MEDUSA‑1 enables “bolt‑on” speedups to existing models, even with quantized backbones (Section 2.2.1).
  - MEDUSA‑2 co‑trains heads with backbone using a combined loss and warmup; self‑distillation removes the need for original SFT data (Sections 2.2.2, 2.3.2).
  - Significance: Offers both a low‑resource adoption path and a higher‑performance “native” path.

## 5. Experimental Analysis
- Evaluation setup (Sections 3, B)
  - Models: Vicuna‑7B/13B/33B (v1.5; Llama‑2 base) and Zephyr‑7B (Sections 3.1–3.2).
  - Data/training:
    - For 7B/13B: ShareGPT fine‑tuning for heads (2 epochs; Section 3.1).
    - For 33B and Zephyr‑7B: self‑distillation from ShareGPT and UltraChat seed prompts; ~100k samples (Section 3.2).
    - Common training choices: 5 heads, `λ_k = 0.8^k` (B.2).
  - Metrics:
    - Speed: tokens per second; “acceleration rate” (tokens accepted per step); “overhead” (per‑step latency vs. vanilla); “speedup” = acceleration rate / overhead (B.1).
    - Quality: MT‑Bench score (0–10) via GPT‑4 judge (Sections 3, 3.2).
  - Baselines: Vanilla decoding; open‑source speculative decoding with draft models (Appendix D; Table 1).
- Main quantitative results
  - Wall‑time speedups (Figure 3a):
    - Vicuna‑7B: MEDUSA‑1 2.18×; MEDUSA‑2 2.83×.
    - Vicuna‑13B: MEDUSA‑1 2.33×; MEDUSA‑2 2.83×.
  - Category‑level speedups (Figure 3b, MEDUSA‑2, 7B):
    - Largest gains on Extraction (3.62×) and Coding (3.29×), indicating many deterministic spans can be accepted per step.
  - Self‑distillation setting (Table 1):
    - Acceleration rate (accepted tokens/step) ≈ 3.0–3.5; overhead ≈ 1.18–1.27; resulting speedups:
      - Vicuna‑7B 2.83×; Zephyr‑7B 2.66×; Vicuna‑13B 2.83×; Vicuna‑33B 2.35×.
    - Quality differences on MT‑Bench are small: for example, Vicuna‑7B change +0.01; Zephyr‑7B −0.07; Vicuna‑33B +0.05.
  - Comparison to speculative decoding with public drafts (Table 1; Appendix D):
    - Reported speedups for speculative decoding are ~1.47–1.60× on the Vicuna lineup, lower than MEDUSA’s 2.35–2.83×.
- Ablations and robustness
  - Tree size vs. speed (Figure 4):
    - “Sparse” optimized trees with ~64 nodes achieve higher acceleration rates than much larger “dense” Cartesian trees; too many candidates reduce tokens/s due to compute overhead (Figure 4b).
  - Typical acceptance vs. RS (Figure 5):
    - As the probability threshold `ε` increases, quality rises and acceleration falls; with `T=0.7`, typical acceptance traces match RS quality while achieving higher acceleration (plot shows acceleration in the 3.0–3.5 range).
  - Two‑stage joint training is necessary (Table 2):
    - Direct joint fine‑tuning lowers quality (MT‑Bench 5.93 vs. baseline 6.17), whereas MEDUSA‑2 keeps quality (6.18) and attains 2.83× speedup.
- Hardware modeling (Appendix G):
  - Roofline plots (Figures 9–17) and simulations (Figures 21–23) show MEDUSA lifts operational intensity and FLOPs/s for attention and MLPs, but speedup saturates or declines beyond ~64 candidate tokens and with very large batches.

> Figure 3a: “MEDUSA‑1 shows a 2.18× speedup on Vicuna‑7B and 2.33× on 13B; MEDUSA‑2 delivers 2.83× on both.”
>
> Table 1: “Acceleration rates ≈ 3.01–3.51 with overhead ≈ 1.18–1.27 yield speedups of 2.35–2.83×; MT‑Bench quality deltas are within ±0.14.”
>
> Table 2: “Direct fine‑tuning hurts quality (5.93). MEDUSA‑1 (6.23, 2.18×) and MEDUSA‑2 (6.18, 2.83×) preserve quality and improve speed.”

Assessment: The evidence is consistent and multi‑sided—absolute speeds, accepted‑token analysis, ablations for tree design and acceptance scheme, and quality measurements—supporting the claim of large wall‑time speedups without quality loss across models and datasets (Sections 3, 3.3; Figures 3–5; Tables 1–2).

## 6. Limitations and Trade-offs
- Distribution changes vs. exact fidelity:
  - Typical acceptance does not reproduce the exact sampling distribution of the original model as RS does (Section 2.3.1). This is a pragmatic trade‑off for speed; it may matter for tasks requiring strict statistical fidelity.
- Tuning and engineering knobs:
  - Performance depends on choices of number of heads K, tree size, top‑k per head, and typical‑acceptance thresholds `ε, δ`. Overly large trees can reduce tokens/s (Figure 4b). Thresholds affect the quality/speed trade‑off (Figure 5).
- Training data dependence:
  - MEDUSA‑2 needs suitable data for joint training; when unavailable, self‑distillation is used. The 33B case shows smaller speedup, possibly due to mismatch between hidden SFT data and self‑distilled data (Section 3.2, Table 1).
- Focus on batch size 1:
  - Most experiments assume batch size 1 (Introduction; Discussion). While the authors mention broader applicability and note that later libraries support it, the empirical evidence here focuses on single‑request latency.
- Compute vs. bandwidth regimes:
  - MEDUSA shifts work toward compute‑bound regimes; too many candidate tokens or very large batches can push into compute saturation, lowering net speedup (Appendix G; Figures 21–23).
- Scope:
  - The method assumes standard decoder‑only transformer LLMs with causal attention and typical KV‑cache serving; applicability to non‑standard architectures would require custom masking and may vary.

## 7. Implications and Future Directions
- Field impact:
  - MEDUSA reframes multi‑token parallelization as a property of the backbone itself rather than a separate draft model. This simplifies deployment and encourages broader adoption of parallel decoding in production LLM systems.
- Practical applications:
  - Lower‑latency interactive assistants, on‑device or edge deployment due to fewer memory transfers, faster coding assistance, and cheaper API serving given the 2.3–2.8× speedups (Figure 3a; Table 1). Typical acceptance is a strong default for creative/temperature‑controlled generation (Figure 5).
- Research directions:
  - Adaptive trees: learn or dynamically select which head depths and top‑i branches to expand based on context, latency targets, or confidence (Section 2.3.3; Appendix C).
  - Quality guarantees: hybrid acceptance schemes that interpolate between typical acceptance and RS to provide tunable fidelity vs. speed.
  - Training integration: joint SFT/RLHF recipes that natively include MEDUSA heads and entropy‑aware acceptance during training.
  - Systems co‑design: combine with KV‑cache optimizations (PagedAttention), quantization, and attention sparsification for compounding gains (A.1; Appendix G).
  - Beyond chat: apply to code generation, extraction, and other domains where Figure 3b shows especially high gains—potentially with domain‑specific head depths and thresholds.

In short, MEDUSA provides a clean, general mechanism to accept multiple tokens per decoding step without separate draft models. Its combination of lightweight heads, tree attention, and typical acceptance delivers substantial and repeatable speedups while preserving output quality across several model families and training regimes.
