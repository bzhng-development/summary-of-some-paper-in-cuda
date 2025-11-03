# EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty

**ArXiv:** [2401.15077](https://arxiv.org/abs/2401.15077)
**Authors:** Yuhui Li, Fangyun Wei, Chao Zhang, Hongyang Zhang
**Institutions:** Not specified in arXiv metadata

## 🎯 Pitch

EAGLE introduces a groundbreaking 'lossless' framework for accelerating large language model decoding, leveraging feature-level drafting and shifted-token conditioning to maintain output integrity while significantly reducing latency by up to 3.5 times. This advancement offers substantial cost savings and responsiveness improvements for applications like chatbots and code assistants, addressing long-standing efficiency challenges in autoregressive decoding without compromising model fidelity.

---

## 1. Executive Summary
EAGLE is a “lossless” acceleration framework for large language model (LLM) decoding that drafts at the feature level rather than the token level and resolves a key source of uncertainty by conditioning on the next token. It achieves 2.7x–3.5x latency speedups on LLaMA2‑Chat 70B while provably preserving the original model’s output distribution in both greedy (temperature=0) and non‑greedy (temperature>0) decoding (Figures 1–2; Sections 2–3).

## 2. Context and Motivation
- Problem addressed
  - Autoregressive decoding generates one token per forward pass, making LLM inference slow and costly.
  - Speculative sampling speeds this up by drafting several tokens with a cheap model and verifying them in parallel with the target LLM, but it has two practical bottlenecks:
    1) Finding a good draft model that is both fast and distribution‑aligned with the target.
    2) Achieving high acceptance so that many drafted tokens survive verification.

- Why it matters
  - Faster decoding reduces serving costs and latency for chatbots, code assistants, and reasoning systems without changing what the model would have produced (“lossless” acceleration). This is crucial for production systems that must preserve output quality and consistency.

- Shortcomings of prior approaches
  - Classic speculative sampling needs a small yet capable draft model. For smaller targets (e.g., 7B), such a draft does not exist; for mid-sized targets (e.g., 13B), using a 7B draft can negate speedups due to overhead (Figures 1–2 note “N/A” where this setup is impractical).
  - Lookahead uses n‑gram heuristics and Jacobi iteration but is limited to greedy decoding and shows modest draft accuracy.
  - Medusa adds MLP heads that directly predict tokens from internal features, but prediction accuracy is only about 0.6 and non‑greedy decoding is not guaranteed to be lossless (Section 1; Figures 1–2).
  
- Positioning of this work
  - EAGLE reframes drafting: instead of predicting tokens directly, it autoregresses the model’s second‑to‑top‑layer hidden states (“features”) and uses the original LM head to produce token distributions.
  - Crucially, it resolves an overlooked source of uncertainty in feature autoregression by conditioning each next‑feature prediction on a token sequence shifted by one step (“shifted-token” input), which boosts acceptance and speed (Figure 4; Section 3.1).

## 3. Technical Approach
EAGLE follows the standard draft‑and‑verify structure of speculative sampling but changes how drafts are produced.

- Key background: speculative sampling verification
  - After the draft model proposes a sequence `t̂_{j+1:j+γ}` with distributions `p̂`, the target LLM computes its own distributions `p` over the same positions in a single pass.
  - Each drafted token `t̂` is accepted independently with probability `min(1, p(t̂)/p̂(t̂))`. On the first rejection, the remainder is discarded and the next token is sampled from a corrected distribution `norm(max(0, p − p̂))`. This guarantees the final output has exactly the same distribution as vanilla decoding (Section 2, “Speculative sampling”).

- EAGLE’s drafting reframed as feature autoregression
  - “Feature” refers to the hidden state right before the LM head (the second‑to‑top layer of the target LLM). These features are continuous vectors of dimension `hidden_dim`.
  - Instead of predicting tokens, EAGLE predicts the next feature and then uses the target LLM’s LM head to turn that feature into a token distribution, from which it samples the next token (Section 3.1; Figure 6).

- Resolving uncertainty with shifted-token inputs
  - Challenge: Features branch with the sampling outcome. From the same current feature `f_I`, sampling “am” vs. “always” leads to different next features (Figure 3). If the draft model doesn’t know which token will be sampled, next‑feature prediction is ambiguous.
  - Solution: Predict `f_{i+1}` from the known past features `F_{1:i}` and a token sequence shifted by one step `T_{2:i+1}` that includes the next token for that position (Section 3.1; Figure 6). Practically:
    1) Given `F_{1:i}` and already sampled `T_{1:i}`, the draft model predicts `f_i` → applies LM head → samples `t_{i+1}`.
    2) It then predicts `f_{i+1}` conditioning on `F_{1:i}` and `T_{2:i+1}` (the token sequence advanced by one time step).
    3) Repeat to grow a chain or a tree of drafted tokens.
  - This conditioning collapses the branching uncertainty at feature level, making feature prediction much easier (Figure 4 shows the jump in both acceptance and speed when using feature+shifted‑token vs. feature‑only or token‑only).

- Draft model architecture (Section 3.1; Figure 6)
  - Shares the target LLM’s embedding layer and LM head (frozen, no training).
  - Adds an `Autoregression Head` composed of:
    - A fully connected layer to fuse and reduce concatenated `[feature; token_embedding]` inputs from dimension `2×hidden_dim` to `hidden_dim`.
    - A single transformer decoder layer that outputs the next feature.
  - The LM head (from the target) converts the predicted feature into a token distribution for sampling; sampled tokens and predicted features are appended to inputs to continue drafting.

- Tree‑structured drafts with tree attention (Section 3.1; Appendix A.1; Figure 6 and Figure 9)
  - EAGLE expands a token tree rather than a single chain: a small number of forward passes in the draft model creates a deeper, wider set of candidate continuations.
  - Example: with 3 draft forward passes, EAGLE can propose a tree containing 10 tokens (Figure 6).
  - During verification, the target LLM evaluates all nodes in the tree in one pass using tree attention.

- Training the draft model (Section 3.2)
  - Objective blends:
    - Regression on features with Smooth L1 loss: `L_reg = SmoothL1(f_{i+1}, DraftModel(T_{2:i+1}, F_{1:i}))`.
    - Classification on next‑token distributions by passing both ground‑truth and predicted features through the LM head and computing cross‑entropy: `L_cls = CrossEntropy(p_{i+2}, p̂_{i+2})`.
  - Final loss: `L = L_reg + w_cls * L_cls` with `w_cls=0.1`.
  - Robustness to accumulated feature errors: add uniform noise `U(−0.1, 0.1)` to training features (Section 3.2).
  - Training data: 68k ShareGPT dialogues; no tuning on evaluation sets (Section “Training”).

- Verification with multi‑round sampling on trees (Section 3.3; Appendix A.2)
  - The target LLM computes probabilities for all drafted nodes in one pass.
  - A recursive “multi‑round speculative sampling” applies the standard acceptance test across a node’s `k` candidates. If all are rejected, it samples from the corrected distribution (Algorithm 1 in Appendix A.2).
  - Guarantees exact preservation of the target model’s output distribution in both greedy and non‑greedy settings (Section 2; Section 3.3).

## 4. Key Insights and Innovations
- Feature‑level autoregression is easier than token‑level
  - Difference: Rather than learning to map context → next token(s), EAGLE learns to map past features + next token context → next feature, and then relies on the original LM head for tokenization.
  - Evidence: With Vicuna‑7B, feature‑only drafting outperforms token‑only drafting in both acceptance and speed; adding shifted tokens further boosts performance (Figure 4).

- Conditioning on shifted tokens resolves feature uncertainty
  - Novelty: Input includes a token sequence advanced by one time step (`feature&shifted-token`). This disambiguates which branch of the stochastic generation the draft is pursuing (Figure 3).
  - Impact: On Vicuna‑7B (MT‑bench, T=0), the speedup improves from ~1.5x (token-only) to ~1.9x (feature-only) and to ~2.8x with `feature&shifted-token` (Figure 4).

- Tree drafting with tree attention for more tokens per pass
  - EAGLE grows a token tree in m forward passes, yielding >m drafted tokens (Figures 6, 9). Verification also uses tree attention so a single target LLM pass evaluates all nodes.
  - Ablation shows tree attention raises average acceptance length τ by ~0.6–0.8 and wall‑time speedup by ~0.3–0.5 across models (Table 5; Figure 7).

- Lossless acceleration in both greedy and non‑greedy decoding
  - Many drafting methods either target only greedy decoding or relax the acceptance test. EAGLE keeps the verification strictly distribution‑preserving (Section 2; 3.3), so outputs are identical in distribution to vanilla decoding for temperature=0 and temperature>0 (Figures 1–2).

## 5. Experimental Analysis
- Evaluation setup (Section 4)
  - Models: Vicuna (7B, 13B, 33B), LLaMA2‑Chat (7B, 13B, 70B), and Mixtral 8×7B Instruct.
  - Tasks: MT‑bench (multi‑turn dialogue), HumanEval (code), GSM8K (math word problems), Alpaca (instruction following).
  - Metrics (Section “Metrics”):
    - Speedup: wall‑time ratio over vanilla decoding.
    - Average acceptance length `τ`: average number of tokens accepted per target‑LLM pass.
    - Acceptance rate `α`: accepted/generated during drafting. For trees, they report `n‑α` using chain drafts, where `n` denotes how many predicted features may include errors (Table 2).

- Main quantitative results
  - MT‑bench, greedy (T=0): Figure 1 shows EAGLE speedups
    - Vicuna‑7B: 2.90x
    - Vicuna‑13B: 3.07x
    - Vicuna‑33B: 2.95x
    - LLaMA2‑Chat‑7B: 2.78x
    - LLaMA2‑Chat‑13B: 3.03x
    - LLaMA2‑Chat‑70B: 3.01x
    - In contrast, speculative sampling with a 7B draft is N/A for 7B targets and often not beneficial for 13B due to overhead (Figure 1).
  - MT‑bench, non‑greedy (T=1): EAGLE sustains strong speedups where Lookahead is not applicable and Medusa is not lossless. Figure 2 reports values typically around 2.1x–2.9x depending on model size.
  - Across tasks (Table 1):
    - τ ranges from 3.2 to 4.5, meaning a single target‑LLM pass often validates 3–4+ tokens.
    - Example: LLaMA2‑Chat‑13B achieves 3.76x (HumanEval), 3.20x (GSM8K), 3.01x (Alpaca) at T=0 with τ ≈ 4.0–4.5.
  - Acceptance quality (Table 2, T=0):
    - `0‑α` (no feature errors) ≈ 0.74–0.79 across models; `1‑α` drops to ≈ 0.69–0.74; performance degrades modestly for `2‑α`–`4‑α`, indicating robustness to accumulated feature errors.
  - Compatibility with other accelerations (Table 4):
    - Combining with gpt‑fast on LLaMA2‑Chat‑7B (RTX 3090): 160.4 tokens/s at int4, up from 106.9 tokens/s for gpt‑fast alone and 24.5 tokens/s vanilla FP16.
  - MoE (Mixtral 8×7B) result (Table 3):
    - Lower speedup of 1.50x and τ=3.25; verification touches multiple experts, reducing gains compared with dense models.
  - Throughput and batching (Table 7):
    - Even though the focus is latency, at the largest feasible batch sizes under fixed memory budgets, EAGLE doubles throughput (~1.97x for Vicuna‑7B; ~1.99x for LLaMA2‑Chat‑70B).
    - Speedups decrease as batch size increases (GPU becomes compute‑bound), but remain substantial.

- Ablations and diagnostics
  - Inputs to the draft model (Figure 8):
    - `feature&shifted-token` dominates `feature&unshifted-token`, `feature`, and `token` across speed, τ, and acceptance—highlighting the benefit of resolving sampling uncertainty.
  - Tree vs. chain drafts (Table 5; Figure 7):
    - Tree attention gives consistent gains in τ (+0.62 to +0.75) and speedup (+0.3 to +0.5).
  - Training data sensitivity (Table 6):
    - Using responses generated by the target LLM helps mildly (2.88x vs. 2.78x speedup on LLaMA2‑Chat‑7B), indicating low sensitivity and justifying training on fixed datasets for cost reasons.

- Do the experiments support the claims?
  - The central claims—higher acceptance and “lossless” acceleration at both T=0 and T>0—are supported by:
    - Exact verification protocol (Section 2; 3.3).
    - Broad model/task coverage with consistent τ>3 and 2–3.5x speedups (Figures 1–2; Tables 1–2).
    - Strong ablations isolating the effect of shifted tokens and tree drafting (Figure 8; Table 5).

## 6. Limitations and Trade-offs
- Access requirements
  - EAGLE needs access to the target LLM’s internal features and LM head weights to share embedding/LM‑head and compute/consume features (Section 3.1). This rules out closed APIs that expose only token probabilities.

- Model‑specific training
  - A small decoder layer (0.24B–0.99B parameters depending on target size) must be trained per target model (Section “Training”). Although training is relatively cheap (1–2 days on a single 3090 for 7B–33B; 4×A100 for 70B), it is not zero‑shot across architectures.

- Diminishing returns with batch size
  - As batch size increases, the speculative‑sampling advantage shrinks because the GPU becomes compute‑bound (Table 7).

- Reduced gains for MoE models
  - Verification on MoE models can require touching more experts per pass, limiting speedups (Table 3).

- Memory and implementation complexity
  - Tree attention and the draft tree introduce extra engineering complexity and memory overhead compared with chain‑only drafting, and the optimal tree shape is workload‑dependent (Appendix A.1).

- Scope of evaluation
  - Most latency results are with batch size 1 (the standard in this literature). Real‑world multi‑tenant serving may have different bottlenecks; while throughput gains are reported, broader system‑level evaluations would be valuable.

## 7. Implications and Future Directions
- How this changes the landscape
  - By reframing drafting at the feature level and conditioning on next tokens, EAGLE raises the ceiling of “lossless” speculative decoding—achieving 3x‑class speedups even for large 70B models (Figures 1–2; Table 1) without modifying the target model. This reduces the need for separate small draft models and makes speculative sampling practical for targets where no obvious draft exists.

- Follow‑up research enabled/suggested
  - Adaptive tree policies: Appendix A.1 uses a fixed tree; learning or dynamically adapting depth/branching by context or hardware budget could improve τ vs. compute trade‑offs.
  - Better robustness training: The noise augmentation in Section 3.2 helps; further techniques (e.g., adversarial feature perturbations, curriculum over error depth) may stabilize long drafts.
  - MoE‑aware verification: Exploit routing coherence across nodes to reduce expert fetches during verification and raise MoE speedups.
  - Cross‑model portability: Study how much of the autoregression head can transfer across related backbones to amortize training.
  - Long‑context regimes: Investigate memory management and caching strategies for tree attention with very long contexts.

- Practical applications
  - Lower‑latency chat assistants and code copilots without quality risk (lossless outputs).
  - Cost‑effective large‑model serving (notably for 70B‑class models) and on‑prem deployments; integration with quantization/compilation stacks like gpt‑fast already yields >6x tokens/s over vanilla FP16 (Table 4).
  - Scenarios where smaller draft models are unavailable or instruction templates differ (Figure 1 discussion), since EAGLE drafts from the target model’s own feature space.

> Representative results:
> - MT‑bench (T=0): 3.01x speedup for LLaMA2‑Chat‑70B; 2.90x–3.07x for Vicuna‑7B/13B; 2.78x–3.03x for LLaMA2‑Chat‑7B/13B (Figure 1).
> - Cross‑tasks: τ ≈ 3.2–4.5 tokens accepted per target pass; up to 3.76x speedup on HumanEval for LLaMA2‑Chat‑13B (Table 1).
> - Acceptance quality: `0‑α` ≈ 0.74–0.79 at T=0 across models; modest degradation with feature errors indicates robustness (Table 2).
> - Compatibility: 160.4 tokens/s on a single RTX 3090 when combined with gpt‑fast int4 (Table 4).

In summary, EAGLE’s two core ideas—feature‑level drafting and shifted‑token conditioning—turn speculative decoding into a practical, high‑gain, and distribution‑preserving accelerator for a wide range of LLMs.
