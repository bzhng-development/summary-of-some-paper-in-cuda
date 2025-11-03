# MEDUSA: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads

**ArXiv:** [2401.10774](https://arxiv.org/abs/2401.10774)

## 🎯 Pitch

MEDUSA introduces a novel framework that augments large language models with lightweight, parallel decoding heads and a tree-structured attention mechanism, enabling the model to propose and verify multiple next-token candidates simultaneously during inference. By eliminating the need for a separate draft model and integrating seamlessly into existing LLM architectures, MEDUSA delivers 2.3–2.8× faster inference without sacrificing output quality—dramatically improving LLM usability in latency-sensitive scenarios like chat and coding assistants. This approach democratizes high-speed LLM serving, making advanced AI more accessible and practical for a wide range of applications.

---

## 1. Executive Summary
MEDUSA is a simple way to speed up large language model (LLM) decoding by adding a few lightweight “decoding heads” on top of the model and verifying many next-token candidates in parallel with a tree-structured attention mask. With two training options—freezing the base model (MEDUSA-1) or jointly fine‑tuning it (MEDUSA‑2)—the method delivers 2.3–2.8× wall‑clock speedups without degrading output quality on common chat benchmarks (e.g., Fig. 3 and Table 2).

## 2. Context and Motivation
- Problem addressed
  - Auto-regressive decoding generates one token at a time, so each step re-reads the full model weights from High‑Bandwidth Memory (`HBM`) into on‑chip cache. This is memory‑bandwidth‑bound, not compute‑bound (Intro; references to Shazeer 2019; Kim 2023).
  - Even though modern accelerators have ample compute, the “1 token per forward pass” loop underutilizes them.  

- Why it matters
  - Lower latency and higher throughput are crucial for interactive applications (chat, coding assistants). Reducing the number of decode steps directly cuts memory traffic and speeds up serving.

- Prior approaches and their gaps
  - Speculative decoding uses a separate, smaller “draft” model to propose a block of tokens that the large model later verifies (Leviathan 2022; Chen 2023).
    - Gaps highlighted in Section 1: acquiring, training, and serving a separate draft model is complex; it may suffer distribution shift from the target model; multi‑model distributed serving is harder.
  - Earlier work on parallel decoding with multiple heads exists (Stern 2018), but was not developed into a practical LLM acceleration framework.

- Positioning
  - MEDUSA revisits multi‑head prediction but integrates it natively inside the original LLM: add small heads to the last hidden state, use a tree‑attention mask to verify many candidates in a single pass, and accept the longest valid prefix (Sections 2.1.1–2.1.2; Fig. 1–2).  
  - No draft model, minimal serving complexity, and optional joint fine‑tuning to maximize gains (Section 2.2).

## 3. Technical Approach
At a high level, each decoding iteration has three substeps (Section 2; Fig. 1):
1) generate candidate continuations, 2) process/verify them in parallel, and 3) accept a consistent prefix to append to the output.

Step-by-step details:

1) Candidate generation with MEDUSA heads (Section 2.1.1)
- What are MEDUSA heads? Extra, very small prediction modules connected to the last hidden state `h_t` at time `t`. The base LM’s head predicts token `t+1`. Head `k` is trained to predict token `t+k+1`, i.e., “look further ahead.”  
- Architecture: each head is a one‑layer feed‑forward network with SiLU activation and a residual connection:
  - p^(k)_t = softmax(W2^(k) · (SiLU(W1^(k) · h_t) + h_t)), with W2^(k) in R^{d×V}, W1^(k) in R^{d×d} (Section 2.1.1).
  - Initialization aligns head logits with the original LM head (W2 cloned from the LM head; W1 zero), so training starts from identical predictions and gradually specializes.
- Output: for each k, take the top‑`s_k` tokens (a small integer per head). Cartesian‑product combinations across heads form candidate sequences of length up to K+1 (Section 2.1.2).

2) Parallel processing with tree attention (Section 2.1.2; Fig. 2)
- Problem: verifying many candidates individually would inflate batch size and cost.
- Mechanism: construct a tree whose levels correspond to positions t+1, t+2, …; each level branches by the head’s top‑`s_k` tokens.  
- Attention mask: modify the causal mask so each token in a branch can only attend to its own ancestors (not to siblings or other branches). This allows a single forward pass to verify all candidates concurrently. Positional indices are adjusted accordingly (Fig. 2).
- Efficiency: the batch size remains 1; the model processes a “fan‑out” of tokens as a single structured sequence.  
- Total new tokens evaluated in one shot: sum over all tree nodes, i.e., Σ_{k=1..K} Π_{i=1..k} s_i (Section 2.1.2).

3) Acceptance step (Section 2.3.1)
- Goal: accept the longest prefix that is “consistent enough” with the base model.
- Two options:
  - Rejection sampling (as in speculative decoding) exactly matches the base model’s distribution but loses speed when sampling temperature increases (Section 2.3.1).
  - Typical acceptance (proposed here): accept token `x_{n+k}` if its probability under the original model `p_original(x_{n+k} | x_{≤n+k-1})` exceeds a threshold that depends on entropy,  
    p_original > min(ε, δ · exp(−H(p_original(·|context)))) (Section 2.3.1).  
    • Intuition: when the model is uncertain (high entropy), multiple continuations are “typical,” so the threshold lowers; when it is certain (low entropy), require higher probability.  
    • First token each step is greedily accepted to guarantee progress (Section 2.3.1).  
- The longest accepted candidate prefix is appended to the output; the next iteration starts from there (Fig. 1). The verification logits computed in step 2 are reused for the next step’s head predictions, avoiding extra passes (Section 2).

Training strategies (Section 2.2)
- MEDUSA‑1 (frozen backbone): train only the heads on cross‑entropy to the ground‑truth future tokens (Eq. (1)); weight farther‑ahead losses by λ_k (e.g., 0.8^k) to stabilize training.  
  • Backbones can be quantized to save memory (QLoRA‑style), enabling single‑GPU training (Section 2.2.1).  
- MEDUSA‑2 (joint training): jointly train heads and backbone with a combined loss (Eq. (2) = LM next‑token loss + λ_0·MEDUSA‑1 loss).  
  • Use higher LR for heads, lower LR for backbone; warm‑up the backbone or gradually increase λ_0 to protect base abilities (Section 2.2.2).  
  • Often yields higher acceptance rates and speedups.
- Self‑distillation when no training data exists (Section 2.3.2):  
  • Generate conversations using the target model on public prompts (e.g., ShareGPT; UltraChat).  
  • For MEDUSA‑2, train the backbone by matching the teacher distribution with a KL term `KL(p_original || p_student)` while training heads normally—implemented with a LoRA adapter so “teacher” = model with adapter off (no extra memory; Section 2.3.2).
- Tree construction optimization (Section 2.3.3):  
  • Estimate per‑head top‑i accuracies on a calibration set; greedily add nodes with highest expected accuracy gain to form a “sparse” tree that maximizes expected accepted length given a node budget (Appendix C; Fig. 4).

Practical choices
- Number of heads: up to five suffices; you can ignore extras at inference (Section 2.2.3).
- Metrics (Appendix B):  
  • `acceleration rate`: average tokens accepted per decode step (baseline = 1).  
  • `overhead`: per‑step latency vs. vanilla decoding.  
  • `speedup`: acceleration rate / overhead.

Why this design?
- It increases arithmetic intensity by turning one‑token steps into multi‑token steps without a draft model, hence avoiding serving complexity and distribution shift (Intro; Section 2.1.1).  
- Tree attention keeps computation compact, leveraging a single pass instead of many batched passes (Section 2.1.2).  
- Typical acceptance is tuned for practical sampling temperatures and maximizes acceptance length, unlike rejection sampling which can waste work (Section 2.3.1).

Hardware perspective (Appendix G)
- Roofline analyses show decoding is often memory‑bandwidth‑bound. MEDUSA raises operational intensity (FLOPs per byte moved) by verifying many tokens at once, shifting parts of the workload toward compute‑bound and improving throughput (Appendix G; Figs. 9–20).

## 4. Key Insights and Innovations
1) In‑model multi‑token prediction heads (Section 2.1.1; Fig. 1)
   - Novelty: reuses the base model’s last hidden state with thin heads to predict t+2, t+3, … directly, avoiding a separate draft model.  
   - Significance: simplifies deployment (no multi‑model serving), reduces risk of distribution mismatch, and enables parameter‑efficient training (MEDUSA‑1) or co‑optimization (MEDUSA‑2).

2) Tree‑structured attention masking for parallel candidate verification (Section 2.1.2; Fig. 2)
   - Novelty: a top‑down tree derived from heads’ top‑k predictions; within a single forward pass, each token only attends to its branch ancestors via a custom mask.  
   - Significance: processes many candidate sequences concurrently without increasing batch size, directly reducing the number of decode iterations.

3) Typical acceptance scheme (Section 2.3.1; Fig. 5)
   - Novelty: accept “typical” tokens based on an entropy‑aware probability threshold instead of strict rejection sampling.  
   - Significance: increases accepted lengths especially at non‑zero temperatures, improving wall‑clock speed while maintaining quality comparable to random sampling.

4) Self‑distillation without extra memory (Section 2.3.2)
   - Novelty: use LoRA adapters so the teacher is the same model with adapters off; train the backbone with a KL loss and the heads with cross‑entropy, all on model‑generated data.  
   - Significance: enables MEDUSA‑2 training for models whose original SFT data is unavailable (e.g., Vicuna‑33B, Zephyr‑7B).

5) Accuracy‑aware sparse tree construction (Section 2.3.3; Fig. 4)
   - Novelty: greedily select nodes that maximize expected acceptance length based on empirical per‑head top‑i accuracies.  
   - Significance: achieves the same (or better) acceleration with fewer nodes than naïve Cartesian‑product trees, controlling compute overhead.

Overall, items (1)–(3) are fundamental innovations for practical LLM decoding; (4)–(5) are strong engineering contributions that make the approach robust and deployable.

## 5. Experimental Analysis
Evaluation design
- Models: Vicuna‑7B/13B/33B (chat‑tuned Llama‑2 family) and Zephyr‑7B (SFT + alignment).  
- Training:
  - Vicuna‑7B/13B use ShareGPT data; Vicuna‑33B and Zephyr‑7B use self‑distillation (Section 3.2).
  - Sequence length 2048–4096; batch size mostly 1 at inference (the primary use case; Section 1).  
- Benchmarks and metrics:
  - MT‑Bench (multi‑turn conversational categories; Section 3).  
  - Quality via GPT‑4 judge (0–10 scale).  
  - Speed reported as tokens/sec and speedup (Appendix B).  
- Baselines: vanilla HuggingFace decoding; speculative decoding with open-source draft models (Appendix D).

Main results
- Overall speedups (Fig. 3a):  
  > Vicuna‑7B: 2.18× (MEDUSA‑1) and 2.83× (MEDUSA‑2) wall‑time speedups over baseline.  
  > Vicuna‑13B: 2.33× (MEDUSA‑1) and 2.83× (MEDUSA‑2).
- Per‑category speedups (Fig. 3b, Vicuna‑7B MEDUSA‑2):  
  > “Extraction” 3.62×; “Coding” 3.29×; others around 2.6–3.0×.  
  This suggests tasks with more predictable local structure benefit most.
- Head‑to‑head with speculative decoding (Table 1):
  > Vicuna‑7B: `S_MEDUSA = 2.83×` vs `S_SpecDecoding = 1.47×`.  
  > Vicuna‑13B: `2.83×` vs `1.56×`.  
  > Vicuna‑33B: `2.35×` vs `1.60×`.  
  Under identical hardware and settings, MEDUSA delivers larger speedups.
- Acceptance vs overhead breakdown (Table 1):
  > Acceleration rate ranges ~3.01–3.51 tokens/step, with per‑step overhead ~1.18–1.27× relative to vanilla; net speedups are 2.35–2.83×.  
- Quality (Table 1; Table 2; Fig. 5):
  > Quality differences on MT‑Bench are negligible: e.g., Vicuna‑7B MEDUSA‑2 6.18 vs baseline 6.17 (Table 2).  
  > MEDUSA‑1 even slightly improves (6.23). Direct end‑to‑end fine‑tuning without warm‑up hurts quality (5.925; Table 2), supporting the MEDUSA‑2 training recipe (Section 2.2.2).  
  > Fig. 5 shows typical acceptance maintains quality comparable to random sampling as the threshold increases, while enabling higher acceleration than greedy at T=0.7.
- Ablations and robustness:
  - Tree configuration (Fig. 4):  
    > An optimized sparse tree with 64 nodes achieves higher acceleration than random dense trees with 256 nodes and better speed due to lower compute overhead.  
    > Speed drops when candidate tokens become too many, showing a compute‑bound regime (Fig. 4b).
  - Typical acceptance thresholds (Fig. 5):  
    > Increasing the posterior threshold ε improves judged quality but reduces acceleration; curves allow trading speed for quality.
  - Contributions to speedup (Table 3 summary in Section 3.3.3):  
    > Heads without tree attention ≈1.5×; +tree attention ≈1.9×; +optimized tree ≈2.2×; +MEDUSA‑2 training ≈2.8×.
- Additional dataset results (Appendix F; Table 4):
  > On AlpacaEval prompts, tokens/sec improves from 37.07→106.76 (2.88×) for Vicuna‑7B, 29.01→91.54 (3.16×) for 13B, and 17.87→40.43 (2.26×) for 33B; Zephyr‑7B gains 2.91×.
- Hardware modeling (Appendix G):
  > Roofline plots (Figs. 9–17) and simulations (Figs. 21–23) confirm that increasing candidate tokens raises operational intensity and acceleration up to a point (often around 64 candidates), after which speedup saturates or declines due to compute overhead.

Assessment
- Evidence is strong that MEDUSA reduces decode steps and yields large wall‑time gains at batch size 1 with minimal quality impact across models and tasks (Sections 3.1–3.3; Figs. 3–5; Tables 1–3).  
- The acceptance/overhead analysis and ablations diagnose where speed comes from and where it can fall off (Appendix G; Fig. 4).

## 6. Limitations and Trade-offs
- Distributional fidelity vs. speed
  - Typical acceptance does not exactly match the base model distribution (Section 2.3.1). For applications demanding strict sampling fidelity (e.g., probabilistic evaluation, likelihood‑based tasks), rejection sampling would be safer but slower.
- Training data availability and quality
  - MEDUSA‑2 benefits from training data that matches the model’s output distribution; when self‑distilled data mismatches the hidden SFT data (e.g., Vicuna‑33B), speedup can be smaller or quality marginally shift (Section 3.2; Table 1).
- Compute overhead from large trees
  - As the number of candidate tokens grows, attention and linear layers become compute‑bound; speed can decline (Fig. 4b; Appendix G). Tuning tree size and sparsity is necessary.
- Focus on batch size 1
  - Most experiments target single‑request latency. While the method conceptually extends to larger batches (Discussion; final section), the paper’s empirical validation for large‑batch serving is limited.
- Engineering complexity
  - Requires custom attention masks, acceptance logic, and training recipes (two‑stage warm‑up, differential LRs). Although simpler than serving a separate draft model, it still adds components to standard decoding stacks.
- Memory and parameter overhead
  - The extra heads are small, but there is still some per‑step overhead (≈1.18–1.27×; Table 1). Gains rely on acceptance rate outpacing this overhead.

## 7. Implications and Future Directions
- How this changes the landscape
  - MEDUSA demonstrates that multi‑token prediction can be made native to the original model, delivering large practical speedups without the complexity of speculative decoding’s draft models. This reframes decoding acceleration as a lightweight architectural augmentation plus masking, not a multi‑model system problem.
- Follow‑up research enabled
  - Better head architectures or deeper multi‑step predictors trained with sequence‑level objectives.  
  - Adaptive, learned tree construction that conditions on context rather than using global calibration (beyond Section 2.3.3).  
  - Formal analysis of accepted‑prefix statistics under typical acceptance and their relation to calibration and entropy.  
  - Integration with other accelerations (KV‑cache compression, quantization, paged attention) and with high‑batch serving; joint scheduling of batch and MEDUSA tree size based on load (Appendix G suggests optimal candidate counts vary with batch, sequence length).
- Practical applications
  - Chat assistants, code generation, extraction, and structured generation: categories that showed the largest gains (Fig. 3b).  
  - On‑device or edge inference: speedups reduce energy use by cutting memory traffic; MEDUSA‑1 can be trained with quantized backbones on a single consumer GPU (Section 2.2.1).  
  - Models without public SFT data: self‑distillation (Section 2.3.2) offers a viable path to adoption.

> Bottom line: Section 3’s results and Appendix G’s analysis converge on a clear takeaway—equipping an LLM with a handful of future‑token heads and verifying them with tree attention can more than double decoding speed at minimal quality cost. The technique is simple to add, general across models, and tunable for different latency–quality targets.
