# The Mamba in the Llama: Distilling and Accelerating Hybrid Models

**ArXiv:** [2408.15237](https://arxiv.org/abs/2408.15237)

## 🎯 Pitch

This paper introduces a practical method to distill large, pretrained Transformer language models into linear RNN (Mamba) architectures by reusing attention weights, enabling the creation of hybrid models that retain most of the original model’s accuracy while drastically reducing inference costs. Additionally, it presents a hardware-aware speculative decoding algorithm tailored for RNNs, delivering significant speedups without sacrificing quality. This work unlocks the potential to deploy highly capable LLMs in environments constrained by memory and latency, paving the way for efficient, large-scale applications beyond what is possible with traditional Transformers.

---

## 1. Executive Summary
This paper shows how to turn a pretrained Transformer language model into a faster “linear RNN” model (Mamba) by reusing most of the Transformer’s weights, and how to further accelerate inference with a speculative decoding algorithm tailored to RNNs. With only modest distillation compute, the resulting hybrid models (mostly Mamba with some attention layers kept) retain much of the teacher’s capability, outperform prior Mamba models trained from scratch, and achieve substantial decoding speedups.

## 2. Context and Motivation
- Problem addressed
  - Transformers are accurate but slow and memory‑hungry at long sequence lengths because attention scales quadratically in length and needs a large key–value (`KV`) cache during generation (Introduction, p.1).
  - Linear RNNs such as `Mamba` achieve higher throughput and constant‑size state during decoding, but when trained from scratch they typically underperform top Transformers on downstream tasks (Introduction, p.1–2).
  - There has been no practical recipe to convert a large, pretrained Transformer into a competitive linear RNN model, nor a good speculative decoding method for RNNs (Sections 2–4).

- Why it matters
  - Many emerging LLM applications are bottlenecked by Transformer memory and latency: multi‑document reasoning, large codebases, and agentic workflows that need long context and large batch decoding (Introduction, p.1–2).
  - If Transformer knowledge can be transferred to a more deployment‑friendly architecture (RNN) without massive retraining, we can get similar quality with faster, cheaper inference.

- Prior approaches and gaps
  - Linear RNNs (Mamba, Mamba2, RetNet, RWKV, etc.) trained from scratch can rival Transformers at small scale, but large‑scale Transformer models are still best on many benchmarks (Introduction, p.1–2).
  - Previous distillation into attention‑free models (e.g., Hyena) either target small scales or incur large degradation (Section 6, Table 6 left).
  - Speculative decoding is well‑studied for Transformers but poorly suited to RNNs due to different state/caching behavior (Section 4.1).

- Positioning
  - This work provides: (1) an attention‑to‑Mamba weight mapping and initialization that immediately yields a viable hybrid model (Section 2.3, Algorithm 1; Figure 1), (2) a multi‑stage distillation pipeline aligned with common LLM post‑training (Section 3, Eq. (2), Eq. (4)), and (3) a hardware‑aware, multi‑step speculative decoding algorithm for linear RNNs and hybrids (Section 4.2, Algorithm 2; Figure 2).

## 3. Technical Approach
Step‑by‑step, the paper builds a path from a Transformer to a fast, accurate hybrid Mamba model and then speeds up its inference.

- From attention to a linear RNN: why a direct mapping is plausible (Section 2.1)
  - If we “linearize” attention by removing its softmax, the attention output at step `t` becomes a dot product between the query `Q_t` and a running sum of past key–value products. This can be written as a linear recurrence over a hidden state `h_t`.
  - In plain language: attention with softmax removed behaves like “accumulate `K_s V_s` over past tokens, then project by `Q_t` now.” Equation (1) in the paper gives the generic linear RNN form: `h_t = A_t h_{t-1} + B_t x_t`, `y_t = C_t^T h_t`.
  - The paper shows the linearized mapping:
    - hidden update: `h_t = m_{t-1,t} h_{t-1} + K_t V_t` (mask `m` enforces causality),
    - output: `y_t = (1/√D) Q_t^T h_t`,
    - and identifies `B_t`, `C_t`, `x_t` with the Transformer’s `W_K o_t`, `W_Q o_t`, and `W_V o_t` respectively (Section 2.1).
  - Problem: without the softmax nonlinearity, capacity drops and performance is poor. The solution is to expand the hidden state and add learnable continuous‑time dynamics using Mamba.

- Mamba as an expanded, learnable linear RNN (Section 2.2)
  - Mamba parameterizes a continuous‑time state‑space model with signals `A(k)`, `B(k)`, `C(k)` and a learned per‑step sampling interval `Δ_t`. A discretization function `Discretize(A, B, C, Δ)` produces the per‑token RNN parameters `Ā_t, B̄_t, C̄_t` used at inference.
  - Crucially, Mamba expands the effective hidden state by a factor `N'` (the “state expansion”); this restores modeling capacity with only modest parameter/compute overhead because the discretization, expansion, and recurrence are executed in a fused, hardware‑efficient kernel (Section 2.2).

- Attention‑to‑Mamba initialization and hybridization (Section 2.3; Algorithm 1; Figure 1)
  - Map Transformer attention weights directly into a Mamba block:
    - Use the Transformer’s linear projections: `W_Q`, `W_K`, `W_V` to compute per‑token vectors `C_t`, `B_t`, and `x_t` (Algorithm 1, steps 10–12).
    - Learn only the new Mamba‑specific parameters: the dynamic matrix signal `A` and the sampling schedule `Δ` (Algorithm 1, steps 13–14; Figure 1, green weights).
  - Replace some attention heads/blocks with these Mamba blocks, keeping the Transformer MLP layers intact (frozen in the first distillation stage). The paper explores hybrids with 50%, 25%, 12.5%, and 0% attention layers remaining (Section 2.3).
  - Stepwise replacement: distill with 50% attention kept, then push to 25%, etc. This “keep every n layers, then repeat” schedule helps stability (Section 2.3; Table 7 right shows perplexity benefits).

- Multistage distillation pipeline aligned with LLM post‑training (Section 3)
  - Stage 1: Distilled supervised fine‑tuning (SFT) with pseudo‑labels
    - Generate teacher outputs for instruction datasets, then train the student to match. The loss blends sequence‑level knowledge distillation (student maximizes likelihood of the teacher’s generated continuation) and token‑level KL divergence to match the teacher’s distribution (Eq. (2), with `α=1`, `β=0.1`).
    - During this stage, freeze the inherited Transformer MLPs so the Mamba blocks learn to “stand in for” the removed attention (Section 3).
  - Stage 2: Standard SFT on public instruction datasets (GenQA, InfinityInstruct, OpenHermes 2.5) with all parameters trainable (Section 5.1).
  - Stage 3: Distilled alignment via Direct Preference Optimization (`DPO`)
    - Use preference datasets (e.g., UltraFeedback, SimPO), but replace the usual “reference model” with the original Transformer teacher. The DPO objective encourages the student to prefer teacher‑preferred completions while regularizing by the teacher’s likelihoods (Eq. (4), Section 3).
    - The paper notes this is the first use of DPO explicitly as a distillation objective (Section 3).

- Speculative decoding adapted to linear RNNs and hybrids (Section 4; Figure 2; Algorithm 2)
  - Background: Speculative decoding uses a fast draft model to propose several next tokens, then a verifier model checks and accepts as many as possible in one go. Transformers verify quickly because they can score many steps in parallel from their `KV` cache. RNNs verify slowly if we must step token‑by‑token and maintain/copy large hidden states (Section 4.1).
  - Core idea: a hardware‑aware multi‑step kernel that recomputes multiple RNN steps without materializing intermediate states and can return both the hidden state “before conflict” and “after” in one pass:
    - `y_{j:k}, h_j, h_k ← MultiStep(h_i, y_{1:k}, i, j, k; A, B, C, Δ)` (Section 4.2).
    - This lets the verifier check a draft of length `K`, find the first mismatch, and then either advance to `h_k` (if all accepted) or roll back and continue from `h_j` (Algorithm 2; Figure 2).
  - Hybrid models: apply the RNN multi‑step verification to Mamba layers and parallel verification to the remaining attention layers (Section 4.2).
  - To perform well on fast GPUs (H100), the implementation fuses recomputation, multi‑step decoding, and caching into single kernels for both verifier and draft models, and uses a circular buffer for the convolutional part of Mamba (Section 4.3; Figure 3).

- Training and setup details (Section 5.1)
  - Distillation token budget is small relative to pretraining: ~20B tokens for the main 7B/8B models, “less than five days on 8×80G A100” per hybrid variant for Llama‑3 8B distillations; later Llama‑3.1/3.2 distillations take “eight days on 8×A100” or “four days on 8×H100” (Section 5.1).
  - Stage‑wise freezing: only the MLP layers are frozen in Stage 1; all parameters are trainable in later stages (footnote 2, p.8).

## 4. Key Insights and Innovations
- Weight‑reusing attention‑to‑Mamba initialization (Section 2.3; Figure 1; Algorithm 1)
  - What’s new: a direct mapping from a Transformer’s `Q/K/V` projections to Mamba’s `C/B/x` signals, plus new Mamba‑specific `A` and `Δ`.
  - Why it matters: it preserves the inductive biases and knowledge encoded in the teacher’s attention projections, yielding strong performance with little compute. Ablations show this initialization is critical:
    - Without attention‑based initialization, perplexity and downstream scores collapse (Table 8; LAMBADA ppl 6.20 vs 55.01; MT‑Bench 6.69 vs 1.04; LC‑win 14.11% vs 0.02%).
- Expanded hidden state via Mamba discretization (Section 2.2)
  - What’s new: rather than accept the low capacity of linearized attention, the method uses Mamba’s continuous‑time SSM with expansion factor `N'` to produce a richer discrete RNN per token without materializing huge tensors.
  - Why it matters: this retains efficiency while restoring modeling power, crucial for matching Transformer performance.
- Multistage, alignment‑aware distillation with DPO as a distillation objective (Section 3; Eq. (2), Eq. (4))
  - What’s new: combine pseudo‑label SFT + token‑level KL with a preference‑based objective that uses the teacher as the reference distribution in DPO.
  - Why it matters: improves downstream chat quality. Ablations show “SFT + DPO” outperforms either alone (Table 6 right; MT‑Bench improves for both 50% and 25% hybrids).
- Hardware‑aware speculative decoding for linear RNNs and hybrids (Section 4.2–4.3; Algorithm 2)
  - What’s new: a multi‑step verification kernel that recomputes and caches efficiently, and fused implementations that realize speedups even on H100 GPUs (Table 1).
  - Why it matters: brings speculative decoding’s speedups to RNNs, where naïve approaches fail due to state handling and kernel overheads.

## 5. Experimental Analysis
- Evaluation methodology and setup (Section 5)
  - Models
    - Teachers: Zephyr‑7B (Mistral‑based) and Llama‑3/3.1/3.2 instruct models at 3B and 8B scales (Section 5.1).
    - Students: hybrid `Mamba` and `Mamba2` with 50%, 25%, 12.5%, and 0% attention kept (“0%” is pure Mamba).
  - Distillation pipeline
    - Stage 1 pseudo‑label KD (UltraChat + UltraFeedback as seed prompts), Stage 2 SFT (GenQA, InfinityInstruct, OpenHermes2.5), Stage 3 DPO with teacher as reference (Section 5.1; Stage choices vary slightly with teacher).
  - Benchmarks and metrics
    - Chat: MT‑Bench (GPT‑4 graded), AlpacaEval 2 LC win rate and overall win rate vs GPT‑4 Turbo (Table 2).
    - General: LM Evaluation Harness zero‑shot tasks (e.g., MMLU, ARC‑C, HellaSwag, TruthfulQA; Table 3).
    - Open LLM Leaderboard/ZeroEval few‑shot (ARC‑C 25‑shot, HellaSwag 10‑shot, MMLU 5‑shot, Winogrande 5‑shot) and ZeroEval for GSM8K/CRUX (Table 4).
    - Long‑context: Needle‑in‑a‑Haystack retrieval accuracy across sequence lengths (Figure 4).
    - Speed: speculative decoding throughput and speedups on 3090 and H100 GPUs for pure Mamba (Table 1) and hybrids (Table 5).

- Main quantitative results
  - Chat quality retained or improved at 50% attention
    - > “Mamba‑Llama3 (50%) 8B achieves MT‑Bench 7.35 and AlpacaEval LC win 29.61%” (Table 2), compared to the teacher Llama‑3‑8B‑Instruct at “MT‑Bench 8.00 and AlpacaEval LC win 22.90%.”
    - Zephyr distills similarly: “Mamba‑Zephyr (50%) 7B MT‑Bench 7.31; AlpacaEval LC win 20.66% vs Zephyr 13.20%” (Table 2).
  - General benchmarks competitive with or better than large Mamba models trained from scratch
    - > On average across 10 zero‑shot LM‑Eval tasks, “Llama3.1‑Mamba2 (50%) DPO avg 65.31” beating NVIDIA Hybrid Mamba‑8B “59.60” and TRI Mamba‑7B “57.65” (Table 3).
  - Long‑context retrieval emerges naturally despite short distillation length
    - > “Distilled 3B models are perfect up to 10k context; 8B models are perfect up to 16k; one 8B hybrid remains strong to ~38k,” despite training on 2k contexts (Figure 4).
  - Speculative decoding speedups realized for Mamba and hybrids
    - Pure Mamba: on 7B verifier with Llama3 1B draft, H100 speedup ~2.0× and ~271 tokens/s; 2.8B verifier gets up to 421 tokens/s, 1.85× (Table 1; Figure 3 shows multi‑step kernel scales sublinearly with steps).
    - Hybrids: Zephyr‑hybrid speedups ≥1.8× with 2‑ or 4‑layer Transformer drafts; Llama‑hybrid ~1.58–1.6× (Table 5). The paper also reports “over 300 tokens/s for a Mamba‑7B model” with the optimized kernels (Abstract).
  - Ablations and robustness
    - More Mamba layers increase speed but can degrade quality; 0% attention (pure Mamba) drops substantially: e.g., LM‑Eval avg 54.74 vs 63.84 for 50% Mamba2‑Llama3 (Table 3), and AlpacaEval LC win 14.49% vs 26.78% (Table 2).
    - Stepwise replacement + interleaving helps perplexity: for 25% hybrid, perplexity 2.20 (with interleave) vs 2.89 (without) after Stage 1 (Table 7 right).
    - Initialization matters enormously:
      - Without attention‑init, 50% hybrid collapses (Table 8: LAMBADA ppl 55.01 vs 6.20; HellaSwag 27.91 vs 75.07).
      - Removing Mamba blocks entirely fails (Table 9: LAMBADA ppl 151.98; MT‑Bench 1.01).
    - Distillation compute is modest but effective: Perplexity rises gradually as more attention is removed (Table 6 left), but remains far better than prior distillations into Hyena at small scale (“Distill Hyena ppl ratio 2.36 vs teacher,” Table 6 left).
    - “SFT + DPO” beats each alone on MT‑Bench for hybrids (Table 6 right).

- Do the experiments support the claims?
  - Yes, for both quality and speed. At 50% attention, hybrids closely track or exceed teacher chat performance (Table 2) and surpass open‑source Mamba baselines on general tasks (Table 3). Speedups from speculative decoding are demonstrated on both Ampere and Hopper GPUs with careful kernel engineering (Table 1; Section 4.3).
  - The quality–speed trade‑off is transparent: pushing to 25% and 12.5% attention yields predictable drops (Tables 2–4), and pure Mamba lags by a wide margin—clarifying limits.

## 6. Limitations and Trade-offs
- Reliance on a strong teacher and its architecture
  - The approach assumes access to a high‑quality Transformer teacher and its attention projections. Results and the ablations (Tables 8–9) show the method’s strength comes from reusing those weights.
- Quality degrades as attention is removed
  - Pure Mamba (0% attention) lags markedly on both chat and general tasks (Tables 2–4). Hybridization is not merely a convenience; some attention seems necessary for top accuracy.
- Specialized kernels and hardware sensitivity
  - Achieving speedups on H100 required fused, hardware‑aware kernels (Section 4.3). Portability to other accelerators may require additional systems work.
- Distillation scope and data
  - Although compute is modest, the pipeline still uses ~20B tokens (Section 5.1) and multiple post‑training datasets. The paper does not explore domain shifts far from the teacher’s data distribution.
- Evaluation coverage
  - Chat scores (e.g., AlpacaEval) depend on GPT‑4‑based judging; while standard, they are not ground truth. Robustness beyond the tested tasks, safety, and factuality are not deeply analyzed.

## 7. Implications and Future Directions
- How this changes the landscape
  - It provides a practical recipe to “convert” large Transformers into deployment‑friendly linear RNN/hybrid models without retraining from scratch, preserving most capabilities while gaining speed and memory efficiency. This makes state‑space models a realistic target for production LLMs, not just research prototypes.

- Follow‑up research enabled/suggested
  - Pushing toward lower attention fractions without quality loss: larger state expansions `N'`, better discretization schemes, or new gating could close the gap to 0% attention.
  - Broader distillation objectives: combine DPO with task‑specific constraints (safety, calibration), or use multi‑teacher distillation.
  - Draft model design for hybrids: smaller, training‑free, or retrieval‑augmented drafts to improve acceptance rate and overall throughput (Section 5.5 notes draft size overhead).
  - Cross‑architecture transfer: apply the attention‑init idea to other sub‑quadratic models (RetNet, recurrent Gemma‑style models) or cross‑modal tasks.

- Practical applications
  - Long‑document assistants and codebase analysis where KV cache limits throughput (Introduction, p.1–2).
  - Agentic systems that value batch throughput and long context (Introduction, p.1–2).
  - Edge or cost‑constrained deployments that benefit from constant‑size RNN states during decoding.

> Representative results to remember:
> - “Mamba‑Llama3 (50%) 8B: MT‑Bench 7.35; AlpacaEval LC win 29.61% vs teacher 22.90%” (Table 2).
> - “On LM‑Eval avg, hybrids beat TRI Mamba‑7B and NVIDIA Hybrid Mamba‑8B trained from scratch” (Table 3).
> - “Speculative decoding speedups up to ~2× on H100 with fused kernels; Mamba‑7B reaches ~271 tokens/s in Table 1, and the abstract reports >300 tokens/s.”

Overall, this work offers a clear and implementable path to distill Transformer knowledge into fast, memory‑efficient linear RNNs, along with an inference algorithm that realizes speedups in practice.
