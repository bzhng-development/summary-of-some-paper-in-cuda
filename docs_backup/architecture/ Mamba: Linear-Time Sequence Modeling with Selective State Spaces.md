# Mamba: Linear-Time Sequence Modeling with Selective State Spaces

**ArXiv:** [2312.00752](https://arxiv.org/abs/2312.00752)
**Authors:** Albert Gu, Tri Dao
**Institutions:** Carnegie Mellon University, Princeton University

## 🎯 Pitch

The paper introduces selective state space models ('S6') and the 'Mamba' architecture, which leverage input-dependent dynamics to achieve Transformer-quality sequence modeling with linear time complexity. This advancement not only allows for efficient processing of up to million-length contexts across modalities like language and genomics but also provides 5× higher inference throughput, making it a significant breakthrough in long-context sequence modeling and hardware efficiency.

---

## 1. Executive Summary (2–3 sentences)
This paper introduces a new class of sequence models, called selective state space models (`S6`), and a simple backbone architecture built from them, `Mamba`. The core idea is to make the state-space dynamics depend on the current input (“selection”), then compute them with a hardware-aware, parallel scan so training and inference run in linear time while matching or exceeding Transformer-level quality on language, audio, and genomics (e.g., 5× higher inference throughput and strong scaling to million-length contexts; see Figure 8 and Sections 4.2–4.4).

## 2. Context and Motivation
- Problem addressed
  - Modern foundation models largely use Transformers, which route information well within a finite context but suffer quadratic time and memory in sequence length and require a growing key–value (`KV`) cache at inference. Prior subquadratic models (linear attention, convolutions, earlier SSMs) are efficient but have struggled to match Transformers on discrete, information-dense modalities like text (Section 1).
  - A key missing capability in those efficient models is content-based reasoning: deciding, based on the current token, what should be remembered or ignored as the sequence progresses (Sections 3.1 and 3.2).
- Why it matters
  - Long-context workloads are proliferating in language, audio, time series, and genomics. Efficient models that can both scale linearly and match Transformer quality unlock longer contexts (up to 1M tokens on real data, Section 4.3) and faster deployment (5× inference throughput without a KV cache, Section 4.5).
- Where prior approaches fall short
  - Most efficient SSMs are linear time-invariant (`LTI`): their parameters do not change over time. LTI recurrences and global convolutions can keep track of time but cannot select which inputs matter based on the content (Figure 2; Section 3.1). This blocks tasks like selective copying or associative recall.
  - Convolution-based SSM training avoids materializing large per-step latent states but depends on time invariance; once input-dependence is introduced, those efficient convolutions break (Section 3.3).
- Positioning relative to existing work
  - This paper removes the LTI constraint by making core SSM parameters input-dependent (“selective”), then replaces convolutional training with a memory- and compute-efficient recurrent computation via a fused, parallel scan (Sections 3.2–3.3). The resulting `Mamba` architecture merges the SSM path with the usual MLP path into a single, homogeneous block (Figure 3; Section 3.4).

## 3. Technical Approach
This section builds from the SSM basics to the selective mechanism, the hardware-aware algorithm, and the final architecture.

- Background: what an SSM computes and why previous ones were fast
  - An SSM maps an input sequence `x(t)` to an output sequence `y(t)` through a latent state `h(t)` (Section 2). In continuous form:
    - `h'(t) = A h(t) + B x(t)` and `y(t) = C h(t)` (Eq. 1).
  - After discretization with a step `Δ` (e.g., Zero-Order Hold, Eq. 4: `Ā = exp(ΔA)`, `B̄ = (ΔA)^{-1}(exp(ΔA) − I)·ΔB`), you get either:
    - A recurrence: `h_t = Ā h_{t−1} + B̄ x_t`, `y_t = C h_t` (Eqs. 2a–2b), or
    - An equivalent global convolution: `y = x * K` with `K = (C B̄, C Ā B̄, C Ā^2 B̄, …)` (Eqs. 3a–3b).
  - Prior SSMs are LTI: `Δ, A, B, C` are constant across time (Section 2, “Linear Time Invariance”). LTI enables very fast training via convolution (FFT), avoiding materializing the latent state of size `(B, L, D, N)` where `N` is the SSM state dimension (Figure 1 and “Structure and Dimensions” in Section 2).

- The central change: selection (input-dependent dynamics)
  - The paper makes key SSM parameters functions of the current input token `x_t` (Algorithm 2; Section 3.2):
    - `B_t = s_B(x_t)`, `C_t = s_C(x_t)`, and `Δ_t = τ_Δ(Parameter + s_Δ(x_t))`.
    - Concretely: `s_B(x) = Linear_N(x)`, `s_C(x) = Linear_N(x)`, `s_Δ(x) = Broadcast_D(Linear_1(x))`, `τ_Δ = softplus`.
  - Intuition: `Δ_t` controls how much to “reset” the state at step `t`. Large `Δ_t` focuses on the current input by damping the previous state; small `Δ_t` preserves the state and largely ignores `x_t` (Section 3.5, “Interpretation of Δ”).
  - Connection to gating: with a particular one-dimensional setup (`N=1`, `A = −1`, `B=1`), the recurrence reduces to a gated RNN update (Theorem 1; Eq. 5):
    - Define `g_t = σ(Linear(x_t))`; then `h_t = (1 − g_t) h_{t−1} + g_t x_t`.
    - This formalizes the selection mechanism as the principled, discretized version of a gate (Sections 3.5.1 and C).

- Why convolution no longer applies and how the model stays fast
  - Once `Ā_t, B̄_t, C_t` vary with time (because `Δ, B, C` are functions of `x_t`), the convolutional equivalence (Eqs. 3a–3b) breaks; only the recurrent path (Eqs. 2a–2b) remains (Algorithm 2, Step 6).
  - Naïvely unrolling the recurrence is memory- and bandwidth-heavy because the latent state has shape `(B, L, D, N)`. The paper avoids this with a hardware-aware algorithm (Section 3.3 and Appendix D):
    - Fused kernel: load small base parameters `(Δ, A, B, C)` from GPU HBM into fast on-chip SRAM, perform discretization and the recurrence in SRAM, then write back only the final outputs `(B, L, D)` to HBM (Figure 1; Section 3.3.2).
    - Parallel scan: despite nonlinearity from input dependence, the recurrence can be composed with an associative parallel scan to remove strict sequential bottlenecks (Section 3.3.2).
    - Recomputation in backward: do not save intermediate states `(B, L, D, N)`; recompute them in the backward pass, so activation memory matches highly optimized attention (FlashAttention) (Appendix D; Section 4.5).
    - Chunking: if a full sequence doesn’t fit in SRAM, process it in chunks and continue the scan across chunks (Appendix D).

- The `Mamba` block and network
  - The architecture merges the “H3-like” SSM path and the MLP path into a single block, removing separate attention or separate MLP blocks (Figure 3; Section 3.4).
    - Input projection expands the model dimension by factor `E` (they use `E=2`).
    - A short convolution + selective SSM (the main path), gated/activated with SiLU, and a final projection.
    - Optional LayerNorm is used similarly to RetNet (Section 3.4).
  - Two such Mamba blocks are stacked to roughly match the parameter count of a standard Transformer layer (MHA + MLP) (Section 3.4).
  - Practical details (Section 3.6):
    - Real-valued SSMs suffice for discrete modalities like text and DNA; complex-valued versions can be better for continuous signals (audio).
    - Initialization follows diagonal SSM variants (e.g., S4D-Real), though random initializations can also work (Table 8).
    - `Δ` can be made a low-rank function of `x` by projecting to a small dimension `R` then to `D` (Section 3.6).

- A concrete “how it works” mental model
  - Think of `Δ_t` as a per-step “valve” that decides how much the model keeps from the past vs. how much it focuses on the current token. `B_t` decides how inputs write into the state; `C_t` decides how the state reads out to the output. Because all three depend on `x_t`, the model can ignore fillers (“um”), reset at document boundaries, or remember a key token until it’s needed (Section 3.5, with examples “Variable Spacing,” “Filtering Context,” and “Boundary Resetting”).

## 4. Key Insights and Innovations
- Selective SSMs: input-dependent `Δ, B, C` (Algorithm 2; Section 3.2)
  - What’s new vs. prior SSMs: prior SSMs were LTI for efficiency. Here, the dynamics depend on content while retaining linear-time computation via a new scan kernel.
  - Why it matters: enables content-based selection and variable spacing along the sequence, solving tasks LTI models couldn’t (Selective Copying, Induction Heads; Figure 2, Tables 1–2).
- Hardware-aware selective scan (Section 3.3.2 and Appendix D)
  - What’s new: fuse discretization + recurrence in SRAM, parallelize with scan, and use recomputation to avoid storing expanded states.
  - Why it matters: training throughput comparable to or faster than FFT-based methods; faster than state-of-the-art attention kernels at long lengths (Figure 8, left). Makes input-dependent SSMs practical at scale.
- A simplified, homogeneous `Mamba` block (Figure 3; Section 3.4)
  - What’s new: merges the SSM and MLP paths into one block (inspired by GAU), removing separate attention and MLP blocks.
  - Why it matters: fewer moving parts, easier scaling, and strong empirical performance that matches or beats strong Transformer recipes (Figure 4; Table 3).
- Principle: selection as the foundation for compressive, long-context sequence modeling (Section 3.1 and 3.5)
  - Conceptual shift: rather than store everything (like attention) or compress blindly (like LTI recurrences), selectively decide what to keep. This leads to monotonic gains with longer context on real data (DNA, audio; Figures 5 and 7), and constant-time-per-token inference without a KV cache (Section 4.5).

## 5. Experimental Analysis
- Evaluation setup (Sections 4.1–4.6; Appendices E.1–E.5)
  - Modalities: synthetic tasks, language modeling (The Pile), DNA (HG38), audio (YouTubeMix, SC09).
  - Baselines: LTI SSMs (S4, Hyena), linear-attention-based architectures (H3, RetNet, RWKV), and strong Transformer recipes (Transformer++).
  - Metrics: accuracy on synthetic tasks, perplexity or bits-per-byte (BPB) for pretraining, zero-shot accuracy for downstream NLP tasks, audio generation metrics (FID, Inception Score), and system metrics (latency/throughput, memory).

- Synthetic tasks demonstrate the new capability (Section 4.1)
  - Selective Copying (variable spacing): LTI models struggle; adding selection solves it.
    - Table 1: full Mamba with `S6` inner layer reaches 99.8% accuracy; H3 with `S6` reaches 99.7%; LTI variants underperform (e.g., S4 without selection: 18.3%).
  - Induction Heads (associative recall): generalization and extrapolation to much longer sequences.
    - Table 2 and Table 11: trained at length 256, `Mamba` achieves perfect generalization all the way to length 1,048,576 (2^20). Attention baselines are memory-limited to ≤ 16k and degrade sharply beyond train length.

- Language modeling: scaling laws and downstream quality (Section 4.2)
  - Scaling laws on The Pile (Figure 4)
    - At sequence length 2k and 8k, for 125M–1.3B models, `Mamba` matches a very strong “Transformer++” recipe and beats all other subquadratic methods (Hyena, H3++, RWKV, RetNet).
    - > “Mamba is the first attention-free model to match the performance of Transformer++ … particularly as the sequence length grows” (Figure 4).
  - Zero-shot downstream (Table 3)
    - Across sizes (~130M to ~2.8B), `Mamba` outperforms Pythia and RWKV on every listed task. Example at ~2.8B:
      - LAMBADA accuracy: 69.2 vs. Pythia-2.8B 64.7; HellaSwag normalized acc: 66.1 vs. 59.3; average across tasks 63.3 vs. Pythia-2.8B 59.1.
    - At 3B scale, `Mamba` often matches or exceeds Transformers roughly twice its size (see discussion beneath Table 3).

- DNA modeling: benefits of longer context appear clearly (Section 4.3)
  - Size scaling at short context (Figure 5, left): at ~40M parameters and length 1024, `Mamba` matches or beats HyenaDNA and Transformer++, often with 3–4× fewer parameters for the same perplexity.
  - Context-length scaling (Figure 5, right): with a ~1.4M-parameter model, `Mamba` improves steadily from length 1k up to 1M; HyenaDNA degrades with longer context (computation-controlled setting).
  - Downstream classification of five great apes (harder than prior species set) (Figure 6; Table 13):
    - At 1M context, `Mamba-7M` fine-tunes to 81.31% accuracy vs. HyenaDNA-1.4M at 54.87%.

- Audio modeling and generation (Section 4.4; Appendix E.4)
  - Long-context autoregressive pretraining on YouTubeMix (Figure 7): `Mamba` (complex SSM for this task) achieves lower BPB than SaShiMi (S4+FFN), and the gap widens at longer contexts up to ~1M samples.
  - SC09 speech generation (Table 4): `Mamba` substantially improves fidelity and diversity over autoregressive, GAN, and diffusion baselines at comparable or smaller parameter counts.
    - Example (smaller model ~6M params): FID 0.94 vs. SaShiMi 1.99 and WaveNet 5.08; Inception Score 6.26 vs. SaShiMi 5.13.
    - Larger `Mamba` (~24M) further improves to FID 0.67 and mIS 144.9.

- Efficiency results (Section 4.5; Appendix E.5)
  - Training-time kernel timing (Figure 8, left): the fused selective scan is 20–40× faster than a PyTorch scan and outpaces FlashAttention-2 beyond length ~2k (with causal mask).
  - Inference throughput (Figure 8, right): at prompt length 2048 and generate length 128, `Mamba-1.4B` achieves up to ~1814 tokens/s at batch size 128, vs. Transformer-1.3B ~515 tokens/s.
    - Quote: “Mamba achieves 4–5× higher inference throughput than a Transformer of similar size …”
  - Memory (Table 15): `Mamba` training memory is comparable to an optimized Transformer with FlashAttention-2; two SSM blocks match the activation memory of one attention + one MLP layer (Appendix D).

- Ablations clarify what drives the gains (Section 4.6)
  - Architecture vs. inner layer (Table 6): swapping among LTI inner layers (S4 real/complex, Hyena) barely changes perplexity; making the inner layer selective (`S6`) yields large improvements. The Mamba block performs similarly or slightly better than H3 while being simpler.
  - Which parameters should be selective? (Table 7): `Δ` is most important (perplexity 9.81 with only `Δ` selective vs. 10.93 with none), and combining selective `Δ, B, C` gives the best results (8.71).
  - State dimension `N` matters only when `B, C` are selective (Table 10): increasing `N` from 1 to 16 improves perplexity massively (from 9.73 to 8.71) when `B, C` are selective, but barely helps when they are not.
  - Δ projection size (Table 9): even a rank-1 projection helps a lot; increasing to 64 yields further gains with small parameter growth.
  - Audio caveat (Figure 10): on raw, continuous audio, an LTI SSM (“Mamba-S4”) can outperform selective SSM (“Mamba-S6”) in the outer U-Net layers; using selection in the center blocks works well. This supports the “continuous vs. discrete” inductive-bias claim (Section 5).

- Overall assessment
  - The experiments are broad, include strong baselines and careful compute control (especially for context-length scaling in DNA and audio), and contain informative ablations. Claims about content-based reasoning are grounded in synthetic tasks (Tables 1–2). System claims are backed by kernel benchmarks (Figure 8) and memory analysis (Table 15).

## 6. Limitations and Trade-offs
- Assumptions and design choices
  - The effectiveness of selection hinges on discretization (`Δ`) behaving like a gate (Theorem 1) and on the model learning when to reset vs. persist; this depends on data characteristics and training signals (Sections 3.5 and 4.6).
  - The hardware speedups presume access to GPUs where SRAM vs. HBM IO dominates and where fused kernels and parallel scan are available (Section 3.3.2; Appendix D).
- Modalities and inductive bias
  - Selection is crucial for discrete, information-dense data (language, DNA). For smoothly varying, uniformly sampled signals (raw audio), LTI behavior can be preferable in early stages (Figure 10; “No Free Lunch: Continuous-Discrete Spectrum” in Section 5).
- Scale and ecosystem
  - Results stop at ~3B parameters for NLP, below the scale of state-of-the-art LLMs. It remains to be shown whether `Mamba` maintains or widens its advantage at 7B–70B+ with instruction tuning, RLHF, tool use, etc. (Section 5, “Scaling”).
- Implementation complexity
  - The approach relies on specialized fused kernels, recomputation strategies, and chunked scans. Portability across frameworks/hardware and ease of integration in diverse training stacks may vary (Appendix D; Section 4.5).
- Not covered
  - The work does not address retrieval-augmented inference, sparse routing, or hybrid attention–SSM mixtures at the largest scales (though brief ablations suggest combining Mamba with MHA is slightly beneficial; Figure 9, left).

## 7. Implications and Future Directions
- How this shifts the field
  - It provides a credible attention-free alternative that delivers Transformer-level quality with linear time and memory in sequence length, constant-time per-token inference, and strong long-context behavior (Figures 4–7). This broadens the design space for foundation models and suggests that content-based selection plus efficient recurrence can substitute dense attention in many settings.
- Follow-up research enabled
  - Scaling studies beyond 7B parameters and across multi-trillion-token training.
  - Hybrid models: interleave or compose `Mamba` blocks with sparse or local attention to combine global retrieval with selective recurrence (Figure 9 hints that mixtures can help).
  - Theory: stability and expressivity analysis of input-dependent discretizations; characterizing when selection vs. LTI is optimal across the continuous–discrete spectrum (Section 5).
  - Systems: generalized selective-scan kernels, CPU/TPU implementations, kernel autotuning, and integration with compiler stacks.
- Practical applications
  - Long-context NLP (million-token analysis), streaming inference without KV caches (chatbots with 5× throughput; Figure 8), genomics modeling at nucleotide resolution over chromosome-scale contexts (Figures 5–6), and audio generation or long-range audio understanding with compute-controlled training (Figure 7).
  - Any deployment where memory is tight and latency matters: on-device models, real-time speech or sensor processing, and reinforcement learning where episode boundaries benefit from selective resets (Section 3.5, “Boundary Resetting”).

> In short: by turning the SSM’s “clock” and input/output couplings into content-aware functions and pairing that with a fused, parallel scan, `Mamba` delivers linear-time sequence modeling that matches Transformer quality while scaling to million-length contexts and substantially improving inference throughput (Sections 3–4; Figures 4–8).
