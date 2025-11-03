# Mamba: Linear-Time Sequence Modeling with Selective State Spaces

**ArXiv:** [2312.00752](https://arxiv.org/abs/2312.00752)

## 🎯 Pitch

Mamba introduces selective state space models (SSMs), enabling fully recurrent, linear-time sequence models to perform content-based reasoning by making their internal parameters input-dependent. This breakthrough allows the Mamba architecture to match or surpass Transformer-level performance on language, audio, and genomics tasks—while scaling efficiently to million-length sequences and providing up to 5× higher inference throughput. By combining principled selectivity with hardware-aware implementation, Mamba offers a compelling new foundation model backbone for applications where long context and computational efficiency are critical.

---

## 1. Executive Summary (2–3 sentences)
This paper introduces selective state space models (selective SSMs) and the Mamba architecture, a fully recurrent, linear-time alternative to Transformers that performs content-based reasoning while scaling to million-length sequences. It achieves Transformer-quality language modeling at up to 5× higher inference throughput and shows strong results across language, audio, and genomics by pairing an input-dependent SSM with a hardware-aware parallel implementation (Figures 4, 7, 8; Table 3).

## 2. Context and Motivation
- Problem addressed:
  - Transformers excel because attention can route information across a context window, but they:
    - Scale quadratically in sequence length during training and need a large key/value cache for inference.
    - Have a fixed context window and struggle with million-length sequences.
  - Subquadratic architectures (linear attention, gated convolutions, recurrent models, and prior structured state space models, SSMs) are efficient but have underperformed on dense, discrete modalities such as language.
  - Root cause identified: time‑invariant (LTI) SSMs cannot perform content-based selection—i.e., deciding which tokens to remember or ignore as the sequence unfolds (Section 3.1; Figure 2).

- Importance:
  - Practical: long-context applications (long documents, genomic sequences, long audio) demand linear scaling and fast inference (Abstract; Sections 1, 3.3).
  - Theoretical: shows how to restore content-based reasoning to efficient recurrent/convolutional mechanisms by making SSM parameters input-dependent, connecting to classical RNN gating (Theorem 1, Section 3.5.1).

- Prior approaches and gaps:
  - LTI SSMs (S4/S5 and variants) compute either as a recurrence or via a global convolution (Equations (2)–(3), Section 2). Their strengths are efficiency and long-range memory, but they lack content awareness.
  - Approximations to attention (linear attention, Hyena, RWKV, RetNet) trade accuracy for speed and still lag on language at scale (Figure 4).
  - Existing SSM architectures (e.g., H3, Hyena) insert SSMs as time-invariant blocks; they cannot filter out irrelevant tokens in an input-dependent way (Section 3.1).

- Positioning:
  - This paper introduces:
    1) a selection mechanism that makes key SSM parameters input-dependent (Algorithm 2);
    2) a hardware-aware parallel “selective scan” kernel to compute this efficiently (Section 3.3.2; Figure 1);
    3) a simplified end-to-end architecture, Mamba, that removes attention and even standalone MLP blocks (Section 3.4; Figure 3).

## 3. Technical Approach
This section explains what an SSM is, what “selectivity” changes, why that breaks convolutional efficiency, and how the paper recovers linear-time speed through a specialized parallel implementation and a new architecture.

- Baseline: structured SSMs (S4) in plain language
  - An SSM maps an input sequence `x` to an output sequence `y` by maintaining a latent state `h` with size `N` per channel. At each step, `h` is updated linearly using matrices `A` (state dynamics) and `B` (how inputs enter the state), and the output is read via `C` (Equations (1a)–(1b), (2a)–(2b), Section 2).
  - “Structured” means `A` has a form (often diagonal) that makes computation efficient; then each of `A`, `B`, `C` is represented by `N` numbers per channel (Section 2, “Structure and Dimensions”).
  - Discretization: continuous-time parameters `(Δ, A, B)` are turned into discrete-time `(Â,  B̂)` by a rule such as zero‑order hold (ZOH):
    > Equation (4): `Â = exp(ΔA)`, `B̂ = (ΔA)^{-1} (exp(ΔA) − I) · ΔB`
    Here `Δ` is a learnable “step size” (Section 2, “Discretization”).
  - LTI property: in standard S4/S5, `(Δ, A, B, C)` are constant over time; this linear time invariance allows two fast computations:
    - a linear recurrence (Equation (2)) for inference,
    - or an equivalent global convolution (Equation (3)) for training (Section 2, “Computation”).

- Why LTI is limiting
  - With constant dynamics, the model cannot “decide” which tokens to remember or ignore based on content (Section 3.1). This shows up on synthetic tasks like Selective Copying and Induction Heads, which require content-aware, variable spacing between relevant tokens (Figure 2).

- Selection mechanism: make the SSM input-dependent
  - Key idea: let certain parameters depend on the current token `x_t`. Concretely, compute
    - `B_t = s_B(x_t)`, `C_t = s_C(x_t)`, `Δ_t = τ_Δ(Parameter + s_Δ(x_t))`
    - where `s_*` are learned linear projections and `τ_Δ` is softplus to keep step sizes positive (Algorithm 2; Section 3.2).
  - This makes the model time‑varying, breaking the convolution equivalence (Equation (3) no longer applies). The model must be computed as a recurrence (a “scan”) but the recurrence is not purely sequential due to associativity (Section 3.2; 3.3).

- Why these selective parameters?
  - `Δ_t` acts like a gate controlling how much to reset vs. carry the state—large `Δ_t` “resets” and focuses on the current input; small `Δ_t` preserves the past (Section 3.5.2, “Interpretation of Δ”).
  - Theorem 1 shows with `N=1`, `A=-1`, `B=1`, `Δ_t=softplus(Linear(x_t))`, the update becomes:
    > Equation (5): `g_t = σ(Linear(x_t))`, `h_t = (1 − g_t) h_{t−1} + g_t x_t`
    i.e., a classical gated RNN update, derived from SSM discretization (Section 3.5.1).
  - Making `B_t` and `C_t` selective modulates what enters the state and what exits to the output (Section 3.5.2).

- Efficient computation: the “selective scan”
  - Naively, a time-varying SSM would materialize the full state tensor `h` of shape `(B, L, D, N)` (large), and would seem sequential.
  - The paper recovers linear-time training by:
    - Using an associative parallel scan to break the recurrence into parallel segments (Section 3.3.2; “parallel scan”).
    - Fusing operations to avoid writing expanded states to slow GPU HBM (high-bandwidth memory). Instead, load `(Δ, A, B, C)` to fast on-chip SRAM, discretize and scan within SRAM, and write only final outputs `(B, L, D)` back to HBM (Figure 1; Appendix D).
    - Recomputing intermediate states in the backward pass (checkpointing) to avoid saving `(B, L, D, N)` activations (Appendix D). This yields activation memory similar to a highly optimized Transformer with FlashAttention (Table 15).
  - Complexity:
    - Recurrent scan uses `O(B L D N)` FLOPs with low constants and linear scaling in `L`. In practice, the fused scan is 20–40× faster than a naïve PyTorch scan and beats FlashAttention-2 beyond length ~2K (Figure 8, left).

- Architecture: the Mamba block
  - Mamba replaces the usual “Attention + MLP” stack with a single homogeneous block that combines a gated MLP path and a selective SSM path (Figure 3; Section 3.4).
  - Design details (Section 3.4):
    - Expansion factor `E = 2` (so intermediate channels are 2× wider).
    - SiLU activation to match “SwiGLU”-style gating.
    - Typically two Mamba stacks per “layer” to roughly match the parameter count of a Transformer layer’s Attention+MLP (to keep comparisons fair).
    - Optional LayerNorm before/after as in modern recipes.

- Additional choices (Section 3.6):
  - Real vs complex SSMs: real-valued SSMs work well for text/DNA; complex SSMs can help continuous signals like audio (Section 3.6 and Figure 10).
  - Initialization and Δ projections: small, low-rank projections into Δ are effective; increasing Δ-projection rank modestly helps (Table 9).

## 4. Key Insights and Innovations
- Selectivity as the missing capability in efficient SSMs (fundamental)
  - Novelty: input-dependent `Δ_t`, `B_t`, `C_t` add content awareness to SSMs without attention (Algorithm 2).
  - Why it matters: enables filtering irrelevant tokens, variable spacing between events, and state resets at boundaries—capabilities LTI models lack (Section 3.5.2). This unlocks strong performance on discrete, information-dense modalities (text, DNA).

- Theory-to-mechanism bridge: gating emerges from SSM discretization (fundamental)
  - Theorem 1 derives a classic gated RNN update from an SSM with selective Δ, clarifying how “gates” arise from discretization (Section 3.5.1). This gives a principled foundation for gating and explains why `Δ_t` is the most important selective parameter (Table 7).

- Hardware-aware “selective scan” (fundamental + systems)
  - The fused, IO-aware kernel computes time‑varying SSMs in linear time without storing expanded states in HBM (Figure 1; Appendix D).
  - Empirical impact: 20–40× faster than a standard scan, and faster than FlashAttention-2 beyond length ~2K (Figure 8, left).

- A simple, homogeneous attention-free backbone (architectural)
  - The Mamba block merges gated MLPs with a selective SSM in one unit (Figure 3), simplifying the stack while matching the parameter budget of Transformer layers (Section 3.4). Despite its simplicity, it matches or exceeds Transformers at the 1.4–2.8B scale in language modeling (Table 3; Figure 4).

- Long-context capability in practice (capability)
  - Mamba trains and improves up to 1M context length for DNA and audio (Figures 5, 7). It also extrapolates on synthetic tasks (Induction Heads) from 256 to 1,048,576 tokens (Table 2).

## 5. Experimental Analysis
- Evaluation methodology
  - Modalities and datasets:
    - Language: The Pile for pretraining and standard zero-shot tasks (LAMBADA, HellaSwag, PIQA, ARC-E/C, WinoGrande) via the EleutherAI harness (Section 4.2; Table 3).
    - DNA: HG38 (human genome) for pretraining; fine-tuning on great apes species classification with up to 1M-token sequences (Sections 4.3.1–4.3.3; Figures 5–6; Table 13).
    - Audio: YouTubeMix (autoregressive next-sample prediction) and SC09 (unconditional speech generation) (Sections 4.4.1–4.4.2; Figure 7; Tables 4–5).
    - Synthetic: Selective Copying and Induction Heads (Sections 4.1.1–4.1.2; Tables 1–2; Figure 2).
  - Metrics:
    - Language: perplexity; accuracy on downstream tasks; scaling-law comparisons vs compute (FLOPs) (Figure 4).
    - DNA: perplexity for pretraining; finetune accuracy for species classification (Figures 5–6; Table 13).
    - Audio: bits per byte (BPB) for pretraining; FID, Inception Score (IS, mIS), and Audio-MSE (AM) for generation (Figure 7; Table 4).
    - Efficiency: kernel runtime, inference throughput, memory (Figure 8; Table 15).
  - Baselines:
    - Transformers (GPT-3 style and an improved “Transformer++” with RoPE, SwiGLU, RMSNorm, higher LR).
    - SSM/linear-time baselines: Hyena, H3, RetNet, RWKV, SaShiMi (Sections 2, 4.2, 4.4; Figure 4; Tables 4–5).
  - Training details:
    - Language scaling laws use Chinchilla-style token counts and model sizes ~125M to ~1.3B (Section E.2; Table 12).
    - DNA scaling controls tokens/batch across lengths and uses sequence-length warmup (Section 4.3.2 and E.3.3).
    - Audio pretraining controls tokens/batch across lengths (Figure 7; Table 14).
    - Efficient kernels used for Mamba; attention baselines use FlashAttention-2 where applicable (Figure 8; Table 15).

- Main quantitative results
  - Synthetic tasks (content selection ability):
    > Table 1 (Selective Copying): the selective SSM solves the task, e.g., “H3 S6” achieves 99.7% vs “H3 S4” 57.0%.  
    > Table 2 (Induction Heads): Mamba generalizes perfectly from training length 256 to 1,048,576, while attention baselines run out of memory beyond 16K.
  - Language modeling:
    - Scaling laws: Mamba is the first attention-free model to match a strong Transformer recipe (“Transformer++”) across 125M–1.3B and benefits more as sequence length increases from 2K to 8K (Figure 4).
    - Zero-shot evaluations (Pile tokenizer matched; 300B tokens):
      > Table 3: For each size, Mamba attains the best average across tasks.  
      > Example at ~1.4B: “Mamba-1.4B” average 59.7 vs “Pythia-1.4B” 55.2 and “RWKV-1.5B” 54.3; LAMBADA accuracy 64.9 and HellaSwag 59.1.  
      > At ~2.8B: “Mamba-2.8B” average 63.3 vs “Pythia-2.8B” 59.1 and “RWKV-3B” 59.6.
    - Pile perplexity also improves over Pythia at matched sizes (Table 3, “Pile ppl”).
  - DNA:
    - Scaling by model size at 1K context: Mamba outscales HyenaDNA and Transformer++ (Figure 5, left).
    - Scaling by context length (1K → 1M) at ~1.3–1.4M params with tokens/batch controlled:
      > Figure 5 (right): Mamba’s pretraining perplexity improves monotonically with longer context up to 1M, while HyenaDNA degrades.
    - Finetune classification (great apes), long context:
      > Table 13 and Figure 6: At 1,048,576 length, “Mamba 7M” reaches 81.31% vs “HyenaDNA 1.4M” 54.87%.
  - Audio:
    - Long-context pretraining (BPB):
      > Figure 7: Mamba improves over SaShiMi across sequence lengths and the gap widens at long sequences (to nearly 1M samples).
    - SC09 generation:
      > Table 4: Mamba 6.1M achieves FID 0.94 vs SaShiMi 1.99 and improves IS/mIS (6.26 / 88.54 vs 5.13 / 42.57). Larger Mamba (24.3M) further reduces FID to 0.67.
      > Table 5 (ablations): With the same 6M budget, using Mamba blocks in the outer and center UNet stages yields the best FID (0.94) and mIS (88.54).
  - Efficiency:
    > Figure 8 (left): the fused scan is 20–40× faster than a standard scan and faster than FlashAttention-2 beyond length ≈2K.  
    > Figure 8 (right): Mamba’s inference throughput is 4–5× higher. For example, at batch size 64 and prompt length 2048, “Mamba 1.4B” reaches ~1688 tokens/s vs “Transformer 1.3B” ~490 tokens/s.  
    > Table 15: Training memory of 125M models is comparable to an optimized Transformer with FlashAttention-2.

- Ablations and robustness
  - Which selective parameters matter (Table 7): Δ is most important, but combining Δ, B, C gives the best perplexity (10.93 → 8.71).
  - Choice among LTI SSMs matters little; selectivity matters a lot (Table 6): swapping S4 variants barely changes performance, but moving from S4 to S6 gives a large gain (e.g., H3 S4 10.30 → H3 S6 8.95 perplexity).
  - State dimension `N` helps only when B and C are selective (Table 10): at `N=16`, perplexity improves from 9.81 (non‑selective) to 8.71 (selective) with negligible parameter increase.
  - Δ-projection rank (Table 9): even a 1‑dimensional projection helps (9.12 → 8.97); larger ranks yield incremental gains (down to 8.71).
  - Audio parameterization (Figure 10): on raw waveforms, the selective mechanism can hurt; LTI (Mamba-S4 or complex S4) performs better in early stages, supporting a “continuous vs. discrete” inductive bias trade-off.

- Do the experiments support the claims?
  - The synthetic tasks isolate content selection and demonstrate the mechanism clearly (Tables 1–2).
  - Language results are competitive up to 2.8–3B parameters and show consistent zero-shot gains (Table 3). Scaling laws (Figure 4) are convincing at ≤1.3B, with the caveat that very large-model regimes remain untested here.
  - Long-context claims are supported with controlled-token DNA/audio studies up to 1M length (Figures 5, 7) and a difficult downstream DNA classification task (Figure 6; Table 13).
  - Efficiency claims are backed by kernel-level and end-to-end measurements (Figure 8; Table 15).

## 6. Limitations and Trade-offs
- Scale of language models:
  - Results top out at ~2.8–3B for downstream and ~1.3B for scaling-law plots (Figure 4). It remains open whether Mamba maintains or widens its advantage at 7B–70B+ (Section 5, “Scaling”).

- Modality-dependent inductive bias:
  - For continuous signals (e.g., raw audio), LTI SSMs can outperform selective SSMs in lower layers (Figure 10). Selection is most beneficial for discrete, information-dense data (text, DNA). This “no free lunch” trade-off across the continuous–discrete spectrum is explicitly discussed (Section 5).

- Implementation complexity:
  - The speedups rely on a custom fused kernel that exploits the GPU memory hierarchy and parallel scan, plus recomputation in backprop (Figure 1; Appendix D). Portability and maintenance across hardware/software stacks may be nontrivial.

- Baseline coverage at very long lengths:
  - Some attention baselines run out of memory beyond 16K in the synthetic extrapolation (Table 2), limiting direct apples-to-apples comparison for million-length contexts. However, DNA and audio experiments do compare long contexts while controlling compute (Figures 5, 7).

- Downstream LLM affordances not fully evaluated:
  - While zero-shot reasoning tasks are covered (Table 3), properties like instruction following, RLHF, tool use, retrieval augmentation, and in-context learning breadth at large scale are open questions (Section 5, “Downstream Affordances”).

## 7. Implications and Future Directions
- How this changes the field
  - Establishes a credible, attention-free backbone that attains Transformer-level quality in language modeling while delivering linear-time scaling and high inference throughput (Figures 4, 8; Table 3). This reframes the design space: content-based reasoning does not require attention if a selective, input-dependent SSM is paired with the right systems implementation.

- Follow-up research directions
  - Scale studies:
    - Train Mamba at 7B–70B+ with strong data/compute recipes to test scaling laws and downstream abilities (Section 5).
  - Hybrid architectures:
    - Combine selective SSMs with attention or retrieval to mix content-aware recurrence with explicit non-local routing where helpful (Section E.2.2 shows Mamba+MHA is slightly better but not necessary).
  - Modality-tailored stacks:
    - For continuous signals, keep early layers LTI (Mamba-S4 or complex S4), add selectivity at higher, more symbolic levels (Figure 10).
  - Memory and state design:
    - Explore learned boundary resets, episodic memory, or expandable state mechanisms for even longer effective horizons (Section 3.5.2, “Boundary Resetting”).
  - Theory:
    - Extend the discretization–gating connection to richer families of selective dynamics; analyze stability and gradient flow of time-varying SSMs at scale (Theorem 1; Sections 3.5, 3.6).

- Practical applications
  - Long-document and streaming LMs with low latency and high throughput (constant-time per step without KV cache; Figure 8).
  - Genomics: pretraining over entire genomes and long-context tasks like regulatory element modeling and species classification up to 1M tokens (Figures 5–6; Table 13).
  - Audio: efficient long-context modeling and speech generation, with state-of-the-art automated metrics on SC09 (Figure 7; Table 4).
  - Any on-device or cost-sensitive deployment where linear-time training and fast recurrent inference are decisive.

Overall, the paper’s core contribution is twofold: it shows that “selectivity” (input-dependent SSM parameters, especially Δ) restores content-aware reasoning to linear-time models, and it makes this practical through a fused, IO-aware parallel scan. The resulting Mamba architecture is simple, fast, and competitive across diverse domains, with clear strengths on long-context and discrete modalities and a principled path for further scaling and hybridization.
