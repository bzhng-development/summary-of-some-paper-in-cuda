# Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality

**ArXiv:** [2405.21060](https://arxiv.org/abs/2405.21060)

## 🎯 Pitch

This paper unveils a deep mathematical connection between state space models (SSMs) and attention mechanisms by showing they are two perspectives on the same class of structured matrices—semiseparable matrices. Leveraging this duality, the authors introduce the SSD (Structured State Space Dual) framework and a new Mamba-2 architecture, achieving unprecedented efficiency: up to 8× faster than prior SSMs and on par with optimized attention for long sequences. This unified theory bridges two major modeling paradigms, enabling more scalable, hardware-friendly sequence models without sacrificing accuracy, and promises new directions for both algorithmic research and practical deployment of long-context neural networks.

---

## 1. Executive Summary (2-3 sentences)
This paper unifies state space models (SSMs) and attention through a single mathematical object—semiseparable matrices—and uses that duality to design faster, scalable algorithms and a new architecture, `Mamba-2`. It shows that a broad class of SSMs and “linearized” attention are two views of the same structured matrix transformation, enabling a hardware-efficient `SSD` (Structured State-Space Dual) algorithm that is 2–8× faster than prior SSM implementations and competitive with optimized attention at long sequences (Section 6; Figure 10), while matching or exceeding language modeling baselines (Sections 9.2–9.3; Figure 9; Tables 1 and 10).

## 2. Context and Motivation
- Problem/gap addressed:
  - Training attention scales quadratically with sequence length and requires caching activations growing linearly with length at inference (Section 1). Recent SSMs (e.g., `Mamba`) scale linearly and have constant-size recurrent state but have been developed somewhat separately from the attention ecosystem, making them harder to analyze and optimize (Sections 1–2.1).
  - Prior “linear attention” derived a limited duality between attention and linear recurrences (Section 2.4), but did not explain selective SSMs nor yield the best possible hardware efficiency or a unifying theory.

- Why it matters:
  - Practical: Long-context models are constrained by attention’s cost; SSMs promise linear scaling but need robust theory, algorithms, and systems support to become mainstream.
  - Theoretical: A common framework lets us transfer proofs, algorithms, and systems techniques between attention and SSMs (Figure 1).

- Prior approaches and limitations:
  - Standard attention uses softmax over all pairs, yielding quadratic cost (Section 2.2).
  - Linear attention variants drop softmax or approximate kernels to enable associative reordering (Section 4.1.3) but typically assume a fixed causal mask and do not characterize when efficient autoregression is possible (Sections 4.2–4.3).
  - SSMs offer linear scaling but require careful structure in their transition matrices and specialized implementations to reach high efficiency (Sections 2.1, 3.4.1; Remark 3).

- Positioning:
  - The paper proposes a general duality—Structured State-Space Duality (`SSD`)—that shows:
    1) SSMs are exactly semiseparable matrix transformations (Theorem 3.5; Figure 2).
    2) Masked attention with a wide class of structured masks (Structured Masked Attention, `SMA`) has an efficient recurrent form, and the largest such class with bounded-order autoregression is precisely the semiseparable masks (Definition 4.2; Theorem 5.2).
    3) A new `SSD` algorithm that combines both views and runs mostly as matrix multiplications, making SSMs as hardware-friendly as attention (Section 6; Listing 1; Theorem 6.1).

## 3. Technical Approach
This section unpacks the methodology step-by-step, introducing concepts only when needed.

A. Start from a sequence transformation
- A sequence layer maps `X ∈ R^(T,P)` to `Y ∈ R^(T,P)` along the sequence axis `T` (Definition 2.1).
- Many sequence layers can be written as a single matrix multiplication along the sequence axis: `Y = M_θ X` (Definition 2.3).

B. SSMs as sequence transformations and their matrix form
- Selective SSM (time-varying SSM) is defined by the recurrence (Equation 2):
  - `h_t = A_t h_{t-1} + B_t x_t` and `y_t = C_t^T h_t`
  - `N` is the state size (hidden dimension).
- Key step: Unroll the recurrence to get a closed-form matrix mapping across the sequence (Equation 3):
  - `y_t = Σ_{s=0..t} C_t^T (A_t ⋯ A_{s+1}) B_s x_s`
  - Equivalently, there exists a lower-triangular matrix `M` with entries `M_{j,i} = C_j^T A_j ⋯ A_{i+1} B_i` such that `Y = M X`.

C. Semiseparable matrices and the SSS representation
- Definition (3.1): A lower-triangular matrix is `N`-semiseparable if every submatrix on/below the diagonal has rank ≤ `N`.
- Sequentially semiseparable (SSS) representation (Definition 3.2): Exactly the parametrization `M_{j,i} = C_j^T A_j ⋯ A_{i+1} B_i`.
- Equivalence: Every `N`-SSS matrix is `N`-semiseparable (Lemma 3.3), and every `N`-semiseparable matrix admits an `N`-SSS representation (Proposition 3.4).
- Main equivalence (Theorem 3.5): The SSM transformation (Equation 2) is exactly multiplication by an `N`-SS semiseparable matrix in SSS form (Figure 2).

D. The scalar case and the “cumprodsum”
- For `N = 1`, the SSS matrix simplifies (Equation 6):
  - `M_{j,i} = a_j × … × a_{i+1}` where each `A_t` is a scalar `a_t`.
  - Multiplication `y = M x` is equivalent to the scalar recurrence `y_t = a_t y_{t-1} + x_t` (Equation 7).
  - This operation generalizes cumsum to “cumulative product of sums” (“cumprodsum”) and will be the primitive used later (Appendix B).

E. Rewriting masked attention as a tensor contraction and generalizing it
- Masked attention with queries/keys/values is typically written in three matrix steps (Equation 11): `G = Q K^T`, mask `M = G ◦ L`, then `Y = M V`.
- The exact same computation is one 4-way tensor contraction (Equation 12):
  - `Y = contract(TN, SN, SP, TS → TP)(Q, K, V, L)`
- “Linear attention” arises by reordering the contractions (Equation 15):
  - Expand features: `Z = contract(SP, SN → SPN)(V, K)`
  - Apply the mask: `H = contract(TS, SPN → TPN)(L, Z)`; if `L` is causal (lower-triangular 1’s), this reduces to feature-wise cumsum.
  - Contract back: `Y = contract(TN, TPN → TP)(Q, H)`
- This yields an `O(T)` recurrent form whenever multiplication by `L` is subquadratic.

F. Structured Masked Attention (SMA)
- Definition (4.2): Use any structured mask `L` with subquadratic matvec to define masked attention by (Equation 12). Two evaluation modes:
  - Quadratic mode: the usual attention evaluation order (Equation 13).
  - Linear mode: the reordered contractions (Equation 15), accelerated by the structured multiplication by `L`.
- Examples (Figure 3): causal mask (linear attention), decay/Toeplitz masks (RetNet-like), Fourier masks, and (importantly) 1-semiseparable masks.

G. The Duality (SSD): When SSMs are attention and vice versa
- Scalar-identity SSMs (Section 5.1): If `A_t = a_t I` (same scalar on the diagonal), then `M = L ◦ (C B^T)` with `L = 1SS(a)` (Equation 16). This is exactly masked kernel attention in quadratic mode.
- 1-SS structured attention (Section 5.2): If `L` is a 1-semiseparable mask (generalizing the causal mask), then the linear-mode computation (Equation 15) is a special case of a diagonal SSM where all diagonal entries of `A` are the same scalar (Corollary 5.1).
- Characterization (Theorem 5.2; Appendix C.2): Any masked attention that is an autoregressive process of bounded order must use a semiseparable mask. In other words, “efficient autoregressive attention = semiseparable SMA.”
- Summary (Figure 4): SSMs (linear-time) and SMA (quadratic attention) intersect in a large class of dual models, `SSD`, with matching linear and quadratic forms.

H. The hardware-efficient `SSD` algorithm (Section 6)
- Idea: Multiply by the SSM’s semiseparable matrix `M` using a block decomposition that combines both modes:
  - Partition the sequence into `Q`-length chunks, yielding a `(T/Q) × (T/Q)` grid of `Q × Q` blocks (Section 6; “Block Decomposition”).
  - Diagonal blocks: compute intra-chunk outputs using the dual quadratic mode (attention-like) with matrix multiplications (Section 6.1).
  - Off-diagonal blocks: use low-rank factorizations intrinsic to semiseparable matrices (Equation 5; Section 6.2), decomposed into:
    - Right factors (input→state per chunk): `B`-block factors via batched matmuls (contract `QN × QP → NP`).
    - Center chain (state→state across chunks): a 1-SS multiplication (scalar SSM scan) over `(N,P)` channels and length `T/Q` (Section 6.2; Appendix B).
    - Left factors (state→output per chunk): `C`-block factors via batched matmuls.
- Complexity and hardware friendliness:
  - If `N = P = Q`, the total training FLOPs are `O(T N^2)`, inference FLOPs `O(N^2)`, and memory `O(T N)` (Theorem 6.1).
  - Most work is batched matrix multiplications on `(N × N)` matrices; the scan is cheap and parallelizable (Section 6.3).
- Minimal implementation: a complete PyTorch reference is provided (Listing 1), demonstrating the algorithm’s simplicity.

I. The `Mamba-2` architecture (Section 7)
- Block design changes (Figure 6; Section 7.1):
  - Parallel parameter projections: produce `A, B, C, X` in one shot at the block start (like `Q, K, V` in attention), which reduces parameters and enables tensor parallelism (Section 8.1).
  - Extra normalization: add a normalization (e.g., GroupNorm/RMSNorm) after the multiplicative gate to improve stability (NormFormer-style; Section 7.1).
- Head patterns for SSMs (Section 7.2; Equations 17–20):
  - Introduce multi-head designs analogous to MHA/MQA/MVA. `Mamba` corresponds to “multi-input SSM” (MIS), analogous to multi-value attention (Proposition 7.2).
  - Grouped variants (GVA/GIS) also supported to match tensor-parallel groups.
- Optional kernel features (Section 7.3):
  - Insert feature maps (`ψ`) on `B, C` (and optionally `X`) to mimic linear-attention kernels (Swish is default; Tables 6–7). Optional attention-like normalization can be recovered by augmenting `X` with a column of ones.

J. Systems support (Section 8; Figure 7)
- Tensor parallelism: single all-reduce per block, parity with attention/MLP layers, by moving projections up-front and using GroupNorm (Section 8.1; Figure 7 left).
- Sequence/context parallelism: split sequence across devices by passing chunk boundary states—cost grows linearly with devices (Figure 7 right)—simpler than ring attention’s quadratic key-query interactions (Section 8.2).
- Variable-length batching: handle mixed-length sequences without padding by zeroing `A_t` at sequence boundaries (Section 8.3).

## 4. Key Insights and Innovations
- Unifying equivalence between SSMs and semiseparable matrices (fundamental)
  - Novelty: Provides an exact matrix characterization of SSMs (Theorem 3.5) and shows that even unstructured real SSMs can be computed in `O(TN)` after preprocessing (Theorem 3.7), bridging recurrent and structured-matrix worlds (Sections 3.1–3.4).
  - Significance: Enables transferring structured-matrix algorithms and theory directly to SSM computation and analysis.

- Generalization of linear attention to Structured Masked Attention (SMA) and its dual linear form (fundamental)
  - Novelty: Derives masked attention as a single tensor contraction and shows two dual computation orders (Equations 12–15; Proposition 4.1), then generalizes the mask to any structured matrix with fast matvec (Definition 4.2; Figure 3).
  - Significance: Clarifies when attention admits efficient autoregression (Theorem 5.2): exactly when the mask is semiseparable.

- State Space Duality (`SSD`) and the hardware-efficient SSD algorithm (core practical innovation)
  - Novelty: Identifies the intersection of SSMs and SMA—1-SS masks / scalar-identity SSMs (Section 5; Figure 4)—and proposes a block-decomposition algorithm that combines matmul-heavy quadratic blocks with cheap recurrent glue (Section 6; Listing 1).
  - Significance: Achieves optimal FLOP/memory scalings (Theorem 6.1), leverages tensor cores, and empirically delivers 2–8× speedups vs optimized SSM scans and competitiveness vs FlashAttention-2 at long sequences (Figure 10).

- Mamba-2: architecture and systems design for large-scale training (incremental but impactful)
  - Novelty: Parallel projections for `A, B, C, X`, head-structure taxonomy (MIS/MCS/MES/MHS), and GroupNorm placement make SSM blocks TP-friendly with one all-reduce (Sections 7–8; Figures 6–7).
  - Significance: Brings SSMs into parity with Transformer training pipelines, supports sequence parallelism, and enables much larger state sizes with minimal slowdown (Figure 10 right).

## 5. Experimental Analysis
Evaluation setup and baselines
- Pretraining and downstream:
  - Dataset: The Pile (Section 9.2; Figure 9; Tables 1 and 10).
  - Model sizes: ~125M to 2.7B; Transformer++ and Mamba baselines; same tokenizer/dataset for fair Pile perplexity comparisons (Sections 9.2–D.3).
  - Metrics: Validation perplexity and zero-shot accuracy on LAMBADA, HellaSwag, PIQA, ARC-E/C, WinoGrande, OpenBookQA (Tables 1, 3, 10).
- Synthetic capability: Multi-Query Associative Recall (MQAR), a difficult phone-book lookup style task requiring memorizing multiple key-value pairs (Section 9.1; Figure 8).
- Efficiency: Microbenchmarks vs FlashAttention-2 and Mamba’s fused scan (Section 9.3; Figure 10).

Main results (with specifics)
- Scaling laws on The Pile (Section 9.2.1; Figure 9):
  - Quote: “Mamba-2 matches or exceeds the performance of Mamba as well as a strong ‘Transformer++’ recipe… and is Pareto dominant on perplexity, FLOPs, and wall-clock time.”
  - The log–log plot shows Mamba-2 slightly below Transformer++ across FLOPs, indicating better perplexity per FLOP for sequence length 8192.

- Zero-shot evaluations (Sections 9.2.2, D.3; Tables 1 and 10):
  - At ~780M–1.3B scale (NeoX tokenizer, 300B tokens), Mamba-2 typically beats Mamba and rivals or exceeds Pythia models with up to 2× parameters.
  - Example (Table 1, 2.7B scale): Mamba-2-2.7B achieves LAMBADA ppl 4.10 and average score 60.2, outperforming Mamba-2.8B (avg 59.9) and Pythia-2.8B (55.7).

- Hybrid model studies (Section 9.2.3; Tables 2–3):
  - Adding a small fraction of attention layers to an SSD stack yields the best perplexity at fixed parameter count.
  - Table 2 (350M model, 48 layers, 7B tokens): ~10% attention layers (e.g., 4–7 layers) achieves the best perplexity (as low as 8.26), improving over pure Mamba-2 (8.60) and pure Transformer++ (8.68).
  - Table 3 (2.7B, 300B tokens): Mamba-2-Attention (58 SSD + 6 attention) attains the best Pile ppl 5.95 and best average 61.0 across tasks, improving over both pure Mamba-2 and Transformer++.

- Synthetic MQAR (Section 9.1; Figure 8):
  - Mamba-2 substantially outperforms Mamba (even at same state size `N=16`) and attention across sequence lengths 256–1024 and model dims 32–256.
  - Larger states improve performance: Mamba-2 with `N=64` and `N=256` clearly increases accuracy curves.

- Efficiency benchmarks (Section 9.3; Figure 10):
  - Left: SSD is 2–8× faster than Mamba’s fused scan at state size `N=64` and surpasses FlashAttention-2 beyond sequence length ≈2K.
  - Right (seq len 4K): Mamba scan time scales linearly with `N`; SSD remains nearly flat as `N` grows to 256, showing large-state scalability.

- Ablations
  - Block design (Table 4): Parallel projections and extra normalization each help; best perplexity 11.49 for “Parallel + Extra Norm” vs 11.76 for the sequential Mamba-1-style block (125M-scale).
  - Head structure (Table 5): MIS/MVA (Mamba-style) clearly outperforms MQA/MKA patterns at equal total state size; e.g., at 361.8M params, MIS achieves ppl 8.73 vs MQA at 9.33.
  - Kernel approximations (Tables 6–7): Swish or none typically outperform exp/ReLU/cosFormer/PRF; Taylor/ReBased mappings do not help; LayerNorm on QK (here on `B,C`) is competitive at 130M and 380M but not consistently better.

Assessment
- The experiments support the main claims:
  - The `SSD` algorithm is faster and scales to larger state size with minimal slowdown (Figure 10).
  - `Mamba-2` is at least competitive with well-tuned Transformer++ on perplexity and wins on some downstream averages at the same token budget (Figure 9; Tables 1 and 3).
  - Hybridization with a small number of attention layers is synergistic (Tables 2–3), consistent with the theoretical duality (Figure 4).
- Robustness:
  - System ablations (TP, sequence parallel) are explained architecturally (Section 8) and reflected in design choices, though large-cluster wall-clock benchmarks are summarized qualitatively.
  - Kernel approximation ablations clarify that softmax-mimicking tricks are unnecessary or unstable in `SSD` (Tables 6–7).

## 6. Limitations and Trade-offs
- Expressivity vs hardware efficiency in `A_t`:
  - `SSD`’s fastest path assumes `A_t` is scalar times identity (Section 5.1), stricter than the diagonal `A_t` in prior SSMs (e.g., `Mamba`). This could limit per-time-step dynamics compared to full diagonal, though the dual algorithm and head patterns compensate in practice.
  - Extending the same matmul-friendly algorithm to general diagonal `A_t` remains open (Section 10.1).

- Not softmax attention:
  - `SSD/SMA` omits softmax normalization (Sections 4.1.3, 7.3). While normalization can be reintroduced by augmenting `X` (Section 7.3), `SSD` primarily targets kernelizable attention. Tasks that benefit from sharp softmax retrieval might still prefer a small number of attention layers—consistent with hybrid results (Tables 2–3).

- Preconditions for `O(TN)` SSM:
  - Theorem 3.7 guarantees `O(TN)` computation only after a potentially expensive preprocessing if `A_t` is unstructured; in practice, useful SSMs rely on structured `A_t` (diagonal/scalar-identity) to avoid that cost (Section 3.4).

- Short-sequence efficiency:
  - At shorter contexts, training time can still favor Transformers because they interleave very efficient MLPs with attention, whereas a pure SSD stack has only SSD blocks (Section 9.3). Mixing SSD with MLP layers mitigates this (Section 9.2.3).

- Stability considerations:
  - Larger models needed an extra normalization inside the block for stable training (Section 7.1; Table 4). Kernel approximations with normalization were sometimes unstable (Table 6).

- Scope of SMA characterization:
  - Theorem 5.2 assumes bounded-order autoregression; tasks requiring unbounded “content-based jumps” might still call for some explicit attention.

## 7. Implications and Future Directions
- How this changes the landscape
  - Provides a single, principled lens—semiseparable matrices—for both SSMs and efficient attention, clarifying when and why linear-time forms exist (Theorems 3.5, 5.2; Figures 2–4). This is a foundation for designing long-context models that are both linear-time and systems-friendly.

- Research directions enabled
  - Explore other `SMA` masks (Toeplitz, Fourier) to encode richer positional or frequency structure with efficient recurrent forms (Figure 3).
  - Extend the `SSD` algorithm to broader `A_t` structures (full diagonal) while keeping matmul dominance (Section 10.1).
  - Non-causal and bidirectional variants using the matrix-mixer view; inverse/closure properties (Appendix C.1) suggest principled constructions (e.g., invertible semiseparable mixers).
  - Theoretical analysis of in-context learning and copying for SSD vs attention, leveraging the autoregressive characterization (Theorem 5.2) and recent benchmarks (Section 9.1).

- Practical applications
  - Long-context language models with lower memory and compute, especially for streaming or on-device settings where constant-state inference is valuable (Sections 1, 8.2).
  - Training at scale with standard parallelism stacks (TP/SP) thanks to the Mamba-2 block design (Section 8; Figure 7).
  - Hybrid stacks combining SSD with a small fraction of attention layers to balance retrieval and compression (Tables 2–3).

> In the authors’ code release, `Mamba-2` and the `SSD` layer target production use (Section 9), with Listing 1 offering an end-to-end reference implementation and the system sections detailing how to integrate with tensor/sequence parallelism.

Overall, this paper contributes a unifying theory, a practical algorithm, and an architecture that together make SSMs a first-class, hardware-efficient alternative (and complement) to Transformers for long-context modeling.
