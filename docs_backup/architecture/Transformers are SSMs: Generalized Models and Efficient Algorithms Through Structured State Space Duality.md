# Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality

**ArXiv:** [2405.21060](https://arxiv.org/abs/2405.21060)
**Authors:** Tri Dao, Albert Gu
**Institutions:** Carnegie Mellon University, Princeton University

## 🎯 Pitch

This paper bridges the gap between Transformers and state-space models (SSMs) through the concept of semiseparable matrices, introducing the novel "State Space Duality" framework. By offering the Mamba-2 architecture, which is 2–8× faster than existing SSMs and competitively efficient compared to optimized Transformers, this work significantly enhances the practicality of long-sequence modeling, enabling more efficient training and inference in large language models.

---

## 1. Executive Summary (2–3 sentences)
This paper connects two major sequence-modeling families—Transformers (attention) and state‑space models (SSMs)—through a single mathematical object: semiseparable matrices. It introduces a duality (“SSD”) that yields new algorithms and a practical architecture (Mamba‑2) that are 2–8× faster than prior SSMs while remaining competitive with optimized Transformers on language modeling and long‑sequence tasks.

## 2. Context and Motivation
- Gap addressed
  - Transformers excel at language modeling but scale quadratically with sequence length in training and require linear‑sized caches in autoregressive generation, which becomes a bottleneck for long contexts (Introduction).
  - Recent SSMs (e.g., S4, Mamba) scale linearly and have constant memory per step but have evolved separately from the Transformer ecosystem, making them harder to optimize and less hardware‑friendly (Section 1; Section 2.1 “Selective SSMs”).
  - Prior connections between attention and RNNs/SSMs (e.g., Linear Attention) exist but are limited; they do not explain the full relationship nor yield the fastest possible SSM implementations (Section 2.2; Section 4).

- Why it matters
  - Long‑context training and inference dominate costs in LLMs. Bridging attention and SSMs promises models that retain Transformer‑level quality with better asymptotic and wall‑clock efficiency (Section 1; Figure 10).

- Prior approaches and limitations
  - Softmax attention: expressive but quadratic in sequence length T (Section 2.2).
  - Linear Attention (LA): uses kernel tricks and associative re‑ordering to reduce complexity, but the “masking with causality” part was not generalized, and its theoretical recurrent form is usually sketched rather than derived cleanly (Section 4.2).
  - SSMs (S4, S4D, Mamba): linear complexity but training relied on specialized kernels; selective SSMs (time‑varying A, B, C) improved quality but still under‑utilized matrix‑multiplication units on GPUs (Section 2.1, “Selective SSMs”).

- Positioning of this work
  - Provides a unifying framework—Structured State Space Duality (SSD)—that: 
    1) exactly matches SSMs with a well‑studied class of structured matrices (semiseparable matrices), and 
    2) generalizes linear attention to Structured Masked Attention (SMA), showing when and how attention has efficient recurrent forms (Sections 3–5).
  - Uses this bridge to design both a fast algorithm (SSD block‑decomposition multiplication) and a practical TP‑friendly architecture (Mamba‑2) (Sections 6–8).

## 3. Technical Approach
This section unpacks how the paper connects SSMs and attention, and how the resulting algorithm works.

A. From SSMs to Semiseparable Matrices (Section 3)
- Definitions (selective and minimal)
  - SSM recurrence (time‑varying/selective form): 
    - `h_t = A_t h_{t-1} + B_t x_t`, `y_t = C_t^T h_t` (Eq. (2)).
  - Key step: Writing the entire input–output sequence map as a single lower‑triangular matrix `M` so that `y = M x` (Eq. (3)).
- Semiseparable matrices (Definition 3.1)
  - A lower‑triangular matrix is N‑semiseparable if every submatrix below the diagonal has rank ≤ N. This compresses a T×T matrix into O(NT) parameters and allows fast multiplication (Section 3.2; Proposition 3.6).
- SSS (sequentially semiseparable) representation (Definition 3.2)
  - Each entry of `M` factorizes as `M_{j i} = C_j^T A_j … A_{i+1} B_i` (Eq. (4)).
- Main equivalence
  - Theorem 3.5: the sequence transformation of an SSM is exactly multiplication by an N‑semiseparable (SS) matrix in SSS form.
  - Implication: all techniques for fast semiseparable multiplication become fast ways to compute SSMs.
- Scalar case (“1‑SS”) and cumprodsum (Section 3.2.2)
  - If the state is 1‑dimensional, `M_{j i} = a_j … a_{i+1}` (Eq. (6)), and multiplication `y = M x` equals the recurrence `y_t = a_t y_{t-1} + x_t` (Eq. (7)), a generalization of cumulative sums/products.

B. From Masked Attention to Structured Masked Attention (SMA) (Section 4)
- Attention as a 4‑way tensor contraction
  - With queries `Q`, keys `K`, values `V`, and a mask `L`, masked attention is:
    - `Y = contract(TN, SN, SP, TS → TP)(Q, K, V, L)` (Eq. (12)).
  - Standard quadratic implementation computes `G = Q K^T`, masks `M = G ◦ L`, then multiplies `Y = M V` (Eq. (13)).
- Linear attention via re‑ordering contractions (Eq. (15))
  - Compute `Z = contract(V, K)` (expansion), then multiply by mask `L` (cumsum for causal masking), then contract with `Q`. This gives an O(T) recurrent form when `L` is causal (Proposition 4.1).
- Generalization: SMA (Definition 4.2)
  - Replace the causal mask `L` with any structured matrix that supports fast multiplication (Toeplitz, Fourier, semiseparable, etc.). SMA inherits a quadratic form (Eq. (13)) and a dual sub‑quadratic form (Eq. (15)).

C. Duality: a shared subfamily (Section 5)
- Scalar‑identity SSM equals masked kernel attention
  - If each `A_t` is a scalar times identity (so all diagonal entries are equal), the SSM’s matrix `M` can be written as `M = L ◦ (C B^T)` with `L` a 1‑SS mask (Eq. (16)). This is exactly SMA with an SS mask.
- Recurrent vs. quadratic “duals”
  - The linear‑time SSM recurrence and the quadratic masked attention computation are two ways to multiply by the same semiseparable matrix.
- Which masked attentions are autoregressive?
  - Theorem 5.2 (Appendix C.2): any masked kernel attention that supports bounded‑order autoregression must have a semiseparable mask `L`. In short, efficient autoregressive attention is semiseparable SMA.

D. The SSD Algorithm: hardware‑efficient block‑decomposition (Section 6)
- Goal
  - Combine the benefits of linear SSM (O(T)) and quadratic attention (GPU‑friendly matmuls).
- Core idea (Figure 5)
  - Block the T×T semiseparable matrix `M` into Q×Q tiles. 
    - Diagonal blocks: compute with the quadratic (attention‑like) form (parallelizable, uses matmuls).
    - Off‑diagonal blocks: are low‑rank by semiseparability; factor them into three parts:
      - Right factors (“input → state”), Center factors (“state → state”), Left factors (“state → output”).
  - The Center factors reduce to a shorter scalar SSM scan across blocks (1‑SS multiplication), which is cheap after blocking.
- Complexity (Theorem 6.1; Section 6.3)
  - With state expansion N and head size P = N:
    - Training FLOPs: `O(T N^2)`.
    - Inference FLOPs: `O(N^2)`.
    - Inference memory: `O(N^2)`.
    - Dominated by batched matrix multiplications; only a small 1‑SS scan remains.
  - Table under “Comparison to Pure SSM and Attention” (Section 6.3): SSD has same asymptotic FLOPs as linear SSMs but is much more hardware‑efficient, and beats quadratic attention for long sequences.

E. The Mamba‑2 architecture (Section 7) and systems (Section 8)
- Block design (Figure 6; Section 7.1)
  - Parallel projections: compute `A, B, C, X` in parallel at the start (rather than deriving `A, B, C` from the SSM input mid‑block as in Mamba‑1).
  - Extra normalization: add a normalization layer (e.g., GroupNorm/RMSNorm) right before the output projection for stability.
- Head patterns (Section 7.2)
  - Introduces SSM analogs of multi‑head attention:
    - `MIS` (multi‑input SSM) ≈ multi‑value attention (MVA): `B, C` shared, `X` per head (Proposition 7.2 identifies Mamba‑1 as MIS).
    - `MCS` ≈ multi‑query attention (MQA), `MES` ≈ multi‑key attention (MKA), and standard multi‑head (`MHS`).
- Systems optimizations (Section 8; Figure 7)
  - Tensor parallelism (TP): Mamba‑2’s parallel projections enable one all‑reduce per block (on par with attention/MLP), unlike Mamba‑1 which needed two.
  - Sequence/context parallelism: split the sequence across devices and pass only the compact recurrent state between devices (linear bandwidth in number of workers), mirroring the SSD block decomposition (Section 8.2).
  - Variable‑length batching: handle different sequence lengths without padding by setting `A_t = 0` at sequence boundaries (Section 8.3).

## 4. Key Insights and Innovations
- SSMs ≡ Semiseparable Matrices (fundamental, Section 3)
  - Theorem 3.5 turns SSM computation into a mature structured‑matrix problem with known O(NT) representations and multiplications (Proposition 3.6). This re‑frames diverse SSM algorithms as structured matrix multiplication.

- Structured Masked Attention (SMA) generalizes Linear Attention (Section 4)
  - Clean tensor‑contraction proof (Eq. (12) → (15)) of the recurrent form and a principled generalization: any structured mask `L` with fast multiply yields an attention layer with both quadratic and linear forms (Definition 4.2; Figure 3).

- State Space Duality (SSD): a shared family with dual forms (Section 5)
  - Shows a large intersection: 1‑semiseparable SMA is a diagonal SSM (Corollary 5.1), and any efficient autoregressive kernel attention must be semiseparable (Theorem 5.2). This is more than an analogy; it is an exact equivalence class.

- SSD algorithm: block‑decomposition semiseparable multiplication (Section 6)
  - New algorithm that mixes the linear SSM recurrence and the quadratic attention form by tiling and exploiting the rank‑structure of off‑diagonal blocks (Figure 5; Listing 1 provides a short PyTorch implementation).
  - Significance: achieves optimal asymptotics while leveraging GPU tensor cores, enabling larger state sizes (N) at almost no extra cost (Figure 10, right).

- Mamba‑2: a TP‑friendly, stable SSM block with new head patterns (Sections 7–8)
  - Parallel parameter projections and grouped/value‑style head sharing bring SSM blocks closer to attention blocks operationally, reducing communication (Section 8.1) and enabling standard large‑scale training techniques.

## 5. Experimental Analysis
- Evaluation methodology
  - Long‑sequence synthetic task: Multi‑Query Associative Recall (MQAR), a demanding associative lookup benchmark (Section 9.1; Figure 8).
  - Language modeling pretraining: The Pile dataset (standard LM setting) with scaling‑law comparisons and zero‑shot evaluations (Sections 9.2, D.2–D.3).
  - Efficiency benchmarks: wall‑clock timing vs sequence length and vs state dimension (Figure 10).
  - Ablations: block design choices, head patterns, and kernel approximations (Sections 9.4.1–9.4.3; Tables 4–7).
  - Hybrid models: interleaving SSD with attention and/or MLP (Sections 9.2.3; Table 2 and Table 3).

- Main quantitative results
  - Speed and scaling (Figure 10)
    - > “SSD is 2–8× faster than a Mamba fused scan for large state expansion (N=64)” (Figure 10, left).
    - Crossover with FlashAttention‑2 at sequence length ≈ 2K, reaching ≈ 6× faster at 16K (Figure 10, left).
    - When increasing state size at fixed length (4K), the scan time grows linearly while SSD stays almost flat (Figure 10, right), enabling much larger `N`.
  - MQAR (Figure 8)
    - Mamba‑2 consistently outperforms Mamba‑1 and even vanilla attention across sequence lengths 256–1024; larger state sizes (`N=64` and `N=256`) markedly improve accuracy, confirming the value of big recurrent memory.
  - Scaling laws on The Pile (Figure 9)
    - Mamba‑2 matches or exceeds both Mamba and a strong “Transformer++” recipe in perplexity vs. FLOPs, and is Pareto‑dominant in wall‑clock (Section 9.2.1; Figure 9).
  - Zero‑shot downstream evaluations (Table 1)
    - At ≈2.7B parameters trained on 300B tokens, `Mamba‑2-2.7B` achieves:
      - Pile ppl 6.09, LAMBADA ppl 4.10, HellaSwag 76.4%, PIQA 69.6%, ARC‑C 36.4%, WinoGrande 64.0% (Table 1).
    - It outperforms `Mamba-2.8B` and open 2.8B–3B Transformers in the provided tasks on average (Table 1).
  - Hybrid models (Table 2 and Table 3)
    - Adding a small number of attention layers (≈10% of layers) to Mamba‑2 further improves quality:
      - For 350M/48‑layer models trained to 7B tokens, adding 6 attention blocks reduces validation perplexity from 8.60 (pure Mamba‑2) to 8.26–8.28, with best performance around 10% attention (Table 2).
      - At 2.7B and 300B tokens, `Mamba‑2-Attention` (58 SSD + 6 attention layers) yields Pile ppl 5.95 and stronger average downstream accuracy than pure Mamba‑2 and Transformer++ (Table 3).

- Ablations and diagnostics
  - Block design (Table 4): 
    - Moving to parallel projections (“Mamba‑2”) and adding an extra normalization layer gives the best perplexity (11.49 vs 11.66–11.76 for other variants at ~125M scale).
  - Head patterns (Table 5):
    - `MIS` (multi‑input ≈ multi‑value attention) substantially outperforms `MCS`/`MES` (multi‑query/multi‑key analogs) at matched parameter counts and state size, especially at 125M and 360M scales.
  - Kernel approximations (Tables 6–7):
    - Various softmax‑mimicking kernel maps (e.g., PRF/Performer, cosFormer, Based/ReBased) did not consistently outperform simple pointwise activations or Swish in SSD. Some introduced instability unless additional normalization was used (Table 6).
  
- Do the experiments support the claims?
  - Yes, on three fronts:
    1) Efficiency: SSD beats both optimized Mamba scans and FlashAttention‑2 for long sequences (Figure 10).
    2) Capability: Larger state sizes directly translate to stronger recall on MQAR (Figure 8).
    3) Quality: Mamba‑2 matches/exceeds strong Transformer baselines across model sizes and improves further when a small fraction of attention layers are added (Figure 9; Tables 1–3).
  - Caveats:
    - At short sequences (e.g., 2K), Transformer training may still be faster overall because it uses fewer token‑mixing layers (SSD vs attention+MLP balance); the authors suggest mixing SSD with MLP to recover speed (Section 9.3).

## 6. Limitations and Trade-offs
- Modeling assumptions and scope
  - The clearest equivalence holds for selective SSMs whose `A_t` are scalar times identity (the SSD “sweet spot”) and for masked kernel attentions whose masks are semiseparable (Sections 5.1–5.2). General diagonal SSMs are theoretically O(TN) (Theorem 3.7) but lack an equally simple attention‑like quadratic form for matmul‑heavy computation.
  - SSD and SMA do not cover softmax attention directly (no finite‑dimensional feature map), so full softmax behavior is approximated only indirectly (Section 10.3).
- Practical constraints
  - Short‑sequence regimes: a pure SSD stack can be less training‑efficient than a Transformer with many MLP layers, since MLPs are extremely hardware‑friendly (Section 9.3).
  - Kernel approximations: many LA‑style kernel tricks do not help or may destabilize training in SSD (Tables 6–7).
  - Preprocessing for fully general (unstructured) SSMs to reach O(TN) involves expensive steps (e.g., SVD/diagonalization) and is not the target in practice (Remark 4; Section 3.4.1).
- Open questions
  - Best mask structures beyond 1‑SS: SMA allows Toeplitz, Fourier, or other structured masks (Figure 3), but their full empirical and theoretical trade‑offs remain open.
  - How far can diagonal SSM expressivity be increased while keeping the matmul‑friendly quadratic dual?
  - Interpretability and behaviors like “attention sinks” for SSMs/SSD (Section 10.3) are not yet fully explored.

## 7. Implications and Future Directions
- Field impact
  - A unifying lens: seeing SSMs as semiseparable matrix mixers (Theorem 3.5) and attention as structured contractions (SMA) tightens the conceptual gap and ports Transformer‑style optimizations into SSMs (Sections 3–5, 8).
  - Practical consequence: with SSD, SSMs become easy to parallelize (one all‑reduce per block), easy to shard across sequence or tensor dimensions, and fast on existing GPU tensor cores (Sections 6–8).

- Research avenues
  - Structured masks in SMA: explore Toeplitz or Fourier masks (Figure 3) as principled positional bias alternatives to RoPE/AliBi, and study their autoregressive capacity (Theorem 5.2).
  - Extending beyond scalar‑identity `A_t`: design semiseparable forms that retain a clean quadratic dual while enlarging expressivity (Section 10.1 “Structure”).
  - Hybrid stacks: systematic recipes for mixing SSD with MLP and attention—Table 2 suggests ≈10% attention layers can be a strong operating point.
  - Theory–practice loop: use semiseparable closure properties (Appendix C.1) to reason about stacking, inverting, or composing SSM layers, and derive new fast algorithms (Appendix B).

- Applications
  - Long‑context LMs with efficient training/inference up to tens or hundreds of thousands of tokens (Figure 10).
  - Domains where constant‑size state is valuable: streaming ASR, online inference, or multi‑device contexts where passing compact states is cheaper than passing activations (Section 8.2).
  - Memory‑intensive retrieval‑like tasks where scaling the recurrent state `N` (now cheap with SSD) improves recall (Figure 8).

> Representative result snapshots:
> - “SSD is 2 − 8× faster than the optimized selective scan implementation of Mamba… crossing over [FlashAttention‑2] at sequence length 2K and 6× faster at 16K” (Section 1 Overview; Figure 10).
> - Theorem 3.5: SSMs are exactly semiseparable matrix transformations (Section 3).
> - Theorem 5.2: Any kernel attention with bounded‑order autoregression must be semiseparable SMA (Section 5.2; Appendix C.2).
> - `Mamba‑2‑2.7B` trained on 300B tokens: Pile ppl 6.09; strong zero‑shot averages vs comparable open models (Table 1).

Overall, the paper contributes both a unifying theory and practical algorithms/architecture that make SSMs competitive and often preferable for long‑sequence modeling, while remaining compatible with modern large‑scale training stacks.
