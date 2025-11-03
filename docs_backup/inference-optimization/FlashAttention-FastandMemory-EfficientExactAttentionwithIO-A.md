# FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness

**ArXiv:** [2205.14135](https://arxiv.org/abs/2205.14135)

## 🎯 Pitch

FlashAttention introduces a novel, IO-aware exact attention algorithm for Transformers that dramatically reduces memory traffic between GPU high-bandwidth memory (HBM) and on-chip SRAM by tiling computations and avoiding the need to store the full N×N attention matrix in slow memory. This breakthrough enables Transformers to process longer sequences much faster and with far less memory—achieving up to 7.6× speedups and unlocking new practical capabilities—making it a foundational step towards truly scalable and efficient large-scale model training and inference.

---

## 1. Executive Summary (2–3 sentences)
FlashAttention is an exact (not approximate) attention algorithm for Transformers that reorganizes the computation to minimize slow memory traffic between GPU high-bandwidth memory (`HBM`) and on-chip static RAM (`SRAM`). By tiling the computation, incrementally computing softmax, and recomputing intermediates during backpropagation, it avoids materializing the full N×N attention matrix in `HBM`, yielding large, demonstrated wall‑clock speedups and memory savings across models and sequence lengths (e.g., up to 7.6× speedup on GPT‑2 attention in Figure 1 right; 15% end‑to‑end BERT‑large speedup in Table 1).

## 2. Context and Motivation
- Problem addressed
  - Standard attention requires computing S = QKᵀ (size N×N), applying softmax to get P, and then O = PV. Common implementations materialize S and P in GPU `HBM` (Algorithm 0), which causes both quadratic memory usage and a large number of slow memory reads/writes. Section 2.2 and Algorithm 0 detail this pipeline and its `HBM` traffic.
- Why it matters
  - On modern GPUs, computation is fast but memory movement is comparatively slow; many deep learning operations are memory‑bound rather than compute‑bound (Section 2.1, “Performance characteristics,” and Figure 1 left). For attention, this means wall‑clock time often scales with memory input/output (IO), not with FLOPs.
  - Long sequences are increasingly important (language, vision, long‑document tasks). Quadratic `HBM` traffic limits feasible context length and training speed.
- Prior approaches and shortcomings
  - Approximate attention methods (sparse or low‑rank) reduce FLOPs to near‑linear but often do not yield wall‑clock speedups and may reduce model quality (Abstract; Section 1). A common reason: they focus on FLOP reduction while ignoring IO/memory movement overheads.
  - Some optimized attention kernels fuse a few steps but still read/write large intermediates to `HBM` (Section 2.1 “Kernel fusion” and Appendix E.4 on Apex FMHA).
- Positioning
  - FlashAttention reframes exact attention as an IO‑aware algorithm—explicitly optimizing the number and pattern of reads/writes between `HBM` and on‑chip `SRAM` (Figure 1 left). It uses two techniques—tiling and recomputation—to compute the same mathematical result while drastically reducing `HBM` traffic (Section 3.1, Algorithm 1).

## 3. Technical Approach
At a high level, FlashAttention changes “when” and “where” attention’s sub-steps are computed so that data live on the fast on‑chip memory (`SRAM`) when they are needed, and the large N×N intermediate matrices are never stored in `HBM`.

Key concepts defined as first used:
- `HBM` (High Bandwidth Memory): GPU’s large but slower off‑chip memory.
- `SRAM`: small, on‑chip memory that is much faster (A100 example bandwidths in Figure 1 left).
- IO‑aware: an algorithm designed to minimize reads/writes between memory levels.
- Tiling: split large matrices into blocks and compute with a small subset in `SRAM` at a time.
- Recomputation: during backpropagation, recompute intermediates on‑chip instead of reading them from `HBM`.
- Fused kernel: implement the entire attention pipeline inside a single GPU kernel to avoid extra IO.

3.1 What the standard pipeline does and why it is slow (Algorithm 0; Section 2.2)
- Compute S = QKᵀ and write S (N×N) to `HBM`.
- Read S, compute P = softmax(S), write P (N×N) to `HBM`.
- Read P and V, compute O = PV, write O to `HBM`.
- Result: Θ(Nd + N²) `HBM` accesses in forward alone (Theorem 2).

3.2 FlashAttention forward: tile and incrementally normalize (Algorithm 1, expanded in Algorithm 2)
- Partition Q into row blocks of size `B_r × d` and K,V into column blocks of size `B_c × d` (Algorithm 1, lines 1–4).
- Outer loop: load one K_j, V_j block from `HBM` to `SRAM` (line 6).
- Inner loop: for each Q_i block (line 8),
  1) Compute S_ij = Q_i K_jᵀ on chip (`SRAM`) (line 9).
  2) Apply masking and compute a numerically stable softmax “partially” for the current block using per-row max and sum, `m̃_i j` and `ℓ̃_i j` (line 10; mask in Algorithm 2 line 11).
  3) Merge these per-block stats into global running stats for the same rows: `m_new = max(m, m̃)` and `ℓ_new = exp(m−m_new)ℓ + exp(m̃−m_new)ℓ̃` (line 11; same formulas appear in Section 3.1 under “Tiling,” and in Algorithm 2 lines 12–13). This exactly reproduces the softmax as if computed over the full row by using algebraic aggregation.
  4) Update the running output block O_i with properly rescaled contributions from this K,V block: O_i ← diag(ℓ_new)^{-1} (diag(ℓ) e^{m−m_new} O_i + e^{m̃−m_new} P̃_ij V_j) (line 12; Algorithm 2 line 15).
  5) Save the updated O_i, ℓ_i, m_i to `HBM` (lines 12–13) before moving to the next Q_i.
- Why it works
  - The key trick is decomposing softmax across concatenated blocks using per-row max `m` and sum `ℓ` (Section 3.1 “Tiling”; the derivation shows how to merge partial softmax results exactly).
  - Correctness is proven in Theorem 1: Algorithm 1 returns O = softmax(QKᵀ)V, with O(N²d) FLOPs and O(N) extra memory.

3.3 FlashAttention backward: recompute instead of reading large intermediates (Algorithm 4; Appendix B.2–B.4)
- Store only O, the per-row softmax stats (`ℓ`, `m`), and the PRNG state for dropout in the forward pass (Algorithm 2, lines 1, 19). Do not store N×N attention matrices.
- During backward:
  - Recreate the same dropout mask from the stored PRNG state (Algorithm 4, lines 1, 14).
  - Recompute the needed P blocks on the fly using Q_i, K_j, `ℓ_i`, `m_i` (Algorithm 4, lines 11–15).
  - Compute gradients without ever forming full N×N matrices:
    - dV accumulates from (P_dropped)ᵀ dO (Algorithm 4, lines 16, 24).
    - dP = dO Vᵀ, and the softmax Jacobian is applied in blocked form to obtain dS = P ∘ (dP − D) with D_i = sum(dO_i ∘ O_i) (lines 17–20; see the scalar derivation in Appendix B.2, Eqs. (3)–(6)).
    - dQ accumulates as dQ_i ← dQ_i + τ dS_ij K_j; dK similarly from dSᵀQ (lines 21–22, 24).
- Benefit
  - Recomputation increases FLOPs slightly but slashes `HBM` IO and thus runtime (Figure 2 left: GFLOPs increase from 66.6→75.2, but `HBM` read/write drops from 40.3 GB→4.4 GB; runtime drops 41.7→7.3 ms).

3.4 Complexity analysis (Section 3.2; Theorem 2 and proof)
- Let `N` = sequence length, `d` = head dimension, `M` = usable on‑chip `SRAM` size.
- Standard attention: Θ(Nd + N²) `HBM` accesses.
- FlashAttention: It loads each K,V block once and makes T_c passes over Q,O, where T_c ≈ N d / M (proof, “We then have: T_c = Θ(N d / M)”). Total `HBM` accesses become Θ(N d · T_c) = Θ(N² d² / M).
- Lower bound: Proposition 3 shows no exact attention algorithm can asymptotically beat this form across all `M` in [d, N d]; i.e., up to constants FlashAttention is IO‑optimal over a wide memory range.
- Empirical confirmation: Figure 2 middle varies block size (affecting `HBM` accesses) and shows runtime drops until arithmetic becomes the bottleneck.

3.5 Block‑sparse FlashAttention (Section 3.3; Algorithm 5; Proposition 4)
- Idea: If attention has a known block‑sparse pattern (only some Q‑K block pairs interact), skip zero blocks entirely but keep the same IO‑aware fused implementation.
- Complexity: Θ(N d + (N² d² / M) · s) `HBM` accesses, where `s` is the fraction of non‑zero blocks (Proposition 4). This yields proportional IO and runtime gains (Figure 2 right).

3.6 Implementation details (Section 3.1 “Implementation details: Kernel fusion”; Section 2.1; Appendix E.4)
- One fused CUDA kernel handles “matmul → mask → softmax → dropout → matmul,” reading inputs once per tile and writing only the final O (Figure 1 left).
- Memory hierarchy on A100: `SRAM` ≈ 20 MB per GPU at ≈19 TB/s vs. `HBM` ≈ 40–80 GB at 1.5–2.0 TB/s (Figure 1 left), motivating aggressive on‑chip reuse.
- Compared with Nvidia Apex FMHA (which still stores the N×N attention matrix in forward), FlashAttention is comparable or faster for short sequences and scales to much longer ones with far lower memory (Appendix E.4, Table 7).

## 4. Key Insights and Innovations
- IO‑aware reformulation of exact attention
  - Novelty: Puts memory traffic—not FLOPs—at the center of algorithm design for attention. Unlike prior “fused” kernels, it avoids ever writing the N×N attention matrix to `HBM` by computing and merging per‑tile softmax statistics (`m`, `ℓ`) (Algorithm 1; Section 3.1).
  - Impact: Reduces `HBM` accesses from Θ(Nd + N²) to Θ(N² d² / M) (Theorem 2), directly predicting large wall‑clock speedups on memory‑bound hardware (confirmed in Figure 2 left).
- Exact incremental softmax with algebraic aggregation
  - What’s new: A practical tiling scheme that maintains exactness via the per‑row max/sum trick for softmax (Section 3.1 “Tiling” equations). This enables exact attention without seeing all keys at once.
  - Why it matters: Prior “memory‑efficient” attention ideas avoided storing intermediates but still incurred quadratic reads/writes; here the incremental normalization makes a fused, IO‑light pass possible.
- Backward recomputation with analytical simplifications
  - What’s different: Instead of generic gradient checkpointing, the derivation uses O(N) statistics (`m`, `ℓ`) and per‑row dot products D_i = ⟨dO_i, O_i⟩ to avoid N×N tensors entirely (Appendix B.2–B.4; Algorithm 4).
  - Benefit: Keeps the backward pass IO‑optimal and fast despite extra FLOPs (Figure 2 left).
- Block‑sparse extension as a first‑class primitive
  - Contribution: A drop‑in IO‑aware kernel that respects a block sparsity mask, achieving further linear improvements in IO proportional to sparsity (Section 3.3; Proposition 4; Figure 2 right).
  - Significance: Outperforms prior approximate methods in runtime at long lengths while often maintaining exactness for the non‑zero pattern.

These are fundamental innovations (changing the algorithmic objective to IO‑minimization and proving near‑optimal IO bounds), not just incremental engineering.

## 5. Experimental Analysis
Evaluation setup (Section 4; Appendix E):
- Hardware: Primarily Nvidia A100 GPUs; additional tests on RTX 3090 and T4 (Appendix E.5).
- Models/datasets:
  - BERT‑large pretraining on Wikipedia, MLPerf 1.1 setup (Table 1).
  - GPT‑2 small/medium on OpenWebText (Table 2), with context lengths up to 4K (Table 4).
  - Long Range Arena (LRA) suite (sequence lengths 1K–4K) for accuracy and speed (Table 3).
  - Long document classification: MIMIC‑III and ECtHR (Table 5).
  - Path‑X (16K tokens) and Path‑256 (64K) tasks (Table 6).
- Baselines: Standard PyTorch attention, Megatron‑LM attention, Apex FMHA (Appendix E.4), and multiple approximate/sparse methods (Linformer, Performer, Reformer, Local, BigBird, Longformer, Smyrf, LSFormer), with fair settings and, when needed, FP32 for methods lacking FP16 support (Appendix E.6).
- Metrics: Wall‑clock time, memory usage, throughput, accuracy/perplexity. Where relevant, dropout and masking are included (Table series 9–21).

Main quantitative results
- Microbenchmarks: attention kernel speed and IO
  - GPT‑2 medium (N=1024, d=64, 16 heads, bs=64) on A100: FlashAttention reduces `HBM` read/write from 40.3 GB to 4.4 GB and runtime from 41.7 ms to 7.3 ms, with a slight FLOP increase (66.6→75.2 GFLOPs). Quote (Figure 2 left):
    > “HBM R/W 40.3 GB → 4.4 GB; Runtime 41.7 ms → 7.3 ms; 7.6× speedup on the attention computation.”
  - Runtime decreases as block size increases (fewer passes over Q), until arithmetic becomes the bottleneck (Figure 2 middle).
  - Block‑sparse FlashAttention runs faster by a factor proportional to sparsity (Figure 2 right).
- End‑to‑end training speed
  - BERT‑large (MLPerf 1.1 target): Table 1 shows
    > “20.0 ± 1.5 min (Nvidia MLPerf) vs. 17.4 ± 1.4 min (FlashAttention)” — a 15% speedup.
  - GPT‑2 on OpenWebText: Table 2 shows
    > GPT‑2 small: 9.5 days (HuggingFace) → 2.7 days (FlashAttention), 3.5× speedup; 4.7 days (Megatron) → 2.7 days, 1.7×.
    > GPT‑2 medium: 21.0 days (HF) → 6.9 days (FlashAttention), 3.0×; 11.5 days (Megatron) → 6.9 days, 1.7×.
  - LRA throughput: Table 3 reports up to 2.4× speedup for FlashAttention over standard Transformer, and 2.8× for block‑sparse FlashAttention; accuracy is on par with baselines.
- Longer context with exact attention
  - GPT‑2 small with FlashAttention at 4K context is still faster than Megatron at 1K and improves perplexity (Table 4):
    > 1K Megatron: 18.2 ppl, 4.7 days; 4K FlashAttention: 17.5 ppl, 3.6 days (1.3× speedup vs Megatron 1K).
- Downstream quality gains from longer sequences
  - Long‑document classification (Table 5):
    > MIMIC‑III micro‑F1: 52.8 (512) → 57.1 (16K), +4.3 points vs 512; ECtHR: 72.2 (512) → 80.7 (8K), +8.5 points vs 512.
- New capabilities at very long lengths
  - Path‑X (16K) and Path‑256 (64K): Table 6
    > First better‑than‑chance Transformer results: FlashAttention 61.4% on Path‑X; block‑sparse FlashAttention 63.1% on Path‑256.
- Runtime and memory scaling vs alternatives (Figure 3; Tables 9–21)
  - Forward+backward runtime: FlashAttention is up to 3× faster than PyTorch attention for N≤2K and remains competitive with many approximate/sparse methods up to a crossover between 512–1024 (Figure 3 left).
  - Memory: FlashAttention uses memory linear in N and is up to 20× more memory‑efficient than exact attention baselines; it reaches N=64K where most baselines OOM (Figure 3 right). Table 21 shows, e.g., at N=8192 the memory is 1672 MB for FlashAttention vs 6784 MB for Local Attention and 24134 MB for Reformer.
- Comparison to Apex FMHA (Appendix E.4, Table 7)
  - For N≤512, forward is slightly faster and backward slightly slower; net is comparable or slightly better. Crucially, FlashAttention scales to long sequences and reduces memory footprint because it does not store N×N attention in forward.

Ablations, robustness, and additional checks
- Block size vs runtime (Figure 2 middle).
- Hardware sensitivity: speedups reported on A100, RTX 3090, and T4; trends match IO analysis (Appendix E.5, Figures 5–8).
- Numerical stability: validation perplexity curves for GPT‑2 small/medium match HuggingFace across training (Appendix E.2, Figure 4).
- Broad baselines and consistent measurement protocols across dropout/masking settings (Tables 9–21).

Assessment
- The experiments convincingly support the core claims: reducing `HBM` IO leads to large, consistent wall‑clock speedups and substantial memory savings without sacrificing exactness. Results span micro‑benchmarks, end‑to‑end training, and downstream tasks.

## 6. Limitations and Trade-offs
- Still quadratic FLOPs in sequence length
  - FlashAttention is exact attention; it avoids quadratic `HBM` traffic but not quadratic compute. Very large N may still be compute‑limited once IO is minimized (Figure 2 middle shows a regime where runtime plateaus as compute dominates).
- Single‑GPU algorithm and kernel specialization
  - Design and proofs focus on a single GPU’s `SRAM`/`HBM` hierarchy (Theorem 2). Multi‑GPU sharding and inter‑GPU bandwidth are not addressed (Section 5 “Multi‑GPU IO‑Aware Methods”).
  - Requires custom CUDA kernels tuned to GPU architectures (Section 5 “Compiling to CUDA”), increasing engineering burden and potential portability issues.
- Tuning block sizes and head dimensions
  - Effective block sizes depend on `SRAM` size and head dimension `d`; for larger `d`, blocks must shrink (Appendix E.5, Figure 6), reducing speedups.
- Applicability of block‑sparse extension
  - Block‑sparse FlashAttention assumes a block sparsity mask (Algorithm 5). Where sparsity is not available or mismatched to the task, the dense kernel is used.
- Dropout and masking assumptions
  - Backward relies on re‑generating the same dropout mask from saved PRNG state (Algorithm 4, line 1). This is standard, but it is an assumption that the random state is preserved correctly across frameworks/runs.

## 7. Implications and Future Directions
- Broader impact on efficient deep learning
  - Demonstrates that IO‑aware algorithm design can unlock large, practical speedups even for “fixed” architectures like attention. This reframes a common optimization goal from FLOPs to data movement—likely applicable to other layers that are memory‑bound (Section 5 “IO‑Aware Deep Learning”).
- Enabling longer‑context Transformers
  - FlashAttention’s linear‑in‑N memory enables training/evaluating with much longer contexts, improving quality in language modeling and long‑document tasks (Tables 4–5) and enabling previously out‑of‑reach benchmarks (Table 6).
- Compiler/DSL opportunities
  - A compelling target is a high‑level compiler that emits IO‑aware kernels from declarative attention specs, akin to Halide for image processing (Section 5 “Compiling to CUDA”).
- Multi‑GPU and system‑level IO analysis
  - Extending IO analysis to include inter‑GPU (and even CPU‑GPU) communication could yield new parallel attention algorithms optimized for modern clusters (Section 5 “Multi‑GPU IO‑Aware Methods”).
- Beyond attention
  - The same principles may accelerate memory‑bound components like normalization layers, certain sequence models, or kernel machines (Appendix D.2 “Kernel machine learning”). Block‑sparse ideas could also be applied to MLPs and other modules if IO‑aware sparsity implementations are devised (Appendix D.2 “Sparse MLP layers”).

Overall, FlashAttention is a fundamental reframing of exact attention around IO‑efficiency, backed by both theory (Theorem 2; Proposition 3) and extensive empirical validation (Figures 1–3; Tables 1–7, 9–21). It provides immediate practical benefits and opens a pathway for IO‑aware design across the deep learning stack.
