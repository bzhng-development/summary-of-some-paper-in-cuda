# FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness

**ArXiv:** [2205.14135](https://arxiv.org/abs/2205.14135)

## 🎯 Pitch

FlashAttention introduces an IO-aware, exact attention algorithm that dramatically speeds up and reduces the memory footprint of Transformer models by minimizing costly reads and writes to GPU high-bandwidth memory. By carefully tiling and recomputing attention blocks using fast on-chip memory, FlashAttention achieves up to 7.6× kernel speedups and linear memory scaling with sequence length, all while preserving exact model accuracy. This breakthrough enables practical, efficient training and inference on long sequences, addressing a core limitation of existing Transformers and unlocking new capabilities for deep learning on large contexts.

---

## 1. Executive Summary (2-3 sentences)
FlashAttention is an exact (non-approximate) attention algorithm that makes the attention step in Transformers much faster and more memory‑efficient by being IO‑aware—i.e., by minimizing slow reads/writes to GPU high‑bandwidth memory (HBM) and using fast on‑chip memory (SRAM) effectively. It tiles the attention computation to avoid ever materializing the N×N attention matrix, and recomputes small pieces on‑chip during backpropagation, yielding up to 7.6× kernel speedups (Figure 1 right), 2–3× end‑to‑end training speedups on common models, and linear memory growth in sequence length (Figure 3 right), while preserving exact attention outputs.

## 2. Context and Motivation
- Problem addressed
  - Self‑attention in Transformers has quadratic time and memory in sequence length N because it forms an N×N matrix S=QKᵀ and often stores both S and the softmax P (Section 2.2, Algorithm 0). This is the main barrier to long‑context training and inference.
- Why this matters
  - Practical: Long contexts are increasingly important (e.g., long documents, code, multi‑modal sequences). Yet common GPUs run out of memory and become slow on long sequences.
  - Architectural: Modern GPUs are often bottlenecked by memory movement, not floating‑point compute. On GPUs like the A100, on‑chip SRAM bandwidth (~19 TB/s) is an order of magnitude faster than HBM (~1.5–2.0 TB/s), but SRAM is tiny (≈20 MB aggregated) (Figure 1 left; Section 2.1).
- Shortcomings of prior approaches
  - Approximate attentions (sparse, low‑rank) reduce FLOPs but often don’t speed up wall‑clock time because they ignore memory access costs, incur overheads, or hurt accuracy (Section 1; Table 3 shows mixed accuracy/speed trade‑offs on LRA).
  - Naïve kernel fusion in deep‑learning frameworks can’t avoid writing large intermediates (S or P) to HBM for backward (Section 2.1 “Kernel fusion”).
- Positioning
  - This work reframes attention optimization around IO (reads/writes between HBM and SRAM) rather than FLOPs, introduces an IO‑optimal exact attention algorithm (FlashAttention), provides an IO‑complexity analysis and lower bound (Theorem 2; Proposition 3), and shows a block‑sparse extension that further reduces IO (Proposition 4).

Definitions used once:
- HBM: GPU off‑chip High Bandwidth Memory; large, relatively slow.
- SRAM: GPU on‑chip memory (registers/shared memory/L1), tiny but very fast.
- IO complexity: Number of transfers between memory levels; here, HBM accesses dominate runtime.
- Tiling: Process input in blocks that fit on‑chip, reusing data to reduce HBM traffic.

## 3. Technical Approach
The core idea is to never form or store the full N×N attention matrix in HBM. Instead, the algorithm streams small blocks through SRAM and incrementally maintains the quantities needed to compute the correct softmax and output.

Step‑by‑step (Algorithm 1; detailed forward Algorithm 2; backward Algorithm 4):

1) Block the inputs to fit on‑chip
- Split Q (queries), K (keys), and V (values) into rectangular blocks Qᵢ (size Bᵣ×d) and Kⱼ, Vⱼ (size B𝚌×d), where d is head dimension.
- Choose block sizes from SRAM capacity M so that all on‑chip temporaries fit:
  - B𝚌 ≈ M/(4d), Bᵣ ≈ min(M/(4d), d) (Algorithm 1 lines 1–4; constraints justified in proof of Theorem 2).

2) Two‑level loop that keeps K/V on‑chip and streams Q
- Outer loop over Kⱼ,Vⱼ blocks: load one Kⱼ and Vⱼ from HBM to SRAM once (Algorithm 1 lines 5–6).
- Inner loop over Qᵢ blocks: for each Qᵢ, load Qᵢ and the running per‑row statistics (explained next), compute the attention contributions for the (i,j) block, update the running output Oᵢ, and write the updated Oᵢ and stats back (lines 7–13).

3) Compute softmax one block at a time via incremental statistics
- Problem: softmax along each row needs the sum over all N keys; with tiling you only see B𝚌 keys at a time.
- Solution (Section 3.1 “Tiling”): maintain two per‑row statistics for each query row:
  - m (row‑wise running max of the logits) for numerical stability, and
  - ℓ (row‑wise running sum of exp(logits − m)).
- When a new (i,j) tile Sᵢⱼ = QᵢKⱼᵀ is computed on‑chip (line 9), compute its temporary row‑max m̃ and temporary exp‑sum ℓ̃ (line 10). Update the running m and ℓ using:
  - m_new = max(m, m̃); ℓ_new = exp(m−m_new)·ℓ + exp(m̃−m_new)·ℓ̃ (line 11).
- Update the partial output Oᵢ to incorporate the contribution of block j using the same normalization (line 12):
  - Oᵢ ← (1/ℓ_new) · [exp(m−m_new)·ℓ·Oᵢ + exp(m̃−m_new)·(exp(Sᵢⱼ−m̃)·Vⱼ)].
- Intuition: this is the softmax of a concatenation trick—sum and max of exponentials can be merged across chunks by rescaling (Section 3.1; equations right above Algorithm 1).

4) Handle masks and dropout while staying on‑chip
- Apply masking to Sᵢⱼ before softmax (Algorithm 2 line 11), and dropout to the block‑level probabilities on‑chip (line 14), then proceed as above.
- To avoid storing a huge dropout mask for backward, save the RNG state once in forward and regenerate the same mask blocks during backward (Algorithm 2 line 1; Algorithm 4 lines 1, 14).

5) Backward pass without storing N×N intermediates
- Standard backward needs P and S; storing them costs O(N²). Instead, recompute block‑wise on‑chip using Q,K,V and the saved per‑row (ℓ,m) and the RNG state (Algorithm 4 lines 11–20).
- A key simplification: the scalar Dᵢ = Σⱼ Pᵢⱼ·dPᵢⱼ equals the dot product dOᵢ·Oᵢ (Eq. (4)), which uses vectors of length d, avoiding reductions over length N (Algorithm 4 line 19).
- Then compute dS = P ∘ (dP − D) block‑wise and accumulate dQ and dK via small GEMMs, while dV accumulates via (Pᵀ dO) (Algorithm 4 lines 20–24). All done tile‑by‑tile in SRAM.

6) Fused implementation
- The entire forward pipeline—QKᵀ, masking, softmax stats, dropout, and PV—runs inside a single CUDA kernel per head per batch, with one write of O to HBM (Section 3.1 “Implementation details”).
- The backward is likewise an on‑chip tiled recomputation fused into a single kernel loop over blocks (Algorithm 4).

7) IO‑complexity analysis and near‑optimality
- Standard attention performs Θ(Nd + N²) HBM accesses (Theorem 2; Algorithm 0).
- FlashAttention reduces this to Θ(N² d / (2M−1)) ≈ Θ(N² d / M) (Theorem 2), by reusing on‑chip Kⱼ,Vⱼ and streaming Q multiple times. The same bound holds for backward (Theorem 5).
- Lower bound: no exact attention algorithm can asymptotically beat o(N² d / (2M−1)) HBM transfers uniformly over M∈[d, Nd] (Proposition 3). Intuition: when M=Θ(Nd) you must at least read inputs/outputs once, already Ω(Nd).

8) Block‑sparse FlashAttention (Algorithm 5)
- If the attention mask is block‑sparse (only a fraction s of blocks are nonzero), skip zero blocks in both loops. IO complexity improves to Θ(Nd + (N² d / (2M−1))·s) (Proposition 4).
- In experiments, a fixed “butterfly” pattern is used (Section 3.3), which empirically offers good coverage with small s.

Why these design choices?
- Tiling leverages the large speed gap between SRAM and HBM (Figure 1 left) to cut down memory traffic—the dominant bottleneck for attention (Section 2.1; arithmetic intensity discussion).
- Recomputing on‑chip in backward costs extra FLOPs but avoids N² reads/writes (Figure 2 left shows more FLOPs but far fewer HBM GB, resulting in 5.7× runtime reduction on GPT‑2 medium).
- Fusing all steps prevents writing intermediate S or P back to HBM and reloading them later.

## 4. Key Insights and Innovations
- IO‑aware exact attention as the primary optimization target
  - Novelty: Shifts the optimization objective from FLOP reduction to minimizing HBM accesses, matching the true bottleneck on GPUs (Section 1; 2.1).
  - Evidence: For GPT‑2 medium at seq=1024, FlashAttention uses 4.4 GB vs 40.3 GB HBM R/W and runs 7.3 ms vs 41.7 ms, despite slightly more FLOPs (75.2 vs 66.6 GFLOPs) (Figure 2 left).
- Incremental softmax with per‑row statistics (m, ℓ) enabling tiling
  - Novelty: An exact, numerically stable decomposition of softmax across blocks (Section 3.1; Algorithm 1 lines 10–12) so the full N×N matrix never exists in HBM.
  - Significance: Reduces memory footprint from quadratic to linear in N (Theorem 1) and enables single‑kernel fusion.
- Recomputation‑based backward with lightweight saved state
  - Novelty: Store only O(N) vectors (O, m, ℓ) and RNG state; recompute S and P block‑wise on‑chip in backward (Algorithm 4), using dO·O to avoid N‑length reductions (Eq. (4)).
  - Significance: Further cuts HBM traffic in backward to Θ(N² d / M) (Theorem 5), turning memory‑bound steps into on‑chip compute even when dropout is used.
- IO‑complexity optimality characterization
  - Novelty: Formal upper bound for FlashAttention and a matching parameter‑range lower bound (Theorem 2; Proposition 3), arguing no exact algorithm can asymptotically do better (for all M).
  - Significance: Provides a principled baseline for future IO‑aware attention designs.
- Block‑sparse FlashAttention as a fast approximate primitive
  - Novelty: Same tiling engine but skip zero blocks; IO scales with sparsity fraction s (Proposition 4).
  - Significance: 2–4× faster than dense FlashAttention and faster than other approximate attentions tested (Figure 2 right; Table 3 speedups).

## 5. Experimental Analysis
Evaluation setup (Sections 4, E; Figures 1–3; Tables 1–7, 8–21):
- Models and datasets
  - BERT‑large pretraining on Wikipedia; metric: time to a target masked‑LM accuracy (Table 1).
  - GPT‑2 small/medium on OpenWebText; metric: validation perplexity and wall‑clock training time (Table 2).
  - Long Range Arena (LRA) with sequence lengths 1K–4K; metrics: accuracy, throughput, training time (Table 3).
  - Long‑document classification on MIMIC‑III and ECtHR; metric: micro‑F1 vs sequence length (Table 5).
  - Path‑X (seq 16K) and Path‑256 (seq 64K) path‑finding tasks; metric: accuracy (Table 6).
  - System microbenchmarks: runtime and HBM traffic (Figures 1–3), hardware variations (Appendix E.5), memory footprint (Figure 3 right; Table 21).
- Baselines
  - Exact attention in PyTorch and Megatron‑LM; NVIDIA Apex FMHA for short sequences (Table 7).
  - Approximate/sparse attentions: Linformer, Linear/Performer, Local, Reformer, BigBird/Longformer, SMYRF, LSFormer, OpenAI Block‑Sparse (Tables 9–20).

Main quantitative findings (representative numbers as block quotes):
- Kernel‑level speed and IO
  - Figure 1 right (GPT‑2 attention microbenchmark): 
    > “FlashAttention … results in a 7.6× speedup on the attention computation.”
  - Figure 2 left (GPT‑2 medium, seq=1024, A100):
    > FLOPs 66.6→75.2 GFLOPs; HBM R/W 40.3→4.4 GB; runtime 41.7→7.3 ms.
  - Figure 2 middle (ablating block size): 
    > larger tiles reduce HBM accesses and runtime until other bottlenecks dominate (beyond block size 256).
- End‑to‑end training speedups
  - Table 1 (BERT‑large, 8×A100):
    > 20.0±1.5 min → 17.4±1.4 min to target accuracy (≈15% faster).
  - Table 2 (GPT‑2 small/medium, 8×A100):
    > small: 9.5 d (HF) → 2.7 d (3.5×); 4.7 d (Megatron) → 2.7 d (1.7×).  
    > medium: 21.0 d (HF) → 6.9 d (3.0×); 11.5 d (Megatron) → 6.9 d (1.8×).  
    > Perplexity matches baselines (18.2/14.3).
- Quality and long‑context capability
  - Table 4 (GPT‑2 small with longer context):
    > At 4K context, training is still 1.3× faster than Megatron at 1K and perplexity improves from 18.2→17.5 (−0.7).
  - Table 5 (Long‑document classification with RoBERTa+FlashAttention):
    > MIMIC‑III: 52.8→57.1 micro‑F1 from 512→16K tokens (+4.3).  
    > ECtHR: 72.2→80.7 at 8K (+8.5), slight drop at 16K (79.2).
  - Table 6 (Path tasks):
    > First better‑than‑chance Transformer results on Path‑X: 61.4% (FlashAttention).  
    > Path‑256 at seq=64K: 63.1% with block‑sparse FlashAttention; all other Transformer baselines fail.
- Benchmarking vs. approximate methods
  - Figure 3 left:
    > FlashAttention beats exact baselines by up to 3× across 128–2048 tokens; approximate methods cross over around 512–1024 tokens; block‑sparse FlashAttention is faster than all tested methods across lengths.
- Memory footprint
  - Figure 3 right; Table 21:
    > Memory scales linearly with N and is up to 20× lower than exact baselines; only Linformer reaches 64K among baselines, but FlashAttention is still ~2× more memory‑efficient there.
- Against NVIDIA Apex FMHA (short sequences; Table 7):
  > At seq=512, forward 1.14→0.81 ms; backward 1.81→2.00 ms; net 2.95→2.81 ms (slightly faster overall).

Ablations/robustness
- Block‑size sensitivity (Figure 2 middle): performance tracks HBM accesses down to tile size limits.
- Hardware sensitivity (Appendix E.5): larger speedups on GPUs with lower HBM bandwidth (RTX 3090), smaller speedups on GPUs with smaller SRAM (T4).
- Head dimension sensitivity: with d=128, speedups decrease but remain significant, especially under causal masks (Appendix E.5).

Assessment
- The experiments convincingly support the central claim: reducing HBM accesses yields large real‑world speedups even with more FLOPs (Figure 2 left).
- End‑to‑end training wins on BERT/GPT‑2 and the LRA suite (Tables 1–3) corroborate system microbenchmarks.
- Quality is maintained (GPT‑2 perplexity) or improved via longer context (Tables 4–6).
- Results are broad (multiple tasks, models, hardware), though training comparisons are against strong but not exhaustive baselines.

## 6. Limitations and Trade-offs
- Still quadratic compute for dense attention
  - FlashAttention does not reduce arithmetic complexity; if compute becomes the bottleneck (e.g., extremely large d or very fast HBM), speedups shrink (Figure 2 middle shows a regime where runtime becomes compute‑bound).
- Engineering effort and portability
  - Requires bespoke CUDA kernels and careful on‑chip memory management; porting across GPU generations and supporting new attention variants demands engineering (Section 5 “Compiling to CUDA” limitation).
- Dependence on SRAM size and GPU characteristics
  - Smaller SRAM reduces tile sizes and speedups (Appendix E.5, T4 results). Benefits depend on the hardware memory hierarchy.
- Backward recomputation adds FLOPs
  - Extra compute is usually masked by memory savings (Figure 2 left), but in compute‑bound regimes could be a trade‑off.
- Multi‑GPU communication not optimized
  - Analysis and kernels target single‑GPU IO; cross‑GPU communication patterns for very long sequences are not addressed (Section 5 “Multi‑GPU IO‑Aware Methods”).
- Block‑sparse quality depends on sparsity pattern
  - The butterfly mask is fixed in experiments (Section 3.3); while fast, it is not learned/adaptive and may not suit all tasks.

## 7. Implications and Future Directions
- Broader shift toward IO‑aware deep learning
  - This work demonstrates that careful control of HBM traffic can unlock large, practical speedups even for “optimized” operations. The analysis and methodology provide a template for other memory‑bound layers (Section 5 “IO‑Aware Deep Learning”).
- Enabling longer contexts in standard Transformers
  - Linear memory scaling allows training and inference at 16K–64K tokens on a single A100 (Figure 3; Path tasks), opening applications in long‑document NLP, long‑horizon RL, genomics, and code modeling.
- Compiler support for IO‑aware kernels
  - A natural next step is a high‑level compiler that emits IO‑efficient fused kernels from PyTorch‑level code (Section 5 “Compiling to CUDA”), akin to Halide for image pipelines.
- Multi‑GPU IO‑aware attention
  - Extending the IO analysis to inter‑GPU communication and designing tiled, overlap‑compute‑with‑communication kernels could scale long‑sequence attention beyond single‑GPU limits (Section 5; Appendix D.2).
- Beyond attention
  - The same principles could accelerate sparse MLPs (memory‑bound in practice) and kernel methods that form implicit N×N matrices (Appendix D.2), as well as other sequence operations with large intermediate tensors.
- Better block‑sparse design
  - Learning task‑specific block patterns or combining block‑sparsity with low‑rank compression (as in Scatterbrain/BigBird) on top of the FlashAttention primitive could preserve quality while achieving near‑linear IO.

In short, FlashAttention reframes attention optimization around memory movement, provides an exact, IO‑optimal algorithm with a fused implementation, and validates that IO‑awareness—not just FLOP counting—drives real speedups and unlocks longer contexts.
