# Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention

**ArXiv:** [2502.11089](https://arxiv.org/abs/2502.11089)

## 🎯 Pitch

Native Sparse Attention (NSA) introduces a novel, hierarchical sparse attention mechanism that is natively trainable and specifically optimized for modern hardware, enabling unprecedented efficiency for long-context language models. By combining token compression, blockwise token selection, and a local sliding window—alongside custom low-level kernels—NSA delivers dramatic real-world speedups (up to 11.6× for decoding) without sacrificing, and sometimes even surpassing, the accuracy of full attention models on challenging benchmarks. This innovation addresses key shortcomings of previous sparse methods, making it possible to efficiently train and deploy LLMs at long context lengths that are essential for tasks like codebase completion, document reasoning, and multi-turn conversational AI.

---

## 1. Executive Summary
This paper introduces NSA (Native Sparse Attention), a new attention mechanism that is both natively trainable and hardware-aligned for efficient long‑context language modeling. NSA replaces full quadratic attention with a hierarchical, query‑dependent sparse scheme that combines token compression, blockwise token selection, and a local sliding window, and it comes with specialized kernels that translate theoretical sparsity into real speedups. On 64k-token sequences it reports up to 11.6× faster decoding and 9.0×/6.0× speedups for forward/backward passes while matching or outperforming full attention on general, long‑context, and chain‑of‑thought evaluations (Figure 1; Tables 1–3).

## 2. Context and Motivation
- Problem addressed
  - Long‑context LLMs are constrained by the high cost of standard attention, whose latency dominates as context grows (Section 1). The authors estimate that with softmax attention architectures, attention accounts for 70–80% of total latency at 64k‑token decoding.
  - Many recent sparse attention methods reduce theoretical computation but often fail to deliver end‑to‑end wall‑clock speedups or support training (Section 2).

- Why this matters
  - Real applications—repository‑level code, long multi‑turn interactions, document‑level reasoning—require long contexts. Making both training and inference efficient expands what models can practically handle (Section 1).

- Where prior approaches fall short (Section 2)
  - Phase-restricted sparsity: Some techniques only help during decoding (e.g., KV eviction like H2O) or only during prefilling; end‑to‑end latency remains high because at least one stage behaves like full attention.
  - Incompatibility with modern attention architectures: Methods that sparsify independently per attention head do not align with Multiple/Grouped‑Query Attention (MQA/GQA), where heads share the KV cache; the union of selected KV across heads can negate memory gains (Section 2.1).
  - Not trainable end‑to‑end: Discrete or heuristic selection (e.g., K‑means in ClusterKV, SimHash in MagicPIG) breaks differentiability; gradients cannot flow to learn sparse patterns (Section 2.2).
  - Inefficient backward passes: Token‑level random access prevents reuse of fast blockwise kernels such as FlashAttention, causing low hardware utilization during training (Section 2.2).

- Positioning
  - NSA is designed as a native, differentiable sparse attention that:
    - Works in all phases (prefilling, decoding, training).
    - Aligns with GQA/MQA to minimize KV memory traffic.
    - Uses blockwise access patterns to fit high‑throughput GPU kernels (Sections 3.2–3.4).

Definitions used in this section:
- `KV cache`: the stored keys and values from previous tokens used during decoding.
- `Prefilling`: processing the initial prompt where many tokens are handled in parallel.
- `Decoding`: generating one token at a time autoregressively; this phase is typically memory‑bandwidth bound.
- `Arithmetic intensity`: ratio of computation to memory access; high intensity favors compute‑bound kernels, low intensity is memory‑bound (Section 3.1).
- `GQA/MQA`: architectures where multiple query heads share the same K and V, reducing KV memory bandwidth during decoding (Section 2.1; citations to Ainslie 2023; Shazeer 2019).

## 3. Technical Approach
NSA constructs, for each query token `q_t`, a compact set of contextual key/value pairs `K̃_t, Ṽ_t` and computes attention only over these, instead of over all past tokens (Equations 3–4). The final output is a gated sum of three attention branches (compression, selection, sliding window) as in Equation (5) and Figure 2.

Step-by-step overview:
1. Replace the full context with a compact, query‑dependent set
   - Formally: compute `K̃_t = f_K(q_t, k_{:t}, v_{:t})`, `Ṽ_t = f_V(q_t, k_{:t}, v_{:t})` (Eq. 3), then attend: `o*_t = Attn(q_t, K̃_t, Ṽ_t)` (Eq. 4).
   - Combine multiple remappings: `o*_t = Σ_{c ∈ {cmp, slc, win}} g^c_t · Attn(q_t, K̃^c_t, Ṽ^c_t)` (Eq. 5), where `g^c_t ∈ [0,1]` are learned gate scores from an MLP.

2. Branch A — Token compression (Section 3.3.1; Eq. 7)
   - Idea: summarize consecutive tokens into a single “compressed” token so that a query can cheaply scan the whole history at a coarse granularity.
   - Mechanism:
     - Partition the sequence into overlapping blocks of length `l` with stride `d` (`d < l` to reduce fragmentation).
     - Apply a learnable function `φ` (an MLP with intra‑block position encoding) to the keys in each block to produce one compressed key; similarly compress values.
     - This yields compressed `K̃_cmp_t, Ṽ_cmp_t` containing roughly `(t - l)/d` items instead of `t`.

3. Branch B — Blockwise selection (Section 3.3.2; Eqs. 8–12)
   - Motivation: compressed tokens may miss fine details; selection retains a small number of full‑fidelity blocks.
   - Key design choices and why:
     - Blockwise (not tokenwise): contiguous blocks align with GPU memory and Tensor Core usage, and match empirical attention continuity (Section 3.3.2 and visualization in Figure 8).
     - Importance scoring with negligible overhead: reuse the attention scores computed between `q_t` and the compressed keys (Eq. 8) to derive block‑level importance.
     - When compression and selection blockings differ, map scores using a structured aggregation (Eq. 9).
     - GQA/MQA alignment: aggregate importance across heads within a GQA group (Eq. 10) so all heads share one subset of selected KV blocks, minimizing KV cache loads.
     - Top‑n selection: choose indices `I_t` of the `n` most important blocks (Eq. 11) and concatenate their tokens to form `K̃_slc_t, Ṽ_slc_t` (Eq. 12).

4. Branch C — Sliding window (Section 3.3.3)
   - Preserve local context by always keeping the most recent `w` tokens intact: `K̃_win_t = k_{t−w:t}`, `Ṽ_win_t = v_{t−w:t}`.
   - Prevent “shortcut learning”: each branch has its own K/V and attention computation, and outputs are combined via learned gates `g_t^c` (Eq. 5), which encourages compression/selection to specialize in longer‑range patterns instead of relying on the local window.

5. Training and gating
   - All operations are differentiable. The gates `g_t^c` come from an MLP with sigmoid, enabling end‑to‑end learning of how much each branch should contribute (Section 3.2).

6. Hardware‑aligned kernel design (Section 3.4; Figure 3)
   - Compressed and sliding‑window branches reuse FlashAttention‑2 kernels.
   - A specialized Triton kernel accelerates the selection branch:
     - Group‑centric query loading: for each time `t`, load all query heads in a GQA group together because they share the same selected KV blocks.
     - Shared KV fetching: load the selected KV blocks once into on‑chip SRAM in continuous chunks (`B_k` divides the selection block size `l'`), then compute attention for all heads in the group.
     - Grid scheduling: put the query/output loops on Triton’s grid (outer loop) since the inner loop length (≈ number of selected blocks) is similar across positions.
   - This design:
     - Maximizes arithmetic intensity during prefilling/training.
     - Minimizes KV memory traffic during decoding by sharing selected blocks across heads.

7. Decoding memory model (Section 5.2; Table 4)
   - Per decoding step, NSA loads at most:
     - ~`(s−l)/d` compressed tokens,
     - `n·l'` tokens from selected blocks, and
     - `w` recent tokens,
     where `s` is current sequence length.
   - This yields a near‑linear relationship between reduced memory access volume and speedup (expected up to 11.6× at 64k; Table 4).

Example hyperparameters used for the 27B model (Section 4.1):
- Compression: block `l=32`, stride `d=16`.
- Selection: block `l'=64`, top‑`n=16` blocks (including one initial and two local blocks).
- Sliding window: `w=512`.
- Architecture: 27B MoE+GQA (4 groups, 64 heads total; `d_k = d_q = 192`, `d_v = 128`) trained on 270B tokens, then long‑context adaptation to 32k with YaRN.

## 4. Key Insights and Innovations
1. Hierarchical, query‑conditioned sparsity that is fully differentiable (Sections 3.2–3.3)
   - Novelty: combines three branches—compressed scan for global context, blockwise selection for detail, and a local window—into a gated mixture (Eq. 5).
   - Significance: preserves both global awareness and local precision while allowing end‑to‑end learning of sparse patterns. This contrasts with inference‑only sparsity and heuristic selection in prior work.

2. Hardware‑aligned blockwise selection with GQA/MQA compatibility (Sections 2.1, 3.3.2, 3.4)
   - Novelty: importance scores are computed via already‑available compressed attention (Eq. 8), then aggregated across heads in a GQA group (Eq. 10) to ensure shared KV subsets.
   - Significance: converts theoretical sparsity into real speed by minimizing redundant KV loads and maximizing continuous memory access, a common bottleneck for sparse methods.

3. Specialized Triton kernel for selection that balances arithmetic intensity (Section 3.4; Figure 3)
   - Novelty: group‑centric query loading and shared KV fetching in continuous chunks; outer‑loop grid scheduling.
   - Significance: achieves FlashAttention‑level throughput on sparse attention during training/prefilling and large speedups during decoding (Figure 6; Figure 1 right).

4. Native trainability without auxiliary losses or discrete operators (Sections 2.2, 6.1)
   - Novelty: avoids non‑differentiable operations (e.g., K‑means, LSH) and the overhead/instability of auxiliary supervision for selection.
   - Evidence: alternative trainable designs (auxiliary‑loss or heuristic selection) underperform on loss curves compared to NSA (Figure 7).

These are fundamental innovations rather than small tweaks: they change how sparse attention is constructed (hierarchical, differentiable), how it interacts with modern attention architectures (GQA/MQA), and how the kernels are scheduled to realize speedups across all phases.

## 5. Experimental Analysis
- Evaluation setup (Section 4)
  - Models: 27B MoE+GQA backbone; both NSA and full attention are pretrained on 270B tokens (8k context) and then continued to 32k with YaRN. NSA uses the hyperparameters listed above.
  - Benchmarks:
    - General knowledge/reasoning/coding: MMLU, MMLU‑PRO, CMMLU, BBH, GSM8K, MATH, DROP, MBPP, HumanEval (Table 1).
    - Long‑context: LongBench subsets (Table 2), Needle‑in‑a‑Haystack (NiH) 64k (Figure 5).
    - Chain‑of‑thought reasoning: AIME’24 via SFT distilled from DeepSeek‑R1; evaluate with 16 samples per question at T=0.7, top‑p=0.95 under 8k and 16k generation limits (Table 3).
  - Kernel speed comparison: Triton NSA kernel vs Triton FlashAttention‑2 (Figure 6). Decoding memory‑access analysis (Table 4).

- Main results
  - General benchmarks (Table 1):
    - NSA achieves higher average score than full attention: 
      > Average 0‑shot: NSA 0.456 vs Full 0.443.
    - Notable gains: 
      > GSM8K +0.034 (0.520 vs 0.486), DROP +0.042 F1 (0.545 vs 0.503), BBH +0.024 (0.521 vs 0.497).
    - Slight trade‑offs in some coding metrics (MBPP 0.466 vs 0.482).
    - Training loss curves show stable convergence with NSA lower than full attention (Figure 4).

  - Long‑context accuracy:
    - NiH retrieval at 64k: 
      > Perfect retrieval across all positions (Figure 5).
    - LongBench (Table 2): NSA has the best average,
      > Avg 0.469 vs Full 0.437 and vs Exact‑Top 0.423.
      - Multi‑hop QA improvements:
        > HPQ: 0.437 vs 0.350 (+0.087); 2Wiki: 0.356 vs 0.305 (+0.051).
      - Code understanding:
        > LCC: 0.232 vs 0.163 (+0.069).
      - Passage retrieval:
        > PassR‑en: 0.905 vs 0.830 (+0.075).

  - Chain‑of‑thought math (Table 3):
    - After SFT, NSA‑R outperforms Full Attention‑R at both generation limits:
      > 8k: 0.121 vs 0.046; 16k: 0.146 vs 0.092.
    - Interpretation given in the paper: the pretrained sparse patterns help capture long‑range logical dependencies important to math reasoning (Section “Chain‑of‑Thought Reasoning Evaluation”).

  - Efficiency (Figures 1 and 6; Table 4):
    - Kernel speed (training/prefilling):
      > At 64k, NSA achieves ~9.0× forward and ~6.0× backward speedups over FlashAttention‑2 (Figure 6).
      - Speedup increases with sequence length (8k→64k).
    - Decoding memory traffic (Table 4): expected speedup ≈ linear in reduced KV loads:
      > 8k: 4×; 16k: 6.4×; 32k: 9.1×; 64k: 11.6×.
    - End‑to‑end stages (Figure 1 right):
      > Decode 11.6×, Forward 9.0×, Backward 6.0× on 64k sequences.

- Ablations and qualitative insights (Section 6)
  - Alternative selection strategies:
    - Auxiliary‑loss‑based and heuristic parameter‑free selection both yield worse training loss than NSA (Figure 7).
  - Attention visualization:
    - Full attention maps show blockwise clustering—nearby keys share similar importance (Figure 8), motivating blockwise selection.

- Overall assessment
  - The experiments comprehensively evaluate capability and efficiency. Gains are consistent on long‑context tasks and reasoning, with competitive general performance.
  - Speedups are grounded in kernel‑level comparisons and decoding memory modeling, not just FLOP counts.
  - NSA’s native trainability is evidenced by successful pretraining to convergence and improved downstream SFT results.

## 6. Limitations and Trade-offs
- Assumptions about attention structure
  - The approach leverages empirical “spatial continuity” of attention (Figure 8). Tasks where important tokens are highly scattered at fine granularity may be less amenable to blockwise selection, though the sliding window mitigates some risk.

- Hyperparameter sensitivity
  - NSA introduces several structural choices—`l`, `d`, `l'`, `n`, `w`—that may require tuning across domains and model sizes (Section 4.1 uses specific values). Fixed `n` imposes a hard cap on selected detail.

- Overheads and memory
  - Maintaining three branches with separate K/V and gates adds parameters and some memory/compute overhead. Compression and selection pre‑compute introduce extra steps, although they are designed to be lightweight and GPU‑friendly (Section 3.4).

- Hardware and architecture coupling
  - The key speed wins rely on GQA/MQA and the custom Triton kernel that exploits group‑wise sharing and continuous memory. Benefits may diminish on architectures lacking such features, or on very different hardware/memory hierarchies.

- Scope of evaluation
  - Results center on one 27B MoE+GQA backbone and A100 GPUs. Generalization to other sizes (especially very small or very large dense models), other accelerators, or multimodal settings remains to be validated.

- Short‑sequence regimes
  - For short contexts (e.g., 8k), speedups and even accuracy gains are smaller; full attention may suffice when latency is already low (Figure 6 shows modest gains at 8k).

## 7. Implications and Future Directions
- Impact on the field
  - NSA demonstrates that sparse attention can be made both trainable and hardware‑efficient, removing a key barrier to long‑context LLMs. This can shift pretraining and fine‑tuning practices toward natively sparse models rather than applying sparsity only at inference.

- Practical applications
  - Faster and cheaper training/inference for long documents, repository‑level code modeling, extended multi‑turn agents, and complex multi‑document reasoning. The perfect NiH retrieval at 64k (Figure 5) and LongBench gains (Table 2) specifically recommend NSA for retrieval‑heavy and multi‑hop tasks.

- Research avenues
  - Adaptive hyperparameters: learning `n`, `l`, `w` per layer or per sequence.
  - Richer compression: explore alternative learnable summarizers (e.g., attention‑based compressors) that remain kernel‑friendly.
  - Multi‑modal extension: extend hierarchical sparsity to vision/audio sequences with similar kernel principles.
  - Theoretical analysis: bounds on approximation error introduced by compression/selection and formal connections to attention’s intrinsic sparsity.
  - Integration with retrieval systems: combine NSA’s compressed scan with external memory indexes for even longer contexts.
  - Portability and systems work: optimized kernels for newer GPUs (e.g., Hopper) and other accelerators; plug‑and‑play support within popular inference/training stacks.

In sum, NSA is a coherent algorithm‑systems co‑design: a hierarchical sparse attention that is natively trainable and paired with kernels that ensure arithmetic‑intensity balance and KV‑sharing in GQA/MQA. The paper provides evidence of both capability (Tables 1–3; Figure 5) and end‑to‑end efficiency (Figure 1; Figure 6; Table 4), positioning NSA as a strong candidate for long‑context LLMs.
