# Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention

**ArXiv:** [2502.11089](https://arxiv.org/abs/2502.11089)
**Authors:** Jingyang Yuan, Huazuo Gao, Damai Dai, Junyu Luo, Liang Zhao, Zhengyan Zhang, Zhenda Xie, Y. X. Wei, Lean Wang, Zhiping Xiao, Yuqing Wang, Chong Ruan, Ming Zhang, Wenfeng Liang, Wangding Zeng
**Institutions:** DeepSeek (Peking University), Peking University, University of Washington

## 🎯 Pitch

Native Sparse Attention (NSA) introduces a revolutionary, sparse attention mechanism designed to align with modern GPU architectures, transforming theoretical speed improvements into true performance gains. This method significantly reduces computational demands of long-context models, enabling efficient training and execution without compromising accuracy, thereby making advanced natural language processing applications feasible and cost-effective in real-world scenarios.

---

## 1. Executive Summary
The paper proposes NSA (Native Sparse Attention), a new attention architecture that is both end‑to‑end trainable and engineered to align with modern GPU hardware. It replaces full quadratic attention with a hierarchical, query‑dependent sparse mechanism that combines block compression, block selection, and a local sliding window, then implements custom Triton kernels to realize the theoretical savings in practice. NSA matches or exceeds full‑attention models on general, long‑context, and reasoning benchmarks while delivering substantial speedups, especially at long sequence lengths (e.g., up to 11.6× in decoding at 64k tokens; Figure 1, right; Table 4).

## 2. Context and Motivation
- Problem addressed
  - Long‑context modeling is computationally expensive because standard attention scales quadratically with sequence length. At 64k contexts, attention dominates latency (estimated 70–80% of total latency in decoding; Section 1).
  - Many sparse attention methods reduce theoretical operations but fail to translate those savings into real wall‑clock speedups, and most are not designed for (or compatible with) training (Sections 2.1–2.2).

- Why it matters
  - Real systems increasingly need long contexts for repository‑level code, multi‑turn agents, and extended reasoning (Section 1). If attention remains the bottleneck, these capabilities are impractical due to cost/latency.

- Shortcomings of prior approaches (Section 2)
  - Phase‑restricted sparsity: some methods speed up only prefilling (processing an input prompt to build the key‑value cache) or only decoding (generating token by token), leaving the other phase near full‑attention cost (Section 2.1).
  - Incompatibility with modern architectures: newer decoding‑efficient designs such as `GQA` (Grouped‑Query Attention) and `MQA` (Multiple‑Query Attention) share key/value caches across heads. Several sparse methods pick different tokens per head, which forces loading the union of all heads’ selections, negating memory benefits (Section 2.1).
  - “Trainability” myths: approaches with discrete operations (e.g., k‑means, hashing) are not differentiable, so gradients cannot shape selection; or they require token‑granular random memory access that breaks fast attention kernels in training (Section 2.2).

- Positioning
  - NSA is designed “natively sparse” (sparsity present throughout pretraining and downstream training) and “hardware‑aligned” (blockwise, contiguous memory access; group‑wise KV sharing) so theoretical savings convert to real speedups in all phases: prefilling, decoding, and backpropagation (Sections 3–4; Figures 1 and 6).

## 3. Technical Approach
NSA restructures attention around three parallel branches and a gating mechanism (Figure 2). For a query at position `t`, NSA replaces attending to all previous tokens with three compact representations of the past:

- Definitions (as used here)
  - `KV cache`: stored keys (`k`) and values (`v`) from preceding tokens used during decoding to avoid recomputation.
  - `Prefilling`: processing the input context to fill the KV cache before generation.
  - `Arithmetic intensity`: compute operations per byte transferred; higher values are compute‑bound, lower ones are memory‑bound (Section 3.1).

Step‑by‑step (Sections 3.2–3.3; Equations 1–12; Figure 2):

1. Start from standard attention
   - Equation (1)–(2) defines normal attention: the output for query `q_t` is a normalized weighted sum over all past values `v_1:t`, with weights from dot‑products between `q_t` and keys `k_1:t`.

2. Replace the full key/value set with a compact, query‑specific set
   - NSA constructs “remapped” keys/values `K̃_t`, `Ṽ_t` tailored to query `q_t` (Eq. 3), then computes attention over them (Eq. 4) rather than all tokens.
   - It combines multiple remappings `c ∈ {cmp, slc, win}` via a learned gate (Eq. 5), where `g_t^c ∈ [0,1]` weights each branch’s contribution.

3. Three branches (C = {cmp, slc, win})
   - Compression (`cmp`): Coarse‑grained summaries
     - Partition the past into overlapping blocks of length `l` with stride `d` (typically `d < l` to reduce information loss).
     - Each block is mapped to a single compressed key (and value) via a learnable MLP `φ` that can include intra‑block positional encoding (Eq. 7). The result is a much shorter list of compressed tokens `K̃_cmp`, `Ṽ_cmp`.
     - Intuition: a quick global scan; cheap to compute, preserves block‑level semantics.

   - Selection (`slc`): Fine‑grained tokens only from important blocks
     - Goal: recover fine detail lost by compression but only where it matters.
     - Compute attention scores between `q_t` and compressed keys: `p_cmp = Softmax(q_t^T K̃_cmp)` (Eq. 8). These scores indicate which coarse blocks are relevant.
     - Translate those coarse scores to the selection blocks’ scheme. If block sizes differ (`l` for compression; `l'` for selection), aggregate based on spatial overlap (Eq. 9). For `GQA/MQA`, sum importance across query heads in the same group to force a common selection and avoid redundant memory reads (Eq. 10).
     - Pick the top‑`n` most important selection blocks (Eq. 11) and concatenate all tokens from them to form `K̃_slc`, `Ṽ_slc` (Eq. 12). The method also includes a small fixed set (e.g., the very first block and a couple of local blocks).
     - Intuition: compression branch cheaply proposes “where to look”; selection branch zooms in there with full token granularity.

   - Sliding window (`win`): Always keep the most recent local context
     - Keep the latest `w` tokens (`K̃_win = k_{t-w:t}`, `Ṽ_win = v_{t-w:t}`), capturing strong local dependencies (Section 3.3.3).
     - NSA uses separate key/value projections per branch to avoid “shortcut learning” where everything routes through the easy local branch. The three outputs are then combined with learned gates `g_t^c` (Eq. 5).

4. Hardware‑aligned kernel design (Section 3.4; Figure 3)
   - Why needed: training/prefilling are compute‑bound, decoding is memory‑bound (Section 3.1). Speed requires contiguous, blockwise memory access with high Tensor Core utilization.
   - Custom Triton kernels for selection branch:
     - Group‑centric query loading: for each time step, load all query heads in a `GQA` group together so they share the same sparse KV blocks (reduces redundant KV fetches).
     - Shared KV fetching: load selected blocks contiguously into on‑chip SRAM, process them there, then move outputs back to HBM.
     - Grid scheduling: outer loops over query positions on Triton’s grid; inner loops iterate over the contiguous selected KV blocks. This balances workloads across streaming multiprocessors.
   - Compression and sliding‑window branches reuse FlashAttention‑2 style kernels since they access contiguous blocks naturally.

5. Design choices and rationale
   - Blockwise (not tokenwise) sparsity: matches GPU memory systems and Tensor Cores; also aligns with observed attention “block clustering” (Figure 8).
   - Use compressed‑attention scores to drive selection: avoids extra indexing networks or discrete, non‑differentiable preprocessing; keeps selection computation cheap (Eqs. 8–10).
   - Group‑wise selection for `GQA/MQA`: preserves their decoding advantages by preventing per‑head scatter/gather (Section 2.1, Section 3.3.2).
   - Separate projections per branch + gating: reduce interference and stabilize training in the presence of a strong local prior (Section 3.3.3).

Hyperparameters in the main experiments (Section 4.1): compression block size `l=32`, stride `d=16`; selection block size `l'=64`, number of selected blocks `n=16` (with the first block and two local blocks always active); sliding window `w=512`. The backbone is a 27B‑parameter MoE transformer with 3B active params, `G=4` GQA groups, 64 heads total (`d_k=192`, `d_v=128`), trained on ~270B tokens at 8k then extended to 32k with YaRN (Section 4.1; Figure 4).

## 4. Key Insights and Innovations
1. Hierarchical sparse attention that is both global and local (Figure 2; Eqs. 5–12)
   - What’s new: jointly using compressed coarse tokens to guide a fine‑grained block selection, plus an explicit local window, with learned gating across branches.
   - Why it matters: preserves long‑range awareness and token‑level precision while remaining cheap enough to train end‑to‑end. This contrasts with many prior methods that either rely on fixed local windows or perform query‑aware selection without a cheap global scan.

2. Hardware‑aligned blockwise design with group‑wise KV sharing (Section 3.4; Figure 3)
   - What’s new: selection is enforced at the GQA group level, and kernels load contiguous KV blocks into SRAM per group. This achieves high arithmetic intensity and avoids scattered memory access that kills decoding throughput.
   - Why it matters: turns theoretical sparsity into actual wall‑clock speedups across forward, backward, and decoding. Many previous methods claimed FLOP reductions but lost the gains to memory and scheduling overhead (Section 2.1).

3. Native trainability without fragile auxiliary objectives (Sections 3.3–3.4; 6.1)
   - What’s new: block importance comes “for free” from compressed‑attention scores (Eq. 8) rather than from separate predictors with extra losses or non‑differentiable algorithms (e.g., k‑means, hashing).
   - Why it matters: enables full‑model pretraining with the sparse mechanism itself, avoiding the mismatch of training with full attention and pruning only at inference (Section 2.2). Figure 4 shows stable convergence with lower loss than the full‑attention baseline; Figure 7 shows alternative selection strategies have worse training loss on a 3B model.

4. Arithmetic‑intensity awareness across phases (Section 3.1; Section 5)
   - What’s new: the method and kernels are designed differently for training/prefilling (compute‑bound) and decoding (memory‑bound). Table 4 connects reduced KV loads directly to expected decoding speedups.
   - Why it matters: yields increasing speedups with longer sequences—precisely where attention becomes a bottleneck.

## 5. Experimental Analysis
- Evaluation setup (Sections 4, 5)
  - Backbone: 27B MoE transformer (3B active), `GQA`, 30 layers, hidden size 2560, heads=64, `d_k=192`, `d_v=128`; MoE uses 72 experts (top‑k=6) with first layer replaced by SwiGLU for stability (Section 4.1).
  - Training data: ~270B tokens at 8k context, then long‑context adaptation at 32k with YaRN. NSA hyperparameters as above (Section 4.1).
  - Baselines: Full attention; inference‑only sparse baselines H2O, InfLLM, Quest; and an “Exact‑Top” upper bound that selects exact top‑n tokens after computing full scores (Sections 4.2–4.3).
  - Kernels: NSA implemented in Triton and compared against Triton‑based FlashAttention‑2 (Section 5.1).

- General benchmark results (Table 1)
  - Benchmarks: MMLU, MMLU‑PRO, CMMLU (knowledge); BBH, GSM8K, MATH, DROP (reasoning); MBPP, HumanEval (coding).
  - Summary: NSA outperforms the full‑attention baseline on 7/9 metrics; improvements include DROP (+0.042 F1: 0.545 vs. 0.503) and GSM8K (+0.034: 0.520 vs. 0.486). Average score improves from 0.443 to 0.456.
  - Interpretation: Despite heavy sparsity, NSA retains or improves capability; the hierarchical mechanism appears to help the model focus on salient information (Section 4.3).

- Long‑context evaluation (Figure 5; Table 2)
  - Needle‑in‑a‑Haystack (64k): NSA achieves perfect retrieval accuracy across all positions (Figure 5).
  - LongBench: To equalize sparsity, each method receives a 2560‑token budget including 128 leading and 512 local tokens (Section 4.3). NSA achieves the highest average (0.469) vs Full Attention (0.437) and Exact‑Top (0.423). Notable gains: HPQ +0.087 (0.437 vs. 0.350), 2Wiki +0.051 (0.356 vs. 0.305), Passage Retrieval EN +0.075 (0.905 vs. 0.830), LCC (code) +0.069 (0.232 vs. 0.163).
  - Interpretation: The combination of compressed scanning and targeted selection preserves global recall and local precision better than fixed or heuristic patterns.

- Chain‑of‑thought reasoning after SFT (Table 3)
  - Setup: Distillation‑based SFT from DeepSeek‑R1 on 10B tokens of 32k math traces; compare NSA‑R vs Full Attention‑R on AIME ’24 with 16 samples per question (temperature 0.7, top‑p 0.95).
  - Results: NSA‑R surpasses Full Attention‑R at 8k (0.121 vs. 0.046) and 16k (0.146 vs. 0.092).
  - Interpretation: NSA’s sparse patterns do not hinder, and may even aid, extended reasoning sequences.

- Speed and efficiency (Figures 1, 6; Table 4)
  - Training/prefilling speed (Figure 6):
    - Forward speedups grow with context: 2.1× (8k), 3.8× (16k), 6.3× (32k), 9.0× (64k).
    - Backward speedups: 1.1× (8k), 2.0× (16k), 3.4× (32k), 6.0× (64k).
  - Decoding speed (Figure 1 right; Table 4):
    - KV tokens loaded per step: Full attention loads all tokens; NSA loads roughly “compressed + selected + window”.
    - Example expected speedups (Table 4): 4.0× (8k), 6.4× (16k), 9.1× (32k), 11.6× (64k).
  - Conclusion: NSA realizes substantial end‑to‑end speedups, especially at long sequences where attention is the bottleneck.

- Ablations/diagnostics (Section 6)
  - Alternative selection strategies:
    - Auxiliary‑loss‑based block predictors and parameter‑free heuristics (Quest‑style) both underperform NSA in training loss on a 3B model (Figure 7).
  - Attention visualization:
    - Full‑attention maps show blockwise clustering—nearby keys often share importance (Figure 8), justifying blockwise selection.

Overall, the experiments support the core claims: NSA maintains or improves accuracy while drastically reducing computation and memory traffic in all phases.

## 6. Limitations and Trade-offs
- Discrete selection remains non‑differentiable
  - The top‑`n` block choice (Eq. 11) is a hard selection; gradients do not flow through the indices. NSA mitigates this by deriving block scores from a differentiable compressed branch (Eq. 8), but the selection threshold itself is not learned. This could, in principle, reduce adaptability around decision boundaries.

- Dependence on hardware‑aligned assumptions
  - Gains rely on blockwise, contiguous memory and `GQA/MQA` KV sharing (Section 3.4). On architectures without strong Tensor Core performance or with different memory hierarchies, speedups may diminish.

- Per‑group shared selection vs. per‑head specialization
  - Enforcing the same sparse blocks across all heads in a `GQA` group (Eq. 10) minimizes memory traffic but may limit diversity among heads within that group.

- Hyperparameter sensitivity and engineering overhead
  - Performance depends on block sizes (`l`, `l'`), stride `d`, selected block count `n`, and window `w` (Section 4.1). The paper gives strong defaults but limited exploration of the full trade‑off surface.
  - Custom Triton kernels and careful scheduling are required; portability to other accelerators/backends may require re‑engineering.

- Sequence regimes where benefits shrink
  - At short contexts (e.g., 8k), speedups are smaller (Figure 6), and kernel overheads can reduce relative gains.

- Reporting discrepancies and external validity
  - The abstract mentions 260B tokens, while Section 4.1 mentions ~270B. Also, most results are on a single 27B MoE backbone; broader scaling and cross‑model validation would strengthen generality.

## 7. Implications and Future Directions
- How this changes the landscape
  - NSA demonstrates that sparse attention can be made native to training and aligned with hardware, delivering both capability and speed. This reduces the long‑context cost barrier and encourages training models that truly learn to use sparse patterns.

- Practical applications
  - Long‑document question answering, repository‑level code understanding, multi‑turn agents, and any application that needs 32k–64k (or longer) contexts with reasonable latency/cost. The perfect retrieval on 64k Needle‑in‑a‑Haystack (Figure 5) and strong LongBench/code results (Table 2) are particularly encouraging.

- Research directions
  - Differentiable or soft block selection: relax the top‑`n` step to enable gradient flow through selection while keeping hardware efficiency (e.g., sparse‑continuous relaxations that still map to contiguous blocks).
  - Adaptive group granularity: explore dynamic grouping that balances memory sharing with head diversity.
  - Learned compression operators: investigate richer block encoders than an MLP `φ` (Eq. 7), including cross‑block context or lightweight attention inside blocks.
  - Cross‑hardware portability: replicate kernels on H100, AMD GPUs, and specialized accelerators; study the impact of memory hierarchies and scheduling.
  - Task‑aware sparsity curricula: curriculum schedules that adjust `n`, `l`, or `w` during training to progressively bias the model toward more efficient patterns without hurting accuracy.

In sum, NSA offers a concrete path for bringing sparse attention from theoretical complexity reductions to practical end‑to‑end speedups, without sacrificing accuracy—and sometimes improving it—on the tasks that matter for long‑context language models.
