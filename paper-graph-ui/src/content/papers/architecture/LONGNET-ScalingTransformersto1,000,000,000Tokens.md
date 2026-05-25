# LONGNET: Scaling Transformers to 1,000,000,000 Tokens

**ArXiv:** [2307.02486](https://arxiv.org/abs/2307.02486)

## 🎯 Pitch

LONGNET introduces a novel 'dilated attention' mechanism that enables Transformers to handle sequences of over one billion tokens with linear computational complexity—a huge leap beyond previous methods limited by quadratic costs or expressivity bottlenecks. By structuring sparsity in a distance-aware way and supporting efficient distributed training, LONGNET unlocks truly long context windows, transforming what Transformers can learn, remember, and reason about—opening the door to treating entire corpora or even the internet as a single sequence. This paves the way for breakthroughs in tasks requiring long-range memory, multi-hop reasoning, and massive in-context learning, all within existing Transformer frameworks.

---

## 1. Executive Summary
LONGNET introduces dilated attention, a drop-in replacement for standard self‑attention that scales Transformer sequence length to the billion-token regime with near-linear compute. It achieves this by sparsifying attention in a structured, distance-aware way and by enabling an efficient distributed algorithm that keeps both computation and communication cost almost constant as length grows (Sections 2–3; Figure 5).

## 2. Context and Motivation
- Problem addressed
  - Transformers scale well in parameters and training tokens but not in sequence length because standard self‑attention has quadratic cost in the number of tokens N. This limits models’ ability to use very long contexts for reasoning, retrieval, and in‑context learning (Introduction; Table 1).
- Why this matters
  - Longer contexts unlock larger memory and receptive fields, capture long causal chains, and enable many‑shot in‑context learning while reducing reliance on spurious short‑range correlations (Introduction).
- Prior approaches and their gaps
  - Recurrent and state-space models increase effective context but either limit parallel training or underperform on common-length language tasks due to reduced expressivity (Introduction; citations [GGR22, SWL23, FDS+23, PMN+23, FPB+23]).
  - Efficient Transformer variants reduce cost via local windows, sparsity, low-rank/kernels, downsampling, recurrence, or retrieval ([CGRS19, WLK+20, CLD+21, etc.]). These help but:
    - Fixed local windows forget very early tokens.
    - Heuristic sparse patterns are hard to scale and tune.
    - Many methods do not scale to hundreds of millions of tokens, let alone 1B (Figure 1).
- Positioning of this work
  - LONGNET provides: (1) linear compute in sequence length and logarithmic dependency between any two tokens; (2) a distributed algorithm that parallelizes along the sequence dimension with nearly constant communication; and (3) a plug‑compatible attention module that reuses existing Transformer optimizations like FlashAttention (Sections 2–3; Table 1; Figure 5).

## 3. Technical Approach
This section explains how dilated attention works and how it enables billion‑token training/inference.

- Key idea in plain terms
  - Instead of letting every token fully attend to every other token (quadratic cost), dilated attention connects tokens with a pattern that becomes sparser as distance grows. Nearby tokens are attended densely; far tokens are sampled sparsely but systematically so information can still flow across the whole sequence with short “routing” paths (Sections 2.2–2.4; Figures 2–3).

- Building blocks
  - Segmenting and sparsifying (Equations 3–8; Figure 2)
    - The sequence is split into segments of length `w` (segment length).
    - Within each segment, only every `r`-th row/column is kept (dilation rate `r`), producing a reduced set `(Q̃, K̃, Ṽ)` for each segment.
    - Attention is computed on these reduced tensors in parallel: `Õi = softmax(Q̃i K̃i^T) Ṽi` (Eq. 6).
    - Outputs are then “scattered back” into their original positions, with zeros for positions that were skipped (Eq. 7), and segments are concatenated (Eq. 8).
    - Implementation detail: this can be implemented as a dense attention sandwiched between a gather (to create `(Q̃, K̃, Ṽ)`) and a scatter (to place `Õ` back). That lets LONGNET reuse high-performance kernels like FlashAttention (Section 2.2).
  - Mixture of multiple dilations (Equations 9–12)
    - A single dilation pattern can miss some connections. LONGNET mixes k dilated attention patterns with increasing `(wi, ri)` so:
      - Small segments with small dilation capture local details exactly.
      - Large segments with large dilation give global coverage at low cost.
    - The mixture is a weighted sum of the outputs from each pattern: `O = sum_i α_i O|ri,wi` (Eq. 9).
    - Weights `α_i` are normalized from each pattern’s softmax denominator `s_i` (Eq. 10). This dynamic weighting better balances the patterns than fixed learned scalars (Section 2.2).
    - Design choice: `w` and `r` are geometric sequences (each grows by a constant factor), producing an exponential growth of the receptive field with pattern index (Eqs. 11–12).
  - Multi‑head coverage via shifting (Equations 13–15; Figure 3)
    - For head j, sparsification is offset by `s_j = j mod r`. This shifts which tokens each head samples within a segment, so different heads cover disjoint subsets and together approximate full coverage.
- Complexity and token dependency
  - Compute cost
    - For one pattern `(r, w)`, the FLOPs are `2N w d / r^2` because each segment is reduced to size `w/r` (Eq. 16).
    - For k patterns, FLOPs are `2 N d sum_i (w_i / r_i^2)` (Eq. 17).
    - With geometric growth of `w_i` and `r_i`, the sum is bounded by a constant factor times `N d`, so the overall complexity is O(N d) (Eq. 18).
  - Token dependency path length
    - With exponentially growing segment sizes, the maximum distance D information can travel in l mixed layers grows exponentially with l (Eq. 19).
    - Therefore, the number of layers needed to connect two arbitrary tokens grows only logarithmically with sequence length: `L ≈ log_α( N(α−1)/w0 )` (Eq. 20). Intuition: the mixture creates “express lanes” that hop further at higher patterns.
- Distributed training along the sequence dimension (Section 3; Figure 4)
  - Setup
    - Split the full sequence across devices along the token axis: `X = [X1, X2, ...]` (Eq. 21).
    - Each device computes its local projections `Qi, Ki, Vi = WQ/WK/WV Xi` (Eq. 22).
  - Local vs. global segments
    - If a dilated pattern has segment length `w ≤ l` (the number of tokens on a device), attention is computed locally using the gathered/sparsified `(Q̃, K̃, Ṽ)` (Eqs. 3–8).
    - If `w > l`, keys/values span multiple devices. Devices all‑gather only the sparse `K̃, Ṽ` (Eq. 23), compute attention with local queries `Q̃` (Eq. 24), then concatenate outputs (Eq. 25).
  - Why communication stays small
    - The all‑gathered `K̃, Ṽ` sizes do not depend on total N but on `w/r`, which is controlled by the dilation. This keeps cross‑device communication essentially constant as N grows (Section 3.1).
- Practical recipe used to reach 1B tokens (Section 3.2; Figure 5)
  - Use ≤3 patterns with segment lengths set to {2,048; tokens per device; full sequence length}.
  - Maintain a fixed number of tokens per batch (1B overall) while increasing sequence length, and measure forward-runtime over 10 runs.
  - Equip both LONGNET and the dense baseline with FlashAttention for a fair runtime comparison.

Illustrative analogy: think of the sequence as a long road. Local lanes (small `w`, `r=1`) cover neighborhood streets thoroughly. Express lanes (large `w`, large `r`) have fewer stops but travel far quickly. Multiple heads shift their stops so, collectively, they “see” most of the road with far fewer total stops than visiting every house.

## 4. Key Insights and Innovations
- Dilated attention with exponential receptive field
  - What’s new: a structured sparsity pattern whose density decays with distance, implemented as a mixture of dilations with geometric segment sizes and rates (Section 2.2; Eqs. 9–12; Figure 2).
  - Why it matters: yields O(N d) compute and O(log N) token dependency without giving up access to distant tokens (Eqs. 18–20), unlike fixed local windows that forget long‑range context.
- Drop‑in, kernel‑friendly implementation
  - What’s new: realize the sparse pattern via gather–dense‑matmul–scatter, letting the method reuse FlashAttention and other optimized kernels (Section 2.2).
  - Why it matters: avoids engineering overhead and performance regressions that often plague custom sparse kernels.
- Sequence‑dimension distributed algorithm with constant communication
  - What’s new: a simple split of the sequence across devices with all‑gather of only sparse `K̃, Ṽ` for large segments (Section 3.1; Figure 4).
  - Why it matters: practical path to 1B‑token contexts with almost constant per‑step runtime (Figure 5) and without specialized hardware assumptions.
- Dynamic mixing weights tied to attention normalization
  - What’s new: combine patterns using weights derived from each pattern’s softmax denominator (Eq. 10), instead of learning fixed scalars.
  - Why it matters: aligns mixing with attention statistics and empirically works better than fixed weights (Section 2.2).

These are fundamental innovations (new attention pattern and distributed scheme) rather than incremental tweaks.

## 5. Experimental Analysis
- Evaluation methodology
  - Tasks and data
    - Language modeling on The Stack (a large code dataset) tokenized with `tiktoken` `cl100k_base` (Section 4.1).
  - Models and architecture
    - Base architecture: MAGNETO with XPOS positional encoding; 12 layers, 768 hidden size, 12 heads (Section 4.1).
    - Replace standard attention with dilated attention for LONGNET.
  - Training setup
    - 300K steps; 0.5M tokens per batch; Adam with β=(0.9, 0.98); polynomial LR decay; LR 6e-4; no dropout; weight decay 0.01 (Appendix Table 3).
    - For scaling‑law study: 125M–2.7B parameter models; the 2.7B model trains on 300B tokens, smaller ones on ~40B tokens (Section 4.4; Appendix Table 4).
  - Baselines
    - Dense Transformer (standard attention).
    - Sparse Transformer with fixed patterns per [CGRS19], tuned to match FLOPs with LONGNET; block size 2048; heads attend distinct subblocks (Section 4.2).
  - Inference beyond training length
    - Use blockwise causal attention (BCA) to extrapolate when evaluation context exceeds model’s trained length; remove absolute position encoding (Section 4.2).
  - Metrics
    - Perplexity (PPL) on The Stack at test lengths from 2K to 32K (Section 4.2; Table 2).
- Main quantitative results
  - Runtime scaling (Figure 5)
    - With FlashAttention for both methods, dilated attention shows near‑constant runtime as length grows from 8K to 1B, while dense attention increases dramatically due to quadratic scaling.
    - Quote: “Dilated attention can successfully scale up the sequence length with almost constant latency… vanilla attention suffers from the quadratic dependency… There is no distributed algorithm for vanilla attention to break sequence length limitation.” (Section 3.2; Figure 5).
  - Language modeling performance (Table 2)
    - With similar compute (FLOPs matched for sparse vs LONGNET), LONGNET consistently achieves lower PPL than Sparse Transformer across training lengths 8K, 16K, 32K.
      - Example at training length 32K: LONGNET test PPLs are 4.37 (2K), 3.33 (8K), 3.01 (32K), while Sparse Transformer is 5.15, 4.00, 3.64.
    - Compared to dense Transformer trained at 2K (due to cost), LONGNET at longer training lengths maintains low PPL even at long test contexts (e.g., 3.24 at 8K in the 8K‑trained setting).
  - Sequence-length scaling curves (Figure 6)
    - As training context increases (1K → 32K), both dense Transformer and LONGNET improve test PPL, but LONGNET achieves a lower loss at substantially lower compute for the same test length.
    - Interpretation: training with long contexts is more effective than relying on inference‑time extrapolation alone; and LONGNET learns long‑range dependencies more compute‑efficiently (Section 4.3).
  - Model‑size scaling (Figure 7a)
    - LONGNET follows a smooth power‑law relationship between compute and test loss from 125M to 2.7B parameters, mirroring dense Transformers’ scaling behavior (Section 4.4).
  - Long‑context prompting (Figure 7b)
    - Holding suffixes fixed and increasing prompt length from 1K to 32K steadily lowers test loss (approx. from ~2.1 to ~1.6), indicating better use of longer contexts (Section 4.5).
- Do the experiments support the claims?
  - Efficiency claims are strongly supported by Figure 5’s runtime scaling and the distributed design (Section 3.1–3.2).
  - Effectiveness claims are supported within the tested domain (code LM on The Stack): Table 2, Figure 6, and Figure 7 show that LONGNET is competitive or superior to matched‑compute sparse baselines and benefits from longer training contexts.
- Ablations and robustness
  - The paper qualitatively notes that dynamic softmax‑based mixing outperforms fixed learned weights (Section 2.2) but does not provide a separate ablation table.
  - Design choices like geometric progression for `w` and `r`, and the precise number of patterns (≤3 in 1B‑length runtime tests) are motivated but not systematically ablated.
- Caveats
  - The 1B‑token result is a runtime/feasibility benchmark for the attention kernel and distributed scheme, not a full end‑to‑end trained LM at that length (Section 3.2; Figure 5).
  - Experiments focus on code modeling; generalization to diverse natural‑language tasks or multimodal settings is proposed as future work (Conclusion).

## 6. Limitations and Trade-offs
- Approximation vs. exactness
  - Far‑range attention is sparse and approximate; fidelity depends on how `w` and `r` are chosen. Very long‑range interactions may be underrepresented if the mixture has too few patterns or too aggressive dilation (Section 2.2).
- Hyperparameter sensitivity
  - Performance depends on the geometric schedules of segment lengths and dilation rates (Eqs. 11–12). The paper suggests reasonable defaults but lacks an extensive ablation grid.
- Evaluation scope
  - Main LM results are on a single code dataset. There is no comprehensive comparison across diverse long‑document NLP benchmarks or tasks like QA and summarization. Long‑range arena (LRA) or retrieval‑heavy tasks are not reported here.
- Training length vs. inference length
  - While BCA helps extrapolation at inference, Table 2 and Figure 6 show extrapolation degrades when test length greatly exceeds training length. LONGNET still benefits from training with longer contexts.
- Distributed assumptions
  - The “constant” communication cost in sequence parallelism depends on the chosen `w/r` for the largest pattern. If `r` is set too small relative to very large `w`, all‑gather volume can rise (Section 3.1).
- Memory and system considerations
  - Although attention FLOPs are linear, full end‑to‑end training at 1B tokens still requires careful memory planning (optimizer state, activation checkpointing, etc.). The paper demonstrates kernel/runtime feasibility (Figure 5) but not the full training cost at 1B.

## 7. Implications and Future Directions
- How this changes the field
  - Introduces a practical, general-purpose way to expand context windows to extreme lengths without sacrificing standard Transformer infrastructure. This removes a major bottleneck for long‑context learning and opens the door to treating massive corpora—even the web—as a single sequence (Abstract; Introduction).
- Follow‑up research enabled
  - Thorough task‑level evaluations on long‑document NLP, long‑horizon reasoning, program synthesis with extended repositories, and multi‑hop retrieval.
  - Integration with memory/retrieval systems to combine structured sparse global attention with learned external memory (related to [WDC+23]).
  - Automated schedule search for `(w, r)` and number of patterns, possibly learned per layer or per head.
  - Theoretical analysis of approximation error vs. dilation schedule and of optimization dynamics with O(log N) dependency paths.
- Practical applications
  - Long‑context prompting for code assistants and LLMs where entire repositories or large projects fit into context (Figure 7b).
  - Processing extremely long sequences in genomics, continuous logs, or legal/financial documents where global dependencies matter.
  - Multimodal long‑horizon modeling (proposed future directions include BEiT and multimodal LLMs; Conclusion).

Overall, LONGNET contributes a principled, efficient mechanism—dilated attention—for billion‑token contexts, a simple yet powerful distributed scheme, and compelling initial evidence on language modeling efficiency and scaling behavior. Further, broader task evaluations and ablations would clarify its limits and best practices.
