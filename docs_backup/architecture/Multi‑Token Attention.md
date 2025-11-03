# Multi‑Token Attention

**ArXiv:** [2504.00927](https://arxiv.org/abs/2504.00927)
**Authors:** Olga Golovneva, Tianlu Wang, Jason Weston, Sainbayar Sukhbaatar
**Institutions:** 

## 🎯 Pitch

Multi-Token Attention (MTA) revolutionizes Transformer models by enabling attention weights to integrate signals from multiple query and key tokens and even different heads, effectively breaking the 'single-token attention' limitation. This innovation significantly enhances long-context retrieval and standard language modeling, opening new frontiers for tasks requiring subtle multi-token interactions, although it incurs higher computational costs pending further optimization.

---

## 1. Executive Summary
Multi-Token Attention (MTA) modifies the way attention weights are computed in Transformers so that each weight can depend on multiple query tokens, multiple key tokens, and even multiple heads at once. By convolving attention scores and probabilities across the query, key, and head dimensions, MTA overcomes the “single-token attention” bottleneck and substantially improves long‑context retrieval and standard language modeling, with small parameter overhead but currently higher training cost.

## 2. Context and Motivation
- Problem addressed
  - Standard attention computes each weight from a single pair of vectors: one query token against one key token via a dot product, then softmax normalization (Eq. 1, Section 2). This makes every attention decision conditioned on a single token pair.
  - Many tasks require jointly matching several pieces of evidence in context (e.g., “find the sentence that mentions both Alice and rabbit”). With ordinary attention, a head that can find “Alice” separately from a head that can find “rabbit” still lacks a built‑in mechanism to combine these signals to focus on where both appear together (Section 3, opening example).

- Why this matters
  - Long‑context performance is known to degrade, especially for content in the middle (“lost in the middle”), and tasks that require pinpointing locations deep in context (Needle‑In‑A‑Haystack, BabiLong) remain challenging (Section 1; references to Liu et al., 2024/2025; Kuratov et al., 2025).

- Prior approaches and gaps
  - “Sharpening” attention: sparsemax, adaptive temperature, and other softmax variants can make distributions more peaked but still rely on a single query–key similarity per weight (Related Work).
  - Cross‑head mixing: Talking Heads attention and Differential Transformer (DIFF) mix attention across heads to denoise, but do not integrate multi‑token interactions across the query and key dimensions (Sections 4.2, 5).
  - Shifting keys/values by one step (KV-shift) is a very special case of multi-token interaction (Related Work).

- Positioning
  - MTA extends attention in three axes simultaneously—query positions, key positions, and heads—by using convolution over attention scores/weights (Figures 1–2; Section 3). This lets a model combine multi-token cues to decide where to attend, rather than requiring those cues to be compressed into a single vector.

## 3. Technical Approach
MTA adds three components on top of standard multi‑head attention (Figures 1–2; Section 3):

1) Recap: standard attention (Section 2, Eq. 1)
- For each head, project hidden states `H` to queries `Q`, keys `K`, and values `V`.
- Compute raw scores (logits) `Â = QK^T / sqrt(d)`; apply a causal mask to forbid attending to future tokens; then softmax over keys to obtain attention weights `A`.
- Output is `AV`.

Key terms used below:
- “Attention logits” `Â`: the unnormalized scores before softmax.
- “Attention weights” `A`: the normalized probabilities after softmax (sum to 1 over keys).
- “Causal mask”: sets illegal (future) positions to negative infinity (or zero before convolution; see below) to prevent information leakage.
- “Convolution kernel”: a small learnable weight matrix/vector that slides across an axis, mixing local neighborhoods.

2) Key–Query convolution (Section 3.1; Eqs. 3–6)
- Goal: Let attention for query position `i` at key position `j` depend on nearby queries and nearby keys, not just on `(qi, kj)`.
- Mechanism:
  - Pre‑softmax option (Eq. 3): apply a 2D convolution `Conv2d_θ` over the logits `Â` across query and key axes, then softmax:
    - Intuition: the new score at `(i, j)` aggregates dot products from a window of queries around `i` and a window of keys around `j`, with learnable kernel weights `θ`.
    - Equation (4) spells this out: the score uses a weighted sum of `q_{i-i'} k_{j-j'}^T`, with masking to preserve causality.
  - Masking detail (Eq. 5): A practical implementation uses “mask to zero” (`Mask0`) before convolution so masked entries do not affect neighbors, then “mask to −∞” (`Mask-∞`) after convolution before softmax. This avoids bespoke CUDA kernels while preserving causality.
  - Post‑softmax option (Eq. 6): alternatively, apply convolution on the attention weights `A` themselves, making the interaction additive rather than multiplicative.
- Design choices:
  - Each head has its own kernel `θ` (heads can specialize).
  - Kernel sizes (`cq` along queries, `ck` along keys) set how far the multi-token interaction spans.
  - Padding with zeros ensures valid convolution at edges.

3) Head mixing convolution (Section 3.2; Eq. 7)
- Goal: Allow different heads’ attention maps to combine their evidence.
- Mechanism:
  - Group the `M` heads into non-overlapping groups of size `ch` and apply a 1D convolution (equivalently, a small learned linear mixing) across heads in each group.
  - Post‑softmax mixing example (Eq. 7): for a pair of heads `(1, 2)`, compute new weights `A_new^1 = w11 A^1 + w12 A^2` and `A_new^2 = w21 A^1 + w22 A^2`.
  - Pre‑softmax mixing similarly combines logits `Â`.
- Interpretation:
  - Appendix B shows this mixing roughly corresponds to increasing the effective rank of an attention head (post‑softmax mixing resembles using a higher‑rank value projection; pre‑softmax mixing resembles higher‑rank Q/K projections). This helps express more complex attention patterns than a single head can capture.

4) Putting it together (Section 3.3; Figure 1 right)
- MTA can perform both key–query convolution and head mixing either pre‑softmax, post‑softmax, or both. If both are pre‑softmax (or both post‑softmax), they are implementable as a single 3D convolution over query, key, and head group axes (Figure 2).
- After mixing, a “group normalization with scalar gating” is applied:
  - Group normalization standardizes activations within groups of channels to stabilize training.
  - A learnable sigmoid gate (a scalar per head or group) can turn heads “on/off” and counter imbalances with the residual stream (Section 3.3; ablations in Table 5).

5) How it solves the “Alice + rabbit” problem
- Separate heads (or nearby queries) can first highlight occurrences of “Alice” and “rabbit” individually.
- Key–query convolution then amplifies positions where both signals co-occur in proximity (across tokens within a sentence), and head mixing reinforces this across heads, producing a sharper joint focus than single-token attention can.

Implementation choices in main experiments (Section 4.2):
- 880M‑parameter models, trained on 105B SlimPajama tokens.
- Key–query convolution applied to every 4th layer; head convolution applied on all layers.
- Kernels fixed at `cq = 6`, `ck = 11`; heads mixed in groups of `ch = 16`.
- Both pre‑ and post‑softmax convolutions used by default (unless stated otherwise).

## 4. Key Insights and Innovations
- Multi-token conditioning of attention weights
  - Novelty: Each attention weight is computed from a neighborhood of queries and keys (2D convolution), rather than a single `(qi, kj)` pair (Section 3.1; Eqs. 3–6).
  - Significance: Enables attention to lock onto spans that require multiple cues (e.g., two entities in the same sentence), which standard attention cannot combine at the weight level.

- Cross-head evidence sharing via local head convolution
  - Novelty: Learnable mixing across groups of heads (Section 3.2) before or after softmax.
  - Significance: Lets heads reinforce or contrast each other’s maps. Appendix B relates this to increasing the effective rank, offering a principled reason for the observed gains.

- Practical, causal-safe convolution over logits/weights
  - Novelty: A simple two-mask trick (Eq. 5) avoids writing custom masked‑convolution kernels while preventing information leakage from the future.
  - Significance: Makes MTA easy to implement in standard frameworks.

- Training stabilization with group normalization and gating
  - Novelty: Group norm with scalar gates after mixing (Section 3.3). Ablations (Table 5) show it consistently improves perplexity and prevents runaway scaling when competing with residual connections.
  - Significance: Critical for making the added mixing layers train well across depth.

- Interpretable learned kernels
  - Observation: Learned key–query kernels range from identity‑like to diagonal “sequence matcher” patterns (Figure 4), edge detectors, and “priming” effects across queries (Section 4.5, Appendix H).
  - Significance: Provides mechanistic interpretability for how MTA sharpens retrieval (e.g., the diagonal kernel in Figure 4 aligns a query sentence with a matching sentence in the context).

## 5. Experimental Analysis
Setup and baselines (Sections 4.2–4.4):
- Pretraining: 880M‑param decoder‑only models on 105B SlimPajama tokens (Table 7 for hyperparams).
- Baselines: `Transformer` (vanilla), `DIFF Transformer` (Difference of two softmax maps), `Talking Heads` (linear mixing across heads pre/post‑softmax).
- Evaluation: validation perplexity per SlimPajama domain; zero‑shot benchmarks; long‑context tasks (LAMBADA, Needle‑In‑A‑Haystack, BabiLong).
- Metrics: Perplexity (lower is better) and accuracy (higher is better).

Main quantitative results
- Language modeling perplexity (Table 2):
  - Pretraining (avg across two runs): 
    - Quote: “MTA … Avg PPL 10.91,” vs Transformer 11.25, DIFF 11.15, Talking Heads 11.04.
  - After long‑context finetuning to 4K context:
    - Quote: “MTA … Avg PPL 10.65,” vs Transformer 11.02, DIFF 10.89, Talking Heads 10.88.
  - Takeaway: MTA consistently reduces perplexity; gains persist after context extension.

- Standard zero‑shot benchmarks (Table 3):
  - Quote: “Avg score 44.9” for MTA vs 43.7 (Transformer), 43.9 (DIFF), 44.4 (Talking Heads).
  - Notable per‑task improvements include BoolQ (62.1 vs 56.2 baseline) and WinoGrande (57.2 vs 56.4), with competitive results elsewhere.

- Long‑range dependency tasks:
  - LAMBADA (Table 4):
    - Quote: “MTA: 13.2 (standard) and 8.4 (OpenAI) perplexity,” improving over Transformer’s 17.6 and 9.5.
  - Needle‑In‑A‑Haystack (Figure 3; detailed curves in Appendix G/Figure 8):
    - Setups with 2/4/6/8/10 needles in 2k or 4k contexts. MTA shows the strongest accuracy across needle counts and depths, especially for deeper insertions.
  - BabiLong QA1–5 with distraction lengths 0K–4K (Figure 5 left; per‑task curves Figure 7):
    - Quote: “MTA consistently outperforms the baseline models,” with the most pronounced gap at 4K distraction.

- Toy task (Section 4.1; Table 1):
  - Task: find the block of letters containing two query letters; output all/first/last token(s) of the block.
  - Quote: “MTA ~0.1% error” on all variants vs Transformer’s high error and instability (e.g., 51.6% ± 43.1 on ‘All’ for N=5).
  - Significance: Isolates the multi‑token retrieval ability that standard attention lacks.

Ablations and diagnostics
- How many MTA layers? (Figure 5 right):
  - With key–query convolution on as few as 2 layers (head mixing on all layers), perplexity already beats baselines; 6 layers strikes a good performance/complexity balance.
- Head kernel size `ch` (Figure 6 left):
  - Larger mixing groups across heads steadily improve perplexity.
- Kernel initialization (Section 4.6):
  - Identity initialization converges faster and yields better final perplexity than zero or constant initializations.
- Component order and normalization (Table 5):
  - Pre‑ and post‑softmax combinations matter modestly (±0.01–0.04 PPL).
  - Group norm with depth scaling or scalar gating consistently helps; plain layer‑norm scaling underperforms.
- Scaling laws (Figure 6 right):
  - Across 300M/550M/1B sizes, MTA maintains a perplexity advantage over all baselines, indicating the effect is not size‑specific.

Complexity and overhead
- Parameter counts (Table 8, 880M scale):
  - Quote: “Transformer 876,553,728 params; MTA 876,583,320,” i.e., a negligible increase (~29K).
- Training speed and memory (Table 9, 32× H200, unoptimized MTA):
  - Quote: “Transformer 54.3k tokens/s, 17.5GB; MTA 5.7k tokens/s, 73.8GB.”
  - Cause: baseline uses optimized scaled dot‑product CUDA kernels; MTA’s convolutions currently do not.
- Limitation A explicitly notes incompatibility with popular optimized attention kernels and leaves runtime optimization to future work.

Kernel interpretability (Section 4.5; Figure 4 and Appendix H)
- Example: a diagonal key–query kernel amplifies matches between sequences of queries and keys, which locks onto the exact sentence containing the “needle” (Figure 4).
- Head kernels often learn “contrast” patterns (subtract one head from another), aligning with the denoising role of head mixing.

Finetuning MTA into pretrained models (Appendix I; Table 10)
- Continual training by inserting identity‑initialized MTA modules improves perplexity for the paper’s own 1.4B model and for Llama‑3.x models (1B/3B/8B) over short runs (5.3–10.5B tokens), e.g.:
  - Quote: “Llama‑3.1‑8B: 8.48 vs 8.53 Avg PPL after MTA finetuning,” and similar small but consistent wins at 1B and 3B scales.
- Indicates MTA can be retrofit without full retraining.

Do the experiments support the claims?
- Yes, particularly for long‑context retrieval and multi‑signal matching:
  - The toy task isolates the core ability (Table 1).
  - LAMBADA and multi‑needle consistently favor MTA (Tables 4; Figure 3).
  - Gains in standard perplexity and zero‑shot tasks show benefits beyond long‑context scenarios (Tables 2–3).
- The main caveat is engineering overhead and limited runtime optimization (Table 9), not modeling effectiveness.

## 6. Limitations and Trade-offs
- Computational cost (major practical limitation)
  - Current MTA implementation lacks optimized CUDA kernels compatible with popular fast attention ops (Limitation A). Empirically it uses ~4.2× memory and ~10× less tokens/s during training at 880M scale (Table 9).
  - The authors mitigate parameter growth by applying key–query convolution on only 1/4 of layers, but runtime overhead still stems from the convolutions.

- Design hyperparameters
  - Performance depends on kernel sizes (`cq`, `ck`, `ch`) and on which layers receive key–query convolution (Figure 5 right; Figure 6 left; Table 5). Good defaults are provided, but optimal choices may vary by model size and task.

- Scope of context lengths and tasks
  - Main long-context tests use 2k–4k tokens. Extrapolation to very long contexts (e.g., 32k–128k) is not evaluated; interaction with advanced positional schemes or memory mechanisms remains open.

- Interpretability vs. complexity
  - While some kernels are interpretable (sequence matchers, edge detectors), others are opaque. Understanding when/where specific kernels help is an open area (Section 4.5, Appendix H).

- Theoretical guarantees
  - Appendix B gives intuition relating head mixing to increased rank, but there is no formal bound showing when MTA strictly dominates standard attention; the case is empirical.

## 7. Implications and Future Directions
- How this changes the landscape
  - MTA reframes attention’s basic unit of decision from a single query–key pair to a multi‑token, multi‑head neighborhood. This is a conceptual shift: attention weights can now encode combinatorial “AND”-like evidence without relying solely on representational compression into single vectors.
  - For practitioners, MTA is a drop‑in module that can be retrofitted via identity initialization and finetuning (Appendix I), enabling a new knob for long‑context performance without rearchitecting the whole model.

- Follow‑up research enabled
  - Systems engineering: write fused, causal‑aware convolution kernels compatible with flash/scaled dot‑product attention to remove the current training overhead (Limitation A; Table 9).
  - Automated kernel design: learn kernel sizes or sparse patterns per layer/head; explore dynamic kernels conditioned on input or depth.
  - Theory: characterize expressivity gains relative to standard attention and identify minimal MTA configurations that provably capture certain multi‑token patterns.
  - Interoperability: combine MTA with retrieval, memory pruning/expiration, and long‑context position encodings or extrapolation methods (e.g., RoPE variants, DAPE v2).

- Practical applications
  - Long‑document QA and summarization where specific multi‑entity co‑occurrence matters.
  - Code or log analysis requiring conjunctions of cues across tokens and lines.
  - Multi‑needle retrieval and open‑book QA settings that benefit from sharper, compositional attention to dispersed evidence.

In short, Multi‑Token Attention offers a principled and empirically validated way to “compose” evidence at the attention-weight level. It substantially improves both general language modeling and long‑context retrieval, while leaving runtime optimization as the key barrier to immediate large‑scale deployment.
