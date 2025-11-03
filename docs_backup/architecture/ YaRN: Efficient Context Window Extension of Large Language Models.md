# YaRN: Efficient Context Window Extension of Large Language Models

**ArXiv:** [2309.00071](https://arxiv.org/abs/2309.00071)
**Authors:** Bowen Peng, Jeffrey Quesnelle, Honglu Fan, Enrico Shippole
**Institutions:** 

## 🎯 Pitch

YaRN introduces a novel method to extend the usable context window of transformer language models using rotary position embeddings, achieving reliable operation at up to 128k tokens with minimal fine-tuning. This approach significantly enhances the practical applicability of large models in tasks requiring long-span attention, such as legal or technical document processing, by offering a compute-efficient solution that maintains benchmark performance and compatibility with fast attention kernels.

---

## 1. Executive Summary (2-3 sentences)
The paper introduces YaRN (“Yet another RoPE extensioN”), a compute‑efficient method for extending the usable context window of transformer language models that use rotary position embeddings (`RoPE`). It combines a targeted frequency‑aware interpolation of positional frequencies with a simple attention‑logit scaling, enabling LLaMA/Llama‑2 and Mistral models to operate reliably at 64k–128k tokens with minimal fine‑tuning, while maintaining standard benchmark performance and compatibility with fast attention kernels (Section 3.4; Tables 2–3; Appendix B.4).

## 2. Context and Motivation
- The specific problem:
  - Transformer LLMs trained with `RoPE` encode token positions as rotations, but they typically fail to generalize beyond the maximum context length used in pretraining; perplexity spikes and retrieval degrades when going past this limit (Section 1; Figure 1).
- Why it matters:
  - Many real tasks (e.g., long-document summarization, codebases, legal/technical documents) require attending to tens or hundreds of thousands of tokens. Extending context post‑pretraining avoids the cost of retraining large models from scratch and broadens practical usability (Section 1).
- Prior approaches and gaps:
  - `Position Interpolation (PI)` scales positions by a factor `s` to “stretch” a fixed context but tends to:
    - Lose high‑frequency positional detail (hurting local token order) and slightly degrade short‑context performance (Section 3.1).
    - Require substantial fine‑tuning tokens (billions) to work well (Section 2.2; [9]).
  - `NTK-aware` interpolation rescales the RoPE frequency base so high frequencies are compressed less, improving zero‑finetune extrapolation but:
    - Slightly extrapolates some dimensions “out‑of‑bounds,” making its stated `s` not match the effective scale and yielding weaker results with fine‑tuning compared to PI (Section 3.1; Def. 1; Eq. 16).
  - `Dynamic NTK` (inference‑time dynamic scaling of `s`) improves zero‑finetune behavior but does not address the frequency‑targeting or short‑context degradation (Section 3.3).
  - Methods that alter attention (e.g., ReRoPE, LM‑Infinite) can extend length but incur incompatibilities (e.g., not working with FlashAttention 2, or requiring two passes) and are not pure embedding‑level approaches (Section 2.4).
- This paper’s position:
  - It consolidates and refines prior ideas into a single, lightweight recipe—`YaRN`—that:
    1) targets which RoPE frequencies to interpolate and by how much (“by parts”);
    2) adds a cost‑free attention‑logit scaling that stabilizes training/inference at long lengths; and
    3) optionally uses dynamic scaling at inference to “gracefully” extend past trained limits (Sections 3.2–3.4).

## 3. Technical Approach
At a high level, RoPE encodes each token position `m` as a rotation in the complex plane applied to query/key vectors. The dot product between a query at position `m` and a key at position `n` depends only on their relative distance `m−n` (Section 2.1; Eqs. 1–9). YaRN modifies how those rotations are parameterized across dimensions and how the attention logits are scaled.

Step 0 — Background: How `RoPE` encodes position
- Each pair of hidden dimensions is interpreted as a complex dimension and rotated by an angle proportional to token position. Different pairs use different frequencies `θ_d = b^(-2d/|D|)`, where `b=10000` (Section 2.1).
- Wavelength `λ_d = 2π/θ_d = 2π b^(2d/|D|)` grows exponentially with dimension index `d` (Eq. 13). Low‑index dimensions are high‑frequency (short wavelength), high‑index dimensions are low‑frequency (long wavelength).

Step 1 — Baseline extension: `Position Interpolation (PI)`
- Idea: scale down positions by `s` when building RoPE, i.e., use `g(m)=m/s`, keeping `θ_d` the same (Section 2.2; Eq. 10 rewritten as Eq. 12).
- Limitation: compresses all frequencies equally, shrinking local angular differences among nearby tokens, which removes high‑frequency detail needed for precise local order (Section 3.1).

Step 2 — Preserve high frequencies: `NTK-aware` base change
- Idea: instead of scaling positions, change the RoPE base from `b` to a new base `b'` so that high‑frequency dimensions are compressed less and low‑frequency dimensions more (Section 3.1; Def. 1).
- Concretely, keep `g(m)=m` and modify `h(θ_d)` by replacing `b` with `b' = b * s^(|D|/(|D|−2))` (Eq. 16). This redistributes the “interpolation burden” across dimensions.
- Trade‑off: some dimensions slightly extrapolate beyond pretraining ranges, which is helpful without fine‑tuning but suboptimal when fine‑tuning (Section 3.1).

Step 3 — Target interpolation where it helps: `NTK-by-parts`
- Motivation: Treating all dimensions the same (“blind” interpolation) harms local relationships. Instead, interpolate only where the model relies more on absolute than relative position, and leave truly local (high‑freq) channels untouched (Section 3.2).
- Mechanism:
  - Define `r(d) = L / λ_d`, the ratio of the pretraining context `L` to the wavelength at dimension `d` (Eq. 17).
  - Choose thresholds `α` and `β` to partition dimensions:
    - If `r(d) < α`: wavelength ≥ L (very low frequency). Interpolate fully (like PI) by using `θ_d/s`.
    - If `r(d) > β`: wavelength ≪ L (very high frequency). Do not interpolate; keep `θ_d`.
    - If `α ≤ r(d) ≤ β`: interpolate partially using a linear ramp `γ(r)` from 0 to 1 (Eq. 18).
  - Implement as a convex combination over the RoPE frequency (Def. 2; Eqs. 19–20):
    - `h(θ_d) = (1−γ(r(d))) * (θ_d/s) + γ(r(d)) * θ_d`.
- Typical hyperparameters: for LLaMA family, `α = 1`, `β = 32` worked well (Section 3.2).

Intuition with a toy analogy:
- Think of RoPE dimensions as rulers of different granularity. Fine rulers measure local token order; long rulers measure global position. PI shrinks all rulers equally—making fine rulers too blunt. NTK‑by‑parts leaves the fine rulers unchanged, only stretching the long rulers and gradually blending for medium ones.

Step 4 — Stabilize attention at long lengths: YaRN’s attention scaling
- Observation (Appendix A.2; Figures 2–4): When sequences get longer (large `s`), scaling the attention logits before softmax improves perplexity consistently across documents and token positions.
- Mechanism (Section 3.4):
  - Replace the attention softmax with a temperature `t`: `softmax(q^T k / (t √|D|))` (Eq. 21).
  - Equivalently (and cheaply), scale both `q` and `k` by `√(1/t)` via the RoPE embedding (the “length scaling” trick). This preserves compatibility with fast attention kernels and adds virtually no overhead.
  - Set the scale empirically as a simple function of `s` that fits several LLaMA variants: `1/t = 0.1 ln(s) + 1` (Eq. 22). This is surprisingly consistent across models and token positions.
- Why this helps:
  - Interpolation compresses angular separations between nearby tokens, which tends to concentrate softmax probabilities. A mild temperature (>1) deconcentrates them, counteracting overconfidence and restoring effective capacity at long ranges.

Step 5 — Optional inference‑time boost: `Dynamic Scaling`
- In generation or streaming inference, current sequence length grows from 1 to the target maximum. Instead of fixing `s = L'/L`, update `s = max(1, l'/L)` at each forward pass, where `l'` is the current length (Section 3.3).
- Benefits:
  - Avoids premature degradation below the trained limit, and bends (rather than breaks) beyond it (Section 3.3; Appendix B.3, Figure 5).
- Implementation note with KV‑cache:
  - Cache key/value tensors before applying RoPE, because RoPE depends on `s`, which changes across steps (Section 3.3).

Putting it together — `YaRN`
- Definition (Section 3.4; Def. 3): YaRN = `NTK-by-parts` interpolation (targeted per dimension) + attention scaling (Eq. 21), with optional `Dynamic Scaling` at inference.
- Training recipe used in the paper (Section 4.1):
  - Base models: Llama‑2 7B and 13B; later, Mistral 7B v0.1 (Appendix B.4).
  - Data: PG19 book corpus chunked into 64k sequences for Llama‑2 (Section 4.1); Long‑Data‑Collections for Mistral (Appendix B.4).
  - Hyperparams: AdamW, lr 2e‑5, β1=0.9, β2=0.95, no weight decay, 20 warmup steps, FlashAttention‑2 + FSDP (Section 4.1).
  - Schedule: train `s=16` for 400 steps (global batch 64). Then start from that checkpoint and train `s=32` for 200 more steps (Section 4.1).

## 4. Key Insights and Innovations
- Targeted, frequency‑aware interpolation (“by parts”), not one‑size‑fits‑all:
  - Novelty: Uses the wavelength ratio `r(d)` to decide how much each RoPE dimension should be interpolated, preserving high‑frequency (local) channels and stretching low‑frequency (global) ones (Section 3.2; Eqs. 17–20).
  - Significance: Maintains local order sensitivity and avoids the short‑context degradation observed with blind interpolation like PI (Section 3.2). This is a conceptual shift from uniform to targeted positional scaling.
- Zero‑overhead attention scaling tied to context extension:
  - Novelty: A simple pre‑softmax scaling `t` implemented as re‑scaling of RoPE output, with the empirical rule `1/t = 0.1 ln(s) + 1` (Section 3.4; Eq. 22).
  - Significance: Robustly improves perplexity across documents and token positions without changing the attention kernel or adding inference cost (Appendix A.2; Figures 2–4).
- Dynamic scaling at inference:
  - Incremental but practical improvement: Adjust `s` with current length to gracefully extrapolate and prevent sharp failures at or beyond trained limits (Section 3.3; Appendix B.3).
  - Significance: Particularly helpful for zero‑finetune scenarios and compatible with KV‑caching if implemented carefully (Section 3.3).
- Compute‑efficient training and transfer:
  - Novelty: Shows effective 64k and 128k extensions with roughly 0.1% of original pretraining tokens and only 400–600 fine‑tuning steps (Section 4; 4.1).
  - Significance: 10× fewer tokens and 2.5× fewer steps than prior PI‑based methods (e.g., [9]) while outperforming them, enabling longer contexts under tight compute budgets (Abstract; Section 4).

## 5. Experimental Analysis
Evaluation setup
- Metrics and datasets:
  - Long‑sequence modeling via sliding‑window perplexity (window S=256) on Proof‑Pile and GovReport (Sections 4.3.1; B.1; Figure 1; Tables 1–2, 4).
  - Passkey retrieval accuracy: synthetic task placing a 5‑digit key at random positions up to 128k (Section 4.3.2; Table 5).
  - Standard benchmarks: ARC‑Challenge (25‑shot), HellaSwag (10‑shot), MMLU (5‑shot), TruthfulQA (0‑shot) (Section 4.3.3; Table 3).
- Baselines:
  - PI‑based Together 32k; “NTK‑aware” Code Llama 100k; original Llama‑2; for Mistral, base v0.1 and MistralLite (NTK‑aware) (Sections 4.3.1; Appendix B.4).

Main quantitative results
- Long‑sequence perplexity (Proof‑Pile; Table 2; Figure 1):
  - 7B:
    - `YaRN s=32 (128k)` shows perplexity 2.45 at 65k, 2.36 at 98k, and 2.37 at 128k.
    - “NTK‑aware” Code Llama 100k: 2.55 at 65k, 2.54 at 98k, rising to 2.71 at 128k.
    - Together 32k fails beyond 32k (perplexity explodes >10² at 65k+).
  - 13B:
    - `YaRN s=32 (128k)`: 2.31 at 65k, 2.23 at 98k, 2.24 at 128k.
    - Code Llama 100k: 2.41 at 65k, 2.37 at 98k, degrades to 2.54 at 128k.
- Short‑to‑medium lengths (Proof‑Pile; Table 1):
  - Extending Llama‑2 7B from 4k→8k:
    - At 8,192 tokens: PI 3.34, NTK‑aware 3.59, `YaRN` 3.35.
    - At 10,240 tokens (beyond trained window): `YaRN` 6.04 vs PI 8.07 vs NTK‑aware 6.24.
  - Takeaway: `YaRN` matches PI at the target length and is more stable beyond it with fewer training steps and tokens (Table 1 vs Section 4).
- GovReport 32k perplexity (Table 4):
  - 7B: `YaRN s=16` achieves 3.59 vs Together 32k at 3.67 and Code Llama 100k at 4.44.
  - 13B: `YaRN s=16` at 3.35 vs Code Llama 100k at 4.22.
- Passkey retrieval (Table 5):
  - 7B: `YaRN s=32` achieves 99.4% through 128k; Code Llama 100k achieves 94.3% at up to ~112k.
  - 13B: `YaRN s=32` achieves 99.4% through 128k; Code Llama 100k 99.4% at 128k.
  - Authors note that passkey accuracy can remain high even when perplexity worsens, suggesting perplexity alone is not a full measure of long‑context usability (Appendix B.2).
- Standard benchmarks (Table 3):
  - 7B:
    - Llama‑2 baseline: ARC 53.1, HellaSwag 77.8, MMLU 43.8, TruthfulQA 39.0.
    - `YaRN s=16`: 52.3, 78.8, 42.5, 38.2.
    - `YaRN s=32`: 52.1, 78.4, 41.7, 37.3.
    - Code Llama 100k: markedly worse (e.g., HellaSwag 60.8, MMLU 31.1).
  - 13B:
    - Llama‑2 baseline: 59.4, 82.1, 55.8, 37.4.
    - `YaRN s=16`: 58.1, 82.3, 52.8, 37.8.
    - `YaRN s=32`: 58.0, 82.2, 51.9, 37.3.
    - Code Llama 100k: much lower (e.g., HellaSwag 63.4, MMLU 32.8).
  - Takeaway: `YaRN` preserves general knowledge/task performance with minimal degradation versus baseline Llama‑2, unlike some prior long‑context models.
- Mistral extension (Appendix B.4; Figure 6; Table 6):
  - Base Mistral v0.1 (8k) and MistralLite (16k) fail at long lengths (>16k), while `YaRN s=16 (128k)` achieves 2.24 at 65k and 2.19 at 128k.

Ablations and robustness checks
- Attention‑scaling ablation across positions/documents (Appendix A.2; Figures 2–4):
  - Shows a broad, consistent improvement from the pre‑softmax scaling factor, with best `1/√t` around the rule in Eq. 22, across multiple token positions.
- Dynamic scaling without any fine‑tuning (Appendix B.3; Figure 5):
  - For Llama‑2 at 4k pretrain length, `Dynamic-YaRN` prevents the perplexity blow‑up beyond 4k and outperforms `Dynamic-PI`.
- Missing or limited ablations:
  - No detailed study varying `α, β` thresholds, or the exact ramp shape in `NTK-by-parts`.
  - Limited analysis of how `t` generalizes beyond LLaMA/Llama‑2/Mistral families or outside the tested scale factors.

Do the experiments support the claims?
- Yes, for the primary claims:
  - YaRN yields lower perplexity at long lengths and sustains capability to 128k for Llama‑2 7B/13B and Mistral 7B, surpassing prior RoPE‑based extensions (Tables 2, 6; Figure 1).
  - It preserves standard benchmark performance (Table 3) and shows strong retrieval (Table 5).
  - It achieves this with far fewer training tokens/steps than prior methods (Sections 4, 4.1).
- Caveats:
  - Perplexity and passkey retrieval do not fully capture all aspects of long‑context reasoning/composition.
  - Comparisons depend on particular datasets and training choices; more diverse tasks would further validate generality.

Representative quotes (verbatim figures/tables/sections)
> “YaRN reaches state-of-the-art performances in context window extensions after fine-tuning on less than ~0.1% of the original pre-training data… Dynamic-YaRN allows for more than 2x context window extension without any fine-tuning.” (Abstract; Sections 3.3–3.4)

> “YaRN (s = 32) models… show continued declining perplexity through 128k, despite the fine-tuning data being limited to 64k tokens in length.” (Section 4.3.1; Table 2)

> “Minimal performance degradation between the YaRN models and their respective Llama 2 baselines.” (Section 4.3.3; Table 3)

## 6. Limitations and Trade-offs
- Scope limitation: Requires models that use `RoPE`. It does not directly apply to architectures with other positional schemes (e.g., pure ALiBi, learned absolute embeddings) without adaptation (Section 1; 2.1).
- Heuristic choices:
  - Thresholds `α, β` for `NTK-by-parts` (e.g., α=1, β=32 for LLaMA) are empirical; sensitivity and optimality are not fully explored (Section 3.2).
  - The temperature rule `1/t = 0.1 ln(s) + 1` is fit empirically on LLaMA variants; its universality is suggested but not theoretically derived (Section 3.4; Appendix A.2).
- Metrics vs capabilities:
  - Perplexity improvements and passkey retrieval success do not guarantee improvements in complex long‑range reasoning or instruction following; broader evaluation suites would strengthen claims (Appendix B.2 discussion).
- Extrapolation limits:
  - Although `Dynamic Scaling` and `YaRN` extend well beyond trained lengths, behavior at extremely long contexts (>128k–355k) is not exhaustively evaluated here for Llama‑2; Code Llama reports larger scales but with attention modifications and different setups (Section 4.2; Table 2 discussion).
- Implementation detail with KV‑cache:
  - Must cache pre‑RoPE tensors; otherwise dynamic adjustments to `s` are incorrect (Section 3.3). This is manageable but is a footgun in custom inference stacks.
- Fine‑tuning is still needed for best results:
  - Zero‑finetune `Dynamic-YaRN` helps but does not match fine‑tuned YaRN at very long lengths; some training remains necessary to achieve the headline results (Appendix B.3 vs Tables 1–2).

## 7. Implications and Future Directions
- Practical impact:
  - Makes long‑context LLMs (64k–128k) accessible with minor code changes and minimal training cost, while remaining compatible with FlashAttention‑2 and standard inference stacks (Section 3.4; 4.1).
  - Immediate applications: long‑document QA/summarization (GovReport), codebase‑level reasoning, legal/financial analysis, multi‑file retrieval, and multi‑session conversational memory.
- Conceptual shift:
  - Demonstrates that positional extension should be frequency‑aware and that light‑touch attention scaling can robustly stabilize long‑range behavior.
- Research directions:
  - Theory: formalize why the attention temperature rule scales approximately with `ln(s)`; analyze the interplay of frequency targeting and softmax calibration.
  - Generality: validate `α, β` and `t(s)` across more architectures (e.g., larger Llama‑2 70B, diverse pretrain corpora) and tasks (long‑horizon reasoning, program synthesis).
  - Tooling: automated selection of thresholds and ramp shapes; adaptive or learned interpolation during fine‑tuning.
  - Integration: combine YaRN with memory modules, retrieval augmentation, or hierarchical attention to stretch beyond 128k with stronger reasoning guarantees.
  - Evaluation: develop richer long‑context benchmarks beyond perplexity/passkey—e.g., multi‑hop narrative reasoning across 100k+ tokens, long‑range contradiction detection, cross‑document synthesis.

In short, YaRN offers a principled yet simple recipe—targeted frequency interpolation plus attention‑logit scaling—that turns standard RoPE‑based models into reliable long‑context models with very low additional compute, while retaining mainstream performance on common benchmarks (Sections 3–4; Tables 2–3; Figure 1).
