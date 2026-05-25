# YaRN: Efficient Context Window Extension of Large Language Models

**ArXiv:** [2309.00071](https://arxiv.org/abs/2309.00071)

## 🎯 Pitch

YaRN introduces a breakthrough method for vastly extending the usable context window of transformer models using Rotary Position Embeddings, such as LLaMA and Mistral. By combining a targeted rescaling of positional frequencies with a simple attention scaling trick, YaRN enables models to handle ultra-long sequences (up to 128k tokens) with 10× less training data and 2.5× fewer steps than previous state-of-the-art, all while retaining strong short-context performance. This innovation unlocks practical scalability for large language models in real-world scenarios that demand processing long documents or dialogues, without the prohibitive costs of model retraining.

---

## 1. Executive Summary
YaRN is a compute-efficient method for extending the usable context window of transformer language models that use Rotary Position Embeddings (`RoPE`). It combines a targeted way to rescale positional frequencies with a lightweight attention scaling trick, enabling LLaMA/Llama 2 and Mistral models to handle up to 128k tokens while using roughly 10× fewer tokens and 2.5× fewer training steps than prior approaches (Abstract; Section 4.1).

The method performs strongly without sacrificing short-context capabilities and remains compatible with fast attention implementations, making it practical for real-world deployment (Sections 3.4, 4.3; Table 3).

## 2. Context and Motivation
- Problem addressed
  - Transformer LLMs struggle to use inputs longer than the context window seen during pretraining. Many positional encoding schemes fail to generalize to much longer sequences (Introduction; Section 2.1–2.4; [22]).
  - Models using `RoPE` are particularly attractive due to relative-position awareness, but still fail when naively extrapolated beyond their training length (Section 2.1).

- Why this matters
  - Long inputs are critical for tasks like understanding long documents, codebases, or multi-turn dialogues. Extending context via light fine-tuning is far cheaper than retraining large models at longer lengths (Introduction; Section 4).

- Prior approaches and their shortcomings
  - Position Interpolation (`PI`): rescales token positions to fit a longer sequence into the original range (Eq. 10). Works with fine-tuning but can lose high-frequency information and slightly hurt short-context performance (Section 3.1).
  - “NTK-aware” interpolation: changes the `RoPE` frequency base (`b → b'`) to preserve more high-frequency detail (Eqs. 14–16). Better zero-shot extension but introduces slight extrapolation in some dimensions, complicating scale calibration and making fine-tuning less effective (Section 3.1).
  - “NTK-by-parts”: selectively interpolates some frequency bands and leaves others untouched to preserve local token order (Eqs. 17–20). Improves both zero-shot and fine-tuned extension (Section 3.2).
  - Dynamic Scaling (“Dynamic NTK”): adapt the scaling factor to the current sequence length at inference for graceful degradation and no-finetune gains (Section 3.3; Figure 5).
  - Other lines (e.g., ReRoPE, LM-Infinite) modify attention itself and are not compatible with FlashAttention-2 or require extra passes (Section 2.4).

- How this work positions itself
  - YaRN unifies and improves these ideas. It adds a simple, universal attention pre-softmax scaling (a temperature) on top of targeted `RoPE` interpolation (“NTK-by-parts”), and optionally combines with dynamic scaling at inference (Sections 3.2–3.4; Definition 3). It emphasizes compatibility with standard attention kernels and minimal training cost.

## 3. Technical Approach
This section explains how `RoPE` works, what goes wrong when you extend it, and how YaRN fixes it.

- Background: how RoPE encodes positions
  - `RoPE` maps even-dimensional real vectors into complex pairs and applies a rotation whose angle grows with position `m` and a per-dimension frequency `θ_d` (Eqs. 3–5).
  - The attention score becomes a function of relative distance `m − n` because the phase difference between query/key rotations depends only on that offset (Eqs. 6–9).
  - Each dimension `d` has a frequency `θ_d = b^(−2d/|D|)` and thus a wavelength `λ_d = 2π/θ_d = 2π b^(2d/|D|)` (Eq. 13). Large `d` → higher frequency → shorter wavelength.

- Why naïve stretching fails
  - PI scales positions by `1/s` so a longer sequence `L'` fits the original range `L` (Eq. 10; summarized in Eq. 12 with `g(m) = m/s, h(θ) = θ`). This compresses all rotations uniformly.
  - Two side effects (Sections 3.1–3.2):
    - Loss of high-frequency detail: small relative shifts become too tiny to detect (“NTK” intuition; Section 3.1).
    - Loss of local distances: compressing all frequencies makes nearby tokens look more similar, confusing local order (Section 3.2).

- Step 1: “NTK-aware” interpolation (target high freq retention)
  - Instead of scaling positions, change the `RoPE` base `b` to `b'` so higher-frequency dimensions are compressed less (Eqs. 14–16). This spreads “interpolation pressure” across dimensions via a frequency rebalance.
  - Benefits: Better zero-shot length extension than PI. Drawback: some dimensions slightly extrapolate “out of bounds,” hurting fine-tuning fit and making the intended scale `s` misaligned with actual usable length (Section 3.1).

- Step 2: “NTK-by-parts” (targeted, frequency-aware scaling)
  - Key idea: don’t treat all dimensions equally. Decide how much to scale per dimension based on the ratio `r(d) = L / λ_d` (Eq. 17), i.e., how many cycles a dimension completes within the original context.
    - If `λ_d >> L` (low frequency; `r` small): safe to interpolate (scale).
    - If `λ_d << L` (high frequency; `r` large): don’t scale; preserve local ordering fidelity.
  - Implemented via a ramp `γ(r)` that transitions from full interpolation (`γ=0`) to no interpolation (`γ=1`) with thresholds `α` and `β` (Eq. 18). Recommended values for Llama-family: `α=1, β=32` (Section 3.2).
  - The per-dimension frequency is blended as:
    - `h(θ_d) = [(1 − γ(r(d))) · θ_d/s] + [γ(r(d)) · θ_d]` (Eqs. 19–20).
  - This avoids extrapolation and preserves high-frequency (local) structure while stretching low-frequency bands (Section 3.2).

- Step 3: Dynamic Scaling at inference
  - During generation, adapt the scale `s = max(1, l'/L)` to the current sequence length `l'` (Section 3.3).
  - Advantages: avoids performance cliffs near the trained limit, enables partial extension even without any long-context fine-tuning, and integrates with “NTK-by-parts” or YaRN (Section 3.3; Figure 5). Implementation detail: cache K/V before applying `RoPE` if using KV-caching (Section 3.3).

- Step 4 (YaRN’s new piece): attention pre-softmax scaling with temperature
  - Modify attention scores by dividing by a temperature `t` (smaller `t` → sharper attention): `softmax((qᵀk)/(t√|D|))` (Eq. 21). This is equivalent to scaling `q` and `k` by `√(1/t)` when implemented via `RoPE`’s complex rotation reparameterization—no code changes or runtime cost (Section 3.4).
  - The best `t` grows gently with `s`: 
    - Quote: “`1/t = 0.1 ln(s) + 1`” (Eq. 22). 
    - Appendix A.2 (Figures 2–4) shows that an appropriate `t` consistently lowers perplexity across samples and token positions for a given `s`.

- Putting it together: YaRN
  - Definition 3: YaRN = “NTK-by-parts” targeted `RoPE` interpolation (Section 3.2) + attention temperature scaling (Eq. 21; Eq. 22), with optional dynamic scaling at inference (Sections 3.3–3.4).
  - Design choices:
    - Targeted rather than blind interpolation to preserve local order and high-frequency detail (Section 3.2).
    - Lightweight attention scaling that acts uniformly and efficiently across positions and samples (Section 3.4; Appendix A.2).
    - Compatibility with FlashAttention-2 and standard attention kernels because the core attention math isn’t restructured (Section 3.4).

- Training recipe (for Llama 2; Section 4.1)
  - Fine-tune Llama 2 7B/13B with `s=16` (to 64k) for 400 steps, global batch 64, AdamW (β1=0.9, β2=0.95), lr=2e−5, 20-step warmup, on PG19 segments of 64k tokens (Section 4.1).
  - For `s=32` (to 128k), continue from the `s=16` checkpoint for 200 steps (Section 4.1).
  - Tooling: PyTorch FSDP + FlashAttention-2 (Section 4.1).

## 4. Key Insights and Innovations
- Targeted, frequency-aware interpolation (“NTK-by-parts”)
  - Novelty: Uses the ratio `r(d)=L/λ_d` to determine per-dimension scaling, preventing over-compression of high-frequency bands (Eqs. 17–20; Section 3.2).
  - Significance: Preserves local-order sensitivity while still extending global context—addresses a core weakness of “blind” methods like PI.

- Lightweight attention temperature scaling integrated into `RoPE`
  - Novelty: A simple pre-softmax scaling (Eq. 21) that can be realized as a constant scale on `RoPE` rotations—no attention kernel changes (Section 3.4).
  - Significance: Uniformly improves perplexity across positions and samples for a given `s` (Appendix A.2; Figures 2–4), with zero runtime overhead.

- Dynamic scaling at inference for graceful extension without fine-tuning
  - Novelty: Adjust `s` online during generation (Section 3.3).
  - Significance: Prevents abrupt performance cliffs beyond pretraining length and helps zero-finetune long-context use (Figure 5).

- Compute efficiency with maintained capability
  - Novelty: Achieves 64k–128k context with only 400 + 200 steps, ≈0.1% of the original pretraining data, “10× less tokens and 2.5× less training steps than previous methods” (Abstract; Section 4).
  - Significance: Makes long-context models accessible under modest compute budgets while preserving standard-benchmark performance (Table 3).

These are more than incremental tweaks: together they define a new, practical regime for long-context extension that respects the structure of `RoPE`, maintains compatibility with optimized attention, and reduces training cost.

## 5. Experimental Analysis
- Evaluation methodology
  - Long-sequence language modeling: sliding-window perplexity with stride `S=256` on long samples from Proof-pile and GovReport (Section 4.3.1; Figure 1; Tables 1–2, 4).
  - Passkey retrieval: a synthetic test for locating a 5-digit number anywhere in long text, measuring retrieval accuracy across positions and lengths (Section 4.3.2; Table 5).
  - Standard benchmarks: ARC-Challenge (25-shot), HellaSwag (10-shot), MMLU (5-shot), TruthfulQA (0-shot) (Section 4.3.3; Table 3).
  - Models: Llama 2 7B and 13B, with baselines including Together 32k (PI) and Code Llama (NTK-aware), plus additional Mistral 7B experiments (Appendix B.4; Figure 6; Table 6).

- Main quantitative results (selected highlights)
  - 8k extension comparison (PI vs NTK vs YaRN; Table 1):
    - At 10,240 tokens, perplexity: PI 8.07; NTK 6.24; YaRN 6.04. 
    - At 8,192 tokens, perplexity: PI 3.34; NTK 3.59; YaRN 3.35.
    - Quote: “YaRN (s=2) matches PI at ≤8k and is better beyond 8k” (Table 1).
  - Long-range (Proof-pile) up to 128k (Table 2; Figure 1):
    - Llama 2 7B at 131,072:
      - Code Llama (NTK-aware): 2.71
      - YaRN s=32: 2.37 (best)
      - Together 32k (PI): diverges (>10^4 by 131k)
    - Llama 2 13B at 131,072:
      - Code Llama: 2.54
      - YaRN s=32: 2.24 (best)
    - Quote: “YaRN (s=32) shows continued declining perplexity through 128k” despite training data being 64k (Section 4.3.1; Table 2).
  - GovReport (32k setting; Table 4):
    - 7B: YaRN s=16: 3.59; YaRN s=32: 3.64; Together 32k: 3.67; Code Llama NTK: 4.44.
    - 13B: YaRN s=16: 3.35; s=32: 3.39; Code Llama NTK: 4.22.
  - Passkey retrieval (Table 5):
    - 7B YaRN s=32 (128k): 99.4% accuracy; 13B s=32 (128k): 99.4%.
    - Code Llama 7B: 94.3% up to 112k.
    - Quote: “Both 7B and 13B YaRN at 128k pass the task with >99% accuracy across the entire window” (Section 4.3.2; Table 5).
  - Standard LLM benchmarks (Table 3):
    - 7B baseline vs YaRN s=32:
      - ARC: 53.1 → 52.1; HellaSwag: 77.8 → 78.4; MMLU: 43.8 → 41.7; TruthfulQA: 39.0 → 37.3.
    - 13B baseline vs YaRN s=32:
      - ARC: 59.4 → 58.0; HellaSwag: 82.1 → 82.2; MMLU: 55.8 → 51.9; TruthfulQA: 37.4 → 37.3.
    - Contrast: Code Llama (NTK-aware) shows substantial degradation (e.g., 7B HellaSwag 60.8; MMLU 31.1).
    - Quote: “Minimal performance degradation vs baselines; average 0.49% drop from s=16 to s=32” (Section 4.3.3; Table 3).
  - No-finetune dynamic scaling (Appendix B.3; Figure 5):
    - Quote: “Dynamic scaling effectively prevents perplexity blow-up beyond pretraining length; Dynamic-YaRN outperforms Dynamic-PI.”

  - Mistral 7B (Appendix B.4; Figure 6; Table 6):
    - At 131,072: YaRN s=16: 2.19; `MistralLite` (NTK-aware) diverges (>10^3). Shows portability beyond Llama.

- Do the experiments support the claims?
  - Yes, on three fronts:
    - Effective long-context utilization up to 128k with smooth perplexity profiles (Table 2; Figure 1).
    - Preservation of general capabilities on standard benchmarks with only minor drops (Table 3).
    - Compute efficiency—short fine-tunes and transfer from s=16 to s=32 (Section 4.1–4.2).
  - Robustness checks:
    - Passkey retrieval across positions and lengths shows attention coverage, not just low perplexity (Section 4.3.2; Table 5).
    - Ablations by method class appear implicitly via method comparisons (PI vs NTK vs YaRN). Appendix A.2 varies `t` and shows consistent optimum ranges across positions/samples (Figures 2–4).

- Mixed or conditional results
  - YaRN s=16 on 7B shows degradation beyond ~64k (perplexity > 10^1 at 98k and 131k in Table 2), indicating that scaling larger than the fine-tuned window benefits from the s=32 transfer step.
  - MMLU drops more than other benchmarks when going to s=32 (Table 3), suggesting modest trade-offs on certain reasoning tasks.

## 6. Limitations and Trade-offs
- Scope limited to `RoPE`-based models
  - Methods operate by modifying `RoPE` frequencies/rotations; not directly applicable to other positional schemes (e.g., T5 bias, ALiBi) without adaptation (Section 2.1–2.4).

- Heuristic components and hyperparameters
  - The ramp thresholds `α, β` and the temperature formula `1/t = 0.1 ln(s) + 1` are empirically chosen (Section 3.2; Eq. 22; Appendix A.2). They generalize across Llama and work well on Mistral (Appendix B.4), but may require tuning for other architectures/domains.

- Memory and compute at inference still grow with context
  - YaRN does not change attention complexity; running 64k–128k contexts still demands substantial memory/time, though the method itself adds no overhead (Section 3.4). Practical deployment may require paging, chunking, or specialized kernels.

- Finetuning data and training choices
  - Results rely on PG19 long-text fine-tuning at 64k segments (Section 4.1). Other domains (e.g., code, math) could require domain-specific long-context data to realize full gains.

- Analysis granularity
  - While the paper compares methods extensively, explicit ablations isolating each component of YaRN (ramp-only vs temperature-only vs both) are limited; most evidence comes from method-level comparisons and temperature sweeps (Table 1; Appendix A.2).

- KV-caching implementation caveat
  - For Dynamic Scaling, incorrect caching (after applying `RoPE`) can produce inconsistent results; keys/values should be cached pre-`RoPE` (Section 3.3).

## 7. Implications and Future Directions
- Field impact
  - YaRN provides a practical path to 64k–128k context for existing `RoPE` models with minimal retraining and preserved capabilities, enabling “train short, test long” regimes (Conclusion; Sections 4.2–4.3). Its compatibility with FlashAttention-2 makes it deployable at scale (Sections 2.4, 3.4).

- Practical applications
  - Long-document QA and summarization (GovReport results; Table 4).
  - Code understanding across large repositories.
  - Multi-document grounding, retrieval-augmented generation with fewer truncation compromises.
  - Synthetic long-context skills (strong passkey retrieval across the entire window; Table 5).

- Research directions
  - Theoretical grounding of the temperature rule (Eq. 22): derive from attention dynamics or information theory; explore layer-wise or head-wise temperatures.
  - Automated selection of `α, β` and ramp shapes; adapt per layer/head using data-driven criteria.
  - Extensions to non-`RoPE` positional encodings and integration with memory-augmented or sparse-attention models.
  - Training curricula that combine YaRN with retrieval or compression to push beyond 128k without quadratic costs.
  - Task-level evaluations beyond perplexity and passkey retrieval (e.g., long-form reasoning, multi-hop over book-length inputs).

> “YaRN reaches state-of-the-art performances in context window extensions after fine-tuning on less than ∼0.1% of the original pre-training data… and, combined with Dynamic Scaling, allows for more than 2× context window extension without any fine-tuning.” (Abstract; Sections 3.3–3.4, 4)

Overall, YaRN is a well-motivated, simple-to-implement, and empirically validated strategy for long-context extension in `RoPE`-based LLMs. It balances frequency-aware interpolation with a universal attention temperature and demonstrates strong results up to 128k with modest compute.
