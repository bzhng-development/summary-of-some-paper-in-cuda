# The End of Manual Decoding: Towards Truly End-to-End Language Models

**ArXiv:** [2510.26697](https://arxiv.org/abs/2510.26697)
**Authors:** Zhichao Wang, Dongyang Ma, Xinting Huang, Deng Cai, Tian Lan, Jiahao Xu, Haitao Mi, Xiaoying Tang, Yan Wang
**Institutions:** 

## 1. Executive Summary (2–3 sentences)
This paper introduces `AutoDeco`, a lightweight extension to transformer language models that predicts token-by-token decoding parameters—temperature and top-p—so the model can modulate its own randomness as it generates. By replacing static, hand-tuned decoding with dynamic per-token control trained end-to-end through a differentiable “soft” top-p, it delivers consistent accuracy gains across eight benchmarks and multiple model families with negligible latency (≈1–2%; Table 3), and it enables a new capability: controlling decoding behavior via natural language prompts (Figure 5, Table 4).

Reasoning: The core problem is that today’s “end-to-end” LLMs still rely on manual, nondifferentiable decoding knobs; the paper’s contribution is to learn these knobs from hidden states per token, trained by backprop via a differentiable approximation of nucleus sampling. The significance is proven by broad empirical gains and by an emergent, natural-language steerability of decoding.

## 2. Context and Motivation
- Problem addressed:
  - Modern LLMs are not fully “end-to-end” because decoding uses nondifferentiable, static hyperparameters such as `temperature` and `top-p` (also called nucleus sampling threshold). These must be tuned manually, are task-dependent, and cannot adapt within a single generated sequence (Section 1).
  - Static settings are inherently suboptimal: early generation may benefit from exploration (higher randomness), while final answers benefit from precision (lower randomness), but current LLMs cannot vary these dynamically within native inference (Section 1).

- Why this matters:
  - Practical impact: Manually tuning temperature/top-p is time-consuming, costly, and brittle across tasks (Introduction; Figure 1). In production, different use cases (e.g., creative writing vs. exact QA) require different settings; even within one answer, the optimal randomness changes over time.
  - Theoretical significance: Nondifferentiable decoding (especially hard top-p) blocks gradients, preventing a truly end-to-end optimization pipeline in which the model learns how to sample, not only what to predict.

- Prior approaches and shortcomings:
  - Deterministic decoding (greedy, beam search) often yields generic or dull outputs (Related Works).
  - Stochastic sampling (top-k, top-p) improves diversity but requires fixed hyperparameters, which are suboptimal and manually tuned (Introduction; Related Works).
  - Model-based decoding (e.g., contrastive decoding, speculative decoding) introduces extra models or fixed algorithms with new hyperparameters (Related Works). None natively learn per-token, context-dependent decoding.
  
- Positioning:
  - `AutoDeco` augments the transformer with two small prediction heads that output token-level `temperature` and `top-p` values, learned end-to-end by introducing a differentiable “soft” nucleus sampling during training (Section 2, Figure 2, Equations (1)–(3)). This makes decoding a parametric, learnable part of the forward pass (Figure 1), rather than an external, hand-tuned procedure.

Reasoning: The motivation builds logically from the mismatch between static heuristics and the dynamic needs of generation. The novelty arises from making decoding parameters themselves learnable, differentiable, and context-aware, overcoming the gradient-flow barrier via a soft top-p relaxation.

## 3. Technical Approach
- Overview:
  - Architecture: Add two lightweight 2-layer MLP heads to a standard transformer: a `temperature head` and a `top-p head`. At each generation step t, given the final hidden state `h_t`, the model predicts `T̂_t` and `P̂_t`, which are then used to modify the next-token probability distribution (Figure 1; Equation (4)).
  - Training: Make top-p differentiable with a smooth “soft mask” so gradients can flow back from the cross-entropy loss to both heads (Section 2.1; Figure 2; Equations (1)–(3)).
  - Inference: Use the predicted `T̂_t` and `P̂_t` to internally rescale and filter logits, with negligible latency overhead (Section 2.2; Table 3).

- Key components explained step-by-step:
  - What is `temperature`?
    - A scalar that rescales logits before the softmax. Higher temperature flattens the distribution (more randomness), lower temperature sharpens it (more determinism).
  - What is `top-p` (nucleus sampling)?
    - A truncation method that keeps the smallest set of tokens whose cumulative probability exceeds p, then samples from that set (more adaptive than top-k).
  - Differentiable soft top-p for training:
    1) Compute temperature-scaled probabilities:
       - p = softmax(l / T̂). This lets the model decide how peaky vs. flat the distribution should be (Equation (1)).
    2) Sort p, compute cumulative sums c, and create a smooth mask m_sorted:
       - m_sorted = exp(-α · ReLU(c - P̂)), with α controlling how steeply the mask decays beyond the nucleus (Equation (2); Figure 2a).
         - For tokens inside the nucleus (c < P̂), mask = 1.
         - For tokens outside, mask decays smoothly towards 0 as they lie further beyond P̂.
    3) Apply the mask and renormalize:
       - p̃ = (p ⊙ m) / (sum(p ⊙ m) + ε) (Equation (3)).
       - This yields a fully differentiable final probability distribution (Figure 2b shows how the distribution changes under the soft mask), enabling end-to-end training via standard cross-entropy with the ground-truth next token y*.
  - Heads and micro-dependency:
    - Predict `T̂_t = temp_head(h_t)`; then predict `P̂_t = top-p_head(h_t, T̂_t)` (Equation (4)). The `top-p head` conditions on the newly predicted temperature to coordinate both controls (Figure 1, dashed arrow).
  - Training protocol:
    - Freeze the base LLM; train only the two heads (Section 2.1 “Training”).
    - Bias mitigation:
      - Easy-token masking: randomly mask loss for a large fraction (e.g., 60%) of positions where greedy prediction already matches the ground-truth, to avoid biasing towards near-zero temperatures (Section 2.1).
      - Dynamic Fine-Tuning (Wu et al., 2025): reweight training loss to focus on tokens where the model is reasonably calibrated, preventing the temperature head from overreacting on highly uncertain/outlier positions (Section 2.1).
  - Inference pipeline:
    - For each step t: compute hidden state `h_t`, predict `T̂_t` and `P̂_t`, rescale logits by `T̂_t`, apply nucleus filtering guided by `P̂_t`, and sample the next token. All of this is embedded in the forward pass, adding only ≈1–2% latency (Section 2.2; Table 3).

- Why this design?
  - Differentiability: The soft top-p mask is the key that restores gradient flow through a sampling-like truncation (Figure 2).
  - Complementary controls: Temperature and top-p influence different aspects of the sampling distribution; learning both per token provides fine-grained control (Figure 4 ablation shows both contribute).
  - Efficiency and simplicity: Two tiny MLPs incur negligible computational cost relative to the base model (Table 3), and deployment is “drop-in” (Section 2.2).

Reasoning: The central mechanism is to turn nondifferentiable, global knobs into differentiable, token-wise predictions that can be learned from the task loss. The soft mask retains the spirit of top-p while allowing backpropagation; the micro-dependency between heads stabilizes and coordinates the two controls.

## 4. Key Insights and Innovations
- Dynamic, token-level decoding control:
  - Novelty: Instead of setting a single static temperature/top-p for an entire output (or entire task), `AutoDeco` predicts `T̂_t` and `P̂_t` at every step from the hidden state (Figure 1; Equation (4)).
  - Why it matters: The “right” amount of randomness depends on context and position in the answer (explore early, exploit late). This unlocks adaptivity current decoders lack.

- Differentiable “soft” top-p for end-to-end training:
  - Novelty: A smooth mask m_sorted = exp(-α · ReLU(c - P̂)) lets the system learn top-p behavior by backpropagating through a nucleus-like operation (Section 2.1; Figure 2; Equation (2)).
  - Significance: Overcomes the fundamental gradient-flow barrier that has kept decoding outside end-to-end training loops.

- Natural-language control of decoding:
  - Novelty: The system exhibits an emergent ability to interpret high-level instructions (“be more diverse/creative” or “be more certain”) and adjust `T̂`/`P̂` accordingly (Figure 5). With targeted fine-tuning using a ranking loss, this behavior becomes reliable (Table 4).
  - Significance: Moves beyond controlling content to controlling the generative process itself via plain language—a new interaction modality.

- Practical performance with negligible overhead:
  - Novelty: Two small heads embedded in the forward pass add ≈1–2% latency and ≈4 MB memory in one evaluated setup (Table 3), yet improve pass@1 across math and general tasks (Tables 1–2).
  - Significance: This makes the method a practical “one-line change” deployment for real systems (Section 2.2).

Reasoning: These elements go beyond incremental tuning: they alter the locus of control (from user-specified knobs to model-predicted parameters), enable learning where it wasn’t possible, introduce a new interface (language-level control over decoding), and do so with production-friendly cost.

## 5. Experimental Analysis
- Setup:
  - Models (Section 3.1): `Llama-3.1-Nemotron-Nano-8B-v1`, `R1-Distill-Qwen-7B`, `Qwen3-30B-A3B-Instruct-2507` (MoE), and `OpenAI-GPT-OSS-20B` (MoE).
  - Training data: Reject-sampling trajectories generated from the DeepMath-103K dataset (Section 3.1).
    - “Reject-sampling trajectories” here are sampled solution attempts, retaining information beyond just the correct final sequence, offering diverse contexts for learning decoding behavior.
  - Evaluation (Section 3.1):
    - In-domain math: AIME (24+25), BRUMO25, HMMT25, BeyondAIME (recent, harder math benchmarks).
    - Out-of-domain: GPQA-Diamond, MMLU-Pro (QA), LiveCodeBenchV6 (code), IFEval (instruction following).
  - Baselines:
    - Greedy Search, and Default Sampling with `T=1.0`, `P=1.0` (Section 3.1).
    - Expert-Guided Tuning (oracle): tune temperature and top-p on the test set via search (interval 0.1), first temperature with `top-p=1.0`, then top-p at the chosen temperature (Figure 3).
  - Metric: `Pass@1` accuracy via oversampling 128 samples per problem (8 seeds × 16 samples; Section 3.1). `Pass@k` (k=16,32,64) in Supplementary Section 8 (Tables 5–7).

- Main quantitative results (Pass@1):
  - In-domain math (Table 1): `AutoDeco` consistently outperforms Greedy and Default Sampling across all four models, e.g.:
    - Llama-Nemotron-8B average: 
      - Greedy 42.50; Default 42.59; AutoDeco 46.05 (+3.5 over Default).
    - R1-Distill-Qwen-7B average: 
      - Greedy 30.58; Default 34.76; AutoDeco 37.37 (+2.61 over Default).
    - Qwen3-30B-A3B average:
      - Greedy 52.25; Default 56.05; AutoDeco 56.54 (+0.49 over Default).
    - OpenAI-GPT-OSS-20B average:
      - Greedy 51.50; Default 56.64; AutoDeco 58.13 (+1.49 over Default).
  - Out-of-domain general tasks (Table 2): `AutoDeco` generalizes beyond math despite math-only training:
    - Llama-Nemotron-8B average:
      - Greedy 48.43; Default 46.35; AutoDeco 49.72 (+3.37 over Default).
    - R1-Distill-Qwen-7B average:
      - Greedy 39.32; Default 42.47; AutoDeco 46.88 (+4.41 over Default).
    - Qwen3-30B-A3B average:
      - Greedy 68.84; Default 69.03; AutoDeco 70.24 (+1.21 over Default).
    - OpenAI-GPT-OSS-20B average:
      - Greedy 56.56; Default 58.63; AutoDeco 59.42 (+0.79 over Default).

  - Notable pattern: On some general tasks, Default Sampling underperforms Greedy (e.g., Llama on GPQA-Diamond, IFEval; Table 2), yet AutoDeco adapts towards more deterministic settings and surpasses both—evidence of effective dynamic control.

- Comparison to oracle-like Expert-Guided Tuning (Figure 3):
  - The per-dataset, test-set-tuned static hyperparameters are an upper bound on what any static policy could achieve. AutoDeco’s single-pass performance is “nearly identical,” with gaps consistently <1 point across models/datasets.
  - This underscores that dynamic per-token control can match a static policy that is unfairly advantaged by test-set access. In real use, test-set tuning is unavailable, giving AutoDeco a practical edge.

- Pass@k (Supplementary Tables 5–7):
  - Gains persist or grow relatively at higher k. Example: For OpenAI-GPT-OSS-20B, the average pass@64 rises from 89.68 (Default) to 91.55 (AutoDeco), which corresponds to a relative error reduction from 10.32% to 8.45%—an 18.1% drop in error (Table 7). This indicates the improvements are not limited to single-sample success.

- Ablations (Figure 4):
  - Using only the `temperature head` or only the `top-p head` yields ≈3–3.5 point gains on AIME (R1-Distill-Qwen-7B), and using both is best. This shows each control contributes meaningfully and that their combination is synergistic.

- Efficiency (Table 3; Section 3.2.2):
  - FLOPs essentially unchanged relative to Default Sampling (e.g., 2.89e+13 vs. 2.89e+13 for 1k prompts).
  - Memory overhead ≈4 MB across prompt lengths.
  - Latency overhead per 1k generated tokens ranges ≈0.29–0.61 s depending on prompt length, averaging ≈1.7% relative increase. For instance, at 24k prompt length: 25.76 s (Default) vs. 26.05 s (AutoDeco), a +0.29 s increase. At 1k prompt length: 18.23 s vs. 18.84 s (+0.61 s).

- Natural-language control (Figure 5; Table 4):
  - Without targeted training, adding “high diversity” or “low diversity” commands visibly raises/lowers predicted `T̂`/`P̂` curves for the same prompt (Figure 5), but not consistently.
  - After a small targeted fine-tuning with a ranking loss, consistent directional control emerges:
    - Baseline Avg. `T̂` = 0.72; “Low Diversity” → 0.61 (↓0.11) with 99% consistency, “High Diversity” → 0.82 (↑0.10) with 96% consistency.
    - Baseline Avg. `P̂` = 0.79; “Low Diversity” → 0.73 (↓0.06; 97% consistency), “High Diversity” → 0.83 (↑0.04; 85% consistency). (Table 4)

- Do the experiments support the claims?
  - Performance: Yes—consistent pass@1 gains across models and tasks (Tables 1–2), matching oracle static tuning (Figure 3), with robust pass@k behavior (Tables 5–7).
  - Efficiency: Yes—measured overheads are small and stable (Table 3).
  - Steerability: Qualitative emergence (Figure 5) is made quantitatively reliable after targeted training (Table 4), though absolute precision (e.g., “temperature ≈ 0”) is not yet achieved (Section 3.3).

Reasoning: The evaluation spans multiple domains, model sizes, and families, uses strong baselines, and includes oracle comparisons and ablations. The result pattern is coherent: dynamic per-token control beats any single static setting; the cost is minimal; and the method enables a new form of natural-language steering.

## 6. Limitations and Trade-offs
- Training-time approximation vs. inference-time behavior:
  - The soft top-p mask is a training-time relaxation (Figure 2; Equations (2)–(3)). If inference uses standard hard top-p truncation, there’s a potential train–test mismatch. The paper does not explicitly detail whether inference applies soft or hard truncation internally (Section 2.2 says “rescale and filter” but not the exact filter form), which could matter for rare-tail behavior.

- Frozen backbone:
  - Only the heads are trained; the base LLM is frozen (Section 2.1). This simplifies training but may limit how precisely decoding control can align with content generation. The authors hypothesize joint training might yield finer control (Section 5 “Future Work”).

- Precision of natural-language control:
  - While directional control becomes consistent after targeted training (Table 4), absolute calibration is limited. For example, “no randomness” doesn’t yield near-zero temperature; it yields a modest decrease (Section 3.3).

- Scope of controls:
  - The method currently predicts only `temperature` and `top-p`. Other decoding knobs (e.g., repetition penalty, presence penalty, top-k, length penalty) are not explored; interactions among more parameters could be complex.

- Data domain and generalization:
  - Heads are trained on math-derived trajectories (Section 3.1) yet show strong out-of-domain gains (Table 2). This is encouraging but also raises questions: Would training on other domains produce even better general-domain decoding? Are there edge cases (e.g., safety-sensitive generation) where token-level increases in randomness are undesirable without additional constraints?

- Evaluation caveats:
  - The Expert-Guided Tuning baseline is an oracle (test-set tuned). While useful to bound static methods, it is not a deployable baseline. Real-world comparisons would benefit from practical developer workflows (e.g., tune-on-validation-then-evaluate-on-blind-test).

- Model-specific variability:
  - Gains are smaller on `Qwen3-30B-A3B-Instruct-2507` (Table 1–2). The paper attributes this to shorter answers (Section 3.2.1), which may reduce sensitivity to decoding parameters. This suggests benefits may vary with model style (e.g., “thinking” vs. “non-thinking”) and output length.

- Safety and robustness:
  - Increasing randomness can raise the risk of off-policy or unsafe outputs in some contexts. There is no explicit safety layer or constraint in the method as presented.

Reasoning: These limitations arise logically from what’s implemented (which parameters, which training scheme), the approximations used, and the reported empirical behavior. They highlight where further engineering or analysis could strengthen the approach.

## 7. Implications and Future Directions
- Paradigm shift in decoding:
  - Decoding ceases to be a fixed, external heuristic and becomes a learned, token-resolved component of the model’s computation graph. This reframes “end-to-end” generation to include how the model samples, not just what it predicts (Figure 1; Section 2).

- Practical benefits:
  - For developers, this promises to eliminate repeated manual sweeps of temperature/top-p per task, while delivering near-oracle static performance across varied inputs automatically (Figure 3). Given the low overhead (Table 3), it is deployment-friendly.

- New interaction modes:
  - Natural-language control of the decoding process (Figure 5; Table 4) suggests user interfaces where people can specify “style of generation” (certainty vs. diversity) directly, and the model adjusts its randomness on the fly—even within the same response.

- Research directions:
  - Joint training of the backbone and decoding heads to improve precision and stability of control (Section 5 “Future Work”).
  - Expanding the control space to other decoding parameters (e.g., repetition penalty, top-k) and studying their interactions.
  - Learning richer policies that condition on intermediate reasoning states (e.g., higher temperature during exploration, lower during finalization) and validating this with token-level analyses.
  - Safety-aware adaptive decoding: incorporate constraints or reward models so increases in diversity do not violate safety or factuality.
  - Task-aware or instruction-aware meta-control: condition decoding heads on task descriptors or system prompts for even better cross-domain adaptation.
  - Theoretical analysis of the soft top-p relaxation: characterize its bias relative to hard top-p and study α’s effect (Figure 2 shows α=30) on gradient quality and final performance.

- Applications:
  - Reasoning systems that explore multiple solution paths early and commit late (as suggested by gains on math and QA; Tables 1–2).
  - Creative writing and brainstorming assistants where users can dial diversity up/down in natural language (Figure 5; Table 4).
  - Code generation where stability is prioritized for structure but diversity is injected at higher-level planning stages (Table 2, LiveCodeBenchV6).

Reasoning: The findings open both practical engineering pathways (drop-in performance gains, user steerability) and conceptual ones (decoding as a learned, controllable policy). Extending the idea to more controls and joint training could make decoding an integral, optimizable part of the LLM stack.

> Representative results cited:
> - “AutoDeco … consistently outperforms standard, non-expert decoding baselines” (Tables 1–2).
> - “AutoDeco’s single-pass performance is nearly identical to [oracle] expert-guided tuning” (Figure 3).
> - “Latency … adds a consistent overhead of 0.29–0.6 s/k tokens … average relative increase of just 1.7%” (Table 3; Section 3.2.2).
> - “Low Diversity: Avg. Temp. ↓0.11 with 99% consistency; Avg. top-p ↓0.06 with 97% consistency” after targeted training (Table 4).
> - Training efficiency: convergence within ~6K samples and ~400 steps (Supplementary Section 7; Figure 6).