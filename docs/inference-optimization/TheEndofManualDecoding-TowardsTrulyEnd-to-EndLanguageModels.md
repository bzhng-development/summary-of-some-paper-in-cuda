# The End of Manual Decoding: Towards Truly End-to-End Language Models

**ArXiv:** [2510.26697](https://arxiv.org/abs/2510.26697)

## 🎯 Pitch

AutoDeco is a novel, lightweight enhancement to transformer-based language models that enables them to dynamically predict their own decoding parameters—temperature and top-p—at each generation step, thus realizing truly end-to-end and context-adaptive text generation. By replacing static, hand-tuned decoding hyperparameters with differentiable, learned predictions (via a new soft top-p mechanism), AutoDeco eliminates manual decoding, consistently boosts accuracy across tasks, and introduces the unprecedented ability for models to interpret and execute natural language decoding instructions—significantly advancing both the usability and control of LLMs.

---

## 1. Executive Summary
This paper introduces AutoDeco, a lightweight addition to standard transformers that lets a language model predict and apply its own decoding hyperparameters—specifically per-token `temperature` and `top-p`—during generation. By making decoding parameters learned, context-dependent, and differentiably trainable (via a new “soft” top‑p), the approach removes manual, static hyperparameter tuning and yields consistent accuracy gains across eight benchmarks with negligible latency overhead (Sections 2–3; Tables 1–3).

## 2. Context and Motivation
- The problem: Modern LLMs still rely on a manual, non-differentiable decoding stage that uses fixed hyperparameters like `temperature` (controls randomness by scaling logits) and `top-p` (nucleus sampling that restricts sampling to the smallest set of tokens whose cumulative probability ≥ p). These are typically hand-tuned per task and kept static for an entire output sequence.
- Why this matters:
  - Practical burden: Choosing good `temperature/top-p` takes time, compute, and expertise; settings are task-dependent and even position-dependent within a single response (Section 1).
  - Suboptimal quality: One static setting cannot balance exploration vs. precision through different stages of generation (e.g., early brainstorming vs. final answer). This undermines “end-to-end” learning because critical generation behavior is controlled outside the model by fixed heuristics (Figure 1).
- Prior approaches and limitations:
  - Deterministic decoders (greedy, beam) or fixed sampling heuristics (top-k/top-p) are static and non-adaptive (Section 4).
  - Model-based steering (e.g., contrastive, speculative decoding) still encodes a fixed external algorithm or guidance model that acts like a hyperparameter choice (Section 4).
  - Importantly, standard `top-p` sampling is non-differentiable, preventing direct training of decoding parameters from task loss (Section 2.1).
- Positioning: AutoDeco makes decoding itself part of the learnable, token-level forward pass. It predicts `temperature` and `top-p` at every step from the model’s hidden state and applies them internally, enabling truly end-to-end optimization with almost no runtime cost (Sections 2.1–2.2; Figure 1).

## 3. Technical Approach
AutoDeco augments a pretrained transformer with two small prediction heads and introduces a differentiable approximation to top‑p so that these heads can be trained directly from task loss.

- Architecture (Figure 1; Section 2.2):
  - Inputs: at generation step t, the transformer produces hidden state `h_t`.
  - Heads:
    - `temp_head(h_t)` → predicts per-token temperature `T̂_t`.
    - `top-p_head(h_t, T̂_t)` → predicts per-token `P̂_t` (note the micro-dependency on temperature; Equation (4)).
  - Application: The predicted `T̂_t` rescales logits; the predicted `P̂_t` constrains sampling. Both are applied inside the same forward pass that produces the next-token distribution.

- The core training challenge: standard top‑p has a hard cutoff (non-differentiable), blocking gradients to the `top-p` head (Section 2.1).  
  Solution: A differentiable “soft top‑p” mask used only during training (Figure 2).
  - Step 1: Temperature scaling (Equation (1)). Given logits `l`, compute probabilities
    p = softmax(l / T̂).
  - Step 2: Compute cumulative probability `c` over sorted `p`, then form a smooth mask (Equation (2)):
    m(sorted) = exp(−α · ReLU(c − P̂)).
    - Intuition: Inside the nucleus (`c < P̂`), mask is 1. Beyond it, weights decay smoothly with steepness controlled by α (Figure 2a).
  - Step 3: Reweight and renormalize to obtain a differentiable final distribution (Equation (3)):
    p̃ = (p ⊙ m) / (sum(p ⊙ m) + ε).
  - Loss: Standard cross-entropy between p̃ and the ground-truth next token y* allows gradients to flow into both heads (Section 2.1; Figure 2b).

- Training protocol (Section 2.1; Section 6; Figure 6):
  - Freeze the base LLM; train only the two small heads.
  - Data: “Reject sampling trajectories” generated from DeepMath-103K prompts (a rigorous math dataset) using four base models (Section 3.1).
  - Debiasing tricks to stabilize learning:
    - Easy-token masking: randomly drop loss on a large fraction (e.g., 60%) of “easy” positions where greedy already matches the label, preventing the temperature head from collapsing toward near-zero (Section 2.1).
    - Dynamic Fine-Tuning: reweight loss to focus on tokens with reasonable prior confidence, discouraging spurious large temperatures on highly uncertain outliers (Section 2.1).
  - Efficiency: ~6k samples and ~400 steps suffice for strong performance; training curves converge fast (Section 7; Figure 6).

- Inference (Section 2.2):
  - For each step t:
    1) compute `h_t`;
    2) compute logits and predict `T̂_t`, `P̂_t` via the two MLP heads (Equation (4));
    3) immediately rescale logits with `T̂_t` and apply nucleus filtering with `P̂_t`.  
  - Overhead: negligible (1–2% latency), because the heads are tiny compared to the transformer (Section 3.2.2; Table 3). No separate passes or external controllers.

- A simple mental model: Treat the LLM as not only deciding “what token comes next” but also “how exploratory should I be right now?” It learns this control signal from the hidden state that already encodes context, uncertainty, and task hints.

## 4. Key Insights and Innovations
- Differentiable soft top‑p (Section 2.1; Figure 2; Equations (1)–(3))
  - What’s new: A smooth nucleus mask `m = exp(−α ReLU(c − P̂))` that approximates top‑p but keeps gradients alive.
  - Why it matters: It finally makes decoding hyperparameters trainable in an end-to-end way from task loss, removing dependence on hand-tuning.

- Token-level, self-predicted decoding parameters (Figure 1; Equation (4))
  - What’s new: Two tiny MLP heads predict `T̂_t` and `P̂_t` per token from `h_t` (with `P̂_t` conditioned on `T̂_t`).
  - Why it matters: Decoding is no longer static. The model dynamically adjusts exploration vs. exploitation across the sequence, matching local context needs.

- Near-oracle performance without test-time tuning (Figure 3)
  - What’s new: Single-pass AutoDeco matches an “expert-guided” oracle baseline that tunes on the test set (an upper bound for any static method).
  - Why it matters: In practice, you cannot tune on unseen test inputs. AutoDeco delivers near-oracle choices automatically and per-token.

- Natural-language steerability of decoding (Section 3.3; Figure 5; Table 4)
  - What’s new: The model begins to interpret instructions like “be more diverse/certain” by shifting predicted `T̂/P̂` on a token-by-token basis; targeted fine-tuning makes this behavior consistent (e.g., −0.11 temperature with 99% consistency for “low diversity”).
  - Why it matters: This is a step toward interactive, instruction-driven control of generation style without external knobs.

- Minimal cost, drop-in design (Section 3.2.2; Table 3; Section 7)
  - What’s new: Two small heads, frozen backbone, ~400-step fine-tuning, 1–2% latency overhead, ~+4 MB memory.
  - Why it matters: Practical to adopt broadly; works across different model families.

## 5. Experimental Analysis
- Setup (Section 3.1; Section 6)
  - Models: `Llama-3.1-Nemotron-Nano-8B-v1`, `R1-Distill-Qwen-7B`, `Qwen3-30B-A3B-Instruct-2507` (MoE), `OpenAI-GPT-OSS-20B` (MoE).
  - Training data: reject-sampled math trajectories derived from DeepMath-103K (Section 3.1).
  - Evaluation: eight benchmarks—four in-domain math (AIME 24/25, BRUMO25, HMMT25, BeyondAIME) and four out-of-domain general tasks (GPQA-Diamond, MMLU-Pro, LiveCodeBench V6, IFEval).
  - Baselines: Greedy Search; Default Sampling (`T=1.0, P=1.0`).  
    An “Expert-Guided Tuning” oracle baseline searches temperature (with `top-p=1.0`), then searches `top-p` with the best temperature—on the test set (Figure 3). This is not a feasible deployment strategy but gives an upper bound for static methods.
  - Metric: Pass@1 from 128 samples per problem (8 seeds × 16) for math; Pass@k (k=16/32/64) reported in Appendix (Tables 5–7).

- Main results (Tables 1–2)
  - In-domain math (Table 1; averages):
    - `Llama-Nemotron-8B`: AutoDeco 46.05 vs Default 42.59 and Greedy 42.50.
    - `R1-Distill-Qwen-7B`: 37.37 vs 34.76 and 30.58.
    - `Qwen3-30B-A3B`: 56.54 vs 56.05 and 52.25 (smaller, but consistent gain).
    - `GPT-OSS-20B`: 58.13 vs 56.64 and 51.50.
  - Out-of-domain general tasks (Table 2; averages):
    - `Llama-Nemotron-8B`: 49.72 vs 46.35 (Default) and 48.43 (Greedy).
    - `R1-Distill-Qwen-7B`: 46.88 vs 42.47 and 39.32.
    - `Qwen3-30B-A3B`: 70.24 vs 69.03 and 68.84.
    - `GPT-OSS-20B`: 59.42 vs 58.63 and 56.56.
  - Takeaway: Gains are consistent across models and domains, despite training only on math data, suggesting the learned control generalizes as a “meta-skill” for balancing determinism and randomness (Section 3.2.1).

- Pass@k analysis (Appendix; Tables 5–7)
  - Improvements persist for k=16/32/64 across models.  
  - Example (`GPT-OSS-20B`): Pass@1 58.13 vs 56.64; Pass@64 91.55 vs 89.68. The paper highlights that equal absolute gains at higher k imply larger relative error reduction (e.g., 18.1% at k=64) because error rates are smaller at high k (Section 3.2.1).

- Comparison to expert-tuned oracle (Figure 3)
  - AutoDeco’s performance is within <1 point of the oracle across datasets and models.  
  - Figure 3 also illustrates how optimal static hyperparameters vary drastically by task (e.g., for `Llama‑Nemotron‑8B`, BRUMO25 prefers `T=0.8, P=0.9` while GPQA prefers `T=0.3, P=0.6`). AutoDeco solves this by adapting per token (Section 3.2.1).

- Efficiency (Table 3; Section 3.2.2)
  - For `R1‑Distill‑Qwen‑7B` over various prompt lengths with 1k generated tokens:
    - FLOPs are unchanged to 3 significant figures.
    - Latency increases by ~0.3–0.6 s per 1k tokens, ≈1–2% relative.
    - Memory +4 MB (e.g., 15546 MB → 15550 MB for 1k prompts).
  - Conclusion: Costs are negligible relative to transformer inference.

- Ablations (Figure 4)
  - Using only the temperature head or only the top‑p head each yields ~3–3.5 point gains on AIME vs Default Sampling.
  - Joint optimization (both heads) is best, confirming complementary roles.

- Emergent and trained instruction control (Section 3.3; Figure 5; Table 4)
  - Without special training, appending “be more diverse” raises predicted `T̂/P̂`; “be as certain as possible” lowers them (Figure 5).
  - After a small targeted fine-tuning with a ranking loss, average temperature changes become consistent:
    - “Low diversity”: 0.72 → 0.61 (−0.11) with 99% consistency; top‑p 0.79 → 0.73 (−0.06) with 97% consistency (Table 4).
    - “High diversity”: 0.72 → 0.82 (+0.10) with 96% consistency; top‑p 0.79 → 0.83 (+0.04) with 85% consistency.
  - Caveat: Control is directional, not precise in absolute values (Section 3.3).

- Do the results support the claims?
  - Yes for performance and efficiency: multiple models, eight benchmarks, consistent improvements over strong non-expert baselines; near-oracle parity (Tables 1–3; Figure 3).
  - Yes for steerability as an emergent and trainable phenomenon, with quantitative evidence (Figure 5; Table 4).
  - Mixed for magnitude: Gains on some large instruct models are modest (Table 1–2), plausibly due to shorter answers and lower sensitivity to sampling (Section 3.2.1).

## 6. Limitations and Trade-offs
- Scope of control:
  - Only `temperature` and `top-p` are learned; other knobs (e.g., repetition penalty, top‑k, beam width) are not modeled. Extension is conceptually straightforward but untested here (implicit in Sections 2–3).
- Training regime:
  - Heads are trained on math-focused trajectories (DeepMath-103K). Generalization is strong but not guaranteed across all domains or modalities (Section 3.1).
  - The base LLM is frozen; instruction-based decoding control is directional rather than precise, suggesting limits of frozen-backbone learning (Section 3.3).
- Non-differentiable inference:
  - The differentiable soft top‑p is a training-time device; inference applies standard rescaling and nucleus filtering (Section 2.2). Any mismatch could, in principle, introduce small train–test gaps (not observed to be problematic empirically).
- Hyperparameters:
  - The soft-mask steepness α is fixed (α=30; Section 6). Sensitivity analyses are not reported.
- Evaluation breadth:
  - Four model families are evaluated with detailed benchmarks; larger released variants are not fully benchmarked due to compute cost (Abstract; Section 3.1).
- Oracle baseline:
  - The “Expert-Guided Tuning” assumes access to the test set for hyperparameter tuning (Figure 3). It is appropriately framed as an upper bound, but actual practitioners cannot do this.

## 7. Implications and Future Directions
- Impact on practice:
  - AutoDeco can eliminate manual decoding sweeps. For production systems, this means higher, more stable performance across diverse inputs with zero per-task retuning and ~1–2% extra latency (Sections 3.2.1–3.2.2; Table 3).
  - The per-token `T̂/P̂` traces can serve as an introspective signal of uncertainty or creativity demand, aiding downstream systems (e.g., when to verify, when to sample more).
- Impact on research:
  - Decoding becomes part of the learnable model. This reframes generation quality as an optimization problem over both “what” to say and “how” to explore while saying it.
  - The differentiable soft top‑p is a tool that can enable training other decoding controls end-to-end (e.g., repetition penalties, length normalization).
  - Natural-language control of decoding opens a new interface—users express style constraints in words rather than API knobs—and suggests work on more precise, quantitative control (Section 3.3).
- Concrete next steps (some noted in Section 5 “Future Work” and Section 3.3):
  - Jointly train the backbone with AutoDeco heads to achieve precise, absolute control (e.g., truly near‑zero temperature on “no randomness” prompts).
  - Expand controllable parameters (top‑k, penalties, contrastive or speculative decoding weights) and study interactions among them.
  - RL or bandit-style objectives that directly reward task outcomes while penalizing unnecessary randomness or verbosity.
  - Task-specific curricula (reasoning vs. creative writing) to learn richer, context-aware decoding policies.
  - System integration: couple `T̂/P̂` with self-verification or tool-use modules to condition verification or drafting depth on predicted uncertainty.

> Bottom line: By turning decoding from a static, manual post-process into a learned, token-level component of the model, AutoDeco delivers practical gains today (accuracy, simplicity, negligible overhead) and opens a rich design space for controllable, instruction-steerable generation tomorrow.
