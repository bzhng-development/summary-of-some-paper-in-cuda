# EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty

**ArXiv:** [2401.15077](https://arxiv.org/abs/2401.15077)

## 🎯 Pitch

EAGLE introduces a novel speculative sampling framework for large language models that accelerates inference by predicting future internal features—rather than tokens—while addressing the inherent uncertainty of feature autoregression by conditioning on advanced token sequences. This allows EAGLE to achieve substantial speedups (up to 3.8x) across a range of LLMs and task types, all while provably preserving the original model's output distribution. By bypassing the need for cumbersome or ill-matched draft models and maintaining generation fidelity, EAGLE advances fast, reliable, and practical LLM deployment for quality-critical applications.

---

## 1. Executive Summary (2-3 sentences)
This paper introduces EAGLE, a “lossless” acceleration framework for large language model (LLM) decoding that predicts future internal features rather than future tokens and conditions these predictions on the next-token choice to resolve sampling uncertainty. Across Vicuna, LLaMA2-Chat, and Mixtral models, EAGLE achieves 2.1x–3.8x latency speedups at temperature 0 and up to 3.5x on LLaMA2-Chat 70B while provably preserving the original model’s output distribution (Figures 1–2, Table 1, Section 2 “Speculative sampling,” Section 3.3).

## 2. Context and Motivation
- Problem addressed:
  - Autoregressive decoding in LLMs generates one token per forward pass, making inference slow and costly (Section 1). 
  - Speculative sampling speeds this up by drafting multiple tokens with a cheap model and verifying them in parallel on the original model, but it depends on a suitable low-overhead draft model and high acceptance accuracy to deliver strong speedups (Section 1; “Speculative sampling” in Section 2).

- Why it matters:
  - Latency and cost are major bottlenecks in deploying LLMs at scale. Faster generation without changing the model’s output distribution is valuable for quality-sensitive applications in dialog, code, math, and instruction following (Abstract; Figures 1–2; Table 1).

- Limitations of prior approaches:
  - Classic speculative sampling uses a smaller LLM as the draft model. For small targets (e.g., 7B), there may be no smaller instruction-tuned draft with compatible prompts; for mid-size targets (e.g., 13B), a 7B draft’s overhead can negate gains (Figure 1, discussion in Section 1).
  - Lookahead and Medusa reduce draft overhead by predicting future tokens directly from n-gram heuristics or MLP heads attached to the target model, but draft accuracy is modest (≈0.6 for Medusa, lower for Lookahead), limiting speedups (Section 1, Figure 4).
  - Many works either focus on greedy decoding only or relax the “lossless” guarantee (distribution preservation) for speed (Related Work, Section 5).

- Positioning:
  - EAGLE keeps the target LLM fixed and “lossless” (same output distribution as vanilla decoding) for both greedy and non-greedy settings (Section 2 “Speculative sampling,” Section 3.3).
  - It replaces token-level drafting with feature-level drafting and resolves feature-level uncertainty by conditioning on a token sequence shifted one step ahead (Sections 3.1–3.2; Figures 3, 5, 6).

## 3. Technical Approach
EAGLE has two phases like speculative sampling: a drafting phase that proposes multiple future tokens, and a verification phase that checks them in parallel on the target LLM. Its core differences are in how the draft is produced.

Key concepts (defined as used here):
- `feature` (or `f`): the hidden state at the second-to-top layer of the target LLM, right before the final projection (`LM Head`) to token logits (Sections 1–2).
- `LM Head`: the target LLM’s final linear layer mapping a feature vector to a token probability distribution (Section 2, Notations).
- `tree attention`: an attention mask/layout that lets the target LLM verify a tree of drafted tokens in one forward pass (used also by SpecInfer and Medusa). Higher-probability branches can be made deeper/wider (Appendix A.1, Figure 9).
- `average acceptance length` (`τ`): average number of tokens accepted by the target LLM per forward pass during verification (Section 4, “Metrics”).
- `acceptance rate` (`α`): fraction of drafted tokens accepted; for EAGLE’s analysis this is measured on a chain draft (no tree) and is reported as `n-α` to account for n predicted features that can propagate error (Section 4, “Metrics”).

Step-by-step methodology:

1) Drafting at the feature level with “shifted” tokens (Sections 3.1–3.2; Figures 5–6)
- Input to the draft model at step i:
  - The sequence of past features `F1:i` produced by the target LLM up to time i.
  - A token sequence `T2:i+1` that is “advanced by one time step,” i.e., it includes the next token choice at each position (Figure 3 shows why this matters).
    - Why: token sampling introduces randomness. If the next token after “I” could be “am” or “always,” their next features differ (Figure 3). Conditioning on the actual sampled token disambiguates the next feature.
- Architecture (Figure 6):
  - Reuse the target LLM’s embedding layer and `LM Head` (frozen; blue “snowflake” blocks in Figure 6).
  - Concatenate token embeddings (for `T2:i+1`) with features (`F1:i`) to form a fused sequence with shape `(bs, seq_len, 2×hidden_dim)`.
  - A small `Autoregression Head` (one fully connected layer for dimension reduction + one transformer decoder layer) predicts the next feature `f_{i+1}` from the fused sequence (red boxes in Figure 6).
  - Use the target `LM Head` on `f_{i+1}` to get a token distribution and sample the token `t_{i+2}`; append (`f_{i+1}`, `t_{i+2}`) and repeat to build a draft.
- Tree-structured draft:
  - EAGLE grows a token tree with depth `m` in only `m` forward passes of the draft model (Figure 6). For example, it drafts 10 tokens with 3 passes.
  - In greedy settings, child nodes are top-k tokens by probability; in sampling settings, k tokens are sampled (Appendix A.1, Figure 9).

2) Training the draft model (Section 3.2)
- Objective combines:
  - Feature regression: predict the next feature accurately using Smooth L1 loss
    - L_reg = SmoothL1(f_{i+1}, DraftModel(T_{2:i+1}, F_{1:i})) (Section 3.2).
  - Distribution matching: ensure that the `LM Head` applied to predicted features yields a token distribution close to the true one
    - p_{i+2} = Softmax(LMHead(f_{i+1})), p̂_{i+2} = Softmax(LMHead(f̂_{i+1}))
    - L_cls = CrossEntropy(p_{i+2}, p̂_{i+2}) (Section 3.2).
  - Total loss: L = L_reg + w_cls L_cls, with w_cls = 0.1 because L_cls is numerically larger (Section 3.2).
- Data and robustness:
  - Trained on 68k ShareGPT dialogues with the target LLM frozen (Training, Section 4).
  - Adds uniform noise U(-0.1, 0.1) to target features during training to make the autoregression head robust to imperfect features and reduce error accumulation (Section 3.2).
- Trainable parameter counts:
  - ≈0.24B/0.37B/0.56B/0.99B for 7B/13B/33B/70B targets; 0.28B for Mixtral 8×7B (Training, Section 4).
  - Reported training time: 1–2 days on 4×A100 40G for 70B; 1–2 days on a single RTX 3090 for 7B/13B/33B (Abstract; Section 4 “Training”).

3) Verification with multi-round speculative sampling over a tree (Section 3.3; Appendix A.2; Algorithm 1)
- The target LLM performs a single forward pass with a tree attention mask to compute probabilities for all nodes in the drafted tree (Section 3.3).
- At each node, EAGLE applies standard speculative sampling acceptance logic: accept a drafted token with probability min(1, p(t)/p̂(t)); if rejected, adjust the distribution and resample (Section 2 “Speculative sampling”).
- Because the draft is a tree (k candidates per position), verification proceeds recursively across candidates (“Multi-round speculative sampling,” Algorithm 1 in Appendix A.2). If all candidates are rejected, sample once from the adjusted distribution.
- This process is mathematically equivalent to sampling from the target LLM directly; hence the output distribution is preserved for both greedy and non-greedy decoding (Section 2 “Speculative sampling,” Section 3.3; Appendix A.2).

Why this design:
- Feature-level autoregression is “easier” (more regular) than token-level, increasing draft accuracy (Figure 4 compares token vs feature vs feature+shifted-token).
- Shifting the token inputs by one step resolves ambiguity in the next feature due to sampling randomness (Figure 3), raising accuracy further (Figure 4; Figure 8 ablation).
- Tree attention increases the number of tokens per verification pass without extra forward passes, improving acceptance length and speed (Section 4.3.1; Table 5; Figure 7).

## 4. Key Insights and Innovations
- Feature-level drafting instead of token-level drafting (fundamental)
  - What’s new: Predict the next internal feature (`f_{i+1}`) then convert it to a distribution via the frozen `LM Head`, rather than predicting the next token directly (Section 3.1; Figure 5).
  - Why it matters: Feature sequences are more regular than token sequences. In Figure 4, feature-only drafting outperforms token-only drafting in both acceptance accuracy and speedup.

- Resolving feature uncertainty with a one-step shifted token sequence (fundamental)
  - What’s new: Include the token sampled at the next step (`T_{2:i+1}`) as input when predicting `f_{i+1}` (Figures 3, 5–6).
  - Why it matters: Sampling introduces branching futures; conditioning on the realized token collapses ambiguity. Figure 4 shows a jump from ≈1.9x (feature-only) to ≈2.8x speedup with feature+shifted-token for Vicuna 7B at T=0; Figure 8 replicates this across epochs and temperatures.

- Lossless, tree-based verification with multi-round speculative sampling (incremental but practical)
  - What’s new: Apply standard speculative sampling acceptance to a tree draft via a recursive scheme (Appendix A.2, Algorithm 1). 
  - Why it matters: Improves `τ` by verifying many candidates per pass while preserving the target model’s output distribution for both greedy and sampling regimes (Section 3.3).

- Lightweight plug-in design and low training cost (practical)
  - What’s new: Reuse target `Embeddings` and `LM Head`; train only a single transformer decoder layer + FC as an `Autoregression Head` (Figure 6).
  - Why it matters: No fine-tuning of the target LLM; training is modest (Section 4 “Training”). It generalizes across datasets without per-task retraining (Section 1 “Beyond performance: Generality/Reliability”).

## 5. Experimental Analysis
Evaluation setup (Sections 4.1–4.4):
- Models: Vicuna (7B, 13B, 33B), LLaMA2-Chat (7B, 13B, 70B), Mixtral 8×7B Instruct (Abstract; Section 4).
- Tasks: dialog (MT-bench), code (HumanEval), math (GSM8K), instruction following (Alpaca) (Abstract; Section 4).
- Metrics: walltime speedup vs vanilla decoding; average acceptance length `τ`; acceptance rates `α` for chain drafts with 0–4 predicted features to test robustness (Section 4 “Metrics”, Tables 1–2, 3, 8).
- Batch size: primarily 1, following prior speculative sampling work; throughput analyses include larger batch sizes (Section 4.4; Table 7).

Main quantitative results:
- Overall speedups on MT-bench (Figures 1–2):
  - Greedy (T=0): EAGLE achieves ≈2.7x–3.5x on LLaMA2-Chat 70B; ≈2.7x–3.1x on 7B/13B; ≈2.8x–3.0x on Vicuna series. It outpaces Lookahead by ≈1.7x–2.1x and Medusa by ≈1.5x–1.6x (Figure 1 caption and text).
  - Sampling (T=1): EAGLE remains “lossless” and obtains ≈2.1x–2.9x across models (Figure 2). Lookahead does not apply here; Medusa is not lossless at T>0 so is omitted (Figure 2 caption).
- Task-wise speedups and `τ` (Table 1):
  - Code (HumanEval) yields the best gains. Example: LLaMA2-Chat 13B at T=0 reaches 3.76x with τ=4.52; Vicuna 33B reaches 3.67x with τ=4.28.
  - At T=1, speedups drop (e.g., LLaMA2-Chat 13B: 2.89x, τ=3.78), but remain substantial.
- Acceptance analysis on MT-bench (Table 2):
  - `τ` ranges ≈3.17–3.98 at T=0; ≈3.17–3.46 at T=1.
  - 0-α (no feature errors) is high (≈0.71–0.79 at T=0), while 1-α drops, showing that feature errors matter; 1-α to 4-α stays relatively stable, indicating robustness to error accumulation.
- Mixtral 8×7B MoE (Table 3):
  - Speedup 1.50x with τ=3.25; lower than dense models because verification may involve more expert weights than vanilla decoding (discussion below Table 4).
- Combination with gpt-fast (Table 4):
  - On LLaMA2-Chat 7B (RTX 3090), EAGLE + gpt-fast reaches 160.4 tokens/s at int4, 100.2 tokens/s at FP16. Vanilla HF is 24.5 tokens/s at FP16.
- Tree attention ablation (Section 4.3.1; Figure 7; Table 5):
  - Using tree attention increases τ by ≈0.62–0.75 and speedups by ≈0.3–0.5, without increasing the number of forward passes; it only increases tokens processed per pass.
- Input ablation (Section 4.3.2; Figure 8):
  - Feature-only > token-only when draft model capacity is small.
  - Combining features with unshifted tokens helps (tokens act as precise anchors).
  - Feature + shifted-token (EAGLE) yields the largest jump in speed and `0-α`, especially at T=0; the gain persists at T=1.
- Training data ablation (Section 4.3.3; Table 6):
  - Training on fixed ShareGPT pairs vs the target-LMM-generated answers makes only a small difference (2.78x vs 2.88x, τ=3.62 vs 3.75 on LLaMA2-Chat 7B), suggesting low sensitivity to training data.
- Batch size and throughput (Section 4.4; Table 7):
  - Speedup decreases as batch size increases (compute becomes less idle). Example on LLaMA2-Chat 70B: 3.01x (bs=1) → 2.40x (bs=4).
  - Throughput roughly doubles vs vanilla (≈1.97x–1.99x), even accounting for slightly higher memory usage; maximum bs is smaller for EAGLE due to tree attention (bs=7 vs 8 on 7B with 24G VRAM; bs=4 vs 5 on 70B with 160G).

Do results support the claims?
- Yes for latency: Across multiple models and tasks, EAGLE consistently increases τ to ≈3.2–4.5 and achieves 2.1x–3.8x speedups at T=0 and ≈2.1x–2.9x at T=1 (Figures 1–2; Tables 1–2).
- “Lossless” claim: The verification phase follows standard speculative sampling theory and an explicit multi-round procedure over trees (Section 2; Section 3.3; Appendix A.2), which preserves the target model’s distribution for both greedy and non-greedy decoding.
- Robustness and mechanism are probed by ablations (Figures 4, 7–8; Tables 5–6): they directly test the impact of feature vs token, shifted vs unshifted tokens, tree attention, and training data.

## 6. Limitations and Trade-offs
- Access and integration requirements:
  - Requires internal access to the target LLM’s second-to-top-layer features and to reuse its `Embedding` and `LM Head` weights. This is straightforward for open models but may be infeasible for closed APIs.
- Additional memory and engineering complexity:
  - Tree attention and token-tree verification increase memory and implementation complexity; maximum batch size is smaller vs vanilla under the same memory budget (Section 4.4; Table 7).
- Temperature sensitivity:
  - Speedups are lower for non-greedy sampling (T=1) than for greedy (T=0) across all models (Figures 1–2; Table 1).
- MoE models:
  - Gains are more modest (1.50x on Mixtral 8×7B; Table 3) because verification can touch more expert weights per pass than vanilla decoding, eroding the benefit (discussion after Table 4).
- Draft accuracy still matters:
  - Although feature+shifted-token significantly improves acceptance, errors in features still reduce `α` (Table 2: drop from 0-α to 1-α). Noise augmentation helps but does not eliminate this.
- Tree shape not optimized:
  - The tree layout is chosen heuristically; the paper notes it is “not rigorously optimized,” and optimal structure may depend on context and batch size (Appendix A.1).

Open questions:
- How does EAGLE perform on languages or domains far from ShareGPT-style dialogs without any retraining?
- What are the best tree policies for different hardware, batch sizes, and temperatures?
- Can similar “feature + shifted-token” ideas extend to other architectures beyond decoder-only LLMs or to multi-modal models?

## 7. Implications and Future Directions
- Field impact:
  - Demonstrates that “lossless” acceleration can be pushed further by moving drafting to the feature space and explicitly resolving sampling uncertainty with shifted tokens. This reframes speculative decoding design space away from token prediction toward internal state prediction (Sections 1, 3; Figures 3–4).
- Practical applications:
  - Low-latency chat systems, code assistants, math solvers, and instruction followers that require the exact same output distribution as the baseline model (Abstract; Tables 1–2). 
  - Production deployments can combine EAGLE with quantization/compilation (e.g., gpt-fast) for additive gains—up to 160.4 tokens/s on LLaMA2-Chat 7B int4 on a single RTX 3090 (Table 4).
- Follow-up research:
  - Auto-tuning tree structures and branching policies for different workloads and hardware (Appendix A.1 notes this as future optimization).
  - Extending the “feature + shifted-token” idea to multi-modal, retrieval-augmented, or structured decoding settings.
  - Investigating better robustness to feature prediction errors (e.g., multi-step consistency losses, teacher-forced vs scheduled sampling in feature space).
  - Specialized designs for MoE verification to avoid reading many experts per pass, improving MoE speedups (Table 3 discussion).
  - Exploring partial fine-tuning of early layers to produce more predictable second-to-top features without changing the LM head distribution.

> Core takeaway: By predicting the next internal feature conditioned on the realized next token and verifying a token tree with a lossless speculative sampler, EAGLE turns each target-LM pass into ≈3–4 accepted tokens on average (Tables 1–2), achieving consistent 2–4x latency gains while keeping the output distribution identical to vanilla decoding (Sections 2–3; Figures 1–2).
