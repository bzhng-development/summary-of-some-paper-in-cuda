# LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens

**ArXiv:** [2402.13753](https://arxiv.org/abs/2402.13753)
**Authors:** Yiran Ding, Li Lyna Zhang, Chengruidong Zhang, Yuanyuan Xu, Ning Shang, Jiahang Xu, Fan Yang, Mao Yang
**Institutions:** (Not specified on arXiv abstract)

## 🎯 Pitch

LongRoPE revolutionizes large language models by extending their context window to over two million tokens with minimal fine-tuning, enhancing their ability to handle multi-document reasoning, comprehensive codebases, and long transcripts. This breakthrough makes it feasible to deploy LLMs in applications requiring extensive memory and interaction without sacrificing short-context performance, drastically expanding their practical utility and efficiency.

---

## 1. Executive Summary
LongRoPE is a method for expanding the usable context window of large language models (LLMs) from thousands of tokens to over two million tokens (2048k) while largely preserving short-context performance. It does this by learning how to rescale the model’s positional encoding in a non-uniform, data-driven way and by using a progressive extension strategy that requires only 1,000 fine-tuning steps at ≤256k training lengths.

## 2. Context and Motivation
- Problem addressed
  - LLMs are typically limited to short context windows (e.g., 4k tokens for `LLaMA2`; Section 1). When inputs exceed this limit, models degrade because they encounter position indices they were never trained on.
  - Extending windows far beyond 128k is hard for three reasons (Section 1):
    - Catastrophic values at unseen positions make optimization unstable when most positions are new.
    - Scarcity of very long training texts (hundreds of thousands to millions of tokens) and high compute costs.
    - Longer contexts can dilute attention and reduce performance within the original short window.

- Why this matters
  - Many real applications—multi-document reasoning, long transcripts, codebases, long-term memory for agents—require hundred-thousand to million-token contexts (Section 1).

- Prior approaches and their gaps
  - Positional Interpolation (PI): uniformly compresses positions by a factor `s` (extension ratio). It crowds positions together and degrades discrimination at high `s` (Section 2.1; “Linear positional interpolation (PI)”).
  - NTK-based scaling: scales different RoPE dimensions differently to spread compression but typically tops out around 4× extension without fine-tuning (Section 2.1).
  - YaRN: groups dimensions into three bands and applies different rules (some extrapolate, some interpolate), but the grouping is hand-designed and can be suboptimal, especially at high extension ratios (Sections 2.1–2.2; Table 1, Figure 3).

- Positioning
  - LongRoPE treats interpolation design as a search problem over two underused sources of non-uniformity (Section 2.2):
    - Different RoPE dimensions carry different frequencies and importance.
    - Early token positions are more critical for attention and benefit from less interpolation.
  - With an efficient evolutionary search and a progressive fine-tuning schedule, LongRoPE extends to 2048k with markedly lower compute than naïve training on million-length sequences (Sections 3 and 3.3).

## 3. Technical Approach
The method centers on smarter rescaling of RoPE positional encodings and on a staged training/extension plan.

- Brief background: RoPE and interpolation
  - RoPE (rotary position embedding) represents token position `n` by rotating pairs of hidden dimensions using angles `nθ_i` that vary across dimensions `i` (Equation (1); Section 2.1).
  - Extending the window from length `L` to `L' = s · L` typically rescales these angles so that unseen positions map into the pretrained range (Equation (2)).
  - The challenge: uniform rescaling (PI) destroys resolution in the original region; fixed rules (NTK, YaRN) help but can’t capture model- or length-specific subtleties (Table 1; Figure 3).

- Key idea: two non-uniformities to exploit (Section 2.2)
  1) RoPE dimension-wise differences
     - Lower-index RoPE dimensions encode higher frequency signals and should be compressed less; higher-index dimensions can absorb more compression or even extrapolation. This is consistent with NTK-inspired intuition but LongRoPE searches the actual per-dimension factors rather than fixing a formula (Table 1 shows gains from per-dimension search).
  2) Position-wise differences
     - The first `n̂` tokens in a sequence often receive large attention and should be left un-interpolated to preserve fidelity (Table 2 shows performance improves when initial tokens are exempted from interpolation).

- Formalization (Equation (3); Section 3.1)
  - For each RoPE dimension `i`, LongRoPE applies a rescale factor `λ̂_i` but only starting from token position `n ≥ n̂`. For the first `n̂` tokens, it uses the original RoPE angles.
  - The goal is to choose `{λ̂_i}` and `n̂` that minimize perplexity on validation texts of at least the target length `L'`.

- Search space and constraints (Section 3.2; Table 4)
  - Search over:
    - Per-dimension factors `λ_i` (the implementation searches `λ_i`, where `λ̂_i = 1/λ_i`) from 1.0 (pure extrapolation, no compression) up to `1.25 × s` (slightly more compression than PI) in steps of 0.01.
    - The starting-token cutoff `n̂` from a small discrete set {0, 1, 2, 4, 8, 12, 16, 20, 24, 28, 32, 64, 128, 256}.
  - Evolutionary search (Algorithm 1):
    - Initialization includes the known baselines (PI, NTK, YaRN) to seed the population with strong candidates.
    - A monotonicity constraint `λ_i ≤ λ_{i+1}` biases the search to “compress less in low-index (high-frequency) dimensions and more in high-index (low-frequency) dimensions,” both shrinking search space and aligning with signal-processing intuition (Section 3.2).

- Progressive extension strategy (Section 3.3)
  - Step A: Search for 128k and 256k rescale factors on the pretrained model.
  - Step B: Fine-tune to 256k using only 1,000 steps total:
    - 400 steps at 128k with the 128k factors, then switch to the 256k factors and run 600 more steps (Section 3.3; Appendix A.2 and Figure 5).
  - Step C: Perform a second search on the 256k-fine-tuned model to reach 2048k without more fine-tuning.
  - Step D (short-context recovery): Run an extra search targeted at 4k–8k to slightly relax interpolation for short sequences; at inference time, choose the rescale set based on the actual prompt length (Section 3.3; Table 10).

- Practical intuition
  - Think of PI as uniform “squeezing” of a ruler: all tick marks get equally compressed, making nearby ticks indistinguishable. LongRoPE instead learns a non-uniform ruler: it preserves the fine ticks near the origin (first `n̂` positions) and in the most informative frequency bands (low-index RoPE dimensions), while compressing other regions more aggressively.

## 4. Key Insights and Innovations
- Non-uniform, searched positional interpolation across RoPE dimensions and token positions
  - What’s new: rather than fixed formulas or hand-set groups, LongRoPE searches per-dimension rescaling and a cutoff `n̂` that leaves the earliest tokens untouched (Section 3.2; Equation (3)).
  - Why it matters: greatly lowers perplexity without fine-tuning at 8k–16k (Table 1) and gives a strong initialization for fine-tuning at longer lengths (Table 3).

- Evolutionary search with domain-informed constraints
  - What’s new: an efficient search that seeds with PI/NTK/YaRN, enforces monotonic `λ_i`, and evaluates perplexity on long validation texts (Algorithm 1; Figure 6).
  - Why it matters: enables an 8× extension without any fine-tuning (Figure 3), where competing methods spike after ~2×–4×.

- Progressive extension to 2048k with minimal fine-tuning
  - What’s new: fine-tune only to 256k (1,000 steps total) and then use a second RoPE search to reach 2048k (Section 3.3).
  - Why it matters: avoids the prohibitive cost and data requirements of training on million-token sequences; still achieves usable perplexities and strong retrieval accuracy (Tables 6 and 9; Figure 4).

- Short-context performance recovery via targeted readjustment
  - What’s new: a dedicated short-length search that reduces over-compression in the original window, applied dynamically at inference (Section 3.3).
  - Why it matters: maintains competitive performance on standard 4k–8k benchmarks (Table 8) and lowers perplexity at 4k–8k (Table 10).

## 5. Experimental Analysis
- Evaluation setup (Section 4.1)
  - Models: `LLaMA2-7B` and `Mistral-7B`.
  - Tasks:
    - Long-sequence language modeling via perplexity on Proof-Pile/Books3/PG19 (Tables 5–7).
    - Passkey retrieval (a synthetic “needle-in-a-haystack” task) up to 2048k (Figure 4).
    - Standard short-context benchmarks from the Hugging Face Open LLM Leaderboard (Table 8).
  - Baselines: fine-tuned models using PI, NTK, or YaRN (e.g., Together-32k, Code LLaMA-100k, LongLoRA-full-FT-100k, YaRN-LLaMA/Mistral).

- Main quantitative results
  - 8× extension without fine-tuning
    - On PG19 and Proof-Pile, LongRoPE maintains low perplexity up to 32k where PI/NTK/YaRN degrade after 8k–16k (Figure 3).
  - Within 256k context (Tables 5 and 7)
    - On Proof-Pile, `LLaMA2-7B LongRoPE-2048k (ft=256k)` achieves a perplexity of 1.87 at 262,144 tokens, outperforming YaRN and NTK baselines that either spike or fail at this length.
    - On PG19, the LongRoPE models show decreasing or stable perplexity up to 128k, e.g., `LLaMA2 LongRoPE-2048k (ft=256k)` reaches 6.31 at 128k (Table 7).
  - Beyond 2000k context (Table 6)
    - On Books3, LongRoPE reaches 2048k with usable perplexity where baselines collapse. For `LLaMA2-7B`:
      - > “LongRoPE-2048k (ft=256k)” achieves perplexity 7.08 at 2048k; competing PI/NTK/YaRN models fail beyond ~128k–256k (Table 6).
    - For `Mistral-7B`, perplexity increases after 256k, reaching 12.78–13.71 at 2048k, but still successfully runs, unlike baselines that crash much earlier.
  - Passkey retrieval (Figure 4)
    - > `LLaMA2 LongRoPE-2048k (ft=256k)` maintains ≥90% accuracy from 4k up to 2048k.
    - > `Mistral LongRoPE-2048k (ft=128k)` maintains 100% up to ~1800k and ~60% at 2048k.
  - Standard short-context benchmarks (Table 8)
    - `Mistral LongRoPE-2048k` matches the original Mistral on ARC/HellaSwag/MMLU and slightly improves TruthfulQA by +0.5 points (43.1 vs. 42.6).
    - `LLaMA2 LongRoPE-2048k (ft=128k)` stays close to original LLaMA2; the `(ft=256k)` variant shows more drop but remains within a few points.
  - Ablations
    - Second interpolation matters: on `LLaMA2-256k`, secondary PI or YaRN interpolation degrades markedly as length grows, while LongRoPE remains stable (Table 9; e.g., 20.17 vs. 7.08 at 2048k).
    - Short-context recovery helps: Proof-Pile perplexity at 4k improves from 4.16→3.71 and average leaderboard accuracy rises from 49.3→52.9 for `(ft=128k)` (Table 10).
    - Which non-uniformity helps most? Dimension-wise search gives the bulk of the gains; the “no interpolation for first `n̂` tokens” helps at 16k–32k but not notably at 2048k (Table 11).

- Compute and data considerations
  - Fine-tuning cost (Appendix A.2): `LLaMA2-128k` uses 8×A100 GPUs for one week (400 steps); `LLaMA2-256k` uses 16×A100 for two weeks (600 steps). `Mistral-128k/256k` fine-tuned on 16k-length data with 4×A100 for two days.
  - Search cost (Appendix A.3): up to 256k can be searched within ~3 days on 1×A100; 512k uses 2 GPUs; 1024k and 2048k use 4 and 8 GPUs within ~5 days.
  - Search efficiency plots (Figure 6) show perplexity steadily dropping across iterations and that the 8× extension from 256k→2048k is easier than 16× from 128k→2048k.

- Do the experiments support the claims?
  - Yes, on three fronts:
    - Non-fine-tuned extension: strong evidence from Figure 3 and Tables 1–3.
    - Million-token window: hard failure of baselines vs. workable perplexity and high passkey accuracy for LongRoPE (Table 6; Figure 4).
    - Short-context retention: competitive leaderboard scores and targeted recovery (Table 8; Table 10).
  - Caveats: `Mistral` shows increasing perplexity at very long lengths; the largest-length evaluations use a limited set of books for tractability (Table 6 uses 20 Books3 samples).

## 6. Limitations and Trade-offs
- Assumptions and scope
  - Designed for RoPE-based models; applicability to other positional schemes is not demonstrated (Section 4.1 notes “RoPE embedding” focus).
  - The objective used in search is perplexity on a small long-text validation set, which may not capture all downstream behaviors.

- Practical constraints
  - Evolutionary search at million-token lengths is expensive (Appendix A.3: a single 2048k perplexity evaluation takes ~50 minutes); though far cheaper than direct fine-tuning at those lengths, it’s still resource-intensive.
  - Fine-tuning still requires substantial GPUs: e.g., `LLaMA2-256k` uses 16×A100 for two weeks (Appendix A.2).

- Performance trade-offs
  - Some loss in short-context benchmarks, especially for `LLaMA2 (ft=256k)` before recovery (Table 8 and Table 10).
  - The “no-interpolation for first `n̂` tokens” helps at moderate lengths but shows limited effect at 2048k (Table 11), suggesting position-based heuristics may not scale uniformly.

- Open questions
  - No formal theory guarantees optimality of the searched `λ_i` or `n̂`. The monotonicity constraint is heuristic, albeit motivated by NTK intuition (Section 3.2).
  - How to automate choosing validation texts and search hyperparameters to generalize across domains remains unclear.

## 7. Implications and Future Directions
- Field impact
  - Demonstrates that million-token contexts are achievable on existing RoPE-based LLMs without architectural changes or massive retraining. This shifts the practical ceiling on context length and opens the door to native long-context workloads.

- Practical applications
  - Long-document understanding: legal/medical documents, books, scientific papers.
  - Software engineering: whole-repository comprehension or refactoring.
  - Conversational systems and agents: persistence over months of interaction logs.
  - Research tools: cross-document synthesis without chunking or retrieval.

- Follow-up research
  - Theory: characterize when and why certain RoPE dimensions and token regions need less/more interpolation; formalize error bounds of interpolation strategies.
  - More efficient search: surrogate models, differentiable relaxations, or Bayesian optimization to reduce 2048k evaluation cost.
  - Dynamic, input-aware interpolation: adjust `λ_i` and `n̂` on-the-fly based on prompt statistics, not only target length.
  - Integration with attention optimizations: combine with streaming attention or memory-augmented transformers to further improve stability at 1–2M tokens (Related Work, Section 5).
  - Beyond RoPE: adapt the non-uniform search idea to other positional encodings or to per-head, per-layer scaling.
  - Broader evals: larger and more diverse long-context tasks beyond Books3 and passkey retrieval, including real-world QA and reasoning benchmarks at million-token scales.

Overall, the paper shows that careful, data-driven control of positional encodings—rather than uniform rules—can unlock multi-million-token contexts with modest fine-tuning, while preserving much of the original model’s capabilities at short lengths. References to Sections 2–4, Tables 1–11, Figure 3–6, and Equations (1)–(3) support both the mechanism and the empirical gains.
