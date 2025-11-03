# Falcon‑H1: A Family of Hybrid‑Head Language Models Redefining Efficiency and Performance

**ArXiv:** [2507.22448](https://arxiv.org/abs/2507.22448)
**Authors:** Jingwei Zuo, Maksim Velikanov, Ilyas Chahed, Younes Belkada, Dhia Eddine Rhayem, Guillaume Kunsch, Hakim Hacid, Hamza Yous, Brahim Farhat, Ibrahim Khadraoui, Mugariya Farooq, Giulia Campesan, Ruxandra Cojocaru, Yasser Djilali, Shi Hu, Iheb Chaabane, Puneesh Khanna, Mohamed El Amine Seddik, Ngoc Dung Huynh, Phuc Le Khac, Leen AlQadi, Billel Mokeddem, Mohamed Chami, Abdalgader Abubaker, Mikhail Lubinets, Kacper Piskorski, Slim Frikha
**Institutions:** Technology Innovation Institute (TII)

## 🎯 Pitch

Falcon-H1 revolutionizes large language models with a hybrid architecture that combines minimal Transformer attention and a robust State Space Model, achieving performance often surpassing 70B-class models while maintaining remarkable efficiency at long contexts. This innovation is crucial as it offers environmentally sustainable AI solutions and expands accessibility, particularly for real-world applications requiring extensive contexts and modest computational resources.

---

## 1. Executive Summary (2-3 sentences)
Falcon-H1 introduces a “hybrid‑head” large language model (LLM) family that mixes a small amount of Transformer attention with a larger State Space Model (SSM, specifically Mamba‑2) inside each layer, run in parallel and then fused. With careful architectural choices, training dynamics (effective LR/WD, µP multipliers), tokenizer design, and data strategy, the 34B model often matches or exceeds 70B-class models, while supporting 256K context and delivering large long‑context throughput gains (up to 4× prefill and 8× generation; §5.3, Fig. 16).

## 2. Context and Motivation
- Gap addressed
  - Transformers scale quadratically with sequence length, making very long contexts expensive (§1). SSMs such as Mamba offer linear-time sequence mixing and strong long-context memory, but pure-SSM models can lose precision on tasks where attention excels (§1–§2).
  - Existing hybrids typically wire SSM and attention sequentially and keep their channel sizes equal, limiting flexibility (e.g., Jamba, Samba, Zamba; §1, §2). Mamba design hyperparameters are also less explored than standard Transformers (§2.2).
- Why it matters
  - Real deployments increasingly require long-context (RAG systems, large documents, multi-turn dialogue), multilingual coverage, and efficient inference on modest hardware. Improving parameter and training efficiency reduces cost and environmental impact while expanding accessibility (§1, §7).
- Prior approaches and their limits
  - Pure Transformers: strong accuracy, poor long-context efficiency (§1). 
  - Pure SSMs (e.g., RWKV, Mamba): efficient, but can struggle on some precision-demanding tasks and lack mature training recipes (§1–§2.2).
  - Earlier hybrids: mostly sequential wiring that forces equal attention/SSM dimensions, curbing design freedom (§2, contrast to Dong et al., 2024).
- Positioning
  - Falcon-H1 adopts a parallel hybrid block that lets attention and SSM see the same input and then concatenates their outputs, so their channel allocations can be tuned independently (§2, Fig. 1; §2.1, Eqs. 3–5). The series revisits SSM hyperparameters, training stability, effective schedule design, tokenizer and data strategy, and distributed training to make hybrids practical at multiple scales (0.5B→34B; Table 1).

## 3. Technical Approach
This section explains how Falcon-H1 is built and trained, why those choices were made, and how the system is evaluated and deployed.

- Hybrid-head block (how a layer works)
  - Each decoder block has two token mixers in parallel: an attention head group and a Mamba‑2 SSM head group. Both consume the same normalized input; their outputs are concatenated and projected back to the model dimension (§2, Fig. 1).
  - The block order is “semi‑parallel”: attention and SSM run in parallel, then MLP runs on the residual updated by both (“SA_M” in §2.1, Eq. 4). This outperformed fully-parallel and fully-sequential variants (Fig. 2 right).

- Channel allocation (how many channels go to SSM vs. attention vs. MLP)
  - Falcon-H1 discretizes the inner dimensions into 8 “chunks” that can be allocated across SSM, attention, and MLP (§2.1, Eq. 1–2).
  - Exhaustive sweeps on ~1.2B proxies show:
    - Putting more channels into attention hurt performance; the best attention fraction is minimal (1/8 of chunks; Fig. 2 left).
    - The best block order is SA_M with a roughly 2:1:5 ratio across SSM:Attention:MLP (within a flat optimum region that eases size-specific adjustments; Fig. 2 right, §2.1).
  - Result: most mixing work is offloaded to SSMs; a small attention slice is retained for precision (§2.1).

- Mamba‑2 SSM design (what happens inside the SSM and why it’s stable)
  - Mamba‑2 processes a sequence recurrently with a hidden state `h`:
    - Update equations (§2.2, Eq. 6): `h_{t+1} = A_t h_t + B_t dt_t x_t`; `y_t = C_t^T h_t + D x_t`.
    - Equivalent view as a causal “attention-like” matrix over time (§2.2, Eq. 7).
    - Inputs are projected and gated with SiLU and a depthwise causal 1D convolution before recurrence (§2.2, Eqs. 8–9).
  - Key ablations and choices (§2.2):
    - State dimension `d_state` vs groups `n_g`: at fixed parameter budget, increasing `d_state` improves accuracy far more than increasing groups; throughput peaks near `d_state=16`, but for long contexts they choose `(n_g, d_state)=(1,256)` (Fig. 3). For 34B with TP=4, `n_g=2` for divisibility (§2.2).
    - SSM head dimension `d_head`: larger heads (≥64) give better accuracy and efficiency (Fig. 4a).
    - Depthwise conv kernel: kernel size 4 minimizes loss; both smaller and larger are worse (Fig. 4b).
    - SSD scan chunk size `cs`: throughput plateaus at 128–256; they fix `cs=256` (§2.2).
    - Cross-document leakage: they reset the SSM hidden state exactly at document boundaries by injecting a large negative bias into `Ā_t` so `exp(-80)≈0` (§2.2 “Hidden State Resetting”), eliminating contamination without extra compute.
  - Training stability: Early runs showed loss spikes originating in the SSM path when width (many heads) pushes the `dt` (“time‑step”) activation too high. Clipping or attenuating positive `dt` removesspikes; they adopt a softer attenuation as a µP multiplier in the forward pass (§3.2.1).

- Positional encoding choice (RoPE base frequency)
  - They raise the RoPE base `b` to 10^11 (very large). Sweeps (Fig. 5a) show that too-small `b` hurts loss, especially when the training sequence length increases; large `b` flattens the loss curve and avoids needing “NTK-aware” tricks when extending context (§2.3.1). With large `b`, many very-low-frequency dimensions remain unused at train time, making later context scaling simple.

- Depth vs. width at fixed parameters (why the “1.5B‑Deep” exists)
  - Under a 1.5B budget, they sweep model depth/width (Fig. 5b). Deeper shapes consistently deliver better pretraining loss, albeit with ~25–30% lower throughput. This motivates releasing both a “balanced” 1.5B and a deeper 1.5B‑Deep (§2.3.2).

- Tokenizer (how text is split to tokens and why it matters)
  - BPE tokenizers with 32k/65k/131k/262k vocabularies are trained on 121+ languages; vocab size scales with model size (Table 5; §2.4.2).
  - Data scaling: more corpus isn’t always better; optimal size depends on vocab (Table 2, §2.4.1).
  - Splitting rules: enabling both punctuation and digit splitting improves code/math and multilingual segmentation—even if fertility (compression) worsens (Table 4; Fig. 6–7; §2.4.1).
  - Inject domain tokens: adding common LaTeX commands to the vocabulary improves math benchmarks during training (§2.4.1, Fig. 8).
  - They reserve 1,024 special tokens for downstream customization (§2.4.2).

- Data and curriculum (how data is organized through training)
  - Sources include filtered web (FineWeb-derived), curated multilingual corpora for 18 languages, large code (file- and repo-level with HQ splits), math corpora, and synthetic/rewritten data (§3.1.1).
  - Deterministic dataloader reads sources sequentially, enabling reproducible runs and flexible on-the-fly mixture changes and multi-epoch reuse (§3.1.2 “Deterministic data loading”).
  - Anti‑curriculum: mix “hard” and “easy” data from the start rather than staging it later; this outperformed curriculum schedules in their setting (§3.1.2 “Data organization and scheduling”).
  - Web fraction is surprisingly low in the final mixtures for large models (e.g., 34B ends at ~15% raw web; Table 6), with rewritten data (web/code/math/curated rewrites) dominating (up to ~52% at 34B end-of-training; Table 6).
  - Memorization window: by checkpoint rollback tests and loss monitoring (Fig. 9), they argue high-quality data can be reused multiple times without overfitting at their scale (§3.1.2).

- Training dynamics and schedules (how to set LR/WD robustly)
  - Parameter norms empirically scale with `sqrt(η/λ)` across layers (Fig. 10), which they interpret via a toy stochastic dynamics model (§3.2.2, Eqs. 10–11 & Appendix B).
  - They define two composite controls:
    - Effective learning rate `η_eff = sqrt(η λ)` governs noise level and loss reduction at LR decay (Fig. 11 right; Eq. 12).
    - Effective weight decay `λ_eff = sqrt(λ/η)` governs parameter norms (Fig. 10 right; Eq. 12).
  - Because `η_eff` and `λ_eff` are orthogonal in log‑space (Eq. 13), grid sweeps can vary noise and norms independently (§3.2.2). 
  - Effective Power Scheduler (EPS): instead of standard Power Scheduler that keeps WD constant and scales LR as `t^{-1/2}` on stable stages, they propose keeping `λ_eff` constant and decaying both LR and WD as `t^{-1/4}` so norms stay stable while `η_eff` decays (Eq. 15). EPS improved convergence in their tests (§3.2.2).

- µP with forward multipliers (how they transfer hyperparameters across sizes)
  - Maximal update parametrization (µP) predicts how LR/WD/initialization/forward multipliers should scale with width for consistent feature learning. Instead of using different LR/WD per size, Falcon‑H1 fixes optimizer hyperparameters across sizes and moves µP scaling into explicit forward multipliers placed throughout the model (Table 7; §3.2.3).
  - They then “tune µP multipliers” at a base shape (L=66, d=1280) via stagewise micro‑sweeps (Appendix C) across 35 multipliers covering forward paths, matrix ELR/EWD groups, and vector LR multipliers (Table 8, Fig. 12). Sensitivity analysis shows ELR multipliers have highest impact, followed by forward multipliers (§3.2.3).
  - Practical bonuses: same LR/WD for all sizes; stable transfers; fewer optimizer parameter groups.

- Rampup, batch scaling, warmup
  - LR scales with sqrt(batch) when batch changes (Eq. 19), which preserves learning better than no scaling (§3.2.4).
  - Batch rampup with LR batch‑scaling may look worse early but wins later, likely because it guides optimization to a better region (Fig. 13).
  - Short LR warmup (~0.1 GT) has a long‑lasting positive impact on loss (Fig. 13, bottom-right).

- Distributed training and inference
  - Training infrastructure (“Mambatron”) supports 5D parallelism: Data, Tensor, Pipeline, Context (long sequences), and a new Mixer Parallelism (MP) (§3.3, Table 9).
  - Mixer Parallelism: split the TP world so attention and SSM compute concurrently per layer; interleaving layers across groups balances load (Fig. 14). Interleaved MP improves training throughput by 1.43× over no MP (Table 10).
  - MP for inference: helps when batch and generation lengths are small; benefits reduce or reverse at large batches/long generations (Fig. 15; §3.3.2).
  - Context Parallelism: attention uses RingAttention (K/V circulate around a ring), SSM uses chunked state passing; both keep per‑GPU memory O(chunk-length) (§3.3.3).

- Post‑training
  - SFT: 6 GT total—3 GT at 16K, then 3 GT at 128K with LR fixed at η/8; data weighted toward high-quality instruction corpora (e.g., Tulu3). No WD during SFT (§4.2, Table 11).
  - DPO: standard DPO loss with β=5; best stopping point is ~1 epoch rather than the full 2‑epoch LR schedule (§4.3, Table 12).

## 4. Key Insights and Innovations
- Parallel hybrid mixer with independent channel allocation (fundamental)
  - Most prior hybrids forced equal attention/SSM dimensions or ran them strictly in series. Falcon‑H1 concatenates parallel outputs and tunes each mixer’s width independently (§2, Fig. 1), allowing a small attention sliver and a large SSM core. Systematic sweeps show the best performance with the minimal attention fraction (1/8) and SA_M block order (Fig. 2). This design drives both accuracy and efficiency, especially at long contexts.

- SSM-specific, long‑context‑oriented design and stability (fundamental)
  - Carefully chosen Mamba‑2 hyperparameters—large `d_state`, head sizes ≥64, conv kernel=4, chunk size=256—plus exact hidden‑state reset at doc boundaries and `dt` attenuation remove instability and preserve long‑range memory (§2.2). These are the practical recipes missing from many SSM reports.

- Effective LR/WD decomposition and schedule (conceptual + practical)
  - The identification of `η_eff = sqrt(ηλ)` as the main “noise” control and `λ_eff = sqrt(λ/η)` as the main “norm” control is supported by loss and norm measurements (Fig. 10–11; Eq. 12). Building on that, the Effective Power Scheduler (Eq. 15) decays LR and WD together to keep parameter norms stable while reducing noise (§3.2.2). This provides a simple, transferable way to set LR/WD across scales.

- µP with tuned forward multipliers (practical transfer)
  - Shifting µP scaling into forward multipliers and tuning a minimal, architecture‑aware set (Table 7–8) lets all model sizes share the same optimizer hyperparameters. Sensitivity diagnostics (Fig. 12) clarify what matters most (matrix ELR multipliers), enabling robust, compute‑efficient HP transfer (§3.2.3, Appendix C).

- Tokenizer provenances that matter for downstream tasks (applied but important)
  - Two decisions—(1) enabling both digit and punctuation splitting and (2) injecting frequent LaTeX tokens—consistently improved code and math performance, despite modest changes in compression metrics (Fig. 6–8; Table 3–4; §2.4.1). This shifts tokenizer tuning away from proxy metrics and toward downstream outcomes.

- Data mixture that deemphasizes raw web and embraces rewriting (applied)
  - The final large‑model mixtures use only ~12–15% raw web and >50% rewrites (Table 6), with an anti‑curriculum schedule and deterministic loader enabling multi‑epoch reuse without observed memorization issues (Fig. 9; §3.1.2). This is a distinctive data philosophy aimed at “knowledge density,” not just token count.

- Mixer Parallelism for hybrid models (systems innovation)
  - Interleaving attention and SSM across TP groups allows genuine concurrency within a layer. It yields 1.43× training throughput versus a non‑MP baseline (Table 10) and can speed up inference in low-latency regimes (Fig. 15; §3.3.2).

## 5. Experimental Analysis
- Evaluation methodology
  - Frameworks: lm‑evaluation‑harness, evalchemy, evalplus, HELMET (§5). For math in evalchemy, they use 16 generation turns and a fixed system prompt, and post‑check with Math‑Verify (§5).
  - Standardization: same Docker environment; “thinking mode” disabled for Qwen3 so inference is comparable (§5).
  - Benchmarks
    - General: BBH, ARC‑C, HellaSwag, Winogrande, MMLU.
    - Math: GSM8K, MATH (lvl5 or 500), AMC‑23, AIME‑24/25.
    - Science: GPQA (+Diamond), MMLU‑Pro, MMLU‑STEM.
    - Code: HumanEval(+), MBPP(+), LiveCodeBench, CRUXEval.
    - Multilingual: Multi‑HellaSwag, Multi‑MMLU, MGSM (6 languages).
    - Long‑context: HELMET LongQA, RAG, Recall (8k→131k).
    - Efficiency: vLLM throughput vs Qwen2.5‑32B (TP=2; H100 GPUs; §5.3).

- Main quantitative results (selected highlights; bold shows standout points)
  - Base models
    - 0.5B (Table 14): 
      > GSM8K 60.20 vs Qwen3‑0.6B 50.04; MATH‑lvl5 15.18 vs 9.29; HumanEval+ 31.10 vs 27.44.  
      This 0.5B model “leads on every Math, Science, and Code benchmark” among sub‑1B baselines (§5.1).
    - 1.5B‑Deep vs peers (Table 15):
      > MMLU 66.29; MMLU‑Pro 41.07 (vs Qwen3‑1.7B 33.81); MBPP 70.90; GSM8K 68.69.  
      Deeper 1.5B consistently outperforms the shallower 1.5B and many 7B-class results on several tasks (§5.1).
    - 3B (Table 16):  
      > MATH‑lvl5 25.83 (SOTA at this scale); MGSM 64.00.  
      Despite only 2.5T tokens (vs Qwen3‑4B’s 36T), math and multilingual math are strong.
    - 7B (Table 17):
      > MMLU 77.38 (best); MATH‑lvl5 34.67 (best); MBPP 78.57 (best); MGSM 74.53 (best).  
      Clearly competitive across reasoning, science, code, and multilingual.
    - 34B (Table 18):
      > BBH 69.36 (best vs 32–72B and 70B baselines); MATH‑lvl5 40.71 (best); GPQA 42.70 (best); HumanEval 70.12 (best); MGSM 82.40 (top‑tier).  
      On general knowledge (MMLU/HellaSwag), 70B models sometimes edge out, but 34B is often second-best.
  - Instruct models
    - 0.5B (Table 20):
      > GSM8K 68.39 (best); MATH‑500 58.40 (best); HumanEval 51.83 (best); IFEval 72.07 (best).  
      Emphasizes robust reasoning and instruction following at tiny scale.
    - 1.5B‑Deep (Table 21):
      > GSM8K 82.34; MATH‑500 77.80; HumanEval 73.78; GPQA_Diamond 40.57; IFEval 83.50.  
      Outperforms same‑size and many larger baselines broadly; 1.5B (shallow) is usually second-best.
    - 3B (Table 22): 
      > MMLU 68.30 (top); GPQA_Diamond 38.72; MBPP 79.63; MGSM 63.90.  
      Very balanced across domains.
    - 7B (Table 23):
      > HumanEval 86.59 (best); GPQA_Diamond 56.90 (best); MMLU‑Pro 51.75 (best); strong across multilingual tasks (e.g., Multi‑MMLU 67.83).  
      Qwen3‑8B does better on some math and preference tests (AIME, Alpaca‑Eval).
    - 34B vs 32–72B and 70B (Table 24):
      > Science suite leadership: GPQA 41.53 (best), GPQA_Diamond 49.66 (best), MMLU‑STEM 83.57 (best).  
      > MTBench 9.20 (best conversational quality).  
      For Math/Code, 70B‑class models sometimes lead; for general reasoning and science, Falcon‑H1‑34B is extremely competitive.
  - Long‑context (HELMET; Table 25)
    - RAG: 
      > 131k tokens — 62.21 (best; Qwen3‑32B 57.08; Llama‑3.3‑70B 55.38; Qwen2.5‑72B 42.33).  
      Strong evidence Falcon‑H1’s hybrid design and training are well‑suited to document‑augmented tasks at extreme lengths.
    - Recall and longQA:
      > At 131k, Recall 56.63 (lower than Qwen3‑32B 86.13 and Llama‑3.3‑70B 82.19), and longQA 33.81 (vs Qwen3‑32B 53.52).  
      Authors attribute this to long‑context data composition; it flags an avenue for further data work (§5.2).
  - Efficiency (Fig. 16; §5.3)
    - Prefill throughput: at long inputs (up to 262k), Falcon‑H1‑34B is up to 4× faster than Qwen2.5‑32B.
    - Generation throughput: at long outputs (up to 262k), up to 8× faster.
    - At short contexts, an optimized Transformer can be slightly faster—SSM kernels are less mature.

- Ablations and robustness
  - Channel allocation and block order sweeps (Fig. 2) substantiate the SA_M choice and minimal attention fraction.
  - SSM ablations (Fig. 3–4) justify `d_state`, head dims, conv kernel, and chunk size.
  - Instability investigations (Fig. 13; §3.2.1) identify the `dt` activation as the culprit and show attenuation solves spikes.
  - ELR/EWD studies (Fig. 10–11) support the composite-control view and motivate the EPS schedule.
  - Tokenizer and data studies (Table 2–4; Fig. 6–8; Table 6; Fig. 9) demonstrate downstream gains from punctuation/digit splitting, LaTeX tokens, and rewritten/anti‑curriculum data.

- Do the experiments support the claims?
  - Yes, for three central claims:
    - Hybrid-head with tuned channel allocation achieves strong accuracy: across scales, Falcon‑H1 holds or leads on many reasoning, science, code, and multilingual tasks (Tables 14–18, 20–24).
    - Long‑context capability and efficiency: HELMET RAG at 131k is best; detailed systems design (CP/MP) plus SSM yields large throughput gains (Table 25; Fig. 16; §3.3).
    - Parameter/training efficiency: 34B rivals or beats 70B models in several categories; 1.5B‑Deep rivals many 7–10B models (summaries in §5.1–§5.2).  
  - Mixed results on a few fronts (e.g., HellaSwag at some scales, HELMET longQA/Recall at 131k) are acknowledged, with plausible causes (data composition, maturity of SSM kernels).

## 6. Limitations and Trade-offs
- Attention is small by design
  - Minimal attention width (1/8 fraction) is optimal in their sweeps (Fig. 2), but edge cases might benefit from more attention (e.g., tasks dominated by global token‑token interactions). The flat optimum near 2:1:5 SSM:Attn:MLP allows some flexibility (§2.1), but the reported settings skew toward SSM.
- Long‑context QA and recall at extreme lengths
  - While RAG excels at 131k, pure recall/QA is weaker than some Transformers (Table 25). The authors attribute this to long‑context data composition rather than architecture, but it remains a practical trade‑off to address in future mixtures.
- Efficiency depends on SSM kernels
  - At short contexts, Qwen2.5‑32B has marginally higher throughput; SSM kernels are newer and less optimized in inference stacks (§5.3). Gains emerge strongly at very long contexts.
- Complexity of training recipe
  - The full recipe includes ELR/EWD‑aware schedules, tuned µP multipliers, SSM‑specific resets and `dt` attenuation, deterministic loading, anti‑curriculum mixing, and MP/CP strategies. Replicating all parts could be nontrivial.
- Data composition and potential biases
  - Heavy reliance on rewritten data (>50% for large models; Table 6) can shape model style and knowledge coverage in ways different from raw web. Memorization analysis (Fig. 9) is suggestive but not exhaustive.
- Multilingual scope
  - Multilingual coverage is strong (18 languages), but the smallest 0.5B is English‑only (§1, Table 1), and performance varies by language (Appendix D.1–D.2). True “100+ language” scaling is stated as feasible, not demonstrated (§1).
- Theory is mostly empirical
  - ELR/EWD motivation includes a toy model (Appendix B), but formal proofs that strictly predict optimal exponents or schedules aren’t provided; results are empirical and could be context‑dependent (§3.2.2).

## 7. Implications and Future Directions
- How it changes the landscape
  - Establishes a practical, strong-performing blueprint for attention–SSM hybrids: parallel mixing with small attention, carefully tuned SSM hyperparameters, and stability methods. It shows hybrids can be parameter‑ and compute‑efficient without sacrificing accuracy, and can excel in long‑context settings (RAG at 131k) where Transformers struggle.
- What it enables next
  - Research:
    - Dynamic channel allocation per layer or per task, or learned attention/SSM ratios.
    - Extending ELR/EWD analysis into formal optimization theory; exploring EPS variants.
    - Better long‑context data recipes to push longQA/Recall at 256k+.
    - Kernel engineering for SSMs (e.g., fused ops, caching) to remove short‑context throughput deficits.
    - Applying large‑`b` RoPE strategy to pure Transformers to test generality (§2.3.1).
  - Practical applications:
    - Enterprise RAG systems over very long corpora (legal, scientific, codebases) with strong speed/quality at 32k–256k contexts.
    - Edge or resource‑constrained deployments: 0.5B and 1.5B‑Deep deliver multi‑domain competence comparable to older 7–10B models (§1, §5.1–§5.2).
    - Multilingual assistants covering 18 languages with good math/code capabilities and 256k context.
    - Code reasoning agents benefiting from tokenizer and repo‑level pretraining (file+repo HQ splits; §3.1.1).

> Summary of impact: Falcon‑H1 is not just a new model family; it is an end‑to‑end methodology—architecture, tokenizer, data, training dynamics, and systems—that shows hybrid attention‑SSM LLMs can be smaller, faster at long contexts, and still competitive with much larger baselines. The open‑weight release across many sizes and the detailed ablations make it a strong foundation for both research and real‑world deployments.
