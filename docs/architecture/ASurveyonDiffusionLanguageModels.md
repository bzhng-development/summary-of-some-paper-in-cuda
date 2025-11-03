# A Survey on Diffusion Language Models

**ArXiv:** [2508.10875](https://arxiv.org/abs/2508.10875)

## 🎯 Pitch

This comprehensive survey maps out the emerging landscape of Diffusion Language Models (DLMs), which generate text by refining entire sequences in parallel through iterative denoising—breaking through the sequential bottlenecks of standard autoregressive models. By offering an in-depth taxonomy, synthesizing state-of-the-art techniques, and benchmarking DLMs’ efficiency and quality, the paper highlights DLMs’ potential to revolutionize language generation with faster inference, richer context handling, and unified multimodal capabilities, setting the agenda for both research and real-world applications.

---

## 1. Executive Summary (2-3 sentences)
This survey systematizes the fast‑growing field of Diffusion Language Models (DLMs)—models that generate text by iteratively denoising masked or noised sequences in parallel rather than predicting tokens one‑by‑one. It builds a unified taxonomy (continuous, discrete, hybrid), formalizes training and inference mechanics with concrete objectives and algorithms, consolidates evidence that DLMs can match comparable autoregressive (AR) models while offering substantial speedups, and maps key challenges and research directions (Sections 2–8; Figs. 1–6; Tables 1–2).

## 2. Context and Motivation
- Problem addressed
  - Language generation is dominated by AR models that emit tokens sequentially. This creates a fundamental latency bottleneck and prevents easy use of bidirectional context during generation. DLMs aim to break this bottleneck by generating many tokens in parallel via iterative denoising (Section 1; Fig. 4).
- Why it matters
  - Practical: Lower latency and higher throughput are critical for interactive systems and large‑scale serving. DLMs promise parallel generation, improved controllability, and bidirectional conditioning that can benefit editing, infilling, and multimodal tasks (Section 1: “Parallel Generation… Bidirectional Context… Controllability… Unified Modeling Across Modalities”; bullets on p. 3).
  - Scientific: Diffusion has transformed vision; this work consolidates how its principles translate to discrete language, clarifying where theory and engineering solutions already exist and where gaps remain (Sections 2–5).
- Prior approaches and shortcomings
  - AR LMs: Scalable and strong but inherently sequential (Eqs. 3–4; Section 2.1.2). Even multi‑token prediction partially retains sequential dependencies (Section 2.1.2).
  - Masked Language Models (MLMs): Excellent for understanding but not designed for open‑ended generation (Eq. 1; Section 2.1.1).
  - Early DLMs: Proofs of concept in continuous embeddings and discrete token spaces, but lagged AR on quality and lacked standardized training/inference tooling (Sections 2.2–2.3; Fig. 1 timeline).
- Positioning
  - The survey provides a complete stack view—formalisms (Eqs. 6–13), model families (Table 1), inference accelerators (Fig. 5), post‑training methods (Table 2), multimodal integrations (Section 5), performance synthesis (Fig. 6), and challenges with concrete examples (Fig. 7)—to establish DLMs as a viable, distinct paradigm.

## 3. Technical Approach
This is a survey; the “approach” is a structured framework that specifies model paradigms, objectives, and inference procedures. It explains mechanics with equations and concrete decoding schedules, then aggregates post‑training and acceleration methods.

- Paradigms and their mechanics (Section 2; Fig. 4)
  - Autoregressive baseline (context for comparison)
    - Factorizes sequence likelihood into left‑to‑right conditionals (Eqs. 3–4).
    - Strength: quality and simplicity of sampling; Weakness: sequential latency.
  - Continuous DLMs (Section 2.2)
    - Idea: Diffuse in a continuous space—either embeddings or logits—then round to tokens.
    - Forward (noising) process: sample a Markov chain q(x1:T|x0) with Gaussian steps; many methods use the closed‑form reparameterization xt = αt x0 + bt ε (Eqs. 6–8).
    - Reverse (denoising) process: learn fθ(xt, t) to predict a target (clean data, noise, or velocity) via a simple squared error objective (Eq. 9).
    - Mapping back to text: nearest‑neighbor or a decoder head converts denoised embeddings/logits to discrete tokens.
    - Variants: SED performs diffusion on a fixed embedding space with self‑conditioning; TESS/TESS‑2 diffuse over a k‑logit simplex to avoid embedding collapse (Section 2.2).
  - Discrete DLMs (Section 2.3)
    - Idea: Diffuse directly in token space using categorical transitions (D3PM; absorbing [MASK] state).
    - Forward (noising) process: apply transition matrices Qt; with an absorbing mask, tokens either stay or become `MASK`. Marginal: q(xt|x0) = Cat(xt; x0 Q̄t) (Section 2.3).
    - Reverse process: learn to reconstruct clean tokens from a partially masked sequence.
    - LLaDA‑style masked diffusion objective: cross‑entropy on masked positions only, weighted by 1/t to emphasize early steps (Eq. 10). Inference starts from an all‑`MASK` sequence; iteratively unmask high‑confidence positions and keep remasking low‑confidence ones (Section 2.3).
  - Hybrid AR‑Diffusion (Section 2.4)
    - Blockwise semi‑autoregression: Generate blocks autoregressively while denoising tokens inside each block in parallel (BD3‑LM). Objective conditions each block on prior blocks while learning to denoise within‑block states (Eq. 11; Fig. 4 “Block‑DLM”).
    - Architectural or decoding hybrids: diffusion as a drafter paired with AR validation (SpecDiff in Section 2.4; also see Section 4.1).
- Training and post‑training (Section 3)
  - Pre‑training: Either from scratch (e.g., `LLaDA‑8B`) or initializing from AR LMs (`DiffuLLaMA`, `Dream`) or image diffusion models (D‑DiT, Muddit) to reuse capabilities (Section 3.1; Table 1).
  - Supervised fine‑tuning (SFT): For masked DLMs, leave prompts visible and selectively mask responses to train conditional generation; continuous DLMs corrupt only response segments (Section 3.1).
  - Reasoning/post‑training with RL and preferences (Section 3.2)
    - Challenge: No tractable AR‑style sequence likelihood; per‑step denoising is non‑factorized.
    - Workarounds:
      - Score‑entropy policy gradients (SEPO): a low‑variance RL objective over diffusion steps using importance sampling (Eq. 12).
      - Practical log‑prob approximations (d1/diffu‑GRPO): mean‑field decomposition + one‑pass per‑token probabilities on masked variants.
      - Masking strategies during RL to expose different denoising stages (UniGRPO).
      - Preference optimization adapted with variance reduction (VRPO in `LLaDA 1.5`).
    - Non‑RL reasoning: DoT reframes chain‑of‑thought as parallel “thought refinement” along the denoising trajectory.
- Inference and acceleration (Section 4; Fig. 5)
  - Parallel decoding: accept multiple tokens per step by confidence thresholds or drafts (Section 4.1).
  - Unmasking/remasking: selectively fix confident tokens and re‑open uncertain ones; new samplers permit revising already decoded tokens (Section 4.2).
  - Guidance: classifier‑free guidance (CFG) combines conditional and unconditional scores to steer outputs: sguided = suncond + λ(scond − suncond) (Eq. 13; Section 4.3).
  - Caching:
    - KV‑cache for semi‑autoregressive or delayed caching settings (BD3‑LM, Fast‑dLLM DualCache, dKV‑Cache; Section 4.4).
    - Feature caching of intermediate activations across diffusion steps (dLLM‑Cache, FreeCache).
  - Step distillation: train a small “student” to emulate multi‑step denoising with few or even one step (Di4C, DLM‑One; Section 4.4).
- Multimodal/unified designs (Section 5)
  - VLMs with vision encoders and projectors (LLaDA‑V, LaViDa, Dimple).
  - Unified token spaces with discrete image tokens (MMaDA, UniDisc, Muddit).
  - Dual‑branch continuous diffusion (D‑DiT) for text and images trained jointly (Section 5).

## 4. Key Insights and Innovations
- A unified, end‑to‑end taxonomy of DLMs with working equations and decoding procedures
  - What’s new: The survey aligns continuous, discrete, and hybrid methods within one formal frame (Eqs. 6–13; Fig. 4), and ties them to concrete training/inference recipes (Figs. 5; Sections 3–4).
  - Why it matters: Readers can directly see how to instantiate, train, and deploy each class, and how to compose hybrids (blockwise or speculative).
- Practical inference playbook for DLMs
  - Novel synthesis: Parallel decoding strategies, adaptive unmasking/remasking, CFG, KV/feature caches, and step distillation are laid out as modular tools (Section 4; Fig. 5).
  - Significance: These make DLMs competitive in latency while preserving or improving quality (quantified speedups below).
- Post‑training for reasoning in non‑AR settings
  - Distinct challenge identified: lack of tractable AR‑style likelihood.
  - Concrete solutions compared (Table 2; Section 3.2): mean‑field likelihood approximations (d1), score‑entropy RL (SEPO), structured noising during RL (UniGRPO), mask coupling to reduce variance (coupled‑GRPO), and preference optimization with variance reduction (VRPO).
- Multimodal diffusion as a first‑class alternative to AR VLMs
  - Contributions: Multiple recipes for visual grounding (projectors vs unified discrete tokens) and joint training objectives (Sections 5.1–5.5).
  - Importance: DLMs’ bidirectional context and iterative refinement map naturally to infilling/editing across modalities.

## 5. Experimental Analysis
While this is a survey, it compiles quantitative evidence (Sections 4–6; Fig. 6) and concrete acceleration numbers. Highlights are grounded with specific citations:

- Evaluation scope (Section 6; Fig. 6)
  - Datasets/metrics include PIQA and HellaSwag (language understanding), HumanEval (code), GSM8K (math reasoning), plus multimodal GenEval, MME, MMMU, GQA.
  - Model scales: predominantly <1B–8B parameters for open DLMs; AR baselines of similar sizes are plotted for comparison.
- Main quantitative takeaways (Section 6; Fig. 6)
  - General language understanding: `LLaDA` and peers are “slightly below or on par” with similarly sized AR models (e.g., LLaMA2, Qwen2.5) on PIQA/HellaSwag.
  - Math and science reasoning: DLMs (`LLaDA`, `Dream`) “consistently” outperform comparable AR models on GSM8K, GPQA, MATH.
  - Code: `DiffuCoder` is competitive among open‑source models on HumanEval.
  - Multimodal: `MMaDA` and `LLaDA‑V` often surpass AR‑based VLMs on understanding/generation.
- Inference speedups (Section 4; with citations)
  - Parallel decoding:
    > “Fast‑dLLM … realizes up to 27.6× speed‑ups without compromising quality.” (Section 4.1)
    > “SlowFast … up to 34× acceleration when combined with caching.” (Section 4.1)
    > “SpecDiff … up to 7.2× speed‑ups over vanilla AR generation.” (Section 4.1)
  - Caching:
    > “Fast‑dLLM … DualCache … up to 27× end‑to‑end throughput gains … with <1% accuracy loss.” (Section 4.4)
    > “dKV‑Cache … achieves 2–10× speed‑ups … with negligible quality drop.” (Section 4.4)
    > “dLLM‑Cache … up to 9× end‑to‑end speed‑ups …” and “FreeCache … pushing acceleration … to 34×” (Section 4.4)
  - Step distillation:
    > “DLM‑One … generates an entire sequence in a single forward pass, realising up to 500× acceleration with near‑teacher quality.” (Section 4.4)
- Trend and scale observations
  - Research volume: Fig. 2 shows discrete DLM papers accelerating sharply since 2023 (Section 1).
  - Timeline: Fig. 1 and Table 1 document the move from small continuous DLMs to large discrete and multimodal variants (2024–2025).
- Robustness/failure cases and ablations
  - Section 8 and Fig. 7 diagnose a core failure mode: quality degrades when too many tokens are accepted per step (the “Parallel Decoding Curse”). Fig. 7 gives concrete failure examples at small step counts for `LLaDA` and `MMaDA`.
- Overall assessment
  - The compiled results substantiate that (a) DLMs can be competitive at similar scale, (b) strong speedups are feasible with the right inference stack, and (c) reasoning and multimodal tasks particularly benefit from bidirectional, iterative refinement. Results are necessarily heterogeneous (drawn from many sources), so exact head‑to‑head comparability depends on training data, step counts, and decoding settings (Section 6).

## 6. Limitations and Trade-offs
- Parallelism vs. coherence (Section 8.1; Fig. 7)
  - Assumption/trade‑off: Accepting many tokens per step increases parallelism but can ignore inter‑token dependencies, yielding incoherent sequences (“Parallel Decoding Curse”).
  - Evidence: Fig. 7 shows correct, fluent outputs only when 1–2 tokens are unmasked per step; fewer steps (more parallelism) produce incorrect or garbled text for both `LLaDA` and `MMaDA`.
- Tooling and infrastructure (Section 8.1)
  - DLMs lack mature open‑source training/serving stacks comparable to HuggingFace/vLLM, complicating deployment and making performance uneven across implementations.
- Long sequences and dynamic length (Section 8.1)
  - Computational scaling: full bidirectional attention at every denoising step implies O(N²) per step; steps often scale with N, giving O(N³) total without KV‑cache or blockwise designs.
  - Dynamic stops: even after an `EOS` is predicted, the whole sequence continues to be processed in subsequent steps.
- Scale and data (Section 8.1)
  - Open DLMs are mostly ≤8B parameters, often adapted from AR models or trained on smaller corpora than leading AR LLMs. Closed DLMs (Mercury, Gemini Diffusion) still trail the very largest AR models on many benchmarks.
- Post‑training complexity (Section 3.2; Table 2)
  - RL and preference methods require approximations to sequence likelihood, specialized masking schedules, and variance‑reduction tricks. These add engineering complexity and hyperparameter sensitivity.

## 7. Implications and Future Directions
- Field impact
  - The survey consolidates DLMs as a legitimate alternative to AR generation with clear recipes for making them fast and controllable (Sections 2–4). It also demonstrates credible multimodal unification routes (Section 5).
- Research avenues (Section 8.2)
  - Training efficiency: Increase token utilization (e.g., complementary masking), better noise schedules, and hybrid architectures to approach AR training efficiency.
  - Low‑bit DLMs: Quantization/binarization for memory and latency reductions remain largely unexplored.
  - Compression: Pruning and knowledge distillation (beyond step distillation) tailored to diffusion schedules and masking patterns.
  - Long‑context scaling: Reduce O(N³) behavior via blockwise diffusion with KV‑cache (Fig. 4 “Block‑DLM”), dynamic step allocation, or sparse attention; adapt AR extrapolation tricks (e.g., RoPE‑NTK) to DLMs (Section 2.3 cites LongLLaDA).
  - RLHF for diffusion: Improve likelihood surrogates, credit assignment across steps, and reward modeling that aligns with parallel refinement (Section 3.2).
  - Unified multimodal reasoning: Expand discrete‑token unification (VQ‑VAEs) and dual‑branch continuous diffusion to robustly co‑generate and edit across modalities (Section 5).
  - Agentic DLMs: Leverage parallel draft‑and‑verify (SpecDiff + AR validators), remasking for self‑correction, and guidance for structural constraints (Section 4.3).
- Practical applications (Section 7)
  - Code assistants that benefit from global planning and fast drafts (`DiffuCoder`, Mercury Coder).
  - High‑throughput creative writing/infilling, paraphrasing, and constrained style transfer via guidance and remasking.
  - Scientific design: molecular/protein design using discrete diffusion with reward optimization (DRAKES, DPLM/2, CFP‑GEN).

---

Selected figure/table/equation references for quick lookup:
- Taxonomy and timeline: Fig. 1 (timeline), Fig. 3 (taxonomy), Fig. 2 (trend).
- Core mechanics: Eqs. 3–4 (AR), Eq. 1 (MLM), Eqs. 6–9 (continuous diffusion), discrete forward with absorbing mask (Section 2.3), Eq. 10 (`LLaDA` loss), Eq. 11 (BD3‑LM block objective), Eq. 12 (SEPO RL objective), Eq. 13 (CFG).
- Inference toolbox: Fig. 4 (training/inference overview), Fig. 5 (parallel decoding, remasking, guidance, caches, distillation).
- Models and post‑training at a glance: Table 1 (models/size/objectives), Table 2 (post‑training algorithms).
- Performance synthesis: Fig. 6 (eight benchmarks).
- Failure mode under high parallelism: Fig. 7 (qualitative outputs across step counts).
