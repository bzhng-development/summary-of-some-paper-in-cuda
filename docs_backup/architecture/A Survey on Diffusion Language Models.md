# A Survey on Diffusion Language Models

**ArXiv:** [2508.10875](https://arxiv.org/abs/2508.10875)
**Authors:** Tianyi Li, Mingda Chen, Bowei Guo, Zhiqiang Shen
**Institutions:** not specified on arXiv

## 🎯 Pitch

This paper systematizes the field of Diffusion Language Models (DLMs), introducing a taxonomy and performance benchmarks that highlight their potential to surpass autoregressive models in speed and multimodal capabilities. By enabling parallel text generation and bidirectional context, DLMs promise significant real-world applications in latency-sensitive tasks, bridging gaps in open-ended and structured generation while uncovering new avenues for robust reasoning and multimodal integration.

---

## 1. Executive Summary
This survey systematizes the fast‑growing field of Diffusion Language Models (DLMs)—models that generate text by iteratively denoising masked or noisy sequences in parallel rather than predicting tokens one by one. It builds a comprehensive taxonomy (continuous, discrete, hybrid, and multimodal variants), explains core training and inference mechanics, consolidates performance evidence across benchmarks, and analyzes practical accelerations and limitations (Sections 2–8; Figs. 1–6, Tables 1–2).

## 2. Context and Motivation
- Problem addressed
  - Large language models based on autoregression (AR) generate text sequentially (token-by-token), which creates an inference‑time bottleneck and limits parallelism (Section 1). DLMs promise parallel generation and bidirectional context but the literature is fragmented across formulations (continuous vs. discrete), training recipes, post‑training for reasoning, and inference accelerations.
- Why it matters
  - Real‑world latency: Parallel denoising can accept or revise multiple tokens per step, potentially yielding much higher throughput than AR decoding (Section 1; Section 4).
  - New capabilities: Bidirectional conditioning supports infilling, structured editing, and unified multimodal generation/understanding in a single framework (Sections 2.2–2.4, 5).
- Prior approaches and gaps
  - AR LLMs excel in quality but are inherently sequential (Section 2.1.2; Eq. (3)–(4)). Masked Language Models (BERT‑style) capture bidirectional context but are not designed for open‑ended generation (Section 2.1.1; Eq. (1)).
  - Early DLMs were small or image‑centric; recent works scale to 7–8B parameters and extend to multimodal tasks (Fig. 1; Table 1). However, there was no single, up‑to‑date, end‑to‑end synthesis of principles, models, training/post‑training, inference, and applications.
- Positioning relative to existing work
  - The survey offers: (i) a unified taxonomy spanning continuous/discrete/hybrid paradigms (Fig. 3), (ii) detailed mechanisms for training and inference—including caching, parallel decoding, and guidance (Section 4; Fig. 5), (iii) a consolidated performance view across tasks (Section 6; Fig. 6), and (iv) a critical assessment of limitations and future directions (Section 8).

## 3. Technical Approach
This section explains “how DLMs work,” organized from foundational formulations to practical training and inference. Citations refer to where each mechanism is defined or exemplified.

- Recap of modern language modeling paradigms (Section 2.1)
  - Masked Language Modeling (MLM): predict randomly masked tokens using surrounding context (Eq. (1)); useful for understanding but not designed for freeform generation.
  - Autoregressive (AR): factorize sequence probability left‑to‑right (Eq. (3)–(4)); great for generation but sequential at inference.
  - Permutation LM (XLNet): trains on random factorization orders (Eq. (5)); captures bidirectional context yet retains AR decoding.

- Continuous-space DLMs (Section 2.2; Eqs. (6)–(9))
  - Idea in plain language: map tokens to continuous vectors (“embeddings”), repeatedly “add noise” to training data, then learn to “remove noise” step‑by‑step; at test time, start from noise and denoise back to a clean embedding sequence; finally “round” embeddings to tokens.
  - Forward (noising) process: a Markov chain gradually corrupts clean embeddings x0 into xt (Eq. (6)–(8)); αt, bt control how much of x0 vs. Gaussian noise appears at time t.
  - Reverse (denoising) process: a Transformer fθ(xt, t) predicts a target (e.g., noise or clean data) and is trained by a simple regression loss (Eq. (9)).
  - Decoding to tokens: after denoising, map the final embeddings back to words by nearest neighbor or a classifier head.
  - Representative mechanisms
    - Classifier-free guidance for controllable generation (adopted from image diffusion; Section 4.3; Eq. (13)).
    - Continuous logit‑space diffusion (TESS/TESS‑2) that diffuses over probability simplices rather than embeddings (Section 2.2).

- Discrete-space DLMs (Section 2.3; Eq. (10))
  - Idea in plain language: operate directly on tokens. The forward process replaces some tokens with a special `[MASK]` symbol; the model learns to recover original tokens from partially masked sequences.
  - Training objective: compute cross‑entropy only on positions that are masked at time t (Eq. (10)); t is sampled, and xt is produced by randomly masking x0.
  - Inference by iterative mask‑predict (Fig. 4 middle-right; Section 2.3; Section 4.2)
    - Start with a fully masked response of desired length.
    - At each step: predict all positions; “unmask” high‑confidence tokens; “remask” low‑confidence or uncertain spans; repeat until no masks remain.
  - Key design levers: noise schedules (how many positions are masked at each step), unmask/remask policies (thresholds, confidence ranking), and acceptance rules (Section 4.1–4.2).

- Hybrid AR–Diffusion (block‑wise) models (Section 2.4; Eq. (11); Fig. 4 bottom)
  - Idea: generate “blocks” of tokens autoregressively to preserve long‑range dependency, but fill tokens within each block in parallel via diffusion steps.
  - Objective (BD3‑LM, Eq. (11)): sum denoising losses over blocks b=1..B, conditioning each block on previous blocks x< b while denoising masked positions inside the current block xb.
  - Benefit: enables effective Key–Value (KV) caching for finished blocks and variable‑length outputs while keeping intra‑block parallelism (Sections 2.4, 4.4).

- Post‑training to elicit reasoning (Section 3.2; Tables 2)
  - Challenge: AR RL methods use exact sequence log‑likelihood; DLM likelihood is intractable because generation is iterative and non‑factorized.
  - Solutions summarized and mechanized:
    - Diffusion‑of‑Thought (DoT): fine‑tune on reasoning traces but refine them in parallel through denoising; uses scheduled/coupled sampling to expose model to its own mistakes (Section 3.2.1).
    - DCoLT: treat each denoising step as a “latent thinking” action and optimize the whole trajectory by outcome‑based RL; adds an Unmasking Policy Module to decide reveal order (Section 3.2.1).
    - SEPO: derive stable policy‑gradient updates by importance‑sampling the “score entropy,” enabling PPO/GRPO‑like updates for discrete diffusion (objective in Eq. (12); Section 3.2.2).
    - diffu‑GRPO (d1) and coupled‑GRPO (DiffuCoder): estimate sequence/per‑token log‑probabilities via masked forward passes; use complementary masking to reduce variance (Section 3.2.2).
    - VRPO (LLaDA 1.5): variance‑reduced preference optimization for DPO‑style training in DLMs using optimal sampling across timesteps and antithetic sampling against a reference policy (Section 3.2.3).

- Inference accelerations and controls (Section 4; Fig. 5)
  - Parallel decoding: accept many tokens per step based on confidence or auxiliary checks; e.g., Fast‑dLLM thresholds (Section 4.1).
  - Unmasking/remasking: revisit previously accepted tokens to improve quality (ReMDM), or schedule slow‑then‑fast acceptance (Section 4.2).
  - Guidance: classifier‑free guidance (CFG) combines conditioned and unconditioned scores to push outputs toward a prompt with tunable strength λ (Eq. (13); Section 4.3).
  - Caches and step distillation (Section 4.4):
    - KV caches adapted for semi‑AR/block schedules or with delayed commits (Fast‑dLLM DualCache; dKV‑Cache).
    - Feature caches reuse stable intermediate activations across steps (dLLM‑Cache; FreeCache).
    - Step distillation collapses many denoising steps into a few or even one (Di4C for discrete; DLM‑One reports one‑step continuous generation).

- Multimodal DLMs (Section 5)
  - Vision‑conditioned DLMs: plug a vision encoder and project features to the text space (LLaDA‑V, LaViDa, Dimple).
  - Unified token spaces: tokenize images as discrete codes (VQ‑VAE) and train a single diffusion Transformer for both text and image inputs/outputs (MMaDA, UniDisc, Muddit; Section 5).

## 4. Key Insights and Innovations
- A unified taxonomy and timeline that clarifies the DLM landscape
  - Fig. 3 organizes paradigms (continuous, discrete, hybrid), training/post‑training, inference, and applications. Fig. 1 and Fig. 2 trace how early continuous models gave way to large discrete and multimodal DLMs. This consolidation is significant for orienting researchers in a rapidly evolving area.

- Mechanistic, side‑by‑side depiction of training/inference across paradigms
  - Fig. 4 contrasts AR training with discrete/continuous diffusion and hybrid block diffusion, making the generative loops concrete (masking strategies, attention patterns). This helps practitioners reason about engineering trade‑offs (e.g., feasibility of caching).

- Systematization of DLM‑specific inference tooling
  - Section 4 and Fig. 5 synthesize a toolkit—parallel decoding, remasking, guidance, KV/feature caches, step distillation—showing how each piece slots into the iterative denoising loop. The survey highlights where large speedups are achieved (e.g., “up to 27×” from Fast‑dLLM; “up to 34×” with SlowFast+cache; “up to 500×” with one‑step distillation) and what is traded off.

- Clear articulation of the “Parallel Decoding Curse” with concrete evidence
  - Section 8.1 explains and Fig. 7 visualizes how aggressively accepting many tokens per step can break global consistency (e.g., math problems solved correctly only when unmasking 1–2 tokens per step; wrong or incoherent answers with fewer steps). This frames a central open problem unique to DLMs.

- Bridge from AR‑style reasoning/RL to diffusion
  - Section 3.2 and Table 2 collect and explain how CoT, PPO/GRPO, and DPO‑like methods are re‑engineered for DLMs (e.g., SEPO objective in Eq. (12), VRPO sampling design). This is a conceptual advance that opens the door to robust reasoning‑aligned DLMs.

## 5. Experimental Analysis
- Evaluation setup as compiled by the survey (Section 6; Fig. 6)
  - Benchmarks cover general language understanding (PIQA, HellaSwag), mathematical reasoning (GSM8K), code generation (HumanEval), and multimodal understanding/generation (GenEval, MME, MMMU, GQA).
  - Comparisons are size‑matched where possible (from <1B to ≈8B parameters), with AR baselines like LLaMA‑2/3, Qwen, Mistral shown alongside DLMs such as LLaDA, Dream, DiffuLLaMA, DiffuCoder, MMaDA, Dimple, and unified models (Fig. 6).

- Main quantitative takeaways (Fig. 6; Section 6; Table 1 for model context)
  - General language understanding: DLMs like `LLaDA-8B` are “slightly below or on par” with similar‑sized AR models on PIQA/HellaSwag.
  - Math/Science: DLMs often shine. The survey notes consistent gains on GSM8K for `LLaDA`/`Dream`, especially after reasoning post‑training (Section 3.2; Fig. 6).
  - Code: `DiffuCoder-7B` is competitive on HumanEval among open‑source models; `Mercury Coder` (industrial) reports AR‑beating throughput “up to 10×” at comparable quality (Sections 5, 7.2; Table 1).
  - Multimodal: `MMaDA` and `LLaDA‑V` outperform several AR‑VLMs on understanding (MME, GQA) and achieve strong T2I/I2T quality; unified discrete diffusion (`UniDisc`, `Muddit`) also performs competitively (Section 5; Fig. 6).

- Acceleration evidence (Section 4)
  - Parallel decoding and caches:
    - “Up to 27.6× speed-ups” (Fast‑dLLM; Section 4.1).
    - “Up to 34× acceleration” with SlowFast sampling plus caching (Section 4.1).
    - KV caches tailored to DLMs (dKV‑Cache): “2–10× speed-ups” with negligible quality drop (Section 4.4).
  - Step distillation:
    - `Di4C` compresses discrete diffusion to 4–10 steps at ≈teacher quality (~2× faster; Section 4.4).
    - `DLM‑One` trains a one‑step continuous DLM achieving “up to 500× acceleration” with near‑teacher quality (Section 4.4).

- Ablations and robustness highlighted by the survey
  - Unmasking/remasking policies matter: confidence‑aware acceptance and revisiting tokens (ReMDM) improve quality/compute trade‑offs (Section 4.2).
  - Parallelism–quality trade‑off is real and visual: Fig. 7 shows correct math reasoning only under conservative per‑step acceptance; aggressive parallelism induces inconsistencies (Section 8.1).
  - Training–inference discrepancy remedies: two‑step diffusion or schedule tweaks improve discrete DLMs’ inference behavior (Section 3.1).

- Do the results support the claims?
  - The aggregated plots (Fig. 6) and concrete speedup reports (Section 4) credibly support that DLMs are now competitive with similarly sized AR models on many tasks while unlocking strong parallel‑time acceleration—conditional on careful decoding/caching and, for the largest speedups, step distillation.

## 6. Limitations and Trade-offs
- Parallelism vs. coherence (“Parallel Decoding Curse”) (Section 8.1; Fig. 7)
  - Accepting many tokens per step ignores inter‑position dependencies; even if each local prediction is high‑probability, the joint sequence can be inconsistent (toy ABAB example in Section 8.1). Fig. 7 confirms quality drops when steps are too few or acceptance too aggressive.

- Infrastructure maturity (Section 8.1)
  - Unlike AR LLMs (with widely used serving stacks like HuggingFace Transformers and vLLM), DLMs lack standardized, optimized open‑source serving infrastructure, complicating deployment.

- Long sequences and dynamic lengths (Section 8.1)
  - Many DLMs are trained for fixed context windows (often ≤4k). Extrapolation techniques common in AR models are underexplored. Inference complexity can be cubic in length: O(N^2) per step with full attention times O(N) steps ≈ O(N^3), unless mitigated by KV‑cache or block designs.

- Scalability and data (Section 8.1)
  - The largest open DLMs are ≈8B parameters; today’s strongest AR systems scale to tens/hundreds of billions or MoE trillions. Many high‑performing DLMs are adapted from AR checkpoints (Table 1), leaving “from‑scratch at scale” relatively untested.

- Assumptions and edge cases
  - Discrete DLMs often assume a good mask schedule and reliable confidence calibration for acceptance. Continuous DLMs require robust rounding from embeddings/logits to tokens; mismatch can harm fluency (Section 2.2–2.3).
  - Reasoning RL for DLMs relies on approximate likelihood estimates (e.g., masked forward passes); bias/variance trade‑offs remain (Section 3.2).

## 7. Implications and Future Directions
- How this work changes the landscape
  - By clarifying mechanisms and limits side‑by‑side (Figs. 3–5), the survey makes DLM design choices tractable: when to prefer discrete vs. continuous, how to mix AR and diffusion, and which inference tools deliver real speed at acceptable quality.
  - It surfaces a coherent agenda for making DLMs practically competitive at scale: robust caching, principled acceptance/remasking, and distillation pathways (Sections 4, 8).

- Follow‑up research enabled or suggested (Section 8.2)
  - Training efficiency: improve token utilization (e.g., complementary masking as in LaViDa, Section 3.1) and reduce train–inference mismatch.
  - Low‑bit and sparsity: quantization/binarization and pruning for DLMs are largely open, with large upside for latency and memory.
  - Compression: distillation for fewer steps or smaller students (Di4C; DLM‑One) and hybridization with AR for cache‑friendly decoding.
  - Unified multimodal reasoning: expand models like MMaDA/UniDisc to reason across text–image jointly with RL‑aligned objectives (Section 5).
  - DLM‑based agents: exploit iterative refinement and bidirectionality for tool use, planning, and self‑correction loops.

- Practical applications
  - High‑throughput code generation (Mercury Coder; DiffuCoder) where parallel revision is handy (Sections 5, 7.2).
  - Structured editing, infilling, and constrained generation via remasking and DFA‑based control (DINGO; Section 4.3).
  - Unified multimodal systems capable of both understanding and generation, with controllable formats and fewer modality‑specific components (MMaDA, UniDisc, Muddit; Section 5).

> In sum, the survey shows that DLMs have moved from small proofs‑of‑concept to 7–8B‑scale models competitive with AR peers on several fronts, especially math and multimodal tasks (Section 6, Fig. 6), while offering inference speedups from smart parallel decoding, caching, and distillation (Section 4). The remaining bottlenecks—coherence under high parallelism, long‑context scaling, and ecosystem maturity—define the next phase of research (Section 8).
