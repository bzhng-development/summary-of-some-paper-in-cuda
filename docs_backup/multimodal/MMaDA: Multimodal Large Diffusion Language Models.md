# MMaDA: Multimodal Large Diffusion Language Models

**ArXiv:** [2505.15809](https://arxiv.org/abs/2505.15809)
**Authors:** Ling Yang, Ye Tian, Bowen Li, Xinchen Zhang, Ke Shen, Yunhai Tong, Mengdi Wang
**Institutions:** Princeton University, ByteDance Seed, Peking University, Tsinghua University

## 🎯 Pitch

MMaDA revolutionizes multimodal AI by unifying textual reasoning, multimodal understanding, and text-to-image generation into a single diffusion-based model. Leveraging a unique reinforcement learning approach and a unified discrete diffusion objective, it excels in cross-modal tasks, simplifying deployment while enhancing cross-domain knowledge transfer, marking a significant step towards more integrated AI systems.

---

## 1. Executive Summary (2–3 sentences)
MMaDA introduces a single diffusion-based foundation model that performs three capabilities at once: textual reasoning, multimodal understanding, and text-to-image generation. It does this by (i) using one unified, discrete diffusion objective for both text and images, (ii) post-training with a mixed, long chain-of-thought (CoT) format shared across tasks, and (iii) a new reinforcement learning method, UniGRPO, that adapts Group Relative Policy Optimization to diffusion models with diversified reward signals (Sections 2.1–2.3, Algorithm 1). The model achieves strong, balanced performance across tasks, e.g., best or near-best scores on image generation metrics and competitive language reasoning (Tables 2–4).

## 2. Context and Motivation
- Problem/gap addressed
  - Unified multimodal foundation models usually rely on autoregressive (AR) text modeling with separate components for vision, or combine AR for text with diffusion for images (Table 1). These systems emphasize pretraining but underexplore post-training—especially reinforcement learning (RL)—for diffusion-based architectures (Introduction; Section 2).
  - Diffusion models for language face practical RL barriers: token log-likelihoods only exist on masked positions, sensitivity to mask ratios, and the lack of an AR chain rule to compute sequence likelihoods (Section 2.3.1: “Challenges…”).
- Why this matters
  - A single model that can reason over text, understand images, and generate images—while sharing a common probabilistic formulation—promises simpler deployment and stronger cross-modal transfer (Section 2; Figure 2). Post-training for diffusion models could unlock the same alignment and reasoning gains that RL brought to AR LLMs.
- Prior approaches and limits
  - “Unified one-model” AR systems (e.g., Emu3, Chameleon, LWM) achieve generality but often lag behind diffusion models in image synthesis quality (Table 3).
  - “Two-model” hybrids (e.g., Show-o, Transfusion) split modalities: AR for text and diffusion for images, complicating post-training and cross-modal transfer (Table 1).
  - Initial diffusion LLMs (e.g., LLaDA) unify objectives but do not present an efficient, on-policy RL pipeline for diffusion (Table 1; Section B).
  - d1 (diff-GRPO) pioneers RL for diffusion LMs but masks the entire answer and randomly masks the question, which limits learning across multi-step denoising and misaligns training with inference (Section B).
- Positioning
  - MMaDA is a “one-model, one-objective” system using discrete diffusion for both text and vision (Table 1). It couples this with a task-agnostic CoT format for post-training and a diffusion-tailored RL algorithm (UniGRPO), aiming to bridge the pretraining–post-training gap in unified diffusion architectures (Sections 2.1–2.3).

## 3. Technical Approach
MMaDA is built around three stages (Figure 2): unified pretraining, mixed long-CoT finetuning, and unified RL (UniGRPO). The system treats both text and images as discrete tokens to be recovered from masked/noised versions, using the same objective and network.

1) Unified discrete diffusion pretraining (Section 2.1)
- Tokenization
  - Text: tokenizer from `LLaDA` (a diffusion-language model) (Section 2.1).
  - Image: MAGVIT-v2 tokenizer from Show-o; a 512×512 image becomes a 32×32 grid of tokens (1024 total) from a codebook of size 8192 (Section 2.1).
- Core modeling idea
  - MMaDA is a single mask-token predictor pθ that receives a noised sequence `x_t` and predicts the original tokens `x_0` only at positions that are masked (Show-o style “mask-and-replace” diffusion; Appendix A.1).
- Unified pretraining loss (Equation (1))
  - Plain-language view: randomly choose a noise level t, corrupt the clean sequence `x_0` into `x_t` by masking some tokens, and train the model to fill in the masked tokens. Only masked positions contribute to the loss.
  - Notation (Eq. (1)):
    - L_unify(θ) = − E_{t, x0, xt}[ (1/t) ∑_{i=1}^L I[x^i_t = [MASK]] log pθ(x^i_0 | x_t) ].
    - `I` is an indicator that selects masked positions; the 1/t factor weights by noise level. This same loss is used for text and image tokens.

2) Mixed Long-CoT finetuning (Section 2.2)
- Motivation
  - Diffusion LMs benefit from explicit reasoning traces before producing the final answer (text) or image (world knowledge-aware generation). However, formats differ across tasks, creating friction.
- Unified CoT format
  - A simple, task-agnostic structure: “|<special_token>| <reasoning_process> |<special_token>| <result>” (Section 2.2).
  - This lets textual and visual tasks share the same “reason-then-produce” template, promoting cross-task transfer (examples in Figure 1 and Figure 5).
- Data curation
  - Mixed long CoT traces for: textual reasoning (math/logical), multimodal reasoning (e.g., GeoQA, CLEVR), and world-knowledge-aware text-to-image prompts. Responses are generated with open models and then verified/filtered to retain accuracy and depth (Section 2.2; Figure 2).
- Training objective (Equation (2))
  - Keep the prompt tokens `p_0` intact, mask only the response segment `r_0` to create `r_t`, and train to reconstruct masked response tokens:
    - L_Mixed-SFT = − E[ (1/t) ∑ I[r^i_t = [MASK]] log pθ(r^i_0 | p_0, r_t) ].
  - Intuition: the model learns to use the clean prompt and partially noised response (including reasoning steps) to reconstruct the correct response.

3) Unified RL for diffusion (UniGRPO) (Sections 2.3.1–2.3.2; Algorithm 1)
- Why GRPO needs rethinking for diffusion
  - In AR models, sequence probability factorizes over tokens; diffusion lacks that chain rule. Also, meaningful token log-probs exist only on masked positions, and performance depends on how much is masked (Section 2.3.1).
- Key design elements
  - Structured noising: For each generated response `o_i`, sample a mask ratio `p_i ∈ [0,1]` and mask only the answer tokens (the question stays unmasked), then compute log-probs on the masked subset (Section 2.3.1, Point 1; Algorithm 1 lines 13–16). This exposes the model to many denoising stages—from heavily masked to lightly masked—matching how diffusion generation actually proceeds.
  - Efficient likelihood approximation:
    - Per-token log-likelihood under a perturbation is defined only on masked tokens and then averaged to approximate a sequence-level score (Equations (3)–(4)).
  - Policy objective with clipping and KL:
    - Compute importance ratios using the approximated likelihoods, apply GRPO-style clipping per token, average per sequence, and add a KL penalty to a reference model (Equation (5); Algorithm 1). Advantage estimates are group-relative (Eq. (15) in Appendix A.2).
  - Uniformly spaced denoising steps:
    - Within each mini-batch update, use multiple timesteps that uniformly cover the diffusion range for stability and better Monte Carlo coverage (Algorithm 1, lines 7–16).
- Diversified reward modeling (Section 2.3.2; Equation (6))
  - A unified reward interface `R_Uni(o)` is instantiated per task:
    - Textual reasoning: +2.0 for correctness, +0.5 for CoT-format compliance.
    - Multimodal reasoning: same as above, plus a CLIP(image,text) reward scaled by 0.1.
    - Text-to-image: CLIP and ImageReward (a learned proxy for human preference) both scaled by 0.1.
  - The overall UniGRPO objective is E[ F(R_Uni) − β·KL ], where F is the clipped surrogate from Eq. (5) (Remark 1).
- Inference strategies (Section 3)
  - Text: semi-autoregressive (Semi-AR) denoising as in LLaDA—divide the sequence into blocks; within a block, iteratively unmask low-confidence tokens and denoise; then move to the next block (length 1024, 512 steps, block size 64; unmask 2 least-confident tokens per step).
  - Image: parallel non-AR remasking with a cosine schedule; 50 denoising steps; classifier-free guidance scale 3.5.

## 4. Key Insights and Innovations
- A single discrete diffusion objective for both text and images (fundamental)
  - What’s new: Most “unified” models either keep modality-specific heads/objectives or split AR/text and diffusion/image (Table 1). MMaDA uses one mask-prediction diffusion architecture and one loss for both modalities (Section 2.1; Eq. (1)).
  - Why it matters: This simplifies the stack and encourages cross-modal transfer—e.g., reasoning learned on text benefits image generation and VQA (Figure 5 and Figure 6 show qualitative and quantitative synergy).
- Mixed Long-CoT with a unified format across tasks (significant)
  - What’s new: A single CoT structure for textual reasoning, multimodal reasoning, and world-knowledge-aware generation (Section 2.2).
  - Why it matters: It “cold-starts” RL (Section 2.2) by giving the diffusion model a consistent way to represent reasoning before producing answers or images, improving robustness and transfer (Table 5: large gains after Stage 2).
- UniGRPO: GRPO tailored to diffusion models (fundamental)
  - What’s new: A practical, efficient policy-gradient method that respects diffusion’s masked-token likelihoods, variable mask ratios, and multi-step denoising (Section 2.3.1; Equations (3)–(5); Algorithm 1).
  - Why it matters: It enables on-policy RL for diffusion LMs without the heavy Monte Carlo masking used by LLaDA, and without d1’s limitations (masking the whole answer and random question masking). Figures 3–4 show higher and more stable rewards during training versus diff-GRPO/d1-style strategies.
- Diversified rewards under one RL objective (incremental but useful)
  - What’s new: A single RL interface (Equation (6)) instantiates task-appropriate rewards, including correctness, format, CLIP, and ImageReward (Section 2.3.2).
  - Why it matters: It unifies post-training across reasoning and generation tasks, aligning diffusion policies with desired outcomes (Table 3 improvements track the optimized metrics).

## 5. Experimental Analysis
- Evaluation setup (Section 4)
  - Pretraining and post-training
    - Stage 1: 600K steps total (200K + 400K) over RefinedWeb (text), ImageNet-1K (class-conditional) early on, and large image–text corpora (Section 4.1).
    - Stage 2: 50K steps of Mixed Long-CoT finetuning with instruction and reasoning data.
    - Stage 3: 50K steps of UniGRPO on math/logical and multimodal tasks.
    - Hardware: 64× A100 80GB, global batch size 1,280; AdamW with LR 5e-5 and cosine scheduler (Section 4.1).
  - Datasets and metrics
    - Multimodal understanding: POPE, MME, Flickr30k, VQAv2, GQA, MMMU (Table 2).
    - Image generation: CLIP Score and ImageReward on 50K prompts; GenEval (object compositionality); WISE (world-knowledge-aware generation) (Table 3).
    - Text generation: MMLU, ARC-C, TruthfulQA, GSM8K, MATH, GPQA (Table 4).
    - Baselines include understanding-only LMMs and unified models (Tables 2–3).
- Main quantitative results
  - Multimodal understanding (Table 2)
    - MMaDA achieves: POPE 86.1, MME 1410.7, Flickr30k 67.6, VQAv2 76.7, GQA 61.3, MMMU 30.2.
    - Compared to unified baselines: it outperforms Show-o (e.g., POPE 80.0, VQAv2 69.4) and SEED-X (e.g., Flickr30k 52.3, MMMU 35.6) on many metrics. Against LLaVA-v1.5 (understanding-only), MMaDA is close or better on some (POPE 86.1 vs 85.9) but behind on MME (1410.7 vs 1510.7).
    - Takeaway: Strong understanding while keeping a single diffusion objective for both modalities.
  - Image generation (Table 3)
    - > “MMaDA (Ours): WISE(Cultural) 0.67, ImageReward 1.15, CLIP 32.46, GenEval Overall 0.63.”  
      These are the best or tied-best among unified models and competitive with specialized generators like SDXL (CLIP 32.12, ImageReward 1.13).
    - Notable compositionality: GenEval Single Obj 0.99; Two Obj 0.76; Counting 0.61. Some weaknesses remain on Position (0.20) and Color Attributes (0.37).
  - Textual reasoning and knowledge (Table 4)
    - MMaDA-8B: MMLU 68.4, ARC-C 57.4, TruthfulQA 43.1, GSM8K 73.4, MATH 36.0, GPQA 28.4.
    - It surpasses LLaDA-8B in math (GSM8K 73.4 vs 70.7; MATH 36.0 vs 27.3) and is competitive on general knowledge. It outperforms LLaMA-3-8B on GSM8K (73.4 vs 53.1) but is below Qwen2-7B on MMLU and GSM8K (70.3 and 80.2 for Qwen2-7B).
- Ablation and design validation
  - Stage-wise gains (Table 5)
    - Stage 1 → Stage 2 (Mixed Long-CoT): GSM8K 17.4→65.2, MATH500 4.2→26.5, CLIP 23.1→29.4.
    - Stage 2 → Stage 3 (UniGRPO): GSM8K 65.2→73.4, MATH500 26.5→36.0, CLIP 29.4→32.5, ImageReward 0.84→1.15.
    - Interpretation: CoT finetuning unlocks reasoning; UniGRPO consolidates and pushes both reasoning and image metrics further.
  - Masking strategy and timestep scheduling (Figures 3–4)
    - > “UniGRPO achieves consistently higher correctness reward over RL steps than diff-GRPO/d1” (Figure 3).
    - > “Uniformly spaced timesteps produce steadier reward curves than fully random masking” (Figure 4).
  - Sampling efficiency (Table 6)
    - Image CLIP only slightly drops from 32.8 (1024 steps) to 32.0 (50 steps) and 31.7 (15 steps).
    - MMLU remains close from 66.9 (1024) to 65.7 (256).
    - Takeaway: diffusion’s parallel denoising provides a good speed–quality trade-off.
- Do experiments support the claims?
  - Yes for the core claims:
    - A single diffusion model can be competitive in multimodal understanding and textual reasoning while excelling in image generation (Tables 2–4).
    - Mixed Long-CoT and UniGRPO are causally linked to gains (Table 5, Figures 3–4).
  - Nuances:
    - Language knowledge accuracy trails strong AR LLMs like Qwen2-7B on some benchmarks (Table 4).
    - Compositional position control in images remains challenging (GenEval Position 0.20 in Table 3).
- Qualitative evidence
  - Cross-task synergy: Figure 5 shows responses become more detailed/grounded across text, VQA, and image generation as training proceeds.
  - Reasoning-then-image examples: Figure 1 and Section C/D show CoT leading to world-knowledge-aligned images (e.g., Statue of Liberty).

## 6. Limitations and Trade-offs
- Modeling/algorithmic assumptions
  - Likelihood approximation: UniGRPO averages masked-token log-probs to approximate sequence-level likelihoods (Eq. (4)). This is efficient but approximate; errors may bias RL updates (Section 2.3.1).
  - Reward shaping: Optimization targets proxy metrics (CLIP, ImageReward) and strict output formats. Over-optimizing proxies can misalign with human preferences in out-of-distribution cases (Section 2.3.2).
- Task coverage
  - Focuses on text and images; audio/video are not included. Some image compositional controls (object position, attribute binding) remain weak (Table 3 breakdown).
- Data and compute
  - Training requires large-scale compute (64×A100 80GB) and many steps across three stages (Section 4.1).
  - CoT data relies on generated-and-verified traces; quality depends on verifiers and filters (Section 2.2).
- Performance trade-offs
  - Compared with top AR LLMs, the diffusion LM is competitive on some reasoning tasks (e.g., GSM8K) but behind on others (MMLU, TruthfulQA versus Qwen2-7B; Table 4).
  - Understanding-only models like LLaVA still lead on specific metrics (MME; Table 2).

## 7. Implications and Future Directions
- How this changes the landscape
  - Demonstrates that a single diffusion objective can power both language and vision in a unified model—narrowing the historical gap where diffusion excels at images while AR dominates text. The paper also provides a viable RL recipe for diffusion LMs (UniGRPO), analogous to PPO/GRPO for AR LLMs (Sections 2.3; Algorithm 1).
- Follow-up research enabled/suggested
  - Extending UniGRPO to other modalities (audio, video) and tasks (e.g., code, tool use) with tailored rewards under the same Equation (6).
  - Improving likelihood estimation for diffusion sequences beyond masked averaging (Eq. (4))—e.g., tighter bounds or learned calibrations.
  - Stronger CoT verification and self-consistency filtering to reduce hallucinations and shallow reasoning in mixed CoT data (Section 2.2).
  - Better compositional controls in generation (especially spatial relations), perhaps via structured prompts, scene graphs, or planning modules before diffusion.
- Practical applications
  - A single assistant that can read diagrams, reason about math/geometry, explain images, and generate faithful images grounded in world knowledge (Figures 1, 5; Tables 2–3).
  - Inpainting and span completion across modalities (text, VQA answers, image regions) with no extra finetuning, since these are native masked-token predictions (Section 5.5; Figure 7).

> Representative quantitative highlights
> - Multimodal understanding: “POPE 86.1; VQAv2 76.7; GQA 61.3” (Table 2).
> - Image generation: “WISE 0.67; CLIP 32.46; ImageReward 1.15; GenEval Overall 0.63” (Table 3).
> - Text reasoning: “GSM8K 73.4; MATH 36.0” with stage-wise gains from CoT and UniGRPO (Tables 4–5).

Overall, MMaDA provides a concrete blueprint for unified diffusion language models—from pretraining to RL—showing that diffusion can be a credible “all-in-one” foundation paradigm when paired with mixed CoT and an RL method that respects diffusion’s mechanics.
