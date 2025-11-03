# MMaDA: Multimodal Large Diffusion Language Models

**ArXiv:** [2505.15809](https://arxiv.org/abs/2505.15809)

## 🎯 Pitch

MMaDA introduces the first unified diffusion-based language model that seamlessly integrates textual reasoning, multimodal understanding, and text-to-image generation within a single network. By pioneering a modality-agnostic diffusion framework, mixed chain-of-thought finetuning, and a novel reinforcement learning algorithm (UniGRPO) tailored for diffusion models, MMaDA achieves state-of-the-art results across all three tasks. This breakthrough not only streamlines AI architectures for practical deployment but also challenges the paradigm that autoregressive models are essential for language and reasoning, opening new avenues for efficient, general-purpose multimodal intelligence.

---

## 1. Executive Summary (2–3 sentences)
MMaDA proposes a single “diffusion language model” that handles three capabilities in one network: textual reasoning, multimodal understanding, and text-to-image generation. It achieves this by (i) modeling both text and images with a unified discrete diffusion objective, (ii) aligning reasoning across tasks via a mixed long chain-of-thought (CoT) format, and (iii) introducing UniGRPO, a reinforcement-learning algorithm adapted to diffusion models, which delivers consistent post-training gains across modalities (sections 2–3).

## 2. Context and Motivation
- Gap addressed
  - Unified multimodal foundation models largely rely on autoregressive (AR) training, or they mix AR for text with diffusion for images (table 1). They emphasize architecture and pretraining, but post-training (especially RL) for non-AR/diffusion settings remains underexplored (section 1).
  - There is no single diffusion model that is strong at both reasoning (text and multimodal) and image generation; diffusion RL for language and multimodal tasks is still immature.

- Why it matters
  - Practical: A single model that reasons, understands images, and generates images reduces engineering overhead and inference cost, and can transfer improvements across tasks (figure 6 shows cross-task synergy).
  - Scientific: Demonstrates diffusion models can serve as general-purpose “language” models for discrete sequences (text and image tokens), challenging the dominance of AR training for language.

- Prior approaches and their limitations
  - Pure AR unification (e.g., Emu3, Chameleon) simplifies training but tends to lag in image quality compared to diffusion (table 3).
  - Hybrid AR+diffusion systems (e.g., Show-o, Transfusion) need modality-specific heads/objectives and focus less on post-training (table 1).
  - LLaDA shows text-only diffusion LMs but does not offer a unified multimodal model or an on-policy RL algorithm suitable for diffusion (sections 2.3 and B).

- Positioning
  - MMaDA is a “one-model, one-objective” approach using discrete diffusion for both language and vision (table 1), adds a cross-modal CoT finetuning stage (section 2.2), and introduces UniGRPO, a GRPO-style RL algorithm tailored to diffusion (section 2.3 and algorithm 1).

## 3. Technical Approach
MMaDA builds a single transformer that predicts masked tokens for both text and images using a discrete diffusion process.

- Tokenization (section 2.1 Data Tokenization)
  - Text: LLaDA tokenizer.
  - Image: MAGVIT-v2 tokenizer (downsampling factor `f=16`, codebook size 8192). A 512×512 image becomes a 32×32 grid of tokens (1024 tokens).
  - Rationale: Treat everything as sequences of discrete tokens so one model and one learning objective can be used.

- Unified discrete diffusion objective (section 2.1; eq. (1))
  - What is “discrete diffusion”? A training process that corrupts sequences by masking tokens and learns to reconstruct the original tokens from the corrupted sequence.
  - The model `pθ(·|xt)` takes the corrupted sequence `xt` and predicts the original token at each masked position. The unified loss is a cross-entropy computed only on masked tokens:
    - Lunify (eq. (1)) averages the negative log-likelihood over masked positions, with the corruption “timestep” `t` sampled uniformly from [0,1].
  - Why this design?
    - A single, modality-agnostic objective avoids separate text/image heads or different losses (contrast table 1 rows “Loss for Language” and “Loss for Vision”).
    - Mask-based diffusion stabilizes training and allows parallel prediction (section A.1).

- Mixed Long-CoT finetuning (section 2.2; eq. (2))
  - Goal: Align reasoning style and structure across tasks to “cold-start” RL and transfer reasoning to generation.
  - Unified CoT format: insert special tags to separate a “thinking” trace and the final result:
    - Format: `|<special>| <reasoning_process> |<special>| <result>` (section 2.2).
    - Data: Curated from open models (e.g., DeepSeek-R1, LMM-R1 on GeoQA/CLEVR, GPT‑4.1 for world-knowledge prompts) and verified for correctness/format (figure 2).
  - Training mechanism (eq. (2)):
    - Keep the original prompt `p0`, mask parts of the response `r0` to produce `rt`, and train the model to reconstruct the masked response from `[p0, rt]`.
    - This teaches the model to produce structured reasoning before answers in a way consistent across text reasoning, VQA-style tasks, and T2I prompts.

- UniGRPO: RL for diffusion models (section 2.3; algorithm 1; eqs. (3)–(6))
  - Background: GRPO is a PPO-style algorithm that normalizes rewards within a group of sampled responses and adds a KL penalty to a reference policy. Standard GRPO assumes AR likelihoods are easy to compute token-by-token (appendix A.2).
  - Challenges for diffusion (section 2.3.1):
    1) Token log-likelihoods are only valid on masked tokens,
    2) Likelihood depends on the mask ratio, and
    3) No AR chain rule for sequence likelihood.
  - Key adaptations:
    - Structured noising (step 1): For each sampled answer `oi`, pick a random mask ratio `pi ∈ [0,1]`, remask only the answer tokens (the question is never masked), and vary `pi` across gradient steps (algorithm 1, lines 7–16; section 2.3.1 “Structured Noising Strategy”).
      - This exposes the model to many denoising stages (from heavily masked to lightly masked) and leverages diffusion’s multi-step nature.
    - Likelihood approximation (step 2): Define the expected per-token log-probability under random remasking (eq. (3)), and approximate sequence likelihood by averaging masked-token log-probs (eq. (4)).
    - Policy gradient (step 3): Use a clipped objective with per-token ratios and a KL penalty to a reference policy (eq. (5)). Advantages are group-normalized as in GRPO (appendix A.2 eq. (15)).
  - Unified reward design (section 2.3.2; remark eq. (6)):
    - Text reasoning: correctness (+2.0 if correct) and format reward (+0.5 if the `<think>...</think>` pattern holds) on GSM8K.
    - Multimodal reasoning: add CLIP alignment reward (scaled by 0.1), plus correctness/format for tasks like GeoQA and CLEVR.
    - T2I generation: CLIP reward and ImageReward (both scaled by 0.1).
  - Why it matters: Unlike LLaDA’s heavy Monte Carlo over many mask ratios, UniGRPO uses one remask per update but varies it across steps for efficiency; unlike d1’s diff-GRPO, it does not mask the question and does not fully mask the answer every time—thereby training genuine multi-step denoising (sections 5.2 and B).

- Inference/sampling (section 3)
  - Text: “semi-autoregressive” (Semi-AR) block-wise denoising (borrowed from LLaDA).
    - Sequence length N=1024, 512 denoising steps, block size 64, unmask the 2 lowest-confidence tokens per step in each block; move block-by-block (section 3).
    - This yields longer, richer generations than parallel fixed-length decoding (qualitative example in section 3).
  - Image: parallel denoising (entire 1024-token image sequence as one block), cosine noise schedule, 50 steps, classifier-free guidance scale 3.5 (section 3).

## 4. Key Insights and Innovations
- One-model, one-objective diffusion architecture (fundamental)
  - What’s new: Both text and vision use the same discrete diffusion, mask-prediction objective (table 1 “Ours MMaDA”). Prior unified models either used AR for text and diffusion for images or required two models (table 1).
  - Why it matters: Removes modality-specific heads, simplifies training, and enables cross-modal transfer during both SFT and RL (sections 2.1–2.3).

- Mixed Long-CoT across modalities (substantial, enabling)
  - What’s new: A single CoT format across three tasks, trained via masked reconstruction of the final response (eq. (2), section 2.2).
  - Why it matters: Establishes reasoning traces as a first-class training signal for both language and vision tasks, and stabilizes downstream RL (“cold start”). Figure 5 and figure 6 visualize cross-task synergy.

- UniGRPO: GRPO adapted to diffusion (fundamental, algorithmic)
  - What’s new: Structured remasking to expose multiple denoising stages, a tractable log-likelihood approximation for diffusion (eqs. (3)–(4)), and a clipped-KL objective (eq. (5)) with diversified rewards across tasks (section 2.3).
  - Why it matters: Makes on-policy RL practical and effective for diffusion LMs; section 5.2 shows more stable and higher-reward training than diff-GRPO (d1) with uniform-timestep remasking (figures 3–4).

- Efficient and flexible inference (incremental but impactful)
  - Semi-AR text decoding mitigates short, premature outputs that appear with naïve parallel decoding in diffusion LMs (section 3), and image generation remains fast with 50 steps while retaining high alignment (table 6).

## 5. Experimental Analysis
- Evaluation setup (section 4.1)
  - Data
    - Pretraining: RefinedWeb for text; standard image–text corpora for multi-modal; ImageNet-1k for class-conditional generation (then replaced with broader caption data).
    - Instruction/Reasoning finetuning: Alpaca and LLaVA-1.5; reasoning sets from ReasonFlux, LIMO, s1k, OpenThoughts, AceMath-Instruct; LMM-R1 outputs filtered on GeoQA/CLEVR; GPT-4.1 prompts for knowledge-aware T2I.
    - RL: GSM8K, GeoQA, CLEVR.
  - Training
    - Stage 1: 200K steps (foundation) + 400K steps (broader image–text); Stage 2: 50K mixed Long-CoT SFT; Stage 3: 50K UniGRPO RL.
    - Hardware: 64×A100 80GB, global batch 1,280, AdamW with lr 5e-5 and cosine schedule.

- Benchmarks and metrics (sections 4.2–4.4; tables 2–4)
  - Multimodal understanding: POPE, MME, Flickr30k retrieval, VQAv2, GQA, MMMU, MMB, SEED.
  - T2I generation: CLIP Score, ImageReward, GenEval (object compositionality), WISE Cultural (world-knowledge alignment) (table 3).
  - Text reasoning and general knowledge: MMLU, ARC-C, TruthfulQA, GSM8K, MATH, GPQA (table 4).

- Main results
  - Multimodal understanding (table 2)
    - MMaDA scores POPE 86.1 and MME 1410.7, close to or above strong VLMs like LLaVA‑v1.5 (POPE 85.9, MME 1510.7) while surpassing many unified baselines (e.g., Show‑o: POPE 80.0, MME 1097.2).
    - On VQAv2 76.7 and GQA 61.3, it’s competitive with understanding-only models and ahead of most unified models.
  - T2I generation (table 3)
    - MMaDA attains the top CLIP Score 32.46 and ImageReward 1.15 among listed models; it beats SDXL (32.12, 1.13) and Janus (29.45, 1.03).
    - On GenEval overall 0.63 (best in table), with notably strong “Two Objects” 0.61 and “Color Attribute” 0.37; WISE Cultural 0.67—well above SDXL 0.43 and Janus 0.16—indicating better world-knowledge grounding.
  - Text benchmarks (table 4)
    - MMaDA-8B vs text-only diffusion LM: improves over LLaDA‑8B on GSM8K (73.4 vs 70.7) and MATH (36.0 vs 27.3).
    - Compared to AR LLMs: it trails Qwen2‑7B and LLaMA‑3‑8B on some general knowledge tasks (e.g., MMLU 68.4 vs 70.3 for Qwen2‑7B), but is competitive and notably better than LLaDA on math reasoning.

- Do experiments support claims?
  - Unified strength: Yes—MMaDA is not merely a generator; it is competitive on understanding and textual reasoning, while leading on text-to-image alignment metrics (tables 2–3).
  - RL matters: Ablations (table 5) show clear gains from Stage 2 (Mixed Long-CoT) to Stage 3 (UniGRPO). Example: GSM8K jumps from 65.2 → 73.4, CLIP from 29.4 → 32.5, ImageReward from 0.84 → 1.15.
  - Masking strategy: Figures 3–4 show higher and more stable GSM8K rewards with UniGRPO’s uniform-timestep remasking than with diff-GRPO or fully random masking.
  - Efficiency: Table 6 shows image CLIP Score remains high even at 15–50 steps (31.7–32.0 vs 32.8 at 1024 steps). Text and multimodal tasks maintain performance at half or quarter steps.
  - Synergy: Figure 6 tracks concurrent improvements on MMLU, CLIP Score, and ImageReward during Stage 2, consistent with the “shared reasoning benefits generation” hypothesis; figure 5 offers qualitative examples of richer descriptions and more faithful generations.

- Qualitative checks
  - Figure 1 and section C/D examples demonstrate: (i) correct geometric reasoning with CoT, (ii) world-knowledge-grounded T2I (e.g., Statue of Liberty), and (iii) coherent textual calculations that follow the unified `<think>` format.

## 6. Limitations and Trade-offs
- Approximate likelihood and RL stability (sections 2.3.1 and B)
  - Sequence log-likelihood is approximated by averaging masked-token log-probs (eq. (4)). This is not an exact factorization like AR. While empirical results are strong, theoretical tightness and potential bias are open questions.
  - Rewards like CLIP and ImageReward (section 2.3.2) are known to be gameable; optimizing them may bias style or diversity.

- Data and compute intensity (section 4.1)
  - Three-stage training on 64×A100 GPUs with hundreds of thousands of steps and curated CoT data—substantial resource requirements and engineering overhead.

- Discrete tokenizers’ ceiling
  - Image quality depends on MAGVIT‑v2 codebook and 512×512 tokenization; higher resolutions or fine textures may be limited by codebook capacity and the 1024-token budget.
  - Text decoding relies on Semi-AR block schedules (section 3). The paper notes that naïve parallel decoding tends to produce short outputs or frequent EOS tokens—a sign that sampling details are crucial.

- Scope constraints
  - Model size is 8B parameters (Conclusion), which may cap performance on broad knowledge tasks versus larger AR LLMs (table 4 comparisons).
  - Modalities beyond text–image (audio, video, 3D) are not included, although the framework is conceptually extendable.

- Evaluation breadth
  - Many T2I metrics focus on alignment (CLIP, GenEval) rather than photorealism or aesthetic diversity. Human studies beyond ImageReward are not reported.

## 7. Implications and Future Directions
- Field impact
  - MMaDA demonstrates that diffusion can underpin a general multimodal foundation model, not just image synthesis. This challenges the default assumption that language must be AR and opens a path to parallel, block-wise language generation with better sampling efficiency (table 6).

- Research avenues
  - Better diffusion RL theory and estimators: tighter sequence-likelihood surrogates, variance reduction, critic-free vs critic-based hybrids, and richer reward models for factuality and safety.
  - Scaling and modalities: larger models, multi-resolution tokenizers, and extending the unified objective to video or audio tokens with the same mask-predict paradigm.
  - Training curricula: more principled CoT data construction and verification; adaptive remasking schedules; combining search/planning with diffusion decoding.
  - Evaluation: beyond CLIP/ImageReward, conduct controlled human studies and measure reasoning faithfulness in multimodal chains (“did the image follow the steps described in `<think>`?”).

- Practical applications
  - Unified assistant that can explain an image, reason about it, and create a new, knowledge-grounded image in one session—useful for education, design ideation, data augmentation, and multimodal agents.
  - Inpainting and span completion across text and image (figure 7) without extra finetuning suggests flexible editing tools powered by one model.

> Bottom line: MMaDA provides a clear recipe—unified discrete diffusion + cross-modal CoT + diffusion-native RL—that delivers strong, balanced performance across understanding, reasoning, and generation (tables 2–5). While compute-heavy and still approximative in its RL objective, it meaningfully advances diffusion models toward truly general multimodal intelligence.
