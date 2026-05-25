# InternVL3: Exploring Advanced Training and Test-Time Recipes for Open-Source Multimodal Models

**ArXiv:** [2504.10479](https://arxiv.org/abs/2504.10479)

## 🎯 Pitch

InternVL3 pioneers a native multimodal pre-training paradigm that enables large models to simultaneously learn language and visual capabilities from both text and multimodal data, eliminating the need for complex, post-hoc adaptation stages. By integrating innovations like Variable Visual Position Encoding (V2PE), advanced supervised fine-tuning and mixed preference optimization, and test-time scaling, InternVL3 achieves state-of-the-art performance among open-source models, dramatically simplifying the training pipeline and closing the gap with leading proprietary MLLMs. This approach greatly enhances scalability, efficiency, and robustness for real-world applications such as document understanding, GUI agents, and multimodal reasoning, while fostering transparency and reproducibility through the release of both data and model weights.

---

## 1. Executive Summary (2-3 sentences)
InternVL3 proposes a “native multimodal pre‑training” recipe that trains a vision–language model to learn language and visual skills together in a single stage, rather than adapting a text-only LLM afterward. Combined with a variable visual position encoding (V2PE), improved post-training (SFT + Mixed Preference Optimization), and test-time scaling with a visual process reward model, the 78B variant reaches state-of-the-art open-source results on many benchmarks, including 72.2 on MMMU and 906 on OCRBench (Figure 1; Tables 2–3).

## 2. Context and Motivation
- Problem/gap
  - Most multimodal LLMs (MLLMs) are built by adapting a text-only LLM through multi-stage “post-hoc” pipelines that add vision later (Section 1). This often requires delicate freezing/unfreezing schedules and extra alignment data to avoid degrading language ability, while still aligning visual and textual representations.
  - These pipelines are resource-heavy and can be brittle for new domains (OCR, GUIs, long videos), leaving a need for a simpler, more scalable approach that preserves language capability and scales to long context (Section 1).

- Why it matters
  - Practical: Robust, scalable MLLMs underpin real-world tasks like document understanding, GUI agents, spatial reasoning, and multi-image reasoning. Efficiency in training and inference is critical for open-source ecosystems.
  - Scientific: Demonstrates whether joint training on text and multimodal data can replace multi-stage adaptation, and whether position encoding innovations like V2PE help extend multimodal context effectively (Sections 2.1–2.2).

- Prior approaches and shortcomings
  - “Post-hoc” alignment (e.g., LLaVA-style pipelines) needs complex stages and special datasets (OCR, charts, etc.) to bridge modality gaps (Section 1). It risks catastrophic forgetting of language skills and adds engineering complexity.
  - Existing long-context methods typically treat all tokens uniformly in positional encoding, so visual tokens quickly eat context budget and limit sequence length (Section 2.1: V2PE motivation).

- Positioning
  - InternVL3 unifies pre-training for text and vision (“native multimodal pre-training,” Section 2.2), adopts a modality-aware positional encoding (V2PE, Section 2.1), adds targeted post-training (SFT + MPO, Section 2.3), and leverages test-time scaling with a visual process reward model (VisualPRM, Section 2.4). Infrastructure enhancements (InternEVO) improve training throughput (Section 2.5).

## 3. Technical Approach
- Architecture (“ViT–MLP–LLM,” Section 2.1; Table 1)
  - Vision encoder: `InternViT-300M` (for 1–14B models) or `InternViT-6B` (for 38B/78B) at 448 px input with pixel unshuffle, which reduces visual tokens by 4×, mapping each 448×448 tile to 256 tokens (Section 2.1).
  - Language model: pre-trained base LLMs from Qwen2.5 (0.5B–72B) or InternLM3-8B; no instruction-tuned LLMs are used for initialization (Table 1).
  - Connector: a 2-layer MLP maps visual embeddings to the LLM space (randomly initialized; Section 2.1).

- Variable Visual Position Encoding (V2PE; Section 2.1, Eqs. (1)–(4))
  - Goal: Stretch multimodal context by “spending” fewer position increments on visual tokens.
  - Mechanism:
    - Sequence tokens `x1..xL` receive position indices `p1..pL` recursively (Eq. (2)). For text tokens, index increases by 1; for visual tokens, by a fraction `δ<1` (Eq. (3)).
    - During training, `δ` is sampled per image from a set {1, 1/2, 1/4, …, 1/256} (Eq. (4)); during inference, `δ` can be chosen based on sequence length to keep positions within the context window.
  - Intuition: Visual tokens consume less “positional budget,” enabling longer sequences without losing text positional resolution.

- Native Multimodal Pre-Training (Section 2.2)
  - Objective: Autoregressive next-token prediction but compute loss only on text tokens (`L_text-only`, Eq. (6)), while visual tokens serve as conditioning context.
    - Formal: Minimize expected loss over a combined dataset of text-only and multimodal samples, updating all parameters jointly (Eq. (8)).
  - Loss scaling: Uses “square averaging” to balance gradients across short/long answers (Eq. (7)); mitigates bias that either favors long or short responses.
  - Data mixture: Approximately 200B training tokens with a 1:3 ratio of language:multimodal tokens (50B language, 150B multimodal; Section 2.2 “Data”). Multimodal sources cover captioning, OCR, charts, math, documents, video, GUIs, tools, 3D scenes, etc.

- Post-Training: SFT and Mixed Preference Optimization (MPO; Section 2.3)
  - SFT (supervised fine-tuning): Uses high-quality, diverse instruction data; preserves InternVL2.5 practices (random JPEG compression, square loss re-weighting, multimodal packing). Expanded data for tools/3D/GUI/long-context/video/diagrams/CoT, growing SFT corpus from 16.3M to 21.7M samples.
  - MPO: Addresses exposure bias and improves chain-of-thought by combining:
    - Preference loss (`L_p`, DPO-style, Eq. (10)) to learn chosen vs rejected responses.
    - Quality loss (`L_q`, BCO, Eqs. (11)–(13)) to model absolute response quality with a reward shift.
    - Generation loss (`L_g`, Eq. (6)) to learn to produce preferred responses.
    - Total loss is a weighted sum `L = w_p L_p + w_q L_q + w_g L_g` (Eq. (9)). MPO uses ~300K preference pairs derived following MMPR v1.2 (Section 2.3 “Data”).

- Test-Time Scaling with VisualPRM (Section 2.4)
  - Best-of-N (BoN): Sample multiple candidate solutions and select one using an external critic.
  - VisualPRM (Visual Process Reward Model): A separate 8B model that assigns step-level correctness probabilities and averages them to score a full solution (Eq. (14)). Trained on VisualPRM400K (extended with InternVL3 rollouts).

- Infrastructure: InternEVO extensions (Section 2.5)
  - Flexible sharding of ViT/MLP/LLM modules; supports data/tensor/sequence/pipeline parallelism and combinations; overlaps communication/computation.
  - Dynamic load balancing between visual and language modules to handle varying token proportions.
  - Supports sequences up to 32K via head-parallel + sequence-parallel. Reports 50–200% training speedup vs. InternVL2.5 at the same compute budget.

## 4. Key Insights and Innovations
- Native multimodal pre-training (fundamental)
  - What’s new: Jointly optimize all parameters on a mixture of text-only and multimodal data from scratch in one stage (Section 2.2), with loss computed only on text tokens (Eq. (6)).
  - Why it matters: Simplifies pipelines, eliminates fragile freeze/unfreeze schedules, and aligns vision/text representations during the earliest stage. Figure 3 shows that even without post-training, native pre-training alone yields strong multimodal capability; with SFT it surpasses the classic multi-stage pipeline.

- Variable Visual Position Encoding (V2PE) for long context (fundamental)
  - What’s new: Modality-specific position increments so visual tokens consume fewer position steps (Eqs. (2)–(4)).
  - Why it matters: Extends effective multimodal context without changing model size. Table 12 shows consistent gains on standard tasks even with moderate context, contradicting earlier reports that V2PE helps only for long contexts.

- Mixed Preference Optimization (MPO) that blends relative and absolute supervision (incremental but impactful)
  - What’s new: Combines DPO-style pairwise preference (Eq. (10)) with BCO-style absolute quality modeling (Eqs. (11)–(13)) plus standard generation loss (Eq. (6)).
  - Why it matters: Improves reasoning stability under self-generated tokens. Table 13 shows across seven reasoning benchmarks, MPO boosts overall scores by +1.5 to +4.5 depending on model size.

- Test-time scaling with VisualPRM as a critic (incremental but effective)
  - What’s new: Step-level process supervision to score and select the best sampled solution (Section 2.4).
  - Why it matters: Substantial test-time boosts in math/reasoning. For instance, InternVL3-8B improves MathVista from 71.6 to 75.2 and MathVision from 29.3 to 37.5 with BoN=8 (Table 2, “w/ VisualPRM-Bo8”).

- Practical training system upgrades (incremental)
  - InternEVO enables 32K sequences and 50–200% faster training than InternVL2.5 (Section 2.5), critical for scaling to 78B parameters with long-context inputs.

## 5. Experimental Analysis
- Evaluation setup
  - Benchmarks span:
    - Multimodal reasoning and math: MMMU, MathVista, MathVision, MathVerse, DynaMath, WeMath, LogicVista (Table 2).
    - OCR, charts, documents: AI2D, ChartQA, TextVQA, DocVQA, InfoVQA, OCRBench, SEED-2-Plus, CharXiv, VCR (Table 3).
    - Multi-image and real-world: BLINK, Mantis-Eval, MMIU, MuirBench, MMT-Bench, MIRB; RealWorldQA, MME-RealWorld, WildVision, R-Bench (Table 4).
    - Comprehensive capability and hallucination: MME, MMBench (EN/CN), MMBench v1.1, MMVet v1/v2, MMStar; HallusionBench, MMHal, CRPE, POPE (Table 5).
    - Grounding: RefCOCO/+/g (Table 6).
    - Multilingual: MMMB, Multilingual MMBench, MTVQA across EN/ZN/PT/AR/TR/RU (Table 7).
    - Video: Video-MME, MVBench, MMBench-Video, MLVU, LongVideoBench, CG-Bench (Table 8).
    - GUI grounding: ScreenSpot and ScreenSpot-V2 (Table 9).
    - Spatial reasoning: VSI-Bench (Table 10).
    - Language-only capability: MMLU, CMMLU, C-Eval, GAOKAO, TriviaQA, NQ, RACE, WinoGrande, HellaSwag, BBH, GSM8K, MATH, TheoremQA, HumanEval, MBPP/MBPP-CN (Table 11).
  - Models: 1B to 78B variants (Table 1). Results compared to Qwen2.5-VL series and closed models like GPT‑4o, Claude 3.5, Gemini 1.5/2.0/2.5 (Figures 1–2; multiple tables).

- Main quantitative highlights
  - Overall multimodal strength
    - > “InternVL3-78B achieves MMMU 72.2” (Table 2), leading open-source MLLMs and close to proprietary models (Figure 1).
    - > “OCRBench 906, AI2D 89.7, ChartQA 89.7, DocVQA 95.4” (Table 3), with competitive or superior scores to Qwen2.5‑VL‑72B and GPT‑4o on several tasks.
  - Reasoning and math (Table 2)
    - > “InternVL3-78B: MMMU 72.2, MathVista 79.0, MathVision 43.1, MathVerse (vision-only) 51.0.”
    - With VisualPRM-Bo8, 38B and 8B models gain +4–8 points on average (Table 2; best-of-8 rows).
  - Multi-image and real-world (Table 4)
    - > “InternVL3-78B: BLINK 66.3, MMT-Bench 73.2, RealWorldQA 78.0, MME-RealWorld 65.4, WildVision 73.6, R-Bench 77.4.”
    - Competes closely with GPT‑4o on RealWorldQA (78.0 vs. 75.4) and is strong on MME-RealWorld.
  - Hallucination & comprehensive (Table 5)
    - > “InternVL3-78B: MME sum 2549.8; MMBench EN/CN 89.0/88.7; MMVet 81.3; HallusionBench 59.1; CRPE 79.2; POPE 90.3.”
    - Gemini 2.5 Pro can lead on some hallucination tests (e.g., HallusionBench in Figure 1).
  - Grounding (Table 6)
    - > “InternVL3-78B overall 91.4,” slightly below InternVL2.5‑78B (92.3), suggesting limited new grounding-specific data.
  - Multilingual (Table 7)
    - > “InternVL3-78B overall 68.9,” edging out InternVL2.5‑78B (68.0) and comparable to Qwen2‑VL‑72B (67.2).
  - Video (Table 8)
    - > “InternVL3-78B: Video-MME 72.7/75.7, MVBench 78.7, MLVU 79.5, LongVideoBench 65.7, CG-Bench 48.4/65.3.” Strong scaling trend across sizes.
  - GUI and spatial reasoning (Tables 9–10)
    - > “ScreenSpot-V2: InternVL3-72B 90.9,” slightly above UI‑TARS‑72B (90.3); “VSI-Bench overall: 38B 48.9; 78B 48.4,” surpassing open and closed baselines on several sub-tasks.

- Ablations and diagnostics
  - Native multimodal pre-training vs. classic pipeline (Figure 3)
    - The pre-trained-only model (no SFT) with native pre-training already reaches near full-pipeline performance; with SFT it exceeds it across many tasks—evidence that the unified training is effective.
  - V2PE δ sweep (Table 12)
    - Using V2PE improves many metrics even for short contexts; small δ (e.g., 1/4–1/16) often best. For fairness, the rest of the paper reports δ=1 unless noted (Section 3.14).
  - MPO impact (Table 13)
    - Improves reasoning by +1.5 to +4.5 overall points depending on scale; largest relative gains at larger scales (38B/78B), suggesting synergy with capacity.

- Are claims supported?
  - The paper triangulates: broad benchmark coverage (Tables 2–10), detailed ablations (Figure 3; Tables 12–13), and comparisons to both open- and closed-source models (Figures 1–2). The gains on MMMU, OCRBench, RealWorldQA, and process-supervised BoN solidly substantiate the core contributions.

- Notable mixed results
  - Visual grounding plateaus or dips slightly at the largest scale (Table 6), likely due to less grounding-specific data.
  - Hallucination scores are competitive but not dominant on every benchmark (Figure 1; Table 5).

## 6. Limitations and Trade-offs
- Data and training assumptions
  - Relies on ~200B tokens total and diverse multimodal corpora (Section 2.2 “Data”). Reproducing results requires large-scale data curation and infrastructure (Section 2.5).
  - The `L_text-only` objective (Eq. (6)) never predicts visual tokens. While simpler and effective, it may limit the model’s ability to generate dense visual sequences (e.g., pixel-level tasks) without further adaptations.

- Task coverage gaps
  - Visual grounding: performance saturates or slightly declines at the largest scale (Table 6), likely because the training expansion did not emphasize grounding data (Section 3.8 discussion).
  - Hallucinations: improvements are not uniform across all hallucination benchmarks (Table 5), leaving room for targeted robustness work.

- Computational costs
  - Largest variant uses `InternViT-6B` + `Qwen2.5-72B` (Table 1) and 32K context support with complex parallelism (Section 2.5). Inference/test-time scaling (Best-of-N with VisualPRM) increases compute at evaluation time.

- Sensitivity and tuning
  - V2PE δ selection impacts results (Table 12). Although the paper fixes δ=1 for fairness outside the ablation, practitioners will need policies to pick δ based on sequence length at inference.

- Dependency on critic at test time
  - The biggest reasoning gains rely on VisualPRM BoN (Table 2), introducing an extra model and sampling budget during evaluation.

## 7. Implications and Future Directions
- Field impact
  - Demonstrates that unified, native multimodal pre-training can replace multi-stage post-hoc adaptation, simplifying pipelines while preserving or improving language skill (Figure 3; Table 11). This can shift the community toward single-stage, mixed-data pre-training for MLLMs.
  - V2PE offers a practical recipe to extend multimodal context without enlarging the model, likely influencing future positional encoding designs for MLLMs (Section 2.1; Table 12).

- Follow-up research
  - Data-targeted improvements: Add grounding-specific corpora and richer hallucination-robust data; probe how the `L_text-only` objective interacts with tasks that benefit from visual-token prediction.
  - Position strategies: Learn δ or schedule it adaptively per input instead of sampling from a fixed set; explore cross-modal relative position schemes.
  - Process supervision: Expand VisualPRM beyond math/science into multi-image reasoning and long video; investigate training-time integration (e.g., reinforcement learning with process rewards).
  - Efficiency: Combine BoN with smarter sampling (e.g., conditional early stopping, diversity-promoting decoding) to reduce test-time cost.

- Applications
  - Strong results suggest readiness for:
    - Document and chart analysis (Table 3: DocVQA 95.4, ChartQA 89.7).
    - Enterprise OCR and form understanding (OCRBench 906).
    - GUI agents and UI automation (Table 9: up to 90.9 on ScreenSpot-V2).
    - Spatial reasoning for robotics/autonomy (Table 10: top object counting/relative distance/appearance order).
    - Long video understanding for surveillance, sports, or education (Table 8: strong scaling on MLVU/LongVideoBench).
  - Public release of code, data, and weights (front page; Figure 1 footers) lowers barriers for practitioners and researchers to build and evaluate next-generation open-source MLLMs.

> Headline result: “InternVL3‑78B reaches MMMU 72.2, OCRBench 906, AI2D 89.7, ChartQA 89.7, DocVQA 95.4” (Figure 1; Tables 2–3), with compelling language-only competence (Table 11) and broad real-world robustness (Tables 4–5), achieved via a unified training paradigm plus V2PE, MPO, and test-time process supervision.
