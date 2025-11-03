# SAIL-VL2 Technical Report

**ArXiv:** [2509.14033](https://arxiv.org/abs/2509.14033)

## 🎯 Pitch

SAIL-VL2 introduces a next-generation open vision–language model suite that achieves state-of-the-art multimodal understanding and reasoning at small parameter scales (2B/8B), leveraging a highly curated data pipeline, a progressively aligned vision encoder, and a hybrid supervised fine-tuning plus reinforcement learning strategy. This innovation matters because it delivers top-tier perception and complex reasoning—across images, documents, and video—at a fraction of the computational cost, paving the way for practical, scalable, and efficient deployment of advanced AI in diverse real-world applications.

---

## 1. Executive Summary
SAIL-VL2 is an open-suite vision–language model (LVM) family that aims to deliver “small-model, strong-performance” multimodal understanding across images and video. It combines a progressively trained vision encoder (`SAIL-ViT`), a data-centric pretraining pipeline, and a thinking-fusion SFT–RL (supervised fine-tuning plus reinforcement learning) recipe to reach state-of-the-art results at 2B and 8B scales on 106 benchmarks, and top open-source scores on the OpenCompass multimodal reasoning leaderboard for its 8B “thinking” model (Table 10; Sections 1, 6.2.3).

## 2. Context and Motivation
- Gap addressed
  - Scaling multimodal models has delivered strong results, but often at high computational and deployment cost. The paper targets efficient architectures and training strategies that inject knowledge effectively without relying on very large dense models (Introduction).
  - Existing LVMs can struggle simultaneously with fine-grained perception (OCR, charts, documents), multi-image/video temporal reasoning, and complex step-by-step reasoning. SAIL-VL2 aims to cover this breadth with models at 2B/8B/A3B scales (Figures 1–2; Tables 1, 8–10).
- Why this matters
  - Practical deployment: smaller models with competitive performance reduce latency and cost.
  - Breadth: fine-grained perception (e.g., OCRBench, DocVQA), multi-image/video understanding, and complex reasoning (MathVista, MathVerse, LogicVista) are common real-world needs (Tables 8–10).
- Prior approaches and limitations
  - Previous open-source LVMs (e.g., Qwen2.5-VL, InternVL3/3.5, Ovis2/U1) show strengths but typically trade off either perception fidelity, efficiency, or reasoning depth at small parameter scales (Tables 8–9).
  - Reasoning-oriented variants exist, but often with larger parameter counts or limited video/multi-image coverage (Table 10).
- Positioning
  - SAIL-VL2 pursues an efficiency-first design via:
    - A progressively aligned vision encoder (`SAIL-ViT`) that reduces the modality gap to language (Section 2.1; Figure 6; Table 7).
    - A data pipeline that scores/filters massive caption, OCR, QA, and video corpora (Figure 3; Sections 3.1, 4.1).
    - A training recipe that blends pretraining, Long-CoT SFT, and two RL stages to shape robust “thinking” strategies (Section 4; Table 4).

## 3. Technical Approach
This section walks through the full pipeline, from architecture to training and infrastructure.

- Architecture overview (Figure 2; Table 1)
  - Vision encoder: `SAIL-ViT` (Section 2.1), a ViT-based encoder with two variants:
    - Fixed-resolution (448×448) encoder; high-resolution images are tiled into 448 crops (Section 2.1.2).
    - `SAIL-ViT-AnyRes`: accepts arbitrary image resolutions by interpolating positional embeddings to match input resolution; tokens scale with resolution (Section 2.1.2; “Interpolate 2D RoPE” shown in Figure 2).
  - Adapter: a lightweight two-layer MLP maps vision features into the LLM’s token space (Section 2).
  - Language backbone: Qwen3 series in dense (1.7B, 8B) and MoE (30B total with ~3B active “A3B”) configurations (Table 1). Visual and text tokens are concatenated and processed together for auto-regressive prediction (Figure 2).

- Progressive `SAIL-ViT` training (Section 2.1.1)
  - Goal: align visual features with LLM space step-by-step.
  - Stage I (warm-up): freeze ViT and LLM, train adapter only on 8M simple caption + OCR samples (1 epoch, LR 2e-4, batch 1920).
  - Stage II (fine-grained alignment): unfreeze ViT+adapter (LLM frozen), broaden data (extra 6.7M captions, more OCR, plus video-caption) with LR 2e-5, batch 512.
  - Stage III (world knowledge injection): unfreeze all modules (ViT, adapter, LLM) and train on 36.5M diverse tasks (caption/OCR/QA/math/text) at LR 1e-5, batch 512 (1 epoch). The resulting encoder is the released `SAIL-ViT`.

- Pretraining data curation at scale (Sections 3.1–3.1.2; Figure 3)
  - `SAIL-Caption2` (Section 3.1.1):
    - Quality scoring with two custom judge models for Visual Information Richness (VIR) and Image–Text Alignment (ITA) trained on SAIL-VL-1.5 (Table 2). Precision/recall >0.90; used to filter 300M captions to 250M high-quality pairs.
    - Adds 1.69M chart captions via code-based chart rendering + open datasets (e.g., DVQA) with annotations/QAs (Section 3.1.1).
  - Synthetic VQA: convert ~80M captions into multi-QA pairs using a strong LVM API; scaling trends hold up to 180M samples (Section 3.1.2; Figure 4).

- Pretraining recipe (Section 3.2; Table 3; Figure 4)
  - Two stages:
    1) Basic multimodal pretraining on 64M samples (caption + OCR). Initialize LR at 2e-4 and let `AdaLRS` search a better LR online; it increases effectively to 6.75e-4 and reduces final loss by >0.06 over a fixed-LR baseline (Section 3.2.1).
       - `AdaLRS` (Section 3.2.3; Eq. (1)): a backtracking line-search on training-loss slope. If the estimated slope improves when LR is scaled up (by α′), keep it; otherwise rollback and scale down (by β′). This converges toward an LR neighborhood where loss descent is fastest.
    2) Multi-task pretraining on 180M samples (visual understanding + instruction-tuning + text math), unfreezing all parameters. No `AdaLRS` here because loss–performance correlation is weak on instruction data (Section 3.2.1).
  - Data resampling: dataset-level balancing in stage 1 and n-gram–level linguistic balancing in stage 2 to avoid language homogenization (Section 3.2.2).
  - Scaling: extending multi-task tokens to 360B yields smooth, monotonic gains across general, natural VQA, and OCR VQA (Figure 4).

- Post-training (reasoning-centric) pipeline (Section 4; Table 4)
  - Basic SFT in phases (Section 4.2.1):
    - Phase 1: general instruction-following (Infinity-MM Stage2).
    - Phase 2: high-quality visual instructions (`SAIL-Instruction2`, 20M samples curated via latent-class clustering and re-annotation; Section 4.1.2; Figure 5).
    - Phase 3: harder reasoning subsets (e.g., LLaVA-CoT, MMPR, Condor).
    - Phase 4: balanced image:video mixture (1:1) using filtered `SAIL-Video` (5.1M high-quality video-QA after alignment, richness, difficulty scoring; Section 4.1.1).
  - Model soup: average merge of homogeneous checkpoints improves metrics consistently, while heterogeneous merges can catastrophically degrade (Table 5).
  - LongCoT SFT (Section 4.2.2):
    - Build a 400K Long-CoT corpus (e.g., VisualWebInstruct, MathV360K, LLaVA-CoT), normalize format: `<think>…</think>` and final answer in `\boxed{…}`; filter redundant/overly trivial chains and balance lengths.
    - Loss is standard next-token prediction over thought plus answer: Eq. (2).
  - RL with verifiable rewards (Section 4.2.3):
    - Data: difficult but solvable problems from Math, Puzzle, Science, OCR, Counting; MCQs converted to free-response; filtered by pass@4 (retain neither trivial nor impossible). Section text reports “70K stem samples.”
    - Rewards: binary Answer correctness (checks `\boxed{…}`) and Format correctness (presence of `<think>` tags).
    - Optimization: PPO-based; `DAPO` for dense models and `GSPO` for MoE; 16,384 context; 4,096 generation; 2,048 rollouts per episode; 8 updates; LR 1e-6; adaptive clip 0.20–0.28.
  - Think-Fusion SFT (Section 4.2.4):
    - Train on 1M instances: 90% direct QA, 10% high-quality CoT harvested from prior RL via rejection sampling (100K CoT + 900K direct). Dual-target loss (Eq. (3)) teaches both concise answers and step-by-step reasoning.
  - RL with mixed rewards (Section 4.2.5):
    - Data: “hard cases” from previous RL (reported as 50K) + 50K general LLaVA-OneVision, for 100K total (the section also mentions “curated 150K,” a minor inconsistency).
    - Rewards: weighted combination of Answer, Think, and Format rewards; for unverifiable tasks, an LVM judge provides the answer reward.
    - Same PPO setup as verifiable RL; observation: model learns to reach correct answers without always emitting full CoT unless asked.

- MoE design and calibration (Section 2.2)
  - Qwen3-based sparse MoE layers with load-balancing auxiliary loss; “expert activation entropy” is tuned via data probing on text-only calibration sets to stabilize activation patterns when moving to multimodal inputs.

- Training infrastructure (Section 5)
  - Stream packing: online concatenation of variable-length samples to minimize padding; balanced micro-batches; periodic inclusion of long samples (Section 5.1).
  - Visual packing: balance the number of vision tokens per device, especially for `AnyRes` inputs (Section 5.1).
  - Reported gains: ~2× SM utilization; +50% speed from data packing; +48% efficiency from visual packing; and +0.7% average metric improvement (Section 5.1).
  - MoE kernels and parallelism: fused kernels yield up to 3× speedup; on NPUs use Megatron-style pipeline/expert parallelism; on NVIDIA, DeepSpeed ZeRO-2 with CPU offload (Section 5.2).

## 4. Key Insights and Innovations
- Progressive visual–language alignment with `SAIL-ViT` (Sections 2.1.1–2.1.2)
  - What’s new: a three-stage regimen that starts by adapting the adapter only, then jointly tuning vision + adapter, and finally injecting broad “world-knowledge” with all modules unfrozen.
  - Why it matters: it directly targets the modality gap. Figure 6 and Table 7 show `SAIL-ViT` features lie closer to LLM embeddings than a strong baseline (`AIMv2`), across several distance metrics, explaining easier fusion in the LLM.
- High-throughput, quality-controlled data pipeline (Sections 3.1, 4.1; Figure 3)
  - What’s new: quality scoring on VIR and ITA with in-house judges (Table 2), chart rendering to generate diverse chart-image/QA, and latent-class bucketing for instruction data (`SAIL-Instruction2`).
  - Why it matters: it systematically removes noisy captions, diversifies domains (charts/tables), and balances instruction styles. Figure 5 shows `SAIL-Instruction2` yields consistently higher SFT performance at the same data budgets compared to prior instruction sets.
- `AdaLRS` for fast pretraining convergence (Section 3.2.3; Eq. (1))
  - What’s new: an online learning-rate search that rolls parameters and optimizer state back when LR increases hurt the loss slope.
  - Why it matters: in basic multimodal pretraining, `AdaLRS` increases the practical LR from 2e-4 to ≈6.75e-4 and beats a fixed-LR baseline by >0.06 final loss (Section 3.2.1), improving training efficiency without manual sweeps.
- Thinking-fusion training stack (LongCoT SFT → RL → Think-Fusion SFT → RL) (Section 4; Table 4)
  - What’s new: a cyclic SFT–RL design that (i) teaches long-chain reasoning with standardized output tags, (ii) enforces correctness via verifiable rewards, (iii) rebalances toward concise answers while preserving reasoning via rejection-sampled CoT examples, and (iv) finishes with mixed rewards to improve both answer quality and reasoning soundness.
  - Why it matters: it yields top open-source scores on the OpenCompass reasoning suite for the 8B-think model (average 54.4 in Table 10), approaching GPT-4o-latest on that benchmark.
- Practical efficiency measures for MoE and variable-length multimodal inputs (Sections 2.2, 5)
  - What’s new: expert-entropy calibration, fused expert kernels, and visual token balancing.
  - Why it matters: stabilizes MoE scaling and reduces computation/communication overhead, while stream/visual packing improves utilization and even final performance (+0.7%) (Section 5.1).

## 5. Experimental Analysis
- Evaluation setup (Section 6.1)
  - Models: `SAIL-VL2-2B/8B/A3B`, plus `AnyRes-2B`, and “thinking” variants for 2B/8B/A3B (Section 6.1).
  - Benchmarks: 106 datasets across:
    - General multimodal understanding (72 sets),
    - OpenCompass (8 sets),
    - Video understanding (9 sets) (Section 6.1).
  - Judging:
    - For non-thinking: re-evaluate baselines with a consistent LLM judge (Doubao-1.5-vision-pro) using a customized VLMEvalKit (Section 6.1).
    - For thinking: scores are from the OpenCompass leaderboard; for two models not present officially, a customized VLMEvalKit with GPT-4o-Mini is used under OpenCompass settings (Section 6.1).
  - Video evaluation: 16 randomly sampled frames per video (Tables 8–9 notes).

- Main quantitative results (highlights with head-to-head numbers)
  - Sub-4B regime (Table 8):
    - “OpenCompassavg” (aggregated across eight datasets): `SAIL-VL2-2B` 70.31 vs Qwen2.5-VL-3B 65.36 and InternVL3.5-2B 66.64.
    - OCR/Docs: `OCRBench` 89.50 (2B) and `DocVQA` 93.10—both leading among <4B models listed.
    - Visual grounding: `AnyRes-2B` reaches 57.82 on RefCOCOavg vs 53.28 for fixed-res 2B (Table 8), indicating `AnyRes` helps fine-grained localization.
    - Math & reasoning (non-thinking models): `SAIL-VL2-2B` scores 28.90 overall, roughly on par with Ovis-U1-3B (28.49) and InternVL3.5-2B (28.86), despite fewer parameters.
  - 8B class and A3B MoE (Table 9):
    - “OpenCompassavg”: `SAIL-VL2-8B` 75.07, surpassing Qwen2.5-VL-7B (70.62) and InternVL3.5-8B (73.49).
    - OCR/Docs: `OCRBench` 91.30 and `DocVQA` 95.28—both strong; `DocVQA` is the highest in Table 9, while `OCRBench` is near the top.
    - Multi-image/video: overall averages are competitive (e.g., `TempCompassavg` 65.66 and `LongVideoBenchval` 58.34).
    - RefCOCOavg: 74.02 for 8B (strong localization).
  - Reasoning (OpenCompass reasoning suite; Table 10):
    - `SAIL-VL2-8B-Thinking` achieves the top open-source average 54.4, with 75.8 on MathVista and 56.4 on LogicVista.
    - `SAIL-VL2-A3B-Thinking` (MoE with ~3B active) hits 53.6—competitive with and often above other open-source MoEs; close to Gemini-2.0-Pro/Flash averages and near GPT-4o-latest on this suite.
    - `SAIL-VL2-2B-Thinking` scores 40.9—smaller but still strong for its scale.

- Component studies and analyses
  - `SAIL-ViT` visual classification (Table 6): average accuracy gains vs `AIMv2` baselines at similar scales (e.g., +1.5% for “Large”, +2.11% for “Huge”), though a much larger InternViT-6B still scores higher overall (67.81 avg).
  - Multimodal feature affinity (Figure 6; Table 7): across multiple LLM sizes, `SAIL-ViT` reduces nearest-neighbor and Wasserstein distances to the text-embedding cluster compared to `AIMv2`, corroborating improved cross-modal alignment.
  - Scaling law (Figure 4): smooth, monotonic improvements as multi-task pretraining data scales to 360B tokens across “Overall,” Natural VQA, and OCR-VQA slices.
  - Model soup (Table 5): homogeneous merges consistently lift averages (e.g., “AVG” from 74.91/74.54 to 76.60); heterogeneous merges can fail dramatically (merge “AVG” 12.86), cautioning against indiscriminate merging.

- Do the experiments support the claims?
  - Efficiency with strong performance: yes, at 2B and 8B, results are consistently competitive or SOTA versus larger open-source baselines across perception and understanding tasks (Tables 8–9).
  - Reasoning advances: clear gains for “thinking” variants, especially 8B-think at the top of OpenCompass open-source scores (Table 10).
  - Breadth: coverage over OCR/document, charts, grounding, multi-image, and video with consistent evaluation notes (Tables 8–9).

## 6. Limitations and Trade-offs
- Data dependence and potential bias
  - Heavy reliance on LVM APIs for label generation, caption scoring (ITA/VIR), and reward judging for unverifiable tasks (Sections 3.1.1, 3.1.2, 4.2.5). This can propagate the annotator model’s linguistic biases and style homogenization despite n-gram balancing (Section 3.2.2).
- Evaluation comparability
  - Non-thinking model evaluations use a customized VLMEvalKit with Doubao; thinking models rely on OpenCompass with GPT-4o-Mini as judge; for two models, the team re-runs under equivalent settings (Section 6.1). Although justified, different judges can influence absolute scores.
  - For RefCOCO, prompt formatting and instruction-following issues caused one strong baseline (Kimi-VL-A3B) to be excluded on that metric (Table 9 notes), complicating direct comparison for that task.
- Numerical inconsistencies
  - Minor mismatches in counts for RL datasets between sections (e.g., 50K vs 70K vs 100–150K; Sections 4.1.3, 4.2.3, 4.2.5) and token summaries in Table 3 vs Introduction (“trained on 776B tokens”). These do not undermine the main results but warrant clarification for reproducibility.
- Complexity and compute
  - The full pipeline is intricate (three-stage `SAIL-ViT`, two-stage pretraining with `AdaLRS`, multi-phase SFT, two RL stages, model soup, MoE calibration). While more efficient than very-large dense models, end-to-end training remains compute-intensive (Sections 2–5).
- MoE stability and distribution shift
  - The paper mitigates expert imbalance via auxiliary losses and data-aware calibration (Section 2.2), but MoE routing can still be sensitive to domain shifts; robustness across highly novel inputs is not deeply analyzed.
- Video modeling
  - Video results are competitive but not dominant across the board (e.g., Table 9 “Video-MME” 62.70; “TempCompassavg” is strong but varies by model). Only 16 frames are sampled, which may miss long-range temporal cues (Tables 8–9 notes).

## 7. Implications and Future Directions
- Field impact
  - Demonstrates that carefully engineered pipelines—progressive visual–language alignment, high-quality multimodal data, and staged SFT–RL—can make smaller LVMs attain broad SOTA-level performance. This rebuts the idea that only massive dense models can be strong generalists at multimodal tasks (Figures 1–2; Tables 8–10).
- Practical applications
  - OCR-heavy and document workflows (DocVQA, OCRBench), chart/table understanding, retail/screenshots UI analysis, and enterprise tools benefit directly from the strong fine-grained perception (Tables 8–9).
  - Education, step-by-step tutoring, and technical support chatbots can leverage the “thinking” variants for math/logical reasoning (Table 10).
  - Video QA and multi-image analytics for surveillance, sports, or media summarization are supported, though further temporal modeling could help.
- Follow-up research
  - Training transparency and reproducibility: publish exact token counts per stage and reconcile dataset-size discrepancies.
  - Better reward models for reasoning: move beyond binary rewards (0/1) to graded signals; explore process-level verifiers tailored to multimodal content.
  - Robustness and safety: stress tests for hallucinations under domain shift; more comprehensive hallucination auditing beyond HallusionBench (Tables 8–9).
  - MoE routing under multimodal distributions: deeper analysis of expert specialization across vision-text domains and time (video).
  - Video scaling: richer temporal sampling policies or long-context architectures to push long-video benchmarks.
  - Data diversification: reduce reliance on closed-source APIs for scoring/annotations; invest in community-governed, audited datasets.

> Bottom line: SAIL-VL2 shows that an efficiency-first recipe—progressive `SAIL-ViT` alignment, a rigorous data pipeline (`SAIL-Caption2`, `SAIL-Instruction2`, curated video), and a staged SFT–RL “thinking” procedure—can deliver leading open-source multimodal performance at 2B/8B scales, with especially strong reasoning in the 8B-think model (Tables 8–10; Sections 2–4, 6).
