# An Empirical Study of Scaling Instruct‑Tuned Large Multimodal Models

**ArXiv:** [2309.09958](https://arxiv.org/abs/2309.09958)
**Authors:** Yadong Lu, Chunyuan Li, Haotian Liu, Jianwei Yang, Jianfeng Gao, Yelong Shen
**Institutions:** Microsoft Research (likely, given authors)

## 🎯 Pitch

This paper revolutionizes large multimodal model training by scaling language backbones from 7B to 70B parameters, optimizing multimodal and language capabilities with cost-effective tuning methods. By showcasing how increased image resolution and strategic data mixing enhance performance, this study provides critical insights for practitioners looking to maximize AI model efficacy without prohibitive resource investment.

---

## 1. Executive Summary
This paper systematically scales the open-source Large Language-and-Vision Assistant (`LLaVA`) from small language backbones (7B/13B parameters) to much larger ones (33B and 65B/70B) and analyzes what actually matters for performance. It disentangles the effects of model size, image resolution, data mixing (adding language-only instruction data), and parameter-efficient tuning (LoRA/QLoRA), showing how to get near–full fine-tuning quality at much lower cost and how visual instruction tuning can sometimes even improve pure language ability.

## 2. Context and Motivation
- Problem addressed:
  - Open-source large multimodal models (LMMs) like LLaVA and MiniGPT-4 showed promising visual instruction-following, but almost all studies evaluated only 7B–13B language backbones due to compute limits. The impact of truly scaling model size (e.g., 33B, 65B/70B) on multimodal and language capabilities remained unclear.
- Why this matters:
  - Practical: If scaling consistently improves real-world multimodal assistants without prohibitive cost, it guides practitioners on where to invest compute (e.g., model size vs. data resolution vs. tuning method).
  - Scientific: Understanding how scaling interacts with data and training choices clarifies what drives LMM capability—language capacity, visual resolution, or instruction data composition.
- Prior work and limitations:
  - LLaVA and MiniGPT-4 showed strong results with small backbones and short training (e.g., LLaVA-7B trained for ~3 hours) but left scaling effects largely unexplored (Introduction, p.1).
  - Other open LMMs (BLIP-2, OpenFlamingo, Otter, InstructBLIP) use various recipes but do not isolate the contribution of model scale, resolution, and data mixing at larger scales (Table 2 context).
- Positioning:
  - The paper performs an empirical scaling study of LLaVA to 33B and 65B/70B backbones and evaluates:
    - Model size (7B/13B → 33B → 65B/70B),
    - Image resolution (224×224 vs. 336×336),
    - Data mixing (multimodal-only vs. adding language-only ShareGPT),
    - Tuning method (full fine-tuning vs. LoRA/QLoRA).
  - It reports end-to-end results on LLaVA-Bench and MM-VET, plus language-only tests (Vicuna-80 and MMLU), establishing stronger baselines for large-scale open LMMs (Sections 2–3).

## 3. Technical Approach
This is an empirical study built on a two-stage LLaVA training pipeline with systematic variations.

- Backbone checkpoints (Section 2: “Model Checkpoints”):
  - `LLaVA-33B`: built on `Vicuna-33B v1.3`.
  - `LLaVA-65B`: trained by the authors because a public 65B Vicuna checkpoint was unavailable; uses 159M tokens of ShareGPT-style data for the Vicuna-65B language backbone (compared to ~370M tokens used for Vicuna-33B).
  - `LLaVA-70B`: built on `LLaMA-2-70B-Chat` for an additional experiment (Table 5).

- Two-stage training (“lightning training,” Section 2):
  1) Stage 1 — Feature alignment:
     - Purpose: let the language model accept visual features as if they were word embeddings.
     - Mechanism: train a single linear projection that maps the frozen image encoder’s features into the LLM’s embedding space. Specifically:
       - Projection dimensions: 1024 → 6656 (33B), 1024 → 8192 (65B).
       - Image encoder: frozen CLIP ViT (features before the last layer).
       - Data: 558K samples from a “concept-balanced subset” of LAION-CC-SBU (internet-scale image–text data).
     - Hyperparameters: LR 1e-4 (linear warmup over 3% of steps), no weight decay, sequence length 2048, optimized with DeepSpeed ZeRO-3 (ZeRO-2 for QLoRA runs).
  2) Stage 2 — Visual instruction tuning:
     - Purpose: teach the system to follow multimodal instructions and hold visual chats.
     - Data: LLaVA-80K multimodal instruction dataset.
     - Variants:
       - Multimodal-only vs. “Data mixing”: mixing in language-only instruction data (`ShareGPT`) together with LLaVA-80K to trade off language and multimodal capabilities.
     - Trainable modules:
       - Full-model fine-tuning (all LLM weights + the projection).
       - Parameter-efficient fine-tuning:
         - `LoRA`: inserts low-rank adapters into attention/FFN modules so only a small number of parameters are trained.
         - `QLoRA`: quantizes the base weights (e.g., 4-bit) and trains adapters, enabling larger backbones to fit into memory.
       - Tuning details:
         - For LoRA, a large LR (1e-4) and `LoRA alpha = 2 × rank` worked best across model sizes.
         - Full fine-tuning LR = 2e-5 (1 epoch).
         - Batch and hardware: full FT uses total batch size 512 on 4 nodes of 8×A100-80G; LoRA/QLoRA uses total batch size 64 on 1 node (33B) or 2 nodes (65B).
     - Why this setup:
       - Aligns with the successful LLaVA recipe while enabling controlled experiments on scaling, resolution, data composition, and tuning efficiency.

- Resolution and inference settings (Section 3.2):
  - Image resolution: a controlled comparison with identical CLIP ViT encoders pretrained at 224×224 vs. 336×336 to isolate resolution effects (Table 3a).
  - Decoding: beam search with beam sizes 1 and 5; observed latency is similar, but beam 5 noticeably improves performance on LLaVA-Bench (Table 1).

## 4. Key Insights and Innovations
- Scaling the LLM backbone consistently helps, especially on knowledge and generation-heavy tasks.
  - What’s new: a controlled, large-scale comparison up to 65B/70B across multiple benchmarks with the same LLaVA recipe.
  - Why it matters: it validates that “language-side capacity” is the primary driver for improved multimodal reasoning and open-ended generation rather than only improving the vision encoder.
  - Evidence:
    - On MM-VET (Table 2), overall score increases from 32.9 (33B) → 35.5 (65B). The largest gains are in Knowledge and Generation categories.

- Higher image resolution (336×336 vs. 224×224) yields consistent gains across model sizes.
  - What’s new: a clean ablation showing resolution alone gives +2–3 points on LLaVA-Bench across 7B, 13B, 33B, 65B (Table 3a).
  - Why it matters: increasing resolution is a low-effort knob that pays off even when the vision encoder remains frozen.

- Data mixing (adding language-only ShareGPT) improves multimodal performance and changes language trade-offs.
  - Novelty: quantifies when and how mixing language-only data helps.
  - Impact:
    - Multimodal: +~2 points on LLaVA-Bench for 33B and 65B (Table 3a), and consistent gains on MM-VET (Table 2).
    - Language: mixed effects—hurts instruction-following on Vicuna-80 for 33B and 65B, but can improve knowledge-heavy MMLU at 33B; see Table 5.

- Parameter-efficient tuning (LoRA/QLoRA) can match full fine-tuning on large backbones with much lower cost.
  - What’s new: a systematic cost–quality analysis on 33B/65B with ranks and quantization.
  - Why it matters: shows practical recipes to train large LMMs without prohibitive hardware.
  - Evidence (Table 4):
    - `13B`: LoRA rank 64 reaches the same 70.1 as full FT.
    - `33B`: LoRA rank 64 (72.0) ≈ full FT (72.2); QLoRA rank 64 gets 71.8 with lower memory/time.
    - `65B`: LoRA rank 64 matches full FT (both 72.3) while using ~32% fewer GPU-hours per node (9.17 vs. 13.50).

- Visual instruction tuning can improve pure language ability at large scale.
  - Significance: challenges the assumption that multimodal tuning dilutes language ability.
  - Evidence:
    - On `LLaMA-2-70B-Chat`, `LLaVA-70B` achieves MMLU 65.1 vs. 63.1 for the base chat model (Table 5), a +2.0 point gain, while maintaining strong multimodal scores.

## 5. Experimental Analysis
- Benchmarks, metrics, and setup:
  - LLaVA-Bench (In-the-Wild) [Section 3.1]:
    - 24 images, 60 questions spanning conversations, detailed descriptions, and reasoning.
    - Scoring uses GPT-4 (gpt4-0314) to compare model output vs. gold responses; relative scores are reported.
  - MM-VET (Table 2):
    - 200 images, 218 questions covering six capabilities: Recognition, OCR, Knowledge, Generation, Spatial, Math.
    - Scored by gpt4-0613; aims to test integrated, combinatorial multimodal skills.
  - MM-Bench (Table 3b):
    - 2,974 questions; evaluates logic, attribute, relation reasoning, and fine/coarse perception categories.
  - Language-only:
    - Vicuna-80: instruction-following on real-world language tasks.
    - MMLU: knowledge-heavy, multi-task academic benchmark.
  - Decoding: Beam search sizes of 1 and 5 reported on LLaVA-Bench (Table 1).

- Main quantitative results:
  - LLaVA-Bench (Table 1):
    - Scaling helps. With beam=5:
      - `LLaVA-13B`: 73.5
      - `LLaVA-33B`: 74.8
      - `LLaVA-65B`: 74.4
    - Beam search helps across sizes:
      - Example: `LLaVA-33B` 72.6 (beam=5) vs. 70.2 (beam=1) on “Conversation.”
    - Larger LLaVA models are “comparable to BingChat in multi-turn, multimodal conversation” and stronger on complex reasoning/long descriptions (Section 3.1 discussion).
  - MM-VET (Table 2):
    - Among end-to-end open models (no tool chaining), scaling improves SOTA:
      - `LLaVA-13B` (LLaMA-2): 32.6
      - `LLaVA-33B`: 32.9
      - `LLaVA-33B` + Mixing: 34.1
      - `LLaVA-65B`: 35.5
      - `LLaVA-65B` + Mixing: 36.4 (best among end-to-end open models in the table)
    - Category breakdown (65B vs. 33B, Table 2):
      - Knowledge: 26.2 (33B) → 26.2 (65B) → 30.4 (65B + mixing)
      - Generation: 28.2 (33B) → 28.3 (65B) → 32.3 (65B + mixing)
      - Recognition/OCR also improve; Spatial/Math are flatter.
  - Image resolution and data mixing (Table 3a):
    - Resolution 224→336 adds ~2–3 points across 7B–65B.
    - Adding language-only data on top of 336×336 yields:
      - `33B`: 72.0 → 73.9
      - `65B`: 72.3 → 74.2
  - MM-Bench (Table 3b):
    - Baseline `LLaVA-7B` overall: 36.2
    - `LLaVA-33B` + 336×336 + data mixing: 55.7
    - `LLaVA-65B` + 336×336 + data mixing: 56.0
    - Gains spread across logic/attribute/relation reasoning and perception categories.
  - Efficiency and tuning method (Table 4):
    - Quote the central trade-off:
      > For 65B, LoRA rank 64 reaches 72.3 on LLaVA-Bench with 9.17 GPU-hours per node, while full fine-tuning also reaches 72.3 but needs 13.50 GPU-hours per node.
    - `33B` provides a strong cost–quality sweet spot: LoRA rank 64 gets 72.0 at 5.80 GPU-hours per node (Table 4).
    - Hyperparameter sensitivity:
      > Using a smaller LR (2e-5) and alpha=16 with rank=64 drops performance from 71.8 to 65.5 on LLaVA-Bench (Section 3.2, point 2).
  - Language ability with multimodal tuning (Table 5):
    - `LLaVA-33B` (no mixing) largely preserves Vicuna-33B’s instruction-following: 85.3 vs. 85.6 on Vicuna-80; MMLU modestly drops (56.1 vs. 59.0).
    - Adding language-only data changes the trade-off:
      - `33B`: Vicuna-80 drops (85.3 → 80.3) while MMLU increases (56.1 → 58.6).
      - `65B`: both Vicuna-80 and MMLU slightly decrease with mixing.
    - `LLaVA-70B` on top of `LLaMA-2-70B-Chat`:
      > MMLU improves from 63.1 to 65.1 (Table 5); multimodal scores (e.g., 69.8 on LLaVA-Bench and 35.4 on MM-VET) remain strong.

- Do the experiments support the claims?
  - Yes, for the core points:
    - Scaling helps: consistently positive on LLaVA-Bench, MM-VET, and MM-Bench.
    - Resolution helps: clear +2–3 points across sizes (Table 3a).
    - Data mixing helps multimodal tasks: clear gains on LLaVA-Bench and MM-VET; language impacts depend on which language metric (Vicuna-80 vs. MMLU) and model size (Table 5).
    - LoRA/QLoRA match full FT at far lower cost: demonstrated across sizes with explicit compute accounting (Table 4).
  - Caveats the paper itself notes:
    - LLaVA-Bench is small; differences may not be statistically significant (Section 3.1).
    - Training datasets are relatively small, so findings are preliminary (Conclusions).

- Ablations, robustness, failure modes:
  - Ablations:
    - Resolution and data-mixing ablations (Table 3a).
    - Tuning method and hyperparameter sensitivity (Table 4 and Section 3.2 point 2).
  - Failure or flat areas:
    - Spatial and Math categories in MM-VET show limited improvements with scaling (Table 2).
    - Language trade-offs depend on the metric and mixing (Table 5).

## 6. Limitations and Trade-offs
- Assumptions and constraints:
  - Frozen vision encoder (CLIP ViT) with limited resolution variants; the study does not scale or adapt the vision backbone beyond 224×224 vs. 336×336 (Conclusions).
  - Stage-1 data is a subset of LAION-CC-SBU (558K); Stage-2 uses LLaVA-80K; total data volume is relatively modest for 33B–70B models (Conclusions).
- Evaluation limitations:
  - LLaVA-Bench is small; even though it’s diverse, results may not be statistically robust (Section 3.1).
  - GPT-4–based evaluators (LLaVA-Bench, MM-VET) introduce potential biases and may reward stylistic alignment.
- Compute and engineering trade-offs:
  - Full fine-tuning at 65B/70B is expensive; LoRA/QLoRA help, but careful hyperparameter tuning is essential to avoid large performance drops (Section 3.2 point 2).
  - LoRA on very large models may hit OOM with ZeRO-3; QLoRA with ZeRO-2 mitigates this (Section 3.2 point 2).
- Capability gaps:
  - Gains are strongest in language-heavy multimodal categories (knowledge, long-form generation), but spatial reasoning and math remain flatter (Table 2).
  - Mixing in language-only data can degrade instruction-following (Vicuna-80), especially at 33B/65B (Table 5).
- Open questions:
  - Optimal ratios and curricula for multimodal–language data mixing.
  - Whether scaling the vision encoder (not just LLM size) yields balanced improvements, especially on spatial/math tasks.

## 7. Implications and Future Directions
- How this changes the landscape:
  - Establishes that scaling the language backbone of open LMMs to 33B–65B is both feasible and beneficial, with clear recipes for practitioners who lack massive compute: use 336×336 resolution, consider moderate beam search, and prefer LoRA/QLoRA with tuned hyperparameters.
  - Demonstrates that multimodal instruction tuning can sometimes enhance language-only capabilities (e.g., MMLU at 70B), encouraging integration rather than separation of multimodal and language training.
- Follow-up research enabled/suggested:
  - Vision-side scaling:
    - Replace or fine-tune the vision encoder; explore higher resolutions, multi-scale features, or better alignment heads to target spatial/math categories.
  - Data strategy:
    - Systematically study data mixtures (ratios, curricula, difficulty), longer-form multimodal reasoning data, and multilingual multimodal corpora.
  - Efficient training:
    - Further optimize parameter-efficient methods (e.g., adaptive ranks, selective module tuning, better quantization) for 70B+ backbones.
  - Evaluation:
    - Larger, more diagnostic multimodal benchmarks; reduce reliance on LLM-as-judge; include safety, robustness, and calibration metrics.
- Practical applications:
  - Enterprise multimodal assistants that require knowledge-grounded visual reasoning (e.g., document OCR + Q&A, product support from images, medical imaging triage).
  - Creative tools needing detailed image descriptions and long-form multimodal generation (e.g., accessibility captioning, educational tutors).
  - Scenarios with constrained compute: QLoRA-based tuning of 33B emerges as a cost–performance sweet spot (Table 4).

Key takeaways to operationalize:
- If you can scale the LLM, do it—performance generally rises, especially for knowledge and long-form generation (Tables 1–2).
- Always use higher input resolution (336×336) when possible; it’s a “cheap” +2–3 points (Table 3a).
- Use LoRA/QLoRA with carefully tuned LR and alpha; rank increases beyond 64 have diminishing returns (Section 3.2 point 2; Table 4).
- Mix in language-only data to boost multimodal scores, but monitor language instruction-following trade-offs (Table 5).
