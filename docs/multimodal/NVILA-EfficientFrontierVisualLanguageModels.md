# NVILA: Efficient Frontier Visual Language Models

**ArXiv:** [2412.04468](https://arxiv.org/abs/2412.04468)

## 🎯 Pitch

NVILA introduces a new family of open visual–language models that set a new benchmark for both accuracy and efficiency by employing a 'scale-then-compress' approach—scaling up input resolutions to capture richer information, then aggressively compressing visual tokens to minimize computational overhead. This innovation is bolstered by data pruning, low-precision training, and quantized inference throughout the model lifecycle, allowing NVILA to meet or surpass top visual–language models on image and video benchmarks while slashing training times and inference latency by up to 5×. The result is a high-performing, resource-friendly VLM architecture poised to unlock real-world deployment in fields like robotics, edge devices, and medical AI—where both performance and speed are critical.

---

## 1. Executive Summary (2–3 sentences)
This paper introduces `NVILA`, a family of open visual–language models (VLMs) designed to set an “efficient frontier” by jointly improving accuracy and end‑to‑end efficiency across training, fine‑tuning, and deployment. The core idea is a “scale‑then‑compress” pipeline: first raise spatial/temporal resolution to capture more information, then aggressively compress visual tokens so computation remains low, complemented by data pruning, low‑precision training, and quantized inference. As a result, `NVILA` matches or surpasses leading open/proprietary VLMs on image and video benchmarks while reducing training time 1.9–5.1× and inference latency/throughput by 1.6–2.8× (Figure 1a–c; Figure 5).

## 2. Context and Motivation
- Problem addressed:
  - Strong VLMs exist, but efficiency has lagged behind accuracy. VLMs are expensive to train (hundreds of GPU‑days even at 7–8B scale), memory‑intensive to fine‑tune, and slow to deploy on limited hardware (Section 1). The reference baseline `VILA-1.5` used 448×448 images and 8–14 frames, discarding detail and underperforming on text‑heavy images and long videos (Section 2.1; Tables 8–9).
- Why it matters:
  - Real‑world applications (robotics, mobile/edge devices, medical imaging) are compute‑ and latency‑constrained. Efficient training lowers entry barriers; efficient inference enables responsive user experiences (Section 1; Figure 5).
- Prior approaches and shortcomings:
  - Many models focus on accuracy via larger backbones or more data; few provide a systematic efficiency methodology across the lifecycle (training → fine‑tuning → deployment). Token reduction methods exist (e.g., Token Merging, TokenLearner, Perceiver Resampler), but have not been shown to move the frontier for VLMs at high resolution/long videos or within a cohesive end‑to‑end recipe (Section 5.2).
- Positioning:
  - `NVILA` builds on `VILA` but redesigns the vision path to support high‑resolution and long‑context inputs efficiently, and adds system‑level accelerations (FP8 training, quantized vision+LLM at inference), curated data pruning, and fine‑tuning guidelines. It presents a complete stack that improves both quality and speed (Sections 2–4; Figure 3; Tables 1–7).

## 3. Technical Approach
`NVILA` is an auto‑regressive VLM with three standard components (Figure 3):
- A vision encoder (SigLIP, a Vision Transformer variant) that converts images/video frames to embeddings.
- A projector (2‑layer MLP) that aligns vision embeddings to the language space.
- A token processor (`Qwen2` LLM of different sizes) that produces text outputs conditioned on visual and textual tokens.

The novelty lies in how the vision path is scaled and then compressed, plus lifecycle‑wide efficiency techniques.

A. Spatial “scale‑then‑compress” for images (Section 2.1.1; Table 1)
- Challenge: Raising resolution improves accuracy but increases tokens and quadratic attention cost.
- Step 1 — Scale with `S2` tiling and `Dynamic-S2`:
  - `S2` (multi‑scale tiling) processes the image at several scales by splitting into 448×448 tiles, running the encoder per tile, stitching feature maps per scale, then concatenating across scales (Section 2.1.1).
  - Problem: `S2` forces square resizing, distorting unusual aspect ratios.
  - `Dynamic-S2` fixes distortion by keeping the largest scale close to the original aspect ratio with tile‑aligned dimensions; all scales are interpolated to the largest scale and concatenated (Section 2.1.1).
  - Effect: Substantial gains on text‑heavy benchmarks. Table 1 shows moving from the `VILA-1.5` baseline to “Scale (Dynamic‑S2)” lifts AI2D 87.0→90.1, DocVQA 61.3→91.1, TextVQA 67.5→77.0, and IM‑10 average 61.2→71.5.
- Step 2 — Compress spatial tokens:
  - Use `spatial-to-channel (STC)` reshaping (e.g., 3×3) to reduce spatial tokens by packing local patches into channels. A naive increase in STC ratio hurts accuracy (roughly −10% DocVQA if done directly, Section 2.1.1).
  - Remedy: Add a `visual encoder pre‑training (VEP)` stage (Table 7, Stage 2) that jointly tunes the vision encoder and projector under the compressed setup. This recovers most accuracy while yielding 2.4× speedups in training/inference (Section 2.1.1; Table 1 “Scale + Compress + VEP”: IM‑10 rises from 67.1→70.8).
  - Why not TokenLearner/Perceiver? With the same token reduction, these learnable compression modules did not outperform simple STC in this training recipe; likely an optimization/training stability issue (Table 1, “Alternative Designs”).

B. Temporal “scale‑then‑compress” for video (Section 2.1.2; Table 2)
- Step 1 — Scale frames:
  - Uniformly sample more frames (e.g., 8→32) and add video‑supervised instruction tuning to teach the model to use longer temporal contexts. This improves Video‑MME overall 55.7→61.0 (Table 2).
- Step 2 — Compress temporally:
  - Use `temporal averaging` (a simple pooling) within frame groups to exploit redundancy across adjacent frames, reducing tokens 4× with only modest accuracy loss: “Scale + Compress (32 frames, 4× pooling)” keeps overall Video‑MME 61.0→60.1 (Table 2).
  - The same approach scales to 256 frames with 8× pooling while improving accuracy relative to 32‑frame setups (Table 2, last row: overall 64.0).

C. Efficient training (Section 2.2)
- `Dataset pruning with DeltaLoss` (Section 2.2.1; Figure 4; Table 3):
  - Goal: Prune supervised fine‑tuning (SFT) data to remove examples that are too easy or distractingly hard, keeping those most informative for learning.
  - Mechanism (Eq. 1): Compute per‑example score log[p_large(x)/p_small(x)] on answer tokens across sub‑datasets, pick top‑K per subset. Intuition:
    - Near 0: both models agree (either both right or both wrong) → low value.
    - Negative: small model right but large wrong → distracting.
    - Positive: large right but small wrong → challenging yet learnable (helpful).
  - Results: Pruning 50% with `DeltaLoss` maintains accuracy almost unchanged relative to 100% data and beats random/cluster pruning on IM‑10, MMMU, DocVQA, TextVQA (Table 3; e.g., IM‑10 75.6→75.5 vs random 74.0).
- `FP8 training with COAT` (Section 2.2.2; Table 4):
  - `FP8` is an 8‑bit floating‑point format supported on NVIDIA Hopper/Blackwell GPUs.
  - Training setup uses FP8 for both weights and activations (COAT) and leverages that VLM batches have highly variable sequence lengths; under‑utilized batches benefit from larger batch sizes (Table 4).
  - Measured on 64× H100: without gradient checkpointing (GC), FP8 raises batch size 4→16 and roughly doubles throughput (199→390 it/s) with similar accuracy. With GC, FP8 still adds ~1.2× throughput (492→580 it/s) at unchanged accuracy (MMMU and Video‑MME nearly equal).
- Other system choices:
  - FlashAttention‑2, DeepSpeed sharding, functional‑preserving sequence packing, and a 5‑stage training curriculum (Table 7) together streamline compute and memory.

D. Efficient fine‑tuning (Section 2.3; Table 5)
- Recipe insights:
  - Use different learning rates for ViT vs LLM; ViT benefits from 5–50× smaller LR than LLM.
  - Tuning only ViT LayerNorm parameters (much cheaper than LoRA) can match LoRA’s performance for many tasks. Best practice: LoRA (or QLoRA) on LLM + ViT LayerNorm with a small LR, selecting LR ratios from {1,5,10,50} per task (Table 5).
  - Memory/throughput: QLoRA halves memory vs LoRA (e.g., 11.1 GB vs 20.1 GB) at modest throughput cost while preserving accuracy after LR tuning.

E. Efficient deployment (Section 2.4; Figure 5; Table 6)
- Two inference phases:
  - `Prefilling`: encode all inputs (vision + prompt) and build attention caches; compute‑bound.
  - `Decoding`: generate tokens step‑by‑step; memory‑bound.
- Optimizations:
  - Compress visual tokens (as above), then the vision tower becomes the prefill bottleneck (>90% latency). Apply `W8A8` quantization to the vision tower to reduce Time‑to‑First‑Token (TTFT), and `W4A16` (AWQ) to the LLM for decoding, with an improved GEMM kernel using FP16 accumulation for an extra 1.7× kernel speedup (Section 2.4; Figure 5).
  - Quality/latency trade‑off: Table 6 shows `W4A16` on the LLM causes small accuracy drops (e.g., MMMU 50.7→49.2), while `W8A8` on ViT is nearly lossless.
- Measured speedups vs `Qwen2‑VL` on a single RTX 4090 (Figure 5):
  - Prefilling TTFT speedup up to 2.22× for video and 1.55× for images.
  - Decoding throughput speedup up to 2.84× for video and 1.24× for images.

F. Training curriculum and data (Section 3.1; Table 7; Table A1)
- 5 stages (Table 7): projector init → visual encoder pre‑training (`VEP`) → token processor pre‑training → image instruction tuning → video instruction tuning (extends long‑video capability).
- Implementation: PyTorch 2.3, Transformers 4.46, DeepSpeed 0.9.5, FlashAttention‑2, gradient checkpointing, sequence packing; trained on 128× H100 with global batch 2048 (Section 3.1).
- Data mixture: curated across recaptioned corpora, documents/OCR, interleaved multi‑modal, chart/diagram, general VQA, text‑only instruction, medical, and video SFT (Table A1).

## 4. Key Insights and Innovations
- “Scale‑then‑compress” vision path (Sections 2.1.1–2.1.2; Tables 1–2)
  - What’s new: Treat high spatial/temporal resolution as a first‑class capability, then compress tokens to keep cost low. The addition of `Dynamic-S2` addresses aspect ratio distortion, and `VEP` stabilizes training under heavy spatial compression.
  - Why it matters: Large accuracy gains on text‑rich images and long videos with roughly the same token budget as low‑resolution baselines. Table 1 shows up to ~30‑point gains on DocVQA (61.3→91.1) before compression; with compression and `VEP`, most gains remain while tokens are cut substantially.
- Lifecycle‑wide efficiency recipe (Sections 2.2–2.4; Tables 3–6; Figure 5)
  - Data: `DeltaLoss` pruning removes 50% of SFT data with negligible accuracy loss (Table 3).
  - Training: FP8 with COAT increases batch size/throughput without hurting accuracy (Table 4).
  - Inference: Joint `W8A8` (ViT) + `W4A16` (LLM) with custom kernels meaningfully improves TTFT and decoding throughput (Figure 5; Table 6).
  - Significance: The combined stack reduces training time by 1.9–5.1× and inference latency/throughput by 1.6–2.8× (Figure 1a–b).
- Simple compression beats complex modules here (Table 1)
  - STC reshaping (with `VEP`) outperforms TokenLearner and Perceiver Resampler at the same reduction ratio in this recipe—highlighting optimization/stability as central for learnable compressors at scale.
- New capabilities via efficient long context (Section 4; Tables 10–12)
  - Temporal localization with discrete time tokens (Table 10), robotics navigation at 1 Hz on a laptop GPU (Figure 6; Table 11), and medical multi‑tasking when paired with expert models (Table 12).

## 5. Experimental Analysis
- Evaluation setup
  - Image benchmarks: AI2D, ChartQA, DocVQA, InfoVQA, MathVista, MMMU (zero‑shot CoT), RealWorldQA, SEED, TextVQA, VQAv2 (Section 3.2.1; Table 8).
  - Video benchmarks: ActivityNet‑QA, LongVideoBench, MLVU, MVBench, NExT‑QA, Video‑MME (with/without subtitles) (Section 3.2.2; Table 9).
  - Efficiency: Speed comparisons vs LLaVA‑OneVision (training) and Qwen2‑VL (inference), measured on H100 for training and single RTX 4090 for inference (Figure 1a–b; Figure 5).
- Main results (selected highlights)
  - End‑to‑end efficiency:
    - > “NVILA trains image and video models 5.1× and 1.9× faster than LLaVA‑OneVision” (Figure 1a).
    - > “Prefilling 1.6–2.2× faster; decoding 1.2–2.8× faster than Qwen2‑VL” (Figure 1b; Figure 5).
  - Image accuracy (Table 8):
    - `NVILA-8B` achieves AI2D 92.3 (best among opens), DocVQA 93.7, TextVQA 68.6, VQAv2 85.4. `NVILA-15B` further improves AI2D 94.1, ChartQA 86.9, DocVQA 94.0, InfoVQA 73.5.
    - Against strong opens (InternVL2‑8B, Qwen2‑VL‑8B), `NVILA-8B` is competitive or better on several image and OCR tasks; vs proprietary models, `NVILA-15B` is competitive on multiple datasets.
  - Video accuracy (Table 9):
    - `NVILA-8B (256 frames)` reaches LongVideoBench 3.7, MLVU val 57.7/test 58.7, MVBench 70.1, NExT‑QA 68.1, Video‑MME 64.2 (w/o subtitles) and 70.0 (with subtitles), surpassing or matching prior open models of similar/larger sizes; notably close to GPT‑4o mini on several metrics.
- Ablations that support claims
  - Spatial ablations (Table 1):
    - “Scale” via `Dynamic-S2` alone substantially improves text‑heavy tasks (e.g., DocVQA +29.8 points). “Scale + Compress” drops some performance, but “+ VEP” recovers most of it while achieving a 2.4× system speedup.
  - Temporal ablations (Table 2):
    - 8→32 frames improves Video‑MME overall 55.7→61.0; compressing 4× keeps 60.1 overall; scaling to 256 frames with 8× compression further raises overall to 64.0.
  - Data pruning (Table 3):
    - At 50% data, `DeltaLoss` maintains average IM‑10 (75.6→75.5) and beats random/cluster pruning across DocVQA/TextVQA/MMMU.
  - FP8 training (Table 4):
    - Throughput doubles without GC (199→390 it/s) and rises 2.5→2.9× with GC (492→580) while keeping benchmark scores within ±0.9 points.
  - Quantization (Table 6):
    - `W4A16` on LLM lowers TTFT (0.90→0.77 s) with small accuracy drop (e.g., Video‑MME 63.9→62.0). Adding `W8A8` on ViT further reduces TTFT to 0.65 s with negligible additional loss.
  - Inference profiling (Figure 5):
    - The vision tower dominates prefill after token compression; quantizing it (`W8A8`) is key to TTFT speedups. Decoding throughput benefits primarily from `W4A16` + FP16‑accumulating kernels.
- Do the experiments support the claims?
  - Yes. The paper provides clear before/after ablations for each design element, strong head‑to‑head efficiency comparisons (vs Qwen2‑VL and LLaVA‑OV), and broad benchmark coverage. Note that Figure 1c shows accuracy normalized to each benchmark’s best score, while Tables 8–9 report absolute numbers; both views are provided.
- Additional capabilities (Section 4):
  - Temporal localization (ActivityNet‑RTL): Mean IoU improves from 32.1 (VILA‑1.5‑8B) to 34.8 with `NVILA-8B` (Table 10) by adding discrete time tokens and smoothed cross‑entropy training.
  - Robotics (VLN‑CE, R2R Val‑Unseen): Navigation success rate improves to 53.3 with lower navigation error (NE 5.43), outperforming reported baselines (Table 11); real‑time 1 Hz demo on laptop GPU (Figure 6).
  - Medical (`NVILA‑M3`): Across VQA/report/classification, `NVILA‑8B` paired with medical experts beats both general Med‑Gemini and task‑specific SOTA on several tasks (Table 12).

## 6. Limitations and Trade‑offs
- Token compression accuracy trade‑off:
  - Spatial STC beyond 2×2 hurts unless `VEP` is added (Table 1). Even with `VEP`, compressed models can be a few points below the “scale‑only” peak on some OCR tasks.
- Vision encoder cost after scaling:
  - Multi‑scale tiling (`Dynamic-S2`) and many frames make the vision tower the prefill bottleneck (Figure 5). The paper mitigates this with `W8A8` quantization, but this assumes compatible hardware and a custom engine (Section 2.4).
- Data pruning dependency:
  - `DeltaLoss` requires evaluating both a “large” and a “small” model over sub‑datasets to score examples (Eq. 1), introducing extra compute and relying on the choice of teacher/student models (Section 2.2.1). The method is validated on their data mixture (Table 3), but generalization across other corpora might need re‑scoring.
- FP8 training assumptions:
  - Gains are measured on H100s with COAT (Table 4). Portability to other accelerators or to mixed hardware is not demonstrated; stability on very long sequences or different LLM backbones may require tuning.
- Quantization accuracy:
  - `W4A16` on the LLM introduces a small accuracy drop (Table 6). For the strictest accuracy targets, some deployments may prefer higher precision at the cost of latency.
- Benchmark scope and context:
  - While coverage is broad, some reported video metrics note with/without subtitles; datasets beyond those listed, or non‑English OCR/scene text scenarios, are not discussed. Figure 1c normalizes scores (helpful for visualization) but should be interpreted alongside absolute values in Tables 8–9.

## 7. Implications and Future Directions
- How this changes the landscape:
  - `NVILA` shows that high‑resolution images and long videos need not be at odds with efficiency if compression is staged and training/inference are optimized end‑to‑end. The work provides a reproducible blueprint—data pruning, FP8 training, flexible fine‑tuning, and dual‑path quantization—that others can adopt (Tables 3–7; Figure 5).
- Follow‑up research enabled/suggested:
  - Learnable compression that trains stably at high reduction ratios (where TokenLearner/Perceiver under‑performed here) could push efficiency further (Table 1).
  - Dynamic token budgets conditioned on input complexity (e.g., content‑aware tiling/frame selection rather than uniform multi‑scale/temporal sampling).
  - Joint optimization of vision and language quantization under accuracy constraints, including per‑layer/adaptive precision.
  - Broader evaluation: non‑English OCR, egocentric/robotic long‑horizon videos, safety/robustness under adversarial degradations.
- Practical applications:
  - Edge deployment of multimodal assistants (robots, AR, mobile) where TTFT matters (Figure 5; Figure 6).
  - Document understanding and chart/diagram QA at high accuracy and lower cost (Table 8; Table 1).
  - Medical imaging workflows when paired with expert models (Table 12) and long‑video analytics for surveillance or instructional content (Table 9).
  
Key citations to ground the above:
- Architecture and paradigm: Figure 3; Sections 2.1.1–2.1.2.
- Spatial results/ablations: Table 1.
- Temporal results/ablations: Table 2.
- Dataset pruning: Equation (1), Figure 4, Table 3.
- FP8 training: Section 2.2.2, Table 4.
- Fine‑tuning: Section 2.3, Table 5.
- Quantization and inference: Section 2.4, Table 6, Figure 5.
- Training curriculum and implementation: Section 3.1, Table 7, Table A1.
- Image benchmarks: Section 3.2.1, Table 8.
- Video benchmarks: Section 3.2.2, Table 9.
- Additional capabilities: Section 4; Tables 10–12; Figure 6.
- Aggregate efficiency/accuracy overview: Figure 1a–c.
