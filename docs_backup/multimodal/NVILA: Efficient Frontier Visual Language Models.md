# NVILA: Efficient Frontier Visual Language Models

**ArXiv:** [2412.04468](https://arxiv.org/abs/2412.04468)
**Authors:** Zhijian Liu, Ligeng Zhu, Baifeng Shi, Zhuoyang Zhang, Yuming Lou, Shang Yang, Haocheng Xi, Shiyi Cao, Yuxian Gu, Dacheng Li, Xiuyu Li, Yunhao Fang, Yukang Chen, Cheng‑Yu Hsieh, De‑An Huang, An‑Chieh Cheng, Vishwesh Nath, Jinyi Hu, Sifei Liu, Ranjay Krishna, Daguang Xu, Xiaolong Wang, Pavlo Molchanov, Jan Kautz, Hongxu Yin, Song Han, Yao Lu
**Institutions:** NVIDIA Research, MIT, UC Berkeley, UC San Diego, University of Washington, Tsinghua University

## 🎯 Pitch

NVILA's innovative "scale-then-compress" architecture for visual-language models revolutionizes efficiency by enabling detailed processing of high-resolution images and longer videos at reduced computational cost. This approach not only achieves frontier-level accuracy but dramatically decreases training and inference times by 1.9–5.1× and up to 2.8× respectively, making state-of-the-art models more accessible and deployable on edge devices, crucial for real-world applications in robotics and healthcare.

---

## 1. Executive Summary
NVILA introduces a “scale-then-compress” design for visual-language models (VLMs) that processes higher spatial resolutions and longer videos without incurring prohibitive costs. It delivers frontier-level accuracy while reducing training time by 1.9–5.1× and inference latency by up to 2.8× through a combination of high-resolution tiling, spatial/temporal token compression, dataset pruning, FP8 training, and deployment-time quantization (Figure 1; Tables 1–6).

## 2. Context and Motivation
- Problem addressed
  - State-of-the-art VLMs have prioritized accuracy, but lag in efficiency across the full lifecycle: training, fine-tuning, and deployment. Specific bottlenecks include long training times, high memory footprint for fine-tuning, and slow inference on resource-limited devices (Introduction; Figure 1).
  - Prior VILA models downsample images to fixed small resolutions (e.g., 448×448) and sample few video frames (e.g., up to 14), losing critical detail and limiting performance on text-heavy images and long videos (Section 2.1; Tables 8–9).

- Why it matters
  - Real-world use (robotics, laptops, edge devices) requires fast inference and adaptation under tight compute/memory budgets.
  - Research accessibility: high training costs (e.g., hundreds of GPU days for 7B models) limit broader participation.

- Prior approaches and gaps
  - Strong open/proprietary VLMs (e.g., Qwen2-VL, InternVL, Gemini, GPT-4o) push accuracy, but their full-stack efficiency—especially vision-tower bottlenecks and end-to-end quantization—has been less systematically optimized (Related Work §5.1; Figure 1).
  - Token reduction is known in pure vision/video models, but not demonstrated end-to-end in frontier VLMs with careful accuracy recovery (Related Work §5.2).

- NVILA’s positioning
  - A full-stack approach that:
    - Scales spatial/temporal resolution to raise the accuracy ceiling, then compresses tokens to keep compute in check (Section 2.1; Tables 1–2).
    - Prunes supervised fine-tuning (SFT) data with a principled “DeltaLoss” criterion to reduce training without hurting accuracy (Section 2.2.1; Table 3; Figure 4).
    - Trains with FP8 to increase batch size and throughput (Section 2.2.2; Table 4).
    - Deploys with quantization for both the vision tower and the LLM, and a specialized inference engine to cut Time-To-First-Token (TTFT) and boost decoding throughput (Section 2.4; Figure 5; Table 6).

## 3. Technical Approach
At a high level, NVILA keeps the standard three-part VLM structure and rethinks how vision inputs are ingested and compressed before reaching the language model.

- Baseline architecture (Figure 3; Section 2.1)
  - `Visual encoder (ViT)`: SigLIP extracts visual features.
  - `Projector (MLP)`: aligns vision features to the language embedding space.
  - `Token processor (LLM)`: Qwen2 family processes mixed visual-text tokens autoregressively.

- Core design: “scale-then-compress”
  - Rationale: Using more pixels and frames raises the information ceiling but explodes token counts and self-attention cost. NVILA first scales to capture detail, then compresses visual tokens to bring cost back down (Section 2.1).

- Spatial scaling with tiling while preserving aspect ratio
  - S2 tiling in spirit: If a ViT is pretrained at 448×448, larger images are broken into 448×448 tiles at multiple scales (e.g., 448², 896², 1344²), encoded tile-by-tile, then stitched back into full-image feature maps (Section 2.1.1).
  - `Dynamic-S2`: Unlike square-resizing that distorts aspect ratios, the largest scale respects the original aspect ratio by choosing tile grids that are multiples of 448, then interpolates and concatenates multi-scale features (Section 2.1.1).
  - Why it matters: It lets the model “see” more detail on large or non-square documents/diagrams—critical for OCR-like tasks—without retraining the ViT at new native resolutions (Table 1).

- Spatial token compression
  - Problem: More tiles → more tokens → more LLM compute.
  - Method: Reduce tokens per tile with spatial-to-channel reshaping or pooling. NVILA uses a stronger compression (3×3 pooling → 121 tokens per tile) vs the earlier VILA’s 2×2 (256 tokens per tile) (Table 1).
  - Stabilization via an extra pretraining stage: More aggressive compression initially hurt accuracy (e.g., DocVQA). NVILA adds an extra “Visual Encoder Pretraining (VEP)” stage to jointly tune the ViT and projector, recovering most of the loss and enabling a 2.4× speedup (Table 1, “Scale + Compress + VEP”).

- Temporal scaling and compression for videos
  - Scaling: Increase frame count (e.g., from 8 to 32 or even 256) with additional video instruction tuning to handle more frames (Section 2.1.2; Table 2).
  - Compression: Partition frames into groups and temporally pool (average) visual tokens within each group—this exploits redundancy across adjacent frames (Table 2).
  - Trade-off: With 32 frames and 4× compression, NVILA largely preserves gains while keeping total tokens near the 8-frame baseline (Table 2). With 256 frames and 8× compression, NVILA attains state-of-the-art scores at similar token budgets (Table 2 and Table 9).

- Data pruning with `DeltaLoss` (Equation 1; Section 2.2.1)
  - Goal: Compress SFT datasets (>100M samples) without sacrificing performance.
  - Mechanism: For each example `x`, compute log(p_large(x)/p_small(x)) over answer tokens—intuitively keeping examples that the large model solves but the small model fails (positive scores), and dropping those that are too easy or misleading (Figure 4).
  - Selection: Take the top-K per sub-dataset to retain diverse, informative samples (Equation (1); Table 3).
  - Effect: Pruning 50% of data via DeltaLoss keeps accuracy almost unchanged and outperforms random or cluster-based pruning (Table 3).

- FP8 training for throughput (Section 2.2.2; Table 4)
  - Background: Hopper/Blackwell GPUs support FP8 matrix multiplies; compressing activations, optimizer states, and gradients reduces memory and communication.
  - Implementation: COAT-style FP8 for weights and activations boosts batch size and throughput, especially helpful for variable-length VLM batches (Table 4).
  - Outcome: 2.0× throughput without gradient checkpointing; still +1.2× with it, with negligible accuracy changes on MMMU and Video-MME (Table 4).

- Fine-tuning recipe optimized for VLMs (Section 2.3; Table 5)
  - Insight 1: Tune the LLM with `LoRA` or `QLoRA`; tune the ViT more conservatively (e.g., only LayerNorm) and at 5–50× smaller learning rate than the LLM.
  - Insight 2: Tuning only ViT LayerNorm achieves accuracy close to ViT-LoRA while being faster and lighter (Table 5).
  - Result: Competitive accuracy under ~24 GB memory; best balance is LLM-LoRA/QLoRA + ViT-LN with small LR (Table 5).

- Deployment-time quantization and specialized engine (Section 2.4; Table 6; Figure 5)
  - Prefilling vs decoding:
    - `Prefilling` (a.k.a. prefill): compute the first forward pass over the full prompt and visual tokens; dominated by the vision tower after token compression.
    - `Decoding`: generate tokens step-by-step; typically memory-bound in LLMs.
  - Quantization:
    - `Vision tower`: `W8A8` (8-bit weights and activations) reduces TTFT materially with near-lossless accuracy on AI2D/TextVQA/Video-MME (Table 6).
    - `LLM`: `W4A16` with FP16 accumulation (improved AWQ kernels) speeds LLM matrix multiplies without notable accuracy loss (Table 6; Figure 5).
  - Engine outcome: Up to 2.2× faster prefilling and up to 2.8× higher decoding throughput vs Qwen2-VL under matched conditions (Figure 5).

- Training pipeline and data (Section 3.1; Table 7; Table A1)
  - Five stages: projector init → visual encoder pretraining (new) → token processor pretraining → image instruction tuning → video instruction tuning (new emphasis for long videos) with stage-wise learning rates (Table 7).
  - Curated multimodal data mixture for each stage (Table A1).

## 4. Key Insights and Innovations
- “Scale-then-compress” for both space and time (Fundamental)
  - What’s new: A principled two-step recipe—first scale resolution/frames to capture detail, then compress tokens to manage compute—validated by ablative tables (Tables 1–2).
  - Why it matters: Yields higher accuracy ceilings without blowing up inference/training cost. Example: Dynamic-S2 boosts DocVQA from 61.3 → 91.1 before compression; after 3×3 pooling + VEP, much of that gain remains (88.8) with far fewer tokens (Table 1).

- `Dynamic-S2` tiled high-res extraction (Incremental but impactful)
  - What’s new: Aspect-ratio-aware multi-scale tiling at 448-sized tiles lets SigLIP see large, non-square inputs without retraining at huge native resolutions (Section 2.1.1).
  - Impact: Up to ~30% accuracy improvements on text-heavy benchmarks (Table 1; note AI2D/DocVQA/TextVQA jumps from baseline to “Scale”).

- Extra visual-encoder pretraining to stabilize aggressive spatial compression (Practical innovation)
  - What’s new: A dedicated Stage 2 that jointly tunes ViT + projector to recover accuracy lost from stronger token compression (Section 3.1; Table 7; Table 1).
  - Impact: Enables 2.4× speedups with limited accuracy loss on DocVQA/TextVQA (Table 1).

- Full-stack efficiency: pruning, FP8 training, and vision+LLM quantization (System-level innovation)
  - What’s new: Demonstrates that FP8-based training and W8A8 (vision)/W4A16 (LLM) quantization can be applied to VLMs at scale with negligible accuracy loss, addressing both training and inference ends (Tables 4 and 6; Figure 5).
  - Impact: End-to-end speedups—training 1.9–5.1× faster than LLaVA-OV (Figure 1a), prefilling 1.6–2.2× and decoding 1.2–2.8× faster than Qwen2-VL (Figures 1b, 5).

- Broadened capabilities: long videos and temporal localization (New capabilities)
  - What’s new: Up to 256-frame processing with compression (Table 2), plus timestamp-aware temporal localization via discrete time tokens and smoothed loss (Table 10).
  - Impact: SOTA results on long-video benchmarks and improved temporal localization IoU/Precision@0.5 (Table 9; Table 10).

## 5. Experimental Analysis
- Evaluation setup
  - Models and sizes: Primarily `NVILA-8B` and `NVILA-15B`; comparisons include Qwen2-VL, InternVL, LLaVA-OV, etc. (Tables 8–9).
  - Hardware for speed:
    - Training time vs LLaVA-OV: NVIDIA H100s (Figure 1a).
    - Inference vs Qwen2-VL: single RTX 4090, both LLMs at `W4A16`; NVILA’s vision tower at `W8A8` (Figure 5).
  - Prefill/Decode definitions and measurements: TTFT breakdown shows vision tower dominates prefilling after token compression (Figure 5a).

- Main accuracy results (images)
  - On general and text-heavy image benchmarks, NVILA is competitive or better at 8B and 15B:
    - AI2D: `NVILA-8B 92.3`, `Qwen2-VL-8B 83.0`; `NVILA-15B 94.1` (Table 8).
    - DocVQA: `NVILA-8B 93.7`, `Qwen2-VL-8B 94.5`; `NVILA-15B 94.0` (Table 8).
    - TextVQA: `NVILA-8B 80.1`, `Qwen2-VL-8B 84.3`; `NVILA-15B 80.0` (Table 8).
    - Overall radar chart in Figure 1c shows NVILA is on-par/superior across a wide range after normalization.

- Main accuracy results (videos)
  - Long video and QA (Table 9):
    - ActivityNet-QA acc: `NVILA-8B 60.9`, competitive with larger/proprietary.
    - NExT-QA multiple-choice: `NVILA-8B 70.1`, higher than many 7–8B open baselines.
    - Video-MME: `w/o subtitles 68.1`, `with subtitles 82.2` for `NVILA-8B`, surpassing other 7–8B entries listed (Table 9).
    - MVBench test: `NVILA-8B 58.7`; MLVU val: `57.7`; competitive among 7–8B class.

- Ablations that validate “how” the gains arise
  - Spatial scaling/compression (Table 1):
    - Baseline (VILA-1.5, 2×2 pooling, 1 tile): AI2D 87.0, DocVQA 61.3, TextVQA 67.5.
    - +Dynamic-S2 (multi-tiles): AI2D 90.1, DocVQA 91.1, TextVQA 77.0.
    - +3×3 spatial pooling: DocVQA dips to 82.3; +VEP recovers to 88.8 (TextVQA 76.1).
    - Alternatives (TokenLearner/Perceiver Resampler) at same reduction don’t beat 3×3 + VEP (Table 1).
  - Temporal scaling/compression (Table 2):
    - 8→32 frames boosts Video-MME overall score 55.7→61.0.
    - Compress 32 frames by 4× keeps 60.1, a small drop for a 4× token reduction.
    - 256 frames with 8× compression yields the best overall (64.0), demonstrating the scale-then-compress path to SOTA.

  - Dataset pruning (Table 3):
    - Pruning 50% with DeltaLoss keeps average IM-10 at 75.5 vs 75.6 baseline; Random drops to 74.0, Cluster to 74.5.
    - Similar trends at 30% and 10% keep DeltaLoss ahead on DocVQA/TextVQA/MMMU.

  - FP8 training (Table 4):
    - No-GC throughput rises from 199.2 to 390.1 it/s (2.0×) with negligible changes on MMMU (47.9→47.0) and Video-MME (52.9→53.0).
    - With GC, FP8 still improves throughput 491.7→579.9 it/s (1.2×).

  - Fine-tuning recipe (Table 5):
    - Best memory-efficient choice: LLM-LoRA/QLoRA + ViT-LN with smaller LR for ViT; achieves FT-5 average ≈71–72 while keeping memory ~10–20 GB and decent throughput.

  - Quantization (Table 6; Figure 5):
    - `W8A8` ViT + `W4A16` LLM keeps accuracy nearly intact (AI2D 91.0→90.9; MMMU 50.7→49.3; Video-MME 63.9→62.1), while improving TTFT from 0.90→0.65 s.
    - Figure 5 shows TTFT and throughput gains over Qwen2-VL: up to 2.22× faster TTFT (video), 1.55× (image), and up to 2.84× higher decoding throughput.

- New capabilities
  - Temporal localization (Table 10): Mean IoU `34.8` and Precision@0.5 `32.1` for NVILA-8B (256 frames), outperforming LITA 7B/13B and a VILA-1.5 reproduction.
  - Robotics (Table 11): In VLN-CE (R2R Val-Unseen), NVILA-8B achieves SR 53.3 and SPL 48.8 using only RGB, improving prior video-VLM agents under identical inputs.
  - Medical (Table 12): Using the NVILA-based M3 setup, medical VQA/reporting/classification metrics improve over domain SOTA and a Gemini baseline.

- Do the results support the claims?
  - Yes, because:
    - The ablations (Tables 1–2) isolate the benefit of scaling first and then compressing.
    - Data/pruning (Table 3), FP8 (Table 4), and quantization (Table 6) each show efficiency gains with minimal accuracy loss.
    - Speedups are measured head-to-head against strong baselines (Figure 1 vs LLaVA-OV for training; Figure 5 vs Qwen2-VL for inference) under stated hardware.

- Caveats in interpretation
  - Inference comparisons use NVILA’s specialized engine vs vLLM for Qwen2-VL; kernel choices and quantization paths differ, which can influence absolute speed (Figure 5).
  - Training-time comparisons (Figure 1a) are versus the only baseline with disclosed costs (LLaVA-OV); broader cost disclosures would further validate generality.

## 6. Limitations and Trade-offs
- Increased vision-tower work before compression
  - More tiles or frames mean more ViT forward passes; while the LLM cost is controlled by compression, prefilling can become dominated by the vision tower (Figure 5a). NVILA addresses this with W8A8 quantization, but it relies on hardware support for 8-bit activation compute.

- Aggressive compression still risks information loss
  - Spatial 3×3 pooling and temporal averaging discard detail; the extra VEP stage recovers much of the loss (Table 1), but extremely fine text or rapid micro-events in video may still degrade.

- Additional training complexity
  - Five-stage training, including a new Stage 2 (VEP) and Stage 5 (video SFT), increases pipeline complexity and hyperparameter surface (Table 7).

- Data pruning requires dual models
  - `DeltaLoss` needs a “large” and a “small” model to score samples (Equation 1), which adds overhead and presumes access to reasonably strong teachers.

- Quantization coverage
  - Results are shown for W8A8 (vision) and W4A16 (LLM) on NVIDIA GPUs; portability to other accelerators or different quantization toolchains may vary (Section 2.4; Table 6).

- Scale and compute assumptions
  - Despite efficiency gains, core training still uses substantial resources (e.g., 128 H100s; Section 3.1). The claimed 1.9–5.1× reductions make training more accessible but not “cheap.”

- Limited exploration of learnable token compressors
  - TokenLearner and Perceiver Resampler underperform the simple spatial-to-channel approach at the same reduction (Table 1). The paper attributes this to optimization challenges but does not fully resolve it.

## 7. Implications and Future Directions
- Field impact
  - Demonstrates that frontier-level VLMs can be made significantly more efficient without sacrificing accuracy by coordinating design across architecture, data, training precision, and deployment. This shifts the goal from “accuracy at any cost” to “accuracy on the efficient frontier.”

- Immediate applications
  - Edge deployment: Robotics and on-device assistants benefit from lower TTFT and higher decoding throughput (Figure 5; Table 11).
  - Document/diagram understanding: High-resolution tiling with aspect-ratio preservation lifts OCR-heavy tasks (Table 1; Table 8).
  - Long-video analytics: Scaled frame ingestion with temporal compression enables hour-scale analysis with manageable compute (Table 2; Table 9).
  - Healthcare: NVILA-based M3 indicates tangible gains when coupled with expert models (Table 12).

- Research directions enabled or suggested
  - Adaptive token budgets: Learn to vary spatial/temporal compression per input (content-aware) rather than fixed pooling ratios.
  - Better learnable compressors: Revisit TokenLearner/Perceiver-style modules with improved optimization, curriculum, or extra supervision, potentially surpassing spatial-to-channel reshaping (Table 1).
  - Cross-stage co-optimization: Jointly tune tiling scales, pooling ratios, and projector capacity with meta-learning or automated curriculum.
  - Universal quantization recipes: Extend W8A8/W4A16 with portability to diverse hardware, auto-tuning calibration, and content-aware mixed precision.
  - Data selection at scale: Extend `DeltaLoss` to multi-task/multi-goal selection, incorporate uncertainty or novelty signals, and reduce the need for dual teacher scoring.
  - Temporal localization and planning: Build on the time-token interface to enable richer event retrieval, step-wise action grounding, and multi-camera fusion.

> Key takeaways grounded in the paper’s evidence:
> - “Scale then compress” is validated by controlled ablations that first raise the accuracy ceiling (Dynamic-S2 tiling; Tables 1–2) and then recover efficiency with minimal loss (spatial/temporal pooling + VEP).
> - Full-stack efficiency measures—DeltaLoss pruning (Table 3), FP8 training (Table 4), and deployment quantization with a specialized engine (Table 6; Figure 5)—compound to yield 1.9–5.1× faster training and up to 2.8× faster decoding without undermining accuracy (Figure 1; Tables 8–9).
> - New capabilities (long videos, temporal localization) emerge naturally from the scaled temporal pathway (Tables 9–10), broadening practical impact (Tables 11–12).
