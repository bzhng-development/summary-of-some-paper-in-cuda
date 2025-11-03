# SmolVLM: Redefining small and efficient multimodal models

**ArXiv:** [2504.05299](https://arxiv.org/abs/2504.05299)
**Authors:** Andrés Marafioti, Orr Zohar, Miquel Farré, Merve Noyan, Elie Bakouch, Pedro Cuenca, Cyril Zakka, Loubna Ben Allal, Anton Lozhkov, Nouamane Tazi, Vaibhav Srivastav, Joshua Lochner, Hugo Larcher, Mathieu Morlon, Lewis Tunstall, Leandro von Werra, Thomas Wolf
**Institutions:** 

## 🎯 Pitch

SmolVLM introduces a family of compact vision-language models that drastically reduce GPU memory usage while maintaining competitive performance. With a design focused on efficient visual token compression and strategic instruction-tuning, these models enable powerful on-device multimodal inference, critical for real-world applications on mobile and edge devices.

---

## 1. Executive Summary
SmolVLM proposes a family of small, memory‑efficient vision‑language models (VLMs) that deliver strong image and video understanding while running with a fraction of the GPU RAM typically required. It does this by combining balanced model sizing, aggressive yet targeted visual token compression, longer context windows, and instruction‑tuning strategies tailored for small models, resulting in variants from 256M to 2.2B parameters that fit in 0.8–4.9 GB of VRAM at batch size 1 and are competitive with or better than much larger VLMs across many benchmarks (Table 1; Fig. 1, Fig. 9).

## 2. Context and Motivation
- Problem addressed
  - Large VLMs perform well but are hard to deploy on mobile/edge devices due to high memory and compute needs. Smaller VLMs often copy design choices from large models (e.g., many image tokens), leading to poor memory efficiency and limited practicality (Introduction; Fig. 1).
  - For video, token counts explode with frame count, making efficient token handling essential (Introduction; §2.3; Fig. 3 right).

- Why this matters
  - Real‑world deployment increasingly requires on‑device, low‑power multimodal inference (Introduction; §4.4). Efficient per‑token computation is crucial as modern reasoning LLMs generate more tokens (Introduction).
  - The paper shows <1 GB VRAM inference for the 256M model and 4.9 GB for the 2.2B model, enabling edge/mobile scenarios without drastic performance loss (Abstract; Table 1; §4.4; Fig. 9).

- Prior approaches and gaps
  - Large VLMs (Flamingo, Idefics) have excellent capability but huge memory footprints (Related Work §5.1).
  - Small VLMs exist (e.g., Qwen2‑VL, InternVL 2.5, PaliGemma, MiniCPM‑V, Moondream2) but often still retain high token counts or design choices that inflate RAM during inference (Introduction; Related Work §5.2–§5.4).
  - Recent VLMs compress vision tokens (e.g., Perceiver Resampler, Q‑Former, pixel shuffle) but compression choices may harm fine‑grained tasks if not tuned to model size (§2.2; §5.3).

- Positioning
  - SmolVLM systematically studies what matters for small multimodal models: how to split compute between vision encoder and LM (§2.1), how to extend context economically (§2.2), how aggressively to compress visual tokens (§2.2; Fig. 3 middle‑right), and how to handle images vs. videos differently (§2.3). It also probes instruction‑tuning choices that stabilize training and avoid capacity overuse in small models (§3.1–§3.5).

## 3. Technical Approach
SmolVLM adopts a joint self‑attention architecture where visual and text tokens are concatenated and processed by a small language model backbone (SmolLM2). The pipeline is summarized in Fig. 2 and detailed in §2–§3.

Step‑by‑step data and model flow (Fig. 2; §2):
1. Image/video preprocessing
   - Images can be split into sub‑images (“tiles”) plus a downsized global image for better coverage of high‑resolution content without exploding tokens (§2.3; “image splitting”; inspired by UReader and SPHINX). For videos, frames are sampled; the paper explicitly avoids averaging frames because it hurts performance in small models (Fig. 3 right; Finding 4).
   - Definition: `image splitting` is cutting a high‑resolution image into tiles (e.g., a grid) and possibly including a smaller global view. This preserves local detail while keeping the token budget manageable.

2. Vision encoding
   - A SigLIP ViT encoder produces dense spatial features (SigLIP‑B/16 at 93M or SigLIP‑SO‑400M at ≈400M; §2.1). SigLIP uses a sigmoid‑based contrastive loss for language‑image pretraining (Related Work; Zhai et al., 2023).

3. Space‑to‑depth compression via pixel shuffle
   - Definition: `pixel shuffle` (a.k.a. space‑to‑depth) rearranges an H×W×C feature map into (H/r)×(W/r)×(C·r²), reducing the number of spatial positions (= tokens) by r² while increasing channel depth (Fig. 4; §2.2).
   - Why: In the joint self‑attention setup, each visual token participates in O(n²) attention. Reducing the number of visual tokens cuts memory and compute drastically. In small models, this reduction outweighs the loss in spatial precision (§2.2; Fig. 3 middle‑right; Finding 3).

4. Projection to LLM token space
   - An MLP (linear layer) maps compressed vision features to the language model’s embedding space to form `visual tokens` (Fig. 2).

5. Token concatenation and joint processing
   - Visual tokens are concatenated or interleaved with text tokens and fed into the language model (`SmolLM2` backbone; sizes: 135M, 360M, 1.7B; §2.1; Fig. 2).

6. Longer context to fit images/videos
   - Single 512×512 images with 16×16 patches produce ~1024 visual tokens even before text; thus SmolVLM extends the context window by increasing the RoPE base from 10k to 273k and fine‑tuning on long‑context text (§2.2).
   - Definition: `RoPE base` is a scaling parameter for rotary positional embeddings; increasing it improves extrapolation to longer sequences (cited approach from Liu et al., 2024c).
   - Stable limits: 16k tokens for the 1.7B LM; 8k for 135M/360M variants (§2.2; Fig. 3 middle; Finding 2).

7. Instruction‑tuning choices specialized for small models (§3)
   - Positional tokenization: Learned `positional tokens` (trainable embeddings indicating tile positions) replace naive string markers like `<row_1_col_2>`. This stabilizes training and avoids the “OCR loss plague,” i.e., early loss drops without OCR gains (Fig. 5 left and center; §3.1; Finding 5).
   - Prompt and media segmentation: Short system prompts and explicit media intro/outro text markers improve zero‑shot reliability, especially for video (Fig. 6; §3.2; Finding 6).
     - Definition: `media intro/outro tokens` are textual delimiters (“Here is an image…”, “Given this image…”) that clearly bracket visual content within the text stream.
   - User‑prompt masking during SFT: During supervised fine‑tuning (SFT), training only on completions (masking user queries) reduces overfitting and improves generalization (Fig. 6; §3.2; Finding 6).
   - Data mix constraints:
     - Keep text to ~14% of the training mix for small VLMs; reusing LLM SFT text degrades multimodal performance (Fig. 7 left; §3.3; Finding 7).
     - Use extremely sparse Chain‑of‑Thought (`CoT`) data (0.02–0.05%); more hurts small‑model performance (Fig. 7 middle; §3.4; Finding 8).
       - Definition: `CoT` refers to training data with explicit step‑by‑step reasoning traces.
     - Moderate average video duration (~3.5 minutes) during training helps; longer brings diminishing returns (Fig. 7 right; §3.5; Finding 9).

8. Variants and deployment targets (§4.1; §4.4)
   - `SmolVLM‑256M`: 93M encoder + 135M LM; <1 GB VRAM inference.
   - `SmolVLM‑500M`: 93M encoder + 360M LM.
   - `SmolVLM‑2.2B`: ~400M encoder + 1.7B LM; 4.9 GB VRAM.
   - ONNX exports and WebGPU demo support browser/mobile deployment (§4.4; Fig. 9).

Design choices and why (§2; §3):
- Encoder/LM balance: In small models, a large vision encoder paired with a tiny LM is wasted capacity. Performance gains from large encoders appear only once the LM is big enough to use the visual signal (§2.1; Fig. 3 left; Finding 1).
- Aggressive visual compression (pixel shuffle r=4) helps small models by reducing attention overhead, even if r=2 is popular in larger VLMs (InternVL/Idefics3) (§2.2; Fig. 3 middle‑right; Finding 3).
- Longer context is critical to fit visual tokens without truncation (§2.2; Fig. 3 middle; Finding 2).
- Image tiling helps; frame averaging (combining frames into one) hurts small models — likely because it discards temporal cues needed for video understanding (§2.3; Fig. 3 right; Finding 4).
- Learned positional tokens and clear media bracketing reduce modality confusion and training instabilities (§3.1–§3.2; Fig. 5–6; Findings 5–6).
- Data composition must reflect small‑model capacity limits: minimal CoT, modest text proportion, moderate video length (§3.3–§3.5; Fig. 7; Findings 7–9).

## 4. Key Insights and Innovations
1) Balanced compute allocation for small VLMs (§2.1; Fig. 3 left)
- Novelty: A systematic study shows that, contrary to common large‑VLM practice, pairing a big encoder with a very small LM underutilizes capacity.
- Evidence:
  - With a 135M LM, using a 428M encoder reduces performance compared to a 93M encoder (Fig. 3 left).
  - With a 360M LM, the 428M encoder gives +11.6% performance but at +66% parameters; trade‑off favors smaller encoder at this scale (§2.1).
- Significance: Guides efficient model sizing for edge deployment.
- Summarized claim:
  > Finding 1. Compact multimodal models benefit from a balanced encoder-LM parameter allocation, making smaller vision encoders preferable for efficiency.

2) Aggressive but task‑aware visual token compression (§2.2; Fig. 3 middle‑right; Fig. 4)
- Novelty: Small VLMs benefit from pixel shuffle r=4 (vs. r=2 popular in larger models) because the reduction in attention cost improves overall modeling.
- Mechanism: Space‑to‑depth reduces token count by r²; fewer tokens → less memory/compute in self‑attention; small LMs gain more from lighter sequences.
- Caveat: High r can harm precise localization (e.g., OCR); SmolVLM still reports strong OCR with larger LM (Table 1).
- Summarized claim:
  > Finding 3. Small VLMs benefit from more aggressive visual token compression.

3) Long context windows are essential and made stable for small models (§2.2; Fig. 3 middle)
- Novelty: Extending context (via RoPE base 273k and long‑context finetuning) produces consistent gains; 16k is stable for 1.7B LM; 8k is the safe limit for 135M/360M.
- Why it matters: Visual tokens are numerous; without long context, images/videos force truncation or over‑compression.
- Summarized claim:
  > Finding 2. Compact VLMs significantly benefit from extended context lengths.

4) Small‑model–specific instruction‑tuning practices (§3; Fig. 5–6; Fig. 7)
- Novelty:
  - Learned positional tokens prevent training stalls (“OCR loss plague”) and improve OCR/generalization (§3.1; Fig. 5).
  - System prompts + media intro/outro tokens reduce ambiguity; masking user prompts during SFT curbs overfitting (§3.2; Fig. 6).
  - Data mix tuned for capacity: avoid reusing LLM SFT text, keep text at ~14%, use minimal CoT, moderate video duration (§3.3–§3.5; Fig. 7).
- Summarized claims:
  > Finding 5. Learned positional tokens outperform raw text tokens for compact VLMs.  
  > Finding 6. System prompts and media intro/outro tokens significantly improve compact VLM performance, particularly for video tasks. During SFT, only train on completions.  
  > Finding 7. Adding text from SFT blend proved worse than new text SFT data.  
  > Finding 8. Excessive CoT data harms compact model performance.  
  > Finding 9. Moderately increasing video duration during training improves both video and image task performance in compact VLMs.

These are largely fundamental design insights for small‑scale multimodal modeling rather than incremental tweaks, because they reverse or qualify common practices borrowed from large models (e.g., encoder sizing, compression ratio, CoT usage).

## 5. Experimental Analysis
- Evaluation setup (§4.2)
  - Framework: VLMEvalKit for reproducibility (Duan et al., 2024).
  - Coverage: 31 multimodal benchmarks tracked by the OpenVLM leaderboard; the paper summarizes 9 image/overall benchmarks and 5 video benchmarks in Table 1.
  - Memory as primary efficiency proxy: The paper argues VRAM at inference correlates better with real cost for VLMs than parameter count (§4.2).

- Data pipeline (§4.1; Fig. 8)
  - Two‑stage training: vision stage then video stage.
  - Vision mixture emphasizes document/OCR, charts/tables, captioning, and reasoning; plus a modest portion of text for general knowledge and logic/maths.
  - Video stage keeps 14% text, 33% video, and includes multi‑image data; video sources include LLaVA‑video‑178k, Video‑STAR, Vript, ShareGPT4Video, Vista‑400k, MovieChat, FineVideo (Fig. 8).

- Main quantitative results (Table 1; Fig. 1; §4.3)
  - Memory footprint at batch size 1:
    - `256M`: 0.8 GB
    - `500M`: 1.2 GB
    - `2.2B`: 4.9 GB
    - Baseline comparison: MolmoE‑A1B‑7B needs 27.7 GB (Table 1).
  - Average across benchmarks:
    - `256M`: 44.0%
    - `500M`: 51.0%
    - `2.2B`: 59.8%
  - Selected single‑image tasks (SmolVLM‑2.2B):
    - OCRBench 72.9%, DocVQA 80.0%, ScienceQA 89.6%, ChartQA 68.7%, TextVQA 73.0% (Table 1).
  - Selected video tasks (SmolVLM‑2.2B):
    - Video‑MME 52.1, MLVU 55.2, TempCompass 53.7, WorldSense 36.2, MVBench 46.3 (Table 1).
  - Notable comparisons:
    - On WorldSense, `2.2B` scores 36.2 vs. Qwen2‑VL‑7B at 32.4 (Table 1).
    - On ScienceQA, `2.2B` at 89.6; some compact baselines do better on specific tasks but with higher memory (text in §4.3).
    - `256M` uses <1 GB VRAM yet outperforms the much larger Idefics‑80B on nearly all benchmarks aggregated in Fig. 1 and discussed in §4.3.

- Ablations and diagnostic studies
  - Encoder/LM sizing (§2.1; Fig. 3 left): shows the diminishing return or even harm of a large encoder with tiny LM; gains grow as LM size increases.
  - Context length (§2.2; Fig. 3 middle): monotonic improvements up to 16k (2.2B LM); 135M/360M unstable beyond 8k.
  - Pixel shuffle ratio (§2.2; Fig. 3 middle‑right): r=4 performs better for small LMs than r=2.
  - Image splitting vs. video frame averaging (§2.3; Fig. 3 right): image splitting helps; frame averaging hurts quickly as averaging factor increases.
  - Tokenization and prompts (§3.1–§3.2; Fig. 5–6): learned positional tokens and system/intro‑outro prompts improve OCR and overall OpenCompass scores; user‑prompt masking helps generalization.
  - Data composition (§3.3–§3.5; Fig. 7): avoid reusing LLM SFT text, use minimal CoT, cap average training video length at ~3.5 minutes.

- On‑device throughput (§4.4; Fig. 9)
  - A100 GPU: `256M` scales from 0.8 to 16.3 examples/s as batch size increases to 64; `2.2B` scales to ~1.7 ex/s at batch size 64 (Fig. 9).
  - L4 GPU: smaller peaks due to memory; `256M` ~2.7 ex/s at batch size 8; `2.2B` ~0.25 ex/s at low batch size (Fig. 9).
  - Browser/WebGPU demo shows up to 80 decode tokens/s for `256M` on a MacBook Pro (M4 Max) (§4.4).

- Do the experiments support the claims?
  - Efficiency: Clear and convincing. Table 1’s RAM numbers (0.8–4.9 GB) vs. baselines like 27.7 GB for MolmoE‑A1B‑7B substantiate edge‑readiness.
  - Effectiveness: Mixed but strong overall. The `2.2B` variant averages 59.8% across many tasks; it wins or is competitive in several tasks but trails in some specialized ones (e.g., MVBench shows a gap to one efficient baseline; Table 1).
  - Causality of design choices: Ablations in Fig. 3, Fig. 5–7 isolate the impact of sizing, compression, context length, tokenization, prompts, and data mix, lending credibility to the findings.

## 6. Limitations and Trade-offs
- Spatial precision vs. compression (§2.2; Fig. 4)
  - Pixel shuffle r=4 reduces tokens aggressively. While beneficial overall for small models (Fig. 3 middle‑right), it can impair localization tasks like OCR in principle. SmolVLM mitigates via learned tokens and scaling the LM, but the trade‑off remains inherent to compression.

- Long‑context stability (§2.2)
  - Small backbones (135M/360M) were unstable above 8k tokens; 16k was stable only for the 1.7B LM (Fig. 3 middle). This caps the maximal image resolution or number of frames for the smallest variants.

- Video temporal modeling (§2.3; Table 1)
  - The design avoids frame averaging and relies on rescaling frames. While this prevents performance drops observed with averaging (Fig. 3 right), it may still be suboptimal for very long videos requiring hierarchical temporal memory. On MVBench, `2.2B` scores 46.3 vs. an efficient baseline at 60.2 (Table 1), suggesting room for improved temporal reasoning.

- Data/composition assumptions (§3.3–§3.5)
  - The ~14% text ratio, tiny CoT fraction (0.02–0.05%), and ~3.5‑minute average video length are tuned to these models and training regimes. Different domains or objectives might require re‑tuning; generality is plausible but not guaranteed.

- Throughput vs. memory (§4.4; Fig. 9)
  - Although VRAM usage is low, the largest variant’s throughput is modest on commodity GPUs (e.g., ~0.25 ex/s on L4); latency‑sensitive applications may prefer the 256M/500M variants or quantization/compilation optimizations.

- Evaluation details that may affect fairness (§4.2)
  - Images are resized differently across variants (longest edge 1920 for 256M/500M; 1536 for 2.2B) and compared against diverse baselines; exact comparability depends on each model’s eval recipe.

## 7. Implications and Future Directions
- Landscape impact
  - SmolVLM shows that careful architectural and training design can make small VLMs both usable and competitive, enabling a new tier of on‑device multimodal AI. The work reframes “efficiency” away from parameter count alone and toward token budgeting, attention cost, and VRAM (Fig. 1; §4.2–§4.3).

- Practical applications (§4.4–§4.5)
  - Edge deployment: Mobile apps and browser‑based inference are feasible with `256M` and `500M` variants (Fig. 9; §4.4).
  - Domain tools: Document conversion (Smol Docling), efficient document retrieval (ColSmolVLM / ColPali), and biomedical VQA (BioVQA) illustrate domain transfer at small scale (§4.5).

- Research enabled or suggested
  - Dynamic token compression: Adaptive pixel‑shuffle ratios or content‑aware token selection could preserve spatial precision only where needed (§2.2; §5.3).
  - Temporal hierarchies for video: Memory‑augmented or hierarchical temporal encoders may close gaps on long‑form tasks (Table 1; §5.4).
  - Stability at long context for small LMs: Methods beyond RoPE base scaling to make >8k stable for 135M/360M backbones (§2.2).
  - Data curriculum for small VLMs: Mechanisms that introduce CoT and long videos progressively without overwhelming capacity (Fig. 7; Findings 8–9).
  - Multi‑image/document layouts: Further exploration of tiling + learned positional tokens for complex documents and UI screenshots (§2.3; §3.1).

- Takeaway for practitioners
  - When building small VLMs for deployment, prioritize:
    - Balanced encoder/LM sizing (Fig. 3 left),
    - Aggressive but validated compression (Fig. 3 middle‑right; Fig. 4),
    - Sufficient context length (Fig. 3 middle),
    - Learned positional tokens and clear visual segmentation (Fig. 5–6),
    - A data blend that respects small‑model capacity (Fig. 7; Fig. 8).
  - These decisions collectively produce large memory savings with competitive accuracy, making real‑time multimodal inference on edge devices practical.

> Example headline claims supported in the paper:
> - “Our smallest model, SmolVLM‑256M, uses less than 1GB GPU memory during inference” (Abstract; Table 1 shows 0.8 GB).
> - “Our largest model, at 2.2B parameters, rivals state‑of‑the‑art VLMs consuming twice the GPU memory” (Abstract; Table 1 shows 4.9 GB vs. 10.5–13.7 GB cited in §4.3 for similar‑scale baselines).
> - “Small VLMs benefit from more aggressive visual token compression” (Finding 3; §2.2; Fig. 3 middle‑right).
> - “System prompts and media intro/outro tokens significantly improve … particularly for video tasks. During SFT, only train on completions” (Finding 6; §3.2; Fig. 6).
