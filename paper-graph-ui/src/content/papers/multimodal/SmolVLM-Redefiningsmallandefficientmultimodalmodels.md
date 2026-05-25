# SmolVLM: Redefining small and efficient multimodal models

**ArXiv:** [2504.05299](https://arxiv.org/abs/2504.05299)

## 🎯 Pitch

SmolVLM pioneers a new class of ultra-efficient vision-language models that deliver robust image and video understanding while using less than 1GB of GPU memory—making them practical for edge devices, smartphones, and browsers. By systematically rethinking architecture, tokenization, and training recipes for compactness rather than merely scaling down large models, SmolVLM closes the performance gap with models hundreds of times larger, unlocking real-world, privacy-preserving multimodal AI for resource-constrained scenarios.

---

## 1. Executive Summary (2-3 sentences)
SmolVLM introduces a family of small, memory‑efficient vision–language models (VLMs) that run on as little as 0.8–1.2 GB of GPU RAM while matching or surpassing much larger systems on many image and video benchmarks. The work’s significance is twofold: it distills a concrete recipe for building strong compact VLMs (architecture, tokenization, training data, and prompts) and validates it with extensive ablations and end‑to‑end results, including on‑device and browser deployments (Figures 1, 3–7, 9; Table 1).

## 2. Context and Motivation
- Problem addressed:
  - Most VLMs achieve high accuracy by being very large, which makes them impractical on phones, browsers, or low‑end GPUs due to memory and compute constraints (Introduction, p.2).
  - Smaller VLMs often copy design choices from large models (e.g., very dense image tokenization), which keeps memory cost high and nullifies the benefits of being small (Abstract; Introduction, p.2).

- Why it matters:
  - Real‑world need for on‑device and edge inference: lower latency, privacy, offline use, and energy efficiency—especially challenging for video tasks, which inflate token counts (Introduction, p.2; Finding 4 in §2.3).
  - Practicality is affected not only by parameter count but by sequence length and tokenization. Efficient per‑token processing is crucial because modern LMs generate many tokens at inference (Introduction, p.2).

- Prior approaches and gaps:
  - Large early VLMs like Flamingo and Idefics (80B) set strong performance but huge memory cost (Introduction, p.2; Related Work §5.1).
  - “Smaller” lines (e.g., Qwen2‑VL 1–2B, InternVL 2.x) still carry heavy overheads or reserve vision only for large models (Introduction, p.2; Related Work §5.2).
  - Efficient models exist (e.g., Moondream, MiniCPM‑V), but the field lacked a systematic, end‑to‑end recipe for compact VLMs that jointly optimizes architecture, tokenization, and data for strict memory budgets.

- Positioning:
  - The paper offers a principled exploration and a unifying design: balanced vision–language compute for small scales, extended context, aggressive but controlled visual token compression, learned positional tokens, and carefully balanced training data (Sections 2–3). It releases open weights, code, and demos, and demonstrates on‑device use (Abstract; §4.4; Figure 9).

## 3. Technical Approach
SmolVLM is a pipeline that compresses visual information into a small number of tokens and interleaves them with text tokens for a compact LM to process. The system is instantiated at three sizes: `SmolVLM-256M`, `SmolVLM-500M`, and `SmolVLM-2.2B` (§4, p.7).

- Architecture and token path (Figure 2; §2):
  1. Image/video ingestion
     - Images can be split into tiles (“sub‑images”), and videos are sampled into frames (§2.3). Image splitting provides both a high‑resolution tiled view and a downscaled global image so the model sees details without losing global context.
  2. Vision encoder
     - Uses `SigLIP` variants: `SigLIP‑B/16` (93M params) for the 256M and 500M models; `SigLIP‑SO400M` (~400M) for the 2.2B model (§4, p.7).
     - SigLIP is a CLIP‑like image encoder trained with a sigmoid contrastive loss (Related Work §5.2; Zhai et al., 2023).
  3. Pixel shuffle (space‑to‑depth) compression
     - Pixel shuffle rearranges spatial features into channels, reducing token count by r² for shuffle ratio `r` (Figure 4; §2.2). Example: 2×2 shuffle turns 4 adjacent patches into 1 token with 4× channels.
     - This lowers attention cost while trying to preserve information density.
  4. MLP projection
     - A small MLP maps vision features to the LM’s embedding space to form “visual tokens” (Figure 2).
  5. Concatenation and LLM
     - Visual tokens are concatenated/interleaved with text tokens and fed to a compact `SmolLM2` language model (135M, 360M, or 1.7B params; §2.2) with an extended context window (§2.2).

- Compute allocation between vision and language (§2.1; Figure 3, left):
  - Design choice: use smaller encoders with smaller LMs; use a larger encoder only when the LM is large enough to exploit it.
  - Evidence (Figure 3, left): pairing a large encoder (428M) with the tiniest LM (135M) reduces performance; the 360M LM benefits modestly from the big encoder but at high parameter cost; only at 1.7B LM scale does the large encoder add value with a small total‑parameter penalty.

- Long‑context capability (§2.2; Figure 3, middle):
  - The team increases the `RoPE` base (rotary positional embedding base) from 10k to 273k to enable stable long‑context attention (to 8k–16k tokens), then fine‑tunes on a mix of long‑context and short‑context text corpora.
  - Stability limit: 1.7B LM variant trains to 16k tokens; 135M/360M are stable up to 8k tokens (text + vision tokens).

- Aggressive visual token compression (§2.2; Figure 3, middle‑right, and Figure 4):
  - Many VLMs choose shuffle ratio `r=2` to protect OCR and localization. Here, small models often benefit from `r=4`, because the reduced token count decreases attention overhead and improves long‑context modeling (Figure 3, middle‑right).

- Image and video handling (§2.3; Figure 3, right):
  - Image splitting helps small models keep detail without exploding tokens (tiles + small global image).
  - Video “frame averaging” (averaging multiple frames into one feature) hurts performance as the averaging factor grows (2→4→8), so it is avoided (Figure 3, right). Instead, frames are rescaled to the encoder’s input resolution.

- Positional encoding for split images and media segmentation (§3.1–3.2; Figures 5–6):
  - Learned positional tokens versus literal string tags: using raw strings such as `<row_1_col_2>` causes unstable plateaus in training (“OCR loss plague”) for small models; learned tokens stabilize optimization and improve OCR and overall scores (Figure 5, left and center).
  - Media intro/outro markers and concise system prompts disambiguate where visual content begins/ends and what the model’s role is; both improve zero‑shot performance, especially on video (Figure 6). During supervised fine‑tuning, masking user prompts (training only on completions) boosts generalization (Figure 6, right).

- Training data composition (§3.3–3.5; Figure 7; §4.1, Figure 8):
  - Two stages: Vision stage (heavy on OCR/docs, charts, tables, VQA, and some reasoning; Figure 8 left) and Video fine‑tuning stage (33% video, 35% image, 12% multi‑image, 20% text; Figure 8 right). Maintain only ~14% pure text to avoid overwhelming compact models with non‑visual data (§4.1).
  - Avoid reusing LLM SFT text (“SmolTalk”): it reduces image and video scores on small VLMs by 3.7% and 6.5% on average (Figure 7, left).
  - Use very sparse Chain‑of‑Thought (CoT): small fractions (0.02–0.05%) help slightly; higher fractions hurt, especially on image tasks (Figure 7, middle).
  - Moderate video durations (≈3.5 minutes) during training improve results; longer yields diminishing returns (Figure 7, right).

- Implementation note on context scaling:
  - Extending context used RoPE base scaling (Liu et al., 2024c). The `SmolVLM-2.2B` uses a 16k limit; smaller variants use an 8k limit (§2.2).

## 4. Key Insights and Innovations
- Balanced encoder–LM capacity for small VLMs (Finding 1; §2.1; Figure 3, left)
  - Novelty: Rather than defaulting to a powerful vision encoder, small LMs pair better with smaller encoders; larger encoders become beneficial only once the LM has enough capacity (≥1.7B).
  - Why it matters: Avoids over‑investing in vision capacity that the LM cannot utilize, saving parameters and memory at small scales.

- Aggressive but targeted visual token compression (Findings 2–4; §2.2–§2.3; Figures 3–4)
  - Novelty: For compact models, using pixel shuffle with `r=4`—more aggressive than the common `r=2`—improves performance by reducing attention load (Figure 3, middle‑right). Combined with a longer context window, this supports higher resolutions without exploding memory.
  - Significance: This departs from prior art that warns against strong compression due to OCR/localization; SmolVLM shows how to compensate (image splitting, learned positional tokens).

- Learned positional tokens and structured prompting for stability and OCR (Finding 5–6; §3.1–3.2; Figures 5–6)
  - Novelty: Replacing string‑based positional tags with learned embeddings eliminates the “OCR loss plague” and improves both image and video scores (Figure 5).
  - Prompting/segmentation tokens and masking user inputs during SFT yield consistent gains, especially on video (Figure 6).
  - Significance: Turns previously brittle training dynamics into stable ones for small multimodal models.

- Data curation rules for small VLMs (Findings 7–9; §3.3–3.5; Figure 7; §4.1, Figure 8)
  - Novelty: Counterintuitive empirical rules—do not reuse LLM SFT text blends; keep CoT minimal; limit average video duration to ~3.5 minutes—optimize capacity usage for small VLMs (Figure 7).
  - Significance: Provides a tested recipe for training compact VLMs without saturating them with text‑heavy or overly long video data.

These are mostly practical innovations grounded in systematic ablations rather than new theory; the novelty lies in the recipe and its interactions.

## 5. Experimental Analysis
- Evaluation setup (§4.2):
  - Toolkit: `VLMEvalKit` (Duan et al., 2024) for reproducibility.
  - Leaderboard: `OpenVLM` (by OpenCompass) and many benchmarks (31 total for the leaderboard; Figure 1). The paper emphasizes RAM usage as a more meaningful proxy for deployment cost than parameter count (§4.2).
  - Image preprocessing size: longest edge 1920 for `256M`/`500M`; 1536 for `2.2B` (§4.2).

- Benchmarks and metrics (Table 1):
  - Single‑image tasks: OCRBench (OCR), AI2D (science diagrams), ChartQA, TextVQA, DocVQA, ScienceQA.
  - Multi‑task: MMMU (college‑level), MathVista (visual math), MMStar (multidisciplinary).
  - Video: Video‑MME (general), MLVU (movie QA + MSRVTT caption), MVBench (multiview), WorldSense (temporal/physics), TempCompass (temporal).
  - Metrics are standard task accuracies or CIDEr where appropriate (Figure 3 caption notes averaging CIDEr and accuracy in their analyses).

- Main quantitative results (Table 1; §4.3):
  - Average across 14 benchmarks:
    > `SmolVLM-256M`: 44.0% | `SmolVLM-500M`: 51.0% | `SmolVLM-2.2B`: 59.8%
  - Memory usage (batch size 1):
    > 0.8 GB (256M), 1.2 GB (500M), 4.9 GB (2.2B) vs. 27.7 GB for `MolmoE‑A1B‑7B` (efficient large baseline in the table).
  - Selected single‑image tasks:
    - OCRBench:
      > 52.6% (256M) → 61.0% (500M) → 72.9% (2.2B) vs. 54.7% `MolmoE‑A1B‑7B`.
    - DocVQA:
      > 58.3% → 70.5% → 80.0% vs. 77.7% `MolmoE‑A1B‑7B`.
    - ScienceQA:
      > 73.8% → 80.0% → 89.6% vs. 87.5% `MolmoE‑A1B‑7B`.
    - MMMU (hard reasoning):
      > 29.0% → 33.7% → 42.0% (close to 33.9% baseline at small scales; improves with the 2.2B variant).
  - Selected video tasks:
    - Video‑MME:
      > 33.7% → 42.2% → 52.1% vs. 45.0% `InternVL2‑2B`.
    - WorldSense:
      > 29.7% → 30.6% → 36.2% vs. 32.4% `Qwen2VL‑7B`.
    - MVBench (challenging for SmolVLM at small scale):
      > 32.7% → 39.7% → 46.3% vs. 60.2% `InternVL2‑2B`.
  - Scaling trend:
    - Almost all tasks improve with model size; even `256M` often beats far larger historical models (Figure 1; §4.3).

- Throughput and on‑device viability (§4.4; Figure 9):
  - A100 GPU:
    > `256M`: 0.8 → 16.3 examples/s (batch 1→64)  
    > `500M`: 0.7 → 9.9 examples/s  
    > `2.2B`: 0.6 → 1.7 examples/s
  - NVIDIA L4 (edge server GPU):
    > Peaks: `256M` ~2.7 ex/s (batch 8); `500M` ~1.4 ex/s; `2.2B` ~0.25 ex/s
  - Browser/WebGPU on MacBook Pro (M4 Max):
    > Up to ~80 decode tokens/s for `256M`.
  - These support the claim that RAM (and per‑token cost) is a better deployment proxy than parameters (§4.2; Figure 9).

- Ablations and training strategy evidence:
  - Encoder–LM balance (Figure 3, left): large encoder hurts with 135M LM; only helps clearly at 1.7B.
  - Context length gains (Figure 3, middle): accuracy increases up to 16k tokens for the large model; small models are stable up to 8k.
  - Pixel shuffle ratio (Figure 3, middle‑right): `r=4` can outperform `r=2` in compact regimes.
  - Frame averaging (Figure 3, right): hurts video performance as averaging factor increases; thus excluded.
  - Learned positional tokens (Figure 5): fix training stalls (“OCR loss plague”) and yield higher scores than string tags.
  - Prompting and masking (Figure 6): system prompts + media intro/outro + user‑prompt masking each add gains; most pronounced on video.
  - Data mix (Figure 7): avoid LLM SFT text; keep CoT tiny; target ~3.5 min videos for training.

- Do the experiments support the claims?
  - Yes, for the paper’s scope. The combination of broad benchmark coverage (Table 1), explicit memory measurements, and many targeted ablations (Figures 3–7) makes a strong case that the proposed recipe yields compact, practical VLMs with competitive accuracy. Where results are mixed (e.g., MVBench), the paper is transparent, and the trends align with known difficulty of long‑range temporal reasoning for small models.

## 6. Limitations and Trade-offs
- Spatial detail vs. compression:
  - Aggressive pixel shuffle (`r=4`) reduces spatial resolution in the token sequence (§2.2). Although compensated with image splitting and learned positional tokens, fine‑grained localization tasks (e.g., dense OCR, small object localization) remain sensitive; the paper hints at this trade‑off when discussing why prior work defaults to `r=2` (Figure 4; §2.2).

- Video long‑range reasoning:
  - On MVBench, even the `2.2B` model reaches 46.3% vs. 60.2% for `InternVL2‑2B` (Table 1). This suggests that very long, complex spatio‑temporal reasoning remains a weakness for compact setups (also consistent with avoiding frame averaging and relying on rescaled frames; §2.3).

- Context stability constraints:
  - Smaller LMs (135M/360M) are stable up to 8k tokens rather than 16k (§2.2). This caps how many visual tokens (e.g., tiles + frames) can be processed at once for the smallest variants.

- Data sensitivity:
  - Small VLMs are sensitive to data composition—LLM SFT text hurts, and excessive CoT degrades performance (§3.3–3.4; Figure 7). This increases curation burden and may limit reuse of popular instruction‑tuning corpora.

- Throughput at larger size:
  - `SmolVLM‑2.2B` offers strong accuracy but comparatively low throughput on modest GPUs (e.g., ~0.25 ex/s on L4; Figure 9). For strict real‑time applications on edge hardware, the `256M/500M` variants are preferable.

- Evaluation scope:
  - While broad, the study focuses on open benchmarks in `VLMEvalKit` and OpenVLM. Domain‑specific edge cases (e.g., industrial inspection, complex multi‑page forms at 4K+) are not directly evaluated. The image resolution was capped (1920 or 1536 longest edge; §4.2).

## 7. Implications and Future Directions
- Field impact:
  - SmolVLM strengthens the case that carefully engineered small VLMs can be practical and competitive, shifting emphasis from parameter count to token efficiency and memory use (Figure 1; §4.3–4.4). It provides a reproducible recipe—balanced encoder–LM capacity, long context, aggressive compression with compensating mechanisms, lean data mixes—that others can adopt.

- Enabled follow‑ups:
  - Adaptive tokenization: dynamic pixel shuffle or learned, content‑aware compression that preserves detail where needed while keeping sequence length short elsewhere.
  - Memory‑efficient video: alternatives to frame averaging (e.g., temporal keyframe selection, learned segment pooling) that retain temporal cues without degrading performance (contrast Figure 3, right).
  - Stability at longer contexts for tiny LMs: optimization and architecture tricks to push 135M/360M models beyond 8k safely (§2.2).
  - Task‑adaptive prompting: learn media segmentation markers and system prompts end‑to‑end rather than hand‑engineering (Figures 5–6).
  - Mixture‑of‑experts at small scale: route only a subset of experts per token to preserve memory while expanding capacity (suggested by comparisons to `MolmoE‑A1B‑7B` in Table 1).

- Practical applications (demonstrated and suggested):
  - On‑device assistants and accessibility tools: demonstrated smartphone app (HuggingSnap) and browser deployment via WebGPU (Figure 9; §4.4).
  - Document AI at the edge: “Smol Docling” uses specialized tags for document conversion with just 256M parameters (§4.5).
  - Retrieval and QA for enterprise documents: “ColSmolVLM” and connections to efficient document retrieval (ColPali) (§4.5).
  - Domain‑specific VQA (e.g., biomedical “BioVQA”) where compute is constrained (§4.5).
  - Robotics and embedded systems: low‑VRAM inference widens feasibility for onboard perception‑language reasoning.

In short, SmolVLM contributes a validated blueprint for building capable, genuinely small VLMs. Its most durable insights are the interplay between token budget and model capacity, and the importance of training/data strategies tailored to compact settings—principles that will likely guide future efficient multimodal systems.
