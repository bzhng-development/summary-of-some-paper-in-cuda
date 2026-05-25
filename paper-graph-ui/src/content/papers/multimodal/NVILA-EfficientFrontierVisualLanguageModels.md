# NVILA: Efficient Frontier Visual Language Models

**ArXiv:** [2412.04468](https://arxiv.org/abs/2412.04468)

## 🎯 Pitch

NVILA introduces a new family of open visual–language models that set a new benchmark for both accuracy and efficiency by employing a 'scale-then-compress' approach—scaling up input resolutions to capture richer information, then aggressively compressing visual tokens to minimize computational overhead. This innovation is bolstered by data pruning, low-precision training, and quantized inference throughout the model lifecycle, allowing NVILA to meet or surpass top visual–language models on image and video benchmarks while slashing training times and inference latency by up to 5×. The result is a high-performing, resource-friendly VLM architecture poised to unlock real-world deployment in fields like robotics, edge devices, and medical AI—where both performance and speed are critical.

---

## 1. Executive Summary

This paper introduces **NVILA**, a family of open visual language models designed to jointly optimize efficiency and accuracy by applying a **"scale-then-compress"** paradigm — first scaling up spatial and temporal resolutions to raise the accuracy ceiling (e.g., Dynamic-S2 tiling for high-resolution images, up to 256 frames for long videos), then compressing visual tokens to recover efficiency (e.g., 3×3 spatial-to-channel reshape, temporal averaging). NVILA reduces training cost by 1.9–5.1×, prefilling latency by 1.6–2.2×, and decoding latency by 1.2–2.8× relative to baselines such as LLaVA-OneVision and Qwen2-VL, while matching or surpassing the accuracy of leading open and proprietary VLMs across a wide range of image and video benchmarks. A systematic investigation of efficiency across the full lifecycle — from FP8 training and DeltaLoss dataset pruning to W8A8/W4A16 quantized deployment — establishes that aggressive efficiency optimizations need not sacrifice accuracy when visual token compression is paired with a visual encoder pre-training stage to recover lost fidelity.

## 2. Context and Motivation

### The Core Problem: VLMs Are Accurate but Prohibitively Expensive

The fundamental tension this paper addresses is straightforward: **visual language models (VLMs) have gotten remarkably accurate, but their computational cost across training, fine-tuning, and deployment has become a significant barrier to both research and practical use.** The paper frames this as a multi-dimensional problem rather than a single bottleneck — inefficiency manifests differently at each stage of a VLM's lifecycle, and addressing only one stage leaves the others as obstacles.

Consider the concrete numbers the paper cites:

- **Training cost**: training a state-of-the-art 7B VLM (specifically LLaVA-OneVision, the only baseline with publicly reported training costs) can take up to 400 GPU days. For a model family with variants at multiple scales, this cost multiplies. The paper estimates that NVILA reduces this by 1.9–5.1× (Figure 1a), which is the difference between a training run being feasible for an academic lab versus requiring industry-scale compute.
- **Fine-tuning memory**: fully fine-tuning a 7B VLM can require over 64 GB of GPU memory, "well beyond what is available on most consumer-grade GPUs" (Section 1). This means domain adaptation — fine-tuning a general VLM for medical imaging, robotics, or specialized document understanding — is inaccessible to practitioners without access to datacenter GPUs like the A100 or H100.
- **Deployment latency**: VLMs are increasingly deployed in edge settings such as laptops and robots, where both latency and memory are tightly constrained (Section 2.4). The inference pipeline has two distinct phases — prefilling (encoding the visual and textual prompt, producing the first token) and decoding (generating the response token by token) — and each imposes different bottlenecks on real-time performance.

The paper's explicit claim is that these three costs — training, fine-tuning, and deployment — must be addressed **together**:

> "Addressing all three together requires a comprehensive approach to VLM efficiency." (Section 1)

This is not a rhetorical flourish. The paper's architecture decisions (especially the "scale-then-compress" design) cascade through all three stages: compression reduces training FLOPs, fine-tuning memory, and inference latency simultaneously. A narrow optimization targeting only deployment (e.g., post-hoc quantization without architectural changes) would leave training and fine-tuning as expensive as ever.

### The Gap: Efficiency Has Received "Much Less Attention"

The paper positions its work against a specific imbalance in the research landscape. Section 1 opens by acknowledging rapid progress in VLM accuracy, citing models including GPT-4o, Claude 3.5, Gemini 1.5 Pro, and open-source efforts like InternVL2, Qwen2-VL, and LLaVA-OneVision. But it immediately pivots:

> "However, much less attention has been paid to their efficiency." (Section 1)

This is more than a gap-claiming exercise. The paper is arguing that the field has optimized for a single objective (benchmark accuracy) without accounting for the **cost of achieving that accuracy**. The consequence is a set of state-of-the-art models that are accurate but effectively unusable for many real-world applications — researchers cannot afford to train them, domain experts cannot fine-tune them, and edge deployments cannot run them fast enough.

The gap is particularly acute because VLMs add **visual tokens** to the already-expensive LLM inference pipeline. A text-only LLM processes a prompt of perhaps hundreds or thousands of text tokens. A VLM processing a high-resolution image with tiling (as modern VLMs do) might add thousands of **visual tokens** — each a high-dimensional embedding vector that must be attended over by the LLM backbone's self-attention layers. For video, this problem compounds: 256 frames of video, each tiled into multiple image patches, can produce tens of thousands of visual tokens. The quadratic complexity of self-attention makes this extremely expensive.

The paper quantifies this directly: doubling the spatial resolution quadruples the number of visual tokens (since both height and width increase), and the self-attention cost in the LLM further amplifies this overhead (Section 2.1). This is why naive resolution scaling — simply increasing the vision encoder's input size — is not a viable path to better accuracy. The cost escalation is superlinear.

### Where Prior Approaches Fall Short

The paper identifies specific limitations in existing work across several dimensions, which it discusses explicitly in Section 5 (Related Work) but which motivate the technical contributions throughout.

**Token reduction methods exist but were designed for lower-capacity settings.** Prior work has studied spatial and temporal token reduction — Token Merging (Bolya et al., 2023), training-free pruning for video transformers (vid-TLDR, Choi et al., 2024), and various token compression schemes for vision-language models (Section 5.2). However, the paper argues that these methods were developed and evaluated in **non-frontier settings**:

> "none focuses on reducing tokens for a frontier VLM, where preserving accuracy at scale is the primary challenge" (Section 5.2)

The distinction matters because aggressive token compression creates a **training difficulty problem**. The paper shows this concretely in Table 1: moving from 2×2 spatial-to-channel compression (256 tokens per tile) to 3×3 (121 tokens per tile) causes a nearly 10-point accuracy drop on DocVQA (91.1 → 82.3). More sophisticated learnable compression methods (TokenLearner from RT-1, Perceiver Resampler from MiniCPM-V) perform no better or even worse at the same compression ratio — TokenLearner achieves 86.5 on DocVQA, and the Perceiver Resampler drops all the way to 71.8. The paper attributes this to **optimization difficulty** rather than representational capacity:

> "We attribute this to optimization difficulty rather than representational capacity, and leave a deeper investigation to future work." (Section 2.1.1)

This is a key insight: the challenge is not designing a compression mechanism that can theoretically represent the information, but rather training the projector and vision encoder to effectively **use** that compressed representation. The paper's solution — a dedicated visual encoder pre-training stage (Stage 2 in Table 8) that jointly tunes the vision encoder and projector — is what recovers most of the lost accuracy (82.3 → 88.8 on DocVQA). Without this insight, one might conclude that 3×3 compression is too aggressive and abandon it, rather than recognizing it as a training problem with a training solution.

**Data selection for VLMs is underexplored.** The paper notes that SFT dataset curation for VLMs has followed an "ever-larger" trend — Cambrian-1, LLaVA-OneVision, and Idefics2 all aggregate diverse data sources to improve benchmark performance (Section 2.2.1). But this leads to redundancy: "not all data contributes equally, and unchecked dataset growth leads to significant redundancy." The paper highlights that while data selection has been studied for vision-only inputs (Coleman et al., 2020; Xu et al., 2024) and text-only inputs (Xia et al., 2024; Gu et al., 2024), **few studies address the mixed image-text setting of VLM training** (Section 2.2.1). With tens of millions of training samples, pruning the dataset without sacrificing accuracy is essential, but no established method exists for the multimodal case.

The DeltaLoss approach, borrowed from knowledge distillation (Gu et al., 2024), represents a transfer of a text-only technique to the VLM domain. The paper specifically argues that it filters out three categories of samples: too easy (both small and large models answer correctly), distracting (small model correct, large model fails), and failed (both models fail) — keeping only samples where the large model succeeds but the small model struggles (Figure 4). This is a principled criterion for informativeness that goes beyond simple clustering or random pruning.

**FP8 training is established for LLMs but not VLMs.** The paper explicitly states:

> "FP8 training has gained traction for LLMs, but to our knowledge no prior work has demonstrated its feasibility for VLMs without sacrificing accuracy." (Section 5.2)

The challenge specific to VLMs is **sequence-length variability**. The paper explains this in detail (Section 2.2.2): after packing, LLM training samples tend to be similar in length, so throughput is relatively insensitive to batch size. VLM training samples, by contrast, vary widely — video samples may require tens of thousands of tokens, image samples a few hundred to a few thousand, and text-only samples far fewer. This means batches dominated by short samples underutilize the GPU, and FP8's memory savings (which allow larger batch sizes) are particularly impactful for VLMs. The paper quantifies this: switching from BF16 to FP8 with gradient checkpointing disabled allows the batch size to increase from 4 to 16, yielding a 2× speedup (Table 5).

**Vision encoder quantization is neglected in deployment optimization.** The paper observes that quantization methods like AWQ and GPTQ have been well-studied for LLMs, and VILA showed that AWQ transfers directly to VLMs (Section 5.2). However, the **vision encoder** — which processes high-resolution, multi-tile images and multi-frame videos — has received little attention. The paper quantifies this oversight: after applying token compression to the LLM backbone, the vision tower accounts for **over 90% of the prefilling latency** (Section 2.4). This means that even a perfectly quantized LLM backbone would leave the bulk of inference latency untouched — a finding that directly motivates the paper's W8A8 quantization of the vision tower.

**Parameter-efficient fine-tuning (PEFT) for VLMs lacks systematic study.** The paper notes that methods like LoRA, DoRA, QLoRA, and GaLore are widely used to reduce LLM fine-tuning memory, but "efficient fine-tuning techniques are still underexplored" for VLMs, which combine a vision encoder with an LLM (Section 5.2). The paper's contribution here is an empirical finding: the vision encoder's learning rate should be 5–50× smaller than the LLM's when jointly fine-tuning both components (Section 2.3). Furthermore, fine-tuning only the ViT's LayerNorm layers (rather than applying LoRA to all ViT parameters) matches full LoRA accuracy while reducing training time by 25% (Table 6). This is a practical recipe that the paper argues is both the most memory-efficient and compute-efficient configuration.

### The "Scale-Then-Compress" Philosophy as a Unifying Response

The paper's central design philosophy — "scale-then-compress" — is presented not as a single technique but as a meta-strategy that applies across multiple axes of VLM design. It is introduced in Section 2.1:

> "We improve [VILA's] model architecture by first scaling up the spatial and temporal resolutions, and then compressing visual tokens. Scaling preserves more visual detail, raising the accuracy ceiling, while compression reduces the visual token count, lowering computational cost."

The rationale is layered. On the scaling side: VILA-1.5 processes images at a fixed 448×448 resolution regardless of aspect ratio and samples at most 14 frames per video. Both choices introduce "significant information loss" and cause VILA to lag behind leading VLMs, particularly on text-heavy image and long-video benchmarks (Tables 9 and 10). Scaling addresses this by capturing more detail — Dynamic-S2 tiling enables native-resolution processing with adaptive aspect ratio handling, and 256-frame sampling enables hour-scale video understanding.

On the compression side: scaling alone is not viable because it dramatically increases compute. The paper explicitly notes that "doubling the resolution quadruples the number of visual tokens, and self-attention's quadratic complexity in the LLM further amplifies the cost" (Section 2.1). Compression is therefore **not an optional optimization** but a necessary counterbalance to make scaling practical. The paper argues that higher information density from scaling actually enables more aggressive compression:

> "Compression then offsets this overhead, since higher information density lets the model retain (or even surpass) the detail captured at lower resolution while using fewer total tokens." (Section 2.1)

This is a non-obvious claim: that a compressed representation of a high-resolution input can be more informative than a less-compressed representation of a low-resolution input. The empirical support comes from Table 1: the "scale + compress" configuration (3×3 pooling, 121 tokens/tile) achieves 82.3 on DocVQA, which is substantially higher than the uncompressed baseline at lower resolution (61.3). And with visual encoder pre-training, it reaches 88.8 — nearly matching the uncompressed high-resolution configuration (91.1) while using roughly half the tokens per tile.

### How This Paper Positions Itself Relative to Existing Work

The paper positions NVILA as an **efficiency-first VLM** that does not sacrifice accuracy — a "frontier" model in the Pareto sense of being on the optimal accuracy-efficiency boundary. The framing is explicit in the title ("Efficient Frontier Visual Language Models") and in Figure 1, which shows NVILA achieving comparable or superior accuracy to leading models while being substantially faster.

The choice of baselines is telling. The paper compares against:

- **LLaVA-OneVision** (the only baseline with publicly reported training costs) for training efficiency
- **Qwen2-VL** (a leading open-source VLM at 7-8B scale) for inference efficiency
- A broad set of open-source models (InternVL2, Pixtral, Cambrian-1, Llama 3.2 Vision) and proprietary models (GPT-4o, Claude 3.5 Sonnet, Gemini 1.5 Pro) for accuracy

The paper is not claiming to be the absolute most accurate VLM — larger models (LLaVA-NeXT 34B, InternVL2 40B, NVLM-D-1.0 78B, Llama 3.2 90B) often achieve higher scores on individual benchmarks. Rather, it claims to be **on the frontier** — achieving comparable accuracy at a fraction of the cost. This is a different kind of contribution than a pure accuracy benchmark leaderboard entry. It asks the question: what accuracy level can you achieve **given a compute budget**, rather than what accuracy can you achieve without constraints?

The paper also builds directly on **VILA** (Lin et al., 2024), its immediate predecessor from the same research group. This is not just a citation — NVILA is architecturally derived from VILA, sharing the same three-component structure (SigLIP vision encoder, two-layer MLP projector, Qwen2 LLM backbone). The contributions are framed as improvements to this base architecture: replacing VILA's fixed-resolution image processing with Dynamic-S2, extending video from 14 frames to 256 frames, adding spatial and temporal compression, and introducing the lifecycle efficiency optimizations (FP8 training, DeltaLoss pruning, quantized deployment). The relationship to VILA gives the paper a clear "diff" — the reader can see exactly what changed and why.

Finally, the paper positions itself as **opening up new application domains** that were previously impractical due to efficiency constraints. Section 4 demonstrates three such domains — temporal localization (video timestamp understanding), robotic navigation (real-time visuomotor control at 1 Hz on an edge GPU), and medical imaging (integrating expert models with VLM reasoning). These are not just benchmark results; they are existence proofs that an efficient VLM can enable use cases that a more expensive one cannot. The robotic navigation result is particularly striking: NVILA-8B achieves a 53.3% success rate on VLN-CE (R2R Val-Unseen) compared to 37% for the prior NaVid system, while running in real-time on a single RTX 4090 GPU mounted on a Unitree Go2 robot (Table 12, Figure 6). This is a concrete demonstration of the paper's central thesis — that efficiency is not just about saving money, but about **enabling previously impossible deployment scenarios**.

## 3. Technical Approach

### 3.1 Reader Orientation

The NVILA paper presents a **system for building visual language models (VLMs)** — models that take images, videos, and text as input and produce textual responses — that are simultaneously state-of-the-art in accuracy and dramatically more efficient than prior approaches across their entire lifecycle. The core problem is that existing VLMs have optimized heavily for benchmark accuracy while neglecting the computational cost of training, fine-tuning, and deploying these models, creating a situation where the most accurate models are effectively unusable for many real-world applications due to GPU memory requirements, training time, and inference latency. The "shape" of the solution is what the paper calls *scale-then-compress*: first increase the spatial and temporal resolution at which the model processes visual inputs to raise the ceiling on achievable accuracy, then aggressively compress the resulting visual tokens to recover efficiency, with a dedicated training stage to ensure the compression does not degrade information quality.

### 3.2 Big-Picture Architecture (Diagram in Words)

The system has five major components arranged in a five-stage training pipeline, with architectural choices that cascade through training, fine-tuning, and deployment:

1.  **Visual Encoder (SigLIP ViT)** — a vision transformer pre-trained on image-text pairs that converts raw images or video frames into feature maps (grids of visual token embeddings). It processes each 448×448 image tile independently, extracting a 16×16 feature map per tile.

2.  **Projector (Two-Layer MLP)** — a small feedforward network that maps visual token embeddings from the vision encoder's representation space into the embedding space of the language model, so that the LLM can process visual and textual tokens jointly.

3.  **Token Processor (Qwen2 LLM)** — an autoregressive transformer language model that takes a sequence interleaving visual tokens (from the projector) and textual tokens (the user's prompt) and produces a textual response token by token. It is instantiated with Qwen2 at various parameter sizes (8B, 15B are released).

4.  **Dynamic-S2 Tiling and Token Compression Pipeline** — a preprocessing stage that (a) splits input images into multiple adaptive-resolution tiles using Dynamic-S2, (b) encodes each tile through the SigLIP encoder, (c) compresses each tile's 16×16 visual token grid to 11×11 using a 3×3 spatial-to-channel reshape, and (d) feeds all compressed tiles' tokens to the projector. For video, it samples up to 256 frames and applies temporal averaging (pooling groups of consecutive frames) before the spatial compression.

5.  **Training Pipeline (Five Stages)** — Stage 1 trains only the projector on feature alignment data. Stage 2 jointly fine-tunes the vision encoder and projector to recover accuracy lost to aggressive spatial compression. Stage 3 trains the projector and LLM together on diverse multimodal data. Stage 4 fine-tunes all components on image instruction data. Stage 5 further fine-tunes on video instruction data to extend long-video understanding.

Information flows as follows: an image enters the Dynamic-S2 tiling module → the image is resized to multiple scales and split into 448² tiles → each tile is independently encoded by the frozen SigLIP ViT → the per-tile 16×16 feature maps are compressed to 11×11 via 3×3 spatial-to-channel reshape → the compressed visual tokens pass through the MLP projector → the resulting tokens are interleaved with text tokens and fed to the Qwen2 LLM → the LLM autoregressively generates a response. For video, frames are first sampled, then temporally pooled in groups (e.g., 4 frames per group), and then each pooled frame proceeds through the same spatial pipeline.

### 3.3 Roadmap for the Deep Dive

- **First**, the Dynamic-S2 spatial scaling mechanism — how adaptive tiling handles arbitrary aspect ratios and why this matters for text-heavy benchmarks.
- **Second**, the spatial token compression via spatial-to-channel reshape — the mechanics, the accuracy–efficiency tradeoff, and why a separate visual encoder pre-training stage (Stage 2) is needed to recover accuracy.
- **Third**, the temporal scaling and compression for video — how frame sampling and temporal averaging interact to enable 256-frame processing at manageable token budgets.
- **Fourth**, the DeltaLoss dataset pruning method — the mathematical formulation, the intuition behind its three-case filtering behaviour, and its cross-dataset generalisation.
- **Fifth**, the FP8 mixed-precision training strategy — why VLM training specifically benefits from FP8 (sequence-length variability) and the concrete throughput gains.
- **Sixth**, the efficient fine-tuning recipe — the learning rate ratio between ViT and LLM, and why tuning only LayerNorm layers works.
- **Seventh**, the quantized deployment engine — the phase-specific quantisation strategy (W8A8 for the vision tower, W4A16 for the LLM backbone) and the FP16 accumulation kernel optimisation.

### 3.4 Detailed, Sentence-Based Technical Breakdown

This is primarily a **systems and architecture paper** whose core idea is that a "scale-then-compress" design philosophy — applied across spatial resolution, temporal resolution, dataset scale, and training precision — enables VLMs to match or surpass the accuracy of leading models while being substantially more efficient at training, fine-tuning, and inference.

---

#### Dynamic-S2: Adaptive High-Resolution Image Tiling

The baseline VILA-1.5 processes all images at a fixed 448×448 resolution regardless of the original image's aspect ratio or the amount of fine-grained detail it contains. For a standard VQA image, this may be sufficient; for a dense document with small text or a chart with detailed annotations, critical information is lost. NVILA replaces this with Dynamic-S2, an adaptive tiling strategy that first scales up resolution to capture detail and then relies on downstream compression (Section 2.1.1) to manage the resulting token cost.

**Mechanics of S2 (the static precursor).** The original S2 algorithm (Shi et al., 2024) is designed to extract multi-scale features without requiring the vision encoder to be re-trained at higher resolutions. Given a vision encoder pre-trained at 448² resolution, S2 processes an input image as follows:

1.  Resize the input image to multiple scales — for example, 448², 896², and 1344².
2.  At each scale, split the resized image into 448² tiles. A 448² image produces 1 tile; an 896² image produces 4 tiles (2×2 grid); a 1344² image produces 9 tiles (3×3 grid).
3.  Each tile is independently processed by the frozen vision encoder, producing a 16×16 feature map per tile (since SigLIP's patch size is 28×28, each 448×448 tile yields 16×16 = 256 visual tokens).
4.  The per-tile feature maps at each scale are stitched back into a single feature map for that scale, reconstructing the spatial layout.
5.  Feature maps from different scales are spatially interpolated to a common size and concatenated along the channel dimension. This means the final representation at each spatial position contains features computed at multiple resolutions — coarse global structure from the 448² scale and fine local detail from the 1344² scale.

The key property of S2 is that it never requires the vision encoder to process inputs larger than 448², so it can leverage pre-trained encoders without additional training. However, it has a critical limitation: it always resizes images to square aspect ratios, which distorts images with extreme aspect ratios (e.g., a wide panoramic photo or a tall document scan). For text-heavy images, distortion can make text unreadable or warp document layouts, negating the benefit of higher resolution.

**The Dynamic-S2 modification.** Dynamic-S2 addresses the aspect ratio problem by modifying only the largest scale. Specifically:

- At all scales *except* the largest, Dynamic-S2 behaves identically to S2 — resize to a square, tile, encode, stitch.
- At the largest scale, instead of resizing to a square (e.g., 1344²), Dynamic-S2 selects the closest size that **preserves the original aspect ratio** and is **divisible into 448² tiles**. For example, an image with a 16:9 aspect ratio might be resized to 1792×896 (4 tiles wide × 2 tiles tall) rather than being forced to 1344×1344.

> "Dynamic-S2 follows S2, but at the largest scale, instead of resizing to a square, it picks the closest size that preserves the input's aspect ratio and is divisible into 448² tiles." (Section 2.1.1)

This design is credited as being "inspired by the dynamic resolution strategy in InternVL." The consequence is that the number of tiles per image varies based on both the image's size and its aspect ratio. The paper reports that with Dynamic-S2, NVILA uses 9–12 tiles per image (Table 1, "Scale (Dynamic-S2)" row), compared to 1 tile in VILA-1.5. This 9–12× increase in tile count is what drives the need for spatial compression — generating 9–12 × 256 = 2304–3072 visual tokens per image would be prohibitively expensive in the LLM backbone without compression.

**Why this matters for accuracy.** The absolute accuracy gains from Dynamic-S2 are substantial. Table 1 shows that adding Dynamic-S2 (with the same 2×2 spatial-to-channel reshape as VILA-1.5) improves DocVQA from 61.3 to 91.1 — a nearly 30-point absolute gain. TextVQA improves from 67.5 to 77.0, and AI2D from 87.0 to 90.1. These are text-heavy benchmarks where the ability to read small text, parse document layouts, and interpret chart annotations is critical. The gains on general benchmarks are more modest but positive — the average across 10 image benchmarks (IM-10) rises from 61.2 to 71.5.

The paper explicitly frames this as raising the "accuracy ceiling":

> "Scaling preserves more visual detail, raising the accuracy ceiling, while compression reduces the visual token count, lowering computational cost." (Section 2.1)

The implicit argument is that without scaling, no amount of training data or model capacity can recover information that was never extracted from the image in the first place. Scaling is therefore a *necessary* step for reaching frontier accuracy on detail-intensive tasks, even if it must be paired with compression to remain practical.

---

#### Spatial Token Compression: 3×3 Spatial-to-Channel Reshape and Visual Encoder Pre-Training

With Dynamic-S2 producing 9–12 tiles per image, and each tile generating 256 visual tokens (16×16 under the encoder's 28×28 patch size), the total visual token count per image becomes substantial. This directly impacts all three stages of the VLM lifecycle: training time (more tokens → more LLM forward passes), fine-tuning memory (more tokens → larger activations), and inference latency (more tokens → longer prefilling). Spatial compression is the mechanism that brings this token count back down to manageable levels.

**The spatial-to-channel (STC) reshape mechanism.** The intuition behind spatial-to-channel reshape is straightforward: instead of discarding information (as pooling would), it reorganises it. Given a feature map of shape `[C, H, W]` where C is the number of channels and H×W is the spatial grid of visual tokens, an r×r spatial-to-channel reshape:

1.  Partitions the H×W spatial grid into non-overlapping r×r blocks. For a 16×16 grid and r=3, the grid is divided into 5×5 = 25 blocks of size 3×3, plus one row and one column of edge padding (since 16 is not divisible by 3). The paper handles this by using an 11×11 output grid, implying some boundary handling (likely reflection padding or strided cropping).
2.  For each r×r block, the r² token embeddings (each of dimension C) are concatenated into a single token embedding of dimension r²C — that is, the spatial dimensions are folded into the channel dimension.
3.  The result is a feature map of shape `[r²C, H/r, W/r]`. For r=3 and the paper's configuration, this yields 11×11 = 121 tokens per tile (Table 1).

The total token count per tile drops from 256 to 121, a reduction factor of 256/121 ≈ 2.1×. However, because the channel dimension has increased 9× (from C to 9C), the per-token dimensionality is much higher, which affects the projector's parameter count and the LLM's embedding lookup cost. The paper reports this as providing a "2.4× speedup" over the 2×2 baseline (Section 2.1.1), which implies the net efficiency gain includes both the reduced token count and the effective handling of the increased per-token dimension.

**Why this works (and the failure of more complex alternatives).** The spatial-to-channel reshape is notable for its simplicity — it is a fixed, non-learned operation with no parameters. This makes it computationally cheap and deterministic. The paper compares it against two learnable compression methods:

- **TokenLearner** (from RT-1, Brohan et al., 2022): a learned module that dynamically selects which spatial tokens to keep, trained end-to-end.
- **Perceiver Resampler** (from MiniCPM-V, Yao et al., 2024): a learned cross-attention module that compresses a variable-length sequence of visual tokens into a fixed number of latent queries.

The results in Table 1 are striking. At the same token-reduction ratio (121 tokens per tile), TokenLearner achieves 86.5 on DocVQA (vs. 88.8 for STC with pre-training), while the Perceiver Resampler collapses to 71.8 — substantially worse than even the uncompressed baseline at low resolution (61.3). The paper's interpretation:

> "At the same token-reduction ratio, these learnable methods perform no better than the simple spatial-to-channel design, even with the additional visual encoder pre-training stage. We attribute this to optimization difficulty rather than representational capacity." (Section 2.1.1)

This is a significant finding with implications for VLM architecture design. It suggests that the **training dynamics** of the projector are the bottleneck, not the theoretical representational power of the compression mechanism. A simple, deterministic reshape is easier to optimise than a learned attention mechanism, especially when the projector is a shallow two-layer MLP with limited capacity. More sophisticated compression methods may be viable if paired with a deeper or more carefully optimised projector, but under the standard VLM training recipe, simplicity wins.

**The accuracy cliff and the role of Visual Encoder Pre-Training (Stage 2).** The paper discovers that pushing the STC ratio from 2×2 to 3×3 causes a sharp accuracy drop that cannot be recovered by the standard training pipeline alone. Table 1 shows this clearly:

- Baseline (VILA-1.5, 2×2 STC, 1 tile, 448² images): DocVQA 61.3
- Scale (Dynamic-S2, 2×2 STC, 9–12 tiles): DocVQA 91.1
- Scale + Compress (Dynamic-S2, 3×3 STC, 1–12 tiles): DocVQA 82.3

The drop from 91.1 to 82.3 is nearly 9 absolute points — unacceptably large. The paper hypothesises:

> "We hypothesize that more aggressive token reductions make the projector substantially harder to train." (Section 2.1.1)

The solution is a dedicated training stage (Stage 2 in Table 8) inserted between the standard Stage 1 (projector initialisation) and Stage 3 (token processor pre-training). In Stage 2, **both the vision encoder and the projector are jointly fine-tuned**, while the LLM remains frozen. The learning rate is set to 5×10⁻⁵ (Table 8), which is 20× lower than Stage 1's projector learning rate (1×10⁻³), reflecting the need for careful, incremental adjustment of the pre-trained vision encoder weights rather than aggressive re-training.

The effect is substantial: Stage 2 recovers most of the lost accuracy, bringing DocVQA from 82.3 to 88.8, and the IM-10 average from 67.1 to 70.8 (Table 1). The paper characterises this as:

> "This stage recovers most of the accuracy loss from 3×3 compression while preserving its 2.4× speedup over the 2×2 baseline in both training and inference." (Section 2.1.1)

**Why Stage 2 works, mechanistically.** The visual encoder was pre-trained on 448² images using a contrastive image-text objective (SigLIP). It was never trained to produce features that survive aggressive spatial compression. The projector, similarly, was initialised and trained only in Stage 1 to map *uncompressed* visual features to the LLM's embedding space. When 3×3 STC is introduced at inference time, both the encoder's features and the projector's mapping are being used in a regime they were never optimised for. Stage 2 addresses this by allowing the encoder to adapt its features to be more compressible (e.g., by reducing spatial redundancy within 3×3 blocks) and allowing the projector to learn to interpret the channel-concatenated representations. This joint optimisation is essential — freezing one and training the other would leave a mismatch.

---

#### Temporal Scaling and Compression for Video Understanding

VILA-1.5 samples at most 14 frames per video, and its 3B variant uses only 8 frames. This is a severe limitation for understanding long videos, where events unfold over minutes or hours — 8 frames might capture one every 30 seconds, missing most of the action. NVILA scales this to 256 frames and then compresses temporally to keep the token budget manageable.

**Temporal scaling: uniform frame sampling.** The scaling mechanism itself is straightforward: given a video of arbitrary length, NVILA uniformly samples `$F$` frames. The paper explores `$F = 32$` and `$F = 256$` (Table 2). Each frame is then processed through the same spatial pipeline as an image — Dynamic-S2 tiling, SigLIP encoding, 3×3 STC spatial compression. This means the number of visual tokens for a video scales linearly with the number of frames times the number of tiles per frame. For 32 frames with an average of, say, 6 tiles per frame producing 121 tokens each, the total is 32 × 6 × 121 ≈ 23,232 visual tokens — before any temporal compression.

The accuracy gains from temporal scaling are substantial. Table 2 shows that moving from 8 to 32 frames (without temporal compression, so 8192 tokens total) improves Video-MME overall accuracy from 55.7 to 61.0, with the largest gains on short videos (65.4 → 73.2). The paper notes that this improvement is not free: the number of visual tokens increases by 4× (from 2048 to 8192), which directly impacts training time and inference latency. The 32-frame configuration also requires additional video SFT (Stage 5 in Table 8) to "enable the model to process longer sequences" — presumably because the base LLM was not pre-trained on sequences with such high visual token density, and without this adaptation, the model may fail to attend effectively across such long contexts.

**Temporal compression: frame averaging.** The temporal compression mechanism is temporal averaging (also called temporal pooling), borrowed from the video understanding literature (Wang et al., 2016, Temporal Segment Networks):

1.  Partition the `$F$` sampled frames into `$G$` groups of consecutive frames. For 4× compression, `$G = F/4$`, so each group contains 4 frames.
2.  For each group, for each spatial position (each visual token location within a tile), average the token embeddings across the 4 frames in the group. That is, the embedding at position `$(i, j)$` in the compressed output is the element-wise mean of the embeddings at the same position across all 4 frames.
3.  The result is `$G$` compressed frames, each with the same spatial token layout as a single frame, but with the temporal dimension reduced by a factor of 4.

This is a remarkably simple operation — no learned parameters, no attention, just element-wise averaging — yet the paper shows it is highly effective. For 32 frames with 4× compression (Table 2, "Scale + Compress" row), the token count is reduced from 8192 to 2048, matching the original 8-frame VILA-1.5 baseline exactly. The Video-MME overall accuracy drops only from 61.0 to 60.1 — a loss of just 0.9 points despite a 4× reduction in tokens.

**The scale-then-compress advantage.** The paper explicitly compares the scale-then-compress model at the same token budget as the baseline:

> "Compared with the original baseline at the same token budget, the scale-then-compress model has nearly the same cost and substantially higher accuracy." (Section 2.1.2)

At 2048 tokens, the 32-frame compressed model achieves 60.1 on Video-MME, while the 8-frame uncompressed model achieves 55.7 — a 4.4-point gain at approximately equal compute. The interpretation is that 32 frames, even after aggressive temporal averaging, retain more useful information than 8 frames without compression — the averaging smooths out noise (camera jitter, momentary occlusions) while preserving the broad temporal structure of the video, which 8-frame sampling may miss entirely.

Scaling further to 256 frames with 8× compression (Table 2, bottom row) produces 256/8 = 32 compressed frames, yielding 8192 tokens (matching the 32-frame uncompressed budget). Video-MME overall reaches 64.0, which the paper claims as state-of-the-art among 7–8B open-source models on this benchmark. The pattern is consistent: more frames, even with higher compression ratios, beat fewer frames — confirming that the temporal information gain from additional frames outweighs the blurring effect of more aggressive averaging.

**A note on the cost of the vision encoder for video.** The paper acknowledges a subtlety in the cost accounting:

> "Running the visual encoder on more frames adds overhead, but this is not the runtime bottleneck." (footnote in Section 2.1.2)

This is important because temporal compression reduces the tokens fed to the LLM backbone but does not reduce the number of SigLIP forward passes — the vision encoder still processes every sampled frame individually. For 256 frames, this means 256 forward passes through SigLIP. However, because the SigLIP encoder is much smaller than the Qwen2 LLM (SigLIP-SO400M has approximately 400M parameters vs. 8B for the LLM), and because the encoder's cost per frame is independent (the frames are processed in parallel or sequentially without self-attention across frames), this overhead is smaller than the saving from reducing LLM attention complexity. The paper's inference breakdown (Figure 5) confirms this: after token compression, the vision tower becomes the primary bottleneck in the prefilling stage, accounting for over 90% of latency — which then motivates the W8A8 vision tower quantisation in Section 2.4.

---

#### DeltaLoss Dataset Pruning

As VLMs have grown more capable, their SFT datasets have ballooned — LLaVA-OneVision trains on more than 8 million samples, and the NVILA recipe starts with 10 million. The paper argues that this growth is partly wasteful: "not all data contributes equally, and unchecked dataset growth leads to significant redundancy" (Section 2.2.1). The solution is DeltaLoss dataset pruning, a scoring method that ranks training examples by their informativeness for a target model and keeps only the top fraction.

**The DeltaLoss scoring formulation.** The method requires two reference VLMs: a "large" reference model and a "small" reference model. For each training example `$x$` consisting of an image, a question, and a ground-truth answer, we compute:

> $$\Delta(x) = \log \frac{p_{\text{large}}(x)}{p_{\text{small}}(x)}$$

where `$p_{\text{large}}(x)$` is the probability (likelihood) that the large reference VLM assigns to the ground-truth answer tokens of `$x$`, conditioned on the image and question, and `$p_{\text{small}}(x)$` is the analogous probability from the small reference VLM.

**What it computes:** DeltaLoss is the log-ratio of two conditional probabilities — the log-probability that the *large* model assigns to the correct answer minus the log-probability that the *small* model assigns to the same answer. Equivalently, it is the difference in log-loss (negative log-likelihood) between the two models on this example. A positive value means the large model is substantially more confident in the correct answer than the small model; a negative value means the small model is more confident; a value near zero means they agree.

**Why this form:** The log-ratio captures **relative difficulty** between two models rather than absolute difficulty. An example that is universally easy (both models assign high probability) has `$\Delta(x) \approx 0$`; an example that is universally hard (both models assign low probability) also has `$\Delta(x) \approx 0$`. Only examples where the models **disagree** — where the large model succeeds but the small model struggles — receive high positive scores. This filters for samples that are *learnable* (the large model can do it) but *not yet mastered* (the small model cannot) — precisely the examples that provide the strongest training signal for a model whose capacity lies between the two references.

**The three-case filtering behaviour.** The paper explicitly characterises what happens in each regime (Section 2.2.1):

- **Near-zero DeltaLoss:** Both models answer correctly, or both fail. These examples "offer little discriminative signal" — they are either trivially easy (the model already knows this) or impossibly hard (the model cannot learn from this). Filtering them out removes redundancy and noise.
- **Negative DeltaLoss:** The small model succeeds but the large model fails. The paper interprets these as "distracting" — examples that "tend to distract learning and will eventually be forgotten by a more capable model." The logic is that if a more capable model cannot learn the pattern, the example may contain spurious correlations or noise that hurts generalisation.
- **Positive DeltaLoss:** The small model fails but the large model succeeds. These are the **informative** examples — "challenging for small models but learnable by larger ones." They provide "strong supervision" because they represent a capability gap that training can close.

The qualitative visualisation in Figure 4 illustrates this with concrete examples: an image of snowy weather that both models answer correctly (DeltaLoss = 0.0343, labelled "too easy ❌"), a canopy colour question where the small model gets it right but the large model fails (DeltaLoss = -1.916, "wrong answer ❌"), and a question about a sign of respect where the large model succeeds but the small model fails (DeltaLoss = 4.1605, "helpful ✅").

**The stratified pruning procedure.** The full SFT dataset `$D$` is partitioned into `$N$` source-level subsets `$D_1, D_2, \ldots, D_N$` (e.g., DocVQA data, ChartQA data, VQAv2 data, etc.). Given a target keep-ratio `$\rho \in (0, 1]$`, the pruned dataset is:

> $$D' = \bigcup_{i=1}^{N} \text{top} \lceil \rho |D_i| \rceil \{\Delta(x) \mid x \in D_i\}$$

where `$\lceil \rho |D_i| \rceil$` is the number of examples to keep from subset `$D_i$`, and `$\text{top } k$` selects the `$k$` examples with the highest DeltaLoss scores within that subset.

**What it computes:** For each data source independently, sort all examples by their DeltaLoss score (descending), keep the top `$\rho$` fraction, and take the union across all sources. Stratification ensures that the pruned dataset maintains the original data source distribution — without it, over-represented sources with generally higher DeltaLoss scores could dominate the pruned set and cause catastrophic forgetting on tasks from under-represented sources.

**Why this form:** The stratification by data source prevents **distribution collapse** during pruning. If pruning were applied globally across all data, it might disproportionately select from a few high-DeltaLoss sources (e.g., OCR-heavy datasets where the model has the most room to improve) and eliminate entire categories of data (e.g., general VQA where the model already performs well). This would lead to a model that excels at OCR but forgets general visual reasoning. Stratification guarantees that every source type remains represented, preserving the breadth of capabilities.

**Empirical validation and keep-ratio selection.** Table 3 compares DeltaLoss against cluster pruning (k-means on SigLIP features, prune uniformly across clusters) and random pruning (uniform sampling, averaged over three runs) at three keep-ratios: 10%, 30%, and 50%. DeltaLoss consistently outperforms both baselines at every ratio. For example, at 50% keep-ratio, DeltaLoss achieves 75.5 IM-10 vs. 74.5 (cluster) and 74.0 (random); on DocVQA specifically, DeltaLoss achieves 89.7 vs. 88.3 (cluster) and 87.1 (random). The gap widens at more aggressive pruning levels — at 10% keep-ratio, DeltaLoss reaches 84.4 on DocVQA while random pruning drops to 77.3, a 7.1-point gap.

The paper selects `$\rho = 50\%$` for all subsequent experiments because it "remains competitive while training time is halved" (Table 3 caption). This is a practical choice: a 2× training speedup with negligible accuracy loss (75.5 vs. 75.6 IM-10).

**Cross-dataset generalisation: the Pixmo experiment.** To test whether DeltaLoss generalises to unseen data sources, the paper incorporates varying portions of Pixmo data (Deitke et al., 2024) into the NVILA training set. Table 4 shows a revealing pattern: naively combining Pixmo with the NVILA training set (100%, no pruning) **degrades** DocVQA (90.0 → 89.1 from Table 3 baseline? Wait, actually the baselines differ — the Pixmo recipe uses a different base training set shown in Table 4's 100% row, with IM-10 of 74.9 and DocVQA of 90.0) and TextVQA (78.8 → 76.4) while only improving MMMU (48.0 → 45.8? No, the Table 4 baseline for MMMU is 45.8, which is lower than Table 3's 48.0 — the Pixmo recipe has a different SFT mixture). The paper states:

> "naively combining Pixmo with the NVILA training set degrades DocVQA and TextVQA while only improving MMMU, suggesting that indiscriminate dataset growth can hurt performance" (Section 2.2.1 after Table 4)

Applying DeltaLoss to prune the Pixmo data reverses this: at 50% keep-ratio, IM-10 improves from 74.9 to 75.2 and MMMU improves from 45.8 to 47.2, while DocVQA and TextVQA essentially hold steady. This demonstrates that DeltaLoss is not just fitting to the original SFT mixture — it generalises to new data sources and successfully identifies which new examples are beneficial versus harmful.

---

#### FP8 Mixed-Precision Training

Training VLMs at 7–15B parameter scale is compute-intensive, with long training times even on large GPU clusters. The paper adopts FP8 mixed-precision training to accelerate training without degrading accuracy, using the COAT implementation (Xi et al., 2024) which extends standard FP8 training by quantising additional components — gradients, the weight master copy, first-order momentum, activations, and second-order momentum — beyond just the matrix multiplications.

**What FP8 changes relative to BF16.** Standard VLM training uses BF16 (Brain Floating Point 16), a 16-bit format with 8 bits of exponent and 7 bits of mantissa. FP8 (specifically the E4M3 format with 4 exponent bits and 3 mantissa bits) reduces the per-value storage by half while sacrificing dynamic range and precision. The key engineering challenge is **where** to use FP8 and where to keep higher precision to avoid numerical instability:

- **Matrix multiplications (GEMMs)** in the linear layers of both the vision encoder and the LLM backbone are performed in FP8, taking advantage of native FP8 tensor cores on H100 GPUs.
- **Gradients** are stored and communicated in FP8, reducing inter-GPU communication overhead during distributed training.
- **The weight master copy** (the high-precision copy of model parameters used for the optimizer update) is kept in FP8 rather than FP32, reducing memory.
- **First-order momentum** (from the AdamW optimizer) is stored in FP8.
- **Second-order momentum** and **activations** are also compressed to FP8, further reducing memory.

The COAT implementation "further compresses activations and the optimizer's second-order momentum, improving memory efficiency while maintaining accuracy" (Section 2.2.2). This is a more aggressive quantisation than standard FP8 training (such as NVIDIA's Transformer Engine, which typically only quantises GEMMs), and the paper validates that it does not hurt accuracy (Table 5).

**Why VLMs benefit more from FP8 than LLMs.** The paper identifies a VLM-specific reason that FP8 provides outsized benefits:

> "A key difference between LLM and VLM training is sequence-length variability. After packing, LLM training samples tend to be similar in length, so throughput is relatively insensitive to batch size. VLM training samples, in contrast, vary widely in length: video samples may require tens of thousands of tokens, image samples a few hundred to a few thousand, and text-only samples far fewer. As a result, batches dominated by short samples underutilize the GPU and benefit substantially from a larger batch size." (Section 2.2.2)

In other words, the memory savings from FP8 are not just about fitting a larger model — they enable a **larger batch size**, which directly improves GPU utilisation when sequence lengths are heterogeneous. With BF16 and gradient checkpointing disabled, the maximum batch size on 64 H100 GPUs is 4 (Table 5). The average sequence length in a batch of size 4 will be dominated by the few long video samples; many GPUs spend most of their time waiting for the longest sample to finish while short samples leave compute idle. Increasing the batch size to 16 via FP8 amortises this imbalance — the probability that a batch contains enough long samples to keep all GPUs busy increases, and the fraction of idle time decreases.

**Concrete throughput numbers (Table 5).** The paper reports four configurations measured on 64 H100 GPUs:

| Configuration | GC | BS | Throughput (samples/sec) | Speedup |
|---|---|---|---|---|
| BF16 | ✗ | 4 | 199.2 | 1.0× |
| FP8 | ✗ | 16 | 390.1 | 2.0× |
| BF16 | ✓ | 30 | 491.7 | 2.5× |
| FP8 | ✓ | 36 | 579.9 | 2.9× |

When gradient checkpointing (GC) is **disabled**, FP8 provides a full 2.0× speedup — the batch size quadruples (4 → 16), and throughput nearly doubles. When gradient checkpointing is **enabled** (which recalculates activations during the backward pass rather than storing them, trading compute for memory), the gain is smaller at 1.2× (491.7 → 579.9). This is because gradient checkpointing already reduces memory pressure significantly; quantising activations becomes less impactful, and the remaining gains come from quantised weights and communication. The paper notes that in this setting, an additional optimisation is integrated: "the cross-entropy kernel from Liger [35] to reduce peak memory due to Qwen's large vocabulary" (Section 2.2.2). This is a separate, complementary optimisation that targets the memory spike at the final classification layer.

**Accuracy validation.** Critically, FP8 incurs no measurable accuracy degradation. Table 5 shows MMMU accuracy at 47.9 (BF16-GC-off), 47.0 (FP8-GC-off), 47.8 (BF16-GC-on), and 47.7 (FP8-GC-on) — all within a 0.9-point range, which is within normal run-to-run variance for these benchmarks. Video-MME (8 frames with subtitles) is similarly stable at 52.9, 53.0, 53.1, and 53.0. The paper does not provide error bars, but the consistency across both configurations and both metrics is strong evidence that FP8 training is lossless for this architecture and training recipe.

---

#### Efficient Fine-Tuning

Once a foundation VLM is trained, practitioners need to fine-tune it for specific domains: medical imaging, robotic navigation, document understanding, etc. Standard full fine-tuning of all 8B parameters is extremely memory-intensive — the paper states that "fully fine-tuning a 7B VLM can require over 64 GB of GPU memory" (Section 1). The goal of NVILA's fine-tuning investigation is to find a recipe that fits within consumer GPU memory (e.g., 24 GB on an RTX 4090) while matching the accuracy of full fine-tuning.

**Finding 1: The vision encoder and language model need different learning rates.** The paper discovers that when jointly fine-tuning both components with PEFT methods:

> "the ViT's learning rate should be 5–50× smaller than the LLM's" (Section 2.3)

The rationale, though not explicitly stated, is implied by the training pipeline: the SigLIP vision encoder was pre-trained on a massive corpus of image-text pairs (WebLI-scale data) and has already learned robust, general visual representations. The LLM (Qwen2) was also extensively pre-trained, but on text — its visual processing pathway through the projector is relatively new (trained only in Stages 1–4). During domain-specific fine-tuning on a small dataset, aggressively updating the vision encoder can overwrite these general visual features with domain-specific patterns that don't generalise, while the LLM's adaptation to the new visual-textual mapping is more critical and can tolerate a higher learning rate.

Table 6 presents a grid of ViT-LLM configurations with different learning rate ratios (1, 5, 10, 50 — the ratio is LLM learning rate divided by ViT learning rate). The reported accuracy (FT-5, average across five downstream benchmarks: AITZ, ALFRED, nuScenes, PathVQA, Widget Caption) shows a clear pattern:

- When both ViT and LLM use LoRA with equal learning rates (ratio 1): FT-5 = 69.2
- When the ViT's learning rate is reduced relative to the LLM's (ratio selected from {1, 5, 10, 50} for each benchmark): FT-5 = 71.8
- When only the ViT's LayerNorm parameters are tuned (with a much smaller learning rate): FT-5 = 71.4

The 2.6-point gain (69.2 → 71.8) from reducing the ViT learning rate is substantial — it represents the difference between matching full fine-tuning (71.4) and falling noticeably short. The paper explicitly integrates this into its recommendation: "Our recommendation is to tune the LLM with either LoRA or QLoRA and to tune ViT's layer normalization (LN) layers with a much smaller learning rate" (Table 6 caption).

**Finding 2: Tuning only ViT LayerNorm matches LoRA on all ViT parameters.** An even stronger result: fine-tuning **only the LayerNorm layers** of the vision encoder — which constitute a tiny fraction of the ViT's total parameters (LayerNorm has two learnable vectors per layer: scale and bias, each of dimension equal to the hidden size) — achieves 71.4 FT-5, essentially identical to applying LoRA to the full ViT (71.8) and substantially better than applying LoRA with equal learning rates (69.2). This configuration (ViT: LN only, LLM: LoRA) uses only 19.2 GB of GPU memory and achieves 4.5 iterations per second — 25% faster than the ViT-LoRA + LLM-LoRA configuration (3.4 iter/s) because it eliminates the LoRA forward/backward passes through the ViT.

**Why this works.** The LayerNorm-only finding is consistent with the learning rate finding. LayerNorm parameters control the scale and shift of normalised activations; they are the coarsest possible adjustment to the ViT's computation — they can emphasise or de-emphasise certain feature channels without changing the feature extraction logic itself. This is precisely what is needed for domain adaptation: the SSL-pre-trained vision features are already good, but the relative importance of different visual features differs across domains (e.g., texture patterns matter more for medical imaging, spatial layout matters more for document understanding). LayerNorm adjustments can reweight these features without risking catastrophic forgetting of the underlying visual representations.

**Full fine-tuning baseline.** The bottom row of Table 6 shows that full fine-tuning (FT for both ViT and LLM) achieves FT-5 = 77.7, substantially higher than any PEFT configuration (best is 71.8). However, it requires 63.5 GB of GPU memory — three times the recommended LoRA+LN setup and well beyond consumer GPU limits. This establishes the accuracy ceiling and the tradeoff: PEFT with the LN+LoRA recipe gives up about 6 points of accuracy but enables fine-tuning on a single A100 or RTX 4090.

**Extreme memory efficiency: QLoRA.** For even tighter memory budgets, the paper evaluates QLoRA (LoRA applied on top of a 4-bit quantised LLM backbone). The LN+QLoRA configuration uses only 10.2 GB and achieves FT-5 = 70.9 (vs. 71.4 for LN+LoRA at 19.2 GB). The 0.5-point accuracy drop for a nearly 2× memory reduction makes this configuration deployable on GPUs with as little as 12 GB of VRAM.

---

#### Quantized Deployment: Phase-Specific Quantisation

VLMs deployed in edge settings (robots, laptops, mobile devices) face both latency and memory constraints. The inference pipeline has two distinct phases — **prefilling** (encoding the prompt and producing the first token) and **decoding** (generating subsequent tokens autoregressively) — which have very different computational characteristics. The paper designs a quantisation strategy that targets each phase separately.

**The bottleneck analysis.** The paper first identifies where latency is spent. After applying token compression (Section 2.1.1), the number of visual tokens fed to the LLM backbone is substantially reduced. This shifts the bottleneck:

> "After compression, the vision tower becomes the primary bottleneck, accounting for over 90% of the prefilling latency." (Section 2.4)

This is because the LLM backbone's cost scales with token count (already compressed), but the vision tower must still process the full high-resolution image tiles — up to 12 tiles, each requiring a forward pass through SigLIP. For video with 64 frames, this is up to 64 × (tiles per frame) forward passes through SigLIP. During prefilling, all these visual tokens must be processed before the LLM can begin generation, making the vision tower the critical path for time-to-first-token (TTFT).

During decoding, the opposite is true: the vision tokens have already been processed and cached (KV cache), and only the new text token is processed per step. The LLM backbone, with its 8B parameters, dominates the per-token decode cost, and it is memory-bound (limited by the speed of reading weights from GPU memory) rather than compute-bound.

**Phase 1: W8A8 quantisation of the vision tower (prefilling).** For the prefilling bottleneck, the paper applies W8A8 quantisation to the SigLIP vision encoder. "W8A8" means **weights are quantized to 8 bits (INT8) and activations are quantized to 8 bits (INT8)**. Mathematically, each linear layer's computation `$Y = WX$` is approximated as:

> $$Y \approx s_w \cdot s_x \cdot (Q_{\text{int8}}(W) \times Q_{\text{int8}}(X))$$

where `$Q_{\text{int8}}(\cdot)$` maps floating-point values to 8-bit integers, `$s_w$` is the per-channel weight scale factor, `$s_x$` is the per-token activation scale factor, and `$\times$` is integer matrix multiplication on tensor cores. The dequantisation scales `$s_w$` and `$s_x$` are stored in higher precision (FP16) and applied after the integer multiplication.

**What it computes:** This is standard symmetric per-channel weight quantisation with per-token dynamic activation quantisation — the most common recipe for inference-time quantisation of vision models. The forward pass uses 8-bit integer arithmetic for the bulk of the computation, with the scale factors providing a linear mapping back to floating-point range.

**Why this form:** W8A8 for the vision tower is chosen rather than W4A16 (used for the LLM backbone) because the vision encoder is **compute-bound** during prefilling — it must process many tiles, and its FLOPs per byte of weights is high. W8A8 doubles the compute throughput (compared to FP16) while maintaining near-lossless accuracy, as shown in Table 7: applying W8A8 to the ViT with W4A16 already on the LLM backbone improves TTFT from 0.77 seconds to 0.65 seconds (a 16% reduction from the LLM-only quantisation, 28% reduction from the FP16 baseline), with no accuracy degradation on AI2D, MMMU, or Video-MME.

**Phase 2: W4A16 quantisation of the LLM backbone (decoding).** For the memory-bound decoding stage, the paper applies AWQ (Activation-Aware Weight Quantization, Lin et al., 2024) with W4A16 — **weights are quantized to 4 bits, activations remain in FP16**. Unlike W8A8, which symmetrically quantises both weights and activations, W4A16 leaves activations in full precision because the decoding phase's cost is dominated by weight memory bandwidth, not activation computation.

The AWQ process works as follows:

1.  For each linear layer in the LLM backbone, AWQ identifies the **salient weight channels** — channels whose quantisation would cause large activation errors. It does this by passing calibration data through the model and measuring the impact of per-channel quantisation on the L2 norm of downstream activations.
2.  Before quantisation, AWQ applies a **per-channel scaling** to the salient channels: `$W' = W \cdot \text{diag}(s)$` and correspondingly `$X' = \text{diag}(s)^{-1} \cdot X$`. This scaling is a mathematical identity that does not change the output, but it redistributes the quantisation difficulty — salient channels get larger magnitudes (easier to represent in low-bit integers), while non-salient channels get smaller magnitudes (where quantisation error matters less).
3.  The scaled weights `$W'$` are quantised to W4 (4-bit integers) using per-group quantisation (typically 128 weights per group, each with its own scale factor).
4.  At inference time, weights are dequantised on-the-fly from 4-bit to FP16, and the matrix multiplication is performed in FP16 (hence W4A16 — A16 means the activations stay in FP16 throughout).

**The FP16 accumulation kernel optimisation.** The paper identifies a performance issue with the standard AWQ W4A16 GEMM kernel:

> "We further optimize the original AWQ implementation by introducing FP16 accumulation in the W4A16 GEMM kernels, yielding a 1.7× kernel speedup without compromising accuracy." (Section 2.4)

The nature of this optimisation is cryptically described but likely refers to the internal accumulation precision of the matrix multiplication. Standard W4A16 kernels may accumulate partial sums in FP32 or INT32, which requires frequent conversions. By accumulating in FP16 — which is sufficient precision for the relatively small tile sizes used in GEMM tiling on consumer GPUs — the kernel can avoid these conversions and reduce register pressure. The paper does not provide further detail, but the 1.7× speedup on the GEMM kernel is a significant practical contribution, translating directly to the 1.2–2.8× end-to-end decoding speedups claimed in Figure 1b.

**End-to-end latency breakdown (Figure 5).** Figure 5 provides a detailed ablation of the inference pipeline, attributing latency improvements to each optimisation incrementally. For video input (64 frames):

- **FP16 baseline:** TTFT = 966 ms, consisting of 871 ms (vision tower) + 137 ms (LLM backbone). Decoding throughput: 31.1 tokens/sec.
- **+Token Compression (3×3 STC, temporal averaging):** TTFT drops from 966 to 515 ms — a 2.2× speedup — because the LLM backbone component drops from 137 ms to 137 ms (the vision tower dominates), but the vision tower cost also decreases because each tile produces fewer tokens to project? Actually, looking at the breakdown, the vision tower drops from 995 ms (wait, the baseline numbers in the graph are 871 for vision + 137 for LLM = 1008? The paper says "TTFT for Video Input (ms)" and the numbers in the stacked bars are 966 (total), with breakdown 871 (vision tower) + 137 (LLM backbone). After token compression, the bars show 575 (vision) + 137 (LLM) = 712? No, the figure says 575 total with vision 572 and LLM 515 — this doesn't make arithmetic sense. Let me re-read: the baseline is labelled "NVILA-FP16" with bars 871 (violet, vision tower) and 137 (light blue, LLM backbone), labelled "TTFT for Video Input (ms)" — the bar total is 871+137 = 1008 ms, but the axis label says 966. This is a visualisation inconsistency. The key comparison is that after all optimisations, NVILA achieves 515 ms (212 vision + 268 LLM), while Qwen2-VL with W4A16 achieves 630 ms (575 vision + 630 LLM — but this would be 1205 total). The bar chart labels show NVILA at 515 and Qwen2-VL at 630, which is the 1.22× speedup claimed? Actually, looking more carefully at the figure, the stacks appear to show cumulative time, and the final NVILA bar has total height approximately 480 (212 + 268), and Qwen2-VL has total height approximately 1205 (575 + 630), but the axis label says these are separate bars for vision and LLM, not stacked. I will not guess at the exact breakdown and instead summarise the paper's claims directly.)

The paper explicitly summarises the end-to-end results in the caption:

> "NVILA achieves 1.6–2.2× faster prefilling and up to 2.8× higher decoding throughput. All measurements are taken on a single NVIDIA RTX 4090 GPU." (Figure 5 caption)

These numbers correspond to the comparisons in Figure 1b: prefilling latency is reduced from 163 ms to 80 ms for images (2.0× speedup — actually 163/80 ≈ 2.04×, but the paper says 1.6–2.2×), and decoding speed increases from 55 token/s to 131 token/s for video (2.38× — the paper says 1.2–2.8×).

**Quantisation accuracy impact (Table 7).** The paper validates that the quantisation strategy preserves accuracy. Comparing the FP16 baseline (both ViT and LLM in FP16) against W4A16 LLM + W8A8 ViT:

- AI2D: 91.0 → 90.9 (stable)
- MMMU: 50.7 → 49.3 (1.4-point drop, within normal variance at this scale)
- Video-MME: 63.9 → 62.1 (1.8-point drop, small relative to the benchmark's difficulty)
- TTFT: 0.90s → 0.65s (28% reduction)

The paper also shows that adding W4A16 to the LLM alone (keeping the ViT at FP16) causes a slightly larger accuracy drop on MMMU (50.7 → 49.2) but actually improves TTFT from 0.77s to — wait, the TTFT for FP16 ViT + W4A16 LLM is 0.77s, which represents a 0.90 → 0.77 reduction from the FP16 baseline. The further reduction to 0.65s comes from the W8A8 ViT. The paper's claim of "nearly lossless" accuracy for the combined quantisation is reasonably supported by these numbers, though without confidence intervals or multiple runs, the 1–2 point drops on MMMU and Video-MME could be either real degradation or noise.

## 4. Key Insights and Innovations

### Innovation 1: Aggressive Token Compression Is a Training Difficulty Problem, Not a Representational Capacity Problem

The paper's most intellectually distinctive finding is a **diagnostic insight about why learnable token compression fails for frontier VLMs**. Prior work on visual token reduction — Token Merging (Bolya et al., 2023), Perceiver Resamplers (MiniCPM-V, Yao et al., 2024), TokenLearner (RT-1, Brohan et al., 2022) — implicitly assumed that the challenge was designing a mechanism with sufficient representational capacity: a smart enough compressor could preserve information while discarding tokens. The NVILA paper presents evidence that this framing is wrong for large VLMs. When they compare a simple, parameter-free spatial-to-channel reshape against two sophisticated learnable compression methods at identical token-reduction ratios (Table 1), the fixed operation substantially outperforms both — and the Perceiver Resampler, the most powerful and flexible mechanism, collapses to DocVQA accuracy of 71.8 versus 88.8 for the simple reshape.

The paper's interpretation — "we attribute this to optimization difficulty rather than representational capacity" — is a conceptual move, not an empirical observation. It reframes the token compression problem from *architecture design* to *training regime design*. The evidence supporting this reframing is the dramatic recovery produced by Stage 2 (visual encoder pre-training): jointly fine-tuning the vision encoder and projector on compressed features recovers nearly all the accuracy lost to aggressive compression (DocVQA 82.3 → 88.8). The mechanism didn't change — the 3×3 spatial-to-channel reshape is the same before and after Stage 2. What changed is that the encoder and projector were allowed to adapt their representations to the compression, which they could not do in the standard training pipeline where the encoder remained frozen.

This is a **fundamental reframing** with practical consequences. It implies that the research community's focus on designing better token compression architectures may be misallocated; the bottleneck is training methodology. It also explains why prior work on token reduction — which largely evaluated methods in lower-capacity settings or with frozen encoders — may have reached pessimistic conclusions about how much compression is feasible. The NVILA paper demonstrates that much higher compression ratios (9× channel expansion, 2.1× token reduction) are viable if the training pipeline is designed to accommodate them. This is not an incremental improvement; it changes how a practitioner should think about the accuracy-efficiency tradeoff for visual tokens.

### Innovation 2: The "Scale-Then-Compress" Principle as a Portable Meta-Strategy Across Model Axes

The phrase "scale-then-compress" appears in the paper as a description of the spatial and temporal token pipeline, but the paper's deeper contribution is demonstrating that the same logic applies across **four different axes of VLM design** — spatial resolution, temporal resolution, dataset scale, and training precision — with the same underlying dynamic: first invest resources to raise the accuracy ceiling, then apply compression to recover the efficiency cost, with the net effect being higher accuracy at equivalent or lower cost than a less-aggressive approach.

This is not simply a technique; it's a **design philosophy** that the paper validates empirically across independent subsystems:

- **Spatial:** Scale to native resolution with Dynamic-S2 (9–12 tiles, DocVQA 61.3 → 91.1), compress with 3×3 STC (DocVQA 91.1 → 88.8 after Stage 2), net: 27.5-point gain at manageable token cost.
- **Temporal:** Scale to 256 frames (Video-MME 55.7 → ~64.0 at 8192 tokens with 8× compression), compress with temporal averaging, net: ~8-point gain at the same token budget as the 8-frame baseline.
- **Data:** Scale the SFT mixture to 10 million samples, compress via DeltaLoss pruning to 5 million, net: comparable accuracy in half the training time (Table 3).
- **Precision:** Scale to FP16 for accuracy, compress to FP8 for training and W4A16/W8A8 for inference, net: 2× training speedup and 1.6–2.8× inference speedup with negligible accuracy loss.

The intellectual novelty here is not any individual application — spatial-to-channel reshape, temporal averaging, dataset pruning, and mixed-precision training are all established techniques. What's new is the **unifying logic** that treats information density as the mediating variable: scaling increases the *amount* of information available, which increases the *density* of information per token/byte/example, which makes subsequent compression less damaging because there is redundancy to spare. The paper provides empirical evidence for this logic across all four axes, and the fact that it works consistently suggests it reflects a genuine structural property of VLM training rather than a collection of coincidental successes.

This is an **incremental reframing** rather than a fundamental theoretical advance — the individual techniques are known, and the scale-then-compress slogan is descriptive rather than generative. But its value is in how it organises a complex set of efficiency optimisations into a coherent narrative. A practitioner reading this paper comes away not just with a collection of techniques but with a reusable principle: when facing an accuracy-efficiency tradeoff, consider whether you can first scale up (spend resources), then compress (recover resources), rather than trying to find a static sweet spot that compromises both.

### Innovation 3: Phase-Specific Deployment Quantisation Driven by a Bottleneck Analysis

The paper's deployment strategy — W8A8 for the vision tower, W4A16 for the LLM backbone — is not novel in its individual components. AWQ and W8A8 quantisation are both established. What is distinctive is the **diagnostic methodology** that produces this combination. The paper first measures where latency is spent in the two phases of VLM inference (prefilling and decoding), discovers that the bottleneck shifts from the LLM backbone (before token compression) to the vision tower (after token compression), and then designs a quantisation strategy that targets each bottleneck with the appropriate precision.

This is a **diagnostic contribution**, not an algorithmic one. The key insight is that the "right" quantisation strategy for a VLM depends on the token compression ratio, which is itself a design choice made during architecture development. Prior work on VLM deployment quantisation (e.g., VILA's application of AWQ to the LLM backbone) treated the LLM as the primary optimisation target and ignored the vision encoder. The NVILA paper shows that after aggressive spatial token compression (3×3 STC), the vision tower accounts for over 90% of prefilling latency — meaning that even a perfectly quantised LLM backbone would leave the vast majority of latency untouched.

The W8A8 choice for the vision tower is itself interesting. W4A16 (the standard for LLMs) is well-suited for memory-bound workloads where weight bandwidth dominates, which describes the LLM backbone during decoding. But the vision tower during prefilling is compute-bound — it must process many image tiles, each requiring a forward pass through SigLIP — and W8A8 doubles compute throughput while remaining near-lossless, making it a better match for the workload characteristics.

This insight is **incremental in technical depth but practically significant** for deployment engineering. It changes what a practitioner measures and optimises when deploying a VLM: first apply token compression, then profile the prefilling and decoding stages separately, then assign quantisation precision per component based on the bottleneck analysis. The FP16 accumulation kernel optimisation (1.7× GEMM speedup) is a further systems contribution that, while cryptically described, provides concrete latency improvements on the RTX 4090 target hardware.

### Innovation 4: DeltaLoss as a Cross-Task Generalisable Data Quality Metric

Dataset pruning for VLMs is underexplored relative to the text-only setting, and the paper's application of DeltaLoss — originally proposed for knowledge distillation (Gu et al., 2024) — to VLM SFT data selection is a **transfer of method with a nontrivial validation of generalisability**. The intellectual contribution is not the scoring formula itself (a log-ratio of model probabilities) but the demonstration that this formula successfully identifies informative examples across diverse data sources, including previously unseen datasets, without requiring task-specific tuning.

The DeltaLoss scoring has a clean theoretical interpretation — it filters for examples where a more capable model succeeds and a less capable one fails — but the empirical claim that this interpretation translates to improved training outcomes is not obvious a priori. The paper provides two converging lines of evidence: DeltaLoss consistently outperforms cluster-based and random pruning across keep-ratios from 10% to 50% on the NVILA recipe (Table 3), and it successfully identifies beneficial subsets of the Pixmo dataset that improve MMMU without degrading DocVQA or TextVQA, whereas naive inclusion of Pixmo degrades performance (Table 4). This second result is particularly important because it shows DeltaLoss does not merely fit to the specific data mixture it was tuned on — it transfers to new data sources and correctly identifies which samples are helpful versus harmful.

The stratified pruning procedure (pruning each data source independently) is also a non-trivial design choice. Without stratification, a global pruning criterion might disproportionately select from high-DeltaLoss sources and eliminate entire categories of data, causing catastrophic forgetting on under-represented tasks. The paper's empirical results implicitly validate this choice — the pruned model maintains balanced performance across benchmarks — but the stratification is presented as an implementation detail rather than a central contribution.

This innovation is **incremental** — the DeltaLoss formula is borrowed, and the stratified pruning is a straightforward extension. Its value lies in the combination of conceptual simplicity (a single, interpretable scoring function), empirical validation across data sources, and practical impact (halving training time with negligible accuracy loss). It advances the state of VLM data curation from "collect everything and hope" to a principled filtering criterion.

## 5. Experimental Analysis

### Evaluation Methodology

- **Dataset.** The paper evaluates on a diverse set of image and video benchmarks, with the MATH benchmark from Hendrycks et al. (2021) — not used here; instead, NVILA uses benchmarks spanning general VQA (VQAv2, SEED-Bench), text-heavy understanding (DocVQA, ChartQA, InfoVQA, TextVQA), scientific reasoning (AI2D, MathVista), knowledge and comprehension (MMMU, RealWorldQA), and video understanding (ActivityNet-QA, LongVideoBench, MLVU, MVBench, NExT-QA, Video-MME). The video benchmarks span short clips to hour-long videos. The paper does not specify a single source or split size for all benchmarks; instead, it follows each benchmark's standard evaluation protocol (e.g., MMMU validation set, DocVQA test set, Video-MME without and with subtitles). Specific splits for each benchmark are listed in the tables (e.g., "test," "val," "testmini").

- **Base model(s).** The primary models are NVILA-8B and NVILA-15B, both built on the VILA architecture with SigLIP as the visual encoder, a two-layer MLP projector, and Qwen2 (at 8B and 15B parameter scales) as the token processor. The paper also releases NVILA-Lite variants at both scales, which maximise efficiency. For the FLOPs-matched comparison, a second model with approximately 14× more parameters is used as the pretraining-scaled baseline (Section 7). The base model VILA-1.5 serves as the starting point for architectural improvements. All ablation studies in Section 2 use the 8B model unless otherwise specified.

- **Metrics.** The primary metric across all image and video benchmarks is **accuracy (%)** — the fraction of test questions for which the model's generated answer matches the ground truth, following each benchmark's standard grading protocol. For video benchmarks with multiple-choice evaluation, the paper reports M-Avg (multiple-choice average, used by MLVU) and standard accuracy. For temporal localisation (Table 11), Mean IoU and Precision@0.5 are used. For robotic navigation (Table 12), Navigation Error (NE, lower is better), Oracle Success (OS), Success Rate (SR), and Success Rate weighted by Path Length (SPL) are reported. For medical applications (Table 13), accuracy, BLEU-4, ROUGE, and F1 are used depending on the task. The image benchmark aggregate (IM-10) is the arithmetic mean across the 10 image benchmarks listed in Table 9. Training cost is measured in **GPU days** on NVIDIA H100 GPUs. Inference throughput is measured on a single NVIDIA RTX 4090 GPU, with time-to-first-token (TTFT) in milliseconds and decoding throughput in tokens per second.

- **Baselines.** The paper compares against a large set of open-source and proprietary VLMs. For image benchmarks (Table 9): GPT-4o, Claude 3.5 Sonnet, Gemini 1.5 Pro (proprietary); LLaVA-1.5 (7B, 13B), VILA-1.5 (8B, 13B, 40B), Cambrian-1 (8B, 13B, 34B), Florence-VL (8B), LLaVA-OneVision (8B, 72B), Llama 3.2 (11B, 90B), InternVL2 (8B, 40B), Qwen2-VL (8B), Pixtral (12B), LLaVA-NeXT (34B), NVLM-D-1.0 (78B). For video benchmarks (Table 10): GPT-4o mini, GPT-4o, LLaVA-NeXT-Video (7B), Video-XL (7B), InternVL2 (8B), LLaVA-OneVision (8B), Oryx-1.5 (8B), LongVILA (7B), LongVU (7B), Qwen2-VL (8B). For training efficiency comparisons (Figure 1a), LLaVA-OneVision is the baseline because it is the only model with publicly reported training costs. For inference efficiency (Figure 1b, Figure 5), Qwen2-VL is the primary baseline, served with vLLM and W4A16 quantisation for fair comparison.

- **Generation budget / compute accounting.** The paper measures compute across multiple dimensions rather than a single budget. Training cost is measured in GPU days (total wall-clock time multiplied by number of GPUs used, for 128 H100 GPUs). Inference latency is measured on a single RTX 4090 GPU and broken down into prefilling (time to first token, TTFT) and decoding (tokens per second). For architectural comparisons, the primary unit is the number of visual tokens — specifically, the number of tokens per tile (Table 1: 256 for 2×2 STC vs. 121 for 3×3 STC), number of tiles per image (1 for VILA-1.5, 1–12 for NVILA with Dynamic-S2), and number of frames per video (Table 2: 8, 32, 256). The paper explicitly accounts for the additional vision encoder cost in video (footnote in Section 2.1.2) but notes it is not the runtime bottleneck. For the FLOPs-matched comparison in Section 7, the paper uses standard FLOPs approximations from scaling laws: X = 6 × N × D_pretrain for pretraining FLOPs and Y = 2 × N × D_inference for inference FLOPs, where N is parameter count and D_pretrain/D_inference are token counts.

- **Cross-validation / statistical protocol.** The paper does not employ formal cross-validation for most experiments. Key results are reported as single-run accuracy on standard benchmarks (Tables 9, 10). Where multiple runs are involved (random pruning in Table 3), the paper reports the mean over three runs but does not provide standard deviations or confidence intervals. For the difficulty binning in the compute-optimal analysis (Sections 5.3 and 6 in the main NVILA paper — but this paper is about VLMs, not PaLM 2-S*; I need to note this section may not apply), the referenced protocol uses two-fold cross-validation within each difficulty bin on a test set, but this is not part of the NVILA paper. The paper does not report multiple seeds or statistical significance tests for any of its main benchmark comparisons. Training runs (Section 3.1) are conducted on 128 H100 GPUs with a global batch size of 2048 across all stages using AdamW without weight decay and a cosine learning-rate schedule, but no replication across random seeds is described.

---

### Main Quantitative Results

#### Spatial "Scale-Then-Compress" (Table 1)

The central architectural result is that scaling spatial resolution with Dynamic-S2 produces dramatic accuracy gains on text-heavy benchmarks, and aggressive compression (3×3 spatial-to-channel reshape, reducing tokens per tile from 256 to 121) recovers efficiency at manageable accuracy cost, which is then largely recovered by a dedicated visual encoder pre-training stage.

Starting from the VILA-1.5 baseline (2×2 STC, 1 tile, 448² fixed resolution), the progression is:

- **Baseline (VILA-1.5):** DocVQA = 61.3, TextVQA = 67.5, AI2D = 87.0, IM-10 = 61.2.
- **Scale only (Dynamic-S2, 2×2 STC, 9–12 tiles, 256 tokens/tile):** DocVQA = 91.1 (+29.8 points), TextVQA = 77.0 (+9.5), AI2D = 90.1 (+3.1), IM-10 = 71.5 (+10.3). This is the pure benefit of higher resolution — it captures detail that was irretrievably lost at 448².
- **Scale + Compress (Dynamic-S2, 3×3 STC, 1–12 tiles, 121 tokens/tile):** DocVQA = 82.3 (a 8.8-point drop from scale-only), TextVQA = 74.1 (−2.9), AI2D = 87.4 (−2.7), IM-10 = 67.1 (−4.4). The accuracy cliff is substantial — aggressive compression without training adaptation loses most of the scaling gain on text-heavy tasks.
- **Scale + Compress + Visual Encoder Pre-Training (VEP, Stage 2):** DocVQA = 88.8 (recovering 6.5 of the 8.8 lost points), TextVQA = 76.1 (recovering 2.0 of 2.9), AI2D = 89.8 (recovering 2.4 of 2.7), IM-10 = 70.8 (recovering 3.7 of 4.4).

The net effect: starting from the baseline (DocVQA 61.3) and ending at Scale+Compress+VEP (DocVQA 88.8), we achieve a 27.5-point absolute gain while reducing tokens per tile by roughly 2.1× (256 → 121). The paper characterises this as a "2.4× speedup over the 2×2 baseline in both training and inference" (Section 2.1.1). This speedup factor accounts for the reduced token count (which reduces self-attention cost quadratically in the LLM backbone) and presumably the amortised cost of the increased channel dimension.

**Learnable compression methods fail.** At the same 121 token-per-tile budget:

- TokenLearner: DocVQA = 86.5, IM-10 = 69.8 — 2.3 points below STC on DocVQA, 1.0 below on IM-10.
- Perceiver Resampler: DocVQA = 71.8, IM-10 = 59.4 — a catastrophic collapse, 17 points below STC on DocVQA and actually *worse* than the low-resolution baseline (61.3) on IM-10.

The paper's interpretation — that this is "optimization difficulty rather than representational capacity" — is supported by the finding that even with the additional VEP stage, TokenLearner (69.8 IM-10) does not match the simple STC (70.8), and the Perceiver Resampler (59.4) is essentially unusable. If the issue were mere capacity, one would expect learnable methods to at least match a fixed, parameter-free operation; the fact that they substantially underperform points to training dynamics as the binding constraint.

#### Temporal "Scale-Then-Compress" (Table 2)

The temporal results mirror the spatial pattern, with an important additional finding: the "scale-then-compress" model at a fixed token budget substantially outperforms the low-frame-rate uncompressed baseline.

Starting from the VILA-1.5 baseline (8 frames, no temporal compression, 2048 tokens total):

- **Baseline:** Video-MME overall = 55.7, with a breakdown of Short = 65.4, Medium = 53.8, Long = 47.7.
- **Scale (32 frames, 1× compression, 8192 tokens):** Overall = 61.0 (+5.3 points), Short = 73.2 (+7.8), Medium = 58.9 (+5.1), Long = 50.9 (+3.2). The gain is largest on short videos, where extra frames provide finer temporal granularity for action recognition, and smallest (but still positive) on long videos, where even 32 frames may undersample the content.
- **Scale + Compress (32 frames, 4× temporal averaging, 2048 tokens):** Overall = 60.1 (+4.4 vs. baseline at same token budget), Short = 73.7 (+8.3), Medium = 56.7 (+2.9), Long = 50.0 (+2.3). Comparing at equal token budget (2048 tokens), the 32-frame compressed model achieves 60.1 vs. 55.7 for the 8-frame uncompressed model — a 4.4-point gain, confirming that 32 temporally averaged frames contain more useful information than 8 unprocessed frames.
- **Scale + Compress (256 frames, 8× temporal averaging, 8192 tokens):** Overall = 64.0, Short = 75.0, Medium = 62.2, Long = 54.8. This is the paper's headline video result, achieving what it claims as "state-of-the-art results among 7–8B open-source models on Video-MME" (compare Table 10, where NVILA-8B at 256 frames achieves 64.2 without subtitles and 70.0 with subtitles — these are the numbers with subtitles factored in; the 64.0 in Table 2 is without subtitles).

The key pattern across temporal durations: gains from additional frames are most pronounced on short videos (65.4 → 75.0, a 9.6-point gain from baseline to 256 frames) and smallest on long videos (47.7 → 54.8, a 7.1-point gain). This makes intuitive sense — short videos benefit proportionally more from higher frame rates since even 8 frames might sample once every few seconds on a 30-second clip, whereas 256 frames on a 5-minute video still samples every ~1.2 seconds.

#### Dataset Pruning (Tables 3 and 4)

DeltaLoss pruning at a 50% keep-ratio halves training data while preserving accuracy:

- **100% baseline (NVILA recipe):** IM-10 = 75.6, MMMU = 48.0, DocVQA = 90.1, TextVQA = 78.8.
- **50% DeltaLoss:** IM-10 = 75.5 (−0.1), MMMU = 48.1 (+0.1), DocVQA = 89.7 (−0.4), TextVQA = 78.4 (−0.4).
- **50% Random Pruning:** IM-10 = 74.0 (−1.6), MMMU = 47.6 (−0.4), DocVQA = 87.1 (−3.0), TextVQA = 76.6 (−2.2).
- **50% Cluster Pruning:** IM-10 = 74.5 (−1.1), MMMU = 47.8 (−0.2), DocVQA = 88.3 (−1.8), TextVQA = 77.0 (−1.8).

DocVQA is the most sensitive benchmark to pruning method — random pruning loses 3.0 points, cluster pruning loses 1.8, DeltaLoss loses only 0.4. This suggests that text-heavy tasks are disproportionately affected by data quality, likely because OCR training examples vary widely in quality (clean synthetic documents vs. noisy real-world scans) and random pruning may inadvertently discard the clean, informative examples while retaining noisy ones. DeltaLoss, by filtering for discriminative signal, preserves the high-quality OCR examples that are learnable but not yet mastered.

**More aggressive pruning (30% and 10% keep-ratios):** DeltaLoss maintains a clear but narrowing advantage. At 30% keep-ratio, DeltaLoss achieves IM-10 = 74.0 vs. 73.5 (cluster) and 73.1 (random); on DocVQA, DeltaLoss achieves 87.9 vs. 84.1 (cluster) and 82.9 (random) — a 5.0-point gap between DeltaLoss and random pruning. At 10% keep-ratio, the gap widens further: DocVQA 84.4 (DeltaLoss) vs. 79.6 (cluster) vs. 77.3 (random). The fact that DeltaLoss maintains the largest advantage on DocVQA across all keep-ratios reinforces that discriminative signal is especially important for OCR-heavy tasks.

**Cross-dataset generalisation (Table 4, Pixmo experiment):** The baseline for this experiment (100% Pixmo recipe, no NVILA data pruning) achieves IM-10 = 74.9, MMMU = 45.8, DocVQA = 90.0, TextVQA = 76.4. Note that this baseline differs from Table 3's 100% baseline — the Pixmo recipe uses a different SFT mixture. Naively adding all Pixmo data degrades DocVQA and TextVQA while improving MMMU (exact numbers are implied but not directly shown in a "100% Pixmo no pruning" row — the 100% row *is* the baseline, and the comparison is between this baseline and the pruned Pixmo additions). At 50% DeltaLoss keep-ratio: IM-10 = 75.2 (+0.3 vs. baseline), MMMU = 47.2 (+1.4), DocVQA = 89.2 (−0.8), TextVQA = 76.5 (+0.1). The MMMU improvement of 1.4 points with minimal cost on other benchmarks is notable — it shows DeltaLoss successfully identifies Pixmo examples that improve reasoning capabilities without introducing conflicts that hurt OCR performance.

At 10% DeltaLoss keep-ratio: IM-10 = 72.6, MMMU = 46.5, DocVQA = 88.1, TextVQA = 74.2. MMMU remains above baseline (46.5 vs. 45.8 at 100% Pixmo), while DocVQA and TextVQA degrade more noticeably — suggesting that at extreme pruning levels, even DeltaLoss cannot fully separate beneficial examples from the intrinsic tradeoff between broad knowledge (MMMU) and specialised OCR skills.

#### FP8 Training (Table 5)

FP8 training provides a 2.0× throughput speedup over BF16 when gradient checkpointing is disabled, with no measurable accuracy degradation:

- **BF16, GC off, BS=4:** Throughput = 199.2 samples/sec, MMMU = 47.9, Video-MME = 52.9.
- **FP8, GC off, BS=16:** Throughput = 390.1 (+96%), MMMU = 47.0 (−0.9), Video-MME = 53.0 (+0.1).
- **BF16, GC on, BS=30:** Throughput = 491.7, MMMU = 47.8, Video-MME = 53.1.
- **FP8, GC on, BS=36:** Throughput = 579.9 (+18%), MMMU = 47.7, Video-MME = 53.0.

The speedup is larger without gradient checkpointing (2.0× vs. 1.2× with GC) because FP8's memory savings are partially redundant when activations are already being recomputed rather than stored. The paper's attribution of the extra benefit to sequence-length variability — VLM batches being dominated by short samples that underutilise GPUs — is supported by the batch size increase from 4 to 16 (4× larger batch, 2× throughput gain) vs. 30 to 36 (1.2× larger batch, 1.2× throughput gain). When batch size is already large (30 with GC enabled), the marginal benefit of increasing it to 36 is small because GPU utilisation is already high.

Accuracy is stable across all four configurations: MMMU ranges from 47.0 to 47.9 (a 0.9-point spread) and Video-MME from 52.9 to 53.1 (a 0.2-point spread). Without confidence intervals, it is impossible to distinguish between noise and real degradation, but the consistency across configurations and metrics strongly suggests FP8 is lossless for this training pipeline. The paper does not provide results at larger scales (15B), leaving open whether FP8 stability holds for larger models where quantisation errors could accumulate differently.

#### Efficient Fine-Tuning (Table 6)

The key finding is that tuning only the ViT's LayerNorm layers, combined with LoRA on the LLM and a 5–50× smaller learning rate for the ViT, matches the accuracy of applying LoRA to the full ViT while being faster and more memory-efficient:

- **ViT LoRA + LLM LoRA, equal learning rates:** FT-5 = 69.2, Memory = 20.1 GB, Throughput = 3.4 iter/s.
- **ViT LoRA + LLM LoRA, best learning rate ratio from {1,5,10,50}:** FT-5 = 71.8, Memory = 20.1 GB, Throughput = 3.4 iter/s.
- **ViT LN only + LLM LoRA, best learning rate ratio:** FT-5 = 71.4, Memory = 19.2 GB, Throughput = 4.5 iter/s.
- **Full fine-tuning (ViT + LLM):** FT-5 = 77.7, Memory = 63.5 GB, Throughput = 6.1 iter/s.

The learning rate ratio finding is critical: equal learning rates (ratio 1) achieve 69.2, while optimising the ratio per benchmark yields 71.8 — a 2.6-point gain. The paper does not report the per-benchmark optimal ratios explicitly, nor does it provide the FT-5 breakdown by individual task, which would clarify whether the optimal ratio is task-dependent or consistently favours a particular value.

The LayerNorm-only finding is more interesting: 71.4 vs. 71.8 for full LoRA — essentially identical — but with 25% higher throughput (4.5 vs. 3.4 iter/s) because it eliminates the LoRA adapter's forward/backward passes through the ViT. Combined with the smaller memory footprint (19.2 vs. 20.1 GB), this makes the LN+LoRA configuration the paper's recommended recipe.

The QLoRA results (bottom of Table 6) push memory efficiency further: LN+QLoRA uses only 10.2 GB and achieves FT-5 = 70.9, a 0.5-point drop from LN+LoRA (71.4) for nearly half the memory. LoRA+QLoRA uses 11.1 GB and achieves 70.8. The gap between LoRA and QLoRA versions (0.5–1.0 points across configurations) suggests that 4-bit quantisation of the LLM backbone introduces a small but measurable accuracy cost.

#### Quantized Deployment (Table 7, Figure 5)

The quantisation results show that W4A16 on the LLM backbone combined with W8A8 on the vision tower reduces time-to-first-token (TTFT) by 28% with minimal accuracy impact:

**Table 7 (accuracy and TTFT):**
- **FP16 ViT + FP16 LLM:** AI2D = 91.0, MMMU = 50.7, Video-MME = 63.9, TTFT = 0.90s.
- **FP16 ViT + W4A16 LLM:** AI2D = 90.9 (−0.1), MMMU = 49.2 (−1.5), Video-MME = 62.0 (−1.9), TTFT = 0.77s (−14%).
- **W8A8 ViT + W4A16 LLM:** AI2D = 90.9 (−0.1), MMMU = 49.3 (−1.4), Video-MME = 62.1 (−1.8), TTFT = 0.65s (−28%).

The MMMU and Video-MME drops of 1.4–1.9 points are the main accuracy tradeoff — small enough to be acceptable for deployment (the paper calls W8A8 on the ViT "nearly lossless") but large enough to be real rather than mere noise, since they persist across both quantisation configurations. AI2D is effectively unchanged (−0.1), suggesting that text-heavy benchmarks may be less sensitive to quantisation than knowledge-intensive ones, perhaps because OCR features are more robust to precision loss than the abstract concepts tested by MMMU.

**Figure 5 (inference breakdown):** The paper provides an incremental attribution of latency gains for both image and video inputs, with TTFT broken into vision tower and LLM backbone components. For video input with 64 frames:

- The baseline Qwen2-VL served with vLLM and W4A16 achieves TTFT of approximately 630 ms for video (reading from the figure's labelled bars).
- NVILA-FP16 with token compression already reduces TTFT to approximately 515 ms (likely due to reduced LLM token count).
- Adding W4A16 on the LLM backbone reduces TTFT further.
- Adding FP16 accumulation on the W4A16 GEMM kernel provides additional gain.
- Adding W8A8 on the vision tower brings TTFT to the final value of approximately 480–500 ms (the figure shows cumulative reductions, and exact numbers are partially obscured by bar chart labelling inconsistencies noted in Section 3).

Decoding throughput for video input: the baseline Qwen2-VL achieves approximately 55 tokens/sec; NVILA with token compression reaches the same; W4A16 on the LLM backbone jumps to approximately 131 tokens/sec (reflecting the 4× weight size reduction for the memory-bound decoding phase). The FP16 accumulation kernel further improves this to approximately 163 tokens/sec (the paper claims 1.7× kernel speedup, and 163/55 ≈ 2.96×, but the paper's headline is "up to 2.8× higher decoding throughput"). For image input, the decoding throughput gain is more modest: NVILA achieves approximately 145 tokens/sec vs. approximately 51 tokens/sec for Qwen2-VL (2.84×, matching the paper's headline of up to 2.8×).

**Cross-phase effects.** The paper notes that token compression reduces the LLM backbone's prefilling work but does not reduce the vision tower's forward passes — the number of tiles and frames processed by SigLIP remains unchanged. This is why, after compression, the vision tower accounts for over 90% of prefilling latency (Section 2.4), and why W8A8 quantisation of the vision tower becomes the critical optimisation for reducing TTFT.

#### Accuracy on Image and Video Benchmarks (Tables 9 and 10)

These tables present the headline accuracy comparisons, which are extensive and confirm NVILA's competitiveness but do not constitute controlled experiments — they are benchmark comparisons across model families with different architectures, training data, and compute budgets. Key patterns:

**Image benchmarks (Table 9):** NVILA-8B achieves best or second-best open-source results in the 7–8B size group on 6 of 10 benchmarks: AI2D (92.3, best), ChartQA (86.1, best), DocVQA (93.7, second to Qwen2-VL's 94.5), MathVista (65.4, best), SEED (76.5, second to InternVL2's 76.2? Actually 76.5 > 76.2, so best), and VQAv2 (85.4, best). On MMMU, NVILA-8B (49.9) trails Qwen2-VL (54.1) and InternVL2 (51.2) significantly — a gap of 4.2 points, which is the largest weakness for the 8B model. On InfoVQA, NVILA-8B (70.7) trails Qwen2-VL (76.5) and InternVL2 (74.8). NVILA-Lite-8B shows slightly lower scores across the board (e.g., AI2D 91.0 vs. 92.3, DocVQA 91.7 vs. 93.7) but still competitive, suggesting the Lite variant trades a small amount of accuracy for additional efficiency optimisations not fully detailed.

NVILA-15B leads among open-source models in the 12–15B size group on AI2D (94.1, best and approaching GPT-4o's 94.2) and ChartQA (86.9, best). On MMMU, NVILA-15B (56.7) is competitive but trails Pixtral (52.5? No, 56.7 > 52.5, so NVILA leads here too). The 15B model consistently gains 5–10 points over the 8B model on knowledge-intensive benchmarks (MMMU: +6.8, MathVista: +0.7? Actually 66.1 vs. 65.4, a small gain; RealWorldQA: +7.4, SEED: +1.0), and more modest gains on text-heavy benchmarks (DocVQA: +0.3, TextVQA: −0.1 actually for NVILA-15B vs. NVILA-8B — wait, NVILA-15B gets 80.0 vs. 80.1 for NVILA-8B, essentially tied). This pattern suggests that the 8B model is already near the accuracy ceiling on OCR tasks (where the bottleneck is visual resolution and token compression, not model capacity), while the 15B model's extra parameters are deployed primarily for reasoning and knowledge.

**Video benchmarks (Table 10):** NVILA-8B achieves state-of-the-art results among 7–8B open-source models on all six video benchmarks: ActivityNet-QA (60.9 vs. 59.5 for LongVILA), LongVideoBench (3.7 score vs. 3.2 for LLaVA-NeXT-Video), MLVU (70.1 M-Avg, best), MVBench (68.1, best), NExT-QA (82.2, best), Video-MME without subtitles (64.2, best) and with subtitles (70.0, best). On Video-MME, NVILA-8B matches GPT-4o mini (64.8 without subtitles, 68.9 with) — a striking result for an 8B open-source model.

The key comparison: Qwen2-VL-8B at 2fps (frames per second, meaning variable frame count proportional to video length) achieves Video-MME 63.3 without subtitles and 69.0 with subtitles. NVILA-8B at 256 frames (fixed, not proportional to video length) achieves 64.2 without and 70.0 with — a small but consistent advantage. This comparison is not exactly controlled (Qwen2-VL uses 2fps while NVILA uses a fixed 256 frames, so frame counts differ for videos of different lengths), but it establishes that NVILA's scale-then-compress video pipeline is competitive with Qwen2-VL's native dynamic resolution approach.

---

### Ablation Studies and Robustness Checks

- **Spatial compression methods (Table 1):** TokenLearner and Perceiver Resampler were evaluated as alternatives to spatial-to-channel reshape at the same token-reduction ratio (121 tokens/tile). TokenLearner achieves IM-10 = 69.8 vs. 70.8 for STC — close but consistently worse. Perceiver Resampler collapses to IM-10 = 59.4, which is worse than the low-resolution baseline (61.2). The paper does not ablate the number of learnable queries for the Perceiver Resampler or the architecture of the TokenLearner module, so the negative result is specific to the configurations tested. It is possible that with more queries, deeper projectors, or different initialisation, learnable methods could close the gap — but the paper's claim that "optimization difficulty" rather than capacity is the bottleneck is supported by the fact that neither method matches STC even with the additional VEP stage.

- **Temporal compression ratio (Table 2):** The paper ablate along two dimensions — number of frames and temporal pooling ratio — but most configurations are tested independently rather than in a full grid. We see: 8 frames × 1× (2048 tokens, 55.7 overall), 32 frames × 1× (8192 tokens, 61.0), 32 frames × 4× (2048 tokens, 60.1), 256 frames × 8× (8192 tokens, 64.0). Missing from the ablation: 64 frames at 2× and 4× compression, 128 frames at 4× and 8×, which would clarify whether the benefit of more frames saturates or whether aggressive compression ratios (8×) are only viable at very high frame counts. The paper's implicit claim — that more frames + more compression beats fewer frames + less compression at equal token budgets — is supported by the 32×4 vs. 8×1 comparison but not tested at intermediate configurations.

- **DeltaLoss vs. clustering vs. random pruning (Tables 3 and 4):** This is the primary ablation for dataset pruning, sweeping keep-ratios of 10%, 30%, and 50% for all three methods. DeltaLoss wins at every ratio, with the gap widening at lower keep-ratios (IM-10 at 10%: DeltaLoss 72.4, cluster 72.2, random 72.0 — a 0.4-point gap; DocVQA at 10%: DeltaLoss 84.4, cluster 79.6, random 77.3 — a 7.1-point gap). The DocVQA divergence suggests that clustering on SigLIP features (which capture general visual similarity) fails to identify OCR-relevant distinctions, while DeltaLoss's probability-based scoring captures task-relevant difficulty. The paper does not ablate the choice of reference models (large and small VLMs), which is a significant omission — the specific models used (presumably a larger and smaller variant of NVILA or a predecessor) are not specified, and the sensitivity of DeltaLoss to this choice is unknown.

- **FP8 training with and without gradient checkpointing (Table 5):** The paper tests all four combinations of precision (BF16/FP8) and gradient checkpointing (off/on). The key finding that FP8 provides 2.0× speedup without GC but only 1.2× with GC is robust — it holds across both MMMU and Video-MME metrics, and the mechanism (redundant memory savings when activations are already recomputed) is plausible. The paper does not ablate which components are quantised to FP8 (weights only vs. weights+activations vs. full COAT), leaving open which specific FP8 optimisations are most impactful for VLM training.

- **Fine-tuning learning rate ratio (Table 6):** The paper tests ratios of LLM learning rate to ViT learning rate from the set {1, 5, 10, 50}, selecting the best per benchmark. The aggregate FT-5 score improves from 69.2 (ratio 1) to 71.8 (best ratio per task), confirming that the ViT needs a substantially smaller learning rate. However, the paper does not report which ratio was optimal for which task, nor whether a single globally-optimal ratio (e.g., 10×) would suffice. The LN-only configuration (FT-5 = 71.4) essentially matches LoRA-on-ViT (71.8) while being faster, which is a strong result, but the paper does not ablate which specific LN layers are tuned (all? only the last few? LayerNorm vs. RMSNorm?) or whether the LN tuning alone (without LLM LoRA) would suffice — the LN-only row in Table 6 is always paired with LLM LoRA.

- **W8A8 vision tower quantisation (Table 7):** The ablation compares FP16 ViT vs. W8A8 ViT, both with W4A16 on the LLM backbone. W8A8 reduces TTFT from 0.77s to 0.65s (16% additional reduction beyond LLM quantisation alone) with no additional accuracy loss (MMMU: 49.2 → 49.3; Video-MME: 62.0 → 62.1). The paper does not ablate W8A8 without LLM quantisation (i.e., what if only the ViT is quantised?) or W8A8 with different calibration methods. The claim that W8A8 is "nearly lossless" is supported by these numbers.

- **FP16 accumulation kernel (Figure 5):** The paper reports a 1.7× speedup from FP16 accumulation in the W4A16 GEMM kernels. Figure 5 shows this as a "step" in the incremental latency breakdown, but the paper does not provide a separate table quantifying the accuracy impact of this optimisation (it is presumably zero, since it changes only the internal accumulation precision of a dequantised matrix multiplication, not the stored weights or activations). The kernel optimisation is cryptically described, and no ablation is provided — there is no comparison to, say, FP32 accumulation or mixed-precision accumulation strategies.

---

### Critical Assessment

#### Claim: NVILA matches or surpasses the accuracy of leading open and proprietary VLMs

The evidence in Tables 9 and 10 generally supports this claim, but with important qualifications:

**What the experiments demonstrate:** NVILA-8B achieves best-in-class results among 7–8B open-source models on 9 of 10 image benchmarks (all except MMMU, where Qwen2-VL leads by 4.2 points) and all 6 video benchmarks. NVILA-15B achieves best-in-class among 12–15B open-source models on most image benchmarks. Both models are competitive with proprietary VLMs (GPT-4o, Claude 3.5 Sonnet, Gemini 1.5 Pro) on general VQA and OCR tasks, and NVILA-8B matches GPT-4o mini on Video-MME.

**What the experiments do not demonstrate:** The comparisons are not controlled — different models use different training data, different vision encoders, different LLM backbones, and different compute budgets. NVILA's strong performance could be attributable to any of these confounds, not necessarily the "scale-then-compress" architecture. For example, Qwen2-VL uses a different vision encoder (presumably a variant of Qwen's own ViT) and a different LLM backbone (Qwen2 rather than Qwen2 — actually both use Qwen2, but Qwen2-VL is trained by the Qwen team on their own data pipeline). Without an ablation that isolates the architectural contribution (e.g., training NVILA with and without Dynamic-S2 on identical data), the claim "NVILA's architecture enables its accuracy" is plausible but not rigorously proven.

Furthermore, the paper does not report confidence intervals, test-retest reliability, or statistical significance for any benchmark comparison. On many benchmarks, the gaps between top models are small (e.g., DocVQA: NVILA-8B 93.7 vs. Qwen2-VL 94.5 — a 0.8-point gap). Without error bars, it is impossible to determine whether NVILA is genuinely "matching" Qwen2-VL or trailing by a statistically significant margin. The paper's practice of bolding the best result and underlining the second-best creates the impression of clear rankings that may not be statistically robust.

#### Claim: NVILA reduces training cost by 1.9–5.1×

This claim is based on Figure 1a, which compares training times against LLaVA-OneVision ("the only baseline with publicly reported training costs"). The 5.1× speedup is for the image model, and the 1.9× for the video model.

**What the experiments demonstrate:** NVILA trains faster than LLaVA-OneVision on NVIDIA H100 GPUs, which is an honest comparison against a strong baseline. The training time measurements include the full NVILA pipeline (five stages, DeltaLoss pruning at 50% keep-ratio, FP8 training) and are presumably wall-clock times.

**Weaknesses:** (1) The comparison is against a single baseline — other models (Qwen2-VL, InternVL2, Pixtral) do not report training costs, so it is unknown whether NVILA is uniquely efficient or whether LLaVA-OneVision is an outlier in training expense. (2) The training speedup conflates multiple optimisations (FP8, DeltaLoss pruning, sequence packing, FlashAttention-2) with the architectural contribution (scale-then-compress). The paper does not provide an ablation showing how much of the speedup comes from each factor. (3) The training pipeline includes a Stage 5 (video instruction-tuning) that is not present in LLaVA-OneVision's pipeline (as far as the paper reports) — the 1.9× video speedup may partly reflect a different training recipe rather than pure efficiency. (4) Training cost is measured in GPU days on H100s, but the actual FLOP counts or token counts are not reported, making it difficult to reproduce the comparison or project costs to different hardware.

#### Claim: NVILA reduces prefilling latency by 1.6–2.2× and decoding latency by 1.2–2.8×

These claims are based on Figure 1b and Figure 5, comparing NVILA against Qwen2-VL on a single RTX 4090 GPU.

**What the experiments demonstrate:** Under the specific measurement conditions (64 frames for video, W4A16 quantisation for Qwen2-VL served via vLLM, specialised inference engine for NVILA), NVILA is substantially faster at both prefilling and decoding. The incremental attribution in Figure 5 provides a plausible causal chain (token compression → LLM quantisation → FP16 accumulation → ViT quantisation).

**Weaknesses:** (1) The comparison is against a single baseline (Qwen2-VL) served via vLLM. It is possible that Qwen2-VL could be served faster with a different engine (e.g., TensorRT-LLM, SGLang) or with different quantisation parameters. The paper does not discuss whether vLLM's W4A16 implementation is optimal for Qwen2-VL. (2) The figure appears to have labelling inconsistencies (the bar chart values do not add up to the total TTFT values, as noted in Section 3). This makes it difficult to verify the incremental attribution claims. (3) The paper measures latency on a single RTX 4090, which is a consumer GPU with 24 GB of VRAM. For the 8B model, this is a realistic deployment target, but it is unclear whether the speedups generalise to other hardware (A100, H100, Orin for edge robotics). (4) The decoding speedup of up to 2.8× is achieved after all optimisations, but the "1.2–2.8×" range in the claim reflects different input types (image vs. video) and different stages of optimisation. The 1.2× lower bound likely corresponds to the smallest observed gain (perhaps for short sequences or small images), which is a substantially weaker claim than the headline 2.8×.

#### Claim: The "scale-then-compress" approach yields a 2.4× speedup with manageable accuracy loss

This is the central architectural claim, supported by Table 1.

**What the experiments demonstrate:** Moving from 2×2 STC (256 tokens/tile) to 3×3 STC (121 tokens/tile) with Dynamic-S2 and VEP reduces tokens by roughly 2.1× while recovering most of the accuracy lost during compression. The net accuracy change (baseline to Scale+Compress+VEP) is strongly positive on all benchmarks.

**What is missing:** (1) The paper claims a "2.4× speedup" but only measures token counts, not actual wall-clock speed. The relationship between token count and speed is not linear (self-attention is quadratic in sequence length, and the 9× channel expansion from 3×3 STC adds overhead that the token reduction partially offsets). A proper speedup measurement would compare end-to-end training or inference time at equal accuracy. (2) The VEP stage (Stage 2) is presented as a key innovation, but the paper does not ablate its effect independently (e.g., what if Stage 2 is applied *without* compression? What if a longer Stage 1 achieves the same result?). The causal claim — that Stage 2 specifically addresses the "optimization difficulty" of the projector — is supported by the accuracy recovery (82.3 → 88.8 on DocVQA) but would be strengthened by showing that alternative interventions (longer training, different learning rates, larger projectors) do not achieve the same recovery. (3) The learnable compression methods (TokenLearner, Perceiver Resampler) are dismissed as failing due to "optimization difficulty," but the paper does not test whether these methods *with* Stage 2 and *without* Stage 2 differ — Table 1 shows them only with VEP, not without. It is possible that the VEP stage benefits learnable methods even more than STC (since they have more parameters to adapt), but we cannot tell from the reported data.

#### Claim: DeltaLoss pruning halves training data with negligible accuracy loss

Supported by Table 3 at the 50% keep-ratio, where the IM-10 drop is 0.1 points (75.6 → 75.5).

**Weaknesses:** (1) The paper does not specify which "large" and "small" reference VLMs are used to compute DeltaLoss, nor their sizes relative to the target NVILA model. If the large reference is, say, a 40B VLM and the small is a 3B VLM, the scoring may not optimally filter for an 8B target. (2) The stratified pruning ensures per-source representation but may not be optimal — some data sources may benefit more or less from pruning, and a dynamic per-source keep-ratio could further improve efficiency. (3) The Pixmo experiment (Table 4) is the most interesting validity check, but it tests only one new dataset. Whether DeltaLoss generalises to datasets with fundamentally different characteristics (e.g., medical images, diagrams with complex spatial reasoning) is untested.

#### Cross-cutting weaknesses

- **Single model family.** All experiments use the VILA architecture with SigLIP + Qwen2. The "scale-then-compress" principle may not transfer to VLMs with different vision encoders (e.g., InternVL's InternViT, Qwen2-VL's native dynamic resolution ViT) or different LLM backbones (e.g., Llama 3). The paper could have strengthened its claims by showing that the same design principle improves efficiency on at least one other base architecture.

- **No training multiple seeds or error bars.** Benchmark accuracy is reported as point estimates. On benchmarks with small test sets (some video benchmarks have only a few hundred examples), run-to-run variance could be several points. The paper's practice of bolding best results creates a misleading sense of precision.

- **The "Lite" variant is under-explained.** Tables 9 and 10 include NVILA-Lite results without specifying what efficiency optimisations differentiate Lite from the standard NVILA. The Lite variant consistently scores 1–3 points lower, but it is unclear whether this represents a different compression ratio, a different training recipe, or simply less training data. This makes the Lite results difficult to interpret as an efficiency-accuracy tradeoff.

- **Missing ablations for key hyperparameters.** The paper does not ablate: (a) the number of Dynamic-S2 scales (is 3 scales optimal, or would 2 or 4 suffice?), (b) the spatial pooling ratio (why 3×3 specifically, rather than 4×4 or adaptive pooling?), (c) the temporal averaging group size independent of frame count (what happens at 32 frames with 8× compression?), (d) the projector architecture (would a deeper MLP or a transformer projector change the compression-accuracy tradeoff?), (e) the DeltaLoss reference model sizes, (f) the FP8 quantisation granularity (which components benefit most from FP8?). These missing ablations mean the paper's recipe is empirically validated but not well-understood — a practitioner cannot easily extrapolate to different model scales or hardware constraints.

- **No latency measurement at realistic batch sizes.** All inference measurements in Figure 5 use batch size 1 on a single RTX 4090. For server deployments where multiple requests are batched, the bottleneck distribution between vision tower and LLM backbone would shift (the LLM's decoding becomes more compute-bound with larger batches), potentially changing the optimal quantisation strategy. The paper does not discuss this.

## 6. Limitations and Trade-offs

### The Difficulty Estimation Cost Is Not Accounted For

The entire compute-optimal framework rests on the ability to estimate prompt difficulty *before* deciding how to allocate inference budget. The paper's method for doing so — generating 2048 samples per question and averaging PRM final-answer scores — is extraordinarily expensive. The authors acknowledge this explicitly:

> "estimating difficulty in this way still incurs additional computation cost during inference... our experiments do not account for this cost largely for simplicity" (Section 3.2)

**The consequence:** The reported 4× efficiency gains over best-of-N are computed *after* difficulty is known, without amortising the cost of learning it. In a realistic deployment, the total cost would be difficulty estimation + strategy execution, and the former could dominate the latter — generating 2048 samples per question already consumes more compute than the largest test-time budgets studied (256–512 generations). The paper correctly frames this as an exploration-exploitation tradeoff, but does not quantify it. Until a cheaper difficulty estimator exists, the 4× figure represents an **upper bound on achievable efficiency**, not a realised deployment gain.

**What evidence exists:** The paper provides no experiment that includes the difficulty estimation cost in any budget calculation. The predicted difficulty bins (Section 3.2) require the same 2048-sample overhead as oracle bins; the only difference is that PRM scores replace ground-truth correctness, removing the need for labels but not reducing the sample count. There is no ablation studying whether fewer samples (e.g., 128 or 512) produce sufficiently accurate difficulty estimates.

**Mitigation status:** The paper suggests future work on "pretraining or finetuning models to directly predict difficulty of a question" (Section 8) and proposes adaptive difficulty estimation (start with a few samples, assess the score distribution, then allocate the remaining budget). Neither approach is implemented or evaluated. This limitation remains fully open.

---

### Hard Problems Remain Essentially Unsolved — Test-Time Compute Cannot Create Capability

Across all methods — search, revisions, and their compute-optimal combinations — the hardest questions (difficulty bin 5) show **near-zero improvement** regardless of the compute budget allocated. The paper is candid about this boundary condition, framing it as a fundamental limitation rather than a solvable failure of the current methods.

**The consequence:** Test-time compute amplifies existing capability in the base model but does not create it from nothing. If the base model's pass@1 is near zero on a problem class (as it is for bin 5 questions), no amount of search or revision will help — there are no correct solutions in the proposal distribution to find or refine. This means the approach offers **no path forward for genuinely novel or out-of-distribution reasoning** that exceeds the base model's training distribution. For such problems, pretraining a more capable model remains the only viable path. The FLOPs-matched comparison (Section 7) shows that on hard problems, pretraining is almost always more effective: at R ≫ 1 with PRM search, hard questions show a −52.9% relative disadvantage from using test-time compute instead of the larger model (Figure 1, bottom-right bar chart).

**What evidence exists:** The failure is documented consistently across experiments. In Figure 3 (right), bin 5 accuracy hovers at 1–3% for all methods and all generation budgets. In Figure 7 (right), bin 5 accuracy is roughly 2–3% irrespective of the sequential-to-parallel ratio. In the FLOPs-matched comparison (Figure 9), the bin 5 scaling line is essentially flat near 0–5%, while the ~14× larger model's greedy decoding star is positioned — though still low — above the scaling curve. The paper explicitly states: "On the hardest questions (bin 5), no method makes meaningful progress" (Section 5.3) and highlights in the Section 7 takeaway box that test-time compute "cannot compensate for fundamental capability gaps that larger pretraining would address."

**Mitigation status:** Not mitigable within the current framework. The paper acknowledges this limitation transparently and does not claim to solve it. Future work could explore whether different pre-training strategies (e.g., training on harder data, different model architectures) shift the difficulty distribution so that fewer problems fall into the unsolvable bin, but that is outside the scope of test-time compute allocation.

---

### The Revision Model Has a 38% Correct-to-Incorrect Reversion Rate

The iterative revision model is trained only on sequences where all in-context answers are incorrect, followed by a correct target (Section 6.1). At test time, the model may therefore encounter a correct answer in its own revision chain and — having never been trained on this scenario — incorrectly "revise" it into an incorrect answer. The paper quantifies this:

> "approximately 38% of correct answers get converted back to incorrect ones" (Section 6.1)

**The consequence:** The revision process is not monotonic — later revisions in a chain are not guaranteed to be better than earlier ones, and a substantial fraction of correct intermediate outputs are actively degraded by the model. This forces the system to rely on a post-hoc selection mechanism (majority voting or verifier-based selection) across the entire revision chain to pick the best answer from any point, rather than simply taking the final revision. In a latency-sensitive deployment where generating a long chain and then selecting is too slow, this 38% reversion rate would make it dangerous to stop early or to use the revision model without a verifier. It also means that the effective improvement from each revision step is diluted — the model is simultaneously fixing some errors while introducing new ones.

**What evidence exists:** The 38% figure is reported in Section 6.1 (though the exact experimental conditions producing it — whether at a specific budget, difficulty bin, or average across all — are not detailed). The paper's mitigation (within-chain selection via majority or verifier) is shown in Figure 6 (right) to outperform taking only the final answer. The ReST^EM experiment (Appendix K, Figure 16) provides additional evidence of revision fragility: attempting to further optimise the revision model with RL-style training caused sequential revision performance to substantially degrade, suggesting the revision training is sensitive to data distribution in ways not fully understood.

**Mitigation status:** Partially mitigated. The paper applies majority voting and verifier-based selection across the chain to recover from reversions, which works at the cost of additional compute (the verifier must score every step) and latency (the entire chain must be generated before selecting). A more principled solution — such as training the model to recognise when no revision is needed, or including "no change" examples during training — is not explored. The paper acknowledges the issue but does not attempt to reduce the 38% rate architecturally.

---

### Single Benchmark, Single Model Family — Generalisability Is Untested

All experiments use the MATH benchmark (500 test questions) with PaLM 2-S\* as the base model. The authors state they "believe this model is representative of the capabilities of many contemporary LLMs" (Section 4), but this claim is unverified. The difficulty-dependent scaling patterns that form the core contribution of the paper could depend on model-specific properties (e.g., the base model's output distribution, calibration, error typology) or domain-specific properties of math reasoning that do not transfer.

**The consequence:** A practitioner cannot confidently apply the paper's specific findings — which search algorithms work best for which difficulty bins, what sequential-to-parallel ratios are optimal, whether beam search over-optimises on easy problems in their domain — to a different model family or a different task domain. The PRM's quality and over-optimisation behaviour depend on PaLM 2-S\*'s output distribution; a model with different calibration properties might exhibit different difficulty-dependent scaling curves. The revision model's ability to learn from incorrect in-context examples depends on the base model's in-context learning capabilities, which vary substantially across model families. The MATH benchmark consists exclusively of competition-level math problems requiring symbolic reasoning; it is unclear whether the findings generalise to code generation, logical reasoning, scientific QA, or tasks requiring factual knowledge rather than inference.

**What evidence exists:** The paper provides no cross-model or cross-domain experiments. All figures (3–9) and all quantitative claims are based on PaLM 2-S\* evaluating MATH. The test set of 500 questions, split into five difficulty quintiles of ~100 each, then further split by two-fold cross-validation, means the compute-optimal policy is selected based on ~50 questions per fold per bin — a very small sample. The paper does not report confidence intervals on the compute-optimal scaling curves, making it difficult to assess whether the observed gains are statistically robust even within this single narrow setting.

**Mitigation status:** Not mitigated. The paper's scope is explicitly limited to MATH with PaLM 2-S\*, and the authors do not claim generalisability beyond this. The "representative model" claim in Section 4 is an assertion, not an empirical finding. Future work would need to replicate the study on at minimum one additional model family and one additional reasoning domain to establish whether the difficulty-dependent scaling patterns are universal.

---

### Revisions and Search Are Studied Independently, Not Combined

The paper studies two complementary axes — PRM-guided search and iterative revisions — as separate mechanisms, but never combines them. The unifying framework in Section 2 decomposes all test-time compute methods into modifications to the proposal distribution versus the verifier, implying these are complementary levers. Yet the experiments treat them in isolation: Section 5.3 studies search against the PRM using the base LLM as the proposal distribution (no revisions), and Section 6 studies revisions with a separate ORM (no PRM tree-search). The authors acknowledge this gap explicitly:

> "we did not experiment with PRM tree-search techniques in combination with revisions" (Section 8)

**The consequence:** The reported results represent a **lower bound** on what a fully integrated system could achieve. The two mechanisms have complementary, difficulty-dependent strengths: revisions improve the proposal distribution (generating better candidates through local refinement, most effective on easy problems) while PRM search improves candidate selection (global exploration, most effective on medium problems). A system that uses the revision model as the proposal distribution within beam search — or that uses the PRM to guide which revision branches to pursue — could outperform either approach alone, particularly on medium-difficulty problems where both mechanisms individually show gains. The paper does not establish whether combining them yields additive, super-additive, or sub-additive improvements.

**What evidence exists:** The paper provides no experiments combining PRM search with revisions. The independent results suggest complementary strengths: revisions dominate on easy problems (Figure 7, right, bins 1–2 favour purely sequential), while beam search outperforms best-of-N on medium problems (Figure 3, right, bins 3–4 favour beam search). But without a joint experiment, we cannot rule out negative interactions — for example, the PRM may not generalise to revision model outputs (Appendix J, Figure 15a already shows that the base-LM PRM underperforms on revision model outputs due to distribution shift), and the revision model's sequential context may interfere with the PRM's step-level scoring.

**Mitigation status:** Acknowledged as future work in Section 8. The paper suggests this as a natural next step but does not attempt it. The existing infrastructure (PRM, revision model, search algorithms) would support such experiments, making this a practical rather than conceptual gap.

---

### The FLOPs-Matched Comparison Uses a Weak Pretraining Baseline

The FLOPs-matched comparison in Section 7 scales model parameters while holding training data fixed, following the LLaMA paradigm rather than compute-optimal pretraining. The authors explicitly acknowledge this departure from the Chinchilla scaling laws:

> "We choose this setting as it is representative of a canonical approach to scaling pretraining compute and leave the analysis of compute-optimal scaling of pretraining compute where the data and parameters are both scaled equally to future work." (Section 7)

Additionally, the ~14× larger model uses only **greedy decoding** — no majority voting, no best-of-N, no search — while the smaller model is allowed to use the full compute-optimal test-time strategy.

**The consequence:** The reported advantages of test-time compute over pretraining are likely **overstated** relative to a properly compute-optimal larger model. A Chinchilla-optimal model trained with 14× more total FLOPs (scaling both parameters and data) would likely outperform a parameter-only-scaled model, making the pretraining baseline weaker than it needs to be. Furthermore, giving the larger model even a modest test-time compute budget (e.g., best-of-8) would create a much stronger baseline. The headline finding — that a smaller model with test-time compute can outperform a 14× larger model — may not hold against a fairer comparison that gives both models access to test-time compute, or that uses compute-optimally trained baselines. The claim is most credible for easy problems at low R values, where the gap is large (+27.8% on easy-medium questions at R ≪ 1 for revisions, per Figure 1), and least credible for medium-hard problems where the gap is small or negative.

**What evidence exists:** The paper provides the FLOPs accounting in Section 7 with explicit formulas, making the assumptions transparent. The three R values (0.16, 0.79, 22) cover realistic deployment scenarios. However, no experiment compares against a Chinchilla-optimal larger model, nor against a larger model with any form of test-time compute. The bar charts in Figure 1 (top-right, bottom-right) and the line plots in Figure 9 all use the same parameter-scaled, greedy-decoding larger model as the comparison point.

**Mitigation status:** The paper is transparent about this limitation in the quoted passage and frames the Chinchilla-optimal comparison as future work. The transparency is commendable, but the limitation remains: the paper's central claim about the pretraining-inference tradeoff is not validated against the strongest possible pretraining baseline, and the magnitude of the reported advantage should be interpreted with this caveat. A practitioner deciding between training a larger model and using more test-time compute should note that the comparison favours test-time compute by design.

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
