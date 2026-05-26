# Qwen2.5-VL Technical Report

**ArXiv:** [2502.13923](https://arxiv.org/abs/2502.13923)

## 🎯 Pitch

Qwen2.5-VL introduces a next-generation vision-language model series that pioneers fine-grained spatial perception, robust document parsing, and long-form video comprehension—delivering precise object localization, advanced document understanding, and second-level event localization in hours-long videos. By innovating with native dynamic resolution processing, efficient window attention, and absolute time-aligned multimodal embeddings, Qwen2.5-VL matches or outperforms leading closed-source models like GPT-4o and Claude 3.5 across key benchmarks. This work sets a new standard for real-world multimodal AI agents, enabling practical applications in business automation, device control, and video analytics at both large and resource-constrained scales.

---

## 1. Executive Summary

This technical report introduces **Qwen2.5-VL**, a flagship vision-language model series that advances multimodal understanding through four named mechanisms: **dynamic resolution processing** (converting images of varying sizes into token sequences of corresponding lengths without coordinate normalization), **window attention** in the vision encoder (reducing quadratic self-attention cost to linear scaling for native-resolution inputs), **Multimodal Rotary Position Embedding aligned to absolute time** (extending MRoPE temporal IDs to encode real timestamps rather than frame indices), and **dynamic FPS sampling** (adapting frame rate during training to match video content tempo). In extensive benchmark comparisons against GPT-4o, Claude 3.5 Sonnet, and InternVL2.5-78B, the flagship Qwen2.5-VL-72B matches or surpasses all competitors—most notably achieving 96.4% on DocVQA, 89.5% on ChartQA, 87.3% on InfoVQA, and 61.5/63.7% on OCRBench_v2 (English/Chinese, exceeding Gemini 1.5 Pro by 9.6 and 20.6 percentage points respectively)—while establishing that strong document parsing and precise object grounding capabilities can be maintained even in the 3B and 7B parameter variants, which outperform all comparably-sized open-source models across general VQA, video understanding, and GUI agent benchmarks.

## 2. Context and Motivation

### The Core Problem: LVLMs Are Competent but Not Exceptional at Fine-Grained Perception

The paper opens with a striking metaphor that frames its entire motivation: current vision-language models are like "the middle layer of a sandwich cookie—competent across various tasks but falling short of exceptional performance." The foundational layer that remains underdeveloped is **fine-grained visual perception**—the ability to precisely localize objects, parse structured information from documents, count instances accurately, and understand temporal dynamics in long videos. The top layer is multimodal reasoning, which builds on this perceptual foundation.

This gap matters because real-world applications demand more than approximate visual understanding. A model that can describe an image in general terms but cannot reliably extract structured data from an invoice, count objects in a crowded scene, or localize an event to a specific second in a multi-hour video is fundamentally limited as an interactive agent. The paper identifies four specific capability deficits that define this gap:

**1. Document parsing beyond text recognition.** Prior models treat document understanding primarily as OCR—recognizing text characters in images. But real documents contain tables, charts, chemical formulas, music sheets, and complex layouts where the spatial arrangement of elements carries meaning. Extracting structured, machine-readable representations (e.g., HTML with bounding boxes) from arbitrary documents requires a unified model that understands both the content and the layout of every element simultaneously, not a pipeline of specialist models.

**2. Object grounding with absolute spatial coordinates.** Most LVLMs represent spatial locations using relative coordinates (e.g., percentages of image dimensions), which discards scale information. For a model to function as a visual agent—clicking buttons on a screen, identifying small objects in high-resolution images, or reasoning about object sizes—it must operate in absolute pixel coordinates that preserve the true spatial extent of objects.

**3. Temporal understanding in long videos at second-level granularity.** Video understanding models typically process uniformly sampled frames and treat time as a frame index. This fails to capture the actual tempo of events (a video recorded at 30 FPS and one at 60 FPS contain the same event at different frame rates) and prevents precise temporal localization (e.g., "the dog jumped at 3:42").

**4. Agent capabilities grounded in visual perception.** For a model to operate computers or mobile devices, it must perceive UI elements, understand their spatial layout, reason about action sequences, and execute precise interactions. Prior models either lack the grounding accuracy to click on small UI elements or rely on auxiliary systems (e.g., Set-of-Mark prompting that overlays numbered bounding boxes) rather than native visual understanding.

These deficits are interconnected: a model cannot be an effective agent without precise grounding, cannot parse complex documents without spatial understanding, and cannot understand long videos without temporal reasoning. The paper positions Qwen2.5-VL as addressing this entire stack simultaneously.

### Why This Problem Matters: The Shift Toward Agentic AI

The paper's emphasis on fine-grained perception is motivated by a broader paradigm shift in AI: models are moving from passive understanding (describing images, answering questions) to active interaction (operating devices, executing multi-step tasks, extracting and transforming data). This shift demands capabilities that general-purpose vision-language models have historically deprioritized in favor of benchmark performance on standard VQA tasks.

**Agentic applications require spatial precision.** When a model clicks a button on a screen, a 5-pixel error makes the difference between success and failure. Prior LVLMs that output relative coordinates (e.g., "the button is at [0.3, 0.5]") must rely on post-hoc coordinate mapping that fails when screenshots are resized or aspect ratios shift. The paper's absolute coordinate approach eliminates this failure mode by training the model to natively understand pixel-level spatial relationships.

**Document processing requires structured extraction, not just reading.** In enterprise settings, the ability to parse an invoice into structured data (line items, totals, dates) or convert a PDF table into a spreadsheet is far more valuable than the ability to answer questions about the document's content. Prior models handle document VQA reasonably well but struggle with the structured extraction task. Qwen2.5-VL's HTML-based omni-parsing format directly targets this gap.

**Video understanding requires temporal precision.** Surveillance, content moderation, and video analytics applications need to answer "when did X happen?" not just "did X happen?" The paper's absolute time encoding enables the model to output precise timestamps (in seconds or HH:MM:SS:frame formats) rather than rough frame ranges, making it deployable in time-critical applications.

**Resource-constrained deployment needs efficient architectures.** The paper produces models at 3B, 7B, and 72B parameters, explicitly targeting "diverse use cases from edge AI to high-performance computing." This reflects the practical reality that agentic applications often run on-device (phones, laptops) where 72B parameter models are infeasible. The challenge is to preserve fine-grained perceptual capabilities at smaller scales, which prior open-source models have generally failed to do.

### Where Prior Approaches Fall Short

The paper's introduction surveys the LVLM landscape and identifies specific limitations that motivate its technical contributions:

**Computational complexity from native resolution processing.** Recent models (Chen et al., 2023; Liu et al., 2024b; Liang et al., 2025) have demonstrated that higher input resolution improves visual understanding quality. However, processing images at native resolution creates quadratic self-attention costs in the Vision Transformer—an image with 4× the resolution produces 16× the attention computation. The paper directly addresses this: "To mitigate this, we introduce windowed attention in most layers, which ensures that computational cost scales linearly with the number of patches rather than quadratically" (Section 2.1.1). This isn't merely an efficiency improvement—it's what makes training on native-resolution images computationally feasible at all.

**Limited contextual understanding across varied sequence lengths.** The paper notes "inconsistent performance across varied sequence length" as a developmental bottleneck. Prior models typically resize images to fixed dimensions or pad them, which either distorts spatial relationships or wastes computation on padding tokens. The paper's dynamic resolution approach processes images of arbitrary aspect ratios at their native dimensions, producing variable-length token sequences that reflect the actual information content of each image.

**Poor fine-grained visual perception.** As noted above, prior models (including Qwen2-VL, InternVL2.5, GPT-4o, and Claude 3.5 Sonnet) all exhibit significant performance gaps on tasks requiring precise spatial localization, counting, and structured document extraction. The paper's benchmarks make this concrete: on OCRBench_v2 (English), Qwen2.5-VL-72B achieves 61.5% compared to Gemini 1.5 Pro's 51.9% and GPT-4o's 46.5%—a gap of 9.6 and 15.0 percentage points respectively. On CountBench, Qwen2.5-VL-72B's 93.6% accuracy dramatically exceeds InternVL2.5-78B's 72.1%. These numbers quantify the perception gap that the paper aims to close.

**Temporal understanding tied to frame indices, not real time.** The paper explicitly critiques Qwen2-VL's temporal encoding: "in Qwen2-VL, the temporal position IDs in MRoPE were tied to the number of input frames, which did not account for the speed of content changes or the absolute timing of events within the video" (Section 2.1.3). This means that Qwen2-VL cannot distinguish between a fast-paced event captured at high FPS and a slow event captured at low FPS—both would produce similar token sequence lengths with similar temporal position IDs. The absolute time alignment in Qwen2.5-VL fixes this by encoding real timestamps, enabling the model to learn the tempo of events through the intervals between temporal IDs.

**OCR-centric document understanding.** Prior approaches to document understanding focus heavily on text recognition (OCR) accuracy but treat layout, charts, tables, and other structured elements as separate problems requiring specialist models. The paper contrasts this with Qwen2.5-VL's unified approach: "Traditional methods for parsing document content typically rely on separate models to handle layout analysis, text extraction, chart interpretation, and illustration processing. In contrast, Qwen2.5-VL is designed to empower a general-purpose model with comprehensive capabilities for parsing, understanding, and converting document formats" (Section 2.2.1).

**Agent capabilities requiring auxiliary systems.** The paper's agent evaluation reveals that models like GPT-4o (34.5% on AndroidWorld) and Gemini 2.0 (26%) require Set-of-Mark prompting—overlaying numbered bounding boxes on screenshots—to achieve even moderate performance. Qwen2.5-VL-72B achieves 35% on AndroidWorld without auxiliary marks, suggesting its native grounding capabilities are more robust. On ScreenSpot Pro (professional high-resolution computer use), Qwen2.5-VL-72B's 43.6% dramatically exceeds GPT-4o (not reported), Claude (17.1%), and Aguvis-72B (23.6%).

### How This Paper Positions Itself

The paper positions Qwen2.5-VL as an **open-source flagship** that "continues the open-source philosophy of the Qwen series, achieving and even surpassing top-tier closed-source models on various benchmarks." This positioning is strategic: by releasing models at 3B, 7B, and 72B scales, the paper provides both a competitive alternative to proprietary systems (GPT-4o, Claude 3.5 Sonnet, Gemini 1.5 Pro) and a resource-efficient option for edge deployment. The availability of all model sizes distinguishes Qwen2.5-VL from prior open-source efforts that typically release only a single large variant.

The paper's four named contributions (Section 1) each address specific gaps in prior work:

1. **Window attention in the visual encoder** directly attacks the quadratic complexity problem that makes native-resolution ViT processing expensive. The design choice to use full self-attention in only 4 of 32 layers (layers {7, 15, 23, 31}) with windowed attention elsewhere is a specific architectural decision that balances global context (needed for spatial reasoning across the full image) with computational efficiency (needed for training at scale).

2. **Dynamic FPS sampling extending dynamic resolution to the temporal dimension** addresses the limitation that prior models treat all videos with uniform frame sampling regardless of content tempo. By training on variable FPS, the model learns to handle both sparse sampling for static content (e.g., lecture videos) and dense sampling for fast action (e.g., sports).

3. **MRoPE aligned to absolute time** is the paper's core temporal innovation, fixing Qwen2-VL's inability to model real timing. The key insight is that the *intervals between temporal IDs* encode tempo information—if frame 1 is at time 0.0s and frame 2 at time 0.5s, the ID gap encodes a 0.5s interval, whereas if frame 2 is at time 5.0s, the gap encodes 5.0s. This allows the same model architecture to handle videos with different FPS without modification.

4. **Scaling pre-training data from 1.2T to 4.1T tokens** with careful data curation (including interleaved image-text scoring, grounding data with absolute coordinates, and the novel HTML-based document parsing format) follows the now-standard recipe of data quality improvement driving model capability, but the paper provides specific, replicable details about the data filtering pipeline and synthetic data generation procedures.

The paper explicitly frames Qwen2.5-VL as building on Qwen2-VL (Wang et al., 2024e) while addressing its identified limitations: the temporal encoding, the computational efficiency of the vision encoder, and the scope and quality of grounding and document parsing data. It also positions itself within the broader LVLM ecosystem by referencing the standard three-component architecture (visual encoder, cross-modal projector, LLM) established by Alayrac et al. (2022), Li et al. (2022a, 2023b), and Liu et al. (2023b,a), while innovating within each component.

A key strategic choice is the **from-scratch training of the Vision Transformer** rather than using a pre-trained vision backbone (e.g., CLIP ViT). This enables the architectural modifications (window attention, SwiGLU activations, RMSNorm) to be integrated from initialization, and allows the ViT to be co-optimized with the LLM throughout the full training pipeline. The paper argues this leads to better alignment with the language model component, though it requires substantially more vision pre-training data (using DataComp and in-house datasets).

Finally, the paper positions its agent capabilities not as an add-on but as an integral capability emerging from the same perceptual foundations: "Leverage advanced grounding, reasoning, and decision-making abilities, boosting the model with superior agent functionality on smartphones and computers." The agent training data includes reasoning traces that explain *why* each UI action is taken, preventing the model from simply memorizing action sequences and instead learning to generalize to novel interfaces.

## 3. Technical Approach

### 3.1 Reader Orientation

Qwen2.5-VL is a vision-language model that takes images, videos, and text as input and produces text (including structured formats like HTML with bounding boxes) as output. The system solves the problem of making a single neural network excel simultaneously at fine-grained visual perception (precise object localization, document parsing, temporal event grounding) and high-level multimodal reasoning by redesigning the vision encoder for native-resolution efficiency, encoding absolute time into positional embeddings, and training on a carefully curated 4.1 trillion token corpus that includes structured grounding and document parsing data in formats the model can directly generate.

### 3.2 Big-Picture Architecture (Diagram in Words)

The system has three major components connected in a feedforward pipeline:

1. **Vision Encoder (ViT)** — receives images or video frames at their native resolutions. Splits them into 14×14 pixel patches, processes them through 32 transformer layers (28 with windowed attention, 4 with full self-attention), and outputs a set of feature vectors. For video, pairs of consecutive frames share patches to reduce token count.

2. **Vision-Language Merger (MLP)** — receives the ViT's output feature grid. Groups spatially adjacent groups of 4 patch features, concatenates them, and projects them through a two-layer MLP to match the LLM's embedding dimension. This compresses the visual token sequence by a factor of 4 before it enters the LLM.

3. **Large Language Model (Qwen2.5 LLM)** — receives the compressed visual tokens interleaved with text tokens. Uses Multimodal Rotary Position Embedding (MRoPE) that encodes temporal, height, and width position components separately, with temporal IDs aligned to absolute time for video inputs. Generates text output autoregressively, including structured formats like HTML with bounding box attributes.

Information flows left-to-right: raw pixels → ViT features → MLP compression → LLM token sequence → generated text. The key innovation is that spatial and temporal information is preserved natively throughout—the ViT processes images at their original resolution without resizing, and the MRoPE encodes real timestamps rather than frame indices.

### 3.3 Roadmap for the Deep Dive

- **First**, the Vision Transformer architecture and window attention mechanism, because this is what makes native-resolution processing computationally feasible and is the paper's primary efficiency innovation.
- **Second**, the dynamic resolution and FPS processing pipeline, because understanding how images and videos of varying sizes and frame rates are converted to token sequences is prerequisite to understanding the spatial and temporal encoding choices.
- **Third**, the Multimodal Rotary Position Embedding (MRoPE) aligned to absolute time, because this is the paper's core temporal reasoning innovation and builds directly on the dynamic FPS concept.
- **Fourth**, the pre-training data construction and training recipe, because the model's capabilities emerge from the scale (4.1T tokens) and specific design of the training corpus (HTML-based document parsing format, absolute coordinate grounding data, interleaved image-text scoring pipeline).
- **Fifth**, the post-training alignment strategy (SFT + DPO), because this is where instruction-following behavior is installed and the specialized capabilities demonstrated in the experiments are refined.
- **Sixth**, the model configuration details and scaling variants, because understanding the three size tiers (3B, 7B, 72B) and what architectural features are shared versus scaled provides context for the experimental comparisons.

### 3.4 Detailed, Sentence-Based Technical Breakdown

This is primarily a **systems and engineering paper** whose core idea is that fine-grained visual perception in LVLMs can be achieved by (1) redesigning the ViT for native-resolution efficiency via window attention, (2) encoding absolute temporal information directly into position embeddings, and (3) training on a large corpus with structured output formats that teach the model to generate spatial and temporal annotations natively.

---

#### Vision Transformer Architecture and Window Attention

The vision encoder is a 32-layer Vision Transformer trained from scratch. All three model sizes (3B, 7B, 72B) share an identical ViT configuration: hidden size 1280, 16 attention heads, intermediate size 3456, patch size 14, window size 112 (Table 1). This shared ViT architecture means that the visual processing pipeline is identical across all model scales—only the LLM and the vision-language merger dimensions vary between the 3B, 7B, and 72B variants.

The critical architectural innovation is **windowed attention in most layers**. In a standard ViT, every patch attends to every other patch, producing self-attention cost that scales as `$O(N^2)$` where `$N$` is the number of patches. For a high-resolution image of dimensions `$H \times W$`, the number of patches is `$(H/14) \times (W/14)$`. An image with twice the linear resolution produces four times the patches and therefore sixteen times the self-attention computation. This quadratic relationship is what makes native-resolution training prohibitively expensive in standard ViTs.

The paper's solution is to apply **self-attention only within local windows** of maximum size 112×112 pixels (which corresponds to 8×8 patches since patch size is 14) in 28 of the 32 layers. The computational cost for these layers scales as `$O(N \cdot W^2)$` where `$W = 64$` (the number of patches per window), which is linear in the total number of patches `$N$`. The remaining 4 layers—specifically layers at indices {7, 15, 23, 31} as specified in Table 1—use full self-attention, giving the model periodic access to global context. These full-attention layers are evenly spaced (every 8 layers starting at layer 7), which provides a balanced trade-off: the model can propagate information globally through these layers while the intervening windowed layers perform efficient local processing.

A subtlety noted in the paper but not elaborated: "Regions smaller than 112×112 are processed without padding, preserving their original resolution." This means the window attention implementation is adaptive—when an image dimension is not a multiple of 112, the edge windows are smaller rather than being zero-padded to the full window size. This avoids wasting computation on padding tokens and preserves the exact spatial relationships at image boundaries, which matters for precise grounding tasks where objects may appear near edges.

The ViT also incorporates **two architectural alignment choices** with the LLM backbone: RMSNorm for normalization (instead of LayerNorm) and SwiGLU as the activation function in the feed-forward networks. These choices are justified as improving "both computational efficiency and compatibility between the vision and language components of the model." The compatibility claim refers to the fact that when the ViT and LLM use the same normalization and activation functions, the cross-modal projector (the MLP merger) operates between components with similar activation statistics, which empirically improves training stability.

For **positional encoding within the ViT**, the paper adopts 2D Rotary Position Embedding (2D-RoPE). Standard 1D RoPE encodes position along a single sequence dimension; 2D-RoPE separately encodes the row and column position of each patch in the 2D grid, allowing the attention mechanism to distinguish spatial relationships (above/below, left/right) rather than treating patches as an undifferentiated 1D sequence. The paper does not provide the exact formulation of 2D-RoPE, but the standard approach is to apply RoPE with different frequency bases to the height and width dimensions.

For **video input**, the ViT uses 3D patch partitioning with a temporal grouping of 2: "two consecutive frames are grouped together" before patch extraction. This means that for a video clip, the ViT processes pairs of adjacent frames as a single unit, reducing the total number of tokens by a factor of approximately 2 compared to processing each frame independently. The patch extraction uses the same 14×14 spatial patch size, but the temporal grouping means each "patch" actually spans 2 frames × 14 pixels × 14 pixels. This design choice preserves compatibility with the image-processing architecture (the patch dimension is still 14×14 spatially) while extending to video with minimal architectural change.

**Training the ViT from scratch** (rather than using a pre-trained vision backbone like CLIP) is a significant engineering decision. The paper argues this enables the architectural modifications (window attention, SwiGLU, RMSNorm) to be integrated from initialization, but it also means the ViT must learn all visual representations from scratch during the multi-stage pre-training process. The first pre-training phase trains only the ViT (the LLM is frozen) on image captions, visual knowledge data, and OCR data for approximately 1.5 trillion tokens. This is the effective equivalent of a CLIP-style pre-training but with the custom architecture and with the explicit goal of aligning the ViT's representations with the frozen LLM's embedding space.

---

#### Dynamic Resolution Processing in the Spatial Domain

Qwen2.5-VL processes images at their **native resolution** without resizing to a fixed dimension. The image is resized such that both height and width are multiples of 28 before being fed into the ViT. This constraint arises from the patch size (14) and the vision-language merger's grouping factor (4 patches grouped together → 2×2 grouping requires even numbers of patches in each dimension, and the alignment to 28 = 14 × 2 ensures whole-patch boundaries). After this minimal resizing, the image is split into 14×14 patches with stride 14 (non-overlapping), producing a `$(H/14) \times (W/14)$` grid of patch features.

The key downstream consequence is that **spatial coordinates are represented in absolute pixel values** rather than normalized to [0,1] or [0,1000]. When the model outputs a bounding box, it uses the actual pixel dimensions of the original input image:

> "unlike traditional approaches that normalize coordinates, our model directly uses the actual dimensions of the input image to represent bounding boxes, points, and other spatial features. This allows the model to learn scale information inherently."

This is a significant departure from prior work. In models that normalize coordinates (outputting values in [0,1] that get mapped back to image dimensions), the model has no access to the absolute scale of objects—a bounding box spanning [0.1, 0.3] in normalized coordinates could represent a small object in a large image or a large object in a small image. By training the model to output absolute pixel coordinates, the model can learn that, for example, a bounding box of width 200 pixels in a 4000-pixel-wide image represents a small object, whereas the same width in an 800-pixel-wide image represents a large object. The scale information is preserved in the absolute values.

This design choice imposes a requirement on training data: all grounding annotations must be provided in absolute pixel coordinates rather than normalized coordinates. The paper describes generating such data through both public datasets and synthetic pipelines using Grounding DINO (for object detection pseudo-labels) and SAM (for segmentation masks converted to bounding boxes).

The **vision-language merger** performs spatial compression. After the ViT produces a `$(H/14) \times (W/14)$` grid of features (each of dimension 1280), the merger groups adjacent 2×2 blocks of patch features, concatenates the 4 feature vectors into a single vector of dimension 4 × 1280 = 5120, and projects this through a two-layer MLP to the LLM's embedding dimension (2048 for 3B, 3584 for 7B, 8192 for 72B). The MLP architecture is:

**Merger input:** 5120-dimensional concatenated feature vectors (4 ViT features × 1280 each).

**Merger hidden layer:** the paper does not specify the hidden dimension, but a standard two-layer MLP would have: input 5120 → hidden → output `$d_{LLM}$`.

**Merger output:** feature vectors of dimension `$d_{LLM}$` (2048, 3584, or 8192 depending on model size).

The effect is a **4× spatial compression**: the `$(H/14) \times (W/14)$` patch grid becomes a `$(H/28) \times (W/28)$` token grid fed into the LLM. This is the origin of the "multiples of 28" constraint—the compressed token grid dimensions must be integers. The compression serves dual purposes: it reduces the sequence length fed to the LLM (which has `$O(N^2)$` attention cost) and it provides a flexible mechanism for handling images of varying sizes (the output sequence length scales with image area divided by 28²).

The paper notes that this method "not only reduces computational costs but also provides a flexible way to dynamically compress image feature sequences of varying lengths." The flexibility comes from the fact that the compression ratio is fixed (always 4× spatially), so the LLM receives a number of visual tokens proportional to the image area. A small 224×224 image produces an 8×8 token grid (64 tokens); a large 4032×3024 image produces a 144×108 token grid (15,552 tokens). The LLM's context window must accommodate these variable-length visual sequences.

---

#### Dynamic FPS Training and Temporal Processing

For video inputs, Qwen2.5-VL extends the dynamic resolution concept to the **temporal dimension** through dynamic FPS (frames per second) sampling. The key idea is that videos contain content at different temporal densities—a lecture video with a static slide changes slowly (low effective FPS would suffice), while a sports clip requires high temporal resolution to capture fast action. Prior models typically extract frames at a fixed rate (e.g., 1 frame per second) or a fixed total number of frames regardless of video duration, which either wastes computation on static content or misses fast events.

The paper's approach is to **vary the FPS during training** so that the model encounters videos sampled at different rates:

> "we dynamically sampled FPS during training to achieve a more evenly distributed representation of FPS within the training dataset."

This means that the same video might be processed at 0.5 FPS in one training epoch (producing few frames, emphasizing long-term structure) and at 30 FPS in another (producing many frames, emphasizing fine-grained motion). The model learns to handle both sparse and dense temporal sampling, and the training distribution is explicitly balanced to avoid over-representing any particular FPS range.

For **long videos** (exceeding half an hour), the paper constructs "a set of long video captions by synthesizing multi-frame captions through a targeted synthesis pipeline." The approach is multi-stage: first, sample frames at some interval; second, generate captions for individual frames or short clips; third, synthesize these into a coherent description covering the full video duration. This addresses the challenge that human-annotated captions rarely exist for multi-hour videos.

The **temporal grounding data** is formatted using two timestamp representations: "second-based formats and hour-minute-second-frame (hmsf) formats." Training the model to both understand and output timestamps in these formats ensures it can handle precise temporal localization queries ("When did the dog jump? → at 3:42:15") and generate temporally grounded descriptions. The inclusion of both formats suggests the model was trained to be flexible—it can output `212.5` (seconds) or `00:03:32:15` (hours:minutes:seconds:frames) depending on the prompt format.

At inference time, video inputs are processed with a **cap of 768 frames** per video and a **maximum of 24,576 visual tokens** (Section 3.3.4). These limits are practical constraints matching the LLM's maximum sequence length (32,768 after long-context pre-training in phase 3). A video at the cap would produce 768 × (patches per frame) visual tokens, which after the 4× spatial compression yields approximately 768 × (H/28) × (W/28) tokens. For typical video resolutions (e.g., 224×224 would yield 8×8=64 tokens per frame, giving 768 × 64 / 4 ≈ 12,288 tokens after compression), this fits within the budget with room for text tokens.

The interaction between **frame rate and the frame cap** creates an implicit trade-off: a 1-hour video at 30 FPS contains 108,000 frames, of which only 768 can be processed. The dynamic FPS sampling during training teaches the model that the frame cap means it sees a subset, and the absolute time encoding (discussed next) tells it where each sampled frame falls in the original timeline. This allows the model to reason about temporal gaps—if frames are sampled at 10-second intervals, the model knows that the gap between consecutive frames represents 10 seconds of elapsed time, not an instantaneous transition.

---

#### Multimodal Rotary Position Embedding (MRoPE) Aligned to Absolute Time

MRoPE is the paper's mechanism for encoding position information across three modalities (text, images, video) while maintaining compatibility with the underlying RoPE machinery. The innovation in Qwen2.5-VL is extending MRoPE to align the temporal dimension with **absolute time** rather than frame indices.

**Background on RoPE.** Rotary Position Embedding (RoPE) encodes position by rotating the query and key vectors in self-attention by an angle proportional to the token's position index. For a token at position `$p$`, the rotation angle for the `$i$`-th dimension pair is `$p \cdot \theta_i$` where `$\theta_i = 10000^{-2i/d}$` (following the original formulation; Su et al., 2024). The key property is that the dot product between query at position `$p$` and key at position `$q$` depends only on their relative distance `$p - q$`, via the trigonometric identity for angle differences.

**MRoPE decomposition.** MRoPE extends this by decomposing the position ID into three components:

- `$p_t$` — temporal position ID
- `$p_h$` — height (row) position ID
- `$p_w$` — width (column) position ID

Each component is assigned a subset of the RoPE frequency dimensions. For a text token, all three components use the same ID (the token's 1D position in the sequence), making MRoPE equivalent to standard 1D RoPE. For an image token, `$p_t$` is constant across all visual tokens (same "time"), while `$p_h$` and `$p_w$` vary based on the token's spatial position in the 2D patch grid. For a video token, `$p_t$` increments with each frame, while `$p_h$` and `$p_w$` follow the same per-frame spatial assignment as images.

The decomposition is not simply additive—the paper implies (building on Qwen2-VL) that different frequency bands of the RoPE embedding are allocated to different position components. For example, low-frequency dimensions might encode temporal position (capturing slow changes across frames) while high-frequency dimensions encode spatial position (capturing fine-grained within-frame structure). The exact allocation is not specified, but the principle is that the attention mechanism can simultaneously attend based on temporal proximity, vertical proximity, and horizontal proximity, with the relevance of each determined by which frequency band dominates the query-key dot product.

**The absolute time alignment problem.** In Qwen2-VL, temporal position IDs were assigned sequentially: frame 1 gets `$p_t = 1$`, frame 2 gets `$p_t = 2$`, etc. This means that the temporal distance between adjacent frames is always 1, regardless of how much real time elapsed between them. A model trained on this encoding cannot distinguish between:
- A 30 FPS video where frames 1 and 2 are 33ms apart (temporal ID difference = 1, real time difference = 0.033s)
- A 0.5 FPS video where frames 1 and 2 are 2 seconds apart (temporal ID difference = 1, real time difference = 2.0s)

Both would produce identical temporal position encodings, destroying the model's ability to learn the tempo of events.

**The solution: align `$p_t$` to timestamps.** Qwen2.5-VL assigns temporal position IDs based on the **absolute timestamp** of each frame:

$$p_t = f(t_{\text{frame}})$$

where `$t_{\text{frame}}$` is the frame's timestamp in seconds (or some other absolute time unit) and `$f$` is a mapping function that converts timestamps to position IDs. The paper does not specify the exact function `$f$`, but the critical property is that it preserves temporal intervals: if two frames are 0.5 seconds apart, their temporal position IDs differ by an amount proportional to 0.5, regardless of how many other frames exist between them. Similarly, if two frames are 5.0 seconds apart, their IDs differ by 10× the amount for a 0.5-second gap.

**What this achieves operationally.** When the model computes attention between two video tokens, the temporal component of the MRoPE rotation encodes their real-time separation. This means:
- The model can learn that events happening at 30 FPS involve rapid changes between frames with small temporal ID gaps.
- The model can learn that events happening at 1 FPS involve slower changes between frames with larger temporal ID gaps.
- When prompted with "What happened at 3:42?", the model can identify which frames have temporal IDs closest to the timestamp 3:42 (converted to the same ID space).
- When generating timestamp-grounded output (e.g., "The dog jumped at 3.5 seconds"), the model can output timestamps that correspond to specific temporal position IDs.

The paper emphasizes the elegance of this approach: "without necessitating any additional computational overhead." Unlike approaches that add separate temporal embedding modules, timestamp tokens, or auxiliary prediction heads, the MRoPE alignment modifies only the assignment of position IDs to frames—the RoPE computation, the attention mechanism, and the model architecture remain unchanged. This is a minimal-modification approach that leverages existing infrastructure.

**Why absolute time over frame-index encoding matters for real-world video.** Consider a surveillance application where the model must answer "When did the person enter the building?" The answer should be a wall-clock time (e.g., "14:32:15"), not a frame index. With frame-index encoding, the model must learn a separate mapping from frame indices to timestamps based on metadata, which is unreliable (different videos have different FPS, and FPS may vary within a single video). With absolute time encoding, the temporal position IDs directly encode the timestamp, so the attention pattern that identifies "the person entering the building" naturally activates at the ID corresponding to the entry time.

**Interaction with dynamic FPS training.** The combination of dynamic FPS (varying frame sampling rates during training) and absolute time MRoPE is synergistic. During training, the model encounters the same video sampled at different rates, producing different numbers of frames with different temporal ID gaps between them. The MRoPE encoding ensures that regardless of the sampling rate, the temporal distances between events are preserved in the position IDs. This teaches the model that the *content* of events is independent of the *sampling rate*, and that temporal reasoning should be based on real time intervals, not frame counts.

---

#### Pre-Training Data Construction

The pre-training corpus was expanded from Qwen2-VL's 1.2 trillion tokens to approximately 4.1 trillion tokens. The paper describes five major data categories, each with specific construction pipelines and purposes.

**Interleaved Image-Text Data Scoring Pipeline.** Interleaved data (documents where images and text alternate naturally, as in web pages, articles, and textbooks) provides three benefits: (1) enabling in-context learning with simultaneous visual and textual cues, (2) maintaining text-only performance when images are absent, and (3) containing diverse general knowledge. However, raw interleaved data from the web is noisy—many image-text pairings are irrelevant or decorative.

The paper develops a **four-stage scoring system** evaluated by an internal model:

1. **Text-only quality:** evaluates the linguistic quality of the text portion in isolation (grammar, coherence, informativeness) to filter out documents where the text itself is low-quality regardless of images.

2. **Image-text relevance:** measures "how much stronger a connection between the image and text" exists, where "the image meaningfully supplements, explains or expands on the text rather than just decorating it." This filters out cases where images are tangentially related or purely decorative.

3. **Image-text complementarity:** evaluates whether "each should provide unique details that together create a complete narrative." High-scoring examples have information distributed between modalities such that removing either would lose essential information. This penalizes redundancy (image and text conveying the same information) and independence (image and text being unrelated).

4. **Balance of information density:** measures whether the distribution of information between image and text avoids "excessive text or image information." A document with 10,000 words of text and one small decorative image scores low; a document with a single image and no explanatory text also scores low.

The pipeline applies standard data cleaning first (following Li et al., 2024e) and then uses the four scoring criteria sequentially. The paper does not provide the thresholds used for filtering or the exact architecture of the scoring model, but the criteria themselves represent a principled decomposition of what makes interleaved data useful for multimodal learning.

The purpose of this elaborate filtering is to address a known problem: "much of the available interleaved data lacks meaningful text-image associations and is often noisy, limiting its usefulness for complex reasoning and creative generation." By training only on data that satisfies all four criteria (high text quality, high relevance, high complementarity, good balance), the model learns to integrate visual and textual information rather than treating images as optional add-ons.

**Grounding Data with Absolute Position Coordinates.** Unlike relative coordinate systems (where positions are expressed as fractions of image dimensions), Qwen2.5-VL is trained on **absolute pixel coordinates** for bounding boxes and points. The data construction involves:

**Data sources:** Public datasets (the paper does not enumerate them, but typical sources include RefCOCO, RefCOCO+, RefCOCOg for referring expression comprehension, and COCO, Objects365, OpenImages for object detection), plus proprietary in-house datasets.

**Format diversity:** The data is synthesized into "various formats, including XML, JSON, and custom formats." This teaches the model to output structured spatial information in multiple representations, making it flexible to different prompting styles and downstream use cases.

**Copy-paste augmentation (Ghiasi et al., 2021):** Objects from one image are copied and pasted into another at various positions and scales, with the bounding box annotations automatically updated. This creates training examples where the model must locate objects in cluttered or unnatural contexts, improving robustness.

**Synthesis with specialist models:** Grounding DINO (Liu et al., 2023c) is used to generate bounding box proposals for objects described by referring expressions, and SAM (Kirillov et al., 2023) is used to convert these boxes to precise segmentation masks. The outputs are used as pseudo-ground-truth for training, expanding the dataset beyond human-annotated examples.

**Expanded object categories:** "To enhance the model's performance on open-vocabulary detection, we expanded the training dataset to include over 10,000 object categories." This is critical for generalization—a model trained on only 80 COCO categories would fail on the long tail of objects encountered in the real world. The 10,000+ categories likely come from combining multiple detection datasets (Objects365 has 365 categories, OpenImages has 600, LVIS has 1200+) with synthetic expansion using LLMs to generate category names.

**Extreme detection scenarios:** The paper synthesizes "non-existent object categories within the queries and constructed image data containing multiple instances for each object." The first type (querying for objects not in the image) teaches the model to correctly respond "not present" rather than hallucinating. The second type (multiple instances of the same category) teaches precise counting and disambiguation between instances.

**Point grounding data:** Separate from bounding boxes, the model is trained to point to specific locations. Data sources include "public pointing and counting data from PixMo (Deitke et al., 2024), publicly accessible object grounding data (from both object detection and instance segmentation tasks), and data synthesized by an automated pipeline for generating precise pointing data towards certain image details." The pointing task requires higher spatial precision than bounding boxes—a box can be 10 pixels off and still contain the object; a point 10 pixels off misses entirely.

**Document Omni-Parsing Data with HTML Format.** The paper's most distinctive data innovation is training the model to parse entire documents into a structured HTML format that captures layout, text content, and spatial coordinates simultaneously.

Traditional document processing pipelines use separate models for:
- Layout analysis (identifying text blocks, tables, figures)
- OCR (extracting text from each block)
- Table extraction (identifying cell boundaries and content)
- Chart interpretation (converting visual data to structured data)
- Formula recognition (converting equation images to LaTeX or MathML)

Qwen2.5-VL is trained to perform all of these tasks in a single forward pass by generating the document's content in the **QwenVL HTML format**. The format uses standard HTML tags with custom attributes:

```
<html><body>
<p data-bbox="x1 y1 x2 y2"> content </p>
<table data-bbox="x1 y1 x2 y2" class="table{id}"> table content </table>
<div class="chart" data-bbox="x1 y1 x2 y2"> <img data-bbox="..."/> <table> chart content </table></div>
<div class="formula" data-bbox="x1 y1 x2 y2"> <img data-bbox="..."/> <div> formula content </div></div>
<div class="image_caption" data-bbox="..."> ... </div>
<div class="image_ocr" data-bbox="..."> ... </div>
<div class="music_sheet" format="abc_notation" data-bbox="..."> ... </div>
<div class="chemical_formula" format="smile" data-bbox="..."> ... </div>
</html></body>
```

**Key design elements of this format:**

The `data-bbox` attribute on every element provides the bounding box in absolute pixel coordinates (x1, y1, x2, y2), allowing the model to learn spatial layout. Elements are organized in reading order (left-to-right, top-to-bottom), teaching the model to reconstruct the document's logical flow, not just its visual appearance.

The format handles **seven distinct element types**, each with its own tag structure:
- **Paragraphs (`<p>`):** standard text blocks with bounding boxes.
- **Tables (`<table>`):** structured data with both bounding box and CSS-style formatting. The table content itself is HTML table markup (`<tr>`, `<td>`, etc.), meaning the model generates complete, machine-readable tables.
- **Charts (`<div class="chart">`):** contains an image reference and a `<table>` with the chart's underlying data. This teaches the model to perform chart de-rendering—extracting the numerical data from a visual chart representation.
- **Formulas (`<div class="formula">`):** contains an image and the LaTeX or similar formula content. The model learns to convert rendered equation images back to their source code.
- **Image captions (`<div class="image_caption">`):** images with descriptive captions, teaching the model to associate visual content with textual description.
- **Image OCR (`<div class="image_ocr">`):** images containing text, with the extracted text in a `<p>` tag. This is the standard OCR task but unified with the document structure.
- **Music sheets (`<div class="music_sheet" format="abc_notation">`):** sheet music images with ABC notation content (a text-based music representation format). This extends document understanding to a specialized domain.
- **Chemical formulas (`<div class="chemical_formula" format="smile">`):** chemical structure diagrams with SMILES string content (a text-based molecular representation). Another specialized domain, demonstrating the format's extensibility.

The format's design principle is to make the model's output directly machine-readable: the generated HTML can be parsed by standard tools to extract structured data (tables as CSV, formulas as LaTeX, chemical structures as SMILES) without post-processing. The inclusion of bounding box coordinates means the output also preserves spatial layout information for applications like document reconstruction or accessibility (e.g., screen readers that need to know element positions).

**Training data generation:** The paper notes that these document parsing examples were "synthesized," meaning they were generated programmatically rather than collected from human annotations. The likely pipeline involves: (1) collect diverse source documents (web pages, scientific papers, invoices, forms); (2) use existing OCR and layout analysis tools to extract text, tables, and bounding boxes; (3) convert the extracted information into the QwenVL HTML format; (4) pair the original document image with the generated HTML as the training target. This synthetic approach allows scaling to "a large corpus of document data" without expensive human annotation.

**OCR Data Construction.** The OCR training data combines three sources:

**Synthetic data:** A "visual text generation engine" produces realistic text images with controlled variation in fonts, sizes, colors, backgrounds, and distortions. This provides dense supervision for text recognition because every character position is known exactly. The paper does not specify the engine used, but common approaches include rendering text with random fonts onto random backgrounds from scene images.

**Open-source data:** Existing OCR datasets (the paper does not enumerate them, but common sources include ICDAR competition datasets, COCO-Text, TextOCR, HierText, and MLT for multilingual text).

**In-house collected data:** Proprietary datasets collected internally, presumably covering real-world scenarios relevant to Qwen's deployment contexts.

**Multilingual support** is explicitly targeted: the dataset covers French, German, Italian, Spanish, Portuguese, Arabic, Russian, Japanese, Korean, and Vietnamese, in addition to the primary Chinese and English data. The inclusion of Arabic and CJK languages is notable because these involve different text characteristics (right-to-left for Arabic, logographic characters for CJK) that require the model to learn distinct visual patterns and reading orders.

**Chart data:** 1 million samples were synthesized using matplotlib, seaborn, and plotly visualization libraries. The paper enumerates chart categories: bar charts, relational diagrams, and heatmaps. For each synthetic chart, the underlying data table is known (since the chart was generated programmatically), providing ground truth for the chart de-rendering task.

**Table data:** 6 million real-world samples were processed through "an offline end-to-end table recognition model." Low-confidence predictions, overlapping tables, and tables with insufficient cell density were filtered out. The filtering is important because table recognition errors (incorrect cell boundaries, merged cells) would teach the model to produce malformed tables.

**Video Data Construction.** Beyond dynamic FPS sampling (discussed above), the video training data includes:

**Long video captions:** For videos exceeding 30 minutes, "a set of long video captions [was constructed] by synthesizing multi-frame captions through a targeted synthesis pipeline." The pipeline likely works by: (1) sampling keyframes at regular intervals; (2) generating captions for each keyframe or short clip using an image-captioning model; (3) using an LLM to synthesize the per-frame captions into a coherent narrative covering the full video duration; (4) potentially refining with temporal alignment (ensuring the synthesized caption's events correspond to actual timestamps in the video).

**Video grounding data:** Timestamps are formatted in both "second-based formats" (e.g., `212.5`) and "hour-minute-second-frame (hmsf) formats" (e.g., `00:03:32:15`). Training on both formats ensures the model can parse and generate timestamps in whichever format the user provides. The inclusion of frames in the hmsf format (the `:15` at the end representing the 15th frame at the video's FPS) teaches the model that frame-level precision is sometimes meaningful.

**Agent Data Construction.** The agent training data teaches the model to perceive UI elements and execute actions:

**Perception data:** Screenshots are collected from mobile, web, and desktop platforms. A "synthetic data engine" generates two types of annotations:
- **Screenshot captions:** natural language descriptions of the UI state, teaching the model to understand what a screen shows (e.g., "A settings page with Wi-Fi toggled on and Bluetooth toggled off").
- **UI element grounding:** bounding boxes for clickable elements (buttons, text fields, dropdowns), teaching the model to locate interactive components.

**Decision-making data:** Multi-step action trajectories are collected from open-source data and synthesized by an agent framework (Wang et al., 2025, 2024b,c) on virtual environments. The operations are "unified across mobile, web, and desktop platforms into a function call format with a shared action space." This means the same action representation (e.g., `click(x=150, y=320)`, `type("hello")`, `scroll(direction="down")`) is used regardless of platform, enabling cross-platform generalization.

**Reasoning annotations:** A critical element is that each action step includes a reasoning trace explaining *why* the action was taken:

> "Given a ground-truth operation, we highlight it on the screenshot. Then, we provide the global query, along with screenshots from before and after this operation, to the annotators and require them to write reasoning content to explain the intention behind this operation."

This creates training data of the form: `[screenshot_before, instruction, action + reasoning]`. The reasoning content (generated by human and model annotators, following Xu et al., 2024) is then filtered by a model-based quality checker to remove low-quality explanations. The purpose is explicitly stated: "Such reasoning content prevents Qwen2.5-VL from overfitting to the ground-truth operations and makes it more robust in real-world scenarios." By learning the principles behind actions (not just action sequences), the model generalizes to novel UIs and tasks.

---

#### Three-Stage Pre-Training Recipe

The pre-training process is divided into three sequential phases, each with different data compositions, parameter update schedules, and sequence length budgets. Table 2 provides the quantitative breakdown:

| Stage | Data Composition | Tokens | Sequence Length | Parameters Trained |
|-------|-----------------|--------|-----------------|-------------------|
| 1: Visual Pre-Training | Image Caption, Knowledge, OCR, Pure text | 1.5T | 8192 | ViT only |
| 2: Multimodal Pre-Training | Interleaved Data, VQA, Video, Grounding, Agent, + Pure text | 2.0T | 8192 | ViT & LLM |
| 3: Long-Context Pre-Training | Long Video, Long Agent, Long Document | 0.6T | 32768 | ViT & LLM |

**Stage 1: Visual Pre-Training (1.5T tokens).** Only the Vision Transformer (ViT) parameters are updated; the LLM is frozen. The primary data consists of image captions, visual knowledge data (celebrity/landmark/flora/fauna identification), and OCR data. The purpose is to train the ViT to produce features that the frozen LLM can interpret—essentially aligning the ViT's representation space with the LLM's embedding space without modifying the LLM's language capabilities. The inclusion of pure text data during this stage (with the ViT presumably receiving no image, or a placeholder) helps maintain the LLM's text-only performance by preventing catastrophic forgetting during the alignment process.

The sequence length is 8,192 tokens, and data is "uniformly packed to a sequence length of 8,192" meaning multiple shorter examples are concatenated into a single training sequence. Dynamic packing balances the computational load across GPUs by ensuring all sequences are the same length, which avoids the inefficiency of padding short examples and the load imbalance of variable-length sequences.

**Stage 2: Multimodal Pre-Training (2.0T tokens).** All parameters (ViT and LLM) are unfrozen and trained jointly. The data composition broadens dramatically to include interleaved image-text data, visual question answering (VQA), multimodal mathematics, agent-based tasks, video understanding, and continued pure text data. This stage teaches the model to perform complex multimodal reasoning by training on diverse tasks simultaneously (multi-task learning). The pure text data continues throughout to prevent catastrophic forgetting of language capabilities.

The paper notes that the data includes "more intricate and reasoning-intensive datasets" compared to Stage 1, specifically calling out VQA, multimodal mathematics, and agent tasks as examples. The sequence length remains at 8,192.

**Stage 3: Long-Context Pre-Training (0.6T tokens).** The sequence length increases to 32,768 tokens, and the data shifts to long-form content: long videos (30+ minutes), long agent trajectories (multi-step interactions spanning many screenshots), and long documents (multi-page PDFs, lengthy articles). The purpose is to train the model to handle extended contexts by extending its effective attention span from 8K to 32K tokens. All parameters continue to be trained.

The total pre-training corpus is 1.5T + 2.0T + 0.6T = 4.1T tokens, as stated in Table 1. This represents a 3.4× increase over Qwen2-VL's 1.2T tokens. The paper does not specify the total training compute (GPU-hours), but the sequence length progression (8K → 8K → 32K) and the parameter training schedule (ViT-only → full → full) suggest a carefully staged approach to managing computational cost while building capability incrementally.

**Load balancing optimization.** The paper addresses a practical training challenge: "varying image sizes and text lengths, which can lead to imbalanced computational loads during training." Since the LLM dominates the computational cost (far more parameters than the ViT, especially for the 72B variant) and the ViT's cost has been reduced by window attention, the paper focuses on balancing the LLM's load:

> "we dynamically packed data samples based on their corresponding input sequence lengths to the LLM, ensuring consistent computational loads."

This means that within each GPU's batch, the total sequence length (visual tokens + text tokens) is kept approximately constant by combining examples of different lengths. A batch might contain one example with a large image (producing many visual tokens) and short text, alongside several examples with small images and long text—the total per-GPU token count is controlled.

---

#### Post-Training Alignment: SFT and DPO

The post-training process consists of two phases: Supervised Fine-Tuning (SFT) followed by Direct Preference Optimization (DPO). The Vision Transformer parameters are **frozen** during both phases—only the LLM and possibly the vision-language merger are updated. This is a standard practice in LVLM training: the visual representations are treated as fixed features, and alignment focuses on teaching the LLM to use these features appropriately for instruction following.

**SFT Data Composition.** The SFT dataset consists of approximately 2 million entries, evenly split by count (50% pure text, 50% multimodal including image-text and video-text). However, the paper notes that "multimodal entries consume significantly more tokens and computational resources during training due to the embedded visual and temporal information." This means that while the *example count* is balanced, the *token consumption* during training is heavily skewed toward multimodal examples, which is appropriate given that the model's primary purpose is multimodal understanding.

The data includes:
- **General VQA:** standard visual question answering across diverse topics
- **Image captioning:** generating descriptions of images
- **Mathematical problem-solving:** visual math problems requiring reasoning
- **Coding tasks:** programming questions, some with visual context (screenshots of code, diagrams)
- **Security-related queries:** the paper mentions this category without elaboration, likely covering content safety and refusal training
- **Document and OCR:** specialized data for the document parsing capabilities
- **Grounding:** referring expression comprehension, object detection, point grounding
- **Video analysis:** video question answering, temporal grounding, dense captioning
- **Agent interactions:** UI understanding, action prediction, multi-step task execution

The query sources include "open-source repositories, curated purchased datasets, and online query data." The online query data likely comes from Qwen's deployed chat services, capturing real user interaction patterns.

The data format uses **ChatML** (OpenAI's chat markup language), which structures conversations as sequences of role-tagged messages:

```
<|im_start|>user
[Image: <image>] What objects are in this image?
<|im_end|>
<|im_start|>assistant
There is a red car, a blue bicycle, and a street sign.
<|im_end|>
```

The ChatML format enables multi-turn dialogues with interleaved images and text, supporting realistic conversational patterns. The paper notes this format "deliberately diverg[es] from the pretraining data schema while maintaining architectural consistency with Qwen2-VL." This divergence is necessary because pre-training data is typically not in dialogue format, while SFT must teach the model to engage in turn-based conversations.

**Data Filtering Pipeline.** The quality of SFT data is controlled through a two-stage pipeline:

**Stage 1: Domain-Specific Categorization.** A specialized classifier (Qwen2-VL-Instag, derived from Qwen2-VL-72B) categorizes each question-answer pair into a hierarchical taxonomy:
- 8 primary domains (e.g., Coding, Planning, Document Processing)
- 30 fine-grained subcategories (e.g., Code_Debugging, Code_Generation, Code_Translation, Code_Understanding under the Coding domain)

This categorization enables "domain-aware and subdomain-aware filtering strategies, enabling the pipeline to optimize data-cleaning processes tailored to each category's specific characteristics." For example, code generation data might need different filtering criteria (checking compilation, checking for hallucinated APIs) than document parsing data (checking bounding box validity, checking table structure).

**Stage 2: Domain-Tailored Filtering.** Two complementary filtering approaches are applied:

*Rule-based filtering* uses predefined heuristics:
- Remove repetitive patterns (common in synthetic data where templates produce near-identical examples)
- Remove incomplete, truncated, or improperly formatted responses
- Remove queries and answers that are "unrelated or could potentially lead to harmful outputs"

*Model-based filtering* uses reward models trained on the Qwen2.5-VL series to score examples across multiple dimensions:
- **Query complexity and relevance:** is the question appropriately challenging and contextually meaningful?
- **Answer correctness:** does the answer factually address the query?
- **Completeness:** does the answer fully address all parts of the query?
- **Clarity:** is the answer well-structured and easy to understand?
- **Helpfulness:** does the answer provide useful information?
- **Visual grounding accuracy (for grounded tasks):** does the model correctly interpret and use visual information?

The multi-dimensional scoring produces a composite quality score, and only high-scoring examples are retained. The paper notes that the reward models are "trained on the Qwen2.5-VL series," suggesting an iterative process where the model being trained is used to evaluate and filter its own training data in subsequent rounds.

**Rejection Sampling for Enhanced Reasoning.** For tasks requiring complex reasoning (mathematical problem-solving, code generation, domain-specific VQA), the paper employs rejection sampling to improve data quality:

1. Start with datasets that have ground truth annotations (correct answers are known).
2. Generate responses using an intermediate version of Qwen2.5-VL (not the final model—this prevents data leakage where the model is evaluated on responses it generated itself).
3. Compare generated responses to ground truth answers. Retain only examples where the model's answer matches the ground truth.

This produces a dataset of **verified correct reasoning traces**: each example contains the problem, the model's generated reasoning (which may include Chain-of-Thought steps), and the correct answer—all verified against ground truth.

**Additional quality constraints** are applied to the reasoning traces:
- Filter out responses exhibiting **code-switching** (unexplained language switching, e.g., a Chinese question with an English reasoning trace, unless the task explicitly requires it). This ensures linguistic consistency.
- Filter out responses with **excessive length** (overly verbose reasoning that doesn't add information).
- Filter out responses with **repetitive patterns** (the model getting stuck in a loop, repeating the same reasoning step).

**The vision-language alignment challenge in CoT.** The paper explicitly identifies a failure mode specific to CoT reasoning in vision-language models:

> "Intermediate reasoning steps may fail to adequately integrate visual information, either by ignoring relevant visual cues or misinterpreting them."

For example, a model solving a geometry problem might generate reasoning that describes geometric relationships in the text of the problem while ignoring the diagram showing that the triangle is actually right-angled. The paper applies "rule-based and model-driven filtering strategies to validate the accuracy of intermediate reasoning steps" but acknowledges this remains "an ongoing challenge that requires further advancements."

**DPO Phase.** After SFT, Direct Preference Optimization (Rafailov et al., 2023) is applied exclusively on image-text and pure text data (no video or agent data in this phase). DPO optimizes the model to prefer chosen responses over rejected responses using pairs of (prompt, chosen_response, rejected_response). The paper states that "each sample [is] processed only once to ensure efficient optimization," meaning the DPO data is used for a single epoch—likely to prevent overfitting to the preference data.

The ViT remains frozen during DPO, consistent with the SFT phase. The paper does not provide the DPO-specific hyperparameters (learning rate, `$\beta$` parameter controlling divergence from the reference model), but the standard DPO objective is:

$$\mathcal{L}_{\text{DPO}} = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right) \right]$$

where `$x$` is the prompt, `$y_w$` is the preferred (winning) response, `$y_l$` is the rejected (losing) response, `$\pi_\theta$` is the model being trained, `$\pi_{\text{ref}}$` is the reference model (the SFT checkpoint), `$\sigma$` is the logistic sigmoid, and `$\beta$` controls how much the model can deviate from the reference.

**What it computes:** the DPO loss increases the log-probability ratio of the preferred response relative to the reference model while decreasing the ratio for the rejected response. The sigmoid saturates when the preference margin (the difference in ratios) is large, preventing the model from over-optimizing on easy preference pairs.

**Why this form:** DPO directly optimizes the policy (the model's response distribution) against a preference dataset without needing to train a separate reward model, unlike RLHF which requires reward model training, RL fine-tuning, and KL penalty scheduling. The implicit reward is `$\beta \log \frac{\pi_\theta}{\pi_{\text{ref}}}$`, which means the model is incentivized to increase the probability of preferred responses while staying close (in KL divergence) to the reference distribution. The paper's choice to apply DPO to only image-text and pure text data (excluding video and agent trajectories) suggests that preference data was more readily available or more reliable for these modalities.

---

#### Model Configuration and Scaling Variants

Table 1 provides the complete configuration of all three model sizes. The key scaling pattern is: the ViT is identical across all sizes; the merger output dimension and LLM dimensions scale; the LLM depth (number of layers) scales.

**Vision Transformer (shared across all sizes):**
- Hidden size: 1280
- Layers: 32
- Attention heads: 16
- Intermediate size (FFN): 3456 (approximately 2.7× hidden size, consistent with SwiGLU which has an extra linear projection)
- Patch size: 14
- Window size: 112 (8×8 patches)
- Full attention at layers: {7, 15, 23, 31}

The hidden size of 1280 is moderate (ViT-Large from the original ViT paper is 1024; ViT-Huge is 1280), reflecting the fact that the ViT is only one component of a much larger system. The window size of 112 = 8×14 means each window contains at most 64 patches (8×8 grid).

**Vision-Language Merger (scales with LLM dimension):**
- Input channels: 1280 (from ViT) for all sizes
- Output channels: 2048 (3B), 3584 (7B), 8192 (72B) — matches the LLM hidden size

The merger is a simple two-layer MLP with these input/output dimensions. The hidden layer dimension is not specified, but the compression from 5120 (4 concatenated ViT features) to 1280 (ViT hidden size) and then expansion to the LLM dimension suggests a bottleneck-then-expand structure: 5120 → 1280 → `$d_{LLM}$`.

**Large Language Model (scales primarily in depth):**
- **Qwen2.5-VL-3B:** hidden size 2048, 36 layers, 2 KV heads, FFN intermediate 4864, embedding tying enabled (input and output embeddings share parameters; saves parameters at small scale)
- **Qwen2.5-VL-7B:** hidden size 3584, 28 layers, 4 KV heads, FFN intermediate 18944, embedding tying disabled
- **Qwen2.5-VL-72B:** hidden size 8192, 80 layers, 8 KV heads, FFN intermediate 29568, embedding tying disabled

The LLM configurations correspond to the Qwen2.5 LLM family: Qwen2.5-3B (hidden 2048, 36 layers), Qwen2.5-7B (hidden 3584, 28 layers), and Qwen2.5-72B (hidden 8192, 80 layers). Note the non-monotonic layer count: the 7B model has fewer (but wider) layers than the 3B model, representing a different depth-vs-width tradeoff in the Qwen2.5 design.

**KV heads (Grouped Query Attention):** The number of key-value heads (2, 4, 8) indicates Grouped Query Attention is used, where multiple query heads share the same key-value head, reducing memory for the KV cache during inference. For the 3B model, 2 KV heads serve 16 query heads (not explicitly stated, but the total head count can be inferred: head size 128, hidden size 2048 → 16 query heads, 2 KV heads → 8 query heads per KV head). For the 7B model: hidden 3584, head size 128 → 28 query heads, 4 KV heads → 7 query heads per KV head. For the 72B model: hidden 8192, head size 128 → 64 query heads, 8 KV heads → 8 query heads per KV head.

**Vocabulary size:** 151,646 tokens, shared across all sizes. This is consistent with the Qwen2.5 tokenizer.

**Total trained tokens:** 4.1T for all sizes, as shown in Table 1. This is the pre-training volume only; SFT and DPO tokens are additional but presumably much smaller (the 2 million SFT entries at an average of perhaps 1K tokens each would total ~2B tokens).

**Embedding tying (3B only):** The 3B model ties the input embedding and output projection weights (the same matrix is used for token embedding lookup and for the final linear projection to vocabulary logits). This saves `$151646 \times 2048 \approx 311$` million parameters, significant at the 3B scale. The 7B and 72B models do not use embedding tying, trading parameter efficiency for representational flexibility.

**Why three sizes?** The paper explicitly states the models address "diverse use cases from edge AI to high-performance computing." The 3B model is suitable for on-device deployment (phones, laptops) where memory and compute are constrained. The 7B model targets mid-range GPU deployment. The 72B model is the flagship for server-class inference. All three share the same ViT, meaning visual processing quality is independent of model size—the difference lies in the LLM's reasoning capacity and the merger's representational capacity.

## 4. Key Insights and Innovations

### Innovation 1: Native Spatial and Temporal Fidelity as a Unified Design Principle

The field's dominant approach to handling variable-resolution multimodal inputs has been **coordinate normalization and temporal subsampling**—resize images to fixed dimensions before encoding, squashing all spatial relationships into a normalized [0,1] range, and sample video frames at a fixed rate regardless of content tempo. Models from CLIP-based VLMs to GPT-4o and InternVL2.5 have all converged on this paradigm because it simplifies engineering: fixed-size inputs mean fixed compute graphs, and normalized coordinates mean the model doesn't need to learn scale-dependent relationships.

Qwen2.5-VL makes the opposite choice across **both** spatial and temporal dimensions simultaneously, and this is what makes the paper conceptually distinctive. The model processes images at their native resolution with absolute pixel coordinates for grounding, and it processes video with frame timestamps encoded as real-time intervals rather than sequential indices. These are not independent improvements—they are manifestations of a single design philosophy: **preserve the physical quantities that matter for real-world interaction** (pixel distances for clicking, seconds for temporal localization) **throughout the entire processing pipeline, and let the model learn to reason about them natively.**

This matters because it reframes the problem. Rather than treating variable-resolution inputs as an inconvenience to be normalized away, the paper treats resolution and frame rate as **informative signals** that the model should exploit. An image that is 4000 pixels wide contains objects at a different absolute scale than one that is 800 pixels wide; a video segment where frames are 33ms apart contains different motion information than one where frames are 2 seconds apart. By preserving these quantities, the model can learn that "a box of width 200 pixels" means something different in the two images, and that "an event spanning 3 temporal ID units" has different real-world duration depending on the sampling rate.

The evidence for this philosophy's effectiveness is distributed across multiple benchmarks: the 93.6% on CountBench (Table 7), which requires precise spatial reasoning about absolute object positions and sizes; the 50.9 mIoU on Charades-STA temporal grounding (Table 8), which requires mapping temporal attention patterns to real timestamps; and the 87.1% on ScreenSpot and 43.6% on ScreenSpot Pro (Table 9), where clicking the right pixel matters absolutely. Critically, Qwen2.5-VL-72B achieves 35% on AndroidWorld without Set-of-Mark prompting—compared to GPT-4o's 34.5% *with* SoM—suggesting the absolute coordinate understanding translates to real interactive capability that auxiliary marking cannot fully substitute for.

This is a **fundamental reframing** rather than an incremental improvement. Prior work (Li et al., 2023c; Ye et al., 2023) has explored dynamic resolution for images, and prior work (Qwen2-VL) introduced MRoPE for multimodal position encoding. But no prior system has committed to native fidelity as a unified spatial-temporal design principle and demonstrated that it yields practical advantages across grounding, document parsing, video temporal localization, and GUI agent tasks simultaneously. The paper's contribution is not the individual mechanisms (window attention for efficiency, absolute time alignment, absolute coordinate training), but the **architectural thesis that these mechanisms should be deployed together because they serve a common purpose**: making the model's internal representations correspond as directly as possible to the physical world's spatial and temporal structure.

---

### Innovation 2: The QwenVL HTML Format as a Scalable Structured Output Curriculum

Prior approaches to document understanding treat parsing as an extraction problem: given an image of a document, use specialist models or pipelines to identify text regions, recognize characters, detect tables, and interpret charts, then assemble these into a structured representation through post-processing. LVLMs have recently been trained to perform some of these subtasks end-to-end, but typically in a fragmented way—one model for DocVQA, another for ChartQA, another for table extraction—without a unified output representation.

The paper's innovation is to define a **single, extensible HTML-based output format** that encodes layout, content, and element type for every component of a document simultaneously, and then train the model to generate this format directly as autoregressive text. The QwenVL HTML format (Section 2.2.1) is not merely a serialization choice—it is a **curriculum design** that teaches the model a unified document representation where spatial layout (bounding boxes in `data-bbox` attributes), content semantics (paragraph text, table data, formula code), and element categorization (seven distinct element types with different tag structures) are learned jointly rather than through separate heads or post-processing.

What makes this distinctive at the idea level is its **scalability implications**. By expressing document parsing as a text generation problem (the model outputs HTML tokens), the same autoregressive training objective used for all other tasks applies without architectural modification. There is no need for task-specific decoders, separate bounding box regression heads, or structured prediction losses. The HTML format itself serves as the "task specification": the model learns that `<div class="chart">` means "you are now de-rendering a chart," that `<table>` means "you are now extracting tabular data," and that `data-bbox="x1 y1 x2 y2"` means "you must output the spatial extent of this element in absolute coordinates." All of this is learned from examples without explicit task definitions.

This stands in contrast to prior work on document parsing with LVLMs, which typically treats layout analysis, text recognition, table extraction, and chart interpretation as separate capabilities requiring distinct training data, loss functions, and sometimes model architectures. The paper's unified approach means that improvements in the base model's language generation capabilities (from scaling the LLM, better pre-training data, or post-training alignment) automatically improve all document parsing subtasks simultaneously. The document understanding benchmarks (Table 5) demonstrate the result: Qwen2.5-VL-72B achieves 96.4% on DocVQA, 89.5% on ChartQA, 87.3% on InfoVQA, and 61.5/63.7% on OCRBench_v2 (English/Chinese)—substantially exceeding prior open-source state-of-the-art and matching or exceeding GPT-4o and Claude 3.5 Sonnet on most metrics. The 3B and 7B variants also show strong performance (93.9% and 95.7% on DocVQA respectively), suggesting the format-based training transfers effectively to smaller models.

The HTML format is also **incrementally extensible** in a way that task-specific architectures are not. The paper demonstrates seven element types; adding an eighth (e.g., `<div class="engineering_drawing" format="step">`) requires only defining the new tag structure and generating training examples, not modifying the model architecture or training procedure. This is a **conceptual advance**: the paper shows that structured output tasks traditionally requiring specialized architectures can be absorbed into the text generation paradigm through careful format design.

---

### Innovation 3: Identifying and Addressing the Frame-Index vs. Absolute Time Representational Gap

The distinction between encoding temporal position as frame index (what frame number is this in the sequence?) versus absolute time (what timestamp does this frame correspond to?) is subtle enough that most video-language models have not treated it as a design choice worth optimizing. Qwen2-VL itself used frame-index-based MRoPE, and the paper's explicit critique of that choice—that it "did not account for the speed of content changes or the absolute timing of events within the video"—represents a **diagnostic insight**: identifying a specific representational inadequacy and proposing a targeted fix.

The conceptual contribution here is not the mechanism (aligning MRoPE temporal IDs to timestamps) but the **recognition that temporal position encoding must be invariant to sampling rate variations** to support robust temporal reasoning. In the frame-index paradigm, a model trained on 1 FPS video learns that "adjacent frames are 1 second apart"; the same model tested on 30 FPS video would incorrectly interpret adjacent frames as also being 1 second apart (rather than 33ms). The model cannot distinguish "fast motion captured densely" from "slow motion captured sparsely" because the position encoding collapses this distinction.

The absolute time alignment solves this by making the temporal position ID a function of real time, not frame count. The consequence is that the same video processed at different FPS produces different frame counts but the same temporal relationship between corresponding content: an event at timestamp 10.0 seconds always maps to the same temporal ID regardless of whether frames were sampled at 1 FPS or 30 FPS around it. The intervals between consecutive temporal IDs encode the actual time elapsed, so attention patterns that span "0.5 seconds" versus "5.0 seconds" are distinguished by the magnitude of the temporal position difference.

This is significant beyond the raw performance gains on temporal grounding (50.9 mIoU on Charades-STA vs. GPT-4o's 35.7; Table 8) because it reveals a **principled design constraint** for multimodal position encoding: when a model is expected to output temporally-grounded predictions (timestamps), its internal temporal representation must be grounded in the same coordinate system. Frame-index encoding requires the model to learn an implicit mapping from frame indices to timestamps, which is unreliable when FPS varies. Absolute time encoding eliminates this mapping problem by construction.

The approach is also notable for its **minimality**—it modifies only the ID assignment function, requiring no additional parameters, no auxiliary loss, and no architecture changes. This is a case where the correct representational choice eliminates complexity rather than adding it. The contrast with approaches that add separate temporal embedding modules or train temporal grounding heads (which would require additional losses and hyperparameters) highlights the elegance of the solution. The paper frames this explicitly: "without necessitating any additional computational overhead."

This insight is not incremental in the sense that it adds a new capability on top of existing ones; rather, it **fixes a representational flaw** that limited prior models' temporal reasoning. The 73.3% on Video-MME (without subtitles) and 74.8% on TempCompass (Table 8)—competitive with GPT-4o and Gemini 1.5 Pro—suggest the fix is practically effective. More importantly, the diagnostic framework (identifying that frame-index encoding conflates sampling rate with content tempo, and that this limits temporal reasoning) provides a conceptual tool for evaluating future video-language architectures.

---

### Innovation 4: Verifying That Fine-Grained Perception Transfers to Small Scales

A persistent belief in the LVLM literature is that fine-grained visual capabilities—precise object localization, structured document parsing, temporal grounding—require large models to achieve competitive performance. The intuition is that these tasks demand both high-resolution visual processing (which is parameter-intensive) and sophisticated reasoning (which requires large LLMs). Prior open-source models have largely validated this view: the gap between flagship models and their smaller variants on perception-heavy tasks is typically large, and models below ~13B parameters rarely compete with proprietary systems on grounding or document understanding.

Qwen2.5-VL provides a **systematic counterexample**. The 7B model achieves 84.9% on TextVQA, 95.7% on DocVQA, 82.6% on InfoVQA, and 87.3% on ChartQA (Table 5)—often exceeding InternVL2.5-78B, a model with more than 10× the parameters. The 3B model achieves 79.3% on TextVQA, 93.9% on DocVQA, 77.1% on InfoVQA, and 84.0% on ChartQA—again competitive with or exceeding much larger models including Qwen2-VL-72B on some metrics. On grounding (Table 6), the 7B model achieves 90.0% on RefCOCO val and 37.3% mAP on ODinW-13; the 3B model achieves 89.1% and 37.5% respectively. On video understanding (Table 8), both smaller models show strong performance: Qwen2.5-VL-7B achieves 71.6% on Video-MME (with subtitles) and 45.3% on LVBench.

What makes this intellectually significant is not the raw numbers but what they imply about **the relationship between model scale and perceptual capability**. The paper's architecture shares the same ViT (1280 hidden, 32 layers) across all three model sizes. The only differences are the LLM dimensions and the merger's output dimension. The fact that the 3B model achieves comparable visual understanding to the 72B model on many benchmarks suggests that **the ViT, not the LLM, is the primary determinant of fine-grained perception quality**—once the ViT can extract precise spatial features (enabled by native resolution processing and window attention) and the training data teaches absolute coordinate understanding (via the grounding data and HTML format), the LLM's role is primarily to interpret and reason about already-well-represented visual information. The LLM's size matters more for complex multimodal reasoning (where the 72B does show larger gaps, e.g., 70.2 vs. 53.1 on MMMU) than for perceptual accuracy per se.

This finding has practical significance for deployment: it means that on-device models (3B parameters) can plausibly handle document scanning, UI grounding, and basic video understanding without cloud offloading. But it also has **theoretical significance**: it suggests a decoupling between perceptual fidelity (governed by the vision encoder and training data) and reasoning depth (governed by LLM scale) that is not obvious a priori. If this decoupling generalizes beyond Qwen2.5-VL, it implies that future work on improving LVLM perception should focus on vision encoder architecture and spatial training objectives rather than simply scaling the LLM, and that perceptual capabilities can be improved independently of reasoning capabilities.

The evidence is not perfectly clean—the 72B model does outperform the smaller variants on most benchmarks, and the gaps are larger on reasoning-heavy tasks—but the fact that a 3B model can achieve 93.9% on DocVQA and 89.1% on RefCOCO val is a genuine empirical surprise that challenges the scaling assumption. The paper does not explicitly theorize this decoupling, but the data supports it as an emergent finding.

## 5. Experimental Analysis

### Evaluation Methodology

- **Datasets.** The paper evaluates on a broad collection of published benchmarks spanning college-level problems (MMMU, MMMU-Pro), mathematics (MathVista, MATH-Vision, MathVerse), general VQA (MMBench series, MMStar, MME, MuirBench, BLINK, CRPE, HallBench, MTVQA, RealWorldQA, MME-RealWorld, MMVet, MM-MT-Bench, MegaBench), document understanding and OCR (AI2D, TextVQA, DocVQA, InfoVQA, ChartQA, CharXiv, SEED-Bench-2-Plus, OCRBench, OCRBench_v2, CC-OCR, OmniDocBench, VCR), spatial understanding and grounding (RefCOCO/+/g, ODinW-13, PointGrounding, CountBench), video understanding and grounding (Video-MME, Video-MMMU, MMVU, MVBench, MMBench-Video, LongVideoBench, LVBench, EgoSchema, PerceptionTest, MLVU, TempCompass, Charades-STA), GUI agent benchmarks (ScreenSpot, ScreenSpot Pro, Android Control, AndroidWorld, MobileMiniWob++, OSWorld), and pure text benchmarks (MMLU-Pro, MMLU-redux, LiveBench, GPQA, MATH, GSM8K, HumanEval, MultiPL-E, IFEval). All are standard published benchmarks; test splits or validation splits are used as specified by each benchmark's evaluation protocol. No custom or proprietary evaluation datasets are introduced.

- **Base model(s).** The primary evaluated models are Qwen2.5-VL at three scales: 3B, 7B, and 72B parameters. All three share an identical Vision Transformer architecture (hidden size 1280, 32 layers), trained from scratch, with the Qwen2.5 LLM backbone (36 layers for 3B, 28 layers for 7B, 80 layers for 72B) initialized from pre-trained weights. The 72B model is the flagship used for state-of-the-art comparisons; the 3B and 7B variants are evaluated to demonstrate capability preservation at smaller scales. The paper argues the model family is "representative of the capabilities of many contemporary LLMs" and sits in a regime where test-time compute and scaling analysis are meaningful.

- **Metrics.** Metric choice is benchmark-dependent and follows each benchmark's standard evaluation protocol. For college-level problems and math tasks: accuracy (exact match or equivalent scoring). For general VQA: accuracy (MMBench series, MMStar, MuirBench, BLINK, CRPE, HallBench, RealWorldQA), MME score (sum of perception and cognition scores), accuracy (MTVQA), LLM-based evaluation (MMVet, MM-MT-Bench). For document understanding: accuracy (AI2D, TextVQA, DocVQA, InfoVQA, ChartQA, CharXiv, SEED-Bench-2-Plus, VCR), OCRBench score (cumulative across subtasks), OCRBench_v2 score (separate English and Chinese tracks), CC-OCR score, OmniDocBench edit distance (↓, lower is better). For grounding: accuracy at Intersection-over-Union threshold (RefCOCO/+/g, typically IoU ≥ 0.5), mean Average Precision (ODinW-13), custom accuracy for point grounding. For counting: accuracy (CountBench). For video understanding: accuracy (Video-MME, Video-MMMU, MMVU, MVBench, LongVideoBench, LVBench, EgoSchema, PerceptionTest, MLVU, TempCompass), score (MMBench-Video, on a 0–6 scale). For video grounding: mean Intersection-over-Union (Charades-STA, mIoU). For GUI agents: accuracy (ScreenSpot, ScreenSpot Pro, Android Control), success rate (AndroidWorld, MobileMiniWob++, OSWorld). For pure text tasks: accuracy (MMLU-Pro, MMLU-redux, GPQA, MATH, GSM8K, HumanEval, MultiPL-E), LiveBench score, IFEval instruction-following accuracy.

- **Baselines.** The paper compares against commercial closed-source models: **GPT-4o-0513** (OpenAI, 2024), **Claude 3.5 Sonnet-0620** (Anthropic, 2024a), **Gemini 1.5 Pro** (Team et al., 2023), and **Gemini 2.0** (Deepmind, 2024). For open-source comparisons: **InternVL2.5-78B** (Chen et al., 2024d) is the primary open-source state-of-the-art baseline, **Qwen2-VL-72B** (Wang et al., 2024e) is the direct predecessor, and **Molmo-72B** (Deitke et al., 2024) appears in grounding and counting comparisons. Additional baselines for specific tasks include **Grounding DINO** (Liu et al., 2023c) for object detection, **Aguvis-72B** (Xu et al., 2024) for GUI agents, and **Llama-3.1-70B/405B**, **Qwen2-72B**, **Qwen2.5-72B** for pure text tasks. For document parsing, the **OmniDocBench** baseline numbers include unnamed prior models. OCRBench_v2 baselines include GPT-4o, Gemini 1.5 Pro, and Claude 3.5 Sonnet in addition to InternVL2.5-78B.

- **Generation budget / compute accounting.** The paper does not use a unified FLOPs-based compute accounting for fair comparison across methods—this is not a test-time compute scaling paper. Instead, evaluation follows each benchmark's standard protocol, with models run at default inference settings (greedy decoding or similar, though the paper does not consistently specify decoding parameters). For video benchmarks, the paper imposes uniform resource limits: "we capped the maximum number of frames analyzed per video at 768, with the total number of video tokens not exceeding 24,576" (Section 3.3.4). This provides a consistent compute budget across the Qwen2.5-VL family and implicitly defines a generation budget for video tasks. For agent benchmarks, some baselines (GPT-4o, Gemini 2.0 on AndroidWorld) use Set-of-Mark (SoM) prompting; Qwen2.5-VL-72B is evaluated both with and without SoM, but this distinction is noted rather than used for compute normalization.

- **Cross-validation / statistical protocol.** No formal cross-validation or statistical significance testing is reported. The paper uses standard benchmark evaluation protocols, which typically use fixed test splits. No confidence intervals, error bars, or significance tests appear anywhere in the experimental tables. Results are reported as point estimates (single numbers per benchmark). For multi-turn agent evaluation, results are aggregated over test episodes, but no variance estimates are provided. This is a limitation: on benchmarks with small test sets (e.g., 500 questions for many reasoning benchmarks, 100–500 images for grounding benchmarks), the reported point estimates could be sensitive to test set composition, and without variance information it is impossible to assess whether small differences between models (1–2 percentage points) are statistically meaningful. The paper's "SOTA" claims (e.g., "surpassing the previous open-source state-of-the-art") should be interpreted cautiously given the absence of statistical rigor.

### Main Quantitative Results

#### Overall State-of-the-Art Comparison

Table 3 presents the headline comparison across college-level problems, math, and general VQA benchmarks. Qwen2.5-VL-72B achieves 70.2 on MMMU (vs. 70.1 for InternVL2.5-78B and 69.1 for GPT-4o), 51.1 on MMMU-Pro (matching GPT-4o's 51.9 and exceeding InternVL2.5's 48.6), 74.8 on MathVista (exceeding InternVL2.5's 72.3 and GPT-4o's 63.8), 38.1 on MATH-Vision (exceeding InternVL2.5's 32.2), and 57.6 on MathVerse (exceeding InternVL2.5's 51.7 and GPT-4o's 50.2). On general VQA, the 72B model achieves 88.4 on MMBench-V1.1-EN (exceeding InternVL2.5's 87.4 and GPT-4o's 83.1), 70.8 on MMStar (exceeding InternVL2.5's 69.5), 2448 on MMEsum (vs. InternVL2.5's 2494—one of the few metrics where Qwen2.5-VL-72B does not hold the SOTA), 70.7 on MuirBench, 64.4 on BLINK, 79.2 on CRPE, 31.7 on MTVQA, 63.2 on MME-RealWorld, 76.2 on MMVet, and 7.6 on MM-MT-Bench. On MegaBench, Qwen2.5-VL-72B achieves 51.3, behind Claude 3.5 Sonnet's 52.1 and GPT-4o's 54.2.

The smaller variants show competitive performance: Qwen2.5-VL-7B achieves 58.6 on MMMU, 68.2 on MathVista, 83.5 on MMBench-ENtest, 63.9 on MMStar, and 70.4 on SEED-Bench-2-Plus—outperforming Qwen2-VL-72B on several metrics despite having roughly 1/10 the parameters. Qwen2.5-VL-3B achieves 53.1 on MMMU, 62.3 on MathVista, 79.1 on MMBench-ENtest, and 55.9 on MMStar—outperforming comparably-sized models and many larger ones.

**Critical reading of Table 3:** The comparison is not fully apples-to-apples. GPT-4o and Claude 3.5 Sonnet numbers are drawn from different sources and may use different evaluation protocols or prompt formats. The paper does not re-evaluate all baselines under identical conditions. The InternVL2.5 numbers come from that model's paper (Chen et al., 2024d), which may use different decoding strategies. Furthermore, several key baselines are missing: Qwen2.5-VL-72B is not compared against Gemini 1.5 Pro on most general VQA benchmarks (no MMBench, MMStar, MME numbers for Gemini appear in Table 3), and Claude 3.5 Sonnet numbers are absent from half of the benchmarks. The table's structure (showing "previous open-source SoTA" as a single column, usually InternVL2.5-78B, alongside proprietary models) obscures the exact comparison landscape.

#### Document Understanding and OCR

Table 5 presents the document understanding results, where Qwen2.5-VL shows its strongest relative advantage. On **OCR-related parsing tasks:** Qwen2.5-VL-72B achieves 79.8 on CC-OCR (exceeding Gemini 1.5 Pro's 73.0 and GPT-4o's 66.9 by large margins) and 0.226/0.324 edit distance on OmniDocBench (English/Chinese, where lower is better—exceeding all baselines including GPT-4o's 0.265/0.435). On **OCR-related understanding tasks:** the 72B model achieves 88.7 on AI2D (vs. InternVL2.5's 89.1 and Gemini 1.5 Pro's 88.4), 83.5 on TextVQA (exceeding InternVL2.5's 83.4 and GPT-4o's 77.4), 96.4 on DocVQA (exceeding Claude 3.5 Sonnet's 95.2), 87.3 on InfoVQA (exceeding InternVL2.5's 84.1 and Gemini 1.5 Pro's 81.0), 89.5 on ChartQA (exceeding InternVL2.5's 88.3 and Claude 3.5 Sonnet's 90.8—one of the rare cases where a proprietary model leads), 49.7/87.4 on CharXiv (RQ/DQ, exceeding InternVL2.5's 42.4/82.3), 73.0 on SEED-Bench-2-Plus (exceeding GPT-4o's 72.0), 885 on OCRBench (exceeding InternVL2.5's 854 and GPT-4o's 736—by 149 points), and 79.8 on VCR (exceeding GPT-4o's 73.2). On **OCR-related comprehensive tasks:** the 72B model achieves 61.5/63.7 on OCRBench_v2 (English/Chinese), exceeding Gemini 1.5 Pro's 51.9/43.1 by 9.6 and 20.6 percentage points respectively—a genuinely large margin that the paper emphasizes.

The smaller variants maintain strong document capabilities: Qwen2.5-VL-7B achieves 84.9 on TextVQA (exceeding InternVL2.5-78B's 83.4), 95.7 on DocVQA, 82.6 on InfoVQA, 87.3 on ChartQA, 864 on OCRBench, 80.5 on VCR, and 56.3/57.2 on OCRBench_v2. Qwen2.5-VL-3B achieves 79.3 on TextVQA, 93.9 on DocVQA, 77.1 on InfoVQA, 84.0 on ChartQA, 797 on OCRBench, and 37.5 on VCR (a notable drop-off on this particular benchmark compared to the 7B model's 80.5, though the paper does not discuss this gap). The 7B model's TextVQA score of 84.9 actually exceeds the 72B model's 83.5—an inversion that could reflect test set noise or an actual capability difference, though the paper does not comment on it.

**Missing comparisons:** Gemini 1.5 Pro numbers are absent from several OCR-related understanding benchmarks (TextVQA, DocVQA, InfoVQA, ChartQA, CharXiv, SEED-Bench-2-Plus, OCRBench) where they would be relevant. The paper attributes some baseline numbers to "Chen et al. (2024d)" (the InternVL2.5 paper) rather than running fresh evaluations, introducing potential protocol inconsistencies.

#### Spatial Understanding and Grounding

Table 6 presents grounding results. Qwen2.5-VL-72B achieves 92.7 on RefCOCO val, 94.6 on RefCOCO testA, 89.7 on RefCOCO testB, 88.9 on RefCOCO+ val, 92.2 on RefCOCO+ testA, 83.7 on RefCOCO+ testB, 89.9 on RefCOCOg val, 90.3 on RefCOCOg test, 43.1 mAP on ODinW-13, and 67.5 on PointGrounding. These numbers are competitive with InternVL2.5-78B (93.7 on RefCOCO val, 95.6 testA, 92.5 testB, 90.4/94.7/86.9 on RefCOCO+, 92.7/92.2 on RefCOCOg, 31.7 on ODinW—though PointGrounding numbers are absent for InternVL2.5) and substantially exceed Gemini 1.5 Pro (which ranges from 62.5–76.2 on the RefCOCO benchmarks and 36.7 on ODinW). The ODinW-13 result of 43.1 mAP is notable because it represents open-vocabulary detection in the wild, bridging generalist LVLMs with specialist object detectors; Grounding DINO achieves 55.0, indicating room for improvement but showing that Qwen2.5-VL-72B is competitive without task-specific architecture.

The smaller variants are again strong: Qwen2.5-VL-7B achieves 90.0 on RefCOCO val, 92.5 testA, 85.4 testB, 84.2/89.1/76.9 on RefCOCO+, 87.2/87.2 on RefCOCOg, 37.3 on ODinW, and 67.3 on PointGrounding. Qwen2.5-VL-3B achieves 89.1/91.7/84.0 on RefCOCO, 82.4/88.0/74.1 on RefCOCO+, 85.2/85.7 on RefCOCOg, 37.5 on ODinW, and 58.3 on PointGrounding. The ODinW results are particularly interesting: the 3B and 7B models achieve essentially identical mAP (37.5 vs. 37.3) while the 72B model achieves 43.1—suggesting that open-vocabulary detection in diverse real-world scenes benefits substantially from the larger LLM's reasoning capacity, unlike the RefCOCO benchmarks where all sizes cluster closely.

Table 7 presents counting results on CountBench. Qwen2.5-VL-72B achieves 93.6, exceeding Molmo-72B's 91.2, Claude 3.5 Sonnet's 89.7, GPT-4o's 87.9, Gemini 1.5 Pro's 85.5, and dramatically exceeding InternVL2.5-78B's 72.1. The paper notes this uses a "detect then count"-style prompt, suggesting that the model benefits from explicit grounding-then-counting instructions rather than being prompted to count directly.

**Missing comparisons:** Molmo-72B numbers are only available for CountBench and two PointGrounding entries. GPT-4o and Claude 3.5 Sonnet grounding numbers are absent from most benchmarks. The PointGrounding benchmark is described as "self-curated," meaning no external baseline comparison exists—only Qwen2.5-VL and Molmo numbers are reported. The lack of a standardized point grounding benchmark weakens this evaluation.

#### Video Understanding and Grounding

Table 8 presents video understanding results. Qwen2.5-VL-72B achieves 73.3 on Video-MME without subtitles (between GPT-4o's 71.9 and Gemini 1.5 Pro's 75.0), 79.1 with subtitles (between GPT-4o's 77.2 and Gemini's 81.3), 60.2 on Video-MMMU (between GPT-4o's 61.2 and Gemini's 53.9), 62.9 on MMVU (below GPT-4o's 67.4), 70.4 on MVBench (exceeding GPT-4o's 64.6 and Gemini's 60.5), 2.02 on MMBench-Video (exceeding GPT-4o's 1.63 and Gemini's 1.30), 60.7 on LongVideoBench (below GPT-4o's 66.7), 47.3 on LVBench (substantially exceeding GPT-4o's 30.8 and Gemini's 33.1), 76.2 on EgoSchema (exceeding GPT-4o's 72.2 and Gemini's 71.2), 73.2 on PerceptionTest, 74.6 on MLVU (exceeding GPT-4o's 64.6), and 74.8 on TempCompass (exceeding GPT-4o's 73.8 and Gemini's 67.1).

On video grounding (Charades-STA), Qwen2.5-VL-72B achieves 50.9 mIoU, dramatically exceeding GPT-4o's 35.7—a 15.2 point absolute improvement. The paper attributes this to the absolute time MRoPE alignment enabling precise temporal localization.

The smaller variants show strong video performance: Qwen2.5-VL-7B achieves 65.1/71.6 on Video-MME (w/o and w/ subtitles), 47.4 on Video-MMMU, 50.1 on MMVU, 69.6 on MVBench, 56.0 on LongVideoBench, 45.3 on LVBench, 65.0 on EgoSchema, 70.5 on PerceptionTest, 70.2 on MLVU, 71.7 on TempCompass, and 43.6 on Charades-STA. Qwen2.5-VL-3B achieves 61.5/67.6 on Video-MME, 67.0 on MVBench, 54.2 on LongVideoBench, 43.3 on LVBench, 64.8 on EgoSchema, 66.9 on PerceptionTest, 68.2 on MLVU, 64.4 on TempCompass, and 38.8 on Charades-STA. The LVBench results (47.3/45.3/43.3 for 72B/7B/3B) are particularly notable—all three sizes exceed GPT-4o's 30.8, and the small gap between sizes suggests that long-video understanding capability transfers well to smaller models.

**Missing comparisons:** Many benchmarks lack Gemini 1.5 Pro and Claude 3.5 Sonnet numbers. Video-MMMU and MMVU are relatively new benchmarks with sparse baseline coverage. Gemini numbers are absent from LongVideoBench, PerceptionTest, MLVU, and Charades-STA—all benchmarks where Qwen2.5-VL-72B claims advantages, making it impossible to verify whether the advantage holds against the strongest proprietary video model.

#### GUI Agent Capabilities

Table 9 presents GUI agent results. Qwen2.5-VL-72B achieves 87.1 on ScreenSpot (vs. Aguvis-72B's 89.2, Gemini 2.0's 84.0, Claude's 83.0, GPT-4o's 18.1), 43.6 on ScreenSpot Pro (dramatically exceeding Aguvis-72B's 23.6, Claude's 17.1, and Qwen2-VL-72B's 1.6), 67.36 on Android Control High (vs. Aguvis-72B's 66.4, Gemini 2.0's 28.5, GPT-4o's 20.8), 93.7 on Android Control Low (exceeding Aguvis-72B's 84.4, Gemini 2.0's 60.2, GPT-4o's 19.4), 35% on AndroidWorld (without SoM, vs. GPT-4o's 34.5% with SoM, Gemini 2.0's 26% with SoM), 68% on MobileMiniWob++ (without SoM, vs. Aguvis-72B's 66% with SoM), and 8.83 on OSWorld (behind Claude's 14.90 and Aguvis-72B's 10.26).

The ScreenSpot Pro result (43.6 vs. 23.6 for Aguvis-72B and 1.6 for Qwen2-VL-72B) represents a roughly 27× improvement over the direct predecessor, suggesting the grounding and agent training data had an outsized effect on professional high-resolution screen understanding. The AndroidWorld result (35% without SoM, exceeding GPT-4o's 34.5% with SoM) demonstrates that native grounding capabilities can substitute for auxiliary prompting techniques on complex mobile tasks.

The OSWorld result (8.83) is notably weak compared to Claude's 14.90, indicating that desktop computer operation remains challenging. The paper does not discuss this relative weakness.

**Missing comparisons:** GPT-4o numbers are absent from ScreenSpot Pro, MobileMiniWob++, and OSWorld (where Claude leads). The Gemini 2.0 numbers on AndroidWorld use the earlier Gemini model, not the "Gemini 2.0" naming elsewhere in the table—this inconsistency is not explained. The ScreenSpot Pro evaluation uses a new benchmark (Li et al., 2025a) with limited baseline coverage, making the large margin difficult to contextualize.

#### Pure Text Performance

Table 4 compares Qwen2.5-VL-72B against pure text LLMs. On general tasks: 71.2 on MMLU-Pro (vs. Qwen2.5-72B's 71.1 and Llama-3.1-405B's 73.3), 85.9 on MMLU-redux (vs. Qwen2.5-72B's 86.8), and 57.0 on LiveBench-0831 (exceeding all baselines including Llama-3.1-405B's 53.2). On math and science: 49.0 on GPQA (matching Qwen2.5-72B, behind Llama-3.1-405B's 51.1), 83.0 on MATH (matching Qwen2.5-72B), 95.3 on GSM8K (vs. Qwen2.5-72B's 95.8). On coding: 87.8 on HumanEval (vs. Qwen2.5-72B's 86.6), 79.5 on MultiPL-E (exceeding all baselines). On alignment: 86.3 on IFEval (exceeding Qwen2.5-72B's 84.1 and matching Llama-3.1-405B's 86.0).

These results demonstrate that the vision-language fine-tuning (adding visual capabilities) did not catastrophically degrade text-only performance—the multimodal model largely matches or exceeds the base Qwen2.5-72B LLM. The LiveBench-0831 result of 57.0 (a 4.7-point improvement over Qwen2.5-72B's 52.3) is notable and suggests the multimodal training may have indirectly improved general capabilities.

**Missing comparisons:** The pure text comparison includes only LLM baselines; it is not a multimodal evaluation. It serves as a sanity check rather than a contribution. Most vision-language models either degrade on text benchmarks or are not evaluated on them, so Qwen2.5-VL maintaining parity is noteworthy.

### Ablation Studies and Robustness Checks

The paper contains **no formal ablation studies** in the traditional sense (controlled experiments varying one component while holding others fixed). The paper does not report:
- Ablation of window attention (comparing full-attention ViT vs. windowed-attention ViT at equal training compute)
- Ablation of absolute time MRoPE (comparing frame-index vs. timestamp-aligned temporal encoding on video benchmarks)
- Ablation of the HTML document format (comparing standard text-based document parsing training vs. the QwenVL HTML format)
- Ablation of training data components (comparing models trained with/without grounding data, document parsing data, agent data)
- Ablation of the 4-stage interleaved data scoring pipeline (comparing unfiltered vs. filtered interleaved data)
- Ablation of the rejection sampling pipeline (comparing SFT with/without rejection-sampled reasoning traces)
- Ablation of the DPO phase (comparing SFT-only vs. SFT+DPO)
- Ablation of ViT training strategy (comparing from-scratch vs. pre-trained ViT initialization)
- Any systematic study varying model scale against data quantity or compute budget

This is a significant methodological gap. The paper makes architectural and data claims (e.g., "window attention reduces computational overhead while maintaining native resolution," "absolute time encoding enables second-level event localization," "the QwenVL HTML format enables unified document parsing") but provides **no controlled experiments isolating these contributions**. The reported benchmark improvements over Qwen2-VL could be attributed to any or all of: the larger training corpus (4.1T vs. 1.2T tokens), the improved LLM backbone (Qwen2.5 vs. Qwen2), the architectural changes (window attention, absolute time MRoPE), the new data formats (HTML document parsing, absolute coordinate grounding), the improved data filtering and scoring pipeline, the rejection sampling procedure, or the DPO alignment phase. Without ablation, the reader cannot assess which of these factors drives the performance improvements.

**Implicit ablations through model size comparison:** The three model sizes (3B, 7B, 72B) sharing an identical ViT architecture provide an implicit ablation of LLM scale's effect on visual capabilities. The relatively small gap between sizes on perceptual benchmarks (e.g., RefCOCO val: 89.1/90.0/92.7 for 3B/7B/72B; DocVQA: 93.9/95.7/96.4; TextVQA: 79.3/84.9/83.5) suggests that perceptual fidelity is largely determined by the ViT and training data, not LLM scale. This is an interesting finding that the paper does not explicitly analyze.

**Implicit ablation through predecessor comparison:** Comparing Qwen2.5-VL-72B against Qwen2-VL-72B on benchmarks where both are evaluated provides a rough measure of cumulative improvement. On MMMU: 70.2 vs. 64.5 (+5.7). On MathVista: 74.8 vs. 70.5 (+4.3). On DocVQA: 96.4 vs. 95.1 from InternVL2.5 (Qwen2-VL not directly compared). On OCRBench: 885 vs. 854 from InternVL2.5. On ScreenSpot Pro: 43.6 vs. 1.6 (+42.0—likely reflecting entirely new agent training data rather than architectural improvements). These comparisons aggregate all changes (architecture, data, training) and cannot isolate individual contributions.

**Data filtering pipeline validation (Section 2.3.2):** The paper describes a two-stage data filtering pipeline (domain categorization + domain-tailored filtering) but reports no experiments comparing filtered vs. unfiltered SFT data. The effectiveness of the pipeline is asserted but not demonstrated.

**Rejection sampling validation (Section 2.3.3):** The paper states that "our post-training experiments confirm this" (that CoT reasoning improves inferential performance) but reports no specific ablation comparing SFT with and without rejection-sampled reasoning traces. The benefit is taken as given from prior work (DeepSeek-AI et al., 2024).

**ChatML format transition:** The paper notes transitioning from pre-training format to ChatML format for SFT (Section 2.3) and argues this enables "explicit dialogue role tagging" and "structured injection of visual embeddings." No comparison between format choices is reported.

This near-total absence of ablation studies makes it impossible to determine whether the paper's four named contributions (window attention, dynamic FPS, absolute time MRoPE, data scaling) are individually necessary or sufficient for the reported performance. The paper reads as a capability demonstration (showing what the full system can do) rather than a scientific analysis (showing why each component matters). This is common for large-scale industrial technical reports but limits the paper's value as a guide for future research—readers cannot determine which innovations to adopt and which are incidental.

### Critical Assessment

The experiments in this paper demonstrate that Qwen2.5-VL achieves competitive or state-of-the-art performance across a broad range of vision-language benchmarks, and that strong perceptual capabilities are achievable even at the 3B and 7B parameter scales. However, the experimental design has several structural limitations that affect what conclusions can be drawn.

**On the claim that Qwen2.5-VL "matches state-of-the-art models like GPT-4o and Claude 3.5 Sonnet":** The benchmarks partially support this. Qwen2.5-VL-72B achieves higher numbers than GPT-4o and Claude 3.5 Sonnet on many benchmarks (e.g., DocVQA 96.4 vs. 95.2 for Claude, OCRBench 885 vs. 788 for Claude and 736 for GPT-4o, LVBench 47.3 vs. 30.8 for GPT-4o). However, the comparison is incomplete—both proprietary models have missing entries across multiple tables. On the benchmarks where all three are compared, the picture is mixed: GPT-4o leads on MMMU-Pro (51.9 vs. 51.1), MegaBench (54.2 vs. 51.3), LongVideoBench (66.7 vs. 60.7), and MMVU (67.4 vs. 62.9); Claude 3.5 Sonnet leads on ChartQA (90.8 vs. 89.5) and OSWorld (14.90 vs. 8.83). The "matches" claim is directionally accurate but overstates the comprehensiveness of the comparison. A more precise characterization: Qwen2.5-VL-72B is competitive with GPT-4o and Claude 3.5 Sonnet, showing particular strength in document understanding and OCR, competitive performance in general VQA and video understanding, and weaker performance on some agent and chart reasoning tasks. This is still a significant achievement for an open-source model, but the "matches" framing glosses over the specific dimensions of advantage and disadvantage.

**On the claim of "powerful document parsing capabilities":** Strongly supported. The OCRBench_v2 results (61.5/63.7 vs. Gemini 1.5 Pro's 51.9/43.1) represent a genuinely large gap. The OmniDocBench results (0.226/0.324 edit distance vs. GPT-4o's 0.265/0.435) demonstrate practical document parsing quality. The CC-OCR result (79.8 vs. 73.0 for Gemini) further reinforces this capability. These are multiple independent benchmarks showing consistent document understanding advantages. The claim is well-supported, though the absence of ablations makes it impossible to attribute the gain specifically to the HTML format versus general data quality improvements.

**On the claim of "precise object grounding across formats":** Partially supported. The RefCOCO results (92.7–94.6 on standard splits) are strong but within the range of prior work (InternVL2.5-78B achieves 93.7–95.6). The ODinW-13 result (43.1) shows open-vocabulary generalization but remains below specialist models (Grounding DINO's 55.0). The PointGrounding benchmark is self-curated, making the 67.5 number difficult to contextualize. The CountBench result (93.6) is genuinely impressive and substantially exceeds competitors. Overall, grounding capabilities are strong but not uniformly demonstrated across standardized benchmarks—the reliance on a self-curated pointing benchmark weakens the "precise" claim.

**On the claim of "ultra-long video understanding and fine-grained video grounding":** Well-supported for video grounding (Charades-STA mIoU 50.9 vs. GPT-4o's 35.7), partially supported for long-video understanding. LVBench results (47.3 vs. 30.8 for GPT-4o) are strong, but LongVideoBench (60.7 vs. 66.7 for GPT-4o) is weaker. The paper attributes temporal grounding improvements to absolute time MRoPE but provides no ablation comparing frame-index vs. absolute time encoding on any video benchmark. The claim that these improvements stem from the architectural change is therefore unsupported. The frame cap of 768 and token cap of 24,576 impose practical limits that are not systematically studied—readers cannot determine how performance degrades as videos approach or exceed these limits.

**On the claim of "enhanced agent functionality":** Mixed support. ScreenSpot Pro (43.6 vs. 23.6 for Aguvis-72B) shows a substantial improvement in professional GUI grounding, and the AndroidWorld result (35% without SoM vs. GPT-4o's 34.5% with SoM) demonstrates native interaction capability. However, OSWorld (8.83 vs. Claude's 14.90) shows a significant gap on desktop tasks. The agent evaluation suite covers only a subset of possible interaction scenarios—no evaluation on web navigation benchmarks (e.g., WebArena, Mind2Web) appears. The claim of "superior agent functionality" is supported on mobile tasks but overstated for general computer use.

**On the claim that small variants "outperform comparable competitors":** Well-supported, with a caveat. Qwen2.5-VL-7B and 3B achieve competitive or leading numbers against models of similar size and, in several cases, against much larger models (7B exceeds InternVL2.5-78B on TextVQA, InfoVQA; 3B exceeds Qwen2-VL-72B on several benchmarks). The caveat is that the "comparable competitors" comparison set is sparse—many similarly-sized open-source models are not evaluated, and the paper rests heavily on InternVL2.5 as the primary comparison point. Whether the small Qwen2.5-VL variants outperform *all* comparable open-source models is asserted but not comprehensively demonstrated.

**Structural limitations of the experimental design:**

1. **No ablation studies whatsoever.** This is the most significant weakness. The paper makes claims about four specific innovations but provides zero controlled experiments to validate any of them. The reported results reflect the aggregate effect of architecture, data, and training changes, and there is no way to attribute gains to specific design choices. This substantially limits the paper's scientific contribution beyond the capability demonstration.

2. **Inconsistent baseline coverage across benchmarks.** GPT-4o, Claude 3.5 Sonnet, and Gemini 1.5 Pro numbers are missing from many tables, making comprehensive comparison impossible. The absence is not random—Gemini numbers tend to be missing from benchmarks where Qwen2.5-VL excels, and present where it doesn't, which could introduce a favorable selection bias in the presented comparisons.

3. **Reliance on previously reported baseline numbers.** Many baseline numbers are drawn from other papers' reported results rather than from re-evaluation under identical conditions. Differences in prompt formatting, decoding strategy, image preprocessing, and evaluation protocol can produce non-trivial performance variation that is not accounted for.

4. **No confidence intervals or statistical testing.** All results are point estimates without variance information. On benchmarks with small test sets (e.g., 500 questions for MMMU, 100–500 images for many grounding benchmarks), differences of 1–3 percentage points may not be statistically significant. Claims of "surpassing" or "outperforming" based on such small margins are not statistically justified.

5. **Missing evaluation dimensions.** The paper evaluates on standard academic benchmarks but provides no analysis of: inference latency or throughput, memory consumption during inference, robustness to distribution shift or adversarial inputs, calibration or reliability of outputs, biases in model behavior, performance across different demographic or linguistic subgroups, or performance on real-world deployment tasks outside the benchmark distribution. These are important practical considerations for a model positioned as deployable from "edge AI to high-performance computing."

6. **The agent evaluation does not control for training data quality.** The dramatic improvement from Qwen2-VL-72B's 1.6 to Qwen2.5-VL-72B's 43.6 on ScreenSpot Pro likely reflects entirely new agent training data rather than architectural improvements. Without ablating the agent training data quantity and quality, the contribution of perceptual architecture to agent capabilities cannot be assessed.

7. **The pure text evaluation (Table 4) compares against LLMs but does not evaluate whether the vision-language training harmed language capabilities relative to a Qwen2.5-72B that underwent equivalent continued training without visual data.** The comparison is against the base Qwen2.5-72B, which was trained only on text—any continued training on 4.1T tokens (even with some pure text data mixed in) could have shifted language capabilities, and the current comparison cannot distinguish whether visual training helped, hurt, or was neutral.

**Experiments that would have strengthened the paper:**

- **Ablation of the architectural innovations:** Train variants with/without window attention (at equal FLOPs), with/without absolute time MRoPE (frame-index baseline), with/without the HTML document format (standard text-based parsing data instead), and measure the isolated contribution of each.
- **Data scaling curve:** Train models on 1T, 2T, 4T tokens and measure how benchmark performance scales, to determine whether the 4.1T corpus was necessary or if diminishing returns had set in earlier.
- **Re-evaluation of all baselines under identical conditions** (same prompts, same image preprocessing, same decoding strategy) to ensure fair comparison.
- **Statistical significance reporting** with confidence intervals on key benchmarks.
- **Latency and memory benchmarks** across the three model sizes, particularly for video and agent tasks where the 768-frame cap interacts with practical deployment constraints.
- **Systematic study of the frame cap's effect on video understanding**—how does performance vary with 128, 256, 512, 768 frames on a fixed set of long videos?
- **Comparison against task-specific specialist models** (e.g., dedicated OCR engines, object detectors, table extraction tools) to contextualize whether the generalist LVLM approach is competitive with or superior to purpose-built systems.

The paper's core value proposition—that Qwen2.5-VL achieves competitive performance through specific architectural innovations in spatial and temporal processing—is directionally supported by the benchmark results but not causally validated. The absence of ablation studies means the paper functions primarily as a **capability report** ("here is what our system can do") rather than a **scientific analysis** ("here is why each design choice matters"). This is appropriate for a technical report from an industrial lab and valuable for practitioners who want to use or build on the system, but readers seeking to understand which innovations to adopt in their own work will find limited guidance.

## 6. Limitations and Trade-offs

### 6.1 No Ablation Studies Isolate the Contribution of Named Architectural Innovations

**The assumption or constraint.** The paper names four specific technical contributions in Section 1: window attention in the visual encoder, dynamic FPS sampling extending dynamic resolution to the temporal dimension, MRoPE aligned to absolute time, and scaling pre-training data from 1.2T to 4.1T tokens with improved curation. The entire narrative frames these as the drivers of Qwen2.5-VL's performance improvements over Qwen2-VL and competitors. However, the paper performs **zero controlled ablation experiments**—no comparison of the full model against variants with any of these mechanisms removed or replaced with alternatives. Every benchmark result in Section 3 reflects the aggregate effect of all architectural changes, all data improvements, and all training recipe modifications simultaneously.

**The consequence.** A practitioner reading this paper cannot determine which innovations to adopt. If a team has limited engineering resources and must choose between implementing window attention, absolute time MRoPE, or the HTML document parsing format, the paper provides no evidence about the marginal benefit of each. The improvement over Qwen2-VL-72B (e.g., +5.7 points on MMMU, +42.0 points on ScreenSpot Pro, unknown on most benchmarks because Qwen2-VL results are not systematically reported) could be explained entirely by the 3.4× increase in pre-training tokens and the new LLM backbone (Qwen2.5 vs. Qwen2), with the architectural innovations contributing minimally or not at all. The paper provides no way to distinguish these possibilities.

The absence of ablations is particularly acute for the absolute time MRoPE claim. The paper positions this as the core temporal reasoning innovation: "Qwen2.5-VL introduces a key improvement: aligning the temporal component of MRoPE with absolute time" (Section 2.1.3), and attributes the Charades-STA temporal grounding result (50.9 mIoU vs. GPT-4o's 35.7) to it. But without comparing frame-index MRoPE against absolute-time MRoPE on the same model trained on the same data, the reader cannot know whether the gain comes from the encoding change, the larger video training corpus, or the dynamic FPS training. A model with frame-index MRoPE trained on 4.1T tokens with the new video data might achieve similar Charades-STA performance—the paper provides no evidence either way.

**What evidence exists in the paper.** None. The paper does not contain an ablation study section. The model size comparison (3B/7B/72B sharing an identical ViT) provides an implicit ablation of LLM scale's effect on visual perception—the small gaps on grounding and OCR benchmarks suggest the ViT, not the LLM, dominates perceptual quality—but this is a data observation the paper does not analyze as an ablation, and it tells us nothing about the marginal contribution of window attention, absolute time MRoPE, or the HTML format relative to alternatives.

The paper's closest approach to ablation is the comparison between Qwen2.5-VL-72B and Qwen2-VL-72B on benchmarks where both appear. But this comparison is: (1) sparse (only a handful of shared benchmarks across Tables 3, 5, 8, 9), (2) confounded by the LLM backbone change (Qwen2.5 vs. Qwen2), the data scale increase (4.1T vs. 1.2T), and the post-training improvements (new SFT data, DPO phase), and (3) not presented as an ablation but as an incidental comparison against a predecessor. The reader cannot isolate any individual innovation's effect.

**Mitigation status.** The paper does not acknowledge the absence of ablation studies as a limitation. It does not suggest future work to isolate component contributions. The four named contributions are presented as a package with the implicit assumption that all are valuable, but no evidence is provided to support this assumption. This is the single most consequential methodological weakness in the paper: it transforms what could be a scientific contribution (demonstrating *why* specific architectural choices help) into a capability report (demonstrating *that* the full system performs well). For a technical report from an industrial lab, the capability demonstration is valuable; for a research contribution intended to guide future work, the absence of causal evidence is a fundamental limitation.

---

### 6.2 Visual Encoder Training Data Quality Depends on an Undisclosed Proprietary Dataset

**The assumption or constraint.** The Vision Transformer is trained from scratch using "DataComp (Gadre et al., 2023) and some in-house datasets as the initialization for the vision encoder" (Section 2.2.2). DataComp is a public dataset; the in-house datasets are not described, enumerated, or released. The nature, scale, and quality of these proprietary datasets are unknown. The ViT's ability to process native-resolution images efficiently via window attention and produce features that the LLM can ground in absolute spatial coordinates depends on the quality of this pre-training data—the architecture alone cannot compensate for poor visual representations.

**The consequence.** Any team attempting to reproduce Qwen2.5-VL's visual capabilities by implementing the described architecture (window attention ViT, 2D-RoPE, SwiGLU, RMSNorm) and training on public data alone may obtain substantially different results. The ViT is shared across all three model sizes (Table 1: 1280 hidden, 32 layers, identical for 3B/7B/72B), meaning the visual backbone's quality propagates to all variants. If the proprietary in-house datasets contain high-quality annotations (e.g., dense object bounding boxes, instance segmentation masks, attribute labels) that are absent from public datasets like DataComp, the from-scratch ViT training will produce inferior visual features when trained on public data alone. The paper provides no characterization of the proprietary data's content, scale, or annotation quality, making it impossible to estimate the performance gap between the reported model and a public-data-only reproduction.

This limitation interacts with the grounding and document parsing capabilities. The ViT pre-training data likely includes images that are relevant to the downstream tasks—documents, screenshots, scene images with objects at various scales. If the proprietary dataset includes, for example, large quantities of document images with layout annotations or UI screenshots with element bounding boxes, the ViT would learn spatial representations that are specifically useful for the grounding and parsing tasks. A ViT trained only on natural images from public datasets would lack this domain-specific pre-training, potentially degrading performance on document parsing and GUI grounding regardless of how well the downstream training data (the 4.1T token corpus) is constructed.

**What evidence exists in the paper.** None. The paper provides no details about the in-house datasets beyond the single sentence mentioning them. There is no comparison between a ViT trained on DataComp alone versus DataComp + in-house data. There is no analysis of how ViT pre-training data composition affects downstream task performance. The DataComp dataset itself is large (~140 million images) but consists primarily of web-crawled image-text pairs with noisy alt-text captions; it contains relatively few document images, screenshots, or images with dense spatial annotations. The gap between DataComp's content distribution and Qwen2.5-VL's target capabilities (document parsing, precise grounding, UI understanding) suggests the proprietary data likely fills a critical role that cannot be replicated from public sources alone.

**Mitigation status.** The paper does not acknowledge this as a limitation. It does not describe the proprietary data's characteristics, does not provide a public-data-only ViT training baseline, and does not release the trained ViT weights separately from the full model (they are available as part of the released model checkpoints, so the *trained* ViT is accessible, but the *training data and procedure* for reproducing it from scratch are not fully specified). The release of model weights on Hugging Face and ModelScope mitigates this limitation for *inference* (users can use the pre-trained ViT) but not for *reproduction* or *adaptation* (users cannot train an equivalent ViT from scratch on new data or for new domains without access to the training data recipe). This is a standard tradeoff in industrial open-source releases—model weights are shared, training data is not—but it limits the paper's value as a blueprint for building similar systems.

---

### 6.3 Video Understanding Is Capped at 768 Frames Without Systematic Study of the Cap's Effect

**The assumption or constraint.** For all video benchmarks, "we capped the maximum number of frames analyzed per video at 768, with the total number of video tokens not exceeding 24,576" (Section 3.3.4). This cap is a practical engineering constraint matching the LLM's context window (32,768 tokens after long-context pre-training) and the computational budget for inference. For a 1-hour video at 30 FPS, the original video contains 108,000 frames; the model sees at most 0.7% of them. For a 2-hour movie, it sees at most 0.35%. The frame selection strategy—how 768 frames are chosen from potentially 100,000+—is not described. The model must reason about events, actions, and temporal relationships based on a sparse, potentially non-uniform sample of the full video.

**The consequence.** The video understanding benchmarks in Table 8 may overestimate real-world performance on long videos because the benchmark videos are likely shorter or less temporally dense than the worst-case scenarios the model will encounter in deployment. Video-MME includes videos ranging from a few seconds to ~1 hour—the 768-frame cap is generous for short videos (it may exceed the video's total frame count) but becomes increasingly restrictive as video length grows. For a 2-hour surveillance video where an event of interest occupies 5 seconds, the probability that the 768-frame sample includes frames from those 5 seconds depends on the sampling strategy. If frames are sampled uniformly, the probability is approximately 768 × 5 / 7200 ≈ 53%. If the sampling strategy is adaptive (e.g., sampling more densely around detected motion), it may be higher, but the paper does not specify the strategy.

The paper claims "ultra-long video understanding and fine-grained video grounding" and "understanding videos lasting hours while extracting event segments in seconds" (Section 1). But the frame cap creates a fundamental tension: longer videos mean sparser temporal sampling, which reduces the model's ability to detect short-duration events. The Charades-STA result (50.9 mIoU, Table 8) demonstrates strong temporal grounding, but Charades videos average ~30 seconds—the 768-frame cap is not binding for these videos, and the result does not validate temporal grounding on multi-hour content. The LVBench result (47.3, exceeding GPT-4o's 30.8) evaluates long-video understanding through question-answering rather than precise temporal localization, so it tests coarse understanding (did X happen?) rather than fine-grained grounding (exactly when did X happen?).

**What evidence exists in the paper.** The paper provides **no systematic study** of how performance varies with the frame cap. There is no experiment comparing 128, 256, 512, and 768 frames on long videos. There is no analysis of how frame sampling density affects temporal grounding accuracy. There is no characterization of the frame selection algorithm (uniform sampling? keyframe detection? content-adaptive?). The paper does not report video duration distributions for the evaluation benchmarks, so the reader cannot assess how often the 768-frame cap is binding. The statement that "the total number of video tokens not exceeding 24,576" suggests that for high-resolution videos, the token cap might bind before the frame cap—a 4K video frame at 3840×2160 produces (3840/28) × (2160/28) ≈ 137 × 77 ≈ 10,549 tokens per frame after the 4× spatial compression, meaning the token cap would be reached at only ~2.3 frames, making the 768-frame cap irrelevant. The interaction between spatial resolution and temporal sampling density is not analyzed.

**Mitigation status.** The paper does not acknowledge the frame cap as a limitation or discuss its implications for video understanding quality. It does not propose strategies for adaptive frame selection, hierarchical temporal processing, or any other mechanism for handling videos that exceed the cap. The dynamic FPS training (Section 2.2.1) partially mitigates the issue by teaching the model to handle variable frame rates, so the model is at least *trained* on sparse temporal sampling—but whether this training translates to robust performance at the extreme sparsity imposed by 768 frames on multi-hour videos is not evaluated. The absolute time MRoPE (Section 2.1.3) helps the model understand the temporal gaps between sampled frames, but it cannot recover information that was never sampled. If an event occurs entirely between two sampled frames, no temporal encoding can help the model detect it. The claim of "understanding videos lasting hours while extracting event segments in seconds" (Section 1) is therefore supported only for videos where events are long enough or frequent enough to be captured by the sampling density—a condition that is neither quantified nor validated.

---

### 6.4 The Point Grounding and ScreenSpot Pro Benchmarks Are Self-Curated or Recently Introduced With Sparse Baseline Coverage

**The assumption or constraint.** Two of the paper's most striking quantitative claims rest on benchmarks with **limited external validation**. The PointGrounding benchmark (Table 6), where Qwen2.5-VL-72B achieves 67.5, is described as "self-curated" (Section 3.3.3): the dataset, annotation procedure, evaluation metric, and baseline numbers are not independently established through prior literature. Only Molmo-72B (69.2) and the three Qwen2.5-VL variants have reported numbers. The ScreenSpot Pro benchmark (Table 9), where Qwen2.5-VL-72B achieves 43.6—dramatically exceeding Aguvis-72B's 23.6, Claude's 17.1, and Qwen2-VL-72B's 1.6—is a recently introduced benchmark from Li et al. (2025a) with limited community adoption and baseline coverage as of the paper's writing. GPT-4o and Gemini numbers are absent from ScreenSpot Pro; the GPT-4o result on the original ScreenSpot (18.1 vs. Qwen2.5-VL's 87.1) suggests the evaluation setup differs substantially between the two benchmarks, but the paper does not analyze this discrepancy.

**The consequence.** The 67.5 PointGrounding result and the 43.6 ScreenSpot Pro result—both presented as evidence of "precise object grounding across formats" and "enhanced agent functionality"—cannot be contextualized against a broad set of strong baselines. On PointGrounding, the only comparison is Molmo-72B (69.2), which is higher than Qwen2.5-VL-72B's 67.5. The paper does not explain why Molmo leads on this metric, what the metric captures, or what a "good" score means in absolute terms. A practitioner cannot determine whether 67.5 represents near-perfect pointing or mediocre pointing that happens to exceed an arbitrary baseline. On ScreenSpot Pro, the 43.6 vs. 1.6 gap relative to Qwen2-VL-72B is so large (+42.0) that it almost certainly reflects the addition of entirely new agent training data (UI screenshots, grounding annotations, reasoning trajectories) that Qwen2-VL-72B never saw, rather than any architectural improvement. The paper does not disentangle these factors.

This matters because these two numbers are prominently featured in the paper's claims. The "precise object grounding" claim in Section 1 includes "pointing" as a key capability. The "enhanced agent functionality" claim includes "superior agent functionality on smartphones and computers." If the PointGrounding benchmark is poorly designed (e.g., the evaluation metric is lenient, the ground truth annotations are noisy, the task is too easy or too hard to discriminate between models), or if the ScreenSpot Pro result is primarily a reflection of training data scale rather than model quality, the claims are misleading. The absence of GPT-4o and Gemini numbers on both benchmarks is particularly concerning—these are the strongest proprietary models, and their omission may indicate that they perform poorly (casting doubt on the benchmark's validity) or that they were not evaluated (leaving an incomplete comparison).

**What evidence exists in the paper.** For PointGrounding: only the numbers in Table 6 for three Qwen2.5-VL variants and Molmo-72B. The paper provides no description of the benchmark's construction, size, annotation procedure, difficulty distribution, or evaluation metric beyond naming it "PointGrounding" and describing it as "self-curated" (Section 3.3.3). For ScreenSpot Pro: the numbers in Table 9, with baseline coverage limited to GPT-4o (absent), Claude (17.1), Aguvis-72B (23.6), and Qwen2-VL-72B (1.6). The benchmark is cited as Li et al. (2025a), a preprint, so its methodology is partially accessible but not described in the Qwen2.5-VL paper itself. The gap between ScreenSpot (where GPT-4o scores 18.1) and ScreenSpot Pro (where Qwen2.5-VL-72B scores 43.6) indicates these are very different evaluations—ScreenSpot Pro focuses on "professional high-resolution computer use" according to its title, while ScreenSpot evaluates general GUI element grounding. The paper does not explain what makes ScreenSpot Pro harder, what the 43.6 score means in practical terms, or why GPT-4o's number is missing.

**Mitigation status.** The paper does not acknowledge the self-curated nature of PointGrounding or the limited baseline coverage of ScreenSpot Pro as limitations. It does not provide sufficient detail for readers to reproduce or critically evaluate the PointGrounding benchmark. It does not analyze why GPT-4o and Gemini are absent from ScreenSpot Pro or what their expected performance would be. The release of the model weights mitigates the issue partially—external researchers can evaluate Qwen2.5-VL on other, better-established pointing and GUI grounding benchmarks—but this shifts the evaluation burden to the community rather than providing it in the paper. For a technical report making strong claims about pointing precision and agent capability, the reliance on self-curated and sparsely-baselined benchmarks represents a significant evidentiary gap.

---

### 6.5 No Guarantees About Cross-Language Generalization for Document Parsing Beyond Chinese and English

**The assumption or constraint.** The paper's document parsing claims are supported primarily by benchmarks in English and Chinese: OCRBench_v2 (English and Chinese tracks), OmniDocBench (English and Chinese edit distance), CC-OCR (language composition not specified but likely Chinese/English-focused given the benchmark name and authors), and the standard English-language OCR benchmarks (TextVQA, DocVQA, InfoVQA, ChartQA, AI2D). The paper mentions that the OCR training data includes "French, German, Italian, Spanish, Portuguese, Arabic, Russian, Japanese, Korean, and Vietnamese" (Section 2.2.1), and the SFT data includes "supplementary multilingual entries to support broader linguistic diversity" (Section 2.3.1). However, **no evaluation is reported for any language other than English and Chinese on document parsing or OCR tasks**. The MTVQA benchmark (Table 3) evaluates multilingual text-centric VQA, where Qwen2.5-VL-72B achieves 31.7 (exceeding InternVL2.5's 31.9—the paper incorrectly claims 31.7 exceeds 31.9 in the text, but the table shows InternVL2.5 at 31.9 and Qwen2.5-VL at 31.7), but this is a single aggregated number across 9 languages and does not provide per-language breakdowns or evaluate structured document parsing.

**The consequence.** A practitioner deploying Qwen2.5-VL for document parsing in Arabic, Japanese, Korean, or any of the other listed languages cannot predict performance from the paper's reported results. The document parsing capability—which the paper positions as a flagship feature ("Powerful document parsing capabilities: Qwen2.5-VL upgrades text recognition to omni-document parsing, excelling in processing multi-scene, multilingual, and various built-in documents")—is validated only for the two highest-resource languages. The risks of deployment without evaluation are substantial:

- **Script-specific failures:** Arabic uses right-to-left text flow and connected cursive characters; Japanese and Korean use logographic and syllabic scripts with thousands of distinct characters. OCR models trained predominantly on Latin and Chinese characters often fail on these scripts in ways that are not predictable from English performance.

- **Layout differences:** Document layouts vary by language and region—Arabic documents flow right-to-left, Japanese documents can be vertical, multilingual documents may mix scripts and writing directions within a single page. The HTML document parsing format's reading-order assumption (left-to-right, top-to-bottom) may not hold for these languages.

- **The HTML format's language assumptions:** The QwenVL HTML format (Section 2.2.1) uses English tag names and attributes ( `<p>`, `<table>`, `data-bbox`). While tag names are language-agnostic, the model's ability to generate this format correctly when processing documents in Japanese or Arabic—where the text content uses different character sets and the spatial layout may follow different conventions—is not validated.

The MTVQA result (31.7) provides some evidence of multilingual text understanding, but this benchmark evaluates VQA (answering questions about text in images) rather than structured parsing (extracting tables, charts, formulas into machine-readable format). The gap between these capabilities is the paper's own rationale for introducing the HTML omni-parsing format—traditional OCR handles text recognition, but document parsing requires layout understanding and structured extraction. The multilingual evaluation only covers the former, not the latter.

**What evidence exists in the paper.** MTVQA (Table 3): 31.7 for Qwen2.5-VL-72B vs. 31.9 for InternVL2.5-78B. No per-language breakdown. No document parsing evaluation in non-English/non-Chinese languages. The paper's description of multilingual OCR training data (Section 2.2.1) confirms the data exists but provides no evaluation. The OmniDocBench evaluation (Table 5) includes English and Chinese tracks with edit distance metrics—the Chinese track (0.324) performs worse than the English track (0.226), suggesting language-dependent performance variation even between the two evaluated languages. Whether this gap widens for languages with more distinct scripts (Arabic, Japanese) is unknown.

**Mitigation status.** The paper does not acknowledge this as a limitation. It does not suggest future work on multilingual document parsing evaluation. The claim of "multilingual" capabilities in the document parsing context is supported only by the presence of multilingual training data and a single aggregated VQA benchmark result. For a model positioned as excelling at "multi-scene, multilingual, and various built-in documents," the absence of multilingual structured parsing evaluation is a significant gap between the claimed capability and the provided evidence.

---

### 6.6 Inference Latency and Memory Costs for Video and Agent Tasks Are Not Characterized

**The assumption or constraint.** The paper reports accuracy and success rate metrics across benchmarks but provides **no characterization of inference cost**: latency (seconds per query), throughput (queries per second), memory consumption (GPU VRAM required), or how these scale with input size (image resolution, video duration, number of agent steps). The 768-frame cap and 24,576-token video budget (Section 3.3.4) define a compute envelope, but the actual wall-clock time and memory required to process a video at that envelope are not reported. For agent tasks involving multi-step interactions (AndroidWorld, OSWorld, MobileMiniWob++), each step requires a full model forward pass, but the paper reports only success rates without step counts or per-step latency.

**The consequence.** The paper positions Qwen2.5-VL as suitable for deployment "from edge AI to high-performance computing" (Section 1) and offers model sizes from 3B to 72B parameters. But a practitioner choosing between these sizes for a specific deployment scenario cannot make an informed decision without understanding the computational requirements. Key unknowns:

- **Video processing latency:** The 72B model processing 768 video frames with up to 24,576 visual tokens requires a forward pass through an 80-layer LLM with 8192 hidden size. Depending on hardware (GPU generation, memory bandwidth, tensor parallelism configuration), this could take seconds to minutes per video. A real-time video understanding application (e.g., live surveillance, video conferencing assistant) may be infeasible if per-frame or per-clip latency exceeds the application's time budget.

- **Memory requirements for long contexts:** The 32,768-token context window (after Stage 3 long-context pre-training) implies a KV cache of size proportional to sequence length × number of layers × hidden size × 2 (for key and value) × number of KV heads. For the 72B model: 32,768 × 80 × 8192 × 2 × 8 KV heads / (group size) bytes. The exact memory depends on precision (FP16 vs. INT4 quantization) and whether KV cache compression techniques are used, but the paper provides no guidance.

- **Multi-step agent interaction cost:** The AndroidWorld success rate of 35% (Table 9) is reported without the average number of steps per episode, the per-step inference latency, or the total wall-clock time per task. If the 72B model requires 2 seconds per step and successful episodes average 10 steps, a single task takes 20 seconds—acceptable for asynchronous automation but too slow for real-time user interaction. If failed episodes involve 50+ steps of exploration before giving up, the average cost per task could be substantially higher than the success rate alone suggests.

- **Scaling behavior from 3B to 72B:** The 3B model achieves competitive performance on many perceptual benchmarks (e.g., 93.9 on DocVQA, 89.1 on RefCOCO val), but the paper does not report whether the 3B model's inference latency makes it suitable for edge deployment—specifically, on-device inference on a smartphone or laptop without cloud offloading. The 3B model's 36-layer, 2048-hidden LLM with a full 32-layer ViT may still be too large for real-time on-device inference, but without latency numbers, this cannot be assessed.

**What evidence exists in the paper.** None. The paper reports no latency, throughput, memory, or FLOPs measurements. The model configurations (Table 1) provide parameter counts and architectural dimensions from which approximate FLOPs can be estimated, but the paper does not perform this calculation or provide empirical measurements. The dynamic resolution processing (Section 2.1.2) produces variable-length visual token sequences—the paper does not characterize the distribution of sequence lengths for typical inputs or how this affects inference cost. The window attention design in the ViT (Section 2.1.1) is motivated by computational efficiency ("significantly reduced computational overhead while maintaining native resolution"), but the paper provides no quantitative comparison of ViT inference cost with and without window attention.

**Mitigation status.** The paper does not acknowledge the absence of inference cost characterization as a limitation. It does not suggest future work on optimizing inference latency or memory. The release of model weights enables the community to benchmark these metrics independently—which will likely happen given the model's open-source release—but this shifts the evaluation burden to users rather than providing it as part of the technical report. For a model series explicitly targeting deployment scenarios "from edge AI to high-performance computing" (Section 1), the omission of deployment-relevant computational metrics is a significant gap between the paper's positioning and its empirical content.

## 7. Implications and Future Directions
- How this work changes the landscape:
  - Demonstrates that native dynamic resolution plus absolute-time positional encoding are practical at scale and translate into concrete gains in document parsing and temporal grounding—two historically difficult areas for LVLMs.
  - Bridges perception and agency by combining precise grounding with decision-making, enabling more autonomous device operation (Table 9).

- Follow-up research enabled/suggested:
  - Component ablations and principled studies:
    - Quantify the contribution of absolute-time MRoPE vs. frame-index time.
    - Explore optimal placement/number of global-attention layers under compute budgets.
    - Study the fidelity–efficiency curve of the 4-patch MLP merger and alternative pooling schemes.
  - Extending temporal modeling:
    - Continuous-time encodings, variable-speed events, multi-camera synchronization.
    - Event-centric memory and retrieval that leverage absolute-time MRoPE.
  - Document intelligence:
    - Expand the HTML target to richer semantics (e.g., logical relations between tables/figures), multi-page long documents, and layout-aware editing.
  - Agents:
    - Tighten grounding→action chains with verification loops; incorporate explicit UI state models; evaluate on more realistic enterprise workflows and security-constrained environments.

- Practical applications:
  - Enterprise document automation: invoices, forms, compliance reports, scientific papers with formulas/diagrams (Section 2.2.1; Table 5).
  - Visual analytics: chart/diagram understanding, counting and open-vocabulary detection (Tables 5–7).
  - Video analytics: surveillance/event detection, sports highlights with precise timestamps, educational video indexing (Table 8).
  - Device agents: automated app workflows on phones/PCs, testing and RPA-style tasks (Table 9).

> Figure 1 illustrates the end-to-end pipeline with native-resolution ViT + MRoPE aligned to absolute time + LLM decoder; Table 1 details ViT and LLM configs; Table 2 outlines the three-stage training with sequence lengths; Tables 3, 5, 6–9 report broad SOTA-competitive results, especially in document/OCR and temporal grounding.
