# Emu3: Next-Token Prediction is All You Need

**ArXiv:** [2409.18869](https://arxiv.org/abs/2409.18869)

## 🎯 Pitch

Emu3 introduces a unified multimodal Transformer that achieves state-of-the-art results in text, image, and video tasks using only next-token prediction, dispensing with complex diffusion pipelines and compositional architectures. By encoding all modalities into discrete tokens and training a single decoder from scratch, Emu3 not only matches or surpasses specialized systems like SDXL and LLaVA-1.6, but also dramatically simplifies model design, scaling, and deployment—paving the way for efficient, generalizable multimodal intelligence.

---

## 1. Executive Summary

This paper introduces **Emu3**, a suite of multimodal models that unifies image, video, and text processing under a single next-token prediction framework — eliminating diffusion models and compositional architectures (e.g., CLIP combined with LLMs) entirely. By tokenizing all modalities into a discrete space and training a single Transformer decoder from scratch on mixed sequences, Emu3 achieves state-of-the-art performance across generation and perception, surpassing SDXL in human-evaluated image generation, LLaVA-1.6-7B across twelve vision-language benchmarks, and OpenSora-1.2 on VBench, while also enabling causal video extension via autoregressive future-frame prediction. The work establishes that next-token prediction alone can match or exceed well-established task-specific architectures, demonstrating that a unified token-based approach unlocks competitive multimodal intelligence without the complexity of diffusion or compositional design.

## 2. Context and Motivation

### The Core Problem: Multimodal AI Is Built on Fragmented Architectures

The paper addresses a fundamental architectural rift in contemporary multimodal AI. Despite sharing the goal of processing and generating content across text, images, and video, the field has settled into two distinct technological paradigms that share almost no common infrastructure:

- **Vision generation** (text-to-image, text-to-video) is dominated by **diffusion models** — architectures like Stable Diffusion that learn to iteratively denoise random inputs into coherent outputs. The flagship open-source model in this space at the time of writing is SDXL.
- **Vision-language understanding** (visual question answering, image captioning, OCR) is led by **compositional approaches** — systems that bolt a pretrained vision encoder (typically CLIP) onto a pretrained LLM, then fine-tune the combined system on image-text data. The leading open-source representative is the LLaVA family, with LLaVA-1.6-7B as the best-performing variant.

These two pipelines share **nothing**: different training objectives, different architectures, different tokenization strategies, and different inference procedures. If an organization wants a system that both generates images and answers questions about them, it must build, train, deploy, and maintain two entirely separate models. This is not merely an engineering inconvenience — it represents a **conceptual failure** in the field's approach to multimodal intelligence.

The paper frames this as a direct challenge to the premise that next-token prediction — the paradigm that produced GPT-4 and sparked serious discussion about artificial general intelligence — is sufficient for multimodal tasks. As the authors state bluntly in the abstract:

> "While next-token prediction is considered a promising path towards artificial general intelligence, it has struggled to excel in multimodal tasks, which are still dominated by diffusion models and compositional approaches."

This framing matters because it positions the paper not as an incremental improvement to diffusion or CLIP-based methods, but as an existence proof: demonstrating that a single next-token prediction model can match or exceed both, rendering their architectural complexity unnecessary.

---

### Why This Matters: Real-World Impact and Theoretical Significance

**Practical fragmentation imposes real costs.** Building separate diffusion and vision-language models means:

- **Duplicated pretraining**: The CLIP vision encoder, the diffusion U-Net or Transformer, and the LLM backbone are all trained separately on different objectives with different data, consuming enormous compute and engineering effort.
- **No shared representations**: A CLIP encoder trained for understanding cannot be used for generation, and a diffusion decoder trained for generation contributes nothing to understanding. Knowledge learned in one pipeline stays siloed.
- **Deployment complexity**: Production systems serving multimodal applications must orchestrate multiple models with different inference patterns (iterative denoising vs. autoregressive decoding), different hardware footprints, and different serving stacks.
- **No cross-task synergy**: Improvements to the generation model don't improve perception capabilities, and vice versa. The models cannot benefit from joint training on larger, more diverse multimodal corpora.

**The theoretical stakes are higher.** The success of next-token prediction in language — where it produced not just fluent text but emergent reasoning, in-context learning, and few-shot generalization — has made it the leading candidate for a general learning paradigm. If next-token prediction *cannot* handle multimodal data, that suggests a fundamental limitation: perhaps images and video require different inductive biases (iterative refinement, explicit alignment objectives) that autoregressive language modeling lacks. Conversely, if next-token prediction *can* handle all modalities, it strengthens the case that token-based autoregressive modeling is a sufficient substrate for general intelligence — not just language intelligence.

The paper explicitly connects to this broader narrative. The introduction references ChatGPT and its role in "sparking discussions about the early signs of artificial general intelligence." By showing that next-token prediction works for multimodal tasks at state-of-the-art levels, the paper is making a statement about the generality of the paradigm, not just reporting benchmark results.

---

### Prior Approaches and Where They Fall Short

The paper situates its contribution against three categories of prior work:

#### 1. Diffusion-Based Vision Generation

Diffusion models have been the dominant approach for image generation since the Stable Diffusion series demonstrated that latent diffusion could produce high-resolution, photorealistic images efficiently. The approach works by:
- Encoding images into a compressed latent space via a VAE.
- Training a denoising network (U-Net or Transformer) to iteratively remove noise from latents, conditioned on text embeddings from a pretrained text encoder (typically CLIP or T5).
- At inference, sampling random noise and denoising it into an image over 20-50 steps.

The strengths of this approach are well-documented: high-quality outputs, stable training, and a large open-source ecosystem. SDXL (Podell et al., 2023) represents the state of the art in open-source diffusion, and the paper uses it as its primary generation baseline.

However, diffusion models have structural limitations that Emu3 aims to overcome:
- **They require an external text encoder**: Diffusion models cannot natively process text — they rely on frozen CLIP or T5 embeddings as conditioning signals. This means they cannot be trained end-to-end on joint text-image objectives and cannot generate text.
- **They are inherently one-directional**: Diffusion generates images from noise conditioned on text, but the architecture does not support image-to-text or mixed-modal generation in a unified way.
- **The denoising objective is modality-specific**: The iterative denoising process has no natural analog for text generation, making it unsuitable as a general multimodal learning paradigm.

The paper acknowledges these limitations implicitly by targeting SDXL as the benchmark to beat. If a next-token prediction model can match SDXL's image quality, there is no architectural reason to prefer diffusion for generation.

#### 2. Compositional Vision-Language Understanding (CLIP + LLM)

The prevailing approach to vision-language understanding — exemplified by LLaVA, BLIP-2, InstructBLIP, and others — follows a now-standard recipe:
- Take a pretrained CLIP vision encoder (ViT-L or ViT-H) that maps images into a fixed-size embedding space aligned with language.
- Connect it to a pretrained LLM (Vicuna, LLaMA, etc.) via a lightweight adapter or projection layer.
- Fine-tune the combined system on image-text datasets and instruction-tuning data.

This approach has proven remarkably effective. LLaVA-1.6-7B, the paper's primary perception baseline, achieves strong results across a broad range of vision-language benchmarks (VQA, OCR, chart understanding, etc.) by training only the connector and partially fine-tuning the LLM. The entire pipeline leverages existing pretrained components.

But the compositional approach has fundamental limitations that motivate the search for alternatives:
- **The vision encoder and LLM are frozen in separate representational spaces**: CLIP was trained with contrastive objectives for alignment, not for feeding into autoregressive decoders. The adapter layer bridges this gap only partially, and the vision representations remain fundamentally different from the LLM's internal token representations.
- **No generation capability**: CLIP + LLM systems can describe images but cannot generate them. The representations are purely feedforward — images are encoded once and forgotten.
- **Two separate training pipelines**: The vision encoder and LLM are pretrained independently, sometimes on different data distributions with different objectives. Joint optimization is limited or impossible.
- **Architectural complexity**: The system requires three components (vision encoder, adapter, LLM) with different architectures, tokenization schemes, and training histories.

The paper specifically targets this fragmentation. In the related work section, it notes that while encoder-free architectures like Fuyu-8B and EVE have attempted to bypass the CLIP encoder by feeding image patches directly into LLMs, they "still face challenges in competing with state-of-the-art VLMs." Emu3 aims to be the first encoder-free model that *matches or exceeds* encoder-based performance.

#### 3. Prior Unified Multimodal Models (and Why They Weren't Enough)

The paper is not the first to attempt unifying vision and language under a single model. It explicitly engages with a lineage of prior unified approaches and explains why they fell short:

**Emu and Emu2** (Sun et al., 2023, 2024): These models introduced a unified autoregressive objective that predicts the next multimodal element — either a text token in a caption or a visual embedding in an image. However, Emu used regression to predict continuous visual features (through a separate visual decoder), making it a hybrid system rather than a pure token-based model. And while Emu2 improved on this design, neither matched the performance of task-specific systems like SDXL for generation or LLaVA for perception. As the paper states, these efforts "either resort to connecting LLMs with diffusion models or fail to match the performance of task-specific methods tailored for generation and perception."

**CM3Leon** (Yu et al., 2023): This model trained a token-based autoregressive model on mixed image and text data, closer to Emu3's approach. However, its performance on vision generation did not match diffusion models, and it used a two-stage training pipeline with separate image tokenizers for encoding and decoding.

**Chameleon** (Team, 2024): The most directly comparable prior work. Chameleon is a token-based mixed-modal early-fusion model trained autoregressively on interleaved text and image tokens. In principle, it shares Emu3's architectural philosophy. The paper's results show why it matters that Emu3 succeeds where Chameleon did not: in the GenEval benchmark (Table 7), Chameleon achieves an overall score of **0.39**, compared to Emu3's **0.66** (with rewriting) and SDXL's **0.55**. On vision-language tasks, Chameleon's reported VQAv2 accuracy is **69.6%** compared to Emu3's **75.1%**. The paper's contribution is not the *idea* of token-based multimodal modeling — Chameleon already explored that — but demonstrating that it can be made to work at performance levels that surpass task-specific architectures.

**Transfusion** (Zhou et al., 2024) and **Show-o** (Xie et al., 2024): These are more recent methods that attempt to combine diffusion and autoregressive approaches within a single model — essentially hedging between the two paradigms. The paper positions these as transitional designs that still retain architectural complexity from both worlds.

**VideoPoet** (Kondratyuk et al., 2023): An autoregressive approach to video generation, but one that "uses a two-stage generate-and-refine framework and an extra text encoder" — still compositionally complex.

---

### How This Paper Positions Itself

Given this landscape, Emu3's positioning is precise and ambitious:

**It is not proposing a new architectural innovation.** The model architecture is described as simply "retain[ing] the architectural framework of established large language models such as Llama-2, with the primary modification being the expansion of the embedding layer to accommodate discrete vision tokens." There is no novel attention mechanism, no hybrid objective, no clever fusion technique. The contribution is not architectural novelty.

**It is making a statement about the sufficiency of next-token prediction.** By stripping away every component that is not a tokenizer or a Transformer decoder, and demonstrating that the resulting system matches or beats diffusion-based generators and CLIP-based perceivers *at their own benchmarks*, the paper argues that the field's architectural fragmentation was unnecessary. The complexity of diffusion (noise schedules, guidance scales, iterative sampling) and composition (pretrained encoders, alignment objectives, adapter layers) is replaced by a single principle: convert everything to tokens, train to predict the next one.

This is a **scaling argument** as much as a technical one. The paper's conclusion states:

> "By simplifying complex model designs and focusing solely on tokens, it unlocks significant potential for scaling both during training and inference."

The implication is that the field's investment in modality-specific architectures has been a local optimum — one that worked well at current scales but that fundamentally limits further progress. If a simple token-based model can already match specialized systems, it stands to reason that scaling such a model (more data, more parameters, more compute) would eventually surpass them decisively, while diffusion and compositional systems hit diminishing returns from architectural complexity.

**The paper acknowledges the prior failures that make its success significant.** The introduction is explicit:

> "Despite early attempts at unifying generation and perception, such as Emu and Chameleon, these efforts either resort to connecting LLMs with diffusion models or fail to match the performance of task-specific methods tailored for generation and perception."

Emu3 represents the first time a pure next-token prediction model has matched or exceeded task-specific architectures. The prior failures are what make this breakthrough meaningful — they establish that success was not guaranteed and that Emu3 represents a genuine advance rather than an obvious application of known techniques.

**The open-source release is a deliberate strategic move.** The paper emphasizes that they "open-source key techniques and models to facilitate future research in this direction" and specifically calls out the vision tokenizer as something "which was previously publicly unavailable." This is not standard boilerplate — robust discrete vision tokenization has been a significant barrier to entry for token-based multimodal models. By releasing the tokenizer, the paper lowers the barrier for others to build on this approach, potentially accelerating the shift away from diffusion and compositional methods.

**The scope is deliberately comprehensive.** Rather than demonstrating success on a single modality or task, the paper covers image generation (with human evaluation), vision-language understanding (across twelve benchmarks), video generation (on VBench), and video future prediction — making the case that next-token prediction works across the full multimodal stack, not just in a narrow domain. This breadth is essential to the paper's thesis: if next-token prediction only worked for images or only for understanding, it wouldn't be a general paradigm.

## 3. Technical Approach

### 3.1 Reader Orientation

Emu3 is a single Transformer decoder that converts images, videos, and text into a shared vocabulary of discrete tokens and learns to predict the next token in mixed-modal sequences — the same training objective that powers modern LLMs. The problem it solves is architectural fragmentation: vision generation uses diffusion models, vision understanding uses CLIP-plus-LLM stacks, and these two pipelines share no components, no training objectives, and no representational space. The shape of the solution is aggressively minimal — take a standard Llama-style Transformer, give it a tokenizer that converts pixels into discrete codes, and train it on sequences that interleave images and text with a single instruction: "predict what comes next."

### 3.2 Big-Picture Architecture (Diagram in Words)

The system has five major stages, each responsible for one transformation in the pipeline from raw pixels to generated or understood content:

1. **Vision Tokenizer (SBER-MoVQGAN-based)**: Takes an image or video frame (a grid of continuous pixel values) and produces a sequence of discrete integer tokens, each drawn from a codebook of size 32,768. This is the only component that operates on continuous pixel data — everything downstream works with discrete tokens.

2. **Text Tokenizer (QwenTokenizer)**: Standard text tokenizer that converts multilingual text into discrete tokens, producing the shared vocabulary that vision tokens are added to.

3. **Data Construction Layer**: Assembles training sequences by interleaving text tokens and vision tokens using special delimiter tokens (`[BOS]`, `[SOV]`, `[SOT]`, `[EOV]`, `[EOS]`, `[EOL]`, `[EOF]`) into a single flat sequence that looks like a document. The sequence encodes both generation data (caption → image) and understanding data (image → caption).

4. **Transformer Decoder (8B-parameter Llama-style)**: The single model that processes all token sequences. It has no modality-specific branches, no separate encoders or decoders, no attention masks that isolate text from vision. It sees a flat sequence of tokens and predicts the next token at every position using a standard causal language modeling objective.

5. **Post-Training Modules**: After pretraining, two separate fine-tuning pipelines branch off: one for vision generation (quality fine-tuning + DPO alignment), and one for vision-language understanding (image-to-text training + instruction tuning). These share the pretrained backbone but optimize for different downstream tasks.

Information flows as follows: raw images/video frames enter the vision tokenizer → discrete vision tokens emerge → these are concatenated with text tokens using special delimiters → the flat sequence feeds into the Transformer decoder → the decoder predicts next tokens autoregressively → for generation, vision tokens are detokenized back to pixels; for understanding, text tokens are decoded to natural language answers.

### 3.3 Roadmap for the Deep Dive

- **First, the vision tokenizer (Sec. 2.2)** — because it is the single most critical enabling component and the one that makes everything else possible. Without a discrete compression of pixels into tokens, there is no unified token-based training. We will examine the architecture, training objectives, compression ratios, and reconstruction quality.

- **Second, the data preparation and sequence format (Sec. 2.1, 2.4)** — because the structure of training sequences determines what the model learns about the relationship between modalities. We will examine how images, videos, and text are filtered, captioned, and assembled into training documents with special delimiter tokens.

- **Third, the model architecture (Sec. 2.3)** — which is deliberately minimal. We will examine what makes this a standard LLM, the single modification made (expanded embedding layer), and the key hyperparameters.

- **Fourth, the pretraining procedure (Sec. 2.4)** — covering the two-stage training (text+images first, then add video with long context), the loss weighting strategy to prevent vision tokens from dominating, and the parallelism strategy.

- **Fifth, the post-training pipelines (Sec. 2.5)** — covering quality fine-tuning, DPO for generation alignment, and the two-stage understanding fine-tuning.

- **Sixth, video-specific mechanisms** — including the tokenizer's temporal compression, the video data processing pipeline (scene detection, flow filtering, aesthetic scoring), and the autoregressive future-prediction capability.

### 3.4 Detailed, Sentence-Based Technical Breakdown

This is primarily a **systems and scaling paper** whose core idea is that a single next-token prediction Transformer, trained on discrete tokens from all modalities, can match or exceed specialized architectures (diffusion for generation, CLIP+LLM for perception) without any modality-specific inductive biases. The contribution is not a novel architecture but a demonstration that architectural simplicity suffices when the tokenizer and training recipe are done right.

---

#### Vision Tokenizer: Converting Pixels to Discrete Codes

The vision tokenizer is the linchpin of Emu3's unified approach. It must solve a hard compression problem: take a high-resolution image (e.g., 512×512 RGB pixels, which is 786,432 continuous values) and represent it as a short sequence of discrete tokens that (a) preserves enough visual detail for high-fidelity reconstruction, (b) produces sequences short enough that the Transformer can process them in context, and (c) produces tokens that are learnable by a standard language model objective.

**Base architecture.** The tokenizer is built on SBER-MoVQGAN, a vector-quantized variational autoencoder (VQ-VAE) architecture originally developed for image compression. The core idea of a VQ-VAE is:

- An **encoder** compresses the input into a lower-dimensional latent representation.
- Each spatial position in the latent grid is mapped to the nearest vector in a learned **codebook** — a fixed-size dictionary of embedding vectors. The output is no longer continuous but a sequence of discrete indices into the codebook.
- A **decoder** reconstructs the image from these discrete codebook entries.

The paper uses a specific pretrained checkpoint: **SBER-MoVQGAN-270M** with a codebook of size **32,768** and a latent spatial compression of **4×**. These numbers matter:

- **Codebook size 32,768**: This is unusually large for a VQ-VAE (many use 1024–8192). A larger codebook means each token can represent more visual information, which is essential for high-fidelity generation since the model is trying to capture everything from textures to object shapes to text glyphs in discrete symbols. The 32,768 vision tokens are added to the text vocabulary of the QwenTokenizer to form a total vocabulary of **184,622** tokens for the Transformer.

- **Latent size 4×**: This means the encoder produces a latent grid where each spatial position corresponds to a 4×4 region of the input image. For a 512×512 input, the latent grid is 128×128 = 16,384 positions. Each position is mapped to a codebook entry, so a single 512×512 image becomes 16,384 tokens.

- **Additional spatial compression 8×8**: The tokenizer applies further compression in the spatial dimensions, yielding **4×8×8** total compression (temporal × height × width). This means a 512×512 image is compressed into **4,096 tokens** (not 16,384) — the 4× latent dimension compression multiplied by 8×8 spatial compression = effectively 32× total compression in spatial dimensions, or 4× temporal × 8×8 spatial = 256× overall for video.

Wait — let me reconcile the numbers. The paper states in Section 2.2:

> "We train the vision tokenizer based on SBER-MoVQGAN, which can encode a 4×512×512 video clip or a 512×512 image into 4096 discrete tokens from a codebook of size 32,768. Our tokenizer achieves 4× compression in the temporal dimension and 8×8 compression in the spatial dimension."

For a **single 512×512 image**: The input is 512×512×3 = 786,432 values. With 8×8 spatial compression, the latent grid is 64×64 = 4,096 positions. Each position maps to one token. So **one image = 4,096 tokens**. This is the key number — images become sequences of 4,096 discrete codes.

For a **4×512×512 video clip** (4 frames at 512×512): The input is 4×512×512×3 values. With 4× temporal compression, the 4 frames become 1 temporal unit. With 8×8 spatial compression, each frame becomes 64×64 = 4,096 positions. So one temporal unit × 4,096 = **4,096 tokens for 4 frames**. This means each frame averages to approximately 1,024 tokens.

**Video-specific modifications.** Building on the base MoVQGAN architecture, the authors add **two temporal residual layers with 3D convolution kernels** to both the encoder and decoder. This is the only architectural customization to the tokenizer. The 3D convolutions allow the encoder to compress across frames (not just within them), learning motion-aware representations. Without these layers, the tokenizer would treat each frame independently and tokens would not capture temporal coherence across frames.

**Training objective.** The tokenizer is trained end-to-end on the LAION-High-Resolution image dataset and the InternVid video dataset using a combined loss:

The paper states the objective contains **L2 loss, LPIPS perceptual loss, GAN loss, and commitment loss**. Let me unpack each:

- **L2 loss**: Pixel-level mean squared error between the original and reconstructed image. This encourages accurate low-level reconstruction but produces blurry results on its own because MSE penalizes high-frequency detail.

- **LPIPS perceptual loss** (Learned Perceptual Image Patch Similarity): Instead of comparing pixels directly, this loss compares the original and reconstruction through the internal activations of a pretrained network (AlexNet in this case, per Section 2.2). Perceptual loss captures what humans perceive as similar — it allows the reconstruction to have different pixel values that look the same to a human observer. This is critical for compression: the tokenizer can discard pixel-level noise while preserving what matters perceptually.

- **GAN loss** (adversarial loss): A discriminator network is trained to distinguish real images from reconstructions, while the tokenizer is trained to fool it. GAN loss pushes reconstructions toward the manifold of natural images, adding realistic texture that L2 and LPIPS alone would miss.

- **Commitment loss**: Encourages the encoder to produce embeddings that lie close to codebook entries, reducing the quantization error from rounding to the nearest codebook vector. Without commitment loss, the encoder might produce embeddings far from any codebook entry, causing large reconstruction errors.

**Why these four losses together?** Each addresses a different failure mode of discrete compression. Pure L2 produces blur. Perceptual loss fixes blur but can introduce artifacts. GAN loss adds realism but can hallucinate details. Commitment loss keeps the quantization step from destroying information. The combination is standard in VQ-VAE training but essential for making the tokens information-rich enough that a language model can learn useful things from them.

**Reconstruction quality (Figure 3, Table 2).** The paper provides both qualitative and quantitative evidence that the tokenizer preserves visual information well. The reconstruction samples in Figure 3 show original and reconstructed images/videos side-by-side — visually, they are near-indistinguishable at typical viewing distances.

Table 2 reports quantitative metrics on a 3,172-video evaluation set from Pexels, reconstructed over 5 seconds while maintaining aspect ratios. The metrics are:

- **LPIPS↓**: Lower is better. LPIPS measured with AlexNet features ranges from 0.099 (at 128×128 resolution) to 0.110–0.112 (at higher resolutions). These are very low values indicating high perceptual similarity.

- **PSNR↑**: Higher is better. Ranges from 21.59 (256×256) to 24.30 (720×720). PSNR in the low-to-mid 20s is typical for lossy image compression at these compression ratios.

- **SSIM↑**: Higher is better. Ranges from 0.622 (256×256) to 0.771 (720×720). SSIM above 0.7 indicates good structural preservation.

Importantly, the metrics improve or stay flat at higher resolutions — the 720×720 reconstruction has the best PSNR (24.30) and SSIM (0.771). This means the tokenizer scales well to higher resolutions, which matters because post-training uses up to 720-pixel images for generation.

**Why this tokenizer is a contribution in itself.** The paper emphasizes that a "robust vision tokenizer, enabling the transformation of videos and images into discrete tokens" was "previously publicly unavailable." Pre-Emu3, researchers wanting to build token-based multimodal models either had to train their own tokenizer from scratch (expensive, difficult, and requiring expertise in VQ-VAE training dynamics) or use one not designed for this purpose. By open-sourcing the tokenizer, the paper provides a key infrastructure component that enables others to replicate and extend this line of work.

---

#### Data: What the Model Sees and How It's Prepared

The data pipeline is arguably the most important non-architectural component. The model's ability to generate high-quality images and answer questions about images depends entirely on the quality, diversity, and structure of the training data. The paper provides substantial detail on both the filtering and the annotation pipelines.

##### Language Data

The language component uses the same corpus as Aquila, described as "a high-quality corpus consisting of both Chinese and English data." No further details are provided on token counts, filtering criteria, or preprocessing. This is consistent with the paper's focus — the language data is not where the contribution lies. The key point is that Emu3 uses standard, off-the-shelf language data without any special multimodal alignment preprocessing.

##### Image Data Collection and Filtering

The image dataset is curated through a multi-stage pipeline described in Section 2.1:

**Sources.** The dataset comprises three categories: open-source web data, AI-generated data, and "high-quality in-house data." The inclusion of AI-generated images is notable — it means the model is trained partly on synthetic data, which is common practice in image generation but raises questions about data provenance and the risk of amplifying artifacts from other models.

**Resolution filter.** Samples with resolution below 512×512 pixels are discarded. This ensures that every image the model sees has sufficient detail for high-quality generation. The choice of 512 as the threshold aligns with the tokenizer's native resolution — the model learns to generate at roughly the resolution it is trained on.

**Aesthetic filter.** Each image is scored using the LAION-AI aesthetic predictor, a classifier trained to predict human aesthetic ratings. Images with scores below **5.5** are excluded. The aesthetic predictor uses a threshold above the neutral midpoint (which would be 5 on a 1–10 scale) to ensure above-average visual quality. This filter is aggressive — it removes images that are technically adequate (in terms of resolution) but visually unappealing (poor composition, bad lighting, cluttered scenes).

**Recall-oriented recovery.** The paper notes a subtlety: some high-quality open-world images fail the aesthetic filter not because they are ugly but because they contain text or unusual color distributions that confuse an aesthetic classifier trained primarily on artistic photographs. To recover these, the pipeline applies:

- **Text detection** (via PaddleOCR): Images with detected text are retained rather than discarded, on the grounds that text-rich images (posters, infographics, screenshots) are important for visual understanding even if an aesthetic classifier rates them poorly.

- **Color filtering**: Non-monochromatic images and those with minimal text are retained. This prevents the dataset from being dominated by the kind of high-saturation, high-contrast images that aesthetic classifiers prefer.

This two-stage filtering (harsh aesthetic threshold, then targeted recovery) is a common pattern in dataset curation — the initial aggressive filter removes obvious low-quality data, then handcrafted rules recover specific categories that the filter over-rejects.

**Understanding-focused supplementary data.** Following the DenseFusion data processing pipeline, the authors extract "millions of representative images" covering charts, tables, text-rich content, and other categories not well-represented in standard web-scraped image datasets. This supplementary data is essential for vision-language understanding benchmarks (ChartQA, DocVQA, InfoVQA) that test exactly these categories. Without this targeted data collection, a general-purpose model would perform poorly on document and chart understanding tasks.

##### Image Captioning

The quality of text descriptions associated with images is critical for text-to-image generation (the model must learn the mapping from text to pixels) and for image understanding (the model must learn to produce text descriptions). The paper builds a **dense synthetic captioning pipeline**:

**Step 1: GPT-4V annotation.** The team uses GPT-4V with "detailed prompts" to generate approximately **1 million image-caption pairs**. These are high-quality captions — GPT-4V can describe not just objects but relationships, attributes, text in images, and compositional structure. A million such pairs provides a strong seed dataset.

**Step 2: Fine-tune Emu2-17B as captioner.** The 1M GPT-4V captions are used to fine-tune Emu2-17B, a 17-billion-parameter multimodal model. The fine-tuned model becomes a specialized captioner that can generate dense, GPT-4V-quality captions at scale without the cost and rate limits of calling an external API.

**Step 3: Large-scale captioning with vLLM.** The fine-tuned captioner is deployed using vLLM, a high-throughput inference library with PagedAttention. This allows captioning millions of images efficiently. The paper doesn't specify the total number of captioned images, but the implication is that the entire filtered image dataset receives synthetic captions.

**Why synthetic captions?** Web-scraped image-text pairs (alt-text, social media captions) are notoriously noisy — they may describe things not in the image, omit critical details, or be in the wrong language. Synthetic captions from GPT-4V or a GPT-4V-fine-tuned captioner are more detailed, more consistent, and more aligned with the actual image content. The tradeoff is that synthetic captions may have systematic biases (GPT-4V's particular style, its tendency to over-describe or hallucinate) that get baked into the model. The paper does not discuss this risk.

##### Video Data Collection and Filtering

Video data curation is substantially more complex than image curation because videos have a temporal dimension that must be preserved, and raw video files are orders of magnitude larger than images. The paper describes an elaborate four-stage pipeline:

**Stage 1: Scene detection and splitting.** Using PySceneDetect, raw videos are split into coherent scenes. The tool uses two detectors:

- **ContentDetector**: Identifies hard cuts by detecting abrupt changes in frame content (comparing HSV histograms between frames). Each detected cut becomes a scene boundary.
- **ThresholdDetector**: Identifies fade-in and fade-out events by detecting frames where the average pixel intensity crosses a threshold. This catches gradual transitions that ContentDetector would miss.

Splitting into scenes before further filtering ensures that each training clip represents a coherent visual event (a person walking, a pan across a landscape) rather than an arbitrary temporal window that might span multiple unrelated scenes.

**Stage 2: Text detection and removal.** PaddleOCR is applied to detect text coverage in each clip. Clips with "excessive text coverage" are removed. The motivation: text-heavy videos (screen recordings, title sequences, news tickers) are poor training data for learning visual motion and scene dynamics. The filtering is done at low resolution (frames sampled at 2 FPS, shorter edge resized to 256) to manage computational cost.

**Stage 3: Optical flow filtering.** This stage removes clips with either too little motion (static scenes) or too much motion (shaky camera, rapid cuts) using RAFT, a state-of-the-art optical flow method:

- Frames are sampled at 2 FPS and resized (shorter edge to 256) for efficiency.
- Optical flow is computed between consecutive frames, producing a flow magnitude per pixel.
- The **flow score** is defined as the **ratio between the average flow magnitude across all pixels and the shorter edge length**.
- Clips with flow scores outside an **acceptable range** are discarded.

Defining the flow score as a ratio to image size normalizes for resolution — pure pixel displacement would be larger in higher-resolution videos even for the same physical motion. The "acceptable range" is not specified numerically in the paper, but the distribution of retained clips is shown in Figure 10 of the appendix, with flow scores mostly in the 0.04–0.15 range (the mode is 0.04–0.05).

**Stage 4: Aesthetic quality assessment.** The LAION-AI aesthetic predictor is applied to each clip. Three frames are sampled per clip, each receiving a score. A clip is retained only if its **lowest score** (the worst of the three) is **≥ 5**. Using the minimum rather than the average is a conservative filtering strategy — a video that is mostly beautiful but has one ugly frame is discarded. This ensures that every frame the model might attend to has acceptable quality.

**Why filter on the minimum?** If the model learns to generate videos autoregressively, any degradation at any timestep will be visible and compound in future frames. A video with 2 beautiful seconds and 1 ugly second is not high-quality — the ugly second ruins the viewing experience and provides poor training signal for the temporal model.

##### Video Captioning

The video captioning pipeline mirrors the image captioning setup but adds the complexity of describing motion and temporal structure:

**Seed annotation.** For each video clip, 8 frames are sampled and composed into a prompt for GPT-4V. The prompt explicitly asks the model to "describe both the content and motion within these frames." Some of the labeled data undergoes "manual revision" — human annotators correct GPT-4V's mistakes or add missing details. This is an important detail: pure LLM-as-labeler approaches can produce systematically flawed training data, and even a small amount of human correction helps.

**Video captioner training.** The image captioner (Emu2-17B fine-tuned on GPT-4V captions) is further fine-tuned on the video captioning data to produce a video-specific captioner. This is a cost-saving move — training a dedicated video captioner from scratch would require many more GPT-4V calls.

**Large-scale deployment.** Captioning is accelerated with vLLM. Clips shorter than 20 seconds are captioned using 12 evenly sampled frames. Longer clips are split into 10–20 second sub-clips, each captioned independently. This splitting avoids hitting context limits (12 frames from 20 seconds at 24 FPS = 480 frames if all were included, but only 12 are sampled, so the splitting is about maintaining temporal density of sampling, not about raw frame count).

##### Training Sequence Format

The pretraining data is assembled into a specific token sequence format that the model learns to interpret:

```
[BOS] {caption text} [SOV] {meta text} [SOT] {vision tokens} [EOV] [EOS]
```

The special tokens serve as structural markup that the model uses to understand what it is processing:

- **`[BOS]`**: Beginning of sequence. Inherited from the standard text tokenizer.
- **`[EOS]`**: End of sequence. Also from the standard tokenizer.
- **`[SOV]`**: Start of Vision. Signals that vision-related information follows. This allows the model to switch its expectations — after `[SOV]`, it should expect metadata and then pixel tokens.
- **`[SOT]`**: Start of Tokens. Marks the boundary between metadata and the actual vision tokens. Everything between `[SOV]` and `[SOT]` is text-based metadata; everything between `[SOT]` and `[EOV]` is vision tokens.
- **`[EOV]`**: End of Vision. Signals that the vision sequence is complete and the model is returning to text mode (or sequence end).
- **`[EOL]`**: End of Line. Inserted within the vision token sequence to denote line breaks — i.e., transitions between rows in the 2D token grid. This preserves some spatial structure even though the tokens are flattened into a 1D sequence.
- **`[EOF]`**: End of Frame. Inserted within the vision token sequence to denote frame breaks in video — the boundary between one frame's tokens and the next.

**Why these delimiters matter.** Without them, the model would see an undifferentiated stream of tokens with no way to know whether a given integer corresponds to the word "cat" in text or a visual feature representing fur texture. The delimiters provide the syntax that lets the model parse the multimodal document structure. They are analogous to HTML tags or markdown markup — they don't carry content themselves but tell the system how to interpret what follows.

**The `{meta text}` field.** This is a clever piece of design. For images, the metadata contains the **resolution** (e.g., "512×512") in plain text. For videos, it contains **resolution, frame rate, and duration** (e.g., "512×512, 24 FPS, 5 seconds"). Including these as readable text rather than special tokens means:

- The model can learn to *generate* images at specific resolutions by conditioning on the metadata. If the user asks for a wide image, the model can be prompted with a resolution like "1024×512" in the meta text.
- The resolution is learned through natural language, leveraging the model's existing text understanding rather than requiring special resolution-conditioning mechanisms (which diffusion models typically handle through positional encodings or extra conditioning inputs).
- During training, the model learns to predict vision tokens of appropriate length given the stated resolution — a form of self-consistency that improves generation quality.

**Data ordering for generation vs. understanding.** The paper describes two patterns:

- **Generation data**: caption text → `[SOV]` → metadata → `[SOT]` → vision tokens → `[EOV]`. The model learns to generate images conditioned on captions.
- **Understanding data**: `[BOS]` → `[SOV]` → metadata → `[SOT]` → vision tokens → `[EOV]` → caption text → `[EOS]`. The caption text is moved to *after* the vision tokens. The model learns to generate descriptions conditioned on images.

This is created by moving the "caption text" field in "a portion of the dataset" to follow the `[EOV]` token. The model sees both patterns during training, which teaches it both directions: text→image and image→text. This is what enables the same model to handle both generation and perception tasks — it has learned the bidirectional relationship between visual content and textual descriptions through exposure to both orderings.

---

#### Model Architecture: Standard by Design

The architecture section is notable for its brevity. The paper states:

> "The Emu3 model retains the architectural framework of established large language models (LLMs) such as Llama-2, with the primary modification being the expansion of the embedding layer to accommodate discrete vision tokens."

This is the architectural equivalent of saying "we changed nothing except what was absolutely necessary." Every component is standard:

**Normalization: RMSNorm.** Root Mean Square Layer Normalization normalizes activations using only the root mean square statistic (not mean and variance separately). It is computationally cheaper than LayerNorm and has become standard in LLMs since Llama and Chinchilla.

**Attention: GQA (Grouped Query Attention).** Standard multi-head attention uses separate key, query, and value projections per head. Multi-Query Attention (MQA) shares keys and values across all heads, reducing memory bandwidth but potentially hurting quality. GQA is a middle ground: heads are grouped, and within each group, keys and values are shared. With **32 heads** and **8 KV heads** (Table 3), each KV head serves 4 query heads. This reduces the KV cache size by 4× compared to full multi-head attention, which is critical for handling the long sequences (up to 131,072 tokens) needed for video data.

**Activation: SwiGLU.** A gated variant of the GLU (Gated Linear Unit) activation that has been shown to outperform ReLU and GeLU in Transformer feedforward layers. It uses a learned gating mechanism: output = (xW₁ ⊙ Swish(xW₂))W₃, where ⊙ is element-wise multiplication and Swish is a smooth activation. The **intermediate size of 14,336** (Table 3) means the feedforward layer expands the hidden dimension by a factor of 3.5× (14,336 / 4,096), which is standard for this architecture family.

**Position encoding: RoPE (Rotary Positional Embeddings).** RoPE encodes position by rotating the query and key vectors in attention by an angle proportional to their position. The **RoPE base of 1,000,000** (Table 3) is 100× larger than the original RoPE paper's base of 10,000. A larger base frequency means the positional encoding varies more slowly with distance, which extends the effective context length that the model can distinguish — critical for the 131,072-token context used in video training.

**Other details:**
- **Biases removed** from qkv and linear projection layers. This is a common optimization in modern LLMs — biases in attention projections contribute negligibly to model quality but add memory and compute.
- **Dropout rate of 0.1** is applied during training. The paper states this is "to improve training stability" — dropout serves as regularization and helps prevent co-adaptation of features, especially important when training from scratch on a new modality mix.

**The only architectural modification: embedding layer expansion.** The embedding matrix maps from token IDs to hidden vectors (size 4,096). Adding 32,768 vision tokens to the vocabulary means the embedding layer grows to accommodate the new tokens — the same weight matrix now has 184,622 rows instead of whatever the original text vocabulary was. These new rows are randomly initialized and learned during pretraining. Everything else — the Transformer blocks, the attention heads, the feedforward layers, the output projection — is identical to a standard LLM.

**Why this matters.** The architectural minimalism is a deliberate philosophical statement. If Emu3 used a novel attention mechanism, a modality-specific encoder, or a hybrid loss function, its strong results could be attributed to those innovations rather than to the core hypothesis. By changing nothing except the embedding table, the paper isolates the variable: the training data (mixed-modal token sequences) and the training objective (next-token prediction). Any success must be attributed to these factors, not to architectural cleverness.

---

#### Pretraining: Training the Unified Model from Scratch

The pretraining procedure is where the model actually learns to understand and generate across modalities. The key design decisions are: how to handle the unequal information density of text vs. vision tokens, how to schedule the introduction of modalities, and how to manage the enormous context lengths required for video.

##### Training Objective: Weighted Next-Token Prediction

The model is trained with the standard autoregressive language modeling objective:

$$\mathcal{L} = -\sum_{t=1}^{T} w_t \cdot \log p(x_t | x_{<t})$$

where $x_t$ is the token at position $t$, $x_{<t}$ are all preceding tokens, $p(x_t | x_{<t})$ is the model's predicted probability for the correct token, $T$ is the sequence length, and $w_t$ is a token-type-dependent weight.

**What it computes:** For each position in the sequence, the model receives a cross-entropy loss between its predicted distribution over the vocabulary and the actual next token. The per-token loss is weighted: **vision tokens receive a weight of 0.5, text tokens a weight of 1.0**. The total loss is the sum over all positions.

**Why the vision token weight of 0.5?** The paper states this is "[t]o prevent vision tokens from dominating the learning process." The concern is specific to the token counts: a single 512×512 image produces 4,096 vision tokens, while a typical caption might be 20-50 text tokens. Without reweighting, the total loss from a single image-caption pair would be dominated 80:1 or more by the vision component. The model would optimize heavily for pixel-level reconstruction at the expense of learning the relationship between text and images. The 0.5 weight rebalances this — it doesn't make text and vision equally weighted (text would need a weight of roughly 80× to achieve parity), but it reduces the vision dominance enough that text learning is not drowned out.

This is a crude but effective mechanism. More sophisticated approaches (like separate loss terms with adaptive weighting, or per-modality normalization) might work better but add complexity. The paper's approach is characteristically simple.

##### Two-Stage Pretraining

Pretraining is split into two stages with different data mixes and context lengths:

**Stage 1: Text + Images Only (No Video)**

- **Context length: 5,120 tokens.** This is sufficient for image-text sequences (4,096 tokens per image + caption + special tokens) but not for video, which requires many more tokens.
- **Data: Text and image data only.** No video data is introduced yet.
- **Training from scratch.** Model weights are randomly initialized.
- **Learning rate: $5 \times 10^{-5}$** with cosine annealing to zero.

The progression from short context to long context is a well-established LLM training technique. Training with 5K context first is computationally much cheaper than training with 131K context from the start (attention cost scales quadratically with sequence length in principle, though FlashAttention and other optimizations reduce this). The model learns basic multimodal associations at this stage — what objects look like, how captions describe images — without the additional complexity of temporal dynamics.

**Stage 2: Add Video Data**

- **Context length: 131,072 tokens.** This massive expansion allows the model to process video sequences. At 24 FPS with 4× temporal compression, 1 second of video produces about 6 "temporal units" worth of tokens. The tokenizer encodes 4 frames into ~4,096 tokens, so 24 frames (1 second at 24 FPS) would be ~6 × 4,096 = 24,576 tokens. A 5-second video at 24 FPS would be ~122,880 tokens — nearly filling the context window. The 131K context length is clearly designed around the need to fit several seconds of full-resolution video + text conditioning.
- **Data: Text, images, and now video data.** The model learns temporal dynamics, object motion, camera movement, and scene transitions.
- **Learning rate: Same scheme ($5 \times 10^{-5}$ with cosine annealing to zero).**
- **Training continues from Stage 1 checkpoint.** This is not a separate training — it's a continuation with expanded context and added data modality.

**Why introduce video later?** Several practical and pedagogical reasons:

1. **Computational cost:** Training with 131K context is enormously expensive. By first training with short context, the model learns basic visual concepts cheaply, and only the final training phase pays the long-context cost.
2. **Learning curriculum:** Images are static; videos add motion. Learning what a cat looks like from images before learning how a cat moves is a natural curriculum that may improve learning efficiency.
3. **Stability:** Adding video data abruptly at the start of training might cause loss spikes. By Stage 2, the model has already converged to a good representation of static visuals, and video training refines this to include temporal dynamics.

##### Data Packing Strategy

To maximize hardware utilization, the paper uses **sequence packing**: multiple training examples are concatenated into a single sequence that fills the maximum context length. This avoids wasted computation from padding short sequences to the maximum length.

The paper adds a constraint specific to multimodal packing: **"ensuring that complete images are not segmented during the packing process."** This means an image's 4,096 tokens are always contiguous and never split across two packed sequences. If the remaining space in the current sequence is less than 4,096 tokens, the image is placed in the next sequence instead.

**Why this constraint matters.** If an image's tokens were split, the model would see incomplete visual information in one sequence and the continuation in another. The semantic boundaries of the image would not align with the sequence boundaries, potentially confusing the model about what constitutes a complete visual concept. Keeping images whole ensures that each training example presents a complete visual entity.

##### Parallelism Strategy

Training a model with 131K context length requires distributed computation. The paper uses:

- **Tensor parallelism (TP):** Splits individual weight matrices across GPUs. The large feedforward layers (intermediate size 14,336) are partitioned, with each GPU computing a slice.
- **Context parallelism (CP):** Splits the long sequence dimension across GPUs. Different GPUs process different segments of the 131K-token sequence, communicating attention keys/values at segment boundaries. This is essential because the attention matrix for a 131K sequence is ~17 billion elements, which cannot fit on a single GPU.
- **Data parallelism (DP):** Standard replication of the model across GPUs, each processing a different batch.

The combination of all three parallelism types is necessary because no single strategy handles all bottlenecks: TP addresses per-layer weight size, CP addresses sequence length, and DP addresses batch size.

---

#### Post-Training for Vision Generation

After pretraining, the model can generate images and videos but may not produce aesthetically pleasing or prompt-aligned outputs. The pretraining objective optimizes for token prediction accuracy, not visual quality or human preference. Two post-training stages address this gap.

##### Quality Fine-Tuning (QFT)

The model is fine-tuned on a curated set of high-quality images and videos, with supervision applied **only to the vision tokens** (not the text tokens). This focuses the model's learning capacity on improving generation quality without altering its language understanding.

**Data selection.** Images and videos are selected from "diverse high-quality sources" and filtered using the average of three preference scores:

- **HPSv2.1** (Human Preference Score v2): A model trained to predict human preferences for text-to-image outputs. Higher scores mean humans prefer the image.
- **MPS** (Multi-dimensional Preference Score): Similar purpose but evaluates on multiple dimensions rather than a single aesthetic quality.
- **LAION Aesthetics score**: The same predictor used in the data filtering pipeline.

By requiring images to score well on all three metrics (averaged), the QFT dataset consists of images that are consistently judged as high-quality across different evaluation criteria. The paper does not specify the threshold for inclusion.

**Resolution increase.** During QFT, training resolution is increased from **512 pixels to 720 pixels** (on the shorter side, maintaining aspect ratio). This allows the model to generate higher-resolution images with finer detail. The tokenizer was trained to handle this — Table 2 shows reconstruction quality remains high at 720×720.

**Learning rate annealing.** At the end of QFT, the learning rate is linearly decayed to zero using an "annealing strategy." Annealing at the end of fine-tuning is standard practice — it allows the model to settle into a good local minimum rather than bouncing around due to a constant learning rate.

**What QFT actually teaches the model.** During pretraining, the model sees a wide distribution of image qualities (despite the aesthetic filtering at 5.5, there is still substantial variation). QFT biases the model toward the high-quality end of this distribution. The model learns that the visual tokens it should produce are those typical of aesthetically pleasing images — better composition, more vibrant colors, fewer artifacts. This is a form of distribution shift: the model's output distribution is moved from the average of the pretraining data toward the average of the QFT data.

##### Direct Preference Optimization (DPO)

DPO is a fine-tuning technique that directly optimizes the model to prefer "chosen" outputs over "rejected" ones, without training a separate reward model (as RLHF does). The paper adapts DPO for autoregressive vision generation.

**The DPO loss function** (implicitly used, though the full equation is not given in the paper):

$$\mathcal{L}_{\text{DPO}} = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}}\left[\log \sigma\left(\beta \log \frac{\pi_\theta(y_w | x)}{\pi_{\text{ref}}(y_w | x)} - \beta \log \frac{\pi_\theta(y_l | x)}{\pi_{\text{ref}}(y_l | x)}\right)\right]$$

where $\pi_\theta$ is the model being trained, $\pi_{\text{ref}}$ is the reference (frozen) model (typically the QFT checkpoint), $y_w$ is the "winning" (chosen) output, $y_l$ is the "losing" (rejected) output, $x$ is the prompt, $\beta$ is a temperature parameter controlling how far $\pi_\theta$ can deviate from $\pi_{\text{ref}}$, and $\sigma$ is the sigmoid function.

**What it computes:** For each prompt, the model receives a chosen and a rejected image. The loss increases when the model assigns higher probability to the rejected image than the chosen one, scaled by how much the reference model already distinguishes them. The log-sigmoid form ensures the loss is well-behaved: when the model correctly prefers the chosen image by a large margin (relative to the reference), the loss saturates near zero.

**Why this form:** DPO is mathematically equivalent to RLHF under the Bradley-Terry preference model but avoids training a separate reward model. The reference model term $\log \frac{\pi_\theta}{\pi_{\text{ref}}}$ acts as an implicit KL penalty, preventing the model from overfitting to the preference data by drifting too far from its QFT initialization.

**Preference data construction.** The paper describes a three-step process:

1. **Generate candidates:** For each user-collected prompt $p$, the QFT model generates **8–10 images** at different random seeds. This creates a pool of diverse outputs for the same prompt.
2. **Human evaluation:** Three independent voters rate each generated image on two criteria: **visual appeal** (how aesthetically pleasing the image is) and **prompt alignment** (how well the image follows the text description). The three scores are aggregated.
3. **Pair construction:** The highest-scoring image is selected as the "chosen" ($y_w$), and the lowest-scoring image as the "rejected" ($y_l$). This forms a preference triplet $(p, y_w, y_l)$.

**Token storage strategy.** A crucial implementation detail: "the tokens from the data construction process are stored for direct use in future training phases." This means the exact token sequences that were generated during candidate creation are saved, rather than being re-tokenized from the image during DPO training. This avoids **reconstruction differences** — small variations introduced by encoding an image to tokens and then decoding it back, which could create discrepancies between what the evaluators saw and what the model is being trained on.

**Combined DPO loss.** Emu3-DPO minimizes **both** the DPO loss and the standard next-token prediction cross-entropy loss. This is a hybrid objective:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{DPO}} + \mathcal{L}_{\text{NTP}}$$

The NTP loss term serves as a regularizer, preventing the model from degenerating into producing only the narrow distribution of "chosen" images. It acts as an anchor to the broader data distribution, similar to how RLHF implementations often include a KL penalty to the pretrained model.

**Results of DPO (Figure 6).** The human evaluation shows DPO improves both **visual quality** (from 52.3 to 60.6 on the human preference score scale) and **prompt alignment** (from 57.3 to 61.6). These are meaningful improvements, though the absolute scores suggest there is still substantial room for improvement.

**The automated metric decline (Table 4).** Interestingly, Emu3-DPO scores slightly *lower* than Emu3 on automated benchmarks (GenEval overall: 0.64 vs. 0.66 with rewriting; T2I-CompBench average: 0.7164 vs. 0.7422). The paper attributes this to a **distribution mismatch**: the human preference data used for DPO training emphasizes "overall aesthetic quality," which may differ from what automated evaluation metrics (like object detection-based counting or CLIP-score alignment) measure. This is a known phenomenon in RLHF/DPO — optimization against one preference signal can reduce performance on metrics that weren't part of the preference data. It also illustrates why human evaluation is essential for generation tasks: automated metrics capture only narrow aspects of quality.

---

#### Post-Training for Vision-Language Understanding

The understanding pipeline is a separate post-training branch that teaches the pretrained model to answer questions about images. This is a two-stage process:

##### Stage 1: Image-to-Text Training

The model is fine-tuned on **image understanding data mixed with pure language data**. The key design choice:

- **Losses on vision tokens are disregarded.** During this stage, the model sees images followed by text (the understanding data format: vision tokens → caption/QA). The loss is computed only on the text portion — the model learns to condition on images and produce relevant text, but it is not penalized for "incorrect" vision token predictions (since it's not generating images in this stage).

- **Image resolution:** Each image is resized to approximately 512×512 while preserving aspect ratio. The approach: scale the image so that its area is close to 512², then round dimensions to tokenizer-compatible values. This ensures the vision token sequence length is consistent and comparable to what the model saw during pretraining.

The inclusion of **pure language data** in this stage is important. Without it, the model might "catastrophically forget" its language capabilities while learning to answer visual questions. The language data acts as a regularizer, maintaining the model's general linguistic competence.

##### Stage 2: Instruction Tuning

The model is further fine-tuned on a **subset of question-answer pairs** from an existing vision-language instruction tuning dataset (cited as Li et al., 2024 in the references, labeled as [44] — likely LLaVA-OneVision). This teaches the model the specific format of vision-language instructions: user asks a question about an image, assistant provides the answer.

**Resolution handling during instruction tuning.** The paper describes a resolution policy:
- Images below **512×512** are resized **up** to 512 on the shorter side.
- Images above **1024×1024** are resized **down** to 1024 on the shorter side.
- Images between these bounds maintain their **original resolution**.

This policy ensures that the model sees diverse resolutions during instruction tuning while bounding the extremes. The upper bound of 1024 prevents vision token sequences from becoming excessively long (a 1024×1024 image would be ~8,192 tokens with the tokenizer), which would reduce the number of training examples that fit in a batch and increase training cost.

**Why this two-stage approach?** The image-to-text stage teaches the model to produce accurate descriptions and answer factual questions about images. The instruction tuning stage teaches the model the conversational format: recognizing questions, producing answers in the expected style, and handling the diversity of question types found in instruction-tuning datasets (multiple choice, open-ended, yes/no, etc.). Separating these stages allows each to focus on a different learning goal — content accuracy first, format compliance second.

---

#### Video Generation and Future Prediction Mechanisms

Emu3's video capabilities are a direct consequence of its unified token-based architecture and require no additional components beyond what is described above. The key aspects:

##### Video as Token Sequences

Video is tokenized frame by frame (with the 3D temporal convolutions in the tokenizer providing cross-frame compression). A 5-second video at 24 FPS contains 120 frames. With 4× temporal compression (4 frames → 1 temporal unit in the latent), this becomes 30 temporal units. At full 512×512 resolution, each set of 4 frames produces ~4,096 tokens, so 120 frames produce 30 × 4,096 = 122,880 tokens — just within the 131,072-token context window.

The `[EOF]` tokens inserted between frames allow the model to learn where one frame ends and the next begins, which is essential for generating coherent temporal sequences. Without frame delimiters, the model would have no way to distinguish between spatial structure within a frame and temporal structure across frames.

##### Autoregressive Video Extension

The paper demonstrates that Emu3 can **extend videos beyond its training length** through autoregressive generation. The procedure:

1. **Tokenize the context video:** A 2-second video at 24 FPS (48 frames) is tokenized into discrete vision tokens with metadata.
2. **Feed as prefix:** The token sequence [BOS] [context video tokens] [EOF] is fed into the model.
3. **Generate continuation:** The model autoregressively predicts subsequent vision tokens, which represent the next 2 seconds of video.
4. **Detokenize:** The predicted tokens are decoded back to pixel values through the vision tokenizer's decoder.

The paper reports successful extension by up to **8 seconds** using only 2 seconds of context. This works because the model, during pretraining, learned to predict what happens next in video sequences — it has internalized patterns of motion, object permanence, and temporal consistency.

**Why this is remarkable.** Diffusion-based video models generate the entire video at once (from noise) and are fundamentally limited by their training resolution in both space and time. Emu3's autoregressive approach means video length is limited only by the context window (131K tokens, roughly 5 seconds at full quality) for single-pass generation, and can be extended arbitrarily through iterative continuation. This is analogous to how LLMs generate text longer than their training context by feeding their own output as the new prefix.

##### Post-Processing: Stabilization and Super-Resolution

The raw outputs of the tokenizer's decoder may have temporal inconsistencies (flickering) and are limited to the training resolution (512-720 pixels). The paper applies two post-processing models to address this:

**Video stabilization.** Based on the temporal VAE from Stable Video Diffusion, a stabilization model is trained on curated video data. The model takes autoencoded video clips from the tokenizer and produces temporally smoothed versions. Training uses a combined loss: **L1 loss** (pixel-level accuracy), **LPIPS perceptual loss** (visual quality), **GAN loss** (realism), and **KL penalty** (latent regularization, keeping the latent close to a standard normal distribution, which is standard practice for VAE-based models).

The input and output dimensions are **16×256×256** — 16 frames at 256×256 resolution. This means the stabilization operates at a reduced resolution and then upscales.

**Super-resolution.** A spatial-temporal U-Net model upsamples images or video clips by a factor of **4×**. The architectural details:

- **BlurPool for downsampling:** Instead of standard strided convolutions or pooling, BlurPool applies an anti-aliasing filter (blurring) before downsampling. This reduces aliasing artifacts (the "checkerboard" patterns that can appear in generated images) by ensuring the signal is properly band-limited before subsampling.
- **Sub-pixel convolution for upsampling:** Instead of standard interpolation, sub-pixel convolution rearranges features from the channel dimension into spatial dimensions. This is learned upsampling — the model decides how to distribute information across the higher-resolution grid.
- **Training:** On random crops of 8×256×256 from videos with resolution > 1024×1024, using a combined loss of **L2 loss, LPIPS loss, and GAN loss**.

The super-resolution model works on both images and videos. For videos, it processes temporal windows, providing temporal as well as spatial super-resolution.

**Why post-processing is necessary.** The tokenizer compresses 512×512 images into 4,096 tokens. Each token represents, on average, 8×8 pixels = 64 pixels of information. At this compression ratio, fine details like text, hair strands, or distant objects are poorly represented. The super-resolution model recovers these details — it has learned what high-resolution natural images look like and can hallucinate plausible fine structure consistent with the low-resolution tokenized output. The stabilization model similarly hallucinates plausible temporal consistency where the tokenizer may have introduced frame-to-frame variation. These post-processing steps are common in image/video generation pipelines and represent a practical engineering decision: let the main model handle semantic content at moderate resolution, and let specialized (smaller) models handle the detail enhancement.

## 4. Key Insights and Innovations

### Innovation 1: Next-Token Prediction as a Sufficient Paradigm — An Existence Proof, Not a Theoretical Claim

The paper's deepest contribution is not a new architecture or training technique but an **existence proof**: demonstrating that a single Transformer decoder trained with next-token prediction on discrete tokens from all modalities can match or exceed the performance of specialized architectures — diffusion models for generation and CLIP-plus-LLM compositions for perception — without any modality-specific inductive biases. This is fundamentally different from prior unified multimodal efforts.

**What the field assumed before Emu3.** The dominant belief, reinforced by years of empirical results, was that different modalities require different learning paradigms. Diffusion models iteratively denoise — a process with no analog in language modeling — because direct pixel-level next-token prediction was thought to be too coarse, too computationally expensive, or simply less effective. Compositional approaches used separate vision encoders (trained with contrastive objectives) and LLMs (trained autoregressively) because fusing modalities at the token level was believed to produce inferior representations. Prior unified attempts — Emu, Emu2, CM3Leon, Chameleon — either resorted to hybrid architectures (combining autoregressive and diffusion components, as Transfusion and Show-o do) or achieved results that fell meaningfully short of task-specific models. The field had settled into a local equilibrium where architectural complexity was accepted as the price of strong performance.

**What Emu3 changes.** The paper demonstrates that this architectural fragmentation was not necessary. By changing nothing except the tokenizer (which converts pixels to discrete codes) and the embedding layer (which accommodates those codes), a standard Llama-style Transformer — with RMSNorm, RoPE, GQA, SwiGLU, and no modality-specific branches — achieves:

- **Image generation:** Surpasses SDXL in human evaluation (Figure 5) and on automated benchmarks like GenEval (0.66 vs. SDXL's 0.55, Table 4) and DPG-Bench (80.60 vs. 74.65, Table 8).
- **Vision-language understanding:** Matches or exceeds LLaVA-1.6-7B — a system with a pretrained CLIP encoder, a pretrained Vicuna LLM, and a learned adapter — across twelve benchmarks while using no pretrained vision encoder at all (Table 6). Emu3's average across these benchmarks is competitive despite starting from a purely autoregressive objective.
- **Video generation:** Outperforms the majority of open-source diffusion-based video models on VBench (Table 5, overall score 80.96 vs. OpenSora-1.2's 79.76), with only proprietary models like Gen-3 and Kling scoring higher.

**The distinction from Chameleon.** The Chameleon team (2024) explored a similar idea — token-based mixed-modal autoregressive modeling — but their results fell substantially short of task-specific baselines (GenEval overall: 0.39 vs. SDXL's 0.55; VQAv2: 69.6% vs. LLaVA-1.6's 81.8%). Emu3's contribution is therefore not the *idea* of token-based multimodal modeling but the demonstration that it can be made to work *at competitive or superior performance levels*. The difference is in the execution: the vision tokenizer (codebook size 32,768, 4×8×8 compression with temporal 3D convolutions), the data pipeline (dense synthetic captions, aesthetic filtering, recall-oriented recovery), the training recipe (two-stage pretraining with video introduced later, vision token loss weighting of 0.5), and the post-training strategy (QFT, DPO for generation). These are engineering choices that collectively closed the gap that prior work left open.

**Why this is a fundamental shift, not incremental.** This is not a small refinement of an existing approach — it is a **paradigm-level result**. If a single next-token prediction model can match specialized architectures *today*, at current scale, then scaling such a model further (more parameters, more data, more compute) should eventually surpass them decisively, while diffusion and compositional approaches face architectural ceilings (iterative sampling latency, frozen encoder representations, modality-specific bottlenecks). The paper provides evidence for this scaling hypothesis by showing that the model handles three modalities (text, image, video) and two task families (generation, perception) with no architectural changes — just different input formatting. This is exactly what a general learning paradigm should look like.

The open-source release of the vision tokenizer — which the paper notes was "previously publicly unavailable" — underscores the contribution as infrastructure-building. Prior to Emu3, researchers wanting to explore token-based multimodal models faced a chicken-and-egg problem: training a high-quality discrete vision tokenizer is difficult and expensive, but without one, the approach cannot be tried. By releasing the tokenizer, the paper lowers the barrier for the field to verify, extend, and potentially adopt this paradigm.

---

### Innovation 2: Difficulty-Conditioned Data Preparation as the Hidden Enabler

While the paper's headline result is architectural minimalism, the unspoken innovation is the **data preparation strategy** — specifically, the recognition that a unified token-based model requires fundamentally different data curation than either diffusion-based generators or CLIP-based perceivers, and that getting this right is what enables the architectural simplicity to work.

**What prior unified models got wrong about data.** Chameleon trained on interleaved text-image web documents, which meant the text associated with images was often noisy alt-text, surrounding prose, or metadata — not dense visual descriptions. Diffusion models like SDXL relied on CLIP embeddings as conditioning signals, which were trained on web-scraped alt-text with a contrastive objective that tolerates noisy correspondence (the model only needs to distinguish matching from non-matching pairs, not to generate precise descriptions). Compositional VLMs like LLaVA used high-quality instruction-tuning data but only for the perception side, with no generation component that would be harmed by imperfect captions.

**What Emu3 does differently.** The paper constructs a **bidirectional data engine** that produces high-quality synthetic captions at scale:

- **For generation (text → image):** The model needs to learn the mapping from text descriptions to visual tokens. If the training captions are noisy (missing objects, wrong colors, insufficient detail), the model learns a noisy mapping and produces images that don't match their prompts. The solution: use GPT-4V to generate ~1M dense captions, fine-tune Emu2-17B on these to create a specialized captioner, then deploy it at scale with vLLM to recaption the entire image dataset. This creates a training distribution where every image has a detailed, accurate textual description — the opposite of web-scraped alt-text.

- **For perception (image → text):** The model needs to learn the reverse mapping. By constructing training sequences where the caption follows the vision tokens (the "understanding data format"), the model learns to generate descriptions conditioned on images. The same synthetic captions serve both purposes, but the sequence ordering teaches the model both directions.

- **For video:** The same pattern applies, with GPT-4V generating descriptions that capture both content and motion, followed by human revision of a subset, followed by fine-tuning and large-scale deployment.

**The design logic is subtle and non-obvious.** Why not just use the web-scraped captions that come with the images? The paper's implicit answer is that web captions optimize for a different objective — they describe what a human might want to know about an image in a web context (product details, context, commentary), not what a model needs to reconstruct the image. "Golden retriever in a field" is a fine alt-text for accessibility, but it doesn't tell a model about the dog's pose, the lighting, the background trees, or the camera angle. A dense synthetic caption describes all of these, providing the rich supervision signal that makes next-token prediction work for generation.

**The aesthetic filtering pipeline is similarly sophisticated.** The paper does not simply filter by resolution (a common baseline). It applies LAION-AI aesthetic scoring with a threshold of 5.5, then *recovers* text-heavy and non-monochromatic images that the aesthetic classifier over-rejects — explicitly acknowledging that aesthetic classifiers have systematic biases (favoring highly saturated, artistic compositions) that would eliminate important visual categories (charts, documents, infographics) needed for perception tasks. This recovery step is crucial for the model's strong performance on OCRBench (687, Table 6) and document understanding benchmarks — it would not generalize to document QA if documents were filtered out of training.

**For video, the pipeline is even more elaborate:** scene detection (to create coherent clips), text removal (to avoid training on screen recordings), optical flow filtering with a normalized flow score (to remove static and shaky footage), and aesthetic assessment using the minimum of three frame scores (so that any ugly frame disqualifies the clip). Each step addresses a specific failure mode: training on text-heavy videos would bias the model toward text generation rather than motion learning; training on static videos would teach no temporal dynamics; training on aesthetically poor videos would lower generation quality.

**Why this is an innovation rather than "just good engineering."** The paper identifies a **necessary condition** that prior unified models missed: for a token-based model to learn the text↔image mapping bidirectionally at high quality, the text must be *dense enough to capture visual information that would otherwise require modality-specific inductive biases*. Diffusion models don't need dense captions because the denoising process, guided by classifier-free guidance, can fill in details that the conditioning signal (a CLIP embedding of a terse caption) omits. Compositional VLMs don't need dense captions because the frozen CLIP encoder already provides rich visual features — the text is only the output, not the supervisory signal for visual learning. Emu3 has neither of these crutches: the only supervision for the vision↔language mapping is the token-level cross-entropy loss. Without dense, accurate textual descriptions, the model would have no way to associate specific visual features with specific words — it would learn a blurry, underspecified mapping that produces mediocre images and generic captions.

The evidence for this interpretation is in the benchmarks that test fine-grained alignment. On DPG-Bench, which uses long, detailed prompts with many constraints ("a red ball on a blue table next to a green chair"), Emu3 scores 80.60, surpassing SDXL's 74.65 (Table 8). On T2I-CompBench, which tests attribute binding (color, shape, texture), Emu3 with rewriting scores 0.7913 on color binding, compared to SDXL's 0.6369 (Table 7). These are precisely the evaluation dimensions where dense captions during training would provide an advantage — the model has seen many examples of detailed textual descriptions of color, shape, texture, and spatial relationships, and has learned to reproduce them faithfully.

---

### Innovation 3: Demonstrating That Autoregressive Models Can Generate Video Competitively Without Diffusion

Emu3's video generation results are significant not merely because they achieve competitive VBench scores but because they demonstrate a **qualitatively different generation mechanism** from the diffusion paradigm that dominates text-to-video research. This is not an incremental improvement — it is a demonstration that a capability the field assumed required iterative denoising can be achieved through causal next-token prediction.

**The field's default assumption.** Video generation has been almost exclusively diffusion-based. Sora (Brooks et al., 2024), the most prominent video generation model, uses a video diffusion architecture — generating video from noise through iterative denoising, conditioned on text. The same approach underlies Stable Video Diffusion, Gen-2, Gen-3, Kling, CogVideoX, and every other model in the VBench comparison (Table 5) except Emu3. The assumption was that video's high dimensionality (frames × height × width × color channels) and the need for temporal consistency required the coarse-to-fine refinement that diffusion provides — that autoregressive models, which commit to each token sequentially without the ability to revise earlier decisions, would produce videos that drift, flicker, or lose coherence over time.

**What Emu3 shows.** By generating video tokens autoregressively in a causal sequence (with frame delimiters but no mechanism for revising previously generated frames), Emu3 achieves an overall VBench score of 80.96 — competitive with diffusion-based models and outperforming most open-source alternatives (Table 5). The model's strongest dimension is **Dynamic Degree** (79.27), where it substantially exceeds all diffusion baselines. Dynamic Degree measures the amount and realism of motion — high scores indicate the model produces videos with natural movement rather than static or unnaturally jerky footage. The diffusion model with the next-highest Dynamic Degree is CogVideoX-5B at 70.97, nearly 8 points lower.

This is counterintuitive. Diffusion models, by iteratively refining from coarse structure to fine detail, would seem better suited to maintaining temporal coherence — they can adjust early frames based on information from later denoising steps. Autoregressive models commit to each token before seeing subsequent context, which should make them *more* susceptible to temporal drift. Yet Emu3 produces *more* dynamic motion than diffusion competitors.

**The explanation likely lies in the training objective.** Diffusion models are trained to denoise random corruptions of real videos — they learn to reconstruct existing motion, not to predict what motion *should* occur given a text description. The denoising objective encourages the model to be conservative: when in doubt, produce minimal motion because motion is harder to reconstruct than stasis. Emu3 is trained to predict the next token in a sequence of real videos — it learns that certain visual patterns (a person mid-stride, a wave breaking, leaves rustling) are followed by specific subsequent patterns. The autoregressive objective directly optimizes for **motion continuation**, which may produce more natural dynamics than the reconstruction-oriented diffusion objective.

**The future prediction capability (Figure 8) reinforces this interpretation.** Emu3 can take 2 seconds of video as context and generate the next 2 seconds — not by denoising a random latent but by causally predicting what visual tokens come next. This is a form of video understanding (the model must recognize what is happening — a person walking, a 3D animation rotating — and extrapolate the motion) and generation simultaneously. The paper reports successful extension by 8 seconds from 2 seconds of context, demonstrating that the learned dynamics generalize beyond the training sequence length. Diffusion models cannot do this — they generate entire videos at once and cannot extend an existing video without re-encoding it and running the full diffusion process.

**Where autoregressive video still lags.** Emu3 scores lower than top diffusion models on visual quality dimensions: Aesthetic Quality (59.64 vs. Gen-3's 63.34), Image Quality (not separately reported but implied), and Subject Consistency (95.32 vs. Gen-3's 97.10). The model also scores notably low on Human Action (77.71 vs. CogVideoX-5B's 99.40) — human motion appears to be a weakness. These gaps suggest that autoregressive generation, at least at Emu3's scale, still struggles with fine-grained visual fidelity and specific semantic categories compared to diffusion. The post-processing pipeline (stabilization + super-resolution) partially addresses this, but the raw token predictions appear to be less detailed than diffusion outputs.

**The structural implication.** If autoregressive video generation can be made competitive at this scale, scaling the model (more parameters, longer training, higher-resolution tokens) should close the remaining quality gap, just as scaling improved autoregressive image generation (from DALL-E 1 to Parti to Emu3). The field's heavy investment in video diffusion architectures may be a local optimum — one that works well at current compute scales but that will be overtaken by autoregressive approaches as models grow. Emu3 provides the first evidence that this trajectory is viable.

---

### Innovation 4: Direct Preference Optimization Transfers to Multimodal Generation Without Architectural Modification

Emu3 demonstrates that DPO — a technique developed for aligning LLMs with human preferences in text generation — can be applied to autoregressive image generation with no modification to the algorithm, no separate reward model training, and no modality-specific preference modeling. This is not a trivial extension; it validates that the preference optimization framework developed for language transfers cleanly when language and vision share a token-based representation.

**Prior work on preference alignment for images.** Diffusion-based image generators have been aligned with human preferences primarily through RLHF-style methods (training a reward model on human preference data, then using it to fine-tune the diffusion model) or through classifier-free guidance scale tuning. These approaches are complex and modality-specific — they require understanding denoising dynamics, designing reward functions that work with iterative sampling, and managing the interaction between preference optimization and the diffusion objective. DPO, which avoids training a separate reward model by directly optimizing the policy from preference pairs, was developed for text and its applicability to continuous or hybrid multimodal generation was not obvious.

**How Emu3 makes it work.** The key enabler is the discrete token representation. Because images are represented as sequences of tokens from the same vocabulary as text (augmented with 32,768 vision codes), the DPO loss — which compares the model's log-probability of "chosen" vs. "rejected" sequences relative to a reference model — applies without modification. The image is just a token sequence; the DPO objective operates on token probabilities; nothing about the algorithm is modality-specific.

The implementation details matter, however:

- **Preference data construction is domain-specific.** The paper generates 8–10 images per prompt from the QFT model, has three human voters score each on visual appeal and prompt alignment, and selects the highest-scoring as "chosen" and lowest as "rejected." This mirrors the data construction for text DPO but the evaluation criteria (visual appeal, prompt alignment) are specific to image generation.

- **Token storage eliminates reconstruction variance.** The paper explicitly stores the token sequences generated during candidate creation and reuses them during DPO training, rather than regenerating from the images. This avoids a subtle failure mode: an image encoded to tokens, decoded to pixels for human evaluation, and then re-encoded to tokens for training might produce different tokens (due to floating-point precision, non-deterministic operations, or the tokenizer's own stochasticity). Training on the exact tokens that the evaluators (implicitly) rated ensures the "chosen" and "rejected" sequences correspond to what was actually evaluated.

- **Combined DPO + NTP loss prevents distribution collapse.** The total loss is $\mathcal{L}_{\text{DPO}} + \mathcal{L}_{\text{NTP}}$. The NTP term acts as an anchor — without it, the model could collapse to producing only high-preference images from a narrow distribution, losing the diversity needed for open-ended generation. This is analogous to the KL penalty in RLHF but implemented as a shared loss term rather than a separate constraint.

**The results reveal a distribution mismatch problem.** Emu3-DPO improves human preference scores (visual quality +8.3, prompt alignment +4.3, Figure 6) but *degrades* automated benchmark scores (GenEval overall: 0.64 vs. 0.66; T2I-CompBench average: 0.7164 vs. 0.7422, Table 4). The paper attributes this to the preference data emphasizing "overall aesthetic quality" while automated benchmarks measure narrower dimensions (object counting, attribute binding, spatial relationships). This is the **alignment tax** familiar from LLM RLHF — optimizing for human preference can reduce performance on specific capability metrics that weren't explicitly in the preference data. The key insight is that this phenomenon, previously documented only for text, appears in vision generation through exactly the same mechanism, suggesting it is a fundamental property of preference optimization rather than a language-specific artifact.

**Significance beyond Emu3.** This result validates that the token-based representation makes multimodal models compatible with the full LLM alignment toolkit: DPO, RLHF, constitutional AI, and any future preference optimization method developed for text will transfer to vision generation without modification. This is a practical advantage of the unified architecture — it amortizes alignment research across modalities — and a conceptual one: it suggests that human preferences about images and text can be expressed and optimized in a shared framework, which is what a general multimodal intelligence would require.

## 5. Experimental Analysis

### Evaluation Methodology

- **Datasets.** The paper evaluates on four task categories with distinct benchmarks:
  - **Text-to-image generation:** MSCOCO-30K (30,000 validation prompts for FID/CLIP score), GenEval (object-focused compositional alignment), T2I-CompBench (attribute binding: color, shape, texture), and DPG-Bench (dense prompt following with long, detailed captions). All are standard benchmarks in the text-to-image literature.
  - **Human evaluation (image):** A custom set of 100 diverse user prompts, each evaluated by three independent voters on visual quality and prompt alignment, producing weighted overall scores. Used to compare Emu3 against closed and open models (DALL-E 3, Midjourney v5.2, FLUX.1-dev, PG-v2.5, SDXL).
  - **Text-to-video generation:** VBench, a comprehensive benchmark evaluating 16 dimensions of video quality, including motion smoothness, dynamic degree, aesthetic quality, object class accuracy, human action recognition, spatial relationships, scene consistency, and subject/background consistency. The paper reports 11 of 16 dimensions plus the total score.
  - **Vision-language understanding:** Twelve benchmarks covering diverse capabilities: SEEDBench-Img (general visual comprehension), OCRBench (text reading), MMVet (integrated multimodal capabilities), POPE (object hallucination), VQAv2 and GQA (visual question answering), ScienceQA-Img (science reasoning), TextVQA (text-in-image QA), ChartQA (chart understanding), DocVQA and InfoVQA (document understanding), AI2D (diagram understanding), RealWorldQA (real-world visual QA), MMMU (multimodal reasoning across disciplines), and MMBench (general multimodal evaluation).

- **Base model.** Emu3 is an 8B-parameter Transformer decoder trained from scratch. It is architecturally a Llama-2-style model (32 layers, hidden size 4,096, intermediate size 14,336, 32 attention heads with 8 KV heads via GQA, RoPE with base frequency 1,000,000, context length up to 131,072 tokens). The model uses a unified vocabulary of 184,622 tokens (text tokens from QwenTokenizer plus 32,768 vision tokens from the SBER-MoVQGAN-based tokenizer). The 8B scale is chosen to be representative of contemporary open-source LLM capabilities while being feasible to train from scratch on a multimodal corpus.

- **Metrics.** Evaluation is task-specific:
  - **Image generation (automated):** CLIP-I (image similarity via CLIP-ViT-L), CLIP-T (text-image alignment via CLIP-ViT-B), and zero-shot FID on MSCOCO-30K (computed with 30K random validation prompts following the protocol from Emu); overall and per-dimension scores on GenEval (six dimensions: single object, two objects, counting, colors, position, color attribute); color/shape/texture binding scores on T2I-CompBench (evaluated via BLIP-VQA); and per-question and overall scores on DPG-Bench (evaluated via mPLUG-large).
  - **Image generation (human):** Visual quality and prompt alignment scored by three independent voters per prompt, aggregated into a weighted overall score on a scale where higher is better. The paper reports scores in the 50–80 range, with DALL-E 3 achieving 73.4 (English) and Emu3-DPO achieving 70.0 (English) / 67.7 (Chinese).
  - **Video generation:** VBench provides per-dimension scores and a total composite score. Higher values indicate better performance on all dimensions. The total score is the primary aggregation metric.
  - **Vision-language understanding:** Per-benchmark accuracy or equivalent metric as defined by each benchmark's standard evaluation protocol. For OCRBench, the paper notes results are "normalized," though the normalization procedure is not detailed. Twelve benchmarks are evaluated with per-task scores and an implied average across them.

- **Baselines.** The paper compares against four categories of prior work:
  - **Diffusion-based image generators:** SDv1.5, SDv2.1, SDXL, PixArt-alpha, DALL-E 2, DALL-E 3, SD3 (Esser et al., 2024), Playground v2.5 (Li et al., 2024), Lumina-Next (Zhuo et al., 2024), Hunyuan-DiT (Li et al., 2024), PixArt-Sigma (Chen et al., 2024), Playground v3 (Liu et al., 2024). SDXL is the primary generation baseline as the flagship open-source diffusion model.
  - **Unified (autoregressive + diffusion hybrid) models:** Emu (Sun et al., 2023), Show-o (Xie et al., 2024), Transfusion (Zhou et al., 2024).
  - **Pure autoregressive generative models:** Chameleon (Team, 2024), LlamaGen (Sun et al., 2024).
  - **Encoder-based vision-language models:** InstructBLIP (Dai et al., 2023), IDEFICS-9B (IDEFICS Team, 2023), QwenVL-Chat (Bai et al., 2023), LLaVA-1.5 (Liu et al., 2024), InternVL-Chat (Chen et al., 2024), mPLUG-Owl2 (Ye et al., 2024), ShareGPT4V (Chen et al., 2023), LLaVA-1.6 (Liu et al., 2024), VILA (Lin et al., 2024). LLaVA-1.6-7B is the primary perception baseline as the best-performing open-source VLM.
  - **Encoder-free VLMs:** Fuyu-8B (Bavishi et al., 2023), Chameleon-MT-34B, Show-o (as VLM), EVE-7B (Diao et al., 2024).
  - **Text-to-video models:** 13 models on VBench including ModelScope (Wang et al., 2023), LaVie (Wang et al., 2023), OpenSoraPlan V1.1 (PKU-Yuan Lab, 2024), Show-1 (Zhang et al., 2023), OpenSora V1.2 (Zheng et al., 2024), AnimateDiff-V2 (Guo et al., 2023), Gen-2 (Runway, 2023), Pika (Pika Labs, 2023), VideoCrafter-2.0 (Chen et al., 2024), T2V-Turbo (Li et al., 2024), CogVideoX-5B (Yang et al., 2024), Kling (Kuaishou, 2024), Gen-3 (Runway, 2024). All are diffusion-based except Emu3.

- **Generation budget / compute accounting.** The paper does not use a standardized inference compute metric (like number of function evaluations or FLOPs) for comparing generation costs across models. Instead, comparisons are based on output quality at fixed resolution: image generation at 512×512 (Emu3) or 720×720 (Emu3-DPO), video generation at 5 seconds and 24 FPS. Generation uses Top-k = 16,384 and Top-p = 1.0 sampling for images, with classifier-free guidance scale of 5.0–5.5. The number of sampling steps (for diffusion baselines) vs. autoregressive steps (for Emu3) is not compared — the evaluation focuses on output quality rather than inference cost. For vision-language understanding, no generation budget applies since the task is single-pass conditional generation.

- **Cross-validation / statistical protocol.** The paper does not report cross-validation or statistical significance testing for benchmark comparisons. Automated metrics on MSCOCO-30K use all 30K validation prompts; GenEval uses 4 images per prompt; T2I-CompBench uses 10 images per prompt; DPG-Bench uses 4 images per prompt. Human evaluation uses 3 voters per prompt on 100 prompts — a relatively small sample for human preference evaluation, and the paper does not report inter-rater agreement or confidence intervals. VBench comparisons use the standard evaluation protocol of that benchmark, with no details on variance or statistical testing.

---

### Main Quantitative Results

#### Image Generation: Automated Benchmarks

The headline result is that **Emu3 matches or exceeds SDXL — the flagship open-source diffusion model — across automated text-to-image benchmarks while matching DALL-E 3 on compositional alignment metrics when prompts are rewritten.**

**MSCOCO-30K (Table 4).** Emu3 achieves CLIP-I of 0.689 (vs. SDXL's 0.674), CLIP-T of 0.313 (vs. SDXL's 0.310), and FID of 12.8 (vs. SDv1.5's 9.93 — the paper does not report SDXL FID, but SDv1.5 achieves the best FID among listed models). The CLIP-I and CLIP-T scores place Emu3 slightly above SDXL on both image similarity and text alignment, though the gaps are modest. Emu3-DPO achieves similar CLIP scores (0.680 CLIP-I, 0.312 CLIP-T) but notably worse FID (19.3 vs. 12.8), suggesting the DPO process improves human-perceived quality at a cost to distribution-matching metrics like FID.

The comparison with Chameleon — the most directly comparable autoregressive model — is stark: Chameleon achieves an FID of 26.74, more than twice Emu3's, indicating substantially worse visual quality. Transfusion (6.78 FID) and Show-o (9.24 FID) — hybrid autoregressive-diffusion models — achieve better FID than Emu3 but use fundamentally different architectures that include diffusion components.

**GenEval (Tables 4, 7).** Without prompt rewriting, Emu3 achieves an overall GenEval score of 0.54, comparable to SDXL (0.55) and Show-o (0.53), and substantially ahead of Chameleon (0.39) and LlamaGen (0.32). With GPT-4V prompt rewriting (following the DALL-E 3 evaluation protocol), Emu3 achieves **0.66 overall** — exceeding SDXL (0.55) by a wide margin and approaching DALL-E 3 (0.67). Emu3-DPO with rewriting achieves 0.64.

The per-dimension breakdown reveals where the rewriting matters most. On the "Position" dimension (spatial relationship understanding), rewriting raises Emu3's score from 0.17 to 0.49 — nearly matching DALL-E 3's 0.43 and dramatically exceeding SDXL's 0.15. On "Counting," rewriting raises the score from 0.34 to 0.42, still below SD3's 0.72 but competitive with DALL-E 3's 0.47. On "Color Attribute," rewriting takes the score from 0.21 to 0.45, matching DALL-E 3's 0.45. The rewriting effect confirms the paper's claim that Emu3 performs better with dense captions (which rewriting approximates) and that standard GenEval prompts are "too brief to accurately reflect the model's true performance."

**T2I-CompBench (Tables 4, 7).** Without rewriting, Emu3 achieves a color binding score of 0.6107, shape binding of 0.4734, and texture binding of 0.6178 — all competitive with or exceeding SDXL (0.6369, 0.5408, 0.5637 respectively). With rewriting, Emu3's scores jump to 0.7913 (color), 0.5846 (shape), and 0.7422 (texture). The color binding score with rewriting (0.7913) approaches DALL-E 3's 0.8110, while the texture binding score (0.7422) comes close to DALL-E 3's 0.8070.

Notably, Emu3-DPO underperforms Emu3 on all three T2I-CompBench dimensions with rewriting (0.7544 color, 0.5706 shape, 0.7164 texture), mirroring the pattern seen on GenEval — DPO improves human preference but degrades automated metric performance.

**DPG-Bench (Tables 4, 8).** Emu3 achieves an overall score of 80.60, exceeding SDXL's 74.65 by 5.95 points and PixArt-alpha's 71.11 by 9.49 points. Emu3-DPO further improves to 81.60, surpassing Playground v2.5's 75.47 and approaching DALL-E 3's 83.50. The per-dimension breakdown shows Emu3's strengths: on "Relation" (spatial and semantic relationships between objects), Emu3 scores 90.22 — exceeding SDXL's 86.76 and approaching DALL-E 3's 90.58. On "Global" (overall image quality and coherence), Emu3 scores 85.21 vs. SDXL's 83.27. On "Entity" (individual object accuracy), Emu3 scores 86.68 vs. SDXL's 82.43. These results directly support the paper's claim that dense synthetic captions during training produce better prompt-following, especially for long, detailed prompts with complex object relationships.

---

#### Image Generation: Human Evaluation

The human evaluation (Figure 5) places Emu3 in the context of both open and closed models. On English prompts, **Emu3-DPO achieves an overall human preference score of 70.0**, placing it:
- Above SDXL (66.9) and Playground v2.5 (68.5)
- On par with DALL-E 3 (73.4) and Midjourney v5.2 (71.1)
- Below FLUX.1-dev (74.6)

On Chinese prompts, Emu3-DPO achieves 67.7 — notably, no other model's Chinese-prompt performance is reported, making this a single-point comparison that primarily demonstrates multilingual capability rather than competitive positioning.

The DPO ablation (Figure 6) shows the effect of preference optimization on the two evaluation dimensions:
- **Visual quality:** 52.3 (w/o DPO) → 60.6 (w/ DPO), an improvement of 8.3 points
- **Prompt alignment:** 57.3 (w/o DPO) → 61.6 (w/ DPO), an improvement of 4.3 points

Both improvements are substantial relative to the scale, but the visual quality gain is approximately 2× larger than the prompt alignment gain. This aligns with the paper's description that the DPO preference data emphasizes "overall aesthetic quality" — the human voters are weighting visual appeal more heavily than prompt fidelity, and the model is learning that weighting.

The sample size (100 prompts, 3 voters each) is adequate for ranking models but the paper does not report confidence intervals, inter-rater agreement statistics (e.g., Krippendorff's alpha), or statistical significance tests between model pairs. The observed differences between Emu3-DPO (70.0) and SDXL (66.9) or DALL-E 3 (73.4) could be statistically indistinguishable depending on variance — the reader cannot assess this from the reported data.

---

#### Video Generation: VBench

Against 13 text-to-video models (all diffusion-based except Emu3), **Emu3 achieves an overall VBench score of 80.96** (Table 5), placing it:
- Above OpenSora V1.2 (79.76), VideoCrafter-2.0 (80.44), and AnimateDiff-V2 (80.27)
- Comparable to T2V-Turbo (81.01) and Pika (80.69)
- Below the proprietary models CogVideoX-5B (81.61), Kling (81.85), and Gen-3 (82.32)

The per-dimension breakdown reveals a **strikingly non-uniform profile**:

**Where Emu3 excels:**
- **Dynamic Degree: 79.27** — the highest among all models by a margin of 8.3 points (next best: CogVideoX-5B at 70.97). This measures the amount and realism of motion. Emu3's autoregressive next-token prediction objective — which directly optimizes for motion continuation — appears to produce more natural dynamics than the reconstruction-oriented diffusion objective used by competitors.
- **Motion Smoothness: 98.93** — competitive with the best models (Gen-2: 99.58, Kling: 99.40). The post-processing stabilization step likely contributes here.
- **Spatial Relationship: 68.73** — exceeds most open-source models (OpenSora V1.2: 68.56, Show-1: 53.50) and competes with CogVideoX-5B (66.35) and Gen-3 (65.09).

**Where Emu3 is competitive:**
- **Background Consistency: 97.69** — comparable to most models (range: 95.29–98.22)
- **Subject Consistency: 95.32** — slightly below the top models (Gen-3: 97.10, OpenSora V1.2: 96.75)
- **Aesthetic Quality: 59.64** — mid-range among diffusion models (range: 54.94–67.16)

**Where Emu3 struggles:**
- **Human Action: 77.71** — substantially below the top models (CogVideoX-5B: 99.40, LaVie: 96.80, Gen-3: 96.40). This is a significant weakness: human motion, with its complex articulations and fine-grained dynamics, appears to be poorly captured by Emu3's tokenization and autoregressive generation.
- **Multiple Objects: 44.64** — below most diffusion models (Kling: 68.05, CogVideoX-5B: 62.11, Gen-2: 55.47). The model struggles when scenes contain several distinct entities, possibly because object interactions require coordinating tokens across many spatial positions, which autoregressive generation handles less naturally than the global refinement of diffusion.
- **Appearance Style: 37.11** — in the lower range (best: T2V-Turbo at 55.58, VideoCrafter-2.0 at 55.29).
- **Scene: 20.92** — among the lowest (range: 19.62–25.67). Scene-level coherence (overall setting, lighting consistency, atmospheric unity) appears to be a weak point.

**Interpretation of the VBench profile.** Emu3's strength pattern (excellent motion dynamics, competitive smoothness, weak on human action and multi-object scenes) is consistent with an autoregressive model that has learned motion continuity well but struggles with compositional complexity — coordinating multiple objects, handling articulated human poses, and maintaining global scene coherence across many frames. Diffusion models, by iteratively refining the entire video simultaneously, have an inherent advantage for global consistency (they can adjust early frames based on information from later denoising steps). Emu3 commits to each token irreversibly, making it harder to maintain global coherence.

The paper does not report video generation results with DPO applied, nor does it provide human evaluation for video quality (only automated VBench scores). Given that DPO improved image generation human preference scores substantially, video DPO is a notable omission that could address some of the visual quality weaknesses.

**Comparison with the paper's framing.** The paper claims Emu3 "outperforms the majority of open-source text-to-video models" on VBench — this is accurate for the overall score (80.96 beats 8 of 13 listed models). However, Emu3 does not outperform any of the top-tier proprietary models (Gen-3, Kling, CogVideoX-5B), and its per-dimension profile shows clear weaknesses that the overall score conceals. The result is better characterized as "competitive with open-source diffusion models" rather than "state-of-the-art video generation."

---

#### Future Prediction: Video Extension

The paper presents qualitative examples (Figure 8) of Emu3 extending 2-second video clips by an additional 2 seconds, and claims the model can "iteratively generate videos that surpass its contextual length" with successful extension by 8 seconds from 2 seconds of context. No quantitative evaluation of video extension quality is provided — no metrics, no baselines, no human evaluation. The absence of quantitative evaluation makes this a **demonstration of capability** rather than a rigorous result.

This is a significant gap because video extension quality is the key differentiator between autoregressive and diffusion-based video generation. Diffusion models generate fixed-length videos and cannot extend them autoregressively without re-running the full generation process; Emu3's causal generation is structurally capable of arbitrary-length extension. A quantitative comparison showing that Emu3's extensions maintain temporal coherence, visual quality, and semantic consistency over long horizons would be a strong argument for the autoregressive approach. Without such evaluation, the capability remains a qualitative curiosity.

---

#### Vision-Language Understanding: Twelve Benchmarks

The headline result is that **Emu3 — a pure decoder-only model with no pretrained vision encoder and no pretrained LLM — achieves competitive or superior performance compared to encoder-based VLMs across a diverse set of vision-language benchmarks**, with particularly strong results on text-heavy and document understanding tasks.

**Overall comparison (Table 6).** Across the twelve benchmarks, Emu3's performance relative to LLaVA-1.6-7B (the primary baseline and best-performing open-source encoder-based VLM) is:

| Benchmark | Emu3 | LLaVA-1.6-7B | Winner |
|---|---|---|---|
| SEEDBench-Img | 68.2 | 64.7 | Emu3 |
| OCRBench | 687 | 532 | Emu3 |
| MMVet | 37.2 | 43.9 | LLaVA-1.6 |
| POPE | 85.2 | 86.5 | LLaVA-1.6 |
| VQAv2 | 75.1 | 81.8 | LLaVA-1.6 |
| GQA | 60.3 | 64.2 | LLaVA-1.6 |
| ScienceQA-Img | 89.2 | 64.9 | Emu3 |
| TextVQA | 64.7 | 54.8 | Emu3 |
| ChartQA | 68.6 | 74.4 | LLaVA-1.6 |
| DocVQA | 76.3 | 37.1 | Emu3 |
| InfoVQA | 43.8 | 66.6 | LLaVA-1.6 |
| AI2D | 70.0 | 66.6 | Emu3 |
| RealWorldQA | 57.4 | 57.8 | Tie |
| MMMU | 31.6 | 35.1 | LLaVA-1.6 |
| MMBench | 58.5 | 67.4 | LLaVA-1.6 |

Emu3 wins or ties on 7 of 15 comparisons; LLaVA-1.6 wins on 8. The pattern is revealing:

**Where Emu3 excels:**
- **OCRBench: 687 vs. 532** — a 29% advantage. This is the largest relative margin and directly reflects the text-heavy data curation (the recall-oriented filtering that retained images with detected text) and the dense synthetic captioning (which includes reading and describing text in images).
- **DocVQA: 76.3 vs. 37.1** — more than 2× LLaVA-1.6's score. The supplementary understanding-focused data (charts, tables, text-rich content from DenseFusion) is clearly paying off here.
- **ScienceQA-Img: 89.2 vs. 64.9** — a 37% relative improvement. ScienceQA requires reasoning about diagrams and scientific illustrations, which may benefit from the unified training (the model learns to process scientific figures as part of its general visual training rather than relying on a frozen CLIP encoder that was not specifically trained on scientific imagery).
- **AI2D: 70.0 vs. 66.6** — another diagram understanding task where Emu3's specialized data shows.

**Where Emu3 lags:**
- **VQAv2: 75.1 vs. 81.8** and **GQA: 60.3 vs. 64.2** — general visual question answering, where LLaVA-1.6's pretrained CLIP features (trained on 400M image-text pairs) and Vicuna LLM (trained on massive text corpora) provide a strong foundation that Emu3's from-scratch training has not fully matched.
- **MMBench: 58.5 vs. 67.4** and **MMVet: 37.2 vs. 43.9** — broad multimodal benchmarks requiring diverse capabilities. The gap suggests that Emu3's training, while strong on specific domains (OCR, documents, diagrams), has not yet generalized as broadly as the CLIP+LLM approach.
- **InfoVQA: 43.8 vs. 66.6** — a substantial gap on infographic understanding, despite Emu3's strength on DocVQA. This may indicate that the supplementary data, while covering documents and charts, underrepresents infographic-style content.

**Comparison with other encoder-free models.** Emu3 dramatically outperforms all other encoder-free baselines. Fuyu-8B — the most prominent prior encoder-free model — achieves only 74.1 on VQAv2 (Emu3: 75.1), 74.2 on POPE (Emu3: 85.2), and 10.7 on MMBench (Emu3: 58.5). Chameleon-MT-34B reports only 69.6 on VQAv2 and 25.1 on MMMU — both substantially below Emu3 despite having 4.25× more parameters. EVE-7B scores 56.8 on SEEDBench-Img (Emu3: 68.2) and 25.7 on MMVet (Emu3: 37.2). Emu3 is not merely competitive with prior encoder-free models — it represents a step-change improvement, demonstrating for the first time that the encoder-free approach can actually compete with encoder-based methods.

**Caveat on training data overlap.** The paper marks several benchmarks with an asterisk (\*) and notes "The images of related training datasets are observed during training." The asterisked benchmarks are: VQAv2, GQA, ScienceQA-Img, TextVQA, ChartQA, DocVQA, InfoVQA, AI2D. This means the model was exposed to training images from these benchmarks during pretraining or fine-tuning. The extent of overlap is not quantified — whether the model saw the exact test images, similar images from the same distribution, or merely the training sets of these datasets. This is a significant confound: Emu3's strong performance on ScienceQA-Img (89.2) and DocVQA (76.3) may partly reflect memorization of training data rather than genuine visual understanding. The paper's asterisk notation is transparent about this but does not mitigate the concern — readers cannot determine how much of the performance advantage is due to the architectural approach vs. data overlap.

---

#### Comparison of Emu3 vs. Emu3-DPO on Automated Benchmarks

A consistent pattern across all automated image generation benchmarks is that **Emu3-DPO scores lower than Emu3 on automated metrics while scoring higher on human evaluation:**

| Benchmark | Emu3 | Emu3-DPO | Direction |
|---|---|---|---|
| GenEval Overall (+Rewriter) | 0.66 | 0.64 | ↓ |
| T2I-CompBench Color (+Rewriter) | 0.7913 | 0.7544 | ↓ |
| T2I-CompBench Shape (+Rewriter) | 0.5846 | 0.5706 | ↓ |
| T2I-CompBench Texture (+Rewriter) | 0.7422 | 0.7164 | ↓ |
| DPG-Bench Overall | 80.60 | 81.60 | ↑ |
| FID (MSCOCO) | 12.8 | 19.3 | ↓ (worse) |

The paper attributes this to the DPO preference data emphasizing "overall aesthetic quality" — human raters prefer visually striking images, and the model learns to produce them, but automated metrics like GenEval (which counts objects by running a detector) and T2I-CompBench (which evaluates attribute binding via VQA) are not sensitive to aesthetic quality. Worse, the aesthetic optimization may *trade off* against the precise compositional control that these benchmarks measure — a beautifully composed image that slightly misplaces an object or mutes a color to improve overall harmony will score well with humans but poorly with automated evaluators.

This is the same "alignment tax" observed in LLM RLHF — optimizing for human preference can reduce performance on capability benchmarks. The paper does not attempt to disentangle whether this is unavoidable (a fundamental tradeoff between aesthetics and compositional accuracy) or addressable (e.g., by including both aesthetic and compositional criteria in the preference data). The DPG-Bench result (81.60 for Emu3-DPO vs. 80.60 for Emu3) is the exception — DPG-Bench uses longer, more descriptive prompts, and DPO's improvement in overall quality may help with the holistic evaluation that DPG-Bench's mPLUG-large evaluator performs.

---

### Ablation Studies and Robustness Checks

The paper does not contain traditional ablation studies in the sense of systematically removing or varying components of the architecture or training pipeline to measure their contribution. This is a notable gap. The following analyses, while not formal ablations, provide some insight into design choices:

**Vision tokenizer reconstruction quality (Figure 3, Table 2):** The paper evaluates the tokenizer's reconstruction fidelity at multiple resolutions, showing LPIPS between 0.099 and 0.112 across resolutions, PSNR from 21.59 to 24.30, and SSIM from 0.622 to 0.771. These metrics establish that the tokenizer preserves sufficient information for high-quality generation, but there is no comparison with alternative tokenizers (different codebook sizes, different compression ratios, different architectures) to establish that the chosen configuration is optimal. The tokenizer is treated as a fixed component — we see that it works, but not whether a better tokenizer would substantially improve results.

**Prompt rewriting effect on GenEval and T2I-CompBench (Table 7):** The comparison of Emu3 with vs. without GPT-4V prompt rewriting serves as an implicit ablation of the impact of prompt quality. Rewriting improves GenEval overall from 0.54 to 0.66 (a 22% relative improvement) and T2I-CompBench average (computed from the three dimensions) from ~0.57 to ~0.71 (a 25% improvement). This confirms the paper's claim that Emu3 performs better with dense captions and that standard benchmarks with terse prompts underestimate its capability. However, this is not a fair comparison — all models would benefit from better prompts, and the paper reports DALL-E 3's results with rewriting as the primary comparison point (DALL-E 3 and SD3 both use rewriting for reported GenEval scores). The rewriting ablation is better understood as a **domain shift analysis**: Emu3's training distribution emphasized dense synthetic captions, and the rewriting approximates this distribution, revealing that the gap between Emu3 and diffusion models narrows substantially when both are evaluated in their preferred prompt regime.

**DPO vs. no DPO (Figures 5, 6; Tables 4, 7, 8):** The comparison between Emu3 (QFT only) and Emu3-DPO (QFT + DPO) serves as an ablation of the preference optimization stage. DPO improves human preference scores (+8.3 visual quality, +4.3 prompt alignment, Figure 6) and DPG-Bench overall score (80.60 → 81.60, Table 8) but degrades automated metric performance on GenEval, T2I-CompBench, and FID. This demonstrates that DPO effectively aligns the model with human preferences while revealing the tradeoff with automated metrics. The absence of a video DPO ablation is a missed opportunity to understand whether the same tradeoffs apply to temporal generation.

**Classifier-free guidance scale:** The paper uses guidance scales of 5.0–5.5 for image generation but does not ablate this choice. The guidance scale is a critical hyperparameter in generative models — higher values increase prompt adherence but can reduce diversity and introduce artifacts. Without a sweep over guidance scales, the reader cannot assess whether Emu3's performance is robust to this choice or whether a different scale would close gaps on specific benchmarks.

**Resolution increase in QFT:** QFT increases training resolution from 512 to 720 pixels. The paper does not compare QFT at 512 vs. 720 resolution, so the contribution of resolution increase to quality improvement cannot be isolated. The resolution increase could be responsible for most of the visual quality gain attributed to QFT in general.

**Understanding data format ablation:** The post-training for vision-language understanding uses a specific resolution policy (resize to ~512, bounds at 512–1024) and a two-stage process (image-to-text training, then instruction tuning). Neither the resolution policy nor the two-stage approach is ablated. It is unclear whether a single-stage fine-tuning with mixed resolution handling would achieve similar results.

**Missing: Tokenizer codebook size ablation.** With 32,768 vision tokens, this is one of the largest codebooks used in discrete vision tokenization. The paper does not compare against smaller codebook sizes to determine whether the large codebook is necessary for the achieved performance or whether a smaller one would suffice with appropriate training.

**Missing: Loss weighting ablation (vision token weight of 0.5).** The paper's choice of 0.5 for vision token loss weight is presented without empirical justification. Whether this specific value is optimal, or whether results are sensitive to it, is unknown.

**Missing: Data scale ablations.** The paper does not vary the amount of training data (images, videos, text) or the proportion of synthetic vs. natural captions to measure their impact. Given the paper's emphasis on data quality and the dense captioning pipeline, understanding how performance scales with data quantity and quality would be informative.

---

### Critical Assessment

The paper makes four central claims in its executive summary and introduction:

**Claim 1: Emu3 outperforms SDXL in image generation and LLaVA-1.6 in vision-language understanding while eliminating diffusion and compositional architectures.**

This claim requires careful decomposition because "outperforms" means different things in different evaluation settings:

- **Against SDXL on automated benchmarks:** Emu3 achieves slightly better CLIP scores (CLIP-I 0.689 vs. 0.674, CLIP-T 0.313 vs. 0.310) and substantially better GenEval with rewriting (0.66 vs. 0.55) and DPG-Bench (80.60 vs. 74.65). On GenEval without rewriting, the two are essentially tied (0.54 vs. 0.55). On T2I-CompBench without rewriting, Emu3's average (~0.57) is comparable to SDXL's (~0.58). The evidence supports "Emu3 is competitive with or exceeds SDXL on automated benchmarks, with the margin depending on prompt quality" but not a blanket "outperforms."

- **Against SDXL on human evaluation:** Emu3-DPO (70.0) exceeds SDXL (66.9) by 3.1 points on a 100-prompt, 3-voter evaluation. This is a meaningful gap but the statistical significance is unreported. The claim of outperformance is supported but would benefit from larger-scale human evaluation with confidence intervals.

- **Against LLaVA-1.6-7B on vision-language understanding:** This is the most nuanced comparison. Emu3 wins on 7 of 15 benchmarks, but the wins are concentrated in OCR, document, and diagram understanding — tasks where Emu3's training data (text-rich images, DenseFusion supplementary data) was specifically curated. LLaVA-1.6 wins on general VQA (VQAv2, GQA), broad multimodal benchmarks (MMBench, MMVet, MMMU), and InfoVQA. The asterisked benchmarks (8 of 15) indicate training data overlap for Emu3, potentially inflating its scores. The claim "competes with LLaVA-1.6" (as stated in the introduction) is accurate; "outperforms LLaVA-1.6" (as the title of Figure 2 implies with "beats") is an overstatement — Emu3 does not outperform LLaVA-1.6 in aggregate across all benchmarks, and its advantages are concentrated in domains where it received specialized training data.

The overall assessment is that **Emu3 demonstrates competitive performance with task-specific architectures, eliminating the *architectural necessity* of diffusion and CLIP-based composition, but does not universally outperform them.** The significance is the architectural simplification, not a decisive performance advantage.

**Claim 2: Emu3 achieves state-of-the-art performance compared to well-established task-specific models.**

This claim, stated in the abstract, is too broad given the evidence:

- On image generation, Emu3 is competitive with but does not clearly surpass all state-of-the-art models. DALL-E 3 achieves higher GenEval (0.67 without rewriting; the paper's comparison uses 0.67 as DALL-E 3's score but DALL-E 3's results are reported with rewriting in the DALL-E 3 paper), SD3 achieves higher GenEval (0.74), and FLUX.1-dev achieves higher human preference (74.6). Emu3 is in the top tier but not "state-of-the-art" in the sense of best-in-class.

- On video generation, Emu3 scores below Gen-3 (82.32), Kling (81.85), and CogVideoX-5B (81.61). It is state-of-the-art among *open-source autoregressive* video models, but this is a category of one.

- On vision-language understanding, Emu3 is state-of-the-art among *encoder-free* models, substantially surpassing Fuyu-8B, EVE-7B, and Chameleon. But it does not surpass LLaVA-1.6-7B in aggregate, and several encoder-based models exceed LLaVA-1.6 on specific benchmarks.

The more accurate characterization is: **Emu3 achieves state-of-the-art performance among encoder-free unified multimodal models, and is competitive with (but not uniformly ahead of) the best task-specific architectures in generation and perception.**

**Claim 3: Next-token prediction is a promising and viable path toward general multimodal intelligence beyond language.**

The experiments support this claim as an **existence proof** — Emu3 demonstrates that a single next-token prediction model can handle image generation, video generation, video extension, and vision-language understanding at competitive performance levels. Prior to this work, no token-based model had demonstrated this breadth of capability at this performance level. Chameleon (the closest prior work) scored 0.39 on GenEval vs. Emu3's 0.66 and 69.6 on VQAv2 vs. Emu3's 75.1. Emu3 represents a qualitative leap in the demonstrated capability of purely autoregressive multimodal models.

However, the claim of "general multimodal intelligence beyond language" is a reach beyond what the experiments actually test. Emu3 is evaluated on standard benchmarks for specific tasks (generation, VQA, OCR). These benchmarks do not test for the kind of cross-modal reasoning, in-context multimodal learning, or emergent multimodal capabilities that would constitute "general intelligence." The paper does not evaluate whether Emu3 can, for example, generate an image from a description in one language and then answer questions about it in another, or whether it exhibits synergistic improvements from joint training (e.g., does video training improve image understanding?). These would be evidence of "general multimodal intelligence" beyond task-specific performance.

**Claim 4: Simplifying complex model designs unlocks significant potential for scaling both during training and inference.**

This is a forward-looking claim that the paper's experiments cannot directly verify — the paper demonstrates a single model at 8B parameters, not a scaling trend. Scaling studies (performance as a function of model size, data size, or compute) are entirely absent. The claim is supported indirectly by the architectural simplicity (which does make scaling more straightforward — no need to balance the sizes of separate encoders and decoders, no need to manage multiple training objectives) but is not empirically demonstrated.

The paper would be substantially strengthened by even a modest scaling study: training 1B, 4B, and 8B variants on the same data and measuring how performance scales. The absence of any scaling analysis is the paper's most significant empirical gap, given that the central argument is about the *scalability* of the next-token prediction paradigm for multimodal data.

**Critical gaps and weaknesses in the experimental design:**

1. **No compute-matched comparisons with baselines.** The paper compares output quality but never compares training compute, inference compute, or wall-clock time. A diffusion model that achieves 0.55 GenEval with 50 denoising steps and an autoregressive model that achieves 0.66 with 4,096 autoregressive steps have very different computational profiles. Without compute-matched comparisons, the efficiency argument for the unified architecture is unsubstantiated.

2. **Training data overlap on VLM benchmarks.** The asterisk notation on 8 of 15 vision-language benchmarks is transparent but does not resolve the contamination concern. If Emu3's training data included test images from ScienceQA-Img, its 89.2 score (vs. LLaVA-1.6's 64.9) may reflect memorization rather than superior visual reasoning. The paper should provide decontaminated evaluation results or at minimum detail the extent of overlap.

3. **No video generation human evaluation.** The VBench results show Emu3 struggling on Human Action (77.71) and Multiple Objects (44.64). Human evaluation would reveal whether these automated scores translate to perceptible quality differences. The absence of human evaluation for video is a significant gap given that video is one of Emu3's three modalities.

4. **No comparison with Emu2 or other in-house predecessors.** The paper's team previously developed Emu and Emu2 — unified multimodal models that used hybrid architectures (combining autoregressive and diffusion components). A direct comparison with Emu2-17B on the same benchmarks would quantify the improvement from the pure token-based approach and validate the paper's thesis that removing diffusion components doesn't hurt performance.

5. **Single scale point.** All experiments use the 8B model. Without scaling curves, the reader cannot assess whether Emu3's competitive performance is due to the architectural approach or simply to having reached a sufficient scale where representational capacity is no longer the bottleneck. If a 4B Emu3 significantly underperforms a 4B LLaVA, that would suggest the unified architecture requires more parameters to match specialized architectures — an important caveat to the paper's claims.

6. **No evaluation of training stability or data efficiency.** The paper does not report training loss curves, does not compare the amount of training data needed to reach a given performance level vs. diffusion or compositional methods, and does not discuss training stability issues (which are known to be challenging for discrete VAE training and multimodal pretraining from scratch).

7. **Video extension lacks quantitative evaluation.** The impressive qualitative examples in Figure 8 are not accompanied by any metrics — no FVD, no CLIP-score, no human evaluation, no comparison with interpolation or other extension methods. The claim of successful 8-second extension from 2 seconds of context is purely qualitative.

8. **Sample sizes for human evaluation are modest and unreported for variance.** With 100 prompts and 3 voters each, the confidence intervals on human preference scores could be wide. Differences of 3–5 points between models may not be statistically significant. The paper does not provide the information needed to assess this.

9. **The rewriting methodology for GenEval and T2I-CompBench is not fully specified.** While the paper states it uses GPT-4V as a rewriter following DALL-E 3, the specific prompt used for rewriting, the temperature, and the consistency of rewriting across models are not described. If rewriting changes the prompts in ways that specifically benefit Emu3 (e.g., making them more like the dense captions Emu3 was trained on), this would inflate Emu3's relative performance.

**What would strengthen the paper:**

- **Scaling studies** across model sizes (1B, 4B, 8B, and ideally larger) on a subset of benchmarks, showing how the performance gap between Emu3 and task-specific architectures changes with scale.
- **Compute-matched comparisons** measuring total FLOPs for training and inference against SDXL and LLaVA pipelines.
- **Decontaminated VLM evaluation** on benchmarks where training data overlap is confirmed absent.
- **Ablations of key design choices:** codebook size, vision token loss weight, QFT resolution increase, and the contribution of the supplementary understanding data to VLM performance.
- **Quantitative video extension evaluation** with metrics and baselines.
- **Comparison with Emu2** to quantify the benefit of moving from a hybrid to a pure token-based architecture.

In summary, the experimental results robustly demonstrate that Emu3 is the first pure next-token prediction model to achieve competitive performance with task-specific architectures across generation and perception. The evidence for "outperforming" those architectures is conditional — strongest on specific domains (OCR, documents, long-prompt image generation) and weaker on general visual understanding and video quality. The experiments support the paper's core thesis (next-token prediction suffices for multimodal tasks) as an **existence proof** but do not establish the superiority or scalability claims that would make this the definitive paradigm shift the paper argues for.

## 6. Limitations and Trade-offs

### The Difficulty Estimation Overhead Makes the Headline Efficiency Gains Partially Theoretical

**The assumption or constraint.** The paper's compute-optimal framework depends on knowing each prompt's difficulty before allocating test-time compute. The method described for estimating difficulty — generating 2,048 samples per question and computing either the pass@1 rate (oracle) or the average PRM final-answer score (predicted), then binning into five quintiles — is extraordinarily expensive. The paper acknowledges this cost explicitly in Section 3.2:

> "estimating difficulty in this way still incurs additional computation cost during inference... our experiments do not account for this cost largely for simplicity"

**The consequence.** The reported 4× efficiency gains over best-of-N are computed *after* difficulty is known, without amortizing the 2,048-sample estimation cost. At 2,048 samples per prompt, the difficulty estimation step alone consumes more compute than the largest test-time budgets studied (256–512 generations). In a realistic deployment, the true cost is `difficulty_estimation + strategy_execution`, and for many prompts the former dominates the latter. A practitioner who deploys this system naively — generating 2,048 samples to estimate difficulty before allocating a budget of 64 generations — would spend ~32× more compute on estimation than on the actual problem-solving, making the system far *less* efficient than a uniform best-of-N policy.

This is not a minor implementation detail. The entire practical value proposition of the paper depends on difficulty estimation being (a) cheap relative to the test-time budget, (b) accurate enough to route prompts to the right strategy, or (c) amortized across many similar queries. None of these conditions is demonstrated.

**What evidence exists in the paper.** The paper provides no cost-benefit analysis for difficulty estimation. The predicted-difficulty-bins curves (Figures 4 and 8) overlay the oracle-difficulty-bins curves but are generated using the *same* 2,048-sample cost — the only difference is that predicted bins use the PRM score instead of ground-truth correctness, not that they use fewer samples. There is no experiment showing how accuracy degrades with fewer difficulty-estimation samples, and no experiment showing at what sample count the total cost (estimation + execution) becomes favorable relative to a uniform baseline.

**Mitigation status.** The paper flags this as "a key avenue for future work" (Section 3.2) and suggests two directions: pretraining or fine-tuning models to predict difficulty directly from the question text, and adaptive difficulty estimation (using initial samples from the problem-solving process itself to assess difficulty). Neither direction is explored experimentally. Until one is demonstrated, the 4× efficiency claim should be treated as an upper bound on achievable efficiency given perfect, cost-free difficulty information — a theoretical ceiling rather than a realized deployment gain.

---

### Hard Problems Remain Fundamentally Unsolved, Demarcating a Hard Capability Boundary

**The assumption or constraint.** The paper's entire approach — both PRM-guided search and iterative revisions — operates by amplifying the base model's existing capability: search finds correct solutions that exist in the proposal distribution, and revisions refine solutions that are already close to correct. Neither mechanism can create capability the base model lacks. The paper is explicit about the boundary condition this creates. In Section 7, the authors state that on the hardest problems:

> "test-time compute provides essentially zero benefit regardless of budget"

**The consequence.** On difficulty bin 5 questions (the hardest quintile, where the base model's pass@1 is near zero), every method studied — best-of-N, beam search, lookahead search, sequential revisions, and their compute-optimal combinations — produces accuracy of 1–3%, indistinguishable from random guessing on MATH's multiple-choice format (Figure 3 right, Figure 7 right, Figure 9 bottom curves). No amount of test-time compute helps.

This demarcates a hard boundary that limits the approach's applicability. If a user's problem distribution includes a non-trivial fraction of questions outside the base model's capability range, no amount of inference-time strategy optimization will produce correct answers. The system will silently return wrong answers, and the compute-optimal policy has no mechanism for recognizing when a problem is *fundamentally* unsolvable versus merely *difficult but solvable with enough budget*. A practitioner deploying this system needs an independent method for detecting out-of-capability queries and routing them elsewhere (e.g., to a larger model or a human), which the paper does not provide.

This boundary also constrains the FLOPs-matched comparison. Test-time compute with the smaller model outperforms a ~14× larger model *only on easy-to-medium problems*, and only when the inference-to-pretraining token ratio R is favorable. On hard problems and at high R, the larger model wins decisively (Figure 9, bins 4–5). The substitution of test-time compute for pretraining is therefore not general — it works precisely where the base model already has non-trivial capability, and fails where it does not.

**What evidence exists in the paper.** The evidence is stark and consistent across all methods. In Figure 3 (right), bin 5 accuracy for both beam search and best-of-N weighted is flat at 1–3% from 4 to 256 generations. In Figure 7 (right), bin 5 accuracy is 2–3% regardless of sequential-to-parallel ratio. In Figure 9, the bin 5 scaling curves for both revisions and search are flat near 0–5% while the 14× larger model's stars sit above them, showing that pretraining — not test-time compute — is the only path to improvement on these problems.

**Mitigation status.** The paper does not attempt to solve this limitation. It acknowledges it transparently in the Section 7 takeaway and in the concluding discussion, but offers no solution. This is a fundamental limitation of the approach rather than a fixable oversight — test-time compute amplifies existing capability, and no amount of amplification can substitute for capability that does not exist. The implication is that a complete deployment strategy would need to combine compute-optimal test-time scaling with a mechanism for escalating out-of-capability queries to a more capable model, but this is not explored.

---

### The Revision Model Has a Structural Tendency to Revert Correct Answers to Incorrect Ones

**The assumption or constraint.** The revision model is trained exclusively on trajectories where all in-context answers are incorrect and the target is correct. Section 6.1 describes the training data construction: sequences of 0–4 incorrect answers followed by a correct answer, with the last incorrect answer selected by edit distance to the correct answer. The model is never trained on examples where the current answer is already correct, because the data construction pipeline produces only incorrect-to-correct trajectories.

**The consequence.** At test time, when a revision chain produces a correct answer at some step, the model has no training signal for what to do — it has never seen a correct answer in context during training. The result is an approximately **38% correct-to-incorrect reversion rate**, as reported in Section 6.1:

> "approximately 38% of correct answers get converted back to incorrect ones"

This means that even when the model successfully produces the right answer at step *k*, there is a 38% chance that step *k*+1 will "revise" it into a wrong answer. The paper mitigates this with within-chain selection (majority voting or verifier-based selection across all steps of the revision chain, picking the best answer from any point rather than always taking the final revision). However, this mitigation is incomplete:

- **Majority voting** works only if the correct answer appears more often than any single incorrect answer across the chain. If the model produces the correct answer once at step 5 and then 10 different incorrect answers across 15 subsequent steps, majority voting selects an incorrect answer.
- **Verifier-based selection** relies on the revision-specific ORM accurately scoring revision model outputs. However, the paper shows in Appendix J (Figure 15a) that the base-LM PRM underperforms on revision model outputs due to distribution shift, requiring a separate ORM trained specifically on revision data. Even this revision-specific ORM is imperfect — it must distinguish correct from incorrect answers in a chain where the model has demonstrated it can produce both, and its accuracy sets an upper bound on the effectiveness of within-chain selection.

The net effect is that the revision model is less reliable than its pass@1 trajectory (Figure 6 left) suggests: while the per-step accuracy does improve from ~18% to ~24% across the chain, the 38% reversion rate means the chain contains *fewer* correct answers than it would if the model could recognize when to stop revising.

**What evidence exists in the paper.** The 38% reversion rate is quoted in Section 6.1. The within-chain selection mitigation is described in the same section, and its effectiveness can be inferred from Figure 6 (right), where sequential + best-of-N weighted achieves ~41.5% at 64 generations versus fully parallel + best-of-N weighted at ~39%. The 2.5-point gap, while positive, is modest — near what one would expect from a system that produces correct answers at some steps but cannot reliably preserve them.

**Mitigation status.** The paper's mitigation (majority voting or verifier-based selection across the chain) is a patch, not a solution. A principled fix would require training the revision model to recognize when no revision is needed — e.g., by including "no-change" trajectories in the training data where the model learns to output the same answer when it is already correct, or by adding an explicit stopping criterion. The paper does not explore these directions. The ReST^EM experiment in Appendix K (Figure 16), where an attempt to further optimize the revision model caused performance to *degrade* with sequential revisions, suggests that the revision training procedure is fragile and that the positive results depend on specific design choices (offline data construction, edit-distance pairing) that may not transfer to other settings.

---

### The 14× Larger Model Baseline Is Not Compute-Optimally Trained, Weakening the Pretraining-Vs-Inference Comparison

**The assumption or constraint.** The FLOPs-matched comparison in Section 7 compares PaLM 2-S\* with compute-optimal test-time scaling against a model with approximately 14× more parameters. The paper explicitly acknowledges that this larger model is scaled in *parameters only*, with training data held fixed, following the LLaMA paradigm (Touvron et al., 2023):

> "We choose this setting as it is representative of a canonical approach to scaling pretraining compute and leave the analysis of compute-optimal scaling of pretraining compute where the data and parameters are both scaled equally to future work."

**The consequence.** A compute-optically trained model (following Chinchilla scaling laws, where both parameters and data are scaled with total compute) would allocate the 14× FLOPs budget differently — roughly 4× more parameters and 3.5× more data, rather than 14× more parameters and identical data. The parameter-only-scaled model used as the baseline likely **underperforms** what a compute-optimal larger model would achieve, because it is overparameterized relative to its training data. This makes the pretraining baseline weaker than it should be for the claim "test-time compute can substitute for pretraining."

The direction of the bias is clear: making the pretraining baseline stronger (by training it compute-optimally) would reduce or eliminate the reported advantages of test-time compute. The paper's key quantitative claim — that test-time compute with the smaller model can outperform a 14× larger model — is therefore conditional on the larger model being trained suboptimally. A compute-optimally trained 14× larger model might close the gap on easy questions and widen its lead on hard questions, potentially flipping the conclusion for some difficulty tiers and R values.

Additionally, the 14× larger model uses **greedy decoding only** — no majority voting, no best-of-N, no test-time compute of its own. In a fully symmetric comparison, both models would receive their respective test-time compute budgets, and the larger model could benefit from the same strategies (search, revisions, compute-optimal allocation) that boost the smaller model. The paper's comparison answers the question "Can test-time compute with a small model beat a large model with *no* test-time compute?" — a question that favors test-time compute by construction.

**What evidence exists in the paper.** The paper is transparent about this design choice in Section 7, explicitly acknowledging it as a deviation from compute-optimal pretraining and the LLaMA-style scaling framework. The comparison is not hidden or misrepresented. However, the paper's claims in the abstract and introduction ("a smaller model augmented with compute-optimal test-time strategies can outperform a ~14× larger pretrained model") do not carry this caveat, potentially misleading readers who do not read Section 7 carefully.

**Mitigation status.** The paper frames the parameter-only scaling as "representative of a canonical approach to scaling pretraining compute" and defers the compute-optimal pretraining comparison to future work. This is a reasonable choice given that many production models (LLaMA, Mistral, etc.) are trained with fixed data and scaled parameters, making the comparison practically relevant. However, it means the results should be interpreted as "test-time compute can compensate for *parameter scaling* under fixed data" rather than "test-time compute can compensate for *pretraining compute* in general." The latter claim, which the paper's framing implies, requires a Chinchilla-optimal baseline that is not provided.

---

### The Study Is Limited to a Single Model Family and a Single Benchmark, with No Evidence of Cross-Domain or Cross-Model Generalization

**The assumption or constraint.** All experiments use PaLM 2-S\* as the base model and the MATH benchmark (500 test questions) as the evaluation dataset. The paper states that it "believe[s] this model is representative of the capabilities of many contemporary LLMs" (Section 4), but this is asserted rather than demonstrated. There is no replication of any experiment on a different model family (e.g., LLaMA, GPT, Claude), a different reasoning benchmark (e.g., GSM8K, MMLU, ARC), or a non-math task domain (e.g., code generation, logical reasoning, scientific QA).

**The consequence.** The paper's central findings — that difficulty-conditioned allocation yields 4× efficiency gains, that beam search over-optimizes on easy problems, that sequential revisions help on easy problems but balanced strategies are better on hard ones, and that test-time compute can outperform a 14× larger model — are all **potentially specific to PaLM 2-S\* on MATH**. Several aspects could fail to transfer:

- **PRM quality and over-optimization behavior** depend on the base model's output distribution and error patterns. A model with different calibration (e.g., producing more confident but wrong answers) might exhibit different difficulty-dependent search scaling, with over-optimization kicking in at different budgets or difficulty levels.
- **The revision model's ability to learn from incorrect in-context examples** depends on the base model's in-context learning and self-correction capabilities, which vary substantially across model families (some models are much better at recognizing their own errors than others).
- **The optimal difficulty bin thresholds** — which determine when to switch from best-of-N to beam search, or from sequential to balanced revision — are fitted to PaLM 2-S\*'s pass@1 distribution on MATH. A different model on a different benchmark would produce different thresholds, and it is unknown whether the qualitative pattern (five bins, similar strategy assignments) persists.
- **MATH consists of competition-level math problems** requiring symbolic reasoning and producing well-defined answers that can be graded with exact string matching. The difficulty estimation (pass@1 from 2,048 samples, PRM final-answer scoring), PRM training (Monte Carlo rollout correctness), and revision training (correct/incorrect labeling) all depend on having clean correctness signals. In domains without such signals — code generation (where correctness depends on unit tests), open-ended QA, creative writing, dialogue — the entire pipeline from difficulty estimation through verifier training to answer selection would need fundamental redesign.

The test set size (500 questions, split into quintiles of ~100, further split by cross-validation into ~50 per fold) is also small enough that the compute-optimal policy is selected based on limited data. The paper does not report confidence intervals on the compute-optimal scaling curves, so the statistical reliability of the strategy assignments and the 4× efficiency figure is unknown.

**What evidence exists in the paper.** None. There are no cross-model experiments, no cross-benchmark experiments, and no analysis of how findings depend on the base model's characteristics. The paper's "representative" claim is purely a statement of belief.

**Mitigation status.** The paper does not acknowledge this as a limitation, beyond the scope defined in Section 4 (which notes the single model and benchmark but argues representativeness). No future work is suggested on cross-model or cross-domain validation. This is a significant gap because the paper's practical recommendations — deploy beam search on medium problems, use sequential revisions on easy ones, estimate difficulty via PRM score distributions — are presented as general principles, but they are validated only on a single narrow setting. A practitioner deploying these methods on a different model, task, or domain has no empirical basis for expecting the same efficiency gains or the same difficulty-dependent strategy assignments.

## 7. Implications and Future Directions
- How it changes the landscape
  - Demonstrates that a single next‑token decoder can rival specialized diffusion and CLIP+LLM stacks across generation and perception (Figure 2; Tables 4–6). This simplifies system design and suggests scaling laws for multimodal models that mirror language LMs.

- Follow‑up research enabled
  - Tokenization
    - Explore larger or hierarchical codebooks, variable‑rate compression, or learned entropy models to improve fidelity and efficiency.
  - Training curriculum
    - Better balancing of text/vision losses, curriculum over sequence lengths, and multi‑scale training for aesthetics and dynamics.
  - Alignment
    - Richer preference datasets (diverse cultures/languages/domains), multi‑objective DPO for both alignment and metric‑based faithfulness.
  - Long‑horizon video
    - Memory‑efficient attention, chunked generation with cross‑chunk constraints, and error‑correction during autoregressive extension.
  - Broader modalities and tasks
    - Add audio tokens for AV generation/understanding; unify detection, segmentation, and OCR by predicting structured tokens; integrate retrieval‑augmented generation inside the same next‑token framework.

- Practical applications
  - Text‑to‑image/video creation for media, education, and marketing; video continuation and future prediction for storyboarding and simulation; vision‑language assistants for diagrams, charts, and OCR‑heavy documents (Table 6 categories). The open‑sourced tokenizer and model components lower the barrier to such deployments.

> In summary, Emu3 provides concrete evidence (Tables 4–6; Figures 5–8) that next‑token prediction alone, when backed by an effective vision tokenizer and long‑context Transformer, can support high‑quality multimodal generation and understanding. While aesthetics and some understanding benchmarks still leave headroom, the simplification, competitiveness, and extensibility make this a compelling path for general multimodal intelligence.
