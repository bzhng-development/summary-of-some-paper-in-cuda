# SAIL-VL2 Technical Report

**ArXiv:** [2509.14033](https://arxiv.org/abs/2509.14033)

## 🎯 Pitch

SAIL-VL2 introduces a next-generation open vision–language model suite that achieves state-of-the-art multimodal understanding and reasoning at small parameter scales (2B/8B), leveraging a highly curated data pipeline, a progressively aligned vision encoder, and a hybrid supervised fine-tuning plus reinforcement learning strategy. This innovation matters because it delivers top-tier perception and complex reasoning—across images, documents, and video—at a fraction of the computational cost, paving the way for practical, scalable, and efficient deployment of advanced AI in diverse real-world applications.

---

## 1. Executive Summary

This paper introduces **SAIL-VL2**, an open-suite vision-language foundation model designed for comprehensive multimodal understanding and reasoning that achieves state-of-the-art performance at the 2B and 8B parameter scales. The system is built on three core innovations: a large-scale data curation pipeline with scoring and filtering strategies (encompassing quality assessment via ITA/VIR judge models and chart-to-caption synthetic generation); a **thinking-fusion SFT–RL hybrid paradigm** that systematically strengthens reasoning capabilities (combining LongCoT supervised fine-tuning, verifiable-reward RL, and mixed-reward RL); and architectural advances extending to sparse Mixture-of-Experts designs. SAIL-VL2 delivers competitive performance across 106 datasets, with SAIL-VL2-2B ranking first on the OpenCompass leaderboard among officially released open-source models under 4B parameters and SAIL-VL2-8B-Thinking setting a new state-of-the-art for open-source models on multimodal reasoning benchmarks (54.4 average, surpassing Gemini-2.0-Flash at 50.6 and approaching GPT-4o-latest at 54.8), establishing that compact architectures can match or exceed far larger models on challenging reasoning tasks when trained with progressive alignment and RL-based thinking mechanisms.

## 2. Context and Motivation

### The Core Problem: Efficiency vs. Scale in Vision-Language Models

The fundamental tension this paper addresses is how to build vision-language models (LVMs) that achieve strong multimodal reasoning capabilities without the prohibitive computational costs of simply scaling up model parameters and training data. The paper frames this through a direct contrast with the dominant industry paradigm:

> "Scaling up model parameters and training data to make LVMs 'larger' has emerged as a pivotal approach for pushing the performance boundaries of LVMs... While this paradigm has yielded substantial performance gains, it also imposes considerable challenges in terms of computational demands and training, as well as deployment costs."

This matters because LVMs have become central infrastructure for a wide range of applications — from document understanding and OCR to scientific reasoning and interactive assistants — but the prevailing scaling approach creates a sharp divide between what's possible in well-resourced industrial labs and what's accessible to the broader research community and practitioners with limited hardware budgets. The paper explicitly positions itself against this trend:

> "In contrast, our SAIL-VL series focuses on developing efficient LVMs, aiming to explore 'how knowledge can be effectively injected through efficient architectures and training strategies,' thereby establishing an open-source model family that embodies the principle of 'small model, strong performance.'"

The question is not merely academic. If effective multimodal reasoning can be achieved at the 2B–8B parameter scale rather than requiring 70B+ parameters, it fundamentally changes the economics of deployment: edge devices, real-time applications, and open-source reproducibility all become viable in ways that large-scale models preclude. The paper's stated goal — to demonstrate that compact architectures can match or exceed far larger models on challenging reasoning tasks — addresses this practical bottleneck directly.

### The Three Interlocking Gaps in Prior Work

The paper identifies deficiencies in existing approaches along three dimensions that collectively limit the efficiency of vision-language models at smaller scales.

#### Gap 1: Data Quality and Diversity Are Underserved in Pre-Training Corpora

The prevailing approach to LVM pre-training relies on large-scale image–caption datasets (e.g., those built from web-crawled alt-text or automated captioning pipelines). While these datasets provide scale, they introduce systematic quality issues that the paper argues are poorly addressed in prior work:

- **Hallucinated or imprecise captions.** Automated captioning models produce text that may misrepresent visual content — describing objects that aren't present, omitting key details, or using generic language that fails to capture fine-grained visual information. The paper's own analysis of SAIL-Caption (their prior dataset from Dong et al., 2025a) revealed that "15–20% of captions fall below an acceptable threshold (score <3)", indicating substantial noise even in a curated corpus.
- **Homogenized linguistic patterns.** When captions are generated by LLMs or LVMs at scale, the output distribution tends toward formulaic phrasings with limited lexical and structural diversity. This biases the model toward learning narrow language patterns rather than robust, generalizable multimodal associations.
- **Missing data modalities.** Standard caption datasets emphasize natural images and general scene descriptions but underrepresent specialized visual domains — particularly charts, tables, and documents — that are critical for real-world reasoning tasks. The paper notes that chart understanding requires dedicated data curation because chart captions differ fundamentally from natural image captions in their emphasis on data relationships, axes, and quantitative interpretation.

Prior dataset curation efforts (ShareGPT4V, LLaVA-OneVision, Cauldron) have increased scale and task diversity but have not systematically addressed the joint problems of *quality filtering* and *distributional balance* in a unified pipeline. The paper argues that this oversight results in models that are trained on noisy signals, learning spurious correlations rather than robust visual-linguistic alignments.

#### Gap 2: Vision Encoders Are Insufficiently Aligned with LLM Representation Spaces

Most LVMs employ a vision encoder (typically a ViT variant) connected to an LLM through a lightweight adapter (often a single- or two-layer MLP). The standard training approach treats the vision encoder as a static feature extractor, either keeping it frozen during multimodal training or fine-tuning it jointly with the LLM only in later stages. The paper argues this creates a **modality gap** — a systematic mismatch between the representation spaces of vision and language tokens — that limits the model's ability to perform fine-grained reasoning that requires tight integration of visual and linguistic information.

The evidence for this gap is both conceptual and empirical:

- Vision encoders trained on image-only objectives (classification, contrastive learning, autoregressive pixel prediction) learn representations optimized for visual tasks, not for seamless integration with language models. As the paper states, "visual and linguistic tokens originate from heterogeneous spaces and must be aligned into a unified representation."
- When the gap is large, the adapter — typically just a two-layer MLP — must perform a difficult compression that can lose visual detail, particularly for tasks requiring precise spatial reasoning, OCR, or fine-grained attribute recognition.
- The paper's visualizations (Figure 6) and distance metric analysis (Table 7) quantify this gap directly: baseline vision encoders produce visual feature clusters that are more dispersed and show less overlap with LLM text embeddings compared to the paper's proposed SAIL-ViT, which is progressively trained to close this gap.

Prior work has attempted to address alignment through various means: using larger adapters (which add parameters), contrastive pre-training objectives (which are computationally expensive), or joint training from the start (which can destabilize the LLM's language capabilities). The paper argues these approaches either add computational overhead or sacrifice one modality for the other, leaving a gap for a method that achieves tight alignment without compromising efficiency.

#### Gap 3: Reasoning Capabilities Require Deliberate Training, Not Just Scale

The most significant gap the paper addresses concerns complex reasoning. Standard LVM training — pre-training on caption/VQA data followed by instruction tuning — produces models that can answer straightforward visual questions but struggle with tasks requiring multi-step logical deduction, mathematical reasoning, or spatial-temporal inference. Prior work has pursued two strategies to address this:

**Chain-of-Thought (CoT) prompting during inference** — instructing the model to "think step by step" — has shown improvements on reasoning benchmarks but is fundamentally limited: the model was not trained to produce structured reasoning, so CoT outputs are often incomplete, logically inconsistent, or irrelevant to the final answer. The paper argues that prompting alone cannot compensate for the absence of explicit reasoning supervision during training.

**Scaling model size** has been the dominant response to reasoning failures. The implicit assumption is that larger models (70B, 100B+ parameters) develop emergent reasoning capabilities through increased capacity. This is evident in the paper's comparison tables (Tables 8, 9, 10), where the strongest baselines — Qwen2.5-VL-72B, InternVL3-78B, GPT-4o, Gemini-2.0-Pro — are orders of magnitude larger than the 2B–8B models the paper targets. However, this approach is fundamentally inefficient: it requires massive computational investment in pretraining for capabilities that may be achievable through more targeted training strategies at smaller scales.

The paper further identifies a subtle failure mode in how prior work has handled the relationship between reasoning and general capabilities. Models fine-tuned exclusively on CoT data often exhibit **reasoning mode collapse** — they generate verbose step-by-step analyses even for simple questions where a direct answer would be more appropriate. Conversely, models trained only on direct-answer formats fail to produce structured reasoning when needed. The paper argues that prior work lacks a mechanism for teaching models to *selectively* employ reasoning based on task difficulty, resulting in either inefficient output or inadequate problem-solving.

### The Training Paradigm Gap: From Teacher Forcing to Hybrid SFT–RL

Underlying all three gaps is a broader shift in training paradigms that the paper positions itself within but argues remains incompletely realized for vision-language models. The introduction notes:

> "Training paradigms have progressed from teacher-forcing supervised learning to hybrid methods, integrating supervised fine-tuning (SFT) and reinforcement learning (RL) for self-improvement."

In the language-only domain, RL-based training (particularly RLHF and its variants) has become standard for aligning models with human preferences and improving reasoning. However, the paper argues that the application of RL to *multimodal* reasoning is nascent and poorly systematized. Prior work has explored:

- **RLHF for multimodal instruction following**, which improves helpfulness and safety but does not specifically target reasoning capabilities.
- **Verifier-guided search at inference time**, which adds compute during generation but does not improve the model's internal reasoning policy.
- **Rejection sampling from SFT models**, which filters for correct reasoning chains but does not provide gradient-based optimization for the reasoning process itself.

The paper positions its thinking-fusion SFT–RL pipeline as filling this gap: a systematic framework for using reinforcement learning with verifiable rewards to optimize reasoning, followed by a fusion stage that integrates reasoning capabilities with general instruction-following without mode collapse.

### The Specific Positioning: SAIL-VL1.5 as the Foundation, Not the Ceiling

The paper builds explicitly on its predecessor, SAIL-VL (Dong et al., 2025a), and an intermediate version (SAIL-VL1.5, referenced in the judge model experiments in Table 2). This lineage is important because it establishes that the contributions are incremental improvements to a working system rather than a de novo architecture. The paper's advances over SAIL-VL include:

- **Data**: Upgrading SAIL-Caption to SAIL-Caption2 with systematic quality filtering (the ITA/VIR judge models and chart caption expansion); introducing SAIL-Instruction2 with latent-class-based diversity balancing; and constructing SAIL-Video with alignment and difficulty scoring.
- **Training**: Adopting a three-stage progressive vision encoder training schedule (warm-up adaptation → fine-grained alignment → world knowledge injection) rather than the simpler alignment approaches in SAIL-VL; introducing AdaLRS for dynamic learning rate adjustment; and developing the full thinking-fusion pipeline (LongCoT SFT → verifiable-reward RL → Think-Fusion SFT → mixed-reward RL) that extends beyond SAIL-VL's SFT-only approach.
- **Architecture**: Extending from dense LLMs to MoE designs (Qwen3-30B-A3B) with specialized load-balancing strategies; introducing SAIL-ViT-AnyRes for arbitrary-resolution support.

The paper's cumulative claim is that these improvements, applied systematically, enable the 2B model to surpass prior sub-4B state-of-the-art models (Qwen2.5-VL-3B, InternVL3.5-2B, Ovis-U1-3B) and the 8B model to match or exceed models with 10–40× more parameters on multimodal reasoning benchmarks.

### The Broader Landscape: Why "Small Model, Strong Performance" Matters Now

The paper's motivation is also shaped by timing within the field. The release of several strong open-weight models in the 2B–8B range (Qwen2.5-VL-3B/7B, InternVL3-2B/8B, Ovis2-2B/8B) has created a competitive landscape where efficiency is a differentiating factor. However, the paper argues that none of these models have simultaneously achieved:

1. **Competitive performance on general multimodal understanding** (MMBench, MMStar, DocVQA, OCRBench).
2. **State-of-the-art complex reasoning** (MathVista, MathVision, LogicVista) through structured thinking mechanisms.
3. **Fine-grained visual grounding** (RefCOCO) and high-resolution understanding.
4. **Video understanding** across short and long temporal horizons.

The paper's explicit goal is to be the first open-source model suite to deliver on all four dimensions at the 2B and 8B scales, demonstrating that comprehensive multimodal intelligence is not exclusively the domain of 70B+ parameter models. This is framed not as a methodological novelty but as an engineering contribution — systematically addressing data quality, training strategy, and architecture to extract maximum capability from compact models.

### The Unstated Stakes: Open-Source vs. Closed-Source Capability Gaps

Though not directly addressed in the introduction, the paper's comparison tables (Tables 8, 9, 10) reveal a deeper motivation. The leading reasoning models — GPT-4o, Gemini-2.0-Pro, Claude-3.7-Sonnet — are closed-source and inaccessible for inspection, modification, or on-device deployment. The open-source models that exist either lag significantly on reasoning (InternVL3-8B at 41.4 on the OpenCompass reasoning average vs. GPT-4o's 54.8) or achieve strong performance only at scales that remain difficult to deploy (InternVL3-78B at 51.0). SAIL-VL2-8B-Thinking's achievement of 54.4 — surpassing Gemini-2.0-Flash (50.6) and approaching GPT-4o-latest (54.8) — represents a meaningful narrowing of this gap at a deployable scale. The paper's emphasis on releasing "the full SAIL-VL2 model suite along with its inference code" underscores this commitment to open access.

## 3. Technical Approach

### 3.1 Reader Orientation

SAIL-VL2 is a vision-language model (a neural network that takes images or videos as input along with text instructions and produces text responses) built by integrating a pre-trained vision encoder with a pre-trained large language model through a lightweight connector, then training the entire system through a carefully sequenced pipeline of data curation, multimodal alignment, and reasoning enhancement. The system solves the problem of how to achieve state-of-the-art multimodal understanding and complex reasoning at small model scales (2B–8B parameters) by systematically addressing three bottlenecks: noisy pre-training data that limits alignment quality, a modality gap between vision and language representations that impairs fine-grained understanding, and the absence of explicit reasoning training that leaves even capable models unable to perform multi-step logical deduction. The shape of the solution is a progressive training curriculum — starting from coarse visual-linguistic alignment and advancing through increasingly sophisticated stages that culminate in a hybrid supervised-fine-tuning-plus-reinforcement-learning pipeline that teaches the model to selectively engage in step-by-step reasoning — combined with a data curation infrastructure that filters, enriches, and rebalances training corpora to maximize the signal-to-noise ratio of every training example.

### 3.2 Big-Picture Architecture (Diagram in Words)

The SAIL-VL2 system consists of five major components arranged in a processing pipeline:

1. **SAIL-ViT Vision Encoder:** A Vision Transformer (ViT) that converts raw images or video frames into sequences of visual feature vectors (tokens). SAIL-ViT comes in two variants: a fixed-resolution encoder that processes 448×448 image crops, and SAIL-ViT-AnyRes that supports arbitrary input resolutions through interpolation-based positional embeddings. The encoder is pre-trained through a three-stage progressive alignment strategy (warm-up adaptation → fine-grained alignment → world knowledge injection) that incrementally aligns its output representations with the downstream LLM's embedding space, rather than treating vision encoding as a separate, frozen preprocessing step.

2. **Vision-Language Adapter:** A lightweight two-layer MLP that projects the SAIL-ViT output from the vision embedding dimension into the LLM's token embedding dimension. This is a standard connector architecture (no cross-attention, no learnable queries) that performs pure dimensional projection. It is randomly initialized and trained from scratch during multimodal pre-training.

3. **Large Language Model Backbone:** The text-processing core, initialized from the Qwen3-Instruct family. The paper explores both dense architectures (Qwen3-1.7B for the 2B variant, Qwen3-8B for the 8B variant) and sparse Mixture-of-Experts architectures (Qwen3-30B-A3B, which has 30.5B total parameters but only 3B activated per token). The LLM receives a concatenated sequence of visual tokens (from the adapter) and text tokens (from the tokenizer) and performs next-token prediction autoregressively.

4. **Data Curation Pipeline:** An offline infrastructure that constructs, filters, and balances the training corpora used at each training stage. This includes: SAIL-Caption2 (250M general captions + 1.69M chart captions, filtered by ITA/VIR judge models), synthetic VQA data (caption-to-QA conversion using LLMs), SAIL-Instruction2 (20M instruction-tuning samples with latent-class-based diversity balancing), SAIL-Video (5.1M video-QA samples scored on alignment, content richness, and difficulty), and multimodal CoT data (400K LongCoT samples + 1M Think-Fusion samples curated through redundancy filtering, answer distillation, and length balancing).

5. **Training Orchestrator:** The training infrastructure that executes the staged training curriculum. This includes: a basic multimodal pre-training stage (training only the adapter on caption/OCR data), a multi-task pre-training stage (unfreezing all parameters and training on the full data mixture with 360B tokens), a basic SFT stage (curriculum learning across four phases with model soup merging), and the thinking-fusion pipeline (LongCoT SFT → verifiable-reward RL → Think-Fusion SFT → mixed-reward RL). The orchestrator also incorporates AdaLRS (adaptive learning rate search) during basic pre-training and stream packing strategies for training efficiency.

**Information flow at inference time:** An image (or video frames) enters through SAIL-ViT → adapter projects visual tokens → these are concatenated with tokenized text instruction → the LLM processes the full multimodal sequence autoregressively → if the model is a "thinking" variant, it generates reasoning steps inside `〈thinking〉〈/thinking〉` tags before producing the final answer in `\boxed{}` tags.

### 3.3 Roadmap for the Deep Dive

- **First, SAIL-ViT progressive training** — because the vision encoder is the foundation upon which all multimodal understanding is built, and its three-stage progressive alignment strategy is the key mechanism for closing the modality gap that limits finer-grained reasoning. Understanding how SAIL-ViT is trained (what freezes/unfreezes at each stage, what data is used, what learning rates are applied) is essential before examining how it integrates with the LLM.

- **Second, pre-training data construction and filtering** — because the quality of the training data directly determines what the model can learn, and SAIL-VL2's data pipeline is substantially more sophisticated than standard approaches. This section covers SAIL-Caption2 (with the ITA/VIR judge models), synthetic VQA generation, SAIL-Video filtering, and the two-step data resampling strategy (dataset-level and linguistic-level).

- **Third, the pre-training recipe** — covering the two-stage pre-training pipeline (basic multimodal pre-training and multi-task pre-training), the AdaLRS algorithm, and the scaling law analysis. This section explains *how* the carefully curated data is consumed during training and what training dynamics emerge.

- **Fourth, post-training data construction** — covering SAIL-Instruction2 (with latent-class-based filtering and re-annotation), SAIL-Video, and multimodal CoT data curation (redundancy filtering, answer distillation, CoT length balancing). This sets up the data foundation for the reasoning enhancement stages.

- **Fifth, the thinking-fusion SFT–RL pipeline** — the most novel contribution. This covers LongCoT SFT (training the model to generate step-by-step reasoning), verifiable-reward RL (using DAPO/GSPO with correctness and format rewards), Think-Fusion SFT (merging reasoning exemplars with direct-answer data to prevent mode collapse), and mixed-reward RL (adding a thinking reward signal via LVM-based judges). This section explains the reward formulations, training recipes, and design rationale.

- **Sixth, architectural extensions** — covering the Mixture-of-Experts design (load-balancing strategies, data probing for expert activation entropy), SAIL-ViT-AnyRes (arbitrary resolution through interpolated 2D RoPE), and the stream packing infrastructure. These are enabling technologies that make the training pipeline feasible at scale.

### 3.4 Detailed, Sentence-Based Technical Breakdown

This is primarily an **engineering contribution paper** whose core idea is that a systematically designed data curation and progressive training pipeline — culminating in a hybrid SFT–RL paradigm that teaches models to selectively engage in structured reasoning — can achieve state-of-the-art multimodal understanding and complex reasoning at compact model scales (2B–8B parameters) without requiring the massive computational investment of training 70B+ parameter models.

---

#### SAIL-ViT Progressive Training Strategy

**What SAIL-ViT is.** SAIL-ViT is not a novel vision transformer architecture but rather a training methodology applied to existing ViT backbones (AIMv2-Large and AIMv2-Huge, which are vision encoders pre-trained via multimodal autoregressive objectives on image-text pairs) to progressively align their output representations with the embedding space of the downstream LLM. The insight driving this approach is that standard vision encoders — trained on image-only objectives like classification or contrastive learning — produce features that are poorly matched to the distribution of linguistic token embeddings, creating a "modality gap" that the adapter must bridge. By progressively injecting multi-granularity multimodal training signals, SAIL-ViT shifts the vision encoder's representation space closer to the LLM's, reducing the burden on the adapter and enabling more precise cross-modal reasoning.

**Stage I: Warm-up adaptation.** In this stage, both the pre-trained vision encoder (AIMv2) and the pre-trained LLM (Qwen3-Instruct) are kept completely frozen — their parameters are not updated. Only the randomly initialized two-layer MLP adapter is trained. The training data consists of 8 million simple multimodal understanding examples: 4.9M captioning samples from the SAIL-Caption dataset and 3.1M OCR samples from the IDL-WDS dataset (which contains industry document images with text annotations). The learning rate is set to $2 \times 10^{-4}$ and training proceeds for one epoch with a batch size of 1920.

**Why this stage exists.** The adapter starts from random initialization and has no prior knowledge of how to map vision features to language space. Training it in isolation — without also modifying the vision encoder or LLM — ensures that the adapter learns a coarse but stable mapping before more complex interactions are introduced. If the vision encoder were also unfrozen at this stage, the adapter's random initialization could produce large gradients that destabilize the pre-trained vision features. If the LLM were unfrozen, the noisy early-stage adapter outputs could degrade the LLM's carefully pre-trained language representations. The caption+OCR data mix is chosen because these are "simple" tasks with clear visual-linguistic correspondences: captions require mapping global image content to descriptive language, and OCR requires mapping localized text regions to character sequences, providing complementary alignment signals.

**Stage II: Fine-grained alignment.** Here, the LLM remains frozen but the vision encoder parameters are now unfrozen alongside the adapter (which continues training). The training data expands beyond Stage I: an additional 6.7M captioning samples from SAIL-Caption are added, the OCR source is supplemented with DocStruct (a document structure understanding dataset that provides hierarchical layout annotations), and video-caption data is incorporated for the first time. The learning rate is reduced to $2 \times 10^{-5}$ and the batch size is 512.

**Why this stage exists.** With the adapter now providing a reasonable initial mapping, the vision encoder can be safely fine-tuned to optimize its features specifically for multimodal understanding. The expanded data diversity — particularly the addition of video and document structure data — exposes the vision encoder to temporal sequences and spatially structured visual information that pure image captioning doesn't provide. The reduced learning rate prevents catastrophic forgetting of the vision encoder's pre-trained visual recognition capabilities while still allowing meaningful adaptation.

**Stage III: World knowledge injection.** In this final stage, all parameters are unfrozen: the vision encoder, adapter, and LLM are trained jointly. The data expands dramatically to 36.5M samples spanning: captioning data (from PixMo and SAIL-Caption), OCR data (DocVQA, IDL-WDS, DocStruct, PixMo), open-ended QA, math reasoning (MathQA, MathV360K, Geometry3K, MAVIS), short-form VQA (ST-VQA, VQAv2, OK-VQA), and pure text corpora (OpenHermes-2.5, MagPie). The learning rate is $1 \times 10^{-5}$ and training proceeds for one epoch with a batch size of 512.

**Why this stage exists.** The previous stages achieved visual-linguistic alignment but did so without modifying the LLM — the language model was essentially treated as a static decoder consuming aligned visual tokens. This stage allows the LLM to adapt its internal representations to better utilize the now-aligned visual features, and simultaneously allows the vision encoder to receive gradients backpropagated through the full LLM, enabling end-to-end optimization for downstream task performance. The inclusion of pure text data is critical: without it, the LLM — which is being trained on predominantly multimodal inputs — risks forgetting its language modeling and comprehension capabilities (a phenomenon known as catastrophic forgetting in continual learning). The math reasoning and diverse QA data expose the vision encoder to stimuli beyond captioning, teaching it to extract features relevant for higher-level reasoning rather than just descriptive generation.

**What the training produces.** The vision encoder that emerges from Stage III is designated as SAIL-ViT. Table 6 provides quantitative validation: on a zero-shot image classification benchmark averaging across ImageNet-1k, ImageNet-A, ImageNet-R, and ImageNet-V2, SAIL-ViT-Huge achieves 63.63% average accuracy compared to AIMv2-Huge's 61.52%, representing a +2.11% improvement. A further expanded version (SAIL-ViT-Huge-v2, trained with additional data during Stage III) achieves 64.25%. More importantly, as shown in Table 7 and Figure 6, SAIL-ViT produces visual feature distributions that are significantly closer to LLM text embedding distributions than the baseline AIMv2: across multiple LLM architectures (Qwen3-0.6B, Qwen3-1.7B, Qwen3-8B, InternLM2.5-1.8B), SAIL-ViT consistently achieves lower average nearest neighbor distance, lower Wasserstein distance, and lower mean overall distance between visual and textual feature clusters. This means the adapter has less "work" to do in bridging the modality gap, preserving more visual information for downstream reasoning.

---

#### SAIL-ViT Architecture Variants

**Fixed-resolution SAIL-ViT.** The standard variant accepts images at a fixed 448×448 pixel resolution. Each image is divided into non-overlapping 14×14 pixel patches, producing a 32×32 grid of 1,024 patches. Each patch is linearly projected into a token embedding, and learnable positional embeddings are added to encode spatial location. For high-resolution images (e.g., a 3580×2260 photograph), the image is tiled into multiple 448×448 crops, each processed independently through the ViT, and the resulting token sequences are concatenated. This tiling approach preserves full-resolution detail at the cost of increased token count (and thus increased LLM computation), but loses cross-tile spatial relationships since each crop is processed independently.

**SAIL-ViT-AnyRes.** This variant removes the fixed-resolution constraint by replacing learnable positional embeddings with an interpolation-based mechanism. Standard ViT positional embeddings are fixed-length vectors learned during pre-training — if you need to process an image at a resolution different from what the model was trained on, the positional embeddings don't generalize naturally. SAIL-ViT-AnyRes addresses this by taking the pre-trained fixed-size positional embeddings and interpolating them (using 2D RoPE, or Rotary Position Embeddings extended to two spatial dimensions) to match the actual input resolution:

> "conventional positional embeddings are fixed-length and hard to generalize, we adopt an interpolation-based mechanism: pre-trained embeddings are resized to match the input resolution, providing a global prior that improves extrapolation"

This means an image of arbitrary aspect ratio and resolution is converted into a token sequence whose length is proportional to its pixel count, not padded or cropped to a fixed size. The largest supported resolution, given the LLM's maximum context length of 16,384 tokens, is 1792×1792 pixels. This is critical for fine-grained visual understanding tasks — OCR at small font sizes, reading text in crowded diagrams, or identifying small objects in high-resolution photographs — where fixed-resolution encoders would lose detail through downsampling. The trade-off is computational: higher-resolution images produce longer token sequences, increasing LLM inference cost.

---

#### Pre-Training Data Construction

**SAIL-Caption2: Automatic quality assessment and filtering.** The core data innovation in SAIL-VL2's pre-training is a systematic approach to filtering noisy captions from the SAIL-Caption dataset (which contains approximately 300M image–caption pairs, many generated by automated captioning models). The paper identifies that 15–20% of captions fall below acceptable quality (score <3 on a 1–5 scale), where "acceptable" means the caption accurately describes the image content without hallucinations, omissions, or generic/vague language. Manually reviewing 300M captions is infeasible, and using expensive commercial LVM APIs to score every caption would be cost-prohibitive. The solution is to train specialized judge models.

The quality assessment framework evaluates captions along two independent dimensions:

**Visual Information Richness (VIR)** measures how much visual content the caption captures across three sub-dimensions: (a) Instances & Entities — the number, diversity, and distinctiveness of objects/entities mentioned; (b) Visual Complexity — color variety, composition, texture descriptions; (c) Fine-Grained Details — specific attributes such as textures, small objects, or contextual relationships between entities.

**Image-Text Alignment (ITA)** measures how well the caption matches the actual image content across three sub-dimensions: (a) Specificity — the precision of details (does the caption specify "a red wooden chair" vs. "furniture"?); (b) Completeness — coverage of all key visual elements; (c) Accuracy — absence of hallucinations (describing things not in the image) or redundancy.

Each dimension is rated on a 1–5 integer scale. The training procedure for the judge models works as follows:

1. Sample approximately 3M captions from the original SAIL-Caption dataset.
2. Send these image–caption pairs to a powerful commercial LVM API (the paper does not name which, but it is a large closed-source model) with scoring prompts that operationalize the ITA and VIR criteria. The API returns 1–5 ratings for each dimension.
3. Train small judge models — fine-tuned versions of SAIL-VL-1.5-2B and SAIL-VL-1.5-8B — to predict these API-generated scores. Two variants are trained for each dimension: a Score Judge that predicts the 1–5 numerical rating (a regression or ordinal classification task) and a Yes-or-No Judge that makes a binary quality decision (acceptable vs. unacceptable, with the threshold at score <3).
4. To ensure robust training, the 500K captions used for fine-tuning are uniformly resampled across the full score distribution, preventing the model from being biased toward the most common score values.

The results in Table 2 show that fine-tuned judge models achieve high accuracy. For example, SAIL-VL-1.5-2B-ITA-Score achieves 0.916 precision and 0.903 recall; SAIL-VL-1.5-2B-VIR-Score achieves 0.983 precision and 0.975 recall. The binary judge variants show similar reliability. Critically, the zero-shot performance of the base SAIL-VL-1.5-2B model (without fine-tuning for judging) is significantly worse: ITA precision of 0.436 and recall of 0.994 (it classifies almost everything as acceptable), while VIR shows the opposite pattern — high precision (0.823) but low recall (0.560), meaning it misses many poor-quality captions. This demonstrates that specialized fine-tuning is necessary; general multimodal understanding capability does not automatically transfer to quality assessment. The paper also shows that InternVL3-2B (an independent baseline) performs even worse in zero-shot mode (ITA: 0.269 precision, 0.070 recall), confirming that the judge capability is not universal across model families.

After training, the judge models are applied to the full 300M SAIL-Caption corpus. Captions receiving a score <3 from the Score Judge or classified as "No" by the Yes-or-No Judge are removed. Given that the judges achieve >90% accuracy, the paper estimates that retained captions are >99% high quality. The final corpus, SAIL-Caption2, contains approximately 250M image–caption pairs after filtering.

**Chart caption data.** Beyond general image captions, the paper constructs a specialized chart caption corpus because standard captioning datasets underrepresent charts, tables, and data visualizations — image types that are critical for scientific reasoning and document understanding benchmarks. The pipeline has two complementary sources:

**Synthetic chart generation.** The paper designs an automated pipeline where a large language model (LLM) receives a text prompt specifying chart parameters and generates chart rendering code (either SVG markup or Python code using libraries like matplotlib). The code is executed to produce a chart image, and the LLM simultaneously generates a caption describing the chart and QA pairs testing comprehension of the data. This pipeline supports a wide range of chart types (bar, line, pie, scatter, area, histogram, heatmap, box plot) and allows flexible configuration of parameters including language (English/Chinese), chart type distribution, and generation volume. The key advantage is scalability: once the pipeline is built, it can generate unlimited chart data without human annotation, and the captions are guaranteed to be accurate (since they're derived from the same code that generated the chart).

**Open-source dataset collection.** To complement synthetic data with real-world chart diversity, the paper collects existing chart caption datasets in both English and Chinese, including PixMo-cap (from the Molmo dataset) and DVQA (Data Visualizations QA). These provide natural chart images (e.g., screenshots from news articles, scientific papers, business reports) that may have visual artifacts, unusual styling, or domain-specific conventions not captured by synthetic generation.

Finally, an LLM-based annotation workflow is used: state-of-the-art multimodal models (commercial APIs) annotate both the auto-rendered and open-source charts under carefully designed prompts that elicit detailed, accurate captions and QA pairs. Manual sampling is performed for quality control. The resulting chart caption corpus contains 400K automatically generated captions and 1.29M captions from conventional datasets, with equal English/Chinese distribution. Combined with the filtered 250M general captions, SAIL-Caption2 totals 251.69M image–caption pairs.

**Synthetic VQA data generation.** Beyond caption data, the paper generates question–answer pairs from captions to support pre-training. The procedure — called "Caption2QA" — works as follows:

1. Sample approximately 80M entries from SAIL-Caption2.
2. For each caption, use a powerful commercial LLM API to generate multiple diverse questions that can be answered based on the caption text. The paper states: "the model generates several diverse questions per caption to ensure broad coverage of the visual content."
3. For each generated question, the LLM produces a corresponding answer grounded in the caption text.

The resulting synthetic VQA dataset serves as additional multimodal training data. The paper acknowledges a limitation: "synthetic data may introduce distributional biases in linguistic expression, as LLMs often produce homogenized phrasing with limited variability." This means the generated questions may use formulaic sentence structures and vocabulary, potentially causing the model to learn narrow linguistic patterns rather than robust question-answering ability. Despite this concern, the paper reports empirical benefits: "performance improves smoothly with increasing training budget, exhibiting a logarithmic scaling trend with up to 180M training samples" — meaning that adding more synthetic VQA data continues to yield (diminishing) returns, consistent with standard neural scaling behavior.

**SAIL-Video: Video QA Data Curation and Filtering.** The paper constructs a video understanding corpus from existing datasets: ShareGPTVideo-QA, NextQA, LongVideoBench-val, PerceptionTest-val, LLaVA-Video-QA, and VideoGPT. The initial collection contains 6.23M video-QA samples. After systematic filtering (described below), this is reduced to 5.1M high-quality samples.

The filtering criteria evaluate each video-QA pair along three dimensions, each scored by a powerful LVM API:

**Video–QA Alignment (score range: -1 to 10).** This addresses a subtle but critical problem: most video QA datasets provide annotations based on *continuous video streams* (where a human annotator watches the full video and writes questions/answers), but most video LVMs process *sparsely sampled frames* (e.g., 16 frames uniformly sampled from a 2-minute video). If the sampled frames don't contain the visual information needed to answer the question — for instance, the question asks about an event that occurs between sampled frames — the model receives an impossible task. The alignment score quantifies the degree to which the question is answerable given only the sampled frames, with higher scores indicating better alignment.

**Video Content Richness (score range: -1 to 7).** This evaluates the visual complexity and informativeness of the video content itself, capturing: element diversity (how many distinct objects, people, scenes appear), scene complexity (visual clutter, motion patterns, lighting variations), and information density (how much is happening per unit time). Videos with low content richness — static scenes, minimal motion, repetitive patterns — provide little learning signal for temporal reasoning.

**QA Difficulty (score range: -1 to 8).** This measures the cognitive complexity of the associated question–answer task, including: reasoning depth (does it require multi-step inference or simple recognition?), spatial–temporal understanding requirements (does it ask about object trajectories, temporal ordering, or causal relationships?), multi-instance interactions (does it involve relationships between multiple entities?), and reliance on external knowledge (does it require world knowledge beyond what's visible in the video?).

The score of -1 is a special value indicating that the LVM API refused to respond (likely due to content policy violations or malformed inputs). The paper applies fixed thresholds: videos must achieve an alignment score ≥5, a content richness score ≥5, and a difficulty score ≥3 to be retained. These thresholds were presumably determined empirically to balance data quality against data quantity, though the paper does not provide an ablation study of threshold values.

**Data resampling strategy.** The paper implements a two-step resampling approach to address distributional biases in the pre-training data, motivated by the observation that large-scale caption and VQA datasets often exhibit imbalanced patterns — certain visual concepts, linguistic structures, or answer formats appear much more frequently than others, which can cause the model to learn spurious shortcuts rather than robust understanding.

**Step 1: Dataset-level resampling during basic pre-training.** Different datasets naturally exhibit different distributional biases (e.g., web-crawled captions emphasize certain object categories; OCR datasets emphasize text-heavy images). During basic multimodal pre-training, the paper adjusts the sampling ratios across datasets to balance the distribution of image–text pairs. The goal is to ensure that the adapter is exposed to "heterogeneous visual signals" — diverse visual content spanning different domains, styles, and task types — rather than overfitting to the characteristics of any single data source.

**Step 2: Linguistic-level resampling during multi-task pre-training.** The SAIL-Caption2 and Synthetic VQA datasets are large in scale but often show imbalanced language patterns because many samples are generated or post-processed by LLMs, which tend to produce homogenized phrasing. To address this, the paper rebalances at the n-gram level, adjusting sampling weights to improve lexical and structural diversity. The paper does not provide full technical details of the n-gram rebalancing algorithm (e.g., whether it uses TF-IDF weighting, frequency-based undersampling, or explicit n-gram coverage objectives), but the stated outcome is that it "boosts data efficiency, allowing SAIL-VL2 to achieve stronger multimodal understanding and more robust instruction-following capabilities."

---

#### Pre-Training Recipe

**Overview of the two-stage pre-training pipeline.** The pre-training of SAIL-VL2 is organized into two sequential sub-stages, each with distinct objectives, data compositions, and trainable parameter sets, as summarized in Table 3.

**Stage 1: Basic multimodal pre-training.** This stage develops fundamental cross-modal alignment — the ability to map visual content to linguistic descriptions and recognize text in images. The model starts with: a pre-trained SAIL-ViT (vision encoder, frozen or partially trainable depending on the SAIL-ViT training stage already completed), a pre-trained Qwen3-Instruct LLM (language model, frozen), and a randomly initialized two-layer MLP adapter. Training uses 64M samples comprising general captions and chart captions from SAIL-Caption2, plus high-quality OCR samples from IDL-WDS. Importantly, only the adapter is trained in this stage (the paper states "trainable params: Connector" in Table 3, though the SAIL-ViT training stages described in Section 2.1.1 also occur during what is called "SAIL-ViT Training" in Table 3 — these are distinct stages). The initial learning rate is $2 \times 10^{-4}$ with AdaLRS applied, and the batch size is 2048.

The sequence length is set to 8,192 tokens throughout both pre-training stages. This is a practical constraint: longer sequences increase memory usage quadratically (due to the self-attention mechanism) and would reduce the maximum batch size that fits in GPU memory. Since pre-training images at 448×448 resolution produce roughly 1,024 visual tokens, and captions/answers typically require a few hundred text tokens, 8,192 provides ample capacity while maintaining training throughput.

**Stage 2: Multi-task pre-training.** This stage comprehensively strengthens visual understanding and instruction-following capabilities. All model parameters are now unfrozen for joint optimization: the vision encoder, adapter, and LLM are all updated. The training data expands to 180M samples with equal-sized visual understanding and instruction-tuning data, encompassing: general captions and OCR data (continuing from Stage 1), open-source VQA data, synthetic VQA data (from the Caption2QA procedure), and text-based math reasoning samples. The inclusion of instruction-tuning data serves dual purposes: it enhances the model's ability to follow diverse instructions, and it preserves the LLM's language capabilities by ensuring the model continues to process text-only inputs, preventing catastrophic forgetting during large-scale vision-language alignment training.

The paper reports training on 128B tokens in basic pre-training and 360B tokens in multi-task pre-training (Table 3), though these figures likely represent the total corpus size processed during one-epoch training rather than a multi-epoch regime.

**AdaLRS: Adaptive Learning Rate Search.** During basic multimodal pre-training, the paper introduces AdaLRS (Dong et al., 2025b) — an algorithm that dynamically adjusts the learning rate during training rather than following a fixed schedule. The motivation comes from an empirical observation:

> "the training loss of LLM/LVM pre-training, as well as the loss descent velocity with respect to the learning rate, exhibits a convex pattern with a shared optimum"

This means: if you were to plot training loss against learning rate (holding all else constant), you would see a U-shaped curve with a clear minimum. The learning rate that minimizes loss — the "optimum" — tends to be the same across different training phases and model scales. However, this optimum is not known in advance and may shift as training progresses. AdaLRS attempts to find and track this optimum online.

The algorithm operates through a backtracking line search procedure:

1. Monitor the loss descent slope (the rate at which training loss decreases) over a sliding window of $k$ steps.
2. If the loss descent slows down (the slope becomes less negative), tentatively increase the learning rate.
3. Evaluate the effect of this increase: if the loss slope improves (becomes more negative, meaning faster descent), retain the higher learning rate and continue training.
4. If the increase further slows convergence, roll back the model parameters and optimizer state to the pre-increase state, then instead decrease the learning rate.
5. If neither increase nor decrease helps, keep the current learning rate.

Formally, the update rule is given in Equation 1:

$$\eta_{t+k} = \begin{cases} \alpha'\eta_t & \text{if } v(\alpha'\eta_t) > v(\eta_t) + 2e \quad \text{(loss slope increases ↑)}, \\ \beta'\eta_t & \text{if } v(\alpha'\eta_t) < v(\eta_t) - 2e \quad \text{(loss slope decreases ↓)}, \\ \eta_t & \text{otherwise.} \end{cases}$$

where $v(\cdot)$ is the estimated loss curve slope obtained from a $k$-step window, $\eta_t$ is the learning rate at step $t$, $\eta_{t+k}$ is the updated learning rate applied for the next $k$ steps, $e$ is the estimation error between the measured slope $v$ and the true loss descent velocity $V$, $\alpha' = \max(\lambda^t \alpha, 1)$ and $\beta' = 1 / \max(\lambda^t \beta, 1)$ are rectified learning rate scaling factors, $\alpha$ and $\beta$ are co-prime integers (mutually prime, having no common factors beyond 1) both greater than 1, and $\lambda \in (0, 1)$ is a decay factor that gradually reduces the magnitude of adjustments over time (preventing large oscillations late in training).

**What this equation computes in operational terms.** At each checkpoint (every $k$ steps), the algorithm estimates how fast the loss is currently decreasing. It then tries a candidate learning rate (either $\alpha'$ times the current rate for an increase, or $\beta'$ times for a decrease). If the candidate produces a substantially better loss slope (exceeding the current slope by more than $2e$, where the factor of 2 accounts for estimation error in both measurements), the candidate is adopted. If it produces a substantially worse slope, the opposite direction is tried. If neither candidate produces a meaningful change, the learning rate stays the same.

**Why this form.** The two key design choices are: (1) using slope (rate of change) rather than absolute loss values — this makes the algorithm robust to the overall scale of the loss, which varies across models and data; and (2) the gradual decay of adjustment magnitude through $\lambda^t$ — this ensures the algorithm makes large exploratory adjustments early in training when the learning rate is far from optimal, but settles into small refinements later when near the optimum, preventing oscillatory behavior.

The paper reports: "AdaLRS adjusts the initial learning rate upwards to $6.75 \times 10^{-4}$ effectively, surpassing the fixed learning rate baseline by a final loss advantage of over 0.06." This means that starting from an initial rate of $2 \times 10^{-4}$, AdaLRS discovered that a higher rate (more than 3× the initial value) was actually better for this training stage, and the final training loss was 0.06 lower than what would have been achieved with a fixed $2 \times 10^{-4}$ schedule.

A notable detail: AdaLRS is not applied during multi-task pre-training because "the training loss on instruction tuning data exhibits poor correlation with model performance." This is a well-known phenomenon in LLM training: loss on diverse instruction-following tasks is dominated by easily predictable tokens and doesn't reliably track downstream benchmark accuracy, making loss-based learning rate optimization unreliable.

**Scaling law analysis.** The paper investigates how multi-task pre-training performance scales with increasing data volume. Using SAIL-VL2-2B as the testbed, the training budget is expanded to 360B tokens (the full multi-task pre-training corpus). Figure 4 plots benchmark accuracy against training data size on a log scale for three evaluation suites: overall benchmarks, natural-scene VQA datasets, and OCR VQA tasks. The paper reports:

> "performance exhibits consistent and monotonic improvements, yielding a smooth empirical scaling curve"

The curves show logarithmic behavior — each doubling of data produces diminishing returns, consistent with neural scaling laws observed across domains. The paper notes that approximately 50% of the training data is synthesized by annotator models (SAIL-Captioner and Qwen3), which "may introduce linguistic biases" but the "increased scale and diversity of training data substantially enhance generalization and reasoning." This is a pragmatic trade-off: synthetic data is cheaper than human annotation and enables larger training corpora, but the biases it introduces may ultimately limit the ceiling of what the model can learn.

---

#### Post-Training Data Construction

**SAIL-Instruction2: Instruction-tuning data with latent-class balancing.** Building on SAIL-Instruction (from the prior SAIL-VL model), the paper constructs a larger, higher-quality instruction-tuning corpus. The starting point includes: LLaVA-OneVision (a large-scale visual instruction dataset), Cauldron (a collection of 50 vision-language datasets unified into a conversation format), and additional high-quality open-source datasets: Mammoth-VL (multimodal reasoning instruction data), MMPR (mixed preference optimization data for reasoning enhancement), and Molmo (pixmo-based instruction data with detailed visual descriptions).

To specifically foster complex reasoning, the paper samples long-answer and reasoning-oriented instances from LLaVA-CoT (chain-of-thought visual reasoning data), MMPR, and Condor datasets. The selection criterion is not explicitly detailed, but the implication is that instances requiring multi-step inference, explanation generation, or structured output formats are preferentially included over simple factoid QA pairs.

All incremental data undergoes a two-stage validation process inherited from SAIL-VL:
1. A quality evaluation to ensure intrinsic reliability — checking for answer correctness, instruction clarity, and absence of hallucinated content.
2. An incremental evaluation to verify that adding this data to the existing training mixture actually improves benchmark performance, rather than just increasing training cost.

Only data passing both stages is retained.

**Latent-class-based filtering and rebalancing.** This is the key innovation in SAIL-Instruction2's construction. The problem being solved is that instruction-tuning datasets, even when diverse in source, often exhibit latent distributional biases: certain types of questions, visual concepts, or answer formats are overrepresented, causing the model to develop brittle, format-specific behaviors rather than robust instruction-following. The solution works as follows:

1. For each VQA sample, a large vision-language model (likely a commercial API) generates a descriptive phrase that characterizes the sample's content and task type. The paper does not provide examples, but this might be something like "counting objects in a cluttered scene" or "reading text from a document and answering a factual question."

2. From this descriptive phrase, logits (the raw output scores from the model's final layer) and latent embeddings (the internal representations from an intermediate layer) are extracted. These serve as a semantic fingerprint of the sample.

3. Each sample is assigned a "latent class" based on clustering of these embeddings. This maps heterogeneous samples from different source datasets into a unified semantic space, bridging the representation gap that exists when samples from different datasets use different formatting conventions, answer styles, or instruction structures even when they test the same underlying capability.

4. The original coarse-level dataset classification (e.g., "this sample came from LLaVA-OneVision, that sample came from MMPR") is replaced with fine-grained semantic buckets derived from latent class clustering. The paper states this "expands the category space by nearly tenfold" — meaning that instead of, say, 20 source-dataset categories, the latent clustering produces ~200 semantic categories.

5. Uniform sampling is applied across these fine-grained buckets, ensuring balanced representation of different task types, visual domains, and reasoning patterns.

6. Samples are re-annotated using APIs from state-of-the-art closed-source models (the paper does not specify which, but this would be models like GPT-4V or Gemini). The re-annotation step serves a dual purpose: it improves answer quality (replacing potentially noisy original annotations) and it ensures formatting consistency (all answers now follow the same style, regardless of original source).

The final SAIL-Instruction2 corpus contains 20M high-quality, diversity-balanced samples. Figure 5 in the paper shows an ablation: SAIL-Instruction2 consistently outperforms prior instruction datasets (including SAIL-Instruction and other open-source collections) at equivalent data budgets, demonstrating that the filtering and rebalancing pipeline produces genuinely higher-quality supervision.

**Multimodal CoT data construction and cleaning.** To support the reasoning-oriented training stages (LongCoT SFT and the RL stages), the paper constructs a corpus of multimodal chain-of-thought data from diverse public sources. The raw data comes from VisualWebInstruct (web-sourced multimodal instruction data), MathV360K (mathematical reasoning with visual contexts), and LLaVA-CoT (chain-of-thought visual reasoning). These heterogeneous sources inevitably contain noise and formatting inconsistencies that must be cleaned before use.

The cleaning protocol consists of:

1. **Removing extraneous content:** System prompts, conflicting hints, and metadata that are not part of the reasoning process are stripped. This ensures the model learns to generate clean reasoning chains rather than reproducing system-level instructions in its outputs.

2. **Unifying output format:** All thinking processes are wrapped in `〈thinking〉〈/thinking〉` tags, all final answers in `\boxed{}` tags. This creates a consistent structure that the model can learn to produce and that the verifier can parse.

3. **Deduplication:** Samples with identical image–question pairs are removed to prevent the model from memorizing specific instances.

For CoT data generation (creating new reasoning chains where raw data only provides questions and answers without intermediate steps), the paper uses a "guided prompting strategy": given a question and its ground-truth answer, a powerful LVM is prompted to generate a detailed reasoning process that logically connects the question to the answer. This ensures the generated reasoning is coherent (it actually leads to the correct answer) rather than being a plausible-sounding but logically flawed chain.

Three additional quality filters are applied:

**Redundancy filtering:** The token overlap between the generated chain-of-thought and the final answer is measured. If the CoT and answer are highly similar (e.g., the CoT essentially just restates the answer in slightly different words), the sample is discarded. This penalizes "trivial reasoning" where the model is not actually performing intermediate inference steps.

**Answer distillation:** Overly verbose ground-truth answers are refined and shortened using a judge model. This mitigates the tendency for models that are trained on long-winded answers to produce unnecessarily verbose output even for simple questions.

**CoT length balancing:** The token length distribution of generated CoT chains is analyzed. To prevent the model from learning a bias toward a particular reasoning length, the dataset is resampled to ensure balanced representation of short, medium, and long reasoning chains.

After cleaning and filtering, the final dataset comprises 400K samples for LongCoT SFT, 1M samples for Think-Fusion SFT, 50K samples for RL with verifiable rewards, and 100K samples for RL with a mixed reward system.

---

#### The Thinking-Fusion SFT–RL Pipeline

This is the most technically novel contribution of the paper. The pipeline consists of four sequential stages that progressively teach the model to reason, verify its reasoning, and integrate reasoning with general capabilities. The goal is to produce a model that can selectively engage in step-by-step thinking when faced with complex problems while answering simple questions directly — avoiding both the failure to reason on hard problems and the inefficiency of verbose reasoning on easy ones.

**Stage 1: LongCoT Supervised Fine-Tuning.** The first stage teaches the model to produce structured reasoning chains through standard supervised learning. The model is fine-tuned on the 400K LongCoT samples, where each training example consists of an image, a question, and a target output containing both a thinking process (wrapped in `〈thinking〉〈/thinking〉`) and a final answer (wrapped in `\boxed{}`). The training objective is next-token prediction over the entire output sequence:

$$\mathcal{L}_{\text{LongCoT SFT}} = -\frac{1}{|\mathcal{D}_{\text{CoT}}|} \sum_{(I,T,A) \in \mathcal{D}_{\text{CoT}}} \log P_\theta(T \circ A \mid I)$$

where $\mathcal{D}_{\text{CoT}}$ is the LongCoT dataset (400K samples), $I$ is the input instruction (image + question), $T$ is the thinking process (the reasoning chain), $A$ is the final answer, $\theta$ represents the model parameters, and $\circ$ denotes string concatenation (the thinking process and answer are treated as a single continuous output during training).

**What it computes:** the average negative log-likelihood of the concatenated thinking+answer sequence given the input. The model predicts each token in $T$ and $A$ one at a time, and the loss penalizes deviations from the ground-truth tokens. Note that the loss is computed over *all* tokens in the output — both thinking and answer — which means the model learns to generate the reasoning chain itself, not just the final answer.

**Why this form:** training only on the answer (ignoring the thinking process during loss computation) would teach the model to produce the right answer without learning *how* to reason — it would learn a direct mapping from question to answer, bypassing the intermediate reasoning steps that are crucial for complex problem-solving. Training on the full sequence forces the model to internalize the reasoning process, learning that the path to the answer matters. The standard negative log-likelihood objective is used because this is a supervised learning stage; there is no reward signal or exploration involved yet.

Training hyperparameters: All model parameters are trained for one epoch using the AdamW optimizer with a global batch size of 1024, a cosine learning rate schedule with a peak learning rate of $1 \times 10^{-6}$, and the standard next-token prediction loss. The context length is set to 20,480 tokens, which is significantly longer than the 8,192 used in pre-training — this is necessary because CoT reasoning chains can be very long, sometimes running to thousands of tokens for complex mathematical proofs or multi-step logical deductions.

**Stage 2: Reinforcement Learning with Verifiable Rewards.** After LongCoT SFT, the model can produce reasoning chains but may still make logical errors, arithmetic mistakes, or fail to arrive at correct answers. This stage uses reinforcement learning to optimize the model's reasoning policy directly for correctness.

**Data curation for RL.** The training data is constructed from diverse public sources covering Math, Puzzle, Science, OCR, and Counting tasks. The raw data undergoes two-stage filtering:

1. **Reward hacking prevention:** For multiple-choice questions that have been converted to free-response format (e.g., "What is the value of X? A) 5 B) 7 C) 12"), an LLM converts them to genuine free-response questions ("What is the value of X?"). This prevents the model from learning to exploit multiple-choice formatting patterns rather than actually solving the problem.

2. **Difficulty-based filtering:** Using the model from the previous stage (LongCoT SFT), each problem is evaluated on its pass@4 score — the fraction of 4 sampled solutions that are correct. For tasks that don't require explicit reasoning, trivial problems where the model achieves perfect pass@4 (=1) are discarded — these are too easy to provide useful learning signal. For reasoning-heavy tasks, a more stringent filter is applied: both the easiest problems (pass@4=1, the model already solves them perfectly) and the hardest problems (pass@4=0, the model never solves them) are removed. This retains problems that are "challenging yet solvable" — the model sometimes gets them right and sometimes wrong, providing a gradient for improvement.

The final dataset for this stage contains 70K STEM samples.

**Reward system.** The reward has two components:

**Answer Reward:** A rule-based binary reward (0 or 1) that checks whether the final answer in the `\boxed{}` tag matches the ground-truth answer. For mathematical problems, this is exact string matching after normalization; for other verifiable problems, a domain-specific checker is used.

**Format Reward:** A binary reward (0 or 1) that verifies the output adheres to the required structure — specifically, that the reasoning process is enclosed in `〈thinking〉〈/thinking〉` tags and the final answer in `\boxed{}` tags.

Both signals are discrete and binary, providing unambiguous feedback. There is no partial credit — an answer is either correct or not. The total reward is a combination of these two signals (the paper does not specify the exact weighting, but typically format reward receives a small weight and answer reward dominates).

**Training recipe.** The LongCoT SFT model is optimized using Proximal Policy Optimization (PPO), a standard RL algorithm for language model training. However, the paper uses specialized PPO variants for different model architectures:

- For the dense model (SAIL-VL2-2B and SAIL-VL2-8B), **DAPO** (Yu et al., 2025) is used — a memory-efficient optimizer that enhances training stability by decoupling the policy and value function optimization and using separate Adam optimizers for each.
- For the MoE model (SAIL-VL2-30B-A3B), **GSPO** (Zheng et al., 2025) is used — Group Sequence Policy Optimization, which provides stable and targeted updates to individual experts by grouping them and applying group-level constraints on policy updates, preventing the expert routing from collapsing during RL training.

The training configuration: context length of 16,384 tokens, maximum generation length of 4,096 tokens per rollout, 2,048 rollouts generated from the policy in each training episode, 8 gradient updates per episode using a mini-batch size of 512, policy network learning rate of $1 \times 10^{-6}$. To encourage exploration (trying different reasoning paths rather than always generating the same one), the PPO clipping value is dynamically adjusted within the range [0.20, 0.28] — this controls how much the policy can change in a single update, with higher values allowing more aggressive exploration.

**Why PPO rather than simpler alternatives.** The paper could have used rejection sampling (generate many solutions, keep the correct ones, fine-tune on those) or direct preference optimization (DPO). PPO was chosen because: (1) it provides gradient-based optimization rather than just filtering, which can discover reasoning strategies that the model never spontaneously generated; (2) it handles the exploration-exploitation trade-off systematically through the clipping mechanism; (3) DAPO and GSPO are specifically designed for large-scale language model training and address stability issues that vanilla PPO exhibits when the policy and value function share parameters.

**Stage 3: Think-Fusion Supervised Fine-Tuning.** The model after RL with verifiable rewards is good at reasoning but may have developed "reasoning mode collapse" — it tends to produce verbose reasoning chains for everything, even simple questions where direct answers would be more appropriate. Additionally, the RL stage focused exclusively on STEM/reasoning tasks, so the model may have lost some of its general instruction-following capabilities. Think-Fusion SFT addresses both issues by training on a strategically composed mixture.

**Data composition.** The dataset consists of two components:

- **90% direct-answer instruction pairs** (approximately 900K samples): General-purpose VQA and instruction-following data that covers a wide range of tasks (image description, factoid QA, simple counting, attribute recognition). These samples have standard input-output format with no reasoning chain — just question and answer.

- **10% high-quality CoT exemplars** (approximately 100K samples): Reasoning chains harvested from the preceding RL stage through rejection sampling. Specifically, the RL-trained model generates multiple solutions (presumably 4–8) for each reasoning problem, the correct solutions are kept, and the best reasoning chains (those that are both correct and logically well-structured) are selected. This means the CoT exemplars are *on-policy* — they represent the model's own best reasoning, not externally generated chains.

The total dataset contains 1M training instances.

**Training recipe.** All model parameters are fine-tuned for one epoch using AdamW, global batch size 1024, cosine learning rate with peak at $1 \times 10^{-6}$. The key innovation is the **dual-target loss function**:

$$\mathcal{L}_{\text{Think-Fusion SFT}} = -\frac{1}{|\mathcal{D}|} \left( \sum_{(I,T,A) \in \mathcal{D}_{\text{CoT}}} \log P_\theta(T \circ A \mid I) + \sum_{(I,A) \in \mathcal{D}_{\text{direct}}} \log P_\theta(A \mid I) \right)$$

where $\mathcal{D}$ is the entire mixed dataset, $\mathcal{D}_{\text{CoT}}$ is the subset with CoT exemplars (100K samples), $\mathcal{D}_{\text{direct}}$ is the subset with direct answers (900K samples), $I$ is the input, $T$ is the thinking process, $A$ is the answer, and $\theta$ represents model parameters.

**What it computes:** a combined loss over two types of training examples. For CoT examples, the loss is computed over the full thinking+answer sequence (teaching the model to produce reasoning when it's appropriate). For direct examples, the loss is computed only over the answer tokens (teaching the model that for simple questions, it should produce just the answer, skipping the reasoning).

**Why this form:** the differential loss computation is the critical design choice. If the loss on direct examples also included the "thinking" portion (which doesn't exist — these examples have no reasoning chain), the model would receive no gradient for producing a direct answer. By computing loss only on the answer tokens for direct examples, the model learns that the optimal policy is to produce just the answer when the question is simple. The 90/10 split ensures that the model retains its general capabilities (the 90% direct data) while maintaining reasoning skills (the 10% CoT data). A 50/50 split would skew the model toward always reasoning; a 99/1 split might cause it to forget how to reason entirely.

**The emergent behavior.** The paper reports a striking observation:

> "the model effectively internalizes the reasoning capabilities it was taught. In other words, it learns the logical pathways to the correct answer without needing to explicitly write out each step. This allows it to provide accurate direct answers even when a user does not specifically prompt it for a step-by-step thinking process"

This means the model learns to perform reasoning "in its head" — the internal representations encode the logical inference even when the output is just the final answer. This is a form of knowledge distillation from the explicit reasoning chains to implicit reasoning capability, and it explains why the model can maintain high accuracy on reasoning benchmarks even when producing direct answers.

**Stage 4: Reinforcement Learning with a Mixed Reward System.** The final stage further refines reasoning by introducing a more sophisticated reward that evaluates not just the final answer but the quality of the reasoning process itself.

**Data curation.** The dataset from the verifiable-reward RL stage (Stage 2) is refined through rejection sampling to identify "hard cases" — problems where the model's performance from the previous stage was suboptimal (it sometimes got them wrong despite having the capability to solve them). This creates a curated dataset of 50K challenging STEM samples. To prevent the model from overfitting to these hard cases at the expense of general capabilities, 50K general-purpose samples from LLaVA-OneVision are added, totaling 100K training samples.

**Mixed reward system.** Three distinct reward signals are combined:

**Answer Reward:** As before, a binary correctness check on the final answer. For verifiable problems (math, puzzles), this uses rule-based checking. For nuanced tasks where ground truth is not deterministically verifiable (e.g., "describe the mood of this painting"), an LVM-based judge provides a reward signal.

**Thinking Reward:** This is the key addition. An LVM-based judge (a powerful multimodal model, likely a commercial API) evaluates the entire reasoning chain on three criteria: (a) logical soundness — does each step follow from previous steps? (b) factual grounding — are claims about the image content actually supported by what's visible? (c) answer consistency — does the final answer actually follow from the reasoning chain, or is there a disconnect?

**Format Reward:** As before, ensuring proper tag usage.

All reward signals are discrete binary values (0 or 1). The final reward is a weighted sum:

$$R_{\text{total}} = w_a R_{\text{answer}} + w_t R_{\text{thinking}} + w_f R_{\text{format}}$$

where $w_a + w_t + w_f = 1$. The paper does not report the specific weights, but the constraint that they sum to 1 means the total reward is always in [0, 1].

**Why add a thinking reward.** The answer reward alone can produce models that arrive at correct answers through flawed reasoning — a lucky guess or a reasoning chain with an error that cancels out. The thinking reward pushes the model toward producing logically coherent chains, which should improve generalization to novel problems where lucky guesses won't work. The format reward ensures the thinking process remains parseable, which is important for transparency and debugging.

**Training recipe.** The training configuration is identical to the verifiable-reward RL stage: DAPO/GSPO with the same hyperparameters, context length, generation length, and batch sizes. This ensures that any performance changes can be attributed to the new reward signal rather than training dynamics.

**The SFT–RL hybrid philosophy.** The full pipeline embodies a deliberate alternation between supervised learning and reinforcement learning. The SFT stages (LongCoT, Think-Fusion) provide stable, grounded training that teaches the model *what good reasoning looks like* through expert demonstrations. The RL stages (verifiable-reward, mixed-reward) provide targeted optimization for *correctness and coherence*, allowing the model to discover reasoning strategies that weren't present in the demonstration data. This hybrid approach addresses a fundamental limitation of pure SFT: the model can only learn to imitate, not to innovate. Pure RL, conversely, can be unstable and may converge to degenerate solutions without a good initialization. The alternation provides the benefits of both while mitigating their respective weaknesses.

---

#### Model Soup: Homologous Model Merging

After the basic SFT stage (Section 4.2.1), the paper applies a simple but effective technique called **model soup** — averaging the weights of multiple trained models to produce a single model that outperforms any individual one. The key insight is that not all model merging is beneficial; the paper distinguishes two cases:

**Homologous models:** Models that follow similar optimization trajectories — comparable hyperparameter settings, similar data compositions, trained from the same initialization. These models converge to different local minima of the loss landscape that are connected by a low-loss basin, so averaging their weights produces a model that is also in the low-loss region.

**Heterologous models:** Models that differ substantially — different data, different hyperparameters, different training procedures. These models converge to different basins that may be separated by high-loss barriers, so averaging their weights can land in a high-loss region, causing catastrophic performance degradation.

Table 5 provides empirical evidence:
- Merging homologous models consistently improves performance. For example, merging two homologous base models improves the average score from 74.91 and 74.54 (individually) to 76.60 (merged), with gains across all benchmarks.
- Merging heterologous models causes catastrophic failure. The merge of two heterologous models produces near-zero accuracy on some benchmarks (0.12 on OCRBench, 5.25 on DocVQA), with performance worse than either individual model on every benchmark.

The paper therefore restricts model soup to homologous models only, averaging weights with equal weighting.

---

#### Mixture-of-Experts Architecture

**Motivation for MoE.** Mixture-of-Experts provides a way to scale model capacity (total parameter count) without proportionally scaling computation per token. In a standard dense transformer, every parameter is used for every input token. In an MoE transformer, the feed-forward network (FFN) layers — which typically account for the majority of parameters — are replaced with multiple parallel "expert" FFNs. A learned gating function (a small neural network) selects a subset of experts (typically 2 out of 8 or more) to activate for each token, meaning only a fraction of the total parameters are used per token.

SAIL-VL2 adopts the Qwen3-MoE architecture, specifically Qwen3-30B-A3B: 30.5B total parameters but only 3B activated per token (hence "A3B"). The activation sparsity means this model has roughly the inference cost of a 3B dense model while having the representational capacity of a 30B model.

**Load balancing.** A critical challenge in MoE training is ensuring that all experts are used roughly equally. If the gating function consistently routes tokens to a few "popular" experts while ignoring others, the popular experts receive most of the gradient updates and dominate the model, while the ignored experts receive negligible training signal and effectively waste capacity. The paper employs two strategies:

1. **Auxiliary load-balancing loss:** An additional loss term is added to the main training objective that penalizes uneven expert utilization. This encourages the gating function to distribute tokens more uniformly across experts.

2. **Average activation across ranks:** In distributed training (where different experts may be on different GPUs), the expert activation statistics are averaged across all devices to ensure consistent load balancing.

**Distribution-aware tuning for expert entropy.** The paper observes that "activation patterns vary with data distribution" — the gating function's behavior depends on the input data. Multimodal data (images + text) has different statistical properties than pure text data, potentially causing the gating function to route tokens differently, which could disrupt the expert specialization learned during the LLM's text-only pre-training. To address this:

> "we conduct data probing and automatic calibration on language data, which maximizes expert activation entropy"

This means: the model's routing behavior on pure text data (where the LLM was originally trained) is measured, and calibration adjustments are applied to ensure that when multimodal data is introduced, the activation entropy (how uniformly tokens are distributed across experts) remains high and the routing pattern on text data stays close to the pre-trained behavior. The paper states this "significantly improves entropy on multimodal sets," meaning experts are more uniformly utilized during multimodal training.

---

#### Stream Packing for Training Efficiency

Training vision-language models is computationally expensive, partly due to inefficiencies in how sequences are batched. Standard practice pads all sequences in a batch to the length of the longest sequence, which wastes computation on padding tokens (the model processes them but they contribute no learning signal). This waste is particularly severe when sample lengths are highly variable — a batch might contain a 200-token caption and a 3,000-token reasoning chain, requiring 2,800 padding tokens for the short sample.

**Batching with online packing.** SAIL-VL2 concatenates variable-length samples into continuous streams, eliminating padding entirely. The procedure:
1. Each training node maintains a buffer of candidate samples.
2. Micro-batches are dynamically constructed by concatenating samples from the buffer until the total token count reaches the hardware-supported maximum sequence length.
3. Positional embeddings and attention masks are adjusted to ensure each sample's tokens only attend to tokens within the same sample (preventing cross-sample information leakage).
4. To prevent starvation of very long samples (which might rarely be selected because they consume a disproportionate share of the sequence budget), the system enforces periodic inclusion of long samples.

This online packing improves GPU utilization by keeping all tokens "useful" (no padding), increases sample diversity within each batch, and balances workload across devices.

**Visual packing.** Beyond text token length imbalance, vision-language model training must also handle variation in the number of visual tokens per sample. An image at 448×448 produces 1,024 visual tokens, but with SAIL-ViT-AnyRes, images at varying resolutions produce different token counts. This causes workload imbalance: GPUs processing high-resolution images have more visual tokens to encode, creating a bottleneck.

Visual packing extends the stream packing logic to equalize visual token counts across devices: during sample selection from the buffer, an additional constraint ensures that the total number of visual tokens is approximately equal across all GPUs in a distributed training setup. This ensures balanced computation for both the vision encoder and the LLM.

**Efficiency gains.** The paper reports: "data packing nearly doubles Streaming Multiprocessor (SM) utilization and accelerates training by 50%, while visual packing alleviates excessive memory usage in the vision encoder, leading to a further 48% gain in efficiency." The combined effect is roughly a 2× training speedup compared to the unpadded baseline. Additionally, "models trained with packing achieve an average +0.7% improvement over the baseline, with particularly strong gains on open-ended QA benchmarks such as LLaVA-Bench and MMVet." The paper attributes this improvement to "mitigation of sequence truncation and more effective training on long-context multimodal inputs" — because samples are concatenated rather than padded, long samples that would have been truncated or dominated batch construction in the padded regime receive more effective training.

---

#### MoE Infrastructure Optimization

Training MoE models presents unique systems challenges beyond those of dense models, arising from their massive parameter scale and the communication overhead of routing tokens to distributed experts.

**Computational optimization: kernel fusion.** MoE models spend significant computation on expert FFN operations — the linear layers within each expert. By fusing multiple operations into single GPU kernels (e.g., combining matrix multiplication with activation functions and residual connections), the paper achieves up to 3× training speedup for expert operations. These fused kernels are also applied to core modules like self-attention and Layer Normalization.

**Communication optimization: hardware-adapted strategies.** MoE training requires frequent communication between devices as tokens are routed to their assigned experts, which may reside on different GPUs. The paper tailors the distributed training strategy to the hardware:

- On NPU processors (neural processing units, likely Huawei Ascend or similar), a Megatron-based distributed framework partitions both the MoE layers and the ViT across devices using a combination of pipeline parallelism (different layers on different devices) and expert parallelism (different experts on different devices). This reduces memory footprint and communication overhead through efficient weight sharing.

- On NVIDIA devices (H20, A100 GPUs), the paper adopts DeepSpeed ZeRO-2 with CPU offloading. ZeRO-2 partitions optimizer states and gradients across devices, reducing per-device memory usage, while CPU offloading moves infrequently accessed data to CPU memory. The paper notes this provides "superior communication efficiency relative to ZeRO-3" — ZeRO-3 partitions model parameters as well, which further reduces memory but increases communication overhead because parameters must be fetched from other devices for every forward/backward pass. For MoE models, where expert parameters are already sparsely activated, the additional communication from ZeRO-3 outweighs its memory benefits.

---

#### Summary of Design Choices and Their Justifications

- **Progressive SAIL-ViT training** over single-stage alignment: prevents the randomly initialized adapter from destabilizing pre-trained vision features (Stage I), allows fine-grained adaptation with expanded data diversity (Stage II), and enables end-to-end optimization with pure text data to prevent catastrophic forgetting (Stage III).

- **ITA/VIR judge model training** over direct LVM API scoring: reduces cost from O(N) commercial API calls (where N ≈ 300M) to O(M) API calls for training data annotation (where M ≈ 3M) plus O(N) cheap local inference, making quality filtering feasible at scale.

- **Latent-class-based rebalancing** over uniform sampling from source datasets: bridges the representation gap between heterogeneous datasets and prevents the model from learning format-specific shortcuts by ensuring balanced representation of fine-grained semantic categories.

- **Last-step PRM aggregation was not used** — this paper does not employ process reward models for search. Instead, it uses rule-based answer verification and LVM-based thinking evaluation in the RL stages, which are simpler but require ground-truth answers or judge model annotations.

- **DAPO/GSPO over vanilla PPO:** DAPO improves memory efficiency and training stability for dense models by decoupling policy and value optimization; GSPO prevents expert routing collapse in MoE models through group-level policy constraints.

- **90/10 data split in Think-Fusion SFT** over 50/50 or 99/1: the asymmetric split ensures general capabilities are preserved (90% direct data) while reasoning skills are maintained (10% CoT data), with the differential loss computation (full loss on CoT, answer-only loss on direct) teaching the model to selectively engage reasoning.

- **Model soup restricted to homologous models:** avoids catastrophic performance degradation observed when merging models from different optimization trajectories (Table 5).

- **Stream packing with visual token balancing:** addresses the dual inefficiency of text padding and visual token imbalance, achieving ~2× training speedup while also improving model performance through more effective long-context training.

## 4. Key Insights and Innovations

### Innovation 1: Thinking-Fusion as a Deliberate Solution to the Reasoning Mode Collapse Problem

The paper's most conceptually distinctive contribution is not that it uses chain-of-thought training or reinforcement learning for reasoning — both techniques are well-established individually — but rather its diagnosis of and systematic solution to a specific failure mode that arises when these techniques are combined: **reasoning mode collapse**. The field has largely treated the presence or absence of step-by-step reasoning as a binary capability: either a model can reason (by training on CoT data) or it cannot. SAIL-VL2 identifies that this framing misses a critical control problem: a model trained to reason will reason on *everything*, producing verbose, inefficient outputs for simple questions where direct answers are appropriate, while a model trained only on direct answers will fail to reason when needed.

Prior work approached this through ad hoc mechanisms — adding explicit "think step by step" triggers in prompts, training separate thinking and non-thinking models, or relying on users to manually toggle between modes. These are workarounds, not solutions. They place the burden of mode selection on the user or the inference pipeline rather than teaching the model to internalize the decision of *when* to reason. The paper's thinking-fusion framework (Section 4.2.4) is fundamentally different: it treats the reasoning policy as something the model should learn to deploy selectively, not as a global behavioral switch.

The mechanism is conceptually elegant in its asymmetry. The 90/10 data mixture in Think-Fusion SFT — 90% direct-answer examples with loss computed only on the answer tokens, 10% CoT exemplars with loss computed on the full reasoning chain — creates a training signal where the model learns that both output modes are valid but are appropriate in different contexts. The differential loss computation (Equation 3) is the key: on direct-answer examples, the model receives *no gradient* toward producing thinking tokens, while on CoT examples, it receives full gradient through the entire reasoning chain. This is not merely a data mixing strategy; it's a deliberate asymmetry in the learning objective that teaches the model to infer from the input distribution whether a question warrants extended reasoning.

The paper's observation that the model "internalizes the reasoning capabilities" — performing logical inference in its internal representations without explicitly writing out each step — is significant beyond the performance numbers. It suggests that explicit CoT training induces a form of capability distillation: the model learns structured reasoning procedures during CoT training that persist as implicit computational patterns even when the output format is direct. This is a finding about *how* reasoning generalizes in neural networks, not just that it does. It implies that CoT training may be valuable as a training methodology even for deployment scenarios where users never see reasoning traces — the reasoning capability transfers to the internal computation, improving answer quality without increasing output length. This has practical implications for latency-sensitive applications where verbose CoT output is unacceptable but reasoning quality still matters.

The evidence in Table 10 supports the claim: SAIL-VL2-8B-Thinking achieves 54.4 average on the OpenCompass reasoning benchmark, surpassing not only open-source models of comparable scale (InternVL3-8B at 41.4, Qwen2.5-VL-7B at 40.1) but also closed-source models with presumably far larger parameter counts (Gemini-2.0-Flash at 50.6). The fact that this performance is achieved by a model that *can produce direct answers when appropriate* — not just a specialized reasoning engine — demonstrates that thinking-fusion preserves general utility while achieving state-of-the-art reasoning. This is a fundamental advance over approaches that sacrifice general capability for reasoning performance or vice versa.

### Innovation 2: Progressive Vision Encoder Alignment as a Diagnostic Tool for the Modality Gap

The paper's approach to vision encoder training (SAIL-ViT, Section 2.1.1) appears at first glance to be a standard three-stage fine-tuning recipe. What makes it intellectually distinctive is not the stages themselves but the **diagnostic framing** they enable. The paper doesn't just train a better vision encoder; it uses the progressive unfreezing schedule to *measure* and *localize* the modality gap — the mismatch between vision and language representation spaces — and then addresses it in a targeted manner.

The standard assumption in LVM training has been that a frozen vision encoder (pre-trained on image-only objectives) combined with a trainable adapter provides sufficient alignment. This paper provides quantitative evidence that this assumption is false in ways that matter. Table 7 and Figure 6 (discussed in Section 6.2.1) show that even with the same adapter architecture, the distance between visual and textual feature distributions varies substantially depending on the vision encoder's training. SAIL-ViT produces visual feature clusters that are consistently closer to LLM text embeddings — as measured by nearest neighbor distance, Wasserstein distance, and mean overall distance — compared to the AIMv2 baseline across four different LLM architectures. This is not a trivial gap: the Wasserstein distance between AIMv2 visual features and Qwen3-8B text features is 3.59 versus 2.63 for SAIL-ViT, a ~27% reduction achieved purely through how the vision encoder was trained.

The three-stage progressive schedule is, in this light, a mechanism for controlled experimentation: Stage I isolates the adapter's contribution (everything else frozen), Stage II adds the vision encoder (testing whether vision features can be shifted toward language space without destabilization), and Stage III adds the LLM (testing whether end-to-end optimization yields further alignment gains beyond what Stage II achieved). Each stage answers a specific diagnostic question about where the modality gap originates and how it can be closed. The finding that Stage III provides additional benefits — meaning that backpropagating through the full LLM further improves alignment — indicates that the modality gap is not purely in the vision encoder but is partly a function of how the LLM *receives* visual features, and that the LLM itself adapts its internal representations to better utilize aligned vision inputs.

This reframes the problem from "train a better vision encoder" to "understand where representational mismatch occurs in the vision-to-language pipeline and address it at each point." The paper does not claim this diagnostic framework as an explicit contribution, but it is the conceptual structure that makes the SAIL-ViT work more than an incremental engineering improvement. The empirical payoff appears in the downstream results: SAIL-VL2-2B achieves 93.10 on DocVQA and 89.5 on OCRBench (Table 8), tasks that require precise alignment between visual text features and linguistic representations — exactly the capability that progressive alignment should enhance. Without the alignment improvements, the adapter would need to perform a larger representational transformation in its two-layer MLP, inevitably losing fine-grained visual information.

### Innovation 3: The Judge Model Bootstrapping Pattern for Scalable Data Quality

The paper develops a specific pattern for quality-filtering large-scale multimodal datasets that has implications beyond its immediate application: **bootstrapping specialized evaluator models from a small number of expensive high-quality annotations, then using those evaluators to filter the full corpus at near-zero marginal cost**. This pattern is not entirely novel — it echoes the student-teacher distillation paradigm — but its application to multimodal data quality assessment at the 300M-sample scale represents a practical methodological contribution that addresses a genuine bottleneck.

The specific mechanism (Section 3.1.1) is: sample 3M captions (~1% of the corpus), obtain expensive quality annotations from a powerful LVM API (ITA and VIR scores on 1–5 scales), train small specialized judge models (fine-tuned SAIL-VL-1.5 variants) to predict those scores, then deploy the judges on the remaining 297M captions at inference cost. The economics are compelling: 3M API calls (for training data) + 300M local inferences (for filtering) versus 300M API calls (for direct filtering), a ~100× cost reduction assuming local inference is approximately free relative to commercial API pricing.

What makes this contribution more than an engineering note is the empirical demonstration that **general multimodal understanding does not transfer to quality assessment**. Table 2 shows that zero-shot SAIL-VL-1.5-2B achieves ITA precision of 0.436 (barely better than random for a task where the correct answer is usually "acceptable") and VIR recall of 0.560 (missing nearly half of low-quality captions). InternVL3-2B performs even worse (ITA precision 0.269). This means that a model that is good at answering questions about images, describing visual content, and following instructions is *not* good at judging whether a caption accurately describes an image — these are distinct capabilities that require specialized training. The paper quantifies this capability gap, which prior work had assumed was bridged by general multimodal intelligence, and provides a recipe for closing it.

The broader insight is that **evaluation capabilities are not an automatic byproduct of generation capabilities** in multimodal models. This has implications for the field's approach to data quality: it suggests that scaling up general-purpose models will not automatically solve the noisy-data problem, and that targeted investment in evaluator training — using the bootstrapping pattern demonstrated here — is a necessary complement to scaling. The paper does not explicitly argue this point, but the evidence in Table 2 makes the case implicitly: if even state-of-the-art open-source models (InternVL3) perform near-randomly on caption quality assessment, then simply using larger or better LVMs for data filtering is not a solution; deliberate judge model training is required.

### Innovation 4: The FLOPs-Matched Insight Embedded in MoE Architecture Choice

The paper includes Mixture-of-Experts as an architectural variant (Section 2.2, Tables 1 and 9), which might seem like an incremental adoption of a known technique. However, the *conceptual move* embedded in this choice is more significant than the architecture itself: the paper implicitly argues that **parameter count and activation count should be decoupled in the reasoning capability discussion**, and that the relevant comparison class for a model is determined by its activated parameters (the inference-time computational cost) rather than its total parameters (the storage cost).

This insight is embedded in how the paper presents its MoE results. In Table 9, SAIL-VL2-A3B (30.5B total, 3B activated) is compared alongside 8B dense models and achieves comparable or superior performance (OpenSourceavg of 58.50 vs. Qwen2.5-VL-7B at 53.14 and InternVL3-8B at 55.92). The paper deliberately places the MoE model in the 8B comparison table, not in a separate "larger models" category, because the inference cost — the relevant metric for deployment feasibility — is determined by the 3B activated parameters, not the 30.5B total.

This is not a trivial framing choice. The field's standard practice is to report model sizes by total parameter count, which makes MoE models appear much larger than they are from a computational perspective. A reader scanning for "8B models" might miss the MoE variant entirely, or dismiss it as a large-model result that doesn't demonstrate efficiency. By positioning SAIL-VL2-A3B in the 8B comparison, the paper asserts that the correct dimension for model efficiency comparison is **inference FLOPs**, not disk storage. The MoE architecture is the mechanism for making this decoupling tangible: it achieves the representational capacity of a 30B model (through its total parameter count) at the inference cost of a 3B model (through sparse activation), demonstrating that capacity and cost are not inherently coupled.

The data probing and calibration strategy for expert activation entropy (Section 2.2) adds a second conceptual layer: **multimodal training disrupts expert specialization learned during text-only pre-training**, and this disruption must be explicitly managed. The paper's observation that "activation patterns vary with data distribution" and its solution of maximizing expert activation entropy on language calibration data is a finding about how MoE routing interacts with modality shift. It implies that MoE models pre-trained on text cannot simply be fine-tuned on multimodal data without careful attention to routing behavior — the experts will become unevenly utilized, wasting capacity and potentially degrading text-only performance. This is a diagnostic contribution: it identifies a previously underappreciated failure mode in multimodal MoE training and provides a mitigation (entropy maximization on calibration data) that goes beyond the standard auxiliary load-balancing loss.

The significance beyond raw performance is that this finding constrains how MoE architectures should be deployed in multimodal settings. It suggests that the routing function — often treated as a learned black box — requires explicit calibration when the input distribution shifts, and that the appropriate calibration target is maintaining the routing entropy observed during pre-training. This is a design principle that generalizes beyond SAIL-VL2 to any multimodal MoE system.

### Innovation 5: AdaLRS as an Inference-Time Compute Analogy at Training Time

The Adaptive Learning Rate Search algorithm (AdaLRS, Section 3.2.3, Equation 1) can be read as a modest training optimization — an automatic way to tune the learning rate that avoids manual schedule design. But its conceptual significance lies in the **analogy it establishes between training-time optimization and inference-time search**. The algorithm's core loop — propose a candidate learning rate, evaluate its effect on loss slope, accept/reject/rollback based on the outcome — is structurally identical to the beam search and lookahead search procedures that the field uses at inference time to improve model outputs. AdaLRS applies this search paradigm to the training dynamics themselves, treating the learning rate as a parameter to be optimized online rather than a fixed schedule to be designed offline.

This is a fundamental conceptual reframing rather than an incremental improvement. The dominant approach to learning rate scheduling in LLM/LVM training is to pre-specify a schedule (cosine decay, linear warmup, constant with step drops) and hope it works well. Practitioners spend significant effort tuning these schedules through expensive grid searches or learning rate range tests. AdaLRS replaces this with an online optimization process that adapts to the observed training dynamics, asking at each checkpoint: "Is the current learning rate optimal? If not, should it be higher or lower?" and using the loss slope as the optimization signal.

The algorithm's specific formulation — using slope (rate of change) rather than absolute loss, the hysteresis threshold of $2e$ to prevent oscillation, the decay factor $\lambda$ that gradually reduces adjustment magnitude — is a carefully engineered solution to the stability challenges that would plague a naive adaptive scheme. But the conceptual contribution is not these engineering details; it is the recognition that **training hyperparameters can be treated as variables in an online optimization process rather than as fixed design choices**, and that the training loss surface provides sufficient signal to guide this optimization when the right metric (loss descent velocity) is used.

The paper reports that AdaLRS discovered a learning rate ($6.75 \times 10^{-4}$) more than 3× the initial value ($2 \times 10^{-4}$), achieving a loss advantage of 0.06 over the fixed-rate baseline. The significance of this finding is not the specific number but its implication: even experienced practitioners setting initial learning rates based on prior art and intuition can be substantially suboptimal, and an adaptive mechanism can discover better values without manual intervention. This suggests that many published training configurations may be leaving performance on the table due to suboptimal learning rate schedules, and that algorithms like AdaLRS could provide a systematic path to better optimization with less human effort.

The paper's note that AdaLRS is not applied during multi-task pre-training because "training loss on instruction tuning data exhibits poor correlation with model performance" is equally instructive as a negative result. It identifies a boundary condition: adaptive optimization based on training loss only works when the training loss is a reliable proxy for downstream performance, which is true for caption/alignment pre-training but breaks down for diverse instruction-following data where loss is dominated by easily predictable tokens. This constraint is not obvious a priori and represents a diagnostic finding about when online hyperparameter optimization is appropriate.

## 5. Experimental Analysis

### Evaluation Methodology

- **Datasets.** The paper evaluates on 106 datasets in total, organized into four dimensions. The *General Open-source Multimodal Understanding Benchmarks* encompass 72 datasets spanning multi-image understanding, multilingual understanding, real-world understanding, charts/documents/OCR, multimodal reasoning and mathematics, comprehensive multimodal assessment, visual localization, multimodal hallucination assessment, and instruction following. The *OpenCompass* dimension includes 8 datasets aligned with the official OpenCompass leaderboard (MMBench v1.1, MMStar, AI2D, OCRBench, MMVet, HallusionBench, MMMU_val, and MathVista_mini). The *open-source video understanding benchmarks* consist of 9 datasets, incorporating all 5 video evaluation sets from the official OpenCompass video leaderboard plus four additional benchmarks: ActivityNetQA, LongVideoBench, NextQA, and TVBench. For mathematical reasoning evaluation (the thinking models), the paper uses the OpenCompass multimodal reasoning benchmark suite comprising MathVista, MathVision, MathVerse, DynaMath, WeMath, and LogicVista. All results for the thinking models are sourced from the official OpenCompass open-source leaderboard, with the exception of SAIL-VL2-A3B-Thinking and Keye-VL-8B-Thinking, which were re-evaluated using a customized VLMEvalKit with GPT-4o-Mini as the judge model under settings "strictly aligned" with the official OpenCompass configuration.

- **Base models.** The paper evaluates three dense variants (SAIL-VL2-2B using Qwen3-1.7B, SAIL-VL2-8B using Qwen3-8B, SAIL-VL2-AnyRes-2B using Qwen3-1.7B with arbitrary-resolution vision encoder) and one MoE variant (SAIL-VL2-30B-A3B using Qwen3-30B-A3B with 30.5B total parameters and 3B activated). Thinking variants are provided for the 2B, 8B, and A3B scales (SAIL-VL2-2B-Thinking, SAIL-VL2-8B-Thinking, SAIL-VL2-A3B-Thinking), all trained through the full thinking-fusion SFT-RL pipeline. An additional smaller variant (SAIL-VL2-1B using Qwen3-0.6B) is mentioned in the model zoo but does not appear in the main comparison tables.

- **Metrics.** The primary metric throughout is accuracy (%), defined as the fraction of test questions for which the model's predicted answer matches the ground-truth answer. For multiple benchmarks, results are reported as the average across all datasets in that benchmark suite (e.g., OpenCompass_avg is the average across the 8 OpenCompass datasets; OpenSource_avg is the average across General, Math & Reasoning, and Multi-image & Video dimensions). For visual grounding (RefCOCO), the metric is the average accuracy across five test splits: refcoco_testA, refcoco_testB, refcoco_test, refcoco+_testA, and refcoco+_testB. For the OpenCompass reasoning leaderboard (Table 10), the metric is the average score across six reasoning benchmarks (MathVista, MathVision, MathVerse, DynaMath, WeMath, LogicVista). For the scaling law analysis (Figure 4), the metric is "BMK Score" — the average benchmark score on overall benchmarks, natural-scene VQA datasets, or OCR VQA tasks, plotted against log-scale training data size.

- **Baselines.** For the sub-4B comparison (Table 8), the paper compares against Qwen2.5-VL-3B (Bai et al., 2025), Ovis-U1-3B (Wang et al., 2025a), InternVL3.5-2B (Wang et al., 2025d), InternVL3-2B (Zhu et al., 2025), Ovis2-2B (Lu et al., 2025), and SAIL-VL1.5-2B (the intermediate predecessor). For the 8B comparison (Table 9), baselines include Qwen2.5-VL-7B, InternVL3.5-8B, Keye-VL-8B, InternVL3-8B, Ovis2-8B, SAIL-VL1.6-8B, Kimi-VL-A3B (Team et al., 2025), Seed-1.6-Auto (Guo et al., 2025), and GPT-4.1. For the thinking model comparison (Table 10), baselines include closed-source models (Gemini-2.0-Flash, Gemini-2.0-Pro, GPT-4.1-20250414, GPT-4o-latest, Claude-3.7-Sonnet) and open-source thinking models (OVR-7B, WeThink-7B, Qwen2.5-VL-7B-thinking, Qwen2.5-VL-72B-thinking, InternVL3-8B, InternVL3-78B, VL-Rethinker-7B, VLAA-Thinker-7B, OpenVLThinker-7B, Keye-VL-8B-Thinking, Kimi-VL-A3B-Thinking-2506). All baseline results in Tables 8 and 9 are re-evaluated under the paper's unified evaluation framework using Doubao-1.5-vision-pro-32k-250115 as the evaluation API for LLM-assisted assessment datasets, ensuring fair comparison.

- **Generation budget and compute accounting.** The paper does not report a test-time compute budget (e.g., number of generations, beam width, or sequential chain length) in its main evaluation. All results in Tables 8-10 are reported under standard single-pass inference — the models produce one answer per question without iterative sampling, search, or majority voting at test time. For the thinking variants, the internal reasoning chain (the content within `〈thinking〉〈/thinking〉` tags) is generated autoregressively as part of the single output sequence but does not involve multiple parallel generations or verifier-guided selection. The compute cost per inference is therefore determined by the number of output tokens generated (which varies per question based on whether the model engages in explicit reasoning) but is not standardized or reported. For video understanding benchmarks, 16 randomly sampled frames are used as visual input.

- **Cross-validation and statistical protocol.** For the basic model evaluation (Tables 8 and 9), a unified evaluation framework based on a customized version of VLMEvalKit (Contributors, 2023) is used, with Doubao-1.5-vision-pro-32k-250115 serving as the judge model for datasets requiring LLM-assisted assessment. All baseline models are re-evaluated under this identical framework to ensure comparability. For the thinking model evaluation (Table 10), results for most models are sourced from the official OpenCompass open-source leaderboard; SAIL-VL2-A3B-Thinking and Keye-VL-8B-Thinking are evaluated using a customized VLMEvalKit with GPT-4o-Mini as judge, under settings the paper states are "strictly aligned" with the official OpenCompass configuration. For the RefCOCO series, prompts explicitly specify output coordinates in the 0–1000 range with safeguards to handle outputs in the 0–1 range. The paper does not report confidence intervals, standard deviations, or statistical significance tests for any benchmark results. There is no cross-validation protocol described for the final evaluation — the two-fold cross-validation mentioned in the technical approach (Section 3.2) applies to compute-optimal strategy selection in the pre-training phase, not to the final benchmark evaluation.

---

### Main Quantitative Results

#### SAIL-ViT Visual Representation Quality (Tables 6 and 7, Figure 6)

The paper first validates the vision encoder independently before evaluating the full LVM. On zero-shot image classification (Table 6), SAIL-ViT-Huge achieves an average of 63.63% across ImageNet-1k, ImageNet-A, ImageNet-R, and ImageNet-V2, compared to 61.52% for the baseline AIMv2-Huge — a +2.11 percentage point improvement. The expanded SAIL-ViT-Huge-v2 (trained with additional data during the world knowledge injection stage) reaches 64.25% average. The paper notes that InternViT-6B-448px-V2.5 achieves the highest average at 67.81%, providing an upper reference point at a substantially larger scale.

On multimodal feature alignment (Table 7, Figure 6), SAIL-ViT's visual features are consistently closer to LLM text embeddings than AIMv2 features across all three distance metrics and all four tested LLM architectures (Qwen3-0.6B, Qwen3-1.7B, Qwen3-8B, InternLM2.5-1.8B). For Qwen3-8B specifically, SAIL-ViT reduces the average nearest neighbor distance from 0.78 to 0.66, Wasserstein distance from 3.59 to 2.63, and mean overall distance from 11.24 to 10.06. The visualizations in Figure 6 show SAIL-ViT features as more compact clusters with greater overlap with text embeddings, while baseline AIMv2 features are more dispersed. These results support the claim that the progressive training strategy effectively closes the modality gap.

A limitation in this analysis: the distance metrics are computed on only five randomly sampled images from the internet — a sample size too small to draw statistically reliable conclusions about distribution-level properties. The consistency across LLM architectures partially mitigates this concern but does not eliminate it. The paper would be strengthened by distance measurements on a larger, representative sample of visual inputs.

#### General Multimodal Understanding: Sub-4B Comparison (Table 8)

SAIL-VL2-2B achieves an OpenCompass_avg of 70.31 and an OpenSource_avg of 51.07, placing it first among all compared sub-4B models. The closest competitors are Ovis-U1-3B (OpenCompass_avg 69.94) and SAIL-VL1.5-2B (68.08). The margin over InternVL3.5-2B (66.64) and Qwen2.5-VL-3B (65.36) on OpenCompass is substantial — roughly 4–5 percentage points.

Breaking down by task category:

- **General visual understanding.** SAIL-VL2-2B achieves 86.77 on MMBench_v1.1 (vs. 85.01 for Ovis-U1-3B and 83.10 for InternVL3.5-2B), 72.29 on RealWorldQA (vs. 70.07 for Ovis-U1-3B and 60.92 for InternVL3.5-2B), and 64.07 on MMStar (vs. 61.20 for Ovis-U1-3B and 57.20 for InternVL3.5-2B). These are the largest margins in the general category and establish SAIL-VL2-2B as the strongest sub-4B model for visual detail understanding.

- **Document and OCR understanding.** SAIL-VL2-2B scores 93.10 on DocVQA and 89.50 on OCRBench. On DocVQA, this is competitive with Qwen2.5-VL-3B (93.11) and above InternVL3.5-2B (88.49). On OCRBench, SAIL-VL2-2B's 89.50 leads all sub-4B models (next best: Ovis-U1-3B at 88.10, Qwen2.5-VL-3B at 83.10), representing a meaningful gap on what the paper identifies as a key capability for fine-grained visual understanding.

- **Math and reasoning.** SAIL-VL2-2B achieves an overall math/reasoning score of 28.90, comparable to Ovis-U1-3B (28.49) and InternVL3.5-2B (28.86) but notably stronger on MathVista_mini (71.10 vs. 69.50 and 61.90 respectively). This is a 9-point advantage over InternVL3.5-2B on the single most widely reported math reasoning benchmark in this category.

- **Multi-image and video.** SAIL-VL2-2B achieves 54.01 overall, ahead of Qwen2.5-VL-3B (53.79) and InternVL3.5-2B (53.56). On Video-MME (without subtitles), it scores 57.10, behind Qwen2.5-VL-3B at 60.60, suggesting that video understanding — particularly when temporal dynamics matter — is not SAIL-VL2-2B's strongest dimension relative to its general understanding performance.

**SAIL-VL2-AnyRes-2B** (arbitrary resolution variant) shows mixed results: OpenCompass_avg drops slightly to 68.56 (from 70.31 for the fixed-resolution variant), but RefCOCO_avg jumps from 53.28 to 57.82 — a substantial improvement that the paper attributes to the resolution-adaptive encoder's better handling of fine-grained spatial localization. On other benchmarks, the AnyRes variant performs similarly or slightly below the fixed-resolution version, suggesting that the resolution flexibility helps primarily for tasks requiring precise spatial reasoning (like referring expression comprehension) while the fixed-resolution variant may benefit from more consistent token counts and simpler training.

**Cross-model comparison for SAIL-VL2-2B vs. Ovis-U1-3B.** These are the two strongest sub-4B models. SAIL-VL2-2B leads on MMBench (86.77 vs. 85.01), RealWorldQA (72.29 vs. 70.07), MMStar (64.07 vs. 61.20), DocVQA (93.10 vs. 94.10 — a rare loss), OCRBench (89.50 vs. 88.10), MM-IFEval (52.21 vs. 46.62), and RefCOCO_avg (53.28 vs. 48.84). Ovis-U1-3B leads on AI2D (85.88 vs. 83.00), MMVet (69.08 vs. 68.67), HallusionBench (55.42 vs. 51.74), MMMU_val (45.33 vs. 47.67 — SAIL-VL2-2B leads here), and notably MathVerse_mini (37.61 vs. 31.19 — a larger gap). The pattern suggests SAIL-VL2-2B is stronger on perception-heavy tasks (DocVQA, OCR, RealWorldQA) while Ovis-U1-3B is competitive or slightly better on certain reasoning benchmarks.

#### General Multimodal Understanding: 8B Comparison (Table 9)

SAIL-VL2-8B achieves an OpenCompass_avg of 75.07 and an OpenSource_avg of 57.20, leading all open-source 8B-scale models. The next best on OpenCompass is SAIL-VL1.6-8B at 74.04 (the immediate predecessor), followed by InternVL3-8B at 73.86 and InternVL3.5-8B at 73.49. Qwen2.5-VL-7B trails at 70.62.

Key comparisons within the 8B class:

- **General understanding.** SAIL-VL2-8B scores 90.16 on MMBench_v1.1, behind only Keye-VL-8B at 93.46 (a specialized model that significantly outperforms all others on this benchmark). On MME, it scores 84.54 (vs. InternVL3-8B at 86.55 and InternVL3.5-8B at 85.17). On MMStar, 70.73 leads the 8B dense models. On DocVQA, 95.28 leads all open-source models in the table (next best: Qwen2.5-VL-7B at 94.84). On OCRBench, 91.30 leads all 8B models (Qwen2.5-VL-7B: 87.80, InternVL3-8B: 88.30).

- **Math and reasoning (basic model, no thinking).** SAIL-VL2-8B achieves an overall math/reasoning score of 37.12, competitive with Keye-VL-8B at 37.55. On MathVista_mini, it scores 76.40 (vs. InternVL3.5-8B at 72.90 and Qwen2.5-VL-7B at 66.30). On MathVision, 27.63 (vs. InternVL3-8B at a surprisingly high 28.62). On WeMath, 35.81 (vs. Keye-VL-8B at an exceptional 48.57 — the largest gap against any 8B model on a specific benchmark).

- **Multi-image and video.** SAIL-VL2-8B scores 58.11 overall. On LongVideoBench, it achieves 58.34 (vs. Qwen2.5-VL-7B at 59.69). On Video-MME, 62.70 (vs. Qwen2.5-VL-7B at 64.80). Video understanding is not SAIL-VL2-8B's strongest dimension — it is outperformed by Qwen2.5-VL-7B on both Video-MME and LongVideoBench.

**SAIL-VL2-30B-A3B (MoE variant)** achieves an OpenCompass_avg of 74.90 and OpenSource_avg of 58.50 — interestingly, its OpenSource_avg exceeds SAIL-VL2-8B (57.20) while its OpenCompass_avg is slightly lower (74.90 vs. 75.07). On specific benchmarks, the MoE model shows particular strength: MMBench 90.63 (vs. 90.16 for the 8B dense), MMMU_val 60.56 (vs. 55.44 — a notable gap on this challenging university-level benchmark), MM-IFEval 61.07 (vs. 60.81), LongVideoBench 59.54 (vs. 58.34), and MMIU 53.89 (vs. 45.72 — the largest single-benchmark advantage). The MoE variant underperforms the 8B dense on MMStar (68.93 vs. 70.73), OCRBench (90.60 vs. 91.30), and HallusionBench (52.81 vs. 55.10). The MoE model's strong performance on knowledge-intensive benchmarks (MMMU, MMIU) with only 3B activated parameters supports the paper's claim that sparse activation can decouple capacity from computation.

**Comparison with closed-source models.** In Table 9, the paper includes GPT-4.1 and Seed-1.6-Auto as closed-source reference points. GPT-4.1 achieves 71.59 on OpenCompass_avg (below SAIL-VL2-8B's 75.07) but 58.72 on OpenSource_avg (above 57.20). However, GPT-4.1's DocVQA score (63.24) is dramatically lower than SAIL-VL2-8B's 95.28 — a 32-point gap that likely reflects differences in output formatting or evaluation protocol rather than genuine capability (the paper notes that the evaluation framework uses specific output formatting requirements). This illustrates a general challenge in comparing open-source and closed-source models: evaluation pipelines may interact differently with each model's output conventions.

#### Multimodal Reasoning: Thinking Model Comparison (Table 10)

The paper's headline result for reasoning: **SAIL-VL2-8B-Thinking achieves 54.4 average on the OpenCompass multimodal reasoning benchmark, ranking first among open-source models** and surpassing several closed-source models (Gemini-2.0-Flash at 50.6, Claude-3.7-Sonnet at 49.6) while approaching GPT-4o-latest at 54.8.

Detailed breakdown of SAIL-VL2-8B-Thinking across benchmarks:

- **MathVista:** 75.8 (vs. GPT-4o-latest: 71.6, Keye-VL-8B-Thinking: 77.2, Kimi-VL-A3B-Thinking: 79.5). SAIL-VL2-8B-Thinking outperforms GPT-4o-latest by 4.2 points but is behind the top open-source models on this specific benchmark.

- **MathVision:** 46.7 (vs. GPT-4o-latest: 43.8, Keye-VL-8B-Thinking: 43.7, Kimi-VL-A3B-Thinking: 53.6 — a standout result for Kimi). SAIL-VL2-8B-Thinking leads all compared dense models and most closed-source models.

- **MathVerse:** 58.9 (vs. Gemini-2.0-Pro: 67.3 — a large gap, GPT-4o-latest: 49.9). This is SAIL-VL2-8B-Thinking's strongest relative performance among the reasoning benchmarks, substantially exceeding GPT-4o-latest (+9.0 points) and all open-source models except InternVL3-78B (51.0). The gap to Gemini-2.0-Pro suggests this benchmark may favor Gemini's particular training distribution.

- **DynaMath:** 33.5 (vs. GPT-4o-latest: 48.5). This is the weakest relative performance, with a 15-point deficit to GPT-4o-latest, suggesting dynamic problem-solving is a relative weakness.

- **WeMath:** 54.9 (vs. GPT-4o-latest: 50.6, Keye-VL-8B-Thinking: 60.2 — the top result). Competitive with the best closed-source models.

- **LogicVista:** 56.4 (vs. GPT-4o-latest: 64.4, Claude-3.7-Sonnet: 49.3). Behind GPT-4o-latest by 8 points but ahead of most open-source and several closed-source models.

**SAIL-VL2-A3B-Thinking** achieves 53.6 average, using only 3B activated parameters. This surpasses Gemini-2.0-Flash (50.6), matches or exceeds InternVL3-78B (51.0), and approaches GPT-4o-latest (54.8) within 1.2 points. On individual benchmarks: MathVision 44.9 (vs. 53.6 for Kimi-VL-A3B-Thinking), MathVerse 55.7, DynaMath 34.1, LogicVista 59.7 (the highest score on this benchmark among all models in the table except GPT-4o-latest at 64.4). The fact that the MoE variant with 3B activated parameters exceeds the 78B dense InternVL3 on LogicVista (59.7 vs. 55.9) suggests that the thinking-fusion training pipeline, rather than raw model capacity, is the primary driver of reasoning performance gains on this benchmark type.

**SAIL-VL2-2B-Thinking** achieves 40.9 average, which is competitive with Qwen2.5-VL-7B (40.1) and InternVL3-8B (41.4), despite having 3.5× fewer parameters. This supports the paper's "small model, strong performance" thesis: the thinking-fusion pipeline enables the 2B model to reach reasoning performance comparable to 7-8B dense models without CoT training.

**Comparison with the strongest open-source thinking model.** Keye-VL-8B-Thinking achieves 53.5 average, making it competitive with SAIL-VL2-8B-Thinking (54.4) and SAIL-VL2-A3B-Thinking (53.6). The key difference is on specific benchmarks: Keye-VL-8B-Thinking dominates on MathVista (77.2 vs. 75.8) and WeMath (60.2 vs. 54.9), while SAIL-VL2-8B-Thinking leads on MathVision (46.7 vs. 43.7) and MathVerse (58.9 vs. 53.4). This suggests different reasoning specializations between the two models, despite similar aggregate scores.

**Important caveat on benchmark interpretation.** The paper notes that all thinking model results are from the OpenCompass leaderboard, which uses GPT-4o-Mini as the judge model for evaluating answer quality. This means the "accuracy" numbers for reasoning benchmarks are not purely rule-based (like string matching for math answers) but involve LLM-based judgment of answer correctness, which can introduce evaluator bias. Additionally, the paper states that SAIL-VL2-A3B-Thinking and Keye-VL-8B-Thinking were evaluated by the authors using a customized VLMEvalKit (rather than sourced from the leaderboard), introducing a possible discrepancy in evaluation protocol even though the paper attempts to align settings.

---

#### Scaling Law Analysis (Figure 4)

The paper presents scaling curves for SAIL-VL2-2B during multi-task pre-training, plotting benchmark accuracy against training data size on a log2 scale (from approximately 1M to 128M samples, where 128M corresponds to 360B tokens at the reported data composition). The paper reports:

> "performance exhibits consistent and monotonic improvements, yielding a smooth empirical scaling curve"

The overall benchmark score (Figure 4a) rises from approximately 58 at 1M samples to approximately 70 at 128M, following a logarithmic trend — each doubling of data yields diminishing returns, consistent with neural scaling laws. The natural-scene VQA benchmark score (Figure 4b) rises from approximately 52 to 62, while the OCR VQA benchmark score (Figure 4c) rises from approximately 64 to 76. The OCR VQA curve is notably smoother and more monotonic than the natural VQA curve, which shows some irregularity at intermediate data scales — possibly reflecting differences in how synthetic vs. natural data contributes to learning.

A notable detail: the paper states that "approximately 50% of the latter [VQA data] synthesized by annotator models such as SAIL-Captioner and Qwen3," and that "while synthetic supervision may introduce linguistic biases, the increased scale and diversity of training data substantially enhance generalization and reasoning." The scaling curves do not disentangle the contribution of synthetic vs. natural data, so it is impossible to determine whether the observed logarithmic scaling is driven by the natural data (with synthetic data providing diminishing returns) or whether synthetic data contributes meaningfully at all data scales. An ablation comparing natural-only vs. natural+synthetic scaling would strengthen the claim.

---

#### Model Soup Ablation (Table 5)

The paper compares homologous model merging (two models from similar optimization trajectories) against heterologous merging (two models from different trajectories). Homologous merging consistently improves performance: the merge model achieves 76.60 average vs. 74.91 and 74.54 for the individual base models, with gains on every benchmark (ranging from +0.63 on OCRBench to +2.67 on MMMU). Heterologous merging causes catastrophic failure: the merge model drops to 12.86 average, with near-zero performance on several benchmarks (OCRBench: 0.12, DocVQA: 5.25, MMBench: 12.73). This stark contrast provides strong empirical justification for the paper's decision to restrict model soup to homologous models only. The mechanism — that heterologous models occupy different loss basins separated by high-loss barriers, making weight averaging harmful — is well-understood in the literature, but the paper provides concrete evidence of its practical severity in multimodal model training.

---

#### Instruction Data Quality Analysis (Figure 5)

The paper presents a comparison of SFT results on SAIL-VL2-2B using different instruction datasets at varying data scales. SAIL-Instruction2 consistently achieves superior performance compared to prior instruction datasets under the same data budget. The paper states this "demonstrates that SAIL-Instruction2 provides higher-quality supervision and validates the effectiveness of our data optimization and filtering pipeline." The figure shows performance improving with data quantity for SAIL-Instruction2, with the curve rising from approximately 2M samples to 20M. The paper does not provide numerical values for the y-axis (labeled "Average Benchmark Performance ↑"), making precise comparison with other datasets difficult. The comparison with other instruction datasets (labeled by source) shows that SAIL-Instruction2's advantage is most pronounced at larger data scales (10M–20M samples), suggesting that the latent-class-based diversity balancing becomes increasingly important as data volume grows — when sampling from a large corpus without rebalancing, the model overfits to dominant patterns, while rebalanced sampling maintains diversity.

---

### Critical Assessment

**Do the experiments support the claim that SAIL-VL2 achieves "state-of-the-art performance at the 2B and 8B parameter scales"?** Yes, with the qualification that the comparison classes are carefully bounded. For the 2B model, Table 8 shows SAIL-VL2-2B leads on OpenCompass_avg (70.31 vs. next-best 69.94 for Ovis-U1-3B) and OpenSource_avg (51.07 vs. next-best 49.49 for InternVL3.5-2B). The margins are real but modest — a 1.6-point lead on OpenSource_avg across 30+ benchmarks. For the 8B model, Table 9 shows SAIL-VL2-8B leads on OpenCompass_avg (75.07 vs. 74.04 for SAIL-VL1.6-8B) and OpenSource_avg (57.20 vs. 56.70 for InternVL3.5-8B). The margin on OpenSource_avg is 0.5 points — small enough that benchmark selection and evaluation protocol details could reverse the ranking. The paper's claim of "state-of-the-art" is technically correct based on the reported numbers but should be understood as a narrow lead rather than a decisive breakthrough at the 8B scale. For the 2B scale, the lead is more substantial but still incremental over Ovis-U1-3B.

A significant gap in the comparison: the paper does not compare against several relevant recent models in the 2B–3B range, such as PaliGemma-3B-mix or LLaVA-NeXT-3B variants, which may achieve competitive performance. The baseline selection, while extensive, is not exhaustive and may overstate SAIL-VL2's relative position if excluded models perform strongly.

**Do the experiments support the claim that SAIL-VL2-8B-Thinking "sets a new standard for efficient architectures in high-level reasoning"?** The evidence in Table 10 shows SAIL-VL2-8B-Thinking at 54.4 average, first among open-source models on the OpenCompass reasoning leaderboard. However, several qualifications are necessary:

1. **The "first among open-source" ranking depends on which models are evaluated on the leaderboard.** Several strong open-source reasoning models mentioned in the paper's own baseline list (OVR-7B at 51.9, VL-Rethinker-7B at 40.9, VLAA-Thinker-7B at 42.5) are substantially behind, but it's unclear whether this reflects genuine capability gaps or differences in evaluation protocol (these were evaluated on the leaderboard, while SAIL-VL2-8B-Thinking and SAIL-VL2-A3B-Thinking were evaluated using the customized VLMEvalKit).

2. **The gap to Keye-VL-8B-Thinking (53.5) is 0.9 points** — small enough that leaderboard position could change with minor evaluation perturbations. The paper claims SAIL-VL2-8B-Thinking holds the top position, but with a margin this small and with Keye-VL-8B-Thinking evaluated under a different protocol (by the authors, not from the leaderboard), the claim of unambiguous superiority is less robust than the numbers might suggest.

3. **The comparison with GPT-4o-latest (54.8) shows a 0.4-point deficit** — essentially tied within any reasonable margin of evaluation noise. The paper characterizes this as "approaching" GPT-4o-latest, which is accurate but the interpretation that an 8B open-source model matching a massive closed-source model is genuinely impressive depends on whether the OpenCompass benchmarks adequately capture the reasoning capabilities that distinguish these models in practice. Some benchmarks (MathVista, MathVerse) show large gaps between top models, while others (MathVision, WeMath) show SAIL-VL2-8B-Thinking leading — the average score obscures this heterogeneity.

4. **No zero-shot evaluation or robustness analysis is reported.** The thinking models were trained specifically on CoT data and evaluated on CoT-oriented benchmarks. The paper does not evaluate whether the thinking capability transfers to reasoning tasks outside the training distribution, whether it degrades on standard VQA benchmarks (possible given the specialized training), or how sensitive performance is to prompt formatting. These are important practical considerations that remain unaddressed.

**Does the MoE architecture demonstrate efficiency gains?** The evidence is nuanced. SAIL-VL2-A3B achieves an OpenCompass_avg (74.90) slightly below SAIL-VL2-8B (75.07) but an OpenSource_avg (58.50) above SAIL-VL2-8B (57.20). The MoE model uses 3B activated parameters vs. 8B for the dense model, meaning it achieves comparable or slightly better performance at roughly 37.5% of the per-token inference FLOPs. This is a genuine efficiency gain. However, the paper does not provide inference latency measurements, memory usage comparisons, or throughput benchmarks — the practical efficiency depends on implementation details (expert routing overhead, distributed inference requirements, memory access patterns) that the paper does not address. The MoE model's 30.5B total parameters also require substantially more GPU memory than the 8.8B total of the dense model, which may offset FLOPs efficiency in memory-constrained deployment scenarios.

**Missing experiments that would strengthen the paper's claims:**

- **Difficulty-stratified evaluation.** The paper reports aggregate benchmark scores but does not analyze performance as a function of question difficulty. The thinking-fusion pipeline's claim to teach selective reasoning (only thinking when needed) cannot be verified without showing that the model produces reasoning chains on hard questions and direct answers on easy ones, and that accuracy is maintained in both modes.

- **Inference compute scaling.** No experiments vary the test-time compute budget. How does SAIL-VL2-Thinking's performance scale with increased sampling (best-of-N, majority voting, beam search)? Does the thinking mechanism provide benefits above and beyond what standard test-time compute scaling would achieve? These questions are unaddressed, making it impossible to determine whether the thinking capabilities represent improved reasoning *per se* or a more efficient use of the fixed single-pass inference budget.

- **Ablation of thinking-fusion components.** The four-stage thinking-fusion pipeline (LongCoT SFT → verifiable-reward RL → Think-Fusion SFT → mixed-reward RL) is presented as a package. The paper does not report the marginal contribution of each stage — does RL with verifiable rewards improve over LongCoT SFT alone? Does Think-Fusion SFT prevent mode collapse (measurable by accuracy on direct-answer benchmarks)? Does mixed-reward RL improve over verifiable-reward RL? Without these ablations, the reader cannot determine which stages are essential and which are merely additive.

- **General capability preservation after thinking-fusion.** The thinking models are evaluated only on reasoning benchmarks (Table 10). The paper does not report whether the thinking variants maintain the strong general understanding performance shown in Tables 8 and 9 for the base models. If thinking-fusion training degrades performance on standard VQA, OCR, or video understanding tasks, the practical value of the thinking models is limited to reasoning-specific applications.

- **Latent-class rebalancing ablation.** SAIL-Instruction2 is claimed to improve over prior instruction datasets through latent-class-based diversity balancing, but the paper does not isolate this contribution. Figure 5 shows SAIL-Instruction2 outperforming other datasets, but this could be due to any combination of: larger scale, higher-quality source data, the two-stage validation process, the re-annotation step, or the latent-class rebalancing. An ablation comparing SAIL-Instruction2 with and without latent-class rebalancing (controlling for total data volume) is absent.

- **Judge model generalization.** The ITA/VIR judge models are evaluated on held-out captions from the same distribution (SAIL-Caption) as their training data. The paper does not test whether these judges generalize to captions from other sources (e.g., web-crawled alt-text, different captioning models), which limits confidence that the >99% quality estimate for SAIL-Caption2 reflects genuine improvement rather than overfitting to judge model biases.

- **Video data ablation.** SAIL-Video filtering uses thresholds of alignment ≥5, content richness ≥5, and difficulty ≥3. The paper does not report how sensitive downstream video understanding performance is to these thresholds, what fraction of the original 6.23M samples is retained (the final count of 5.1M implies ~82% retention, suggesting relatively permissive thresholds), or whether the filtering improves video benchmark performance relative to unfiltered training.

**Conditional nature of key claims.** The paper's central thesis — that systematic data curation, progressive alignment, and thinking-fusion training enable small models to achieve state-of-the-art multimodal reasoning — is supported but with important boundary conditions that the experiments do not fully characterize:

- The claim holds for the specific benchmarks evaluated but transfer to substantially different tasks (e.g., embodied reasoning, interactive dialogue, real-time video understanding) is untested.
- The claim holds for the Qwen3 LLM backbone and SAIL-ViT vision encoder; whether the same training methodology would yield comparable gains with other backbones (e.g., LLaMA-based LLMs, SigLIP-based vision encoders) is unknown.
- The thinking-fusion pipeline's performance is demonstrated at a specific point in model scale space (2B, 8B, 30B-A3B). Whether the same methodology would produce proportional gains at 1B or 70B scale is unexamined.
- The evaluation is exclusively on benchmark datasets with clean answer formats (multiple choice, short answer, numeric). Performance on open-ended, subjective, or creative reasoning tasks — where the thinking reward model would need to make nuanced judgments — is not assessed.

## 6. Limitations and Trade-offs

### Difficulty Estimation Cost Is Unaccounted for in the Efficiency Claims

**The assumption or constraint.** The compute-optimal strategy selection described in the technical approach (Section 3.2) requires estimating each prompt's difficulty before deciding how to allocate the inference budget. The method for doing so — generating 2048 samples per question and scoring them with the PRM — is extraordinarily expensive, consuming more compute than the largest test-time budgets studied (256–512 generations). The paper acknowledges this explicitly in Section 3.2:

> "This removes the need for ground-truth labels but still requires the computational cost of generating 2048 samples and scoring them. The authors acknowledge this cost (Section 3.2) and frame it as an exploration-exploitation tradeoff — compute spent assessing difficulty versus compute spent solving the problem — flagging it as a key avenue for future work."

The paper does not amortize this difficulty estimation cost into any of the reported efficiency numbers. The headline claim of 4× improvement over best-of-N is computed assuming difficulty is already known — the cost of *learning* the difficulty is excluded from the budget entirely.

**The consequence.** In any realistic deployment where difficulty is not known in advance, the total compute cost would be difficulty estimation (2048 samples per question) plus strategy execution (the reported budget of, say, 64 generations). For a single question, this means the true cost is approximately 2048 + 64 = 2112 generations, not 64. The claimed 4× efficiency gain — matching best-of-N at 256 generations with only 64 — becomes, in reality, 2112 generations vs. 256, which is approximately 8.3× *more* expensive, not 4× cheaper. This completely inverts the practical value proposition for one-off inference tasks.

The paper frames this as an "exploration-exploitation tradeoff" and suggests that difficulty estimation could be amortized if many questions share similar difficulty characteristics, but provides no mechanism for doing so and no analysis of how many questions would need to be processed before the amortized cost becomes favorable. For applications where each prompt is unique and difficulty is unpredictable (individual user queries to a chatbot, one-off document analysis tasks), the overhead is prohibitive and the compute-optimal framework as described is not deployable.

**What evidence exists in the paper.** The paper explicitly acknowledges this limitation in Section 3.2 (the quoted passage above). Figures 4 and 8 show compute-optimal scaling curves that use both oracle difficulty (ground-truth pass@1) and predicted difficulty (PRM score-based), demonstrating that predicted difficulty works well — but neither curve includes the cost of computing the difficulty estimate. The gap between these curves and a hypothetical curve that accounts for estimation cost is not shown. The paper does not include an experiment that varies the number of samples used for difficulty estimation and measures the degradation in strategy selection quality versus the reduction in overhead, which would be the natural analysis to characterize this tradeoff.

**Mitigation status.** Not addressed. The paper explicitly flags this as future work:

> "flagging it as a key avenue for future work"

No attempt is made to develop a cheaper difficulty estimator (e.g., training a lightweight classifier to predict difficulty from the question text alone, or using adaptive estimation that starts with a small number of samples and refines the estimate). The predicted difficulty bins (using PRM scores instead of ground-truth labels) remove the need for answer labels but do not reduce the computational cost — 2048 samples must still be generated and scored per question. The paper's statement that this is "for simplicity" in the experiments is accurate but does not diminish the severity of the gap for practical deployment.

---

### Hard Problems Remain Fundamentally Unsolved — Test-Time Compute Cannot Create Capability

**The assumption or constraint.** The paper's entire framework — both the search-based approach against the PRM and the revision-based approach modifying the proposal distribution — operates under the assumption that the base model already possesses the capability to produce correct solutions for a given problem at some non-trivial rate. If the base model's pass@1 on a problem is zero or near-zero, test-time compute cannot help: there are no correct solutions in the proposal distribution to find (via search) or refine (via revisions). The paper is explicit about this:

> "on the hardest questions (bin 5), no method makes meaningful progress — the base model simply lacks the capability to produce correct solutions regardless of how the budget is allocated"

**The consequence.** This limitation creates a hard ceiling on what test-time compute can achieve, and it is a ceiling that no amount of scaling the inference budget can break through. For problem distributions where a substantial fraction of questions are in the hardest difficulty bin (where the base model's pass@1 is near zero), the compute-optimal framework provides no benefit — the policy will correctly identify these problems as hard but will be unable to solve them regardless of strategy selection.

This is not a minor edge case. The paper's own difficulty binning shows that approximately 20% of the MATH test set falls into bin 5 (the hardest quintile), and on these problems, accuracy remains at 1–3% regardless of method or budget (Figure 3, right; Figure 7, right). This means that even with the compute-optimal strategy, one-fifth of the benchmark is essentially unsolvable. In practical terms, this limits the approach to problems within the base model's "capability frontier" — the set of problems for which the model can produce at least occasional correct solutions. For problems outside this frontier, pretraining a larger model (or training on more diverse data) remains the only viable path.

The FLOPs-matched comparison (Section 7) makes this boundary explicit: on bin 5 problems, the 14× larger pretrained model substantially outperforms test-time compute with the smaller model (e.g., -52.9% relative disadvantage on hard problems at R ≫ 1 for PRM search). This means that for genuinely hard problems — those requiring capabilities the base model does not possess — scaling pretraining is unambiguously superior, and test-time compute cannot substitute.

**What evidence exists in the paper.** The difficulty-bin breakdowns consistently show near-zero performance on bin 5 across all methods. In Figure 3 (right), bin 5 accuracy is 1–3% for both beam search and best-of-N at all budgets from 4 to 256 generations. In Figure 7 (right), bin 5 accuracy is approximately 2–3% regardless of the sequential-to-parallel ratio at a budget of 128 generations. In Figure 9, the bin 5 scaling line (blue, bottommost) is essentially flat near 0–5% for all test-time compute budgets, while the 14× larger model's greedy performance (star) sits above this line for all values of R. The paper also provides qualitative examples in Appendix M showing degenerate search outputs, reinforcing that aggressive optimization cannot compensate for missing capability.

**Mitigation status.** The paper is transparent about this limitation in the Section 7 takeaway box and does not attempt to solve it. No method is proposed for extending the capability frontier through test-time compute — the paper's contribution is optimizing within the frontier, not expanding it. This is a fundamental constraint, not an oversight, and the paper correctly identifies it as such. However, practitioners evaluating whether to adopt this approach must understand that its benefits are bounded by base model capability, and that problems requiring capabilities beyond the base model's training distribution will see no improvement regardless of inference budget.

---

### The Thinking-Fusion Pipeline Is Evaluated as a Monolithic Block Without Stage-Level Ablations

**The assumption or constraint.** The thinking-fusion SFT–RL pipeline consists of four sequential stages: LongCoT SFT, verifiable-reward RL, Think-Fusion SFT, and mixed-reward RL. The paper presents the full pipeline as a package and evaluates the final model (SAIL-VL2-Thinking variants) against baselines in Table 10. However, it does not report the marginal contribution of each stage — what does LongCoT SFT alone achieve? What does adding verifiable-reward RL contribute? What does Think-Fusion SFT add beyond the RL stage? What does the mixed-reward stage add beyond verifiable-reward RL?

**The consequence.** Without stage-level ablations, a practitioner cannot determine which components of the pipeline are essential and which are optional. The full four-stage pipeline is computationally expensive: it requires curating 400K LongCoT samples, 70K STEM RL samples, 1M Think-Fusion samples, and 100K mixed-reward samples, plus running two separate RL training phases. If, for example, the mixed-reward RL stage provides only marginal improvement over verifiable-reward RL, a practitioner with limited compute might reasonably skip it. Conversely, if Think-Fusion SFT is critical for preventing reasoning mode collapse (the paper's stated motivation), a practitioner who omits it may produce a model that performs well on reasoning benchmarks but generates verbose CoT for every query, making it unusable in latency-sensitive applications.

The paper claims that Think-Fusion SFT prevents mode collapse: "it learns the logical pathways to the correct answer without needing to explicitly write out each step. This allows it to provide accurate direct answers even when a user does not specifically prompt it for a step-by-step thinking process." This is an important claim about a specific behavior (selective reasoning) that cannot be verified without an ablation comparing a model trained with Think-Fusion SFT against one trained with only LongCoT SFT + RL, evaluated on both reasoning benchmarks (where CoT is appropriate) and standard VQA benchmarks (where direct answers are appropriate). The paper does not report standard VQA performance for the thinking variants at all, so there is no evidence that the base model's general capabilities are preserved.

**What evidence exists in the paper.** The paper does not provide any ablation of the thinking-fusion pipeline stages. Table 10 reports only final thinking model performance on reasoning benchmarks. The base (non-thinking) model performance on general understanding benchmarks is reported in Tables 8 and 9, but no corresponding table exists for the thinking variants on these same benchmarks, making it impossible to compare thinking vs. non-thinking performance on standard tasks. The paper's statement about the model "internalizing reasoning capabilities" is an interpretation of behavior, not a measured outcome supported by comparative experiments.

**Mitigation status.** Not addressed. The paper presents the pipeline as a complete recipe without analyzing component contributions. This is a significant gap for a paper whose primary contribution is the training methodology — readers cannot determine which innovations within the pipeline are responsible for the observed performance gains. Future work that isolates each stage's contribution would substantially increase the practical value of the thinking-fusion approach by identifying the minimal effective training recipe.

---

### SAIL-ViT Alignment Metrics Are Measured on an Implausibly Small Sample

**The assumption or constraint.** The paper's claim that SAIL-ViT closes the modality gap between vision and language representations rests on quantitative evidence in Table 7 and Figure 6, which measure distribution distances (nearest neighbor distance, Wasserstein distance, mean overall distance) between visual features extracted by SAIL-ViT (and baseline AIMv2) and text embeddings from various LLMs. The paper states:

> "We randomly sampled five images from the internet, extracted their visual features using both SAIL-ViT and its baseline, and concatenated them to form a feature collection of size [5120, 1024]."

The distance metrics are computed on features from **five images**. The concatenated feature collection of 5,120 tokens represents 1,024 tokens per image from five images. Five images drawn from an unspecified internet source is not a statistically meaningful sample for drawing conclusions about distribution-level properties of a vision encoder's output space.

**The consequence.** The distance metrics reported in Table 7 — showing SAIL-ViT consistently achieving lower distances than AIMv2 across all three metrics and all four LLM architectures — may reflect properties of the particular five images chosen rather than genuine distribution-level improvements in visual-linguistic alignment. The consistency across LLM architectures partially mitigates this concern (if the improvement were noise, it would likely not replicate across four different LLMs), but five images are simply too few to establish that the observed differences are statistically reliable.

The stakes of this limitation are moderate rather than catastrophic: the downstream benchmark results (Tables 8 and 9) provide independent evidence that SAIL-ViT contributes to improved multimodal understanding, and the zero-shot image classification results (Table 6) show improved visual representation quality. The alignment distance metrics are supplementary evidence, not the primary claim. However, the paper presents them as direct quantitative validation of the progressive training strategy's mechanism — that it "closes the gap between visual and textual feature spaces" — and the evidence supporting this specific mechanistic claim is weak.

The visualization in Figure 6 shows 2D PCA projections of the feature distributions, with SAIL-ViT features appearing more compact and overlapping more with text embeddings than AIMv2 features. PCA projections from five images onto two dimensions are highly sensitive to sampling — different random images would produce different projections, and the apparent overlap patterns could change substantially.

**What evidence exists in the paper.** The paper provides the distance metric table (Table 7) and PCA visualizations (Figure 6) as the sole quantitative evidence for the modality gap reduction claim. The sample size of five images is explicitly stated in the text. No confidence intervals, standard deviations, or statistical tests are reported. No analysis of how the distance metrics vary across different random samples of images is provided. The downstream benchmark results (Section 6.2.2) show SAIL-VL2 models with SAIL-ViT outperforming baselines, but these results are confounded by all other differences between SAIL-VL2 and comparison models (different training data, different LLM backbones, different training recipes), so they do not isolate SAIL-ViT's contribution.

**Mitigation status.** Not addressed. The paper does not acknowledge the small sample size as a limitation of the alignment analysis. A straightforward improvement would be to compute the distance metrics on a larger sample (e.g., 1,000 images from a standard validation set like ImageNet-val or COCO-val) and report confidence intervals. This would require modest additional computation (extracting features for 1,000 images is fast for a pre-trained ViT) and would substantially strengthen the mechanistic evidence.

---

### The 14× Larger Pretrained Model Baseline Is Weakened by Non-Compute-Optimal Training

**The assumption or constraint.** The FLOPs-matched comparison in Section 7 compares PaLM 2-S* with compute-optimal test-time scaling against a model with approximately 14× more parameters that uses greedy decoding and no test-time compute augmentation. The paper scales model parameters while holding training data fixed, departing from compute-optimal pretraining (where both data and parameters would be scaled). The paper acknowledges this:

> "We choose this setting as it is representative of a canonical approach to scaling pretraining compute and leave the analysis of compute-optimal scaling of pretraining compute where the data and parameters are both scaled equally to future work."

The comparison model uses only greedy decoding — no majority voting, no best-of-N sampling, no verifier-based selection. The larger model receives none of the inference-time optimization that the smaller model benefits from.

**The consequence.** This design choice systematically favors test-time compute in two ways. First, a Chinchilla-optimal model trained with 14× more total FLOPs — scaling both parameters and data according to Hoffmann et al. (2022) — would likely outperform a parameter-only-scaled model, making the pretraining baseline weaker than it could be. The paper's reported advantages of test-time compute (e.g., +27.8% on easy questions at R ≪ 1) would likely shrink against a properly compute-optimal larger model.

Second, giving the larger model even a modest test-time compute budget — say, best-of-8 or best-of-16 with the same PRM — would create a much stronger baseline. The paper's framing ("test-time compute with a smaller model vs. pretraining a larger model") implies a tradeoff between two modes of spending compute, but the comparison actually pits a comprehensively optimized small model (with difficulty-aware strategy selection, PRM search or revision chains, and aggressive test-time compute) against a minimally optimized large model (greedy single-pass decoding). This is not a fair comparison between inference compute and pretraining compute — it is a comparison between a highly optimized system and a baseline system, and the optimization, not the underlying compute allocation philosophy, may drive the observed differences.

**What evidence exists in the paper.** The paper explicitly describes the baseline in Section 7: the 14× larger model is used with greedy decoding and no extra test-time compute. The bar charts in Figure 1 and the line plots in Figure 9 show the comparison results. No ablation is provided that gives the larger model any test-time compute budget, which would be the natural robustness check. The paper's acknowledgment of the non-compute-optimal pretraining (quoted above) is honest but does not quantify how much this weakens the baseline.

**Mitigation status.** Partially addressed through transparency — the paper states the limitation clearly. However, the limitation is not mitigated experimentally. A fairer comparison would include at minimum: (a) a Chinchilla-optimal larger model (scaling both parameters and data), and (b) giving the larger model a small test-time compute budget (best-of-8 or best-of-16) to see whether test-time compute with the smaller model still outperforms. Without these comparisons, the paper's claim that "test-time compute can substitute for pretraining" should be understood as an upper bound on the substitution effect, valid under the specific (and favorable-to-test-time-compute) comparison conditions used.

---

### Single Benchmark, Single Model Family — No Evidence of Cross-Domain or Cross-Architecture Generalization

**The assumption or constraint.** All experiments in the paper use the MATH benchmark (500 test questions) with PaLM 2-S* as the base model. The paper states that it "believes this model is representative of the capabilities of many contemporary LLMs" (Section 4), but provides no evidence for this claim. The experimental findings — about difficulty-dependent scaling behavior, about verifier over-optimization thresholds, about the optimal sequential-to-parallel ratio, about the 4× efficiency gain — are demonstrated on exactly one task domain (competition-level mathematical reasoning) with exactly one model family.

**The consequence.** A practitioner cannot determine whether the paper's central findings — particularly the difficulty-conditioned compute-optimal policy and the 4× efficiency gain — generalize to other domains or model families. Several aspects of the results could be specific to MATH and PaLM 2-S*:

- **Task domain specificity.** Math problems have clean, verifiable answers and structured multi-step solutions, making PRM training via Monte Carlo rollouts straightforward (correctness is deterministically checkable). For domains without such clean correctness signals — code generation (where solutions compile but may be logically wrong), open-ended QA, creative writing, dialogue — the PRM training pipeline would need fundamental modification, and it is unclear whether difficulty-conditioned allocation would show similar patterns.

- **PRM quality dependence.** The verifier over-optimization thresholds observed in Figure 3 (beam search degrading on easy problems at high budgets) depend on the PRM's calibration. A PRM trained on a different model's outputs or on a different task distribution might exhibit different over-optimization behavior, shifting the difficulty thresholds at which beam search becomes harmful vs. helpful. The compute-optimal policy is contingent on the specific PRM quality achieved with PaLM 2-S* on MATH.

- **Revision model transferability.** The revision model's ability to learn from incorrect in-context examples depends on the base model's in-context learning capabilities. PaLM 2-S* has specific in-context learning properties that may not generalize to other model families (LLaMA-based models, Qwen-based models). The edit-distance-based pairing strategy for revision training data (selecting incorrect answers close to correct ones) may be more or less effective depending on the base model's typical error patterns.

- **MATH-specific difficulty distribution.** The paper bins questions into difficulty quintiles based on the base model's pass@1 rate. The distribution of difficulty across quintiles — and therefore the optimal allocation policy — is specific to MATH's problem characteristics. A different benchmark with different difficulty characteristics (e.g., more problems clustered at medium difficulty, fewer at extreme difficulty) would yield a different policy.

**What evidence exists in the paper.** All experiments in Sections 5–7 are on MATH with PaLM 2-S*. The paper does not report results on any other benchmark (GSM8K, MMLU, HumanEval, etc.) or with any other base model (LLaMA, Qwen, etc.). The claim that PaLM 2-S* is "representative" is an assertion, not a supported claim. The fact that the paper's own PRM training encountered distribution shift issues with the PRM800k dataset (Section 5.1: "largely ineffective for their PaLM 2 models, likely due to distribution shift between GPT-4 and PaLM 2 outputs") ironically demonstrates that findings do not transfer cleanly across model families — the very lesson from that result should caution against assuming PaLM 2-S*-specific findings are universal.

**Mitigation status.** The paper acknowledges the single-benchmark limitation implicitly through scope but does not address it as a limitation. Section 8 suggests extending to other domains as future work ("code generation, logical reasoning, scientific QA") but provides no evidence that the current findings transfer. For practitioners, this means the paper's methodology should be viewed as a promising direction requiring domain-specific validation, not as a turnkey solution that can be applied to any reasoning task without modification.

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
