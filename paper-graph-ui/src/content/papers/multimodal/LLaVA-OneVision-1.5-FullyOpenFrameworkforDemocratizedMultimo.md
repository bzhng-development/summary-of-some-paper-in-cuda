# LLaVA-OneVision-1.5: Fully Open Framework for Democratized Multimodal Training

**ArXiv:** [2509.23661](https://arxiv.org/abs/2509.23661)

## 🎯 Pitch

LLaVA-OneVision-1.5 delivers a fully open-source, cost-efficient framework for training large multimodal models (LMMs) entirely from scratch—combining a state-of-the-art, region-aware vision encoder (RICE-ViT), vast curated datasets, and highly optimized training pipelines. This breakthrough enables researchers and practitioners to build high-performing vision-language models without proprietary data, expensive infrastructure, or opaque methods, democratizing access to cutting-edge multimodal AI and accelerating advances across diverse real-world tasks.

---

## 1. Executive Summary

This paper introduces **LLaVA-OneVision-1.5**, a family of fully open-source Large Multimodal Models that achieves state-of-the-art performance under a strict $16,000 compute budget by scaling mid-training data and curating instruction data, rather than relying on complex training paradigms. Built on a RICE-ViT vision encoder for region-aware visual semantics and Qwen3 language backbones, the models undergo a three-stage pipeline — language-image alignment, high-quality knowledge learning (an 85M concept-balanced dataset), and visual instruction tuning (a 22M curated dataset) — with optional RL post-training to elicit chain-of-thought reasoning. The 8B variant outperforms Qwen2.5-VL-7B on 18 of 27 benchmarks, the 4B variant surpasses Qwen2.5-VL-3B on all 27 benchmarks, and RL post-training yields gains of +7.9 on WeMath and +10.5 on MMMU-Pro-vision in thinking mode, establishing that a compute-constrained open framework can match or exceed proprietary-scale models only when the mid-training data is concept-balanced and the instruction-tuning corpus is aggressively scaled.

## 2. Context and Motivation

### The Core Problem: Building High-Performance Multimodal Models Is Expensive and Opaque

The fundamental challenge this paper addresses is not whether large multimodal models (LMMs) can achieve impressive performance — they demonstrably can — but rather **how to build them from scratch when you lack the resources of a major industrial lab**. The most capable multimodal systems as of the paper's writing (Gemini 2.5, GPT-4V) remain proprietary, with their training data, source code, and detailed recipes hidden from the broader research community. This opacity creates a specific and consequential problem: researchers outside of well-funded organizations cannot understand *how* these models are constructed, cannot reproduce their results, and cannot iterate on them to produce new innovations.

The paper frames this as a **democratization gap**. It is not simply that proprietary models outperform open ones — it is that the community lacks a verifiable blueprint for building competitive LMMs under realistic resource constraints. Without such a blueprint, progress is bottlenecked to a small number of actors, and the scientific understanding of what architectural choices, data strategies, and training procedures actually matter remains fragmented and anecdotal.

This gap matters for several overlapping reasons the authors articulate throughout Section 1:

- **Scientific reproducibility**: When training data and code are proprietary, claims about model capabilities cannot be independently verified, and the field cannot build cumulative knowledge about which design decisions drive performance improvements. A finding that a particular model excels at OCR might be due to a novel vision encoder, a carefully curated dataset, or simply a larger training budget — and without openness, there is no way to disentangle these factors.
- **Resource-constrained innovation**: Academic labs and smaller companies cannot afford the million-dollar training budgets that the largest models require. If the only path to state-of-the-art performance is massive pretraining spend, then entire segments of the research community are structurally excluded from contributing to LMM development. This paper explicitly targets a constraint — $16,000 total compute — to show that ingenuity in data curation and training strategy can partially substitute for raw financial scale.
- **Specialized applications**: Proprietary models are general-purpose. An open, reproducible pipeline allows practitioners to adapt models to specific domains (medical imaging, legal document analysis, scientific figure interpretation) by retraining or fine-tuning with domain-specific data, something a closed API cannot support.
- **Safety and auditability**: Understanding the training data composition and training procedure is a prerequisite for auditing models for bias, factual accuracy, and safety properties. Proprietary models offer no such transparency.

### The Widening Performance Gap Between Open and Proprietary Models

The paper situates itself within a specific historical trajectory of open-source multimodal efforts, each of which pushed the frontier but ultimately fell behind. Understanding this progression is essential to understanding why LLaVA-OneVision-1.5 is not simply "another open model" but a deliberate attempt to close a gap that has been growing despite prior efforts.

**Early LLaVA releases (LLaVA, LLaVA-NeXT, LLaVA-OneVision).** The LLaVA series, pioneered by Liu et al. (2023, 2024a), established a foundational contribution: the release of fully open training data and code. LLaVA demonstrated that a vision encoder (CLIP) connected to a language model (Vicuna) via a simple projection layer, trained on GPT-4-generated multimodal instruction data, could produce a competent visual assistant. Subsequent iterations — LLaVA-NeXT (Liu et al., 2024b), which introduced higher-resolution input handling, and LLaVA-OneVision (Li et al., 2025a), which unified single-image, multi-image, and video understanding — progressively improved capabilities while maintaining openness. However, the paper acknowledges candidly that "their performance now falls substantially behind that of current state-of-the-art models." The gap is not a minor lag — it represents a qualitative difference in what the models can do, particularly on difficult reasoning tasks.

**Why the early LLaVA models fell behind.** The paper does not extensively diagnose this, but the implicit reasoning is clear: as the frontier advanced, merely being open was insufficient. The proprietary models benefited from (a) vastly larger and more carefully curated pretraining datasets, (b) more capable vision encoders trained with larger budgets and better objectives, (c) stronger language backbones, and (d) more sophisticated multi-stage training pipelines. Early LLaVA models, while pioneering openness, used relatively simple recipes — CLIP-based vision encoders, modest-scale instruction data, and straightforward training schedules — that did not scale with the frontier.

**Molmo and PixMo: the first credible attempt to close the gap.** Deitke et al. (2025) represent a significant inflection point. Molmo released not just model weights but also datasets and source code, enabling the community to train LMMs from scratch with a reproducible pipeline. The paper highlights that "through careful architectural choices, a refined training pipeline, and high-quality data, Molmo achieves near-parity with GPT-4V on both academic benchmarks and user preference evaluations." This was the first demonstration that an open model could approach proprietary performance, and it validated the hypothesis that data quality and training strategy could partially substitute for raw scale. However, Molmo's achievement came at substantial computational cost — it was not a budget-constrained exercise — and its performance, while competitive at the time, would quickly be surpassed as the proprietary frontier continued advancing.

**Open-Qwen2VL: pushing efficiency further.** Wang et al. (2025b) introduced a 2B-parameter model pretrained on only 0.36% of the 1.4T multimodal tokens used in Qwen2-VL, yet outperforming Qwen2-VL-2B across various benchmarks. This was an important proof point: it suggested that much of the data used in large-scale pretraining might be redundant, and that smarter data selection could dramatically reduce training costs without sacrificing performance. The paper implicitly builds on this insight — if 0.36% of the data can work, then careful curation is the key lever, not raw volume — but extends it to larger model scales (4B, 8B) and a more comprehensive set of benchmarks.

**Where these prior efforts collectively fall short.** Despite the progress from Molmo and Open-Qwen2VL, the paper identifies persistent limitations that motivate LLaVA-OneVision-1.5:

- **Computational demand remains prohibitive.** Even open models that achieve strong performance typically require substantial GPU budgets that are out of reach for many academic labs. The paper cites a $16,000 budget as its constraint, which is explicitly positioned as affordable for a broader community. Prior open models either did not report or did not optimize for such tight budgets.

- **Training data quality and curation are underexplored.** While Molmo demonstrated the importance of high-quality data, the specific strategies for ensuring balanced concept coverage, cross-lingual alignment, and effective instruction mixing were not fully systematized. The paper argues that the composition and curation of both pretraining and instruction-tuning data — not just their scale — are the critical determinants of performance under budget constraints.

- **Vision encoder choice has been stagnant.** Most open models, including early LLaVA releases, used CLIP (Radford et al., 2021) or SigLIP (Zhai et al., 2023) as their vision encoder. These models are trained with global image-text contrastive objectives that fail to capture local region-level semantics, limiting fine-grained capabilities like OCR, grounding, and detailed visual reasoning. The paper identifies this as a key architectural bottleneck: global alignment produces representations that are good for coarse image understanding but poor for tasks requiring precise spatial or textual information.

- **No unified framework exists.** The community had individual success stories (Molmo, Open-Qwen2VL) but lacked a complete, documented, and reproducible pipeline covering data curation, efficient training, and post-training optimization — all operating under a transparent budget constraint. Researchers wanting to build their own LMMs had to piece together disparate techniques with no guarantee of synergistic performance.

### The Specific Gap: Mid-Training as an Underexplored Lever

The paper makes a specific conceptual contribution to the training pipeline taxonomy. Standard LMM training typically involves two stages: (1) pretraining the vision-language connector (also called the projector) to align visual features with the language model's embedding space, and (2) instruction tuning to teach the model to follow diverse visual instructions. The paper introduces a **Stage-1.5: High-Quality Knowledge Learning** (or "mid-training") that sits between these two stages.

This is not merely a relabeling of existing practice. The insight is that after the projector is aligned (Stage-1), the model can process visual inputs but lacks *multimodal world knowledge* — it has not seen enough diverse image-text pairs to build robust associations between visual concepts and linguistic descriptions. Stage-2 (instruction tuning) teaches the model to follow instruction formats, but the underlying knowledge must already be present; instruction tuning is primarily about eliciting existing knowledge in the desired output format, not about injecting new knowledge. The mid-training stage therefore serves as a **critical knowledge injection phase** that fills the gap between basic alignment and instruction-following capability.

The paper's key finding with respect to mid-training is that **simply scaling the volume of mid-training data, with careful concept balancing, is sufficient to produce state-of-the-art LMMs** — eliminating the need for complex training paradigms that previous work had explored (e.g., multi-stage curriculum learning, iterative data refinement, or architectural modifications). This is both a methodological contribution (a specific recipe) and a philosophical one (a bet that data curation beats algorithmic complexity under budget constraints).

### Where Vision Encoders Fall Short: The Case for Region-Aware Representations

The paper devotes significant attention (Section 2.2) to the limitations of standard vision encoders, and this analysis is central to understanding why LLaVA-OneVision-1.5's choice of RICE-ViT is consequential rather than incidental.

**The global alignment problem.** Vision encoders like CLIP and SigLIP are trained with instance-level contrastive objectives: an entire image is matched against its corresponding caption, with all other images in the batch treated as negatives. This produces a single global embedding per image that captures high-level semantic content — is this a dog, a car, a document? — but **discards all spatial information below the global level**. The paper explains: "they fail to capture the similarity structure of training data or the local region-level semantics within images."

Why does this matter concretely? Consider an OCR task where the model must read text in a specific region of a document. A CLIP embedding represents the entire document as one vector, collapsing spatial distinctions. The model can learn that "this image contains text" but cannot easily associate specific text strings with specific locations. Similarly, for grounding tasks where the model must identify an object described in the prompt, a global embedding provides no spatial anchors. The LLM must rely on positional information injected through other means (e.g., tiling the image into patches), which is a workaround rather than a solution.

**The false negative problem.** The paper identifies a subtler issue: "instance-wise contrastive learning... treats all instances as negatives regardless of their semantic similarity." In CLIP-style training, if a batch contains two images of different dogs, one is treated as a negative for the other even though they are semantically similar. This pushes the model to represent all images as far apart as possible in embedding space, destroying the similarity structure that could be useful for fine-grained discrimination. A better objective would recognize that these images are semantically related while still being distinct — which is precisely what cluster discrimination approaches aim to achieve.

**Multi-objective complexity in SigLIP2.** The paper provides a specific comparison point: SigLIP2 (Tschannen et al., 2025), a state-of-the-art vision encoder, "depends on multiple specialized losses (SILC, TIPS, LocCa, and Sigmoid)" to achieve its strong performance. Each loss targets a different capability: SigLIP's original sigmoid loss for global alignment, SILC for local-global consistency, TIPS for text-image positional supervision, and LocCa for localization-aware captioning. This multi-loss approach works but introduces architectural complexity, training instability risks, and hyperparameter sensitivity — and critically, it still does not fundamentally solve the region-level representation problem, instead patching it with auxiliary objectives.

**How RICE-ViT addresses these limitations.** The paper positions RICE-ViT (Xie et al., 2025) as a cleaner solution that unifies global understanding, OCR, and localization under a single **region cluster discrimination loss**. Rather than treating images as atomic units, RICE-ViT operates on candidate regions within images — trained on 450M images and 2.4B candidate regions — learning to discriminate whether two regions belong to the same semantic cluster (e.g., both contain a dog face, both contain a text string, both contain a car wheel). This objective naturally produces representations that are:
- **Region-aware**: individual spatial locations have meaningful embeddings that capture local semantics.
- **Similarity-preserving**: semantically related regions (even from different images) are close in embedding space, enabling the downstream LLM to reason about visual similarity.
- **Resolution-flexible**: RICE-ViT uses 2D rotary positional encoding (RoPE), which naturally supports variable input resolutions without requiring resolution-specific fine-tuning — a practical advantage over models like Qwen2-VL and InternVL 2.5 that require explicit architectural modifications for different resolutions.

The paper also introduces a design choice that initially seems counterintuitive but reflects deliberate reasoning about multimodal alignment: during pretraining, RICE-ViT jointly models both **object regions** and **OCR regions** (Figure 2). This means the same encoder learns to represent both semantic objects (dogs, cars, faces) and text-bearing regions (words, paragraphs, document sections) in a shared embedding space. This is a specific architectural commitment: it assumes that the visual features useful for recognizing text (stroke patterns, spacing, font characteristics) and the features useful for recognizing objects (shape, texture, context) can coexist in a single representation without destructive interference. The paper's empirical results validate this assumption, but it is a non-trivial design choice — alternative approaches might use separate encoders for visual and textual regions, at the cost of increased model complexity.

### How This Paper Positions Itself

The paper positions LLaVA-OneVision-1.5 not as an incremental improvement within the LLaVA series but as a **systematic response to a specific set of identified failures in the open LMM ecosystem**. The positioning can be understood along four axes:

**1. Against proprietary models: performance parity through openness.** The paper does not claim to beat the largest proprietary models (Gemini 2.5, GPT-4V) on all metrics. Instead, it claims that open models — built with transparency and under budget constraints — can achieve *competitive* performance against similarly-sized proprietary-adjacent models (specifically Qwen2.5-VL) across a broad benchmark suite. The comparison against Qwen2.5-VL is strategically chosen: Qwen2.5-VL is itself open-weight but not fully open (training data and code are partially released), making it a relevant and strong baseline that represents the current frontier for accessible multimodal models.

**2. Against prior open models: a complete, budget-constrained recipe.** The paper differentiates itself from Molmo, Open-Qwen2VL, and earlier LLaVA releases by committing to a fully documented pipeline that includes: (a) the exact datasets with download links, (b) the training framework with efficiency optimizations, (c) the model checkpoints at multiple scales, and (d) an explicit budget constraint ($16,000) that serves as a proof of accessibility. No prior work, the paper argues, provided all four elements simultaneously.

**3. Against complex training paradigms: data scaling over algorithmic complexity.** The paper makes a deliberate philosophical claim: "simply scaling data at the mid-training stage alone can produce state-of-the-art LMMs, eliminating the need for complex training paradigms." This is an argument against the trend toward increasingly elaborate training recipes (multi-stage curricula, iterative self-improvement loops, architectural modifications for specific capabilities). The paper's bet is that careful data curation — concept balancing, cross-lingual coverage, diverse instruction mixing — yields better returns than additional algorithmic complexity when operating under a fixed budget. This is a falsifiable claim, and the ablation studies (Section 6.7) are designed to support it.

**4. As a foundation for community-driven specialization.** The paper explicitly frames LLaVA-OneVision-1.5 as a "foundational resource that empowers the community to build specialized applications." This is not merely aspirational language — it reflects a strategic decision to release all assets under permissive licenses (the code is on GitHub, the models on HuggingFace, the datasets with direct links) specifically to enable downstream fine-tuning, domain adaptation, and architectural experimentation. The paper envisions a model that is not an end product but a starting point that the community can fork, modify, and improve.

### The Compute Budget Constraint as a Research Contribution

The $16,000 compute budget is not incidental to the paper's contribution — it is a central part of the research framing. The paper treats the budget as a **hard constraint that forces design tradeoffs**, analogous to how the Chinchilla scaling laws (Hoffmann et al., 2022) treated total FLOPs as a constraint that forced principled allocation between model size and data volume.

By stating the budget explicitly, the paper enables:
- **Reproducibility**: Another lab with $16,000 of GPU credit can verify the results exactly.
- **Calibration**: The community can assess whether future improvements come from better methods or simply larger budgets — a crucial distinction that is often muddied when budgets are unreported.
- **Constraint-driven innovation**: Specific design choices in the paper — offline data packing, hash-bucket-based batching, hybrid parallelism — are direct responses to the budget constraint. Without the constraint, simpler (but more expensive) approaches might have been chosen.

The paper reports that training the 8B model on 85 million mid-training samples at native resolution took 128 A800 GPUs over 3.7 days (Section 4.2). At typical cloud GPU pricing, this is roughly consistent with the stated $16,000 figure, though the exact cost depends on the provider. The important point is that this is an order of magnitude cheaper than what is typically required to train competitive LMMs from scratch — and the paper's technical contributions (data packing, efficient infrastructure) are what make this possible.

## 3. Technical Approach

### 3.1 Reader Orientation

LLaVA-OneVision-1.5 is a production pipeline for building vision-language models from scratch — a recipe, not just a model — that takes raw image-text pairs and instruction data as input and produces a multimodal assistant capable of answering questions about images, reading documents, solving math problems, and grounding objects in space. The system solves the problem of building competitive multimodal models under a strict $16,000 compute budget by substituting careful data curation (concept-balanced pretraining, aggressive instruction scaling) for algorithmic complexity, and by choosing a vision encoder (RICE-ViT) that provides region-level representations natively rather than relying on workarounds like image tiling.

### 3.2 Big-Picture Architecture (Diagram in Words)

The system is a multi-stage training pipeline with six major components, each producing an intermediate artifact consumed by the next stage:

1. **Data Ingestion and Concept Balancing** — Takes raw image sources (COYO-700M, Obelics, DataComp-1B, SAM-1B, ImageNet-21K, LAION-CN, MINT, Zero250M) and produces an 85M-image dataset with balanced concept coverage by assigning images to a 500K-concept vocabulary via embedding similarity, then sampling inversely proportional to concept frequency. This step is critical for preventing the long-tail distribution in raw web data from starving the model of rare concepts.

2. **Captioning and Filtering** — A powerful captioner generates English and Chinese captions for the balanced images, and a validity filter removes duplicates and excessively long outputs. The output is LLaVA-OneVision-1.5-Mid-Traning: 65M English and 20M Chinese image-text pairs.

3. **Vision Encoder (RICE-ViT)** — A ViT-L-14 pretrained on 450M images with 2.4B candidate regions using a unified region cluster discrimination loss, producing spatially-aware visual features with 2D RoPE for native resolution support. Unlike CLIP/SigLIP, this encoder preserves local region semantics rather than collapsing everything to a global embedding.

4. **Projector (2-layer MLP with spatial grouping)** — Groups adjacent sets of four patch features spatially, concatenates them, and projects them through a two-layer MLP into the LLM's text embedding space, bridging the modality gap between visual tokens and language tokens.

5. **Language Model (Qwen3)** — The reasoning core that receives the sequence of projected visual tokens interleaved with text tokens and autoregressively generates responses. The 4B variant uses Qwen3-4B, the 8B variant uses Qwen3-8B.

6. **RL Post-Training Module (AReaL GRPO)** — An optional asynchronous reinforcement learning stage that uses outcome-based rewards on 67K curated instances to elicit chain-of-thought reasoning, with a two-stage curriculum (answer-only warmup → chain-of-thought reasoning with interleaved anchor tasks).

**Information flow in the standard pipeline:** Raw image sources → concept balancing → captioning → [85M balanced image-text pairs] → Stage-1.5 full-parameter training with LLaVA-558K-aligned projector → [LLaVA-OneVision-1.5-Base model] → Stage-2 full-parameter training with 22M instruction samples + FineVision → [LLaVA-OneVision-1.5-Instruct model] → (optional) RL post-training with 67K discrepancy-selected instances → [LLaVA-OV-1.5-RL model].

The three training stages are: **Stage-1** (projector-only on 558K LLaVA-1.5 data, aligning visual features to LLM space), **Stage-1.5** (full-parameter on 85M mid-training data, injecting multimodal world knowledge), and **Stage-2** (full-parameter on 22M instruction data, teaching instruction following and task-specific behaviors).

### 3.3 Roadmap for the Deep Dive

- **First**, the concept-balanced mid-training dataset — how raw images are assigned to a 500K-concept vocabulary, how inverse-frequency sampling flattens the concept distribution, and why this matters for downstream performance — because this is the paper's central data contribution and the foundation on which all training stages depend.
- **Second**, the instruction dataset construction — the seven-category taxonomy, the 124 data sources, and the merging strategy with FineVision — because this determines what tasks the final model can perform and how balanced its capabilities are.
- **Third**, the vision encoder (RICE-ViT) in detail — the region cluster discrimination loss, the 2D RoPE mechanism, the joint object-OCR region modeling, and how these differ from CLIP/SigLIP alternatives — because the vision encoder choice is the primary architectural differentiator from prior open LMMs.
- **Fourth**, the training pipeline and infrastructure — the three stages, the data packing optimization (hash buckets, 11× compression), the hybrid parallelism framework (AIAK-Training-LLM on Megatron-LM, 128 A800 GPUs, 3.7 days), and the specific hyperparameter configurations — because the $16,000 budget constraint forces specific engineering decisions that would be unnecessary with unlimited compute.
- **Fifth**, the RL post-training system — the discrepancy-driven data selection (Pass@N vs. Pass@1 gap), the reward system (type-specific verification rules, symbolic equivalence for math), the GRPO algorithm with asynchronous AReaL infrastructure, and the two-stage curriculum design — because RL is a separable but important component that unlocks reasoning capabilities latent in the supervised model.
- **Sixth**, the training of the vision encoder itself — the cluster discrimination loss formulation, the 450M image / 2.4B region scale, and the comparison with multi-loss SigLIP2 — because understanding why RICE-ViT works requires understanding what it was trained to do.

### 3.4 Detailed, Sentence-Based Technical Breakdown

This is primarily a **systems and data engineering paper** whose core idea is that carefully curated data (concept-balanced pretraining, aggressively scaled instruction tuning) combined with a region-aware vision encoder can produce state-of-the-art multimodal models under a strict $16,000 compute budget, without requiring novel architectures or complex training paradigms.

---

#### Concept-Balanced Mid-Training Dataset (85M Image-Text Pairs)

The mid-training dataset is the paper's most significant data contribution and the engine of its performance. The problem it solves is specific: raw web-scale image datasets have extremely skewed concept distributions (long-tail bias), meaning common concepts like "dog," "car," and "person" appear millions of times while rare concepts like "abacus," "zygote," or "isosceles triangle" appear a handful of times. A model trained on such data will excel at recognizing common concepts but fail on rare ones — and importantly, it will **waste most of its training budget on redundant examples of already-mastered concepts** while starving on the long tail.

The solution is a concept-balancing pipeline that operates in four stages: embedding projection, top-K concept assignment, inverse-frequency weighting, and caption generation.

**Stage 1: Embedding-based concept matching.** The paper collects raw images from eight sources: COYO-700M (Byeon et al., 2022), Obelics (Laurençon et al., 2023), DataComp-1B (Gadre et al., 2023), LAION-CN (Zhang et al., 2022), ImageNet-21K (Russakovsky et al., 2015), SAM-1B (Kirillov et al., 2023), MINT (Wang et al., 2024c), and Zero250M (Xie et al., 2023). These sources collectively contain diverse image types but have highly variable caption quality — some have detailed captions (Obelics), some have brief or noisy captions (COYO-700M), and some have no captions at all (SAM-1B, ImageNet-21K). This heterogeneity makes traditional caption-based concept matching (as in MetaCLIP) unreliable, because poor captions lead to poor concept assignments.

Instead, the paper adopts a **feature-based matching approach** that operates purely in embedding space, bypassing caption quality entirely. Given:
- An image set `$\mathcal{I} = \{i_0, i_1, \ldots, i_N\}$` (all images from the eight sources)
- A concept vocabulary set `$\mathcal{V} = \{v_0, v_1, \ldots, v_M\}$` consisting of MetaCLIP's 500,000 concept entries (a predefined set of textual concepts derived from balanced web data)

The procedure uses the pretrained MetaCLIP-H/14-Full-CC2.5B encoders (Xu et al., 2024) as frozen embedding functions:

$$\mathcal{E}_i = \{\Phi_v(i), i \in \mathcal{I}\}$$
$$\mathcal{E}_t = \{\Phi_t(v), v \in \mathcal{V}\}$$

where `$\Phi_v$` is the MetaCLIP image encoder (ViT-H/14), `$\Phi_t$` is the MetaCLIP text encoder, `$\mathcal{E}_i$` is the set of all image embeddings (each a vector in the shared CLIP embedding space), and `$\mathcal{E}_t$` is the set of all concept embeddings (each a vector in the same shared space).

**What this computes:** every image from the eight raw sources is encoded into a 1024-dimensional CLIP embedding vector. Separately, every concept from the 500K MetaCLIP vocabulary is encoded into a 1024-dimensional vector using the same text encoder. Both sets of vectors live in the **same semantic space** — this is the critical property inherited from CLIP-style joint training, where images and text are aligned such that semantically related images and texts have similar embedding vectors.

**Why this form:** the alternative (MetaCLIP's original approach) would be to match images to concepts using the raw captions that come with each dataset — for each image, look at its caption, extract noun phrases, and match those to the concept vocabulary. This fails when captions are missing (SAM-1B has no captions at all — it's a segmentation dataset) or when captions are brief and incomplete (COYO-700M captions are alt-text scraped from the web, often just a few words like "nice photo"). The embedding-based approach bypasses this entirely: the image itself is directly compared to concept embeddings, requiring no intermediate text and being robust to caption quality. The key insight is that MetaCLIP's embedding space is already concept-balanced (because MetaCLIP was trained on a concept-balanced dataset), so projecting images into this space implicitly maps them onto a balanced concept manifold.

**Stage 2: Top-K concept assignment.** For each image `$i$`, the paper computes the cosine similarity between its L2-normalized embedding `$\Phi_v(i) / ||\Phi_v(i)||_2$` and every concept embedding `$\Phi_t(v) / ||\Phi_t(v)||_2$` in the 500K vocabulary. The top-K concepts (the paper does not specify the exact value of K — it says "top-K nearest concepts" without stating K explicitly) with the highest cosine similarity are assigned to that image as pseudo-labels.

This creates a mapping from each image to a small set of concepts that are semantically related to its visual content. For example, an image of a dog at the beach might get assigned concepts like "dog," "beach," "sand," "ocean," "pet," and "outdoors." Crucially, this mapping is derived purely from visual similarity in embedding space, not from any textual annotation of the image.

**Stage 3: Inverse-frequency weighted sampling.** Once every image has its assigned concepts, the paper weights each image by the **inverse frequency** of its assigned concepts across the entire dataset. Formally, for an image `$i$` assigned a set of concepts `$\mathcal{C}(i) \subset \mathcal{V}$`, its weight is:

$$w(i) \propto \sum_{v \in \mathcal{C}(i)} \frac{1}{\text{freq}(v)}$$

where `$\text{freq}(v)$` is the number of images in the dataset that have concept `$v$` among their top-K assignments.

**What this computes:** images assigned to rare concepts get high weights; images assigned to common concepts get low weights. When the final dataset is sampled according to these normalized weights, rare concepts are upsampled and common concepts are downsampled, flattening the overall concept distribution.

**Why this form:** without inverse-frequency weighting, the sampling distribution would mirror the natural concept distribution of web data — heavily skewed toward common concepts. This would mean the model sees millions of redundant examples of "person" and "car" but almost no examples of "abacus." In a fixed compute budget, this is wasteful: each additional example of an already-mastered concept provides diminishing returns, while examples of rare concepts provide disproportionately high information gain. The inverse-frequency weighting is an **information-theoretic argument about efficient data utilization** — it allocates the training budget approximately equally across the concept space rather than proportionally to natural frequency.

Figure 3(a) visualizes the effect: the vocabulary coverage proportion curve after balancing is substantially flatter and higher than before balancing, indicating that more concepts are represented in the final dataset at more uniform frequencies.

The balancing process yields **85M images** with balanced concept coverage. This number (85M) is the result of applying the inverse-frequency sampling to the union of the eight source datasets — the paper does not specify exactly how many raw images were in the initial pool before sampling, but the balanced output is 85M. Figure 3(b) shows the source distribution of these 85M images across the contributing datasets, with Obelics being the largest contributor (it has the broadest concept distribution, as shown in Appendix B, Figure 10).

**Stage 4: Captioning and filtering.** The balanced 85M images — many of which had noisy, missing, or non-English captions — are then re-captioned using "a powerful captioner" (the paper does not specify which captioner, but the context suggests a strong proprietary or open-source image captioning model). Captions are generated in both English and Chinese, producing the final LLaVA-OneVision-1.5-Mid-Traning dataset: **65M English image-text pairs and 20M Chinese image-text pairs**. A validity filter removes duplicate images and captions that are excessively long.

The re-captioning step is important because the original captions from the source datasets are of variable quality (COYO-700M has brief alt-text, SAM-1B has no captions, Obelics has interleaved document text rather than clean captions). By generating fresh captions after sampling, the paper ensures that all 85M examples have consistent, high-quality textual descriptions — a property that is essential for the downstream language model to learn reliable image-text associations.

**Why this entire pipeline over simpler alternatives:** the paper explicitly contrasts with MetaCLIP (Xu et al., 2024), which "depends on raw captions for concept matching and struggles with caption-free or interleaved datasets." MetaCLIP's approach requires every image to have a caption that can be parsed for noun phrases, which then serve as concept queries. This fails for datasets like SAM-1B (no captions), ImageNet-21K (class labels only, not captions), and Obelics (interleaved document context rather than clean image captions). The embedding-based approach eliminates the dependency on caption quality entirely, making it applicable to any image collection regardless of its associated text. The paper frames this as reducing "reliance on caption quality" — a practical engineering consideration that enables the concept-balancing machinery to work across heterogeneous data sources.

---

#### Instruction Dataset Construction (22M + FineVision Merging)

The instruction tuning dataset (LLaVA-OneVision-1.5-Instruct) is the second major data contribution. While the mid-training dataset injects multimodal world knowledge, the instruction dataset teaches the model to follow specific task formats, produce appropriate response styles, and handle the diverse output types required by different benchmarks (multiple choice, free-form answer, bounding box coordinates, code, etc.).

**Data aggregation strategy.** The paper aggregates instruction-tuning datasets "from a wide range of instruction-tuning datasets from diverse sources," totaling **124 distinct data sources**. These are organized into a seven-category taxonomy:

- **Caption** — image captioning tasks (describing image content in natural language)
- **Chart & Table** — understanding structured visual data (bar charts, line plots, tables as images)
- **Code & Math** — reasoning tasks involving code generation from visual specifications and mathematical problem-solving with visual inputs
- **Domain-specific** — specialized domains (medical imaging, satellite imagery, industrial inspection)
- **General VQA** — open-ended visual question answering across diverse image types
- **Grounding & Counting** — spatial localization (identifying object locations, producing bounding boxes) and object counting
- **OCR** — reading and understanding text in images, from document text to scene text
- **Science** — scientific reasoning tasks (diagrams, experimental setups, data interpretation)

The resulting corpus contains **22 million samples**. Figure 3(c) shows the proportional distribution across these categories. The paper does not enumerate all 124 sources or provide the exact proportion for each category (the figure is a pie chart without numerical percentages), but the taxonomy itself encodes a design philosophy: **capability coverage matters more than data source prestige**. The categories are chosen to ensure the model has exposure to the full range of task types that multimodal benchmarks evaluate.

**Scaling via FineVision merging (Merged46M).** The paper goes beyond its own 22M dataset by incorporating FineVision (Wiedmann et al., 2025), a recently proposed open instruction dataset. The merging process involves deduplication (removing samples that appear in both datasets) and concatenation, resulting in a **Merged46M dataset** (the name indicates 46M total samples: 22M from LLaVA-OneVision-1.5-Instruct plus approximately 24M from FineVision after deduplication).

To maintain consistent training steps despite the doubled data volume, the batch size is doubled when training on Merged46M compared to training on LLaVA-OneVision-1.5-Instruct alone. This is a practical consideration: if the batch size remained the same, training on twice the data would require twice the number of steps, violating the compute budget. Doubling the batch size keeps the step count constant while processing all the data, at the cost of larger memory requirements per step (which the infrastructure accommodates).

Figure 8 shows the performance comparison across 16 benchmarks when using LLaVA-OneVision-1.5-Inst-Data alone, FineVision alone, and Merged46M. The Merged46M dataset "delivers the best results across nearly all benchmarks," confirming the paper's broader thesis that data scaling at the instruction-tuning stage yields monotonic improvements — more diverse, higher-quality instruction data consistently improves downstream performance.

**Why instruction data over architecture changes:** the paper's design philosophy is evident here. Rather than designing separate model heads for different task types (one for VQA, one for grounding, one for OCR), the approach is to convert all tasks into a unified text-generation format and rely on data diversity to teach the model the appropriate output format for each task. This is a bet on the power of in-context learning and instruction following: the model learns from examples that "when the instruction says 'give bounding box coordinates,' I should output [x1, y1, x2, y2]" without requiring any architectural specialization. This keeps the model simple (no task-specific modules) but places heavy demands on the instruction dataset's quality and coverage.

---

#### Vision Encoder (RICE-ViT): Region-Aware Cluster Discrimination

The vision encoder is the primary architectural differentiator from prior open LMMs, and understanding its training objective is essential to understanding why it produces representations that are qualitatively different from CLIP or SigLIP.

**The limitation of global contrastive learning.** Standard vision-language encoders (CLIP, SigLIP) are trained with a global alignment objective: an image is encoded into a single vector, a text caption is encoded into another vector, and the training objective pushes matching image-text pairs together while pushing non-matching pairs apart. This produces representations that capture **what** is in an image at a coarse semantic level but lose all spatial information about **where** things are and **how** they relate to each other locally.

The paper identifies two specific failures of this approach:

1. **False negative suppression of semantic structure.** In instance-wise contrastive learning, if a batch contains two images of different dogs with different captions, they are treated as negatives for each other. The loss pushes their embeddings apart, even though they are semantically related (both contain dogs). Over millions of training examples, this destroys the similarity structure that could enable fine-grained discrimination — the model learns that all images are maximally distinct, which is the wrong inductive bias for tasks requiring comparison or analogy.

2. **Global pooling collapses spatial information.** CLIP/SigLIP use global average pooling over spatial positions to produce a single embedding vector. This means the model cannot represent that "a dog is in the top-left corner and a cat is in the bottom-right" — all spatial distinctions are averaged away. For OCR, grounding, and detailed visual reasoning, this is catastrophic.

**The cluster discrimination alternative.** RICE-ViT replaces instance-wise contrastive learning with **cluster discrimination**: instead of asking "is this image the match for this caption?" it asks "does this image region belong to the same semantic cluster as this concept?" Operationally, this means:

- The model operates on **image regions** (candidate bounding boxes or segmentation masks), not whole images.
- Regions are assigned to semantic clusters based on their visual and textual features.
- The training objective is to predict which cluster a region belongs to, where clusters are discovered automatically during training (via clustering in the learned representation space).

This objective has two key properties that the paper argues are essential for LMM vision encoders:

- **Similarity preservation:** regions belonging to the same semantic cluster (e.g., all dog-face regions, all text-string regions, all car-wheel regions) are mapped to similar embeddings, even if they come from different images. This preserves the similarity structure that global contrastive learning destroys.
- **Spatial resolution:** because the model operates on regions rather than whole images, spatial information is preserved in the representation — each spatial location has its own embedding that captures local semantics.

**Training scale.** RICE-ViT is trained on **450M images** with **2.4B candidate regions** (the paper states this explicitly in Section 2.2). The 2.4B regions are presumably generated by a region proposal method (not specified, but likely based on object detection or segmentation proposals) applied to the 450M images, yielding roughly 5.3 regions per image on average. This scale is significant: it means the model has seen billions of individual region-concept associations, enabling it to learn fine-grained distinctions that would be impossible with smaller-scale training.

**2D Rotary Positional Encoding (RoPE).** RICE-ViT uses 2D RoPE rather than learned absolute position embeddings or 1D RoPE. Standard 1D RoPE (used in LLMs) encodes position along a single dimension (token index). 2D RoPE encodes position along two spatial dimensions (height and width), applied separately to the horizontal and vertical axes.

The practical advantage is **native resolution support**. Models that use absolute position embeddings (learned vectors for each spatial position) are locked to a specific input resolution — if you train at 224×224, you cannot process a 448×448 image without interpolation or architectural hacks because the positions beyond 224 have no learned embeddings. Models that use 1D RoPE treat the image as a 1D sequence of patches (flattening the 2D grid), which discards spatial structure and makes it harder to represent 2D spatial relationships.

2D RoPE solves both problems: because the encoding is a continuous function of spatial coordinates (using sine/cosine bases applied to the x and y positions separately), it naturally generalizes to any resolution. A model trained at 224×224 can process a 1024×1024 image simply by computing the RoPE encodings for the additional spatial positions — no fine-tuning, no interpolation, no architectural changes. This is a specific advantage the paper claims over Qwen2-VL and InternVL 2.5, which "require resolution-specific fine-tuning."

**Joint object and OCR region modeling (Figure 2).** During pretraining, RICE-ViT is trained to model both **object regions** (regions containing semantic objects — dogs, cars, faces, buildings) and **OCR regions** (regions containing text — words, paragraphs, document sections) using the same architecture and the same cluster discrimination loss. This is a deliberate design choice: rather than having separate encoders for visual and textual content, a single encoder learns to represent both in a shared embedding space.

This design assumes that the features useful for recognizing text (stroke patterns, spacing, font characteristics, character shapes) and the features useful for recognizing objects (shape, texture, color, context) can coexist in a single representation without destructive interference. The paper's empirical results in Table 2 (strong performance on both general VQA and OCR benchmarks) validate this assumption, but the choice is not obviously correct a priori — an alternative would be to have a dedicated OCR encoder (or OCR-specific layers) to prevent the text-recognition features from being diluted by the object-recognition features.

**Comparison with SigLIP2.** The paper explicitly positions RICE-ViT against SigLIP2 (Tschannen et al., 2025), which represents the state-of-the-art in multi-loss vision encoder training. SigLIP2 uses four separate loss functions:
- **Sigmoid loss** (SigLIP's original objective): global image-text matching
- **SILC loss** (Self-supervised Image-Language Consistency): local-global feature consistency
- **TIPS loss** (Text-Image Positional Supervision): predicting text positions from image features
- **LocCa loss** (Localization-aware Captioning): generating captions that describe specific image regions

The paper argues that this multi-loss approach "depends on multiple specialized losses" and that RICE-ViT's single cluster discrimination loss "simultaneously strengthens general understanding, OCR, and localization" — providing "an elegant, computationally efficient solution that matches SigLIP2's performance while substantially reducing architectural complexity and training overhead."

Table 2 (vision encoder comparison) provides the empirical evidence for this claim. At comparable resolutions, RICE-ViT (ViT-L-14-378px) achieves:
- InfoVQA: 48.1 vs. SigLIPv2 (SO400M-14-384px)'s 43.7 (+4.4%)
- DocVQA: 82.6 vs. 79.1 (+3.5%)
- ChartQA: 75.1 vs. 70.2 (+4.9%)

while matching or slightly exceeding SigLIP2 on general vision benchmarks. This is the paper's core architectural argument: you don't need multiple special-purpose losses to get strong region-level representations — a single well-designed objective (cluster discrimination) with sufficient training data (450M images, 2.4B regions) suffices.

**The [CLS] token preservation (Figure 2).** The architecture preserves the standard ViT [CLS] token, which aggregates global image-level information, alongside the spatial patch tokens that encode region-level information. This is mentioned in the Figure 2 caption: "the [CLS] token is preserved to retain global semantic capacity during multimodal alignment." The rationale is that while region-level features are essential for fine-grained tasks, global features are still useful for tasks requiring holistic image understanding (scene classification, overall image captioning, aesthetic judgment). The projector maps both the [CLS] token and the spatial patch tokens into the LLM's embedding space, giving the language model access to both global and local visual information.

---

#### Training Pipeline: Three Stages and Infrastructure

The training pipeline has three stages with specific data, parameter freezing, and compute configurations.

**Stage-1: Language-Image Alignment.**

- **Data:** LLaVA-1.5 558K (Liu et al., 2024a) — the original LLaVA pretraining dataset of image-text pairs designed for projector training.
- **What is trained:** Only the projector (the 2-layer MLP that maps grouped visual features to LLM embedding space). Both the vision encoder and LLM are frozen.
- **Purpose:** Align the visual feature space with the LLM's text embedding space so that visual tokens can be interleaved with text tokens without causing the LLM to produce garbage. This is the standard first stage in LLaVA-family models — the projector needs to learn the mapping before the LLM can process visual information meaningfully.
- **Rationale for freezing:** If the LLM were trained at this stage, it would overfit to the small 558K dataset and lose its pretrained language capabilities. If the vision encoder were trained, the projector would be chasing a moving target. Freezing both and training only the projector establishes a stable visual-textual interface without damaging either the visual or linguistic representations.

**Stage-1.5: High-Quality Knowledge Learning (Mid-Training).**

- **Data:** LLaVA-OneVision-1.5-Mid-Traning (85M concept-balanced image-text pairs — 65M English, 20M Chinese).
- **What is trained:** All parameters — vision encoder, projector, and LLM — are unfrozen and trained jointly. This is full-parameter training.
- **Purpose:** Inject multimodal world knowledge into the model. The LLM, which was pretrained on text only, now learns to associate visual concepts with linguistic descriptions across a vast and balanced concept space. The vision encoder, which was pretrained on cluster discrimination, now adapts to the specific distribution of images and concepts in the mid-training dataset. The projector continues to refine the visual-to-textual mapping.
- **Context length:** 8K tokens — sufficiently long to accommodate the visual tokens from the image encoding (typically hundreds to thousands of tokens depending on resolution) plus the caption text.
- **Rationale for full-parameter training:** This is the knowledge injection phase. If the LLM were frozen, it could not learn new associations between visual concepts and linguistic terms that were absent from its text-only pretraining. If the vision encoder were frozen, it could not adapt to the concept distribution of the mid-training data. Full-parameter training is expensive but necessary for the model to genuinely acquire new multimodal capabilities rather than just routing pre-existing visual features through a new interface.

**Stage-2: Visual Instruction Tuning.**

- **Data:** LLaVA-OneVision-1.5-Instruct (22M instruction samples) plus FineVision (Wiedmann et al., 2025), merged and deduplicated to form the Merged46M dataset.
- **What is trained:** All parameters, continuing full-parameter training.
- **Purpose:** Teach the model to follow diverse visual instructions, produce task-appropriate response formats, and handle the specific output types required by different benchmarks (multiple choice selections, free-form answers, bounding box coordinates, code blocks, mathematical derivations). This is where the model transitions from a general-purpose multimodal knowledge base to a practical assistant capable of understanding and executing user instructions.
- **Batch size adjustment:** When training on Merged46M (twice the size of LLaVA-OneVision-1.5-Instruct alone), the batch size is doubled to maintain consistent training steps within the compute budget.
- **Rationale for full-parameter training:** Instruction tuning benefits from updating all parameters because instruction-following behavior requires coordination across the entire pipeline — the vision encoder needs to attend to task-relevant visual features, the projector needs to route those features appropriately, and the LLM needs to generate the correct response format.

**Infrastructure: Data Packing for Padding Efficiency.**

A major source of inefficiency in multimodal training is padding. Multimodal data is heterogeneous: some images are small (producing few visual tokens), some are large (producing many visual tokens), some captions are short (few text tokens), and some are long (many text tokens). In standard batching, all samples in a batch are padded to the length of the longest sample in that batch, meaning that a batch containing one very long sample wastes compute on padding tokens for all other samples.

The paper's **offline parallel data packing** addresses this by consolidating multiple shorter samples into packed sequences during preprocessing, before training begins:

- **Hash buckets:** Samples are grouped into buckets based on their sequence lengths using hash functions, enabling efficient lookup and combination of samples with compatible lengths.
- **Multi-threaded, strategy-aware batching:** The packing algorithm uses multiple threads in parallel and considers both the packing success rate (what fraction of samples can be successfully packed) and batch composition (ensuring packed sequences contain diverse content rather than all-similar samples).
- **Offline execution:** Unlike online packing, which dynamically combines samples during training (adding latency and complexity to each training step), the offline approach processes entire datasets or large contiguous chunks in advance, producing uniform-length packed sequences that can be fed directly to the model.
- **Compression ratio:** The paper reports "up to an 11× compression ratio on 85 million pretraining samples" — meaning the 85M individual samples are packed into roughly 7.7M sequences, each containing (on average) 11 original samples concatenated together with appropriate attention masking to prevent cross-sample attention.

The 11× compression directly translates to 11× fewer training steps (each step processes 11× more data on average), which is how the paper achieves the $16,000 budget while processing 85M mid-training samples. Without packing, training on 85M samples would require approximately 11× more GPU-hours, far exceeding the budget.

**Infrastructure: Hybrid Parallelism Framework.**

The training uses AIAK-Training-LLM, a framework built on Megatron-LM (Shoeybi et al., 2019) with Baidu Cloud optimizations. The parallelism strategy combines:

- **Distributed optimizer parallelism:** The optimizer state (AdamW moments for every parameter) is sharded across GPUs, reducing per-GPU memory requirements. This is essential for training an 8B-parameter LLM with full-parameter updates on 128 GPUs — without sharding, the optimizer state would exceed individual GPU memory.
- **Uniform recomputation:** Intermediate activations that would normally be stored for backpropagation are instead recomputed during the backward pass. This trades additional computation (re-running forward passes) for lower memory usage, enabling larger batch sizes and longer sequence lengths within the GPU memory constraint. "Uniform" means the recomputation pattern is the same across all layers, simplifying the implementation.

The mid-training of LLaVA-OneVision-1.5-8B is conducted "at native resolution on 85 million captions using 128 × A800 GPUs over 3.7 days." The A800 is a GPU with 80GB of memory (similar to the A100 80GB), and 128 GPUs over 3.7 days is approximately 11,366 GPU-hours. At typical cloud pricing (~$1.40/GPU-hour for A100-80GB equivalents), this translates to roughly $15,900 — consistent with the stated $16,000 budget.

---

#### RL Post-Training: Discrepancy-Driven GRPO with Asynchronous Infrastructure

The RL post-training stage is an optional but impactful addition that targets a specific capability gap: complex multimodal reasoning that requires multi-step chain-of-thought processes. The supervised model already has strong performance on perception and understanding tasks (General VQA, OCR, Chart), but the paper identifies that its reasoning capabilities — particularly on math, science, and structured problem-solving — can be substantially improved through RL.

**Problem framing: elicitation, not injection.** The paper is explicit about what RL is doing: it is an **elicitation mechanism** that redirects probability mass toward reasoning paths the model can already generate but does not reliably prioritize. This framing is important because it distinguishes RL post-training from the earlier supervised stages. Mid-training injects new knowledge (the model learns associations it didn't have before). Instruction tuning teaches format and task structure. RL, by contrast, works with the existing knowledge in the model and reshapes its output distribution to favor correct reasoning paths.

**Discrepancy-driven data selection.** The key insight for data curation is to select training instances where there is a large gap between `Pass@N` and `Pass@1` performance. Here:

- `Pass@1` is the probability that a single sampled response from the supervised model is correct — essentially, how often the model's most likely output is right.
- `Pass@N` (for some N > 1) is the probability that at least one of N sampled responses is correct — how often a correct answer exists somewhere in the model's sampling distribution.

A large gap between `Pass@N` and `Pass@1` means: **the model knows how to solve this problem** (a correct solution exists in its output distribution) but **does not reliably produce the correct solution** (its policy assigns low probability to the right reasoning path). This is the ideal case for RL: the model already possesses the capability, and RL just needs to shift probability mass toward the correct path.

Conversely, instances where `Pass@1` is already high are trivial (the model already solves them reliably — RL provides no benefit), and instances where `Pass@N` is near zero are unsolvable (the model has no correct solution in its distribution — RL cannot create capability from nothing). The discrepancy-based selection biases the RL corpus toward **medium-difficulty instances at the model's effective learnable boundary**, which provide the most valuable learning signal per training step.

The exact value of N and the specific threshold for the discrepancy gap are not specified in the paper, but the principle is clearly articulated.

**Data composition (67K instances, Figure 4).** The final RL corpus of 67,000 instances is drawn from nine public data sources organized by task category:

- **STEM reasoning (38.9K, 58.0%):** From ViRL (Wang et al., 2025a) — ViRL39K specifically. This is the largest category, reflecting the paper's emphasis on reasoning.
- **Grounding (15K, 22.4%):** Aggregated from Ref-L4 (Chen et al., 2024a) and VigoRL-SA (Sarch et al., 2025) — tasks requiring spatial localization of objects based on textual descriptions.
- **Spatial reasoning (4.2K, 6.3%):** From VigoRL-SAT (Sarch et al., 2025) — tasks involving spatial relationships and reasoning about object positions.
- **Counting (2.8K, 4.2%):** From PixmoCount (Deitke et al., 2025) — object counting tasks.
- **Coding (4K, 6.0%):** From WebCode2M (Yun et al., 2024) and UniSVG (Li et al., 2025b) — code generation from visual specifications and SVG generation.
- **OCR (2K):** From InfoVQA (Mathew et al., 2022a) — reading text from infographics.
- **Diagram understanding (0.2K, 0.3%):** From AI2D (Kembhavi et al., 2016b) — understanding scientific diagrams.

Each instance is labeled with whether it uses a **short answer-only prompt** (output only the final answer) or a **chain-of-thought prompt** (output reasoning steps then answer). This labeling is exploited in the two-stage curriculum.

**Reward-based filtering.** Beyond discrepancy-based selection, the paper applies a second, finer-grained filter: for each candidate instance, the base model generates multiple candidate responses, and automatic rewards are computed for each. Only instances where the **average reward across candidates falls within a specified range** are retained. The range is designed to filter out both:
- Instances where the average reward is very high (the model gets them right almost every time — these are too easy and provide no learning signal).
- Instances where the average reward is near zero (the model gets them wrong almost every time — these are unsolvable and RL cannot help).

This filtering biases the corpus toward medium-difficulty instances, where rewards are intermediate (the model sometimes gets them right, sometimes wrong), maximizing the gradient signal for policy improvement.

**Reward system: type-specific verification.** The paper implements a rule-based reward paradigm — rewards are computed directly from task outcomes using deterministic rules, not from a learned reward model. The key challenge is that different answer types require fundamentally different verification strategies:

- **STEM (ViRL39K):** The primary challenge is format variation — the same correct answer can be expressed in many ways. The solution is a multi-stage pipeline: (1) extract the answer via flexible parsing (preferring structured tags like `<answer>` but falling back to heuristics like "Final Answer:" when necessary), (2) normalize LaTeX artifacts (e.g., unifying `\frac12` and `\frac{1}{2}`), and (3) for numerical reasoning, perform **symbolic equivalence checking** rather than string matching. Symbolic equivalence checking means the system parses both the model's answer and the ground truth as mathematical expressions and checks whether they are mathematically identical — so `1/2`, `0.5`, `2/4`, and `\frac{1}{2}` all register as the same correct answer.

- **Multiple-choice (various sources):** Extract the option label (A/B/C/D) from the model's response, tolerating formatting variations like `(A)`, `**A**`, `A.`, or just `A`, and match against the reference label. This is simpler than STEM because the output space is constrained to a few discrete options.

- **Coding (WebCode2M):** Reward based on **token- and tag-level overlap** between the generated code and the reference code. This measures how structurally similar the generated code is to the ground truth at the level of HTML tags and code tokens, rather than requiring exact string match.

- **Coding (UniSVG):** Extends the WebCode2M approach with an additional **SVG rendering similarity score** in `[0, 1]`. The generated SVG is actually rendered to an image, and the rendered image is compared to the rendering of the reference SVG. This rewards the model for producing code that *looks right* even if the exact code structure differs from the reference — a more meaningful metric for visual output.

- **Grounding (Ref-L4, VigoRL-SA):** Evaluated by **intersection-over-union (IoU)** between predicted and reference bounding boxes. IoU measures the overlap area divided by the union area — a standard metric for object detection and grounding. For associated multiple-choice queries within grounding tasks, standard accuracy is used.

- **Spatial reasoning (VigoRL-SAT):** Scored by answer accuracy — exact match against the reference answer.

- **Counting (PixmoCount):** Extract the final numeric token from the model's response and require **exact equality** with the gold count. No tolerance for off-by-one errors — the count must be exactly right.

- **OCR (InfoVQA):** Text-similarity-based reward between the predicted and reference strings — likely using a metric like BLEU, ROUGE, or edit distance, though the paper doesn't specify exactly which.

- **Diagrams (AI2D):** Multiple-choice accuracy for diagram understanding questions.

All these type-specific rewards are collapsed into a single scalar per response, which serves as the RL training signal.

**GRPO algorithm and AReaL infrastructure.** The paper uses **Group Relative Policy Optimization (GRPO)** (Shao et al., 2024) as the core RL algorithm, implemented in the AReaL framework (Fu et al., 2025). GRPO is a variant of policy gradient methods that:

- Samples a group of responses for each prompt from the current policy.
- Computes rewards for all responses.
- Normalizes rewards within the group (hence "relative" — the policy is optimized to prefer responses that are better relative to other responses from the same model, not against some absolute reward threshold).
- Updates the policy to increase the probability of above-average responses and decrease the probability of below-average responses.

The paper makes two simplifications to the standard GRPO formulation:

1. **Omission of KL divergence penalty.** Standard GRPO includes a KL divergence term that penalizes the policy for deviating too far from a reference policy (usually the supervised model). The paper omits this penalty, relying instead on **PPO-style clipping** to maintain training stability. PPO clipping limits how much the policy can change in a single update by clipping the probability ratio between old and new policies to a fixed range (typically `[1-ε, 1+ε]`). The rationale is that clipping provides sufficient regularization without the computational overhead of computing KL divergences.

2. **No format reward.** Many RL implementations for language models include an explicit format reward that encourages the model to produce properly structured outputs (e.g., with XML tags like `<answer>`). The paper discards this, relying "solely on outcome-based correctness rewards." The model learns to use the correct format because format errors lead to parsing failures, which result in zero reward — the outcome-based signal is sufficient to shape formatting behavior without an explicit format reward term.

**AReaL asynchronous architecture.** AReaL is "a state-of-the-art asynchronous RL framework" that decouples generation from training:
- **Rollout workers** continuously generate responses from the current policy, compute rewards, and add (prompt, response, reward) tuples to a replay buffer.
- **Trainer workers** sample from the replay buffer and update the model parameters using GRPO, asynchronously from the rollout generation.
- This decoupling "significantly improves GPU utilization compared to synchronous implementations" because generation and training can proceed in parallel without waiting for each other. In a synchronous setup, training workers would idle while rollouts are being generated, and rollout workers would idle while training is happening. The asynchronous design keeps all GPUs busy, maximizing throughput.

**Two-stage curriculum (Figure 4).** The RL training uses a deliberate curriculum to avoid catastrophic forgetting of basic perceptual skills while developing chain-of-thought reasoning:

- **Stage 1: Answer-only RL (19.9K instances, Figure 4b).** The model is trained exclusively on the "normal split" of the data, where instructions ask for only the final answer using the prompt: "Put ONLY your final answer within `<answer></answer>`." This stage covers 19.9K instances drawn from Grounding (75.0%), Counting (14.1%), and OCR & Diagram (10.9%) — predominantly perception-heavy tasks. The purpose is to **solidify basic perceptual skills** before introducing complex reasoning. This serves as a warm-up that ensures the model retains precision on simple problems and "avoids 'over-thinking' when later advancing to complex reasoning chains."

- **Stage 2: Chain-of-thought RL (49.2K instances, Figure 4c).** The model switches to the "long-reasoning split" with the prompt: "Think and solve the following question step by step. Please put your thinking and analysis procedure within `thinking` response. Put ONLY your final answer within `<answer></answer>`." This stage covers 49.2K instances dominated by STEM (79.0%), with smaller proportions of Spatial (8.5%), Coding (8.1%), Grounding (3.0%), Counting (0.6%), and OCR & Diagram (0.8%). The reward is computed only from the content within `<answer></answer>`, meaning the optimization target is answer correctness while the reasoning tokens inside `thinking` serve as auxiliary guidance.

- **Mitigating forgetting:** To prevent the model from losing the short-answer skills learned in Stage 1, Stage 2 mini-batches **interleave a small proportion of normal-set examples** (using the Stage 1 answer-only prompt and reward). These anchor samples act as a regularizer that preserves competence on concise tasks while the RL emphasis shifts toward deeper reasoning. The paper does not specify the exact interleaving ratio.

**Why this curriculum design:** naive RL that immediately trains on chain-of-thought tasks can cause the model to develop "over-thinking" — generating unnecessarily verbose reasoning even for simple tasks that require only a direct answer, wasting inference compute and potentially introducing errors. The two-stage design with interleaved anchor samples ensures the model learns to modulate its reasoning depth based on task complexity: simple tasks get concise answers, complex tasks get detailed chains of thought.

## 4. Key Insights and Innovations

### Innovation 1: Mid-Training as the Primary Lever for Multimodal Capability, Not Architectural Complexity

The paper's most fundamental conceptual contribution is the claim — and the empirical demonstration — that **a dedicated mid-training stage with concept-balanced data at scale is sufficient to produce state-of-the-art LMMs, eliminating the need for complex or specialized training paradigms**. This is not merely a new stage in a pipeline; it is an argument about *where capability comes from* in multimodal models, and the answer this paper gives is different from what the field had implicitly assumed.

**What the field assumed before this work.** The dominant trajectory in LMM development, particularly as models pushed toward stronger OCR, grounding, and reasoning, was toward increasing architectural and training complexity. Qwen2-VL (Wang et al., 2024b) introduced dynamic resolution processing with specialized vision transformer modifications. InternVL 2.5 (Chen et al., 2024c) developed progressive scaling strategies with carefully calibrated vision encoder-language model capacity ratios. SigLIP2 (Tschannen et al., 2025) combined four separate loss functions to capture different aspects of visual understanding. The implicit assumption was that to get fine-grained capabilities — reading text in images, localizing objects, understanding spatial relationships — you needed specialized architectural components or multi-objective training.

The early LLaVA models themselves reflected a simpler era of this assumption: they used a straightforward CLIP encoder, a basic projector, and relatively small-scale instruction data, and their performance had fallen "substantially behind" the state of the art (as the paper candidly acknowledges). The natural inference was that catching up would require adopting the same complexity the frontier models used.

**What this paper argues instead.** The paper's central bet is that **data curation beats algorithmic complexity under a fixed compute budget**. The key finding, stated explicitly in Section 1, is that "simply scaling data at the mid-training stage alone can produce state-of-the-art LMMs, eliminating the need for complex training paradigms." The word "simply" is doing significant argumentative work here — it is a direct claim that the elaborate multi-stage curricula, progressive resolution training, and multi-loss encoder objectives that characterize competing models are unnecessary *if* you get the mid-training data right.

The evidence for this claim is distributed across the paper's architecture and results. The architecture itself is deliberately minimal: a standard ViT encoder (RICE-ViT), a standard 2-layer MLP projector, and a standard Qwen3 LLM — connected in the same "ViT–MLP–LLM" paradigm that LLaVA introduced years ago. There are no dynamic resolution modules, no separate OCR heads, no multi-scale feature pyramids, no auxiliary training objectives beyond the standard autoregressive language modeling loss. All of the performance comes from what data is fed into this simple architecture and how it is balanced.

Figure 6 provides the direct evidence for the mid-training scaling claim: across ten diverse benchmarks, increasing mid-training data from 0M to 4M to 85M produces monotonic improvements at every step. The jump from 4M to 85M yields particularly large gains on OCR-heavy benchmarks like InfoVQA and DocVQA, and on reasoning benchmarks like MathVista and MMMU. This is not a saturation curve approaching diminishing returns — the 85M point is substantially better than 4M across the board, suggesting that even larger mid-training datasets would continue to improve performance.

**Why this is a fundamental shift, not an incremental refinement.** The argument the paper is making — that data curation is the primary driver of multimodal capability, and architecture is secondary — is a **framing shift** with significant implications for how the field allocates research effort. If the paper is right, then progress in open LMMs depends less on inventing new architectures and more on solving the (admittedly difficult) data engineering problems: how to source diverse images, how to annotate them with high-quality captions, how to balance concept coverage across languages, and how to scale instruction data efficiently. This reframes the research agenda from architecture design to data curation methodology.

The paper makes this argument explicit in its comparison with SigLIP2. SigLIP2, with its four specialized loss functions, represents the architectural-complexity approach to improving vision encoders. RICE-ViT, with a single cluster discrimination loss, represents the data-driven approach. Table 2 shows that at comparable resolutions, the simpler RICE-ViT matches or exceeds SigLIP2 on both OCR and general vision benchmarks. The paper's framing is deliberate: "This unified formulation provides an elegant, computationally efficient solution that matches SigLIP2's performance while substantially reducing architectural complexity and training overhead." The word "elegant" here is a value judgment that encodes the paper's philosophy — simple methods with good data beat complex methods with good data, and when operating under a budget constraint, the simplicity premium matters.

This is significant beyond raw performance because it is a **falsifiable claim about the nature of multimodal learning**. If future work finds that certain capabilities (e.g., very fine-grained spatial reasoning, multi-page document understanding, or video temporal dynamics) *cannot* be achieved by a simple ViT-MLP-LLM architecture regardless of data scale, then the paper's central thesis is wrong. The paper is making a strong bet, and the fact that it works across 27 diverse benchmarks with a $16,000 budget is evidence for the bet — but the bet's limits have not been fully mapped.

### Innovation 2: Concept-Balanced Data Curation via Embedding-Based Matching as a General Solution to Long-Tail Visual Concepts

The second major innovation is a specific data curation methodology — **embedding-based concept assignment with inverse-frequency weighted sampling** — that solves a problem the field had previously addressed only with fragile, caption-dependent approaches. This is a methodological contribution with implications beyond the specific model: it provides a general recipe for transforming any collection of images (regardless of caption quality) into a balanced pretraining dataset.

**How the field handled concept balancing before.** The dominant approach to ensuring diverse concept coverage in multimodal datasets was MetaCLIP (Xu et al., 2024), which matches images to a curated concept vocabulary by parsing their natural language captions. Specifically, MetaCLIP extracts noun phrases from captions, looks them up in a predefined concept list (e.g., WordNet synsets or Wikipedia entries), and uses substring matching to assign concepts. Images are then sampled with inverse-frequency weights based on their assigned concepts, producing a balanced dataset.

This approach works when captions are detailed and accurate but fails systematically when they are not. Three failure modes are common: (1) captions may be missing entirely (as in segmentation datasets like SAM-1B, which contain images with pixel-level masks but no natural language descriptions), (2) captions may be brief or noisy (as in COYO-700M, where alt-text captions are often just a few words like "nice photo" or "image of a product"), or (3) captions may be interleaved with non-caption text (as in Obelics, where images appear in the context of web documents with surrounding prose rather than clean image-description pairs). In all three cases, MetaCLIP's caption-dependent matching produces poor or empty concept assignments, and the balancing procedure fails because it cannot assign concepts to images it cannot parse.

**What this paper contributes that is genuinely new.** The embedding-based approach bypasses caption quality entirely. Instead of extracting concepts from captions, it projects images directly into a concept-balanced embedding space (MetaCLIP's shared image-text space) and assigns concepts based on embedding similarity. The key insight is that MetaCLIP's embedding space is *already concept-balanced* — because MetaCLIP was trained on concept-balanced data — so projecting arbitrary images into this space implicitly maps them onto a balanced concept manifold regardless of whether the original images had captions, or whether those captions were good.

This is a **diagnostic insight**: the paper recognized that the bottleneck in prior concept-balancing approaches was not the balancing algorithm itself (inverse-frequency weighting is straightforward) but the *concept assignment step* that precedes it. By changing the assignment mechanism from caption-parsing to embedding-matching, the paper made concept balancing work across heterogeneous data sources with variable or absent captions.

The evidence for the effectiveness of this approach is Figure 3(a) and Figure 7. Figure 3(a) shows that after concept balancing, the vocabulary coverage distribution is substantially flatter — more concepts are represented, and at more uniform frequencies, than in the unbalanced raw data. Figure 7 provides the causal evidence: models trained on 2M concept-balanced samples consistently outperform models trained on 2M randomly sampled (unbalanced) samples across 25 of 27 benchmarks. This is a controlled comparison — same data sources, same model architecture, same training procedure — confirming that the balancing procedure, not just the data volume, is responsible for the performance difference.

**Why this matters beyond this paper.** The embedding-based balancing methodology is a **transferable recipe**. Any research group with access to a pretrained CLIP-style model and a concept vocabulary can apply it to their own image collections, regardless of the quality or presence of captions. This is particularly valuable in specialized domains (medical imaging, satellite imagery, industrial inspection) where images rarely come with high-quality captions but where balanced concept coverage is essential for building capable domain-specific LMMs. The paper's release of the LLaVA-OneVision-1.5-Mid-Traning dataset provides a concrete instantiation, but the methodology itself is the reusable contribution — a solution to the long-tail concept problem that does not depend on the quality of the text associated with images.

The limitation the paper acknowledges is that the embedding-based matching is only as good as the underlying embedding space. If MetaCLIP's shared space does not represent certain visual concepts well (e.g., very rare or highly specialized concepts that were underrepresented in CLIP training data), the embedding similarity will be unreliable for those concepts, and the balancing procedure will fail to upweight them appropriately. This is not a flaw in the method per se but a boundary condition: the approach works for concepts within the embedding model's conceptual coverage, which for MetaCLIP's 500K-concept vocabulary covers a very broad range but is not universal.

### Innovation 3: Discrepancy-Driven Data Selection as a Principled Elicitation Framework for RL Post-Training

The third innovation is a **diagnostic framework for when RL post-training can help** — and, by implication, when it cannot — based on the gap between Pass@N and Pass@1 performance. This is not a new RL algorithm (the paper uses existing GRPO) but a new way of thinking about *which problems to apply RL to* and *what RL is actually doing* in the context of multimodal reasoning.

**What the field typically assumes about RL for reasoning.** The dominant narrative around RL for language models, particularly following DeepSeekMath (Shao et al., 2024) and subsequent reasoning-focused models, is that RL "teaches" models to reason — that it injects new reasoning capabilities that were absent from the supervised model. This framing treats RL as a form of continued training that adds capabilities beyond what supervised learning can achieve, often through exploration of solution spaces that the supervised model would not naturally explore.

Under this framing, the natural approach to data selection is to curate a dataset of difficult reasoning problems — the harder the better — and apply RL to teach the model to solve them. The assumption is that RL's exploration mechanisms will discover reasoning paths that supervised learning missed, and that capability emerges from this exploration process.

**What this paper argues instead.** The paper introduces a fundamentally different framing: RL is an **elicitation mechanism, not a knowledge injection mechanism**. The key diagnostic is the gap between Pass@N and Pass@1:

- If Pass@N is high (a correct answer exists in the model's output distribution) but Pass@1 is low (the model doesn't reliably produce it), then RL can help by **redistributing probability mass** toward the correct reasoning path. The capability already exists; RL just makes the model more likely to use it.
- If Pass@N is near zero (no correct answer exists in the output distribution), then RL **cannot help** — no amount of policy optimization can surface a capability that isn't there. The model needs more supervised training (more knowledge injection) before RL becomes useful.
- If Pass@1 is already high, RL provides minimal benefit — the model already solves the problem reliably.

This framing has an immediate practical consequence: **you should select RL training data specifically from instances where the Pass@N vs. Pass@1 gap is large**, filtering out both too-easy and too-hard problems. The paper implements this with both the discrepancy-driven selection (explicitly selecting instances with a large gap) and the reward-based filtering (retaining only instances where the average reward across candidates falls in an intermediate range).

**Why this is a conceptual advance, not just a filtering heuristic.** The elicitation-versus-injection distinction is a **diagnostic tool** that changes how practitioners should think about RL for multimodal models. It implies that before applying RL, one should measure the supervised model's Pass@N on the target distribution — and if Pass@N is low, the correct intervention is *more supervised training* (more data, better data, or more training steps), not RL. RL should be reserved for problems where the model already demonstrates latent capability that just needs to be elicited.

This reframing also explains several patterns in the paper's results that would otherwise be puzzling:

- **Why RL helps most on reasoning benchmarks (WeMath +7.9, MathVision +8.8, MMMU-Pro-vision +10.5) but minimally on General VQA and OCR.** Reasoning tasks require multi-step chains of thought; the supervised model often *can* produce correct solutions when sampled multiple times (high Pass@N) but its greedy output is wrong (low Pass@1) because there are many more wrong reasoning paths than right ones. RL narrows this gap by making the right path more probable. General VQA and OCR tasks, by contrast, are more deterministic — the model's first answer is usually its best answer, so Pass@1 is already close to Pass@N, and RL has little room to improve.

- **Why "thinking" mode sometimes underperforms "fast" mode on perceptual metrics (Ref-L4 IoU in Figure 5).** The elicitation framing predicts this: when the model generates verbose reasoning chains, it may wander into irrelevant considerations that dilute the signal for tasks that are fundamentally about direct perception (like producing bounding box coordinates). The capability to produce the right coordinates exists in the model (high Pass@N in fast mode), but the chain-of-thought process can interfere with the direct mapping from visual input to coordinate output. RL optimizes for answer correctness, not for reasoning efficiency, so the model may learn to produce reasoning even when it hurts, if the training data is biased toward reasoning-heavy tasks.

- **Why the two-stage curriculum (answer-only warmup before chain-of-thought RL) is necessary.** If RL were simply "teaching reasoning," you would want to apply it as aggressively as possible from the start. But if RL is eliciting latent capabilities, then you need to ensure that the basic perceptual capabilities are already well-elicited (the answer-only stage) before adding the complexity of chain-of-thought reasoning, to prevent the exploration process from destabilizing already-reliable behaviors. The interleaving of normal-set anchor samples in Stage 2 is a direct operationalization of the elicitation framing: it maintains the probability mass on correct short-answer paths while shifting mass toward correct reasoning paths.

**Evidence and limitations.** The paper's evidence for the elicitation framing is primarily the performance pattern in Table 1: large gains on reasoning tasks where the gap between greedy and best-of-N is presumably large, and small or negligible gains on perceptual tasks where the gap is presumably small. However, the paper does not directly measure the Pass@N vs. Pass@1 gap for its supervised model — it describes the principle and uses it implicitly in data selection, but never presents the empirical gap measurements that motivated specific data inclusion decisions. This is a limitation: the framing is conceptually compelling but the paper provides qualitative evidence for it (the pattern of RL gains) rather than quantitative validation (direct gap measurements before and after RL).

Nevertheless, the framing itself is the contribution — it provides a **principled answer to the question of when RL is the right tool** that the field previously lacked. Prior work on RL for multimodal models (e.g., VigoRL, ViRL) applied RL broadly to curated reasoning datasets without explicitly distinguishing between capability injection and capability elicitation, and without using the Pass@N/Pass@1 gap as a selection criterion. The paper's discrepancy-driven approach makes this distinction operational and explains *why* RL works on some problems and not others, moving the conversation from "RL can improve reasoning" to "RL can improve reasoning specifically when the model already has latent capability, and here's how to identify those cases."

### Innovation 4: The $16,000 Budget as a Constraint-Driven Research Methodology

The fourth innovation is methodological rather than technical: the paper treats the $16,000 compute budget not as a limitation to be acknowledged but as a **research tool that forces and justifies specific design decisions**. This is a contribution to *how* LMM research is conducted and reported, with implications for reproducibility and democratization that extend beyond any single model.

**What the field typically does with compute constraints.** Most LMM papers either (a) do not report training costs at all, leaving the community to guess whether results are achievable with reasonable resources, (b) report costs indirectly (e.g., "trained on 256 GPUs for 5 days") without translating to a dollar figure, or (c) acknowledge budget constraints only in the limitations section, treating them as obstacles that prevented even better results. The implicit message is that higher performance would have been achieved with more compute — the budget is a ceiling, not a feature.

This creates several pathologies in the research ecosystem. Results are not reproducible because the required budget is unknown or prohibitive. Comparisons between methods are confounded by budget differences — a method that performs better may simply have used more compute. And researchers without access to large GPU clusters are structurally excluded from contributing to the frontier, since they cannot verify or build upon existing results.

**What this paper does differently.** The paper makes three specific methodological commitments that collectively treat the budget as a research tool:

1. **The budget is stated upfront and precisely** — $16,000 — and the training configuration is described in enough detail (128 A800 GPUs, 3.7 days) that the cost can be independently verified. This is not "we trained on a modest budget" but rather "here is exactly what we spent, and here is exactly what hardware we used, so you can replicate this exactly."

2. **Design decisions are justified by reference to the budget constraint.** The offline data packing (11× compression) is not presented as a nice-to-have optimization but as a necessary step to stay within the budget while processing 85M mid-training samples. The choice of Qwen3 as the language backbone is partly justified by efficiency. The decision to forgo complex multi-stage curricula or architectural modifications is explicitly argued as a budget-driven tradeoff — simpler methods that work with less compute are preferred over complex methods that would exceed the budget. The budget is not an afterthought; it is the organizing principle for the entire technical approach.

3. **The budget constraint is used to make a positive argument about accessibility.** The paper frames the $16,000 figure as evidence that "state-of-the-art performance" is achievable with "limited computational cost" — the constraint is proof that high-quality LMMs can be built without industrial-scale resources, which is a substantive claim about the democratization of AI research. If the paper had achieved similar results with a $500,000 budget, the democratization argument would be weaker; the specific dollar figure matters because it is low enough to be within reach of academic labs, small companies, and even well-funded individual researchers.

**Why this is an innovation in research practice.** The budget-as-methodology approach is a **contribution to research norms**, not to algorithms or architectures. It provides a template for how LMM papers *should* report their work: with explicit budgets, cost-justified design decisions, and enough detail for independent reproduction. This matters because the current norms in the field — where costs are opaque and results are difficult to verify — create a "reproducibility gap" that this paper directly addresses by releasing all assets (datasets, code, model weights, training configuration) with explicit cost figures.

The paper's ablation in Figure 6 is a direct product of this methodology. The question "does mid-training data scale help?" could be answered in the affirmative by any team with a large GPU budget — just train on 100M, 200M, 500M examples and show the curve. But the paper asks a more constrained question: "does mid-training data scale help *under a fixed budget*?" The answer matters precisely because the budget is the bottleneck. If scaling from 4M to 85M had produced only marginal gains, the paper's central thesis would be weakened — the budget would be better spent elsewhere. The fact that the gains are substantial and monotonic validates the budget allocation: most of the $16,000 went to mid-training data processing and training, and that was the right choice.

**Limitations of this perspective.** The paper does not explore whether alternative budget allocations would have produced even better results. Could $8,000 of mid-training with a more sophisticated instruction-tuning stage outperform $12,000 of mid-training with $4,000 of instruction tuning? Could a different vision encoder (e.g., SigLIP2 at a lower resolution) have freed up budget for more training steps? The paper's single budget point ($16,000) is a proof of concept — it shows what is possible at this price point — but it does not provide a budget scaling curve showing how performance varies as the constraint is tightened or relaxed. This is a natural extension of the methodology but is absent from the current paper.

Nevertheless, the methodological contribution stands: by treating the budget as a transparent constraint that shapes design decisions rather than a limitation to be overcome or ignored, the paper provides a model for how open LMM research can be conducted and communicated in a way that genuinely enables reproducibility and participation from resource-constrained researchers.

## 5. Experimental Analysis

### Evaluation Methodology

- **Dataset.** All primary evaluations use the **LMMs-Eval** framework (Zhang et al., 2025a) with the default prompt, spanning 27 benchmarks across four task categories: General VQA (9 benchmarks including MMStar, MMBench, MME-RealWorld, SeedBench, CV-Bench, RealWorldQA), Multimodal Reasoning (6 benchmarks including MathVista, WeMath, MathVision, MMMU, MMMU-Pro), OCR & Chart Understanding (7 benchmarks including ChartQA, CharXiv, DocVQA, OCRBench, AI2D, InfoVQA), and Others (4 benchmarks including PixmoCount, CountBench, VL-RewardBench, V∗). The paper also uses SAT, TreeBench, Ref-L4, RefCOCO, WebCode2M, Design2Code, and UniSVG for spatial reasoning, grounding, and coding evaluations in Figure 5. Ablation experiments on vision encoders use LLaVA-NeXT's tiling strategy (up to 2×2+1 tiles) for fair comparison across encoders that do not natively support resolution adaptation.

- **Base model(s).** The primary models are **LLaVA-OneVision-1.5-4B** and **LLaVA-OneVision-1.5-8B**, both using RICE-ViT (ViT-L-14) as the vision encoder and Qwen3 as the language backbone. The 4B variant uses Qwen3-4B; the 8B variant uses Qwen3-8B. These are compared against **Qwen2.5-VL-7B** and **Qwen2.5-VL-3B** as the primary baselines, and against **LLaVA-OneVision-7B** (the prior generation) for historical context. The RL-enhanced variant, **LLaVA-OV-1.5-RL-8B**, is evaluated against the supervised LLaVA-OneVision-1.5-8B and Qwen2.5-VL-7B. An additional **LLaVA-OneVision-1.5-3B** variant, trained with Qwen2.5-3B-Instruct as the LLM (rather than Qwen3), is evaluated in Appendix A for a same-LLM comparison against Qwen2.5-VL-3B.

- **Metrics.** All benchmarks report **accuracy (%)** as the primary metric, with evaluation handled by the LMMs-Eval framework using each benchmark's standard grading protocol. For spatial reasoning and grounding tasks in Figure 5, **Intersection-over-Union (IoU)** is reported for bounding box prediction tasks (Ref-L4, RefCOCO), and **accuracy** is reported for multiple-choice spatial reasoning tasks (SAT, TreeBench). For coding tasks in Figure 5, the paper reports benchmark-specific scores (WebCode accuracy, Design2Code score, UniSVG score) — exact metric definitions are determined by each benchmark's evaluation protocol. The paper does not report confidence intervals, standard deviations, or statistical significance tests for any results.

- **Baselines.** The primary baselines are:
  - **Qwen2.5-VL-7B** and **Qwen2.5-VL-3B** (Bai et al., 2025): the most recent open-weight multimodal models from the Qwen family at comparable parameter scales, representing the current frontier for accessible LMMs.
  - **LLaVA-OneVision-7B** (Li et al., 2025a): the prior generation in the LLaVA series, included to show the performance improvement from LLaVA-OneVision-1.5's design choices.
  - For vision encoder ablations (Table 2): **CLIP** (Radford et al., 2021, ViT-L-14-336px), **MLCD** (An et al., 2024, ViT-L-14-336px), **AIMv2** (Fini et al., 2025, ViT-L-14-336px), **DFN5B** (Fang et al., 2023, ViT-H-14-378px), **SigLIP** (Zhai et al., 2023, ViT-SO400M-14-384px), **SigLIPv2** (Tschannen et al., 2025, at 384px and 560px), and **Qwen-ViT** (from Qwen2.5-VL-7B, ViT-H-14-560px). All vision encoder comparisons use the LLaVA-NeXT framework with Qwen2.5-7B as the language model and identical training data and pipeline.
  - For instruction data ablations (Figure 8): **FineVision** (Wiedmann et al., 2025) is used as a standalone SFT dataset for comparison against LLaVA-OneVision-1.5-Inst-Data.

- **Generation budget / compute accounting.** The paper's compute accounting focuses on **training budget** rather than inference budget. The primary constraint is a **$16,000 total training budget**, with the 8B model's mid-training conducted on 128 × A800 GPUs over 3.7 days (approximately 11,366 GPU-hours). For the RL post-training stage, generation and training are decoupled via the AReaL asynchronous framework, but no explicit inference compute budget is reported — the RL stage is described as "lightweight" but no FLOP or GPU-hour count is provided. The offline data packing achieves an 11× compression ratio on 85 million samples, meaning the effective training throughput is approximately 11× higher than naive batching with padding. For fair comparison in vision encoder ablations, all models are trained with identical configurations (same LLM, same training data, same pipeline) using LLaVA-NeXT's tiling strategy to normalize for resolution handling differences.

- **Cross-validation / statistical protocol.** The paper does not report any cross-validation, statistical significance testing, or confidence intervals. Results are reported as single-point accuracy scores on each benchmark's standard test set. For the RL post-training experiments, the "thinking" mode and "fast" mode are both evaluated, but no ensemble or multi-seed results are reported. The difficulty-based discrepancy selection for RL data (Pass@N vs. Pass@1 gap) is described qualitatively without specific threshold values, and no sensitivity analysis on the selection threshold is provided. The reward-based filtering for RL data uses a "specified range" for average rewards, but this range is not numerically specified.

### Main Quantitative Results

#### Overall Performance Against Qwen2.5-VL (Table 1)

**Headline comparison.** LLaVA-OneVision-1.5-8B outperforms Qwen2.5-VL-7B on **18 of 27 benchmarks**, and LLaVA-OneVision-1.5-4B outperforms Qwen2.5-VL-3B on **all 27 benchmarks**. This is the paper's primary competitive claim and is supported by Table 1.

**Same-LLM controlled comparison (Appendix A, Figure 9).** When both models use the same language backbone (Qwen2.5-3B-Instruct), LLaVA-OneVision-1.5-3B outperforms Qwen2.5-VL-3B on **17 of 27 benchmarks**. This controls for LLM quality differences and isolates the effect of the vision encoder and data pipeline. The paper does not report the analogous same-LLM comparison for the 8B scale (i.e., LLaVA-OneVision-1.5 with Qwen2.5-7B vs. Qwen2.5-VL-7B), which would be the most direct test of the vision encoder and data pipeline's contribution versus the LLM upgrade from Qwen2.5 to Qwen3.

**Category-level analysis (Table 1).** The performance advantage varies systematically by task category:
- **General VQA (9 benchmarks):** The 8B model averages 71.8% vs. Qwen2.5-VL-7B's average of 72.2% across the 9 benchmarks shown — actually *slightly behind* on average. The 4B model averages 72.1% vs. Qwen2.5-VL-3B's 66.4%. The 8B's advantage is concentrated on MMStar (+0.1 over Qwen2.5-VL-7B after subtracting — actually, reading Table 1 more carefully: MMStar 67.7 vs. 68.3, so -0.6; MMBen 84.1 vs. 85.7, so -1.6; MMBen-cn 81.0 vs. 81.5, so -0.5; MME-RealWorld-en 61.7 vs. 63.3, so -1.6; MME-RealWorld-cn 56.1 vs. 56.3, so -0.2; SeedBench 77.3 vs. 77.6, so -0.3; CV-Bench 80.7 vs. 81.1, so -0.4; SEED-Bench-2-Plus 69.2 vs. 69.2, tie; RealWorldQA 68.1 vs. 70.6, so -2.5). On General VQA, **Qwen2.5-VL-7B actually outperforms LLaVA-OneVision-1.5-8B on most benchmarks**, with the 8B model trailing on 8 of 9 benchmarks (all except SEED-Bench-2-Plus, which ties). This is a notable pattern that the paper does not explicitly discuss.
- **Multimodal Reasoning (6 benchmarks):** The 8B model averages 45.8% vs. Qwen2.5-VL-7B's 45.5% — essentially tied on average, with specific advantages on MathVista (+0.6 at 69.6 vs. 71.8, but the RL model gets 72.3 — the base model trails Qwen2.5-VL-7B slightly), WeMath (+0.7 at 61.5 vs. 60.8), MathVision (+3.2 at 25.6 vs. 22.4), MMMU-val (+4.1 at 55.4 vs. 51.3), and MMMU-Pro-standard (+1.1 at 37.4 vs. 36.3). However, on MMMU-Pro-vision, the base model trails Qwen2.5-VL-7B (25.2 vs. 32.8), suggesting that Qwen2.5-VL has stronger vision-heavy reasoning.
- **OCR & Chart (7 benchmarks):** The 8B model averages 84.6%, essentially tied with Qwen2.5-VL-7B's 84.4%. Individual benchmark differences are small and mixed: ChartQA (86.5 vs. 87.1, -0.6), CharXiv (70.9 vs. 69.8, +1.1), DocVQA (95.0 vs. 94.9, +0.1), OCRBench (82.9 vs. 84.2, -1.3), InfoVQA (78.4 vs. 81.7, -3.3). The paper's strongest OCR advantage is at the 4B scale against Qwen2.5-VL-3B, where LLaVA-OneVision-1.5-4B outperforms on all 7 benchmarks.

**Key insight from the 4B vs. 3B comparison.** The 4B model's consistent advantage over Qwen2.5-VL-3B is the paper's strongest result, since it is unambiguous (27 of 27 benchmarks) and spans all task categories. The 8B vs. 7B comparison is much closer — the 8B model leads on only 18 of 27 benchmarks, and the advantages are often marginal (single-digit percentage points). The paper's claim of "state-of-the-art performance" at the 8B scale should be understood as broadly competitive rather than clearly dominant.

**Comparison against LLaVA-OneVision-7B (prior generation).** The 4B model outperforms the prior 7B model on every single reasoning benchmark (MathVista: 67.9 vs. 58.5; WeMath: 24.9 as printed — but checking Table 1, WeMath for LLaVA-OV-1.5-4B is listed as 62.0 under the Qwen2.5-VL-3B column, not 24.9; MathVision: 24.2 vs. 18.5; MMMU-val: 52.7 vs. 48.8; MMMU-Pro-standard: 35.3 vs. 28.0; MMMU-Pro-vision: 25.4 vs. 14.3). This is a striking finding: a 4B model with better data and vision encoder significantly outperforms a 7B model from the prior generation across all reasoning tasks, demonstrating that data quality and vision encoder choice can more than compensate for a 1.75× parameter disadvantage.

---

#### RL Post-Training Performance (Table 1, Figure 5)

**Overall RL gains.** Comparing LLaVA-OV-1.5-RL-8B against the supervised LLaVA-OneVision-1.5-8B in "thinking" mode and "fast" mode (Table 1):

- **Multimodal Reasoning (thinking mode):** WeMath: 69.4 vs. 61.5 (+7.9); MathVision: 34.4 vs. 25.6 (+8.8); MMMU-val: 58.8 vs. 55.4 (+3.4); MMMU-Pro-standard: 39.9 vs. 37.4 (+2.5); MMMU-Pro-vision: 35.7 vs. 25.2 (+10.5). The average reasoning improvement across all 6 benchmarks in thinking mode is +6.0 percentage points. These are the largest RL gains in the paper, concentrated on the most complex reasoning tasks.
- **Multimodal Reasoning (fast mode):** Gains are smaller: WeMath: 60.8 vs. 61.5 (-0.7 — a slight regression); MathVision: 26.2 vs. 25.6 (+0.6); MMMU-val: 54.9 vs. 55.4 (-0.5); MMMU-Pro-standard: 38.0 vs. 37.4 (+0.6); MMMU-Pro-vision: 29.0 vs. 25.2 (+3.8). Average improvement in fast mode is +1.0 percentage points — substantially smaller than thinking mode gains.
- **General VQA:** RL provides modest improvements in thinking mode (+1.0 average) and essentially flat performance in fast mode (+0.8). RL does not hurt general VQA, but the gains are small, consistent with the paper's elicitation framing (general VQA tasks have a narrower Pass@N-to-Pass@1 gap).
- **OCR & Chart:** RL in fast mode maintains performance (84.6% average for both supervised and RL-fast). RL in thinking mode shows a slight regression (83.3% vs. 84.6% supervised), likely because chain-of-thought generation interferes with the direct visual perception required for OCR tasks. This is consistent with the Ref-L4 IoU decline shown in Figure 5.

**Spatial reasoning and grounding (Figure 5).** RL in fast mode consistently outperforms the SFT baseline: SAT (test): ~59% vs. ~57%; TreeBench: ~61% vs. ~58%; Ref-L4 (IoU): ~87% vs. ~81%; RefCOCO (IoU): ~94% vs. ~92%. However, RL in thinking mode *degrades* performance on these strictly perceptual metrics: Ref-L4 IoU drops from SFT's ~81% to ~64% in thinking mode, a substantial regression. The paper attributes this to "verbose generation occasionally interfering with precise coordinate regression." On coding tasks (WebCode, Design2Code, UniSVG), RL in thinking mode is the best configuration, achieving the highest scores on Design2Code and UniSVG — suggesting chain-of-thought reasoning is beneficial for structured code generation but detrimental for spatial precision.

**Key pattern: thinking mode helps reasoning and coding, hurts perception.** The RL results reveal a systematic tradeoff that the paper partially acknowledges but does not fully quantify: the "thinking" mode that dramatically improves math and science reasoning (+7.9 to +10.5 on the hardest benchmarks) simultaneously degrades performance on tasks requiring precise spatial outputs (Ref-L4 IoU regression, RefCOCO IoU decline). This is a **capability tradeoff** — the same mechanism that elicits multi-step reasoning can interfere with direct perceptual mappings. The fast mode serves as a compromise, achieving modest reasoning gains without sacrificing perceptual accuracy.

---

#### Vision Encoder Comparison (Table 2)

**RICE-ViT vs. alternative encoders at comparable resolutions.** Table 2 provides the most granular evidence for the paper's claim that RICE-ViT's cluster discrimination objective produces representations superior to both global contrastive encoders (CLIP, SigLIP) and multi-loss encoders (SigLIP2):

- **At 336px:** RICE-ViT (ViT-L-14-336px) achieves 56.2% OCR average (InfoVQA: 45.2, DocVQA: 79.2, ChartQA: 72.3, TextVQA: 65.9, OCRBench: 57.5, OCRBenchV2: 24.1, LiveXivVQA: 48.9) vs. CLIP's 52.3% (+3.9 percentage points) and AIMv2's 54.2% (+2.0 points). In General Vision, RICE-ViT achieves 70.5% average vs. CLIP's 67.6% (+2.9 points).

- **At 378px:** RICE-ViT (ViT-L-14-378px) achieves 58.0% OCR average vs. SigLIPv2 (ViT-SO400M-14-384px)'s 56.0% (+2.0 points) and substantially outperforms DFN5B (ViT-H-14-378px)'s 49.8% (+8.2 points). On individual OCR benchmarks: InfoVQA 48.1 vs. SigLIPv2 43.7 (+4.4); DocVQA 82.6 vs. 79.1 (+3.5); ChartQA 75.1 vs. 70.2 (+4.9). General Vision performance is comparable: RICE-ViT 70.1% vs. SigLIPv2 69.5%.

- **At 560px:** RICE-ViT (ViT-L-14-560px) achieves 61.1% OCR average vs. SigLIPv2 (ViT-SO400M-16-560px)'s 60.9% — essentially tied, with RICE-ViT ahead on InfoVQA (53.2 vs. 50.2, +3.0) and DocVQA (87.4 vs. 86.2, +1.2) but behind on OCRBench (60.7 vs. 62.7, -2.0). Qwen-ViT (ViT-H-14-560px, from Qwen2.5-VL-7B) achieves 62.9% OCR average, surpassing both — though the paper emphasizes that after LMM training, RICE-ViT from LLaVA-OneVision-1.5-3B (64.8% OCR average) outperforms Qwen-ViT from Qwen2.5-VL-7B (62.9% OCR average).

**The critical finding: RICE-ViT's advantage is primarily on OCR-intensive benchmarks.** Across resolution comparisons, RICE-ViT's largest margins over SigLIPv2 are on InfoVQA (+4.4% at 378px, +3.0% at 560px), DocVQA (+3.5% at 378px), and ChartQA (+4.9% at 378px) — all tasks requiring fine-grained text reading in images. On general vision understanding, RICE-ViT and SigLIPv2 are within 1-2 percentage points across resolutions. This validates the paper's architectural argument: the region cluster discrimination objective specifically improves region-level text and visual semantics without sacrificing global understanding.

**The tradeoff at higher resolutions.** At 560px, RICE-ViT (ViT-L-14) faces compute disadvantages against the larger SigLIPv2 (ViT-SO400M) and Qwen-ViT (ViT-H) architectures. RICE-ViT's 61.1% OCR average at 560px is comparable to SigLIPv2's 60.9% but trails Qwen-ViT's 62.9%. However, the paper argues this is the wrong comparison — after full LMM training (the "RICE-ViT from OV-1.5 3B" row), the same RICE-ViT encoder achieves 64.8% OCR average, surpassing Qwen-ViT's 62.9%. This suggests that joint training of the vision encoder with the LLM during mid-training and instruction tuning further improves RICE-ViT's representations beyond what the frozen encoder alone provides — an interaction effect that the vision-encoder-only comparison cannot capture.

---

#### Mid-Training Data Scaling (Figure 6)

**Monotonic improvement with increasing mid-training data.** Figure 6 compares models trained with 0M, 4M, and 85M mid-training samples (all using LLaVA-558K for Stage-1 alignment and LLaVA-NeXT for instruction tuning). Across all ten benchmarks, increasing mid-training data produces monotonic accuracy improvements:

- **OCR-heavy benchmarks show the largest gains from scaling.** InfoVQA: 0M → 4M → 85M shows a progression from approximately 56% → 68% → 75% (a roughly +19 percentage point total gain). DocVQA: approximately 80% → 84% → 87%. OCRBench: approximately 52% → 60% → 65%. This is consistent with the paper's argument that mid-training injects the multimodal world knowledge necessary for OCR — without sufficient mid-training data, the model lacks the visual-textual associations needed to read text in images.

- **General VQA and reasoning benchmarks show substantial but smaller gains.** AI2D: approximately 63% → 72% → 78%. MMBench: approximately 72% → 76% → 79%. MathVista: approximately 42% → 52% → 58%. The gains from 0M to 4M are larger than the gains from 4M to 85M on these benchmarks — suggesting some diminishing returns, though the trend is still positive.

- **The 0M baseline is not zero-capability.** Even without mid-training (using only LLaVA-558K for projector alignment and then instruction tuning), the model achieves non-trivial performance (e.g., ~56% on InfoVQA, ~80% on DocVQA, ~72% on MMBench). Mid-training improves upon this baseline but does not create capabilities from nothing — the base LLM and vision encoder already possess substantial multimodal knowledge from their respective pretraining.

**The 85M point is not a saturation point.** The slopes from 4M to 85M are positive across all ten benchmarks, with no plateau in sight. This suggests that even larger mid-training datasets would continue to improve performance — the 85M figure is a practical budget-constrained choice, not a ceiling imposed by diminishing returns.

---

#### Concept Balancing Ablation (Figure 7)

**Concept-balanced sampling consistently outperforms random sampling.** Comparing models trained on 2M concept-balanced mid-training samples vs. 2M randomly sampled (unbalanced) samples, both using the same data sources and LLaVA-NeXT-780k for instruction tuning:

- The balanced model outperforms the unbalanced model on **25 of 27 downstream benchmarks**.
- The gains are most pronounced on OCR and document understanding tasks, which benefit from exposure to rare text-bearing concepts that random sampling would undersample.
- The two benchmarks where random sampling matches or exceeds balanced sampling are not specified by name in the paper, but Figure 7 shows them as near-ties.

This ablation provides causal evidence that the concept-balancing procedure — not just the data volume or data sources — is responsible for performance improvements. Random sampling from the same data pool produces a model that is nearly as good on common concepts but worse across the board, confirming that the long-tail concept coverage the balancing procedure provides is valuable.

---

#### Instruction Data Quality and Scaling (Figure 8)

**Merged46M consistently outperforms smaller instruction datasets.** Comparing LLaVA-OneVision-1.5-Inst-Data (22M), FineVision alone, and Merged46M (LLaVA-OneVision-1.5-Inst-Data + FineVision after deduplication, 46M total) across 16 benchmarks:

- Merged46M achieves the best or near-best performance on all 16 benchmarks.
- The gains from merging are not simply additive — some benchmarks show substantial jumps from 22M to 46M (e.g., InfoVQA, AI2D), while others show smaller gains (e.g., MMBench, MME).
- FineVision alone performs comparably to LLaVA-OneVision-1.5-Inst-Data on most benchmarks, suggesting both datasets cover similar capability distributions but with complementary examples — merging them provides additional diversity rather than redundancy.

The paper uses doubled batch size for Merged46M to maintain consistent training steps, meaning the comparison controls for number of optimization steps but not for total data processed. The Merged46M model sees twice the data within the same step budget, so the improvement could be attributed to data volume, data diversity, or both.

---

### Ablation Studies and Robustness Checks

**Vision encoder architecture comparison (Table 2):** RICE-ViT at ViT-L-14-336px outperforms CLIP (ViT-L-14-336px) by 3.9% on OCR average and 2.9% on General Vision average, and outperforms AIMv2 (ViT-L-14-336px) by 2.0% on OCR. At 378px, RICE-ViT matches or exceeds SigLIPv2 (the state-of-the-art multi-loss encoder) on 9 of 14 benchmarks while using a simpler training objective. A non-obvious finding: RICE-ViT's advantage over SigLIPv2 is concentrated on OCR benchmarks (InfoVQA +4.4%, DocVQA +3.5%, ChartQA +4.9%), not on general vision tasks, confirming that the region cluster discrimination objective specifically improves fine-grained visual semantics while maintaining comparable global understanding.

**Same-LLM controlled comparison (Appendix A, Figure 9):** When both LLaVA-OneVision-1.5-3B and Qwen2.5-VL-3B use the same Qwen2.5-3B-Instruct LLM, LLaVA-OneVision-1.5-3B wins on 17 of 27 benchmarks. This isolates the vision encoder (RICE-ViT vs. Qwen-ViT) and data pipeline (concept-balanced mid-training + instruction data) as the sources of advantage, ruling out the LLM upgrade (Qwen2.5 → Qwen3) as the primary driver. However, the paper does not report this controlled comparison at the 8B scale, which is a missing ablation — without it, we cannot determine how much of the 8B model's performance comes from Qwen3-8B being a stronger LLM than Qwen2.5-7B's language backbone.

**Mid-training data scale (Figure 6):** 0M → 4M → 85M shows monotonic improvements across all ten benchmarks, with the largest gains on OCR tasks. The fact that 0M (no mid-training) still achieves non-trivial performance confirms that mid-training amplifies existing capabilities rather than creating them from nothing. The lack of a plateau at 85M suggests the paper's central claim — "simply scaling data at the mid-training stage alone can produce state-of-the-art LMMs" — is supported but not fully bounded: we don't know at what scale diminishing returns would set in.

**Concept balancing vs. random sampling (Figure 7):** 2M balanced outperforms 2M random on 25 of 27 benchmarks. This is a clean ablation controlling for data volume, confirming that the concept-balancing procedure (not just the data sources or training recipe) is causally responsible for improved performance. The two benchmarks where random matches balanced are not identified, which would be informative for understanding where concept coverage matters least.

**Instruction data merging (Figure 8):** Merged46M > LLaVA-OneVision-1.5-Inst-Data (22M) ≈ FineVision on nearly all benchmarks, confirming that scaling instruction data via merging produces benefits without negative interference. However, the doubled batch size for Merged46M confounds the comparison — the improvement could be due to larger batch size rather than additional data. A controlled ablation with identical batch size (processing 46M samples over twice the steps) is not provided.

**RL mode comparison: thinking vs. fast (Table 1, Figure 5):** Thinking mode substantially outperforms fast mode on reasoning (+6.0 vs. +1.0 average improvement over supervised baseline) but underperforms on perceptual tasks (Ref-L4 IoU decline, OCRBench slight regression). This reveals a capability tradeoff: the chain-of-thought reasoning that benefits math and science can interfere with direct visual perception. The paper's two-stage curriculum and interleaved anchor samples partially mitigate this but do not eliminate it.

**RL data selection strategy (qualitative, Section 5.1):** The paper describes discrepancy-driven selection (Pass@N vs. Pass@1 gap) and reward-based filtering (average reward within a "specified range") but provides no ablation comparing these selection strategies against random selection or other heuristics. The specific thresholds (N, the gap magnitude, the reward range) are not reported. This makes the RL data curation procedure not reproducible from the paper alone.

---

### Critical Assessment

#### Does the claim "LLaVA-OneVision-1.5-8B outperforms Qwen2.5-VL-7B on 18 of 27 benchmarks" demonstrate genuine superiority?

The claim is factually correct as stated — the 8B model leads on 18 benchmarks and trails on 9. However, a closer reading of Table 1 reveals that the superiority is **uneven and task-dependent**:

- **On General VQA (9 benchmarks):** LLaVA-OneVision-1.5-8B trails Qwen2.5-VL-7B on 8 of 9 benchmarks, with only SEED-Bench-2-Plus showing a tie. The 8B model is *systematically worse* at general visual question answering than the comparably-sized Qwen model. The paper does not discuss this pattern, instead focusing on the 18-of-27 headline and the average metrics that obscure this category-level weakness.

- **On OCR & Chart (7 benchmarks):** The two models are essentially tied (84.6% vs. 84.4% average), with mixed individual results. The paper's RICE-ViT advantage over SigLIP/Qwen-ViT shown in Table 2 does not translate to a consistent OCR advantage at the full LMM level against Qwen2.5-VL-7B — likely because Qwen2.5-VL's architecture and training pipeline compensate for its vision encoder's limitations through other mechanisms (e.g., dynamic resolution, more sophisticated projector).

- **On Multimodal Reasoning (6 benchmarks):** The 8B model shows a genuine advantage on MathVision (+3.2), MMMU-val (+4.1), and MMMU-Pro-standard (+1.1), but trails on MMMU-Pro-vision (-7.6). The advantage is not uniform across reasoning types — it is concentrated on math-heavy reasoning (MathVista, MathVision, WeMath) and STEM reasoning (MMMU), while the Qwen model excels on vision-heavy reasoning (MMMU-Pro-vision).

The claim "outperforms on 18 of 27" is therefore technically true but masks a more nuanced reality: LLaVA-OneVision-1.5-8B is **stronger at math and STEM reasoning, weaker at general visual understanding, and comparable at OCR** compared to Qwen2.5-VL-7B. This is a specific capability profile, not a blanket superiority.

---

#### Does the claim "LLaVA-OneVision-1.5-4B surpasses Qwen2.5-VL-3B on all 27 benchmarks" hold up?

Yes, and this is the paper's strongest and most unambiguous result. The 4B model leads on every single benchmark, often by substantial margins (e.g., MathVista 67.9 vs. 60.2, MMMU-val 52.7 vs. 46.4). The 4B model also surpasses the prior-generation 7B model on all reasoning benchmarks. This provides strong evidence that the paper's data pipeline and vision encoder are genuinely effective — the gains are too consistent and too large to be attributed to noise.

However, a careful reader will note that the 4B model uses **Qwen3-4B** as its LLM while Qwen2.5-VL-3B uses **Qwen2.5-3B** as its LLM. These are different language backbones — Qwen3 is a newer, presumably stronger model than Qwen2.5 (the Qwen3 technical report is cited as Team, 2025, suggesting it postdates Qwen2.5). The paper's same-LLM comparison in Appendix A (LLaVA-OneVision-1.5-3B with Qwen2.5-3B vs. Qwen2.5-VL-3B) shows a reduced but still substantial advantage (17 of 27). This means part of the 4B's advantage comes from the LLM upgrade, and part comes from the vision encoder and data pipeline. The paper cannot fully disentangle these factors because the 4B variant using Qwen2.5-4B (for a true same-LLM comparison) is not evaluated.

---

#### Does the concept-balancing ablation (Figure 7) demonstrate what it claims?

Yes — the 2M balanced vs. 2M random comparison is a clean, well-controlled ablation. Same data sources, same data volume, same training procedure, different sampling strategy. The balanced model wins on 25 of 27 benchmarks. This provides strong causal evidence that the concept-balancing procedure specifically improves performance.

However, the ablation is conducted at the 2M data scale, not the 85M scale used for the final model. The paper implicitly assumes that the benefits of concept balancing scale with data volume, but this is not directly tested — a 4M balanced vs. 4M random comparison or 85M balanced vs. 85M random comparison would strengthen the claim. It is possible (though unlikely) that the benefits of balancing diminish at larger scales because random sampling eventually covers the long tail through sheer volume.

---

#### Does the mid-training scaling experiment (Figure 6) demonstrate that "simply scaling data at the mid-training stage alone can produce state-of-the-art LMMs"?

The experiment shows monotonic improvement from 0M to 4M to 85M, which supports the claim that mid-training data scale matters. However, the claim contains a stronger assertion — that scaling *alone* (without complex training paradigms) *produces state-of-the-art performance*. The experiment compares 0M, 4M, and 85M within the LLaVA-OneVision-1.5 pipeline, but it does not compare against "complex training paradigms" as a control. To truly support the claim, the paper would need to show that a LLaVA-OneVision-1.5 trained with 85M mid-training data outperforms models trained with complex paradigms (multi-stage curricula, iterative refinement, progressive resolution training) at the same budget. The paper's comparisons are against Qwen2.5-VL (which uses a different architecture and data pipeline, not just a different training paradigm), so the specific claim about "eliminating the need for complex training paradigms" is an inference rather than a directly tested hypothesis.

---

#### Does the vision encoder comparison (Table 2) support the claim that RICE-ViT matches or exceeds SigLIP2 while being simpler?

Yes, with qualifications. At 378px resolution, RICE-ViT (ViT-L-14, single cluster discrimination loss) vs. SigLIPv2 (ViT-SO400M-14, four specialized losses) shows RICE-ViT leading on 9 of 14 benchmarks. The advantage is specific to OCR-heavy benchmarks (InfoVQA +4.4%, DocVQA +3.5%, ChartQA +4.9%), while general vision performance is comparable (70.1% vs. 69.5% average). This supports the claim that the single-loss approach matches or exceeds the multi-loss approach, particularly for the fine-grained capabilities that matter most for LMM tasks.

However, Table 2 uses the **LLaVA-NeXT framework** for all comparisons, which employs tiling (2×2+1) for high-resolution processing. This means the comparison tests how well each encoder's features work when *combined with tiling*, not how well they work natively. RICE-ViT's 2D RoPE is designed for native resolution support without tiling — the Table 2 evaluation may not fully capture this advantage because all encoders benefit from tiling. A comparison without tiling (using each encoder's native resolution handling) would better isolate RICE-ViT's resolution flexibility advantage.

Additionally, the encoder comparison uses Qwen2.5-7B as the LLM, not Qwen3. This means the Table 2 results reflect a different LLM backbone than the final LLaVA-OneVision-1.5 models, and the encoder rankings might shift if Qwen3 were used instead.

---

#### Does the RL post-training demonstrate elicitation rather than injection?

The paper's conceptual framing — RL as elicitation, not injection — is supported qualitatively by the pattern of results (large gains on reasoning where Pass@N is presumably high, small gains on perception where Pass@N is already close to Pass@1) but is **not empirically validated**. The paper never reports actual Pass@N and Pass@1 measurements for its supervised model, so we cannot verify that the selected RL instances indeed have a large discrepancy gap. The discrepancy-driven selection is described as a principle, but no quantitative evidence is provided that it works better than random selection or difficulty-based selection.

The RL experiments also lack several controls that would strengthen the elicitation claim:
- **No comparison of discrepancy-selected vs. randomly-selected RL data:** Without this ablation, we cannot determine whether the selection strategy matters.
- **No measurement of Pass@N before and after RL:** If RL is truly elicitation, we would expect Pass@1 to increase (probability mass shifts to correct paths) while Pass@N stays relatively constant (the model doesn't gain new capabilities). This prediction is not tested.
- **No breakdown of RL gains by instance difficulty:** The elicitation framing predicts that RL gains should be concentrated on medium-difficulty instances (where the discrepancy gap is large), with minimal gains on easy instances (already solved) or hard instances (no latent capability). This prediction is not tested.

The two-stage curriculum and the thinking-vs-fast mode comparison provide indirect evidence for the elicitation framing (the fact that thinking mode helps reasoning but hurts perception is consistent with probability mass being rearranged rather than capabilities being added), but these are consistent with multiple interpretations, not just elicitation.

---

#### Missing experiments that would have strengthened the paper

- **8B same-LLM comparison:** LLaVA-OneVision-1.5-8B with Qwen2.5-7B vs. Qwen2.5-VL-7B — this would isolate the vision encoder and data pipeline contributions at the 8B scale, where the current comparison confounds LLM quality (Qwen3 vs. Qwen2.5) with vision pipeline quality.

- **Budget scaling curve:** The paper makes a strong claim about achieving results under a $16,000 budget, but provides only one budget point. How does performance scale with budget? If the budget were $8,000 or $32,000, what would the performance be? Without a scaling curve, the $16,000 figure is a proof of existence rather than a characterization of the cost-performance tradeoff.

- **RL data selection ablation:** Discrepancy-driven selection vs. random selection vs. difficulty-based selection, with Pass@N measurements before and after RL for each. This would validate the paper's conceptual contribution to RL data curation.

- **85M concept-balanced vs. 85M random sampling:** The 2M balanced-vs-random ablation is strong, but the paper's final model uses 85M samples. A corresponding ablation at the 85M scale would confirm that concept balancing remains beneficial at the scale actually used.

- **Separate evaluation of the Chinese-language capability:** The mid-training dataset contains 20M Chinese samples, but all reported benchmarks appear to be English-language. The paper does not evaluate Chinese-language multimodal performance separately, making it impossible to assess whether the Chinese mid-training data improved Chinese capabilities or whether the benefits transferred primarily to English tasks.

- **Statistical significance or confidence intervals:** None of the 27 benchmark comparisons include confidence intervals, so we cannot distinguish between genuine differences and sampling noise, particularly for small-margin results (e.g., the 8B model's +0.1 on DocVQA, which is clearly within noise). Given that many comparisons show sub-2% differences, this is a genuine limitation for interpreting the results.

- **Inference cost analysis:** The paper carefully accounts for training cost but never discusses inference cost. RICE-ViT's native resolution support should reduce inference compute compared to tiling-based approaches, but this is never quantified. The RL model's "thinking" mode generates longer responses (chain-of-thought), which increases inference cost — this tradeoff between accuracy and latency is never discussed.

---

#### Summary of what the experiments do and do not establish

**What is well-established:**
- Concept-balanced mid-training data (85M samples) produces strong multimodal models, with balanced sampling outperforming random sampling at the 2M scale.
- RICE-ViT provides superior OCR-centric visual representations compared to CLIP, DFN, and SigLIPv2 at comparable resolutions.
- The 4B model outperforms Qwen2.5-VL-3B across all evaluated benchmarks.
- RL post-training substantially improves reasoning performance in thinking mode (+6.0 average), with the largest gains on the hardest reasoning tasks.
- Data scaling (mid-training, instruction tuning) produces monotonic improvements across diverse benchmarks.

**What is partially established but incompletely tested:**
- The 8B model's overall superiority to Qwen2.5-VL-7B — true on 18 benchmarks, but the model is systematically worse on General VQA tasks.
- The claim that "simply scaling data eliminates the need for complex training paradigms" — no direct comparison against complex paradigms under matched budgets is performed.
- The elicitation-vs-injection framing for RL — conceptually plausible and consistent with results, but not directly validated with Pass@N measurements or selection ablations.

**What is not established:**
- Whether the $16,000 budget is near-optimal — no budget scaling curve is provided.
- Whether concept balancing benefits persist at the 85M scale as they do at 2M.
- The model's Chinese-language multimodal capabilities, despite 20M Chinese mid-training samples.
- The statistical reliability of the 27-benchmark comparisons, given the absence of confidence intervals.

## 6. Limitations and Trade-offs

### The "18 of 27" Headline Masks Systematic Weakness on General Visual Understanding

**The assumption or constraint.** The paper's primary performance claim — "LLaVA-OneVision-1.5-8B outperforms Qwen2.5-VL-7B on 18 of 27 benchmarks" — aggregates results across four task categories (General VQA, Multimodal Reasoning, OCR & Chart, and Others) and counts benchmark-level wins without weighting by category size or practical importance. The implicit assumption is that benchmark-counting provides a fair summary of relative model capability.

**The consequence.** The aggregate claim obscures a stark category-level pattern that a practitioner would want to know about: on General VQA (9 benchmarks — the largest category, representing core multimodal understanding), LLaVA-OneVision-1.5-8B trails Qwen2.5-VL-7B on **8 of 9 benchmarks**, with only SEED-Bench-2-Plus showing a tie. The paper's own Table 1 shows these deficits: MMStar 67.7 vs. 68.3 (−0.6), MMBench-en 84.1 vs. 85.7 (−1.6), MMBench-cn 81.0 vs. 81.5 (−0.5), MME-RealWorld-en 61.7 vs. 63.3 (−1.6), MME-RealWorld-cn 56.1 vs. 56.3 (−0.2), SeedBench 77.3 vs. 77.6 (−0.3), CV-Bench 80.7 vs. 81.1 (−0.4), RealWorldQA 68.1 vs. 70.6 (−2.5). The "18 of 27" framing is technically true but misleading about the capability profile: the model is genuinely better at math and STEM reasoning (+4.1 on MMMU-val, +3.2 on MathVision) and genuinely worse at general-purpose visual understanding.

The consequence for a practitioner is that deploying this model for a general-purpose visual assistant — where users ask open-ended questions about everyday images — would yield systematically worse performance than deploying the comparably-sized Qwen2.5-VL-7B, which was designed and trained for that regime. The LLaVA-OneVision-1.5-8B is better suited to deployment scenarios weighted toward math, science, and technical diagram understanding rather than broad visual comprehension.

**What evidence exists in the paper.** Table 1, when read by category rather than by benchmark count, reveals this pattern directly. The paper does not discuss it — the text in Section 6.2 (General VQA) lists the model's absolute scores on individual benchmarks without comparing them to Qwen2.5-VL-7B's scores or acknowledging the systematic gap.

**Mitigation status.** Not addressed. The paper reports the benchmark-level comparison without category-level analysis and does not acknowledge this tradeoff. There is no discussion of why the model might underperform on General VQA — whether the concept-balanced mid-training overweights rare and technical concepts at the expense of common visual concepts, whether RICE-ViT's region-aware representations are less suited to holistic scene understanding than to fine-grained analysis, or whether the instruction-tuning mixture underrepresents general VQA formats. A practitioner choosing between these models would need to perform their own category-level analysis of Table 1, which the paper itself does not provide.

---

### Difficulty Estimation for RL Data Selection Is Described in Principle but Not Empirically Validated

**The assumption or constraint.** The RL post-training pipeline is built on a specific data selection strategy: curate training instances where the gap between Pass@N and Pass@1 is large, filtering out both too-easy instances (already solved) and too-hard instances (no latent capability to elicit). The paper frames this as a principled "discrepancy-driven data selection" approach and further refines it with reward-based filtering that retains only instances where average candidate rewards fall within a "specified range" (Section 5.1).

The implicit assumption is that (a) this selection strategy is causally responsible for RL's effectiveness, (b) the Pass@N/Pass@1 gap is measurable and meaningful for the specific model and tasks, and (c) the unspecified thresholds (N, the gap magnitude, the reward range) are set to values that produce an effective training distribution.

**The consequence.** None of these assumptions are tested. The paper provides no empirical evidence — zero measurement, zero ablation — that the discrepancy-driven selection outperforms simpler alternatives such as random selection from the same source datasets, uniform difficulty sampling, or simply using all available data. The specific N used to measure Pass@N, the gap threshold, and the reward range are never reported, making the procedure irreproducible from the paper alone.

Without these measurements, we cannot determine whether RL's gains (+7.9 on WeMath, +10.5 on MMMU-Pro-vision) are attributable to the selection strategy, to the GRPO algorithm, to the two-stage curriculum, to the AReaL asynchronous infrastructure, or to some interaction among them. A practitioner attempting to replicate the RL results would not know how to curate their own training data and would have no guidance on whether the selection strategy is essential or incidental.

The paper's conceptual framing — RL as elicitation rather than injection — is consistent with the results (large gains on reasoning where Pass@N is plausibly high, small gains on perception where Pass@1 is already high) but is never directly tested. The paper does not report a single Pass@N measurement for its supervised model, does not show Pass@N before and after RL, and does not break down RL gains by instance difficulty. The elicitation framing remains a plausible interpretation rather than an empirically validated mechanism.

**What evidence exists in the paper.** No direct evidence for the efficacy of the selection strategy exists. The paper describes the strategy qualitatively (Section 5.1) and shows that RL improves performance overall (Table 1, Figure 5), but there is no ablation comparing discrepancy-selected data to randomly selected data, no Pass@N measurements, and no sensitivity analysis on the selection thresholds. The reward-based filtering range is mentioned as a design choice but never numerically specified.

**Mitigation status.** Not addressed. The paper treats the data selection strategy as a completed design decision and evaluates only the endpoint (RL model performance vs. supervised baseline), not the intermediate steps. Section 5.1 acknowledges that the strategy "biases the corpus toward medium-difficulty instances that provide the most valuable learning signal," but this is stated as a claim about the strategy's effect, not as a tested hypothesis. The omission of selection ablations is a significant gap given that the paper positions discrepancy-driven selection as a conceptual contribution (see Innovation 3 in the prior analysis).

---

### The $16,000 Budget Figure Excludes Difficulty Estimation Cost and RL Inference Overhead

**The assumption or constraint.** The paper's central methodological commitment is to an explicit $16,000 compute budget that makes the results "democratized" and reproducible for resource-constrained researchers. The training cost accounting (Section 4.2) covers the mid-training stage (128 × A800 GPUs over 3.7 days ≈ 11,366 GPU-hours ≈ ~$15,900 at typical cloud pricing) and implies that Stages 1 and 2 (projector alignment on 558K samples, instruction tuning on 22M–46M samples) fit within the remaining ~$100 of the stated budget.

**The consequence.** Two significant costs are excluded from the $16,000 figure:

1. **Concept balancing requires a pretrained MetaCLIP-H/14-Full-CC2.5B encoder** to project images into a shared embedding space and compute top-K similarities against a 500K-concept vocabulary. The paper does not account for the cost of (a) acquiring or training this MetaCLIP model, (b) encoding 85M+ raw images through it to produce embeddings, and (c) encoding 500K concept entries. For 85M images at ViT-H/14 resolution, even inference-only encoding is non-trivial — at conservative throughput estimates (~100 images/second/GPU on an A100-class GPU), encoding 85M images would require approximately 236 GPU-hours, adding roughly $330 to the budget. Encoding the 500K concept entries adds additional (smaller) cost. While this is a one-time preprocessing cost rather than a per-run training cost, it is real compute that must be spent to produce the dataset, and excluding it makes the $16,000 figure a lower bound.

2. **RL inference during post-training is not budgeted.** The AReaL framework decouples rollout generation from training — rollout workers continuously generate responses from the current policy while trainer workers update the model. This generation requires running the full 8B model in inference mode to produce multiple candidate responses per training instance. The paper describes the RL stage as "lightweight" but provides no GPU-hour accounting for the inference side of the asynchronous system. For a 67K-instance corpus with (presumably) multiple rollout generations per instance across multiple training iterations, this inference cost is non-trivial. Excluding it makes the RL results not directly comparable to the supervised results in budget terms.

For a practitioner attempting to replicate the full pipeline, the actual cost is $16,000 + concept-balancing preprocessing + RL inference, with the latter two components potentially adding hundreds to low thousands of dollars depending on implementation efficiency.

**What evidence exists in the paper.** Section 3.1 describes the MetaCLIP-based concept matching pipeline without cost accounting. Section 3.2 describes difficulty estimation for RL data selection without quantifying the cost of generating the multiple candidate responses needed to measure Pass@N and perform reward-based filtering. Section 5.3 describes the AReaL asynchronous system and mentions that it "significantly improves GPU utilization compared to synchronous implementations" without providing absolute GPU-hour figures for the RL stage.

**Mitigation status.** Partially addressed by transparency. The paper is explicit about the training infrastructure (128 A800 GPUs, 3.7 days) and the offline data packing compression ratio (11×), which enables independent verification of the core training cost. The paper does not explicitly claim that $16,000 covers preprocessing or RL inference — it states that the training framework "enables the training of LLaVA-OneVision-1.5 within a $16,000 compute budget" (Section 1), which is technically compatible with preprocessing and RL inference being additional costs. However, the paper does not flag these exclusions, and a reader unfamiliar with the practical details of data preprocessing and RL infrastructure would reasonably interpret $16,000 as the total cost of producing the final model, which it is not.

---

### All Evaluations Are on Single-Turn Academic Benchmarks with No Deployment-Relevant Metrics

**The assumption or constraint.** The entire evaluation suite — 27 benchmarks in the main comparison plus additional coding and grounding benchmarks in Figure 5 — consists of single-turn, static benchmarks with known answer formats, ground-truth labels, and automated scoring. The paper implicitly assumes that performance on these benchmarks is a sufficient proxy for real-world multimodal capability and that the ranking of models on benchmarks transfers to deployment scenarios.

**The consequence.** Several deployment-relevant properties are completely unmeasured:

- **Multi-turn interaction and instruction following over extended conversations.** All benchmarks present a single image and a single question, with the model producing a single answer. This evaluates zero-shot instruction following but provides no signal about how the model behaves in multi-turn dialogues where context accumulates, the user asks follow-up questions about the same image, or the task requires the model to maintain coherent visual reference across turns.

- **Calibration and refusal behavior.** Benchmarks measure top-1 accuracy — whether the model's most likely answer is correct. They do not measure whether the model knows when it is uncertain (calibration), whether it appropriately refuses to answer when it lacks information, or whether it produces plausible-sounding but incorrect responses (hallucination) on out-of-distribution inputs. A model that achieves 70% accuracy on a benchmark but confidently hallucinates on 15% of its wrong answers may be less deployable than a model with 65% accuracy that reliably expresses uncertainty.

- **Latency, throughput, and inference cost.** The paper carefully accounts for training cost but provides zero inference-cost measurements. RICE-ViT's native resolution support with 2D RoPE should reduce inference compute compared to tiling-based approaches (which process the image multiple times at different resolutions), but this is never quantified. The RL model's "thinking" mode generates substantially longer outputs (chain-of-thought reasoning traces), which increases both latency and per-query inference cost — a tradeoff the paper reports in accuracy terms (+6.0 average reasoning gain) but never translates to latency or cost terms. A practitioner choosing between the fast and thinking modes needs to know whether the accuracy gain justifies the latency increase.

- **Robustness to input perturbations.** The benchmarks evaluate clean, curated test examples. There is no evaluation of robustness to common real-world variations: different image resolutions, compression artifacts, rotated or cropped images, noisy captions, adversarial inputs, or distribution shifts between training and deployment data.

**What evidence exists in the paper.** None. The paper provides no latency measurements, no calibration curves, no multi-turn evaluations, and no robustness testing. The evaluation is entirely within the LMMs-Eval framework's standard single-turn benchmark protocol.

**Mitigation status.** Not addressed. The paper presents the benchmark results as the primary evidence of model quality (Section 6, Table 1, Figure 5) and draws deployment-relevant conclusions from them (the model "empowers the community to build specialized applications" — Section 7). The gap between benchmark performance and deployment readiness is not discussed. This is a standard limitation in the LMM literature (most papers evaluate only on academic benchmarks), but it is particularly relevant here because the paper's explicit framing is about democratizing access to *practical* multimodal AI, making the absence of deployment-relevant metrics more consequential than it would be for a purely methodological contribution.

---

### The 8B Model's Gains Over Qwen2.5-VL-7B Are Partially Attributable to the Stronger Qwen3 Language Backbone, and This Is Not Controlled For

**The assumption or constraint.** The paper compares LLaVA-OneVision-1.5-8B (using Qwen3-8B as the language model) against Qwen2.5-VL-7B (using Qwen2.5-7B as the language model). The implicit assumption in presenting these as comparable is that the language model quality difference (Qwen3 vs. Qwen2.5) is not a major driver of the performance difference, and that the vision encoder (RICE-ViT vs. Qwen-ViT) and data pipeline are the primary differentiators.

**The consequence.** This assumption is untestable with the data provided. Qwen3 is a newer, presumably stronger language model than Qwen2.5 — the Qwen3 technical report (Team, 2025) is cited but not characterized in the paper, so we do not know the magnitude of the text-only capability gap between these two LLMs. A meaningful fraction of LLaVA-OneVision-1.5-8B's performance on any given benchmark could come from the LLM being better at reasoning, reading comprehension, or mathematical computation — capabilities that have nothing to do with the vision encoder or the multimodal data pipeline.

The Appendix A experiment partially acknowledges this concern by providing a same-LLM comparison at the 3B scale: LLaVA-OneVision-1.5-3B with Qwen2.5-3B vs. Qwen2.5-VL-3B. This controlled comparison shows a reduced but still substantial advantage (17 of 27 benchmarks), confirming that the vision encoder and data pipeline contribute genuine value beyond the LLM difference. However, the paper does not report the analogous comparison at the 8B scale — there is no LLaVA-OneVision-1.5-8B variant using Qwen2.5-7B (to match Qwen2.5-VL-7B's LLM), nor is there a LLaVA-OneVision-1.5-7B variant that would be more directly parameter-matched. The 8B model is compared against a 7B model with a different LLM family, and the LLM quality difference is a confounding variable.

The practical consequence: a practitioner deciding between these models cannot determine how much of the 8B model's advantage comes from the vision pipeline (which is the paper's contribution) versus the LLM upgrade (which is available to any model builder by simply swapping Qwen2.5 for Qwen3). The paper's claim that its data and training pipeline produce state-of-the-art multimodal models is partially conflated with the independent improvement in language model quality between Qwen2.5 and Qwen3.

**What evidence exists in the paper.** The same-LLM comparison at the 3B scale (Appendix A, Figure 9) provides strong evidence that the vision pipeline contributes genuine value independent of LLM quality. However, the 8B comparison (which is the headline result — "LLaVA-OneVision-1.5-8B outperforms Qwen2.5-VL-7B on 18 of 27 benchmarks") has no same-LLM control. The paper lists Qwen3 as the language backbone in Section 2.1 but does not discuss the implications of comparing Qwen3 against Qwen2.5.

**Mitigation status.** Partially addressed at the 3B scale but not at the 8B scale. The Appendix A experiment shows awareness of the LLM-confounding issue, but the paper does not extend this control to the primary comparison or acknowledge that the 8B-vs-7B comparison confounds LLM quality with vision pipeline quality. A future practitioner interested in the pure contribution of the vision pipeline would need to train a LLaVA-OneVision-1.5-8B with Qwen2.5-7B (keeping the parameter count at 7B for a fully matched comparison) to isolate the effect — an experiment the paper does not provide.

---

### The RL Post-Training Introduces a Fundamental Accuracy-vs-Perception Tradeoff That the Paper Does Not Quantify or Resolve

**The assumption or constraint.** The RL post-training uses a two-stage curriculum (answer-only warmup → chain-of-thought RL with interleaved anchor samples) designed to develop reasoning capabilities while maintaining performance on perception-heavy tasks. The implicit assumption is that this curriculum successfully mitigates the tradeoff between reasoning depth and perceptual precision.

**The consequence.** The mitigation is partial, and the tradeoff persists in the final model, but the paper does not systematically characterize it. The evidence is scattered across Table 1 and Figure 5:

- **Thinking mode helps reasoning but hurts OCR and perception.** In thinking mode, the RL model gains +7.9 on WeMath, +8.8 on MathVision, and +10.5 on MMMU-Pro-vision over the supervised baseline — dramatic improvements. But on OCR & Chart benchmarks, thinking mode averages 83.3% vs. the supervised model's 84.6% and the fast-mode RL model's 84.6% — a clear regression. On Ref-L4 (IoU), a grounding benchmark in Figure 5, thinking mode scores approximately 64% vs. the SFT baseline's approximately 81% — a 17-percentage-point drop on a core perceptual metric.

- **Fast mode partially recovers perception but leaves most reasoning gains on the table.** In fast mode, the RL model achieves OCR & Chart parity with the supervised model (84.6% both) and improves spatial reasoning slightly (Ref-L4 IoU ~87% vs. ~81% SFT baseline in Figure 5). But the reasoning gains largely disappear: WeMath regresses slightly (−0.7), MathVision gains only +0.6, and MMMU-Pro-vision gains +3.8 compared to +10.5 in thinking mode. The average reasoning improvement in fast mode (+1.0) is one-sixth of the thinking mode gain (+6.0).

This is a genuine tradeoff that a practitioner must navigate: **do you want the model that excels at math and science reasoning but is worse at reading documents and localizing objects (thinking mode), or the model that maintains strong perception but gains little on reasoning (fast mode)?** The paper provides no unified model that achieves both, and no guidance on how to select between modes based on task type.

The interleaved anchor samples in Stage 2 are described as "a small proportion" (Section 5.3) but the exact ratio is not specified, and no ablation tests whether a different ratio would shift the tradeoff. It is possible that the regression on perception in thinking mode is fixable with more aggressive interleaving or a different curriculum design, but the paper does not explore this.

**What evidence exists in the paper.** Table 1 shows the OCR regression in thinking mode (83.3% vs. 84.6%) and the minimal reasoning gains in fast mode (+1.0 vs. +6.0). Figure 5 shows the Ref-L4 IoU collapse in thinking mode and the Design2Code/UniSVG advantage in thinking mode — different tasks pull in opposite directions. The paper acknowledges the Ref-L4 IoU decline briefly (Figure 5 caption: "verbose generation may occasionally interfere with precise coordinate regression") but does not treat the tradeoff as a systematic limitation of the RL approach.

**Mitigation status.** Partially addressed but not resolved. The two-stage curriculum with interleaved anchors is presented as the mitigation, but the evidence shows it does not eliminate the tradeoff — it only provides two operating points (fast mode for perception-optimized, thinking mode for reasoning-optimized) rather than a single Pareto-optimal model. The paper does not explore whether further curriculum adjustments, different reward designs (e.g., adding a brevity penalty or perception-specific reward terms), or architectural modifications (e.g., separate reasoning and perception heads) could bridge the gap. For a practitioner, the current state of the RL model requires an explicit choice between reasoning and perception that the paper's headline RL gains (+10.5 on MMMU-Pro-vision) do not capture.

## 7. Implications and Future Directions
- Field impact:
  - Demonstrates that a carefully engineered open pipeline—region-aware vision encoder, concept-balanced mid-training, and efficient offline packing—can match or surpass contemporary open baselines with lower cost and data friction. This lowers the barrier for labs and startups to train capable LMMs from scratch.
- Enabled research:
  - Reproducible mid-training datasets and scripts allow the community to:
    - Probe scaling laws for Stage-1.5 and instruction tuning (Figures 4 & 6).
    - Experiment with alternative encoders or loss functions within the same pipeline (Table 2).
    - Explore concept balancing for other modalities (e.g., video, audio) using feature-based retrieval.
- Practical applications:
  - Strong results on document OCR and chart understanding (Table 1; OCR & Chart avg 85.0 for 8B) suggest immediate utility in enterprise document processing, business intelligence, and scientific literature analysis.
  - General VQA and ScienceQA improvements imply broader applicability in education, knowledge assistance, and multimodal search.
- Future directions suggested by the paper’s findings:
  - Add RL or preference alignment to improve reward-based and grounding-sensitive benchmarks (Abstract; Table 1, “Others”).
  - Expand concept vocabularies and multilingual coverage, perhaps learning concept spaces jointly with the model.
  - Push native-resolution reasoning further (RICE-ViT + 2D RoPE) for ultra-high-resolution documents and complex charts, and extend to multi-image or video scenarios.
  - Integrate tool use (OCR post-processing, retrieval) and test-time scaling strategies to address the few lagging benchmarks (e.g., MMMU-Pro-vision).

Overall, LLaVA-OneVision-1.5 offers a clear, reproducible path to high-quality open LMMs: a region-aware encoder, a caption-light concept-balancing strategy to build a massive yet diverse pretraining set, and an efficiency-first training framework. The evidence across Tables 1–2 and Figures 3–6 supports both performance and cost-effectiveness, while also highlighting where additional alignment or grounding could yield further gains.
