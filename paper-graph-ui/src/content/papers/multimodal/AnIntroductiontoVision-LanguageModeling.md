# An Introduction to Vision-Language Modeling

**ArXiv:** [2405.17247](https://arxiv.org/abs/2405.17247)

## 🎯 Pitch

This paper delivers a unified, practice-driven introduction to Vision-Language Models (VLMs), mapping out the core training paradigms—contrastive, masking, generative, and pretrained-backbone—and providing hands-on guidance for efficient data curation, model alignment, and responsible evaluation. By critically synthesizing what works (and what can go wrong), it empowers practitioners and newcomers to build and assess more reliable, real-world VLMs, addressing persistent challenges like spatial reasoning, prompt adherence, and hallucination that currently limit the technology’s impact.

---

## 1. Executive Summary

This paper provides a pedagogical introduction to vision-language models (VLMs), surveying the dominant training paradigms—contrastive training (CLIP, SigLIP), masking objectives (FLAVA, MaskVLM), generative approaches (CoCa, Chameleon), and pretrained-backbone methods (Frozen, MiniGPT)—alongside practical guidance on data curation, alignment, grounding, and evaluation. The work catalogs critical but often overlooked evaluation pitfalls, including benchmarks solvable via blind language priors and a warning that models outputting uniform probabilities can achieve 100% on binary-choice benchmarks when the correct answer is placed first due to argmax behavior. The paper establishes that current VLMs fail at spatial reasoning beyond random chance on synthetic benchmarks like PUG, revealing a fundamental gap between surface-level visio-linguistic alignment and genuine compositional understanding.

## 2. Context and Motivation

### The Core Problem: The Field of VLMs Needs an Accessible Entry Point, Not Just Another Survey

The fundamental challenge this paper addresses is not a technical gap but a **pedagogical and structural one**: the rapidly growing field of vision-language modeling has become fragmented, difficult to navigate, and inaccessible to newcomers. As the authors state in Section 1, while "several works have already extended large language models to vision, connecting language to vision is not completely solved." The result is a research landscape where training paradigms proliferate, evaluation practices are inconsistent, and critical methodological pitfalls are often discovered only through costly trial and error.

The problem is not that VLM research lacks technical depth—it is that the field has accumulated so much complexity so quickly that **the barrier to entry has become prohibitive**. A researcher entering the field in 2024 confronts a bewildering array of choices: contrastive or generative? Masking or pretrained backbones? CLIP-style scaling or parameter-efficient fine-tuning? Each choice carries downstream consequences for data requirements, compute budgets, model capabilities, and evaluation protocols, yet these tradeoffs are rarely articulated clearly in individual papers, which naturally focus on demonstrating the superiority of their proposed method rather than providing a comparative landscape.

This gap is significant for several practical reasons the paper implies throughout its structure but does not always spell out explicitly:

- **Resource allocation decisions**: A junior researcher deciding whether to train CLIP from scratch (requiring hundreds of GPUs and hundreds of millions of image-text pairs, as noted in Section 3.2.2—"using between 256 and 600 GPUs across multiple days or weeks") versus fine-tuning a pretrained LLM backbone (which can be done on "four A100 GPUs for around ten hours," Section 2.5.2) needs guidance on what each path enables and what it sacrifices. The paper aims to provide exactly this kind of comparative framing.

- **Evaluation validity**: The paper documents numerous cases where standard benchmarks produce misleading results. As Section 4.1.6 warns, many binary-choice benchmarks place the correct answer first, and because PyTorch's `argmax` returns the first element when probabilities are equal, "a model whose parameters are all equal to zero could achieve 100% accuracy in these benchmarks." Without this knowledge, a new researcher could easily misinterpret a catastrophic model failure as state-of-the-art performance. The paper positions itself as a guardian against such interpretive errors.

- **Understanding failure modes**: Section 4.1.8 reports that on the synthetic PUG benchmark, "current VLMs are not performing better than random chance when evaluating spatial relations." This finding—that models achieving impressive results on standard benchmarks collapse to chance-level performance on controlled spatial reasoning—highlights a fundamental gap between surface-level visio-linguistic alignment and genuine compositional understanding. The paper contextualizes such failures as systematic rather than anecdotal, giving newcomers a framework for interpreting model limitations.

### Why Existing Surveys Don't Fully Address This Gap

The paper explicitly acknowledges that "several complete and more technical surveys on VLMs" exist (Section 1), citing work by Zhang et al. [2024a], Ghosh et al. [2024], Zhou and Shimada [2023], Chen et al. [2023a], Du et al. [2022], Uppal et al. [2022], and Liang et al. [2024]. These surveys serve an important function—they are comprehensive, technically detailed, and heavily cited—but they are written for **domain experts who already possess the conceptual vocabulary of the field**.

The authors identify a specific type of reader that existing literature fails to serve: "students or researchers in other areas who want to enter the field" (Section 1). For this reader, the existing surveys present several barriers:

- **Conceptual overload**: A comprehensive survey might present twenty different VLM architectures in sequence without first establishing the conceptual axes that distinguish them—contrastive vs. generative, masked vs. autoregressive, trained from scratch vs. assembled from pretrained components. The reader is left memorizing model names rather than understanding design principles.

- **Missing meta-knowledge**: Standard surveys describe what models do and how well they perform, but they rarely capture the tacit knowledge that experienced practitioners possess: which decisions matter most (Section 3.2.4: "image resolution, visual encoder capacity, and visual pretraining data are the choices that most impact model performance"), how to allocate a limited GPU budget (Section 3.2.2), or which benchmarks are vulnerable to shortcut solutions (Section 4.2.3).

- **Insufficient connective tissue**: Sections 2.2.1 and 2.2.2 discuss CLIP and SigLIP, but a reader unfamiliar with contrastive learning would not understand *why* these models differ—the key distinction is that CLIP uses InfoNCE (a multi-class softmax over negative pairs within a batch, Equation 2), while SigLIP uses the original NCE formulation (a binary cross-entropy per pair, Equation 1). This difference explains why SigLIP achieves "better 0-shot performances on smaller batch sizes than CLIP" (Section 2.2.2)—because binary classification is less batch-dependent than multi-class softmax. A standard survey might list both models; this paper aims to explain what the distinction means.

### The Paper's Positioning: A Pedagogical Introduction, Not a Survey

The paper positions itself quite explicitly by describing what it is **not**: "This work should not be considered as a survey or a complete guide on VLMs. Hence, we do not aim to cite every work from the VLM research field; nor does this work capture every best practice in this space" (Section 1). This is an unusual and deliberate framing choice. By explicitly declining to be comprehensive, the paper claims a different kind of authority—not the authority of exhaustive coverage, but the authority of **curated, explanatory depth**.

This positioning has consequences for how the paper should be read:

**Selection is deliberate, not arbitrary.** When the paper chooses to present CLIP (Section 2.2.1) as the representative contrastive model rather than, say, ALIGN (Jia et al., 2021) or BASIC (Pham et al., 2021), it is making a pedagogical decision: CLIP's architecture is simpler, its training objective is cleaner to derive from first principles (which the paper does, walking from energy-based models through NCE to InfoNCE in Section 2.2), and its downstream impact has been more pervasive. The paper is not claiming CLIP is the best contrastive VLM—only that it is the best one to *teach from*.

**The taxonomy creates the narrative.** Categorizing VLMs into four families (contrastive, masking, generative, pretrained backbones) is not an objective classification—as the paper acknowledges, "these paradigms are not mutually exclusive; many approaches rely on a mix of contrastive, masking, and generative criteria" (Section 2). The taxonomy is a **teaching device**. It creates a conceptual map that helps newcomers locate any specific model within a broader design space, even if the boundaries are fuzzy at the edges.

**Practical guidance is prioritized over theoretical completeness.** Sections 3.2.2–3.2.3 offer concrete advice about GPU requirements ("training a contrastive model like CLIP on hundreds of millions of images from scratch should not require more than 64 GPUs"), software tools (torch.compile, xformers, FFCV), and data loading bottlenecks ("data loading often becomes a bottleneck that significantly slows down training"). This type of operational knowledge is almost never found in research papers or technical surveys, yet it is precisely what a new practitioner needs to know to go from reading to doing. The paper positions this practical knowledge as equally valuable to the conceptual taxonomy.

### The Information-Theoretic Thread: A Unifying Lens

One subtle but important contribution is the paper's use of information theory (Section 2.3.3) to provide a **unified mathematical perspective on seemingly disparate training objectives**. Building on Federici et al. [2020] and Dubois et al. [2021], the paper shows that masking, contrastive learning, and auto-encoding can all be understood as solutions to the same rate-distortion problem (Equation 3):

$$\arg \min_{p(z|x)} I(f(X); Z) + \beta \cdot H(X|Z)$$

where $I(f(X); Z)$ is the rate (how much information the representation $Z$ retains about the transformed input $f(X)$) and $H(X|Z)$ is the distortion (how much uncertainty about $X$ remains given $Z$). The key insight is:

- **Masking VLMs** implement the rate term through an entropy bottleneck (the amount of information removed by masking is bounded), and the distortion term through reconstruction (auto-encoding).
- **Contrastive losses** implement the distortion term through the InfoNCE classification objective, which "retains the necessary information by classifying which $Z$ is associated with an equivalent example $X$" (Section 2.3.3).

This framing unifies methods that are typically presented as entirely different paradigms. For a newcomer who might wonder whether to use masking or contrastive training, the rate-distortion view reframes the question: both methods are optimizing the same fundamental tradeoff between compression and information preservation, but they implement the distortion term differently. This theoretical unification is not a novel contribution of the paper—it builds explicitly on Federici et al. and Dubois et al.—but the paper's pedagogical value lies in making this connection accessible and showing how it maps onto practical model choices.

### The Companion to the Executive Summary: What This Section Adds

The executive summary identified what the paper *does*: catalog training paradigms, provide practical guidance, warn about evaluation pitfalls. This context section explains *why* these contributions matter: because the field has grown faster than its pedagogical infrastructure, leaving newcomers without a roadmap. The paper fills this gap not by competing with comprehensive surveys but by serving a qualitatively different function—teaching principles rather than cataloging models, exposing pitfalls rather than reporting benchmarks, and connecting practical decisions to theoretical foundations.

The remainder of this analysis will trace how the paper executes this pedagogical mission across training paradigms (Section 2), practical guidance (Section 3), and evaluation (Sections 4–5), always attending to what the paper chooses to explain in depth, what it omits, and what the implications of those choices are for the reader it aims to serve.

## 3. Technical Approach

### 3.1 Reader Orientation

This paper is a **pedagogical introduction**, not a traditional research paper proposing a novel method. The "system" being described is the **conceptual and practical ecosystem of vision-language models** — the set of training paradigms, architectural choices, data strategies, and evaluation protocols that constitute the VLM landscape as of early 2024. The core idea is that by organizing this landscape around a few clean conceptual axes (contrastive vs. generative, masking vs. autoregressive, trained from scratch vs. assembled from pretrained components), and by explicitly surfacing the tacit practical knowledge that experienced researchers possess, the paper can dramatically lower the barrier to entry for newcomers while also helping established practitioners avoid common pitfalls. The approach is **taxonomic, explanatory, and cautionary** — it builds a mental model of how VLMs work at the level of training objectives and architectural decisions, then uses that model to explain why certain design choices matter, why certain benchmarks are misleading, and where current models systematically fail.

### 3.2 Big-Picture Architecture (Diagram in Words)

The paper does not propose a single architecture but rather describes a **design space** spanning four major families of VLMs, a **training pipeline** common across families, and an **evaluation infrastructure** for assessing model capabilities. The major components are:

1.  **Training Paradigm Taxonomy (Section 2):** Four conceptual families — contrastive (CLIP, SigLIP), masking (FLAVA, MaskVLM), generative (CoCa, Chameleon, diffusion-based classifiers), and pretrained-backbone (Frozen, MiniGPT, BLIP-2) — each defined by their core training objective and architectural composition. These are not mutually exclusive; most modern VLMs mix objectives.
2.  **Unifying Mathematical Framework (Section 2.2–2.3):** A progression from energy-based models through Noise Contrastive Estimation to InfoNCE, and a parallel rate-distortion view that frames contrastive and masking objectives as different implementations of the same compression-preservation tradeoff (Equation 3).
3.  **Training Pipeline (Section 3):** A sequential recipe spanning data curation (heuristic filtering, CLIPScore ranking, diversity balancing, synthetic captioning, data augmentation, interleaved data construction), software infrastructure (OpenCLIP, torch.compile, xformers, FFCV), model selection decision criteria (when to use contrastive vs. masking vs. generative vs. pretrained backbone), grounding improvement techniques (bounding boxes, negative captioning), alignment via instruction tuning and RLHF (LLaVA family), text-rich image understanding strategies (high-resolution patching, decoupled OCR modules), and parameter-efficient fine-tuning methods (LoRA variants, prompt-based, adapter-based, mapping-based).
4.  **Evaluation Infrastructure (Section 4):** A multi-dimensional assessment framework covering visio-linguistic abilities (image captioning, text-to-image consistency, VQA, text-centric VQA, zero-shot classification, compositional reasoning, dense captioning, synthetic data evaluation), bias measurement (classification-based and embedding-based methods), hallucination detection (CHAIR, POPE, model-based evaluation), memorization testing (déjà vu memorization via k-nearest neighbor tests), and red teaming protocols for safety assessment.
5.  **Video Extension Framework (Section 5):** A parallel taxonomy for video-language models, covering early fusion approaches (VideoBERT, MERLOT), generative video-text models (VideoOFA), and pretrained LLM-based methods (Video-LLaMA, MiniGPT4-Video), alongside video-specific challenges in temporal understanding, data scarcity, and evaluation methodology.

Information flows as follows: a practitioner enters the design space → selects a training paradigm based on resource constraints and desired capabilities (Section 3.3) → curates training data using filtering and augmentation strategies (Section 3.1) → trains the model, potentially improving grounding and alignment through additional fine-tuning stages (Sections 3.4–3.7) → evaluates the resulting model across multiple dimensions, being careful to avoid known benchmark pitfalls (Section 4) → optionally extends the approach to video (Section 5).

### 3.3 Roadmap for the Deep Dive

- **First,** the contrastive training paradigm (CLIP) and its mathematical foundation — from energy-based models through NCE to InfoNCE — since contrastive learning is the most widely used VLM objective, the simplest to derive from first principles, and the foundation upon which many subsequent methods build.
- **Second,** the masking paradigm (FLAVA, MaskVLM) and the information-theoretic rate-distortion view that unifies masking with contrastive learning under a common compression-preservation framework.
- **Third,** the generative paradigm, including autoregressive models (CoCa, Chameleon), diffusion models, and the counter-intuitive use of generative models as zero-shot classifiers via Bayes' theorem — because generative methods represent the current frontier of VLM capability but require fundamentally different training infrastructure.
- **Fourth,** the pretrained-backbone paradigm (Frozen, MiniGPT, BLIP-2) — because this approach minimizes computational cost by learning only a mapping between frozen pretrained encoders, making it the most accessible entry point for resource-constrained practitioners.
- **Fifth,** the training pipeline in its full operational detail, from data curation through alignment to parameter-efficient fine-tuning — because understanding *how* to train is equally important to understanding *what* to train.
- **Sixth,** the evaluation framework and its documented pitfalls — because benchmark interpretation errors can lead researchers to fundamentally incorrect conclusions about model capabilities, and the paper's warnings about blind language priors and argmax artifacts are among its most practically valuable contributions.

### 3.4 Detailed, Sentence-Based Technical Breakdown

This is a **pedagogical survey paper** whose core contribution is not a novel technique but a structured, accessible introduction to the VLM field that surfaces tacit practical knowledge, connects disparate methods through unified mathematical frameworks, and documents evaluation pitfalls that are typically learned only through painful experience.

---

#### The Contrastive Training Paradigm: From Energy-Based Models to CLIP

The paper builds the contrastive learning framework from first principles, starting with Energy-Based Models (EBMs) and progressing through Noise Contrastive Estimation (NCE) to the InfoNCE loss that powers CLIP. This progression is pedagogically motivated: it shows that contrastive learning is not an arbitrary objective but rather a computationally tractable approximation to maximum likelihood estimation in energy-based models, where the intractable normalization constant is avoided by reframing the problem as discrimination between real and noise samples.

##### Energy-Based Models and the Maximum Likelihood Objective

An Energy-Based Model (EBM) is a function $E_\theta(x)$ parameterized by $\theta$ that assigns a scalar energy to each input $x$. The model is trained so that desirable inputs (those from the true data distribution) receive low energy, while undesirable inputs receive high energy. The probability of an input under the model is given by the Boltzmann distribution:

$$p_\theta(x) = \frac{e^{-E_\theta(x)}}{Z_\theta}$$

where $Z_\theta = \sum_x e^{-E_\theta(x)}$ is the normalization factor (also called the partition function) that ensures $p_\theta(x)$ sums to 1 over all possible $x$.

**What it computes:** a properly normalized probability distribution over inputs, where the probability of $x$ is proportional to the exponential of negative energy. Lower energy means higher probability; higher energy means lower probability.

**Why this form:** the exponential ensures non-negative probabilities. The minus sign maps energy (where 0 is low and positive is high) to probability (where high is likely and near-zero is unlikely). The normalization factor $Z_\theta$ is the mathematical price of having a valid probability distribution: without it, $e^{-E_\theta(x)}$ could be any positive number, but probabilities must sum to 1.

The ideal training objective is maximum likelihood:

$$\arg \min_\theta \mathbb{E}_{x \sim P_D(x)}[-\log p_\theta(x)]$$

where $P_D$ is the true data distribution. The gradient of this objective decomposes into two terms:

$$\frac{\partial \mathbb{E}_{x \sim P_D(x)}[-\log p_\theta(x)]}{\partial \theta} = \mathbb{E}_{x^+ \sim P_D(x)}\left[\frac{\partial E_\theta(x^+)}{\partial \theta}\right] - \mathbb{E}_{x^- \sim P_\theta(x)}\left[\frac{\partial E_\theta(x^-)}{\partial \theta}\right]$$

where $x^+$ is a positive sample from the data distribution and $x^-$ is a negative sample from the model's own distribution $P_\theta$.

**What it computes:** the gradient of the negative log-likelihood. The first term pushes down the energy of real data points (makes them more likely). The second term pushes up the energy of samples from the model's current distribution (makes the model's own "fantasies" less likely). The learning process is a competition: the model must simultaneously make real data probable while making its own generated samples improbable.

**Why this form:** this is the standard gradient of log-likelihood for energy-based models. The challenge is the second term: sampling from $P_\theta(x)$ requires computing the intractable normalization constant $Z_\theta$, which involves summing over all possible inputs.

##### Noise Contrastive Estimation (NCE): Avoiding the Intractable Partition Function

The key practical problem is that $\mathbb{E}_{x^- \sim P_\theta(x)}[\cdot]$ requires sampling from the model distribution, which is intractable because $Z_\theta$ involves a sum over all possible $x$. NCE [Gutmann and Hyvärinen, 2010] proposes a radical simplification: instead of using samples from the model distribution, use samples from a known noise distribution $p_n(u')$.

The NCE framework reformulates the problem as binary classification: train a model to discriminate between samples from the real data distribution (label $C=1$) and samples from the noise distribution (label $C=0$). The loss is standard binary cross-entropy:

$$\mathcal{L}_{\text{NCE}}(\theta) := -\sum_i \log P(C_i=1 \mid x_i; \theta) - \sum_j \log P(C_j=0 \mid x_j; \theta)$$

where $x_i$ is sampled from the data distribution and $x_j \sim p_n(x)$ is sampled from the noise distribution.

**What it computes:** the cross-entropy of a binary classifier that must distinguish real data from noise. The first sum penalizes the model for failing to recognize real data as real. The second sum penalizes the model for failing to recognize noise as noise.

**Why this form:** the classifier $P(C=1 \mid x; \theta)$ can be expressed in terms of the model's unnormalized density $e^{-E_\theta(x)}$ and the noise distribution $p_n(x)$ without ever computing the intractable $Z_\theta$. The binary classification problem is tractable because it only requires comparing the model's score against the known noise distribution, not against all possible inputs.

The paper notes an important caveat: "Even if it can be theoretically difficult to justify why such an approach might work, there is ample empirical evidence of the success of NCE-based methods in recent Self-Supervised Learning (SSL) literature" (Section 2.2). This is an honest acknowledgment: NCE is a theoretically imperfect approximation, but it works remarkably well in practice.

##### InfoNCE: The Loss Function Behind CLIP

Wu et al. [2018] introduced NCE without positive pairs using a non-parametric softmax with explicit normalization and a temperature parameter $\tau$. Oord et al. [2018, CPC] kept the non-parametric softmax while adding positive pairs, naming this formulation InfoNCE:

$$\mathcal{L}_{\text{InfoNCE}} = -\sum_{(i,j) \in \mathcal{P}} \log\left(\frac{e^{\text{CoSim}(z_i, z_j)/\tau}}{\sum_{k=1}^N e^{\text{CoSim}(z_i, z_k)/\tau}}\right)$$

where $(i,j) \in \mathcal{P}$ are positive pairs, $\text{CoSim}(z_i, z_j)$ is the cosine similarity between representations $z_i$ and $z_j$, $\tau$ is a temperature parameter controlling the concentration of the distribution, and $N$ is the batch size.

**What it computes:** for each positive pair $(i,j)$, the model computes the cosine similarity between their representations, exponentiates it (scaled by temperature), and divides by the sum of exponentiated similarities between $i$ and *all* items in the batch (including the positive $j$ and all negatives $k \neq j$). The negative log of this fraction is the loss. Minimizing this loss means maximizing the similarity of positive pairs relative to all negative pairs in the batch.

**Why this form:** three key design choices matter here. First, the softmax over the batch turns representation learning into a classification problem: given representation $z_i$, which of the $N$ items in the batch is its true partner? This forces the model to develop representations where positive pairs are more similar than any negative pair. Second, the temperature $\tau$ controls the sharpness of the softmax: low $\tau$ makes the distribution peakier (harder classification, more focus on the hardest negatives), while high $\tau$ makes it flatter (easier classification, less discriminating). Third, and critically, the denominator sums over the *current mini-batch*, making the loss batch-dependent — small batches provide fewer negatives, making the task easier and potentially leading to worse representations. This explains why CLIP needs "large mini-batches to make the contrastive training criterion between the positive and negative samples more effective" (Section 2.2).

For SSL methods like SimCLR, positive pairs are two augmented views of the same image. For CLIP (Section 2.2.1), positive pairs are (image, correct caption) pairs, while negatives are (image, all other captions in the batch) and (caption, all other images in the batch).

##### CLIP: Dual-Encoder Architecture with Shared Representation Space

CLIP trains two encoders from scratch: a vision encoder (typically a ViT or ResNet) and a text encoder (a transformer). Both produce fixed-dimensional embedding vectors. The training objective is a symmetric InfoNCE loss: given a batch of $N$ (image, caption) pairs, the model computes an $N \times N$ similarity matrix where entry $(i,j)$ is the cosine similarity between the $i$-th image embedding and the $j$-th text embedding. The diagonal entries $(i,i)$ are the positive pairs. The loss has two directions:

- **Image-to-text:** for each image $i$, classify which of the $N$ captions is correct (this is an $N$-way classification problem).
- **Text-to-image:** for each caption $j$, classify which of the $N$ images is correct.

The total loss is the average of these two directional losses.

The original CLIP was trained on 400 million image-text pairs collected from the web (Section 2.2.1). A ResNet-101 CLIP matched the performance of a supervised ResNet on ImageNet zero-shot classification (76.2% top-1), demonstrating that natural language supervision at scale can produce visual representations competitive with supervised training. Crucially, CLIP's zero-shot capability means it can classify images into categories it was never explicitly trained to recognize, simply by comparing the image embedding to text embeddings of class names like "a photo of a {class}."

##### SigLIP: A Binary Alternative to the Multi-Class Softmax

SigLIP [Zhai et al., 2023b] replaces CLIP's InfoNCE-based multi-class softmax with the original NCE binary cross-entropy formulation (Section 2.2.2). Instead of computing an $N \times N$ softmax over all pairs in the batch, SigLIP treats each (image, text) pair independently: the model predicts whether this specific pair is a match (positive) or not (negative).

The consequence is that SigLIP achieves "better 0-shot performances on smaller batch sizes than CLIP" (Section 2.2.2). This is because the binary formulation removes the batch dependency: each pair is evaluated in isolation rather than relative to all other pairs in the batch. For practitioners with limited GPU memory who cannot use large batch sizes, SigLIP is a preferable choice. The paper does not provide specific performance numbers comparing CLIP and SigLIP but establishes the conceptual distinction clearly.

##### Llip: Conditioning on Caption Diversity

Llip [Lavoie et al., 2024] addresses the problem that "an image can be captioned in several different ways" (Section 2.2.2). Standard CLIP treats a single caption as the unique correct description of an image, but in reality, the same image might be described as "a dog in a park," "a golden retriever playing fetch," or "a sunny afternoon with a pet." Each caption emphasizes different aspects of the image.

Llip proposes conditioning the image encoding on the target caption through a cross-attention module. This means the image representation is not fixed for a given image but varies depending on which caption it is being compared to. When compared with the caption "a golden retriever," the image encoder can learn to emphasize the dog and its breed; when compared with "a sunny afternoon," it can emphasize the lighting and weather conditions.

The paper reports that "accounting for the caption diversity increases the representation's expressivity and it generally improves the downstream zero-shot transfer classification and retrieval performance" (Section 2.2.2). This result underscores a fundamental limitation of the CLIP training objective: by treating each image as having one correct caption, CLIP implicitly assumes that all visual information relevant to any possible downstream task is captured in that single caption, which is rarely true for web-scraped data.

---

#### Masking Objectives and the Information-Theoretic View

The masking paradigm trains VLMs by randomly hiding portions of the input (image patches, text tokens, or both) and requiring the model to reconstruct the missing information. The paper presents two representative models — FLAVA and MaskVLM — before connecting masking to contrastive learning through a unified rate-distortion framework.

##### FLAVA: Multimodal Masking with Separate Encoders

FLAVA [Singh et al., 2022] employs a three-component architecture: an Image Encoder (ViT-based), a Text Encoder (transformer-based), and a Multimodal Encoder that fuses representations from both unimodal encoders via cross-attention (Section 2.3.1). The training regimen combines:

- **Masked Image Modeling (MIM):** random image patches are masked, and the Image Encoder must reconstruct them.
- **Masked Language Modeling (MLM):** random text tokens are masked, and the Text Encoder must reconstruct them, following the BERT paradigm.
- **Contrastive loss:** an InfoNCE-style objective between image and text representations, similar to CLIP.

The Multimodal Encoder receives hidden states from both unimodal encoders (after projection) and processes them jointly. It also has its own classification token `[CLSM]`. The model is pretrained on 70 million image-text pairs.

The key design insight is that FLAVA trains both unimodal and multimodal representations simultaneously. The unimodal objectives (MIM and MLM) ensure that each encoder learns strong representations of its own modality independently, while the contrastive and multimodal objectives enforce cross-modal alignment. The paper reports that FLAVA achieved "state-of-the-art performance across an array of 35 diverse tasks which span vision, language, and multimodal benchmarks" (Section 2.3.1).

##### MaskVLM: Masking Without Pretrained Tokenizers

FLAVA relies on a pretrained dVAE tokenizer [Zhang et al., 2019] for the vision side. MaskVLM [Kwon et al., 2023] removes this dependency by applying masking "directly in the pixel space and in the text token space" (Section 2.3.2).

The critical mechanism enabling this is **cross-modal information flow**: when reconstructing masked text tokens, the model has access to information from the image encoder; when reconstructing masked image patches, the model has access to information from the text encoder. This bidirectional flow ensures that each modality benefits from the other during reconstruction, unlike FLAVA where (in the unimodal masking objectives) each encoder works independently.

##### The Rate-Distortion Unification

The paper's information-theoretic framing (Section 2.3.3) is one of its most conceptually valuable contributions for newcomers, as it provides a mathematical language for understanding *why* masking and contrastive learning are not fundamentally different approaches but rather different implementations of the same underlying optimization.

Following Federici et al. [2020] and Dubois et al. [2021], the paper introduces the concept that any transformation $f(X)$ on data $X$ implicitly induces an equivalence relationship that partitions the representation space $f(X)$ into disjoint equivalence classes. The goal is to constrain conditional densities to be constant within each class: $f(x) \sim f(x') \implies p(z \mid f(x)) = p(z \mid f(x'))$, where $Z$ is the learned representation of $X$. This view unifies masking, data augmentation, and the choice function between two modalities — all can be represented as transformations of the data.

The rate-distortion problem is formulated as:

$$\arg \min_{p(z \mid x)} I(f(X); Z) + \beta \cdot H(X \mid Z)$$

where $I(f(X); Z)$ is the mutual information between the transformed input $f(X)$ and the representation $Z$ (the **rate** — how much information $Z$ retains about $f(X)$), $H(X \mid Z)$ is the conditional entropy of $X$ given $Z$ (the **distortion** — how much uncertainty remains about $X$ after observing $Z$), and $\beta > 0$ is a trade-off parameter.

**What it computes:** this objective seeks a representation $Z$ that minimizes the information retained about the specific transformation $f(X)$ (compressing away irrelevant details) while maximizing the information preserved about the original input $X$ (retaining what matters). The $\beta$ parameter controls the tension: large $\beta$ prioritizes reconstruction fidelity; small $\beta$ prioritizes compression.

**Why this form:** the rate-distortion framework formalizes the intuitive idea that good representations should discard irrelevant variation (the specific instance of a transformation, the particular caption wording) while preserving essential content (what the image actually depicts). This is exactly the tradeoff that both masking and contrastive learning navigate, but through different mechanisms.

The paper shows how to bound Equation 3 to recover practical losses:

$$\mathcal{L} = -\sum_{x \in \mathcal{D}} \mathbb{E}_{p(f) p(Z \mid f(x))} \left[\log q(z) + \beta \cdot \log q(x \mid z)\right]$$

where $\log q(z)$ is an entropy bottleneck bounding the rate $I(f(X); Z)$ by removing superfluous information, and $\log q(x \mid z)$ bounds the distortion $H(X \mid Z)$ and ensures information preservation.

**What it computes:** an upper bound on the rate-distortion objective that can be optimized in practice. The first term penalizes $Z$ for containing too much information (encouraging compression). The second term penalizes $Z$ for failing to reconstruct $X$ (encouraging preservation).

**Why this form for masking VLMs:** in masking, the entropy bottleneck is bounded by a constant that depends on the amount of information removed by masking — you cannot retain more information in $Z$ than what remains in the unmasked portion of the input. The distortion term is realized by auto-encoding: the model must reconstruct the original input (pixels or tokens) from the compressed representation.

**Why this form for contrastive VLMs:** contrastive losses are "compression without data reconstruction" (Section 2.3.3). The InfoNCE loss scores the equivalence of two representations — it retains necessary information by classifying which $Z$ is associated with an equivalent example $X$. The distortion is implemented through the classification objective rather than through reconstruction.

This unification gives newcomers a framework for reasoning about model design: if you need explicit reconstruction capability (e.g., for image generation), you need a form of distortion that involves decoding back to input space (masking or auto-encoding). If you only need a discriminative embedding (e.g., for retrieval or zero-shot classification), a contrastive distortion may suffice and is typically more computationally efficient since it avoids training a decoder.

---

#### Generative VLMs: From Captioning to Full Multi-Modal Generation

The generative paradigm differs fundamentally from contrastive and masking approaches: instead of learning representations and then comparing or reconstructing them, generative models learn to directly produce text, images, or both from multimodal inputs. The paper covers three subtypes: text generation from images (CoCa), full multimodal generation (Chameleon, CM3leon), and the counter-intuitive use of text-to-image models as zero-shot classifiers.

##### CoCa: Contrastive Captioner with a Generative Decoder

CoCa [Yu et al., 2022b] extends the CLIP architecture by adding a **multimodal text decoder** on top of the image encoder and the unimodal text decoder (Section 2.4.1). The model has two training objectives:

- **Contrastive loss:** same as CLIP, between image encoder outputs and unimodal text decoder outputs.
- **Generative captioning loss:** the multimodal text decoder receives (1) image encoder outputs and (2) representations from the unimodal text decoder, and generates captions autoregressively — token by token, each token conditioned on the image and all previous caption tokens.

The key architectural detail is that the unimodal text decoder processes text independently (enabling the contrastive objective), while the multimodal text decoder conditions on both modalities (enabling caption generation). By sharing the text decoder's lower layers, CoCa learns representations useful for both matching and generation.

CoCa is pretrained on ALIGN (~1.8B images with alt-text) and JFT-3B (an internal dataset with >29.5k classes, treating labels as alt-text). The paper notes that "pretraining relies on two datasets" and that the new generative loss "allows the ability to perform new multimodal understanding tasks (e.g., VQA) without the need for further adaptation using multimodal fusion modules" (Section 2.4.1). This is significant because it means CoCa can answer visual questions — a task requiring understanding of both image and question — without any architecture modifications beyond what was used for captioning.

##### Chameleon and CM3leon: Token-Based Unified Multimodal Models

CM3leon [Yu et al., 2023] (Section 2.4.2) represents a more radical unification: instead of having separate encoders for vision and text, it tokenizes everything into a shared vocabulary and processes interleaved image-text sequences with a single decoder-only transformer.

The image tokenizer (borrowed from Gafni et al., 2022) encodes a $256 \times 256$ image into 1024 tokens from a vocabulary of 8192. The text tokenizer (borrowed from Zhang et al., 2022) has a vocabulary of 56,320. A special `<break>` token marks transitions between modalities.

Training proceeds in two stages:

1.  **Retrieval-augmented pretraining:** a CLIP-based encoder acts as a dense retriever to fetch relevant multimodal documents, which are prepended to the input sequence. The model is trained with standard autoregressive next-token prediction on the augmented sequence. The paper notes that this "effectively increases the tokens available during pretraining thereby increasing data-efficiency" — by showing the model relevant external context, it learns faster from the same number of training examples.

2.  **Supervised fine-tuning (SFT):** the model undergoes multi-task instruction tuning, processing and generating content across different modalities. This stage "significantly improves its performance on a variety of tasks including text-to-image generation and language-guided image editing" (Section 2.4.2).

Chameleon [Team, 2024] extends this approach by being "uniquely designed to be mixed-modal from the beginning, utilizing a uniform architecture trained from scratch in an end-to-end manner on a blend of all modalities—images, text, and code" (Section 2.4.2). The critical technical challenges are **optimization stability and scaling** in a mixed-modal environment. The solutions include "novel modifications to the transformer architecture such as query-key normalization and revised layer norm placements, which are crucial for stable training."

The paper does not provide specific quantitative results for Chameleon or CM3leon, treating them as existence proofs of the unified token-based approach rather than benchmarks to compare against.

##### Image Tokenization: The VQ-VAE Framework

Since autoregressive models act on discrete tokens, continuous images must be discretized. The paper provides a clear explanation of the VQ-VAE [Van Den Oord et al., 2017] framework that underlies most modern image tokenizers:

The architecture is "a Convolutional Neural Network (CNN) encoder, followed by a Vector Quantization layer, followed by a CNN decoder" (Section 2.4.3). The Vector Quantization layer maps each encoder output vector to the closest embedding in a learned codebook (an embedding table updated during training). The loss combines:

- **Reconstruction loss:** L2 distance between input and reconstructed pixels.
- **Codebook commitment losses:** penalties that encourage encoder outputs and codebook embeddings to be close to each other, preventing the codebook from drifting away from the encoder's output distribution.

VQ-GAN [Esser et al., 2021] improves on this by adding **perceptual losses** (comparing feature maps from a pretrained network, not just pixels) and **adversarial losses** (a discriminator network distinguishing real from reconstructed images). These additions capture more fine-grained details that pixel-level L2 loss would miss.

VIT-VQGAN [Yu et al., 2022a] replaces the CNN encoder/decoder with a Vision Transformer, demonstrating that the VQ framework is orthogonal to the specific architecture choice.

##### Generative Models as Zero-Shot Classifiers

Section 2.4.3 presents one of the paper's most counter-intuitive ideas: text-to-image generative models, trained only to produce images from text descriptions, can be used directly for discriminative tasks like classification without any retraining.

Given a generative model trained to estimate $p_\theta(x \mid c)$ — the conditional likelihood of image $x$ given text prompt $c$ — classification follows from Bayes' theorem:

$$p_\theta(c_i \mid x) = \frac{p(c_i) p_\theta(x \mid c_i)}{\sum_j p(c_j) p_\theta(x \mid c_j)}$$

where $p(c_i)$ is the prior probability of class $c_i$ (usually assumed uniform), $p_\theta(x \mid c_i)$ is the model's estimated likelihood of the image under class description $c_i$, and the denominator normalizes across all $n$ classes.

**What it computes:** given an image $x$ and $n$ possible class descriptions $\{c_i\}_{i=1}^n$, this computes the posterior probability of each class by evaluating how likely the image is under each class's description and normalizing. The model's prediction is the class with highest posterior.

**Why this form:** this is literally Bayes' theorem — no approximation. It converts a generative model $p(x \mid c)$ into a discriminative model $p(c \mid x)$ by applying the prior and normalizing. The quality of the classifier depends entirely on the quality of the generative model's likelihood estimates.

For autoregressive models (like Parti), likelihood estimation is straightforward after tokenization. The image is mapped to a sequence of discrete tokens $(t_1, \ldots, t_K)$, and:

$$\log p_\theta(x \mid c_i) = \sum_{j=1}^K \log p_\theta(t_j \mid t_{<j}, c_i)$$

where $p_\theta(t_j \mid t_{<j}, c_i)$ is the model's predicted probability of token $t_j$ given all previous tokens and the class description $c_i$.

**What it computes:** the log-likelihood of the image under class $c_i$ is the sum of log-probabilities of each token given all previous tokens and the class description. This is just the autoregressive factorization of the joint probability — the standard way to compute likelihoods from autoregressive models.

For diffusion models (like Imagen, Stable Diffusion), likelihood estimation is more complex because diffusion models do not output $p_\theta(x \mid c)$ directly. Instead, they estimate the noise $\epsilon$ added to a noisy image $x_t$. The paper notes that diffusion-based classification techniques [Li et al., 2023a, Clark and Jaini, 2023] estimate a variational lower bound:

$$\log p_\theta(x \mid c_i) \propto -\mathbb{E}_{t, \epsilon} \left[ \|\epsilon - \epsilon_\theta(x_t, c_i)\|^2 \right]$$

where $t$ is a randomly sampled timestep, $\epsilon$ is the actual noise added at that timestep, $x_t$ is the noised image, and $\epsilon_\theta(x_t, c_i)$ is the model's noise prediction.

**What it computes:** the conditional log-likelihood is proportional to the negative expected squared error between the true noise and the model's prediction. Lower prediction error means the model better understands how the image was constructed from noise, which (under the diffusion formulation) means higher likelihood.

**Why this form:** this is the evidence lower bound (ELBO) for diffusion models. It avoids the intractable integration over all possible diffusion paths by sampling timesteps and noise. The proportional sign ($\propto$) indicates that there may be weighting terms depending on the specific bound used.

The paper reports that while diffusion-based zero-shot classification "performance ... is quite good," it is "still computationally expensive, scaling with the number of classes and requiring hundreds or thousands of network evaluations per test image" (Section 2.4.3). Li et al. [2023a] and Clark and Jaini [2023] develop techniques to reduce sample count (dynamically allocating samples to likely classes, matching noise across classes), but inference remains impractical without further optimization.

Despite computational cost, generative classifiers offer advantages over discriminative ones like CLIP:

- **Better out-of-distribution performance** ("more 'effective robustness'" — Section 2.4.3, citing Li et al., 2023a).
- **Stronger compositional reasoning** — on Winoground, "generative classifiers far outperform discriminative methods like CLIP" (Section 2.4.3).
- **Better alignment with human judgment** — "more shape bias" (Jaini et al., 2024), meaning they rely more on object shapes (as humans do) rather than textures (as CNNs often do).
- **Test-time adaptation** — generative classifiers can be jointly adapted with discriminative models using only unlabeled test examples (Prabhudesai et al., 2023).

---

#### VLMs from Pretrained Backbones: The Resource-Efficient Path

The pretrained-backbone paradigm represents the most accessible entry point to VLM research. Instead of training vision and text encoders from scratch (requiring hundreds of GPUs and hundreds of millions of image-text pairs), these methods leverage existing pretrained large language models (LLMs) and/or visual encoders, training only a relatively small mapping between them.

##### Frozen: The First to Connect Frozen LLMs to Vision

Frozen [Tsimpoukelli et al., 2021] (Section 2.5.1) makes a simple but crucial design choice: the language model (a 7-billion-parameter transformer trained on C4) is **kept frozen** — "this is crucial to maintain the features that the pre-trained model had already learned." Only the vision encoder (an NF-ResNet-50) and a lightweight linear mapping network are trained from scratch.

The mapping network projects visual features into the token embedding space of the frozen LLM, producing vectors that the LLM can interpret as if they were text tokens. The model is supervised with a text generation objective on Conceptual Captions. At inference time, the LLM can be conditioned on interleaved text and image embeddings — the image embeddings are inserted into the token sequence wherever an image appears.

The paper characterizes Frozen as "an important first step toward the current Multimodal LLMs capable of open-ended multimodal zero/few-shot learning" (Section 2.5.1), acknowledging that its performance was "only modest" but its architectural insight was foundational.

##### MiniGPT: The Minimalist Recipe

MiniGPT-4 [Zhu et al., 2023a] (Section 2.5.2) crystallizes the pretrained-backbone approach into its simplest form: a pretrained vision encoder (the same one used in BLIP-2, based on Q-Former and a ViT backbone), a pretrained LLM (Vicuna), and a single linear projection layer to map between them.

Training proceeds in two rounds:

1.  **Initial alignment:** 20k training steps with batch size 256 on approximately 5M image-text pairs from Conceptual Caption, SBU, and LAION. Only the linear projection layer is trained. The paper notes this required "only four A100 GPUs for around ten hours" — a dramatic contrast with CLIP's "256 and 600 GPUs across multiple days or weeks" (Section 3.2.2).

2.  **Instruction tuning:** 400 training steps with batch size 12 on highly-curated instruction-tuning data. This second round teaches the model to follow specific instruction formats.

MiniGPT-5 extends the output capability to include images interleaved with text. It introduces "generative tokens" — special visual tokens that can be mapped through transformer layers to feature vectors, which are then fed into a frozen Stable Diffusion 2.1 model to generate images. The training uses supervised data on multimodal dialogue generation and story generation tasks.

MiniGPT-v2 unifies multiple vision-language tasks (image captioning, VQA, object grounding) through a single interface by introducing "unique identifiers for different tasks when training, enabling the model to distinguish each task instruction effortlessly" (Section 2.5.2).

##### BLIP-2: The Q-Former Architecture

BLIP-2 [Li et al., 2023e] (Section 2.5.3) introduces the Q-Former, a trainable component (approximately 100-200M parameters) that sits between the frozen vision encoder and frozen LLM. The Q-Former is a transformer that takes in a fixed number of randomly-initialized "query" vectors. In the forward pass:

1.  The queries interact with image embeddings via cross-attention in the Q-Former.
2.  A linear layer projects the query outputs to the LLM's input space.

The Q-Former architecture is more expressive than a simple linear projection (as in MiniGPT-4) because the cross-attention mechanism allows the queries to selectively attend to different parts of the image, effectively learning which visual information is most relevant for the language model. The paper does not provide specific performance numbers for BLIP-2 but establishes it as the architectural foundation for MiniGPT-4's vision encoder.

##### Qwen: Cross-Attention Compression

Qwen-VL [Bai et al., 2023b] (Section 2.5.3) initializes its LLM from Qwen-7B and its vision encoder from ViT-bigG. Instead of a linear projection or Q-Former, it uses a one-layer cross-attention module to compress the visual representation to a fixed-length sequence of 256 tokens. This compressed sequence is then fed into the LLM.

The fixed-length compression addresses a practical challenge: ViT encoders produce a variable number of tokens depending on image resolution (one token per patch), but LLMs work best with consistent input lengths. By always producing 256 visual tokens regardless of the number of input patches, Qwen-VL creates a consistent interface between vision and language.

---

#### The Training Pipeline: Data, Software, and Practical Decisions

Section 3 of the paper shifts from "what VLMs exist" to "how to train them effectively." This section is the paper's most practically valuable contribution, as it surfaces operational knowledge rarely found in research papers.

##### Training Data: The Foundation of VLM Performance

The paper's overarching thesis about data is that "data pruning is a crucial step in training highly efficient and performant VLMs" (Section 3.1). It categorizes data-pruning methods into three families:

**Heuristic filters** can be unimodal or multimodal:

- **Unimodal text heuristics:** remove captions with low text complexity (measured by number of objects, attributes, and actions — Radenovic et al., 2023a), eliminate non-English alt-text using fastText, remove images based on resolution and aspect ratio.
- **Multimodal heuristics:** use image classifiers to filter out pairs where none of the detected objects map to any text token (Sharma et al., 2018a), or remove pairs with high text-in-image overlap using off-the-shelf text spotters (Kuang et al., 2021) — the latter prevents the model from learning to read text in images rather than understanding visual semantics, "thereby preventing low performance on object-centric and scene-centric downstream zero-shot tasks" (Section 3.1, Radenovic et al., 2023a).

**Ranking based on pretrained VLMs** uses existing models to score alignment:

- **CLIPScore** [Hessel et al., 2021, Schuhmann et al., 2021]: compute cosine similarity between image and text embeddings from a pretrained CLIP model, then rank and filter out low-scoring pairs.
- **LAION filtering** [Schuhmann et al., 2021]: uses an OpenAI CLIP model pretrained on 400M pairs to evaluate image-text alignment at web scale.
- **T-MARS** [Maini et al., 2023]: detects and masks text regions in images before computing CLIPScore, preventing in-image text from artificially inflating alignment scores.
- **Sieve** [Mahmoud et al., 2024]: uses generative image captioning models pretrained on small but curated datasets to produce more accurate alignment scores, minimizing false positives and negatives from CLIPScore ranking.

**Diversity and balancing** methods ensure broad concept coverage:

- **DataComp** [Gadre et al., 2023]: samples image-text pairs semantically similar to curated datasets like ImageNet, using either text-based sampling (retain captions overlapping with ImageNet classes) or image-based sampling (cluster web-scale image embeddings with FAISS, then select clusters closest to ImageNet training samples). The paper warns this "could bias the CLIP model, potentially limiting its generalization to new downstream tasks."
- **MetaCLIP** [Xu et al., 2024]: uses 500,000 queries from Wikipedia/WordNet as metadata to create a balanced pretraining distribution, sampling up to 20,000 examples per query to balance well-represented and under-represented concepts.
- The paper acknowledges an inherent limitation: "collecting a perfectly balanced dataset is impractical due to the natural long-tailed distribution of web data. Consequently, all these CLIP variants still exhibit imbalanced performances across downstream visual concepts" (Section 3.1, Parashar et al., 2024).

**The critical finding about zero-shot capability:** Udandarao et al. [2024] demonstrate that "the zero-shot performances of VLMs depend mostly on how much those zero-shot downstream concepts are present in the training data" (Section 3.1). This challenges the common narrative of "zero-shot" as a magical emergent capability — it is better understood as "few-shot with massive implicit exposure." If a concept never appeared in training, the VLM cannot recognize it; if it appeared frequently, the VLM appears to "zero-shot" recognize it.

##### Synthetic Data Generation: Improving Caption Quality

The paper describes two directions for synthetic data:

**Improving captions** (Section 3.1.1):

- BLIP [Li et al., 2022b] performs bootstrapping: generate synthetic captions (positive examples) and filter out noisy captions (negative examples) simultaneously.
- Santurkar et al. [2022] use BLIP to approximate caption descriptiveness and show that "models trained on consistent and complete synthetic captions generated by BLIP outperform a model trained on human-written captions" — a surprising result suggesting that human-written web captions (often terse, incomplete, or only loosely related to image content) may be worse training targets than model-generated captions.
- Nguyen et al. [2023] use BLIP2 to replace poorly aligned alt-text with descriptive synthetic captions, demonstrating that a mixture of real and synthetic captions is effective. However, they identify a scaling limitation: "at scale, the improvement provided by synthetic captions is capped by the limited diversity of generated captions compared to the high diversity of noisy text labels." Synthetic captions, being model-generated, tend toward a narrower distribution than the wild diversity of human-written web text.
- Chen et al. [2024] use LLaVA as a captioning model to train text-to-image generative models efficiently.

**Generating synthetic images** (Section 3.1.1):

- Tian et al. [2023b] use multiple synthetic images from the same text prompt as multiple positive pairs in contrastive learning, demonstrating improved performance over CLIP and SimCLR using only synthetic data.
- SynCLR [Tian et al., 2023a] and SynthCLIP [Hammoud et al., 2024] train VLMs entirely on synthetic data: an LLM generates captions, and a text-to-image model generates images from those captions. The paper presents this as a proof of concept that real data is not strictly necessary, though the approach is not yet competitive with models trained on web-scale real data.

##### Data Augmentation: SLIP and CLIP-rocket

Can self-supervised augmentation techniques (like those in SimCLR) benefit vision-language pretraining? SLIP [Mu et al., 2022] addresses this by "introducing an auxiliary self-supervised loss term on the vision encoder" (Section 3.1.2). The input image is augmented twice to create a positive pair, contrasted against all other images in the batch — identical to SimCLR's objective, but running alongside the standard CLIP contrastive loss.

However, SLIP only applies the SSL loss to the visual encoder, "which does not fully exploit the important signal coming from text" (Section 3.1.2). CLIP-rocket [Fini et al., 2023] addresses this by making the augmentations cross-modal. The key mechanism:

1.  The input image-text pair is augmented in an asymmetrical way: one weak set of augmentations, one strong set.
2.  Both augmented pairs are embedded with the standard CLIP encoder.
3.  Each is projected through a different projector: the weak augmentation uses a linear layer (standard CLIP), the strong augmentation uses a 2-layer MLP ("to cope with the noisier embeddings").
4.  At inference time, weak and strong representations are interpolated to get a single vector.

The paper notes a crucial design choice (building on Bordes et al., 2022): "it is crucial to separate the two projectors as the strong one is learning more invariant, too invariant, representations for downstream tasks." If both augmentations shared a projector, the strong augmentations' invariance would interfere with the weak augmentations' discriminative power.

##### Interleaved Data: Natural vs. Synthetic Construction

For autoregressive VLMs (like Flamingo, MM1), interleaved text-image data during training improves few-shot performance (Section 3.1.3). The paper describes two construction strategies:

**Natural interleaved data (OBELICS)** [Laurençon et al., 2023]: preserves the intrinsic structure of web documents where text and images co-occur. The curation pipeline: collect English data from Common Crawl → deduplicate → pre-process HTML to identify useful DOM nodes → filter images (remove logos) → apply document-level text filtering (remove incoherent or poorly-formed text). This preserves the authentic context in which images and text appear together on the web.

**Synthetic interleaved data (MMC4)** [Zhu et al., 2023b]: retrofits text-only corpora with images. For each text passage, images are collected from the internet based on CLIP similarity between the text and candidate images. The paper notes this "may lack the contextual nuance of naturally interleaved datasets" but "allows for the scalable creation of multimodal data from well-established text-only resources."

##### Data Quality Assessment

The paper identifies quantifying multimodal data quality as "a very active area for research" (Section 3.1.4). Existing work addresses quality along separate dimensions:

- **Text quality:** QuRating [Wettig et al., 2024], Data efficient LMs [Sachdeva et al., 2024], text-quality-based pruning [Sharma et al., 2024].
- **Image quality:** VILA [Ke et al., 2023], LAION-aesthetics [Schuhmann, 2023].
- **Alignment quality:** CLIP-based approaches [Radford et al., 2021, Xu et al., 2024, Gao et al., 2024].

The paper's key observation is that "we lack a holistic way of evaluating the quality of multimodal and interleaved data, which remains an active area of research" (Section 3.1.4). Current methods treat text quality, image quality, and alignment quality as independent properties, but a truly interleaved dataset requires co-assessment: a high-quality image paired with irrelevant text is poor data, regardless of the image's aesthetic score.

##### Human Annotation: The Power and Limitation

The paper acknowledges the value of human-annotated data (Section 3.1.5): DCI [Urbanek et al., 2023] provides fine-grained annotations for images from SA-1B [Kirillov et al., 2023], and datasets like OKVQA, A-OKVQA, Image Paragraph Captioning, VisDial, Visual Spatial Reasoning, and MagicBrush all rely on detailed human annotations. However, the limitation is stark: "human-annotated data ... is often costly to get, especially when requesting fine-grained annotations. In consequence, the number of images with highly detailed annotations is often low which makes often those datasets more suited for evaluation or fine-tuning than for large-scale pre-training."

##### Software Infrastructure: Tools and Practical Scaling Advice

Section 3.2 provides concrete operational guidance:

**Existing repositories** (Section 3.2.1): OpenCLIP and Hugging Face transformers implement most VLMs and are "extremely useful when making benchmarks or comparing different models."

**GPU requirements** (Section 3.2.2): CLIP and OpenCLIP "leveraged more than 500 GPUs to train their models" with public cloud costs "equivalent to hundreds of thousands of dollars." But the paper offers a more accessible path: "when using the right ingredients such as having a high-quality dataset and leveraging masking strategies when using bigger models, training a contrastive model like CLIP on hundreds of millions of images from scratch should not require more than 64 GPUs (which should be equivalent to spending around 10K USD in compute)." For pretrained-backbone approaches, costs are much lower — MiniGPT-4 required "only four A100 GPUs for around ten hours."

**Training speedups** (Section 3.2.3):

- `torch.compile` (PyTorch team) and `xformers` (Lefaudeux et al., 2022) provide software-level acceleration through more efficient attention mechanisms.
- **Data loading is a bottleneck:** "data loading often becomes a bottleneck that significantly slows down training" because "large-scale datasets are often saved in chunks of tar files that have to be uncompressed on the fly."
- **Recommendation:** "store as many uncompressed files as possible to speed up training."
- **FFCV** [Leclerc et al., 2023] creates data files "much faster to load" and "can significantly speed up VLM training." The tradeoff is "storage might be more costly than storing compressed files," but "since the training speed will be much faster, the additional storage cost should be compensated quickly by the lower amount of compute needed."
- **Masking for efficiency:** Li et al. [2023f] show that "by randomly masking image tokens one can significantly speed up training time while improving model performances" (Section 3.2.3). This is because masked tokens don't require forward/backward computation for the masked positions.

**Most impactful hyperparameters** (Section 3.2.4): McKinzie et al. [2024] find that "image resolution, visual encoder capacity, and visual pretraining data are the choices that most impact model performance," while "while there are many ways to connect modalities, this choice is much less important." The paper also notes the importance of the training data mixture: "the right mix achieves the best performance across both zero-shot classification and visual-question answering tasks."

##### Model Selection: A Decision Framework

Section 3.3 provides explicit criteria for choosing a VLM approach:

**Use contrastive models like CLIP when** (Section 3.3.1):
- You need representations with meaning in both image and text space (enabling text-to-image retrieval).
- You want a simple, well-understood training paradigm.
- You want a good base for building more complex models (especially for improving grounding).
- You need a model for data curation pipelines (CLIP embeddings enable semantic search).
- **Caveat:** CLIP is not generative — you cannot generate captions, only retrieve the best from a list.
- **Caveat:** CLIP needs "very large dataset as well as large batch sizes to offer decent performances, which implies that CLIP usually needs significant resources to be trained from scratch."

**Use masking when** (Section 3.3.2):
- You want to jointly model text and image distributions with reconstruction capability.
- You want to avoid batch dependency (no need for negative examples, so smaller mini-batches work without temperature tuning).
- **Caveat:** masking methods "might need to leverage a decoder to map back the representation to the input space" which "might add an additional bottleneck which might make these methods less efficient than a purely contrastive one."

**Use generative models when** (Section 3.3.3):
- You need the ability to generate photorealistic images from text (diffusion/autoregressive models).
- You want to directly visualize what the model has learned (rather than doing expensive k-NN searches).
- You believe that "having the ability to generate images given words is an important step towards creating a good world model" (the paper notes this is debated: "other researchers argue that such a reconstruction step is not needed").
- **Caveat:** generative models "are more computationally expensive to train than their contrastive learning counterpart."

**Use pretrained backbones when** (Section 3.3.4):
- You have limited compute resources (only the mapping between representations needs to be learned).
- You want rapid prototyping and experimentation.
- **Caveat:** "the VLM will be impacted by the potential hallucination of the LLM" and "any bias coming from the pretrained models."
- **Caveat:** there may be "additional overhead in trying to correct the defect of the vision model or of the LLM."

##### Improving Grounding: Bounding Boxes and Negative Captioning

The paper defines grounding as addressing the problem of "models not understanding well the text prompt which could either lead to ignoring some part of the prompt or to hallucinating something that is not even part of the prompt" (Section 3.4). Specific challenges include "understanding relations such as an object being on the left or right, negations, counting, or understanding attributes (such as colors or textures)."

**Bounding box annotations** (Section 3.4.1):

- **X-VLM** [Zeng et al., 2022]: incorporates box regression and IoU loss to locate and align visual concepts with textual descriptions. Trained on up to 16M images from COCO, Visual Genome, SBU, and Conceptual Captions with bounding box annotations. The paper reports it "outperforms existing methods across a variety of vision-language tasks."
- **Kosmos-2** [Peng et al., 2024]: avoids manual annotation by bootstrapping bounding boxes from web-crawl data. The pipeline: extract nouns from captions using spaCy → use GLIP [Li et al., 2022c] to predict bounding boxes for extracted nouns → use spaCy to extract expression spans associated with each noun → produce captions associated with detected bounding boxes. The paper notes the limitation: "such an approach is limited by how strong the grounding model for bounding box detection is. It is likely that if this base model fails on some rare nouns or instances, the downstream model would make similar mistakes."

**Negative captioning** (Section 3.4.2): the ARO benchmark [Yuksekgonul et al., 2023] evaluates VLMs by presenting negative samples — captions that are incorrect or nonsensical pairings with the image — and testing whether the model correctly identifies them as such. The paper frames this as an extension of contrastive learning principles: "by contrasting positive pairs ... with negative pairs ..., models are forced to develop a nuanced understanding of the data, going beyond mere superficial features to grasp the underlying patterns that distinguish different classes or categories."

##### Improving Alignment: Instruction Tuning and RLHF for VLMs

Section 3.5 describes techniques adapted from the language domain to align VLM outputs with human expectations.

**Instruction tuning** involves fine-tuning on supervised data containing "instructions, inputs, and the desired response" (Section 3.5). Datasets are "much smaller compared to pretraining data—with instruction tuning data sizes ranging from a few to one hundred thousand samples."

**RLHF** trains a reward model to match human preferences, then fine-tunes the primary model with the reward model. The paper notes that "while instruction tuning requires supervised training samples, which can be costly to gather, RLHF takes advantage of an auxiliary reward model to mimic human preferences."

**The LLaVA family** (Section 3.5.1) illustrates the progression:

- **LLaVA** [Liu et al., 2023d]: among the first to incorporate instruction fine-tuning in VLMs. Uses 150k synthetically generated visual instruction samples, a pretrained Vicuna LLM, a pretrained CLIP ViT-L/14 vision encoder, and a linear projector to fuse embeddings. Shows improvements on synthetic instruction following and Science QA.

- **LLaVA 1.5** [Liu et al., 2023c]: improves instruction fine-tuning with a cross-modal MLP (replacing the linear projector) and academic VQA instruction data. Trained on 600k image-text pairs, "making it much more efficient to train compared to other instruction-tuned models." Training takes "approximately one day on 8-A100 GPUs."

- **LLaVA-RLHF** [Sun et al., 2023]: addresses the scarcity of high-quality visual instruction tuning data through Factually Augmented RLHF. The reward model is "augmented with extra factual information of image captions and ground-truth multi-choice to reduce reward hacking." Uses GPT-4-generated training data and human-written image-text pairs. On LLaVA-Bench, achieves "94% performance level of GPT-4." On MMHAL-BENCH (focused on penalizing hallucinations), "outperforms baselines by 60%."

- **LLaVA-NeXT (v1.6)** [Liu et al., 2024a]: increases image resolution by "concatenating visual features from the full image and smaller image patches, which are separately fed through the vision encoder." Improves the visual instruction tuning data mixture with "better visual reasoning, OCR, world knowledge, and logical reasoning examples." The largest variant uses a 34B-parameter LLM backbone. Achieves state-of-the-art among open-source multimodal LLMs.

**Multimodal in-context learning** (Section 3.5.2): Otter [Li et al., 2023c] demonstrates that VLMs can learn from a few in-context examples without extra fine-tuning, analogous to text-only LLM in-context learning. The training dataset, MIMIC-IT [Li et al., 2023b], contains ~2.8M multimodal instruction-response pairs, each including in-context instruction-image-answer tuples plus a test example. The in-context tuples relate to the test example in three ways: (1) similar instructions but different images, (2) same images but different instructions, (3) sequentially related images from video repositories. Fine-tuning OpenFlamingo on MIMIC-IT produces Otter, which "exhibits stronger instruction following ability as well as multimodal in-context learning ability."

##### Improving Text-Rich Image Understanding

Section 3.6 addresses VLMs' struggles with text in images, which the paper attributes to "the prevalence of natural images in their training data (for instance, Conceptual Captions and COCO)" — these datasets prioritize photographic scenes where text rarely appears.

**LLaVAR** [Zhang et al., 2023c]: enhances visual instruction tuning with text-rich images (movie posters, book covers). The pipeline: collect 422K text-rich images from LAION → use OCR tools to extract text → prompt text-only GPT-4 with recognized text and image captions to generate 16K conversations with question-answer pairs → combine with existing instruction-following data. Results: "up to a 20% accuracy improvement on text-based VQA datasets and a slight improvement on natural images."

**Monkey** [Li et al., 2023h]: addresses the resolution limitation (most MM-LLMs cap input images at 224×224). Uses a sliding window approach to process images in uniform patches, each matching the original training resolution of the well-trained vision encoder. Each patch is processed independently by the vision encoder (enhanced with LoRA adjustments and a trainable visual resampler). This enables handling resolutions up to 1344×896. The paper notes it "also employs a multi-level description generation method, enriching the context for scene-object associations."

**Lumos** [Shenoy et al., 2024]: takes a different approach — instead of trying to get the VLM itself to read text, Lumos decouples scene text recognition into a separate module. The STR module contains four sub-components: ROI detection (identifies salient areas), Text detection (finds word bounding boxes), Text recognition (reads the words), and Reading-order reconstruction (organizes recognized words into paragraphs in reading order). The recognized text and coordinates are then fed to a cloud-hosted multimodal LLM. The key advantage: the STR module can process 3k×4k images, and running it on-device "reduces power and latency from transferring high-resolution images to the cloud."

##### Parameter-Efficient Fine-Tuning (PEFT)

Section 3.7 addresses the practical challenge that "as the size of pre-trained models continues to grow, fine-tuning the entire parameter set of these models becomes impractical due to computational constraints." The paper categorizes PEFT methods into four groups:

**LoRA-based** [Hu et al., 2022]: injects trainable low-rank matrices into frozen pretrained weights. Variants include QLoRA (integrates LoRA with 4-bit quantization — Dettmers et al., 2023), VeRA (uses a single shared pair of low-rank matrices across layers with learned scaling vectors — Kopiczko et al., 2024), and DoRA (decomposes pretrained weights into magnitude and direction components for fine-tuning — Liu et al., 2024b).

**Prompt-based:** CoOp [Zhou et al., 2022] optimizes the context words of prompts using learnable vectors, outperforming hand-crafted prompts and linear probe models in few-shot learning. VPT [Jia et al., 2022] introduces "less than 1% of model parameters" as trainable tokens in the input space while keeping the backbone frozen.

**Adapter-based:** CLIP-Adapter [Gao et al., 2024] adds bottleneck layers with residual connections to either the visual or language branch. VL-adapter [Sung et al., 2022] evaluates adapter-based methods across diverse benchmarks and finds that "the application of the weight-sharing technique in conjunction with adapters can effectively rival the performance of full fine-tuning, while necessitating updates to only ... 4.18% for image-text tasks and 3.39% for video-text tasks." LLaMA-Adapter V2 [Gao et al., 2023] unlocks more learnable parameters (norm, bias, scale) and uses early fusion to incorporate visual tokens into LLM layers, requiring "much fewer additional parameters" compared to full fine-tuning approaches.

**Mapping-based:** these methods avoid modifying the pretrained network's architecture entirely. LiMBeR [Merullo et al., 2022] uses a single linear layer to project visual features to the LLM's hidden state dimension, applied independently to each visual feature vector — the sequence length passed to the LLM equals the number of visual tokens. MAPL [Mañas et al., 2023] addresses the sequence length issue by aggregating visual features into a smaller set of query tokens: input features are projected and concatenated to learnable query tokens, and only the query token outputs are fed to the LLM. The paper notes this approach "requires fewer trainable parameters and leads to increased data-efficiency" (Section 3.7, Vallaeys et al., 2024).

---

This completes the deep dive into the paper's technical approach. The paper's contribution is not a single method but rather a structured journey through the VLM design space: from the mathematical foundations of contrastive learning (energy-based models → NCE → InfoNCE), through the major training paradigms (contrastive, masking, generative, pretrained-backbone), to the practical infrastructure of data curation, software tooling, model selection, grounding, alignment, and parameter-efficient fine-tuning. Throughout, the paper emphasizes *why* design choices matter — why SigLIP's binary loss enables smaller batch sizes, why VQ-VAE tokenization is necessary for autoregressive image modeling, why data loading bottlenecks are often the limiting factor in training speed — rather than merely cataloging what exists. This pedagogical depth, combined with the explicit surfacing of tacit practical knowledge (GPU requirements, software recommendations, dataset construction strategies), is what distinguishes this paper from a standard survey and makes it accessible to "students or researchers in other areas who want to enter the field."

## 4. Key Insights and Innovations

### Innovation 1: The VLM Design Space Can Be Organized Into Four Training Paradigms That Are Conceptually Unified by the Same Information-Theoretic Objective

The paper's most fundamental conceptual contribution is not the taxonomy itself—categorizing VLMs as contrastive, masking-based, generative, or built from pretrained backbones—but the demonstration that these four categories are not separate islands of research. Section 2.3.3 shows they are different implementations of the same rate-distortion tradeoff: all VLMs must decide how much information to compress away (rate) and how faithfully to preserve what remains (distortion). Contrastive learning implements the distortion term through a classification objective (which $Z$ is associated with an equivalent $X$?), while masking implements it through reconstruction (can we recover $X$ from $Z$?). The mathematical machinery differs, but the optimization problem is shared.

This reframing matters because prior to this paper, the VLM landscape appeared to newcomers as a pile of incompatible methods. A researcher reading a CLIP paper and a FLAVA paper would encounter entirely different loss functions with no obvious connection between them. The information-theoretic lens—building on Federici et al. (2020) and Dubois et al. (2021)—provides a conceptual Rosetta Stone. It answers the question "when should I use masking versus contrastive training?" not with a heuristic ("CLIP works better for retrieval") but with a principle: if you need explicit reconstruction capability, you need a decoder-based distortion; if you only need discriminative embeddings, contrastive distortion may suffice and is computationally cheaper. This is not an incremental finding but a **fundamental conceptual reframing** that changes how the field should think about model design.

The idea is intellectually distinctive because it operates at the level of *why* rather than *what*. Standard surveys describe what each model does; this paper explains why the models' objectives are mathematically coherent variants of the same underlying optimization. For a newcomer, this transforms VLM research from a memorization task (remembering dozens of model names and architectures) into a reasoning task (understanding the design axes along which any VLM must make tradeoffs).

---

### Innovation 2: "Zero-Shot" VLM Capabilities Are a Function of Training Data Coverage, Not Emergent Reasoning

The paper does not originate this finding—it credits Udandarao et al. (2024)—but it elevates it to a central diagnostic principle with broad implications for how the field interprets VLM evaluation. The key claim, stated in Section 3.1, is that "the zero-shot performances of VLMs depend mostly on how much those zero-shot downstream concepts are present in the training data." In other words, when a CLIP model achieves 76.2% zero-shot accuracy on ImageNet, it is not performing genuine zero-shot generalization from language descriptions to visual categories. It is performing retrieval from a training distribution that likely contained many images of those categories, even if not labeled with the exact class names.

This challenges a narrative that has been implicit in VLM research since Radford et al. (2021): that contrastive pretraining on image-text pairs produces representations capable of genuinely novel generalization. The paper's reframing is more sober: zero-shot capability is better understood as "few-shot with massive implicit exposure." If a concept (e.g., "okapi") never appeared in the 400M training pairs, the model cannot recognize it; if it appeared frequently, the model appears to "zero-shot" recognize it. The distinction between zero-shot and few-shot collapses when the training set is large enough that most evaluation concepts are implicitly present.

This insight is significant beyond raw performance because it changes how researchers should design and interpret evaluations. Section 4.2.4 describes the practical methodology: find the concepts that describe a downstream task (class names for classification), use recognition models like RAM to detect how much those concepts appear in the training data, and then predict whether the VLM can solve the task. If a model performs well on a "zero-shot" benchmark where the concepts were well-represented in training, that is not evidence of generality—it is evidence of coverage. This shifts evaluation from "does the model succeed on this task?" to "does the model succeed on this task for reasons beyond training data memorization?"—a fundamentally different and more rigorous standard.

The idea is a **diagnostic reframing**, not a technical advance. No new model or algorithm is proposed; instead, the paper changes what counts as a valid inference from evaluation results. This is precisely the kind of meta-knowledge that Section 1 identifies as missing from standard surveys: not what the benchmarks measure, but what they *actually* measure versus what researchers think they measure.

---

### Innovation 3: Standard VLM Benchmarks Contain Systematic Evaluation Pitfalls That Can Produce Catastrophically Misleading Results

The paper's Section 4.1.6 contains what may be its most practically actionable contribution: a warning that many binary-choice VLM benchmarks (including ARO, Winoground variants, and others) are structurally vulnerable to trivial models achieving perfect or near-perfect scores. The mechanism is deceptively simple: these benchmarks present a correct caption and a negative caption, and evaluate whether the model assigns higher probability to the correct one. If the model outputs equal probabilities for both captions (a representation collapse where both captions map to the same vector), PyTorch's `argmax` returns the first element. Since "many benchmarks put the correct caption as the first element," the paper notes, "a model whose parameters are all equal to zero could achieve 100% accuracy."

This is not a theoretical concern—it is a real failure mode that, if undetected, could lead a researcher to believe their model has achieved state-of-the-art compositional reasoning when it has in fact learned nothing. The paper's framing is significant because it identifies a **class of evaluation error**, not an isolated bug. The underlying problem is that argmax-based evaluation assumes a discriminative signal where none may exist. Any benchmark where (1) the task reduces to binary choice between correct and incorrect options, (2) the correct option is placed consistently first, and (3) the model can collapse to uniform outputs is vulnerable to this artifact.

The recommended mitigation—"adding a small epsilon random number or keeping track if the captions are assigned the same probabilities" (Section 4.1.6)—is simple but the insight is that **most researchers would not think to check for this**. The paper surfaces a failure mode that is almost certainly present in published results but rarely detected because the default behavior (correct placement first + argmax) produces results that look plausible. A model achieving 60% accuracy on ARO might seem believable; the researcher moves on without realizing that 50% of that accuracy came from the artifact, and the model's true discriminative ability is near chance.

This contributes to the field as a **methodological diagnostic**, analogous to how the machine learning community eventually recognized that test set reuse can inflate reported performance. It changes the standard of evidence: after this paper, any benchmark using binary choice with consistent correct-answer placement must explicitly demonstrate that the artifact is not driving results. The idea is intellectually distinctive because it requires thinking at the level of **evaluation protocol** rather than model architecture—a perspective that is underrepresented in the VLM literature, which tends to focus on new training objectives and architectures.

---

### Innovation 4: The Training–Inference Compute Tradeoff Has Structure: Pretrained Backbones Can Match $500\times$ More Expensive Training with $50\times$ Less Compute

The paper does not present this as a single headline result but rather as a **scaffolding insight** that emerges from the practical guidance in Sections 3.2.2 and 3.3.4. The raw numbers tell the story: training CLIP from scratch required "using between 256 and 600 GPUs across multiple days or weeks" (Section 3.2.2) on 400M image-text pairs, with cloud costs equivalent to "hundreds of thousands of dollars." Training MiniGPT-4 required "only four A100 GPUs for around ten hours" (Section 2.5.2) and "around 10K USD in compute" (Section 3.2.2) for a more modest setup. The pretrained-backbone approach is roughly one to two orders of magnitude cheaper in compute while achieving competitive performance on many benchmarks.

This is not just a cost comparison—it is a **structural finding about the modularity of VLM capabilities**. The fact that a frozen LLM (Vicuna) paired with a frozen vision encoder (BLIP-2's Q-Former + ViT) and a single trained linear projection layer can perform visual question answering, captioning, and multi-turn dialogue implies that **the latent spaces of independently trained vision and language models are already sufficiently aligned** that only a thin mapping layer is needed. The heavy lifting of visual understanding and language generation was done during unimodal pretraining; the VLM's job is merely to connect them.

The significance beyond raw performance is that this finding reshapes the economics of VLM research. If a graduate student with four A100s can build a competitive VLM in ten hours, the barrier to entry collapses. The paper's framing of this tradeoff (Section 3.3.4) is nuanced: pretrained backbones come with costs (inherited LLM hallucinations, biases from pretrained models, additional overhead in correcting defects), but for resource-constrained settings, the tradeoff is strongly favorable. This is not an incremental improvement—it is a **fundamental shift in accessibility** that democratizes VLM research.

The paper does not claim this as an original discovery (Frozen, BLIP-2, and MiniGPT all predate it), but the pedagogical contribution is to make the tradeoff **explicit and quantified**. Prior work presented individual pretrained-backbone models as architectural innovations; this paper presents the entire paradigm as a **practical decision framework** with clear resource implications. For a newcomer choosing between training from scratch and using pretrained components, the paper provides concrete numbers (GPU counts, training hours, cost estimates) rather than vague advice. This transforms the paper from a literature review into a **decision support tool**, which is a distinctive type of intellectual contribution rarely seen in academic surveys.

## 5. Experimental Analysis

### Evaluation Methodology

**Dataset.** The paper does not conduct new experiments. Rather, it synthesizes results from existing literature. When specific benchmarks are discussed for evaluation purposes (as opposed to training), the paper references standard VLM evaluation suites: ImageNet for zero-shot classification (Section 4.1.5), COCO captioning (Section 4.1.1), VQAv2 and GQA for visual question answering (Section 4.1.3), TextVQA for text-centric VQA (Section 4.1.4), Winoground and ARO for compositional reasoning (Section 4.1.6), DCI for dense captioning (Section 4.1.7), and PUG for synthetic spatial reasoning evaluation (Section 4.1.8). No original training data or evaluation splits are introduced.

**Base model(s).** The paper discusses a wide range of models without conducting experiments on them itself: CLIP variants (Radford et al., 2021, trained on 400M image-text pairs), SigLIP, Llip, FLAVA, MaskVLM, CoCa, Chameleon, CM3leon, Frozen, MiniGPT-4/5/v2, BLIP-2, Qwen-VL, LLaVA family (LLaVA, LLaVA 1.5, LLaVA-RLHF, LLaVA-NeXT), Otter, Video-LLaMA, VideoOFA, and MiniGPT4-Video. The paper's function is to describe these models and report their published results rather than to evaluate them under a common protocol. For cost comparisons, the paper quotes MiniGPT-4's setup (4 A100 GPUs, ~10 hours) against CLIP's (256-600 GPUs, multiple days/weeks) from their respective publications.

**Metrics.** The paper describes multiple metrics without computing them: zero-shot classification accuracy (ImageNet, CIFAR10/100, Caltech101, etc. — percentage of correct predictions using prompt-engineered class names), VQA accuracy (exact string match between generated and reference answers), CLIPScore for image-text alignment (cosine similarity between CLIP embeddings), BLEU and ROUGE for caption quality (noted to be insufficient and potentially misleading), binary-choice accuracy for compositional benchmarks (percentage of correctly ordered caption pairs identified, with warnings about argmax artifacts), and pass@1 for generative model evaluation.

**Baselines.** The paper does not run controlled experiments comparing models against baselines. Instead, it reports comparisons that exist in the original papers: CLIP vs. supervised ResNet on ImageNet (CLIP ResNet-101 matched 76.2% zero-shot classification), generative classifiers vs. discriminative CLIP on Winoground (generative "far outperform" — Section 2.4.3), LLaVA-RLHF vs. baselines on MMHAL-BENCH (LLaVA-RLHF "outperforms baselines by 60%" — Section 3.5.1), and MiniGPT4-Video vs. Video-LLaMA on video benchmarks (MiniGPT4-Video "consistently outperforms ... by a large margin" — Section 5.3). For data curation, the DataComp benchmark (Section 3.1) provides a standardized comparison framework where CLIP architecture and hyperparameters are fixed and only datasets vary.

**Generation budget / compute accounting.** For cost comparisons, the paper quotes two regimes: CLIP-scale training (256-600 GPUs, multiple days/weeks, "hundreds of thousands of dollars" in cloud costs — Section 3.2.2) and pretrained-backbone fine-tuning (64 GPUs, ~$10K USD for contrastive training from scratch with good data; 4 A100s for ~10 hours for MiniGPT-4's linear projection layer — Section 2.5.2). The paper does not define a unified compute metric (e.g., FLOPs or GPU-hours) across methods. For diffusion-based zero-shot classification, the paper notes computational cost "scaling with the number of classes and requiring hundreds or thousands of network evaluations per test image" (Section 2.4.3), making inference impractical without further optimization.

**Cross-validation / statistical protocol.** The paper does not report any cross-validation procedures, likely because it conducts no original experiments. It does, however, surface statistical vulnerabilities in benchmark design: Section 4.1.6 warns that binary-choice benchmarks with consistent first-position placement of correct answers can produce 100% accuracy from a zero-parameter model due to argmax behavior, and recommends adding small epsilon random numbers or tracking equal-probability cases. Section 4.2.3 documents the VQA benchmark's susceptibility to blind language priors (e.g., "Is there a clock" has answer "yes" 98% of the time — Goyal et al., 2017), meaning a model ignoring images entirely can achieve high accuracy.

---

### Main Quantitative Results

Since the paper does not conduct original experiments, this section catalogs the quantitative claims from prior work that the paper reports as evidence for its pedagogical framing. The organization follows the paper's conceptual structure: data curation effectiveness, training paradigm comparisons, and evaluation benchmark vulnerabilities.

#### Data Curation and Training Efficiency

The paper's central practical claim is that data quality matters more than model scale for VLM performance. Several reported results support this:

**CLIP zero-shot performance.** The original CLIP (Radford et al., 2021) trained on 400M image-text pairs achieved 76.2% zero-shot top-1 accuracy on ImageNet with a ResNet-101 backbone, matching a fully supervised ResNet-101 (Section 2.2.1). This is the paper's primary evidence that natural language supervision at scale can produce visual representations competitive with supervised training. No confidence intervals or error bars are reported.

**DataComp benchmark findings.** Section 3.1 reports that the DataComp benchmark (Gadre et al., 2023) — which fixes CLIP architecture and hyperparameters while varying only the dataset — demonstrates that "data pruning is a crucial step in training highly efficient and performant VLMs." The paper reports no specific numbers from DataComp, instead describing filtering strategies (CLIPScore ranking, text spotting, diversity sampling) that improve performance. The claim that pruning matters is supported by citation to DataComp's benchmark design rather than by specific quantitative comparisons reproduced in this paper.

**Synthetic caption improvements.** Section 3.1.1 reports several quantitative claims: Santurkar et al. (2022) found that "models trained on consistent and complete synthetic captions generated by BLIP outperform a model trained on human-written captions" — a surprising result the paper presents without specific numbers. LLaVAR (Zhang et al., 2023c) achieved "up to a 20% accuracy improvement on text-based VQA datasets" when trained on LLM-generated conversations about text-rich images (Section 3.6). The paper treats these as existence proofs that synthetic data can improve performance rather than as benchmark results to be compared across methods.

**Training speed via masking.** Section 3.2.3 reports that Li et al. (2023f) showed "by randomly masking image tokens one can significantly speed up training time while improving model performances." No specific speedup factor or performance delta is provided.

#### Training Paradigm Comparisons

**SigLIP vs. CLIP on batch size sensitivity.** Section 2.2.2 claims SigLIP achieves "better 0-shot performances on smaller batch sizes than CLIP" — a consequence of replacing InfoNCE's batch-dependent softmax with binary cross-entropy. The paper provides no specific numbers for this comparison but frames it as a practical advantage for resource-constrained settings.

**LLaVA-RLHF vs. baselines on hallucination benchmarks.** Section 3.5.1 reports that LLaVA-RLHF (Sun et al., 2023) achieves "94% performance level of GPT-4" on LLaVA-Bench, and on MMHAL-BENCH (a benchmark with "special focus on penalizing hallucinations"), LLaVA-RLHF "outperforms baselines by 60%." The baseline models are not specified in the paper, and the metric for "outperforms by 60%" (absolute accuracy difference? relative error reduction?) is not clarified.

**Generative vs. discriminative classifiers on compositional reasoning.** Section 2.4.3 reports that on Winoground (Thrush et al., 2022), "generative classifiers far outperform discriminative methods like CLIP." Li et al. (2023a) and Clark and Jaini (2023) are cited as showing that diffusion-based and autoregressive classifiers have "better out-of-distribution performance for a given in-distribution accuracy" — what the paper calls "more effective robustness." No specific Winoground scores are reported for CLIP or generative classifiers.

**Pretrained backbone cost comparison.** Section 3.2.2 provides the paper's most concrete resource comparison: CLIP training used "more than 500 GPUs" and cost "hundreds of thousands of dollars" (with OpenCLIP using "256 and 600 GPUs across multiple days or weeks"), while MiniGPT-4 trained on "only four A100 GPUs for around ten hours" (Section 2.5.2). A contrastive model trained from scratch on hundreds of millions of images with good data "should not require more than 64 GPUs (which should be equivalent to spending around 10K USD in compute)" — a ~50× cost reduction relative to CLIP-scale training. These numbers compare different model families doing different tasks (CLIP does retrieval and zero-shot classification; MiniGPT-4 does multimodal dialogue), so they represent a resource comparison rather than an apples-to-apples performance-per-dollar analysis.

#### Evaluation Benchmark Vulnerabilities

**Spatial reasoning on PUG.** Section 4.1.8 reports the paper's most striking negative result: on the synthetic PUG benchmark (Bordes et al., 2023), which evaluates spatial relations through controlled image generation, "current VLMs are not performing better than random chance when evaluating spatial relations." This means models achieving high accuracy on standard VQA and compositional benchmarks collapse to chance-level performance when tested on controlled spatial reasoning tasks. The paper does not specify which VLMs were tested on PUG or report exact numbers, citing the original PUG paper.

**Blind language priors in VQA.** Section 4.2.3 documents that the VQA benchmark (Antol et al., 2015) can be partially solved without visual input: questions starting with "Is there a clock" have answer "yes" 98% of the time (Goyal et al., 2017). More broadly, Lin et al. (2024a) discovered that "a blind language prior (P(text)) estimated using image-captioning models like BLIP perform well on contemporary image-text retrieval benchmarks, including ARO, Crepe, VL-CheckList, and SugarCrepe" (Section 4.2.3). In contrast, "balanced benchmarks like Winoground and EqBen actually penalize unimodal shortcuts." No specific accuracy numbers for blind models on these benchmarks are provided.

**Argmax artifact in binary-choice benchmarks.** Section 4.1.6 warns that "a model whose parameters are all equal to zero could achieve 100% accuracy" on binary-choice benchmarks where the correct answer is consistently placed first. This is because zero-parameter models produce uniform output probabilities, and PyTorch's argmax returns the first element when values are equal. The paper does not report how many existing benchmarks exhibit this vulnerability or provide empirical evidence of models exploiting it, but frames it as an urgent methodological warning.

**Déjà vu memorization.** Section 4.4 reports findings from Jayaraman et al. (2024) that "CLIP models can effectively 'remember' objects present in the training images, even if they are not described in the caption." The quantification uses "the gap between the object detection precision/recall scores of the target and reference models, whereby a larger gap indicates a higher degree of memorization." No specific precision/recall numbers or gap magnitudes are provided. The paper reports that "text randomization to be the most effective regularization technique that significantly reduces memorization without severely penalizing the model utility" — a technique where "a random fraction of text tokens from training captions are masked in each training epoch" (Section 4.4).

---

### Ablation Studies and Robustness Checks

Given that the paper conducts no original experiments, there are no new ablations. However, the paper reports ablation-like findings from prior work that illuminate design choices:

**Batch size dependence in contrastive losses (informal ablation):** SigLIP's binary loss vs. CLIP's InfoNCE represents an implicit ablation of the batch-dependent softmax normalization. The finding — SigLIP achieves better zero-shot performance at small batch sizes (Section 2.2.2) — demonstrates that the multi-class softmax over the batch is the source of CLIP's batch size sensitivity. No specific batch sizes or performance deltas are reported.

**Projector separation in CLIP-rocket:** The finding that "it is crucial to separate the two projectors as the strong one is learning more invariant, too invariant, representations for downstream tasks" (Section 3.1.2, citing Bordes et al., 2022) is an ablation result: using a shared projector for weak and strong augmentations degrades downstream performance because strong augmentations' invariance interferes with weak augmentations' discriminative power. The paper does not report specific performance numbers for shared vs. separate projectors.

**Revision history in verifier context:** Not strictly an ablation in this paper, but referenced from the LLaVA literature: LLaVA 1.5's use of a cross-modal MLP (replacing LLaVA's linear projector) improved performance, demonstrating that the modality fusion mechanism matters. The paper reports qualitative improvements rather than specific ablation numbers.

**Data type mixture for interleaved training:** Section 3.2.4 reports that McKinzie et al. (2024) found "the right mix achieves the best performance across both zero-shot classification and visual-question answering tasks" when combining text-only, image-caption paired, and interleaved data. This is an ablation of data mixture ratios, but no specific ratios or performance numbers are reproduced.

**Text randomization for memorization mitigation:** Section 4.4 reports that "text randomization ... significantly reduces memorization without severely penalizing the model utility" compared to other regularization techniques. This is a comparative ablation of regularization strategies for mitigating déjà vu memorization, but specific comparisons between techniques are not provided.

**Negative result — video temporal understanding:** Section 5.4 reports that on the synthetic physics-understanding benchmark from Jassim et al. (2023), "models such as VideoLLaMA or PandaGPT do not exceed random performance, whereas humans achieve more than 80% accuracy." This is a clear negative result: current video VLMs lack basic physical reasoning capabilities that are trivially easy for humans.

---

### Critical Assessment

The paper's central claim is that it provides a "clear and easy-to-understand introduction to VLM research" highlighting "effective practices for research in this space" (Section 1). This is fundamentally a pedagogical claim, not an empirical one: the paper's success should be measured by whether it organizes knowledge in a way that lowers the barrier to entry, not by whether it advances state-of-the-art performance.

**The paper succeeds as a pedagogical resource** in several concrete ways that are unusual for academic surveys. The progression from energy-based models through NCE to InfoNCE (Section 2.2) builds contrastive learning from first principles rather than presenting CLIP's loss function as received wisdom. The information-theoretic unification of masking and contrastive objectives (Section 2.3.3) provides conceptual coherence that enables reasoning across paradigms rather than memorizing them separately. The practical guidance on GPU requirements, software tools, and data loading bottlenecks (Sections 3.2.2–3.2.3) contains operational knowledge that is almost never found in research papers yet is precisely what a new practitioner needs to know to go from reading to doing.

**However, the paper's strength as a pedagogical introduction is inseparable from its limitations as an empirical document.** Several critical claims are presented without the quantitative support that would be expected in a research paper:

**The claim that "data pruning is a crucial step" (Section 3.1) is well-motivated by citations to DataComp and the CLIP literature, but the paper provides no comparative numbers.** A newcomer reading this section learns that pruning matters but cannot answer basic practical questions: How much does CLIPScore filtering improve accuracy? At what filtering threshold do returns diminish? The DataComp benchmark was specifically designed to answer these questions, but the paper treats it as a citation rather than mining it for actionable numbers. This pattern recurs throughout: synthetic captions "improve" performance (by how much?), masking "significantly speed[s] up training" (by what factor?), and LLaVA-RLHF "outperforms baselines by 60%" (absolutely or relatively? on what metric?). The paper's decision to omit specific numbers may be pedagogically motivated — avoiding overwhelming newcomers with a firehose of percentages — but it leaves the reader unable to assess the magnitude of claimed effects.

**The spatial reasoning result on PUG (Section 4.1.8) is the paper's most striking negative finding — "current VLMs are not performing better than random chance" — but the paper does not specify which VLMs were tested, at what scale, or with what prompting strategy.** A skeptical reader might wonder: was the 76.2% ImageNet CLIP model tested on PUG, or a smaller variant? Were the models evaluated in a zero-shot setting, or were they fine-tuned? Did any model achieve even slightly above chance, or was performance uniformly at chance? Without these specifics, the claim that current VLMs "are not performing better than random chance" is imprecise. The original PUG paper (Bordes et al., 2023) presumably provides these details, but this paper — which positions itself as a self-contained introduction — requires the reader to consult external sources to verify one of its most important claims.

**The evaluation pitfall warnings (Sections 4.1.6, 4.2.3) are valuable but would benefit from empirical quantification.** The argmax artifact warning is well-described, but the paper does not report how many popular benchmarks exhibit this vulnerability, whether any published results are suspected of being inflated by it, or what fraction of models tested actually produce uniform outputs. Similarly, the blind language prior warning (Section 4.2.3) accurately describes the VQA benchmark's vulnerability to unimodal shortcuts, but does not provide the blind model's accuracy on VQA — a number that would dramatically illustrate the problem. A reader learning about these pitfalls for the first time might reasonably ask: are these edge cases affecting 1% of evaluations, or systematic problems affecting the majority of published benchmarks? The paper does not provide the evidence to distinguish these scenarios.

**The cost comparison between CLIP-scale training and pretrained-backbone fine-tuning (Sections 3.2.2, 3.3.4) is the paper's most actionable practical guidance, but it compares fundamentally different capabilities.** CLIP trained on 400M image-text pairs can perform zero-shot classification on 1000 ImageNet classes, text-to-image retrieval, and image-to-text retrieval. MiniGPT-4 (trained on 5M pairs with a frozen LLM) can generate captions and engage in multimodal dialogue. These are not the same task, and the paper's claim that pretrained backbones offer a resource-constrained alternative implicitly assumes that the researcher's downstream task is within the capabilities of the pretrained-backbone approach. For a researcher who needs a robust visual encoder for downstream fine-tuning (the primary use case for CLIP embeddings), MiniGPT-4 is not a substitute — it is a different tool for different tasks. The paper's framing of "which model to use" (Section 3.3) partially addresses this by specifying when each paradigm is appropriate, but the cost comparison risks being misinterpreted as "pretrained backbones are cheaper and equally capable" when the actual finding is "pretrained backbones are cheaper and have different capabilities."

**The paper's claim to provide "effective practices" (Section 1) is unevenly supported.** Some recommendations are concrete and actionable: store data uncompressed to speed training; use FFCV for faster data loading; separate projectors for strong and weak augmentations; add epsilon noise or track equal-probability cases in binary-choice benchmarks. Others are vague: "having a high-quality dataset" (how is quality measured?); "leveraging masking strategies when using bigger models" (at what model size does masking become beneficial?); "the right mix achieves the best performance" (what is the right mix?). The uneven specificity reflects the paper's reliance on cited prior work: when the original paper provides concrete findings, this paper transmits them; when the original paper is itself qualitative, this paper cannot add precision without conducting original experiments.

**The absence of any original experiments is both the paper's defining characteristic and its primary limitation.** A pedagogical introduction that also included a small controlled experiment — for example, demonstrating the argmax artifact on a popular benchmark, or measuring CLIPScore's correlation with downstream performance at different filtering thresholds — would transform several of the paper's warnings from plausible concerns into demonstrated facts. The paper's reliance on external citations for all quantitative claims means that a reader who wants to verify any specific number must trace the full citation chain, which partially defeats the paper's purpose as a self-contained introduction. This is not a failure of rigor — the paper is transparent about being "not a survey" and "not a complete guide" — but it means the paper's authority derives from its synthesis and framing rather than from empirical demonstration. For a reader who trusts the authors' curation and interpretation, this is sufficient. For a reader who wants to see the evidence directly, the paper is a starting point, not a destination.

## 6. Limitations and Trade-offs

### 1. The Paper Has No Original Experiments to Support Its Most Actionable Claims

The paper explicitly states in Section 1 that it "should not be considered as a survey or a complete guide on VLMs" and does "not aim to cite every work from the VLM research field." This is a transparent and honest framing, but it conceals a deeper limitation: the paper's most practically valuable contributions—warnings about benchmark pitfalls, claims about spatial reasoning failures, and cost comparisons between training paradigms—are reported from prior work without independent verification or controlled demonstration.

**What this means in practice.** The paper's most memorable and actionable claim is the warning in Section 4.1.6 about the argmax artifact: "a model whose parameters are all equal to zero could achieve 100% accuracy in these benchmarks." This is a devastating critique of standard evaluation practice, but the paper provides no empirical demonstration. It does not report which specific benchmarks exhibit the vulnerability, test existing models to see whether they produce uniform outputs, or estimate how many published results might be inflated by the artifact. A reader encountering this warning may wonder: is this a theoretical possibility or a demonstrated problem? The paper does not provide the evidence to answer that question.

The same pattern applies to the spatial reasoning claim in Section 4.1.8: "current VLMs are not performing better than random chance when evaluating spatial relations." This is the paper's headline finding about fundamental capability gaps in VLMs, but it does not specify which VLMs were tested, at what scale, or under what prompting conditions. It attributes the finding entirely to Bordes et al. (2023). The reader must leave this paper and consult the original PUG paper to understand the scope and strength of the evidence for this claim.

The cost comparison in Section 3.2.2—CLIP training at "hundreds of thousands of dollars" versus MiniGPT-4 at "four A100 GPUs for around ten hours"—is compelling but compares different model families performing different tasks under different training protocols. The paper does not define a unified compute metric (FLOPs, GPU-hours) or attempt to normalize for capability. A practitioner trying to decide between training CLIP from scratch and fine-tuning a pretrained backbone cannot extract a true apples-to-apples cost-effectiveness comparison from the paper's numbers.

**Mitigation status.** The paper does not attempt to address this limitation—indeed, the limitation follows directly from the paper's stated purpose. The authors position the work as a pedagogical introduction, not an empirical investigation. This is a legitimate choice, but it means the paper's authority derives from synthesis and interpretation rather than from demonstrated fact. The limitation is structural: by choosing not to conduct original experiments, the paper cannot independently verify or quantify the claims it surfaces from prior work. The consequence is that a reader who wants to verify the strength of evidence for any specific quantitative claim must trace the full citation chain. For a paper that explicitly aims to serve "students or researchers in other areas who want to enter the field" (Section 1), this partial reliance on external verification is a non-trivial barrier.

---

### 2. The Recommendations Are Qualitative and Lack Actionable Thresholds

Section 3 positions itself as "A Guide to VLM Training" and makes numerous practical recommendations. However, these recommendations are almost entirely qualitative—describing what matters without specifying how much it matters, at what scale it matters, or what specific thresholds should guide decisions.

**What the paper says versus what a practitioner needs.** Section 3.2.3 recommends storing "as many uncompressed files as possible to speed up training." This is directionally correct but operationally incomplete. How much speedup does uncompressed storage provide? At what dataset size does the storage cost outweigh the compute savings? Should the practitioner store everything uncompressed, or only the most frequently accessed shards? Section 3.2.4 reports that "image resolution, visual encoder capacity, and visual pretraining data are the choices that most impact model performance" but provides no guidance on how to choose resolution (is doubling resolution worth 4× the compute?), what "visual pretraining data" means concretely (ImageNet? COCO? LAION?), or how to trade off these three factors against each other when operating under a fixed compute budget.

Section 3.3.1 recommends using contrastive models like CLIP when the practitioner needs representations with "meaning in both the image and text space" and when the model will be "a good base for building more complex models." These are conceptual guidelines, not decision criteria. At what scale does CLIP become preferable to a pretrained-backbone approach for a given downstream task? How does one quantify "good enough" for the base representations? The paper provides no frameworks for making these choices quantitatively.

Section 3.1 emphasizes that "data pruning is a crucial step in training highly efficient and performant VLMs" and describes filtering strategies (CLIPScore ranking, text spotting, diversity sampling) without comparing their effectiveness. A practitioner implementing CLIPScore filtering needs to know: what CLIPScore threshold should be used? How does the choice of threshold trade off dataset size against quality? Does the optimal threshold depend on the downstream task? The DataComp benchmark was specifically designed to answer these questions, but the paper cites it without mining it for actionable thresholds.

**Mitigation status.** The paper does not frame this as a limitation; it presents its qualitative guidance as the intended product. The limitation arises from the gap between the paper's identity as a pedagogical introduction—where conceptual understanding is the goal—and its simultaneous aspiration to provide "effective practices for research in this space" (Section 1). Effective practices in machine learning typically require quantitative decision thresholds, but the paper operates almost entirely at the level of principles and directions. For a newcomer who needs to make specific implementation choices (what batch size? what learning rate? what CLIPScore threshold?), the paper provides conceptual orientation but not operational specifics. The vast majority of these specifics exist in the original papers the present work cites; the limitation is that this paper does not extract and organize them into a usable reference.

---

### 3. The Taxonomy Overstates the Cleanliness of the Design Space

Section 2 organizes VLMs into four families—contrastive, masking-based, generative, and pretrained-backbone—and the paper's pedagogical strategy depends on this categorization as a conceptual map. However, the paper itself acknowledges in the caption of Figure 1 that "these paradigms are not mutually exclusive; many approaches rely on a mix of contrastive, masking, and generative criteria." This acknowledgment raises a question the paper does not fully address: if the boundaries between families are porous in practice, how useful is the taxonomy as a guide to action?

**The taxonomy as a teaching device versus the taxonomy as a design tool.** The paper's Section 3.3 provides explicit decision criteria for choosing between paradigms: use CLIP when you need shared representations, use masking when you want to avoid batch dependency, use generative models when you need image generation, use pretrained backbones when compute is limited. But modern VLMs routinely mix these objectives. FLAVA (Section 2.3.1) combines masked image modeling, masked language modeling, and a contrastive loss—it is simultaneously a masking model and a contrastive model. CoCa (Section 2.4.1) combines a contrastive loss with a generative captioning loss—it is simultaneously contrastive and generative. A practitioner reading Section 3.3 might reasonably ask: "Should I use FLAVA or CoCa?" The taxonomy does not provide a clear answer because these models span categories. The decision criteria in Section 3.3 apply cleanly to the pure cases (pure CLIP, pure generative) but do not address how to reason about hybrid models that are increasingly the norm.

The paper's taxonomy also creates a false sense of architectural orthogonality. The contrastive family is defined by its training objective (InfoNCE), the masking family by its training objective (reconstruction), the generative family by its output capability (generating text/images), and the pretrained-backbone family by its initialization strategy (frozen pretrained components). These are not dimensions of the same space; they are different kinds of properties. A model can be both contrastive and use a pretrained backbone (e.g., fine-tuning CLIP), or both generative and use masking (e.g., masked autoregressive models). The taxonomy conflates objective function, output capability, and training strategy into a single flat categorization, which simplifies the landscape for newcomers but obscures the actual combinatorial design space that practitioners navigate.

**Mitigation status.** The paper acknowledges the fuzziness of the boundaries but treats this as a minor caveat rather than a structural limitation of the taxonomy. Section 3.3 provides decision criteria that assume the practitioner is choosing between pure forms of each paradigm, which does not match the reality of contemporary VLM development where hybrid approaches dominate. The limitation is not that the taxonomy is wrong—it is pedagogically useful, and the information-theoretic unification in Section 2.3.3 partly bridges the categories by showing they optimize the same underlying objective—but that the paper does not provide guidance on how to design or evaluate hybrid models. A newcomer who internalizes the four-family taxonomy may struggle to understand models that do not fit neatly into a single box, which includes many of the most important recent VLMs.

---

### 4. The Evaluation Pitfall Warnings Are Not Quantified or Systematically Surveyed

Sections 4.1.6, 4.2.3, and 4.4 contain some of the paper's most practically valuable content: warnings about benchmark artifacts, blind language priors, and memorization that can produce misleading evaluation results. However, these warnings are presented as qualitative observations rather than as a systematic audit of the VLM evaluation landscape.

**What the paper warns about but does not measure.** Section 4.1.6 warns that binary-choice benchmarks where the correct answer appears first can be solved by zero-parameter models due to argmax behavior. The paper recommends adding "a small epsilon random number or keeping track if the captions are assigned the same probabilities." But it does not report which specific benchmarks exhibit this vulnerability, what fraction of published VLM evaluations use benchmarks with this property, or whether any existing models actually produce uniform outputs on these benchmarks. A practitioner reading this section learns that a potential problem exists but cannot assess its prevalence or severity.

Section 4.2.3 documents that the VQA benchmark (Antol et al., 2015) contains blind language priors: "questions starting with 'Is there a clock' has the answer 'yes' 98% of the time." The paper also reports that Lin et al. (2024a) found blind language priors perform well on "contemporary image-text retrieval benchmarks, including ARO, Crepe, VL-CheckList, and SugarCrepe." But the paper does not report the accuracy of blind models on these benchmarks. A newcomer reading this section might reasonably ask: how much of reported VLM performance on these benchmarks is attributable to visual understanding versus language shortcuts? If blind models achieve 60% accuracy and VLMs achieve 65%, the benchmark is measuring mostly language priors. If blind models achieve 30% and VLMs achieve 65%, the benchmark is measuring mostly visual understanding. Without these numbers, the severity of the problem is unclear.

Section 4.4 reports that Jayaraman et al. (2024) found CLIP models exhibit déjà vu memorization—"remembering" objects in training images even when they are not described in the caption. The paper describes the quantification method (gap between target and reference model precision/recall) but does not report the magnitude of this gap. How severe is the memorization? At what training set size does it become problematic? Does it affect all object categories equally or concentrate on frequent ones? The paper does not provide answers.

**Mitigation status.** The paper does not acknowledge this as a limitation; it presents these warnings as contributions. The limitation is one of depth: the warnings are valuable as flags, but without quantification they are difficult to act on. A researcher designing a new VLM evaluation cannot determine from this paper alone whether a proposed benchmark design is vulnerable to these artifacts. The paper raises awareness but does not provide the systematic analysis needed to solve the problems it identifies. For a field where benchmark design directly shapes research priorities, the absence of quantitative characterization of these pitfalls limits the paper's ability to change practice.

---

### 5. The Paper Does Not Address Latency, Deployment, or Inference-Time Practicalities

The paper's practical guidance (Section 3) focuses almost exclusively on training considerations: how to curate data, which software to use, how many GPUs training requires, and how to choose a training paradigm. There is no corresponding treatment of inference-time considerations: latency, throughput, memory footprint, quantization, or deployment hardware constraints.

**Why this matters for a practitioner.** A researcher who follows the paper's guidance to train a VLM successfully will immediately confront deployment questions the paper does not address. Generative VLMs (like Chameleon or CoCa) produce outputs autoregressively—token by token—which imposes serial latency that contrastive models (like CLIP) avoid. A diffusion-based zero-shot classifier (Section 2.4.3) requires "hundreds or thousands of network evaluations per test image," making it impractical for real-time applications. The paper's recommendation to use pretrained backbones (Section 3.3.4) is motivated by training cost but does not discuss whether the resulting models have acceptable inference latency or whether the frozen LLM's memory footprint is compatible with on-device deployment.

Section 3.6 discusses text-rich image understanding and notes that Lumos uses a decoupled STR module running on-device to "reduce power and latency from transferring high-resolution images to the cloud." This is the only substantive discussion of deployment architecture in the paper. There is no treatment of model compression (quantization, pruning, distillation), no comparison of inference FLOPs across model families, and no guidance on choosing models based on latency budgets. For a paper that aims to provide "a guide to VLM training" and to help practitioners make decisions "given different research goals," the omission of inference-time considerations is consequential: a model that is cheap to train but expensive to serve may be the wrong choice for a production application, and vice versa.

**Mitigation status.** The paper does not address inference-time considerations at all, and does not acknowledge this as a limitation. Section 3.2 focuses on training software (torch.compile, xformers, FFCV) and training-time speedups. Section 3.3 provides model selection criteria that consider training cost and capabilities but not deployment constraints. The paper's scope—introducing VLMs to newcomers—could reasonably include a discussion of deployment considerations, since a practitioner's end goal is typically to use the model, not just to train it. The absence of this discussion means a reader who follows the paper's guidance through training will be unprepared for the next stage of the pipeline.

---

### 6. The Discussion of Video VLMs Highlights Fundamental Capability Gaps but Offers No Diagnostic Framework

Section 5 extends the paper's treatment to video-language models and contains one of the paper's most striking negative results: on synthetic benchmarks testing physics understanding, "models such as VideoLLaMA or PandaGPT do not exceed random performance, whereas humans achieve more than 80% accuracy" (Section 5.4). This finding—that current video VLMs lack basic physical reasoning—parallels the spatial reasoning failure on PUG for image VLMs and suggests a systematic gap between surface-level visio-linguistic alignment and genuine world understanding.

**What is missing.** The paper identifies this failure but does not provide a diagnostic framework for understanding *why* video VLMs fail at physical reasoning or what would be required to close the gap. The image VLM sections provide some diagnostic structure: the taxonomy (contrastive vs. masking vs. generative) helps reason about capabilities, the evaluation section (Section 4) characterizes specific failure modes (blind language priors, argmax artifacts, spatial reasoning failures), and the training guide (Section 3) offers techniques for improving grounding and alignment. The video section (Section 5) is comparatively underdeveloped: it describes architectures (VideoBERT, MERLOT, VideoOFA, Video-LLaMA, MiniGPT4-Video) and evaluation benchmarks but does not connect capability failures to architectural choices or training data limitations.

Section 5.5 identifies challenges in video data: "a challenge for video-text pretraining is the current scarcity of (weak) supervision on temporal space," "CLIP models trained on video can also exhibit a noun bias which makes it harder to model interactions," and "processing videos is more expensive than images yet it's an even more redundant modality." These are important observations, but the paper does not connect them back to the physical reasoning failure. Is the failure due to insufficient temporal supervision in training data? To architecture limitations (treating videos as bags of frames)? To the noun bias preventing verb and action understanding? The paper does not provide experimental evidence or even a structured hypothesis space for diagnosing these failures.

**Mitigation status.** The paper presents Section 5 as a forward-looking extension, acknowledging that video VLMs are less mature than image VLMs. It does not claim to provide the same level of diagnostic depth for video as for images, and the section ends with the observation that "all of these challenges, whether regarding pretraining data, compute or quality of evaluations, point to promising research directions towards video VLMs with better understanding of the world" (Section 5.5). This is a reasonable scope limitation for a paper primarily focused on image-language models, but it means the paper's strongest cautionary finding—that current video VLMs lack basic physical reasoning—is presented without the analytical scaffolding that would help a newcomer understand and address it. The section identifies an important failure mode without providing a framework for investigating it, which limits its practical utility for researchers trying to build better video VLMs.

## 7. Implications and Future Directions
- How this work changes the field’s landscape
  - Provides a shared vocabulary (Figure 1 taxonomy; rate–distortion view) and an “engineering playbook” (Figure 2 and Section 3) to design VLMs deliberately rather than by trial and error. It also sharpens evaluation practice (Figure 3; Section 4), warning against common traps.

- Follow‑up research enabled or suggested
  - Objective design: Mix contrastive/masking/generative in principled ways guided by the rate–distortion perspective (Section 2.3.3).
  - Generative‑discriminative fusion: Practical hybrids that get generative classifiers’ robustness without prohibitive inference cost (Section 2.4.3).
  - Data‑centric VLMs: Automated, holistic multimodal data quality measures; concept‑coverage planning; interleaved data construction at scale (Sections 3.1.3–3.1.4).
  - Grounding and alignment: Large‑scale, high‑quality pseudo‑grounding pipelines (GLIP‑style) plus instruction‑tuning/RLHF with factual constraints (Sections 3.4–3.5).
  - Reliable evaluation: Benchmarks resilient to language priors, with selective prediction, dense grounding, and synthetic diagnostics (Sections 4.1–4.2).
  - Video VLMs: Efficient temporal modeling, better supervision for actions/motion, physics‑aware reasoning (Section 5.5).

- Practical applications and use cases
  - Retrieval, captioning, visual dialog, OCR‑VQA, and document understanding (Sections 4.1.1–4.1.4).
  - Multimodal assistants (LLaVA‑style) aligned via instruction tuning/RLHF for safer deployment (Section 3.5).
  - Content moderation and fairness auditing using bias/embedding analyses and red teaming (Sections 4.2, 4.5).
  - Video search and understanding for education, robotics, and AR, with a roadmap to handle long‑context reasoning and physics (Section 5.4).

> Bottom line: If you need to build or evaluate a VLM today, use Figure 1 to pick an objective family, Section 3 to design a data‑efficient training pipeline (with grounding/alignment add‑ons), and Section 4 to choose robust evaluations—keeping in mind the specific pitfalls (language priors, binary tie bug) and the trade‑offs of each method.
