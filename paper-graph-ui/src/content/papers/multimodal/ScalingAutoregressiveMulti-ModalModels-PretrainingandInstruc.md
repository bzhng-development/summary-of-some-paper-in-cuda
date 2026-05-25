# Scaling Autoregressive Multi-Modal Models: Pretraining and Instruction Tuning

**ArXiv:** [2309.02591](https://arxiv.org/abs/2309.02591)

## 🎯 Pitch

CM3Leon introduces a single retrieval-augmented, autoregressive multi-modal model capable of both generating and editing text and images using a unified decoder-only architecture. By adapting large language model pretraining and instruction tuning recipes to the multi-modal setting, and integrating a novel contrastive decoding method, CM3Leon achieves state-of-the-art text-to-image performance with dramatically reduced compute cost—demonstrating that LLM-style scaling and flexibility can transform how vision and language models are trained and applied. This breakthrough paves the way for more general, controllable, and efficient AI systems capable of seamlessly bridging text and image domains in practical, real-world workflows.

---

## 1. Executive Summary

This paper introduces **CM3Leon**, a retrieval-augmented, token-based, decoder-only multi-modal language model that generates and infills both text and images, and demonstrates that adapting a text-only language model training recipe — large-scale retrieval-augmented pretraining followed by multi-task supervised fine-tuning (SFT) — to the multi-modal setting yields state-of-the-art performance with dramatically improved efficiency. On the zero-shot MS-COCO text-to-image benchmark, CM3Leon achieves an FID of 4.88 while using 5× less training compute than comparable methods (e.g., PARTI), establishing that autoregressive models can surpass diffusion models in both quality and cost-effectiveness when trained with retrieval augmentation and a self-contained contrastive decoding method (CD-K, which subtracts log-probabilities between conditional and unconditional token streams to guide sampling). The model further demonstrates unprecedented controllability after SFT across tasks including language-guided image editing, image-to-image grounded generation, and visual question answering, though the paper's findings establish that the benefits of this recipe are most pronounced when the model can leverage retrieval augmentation during both training and inference, with zero-shot FID degrading from 4.88 (2 retrieved documents) to 10.82 (0 retrieved documents).

## 2. Context and Motivation

### The Core Problem: Autoregressive Models Have Been Left Behind in Multi-Modal Generation

The fundamental problem this paper addresses is a perceived **inefficiency and narrowness** in autoregressive token-based models for multi-modal generation tasks. As of early 2023, diffusion models (Rombach et al., 2022; Saharia et al., 2022; Ramesh et al., 2022) had become the de facto standard for text-to-image generation, widely perceived as both more performant and more computationally practical than their autoregressive counterparts. The paper opens by explicitly stating this conventional wisdom:

> "Diffusion models have recently dominated image generation work due to their strong performance and relatively modest computational cost. In contrast, token-based autoregressive models... are known to also produce strong results, with even better global image coherence in particular, but are much more expensive to train and use for inference."

This framing sets up a clear **narrative to flip**: autoregressive models, the paper argues, can be made efficient and performant while also generalizing beyond text-to-image to a wide range of mixed-modal tasks — but only if they borrow training and inference techniques originally developed for text-only language models.

The gap, then, is not that autoregressive models are incapable — prior work like DALL-E (Ramesh et al., 2021) and PARTI (Yu et al., 2022) had already shown they can produce high-quality images — but rather that **no one had successfully adapted the full training recipe of modern large language models (LLMs) to the multi-modal autoregressive setting**. Text-only LLMs by this point had established a powerful two-stage paradigm: large-scale pretraining (often with retrieval augmentation) followed by multi-task supervised instruction tuning (Iyer et al., 2022). Multi-modal models, by contrast, were typically trained in a single stage for a narrow set of tasks (usually text-to-image generation) and lacked the instruction-tuning phase that had proven so critical for the flexibility and controllability of text-only LLMs like ChatGPT.

### Why This Gap Matters: Practical and Scientific Significance

The importance of closing this gap spans both practical deployment and scientific understanding.

**From a practical standpoint**, the dominance of diffusion models created a fork in the research ecosystem. Techniques developed for one model family (e.g., classifier-free guidance for diffusion, instruction tuning for autoregressive LLMs) did not transfer cleanly to the other. Practitioners building multi-modal systems faced an uncomfortable choice: use diffusion models for image generation quality and efficiency, or use autoregressive models for their unified sequence modeling paradigm (which naturally handles interleaved text and image tokens, infilling, and arbitrary input-output combinations). There was no model that combined the efficiency and performance of diffusion with the flexibility and tunability of autoregressive LLMs. CM3Leon aims to be that unified model.

**From a scientific standpoint**, the paper challenges the assumption that different generative modeling paradigms (autoregressive vs. diffusion) have fundamentally different scaling properties. Figure 2 (a log-scale plot of FID against equivalent A100 GPU hours) is the key piece of evidence here: the CM3Leon model family (350M, 760M, 7B parameters) shows a scaling curve that is steeper and lower (better) than both diffusion models (Stable Diffusion v1.5, v2.1) and prior autoregressive models (DALL-E, PARTI). The claim embedded in this figure is not just that CM3Leon performs well, but that **autoregressive models scale better with compute when trained with the right recipe** — retrieval augmentation, the CM3 objective, and the specific data and architectural choices described in Section 2. If true, this has implications for how the field should allocate research effort: it suggests that continued investment in autoregressive multi-modal models may yield higher returns than the then-dominant focus on diffusion.

**A second scientific contribution** is demonstrating that the **supervised fine-tuning (SFT) stage** — which was by this point well-established for text-only LLMs — transfers to multi-modal settings and unlocks capabilities that single-stage models lack. The paper's SFT stage (Section 4) takes the pretrained CM3Leon and fine-tunes it on a mixture of text-to-image, image-to-text, and image-to-image tasks using the same CM3 objective. The result is a single model that can do text-guided image editing, structure-conditioned generation, visual question answering, and long-form captioning — tasks that would typically require separate specialized models. This unification is significant because it suggests that the multi-task instruction tuning paradigm is not specific to text, but is a general property of sequence models trained on diverse token streams.

### Where Prior Approaches Fall Short

The paper identifies shortcomings in prior work across four categories of models:

**1. Diffusion models (the dominant paradigm).** While diffusion models like Stable Diffusion and Imagen achieved strong text-to-image performance, they had structural limitations that the paper implicitly critiques:

- **Task narrowness**: Diffusion models are inherently designed for image generation (or image-to-image translation). They do not naturally produce text outputs, answer questions about images, or perform the kind of mixed-modal generation that an autoregressive sequence model handles natively. Each new task typically requires a different model architecture or training procedure.
- **Lack of a unified instruction-tuning paradigm**: By 2023, there was no established method for fine-tuning a diffusion model on a broad mixture of vision-and-language tasks in the way that OPT-IML (Iyer et al., 2022) had done for text-only LLMs. The paper positions CM3Leon's SFT stage as filling this gap for autoregressive models specifically.
- **Retrieval augmentation was underexplored in practice**: Although retrieval-augmented diffusion models existed (KNN-Diffusion, RE-IMAGEN; Chen et al., 2022), they were relatively recent and not part of the standard training recipe. The paper shows that retrieval augmentation is critical for efficient training, with retrieval during inference providing substantial FID improvements (Table 1: 4.88 with 2 retrieved documents vs. 10.82 with 0).

**2. Prior autoregressive token models (DALL-E, PARTI, Make-A-Scene).** The key predecessors demonstrate capability but fall short on efficiency and scope:

- **DALL-E** (Ramesh et al., 2021): Established that a discrete VAE tokenizer + autoregressive transformer could generate coherent images from text. However, its training was expensive, and its inference required temperature sampling and a computationally costly re-ranking stage over 512 candidates using CLIP. It was a single-purpose text-to-image model with no SFT stage.
- **PARTI** (Yu et al., 2022): Scaled the autoregressive approach to 20B parameters and achieved strong results (FID of 7.23 on zero-shot MS-COCO), but used 5× more training compute than CM3Leon (as the paper claims in its abstract and Section 3.2). PARTI used classifier-free guidance to reduce the re-ranking burden to 16 samples but still operated as a text-to-image specialist. It did not incorporate retrieval augmentation.
- **Make-A-Scene** (Gafni et al., 2022a): Introduced token-based classifier-free guidance for autoregressive models, which CM3Leon adopts. However, it did not include retrieval augmentation or an SFT stage.

**3. Retrieval-augmented models (RA-CM3, KNN-Diffusion, RE-IMAGEN).** The direct predecessor is RA-CM3 (Yasunaga et al., 2022), which introduced retrieval-augmented pretraining for the CM3 architecture. CM3Leon builds directly on RA-CM3 but identifies specific shortcomings:

- **Objective function weight imbalance**: RA-CM3 up-weighted the query image-caption pair loss relative to retrieved documents to encourage the model to focus on using retrieved samples during generation. However, CM3Leon finds that "this method adversely affects the zero-shot scenario, where the goal is to generate an image without retrieval" (Section 2.2). The paper removes this weighting entirely.
- **Masking across modality boundaries**: The original CM3 objective allowed masking spans that crossed the `<break>` token (the modality transition marker). CM3Leon prevents this, justified by the observation that "allowing masking across `<break>` tokens may lead to the model generating image content from an arbitrary midpoint, which is not a desirable outcome" (Section 2.2). This is a subtle but important design choice: it ensures that infilling operations respect modality boundaries, preventing the model from learning to complete partial images starting from mid-token sequences.
- **Data and scale**: RA-CM3 was trained on 150M examples with a 2.7B parameter model, achieving an FID of 15.70 (Table 1). CM3Leon uses a larger, fully licensed dataset (Shutterstock, 340M examples) with models up to 7B parameters, trained to 2.4T tokens, and achieves an FID of 4.88 — a 3× improvement.
- **No SFT stage**: RA-CM3 was a pretrained model only. CM3Leon demonstrates that adding a multi-task SFT stage dramatically expands the model's capabilities.

The paper also distinguishes CM3Leon from retrieval-augmented diffusion models (KNN-Diffusion, RE-IMAGEN) primarily on the basis of performance (Table 1) and the unified sequence modeling paradigm that the autoregressive approach enables.

**4. Instruction-tuned vision-language models (Flamingo, OpenFlamingo).** On the image-to-text side, Flamingo (Alayrac et al., 2022) had demonstrated that vision-language models could achieve strong few-shot performance across captioning and VQA tasks. However, Flamingo uses a fundamentally different architecture (perceiver resamplers connecting a frozen vision encoder to a frozen language model) and was trained on substantially more text data (100B tokens vs. CM3Leon's ~3B Shutterstock text tokens). CM3Leon shows competitive zero-shot performance on several vision-language tasks despite seeing far less text data (Table 2), even beating Flamingo on VizWiz (37.6 vs. 28.8). This suggests that the token-based decoder-only architecture — where image tokens and text tokens share the same transformer — may be more data-efficient for learning vision-language connections than architectures with separate encoders and cross-attention mechanisms.

### How CM3Leon Positions Itself Relative to Existing Work

CM3Leon's positioning can be understood along three axes:

**First, as a training recipe contribution rather than an architectural novelty.** The paper does not introduce a fundamentally new model architecture. It uses the same CM3 framework as Aghajanyan et al. (2022) and the same retrieval-augmented pretraining approach as RA-CM3 (Yasunaga et al., 2022). The novel contribution is the **integration and scaling** of techniques from text-only LLM training: (1) removal of the query up-weighting from RA-CM3 to improve zero-shot performance, (2) prevention of cross-modality masking, (3) a fully licensed dataset at larger scale, (4) a multi-task SFT stage adapted from OPT-IML (Iyer et al., 2022), and (5) the CD-K contrastive decoding method adapted from text-only contrastive decoding (Li et al., 2022). The paper's message is that **following the LLM recipe matters more than architectural novelty** — a claim that directly challenges the diffusion-dominant narrative in the field.

**Second, as an efficiency argument for autoregressive models.** Figure 2 is the centerpiece of this positioning: by plotting FID against equivalent A100 GPU hours on a log scale, the paper makes the case that CM3Leon's scaling curve lies below (better FID at equivalent compute) and is steeper than both diffusion and prior autoregressive models. The 5× less training compute claim in the abstract is a comparison to PARTI, but the figure suggests the advantage extends more broadly. The paper is not claiming autoregressive models are inherently better — it is claiming that **autoregressive models trained with the right recipe** are better, and that the recipe (retrieval augmentation, CM3 objective modifications, SFT) is what makes the difference.

**Third, as a demonstration of generality.** The SFT results in Section 4 are positioned as evidence that the token-based decoder-only paradigm is uniquely flexible. Because the model treats images and text as a single token stream (separated by `<break>` tokens), it can be fine-tuned on tasks with arbitrary mixtures of image and text in both inputs and outputs — text-guided image editing, structure-conditioned image generation, spatial grounding, VQA, and long-form captioning — all using the same next-token prediction objective. This is a capability that diffusion models, by their architectural nature, cannot match without auxiliary components. The paper implicitly argues that this generality is not just a nice-to-have but is the future of multi-modal AI: a single model that can be instructed to perform any vision-and-language task.

### The Tension the Paper Resolves

Underlying all of this is a tension in the field circa 2023: **diffusion models were winning on benchmarks and deployment, but autoregressive models offered a more unified and flexible paradigm**. The diffusion community had optimized heavily for FID on MS-COCO, achieving impressive results with relatively small models. The autoregressive community had demonstrated more general capabilities (infilling, interleaved generation) but at higher computational cost and with worse benchmark numbers. CM3Leon's core claim is that this tradeoff was a false dichotomy — the right training recipe can make autoregressive models simultaneously more efficient, higher-performing, and more general than diffusion models. Whether this claim has held up in the subsequent literature is a separate question, but the paper's framing and evidence make a clear case for why researchers should reinvest in the autoregressive paradigm.

## 3. Technical Approach

### 3.1 Reader Orientation

CM3Leon is a **retrieval-augmented, token-based, decoder-only transformer** that treats images and text as a single unified sequence of discrete tokens, enabling it to generate and infill both modalities using the same next-token prediction objective. The core problem it solves is the perceived inefficiency and task narrowness of prior autoregressive multi-modal models: CM3Leon shows that by adapting the full training recipe of modern text-only LLMs — specifically, retrieval-augmented pretraining with a cleaned-up CM3 objective, followed by multi-task supervised instruction tuning, and capped with a self-contained contrastive decoding method — autoregressive models can surpass diffusion models in both image quality and computational efficiency while simultaneously handling a far broader range of text-and-image tasks than any single diffusion model can.

### 3.2 Big-Picture Architecture (Diagram in Words)

The system has six major components, organized into two stages (pretraining and fine-tuning) and an inference procedure:

1. **Image Tokenizer** (`$f_{\text{img}}$`): a pretrained VQ-VAE encoder that converts a $256 \times 256$ RGB image into 1024 discrete tokens from a vocabulary of 8,192. This is the bridge between continuous pixel space and the discrete token space the transformer operates on.

2. **Text Tokenizer** (`$f_{\text{text}}$`): a custom byte-pair encoding (BPE) tokenizer with a vocabulary of 56,320 tokens, trained on the OPT text corpus (Zhang et al., 2022). It converts natural language prompts into variable-length sequences of text tokens.

3. **Dense Retriever** (`$R$`): a CLIP-based bi-encoder (ViT-B-32, frozen) that encodes multi-modal documents (image + caption) into a single vector by averaging separately-computed CLIP text and image embeddings. During training, for every caption-image pair, the retriever fetches relevant but diverse documents from a memory bank of the Shutterstock training set using maximum inner product search (MIPS), and these retrieved documents are prepended to the training sequence as context.

4. **Decoder-Only Transformer** (`$T$`): the main generator, a causal transformer with sizes ranging from 350M to 7B parameters. It receives a sequence of interleaved text tokens, image tokens, and special tokens (`<break>` for modality boundaries, `<mask>` for the CM3 infilling objective, `<eos>` for document separation) and is trained with a next-token prediction loss. The same architecture and weights are used for both text-to-image, image-to-text, and mixed-modal generation — the model learns to predict whatever token comes next, regardless of modality.

5. **Supervised Fine-Tuning (SFT) Data Pipeline**: after pretraining, the model is fine-tuned on a diverse mixture of tasks, each formatted as a sequence of interleaved text and image tokens with task-specific instruction prefixes (e.g., "Edit the image following the text instruction", "Make high quality image from canny edge features"). The same CM3 objective is used during SFT, meaning the model never changes its loss function — it simply sees new token sequences that encode new tasks.

6. **Inference Decoding Strategy** (`$\text{CD-K}$` + CFG + CLIP re-ranking): at generation time, the model runs two parallel token streams — one conditioned on the text input (conditional), one conditioned on a `<mask>` token (unconditional) — and combines their logits using either classifier-free guidance (CFG) or the proposed Contrastive Decoding TopK (CD-K) method. Multiple candidates are generated (8 per prompt), and a CLIP model selects the best one via re-ranking.

**Information flow during pretraining:** a Shutterstock caption-image pair enters the system → the dense retriever fetches 3 relevant multi-modal documents from the memory bank → these are concatenated with the query pair (separated by `<eos>` tokens) → the CM3 masking scheme is applied (random spans are masked and moved to the end, but masking never crosses `<break>` tokens) → the transformer predicts every token in the sequence autoregressively → the loss is computed only over the query pair's tokens (retrieved documents serve as context only, their tokens are not part of the loss).

**Information flow during SFT:** a task instruction + input (text, image, or both) is formatted into a sequence with a task prefix → the model generates the output tokens autoregressively (conditioned on the input prefix) → the loss is computed over the output tokens only, using the same CM3 objective weighting during pretraining.

**Information flow during inference (text-to-image):** a text prompt is tokenized → optionally, retrieved documents are prepended as context → the model generates two token streams (conditional and unconditional) → CD-K or CFG combines their logits to steer generation → 8 candidate images are generated → a CLIP model selects the best one based on text-image similarity.

### 3.3 Roadmap for the Deep Dive

The technical breakdown follows the chronological pipeline — pretraining first, then fine-tuning, then inference — because each stage builds on the outputs of the previous one:

- **First, the CM3 objective function** (Section 3.4.1): the core training objective inherited from Aghajanyan et al. (2022) and RA-CM3, with the two critical modifications CM3Leon introduces (removal of query up-weighting, prevention of cross-modality masking). This is the mathematical foundation that every other component rests on.

- **Second, the tokenization and data representation** (Section 3.4.2): how images and text are converted into discrete tokens, how modality boundaries are marked, and how retrieval-augmented training sequences are constructed. Understanding the token stream is prerequisite to understanding the objective and the retrieval mechanism.

- **Third, retrieval augmentation** (Section 3.4.3): the dense retriever architecture, the retrieval strategy (relevance + modality + diversity), and the three-document training construction. This is what makes the pretraining efficient and gives CM3Leon its strong FID with retrieval.

- **Fourth, the model architecture and training** (Section 3.4.4): the decoder-only transformer design, the specific deviations from OPT (removal of bias, dropout, and learnable layer norms; extended sequence length), weight initialization, and the training hyperparameters for the 350M, 760M, and 7B model sizes.

- **Fifth, the supervised fine-tuning stage** (Section 3.4.5): the task mixture, the data formatting (task prefixes, interleaved tokens), and the hyperparameters. This is the stage that unlocks the controllability and multi-task capabilities.

- **Sixth, the decoding strategies for inference** (Section 3.4.6): classifier-free guidance, the proposed CD-K method (and why the original contrastive decoding constraint was too strict), top‑p sampling, temperatured sampling, and the CLIP re-ranking pipeline. This is the inference-time recipe that turns pretrained weights into high-quality outputs.

### 3.4 Detailed, Sentence-Based Technical Breakdown

This is a **systems and training methodology paper** whose core idea is that autoregressive multi-modal models achieve state-of-the-art performance and efficiency when trained with a recipe faithfully adapted from text-only LLMs: retrieval-augmented pretraining with a cleaned-up infilling objective, followed by multi-task supervised instruction tuning, and decoded with a self-contained contrastive method that repurposes the conditional-vs-unconditional logit difference without requiring a separate weaker model.

---

#### 3.4.1 The CM3 Objective Function and CM3Leon's Modifications

The CM3 (Causal Masked Multi-modal Model) objective, originally introduced by Aghajanyan et al. (2022), is an infilling-based training objective that unifies autoregressive generation and masked-span prediction under a single next-token prediction loss. Rather than training the model to only generate left-to-right (like standard GPT models) or only fill in corrupted spans (like masked language models), CM3 transforms training examples by masking random spans of text and image tokens and relocating them to the end of the sequence, then trains the model to predict the entire sequence autoregressively — including reconstructing the masked spans from the relocated tokens at the end.

**The basic CM3 transformation.** Given a multi-modal input sequence `$x = [t_1, t_2, ..., t_n]$` where each `$t_i$` is either a text token or an image token, the CM3 objective:

1. Selects one or more random contiguous spans `$\{[s_j, e_j]\}_{j=1}^k$` within the sequence.
2. Replaces each span with a single `<mask>` token.
3. Appends a sentinel token (an `<infill>` marker) to the end of the sequence, followed by the actual tokens that were in the masked spans, in order.

For example, a caption-to-image pair like `"A photo of a chameleon: <break> IMG5432 IMG12 ... IMG1991"` might become `"A photo of <mask>: <break> IMG5432 IMG12 ... IMG1991 <infill> a chameleon"`. The model then sees the entire transformed sequence (including the `<infill>` marker and the relocated tokens) and is trained with the standard autoregressive loss:

$$\mathcal{L}_{\text{CM3}} = -\sum_{i=1}^{N} \log p(t_i \mid t_{<i})$$

where `$N$` is the total number of tokens in the transformed sequence (including the relocated spans), and `$p(t_i \mid t_{<i})$` is the model's predicted probability of token `$t_i$` given all previous tokens `$t_{<i}$`.

**What this objective computes:** for every position in the transformed sequence, the model predicts what token comes next, and its loss is the negative log-likelihood of the true token at that position. Crucially, when the model reaches the `<infill>` marker at the end of the masked-prefix portion, it must generate the tokens that were masked from the prefix — effectively doing autoregressive infilling.

**Why this form:** the CM3 objective is a unification trick. A standard autoregressive model can only generate left-to-right continuations. A masked model can fill in gaps but cannot generate. By physically relocating the masked spans to the end and using a single autoregressive loss over the entire sequence, CM3 trains one model to do both: generate new content (left-to-right continuation) and fill in missing content (by conditioning on the relocated tokens at the end). This is more parameter-efficient than training separate models or objectives, and it means the same model can be used at inference time for both generation tasks (e.g., caption-to-image) and infilling tasks (e.g., text-guided image editing, where parts of an image are masked and regenerated).

**CM3Leon's first modification: preventing masking across `<break>` tokens.** The original CM3 objective allowed the random span selection to cross the `<break>` token — the special token marking the boundary between modalities (e.g., between the caption text and the image tokens). CM3Leon introduces a hard constraint: **spans cannot cross `<break>` boundaries**. In practice, this means each masked span is either entirely within the text portion or entirely within the image portion of a training example, never straddling both.

The justification, stated in Section 2.2, is that "allowing masking across `<break>` tokens may lead to the model generating image content from an arbitrary midpoint, which is not a desirable outcome." The operational concern is this: if a mask could start in the text (e.g., partially through a caption) and extend into the image tokens, the model would learn to complete partial image token sequences that begin at semantically arbitrary positions — not at the natural starting point of an image token sequence. Since images are encoded as a fixed sequence of 1024 tokens from an encoder with a specific causal structure, generating image tokens from a midpoint would bypass the encoder's intended ordering and produce incoherent images. By constraining masks to single modalities, CM3Leon ensures that image generation always starts from the beginning of an image token sequence (or from a complete image condition), maintaining the integrity of the VQ-VAE's token structure.

**CM3Leon's second modification: removing query pair up-weighting.** RA-CM3 (Yasunaga et al., 2022), the direct predecessor, used retrieval-augmented training sequences that prepended several retrieved documents before the target query pair. It then up-weighted the loss contribution of the query pair relative to the retrieved documents, encouraging the model to focus on using the retrieved context during generation. CM3Leon removes this weighting entirely — all tokens in the sequence contribute equally to the loss.

The reason is stated in Section 2.2: the weighting "adversely affects the zero-shot scenario, where the goal is to generate an image without retrieval." The mechanism behind this is likely that up-weighting the query pair teaches the model to rely heavily on the retrieved context during training. When no retrieved context is available at inference time (the zero-shot setting), the model underperforms because it has learned to expect and depend on those extra conditioning signals. By using uniform weighting, CM3Leon learns to use retrieved context when it is present but does not become dependent on it, enabling strong zero-shot generation (as evidenced by the FID of 10.82 with 0 retrieved documents, still competitive with diffusion models).

**Connection to downstream capabilities.** This objective is what gives CM3Leon its unique flexibility at SFT time. Because the model has been trained to handle sequences with arbitrary `<mask>` placements (provided they respect modality boundaries), it can be fine-tuned on tasks that require different infilling patterns — text-guided editing (where an image is provided and text specifies what to change, requiring the model to regenerate parts of the image), structure-conditioned generation (where a control image like an edge map is provided and the model generates the corresponding photorealistic image), and spatial grounding (where object locations are specified and the model generates an image with objects at those positions). All of these boil down to the same operation: the model sees a partially-masked sequence and autoregressively generates the missing tokens. The CM3 pretraining ensures the model has extensive practice with this operation across diverse masking patterns.

---

#### 3.4.2 Tokenization and Multi-Modal Data Representation

CM3Leon represents everything — captions, image pixels, retrieval context, task instructions — as a single flat sequence of discrete tokens drawn from two vocabularies, separated by special control tokens.

**Image tokenization.** Images are processed through a pretrained VQ-VAE encoder-decoder pair (from Gafni et al., 2022a, the Make-A-Scene model). The encoder compresses a $256 \times 256$ RGB image into a $32 \times 32$ grid of latent codes, where each code is an index into a learned codebook of 8,192 discrete vectors. Flattened row-wise, this yields exactly **1024 image tokens** per image. The decoder (unused during CM3Leon training, only used to convert generated tokens back to pixels for evaluation and display) reconstructs the image from these 1024 tokens.

**Why this tokenizer:** a discrete token representation is essential because the autoregressive transformer operates over a categorical distribution — at each position, it predicts a probability distribution over the token vocabulary and samples from it. Continuous latent representations (like those used in latent diffusion models) cannot be directly generated by an autoregressive model without an additional decoding step. The 1024-token representation means an image is as "long" as a moderate-length text sequence, making the context length requirements tractable (a 4096-token sequence can hold roughly 4 images or 3 images plus text).

**Text tokenization.** Text is tokenized using a custom BPE (Byte Pair Encoding) tokenizer with a vocabulary size of **56,320 tokens**, trained on the text corpus from Zhang et al. (2022) (the OPT pretraining data). This is a standard subword tokenization scheme: common words are single tokens, rare words are split into multiple subword tokens, and the vocabulary size balances coverage (enough tokens to represent diverse text efficiently) against the softmax computational cost (which scales linearly with vocabulary size).

**Special tokens.** The tokenizer includes three special-purpose tokens that structure multi-modal sequences:

- **`<break>`**: marks the boundary between modalities within a single document. When the sequence transitions from text tokens to image tokens (or vice versa), a `<break>` token is inserted. This token is critical for three reasons: (1) it tells the model that the following tokens belong to a different modality and should be interpreted through the VQ-VAE decoder rather than as text, (2) it is the stopping condition for the CM3 masking scheme (masks cannot cross it), and (3) it provides an explicit signal that the model can learn to use during generation to know when to switch from generating text to generating images.
- **`<mask>`**: used in two contexts. During training, it replaces masked spans in the CM3 objective (as described in Section 3.4.1). During inference for classifier-free guidance, it replaces the text conditioning to create an unconditional generation stream — the model conditions on `<mask>` instead of the text prompt, producing an unconditional output.
- **`<eos>`**: marks the boundary between different multi-modal documents in a training sequence. In retrieval-augmented training, multiple retrieved documents (each consisting of a caption and image, separated by `<break>`) are concatenated with `<eos>` tokens, and the final query pair also ends with `<eos>`. This token tells the model that the following content belongs to a semantically distinct document, preventing information from adjacent retrieved documents from bleeding together.

**Training sequence construction (Figure 9 in the paper).** A full training example for CM3Leon consists of **three retrieved multi-modal documents** followed by the **query caption-image pair**. Each document is formatted as:

```
<text_tokens> <break> <image_tokens> <eos>
```

For the query pair specifically, the sequence is:

```
<text_tokens> <break> <image_tokens> <eos>
```

And the full training example is the concatenation of 3 retrieved documents + 1 query document:

```
<retrieved_1_text> <break> <retrieved_1_img> <eos> 
<retrieved_2_text> <break> <retrieved_2_img> <eos> 
<retrieved_3_text> <break> <retrieved_3_img> <eos> 
<query_text> <break> <query_img> <eos>
```

**Why three retrieved documents:** the paper states that "in training, we randomly select three retrieved samples for every caption-image pair in our dataset, effectively 4× the number of tokens available in the pretraining." The choice of three is a practical tradeoff: each retrieved document adds approximately 1024 image tokens plus text tokens, so three documents roughly quadruple the sequence length on average. This provides substantial context for the query pair (exposing the model to diverse related examples) without exceeding the 4096-token sequence length limit for the majority of examples. The 4× token multiplier is important because it means that even though CM3Leon's "raw" training dataset is 340M Shutterstock pairs, the effective training data seen by the model is approximately 1.36B tokens, amortizing the cost of the retriever across more learning signal.

**The CM3 transformation applied to this sequence.** After constructing the concatenated sequence, the CM3 masking scheme is applied (respecting `<break>` boundaries). However, the masking is applied **only to the query pair**, not the retrieved documents. The retrieved documents serve as static context — they tell the model what related content looks like — while the query pair is the target for generation/infilling. This means the model sees unmasked retrieved documents and a partially-masked query document, and must use the retrieved context to help reconstruct the masked portions of the query.

**Training vs. inference sequence construction.** During training, retrieved documents are prepended because the model needs to learn to use retrieved context. During inference (zero-shot generation), no retrieved documents are prepended, and the sequence is simply the text prompt followed by `<break>` and the model generates image tokens autoregressively. During retrieval-augmented inference (reported in Table 1 with 1 or 2 retrieved documents), one or two documents retrieved from the full Shutterstock training set are prepended before the generation prompt, providing the model with relevant visual and textual examples to condition on.

---

#### 3.4.3 Retrieval Augmentation Mechanism

Retrieval augmentation is the engine behind CM3Leon's training efficiency and strong performance. The mechanism has two components: a **dense retriever** (which finds relevant documents) and a **retrieval strategy** (which selects which retrieved documents to include in training sequences based on relevance, modality, and diversity criteria).

**The dense retriever architecture.** The retriever uses the bi-encoder design from Karpukhin et al. (2020), adapted to multi-modal documents. It is built on **frozen CLIP encoders** (the ViT-B-32 variant from Radford et al., 2021), which are kept fixed throughout training — the retriever is not fine-tuned.

For a given multi-modal document (which contains both text and an image), the retriever computes a single vector representation by:

1. Encoding the text portion using the frozen CLIP text encoder, producing a text embedding vector `$v_{\text{text}} \in \mathbb{R}^d$`.
2. Encoding the image portion using the frozen CLIP image encoder (ViT-B-32), producing an image embedding vector `$v_{\text{image}} \in \mathbb{R}^d$`.
3. Normalizing both vectors.
4. Averaging them: `$v_{\text{doc}} = \frac{1}{2}(v_{\text{text}} + v_{\text{image}})$`.

The query during training is typically the caption of the training pair (since the goal is to retrieve documents relevant to that caption). The query is encoded using the same frozen CLIP text encoder (if it is text), the CLIP image encoder (if it is an image), or both averaged (if multi-modal). The paper notes that they use **query dropout** during retrieval: 20% of the query tokens are randomly dropped before encoding, which serves as regularization and encourages the retriever to rely on diverse signals rather than overfitting to exact token matches.

**Why CLIP:** CLIP was trained on 400M image-text pairs to produce aligned embeddings — an image of a dog and the caption "a photo of a dog" produce similar vectors. This alignment means that encoding a multi-modal document by averaging its text and image CLIP embeddings naturally places similar documents (in terms of joint visual-semantic content) near each other in the embedding space. The frozen nature is a practical choice: fine-tuning the retriever would require periodically re-indexing the entire memory bank (all 340M Shutterstock pairs), which is computationally prohibitive at scale. Using frozen CLIP means the memory bank can be encoded once and reused throughout training.

**The memory bank and search.** All documents in the Shutterstock training set are encoded using the bi-encoder and stored in a memory bank `$M$`. For each training query, the retriever performs **Maximum Inner Product Search (MIPS)** over `$M$` using the Faiss library, returning a ranked list of the most similar documents (those with the highest cosine similarity / inner product to the query embedding).

**The retrieval strategy: three selection criteria.** Simply taking the top-K documents by relevance score would be problematic because high-similarity documents are often near-duplicates or highly redundant (e.g., different photos of the same scene with similar captions). CM3Leon's retrieval strategy applies three filters to produce informative and diverse training context:

1. **Relevance threshold**: the paper states "we only use retrieved documents with relevance score $\leq 0.9$" (where the relevance score is the inner product / cosine similarity). This is counterintuitive — one would typically want *higher* relevance — but the justification is that scores above 0.9 indicate near-duplicate or identical content to the query, which provides no new information to the model. By capping the relevance, the retriever is forced to return documents that are related but not identical, providing genuinely useful conditioning signals.

2. **Modality**: the paper emphasizes that "retrieving a multi-modal document consisting of images and text leads to better generator performance than retrieving either image or text." The retriever always returns full multi-modal documents (both the image and its caption), never text-only or image-only documents. This is because the training objective requires both modalities in context — the model needs to see how captions relate to images in the retrieved examples to apply that understanding to the query pair.

3. **Diversity through deduplication**: the paper states that "simply taking the top K documents based on relevance score can result in duplicates or highly similar documents, hurting downstream pretraining." To prevent this, the retriever skips candidate documents that are too similar to documents already selected for the training example. The similarity check is performed by comparing CLIP embeddings; if a candidate's embedding is too close to any already-selected document, it is skipped and the next candidate in the MIPS ranking is considered. Combined with the relevance cap, this ensures the three retrieved documents provide complementary information about the query concept rather than redundant views.

**Training-time retrieval vs. inference-time retrieval.** During training, the three retrieved documents are randomly selected from the filtered MIPS results, meaning different training epochs may use different retrieved context for the same query pair — this acts as additional data augmentation. During inference (when retrieval is used), the paper reports results with 0, 1, or 2 retrieved documents, retrieved from the full Shutterstock training set using the same frozen retriever. The dramatic FID improvement from 0 to 2 documents (10.82 to 4.88, Table 1) demonstrates that retrieval provides crucial world knowledge that the model's parameters alone cannot fully capture — even at 7B parameters.

**Computational cost of retrieval.** The paper does not report the computational overhead of the retriever itself, but it is worth noting: encoding 340M documents with CLIP (or rather, using pre-encoded CLIP embeddings if the Shutterstock dataset was preprocessed) and running MIPS for every training example is non-trivial. However, since the retriever is frozen, the memory bank embeddings are computed once (a one-time cost) and the MIPS per query is a sub-linear operation using approximate nearest neighbor indices. The paper's compute efficiency claims (5× less than PARTI) include this cost, as it is part of the training pipeline.

---

#### 3.4.4 Model Architecture and Pretraining

CM3Leon's transformer architecture is a **decoder-only, causally-masked** design, meaning each token can only attend to tokens that appear before it in the sequence (no bidirectional attention). This is the same basic architecture as GPT models and OPT (Brown et al., 2020; Zhang et al., 2022), but with specific modifications for the multi-modal setting.

**Architecture specification.** The paper trains three model sizes, detailed in Table 3:

| Model | Layers (`# L`) | Embedding dim (`d_model`) | Sequence length | Batch size (tokens) | Peak LR | Warmup steps | GPUs used | Tokens consumed |
|-------|-----------|--------------------|-----------------|---------------------|---------|--------------|-----------|-----------------|
| 350M  | 24        | 1024               | 4096            | 8 million            | 6e-04   | 1500         | 256       | 1.4T            |
| 760M  | 24        | 1536               | 4096            | 8 million            | 5e-04   | 1500         | 256       | 1.9T            |
| 7B    | 32        | 4096               | 4096            | 8 million            | 1.2e-04 | 1500         | 512       | 2.4T            |

**Deviations from OPT / standard GPT architectures.** Compared to the OPT model family (Zhang et al., 2022), which CM3Leon otherwise follows closely, the paper makes four architectural changes:

1. **Removal of bias terms**: standard transformers include bias terms in the linear projections for query, key, value, and output. CM3Leon removes these, following recent evidence that bias terms have negligible effect on transformer performance while slightly reducing parameter count and memory usage.

2. **Removal of dropout**: CM3Leon uses no dropout anywhere in the model. Dropout was historically used in transformers for regularization, but the scale of modern training data (2.4T tokens for the 7B model) makes explicit regularization unnecessary — the data volume itself prevents overfitting.

3. **Removal of learnable parameters for layer normalization**: in standard transformers, each layer norm has a learnable scale and bias parameter (`$\gamma$` and `$\beta$`). CM3Leon uses fixed (non-learnable) layer normalization, reducing parameter count and simplifying optimization without hurting performance at scale. This is a design choice also used in some recent efficient transformers.

4. **Extended sequence length from 2048 to 4096**: OPT used a sequence length of 2048 tokens. CM3Leon doubles this to 4096. This is critical for the multi-modal setting: a single image is 1024 tokens, and with 3 retrieved documents (each ~1050 tokens on average) plus the query pair (~1050 tokens), the total sequence can easily exceed 3,000 tokens. A 2048-token limit would force truncation of retrieved context or query data; 4096 provides enough headroom.

The architecture otherwise uses standard multi-head self-attention with causal masking, feed-forward networks (presumably SwiGLU or standard ReLU, though the paper does not specify the activation function), and absolute learned positional embeddings.

**Weight initialization.** The paper specifies three initialization details:

- **General weight initialization**: truncated normal distribution with mean 0 and standard deviation 0.006, truncated to 3 standard deviations. This means weights are sampled from `$\mathcal{N}(0, 0.006^2)$` and any value more than 3 standard deviations from the mean (i.e., outside `$[-0.018, 0.018]$`) is resampled. The standard deviation 0.006 is relatively small, ensuring that initial activations don't saturate non-linearities early in training.

- **Output layer initialization**: all zeros. The output projection layer (which maps the final hidden state to vocabulary logits) is initialized to zero, meaning the initial prediction is a uniform distribution over the vocabulary. This is a common initialization trick: starting with uniform predictions prevents the model from being overconfident about wrong tokens early in training, making the loss landscape smoother.

- **Positional embedding initialization**: near-zero, specifically "near zero with a standard deviation of 0.0002." This is much smaller than the 0.006 used for other weights. The rationale is likely that positional embeddings should start small and gradually learn meaningful position-dependent offsets, rather than imposing a strong initial positional prior that might conflict with the specific sequence structures (retrieved documents, modality boundaries) in CM3Leon's data.

**Training objective and data.** The training uses the CM3 objective described in Section 3.4.1, applied to the retrieval-augmented sequences described in Section 3.4.3, with the modifications described in Section 3.4.1 (no cross-`<break>` masking, no query up-weighting). All tokens in the sequence contribute equally to the loss — there is no special weighting for image tokens vs. text tokens, for retrieved context vs. query tokens, or for masked vs. unmasked positions.

The training data is the Shutterstock dataset of 340M licensed caption-image pairs. The paper emphasizes that they "use only licensed images from Shutterstock" to "avoid concerns related to images ownership and attribution, without sacrificing performance." This is both an ethical choice and a practical demonstration — it shows that state-of-the-art text-to-image generation does not require web-scraped data with unclear copyright status.

**Training dynamics (Figure 3).** Figure 3 plots validation perplexity against training updates for all three model sizes. The key observation is that "the losses for all three models decrease steadily throughout training, strongly suggesting they have not saturated." This is an important empirical claim: it means CM3Leon has not hit the point of diminishing returns at 2.4T tokens, and further training (scaling tokens even more) would likely yield additional improvements. The purple dashed line marks the point where the 760M and 7B models resume training after a full epoch, and "the small rise in the PPL is due to the sudden increase of the learning rate" — meaning the learning rate schedule resets at epoch boundaries (likely a cosine schedule with restarts), causing a temporary perplexity spike that recovers quickly.

**Training implementation.** The models are trained using Metaseq (the same framework used for OPT at Meta), with experiment tracking via Aim (Arakelyan et al., 2020). The 350M and 760M models use 256 A100 GPUs; the 7B model uses 512 A100 GPUs. The 8M-token batch size (at 4096 sequence length) means each GPU processes approximately 64 sequences per step (for 256 GPUs: 8M / 4096 / 256 ≈ 7.6 sequences per GPU).

**Scaling behavior (Figure 2).** Figure 2 plots FID against equivalent A100 GPU hours on a log scale, comparing CM3Leon (350M, 760M, 7B) with diffusion models (Stable Diffusion v1.5, v2.1), prior autoregressive models (DALL-E, PARTI), and encoder-decoder models (PARTI-20B). The CM3Leon curve lies below all others, and its slope is steeper, indicating that each additional unit of compute yields a larger FID improvement for CM3Leon than for competitors. The 7B model achieves the best FID (4.88) at far fewer GPU hours than PARTI's 20B model (which achieves FID 7.23). This is the quantitative basis for the paper's "5× less training compute" claim.

---

#### 3.4.5 Supervised Fine-Tuning (SFT) Stage

The SFT stage is what transforms CM3Leon from a general-purpose multi-modal generative model (capable of text-to-image and image-to-text generation but not specifically trained to follow nuanced instructions) into a controllable, multi-task model that can perform text-guided editing, structure-conditioned generation, question answering, and long-form captioning — all with the same weights.

**Why SFT matters (and why it was missing from prior multi-modal autoregressive models).** Text-only LLMs had by this point firmly established that SFT (or "instruction tuning") dramatically improves zero-shot generalization, task-following ability, and the stability of outputs (Iyer et al., 2022; Wei et al., 2021; Ouyang et al., 2022). However, prior multi-modal autoregressive models (DALL-E, PARTI, Make-A-Scene, RA-CM3) were trained in a single stage for a narrow task distribution, typically text-to-image generation with simple captions. CM3Leon demonstrates that the same SFT recipe — fine-tune the pretrained model on a diverse mixture of tasks, each formatted as an instruction-and-output pair — transfers effectively to the multi-modal setting, unlocking capabilities that the pretrained model has the raw knowledge for but cannot reliably deploy.

**The SFT task mixture.** CM3Leon is fine-tuned on a diverse set of tasks that span three modalities of interaction:

*Image-to-image tasks (the model must generate an image given an image and optional text):*
- **Text-guided image editing** (InstructPix2Pix; Brooks et al., 2023): 127,000 examples. The input is an image and a text instruction (e.g., "make her an alien"), and the output is the edited image. The paper used the InstructPix2Pix methodology with proprietary face-filtering to remove personally identifiable information.
- **Structure-conditioned generation** (ControlNet; Zhang & Agrawala, 2023): 7 million examples across multiple control modalities — canny edge maps (1M), segmentation maps (1M), HED boundary maps (1M), depth maps (1M), 3D normal maps (1M), human pose keypoints via OpenPose (142K), and synthesized scribbles (500K). For each, the input is a control image (e.g., a Canny edge map) plus a text caption, and the output is the corresponding photorealistic image.
- **Spatially grounded generation**: 3 million examples from object detection datasets (MS-COCO, OpenImages, Objects365), where objects are specified with bounding box coordinates in the text prompt and the model must generate an image with objects at those locations.
- **Text rendering / "how-to-write"**: 200,000 examples from Shutterstock images containing text, where the model must generate an image of a sign or logo with specified text content.

*Text-to-image tasks:*
- **Standard caption-to-image generation**, using the same format as pretraining but with task-specific prefixes (e.g., "Make high quality image from children's scribbles and text description").

*Image-to-text tasks (the model must generate text given an image):*
- **MS-COCO Captioning** (Chen et al., 2015): 591,000 examples. The input is "Describe the given picture." followed by the `<break>` token and the image tokens; the output is the caption text.
- **Flickr30k** (Young et al., 2014): 144,000 examples, similar format.
- **Image Paragraph Captioning** (Krause et al., 2017): 14,000 examples, with prompts like "Describe the given picture in very detail" or "Generate a long caption for the given image."
- **Localized Narratives** (Pont-Tuset et al., 2020): 164,000 examples of long-form image descriptions.
- **Visual Question Answering**: combines VQA2 (Goyal et al., 2017; 1.3M examples), VizWiz (Gurari et al., 2018; 92K examples), OKVQA (Marino et al., 2019; 26K examples), and ScienceQA (Lu et al., 2022; 6K examples). Each uses templates like "Question: {question} Answer: {answer}" or "Question: {question} Answer: Let's think step-by-step: {explanation} So the answer is {answer}."

**SFT data formatting.** Every task is formatted as a sequence of interleaved text and image tokens, with the task instruction serving as a prefix that conditions the model on what type of output is expected. For example, the text-guided editing task is formatted as:

```
Edit the image following the text instruction <break> {input_image_tokens} <break> 
{edit_instruction} <break> {output_image_tokens}
```

For structure-conditioned generation:

```
Make high quality image from canny edge features <break> {edge_image_tokens} <break> 
{caption_text} <break> {output_image_tokens}
```

For VQA:

```
Question: {question} Answer: {answer}. <break> {image_tokens}
```

**Why task prefixes:** the task prefix serves the same role as instruction tuning in text-only LLMs — it tells the model which capability to deploy. Without task prefixes, a model that can do everything would have ambiguous conditioning: given an image and some text, should it answer a question about the image, describe the image, or edit the image? The prefix resolves this ambiguity by specifying the intended output format before the model sees the input. During inference, users specify which task to perform by prepending the appropriate prefix to their input.

**Why the same CM3 objective for SFT:** the paper uses the identical CM3 loss during SFT as during pretraining (no special objective for different tasks). This is a design choice that ensures consistency: the model never has to adapt to a new loss landscape. Instead, SFT is purely a data distribution shift — the model sees new types of sequences (with task prefixes) but the underlying operation (predict next token) remains unchanged. This is exactly how text-only LLM instruction tuning works (e.g., fine-tuning GPT-3 on instruction-output pairs), and CM3Leon demonstrates the same principle works for multi-modal tokens.

**SFT hyperparameters (Table 4).** The fine-tuning uses the following configuration:

| Model | GPUs | Sequence Length | Batch Size | Learning Rate | Warmup Steps | Tokens Processed |
|-------|------|-----------------|------------|---------------|--------------|------------------|
| 760M  | 64 × 80GB A100 | 4096 | 2M tokens | 5e-05 | 150 | ~30B |
| 7B    | 128 × 80GB A100| 4096 | 2M tokens | 5e-05 | 150 | ~30B |

Both models process approximately **30 billion tokens** during SFT. This is roughly 1/50th of the pretraining token budget for the 7B model (2.4T pretraining tokens vs. 30B SFT tokens), consistent with the observation in text-only LLMs that SFT requires far less data than pretraining to be effective.

The paper notes that "preliminary experiments were conducted to identify optimal learning rates from a range of 1e−5, 3e−5, 5e−5, 1e−4 and per-GPU batch sizes from 4, 8, 16 using our validation split." The selected values (5e-05 learning rate, 2M-token batch size) were the best among the sweep.

**Data balancing.** To handle the highly imbalanced task distribution (e.g., 7M structure-conditioned examples vs. 6K ScienceQA examples), the paper implements an **up/down sampling strategy with a threshold of 3/0.3** (Section E.1). This means: if a task has fewer than `$0.3 \times$` the average number of examples per task, its examples are up-sampled (repeated) to reach that threshold. If a task has more than `$3 \times$` the average, its examples are down-sampled (sub-sampled) to that threshold. This prevents the model from overfitting to the largest datasets (like structure-conditioned generation) while still learning effectively from small datasets (like ScienceQA).

**What SFT enables that pretraining doesn't.** The pretrained CM3Leon can already generate images from text and captions from images, but it has no notion of "editing," "answering questions," or "following spatial constraints" as distinct controllable tasks. SFT introduces these capabilities by training the model to condition its outputs on task-specific prefixes and, critically, on multi-modal conditioning signals (e.g., an input image to be edited plus a text instruction). The pretrained model has the visual understanding and generation capability; SFT teaches it to deploy that capability in response to specific instructions and structured inputs. The qualitative examples in Figure 6 and Figure 15 show that SFT yields coherent text-guided edits (changing facial hair, adding sunglasses) and structure-conditioned generation (different scenes from the same pose skeleton) — capabilities that the raw pretrained model cannot perform.

---

#### 3.4.6 Inference Decoding Strategies

The quality of autoregressive text-to-image generation depends heavily on the decoding strategy used at inference time. CM3Leon employs a multi-stage decoding pipeline: temperatured or top-p sampling with guidance (CFG or CD-K), multiple candidate generation, and CLIP-based re-ranking.

**Temperatured sampling.** Temperatured sampling modifies the softmax distribution over the token vocabulary by dividing logits by a temperature parameter `$T$`:

$$p(t_i \mid t_{<i}) = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$

where `$z_i$` is the raw logit (pre-softmax score) for token `$i$` at position `$t_i$`, and `$T > 0$` is the temperature.

**What it computes:** the probability of each next token, scaled by temperature. When `$T < 1$`, the distribution becomes more peaked (the highest-logit token gets more probability mass), making sampling more deterministic. When `$T > 1$`, the distribution flattens, making sampling more random (exploratory).

**Why this form:** temperature scaling preserves the relative ranking of token probabilities (unlike top-k truncation, which zeroes out low-probability tokens) while controlling the randomness-quality tradeoff. For image generation, where each token controls a small patch of the image, modest temperature values (typically `$T < 1$` in practice) prevent the model from being too random (which would produce incoherent images) or too deterministic (which would produce boring, repetitive images). The paper does not report specific temperature values used, but standard practice in autoregressive image generation ranges from 0.8 to 1.0.

**Top-P (nucleus) sampling.** Top-P sampling (Holtzman et al., 2020) truncates the token distribution to the smallest set of most-likely tokens whose cumulative probability exceeds a threshold `$p$`. Specifically, tokens are sorted by probability from highest to lowest, and the model samples only from the first `$k$` tokens where `$\sum_{j=1}^k p(t_j) \geq p$`. All tokens outside this set have their probabilities zeroed out (they are never sampled).

**What it computes:** a dynamic truncation of the token distribution that adapts to the entropy of the distribution at each position. If the model is very confident (one token has probability 0.95), Top-P may only include that one token. If the model is uncertain (probabilities are spread across many tokens), Top-P includes many tokens.

**Why this form:** Top-P avoids the brittleness of fixed top‑k truncation. With top‑k, a fixed number of tokens (e.g., k=50) are kept regardless of context — but in some positions, the model's probability mass is concentrated in 5 tokens, while in others it is spread across 200. Top-P adapts to this: it keeps enough tokens to cover `$p$` probability mass, ensuring the model has enough options when uncertain but not too many when confident. The paper does not report the specific `$p$` threshold used.

**Classifier-Free Guidance (CFG).** CFG is a technique that steers generation toward stronger alignment with the conditioning signal (the text prompt) by contrasting the conditional and unconditional token distributions. It was originally developed for diffusion models (Ho & Salimans, 2022) and adapted to autoregressive image models by Gafni et al. (2022a) (Make-A-Scene). CM3Leon adopts this adaptation.

At each generation step, the model runs two forward passes:

1. **Conditional pass**: the model conditions on the actual input text `$t_x$` (e.g., "A photo of a chameleon"), producing logits `$z_{\text{cond}}$` for each token in the vocabulary.
2. **Unconditional pass**: the model conditions on a `<mask>` token instead of the text (the CM3 pretraining objective provides this capability, since the model was trained with masks replacing text spans), producing logits `$z_{\text{uncond}}$`.

The guidance operation combines these logits:

$$z_{\text{cf}} = z_{\text{uncond}} + \alpha_c \cdot (z_{\text{cond}} - z_{\text{uncond}})$$

where `$\alpha_c$` is the guidance scale (CFG weight), typically set to values greater than 1.

**What this equation computes:** it extrapolates along the vector from unconditional to conditional logits. When `$\alpha_c = 0$`, the result is purely unconditional generation (ignoring the text prompt). When `$\alpha_c = 1$`, the result is standard conditional generation. When `$\alpha_c > 1$`, the model amplifies the difference between conditional and unconditional distributions — effectively saying "generate what this text prompt would produce, and push it further away from what you would produce without any prompt." This sharpens the model's focus on the text conditioning.

**Why this form:** the subtraction `$z_{\text{cond}} - z_{\text{uncond}}$` isolates the effect of the text conditioning on the token distribution. Tokens that appear in both conditional and unconditional distributions have a small difference; tokens specific to the text prompt have a large difference. Multiplying by `$\alpha_c > 1$` and adding back to `$z_{\text{uncond}}$` emphasizes prompt-relevant tokens while suppressing prompt-irrelevant ones. For image generation, this produces images that are more faithful to the text description at the cost of some diversity (the `$\alpha_c$` parameter trades off fidelity vs. diversity).

**Contrastive Decoding TopK (CD-K): CM3Leon's novel contribution.** The paper observes a structural similarity between CFG's logit subtraction and contrastive decoding (CD), a technique from text-only language models (Li et al., 2022). Standard CD defines a score per candidate next token:

$$\text{CD}(y_i; y_{<i}) = \log \frac{p_{\text{EXP}}(y_i \mid y_{<i})}{p_{\text{AMA}}(y_i \mid y_{<i})}$$

where `$p_{\text{EXP}}$` is the probability under an "expert" model (stronger, larger, or conditioned on more information) and `$p_{\text{AMA}}$` is the probability under an "amateur" model (weaker, smaller, or unconditioned). Tokens that the expert model considers much more likely than the amateur get positive scores; tokens the amateur overfits get negative scores.

In the original CD formulation, a constraint set `$V(y_{<i})$` restricts which tokens are eligible for sampling:

$$V(y_{<i}) = \{y_i \in \mathcal{V} \mid p_{\text{EXP}}(y_i \mid y_{<i}) \geq \alpha \cdot \max_w p_{\text{EXP}}(w \mid y_{<i})\}$$

This means: only tokens whose probability under the expert is at least `$\alpha$` times the probability of the most likely token are considered. Tokens below this threshold get `$-\infty$` scores (never sampled). The paper found this constraint **too strict for image generation** — it would "consistently become greedy decoding" (only the single most likely token survives the filtering), losing the diversity needed for coherent images.

**CM3Leon's CD-K modification** replaces the `$\max_w$` with the `$k$`-th largest probability:

$$V(y_{<i}) = \{y_i \in \mathcal{V} \mid p_{\text{EXP}}(y_i \mid y_{<i}) \geq \alpha \cdot \text{kmax}_{k,w} \left(p_{\text{EXP}}(w \mid y_{<i})\right)\}$$

where `$\text{kmax}_{k,w}$` returns the `$k$`-th largest probability among all tokens (not the maximum).

**What this modification computes:** instead of comparing every token's probability to the single most likely token, CD-K compares to the `$k$`-th most likely token. Since the `$k$`-th largest probability is lower than the maximum (for `$k > 1$`), the threshold is relaxed — more tokens are eligible for sampling. This preserves the diversity needed for image generation while still filtering out very low-probability tokens.

**Why this modification helps:** in text generation, CD's original formulation works well because language has low entropy — at each position, the model is often very confident in a few tokens. In image generation, the 8,192-token VQ-VAE vocabulary means the distribution is inherently more spread out (many tokens are plausible for a given image patch), so comparing to `$\max$` would eliminate too many candidates. Using the `$k$`-th largest probability (for a suitable `$k$`) expands the candidate set enough to generate diverse and coherent images while still leveraging the contrastive signal to emphasize text-relevant tokens. The paper does not report the specific value of `$k$` used.

**How CD-K relates to CFG.** Both CD-K and CFG use the same core idea — contrasting conditional and unconditional token distributions — but they implement it differently:

- CFG applies the contrast at the **logit level** (before softmax), scaling by `$\alpha_c$`, then samples from the full vocabulary.
- CD-K applies the contrast as a **log-probability ratio** (after softmax), then applies an additional truncation step (the `$\alpha$` and `$k$` thresholds) to further filter tokens.

Both can be seen as special cases of a general "guidance by distributional contrast" framework. The paper shows that CD-K is **competitive with CFG** (Figure 4, right panel) and, importantly, **complementary to CFG** — generating some candidates with CFG and some with CD-K, then re-ranking with CLIP, produces better FID than using either method alone. This complementarity suggests that the two methods produce different types of images (possibly with different diversity-fidelity tradeoffs), and combining them via re-ranking captures the best of both.

**CLIP re-ranking.** For quantitative evaluation (FID), CM3Leon generates **8 candidate images** per text prompt, and a CLIP model (ViT-B-32, the same architecture as the retriever but presumably a separate instance) scores each image for text-image similarity. The image with the highest CLIP score is selected as the final output. This follows the DALL-E approach (which used 512 candidates) and the PARTI approach (which reduced to 16 with CFG), and serves to filter out the occasional low-quality generation that even guided sampling produces.

**Ablation of decoding strategies (Figure 4).** Figure 4 provides three key experimental findings about decoding:

- **Left panel (CFG weight sweep)**: The optimal CFG weight is consistent across model sizes (350M, 760M, 7B), all showing minimum FID at a CFG weight of approximately 3–4 (reading from the log-scale FID axis, the trough is around CFG weight 3). Higher CFG weights (above 7) degrade FID, likely because over-amplifying the conditional signal produces images that are overly "sharp" or stereotyped, losing visual diversity.

- **Right panel (number of candidates per prompt)**: Both TopP sampling and CD-K show decreasing FID as the number of generated candidates increases (from 1 to 32), which is expected — more candidates means a higher chance that at least one is high quality. Importantly, "TopP and CD-K are similar across sample counts but exhibit complementary behavior when combined" — mixing candidates from both decoding methods yields lower FID than either method alone at a given total candidate count.

- **CFG weight × batch size interaction (not in the main figure but discussed)**: The paper notes that values around 3 are used for most experiments, with the exception of text-guided editing where separate image CFG (1.5) and text CFG (7.5) are used to balance fidelity to the original image with faithfulness to the editing instruction.

**Unified inference procedure.** Putting it all together, the inference pipeline for a text-to-image generation is:

1. **Text prompt** is tokenized into text tokens.
2. **Optional retrieval**: if retrieval augmentation is enabled, 1–2 multi-modal documents are retrieved from the Shutterstock training set using the frozen CLIP retriever and prepended to the prompt sequence (each as `{text} <break> {image} <eos>`).
3. **Parallel generation**: 8 candidate images are generated. Some candidates use CFG (with `$\alpha_c \approx 3$`), some use CD-K (with a specific `$k$` and `$\alpha$`). For each candidate, the model autoregressively generates 1024 image tokens following the `<break>` token, with each token sampled from the guided conditional distribution.
4. **CLIP re-ranking**: the 8 generated images are decoded from VQ-VAE tokens to pixel space, and a CLIP model scores each image against the text prompt. The highest-scoring image is returned.
5. **Optional continuation**: if the task is interleaved text-and-image generation (e.g., a multi-turn editing dialogue), the selected image tokens are appended to the sequence, and the process repeats for the next generation step.

This pipeline is what produces the state-of-the-art FID of 4.88 (with 2 retrieved documents) and 10.82 (without retrieval) reported in Table 1.

---

#### 3.4.7 Unified Design: Why This Recipe Works

CM3Leon's approach can be understood as a systematic application of three principles that had proven successful in text-only LLMs but had not been fully applied to multi-modal autoregressive models:

**Principle 1: Training data as sequence construction.** In text-only LLMs, the power of pretraining comes from treating all text as a single sequence prediction problem — next token prediction over diverse web text, books, code, etc. CM3Leon extends this to multi-modal data by constructing unified sequences of text tokens and image tokens separated by control tokens (`<break>`, `<eos>`). The model never processes modalities separately; it always sees them interleaved, learning the joint distribution `$p(\text{text}, \text{image})$` rather than `$p(\text{image} \mid \text{text})$` or `$p(\text{text} \mid \text{image})$` in isolation. This joint training is what enables the model to go both directions (image-to-text and text-to-image) with a single set of weights.

**Principle 2: Retrieval augmentation as compute multiplier.** Text-only LLMs (like RETRO; Borgeaud et al., 2022) and retrieval-augmented LMs (like REALM; Guu et al., 2020) had shown that retrieval reduces the parameter count needed for factual knowledge. CM3Leon shows the same holds for visual knowledge: the retriever provides visual and textual examples of concepts (objects, scenes, styles, compositions), reducing the burden on the model's parameters to memorize the visual appearance of every concept. This is why retrieval during inference improves FID so dramatically (from 10.82 to 4.88) — the model with retrieval can lean on the retrieved visual examples to guide its generation, rather than having to recall every visual detail from its weights.

**Principle 3: Instruction tuning as capability unlocker.** Text-only LLMs demonstrated that pretrained models contain latent capabilities that are not accessible through naive prompting but become accessible after fine-tuning on diverse instruction-output pairs. CM3Leon applies this exact logic to the multi-modal domain: the pretrained model has the raw capacity to edit images (because the CM3 objective trains it to regenerate masked image regions) and to answer questions about images (because it has learned visual concepts through the joint embedding), but it needs SFT to learn to deploy these capacities in response to specific instructions. The task prefixes in SFT serve as learned "API calls" — compact tokens that route the model's latent capabilities toward specific behaviors.

## 4. Key Insights and Innovations

### Innovation 1: The Full Text-LLM Training Recipe Transfers to Multi-Modal Models — and That Transfer Is the Core Contribution

The dominant framing in the field circa 2023 treated diffusion models and autoregressive models as fundamentally different beasts requiring fundamentally different training approaches. Diffusion models had their own ecosystem of techniques — classifier-free guidance, latent spaces, noise schedules — while autoregressive image models (DALL-E, PARTI) borrowed some ideas from text LLMs but never adopted the full two-stage paradigm that had proven so successful for models like ChatGPT: large-scale pretraining followed by multi-task supervised instruction tuning.

CM3Leon's most conceptually significant move is to argue that this divergence was unnecessary. The paper treats the multi-modal autoregressive setting as **just another sequence modeling problem**, and systematically ports over the three pillars of modern text-LLM training:

1. **Retrieval-augmented pretraining** (analogous to RETRO for text; Borgeaud et al., 2022), which amortizes world knowledge across retrieved context rather than forcing the model parameters to memorize everything.
2. **A cleaned-up infilling objective** that respects modality boundaries (no cross-`<break>` masking), analogous to how text infilling models learn to respect sentence or paragraph boundaries.
3. **Multi-task supervised instruction tuning** on a diverse mixture of tasks formatted as instruction-input-output sequences, exactly as OPT-IML (Iyer et al., 2022) did for text-only models.

None of these individual components is architecturally novel. The retriever is frozen CLIP (Radford et al., 2021). The CM3 objective is from Aghajanyan et al. (2022). The SFT paradigm is from Iyer et al. (2022). What is novel is the **integration of all three into a single training pipeline for a multi-modal model** and the empirical demonstration that this integration yields both efficiency gains (5× less training compute than PARTI for a better FID of 4.88; Table 1, Figure 2) and capability gains (a single model doing text-to-image, image-to-text, text-guided editing, structure-conditioned generation, VQA, and long-form captioning after SFT; Figures 5–7).

**Why this is a fundamental shift, not incremental.** Prior to this work, the field implicitly assumed that multi-modal generation required specialized training recipes. Diffusion models had their own training paradigm. Autoregressive image models had their own (typically single-stage, narrow-task pretraining). The SFT stage — arguably the single most important technique behind ChatGPT's flexibility and instruction-following ability — had no counterpart in multi-modal autoregressive models. CM3Leon demonstrates that the SFT stage is not text-specific; it is a general property of sequence models trained on diverse token streams. This reframes multi-modal model development from "build a specialized architecture for each task" to "train a generalist sequence model with the LLM recipe and teach it tasks through instruction tuning." The paper's ablation in Table 1 (zero-shot FID of 10.82 without retrieval vs. 4.88 with retrieval) and the SFT results in Table 2 (competitive zero-shot performance with Flamingo on VQA/VizWiz despite seeing ~33× less text data) provide empirical anchors for this reframing.

**The significance beyond performance.** The 4.88 FID is impressive but the deeper contribution is the **unification argument** — that the same model weights, the same training objective, and the same inference procedure can handle text-to-image generation, image editing, structure-conditioned generation, spatial grounding, VQA, and long-form captioning. This is not possible with diffusion models without auxiliary components (e.g., separate text decoders for captioning, separate encoders for conditioning). The paper's Figure 5 illustrates this unification visually: a single model takes in diverse input formats (task instruction + image + text + bounding boxes) and produces the appropriate output format (image or text), all through the same next-token prediction operation. This is fundamentally a different model of what a multi-modal system can be — not a collection of specialists but a single generalist — and the paper's contribution is showing that the LLM training recipe is what makes this possible.

### Innovation 2: Retrieval Augmentation as a Compute Multiplier — and the Clean Separation of Zero-Shot vs. Retrieval-Augmented Performance

Retrieval-augmented generation for multi-modal models was not new — RA-CM3 (Yasunaga et al., 2022) had introduced it for the CM3 architecture, and RE-IMAGEN (Chen et al., 2022) had applied it to diffusion models. What CM3Leon contributes is a **diagnostic insight** about the interaction between retrieval augmentation and zero-shot capability, and a concrete design fix (removing query-pair up-weighting) that dramatically improves the tradeoff.

The diagnostic: RA-CM3 up-weighted the loss on the query pair relative to the retrieved context during training. The stated motivation was to encourage the model to focus on the retrieved examples during generation. The unintended consequence, which CM3Leon identifies and fixes, is that this up-weighting creates a **dependency on retrieved context** — the model learns to rely heavily on retrieval because the training signal heavily penalizes getting the query pair wrong relative to getting the retrieved context right. When retrieval is unavailable at inference time (the zero-shot setting), performance collapses because the model expects conditioning signals it no longer receives.

CM3Leon's fix is elegantly simple: remove the up-weighting entirely. All tokens in the training sequence — retrieved documents and query pair alike — contribute equally to the loss. The result, shown in Table 1, is that CM3Leon achieves **both** strong retrieval-augmented performance (FID 4.88 with 2 retrieved documents) and strong zero-shot performance (FID 10.82 with 0 retrieved documents). The zero-shot FID of 10.82 is competitive with or better than several diffusion models (Stable Diffusion v2.1 at 12.60, DALL-E) that do not use retrieval at all, meaning the model hasn't sacrificed its inherent generation capability for retrieval-dependence.

**Why this is a conceptual advance, not just a hyperparameter tweak.** The field's default assumption — implicit in RA-CM3's design — was that retrieval augmentation and zero-shot capability were in tension: making the model better at using retrieved context would make it worse without it. CM3Leon demonstrates that this tension is **a training artifact, not a fundamental tradeoff**. By using uniform weighting, the model learns to use retrieved context when it's present (achieving state-of-the-art FID) without becoming dependent on it (achieving competitive zero-shot FID). This is a clean reframing of the retrieval design problem: the goal is not to maximize retrieval utilization but to make retrieval a **helpful but optional** conditioning signal that the model can flexibly use or ignore.

The evidence for this separation is in the large FID gap between 0 and 2 retrieved documents (10.82 → 4.88). The model's inherent image generation capability (zero-shot FID) is good; retrieval provides a massive additional boost by supplying relevant visual examples. But the boost is additive, not essential — the model doesn't fall apart without retrieval. This is the ideal behavior for a retrieval-augmented system, and CM3Leon shows how to achieve it through a simple training objective modification.

**Practical significance.** This finding has direct implications for deployment: the same model can be used in both resource-constrained settings (no retrieval, faster inference, FID 10.82) and quality-sensitive settings (retrieval enabled, slower inference, FID 4.88), with no architectural changes or fine-tuning needed — just a flag at inference time. This is significantly more flexible than a model that was trained with query up-weighting and collapses without retrieval.

### Innovation 3: Contrastive Decoding as a Self-Contained Alternative to Classifier-Free Guidance — and the Complementarity Insight

Classifier-free guidance (CFG) was, by this point, the standard method for improving text-to-image generation quality in both diffusion and autoregressive models (Gafni et al., 2022a; Ho & Salimans, 2022). It works by contrasting conditional and unconditional token distributions, using a guidance scale α_c to amplify the text-conditioning signal. It requires no additional models or training — just two forward passes per generation step.

CM3Leon's innovation in this space is twofold:

**First, the recognition that contrastive decoding (CD) from text-only LLMs (Li et al., 2022) is structurally equivalent to CFG** — both involve subtracting log-probabilities between a stronger (conditional) and weaker (unconditional) distribution — and can therefore be adapted as an alternative guidance method for multi-modal generation. This is a conceptual bridge between two previously separate research threads: guidance methods in image generation (CFG) and contrastive decoding in text generation (CD). The paper shows they are instances of the same underlying principle: steering generation by amplifying the difference between two distributions.

**Second, the CD-K modification that makes contrastive decoding work for image generation.** The original CD method uses a strict constraint — only tokens whose probability exceeds α times the maximum probability are eligible for sampling — which works for low-entropy text but collapses to greedy decoding for the high-entropy image token distribution (8,192 VQ-VAE tokens). CM3Leon's fix, replacing the maximum with the k-th largest probability, is simple but motivated by a clear diagnostic: image token distributions are inherently more spread out than text token distributions, so the eligibility threshold needs to be based on a softer reference point.

**The complementarity finding is the key conceptual result here.** Figure 4 (right panel) shows that TopP sampling and CD-K produce similar FID-vs-candidates curves when used individually, but **combining candidates from both methods yields lower FID at any given total candidate count**. This means CD-K and CFG produce **different, complementary sets of high-quality images** — they don't just both work; they work better together because they sample from different regions of the image distribution. This is a non-obvious finding that suggests guidance methods can be treated as **diversity mechanisms**, not just quality-control mechanisms. The practical recipe — generate some candidates with CFG, some with CD-K, then re-rank with CLIP — is a concrete instantiation of this insight.

The significance of this contribution is that it opens up a new dimension for improving autoregressive image generation: rather than developing ever-more-sophisticated single guidance methods, combine diverse guidance methods to explore a broader range of high-quality images and let a re-ranker select the best. This is analogous to ensemble methods in classification — multiple weak classifiers can outperform a single strong one — but applied to generative decoding.

### Innovation 4: Modality-Boundary-Aware Masking as a Principled Design Choice for Unified Multi-Modal Models

This is a subtle but deep contribution that is easy to overlook because it appears as a minor implementation detail ("prevent masking across `<break>` tokens"; Section 2.2). The conceptual significance is larger than the brevity of description suggests.

The original CM3 objective (Aghajanyan et al., 2022) allowed random spans to cross modality boundaries — a mask could start in a text caption and extend into the image tokens. CM3Leon introduces a hard constraint: masks must be contained within a single modality. The justification — "allowing masking across `<break>` tokens may lead to the model generating image content from an arbitrary midpoint, which is not a desirable outcome" — points to a fundamental issue with multi-modal sequence modeling that had not been explicitly articulated before: **modality boundaries are semantically meaningful and should be treated as hard constraints, not as just another token**.

The deeper reasoning: image token sequences produced by a VQ-VAE encoder have an internal causal structure. The 1024 tokens representing a 256×256 image are not an arbitrary set; they are a raster-ordered grid where adjacent tokens correspond to adjacent spatial patches, and the encoder's learned codebook imposes statistical dependencies between tokens that respect this spatial structure. Starting generation from an arbitrary midpoint of this sequence — which is what cross-modality masking would train the model to do — would require generating image tokens without the spatial context that normally precedes them, likely producing incoherent images.

By contrast, text tokens have a different kind of structure. A mask that starts mid-sentence and extends into the image would train the model to complete partial sentences and then immediately generate images from mid-sequence. This conflates two very different generation problems (sentence completion and image generation) that have different optimal token distributions, making the joint optimization harder.

The `<break>` token constraint is thus not just a practical fix — it is a **principled recognition that multi-modal sequence models need to respect modality boundaries as structural elements, not just as token-type markers**. The `<break>` token should be a hard barrier for the masking operation because it marks a genuine shift in the data-generating process: text is generated by linguistic rules, images are generated by visual and physical rules, and the model should not be trained on artificial "completions" that mix partial generations across these regimes.

**Why this matters for future work.** This design choice has implications for any model that builds on CM3Leon or similar multi-modal sequence architectures. It suggests that special tokens should not just be passive markers that the model might learn to interpret; they should actively constrain training operations in ways that reflect the underlying structure of the data. Future multi-modal sequence models with more modalities (audio, video, code) will face analogous questions: should masking cross audio-video boundaries? Should infilling ever require generating half an audio clip followed by half a video frame? CM3Leon's answer — no, and here's why — provides a template for reasoning about these cases.

## 5. Experimental Analysis

### Evaluation Methodology

- **Dataset.** The primary quantitative evaluation for text-to-image generation uses the **zero-shot MS-COCO (30K) task**, specifically the 30,000-image subset of the Microsoft Common Objects in Context dataset (Lin et al., 2014) with captions from Chen et al. (2015). For text-to-image tasks, captions are used as prompts without any fine-tuning on MS-COCO training data (hence "zero-shot"). For supervised fine-tuning evaluation, the paper uses the standard test splits of: MS-COCO Captioning (test), VQA2 (test-dev), VizWiz (test-dev), OKVQA (validation), Image Paragraph (test), and VisDial (validation), as specified in Table 2.

- **Base model(s).** Three model sizes of the CM3Leon architecture are evaluated: **350M, 760M, and 7B parameters**. All are decoder-only transformers trained on the Shutterstock dataset (340M licensed caption-image pairs) with the retrieval-augmented CM3 objective. The 7B model serves as the flagship, while the 350M and 760M provide scaling law data points. The pretrained models are evaluated before and after retrieval augmentation; the SFT models are evaluated after fine-tuning on the multi-task mixture described in Section 4.

- **Metrics.** The primary metric for text-to-image generation is **Fréchet Inception Distance (FID)**, computed using the clean-fid implementation from Seitzer (2020) which addresses biases in the original FID calculation. Lower FID indicates higher image quality and diversity (closer distributional match to real MS-COCO images). For image-to-text tasks (Table 2), the paper reports **CIDEr** (caption quality), **accuracy** (VQA2, VizWiz, OKVQA), and **NDCG** (VisDial). These are standard metrics in their respective tasks and enable direct comparison with prior work.

- **Baselines.** The paper compares against a broad set of prior models, organized by architecture family:
  - **Diffusion models**: StableDiffusion (Rombach et al., 2022; FID 12.60), Imagen (Saharia et al., 2022), LDM.
  - **Retrieval-augmented diffusion models**: KNN-Diffusion (FID 12.50), RE-IMAGEN (Chen et al., 2022; FID 5.25).
  - **Autoregressive token models**: DALL-E (Ramesh et al., 2021), PARTI (Yu et al., 2022; FID 7.23 at 20B parameters), Make-A-Scene (Gafni et al., 2022a). PARTI is the primary compute-matched comparison (5× more training compute).
  - **Retrieval-augmented autoregressive models**: RA-CM3 (Yasunaga et al., 2022; FID 15.70 at 2.7B parameters).
  - **Non-autoregressive token models**: MUSE (Chang et al., 2023; FID 7.88).
  - **Vision-language models (for SFT tasks)**: Flamingo-9B (Alayrac et al., 2022) and OpenFlamingo-9B, evaluated zero-shot on captioning and VQA tasks.
  
  For the SFT evaluation (Table 2), Flamingo and OpenFlamingo serve as the primary baselines, with the notable distinction that they were trained on substantially more text data (100B and 40B tokens respectively) than CM3Leon (~3B Shutterstock text tokens).

- **Generation budget / compute accounting.** Compute is measured in two ways:
  - **Training compute**: Equivalent A100 GPU hours, plotted in Figure 2. This enables direct comparison of training efficiency across models with different architectures and hardware requirements. CM3Leon's 5× less training compute claim is relative to PARTI-20B, computed from the total A100-hours consumed during pretraining.
  - **Inference compute**: Measured as the number of generated candidates per prompt before CLIP re-ranking (Figure 4, right). CM3Leon generates 8 candidates per prompt for the main FID evaluation (Table 1), compared to DALL-E's 512 and PARTI's 16. The paper also reports inference latency (seconds per image) in Figure 10 and throughput (seconds per batch size) in Figure 11 for the 7B model across different precision formats (FP32, BF16, INT8) and model parallelism configurations.

- **Cross-validation / statistical protocol.** The paper does not describe a formal cross-validation or statistical significance testing protocol. The primary FID evaluation is on the full 30K MS-COCO zero-shot set, which is large enough to yield stable FID estimates. For SFT, standard test splits from each benchmark are used. Hyperparameter selection (learning rate, batch size for SFT) used a held-out validation split to choose from the ranges 1e−5 to 1e−4 (LR) and 4 to 16 (per-GPU batch size), as described in Appendix E.1. The paper does not report confidence intervals, standard deviations across random seeds, or significance tests for any of the reported metrics, which is a limitation for interpreting the reliability of small differences (e.g., FID 4.88 vs. 5.25 for RE-IMAGEN).

### Main Quantitative Results

#### Text-to-Image Generation: Scaling and Retrieval

The headline result is that **CM3Leon-7B achieves a zero-shot MS-COCO FID of 4.88** when using 2 retrieved documents during inference, which is state-of-the-art at the time of publication and substantially better than the next-best retrieval-augmented model RE-IMAGEN (FID 5.25) and the best non-retrieval autoregressive model PARTI-20B (FID 7.23). Critically, this is achieved with approximately **5× less training compute than PARTI**, as shown in Figure 2 where the CM3Leon-7B point sits at roughly 1/5 the A100 hours of the PARTI-20B point while achieving a lower FID.

Table 1 provides the detailed breakdown:

| Configuration | FID-30K |
|--------------|---------|
| CM3Leon-7B, 2 retrieved docs | 4.88 |
| CM3Leon-7B, 1 retrieved doc | 5.78 |
| CM3Leon-7B, 0 retrieved docs (zero-shot) | 10.82 |
| CM3Leon-760M, 2 retrieved docs | 6.61 |
| CM3Leon-350M, 2 retrieved docs | 14.20 |

The retrieval dependence is stark: adding 2 retrieved documents during inference improves FID from 10.82 to 4.88 — a more than 2× improvement. This demonstrates that retrieval provides visual-world knowledge that the model's 7B parameters, even after training on 340M Shutterstock images, cannot fully internalize. The scaling trend is also clear: at each retrieval level, larger models achieve better FID (350M: 14.20, 760M: 6.61, 7B: 4.88 with 2 retrieved docs).

**Comparison to prior models (Table 1):**
- CM3Leon-7B (FID 4.88) vs. RA-CM3-2.7B (FID 15.70): A 3.2× FID improvement while scaling from 2.7B to 7B parameters, but also reflecting the training recipe improvements (no cross-`<break>` masking, no query up-weighting, larger dataset).
- CM3Leon-7B (FID 4.88) vs. RE-IMAGEN (FID 5.25): A retrieval-augmented diffusion model that uses a larger training set (450M vs. 340M examples) and a 3.6B effective model (T5-XXL + diffusion), yet CM3Leon outperforms it by 0.37 FID.
- CM3Leon-7B zero-shot (FID 10.82) vs. StableDiffusion (FID 12.60): Even without retrieval, CM3Leon outperforms the widely-used SD v2.1 by 1.78 FID, demonstrating that the autoregressive approach is inherently competitive with diffusion models for image quality, even without its key advantage (retrieval).

**Scaling behavior (Figure 2).** The log-scale FID-vs-compute plot reveals that CM3Leon's scaling curve is steeper than both diffusion models (Stable Diffusion v1.5, v2.1) and prior autoregressive models (DALL-E, PARTI). The 350M model already achieves competitive FID (14.20 with retrieval) at relatively low compute, and each doubling of model size yields a substantial FID reduction. The 7B model's point lies below and to the left of all competitors, meaning it achieves both better FID and lower training cost. The paper interprets this as evidence that "autoregressive models scale better with compute when trained with the right recipe."

**Training dynamics (Figure 3).** Validation perplexity decreases steadily for all three model sizes across their full training runs (350M: 1.4T tokens, 760M: 1.9T tokens, 7B: 2.4T tokens), with no sign of saturation — the curves are still declining at the final checkpoint. This strongly suggests that further scaling (more tokens, larger models) would yield additional improvements. The purple dashed line marks the epoch boundary where the 760M and 7B models resume training after a full pass through the 340M dataset, and the small perplexity spike is attributed to the learning rate schedule resetting (cosine with restarts).

#### Decoding Strategy Experiments

The paper conducts ablation experiments on decoding strategies (Figure 4) using an 8K held-out subset of MS-COCO:

- **CFG weight sweep (Figure 4, left):** The optimal CFG weight is approximately 3–4 for all model sizes (350M, 760M, 7B), with FID degrading at higher weights (above ~7). This consistency across model scales suggests the guidance scale is a property of the tokenizer and training objective, not the model capacity. The minimum FID at CFG weight ~3 corresponds to the model achieving the best balance between prompt fidelity and image diversity — higher weights over-constrain generation toward stereotyped conditional outputs.

- **Candidate count and decoding strategy comparison (Figure 4, right):** Both TopP sampling and CD-K show monotonically decreasing FID as the number of generated candidates increases from 1 to 32 (more candidates → higher probability of at least one high-quality image). The two methods produce similar FID-vs-candidates curves when used individually. However, the key finding is that **combining candidates from both methods yields lower FID than either method alone** at any given total candidate count. For example, at 32 total candidates, generating 16 with TopP + 16 with CD-K outperforms generating 32 with only TopP or only CD-K. This is the empirical basis for the "complementarity" claim: CD-K and CFG-based TopP explore different regions of the image distribution, and CLIP re-ranking can select the best from the combined pool.

- **CFG weight for SFT tasks:** For text-guided image editing (Section 4.1, Figure 6), the paper reports using **separate image CFG (1.5) and text CFG (7.5)** weights rather than a single CFG value. This is because editing requires balancing two constraints: fidelity to the original input image (lower image CFG keeps the output close to the input) and faithfulness to the editing instruction (higher text CFG ensures the edit is applied). For structure-conditioned generation, a single CFG value of 3 is used.

#### Supervised Fine-Tuning Results

**Image generation tasks (qualitative evaluation).** The SFT model's capabilities are demonstrated primarily through qualitative examples (Figures 6, 7, 15, 16) rather than quantitative metrics. Figure 6 shows text-guided editing (row 1: changing facial hair, adding sunglasses, aging, applying face paint) and structure-conditioned generation (row 2: the same OpenPose skeleton generating different scenes — businessman in city, boy on grass, girl on mountain trail, woman on beach). Figure 15 extends this to additional conditioning modalities: Canny edge maps, depth maps, line maps, scribbles, and spatial bounding boxes, all generating coherent photorealistic images that respect both the structural constraints and the text prompt.

The paper does not report quantitative metrics (e.g., FID, CLIP score, human evaluation) for the SFT generation tasks. This is a notable gap: without quantitative evaluation, it is difficult to assess whether the SFT model's editing and grounding capabilities are competitive with specialized models (e.g., InstructPix2Pix, ControlNet, GLIGEN) or whether the unified model trades off task-specific performance for generality.

**Image-to-text tasks (Table 2).** On vision-language benchmarks, the SFT-CM3Leon-7B model achieves competitive zero-shot performance despite seeing substantially less text data than the baselines:

| Task | SFT-CM3Leon-7B (0-shot) | Flamingo-9B (0-shot) | OpenFlamingo-9B (0-shot) |
|------|--------------------------|----------------------|--------------------------|
| MS-COCO CIDEr | 61.6 | 79.4 | 65.5 |
| VQA2 Accuracy | 47.6 | 51.8 | 43.5 |
| VizWiz Accuracy | **37.6** | 28.8 | - |
| OKVQA Accuracy | 23.8 | 44.7 | - |
| Image Paragraph CIDEr | 10.5 | - | - |
| VisDial NDCG | 22.6 | 48.4 | - |

The key comparison points:
- **CM3Leon outperforms Flamingo on VizWiz (37.6 vs. 28.8, a +8.8 absolute improvement)**, which is notable because VizWiz consists of questions asked by blind users about images they've taken, often with unusual framing, lighting, or image quality — scenarios where a joint image-text model trained end-to-end (CM3Leon processes image tokens natively) may have advantages over a frozen-vision-encoder + frozen-language-model architecture like Flamingo.
- **CM3Leon is competitive with OpenFlamingo-9B on VQA2 (47.6 vs. 43.5) and within 4 points on MS-COCO CIDEr (61.6 vs. 65.5)**, despite seeing ~13× less text data (3B vs. 40B tokens during training).
- **CM3Leon substantially underperforms Flamingo on OKVQA (23.8 vs. 44.7) and VisDial (22.6 vs. 48.4)**. OKVQA requires external knowledge retrieval (the questions are designed to require Wikipedia-level knowledge beyond what's in the image), suggesting CM3Leon's 3B text tokens are insufficient for acquiring broad world knowledge compared to Flamingo's 100B tokens. VisDial requires multi-turn dialogue reasoning over an image, which may benefit from Flamingo's architecture that uses perceiver resamplers to process image features at each dialogue turn.

**Qualitative text generation (Figures 7, 16).** Figure 16 shows standard captioning and VQA capabilities: the model correctly describes images ("A man is standing on a beach with a surfboard," "A herd of sheep standing on top of a snow covered field") and answers straightforward visual questions ("What are people flying? Kites," "What sign is on the street? stop"). Figure 7 demonstrates more complex capabilities: long-form detailed captioning ("A street sign is on a metal pole. The sign is blue with white writing. There is a red light on the pole...") and step-by-step reasoning for ScienceQA ("Think about each object. Potato chips have a salty taste. The pretzel is salty. So the answer is (B)."). The model's ability to generate chain-of-thought reasoning over an image is notable given that it was trained on only 6K ScienceQA examples (Table 5), suggesting the CM3 objective's infilling training provides a strong prior for structured reasoning.

#### Inference Efficiency

Figure 10 reports inference latency for CM3Leon-7B compared to other models:
- CM3Leon-7B (BF16): 11.8 seconds per 256×256 image
- CM3Leon-7B (INT8): 9.1 seconds per image
- PARTI-3B: 6.4 seconds per image (at 256×256)
- MUSE-3B: 0.5 seconds per image (at 256×256, leveraging non-autoregressive iterative decoding)
- LDM (50 steps): 3.7 seconds per 512×512 image
- Imagen: 9.1s (256×256) to 13.1s (1024×1024)

CM3Leon is slower than both non-autoregressive (MUSE) and latent diffusion (LDM) models, reflecting the fundamental sequential bottleneck of autoregressive decoding: generating 1024 image tokens requires 1024 sequential forward passes through the 7B-parameter transformer. The INT8 quantization provides a 23% speedup (11.8s → 9.1s) by reducing memory bandwidth and compute requirements per forward pass.

Figure 11 shows inference throughput scaling across batch sizes (1-256) and model parallelism configurations (MP1, MP2, MP4, MP8). At batch size 1, throughput is approximately 0.08 images/second (≈12.5 seconds/image, consistent with Figure 10). Throughput scales sub-linearly with batch size due to the autoregressive bottleneck: even with large batches, each of the 1024 generation steps must complete before the next step begins. Higher model parallelism (MP4, MP8) actually reduces throughput at small batch sizes (the communication overhead of splitting the model across GPUs dominates) but enables larger total batch sizes and higher absolute throughput.

### Ablation Studies and Robustness Checks

**Retrieval augmentation during training (Table 1, Figure 2):** The paper demonstrates that retrieval is critical for training efficiency and final performance. The CM3Leon-7B model trained with retrieval augmentation achieves FID 4.88 (2 retrieved docs) vs. 10.82 (0 retrieved docs). However, this is not a controlled ablation with a retrieval-free training run — the same model is evaluated with and without retrieval at inference time. The paper does not report the FID of a model trained entirely without retrieval augmentation, which would isolate the effect of retrieval on training dynamics (as opposed to inference-time conditioning). The RA-CM3 comparison (trained with retrieval, achieving FID 15.70) provides a weak baseline but does not fully control for the other recipe changes (no query up-weighting, no cross-`<break>` masking, larger dataset).

**Query up-weighting removal (Section 2.2, implicit in Table 1):** The paper claims that RA-CM3's query up-weighting "adversely affects the zero-shot scenario" and removes it. The evidence is indirect: RA-CM3 (with up-weighting) achieves FID 15.70, while CM3Leon-7B (without up-weighting) achieves FID 10.82 zero-shot — but these models differ in dataset size (150M vs. 340M), model size (2.7B vs. 7B), and other recipe changes. There is no controlled experiment where CM3Leon is trained with and without query up-weighting to isolate its effect on zero-shot FID.

**Cross-modality masking prevention (Section 2.2, no explicit table/figure):** The paper states that preventing masking across `<break>` tokens prevents the model from "generating image content from an arbitrary midpoint, which is not a desirable outcome." However, there is **no ablation comparing CM3Leon trained with vs. without this constraint**. The justification is presented as a logical argument rather than an empirical finding. Given that RA-CM3 (which allowed cross-modality masking) achieved FID 15.70 and CM3Leon achieves 4.88, and many other factors changed simultaneously, the specific contribution of this design choice to the final performance cannot be disentangled from the reported results.

**CFG weight across model sizes (Figure 4, left):** The optimal CFG weight is approximately 3–4 for all model sizes (350M, 760M, 7B). This consistency is a robustness check: it suggests that the guidance scale is not an artifact of model capacity or overfitting, but rather a fundamental property of the training objective and tokenizer. The FID penalty at high CFG weights (above ~7) is also consistent across sizes, indicating that over-constraining the conditional distribution degrades image quality in a model-size-independent manner.

**Decoding strategy complementarity (Figure 4, right):** The finding that TopP and CD-K produce complementary candidates is demonstrated by the "½ TopP + ½ CD-K" curve lying below both individual curves. The paper does not explore why the methods are complementary — one hypothesis (not tested) is that CFG-based TopP amplifies the text-conditioning signal more aggressively (producing higher-fidelity, lower-diversity images), while CD-K with the relaxed k-th threshold produces more diverse images that explore different visual interpretations of the prompt, and CLIP re-ranking can select the best from either pool.

**CFG weights for SFT editing tasks (Section 4.1):** The paper reports using separate image CFG (1.5) and text CFG (7.5) for text-guided editing. There is no ablation comparing single vs. separate CFG weights for editing, nor a sweep over the image CFG/text CFG parameter space. The reported values represent a tuned configuration, but without the tuning curve, it is unclear how sensitive editing quality is to these choices or whether a single CFG weight with careful tuning could achieve similar results.

**SFT data balancing (Section E.1):** The paper implements up/down sampling with a threshold of 3/0.3 to handle the imbalanced task distribution. No ablation is reported for different threshold values or for training without balancing. Given the extreme imbalance (7M structure-conditioned examples vs. 6K ScienceQA), the choice of balancing strategy could significantly affect per-task performance, but the paper provides no sensitivity analysis.

### Critical Assessment

**Claim: "CM3Leon achieves state-of-the-art performance in text-to-image generation with 5× less training compute than comparable methods (zero-shot MS-COCO FID of 4.88)."**

This claim is **well-supported quantitatively** by Table 1 and Figure 2. The FID of 4.88 is indeed better than all comparison models listed (next best: RE-IMAGEN at 5.25, PARTI at 7.23). The compute comparison is based on Figure 2, where CM3Leon-7B sits at approximately 1/5 the A100 hours of PARTI-20B. However, there are important caveats:

- **The comparison is to PARTI specifically**, not to all models in the table. PARTI used 5B examples vs. CM3Leon's 340M, and the "5× less compute" reflects the total pretraining FLOPs difference, not inference cost. Other models (Stable Diffusion, MUSE) use even less training compute than CM3Leon but achieve worse FID, so the claim is specifically about the *Pareto frontier* — CM3Leon achieves the best FID while using less compute than the previous best model, not less compute than all models.
- **The compute accounting includes retrieval overhead?** Unclear. The paper does not specify whether Figure 2's "equivalent A100 hours" includes the cost of building and querying the memory bank during training, or only the transformer training cost. If retrieval indexing and MIPS queries are excluded, the compute advantage is somewhat overstated — retrieval is not free, especially for 340M-document memory banks.
- **The 4.88 FID requires retrieval at inference time**, which adds latency and compute (encoding the query, running MIPS over 340M documents, prepending retrieved documents to the context). The zero-shot FID of 10.82 is competitive but not state-of-the-art (e.g., PARTI achieves 7.23 without retrieval). The paper's strongest result is therefore a retrieval-augmented result, not a purely generative one, which matters for fair comparison to non-retrieval models.

**Claim: "CM3Leon is the first multi-modal model trained with a recipe adapted from text-only language models, including a large-scale retrieval-augmented pretraining stage and a second multi-task supervised fine-tuning (SFT) stage."**

This is a **prior-work claim that is supported by citation but not experimentally verified**. The paper cites RA-CM3 as using retrieval-augmented pretraining but not SFT, and Iyer et al. (2022) (OPT-IML) as the text-only SFT paradigm. To the extent that no prior work combined *both* retrieval-augmented pretraining *and* multi-task SFT for a token-based multi-modal autoregressive model, this claim holds. However:

- **Flamingo (Alayrac et al., 2022) used a multi-task training approach** with diverse image-text tasks, albeit with a different architecture (frozen vision encoder + frozen LM with perceiver resamplers) and without retrieval augmentation. CM3Leon's novelty is specifically the *token-based decoder-only* instantiation of the LLM recipe, not the concept of multi-task multi-modal training itself.
- **The paper does not ablate the SFT stage** to demonstrate that it is the SFT specifically, rather than the diverse pretraining data or the CM3 objective, that enables the multi-task capabilities. There is no comparison of pretrained CM3Leon vs. SFT-CM3Leon on the vision-language tasks in Table 2 — the pretrained model could potentially achieve reasonable zero-shot performance on captioning and VQA without SFT, given that the CM3 objective trains it to go in both directions (image→text and text→image). Without this ablation, the causal contribution of SFT to the quantitative results in Table 2 is not established.

**Claim: "CM3Leon demonstrates unprecedented levels of controllability in tasks ranging from language-guided image editing to image-controlled generation and segmentation."**

This claim is supported only by **qualitative examples** (Figures 6, 7, 15, 16). There are no quantitative metrics for:
- Editing quality (e.g., how well the edited image preserves the original content while applying the edit, measured by LPIPS or CLIP directional similarity).
- Structure-conditioned generation accuracy (e.g., how well the generated image matches the input edge map/depth map/pose, measured by keypoint accuracy or edge F1).
- Spatial grounding accuracy (e.g., whether objects are actually placed at the specified bounding box coordinates).

Without quantitative evaluation, "unprecedented levels of controllability" is an impressionistic claim based on cherry-picked examples. The paper would need to compare against specialized models (InstructPix2Pix for editing, ControlNet for structure-conditioned generation, GLIGEN or LayoutGPT for spatial grounding) on standard benchmarks with automated metrics to substantiate this claim. The current evidence only demonstrates that the model *can* perform these tasks, not that it performs them *well* or *better than alternatives*.

**What experiments would strengthen the paper:**

1. **Controlled ablation of the SFT stage**: Evaluate the pretrained CM3Leon-7B (no SFT) on the vision-language tasks in Table 2 using zero-shot prompting (e.g., "Describe the given picture." without having seen that format during training). This would isolate the effect of SFT on task performance.

2. **Ablation of query up-weighting**: Train two versions of CM3Leon (or a smaller variant) with and without query up-weighting, holding all else constant, to quantify the effect on zero-shot FID. The current comparison to RA-CM3 conflates too many variables.

3. **Quantitative SFT task evaluation**: Report standard metrics for editing (LPIPS vs. InstructPix2Pix), structure-conditioned generation (keypoint PCK, edge F1 vs. ControlNet), and spatial grounding (bounding box accuracy vs. GLIGEN) on held-out test sets. The qualitative examples are promising but do not constitute scientific evidence of "unprecedented controllability."

4. **Text-to-image evaluation beyond FID**: FID measures distributional similarity to MS-COCO but does not capture prompt faithfulness. Report CLIP score (image-text similarity) or human evaluation of text-image alignment to assess whether the low FID comes at the cost of prompt following.

5. **Training data ablation**: Train a CM3Leon variant on a subset of the Shutterstock data (e.g., 100M, 200M, 340M) to characterize how FID scales with data quantity, and whether the retrieval augmentation advantage persists at smaller data scales.

6. **Inference compute vs. quality tradeoff**: Report FID for the 7B model at different generation budgets (1, 2, 4, 8, 16 candidates) to quantify the marginal benefit of additional candidates, and whether the complementarity of CFG and CD-K persists at lower candidate counts.

**Summary of the experimental evidence:**

The paper provides strong quantitative evidence for its core text-to-image generation claims: CM3Leon achieves state-of-the-art FID (4.88) with ~5× less training compute than the previous best autoregressive model (PARTI), and retrieval augmentation is the key enabler of this efficiency (FID drops from 4.88 to 10.82 without retrieval). The scaling analysis (Figures 2, 3) suggests continued benefits from further scaling, and the decoding strategy experiments (Figure 4) provide practical guidance on CFG weights and the benefits of combining decoders.

The SFT results are substantially weaker. The vision-language task numbers (Table 2) are competitive but not state-of-the-art (except VizWiz), and the image generation capabilities unlocked by SFT (editing, grounding, structure-conditioning) are demonstrated only qualitatively. The paper's strongest claim — "unprecedented levels of controllability" — is not experimentally substantiated. The quantitative evaluation gap for SFT is the most significant weakness in the experimental analysis: without it, the paper demonstrates that the LLM training recipe can be applied to multi-modal models (a meaningful methodological contribution) but does not convincingly demonstrate that the resulting model is practically competitive with specialized approaches for the tasks it claims to unify.

## 6. Limitations and Trade-offs

### 6.1 The Retrieval Pipeline Makes the Headline FID Number Misleading for Zero-Shot Generation Use Cases

The paper's central quantitative claim — an FID of 4.88 on zero-shot MS-COCO — is achieved with **retrieval augmentation during inference**: two multi-modal documents are fetched from the Shutterstock training set and prepended to the generation context. The zero-shot FID without retrieval is 10.82 (Table 1), more than 2× worse. Section 2.2 acknowledges that the original RA-CM3 objective's query up-weighting "adversely affects the zero-shot scenario" and that CM3Leon removes this weighting to improve zero-shot performance, but the paper never reframes its headline result around this distinction.

**The consequence:** A practitioner evaluating CM3Leon for a deployment where retrieval is infeasible — because latency requirements prohibit MIPS over a 340M-document memory bank, because the memory bank cannot be shipped with the model, or because the query images are out-of-distribution relative to Shutterstock — should expect FID closer to 10.82 than 4.88. The retrieval pipeline adds substantial inference cost: encoding the query with CLIP, running approximate nearest-neighbor search over 340M documents, and prepending 2+ retrieved documents (each roughly 1050 tokens) to the generation context, increasing the sequence length and thus the per-step autoregressive cost. The paper does not report the latency overhead of retrieval. Figure 10 reports 11.8 seconds per image (BF16) and 9.1 seconds (INT8) but does not specify whether this includes retrieval time or is generation-only — if it is generation-only, the true per-image latency with retrieval is higher by an unknown amount.

**What evidence exists in the paper:** The retrieval dependence is documented directly in Table 1: FID progressions for the 7B model are 10.82 (0 retrieved), 5.78 (1 retrieved), and 4.88 (2 retrieved). The paper does not ablate retrieval overhead or provide a cost-quality curve that would let practitioners decide whether the FID improvement from 5.78 to 4.88 is worth the additional retrieval cost. Figure 11 reports throughput scaling across batch sizes but does not isolate the retrieval component.

**Mitigation status:** The paper does not address this as a limitation. The retrieval-augmented and zero-shot numbers are both reported transparently in Table 1, but the abstract and introduction emphasize 4.88 without qualification, and the paper never explicitly states that this is a retrieval-augmented rather than purely generative result. A practitioner would need to read Table 1 carefully to discover the 10.82 baseline.

---

### 6.2 The Supervised Fine-Tuning Stage Is Evaluated Almost Entirely Qualitatively, Undermining the "Unprecedented Controllability" Claim

Section 4.1 claims that after SFT, CM3Leon "can also demonstrate unprecedented levels of controllability in tasks ranging from language-guided image editing to image-controlled generation and segmentation." The evidence for this claim consists of qualitative examples (Figures 6, 7, 15, 16) and a single quantitative table for vision-language tasks (Table 2). There are **no quantitative metrics for the image generation tasks unlocked by SFT**: text-guided image editing, structure-conditioned generation (Canny edge, depth, pose, scribble, segmentation map), spatially grounded generation, or text rendering.

**The consequence:** The paper provides no evidence that CM3Leon's editing quality, structural fidelity, or spatial accuracy is competitive with specialized models. A practitioner choosing between CM3Leon and dedicated tools — InstructPix2Pix for editing, ControlNet for structure-conditioned generation, GLIGEN for spatial grounding — has no basis for comparison. Without metrics like LPIPS (perceptual distance between input and edited image), keypoint accuracy (for pose-conditioned generation), edge F1 (for edge-conditioned generation), or bounding box IoU (for spatial grounding), the claim of "unprecedented controllability" is unsubstantiated. The qualitative examples may be cherry-picked; the paper provides no information about failure modes, failure rates, or the sensitivity of generation quality to the CFG weights (image CFG 1.5, text CFG 7.5 for editing; 3.0 for structure-conditioning).

**What evidence exists in the paper:** The only quantitative SFT results are in Table 2 (image-to-text tasks: captioning, VQA). For image-to-image and image-to-text-image tasks that constitute the "controllability" claim, the evidence is exclusively Figures 6, 7, 15, and 16 — approximately 20–30 hand-selected examples across all task types. The paper does not report FID, CLIP score, or any automated metric on held-out test sets for any SFT image generation task.

**Mitigation status:** The paper does not acknowledge this as a limitation. Section 4.1 describes the tasks, the data sources, and the template formats, but never transitions from qualitative demonstration to quantitative evaluation. The supplementary figures (Figures 15, 16 in the Appendix) provide additional qualitative examples but no metrics.

---

### 6.3 Inference Latency Is Fundamentally Bottlenecked by Autoregressive Decoding, Making CM3Leon Impractical for Interactive or Real-Time Applications

CM3Leon generates images autoregressively: producing a 256×256 image requires 1024 sequential forward passes through the transformer, one per image token. Figure 10 reports 11.8 seconds per image (BF16) and 9.1 seconds (INT8) on a single generation, and this does not include retrieval time or the CLIP re-ranking step (which requires generating 8 candidates and running CLIP inference on each). Compare this to MUSE (0.5 seconds via non-autoregressive iterative decoding with 24 steps) and Stable Diffusion (3.7 seconds for a larger 512×512 image with 50 diffusion steps). The autoregressive bottleneck means CM3Leon is approximately 18× slower than MUSE and 3× slower than SD at generating images, while producing lower-resolution outputs (256×256 vs. 512×512 for SD).

**The consequence:** For any application requiring sub-second or even sub-5-second image generation — interactive editing tools, real-time content creation, chatbot-integrated image generation, iterative refinement workflows — CM3Leon is impractical regardless of its FID advantage. The latency is structural, not an implementation artifact: the 1024 sequential forward passes through a 7B-parameter model each depend on the previous token, so the generation cannot be parallelized. Figure 11 confirms this: throughput scales sub-linearly with batch size because the sequential dependency within each image dominates. The INT8 quantization provides only a 23% speedup (faster per-step math, but the same number of steps). Even with infinite hardware, the minimum latency is bounded by the time to run 1024 sequential forward passes.

**What evidence exists in the paper:** Figures 10 and 11 provide the latency and throughput numbers. The paper does not explicitly discuss the latency implications or compare wall-clock time to diffusion and non-autoregressive models — the numbers are presented without framing.

**Mitigation status:** The paper does not address the latency limitation or propose mitigation strategies. Section D (Figures 10, 11) reports the numbers but offers no analysis of whether speculative decoding, early exiting, distillation, or reducing the number of image tokens (e.g., a coarser VQ-VAE with 256 or 512 tokens per image) could improve inference speed. The paper's focus on training efficiency (5× less training compute) does not extend to inference efficiency.

---

### 6.4 The Model's Image-to-Text Performance Collapses on Knowledge-Intensive Tasks, Exposing a Stark Text-Data Bottleneck

CM3Leon's pretraining text data is extremely limited compared to its vision-language competitors: it is trained on only the **~3B text tokens** contained in the 340M Shutterstock captions, compared to Flamingo's 100B tokens and OpenFlamingo's 40B tokens (Section 4.2). Table 2 reveals the consequence: while CM3Leon is competitive on tasks that primarily require visual understanding with limited world knowledge (VizWiz: 37.6 vs. Flamingo's 28.8; VQA2: 47.6 vs. Flamingo's 51.8), it **collapses on knowledge-intensive benchmarks**. On OKVQA, which explicitly requires external knowledge to answer visual questions (e.g., identifying a landmark, knowing the function of an obscure object), CM3Leon achieves 23.8 compared to Flamingo's 44.7 — a 21-point gap. On VisDial, which requires multi-turn reasoning grounded in visual and world knowledge, CM3Leon scores 22.6 NDCG compared to Flamingo's 48.4 — less than half the performance.

**The consequence:** CM3Leon cannot serve as a general-purpose vision-language model for tasks requiring factual knowledge beyond what is visually apparent. The 3B Shutterstock text tokens are captions (short, descriptive, focused on visual content), providing effectively zero coverage of encyclopedic knowledge, historical facts, scientific concepts, or cultural references that a model trained on web-scale text would acquire. A practitioner needing a model that can answer "What monument is this?" or "When was this building constructed?" or "What is this tool used for?" should expect CM3Leon to fail where Flamingo or a GPT-4V-style model would succeed. The paper implicitly acknowledges this by noting Flamingo's text-data advantage but does not position it as a fundamental capability ceiling.

**What evidence exists in the paper:** Table 2 provides the direct comparison. The large performance gap on OKVQA (23.8 vs. 44.7) and VisDial (22.6 vs. 48.4) is the quantitative evidence that the text-data bottleneck limits knowledge-intensive performance. Section 4.2 notes that Flamingo was trained on 100B tokens and OpenFlamingo on 40B, but frames this as making CM3Leon's competitiveness "notable" rather than identifying the knowledge gap as a limitation.

**Mitigation status:** The paper does not explicitly acknowledge this as a limitation or propose mitigation strategies. The SFT stage adds only 30B tokens of vision-language task data (Table 4), which is itself predominantly task-specific (VQA, captioning) rather than broad world knowledge. The paper does not discuss whether pre-training on additional text-only data, interleaving text-only and multi-modal documents during pretraining, or using a retrieval system that can fetch text-only knowledge documents would address this gap.

---

### 6.5 All Results Are on a Single Licensed Dataset with a Single Model Architecture, Leaving Generalization Uncharacterized

The paper's entire training and evaluation pipeline is built on a single dataset — licensed images from Shutterstock — and a single model architecture family (decoder-only transformers with the CM3 objective). Section 2.1 positions the exclusive use of Shutterstock as an ethical strength ("we use only licensed images... to avoid concerns related to images ownership and attribution, without sacrificing performance"), but it also means the model's behavior on other image distributions — photographs from different sources, artistic illustrations, scientific diagrams, medical images, satellite imagery — is entirely unknown.

**The consequence:** A practitioner deploying CM3Leon on in-the-wild images (user-uploaded photos from smartphones, screenshots, scanned documents, historical photographs) has no basis for predicting performance. The Shutterstock dataset consists of professionally composed, well-lit, high-resolution stock photographs with descriptive captions — it is not representative of the long tail of real-world images. The model may fail in unpredictable ways on images with different lighting conditions, unusual compositions, non-photographic styles, or caption styles that differ from Shutterstock's descriptive format. Similarly, the exclusive use of a decoder-only transformer means the paper provides no evidence about whether the training recipe (retrieval augmentation, CM3 objective modifications, SFT) transfers to other architectures — encoder-decoder models, diffusion backbones adapted for token prediction, or non-autoregressive token models. The claim that "the full text-LLM training recipe transfers to multi-modal models" (Section 4, paraphrased) is tested on exactly one architecture with one dataset.

**What evidence exists in the paper:** The paper evaluates only on standard benchmarks derived from MS-COCO (text-to-image FID, image-to-text CIDEr, VQA2, VizWiz, OKVQA) — all of which feature natural photographs. There are no evaluations on out-of-distribution image types, artistic styles, diagrammatic reasoning, document understanding, or any non-photographic visual domain. The SFT stage includes some non-photographic tasks (Canny edge maps, segmentation maps, pose skeletons, scribbles) but all are derived from or conditioned on the Shutterstock training images; the model never sees truly out-of-domain visual inputs.

**Mitigation status:** The paper does not acknowledge this as a limitation or discuss generalization. The Shutterstock-only training is framed as an ethical benefit throughout, never as a potential constraint on the breadth of visual understanding. Future work would need to evaluate CM3Leon on diverse image distributions (e.g., DomainNet, ImageNet-R, or domain-specific benchmarks) and potentially train variants on broader data mixtures (licensed + public domain + synthetic) to characterize and improve generalization.

---

### 6.6 The Difficulty Estimation and Adaptation Framework That Drives 4× Efficiency Gains in the Reference Paper Has No Analog in CM3Leon — Retrieval Is Applied Uniformly Regardless of Prompt Difficulty

This limitation is structural rather than experimental: CM3Leon applies retrieval augmentation uniformly — 2 retrieved documents for every prompt, regardless of whether the prompt describes a common concept that the model's parameters already represent well (e.g., "a cat sitting on a couch") or a rare tail entity that the model has never seen (e.g., "an Armenian khachkar surrounded by pomegranates"). The paper's Figure 1 showcases CM3Leon's ability to generate tail entities like khachkars (Armenian stone crosses), but provides no analysis of whether retrieval is more valuable for some prompts than others or whether the 2-document budget could be allocated more efficiently.

**The consequence:** The uniform retrieval strategy leaves two types of efficiency on the table. **First**, for common concepts well-represented in the pretraining data, retrieval may be unnecessary — the model likely achieves similar FID with 0 or 1 retrieved documents, and the retrieval latency is wasted. **Second**, for extremely rare concepts, 2 retrieved documents may be insufficient — the model might benefit from 5 or 10 retrieved examples but is capped at 2. A compute-optimal allocation (analogous to the reference paper's difficulty-conditioned strategy selection) would estimate the rarity of the prompt's concepts, allocate retrieval budget accordingly (0 documents for common concepts, 5+ for rare ones), and potentially combine retrieval with different decoding strategies (CFG-heavy for well-represented concepts, retrieval-heavy for rare ones). The paper's framework provides no mechanism for this adaptation.

**What evidence exists in the paper:** The paper provides no evidence on this limitation because it does not study prompt-conditioned retrieval allocation. Table 1 shows overall FID improves with more retrieved documents (0 → 1 → 2), but this is aggregated across all 30K MS-COCO prompts. There is no per-prompt difficulty analysis, no stratification by concept frequency in the training data, and no experiment varying the number of retrieved documents as a function of prompt characteristics.

**Mitigation status:** The paper does not address this limitation. Section 2.1 describes the retrieval strategy as fixed (2 documents each for image and text, with 3 randomly selected for training), and the inference setting in Table 1 sweeps over 0, 1, 2 retrieved documents only as fixed global configurations. The paper does not discuss adaptive retrieval budgets, prompt difficulty estimation, or any mechanism for allocating retrieval compute non-uniformly across prompts. This is a missed opportunity: the Shutterstock training set provides a natural signal for concept rarity (the frequency of each caption n-gram or CLIP embedding cluster size), and a difficulty-conditioned retrieval policy could improve the cost-quality Pareto frontier beyond the uniform 2-document baseline.

## 7. Implications and Future Directions

### How This Work Changes the Landscape

CM3Leon does not introduce a fundamentally new architecture or training objective — it uses the same CM3 framework as Aghajanyan et al. (2022) and the same retrieval-augmented pretraining approach as RA-CM3 (Yasunaga et al., 2022). Its contribution is better understood as a **methodological reframing**: the paper demonstrates that the dominant narrative circa 2023 — diffusion models are more efficient and practical than autoregressive models for multi-modal generation — was a product of suboptimal training recipes, not an inherent property of the model families. By faithfully porting the full text-only LLM training paradigm (retrieval-augmented pretraining + multi-task SFT + contrastive decoding) to the token-based autoregressive setting, CM3Leon achieves state-of-the-art text-to-image FID (4.88) with ~5× less training compute than the previous best autoregressive model (PARTI-20B; Figure 2, Table 1), while simultaneously handling a far broader range of tasks after SFT (text-guided editing, structure-conditioned generation, VQA, long-form captioning; Figures 5–7, 15–16).

This reframing has three concrete consequences for the field:

**First, it reopens the autoregressive-vs-diffusion debate on new terms.** Prior to this work, the debate was largely settled in diffusion's favor on practical grounds: diffusion models were cheaper to train, faster at inference, and achieved better benchmark numbers. CM3Leon shows that the cost and quality gap can be closed — and even reversed — through training recipe improvements alone, without architectural innovation. The scaling curve in Figure 2 (FID vs. A100 hours, log scale) suggests autoregressive models may scale *better* with compute than diffusion models when trained with retrieval augmentation. This does not mean diffusion models are obsolete, but it does mean that the field's near-exclusive focus on diffusion for image generation was premature — autoregressive models deserve significant reinvestment, particularly for applications requiring unified text-and-image generation.

**Second, it establishes SFT as a first-class capability unlocker for multi-modal models.** The paper's most conceptually significant move is showing that supervised instruction tuning — the technique behind ChatGPT's flexibility — transfers cleanly to the multi-modal domain. The SFT stage (Section 4) takes a pretrained text-to-image model and, by fine-tuning on a diverse mixture of tasks formatted as instruction-input-output sequences (using the same CM3 objective, no architectural changes), produces a single model that can edit images, generate from structural conditioning, answer visual questions, and produce long-form captions. This is not an incremental capability improvement — it is a *category change*: the model goes from being a text-to-image generator to being a general-purpose multi-modal agent. The fact that this works with only ~30B SFT tokens (Table 4) on top of 2.4T pretraining tokens suggests that multi-modal instruction tuning is highly data-efficient, consistent with findings from text-only LLMs.

**Third, it provides a clean diagnostic for the retrieval-dependence problem in retrieval-augmented models.** The paper identifies a specific design flaw in RA-CM3 (query-pair loss up-weighting) that creates an unintended dependency on retrieval at inference time, and shows that simply removing this weighting produces a model that is strong both with retrieval (FID 4.88) and without it (FID 10.82; Table 1). This is a reusable insight: retrieval augmentation should be treated as a helpful but optional conditioning signal, not baked into the training objective in a way that makes the model fail without it. The paper provides no controlled ablation isolating exactly this variable (the comparison to RA-CM3 conflates dataset size, model size, and multiple recipe changes), but the diagnostic logic is clear and actionable for future retrieval-augmented training efforts.

**What this work does NOT do:** It does not provide a systematic scaling law for multi-modal autoregressive models (the three model sizes — 350M, 760M, 7B — are suggestive but insufficient for fitting a power law). It does not demonstrate that the SFT-enabled capabilities (editing, grounding, structure-conditioning) are quantitatively competitive with specialized models — the evidence is exclusively qualitative. It does not solve the autoregressive inference latency problem (11.8 seconds per 256×256 image; Figure 10). And it does not characterize generalization beyond the Shutterstock image distribution or the specific tokenizer and architecture used.

**Which research directions become more attractive after this work:**
- **Autoregressive multi-modal models as generalist agents:** The SFT results, while qualitatively demonstrated, suggest that a single token-based decoder-only model can be a unified interface for arbitrary vision-and-language tasks. This makes research on scaling these models (more parameters, more SFT tasks, more modalities) a high-upside bet.
- **Training recipe engineering for multi-modal models:** The paper's core message is that recipe matters more than architecture. This shifts attention from novel architectures toward systematic ablation of training design choices: objective modifications, data mixtures, retrieval strategies, SFT task selection and balancing.
- **Retrieval augmentation as a compute multiplier:** The 2×+ FID improvement from adding retrieval (10.82 → 4.88) makes retrieval-augmented training arguably the single highest-impact technique for improving multi-modal generation efficiency. Research on better retrievers, dynamic retrieval budgets, and multi-modal memory banks becomes more compelling.

**Which directions become less attractive:**
- **Small-scale autoregressive image models without retrieval or SFT:** The paper shows that the performance of an autoregressive model without retrieval (FID 10.82) is competitive but not state-of-the-art, and that the biggest gains come from retrieval and SFT — not from incremental architectural tweaks to the base autoregressive framework.
- **Narrow text-to-image specialists:** The SFT results, even qualitatively, demonstrate the value of a unified model. Research on single-purpose text-to-image models (whether autoregressive or diffusion) now must justify why generality is not needed for the target application.

---

### Follow-Up Research This Work Enables

**Controlled ablation of the SFT stage to isolate its causal contribution to multi-task capabilities.** The paper demonstrates that SFT-CM3Leon can perform text-guided editing, structure-conditioned generation, VQA, and long-form captioning, but provides no comparison to the pretrained model on these tasks. A critical open question is: how much of this capability is *latent* in the pretrained model (accessible through careful prompting) and how much is *learned* during SFT? A strong follow-up would evaluate the pretrained CM3Leon-7B (without SFT) on the same vision-language benchmarks (Table 2) using zero-shot prompting with the same templates used during SFT (e.g., "Describe the given picture. <break> {image}"). If the pretrained model achieves non-trivial performance, then SFT primarily improves instruction-following and format consistency rather than teaching fundamentally new capabilities. If the pretrained model performs near chance, then SFT is teaching the model to extract and articulate visual understanding that was only implicit during pretraining. Either outcome would refine our understanding of what multi-modal pretraining actually learns.

**Quantitative benchmarking of SFT image generation capabilities against specialized models.** The paper's "unprecedented controllability" claim is backed by approximately 20–30 qualitative examples. A necessary follow-up is a rigorous quantitative comparison of SFT-CM3Leon against specialized models on standard benchmarks: text-guided editing against InstructPix2Pix (Brooks et al., 2023) on the InstructPix2Pix benchmark (measuring LPIPS for input fidelity and CLIP directional similarity for edit faithfulness), structure-conditioned generation against ControlNet (Zhang & Agrawala, 2023) on MS-COCO with edge/depth/pose conditioning (measuring FID and conditioning accuracy via edge F1, keypoint PCK, or depth RMSE), and spatial grounding against GLIGEN (Li et al., 2023) or LayoutGPT on the COCO/OpenImages layout-to-image benchmarks (measuring bounding box accuracy and FID). This would establish whether the unified model's generality comes at a cost in per-task performance, and if so, how large that cost is. The paper's own SFT data pipeline (Section 4.1) uses the exact same data sources as these specialized models (InstructPix2Pix data, ControlNet-processed Shutterstock), making a direct comparison straightforward to implement.

**Characterization of retrieval dependence as a function of prompt concept frequency.** Table 1 shows aggregate FID improves with more retrieved documents (0: 10.82, 1: 5.78, 2: 4.88), but this is averaged across all 30K MS-COCO prompts. A critical follow-up would stratify FID by the frequency of the prompt's key concepts in the Shutterstock training data. The hypothesis (motivated by the paper's Figure 1 showcasing tail entities like "khachkar"): retrieval provides most of its benefit for rare concepts that the model's parameters cannot memorize, while common concepts (dog, car, person) see minimal improvement from retrieval because the 7B parameters already capture their visual appearance. A researcher could bin MS-COCO prompts by the training-set frequency of their noun phrases (using the Shutterstock caption corpus), then measure FID at 0, 1, and 2 retrieved documents per frequency bin. If retrieval benefit is concentrated in the long tail, this would motivate adaptive retrieval budgets — 0 documents for common concepts (saving latency), 5+ documents for rare concepts (improving quality) — analogous to the difficulty-conditioned compute allocation in the reference paper's framework. The paper's frozen CLIP retriever and pre-encoded Shutterstock memory bank make this experiment straightforward: it requires only re-running inference with different retrieval configurations and no retraining.

**Scaling laws for multi-modal autoregressive models with and without retrieval.** The paper trains three model sizes (350M, 760M, 7B) but this is insufficient to fit a reliable power law relating model size, training tokens, and FID. A systematic scaling law study — following the methodology of Hoffmann et al. (2022) for text or Aghajanyan et al. (2023) for multi-modal models — would train a range of CM3Leon variants (e.g., 150M through 30B parameters) to full convergence, measuring FID as a function of model size, training tokens, and retrieval configuration (0, 1, 2 documents). The key question: does retrieval augmentation change the exponent of the scaling law? If retrieval makes smaller models disproportionately better (the 350M model's FID of 14.20 with retrieval is already better than some non-retrieval models), then retrieval is not just a constant quality boost but a compute multiplier that shifts the entire scaling curve. Alternatively, if retrieval provides a constant additive improvement regardless of model size, then the scaling exponent is unchanged and retrieval can be treated as an independent quality lever. Figure 2 provides suggestive evidence (the CM3Leon curve is steeper than competitors) but cannot distinguish between these hypotheses without more data points and controlled ablations.

**Non-autoregressive decoding for token-based multi-modal models to close the inference latency gap.** CM3Leon's 11.8 seconds per image (Figure 10) compared to MUSE's 0.5 seconds (Chang et al., 2023) highlights the fundamental latency bottleneck of autoregressive decoding: 1024 sequential forward passes. A research direction the paper enables but does not pursue is adapting non-autoregressive decoding methods from text (mask-predict; Ghazvininejad et al., 2019) or from MUSE to CM3Leon's architecture and tokenizer. The key experiment: take the pretrained CM3Leon-7B and fine-tune it with a mask-predict objective where some fraction of the 1024 image tokens are masked and the model predicts them in parallel, then iteratively unmasks tokens over ~24 steps (as in MUSE). If this yields an FID of, say, 6-8 (worse than the autoregressive 4.88 but far better than the 10.82 zero-shot baseline), while achieving sub-second inference, it would make token-based models practical for interactive applications while preserving their unified text-and-image generation capabilities. This is a straightforward extension because the CM3 objective already trains the model for infilling (filling in masked image regions), and the only required change is the decoding procedure. The paper's CD-K and CFG guidance methods would transfer directly to the non-autoregressive setting as well.

**Text-data augmentation for knowledge-intensive vision-language tasks.** Table 2 reveals a sharp capability cliff on knowledge-intensive benchmarks: CM3Leon achieves 23.8 on OKVQA vs. Flamingo's 44.7, and 22.6 NDCG on VisDial vs. 48.4. This is almost certainly driven by CM3Leon's ~3B Shutterstock text tokens (captions only) vs. Flamingo's 100B tokens of diverse text. A natural follow-up would augment CM3Leon's pretraining with additional text-only data — interleaving text-only documents (from Wikipedia, books, or web text) with the multi-modal Shutterstock sequences using a special modality marker to distinguish text-only from multi-modal context. The key experiment: pretrain a CM3Leon variant where 50% of training sequences are text-only documents (no images) and 50% are the standard retrieval-augmented multi-modal sequences, then evaluate on OKVQA, VisDial, and other knowledge-intensive vision-language benchmarks. If this closes the gap with Flamingo, it demonstrates that the text-data bottleneck is the primary limitation, not the token-based decoder-only architecture. This also connects to broader questions about whether multi-modal models should be trained on joint image-text data from the start or whether visual understanding can be effectively post-hoc grafted onto a text-only LLM — a question that subsequent models like GPT-4V and Gemini have made central to the field.

---

### Practical Applications and Downstream Use Cases

**Licensed-data text-to-image generation for commercial deployment where copyright compliance is mandatory.** The paper's exclusive use of Shutterstock's 340M licensed images (Section 2.1) directly addresses the legal and ethical uncertainties surrounding models trained on web-scraped data. For enterprises that cannot risk copyright infringement claims — advertising agencies generating campaign imagery, e-commerce platforms creating product visualizations, publishing houses producing book covers — CM3Leon provides a state-of-the-art FID (4.88) with a fully attributable training data pipeline. The retrieval augmentation during inference further strengthens the attribution story: each generated image is conditioned on specific Shutterstock reference images that can be cited, providing a transparent provenance chain. The 9.1 seconds per image (INT8; Figure 10) is acceptable for batch creative workflows where images are generated in the background, not in real-time interactive tools.

**Unified multi-modal content creation pipelines where a single model replaces multiple specialized tools.** After SFT, CM3Leon can perform text-to-image generation, text-guided image editing, structure-conditioned generation (pose, edge, depth, segmentation), spatial grounding, and image captioning — all with the same model weights and inference infrastructure (Section 4). For a creative team that currently uses separate models for each of these tasks (Stable Diffusion + ControlNet for conditioning, InstructPix2Pix for editing, BLIP-2 for captioning), deploying a single SFT-CM3Leon instance reduces operational complexity (one model to maintain, update, and monitor) and enables workflows that chain these capabilities without switching models — for example, generating an image from a text prompt, editing it with follow-up instructions, generating a caption for the edited result, and extracting a pose skeleton for further generation, all within a single autoregressive sequence. The caveat is that the paper provides no quantitative evidence that the unified model's per-task quality matches dedicated tools, so deployment should be preceded by task-specific benchmarking.

**Assistive technology for visually impaired users, leveraging CM3Leon's strong zero-shot VizWiz performance.** The paper's SFT-CM3Leon-7B achieves 37.6 accuracy on VizWiz (Table 2), outperforming Flamingo-9B (28.8) by a substantial margin. VizWiz consists of questions asked by blind users about images they've taken with their smartphones — often poorly framed, blurry, or oddly lit — making it a realistic proxy for real-world assistive technology deployments. CM3Leon's advantage may stem from its end-to-end token-based architecture: image tokens are processed natively by the same transformer that generates text, rather than being compressed through a frozen vision encoder whose features may not capture the unusual visual characteristics of user-taken photos. A deployment scenario would integrate SFT-CM3Leon into a mobile assistive app where users photograph their surroundings and ask questions ("What does this sign say?", "Is this medication bottle the right one?", "What color is this shirt?"). The model's ability to generate long-form descriptions (Figure 7) and chain-of-thought reasoning (ScienceQA examples) extends its utility beyond simple VQA to more complex assistive queries.

**Efficient training for research groups with limited compute budgets, via retrieval augmentation as a compute multiplier.** Figure 2 and Table 1 demonstrate that retrieval augmentation provides a massive efficiency boost: the 350M CM3Leon with retrieval achieves FID 14.20, competitive with or better than several larger non-retrieval models, while the 760M model achieves FID 6.61. For academic labs or startups that cannot afford to train 7B+ parameter models from scratch, the paper's recipe suggests a high-return strategy: train a smaller model (350M–760M parameters) with retrieval augmentation on a modest licensed or public-domain dataset, then apply SFT for task-specific capabilities. The 350M model was trained for only 1.4T tokens (Table 3) — far less than the 2.4T tokens for the 7B model — making it feasible on a handful of GPUs over weeks rather than months. The key infrastructure requirement is the retriever (frozen CLIP + MIPS over the training set), which is a one-time indexing cost amortized across the entire training run. This recipe lowers the barrier to entry for multi-modal model research and enables domain-specific adaptations (medical imaging, scientific figures, architectural renderings) without requiring web-scale training data.

---

### When to Prefer This Method

The paper does not articulate an explicit tradeoff matrix against named alternatives. It positions CM3Leon as a general demonstration that the text-only LLM training recipe transfers to multi-modal autoregressive models, achieving state-of-the-art text-to-image FID with 5× less training compute than PARTI (Section 3.2, Figure 2, Table 1), and additionally enabling diverse SFT capabilities (Section 4). The comparison to diffusion models is made through the scaling plot (Figure 2) and inference latency table (Figure 10), but the paper does not provide a decision rule for when a practitioner should choose CM3Leon over, say, Stable Diffusion or MUSE. The absence of quantitative SFT task metrics (editing accuracy, structure-conditioning fidelity, spatial grounding precision) further prevents a principled tradeoff analysis: we cannot say whether CM3Leon's unified model sacrifices per-task quality relative to specialized alternatives because the paper provides no per-task quality numbers.

The paper does, however, implicitly identify conditions where CM3Leon is **clearly preferred** over the baseline it directly improves upon (non-retrieval autoregressive models like PARTI) and conditions that remain **open questions** pending further benchmarking. Based on the evidence provided:

**Prefer the CM3Leon recipe (retrieval-augmented pretraining + SFT) when:**
- Training compute efficiency for text-to-image generation is a primary constraint, and retrieval infrastructure (frozen CLIP + MIPS memory bank) is available during both training and inference. The ~5× training compute reduction vs. PARTI (Figure 2) and the massive FID improvement from retrieval (10.82 → 4.88; Table 1) make this the dominant training paradigm among autoregressive methods.
- The deployment requires a single model handling diverse vision-and-language tasks (text-to-image, image-to-text, editing, spatial grounding) and operational simplicity outweighs per-task optimality. The SFT stage enables this unification within a single set of weights (Section 4).
- Copyright-compliant training data is a hard requirement. The exclusive use of licensed Shutterstock images (Section 2.1) provides an attributable data pipeline that web-scraped alternatives cannot offer.

**The paper does not provide sufficient evidence to prefer CM3Leon over alternatives when:**
- Inference latency is a primary constraint. MUSE generates images in 0.5 seconds vs. CM3Leon's 9.1–11.8 seconds (Figure 10), a ~18–24× difference that dominates any FID advantage for real-time applications.
- Per-task image editing or structure-conditioning quality must match specialized models. The paper provides no quantitative metrics for SFT image generation tasks to enable this comparison.
- The application requires broad world knowledge for vision-language tasks. The 3B text-token training budget creates a substantial performance gap on knowledge-intensive benchmarks like OKVQA and VisDial (Table 2) compared to text-rich models like Flamingo.
- The image distribution is far from Shutterstock's professional stock photography. The paper provides no out-of-distribution evaluation to characterize generalization.
