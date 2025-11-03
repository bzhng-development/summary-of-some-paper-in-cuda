# Scaling Autoregressive Multi‑Modal Models: Pretraining and Instruction Tuning

**ArXiv:** [2309.02591](https://arxiv.org/abs/2309.02591)
**Authors:** Lili Yu, Bowen Shi, Ramakanth Pasunuru, Benjamin Muller, Olga Golovneva, Tianlu Wang, Arun Babu, Binh Tang, Brian Karrer, Shelly Sheynin, Candace Ross, Adam Polyak, Russell Howes, Vasu Sharma, Puxin Xu, Hovhannes Tamoyan, Oron Ashual, Uriel Singer, Shang‑Wen Li, Susan Zhang, Richard James, Gargi Ghosh, Yaniv Taigman, Maryam Fazel‑Zarandi, Asli Celikyilmaz, Luke Zettlemoyer, Armen Aghajanyan
**Institutions:** 

## 🎯 Pitch

CM3Leon introduces a groundbreaking multi-modal transformer model that excels in both text-to-image and image-to-text generation by integrating retrieval-augmented training with a unified decoder-only architecture. This innovation significantly reduces compute requirements while achieving state-of-the-art quality, enabling practical applications in creative design, accessibility, and content moderation, and demonstrating the efficacy of adapting language model strategies to multi-modal tasks.

---

## 1. Executive Summary
CM3Leon is a retrieval-augmented, token-based, decoder-only multi-modal model that can both generate and infill text and images using a single autoregressive transformer. By scaling pretraining on a licensed Shutterstock corpus and then instruction-tuning across diverse vision–language tasks, it achieves state-of-the-art text-to-image quality on MS-COCO with much lower training compute than prior systems and adds fine-grained controllability for editing and image-conditioned generation.

## 2. Context and Motivation
- Problem addressed
  - Autoregressive image generators produce globally coherent images but have been considered too expensive to train and decode compared to diffusion models. Existing multi-modal systems also often specialize (e.g., text-to-image only) rather than supporting a broad set of image and text tasks within a single model.
- Why it matters
  - Practical: Efficient, general-purpose models unlock applications like precise text-guided editing, pose/edge/segmentation-conditioned generation, and open-ended image captioning/VQA.  
  - Scientific: It tests whether the “language model recipe” (large-scale retrieval-augmented pretraining followed by instruction-style supervised fine-tuning) transfers to multi-modal modeling.
- Prior approaches and gaps
  - Diffusion models (e.g., Stable Diffusion, Imagen) are strong but typically need many denoising steps and often rely on separate text encoders (§1; Related Work).  
  - Token-based autoregressive models (e.g., DALL·E, Parti) show good global coherence but are viewed as compute-heavy; earlier retrieval-augmented variants (RA-CM3) improved efficiency but did not scale to state-of-the-art quality or broad controllability (§1, §3.2 Table 1).  
  - Instruction tuning is well-studied for text-only LLMs but underexplored for multi-modal models (§4).
- Positioning
  - CM3Leon adapts the text-LM recipe to multi-modality: (1) large-scale retrieval-augmented pretraining over licensed data; (2) multi-task supervised instruction tuning with interleaved image–text tokens; and (3) a single decoder-only architecture that unifies text-to-image and image-to-text tasks (§1–§4).

## 3. Technical Approach
CM3Leon is an autoregressive transformer that learns to generate sequences containing both text and image tokens. The core pipeline has four parts.

- Data and tokenization (§2.1)
  - Images (256×256) are discretized into 1024 tokens using a VQ-style tokenizer (vocab size 8192).  
  - Text is tokenized with a custom BPE-like tokenizer (vocab 56,320).  
  - A special `<break>` token marks modality transitions so the model can read sequences like “text <break> image <break> text …”.
  - A licensed Shutterstock corpus is used throughout, addressing ownership/attribution concerns.
- Retrieval augmentation for pretraining (§2.1 Data; “Retrieval Augmentation”)
  - Goal: Enrich the context with relevant multi-modal documents to reduce the need for the generator to memorize long-tail knowledge.  
  - How retrieval works:
    - A CLIP-based bi-encoder builds embeddings for the text and image parts of each memory-bank document; the two embeddings are averaged into a single vector.  
    - Given a query sequence, Maximum Inner Product Search (MIPS) retrieves candidates ranked by CLIP similarity.  
  - Sampling strategy balances relevance, modality, and diversity:
    - It requires multi-modal retrieved items (both text and image parts).  
    - It drops near-duplicates by skipping results with too-high similarity (uses a threshold; only keep results with relevance score ≤ 0.9).  
    - It uses “query dropout” (randomly removes 20% of query tokens) to diversify retrieval.
  - Training assembly: For each caption–image pair, the model samples three retrieved examples and interleaves them with the query pair, effectively quadrupling the number of tokens processed per training example (§2.1, Fig. 9 shows the packed sequence layout).
- Objective: CM3-style causal masked infilling (§2.2)
  - Idea in plain terms: convert any input into a fill-in-the-blank problem, then train with next-token prediction.  
  - Mechanism:
    - The input sequence is transformed by masking spans and appending the masked content after a special `<infill>` marker. Example: “Image of a chameleon: [image]” becomes “Image of <mask>: [image] <infill> a chameleon”.  
    - The model is trained with standard autoregressive loss over the full sequence.  
    - Two notable tweaks:
      - Remove RA-CM3’s extra loss weight on the query pair so the model doesn’t overfit to relying on retrieval, which harms “no-retrieval” zero-shot generation (§2.2).  
      - Do not allow masks to span across `<break>` tokens so the model never learns to start generating an image from the middle of an image segment (§2.2).
  - This single objective covers text-to-image (“Image of a …:”), image-to-text (prompt “Image of <mask>: [image] <infill> …”), and mixed infilling.
- Architecture and training (§2.3–§2.4; Appx. B.2 Table 3)
  - Decoder-only transformer (no biases, no dropout, 4096 context length).  
  - Three sizes: 350M, 760M, 7B parameters; trained for roughly 1.4T, 1.9T, and 2.4T tokens, respectively.  
  - Weight initialization is carefully tuned for stability; absolute positional embeddings initialized near zero (§2.3).  
  - Figure 3 shows validation perplexity improving smoothly over training and not saturating even at the end.
- Decoding strategies (§3.1, Eqs. 1–3; Fig. 4)
  - Classifier-Free Guidance (CFG) for autoregressive generation:
    - Because the CM3 objective already uses a `<mask>` token to represent “no text,” the model can produce an unconditional token stream by conditioning on `<mask>` instead of the prompt.  
    - Two logits are computed at each step: conditional `T(ty | tx)` and unconditional `T(ty | <mask>)`. They are mixed with weight α as in Eq. (2): `logits_cf = logits_uncond + α (logits_cond − logits_uncond)`.  
  - Contrastive Decoding with Top-K constraint (CD-K), a new self-contained alternative:
    - Contrastive decoding typically needs a strong and a weak model; CM3Leon reuses the same model twice—conditional stream as strong (`p_EXP`) and unconditional stream as weak (`p_AMA`)—and scores tokens by the log-probability ratio (§3.1 “Contrastive Decoding TopK”).  
    - To avoid degenerating into greedy decoding, the candidate set is relaxed from “α times the max probability” to “α times the k-th largest probability” (Eq. 3).  
    - Fig. 4 (right) shows CD-K and TopP have complementary strengths; mixing them lowers FID further than either alone.

- Supervised Fine-Tuning (SFT) with instruction-style tasks (§4; Fig. 5; Appx. E.1–E.3)
  - Format: Concatenate task instruction, inputs (text and/or images), and target outputs into one sequence and train with the same CM3 objective.  
  - Image-to-image grounded generation tasks are created by applying ControlNet-like feature extractors (edge maps, HED boundaries, human pose, depth) on Shutterstock images to build ≈7M examples; object detection data provides ≈3M spatially grounded examples where objects are specified by discrete layout tokens; an OCR-derived “how-to-write” sign/logo dataset adds ≈200k examples (§4.1, Fig. 5).  
  - Text-conditioned tasks include COCO, Flickr30k, Image Paragraph, Localized Narratives, VQA2, VizWiz, OKVQA, and ScienceQA with multiple prompt templates to increase robustness (§4.2; Appx. Table 5).  
  - SFT scale: ≈30B tokens with learning rate 5e-5 on 760M and 7B models (Appx. Table 4).

## 4. Key Insights and Innovations
- Retrieval-augmented pretraining scaled for multi-modality
  - What’s new: Large-scale RA-CM3-style training on a licensed Shutterstock memory bank with careful retrieval sampling (multi-modal documents favored; similarity thresholding; query dropout) and a simple, unified CM3 infilling objective (§2.1–§2.2).  
  - Why it matters: It significantly improves data efficiency and expands knowledge coverage for an autoregressive model, enabling strong zero-shot text-to-image quality with much less compute (Table 1; Fig. 2).
- A single decoder-only model that natively handles both text-to-image and image-to-text
  - What’s new: Unified objective and token stream that interleaves text and image tokens with `<break>`, plus an explicit change not to mask across modality boundaries (§2.2).  
  - Why it matters: The same model supports generation, infilling, and cross-modal tasks and becomes a natural target for instruction tuning (Fig. 5), enabling controllable editing and image-grounded generation (Fig. 6).
- Self-contained contrastive decoding (CD-K)
  - What’s new: A contrastive-decoding variant that uses the same model twice (conditional vs. unconditional) and relaxes the candidate set via a k-th largest probability criterion (Eq. 3).  
  - Why it matters: It delivers generations complementary to CFG/TopP; mixing half TopP and half CD-K candidates reduces FID further (Fig. 4 right).
- Demonstration that instruction tuning substantially improves multi-modal controllability
  - What’s new: An SFT recipe that combines text-guided editing, structure-conditioned generation (pose/edge/segmentation/depth), spatial layout control, OCR-like sign creation, and multiple VQA/captioning tasks (Fig. 5; Appx. Table 5).  
  - Why it matters: After SFT, CM3Leon can follow nuanced instructions for editing and grounding without changing the architecture (§4.1–§4.2, Fig. 6–7).

## 5. Experimental Analysis
- Evaluation setup
  - Text-to-image quality on MS-COCO 30k is measured using FID (Fréchet Inception Distance), where lower is better; the model generates 8 samples per prompt and a CLIP model re-ranks them to pick the best sample (Table 1).  
  - Ablations on decoding and guidance weights are done on held-out MS-COCO prompts (Fig. 4).  
  - SFT evaluation covers captioning (CIDEr metric), VQA accuracy, and dialog/narrative metrics on public benchmarks (Table 2).  
  - Training dynamics are monitored via validation perplexity (Fig. 3). Inference latency and throughput are compared against other model families (Fig. 10–11).
- Main quantitative results
  - State-of-the-art FID with less compute:
    - Table 1 shows `CM3Leon-7B` achieves FID 4.88 on MS-COCO 30k when using two retrieved documents at inference, outperforming RE-IMAGEN (5.25), PARTI (7.23), MUSE (7.88), Stable Diffusion (12.60), and RA-CM3 (15.70).  
    - With zero retrieval at inference, `CM3Leon-7B`’s FID is 10.82; adding one retrieved example improves it to 5.78; two retrieved examples reach 4.88 (Table 1).  
    - Figure 2 plots FID vs. “Equivalent A100 GPU hours” and shows CM3Leon scales better than DALLE, Stable Diffusion, and PARTI as compute increases.
  - Decoding ablations (Fig. 4):
    - CFG weight has a similar optimal value across model sizes (left panel); deviating in either direction hurts FID.  
    - CD-K vs. TopP: each is competitive alone, but combining “half TopP + half CD-K” samples before CLIP re-ranking lowers FID further as the number of candidates grows (right panel). This shows complementary sample diversity.
  - Training dynamics (Fig. 3):
    - Perplexity steadily decreases across 350M/760M/7B models without saturation up to 1.4T–2.4T tokens; brief bumps correspond to learning-rate schedule changes when resuming after an epoch.
  - SFT zero-shot multi-modal text tasks (Table 2):
    - `SFT-CM3Leon-7B` reaches COCO CIDEr 61.6 (vs. Flamingo-9B 79.4), VQA2 47.6 (vs. 51.8), VizWiz 37.6 (better than Flamingo-9B’s 28.8), OKVQA 23.8 (vs. 44.7).  
    - Despite seeing far fewer text tokens in pretraining (≈3B vs. 40–100B noted in §4.2), it is competitive and even surpasses Flamingo on VizWiz (blind-user photographs).
  - Inference speed (Fig. 10–11):
    - For 256×256 images, `CM3Leon-7B` runs at 11.8s (BF16) or 9.1s (INT8), slower than MUSE 3B (0.5s) and Parti 3B (6.4s). Throughput improves with FasterTransformer kernels and model parallelism (Fig. 11), but autoregressive decoding remains comparatively slow.
- Qualitative evidence
  - Post-SFT examples show high-quality editing and structure-conditioned generation (Fig. 6), spatial grounding (Fig. 15), and long-form reasoning/captioning (Fig. 7, Fig. 16).
- Do the experiments support the claims?
  - The text-to-image results and ablations convincingly support that (a) retrieval helps substantially at inference, (b) the decoding recipe matters, and (c) the autoregressive approach can be compute-efficient when paired with retrieval (Table 1, Fig. 2, Fig. 4).  
  - The SFT results demonstrate generality and instruction-following, with strengths on VizWiz and reasonable zero-shot performance given limited text pretraining (Table 2).  
  - Latency data (Fig. 10–11) fairly shows autoregressive inference trade-offs compared to masked or diffusion models.

## 6. Limitations and Trade-offs
- Dependence on retrieval
  - The best FID uses retrieval at inference (Table 1). Without retrieval, `CM3Leon-7B`’s FID (10.82) is competitive but not state of the art. Performance relies on the quality/diversity of the memory bank and the CLIP-based retriever (§2.1).  
  - Although the training pipeline avoids overly similar items (similarity threshold and dropout), the paper does not detail near-duplicate filtering for the inference-time memory bank beyond retrieval thresholds; copying risk is mitigated but not fully analyzed.
- Resolution and tokenization
  - The model operates at 256×256 with 1024 tokens per image (§2.1). Scaling to higher resolutions would require more tokens and longer sequences, increasing compute and latency.  
- Inference speed
  - Autoregressive decoding is slower than diffusion or non-autoregressive masked generation. Even with INT8 and optimized kernels, 256×256 generation is ~9–12s vs. 0.5s for MUSE 3B (Fig. 10).  
- Data scope
  - Pretraining uses licensed Shutterstock data—ethically strong but domain-limited. Some visual concepts or world knowledge outside stock-image distributions might be underrepresented (§2.1).  
- Multi-modal text tasks after SFT
  - On several benchmarks (COCO CIDEr, VQA2, OKVQA), the model trails Flamingo (Table 2). With only ≈3B text tokens during pretraining (§4.2), language knowledge may limit zero-shot reasoning without additional data or retrieval augmentation for text tasks.
- Compute footprint
  - Training uses 1.4T–2.4T tokens (Appx. Table 3), which is still substantial. Figure 3 indicates under-saturation, suggesting even more compute could be beneficial but costly.

## 7. Implications and Future Directions
- Field impact
  - CM3Leon shows that the LLM recipe—retrieval-augmented pretraining plus instruction tuning—transfers effectively to multi-modal autoregressive models. It reopens the case for token-based generation as competitive with diffusion on both quality and controllability (Table 1; §4).
- Research directions enabled
  - Scaling to higher resolutions and longer contexts by improving tokenizers or using hierarchical generation (multi-scale image tokens).  
  - Better retrieval systems: joint training of the retriever and generator; adaptive memory construction; deduplication and provenance-aware retrieval.  
  - Decoding research: extend CD-K with sequence-level objectives, adaptive k/α, and hybrid CFG/CD-K schedules conditioned on prompt type (Fig. 4 indicates complementarity).  
  - More powerful instruction tuning: integrate chain-of-thought or rationale data for image reasoning, and expand cross-modal datasets with richer instructions (Appx. Table 5 hints at the value of diverse templates).  
  - General-purpose multi-modal assistants: The unified architecture already supports editing, layout control, and image-grounded text. Adding tools (e.g., segmentation, retrieval plugins) could make a versatile visual assistant.
- Practical applications
  - Creative design and marketing: controllable editing, logo/sign creation (“how-to-write” task), and layout-aware generation (§4.1; Fig. 6, Fig. 15).  
  - Accessibility: Strong performance on VizWiz suggests promise for assisting blind users with scene descriptions and Q&A (Table 2).  
  - Content moderation and data governance: The licensed-only training and explicit face filtering for examples show a path toward responsible deployment (§2.1; §4.1).

> Headline result: “CM3Leon-7B … achieves zero-shot MS-COCO FID of 4.88 with two retrieved documents” (Table 1), while using “5x less training compute than comparable methods” (highlighted in §1 and visualized in Fig. 2), and after SFT it supports nuanced editing and structure-controlled generation (Fig. 6, Fig. 15) as well as image-conditioned text tasks (Table 2, Fig. 7, Fig. 16).

Overall, the work demonstrates that retrieval-augmented, decoder-only token models can be efficient, high-quality, and broadly controllable across image and text modalities when trained with a unified infilling objective and instruction-tuned on diverse multi-modal tasks.
