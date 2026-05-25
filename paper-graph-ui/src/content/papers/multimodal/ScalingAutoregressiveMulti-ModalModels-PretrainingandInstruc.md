# Scaling Autoregressive Multi-Modal Models: Pretraining and Instruction Tuning

**ArXiv:** [2309.02591](https://arxiv.org/abs/2309.02591)

## 🎯 Pitch

CM3Leon introduces a single retrieval-augmented, autoregressive multi-modal model capable of both generating and editing text and images using a unified decoder-only architecture. By adapting large language model pretraining and instruction tuning recipes to the multi-modal setting, and integrating a novel contrastive decoding method, CM3Leon achieves state-of-the-art text-to-image performance with dramatically reduced compute cost—demonstrating that LLM-style scaling and flexibility can transform how vision and language models are trained and applied. This breakthrough paves the way for more general, controllable, and efficient AI systems capable of seamlessly bridging text and image domains in practical, real-world workflows.

---

## 1. Executive Summary
CM3Leon is a retrieval-augmented, autoregressive multi-modal model that generates and infills both text and images using a single decoder-only Transformer. It adapts the “pretrain + instruction-tune” recipe from text-only large language models to vision-language, introduces a self-contained contrastive decoding method, and achieves state-of-the-art zero-shot text-to-image quality on MS-COCO while using substantially less training compute (e.g., FID 4.88 with two retrieved examples; Table 1, §3.2).

## 2. Context and Motivation
- Problem the paper addresses
  - Autoregressive image token models produce globally coherent images but have been considered too compute-intensive compared to diffusion models for both training and inference. They also rarely support the full spectrum of tasks (text-to-image, image-to-text, editing, controllable generation) within one model (§1).
- Why it matters
  - Practical impact: Real applications need more than text-to-image—editing, spatial control, and instruction following are crucial. A single model that can both read and write images and text simplifies productization and enables new workflows (Figure 5).
  - Scientific impact: Demonstrates that scaling laws and recipes from LLMs (retrieval-augmented pretraining and supervised instruction tuning) carry over to multi-modal models, potentially shifting the field’s default approach beyond diffusion-only solutions (§1, §2, Figure 2).
- Prior approaches and their gaps
  - Diffusion models (e.g., Stable Diffusion, Imagen, Parti) excel at quality/efficiency but often rely on external text encoders and struggle with some forms of global coherence or tight multi-turn controllability (§1, §5).
  - Autoregressive token models (e.g., DALL·E, Parti) show coherence but are expensive and typically specialized for text-to-image without unified image-to-text/editing capabilities (§1, §5).
  - Retrieval-augmented models had been explored (RA-CM3; retrieval-augmented diffusion), but not fully scaled with the LLM-style two-stage training and broad instruction-tuning used here (§1, §2.1, §5).
- Positioning
  - CM3Leon scales a decoder-only, token-based architecture with retrieval during pretraining and multi-task supervised fine-tuning (SFT) across mixed image–text tasks, and introduces a decoding method that uses the model’s own conditional/unconditional variants for guidance (§1, §2, §3.1, §4).

## 3. Technical Approach
High-level pipeline: tokenize images and text into one sequence; augment each training example with retrieved image–text documents; train a single decoder to do next-token prediction on an infilling objective; then instruction-tune on diverse multi-modal tasks; finally, use improved decoding for generation.

Step-by-step:

1) Data and tokenization (§2.1; Appendix B.1)
- Only licensed images from Shutterstock are used to avoid ownership/attribution concerns.
- Image tokenization: a pretrained image tokenizer (from Gafni et al., 2022a) compresses a 256×256 image into 1024 discrete tokens from a codebook of size 8192.
- Text tokenization: a custom tokenizer with 56,320 vocabulary size.
- Special tokens:
  - `<break>` separates modalities inside a single sequence.
  - `<mask>` marks the span to infill.
  - `<infill>` indicates the start of the content that fills the masked span.
- Figures B.1–B.2 (Figure 8–9) visualize how captions and images interleave with `<break>` and how training samples with retrieved documents are concatenated.

2) Retrieval augmentation during pretraining (§2.1)
- Goal: expand the context with relevant and diverse multi-modal documents to reduce the burden on the generator and make training more compute-efficient.
- Retriever:
  - A CLIP-based bi-encoder produces embeddings for text and image parts of each memory document; the two embeddings are averaged to represent the document (§2.1; CLIP is an image–text model that maps images and captions into the same embedding space).
  - Relevance is scored by inner product; efficient search is done via Maximum Inner Product Search (MIPS).
- Diversity/relevance controls:
  - Skip candidates too similar to the query (they report keeping only those with “relevance score ≤ 0.9” and removing near-duplicates).
  - Query dropout: randomly drop 20% of query tokens to encourage diverse retrieval.
  - Modality balance: retrieve multi-modal documents (images + text) rather than only text or only image.
- Training usage:
  - They retrieve from both image- and text-based queries and include three retrieved samples per training example; combined with the original pair, this yields roughly 4× more tokens per instance (§2.1).

3) Objective: CM3 in-filling for mixed modalities (§2.2)
- Idea in plain language: Turn any mixed image–text input into an in-filling task by masking a span and moving it to the end of the sequence; train with standard next-token prediction.
- Example:
  - Input: “Image of a chameleon: [image]”
  - Infilling form: “Image of <mask>: [image] <infill> a chameleon”
- Important detail: masking does not cross `<break>` tokens, preventing the model from starting an image in the middle of an existing image segment (§2.2).
- Why this objective?
  - It unifies text-to-image, image-to-text, and infilling/editing within a single next-token model. It also enables classifier-free guidance without extra finetuning (see decoding below).

4) Architecture and training setup (§2.3–§2.4; Table 3; Figure 3)
- Decoder-only Transformer, 4096 sequence length, no dropout/biases/learnable LayerNorm parameters; careful initialization (positional embeddings near zero).
- Model sizes:
  - 350M parameters (24 layers, d_model=1024), trained on 1.4T tokens.
  - 760M parameters (24 layers, d_model=1536), trained on 1.9T tokens.
  - 7B parameters (32 layers, d_model=4096), trained on 2.4T tokens.
- Validation perplexity curves (Figure 3) decrease steadily with more updates, suggesting training did not saturate.

5) Decoding strategies for image generation (§3.1; Eqs. 1–3; Figure 4)
- Background terms:
  - `temperature` sampling controls randomness by flattening/sharpening the probability distribution.
  - `top-p` (nucleus) sampling draws from the smallest set of tokens whose cumulative probability exceeds a threshold.
- Classifier-free guidance (CFG) inside one model (Eqs. 1–2):
  - They form two streams during generation:
    - Conditional: use the actual text prompt.
    - Unconditional: replace the text with the `<mask>` token (enabled by the CM3 objective).
  - Combine logits as `logits_uncond + α · (logits_cond − logits_uncond)`, where `α` controls guidance strength.
- Contrastive Decoding with Top‑K set (CD‑K) (§3.1; Eq. 3):
  - Contrastive decoding scores each token by how much more the “strong” model likes it than the “weak” model.
  - Here, “strong” is the conditional model (with text prompt) and “weak” is the unconditional model (with `<mask>`).
  - Token candidates are restricted to those whose conditional probability is at least `α` times the k‑th largest probability (Eq. 3), which is less brittle than using only the single maximum probability threshold.
  - Practical effect: CD‑K produces generations complementary to CFG; combining them allows further FID improvement as you increase samples (Figure 4, right).

6) Supervised instruction tuning (SFT) across tasks (§4; Figure 5; Table 5; Appendix E)
- After pretraining, CM3Leon is fine-tuned on diverse, instruction-style tasks that interleave text and images, e.g.:
  - Text-guided image editing (InstructPix2Pix-like; ~600k examples; §4.1).
  - Image-to-image grounded generation via structural features (edge maps, pose, segmentation, depth) extracted with ControlNet-style preprocessing; ~7M examples (§4.1).
  - Spatially grounded generation with object tokens and bounding boxes; ~3M examples (§4.1).
  - “How-to-write” for rendering text in images; ~200k examples (§4.1).
  - Image captioning, visual question answering (VQA), long-caption paragraphing, and multi-step reasoning prompts with images (§4.2).
- All tasks are formatted as single sequences with a task prefix (e.g., “Edit the image following the text instruction”) and trained with the same infilling objective (Figure 5).
- SFT hyperparameters and scale: about 30B tokens processed; see Table 4 and §E.1.

7) Inference efficiency (§C; Figures 10–11)
- For 256×256 images with the 7B model: 11.8 seconds in BF16 and 9.1 seconds in INT8 (Figure 10).
- Throughput trade-offs with model parallelism and precision are reported in Figure 11.

## 4. Key Insights and Innovations
- Retrieval-augmented pretraining at scale for multi-modal autoregressive models
  - Distinctive aspect: retrieval of multi-modal documents (image+text) from a large licensed corpus (Shutterstock) is integrated throughout pretraining (§2.1).
  - Significance: training is more compute-efficient (Figure 2 shows better FID vs GPU hours than DALLE, Stable Diffusion, and Parti).
  - Design twist: unlike RA‑CM3, they remove the loss up-weighting of the query pair to preserve zero-shot performance without retrieval (§2.2).

- A single infilling objective enabling both modalities and enabling CFG without extra finetuning
  - Distinctive aspect: the CM3 infilling objective with `<mask>`/`<infill>` and `<break>` allows text-to-image, image-to-text, and editing within one decoder (§2.2).
  - Significance: the same model provides the conditional and unconditional streams required for classifier-free guidance (Eqs. 1–2), avoiding the need for separate conditioning tricks (§3.1).

- Contrastive Decoding Top‑K (CD‑K)
  - Distinctive aspect: reinterprets contrastive decoding using the model’s own conditional/unconditional distributions and relaxes the candidate set with a k‑th maximum probability threshold (Eq. 3), avoiding degeneration into greedy decoding (§3.1).
  - Significance: yields generations complementary to CFG; mixing Top‑P and CD‑K keeps reducing FID as you increase the number of samples (Figure 4, right).

- Broad instruction tuning with interleaved image–text tasks
  - Distinctive aspect: multi-task SFT that covers editing, conditional synthesis from structural controls, spatially grounded generation, OCR-like text rendering, captioning, VQA, and long-form descriptions (Figure 5, Table 5).
  - Significance: enables unprecedented controllability for an autoregressive multi-modal model (qualitative results in Figures 6, 15, 16).

## 5. Experimental Analysis
- Evaluation setup
  - Primary text-to-image benchmark: MS‑COCO 30K zero-shot, metric FID (lower is better). FID (Fréchet Inception Distance) measures how close the distribution of generated images is to real images using Inception features.
  - Decoding study: CFG weights and sampling strategies vs FID (Figure 4).
  - Comparisons include diffusion (Stable Diffusion, RE‑Imagen), autoregressive (DALLE, Parti, MUSE), and retrieval-augmented variants (KNN‑Diffusion, RA‑CM3) (§3.2; Table 1).
  - Vision–language tasks after SFT: COCO CIDEr, VQA2 accuracy, VizWiz accuracy, OKVQA accuracy, Image Paragraph CIDEr, VisDial NDCG (Table 2).

- Main quantitative results
  - State-of-the-art FID with less compute:
    - > “CM3Leon‑7B ✓ retrieval ✓ responsible data, with 2 retrieved documents at inference: Zero-shot FID‑30K = 4.88” (Table 1).
    - With 0 retrievals: FID 10.82; with 1 retrieval: FID 5.78; with 2 retrievals: FID 4.88 (Table 1). This isolates the benefit of retrieval at inference.
    - Figure 2 plots FID (log scale) vs equivalent A100 GPU hours: CM3Leon’s curve sits below DALLE, Stable Diffusion, and Parti, indicating better quality for a given compute budget.
  - Decoding ablations (Figure 4):
    - Optimal CFG weight is consistent across model sizes (left panel).
    - Top‑P and CD‑K achieve similar FID individually, but combining them (half Top‑P + half CD‑K) continues to reduce FID as the number of samples per prompt increases (right panel).
  - Instruction tuning outcomes (Table 2):
    - COCO CIDEr (0-shot): 61.6 (SFT‑CM3Leon‑7B).
    - VQA2 accuracy (0-shot): 47.6.
    - VizWiz accuracy (0-shot): 37.6, exceeding Flamingo‑9B’s 28.8.
    - OKVQA accuracy (0-shot): 23.8 (below Flamingo‑9B’s 44.7).
    - Image Paragraph CIDEr: 10.5; VisDial NDCG: 22.6.
    - Notably, these are achieved after training on only ≈3B text tokens vs 40–100B in OpenFlamingo/Flamingo (§4.2; Table 2).
  - Inference efficiency (Figure 10):
    - 7B model: 11.8 s (BF16) and 9.1 s (INT8) for 256×256; this is slower than MUSE 3B at 0.5 s but comparable to some diffusion settings.

- Convincingness of evidence
  - Text-to-image quality is strongly supported by FID comparisons and scaling curves (Table 1, Figure 2) and by decoding ablations (Figure 4).
  - The retrieval-at-inference ablation (0 vs 1 vs 2 retrieved docs) clearly quantifies retrieval’s impact (Table 1).
  - Multi-task capabilities are shown with both quantitative (Table 2) and qualitative examples (Figures 6, 15, 16). Results are mixed across vision–language tasks: strong on VizWiz, weaker on knowledge-heavy OKVQA, consistent with the relatively small text-only corpus (≈3B tokens).

- Notable observations/nuances
  - The model uses CLIP-based re-ranking to select the best image among 8 samples during FID evaluation (Table 1 caption notes re-ranking via CLIP for their models), which is a common practice but adds a learned selection step on top of raw sampling.
  - Removing RA‑CM3’s loss up-weighting improves zero-shot behavior without retrieval (§2.2), aligning training more closely with inference.

## 6. Limitations and Trade-offs
- Dependency on retrieval
  - Quality drops without retrieval at inference (FID 10.82 with 0 retrievals vs 4.88 with 2; Table 1). If the memory bank is missing rare concepts or stylistic niches, generation quality may suffer.
- Data scope and licensing
  - Only Shutterstock-licensed data are used (§2.1). This strengthens ethical sourcing but may limit coverage of tail entities compared to web-scale crawls, which could explain weaker performance on knowledge-heavy tasks (Table 2 OKVQA).
- Resolution and latency
  - Results and latency are reported for 256×256 image tokenization. Inference at this resolution takes 9–12 seconds on the 7B model (Figure 10); this is slower than some non-autoregressive models (e.g., MUSE 3B at 0.5 s).
- Compute and scale
  - While more compute-efficient than prior AR models at the same quality (Figure 2), pretraining still consumes up to 2.4T tokens and large GPU fleets (Table 3).
- Metrics and evaluation breadth
  - FID is a distributional metric and may not fully capture compositional controllability; the paper reports extensive qualitative examples (Figures 6, 12–16), but systematic evaluations of editing fidelity, spatial grounding accuracy, or text rendering robustness are limited.
- Decoding complexity
  - Best results rely on mixing strategies (Top‑P + CD‑K) and CLIP-based re-ranking (Table 1), which adds engineering complexity and inference-time cost proportional to the number of samples per prompt (Figure 4, right).

## 7. Implications and Future Directions
- Field impact
  - Demonstrates that the LLM recipe—retrieval-augmented pretraining followed by instruction tuning—transfers effectively to multi-modal generation, suggesting a viable alternative (or complement) to diffusion-first pipelines (§1, §4; Figure 2).
  - Shows that a single decoder can read and write both modalities with fine-grained control, potentially simplifying product stacks for image editing, controllable generation, and captioning (Figure 5; §4.1–§4.2).
- Follow-up research enabled or suggested
  - Higher-resolution tokenizers and hierarchical decoding to reduce latency while scaling image quality beyond 256×256.
  - Retrieval store design: dynamic, domain-specific memory banks; learned retrieval policies; retrieval for safety/attribution metadata.
  - Stronger instruction tuning with more text-heavy and knowledge-intensive corpora to close gaps on OKVQA-like tasks (Table 2).
  - Formal evaluations of controllability: quantitative metrics for editing consistency, spatial accuracy, and text rendering fidelity, complementing the qualitative showcases (Figures 6, 15).
  - Extending the same framework to video (tokenized spatio-temporal streams) and to 3D or multimodal dialogue settings.
- Practical applications
  - Content creation: text-guided editing, pose/edge/segmentation-conditioned generation for design workflows (§4.1; Figure 6).
  - Accessibility: improved zero-shot performance on non-ideal images (e.g., VizWiz; Table 2) hints at utility for assistive technologies.
  - Visual reasoning assistants: combined captioning, VQA, and long-form narration in one model (Figure 7, Figure 16), with potential for grounded assistants that can both generate and understand images.

Block-quoted key results and where to find them:
- “CM3Leon‑7B … 2 retrieved documents … Zero-shot FID‑30K = 4.88” — Table 1 (§3.2).
- “FID vs compute shows CM3Leon scaling better than DALLE, Stable Diffusion, PARTI” — Figure 2.
- “Validation perplexity decreases steadily; training not saturated” — Figure 3.
- “Top‑P and CD‑K are complementary; mixing them keeps reducing FID with more samples” — Figure 4 (right).
- “Optimal CFG weight stable across sizes” — Figure 4 (left).
- “SFT zero-shot: VizWiz 37.6 (beats Flamingo‑9B’s 28.8), COCO CIDEr 61.6, VQA2 47.6” — Table 2.
- “Latency: 7B BF16 11.8s; INT8 9.1s at 256×256” — Figure 10.
