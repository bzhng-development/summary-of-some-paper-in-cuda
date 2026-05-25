# Next Token Prediction Towards Multimodal Intelligence: A Comprehensive Survey

**ArXiv:** [2412.18619](https://arxiv.org/abs/2412.18619)

## 🎯 Pitch

This survey introduces and systematically analyzes the 'next token prediction' (NTP) paradigm as a universal training objective that unifies multimodal understanding and generation—spanning text, images, audio, and video—within a single modeling framework. By providing a comprehensive taxonomy across tokenization strategies, model architectures, unified task formats, datasets, evaluation, and open challenges, the paper demonstrates how NTP enables scalable, end-to-end multimodal AI, bridging the gap between traditionally siloed approaches. This matters because it paves the way for more seamless, efficient, and integrated AI models—opening the door to assistants that can reason about and generate content across all modalities using shared infrastructures and learning principles.

---

## 1. Executive Summary (2–3 sentences)
This survey systematizes “multimodal next-token prediction” (MMNTP) as a unifying paradigm to train and operate models that understand and generate text, images, audio, and video using the same objective: predict the next token in a sequence. It proposes a complete taxonomy (tokenization, model architectures, unified task formats, datasets/evaluation, open challenges) and shows with concrete model families, datasets, and results that NTP can cover both multimodal understanding and generation, clarifying design choices that make this work across modalities (Fig. 2; Table 1).

## 2. Context and Motivation
- Problem addressed
  - Modern multimodal AI is fragmented: vision-language understanding often uses contrastive or cross-attention encoders (e.g., CLIP), while generation relies on diffusion/autoencoding pipelines; audio has its own codecs and objectives. A single learning objective that supports both understanding and generation across modalities has been missing.
  - This work frames multimodal learning as Next Token Prediction (NTP), i.e., transform multimodal data into tokens and learn to predict the next one given context (Abstract; Fig. 2).

- Why this matters
  - Practical: A unified objective simplifies training and deployment, allows reuse of LLM infrastructure (tooling, scaling, inference acceleration), and opens end-to-end multimodal assistants that both perceive and create content (Sec. 1; Fig. 1 timeline).
  - Scientific: It connects modalities that live in continuous spaces (images, audio) with language-style sequence modeling via tokenization, enabling a common probabilistic modeling lens (Sec. 2).

- Prior approaches and shortcomings
  - Vision-language understanding: contrastive pretraining (e.g., CLIP [326]) aligned images with text but did not generate images, and often used separate encoders/heads (Sec. 2.1.3; Table 2).
  - Generation: diffusion models and VAE variants produce high-quality media but use different training objectives and are awkward to integrate with LLMs (Sec. 2.2; 3.3.2).
  - Early multimodal LLMs took a compositional approach (external encoders/decoders glued to an LLM) rather than a single unified backbone (Sec. 3.2 vs. 3.3).

- How this survey positions itself
  - It articulates a unified taxonomy for MMNTP with three core components—tokenization (discrete vs. continuous), backbone modeling (compositional vs. unified; attention masks), and training objectives (discrete-token prediction vs. continuous-token prediction)—and maps existing models onto this space (Fig. 2; Table 3; Fig. 8–10).
  - It also standardizes task formatting (alignment pretraining, instruction and preference finetuning, prompt-time ICL/CoT) and compiles datasets/benchmarks to evaluate progress and gaps (Sec. 4–5; Table 5–7; Fig. 17–18).

## 3. Technical Approach
The “approach” here is a framework—how to turn heterogeneous media into sequences that a transformer can model with NTP, then put them back into media.

Step 1 — Tokenize multimodal inputs into sequences (Sec. 2; Fig. 4–7; Table 2)
- Discrete tokenization (quantization)
  - Goal: map continuous inputs (pixels, waveforms) into a finite vocabulary (like text tokens) via a learned codebook.
  - How it works
    - Encode input to a latent `Z` with an autoencoder encoder `E`.
    - Quantize each latent vector to its nearest codebook entry (`argmin ||z − c_i||`; Fig. 6).
    - Decode quantized `Ẑ` back to the input space with decoder `D`.
    - Train with reconstruction loss + codebook/commitment losses; gradients bypass the non-differentiable quantizer via a straight-through estimator (Sec. 2.2.1).
  - Variants (Sec. 2.2.1)
    - `RVQ` (Residual VQ): quantize in coarse-to-fine stages, improving precision and reducing compute (RQVAE/RQ-Transformer [217]).
    - `Group VQ`: split information by dimensions (HiFiCodec [445]; FACodec [190]).
    - `Multi-scale VQ`: encode multiple scales (VAR [379]) enabling “next-scale” rather than raster token order.
    - `FSQ` (finite scalar quantization) and `LFQ` (lookup-free quantization): enlarge vocabularies while mitigating codebook collapse; LFQ improved ImageNet reconstruction rFID from 2.5 to 1.4 by increasing vocab from 2^10 to 2^16 (Sec. 2.2.1).
    - Auxiliary losses (e.g., perceptual/adversarial in VQGAN [112]) to improve perceptual quality (reduce blur).

- Continuous tokenization (no quantization)
  - Goal: keep rich continuous features and align them with the LLM’s embedding space.
  - How it works
    - Encode raw media with modality encoders (e.g., ViT/CLIP for images, Whisper/WavLM/HuBERT for audio) to get continuous features (Sec. 2.4.1; Table 2).
    - Align features to LLM with either:
      - Slot-based resampler (e.g., `Q-Former` in BLIP-2 [228]; `Perceiver Resampler` in Flamingo [3]) that compresses many patch tokens to a few learned queries via cross-attention; or
      - Simple projection (linear/MLP) into the LLM embedding space (LLaVA [255], Fuyu [18]) (Sec. 2.4.1; Fig. 7).

- Output tokenization (for generation from an LLM)
  - Discrete outputs: just predict vocabulary IDs (merged text + visual/audio vocabulary) and detokenize via VQ decoders (Sec. 2.4.2).
  - Continuous outputs: regress dense features as conditions or latents for decoders (e.g., diffusion or neural codec); position them in the sequence with placeholders or BOS/EOS markers; and add an “output alignment” head if needed (Sec. 2.4.2).

Step 2 — Model sequences with a multimodal transformer (Sec. 3; Fig. 8–10)
- Two design families (Fig. 8; Table 3)
  - `Compositional`: powerful external encoders/decoders (e.g., EVA-CLIP + Stable Diffusion) connected to an LLM by small aligners; great when you want to reuse best-in-class components (Sec. 3.2).
  - `Unified`: minimal (or lightweight) encoders/decoders; most understanding and generation happens in a single AR transformer. Often uses discrete tokens via VQ and merges vocabularies across modalities (Sec. 3.3.1); newer work also trains AR diffusion jointly (Sec. 3.3.2).

- Attention masks (Fig. 10)
  - `Causal`: standard autoregressive attention for generation.
  - `Prefix/Non-causal on the prefix`: bidirectional attention allowed over conditioning input (e.g., an image or document) but causal over the generated segment—useful for summarization or VQA with rich contexts.
  - `Semi-causal`: a token may attend to all past + a portion of future (visual-specific accelerations, e.g., VAR [379]).

- Task templates as sequences (Fig. 11–12)
  - Vision: VQA (`image + question → answer tokens`), text-to-image (`text → image tokens`), image editing (`image + instruction → edited image tokens`). Depending on generation style: autoregressive discrete tokens (LlamaGen [361], VAR [379]) or continuous latents to diffusion (MAR [235], Transfusion [513]).
  - Audio: understanding via encoder + adapter; generation via discrete codec tokens (e.g., Encodec [104]) or continuous tokens (MELLE [291]); full-duplex streaming dialogues model simultaneous input/output audio streams (Moshi [78]) (Fig. 12).

Step 3 — Train with unified objectives and stages (Sec. 4; Fig. 13–16)
- Core objective (Eq. 2): predict the next token given the previous tokens; the loss is cross-entropy for discrete, MSE-like for continuous (Sec. 4.1).
- Discrete-token prediction (`DTP`) vs. continuous-token prediction (`CTP`) (Fig. 13)
  - DTP scales like language modeling; supports mixed vocabularies (text + VQ image/audio codes) and parallel prediction schemes (e.g., MaskGIT [44], multi-scale VAR [379]) for images/videos (Sec. 4.1.1).
  - CTP regresses dense features used directly as conditions/latents for decoders; e.g., Emu [367] uses an LLM head to regress diffusion conditions, trained jointly with diffusion during instruction tuning (Sec. 4.1.2).

- Training stages (Fig. 14)
  1) Modality alignment pretraining (Eq. 4–5): align image/video/audio with language on web-scale data—either to predict text given media or to predict media tokens given text.
  2) Instruction finetuning (Eq. 6–7): teach the model to follow multimodal instructions for both understanding (triplets of image, question, answer) and generation (text-to-media, edit-to-media).
  3) Preference alignment: mitigate hallucination and align with human preferences via RLHF or DPO variants (Eq. 8–11), including diffusion-specific DPO (DPO-Diffusion, Eq. 11) (Sec. 4.3.3–4.3.4).

- Inference-time prompt engineering (Sec. 4.4; Fig. 15–16; Table 4)
  - Multimodal in-context learning (ICL): provide few multimodal demonstrations; proven effective but often text-dominated, so careful example construction matters.
  - Multimodal Chain-of-Thought (CoT): elicit stepwise reasoning with visual/audio grounding to reduce hallucinations (e.g., VisualCoT [351], V* [422]).

## 4. Key Insights and Innovations
- A unified taxonomy that spans understanding and generation across modalities (Sec. 1.1; Fig. 2; Table 1)
  - What’s new: prior surveys emphasized understanding or single-modality generation; here both are placed under the same NTP lens with parallel design choices (discrete vs. continuous tokenization; compositional vs. unified backbones; DTP vs. CTP).
  - Why it matters: makes design trade-offs explicit and comparable across modalities and tasks.

- Clear architectural dichotomy: `compositional` vs. `unified` models (Sec. 3; Fig. 8; Table 3)
  - New framing: many recent models can be categorized by how much work is delegated to external encoders/decoders versus a single AR transformer.
  - Significance: clarifies when to leverage best-in-class modules (compositional) versus when to push end-to-end AR modeling (unified) for efficiency and scaling.

- Tokenization as the gateway to NTP for continuous media (Sec. 2; Fig. 4–7; Table 2)
  - Novel synthesis: detailed comparison of discrete methods (VQ/RVQ/FSQ/LFQ, multi-scale, product VQ) and continuous methods (encoders + aligners), including modality-specific tips (e.g., video 3D tokenizers; audio neural codecs).
  - Impact: helps practitioners pick tokenizers for their modality and task while understanding reconstruction vs. representation vs. token efficiency trade-offs (Sec. 2.3).

- Unified training curriculum (alignment → instruction → preference) and unified inference tooling (ICL/CoT) across modalities (Sec. 4; Fig. 14–16)
  - Contribution: translates the well-known LLM pipeline into multimodal settings with concrete loss formulations (Eq. 4–11) and pitfalls (e.g., hallucination).
  - Value: a blueprint to train MMNTP systems with aligned objectives and consistent evaluation.

- Identification of open challenges unique to MMNTP (Sec. 6)
  - Fundamental issues: scaling laws on unlabeled multimodal data, modality interference within a single AR backbone, long-sequence and heterogeneous system bottlenecks, and the boundary between NTP and diffusion for generation.

## 5. Experimental Analysis
Because this is a survey, the “experiments” are aggregated plots and curated comparisons rather than the authors running new models. The paper grounds claims using public results and its own synthesis figures.

- Evaluation methodology and resources
  - Datasets
    - Pretraining corpora for text (e.g., `C4`, `mC4`, `The Pile`, `Dolma`, `FineWeb`) and for vision/audio/video (e.g., `LAION-5B`, `COYO-700M`, `WebVid`, `ACAV100M`, `LibriLight`, `WavCaps`) with sizes and provenance summarized in Table 5.
    - Instruction tuning datasets spanning understanding and generation, including VQA-style, video-chat, OCR-rich, detection-augmented, and image-editing corpora (Table 6).
  - Benchmarks/metrics
    - Holistic: MME, MMBench, SEED-Bench(-2), MMMU, MVBench, VBench, CMMMU (Table 7).
    - Generation quality: ImageNet FID/rFID/gFID (Sec. 2.2.2; Fig. 18), GenEval for text-image alignment (Fig. 18).
    - Tokenizer evaluation: reconstruction (PSNR, rFID) vs. generation (IS, gFID) (Sec. 2.2.2).

- Main quantitative comparisons
  - Understanding (Fig. 17)
    - VQAv2: a steady accuracy climb from pre-2020 methods to LMMs such as LLaVA-1.5/1.6, Qwen-VL, InternVL, Emu2. The figure illustrates the lineage and gains achieved as models adopt an NTP framing and larger backbones.
    - MMMU (multi-discipline expert benchmark): both open-source (LLaVA-1.6, Qwen2-VL, InternVL2) and closed APIs (GPT-4V, Gemini 1.5, Claude 3.5, o1-preview) show rapid gains, indicating that NTP-style multimodal LLMs compete in advanced, cross-domain reasoning tasks.
  - Generation (Fig. 18)
    - ImageNet FID: the curve shows continued improvement from VQGAN+Transformer through MaskGIT, RQ-Transformer and up to VAR/MAGViT-v2 and modern AR models; NTP-based approaches are competitive with diffusion.
    - GenEval: NTP models (e.g., LlamaGen, Emu3, Chameleon, Show-o, Transfusion) show rising scores over time, with DALL·E 3 (diffusion-based) as a strong reference—underscoring that AR-NTP can deliver high alignment while quality gaps shrink.

- Specific, grounded observations
  - Tokenizer quality matters: in Sec. 2.2.1, LFQ’s ability to reduce reconstruction rFID from 2.5 to 1.4 by enlarging the vocabulary (2^10 → 2^16) on ImageNet highlights the practical effect of codebook design on generation (not just abstract theory).
  - Multi-scale and parallel prediction improve visual AR modeling: VAR’s next‑scale and MaskGIT’s parallel denoising-like decoding produce better sample quality and speed than strict raster scanning (Sec. 4.1.1).

- Are claims supported?
  - The synthesized plots (Fig. 17–18) and large model table (Table 3) credibly show that:
    - NTP-based LMMs now dominate multimodal understanding leaderboards.
    - AR-NTP generation is closing the gap with diffusion for images and is mainstream in audio via neural codecs (Table 3; Sec. 2.3.2, 3.3.1).
  - Caveat: the paper does not run controlled head-to-head experiments; it aggregates reported results. Differences in training data and evaluation setups mean causality (NTP vs. non-NTP) should be interpreted with care (Sec. 5.2).

- Ablations / failure cases / robustness
  - Tokenizers: reconstruction vs. representation trade-off and codebook collapse are discussed with mitigations (FSQ/LFQ, perceptual/adversarial losses) (Sec. 2.2.1–2.2.2).
  - Hallucination: preference alignment methods (RLHF/DPO) and CoT/visual grounding strategies are covered (Sec. 4.3.3–4.4.2), but systematic robustness audits remain an open need (Sec. 6.2; Table 7 includes HallusionBench, VQAv2‑IDK).

## 6. Limitations and Trade-offs
- Assumptions and dependencies
  - MMNTP relies on high-quality tokenization: discrete requires good codebooks/decoders; continuous requires strong encoders and careful alignment (Sec. 2).
  - Instruction and preference finetuning assume access to sizable, quality multimodal supervision and preference data (Sec. 4.3), which is uneven across domains (e.g., medical, non-English).

- Open scenarios not fully addressed
  - Ultra-long multimodal contexts (multi-image RAG, long video) strain sequence length and memory; current sequence parallelism is LLM-centric and not optimized for mixed-modality pipelines (Sec. 6.3).
  - Real-time, full-duplex audio+vision interactions are emerging (Moshi [78]), but unified latency-aware MMNTP training is nascent (Fig. 12c).

- Computational and data constraints
  - Heterogeneous pipelines cause “GPU bubbles” (idle time) in distributed training when encoders and LLMs are pipelined; specialized schedulers (Optimus, DistTrain) are needed, but production-scale training is still challenging (Sec. 6.3).
  - Video tokenization can explode sequence lengths unless temporally compressed 3D tokenizers are used; frame-by-frame tokenizers cause redundancy and temporal inconsistency (Sec. 2.3.3).

- Modalities interference and optimization stability
  - A single AR backbone that predicts both text and image/audio tokens can experience gradient/pathology issues and capability interference (e.g., language skill regressions), requiring techniques such as QK-Norm and careful curriculum (Sec. 6.2; [375], [496]).

- Unsettled modeling choices
  - AR-NTP vs. diffusion for generation is undecided; different works report opposite advantages depending on architecture and data (Sec. 3.3.2). Hybrid approaches (e.g., Transfusion [513]; MAR [235]) blur boundaries but complicate training.

## 7. Implications and Future Directions
- Field-level impact
  - MMNTP reframes multimodal AI around one objective, letting the community port decades of language-model know‑how (scaling, inference, alignment, prompting) to vision/audio/video (Sec. 3–4).
  - By making tokenization the bridge, it enables a common interface for agents that must both perceive and act/generate (Fig. 2; Fig. 11–12).

- Research directions suggested (Sec. 6)
  - Scaling laws for multimodal data: establish how data mixture, tokenization granularity, and model size jointly determine loss/performance across modalities; determine whether emergent abilities—well documented in LLMs—appear for multimodal tasks (Sec. 6.1).
  - Reducing modality interference: routing, parameter decoupling, or optimizer schedules that preserve each modality’s strengths while enabling synergy (Sec. 6.2).
  - Token efficiency and redundancy: principled token pruning/merging for continuous and discrete tokens; leverage findings that many visual tokens receive little attention (Sec. 6.3; [54]).
  - Systems co-design: bubble-free heterogeneous pipelines, unified sequence parallelism for mixed modality and long contexts, streaming training for duplex audio/vision (Sec. 6.3).
  - Universal interfaces beyond AV/VL: formulate robotics actions, GUIs, molecules, and proteins as next-token sequences with appropriate tokenizers (e.g., action tokens, graph tokens), and compare NTP with diffusion/planning hybrids in these domains (Sec. 6.4).

- Practical applications
  - Unified assistants that see, listen, speak, and draw: medical VQA and report generation (LLaVA‑Med), design tools (instruction-based image editing, Table 6), multimedia creation (Emu3/Chameleon/LlamaGen; Fig. 18), accessible captioning and TTS/ASR with a single backbone (VIOLA, LauraGPT; Table 3).
  - Multimodal agents and robotics: instruction-following in embodied settings using NTP-formatted perception and action tokens (Sec. 2.3.4; [32], [184]).

> Core takeaway (Fig. 2; Sec. 1): “The process can be divided into three key components: Tokenization, Modeling, and Training Objectives.” By making these components explicit and mapping today’s models and datasets onto them, the survey provides a playbook to build, train, and evaluate unified multimodal systems using next-token prediction.
