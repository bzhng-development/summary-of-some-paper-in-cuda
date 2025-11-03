# Next Token Prediction Towards Multimodal Intelligence: A Comprehensive Survey

**ArXiv:** [2412.18619](https://arxiv.org/abs/2412.18619)
**Authors:** Liang Chen, Zekun Wang, Shuhuai Ren, Lei Li, Haozhe Zhao, Yunshui Li, Zefan Cai, Hongcheng Guo, Lei Zhang, Yizhe Xiong, Yichi Zhang, Ruoyu Wu, Qingxiu Dong, Ge Zhang, Jian Yang, Lingwei Meng, Shujie Hu, Yulong Chen, Junyang Lin, Shuai Bai, Andreas Vlachos, Xu Tan, Minjia Zhang, Wen Xiao, Aaron Yee, Tianyu Liu, Baobao Chang
**Institutions:** Peking University, Beihang University, University of Hong Kong, Shenzhen Institute of Advanced Technology, Chinese Academy of Sciences, Tsinghua University, M‑A‑P, The Chinese University of Hong Kong, Alibaba Group, University of Cambridge, Microsoft Research, UIUC, Humanify Inc., Zhejiang University

## 🎯 Pitch

This survey revolutionizes multimodal learning by unifying the diverse modalities of images, audio, video, and text under a single next-token prediction (NTP) framework. By offering a unified approach, it simplifies engineering complexity and harnesses LLM advances, paving the way for integrated systems capable of seamless perception and generation, which could transform how we build universally intelligent agents.

---

## 1. Executive Summary (2-3 sentences)
This survey reframes multimodal learning—across images, audio, video, and text—under one training objective: next‑token prediction (NTP). It proposes a concrete taxonomy spanning tokenization, backbone architectures, unified task representation and training, datasets/evaluation, and open challenges (Fig. 2; Secs. 2–6), and shows how both understanding and generation can be implemented with the same autoregressive interface.

## 2. Context and Motivation
- Gap addressed
  - Multimodal research has splintered across architectures (e.g., diffusion for images, encoder–decoder for ASR, CLIP-style dual encoders for alignment) with incompatible training objectives and interfaces. There has been no unified, end‑to‑end view of “how to do multimodality” using the same modeling principle for both understanding and generation (Sec. 1; Fig. 1).
- Why it matters
  - Practical: One objective simplifies training, serving, and tool integration across modalities, enabling unified assistants that can see, hear, and generate (e.g., GPT‑4o, Moshi) (Sec. 1).
  - Scientific: NTP has well-studied scaling behavior in language; understanding whether and how those benefits carry over to multimodality could unlock progress akin to LLMs (Sec. 6.1).
- Prior approaches and their limits
  - Unimodal specialists: great performance but do not generalize across modalities/tasks (Sec. 1).
  - Cross‑modal encoders (e.g., CLIP) enable understanding but not generation directly (Sec. 2.1.3).
  - Diffusion and codecs enable generation but with different training paradigms from language (Secs. 2.2–2.4).
  - Fragmentation leads to engineering complexity, duplicated effort, and difficult transfer of improvements across modalities.
- Positioning
  - The survey unifies multimodal understanding and generation into the NTP lens and organizes the field around five pillars: multimodal tokenization (Sec. 2), MMNTP architectures (Sec. 3), unified task/training objectives (Sec. 4), datasets & evaluation (Sec. 5; Tables 5–7), and open challenges (Sec. 6). It also distinguishes two implementation families—`compositional` versus `unified` models (Fig. 8; Table 3)—and compares their trade‑offs.

## 3. Technical Approach
The survey’s “approach” is a systematization of how to build multimodal NTP systems end‑to‑end. The canonical pipeline is in Fig. 2 and Sec. 3.1:

1) Tokenize multimodal inputs (Sec. 2; Fig. 4)
- Goal: map raw data `x ∈ X` to representations `z ∈ Z_f` via a tokenizer `f` (Eq. 1).
- Two families:
  - `Discrete tokens` (quantized codes): use vector quantization (VQ) to map continuous features into a finite codebook (Sec. 2.2; Fig. 6).
    - Core mechanism (`VQVAE`, `VQGAN`): an encoder produces latents; a quantizer replaces each latent with its nearest codebook vector; a decoder reconstructs the input. Training uses reconstruction loss + codebook loss + commitment loss with a straight‑through estimator for gradients (Sec. 2.2.1).
    - Variants to improve fidelity/efficiency:
      - `Residual VQ (RVQ)` stacks quantizers to refine residuals (Sec. 2.2.1).
      - `Product Quantization (PQ)` factors large codebooks into smaller ones (Sec. 2.2.1).
      - `Multi‑scale VQ / VAR`: encode low‑to‑high scales; generate “next‑scale” rather than raster tokens (Sec. 2.2.1; Visual AR Modeling).
      - `FSQ` and `LFQ`: scalar and lookup‑free quantization to avoid codebook collapse (Sec. 2.2.1). Notably, LFQ reports rFID improvements on ImageNet when vocabulary grows from 2^10 to 2^16 (rFID ~2.5 → ~1.4) (Sec. 2.2.1).
  - `Continuous tokens` (dense features): keep features continuous and align them to the LLM’s embedding space (Sec. 2.4; Fig. 7).
    - Input alignment: transform image/audio/video embeddings into a compact set of LLM‑consumable vectors via either
      - `slot-based resamplers` (e.g., `Q‑Former`, `Perceiver` cross‑attention) that compress many patches/frames into K “slots” (Sec. 2.4.1), or
      - `projection` (linear/MLP) of encoder features to the LLM’s token space (Sec. 2.4.1).
    - Output alignment for generation (Sec. 2.4.2): LLM outputs continuous features that must be shaped into the conditioning interface of external decoders (e.g., Stable Diffusion) via a regression head or Q‑Former‑like module. “Positioning” the non‑text output in the sequence uses either placeholder tokens (e.g., `[IMG1] … [IMGr]`) or a BOS marker like `<dream>` (Sec. 2.4.2).

2) Autoregressively model the joint sequence with a transformer (Sec. 3.1; Fig. 9)
- Concatenate text + multimodal tokens (discrete or continuous). Predict the next token conditioned on the prefix.
- Attention masks (Fig. 10): causal for generation; prefix‑bidirectional for rich conditioning; encoder–decoder also supported. Masks can be set per modality/prefix to match task structure (Sec. 3.1).

3) Decode back to target modality
- Discrete case: pass predicted codes to the corresponding VQ decoder (Sec. 3.1).
- Continuous case: pass LLM’s predicted features to a modality‑specific decoder (e.g., diffusion, VAE latent decoder) (Sec. 2.4.2).

Architectural choices (Sec. 3; Fig. 8; Table 3)
- `Compositional models`: keep strong external encoders/decoders and connect them via small aligners (e.g., `CLIP` + MLP for understanding; LLM + regression head → Stable Diffusion for generation). Examples: LLaVA, Emu1/2, MiniGPT‑4, Qwen‑VL; audio: SALMONN, Qwen‑Audio, LLaMA‑Omni (Secs. 3.2.1–3.2.2).
- `Unified models`: lightweight encoders/decoders; most understanding and generation learned within the same autoregressive backbone. Two subtypes:
  - `Quantization-based AR`: generate discrete codes for images/audio/video, sometimes with parallel or multi‑scale decoding (Unified‑IO, Chameleon, Emu3, Moshi; Sec. 3.3.1).
  - `AR‑Diffusion hybrids`: train the diffusion process jointly under NTP to improve quality while maintaining a unified interface (Transfusion, MAR, CosyVoice; Sec. 3.3.2).

Unified training formulation (Sec. 4; Fig. 13–14)
- Generic next‑token loss (Eq. 2): predict `x_i` given `x_{1..i-1}`; the target `y_i` can be a discrete distribution (`cross‑entropy`) or a continuous vector (`MSE`).
- Two targets (Eq. 3):
  - `Discrete‑token prediction (DTP)`: generate textual tokens and/or VQ codes; enables pure understanding (text‑only outputs) and generation (multimodal tokens) (Sec. 4.1.1).
  - `Continuous‑token prediction (CTP)`: regress dense features for decoders, either as conditions (e.g., SD latents) or as direct outputs in latent space (Sec. 4.1.2).
- Training stages (Fig. 14):
  - `Pretraining / modality alignment`: learn to condition text on visuals or vice versa (Eqs. 4–5; Sec. 4.2). Works for images (LAION‑5B), video (WebVid), and audio (Clotho), including interleaved image–text corpora (MMC4/OBELICS).
  - `Instruction fine‑tuning`: supervised pairs `(I, Q, A)` for understanding (Eq. 6) and generation pairs `(Q, S)` for images/audio/videos (Eq. 7) (Sec. 4.3.1–4.3.2). Table 6 aggregates large instruction corpora (e.g., LLaVA, M3IT, SVIT, MIMIC‑IT).
  - `Preference alignment`: RLHF or `DPO` for understanding (Eq. 8; LLaVA‑RLHF, RLHF‑V, Silkie) and nascent DPO variants for diffusion models (Eqs. 9–11; DPO‑Diffusion, D3PO) to mitigate hallucinations and align outputs with human preferences (Sec. 4.3.3–4.3.4).
- Inference prompt engineering (Sec. 4.4; Table 4; Fig. 15–16):
  - `Multimodal ICL`: few interleaved demos to guide behavior.
  - `Multimodal CoT`: generate step‑by‑step rationales (e.g., Video/Audio CoT) to improve reasoning.

Modality specifics (Sec. 2.3, 2.5; Figs. 11–12)
- Images: balance `representation` (alignment with text), `reconstruction`, and `token efficiency` (Sec. 2.3.1). Support high‑resolution, arbitrary aspect ratios, and document layouts (Sec. 2.5.1).
- Audio: codecs (`SoundStream`, `Encodec`) + RVQ for high‑fidelity discrete tokens; semantic distillation for better language modeling (`SpeechTokenizer`, `Mimi`) (Sec. 2.3.2). Continuous mel features work for AR audio synthesis (MELLE; Sec. 2.5.2).
- Video: 3D tokenizers (MAGVIT‑v2, C‑ViViT) and causal temporal structures reduce redundancy and preserve temporal coherence (Sec. 2.3.3); or fuse frame features (Sec. 2.5.3).

## 4. Key Insights and Innovations
- A unifying taxonomy of multimodal NTP (fundamental)
  - The pipeline Tokenize → Autoregress → Decode (Fig. 2) with explicit choices at each step (discrete vs continuous tokens; compositional vs unified backbones; attention masks) gives a design map that had been missing. This framework systematically connects understanding and generation under one objective (Secs. 2–4).
- Clear articulation of tokenization trade‑offs (fundamental)
  - For images, three axes—`representation`, `reconstruction`, `token efficiency`—must be balanced (Sec. 2.3.1). The survey explains how design choices (e.g., Q‑Former vs projection; VQ vs LFQ/FSQ; multi‑scale codes) map to these axes and downstream cost/quality.
- Precise mechanism for continuous output generation (incremental but clarifying)
  - Two challenges are named and solved structurally: `positioning` (where non‑text outputs appear in the sequence) and `output alignment` (shaping LLM hidden states into decoder‑ready features) (Sec. 2.4.2). This crystallizes scattered practices in the literature.
- Systematic comparison of `compositional` vs `unified` models (fundamental)
  - Compositional models leverage strong pretrained components for faster progress; unified models promise better scaling and deployment efficiency but face stability and quality hurdles (Sec. 3.4). The survey grounds this with examples in Table 3 and details like QK‑Norm to stabilize mixed‑modality training (Sec. 6.2).
- Honest synthesis on AR vs diffusion for generation (fundamental)
  - It explicitly notes there is no consensus: some AR quantization models (Emu3) beat diffusion baselines on quality, while AR‑diffusion hybrids improve consistency (Sec. 3.3.2). The survey refrains from one‑size‑fits‑all claims.

## 5. Experimental Analysis
Evaluation design in a survey context (Sec. 5)
- Datasets
  - Pretraining: Tables 5 summarizes text (e.g., `C4`, `mC4`, `Pile`, `FineWeb`, `Dolma`), image (e.g., `LAION‑5B`, `COYO‑700M`, `MMC4`, `OBELICS`), video (`WebVid`, `InternVid`), audio (`LibriLight`, `AudioSet`, `WavCaps`, `Yodas`) (Sec. 5.1.1).
  - Instruction tuning: Table 6 covers understanding (LLaVA, MultiInstruct, SVIT, M3IT, etc.) and generation‑oriented editing sets (e.g., `InstructPix2Pix`, `HIVE`, `HQ‑Edit`, `UltraEdit`) (Sec. 5.1.2).
- Benchmarks and metrics
  - Holistic: `MME`, `MMBench`, `SEED‑Bench`, `MMMU`, `SEED‑Bench‑2`, `MVBench`, `VBench`, `CMMMU` (Table 7).
  - Specialty/emergent: `SparklesEval` (multi‑image dialog), `MathVista/Math‑Vision` (visual math), `HallusionBench` and `VQAv2‑IDK` (hallucination and uncertainty), `MMC‑Benchmark` (charts), `TempCompass` (temporal video understanding), `MMEvalPro` (reasoning and calibration) (Table 7).
- Main quantitative takeaways (Figs. 17–18)
  - Visual understanding: accuracy on `VQAv2` trends from ~55% in early systems to >80–85% in recent open‑source NTP‑style LMMs (Fig. 17, left). On `MMMU` (college‑level, 183 subfields), both open‑source and frontier APIs show steady gains from roughly the 20–30% range to ≈60–70%+ for top models (Fig. 17, right).
  - Image generation: FID on ImageNet steadily improves across AR families (e.g., VQGAN+Transformer → RQ‑Transformer → VAR, MaskBit, Infinity), with best open models around FID ~1–2 (Fig. 18, left). On `GenEval` (object‑centric alignment), unified AR systems (e.g., LlamaGen, Emu3, Chameleon, Show‑o) reach higher composite scores than earlier diffusion‑only pipelines, and close to strong commercial systems (Fig. 18, right).
  - Tokenizer evidence: `LFQ` demonstrates measurable reconstruction gains as vocabulary increases (rFID ↓ from ~2.5 to ~1.4 on ImageNet) (Sec. 2.2.1).
- Do experiments support the claims?
  - The survey’s cross‑model plots suggest that NTP‑style models are competitive or superior on both understanding and generation benchmarks (Secs. 5.2; Figs. 17–18), while also acknowledging that AR vs diffusion quality is task‑ and design‑dependent (Sec. 3.3.2).
- Robustness, failures, ablations
  - Hallucination remains a core failure mode; specialized datasets (`HallusionBench`, `VQAv2‑IDK`) and alignment methods (RLHF/DPO) are being adapted to multimodality (Secs. 4.3.3, 5.2.2).
  - For video understanding, `TempCompass` shows significant gaps in temporal reasoning (Sec. 5.2.2).
  - Prompting helps: multimodal ICL and CoT improve performance in some settings (Sec. 4.4; Table 4), but contributions can be “text‑driven,” with visuals under‑utilized in ICL unless carefully designed.

> “It is not concluded yet which modeling method has superior performance [between quantization‑based AR and diffusion‑based AR].” (Sec. 3.3.2)

## 6. Limitations and Trade-offs
- Tokenization choices
  - `Discrete` vs `continuous`:
    - Discrete codes ease NTP but may lose fidelity (quantization error) or create long raster sequences; continuous features preserve nuances but need `positioning` and `output alignment` modules (Secs. 2.2, 2.4).
  - Image tokenizers must trade `representation` vs `reconstruction` vs `token efficiency` (Sec. 2.3.1).
- Architecture
  - `Compositional` models rely on external encoders/decoders; faster to build but harder to end‑to‑end scale; potential mismatch between components (Sec. 3.4).
  - `Unified` models centralize learning but can be less sample‑efficient initially and face optimization instabilities (e.g., gradient norm issues across modalities; mitigated by QK‑Norm) (Secs. 3.4, 6.2).
- Training objectives
  - Joint prediction of text and multimodal codes can cause interference; text generation does not always help multimodal token prediction, and vice versa (Sec. 4.1.1; modality interference in Sec. 6.2).
  - Preference alignment for generation is in its infancy; diffusion DPO is promising but computationally heavy (Sec. 4.3.4).
- Systems and scalability
  - Heterogeneous modules and variable sequence lengths cause pipeline bubbles and poor hardware utilization; specialized schedulers (Optimus) and disaggregated training (DistTrain) are needed (Sec. 6.3).
  - Long‑context multimodal sequences (multi‑image/video) need sequence parallelism adapted to heterogeneous tokens (Sec. 6.3).
- Generalization scope
  - Scaling laws for multimodal NTP are not established; it is unclear how unlabeled interleaved data should be leveraged relative to paired data to maximize emergent abilities (Sec. 6.1).
  - Some modalities (e.g., 3D, actions/robotics) are at early stages of NTP formulation, with open questions about efficient tokenization and training objectives (Sec. 2.3.4; Sec. 6.4).

## 7. Implications and Future Directions
- Field impact
  - Treating multimodality as NTP gives a common interface to plug in new modalities and tasks, much as language modeling unified NLP. It clarifies how to combine perception and generation in one backbone and makes LLM advances (e.g., attention/kernel optimization, inference serving) directly transferable to multimodal systems (Secs. 1, 3.4).
- Research directions
  - `Scaling laws & unlabeled data`: establish compute‑optimal regimes and recipes for interleaved, weakly labeled, or modality‑only corpora (Sec. 6.1).
  - `Interference → synergy`: curriculum, masking, or routing strategies to reduce negative transfer between modalities; theoretical understanding of mixed‑modality optimization (Sec. 6.2).
  - `Efficient tokenization`: fewer, more expressive tokens (LFQ/FSQ, multi‑scale, learned pruning) that preserve semantics while minimizing context length (Secs. 2.2.1, 6.3).
  - `AR vs diffusion hybrids`: principled integration of next‑token prediction and denoising processes to combine quality and interface uniformity (Sec. 3.3.2).
  - `Preference alignment for generation`: scalable, modality‑aware DPO/RLHF (Eqs. 9–11) and better reward/detector models against hallucination (Secs. 4.3.4, 5.2.2).
  - `Systems`: bubble‑aware pipelines, unified sequence parallelism for heterogeneous tokens, and memory‑efficient KV‑caching across long multimodal contexts (Sec. 6.3).
  - `Universal interfaces beyond media`: formulate robotics, molecular design, and protein engineering as NTP with modality‑appropriate tokenization and objectives (Sec. 6.4).
- Practical applications
  - Unified assistants that can: answer visual/audio/video queries; edit/generate images and audio; conduct full‑duplex speech dialog (Fig. 12c); handle documents (Sec. 2.5.1) and charts (MMC‑Benchmark in Table 7); and perform instruction‑following or tool‑use across modalities (Tables 3–4).

Key references to locate details inside the paper:
- Pipeline overview: Fig. 2; Secs. 2–4
- Tokenizers: Table 2; Secs. 2.2–2.5; Figs. 4–7
- Model families and examples: Table 3; Fig. 8–12; Sec. 3
- Training objectives and stages: Sec. 4; Eqs. (2)–(7); Figs. 13–16
- Datasets and benchmarks: Tables 5–7; Sec. 5
- Results trends: Figs. 17–18
- Open challenges: Sec. 6

> “The process can be divided into three key components: Tokenization, Modeling, and Training Objectives.” (Sec. 1; Fig. 2)

> “LFQ… allows VQVAE to improve the quality of image reconstruction and generation by vastly increasing the vocabulary size… rFID … decreases from 2.5 to 1.4 when the LFQ vocabulary size increases from 2^10 to 2^16 on ImageNet.” (Sec. 2.2.1)

> “Unifying understanding and generation presents unique challenges, including the diversity of modalities and resolving conflicts between them.” (Sec. 1; expanded in Sec. 6.2)

> “It is not concluded yet which modeling method has superior performance [AR quantization vs AR‑diffusion].” (Sec. 3.3.2)
