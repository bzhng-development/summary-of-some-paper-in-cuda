# LLaVA-OneVision-1.5: Fully Open Framework for Democratized Multimodal Training

**ArXiv:** [2509.23661](https://arxiv.org/abs/2509.23661)

## 🎯 Pitch

LLaVA-OneVision-1.5 introduces a fully transparent, cost-efficient framework—including novel large-scale concept-balanced datasets and an end-to-end training recipe—for building state-of-the-art large multimodal models from scratch. By publicly releasing all assets and demonstrating competitive results on a modest budget, it lowers barriers to advanced vision-language research and empowers the community to innovate and specialize robust multimodal AI systems for diverse applications.

---

## 1. Executive Summary
LLaVA-OneVision-1.5 is a fully open training recipe, dataset suite, and model family for large multimodal models (LMMs) that see images and read text. It tackles the long-standing barrier that top multimodal systems are either closed or expensive to reproduce by showing how to reach state-of-the-art results with a transparent pipeline, concept-balanced data, and cost-efficient training.

## 2. Context and Motivation
- Problem addressed
  - High-performing vision–language models are typically proprietary, and even the open ones often require huge, opaque data mixtures and heavy compute. This makes it hard for researchers and practitioners to build strong models “from scratch,” learn what matters in training, or adapt models to new domains.
- Why it matters
  - Practical impact: many applications (document understanding, chart analysis, OCR-heavy workflows, scientific/technical Q&A, general visual reasoning) depend on reliable vision–language systems.
  - Scientific impact: the community needs reproducible insights about what data, architectures, and training steps most affect performance.
- Prior approaches and gaps
  - Earlier open projects (e.g., LLaVA, LLaVA-Next, LLaVA-OneVision) published code and data, but their performance now lags newer systems (Context: Introduction, p.1–2).
  - Newer open efforts (Molmo, Open-Qwen2VL) improved transparency and efficiency, but the gap with top proprietary models remains, partly due to data scale/quality and training efficiency limits (Introduction, p.1–2).
- How this work positions itself
  - A complete, reproducible pipeline: curated large-scale mid-training and instruction-tuning datasets; an efficiency-first training framework with offline parallel data packing; and models trained end-to-end on a modest budget (Abstract; Sections 3–4).
  - Targeted capability focus: strong OCR, document, and chart understanding, plus general VQA and reasoning (Figures 1, 3; Table 1).

## 3. Technical Approach
This section explains, step-by-step, how the system is built and trained.

- System architecture (Figure 2)
  - High-level design: “ViT–MLP–LLM” stack.
    - `Vision encoder`: RICE-ViT, a region-aware image encoder.
    - `Projector`: a small two-layer MLP that maps vision features into the LLM token space. To reduce sequence length, spatially adjacent 2×2 patch features are first grouped and concatenated (Section 2.1).
    - `LLM`: Qwen3 acts as the language backbone for multimodal reasoning and generation (Section 2.1).
  - Resolution handling: The vision encoder uses `2D RoPE` (two-dimensional rotary positional encoding) so it can process images at native resolution without fixed-size tiling, helping preserve fine detail (Figure 2; Section 2.2).
  - Global context: the encoder’s `[CLS]` token is kept to retain a global semantic summary alongside local region tokens during alignment (Figure 2 caption).

- RICE-ViT vision encoder (Section 2.2)
  - Goal: stronger fine-grained understanding for OCR, grounding, and local reasoning than purely global contrastive encoders (e.g., CLIP/SigLIP).
  - Mechanism:
    - `Region-aware attention`: explicitly models local regions inside images.
    - `Cluster discrimination loss`: a single, unified loss encourages semantically similar regions to cluster while distinguishing different regions. This substitutes a collection of specialized losses used by some alternatives and is trained on 450M images with 2.4B candidate regions.
    - Native-resolution capability via 2D RoPE.
  - Why this design: instance-level contrastive learning often treats all other images as negatives and misses local semantics. RICE-ViT’s region-level learning captures OCR text boxes and object regions jointly, improving text-in-image understanding (Section 2.2).

- Training pipeline (Section 4.1)
  - Three stages:
    1) `Stage-1: Language–Image Alignment`  
       - Train only the projector (keeps the LLM largely fixed) using `LLaVA-1.5 558K` to quickly align vision features with the LLM token space.
    2) `Stage-1.5: High-Quality Knowledge Learning` (the “mid-training” stage)  
       - Full-parameter training (vision encoder, projector, LLM) on an `85M` concept-balanced image–text corpus (`LLaVA-OneVision-1.5-Mid-Training`).
       - Key finding: scaling mid-training data alone substantially boosts performance across many benchmarks, reducing the need for more complex recipes (Section 1; Figure 4).
    3) `Stage-2: Visual Instruction Tuning`  
       - Full-parameter supervised fine-tuning on a curated `22M` instruction dataset (`LLaVA-OneVision-1.5-Instruct`) covering seven capability areas (Caption; Chart & Table; Code & Math; Domain-specific; General VQA; Grounding & Counting; OCR; Science; Figure 3c). The team also uses the FineVision collection and a merged set for larger-scale SFT (Section 5.6.4; Figure 6).

- Concept-balanced mid-training data (Section 3.1; Figure 3)
  - Challenge: open web corpora are long-tailed—some concepts dominate while many are rare—leading to uneven learning (Appendix B, Figure 8).
  - Strategy:
    - Use the `MetaCLIP-H/14-Full-CC2.5B` encoders and its `~500K` concept vocabulary.
    - For each image: embed the image and retrieve its top-K nearest “concept embeddings.” These serve as refined pseudo-concepts even when the original caption is missing or poor (Section 3.1).
    - Assign a weight to each image inversely proportional to the frequency of its associated concepts; sample images by these normalized weights (“inverse-frequency sampling”). This evens out the concept distribution (Section 3.1; Figure 3a).
    - Caption all selected images in English and Chinese using a strong captioner; filter for quality and deduplicate to finalize the 85M set (Section 3.1).
  - Intuition: balance what the model sees conceptually, not just by dataset source counts. Figure 3a shows a smoother post-balancing coverage curve.

- Efficient training infrastructure (Section 4.2)
  - `Offline parallel data packing`:
    - Problem: multimodal batches vary widely in length; padding wastes compute.
    - Solution: pre-process the dataset to pack multiple short samples into fixed-length sequences before training. This uses hash buckets and multi-threaded, strategy-aware batching to maximize packing success and maintain batch composition.
    - Outcome: up to `11×` compression in the 85M pretraining set, meaning far fewer tokens are wasted as padding (Section 4.2).
    - Why offline: unlike on-the-fly packing, offline ensures consistent, uniform packed sequences and avoids runtime overhead.
  - `Hybrid parallelism` with `AIAK-Training-LLM` (built on Megatron-LM): uniform recomputation, distributed optimizer parallelism, 8K context, native-resolution inputs. The 85M-stage training of the 8B model runs on `128 × A800` GPUs for about `3.7 days` (Section 4.2).

## 4. Key Insights and Innovations
- Region-centric vision encoder for OCR-heavy tasks
  - What is new: using `RICE-ViT` with a single cluster discrimination loss and region-aware attention for local semantics, paired with 2D RoPE for variable resolution (Section 2.2).
  - Why it matters: Table 2 shows consistent gains vs. CLIP/SigLIP/DFN in OCR and document tasks with the same LLaVA-NeXT training pipeline. Example: at 560px, the RICE-ViT from OV-1.5-3B yields higher averages in OCR & Document Understanding (80.3%) and competitive General Vision Understanding (73.4%).
- Concept-balanced mid-training at scale is a decisive lever
  - What is new: feature-based concept induction via MetaCLIP embeddings enables balancing even for caption-poor or interleaved datasets (Section 3.1).
  - Why it matters: Figure 4 shows performance rises steadily when scaling mid-training from 0M → 4M → 85M across 10 benchmarks. Figure 5 shows that even at the same size (`2M`), concept-balanced sampling beats random sampling on `25/27` benchmarks.
- Cost-efficient training via offline parallel data packing
  - What is new: a practical, offline, multi-threaded packing pipeline tailored for heterogeneous multimodal lengths (Section 4.2).
  - Why it matters: up to `11×` effective compression, enabling the reported <$16k budget training (Abstract; Section 4.2). This makes large open models financially reachable.
- Fully open, end-to-end assets
  - What is new: releasing the 85M mid-training data distribution, the 22M instruction data, code, and checkpoints (Abstract; Figure 1 links).
  - Why it matters: reproducibility and extensibility—others can retrain or adapt.

## 5. Experimental Analysis
- Evaluation setup (Section 5.1)
  - Framework: `LMMs-Eval` with default prompts (Zhang et al., 2024).
  - Benchmarks grouped into four categories, spanning 27 datasets:
    - General VQA: MMStar, MMBench (en/cn), MME-RealWorld (en/cn), SeedBench, CV-Bench, ScienceQA, SEED-Bench-2-Plus, RealWorldQA.
    - Multimodal reasoning: MathVista, WeMath, MathVision, MMMU, MMMU-Pro (standard and vision).
    - OCR & Chart: ChartQA, CharXivDQ, DocVQA, OCRBench, AI2D (w/ and w/o M), InfoVQA.
    - Others: PixmoCount, CountBench, VL-RewardBench, V*.
  - Models compared: `LLaVA-OV-1.5-8B` vs `Qwen2.5-VL-7B` and `LLaVA-OV-1.5-4B` vs `Qwen2.5-VL-3B`. Extra fairness check with the same LLM in Appendix A.

- Main quantitative results (Table 1)
  - Overall head-to-head:
    - Quote: “LLaVA-OneVision-1.5-8B surpasses Qwen2.5-VL-7B on 18 of 27 benchmarks and LLaVA-OneVision-1.5-4B surpasses Qwen2.5-VL-3B on 27 of 27” (Section 5.1; Table 1).
  - Category averages:
    - General VQA average: `74.2` (OV-1.5-8B) vs `72.2` (Qwen2.5-VL-7B).
    - Reasoning average: `41.1` vs `40.8` (slight edge; OV-1.5-4B also leads its 3B counterpart by 38.4 vs 33.1).
    - OCR & Chart average: `85.0` vs `84.4`.
    - Others average: `68.8` vs `69.1` (near parity; Qwen leads slightly on this group).
  - Notable single-benchmark highs:
    - ScienceQA: `95.0` (OV-1.5-8B) vs `88.8` (Qwen2.5-VL-7B).
    - ChartQA: `86.5` (OV-1.5-8B).
    - DocVQA: `95.0` (OV-1.5-8B).
    - CountBench: `88.2` (OV-1.5-8B).
  - Reasoning tasks (Section 5.3): the 8B model edges out the 7B baseline on MathVista-mini (+1.0%), WeMath (+0.3%), MathVision (+3.2%), MMMU-val (+4.1%), MMMU-Prostandard (+1.1%). The 4B model is consistently ahead of Qwen2.5-VL-3B on all reasoning benchmarks listed.

- Vision encoder ablation (Table 2; Section 5.6.1)
  - Setup: fix LLM to Qwen2.5-7B and use LLaVA-NeXT data/pipeline; compare vision encoders at multiple resolutions.
  - Finding: RICE-ViT generally improves OCR/document tasks and remains competitive on general vision. At 560px, the RICE-ViT derived from OV-1.5-3B shows `53.7/87.1/81.9/73.8/73.3/30.4/53.6/64.8` on a set of OCR/Doc benchmarks and performs strongly on AI2D and MMBench variants (Table 2).

- Data scaling and balancing (Figures 4–5; Sections 5.6.2–5.6.3)
  - Figure 4: moving from no mid-training to 4M and then 85M mid-training samples improves 10 benchmarks across science diagrams, MME variants, MMMU, DocVQA, and ChartQA.
  - Figure 5: with the same `2M` size, concept-balanced mid-training consistently beats randomly sampled mid-training on `25/27` downstream benchmarks.

- Instruction data quality & scale (Figure 6; Section 5.6.4)
  - Three SFT datasets tested: `LLaVA-OV-1.5-Inst-Data`, `FineVision`, and `Merged46M` (a deduplicated/merged enlargement).
  - Finding: `Merged46M` yields the best curves across nearly all 16 SFT-tracked benchmarks; the 22M in-house instruction set matches FineVision closely.

- Fairness check with the same LLM (Appendix A; Figure 7)
  - To isolate the impact of the visual stack and training pipeline, `LLaVA-OV-1.5-3B` is trained on `Qwen2.5-3B-Instruct`. Result: outperforms `Qwen2.5-VL-3B` on `17/27` benchmarks (Figure 7).

- Are the experiments convincing?
  - Strengths
    - Broad benchmark coverage, clear category breakdown, and multiple slice analyses (vision encoder choice, data scaling, data balancing, SFT datasets).
    - Consistent use of public evaluation harness (`LMMs-Eval`) and explicit reporting of per-benchmark scores (Table 1).
    - Fairness appendix addressing backbone differences (Figure 7).
  - Caveats
    - The “Others” category average is slightly lower than Qwen2.5-VL-7B (68.8 vs 69.1), though many individual benchmarks are close.
    - No user preference studies or human evaluations are reported; results are purely benchmark-driven.
    - The exact “powerful captioner” used for the 85M set is not specified, which affects replicability of the data generation stage.

## 6. Limitations and Trade-offs
- Data construction assumptions
  - Concept balancing depends on MetaCLIP’s 500K concept vocabulary and encoders. This may bias sampling towards concepts well represented by MetaCLIP and underrepresent nuances outside its embedding space (Section 3.1; Appendix B).
  - The mid-training captions are generated by an unspecified captioner; caption quality and stylistic biases can shape downstream behavior (Section 3.1).
- Scope limitations
  - The work focuses on single-image understanding; video, audio, or multi-image dialogue are not addressed in training or evaluation.
  - Reasoning improvements are moderate on average; the approach does not include reinforcement learning or preference optimization yet (though an RL variant is “anticipated”; Abstract).
- Efficiency trade-offs
  - Offline data packing requires substantial preprocessing, hashing, and storage of packed sequences. This reduces training-time cost but shifts complexity and compute to preprocessing (Section 4.2).
  - The training run cited uses `128 × A800` GPUs for `3.7 days` at 8K context (Section 4.2). While the budget is claimed to be ~$16k, access to this hardware and orchestration remains a practical barrier for many labs.
- Generalization risks
  - As with most web-scale mixtures, even concept-balanced data can contain spurious correlations and long-tail gaps (Figure 8 highlights strong long-tail behavior in source datasets).
  - Benchmarks may not fully capture deployment scenarios (e.g., safety, long-form multi-step tool use, tabular PDF-heavy workflows).

## 7. Implications and Future Directions
- Shifts in the landscape
  - LLaVA-OneVision-1.5 shows that “just scale mid-training data—if it’s high-quality and concept-balanced—plus an OCR/region-strong vision encoder” is a simple, reproducible path to state-of-the-art open LMMs (Figure 4; Table 1). This lowers the barrier to building capable multimodal systems outside large organizations.
- What this enables
  - Re-trainable stacks: because datasets, code, and checkpoints are open, researchers can swap LLM backbones, specialize the instruction mix to domains (e.g., medicine, law), or extend the concept-balancing vocabulary to niche areas.
  - OCR/document-first applications: the RICE-ViT encoder and the strong OCR/Chart results (Table 1; Table 2) make the models attractive for enterprise document processing, scientific diagrams (AI2D, InfoVQA), and forms understanding (DocVQA).
  - Training systems research: the offline parallel packing method (Section 4.2) is a concrete recipe others can adopt to cut costs in multimodal training.
- Next research steps
  - RL and preference optimization: the paper previews an `LLaVA-OneVision-1.5-RL` release. Aligning responses to human preferences could improve instruction following and factuality.
  - Richer modalities and contexts: extend to video, multi-image narratives, and longer contexts beyond 8K; evaluate on tasks requiring temporal reasoning and tool use.
  - Transparent captioning and filtering: release or standardize the captioner and filtering stack for end-to-end reproducibility and bias auditing.
  - Concept balancing beyond MetaCLIP: explore adaptive vocabularies and cross-dataset alignment methods that reduce dependence on a single concept library and improve coverage of underrepresented domains.

In short, LLaVA-OneVision-1.5 contributes a clear, open, and empirically validated path to training competitive multimodal models at moderate cost. Its most substantive advances are the combination of region-aware vision encoding, large yet balanced mid-training data, and a pragmatic efficiency toolkit—together yielding strong OCR/document and broad VQA performance with a reproducible footprint.
