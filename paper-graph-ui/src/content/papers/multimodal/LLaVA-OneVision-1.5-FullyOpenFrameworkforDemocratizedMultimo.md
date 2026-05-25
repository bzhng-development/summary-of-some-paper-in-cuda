# LLaVA-OneVision-1.5: Fully Open Framework for Democratized Multimodal Training

**ArXiv:** [2509.23661](https://arxiv.org/abs/2509.23661)

## 🎯 Pitch

LLaVA-OneVision-1.5 delivers a fully open-source, cost-efficient framework for training large multimodal models (LMMs) entirely from scratch—combining a state-of-the-art, region-aware vision encoder (RICE-ViT), vast curated datasets, and highly optimized training pipelines. This breakthrough enables researchers and practitioners to build high-performing vision-language models without proprietary data, expensive infrastructure, or opaque methods, democratizing access to cutting-edge multimodal AI and accelerating advances across diverse real-world tasks.

---

## 1. Executive Summary (2-3 sentences)
This paper introduces LLaVA-OneVision-1.5, a fully open framework for training large multimodal models (LMMs) from scratch with strong performance and a constrained budget. It combines a region-aware vision encoder (`RICE-ViT`), concept-balanced large-scale datasets (85M pretraining pairs and 22M instructions), and an efficient offline data-packing training pipeline to outperform comparable open models across many benchmarks (Table 1), while keeping total compute around $16k (Section 4.2).

## 2. Context and Motivation
- Problem addressed:
  - High-performing multimodal models (image+text) are mostly proprietary, with undisclosed training data and pipelines, which blocks reproducibility and community progress (Introduction, p.1–2).
  - Open LMMs exist (e.g., LLaVA, LLaVA-Next, LLaVA-OneVision, Molmo, Open-Qwen2VL), but either lag behind state of the art (SOTA) or require heavy compute (Introduction, p.1–2).
- Why this matters:
  - Multimodal systems power applications involving images, documents, charts, and PDFs. An open, efficient, reproducible approach lowers the barrier for research and deployment (Introduction, p.1).
- Prior approaches and their gaps:
  - Early fully open releases (LLaVA series) now trail top models on benchmarks (Introduction, p.1–2).
  - Molmo shows strong results with open weights/data/code but relies on refined pipelines and significant resources (Introduction, p.1–2).
  - Open-Qwen2VL targets compute efficiency but still leaves a gap to larger closed or semi-closed systems (Introduction, p.2).
- Positioning:
  - LLaVA-OneVision-1.5 aims to be a fully open, efficient “from-scratch” training recipe with curated large-scale data, an encoder specialized for region-level semantics, and an engineering stack that fits within a modest budget (Abstract; Section 2; Sections 3–4).

## 3. Technical Approach
The system keeps the classic “ViT–MLP–LLM” layout (Figure 2) but modernizes each part and the data/training pipeline.

- Model architecture (Section 2; Figure 2):
  - Vision encoder: `RICE-ViT`
    - What it is: a ViT trained with a region-based cluster discrimination loss that encourages images to be understood as collections of semantically coherent regions rather than only global embeddings (Section 2.2).
    - Why it helps: conventional contrastive encoders (e.g., CLIP/SigLIP) align whole-image features and often miss fine-grained local semantics critical for text-in-image (OCR), grounding, and dense tasks (Section 2.2).
    - How it works:
      - Region-aware attention enhances local semantic modeling.
      - 2D RoPE: “2D rotary positional encoding,” which rotates query/key vectors by position-dependent phases in two spatial dimensions so the encoder natively supports variable resolutions without special fine-tuning (Section 2.2).
      - Unified loss: a single cluster discrimination objective trained on 450M images and 2.4B candidate regions improves general understanding, OCR, and localization without multiple specialized losses (Section 2.2). This contrasts with SigLIP2’s mixture of SILC/TIPS/LocCa/Sigmoid losses.
  - Projector (Section 2.1):
    - Maps vision features to the LLM embedding space.
    - Design: spatially adjacent groups of 4 patch features are concatenated, then passed through a 2-layer MLP (multi-layer perceptron) into the LLM space (following Qwen2.5-VL).
    - The `[CLS]` token is preserved to keep global semantics (Figure 2 caption).
  - Large Language Model (LLM) (Section 2.1):
    - Backbone: `Qwen3` (Team, 2025), acting as the reasoning/generation core.
- Training pipeline (Section 4.1):
  - Terminology: “mid-training” (Stage-1.5) is a full-parameter training phase between initial alignment and instruction tuning.
  - Stage-1 (Language–Image Alignment):
    - Goal: teach the projector to map vision features into the LLM’s token space.
    - Data: `LLaVA-1.5 558K` (Section 3.1; 4.1).
    - Setup: pretrain the projection layer against this small, curated alignment set.
  - Stage-1.5 (High-Quality Knowledge Learning):
    - Full-parameter training (vision encoder + projector + LLM).
    - Data: `LLaVA-OneVision-1.5-Mid-Training` (85M pairs, English+Chinese; Section 3.1).
    - Key insight: scaling this stage alone markedly boosts capability (Figure 4).
  - Stage-2 (Visual Instruction Tuning):
    - Full-parameter training on instruction-following data to shape outputs and task adherence.
    - Data: `LLaVA-OneVision-1.5-Instruct` (22M) plus `FineVision` (Section 4.1; 5.6.4).
- Data construction (Section 3; Figure 3):
  - 85M Mid-Training (Section 3.1):
    - Sources: COYO-700M, Obelics, DataComp-1B, LAION-CN, ImageNet-21K, SA-1B, MINT, Zero250M (Figure 3b).
    - Problem: long-tail and uneven concepts across sources (Appendix B; Figure 8).
    - Solution—Concept-balanced sampling without relying on noisy captions:
      - Feature-based top-K concept assignment:
        1) Use pretrained MetaCLIP-H/14 encoders to embed images and a 500K concept vocabulary into the same space.
        2) Retrieve the top-K nearest concept embeddings per image.
        3) Weight each image by inverse frequencies of its assigned concepts (rare concepts get higher weight).
        4) Sample images according to normalized weights to flatten the concept distribution (Section 3.1; “Top-K Concept Assignment and Balance Sampling”).
      - After sampling 85M images, generate English/Chinese pseudo-captions with a strong captioner and filter for validity (Section 3.1).
      - Outcome: smoother concept coverage (Figure 3a) and a balanced 85M corpus (Figure 3b).
  - 22M Instruction (Section 3.2; Figure 3c):
    - Composition: balanced across 7 categories—Caption, Chart & Table, Code & Math, Domain-specific, General VQA, Grounding & Counting, OCR, and Science.
  - Additional SFT mixing (Section 5.6.4; Figure 6):
    - `Merged46M`: deduplicated merge of the paper’s instruction set with `FineVision`, showing best SFT performance on most of 16 benchmarks.
- Efficient training infrastructure (Section 4.2):
  - Offline parallel data packing:
    - What it is: a preprocessing method that concatenates multiple short samples into fixed-length packed sequences to minimize padding waste.
    - How it works: uses hash buckets and multi-threaded, strategy-aware batching during preprocessing, not at runtime; this yields uniform output lengths and better GPU utilization (Section 4.2).
    - Benefits: up to 11× compression on the 85M pretraining set (Section 4.2).
  - Hybrid parallelism:
    - Framework: AIAK-Training-LLM (Baidu Cloud’s optimized Megatron-LM) with a transformer engine, distributed optimizer parallelism, and uniform recomputation (Section 4.2).
    - Scale/time: mid-training LLaVA-OV-1.5-8B with 8K context length at native resolution on 85M captions using 128× A800 GPUs over 3.7 days (Section 4.2).
    - Budget: approximately $16,000 for the full training (Abstract; Section 4).

## 4. Key Insights and Innovations
- Region-aware vision encoder with a unified loss (fundamental):
  - `RICE-ViT` improves local semantic modeling via region-aware attention and 2D RoPE, trained with a single cluster discrimination loss that jointly enhances general understanding, OCR, and localization (Section 2.2).
  - Significance: rivals systems like SigLIP2 without multiple losses and reduces architectural complexity and training overhead (Section 2.2).
- Caption-free concept balancing at scale (fundamental):
  - Moves from caption-based balancing (MetaCLIP) to feature-based top-K concept assignment and inverse-frequency sampling, letting the pipeline handle sources with poor or missing captions (COYO, SA-1B, ImageNet-21K, Obelics) (Section 3.1; Figure 3a–b; Appendix B).
  - Significance: creates an 85M balanced corpus with broad and smoother concept coverage (Figure 3a; Appendix B, Figure 8 and Table 3).
- Mid-training scaling as the main driver (insightful empirical finding):
  - Simply scaling Stage-1.5 (high-quality knowledge learning) is enough to reach SOTA-like behavior; extra complex stages are not required (Introduction, p.2; Figure 4).
- Offline parallel data packing for efficiency (engineering innovation):
  - Shifts packing offline to deliver up to 11× sequence utilization improvement, enabling training within a modest budget and shorter wall time (Section 4.2).
- Broad, open release (ecosystem contribution):
  - Code, checkpoints (`Base`, `Instruct`), and both datasets are released, enabling full reproducibility and community extension (Abstract; Section 1; links on p.1).

## 5. Experimental Analysis
- Evaluation setup (Section 5.1):
  - Framework: `LMMs-Eval` with default prompts.
  - Four benchmark categories:
    - General VQA: MMStar, MMBench (en/cn), MME-RealWorld (en/cn), SeedBench-image, SEED-Bench-2-Plus, CV-Bench, ScienceQA, RealWorldQA.
    - Multimodal Reasoning: MathVista(mini), WeMath, MathVision, MMMU(val), MMMU-Pro (standard/vision).
    - OCR & Chart: ChartQA, CharXivDQ, DocVQA, OCRBench, AI2D (with/without materials), InfoVQA.
    - Others: PixmoCount, CountBench, VL-RewardBench, V* (visual search).
- Main results (Table 1):
  - Overall head-to-heads:
    - 8B vs Qwen2.5-VL-7B:
      - Wins on 18/27 benchmarks. Category averages:
        - General VQA avg: 74.2 vs 72.2.
        - Reasoning avg: 41.1 vs 40.8.
        - OCR & Chart avg: 85.0 vs 84.4.
        - Others avg: 68.8 vs 69.1 (slightly lower).
      - Standout wins:
        - ScienceQA: 95.0 vs 88.8.
        - MME-RealWorld (en): 62.3 vs 57.3; (cn): 56.1 vs 51.5.
        - MMStar: 67.7 vs 62.5.
        - ChartQA: 86.5 vs 84.1.
        - CountBench: 88.2 vs 86.4.
      - Notable losses or ties:
        - SEED-Bench-2-Plus: 69.2 vs 70.9 (−1.7).
        - MMMU-Pro(vision): 25.2 vs 32.8 (−7.6).
        - OCRBench: 82.9 vs 84.2 (−1.3).
        - PixmoCount: 62.2 vs 63.3 (−1.1).
        - VL-RewardBench: 46.7 vs 49.7 (−3.0).
    - 4B vs Qwen2.5-VL-3B:
      - Wins on all 27 benchmarks. Category averages:
        - General VQA avg: 72.1 vs 66.4.
        - Reasoning avg: 38.4 vs 33.1.
        - OCR & Chart avg: 82.6 vs 79.8.
        - Others avg: 63.8 vs 58.8.
  - Representative quote:
    > “LLaVA-OneVision-1.5-8B surpasses Qwen2.5-VL-7B on 18 of 27 benchmarks and LLaVA-OneVision-1.5-4B surpasses Qwen2.5-VL-3B on 27 of 27 benchmarks.” (Section 5.1; Table 1)
- Ablations and diagnostics:
  - Encoder choice (Table 2; Section 5.6.1):
    - Within LLaVA-NeXT settings, `RICE-ViT` consistently improves OCR/Doc and general vision metrics compared to CLIP, SigLIP, SigLIP2, DFN across multiple resolutions.
    - At higher resolution (560px), “RICE-ViT from OV-1.5 3B” shows strong OCR averages (80.3) and competitive general metrics (73.4), with average improvements over “Qwen-ViT from Qwen2.5-VL 7B” in OCR & Document (+1.9%) and General (+0.9%).
  - Mid-training data scaling (Figure 4; Section 5.6.2):
    - Across 10 benchmarks, moving from 0M → 4M → 85M mid-training samples yields monotonic gains after the same Stage-1 alignment and LLaVA-Next SFT.
    - This empirically supports the claim that Stage-1.5 scale is the primary driver of capability.
  - Concept balancing effectiveness (Figure 5; Section 5.6.3):
    - With only 2M mid-training samples, the concept-balanced subset beats random sampling on 25/27 benchmarks.
    - Quote:
      > “Using a balanced mid-training dataset yields consistent improvements over a random sampling strategy.” (Figure 5, caption)
  - Instruction data quality/scale (Figure 6; Section 5.6.4):
    - `Merged46M` (the deduplicated merge with FineVision) outperforms either individual dataset on most of 16 SFT benchmarks, at the same number of training steps (batch size doubled to keep steps equal).
- Fairness check with same LLM (Appendix A; Figure 7):
  - To isolate the vision/data/training recipe from the language backbone, a 3B variant trained on `Qwen2.5-3B-Instruct` is compared to `Qwen2.5-VL-3B`. It performs better on 17/27 benchmarks, supporting the encoder/data/training benefits beyond the LLM choice.
- Efficiency evidence (Section 4.2):
  - Offline packing achieves up to 11× compression; mid-training of 8B model at native resolution completes in 3.7 days on 128×A800s; budget ~$16k.
- Do the experiments support the claims?
  - Yes for the central claims:
    - Performance: clear gains vs same-size open baselines in most areas (Table 1), especially at the 4B scale.
    - Encoder/data effects: multiple ablations (Table 2; Figures 4–6) demonstrate benefits from `RICE-ViT`, mid-training scale, and concept balance.
    - Efficiency: concrete utilization/compute details (Section 4.2) and a budget estimate anchor the “cost-effective” claim.
  - Mixed outcomes:
    - On “Others” category, the 8B model is slightly below Qwen2.5-VL-7B on average (68.8 vs 69.1) and underperforms on `MMMU-Pro(vision)` (Table 1), suggesting possible headroom for visual grounding or reward-model alignment.

## 6. Limitations and Trade-offs
- Data creation assumptions:
  - Concept balancing relies on MetaCLIP feature space and a fixed 500K concept vocabulary; unusual domains not well represented in these concepts may remain under-covered (Section 3.1; Appendix B).
  - The 85M pretraining captions are generated by an external captioner; while filtered, any captioner bias or errors propagate into training (Section 3.1).
- Compute and system constraints:
  - Although cost-optimized, the recipe still requires substantial infrastructure—128×A800 for several days—and an 8K context setup (Section 4.2).
  - Offline packing improves utilization but increases preprocessing complexity and storage of packed corpora; repacking is needed if sequence-length or batching strategies change (Section 4.2).
- Task-specific gaps:
  - Relative weaknesses on certain benchmarks (e.g., `MMMU-Pro(vision)`, `VL-RewardBench`) indicate that advanced grounding, preference alignment, or vision-heavy reasoning may need additional training (Table 1).
- Generalization scope:
  - The instruction data cover seven broad categories (Figure 3c), but certain specialized domains (medical imaging, remote sensing, CAD) are not explicitly targeted; further domain adaptation may be needed.
- Pending components:
  - A reinforcement learning (RL) variant is “anticipated shortly” but not included; several “Others” tasks (e.g., reward-based metrics) might benefit from it (Abstract).

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
