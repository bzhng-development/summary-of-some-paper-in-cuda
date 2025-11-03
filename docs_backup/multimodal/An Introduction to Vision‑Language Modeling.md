# An Introduction to Vision‑Language Modeling

**ArXiv:** [2405.17247](https://arxiv.org/abs/2405.17247)
**Authors:** Florian Bordes, Richard Yuanzhe Pang, Anurag Ajay, Alexander C. Li, Adrien Bardes, Suzanne Petryk, Oscar Mañas, Zhiqiu Lin, Anas Mahmoud, Bargav Jayaraman, Mark Ibrahim, Melissa Hall, Yunyang Xiong, Jonathan Lebensold, Candace Ross, Srihari Jayakumar, Chuan Guo, Diane Bouchacourt, Haider Al‑Tahan, Karthik Padthe, Vasu Sharma, Hu Xu, Xiaoqing Ellen Tan, Megan Richards, Samuel Lavoie, Pietro Astolfi, Reyhane Askari Hemmat, Jun Chen, Kushal Tirumala, Rim Assouel, Mazda Moayeri, Arjang Talattof, Kamalika Chaudhuri, Zechun Liu, Xilun Chen, Quentin Garrido, Karen Ullrich, Aishwarya Agrawal, Kate Saenko, Asli Celikyilmaz, Vikas Chandra
**Institutions:** Meta Research and affiliated academic institutions

## 🎯 Pitch

This tutorial paper organizes the burgeoning field of vision-language models into four paradigms, providing actionable recipes for training, data curation, and evaluation, thus guiding practitioners on choosing the right approach for building reliable models. The significance lies in its ability to demystify complex VLM processes, enhancing application accuracy across AI-driven retrieval, captioning, and generative systems while addressing critical issues such as bias and hallucinations.

---

## 1. Executive Summary (2-3 sentences)
This tutorial paper systematizes the rapidly growing area of vision–language models (VLMs) into four training paradigms—contrastive, masking, generative, and pretrained-backbone mapping—then turns that taxonomy into actionable training recipes, evaluation practices, and failure analyses (Figures 1–3; Sections 2–5). Its significance is pragmatic: it explains how to choose an approach, curate data, scale training affordably, benchmark responsibly (bias, hallucinations, memorization), and extend from images to video.

## 2. Context and Motivation
- Problem/gap addressed
  - Building reliable models that connect images and text remains unsolved. Section 1 lists persistent failures: spatial relations and counting, attribute/order understanding, prompt under-specification, and hallucinations.
  - The field lacks a clear, practical guide for newcomers on which modeling family to pick, how to train efficiently, and how to evaluate responsibly without being misled by biased benchmarks (Sections 1, 4).

- Why it matters
  - Real-world impact: VLMs underpin assistants, retrieval, captioning, and generative systems. Reliability issues like hallucinations and weak grounding can mislead users and propagate harms (Sections 1, 4.2–4.5).
  - Theoretical significance: connecting continuous vision and discrete language requires reconciling learning objectives across modalities; the paper offers unifying views (energy-based, information-theoretic) that clarify when and why objectives work (Sections 2.2, 2.3.3).

- Prior approaches and their shortfalls
  - Early BERT-style multimodal transformers (e.g., `VisualBERT`, `ViLBERT`) relied on masked modeling and image–sentence prediction but struggled with scale and generalization (Section 2.1).
  - Contrastive models like `CLIP` unlocked zero-shot transfer but need huge batches and data; they may rely on language priors and can be weak on compositionality or spatial reasoning (Sections 2.2–2.2.1, 4.1.6, 4.2.3).
  - Generative models add flexibility but are compute-intensive and slow to evaluate; diffusion-based classification requires hundreds to thousands of network evaluations per image (Section 2.4.3; Eq. 7).
  - Pretrained LLM backbones reduce compute but can inherit LLM hallucinations and biases; mapping quality and data alignment become bottlenecks (Section 2.5, 3.3.4).

- How this paper positions itself
  - It is an introduction that bridges conceptual understanding and practical know-how: a clear taxonomy (Figure 1; Section 2), training and curation recipes (Section 3; Figure 2), evaluation pitfalls and remedies (Section 4; Figure 3), and a road-map for video (Section 5). It is not a comprehensive survey, but it curates representative exemplars and concrete tactics (Section 1, footnotes).

## 3. Technical Approach
The “approach” is a framework that explains how today’s VLMs are trained, why the objectives work, and how to build and evaluate them in practice.

- Four training paradigms (Figure 1; Section 2)
  1) Contrastive learning with an energy-based lens (Section 2.2)
     - Core idea: learn embeddings where positive pairs (matching image–text) have low “energy” (are close) and negatives have high energy (are far). This is cast via Noise-Contrastive Estimation (NCE) and `InfoNCE` (Eq. 2).
       - `NCE` reframes density estimation as binary classification against a noise distribution (Eq. 1, described in Section 2.2).
       - `InfoNCE` uses a softmax over dot-product or cosine similarities between a query and a set of positives/negatives (Eq. 2).
       - Trade-off: effectiveness often scales with batch size because negatives come from the mini-batch.
     - `CLIP` (Section 2.2.1) applies `InfoNCE` to image–caption pairs; positives are matched pairs, negatives are other captions in the batch. Notable property:
       > “a ResNet-101 CLIP matched the performance of a supervised ResNet … attaining 76.2% zero-shot classification accuracy” (Section 2.2.1).
     - Variants:
       - `SigLIP` replaces `InfoNCE` with the original binary NCE, improving zero-shot at small batches (Section 2.2.1).
       - `Llip` conditions an image encoder on the target caption via cross-attention to capture caption diversity (Section 2.2.1).

  2) Masked modeling (Section 2.3)
     - Core idea: reconstruct masked tokens either in image space (masked patches) or text space (masked words), optionally sharing information cross-modally.
     - `FLAVA` (Section 2.3.1) has separate image and text encoders and a multimodal transformer; it pretrains with masked unimodal losses and a multimodal masked objective plus a contrastive loss, achieving strong across-task coverage.
     - `MaskVLM` (Section 2.3.2) applies masking directly in pixels and tokens; crucially, each modality’s reconstruction is informed by the other (cross-modal flow).
     - Unifying theory (Section 2.3.3): VLM objectives are framed as a rate–distortion trade-off (Eq. 3–4):
       - Intuition: choose representations Z that compress away superfluous details (`rate`) while preserving predictive information needed for reconstruction or discrimination (`distortion`).
       - Contrastive = “compression without reconstruction” (distortion assesses pairwise equivalence); auto-encoding = explicit reconstruction distortion.

  3) Generative modeling (Section 2.4)
     - Text generation: `CoCa` adds a generative captioning loss on top of contrastive pretraining (Section 2.4.1).
     - Mixed-modality generation: `CM3leon` and `Chameleon` tokenize images into discrete tokens (e.g., 1024 tokens from an 8192-codebook) and interleave with text via a special `<break>` token in a decoder-only transformer; training uses retrieval-augmented next-token prediction followed by instruction tuning (Section 2.4.2).
     - Generative models for discriminative tasks (Section 2.4.3):
       - Autoregressive: classify by Bayes’ rule using `log pθ(x|c)` = sum of log-probs over image tokens (Eq. 5–6); tokenizers are typically VQ-VAE/VQ-GAN/ViT-VQGAN.
       - Diffusion: approximate `log pθ(x|c)` via a reweighted noise-prediction loss (Eq. 7); methods reduce sampling cost but remain expensive.
       - Advantages cited: better out-of-distribution robustness and compositional reasoning vs. discriminative models (Section 2.4.3).

  4) Pretrained backbones (Section 2.5)
     - Core idea: keep the LLM and often the vision encoder frozen; learn a small mapping network to pass visual tokens into the LLM’s embedding space.
     - `Frozen` maps NF-ResNet features into a 7B LLM’s token space; conditions generation on interleaved text–image embeddings (Section 2.5.1).
     - `MiniGPT-4` uses a simple linear projector from `BLIP-2`’s vision stack into `Vicuna`’s input space; trains in two rounds (5M pairs; then instruction-tuning with curated data), requiring only a few GPUs-hours because only the projector is trained (Section 2.5.2).
     - Extensions: `MiniGPT-5` adds image generation; `MiniGPT-v2` unifies multiple VL tasks with task identifiers; `Qwen-VL` and `BLIP-2` use cross-attention or Q-Former bridges (Section 2.5.2–2.5.3).

- Training recipes and systems (Section 3; Figure 2)
  - Data curation (Section 3.1): three strategies
    - Heuristics (language filters, resolution checks, text spotting).
    - Ranking by pretrained VLMs (`CLIPScore`, `T-MARS`, `Sieve`) to keep well-aligned pairs.
    - Diversity/balancing (`DataComp`, `MetaCLIP`) to cover long-tail concepts.
    - Key observation:
      > “Having a wide range of training data concepts seems to be one of the most important components behind the ‘zero-shot abilities’ of VLMs.” (Section 3.1)
  - Synthetic data (Sections 3.1.1, 3.1.2): replace noisy captions with synthetic ones (`BLIP`, `BLIP2`, `LLaVA`); generate images via diffusion/AR and train contrastively (`SynCLR`, `SynthCLIP`).
  - Interleaved training data (Section 3.1.3): natural interleaving (`OBELICS`) vs. retrofitting text corpora with images (`MMC4`).
  - Data quality measurement (Section 3.1.4): text quality (`QuRating`, pruning), image aesthetics (`VILA`, LAION-aesthetics), alignment (CLIP-family).
  - Human annotation (Section 3.1.5): high-quality, fine-grained datasets (e.g., `DCI` for dense captions) are valuable but costly.
  - Software and compute (Section 3.2):
    - Use `OpenCLIP`/`Transformers` to reproduce/compare.
    - Compute budgeting:
      > “training a contrastive model like CLIP … should not require more than 64 GPUs (… around 10K USD)” when data are high quality and masking is leveraged (Section 3.2.2).
    - Speed-ups: `torch.compile`, `xformers`, and, crucially, fast data loading (store uncompressed, use `FFCV`) (Section 3.2.3). Masking also reduces training cost for large models.
    - What matters most: image resolution, vision encoder capacity, and data (Section 3.2.4; `MM1`).
  - Choosing a model family (Section 3.3)
    - Contrastive (`CLIP`) for reusable embeddings/retrieval; needs large data and batches; not generative.
    - Masking to avoid batch-dependence; simpler small-batch training; may need decoders.
    - Generative for world-modeling and direct decoding; costly but more interpretable results.
    - LLM-backbone mapping when compute/data are limited; may inherit LLM hallucinations.
  - Improving grounding (Section 3.4): supervised grounding (`X-VLM` with boxes/IoU loss) vs. auto-generated boxes (`Kosmos-2` via `spaCy` + `GLIP`); negative captioning to force discrimination.
  - Alignment via instruction tuning and RLHF (Section 3.5):
    - `LLaVA` family shows scalable recipes. Notable result:
      > “LLaVA-RLHF … achieves 94% performance level of GPT-4 on LLaVA-Bench … [and] outperforms baselines by 60% on MMHAL-BENCH” (Section 3.5.1).
  - Text-rich image understanding (Section 3.6): `LLaVAR` augments instruction tuning with OCR-heavy data (up to +20% on text-based VQA); `Monkey` uses higher resolution and sliding-window patches; `Lumos` decouples an on-device scene-text recognition module from the cloud LLM for latency/accuracy.
  - Parameter-efficient fine-tuning (Section 3.7): `LoRA` variants (`QLoRA`, `VeRA`, `DoRA`), prompt-based (`CoOp`, `VPT`), adapters (`CLIP-Adapter`, `VL-Adapter`, `LLaMA-Adapter V2`), and minimal mapping networks (`MAPL`, `LiMBeR`).

- Responsible evaluation (Section 4; Figure 3)
  - Captioning metrics: classic n-gram metrics (BLEU/ROUGE) vs. embedding-based `CLIPScore`; each has caveats (Sections 4.1.1–4.1.2).
  - Text-to-image consistency: LLM-generated QA (`TIFA`, `DSG`), visual programs (`VPEval`), and `VQAScore` (Sections 4.1.2).
  - VQA: many datasets; exact-match `VQA Accuracy` underestimates open-ended answers; using LLMs as judges improves fairness (Section 4.1.3).
  - Selective prediction and visual dialog (Sections 4.1.3).
  - Text-centric VQA and OCR benchmarks (Section 4.1.4).
  - Zero-shot classification and OOD alignment influences (Sections 4.1.5).
  - Compositionality: `Winoground`, `ARO`, `SugarCrepe`—with pitfalls such as language-only shortcuts and even a tie-breaking bug:
    > “a model whose parameters are all equal to zero could achieve 100% accuracy” if equal probabilities are always argmaxed to the first option (Section 4.1.6, Warning).
  - Dense captioning and crop–caption matching with `DCI` (Section 4.1.7).
  - Synthetic evaluations (`PUG`) for controlled spatial testing (Section 4.1.8).

- Bias, hallucination, memorization, red teaming (Sections 4.2–4.5)
  - Bias via classifications and embeddings; synthetic contrast sets help diagnose (Section 4.2).
  - Language priors can solve “multimodal” benchmarks without images (Section 4.2.3).
  - Training-data concept frequency predicts “zero-shot” success (Section 4.2.4).
  - Hallucinations: `CHAIR`, `POPE`, LLM-as-judge evaluations (Section 4.3).
  - Memorization: “déjà vu” test shows CLIP can recall unlabeled objects; text randomization regularization helps (Section 4.4).
  - Red teaming: define harms, probe with adversarial prompts, and mitigate with post-processing or RLHF (Section 4.5).

- Extending to video (Section 5)
  - Early fusion (`VideoBERT`, `MERLOT`) vs. LLM-aligned video Q-formers (`Video-LLaMA`, `Video-LLaVA`, `MiniGPT4-Video`) (Sections 5.1–5.3).
  - New evaluation opportunities (long-form understanding in `EgoSchema`) and synthetic physics tests:
    > GRASP finds `VideoLLaMA`/`PandaGPT` “do not exceed random performance” on physical law violations despite humans >80% (Section 5.4).
  - Challenges: scarcity of temporal supervision, noun bias in video-CLIP, and compute/memory constraints; masking and compressed video pipelines are promising (Section 5.5).

## 4. Key Insights and Innovations
- A unifying conceptual lens for VLM objectives (fundamental)
  - The energy-based view clarifies contrastive training mechanics (Section 2.2), while the information-theoretic rate–distortion framing (Eq. 3–4) unifies masking and contrastive losses as different choices of “distortion” and “rate” (Section 2.3.3). This helps reason about when to favor reconstruction vs. discrimination.

- Actionable, end-to-end training guidance (practical, high impact)
  - Data curation playbook with concrete filters and ranking (`CLIPScore`, `T-MARS`, `Sieve`), diversity balancing (`DataComp`, `MetaCLIP`), and when to inject synthetic captions/images (Sections 3.1–3.1.2). The guidance on throughput bottlenecks (data loading, FFCV, uncompressed storage) is unusually specific and impactful (Section 3.2.3).

- Clear decision criteria for model family choice (practical)
  - The “When to use …” subsections (3.3.1–3.3.4) go beyond taxonomy; they map objectives to constraints (batch size, compute, generative needs) and to risk factors (LLM hallucinations). This reduces trial-and-error for practitioners.

- Responsible evaluation and pitfalls (fundamental for reliability)
  - The paper catalogs where evaluation can mislead—language priors (Section 4.2.3), hackable compositionality sets (Section 4.1.6), and metric brittleness (Section 4.1.1–4.1.3)—and offers fixes: LLM judges for VQA, balanced datasets (e.g., `Winoground`), dense/synthetic evaluations (`DCI`, `PUG`).

- Compact recipes for alignment and OCR-heavy scenarios (practical)
  - The `LLaVA` series demystifies instruction tuning + RLHF for multimodal chat, including concrete scales and training time (Section 3.5.1). The OCR-focused designs (`Monkey` high-res patches; `Lumos` decoupled STR) show how to retrofit VLMs for text-rich images (Section 3.6).

## 5. Experimental Analysis
- Evaluation methodology synthesized in the paper
  - Datasets and tasks: captioning (COCO Captions, Section 4.1.1), text–image consistency (`TIFA`, `DSG`, `VQAScore`, Section 4.1.2), VQA families (VQAv2, GQA, VizWiz, OK-VQA, ScienceQA, MMMU; Section 4.1.3), text-centric OCR VQA (Section 4.1.4), zero-shot classification (ImageNet and many others; Section 4.1.5), compositional reasoning (`Winoground`, `ARO`, `SugarCrepe`; Section 4.1.6), dense captions (`DCI`; Section 4.1.7), and synthetic (`PUG`; Section 4.1.8).
  - Metrics: BLEU/ROUGE (captioning), `CLIPScore` (embedding alignment), VQA Accuracy vs. LLM-as-judge scoring, retrieval accuracy on compositional datasets, crop–caption matching accuracy, and text-to-image VQA probabilities (`VQAScore`) (Sections 4.1.1–4.1.3).

- Representative quantitative results cited
  - `CLIP` zero-shot ImageNet:
    > “ResNet-101 CLIP … attaining 76.2% zero-shot classification accuracy” (Section 2.2.1).
  - `LLaVA-RLHF`:
    > “94% performance level of GPT-4 on LLaVA-Bench … [and] outperforms baselines by 60% on MMHAL-BENCH” (Section 3.5.1).
  - OCR-enhanced instruction tuning:
    > `LLaVAR` improves “up to a 20% accuracy” on text-based VQA with minimal impact on natural images (Section 3.6).
  - Training efficiency:
    > Contrastive pretraining “should not require more than 64 GPUs … around 10K USD” given strong data and masking (Section 3.2.2).
  - Video physics reasoning:
    > Some modern video-VLMs “do not exceed random performance” on GRASP physics violations while humans exceed 80% (Section 5.4).

- Do these support the guidance?
  - Yes, but note the paper aggregates results from many sources rather than running new experiments. The evidence is appropriately scoped: big-picture trends (data quality matters; instruction tuning + RLHF helps alignment; language priors can confound benchmarks; synthetic tests reveal missed capabilities) are tied to specific, cited studies (Sections 3–5).

- Ablations, failure cases, robustness checks discussed
  - DataComp’s pruning and diversity ablations (Section 3.1): show dataset design can rival raw scale.
  - Negative captioning and compositional benchmarks (`ARO`, `SugarCrepe`) expose weaknesses in relation/attribute/order reasoning (Section 4.1.6).
  - The tie-breaking pitfall demonstrates how implementation details can invalidate binary-choice benchmarks (Section 4.1.6, Warning).
  - Memorization analysis uses a reference model to separate correlation learning from copy-overfit; text randomization ablates memorization without severe utility loss (Section 4.4).

- Mixed/conditional results and trade-offs
  - Generative classifiers improve OOD and compositionality (Section 2.4.3) but are computationally heavy at inference (diffusion Eq. 7).
  - Synthetic captions/images help when labels are noisy but can cap diversity compared to web-scale alt-text (Section 3.1.1).
  - Interleaved data (OBELICS) offers more authentic multimodal context but requires heavy crawling/curation (Section 3.1.3).

## 6. Limitations and Trade-offs
- Assumptions and scope
  - The paper is explicitly an introduction, not an exhaustive survey; coverage is selective (Section 1). It assumes readers will draw from the cited implementations to realize the recipes.

- Scenarios not fully addressed
  - End-to-end, apples-to-apples comparisons across all four paradigms on a common compute/data budget are out of scope.
  - Detailed security/privacy risks for multimodal data collection and annotation are not analyzed beyond red teaming pointers (Section 4.5).

- Computational and data constraints
  - Contrastive methods still benefit from large batches and massive curated data; high-quality deduplication and alignment filtering increase pre-processing cost (Section 3.1–3.2).
  - Generative models (especially diffusion) are expensive during inference for classification-by-likelihood (Section 2.4.3).
  - Video VLMs multiply memory and compute barriers; temporal supervision remains scarce, and models show noun bias (Section 5.5).

- Remaining weaknesses/open questions
  - Persistent failures in spatial reasoning, counting, and compositionality even after scaling (Sections 1, 4.1.6).
  - Benchmark validity: mitigating language priors in evaluation and ensuring balanced, tie-robust protocols (Sections 4.1.6, 4.2.3).
  - How to systematically balance diverse, long-tail concepts while avoiding biases and privacy issues in web data (Sections 3.1, 4.2).

## 7. Implications and Future Directions
- How this work changes the landscape
  - It provides a coherent mental model and a practical toolbox. For teams entering VLM research, this paper shortens the path from idea to reliable evaluation: pick a paradigm (Figure 1), curate data pragmatically (Section 3.1), train efficiently (Section 3.2), align responsibly (Sections 3.4–3.5), and evaluate with robust, multi-angle tests (Figure 3; Section 4).

- Follow-up research directions enabled/suggested
  - Objective design: leverage the rate–distortion view to invent hybrids that tune “rate” (compression) and “distortion” (reconstruction vs. discrimination) for particular capabilities (Section 2.3.3).
  - Data-centric VLMs: new, scalable, interleaved datasets and principled quality metrics that combine text, image, and alignment scores (Section 3.1.4).
  - Grounding: self-supervised or weakly supervised supervision signals beyond boxes (e.g., referential games, precise negative captions) to improve attribute/relation understanding (Sections 3.4, 4.1.6).
  - Evaluation science: tie-robust compositionality tests, LLM-judged open-ended answers with error taxonomies, synthetic environments for counterfactual spatial/temporal reasoning (Sections 4.1, 4.1.6–4.1.8).
  - Safety & privacy: standardized multimodal red teaming suites that connect harms to risks and mitigation playbooks (Section 4.5).
  - Video VLMs: efficient temporal modeling (masking, compressed tokens), datasets with motion-centric supervision, and synthetic physics/ecology to probe world knowledge (Sections 5.4–5.5).

- Practical applications
  - Retrieval and search with contrastive embeddings; captioning and accessible interfaces via generative or LLM-mapped VLMs; document/chart/scene understanding through OCR-augmented pipelines (`Lumos`, `Monkey`); visual assistants aligned with instruction tuning and RLHF (`LLaVA` series); and cautious deployment supported by bias/hallucination/memorization monitoring (Sections 3, 4).

In sum, the paper is a practitioner-oriented map of the VLM landscape: it explains how the major objectives work and relate (Sections 2.2, 2.3.3), when to pick each paradigm (Section 3.3), how to assemble a competitive training pipeline on a budget (Sections 3.1–3.2), and how to evaluate models in ways that surface real capabilities and risks (Section 4), with an outlook toward video (Section 5).
