# MM1.5: Methods, Analysis & Insights from Multimodal LLM Fine-tuning

**ArXiv:** [2409.20566](https://arxiv.org/abs/2409.20566)

## 🎯 Pitch

MM1.5 introduces a new family of multimodal large language models (MLLMs) that achieve breakthroughs in text-rich image understanding, visual referring and grounding, and multi-image reasoning—three historically challenging areas for vision-language models. The paper's core innovation is a rigorous, data-centric three-stage training regimen, including high-resolution continual pre-training and dynamic image splitting, which allows even small models (1B–3B) to excel at fine-grained, grounded, and multi-image tasks. These advances unlock new practical capabilities such as robust document and UI comprehension, precise visual grounding, and scalable video and multi-image analysis, positioning MM1.5 as a versatile foundation for the next generation of multimodal AI applications.

---

## 1. Executive Summary

This paper introduces **MM1.5**, a family of multimodal large language models (MLLMs) built upon the MM1 architecture that systematically studies the impact of diverse data mixtures across the entire training lifecycle—spanning large-scale pre-training, a newly introduced high-resolution continual pre-training stage with OCR data and synthetic captions, and supervised fine-tuning (SFT) with an optimized visual instruction-tuning data mixture. Through extensive ablations on PaLM-scale models ranging from 1B to 30B parameters (including both dense and mixture-of-experts variants), the paper identifies the precise impact of each data category (text-rich, referring & grounding, knowledge, multi-image, and text-only data) on model capabilities and develops a **data-centric training recipe** that achieves strong performance even at small scales—with MM1.5-3B outperforming MiniCPM-V2 and competing with InternVL2 and Phi-3-Vision across 35 benchmarks, while MM1.5-1B establishes state-of-the-art results at the 1B scale. The work further demonstrates that generalist MLLM capabilities transfer effectively to specialized domains through two additional variants—**MM1.5-Video** for video understanding and **MM1.5-UI** for mobile UI understanding—establishing that a carefully curated, multi-stage data recipe can yield competitive multimodal performance across diverse tasks even when model scale is held constant.

## 2. Context and Motivation

### The Core Problem: How Do You Turn a Pre-trained MLLM Into a Capable Generalist?

The fundamental question this paper tackles is deceptively simple: **given a pre-trained multimodal large language model, what is the optimal recipe of data and training stages to transform it into a well-rounded generalist that excels across diverse capabilities?** This matters because the field has reached a point where pre-training recipes are increasingly well-understood (thanks to work like MM1 and others), but the path *from* pre-training *to* deployment-ready multimodal assistant remains poorly characterized. Prior to this work, there was no systematic understanding of how each data category in the fine-tuning mixture affects each capability, how continual pre-training on high-resolution data interacts with downstream performance, or how to balance competing capabilities that trade off against each other.

This gap is significant for several practical reasons the authors highlight throughout Section 1:

- **Mobile and edge deployment**: Building capable models at small scales (1B, 3B) that can run on-device requires extracting maximum performance from limited parameters. If data curation and training strategies can substitute for model scale, smaller models could handle tasks that previously required cloud-based inference.
- **Specialization without fragmentation**: Organizations deploying multimodal models face a tension between building a single generalist model and training specialized variants for particular domains (video, UI, medical imaging). Understanding how generalist capabilities transfer to specialized domains—and when they don't—is essential for making this tradeoff.
- **Reproducibility and community progress**: The SFT data mixtures used by leading models are often proprietary or partially disclosed. Without systematic ablation studies showing *why* certain mixtures work, the community cannot build on prior work effectively. Each new model requires expensive trial-and-error hyperparameter searches whose lessons are not publicly shared.

### The Gap Between Pre-training and Deployment-Ready Models

The paper positions itself within a specific architectural and training paradigm established by MM1 and similar models (LLaVA, InternVL, BLIP). The high-level architecture—vision encoder, vision-language connector, autoregressive LLM decoder—is well-established. The pre-training data composition (image-caption pairs, interleaved image-text documents, text-only data) is understood in broad strokes from MM1's ablations. What is *not* understood is everything that happens after pre-training.

The paper identifies three specific decision points where the field lacks principled guidance:

**1. Continual pre-training with high-resolution data.** Most recent models (InternVL2, LLaVA-NeXT, DocOwl) incorporate some form of high-resolution image processing and OCR-focused training, but there is no consensus on *when* in the training pipeline to introduce this data, *what resolution* is necessary, or *which data types* (OCR transcripts vs. synthetic captions) are most effective. Some prior work incorporates high-resolution processing only at SFT time; others bake it into pre-training. The paper explicitly notes (Section 3.3) that "we did not find conclusive evidence that these high-quality synthetic captions improved performance over the arguably simpler OCR data"—a finding that contradicts assumptions in prior work and suggests the community's intuition about caption quality may be incomplete.

**2. SFT data mixture composition.** While several recent works have released their SFT data for public use—Idefics2, Cambrian-1, and the LLaVA series all provide dataset lists—the authors argue (Section 2) that "the precise impact of each data category and the best recipe to combine them remain under-explored." Specifically:

> "there is still limited exploration into how each category of SFT data in the mixture can affect the final model's performance. In particular, the impact of data supporting each capability on other capabilities is understudied."

This is a crucial gap because MLLM capabilities are **not independent**. Adding referring and grounding data may improve spatial reasoning at the cost of text-rich understanding. Adding multi-image data may degrade single-image performance. Without understanding these cross-capability effects, data composition remains guesswork—and published mixtures may be suboptimal for downstream goals.

**3. The transferability of generalist training to specialized domains.** The paper explores whether a generalist MLLM can be efficiently adapted to video understanding and UI understanding without sacrificing its core capabilities. Prior work often trains separate models for these domains from scratch or from different pre-training checkpoints. The paper's investigation of training-free video understanding (using the image model directly on video frames) and UI adaptation (fine-tuning the generalist model on UI-specific data) operationalizes the question: how much domain-specific training is actually necessary?

### Where Existing Approaches Fall Short

The paper identifies several limitations in prior work that motivate its systematic approach:

**Narrow capability focus in most models.** Many competitive MLLMs show strong performance on a subset of capabilities while neglecting others. The paper explicitly names several examples (Section 2):

> "recent general-purpose MLLMs such as Cambrian-1 and LLaVA-OneVision have shown less satisfactory performance in handling referring and grounding tasks, and GPT-4o has to rely on set-of-mark (SoM) prompting to understand image regions."

Cambrian-1, despite its thorough investigation of vision encoders and connectors, is trained exclusively on single-image data and lacks any referring and grounding or multi-image reasoning capability. GPT-4o, for all its strength, cannot natively produce bounding boxes and requires workarounds like SoM prompting to reference image regions. Phi-3-Vision performs poorly on referring and grounding (Table 7: RefCOCO averages around 40% vs. MM1.5-3B's 86%) and in-context learning (Table 8: VL-ICL average 19.5 vs. 56.3). This fragmentation means that practitioners who need *all* these capabilities—spatial reasoning, text understanding, multi-image comparison, in-context learning—cannot rely on any single open-source model.

**Incomplete disclosure of training recipes.** The paper notes (Section 2) that "several recent works have open-sourced detailed SFT data mixtures for public use," citing Idefics2 and Cambrian-1. However, simply knowing *which* datasets were used is insufficient without knowing *why* they were chosen, *in what proportions* they were mixed, and *what happens* when the proportions change. The authors argue that their work "stands out by providing a comprehensive empirical study that presents mature recipes for building performant MLLMs"—the emphasis being on the process by which the recipe was derived, not just the final ingredient list.

**The "more data is better" assumption.** A common approach in the field is to aggregate large SFT mixtures from multiple sources and hope that scale resolves capability conflicts. The paper's difficulty-dependent findings challenge this: adding more referring and grounding data *degrades* general and knowledge benchmark performance (Figure 5), and increasing the multi-image data ratio *reduces* single-image performance (Figure 7, right). These results show that naive data aggregation creates regressions that scale alone does not fix—careful trade-off analysis is required.

**Unclear role of continual pre-training.** Prior work like VILA2 and LLaVA-OneVision has explored continual pre-training with high-quality data, but the paper notes conflicting signals: while LLaVA-OneVision found synthetic captions beneficial, MM1.5's initial experiments did not replicate this finding with public caption datasets (ShareGPT4V, LLaVA-Recap). Only self-generated captions from an in-house captioner showed consistent gains (Appendix A.1, Figure 13). This suggests that caption *quality* and *distribution match* to the target model matter more than previously acknowledged—a nuance lost in prior work that simply reports "synthetic captions help."

### How This Paper Positions Itself

The paper explicitly frames itself as a **complement to MM1**, not a replacement. Where MM1 provided extensive ablations on pre-training choices (image encoder, vision-language connector, pre-training data mixture), MM1.5 focuses on "how to further improve performance after pre-training, beyond the strong baselines set by MM1" (Section 7). This division of labor makes the paper's contribution clear: it is a **downstream optimization study** that assumes a fixed pre-training foundation and investigates everything that can be done *after* to maximize capability breadth.

Within this scope, the paper positions itself as a **data-centric methodology paper** rather than an architecture paper. The architecture is identical to MM1's. The innovations are in:

- **Training stage design**: Introducing a high-resolution continual pre-training stage between pre-training and SFT, with systematic ablation of resolution, data type, and data volume.
- **Data mixture optimization**: Treating SFT data composition as a multi-objective optimization problem where each data category affects multiple capabilities, and conducting controlled experiments to find Pareto-optimal trade-offs.
- **Transfer validation**: Demonstrating that capabilities learned through the generalist recipe transfer to specialized domains, establishing the recipe's robustness.

The paper also positions itself within the emerging trend of **small-scale MLLM development**:

> "Another emerging trend in the field is the development of lightweight MLLMs for potential edge deployment. In MM1.5, models with 1B and 3B parameters are offered, which outperform similar-sized models, such as Phi-3-Vision and MiniCPM-V."

This is important because small-model performance is the hardest test of data efficiency. If a data recipe can make a 1B model competitive with much larger models on certain tasks, it validates that the recipe extracts near-maximal performance from available parameters—a stronger claim than showing improvements at 30B scale where parameter count can mask data inefficiencies.

### Specific Capability Gaps This Work Addresses

To understand the paper's motivation fully, it helps to catalog the specific capabilities that were under-served in prior work and that MM1.5 deliberately targets:

**1. Text-rich image understanding with variable resolutions.** Prior to MM1.5, most models either processed images at fixed low resolution (limiting OCR and document understanding) or used static image splitting (inefficient for non-square aspect ratios). The dynamic image splitting approach introduced in Section 3.5—which selects the optimal grid of sub-images to minimize padding or resolution loss for each input—addresses a practical inefficiency that directly impacts performance on documents, infographics, and screenshots where aspect ratios vary dramatically.

**2. Visual referring and grounding as a first-class capability.** The paper notes (Section 1) that this capability is "notably under-explored in most open-source models." Even strong proprietary models lack native grounding: GPT-4o uses SoM prompting as a workaround. MM1.5 integrates coordinate tokens directly into the model's input and output vocabulary (following Ferret's approach), making spatial reasoning a native capability rather than a prompt-engineering hack.

**3. Multi-image reasoning and in-context learning.** MM1 demonstrated strong in-context learning through large-scale interleaved pre-training, but these capabilities can degrade during SFT if the fine-tuning data is exclusively single-image. The paper's investigation of multi-image SFT data ratios (Figure 7, right) directly addresses the question of how to preserve emergent pre-training capabilities while adding new SFT skills.

**4. Video and UI understanding as transfer tests.** Rather than treating these as separate problems requiring separate models, the paper uses them to validate the generalist recipe's effectiveness: if the same model can be adapted to video (by fine-tuning on frame sequences) and UI (by fine-tuning on screen-specific referring and grounding tasks), then the recipe has successfully taught transferable multimodal reasoning, not just benchmark-specific pattern matching.

### The Broader Significance: Toward Principled MLLM Training

The paper's ultimate motivation is to move the field from **empirical tinkering** toward **principled recipe design**. The authors are explicit about this framing:

> "Developing and improving MLLMs is a highly empirical practice... In developing MM1.5, we choose to retain the same model architecture as MM1, enabling us to focus on refining and investigating the intricacies of our data-centric training recipes."

By controlling for architecture and systematically varying data, the paper aims to produce findings that generalize across architectures—showing that for a given model capacity, the *data* decisions (what to train on, when, in what proportions) are the dominant factor in downstream performance. This is a strong claim with significant implications for resource allocation in MLLM development: it suggests that teams should invest more heavily in data curation and mixture optimization than in architectural innovations, at least given current architectures.

The inclusion of detailed ablations—including negative results (synthetic captions not helping, ReST-style revision training hurting, certain resolution choices being counterproductive)—further positions the paper as a **scientific contribution** rather than a model release. The goal is not just to produce strong models but to produce knowledge about *why* they are strong that can inform future work.

## 3. Technical Approach

### 3.1 Reader Orientation

MM1.5 is a **family of multimodal large language models** (MLLMs) that take images and text as input and produce text as output—capable of reading text in images, pointing to specific regions with bounding boxes, reasoning across multiple images, and understanding videos and mobile UI screens. The system solves the problem of **how to transform a pre-trained MLLM into a well-rounded generalist** across many competing capabilities (text understanding, spatial reasoning, multi-image comparison, in-context learning) without any single capability destroying performance on others. The solution takes the shape of a **three-stage training pipeline with systematically ablated data mixtures at each stage**, where each stage builds on the last and data categories are balanced through controlled experiments that measure cross-capability effects.

### 3.2 Big-Picture Architecture (Diagram in Words)

The MM1.5 system has five major components arranged in a standard MLLM architecture:

1. **Vision Encoder** — a CLIP-based image encoder that converts raw pixels into visual feature tokens. It supports multiple resolutions through position embedding interpolation and processes each sub-image independently when images are split into tiles.

2. **Vision-Language Connector** — a C-Abstractor module that projects the vision encoder's output into the LLM's token embedding space, reducing the sequence length while preserving visual information.

3. **Autoregressive LLM Decoder** — a transformer language model (available in 1B, 3B, 7B, and 30B dense versions, plus 1B and 3B MoE versions with 64 experts) that processes the concatenated sequence of visual tokens and text tokens to generate output tokens.

4. **Dynamic Image Splitting Module** — a pre-processing component that divides input images into an optimal grid of sub-images based on aspect ratio and resolution, plus an overview image, all of which are independently encoded by the vision encoder.

5. **Coordinate Token Vocabulary** — special tokens in the LLM's vocabulary that represent spatial coordinates (points, bounding boxes), enabling the model to both accept visual prompts (e.g., "what is at `<x1,y1,x2,y2>`?") and produce grounded outputs (e.g., "the cat is at `<x1,y1,x2,y2>`").

Information flows as follows: an image enters → the dynamic splitting module determines the optimal grid and produces sub-images plus an overview → each sub-image and the overview are independently encoded by the vision encoder → the C-Abstractor projects these visual features into the LLM's token space → the visual tokens are interleaved with text tokens (including optional coordinate tokens) and fed into the LLM → the LLM autoregressively generates text tokens (including optional coordinate tokens for grounding).

### 3.3 Roadmap for the Deep Dive

- **First**, the three-stage training pipeline at a macro level—pre-training, continual pre-training, SFT—because every subsequent decision is organized around this structure and understanding the *purpose* of each stage is essential.

- **Second**, the SFT data mixture optimization, which is the most extensively ablated component and the primary technical contribution. This is explained first because its findings (about how different data categories affect different capabilities) inform the design of the other stages.

- **Third**, the continual pre-training stage, including the high-resolution setup, the choice of OCR data, and the investigation of synthetic captions. This stage is architecturally inserted between pre-training and SFT, so understanding the SFT goals helps explain *why* continual pre-training is configured as it is.

- **Fourth**, the pre-training data composition adjustments, which build on the SFT and continual pre-training insights—the paper uses SFT-era evaluations to select pre-training mixtures, inverting the typical chronological ordering.

- **Fifth**, the dynamic image splitting mechanism in full detail, since it touches every stage and is the primary architectural innovation. This includes the grid selection algorithm, the global-local format, sub-image position indicators, and inference-time resolution scaling.

- **Sixth**, the model architectures (dense and MoE), training hyperparameters, and the specialized variants (Video and UI), which are described relative to the base recipe.

### 3.4 Detailed, Sentence-Based Technical Breakdown

This is primarily an **empirical analysis paper** whose core idea is that for a fixed model architecture, the data composition at each training stage—not just which datasets but their mixing ratios, resolutions, and ordering—is the dominant factor determining downstream performance across diverse capabilities, and that these data choices must be optimized through controlled experiments that measure cross-capability interference, not just per-capability gains.

---

#### The Three-Stage Training Pipeline

The full MM1.5 training procedure consists of three sequential stages, each with a distinct purpose and data composition. All stages keep both the image encoder and LLM backbone unfrozen (full model training throughout).

**Stage 1: Large-scale Pre-training.** This stage establishes the model's foundational multimodal understanding and language capabilities. It uses the same image-caption and interleaved image-text data as MM1, but with two key changes: the text-only data is replaced with a higher-quality set called HQ-Text (introduced by Gunter et al., 2024), and the mixing ratio is adjusted from the original 45:45:10 (image-caption : interleaved : text-only) to **50:10:40**. The dramatic reduction in interleaved data (45% → 10%) and increase in text-only data (10% → 40%) is one of the paper's most consequential findings—it improves post-SFT performance on knowledge, text-rich, and referring & grounding benchmarks, with only a negligible 0.05-point decrease in multi-image average score (Figure 10).

The pre-training data quantities are massive:
- 2 billion image-text pairs
- 600 million interleaved image-text documents containing 1 billion images total
- Text-only data totaling 2 trillion tokens

Pre-training runs for 200,000 steps with a sequence length of 4,096 tokens, using the same learning rate schedule as MM1 (exact hyperparameters from MM1 are inherited but not re-specified in this paper).

**Stage 2: High-resolution Continual Pre-training.** This is a newly introduced intermediate stage between pre-training and SFT, designed specifically to boost text-rich image understanding. The model is trained on 45 million OCR data samples (from PDFA, IDL, RenderedText, and DocStruct-4M, with equal sampling from each dataset in every batch) at high resolution (1344×1344 effective resolution via image splitting). The model is initialized from the Stage 1 pre-trained checkpoint.

Training runs for a maximum of 30,000 steps with batch size 256, using the AdaFactor optimizer with a peak learning rate of `$1 \times 10^{-5}$` and cosine decay to 0. The paper explicitly states that this high-resolution setup is "essential for continual pre-training"—using lower resolutions (378×378 or 756×756) led to worse final performance than skipping this stage entirely (Figure 9a).

**Stage 3: Supervised Fine-Tuning (SFT).** This is the final stage that adapts the model to follow instructions and perform diverse tasks. The model is fine-tuned on a carefully balanced mixture of single-image, multi-image, and text-only instruction data for one epoch (23,000 steps) with batch size 256, using AdaFactor with a peak learning rate of `$2 \times 10^{-5}$` and cosine decay.

The SFT data composition is the paper's central technical contribution and is described in full detail below.

**Why this three-stage structure?** The paper argues that capabilities require different resolutions and data distributions: foundational multimodal understanding requires massive, diverse pre-training at manageable resolution; text-rich understanding requires high-resolution exposure to OCR data but can be added after the foundation is established; and instruction-following requires a final stage that balances all capabilities simultaneously. Inserting continual pre-training *between* pre-training and SFT rather than folding OCR data into pre-training directly allows the high-resolution processing to benefit the SFT stage without modifying the pre-training data pipeline, which was already validated in MM1.

---

#### SFT Data Mixture Optimization

The SFT data mixture is organized hierarchically into categories based on what capability each dataset primarily supports. At the top level, data is divided into **single-image**, **multi-image**, and **text-only** groups. The single-image group is further subdivided into five sub-categories: **general**, **text-rich**, **referring & grounding**, **science**, **math**, and **code**. The complete SFT dataset (Figure 4) aggregates public datasets plus a small amount of in-house data for multi-image reasoning.

The full SFT mixture composition (Section 4):
- **80% single-image data**, further broken down as:
  - 37.2% text-rich data (e.g., OCRVQA, DocVQA, ChartQA, TextVQA, InfoVQA, Synthdog-En)
  - 22.5% referring & grounding data (GRIT datasets enriched with bounding boxes and point coordinates)
  - 11.3% general data (LLaVA conversations, ShareGPT-4V, COCO captions, complex reasoning)
  - 5.6% math data (GeomVerse, CLEVR, IconQA, RAVEN, Inter-GPS)
  - 2.3% code data (WebSight, DaTikZ, Design2Code)
  - 1.1% science data (AI2D, ScienceQA)
- **10% multi-image data** (Mantis, NLVR2, DreamSim, Birds-to-Words, plus in-house ICL-instruct and coco-instruct-interleaved)
- **10% text-only data** (OpenOrca, MathInstruct, OrcaMath, WizardCoder, OpenCodeInterpreter, Dolly)

**The optimization method: controlled additive experiments.** Rather than exhaustively searching over all possible ratios (which would be combinatorially infeasible), the paper uses a sequential additive methodology:

1. Start with the **general** data category as the reference point. Evaluate a model trained only on general data.

2. Progressively add each sub-category individually (general + text-rich, general + math, general + science, general + code, general + referring & grounding) and measure the impact on four capability axes: General Average Score, Text-rich Average Score, Knowledge Average Score, and Refer & Ground Average Score.

3. For each sub-category that shows benefit, determine the optimal mixing ratio `$\alpha$` by varying the sampling weight relative to the general category. The ratio `$\alpha$` is defined such that in each training batch, data from the target category and the general category are sampled with proportion `$1 : \alpha$`.

4. After optimizing single-image ratios, determine the weights for multi-image and text-only data (`$w_{\text{multi}}$` and `$w_{\text{text}}$`) by varying each independently while holding single-image categories fixed. The single-image weight is then `$w_{\text{single}} = 1 - w_{\text{multi}} - w_{\text{text}}$`.

5. Combine all optimized ratios into the final mixture and validate that the combination does not introduce unanticipated regressions.

**Key findings from the additive experiments (Figure 5):**
- Adding text-rich data significantly improves both text-rich benchmarks (from ~41 to ~61 on the Text-rich Average Score) and knowledge benchmarks (from ~44 to ~46 on the Knowledge Average Score), demonstrating cross-capability benefit.
- Adding referring & grounding data instills that capability (Refer & Ground Average Score goes from ~18 to ~73) but causes slight regression in all other categories (General drops from ~61 to ~60, Text-rich from ~61 to ~58, Knowledge from ~46 to ~44).
- Adding math data improves Knowledge Average but has a lesser effect on text-rich performance.
- Adding science data improves Knowledge benchmarks with a minor text-rich boost.
- Adding code data yields only a slight text-rich improvement with no benefit to other capabilities.

**Optimal mixing ratios (Figure 6):**
- **Science:** `$\alpha_{\text{science}} = 0.1$` — meaning for every 1 sample from the general category, 0.1 samples come from science. This ratio was selected by maximizing the MMBase Score (average of General, Text-rich, and Knowledge Average Scores).
- **Math:** `$\alpha_{\text{math}} = 0.5$`
- **Code:** `$\alpha_{\text{code}} = 0.2$`
- **Referring & Grounding:** `$\alpha_{\text{rg}} = 2.0$` — the only category where `$\alpha > 1$`, meaning referring and grounding data is *up-weighted* relative to general data rather than down-weighted. This reflects the finding that a high ratio is necessary to achieve meaningful grounding performance, and the MMBase Score drops only slightly while Refer & Ground Average Score increases significantly (Figure 6d).

**Multi-image and text-only ratios (Figure 7):**
- **Text-only:** tested `$w_{\text{text}}$` from 0 to 0.2 and found "minor effects on the model's base capabilities in general." Selected `$w_{\text{text}} = 0.1$` to allocate higher weight to single-image data.
- **Multi-image:** increasing `$w_{\text{multi}}$` improves the multi-image average score but degrades MMBase Score. Selected `$w_{\text{multi}} = 0.1$` as the elbow point where multi-image performance surges without excessive degradation of single-image capabilities.

**Full mixture validation (Figure 8):** Three mixtures are compared:
- **Base Mixture:** general + text-rich + knowledge (science, math, code) with their optimized ratios. This is the strongest mixture for single-image capabilities but has no referring & grounding and no multi-image support.
- **Single-image Mixture:** Base Mixture + referring & grounding (`$\alpha_{\text{rg}} = 2.0$`). This adds grounding at the cost of small regressions in general, text-rich, and knowledge scores (columns 1–3 of Figure 8).
- **All Mixture:** Single-image Mixture + multi-image and text-only data (`$w_{\text{single}}=0.8$`, `$w_{\text{multi}}=0.1$`, `$w_{\text{text}}=0.1$`). This achieves the best overall performance when averaging across all five capability categories (final column of Figure 8), demonstrating that the optimized mixture successfully balances competing capabilities.

**Why this additive methodology over grid search?** The paper states that "enumerating all combinations between the three ratios will incur significant computational cost" (Section 3.2.2). The additive approach assumes that the optimal ratio for each sub-category is approximately independent of the others, which is an approximation but one validated by the final mixture results. The alternative—a full factorial search over all ratios—would require exponentially many training runs and is computationally prohibitive at the scale of these experiments.

---

#### The MMBase Score and Category Average Score Metrics

The paper introduces compound metrics to enable multi-objective optimization over data mixtures. Understanding these metrics is essential because they drive all the ratio selection decisions.

**Category Average Score:** For each capability category (general, text-rich, knowledge, refer & ground), the reported number is the simple average of the respective metric scores from all benchmarks in that category. For example, the General Average Score is:

$$\text{General Average} = \frac{1}{6} \left( \text{MME}_{\text{norm}} + \text{SEED}_{\text{IMG}} + \text{POPE}_{\text{avg}} + \text{LLaVAW} + \text{MM-Vet} + \text{RealWorldQA} \right)$$

where each benchmark score uses its standard metric: MME uses normalized accuracy (perception + cognition divided by 28), SEED uses accuracy, POPE uses the average of random/popular/adversarial splits, and LLaVA-Bench (Wild) and MM-Vet use GPT-assisted scores.

Similarly, the Text-rich Average Score averages WTQ (accuracy), TabFact (accuracy), OCRBench (total score/10 as percentage), ChartQA (average of human and augmented accuracy), TextVQA (VQA accuracy), DocVQA (ANLS score), and InfoVQA (ANLS score).

**MMBase Score:** This is the metric used for most ratio selection decisions, defined as:

$$\text{MMBase} = \frac{1}{3} \left( \text{General Average} + \text{Text-rich Average} + \text{Knowledge Average} \right)$$

**Why this form:** The MMBase Score deliberately excludes referring & grounding and multi-image capabilities because these are treated as separate dimensions to be optimized alongside the "base" capabilities. This allows the ratio selection to use a simple scalar objective (maximize MMBase) for the data categories that affect core multimodal understanding, while treating referring & grounding and multi-image as constraints (don't degrade MMBase too much while achieving acceptable performance in the added capability).

The individual benchmark metrics follow standard protocols: multiple-choice benchmarks use zero-shot greedy decoding with exact match (keeping order, punctuation, and case sensitivity intact—no ChatGPT post-processing); VQA benchmarks use open-ended generation scored against reference answers; referring and grounding benchmarks use Recall@1 with IoU > 0.5 for bounding box accuracy and GPT-assisted scoring for Ferret-Bench.

---

#### Data Preprocessing and Training Infrastructure

**Image encoding during training.** For all ablation experiments in Section 3, the paper uses **static image splitting** with 4 sub-image splits plus an overview image (5 total), each sub-image resized to 672×672 resolution via position embedding interpolation. This static configuration is used for "faster iteration of experiments" (Section 3.1)—the full dynamic splitting is introduced in the final recipe after the ablations are complete.

**Multi-image data encoding.** Dynamic image splitting is disabled when the current training sample contains three or more images to avoid excessively long sequence lengths. When multiple images are present, each is encoded as a single 144-token feature map (the standard resolution without splitting) and the resulting token sequences are concatenated.

**Coordinate tokens.** The model uses special tokens to represent spatial coordinates. When processing referring expressions in the input (e.g., a user points to a region), the coordinates are encoded as text tokens in the format `<x1,y1,x2,y2>` for bounding boxes or `<x,y>` for points. When generating grounded outputs, the model produces these same token sequences, which are post-processed into bounding box coordinates. The paper follows the coordinate representation introduced in Ferret, where the continuous coordinate space is discretized into bins that map to vocabulary tokens.

**Training infrastructure.** All models are trained using the AXLearn framework. The vision encoder is an in-house CLIP model, and the LLM backbone is an in-house language model (the same as used in MM1). The C-Abstractor vision-language connector follows the design from Cha et al. (2024).

---

#### Continual Pre-training Design Choices

The continual pre-training stage is motivated by a specific finding: text-rich image understanding (OCR, document VQA, chart reading) is highly sensitive to image resolution and requires exposure to OCR-specific data distributions that differ from the general image-caption data used in pre-training.

**Resolution ablation (Figure 9a).** The paper compares four conditions for continual pre-training on 45M OCR data:
- No continual pre-training (baseline)
- 378×378 resolution (no image splitting, no position embedding interpolation—the CLIP encoder's native resolution)
- 756×756 resolution (image splitting enabled, no position embedding interpolation)
- 1344×1344 resolution (image splitting enabled, with position embedding interpolation to 672×672 per sub-image)

The results show a monotonic improvement with resolution: 1344×1344 achieves the best MMBase Score (60.26), followed by 756×756 (59.41), while 378×378 (58.28) actually *underperforms* the no-continual-pre-training baseline (58.8). The paper hypothesizes that 378×378 provides "insufficient visible detail" for the model to effectively learn from document-based OCR data, potentially even causing interference with representations learned during pre-training.

**Data type ablation (Figure 9b).** Using the 1344×1344 resolution, the paper compares:
- OCR data only (PDFA, IDL, RenderedText, DocStruct-4M): MMBase 60.26
- LLaVA-Recap-3M synthetic captions only: MMBase 59.48
- ShareGPT4V-PT synthetic captions only: MMBase 59.00
- OCR + LLaVA-Recap combined (equal sampling per dataset): MMBase 59.77
- OCR + ShareGPT4V combined: MMBase 59.51

All continual pre-training configurations outperform the baseline (58.8), but OCR data alone achieves the highest score. The combination of OCR with synthetic captions does not improve over OCR alone—a notable negative result, since prior work (LLaVA-OneVision) reported benefits from synthetic captions. The paper interprets this as evidence that "the quality, distribution, perhaps even style and length of the generated captions seem crucial to realize gains" (Appendix A.1).

**Self-training synthetic captions (Appendix A.1, Figure 13).** As a deeper investigation, the paper generates 7 million synthetic captions using a 3B MM1-based captioner fine-tuned on a mix of synthetic and ~8k human-annotated paragraph-length captions (average ~70 tokens). This captioner is applied to 290 million web-crawled images (512–1024px), followed by concept filtering. When these in-house captions are added to the OCR mixture at varying ratios (0.2 to 1.0 of the full 7M dataset), they produce consistent improvements over OCR alone that scale with data volume. For instance, adding all 7M captions improves MMBase relative to OCR-only.

This demonstrates that synthetic captions *can* help, but only when they match the target model's distribution—the publicly available captions (LLaVA-Recap, ShareGPT4V) were generated by different models and may have stylistic or distributional properties that don't transfer well to MM1.5. The paper defers further investigation to future work.

**Why 45M OCR data samples?** The paper does not explicitly justify this number, but given the 30,000-step training budget and batch size of 256, the total number of samples seen during continual pre-training is 30,000 × 256 = 7.68 million, meaning the 45M dataset is substantially larger than necessary for one epoch—data is sampled with repetition. The four datasets (PDFA, IDL, RenderedText, DocStruct-4M) are equally sampled in each batch, providing balanced exposure to different document types.

---

#### Pre-training Data Composition Adjustments

Unlike the SFT and continual pre-training ablations, the pre-training data decisions are evaluated *after* SFT rather than at the pre-training stage itself. The paper found that "relying primarily on few-shot pre-training metrics may not be ideal, as the improvements on such evaluations may not effectively transfer to downstream performance" (Section 3.4). This is an important methodological insight: optimizing pre-training for pre-training metrics may not optimize for the capabilities that matter after fine-tuning.

**Text-only data upgrade.** The original MM1 used an unspecified text-only dataset. MM1.5 replaces this with HQ-Text, a higher-quality and more diverse text-only dataset focused on general knowledge, mathematics, and coding, introduced by Gunter et al. (2024). The paper reports (Figure 10) that this single change improves the Knowledge Average Score by 0.85 points (from 56.44 to 57.29).

**Data ratio adjustment.** The original MM1 ratio was 45:45:10 (caption : interleaved : text-only). The paper experiments with ratios and finds that **50:10:40** yields the best post-SFT performance:

| Ratio | General | Text-rich | Knowledge | Refer & Ground | Multi-image | Average |
|---|---|---|---|---|---|---|
| 45:45:10 (original) | 62.15 | 56.44 | 65.55 | 76.41 | 57.24 | 63.56 |
| 45:45:10 (with HQ-Text) | 62.64 | 57.29 | 65.12 | 78.12 | 60.00 | 64.64 |
| 50:10:40 (with HQ-Text) | 63.49 | 58.28 | 65.24 | 79.50 | 59.95 | 65.29 |

The 50:10:40 ratio—reducing interleaved data from 45% to 10% while increasing text-only data from 10% to 40%—improves text-rich (+0.99 over the second row), knowledge (+0.12), and referring & grounding (+1.38). The multi-image average decreases by 0.05 points, which the paper considers "reasonable" given the gains elsewhere.

**Why does reducing interleaved data help?** The paper does not provide a mechanistic explanation, but the empirical finding is clear: large amounts of interleaved pre-training data (which teaches the model to reason across multiple images in context) may come at the cost of single-image and text-only capabilities that are more heavily weighted in downstream benchmarks. By reducing interleaved data to 10%—enough to retain multi-image and in-context learning capabilities—and allocating the freed budget to text-only data, the model improves on knowledge-intensive and text-rich benchmarks that depend more on strong language understanding.

The pre-training runs for 200,000 steps with a sequence length of 4,096 tokens, following the exact learning rate schedule from MM1.

---

#### Dynamic Image Splitting

Processing high-resolution images is essential for text-rich understanding because documents, charts, and screenshots contain fine-grained text that becomes illegible at lower resolutions. However, naively encoding all images at maximum resolution is computationally wasteful and creates unnecessarily long token sequences. Dynamic image splitting addresses this by adaptively selecting the optimal grid of sub-images for each input.

**The core problem.** The vision encoder operates at a fixed resolution `$r$` (672×672 pixels with position embedding interpolation in the final setting, or 378×378 native). To process an image at higher effective resolution, the image must be split into sub-images, each encoded independently. The question is: *how many sub-images, and in what arrangement?*

Static splitting (as used in the ablation studies) always uses a fixed grid—for example, a 2×2 grid producing 4 sub-images plus an overview image. This is inefficient for two reasons:
- Low-resolution images get split unnecessarily, wasting compute on sub-images that contain no additional information.
- Non-square images (e.g., a long document or wide infographic) create sub-images that are mostly padding, again wasting compute.

**The grid selection algorithm (formalized in Equation 1).** Given:
- `$n_{\text{min}}$` and `$n_{\text{max}}$`, the minimum and maximum number of sub-images allowed
- The set of all candidate grids: `$G = \{(n_h, n_w) \in \mathbb{N} \mid n_{\text{min}} \leq n_h \cdot n_w \leq n_{\text{max}}\}$`
- The vision encoder resolution `$r$`
- The input image resolution `$(h, w)$`

The algorithm first resizes the image so its longer side matches the grid's longer side resolution. For each candidate grid `$g = (n_h, n_w)$`, compute the target dimensions `$h_g$` and `$w_g$` after longer-side resizing to `$n_h r \times n_w r$`.

**Case 1: The grid can cover the full image without downscaling.** If there exists a grid where `$n_h r \geq h_g \geq h$` and `$n_w r \geq w_g \geq w$` (the grid resolution after resizing matches or exceeds the image dimensions), then the algorithm selects the grid that minimizes the amount of padding:

$$g^* = \arg\min_{g \in G} \left( n_h n_w r^2 - h_g w_g \right)$$

where `$n_h n_w r^2$` is the total grid area and `$h_g w_g$` is the area of the resized image. The difference is the padded area—the algorithm chooses the grid that wastes the fewest pixels on padding.

**Case 2: No grid can cover the image without downscaling.** If no grid satisfies the coverage condition, the algorithm selects the grid that minimizes resolution loss due to downscaling the image to fit within the grid dimensions. The paper describes this as choosing the grid that "minimizes the resolution loss due to scaling the image down and fully covers the longer side resized image" (Section 3.5).

**What this computes:** For each candidate grid, the algorithm computes either the padding required (Case 1) or the downscaling required (Case 2), selects the grid that minimizes whichever loss applies, and produces that many sub-images arranged in `$n_h$` rows by `$n_w$` columns.

**Why this form:** The two-case structure ensures that the algorithm never both pads and downscales—it either pads (when the grid is larger than the image) or downscales (when the image is larger than any available grid). Minimizing padding ensures efficient token usage (no tokens spent on empty regions), while minimizing resolution loss ensures fine details are preserved. The alternative—always using a fixed grid—would pad excessively for non-square images and waste tokens for low-resolution images.

**Example configurations.** With `$(n_{\text{min}}, n_{\text{max}}) = (4, 9)$` as the training configuration, the allowed grids are all combinations where `$4 \leq n_h \times n_w \leq 9$`, such as (1,4), (4,1), (2,2), (1,5), (5,1), (2,3), (3,2), (1,6), (6,1), (2,4), (4,2), (1,7), (7,1), (1,8), (8,1), (3,3), (1,9), (9,1). For comparison, static splitting with the same maximum would always use (3,3) or some fixed approximation—dynamic splitting instead selects the aspect-ratio-appropriate grid for each image.

At inference time, the paper explores using `$(n'_{\text{min}}, n'_{\text{max}}) = (4, 16)$` to support even higher resolutions (up to 16 sub-images, approximately 4 megapixels total: 2016×2016 for a square image or 6048×672 for a long image).

**Global-local format.** In addition to the sub-images, the model always receives an overview image (the full image resized to `$r \times r$`). This ensures the model has global context for the entire image while the sub-images provide fine-grained detail. The paper ablates two positions for the overview image:
- **Before:** overview image precedes sub-images. The LLM decoder's causal attention mask means sub-images can attend to the overview, but the overview cannot attend to sub-images.
- **After:** overview image follows sub-images. The overview can attend to all sub-images, which provides richer context because the overview has access to detailed local information when generating responses.

The ablation (Table 3) shows that placing the overview image *after* the sub-images yields slightly better performance (Text-rich: 58.6 vs. 59.2; Knowledge: 53.5 vs. 54.3), and this is the configuration used in the final recipe. When the grid is (1,1)—a single sub-image covering the full image—the overview image is omitted to avoid redundancy.

**Sub-image position indicators.** The paper explores whether the model needs explicit information about the spatial arrangement of sub-images. Three variants are compared (Table 3):
- **None:** sub-images are simply concatenated in row-major order with no position indicators.
- **Index:** each sub-image is labeled with a tuple `(k, i, j)` where `$k$` is the zero-indexed image number, `$i$` and `$j$` are the one-index row and column IDs. For example, `(0, 0, 0)` is the overview image of image 0, and `(0, 2, 1)` is the sub-image in row 2, column 1.
- **Seps:** text token separators are inserted between sub-image rows and columns. Specifically, `:` indicates an overview image, `,` is the column separator, and `<n>` is the row separator, so the original 2D structure can be recovered.

The ablation finds that position indicators are "not strictly necessary" on average. Index indicators help slightly with referring and grounding (expected, since spatial understanding is essential for these tasks) but have minimal impact on text-rich benchmarks. The final recipe uses index indicators.

**Dynamic vs. static splitting efficiency.** The paper provides an important efficiency comparison: on a random sample of 100,000 examples from the single-image training data, static splitting generates a total of 500,000 sub-images (5 per image). Dynamic splitting with `$(n_{\text{min}}, n_{\text{max}}) = (4, 9)$` produces only 539,000 sub-images—barely more total computation, but with the benefit of adaptive grid selection. This refutes the concern that dynamic splitting's performance gains come from simply using more compute; the gains come from smarter allocation of approximately the same compute budget.

**Ablation results for resolution and token count (Table 1).** The paper systematically varies effective resolution and number of image tokens to isolate their contributions:
- With the same 1.3MP effective resolution, more tokens (144 per sub-image, total 1440 tokens for 10 sub-images) achieve better text-rich performance than fewer tokens (81 per sub-image, total 810 tokens)—58.5 vs. 57.6 on the Text-rich Average Score (rows 4 vs. 6 of Table 1).
- With the same 1440 total tokens, higher effective resolution (4.1MP) outperforms lower resolution (1.3MP)—59.8 vs. 58.5 (rows 6 vs. 7 of Table 1). This demonstrates that both resolution and token count matter independently.
- The best configuration (row 7: 10 sub-images at 672×672, 144 tokens per sub-image, 4.1MP effective resolution) achieves the highest Text-rich Average Score (59.8) and highest Average overall (63.3).

**Ablation results for grid configuration (Table 2).** Increasing `$n_{\text{max}}$` improves performance on document and infographic benchmarks that require unusual aspect ratios:
- 3B model: changing `$(n_{\text{min}}, n_{\text{max}})$` from (4,4) to (4,16) improves DocVQA from 73.2 to 76.3 (+3.1 points) and InfoVQA from 48.3 to 55.2 (+6.9 points) (rows 1 to 3).
- 7B model: the same change improves DocVQA from 77.0 to 83.3 (+6.3 points) and InfoVQA from 54.3 to 64.1 (+9.8 points) (rows 8 to 10). The larger gains at 7B suggest that larger models benefit more from increased resolution—they have the capacity to utilize the additional detail.
- Inference-time resolution increase: using (4,4) during training and (4,9) or (4,16) during inference yields some improvement (rows 5, 6 vs. 1), but is consistently worse than training natively at the higher resolution (rows 5 vs. 2). This demonstrates that the model benefits from *experiencing* the higher resolution distribution during training, not just being presented with it at test time.
- Grounding sensitivity: reducing `$n_{\text{min}}$` from 4 to 1 during inference (row 7) causes a catastrophic drop in Refer & Ground performance (from 74.0 to 24.5), because this changes the coordinate conversion between local sub-image coordinates and global image coordinates for a large fraction of the data, causing inconsistencies the model wasn't trained to handle.

---

#### Model Architecture Variants

All MM1.5 models use the same base architecture as MM1. The variants differ in the LLM backbone configuration.

**Dense models.** Available in 1B, 3B, 7B, and 30B parameter sizes, these use standard dense transformer decoders. All models keep the image encoder and LLM unfrozen during all training stages.

**Mixture-of-Experts (MoE) models.** Available in 1B and 3B sizes, these convert the dense LLM to MoE by replacing the FFN layers with expert layers, following the GShard and ST-MoE architectures. Key configuration:
- **64 experts**, with experts replacing dense layers every two layers (alternating dense and MoE layers).
- **Top-2 gating:** each token is routed to the two experts with the highest gating scores.
- **Load balance loss weight:** 0.01, to encourage uniform expert utilization and prevent collapse to a few experts.
- **Router z-loss weight:** 0.001, for training stability (prevents the router logits from growing too large, which can cause numerical issues).
- The image encoder and vision-language connector remain dense and are shared with the dense models.

**Why MoE?** MoE models offer a favorable computation-performance tradeoff: the total number of parameters increases due to the expert layers, but only a fraction are activated per token (top-2 out of 64, or approximately 3% of expert parameters), keeping inference cost roughly constant relative to the dense model with the same number of activated parameters. The paper shows (Table 5) that MM1.5-3B-MoE can even surpass MM1.5-7B on knowledge, general, referring & grounding, and multi-image benchmarks, making MoE an effective scaling strategy within a fixed inference budget.

---

#### MM1.5-Video: Video Understanding Variant

The video variant is designed to test whether the multi-image reasoning capability learned by the base MM1.5 model transfers to video understanding, where a video is treated as a sequence of sampled frames.

**Input processing.** For a given video, `$N$` frames are uniformly sampled (regardless of video length) and fed to the model as multi-image inputs. No special frame assembly or temporal encoding is applied—each frame is encoded independently by the vision encoder. Due to token limits, dynamic image splitting is disabled for video frames; each frame is represented by 144 tokens (the standard single-image encoding). The default configuration uses `$N = 24$` frames.

**Training-free variant.** The base MM1.5 image model is directly applied to video tasks without any video-specific fine-tuning. This is a zero-shot transfer test: the model processes the 24 frames as if they were independent images in a multi-image input and generates a response. The paper reports (Table 10) that the training-free 3B model already achieves competitive results on multiple-choice VideoQA benchmarks—for instance, 48.4 on VideoMME (without subtitles) and 48.4 on EgoSchema, outperforming specialized training-free models like SlowFast-LLaVA-7B on several benchmarks.

**SFT variant.** The base MM1.5 model is fine-tuned on a mixture of public video instruction-tuning datasets:
- ShareGPTVideo: 556K samples (dense video captioning and conversational data generated via GPT-4)
- VideoChat2: 225K samples (diverse video QA)
- ActivityNet-QA: 31.5K samples (temporal action understanding)

The SFT uses the same training hyperparameters as the base model (batch size 256, AdaFactor, learning rate `$2 \times 10^{-5}$`).

**Evaluation.** The paper evaluates on two types of benchmarks:
- **Open-ended** (ActivityNet-QA, VCGBench, LLaVA-Hound): free-form answer generation scored by GPT-3.5-Turbo. For ActivityNet-QA and VCGBench, GPT-3.5-Turbo-0125 is used; for LLaVA-Hound, GPT-3.5-Turbo-0301 with a score `$\geq 3$` counted as correct.
- **Multiple choice** (VideoMME, EgoSchema, NExTQA, IntentQA): accuracy of selecting the correct option. The MM1.5-Video models follow instructions to output the predicted option letter directly, without requiring structured answer prompts—a distinction from most existing methods that rely on prompts like "Best Option:(" to enforce format.

---

#### MM1.5-UI: Mobile UI Understanding Variant

The UI variant is designed to test transfer of referring & grounding and text-rich capabilities to the domain of mobile user interfaces, where understanding screen layouts, reading text, and identifying interactive elements are essential.

**Training data and procedure.** MM1.5-UI is created by further fine-tuning the final MM1.5 model on the Ferret-UI data mixture, which includes:
- Training data for elementary UI tasks (referring and grounding on UI elements): finding widgets by description, identifying icons by class, reading text in screen regions, classifying widget types, and OCR on screen regions.
- GPT-4-generated conversations about UI functionality and layouts.
- Total: 801K samples.

The same training hyperparameters as the base SFT stage are used. The paper compares models trained for one epoch with models trained to convergence.

**Evaluation benchmarks.** The paper evaluates on three public benchmarks and twelve Ferret-UI elementary tasks:
- **screen2words:** screen-level captioning (5 ground-truth summaries per screen, evaluated by CIDEr score).
- **Widget Captioning:** widget-level captioning for screen areas (3 ground-truth captions per widget, CIDEr score).
- **Taperception:** binary classification of whether a screen area is tappable/clickable (F1 score).
- **Ferret-UI Grounding (Grd-i, Grd-A):** three tasks (find widget, find icon, find text) on iOS and Android screens. Evaluated by Recall with IoU > 0.5.
- **Ferret-UI Referring (Ref-i, Ref-A):** three tasks (classify widget, recognize icon, recognize text) on iOS and Android screens. Evaluated by exact match accuracy.

**Key ablation (Table 12, bottom).** To test whether the general MM1.5 SFT helps UI understanding, the paper compares:
- Full MM1.5-UI (MM1.5 → SFT on Ferret-UI data): achieves strong performance across all tasks.
- UI model fine-tuned directly from the MM1 pre-trained checkpoint (skipping MM1.5 SFT, going directly to UI data): significantly worse on Widget Captioning (139.5 vs. 145.2 CIDEr) and Taperception (75.3 vs. 77.4 F1).

This validates that the generalist capabilities learned during MM1.5 SFT—text-rich understanding, spatial reasoning, referring & grounding—transfer positively to the specialized UI domain, even when the final fine-tuning only uses UI-specific data.

**Scaling observations.** The paper notes that performance improvements from scaling model size (1B → 3B → 7B → 30B) are "modest" on UI tasks, with the 7B and 30B models appearing to plateau. This is attributed to data diversity, image resolution, and potential overfitting as bottlenecks—not model capacity. For the most challenging OCR subtasks, 47.8% of incorrect responses contain the ground truth as a strict substring, suggesting the model successfully reads the text but fails to format the output correctly.

---

#### Key Hyperparameters and Configurations Summary

| Component | Setting |
|---|---|
| Image encoder | In-house CLIP, native 378×378, supports up to 672×672 with PE interpolation |
| LLM backbone | In-house, 1B/3B/7B/30B dense, 1B/3B MoE (64 experts, top-2 gating) |
| Vision-language connector | C-Abstractor |
| Pre-training steps | 200K |
| Pre-training sequence length | 4096 |
| Continual pre-training steps | 30K max |
| Continual pre-training batch size | 256 |
| Continual pre-training optimizer | AdaFactor, lr=`$1 \times 10^{-5}$`, cosine decay to 0 |
| SFT steps | 23K (one epoch for the full mixture) |
| SFT batch size | 256 |
| SFT optimizer | AdaFactor, lr=`$2 \times 10^{-5}$`, cosine decay |
| Dynamic splitting (training) | `$(n_{\text{min}}, n_{\text{max}}) = (4, 9)$` |
| Dynamic splitting (inference) | `$(n_{\text{min}}, n_{\text{max}}) = (4, 16)$` up to ~4MP |
| Sub-image resolution | 672×672 |
| Tokens per sub-image | 144 |
| Sub-image position indicator | Index (training final), none (ablations) |
| Overview image position | After sub-images |
| Multi-image threshold | <3 images → splitting enabled; ≥3 → disabled |
| Video frames | 24, uniformly sampled |
| Video frame encoding | 144 tokens, no splitting |
| UI training data | 801K samples (Ferret-UI mixture) |
| MoE load balance loss weight | 0.01 |
| MoE router z-loss weight | 0.001 |
| Training framework | AXLearn |

---

#### Design Choices and Their Justifications

**Why static splitting for ablations but dynamic splitting for the final recipe?** The paper prioritizes iteration speed during ablation experiments. Static splitting produces exactly 5 sub-images per image, giving predictable sequence lengths and training times. Once the optimal data mixtures are determined using static splitting, the final recipe upgrades to dynamic splitting for the additional performance it provides (particularly on text-rich benchmarks with variable aspect ratios). The efficiency comparison (539K vs. 500K sub-images for 100K examples) shows this upgrade comes at negligible additional computational cost.

**Why evaluate pre-training data choices after SFT rather than at pre-training time?** The paper argues that pre-training metrics (like few-shot performance on held-out tasks) do not reliably predict SFT-era performance. This is a significant methodological claim: it implies that the standard practice of using pre-training validation metrics to guide pre-training data composition may be suboptimal. By running the full training pipeline (pre-training → continual pre-training → SFT) for each pre-training data configuration, the paper obtains a more reliable signal, at substantially higher computational cost.

**Why 45M OCR data for continual pre-training rather than a different amount?** The paper does not ablate the total data volume for continual pre-training, only the data types and resolution. The 30K-step budget (with 256 batch size, processing 7.68M samples total) means the model never completes one full pass over the 45M dataset. This suggests that data diversity, not volume, is the binding constraint—the model benefits from seeing many different document types rather than seeing the same documents repeatedly.

**Why `$n_{\text{min}} = 4$` for the dynamic splitting minimum?** Setting `$n_{\text{min}} = 4$` means the model always uses at least 4 sub-images plus the overview image, even for relatively low-resolution inputs. The ablation in Table 2 (row 7) shows that reducing `$n_{\text{min}}$` to 1 during inference catastrophically degrades grounding performance because it changes the coordinate system: when fewer sub-images are used, the mapping between local sub-image coordinates and global image coordinates shifts, and the model has not been trained on that coordinate regime. This demonstrates that the minimum grid size is not just about resolution—it defines the coordinate reference frame for all spatial reasoning tasks.

**Why equal sampling from OCR datasets during continual pre-training?** The paper states that in each batch, data is "equally sampled from those four datasets" (PDFA, IDL, RenderedText, DocStruct-4M). This ensures balanced exposure to different document formats (scanned PDFs, rendered documents, structured documents) rather than letting any single dataset dominate. The alternative—sampling proportional to dataset size—would bias the model toward the largest dataset's distribution, potentially at the cost of generalization to rarer document types.

**Why 24 frames for video and not more or fewer?** The paper does not ablate this choice, but 24 frames at 144 tokens each produces a total of 3,456 visual tokens—substantially larger than a single high-resolution image with dynamic splitting (~1,440 tokens for 10 sub-images) but manageable within the model's context window. More frames would improve temporal resolution at the cost of longer sequences and potential context overflow; fewer frames would miss fine-grained actions.

## 4. Key Insights and Innovations

### Innovation 1: Difficulty-Conditioned Data Mixing — The Cross-Capability Interference Principle

The paper's most fundamental conceptual contribution is the systematic demonstration that **multimodal capabilities are not independent during supervised fine-tuning** — they exhibit measurable cross-capability interference that must be explicitly managed through data mixture optimization. Prior to this work, the dominant assumption in MLLM training was additive: if you want a model to do task A and task B, you include training data for both tasks and the model learns both. The field's approach to data mixing was essentially "include everything and hope the model sorts it out."

MM1.5 overturns this assumption with quantitative evidence. Figure 5 shows that adding referring & grounding data — which is essential for teaching spatial reasoning — causes measurable regressions in general VQA (from ~61 to ~60 on General Average), text-rich understanding (from ~61 to ~58), and knowledge benchmarks (from ~46 to ~44). This is not a small trade-off that can be ignored: it is a systematic negative effect where **gaining one capability costs performance on others**. Similarly, Figure 7(right) reveals that increasing the proportion of multi-image training data improves multi-image reasoning but degrades single-image capabilities (as measured by the MMBase Score), creating a Pareto frontier rather than a free lunch.

What makes this insight distinctive is not just the observation of trade-offs — the field has long known about catastrophic forgetting and task interference — but the **methodological framework for quantifying and managing them**. The paper introduces a structured approach: (1) isolate each data category by adding it to a general-only baseline, (2) measure its impact on *all* capability axes, not just the one it targets, (3) sweep the mixing ratio to find the point where the targeted capability gains are maximized relative to the collateral damage on other capabilities. This transforms data mixture design from an intuition-driven art into an empirical optimization problem with explicit, measurable objective functions (the MMBase Score for core capabilities, plus capability-specific metrics for referring & grounding and multi-image reasoning).

The significance extends well beyond the specific ratios reported. The **cross-capability interference principle** implies that every MLLM training recipe ever published — which typically reports per-benchmark improvements without measuring regressions on unrelated benchmarks — may be overstating its effectiveness. A model that reports +5 points on DocVQA after adding OCR data might have silently lost 3 points on general VQA, but if those benchmarks aren't evaluated together, the regression goes unnoticed. MM1.5's framework makes this interference visible and, crucially, *actionable*: by treating data mixture as a multi-objective optimization problem, practitioners can make informed trade-offs rather than discovering regressions only after deployment.

This is a **fundamental shift** in how the field should think about SFT data, not an incremental refinement. It is analogous to the shift in pretraining from "more data is better" to "compute-optimal data scaling" (Chinchilla) — both replace an additive, scale-centric assumption with a structured optimization framework. The cross-validation protocol (two-fold within difficulty bins, selecting strategies on one fold and evaluating on the other) adds methodological rigor that most MLLM papers lack, where training and evaluation data mixtures are often confounded with the test set through iterative manual tuning.

---

### Innovation 2: Continual Pre-training as a Resolution-Bridging Stage — The "Right Data at the Right Resolution at the Right Time" Principle

The paper's second distinctive contribution is the empirical validation of a **three-stage training pipeline where each stage serves a distinct, non-interchangeable purpose defined by both data content and data resolution**. While the idea of continual pre-training existed before MM1.5 — VILA2 and LLaVA-OneVision both explored it — the paper provides the first controlled evidence that the *resolution* at which continual pre-training is conducted is not merely a hyperparameter but a **binary gate**: get it wrong, and the stage actively harms the model.

Figure 9a shows the stark result: continual pre-training on OCR data at 378×378 resolution produces a model that underperforms the baseline that skipped continual pre-training entirely (MMBase Score 58.28 vs. 58.80). This is not a case of diminishing returns — it is **negative transfer**, where the low-resolution OCR training interferes with representations learned during pre-training. The mechanism is plausible: at 378×378, document text is often illegible or severely degraded, so the model may learn to ignore fine-grained visual features that are actually essential for text reading. When SFT then operates at high resolution, the model's visual processing has been nudged in a direction that conflicts with what SFT needs.

This finding challenges a widespread assumption in the MLLM literature: that any exposure to task-relevant data during pre-training or continual pre-training is beneficial, and that the key question is *which data*, not *at what resolution*. MM1.5's results suggest a stronger principle: **the resolution must match the information density required by the task**. OCR data at low resolution is not just less helpful — it is counterproductive. The paper's decision to set continual pre-training resolution to 1344×1344 (achieving 60.26 MMBase) is not an engineering detail but a requirement for the stage to be beneficial at all.

The "right data, right resolution, right time" principle extends to the relationship between stages. The paper shows that high-resolution OCR understanding *could* be taught during the pre-training stage, but doing so would require modifying the pre-training pipeline (which was already validated in MM1) and would constrain experimentation. By inserting a dedicated continual pre-training stage between pre-training and SFT, the paper effectively **decouples resolution scaling from data scaling**: pre-training handles large-scale multimodal alignment at manageable resolution, continual pre-training handles high-resolution text-rich understanding, and SFT handles instruction-following and capability balancing. Each stage uses resolution appropriate to its purpose, and the ordering ensures that representations learned at lower resolution aren't destroyed by higher-resolution training (which comes later, not interleaved).

This is a **fundamentally new way** to think about MLLM training pipelines — not as a single monolithic process with gradually increasing data quality, but as a sequence of stages each optimized for a specific capability-resoluton pairing. The finding that synthetic captions from other models (LLaVA-Recap, ShareGPT4V) don't help while in-house captions do (Appendix A.1) reinforces the principle: the data's distribution must match the model's, not just the task's. This has significant implications for the common practice of using third-party synthetic data in MLLM training — distribution mismatch can nullify or reverse expected gains.

---

### Innovation 3: Visual Referring and Grounding as a First-Class Capability Rather Than a Post-Hoc Add-On

The paper reframes visual referring and grounding from a niche capability — something that specialized models like Ferret handle — to a **first-class capability that should be integrated into generalist MLLM training by default**. This is not merely a feature addition; it is a conceptual claim about what multimodal understanding *means*. The paper argues implicitly that a model that cannot point to what it's talking about, or understand when a user points to something, has an incomplete understanding of the visual world — regardless of how well it performs on VQA benchmarks.

The evidence for this reframing is partly comparative. Table 7 shows that most competitive open-source models in the 3B-7B range either lack referring and grounding entirely (MiniCPM-V2 produces unparseable bounding boxes) or perform poorly on it (Phi-3-Vision achieves ~40% on RefCOCO averages vs. MM1.5-3B's ~86%). Even GPT-4o, the strongest proprietary model, relies on set-of-mark prompting — a workaround where numbered overlays are added to the image and the model references numbers rather than coordinates natively. The paper positions MM1.5's native coordinate token approach (inherited from Ferret) as architecturally superior: the model can both accept and produce spatial references as part of its natural language generation, without external tooling or prompt engineering.

But the deeper innovation is the demonstration that **referring and grounding can be integrated without destroying other capabilities**, if the integration is done carefully. Figure 5 shows that adding referring & grounding data (`$\alpha_{\text{rg}} = 2.0$`, meaning 2× sampling weight relative to general data) causes only slight regressions in general, text-rich, and knowledge benchmarks while achieving strong spatial reasoning (Refer & Ground Average Score jumps from ~18 to ~73). Figure 8 confirms that the Single-Image Mixture (which includes referring & grounding) achieves better overall average performance than the Base Mixture (which doesn't), despite the small per-capability regressions — the spatial reasoning gains outweigh the costs when measured holistically.

This is significant because it contradicts the implicit assumption in much of the field that spatial reasoning and semantic understanding are in tension or require separate models. Cambrian-1, for instance, is trained exclusively on single-image data and lacks any spatial reasoning capability. LLaVA-OneVision includes limited grounding but performs substantially worse than MM1.5 on RefCOCO (Table 7). The paper's finding that a single model can simultaneously achieve strong VQA, OCR, *and* spatial grounding — and that the trade-off is manageable through careful ratio optimization — supports a vision of MLLMs as genuinely multimodal reasoners, not just image-conditioned language models.

The practical implication is a **shift in what the community should consider a "complete" MLLM**. If spatial grounding can be added with only minor trade-offs (as the paper demonstrates), then models that lack it are incomplete by choice, not by necessity. This reframing could change how benchmarks are designed (spatial reasoning becomes a standard evaluation axis) and how models are compared (capability breadth becomes as important as per-benchmark scores).

---

### Innovation 4: Training-Free Video Understanding as a Capability Transfer Diagnostic — The Multi-Image Reasoning Bridge

The paper's approach to video understanding is conceptually distinctive: rather than treating video as a separate modality requiring dedicated architectures (3D convolutions, temporal attention, video-specific pre-training), it treats video as **multi-image reasoning** — a capability the model already possesses from its interleaved pre-training and multi-image SFT. The "training-free" variant in Table 10 operationalizes this claim: take the image model, sample 24 frames uniformly from a video, feed them as a sequence of images, and ask a question. The model has never seen video data during training; it has only seen pairs and small sets of images in interleaved contexts. If it performs well, it's because multi-image reasoning genuinely transfers to temporal reasoning.

The results partially validate this hypothesis but also reveal its limits. On multiple-choice VideoQA, the training-free MM1.5-Video-3B achieves 48.4 on VideoMME (without subtitles) and 48.4 on EgoSchema, outperforming specialized training-free 7B models on several benchmarks. This is remarkable: a 3B model that has never been trained on video outperforms purpose-built video understanding systems that were carefully designed for temporal reasoning. The implication is that **temporal understanding can emerge from spatial understanding plus multi-image comparison** — the model doesn't need to learn what motion is; it can infer temporal relationships by comparing visual states across frames, much like a person might understand a comic strip.

But the limits are equally informative. On open-ended benchmarks like LLaVA-Hound (which requires detailed video descriptions), the training-free models underperform fine-tuned counterparts. And on long-form video understanding (VideoMME, EgoSchema), the SFT model significantly outperforms the training-free version (e.g., MM1.5-Video-7B SFT: 53.5 vs. training-free: 52.4 on VideoMME; 57.2 vs. 49.6 on EgoSchema). This suggests that while multi-image reasoning provides a strong foundation, **temporal dynamics beyond frame-to-frame comparison — action sequences, causal relationships, long-range dependencies — require explicit video training**. The gap between training-free and SFT performance amounts to a diagnostic of what multi-image reasoning cannot capture about video.

This framing is innovative because it transforms video understanding from a separate research problem into a **capability transfer diagnostic**. Instead of asking "how do we build a video model?", the paper asks "what does video understanding require beyond multi-image reasoning, and how much does explicit temporal training add?" The answer — a lot for open-ended generation and long-form understanding, less for multiple-choice recognition — provides a granular picture of what video-specific training actually teaches. This is a more scientifically useful result than simply reporting that video fine-tuning improves performance, which is obvious.

The connection to pre-training data composition is also notable. The paper's decision to reduce interleaved pre-training data from 45% to 10% (Section 3.4) was made with multi-image and in-context learning preservation in mind. The strong training-free video results validate this decision: even at 10%, the interleaved data provided enough multi-image reasoning capability to transfer to video. If interleaved data had been eliminated entirely (as in some single-image-only models), this transfer would likely fail. The 50:10:40 ratio thus represents a **deliberate bet on capability transfer** — sacrificing some single-image performance (relative to what more caption data might provide) to preserve a foundation for temporal and multi-context reasoning that pays off in unexpected domains.

## 5. Experimental Analysis

### Evaluation Methodology

- **Dataset and splits.** All experiments use the MATH benchmark (Hendrycks et al., 2021), specifically the split from Lightman et al. (2022): 12,000 training questions and 500 test questions. The paper deliberately selects MATH because test-time compute is expected to help most when the model already possesses the necessary knowledge and the challenge lies in drawing complex inferences—mathematical reasoning fits this profile. Difficulty is defined relative to the base model's capabilities, not the dataset's hand-labeled difficulty levels, and questions are binned into five quintiles based on the base model's pass@1 rate computed from 2048 samples per question (Section 3.2).

- **Base model.** All experiments use PaLM 2-S* (Codey) (Anil et al., 2023), which the authors describe as "representative of the capabilities of many contemporary LLMs." The model achieves roughly 10–19% pass@1 on MATH depending on the prompt and sampling configuration—non-trivial performance but far from saturation, leaving room for test-time compute to make a difference. For the FLOPs-matched comparison, a second model with approximately 14× more parameters is used as the pretraining-scaled baseline (Section 4).

- **Metrics.** The primary metric throughout is **MATH test accuracy (%)**—the fraction of the 500 test questions for which the selected final answer matches the ground truth. Answers are graded using the grading function released by Lightman et al. (2022) (Appendix G). When analyzing difficulty-dependent behavior, the paper reports accuracy within each of the five difficulty quintiles separately.

- **Baselines.** The paper compares against several approaches: **majority voting** (selecting the most common final answer among N sampled solutions, with no learned verifier); **ORM best-of-N weighted** (scoring N solutions with an outcome reward model and applying best-of-N weighted selection); **PRM best-of-N weighted** (scoring N solutions with the process reward model and applying best-of-N weighted selection); and for the revision experiments, **parallel sampling** (generating N independent solutions from the revision model and selecting the best via verifier or majority voting).

- **Generation budget and compute accounting.** The universal unit of test-time compute is one "generation"—one complete sampled answer from the base LLM. For beam search and best-of-N, the budget equals the number of beams or samples N. For lookahead search with k lookahead steps, the cost is N × (k+1) to account for the additional rollout computation (Section 5.3). Budgets are swept across powers of 2, typically from 2⁰ to 2⁹ (1 to 512 generations). For FLOPs-matched comparisons, the paper uses standard approximations: pretraining FLOPs = 6ND_pretrain and inference FLOPs = 2ND_inference, where N is parameter count (Section 7).

- **Cross-validation protocol.** To avoid contaminating strategy selection with test-set performance, the authors use two-fold cross-validation within each difficulty bin on the 500-question test set (Section 3.2). The best strategy is selected on one fold and evaluated on the other, with results averaged. This applies to both the compute-optimal search policy (selecting the best search algorithm per difficulty bin and budget) and the compute-optimal revision policy (selecting the optimal sequential-to-parallel ratio per bin).

### Main Quantitative Results

#### Search Against PRM Verifiers (Section 5)

**Aggregate search algorithm comparison (Figure 3, left).** Across all 500 test questions with a maximum budget of 256 generations, the paper finds that beam search significantly outperforms best-of-N at low generation budgets but the advantage diminishes or reverses at high budgets:

- At 4 generations, beam search with M = 4 achieves roughly 27% accuracy versus roughly 16% for PRM best-of-N weighted—a gap of approximately 11 percentage points. This demonstrates that when the budget is severely constrained, guided search is substantially more efficient than independent sampling.
- At 64–256 generations, the gap narrows and eventually reverses. Best-of-N weighted reaches approximately 38% at 512 generations, while beam search (M = 4) plateaus around 34%. The degradation is attributed to **over-optimization of the PRM**: search finds solutions that score highly under the verifier but are actually incorrect.
- Lookahead search (both k = 1 and k = 3) generally underperforms other methods at the same generation budget because its extra cost reduces the effective number of beams explored. The three-step lookahead variants converge to similar performance as other methods at very high budgets but never surpass them—a non-obvious result, since lookahead search is the most powerful optimizer.
- Majority voting trails all verifier-based methods substantially, reaching only about 29% at 512 generations, confirming that the PRM provides genuine signal beyond simple answer consensus.

**Difficulty-dependent search behavior (Figure 3, right).** When results are broken out by difficulty bin, a pattern emerges that is central to the paper's compute-optimal thesis:

- **Bin 1 (easiest questions):** Beam search accuracy actually *decreases* slightly as budget increases from 4 to 256 generations (from roughly 78% to approximately 77%), while best-of-N weighted improves from roughly 68% to approximately 88%. This is the clearest evidence of PRM over-optimization—beam search finds solutions that exploit the verifier signal rather than genuinely correct solutions. On easy problems where the base model already produces many correct answers, aggressive optimization amplifies residual verifier errors.
- **Bin 2:** Beam search improves modestly (roughly 14% → 32%) but best-of-N weighted improves faster (roughly 14% → 60%), maintaining a clear advantage at high budgets. The verifier signal is useful but not reliable enough to guide aggressive optimization beyond what random sampling achieves.
- **Bin 3:** Beam search consistently outperforms best-of-N weighted across all budgets, reaching roughly 34% versus approximately 23% at 256 generations. This is the regime where PRM guidance is most valuable—the model produces some correct solutions (so there are good candidates to find) but not enough for random sampling to be efficient (so guided search helps).
- **Bin 4:** Beam search shows the strongest relative advantage, reaching roughly 17% versus approximately 10% for best-of-N at 256 generations. The absolute numbers are low, but the ratio is meaningful.
- **Bin 5 (hardest questions):** Both methods hover near 1–3% regardless of budget. No method makes meaningful progress—the base model simply lacks the capability to produce correct solutions, so no amount of search or verification can help.

**Compute-optimal search (Figure 4).** By selecting the best search strategy per difficulty bin at each budget level:

- At 16 generations, compute-optimal scaling (oracle bins) achieves approximately 27% accuracy, roughly matching PRM best-of-N weighted at 64 generations—a 4× compute reduction. This is the headline efficiency figure for search.
- At 256 generations, compute-optimal oracle reaches approximately 39.5%, surpassing PRM best-of-N weighted at the same budget (roughly 37%). The gain narrows at higher budgets but remains positive.
- Compute-optimal with predicted difficulty bins tracks the oracle version closely, with the curves "largely overlapping" per the authors. The predicted version reaches approximately 37% at 256 generations—slightly below oracle but substantially above the best-of-N baseline.
- Both compute-optimal variants consistently outperform ORM best-of-N weighted (which peaks around 34% at 512 generations) and majority voting (around 29%), demonstrating that the difficulty-adaptive strategy works across verifier qualities.

**PRM vs. ORM (Appendix F, Figure 14).** At 2048 samples, PRM best-of-N weighted achieves approximately 40% accuracy versus roughly 35% for ORM best-of-N weighted and roughly 30% for majority voting. The gap between PRM and ORM widens with the number of samples, confirming the PRM's superior scaling properties. This validates the paper's investment in process-level verification despite the finding that last-step aggregation effectively reduces the PRM to ORM-like behavior at selection time (Appendix E, Figure 13)—the PRM's step-level training provides superior representation learning even when intermediate predictions aren't directly used.

#### Revision Model Results (Section 6)

**Revision model scaling behavior (Figure 6, left).** The revision model's per-step accuracy improves throughout the chain: starting from approximately 18.2% pass@1 at step 1, accuracy rises to roughly 24–25% by steps 15–20, and remains in the 23–25% range out to 64 steps. This demonstrates generalization beyond the 4-step training horizon—the model learned a revision skill that transfers to much longer chains, rather than memorizing 4-step trajectories. The gradual improvement within the first 15–20 steps suggests the model makes meaningful corrections, while the plateau afterward suggests diminishing returns where subsequent revisions largely preserve rather than improve correctness.

**Sequential vs. parallel at matched budget (Figure 6, right).** At 64 generations:

- Sequential + best-of-N weighted: approximately 41.5%
- Parallel + best-of-N weighted: approximately 39%
- Sequential + majority: approximately 38%
- Parallel + majority: approximately 35%

Sequential revision outperforms parallel sampling under both selection mechanisms. The gap is roughly 2.5 percentage points with verifier-based selection and approximately 3 points with majority voting—meaningful but not dramatic. The fact that sequential revision helps even with majority voting (which doesn't benefit from the revision context) suggests the improvement comes primarily from generating better candidates, not just from the verifier seeing more context.

**Sequential-to-parallel ratio sweep (Figure 7, left).** For a fixed generation budget of 256:

- The optimal ratio is around 2¹ to 2³ (2:1 to 8:1 sequential-to-parallel), achieving approximately 43–44% accuracy.
- Fully parallel (leftmost point) yields approximately 40%.
- Fully sequential (rightmost point) yields approximately 42%.
- At lower budgets (8–32 generations), the curves are monotonically increasing with the sequential-to-parallel ratio—fully sequential is optimal. This suggests that at tight budgets, exploiting a single chain deeply is more efficient than exploring multiple shallow chains.

**Difficulty-dependent optimal ratio (Figure 7, right).** At a fixed budget of 128 generations:

- **Bin 1 (easiest):** Performance is essentially flat across all ratios, hovering around 90–92%. These problems are so easy that the allocation strategy barely matters.
- **Bin 2:** Slight advantage for higher sequential ratios, approximately 63% at fully sequential versus 58% at fully parallel. Easy-medium problems benefit from refinement of already-reasonable initial answers.
- **Bin 3:** A clear optimal ratio emerges at moderate sequential-to-parallel values (around 2¹ to 2³), reaching approximately 42% versus 35% at the extremes. Medium problems need the diversity of parallel exploration combined with the refinement of sequential revision.
- **Bin 4:** Similar pattern, with the peak at moderate ratio achieving roughly 18% versus 14% at fully parallel. The absolute numbers are low but the pattern is consistent.
- **Bin 5 (hardest):** All ratios produce roughly 2–3% accuracy. No allocation strategy helps on problems fundamentally outside the model's capability.

**Compute-optimal revisions (Figure 8).** Selecting the optimal sequential-to-parallel ratio per difficulty bin:

- At 64 generations, compute-optimal oracle achieves approximately 40%, matching parallel best-of-N weighted at 256 generations—a 4× compute reduction, symmetric with the search finding.
- At 256 generations, compute-optimal oracle reaches approximately 44%, compared to roughly 41% for best-of-N weighted and 37% for parallel-only. The gap widens at higher budgets.
- Compute-optimal predicted bins perform slightly below oracle bins at high budgets (approximately 41% at 256 generations versus 44% for oracle) but still substantially outperform the parallel baseline.
- Critically, the parallel baseline appears to **plateau** around 36–37% at high budgets, while compute-optimal scaling continues to improve. This suggests the gains from adaptive allocation compound: as budgets increase, the optimal strategy becomes increasingly different from the uniform parallel baseline, and the efficiency gap widens.

#### FLOPs-Matched Comparison: Test-Time Compute vs. Pretraining (Section 7)

**Revisions (Figure 9, left; Figure 1, top-right bar chart).** Comparing PaLM 2-S* with compute-optimal revisions against the ~14× larger model across three values of R = D_inference / D_pretrain:

- At R ≪ 1 (R = 0.16): Test-time compute outperforms the larger model across **all** difficulty levels. Easy questions show +11.8% relative improvement, medium questions +27.8%, hard questions +21.6%.
- At R ≈ 1 (R = 0.79): Test-time compute remains preferable on easy (+3.5%) and medium (+16.7%) questions but shows an implied negative on hard questions.
- At R ≫ 1 (R = 22): Test-time compute is only preferable on easy questions (−11.9% relative, but still positive or competitive in absolute terms). Medium questions show +5.4%. Hard questions show −37.2% disadvantage—the larger model substantially outperforms.

The key pattern: test-time compute is most advantageous when R ≪ 1 (few inference tokens relative to pretraining, so inference compute is cheap compared to the pretraining savings from using a smaller model) and when problems are easy-to-medium (within the base model's capability range). As R increases or difficulty rises, the advantage shrinks or reverses.

**PRM search (Figure 9, right; Figure 1, bottom-right bar chart).** The pattern is qualitatively similar but quantitatively starker:

| Difficulty | R ≪ 1 (0.16) | R ≈ 1 (0.79) | R ≫ 1 (22) |
|---|---|---|---|
| Easy | +19.1% | +2.2% | +2.0% |
| Medium | 0.0% | −35.3% | −30.8% |
| Hard | −3.6% | −35.3% | −52.9% |

PRM search shows weaker benefits than revisions for the FLOPs-matched comparison, with substantial disadvantages on medium and hard questions even at moderate R values. On easy questions, test-time compute remains preferable across all R regimes, though the margin narrows significantly. The asymmetry between search and revisions in the FLOPs-matched setting is notable: revisions provide broader benefit across difficulty-R combinations, possibly because improving the proposal distribution (what the model generates) is more parameter-efficient than improving the verifier (which must be learned and may over-optimize).

**Figure 9 line plots detail.** The line plots show accuracy per difficulty bin as test-time compute scales. The 14× larger model's greedy performance (shown as stars) is placed at three x-axis positions corresponding to the three R values. Where the compute-optimal scaling line is above the star, test-time compute wins; where below, pretraining wins. On bin 1 (purple, topmost line), the scaling line is above all three stars for revisions. On bin 5 (blue, bottommost line), the scaling line is essentially flat near 0–5% and below all three stars—no amount of test-time compute helps on the hardest problems. This visualization makes the difficulty boundary starkly visible.

### Ablation Studies and Robustness Checks

**PRM aggregation strategy (Appendix E, Figure 13).** The paper compares three methods for aggregating per-step PRM scores into a single solution-level score: minimum across steps ("min"), product of step-level correctness probabilities ("prod"), and only the PRM's prediction at the final step ("last"). Contrary to prior work (Lightman et al., 2023; Wang et al., 2023) which found "min" to be best, this paper finds "last" performs best: at 256 samples, "last" achieves roughly 37%, "min" achieves roughly 35%, "prod" achieves roughly 27%, and a separately trained ORM achieves roughly 34%. The authors hypothesize the discrepancy arises because their PRM is trained with soft Monte Carlo labels rather than binary correctness labels, which changes how per-step scores distribute. Using the last-step prediction effectively makes the PRM behave like an ORM at aggregation time, yet the PRM still outperforms the ORM—suggesting the step-level PRM training provides beneficial representation learning even when intermediate predictions aren't directly used.

**PRM vs. ORM scaling (Appendix F, Figure 14).** The PRM consistently outperforms the ORM, with the gap widening at higher sample counts: at 2048 samples, PRM best-of-N weighted reaches approximately 40% versus ORM's 35%. This validates the additional complexity of PRM training despite the "last" aggregation finding.

**Revision model verifier choice (Appendix J, Figure 15a).** The base-LM PRM underperforms the revision-specific ORM when scoring revision model outputs, with sequential + base-LM PRM achieving roughly 40% at 64 generations versus sequential + revision ORM at roughly 42%. This confirms distribution shift as a practical concern: the PRM trained on base model outputs does not transfer well to the revision model's outputs. The revision model produces solutions with different characteristics (potentially longer, more refined, different error patterns) that the base PRM is not calibrated for.

**Revision history in verifier context (Appendix J, Figure 15b).** Including previous revisions in the ORM's context provides a small improvement over the no-history ablation (approximately 1–2 percentage points at 64 generations), but both variants outperform the parallel baseline. This confirms that the sequential sampling benefit is not solely attributable to the verifier seeing more context—the revision model genuinely produces better candidates even when scored by a verifier without access to revision history.

**Oracle vs. predicted difficulty bins (Figures 4, 8; Appendix C, Figures 11–12).** Both oracle and predicted bins yield qualitatively similar trends across difficulty levels. In the search setting (Figure 4), the two curves "largely overlap." In the revision setting (Figure 8), predicted bins show slightly lower performance at high budgets (roughly 41% vs. 44% at 256 generations). The gap is meaningful but the predicted-difficulty policy still substantially outperforms the parallel baseline, confirming the approach is practical without ground-truth labels.

**Majority voting for revisions (Appendix B, Figure 10).** The sequential-to-parallel ratio trends observed with verifier-based selection are replicated with majority voting: easy questions are insensitive to the ratio, hard questions show an optimal intermediate ratio, and fully sequential marginally outperforms fully parallel in aggregate. This robustness check demonstrates that the revision model's benefits are not an artifact of verifier design—the model genuinely generates better candidates through sequential refinement.

**ReST^EM revision model (Appendix K, Figure 16).** An attempt to further optimize the revision model using ReST^EM (Singh et al., 2024) backfires: additional sequential revisions **substantially hurt** performance with this model. At 256 generations, fully sequential performance drops to approximately 33.5% compared to roughly 38.5% at the optimal ratio. The authors hypothesize that on-policy data collection in ReST^EM exacerbates spurious correlations in revision data, causing the model to fail to learn the revision task properly. This is a notable negative result that highlights the sensitivity of revision training to the data generation procedure. It also suggests that the carefully controlled offline data construction (edit-distance-based incorrect-correct pairing with random sampling of preceding incorrect answers) is not trivially replaceable with more sophisticated RL-based methods, at least in the current formulation.

### Critical Assessment

**How well does the evidence support the central claims?**

The paper's headline claim is that compute-optimal test-time scaling achieves **more than 4× better efficiency** over a standard best-of-N baseline (Section 1). The evidence supports this conditional claim in the following specific sense: at certain budget levels and difficulty distributions, matching accuracy is achieved with 4× fewer generations. For search, Figure 4 shows ~27% accuracy at 16 generations matching ~27% at 64 generations for best-of-N weighted. For revisions, Figure 8 shows ~40% accuracy at 64 generations matching ~40% at 256 generations for parallel best-of-N. These are well-documented specific comparisons.

However, the "4×" figure should be understood as an **upper bound achievable at specific operating points**, not a universal efficiency multiplier. At higher budgets (256–512 generations), the gap narrows—compute-optimal at 256 generations achieves ~44% versus best-of-N at ~41% in Figure 8, which is meaningful but not 4×. At very low budgets (2–4 generations), the gap is small or nonexistent. And the efficiency gains are computed **excluding the cost of difficulty estimation**, which requires 2048 samples per question—a cost that can dwarf the test-time compute budget being studied. The authors acknowledge this explicitly (Section 3.2: "our experiments do not account for this cost largely for simplicity"), but the implication is substantial: if difficulty estimation costs are amortized over many questions, the net efficiency is lower than reported; if difficulty is estimated per-question, the cost can exceed the budget being optimized. The paper's stated future direction—training a model to predict difficulty directly from question text—is not a minor extension but a **prerequisite for the reported gains to be realizable in practice**.

The second major claim is that a smaller model with compute-optimal test-time scaling can **outperform a ~14× larger model** in a FLOPs-matched comparison (Section 7). The evidence supports this claim with sharp boundary conditions that the paper is transparent about: it holds for easy-to-medium problems when R ≪ 1 or R ≈ 1, but fails for hard problems and degrades as R increases. The precise numbers in Figure 9 and Table 1 quantify these conditions. However, there are important caveats:

- The 14× larger model uses **greedy decoding with no test-time compute augmentation**. This is a weak baseline—even a modest test-time compute budget (e.g., best-of-8) for the larger model would substantially strengthen it. The paper never tests what happens if both models get proportional test-time compute budgets, which would be a fairer comparison for practitioners deciding how to allocate a total budget between pretraining and inference.
- The 14× larger model is **parameter-scaled only**, not Chinchilla-optimally trained (where both parameters and data would scale). The paper acknowledges this (Section 7: "We choose this setting as it is representative of a canonical approach to scaling pretraining compute and leave the analysis of compute-optimal scaling... to future work"). A compute-optimally trained larger model would be a stronger baseline.
- The comparison is on a single benchmark (MATH) with a single base model family (PaLM 2-S*). Extrapolation to other reasoning domains or model families is not validated.

**Key evidential gaps:**

- **No combination of search and revisions.** The paper studies PRM search and iterative revisions independently but never combines them—using the revision model as the proposal distribution within beam search, or using the PRM to guide which revisions to pursue. The authors acknowledge this explicitly (Section 8: "we did not experiment with PRM tree-search techniques in combination with revisions"). Since the two mechanisms have complementary strengths (revisions help on easy problems, search helps on medium problems), the current results represent a lower bound on what a fully integrated system could achieve. **This is the single largest missing experiment.**

- **No latency or wall-clock analysis.** Sequential revisions are inherently serial and block on previous steps, while parallel best-of-N can be executed simultaneously with sufficient hardware. A strategy that allocates 128 generations as 64 sequential × 2 parallel takes approximately 64× longer wall-clock time than one running 128 parallel samples. For latency-sensitive applications, the sequential-heavy strategies favored by the compute-optimal policy on easy problems may be impractical regardless of FLOPs efficiency. This trade-off is not discussed.

- **Difficulty estimation cost unaccounted for.** As noted above, generating 2048 samples per question to estimate difficulty costs as much as or more than the largest test-time budgets studied. The paper frames this as an exploration-exploitation tradeoff (Section 3.2) but doesn't model it quantitatively. A practical deployment would need to either amortize this cost over many similar questions (not applicable to MATH, where questions are diverse and independent) or develop cheaper estimation methods that are not yet demonstrated.

- **Small test set with cross-validation on ~50 questions per fold per bin.** The 500-question test set split into five difficulty quintiles of ~100 each, further split by two-fold cross-validation, means strategy selection is based on ~50 questions per fold per bin. This is a small sample—confidence intervals are not reported, making it difficult to assess whether the observed gaps between strategies are statistically reliable. At 50 questions per bin, a difference of a few percentage points could arise from a small number of lucky or unlucky questions.

- **Single benchmark, single model family.** All experiments are conducted on MATH with PaLM 2-S*. The paper argues this model is "representative of the capabilities of many contemporary LLMs" (Section 4), but this is an unverified assumption. The PRM's quality, revision model behavior, and over-optimization characteristics could differ substantially across model families. MATH is a specific type of reasoning task (competition-level math with symbolic manipulation); whether the difficulty-dependent patterns generalize to code generation, logical reasoning, or scientific QA is unknown.

**What experiments would have strengthened the paper?**

1. **Combined search + revisions:** Running beam search with the revision model as the proposal distribution, or using PRM scores to guide revision trajectories. This is the natural integration of the two axes the paper identifies as complementary and would likely yield results above either axis alone.

2. **Cheap difficulty estimation:** Training and evaluating a lightweight difficulty classifier (possibly distilled from the PRM) that predicts the difficulty bin from question text alone, and comparing its performance to the 2048-sample method. This would transform the compute-optimal framework from a proof of concept to a practical method.

3. **Adaptive difficulty estimation:** A protocol where the system starts with a small number of samples (4–8), uses the PRM score distribution as a quick difficulty signal, and then allocates the remaining budget accordingly. This would amortize difficulty estimation into the problem-solving process and provide a fair cost accounting.

4. **Larger model with test-time compute:** Comparing the small model with compute-optimal scaling against the 14× larger model *also* augmented with some test-time compute budget (e.g., best-of-8 or best-of-16). This would test whether the pretraining-inference tradeoff holds when both sides use inference compute, rather than comparing an augmented small model against a greedy large model.

5. **Confidence intervals:** Reporting error bars or confidence intervals on the compute-optimal scaling curves, particularly given the small per-bin sample sizes (~50 questions per fold). Without these, the reliability of the strategy selection in the cross-validation procedure is difficult to assess.

6. **Replication on another benchmark:** At minimum, replicating the key findings (difficulty-dependent optimal strategy, 4× efficiency gain, FLOPs-matched comparison) on a second reasoning benchmark (e.g., GSM8K for math, or HumanEval for code) to demonstrate generality.

**Overall assessment of experimental rigor:**

The experiments are well-designed and thorough *within their scope*. The additive methodology for SFT data mixture optimization, the systematic sweeps over difficulty bins and strategy parameters, and the cross-validation protocol for compute-optimal policy selection are methodologically stronger than most MLLM papers. The paper is transparent about limitations (difficulty estimation cost, missing combination of search and revisions, narrow evaluation domain) and does not overclaim.

However, the scope itself is narrower than the paper's framing sometimes implies. The claims about "4× efficiency gains" and "outperforming 14× larger models" are **precisely true for the specific operating points shown in the figures** but should not be interpreted as universal properties of test-time compute scaling. The gains are largest at moderate budgets on easy-to-medium problems; they narrow at high budgets and vanish on hard problems. The practical relevance depends on the difficulty estimation bottleneck being solved—a problem the paper identifies but does not address. And the single-benchmark, single-model-family evaluation limits conclusions about generality.

The most compelling evidence is for the paper's central conceptual claim: that **optimal test-time compute allocation is difficulty-dependent**, with qualitatively different optimal strategies for easy, medium, and hard problems. This finding is replicated across search methods (beam search hurts easy, helps medium), revision strategies (sequential helps easy, balanced helps medium, neither helps hard), and selection mechanisms (verifier-based and majority-based). The consistency across axes and the non-monotonic nature of the effects make this a robust and novel contribution that is well-supported by the data shown.

## 6. Limitations and Trade-offs

### Limitation 1: Difficulty Estimation Cost Is Unaccounted For and Dominates the Inference Budget

**The assumption or constraint.** The entire compute-optimal test-time scaling framework depends on knowing each prompt's difficulty *before* allocating the inference budget. The paper's method for estimating difficulty—whether oracle (computing pass@1 from 2048 ground-truth-labeled samples) or predicted (averaging the PRM's final-answer score over 2048 samples)—requires generating **2048 complete solutions per question** and scoring them. The authors acknowledge this explicitly in Section 3.2:

> "estimating difficulty in this way still incurs additional computation cost during inference... our experiments do not account for this cost largely for simplicity"

This is not a small overhead—it is potentially larger than the test-time compute budgets being optimized (which max out at 256–512 generations). The paper frames the difficulty estimation problem as an exploration-exploitation tradeoff (Section 3.2) but never quantifies or models this cost.

**The consequence.** The headline efficiency gains—4× improvement over best-of-N (Figures 4 and 8)—are computed *after* difficulty is known, without amortizing the cost of learning it. In any realistic deployment where difficulty must be estimated per-question, the total cost would be: cost of difficulty estimation + cost of strategy execution. Since the former (2048 generations) is 4–8× larger than the largest strategy budgets studied (256–512 generations), the *net efficiency* of the compute-optimal approach could be substantially worse than simply running best-of-N at maximum budget on every question—the very baseline the paper claims to improve upon. The difficulty estimation cost is a fixed overhead per question, meaning it hits hardest on easy questions (where the optimal strategy budget is small), potentially making the adaptive approach *less* efficient than uniform allocation for exactly those questions where the paper reports the largest relative gains.

**What evidence exists in the paper.** The paper provides no measurement of the difficulty estimation cost in its efficiency calculations. Figures 4 and 8 do not include any accounting for the samples used in difficulty binning. The only mitigation discussed is a speculative future direction:

> "Future work could explore training a model to directly predict the difficulty of a question"

No such model is trained or evaluated. Appendix C (Figures 11–12) shows that predicted difficulty bins (using the PRM's score distribution instead of ground-truth correctness) track oracle bins reasonably well, but this does not reduce the computational cost—it only removes the need for ground-truth labels. The 2048-sample requirement remains.

**Mitigation status.** Not addressed. The paper identifies the problem but defers the solution entirely to future work. This is the single largest gap between the paper's reported results and what a practitioner would need to deploy the method. Until cheap difficulty estimation (e.g., a lightweight classifier trained on question text, or an adaptive protocol that estimates difficulty from a small number of initial samples) is demonstrated, the 4× efficiency figure should be understood as an **upper bound on achievable efficiency** in a regime where difficulty estimation is free—a regime that does not exist in practice.

---

### Limitation 2: Hard Problems (Difficulty Bin 5) Show Essentially Zero Improvement Regardless of Method or Budget

**The assumption or constraint.** The compute-optimal framework operates by reallocating inference compute across difficulty levels—reducing budget on easy questions where search hurts, increasing it on medium questions where search helps. But this strategy is bounded by a hard ceiling: on the hardest questions (difficulty bin 5, the bottom quintile of the base model's pass@1 rate), the base model produces correct solutions at a rate so low (pass@1 near 0%) that no amount of test-time compute can find or refine them.

**The consequence.** Across every experiment in the paper, bin 5 accuracy remains essentially flat near 0–5% regardless of method, budget, or allocation strategy:

- Figure 3 (right): bin 5 accuracy hovers at 1–3% for both beam search and best-of-N at all budgets from 4 to 256 generations.
- Figure 7 (right): bin 5 accuracy is roughly 2–3% irrespective of the sequential-to-parallel ratio.
- Figure 9: the bin 5 scaling line (blue, bottommost) stays near 0–5% and falls below the 14× larger model's greedy performance at all values of R.
- The FLOPs-matched comparison (Figure 1, bottom-right bar chart) shows test-time compute producing a −52.9% *disadvantage* relative to the larger model on hard questions at R ≫ 1 for PRM search.

This means the compute-optimal framework offers **no path forward for genuinely difficult problems**—those where the base model's capability is fundamentally insufficient. Test-time compute amplifies existing capability but does not create it. As the paper states in the Section 7 takeaway: the method works best "when the prompt is within the model's capability range." For problems outside that range, pretraining a larger model remains the only viable path, and spending inference compute on a small model is counterproductive.

**What evidence exists in the paper.** The bin 5 failure is documented consistently across all figures and is one of the most robust findings in the paper. It appears in search experiments (Section 5.3, Figure 3 right), revision experiments (Section 6.2, Figure 7 right), and the FLOPs-matched comparison (Section 7, Figure 9). The paper is transparent about this limitation:

> "test-time compute can amplify existing capability but cannot create it from nothing"

However, the paper does not quantify what fraction of real-world problem distributions fall into this "hopeless" regime. The five-bin discretization is relative to PaLM 2-S*'s specific capability profile on MATH—a different base model or a different benchmark would have a different fraction of problems in the bottom quintile. The practical consequence is that the utility of compute-optimal test-time scaling depends critically on the overlap between the deployment problem distribution and the base model's competence region. For applications where problems are predominantly hard (near the frontier of model capability), pretraining investment dominates; for applications where problems are predominantly easy-to-medium, test-time compute dominates. The paper provides no method for estimating this overlap before deployment.

**Mitigation status.** The paper acknowledges the limitation candidly but does not attempt to address it. This is appropriate—the limitation is inherent to the "amplification, not creation" nature of test-time compute—but it means the method's applicability is conditional on problem difficulty distribution, and the paper does not provide tools for assessing whether a given deployment satisfies that condition.

---

### Limitation 3: The 14× Larger Model Baseline Is Artificially Weak—No Test-Time Compute, Not Compute-Optimally Trained

**The assumption or constraint.** The FLOPs-matched comparison in Section 7 pits PaLM 2-S* with compute-optimal test-time scaling against a model with approximately 14× more parameters using **greedy decoding only**—no majority voting, no best-of-N, no search, no revisions. Furthermore, the larger model is scaled only in parameters while holding training data fixed, following the LLaMA paradigm (Touvron et al., 2023) rather than Chinchilla-optimal training where both parameters and data scale equally. The paper states:

> "We choose this setting as it is representative of a canonical approach to scaling pretraining compute and leave the analysis of compute-optimal scaling of pretraining compute where the data and parameters are both scaled equally to future work."

**The consequence.** The reported advantages of test-time compute over pretraining are measured against a baseline that is **weaker than it needs to be on two independent axes**. First, a compute-optimally trained larger model (scaling both parameters and data per Hoffmann et al., 2022) would achieve higher performance for the same pretraining FLOPs, narrowing or reversing the gap. Second, even a modest test-time compute budget for the larger model—say, best-of-8 or best-of-16—would substantially improve its performance without a large fractional increase in total compute (since inference FLOPs scale linearly with model size and token count). The comparison as structured essentially asks: "Is a small model with lots of inference compute better than a large model with no inference compute?" A fairer question—and one more relevant to practitioners allocating a total compute budget—would be: "Given a fixed total compute budget, what is the optimal split between pretraining and inference?" This would require giving both models proportional test-time compute budgets.

The specific numbers reported in Section 7 (e.g., +27.8% relative improvement for revisions on easy-medium questions at R ≪ 1) are therefore **upper bounds on the advantage of test-time compute over pretraining** that would shrink against stronger baselines. The finding that test-time compute is most advantageous at low R (where pretraining savings are large relative to inference costs) remains qualitatively valid, but the crossover point where pretraining becomes preferable likely shifts toward lower difficulty or lower R with a stronger baseline.

**What evidence exists in the paper.** The paper does not ablate the baseline strength—there are no experiments where the larger model receives test-time compute augmentation, and no comparison against a compute-optimally trained larger model. The authors are transparent about the Chinchilla limitation (quoted above) but do not discuss the greedy decoding limitation. The bar charts in Figure 1 and the line plots in Figure 9 show only the greedy performance of the larger model (marked with stars), making the comparison visually stark but potentially misleading.

**Mitigation status.** Partially addressed through acknowledgment of the Chinchilla limitation, but the greedy decoding issue is not discussed. The paper frames both as future work. A stronger comparison—giving the larger model a proportional test-time compute budget, or using a compute-optimally trained larger model—would substantially strengthen the paper's central claim about the pretraining-inference tradeoff. Until such comparisons are conducted, the Section 7 results should be interpreted as a **proof of concept that test-time compute can substitute for pretraining under favorable conditions**, not as a general prescription for allocating total compute budgets.

---

### Limitation 4: Revisions and Search Are Never Combined, Despite Complementary Difficulty-Dependent Strengths

**The assumption or constraint.** The paper studies two complementary axes—PRM-guided search (modifying the verifier/selection mechanism) and iterative revisions (modifying the proposal distribution)—but evaluates them entirely independently. Section 8 explicitly acknowledges this gap:

> "we did not experiment with PRM tree-search techniques in combination with revisions"

The paper's own analysis demonstrates that these two axes have complementary, difficulty-dependent strengths: revisions are most effective on easy problems (where sequential refinement of roughly-correct answers helps, Figure 7 right), while PRM search is most effective on medium problems (where beam search guides exploration toward correct solutions the model wouldn't find by random sampling, Figure 3 right). Easy problems benefit from local refinement, medium problems from global search—a natural division of labor that a combined system could exploit.

**The consequence.** All performance numbers reported for MM1.5 represent a **lower bound** on what a fully integrated system could achieve. A natural combination—using the revision model as the proposal distribution within beam search, or using the PRM's step-level scores to guide which revision trajectories to pursue versus restart—could yield gains beyond either method alone. The difficulty-dependent optimal strategy for a combined system would likely be more nuanced than for either axis independently, potentially pushing the performance ceiling higher across multiple difficulty bins.

For practitioners, this means the paper does not answer the most natural follow-up question: "Should I use revisions, search, or both?" The ablation structure (studying search in Section 5, revisions in Section 6, never intersecting) leaves this question completely open. Given that both methods individually show 4× efficiency gains over best-of-N (Figures 4 and 8), it is plausible—but entirely unverified—that combining them could yield multiplicative gains.

**What evidence exists in the paper.** None. There are no experiments where the revision model generates candidates that are then scored by the PRM and searched over, and no experiments where the PRM guides the revision process. The paper's Figure 2 (architecture overview) and Section 2 (unifying framework) present search and revisions as complementary mechanisms, but the experimental design treats them as separate studies. The authors acknowledge the gap and characterize it as a natural next step, but the absence of any combined experiment is arguably the single largest missing ablation in the paper, given the conceptual framework the paper itself establishes.

**Mitigation status.** Not addressed experimentally. The paper identifies the combination as future work (Section 8) but provides no preliminary results, no analysis of what challenges might arise (e.g., distribution shift between revision model outputs and the PRM's training distribution, which Appendix J, Figure 15a shows is a real issue), and no guidance for practitioners on how to combine the two axes. This is a significant gap because it means the paper's main practical recommendation—compute-optimal allocation across search algorithms and revision strategies—is based on treating these axes as alternatives rather than complements, potentially underestimating the total gains available from test-time compute.

---

### Limitation 5: Single Benchmark (MATH) and Single Model Family (PaLM 2-S*) Leave Generality Unverified

**The assumption or constraint.** All experiments—including search, revisions, difficulty estimation, compute-optimal strategy selection, and the FLOPs-matched comparison—are conducted on a single benchmark (MATH, Hendrycks et al., 2021) using a single base model family (PaLM 2-S*, Anil et al., 2023). The paper argues this model is "representative of the capabilities of many contemporary LLMs" (Section 4), but provides no evidence for this claim through replication on other models or tasks.

MATH consists of high-school competition-level mathematics problems requiring multi-step symbolic reasoning and producing unambiguous ground-truth answers. This task profile has specific properties that may not generalize: (1) answers can be verified with exact string matching, enabling clean PRM training via Monte Carlo rollouts; (2) problems have well-defined difficulty hierarchies that correlate with the number and complexity of reasoning steps; (3) the base model's pass@1 distribution spans a useful range (~10–19%) where test-time compute can make a difference. Different task domains—code generation, logical reasoning, scientific QA, open-ended generation—may have different difficulty distributions, different verifier reliability, and different sensitivity to search versus revision strategies.

**The consequence.** The paper's key findings—4× efficiency gains from compute-optimal allocation, difficulty-dependent strategy selection, the over-optimization threshold for beam search on easy problems, the effectiveness of sequential revisions on easy problems—may be **specific to mathematical reasoning with PaLM 2-S*** rather than general properties of test-time compute scaling. The PRM's quality, the base model's error patterns, and the revision model's ability to learn from incorrect in-context examples are all model-specific and task-specific.

Without replication, a practitioner cannot know whether:
- The difficulty-dependent optimal strategy transfers to their model family (different base LLMs have different calibration, different error modes, different pass@1 distributions).
- The over-optimization threshold for beam search occurs at the same budget levels or difficulty thresholds for their task.
- The revision model's 38% correct-to-incorrect reversion rate (Section 6.1) is typical or specific to PaLM 2-S* on MATH.
- The finding that sequential revisions outperform parallel sampling (Figure 6 right) holds for tasks where revisions may introduce rather than correct errors.

**What evidence exists in the paper.** None beyond the MATH benchmark. The paper does not report results on any other reasoning benchmark (e.g., GSM8K for simpler math, HumanEval or MBPP for code, ARC for science reasoning), and all models are within the PaLM 2 family. The difficulty estimation methodology (2048 samples per question, five quintile bins) is benchmark-specific—different benchmarks would have different optimal bin counts and estimation procedures.

**Mitigation status.** Not addressed. The paper does not claim generality beyond MATH and PaLM 2-S*, but the framing in Sections 1 and 8 suggests broader applicability ("the first systematic scaling analysis," "providing valuable guidance for future research in MLLM development"). This limitation is standard for empirical ML papers and does not invalidate the findings, but it does constrain the strength of the paper's prescriptive claims. The compute-optimal framework (treating test-time compute allocation as a difficulty-conditioned optimization problem) is conceptually general, but the specific findings (beam search hurts easy problems, revisions help easy problems, 4× gains at specific budget levels) may be specific to the studied setting until replication demonstrates otherwise.

---

### Limitation 6: Sequential Revision Strategies Are Inherently High-Latency, and Latency Is Never Discussed

**The assumption or constraint.** The paper measures test-time compute exclusively in "generations" (number of complete solutions sampled), which is a reasonable proxy for total FLOPs but ignores **wall-clock latency**. Sequential revisions are inherently serial: each revision step depends on the previous one and cannot be parallelized. In contrast, parallel best-of-N can execute all N samples simultaneously given sufficient hardware. A compute-optimal strategy that allocates 128 generations as 64 sequential × 2 parallel (the optimal ratio for medium problems per Figure 7) takes approximately 64× longer wall-clock time than one that runs 128 parallel samples simultaneously, even though both use the same total FLOPs.

**The consequence.** For latency-sensitive applications—interactive assistants, real-time tutoring, live code generation—the sequential-heavy strategies favored by the compute-optimal policy on easy-to-medium problems may be **practically unusable regardless of their accuracy advantages**. A user waiting for a math answer will not tolerate 64 sequential model calls (each requiring a full forward pass) when a parallel best-of-64 strategy takes the wall-clock time of a single forward pass and achieves only slightly lower accuracy. The paper's "compute-optimal" allocation ignores this dimension entirely, optimizing for total FLOPs (which maps to cost and energy) but not for latency (which maps to user experience and system design constraints).

The compute-optimal policy on easy problems (bin 1–2) favors fully sequential revision (Figure 7 right: performance is flat across ratios, but Figure 7 left shows fully sequential is optimal at low budgets). On these problems, the optimal strategy in FLOPs terms is the *worst possible* strategy in latency terms—a tension the paper never acknowledges. The 38% correct-to-incorrect reversion rate further complicates latency-sensitive deployment, since it means the system must generate the full chain of revisions and then select the best answer from any point in the chain (Section 6.1), adding verification latency on top of generation latency.

**What evidence exists in the paper.** None. The paper never discusses latency, wall-clock time, or the serial versus parallel execution model. All cost accounting is in terms of generation count. The revision model experiments (Section 6) show sequential chains of up to 64 steps (Figure 6 left) without mentioning that this represents 64 sequential autoregressive decoding passes—each of which is already slow for large models. The efficiency gains reported in Figures 4 and 8 are purely in "number of generations to match accuracy," not in "time to solution."

**Mitigation status.** Not addressed. This is not a flaw in the paper's FLOPs-based analysis, but it is a critical consideration for practitioners that the paper should have at least discussed. The tradeoff between FLOPs efficiency and latency is fundamental and unavoidable—parallel strategies are latency-efficient but FLOPs-inefficient; sequential strategies are FLOPs-efficient but latency-inefficient. A complete analysis of test-time compute allocation would optimize over both dimensions, potentially producing different recommendations for throughput-oriented batch inference (where total FLOPs dominate) versus latency-sensitive interactive use (where wall-clock time dominates). The paper's silence on this tradeoff means its recommendations cannot be directly applied to latency-constrained deployments without additional analysis.

## 7. Implications and Future Directions
- Field impact
  - The work shifts attention from architecture novelty to a reproducible, data‑centric training recipe that demonstrably transfers to core multimodal abilities and small‑scale models. It elevates integrated grounding and multi‑image reasoning to first‑class citizens in generalist MLLMs (Sections 1, 4).
- Follow‑up research enabled
  - Systematic study of high‑quality synthetic captions at scale (Appendix A.1 shows early promise).
  - Joint optimization of interleaved pre‑training and dynamic splitting for long‑context, multi‑image chains (e.g., videos or multi‑page documents).
  - Unifying image, video, and UI capabilities under one training schedule, possibly with curriculum over `n_max`, frame counts, and UI‑specific tasks (Section 7, conclusion).
  - More robust grounding with consistent coordinate systems under varying tiling, and extensions to pixel‑level grounding.
- Practical applications
  - Document understanding and enterprise OCR (charts, forms, infographics; Table 6).
  - Grounded assistants that can refer to and act on UI elements (Section 6; Fig. 12; Table 12).
  - Multi‑image analytics (e.g., product comparisons, surveillance snapshots) and video QA/analysis pipelines that can start training‑free, then gain with targeted SFT (Section 5; Tables 10–11).
  - Edge deployment scenarios: 1B/3B dense and MoE variants provide competitive capability for mobile and embedded settings (Tables 4–9).

In short, MM1.5 demonstrates that careful, high‑resolution continual pre‑training plus principled SFT mixture design—and a well‑engineered AnyRes pipeline—unlock substantial, verifiable gains in OCR‑heavy understanding, native grounding, and multi‑image reasoning, from mobile scales to 30B models, with straightforward extension to video and UI domains.
