# InternVL3: Exploring Advanced Training and Test-Time Recipes for Open-Source Multimodal Models

**ArXiv:** [2504.10479](https://arxiv.org/abs/2504.10479)

## 🎯 Pitch

InternVL3 pioneers a native multimodal pre-training paradigm that enables large models to simultaneously learn language and visual capabilities from both text and multimodal data, eliminating the need for complex, post-hoc adaptation stages. By integrating innovations like Variable Visual Position Encoding (V2PE), advanced supervised fine-tuning and mixed preference optimization, and test-time scaling, InternVL3 achieves state-of-the-art performance among open-source models, dramatically simplifying the training pipeline and closing the gap with leading proprietary MLLMs. This approach greatly enhances scalability, efficiency, and robustness for real-world applications such as document understanding, GUI agents, and multimodal reasoning, while fostering transparency and reproducibility through the release of both data and model weights.

---

## 1. Executive Summary

This report introduces InternVL3, the latest generation in the InternVL model family, distinguished by a **native multimodal pre-training** paradigm that jointly optimizes linguistic and multimodal capabilities from interleaved text and multimodal data in a single pre-training stage, rather than retrofitting a pre-trained text-only LLM. Evaluated against benchmarks including MMMU, MathVista, and OCRBench using InternVL3-78B (built on InternViT-6B and Qwen2.5-72B), the model incorporates variable visual position encoding (V2PE — small fractional position increments for visual tokens to extend multimodal context), mixed preference optimization (MPO — combining DPO, BCO, and LM losses on preference pairs), and test-time scaling via best-of-N with step-level visual process reward models (VisualPRM-8B as the critic). InternVL3-78B achieves 72.2 on MMMU, setting a new state-of-the-art among open-source MLLMs, while the compute-optimal test-time strategies yield substantial efficiency gains — for instance, the best-of-8 strategy improves MathVerse Vision-Only scores by 3.2–6.0 points across model scales, establishing that native multimodal pre-training enables open-source models to remain competitive with proprietary counterparts like GPT-4o and Claude 3.5 Sonnet across multimodal reasoning, document understanding, and GUI grounding tasks.

## 2. Context and Motivation

### The Core Problem: Post-Hoc Multimodal Adaptation Creates an Artificial Barrier Between Language and Vision Learning

The fundamental problem InternVL3 addresses is architectural and procedural rather than task-specific. The vast majority of multimodal large language models (MLLMs)—both open-source and proprietary—are built through a multi-stage pipeline that treats vision as an afterthought: (1) pretrain a text-only LLM on massive language corpora, (2) independently train a vision encoder, and then (3) connect them through a lightweight adapter with multimodal alignment training, often freezing or partially freezing the LLM backbone to avoid catastrophic forgetting of linguistic capabilities.

The paper argues in Section 1 that this "post-hoc" paradigm introduces an inherent tension: the LLM was optimized *without any visual signal* during its foundational pre-training, so when multimodal data is later introduced, the model must simultaneously learn to process visual information while retaining linguistic competence. This creates what the paper calls "alignment challenges" that manifest practically as:

- **Training complexity**: Sophisticated parameter-freezing or multi-stage fine-tuning schedules are required to ensure core linguistic capacities remain uncompromised when vision is bolted on. The paper explicitly cites prior work [73, 7, 5, 18] that resorts to these resource-intensive strategies.
- **Data dependency**: Bridging modality gaps often necessitates incorporating auxiliary data from specialized domains—particularly OCR—to compensate for the fact that the model did not learn to read text in images during its most formative training phase.
- **Catastrophic forgetting tradeoff**: When the LLM is fully fine-tuned on multimodal data, language capabilities degrade; when it is frozen, multimodal integration is suboptimal. Both choices create a performance ceiling.

This is not merely an inconvenience. The paper's position (articulated throughout Section 2.2) is that this separation is **fundamentally artificial**—there is no principled reason language understanding and visual understanding must be acquired sequentially rather than jointly. Text and images are complementary carriers of information about the world, and learning them together could, in theory, produce richer representations than learning them separately. The post-hoc paradigm persists primarily because of engineering momentum (most LLMs were developed as text-only systems before multimodal applications became a priority) rather than because it is optimal.

### Why This Matters: Real-World Impact and Theoretical Significance

**Practical deployment implications.** The paper's concern with post-hoc training is not academic. The complexity of multi-stage pipelines directly translates to higher development costs, slower iteration cycles, and greater difficulty in reproducing results. InternVL3's stated goal of releasing both training data and model weights "in pursuit of open-science principles" implicitly critiques a status quo where reproducing leading MLLMs requires navigating proprietary training recipes and opaque multi-stage schedules. A simplified, unified training paradigm lowers the barrier to entry for the broader research community.

**The scaling efficiency argument.** A subtler but equally important point is about compute efficiency. Multi-stage pipelines effectively train the LLM parameters *twice*: once on text-only data, then again during multimodal adaptation, with different objectives and potentially conflicting gradient signals. If joint pre-training can achieve comparable or superior performance without this redundancy, it represents a more compute-efficient path to strong multimodal capabilities—a consideration that becomes increasingly important as models grow to 78B parameters and beyond.

**The representation learning insight.** The paper's approach embodies a hypothesis that is not yet widely tested at scale: that visual and linguistic signals are mutually informative during pre-training, not just during post-hoc alignment. When a model learns the concept of "cat" from text, simultaneously seeing images of cats could produce a richer, more grounded representation than either modality alone. This has theoretical significance for the debate about whether language models can truly "understand" concepts without embodied or perceptual experience—joint pre-training is a step toward models that learn from multiple modalities concurrently, as humans do.

### Where Prior Approaches Fall Short

The paper identifies several specific limitations in existing MLLM training paradigms:

**Sequential modality learning creates a representational mismatch.** When an LLM is pre-trained on text alone, its internal representations are optimized for predicting the next text token in a text-only context. When images are later introduced via an adapter (typically an MLP projector), the visual representations must be squeezed into a representational space that was never designed to accommodate them. This is fundamentally different from an architecture where visual tokens and text tokens share the same representational space from the beginning of pre-training. The paper's joint optimization in Equation 8—`θ* = arg min_θ E_x∈D_multi [L_text-only(θ)]`—is designed so that "text representations and visual features are learned in concert, reinforcing alignment across modalities" rather than one modality being warped to fit the other's pre-existing representational geometry.

**Vision encoders are treated as static feature extractors rather than jointly learned components.** In the standard post-hoc paradigm, the ViT is pre-trained independently (often on image-text contrastive objectives like CLIP) and then frozen or lightly fine-tuned during multimodal alignment. This means the visual features are optimized for a different objective (e.g., image-text matching) than the one they serve in the MLLM (conditioning LLM generation). InternVL3's approach of training the ViT, MLP projector, and LLM simultaneously ensures that visual features are optimized directly for the autoregressive language modeling objective, eliminating this objective mismatch.

**The data imbalance problem is typically handled through ad-hoc rebalancing rather than integrated sampling.** The paper notes (Section 2.2, Data subsection) that multimodal datasets tend to have "relatively short and less diverse textual content" compared to pure language corpora. In post-hoc pipelines, this is addressed by first training on massive text-only data and then fine-tuning on smaller multimodal datasets—which means the model never learns to balance linguistic and visual learning objectives simultaneously. InternVL3's two-stage sampling strategy (first finding optimal ratios within each modality, then determining the cross-modal ratio under a fixed total budget) is a principled approach to this problem that is only possible in a joint training framework. The empirical finding that a 1:3 ratio of language to multimodal data yields optimal performance highlights how much multimodal learning benefits from continued text exposure during joint training.

**Instruction-tuned LLMs are used as initialization, but base models would be more appropriate.** The paper makes a specific design choice that differs from common practice: "our LLM components are initialized solely from pre-trained base models, without employing instruction-tuned variants" (Section 2.1). This is significant because instruction-tuned LLMs have already been optimized for a particular interaction pattern (following instructions, generating helpful responses) that may not align with the raw autoregressive pre-training that InternVL3 employs. Starting from base models means the LLM begins with only its foundational next-token prediction capability, allowing the joint pre-training process to shape how linguistic and multimodal capabilities develop together without interference from prior instruction-tuning biases.

**Existing models show uneven capability profiles.** The comprehensive evaluation in Section 3 reveals a pattern where previous open-source MLLMs (including InternVL2.5) show strong performance on some task categories but significant weaknesses on others. For example, InternVL2.5-78B achieves 89.1 on AI2D but only 72.3 on MathVista—a gap that suggests specialized visual understanding (diagrams) was better learned than mathematical reasoning with visual elements. InternVL3's more balanced improvements across benchmarks (e.g., 79.0 on MathVista, a 7.3-point gain) suggest that joint pre-training produces more uniformly capable multimodal representations rather than representations that excel in domains where the vision-text alignment happens to be easier.

### How This Paper Positions Itself Relative to Existing Work

**Relative to the InternVL lineage.** This paper explicitly positions itself as building on InternVL [21], InternVL2 [19, 20], and InternVL2.5 [18]. The architecture ("ViT-MLP-LLM") is inherited from InternVL2.5, as are techniques like pixel unshuffle for high-resolution image processing and square loss re-weighting for SFT. What is new is the *training paradigm*: rather than the multi-stage pipeline (MLP warmup → instruction tuning → optional RLHF) used by predecessors, InternVL3 advocates a single unified pre-training stage followed by SFT and MPO. The paper frames this as a simplification that simultaneously improves performance—contrary to what one might expect if the multi-stage approach were optimal.

**Relative to contemporary open-source MLLMs.** The paper's primary comparison targets are the Qwen2-VL [121] and Qwen2.5-VL [7] series, which represent the strongest open-source competition at the time of writing. Qwen2.5-VL uses a similar "ViT-MLP-LLM" architecture but, critically, follows the conventional approach of adapting pre-trained Qwen2.5 LLMs through multimodal training stages. By benchmarking against Qwen2.5-VL models of comparable scale (e.g., InternVL3-8B vs. Qwen2.5-VL-7B, InternVL3-78B vs. Qwen2.5-VL-72B), the paper aims to demonstrate the superiority of native multimodal pre-training over post-hoc adaptation when architectural components are similar. The results in Table 2 (Section 3.2) bear this out: InternVL3-78B achieves 72.2 on MMMU vs. Qwen2.5-VL-72B's 68.2, a sizable margin for this benchmark.

However, the paper is careful to acknowledge that the comparison is not perfectly apples-to-apples. InternVL3 uses InternViT-6B as its vision encoder, while Qwen2.5-VL uses its own vision encoder architecture (Qwen-ViT). The overall parameter counts are also not identical (78B vs. 72B). The paper does not exhaustively control for these differences, leaving some ambiguity about whether the gains come from the pre-training paradigm specifically or from differences in model scale and vision encoder quality.

**Relative to proprietary models.** The paper positions InternVL3 as "competitive with leading proprietary models, including ChatGPT-4o, Claude 3.5 Sonnet, and Gemini 2.5 Pro" (Section 1). The evidence supports this but with important nuance. InternVL3-78B surpasses GPT-4o-20241120 on MMMU (72.2 vs. 70.7) and MathVista (79.0 vs. 60.0), but trails Gemini-2.5-Pro on both benchmarks (74.7 and 80.9, respectively). The paper frames this gap not as a fundamental limitation but as evidence that "there remains room for further refinement" (Section 3.1), positioning InternVL3 as closing the open-proprietary gap rather than closing it entirely.

**Relative to the preference optimization literature.** The adoption of Mixed Preference Optimization (MPO) [124] positions InternVL3 at the intersection of the RLHF/DPO lineage and multimodal training. MPO is a composite loss combining DPO (relative preference), BCO (absolute quality), and LM loss (generation quality). The paper argues that this addresses a specific failure mode: the distribution shift between training (conditioned on ground-truth tokens) and inference (conditioned on model-generated tokens) impairs Chain-of-Thought reasoning. By training on rollouts from the model itself (positive and negative), MPO aligns the training and inference distributions. This is a more nuanced intervention than standard SFT, and the ablation in Table 13 (Section 3.14) shows consistent gains of 0.5–4.5 points across model scales, with larger gains on harder reasoning benchmarks like MathVerse (e.g., 44.2 → 51.0 for InternVL3-78B).

**Relative to test-time scaling literature.** The paper's test-time scaling approach (Section 2.4) uses VisualPRM, a step-level process reward model that scores individual reasoning steps and is used for best-of-N selection at inference time. This positions InternVL3 within the growing body of work on inference-time compute scaling [108, 94, 87, 70], but with a multimodal twist: the process reward model must assess reasoning about visual inputs, not just text. The paper's contribution is not a new test-time scaling method but rather demonstrating that such methods transfer effectively to multimodal reasoning and that they compound with the gains from native multimodal pre-training.

**The paper's unique value proposition.** What distinguishes InternVL3 from contemporary MLLMs is not any single technique—V2PE, MPO, test-time scaling, and joint training have each been explored in isolation—but rather the integration of all of these into a coherent training recipe that starts from base models and jointly learns vision and language from scratch. The paper's central claim is that this integrated approach yields a more capable, more balanced multimodal model than the sum of individually applied techniques would suggest. The evidence for this is the consistent and broad-based improvements across 8 benchmark categories (Tables 2–8) rather than isolated gains on a few tasks.

## 3. Technical Approach

### 3.1 Reader Orientation

InternVL3 is a family of multimodal large language models that process images, videos, and text through a unified Transformer architecture, producing text responses. The system solves the problem that conventional MLLMs train language and vision separately—first building a text-only LLM, then retroactively bolting on vision—by **jointly learning linguistic and multimodal capabilities from interleaved text and multimodal data in a single pre-training stage**. The "shape" of the solution is a single training phase that replaces the standard two-to-three-stage pipeline (language pre-training → multimodal alignment → instruction tuning) with an integrated process, followed by supervised fine-tuning and preference optimization for refinement.

### 3.2 Big-Picture Architecture (Diagram in Words)

The InternVL3 system has five major components connected in a pipeline:

1. **Vision Encoder (InternViT)**: Two configurations available—InternViT-300M (for smaller models up to 14B parameters) and InternViT-6B (for 38B and 78B models). Converts images into visual tokens through a Transformer-based vision backbone. Operates at 448×448 pixel resolution, with pixel unshuffle reducing token count by 4× (each tile produces 256 visual tokens).

2. **MLP Projector**: A randomly initialized two-layer multilayer perceptron that maps visual tokens from the ViT's output space into the LLM's input embedding space. Unlike conventional approaches where only this component is trained during multimodal alignment, here it is trained jointly with all other parameters from the start.

3. **Language Model (LLM)**: Pre-trained base models from the Qwen2.5 series (0.5B, 1.5B, 7B, 14B, 32B, 72B) or InternLM3-8B, initialized from base (non-instruction-tuned) checkpoints. Processes both text tokens and projected visual tokens autoregressively.

4. **Variable Visual Position Encoding (V2PE)**: A modified positional encoding scheme where visual tokens receive fractional position increments (δ < 1) while text tokens receive the standard increment of 1. This allows extended multimodal contexts without blowing up the position index range.

5. **Training and Post-Training Pipeline**: Native multimodal pre-training on ~200B tokens (50B language + 150B multimodal) using an autoregressive objective with square averaging loss re-weighting, followed by Supervised Fine-Tuning (SFT) on 21.7M samples and Mixed Preference Optimization (MPO) on ~300K preference pairs, with test-time best-of-N selection using VisualPRM-8B as a step-level critic.

Information flows as follows: an image enters → pixel unshuffle reduces spatial resolution → ViT processes patches into visual tokens → MLP projects into LLM embedding space → V2PE assigns fractional position indices to visual tokens → LLM processes the combined sequence of visual and text tokens autoregressively → text tokens are generated as output → during training, loss is computed only on text tokens using square averaging.

### 3.3 Roadmap for the Deep Dive

- **First**, the multimodal autoregressive formulation (Equations 1–8), which defines how visual and text tokens are combined into a single sequence, how position indices are computed with V2PE, how the loss function works, and why the loss is restricted to text tokens only. This is the mathematical core of the model.
- **Second**, the native multimodal pre-training procedure, including the data composition, the two-stage sampling ratio determination strategy, the 1:3 language-to-multimodal ratio finding, and why joint parameter optimization departs from conventional practice.
- **Third**, the supervisied fine-tuning (SFT) phase, covering data expansion (16.3M → 21.7M samples), techniques inherited from InternVL2.5 (random JPEG compression, square loss re-weighting, multimodal data packing), and new domain coverage.
- **Fourth**, Mixed Preference Optimization (MPO), including the composite loss function (Equations 9–13), why each component (preference loss, quality loss, generation loss) is needed, and how preference pairs are constructed from model rollouts.
- **Fifth**, test-time scaling with VisualPRM, covering the multi-turn chat formulation for step-level scoring (Equation 14), the best-of-N selection strategy, and the data expansion from VisualPRM400K.
- **Sixth**, the training infrastructure (InternEVO), including the decoupled sharding strategies for ViT/MLP/LLM, the dynamic load balancing for imbalanced visual-text token ratios, and the 50–200% training speedup claim.

### 3.4 Detailed, Sentence-Based Technical Breakdown

This is primarily a **systems and methods paper** whose core idea is that jointly pre-training vision and language components from base model initializations—rather than adapting a pre-trained text-only LLM to vision—produces a more capable, more balanced multimodal model with a simpler training pipeline.

---

#### Multimodal Autoregressive Formulation

The fundamental representational decision in InternVL3 is how to combine tokens from different modalities into a single sequence that a Transformer can process. The model represents every training sample as a sequence of tokens drawn from a shared vocabulary, where each position can hold either a text token embedding, a visual patch embedding, or other modality-specific representations (e.g., video patch embeddings).

**Token sequence representation.** For an arbitrary training sample with token length L, the sequence is written as:

$$x = (x_1, x_2, \ldots, x_L)$$

where each `$x_i$` can be a textual token embedding, a visual embedding, or another modality-specific representation.

**What it represents:** This is simply a list of L vectors, each of which is the embedding of either a word/subword (text token), an image patch (visual token), or other modality data. The Transformer processes this sequence left-to-right, attending across all positions regardless of modality.

**Why this matters:** This unified sequence representation is what makes native multimodal pre-training possible. There is no separate processing pathway for images versus text—all tokens enter the same Transformer and interact through the same attention mechanism. This means that from the very first gradient update, the model learns to relate visual and textual information through shared attention patterns, rather than having these relationships mediated by a separately-trained projector.

**Position encoding with V2PE.** The position index for each token is computed sequentially using a modality-dependent recursive function:

$$p_i = \begin{cases} 0, & \text{if } i = 1 \\ f_{\text{pos}}(p_{i-1}, x_i), & \text{for } i = 2, 3, \ldots, N \end{cases}$$

where `$p_i$` is the position index assigned to token i, and `$f_{\text{pos}}$` is a function that determines the next position index based on the previous index and the current token's modality.

This general formulation is then specialized to:

$$p_i = p_{i-1} + \begin{cases} 1, & \text{if } x_i \text{ is a textual token} \\ \delta, & \text{if } x_i \text{ is a visual token} \end{cases}$$

where `$\delta$` is a fractional increment (`$\delta < 1$`) drawn randomly during training from the set:

$$\delta \in \Delta = \left\{1, \frac{1}{2}, \frac{1}{4}, \frac{1}{8}, \frac{1}{16}, \frac{1}{32}, \frac{1}{64}, \frac{1}{128}, \frac{1}{256}\right\}$$

**What it computes:** For text tokens, position indices increase by 1 as usual (the first text token gets position p, the next gets p+1, etc.). For visual tokens, position indices increase by the much smaller fraction δ (e.g., 1/256). This means a sequence of 256 visual tokens advances the position counter by only 1 unit rather than 256 units. During training, δ is randomly sampled per image from the set above, so the model learns to handle visual tokens at various effective "resolutions" in the position space. During inference, δ can be chosen flexibly based on the total sequence length needed.

**Why this form:** The motivation is straightforward but the mechanism is clever. Standard position encodings treat every token identically: position p+1 follows position p regardless of what p+1 contains. When processing high-resolution images, this causes the position counter to advance rapidly—a single 448×448 image tile contributes 256 tokens, consuming 256 position slots. For multi-image or video contexts, the position budget explodes. V2PE exploits the intuition that visual tokens are spatially local and inherently ordered by their 2D arrangement, so they don't need the same "positional bandwidth" as text tokens, where word order is critical to meaning. By giving visual tokens fractional increments, the position counter advances slowly through visual content, leaving more room for text tokens and enabling longer multimodal contexts.

The training-time randomization of δ is critical: if the model were trained with a single fixed δ, it would learn position embeddings specialized to that specific increment and might fail to generalize when a different δ is used at inference. Randomizing δ forces the position embeddings to learn interpolation behavior across a range of visual token densities.

It is important to note (Section 3.14) that for all results in the main paper besides the V2PE ablation in Table 12, δ is fixed to 1—meaning standard position encoding is used. The V2PE benefits are demonstrated separately in the ablation; the main results use the simpler scheme to ensure fair comparison with prior work.

**Autoregressive training objective.** InternVL3 uses the standard left-to-right autoregressive language modeling objective:

$$\mathcal{L}_{\text{full}}(\theta) = -\sum_{i=2}^{L} w_i \cdot \log p_\theta(x_i \mid x_1, \ldots, x_{i-1})$$

where `$\theta$` represents all model parameters, `$L$` is the sequence length, `$p_\theta(x_i \mid x_1, \ldots, x_{i-1})$` is the model's predicted probability for the actual token at position i given all previous tokens, and `$w_i$` is a per-token loss weight.

However, this full formulation would compute loss on visual tokens as well as text tokens—that is, the model would be penalized for incorrectly predicting image patches. The paper argues that this is not the desired behavior: the model should use visual information as conditioning context, not as prediction targets. Therefore, the loss is restricted:

$$\mathcal{L}_{\text{text-only}}(\theta) = -\sum_{\substack{i=2 \\ x_i \in \text{Text}}}^{L} w_i \cdot \log p_\theta(x_i \mid x_1, \ldots, x_{i-1})$$

**What it computes:** For each position i in the sequence, if the token at position i is a text token, the model computes the negative log-probability of that token under the model's predicted distribution (conditioned on all previous tokens, which may include both text and visual tokens). These per-token losses are weighted by `$w_i$` and summed. Visual tokens contribute to the conditioning (they appear in `$x_1, \ldots, x_{i-1}$`) but do not generate loss signals themselves.

**Why this form:** This selective objective is the standard approach in MLLM pre-training (used in InternVL2.5 and elsewhere). The rationale has two parts. First, predicting visual tokens is a fundamentally different task than predicting text tokens—it requires modeling continuous pixel distributions or quantized patch identities, which introduces complexity and potential optimization conflicts with the primary goal of language generation. Second, and more importantly, the downstream use case is always text generation conditioned on images, not image generation. Training the model to predict visual patches would consume compute on a capability that is never used at inference time, while potentially distorting the representations learned for the actual task. By restricting loss to text tokens, the model learns to embed visual information in a way that is maximally useful for language decoding—exactly the skill needed for VQA, captioning, reasoning, and all other evaluated tasks.

**Token weight selection: square averaging.** The per-token weight `$w_i$` in the loss function controls how different-length training samples contribute to the total gradient. The paper discusses three strategies:

$$w_i = \begin{cases} \frac{1}{l^0}, & \text{for token averaging} \\ \frac{1}{l^{0.5}}, & \text{for square averaging} \\ \frac{1}{l^1}, & \text{for sample averaging} \end{cases}$$

where `$l$` is the number of tokens in the training sample on which the loss needs to be calculated (text tokens only, since visual tokens are excluded from loss computation).

**What it computes:** These formulas determine how much each token's loss contributes to the total, relative to the sample's length. With token averaging (`$l^0 = 1$`), every token has equal weight regardless of sample length—this biases gradients toward longer responses because they contain more tokens. With sample averaging (`$l^1 = l$`), the total loss for each sample is divided by its length, making all samples contribute equally—this biases gradients toward shorter responses because each token in a long sample contributes less than each token in a short sample. Square averaging (`$l^{0.5} = \sqrt{l}$`) interpolates between these extremes.

**Why square averaging:** This is inherited from InternVL2.5 and addresses a practical training instability. If long responses dominate gradients (token averaging), the model may overfit to verbose patterns and fail on tasks requiring concise answers. If short responses dominate (sample averaging), the model may not learn to produce the detailed, multi-step reasoning needed for complex tasks. Square averaging provides a middle ground where longer samples contribute more total gradient than shorter ones (rewarding detail) but not proportionally to their length (preventing dominance). The square root specifically is an empirical choice with no deeper theoretical justification beyond working well in practice (as demonstrated in InternVL2.5).

**Joint parameter optimization.** The complete pre-training objective is:

$$\theta^* = \arg\min_\theta \mathbb{E}_{x \in \mathcal{D}_{\text{multi}}} \left[\mathcal{L}_{\text{text-only}}(\theta)\right]$$

where `$\mathcal{D}_{\text{multi}}$` is the union of large-scale text-only and multimodal corpora (image-text pairs, video-text pairs, interleaved image-text sequences, and pure text documents).

**What it computes:** This is a straightforward empirical risk minimization: find parameters θ (all of them—ViT, MLP, LLM) that minimize the expected text-only autoregressive loss over the combined training data distribution.

**Why this form over alternatives:** The key phrase is "all model parameters jointly." In conventional post-hoc MLLM training, this optimization would be split across stages with different parameter subsets frozen at each stage. For example, a typical pipeline might: (1) train the LLM on text only (θ_LLM optimized, everything else not yet connected), (2) train the MLP projector while freezing the LLM and ViT (only θ_MLP updated), (3) optionally fine-tune the LLM with the ViT still frozen or with a lower learning rate. InternVL3 collapses all of this into a single optimization where every parameter is updated from every batch—visual features, the projection mapping, and the language model all evolve together in response to the same loss signal. The paper argues that this ensures "both linguistic and visual features evolve synchronously" rather than visual features being warped to fit a frozen linguistic representational space established during text-only pre-training.

---

#### Native Multimodal Pre-Training

**What makes it "native."** The term "native multimodal pre-training" refers to the fact that InternVL3 acquires multimodal capabilities *during* the pre-training phase itself, rather than in a separate post-hoc adaptation stage. In conventional MLLM pipelines, the model first undergoes language-only pre-training, then (optionally) language post-training (instruction tuning), and only then is exposed to multimodal data through an alignment stage. InternVL3 eliminates this separation: multimodal data (images, videos, interleaved sequences) and pure text data are mixed together from the very first pre-training step.

This is not merely a scheduling change—it changes what the model learns at the representational level. When a model sees "A cat is sitting on a mat" and an image of a cat on a mat in the same training batch, the gradients from both samples update the same parameters simultaneously. The LLM's internal representations for the word "cat" are shaped not only by text co-occurrence statistics (which cats appear with in sentences) but also by the visual features of actual cats. This is fundamentally different from a model that first learns "cat" from text alone and then has visual cat features projected into that pre-existing representational neighborhood.

**Data composition.** The pre-training data is divided into two categories:

*Multimodal data*: Built on the pre-training corpus from InternVL2.5, covering image captioning, general VQA, mathematics, charts, OCR, knowledge grounding, document understanding, multi-turn dialogue, and medical data. Additionally, new data is incorporated for GUI understanding, tool usage, 3D scene understanding, and video comprehension. The total scale of multimodal data is approximately 150 billion tokens.

*Pure language data*: Primarily constructed from InternLM2.5's pre-training data, augmented with open-source text datasets including SmolLM-Corpus and additional mathematical/reasoning corpora. The language data serves to "preserve and amplify the model's capabilities in language understanding and generation" and compensates for the fact that multimodal data tends to have "relatively short and less diverse textual content." The total scale of language data is approximately 50 billion tokens.

**Two-stage sampling ratio strategy.** Determining the right balance between multimodal and language data is non-trivial because the two data types have different characteristics (length distributions, difficulty levels, domain coverage) and the model must perform well on both pure language and multimodal benchmarks. The paper describes a two-stage empirical procedure:

*Stage 1 (within-modality optimization)*: Train separate models on only the multimodal dataset and only the language dataset. Evaluate each on its respective benchmarks to identify optimal sampling ratios *within* each modality (e.g., what fraction of multimodal data should be chart understanding vs. OCR vs. general VQA).

*Stage 2 (cross-modality optimization)*: Under a fixed total training budget, combine the two modalities and sweep their relative sampling ratio. The paper reports: "Empirical studies show that a 1:3 ratio of language to multimodal data yields the best overall performance across both unimodal and multimodal benchmarks." This means for every 1 token of pure text data, the model sees 3 tokens of multimodal data.

The total training volume is approximately 200 billion tokens (50B language + 150B multimodal), which is modest compared to the multi-trillion-token budgets used for pure language model pre-training (reflecting the higher cost and lower availability of multimodal data).

**Why joint training improves language capabilities.** A surprising result (Section 3.13, Table 11) is that InternVL3 models outperform their Qwen2.5 Chat counterparts on pure language benchmarks, despite both being initialized from the same Qwen2.5 base models. The InternVL3-78B achieves 86.9 on MMLU versus Qwen2.5-72B Chat's 84.4, and 90.5 on GSM8K versus 88.2. The paper attributes this to three factors: (1) approximately 25% of training data is pure language, maintaining linguistic competency; (2) joint optimization allows language representations to benefit from multimodal grounding (e.g., mathematical concepts learned from both text proofs and visual diagrams); and (3) the subsequent SFT and MPO stages use high-quality textual corpora that further improve language performance. This is a significant finding because it challenges the assumption that multimodal training inevitably degrades language capabilities—with the right data mixture and joint optimization, multimodal training can actually *improve* language understanding.

---

#### Supervised Fine-Tuning (SFT)

**Purpose.** After native multimodal pre-training produces a model with broad multimodal and linguistic capabilities, the SFT phase trains the model to follow instructions and produce high-quality responses in a conversational format. This is standard for LLMs and MLLMs but with specific enhancements.

**Techniques inherited from InternVL2.5.** Three techniques are explicitly mentioned as carried forward:

- **Random JPEG compression**: During training, images are randomly JPEG-compressed at varying quality levels. This acts as data augmentation, making the model robust to image artifacts, compression noise, and varying image quality—all common in real-world inputs (screenshots, user-uploaded photos, scanned documents).

- **Square loss re-weighting**: The same square averaging scheme (`$w_i = 1/\sqrt{l}$`) from pre-training is used during SFT to balance contributions from short and long responses.

- **Multimodal data packing**: Multiple training samples are concatenated into single sequences to maximize GPU utilization and reduce padding waste. This requires careful attention masking to prevent cross-sample attention.

**Data expansion relative to InternVL2.5.** The SFT training data grows from 16.3M samples in InternVL2.5 to 21.7M samples in InternVL3—a 33% increase. The new data specifically targets domains that were underrepresented in InternVL2.5: tool usage (training the model to interact with external APIs and tools), 3D scene understanding (spatial reasoning about three-dimensional environments), GUI operations (interpreting and interacting with graphical user interfaces), long context tasks, video understanding, scientific diagrams, creative writing, and multimodal reasoning.

This expansion addresses a key limitation of InternVL2.5: while it performed well on standard multimodal benchmarks, it lacked capabilities for emerging application areas like GUI agents and spatial reasoning. The paper validates these targeted improvements through dedicated evaluations: Table 9 (GUI grounding) shows InternVL3-78B achieving 88.7% on ScreenSpot and 90.9% on ScreenSpot-V2, and Table 10 (spatial reasoning) shows InternVL3-38B outperforming GPT-4o on VSI-Bench (48.9 vs. 34.0 overall).

---

#### Mixed Preference Optimization (MPO)

**The distribution shift problem.** SFT trains the model to predict the next token conditioned on *ground-truth* previous tokens (teacher forcing). At inference time, however, the model conditions on its *own previously generated* tokens. This discrepancy means that errors accumulate: if the model makes a small mistake early in a chain-of-thought reasoning trace, subsequent tokens are conditioned on an erroneous context that was never seen during training. This is particularly damaging for multi-step reasoning tasks like mathematical problem solving.

MPO addresses this by introducing training on model-generated rollouts (both correct and incorrect), aligning the training distribution more closely with the inference distribution.

**The MPO loss function.** MPO combines three loss terms with weighting coefficients:

$$L = w_p L_p + w_q L_q + w_g L_g$$

where `$w_p$`, `$w_q$`, and `$w_g$` are hyperparameters controlling the contribution of the preference loss, quality loss, and generation loss respectively.

*Preference loss (DPO)*: This is the standard Direct Preference Optimization loss, which teaches the model to prefer chosen responses over rejected ones by comparing their relative likelihoods under the current policy versus a reference policy:

$$L_p = -\log \sigma\left(\beta \log \frac{\pi_\theta(y_c \mid x)}{\pi_0(y_c \mid x)} - \beta \log \frac{\pi_\theta(y_r \mid x)}{\pi_0(y_r \mid x)}\right)$$

where `$x$` is the user query (which may include images), `$y_c$` is the chosen (preferred) response, `$y_r$` is the rejected response, `$\pi_\theta$` is the current policy model, `$\pi_0$` is the reference model (the SFT checkpoint, frozen), `$\beta$` is the KL penalty coefficient controlling how far the policy can diverge from the reference, and `$\sigma$` is the sigmoid function.

**What it computes:** The DPO loss encourages the model to increase the log-ratio `$\log(\pi_\theta(y_c|x)/\pi_0(y_c|x))$` for chosen responses (making them more likely under the current policy than they were under the reference) while decreasing the same ratio for rejected responses. The sigmoid converts the difference of these log-ratios into a probability that the chosen response is better; the negative log makes this a minimization objective.

**Why this form:** DPO is chosen over RLHF because it avoids training a separate reward model—the preference signal comes directly from the comparison of policy and reference likelihoods. The KL penalty β prevents the policy from diverging too far from the reference, which helps maintain generation quality and prevents reward hacking.

*Quality loss (BCO)*: Binary Classifier Optimization loss teaches the model to assess the absolute quality of individual responses, not just relative preferences:

$$L_q = L_q^+ + L_q^-$$

where the individual terms are:

$$L_q^+ = -\log \sigma\left(\beta \log \frac{\pi_\theta(y_c \mid x)}{\pi_0(y_c \mid x)} - \delta\right)$$

$$L_q^- = -\log \sigma\left(-\left(\beta \log \frac{\pi_\theta(y_r \mid x)}{\pi_0(y_r \mid x)} - \delta\right)\right)$$

with `$\delta$` being a reward shift computed as the moving average of previous rewards to stabilize training.

**What it computes:** `$L_q^+$` encourages the log-ratio for chosen responses to be greater than the threshold δ—if the chosen response is already above δ, the loss is small; if below, the loss pushes it up. `$L_q^-$` encourages the log-ratio for rejected responses to be below δ—if already below, small loss; if above, pushes it down. Together, these terms establish δ as a "quality threshold" that separates good from bad responses.

**Why this form:** DPO alone only captures relative preferences (A is better than B) but not absolute quality (both A and B might be bad, or both good). BCO adds an absolute quality signal that helps the model learn what makes a response good regardless of comparison. The moving average δ adapts the threshold during training, preventing it from being a fixed hyperparameter that might become inappropriate as the model improves.

*Generation loss (LM loss)*: This is the standard language modeling loss from Equation 6, computed only on the chosen (preferred) responses:

$$L_g = -\sum_{\substack{i=2 \\ x_i \in \text{Text}}}^{L} w_i \cdot \log p_\theta(x_i \mid x_1, \ldots, x_{i-1})$$

**What it computes:** This is identical to the pre-training loss but applied only to chosen responses during MPO training. It ensures the model maintains the basic ability to generate the preferred responses fluently, rather than only learning preference distinctions.

**Why this form:** Preference optimization methods like DPO can sometimes degrade generation quality because they focus on relative likelihood ratios rather than absolute likelihoods. The LM loss term anchors the policy to actually produce the chosen responses, preventing degeneration.

**Preference data construction.** The MPO data is built on the MMPR v1.2 pipeline, covering domains including general VQA, science, charts, mathematics, OCR, and documents. The key procedural detail: "We use the SFT versions of InternVL3-8B, 38B, and 78B to generate rollouts." This means that to create preference pairs, the SFT-trained model generates multiple candidate responses to each query. Some of these are correct (chosen) and some are incorrect (rejected). The preference pairs cover about 300K samples, and notably "all models [8B, 38B, 78B] are trained on the same dataset"—only the rollout generation uses model-specific checkpoints; the final preference data is shared.

This approach of using the model's own SFT checkpoint to generate rollouts is what theoretically addresses the distribution shift: the rollouts come from a distribution closer to what the model will actually produce at inference time, so training on them reduces the train-test discrepancy.

**MPO effectiveness (Table 13).** The ablation in Section 3.14 shows consistent gains from MPO across all model scales and reasoning benchmarks. InternVL3-78B improves from 74.0 to 79.0 on MathVista (+5.0) and from 44.2 to 51.0 on MathVerse (+6.8). The overall average across seven reasoning benchmarks improves by 4.1 points for the 78B model. Critically, the paper notes: "the training data used for MPO is a subset of that used for SFT, indicating that the performance improvements primarily stem from the training algorithm rather than the training data." This is a strong claim that the MPO loss formulation itself—not additional or higher-quality data—drives the gains.

---

#### Test-Time Scaling with VisualPRM

**Overview.** Test-time scaling refers to spending additional computation at inference time to improve output quality. InternVL3 uses a Best-of-N strategy: generate N candidate solutions, score each with a process reward model, and select the highest-scoring one.

**Visual Process Reward Model (VisualPRM).** VisualPRM-8B is a separate 8B-parameter MLLM trained to evaluate the quality of individual reasoning steps. Unlike outcome reward models (which score complete solutions) or conventional process reward models for text-only math, VisualPRM must assess reasoning about visual inputs.

The model is used through a multi-turn chat formulation:

$$c_i \sim M(y_i \mid I, q, s_{\leq i})$$

where `$I$` is the input image, `$q$` is the question, `$s_{\leq i}$` represents the solution steps up to and including step i, `$M$` is the VisualPRM model, `$y_i$` is the model's binary judgment ("+" or "−"), and `$c_i \in \{+, -\}$` is the predicted correctness of step i.

**What it computes:** In the first turn, VisualPRM receives the image, the question, and the first step of the solution. It outputs "+" if it believes the first step is correct, "−" otherwise. In the second turn, it receives the image, question, first step, and second step, and judges the second step. This continues for all steps. During inference, rather than discrete "+"/"−" tokens, the score for each step is the probability the model assigns to generating "+", providing a continuous score. The overall solution score is the average of per-step scores.

**Why this multi-turn formulation:** By presenting steps sequentially in a chat format, VisualPRM can leverage the MLLM's pre-existing conversational capabilities and attention mechanisms. Each step is evaluated in the context of all previous steps, allowing the model to detect inconsistencies or reasoning errors that only become apparent when considering the full chain of reasoning. The binary "+"/"−" output simplifies training to a classification task with clear supervision.

**Training data expansion.** VisualPRM is trained on VisualPRM400K, which was constructed from MMPR v1.2 questions. The paper states: "we further expand VisualPRM400K by sampling rollouts from the 8B and 38B variants of InternVL3." This means the training data for the process reward model includes solutions generated by InternVL3 itself (not just external models), which should improve the PRM's calibration on InternVL3's specific error patterns.

**Best-of-N procedure.** During evaluation on reasoning and mathematics benchmarks, the model generates N candidate solutions (N=8 is used for the "w/ VisualPRM-Bo8" results). VisualPRM scores each solution by evaluating each step and averaging the scores. The solution with the highest average score is selected as the final answer.

**Effectiveness.** The best-of-8 strategy yields substantial gains. For InternVL3-38B on MathVerse Vision-Only, the score improves by 6.0 percentage points (48.2 → 54.2). For InternVL3-78B, the gain is 3.2 points (51.0 → 54.2). Smaller models benefit even more proportionally: InternVL3-1B improves from 18.7 to 28.9 (+10.2 points), and InternVL3-2B improves from 25.3 to 36.7 (+11.4 points). This pattern—larger relative gains for smaller models—is consistent with the intuition that smaller models produce more diverse (and error-prone) candidate solutions, giving the PRM more room to discriminate quality.

---

#### Training Infrastructure (InternEVO)

**The computational challenge.** Training a 78B-parameter MLLM on 200B tokens with high-resolution images presents several infrastructure challenges: (1) the ViT, MLP projector, and LLM have very different computational profiles (the ViT processes fixed-size image patches with quadratic attention, while the LLM processes variable-length sequences of text and visual tokens), (2) visual and text tokens appear in different ratios per sample, causing load imbalance across GPUs, and (3) long multimodal sequences (up to 32K tokens) require memory-efficient parallelism strategies.

**InternEVO extensions.** The InternEVO framework, originally designed for LLM training with ZeRO optimization, is extended for MLLM training with several key features:

- **Decoupled sharding strategies**: The ViT, MLP, and LLM components can use different parallelism configurations independently. This is important because the ViT benefits from data parallelism (processing different images on different devices) while the LLM benefits from tensor parallelism (splitting large weight matrices across devices). A unified sharding strategy would be suboptimal for one or both components.

- **Comprehensive parallelism support**: The framework supports data, tensor, sequence, and pipeline parallelism in arbitrary combinations. For sequences up to 32K tokens, head-parallel and sequence-parallel techniques are used to overcome memory bottlenecks.

- **Dynamic load balancing**: The varying proportions of visual to textual tokens per sample create computational imbalances—some GPUs might get image-heavy batches (more ViT work) while others get text-heavy batches (more LLM work). InternEVO includes techniques to "dynamically balance computational workloads across modules, ensuring efficient and equitable resource utilization."

- **Communication-computation overlapping**: By overlapping gradient communication with forward/backward computation, the framework reduces idle GPU time.

**Reported speedups.** Compared to InternVL2.5 training, InternEVO provides "a training speedup of 50% to 200% for models of comparable size, given the same computational budget." The wide range (50–200%) suggests the speedup depends heavily on model scale and configuration—the larger speedups likely apply to the largest models where parallelism overhead was most severe in the previous system. The paper notes that InternEVO formulates an "optimization objective that identifies the optimal configuration to minimize both memory consumption and communication overhead"—essentially an auto-tuning step that searches over parallelism strategies to find the most efficient setup for each model scale.

**Why this matters for the paper's contributions.** The infrastructure improvements are not the main technical contribution, but they are pragmatically essential. Native multimodal pre-training requires training larger models with more complex data mixtures than conventional post-hoc pipelines. Without the 50–200% speedup, the computational cost of this approach might be prohibitive for many research groups, limiting the impact of the open-source release. The paper's commitment to releasing training data alongside model weights is partially enabled by having infrastructure that makes large-scale multimodal training feasible.

## 4. Key Insights and Innovations

### Innovation 1: Native Multimodal Pre-Training Reframes Vision-Language Learning as Integrated Foundation Building Rather Than Retroactive Alignment

The dominant paradigm for building multimodal large language models—exemplified by Qwen2-VL [121], LLaVA-OneVision [60], and InternVL's own predecessors [21, 19, 18]—treats vision as a modality to be *attached* to a pre-existing language model. The standard pipeline is: pretrain a text-only LLM on trillions of tokens, independently train a vision encoder on image-text contrastive objectives, then connect them through a lightweight adapter with multimodal alignment training, typically freezing or partially freezing the LLM to prevent catastrophic forgetting. This approach implicitly assumes that linguistic competence must be established first and protected during subsequent multimodal exposure.

InternVL3 challenges this assumption at a fundamental level. The paper's central conceptual move is the claim that linguistic and visual capabilities are not competing objectives that must be sequenced to avoid interference, but are *mutually informative* when learned jointly from the start. By initializing from base (non-instruction-tuned) LLM checkpoints and jointly optimizing all parameters—ViT, MLP projector, and LLM—on interleaved text and multimodal data, InternVL3 eliminates the artificial separation between "language learning" and "vision learning" that characterizes prior work.

This is not merely a scheduling change. The post-hoc paradigm creates a representational bottleneck: the LLM's internal geometry is optimized exclusively for text during its foundational training, and visual features must later be warped into this pre-existing space through the MLP projector. No amount of fine-tuning can fully compensate for the fact that the representational space was never shaped by visual signal during its most plastic phase. By contrast, native multimodal pre-training allows the LLM's internal representations to develop in response to both text co-occurrence statistics and visual grounding signals simultaneously—a learning dynamic that is fundamentally unavailable in sequential training.

The empirical evidence for this conceptual claim is substantial and appears across multiple lines:

**Language capabilities improve rather than degrade (Table 11).** The most striking evidence for mutual informativeness is that InternVL3 models *outperform* their Qwen2.5 Chat counterparts on pure language benchmarks, despite both being initialized from the same Qwen2.5 base models. InternVL3-78B achieves 86.9 on MMLU versus Qwen2.5-72B Chat's 84.4—a 2.5-point advantage on a benchmark with no visual component whatsoever. On GSM8K, the gap is 90.5 versus 88.2. If vision and language were competing objectives, we would expect multimodal training to degrade language performance, not improve it. The fact that language capabilities are enhanced suggests that visual grounding provides a complementary training signal that enriches linguistic representations—concepts learned from both text descriptions and visual examples produce more robust understanding than text alone.

**Even without post-training, multimodal capabilities emerge (Figure 3).** In the ablation study comparing InternVL2-8B trained with conventional MLP warmup versus native multimodal pre-training, the natively pre-trained model achieves performance comparable to the fully multi-stage-trained InternVL2-8B *before any instruction tuning*. After subsequent SFT on the same data, the natively pre-trained model further outperforms the conventional pipeline. This demonstrates that joint pre-training imparts multimodal capabilities more efficiently than the multi-stage approach—the model learns more from the same data because vision and language parameters co-evolve rather than one being frozen while the other adapts.

**Balanced improvements across diverse benchmarks (Tables 2–8).** Prior InternVL models showed uneven capability profiles—strong on diagram understanding (89.1 on AI2D for InternVL2.5-78B) but proportionally weaker on mathematical reasoning with visual elements (72.3 on MathVista, a significant gap). InternVL3-78B achieves 89.7 on AI2D and 79.0 on MathVista, closing this gap substantially. The 7.3-point improvement on MathVista versus a 0.6-point improvement on AI2D suggests that joint pre-training is particularly beneficial for tasks requiring deep integration of visual and linguistic reasoning, as opposed to tasks where the visual component is primarily about pattern recognition (diagrams). This pattern is consistent with the idea that joint training produces representations where visual and linguistic information are more tightly integrated.

This innovation is **fundamental rather than incremental**. It challenges a core assumption of the MLLM field—that language must be learned before vision—and provides empirical evidence that the opposite may be true. If validated across other model families and scales, this insight could reshape how future multimodal models are trained, potentially eliminating the multi-stage pipelines that currently dominate the field.

The primary caveat is that the comparison to Qwen2.5-VL is not perfectly controlled: InternVL3 uses InternViT as its vision encoder while Qwen2.5-VL uses its own vision architecture, and the overall parameter counts differ (78B vs. 72B). The language capability improvement is the cleaner comparison (same base LLM, same benchmarks) and carries more of the argument's weight.

---

### Innovation 2: Variable Visual Position Encoding Reframes Position Allocation as a Modality-Aware Resource, Not a Uniform Sequential Index

Position encoding in Transformer-based models has traditionally been modality-agnostic: every token, whether representing a word, an image patch, or a special delimiter, advances the position counter by exactly 1. This design choice dates back to the original Transformer architecture and persists in most MLLMs because it is simple and compatible with existing positional encoding mechanisms (RoPE, learned absolute positions, etc.). The implicit assumption is that all tokens contribute equally to positional context and therefore deserve equal positional bandwidth.

V2PE challenges this assumption by introducing a modality-specific position increment: text tokens advance the position counter by 1, while visual tokens advance it by a fractional increment δ (chosen randomly during training from {1, 1/2, 1/4, ..., 1/256}). The insight driving this design is that visual tokens have a fundamentally different positional structure than text tokens. Text semantics are highly order-dependent—swapping two words typically changes meaning. Visual tokens from an image have strong 2D spatial structure that is largely independent of their 1D sequence position, and the model already captures this through patch adjacency and attention patterns. Allocating full positional bandwidth to visual tokens is therefore wasteful: it consumes position index range that could be used for processing longer contexts, without providing commensurate benefits for visual understanding.

This reframing of position encoding as a *resource to be allocated* rather than a *mechanical index* is the conceptual contribution. Prior work on long-context MLLMs focused on architectural solutions (larger position embeddings, sparse attention, memory mechanisms) that preserve the uniform-increment assumption. V2PE instead questions whether visual tokens need the same positional granularity as text tokens at all.

The empirical findings from the V2PE ablation (Table 12) reveal a non-obvious pattern that supports the conceptual argument: even for tasks with *short* multimodal contexts, small δ values (e.g., 1/4 or 1/16) outperform the standard δ=1 encoding. On AI2D, δ=1/4 achieves 81.8 versus 81.7 for δ=1; on InfoVQA, δ=1/4 achieves 71.7 versus 71.4. These are modest improvements, but their existence on benchmarks where context length is not a limiting factor suggests that V2PE is doing more than simply compressing position indices—it may be providing a beneficial inductive bias about the relative importance of positional information across modalities. Visual tokens with smaller position increments are effectively encoded as being "closer together" in position space, which could encourage the model to attend to them as a group, reinforcing spatial coherence.

The training-time randomization of δ is an important practical innovation: by exposing the model to multiple fractional increments during training, V2PE learns to interpolate position embeddings across a range of visual token densities. This means the model can flexibly adapt to different context length requirements at inference time—use a larger δ when the sequence is short, a smaller δ when processing long videos or multi-image inputs—without retraining or architectural modification.

This innovation is **incremental at the mechanism level** (it modifies position encoding, a well-studied component) but **conceptual at the framing level** (it introduces modality awareness into a component previously treated as modality-agnostic). The specific mechanism—fractional increments with randomized training—is a refinement of existing V2PE work [42], but the paper's demonstration that this benefits even short-context tasks distinguishes it from prior long-context techniques that only help when context length is the bottleneck.

The main limitation is that the paper does not extensively evaluate V2PE in the long-context scenarios that motivate it. The V2PE ablation (Table 12) uses standard benchmarks with moderate context lengths; there is no dedicated long-context multimodal evaluation showing, for example, that V2PE enables processing more images or longer videos than would be possible with standard position encoding. The paper also notes that for all main results, δ is fixed to 1 to ensure fair comparison—meaning the V2PE contribution is demonstrated in isolation but not integrated into the largest-scale evaluations.

---

### Innovation 3: The InternVL Scaled Architecture Demonstrates That Open-Source MLLMs Can Achieve Proprietary-Level Performance Through Training Paradigm Innovation, Not Just Scale

The most commonly cited barrier between open-source and proprietary MLLMs is scale: proprietary models like GPT-4o and Gemini benefit from massive training compute budgets, proprietary datasets, and engineering resources that are unavailable to the open-source community. The dominant narrative is that open-source models can approach but not match proprietary performance, and that closing the gap requires either comparable scale (which is prohibitively expensive) or distillation from proprietary models (which raises legal and ethical concerns).

InternVL3 challenges this narrative by demonstrating that **training paradigm innovation can substitute for scale**. InternVL3-78B—trained on approximately 200B tokens with publicly reported data sources and no distillation from proprietary models—achieves results that are competitive with or surpass GPT-4o, Claude 3.5 Sonnet, and Gemini 2.5 Pro on several major benchmarks:

- **MMMU (multidisciplinary reasoning):** InternVL3-78B scores 72.2, surpassing GPT-4o-20241120 (70.7) and Claude-3.7-Sonnet (75.0 versus 72.2, with Gemini at 74.7). This is the headline result because MMMU is considered one of the most challenging and comprehensive multimodal benchmarks, requiring college-level reasoning across 30 subjects.

- **MathVista (mathematical reasoning):** InternVL3-78B scores 79.0, substantially outperforming GPT-4o-20241120 (60.0) and Claude-3.7-Sonnet (66.8), and approaching Gemini-2.5-Pro (80.9). The 19-point gap over GPT-4o is particularly striking and suggests that native multimodal pre-training provides advantages for mathematical reasoning with visual elements that are difficult to replicate through post-hoc alignment.

- **AI2D (diagram understanding):** InternVL3-78B scores 89.7, surpassing both GPT-4o-20240513 (84.6) and GPT-4V (78.2), and matching Gemini-2.5-Pro (89.5).

The significance lies not in any individual benchmark win but in the *pattern of balanced competitiveness*. InternVL3-78B's overall score on the OpenCompass academic leaderboard (79.5, Table 1) places it above Qwen2.5-VL-72B and competitive with Gemini-2.5-Pro, despite using roughly comparable parameter counts to other open-source models. This suggests that the native multimodal pre-training paradigm—not simply scaling up parameters or training data—is responsible for the performance gains.

The paper also provides evidence of scaling efficiency: across model sizes from 1B to 78B, InternVL3 consistently outperforms comparably-sized Qwen2.5-VL models (e.g., InternVL3-8B achieves 73.3 on OpenCompass versus Qwen2.5-VL-7B; InternVL3-14B at 75.5 versus Qwen2.5-VL-72B at comparable levels despite the parameter disadvantage). This suggests the training paradigm advantage persists across scales and is not merely an artifact of the 78B model being larger than some competitors.

This innovation is **fundamental at the level of field narrative but incremental at the level of individual techniques**. The specific methods—joint pre-training, MPO, V2PE—are each refinements of existing ideas. The contribution is their integration into a coherent training recipe that collectively demonstrates open-source models can compete with proprietary ones without relying on proprietary data or distillation. This has practical significance for the open-source community and theoretical significance for debates about whether MLLM performance is primarily a function of compute scale or training methodology.

The paper is appropriately cautious about this claim. It acknowledges that Gemini-2.5-Pro maintains a performance edge on several benchmarks (HallusionBench: 64.1 vs. 59.1; MathVista: 80.9 vs. 79.0) and that the comparison is not perfectly controlled (different vision encoders, slightly different parameter counts). The fact that some benchmarks still favor proprietary models suggests that scale and proprietary data do matter—native multimodal pre-training narrows but does not eliminate the gap.

---

### Innovation 4: The MPO Ablation Reveals That Preference Optimization Improves Reasoning Through Algorithm-Induced Distribution Alignment, Not Data Volume

The paper's ablation study on Mixed Preference Optimization (Table 13) contains a finding that is methodologically significant beyond InternVL3's specific results: MPO training on ~300K preference pairs—a subset of the 21.7M samples used for SFT—improves reasoning performance by 0.5–4.5 points across benchmarks, with the largest gains on the hardest reasoning tasks (MathVerse: +6.8 for InternVL3-78B, MathVision: +7.9). The paper explicitly states that "the training data used for MPO is a subset of that used for SFT, indicating that the performance improvements primarily stem from the training algorithm rather than the training data."

This is a clean demonstration of a principle that is often claimed but rarely isolated: **the value of preference optimization lies in its ability to align the training distribution with the inference distribution, not in providing additional supervision signal**. SFT already exposed the model to correct solutions for all the problems in the MPO dataset. The additional gains come from exposing the model to its *own incorrect rollouts* (the rejected responses in preference pairs) and teaching it to distinguish them from correct ones. This addresses the teacher-forcing discrepancy described in Section 2.3—during SFT, the model never sees its own mistakes because it is always conditioned on ground-truth tokens; during MPO, it learns from negative examples generated from its own distribution.

This finding has implications for the broader RLHF/DPO literature. It suggests that preference optimization is not primarily about providing more data or better data, but about providing *data of a different type*—negative examples from the model's own distribution that teach it to avoid specific failure modes. This reframes the role of preference optimization from "more supervision" to "distribution alignment," which has practical implications for data collection strategies (focus on generating diverse negative rollouts rather than curating additional positive examples) and for understanding when preference optimization is most beneficial (when the model's mistakes follow predictable patterns that can be captured in preference pairs).

This innovation is **incremental as a technique** (MPO combines known loss functions—DPO, BCO, LM loss—in a straightforward way) but **conceptually significant as an empirical finding** about the mechanism by which preference optimization improves reasoning. The paper provides a rare controlled comparison where data quantity is held constant and the algorithmic contribution is isolated.

## 5. Experimental Analysis

### Evaluation Methodology

- **Dataset.** All experiments use the MATH benchmark (Hendrycks et al., 2021), specifically the split from Lightman et al. (2022) consisting of 12,000 training questions and 500 test questions. MATH consists of high-school competition-level math problems spanning algebra, geometry, precalculus, and other topics. The choice is deliberate: test-time compute is expected to help most when the model already possesses the necessary knowledge and the challenge lies in drawing complex inferences—mathematical reasoning fits this profile because it requires multi-step logical deduction rather than novel factual recall. Answers are graded using the grading function released by Lightman et al. (2022), as noted in Appendix G.

- **Base model(s).** All experiments use PaLM 2-S* (Codey) (Anil et al., 2023). The authors argue this model is "representative of the capabilities of many contemporary LLMs" and sits in a useful regime: non-trivial performance on MATH (roughly 10–19% pass@1 depending on the prompt and sampling configuration) but far from saturation, leaving room for test-time compute to make a difference. For the FLOPs-matched comparison, a second model with approximately 14× more parameters is used as the pretraining-scaled baseline. The base LLM is used in a few-shot prompted configuration; specific prompts are not described in detail in the main text.

- **Metrics.** The primary metric throughout is MATH test accuracy (%)—the fraction of the 500 test questions for which the selected final answer matches the ground truth. When analyzing difficulty-dependent behavior, the paper reports accuracy within each of five difficulty quintiles separately. Difficulty is defined relative to the base LLM's pass@1 rate: for each question, 2048 solutions are sampled, and the fraction correct defines its difficulty. Oracle difficulty bins use actual correctness; predicted difficulty bins use the PRM's final-answer score averaged across the same 2048 samples instead.

- **Baselines.** The paper employs several baselines, each representing a point in the design space:
  - **Majority voting**: select the most common final answer among N sampled solutions, using no learned verifier. This represents the simplest test-time compute strategy.
  - **ORM best-of-N weighted**: score N solutions with an outcome reward model (trained separately to predict final-answer correctness) and apply best-of-N weighted selection, where solutions arriving at the same answer have their scores summed and the answer with the highest total score wins.
  - **PRM best-of-N weighted**: score N solutions with the process reward model (trained with Monte Carlo rollouts) using last-step aggregation, then apply best-of-N weighted selection.
  - **Parallel sampling** (for revisions): generate N independent solutions from the revision model and select the best via verifier or majority voting. This serves as the parallel baseline against which sequential and hybrid revision strategies are compared.

- **Generation budget / compute accounting.** One "generation" equals one complete sampled answer from the base LLM. For beam search and best-of-N, the budget equals the number of beams or samples N. For lookahead search with k lookahead steps, the cost is N × (k+1) to account for the additional rollout computation. Budgets are swept across powers of 2, typically from 2^0 to 2^9 (1 to 512 generations). The paper does not account for the cost of difficulty estimation (2048 samples per question) in any budget calculation, which the authors explicitly acknowledge in Section 3.2.

- **Cross-validation / statistical protocol.** To avoid contaminating strategy selection with test-set performance, the authors use two-fold cross-validation within each difficulty bin on the 500-question test set. The best-performing strategy is selected on one fold and evaluated on the other, with results averaged. This applies to both the compute-optimal search strategy (selecting which algorithm and beam width per bin) and the compute-optimal revision strategy (selecting the optimal sequential-to-parallel ratio per bin). The authors note that with 500 questions split into five difficulty quintiles of approximately 100 each, and each bin further split in half by cross-validation, strategy selection is based on approximately 50 questions per fold per bin.

---

### Main Quantitative Results

#### Search Against PRM Verifiers (Section 5)

The paper evaluates three search algorithms—best-of-N weighted, beam search (with beam widths M = √N and M = 4), and lookahead search (with k = 1 and k = 3 applied to both beam width settings)—using a PRM trained via Monte Carlo rollouts as the verifier.

**Aggregate comparison across all questions (Figure 3, left).** At the aggregate level across all 500 test questions:

- At low generation budgets (2–8 generations), beam search with M = 4 significantly outperforms PRM best-of-N weighted. For example, at 4 generations, beam search (M = 4) achieves roughly 27% accuracy compared to roughly 16% for best-of-N weighted—an absolute gap of approximately 11 percentage points.
- At moderate budgets (16–64 generations), beam search (M = 4) maintains a lead, though the gap narrows as best-of-N weighted catches up. At 64 generations, both methods converge to roughly 32–34%.
- At high budgets (128–256), beam search performance flattens and falls slightly *below* best-of-N weighted. Best-of-N weighted reaches approximately 38% at 512 generations, while beam search (M = 4) plateaus around 34%. This reversal at high budgets is attributed to PRM over-optimization—beam search finds solutions that score highly under the verifier but are actually incorrect.
- Lookahead search (both k = 1 and k = 3) generally underperforms all other methods at equivalent generation budgets. The 3-step lookahead variants are particularly expensive (costing 4× the base budget per beam) and never surpass simpler methods.
- Majority voting trails all verifier-based methods substantially, reaching only about 29% at 512 generations, demonstrating the value of learned verification over simple consensus.

**Difficulty-dependent analysis (Figure 3, right).** The per-difficulty breakdown (comparing beam search M = 4 vs. best-of-N weighted at four budget levels: 4, 16, 64, 256 generations) reveals the core pattern that motivates compute-optimal allocation:

- **Bin 1 (easiest questions):** Beam search accuracy *decreases* from roughly 78% to 77% as the budget increases from 4 to 256, while best-of-N weighted increases from 68% to 88%. This is the clearest evidence of verifier over-optimization: aggressive search finds solutions optimized for the PRM's preferences rather than actual correctness. Best-of-N's gentler optimization avoids this trap.
- **Bin 2:** Similar pattern—best-of-N weighted improves from roughly 14% to 60% across budgets, while beam search improves more slowly (roughly 14% to 32%). Best-of-N weighted maintains a clear advantage, though the gap narrows at higher budgets.
- **Bin 3:** A qualitative shift occurs: beam search consistently *outperforms* best-of-N weighted across all budgets. At 256 generations, beam search reaches roughly 34% versus 23% for best-of-N weighted. The PRM's guidance genuinely helps navigate toward correct solutions that random sampling would miss.
- **Bin 4:** Beam search shows its strongest relative advantage, reaching roughly 17% at 256 generations versus 10% for best-of-N weighted. However, both methods make only modest progress on these difficult questions.
- **Bin 5 (hardest questions):** Both methods hover near 1–3% accuracy regardless of budget. No amount of search helps—the base model simply does not produce correct solutions for these problems.

This difficulty-dependence pattern—beam search hurting easy questions but helping medium-hard ones—is the paper's key empirical finding for search strategies. It directly motivates the compute-optimal allocation policy.

**Compute-optimal search results (Figure 4).** By selecting the best search strategy per difficulty bin at each budget level (using two-fold cross-validation):

- At 16 generations, compute-optimal search with oracle difficulty bins achieves approximately 27% accuracy, roughly matching PRM best-of-N weighted at 64 generations—a 4× compute reduction for equivalent accuracy.
- At 256 generations, compute-optimal oracle reaches approximately 39.5%, surpassing PRM best-of-N weighted at the same budget (roughly 37%) and ORM best-of-N weighted (roughly 34% at 512 generations).
- Compute-optimal search with *predicted* difficulty bins (using the PRM's average score instead of ground-truth labels) tracks the oracle version closely, with the two curves "largely overlapping" per the authors. The predicted version reaches approximately 37% at 256 generations.
- Both compute-optimal variants consistently outperform majority voting (roughly 29% at 512 generations).

The critical finding is that the predicted difficulty version works nearly as well as the oracle version—the difficulty estimation does not require ground-truth labels and is therefore deployable in principle, though the computational cost of the estimation itself (2048 samples) is not accounted for in the budget comparison.

**PRM vs. ORM comparison (Appendix F, Figure 14).** At 2048 samples, PRM best-of-N weighted achieves approximately 40% accuracy versus roughly 35% for ORM best-of-N weighted. The gap between PRM and ORM widens as the number of samples increases, indicating superior scaling properties for the step-level verifier. This is notable because the PRM uses last-step aggregation (Section 5.1, Appendix E), which effectively reduces it to ORM-like behavior at aggregation time—yet it still substantially outperforms a separately trained ORM. The paper interprets this as evidence that step-level PRM training provides beneficial representation learning that transfers to better final-answer predictions.

---

#### Revision Model Results (Section 6)

The revision model is a fine-tuned variant of the base PaLM 2-S* that conditions on its own previous incorrect answers (up to 4 in training, more at inference) to produce improved solutions. Training data is constructed by pairing incorrect and correct solutions using edit-distance-based matching, with incorrect solutions serving as context and correct solutions as targets.

**Revision model pass@1 trajectory (Figure 6, left).** Starting from approximately 18.2% pass@1 at step 1 (the initial answer), the revision model's per-step accuracy improves to roughly 24–25% by steps 15–20, and remains in the 23–25% range out to 64 steps. This demonstrates two important properties: (1) the model has learned a generalizable revision skill, improving over the initial pass@1 by approximately 6–7 percentage points, and (2) the skill generalizes beyond the 4-step training horizon—the model was only trained with up to 4 previous answers in context, yet continues to improve and maintain performance through much longer chains.

**Sequential vs. parallel sampling (Figure 6, right).** At a fixed budget of 64 generations:
- Sequential + best-of-N weighted (verifier): approximately 41.5%
- Parallel + best-of-N weighted (verifier): approximately 39%
- Sequential + majority voting: approximately 38%
- Parallel + majority voting: approximately 35%

Sequential revisions outperform parallel sampling under both selection mechanisms. The gap is roughly 2.5 percentage points with verifier-based selection and roughly 3 points with majority-based selection. This is an important finding because it holds even when the verifier-based gap is relatively small—the benefit of sequential revisions is not solely attributable to the verifier seeing more context from the revision chain. Majority voting, which does not benefit from additional context, also shows a preference for sequential strategies.

**Sequential-to-parallel ratio sweep (Figure 7, left).** For a fixed total generation budget, the paper explores the tradeoff between sequential depth (length of each revision chain) and parallel breadth (number of independent chains), with the constraint that sequential depth × parallel breadth = total budget. At 256 total generations:
- Fully parallel (256 independent samples, no revisions): approximately 40% accuracy
- Fully sequential (1 chain of 256 revisions): approximately 42% accuracy
- Optimal intermediate ratios (2:1 to 8:1 sequential-to-parallel): approximately 43–44% accuracy

At lower budgets (8–32 generations), the curves are monotonically increasing with the sequential-to-parallel ratio—purely sequential is optimal when the total budget is small. This makes intuitive sense: with only 8 total generations, the diversity provided by parallel sampling is minimal, while sequential refinement can meaningfully improve each attempt.

**Difficulty-dependent ratio (Figure 7, right).** At a fixed budget of 128 generations, analyzed per difficulty bin:
- **Bin 1 (easiest):** Performance is essentially flat across all ratios, around 90–92%. Easy questions are insensitive to the allocation strategy—the model's initial answers are good enough that both refinement and exploration add little.
- **Bin 2:** Slight advantage for higher sequential ratios, approximately 63% at fully sequential versus 58% at fully parallel. Local refinement helps when initial answers are roughly on the right track.
- **Bin 3:** A clear optimal ratio emerges at moderate sequential-to-parallel values (around 2^1 to 2^3), reaching approximately 42% versus 35% at the extremes. These medium-difficulty questions benefit from both the diversity of parallel sampling (exploring different approaches) and the depth of sequential refinement (improving each approach).
- **Bin 4:** Similar pattern with a peak at moderate ratios achieving roughly 18% versus 14% at fully parallel. The optimal ratio shifts slightly toward more parallel sampling compared to bin 3, suggesting that harder problems require more exploration.
- **Bin 5:** All ratios produce roughly 2–3% accuracy. No allocation strategy helps on the hardest questions.

This mirrors the difficulty-dependent pattern from search: easy problems benefit from exploitation (sequential refinement), harder problems benefit from exploration (parallel diversity), and the hardest problems benefit from neither.

**Compute-optimal revision results (Figure 8).** Selecting the optimal sequential-to-parallel ratio per difficulty bin:
- At 64 generations, compute-optimal oracle achieves approximately 40%, matching parallel best-of-N weighted at 256 generations—a 4× compute reduction.
- At 256 generations, compute-optimal oracle reaches approximately 44%, compared to roughly 41% for best-of-N weighted and roughly 37% for the parallel-only baseline (parallel + best-of-N weighted at 256 generations).
- Compute-optimal with predicted difficulty bins performs slightly below oracle bins at high budgets (approximately 41% at 256 generations versus 44% for oracle) but still substantially outperforms the parallel baseline (approximately 37%).
- Notably, the parallel baseline appears to *plateau* around 36–37% at high budgets, while compute-optimal scaling with oracle bins continues to improve. This suggests that the gains from adaptive allocation are not merely shifting the curve but changing its asymptotic behavior—compute-optimal strategies extract value from additional budget that parallel sampling leaves on the table.

---

#### FLOPs-Matched Comparison: Test-Time vs. Pretraining Compute (Section 7)

This section addresses the question: given a fixed total FLOPs budget, is it better to train a larger model or to keep the smaller model (PaLM 2-S*) and spend the extra FLOPs on inference-time computation?

**FLOPs accounting model.** The paper uses standard approximations from the scaling laws literature:

Pretraining FLOPs: X = 6ND_pretrain
Inference FLOPs: Y = 2ND_inference

where N is the number of model parameters, D_pretrain is pretraining tokens, and D_inference is total inference tokens generated. Scaling model parameters by a factor of M multiplies both X and Y by M (the larger model costs more per token at both training and inference). The critical parameter is the ratio R = D_inference / D_pretrain, which determines how much inference budget the smaller model gets in exchange for its pretraining savings.

Three values of R are tested:
- R ≪ 1 (0.16): Few inference tokens relative to pretraining (typical of one-time evaluation or self-improvement pipelines).
- R ≈ 1 (0.79): Comparable inference and pretraining tokens.
- R ≫ 1 (22): Many inference tokens relative to pretraining (typical of high-throughput production deployment).

**Revisions against the larger model (Figure 9, left; Figure 1, top-right bar chart).** PaLM 2-S* with compute-optimal revisions versus the ~14× larger model using greedy decoding:

- At R ≪ 1 (0.16), test-time compute with the smaller model outperforms the larger model across all difficulty levels: +11.8% on easy questions, +27.8% on medium questions, +21.6% on hard questions (values from the bar chart in Figure 1, top-right). This is the strongest case for test-time compute as a substitute for pretraining.
- At R ≈ 1 (0.79), test-time compute remains preferable on easy (+3.5%) and medium (+16.7%) questions, but the advantage narrows substantially.
- At R ≫ 1 (22), test-time compute is only preferable on easy questions (+5.4% for medium), with hard questions showing a −37.2% relative disadvantage. At high inference volume, the per-query cost of additional test-time compute accumulates, and the larger model's superior per-token capabilities dominate.

**PRM search against the larger model (Figure 9, right; Figure 1, bottom-right bar chart).** The pattern is starker and generally less favorable to test-time compute:

- At R ≪ 1, test-time compute wins on easy questions (+19.1%), is roughly even on medium (0.0%), and loses on hard (−3.6%).
- At R ≈ 1, test-time compute wins modestly on easy (+2.2%) but loses substantially on medium (−35.3%) and hard (−35.3%).
- At R ≫ 1, test-time compute wins marginally on easy (+2.0%) but loses badly on medium (−30.8%) and hard (−52.9%).

The PRM search approach fares worse than revisions in the FLOPs-matched comparison because search requires more inference tokens per question (generating and scoring multiple complete solutions) while providing smaller per-token accuracy improvements than sequential revisions.

**Per-difficulty-bin analysis (Figure 9).** The line plots show accuracy for the compute-optimal strategy across difficulty bins, with the larger model's greedy performance indicated by stars at the three R values:
- On bin 1 (easiest, purple line): The scaling line is above all three stars for revisions, and above all three stars for PRM search as well. Easy questions are where test-time compute most reliably beats pretraining.
- On bin 5 (hardest, blue line): The scaling line is below all three stars and essentially flat near 0–5% for both methods. No amount of test-time compute helps—the base model's pass@1 is near zero, so there are no correct solutions to find or refine. This establishes a hard boundary: test-time compute cannot create capability that does not exist in the base model's output distribution.
- On intermediate bins, the relative position depends on both R and the method—revisions are more competitive across a wider range of R values than search.

**Key finding: easy problems favor test-time compute across all R; hard problems favor pretraining across all R; medium problems depend on R and method.** This nuanced conditional claim is the paper's most important empirical contribution to the pretraining-vs-inference debate.

---

### Ablation Studies and Robustness Checks

**PRM aggregation strategy (Appendix E, Figure 13):** Comparing "min" (minimum step score), "prod" (product of step scores), and "last" (final step score only) aggregation methods for the PRM, "last" achieves roughly 37% at 256 samples, "min" achieves roughly 35%, "prod" achieves roughly 27%, and a separately trained ORM achieves roughly 34%. The "last" aggregation's superiority is non-obvious: prior work (Lightman et al., 2023; Wang et al., 2023) found "min" to be best. The paper hypothesizes that the discrepancy arises because their PRM is trained with soft Monte Carlo labels rather than binary correctness labels, which changes how per-step scores distribute. An important consequence: using last-step prediction effectively makes the PRM behave like an ORM at aggregation time, yet the PRM still outperforms a separately trained ORM, suggesting that the step-level PRM training serves as beneficial representation learning for the final-step predictor.

**PRM vs. ORM (Appendix F, Figure 14):** The PRM consistently outperforms the ORM, with the gap widening at higher sample counts. At 2048 samples, PRM best-of-N weighted reaches approximately 40% versus ORM's 35%. This is a clean demonstration that process-level training improves outcome prediction even when intermediate predictions are discarded at aggregation time.

**Revision model verifier choice (Appendix J, Figure 15a):** The base-LM PRM (trained on standard base model outputs) underperforms the revision-specific ORM when scoring revision model outputs. At 64 generations, sequential + base-LM PRM achieves roughly 40% versus sequential + revision ORM at roughly 42%. This confirms that distribution shift between base model outputs and revision model outputs is a practical concern—a verifier trained on one distribution does not transfer perfectly to the other.

**Revision history in verifier context (Appendix J, Figure 15b):** Including previous revisions in the ORM's context provides a small improvement over the no-history ablation (approximately 1–2 percentage points at 64 generations), but both variants substantially outperform the parallel baseline. This confirms that the sequential sampling benefit is not solely attributable to the verifier seeing more context—the revision process itself generates better candidates, independent of how those candidates are scored.

**Oracle vs. predicted difficulty bins (Figures 4, 8; Appendix C, Figures 11–12):** Both oracle and predicted bins yield qualitatively similar trends across difficulty levels. Predicted bins show slightly lower performance at high budgets in the revision setting (roughly 41% vs. 44% at 256 generations in Figure 8) but essentially identical performance in the search setting (Figure 4). This is the critical robustness check for deployability: the compute-optimal strategy works without ground-truth labels, though the degradation in the revision setting at high budgets warrants attention.

**Majority voting for revisions (Appendix B, Figure 10):** The sequential-to-parallel ratio trends observed with verifier-based selection are replicated with majority voting: easy questions are insensitive to ratio, hard questions show an optimal intermediate ratio, and fully sequential marginally outperforms fully parallel in aggregate. This confirms that the revision model's benefit is not an artifact of the verifier selection mechanism.

**ReST^EM revision model (Appendix K, Figure 16):** An attempt to further optimize the revision model using ReST^EM (Singh et al., 2024) backfires: additional sequential revisions *substantially hurt* performance with this model. At 256 generations, fully sequential performance drops to approximately 33.5% compared to roughly 38.5% at the optimal ratio. The authors hypothesize that on-policy data collection in ReST^EM exacerbates spurious correlations in revision data, causing the model to fail to learn the revision task properly. This is a notable negative result highlighting the sensitivity of revision training to the data generation procedure—the positive results in the main paper depend on the specific offline, edit-distance-based data construction.

---

### Critical Assessment

**Claim from the executive summary: "Compute-optimal test-time scaling strategies yield more than 4× better efficiency over a standard best-of-N baseline."**

This claim is supported under specific conditions but overstated in its generality. The 4× figure comes from two specific comparisons: (1) Figure 4, where compute-optimal search at 16 generations matches best-of-N weighted at 64 generations; and (2) Figure 8, where compute-optimal revisions at 64 generations match best-of-N weighted at 256 generations. Both comparisons use *oracle* difficulty bins. When predicted difficulty bins are used (the deployable setting), the revision efficiency gain at high budgets shrinks—Figure 8 shows predicted bins achieving approximately 41% at 256 generations versus oracle bins at approximately 44%, suggesting the 4× figure is most reliable in the lower-to-moderate budget regime and may degrade at high budgets.

More importantly, the difficulty estimation cost is entirely unaccounted for. Generating 2048 samples per question to estimate difficulty consumes *more* compute than the largest test-time budgets studied (256–512 generations). If this cost were amortized, the effective efficiency gain would be substantially lower than 4×—potentially negative for one-off queries. The paper acknowledges this explicitly in Section 3.2 ("our experiments do not account for this cost largely for simplicity") but the 4× claim does not carry this caveat. For the claim to hold in practice, a cheap difficulty estimator (e.g., a lightweight classifier trained on question text alone) would need to be developed and validated. No such estimator is provided.

Additionally, the 4× improvement is benchmarked against PRM best-of-N weighted (a strong baseline that already uses a learned verifier) but not against simpler or cheaper baselines like ORM best-of-N weighted or majority voting. The improvement over majority voting is substantially larger (>4× at many budgets), but that comparison is less relevant for practitioners who would already use a verifier.

**Claim: "A smaller model augmented with compute-optimal test-time strategies can outperform a ~14× larger pretrained model on easy-to-medium difficulty problems."**

This claim is supported with sharp and well-characterized conditions. Figure 9 and the associated bar charts clearly show:
- Test-time compute wins on easy questions (bin 1) across all R values for both revisions and PRM search.
- Test-time compute wins on medium questions (bins 2–3) when R ≪ 1 or R ≈ 1 for revisions, but only at R ≪ 1 for PRM search.
- Test-time compute loses on hard questions (bins 4–5) at all R values for PRM search, and only wins at R ≪ 1 for revisions.

The paper is transparent about these boundaries, which strengthens credibility. The claim is specifically about easy-to-medium problems, and the evidence for this scope is solid.

However, several caveats weaken the practical interpretation:

First, the 14× larger model uses only greedy decoding. It receives no test-time compute budget of its own—no majority voting, no best-of-N, no search. This stacks the deck in favor of test-time compute. A fairer comparison would give the larger model at least a modest test-time budget (e.g., best-of-8) to see whether the gap narrows or reverses. The paper does not run this experiment.

Second, the larger model's pretraining is *not* compute-optimal by the Chinchilla scaling law standard. The paper scales parameters while holding data fixed (following the LLaMA paradigm), but Hoffmann et al. (2022) showed that compute-optimal pretraining scales both parameters and data equally. A Chinchilla-optimal 14× larger model would likely achieve better performance than the parameter-only-scaled model used here, making the pretraining baseline weaker than it could be. The paper acknowledges this limitation ("We choose this setting as it is representative of a canonical approach...") but the consequence is that the reported advantages of test-time compute may shrink or reverse against a properly compute-optimal larger model.

Third, the comparison relies on a specific FLOPs accounting model with standard approximations. The ratio R = D_inference / D_pretrain is the critical variable, and the three tested R values (0.16, 0.79, 22) span a wide range, but the conclusions are highly sensitive to the exact R value. For practitioners considering this tradeoff, knowing their specific R is essential—the paper's results cannot be extrapolated without it.

**Claim: "The effectiveness of any given test-time strategy is highly dependent on prompt difficulty."**

This is the most robustly supported claim in the paper. The evidence is overwhelming and consistent across methods:
- Figure 3 (right) shows beam search *hurting* easy questions while *helping* medium-hard ones, with qualitatively different trends per bin.
- Figure 7 (right) shows that the optimal sequential-to-parallel ratio varies from "fully sequential" (easy) to "intermediate" (medium) to "no allocation matters" (hard), with different shapes per bin.
- Figure 9 shows that the FLOPs-matched advantage of test-time compute varies dramatically by difficulty bin, from always favorable on easy to almost never favorable on hard.
- The same difficulty-dependent qualitative patterns appear for both oracle and predicted difficulty bins, for both search and revision strategies, and for both verifier-based and majority-based selection—a degree of replication that inspires confidence.

The difficulty bins are defined relative to the base model's capabilities, not the dataset's own difficulty labels, and the paper explicitly states that model-specific bins are more predictive than the MATH dataset's built-in difficulty levels. This is an important methodological point—difficulty is not an intrinsic property of a question but a property of the question *relative to a specific model*.

**Genuine weaknesses in the experimental design:**

- **Single benchmark, single model family.** All results are on MATH with PaLM 2-S*. The paper's claim about representativeness ("this model is representative of the capabilities of many contemporary LLMs") is untested. Without replication on at least one other model family (e.g., LLaMA, Gemma) and at least one other reasoning benchmark (e.g., GSM8K, MMLU), the findings cannot be assumed to generalize. The specific difficulty distributions, PRM quality, and revision model effectiveness may all be particular to PaLM 2-S*.

- **Small test set for strategy selection.** The 500-question test set is split into five difficulty quintiles of ~100 questions each, then further split by two-fold cross-validation. Strategy selection is based on ~50 questions per fold per bin. This is a small sample, and the selected strategies may not be robust. The paper does not report confidence intervals or standard errors on the compute-optimal scaling curves, making it impossible to assess whether the observed gains are statistically reliable or could be noise at this sample size. Given that the strategies being selected are discrete (e.g., which search algorithm, which beam width), small-sample strategy selection could exhibit high variance.

- **The compute-optimal policy is a lookup table, not a learned function.** The paper's approach requires pre-computing the best strategy for each difficulty bin at each budget level on the validation fold. This means the "compute-optimal" policy is specific to the (budget, difficulty) grid used in the experiments and does not generalize to unseen budgets or intermediate difficulties without interpolation assumptions. A parameterized policy function (e.g., a small neural network mapping difficulty estimate and budget to strategy parameters) would be more flexible and could be trained on the same data, but this is not explored.

- **No combination of search and revisions.** The paper studies PRM search and iterative revisions as independent mechanisms. A natural combination—using the revision model as the proposal distribution within beam search, or using the PRM to guide which revision paths to pursue—is never tested. Section 8 acknowledges this gap explicitly, but the consequence is that the reported gains represent a lower bound. It is possible that combined approaches would substantially outperform either method alone, potentially changing the FLOPs-matched comparison results.

- **No latency or wall-clock time accounting.** The paper measures compute in "generations," which is a reasonable proxy for total FLOPs. However, sequential revisions are inherently serial (each revision depends on the previous one), while parallel best-of-N can be executed simultaneously with sufficient hardware. A strategy allocating 128 generations as 64 sequential × 2 parallel takes approximately 64× longer wall-clock time than 128 parallel samples. For latency-sensitive applications, the sequential-heavy strategies favored by the compute-optimal policy on easy problems may be impractical regardless of their total FLOPs advantage. This tradeoff is never discussed.

- **Difficulty estimation cost makes the 4× claim potentially misleading for deployment.** The paper generates 2048 samples per question *just to estimate difficulty*. This is 2048 × the cost of a single forward pass, which dwarfs the test-time budgets studied (up to 512 generations). If difficulty estimation were amortized over many queries (e.g., estimating difficulty once and caching it), the cost per query could be low—but only if the same questions are asked repeatedly, which is not typical for open-ended assistants. The paper frames this as future work (cheap difficulty estimation) but the 4× efficiency number does not reflect the cost of getting there.

**Experiments that would have strengthened the paper:**

- **Cheap difficulty estimation validation.** Training a lightweight difficulty predictor (e.g., a logistic regression on question embeddings) and showing that it achieves comparable binning accuracy to the 2048-sample PRM-based method, with an analysis of how much accuracy degrades when using this cheap estimator.

- **Larger model with test-time compute baseline.** Giving the 14× larger model even a modest test-time budget (best-of-8, best-of-16) in the FLOPs-matched comparison, to test whether the advantage of the smaller model with heavy test-time compute persists when both models are allowed to use inference-time strategies.

- **Combined search + revisions.** Running beam search with the revision model as the proposal distribution, or using the PRM to score revision steps, to establish whether the two mechanisms are complementary or redundant.

- **Replication on a second benchmark and model family.** Running the entire compute-optimal pipeline on GSM8K or MMLU with a LLaMA-based base model to test whether the difficulty-dependent patterns and 4× efficiency claims generalize.

- **Confidence intervals.** Reporting standard errors or bootstrapped confidence intervals on the compute-optimal scaling curves, particularly given the small per-bin sample sizes (~50 questions per fold) used for strategy selection.

## 6. Limitations and Trade-offs

### 6.1 Difficulty Estimation Requires More Compute Than the Test-Time Budget It Optimizes

**The assumption or constraint.** The entire compute-optimal allocation framework depends on estimating each prompt's difficulty before selecting a strategy. The paper's method for doing so—generating 2048 complete solutions per question and averaging the PRM's final-answer scores across them—consumes computational resources that vastly exceed the test-time budgets being optimized. The authors acknowledge this explicitly in Section 3.2:

> "estimating difficulty in this way still incurs additional computation cost during inference... our experiments do not account for this cost largely for simplicity"

The paper further describes this as "an exploration-exploitation tradeoff—compute spent assessing difficulty versus compute spent solving the problem—flagging it as a key avenue for future work."

**The consequence.** The headline 4× efficiency gains (e.g., Figure 4: compute-optimal search at 16 generations matching best-of-N at 64 generations; Figure 8: compute-optimal revisions at 64 generations matching best-of-N at 256) are computed *after* difficulty is known, without amortizing the cost of learning it. In a realistic single-query deployment, the total cost would be 2048 + N generations, where N is the selected strategy budget. Since N is at most 256–512 in the experiments and 2048 dominates, the effective cost with difficulty estimation is 2048/256 ≈ 8× larger than the largest test-time budget studied. The 4× efficiency claim is therefore an *upper bound on achievable efficiency* in a setting where an oracle or a pre-existing cheap difficulty estimator is available—neither of which is demonstrated.

The predicted difficulty bins (using PRM scores instead of ground-truth labels) remove the need for correct answers but do not reduce the computational cost—they still require 2048 forward passes per question. The paper does not present any method for estimating difficulty with substantially fewer samples. The performance of predicted difficulty bins relative to oracle bins (Figures 4 and 8) degrades at higher budgets in the revision setting (roughly 41% vs. 44% at 256 generations in Figure 8), suggesting that even with the full 2048-sample estimation, the predicted bins do not perfectly replicate oracle performance. A cheaper estimator with fewer samples would likely degrade further.

**What evidence exists in the paper.** The difficulty estimation procedure is described in Section 3.2, where the 2048-sample approach is specified. The oracle vs. predicted comparison appears in Figures 4 and 8, and in Appendix C. The paper explicitly notes the unaccounted cost in Section 3.2 but provides no experiments measuring how performance degrades with fewer difficulty-estimation samples, nor any attempt to train a cheap difficulty predictor.

**Mitigation status.** The paper acknowledges this as a limitation and suggests future work on "pretraining or finetuning models to directly predict difficulty of a question" (Section 8). No such model is developed or evaluated. The limitation is therefore entirely unmitigated in the current work.

---

### 6.2 Hard Problems Show Near-Zero Benefit from Any Test-Time Compute Strategy

**The assumption or constraint.** The paper's framework assumes that the base model's pass@1 on a problem is non-trivially above zero—that is, the model occasionally produces correct solutions, and test-time compute can amplify this signal. This assumption fails systematically for the hardest problems.

**The consequence.** Across all methods—search, revisions, and their compute-optimal combinations—the hardest questions (difficulty bin 5) show essentially no improvement regardless of compute budget. In Figure 3 (right), bin 5 accuracy hovers at 1–3% for both beam search and best-of-N weighted at all budgets from 4 to 256 generations. In Figure 7 (right), bin 5 shows roughly 2–3% accuracy regardless of the sequential-to-parallel ratio at 128 generations. In the FLOPs-matched comparison (Figure 9), the bin 5 scaling lines are essentially flat near 0–5%, and all three stars (representing the ~14× larger model's greedy performance) sit above the scaling curves.

This establishes a hard boundary on the utility of test-time compute: it can amplify existing capability but cannot create it. If the base model's pass@1 is near zero on a problem class, no amount of search or revision will help because there are no correct solutions in the proposal distribution to find or refine. For genuinely novel or out-of-distribution reasoning tasks that exceed the base model's training distribution, pretraining remains the only viable path. This boundary is important for practitioners deciding whether to invest in test-time compute infrastructure: if their query distribution contains a substantial fraction of problems where the base model's pass@1 ≈ 0, the returns on inference-time compute investment will be near zero for that fraction.

**What evidence exists in the paper.** The evidence is pervasive and consistent across Figures 3 (right), 7 (right), and 9. The paper is transparent about this in the Section 7 discussion and the takeaway box:

> "test-time compute provides minimal gains on problems that are fundamentally outside the base model's capability range"

**Mitigation status.** There is no mitigation. The paper correctly identifies this as a fundamental characteristic, not a bug that can be fixed. The only path forward for hard problems identified in the paper is scaling pretraining (Section 7). No hybrid approach (e.g., using the PRM to decide when to route a query to a larger model) is explored.

---

### 6.3 The ~14× Larger Model Baseline Is Not Compute-Optimal and Uses No Test-Time Compute of Its Own

**The assumption or constraint.** The FLOPs-matched comparison in Section 7 compares PaLM 2-S* with compute-optimal test-time strategies against a model with approximately 14× more parameters that uses only greedy decoding (no majority voting, no best-of-N, no search). The larger model's pretraining scales only parameters while holding training data fixed, following the LLaMA paradigm rather than Chinchilla-optimal scaling (which would scale both parameters and data equally).

The paper acknowledges this in Section 7:

> "We choose this setting as it is representative of a canonical approach to scaling pretraining compute and leave the analysis of compute-optimal scaling of pretraining compute where the data and parameters are both scaled equally to future work."

**The consequence.** Both design choices weaken the pretraining baseline, making the case for test-time compute appear stronger than it would be against a properly optimized baseline:

First, using only greedy decoding means the larger model has no inference-time optimization at all. A practitioner deciding between scaling pretraining and scaling test-time compute would likely give the larger model *some* test-time budget—even a modest best-of-8 would improve the larger model's scores. The current comparison answers the question "test-time compute on a small model vs. *greedy decoding from a large model*," which is less informative than "test-time compute on a small model vs. *comparable test-time compute on a large model*." The paper never runs the latter experiment.

Second, a Chinchilla-optimal 14× larger model (scaling both parameters and data equally) would likely outperform the parameter-only-scaled model used here, since it would be trained more efficiently with more data per parameter. The paper acknowledges this departure from compute-optimal pretraining but does not quantify how much it affects the comparison.

The practical consequence is that the reported advantages—e.g., +27.8% on easy questions at R ≪ 1 for revisions (Figure 1 bar chart)—may shrink or reverse against a stronger pretraining baseline. The paper's claim that "test-time compute can substitute for pretraining" should be understood as applying specifically to the LLaMA-style scaling paradigm, not necessarily to compute-optimal pretraining.

**What evidence exists in the paper.** The FLOPs-matched results are in Figure 9 and the associated bar charts in Figure 1. The baseline description (14× larger model, greedy decoding) is in Section 7. The parameter-only scaling choice is explicitly noted in Section 7. However, no ablation tests the larger model with test-time compute, and no comparison to a Chinchilla-optimal larger model is provided.

**Mitigation status.** The paper flags the parameter-only scaling choice as a limitation and defers the compute-optimal pretraining comparison to future work. The lack of test-time compute for the larger model is not acknowledged as a limitation—it is simply the chosen experimental design. No experiments address either concern.

---

### 6.4 PRM Search and Iterative Revisions Are Studied Independently, Not Combined

**The assumption or constraint.** The paper studies two complementary mechanisms—PRM-guided search (Section 5) and iterative revisions (Section 6)—as independent axes of test-time compute scaling. They are never combined into a single system.

Section 8 acknowledges this:

> "we did not experiment with PRM tree-search techniques in combination with revisions"

**The consequence.** The paper demonstrates that search and revisions have complementary strength profiles across difficulty levels: revisions excel on easy problems (local refinement of roughly-correct initial answers), while search excels on medium problems (global exploration of different solution strategies). The natural hypothesis—supported by the paper's own framing in Section 2, where revisions modify the proposal distribution and search modifies the verifier—is that combining both would yield gains beyond either alone.

Because this combination is never tested, the reported results represent a *lower bound* on what is achievable with the paper's components. Specifically:
- Beam search with the revision model as the proposal distribution (generating each beam step conditioned on previous revision context) could improve candidate quality over standard beam search.
- Using the PRM to guide which revision paths to pursue (e.g., scoring partial revisions and pruning unpromising chains) could make sequential revision more efficient by avoiding entire chains of bad revisions.
- The compute-optimal policy could select not just between search and revisions but also combined strategies, potentially finding regimes where the combination beats either individual approach.

Without these experiments, it is unknown whether the gains from search and revisions are additive, sub-additive (redundant), or super-additive (synergistic). The paper's compute-optimal policy selects the best *single* strategy per difficulty bin; it does not explore whether a combination of strategies would perform better.

**What evidence exists in the paper.** None. The search experiments (Section 5) all use the base PaLM 2-S* model as the proposal distribution. The revision experiments (Section 6) all use the fine-tuned revision model without PRM-guided search. The two lines of work are presented in separate sections with separate evaluation protocols. Section 8 explicitly notes the gap.

**Mitigation status.** The paper identifies this as future work in Section 8 but offers no experimental results. The limitation is entirely unmitigated in the current work.

---

### 6.5 Sequential Revision Strategies Introduce Latency That Parallel Strategies Avoid

**The assumption or constraint.** The paper measures test-time compute exclusively in "generations"—the number of complete solutions sampled—which is a reasonable proxy for total FLOPs but ignores wall-clock latency. This assumption is implicit throughout Sections 5–7; the paper never discusses the time dimension of compute allocation.

**The consequence.** Sequential revisions are inherently serial: each revision depends on the output of the previous one, so a chain of N sequential revisions requires N serial forward passes through the model. Parallel best-of-N generates all N samples independently and can complete in the time of a single forward pass with sufficient hardware (batching or model parallelism). For the compute-optimal policies that allocate heavily toward sequential revisions:

- On easy problems (bins 1–2), the optimal strategy is pure or near-pure sequential (Figure 7, right). At 256 total generations, a fully sequential strategy requires 256 serial forward passes. A parallel baseline with the same budget takes the wall-clock time of 1 forward pass (assuming batch size ≥ 256). The sequential strategy therefore incurs a *256× wall-clock time penalty* for latency-sensitive applications.
- The compute-optimal hybrid strategies (e.g., 64 sequential × 4 parallel) reduce total FLOPs but still take 64× longer than purely parallel execution.

For interactive applications (chatbots, assistants, real-time QA), this latency penalty may render compute-optimal strategies impractical regardless of their FLOPs advantages. A practitioner choosing between strategies must trade off total compute cost against user-perceived response time—a dimension the paper does not address.

**What evidence exists in the paper.** None. The paper does not report wall-clock time, latency, throughput, or any time-based metric. The generation budget is used throughout as the sole unit of cost. The serial nature of sequential revisions is implicit in the method description (Section 6.1: the revision model "conditions on that answer to produce a revision, then conditions on the revision to produce another revision") but the latency implications are never quantified or discussed.

**Mitigation status.** Not addressed. The paper treats total FLOPs as the only resource constraint, which is appropriate for the scaling analysis at the level of the research contribution but incomplete for practitioners concerned with deployment latency. No experiments explore latency-aware allocation (e.g., constraining the total budget while also limiting the sequential depth, or trading off sequential depth against parallel breadth under a latency constraint).

---

### 6.6 All Results Are on a Single Benchmark (MATH) with a Single Model Family (PaLM 2-S*)

**The assumption or constraint.** The entire experimental framework—all search results, all revision results, all difficulty-dependent analyses, the FLOPs-matched comparison, and every ablation—is conducted on the MATH benchmark (500 test questions) using PaLM 2-S* (Codey) as the base model. The authors state in Section 4 that they "believe this model is representative of the capabilities of many contemporary LLMs," but no evidence is presented to support this representation claim.

**The consequence.** Several aspects of the findings could be specific to this model-benchmark combination:

- **PRM quality and over-optimization behavior.** The PRM's performance and the point at which over-optimization begins (Figure 3, right) depend on PaLM 2-S*'s specific error patterns and output distribution. A model with different calibration properties or different types of reasoning errors might exhibit different difficulty-dependent scaling curves and different optimal strategies. The paper's central finding—that beam search hurts easy problems but helps medium ones—could be partially specific to how PaLM 2-S*'s errors interact with the PRM's scoring.

- **Revision model effectiveness.** The ability of the revision model to learn from incorrect in-context examples depends on the base model's in-context learning capabilities, which vary substantially across model families. The edit-distance-based training data construction and the ~38% correct-to-incorrect reversion rate may be particular to PaLM 2-S*.

- **Task domain.** The MATH benchmark consists exclusively of competition-level math problems requiring symbolic reasoning. The difficulty-dependent patterns—and the relative effectiveness of search vs. revisions—may not generalize to other reasoning domains. Code generation, for instance, has different error patterns (syntax errors are immediately detectable; logic errors may persist through revision). Scientific QA requires factual recall that revisions cannot fix. Open-ended tasks without clean verifiability cannot use PRM-based search or the Monte Carlo rollout training procedure at all.

- **Sample size for strategy selection.** The 500-question test set is split into five difficulty quintiles (~100 questions each), then further split by two-fold cross-validation for strategy selection (~50 questions per fold per bin). This is a small sample for selecting discrete strategies from a combinatorial space (multiple search algorithms × multiple beam widths × multiple budgets). The paper does not report confidence intervals, so it is impossible to assess whether the selected compute-optimal policies are stable or whether the observed performance differences between strategies are statistically significant. A different model on a different benchmark might produce different strategy rankings from this small sample, leading to different (and perhaps less favorable) compute-optimal scaling curves.

**What evidence exists in the paper.** All results in Sections 5–7 are from MATH with PaLM 2-S*. The paper acknowledges the single-model limitation in Section 4 by claiming representativeness but provides no replication. The MATH split (12,000 training, 500 test, from Lightman et al., 2022) is described in Section 4. No experiments on other benchmarks (GSM8K, MMLU) or other model families (LLaMA, Gemma, Qwen) are presented.

**Mitigation status.** The paper does not address this limitation experimentally. The claim of representativeness is asserted without evidence. Future work replicating the compute-optimal framework on other models and benchmarks is neither performed nor suggested in the paper's discussion of future directions (Section 8). Given that the paper's core contributions are empirical findings about the scaling behavior of test-time compute, replication across model families and task domains is essential for establishing the generality of those findings—and its absence is a significant gap.

## 7. Implications and Future Directions
- Field impact
  - Demonstrates that unified, native multimodal pre-training can replace multi-stage post-hoc adaptation, simplifying pipelines while preserving or improving language skill (Figure 3; Table 11). This can shift the community toward single-stage, mixed-data pre-training for MLLMs.
  - V2PE offers a practical recipe to extend multimodal context without enlarging the model, likely influencing future positional encoding designs for MLLMs (Section 2.1; Table 12).

- Follow-up research
  - Data-targeted improvements: Add grounding-specific corpora and richer hallucination-robust data; probe how the `L_text-only` objective interacts with tasks that benefit from visual-token prediction.
  - Position strategies: Learn δ or schedule it adaptively per input instead of sampling from a fixed set; explore cross-modal relative position schemes.
  - Process supervision: Expand VisualPRM beyond math/science into multi-image reasoning and long video; investigate training-time integration (e.g., reinforcement learning with process rewards).
  - Efficiency: Combine BoN with smarter sampling (e.g., conditional early stopping, diversity-promoting decoding) to reduce test-time cost.

- Applications
  - Strong results suggest readiness for:
    - Document and chart analysis (Table 3: DocVQA 95.4, ChartQA 89.7).
    - Enterprise OCR and form understanding (OCRBench 906).
    - GUI agents and UI automation (Table 9: up to 90.9 on ScreenSpot-V2).
    - Spatial reasoning for robotics/autonomy (Table 10: top object counting/relative distance/appearance order).
    - Long video understanding for surveillance, sports, or education (Table 8: strong scaling on MLVU/LongVideoBench).
  - Public release of code, data, and weights (front page; Figure 1 footers) lowers barriers for practitioners and researchers to build and evaluate next-generation open-source MLLMs.

> Headline result: “InternVL3‑78B reaches MMMU 72.2, OCRBench 906, AI2D 89.7, ChartQA 89.7, DocVQA 95.4” (Figure 1; Tables 2–3), with compelling language-only competence (Table 11) and broad real-world robustness (Tables 4–5), achieved via a unified training paradigm plus V2PE, MPO, and test-time process supervision.
