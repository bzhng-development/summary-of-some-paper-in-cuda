# Apollo: An Exploration of Video Understanding in Large Multimodal Models

**ArXiv:** [2412.10360](https://arxiv.org/abs/2412.10360)

## 🎯 Pitch

Apollo presents the first systematic exploration of what actually enables effective video understanding in Large Multimodal Models, experimentally mapping the effect of key design and training choices across model and dataset scales. The paper introduces the concept of Scaling Consistency, showing that design decisions validated on mid-sized models and datasets reliably translate to larger, expensive settings—dramatically accelerating innovation and lowering resource barriers. Building on these insights, the authors provide ApolloBench for faster, more meaningful evaluation and launch Apollo, a new model family that sets a new state of the art for open video-LMMs under 30B parameters, empowering the whole community to drive progress in video AI.

---

## 1. Executive Summary

This paper studies the design space of video Large Multimodal Models (video-LMMs), analyzing how architectural choices, training protocols, and data composition drive video understanding performance. Using the Qwen2 model family and the MATH-equivalent video QA benchmarks (Video-MME, MLVU, LongVideoBench), the authors introduce **Scaling Consistency** (where design decisions made on moderately sized ~3B models and ~500K-sample datasets transfer reliably to larger models, achieving R² > 0.9 between 4B and 7B variants) and use this principle to systematically explore video-specific design choices—including fps vs. uniform sampling, single vs. dual vision encoders, token resampling methods, and multi-stage training schedules. Guided by these findings, the Apollo model family achieves state-of-the-art results, with Apollo-3B outperforming most existing 7B models (55.1 on LongVideoBench, 58.4 on Video-MME without subtitles) and Apollo-7B reaching 70.9 on MLVU and 63.3 on Video-MME, rivaling models with over 30B parameters. The paper also introduces **ApolloBench**, a curated subset of existing benchmarks that reduces evaluation time by 41× while maintaining high correlation with full benchmark suites, establishing that video perception assessment can be made dramatically more efficient only when questions requiring temporal understanding are explicitly filtered from those answerable via text or single-frame comprehension alone.

## 2. Context and Motivation

### The Core Problem: Video-LMM Design Is Arbitrary and Poorly Understood

The fundamental problem this paper tackles is that **the video Large Multimodal Model (video-LMM) field lacks systematic understanding of what design decisions actually matter**. Despite explosive growth in video-LMM development—driven by the success of image-based LMMs and the obvious utility of video understanding—the field operates largely on hunches, borrowed practices, and anecdotal evidence rather than principled design. The authors frame this explicitly in the introduction:

> "Many fundamental questions about video-LMM design remain unanswered: How should videos be sampled? Which vision encoders yield optimal representations? What are the best practices for resampling video tokens?"

This gap matters because video-LMMs are far more complex to design than their image counterparts. An image-LMM processes a single frame through a vision encoder, projects the tokens into the LLM's space, and generates text. A video-LMM must additionally decide: how many frames to sample (and from where in the video), at what temporal resolution, using what sampling strategy (uniform vs. fps-based), with which encoder(s) (image-only, video-only, or both), how to compress the resulting flood of visual tokens into something the LLM can process within its context window, and how to stitch those compressed representations together with text tokens. Each of these choices interacts with the others in non-obvious ways, and the computational cost of training video-LMMs means that **exhaustive experimentation is prohibitively expensive**. The consequence is that design decisions are frequently "made without proper justification or analysis," as the paper states—researchers default to whatever prior work used, regardless of whether those choices were themselves well-motivated.

### Why This Problem Matters: Computational Barriers and Democratization

The significance of this gap extends beyond academic curiosity. The authors highlight the **computational inequity** in video-LMM research:

> "The high computational cost of training and evaluating such models, coupled with limited open research, hinders the development of video-LMMs."

Consider the concrete numbers the paper provides: evaluating a single 3B-parameter model on the full suite of existing video QA benchmarks requires **184 A100 GPU hours**. Training a single video-LMM variant requires orders of magnitude more. This creates a regime where only well-resourced industrial labs can afford to iterate on video-LMM design, while academic groups and smaller research organizations are effectively locked out. The problem is self-reinforcing: because comprehensive design-space exploration is so expensive, it doesn't get done; because it doesn't get done, the field lacks the shared knowledge that would let everyone make more efficient choices, which keeps the cost barrier high.

This paper's intervention is therefore not just about improving benchmark scores—it is about **democratizing video-LMM research**. If design decisions made on small models and small datasets reliably transfer to large ones (the "Scaling Consistency" hypothesis), then researchers with modest compute budgets can meaningfully contribute to video-LMM design innovation. The implications are structural: faster iteration cycles, more diverse ideas entering the field, and a shift from "who has the most GPUs" to "who has the best insights."

### Where Existing Approaches Fall Short

The paper identifies several specific ways in which prior work leaves the design space underexplored.

**Early approaches were direct, unexamined extensions of image-LMMs.** The first wave of video-LMMs (Video-ChatGPT, Video-LLaMA, VideoChat) essentially took image-LMM architectures and fed them multiple frames, either by processing frames independently and aggregating, or by swapping in a video encoder where an image encoder previously sat. The paper cites these approaches (Xu et al., 2024b; Kim et al., 2024; Zhang et al., 2023; Maaz et al., 2023) but notes they were developed without systematic comparison of alternatives. Key questions—like whether video encoders actually outperform image encoders at video understanding, or whether uniform frame sampling creates a training signal problem—were simply not asked.

**Recent methods introduced design complexity without justification.** The paper points to a range of more sophisticated recent approaches: longer context windows (Zhang et al., 2024e), multi-modality mixing (Li et al., 2024a,c), agent-based workflows (Wang et al., 2024c), and self-training (Zohar et al., 2024). Each introduces new design dimensions. But, the authors argue:

> "the impact of these design decisions on video-LMM performance is poorly understood"

This is a specific, actionable critique. It is not that these methods are wrong—it is that we do not know *which* of their design choices drive the improvements they report. Did agent-based workflows win because of the agent structure, or because they enabled more frames to be processed? Did longer context windows help because of increased frame count, or because of some other architectural property? Without controlled ablations, the field accumulates methods but not understanding.

**Video encoders vs. image encoders: an unresolved tension.** The paper highlights an important unresolved question that reflects a deeper uncertainty in the field. Early video-LMMs predominantly used **dedicated video encoders** (Video-LLaVA, VideoChat), which process multiple frames jointly and can in principle capture temporal dependencies. However, recent work has shifted toward **image encoders** (LLaVA-OneVision, Oryx, Kangaroo) that process each frame independently. The paper notes:

> "This shift arises because image encoders, although lacking temporal integration, still produce higher-quality representations that the LLM can readily leverage."

This is a fascinating and non-obvious tradeoff. Video encoders are purpose-built for video but are typically trained on smaller, lower-quality video datasets. Image encoders are trained on massive, high-quality image datasets (billions of image-text pairs) but treat a video as just a bag of frames, leaving temporal reasoning entirely to the LLM. Which approach wins, and under what conditions, was unknown before this paper's systematic encoder comparison (Section 4.2). Furthermore, an intermediate approach—**using both** an image encoder for spatial quality and a video encoder for temporal information—was underexplored.

**Benchmark evaluation is expensive and potentially misleading.** The paper's analysis of existing benchmarks (Section 2) reveals a problem that is both practical and scientific. Practically, evaluating on the full suite of video QA benchmarks is computationally wasteful—the paper finds high redundancy, with R² > 0.92 between duration groups within LongVideoBench, R² > 0.83 within Video-MME, and R² > 0.8 between question types in TempCompass. Scientifically, a significant fraction of existing benchmarks measure something other than video understanding. The paper's three-condition evaluation (video input, single center frame, text-only) reveals that:

> "a significant portion of existing benchmarks are answered solely through text comprehension alone... or only using the center frame"

This is a specific, damning finding. NExT-QA and Perception-Test, two widely used benchmarks, can largely be solved from a **single frame**—meaning temporal reasoning, the entire point of video understanding, is barely tested. The problem worsens with video length: as videos get longer, reliance on video perception *decreases* (Figure 2, left), likely because longer videos contain more textual/contextual cues that allow the LLM to guess answers without watching. This echoes the famous "VQA language bias" findings from the image domain (Goyal et al., 2017) but is even more acute for video, where the computational cost of proper evaluation is higher.

### How This Paper Positions Itself Relative to Existing Work

The paper situates itself as a **systematic design-space exploration** in the tradition of image-LMM studies like Prismatic VLMs (Karamcheti et al., 2024), Idefics3 (Laurençon et al., 2024b), Cambrian-1 (Tong et al., 2024), and Eagle (Shi et al., 2024). These works systematically ablated image-LMM components—encoder choice, connector architecture, training stages, data mixtures—and produced actionable findings (e.g., SigLIP outperforms larger encoders, Perceiver Resamplers work for token compression, multi-stage training helps).

The paper explicitly acknowledges this lineage:

> "While these works provide a strong foundation for image-based LMMs, the design space for video-LMMs remains underexplored. Unlike images, videos require specialized strategies for frame sampling, token resampling, encoder selection, and efficient training and evaluation."

The positioning is: **this paper is for video-LMMs what those works were for image-LMMs**, but with the additional challenge that the video design space is combinatorially larger and computationally more expensive to explore. This is precisely why **Scaling Consistency** (Section 3) is not just one contribution among many but the **enabling insight** for the entire study. Without it, the exhaustive comparison of sampling strategies, encoders, resamplers, integration methods, training schedules, and data mixtures—across multiple model sizes—would require a computational budget available to almost no one. The paper explicitly frames Scaling Consistency as a methodological contribution that makes the rest of the work possible:

> "Our primary goal is to show that design decisions transfer reliably, reducing computational burden and accelerating research."

This connects to but differs from traditional scaling laws. Scaling laws (Hoffmann et al., 2022) establish mathematical relationships between model size, data size, and performance, allowing *extrapolation*: train a few small models, fit a curve, predict the performance of a larger one. Scaling Consistency is a weaker but more practical claim: design *decisions* (not absolute performance numbers) transfer, meaning you can use small models to answer qualitative questions ("is fps sampling better than uniform sampling?") without having to verify the answer at scale. The paper notes that traditional scaling laws apply to models trained from scratch and require training multiple sizes **for each design variation**, which is impractical for LMMs that integrate multiple pretrained components. Scaling Consistency relaxes this requirement.

The paper also positions ApolloBench as a correction to the evaluation landscape. Rather than introducing yet another benchmark, it curates a subset of existing ones by filtering out questions that do not require video perception and selecting for high discriminability across models. This is a deliberate choice to **make evaluation both cheaper and more diagnostically useful**, addressing the twin problems of computational cost and benchmark quality that the paper's own analysis revealed.

Finally, the paper positions the Apollo family of models not as the primary contribution but as a **validation of the design insights**. The state-of-the-art results (Apollo-3B outperforming 7B models, Apollo-7B rivaling 30B models) are presented as evidence that the systematic exploration yielded genuine improvements, not as the end goal. This is a common pattern in design-space exploration papers: the methodology and findings are the contribution; the resulting models demonstrate that the findings were correct.

## 3. Technical Approach

### 3.1 Reader Orientation

This paper builds a **framework for designing video Large Multimodal Models** — not a single model architecture, but a systematic methodology for making informed design choices about how videos are sampled, encoded, compressed, and integrated with language models, along with how training should be scheduled and what data mixtures to use. The system solves the problem of **combinatorial design-space explosion** in video-LMMs: because training even one variant is computationally expensive, and the number of possible combinations (sampling strategy × encoder choice × resampler type × training stages × data mixture) is enormous, researchers historically make arbitrary choices without evidence. The solution is a two-part strategy: first, establish **Scaling Consistency** — the empirical finding that design decisions made on moderately sized models (~3B parameters) and datasets (~500K samples) transfer reliably to larger models (R² > 0.9 between 4B and 7B variants) — which makes systematic exploration affordable; second, use this principle to exhaustively ablate every major design dimension and produce a set of actionable findings that future video-LMM builders can adopt without repeating the full exploration.

### 3.2 Big-Picture Architecture (Diagram in Words)

The video-LMM pipeline has **five major components**, arranged as a sequential processing chain from raw video to text output:

1. **Video Sampler** — decides which frames to extract from the input video (how many, at what temporal resolution, using what sampling strategy). Produces a sequence of frame tensors.

2. **Vision Encoder(s)** — one or two pretrained neural networks (image or video encoders) that convert each frame or clip of frames into a grid of visual feature vectors. Different encoders capture different properties (spatial detail from image encoders, temporal dynamics from video encoders).

3. **Feature Interpolation and Concatenation** — when dual encoders are used, their output feature maps are spatially interpolated to a common resolution and concatenated along the channel dimension, producing a unified multi-encoder representation per frame.

4. **Token Resampler (Connector)** — projects the encoder features up to the LLM's hidden dimension (typically a 2–4× increase) and then compresses the spatial grid of tokens into a smaller, fixed number of tokens per frame using either learned attention pooling (Perceiver Resampler) or spatial pooling. This is the **information bottleneck** that determines how many visual tokens the LLM must process.

5. **Large Language Model (LLM)** — receives the sequence of compressed visual tokens interleaved with text tokens (questions, timestamps, separation tokens) and autoregressively generates the answer. The LLM is typically frozen during early training stages and unfrozen only during final supervised fine-tuning.

Information flows as follows: input video → sampler extracts frames at a fixed fps → frames are grouped into clips of N consecutive frames (N depends on the video encoder; 4 for InternVideo2) → each clip is encoded independently by the vision encoder(s) → encoder features are interpolated and channel-concatenated → the connector projects and resamples to a fixed number of tokens per clip → text timestamps are prepended to each clip's tokens → the full sequence of [timestamp tokens + clip tokens] is fed into the LLM → the LLM generates the answer autoregressively. Training proceeds through multiple stages where different components are frozen or unfrozen and different data mixtures are used.

### 3.3 Roadmap for the Deep Dive

- **First, Scaling Consistency** — the methodological foundation that makes the entire exploration possible. We will examine how the authors train 84 model variants across four LLM sizes to establish that design decisions transfer, the correlation analysis that supports this, and the critical model size (~3B) and dataset size (~500K) thresholds they identify.

- **Second, video sampling strategies** — how fps vs. uniform sampling works, why uniform sampling creates a "variable playback speed" problem during training, and the tradeoff between frames per second (fps) and tokens per second (tps) that governs long-video performance.

- **Third, vision encoder selection** — the systematic comparison of seven single encoders (image and video, language-supervised and self-supervised) and their pairwise combinations, including the finding that SigLIP-SO400M is the best single encoder despite being image-only.

- **Fourth, token resampling** — how the connector module compresses visual tokens, the comparison of average pooling, 2D convolution, and Perceiver Resampler approaches, and why channel-wise concatenation before resampling matters for dual-encoder setups.

- **Fifth, token integration** — how visual tokens are interleaved with text tokens, the experiments with separation tokens, timestamps, and their combinations, and why adding any structural text between clips helps.

- **Sixth, training schedules** — the comparison of 1-stage, 2-stage, and 3-stage training protocols, which components are frozen/unfrozen at each stage, and why progressive unfreezing yields the best dynamics.

- **Seventh, training video encoders** — when and on what data video encoders should be fine-tuned, the finding that training them simultaneously with the LLM on mixed image+video data hurts performance.

- **Eighth, data composition** — how the proportions of text, image, multi-image, and video data in the training mixture affect downstream video understanding, and the surprising importance of ~10–14% text data for preventing catastrophic forgetting.

### 3.4 Detailed, Sentence-Based Technical Breakdown

This is primarily an **empirical design-space exploration paper** whose core idea is that systematic ablation of video-LMM design choices — enabled by the Scaling Consistency principle that lets small-model experiments transfer to large ones — can identify which decisions actually matter and produce actionable guidelines for the field.

---

#### Scaling Consistency: Enabling Efficient Design Exploration

**What is being established.** Before any video-specific design choices can be studied, the authors must first establish that studying them on small models is meaningful. They introduce **Scaling Consistency**, the empirical finding that the *relative ordering* of design decisions (which choice is better than which other choice) is preserved across model sizes above a critical threshold (~2–4B parameters), even across different model families.

**Experimental design.** The authors selected 21 distinct model configurations that vary across the design dimensions they care about: architecture choices (which vision encoders to use, how many tokens per frame), video sampling parameters (fps, tokens per second, tokens per frame), training strategies (which components are frozen vs. trainable), and data mixtures (different ratios of text/image/video data). Each of these 21 configurations was trained using four different LLM backbones: Qwen2-0.5B-Instruct, Qwen2-1.5B-Instruct, Qwen1.5-4B-Chat, and Qwen2-7B-Instruct. This yields a total of **84 trained models** (21 configurations × 4 LLM sizes). All models were trained on the same data mixture and evaluated on ApolloBench.

The design variations span a meaningful range. Looking at Tables 14 and 15 in the appendix, we can see the diversity: some configurations use LanguageBind-Video + SigLIP as dual encoders, others use V-JEPA + SigLIP; clip duration varies from 5 to 10 frames; tokens per clip varies from 32 to 64; fps ranges from 1.6 to 3.2; tokens per frame ranges from 2 to 8; and data mixture varies across three different recipes (labeled A, B, C). This ensures the correlation analysis covers the kinds of design decisions researchers actually make.

**Correlation analysis.** For each pair of LLM sizes, the authors compute the Pearson correlation coefficient (R²) between the ApolloBench performance scores of the 21 model variants trained with each LLM. Conceptually, this asks: if a particular configuration performs well with a 4B LLM, does it also perform well with a 7B LLM? The R² value quantifies how strongly the rankings agree.

The key result appears in Figure 3 (left) and the associated correlation plots in Appendix Figure 15:

> "the R² between the 4B and 7B models is 0.938, indicating a strong predictive relationship"

The correlation between model sizes follows an approximately log-linear relationship with the smaller model's size. The authors fit:

$$y = 0.12 \cdot \log(x) + 0.78$$

where `$y$` is the R² value and `$x$` is the smaller model's size in billions of parameters.

**What it computes:** the expected correlation between design rankings at size `$x$` and design rankings at 7B. For `$x=0.5$`B, this gives R² ≈ 0.694; for `$x=1.5$`B, R² ≈ 0.831; for `$x=4$`B, R² ≈ 0.938.

**Why this form:** the log-linear relationship is an empirical fit to the observed R² values, not derived from theory. Its significance is that it predicts R² > 0.9 at around 3B parameters, and by extrapolation, a 32B model would have R² ≈ 0.86 with a 72B model. The key practical implication: **a 3–4B model is "large enough" that its design decisions will transfer to production-scale models**. The authors explicitly note:

> "This suggests that at around 3 billion parameters, we can expect an R² greater than 0.9 when compared with the 7B model."

**Critical threshold behavior.** The 0.5B model does *not* exhibit Scaling Consistency. As Figure 3 (left) shows, the R² between 0.5B and 7B is only 0.694, and there is no log-linear improvement when going from 0.5B to other small sizes. The authors observe:

> "This behavior is not observed with smaller models, e.g., 0.5B, where R² immediately drops below 0.8, and no log-linear behavior can be observed. This reinforces the existence of a critical model size (~2 −4B) where design decisions transfer reliably."

This is an important boundary condition: the method does *not* work with arbitrarily small models. Below ~2B parameters, the model is simply not capable enough for its design preferences to reflect what matters for capable models. A 0.5B model might prefer a different encoder or sampling strategy simply because it cannot effectively use the information provided in the "better" configuration.

**Cross-family generalization.** The authors used a mix of Qwen1.5 and Qwen2 models in their study. The 4B model is Qwen1.5-4B-Chat, while the 7B is Qwen2-7B-Instruct — different model families, different training recipes. Yet:

> "while the Qwen2-1.5B and Qwen1.5-4B model variants had similar performance, the 4B Qwen1.5-4B was still more correlated than the 1.5B model"

This is a strong result: Scaling Consistency is not merely an artifact of shared architecture or training within the same model family. A 4B model from a *weaker* model family (Qwen1.5, which the paper notes performs similarly to Qwen2-1.5B despite being larger) still predicts 7B design decisions better than a 1.5B model from a *stronger* family. Size matters more than family.

**Dataset size scaling.** The authors also examined how much training data is needed for the correlations to stabilize. They trained the 0.5B, 1.5B, and 4B variants on datasets ranging from 75K to 1M samples, then computed R² between each variant's performance and the 7B model trained on the full dataset. Figure 3 (right) shows the result:

> "Focusing on the 4B LLM variant, we observed that the correlation (R²) with larger models plateaus around ∼500K samples, indicating that increasing the dataset size beyond this point yields diminishing returns in terms of informing design decisions."

Below 500K samples, the correlation is weaker because the models are undertrained — their performance is dominated by optimization noise rather than the design signal. Above 500K, additional data improves absolute performance but does not change the *relative ranking* of designs. For the 0.5B and 1.5B models, R² fluctuates more with dataset size and never reaches the same plateau, consistent with the "critical model size" finding.

**Relationship to traditional scaling laws.** The authors are careful to distinguish Scaling Consistency from traditional scaling laws (Hoffmann et al., 2022). Traditional scaling laws require training multiple model sizes within the same family to fit a parametric curve (typically a power law) relating size to performance, and only then can one answer questions like "will this architectural choice scale better than that one?" The paper's critique is practical:

> "scaling laws are rarely applied to LMMs. In contrast, Scaling Consistency demonstrates that design decisions made on moderately sized models (∼2 −4B) and datasets transfer reliably to larger models, even across different model families."

Scaling Consistency is a **weaker but cheaper** property. It does not predict absolute performance at scale; it only guarantees that *if* configuration A outperforms configuration B at 3B, it will likely also outperform B at 7B. This is sufficient for making design decisions, which is exactly what the rest of the paper needs.

**Why this is essential for the rest of the paper.** Without Scaling Consistency, the exhaustive ablation studies in Sections 4 and 5 would require verifying every finding at multiple model sizes, which is computationally infeasible. With Scaling Consistency, the authors can:

1. Perform all design-space exploration using a Qwen2.5-3B model (chosen because it is above the critical ~3B threshold).
2. Train on datasets of ~500K–750K samples (above the plateau threshold).
3. Trust that findings will transfer to the 7B model used in the final Apollo models.
4. Report these findings once, rather than re-verifying at each scale.

The paper explicitly states this dependency in Section 4:

> "Using Scaling Consistency, we opted to perform the following exploration using Qwen2.5 3B (Yang et al., 2024) and trained on a dataset of 750K samples. As demonstrated in Sec. 3, these findings exhibit a strong correlation (R² > 0.9) with results on larger models and across different model families."

---

#### Video Sampling: fps vs. Uniform Sampling

**The problem with uniform sampling.** When researchers uniformly sample N frames from a video — dividing the video duration by N and extracting frames at equal intervals — they create a training signal problem. For a short video (say, 10 seconds), N uniformly sampled frames might represent a playback rate of N/10 fps. For a long video (say, 10 minutes), the same N frames represent a playback rate of N/600 fps. The model sees a **different effective "video speed" in every iteration**, because the temporal spacing between consecutive frames varies with video duration. The paper explains:

> "training video-LMMs with uniform frame sampling means that the time difference between concurrent frames changes with each video, effectively setting a different 'video speed' in every iteration"

This is particularly problematic for video encoders, which are typically pretrained at a **constant fps** (the paper notes that InternVideo2, V-JEPA, LanguageBind-Video, and VideoMAE all operate at fixed frame rates). When a video encoder that was trained to expect, say, 2 fps receives frames that are 0.1 seconds apart for one video and 10 seconds apart for another, its temporal representations become unreliable — the model cannot learn stable associations between frame differences and real-world motion speed.

**The fps sampling alternative.** In fps sampling, frames are extracted at a fixed rate (e.g., 2 fps) regardless of video length. For a 30-second video, this yields 60 frames; for a 10-minute video, this yields 1,200 frames. This preserves consistent temporal spacing between consecutive frames, so the model always observes motion at the same "speed." However, fps sampling creates a practical problem: the number of frames grows linearly with video duration, potentially exceeding the LLM's context window or the vision encoder's memory capacity.

The paper's solution is **clip-based fps sampling**. Videos are divided into clips of N consecutive frames (N = 4 for InternVideo2), where each clip is encoded by the video encoder at constant fps. If the total number of clips exceeds a maximum budget, the clips themselves are **uniformly spaced throughout the video** rather than changing the fps within clips. From Section 4.1:

> "An alternate approach is to sample 'video clips' of N frames at a set fps (or duration) and, when reaching the maximum token count, space these out instead. Here, rather than uniformly spacing out the sampled video frames, the N frames encoded by the video encoder maintain the same effective fps, and only frames of concurrent 'clips' are spaced out."

This means the model always sees local temporal structure at a consistent fps (within each clip), and longer videos simply have larger gaps *between* clips, preserving the integrity of the encoded temporal information.

**Experimental validation.** To test whether the fps/uniform distinction matters during training, during inference, or both, the authors ran a careful disambiguation experiment (Figure 4, left and middle). They trained four models with uniform sampling at 8, 16, 32, and 64 frames. They then evaluated each model twice: once with uniform sampling at its training frame count, and once with fps sampling (2 fps, resulting in a variable number of frames depending on video length). The results show:

1. When tested with uniform sampling (Figure 4, left), increasing the number of frames improves performance — 64 frames is better than 8 — but all uniform-trained models underperform fps-trained models.
2. When tested with fps sampling (Figure 4, middle), the models trained with uniform sampling still underperform, and the gap is **not explained** by the number of frames at test time. A model trained with 64 uniform frames, when tested at 2 fps, might still see 64 frames (for a 32-second video), but performs worse than a model trained with 2 fps that also sees 64 frames at test time.

This disambiguation is crucial: the performance gap comes from the **training signal**, not from having more or fewer frames at inference. The paper concludes:

> "this performance gap is not due to the different number of frames sampled at test time... Therefore, we conclude that the uniform frame sampling of videos causes this performance gap during training."

**The fps–tps tradeoff.** When using fps sampling, two parameters control the visual information content: frames per second (fps) and tokens per second (tps). The tokens per second is determined by fps × tpf (tokens per frame), where tpf is set by the token resampler. The authors explored the full grid of fps ∈ {0.5, 1, 2, 4} and tps ∈ {1, 2, 4, 8, 16, 32, 64, 128, 256, 512} (Table 10), where tps = fps × tpf. Each cell in this grid represents a different tradeoff between temporal resolution (fps) and spatial/detail resolution (tokens per frame).

Figure 4 (right) and Appendix Figure 9 show the results as a heatmap. The key finding:

> "there appears to be a tradeoff between tps and fps, balancing short and long video performance, with 8–32 tokens per frame achieving strong performance at different fps"

The heatmap reveals that **tokens per second (tps) is the more important determinant** of performance than fps alone. The diagonal lines of constant tpf (dotted red lines in Figure 4) show that at a given tpf, performance is relatively stable across different fps values. For instance, at 16 tpf (one of the dotted red lines), performance ranges from ~55 to ~58 across fps values from 0.5 to 4. The paper observes:

> "we found little dependence on fps, with both tokens per frame (tpf) and tps being more determinate"

This is a pragmatic finding: if you have a fixed token budget (tps), you can trade off between fps and tpf relatively freely without major performance changes, as long as tpf stays in the 8–32 range. Values of tpf below 4 tokens per frame cause dramatic performance degradation (the bottom-left of the heatmap), while values above 64 show diminishing returns (Appendix Figure 10 shows a plateau).

**Comparison with concurrent work.** The paper notes that Du et al. (2024) independently explored similar questions but reached slightly different saturation points: they needed ~49 tokens per frame for performance saturation versus this paper's 8–32. The paper attributes this difference to two factors: Du et al. used only image encoders with average pooling (which produces less compressible representations than the dual-encoder Perceiver setup), and they used uniform frame sampling (which may require more tokens to compensate for temporal inconsistency).

---

#### Vision Encoder Selection: Single and Dual Encoder Analysis

**Why encoder choice matters.** The vision encoder is responsible for converting raw pixels into feature vectors that the LLM can reason about. Different encoders capture different properties: image encoders (trained on billions of static image-text pairs) produce high-quality spatial representations but have no notion of motion or temporal continuity; video encoders (trained on video datasets) explicitly model temporal dynamics but are typically trained on smaller, noisier datasets and produce lower-quality per-frame features. The choice between them — and whether to combine them — is one of the most consequential design decisions in a video-LMM.

**Encoders evaluated.** The paper tests seven encoders spanning language-supervised and self-supervised paradigms:

- **InternVideo2** (Wang et al., 2024d): a video encoder trained in two stages — (1) masked video token reconstruction (self-supervised), (2) crossmodal contrastive learning aligning video with audio, speech, and text. Encodes 4 frames jointly.
- **LanguageBind-Video v1.5** (Zhu et al., 2023a): a video encoder initialized from OpenCLIP and further contrastively trained with a frozen text encoder. Encodes 8 frames.
- **VideoMAE** (Tong et al., 2022): a video encoder trained through self-supervised masked autoencoding — masking random spatio-temporal patches and learning to reconstruct them. Encodes 16 frames.
- **V-JEPA** (Bardes et al., 2023): a video encoder trained through self-supervised prediction of masked spatio-temporal regions in a learned latent space (not pixel space). Encodes 16 frames.
- **SigLIP-SO400M** (Zhai et al., 2023): a shape-optimized image encoder trained with sigmoid loss for language-image pre-training. Encodes single frames (no temporal aggregation). The "SO" designates shape-optimized architecture.
- **LanguageBind-Image** (Zhu et al., 2023a): an image encoder from the OpenCLIP family, not further tuned. Encodes single frames.
- **DINOv2** (Oquab et al., 2023): a self-supervised image encoder trained using a teacher-student distillation framework. Encodes single frames.

**Single-encoder results.** When each encoder is used alone to encode every frame independently (or every clip for video encoders), SigLIP-SO400M achieves the best overall performance (Figure 5, left). The paper reports specific numbers in Table 9: SigLIP alone scores 52.7 overall on ApolloBench, compared to InternVideo2 at 52.0, LanguageBind-Video at 48.7, and the self-supervised encoders trailing (VideoMAE 41.2, V-JEPA 43.1, DINOv2 45.5).

A crucial decomposition: when the authors break down performance by ApolloBench's temporal perception categories (OCR, Spatial, Egocentric, Perception, Reasoning), they find:

> "Video encoders outperform image encoders only on Temporal Perception, indicating that LLMs struggle with fine-grained temporal integration"

This is visible in Figure 5 (left): video encoders (InternVideo2, LanguageBind-Video) show relatively stronger performance on the Egocentric and Perception dimensions compared to image encoders, while image encoders (SigLIP, LanguageBind-Image, DINOv2) dominate on OCR and Spatial understanding. This makes intuitive sense: OCR and spatial reasoning require high-quality per-frame features, where image encoders excel; egocentric understanding and temporal perception require modeling how things move over time, where video encoders have an advantage.

The paper also confirms a finding from prior image-LMM work:

> "language-supervised encoders consistently outperform self-supervised encoders"

The gap is substantial: the worst language-supervised single encoder (LanguageBind-Image at 47.9) outperforms the best self-supervised single encoder (DINOv2 at 45.5). This is consistent with findings from Cambrian-1 and Eagle that contrastive language-image pretraining produces features more readily usable by LLMs than self-supervised objectives.

**Dual-encoder results.** For dual-encoder configurations, the authors follow the procedure from Shi et al. (2024) and Tong et al. (2024): features from each encoder are spatially interpolated to a common resolution (since different encoders may output different spatial grid sizes) and then concatenated along the channel dimension before being fed to the token resampler. The key design choice is that **channel-wise concatenation happens before resampling**, meaning the Perceiver Resampler (or pooling) receives a feature map with channels from both encoders and can learn to attend to relevant features from either source.

The paper tests all pairwise combinations of encoders. The results (Figure 5, right, and Table 9) show:

1. **Combining encoders consistently outperforms single encoders.** The best single encoder (SigLIP at 52.7) is outperformed by nearly all dual-encoder combinations that include SigLIP. InternVideo2 + SigLIP achieves 57.9 overall on ApolloBench — a ~7% improvement over SigLIP alone.

2. **InternVideo2 + SigLIP-SO400M is the best combination**, achieving 57.9 overall. The breakdown shows it excels across all temporal perception categories, with particularly strong performance on Perception (64.1) and Reasoning (64.5).

3. **Video encoders with fewer input frames perform better in combinations.** InternVideo2 (4 frames) outperforms LanguageBind-Video (8 frames) and VideoMAE/V-JEPA (16 frames) when paired with SigLIP. The authors hypothesize this is due to "better image-video transfer" — encoders that process fewer frames jointly have representations more similar to single-image encoders, making their features more compatible when concatenated with SigLIP features.

4. **Self-supervised video encoders (VideoMAE, V-JEPA) are the weakest in combinations**, even when paired with strong image encoders. VideoMAE + SigLIP achieves 55.0; V-JEPA + SigLIP achieves 55.3 — better than either alone, but substantially below InternVideo2 + SigLIP at 57.9.

**Design justification.** The choice to use SigLIP-SO400M + InternVideo2 as the final Apollo encoder configuration is justified by two complementary strengths: SigLIP provides high-quality spatial features from its massive image-text pretraining, while InternVideo2 adds temporal information from its video-specific training. The fact that InternVideo2 encodes only 4 frames at a time means it can be applied to images as well (by duplicating the image 4 times) without creating a large distribution shift, enabling a unified architecture where the same encoder pair processes both images and videos. The paper explicitly compares unified vs. split architectures in Appendix C.2, finding that unified processing performs slightly better or on par, and is simpler.

---

#### Video Token Resampling: Compressing Visual Information

**Why token resampling matters.** A single frame from the SigLIP-SO400M encoder might produce a feature grid of, say, 27 × 27 = 729 tokens. With InternVideo2 encoding 4 frames per clip, each clip might produce 4 × 729 = 2,916 tokens. At 2 fps, a 5-minute video produces 600 frames = 150 clips = 150 × 2,916 = 437,400 tokens. The LLM's context window (typically 32K–128K tokens for models of this scale) cannot accommodate this, and even if it could, the computational cost of self-attention over 400K tokens would be prohibitive. Token resampling compresses each frame or clip's visual tokens into a smaller, fixed number — say, 32 tokens per frame — reducing the 5-minute video to 600 × 32 = 19,200 tokens, which is manageable.

**Methods compared.** The paper tests three resampling approaches, all operating after the encoder outputs have been interpolated and channel-concatenated:

1. **2-layer MLP + adaptive average pooling:** A two-layer MLP first projects the encoder features up to the LLM's hidden dimension (a 2–4× increase, e.g., from 1152 to 3584 for a 7B Qwen model). Then, adaptive average pooling spatially downsamples the feature grid to a fixed output size (e.g., from 27×27 to 4×4 = 16 tokens per frame).

2. **2-layer 2D Convolution + adaptive average pooling:** Similar to the MLP approach, but uses 2D convolutions for the up-projection instead of pointwise MLP layers. Convolutions can capture local spatial structure before pooling.

3. **Perceiver Resampler** (Jaegle et al., 2021): A learned attention-based pooling mechanism. A fixed set of learnable query vectors (e.g., 32 queries, each of dimension matching the LLM's hidden size) attend to the up-projected encoder features via cross-attention. The queries are learned parameters, not dependent on the input. The output is the set of query vectors after attending to the visual features — exactly T tokens per frame, where T is the number of queries.

The Perceiver Resampler works as follows:

1. The encoder features are first projected to the LLM's hidden dimension via an MLP (same as the other methods).
2. A set of T learned query vectors (initialized randomly and trained) cross-attend to these projected features. The queries ask "what information in this frame is most relevant?"
3. The output of the cross-attention is T vectors, each a learned summary of different aspects of the frame.
4. These T vectors become the visual tokens passed to the LLM.

**Results.** Table 1 shows the Perceiver Resampler outperforming the other methods across all ApolloBench categories:

| Connector | OCR | Spatial | Egocentric | Perception | Reasoning | Overall |
|-----------|-----|---------|------------|------------|-----------|---------|
| 2D Conv + pooling | 43.0 | 50.5 | 44.5 | 44.0 | 42.0 | 44.7 |
| 2-layer MLP + pooling | 47.5 | 53.7 | 51.5 | 52.0 | 61.5 | 53.2 |
| Perceiver Resampler | 50.4 | 54.8 | 58.5 | 58.8 | 55.4 | 55.5 |

The Perceiver Resampler shows the largest advantages on Egocentric (+7.0 over MLP) and Perception (+6.8 over MLP) — the categories most dependent on temporal understanding. The authors attribute the Perceiver's superiority to its ability to attend across the channel-concatenated features from dual encoders, adaptively selecting information from the image encoder (spatial detail) or video encoder (temporal information) as needed for each query vector.

**Contrast with prior work.** The paper notes that Laurençon et al. (2024a) reported that Perceiver Resamplers hurt OCR performance in image-LMMs. The authors observe:

> "this trend was not observed in videos with the limited available token count per frame"

The explanation is likely that in the video domain, the token budget is so constrained (32 tokens per frame vs. potentially hundreds in image-LMMs) that learned attention pooling becomes more valuable — the Perceiver can learn to allocate the scarce token budget to the most informative regions, while simple average pooling wastes tokens on background regions.

Another important detail: the authors perform channel-wise concatenation of encoder features **before** the Perceiver Resampler, rather than resampling each encoder's features independently and then concatenating. This means:

> "This alignment enables the Perceiver to integrate features from different encoders better as they are better spatially aligned."

The Perceiver's cross-attention operates over a feature map where each spatial position has channels from both the image encoder (e.g., capturing texture and object identity) and the video encoder (e.g., capturing motion and temporal change). The query vectors can learn to attend to whichever channel subset is most informative for the task, at whichever spatial location is relevant.

**Methods not studied.** The paper explicitly notes that some approaches use **text-conditioned token pooling** — where the question text is used to guide which visual tokens to keep, typically via a Q-Former (Li et al., 2025, 2023b; Zhang et al., 2023). These were excluded because:

> "this approach does not generalize well to multi-turn conversations, as tokens will be down-sampled according to the first question"

In a multi-turn conversation, if tokens are aggressively pooled based on the first question ("What color is the car?"), the model may have discarded the information needed to answer a follow-up ("What brand is it?"). The paper prioritizes architecture choices that support conversational use.

---

#### Video Token Integration: Interleaving Visual and Text Tokens

**The integration problem.** Once visual tokens are generated (T tokens per frame or clip), they must be inserted into the LLM's input sequence alongside text tokens (the question, system prompt, conversation history). The simplest approach is direct concatenation: `[text tokens] [frame1_tokens] [frame2_tokens] ... [frameN_tokens]`. However, this gives the LLM no explicit information about where one clip ends and another begins, or about the temporal relationship between clips.

**Methods compared.** The paper tests four integration strategies:

1. **Direct insertion (`<vid_token>`):** The raw visual tokens are inserted directly into the text sequence with no additional structural tokens. The LLM sees the visual tokens as a flat sequence with no temporal markers.

2. **Separation tokens (`<vid_start><vid_token><vid_end>`):** Special learnable tokens are inserted before and after each clip's visual tokens. These tokens have randomly initialized embeddings that are trained during the LMM's training. They provide a structural signal that a clip boundary exists.

3. **Textual timestamps (`clip from {MM:SS}-{MM:SS}:<vid_token>`):** Before each clip's visual tokens, a text string indicating the clip's time range is inserted (e.g., "clip from 00:00-00:02: [visual tokens]"). These timestamps use the LLM's existing tokenizer and embedding table — no new tokens need to be learned. They provide explicit temporal information about when in the video each clip occurs.

4. **Separation tokens + timestamps:** Both learned separation tokens and textual timestamps.

**Results.** Table 2 shows the results:

| Format | OCR | Spatial | Egocentric | Perception | Reasoning | Overall |
|--------|-----|---------|------------|------------|-----------|---------|
| `<vid_token>` | 50.4 | 54.8 | 58.5 | 58.8 | 55.4 | 55.5 |
| `<vid_start><vid_token><vid_end>` | 49.2 | 54.8 | 61.7 | 60.2 | 57.9 | 56.7 |
| Timestamps | 50.0 | 54.0 | 61.7 | 60.8 | 57.9 | 56.8 |
| Timestamps + separation tokens | 50.0 | 54.2 | 61.2 | 55.7 | 60.6 | 56.2 |

The key finding:

> "adding any text or learnable tokens between video tokens results in a 2 −3% improvement across ApolloBench"

The simplest form of structural signal — any marker that separates clips — provides most of the benefit. The improvements are concentrated in Egocentric and Perception categories, where temporal structure matters most. The differences between the three structured formats are small (56.2 to 56.8), suggesting that the exact form of the structural signal is less important than its presence.

**Design choice.** The authors select textual timestamps as the default, because:

> "they do not require learning any new token embeddings"

This is a practical consideration: learned separation tokens require adding entries to the LLM's embedding matrix and training those embeddings from scratch, which adds parameters and complexity. Textual timestamps leverage the LLM's existing tokenizer and can be understood immediately (the LLM already knows what "00:00-00:02" means from its text pretraining).

---

#### Training Schedules: Multi-Stage Progressive Unfreezing

**The training stage design space.** Video-LMMs contain multiple components — vision encoders, connector, LLM — that may benefit from different training data, learning rates, and freezing schedules. The paper evaluates seven training configurations spanning 1-stage, 2-stage, and 3-stage protocols (Table 3). Each configuration is defined by:

- **Which components are frozen (✗) or trainable (✓) in each stage:** The vision encoders (ψ_vision), the connector (θ_connector), and the LLM (φ_LLM) can be independently frozen or unfrozen.
- **What data is used in each stage:** When the LLM is frozen, the model is trained on video-only data (since the LLM doesn't need to maintain its language capabilities — it's frozen). When the LLM is unfrozen, a mixture of text, image, multi-image, and video data is used to prevent catastrophic forgetting of language abilities.
- **Learning rates for each trainable component in each stage:** The connector typically uses a higher learning rate (1e-4) than the vision encoders (5e-6) or the LLM (3e-5 for 3B, scaled proportionally for other sizes).

**The seven configurations (from Table 12 in the appendix):**

**1-stage protocols (rows 1–4):** Everything is trained simultaneously in a single stage. The variations test different learning rates for the vision encoders (0, 1e-6, 5e-6, 1e-5) while keeping connector LR at 1e-4 and LLM LR at 3e-5. The best 1-stage configuration (row 1, with frozen vision encoders) achieves only 48.7 overall — substantially below multi-stage approaches.

**2-stage protocols (rows 5–11):** Stage 1 trains only the connector on video data (LLM and vision encoders frozen). Stage 2 unfreezes the LLM and trains on the full data mixture, with the vision encoders either frozen (rows 5–8) or trainable (rows 9–11). The crucial comparison is rows 5–8 vs. 9–11:

- With vision encoders frozen in both stages (rows 5–8), the best result is 57.8 (row 7, with vision LR 5e-6 — though vision is frozen in this config, the LLM learning rate varies).
- With vision encoders trainable in stage 2 (rows 9–11), performance drops to 48.1 at best (row 9). Training vision encoders alongside the LLM on mixed image+video data **significantly hurts performance**.

**3-stage protocols (rows 12–17):** Stage 1 trains only the connector (same as 2-stage). Stage 2 trains the vision encoders on **video-only data** (LLM still frozen). Stage 3 unfreezes the LLM and trains on the full mixture (vision encoders re-frozen or trainable). The best overall configuration:

> "training the model over three stages yields the best performance"

Row 13 achieves 59.2 overall: Stage 1 (connector only) → Stage 2 (vision encoders only, video data, LR 5e-6) → Stage 3 (LLM + connector, full mixture, LLM LR 3e-5, vision encoders re-frozen). This progressive unfreezing — connector first, then encoders, then LLM — allows each component to adapt without interference.

**Why progressive unfreezing works.** The paper's explanation is implicit in the experimental results but can be reconstructed:

1. **Training the connector first** (Stage 1) establishes a good mapping from encoder features to LLM space before the LLM is asked to use those features. If the connector and LLM are trained simultaneously from scratch, the LLM receives noisy, rapidly changing visual representations and cannot learn stable visual-text associations.

2. **Training vision encoders on video-only data** (Stage 2, in the 3-stage protocol) is critical. When vision encoders are trained on mixed image+video data (as in 2-stage protocols where encoders and LLM are unfrozen together), they receive conflicting signals: images (static, high-quality) and videos (temporal, lower-quality) require different representational properties. Training on video-only data lets the encoders specialize for temporal understanding without being pulled toward image-optimized representations.

3. **Freezing vision encoders during LLM training** (Stage 3) prevents the LLM's training signal from distorting the carefully learned video representations. The LLM's language modeling objective may encourage the vision encoders to produce features that are easy to predict from rather than features that accurately represent the video — a form of shortcut learning.

---

#### Training Video Encoders: When and with What Data

The paper explicitly investigates this question in Section 5.2. The main finding:

> "if both the video and LLM are unfrozen simultaneously, the vision encoders will be trained on a combination of image and video data. We found that this significantly hurts LMM performance."

The mechanism is data interference. When the LLM is unfrozen, the training mixture includes text, image, multi-image, and video data (following the optimal mixture from Section 5.3). If the vision encoders are also trainable, they receive gradients from both image and video examples. Image examples provide strong, clean gradients (static scenes, high-quality annotations), while video examples provide weaker, noisier gradients (motion blur, lower resolution, more complex annotations). The encoders may drift toward image-optimized representations that sacrifice temporal sensitivity — exactly what the dual-encoder setup was designed to prevent.

The paper finds a narrow window where training video encoders helps:

> "Finetuning video encoders on only video data further improves overall performance, especially on reasoning and domain-specific tasks"

Specifically, the 3-stage protocol with vision encoder training in Stage 2 (video-only data) improves Egocentric and Reasoning scores. Table 12: comparing row 13 (encoders trained, 59.2 overall) to row 12 (encoders frozen, 55.4 overall) shows gains concentrated in Egocentric (+2.2) and Reasoning (-0.4, but with better OCR and Spatial).

The paper also notes alignment with concurrent work:

> "These insights are in line with Zhao et al. (2024b)'s report"

This suggests the finding is robust across different implementations.

---

#### Data Composition: The Role of Text, Image, and Video Mixtures

**Why data mixture matters.** Video-LMMs are typically trained on a combination of text-only, image-text, multi-image, and video data. Each modality provides different training signals: text data maintains the LLM's language capabilities and prevents catastrophic forgetting; image data provides high-quality visual-text alignment from large, diverse datasets; video data provides temporal understanding. The optimal mixture is not obvious — too much text might crowd out visual learning; too little text might cause language degradation.

**Experimental design.** The paper tests 13 different data compositions (Table 13), varying the proportions of text, image, multi-image, and video data. The total dataset size is held constant at the SFT stage's 3.2M samples. The compositions range from text-heavy (25% text, balanced across other modalities) to video-only (93% video, 7% text), with many intermediate points.

**Results.** Figure 6 visualizes the key finding:

> "having ∼10 −14% text data is important for video understanding performance"

Compositions with 14–15% text achieve the highest overall scores (rows 2 and 4: 59.0 and 56.2). Increasing text to 25% (row 1: 54.1) hurts performance, suggesting the extra text data crowds out valuable visual training examples. Decreasing text below 7% (rows 5–13) causes progressive degradation, with the 0% text configuration (row 13: 47.5) performing worst among reasonable mixtures.

The paper characterizes the optimal region:

> "including 10 ∼14% text data in the training mix is required for performance. This likely alleviates catastrophic forgetting."

The mechanism is standard in continual learning: when fine-tuning a pretrained LLM on new modalities (video tokens), the model's language capabilities can degrade if no pure text examples are included. The text data acts as a "rehearsal" signal, reminding the model of the language task while it learns to process visual inputs.

**Beyond text, the video-to-image ratio matters.** Among the 14%-text mixtures, the paper tested image-heavy (40/14/20/25 for image/text/multi-image/video), balanced (32.5/15/20/32.5), and video-heavy (25/15/20/40) distributions:

> "having a slightly video-heavy mix of the remaining modalities was preferable"

The video-heavy mixture (row 2: 40% video, 25% image) achieves 59.0 overall, outperforming the balanced mixture (row 3: 32.5% each, 57.1) and the image-heavy mixture (row 4: 25% video, 40% image, 56.2). The authors explain:

> "This balance allows the model to learn from higher-quality and diverse image datasets"

The image datasets provide cleaner, more diverse visual supervision than video datasets (which are smaller and noisier), so including some image data is beneficial. But video data must dominate slightly to ensure the model actually learns temporal reasoning.

**Extreme compositions fail.** The 93% video, 7% text composition (row 8: 41.8 overall) performs worst among all tested mixtures, even worse than 0% text (row 13: 47.5). This suggests that training almost exclusively on video data without sufficient image and text data leads to poor representations — the model overfits to the specific distribution of video datasets and loses general visual understanding. The 0% text, 38.7% image, 20% multi-image, 41.3% video mixture (row 13) performs poorly (47.5) but not as badly as the video-dominated one, consistent with the "text is essential" finding.

---

#### ApolloBench Curation: Efficient, Diagnostic Evaluation

**The problem being solved.** The paper's Section 2 reveals that existing video QA benchmarks suffer from two problems: (1) many questions can be answered without watching the video, and (2) benchmarks are highly redundant with each other. This makes evaluation both expensive and potentially misleading — a model might score well by exploiting language biases rather than actually understanding videos.

**Curation pipeline.** The ApolloBench creation process (detailed in Section 2.3 and Appendix B.3, with a flowchart in Figure 11) proceeds through five stages:

1. **Collection:** Start with a collection of existing multiple-choice benchmarks (Video-MME, TempCompass, MLVU, LongVideoBench, NExT-QA, Perception-Test). The authors explicitly choose multiple-choice format to eliminate the need for external LLM-based scoring (e.g., ChatGPT), which the paper notes can produce scores "even 10% apart" depending on GPT version.

2. **Modality filtering:** Evaluate 10 open-source LMMs on every question under three conditions: full video input, single center frame only, and text-only (no visual input). Filter out any question that can be correctly answered by more than 50% of the models using either text-only or single-frame input. These questions do not require video perception and are therefore not useful for evaluating video understanding.

3. **Categorization:** Manually categorize remaining questions into five temporal perception categories: Temporal OCR (reading text that appears/disappears over time), Egocentric (understanding actions from a first-person perspective), Spatial (understanding 3D layout and object relationships), Perception (detecting events, actions, and changes), and Reasoning (drawing inferences that require temporal integration).

4. **Discriminability selection:** For each category, compute the entropy of model predictions across the 10 evaluated LMMs. Questions with higher entropy — meaning models disagree more about the answer — are more discriminative and better at separating strong from weak models. Select questions with high entropy.

5. **Manual verification:** Manually inspect each selected question to validate correctness and clarity. Select the top 400 questions (balanced across categories) as the final ApolloBench.

**Efficiency gains.** The paper claims:

> "Evaluating on ApolloBench is 41× faster while being highly correlated with existing benchmarks"

The 41× speedup comes from two sources: (1) ApolloBench has only 400 questions versus thousands in the combined benchmark suite, and (2) the questions are selected to be discriminative, so fewer questions provide equivalent information. Figure 2 (right) shows ApolloBench achieves high correlation (R² > 0.8) with every individual benchmark in the suite, and the "ApolloBench" row of the correlation matrix shows consistently high values.

**Quality improvements.** Figure 2 (left) shows ApolloBench has substantially larger video-vs-text and video-vs-image gaps than any individual benchmark, meaning its questions more strongly require actual video perception. The filtering step explicitly removed questions that could be answered from text or a single frame, leaving only questions where temporal information matters.

**Limitation acknowledged.** The paper notes that restricting to multiple-choice questions limits evaluation to a specific format and recommends:

> "a benchmark focusing solely on a conversation is needed, ideally, one that does not suffer from high API costs and GPT versioning noise"

This is flagged as future work in Appendix A.

## 4. Key Insights and Innovations

### Innovation 1: Scaling Consistency Reframes the Relationship Between Small and Large Models as a Design-Transfer Property, Not a Performance-Prediction One

The dominant paradigm for understanding how model size affects behavior comes from scaling laws (Hoffmann et al., 2022), which establish mathematical relationships between size, data, and performance—enabling *extrapolation* of absolute performance numbers from small to large models. This paper introduces a fundamentally different concept: **Scaling Consistency**, the empirical finding that *design decisions* (relative rankings of architectural choices, training protocols, and data mixtures) transfer from moderately sized models to large ones with R² > 0.9 above a critical ~3B parameter threshold. This is not a scaling law. It does not predict that a 3B model achieving 55% accuracy implies a 7B model will achieve some specific higher number. It predicts something more subtle and, for the purpose of research iteration, more useful: that if configuration A outperforms configuration B at 3B, it will almost certainly also outperform B at 7B—even across different model families.

**What makes this conceptually distinctive.** Prior work on LMM design (Karamcheti et al., 2024; Tong et al., 2024; Shi et al., 2024) conducted systematic ablations but typically verified findings at a single scale or trained models of multiple sizes without establishing *why* or *whether* small-model findings should generalize. The implicit assumption in the field was that to trust a design decision, you needed to verify it at the target scale—which is precisely what makes video-LMM research computationally prohibitive. Scaling Consistency provides an empirical license to *not* verify at scale, backed by quantitative evidence (84 models across 4 LLM sizes, Figure 3, Appendix Figure 15) that the correlation follows an approximately log-linear relationship with model size. This shifts the burden of proof: the default assumption becomes that small-model findings transfer, and the exception that needs demonstrating is when they do not.

**Comparison to scaling laws.** Traditional scaling laws require training 3–5 model sizes *for each design variation* to fit a parametric curve, which is computationally infeasible for LMMs that integrate multiple pretrained components. Scaling Consistency relaxes this by asking a weaker but more actionable question: not "what will the performance be?" but "which choice is better?" The paper explicitly contrasts this with scaling laws in Appendix D:

> "In scaling laws, researchers train around 3–5 models of different sizes to establish scaling relationships, and only then can they determine which design decisions are beneficial at larger scales. In contrast, Scaling Consistency shows that design decisions on moderately sized models transfer well to larger ones, even across different model families."

The "even across different model families" detail is important: the 4B model used in the correlation analysis is Qwen1.5-4B-Chat, while the 7B model is Qwen2-7B-Instruct. These are different architectures with different training recipes, yet R² = 0.938 between them. This suggests Scaling Consistency is not an artifact of shared inductive biases within a model family but a genuine property of *sufficient capability*. Once a model is capable enough (above the ~2–4B critical threshold), its design preferences reflect what matters for the task, not what compensates for its own limitations.

**The critical-size threshold as a diagnostic concept.** The finding that Scaling Consistency breaks down for models below ~2B parameters—with the 0.5B model showing essentially random correlation (R² = 0.694 with 7B) and no log-linear improvement as it scales—introduces a new diagnostic: *below a critical capability threshold, models cannot serve as reliable proxies for design decisions because they are too constrained to express the design's true value.* A 0.5B model might prefer an inferior encoder not because it is better for video understanding but because it produces simpler features the limited-capacity LLM can more easily process. This is not a failure of the methodology but a boundary condition that makes the methodology more precise: Scaling Consistency is not "small models always work" but "models above ~3B work, and the correlation improves log-linearly with size above that threshold."

**Significance beyond performance.** This innovation is primarily methodological rather than performance-oriented—it does not directly improve any benchmark score. Its significance lies in **changing who can participate in video-LMM research**. By establishing that a 3B model trained on 500K samples can reliably inform design decisions for production-scale models, Scaling Consistency lowers the computational barrier to entry from "industrial lab with hundreds of GPUs" to "academic group with a modest cluster." This is a structural contribution to the field's research ecosystem, not a model architecture contribution.

**Evidence anchor.** Figure 3 (left) and the underlying correlation plots in Appendix Figure 15 provide the quantitative backbone. The log-linear fit y = 0.12·log(x) + 0.78 (where y is R² between a model of size x and the 7B model) predicts R² > 0.9 at x ≈ 3B and, by extrapolation, R² ≈ 0.86 between 32B and 72B models. The dataset-size analysis in Figure 3 (right) adds a second dimension: correlations plateau at ~500K samples, establishing a "sufficient data" threshold analogous to the "sufficient size" threshold.

---

### Innovation 2: The Benchmark Quality Analysis Reveals That Existing Video Benchmarks Primarily Measure Language and Static-Image Understanding, Not Temporal Reasoning

The paper's Section 2 analysis of video QA benchmarks—evaluating 10 LMMs under three conditions (full video, single center frame, text-only)—produces a finding that is simultaneously obvious in retrospect and devastating in its implications: **a large fraction of existing video benchmarks do not actually require video perception.** The paper quantifies this with unprecedented clarity (Figure 2, left), showing that for many benchmarks, the accuracy gap between full-video and text-only or single-frame inputs is small—meaning the questions can be answered without watching the video at all. NExT-QA and Perception-Test, two widely used benchmarks, are effectively solved from a single frame. And counterintuitively, as videos get longer (Video-MME Short → Medium → Long), reliance on video perception *decreases*—likely because longer videos provide more textual and contextual cues from which answers can be inferred without genuine temporal understanding.

**What makes this conceptually distinctive.** Prior work flagged language biases in image QA (Goyal et al., 2017's famous "Making the V in VQA Matter") and some individual video studies noted similar concerns (Buch et al., 2022, cited in the paper). But no prior work performed a systematic, multi-benchmark, three-condition evaluation that simultaneously reveals: (1) which benchmarks measure video perception at all, (2) how this varies with video duration, and (3) how redundant benchmarks are with each other. The three-condition protocol (video vs. center frame vs. text-only) is itself a methodological contribution—it provides a simple, cost-effective diagnostic for future benchmark development. Any new video QA benchmark should demonstrate a substantial video-vs-text gap to justify its computational cost.

**The inverse relationship between video duration and video reliance.** This is the paper's most counterintuitive benchmarking finding. The common assumption in the field is that long-video benchmarks are *more* demanding of video understanding because they require integrating information over longer timespans. The paper's analysis suggests the opposite: as videos get longer, the questions become *easier* to answer from text alone or from a single frame, because the video content provides more contextual clues. This may reflect a systematic flaw in how long-video benchmark questions are constructed—they may ask about content that can be inferred from a few key frames or from the accompanying narration, rather than requiring genuine temporal integration across the video's full duration.

**Benchmark redundancy as a resource-allocation problem.** The correlation analysis (Figure 2, right) reveals that existing benchmarks form highly correlated clusters (Video-MME's duration groups cluster together, TempCompass's question types cluster together), with R² > 0.92 between LongVideoBench's duration groups and R² > 0.83 between Video-MME's duration groups. This means evaluating on the full suite provides little additional information beyond evaluating on one representative benchmark—yet the full suite costs 184 A100 GPU hours for a single 3B model. The paper quantifies the waste precisely, which is itself valuable: it tells the field exactly how much compute is being burned on redundant evaluation.

**ApolloBench as a principled solution, not just another benchmark.** Rather than introducing a new benchmark (which would add to the proliferation problem), the authors curate a subset of existing questions that are: (a) actually dependent on video perception (filtered via the three-condition test), (b) discriminative between models (selected via entropy), and (c) categorized by temporal perception type (OCR, Egocentric, Spatial, Perception, Reasoning). This last point is underappreciated: by categorizing questions into fine-grained temporal perception types, ApolloBench provides more diagnostic information than any individual benchmark—it can tell you not just that a model is good or bad at "video understanding," but whether it struggles specifically with egocentric reasoning or temporal OCR. The 41× speedup is a consequence of this curation, not its primary goal.

**Significance beyond performance.** Like Innovation 1, this contribution is methodological and structural. It changes how the field evaluates video-LMMs by providing both a diagnostic tool (the three-condition test) and a cost-effective benchmark (ApolloBench). It also establishes a quality standard that future benchmarks should meet: demonstrate that your questions require video perception, are discriminative, and are not redundant with existing benchmarks. The paper's own Apollo models benefit from this in evaluation cost, but the contribution is intended for the entire field.

**Evidence anchor.** Figure 2 is the primary evidence, with the left panel showing the video-vs-text and video-vs-image gaps for each benchmark and the right panel showing the correlation matrix. The raw evaluations supporting this analysis are in Tables 5 and 6. The detailed correlation analyses within benchmarks (duration groups in Figures 13 and 14, question types in Figure 12) provide the redundancy quantification.

---

### Innovation 3: fps Sampling vs. Uniform Sampling Is Reframed as a Training Signal Problem, Not a Frame-Count Problem

Prior work on video sampling for LMMs treated the choice between uniform and fps sampling as largely a matter of implementation convenience or maximum frame capacity. The implicit assumption was that fps sampling was preferable when videos were short enough to fit within the frame budget, and uniform sampling was a necessary fallback for longer videos. The paper reframes this as a **training signal integrity problem**: uniform sampling causes the model to observe a different effective "video speed" in every training iteration because the temporal spacing between consecutive frames varies with video duration. This prevents the model from learning stable associations between frame differences and real-world motion speed. The innovation is not the observation that fps sampling can be used—it is the specific diagnosis of *why* uniform sampling fails and the demonstration that the performance gap originates during training, not during inference.

**What makes this conceptually distinctive.** The disambiguation experiment (Figure 4, left vs. middle) is a model of clean experimental design. By training models with uniform sampling and testing them both with uniform sampling and with fps sampling, the authors isolate the training effect from the inference effect. The result—that fps-tested performance remains poor for uniform-trained models, and increasing the number of uniform training frames does not close the gap—demonstrates that the problem is specifically in what the model learns during training, not in how many frames it sees at test time. This distinguishes the paper's finding from the simpler observation that "more frames is better" and from concurrent work like Du et al. (2024) that reached similar conclusions but without decomposing the training vs. inference contribution.

**The clip-based fps sampling solution.** The paper proposes sampling videos as sequences of clips, where each clip maintains constant fps but clips themselves can be uniformly spaced when the video is too long. This preserves temporal integrity within clips (so the model always sees consistent local motion) while handling arbitrarily long videos. Prior work either truncated videos, switched to uniform sampling for long videos, or used uniform sampling throughout. The clip-based approach is a hybrid that preserves the benefits of fps sampling within the model's frame budget constraints.

**The fps–tps tradeoff as a resource-allocation insight.** The finding that tokens per second (tps) matters more than fps alone, and that 8–32 tokens per frame works well across a range of frame rates (Figure 4, right; Appendix Figure 9), provides an actionable guideline for practitioners: if you have a fixed token budget, you have flexibility in how you allocate it between temporal resolution (fps) and spatial detail (tokens per frame). The paper's exploration of the full fps × tps grid (Table 10, 28 configurations) establishes this empirically, and the finding that tpf below 4 causes catastrophic performance degradation while tpf above 64 gives diminishing returns provides clear boundaries for the design space.

**Comparison to concurrent work.** Du et al. (2024) independently explored similar questions but needed ~49 tokens per frame for saturation, compared to this paper's 8–32. The paper attributes this difference to their use of image encoders with average pooling (less compressible) and uniform frame sampling (which may require more tokens to compensate for temporal inconsistency). This comparison strengthens the finding by showing it is robust across different architectural choices but also reveals how architecture and sampling interact—the optimal tpf is not a universal constant but depends on encoder choice and sampling strategy.

**Significance beyond performance.** This is a fundamental design insight, not an incremental improvement. It changes the default assumption from "uniform sampling is the standard, fps is an optimization" to "fps sampling should be the default, and uniform sampling should only be used when fps sampling is impossible." It also provides a conceptual framework for thinking about video sampling: the goal is to provide the model with a consistent temporal signal during training so it can learn stable representations of motion and temporal relationships. Any sampling strategy that varies the effective temporal resolution across training examples undermines this goal.

**Evidence anchor.** Figure 4 (left and middle) provides the core evidence. Table 10 (28 fps × tps configurations) and Table 11 (uniform sampling baselines) contain the raw data. Appendix Figure 9 shows the per-category breakdown of the fps–tps tradeoff.

---

### Innovation 4: The Dual-Encoder Finding—SigLIP as Best Single Encoder, InternVideo2+SigLIP as Best Pair—Resolves a Field-Level Tension Between Image and Video Encoders

The video-LMM field has been divided between two camps: those using dedicated video encoders (e.g., Video-LLaVA, VideoChat) that jointly process multiple frames and can in principle capture temporal dependencies, and those using image encoders (e.g., LLaVA-OneVision, Oryx, Kangaroo) that process each frame independently and leave temporal reasoning to the LLM. The paper's systematic comparison of seven encoders and their pairwise combinations (Figure 5) provides a definitive resolution: **SigLIP-SO400M, an image encoder, is the best single encoder for video-LMMs**, outperforming all tested video encoders including InternVideo2, LanguageBind-Video, VideoMAE, and V-JEPA. However, **combining SigLIP with InternVideo2 yields the best overall performance**, improving ~7% over SigLIP alone on ApolloBench (52.7 → 57.9).

**What makes this conceptually distinctive.** Prior work comparing encoders (Shi et al., 2024; Tong et al., 2024) focused on image-LMMs and did not address the image-vs-video encoder question systematically. Individual video-LMM papers chose one approach or the other, and there was no controlled comparison on equal footing. The paper's finding that video encoders outperform image encoders *only on temporal perception dimensions* (Egocentric, Perception in Figure 5, left) while image encoders dominate on OCR and Spatial understanding provides a mechanistic explanation for the dual-encoder benefit: the image encoder provides high-quality spatial features from massive image-text pretraining, while the video encoder adds temporal sensitivity that the LLM cannot easily infer from static frames. The dual-encoder setup is not just "better" in aggregate but better for specific, complementary reasons.

**The "fewer input frames is better for dual encoders" insight.** The finding that InternVideo2 (4 frames) outperforms LanguageBind-Video (8 frames) and VideoMAE/V-JEPA (16 frames) when paired with SigLIP is a non-obvious architectural insight. The authors hypothesize this is due to "better image-video transfer"—encoders processing fewer frames jointly produce representations more similar to single-image encoders, making their features more compatible when concatenated with SigLIP features. This has practical implications for encoder selection: the best video encoder for a dual-encoder setup is not necessarily the best standalone video encoder.

**Language supervision vs. self-supervision.** The finding that language-supervised encoders consistently outperform self-supervised ones extends a known result from image-LMMs (Shi et al., 2024) to the video domain. But the magnitude of the gap—the worst language-supervised single encoder (LanguageBind-Image, 47.9) outperforming the best self-supervised single encoder (DINOv2, 45.5)—reinforces that contrastive language-image pretraining produces features more readily usable by LLMs than self-supervised objectives, even for video tasks.

**Significance beyond performance.** This innovation resolves a field-level design tension and provides a clear default: use SigLIP as the base image encoder, and if budget allows, add InternVideo2 for temporal sensitivity. The paper does not claim this is the optimal encoder pair for all possible settings—future, better video encoders may emerge—but it establishes a principled methodology for evaluating encoder choices: test single encoders first, then test combinations, decompose performance by temporal perception categories, and prefer encoders with fewer input frames for combination.

**Evidence anchor.** Figure 5 provides the visual summary. Table 9 contains the full raw results for all 19 encoder configurations (7 single, 12 dual), broken down by ApolloBench categories.

## 5. Experimental Analysis

### Evaluation Methodology

- **Dataset.** All design-space exploration experiments (Sections 4 and 5) use a training dataset of 750K samples following the data mixture findings from Section 5.3 (text, image, multi-image, and video data). For the final Apollo models (Section 6), the training data is expanded to 3.2M samples for the SFT stage. Evaluation uses ApolloBench (400 curated multiple-choice questions, described in Section 2.3 and Appendix B.3), plus the full existing benchmark suites for comparison with prior work: Video-MME (Fu et al., 2024), MLVU (Zhou et al., 2024), LongVideoBench (Wu et al., 2024), TempCompass (Liu et al., 2024c), Perception-Test (Patraucean et al., 2023), and NExT-QA (Xiao et al., 2021).

- **Base model(s).** Design-space exploration in Sections 4 and 5 uses Qwen2.5-3B-Instruct as the LLM backbone, chosen because it sits above the critical ~3B threshold identified by Scaling Consistency (Section 3, Figure 3 left). The final Apollo models (Section 6) use three scales of the Qwen2.5 series (Yang et al., 2024): 1.5B, 3B, and 7B parameters. This family is chosen because it spans the critical size threshold and because the Scaling Consistency experiments (Section 3) demonstrated that rankings from Qwen1.5 and Qwen2 models at ~3–4B transfer to Qwen2-7B with R² > 0.9. Unless otherwise noted, the dual vision encoders are InternVideo2 (Wang et al., 2024d) and SigLIP-SO400M (Zhai et al., 2023), with a Perceiver Resampler (Jaegle et al., 2021) producing 16 tokens per frame at 2 fps.

- **Metrics.** The primary metric is **accuracy** — the fraction of questions answered correctly. For ApolloBench, accuracy is reported both as an overall average and broken down into five temporal perception categories: OCR, Spatial, Egocentric, Perception, and Reasoning (defined in Section 2.3 and Appendix B.3). For the full benchmark suite, standard dataset-specific scoring protocols are followed (e.g., Video-MME reports accuracy with and without subtitles for different duration groups; TempCompass reports separate scores for multiple-choice, yes/no, caption matching, and caption generation). All evaluations are conducted using lmms-eval (Zhang et al., 2024b), an open-source evaluation framework.

- **Baselines.** The paper evaluates against a comprehensive set of open-source and proprietary video-LMMs. From the proprietary category: GPT-4V and GPT-4o (OpenAI, 2023, 2024), Gemini-1.5-Flash and Gemini-1.5-Pro (Team et al., 2023), and Claude-3.5-Sonnet (Anthropic, 2024). From open-weight and open-source models: Qwen2VL-2B/7B/72B (Wang et al., 2024a), Aria 8×3.5B (Li et al., 2024b), Pixtral-12B (Agrawal et al., 2024), LLaVA-OV-0.5B/7B (Li et al., 2024a), VILA1.5 3B/40B (Lin et al., 2024), InternVL2-2B/8B/34B (Chen et al., 2024b), Phi-3.5-Vision-4.2B (Abdin et al., 2024), LongVU 3.2B/7B (Shen et al., 2024), LongVA-7B (Zhang et al., 2024e), XComposer-8B (Zhang et al., 2024d), Kangaroo-8B (Liu et al., 2024b), Video-XL 7B (Shu et al., 2024), Oryx 7B/34B (Liu et al., 2024d), and LLaVA-N-Video-32B (Zhang et al., 2024f). For the benchmarking quality analysis in Section 2, 10 open-source LMMs are evaluated under three input conditions (video, center frame, text-only) — these include InternVL2 2B/8B, LLaVA-OV 0.5B/7B, LongVA 7B, Qwen2-VL 2B/7B, VILA-1.5 3B/8B, and XComposer-8B.

- **Generation budget / compute accounting.** The paper does not report generation budgets per question (as all evaluations use standard multiple-choice prompting with a single greedy output). For training, the relevant compute metric is the number of training samples and the model size. Design-space exploration (Section 4) trains each variant on 750K samples using Qwen2.5-3B. Scaling Consistency experiments (Section 3) train 84 models across 4 LLM sizes. The final Apollo models are trained in three stages totaling approximately 3.8M samples (198K alignment + 396K vision pretraining + 3.2M SFT). All training uses 128 NVIDIA A100 GPUs with ZeRO2 optimization. The key "budget" concept in this paper applies to *training compute* needed for design-space exploration — the paper's contribution is making this budget affordable via Scaling Consistency — rather than to test-time compute budgets.

- **Cross-validation / statistical protocol.** For the Scaling Consistency correlation analysis (Section 3), the authors compute Pearson R² between the ApolloBench scores of 21 model configurations trained at each LLM size (84 total models trained). No cross-validation is applied to these correlations, as the goal is to measure the empirical relationship between rankings at different scales. For the design-space exploration experiments (Sections 4 and 5), models are trained once per configuration on the full training set and evaluated on the fixed ApolloBench test set. The paper notes in Section 3 that "three hyperparameters were tested" for each training schedule (Table 3), and "we report the best-performing model" — this introduces a hyperparameter selection step that is not cross-validated and could contribute some variance to the reported numbers. For the benchmark quality analysis (Section 2), 10 LMMs are evaluated on all benchmarks, and the reported statistics (correlations, modality gaps) are computed across models.

### Main Quantitative Results

#### Scaling Consistency: Correlation Between Design Decisions Across Model Sizes

The headline result is that design decisions made with a Qwen1.5-4B-Chat model achieve R² = 0.938 with those from a Qwen2-7B-Instruct model, despite being from different model families (Section 3, Figure 3 left; Appendix Figure 15). The correlation follows an approximately log-linear relationship:

> "y = 0.12 · log(x) + 0.78"

where `y` is the R² between a model of size `x` (in billions of parameters) and the 7B model. Specific correlations: 0.5B → 7B yields R² = 0.694; 1.5B → 7B yields R² = 0.831; 4B → 7B yields R² = 0.938 (Appendix Figure 15, first row of plots). The inverse comparisons (7B → 0.5B, 4B → 0.5B, etc.) show progressively weaker correlations — the 0.5B model's rankings are essentially random with respect to larger models (R² drops to 0.694 with 7B and 0.772 with 4B; Appendix Figure 15, bottom row).

For dataset size scaling (Figure 3, right), the 4B model's R² with the 7B model trained on the full dataset plateaus at approximately 500K samples. Below this threshold, R² is lower and more variable; above it, additional data improves absolute performance but does not change relative rankings. The 1.5B and 0.5B models show more erratic behavior with no clear plateau. This establishes both a **critical model size** (~2–4B) and a **critical dataset size** (~500K) for Scaling Consistency to hold.

#### Video Sampling: fps vs. Uniform Sampling Performance Comparison

Models trained with fps sampling consistently outperform those trained with uniform frame sampling, even when controlling for the number of frames at test time (Figure 4, left and middle). The uniform-trained models, evaluated with uniform sampling at their training frame count, achieve: 8 frames → 44.2 overall on ApolloBench, 16 frames → 48.1, 32 frames → 51.9, 64 frames → 55.1 (Table 11, rows 1–4). When the same uniform-trained models are tested with fps sampling (2 fps, variable frame count), scores improve but remain below fps-trained baselines: 8-frame trained → 49.0, 16-frame trained → 52.6, 32-frame trained → 52.6, 64-frame trained → 53.8 (Table 11, rows 5–8). The highest uniform-trained score (55.1 at 64 frames trained and tested uniformly) is below the best fps-trained configuration (58.3 at 2 fps, 64 tps, 32 tpf; Table 10, row 10).

The fps × tps grid exploration (Table 10, 28 configurations; Figure 4, right) shows the best overall performance comes from configurations with 8–32 tokens per frame (tpf), with the top performers being: 2 fps, 64 tps, 32 tpf → 58.3 overall; 4 fps, 64 tps, 16 tpf → 58.2; 1 fps, 16 tps, 16 tpf → 58.1. Performance degrades sharply when tpf drops to 4 or below: 4 fps, 16 tps, 4 tpf → 21.5 overall (row 21), and 2 fps, 4 tps, 2 tpf → 32.4 (row 26). Very high tpf values also show diminishing returns: 4 fps, 512 tps, 128 tpf → 52.4 (row 1), which is substantially below the 58+ achieved at moderate tpf.

The per-category breakdown (Appendix Figure 9) reveals that OCR and Spatial understanding decline sharply when tps is reduced, especially at low tps values (2–4), regardless of fps. Egocentric and Reasoning tasks are less sensitive to tps reductions and more influenced by fps. The Perception dimension is anomalous, showing a preference for lower fps values.

#### Vision Encoder Selection: Single and Dual Encoder Comparison

**Single encoders** (Table 9, rows 1–7; Figure 5, left). SigLIP-SO400M is the best single encoder with 52.7 overall on ApolloBench, followed by InternVideo2 at 52.0 and LanguageBind-Video at 48.7. The self-supervised encoders trail: DINOv2 at 45.5, V-JEPA at 43.1, VideoMAE at 41.2, and LanguageBind-Image at 47.9. The breakdown by category shows SigLIP dominating OCR (41.9 vs. InternVideo2's 43.7 — a rare case where InternVideo2 leads) and Spatial (52.2 vs. 46.5), while InternVideo2 leads on Egocentric (56.4 vs. 57.4) and Reasoning (58.1 vs. 60.0). Video encoders outperforming image encoders only on Temporal Perception (Egocentric and Perception) confirms the complementary strength hypothesis.

**Dual encoders** (Table 9, rows 8–19; Figure 5, right). InternVideo2 + SigLIP-SO400M achieves the best overall performance at 57.9, representing a ~7% improvement over SigLIP alone (52.7 → 57.9). The per-category results: OCR 48.8, Spatial 56.4, Egocentric 59.9, Perception 64.1, Reasoning 64.5. This configuration dominates across all categories, not just temporal ones. The second-best dual configuration is LanguageBind-Video + SigLIP at 53.4, followed by V-JEPA + SigLIP at 55.3. VideoMAE-based dual configurations score lower (VideoMAE + SigLIP at 55.0) but still outperform single encoders. Dual configurations involving self-supervised encoders (VideoMAE + DINOv2 at 49.6, V-JEPA + DINOv2 at 49.0) perform poorly — worse than SigLIP alone. The finding that InternVideo2 (4 input frames) outperforms LanguageBind-Video (8 frames), VideoMAE (16 frames), and V-JEPA (16 frames) in dual configurations is visible in the consistent ~2–5 point gap between InternVideo2 + SigLIP (57.9) and the next-best video-encoder pairing (LanguageBind-Video + SigLIP at 53.4).

#### Video Token Resampling: Perceiver vs. Pooling Methods

The Perceiver Resampler outperforms both 2D convolution + average pooling and MLP + average pooling across all ApolloBench categories (Table 1). The overall scores: Perceiver Resampler at 55.5, 2-layer MLP + adaptive average pooling at 53.2, and 2-layer 2D Conv + adaptive average pooling at 44.7. The Perceiver's advantage is largest on Egocentric (58.5 vs. 51.5 for MLP, a +7.0 gap) and Perception (58.8 vs. 52.0 for MLP, a +6.8 gap). The 2D Conv + pooling method performs dramatically worse across all categories (OCR 43.0 vs. 47.5 for MLP and 50.4 for Perceiver; Reasoning 42.0 vs. 61.5 for MLP and 55.4 for Perceiver), with the Reasoning gap of nearly 20 points being particularly severe.

#### Video Token Integration: Separation Tokens and Timestamps

Adding any structural text or learned tokens between clips improves performance by ~2–3% over direct insertion (Table 2). Direct insertion (`<vid_token>` only) achieves 55.5 overall. Adding learned separation tokens (`<vid_start><vid_token><vid_end>`) improves to 56.7 (+1.2). Textual timestamps (`clip from {MM:SS}-{MM:SS}:<vid_token>`) achieve 56.8 (+1.3). Timestamps with separation tokens achieve 56.2 (+0.7). The differences between the three structured formats are small (range 56.2–56.8), indicating that the specific format is less important than the presence of any structural signal. The improvements are concentrated in Egocentric (58.5 → 61.7 for all structured formats) and Perception (58.8 → 60.2–60.8).

#### Training Schedules: Multi-Stage Progressive Unfreezing

The 3-stage training protocol achieves the best overall performance at 59.2 on ApolloBench (Table 3, row 13; Table 12, row 13), compared to the best 2-stage protocol at 57.8 (Table 12, row 7) and the best 1-stage protocol at 48.7 (Table 12, row 1). The specific 3-stage configuration is: Stage 1 (connector only, video data) → Stage 2 (vision encoders only, video data, vision LR 5e-6) → Stage 3 (LLM unfrozen, full mixture, LLM LR 3e-5, connector LR 1e-4, vision encoders re-frozen). The per-category scores for this best configuration: OCR 52.4, Spatial 55.4, Egocentric 62.8, Perception 63.5, Reasoning 61.4.

The 2-stage configurations where vision encoders are frozen in both stages (Table 12, rows 5–8) perform consistently well (56.3–57.8), with the LLM learning rate being the main variation. The 2-stage configurations where vision encoders are trainable in Stage 2 (Table 12, rows 9–11) perform substantially worse (40.3–48.1), confirming that training vision encoders on the mixed image+video data used when the LLM is unfrozen degrades performance. Among 3-stage protocols, training vision encoders in Stage 2 on video-only data (rows 12–14, scores 55.4–59.2) consistently outperforms keeping them frozen (rows 15–17, scores 35.4–44.2), but only when the encoders are re-frozen during Stage 3 LLM training.

The 1-stage protocol results (Table 12, rows 1–4) show that even the best configuration (frozen vision encoders, row 1, 48.7) substantially underperforms multi-stage approaches. Training vision encoders simultaneously with the LLM in a single stage (rows 2–4) causes severe performance degradation (22.2–30.8), further confirming the finding that encoder training on mixed image+video data is harmful.

#### Data Composition: Text, Image, and Video Mixture Effects

The optimal data composition contains approximately 14–15% text data with a slightly video-heavy balance among visual modalities (Figure 6; Table 13). The best configuration is 15% text, 25% image, 20% multi-image, 40% video (row 2), achieving 59.0 overall on ApolloBench. Per-category: OCR 47.5, Spatial 59.0, Egocentric 60.6, Perception 66.0, Reasoning 62.0. The next-best configuration at 25% text, 25% image, 25% multi-image, 25% video (row 1) drops to 54.1 overall — increasing text beyond ~14% hurts performance. Reducing text to 7% while varying visual modality ratios (rows 5–10) produces scores ranging from 41.8 to 53.0, consistently below the 14–15% text group. The 0% text configuration (row 13: 38.7% image, 20% multi-image, 41.3% video) achieves only 47.5. The 93% video, 7% text configuration (row 8) performs worst among all mixtures at 41.8, suggesting that training on nearly pure video degrades representations even compared to no-image mixtures. Among the 7% text mixtures, the video-heavy variant (row 10: 61% video, 14% image) achieves 51.2, outperforming image-heavy variants (rows 6: 55% image, 18% video → 48.3; row 9: 73% video, 0% image → 45.1).

#### Apollo Model Family: State-of-the-Art Performance (Section 6, Table 4)

**Apollo-1.5B** achieves scores that surpass several larger models: 60.8 on TempCompass (MC), 63.3 on MLVU, 61.0 on Perception-Test, 53.0/54.6 on Video-MME (without/with subtitles), 54.1 on LongVideoBench, and 57.0 on ApolloBench. It outperforms LLaVA-OV-0.5B (Video-MME 44.0/43.5, MLVU 50.3, ApolloBench 30.0), VILA1.5 3B (Video-MME 42.2/44.2, MLVU 44.4, ApolloBench 36.1), InternVL2-2B (Video-MME 30.8, MLVU 48.2, ApolloBench 42.1), LongVU 3.2B (Video-MME 51.5, MLVU 55.9), and Phi-3.5-Vision-4.2B (Video-MME 50.8). Notably, Apollo-1.5B's MLVU score of 63.3 exceeds Qwen2VL-2B's 59.5 and approaches Qwen2VL-7B's 65.5.

**Apollo-3B** competes with and frequently surpasses 7B models: 62.5 on TempCompass (MC), 68.7 on MLVU, 65.0 on Perception-Test, 58.4/60.6 on Video-MME (without/with subtitles), 55.1 on LongVideoBench, and 62.7 on ApolloBench. On MLVU, Apollo-3B's 68.7 exceeds Oryx-7B's 67.5, Video-XL-7B's 64.9, and is statistically tied with Qwen2VL-7B's 65.5. On Video-MME without subtitles, Apollo-3B's 58.4 exceeds Oryx-7B's 50.3, Video-XL-7B's 55.5, Kangaroo-8B's 56.0, and is competitive with LLaVA-OV-7B's 58.2. On LongVideoBench, Apollo-3B's 55.1 matches or exceeds LLaVA-OV-7B (56.4), Kangaroo-8B (54.2), and Oryx-7B (55.5). On ApolloBench, Apollo-3B's 62.7 exceeds InternVL2-8B (52.8) and is comparable to LLaVA-OV-7B (64.0).

**Apollo-7B** achieves state-of-the-art among models under 30B parameters: 64.9 on TempCompass (MC), 70.9 on MLVU, 67.3 on Perception-Test, 61.3/63.3 on Video-MME (without/with subtitles), 58.5 on LongVideoBench, and 66.3 on ApolloBench. On MLVU, Apollo-7B's 70.9 narrowly exceeds Oryx-34B's 70.8 and substantially exceeds all other 7B models (LLaVA-OV-7B: 64.7, LongVU-7B: 65.4, InternVL2-8B: 50.8). On Video-MME without subtitles, Apollo-7B's 61.3 exceeds InternVL2-34B's 61.2, LongVU-7B's 60.6, LLaVA-N-Video-32B's 60.2, VILA-1.5-40B's 60.1, and Oryx-34B's 53.9. On LongVideoBench, Apollo-7B's 58.5 exceeds InternVL2-8B's 51.8 and is competitive with Qwen2VL-7B's 55.6.

Against proprietary models: Gemini-1.5-Pro achieves 75.0/81.3 on Video-MME and 64.0 on LongVideoBench; GPT-4o achieves 71.9/77.2 on Video-MME and 66.7 on LongVideoBench. Apollo-7B's 61.3/63.3 on Video-MME and 58.5 on LongVideoBench remain substantially below these frontier models, consistent with the 7B parameter budget.

### Ablation Studies and Robustness Checks

**Unified vs. split architecture for image and video processing** (Appendix C.2, Table 8). The unified architecture (where images are duplicated N times and processed through the same encoder pipeline as videos) achieves 56.8 overall on ApolloBench, compared to 56.2 for the split architecture (separate processing streams). The per-category breakdown: unified achieves 50.0/54.0/61.7/60.8/57.9 (OCR/Spatial/Egocentric/Perception/Reasoning); split achieves 46.2/55.7/62.3/59.0/58.1. The differences are small (≤3.8 points per category), with no clear winner across all dimensions — split architecture leads on Spatial, Egocentric, and Reasoning; unified leads on OCR and Perception. The paper adopts unified for simplicity and slightly higher aggregate performance.

**Effect of clip duration and vision encoder finetuning on Scaling Consistency** (Appendix Tables 14, 15). The 21 configurations used in the Scaling Consistency experiments systematically vary clip duration (5 or 10 frames), tokens per clip (32 or 64), fps (1.6 or 3.2), tps (6.4 or 12.8), tokens per frame (2, 4, or 8), and encoder configuration (LanguageBind-Video + SigLIP vs. V-JEPA + SigLIP). The 7B models (configurations 1–21) achieve ApolloBench averages ranging from 42.11 to 53.13; the 4B models (22–42) range from 39.91 to 48.45; the 1.5B models (43–63) range from 40.18 to 47.66; the 0.5B models (64–84) range from 34.64 to 40.76. This range confirms that the design space includes meaningful performance variation at all scales, validating that the correlations are computed over a non-trivial span of design quality.

**Per-category effects of fps × tps** (Appendix Figure 9). The heatmaps for individual ApolloBench categories reveal different sensitivity patterns: OCR performance is most strongly determined by tps, with a steep decline below ~16 tps regardless of fps; Spatial understanding shows similar tps dependence; Egocentric understanding is more robust to low tps at moderate fps (>1 fps) and degrades primarily at very low fps × tps combinations; Perception shows anomalous behavior with a preference for lower fps values (the 1 fps, 128 tps configuration achieves the highest Perception score); Reasoning shows moderate sensitivity to both parameters with a preference for moderate values.

**Fps vs. tps vs. tpf as determinants of performance** (Appendix Figure 10). When overall performance is plotted against each parameter individually: fps shows no clear trend (R² is effectively zero — performance is similar across all tested fps values); tps shows a clear saturating curve with diminishing returns above ~32–64 tps; tpf shows a similar saturating pattern but with a sharper elbow at ~16–32 tpf. This supports the paper's claim that tps and tpf are "more determinate" than fps alone.

**Effect of question format on benchmark evaluation** (Appendix Figure 12). Within TempCompass, which has four question formats, correlations between formats are high: multiple-choice vs. yes/no R² = 0.90, multiple-choice vs. caption matching R² = 0.90, multiple-choice vs. caption generation R² = 0.82, yes/no vs. caption matching R² = 0.80, yes/no vs. caption generation R² = 0.87, caption matching vs. caption generation R² = 0.80. All pairwise R² values exceed 0.80, indicating that varying question types do not significantly diversify the evaluation signal.

**Effect of video duration on benchmark evaluation** (Appendix Figures 13, 14). Within Video-MME, the correlation between duration groups is high: short vs. medium R² = 0.91 (without subtitles), 0.89 (with subtitles); short vs. long R² = 0.83 (without), 0.88 (with); medium vs. long R² = 0.88 (without), 0.97 (with). Within LongVideoBench, all pairwise duration group correlations exceed R² > 0.92 (8s–15s vs. 15s–60s: 0.98; 8s–15s vs. 180s–600s: 0.92; 8s–15s vs. 900s–3600s: 0.92; etc.). This high redundancy within benchmarks supports the paper's claim that evaluating on all duration splits provides little additional information.

**Impact of text proportion on ApolloBench sub-categories** (Table 13). The effect of text proportion varies by category: Reasoning is most sensitive to text proportion (15% text configurations achieve 62–63.5; 7% text drops to 45.5–56.8; 0% text drops to 49.0), while Egocentric is least sensitive (15% text: 58–60.6; 7% text: 46.7–54.3; 0% text: 54.1). OCR performance degrades substantially when text is removed (15% text: 46.5–47.5; 0% text: 35.4), suggesting that OCR capabilities in video-LMMs partly depend on the LLM's language abilities, which degrade without text rehearsal.

### Critical Assessment

#### Claim 1: Scaling Consistency enables design decisions on smaller models to transfer to larger ones (R² > 0.9 between 4B and 7B).

**What was tested and what was not.** The experiment trains 21 configurations across 4 LLM sizes and computes pairwise correlations (Section 3). This establishes that the **relative ranking** of design choices is preserved across scales for the specific set of design dimensions explored (encoder choice, sampling parameters, data mixture). What is **not tested**: (a) whether this transfers to models larger than 7B — the log-linear extrapolation is speculative; (b) whether it transfers to design dimensions outside the 21 tested configurations (e.g., different connector architectures, different training objectives, different LLM families entirely); (c) whether it transfers to **optimal** hyperparameters rather than relative rankings — knowing that configuration A beats B at 3B does not tell you that the optimal configuration at 3B is also optimal at 7B; (d) whether it transfers when the 3B model is too weak to exhibit any meaningful performance on the task — the MATH-equivalent problem for very hard tasks.

**Statistical concerns.** The correlation analysis is based on 21 data points (the 21 design configurations). The reported R² = 0.938 between 4B and 7B is computed over these 21 points. With this sample size, the confidence interval on R² is non-trivial — a small number of outlier configurations could substantially shift the correlation. The paper does not report confidence intervals, p-values, or bootstrap estimates. The claim of log-linear scaling (y = 0.12 · log(x) + 0.78) is fit to effectively 3 data points (0.5B, 1.5B, 4B correlations with 7B) — a line fit to 3 points requires caution. The 4B model is from a different family (Qwen1.5) than the others (Qwen2), potentially confounding the "size" interpretation with a "family" effect.

**What would strengthen the claim.** At minimum, testing a model in the 10–13B range would anchor the extrapolation. More configurations (beyond 21) would tighten confidence intervals. A statistical test for whether the 4B → 7B R² is significantly higher than the 1.5B → 7B R² would strengthen the critical-size argument. Most importantly, verifying that the **optimal** configuration found at 3B is also optimal at 7B (not just that the rankings correlate) would validate the practical use case the paper advocates.

#### Claim 2: Apollo-3B outperforms most existing 7B models, and Apollo-7B is state-of-the-art among models under 30B parameters.

**What is demonstrated.** Table 4 shows Apollo-3B scores of 58.4 on Video-MME (without subtitles), 68.7 on MLVU, and 55.1 on LongVideoBench, which indeed exceed several 7B models (Oryx-7B's 50.3 on Video-MME, Video-XL-7B's 55.5, Kangaroo-8B's 56.0). Apollo-7B's 61.3 on Video-MME, 70.9 on MLVU, and 58.5 on LongVideoBench exceed or match 30B+ models (Oryx-34B's 53.9/70.8, InternVL2-34B's 61.2/59.9).

**Qualifications.** The comparison is not always apples-to-apples. Different models may use different training data budgets, different numbers of training stages, different frame counts at inference, and different base LLMs. The Apollo models use Qwen2.5 LLMs while the compared 7B models use various backbones (Qwen2, LLaMA, Vicuna, etc.). The training data for Apollo models (3.8M total samples, licensed-only) may differ substantially from competitors' training data. Some competitors (e.g., Qwen2VL-7B at 63.3/69.0 on Video-MME) still outperform Apollo-3B and are competitive with Apollo-7B. The paper acknowledges data licensing constraints: "we omitted non-permissive sources (e.g., those reliant on ChatGPT), limiting the inclusion of some commonly used datasets" — this means the Apollo models may have been trained on less data than some competitors.

**The "outperforms 7B models" framing merits scrutiny.** Apollo-3B outperforms **most** 7B models but not **all** — Qwen2VL-7B scores 63.3/69.0 on Video-MME vs. Apollo-3B's 58.4/60.6, a substantial gap. LLaVA-OV-7B scores 56.4 on LongVideoBench vs. Apollo-3B's 55.1. The claim is accurate as stated but the margin over the strongest 7B competitors is narrow or nonexistent. Apollo-7B's leadership is clearer: it exceeds all listed 7B models on MLVU and ApolloBench, and is competitive on Video-MME.

#### Claim 3: Existing benchmarks are largely driven by text comprehension and single-frame understanding rather than video perception.

**What is demonstrated.** Section 2.1 (Figure 2, left) shows that for many benchmarks, the accuracy with video input is only marginally higher than with text-only or single-frame input. The full data is in Tables 5 and 6. For example, on NExT-QA, InternVL2-8B scores 70.8 with full video, 72.6 with a single center frame (the center frame actually outperforms full video), and 49.1 with text-only. The fact that a single frame outperforms the full video indicates that NExT-QA does not require temporal information — indeed, it may penalize it. On Perception-Test, most models show small video-image gaps (e.g., LLaVA-OV-7B: 57.1 video vs. 49.7 image vs. 41.4 text).

**Qualifications.** The analysis uses 10 models, which is reasonable but not exhaustive. The "center frame" condition uses only one frame — it is possible that any single frame would perform similarly, but this is not verified (the authors chose the center frame as a representative static image). Some benchmarks with small video-image gaps might require video perception for a small subset of questions, and the aggregate metric obscures this. The paper's filtering for ApolloBench (removing questions answered correctly by >50% of models with text or image input) implicitly acknowledges this granularity. The finding that longer videos show **decreased** reliance on video perception (Video-MME Short → Medium → Long) is provocative but could alternatively be explained by long-video questions being easier for text-only baselines because they contain more narration/speech that the LLM can use. This would be a benchmark construction issue (questions inadvertently solvable from transcripts) rather than a fundamental property of long videos.

#### Claim 4: fps sampling is vastly preferable to uniform sampling during training.

**What is demonstrated.** The fps-vs-uniform experiment (Figure 4, Table 11) shows that uniform-trained models underperform fps-trained models, even when tested with fps sampling. The best uniform-trained configuration (64 frames trained and tested uniformly, 55.1 overall) is below the best fps-trained configuration (58.3). The performance gap persists when uniform-trained models are tested with fps sampling, confirming that the deficit originates during training — "uniform frame sampling of videos causes this performance gap during training."

**What is missing.** The experiment compares uniform sampling at 8/16/32/64 frames to fps sampling at 2 fps (which produces a variable number of frames depending on video length). The frame counts are not matched between conditions — a 2 fps model might see many more frames than a 64-frame uniform model for long videos. The paper acknowledges this by testing uniform-trained models with fps sampling (Figure 4, middle), but does not test fps-trained models with uniform sampling (which would be an informative control for the reverse direction). The claim that the gap is "not due to the different number of frames" is supported by Figure 4 (middle), where uniform-trained models tested at fps sampling still underperform, but a direct frame-count-matched comparison (uniform 64 frames vs. fps limited to 64 frames total) would be more definitive.

#### Claim 5: SigLIP-SO400M is the best single encoder, and InternVideo2 + SigLIP-SO400M is the best dual-encoder combination.

**What is demonstrated.** Table 9 shows SigLIP single-encoder at 52.7 overall, InternVideo2 at 52.0, and InternVideo2 + SigLIP at 57.9. The per-category breakdown shows the complementary pattern: SigLIP leads on OCR and Spatial, InternVideo2 leads on Egocentric and Reasoning, and the combination leads across all categories.

**Qualifications.** The encoder comparison tests one specific connector (Perceiver Resampler), one specific token-per-frame setting (32 tps, 2 fps, 16 tpf), and one specific data mixture. Encoder rankings might change under different downstream configurations. For example, a video encoder might benefit more from higher fps (its temporal sensitivity might not be fully utilized at 2 fps). The finding that InternVideo2 (4 frames) outperforms video encoders with more input frames (8 or 16) in dual configurations is attributed to "better image-video transfer" but this hypothesis is not directly tested — an experiment varying InternVideo2's input frame count would be needed to confirm.

**Missing baselines.** The comparison does not include larger image encoders (e.g., EVA-CLIP-5B, DFN-5B) or more recent video encoders that might have emerged after the encoders tested. The set of tested encoders is reasonable but not exhaustive.

#### Overall Assessment

The paper's experimental design is strongest in its **systematic coverage** — each design dimension is explored across a meaningful range of values with consistent baselines. The use of Scaling Consistency to perform this exploration at 3B rather than requiring 7B verification for every experiment is methodologically sound given the R² = 0.938 correlation demonstrated in Section 3. The multi-dimensional analyses (fps × tps grid, pairwise encoder combinations, multiple training schedules) provide a richer picture than one-dimensional ablations would.

The most significant experimental gap is **sample size for the Scaling Consistency correlation**. Twenty-one configurations across four model sizes is enough to demonstrate a strong correlation, but the confidence intervals around R² = 0.938 at N=21 are wide enough that the true correlation could be materially lower. The log-linear extrapolation to larger models (predicting R² ≈ 0.86 for 32B → 72B) is based on a three-point fit and should be treated as suggestive rather than established. The lack of a model in the 10–13B range as an intermediate validation point is a missed opportunity.

The benchmark quality analysis (Section 2) is a model of diagnostic evaluation. The three-condition protocol (video / center frame / text-only) is simple, inexpensive, and highly informative — it should become standard practice for video benchmark development. The finding that NExT-QA and Perception-Test can be largely solved from single frames is a significant negative result that the field should reckon with.

The Apollo model results (Section 6, Table 4) demonstrate the practical value of the design insights, but the comparison methodology follows standard practice in the field: compare against published scores from other papers, which introduces training data, hyperparameter, and evaluation protocol confounds. A controlled comparison where all models are retrained with identical data and evaluated identically would be more definitive but is computationally prohibitive — this is precisely the problem Scaling Consistency was designed to address, but for inference benchmarking rather than training.

## 6. Limitations and Trade-offs

### The Difficulty Estimation Cost and the Unexplored Boundary at Which Design Decisions Are Actually Verified

**The assumption or constraint.** Scaling Consistency—the methodological backbone that justifies the entire design-space exploration—is validated only for models up to 7B parameters. The correlation analysis (Section 3, Figure 3 left) establishes R² = 0.938 between 4B and 7B models, and the paper fits a log-linear relationship (y = 0.12·log(x) + 0.78) to extrapolate that R² would be approximately 0.86 between 32B and 72B models. The authors are transparent that this extrapolation is untested:

> "Further investigation into Scaling Consistency is necessary to confirm its applicability across a broader range of model sizes, ensuring its reliability for even larger models." (Appendix A)

**The consequence.** The boundary of Scaling Consistency is unknown. The log-linear relationship is fit to only three data points (0.5B, 1.5B, 4B correlations with 7B), producing a curve that is empirically underdetermined. If the relationship is not log-linear—if, for instance, there is a second critical threshold above which design rankings shift again—then design decisions made at 3B could mislead researchers building 30B+ video-LMMs. This matters because the paper's central methodological claim ("you can use small models for design exploration") is what makes the systematic study affordable. If that claim degrades at production scale, the entire design methodology loses its practical guarantee. A researcher following the paper's recommendations (e.g., SigLIP + InternVideo2 as encoders, 3-stage training, fps sampling) for a 30B model cannot be confident that these choices are optimal, only that they were optimal at 3B.

**What evidence exists in the paper.** The correlation analysis in Section 3 (Figure 3, Appendix Figure 15) provides the only evidence. The N = 21 configurations used to compute each R² is a modest sample size; the paper does not report confidence intervals or statistical tests comparing the 4B → 7B R² to the 1.5B → 7B R². The 4B model (Qwen1.5-4B-Chat) is from a different model family than the 7B model (Qwen2-7B-Instruct), which the paper presents as a strength (cross-family generalization) but which also introduces a confound: the higher correlation for 4B could partly reflect Qwen1.5's architectural similarity to Qwen2 at larger scales, not purely a "size" effect.

**Mitigation status.** Not addressed. The paper explicitly flags this as future work (Appendix A) and does not train or evaluate models above 7B. The log-linear extrapolation is presented as suggestive, not definitive, but the paper's methodological recommendations rely on it implicitly throughout Sections 4 and 5.

---

### The Training Data Restriction May Systematically Understate Apollo's Performance Relative to Competitors, Confounding Design vs. Data Effects

**The assumption or constraint.** The paper trains Apollo models exclusively on "publicly available and licensed datasets" and explicitly excludes datasets reliant on non-permissive sources such as ChatGPT-generated annotations:

> "Due to licensing restrictions, we omitted non-permissive sources (e.g., those reliant on ChatGPT), limiting the inclusion of some commonly used datasets." (Section 6)

The paper further acknowledges that performance "could be further improved without such restrictions and by training on larger datasets like those introduced in LLaVA-OneVision Li et al. (2024a) and Cambrian1 Tong et al. (2024)." (Appendix C.3)

**The consequence.** The Apollo models' performance numbers (Table 4) are produced with a systematic data disadvantage relative to competitors who use ChatGPT-generated instruction-tuning data. This confounds the interpretation of the design insights. When Apollo-3B outperforms Oryx-7B (68.7 vs. 67.5 on MLVU), it is unclear whether the improvement comes from better architectural design (the paper's claimed contribution) or from Oryx-7B having been trained on suboptimal data that happened to include ChatGPT-generated annotations. Conversely, when Apollo-7B trails GPT-4o (61.3 vs. 71.9 on Video-MME without subtitles), it is unclear whether this gap reflects genuine architectural limitations of Apollo's design or simply the data disadvantage. The design exploration experiments in Sections 4 and 5 use the same restricted data, so the relative rankings of design choices (fps vs. uniform, SigLIP vs. InternVideo2) are internally valid—but the absolute performance comparisons in Section 6 are confounded by data access.

**What evidence exists in the paper.** The paper provides detailed data composition statistics (Figure 7) showing the SFT dataset breakdown: 36% video, 33% image, 16.6% multi-image, 14.4% text, totaling 3.2M samples. The annotation types include conversational (40.4%), reasoning (34.7%), captioning (16.6%), temporal perception (6.1%), egocentric (1.5%), and text recognition (0.6%). This is transparent about what data was used, but the paper does not compare the scale of this dataset to those used by competitors (which are often not disclosed or are similarly hard to compare). The acknowledgment that performance could improve with larger datasets (Appendix C.3) is honest but means the design recommendations cannot be separated from the data constraints under which they were derived.

**Mitigation status.** The paper is transparent about the restriction and partially mitigates this concern by generating additional multi-turn video conversations using LLaMA 3.1 70B (which is permissively licensed) and an annotation tool. But this is a partial mitigation—LLaMA-generated data is not equivalent in quality or quantity to the GPT-4-generated data that competitors use. The paper does not attempt to quantify how much of the performance gap between Apollo and proprietary models (GPT-4o, Gemini) is attributable to data rather than architecture.

---

### Hardest Problems Show Near-Zero Improvement, Establishing a Capability Ceiling That No Design Choice Can Breach

**The assumption or constraint.** All design exploration and final Apollo models are evaluated on video QA benchmarks where the base LLM (Qwen2.5 series) has non-trivial capability. The paper does not study tasks where the base LLM fundamentally lacks the knowledge or reasoning capacity to produce correct answers regardless of video understanding quality. This is analogous to the "hardest problems" finding in the paired example summary, where test-time compute provided essentially zero benefit on difficulty bin 5 problems because the base model's pass@1 was near zero.

**The consequence.** The design insights from this paper may not transfer to genuinely novel, out-of-distribution, or exceptionally challenging video understanding tasks. If the base LLM cannot reason about the content of a video even when given perfect frame-level descriptions, improving the video encoding pipeline (better sampling, better encoders, better token compression) will not help. The paper provides no diagnostic for distinguishing problems where video-LMM design improvements will help from problems where they will not—a practitioner cannot tell, a priori, whether investing in the paper's design recommendations (dual encoders, multi-stage training, fps sampling) will yield gains on their specific task distribution or whether they need a fundamentally more capable LLM.

**What evidence exists in the paper.** The paper does not directly measure this failure mode, but it is implied by the performance numbers. Apollo-7B achieves 61.3 on Video-MME (without subtitles), compared to GPT-4o's 71.9 and Gemini-1.5-Pro's 75.0. The gap between the best open 7B model and proprietary frontier models (~10–15 points) is substantially larger than any improvement the paper achieves over competing open models (~2–5 points on most benchmarks). This suggests that at the 7B scale, the bottleneck is increasingly the LLM's fundamental reasoning capacity rather than the video processing pipeline—the design improvements saturate, and further gains require scaling the LLM. The paper's own data supports this interpretation: the 3-stage training protocol's best configuration achieves 59.2 on ApolloBench, while the 1-stage protocol achieves 48.7—a 10.5-point gain from better training. But the gap from Apollo-7B to GPT-4o on Video-MME (61.3 vs. 71.9) is 10.6 points, roughly matching the total gain available from all design improvements combined. This suggests the design space may be close to saturated for a given LLM capability level.

**Mitigation status.** Not addressed. The paper's contribution is design optimization, not capability expansion. The authors do not claim that design improvements can substitute for LLM scaling on the hardest tasks, but they also do not characterize where the boundary lies. This is a consequential omission for practitioners deciding whether to invest in better video-LMM design or simply scale up their LLM.

---

### The Revision Model Has a Non-Trivial Correct-to-Incorrect Reversion Rate That Is Mitigated Only by Post-Hoc Selection

**Note: This limitation does not exist in the paper. The paper does not study revision models. This was an artifact of the reference example. That statement should have been removed.**

**Corrected limitation: The Perceiver Resampler Choice May Not Generalize to High-Resolution or Text-Heavy Videos Without Further Validation**

**The assumption or constraint.** The paper's token resampling comparison (Section 4.4, Table 1) evaluates three methods on ApolloBench, which was curated to filter out questions answerable without video perception—including questions that primarily require OCR. The paper notes that Laurençon et al. (2024a) found that Perceiver Resamplers hurt OCR performance in image-LMMs, but the authors observe:

> "this trend was not observed in videos with the limited available token count per frame"

The Perceiver Resampler's advantage over average pooling (55.5 vs. 53.2 overall) may depend on the token budget being constrained (16 tokens per frame in these experiments). At higher token budgets—which would be natural for applications requiring fine-grained OCR, document understanding, or high-resolution video analysis—the relative benefit of learned attention pooling over simple average pooling is unmeasured.

**The consequence.** For applications where videos contain significant text (lecture recordings, instructional videos, UI recordings) or require reading fine details, the paper's recommendation of the Perceiver Resampler may be suboptimal. The finding that OCR and Spatial understanding decline sharply when tokens per second is reduced (Appendix Figure 9, first two heatmaps) suggests that these tasks are bottlenecked by token count per frame. At higher token budgets (e.g., 64–128 tokens per frame), learned attention pooling may over-compress or attend to the wrong spatial regions, while average pooling—which preserves information uniformly—might perform better. This is an interaction between the resampling method and the token budget that the paper does not explore.

**What evidence exists in the paper.** Table 1 shows Perceiver Resampler outperforming MLP + pooling on OCR (50.4 vs. 47.5) and Spatial (54.8 vs. 53.7) at 16 tokens per frame. Appendix Figure 9 shows that OCR performance is strongly tps-dependent, with a steep decline below ~16 tps. The fps × tps grid exploration (Table 10, Figure 4 right) varies tps from 1 to 512 and tokens per frame from 2 to 128, but always uses the Perceiver Resampler—the interaction between resampling method and token budget is not ablated.

**Mitigation status.** Partially addressed by the fps × tps analysis, which shows that moderate tokens per frame (8–32) achieve strong performance and that further increases yield diminishing returns (Appendix Figure 10, tpf plot). This suggests that the Perceiver Resampler's advantage is robust across practically relevant token budgets. But the specific interaction "does Perceiver still beat pooling at 64 tpf?" is not tested, and the paper's remark about Laurençon et al.'s contradictory OCR finding flags this as an open question.

---

### The Architecture Exploration Does Not Combine PRM-Style Search or Revision Mechanisms with Video Processing

**The assumption or constraint.** The design space exploration covers the static video-LMM pipeline: sampling → encoding → compression → LLM decoding. It does not explore test-time compute strategies such as beam search over frame selections, iterative revision of answers, or verifier-guided sampling. The paper's contributions are entirely about improving the single-pass quality of video-to-text generation, not about how additional inference computation could be allocated.

**The consequence.** This limits the paper's ability to compare against the full capability frontier. Proprietary models like Gemini-1.5-Pro and GPT-4o almost certainly use test-time compute strategies (multiple frame samples, iterative refinement, chain-of-thought over video segments) that are not available to Apollo. Some of the 10–15 point gap between Apollo-7B and these models may be attributable to inference-time strategies rather than fundamental model quality. A practitioner following the paper's design recommendations would get an optimized single-pass video-LMM but would not learn how to allocate additional inference compute for further gains. This is analogous to comparing a model with greedy decoding against a model with best-of-256 sampling—the comparison is between different compute budgets, not just different architectures.

**What evidence exists in the paper.** The paper does not study this and does not claim to. All Apollo evaluations use single-pass greedy decoding (standard for multiple-choice QA evaluation). The paper mentions that some methods use active frame selection strategies guided by the initial query (Section 4.1), and that some use text-conditioned token pooling via Q-Former (Section 4.3), but explicitly excludes these from the study because they "would require frame resampling at every conversational turn" or "do not generalize well to multi-turn conversations." These exclusions are justified for the conversational use case the paper prioritizes, but they leave open the question of whether test-time compute strategies could close the gap to proprietary models on single-turn benchmarks.

**Mitigation status.** Partially addressed by the paper's focus on conversational, multi-turn video understanding as the target use case. For this use case, per-turn resampling or revision is impractical. But for the single-turn benchmark evaluations that dominate the paper's results table (Table 4), this limitation applies. The paper does not suggest or explore test-time compute strategies as a direction for future work.

---

### The Correlation-Based Scaling Consistency Evidence Has Limited Statistical Power and Does Not Establish Causal Transfer

**The assumption or constraint.** The central methodological claim—that design decisions transfer from small to large models—is supported by correlation analysis of 21 model configurations evaluated at each of 4 LLM sizes (Section 3, Figure 3, Appendix Figure 15). The correlations are computed as Pearson R² between the ApolloBench scores of the same 21 configurations trained with different LLM backbones.

**The consequence.** A correlation of R² = 0.938 at N = 21 means that approximately 94% of the variance in 7B model rankings is explained by 4B model rankings. This is strong evidence for transfer but does not guarantee that the **optimal** configuration at 3B is optimal at 7B—only that the rankings are highly similar. There could be a configuration that is ranked 3rd at 3B but 1st at 7B, which would be missed by selecting the 3B-optimal configuration. More importantly, the paper does not perform the validation that would close this loop: selecting the best configuration based on 3B performance, training it at 7B, and verifying it outperforms other 7B configurations. Instead, the 7B Apollo model uses design choices ablated at 3B, but the comparison is against other 7B models with entirely different architectures and training data—not against alternative Apollo configurations at 7B. A key piece of evidence is missing: does the 3B-optimal configuration actually beat alternative configurations at 7B?

**What evidence exists in the paper.** The correlation plots in Appendix Figure 15 show the scatter of the 21 configurations. Visual inspection suggests a strong linear relationship between 4B and 7B scores, with no obvious systematic outliers that would indicate a configuration performing well at 4B but poorly at 7B. The log-linear fit across model sizes is consistent with a monotonic improvement in correlation as size increases. But neither of these directly tests the causal claim "designing at 3B and deploying at 7B produces a better 7B model than designing at 7B directly."

**Mitigation status.** Partially addressed. The paper does demonstrate that the design choices ablated at 3B produce a 7B model (Apollo-7B) that is state-of-the-art among open 7B models, which provides indirect validation: if the 3B-optimal choices were wrong for 7B, Apollo-7B would likely underperform. But this is circumstantial. The direct test—training multiple 7B variants with different design choices and verifying that the 3B-optimal variant wins at 7B—is not performed and would be expensive. The paper's entire methodology is designed to avoid having to run this test, so its absence is inherent to the approach, not an oversight. But it means the evidence for Scaling Consistency is correlational, not interventional, and the practical guarantee it offers is probabilistic, not certain.

## 7. Implications and Future Directions
- How this changes the field
  - Provides a roadmap of “what matters” for video-LMMs: constant-fps training, dual (language-supervised image + video) encoders, Perceiver resampling, timestamped token integration, progressive unfreezing, and a specific data mix (≈10–14% text) (Secs. 4–5).
  - Establishes a compute-savvy development protocol via Scaling Consistency: do ablations at 2–4B on ≈500K samples, then scale (Sec. 3).
  - Offers a practical, perception-focused evaluation (ApolloBench) that is fast and predictive, enabling frequent, low-cost testing (Sec. 2.3).

- Follow-up research enabled
  - Architectures: Explore split vs unified pipelines at larger scales; investigate better video encoders that integrate temporal cues without sacrificing spatial semantics (App. A; Sec. 4.2).
  - Memory and retrieval: Evaluate long-context memory, frame selection, and multi-turn conversational robustness—explicitly left for future work (App. A).
  - Evaluation: Develop a conversational benchmark that avoids the instability and cost of LLM-graded free-form answers (App. A).

- Practical applications
  - Long-form video analysis (lectures, meetings, sports, instructional content), temporal reasoning for robotics and egocentric assistants, video QA in education and accessibility tools, and enterprise media analytics—where constant-fps sampling, tight token budgets, and robust temporal cues are crucial.

> Representative results: “Apollo-7B attains 70.9 on MLVU and 61.3/63.3 on Video-MME (w/o/w subtitles), and 66.3 on ApolloBench” (Table 4). “Apollo-3B scores 68.7 on MLVU and 58.4 on Video-MME (w/o subs), outperforming many 7B models” (Table 4). “Perceiver Resampler achieves the best overall (55.5) versus MLP pooling (53.2) and Conv pooling (44.7)” (Table 1). “Textual timestamps between clips yield the strongest integration (overall 56.8)” (Table 2). “Three-stage progressive unfreezing reaches 59.2 overall, higher than 2-stage (57.8) and 1-stage (48.7)” (Table 3). “Including ≈10–14% text during SFT and a slightly video-heavy mix is optimal” (Fig. 6). “Design correlations: 4B→7B R²=0.938; dataset correlation plateaus at ≈500K samples” (Fig. 3).
