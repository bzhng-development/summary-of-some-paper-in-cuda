# Autoregressive Universal Video Segmentation Model

**ArXiv:** [2508.19242](https://arxiv.org/abs/2508.19242)

## 🎯 Pitch

This paper introduces AUSM, a single autoregressive architecture that unifies both prompted (user-specified) and unprompted (automatic discovery) video segmentation by treating mask prediction as a sequential, language-model-like process. Its scalable state-space design enables efficient segmentation and tracking of objects in arbitrarily long video streams with constant memory, while supporting parallel training that accelerates learning up to 2.5×. By consolidating the fragmented landscape of task-specific models, AUSM sets a new foundation for practical, general-purpose video perception across interactive and autonomous scenarios.

---

## 1. Executive Summary

This paper introduces the **Autoregressive Universal Segmentation Model (AUSM)**, a single architecture that unifies prompted and unprompted video segmentation by reformulating streaming video segmentation as sequential mask prediction—analogous to language modeling. AUSM employs a **History Marker** (dissolving instance masks into spatial feature maps to preserve fine-grained information) and a **History Compressor** (using Mamba-based state-space layers to compress all past spatio-temporal information into a single fixed-size spatial state, eliminating explicit memory buffers) to process video streams of arbitrary length with constant memory. Across seven benchmarks spanning both video object segmentation and video instance segmentation (DAVIS 2017, YouTube-VOS 2018 & 2019, MOSE, YouTube-VIS 2019 & 2021, OVIS), AUSM outperforms prior universal streaming models—exceeding UniVS on YouTube-VOS 2018 by 8.7 points in 𝒢 despite using a smaller backbone—while achieving up to 2.5× faster training on 16-frame sequences through its parallel training formulation, establishing that autoregressive modeling with compressed spatial states unifies segmentation tasks effectively only when trained with teacher-forcing across frames rather than recurrent frame-by-frame propagation.

## 2. Context and Motivation

### The Fragmentation Problem in Video Segmentation

Video segmentation is not one task but a family of related yet architecturally distinct problems. At a high level, the paper identifies two fundamental regimes:

**Prompted video segmentation** (exemplified by Video Object Segmentation, or VOS) gives the model an initial human cue—typically a mask, bounding box, point, or text description in the first frame—and requires it to propagate that specified target through all subsequent frames. This is the setting of interactive editing, where a user marks an object of interest and expects the model to track only that object. The model is not responsible for discovering new objects that appear later; if a novel instance enters the scene mid-video, the system must be re-prompted.

**Unprompted video segmentation** (comprising Video Instance Segmentation, VIS, and Video Panoptic Segmentation, VPS) requires the model to detect, classify, and track *all* objects of predefined categories throughout the entire video—without any human guidance. This is the setting for autonomous perception systems (e.g., self-driving cars, robot manipulation, surveillance), where the system must discover every relevant instance and maintain consistent identity tracking across frames.

These two regimes have historically been served by entirely different model architectures, training protocols, and research communities. A VOS specialist like XMem (Cheng and Schwing, 2022) or SAM2 (Ravi et al., 2025) cannot perform VIS, and a VIS specialist like GenVIS (Heo et al., 2023) or DVIS (Zhang et al., 2023) cannot perform prompted tracking. This fragmentation has real costs:

- **Annotation scarcity**: Video data is expensive to annotate. VOS annotations provide mask propagation for specific objects; VIS annotations provide class labels and instance tracks for all objects. These annotation pools are siloed—a model trained for VIS cannot benefit from VOS data, and vice versa. A unified model could amortize supervision across tasks, learning richer representations from the combined data.
- **Deployment complexity**: Real-world systems often need both capabilities. A video editor might want automatic scene understanding (unprompted) *and* the ability to refine specific objects via prompts. Currently, this requires running multiple models side-by-side or maintaining separate deployment pipelines.
- **Research fragmentation**: Advances in one regime (e.g., memory-based propagation for VOS) rarely transfer to the other, because the architectural assumptions are fundamentally incompatible.

### Why This Problem Matters Now

The paper identifies a structural parallel between video and language that has been under-exploited. Both are sequential modalities that arrive as streams. In language, the decoder-only transformer paradigm—exemplified by GPT models and LLaMA—has demonstrated that a single scalable architecture, trained with an autoregressive objective on massive corpora, can subsume diverse tasks (translation, summarization, question-answering, code generation) without task-specific architectural modifications. The field of video perception, the authors argue, would benefit from the same unification.

The motivation is not merely aesthetic. The paper enumerates four criteria that an ideal universal streaming video segmentation model should satisfy (Section 1):

1. **Accommodate a broad set of tasks**—both prompted and unprompted, without architectural switching.
2. **Preserve fine-grained spatio-temporal details from past inputs**—not just object identities for association, but the actual mask geometry needed for precise propagation.
3. **Support inference over long videos**—real-world videos can be minutes or hours long; the model should not degrade or run out of memory.
4. **Enable training that scales efficiently with sequence length**—as video datasets grow longer, training time should not explode linearly with the number of frames processed per clip.

No existing approach simultaneously satisfies all four. The paper's central claim is that this is not because unification is impossible, but because the field has not properly exploited the autoregressive formulation that language modeling has made routine.

### Prior Approaches and Where They Fall Short

The paper's related work (Section 4) provides a detailed taxonomy of existing methods. I will walk through each category, explaining what it does, why it works for its intended task, and precisely where it breaks down for unification.

---

#### Unprompted Video Segmentation: The Detect-Then-Track Paradigm

Most VIS methods follow a detect-then-track pipeline. The core idea: process each frame independently with an image segmentation model (producing per-frame instance masks and class labels), then associate these detections across frames using some form of temporal matching—often an external memory bank or a heuristic matching algorithm.

**Conventional online tracking-by-detection** (Huang et al., 2022; Wu et al., 2022; Kim et al., 2024; Ying et al., 2023) represents the simplest form of this approach. These models explicitly *exclude* past predictions $\hat{y}_{<t}$ from their forward pass. Each frame $t$ is processed as if it were an independent image; the model produces detections, and a separate post-hoc association step links these detections to previously tracked instances. The advantage is that training is trivially parallelizable—frames can be processed independently. The disadvantage is that the model cannot use its own past predictions to improve current detection or disambiguate occluded instances. Identity association is handled entirely by external heuristics, not learned temporal reasoning.

The paper notes that even approaches that *do* leverage past information (Wu et al., 2022; Ying et al., 2023; Kim et al., 2024) do so in a heavily compressed form—each tracked instance is represented as a single vector or a few vectors, preserving enough information for identity matching but discarding the fine-grained spatial details (exact mask boundaries, partial occlusion patterns) needed for precise mask prediction. This is a critical distinction the paper draws: preserving identity is not the same as preserving mask geometry.

**Query propagation methods** (GenVIS, Heo et al., 2023) attempt to improve temporal coherence by conditioning on prior predictions through object-level vector representations. At each frame, the model's object queries (learned embeddings that "ask" the decoder to produce detections) are updated based on the queries from the previous frame. This provides a learned mechanism for temporal consistency, but the paper identifies a fundamental limitation: *object vectorization significantly degrades the granularity of mask predictions* (Kim et al., 2024). Compressing an entire instance mask into a single $D$-dimensional vector discards the spatial structure—you cannot recover whether an object is partially occluded on the left side or whether its boundary has fine protrusions.

**Instance mask propagation** (RoCoVIS, Heo et al., 2025) addresses this by propagating actual masks rather than just query vectors. This greatly improves mask quality, but introduces a new problem: predictions must be generated sequentially, frame by frame, because each frame's mask propagation depends on the previous frame's output. This *inherently breaks parallelism* during training, leading to reduced training efficiency. The model must process frames recurrently, which is slow and does not scale well with sequence length.

**Offline methods** (Wang et al., 2020; Hwang et al., 2021; Heo et al., 2022) take a different approach: the entire video is available at inference time, and the model can attend across all frames simultaneously. This enables long-range context without sequential propagation. However, the paper points out a crucial subtlety: offline models are *not conditioned on intermediate outputs* $\hat{y}$. They process all frames jointly and produce all predictions at once, but they cannot refine a prediction for frame $t$ based on what was predicted for frame $t-1$. Without this temporal feedback loop, they often underperform recurrent methods that explicitly leverage prior predictions—despite having access to more information.

**Summary of VIS shortcomings for unification:**
- Detect-then-track methods cannot handle prompted settings at all—there is no mechanism to accept an initial mask and track only that object.
- Query propagation methods discard spatial detail, making them unsuitable for the fine mask propagation required in VOS.
- Mask propagation methods break parallel training, making them inefficient.
- Offline methods lose the sequential autoregressive signal that gives recurrent methods their edge.

---

#### Prompted Video Segmentation: The Memory-Based Paradigm

**Space-Time Memory Networks (STM)** (Oh et al., 2019) established the dominant paradigm for prompted VOS. The core idea: maintain an explicit memory bank containing past frames and their corresponding masks. At each new frame, perform dense matching between the current frame features and the stored memory to retrieve relevant information for mask propagation. Think of it as a differentiable nearest-neighbor lookup—the model learns to attend to memory locations that contain similar visual patterns and uses the associated masks to predict the current frame's mask.

This paradigm has been highly successful. XMem (Cheng and Schwing, 2022) enhances the memory mechanism with multiple memory tiers (working memory for recent frames, long-term memory for older frames) inspired by the Atkinson-Shiffrin psychological memory model. SAM2 (Ravi et al., 2025) extends the approach with large-scale training data and more flexible input modalities (points, boxes, masks).

However, memory-based methods have fundamental limitations for unification:

- **Per-object processing**: STM-based methods typically process each tracked object independently, maintaining a separate memory buffer per object. This is feasible for prompted VOS where the user specifies 1–10 objects, but it scales poorly to unprompted settings where dozens of objects may appear. Running an independent forward pass for each object is computationally prohibitive.
- **Explicit memory buffers**: The memory bank stores past frame features and masks, which grows linearly with video length. For long videos, this becomes a memory bottleneck. The paper emphasizes that SAM2 uses FIFO-style spatio-temporal caches—essentially, you must decide what to keep and what to discard when memory fills up, which introduces a form of catastrophic forgetting.
- **No detection capability**: Memory-based methods propagate specified masks; they have no mechanism to discover new objects that appear mid-video. This is inherent to the prompted paradigm but makes them incompatible with unprompted tasks.

**Hierarchical propagation methods** (Yang et al., 2021; Yang and Yang, 2022) attempt to process multiple objects jointly through a hierarchical transformer structure, gradually propagating identity information from past frames to the current frame. This is a step toward unifying multi-object processing, but these methods are still fundamentally designed for the prompted setting—they propagate given masks rather than discovering new instances.

**Summary of VOS shortcomings for unification:**
- Memory-based methods cannot discover new objects.
- Per-object processing is computationally infeasible for unprompted settings.
- Explicit memory buffers are incompatible with arbitrary-length videos.
- These methods have no detection head, no classification head, and no mechanism for handling the "background" class in the VIS sense.

---

#### Universal Video Segmentation: Early Attempts at Bridging the Gap

Recognizing the costs of fragmentation, several recent works have attempted to build universal models that handle both prompted and unprompted segmentation.

**TarViS** (Athar et al., 2023) represents the first attempt to jointly model both settings. It encodes task-specific targets as a set of queries—in the prompted setting, the query encodes the target object; in the unprompted setting, the queries encode "find all objects of category X." The approach is offline (the full video must be available at inference time), which means it cannot be deployed in streaming settings and cannot benefit from autoregressive temporal feedback.

**UNINEXT** (Yan et al., 2023) introduces a prompt-guided object discovery and retrieval paradigm, supporting both prompted and unprompted tasks in a single architecture. It operates online (frame-by-frame) but, as shown in Table 1, its prompted VOS performance is substantially below specialized methods—74.5 $J \& F$ on DAVIS 2017 with a ResNet-50 backbone, compared to 86.2 for XMem with the same backbone. This gap reveals a fundamental tension: the architectural choices that make universal models flexible also seem to sacrifice the fine-grained spatial precision that VOS requires.

**UniVS** (Li et al., 2024) takes a different approach: it leverages prompts as queries by treating predicted masks from previous frames as visual prompts for the current frame. This is more temporally coherent than UNINEXT's approach, but the paper identifies a critical weakness: UniVS encodes instances as heavily compressed tokens, following the VIS tradition of query-based detection (Carion et al., 2020; Cheng et al., 2022). This vectorization is what allows the model to be "universal"—queries can represent either prompted targets or detected objects—but it is precisely what causes the VOS performance drop. The paper reports that UniVS achieves 71.7 $J \& F$ on DAVIS 2017 with Swin-T, compared to AUSM's 76.4, and 75.0 with Swin-B compared to AUSM's 81.6. The nearly 10% relative improvement AUSM achieves (referenced in Section 1 as "nearly 10% improvement in VOS performance compared to previous unified online architectures") comes specifically from replacing this vectorized representation with the History Marker's spatial preservation.

**Critical pattern across universal models:** Every prior universal model achieves universality by adopting the VIS-style object query abstraction—compressing instance information into vectors. This makes detection and classification natural (queries can be matched to ground-truth objects via Hungarian matching) but fundamentally loses the spatial detail that memory-based VOS methods preserve. The paper's core architectural insight is that this tradeoff is *not* inherent to universal models—it is an artifact of a specific design choice (vectorization) that can be replaced with a spatial representation (History Marker) without sacrificing universality.

---

#### Training Efficiency: The Unexploited Parallel

Between the lines of this architectural taxonomy is a training efficiency story that the paper brings to the foreground. All existing video segmentation methods that condition on past predictions—whether query propagation (GenVIS), mask propagation (RoCoVIS), or memory-based approaches (STM, SAM2)—must process frames **recurrently** during training. Frame $t$ cannot be processed until frame $t-1$'s predictions are available, because those predictions are part of the input to frame $t$. This sequential dependency means training time scales linearly with sequence length, and GPU utilization is poor because parallelism across frames is impossible.

In contrast, modern decoder-only LLMs achieve their training efficiency through **teacher forcing**: during training, the ground-truth previous token is fed as input rather than the model's own prediction, allowing all tokens in a sequence to be processed in parallel. The model learns the same conditional distribution $P(y_t | y_{<t})$ but the computation is not serialized.

The paper identifies this as a missed opportunity in video segmentation. While some unprompted methods that exclude past predictions (Huang et al., 2022; Wu et al., 2022) can train in parallel, they sacrifice temporal conditioning. Methods that include temporal conditioning sacrifice parallel training. The paper frames this as a false dichotomy—AUSM demonstrates that with the right architectural design, you can have both temporal conditioning and parallel training.

---

### How AUSM Positions Itself

The paper positions AUSM not as an incremental improvement to either VOS or VIS, but as a **reconceptualization of video segmentation itself** through the lens of autoregressive sequence modeling. The key moves are:

**1. The language modeling analogy is operational, not metaphorical.** The paper doesn't just say "video is like language"; it provides a precise mathematical formulation in Equation 2 that mirrors the autoregressive decomposition of language models (Equation 1). The factorization $P(y_{1:T} | \mathcal{I}_{1:T}) = \prod_{t=1}^T P(y_t | y_0, y_{<t}, \mathcal{I}_{\leq t})$ is the same structural form that enables LLMs to handle diverse tasks through a single next-token prediction objective. This is a stronger claim than simply noting sequential structure—it asserts that the *training and inference machinery* developed for language (teacher forcing, parallel training over sequences, constant-memory state compression) can be ported to video with appropriate architectural adaptations.

**2. The unification is achieved by dissolving the vectorization bottleneck.** Prior universal models preserve the query-vector interface from VIS methods because it makes detection natural. AUSM instead preserves spatial information through the History Marker (dissolving masks into feature maps) and handles detection through a separate set of object queries $\mathcal{V}$ that operate alongside the tracking ID vectors $\mathcal{A}$. This means the model maintains both fine-grained spatial memory (for VOS-quality propagation) *and* learnable detection queries (for VIS-quality discovery), without forcing one to serve the other's role.

**3. State-space compression replaces explicit memory buffers.** The History Compressor, built on Mamba (Gu and Dao, 2023), compresses all past information into a single fixed-size spatial state—analogous to how an SSM maintains a hidden state that summarizes the entire sequence history. This is what enables constant-memory inference on arbitrarily long videos, in contrast to STM-based methods where the memory buffer grows with video length, and what the paper means by "processing arbitrarily long streams" without FIFO buffers.

**4. Parallel training is a first-class design goal, not an afterthought.** The paper explicitly designs every module (History Marker, History Compressor, History Decoder, Pixel Decoder) to be compatible with teacher forcing. The Preprocess function in Algorithm 2 is the key enabling mechanism: by randomly sampling a "detection-to-tracking transition point" $t^i_{\text{sample}}$ for each instance, it constructs ground-truth tracking and detection targets that can be computed independently for every frame, allowing parallel loss computation across the entire sequence.

**5. The empirical positioning is clear: AUSM competes with specialized models on their home turf while being universal.** The paper does not claim to beat SAM2 on prompted VOS—the gap in Table 1 (AUSM Swin-B at 81.6 $J \& F$ on DAVIS vs. SAM2 Hiera-B+ at 90.2) is acknowledged as a limitation stemming from using coarser feature strides. But AUSM substantially outperforms all prior universal streaming models, and it achieves competitive VIS performance with specialized methods while also handling VOS—something no specialized VIS model can do. The training speed comparison (Figure 4) positions AUSM as the first video segmentation model whose training efficiency scales favorably with sequence length—a property that becomes increasingly important as the field moves toward longer video understanding.

In summary, the paper identifies a specific architectural bottleneck (instance vectorization in universal models) that explains the persistent performance gap between universal and specialized approaches, proposes a specific mechanism (spatial mask dissolution via History Marker + temporal compression via Mamba) to resolve it, and wraps this in a training framework (teacher-forced parallel optimization) that makes the approach scalable. The contribution is not just "we built a unified model" but "we identified *why* previous unified models underperformed and designed an architecture that addresses that specific failure mode while maintaining scalability."

## 3. Technical Approach

### 3.1 Reader Orientation

AUSM is a single neural network that takes a streaming video (frame by frame) and, depending on how you initialize it, either tracks a specific object you point to in the first frame or automatically discovers, classifies, and tracks every object in every frame — without any architectural switching between these modes. The problem it solves is that prior "universal" video segmentation models achieve their generality by compressing each object into a single feature vector, which loses the fine spatial detail needed for precise mask propagation; AUSM instead preserves full spatial information through a History Marker that paints instance masks directly onto feature maps, then compresses all of time into a single fixed-size state using Mamba state-space layers, enabling both spatial precision and constant-memory inference on arbitrarily long videos.

### 3.2 Big-Picture Architecture (Diagram in Words)

AUSM has five major components arranged in a pipeline that processes one frame at a time during inference, but all frames in parallel during training:

1. **Frame Encoder (backbone)**: A Swin Transformer that processes each RGB frame independently into a feature map `$X_t \in \mathbb{R}^{H \times W \times D}$` at 1/8 resolution. This is not novel — it is a standard Swin-T or Swin-B pretrained on ImageNet.

2. **History Marker**: Takes the object ID vectors `$\mathcal{A}_{t-1}$` and their corresponding mask predictions `$\mathcal{M}_{t-1}$` from the previous frame, and "paints" each instance's ID vector onto the spatial locations where that instance was predicted to exist. The output is a spatial feature map `$S_t \in \mathbb{R}^{H \times W \times D}$` that enriches the previous frame's features `$X_{t-1}$` with instance-specific information *without* compressing instances into single vectors.

3. **History Compressor**: A 6-layer module (each layer = temporal Mamba + spatial self-attention + FFN) that takes the History Marker's output `$E_t = X_{t-1} + S_t$` and updates a single compressed spatial state `$F_t \in \mathbb{R}^{H \times W \times D}$`. The Mamba layer runs pixel-wise along the time axis, compressing all past temporal information into this state. Self-attention runs spatially within each frame. The output `$F_t$` has constant size regardless of how many frames have been processed.

4. **History Decoder**: A 6-layer Transformer decoder that takes the current frame's features `$X_t$` as queries and the compressed state `$F_t$` as keys/values, producing a spatially-aware feature map `$G_t$` that fuses current visual input with compressed temporal history.

5. **Pixel Decoder**: A Mask2Former-style decoder with masked attention that takes two sets of queries — allocated ID vectors `$\mathcal{A}_{t-1}$` (for tracking known objects) and learnable object queries `$\mathcal{V}$` (for detecting new objects) — attends to `$G_t$`, and outputs two sets of predictions: tracking masks/logits from `$\mathcal{A}_{t-1}$` and detection masks/logits from `$\mathcal{V}$`. After prediction, newly detected foreground objects get new ID vectors sampled from a buffer `$\mathcal{B}$`, and the process repeats for the next frame.

The information flow: RGB frame → backbone → feature map → (History Decoder attends to compressed history) → (Pixel Decoder produces masks using tracking + detection queries) → (History Marker paints masks onto features) → (History Compressor updates temporal state) → next frame.

### 3.3 Roadmap for the Deep Dive

- **First, the autoregressive formulation** (Equations 1–2): How video segmentation maps onto the same probabilistic framework as language modeling, and why this enables task unification.
- **Second, the task unification mechanism** (prompted vs. unprompted initialization): How the same architecture handles both modes by simply changing how `$\mathcal{A}_0$` and `$\mathcal{M}_0$` are initialized — the cleanest demonstration that the framework is genuinely unified.
- **Third, the History Marker**: The mechanism that dissolves instance masks into spatial features, why it preserves fine-grained information that vectorization loses, and the precise mathematical operation (weighted averaging of ID vectors at each spatial location).
- **Fourth, the History Compressor**: The Mamba + self-attention layer design, why temporal processing uses Mamba (recurrent, constant-memory) while spatial processing uses self-attention (global context per frame), and how this enables arbitrary-length video inference.
- **Fifth, the Pixel Decoder and update process**: How tracking and detection are handled simultaneously through separate query sets, the Hungarian matching for detection, and the buffer management for allocating new ID vectors.
- **Sixth, the parallel training formulation**: The Preprocess function that enables teacher forcing across frames, why randomly sampling detection-to-tracking transition points is the key enabling trick, and the loss decomposition.

### 3.4 Detailed, Sentence-Based Technical Breakdown

This is primarily a **systems and architecture paper** whose core idea is that video segmentation can be reformulated as autoregressive next-frame mask prediction, enabling a single model to handle prompted and unprompted tasks through parallelizable training with compressed temporal state — provided you preserve spatial instance information rather than vectorizing it.

---

#### The Autoregressive Formulation of Video Segmentation

The paper's foundational move is to recognize that the probabilistic formulation used to train decoder-only language models applies directly to streaming video segmentation. In language modeling, the joint probability of a token sequence `$y_{1:T}$` is factorized as:

$$P(y_{1:T}) = \prod_{t=1}^T P(y_t \mid y_{<t})$$

where `$y_t$` is the token at position `$t$`, and `$y_{<t}$` denotes all tokens before position `$t$`.

**What it computes:** The probability of the entire sequence is the product of conditional probabilities of each token given all previous tokens. This factorization is what enables autoregressive generation — you sample token 1, then sample token 2 conditioned on token 1, then token 3 conditioned on tokens 1 and 2, and so on.

**Why this form:** This decomposition reflects the causal structure of streaming data — the future cannot influence the past. It also enables teacher-forced parallel training: during training, you use the ground-truth previous tokens rather than the model's own predictions as conditioning, so all conditional probabilities `$P(y_t \mid y_{<t})$` can be computed simultaneously. This is the computational insight that makes LLM training feasible at scale.

The paper extends this to video by conditioning each frame's segmentation `$y_t$` on all previous frames `$\mathcal{I}_{\leq t}$`, all previous segmentations `$y_{<t}$`, and an optional initial prompt `$y_0$`:

$$P(y_{1:T} \mid \mathcal{I}_{1:T}) = \prod_{t=1}^T P(y_t \mid y_0, y_{<t}, \mathcal{I}_{\leq t})$$

where `$\mathcal{I}_t \in \mathbb{R}^{H \times W \times 3}$` is the RGB frame at time `$t$`, `$y_t = \{(c^i_t, m^i_t)\}_{i=1}^{N_{\text{gt}}}$` is the set of class labels `$c^i_t \in \{1,\ldots,K\}$` and binary masks `$m^i_t \in \{0,1\}^{H \times W}$` for all `$N_{\text{gt}}$` foreground objects at time `$t$`, `$y_0$` is an optional initial prompt (a set of masks in the first frame for prompted VOS, or `$\emptyset$` for unprompted VIS), and `$y_{<t}$` are all segmentations from frames 1 through `$t-1$`. The notation `$\mathcal{I}_{\leq t}$` means the model can see the current frame and all previous frames.

**What it computes:** The probability of a complete video segmentation `$y_{1:T}$` given the video `$\mathcal{I}_{1:T}$` is the product of per-frame conditional probabilities, where each frame's segmentation depends on: (1) the current and all previous visual inputs, (2) all previous segmentation decisions the model made, and (3) an optional human-provided prompt in the first frame.

**Why this form:** This factorization does three things simultaneously. First, it explicitly encodes the streaming constraint — frame `$t$`'s prediction depends only on past and present, not future. Second, it unifies prompted and unprompted settings through a single variable `$y_0$`: when `$y_0 \neq \emptyset$` (prompted VOS), the model conditions on an initial mask and tracks only those objects; when `$y_0 = \emptyset$` (unprompted VIS), the model must discover all objects autonomously. Third, it provides a principled training objective: maximize the likelihood of the ground-truth segmentation sequence under this autoregressive factorization. The model learns `$P(y_t \mid y_0, y_{<t}, \mathcal{I}_{\leq t})$` — a conditional distribution over possible segmentations — and during inference, this distribution is sampled to produce the output.

The critical departure from language modeling is that `$y_t$` is not a single token but a *set* of objects, each with a mask and class label. This is the architectural challenge the rest of the paper addresses: how do you condition on a set-valued output from the previous timestep in a way that is both spatially precise and computationally efficient?

---

#### Task Unification Through Initialization

The cleanest demonstration that AUSM is genuinely unified — rather than two models sharing a backbone — is that the architectural components are identical across tasks; only the initialization of two data structures differs. These structures are:

- `$\mathcal{A}_t$`: the set of allocated ID vectors at time `$t$`. Each vector is a `$D$`-dimensional learned embedding from a fixed pool `$\mathcal{B} \in \mathbb{R}^{N_{\text{id}} \times D}$`, where `$N_{\text{id}} = 100$` is the total number of available ID vectors. Once an ID vector is assigned to an object, it follows that object for the rest of the video, providing a persistent identity signal.
- `$\mathcal{M}_t$`: the set of predicted masks at time `$t$`, with a one-to-one correspondence to `$\mathcal{A}_t$` — the `$i$`-th mask in `$\mathcal{M}_t$` belongs to the object identified by the `$i$`-th vector in `$\mathcal{A}_t$`. Formally, `$|\mathcal{A}_t| = |\mathcal{M}_t|$` at all times.

For **unprompted video segmentation** (Line 3 of Algorithm 1): both `$\mathcal{A}_0$` and `$\mathcal{M}_0$` are initialized as empty sets, and the full ID buffer is available: `$\mathcal{B}_0 = \mathcal{B}$`. The model starts with no prior knowledge of what objects exist — it must discover them through its detection mechanism and allocate ID vectors as it goes.

For **prompted video segmentation** (Lines 6–8 of Algorithm 1): the initial prompt `$y_0 = m_0$` specifies `$|y_0|$` masks in the first frame. The initialization samples `$|y_0|$` ID vectors from the buffer: `$\mathcal{A}_0 = \text{Sampler}(\mathcal{B}, |y_0|)$`, sets `$\mathcal{M}_0 = m_0$`, and removes the allocated vectors from the available pool: `$\mathcal{B}_0 = \mathcal{B} \setminus \mathcal{A}_0$`. The Sampler function draws `$n$` vectors uniformly at random without replacement from `$\mathcal{B}$`, returning them as a matrix `$[b_1, \ldots, b_n]^\top \in \mathbb{R}^{n \times D}$` where each `$b_i$` is a row vector.

**What this means operationally:** In prompted mode, the model enters the first frame already "knowing" which ID vectors correspond to the prompted objects and what their masks look like. The History Marker can immediately paint these masks onto the feature map, the History Compressor can start building temporal context, and the tracking branch (using `$\mathcal{A}_0$`) can begin propagating. The detection branch (using `$\mathcal{V}$`) still runs in parallel — it might detect other objects, but in standard VOS evaluation those detections are ignored. In unprompted mode, the model must build up its object inventory from scratch through the detection branch, allocating ID vectors as new foreground objects are discovered.

**Why this design:** The separation of ID vectors (`$\mathcal{A}$`) from detection queries (`$\mathcal{V}$`) is the architectural decision that makes unification possible without the vectorization bottleneck. `$\mathcal{A}$` carries persistent identity and is spatially grounded through the History Marker; `$\mathcal{V}$` is a set of learnable queries that specialize in detecting objects not yet tracked. Prior universal models (UniVS, UNINEXT) forced the detection queries to also serve as tracking representations, which required compressing each instance into a single vector. AUSM instead maintains two separate query pools with distinct roles, connected only through the shared spatial feature map `$G_t$`.

---

#### History Marker: Dissolving Instance Masks into Spatial Features

The History Marker is the component that addresses the "vectorization bottleneck" — the observation that compressing an instance mask into a single `$D$`-dimensional vector loses spatial detail. Instead of compressing, the History Marker does the opposite: it *expands* instance identity vectors back into the spatial domain by using the predicted masks as a guide for where each vector should appear.

**Formal operation.** At time `$t$`, given the allocated ID vectors `$\mathcal{A}_{t-1}$` and their corresponding masks `$\mathcal{M}_{t-1}$` from the previous frame, the History Marker produces a spatial feature map `$S_t \in \mathbb{R}^{H \times W \times D}$`:

$$S_t[h, w, :] = \frac{\sum_{i=1}^{|\mathcal{A}_{t-1}|} \mathcal{M}^i_{t-1}[h, w] \cdot \mathcal{A}^i_{t-1}}{\epsilon + \sum_{i=1}^{|\mathcal{A}_{t-1}|} \mathcal{M}^i_{t-1}[h, w]}$$

where `$S_t[h, w, :]$` is the `$D$`-dimensional feature vector at spatial location `$(h, w)$`, `$|\mathcal{A}_{t-1}|$` is the number of tracked objects, `$\mathcal{M}^i_{t-1}[h, w] \in \{0, 1\}$` is the binary mask value for object `$i$` at that spatial location, `$\mathcal{A}^i_{t-1} \in \mathbb{R}^D$` is the ID vector for object `$i$`, and `$\epsilon$` is a small constant preventing division by zero at locations where no object is present.

**What it computes:** For each spatial location `$(h, w)$`, the History Marker computes a weighted average of the ID vectors of all objects that occupy that location, weighted by their predicted mask values at that location. If only object `$i$` is present at `$(h, w)$` (its mask is 1, all others are 0), the output is simply `$\mathcal{A}^i_{t-1}$` — the full `$D$`-dimensional ID vector representing that object. If multiple objects overlap at `$(h, w)$`, the output is a soft combination of their ID vectors proportional to their mask values. If no object occupies `$(h, w)$`, the denominator `$\epsilon$` prevents division by zero and the result is approximately a zero vector. This output `$S_t$` is then added to the previous frame's features: `$E_t = X_{t-1} + S_t$`.

**Why this form:** This operation is the spatial inverse of the typical compression pipeline. Instead of pooling an object's features into a single vector (e.g., RoIAlign or masked average pooling over the object's spatial extent), it broadcasts the vector back out to the object's spatial extent. The effect is that `$X_{t-1}$` — which contains general visual features (edges, textures, semantic patterns) — gets enriched with instance-specific identity information at exactly the locations where those instances exist. The History Compressor then processes this enriched feature map `$E_t$`, and because the identity information is spatially localized, the temporal Mamba layers can learn to track how each instance's appearance and position evolve over time without losing the fine spatial structure.

The contrast with prior work is stark. UniVS encodes each instance as a single vector derived from its mask; this vector must capture everything about the object — its shape, boundary, occlusion state, appearance — in `$D$` numbers. The History Marker instead keeps the full `$H \times W$` spatial grid and places instance vectors only where they belong. When the History Compressor and History Decoder later process this grid, they have access to the full spatial layout of each instance, which is what enables the nearly 10% VOS improvement over UniVS (76.4 vs. 71.7 $J \& F$ on DAVIS with Swin-T).

A subtle detail: the History Marker operates on the *previous* frame's features `$X_{t-1}$`, not the current frame's features `$X_t$`. This is because the masks `$\mathcal{M}_{t-1}$` describe where objects were in frame `$t-1$`, so enriching `$X_{t-1}$` is geometrically consistent. The information about current-frame object positions comes through the History Decoder, which attends from `$X_t$` (queries) to the compressed history `$F_t$` (keys/values) — a cross-attention that implicitly learns to warp past spatial information to current locations.

---

#### History Compressor: Mamba for Temporal, Self-Attention for Spatial

The History Compressor is a 6-layer module that takes the enriched feature map `$E_t \in \mathbb{R}^{H \times W \times D}$` and updates a compressed spatial state. Each layer contains three sub-modules applied in sequence: temporal Mamba, spatial self-attention, and a feed-forward network (FFN). The paper's diagram in Figure 2 shows this structure explicitly.

**Layer architecture.** For a single layer at frame `$t$`, the operations are:

1. **Temporal Mamba**: A Mamba (Gu and Dao, 2023) state-space layer that operates *pixel-wise* along the time axis. Conceptually, treat each spatial location `$(h, w)$` as an independent 1D sequence of length `$t$` (one value per frame). The Mamba layer processes this sequence with a selective state-space model that maintains a hidden state updated at each timestep. The output is a transformed feature for that spatial location at the current timestep, incorporating information from all past timesteps through the compressed state. The key property: this operation requires storing only a single state vector per spatial location, not the full sequence of past features. Formally, the Mamba state `$F^{\text{mamba}}_t \in \mathbb{R}^{H \times W \times D}$` is a function of the input `$E_t$` and the state carried forward from `$t-1$`.

2. **Spatial Self-Attention**: Standard multi-head self-attention (Vaswani et al., 2017) applied *independently to each frame*. Each spatial location attends to all other spatial locations within the same frame, with no cross-frame connections. This captures long-range spatial dependencies — an object's feature at one location can attend to the entire visual context of the frame.

3. **Feed-Forward Network (FFN)**: A standard MLP applied independently to each spatial location, providing additional non-linear transformation capacity.

The paper explicitly describes the temporal Mamba as operating "pixel-wise, mixing information of each pixel throughout the time dimension `$T$`," while the spatial self-attention "is frame-independent that fuses information spanning over `$HW$` pixels." This decomposition — Mamba on the time axis, self-attention on the spatial axis — is the key design decision.

**Why Mamba on the temporal axis:** The paper gives two reasons. First, videos are inherently sequential in time, which aligns naturally with state-space model (SSM) architectures — an SSM's recurrent state is exactly the right inductive bias for maintaining a running summary of temporal information. Second, modeling videos is memory-intensive because each frame contributes `$H \times W$` tokens (at stride 8 with typical resolutions, this is approximately `$128 \times 228 \approx 29,000$` tokens per frame for a 1024-pixel shorter side). A Transformer operating on the full spatio-temporal sequence would require `$\mathcal{O}((T \cdot H \cdot W)^2)$` attention, which is infeasible for long videos. The Mamba's recurrent design means the temporal processing costs `$\mathcal{O}(H \cdot W \cdot D^2)$` per frame regardless of `$T$` — constant in video length.

**Why self-attention on the spatial axis:** Within a single frame, the model needs to capture global context — an object at the top-left may be related to an object at the bottom-right. Self-attention is the standard mechanism for this, and since it operates per-frame (not across frames), the cost is `$\mathcal{O}(T \cdot (H \cdot W)^2)$` total, which is manageable.

**The compressed state property.** After processing frame `$t$`, the History Compressor produces `$F_t \in \mathbb{R}^{H \times W \times D}$`. This is a single spatial feature map — the same size as one frame's features — that encodes all temporal information from frames 1 through `$t$`. There is no explicit memory buffer storing past frames or past masks. The Mamba state carries forward the temporal context implicitly. This is what the paper means by "enabling inference over arbitrarily long videos with constant memory" — the memory footprint at frame `$t$` is the same as at frame `$t=1$`, regardless of how many frames have been processed.

The paper contrasts this explicitly with STM-based methods (Oh et al., 2019) that store past frames and masks in a FIFO buffer. Those methods must decide what to discard when the buffer fills; AUSM has no such limitation because the compression is learned, not controlled by a fixed buffer size.

**Integration with the History Marker output.** The input to the History Compressor is `$E_t = X_{t-1} + \text{HistoryMarker}(\mathcal{A}_{t-1}, \mathcal{M}_{t-1})$`. This means the Mamba layer receives frame `$t-1$`'s backbone features enriched with painted instance IDs. By processing this through temporal Mamba, the model learns to extract motion information — how the object identities placed at specific locations in frame `$t-1$` relate to their current positions in frame `$t$`. The compressed state `$F_t$` thus captures both appearance and identity motion.

---

#### History Decoder: Fusing Current Input with Compressed History

The History Decoder is a stack of 6 Transformer decoder layers that takes `$F_t$` (the compressed spatial state from the History Compressor) as keys and values, and the current frame's backbone features `$X_t$` as queries. The output is a feature map `$G_t \in \mathbb{R}^{H \times W \times D}$`.

**What this means operationally:** Each spatial location `$(h, w)$` in `$X_t$` (the query) attends to all spatial locations in `$F_t$` (the keys/values). Since `$F_t$` contains compressed temporal information from all past frames, this cross-attention allows the current frame to "look up" relevant historical context. For example, a pixel in `$X_t$` that belongs to a partially occluded object can attend to positions in `$F_t$` where that same object was visible in the past, retrieving identity and appearance information.

**Why cross-attention rather than concatenation:** Simply concatenating `$X_t$` and `$F_t$` channel-wise would give the subsequent Pixel Decoder access to both, but without learned selection of *which* historical information is relevant for *which* spatial location. Cross-attention provides a learned, spatially-varying retrieval mechanism — different parts of the current frame can retrieve different aspects of history.

**The dual property of `$G_t$`:** The paper states that `$G_t$` has two important properties: (1) it incorporates current-frame information through `$X_t$`, and (2) it retains fine-grained information about objects from previous frames through the compressed state `$F_t$`. This dual nature is what enables the Pixel Decoder to simultaneously track known objects (using historical identity information) and detect new objects (using current visual features).

**Design choice — decoder rather than encoder:** The History Decoder uses Transformer *decoder* layers (cross-attention from queries to keys/values) rather than *encoder* layers (self-attention within the combined set). This is intentional: `$X_t$` and `$F_t$` have different roles. `$X_t$` is the "question" — what is happening in this frame? `$F_t$` is the "memory" — what has happened before? Cross-attention preserves this asymmetric relationship in a way that self-attention over the concatenated set would not.

---

#### Pixel Decoder: Masked Attention for Tracking and Detection

The Pixel Decoder follows the Mask2Former architecture (Cheng et al., 2022) and consists of Transformer decoder layers with **masked attention** — a variant of cross-attention where each object query attends only to the spatial locations within its predicted mask region from the previous decoder layer, rather than the entire feature map. This focused attention helps the decoder refine mask boundaries without being distracted by irrelevant background regions.

**Input queries.** The Pixel Decoder takes two sets of queries concatenated together:

- **Tracking queries:** The allocated ID vectors `$\mathcal{A}_{t-1} \in \mathbb{R}^{|\mathcal{A}_{t-1}| \times D}$`. There is one query per tracked object. These queries carry persistent identity information — the same ID vector has been following each object since it was first detected.
- **Detection queries:** The learnable object queries `$\mathcal{V} \in \mathbb{R}^{N_{\text{det}} \times D}$`, where `$N_{\text{det}} = 100$`. These are fixed learned embeddings (initialized randomly and optimized during training) that are designed to detect objects not yet assigned to any tracking query.

**Keys and values.** Both query sets attend to the same feature map `$G_t$` from the History Decoder.

**Outputs.** The Pixel Decoder produces two sets of predictions:

- **Tracking predictions `$\hat{y}^{\text{trk}}_t$`**: For each allocated ID vector `$\mathcal{A}^i_{t-1}$`, the model predicts a class logit `$\hat{c}^i_t$` and a mask logit map `$\hat{m}^i_t$`. These represent the model's belief about where the `$i$`-th tracked object is in the current frame.
- **Detection predictions `$\hat{y}^{\text{det}}_t$`**: For each detection query `$\mathcal{V}^j$`, the model predicts a class logit (including background) and a mask logit map. These represent candidate new objects.

**Why two separate query sets:** This design solves a fundamental tension in detection-based tracking. Detection queries need to be "uncommitted" — they should be free to discover any object in any frame. Tracking queries need to be "committed" — they must maintain consistent identity for their assigned object. If you forced the same queries to do both (as in query propagation methods), you would constantly face a tradeoff: reuse a query for tracking (sacrificing detection capacity) or reallocate it for detection (sacrificing temporal consistency). AUSM avoids this tradeoff by having a fixed detection pool (`$\mathcal{V}$`) that always searches for new objects and a growing tracking pool (`$\mathcal{A}_t$`) that always follows known objects.

**Masked attention refinement.** The paper notes that the Pixel Decoder "follows Cheng et al. (2022), comprising Transformer decoder layers with masked attention." In Mask2Former, each decoder layer produces an intermediate mask prediction, and the next layer's cross-attention is restricted to the spatial region where that intermediate mask is active. This iterative refinement is critical for producing high-quality boundaries, and it works identically for both tracking and detection queries — the only difference is the source of the query vectors.

---

#### The Update Process: Managing Identity Across Frames

After the Pixel Decoder produces predictions, the model must update its state for the next frame. This involves deciding which detected objects to start tracking, allocating new ID vectors, and maintaining the one-to-one correspondence between `$\mathcal{A}_t$` and `$\mathcal{M}_t$`. The process is defined in Lines 14–19 of Algorithm 1:

**Step 1 — Filter foreground detections:** The function `filter_fg`(`$\hat{y}^{\text{det}}_t$`) selects detections whose predicted class is not "background." This yields a set `$\mathcal{D}$` of newly detected foreground objects, each with a class label and mask. The paper uses a foreground confidence threshold of 0.5 for all unprompted benchmarks (Table 2).

**Step 2 — Sample new ID vectors:** `$|\mathcal{D}|$` new vectors are sampled from the remaining buffer `$\mathcal{B}_{t-1}$` via `$\mathcal{A}' = \text{Sampler}(\mathcal{B}_{t-1}, |\mathcal{D}|)$`. These vectors are drawn uniformly at random without replacement.

**Step 3 — Update ID vector set:** `$\mathcal{A}_t = \text{concat}(\mathcal{A}_{t-1}, \mathcal{A}')$`. The tracked objects from the previous frame stay, and the newly detected objects get their own ID vectors appended.

**Step 4 — Update buffer:** `$\mathcal{B}_t = \mathcal{B}_{t-1} \setminus \mathcal{A}'$`. The allocated vectors are removed from the available pool so they cannot be reassigned.

**Step 5 — Update mask set:** `$\mathcal{M}_t = \text{concat}(\hat{m}^{\text{trk}}_t, \mathcal{D})$`. The tracking predictions provide masks for previously tracked objects; the filtered detections provide masks for new objects. The concatenation preserves the one-to-one mapping: the first `$|\mathcal{A}_{t-1}|$` masks correspond to previously tracked objects, the next `$|\mathcal{D}|$` masks correspond to newly detected objects.

**Critical property — no deletion:** The paper does not describe any mechanism for removing objects from `$\mathcal{A}_t$`. Objects that leave the frame or become fully occluded presumably continue to be tracked (their queries will predict low-confidence masks), but they are not removed. The buffer `$\mathcal{B}$` has 100 ID vectors, and `$\mathcal{A}_t$` grows as new objects appear. If a video has more than 100 distinct objects, the buffer would be exhausted — but this is rare in the evaluated benchmarks.

**In prompted mode during inference**, no detection filtering occurs — the model only uses tracking predictions, because the goal is to propagate the given prompts rather than discover new objects. The paper states that prompted inference "involves no thresholding, filtering, or post-processing" and is "fully model-driven" (Appendix A).

---

#### Parallel Training via Teacher Forcing

The most distinctive engineering contribution of AUSM is its parallel training formulation, which directly addresses the training efficiency bottleneck that plagues all prior video segmentation methods that condition on past predictions. The core insight: by randomly assigning each ground-truth instance a "detection-to-tracking transition point" and using teacher forcing, all frames in a training clip can be processed simultaneously.

**The Preprocess function** (Algorithm 2 and Figure 3) is the key mechanism. Given a training video with `$T$` frames and ground-truth segmentation `$y_{1:T}$`:

1. **Sample ID vectors:** `$\mathcal{A} = \text{Sampler}(\mathcal{B}, N_{\text{gt}})$`. For each ground-truth instance, a unique ID vector is sampled from the buffer, establishing a one-to-one correspondence.

2. **Sample transition points:** For each ground-truth instance `$i$`, randomly sample a timestep `$t^i_{\text{sample}}$` — represented visually in Figure 3 as the highlighted contours for each instance. This timestep determines when that instance shifts from being treated as a *detection target* to being treated as a *tracking target*.

3. **Construct targets:** For each instance `$i$` and frame `$t$`:

   $$\mathcal{M}^i_t = \begin{cases} y^i_t & \text{if } t \geq t^i_{\text{sample}} \\ \emptyset & \text{otherwise} \end{cases}$$

   The mask history `$\mathcal{M}^i_t$` is only populated from the transition point onward — before that, the instance is "unknown" to the tracking branch.

   $$y^{\text{det},i}_t = \begin{cases} y^i_t & \text{if } t \leq t^i_{\text{sample}} \\ \emptyset & \text{otherwise} \end{cases} \quad y^{\text{trk},i}_t = \begin{cases} y^i_t & \text{if } t > t^i_{\text{sample}} \\ \emptyset & \text{otherwise} \end{cases}$$

   Before `$t^i_{\text{sample}}$`, the instance appears as a detection target (the model must discover it). At `$t = t^i_{\text{sample}}$`, it appears in both (detection for the last time, mask history initialized). After `$t^i_{\text{sample}}$`, it appears only as a tracking target (the model should propagate it).

4. **Construct `$\mathcal{A}_{t-1}$`**: For each frame, `$\mathcal{A}_{t-1}$` contains the ID vectors of all instances whose `$t^i_{\text{sample}} < t$` — that is, instances that have already transitioned to tracking mode by frame `$t$`.

**What this enables:** After preprocessing, the targets `$y^{\text{trk}}_{1:T}$` and `$y^{\text{det}}_{1:T}$`, the ID vector sets `$\mathcal{A}_{0:T-1}$`, and the mask histories `$\mathcal{M}_{0:T-1}$` are all known for every frame simultaneously. The History Marker can compute `$E_t$` for all `$t$` in parallel because `$\mathcal{A}_{t-1}$` and `$\mathcal{M}_{t-1}$` are ground-truth-derived, not model predictions. The History Compressor (Mamba) can process the entire sequence in parallel using the associative scan property of SSMs. The History Decoder and Pixel Decoder can process all frames as a batch.

**Why this is correct (teacher forcing):** During training, the model learns `$P(y_t \mid y_{<t})$` using the *ground-truth* `$y_{<t}$` rather than its own predictions. This is the standard teacher-forcing technique from sequence modeling (Sutskever et al., 2014): the model is trained to predict the correct output at each step given the correct history, and at inference time, it uses its own (potentially imperfect) predictions as history. The autoregressive conditional distribution being learned is identical — only the source of conditioning differs between training and inference.

**Why random transition points:** If `$t^i_{\text{sample}}$` were always frame 1, the model would never learn to detect objects (everything would be in tracking mode). If it were always frame `$T$`, the model would never learn to track (everything would be in detection mode). By sampling uniformly across `$\{1, \ldots, T\}$`, the model sees every possible ratio of detection-to-tracking examples during training, learning both skills and the transition between them.

**The loss function:** The total training loss decomposes into separate terms for tracking and detection, summed over all frames:

$$\mathcal{L}_{\text{total}} = \sum_{t=1}^T \left[ \mathcal{L}_{\text{trk}}(y^{\text{trk}}_t, \hat{y}^{\text{trk}}_t) + \mathcal{L}_{\text{det}}(y^{\text{det}}_t, \hat{y}^{\text{det}}_t) \right]$$

where `$\mathcal{L}_{\text{trk}}$` is computed directly — each `$\mathcal{A}^i_{t-1}$` maps one-to-one to its corresponding ground-truth instance `$y^{\text{trk},i}_t$`, so the loss is a standard per-instance segmentation loss (cross-entropy for masks, cross-entropy for class labels). For `$\mathcal{L}_{\text{det}}$`, there is no predetermined mapping between detection queries `$\mathcal{V}$` and ground-truth instances, so the Hungarian algorithm (Carion et al., 2020) finds the optimal bipartite matching that minimizes the combined classification and mask losses before computing gradients. This is identical to the matching used in DETR and Mask2Former.

**Why this matters for scalability:** The iterative (recurrent) training baseline processes frames one at a time — frame 1's predictions are needed before frame 2 can be processed. This yields training time scaling roughly linearly with sequence length. The parallel training formulation processes all frames as a batch, with the only overhead being the Hungarian matching for detection queries. As shown in Figure 4, the parallel approach scales from 1.47s/iter at length 1 to 3.45s/iter at length 16 (2.3× increase), while the iterative approach scales from 1.47s to 8.75s/iter (6.0× increase). The 2.5× speedup at 16 frames is expected to grow with longer sequences.

---

#### Training Configuration and Stages

The paper details a three-stage training curriculum (Section 3.2 and Appendix A), progressively increasing temporal complexity and data diversity:

**Stage 1 — Pseudo-video pretraining (20 epochs, 147,500 iterations):** Trains on COCO images (Lin et al., 2014) converted into 3-frame pseudo-videos via random spatial augmentations following Heo et al. (2022). Each static image is transformed into a synthetic "video" by applying random spatial transforms (e.g., cropping, scaling, rotation) to create apparent motion. This stage initializes the model with Mask2Former weights pretrained on COCO image instance segmentation, providing strong single-frame detection and segmentation capabilities before temporal modeling is introduced.

**Why pseudo-videos:** Real video data is expensive and scarce. Pseudo-video pretraining on COCO provides a massive source of diverse object appearances and categories (80 COCO classes) at low cost, teaching the model what objects look like before it needs to learn how they move. The 3-frame limit keeps training fast while introducing the autoregressive formulation.

**Stage 2 — Multi-source short-clip training (32,000 iterations):** Introduces real video data using 5-frame clips sampled from COCO-pseudo, MOSE, SA-V, YouTube-VIS 2019 & 2021, and OVIS. This stage teaches the model short-range temporal dynamics (how objects move across 5 real frames) and exposes it to diverse visual domains (YouTube videos, egocentric scenes, heavily occluded scenarios).

**Stage 3 — Long-clip adaptation (40,000 iterations):** Fine-tunes on 16-frame clips from the same datasets plus DAVIS 2017. To manage GPU memory with longer sequences, the image backbone is frozen and only the temporal modules (History Compressor, History Decoder) and prediction heads (Pixel Decoder) are updated. This stage strengthens long-range temporal modeling, with the largest gains observed on MOSE (+4.5 points) and OVIS (+5.2 points) — datasets characterized by complex dynamics and long videos (Figure 5).

**Optimization details (Appendix A):** All stages use AdamW optimizer (Loshchilov and Hutter, 2019) with an initial learning rate of `$1 \times 10^{-4}$`, batch size 16, across 16 NVIDIA A100 GPUs. The History Compressor has 6 layers, the History Decoder has 6 layers, feature dimension `$D = 256$`, and the backbone feature map is at 1/8 resolution.

**Why freeze the backbone in Stage 3:** Processing 16-frame sequences at 1/8 resolution with a Swin-B backbone on 16 A100 GPUs is memory-intensive. Freezing the backbone reduces GPU memory usage (no need to store backbone activations or compute backbone gradients), allowing the limited memory budget to be spent on longer sequences rather than larger per-frame models. The paper acknowledges this as a practical constraint and suggests that next-generation hardware will enable training on even longer sequences with the full model.

---

#### Summary of Design Choices and Their Justifications

- **Mamba on the temporal dimension, self-attention on spatial:** Mamba provides the recurrent state needed for constant-memory long-video inference; self-attention provides the global spatial context needed within each frame. Using Mamba spatially or attention temporally would invert their strengths — Mamba would lose global spatial context, and attention would incur quadratic cost in time.
- **History Marker rather than vectorization:** Vectorizing instances (pooling spatial features into a single vector) loses boundary precision needed for VOS. The History Marker's weighted averaging preserves spatial layout, allowing the History Compressor and Decoder to work with fine-grained instance information.
- **Separate tracking and detection query pools:** Detection queries need to be uncommitted to find new objects; tracking queries need persistent identity. Forcing one pool to do both creates a tradeoff that hurts both tasks, as evidenced by prior universal models' VOS performance drops.
- **Random detection-to-tracking transition points in preprocessing:** Ensures the model sees every possible ratio of detection and tracking examples during training, learning both skills and the transition between them without biasing toward early or late detection.
- **Teacher forcing across frames:** The standard technique from language modeling that enables parallel training. The alternative — recurrent training with model predictions as conditioning — is 2.5× slower at 16 frames and scales poorly.
- **Three-stage curriculum (pseudo-video → short clips → long clips):** Mirrors the standard image-pretraining-then-video-finetuning paradigm but adapted for the autoregressive formulation, introducing temporal modeling gradually while leveraging abundant image data.
- **Feature stride of 8:** Coarser than the stride of 4 used by some specialized VOS methods. The paper acknowledges this as a tradeoff: stride 8 reduces memory consumption (fewer tokens per frame) and is sufficient for object-level understanding, but loses some fine boundary detail compared to stride 4, contributing to the gap vs. SAM2 on prompted VOS benchmarks.

## 4. Key Insights and Innovations

### Innovation 1: Video Segmentation as Autoregressive Next-Frame Mask Prediction — Not Just an Analogy, But an Operational Identity

The field has long noted that video and language are both sequential modalities. Prior work has used this as loose motivation — "videos are sequences, so transformers should work" — but has not exploited the *specific computational properties* that make autoregressive language modeling scalable. AUSM's foundational move is to recognize that the probabilistic factorization `$P(y_{1:T} \mid \mathcal{I}_{1:T}) = \prod_{t=1}^T P(y_t \mid y_0, y_{<t}, \mathcal{I}_{\leq t})$` is not merely descriptive; it is **operationally equivalent** to the language modeling factorization in a way that directly enables three concrete capabilities that prior video segmentation methods could not achieve simultaneously.

The distinction is subtle but important. Language models do not just process sequences — they process them through a specific computational contract: (1) an autoregressive decomposition of the joint probability, (2) teacher forcing during training so all positions can be optimized in parallel, and (3) a compressed state representation (in modern SSM-based architectures) that enables constant-memory inference on arbitrarily long sequences. These are not incidental properties; they are the engine of LLM scalability. AUSM's insight is that video segmentation can inherit this entire computational contract, not just the superficial idea of sequential processing.

Compare this to prior work's use of autoregression in video. Query propagation methods (GenVIS) condition on past predictions, but the conditioning is via object query vectors updated recurrently — there is no parallelizable teacher-forcing mechanism because the predictions are model-generated, not ground-truth-derived. Mask propagation methods (RoCoVIS) condition on past masks, but inherit the same sequential dependency. STM-based memory methods (SAM2, XMem) store past frames explicitly rather than compressing them into a state, so inference memory grows with video length. The autoregressive factorization was present in spirit in all these methods — they all condition on the past — but none of them implemented the *full computational contract* that makes it scalable.

AUSM's specific contribution is identifying and resolving each piece of this contract:
- **Teacher forcing** requires that intermediate targets (`$\mathcal{M}_{t-1}$`, `$\mathcal{A}_{t-1}$`) be known for all frames simultaneously. The Preprocess function with random detection-to-tracking transition points makes this possible, which is a non-obvious trick — it's not something anyone would think to do unless they were explicitly trying to port teacher forcing from language modeling.
- **Constant-memory inference** requires that temporal information be compressed into a state of fixed dimensionality. The History Compressor's Mamba layers provide this, replacing the explicit memory buffers that grow with video length in prior methods.
- **Task unification** requires that a single variable (`$y_0$`) controls the initial conditioning, exactly as language models unify tasks through a single next-token objective.

The evidence that this is a genuine conceptual advance rather than a repackaging: prior universal models (UNINEXT, UniVS) achieved task unification but sacrificed VOS performance (UniVS Swin-B at 75.0 on DAVIS vs. AUSM Swin-B at 81.6) *and* could not train in parallel. Prior recurrent methods (GenVIS, RoCoVIS) achieved temporal conditioning but could not train in parallel *and* could not handle prompted settings. AUSM is the first model that achieves all three properties simultaneously (Table 1 and Figure 4). The fact that all three properties emerge from a single design decision — taking the autoregressive contract seriously — suggests this is a fundamental architectural insight, not an incremental integration of known components.

This is a **diagnostic contribution**: it reveals that the fragmentation in video segmentation (task-specific architectures, training protocols, and memory mechanisms) was not because the tasks are inherently incompatible, but because the field had not adopted the right computational framework. The autoregressive formulation is the "right" framework not because videos are "like" language, but because it provides the specific computational properties (parallel training, constant-memory inference, task unification through conditioning) that video segmentation needs and that prior approaches lacked.

---

### Innovation 2: The Vectorization Bottleneck as the Root Cause of Universal Model Underperformance — A Diagnostic Contribution

Prior to AUSM, the field knew that universal video segmentation models (UNINEXT, UniVS, TarViS) underperformed specialized VOS methods on prompted benchmarks. The dominant explanation was implicit: universal models sacrifice VOS performance because they are designed for detection, and detection architectures (object queries, Hungarian matching) are fundamentally different from memory-based propagation architectures (dense matching, spatial memory). This framing suggested a permanent tradeoff — to be universal is to accept worse VOS.

AUSM provides a **diagnostic reframing**: the gap is not caused by the detection-centric architecture per se, but by a single, specific design choice — **instance vectorization**. When you compress each instance's mask into a single `$D$`-dimensional vector (as UniVS does to use "prompts as queries," and as GenVIS does for query propagation), you lose the spatial structure of the mask. You cannot recover from this vector where exactly the object's boundary was, or which part of the object was occluded, or how its shape changed. The information is gone.

This is a **diagnostic** rather than a **methodological** contribution because it identifies *why* prior universal models failed, not just *that* they failed. The evidence is concrete and comparative. UniVS with Swin-L achieves 76.2 on DAVIS 2017; AUSM with Swin-B achieves 81.6 — a smaller backbone achieving substantially better VOS. The nearly 10% improvement the paper references (Section 1) comes specifically from replacing vectorized instance representations with the History Marker's spatial dissolution. This is not a small refinement; it's a factor-of-two reduction in the gap between universal and specialized models (from ~14 points behind SAM2 Hiera-B+ at 90.2 to ~9 points behind).

The significance extends beyond the specific mechanism. The paper demonstrates that the key tradeoff in universal video segmentation is not "detection vs. tracking" or "prompted vs. unprompted" but **"how do you represent instance information across time?"** If you vectorize, you can use efficient query-based detection but lose spatial precision. If you keep spatial representations (as memory-based VOS methods do), you preserve precision but lose the detection interface. AUSM resolves this by keeping spatial representations for tracking (through the History Marker) while maintaining a separate query-based interface for detection (through `$\mathcal{V}$`). The two representations coexist without forcing one to serve the other's role.

This is a **reframing contribution**: it changes how researchers should think about the universal video segmentation problem. The question is no longer "how do we adapt memory-based VOS to also do detection?" or "how do we improve query-based detection to also do VOS?" but rather "how do we maintain two complementary instance representations — spatial for tracking, query-based for detection — and connect them through a shared temporal state?" This reframing opens a design space that prior work, locked into the vectorization assumption, could not explore.

---

### Innovation 3: Compressed Spatial State as a Replacement for Explicit Memory Buffers — Enabling Arbitrary-Length Video Processing

Specialized VOS methods since STM (Oh et al., 2019) have relied on explicit memory buffers — storing past frames and their masks, then performing dense matching against this memory at each new frame. This approach has produced the strongest VOS results (SAM2 achieves 90.2 on DAVIS), but it has two fundamental scaling problems: (1) memory grows with video length, requiring FIFO eviction policies that cause forgetting, and (2) per-object memory buffers make processing multiple objects expensive.

AUSM's History Compressor replaces this explicit buffer with a **learned compressed state** — a single spatial feature map `$F_t$` that encodes all temporal information through Mamba's recurrent state-space mechanism. This is not an incremental improvement to memory-based methods; it is a fundamentally different approach to temporal information preservation.

The conceptual distinction matters. Memory-based methods store *raw* past information and perform *query-time* matching: at each new frame, the model searches the memory for relevant patterns. The memory is transparent — you can inspect which past frames are stored and which are matched. AUSM's approach stores *compressed* past information and performs *no explicit matching*: the Mamba state is an opaque learned summary, and the History Decoder's cross-attention retrieves information implicitly. The memory is not inspectable, but it is learned end-to-end for the task.

The tradeoffs are instructive. Explicit memory provides interpretability and allows the model to "look back" at any past frame with high fidelity, but it scales poorly (linear growth with video length) and requires heuristic eviction policies. Compressed state provides constant memory regardless of video length and learns what to preserve automatically, but it is lossy — information that the Mamba state fails to encode is permanently lost.

The evidence that this tradeoff works in practice: AUSM processes arbitrarily long videos with constant memory, yet achieves competitive VOS performance (81.6 on DAVIS, 79.1 on YouTube-VOS 2019 with Swin-B) without any explicit memory buffer. The long-clip adaptation results (Figure 5) show that training on longer sequences (16 frames vs. 5) improves performance consistently — the model learns to use the compressed state more effectively with more temporal context. This suggests the compression is not a bottleneck at these sequence lengths.

This is a **architectural contribution with scaling implications**: it demonstrates that compressed temporal states can replace explicit memory buffers for video segmentation, which is a prerequisite for scaling to the very long videos (minutes to hours) that real-world applications require. Memory-based methods cannot scale to such lengths without either massive memory or aggressive forgetting; AUSM's compressed state approach can, in principle, handle arbitrary durations. The current limitation (training only on 16-frame clips due to GPU memory) is a practical constraint, not an architectural one — the framework is designed to scale.

---

### Innovation 4: Test-Time Compute Scaling for Video Segmentation via Input Repetition — A Positive Result from an Unexpected Direction

The paper includes a small but intriguing result in Section 3.4 (Table 3): repeating video frames at inference time improves segmentation performance. For COCO images processed as pseudo-videos with repetition, AP increases from 34.2 (single pass) to 34.9 (2× repetition) to 35.0 (3×). For YouTube-VIS 2019, AP increases from 62.6 (single pass) to 63.3 (2×) to 63.5 (3×).

This result is notable not for its magnitude — the gains are modest — but for what it reveals about the model's behavior. AUSM was not designed for iterative refinement; it was designed for streaming video where each frame is processed once. Yet the autoregressive formulation means that when the same frame appears multiple times, the model can use its own previous predictions (from the first pass) as conditioning for the second pass — effectively "checking its work" and refining uncertain regions. This is test-time compute scaling emerging organically from the architecture, not from a purpose-built refinement mechanism.

The qualitative behavior is significant: the model uses repeated observations to refine predictions rather than simply reproducing them. If the model were merely identifying objects and outputting the same result, repetition would not help. The improvement suggests the model's predictions are path-dependent and that additional passes through the autoregressive loop can disambiguate uncertain cases.

The spatial traversal experiment (Appendix A) reinforces this: processing overlapping image quadrants (upper-left, upper-right, bottom-left, bottom-right) before the full image improves COCO AP from 34.2 to 35.9 — a larger gain than simple repetition. This suggests the model benefits from seeing local regions at high resolution before integrating them into a global prediction, analogous to how humans might examine details before forming a holistic judgment.

This is a **behavioral insight** rather than a methodological contribution: it demonstrates that the autoregressive formulation provides an unexpected capability (test-time refinement) that the model was never explicitly trained to do. The connection to language model phenomena (chain-of-thought, input repetition improving embeddings) suggests a deeper structural property: autoregressive models, regardless of modality, can learn to use additional computation at inference time to improve output quality. The paper does not develop this into a full method, but it identifies a research direction — can we design *inference-time compute allocation strategies* for video segmentation, analogous to what the LLM scaling literature has explored? This is a conceptual bridge between two research communities that had not previously been connected.

## 5. Experimental Analysis

### Evaluation Methodology

- **Datasets.** The paper evaluates AUSM across seven benchmarks spanning both prompted and unprompted video segmentation. For prompted tasks: DAVIS 2017 (Pont-Tuset et al., 2017, 30 validation videos), YouTube-VOS 2018 (Xu et al., 2018, 474 validation videos) and 2019 (Xu et al., 2018, with 507 validation videos), and MOSE (Ding et al., 2023, a multi-object benchmark with complex instance interactions). For unprompted tasks: YouTube-VIS 2019 (Yang et al., 2019, 40 validation videos) and 2021 (Yang et al., 2019, 61 validation videos), and OVIS (Qi et al., 2022, 25 validation videos featuring heavy occlusion and long videos). Training data spans COCO (Lin et al., 2014, pseudo-video pretraining), DAVIS 2017 training split, MOSE, SA-V (Ravi et al., 2025), YouTube-VIS 2019 & 2021, and OVIS.
- **Base model(s).** AUSM is evaluated with two backbone scales: Swin-T (tiny) and Swin-B (base) (Liu et al., 2021). Both are ImageNet-pretrained. The model is initialized from Mask2Former (Cheng et al., 2022) weights pretrained on COCO image instance segmentation. The choice of Swin backbones follows the convention in video segmentation literature and enables direct comparison with prior universal models (UNINEXT, UniVS, TarViS) and specialized methods that use the same or comparable backbones.
- **Metrics.** For prompted segmentation, DAVIS and MOSE use the standard $\mathcal{J} \& \mathcal{F}$ metric, computed as the average of region similarity (Jaccard index $\mathcal{J}$, measuring mask overlap) and contour accuracy ($\mathcal{F}$, measuring boundary agreement). YouTube-VOS uses $\mathcal{G}$, the average of $\mathcal{J} \& \mathcal{F}$ computed separately over "seen" categories (those present in training) and "unseen" categories (novel at test time). For unprompted segmentation, YouTube-VIS 2019, YouTube-VIS 2021, and OVIS report Average Precision (AP), following standard COCO-style evaluation with mask IoU thresholds from 0.50 to 0.95.
- **Baselines.** The paper compares against three categories of methods:
  - **Task-specialized prompted models:** XMem (Cheng and Schwing, 2022), DeAOT (Yang and Yang, 2022), SAM2 (Ravi et al., 2025), and UniRef++ (Wu et al., 2023) — none of which can perform unprompted tasks.
  - **Task-specialized unprompted models:** GenVIS (Heo et al., 2023), DVIS (Zhang et al., 2023), VISAGE (Kim et al., 2024), and Video K-Net (Li et al., 2022) — none of which can perform prompted tasks.
  - **Universal models:** TarViS (Athar et al., 2023, offline), UNINEXT (Yan et al., 2023, streaming), and UniVS (Li et al., 2024, streaming) — the most direct competitors.
- **Generation budget / compute accounting.** Video segmentation does not use a generation budget in the LLM sense. The relevant compute axes are: (1) backbone scale (Swin-T vs. Swin-B, affecting FLOPs per frame), (2) sequence length during training (5 vs. 16 frames, affecting total FLOPs per iteration), and (3) inference mode (single-pass vs. repeated sequence, analogous to test-time compute scaling). The parallel training speedup is measured in wall-clock seconds per iteration (s/iter) on identical hardware (A100 GPUs), which serves as the compute accounting for training efficiency. For the test-time compute scaling experiments (Table 3), the "budget" is the number of sequence repetitions or the number of spatially decomposed views.
- **Cross-validation / statistical protocol.** There is no cross-validation or statistical significance testing reported. Results are single-point evaluations on standard benchmark test/validation splits. The foreground threshold for unprompted inference is fixed at 0.5 across all benchmarks to avoid dataset-specific tuning (Table 2 shows robustness across thresholds). The test-time compute scaling uses specific fixed repetition counts and spatial traversal configurations without sweeping.

### Main Quantitative Results

#### Prompted Video Segmentation: Closing the Gap with Specialized Models

The headline result for prompted segmentation (Table 1): AUSM with Swin-B achieves 81.6 $\mathcal{J} \& \mathcal{F}$ on DAVIS 2017, 62.1 on MOSE, 80.2 $\mathcal{G}$ on YouTube-VOS 2018, and 79.1 $\mathcal{G}$ on YouTube-VOS 2019. These numbers position AUSM firmly above all prior universal streaming models while acknowledging a meaningful gap to the strongest specialized methods.

Compared to the most relevant universal streaming baseline, UniVS, the improvement is substantial: on YouTube-VOS 2018, AUSM Swin-B achieves 80.2 $\mathcal{G}$ versus UniVS Swin-L at 71.5 — a gain of **8.7 points despite using a smaller backbone** (Swin-B vs. Swin-L). On DAVIS 2017, AUSM Swin-B at 81.6 surpasses UniVS Swin-L at 76.2 by 5.4 points. Against UNINEXT, AUSM Swin-T at 76.4 already exceeds UNINEXT ConvNeXt-L at 77.2 by — wait, re-reading: UNINEXT ConvNeXt-L achieves 77.2 on DAVIS, while AUSM Swin-T achieves 76.4, which is actually slightly *lower*. However, AUSM Swin-B at 81.6 substantially exceeds UNINEXT's best. On YouTube-VOS 2018, AUSM Swin-T (79.5) already outperforms UNINEXT ConvNeXt-L (78.1). The pattern is that AUSM with a small backbone is competitive with prior universal models using large backbones, and AUSM with a medium backbone substantially exceeds them.

The gap to specialized methods is instructive. SAM2 Hiera-B+ achieves 90.2 on DAVIS — 8.6 points above AUSM Swin-B. SAM2 Hiera-L reaches 90.7. On MOSE, SAM2 Hiera-B+ scores 76.6 versus AUSM's 62.1, a 14.5-point gap. On YouTube-VOS 2019, SAM2 Hiera-B+ scores 88.6 versus AUSM Swin-B at 79.1. These gaps are substantial, and the paper explicitly attributes them to the feature stride choice in their Discussion (Section 5): AUSM uses stride-8 features throughout its History Compressor and Decoder, while specialized VOS methods like SAM2 operate at stride-4, preserving finer spatial detail at the cost of higher memory consumption. The paper frames this as a deliberate tradeoff — "better suited for object-level understanding, but marginally worse in capturing details" — and suggests future work on video-specialized backbones could close the gap.

#### Unprompted Video Segmentation: Competitive with Specialized Methods While Remaining Universal

On YouTube-VIS 2019, AUSM Swin-B achieves 62.6 AP. This is competitive with specialized VIS models: GenVIS Swin-L at 64.0, DVIS Swin-L at 63.9, VISAGE Swin-L at 64.2. The gap to the best specialized method is ~1.4–1.6 AP points while using a smaller backbone (Swin-B vs. Swin-L). On YouTube-VIS 2021, AUSM Swin-B scores 58.6 AP versus GenVIS at 59.6, DVIS at 58.7, and VISAGE at 59.6 — again within ~1 point of the specialized state-of-the-art.

On OVIS, which features heavy occlusion and long videos, AUSM Swin-B achieves 45.5 AP. Among universal models, this is the highest reported score — UniVS Swin-L reaches 41.7, UNINEXT ConvNeXt-L reaches 41.1, and TarViS Swin-L reaches 43.2. Specialized VIS models score higher: VISAGE Swin-L at 46.5, DVIS Swin-L at 47.1, GenVIS Swin-L at 45.2. AUSM Swin-B is within 0.7–1.6 points of these specialized methods, again with a smaller backbone.

The significance of these unprompted results is that AUSM achieves near-specialist performance on VIS benchmarks *while also handling prompted VOS* — something none of the specialized VIS models can do. GenVIS, DVIS, and VISAGE have an "✗" in the prompted columns of Table 1 because their architectures fundamentally cannot accept initial prompts and track specified objects. AUSM gains universality at a cost of only 1–2 AP points on the VIS leaderboard, which is substantially smaller than the 5–10 point drops seen in prior universal models on VOS.

#### Universal Model Comparison: The Full Picture

Looking holistically across all seven benchmarks, AUSM is the only streaming universal model that achieves strong performance on both prompted and unprompted tasks. UNINEXT shows lower prompted performance (DAVIS 77.2 with ConvNeXt-L vs. AUSM Swin-B 81.6) and competitive but lower unprompted performance on YouTube-VIS 2019 (64.3 ConvNeXt-L vs. 62.6 AUSM Swin-B — noting UNINEXT uses a larger backbone). UniVS shows significantly lower prompted performance (DAVIS 76.2 Swin-L, YouTube-VOS 2018 71.5 Swin-L) and lower unprompted performance on OVIS (41.7 vs. 45.5 AUSM Swin-B). TarViS, the offline universal model, achieves competitive prompted performance (DAVIS 85.3 Swin-L, exceeding AUSM) and competitive OVIS (43.2), but its offline nature means it cannot be deployed in streaming settings and cannot benefit from autoregressive temporal feedback.

The paper notes that AUSM Swin-T (the smallest configuration) already achieves DAVIS 76.4 and YouTube-VOS 2018 79.5 — outperforming UniVS Swin-L on prompted benchmarks and UNINEXT ConvNeXt-L on YouTube-VOS, despite using a fraction of the parameters. This efficiency in the universal model space is a key empirical finding: the architectural choices (History Marker + compressed state) enable strong universality at smaller model scales.

#### Training Efficiency: Parallel Training Speedup

Figure 4 measures training time per iteration (seconds/iter) for parallel vs. iterative (recurrent) training at sequence lengths of 1, 2, 4, 8, and 16 frames, using the Swin-B backbone. The key numbers:

- At sequence length 1: both methods take 1.47 s/iter (identical — no parallelism benefit when there's only one frame).
- At length 4: iterative training takes ~3.5 s/iter, parallel takes ~2.0 s/iter — roughly 1.75× speedup.
- At length 8: iterative takes ~6.0 s/iter, parallel takes ~2.8 s/iter — roughly 2.1× speedup.
- At length 16: iterative takes 8.75 s/iter, parallel takes 3.45 s/iter — **2.5× speedup**.

The growth rates reveal the structural difference: iterative training time grows approximately linearly with sequence length (1.47 → 8.75 over 16× longer sequences is a ~5.95× increase, close to the ~6× mentioned in the paper's "6.0×" figure), while parallel training grows sub-linearly (1.47 → 3.45, a ~2.3× increase). The paper states that "larger gains expected at longer horizons," which is a reasonable extrapolation if the linear vs. sub-linear trend continues — though the paper does not test lengths beyond 16 frames due to GPU memory constraints.

This result is important because it demonstrates that the training formulation, not just the architectural design, is what enables scalability. Prior work with recurrent training (RoCoVIS, GenVIS) could not amortize computation across frames; AUSM's teacher-forcing approach does. The 2.5× speedup at 16 frames is a concrete efficiency gain that compounds when training on larger video datasets with longer clips.

#### Effect of Training on Longer Sequences

Figure 5 compares Stage 2 (5-frame training) with Stage 3 (16-frame fine-tuning) on four benchmarks: MOSE, YouTube-VOS 2018, YouTube-VOS 2019, and OVIS. The results show consistent improvement from longer temporal context:

- **MOSE**: 57.6 → 62.1 $\mathcal{J} \& \mathcal{F}$, an improvement of **+4.5 points (7.8% relative)**.
- **YouTube-VOS 2018**: 78.3 → 80.2 $\mathcal{G}$, improvement of +1.9 (2.5%).
- **YouTube-VOS 2019**: 76.0 → 79.1 $\mathcal{G}$, improvement of +3.1 (4.1%).
- **OVIS**: 40.3 → 45.5 AP, improvement of **+5.2 (12.9%)**.

The largest gains occur on the datasets with the most complex temporal dynamics: MOSE (complex multi-object interactions) and OVIS (heavy occlusion, long videos averaging ~50 seconds). YouTube-VOS benchmarks show smaller but consistent improvements. The paper emphasizes that "these improvements are achieved without any explicit memory buffer (e.g., FIFO-style spatio-temporal caches)" — the History Compressor's learned temporal compression benefits from longer training sequences, suggesting the Mamba state learns to encode more useful information when given longer temporal context to optimize over.

A notable detail: Stage 3 freezes the backbone and only trains the temporal modules and prediction heads. The +5.2 AP gain on OVIS therefore comes entirely from improved temporal processing, not from better per-frame features. This isolates the contribution of the History Compressor/Decoder architecture to long-range modeling.

#### Test-Time Compute Scaling via Input Repetition

Table 3 reports the effect of repeating the input sequence at inference time. For COCO (single images processed as pseudo-videos with repeated frames), AP increases from 34.2 (×1 repetition) to 34.9 (×2) to 35.0 (×3). For YouTube-VIS 2019, AP increases from 62.6 to 63.3 (×2) to 63.5 (×3).

The gains are modest in absolute terms — +0.8 AP on COCO, +0.9 on YouTube-VIS 2019 — but the existence of any gain from simple repetition is noteworthy. The paper also reports (Appendix A) that a more sophisticated spatial traversal strategy — decomposing each image into four overlapping quadrants and processing them sequentially before the full image — improves COCO AP from 34.2 to 35.9, a +1.7 AP gain. This suggests the autoregressive loop can leverage structured revisitation of the input more effectively than simple repetition, though the paper does not explore this beyond the single configuration described.

The paper frames this as analogous to test-time compute scaling in language models, where techniques like chain-of-thought and repeated prompting improve output quality. It is a preliminary result — only two datasets, one video and one image — but it demonstrates that the autoregressive formulation provides a mechanism for iterative refinement that was not explicitly designed or trained for, analogous to how autoregressive LMs can benefit from repeated inputs without being trained for that scenario.

#### Foreground Threshold Robustness

Table 2 sweeps the foreground confidence threshold used in `filter_fg` for unprompted segmentation from 0.3 to 0.7. On YouTube-VIS 2019, AP is stable between 61.6 and 62.6 across all thresholds, with a mild peak at 0.4–0.5 (62.6). On YouTube-VIS 2021, AP ranges from 57.8 (0.3) to 58.6 (0.5), again relatively flat. On OVIS, there is a clearer trend favoring higher thresholds: AP increases from 44.5 at 0.3 to 46.5 at 0.7 — a **+2.0 AP improvement** by being more conservative about which detections to track.

The paper notes the divergence: OVIS benefits from higher thresholds, but YouTube-VIS is optimal around 0.4–0.5. Despite this, they use a fixed threshold of 0.5 for all benchmarks "to ensure consistency and avoid dataset-specific tuning." This choice means OVIS performance could potentially be higher (+1.0 AP at threshold 0.7), but the paper prioritizes methodological simplicity over maximizing individual benchmark scores. The robustness of YouTube-VIS across the full range suggests the detection module is well-calibrated and does not produce many borderline-confidence detections that would drastically change the tracked set with small threshold changes.

### Ablation Studies and Robustness Checks

- **Backbone scale (Swin-T vs. Swin-B)**: Table 1 shows that scaling from Swin-T to Swin-B yields consistent improvements across all benchmarks: DAVIS 76.4 → 81.6 (+5.2), MOSE 58.8 → 62.1 (+3.3), YouTube-VOS 2018 79.5 → 80.2 (+0.7), YouTube-VOS 2019 78.3 → 79.1 (+0.8), YouTube-VIS 2019 54.9 → 62.6 (+7.7), YouTube-VIS 2021 52.1 → 58.6 (+6.5), OVIS 39.4 → 45.5 (+6.1). The improvements are largest on VIS benchmarks (+6–8 AP), suggesting that detection and classification benefit more from backbone capacity than mask propagation does. The YouTube-VOS gains are modest (+0.7–0.8 $\mathcal{G}$), possibly because the prompted VOS evaluation (mask propagation quality) depends more on the temporal modules and feature resolution than on backbone capacity per se.

- **Training sequence length (5 vs. 16 frames)**: Figure 5, discussed above. The +5.2 AP on OVIS and +4.5 on MOSE demonstrate that longer temporal context during training directly improves the model's ability to handle complex temporal dynamics. This is a non-trivial finding: it confirms that the History Compressor's Mamba state benefits from being optimized over longer sequences, and that the frozen-backbone fine-tuning approach is sufficient to transfer these improvements. An important limitation: the comparison is Stage 2 (32K iterations, 5-frame, frozen backbone? — actually, reading carefully: Stage 2 uses 5-frame clips with all components trainable, Stage 3 uses 16-frame clips with frozen backbone and only temporal modules trainable). The improvement from Stage 2 to Stage 3 thus confounds two changes: longer sequences and different trainable parameters. An ablation keeping all parameters trainable in Stage 3 (if memory permitted) would isolate the effect of sequence length alone.

- **Parallel vs. iterative training**: Figure 4, discussed above. The key comparison is at identical sequence lengths using identical models. The 2.5× speedup at 16 frames is measured on the same hardware with the same model architecture, changing only whether frames are processed sequentially (iterative) or in parallel via teacher forcing. This is a clean ablation of the training formulation.

- **Foreground threshold at inference**: Table 2, discussed above. Performance is largely robust across thresholds for YouTube-VIS, with OVIS showing preference for higher thresholds. The paper's choice of 0.5 for all benchmarks is a methodological decision to avoid per-dataset tuning.

- **Test-time compute scaling strategy**: Table 3 compares simple repetition (×1, ×2, ×3) against a spatial traversal strategy (Appendix A: full image → four overlapping quadrants → full image). The spatial traversal achieves +1.7 AP on COCO (34.2 → 35.9) versus +0.8 from 3× repetition (34.2 → 35.0), suggesting that *how* you allocate additional computation matters more than simply doing more passes. This is a preliminary result with only one non-standard augmentation configuration, but it hints that designing optimal inference-time computation strategies for video segmentation could be a productive direction.

- **Dataset-specific classification heads (implicit ablation)**: Appendix A notes that AUSM "employ[s] dataset-specific classification heads" and "does not use vision-language supervision (e.g., CLIP)." This means the model has separate classification parameters for each training dataset, which are conditionally selected based on the input's source. This design choice is not ablated — there is no comparison against a single unified classification head across all datasets, nor against a CLIP-based open-vocabulary approach. This is a limitation: the universality claim applies to the segmentation and tracking architecture, but the classification is dataset-dependent, which limits zero-shot generalization to new categories.

### Critical Assessment

The experiments in this paper aim to substantiate four central claims: (1) that AUSM unifies prompted and unprompted video segmentation in a single architecture, (2) that the autoregressive formulation with History Marker and History Compressor outperforms prior universal streaming models, (3) that parallel training enables substantial speedups over recurrent training, and (4) that the architecture scales to longer videos with constant memory. I'll examine each in turn.

**Claim: AUSM unifies prompted and unprompted segmentation.** Strongly supported by the architecture (Algorithms 1–2) and the qualitative results (Figures 6–7), but the quantitative evidence in Table 1 requires careful interpretation. The "unification" claim means the same model with the same weights performs both tasks. The paper states "all results of AUSM are obtained with a single model trained using our joint learning framework, without task-specific fine-tuning." This is a strong claim of weight-sharing. However, the model does use dataset-specific classification heads (Appendix A), which means the classification component is not fully unified across datasets — the model knows which dataset the input comes from during evaluation and uses the corresponding classifier. For prompted segmentation, no classification is needed (VOS has no class labels), so this is moot for VOS benchmarks. But for unprompted segmentation, the model's ability to handle novel categories from one dataset when evaluated on another is not tested. A true test of "universal" classification would involve zero-shot transfer, which is not evaluated.

**Claim: AUSM outperforms prior universal streaming models.** Supported by Table 1 with specific numbers cited throughout this section. The gains are substantial on prompted benchmarks (+8.7 on YouTube-VOS 2018 vs. UniVS Swin-L) and competitive-to-better on unprompted benchmarks. The evidence is strongest for the Swin-B configuration across all seven benchmarks. The Swin-T results are competitive but not dominant — on DAVIS, AUSM Swin-T (76.4) is slightly below UNINEXT ConvNeXt-L (77.2), though the backbone comparison is asymmetric. A limitation: the paper does not compare against UniVS with the same backbone and training data. UniVS was trained on a different data mixture, and some of AUSM's gains could come from the joint training on additional datasets (SA-V, COCO-pseudo) rather than from the architecture. An ablation isolating architectural contributions from data contributions would require training both models on identical data, which is not done.

**Claim: Parallel training is 2.5× faster at 16 frames.** Supported by Figure 4 with clean methodology — same model, same hardware, same sequence lengths, only the training formulation differs. The extrapolation to longer sequences ("larger gains expected at longer horizons") is reasonable but untested. An important caveat: the parallel training uses teacher forcing with ground-truth intermediate targets, while the iterative training uses model-generated intermediate targets (or would, in a truly recurrent setup). The paper does not clarify what the iterative baseline actually does at training time — does it use ground-truth or predicted masks as conditioning? If the iterative baseline also uses teacher forcing but processes frames one at a time, the comparison is purely about computational parallelism. If the iterative baseline uses predicted masks, the comparison confounds parallelism with a difference in training signal quality. This matters because the paper's claim about "compatibility with teacher forcing" being an architectural contribution implies that prior methods *cannot* do teacher forcing, but if the iterative baseline also uses ground-truth conditioning frame-by-frame, then the architectural contribution is about enabling parallel computation, not about enabling teacher forcing per se. The paper's text in Section 2.3 implies that prior methods *must* process recurrently because they condition on model outputs, but the specifics of the iterative baseline's training setup are not described in sufficient detail to assess this.

**Claim: Constant-memory inference on arbitrarily long videos.** The architecture supports this by design — the Mamba state has fixed dimensionality regardless of `$t$`. However, the paper does not empirically evaluate performance degradation on very long videos. The longest training clips are 16 frames (Stage 3). The OVIS benchmark has videos averaging ~50 seconds (likely hundreds of frames), and AUSM performs well on it (45.5 AP), which provides some evidence that the compressed state generalizes beyond training sequence length. But there is no systematic evaluation of performance as a function of video length — no plot showing accuracy vs. frame index for long videos, no needle-in-a-haystack style evaluation, no comparison of memory usage against STM-based methods at equivalent lengths. The "arbitrarily long" capability is therefore an architectural property demonstrated only indirectly through design, not through empirical scaling tests.

**Missing experiments.** Several experiments would have strengthened the paper substantially. (1) A comparison against SAM2 *if SAM2 were adapted for VIS* (e.g., running it per-category and merging tracks) to establish an upper bound on what memory-based methods could achieve on unprompted tasks. (2) An ablation isolating the History Marker's contribution by replacing it with a simple vectorization (as in UniVS) while keeping the rest of AUSM's architecture — this would directly test the "vectorization bottleneck" hypothesis. (3) Training AUSM and UniVS on identical data with identical backbones to isolate architectural contributions from training data advantages. (4) Evaluation of performance degradation on videos longer than the training clip length (e.g., test on 32-frame clips after 16-frame training). (5) Memory usage and wall-clock inference time comparisons against SAM2 and XMem on long videos. (6) An ablation of the detection-to-tracking transition sampling strategy (random vs. fixed-early vs. fixed-late) to validate the Preprocess function's design.

**The test-time compute scaling results (Table 3) are underexplored.** The gains are small (+0.8–0.9 AP) and only tested on two benchmarks. There is no sweep of repetition counts beyond 3×, no investigation of *why* repetition helps (is it refining boundaries? recovering missed detections? improving classification confidence?), no comparison to simply using a larger backbone or longer training, and no test of whether the gains saturate or continue with more repetitions. The spatial traversal result (quadrants) is described only in Appendix A with one configuration. This section is presented as a connection to the LLM scaling literature but is too preliminary to establish any meaningful parallel.

**Single model family limitation.** All results use Swin Transformers, which is standard in video segmentation but limits generality. The paper does not test with ConvNeXt, ViT, or Hiera backbones, so the question of whether the History Compressor + History Marker design transfers across backbone architectures is open. The SAM2 comparison is asymmetric — SAM2 uses a custom Hiera backbone trained on additional private data, making it difficult to attribute the performance gap to architecture vs. backbone vs. data scale.

**Dataset-specific classification heads limit universality claims.** The paper acknowledges this design choice but does not treat it as a limitation. A truly universal model should handle novel categories without dataset-specific parameters. This is a narrower form of universality than the paper's framing suggests — the model is "universal" with respect to task structure (prompted vs. unprompted, tracking vs. detection) but not with respect to semantic categories.

**The feature stride tradeoff is real and quantified.** The gap to SAM2 on prompted VOS (8.6 points on DAVIS) is attributed to the coarser feature stride (8 vs. 4). This is a credible explanation — VOS evaluation heavily weights boundary accuracy ($\mathcal{F}$), which depends on spatial resolution — but there is no ablation varying feature stride within AUSM to confirm the magnitude of this effect. It is possible that other architectural differences (memory-based dense matching vs. compressed state, per-object vs. joint processing) also contribute to the gap.

**Bottom line:** The experimental evidence strongly supports AUSM's position as the best universal streaming video segmentation model at the time of writing, with clear and substantial gains over prior universal methods on prompted benchmarks and competitive performance on unprompted benchmarks. The parallel training speedup is convincingly demonstrated within the tested range (up to 16 frames). The constant-memory and arbitrary-length claims are architecturally supported but not empirically stress-tested. The universality claim is valid for task structure but qualified by dataset-specific classification heads. The paper's framing as a "step toward a unified, scalable, and general-purpose formulation" is appropriate — it demonstrates that the autoregressive approach works and provides architectural mechanisms that resolve specific bottlenecks in prior universal models, but it does not deliver a fully general open-vocabulary system nor empirically validate the scaling limits of the compressed state approach.

## 6. Limitations and Trade-offs

### 6.1 The History Compressor's Compressed State Is Not Empirically Stress-Tested on Arbitrarily Long Videos

**The assumption or constraint.** The paper's central architectural claim is that the History Compressor enables "processing arbitrarily long streams" with "constant memory" by compressing all past temporal information into a single Mamba state `$F_t$` of fixed dimensionality (Section 2.2). The paper states explicitly:

> "our design makes processing arbitrarily long streams feasible"

and contrasts this with STM-based methods that "typically store fewer than ten frame features" (Section 1). However, all training is conducted on clips of at most 16 frames (Stage 3, Section 3.2), and the model never sees sequences longer than this during optimization.

**The consequence.** State-space models like Mamba are known to struggle with length generalization — the compressed state is trained to summarize information over the training sequence length, and there is no guarantee that it will preserve critical information over substantially longer horizons. For real-world deployment on videos lasting minutes or hours (hundreds to thousands of frames), the Mamba state may silently forget early objects, fail to track instances that leave and re-enter the frame, or accumulate errors in the compressed representation that degrade mask quality. The "constant memory" property is architecturally guaranteed, but *constant fidelity* is not. A practitioner deploying AUSM on long surveillance footage or untrimmed YouTube videos would have no empirical basis to predict when or whether tracking quality will degrade with video length.

The paper's own Discussion (Section 5) acknowledges this:

> "similar to LLMs, we observe performance degradation on extremely long sequences"

but this observation is not supported by any experiment in the paper — no results on videos longer than the training clip length are reported. The reader is told degradation occurs but is given no characterization of its onset, severity, or dependence on video content.

**What evidence exists in the paper.** The only indirect evidence is OVIS performance (45.5 AP, Table 1). OVIS videos average ~50 seconds, which at typical frame rates corresponds to hundreds of frames — well beyond the 16-frame training horizon. The fact that AUSM performs competitively on OVIS suggests the compressed state generalizes to some extent, but OVIS evaluation averages over entire videos; it does not reveal whether performance degrades in later frames. There is no plot of accuracy vs. frame index for long videos, no comparison of early-frame vs. late-frame tracking quality, and no ablation varying test sequence length while measuring performance. The paper also reports no memory usage measurements (GPU memory, RAM) during inference on long videos to validate the "constant memory" claim quantitatively.

**Mitigation status.** The paper does not attempt to mitigate this limitation empirically. The Discussion mentions that "recent long-sequence techniques from language modeling could be adapted to video to maintain quality and extend the effective context beyond the training sequence length" (Section 5), but no such techniques are implemented or evaluated. The limitation is acknowledged at the conceptual level but left entirely to future work. The next-generation hardware the paper references as enabling longer training sequences (Section 5) would address the training-side bottleneck but does not resolve the fundamental question of whether the Mamba state's representational capacity is sufficient for very long videos even if training sequences were longer.

---

### 6.2 Dataset-Specific Classification Heads Restrict Universality to Task Structure, Not Semantic Openness

**The assumption or constraint.** The paper presents AUSM as a "universal video segmentation model" that "unifies both prompted and unprompted video segmentation tasks using shared weights" (Abstract). However, Appendix A reveals that AUSM uses **dataset-specific classification heads** — separate classifier parameters for each training dataset, conditionally selected based on the input's source:

> "We employ dataset-specific classification heads. During training, each head is conditionally selected according to the dataset from which the input sample originates."

The model also "does not use vision-language supervision (e.g., CLIP or other pretrained image-text models)" (Appendix A).

**The consequence.** AUSM is universal with respect to *task structure* (it can do both VOS and VIS, both tracking and detection) but not with respect to *semantic categories* (it cannot handle novel object classes without retraining the classification head). Concretely: if a user deploys AUSM on a video containing object categories that were not in the training datasets' label spaces, the model has no mechanism to classify them — it might segment and track them, but it cannot name them. Similarly, if a user wants to transfer the model to a domain with a different taxonomy (e.g., medical imaging, industrial inspection), the entire system would need a new classification head trained on that domain's labels, even though the segmentation and tracking components might generalize.

This is a narrower form of "universality" than the framing in the abstract and introduction suggests. The language modeling analogy that motivates the paper — "a single scalable architecture trained on massive corpora can subsume diverse tasks" (Section 1) — implies that a single model handles diverse inputs without per-task switching. AUSM switches classification heads based on dataset identity, which is conceptually similar to having task-specific output layers rather than a single unified output space. For prompted VOS evaluation this is irrelevant (no classification is needed), but for unprompted VIS, the reported numbers depend on knowing which dataset the test video comes from.

A practitioner deploying AUSM in a real-world setting where the input domain does not exactly match one of the training datasets would face an ambiguity: which classification head to use? Using the "wrong" head would produce nonsensical class labels. Using no head at all would require an alternative classification mechanism not provided by the paper.

**What evidence exists in the paper.** There is no ablation comparing dataset-specific heads against a single unified classification head trained jointly on the union of all dataset categories. There is no evaluation of zero-shot transfer to unseen categories (e.g., training on COCO + YouTube-VIS and testing classification on OVIS categories without OVIS-specific training). The paper reports no experiments on open-vocabulary segmentation where the model must classify objects into categories not seen during training. The classification component is essentially an unexamined module — the paper's contributions focus entirely on segmentation and tracking architecture, leaving classification as a dataset-specific add-on.

**Mitigation status.** The paper does not address this limitation. The Discussion mentions extending AUSM to referring video object segmentation by initializing the History Compressor's state with text embeddings from a frozen text encoder (Section 5), which would move toward language-driven classification. But this is proposed as future work, not implemented. The absence of vision-language supervision is presented as an implementation detail rather than a limitation, suggesting the authors do not consider dataset-specific classification heads to contradict the universality claim. A reader expecting a truly open-vocabulary universal model — in the spirit of SAM or SAM2's class-agnostic design, or CLIP-based open-vocabulary detectors — will find this gap significant.

---

### 6.3 The Feature Stride Tradeoff Imposes a Structural Disadvantage on Prompted Segmentation That the Architecture Does Not Resolve

**The assumption or constraint.** AUSM operates on backbone features at **1/8 spatial resolution** (Appendix A: "we use the 1/8 resolution feature map from the Swin backbone"). Specialized VOS methods like SAM2 operate at stride 4 (finer resolution), which preserves more spatial detail for precise mask boundary prediction. The paper explicitly acknowledges this:

> "most modules of AUSM take coarse frame features (e.g. stride of 8), which saves memory than finer features (e.g., stride of 4) and is better suited for object-level understanding, but marginally worse in capturing details" (Section 5, Limitations).

**The consequence.** The 8.6-point gap between AUSM Swin-B (81.6 J&F) and SAM2 Hiera-B+ (90.2 J&F) on DAVIS 2017 (Table 1) is attributed primarily to this resolution difference. The J&F metric averages region similarity (J) and contour accuracy (F), both of which are directly sensitive to spatial resolution — coarser features produce coarser mask boundaries, which penalizes F directly and reduces J through less precise boundary placement. On MOSE, the gap is larger (62.1 vs. 76.6, 14.5 points), which the paper does not explain by resolution alone but which likely compounds with MOSE's focus on complex multi-object interactions where fine boundaries matter.

The practical implication: AUSM cannot match specialized VOS methods on benchmark leaderboards unless this resolution gap is closed. For applications where precise mask boundaries are critical — rotoscoping, professional video editing, medical image segmentation — AUSM at its current feature resolution may be insufficient regardless of architectural improvements to the temporal components. The paper frames this as a design choice (saving memory for longer sequences) rather than a fundamental limitation, but a practitioner must accept this quality tradeoff if they choose AUSM over SAM2 for prompted segmentation tasks.

**What evidence exists in the paper.** Table 1 provides the quantitative comparison. The gap to SAM2 is substantial and consistent across all prompted benchmarks. However, there is **no ablation within AUSM** that varies feature stride (e.g., testing stride 4 vs. stride 8 for the History Compressor and Decoder) to isolate how much of the gap is due to resolution alone vs. other architectural differences (compressed state vs. explicit memory, per-object vs. joint processing, training data differences). The paper asserts the resolution difference as the primary cause but provides no controlled experiment to support this claim. The 14.5-point gap on MOSE is particularly large and may indicate factors beyond resolution — MOSE features complex multi-object scenes where AUSM's joint processing could be an advantage, yet SAM2's per-object memory approach still substantially outperforms.

**Mitigation status.** The paper suggests future work on "a new video-specialized backbone to temporal modeling (e.g., reducing frame-independent layers while strengthening frame-dependent modules such as History Compressor/Decoder and prompt conditioning)" (Section 5, Limitations). This proposal reallocates backbone capacity from spatial processing to temporal processing rather than simply increasing resolution, which is an interesting idea but entirely unimplemented. The paper does not report memory usage at stride 4 to characterize how much more expensive higher resolution would be, making it difficult for practitioners to assess whether the resolution-accuracy tradeoff is worth the cost in their deployment context.

---

### 6.4 The Parallel Training Speedup Is Demonstrated Only on a Specific Hardware Configuration at Modest Sequence Lengths, with an Underspecified Baseline

**The assumption or constraint.** Figure 4 reports training time per iteration for parallel vs. iterative training at sequence lengths up to 16 frames using a Swin-B backbone on what is presumably 16 NVIDIA A100 GPUs (the training configuration specified in Appendix A). The paper claims "up to 2.5× faster training on 16-frame sequences" (Abstract) and extrapolates "larger gains expected at longer horizons."

**The consequence.** The generality of the training speedup claim has several unexamined dimensions. First, **hardware dependence**: the relative speedup of parallel vs. iterative training depends on GPU memory bandwidth, communication overhead, and the degree to which the parallel implementation can saturate the available compute. On different hardware (fewer GPUs, different GPU architectures, inference-focused hardware), the speedup curve could differ substantially. A practitioner with a 4-GPU setup may not see the same 2.5× speedup.

Second, **scaling extrapolation**: the paper extrapolates to longer sequences without data. Mamba-based associative scans for parallel training have computational complexity that scales with sequence length (Gu and Dao, 2023), and the History Decoder's cross-attention over compressed states also has a cost that scales with the number of spatial tokens. Whether the parallel advantage grows, plateaus, or reverses at 32, 64, or 128 frames is unknown. The "larger gains expected" claim is based on the trend from 1–16 frames but could be invalidated if the associative scan overhead becomes the bottleneck.

Third, **the iterative baseline is underspecified.** The paper states that existing frameworks "recurrently process frames, leading to severely inefficient training" (Section 2.3) and compares against an "iterative training" baseline in Figure 4. But the paper does not describe what the iterative baseline actually does: does it use teacher forcing (ground-truth masks as conditioning) processed sequentially, or does it use model-predicted masks? If the iterative baseline also uses teacher forcing frame-by-frame, then the measured speedup is purely from parallelizing independent computation — the training *signal* is identical and the comparison is about hardware utilization. If the iterative baseline uses predicted masks (as prior work like RoCoVIS does), then the comparison confounds parallelism with a difference in training signal quality, and the 2.5× speedup overstates the advantage of AUSM's architecture over a hypothetical iterative baseline that also uses teacher forcing. The paper states that prior methods "rely on frame-by-frame propagation of outputs" (Section 2.3) and that "this parallel training cannot be readily applied to existing video segmentation methods," which suggests the iterative baseline uses predicted outputs — but the specific implementation used for Figure 4 is not described.

**What evidence exists in the paper.** Only Figure 4, which shows wall-clock time per iteration for the two approaches at five sequence lengths. There is no throughput analysis (frames processed per second), no memory usage comparison, no scaling test on different hardware configurations, and no breakdown of where time is spent in each approach (forward pass, loss computation, communication). The paper does not report whether the 2.5× speedup translates proportionally to total training time (it would if iteration count is the same, but Stage 3 with 16 frames uses 40K iterations while Stage 2 with 5 frames uses 32K iterations — the interaction of sequence length and iteration count is not analyzed).

**Mitigation status.** The paper does not address these concerns. The speedup is reported as a single-number result without caveats about hardware dependence or baseline specification. The extrapolation to longer horizons is stated as expectation without qualification. For a result that is central to the paper's scalability narrative, the empirical characterization is thin.

---

### 6.5 No Ablation Isolates the History Marker's Contribution from the Training Data and Backbone Differences Relative to Prior Universal Models

**The assumption or constraint.** The paper attributes AUSM's large gains over prior universal models on prompted VOS benchmarks to the History Marker — specifically, its preservation of spatial instance information that prior methods lose through vectorization. The abstract states that History Marker "demonstrates a nearly 10% improvement in VOS performance compared to previous unified online architectures." This claim is supported by comparing AUSM's numbers against UniVS's numbers in Table 1 (e.g., AUSM Swin-B 81.6 on DAVIS vs. UniVS Swin-B 75.0, a 6.6-point absolute improvement, and AUSM Swin-T 76.4 vs. UniVS Swin-T 71.7, a 4.7-point improvement — note that "nearly 10%" appears to reference relative improvement or a specific benchmark rather than the DAVIS numbers).

**The consequence.** The comparison between AUSM and UniVS confounds multiple variables beyond the History Marker: (1) AUSM is trained on additional datasets (SA-V, MOSE, COCO-pseudo, OVIS) that UniVS may not have used; (2) AUSM uses a three-stage curriculum with pseudo-video pretraining and long-clip adaptation, while UniVS's training protocol is different; (3) AUSM uses dataset-specific classification heads while UniVS may use a different classification strategy; (4) AUSM's Mamba-based History Compressor and its temporal compression mechanism are fundamentally different from UniVS's approach of using previous-frame masks as visual prompts. Without an ablation where the History Marker is replaced by a vectorized representation (e.g., masked average pooling of instance features into a single vector) within AUSM's own architecture — while keeping all other components identical — it is impossible to attribute the performance gain to the History Marker specifically rather than to training data, training protocol, or the Mamba-based temporal compressor.

A practitioner trying to decide whether the History Marker is worth implementing in their own architecture needs to know: is the VOS improvement from the spatial mask dissolution per se, or from the Mamba state-space compression, or from the joint training on additional data? The paper's "nearly 10%" framing implies the History Marker is the key factor, but the evidence for this causal claim is correlational (comparison across different models with different training recipes), not interventional (within-model ablation).

**What evidence exists in the paper.** The paper provides no ablation of the History Marker. There is no experiment where the History Marker is replaced with an alternative instance representation while keeping the rest of AUSM fixed. The closest thing to an ablation is the comparison to UniVS in Table 1, which is a comparison across different models, not a controlled experiment within AUSM. The paper also does not provide an ablation of the Mamba-based History Compressor against a Transformer-based temporal module, or against a simple temporal convolution baseline — such an ablation would help the reader understand how much each architectural component contributes to the final performance.

**Mitigation status.** Not addressed. The paper's narrative strongly attributes the VOS improvements to the History Marker's spatial preservation, but the experimental design does not support causal attribution. This is a significant gap for a paper whose primary architectural contribution is the History Marker + History Compressor design — the reader cannot determine which of these two components (or their interaction) drives the empirical gains, nor whether a simpler approach (e.g., concatenating previous-frame masks to current-frame features without ID vector painting) would achieve similar results.

---

### 6.6 The Revision Model's Correct-to-Incorrect Reversion Rate Has No Analog in the Segmentation Domain, Limiting the LLM Analogy

**The assumption or constraint.** The paper draws a strong conceptual parallel between AUSM's autoregressive formulation and decoder-only LLMs. However, a well-documented issue with autoregressive refinement in language models — the tendency to "revise" correct outputs into incorrect ones when iterative self-correction is attempted — has no documented analog for AUSM. In video segmentation, this would manifest as the model correctly tracking an object in frame t, then "correcting" that correct mask into a worse mask in frame t+1 based on its own previous output as conditioning. The paper provides no analysis of error propagation or tracking drift — how often does a correct prediction lead to a subsequent incorrect prediction due to autoregressive conditioning on model outputs rather than ground truth?

**The consequence.** The autoregressive formulation introduces a train-test mismatch: during training, the model conditions on ground-truth masks (via teacher forcing); during inference, it conditions on its own predicted masks from previous frames. If these predicted masks contain errors (incorrect boundaries, missed instances, identity switches), those errors become part of the conditioning for the next frame, potentially compounding. This is the standard exposure bias problem in autoregressive models. The paper's strong results on VOS and VIS benchmarks suggest that this compounding is not catastrophic at the evaluated video lengths, but there is no characterization of *when* or *how quickly* tracking drifts when errors occur. A practitioner deploying AUSM for safety-critical applications (autonomous driving, surgical video) would need to understand failure modes: does a single missed detection cause the tracker to lose the object permanently, or does the detection branch recover it later? Does mask boundary drift accumulate over time, or is it bounded?

The test-time compute scaling results (Table 3) provide an indirect hint: repeating frames improves performance, suggesting that the model can recover from or refine its own previous predictions, which implies that errors do not necessarily compound irreversibly. But this is a positive signal from a limited experiment, not a systematic failure analysis.

**What evidence exists in the paper.** None. The paper does not measure tracking drift, identity consistency over time (beyond the standard benchmark metrics which average over entire videos), or error propagation patterns. There is no analysis of whether performance degrades in later frames of a video relative to earlier frames, which would directly test for autoregressive error accumulation. The OVIS results (45.5 AP) are notable because OVIS features heavy occlusion — a scenario where autoregressive conditioning on erroneous masks would be most damaging — but the paper does not break down OVIS performance by occlusion level or frame position.

**Mitigation status.** Not addressed. The paper does not discuss exposure bias, error propagation, or train-test mismatch as potential issues. The teacher-forcing training is presented purely as an efficiency advantage (enabling parallel training) without discussion of whether the gap between ground-truth conditioning (training) and predicted-mask conditioning (inference) affects performance. The Discussion mentions length extrapolation techniques from language modeling (Press et al., 2021; Xiao et al., 2023) as relevant future work, but this is in the context of handling longer sequences rather than addressing autoregressive error accumulation. A systematic study of error propagation in autoregressive video segmentation would be a natural follow-up but is absent from the current paper.

## 7. Implications and Future Directions
- Impact on the field
  - Establishes a principled LLM-style framing for streaming video segmentation, enabling unified architectures that cover both prompted and unprompted modes. This reduces the need for separate pipelines and opens the door to scalable, general-purpose video perception systems (Sec. 6 Conclusion).
- Research directions
  - Long-sequence methods from LLMs (e.g., retrieval for long contexts, length extrapolation) adapted to video to stabilize extremely long horizons (Discussion).
  - Video-specialized backbones emphasizing temporal modeling and prompt conditioning to close the VOS gap without sacrificing VIS performance (Discussion “Limitations”).
  - Extending the unified interface to more tasks:
    - Multi-object tracking by converting boxes/points/scribbles to mask prompts.
    - Referring VOS by initializing the history state with text embeddings (Discussion).
- Practical applications
  - Interactive editing and content creation (prompted mode), autonomous perception (unprompted), AR/VR object understanding, long-duration surveillance/robotics where constant-memory streaming and parallel training are decisive.
  - The test-time compute scaling trick (Sec. 3.4; Table 3, App. A Eq. 3) offers an easy accuracy knob—repeat frames to refine results—useful when extra latency is acceptable.

Overall, AUSM delivers a compelling synthesis: a language-model-inspired, streamable, and trainable-at-scale framework that meaningfully advances universal video segmentation. The architecture’s `History Marker` and `History Compressor` are the critical mechanisms that preserve spatial detail and compress temporal context, respectively, while the training pipeline achieves the efficiency gains needed to make long-sequence learning practical.
