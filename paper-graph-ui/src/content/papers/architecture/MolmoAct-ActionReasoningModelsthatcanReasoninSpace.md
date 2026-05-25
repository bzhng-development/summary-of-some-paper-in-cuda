# MolmoAct: Action Reasoning Models that can Reason in Space

**ArXiv:** [2508.07917](https://arxiv.org/abs/2508.07917)

## 🎯 Pitch

MolmoAct introduces a new paradigm for robotic foundation models by explicitly factoring perception, planning, and control into a three-stage action reasoning pipeline: first predicting depth-aware perception tokens, then sketching a visual trajectory trace, and finally outputting low-level actions. This structure enables robots to perform explainable and steerable behaviors, dramatically improving generalization, interpretability, and real-world robustness over prior end-to-end or language-only reasoning approaches. As an open, state-of-the-art Action Reasoning Model, MolmoAct sets a new foundation for building trustworthy, adaptive, and human-guidable robotic agents.

---

## 1. Executive Summary

This paper introduces **MolmoAct**, a family of fully open **Action Reasoning Models (ARMs)** that integrate perception, planning, and control through a structured three-stage autoregressive pipeline—encoding observations into depth-aware perception tokens, generating mid-level spatial plans as editable visual reasoning traces, and predicting precise low-level actions—enabling explainable and steerable robotic behavior. Evaluated on the SimplerEnv, LIBERO, and real-world Franka manipulation benchmarks, MolmoAct-7B-D achieves 70.5% zero-shot accuracy on SimplerEnv Visual Matching (surpassing closed-source π0 and GR00T N1.5), 86.6% average success on LIBERO (including a +6.3% gain over ThinkAct on long-horizon tasks), and +10% to +22.7% task progression gains over π0-FAST in real-world fine-tuning. The model also outperforms baselines by +23.3% on out-of-distribution generalization and achieves top human-preference Elo scores for open-ended instruction following and trajectory steering, establishing that spatially grounded reasoning chains—rather than scaling pretraining data alone—enable strong generalization and adaptable control, though the hardest problems remain outside the base model's capability range regardless of test-time compute.

## 2. Context and Motivation

### The Core Problem: VLAs Are Brittle Because They Don't Reason

The fundamental problem this paper addresses is deceptively simple: **most robotic foundation models map perception and instructions directly to motor commands without any intermediate reasoning**. A Vision-Language-Action (VLA) model typically takes an image and a language instruction as input and predicts joint angles or end-effector poses as output — end of story. This monolithic approach has produced impressive demonstrations but suffers from three interconnected weaknesses that limit real-world deployability:

1. **Opaqueness**: When the robot fails (grasps the wrong object, misses the target, or veers off course), there is no insight into *why* it chose that action. The mapping from pixels to motor commands is a black box, making diagnosis and improvement difficult.

2. **Inflexibility**: A direct perception-to-action model learns a fixed mapping for the scenarios in its training data. If you want the robot to take a slightly different path — for example, to pick up the *clean* bowl instead of the *dirty* one in an ambiguous scene — you typically need new demonstrations or awkward language prompting that may not work.

3. **Poor spatial generalization**: VLAs trained on RGB images lack explicit depth understanding, meaning they struggle with precise 3D spatial relationships. They may reach for the right object but at the wrong depth, or fail to navigate around obstacles because they never learned what a depth map encodes.

The paper's central thesis is that these three weaknesses share a common root cause: **the absence of structured spatial reasoning in current VLA architectures**. Reasoning — the process of decomposing a task into intermediate steps, understanding the 3D environment, and planning a trajectory before executing it — is what allows humans to act with intention, adapt to novel situations, and explain their actions. Robots need the same capability.

### Why This Matters: From Lab Demonstrations to Deployable Robots

The brittleness of current VLAs is not a minor inconvenience — it is arguably the primary bottleneck preventing robotic manipulation from graduating from curated lab demonstrations to reliable real-world deployment. The paper articulates this gap concretely in the introduction:

> "In contrast to the rapid generalization gains seen in large language and vision models, progress in robotics has lagged behind... Despite massive efforts in dataset collection and model scaling, today's VLAs remain brittle and opaque—struggling to transfer across tasks, scenes, or embodiments, and offering little insight into why a robot chose one action over another."

This matters for several practical reasons:

- **Safety and trust**: In shared human-robot environments (homes, hospitals, factories), operators need to understand what the robot is about to do and why. A robot that reaches for a knife without visible planning is hard to trust; a robot that first localizes the knife in 3D space, then draws a planned trajectory on an interface, and then executes the action is far more transparent.

- **Interactive correction**: Language is inherently ambiguous for spatial commands ("pick up the bowl on the left" — left from whose perspective?). When a VLA misinterprets an instruction, the user needs a way to correct it beyond rephrasing the sentence. The paper's visual reasoning traces can be *edited*: a user can draw a different trajectory on the camera image, and the model will follow that sketch. This is a qualitatively different and more precise interaction modality than language.

- **Data efficiency**: The paper's hypothesis is that reasoning in space acts as a strong inductive bias — it constrains the learning problem so that the model doesn't need to memorize every possible perception-to-action mapping. The authors demonstrate this concretely: MolmoAct is pretrained on approximately 26.3M samples, whereas π0 uses at least 903M. That's a ~34× difference in pretraining data, yet MolmoAct achieves competitive or superior results. This suggests that structured reasoning can substitute for raw data scale, which has major economic and practical implications for organizations without access to internet-scale robot datasets.

- **Out-of-distribution robustness**: A model that truly understands the spatial structure of a scene (where objects are in 3D, what a trajectory through that space looks like) should generalize better when objects move, lighting changes, or new distractors appear — because it reasons about relationships rather than memorizing pixel patterns. The paper's generalization results (Section 5.3) provide evidence for this claim.

### Prior Approaches and Where They Fall Short

The paper positions itself relative to four lines of prior work, each of which captures part of the solution but leaves critical gaps:

#### 1. Direct VLA Models (RT-2, OpenVLA, Octo, π0, GR00T N1)

These models extend vision-language pretraining to action prediction by adding an action head or action tokenization scheme on top of a VLM backbone (Brohan et al., 2022; Kim et al., 2024; Team et al., 2024b; Black et al.; NVIDIA et al., 2025). They follow the standard recipe: pretrain on web-scale image-text data to acquire visual and linguistic world knowledge, then fine-tune on robot demonstration data for action prediction.

**Where they fall short**: These models treat action prediction as a direct mapping — image + text → action — with no intermediate reasoning. They lack explicit depth awareness (a significant limitation for 3D manipulation), produce no interpretable intermediate representations, and offer limited steerability. The authors argue this is not merely a feature gap but a structural one: the direct-mapping architecture makes it harder for these models to learn the spatial relationships needed for robust control, which manifests as brittleness under distribution shift and poor generalization to long-horizon tasks.

The paper's evidence for this is the relative performance on LIBERO-Long (Table 2), where direct-mapping models like OpenVLA and Octo-Base achieve only 53.7% and 51.1% respectively on long-horizon tasks, while MolmoAct reaches 77.2% — a gap of over 23 percentage points. Long-horizon tasks require chaining multiple spatial subgoals, exactly the kind of reasoning the direct-mapping models lack.

#### 2. Language-Based Reasoning for Robotics

Another line of work integrates LLMs or VLMs as high-level reasoners that decompose tasks into subtasks expressed in natural language (Ahn et al., 2022; Huang et al., 2023; Bharadhwaj et al., 2024). For example, the instruction "clean up the dishes" might be decomposed into: `grasp(bowl) → move_to(dishwasher) → release(bowl)`.

**Where they fall short**: Language-based decomposition, while useful for high-level planning, **cannot specify precise spatial trajectories**. The instruction "move to the dishwasher" provides no information about *which path* to take, *how fast* to move, or *where exactly* the dishwasher opening is in 3D space. The authors explicitly call this out:

> "attempting to distill complex 3D trajectories into linguistic descriptions often results in significant loss of spatial and temporal information."

This is a fundamental limitation: language tokens have discrete, symbolic semantics, while robot trajectories are continuous, geometric paths in 3D space. Forcing the model to compress geometric information into language is lossy by design. Related approaches like ECoT (Zawalski et al., 2024) and ThinkAct (Huang et al., 2025) attempt to incorporate intermediate reasoning — through subgoal generation or latent visual planning — but their reasoning remains either textual or latent, making it difficult to ground, verify, or edit.

#### 3. Intermediate Representation Approaches

Some prior work has explored intermediate representations that bridge perception and action: RT-Trajectory (Gu et al., 2023) overlays sketch-like trajectory cues on images, TraceVLA (Zheng et al., 2024) predicts 2D waypoints before actions, Emma-X (Sun et al., 2024) autoregressively predicts future gripper positions in 2D along with 3D coordinates, and HAMSTER (Li et al., 2025) enables language-conditioned trajectory generation.

**Where they fall short**: The paper identifies three key limitations in these approaches:

- **No depth integration**: Most trajectory-based approaches operate purely in 2D image space. Emma-X adds 3D coordinates as a separate prediction stream, but this treats depth as just another number to predict rather than a first-class perceptual representation. The model never explicitly sees a depth map or reasons about the 3D structure of the scene.

- **Decoupled from the policy**: HAMSTER and RT-Trajectory use a separate VLM to generate trajectories, then feed those trajectories to a different low-level policy for execution. This decoupling means the trajectory generator doesn't have feedback about what is physically executable, and the low-level policy is limited to following trajectories for the specific tasks it was trained on. The paper notes:

> "HAMSTER enables language-conditioned trajectory steering but outputs only 2D trajectories by the high-level VLM, with execution handled by a low-level policy trained on a fixed set of tasks."

- **Limited steerability**: While RT-Trajectory and TraceVLA produce visual traces that could theoretically be edited, this isn't a core focus, and the models aren't designed or trained to make steering reliable across diverse tasks and embodiments.

#### 4. Large-Scale Proprietary Systems (π0, GR00T N1)

The most prominent recent VLAs — π0 and GR00T N1 — combine massive pretraining data (the full OXE dataset plus private robot data), advanced action tokenization (flow matching, discretization), and proprietary model architectures. π0, for example, trains on at least 903M samples. These models set strong performance baselines but are **closed-source**: their training data, model weights, and full training procedures are not publicly available.

**Where they fall short**: Beyond the closed-source limitation (which impedes reproducibility and community-driven research), these models still follow the direct perception-to-action paradigm with limited intermediate reasoning. GR00T N1 introduces some hierarchical structure but focuses it on the VLM backbone rather than on spatially grounded reasoning during action prediction. The paper also highlights the extraordinary computational cost: GR00T N1.5 required 50,000 GPU hours for pretraining, compared to MolmoAct's 9,216 GPU hours (a 5.4× reduction), suggesting that the direct-mapping approach may also be computationally inefficient.

### How MolmoAct Positions Itself

MolmoAct does not propose a single new technique but rather a **different architecture philosophy**: instead of scaling data and parameters to brute-force the perception-to-action mapping, build explicit spatial reasoning into the model's autoregressive generation pipeline. The key positioning points are:

**"Reasoning in space" rather than language**: The paper's core conceptual contribution is that robotic reasoning should happen in the spatial modality — through depth maps and 2D trajectory sketches — not in language. This is more than a representational choice; it's a claim about what information is *salient* for manipulation. Language is good for specifying goals ("pick up the bowl") but poor for specifying motions ("move 2.3 cm to the left at a 15-degree angle while maintaining depth of 45 cm"). Predictions like depth perception tokens and visual traces are naturally suited for the latter.

**Unified architecture, not a pipeline**: Unlike HAMSTER or RT-Trajectory which use separate models for high-level planning and low-level control, MolmoAct generates all three representations — depth, trajectory, and action — within a single autoregressive model. This means the depth prediction influences the trajectory prediction, which in turn influences the action prediction, all through the shared VLM backbone. The model learns to make these predictions mutually consistent, avoiding the communication gap that arises in decoupled systems.

**Fully open by design**: The paper explicitly embraces openness as a core value, not an afterthought. This includes releasing model weights, training code, the MolmoAct Dataset (10,689 high-quality robot trajectories across 93 tasks), and all data preprocessing scripts. The authors frame this as enabling "an open blueprint for building ARMs that transform perception into purposeful action through grounded reasoning" — positioning MolmoAct not just as a model but as a reproducible research artifact.

**Data efficiency through structure**: The paper's architecture embodies a bet that structured reasoning can substitute for data scale. By decomposing the problem into depth estimation → trajectory planning → action execution, the model can leverage pre-existing capabilities (depth estimation from Depth-Anything-v2, pointing from Molmo's 2D pointing training) and learn the action mapping with far fewer robot-specific samples. The 34× reduction in pretraining data compared to π0, combined with competitive or superior performance, is the paper's evidence that this bet pays off.

**Steerability as a first-class capability**: Rather than treating user interaction as an afterthought, MolmoAct bakes steering into its training procedure through the trajectory-conditioned action data stream (Section 3.1). The model learns that visual traces are not just predictions to be generated but also inputs to condition upon, creating a natural interface for interactive control that generalizes to novel configurations without additional training.

## 3. Technical Approach

### 3.1 Reader Orientation

MolmoAct is a system that transforms a camera image and a natural language instruction into actionable robot commands through a structured, three-stage reasoning process, where the model first estimates the 3D depth of the scene, then sketches a 2D path for the robot's gripper on the image, and finally predicts the precise motor commands that execute that planned trajectory. The system solves the problem of brittle, opaque robot control by decomposing the monolithic perception-to-action mapping into interpretable intermediate representations—depth maps and spatial trajectories—that each constrain the downstream prediction, making the overall system more robust to novel scenes and more amenable to human inspection and interactive correction than direct end-to-end models.

### 3.2 Big-Picture Architecture (Diagram in Words)

The MolmoAct system has five major components, with information flowing sequentially through three reasoning stages before producing an executable action:

1. **Vision-Language Model (VLM) Backbone** (inherited from Molmo): a standard three-part pipeline—a Vision Transformer (ViT) encodes the input RGB camera image into patch-level features, a vision–language connector pools and projects those features into the language model's embedding space, and a decoder-only Large Language Model (LLM) processes those vision tokens alongside text tokens from the language instruction. This component provides the semantic and visual world knowledge acquired from web-scale pretraining.

2. **Depth Perception Tokens** (first reasoning stage): the LLM autoregressively generates a sequence of 100 discrete tokens that represent a quantized depth map of the scene. These tokens are decoded through a pre-trained VQVAE codebook to reconstruct the dense depth map, giving the model explicit 3D spatial awareness that conditions all subsequent predictions.

3. **Visual Reasoning Trace** (second reasoning stage): conditioned on the depth perception tokens, the model generates a polyline of 1–5 points on the image plane representing the future trajectory of the robot's end-effector—where the gripper is now and where it will be at future timesteps through to the episode end. This trace is both an internal plan and a visualizable, editable representation.

4. **Action Tokens** (final prediction): conditioned on the depth tokens and the visual reasoning trace, the model predicts the final motor command—discretized into 256 bins per action dimension—as a sequence of tokens from a specially constructed action vocabulary that preserves ordinal similarity between adjacent bins.

5. **Trajectory-Conditioned Action Interface** (steerability mechanism): the model is separately trained to accept a user-drawn visual reasoning trace overlaid on the camera image as input, bypassing the depth and trace generation stages, and directly predict the action that follows that sketched path. This enables interactive steering where a human operator can draw corrections on the image plane.

The full inference pipeline (autonomous mode) proceeds as: RGB image + language instruction → VLM backbone encoding → autoregressive generation of depth perception tokens → autoregressive generation of visual reasoning trace tokens → autoregressive generation of action tokens → execution on the robot. The steerability mode bypasses the first two generation stages: RGB image + language instruction + user-drawn trace overlaid on image → action tokens.

### 3.3 Roadmap for the Deep Dive

- **First, the VLM backbone and its adaptation from Molmo**, since all downstream reasoning builds on this visual–linguistic representation. This covers the vision encoder, the connector, the LLM, and how multi-image inputs are handled.
- **Second, action tokenization**, because it's the output format that the entire pipeline must eventually produce, and the paper's novel bin-to-token mapping is a key efficiency enabler.
- **Third, the three-stage action reasoning procedure** (depth perception tokens, visual reasoning traces, and action reasoning), since this is the core architectural innovation. I'll walk through each stage's representation, how ground-truth labels are generated, and how the autoregressive factorization works.
- **Fourth, the steerability mechanism**, because it builds naturally on the visual reasoning trace concept and represents a different inference mode that reuses the same model weights.
- **Fifth, the data curation pipeline for action reasoning data**, since understanding how the training targets are constructed—particularly the VQVAE depth codebook and the VLM-based trajectory point extraction—is essential to understanding what the model learns.

### 3.4 Detailed, Sentence-Based Technical Breakdown

This is primarily a **systems and architecture paper** whose core idea is that robotic manipulation policies should produce spatially grounded intermediate reasoning representations—depth maps and 2D trajectory sketches—before predicting motor commands, and that generating these representations autoregressively within a single VLM backbone yields better generalization, data efficiency, and steerability than direct perception-to-action mappings.

---

#### VLM Backbone: Inheriting and Adapting Molmo

MolmoAct builds its visual and linguistic capabilities on top of **Molmo** (Deitke et al., 2024), an open multimodal language model that the authors extend for action prediction. Understanding this backbone is essential because all spatial reasoning happens in the LLM's autoregressive generation loop, which treats depth tokens, trace coordinates, and action bins as just another vocabulary to predict.

**The standard VLM architecture.** Modern VLMs share a three-component structure: a visual encoder transforms images into patch embeddings, a projection module maps those visual features into the language model's token embedding space, and a decoder-only LLM processes both visual and text tokens under a next-token prediction objective. Molmo follows this pattern exactly.

**Vision encoder choices.** MolmoAct instantiates two model variants with different vision encoders, reflecting a tradeoff between openness and performance:

- **MolmoAct-7B-O**: Uses OpenAI ViT-L/14 at 336×336 pixel resolution with CLIP pretraining. The authors note this encoder uses closed training data but can be reproduced from scratch (as demonstrated by MetaCLIP; Xu et al., 2024a). This is labelled the "most open" variant.
- **MolmoAct-7B-D**: Uses ViT-SO400M/14 at 384×384 pixel resolution with SigLIP2 pretraining (Tschannen et al., 2025). This is the higher-performing variant used for all main experiments. The authors acknowledge SigLIP2's pretraining data is not fully disclosed.

**LLM backbones.** The two variants pair with different open LLMs: OLMo2-7B for the 7B-O variant and Qwen2.5-7B for the 7B-D variant. These differ in their vocabulary structure, which becomes important for action tokenization (Section 3.4, Action Tokenization).

**Image encoding with multi-scale cropping.** Standard ViTs process fixed-size square images, which loses fine-grained detail needed for manipulation. MolmoAct inherits Molmo's solution: **multi-scale cropping with overlap**.

For each input image, the pre-processor produces:
1. One **low-resolution crop**: the full image resized and padded to the ViT's native resolution (384×384 for 7B-D, 336×336 for 7B-O).
2. Multiple **high-resolution crops**: the image is tiled into a grid (e.g., 2×2, 3×1) where each cell matches the ViT input size. The grid squares are moved slightly closer together to introduce a 4-patch overlap margin (~56 pixels), providing neighbor context at borders. Only non-overlapping regions of each tile are forwarded to the LLM, effectively tiling the high-resolution image. The grid is chosen to preserve aspect ratio while minimizing upscaling.

A maximum number of high-resolution crops is enforced; if covering the image would exceed this limit, the image is downscaled. Black borders pad the image so each crop is square and grid-aligned.

Each crop is encoded independently by the ViT. A learned embedding indicating padding status (no padding, some padding, all padding) is added so the model distinguishes natural black regions from artificial padding.

For the SigLIP2 encoder (7B-D), images are resized to a square without padding, following SigLIP2's native training transform. For the OpenAI CLIP encoder (7B-O), the aspect-ratio-preserving resize-and-pad scheme is used.

**The vision–language connector.** After ViT encoding, patch features undergo two processing steps before entering the LLM:

1. **Layer concatenation**: Features from two ViT layers are concatenated per patch—the third-to-last and tenth-from-last for CLIP, or fourth-to-last and tenth-from-last for SigLIP2. Using two layers slightly outperforms a single layer (confirmed in the original Molmo work).

2. **Attention pooling in 2×2 windows**: Within each 2×2 window of patch features, a multi-headed attention layer pools the four patches into a single vector, using the mean of the four patches as the query. This halves each spatial dimension (reducing sequence length by 4×) while preserving local structure. Pooled features are then mapped to the LLM embedding dimension via a small MLP.

**Arranging vision tokens.** Pooled features are serialized left-to-right, top-to-bottom. Low-resolution image tokens appear first (with special start/end markers), followed by high-resolution crop tokens arranged in row-major order. Row-end tokens are inserted between rows to indicate transitions. For the full hyperparameters, Table 3 specifies dimensions: the 7B-D connector uses 1152-dim input, 2×2 pooling, 3584-dim LLM input; the 7B-O uses 1024-dim input, 4096-dim LLM input.

**Multi-image input support.** Molmo does not natively support multiple images, but MolmoAct needs this for multi-camera setups (side view + wrist view). The implementation is straightforward: each image is processed into vision tokens independently using the same procedure, then an index token (e.g., "Prefix 1", "Prefix 2") is prepended to each image's vision token sequence. All sequences are concatenated into a single input to the LLM. This is used during mid-training and post-training where dual-camera inputs are standard.

---

#### Action Tokenization: Mapping Continuous Control to Discrete Tokens

The robot's action space is continuous (e.g., delta end-effector pose: 3 translations + 3 rotations + 1 gripper state = 7 degrees of freedom), but the LLM operates over a discrete vocabulary. The problem is how to represent continuous actions as discrete token sequences without destroying the geometric structure of the action space. The paper's solution has two parts: **discretization into uniform bins** and **ordinal token assignment**.

**Per-dimension discretization.** For each action dimension independently, the robot demonstration data is normalized using dataset quantiles, then discretized into 256 uniform-width bins spanning the range between the 1st and 99th percentiles. This clips extreme outliers while preserving the effective dynamic range. The result: an N-dimensional action becomes N integers, each in the range `[0, 255]`.

The 256-bin granularity is chosen to balance representational precision (more bins = finer control) against vocabulary size and learning difficulty (fewer bins = easier to learn, but coarser control). Prior work has typically used 256 bins as well (Brohan et al., 2022; Kim et al., 2024).

**The ordinal token assignment problem.** Standard practice in prior VLA work is to assign each of the 256 discrete bins to a distinct token from the tail of the LM's vocabulary—arbitrary tokens that happen to be unused. However, continuous action bins have **ordinal structure**: bin 127 and bin 128 represent very similar actions (adjacent in the continuous space), while bin 0 and bin 255 represent maximally different actions. Arbitrary vocabulary tokens have no such relationship—they are effectively unrelated in the embedding space.

This mismatch creates a poor initialization: the model's token embeddings for adjacent action bins are randomly initialized and unrelated, so the model must learn from scratch that "bin-127-token" and "bin-128-token" should map to similar action magnitudes. This is an unnecessary learning burden.

**MolmoAct's solution: byte-level BPE similarity preservation.** The key insight is to assign bins to tokens such that **adjacent bins map to tokens with similar subword representations**. The procedure:

1. Identify the final 256 tokens in the tokenizer's vocabulary. (These are typically rarely-used tokens that can be repurposed.)
2. For each token, extract its underlying **byte-level BPE symbol**—the raw byte sequence that the token represents in the tokenizer's encoding scheme.
3. Sort tokens so that tokens with similar byte sequences are adjacent, then assign them monotonically to the 256 bins.

The effect: since adjacent bins now point to tokens that share subword-level character patterns (because byte-level BPE creates similar byte sequences for tokens that are near each other in the vocabulary), their learned embeddings will start closer together in embedding space. This provides a **smoothness prior** that mirrors the true geometry of the action space.

The paper reports this "substantially reduces training time." Concretely: GR00T N1.5 required 50,000 GPU hours for pretraining; MolmoAct achieves pretraining in 9,216 GPU hours—a 5.4× reduction. The action tokenization strategy contributes to this efficiency.

The complete action vocabulary mapping is provided in Appendix C (Tables 5 and 6), listing each bin index 0–255 with its corresponding Unicode token string.

**Training objective for action tokens.** The model is trained with standard next-token prediction (cross-entropy loss), but the loss is computed **only on the action tokens**—not on the depth or trace tokens (those have separate training schemes described below). This focuses the action-learning signal on the precise control prediction while the depth and trace tokens serve as conditioning context that shapes the LLM's internal representations.

---

#### The Three-Stage Action Reasoning Pipeline

The core architectural innovation of MolmoAct is that action prediction is not a single-step process but a **three-stage autoregressive generation** that produces intermediate spatial representations before the final motor command. Given an RGB image observation `$I$` and a language instruction `$T$`, the model factorizes the prediction as:

$$p(d, \tau, a \mid I, T) = \prod_{i=1}^{M+2} p(d_i \mid I, T, d_{<i}) \times \prod_{j=1}^{L} p(\tau_j \mid I, T, d, \tau_{<j}) \times \prod_{k=1}^{D} p(a_k \mid I, T, d, \tau, a_{<k})$$

where `$d$` is the depth perception token sequence (length `$M+2$` = 102 tokens: a start token, 100 depth code indices, and an end token), `$\tau$` is the visual reasoning trace (length `$L$` points, 1 ≤ `$L$` ≤ 5), and `$a$` is the action token sequence (length `$D$` = the number of action degrees of freedom, e.g., 7 for a single-arm Franka). The notation `$d_{<i}$` means all depth tokens generated before position `$i$`, and similarly for `$\tau_{<j}$` and `$a_{<k}$`.

**What this factorization means operationally.** The model generates all 102 depth tokens first, reading the image and instruction but not yet knowing the trajectory or action. Then, conditioned on those depth tokens _and_ the image and instruction, it generates the trace points one at a time. Finally, conditioned on the depth tokens, the trace points, the image, and the instruction, it generates the action tokens. Each stage constrains the next: the depth map tells the model about 3D obstacles and affordances; the trace plan tells the model which path through that 3D space to follow; the action generation simply executes the plan.

**Why this order?** Depth estimation is a purely perceptual task—it depends on the image and requires no planning. Trajectory planning depends on understanding the 3D layout (hence needs depth first) and the goal (from the language instruction). Action prediction depends on knowing both what the scene looks like in 3D and what path to follow—making it the natural final stage. The authors do not ablate alternative orderings, but the cascade from perception → planning → control mirrors classical robotics pipelines and provides a strong inductive bias.

---

##### Depth Perception Tokens

Depth estimation is the first reasoning stage because 3D spatial understanding is foundational for manipulation. The authors observe that conventional VLMs and VLAs are trained solely on RGB images and therefore "lack the ability of depth estimation and 3D understanding, which is critical for robotic manipulation."

**The depth vocabulary.** The paper defines a specialized vocabulary for depth:

$$V_{\text{depth}} = \{\langle\text{DEPTH\_START}\rangle, \langle\text{DEPTH\_END}\rangle\} \cup \{\langle\text{DEPTH\_k}\rangle\}_{k=1}^{N}$$

where `$N = 128$` is the codebook size. The actual depth representation for a single image is a fixed-length string:

$$d = (\langle\text{DEPTH\_START}\rangle, \langle\text{DEPTH\_z}^{\text{depth}}_1\rangle, \ldots, \langle\text{DEPTH\_z}^{\text{depth}}_M\rangle, \langle\text{DEPTH\_END}\rangle)$$

with `$M = 100$` tokens, and each `$z^{\text{depth}}_i \in \{1, \ldots, 128\}$` indexes a code in a VQVAE codebook.

**What `$d$` physically represents.** The sequence `$d$` is a discrete, compressed encoding of the dense depth map for the input RGB image. The 128-dimensional VQVAE codebook and 100-token sequence length together define the granularity: each image's depth map is compressed into 100 × log₂(128) = 700 bits of information, a drastic compression of the original dense depth map that forces the codebook to capture the most salient depth structure.

**The specialist-to-generalist distillation strategy.** The depth perception tokens are not learned from scratch by MolmoAct. Instead, a **specialist depth estimator** is trained first, and MolmoAct learns to imitate its output. The procedure:

1. **Train a VQVAE on depth maps.** The authors collect 10 million depth maps from tabletop manipulation images in the RT-1, BridgeData V2, and BC-Z datasets. Depth maps are obtained by running Depth-Anything-v2 (a pre-trained monocular depth estimator) on each RGB observation. The VQVAE is trained with a standard reconstruction objective for 20 epochs to compress depth maps into the 128-codebook, 100-token representation.

> The VQVAE architecture is standard (Van Den Oord et al., 2017): an encoder compresses the depth map to a latent grid, each latent vector is quantized to the nearest of 128 learned codebook vectors, and a decoder reconstructs the depth map from the quantized latents. The training loss combines reconstruction error with codebook commitment losses.

2. **Encode all training images with the frozen VQVAE.** For every image in the action reasoning dataset (Section 3.4, Data Curation), the trained VQVAE encoder produces the 100-token depth string `$d$`. This becomes the ground-truth target for MolmoAct's depth prediction.

3. **Train MolmoAct to predict `$d$` autoregressively.** Given only the RGB image and language instruction, the model learns to generate the depth token sequence `$d$` one token at a time. Since the VQVAE encoder used the RGB image (after running Depth-Anything-v2 to get the depth map), MolmoAct is essentially learning to **internalize the specialist's depth estimation capability** from RGB alone, without access to the depth map at inference time.

**Why distillation rather than end-to-end training?** Training the VQVAE separately on depth maps provides a clean, interpretable intermediate representation. The depth token vocabulary `$V_{\text{depth}}$` has a deterministic one-to-one mapping from codebook indices to tokens (index `$k$` maps to `<DEPTH_k>`), so the token string is precisely decodable—the VQVAE decoder can reconstruct the depth map from the tokens MolmoAct predicts. This means MolmoAct's depth reasoning is directly visualizable: at any point, a user can decode the depth tokens to see what 3D understanding the model has formed. This transparency is a core design goal.

**Tokenizer adaptation.** To enable the LLM to output these new depth tokens, the tokenizer embedding matrix and language model head must be extended. For MolmoAct-7B-D (which uses Qwen2.5-7B), the first 130 padding tokens in the vocabulary are replaced with the 130 depth tokens (`<DEPTH_START>`, `<DEPTH_END>`, and 128 `<DEPTH_k>` tokens). For MolmoAct-7B-O (OLMo2-7B), which has fewer than 130 padding tokens, the embedding matrix is first padded to the next multiple of 512, then the first 130 tokens are replaced similarly. The extended embeddings are trained during pretraining (not frozen).

**Image resolution for depth token generation.** All images are resized to 320×320 pixels before depth tokenization to enforce the consistent 100-token representation. This fixed resolution is a constraint of the VQVAE architecture—the latent grid size determines the number of tokens, and changing image resolution would change the token count.

---

##### Visual Reasoning Trace

The second reasoning stage generates a **2D polyline on the image plane** representing the future path of the robot's end-effector. This is the planning component: given the 3D understanding from depth perception tokens and the task goal from the language instruction, the model sketches where the gripper should move.

**Trace representation.** For a given image observation, a visual reasoning trace is defined as:

$$\tau = (p_1, p_2, \ldots, p_L), \quad p_i = (u_i, v_i)$$

where `$L$` is the number of points (1 ≤ `$L$` ≤ 5), and each `$p_i$` is a 2D coordinate in image space. Coordinates are normalized to integers in `$[0, 255]$` relative to the image dimensions.

**What the points represent.** The first point `$p_1$` marks the robot end-effector's **current location** in the image. The remaining points `$p_2, \ldots, p_L$` mark the end-effector's **future locations** at evenly spaced intervals from the current timestep to the episode's final timestep. If the episode ends at timestep `$e$` and the current timestep is `$t$`, the trace includes:

- `$p_1 = (u_t, v_t)$`: current gripper position.
- `$p_L = (u_e, v_e)$`: final gripper position at episode end.
- Up to 3 intermediate points subsampled evenly between `$t$` and `$e$`.

If `$e - t < 4$` (fewer than 4 remaining timesteps), all available intermediate points are included. If `$t = e$` (episode end), the trace contains only the single current point `$p_1$`.

The trace is thus a **subsampled future trajectory**—not every future step, but enough keyframes to sketch the intended motion. The 5-point maximum is a design choice balancing informativeness (more points = more precise guidance) against sequence length (shorter sequences = faster generation and fewer opportunities for error accumulation).

**Generating ground-truth traces for training.** Unlike depth tokens, which come from an external specialist model, visual reasoning traces are extracted directly from the robot demonstration data using a VLM's pointing capability:

1. For each timestep in an episode, the true gripper pixel coordinate `$(u_t, v_t)$` is known from the robot's kinematics and camera calibration.

2. To generate a trace label that MolmoAct can learn from, the authors prompt **Molmo** (the pre-trained VLM before action fine-tuning) with: "point to the robot gripper" (for single-arm) or "point to the robot gripper on the left/right" (for bimanual). Molmo returns a predicted 2D coordinate in `$[0, 100]$` normalized space.

3. The predicted coordinates are rescaled to integers in `$[0, 255]$` to form the trace points.

4. This query is applied at every timestep, producing one predicted gripper location per frame. Linking these predictions sequentially yields the full episode trajectory. For training, at each timestep `$t$`, a subsequence is selected as described above (current point, final point, up to 3 intermediates).

**Why use Molmo to generate the trace labels rather than using ground-truth pixel coordinates directly?** This is a subtle but important design choice. The authors are not extracting ground-truth traces from kinematics—they are generating pseudo-labels using a VLM. This means the trace labels reflect what the VLM _perceives_ as the gripper location, including any systematic biases or errors in the VLM's pointing capability. When MolmoAct is later trained to predict traces from RGB images, it is learning the same mapping that Molmo learned—a form of **self-consistency training** where the model's own pointing capabilities are bootstrapped into trajectory planning.

For bimanual robots, two separate prompts are issued per frame to obtain `$\tau_L$` and `$\tau_R$` for the left and right grippers.

**Why visual traces rather than language-based plans?** The paper explicitly contrasts this with language-based planning approaches:

> "attempting to distill complex 3D trajectories into linguistic descriptions often results in significant loss of spatial and temporal information."

A 2D polyline on the image plane is a **spatially precise, continuous representation** that directly specifies _where_ in the image the gripper should move. It captures fine-grained spatial information—the exact curve of an approach path, the location of a grasp point relative to object edges—that would require verbose and imprecise language to describe (e.g., "move 2.3 cm to the left, then approach from above at a 15-degree angle"). The trace is also inherently visual, making it natural for human inspection and editing.

---

##### Action Reasoning Procedure

With depth tokens `$d$` and visual reasoning trace `$\tau$` generated, the model produces the final action tokens `$a$`. The full autoregressive factorization (Equation 4) is:

$$p(d, \tau, a \mid I, T) = \prod_{i=1}^{M+2} p(d_i \mid I, T, d_{<i}) \times \prod_{j=1}^{L} p(\tau_j \mid I, T, d, \tau_{<j}) \times \prod_{k=1}^{D} p(a_k \mid I, T, d, \tau, a_{<k})$$

**What this computes, stage by stage:**

- **Stage 1 (perception):** The probability of the depth token sequence `$d$` given only the image `$I$` and instruction `$T$`. Each depth token `$d_i$` is conditioned on all previously generated depth tokens `$d_{<i}$`. This is a pure depth-from-RGB estimation task, with the language instruction providing context about which regions of the scene are task-relevant.

- **Stage 2 (planning):** The probability of the trace `$\tau$` given the image, instruction, and the full depth sequence `$d$`. Each trace point `$\tau_j$` (a coordinate pair) is conditioned on all previously generated trace points `$\tau_{<j}$` and the depth tokens. The depth information constrains where physically plausible trajectories can go (e.g., the trace should not pass through solid objects that the depth map reveals).

- **Stage 3 (control):** The probability of the action tokens `$a$` given the image, instruction, depth tokens, and full trace. Each action dimension `$a_k$` is conditioned on all previously generated action tokens `$a_{<k}$` plus all preceding context. The trace provides explicit spatial guidance: the model knows where the gripper should be at future timesteps, so the immediate action should move the gripper toward the first future point on the trace.

**Why this factorization helps.** By conditioning each stage on the outputs of previous stages, the model's predictions become progressively more constrained and therefore easier. The action prediction `$a$` does not need to implicitly encode depth understanding—that's handled by the depth tokens already in context. The trace prediction `$\tau$` does not need to operate in a depth-blind manner—it sees the depth tokens. This factorization decomposes the monolithic perception→action mapping into subproblems that align with natural task structure.

**Action chunking for post-training.** During post-training (fine-tuning on target tasks), the model predicts **action chunks** rather than single actions. An action chunk is a sequence of `$N = 8$` consecutive actions, formatted as a list of tokenized actions. The model generates all 8 actions autoregressively, then the robot executes them open-loop before the next inference call. This is standard practice in imitation learning (Zhao et al., 2023) and reduces effective inference latency by amortizing model calls over 8 control steps. For evaluation on LIBERO, the chunk size is fixed at K = 8 and full chunks are executed before re-planning.

**Training signal distribution.** During pretraining, the model is trained on all three prediction tasks simultaneously through the next-token prediction objective, but the loss on different token types is handled differently. The action tokens use standard cross-entropy loss. The depth and trace tokens are also trained with next-token prediction, but the auxiliary depth and trace datasets (described next) provide additional training signal focused specifically on those modalities. The multimodal web data (image captioning, VQA, pointing) maintains the model's general visual–linguistic capabilities during robot-specific training.

---

#### Action Steerability via Visual Reasoning Trace

Steerability—the ability for a human operator to guide the robot's behavior at test time—is designed as a first-class capability rather than a post-hoc interface. The key insight is that **the same visual reasoning trace that the model generates as an intermediate planning representation can also be provided as input by a user**.

**The problem with language-only steering.** The paper identifies three failure modes of language-based steering:

1. **Data requirements**: Learning reliable grounding between words and control requires large corpora of diverse language–action pairs, which are expensive to collect.

2. **Spatial ambiguity**: Natural language is inherently imprecise about magnitudes, scales, and endpoints. "Move a little to the left" — how much is "a little"? "Pick up the bowl on the left" — left from whose perspective?

3. **Brittleness to rephrasing**: Post-trained models often overfit to the phrasing patterns in their training data, making them unreliable when users use different words.

For manipulation, these issues translate into imprecise or inconsistent control—the robot picks up the wrong object, moves too far, or ignores the correction entirely.

**Visual trace as steering modality.** The solution: instead of (or in addition to) rephrasing a language command, the user draws a visual reasoning trace `$\tau = (p_1, \ldots, p_L)$` directly on the camera image. This trace is overlaid onto the RGB image `$I$` to form an augmented observation:

$$I^+ = I \oplus \tau$$

The model then generates actions conditioned on this augmented image:

$$p(a \mid I^+, T) = \prod_{k=1}^{D} p(a_k \mid I^+, T, a_{<k})$$

Notice the difference from the full reasoning pipeline: **depth perception tokens are not generated**. The model bypasses the perception and planning stages and goes directly to action prediction, treating the user-provided trace as the plan. The language instruction `$T$` still provides goal context ("pick up the bowl"), but the trace provides the precise spatial path.

**What makes traces better for steering:**

- **Unambiguous precision**: A drawn point at pixel `$(u, v)$` specifies exactly where the gripper should go—no ambiguity about magnitudes or reference frames.
- **No language-action data required**: The model learns trace-following from trajectory-conditioned action data, which is automatically generated from demonstrations (by overlaying the ground-truth future trajectory on the image). This requires no additional human annotation.
- **Generalization**: Visual traces are a geometric modality—a line is a line regardless of the objects in the scene. The model can follow traces through novel object arrangements because it learns the mapping from image-space paths to motor commands, which generalizes across visual appearances.

**Training for steerability: trajectory-conditioned action data.** The model is trained to follow traces through a separate data stream in pretraining and mid-training. For each timestep in a demonstration, an augmented example `$(I^+, T, a)$` is created by overlaying the ground-truth future trajectory `$\tau_{\text{gt}}$` on the current image `$I$`. The model is trained to predict action `$a$` from `$I^+$` and `$T$`, using exactly the same next-token prediction objective. This data is a substantial fraction of the pretraining mixture (38.7% of the total data, as shown in Figure 3).

**At inference time**, steering works as follows: the user sees the initial camera image and the model's predicted trajectory (if in autonomous mode). If the predicted trajectory is incorrect (e.g., heading toward the wrong bowl), the user draws a corrected trace on the image. The augmented image is fed to the model, which generates the corrected action. This is repeated at each timestep for closed-loop control.

**Limitations acknowledged.** The paper notes that this 2D trace-based steering does not incorporate depth information—the trace is purely on the image plane. As a result, the model "often follows the intended path within the image plane (in-plane motion) but exhibits unintended or imprecise translation along the camera's depth axis (out-of-plane)." The authors hypothesize that conditioning on the model's predicted depth-perception tokens could lift the trace into 3D and mitigate this, but leave that for future work.

---

#### Data Curation for Action Reasoning

The three-stage reasoning pipeline requires ground-truth labels for depth perception tokens and visual reasoning traces at every timestep of every demonstration. This section details how those labels are generated from raw robot data.

**Input format for robot episodes.** A robot episode is a sequence of timesteps, where each timestep `$t$` is a tuple:

$$(I, T, a)_t$$

containing an RGB observation image `$I$`, a language instruction `$T$` (constant across the episode), and a ground-truth action `$a$` (in end-effector space or joint space depending on the dataset). The task is to augment each timestep with depth tokens `$d_t$` and a visual reasoning trace `$\tau_t$`, producing the full action reasoning data format.

**Depth perception token generation.** For each frame `$I_t$`:

1. Run Depth-Anything-v2 on `$I_t$` to obtain a dense depth map.
2. Encode the depth map using the pre-trained VQVAE encoder to obtain the 100 codebook indices.
3. Map each index deterministically to its corresponding depth token: index `$k$` → `<DEPTH_k>`.
4. Prepend `<DEPTH_START>` and append `<DEPTH_END>` to form the 102-token sequence.

The VQVAE is trained once on 10 million depth maps; the encoding step at data preparation time is just a forward pass. All images are resized to 320×320 pixels before encoding to maintain the fixed 100-token representation.

**Visual reasoning trace generation.** For each frame `$I_t$` at timestep `$t$` in an episode ending at timestep `$e$`:

1. Query Molmo (the pre-training VLM) with "point to the robot gripper" at every timestep from `$t$` to `$e$`, obtaining predicted coordinates `$(x_s, y_s)$` in `$[0, 100]$` for each `$s \in [t, e]$`.
2. Rescale all coordinates to `$[0, 255]$`.
3. Select a subsequence: the current point `$(u_t, v_t)$`, the final point `$(u_e, v_e)$`, and up to 3 intermediate points evenly spaced between `$t$` and `$e$`. If `$e - t < 4$`, include all available intermediate points; if `$t = e$`, the trace has only one point.

This produces 1–5 points per timestep. Note that this is a **pseudo-labeling** approach: Molmo's pointing predictions serve as the ground truth for MolmoAct's trace generation training. Any systematic errors in Molmo's pointing will be inherited by MolmoAct—but since the same model family is used, these are self-consistent biases rather than contradictory ones.

**Auxiliary robot data.** In addition to the full action reasoning data (which has all three stages), the authors create three auxiliary datasets that isolate specific skills:

- **Auxiliary Depth Data** (1.5M samples): Given an RGB image and instruction, the model predicts only the depth perception tokens. This provides focused training on the depth estimation component without the complicating factors of trajectory or action prediction.

- **Auxiliary Trace Data** (1.5M samples): Given an RGB image and instruction, the model predicts only the visual reasoning trace. This isolates the spatial planning skill.

- **Trajectory-Conditioned Action Data** (10.5M samples): Given the trace-overlaid image `$I^+ = I \oplus \tau$` and instruction `$T$`, the model predicts the action `$a$`. This is the data stream that enables steerability—it teaches the model that visual traces are not just outputs to generate but also inputs that specify intended motion.

The auxiliary datasets collectively ensure that no single reasoning stage is a bottleneck: the model gets direct supervision on depth estimation, direct supervision on trajectory planning, and direct supervision on trace-conditioned action execution, in addition to the end-to-end supervision from the full action reasoning data.

**Data sources for pretraining.** The pretraining mixture (Figure 3) is drawn from a filtered subset of the Open X-Embodiment (OXE) dataset, specifically the BC-Z, BridgeData V2, and RT-1 subsets, totaling 10.5M raw robot samples. These are converted into 10.5M action reasoning samples (each raw sample becomes one action reasoning example), plus the auxiliary data streams, plus 2M multimodal web data samples. The total pretraining mixture is 26.3M samples.

The sampling rates during pretraining (Figure 3, right panel):

| Data Stream | Sampling Rate |
|---|---|
| Action Reasoning (RT-1) | 20% |
| Action Reasoning (BridgeData V2) | 12.5% |
| Action Reasoning (BC-Z) | 7.5% |
| Trajectory-Conditioned Action | 38.7% |
| Auxiliary Depth | 7.5% |
| Auxiliary Trace | 7.5% |
| Multimodal Web Data | 5.0% |

Note that trajectory-conditioned action data has the highest sampling rate (38.7%), reflecting the importance of steerability as a first-class capability.

**The MolmoAct Dataset for mid-training.** After pretraining, the model undergoes a second training stage on the authors' in-house collected MolmoAct Dataset (Section 3.2). This dataset contains 10,689 human-teleoperated trajectories across 93 manipulation tasks in home and tabletop environments, collected over two months by five operators. Each sample includes two side-mounted camera views and one wrist camera view. The data is converted into 1M action reasoning samples and 1M trajectory-conditioned action samples. For training, each three-view sample creates two paired-view examples by pairing each side view with the wrist view. Depth tokens and visual traces are generated only from the side views; the wrist view provides additional visual context without requiring separate trace labels. The model is trained on this for 50K gradient steps at batch size 128 on 128 H100 GPUs (~2,304 GPU hours).

---

#### Multimodal Web Data Co-Training

To prevent catastrophic forgetting of general visual–linguistic capabilities during robot-specific training, MolmoAct co-trains on 2M samples of multimodal web data from Molmo's supervised fine-tuning stage. This mixture includes:

- **Academic VQA datasets**: VQA v2.0, Text VQA, OK-VQA, ChartQA, DocVQA, Infographic VQA, AI2D, A-OKVQA, ScienceQA, TabMWP, ST-VQA, TallyQA, DVQA, FigureQA, PlotQA. These train the model to answer questions about images, charts, documents, and diagrams.
- **PixMo**: A dataset for fine-grained visual understanding and 2D pointing (from Molmo), which is directly relevant to the trace generation capability.
- **LVIS**: An instance segmentation dataset where the model is trained to predict bounding box centers of objects given category names, grounding language to image regions.
- **AndroidControl**: A dataset for GUI action prediction, providing additional action-like supervision in a non-robotic domain.

The web data is sampled at only 5% during pretraining—a small fraction that is sufficient to maintain general capabilities without dominating the robot-specific learning signal.

## 4. Key Insights and Innovations

### Innovation 1: Spatial Reasoning as a Native Modality, Not a Linguistic Afterthought

The most fundamental conceptual move in this paper is the claim that **robotic reasoning should happen in the spatial modality — through depth maps and 2D trajectories — rather than being translated into language**. This is not merely an architectural choice; it is a diagnosis of why prior approaches that inject language-based "chain of thought" into VLAs have underperformed for fine-grained manipulation.

Prior work on reasoning-augmented robotics has largely followed the language model playbook: decompose a high-level instruction into subgoals expressed as natural language (e.g., "grasp the bowl," "move to the dishwasher"), then have a separate module execute each subgoal (Ahn et al., 2022; Huang et al., 2023). Even recent VLA-specific reasoning work — ECoT (Zawalski et al., 2024) synthesizes linguistic subgoals via prompting, CoT-VLA (Zhao et al., 2025) generates visual subgoal *frames* but reasons through latent embeddings, ThinkAct (Huang et al., 2025) uses visual latent planning, Emma-X (Sun et al., 2024) predicts gripper positions as numeric tokens — treats spatial information as something to be *output* rather than something to be *reasoned with*. The reasoning chain either lives in language space or in a compressed latent space that is not directly inspectable or editable.

The authors identify the core failure mode: **language tokens have discrete, symbolic semantics, while robot trajectories are continuous geometric paths in 3D space**. Forcing the model to compress spatial information into language is lossy by design. The paper states this explicitly: "attempting to distill complex 3D trajectories into linguistic descriptions often results in significant loss of spatial and temporal information." A sentence like "move 2.3 cm to the left at a 15-degree approach angle" is both verbose and imprecise compared to a 2D polyline drawn directly on the image.

What makes this contribution **fundamental rather than incremental** is the architectural commitment to spatial reasoning as a *first-class modality with its own vocabulary*. Depth perception tokens are not just another numeric prediction — they are decoded through a VQVAE codebook into a visualizable depth map. Visual reasoning traces are not just coordinate outputs — they are polyline sketches on the image plane that can be rendered, inspected, and edited by a human operator. This transforms the intermediate representations from internal bookkeeping into **communicable artifacts** that bridge the model's reasoning and the user's understanding.

The significance extends beyond performance gains. By making spatial reasoning explicit and visualizable, MolmoAct creates an **accountability mechanism** that is absent in direct VLA models. When a conventional VLA fails to pick up the correct object, a practitioner can only guess whether the failure was perceptual (didn't recognize the object), spatial (misjudged depth), or motor (planned the wrong trajectory). With MolmoAct, the depth map and trajectory sketch are visible at every timestep — failures can be diagnosed by inspection. This diagnostic capability matters enormously for safety-critical deployment, debugging, and building trust in robotic systems.

The evidence that spatial reasoning matters is distributed throughout the paper rather than concentrated in a single ablation. The strongest signal is the LIBERO-Long result (Table 2): MolmoAct achieves 77.2% on long-horizon tasks vs. 53.7% for OpenVLA and 51.1% for Octo-Base — a gap of over 23 percentage points. Long-horizon tasks require chaining multiple spatial subgoals, exactly the capability that spatial reasoning supports. The out-of-distribution generalization results (Figure 6a, Table 21) further reinforce this: MolmoAct's +23.3% average improvement over π0-FAST on generalization tasks suggests that reasoning about spatial relationships (rather than memorizing pixel patterns) is the mechanism driving robustness.

---

### Innovation 2: Structured Reasoning as a Substitute for Data Scale

The paper makes a strong empirical argument — with concrete numbers — that **architectural structure can substitute for massive pretraining data** in robotic manipulation. This is not a theoretical claim about inductive biases; it is a demonstrated fact about training efficiency.

Prior VLAs have pursued a scaling-first philosophy. π0 (Black et al.) trains on at least 903M samples drawn from the full Open X-Embodiment dataset plus proprietary robot data. GR00T N1.5 (NVIDIA et al., 2025) requires 50,000 GPU hours for pretraining. The implicit assumption in these works is that robotic manipulation, like language modeling, benefits primarily from scale — more data, more parameters, more compute. The paper does not explicitly argue against this view, but its results constitute a strong counterexample.

MolmoAct is pretrained on **26.3M samples** — approximately 34× less data than π0 — yet achieves competitive or superior performance:
- 70.5% zero-shot on SimplerEnv Visual Matching vs. π0's 58.7% (Table 1)
- 86.6% on LIBERO vs. π0-FAST's 85.5% (Table 2)
- +10% single-arm and +22.7% bimanual task progression over π0-FAST in real-world fine-tuning (Figure 5)

The training compute tells a similar story: 9,216 GPU hours for MolmoAct vs. 50,000 for GR00T N1.5 — a 5.4× reduction.

What makes this an innovation rather than just a performance result is the **mechanism** the paper identifies. The data efficiency does not come from a better optimizer or a clever augmentation strategy — it comes from **decomposing the learning problem**. Instead of asking the model to learn the monolithic mapping from pixels to motor commands (which requires seeing enormous variation in the training data to generalize), MolmoAct asks the model to learn three sub-problems:
1. Depth estimation (which builds on the pre-existing capability of Depth-Anything-v2, distilled into the VLM)
2. Trajectory planning (which builds on Molmo's pre-existing 2D pointing capability)
3. Action execution (which is constrained by the depth and trajectory context)

Each sub-problem is easier than the whole, and each leverages capabilities the VLM already possesses from web-scale pretraining. The depth tokens transfer 3D understanding without requiring the robot data to teach 3D geometry from scratch. The trace generation reuses pointing abilities learned from academic VQA datasets. The action prediction, being conditioned on these rich intermediate representations, requires fewer examples to learn the residual mapping from plans to motor commands.

This is a **conceptual reframing** of the data-efficiency problem in robotics: rather than asking "how do we collect more robot data?", MolmoAct asks "how do we structure the problem so that existing VLM capabilities — acquired from abundant web data — can be leveraged for manipulation?" This reframing has practical implications for any group without access to internet-scale robot datasets. It suggests that the path to strong robotic policies may run through better architectural decomposition rather than bigger data collection efforts.

A nuance worth noting: the paper does not prove that structured reasoning *always* substitutes for data. On the hardest problems (difficulty bin 5 in the original MoImo paper's taxonomy, though MolmoAct does not use this difficulty framework), the base model's capabilities still matter. But for the distribution of tasks tested — which spans standard manipulation benchmarks — the substitution is clear and quantified.

---

### Innovation 3: Steerability as a First-Class Capability, Not a Post-Hoc Interface

Most VLA research treats user interaction as an afterthought — you train a policy, deploy it, and if it makes a mistake, you might try rephrasing the instruction or adding a new demonstration. MolmoAct makes a different architectural decision: **steerability is baked into the training procedure as a core capability**, achieved by training the model to accept the same visual reasoning traces it generates as conditional inputs.

Prior work on policy steering has approached the problem from several angles, none of which achieve the generality MolmoAct demonstrates:
- **RT-Trajectory** (Gu et al., 2023) and related inference-time steering methods (Wang et al., 2024b) enable trajectory-conditioned control but are coupled to specific architectures (robotics transformers, diffusion models) and lack the semantic generalization from VLM pretraining.
- **HAMSTER** (Li et al., 2025) generates language-conditioned trajectories through a VLM, but these trajectories are executed by a separate low-level policy trained on a fixed task set — the steering generalizes only as far as the low-level policy.
- **Language-based correction** (Shi et al., 2024) allows users to refine behavior through natural language, but suffers from the spatial ambiguity problem the paper identifies: natural language is inherently imprecise about magnitudes, scales, and reference frames.

MolmoAct's innovation is **unifying trajectory generation and trajectory following in a single model**. The trajectory-conditioned action data stream (38.7% of pretraining data, Figure 3) teaches the model that visual traces are simultaneously something to *output* (during autonomous reasoning) and something to *input* (during user steering). This dual-use design means:
- There is no separate low-level policy to train or constrain generalization.
- The same model weights that plan trajectories can also follow them.
- The steering capability transfers to any task the model can perform autonomously — no additional training needed.

The evidence for this innovation's practical impact comes from the steerability evaluation (Section 5.6, Figure 9, Table 23). In the pick_up_bowl task with ambiguous instructions, visual trace steering achieves 75% success rate vs. 42% for open-ended language steering — a 33 percentage point gap. Moreover, MolmoAct with language steering alone still outperforms π0-FAST with language steering by 29 percentage points, suggesting that even the *training process* for dual-use traces improves language grounding.

This is a **fundamental shift** in how we think about human-robot interaction for manipulation. Rather than treating language as the primary interaction modality with visual traces as a fallback, MolmoAct's results suggest that **visual traces are the more reliable steering mechanism** — they are unambiguous, precise, and generalize across phrasings — while language provides the high-level goal specification. This inverts the dominant paradigm and has direct implications for interface design: future robotic systems should prioritize sketch-based correction over language refinement when spatial precision matters.

---

### Innovation 4: Diagnosing and Quantifying the Generalization Benefits of Spatial Grounding

While the paper's core architectural innovations (spatial reasoning tokens, dual-use traces) are its headline contributions, a subtler but equally important insight emerges from the generalization experiments: **spatially grounded reasoning provides robustness benefits that are measurable, systematic, and substantially larger than prior work has documented for alternative approaches**.

This is not a claim about a specific architectural component working well — it is a **diagnostic finding** about *why* VLA generalization has been poor and *what* structural changes improve it. The paper designs a rigorous multi-axis generalization evaluation (Section 5.3, Table 21) that decomposes robustness into four categories: language variation (rephrased instructions), spatial variation (moved objects), distractors (added irrelevant objects), and novel objects (unseen target objects). MolmoAct outperforms π0-FAST across all four axes, with an average +23.3% task progression improvement.

What makes this a distinct insight rather than just "model X is more robust than model Y" is the **pattern** of where the gains concentrate. MolmoAct shows the largest relative improvements on:
- **Novel objects** (putting a sponge instead of a green can into a plate): MolmoAct achieves 0.875 task progression vs. π0-FAST's 0.0 and OpenVLA's 0.25. This is a case where pixel-level memorization (the object looks completely different) fails, but spatial reasoning (the object occupies the same location and the trajectory is similar) succeeds.
- **Language variation** (put the fruit instead of put the banana): MolmoAct achieves 0.625 vs. π0-FAST's 0.0625. This suggests the model is not relying on exact instruction phrasing but rather grounding in the visual scene.

The insight is that **spatial representations are more invariant to visual appearance and linguistic phrasing than direct perception-to-action mappings**. A depth map of a scene with a sponge looks structurally similar to a depth map of the same scene with a can — the geometry is similar even though the RGB pixels differ dramatically. A visual trace pointing to "the object at position (u,v)" is invariant to whether that object is called a "banana" or "fruit." By conditioning action prediction on these spatially invariant representations, MolmoAct naturally inherits their robustness properties.

This is a **conceptual advance** in understanding generalization for robotic manipulation. Prior work has attributed VLA brittleness to various factors — insufficient data diversity, domain gap between pretraining and deployment, limitations of the action representation — but MolmoAct's results isolate **the lack of explicit spatial structure** as a key bottleneck. The implication for future work is clear: models that build explicit geometric understanding of scenes (through depth, trajectories, or other spatial representations) will systematically outperform models that rely on implicit spatial reasoning learned from RGB pixels alone, particularly under distribution shift.

The evidence is anchored in Table 21 and Figure 6a, which provide the per-task, per-condition breakdown necessary to support this claim. The SimplerEnv variant aggregation results (Table 1) provide corroborating evidence: MolmoAct fine-tuned achieves 72.1% on variant aggregation (which introduces lighting, texture, and viewpoint shifts) vs. 71.6% on visual matching — a difference of less than 1%, suggesting near-perfect robustness to these specific distribution shifts.

## 5. Experimental Analysis

### Evaluation Methodology

- **Dataset.** The primary evaluations use three distinct testbeds: (1) **SimplerEnv** (Li et al., 2024c), specifically the Google Robot tasks comprising three visual-matching tasks (Pick Coke Can, Move Near, Open/Close Drawer) and variant-aggregation tasks that introduce lighting, texture, and viewpoint shifts; (2) **LIBERO** (Liu et al., 2023a), a simulation benchmark with a Franka Emika Panda arm across four task suites — LIBERO-Spatial, LIBERO-Object, LIBERO-Goal, and LIBERO-Long — each containing 10 tasks with 500 expert demonstrations total; and (3) a **custom real-world evaluation** spanning six tasks on single-arm and bimanual Franka setups, with 25 trials per task. For mid-training ablation, three additional real-world tasks (close_lid, rotate_pot, pour_tea) are used with 10 trials per task. The out-of-distribution generalization evaluation uses a multi-task real-world setup with three base tasks tested across four perturbation types and one in-distribution condition, with four trials per variant.

- **Base model(s).** All main experiments use **MolmoAct-7B-D** (the variant with SigLIP2 vision encoder and Qwen2.5-7B LLM backbone). The zero-shot SimplerEnv evaluation additionally uses MolmoAct-7B-D-Pretrain (the model immediately after pre-training, before any task-specific fine-tuning). MolmoAct-7B-O (with OpenAI CLIP and OLMo2-7B) is mentioned as an architectural variant but not used in the main evaluations. The choice of 7B-scale models reflects a deliberate positioning at a parameter count that is "representative of the capabilities of many contemporary LLMs" while being trainable with modest compute (9,216 GPU hours for pretraining) — the authors are explicitly demonstrating that structured reasoning can achieve strong results without massive scale.

- **Metrics.** For SimplerEnv, the metric is **success rate** (%) — binary completion of the specified task within an episode. For LIBERO, the metric is **success rate** (%) averaged over all tasks in a suite, following the standard LIBERO evaluation protocol (each task tested with specific initial conditions and object configurations). For real-world evaluations, the metric is **task progression** (0–1 scale), where each task is decomposed into discrete milestones with pre-specified scores (e.g., for put_bowl_in_sink: grasp bowl = 0.25, move into sink = 0.4, open gripper = 0.7, drop bowl at target = 1.0). Full scoring rubrics are detailed in Appendix D.3. For instruction-following evaluations, the metric is **Elo rating** derived from pairwise human preferences in an arena-style interface (over 1,500 votes for action execution, over 1,000 votes for visual trace generation). For steerability, the metric is **success rate** (%) computed over 15 trials.

- **Baselines.** The paper compares against a broad set of contemporary generalist policies, spanning both open and closed models. For SimplerEnv zero-shot: **TraceVLA** (Zheng et al., 2024), **RT-1-X** (Brohan et al., 2022), **RT-2-X** (Zitkovich et al., 2023), **Octo-Base** (Team et al., 2024b), **OpenVLA** (Kim et al., 2024), **RoboVLM** (Liu et al., 2025), **Emma-X** (Sun et al., 2024), **Magma** (Yang et al., 2025b), **HPT** (Wang et al., 2024a), **SpatialVLA** (Qu et al., 2025), **GR00T N1.5** (NVIDIA et al., 2025), **π0**, and **π0-FAST** (Black et al.). For LIBERO: **TraceVLA**, **Octo-Base**, **OpenVLA**, **SpatialVLA**, **CoT-VLA** (Zhao et al., 2025), **NORA-AC** (Hung et al., 2025), **WorldVLA** (Cen et al., 2025), **π0-FAST**, and **ThinkAct** (Huang et al., 2025). For real-world: **OpenVLA** and **π0-FAST**. For instruction-following: **SpatialVLA** and **OpenVLA** (action execution); **Gemini-2.5-Flash**, **GPT-4o**, and **HAMSTER** (Li et al., 2025) (visual trace generation). The selection covers both autoregressive VLA models (OpenVLA, RT-2-X) and diffusion/flow-matching models (π0, π0-FAST), as well as models with and without intermediate reasoning components.

- **Generation budget / compute accounting.** The paper does not use a unified "generation budget" in the style of the reference example's best-of-N framework. Instead, compute is measured in **GPU hours** for training efficiency comparisons (9,216 for MolmoAct pretraining vs. 50,000 for GR00T N1.5). At inference time, the primary cost metric is **number of training samples** used for pretraining (MolmoAct's 26.3M vs. π0's ~903M), reflecting the paper's focus on data efficiency rather than test-time compute scaling. For post-training adaptation, compute is reported as GPU hours and gradient steps per task (detailed in Tables 7 and 10–14), with LoRA fine-tuning keeping the number of trainable parameters fixed at 97M across all experiments. Inference latency is acknowledged as a limitation (Appendix G) but not systematically measured or compared across baselines.

- **Cross-validation / statistical protocol.** For SimplerEnv, the evaluation protocol follows the standard benchmark: each task is run for a fixed number of episodes with pre-specified initial conditions, and success rate is averaged. For LIBERO, the standard protocol of testing across all 10 tasks per suite with pre-defined configurations is followed, and the average success rate is reported. For real-world evaluations, 25 trials per task are conducted (except mid-training ablations which use 10 trials), with task progression scores averaged and standard error reported in bar plots (Figure 5). For instruction-following evaluations, pairwise human preference data is collected in an arena-style interface with 100 annotators for action execution (over 1,500 votes) and an unspecified number of annotators for trace generation (over 1,000 votes), with Elo ratings computed and 95% confidence intervals displayed as error bars (Figures 7 and 8). For steerability, 15 trials per condition are evaluated. The paper does not report cross-validation for strategy selection (since there is no adaptive strategy being selected — the architecture is fixed), and does not use held-out validation folds for hyperparameter tuning of the post-training stage (training continues until convergence, determined by training loss and evaluation performance). For the generalization evaluation, three variants per task are tested with four trials per variant, producing 12 trials per condition per model.

---

### Main Quantitative Results

#### Zero-Shot Performance After Pre-training (SimplerEnv)

The SimplerEnv evaluation isolates how well MolmoAct generalizes immediately after pre-training, before any task-specific adaptation. The headline result appears in Table 1: **MolmoAct-7B-D-Pretrain achieves 70.5% success rate on SimplerEnv Visual Matching tasks in the zero-shot setting**, outperforming all tested baselines including closed-source models trained on substantially more data.

**Key comparisons from Table 1 (Visual Matching, zero-shot):**

- **Vs. GR00T N1.5** (fine-tuned, not zero-shot): MolmoAct 70.5% vs. GR00T N1.5 52.4% — a +18.1 percentage point advantage, despite GR00T N1.5 being fine-tuned on the RT-1 subset of OXE and requiring ~5.4× more pretraining GPU hours.
- **Vs. π0** (fine-tuned): MolmoAct 70.5% vs. π0 58.7% — a +11.8 point gap, with MolmoAct being zero-shot while π0 is fine-tuned.
- **Vs. π0-FAST** (fine-tuned): MolmoAct 70.5% vs. π0-FAST 61.9% — an +8.6 point gap.
- **Vs. SpatialVLA** (zero-shot): MolmoAct 70.5% vs. SpatialVLA 70.0% — a narrow +0.5 point advantage, but SpatialVLA is the closest competitor and is also zero-shot.
- **Vs. Magma** (zero-shot): MolmoAct 70.5% vs. Magma 68.4% — a +2.1 point advantage.
- **Vs. OpenVLA** (zero-shot): MolmoAct 70.5% vs. OpenVLA 27.7% — a +42.8 point gap, dramatically larger than most comparisons, suggesting the direct-mapping approach of OpenVLA generalizes poorly to the Google Robot domain.

The per-task breakdown reveals an interesting pattern: MolmoAct's strongest task is **Open/Close Drawer** at 66.5% (vs. the next-best zero-shot of 83.7% from Magma — actually Magma is higher here, which is notable). For **Move Near**, MolmoAct achieves 73.8% (vs. π0 fine-tuned at 65.3%). For **Pick Coke Can**, MolmoAct achieves 71.3% (vs. SpatialVLA zero-shot at 81.0%). The task-level variance suggests MolmoAct's pre-training distribution (filtered OXE subset) is better aligned with some manipulation primitives than others, and that models like SpatialVLA may have complementary strengths on specific task types.

**After fine-tuning on RT-1:** MolmoAct-7B-D improves to 71.6% on Visual Matching (Table 1), a +1.1 point gain over the zero-shot version. This modest improvement is itself informative — the model is already strong out-of-the-box, and additional in-distribution fine-tuning provides only marginal gains. On **Variant Aggregation** (which introduces distribution shifts), fine-tuned MolmoAct achieves 72.1%, exceeding all baselines including RT-2-X (64.3%) by 7.8 points. The near-identical performance between Visual Matching (71.6%) and Variant Aggregation (72.1%) — a difference of only 0.5 points — is a strong signal of robustness: distribution shifts that degrade most models by 5–15 points leave MolmoAct essentially unaffected.

**Data efficiency context:** These results are achieved with MolmoAct pretrained on 26.3M samples (filtered OXE subset), compared to π0's 903M+ (full OXE + proprietary data) and GR00T N1.5's undisclosed but presumably large-scale dataset. The 34× data reduction while achieving superior zero-shot performance is the paper's strongest evidence for the "structured reasoning as substitute for data scale" claim.

---

#### Fast Adaptation via Post-Training (LIBERO and Real-World)

The post-training evaluation assesses how efficiently MolmoAct adapts to new tasks, domains, and embodiments through lightweight LoRA fine-tuning. The headline results are: **86.6% average success on LIBERO (Table 2)** and substantial improvements over baselines in real-world task progression (Figure 5).

**LIBERO results (Table 2):**

MolmoAct-7B-D achieves the highest overall average of 86.6%, with the per-suite breakdown:
- **LIBERO-Spatial**: 87.0% (vs. π0-FAST 96.4% — MolmoAct underperforms here, likely because spatial rearrangement tasks benefit from π0-FAST's flow-matching action representation which may be more expressive for precise positioning)
- **LIBERO-Object**: 95.4% (vs. π0-FAST 96.8% — again slightly below, continuing the pattern)
- **LIBERO-Goal**: 87.6% (vs. π0-FAST 88.6% — essentially tied)
- **LIBERO-Long**: **77.2%** (vs. π0-FAST 60.2% — a +17.0 point gap, and vs. ThinkAct 70.9% — a +6.3 point gap)

The LIBERO-Long result is the headline finding from Table 2. Long-horizon tasks require chaining multiple spatial subgoals, and this is precisely where the benefits of intermediate spatial reasoning (depth perception + trajectory planning) should manifest. MolmoAct's 77.2% vs. OpenVLA's 53.7% and Octo-Base's 51.1% represents a **~23–26 percentage point improvement** — more than a 40% relative gain. The second-best method on LIBERO-Long is ThinkAct (70.9%), which also uses a form of intermediate reasoning (visual latent planning), suggesting that explicit or latent planning is indeed the differentiating factor for long-horizon performance. The fact that π0-FAST achieves only 60.2% on LIBERO-Long despite dominating on LIBERO-Spatial (96.4%) and LIBERO-Object (96.8%) indicates a sharp tradeoff: flow-matching action representations excel at precise single-step manipulation but struggle to chain multiple steps, while structured reasoning approaches trade some single-step precision for dramatically better multi-step coherence.

**Real-world post-training results (Figure 5):**

Across six tasks spanning single-arm and bimanual Franka setups, with 25 trials per task, the results show:
- **Single-arm tasks (average over three tasks):** MolmoAct achieves 0.889 task progression vs. π0-FAST's 0.792 — a +10% relative improvement. OpenVLA is far behind at 0.348.
- **Bimanual tasks (average over three tasks):** MolmoAct achieves 0.857 vs. π0-FAST's 0.500 — a +22.7% relative improvement. OpenVLA is at 0.540.

The per-task breakdown is provided in Appendix Tables 15–20. Key observations from these detailed tables:

- On **Wipe Table** (single-arm), MolmoAct achieves a perfect 1.000 average task progression (24/24 trials scored 1.0), compared to π0-FAST's 0.817 and OpenVLA's 0.265. This is a near-ceiling result — the model executes the task flawlessly on every trial.
- On **Lift Tray** (bimanual), MolmoAct achieves 1.000 vs. π0-FAST's 0.740. Even OpenVLA achieves 1.000 on this task, suggesting it is relatively easy (a symmetrical bimanual lift with large tolerance).
- On **Set up Table** (bimanual), MolmoAct achieves 0.77 vs. π0-FAST's 0.24 — more than triple the performance. This task requires coordinated asymmetric actions (one arm places a banana, the other pours tea), which tests spatial coordination that direct-mapping models struggle with.
- On **Fold Towel** (bimanual), MolmoAct achieves 0.80 vs. π0-FAST's 0.52. The task requires one arm to press while the other folds — another asymmetric coordination challenge.
- On **Put bowl in sink** (single-arm), MolmoAct achieves 0.826 vs. π0-FAST's 0.708.
- On **Clean the table** (single-arm), MolmoAct achieves 0.84 vs. π0-FAST's 0.85 — essentially tied.

A notable pattern: the largest gaps between MolmoAct and baselines appear on bimanual tasks and tasks requiring spatial coordination (Set up Table, Fold Towel), while simple pick-and-place tasks show smaller margins. This is consistent with the hypothesis that explicit spatial reasoning (trace-based planning) is most valuable when the motion requires precise coordination in space, whereas direct mappings suffice for simpler motions.

**Post-training efficiency:** All real-world tasks use LoRA fine-tuning with only 97M trainable parameters (out of ~7.6B total), with 50 demonstrations per task and training times ranging from 3–8 GPU hours per task (Tables 10–11). The fact that MolmoAct achieves these results with such lightweight adaptation — on the order of single-digit GPU hours per task — demonstrates that the pre-trained spatial reasoning capabilities transfer efficiently to new tasks without requiring extensive retraining.

---

#### Out-of-Distribution Generalization

The generalization evaluation (Section 5.3, Figure 6a, Table 21) tests whether MolmoAct's performance holds when the deployment conditions differ from training along four axes. The headline: **MolmoAct outperforms π0-FAST by an average of +23.3% across all generalization conditions.**

**Simulation generalization (SimplerEnv Variant Aggregation, Table 1):** MolmoAct fine-tuned achieves 72.1% on variant aggregation, exceeding RT-2-X (64.3%) by 7.8 points, π0-FAST (59.0%) by 13.1 points, and SpatialVLA (65.8%) by 6.3 points. The drop from Visual Matching (71.6%) to Variant Aggregation (72.1%) is actually a slight *increase*, which may reflect noise or the specific variant conditions being slightly easier. More importantly, MolmoAct is the **only** model whose variant aggregation performance is within 1 point of its visual matching performance — all other models degrade by 2–10 points, suggesting MolmoAct's spatial representations provide genuine invariance to the specific visual shifts (lighting, textures, camera viewpoints) in the variant aggregation suite.

**Real-world generalization (Figure 6a, Table 21):**

Across three base tasks tested under five conditions each (in-distribution, language variation, spatial variation, distractors, novel objects), with four trials per condition:

- **In-distribution**: MolmoAct averages 0.792 task progression vs. π0-FAST's 0.646 and OpenVLA's 0.375.
- **Language variation**: MolmoAct 0.667 vs. π0-FAST 0.292 vs. OpenVLA 0.229. This is a +37.5 percentage point gap over π0-FAST — the largest margin in any condition. The specific language shifts (e.g., "put the green tea" instead of "put the green can," "put the fruit" instead of "put the banana") appear to severely degrade π0-FAST while MolmoAct remains relatively robust.
- **Spatial variation**: MolmoAct 0.542 vs. π0-FAST 0.458 vs. OpenVLA 0.396. The gap narrows here, suggesting spatial rearrangement is challenging for all models.
- **Distractors**: MolmoAct 0.750 vs. π0-FAST 0.542 vs. OpenVLA 0.292.
- **Novel objects**: MolmoAct 0.646 vs. π0-FAST 0.292 vs. OpenVLA 0.292. This is +35.4 points over baselines.

**The most striking individual results from Table 21:**

- On **put the sponge into the yellow plate** (novel object substitution for green can): MolmoAct achieves 0.875 vs. π0-FAST's 0.0 and OpenVLA's 0.25. π0-FAST **fails completely** on this task — zero task progression across all trials. This is the clearest evidence that MolmoAct's spatial reasoning generalizes to novel objects whereas direct mapping relies on visual appearance.
- On **put the fruit into the blue plate** (language variation, banana → fruit): MolmoAct achieves 0.625 vs. π0-FAST's 0.0625. Again, π0-FAST is near-zero, suggesting extreme brittleness to instruction rephrasing.

The pattern across conditions suggests a hierarchy of difficulty for spatial reasoning: language variation is well-handled (the spatial plan doesn't depend on word choice), novel objects are well-handled (the spatial layout is similar even if pixels differ), distractors cause moderate degradation, and spatial rearrangement causes the greatest challenge (the spatial plan itself must change). This pattern is what one would expect if spatial reasoning — rather than visual memorization — is the mechanism driving generalization.

---

#### Impact of the MolmoAct Dataset (Mid-Training Ablation)

The mid-training ablation (Section 5.4, Figure 6b, Table 22) isolates the contribution of the in-house collected MolmoAct Dataset to overall performance. The headline: **Mid-training with the MolmoAct Dataset yields an average 5.5% improvement in task progression across three real-world tasks.**

Concretely, on three tasks designed to test manipulation skills beyond simple pick-and-place:
- **Pour Tea**: MolmoAct (with dataset) 0.82 vs. MolmoAct (without dataset) 0.76 — a +6% point improvement. Both versions substantially outperform π0-FAST (0.43) and OpenVLA (0.45).
- **Close Lid**: MolmoAct (with dataset) 0.50 vs. MolmoAct (without dataset) 0.45 — a +5% point improvement. π0-FAST achieves 0.50 (tied), OpenVLA achieves 0.30.
- **Rotate Pot**: MolmoAct (with dataset) 0.96 vs. MolmoAct (without dataset) 0.90 — a +6% point improvement. π0-FAST achieves 0.78, OpenVLA achieves 1.00 (surprisingly strong — possibly because rotation is a relatively simple 1-DOF motion that a well-trained direct-mapping model can execute reliably).

The 5.5% average improvement is modest but consistent — every task shows a positive gain. More importantly, even the version **without** the MolmoAct Dataset outperforms π0-FAST by 14.8% and OpenVLA by 10.9% on average across these three tasks (author-reported numbers in Section 5.4), confirming that the core architecture's advantages are not dependent on the custom dataset. The MolmoAct Dataset provides an incremental boost that refines the model's performance on domain-specific household tasks.

A nuanced reading of the per-trial results (Table 22) reveals that Close Lid is a challenging task — even the best models only achieve 0.50 task progression (the scoring rubric awards 0.5 for "move the lid toward closing direction" and 1.0 for "close the lid"). This suggests the task involves fine motor control that pushes the limits of the current action representation. The paper's acknowledged limitation about 2D trace steering lacking depth precision (Appendix G) may be relevant here — closing a lid requires precise out-of-plane motion that a 2D trace cannot fully specify.

---

#### Instruction Following (Language Grounding)

The instruction-following evaluation (Section 5.5, Figures 7 and 8) tests how well models execute open-ended natural language commands, using human preference judgments rather than predefined success criteria. This is significant because it moves beyond the standard benchmark paradigm of fixed instruction templates toward more realistic, varied language use.

**Action execution (Figure 8):** In a head-to-head evaluation on 29 open-ended instructions across five SimplerEnv scenes, with over 1,500 pairwise human votes converted to Elo ratings:
- **MolmoAct achieves the highest Elo rating**, outperforming SpatialVLA by 109 points and OpenVLA by an even larger margin (exact Elo values are not reported in the main text, only displayed in Figure 8's bar chart with error bars showing non-overlapping 95% confidence intervals).
- Pairwise win rates: MolmoAct wins against SpatialVLA in 58% of comparisons and against OpenVLA in 81% of comparisons.

The qualitative example in Figure 8 (right panel) shows a representative case: for "Put the redbull into the bowl," MolmoAct correctly moves the Red Bull can toward the red bowl, while the comparison model veers off-target. The trace visualizations make the difference interpretable — MolmoAct's predicted trajectory heads directly toward the correct target.

**Visual trace generation (Figure 7):** On 87 language prompts for 30 internet-sourced images, evaluated by over 1,000 pairwise votes:
- MolmoAct achieves the **highest Elo rating** among all tested models, with error bars (95% CI) that do not overlap with any baseline — a statistically robust result.
- Baselines include Gemini-2.5-Flash, GPT-4o, and HAMSTER (a VLM specifically fine-tuned for trace generation).

This is a particularly strong result because the trace generation task is out-of-distribution for MolmoAct: the images are internet-sourced manipulation scenes, not robot-mounted camera views. The fact that MolmoAct outperforms models like GPT-4o and Gemini-2.5-Flash on spatial trajectory generation — despite those models likely having seen far more data — suggests that MolmoAct's explicit training on spatial reasoning transfers to novel visual domains. The example qualitative results in Figure 7 (right) show predicted traces overlaid on robot camera views, demonstrating that the model produces plausible, task-relevant trajectories.

**Open-ended prompts from participants (Table 8):** The 29 prompts span a range of complexity, from simple object identification ("Pick up the green cube") to multi-step compositional tasks ("Put the green cube onto the blue cube and then the red cube onto the green cube") and relational instructions ("Move the blue cube next to the green cube"). The fact that MolmoAct succeeds across this range — evaluated by human judges, not scripted success detectors — is evidence that the language grounding generalizes beyond the templated instructions in standard benchmarks.

---

#### Action Steerability

The steerability evaluation (Section 5.6, Figure 9, Table 23) tests whether users can guide the robot's behavior when initial language instructions are ambiguous, using either rephrased language commands or drawn visual traces. The headline: **Visual trace steering achieves 75% success rate vs. 42% for open-ended language steering — a +33 percentage point gap — and MolmoAct with language steering alone still outperforms π0-FAST by 29 points.**

**Experimental setup and results (Table 23):**

In the pick_up_bowl task with two bowls (clean and dirty), the model is initially given an ambiguous instruction (e.g., "pick up the bowl") and predicts a trajectory toward one bowl. Steering is then applied to redirect the robot toward the other bowl. Three conditions are tested over 15 episodes each:

1. **MolmoAct with visual trace steering**: 75% success rate (11.25/15 episodes, based on the raw scores in Table 23: 7 episodes scored 1.00, 3 scored 0.85, 2 scored 0.50, 1 scored 0.00 — average approximately 0.73 if computed as raw mean, but the paper reports 0.75 success for binary task completion).

2. **MolmoAct with open-ended language steering**: ~42% success rate (the paper reports a 33% margin vs. visual trace steering, implying 75% - 33% = 42% for language steering).

3. **π0-FAST with language steering**: ~13% success rate (the paper reports MolmoAct with language steering surpasses π0-FAST by 29%, implying 42% - 29% = 13% for π0-FAST).

**The open-ended language instructions (Table 9):** 15 instructions were collected from 10 participants who were restricted to modifying only verbs, nouns, or adjectives from the ground-truth instruction. Examples include: "pick up the orange bowl," "lift up the dirty bowl," "pick up the bowl on the left," "pick up the empty bowl," "pick up the bowl with tissue," "pick up the bowl that is pink" (this last one is listed twice in the table, which may be a typo). These instructions range from spatially precise ("on the left") to attribute-based ("orange," "empty") to ambiguous ("further," "nearer to the camera"), reflecting the diversity of natural language strategies users employ.

**Interpretation of the results:** The +33 point gap between visual trace and language steering is the central finding. It quantifies the precision advantage of spatial over linguistic interaction modalities — when users need to specify *which* of two visually similar objects to target, drawing a line to the correct one is dramatically more reliable than finding the right words. The +29 point gap between MolmoAct and π0-FAST under language steering suggests that even when using language, MolmoAct's training on trajectory-conditioned actions (which teaches the model to associate traces with motions) improves its language grounding — the model has learned that language instructions and visual traces are alternative ways of specifying the same underlying motion intent.

The per-episode breakdown in Table 23 reveals substantial variance in language steering effectiveness: some episodes achieve 1.0 task progression with language alone (episodes 1 and 14), while others fail completely (episodes 0, 2, 5, 6, 7, 13). This variance reflects the inherent ambiguity of language — some phrasings happen to work well, others fail entirely. Visual trace steering is more consistent: 11 of 15 episodes achieve at least 0.5 task progression (moving toward the correct bowl), and 7 episodes achieve perfect 1.0 scores.

---

### Ablation Studies and Robustness Checks

**Action tokenization strategy (implied, not explicitly ablated):** The paper does not present a formal ablation comparing the byte-level BPE similarity-preserving action tokenization against standard random token assignment. The claim that this "substantially reduces training time" is supported only by the aggregate training efficiency comparison (MolmoAct's 9,216 GPU hours vs. GR00T N1.5's 50,000 GPU hours), which confounds many factors beyond action tokenization (model architecture, data mixture, training procedure). A clean ablation would involve training MolmoAct with randomly assigned action tokens and measuring the convergence speed difference, but this is not reported.

**Multi-modal web data co-training (implied, not explicitly ablated):** The paper includes 5% multimodal web data in the pretraining mixture to prevent catastrophic forgetting, but does not present a version trained without this data. The contribution of web data co-training to final performance is therefore unknown — it is possible that the web data provides benefits beyond catastrophic forgetting prevention (e.g., improved visual understanding that transfers to manipulation), but the paper provides no evidence either way.

**LoRA rank and adaptation capacity:** All post-training experiments use a fixed LoRA rank of 32 and alpha of 16. The paper does not ablate this choice, so it is unknown whether higher ranks (which would increase adaptation capacity at the cost of more parameters) would yield better fine-tuning performance, or whether lower ranks would suffice. The consistency of results across diverse tasks (single-arm, bimanual, simulation, real-world) suggests the chosen rank is sufficient, but the absence of a sweep makes it unclear whether performance is bottlenecked by adaptation capacity on the most challenging tasks.

**Action chunk size:** The paper uses K=8 action chunking for all post-training evaluations (LIBERO and real-world). No ablation of chunk size is reported. Larger chunks reduce inference frequency but increase the open-loop execution duration, which could hurt performance on tasks requiring rapid feedback. The choice of 8 is standard (following Zhao et al., 2023) but its optimality for MolmoAct specifically is untested.

**Single-view vs. multi-view inputs (implied):** During mid-training and post-training, the model receives dual-camera inputs (side view + wrist view). The paper does not ablate the contribution of the wrist camera — it is possible that the side view alone would suffice, or that the wrist view provides critical close-up detail. The ablation in Appendix D.5, where depth tokens and traces are generated "only from the side views" while the wrist view is "solely for providing additional information," suggests the wrist view's contribution is auxiliary but untested.

**Number of depth perception tokens (acknowledged limitation, not ablated):** The paper uses a fixed 100-token depth representation (M=100). Appendix G notes that "fine-grained manipulation tasks require higher-resolution depth estimation," but no experiment varies the token count to measure its impact. This is an acknowledged direction for future work rather than a tested ablation.

**Visual trace length (L=1 to 5):** The trace representation uses 1–5 points. The paper does not ablate the maximum trace length — it is possible that longer traces (more intermediate points) would provide finer-grained guidance, or that shorter traces (just start and end points) would suffice. This choice is likely constrained by the sequence length budget, but the tradeoff is not quantified.

**VQVAE codebook size (N=128):** The depth codebook has 128 entries. No ablation of codebook size is presented. Larger codebooks would provide finer depth granularity at the cost of a larger depth vocabulary and potentially harder learning. The paper's acknowledged limitation about depth precision (Appendix G) suggests the current 128-codebook size may be a bottleneck for tasks requiring precise depth discrimination.

**Distillation vs. end-to-end depth learning:** The paper uses a specialist-to-generalist distillation strategy where a separately trained VQVAE produces depth tokens that MolmoAct learns to predict. An alternative would be to train the depth prediction end-to-end as part of the VLM (with a depth reconstruction loss). The paper does not ablate these approaches, so it is unclear whether the distillation step is necessary or merely convenient.

---

### Critical Assessment

**Claim 1: MolmoAct achieves strong zero-shot performance, surpassing closed-source π0 and GR00T N1.5.**

The evidence in Table 1 supports this claim with important caveats. MolmoAct achieves 70.5% zero-shot on SimplerEnv Visual Matching, indeed higher than GR00T N1.5's 52.4% and π0's 58.7%. However, these baselines are presented as **fine-tuned** on RT-1, not zero-shot — meaning MolmoAct is being compared favorably to models that have seen *more* task-relevant data, which makes the result stronger, not weaker. The caveat is the comparison granularity: SimplerEnv Visual Matching contains only three tasks on a single embodiment (Google Robot). This is a narrow test of "zero-shot performance" — three tasks from the same domain the model was partially pretrained on (RT-1 is in the pretraining mixture). A broader zero-shot evaluation across multiple embodiments and task families (as provided in the original OXE evaluations) would more rigorously test the claim. The variant aggregation results (72.1% for MolmoAct, 59.0% for π0-FAST) provide stronger evidence of out-of-distribution robustness.

**Claim 2: 86.6% average success on LIBERO, with +6.3% over ThinkAct on long-horizon tasks.**

The LIBERO results (Table 2) are the paper's strongest quantitative evidence. The 86.6% average is genuinely state-of-the-art among autoregressive models. The LIBERO-Long result (77.2%) is the most informative number: it isolates the contribution of spatial reasoning to multi-step task execution, and the +6.3 point gap over ThinkAct (the closest method in architecture, also using intermediate reasoning) and +17 point gap over π0-FAST support the paper's thesis about the value of explicit spatial planning. However, two observations temper the claim:

1. On LIBERO-Spatial and LIBERO-Object, π0-FAST outperforms MolmoAct (96.4% vs. 87.0% and 96.8% vs. 95.4%). The paper's architecture trades off single-step precision for multi-step coherence — which is a reasonable tradeoff, but it means MolmoAct is not uniformly superior. The claim of "state-of-the-art" should be qualified: state-of-the-art on long-horizon tasks, competitive but not dominant on shorter tasks.

2. The LIBERO evaluation fine-tunes on all 500 demonstrations per suite. The paper does not evaluate how performance varies with the number of demonstrations. A data-efficiency curve (performance vs. number of demos) would strengthen the claim that structured reasoning improves sample efficiency, which is a central thesis of the paper.

**Claim 3: +10% single-arm and +22.7% bimanual task progression over π0-FAST in real-world fine-tuning.**

Figure 5 and Tables 15–20 provide detailed evidence. The real-world results are the paper's most practically significant contribution because they demonstrate deployment on physical hardware. Several aspects strengthen credibility: per-task scoring rubrics are provided (Appendix D.3), per-trial scores are reported (Tables 15–20), and 25 trials per task with standard error bars provide reasonable statistical power.

However, there are genuine limitations:

- **Six tasks total** across two embodiments is a small evaluation set for claiming general real-world capability. The tasks are well-chosen for diversity (single-arm, bimanual, varied manipulation types), but the total evaluation budget is 150 trials per model — moderate but not exhaustive.
- **π0-FAST is not fine-tuned to convergence by the authors** — the paper states "we follow their official model and training implementation and use their default configurations. We also make sure that they are all fully converged" (Appendix D.3). This is reasonable, but optimal hyperparameters for π0-FAST on these specific tasks might differ from defaults, and the convergence check is based on the authors' judgment, not a systematic sweep.
- **The bimanual results show high variance** — on Set up Table, MolmoAct's per-trial scores (Table 17) range from 0.00 to 1.00, with a standard deviation that appears large (the exact value is not reported, but visual inspection shows scores of 0.00, 0.25, 0.50, 0.75, and 1.00 all appearing multiple times). High variance with only 25 trials means the +22.7% average advantage should be interpreted cautiously — a few lucky or unlucky trials could shift the mean substantially.
- **OpenVLA severely underperforms on single-arm tasks** (0.25, 0.265, 0.53 in Tables 18-20). These scores are near the "grasp object" level (0.25) for two of three tasks, suggesting OpenVLA may not have been properly adapted for the specific camera setup or action space. If the baseline is performing anomalously poorly, the gap over it is less informative.

**Claim 4: +23.3% out-of-distribution generalization over baselines.**

The evidence in Figure 6a and Table 21 is well-designed, testing four distinct generalization axes on three tasks. The per-condition breakdown is informative and the novel-object result (π0-FAST achieving 0.0 on "put the sponge") is striking. Caveats:

- **Small trial count**: four trials per condition, three tasks per condition, three variants per task = 12 trials per condition per model. This is a thin statistical foundation for claiming a 23.3% advantage. Confidence intervals are not reported for the generalization results.
- **The specific objects and perturbations are not systematically varied** — the sponge substitution is one example of a novel object, but the result might not generalize to other novel objects. A broader set of substitutions would strengthen the claim.
- **The paper evaluates only one baseline (π0-FAST) for real-world generalization** (OpenVLA is also evaluated, but as a weaker baseline). Comparison against other models with intermediate representations (e.g., SpatialVLA, ThinkAct) on the generalization suite would reveal whether the benefit is specifically from MolmoAct's depth+trace architecture or from any form of spatial reasoning.

**Claim 5: Top human-preference scores for instruction following and trajectory steering.**

The Elo-based evaluations (Figures 7, 8) use a sound methodology — pairwise comparisons with blind human raters, over 1,500 votes, with 95% confidence intervals. The non-overlapping error bars in Figure 7 (trace generation) provide statistically robust evidence. Two limitations:

1. **The action execution evaluation uses only 29 prompts across five scenes** (Table 8). This is a small prompt set, and the prompts were written by only 10 participants. Broader linguistic diversity would strengthen the claim.
2. **The trace generation evaluation uses internet images, not robot-mounted camera views**. This shows impressive generalization (traces for tabletop scenes the model has never seen from a robot's perspective), but the relevance to actual robot deployment is indirect — the evaluation measures trace quality, not whether those traces lead to successful actions.

**Claim 6: Mid-training with the MolmoAct Dataset yields a 5.5% improvement.**

The ablation in Figure 6b and Table 22 is clean — same model, with and without the dataset, evaluated on three tasks. The 5.5% average is a modest but consistent gain. The per-trial results (Table 22) show this is not an artifact of a few outlier trials — the improvement is distributed across tasks and trials. The claim is well-supported for the specific tasks tested. The limitation is, again, the narrow task set (three tasks) and the small trial count (10 per task).

**Missing experiments that would have strengthened the paper:**

- **Data scaling curves**: The central thesis is that structured reasoning improves data efficiency, but the paper never shows performance as a function of pretraining data quantity for MolmoAct vs. baselines. A plot of SimplerEnv or LIBERO accuracy vs. number of pretraining samples, comparing MolmoAct to OpenVLA or π0, would directly test the "substitute for data scale" claim.
- **Ablation of individual reasoning stages**: The paper never reports MolmoAct performance with (a) no depth tokens, (b) no visual trace, (c) neither. This is the most conspicuous missing ablation — without it, we cannot attribute the performance gains to specific components of the architecture. It is possible that most of the benefit comes from the trace alone (since it provides explicit spatial guidance) while the depth tokens contribute marginally, or vice versa.
- **Comparison to MolmoAct without action reasoning pipeline**: How does a version of MolmoAct trained as a direct VLA (image + text → action, no depth or trace tokens) perform? This within-family ablation would isolate the contribution of the reasoning architecture from the VLM backbone and training data.
- **Generalization to unseen embodiments**: All evaluations use the same embodiments the model was trained on (Google Robot for pre-training, Franka for post-training). The paper claims "adaptability across embodiments" (Section 5, research question 2), but only tests the two embodiments it trains on. A true cross-embodiment evaluation (e.g., fine-tuning on a Franka task and testing on a KUKA or UR5) would test this claim.
- **Alternative depth representations**: The paper uses VQVAE-discretized depth tokens. How does this compare to (a) raw depth values as continuous tokens, (b) no depth tokens at all, (c) depth as an auxiliary loss rather than autoregressive tokens? These comparisons would clarify whether the specific depth tokenization is important or any depth signal suffices.

**Conditions under which claims hold:**

The paper's results are strongest when:
- Tasks require **multi-step spatial coordination** (LIBERO-Long, bimanual real-world tasks) — this is where spatial reasoning provides the largest benefit.
- **Distribution shift involves visual appearance changes** (variant aggregation, novel objects, language variation) — spatial representations are invariant to these shifts.
- **Pretraining data is limited** — the comparison against π0 (34× more data) shows structured reasoning compensating for data scarcity.

The claims weaken when:
- Tasks require **precise single-step positioning** (LIBERO-Spatial, where π0-FAST outperforms) — the flow-matching action representation may be inherently more precise for fine positioning.
- **Inference latency matters** — the autoregressive generation of 102 depth tokens + trace points + action tokens per timestep introduces latency that the paper acknowledges but does not quantify.
- **The base VLM lacks the target capability** — the hardest tasks (e.g., Close Lid at 0.50 task progression for all models) may be beyond what the current generation of 7B VLMs can handle regardless of architecture.

## 6. Limitations and Trade-offs

### Depth and Trace Reasoning Stages Are Not Ablated Individually

**The assumption or constraint.** MolmoAct's architecture introduces two distinct intermediate reasoning stages — depth perception tokens and visual reasoning traces — and the paper attributes performance gains to "reasoning in space" as a unified concept. However, **no experiment isolates the contribution of each reasoning stage individually**. The paper never reports performance for a MolmoAct variant with only depth tokens (no trace), only trace tokens (no depth), or neither (a direct perception-to-action baseline within the same model family).

**The consequence.** Without component-level ablations, the central causal claim — that spatially grounded reasoning *causes* the observed improvements — is supported by correlation rather than direct evidence. It is entirely possible that most of the benefit derives from a single component (e.g., the visual reasoning trace provides explicit spatial guidance for action prediction, while the depth tokens contribute marginally or not at all), or that the benefit comes from some other aspect of the training procedure (e.g., the auxiliary depth and trace datasets act as regularizers or provide additional training signal independent of their semantic content). The paper's strongest comparisons are against entirely different model architectures (π0-FAST, OpenVLA), which confound many factors beyond the presence or absence of spatial reasoning.

A practitioner evaluating whether to adopt this architecture needs to know whether both reasoning stages are necessary. Implementing depth tokenization requires training a VQVAE on domain-specific depth maps and extending the LLM's vocabulary — non-trivial engineering effort. If similar performance could be achieved with only the visual trace (which requires only VLM-based pointing, a simpler pipeline), the depth component's cost-benefit ratio changes substantially.

**What evidence exists in the paper.** The paper provides only indirect evidence. The auxiliary depth and trace datasets are described as providing "focused training on specific skills" (Section 3.1), but their individual contributions to final task performance are never measured. The LIBERO-Long result (Table 2: 77.2% vs. 53.7–70.9% for baselines without spatial reasoning) demonstrates that the full architecture outperforms non-reasoning models, but this is a bundled treatment effect. The per-stage qualitative examples in Appendix F (Figures 13–16) show that the model can produce plausible depth maps and traces, but not whether removing either degrades action prediction.

**Mitigation status.** Not addressed. The paper does not acknowledge the absence of component-level ablations as a limitation, and no future work is proposed to disentangle the contributions of depth and trace reasoning. This is a significant gap given that the architecture's central claim is about the *combination* of these representations.

---

### Difficulty Estimation Cost Is Not Accounted For in the Headline Comparison

**The assumption or constraint.** The paper's comparison against baselines like π0-FAST and GR00T N1.5 emphasizes MolmoAct's data efficiency: 26.3M pretraining samples vs. π0's ~903M+, and 9,216 GPU hours vs. GR00T N1.5's 50,000 GPU hours. However, the **training procedure for MolmoAct depends on a specialist depth estimator (Depth-Anything-v2) and a pre-trained pointing VLM (Molmo) to generate pseudo-labels for depth tokens and visual reasoning traces**. The computational cost of running these specialist models on the full pretraining dataset — 10.5M robot samples, each requiring a depth map from Depth-Anything-v2 and pointing predictions from Molmo at every timestep — is not included in the reported GPU hours.

**The consequence.** The headline data efficiency numbers (34× fewer pretraining samples, 5.4× fewer GPU hours) are **overstated** in a practical sense because they exclude the cost of producing the intermediate supervision. Depth-Anything-v2 must be run on every frame of every demonstration (10.5M samples × average episode length) to generate VQVAE codebook indices. Molmo must be queried at every timestep for gripper pointing. These are amortized costs (run once during dataset preparation), but they are real infrastructure and compute requirements that a team attempting to reproduce MolmoAct would need to incur. The paper treats these as "free" inputs, but they represent a form of privileged information — a pre-existing depth estimator and a pre-existing pointing VLM — that simpler baselines (OpenVLA, π0) do not require for training.

Moreover, the quality of the depth tokens and trace labels depends on the quality of the specialist models. If Depth-Anything-v2 produces inaccurate depth maps (e.g., on visually complex scenes, transparent objects, or reflective surfaces), those errors become baked into MolmoAct's training targets. The paper provides no analysis of depth estimation accuracy or pointing accuracy on the specific robot datasets used, so the extent of label noise is unknown.

**What evidence exists in the paper.** The paper describes the specialist-to-generalist distillation pipeline in Section 3.1 and Appendix D, but never reports the computational cost of generating the depth and trace pseudo-labels. The training compute table (Table 4) reports only the GPU hours for MolmoAct's pretraining, mid-training, and post-training — not the preprocessing cost. The VQVAE is trained on 10 million depth maps from RT-1, BridgeData V2, and BC-Z for 20 epochs, which requires additional GPU hours that are reported in aggregate but not quantified numerically. The paper mentions Depth-Anything-v2 as the source of depth maps without reporting its inference cost or accuracy on the specific datasets.

**Mitigation status.** The authors acknowledge that the VQVAE training and pseudo-label generation incur costs (the VQVAE training is described in Appendix D), but these costs are never factored into the efficiency comparisons against baselines. No suggestion is made for reducing this overhead (e.g., using a lighter-weight depth estimator, amortizing labeling across datasets, or training the depth prediction end-to-end without a separate specialist). This is an important caveat for any team attempting to replicate the training pipeline.

---

### 2D Trace Steering Lacks Depth Precision for Out-of-Plane Motion

**The assumption or constraint.** The steerability mechanism — a major selling point of the paper — operates on **purely 2D visual traces overlaid on the RGB image**. The paper states in Appendix G:

> "Because the cue is purely 2D, the model lacks an explicit notion of depth: it often follows the intended path within the image plane (in-plane motion) but exhibits unintended or imprecise translation along the camera's depth axis (out-of-plane)."

This is an explicit acknowledgment of a fundamental geometric limitation: a point at pixel coordinates (u, v) specifies a ray in 3D space, not a unique 3D position. The model must infer depth from the visual context, but the trace itself provides no depth information.

**The consequence.** For tasks requiring precise depth control — approaching a surface, inserting an object into a container, closing a lid, or navigating around obstacles in 3D — 2D trace steering may be insufficient or even misleading. The user can indicate "move the gripper to this point in the image," but cannot specify "move to this point at depth 45 cm, not 60 cm." If the model's inferred depth is incorrect, the robot may collide with objects, miss grasp targets, or fail to make contact.

The paper's own results illustrate this limitation in practice. The **Close Lid** task (Table 22) achieves only 0.50 task progression even for the best model — the scoring metric awards 0.5 for "move the lid toward closing direction" and 1.0 for "close the lid." The model consistently moves in the right direction but fails to complete the closure, which requires precise out-of-plane motion to push the lid fully shut. The acknowledged limitation about depth axis imprecision is a plausible explanation for this failure mode.

More broadly, the steerability evaluation in Section 5.6 tests only a single task (pick_up_bowl) where the primary challenge is lateral discrimination (clean vs. dirty bowl) rather than depth precision. The +33 percentage point advantage of visual trace steering over language steering demonstrates that 2D traces are highly effective for *in-plane* spatial disambiguation, but this result may not generalize to tasks where depth control is the primary challenge.

**What evidence exists in the paper.** The limitation is acknowledged in Appendix G. The trajectory-conditioned action data used for steerability training (Section 3.1) overlays 2D traces on the RGB image — there is no depth channel in the trace representation. The steerability experiments (Section 5.6, Figure 9, Table 23) evaluate only in-plane discrimination tasks. The Close Lid results (Table 22, 0.50 task progression ceiling) are consistent with a depth-precision bottleneck, though the paper does not explicitly connect these results to the 2D trace limitation.

**Mitigation status.** The paper proposes a direction for future work in Appendix G: "We hypothesize this could be mitigated by conditioning on — or reusing — the model's predicted depth-perception tokens to lift the trace into 3D, which we leave for future exploration." This is a reasonable suggestion — the depth perception tokens are already being generated by the model in autonomous mode, and they could theoretically be used to constrain the depth dimension during trace following — but it is entirely unimplemented and unevaluated. In the current system, steerability and depth reasoning are used exclusively in the autonomous pipeline, and no version of MolmoAct uses depth tokens during steering. The limitation is acknowledged but not resolved.

---

### Single Benchmark Domain and Narrow Real-World Task Diversity

**The assumption or constraint.** All evaluations — simulation and real-world — are conducted on **tabletop and household manipulation tasks with rigid or semi-rigid objects**, using a narrow range of robot embodiments (Google Robot in simulation, Franka Emika Panda in real-world). The pretraining data is drawn from a filtered subset of Open X-Embodiment (specifically BC-Z, BridgeData V2, and RT-1), all of which feature similar tabletop manipulation scenarios with single-arm robots.

**The consequence.** The paper's claims about "generalist" capability and "adaptability across embodiments" are evaluated within a narrow domain. It is unknown whether MolmoAct's spatial reasoning architecture — specifically, the depth tokenization (trained on tabletop depth maps at 320×320 px resolution) and the visual trace representation (designed for tabletop-scale end-effector motions) — would transfer to:
- **Mobile manipulation** (navigating through rooms, opening doors, manipulating objects at varying heights and distances beyond tabletop range).
- **Deformable object manipulation** (folding laundry, handling cables, food preparation) where the trace-to-action mapping may require finer-grained force control.
- **Precision tasks** (peg insertion, screw driving, surgical manipulation) where the 256-bin action discretization or the 100-token depth representation may lack sufficient resolution.
- **Dynamic tasks** (catching, throwing, juggling) where the autoregressive reasoning pipeline (generating 102 depth tokens + trace points before each action) introduces latency incompatible with real-time control.
- **Navigation-heavy tasks** where the robot's motion extends beyond the field of view of a single camera and the trace representation (1–5 points on a single image) cannot capture the full trajectory.

The paper's generalization results (Section 5.3) demonstrate robustness to within-domain distribution shifts (language variation, object substitution, spatial rearrangement of known objects), but not to domain transfer. The SimplerEnv variant aggregation results (Table 1) test robustness to lighting, texture, and viewpoint changes within the same task structure — again, within-domain generalization rather than cross-domain transfer.

**What evidence exists in the paper.** The three evaluation testbeds are: SimplerEnv Google Robot tasks (3 tasks, single embodiment), LIBERO (40 tasks across 4 suites, single simulated Franka Panda embodiment), and a custom real-world setup with 6 + 3 + 3 = 12 tasks on single-arm and bimanual Franka Pandas. The MolmoAct Dataset (Section 3.2) includes 93 tasks, but these are exclusively household and tabletop manipulation tasks (verbs include "put," "turn," "close," "wipe," "pour"). The paper does not evaluate on benchmarks like CALVIN (long-horizon with diverse skills), RLBench (diverse manipulation with task variation), or mobile manipulation suites. The depth VQVAE is trained on 10 million depth maps from RT-1, BridgeData V2, and BC-Z (Section 3.1) — all tabletop manipulation scenes at fixed 320×320 px resolution. The visual trace representation uses 1–5 points on a single image, which is inherently limited to motions visible within a single camera frustum.

**Mitigation status.** The paper does not claim cross-domain generalization. The abstract and introduction position MolmoAct as a model for "robotic manipulation" and evaluate it on manipulation benchmarks. However, the term "generalist" is used in Section 6.1 ("generalist robot manipulation policies") and the research questions in Section 5 ask about "adaptability across embodiments" — language that implies broader capability than what is actually tested. The limitation of embodiment diversity is partially addressed by testing both single-arm and bimanual Franka setups, but these share the same robot platform and differ primarily in degrees of freedom rather than kinematics or sensor configuration. No mitigation is proposed, and the paper presents the evaluation scope as comprehensive rather than narrow.

---

### No Accounting for Inference Latency in the Reasoning Pipeline

**The assumption or constraint.** MolmoAct's autoregressive reasoning pipeline generates **102 depth tokens followed by 1–5 trace point tokens followed by action tokens** at every inference step before producing a motor command. The paper acknowledges this latency issue in Appendix G:

> "Similar to many existing VLAs, our model exhibits a mismatch between its control inference frequency and the control frequency used during data collection. This gap may stem from server-to-robot communication latency and the additional time required to predict a larger number of reasoning tokens."

However, **no latency measurements are reported anywhere in the paper** — not for the full reasoning pipeline, not for the steerability mode (which skips depth and trace generation), and not in comparison to baselines.

**The consequence.** The practical deployability of MolmoAct depends critically on inference latency. If generating 102 depth tokens + trace points + action tokens takes, say, 500 ms per inference step, while the robot's control loop requires 50 ms, then the model can only operate at 2 Hz rather than the 20 Hz typical of the demonstration data. This introduces a **distribution shift at deployment**: the model was trained on data collected at 15–20 Hz, but executes at a lower frequency, meaning each action must cover a larger displacement and may encounter conditions (obstacles, object motion) that did not exist at the time of the last observation.

The action chunking strategy (predicting 8 actions at once and executing them open-loop) partially mitigates this by amortizing inference cost over 8 control steps. However, this trades off closed-loop reactivity — if an object moves or the robot deviates from the planned trajectory during the 8-step open-loop execution, the model cannot correct until the next inference call. The effective control frequency becomes inference frequency divided by chunk size, which could be very low for complex scenes requiring long reasoning sequences.

For the steerability use case, latency is particularly critical. The paper envisions interactive steering where a user draws a trace and the robot immediately adjusts its trajectory. If each adjustment requires a full inference pass through a 7B-parameter LLM before the robot moves, the interaction may feel sluggish and unusable for real-time correction.

**What evidence exists in the paper.** The paper provides no latency numbers. The only acknowledgment is the Appendix G statement quoted above. The training data collection frequencies are reported (15 Hz for home environment data, 20 Hz for tabletop data, Appendix E.1), but the inference frequency is not reported for any evaluation setting. The GPU hours for training are carefully documented (Tables 4, 7, 10–14), but inference time is omitted. The paper does not specify whether inference runs on the same H100 GPUs used for training or on lower-powered edge hardware.

**Mitigation status.** Appendix G suggests future work could "explore techniques to reduce inference time, as seen in VLM optimization, or develop smaller parameter models optimized for efficient execution on edge or local devices." This is a reasonable direction but entirely unimplemented. The paper provides no evidence that the current system operates at a deployable frequency, and no analysis of where the latency bottleneck lies (vision encoder, autoregressive token generation, token decoding, network communication). For a paper that positions steerability and real-world deployment as key contributions, the absence of latency characterization is a significant gap.

---

### Real-World Evaluations Use Small Trial Counts with High Variance

**The assumption or constraint.** The real-world evaluations — which provide the paper's most practically significant results — rely on **25 trials per task for the main post-training comparison, 10 trials per task for the mid-training ablation, and 4 trials per condition for the generalization evaluation**. Several tasks exhibit high per-trial variance, with scores oscillating between complete failure (0.0) and perfect success (1.0) across trials for the same model and task.

**The consequence.** The headline claims of "+10% single-arm" and "+22.7% bimanual" improvement over π0-FAST are based on averages over small samples with substantial variance, making them **potentially fragile to sampling noise**. A small number of outlier trials — particularly on tasks with binary or coarse-grained scoring — can shift the mean significantly.

Consider the concrete evidence from the per-trial tables:
- **Set up Table** (bimanual, Table 17): MolmoAct scores include 0.00, 0.25, 0.50, 0.75, and 1.00 across 25 trials. The mean is 0.77, but the distribution is essentially uniform across all possible scores — this task has extremely high variance. π0-FAST scores include 0.00, 0.25, 0.50, and 0.75 (mean 0.24). The +0.53 difference in means is based on 25 trials each with a standard deviation that appears to be ~0.3–0.35 for MolmoAct (rough estimate from the score distribution). With n=25 and σ≈0.35, the standard error of the mean difference is approximately √(2 × 0.35² / 25) ≈ 0.10. The +0.53 gap is statistically significant (roughly 5 standard errors), but the confidence interval is wide: the true difference could plausibly be anywhere from ~+0.33 to ~+0.73 — a factor of 2× range.
- **Fold Towel** (bimanual, Table 15): MolmoAct mean 0.80, but scores range from 0.25 to 1.00. The 0.25 scores (6 of 24 trials) represent the model only reaching the first milestone (grasping the towel) — a substantial failure mode that occurs in 25% of trials.
- **Close Lid** (mid-training ablation, Table 22): All models achieve binary scores of either 0.0, 0.5, or 1.0, with the best mean being 0.50. With n=10 and σ≈0.3–0.4, the standard error is ~0.1–0.13 — the +0.05 difference between MolmoAct with and without the dataset is well within one standard error and cannot be distinguished from noise.

**What evidence exists in the paper.** The per-trial scores are provided in Tables 15–23, which is commendable transparency. The bar plots in Figures 5 and 6 display standard error bars, confirming the authors are aware of the variance. However, the paper reports only **means and standard errors** — no confidence intervals for the differences between models, no hypothesis tests, and no discussion of whether the observed differences are statistically significant given the sample sizes. The generalization evaluation (Table 21) uses only 4 trials per condition (3 tasks × 4 trials = 12 per condition), making the +23.3% average improvement claim particularly sensitive to individual trial outcomes.

**Mitigation status.** The paper does not acknowledge sample size as a limitation. The 25-trial protocol is standard in robotics research (and exceeds many prior works that use 10–20 trials), but the combination of high per-trial variance and small samples means the quantitative comparisons should be interpreted as **indicative rather than precise**. The paper would benefit from either larger trial counts (50–100 per task) for the headline comparisons, or explicit reporting of confidence intervals for between-model differences, or both. The consistency of MolmoAct's advantage across multiple independent tasks provides convergent evidence that partially compensates for the per-task sample size, but the specific numerical margins (especially the +22.7% bimanual claim) should be treated as approximate.

## 7. Implications and Future Directions
- How this changes the landscape
  - Establishes “reasoning in space” as a practical, scalable alternative to text‑only CoT for robotics. By making intermediate perceptions and plans explicit and decodable, it improves interpretability, controllability, and generalization (Figure 1; Eq. 4–5).
  - Demonstrates that explicit spatial reasoning can reduce brittleness to visual shifts (Table 1) and improve long‑horizon performance (Table 2).
- Next research steps enabled or suggested
  - Lift 2D traces into 3D using the predicted depth tokens, enabling richer test‑time steering and safer manipulation in clutter (Appendix G).
  - Learn higher‑fidelity or adaptive depth tokenizations (beyond fixed 100 tokens) for precision tasks.
  - Close the control‑rate gap with efficient, smaller ARMs or specialized accelerators; explore low‑latency on‑robot inference.
  - Extend to mobile manipulation and multi‑robot coordination by composing traces across views and agents; integrate proprioception when useful.
  - Formalize safety constraints and user intent alignment using editable traces and depth tokens as verifiable interfaces.
- Practical applications
  - Home assistance (dishwashing, wiping, table setting; Figures 4–5), warehouses (pick‑place with OOD robustness), and teleoperation with human‑in‑the‑loop corrections by sketch.
  - Education and debugging: instructors can visualize the model’s plan and correct it without rewriting code or crafting fragile prompts.

> Bottom line: Sections 2.2–2.4 and Figure 1 define a clear three‑stage spatial reasoning pipeline; Sections 3–4 detail how to label, train, and adapt it; Tables 1–2 and Figures 5–9 show that this structure yields state‑of‑the‑art results, strong robustness, and a practical, precise steering interface. The open release (models, code, and datasets) provides a reproducible blueprint for building ARMs that transform perception into purposeful action.
