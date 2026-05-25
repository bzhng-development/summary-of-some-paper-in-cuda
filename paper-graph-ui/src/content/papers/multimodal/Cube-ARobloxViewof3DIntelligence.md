# Cube: A Roblox View of 3D Intelligence

**ArXiv:** [2503.15475](https://arxiv.org/abs/2503.15475)

## 🎯 Pitch

This paper introduces Cube, a pioneering step toward a 'foundation model' for 3D intelligence by developing a discrete 3D shape tokenizer that converts meshes into tokens and back. This innovation unlocks powerful applications like text-to-shape, shape-to-text, and text-to-scene generation, making it possible for AI systems and humans to collaborate in creating rich, interactive 3D experiences. By enabling seamless multimodal integration and robust 3D asset generation, Cube addresses a foundational gap in AI, paving the way toward universal assistants for 3D content creation in platforms like Roblox and beyond.

---

## 1. Executive Summary (2-3 sentences)
This paper introduces Cube, Roblox’s first step toward a “foundation model for 3D intelligence.” Its central technical contribution is a discrete 3D shape tokenizer that turns meshes into tokens and back, enabling autoregressive generation and multimodal collaboration with language models. Built on this tokenizer, the paper demonstrates text-to-shape, shape-to-text, and text-to-scene systems, and reports strong reconstruction quality and practical scene-building workflows.

## 2. Context and Motivation
- Problem addressed
  - There is no widely usable “foundation model” for 3D content comparable to models for text, images, or video. 3D assets are diverse (meshes, constructive solid geometry, textures, rigs, scripts) and datasets are relatively small, making it hard to train large models that can both reason about and generate 3D content at scale (Introduction, p.2).
- Why it matters
  - Roblox (and 3D content creation broadly) needs assistants that can:
    - Generate individual assets (e.g., “a motorbike with wings”).
    - Assemble complete scenes (e.g., “a futuristic cloud city”).
    - Produce riggable characters and scripts describing behaviors (p.2).
  - A 3D foundation model should:
    - Learn jointly from sparse, multimodal data.
    - Handle unbounded input/output sizes via autoregression with long context.
    - Collaborate with people and other AIs through multimodal I/O (bullets on p.2–3).
- Prior approaches and shortcomings
  - Continuous latent 3D representations (e.g., 3DShape2VecSet; CraftsMan) support high fidelity but are not natively compatible with token-based, multimodal, autoregressive models (p.3, §2).
  - Existing text-to-3D systems often rely on diffusion or rectified-flow methods and do not offer a general-purpose token interface for language models (p.10).
- Positioning
  - Cube focuses on a discrete shape tokenizer as the core 3D data type, analogous to subword tokenizers in NLP. This enables:
    - Autoregressive generation and long-context modeling.
    - Seamless composition with LLMs for scene analysis and planning.
    - A single “shape token” interface for both input perception and output generation (Overview, Fig. 2; Architecture, Fig. 3).

## 3. Technical Approach
The core pipeline turns a mesh into tokens and back (Fig. 3). The design choices target three issues: spatial disambiguation during attention, stable vector quantization, and a geometry-aware latent space.

Step-by-step
1. Sampling and encoding the shape
   - From each mesh, sample `Np` points on the surface to form a point cloud `P ∈ R^{Np×3}` (p.4, §2.1).
   - Embed each 3D point using a new `Phase-Modulated Positional Encoding (PMPE)`:
     - Standard sinusoidal encodings (`γ(·)`) can map distant points to similar embeddings because of periodicity, which harms cross-attention’s ability to distinguish spatially distant regions (Fig. 4a, p.4).
     - PMPE adds a phase-modulated term `γ′(p)` with a constant base frequency but channel-wise non-linear phase offsets, then uses `γ_PM(p) = γ(p) + γ′(p)` (Eq. 2, p.5). This makes dot-product similarity track spatial proximity more faithfully (Fig. 4b).

2. Encoding to continuous latents (Perceiver-style)
   - A Perceiver-based transformer uses cross-attention from learnable queries to the PMPE-embedded point cloud, followed by self-attention stacks, producing a compact sequence of continuous latent vectors (Fig. 3; p.3–5).
   - This encoder is later reused in a teacher–student setup for self-supervised regularization (§2.3).

3. Discretization via vector quantization
   - Convert continuous latents to discrete tokens with `OptVQ` (optimal-transport vector quantization; p.5, §2.2 and p.7, §2.4). Vector quantization maps each latent to a codebook index to produce a sequence of discrete shape tokens.

4. Training stabilization: stochastic linear shortcut
   - VQ layers are non-differentiable and can destabilize training. The paper introduces a `stochastic linear shortcut` that, with 50% probability, bypasses quantization by projecting continuous latents through a learnable linear layer directly into the decoder (p.5–6, §2.2; “Stochastic Gradient Shortcut (50% of training time)” in Fig. 3).
   - Intuition: the linear projection offers a smooth gradient path and acts like a “teacher” pathway; the quantized path learns to match this easier route. Prior “identity” shortcuts were reported as ineffective; the learnable linear projection performs better (p.6).

5. Latent-space regularization with self-supervision
   - The encoder is trained to produce a geometry-aware latent space using a DINOv2-style teacher–student loss (Fig. 5; p.6, §2.3).
   - Mechanism:
     - Maintain an EMA teacher encoder; pass masked queries to the student and full queries to the teacher.
     - Both produce “prototype scores” via MLP heads; minimize cross-entropy between teacher and student distributions.
   - Effect: shapes with similar geometry map to nearby latents; cosine similarity better reflects semantic/structural similarity (Fig. 6b vs 6a).

6. Decoding tokens back to geometry
   - The decoder maps discrete tokens (or continuous latents through the shortcut during training) to an implicit field, specifically an `occupancy field` f(x) ∈ {0,1} that predicts whether a 3D location is inside/outside the shape (Fig. 3).
   - `Mesh extraction`: run Marching Cubes on the occupancy field, then simplify mesh with quadric-error decimation and remove small floaters (p.8, §3.1).

Model and training specifics (baseline, March 2025 version; p.7, §2.4)
- Encoder: 13 transformer layers; Decoder: 24 layers; width 768; 12 heads; ~273M parameters.
- Latent: 512 code tokens; codebook size 16,384; embedding dim 32.
- PMPE hyperparameter `β=0.125`; self-supervised weight `λ_SSL=0.0005`.
- Data: ~1.5M assets (licensed/public + opted-in Roblox assets); each shape normalized to [-1,1]. Sample 8,192 surface points for input; 8,192 points for occupancy supervision (half uniform, half near-surface; p.7).

Applications built on the tokenizer
- `Text-to-Shape` (p.8–9, §3.1):
  - Architecture: GPT-2–like decoder-only transformer predicts shape tokens autoregressively conditioned on a CLIP-encoded prompt; uses dual-stream attention and classifier-free guidance.
  - Training data: For each of the ~1.5M assets, render multiple views; use GPT-4o to caption at various lengths.
- `Shape-to-Text` (p.9–11, §3.2):
  - Feed shape tokens into a pretrained language model backbone (`InternVL 2.5-2B`) via a small 2-layer MLP projection, following LLaVA’s two-stage training (align projection, then jointly finetune projection + LLM; tokenizer frozen).
  - Control caption length by appending an instruction token: “caption short:”, “caption medium:”, or “caption long:”.
- `Text-to-Scene` (p.11–14, §3.3):
  - Represent scenes as JSON `scene graphs` listing objects with `object_category`, `object_caption` (from shape-to-text), and layout (`position`, `extent` (bounding box), `rotation`)—Fig. 11.
  - An LLM generates scene JSON from a prompt, assisted by in-context exemplars built from existing scenes processed by the shape-to-text model.
  - Each object’s geometry is then generated with the text-to-shape model; optional iterative edits via conversation.

Updates (July 2025, §5)
- Larger VQ latent length (512→1024); two-stage supervision (occupancy pretraining then TSDF fine-tuning), Eikonal loss, REPA-style regularization, denser surface sampling (8,192→32,768), 3D bounding-box conditioning with random perturbation, and hierarchical volume decoding for faster mesh extraction (p.15–18; Fig. 13).

Key terms (brief)
- `Occupancy field`: function that outputs 1 if a 3D point is inside the shape volume, else 0.
- `TSDF (Truncated Signed Distance Function)`: signed distance to the nearest surface, clipped to a range; provides richer gradients than occupancy.
- `Eikonal loss`: regularizes SDF so that the gradient norm is ≈1 almost everywhere, helping ensure valid distance fields.
- `Vector Quantization (VQ)`: maps continuous vectors to the nearest codebook entry to obtain discrete tokens.
- `Optimal Transport VQ (OptVQ)`: assigns latents to codes with an optimal-transport objective to avoid poor local minima during VQ.

## 4. Key Insights and Innovations
1. Phase-Modulated Positional Encoding (PMPE) for 3D point inputs
   - What’s new: Adds a phase-modulated sinusoid (`γ′`) to standard encoding (`γ`) with a constant base frequency but channel-wise non-linear phases (Eq. 2).
   - Why it matters: Dot-product similarity better respects spatial distances, improving cross-attention’s ability to tell far-apart points apart (Fig. 4b). The paper reports fewer artifacts and better reconstruction of complex details (§2.1).

2. Stochastic linear shortcut for stable VQ training
   - What’s new: With 50% probability, bypass the quantization bottleneck using a learnable linear projection before decoding (Fig. 3; §2.2).
   - Why it matters: Provides stable gradients and acts as a “teacher” pathway; empirically reduces training/validation loss and widens the range of stable hyperparameters (p.6). Prior identity shortcuts performed poorly.

3. Self-supervised latent regularization that encodes geometric similarity
   - What’s new: A DINOv2-like teacher–student objective encourages latents from geometrically similar shapes to be close (Fig. 5).
   - Why it matters: The latent space becomes geometry-aware; cosine similarity aligns with actual shape similarity (Fig. 6b). This benefits retrieval, captioning, and downstream generative models (§2.3).

4. A discrete shape-token interface enabling multimodal 3D systems
   - What’s new: Use the same tokens to drive both text-conditioned generation and text captioning; demonstrate shape–text cycle consistency (Fig. 10) and text-to-scene pipelines using LLM reasoning with JSON scene graphs (Fig. 11–12).
   - Why it matters: Tokens unlock autoregression, long-context planning, and collaboration with LLMs—capabilities difficult with purely continuous latents.

5. July 2025 training recipe for higher fidelity and controllability
   - Two-stage Occupancy→TSDF fine-tuning with Eikonal loss; REPA-style latent smoothing; larger latent length; denser input sampling; 3D bounding-box conditioning with random perturbations for prompt adherence; hierarchical volume decoding for speed (Section 5; Fig. 13).
   - Significance: Improves reconstruction fidelity and controllability, and accelerates inference.

Collectively, (1)–(3) are fundamental innovations at the representation and training level; (4) is a capability-level innovation demonstrating a workable 3D token ecosystem; (5) is a set of substantial engineering refinements.

## 5. Experimental Analysis
Evaluation setup
- Datasets
  - Tokenizer training: ~1.5M assets (Objaverse + licensed + opted-in Roblox assets), normalized to [-1,1]; 8,192 input surface points; 8,192 occupancy samples (half uniform, half near-surface) (p.7).
  - Benchmarks: Toys4K for reconstruction evaluation; Toys4K was not used for training (p.7).
- Metrics
  - `S-IoU` (Surface IoU): IoU near the mesh surface.
  - `V-IoU` (Volumetric IoU): IoU over uniformly sampled points in the bounding volume (p.7).
- Baselines
  - `CraftsMan` (a 3DShape2VecSet variant; trained on 170K Objaverse objects).
  - `Ours-KL`: Cube continuous variant (no VQ, KL-regularized), trained on the same 1.5M objects.
  - `Ours-VQ`: Cube discrete tokenizer (the focus).

Main quantitative results
- Table 1 (p.7) reports:
  > S-IoU: CraftsMan 68.8%, Ours-VQ 91.7%, Ours-KL 94.8%  
  > V-IoU: CraftsMan 83.6%, Ours-VQ 94.5%, Ours-KL 95.4%
- Interpretation
  - The discrete tokenizer dramatically improves reconstruction over CraftsMan, approaching the continuous variant.
  - The small but consistent gap between Ours-KL and Ours-VQ quantifies the fidelity cost of discretization.

Qualitative results and analyses
- Reconstruction quality (Fig. 7, p.7): Fewer artifacts and better detail preservation compared with CraftsMan (e.g., limbs and bike details).
- Text-to-Shape gallery (Fig. 8, p.9): Diverse results—furniture, tools, stylized characters—show clean surfaces and crisp edges.
- Shape-to-Text captions (Fig. 9, p.10): Short/medium/long control works: short captures category, longer adds part structure and style attributes.
- Shape→Text→Shape cycle consistency (Fig. 10, p.11): Regenerated meshes preserve global structure and key features, though some fine detail is lost—evidence that captions carry sufficient 3D information for the tokenizer-driven generator.
- Text-to-Scene (Fig. 12, p.13): LLM-generated layouts produce plausible placements and consistent stylistic geometry and textures. The workflow supports iterative user edits.
- Scene analysis assistant (Table 2, p.14): Given a diner scene, the system provides useful suggestions (condiment placement, seating alternatives, background music) using only text scene graphs.

Updates (July 2025)
- Synthetic paired data (~3M) added to improve prompt adherence, especially on compositional prompts (p.15, §5.1).
- VQ-VAE improvements deliver visibly sharper reconstructions (Fig. 13, p.16–17).
- 3D bounding box conditioning improves spatial control; random perturbation prevents the model from ignoring the text prompt (p.17, §5.3).
- Hierarchical volume decoding reduces SDF query cost from O(N^3) toward O(N^2), speeding extraction (p.18, §5.4).

Ablations and robustness
- While not presented as classical ablations, the paper compares with/without the SSL loss in latent-space similarity heatmaps (Fig. 6) and discusses training stability with/without the stochastic shortcut (§2.2). The update logs function as a large ablation-style set of improvements (e.g., TSDF fine-tuning, REPA, input density).

Do the experiments support the claims?
- Yes for reconstruction quality (clear gains in Table 1 and Fig. 7).  
- The application demos substantiate capability claims (Figs. 8–12), though they are qualitative. The July updates add stronger engineering evidence for fidelity and control (Fig. 13, §5.2–5.4).

## 6. Limitations and Trade-offs
- Discretization cost
  - The discrete model (`Ours-VQ`) still trails the continuous variant (`Ours-KL`) in both S-IoU and V-IoU (Table 1), indicating some fidelity loss due to quantization.
- Layout reasoning relies on a separate LLM
  - Text-to-scene leverages an external LLM with in-context examples and may struggle with precise spatial reasoning; manual edits are sometimes required (p.12–13).
- Rotation simplification in scene graphs
  - Early JSON schema supports only Y-axis rotations, not full 3D orientation (Fig. 11 caption notes a limitation).
- Data dependence and caption quality
  - Text-to-shape and shape-to-text rely on captions produced with GPT-4o from rendered views (p.8); errors or biases in captions can influence both training and controllability.
  - July 2025 synthetic data (~3M assets) improve prompt adherence but may import biases from upstream text-to-image and image-to-shape models (§5.1).
- Computational footprint
  - The base tokenizer is ~273M parameters with non-trivial training requirements (p.7). Mesh extraction via implicit fields requires dense evaluations; although hierarchical decoding reduces cost, high-resolution extraction remains expensive (§5.4).
- Geometry only (current step)
  - The paper focuses on geometry; textures are provided by a separate in-house model (Fig. 12 caption). Rigging, animation, and scripts (“4D behavior”) remain future work (Conclusion, p.14–15).

## 7. Implications and Future Directions
How this work changes the landscape
- Establishes `shape tokens` as a practical 3D primitive for foundation models:
  - Enables autoregression over 3D content, long-context planning, and multimodal collaboration with LLMs.
  - Demonstrates a full pipeline: tokenize meshes → train token models → build text↔shape↔scene systems.

What it enables next
- Unified multimodal 3D foundation models
  - With a token vocabulary for shapes, future models can mix sequences of text, shapes, images, scripts, and physics events in a single autoregressive stream—similar to “mixed-modal” models (p.3; analogy to Chameleon).
- Richer 3D editing agents
  - LLM-driven scene planning combined with token-level constraints (e.g., bounding box conditioning) enables interactive assistants for asset placement, replacement, and global style adjustments (p.12–14; §5.3).
- Learning across modalities with sparse data
  - The SSL-regularized, geometry-aware latent space (Fig. 6) can support retrieval, clustering, and few-shot learning over 3D assets.

Practical applications
- Roblox Creator tools
  - Text-to-asset generation, automated scene kitbashing, content-aware suggestions (Table 2), and controllable spatial design via bounding boxes (§5.3).
- Game and XR pipelines
  - Rapid prototyping of stylized assets (Fig. 8) and scenes (Fig. 12), with token-level control to match art direction.

Future research directions highlighted by the paper
- Mixed meshes and CSG generation
  - Integrate constructive solid geometry (CSG) tokens for efficient, “blocky” Roblox-native styles and edge-device rendering (Conclusion, p.15).
- Avatar generation with rigging
  - Produce riggable heads, bodies, and layered clothing tailored to animation constraints (p.15).
- 4D behavior generation
  - Extend from static geometry to rigging, scripted interactions, and physics-aware behaviors—“4D behavior” (p.15).
- Bridging the VQ fidelity gap
  - Close the remaining gap between discrete and continuous latents via improved codebooks, alignment losses (e.g., REPA), and TSDF-centric training (§5.2).
- Stronger spatial reasoning in scene layout
  - Combine geometric constraints directly into the LLM planning loop (e.g., collision-free placement, visibility and affordance constraints), and expand rotation handling beyond Y-axis (Fig. 11).

Quotable highlights
- Reconstruction gains (Table 1):
  > “S-IoU↑ 68.8% (CraftsMan) → 91.7% (Ours-VQ) → 94.8% (Ours-KL); V-IoU↑ 83.6% → 94.5% → 95.4%.”
- Training stability (Fig. 3; §2.2):
  > “Stochastic Gradient Shortcut (50% of training time)” projects continuous latents through a linear layer to stabilize gradients and guide the quantized path.
- Latent geometry awareness (Fig. 6):
  > With the self-supervised loss, “latent space similarity aligns with ground-truth geometric relationships.”

Bottom line
Cube reframes 3D as a token modality and shows that, with the right positional encoding, VQ training stabilization, and self-supervised regularization, discrete shape tokens can reach high reconstruction quality while unlocking autoregressive, multimodal workflows. The demonstrated text-to-shape, shape-to-text, and text-to-scene systems, along with the July 2025 upgrades for fidelity and control, lay a credible path toward a unified foundation model for 3D creation and reasoning.
