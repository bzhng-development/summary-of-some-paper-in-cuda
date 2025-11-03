# Cube: A Roblox View of 3D Intelligence

**ArXiv:** [2503.15475](https://arxiv.org/abs/2503.15475)
**Authors:** Foundation AI Team, Kiran Bhat, Nishchaie Khanna, Karun Channa, Tinghui Zhou, Yiheng Zhu, Xiaoxia Sun, Charles Shang, Anirudh Sudarshan, Maurice Chu, Daiqing Li, Kangle Deng, Jean‑Philippe Fauconnier, Tijmen Verhulsdonck, Maneesh Agrawala, Kayvon Fatahalian, Alexander Weiss, Christian Reiser, Ravi Kiran Chirravuri, Ravali Kandur, Alejandro Pelaez, Akash Garg, Michael Palleschi, Jessica Wang, Skylar Litz, Leon Liu, Anying Li, David Harmon, Derek Liu, Liangjun Feng, Denis Goupil, Lukas Kuczynski, Jihyun Yoon, Naveen Marri, Peiye Zhuang, Yinan Zhang, Brian Yin, Haomiao Jiang, Marcel van Workum, Thomas Lane, Bryce Erickson, Salil Pathare, Kyle Price, Anupam Singh, David Baszucki
**Institutions:** Roblox Foundation AI Team

## 🎯 Pitch

Cube pioneers a transformative approach to 3D intelligence by converting 3D shapes into discrete tokens for GPT-style models, unlocking applications such as text-to-shape, shape-to-text, and text-to-scene generation. This innovation significantly streamlines 3D content creation, impacting industries like AR/VR and robotics by bridging traditional 3D modeling with advanced language models, enabling more efficient and interactive multi-modal workflows.

---

## 1. Executive Summary (2-3 sentences)
Cube introduces a practical path toward a “foundation model for 3D intelligence” by turning 3D shapes into discrete tokens that can be used by GPT‑style models. The paper’s core contributions are a robust 3D shape tokenizer and three end‑to‑end applications—text‑to‑shape, shape‑to‑text, and text‑to‑scene—that show how these tokens enable multi‑turn 3D creation and reasoning with large language models.

## 2. Context and Motivation
- Problem addressed
  - Building a general AI assistant that can generate and reason about 3D experiences (objects, scenes, characters, behaviors) requires a common representation that works across modalities and scales. The paper focuses on the foundation of such a system: a way to convert 3D geometry into discrete tokens so that standard autoregressive models can “read and write” 3D shapes (Section 1, Figures 1–2).
- Why it matters
  - 3D creation is central to platforms like Roblox; automatic generation of assets and scenes can dramatically reduce content creation time. Beyond entertainment, 3D understanding and generation is key for simulation, robotics, AR/VR, and digital twins.
- Gaps in prior work
  - Existing strong 3D generative models (e.g., continuous latent approaches such as 3DShape2VecSet, CraftsMan, rectified‑flow models like Trellis/Hunyuan3D‑2/TripoSG) typically use continuous latent vectors. These are not naturally compatible with token‑based, mixed‑modal autoregressive models, which excel at long‑context reasoning and multi‑turn workflows (Section 1).
- Positioning relative to existing work
  - Cube adapts continuous 3D representations into a discrete token space using a VQ‑VAE‑style approach, then shows that these tokens plug into standard GPT‑style decoders and LLM workflows. The system emphasizes three design principles for a 3D foundation model: joint learning across sparse multi‑modal data, handling variable‑size 3D content with long contexts, and collaborating with humans/LLMs via multi‑modal I/O (bullets in Section 1).

## 3. Technical Approach
The paper builds a shape tokenizer and then stacks applications on top of it.

A. Shape Tokenizer (Figure 3; Section 2)
1) Input preprocessing
- Start from a triangle mesh. Sample Np surface points to obtain a point cloud P ∈ R^(Np×3). Points are embedded with a custom Phase‑Modulated Positional Encoding (PMPE) before being fed to a transformer (Section 2.1).

2) Phase‑Modulated Positional Encoding (PMPE) to fix ambiguity in dot‑product attention
- Challenge: Traditional sinusoidal positional encoding `γ(p)` uses exponentially increasing frequencies (Eq. 1). Because sinusoids repeat, spatially distant points can map to similar embeddings along some channels, causing cross‑attention to confuse far‑apart points (Figure 4a; explained in Section 2.1).
- Idea: Add a second encoding `γ′(p)` that uses a constant base frequency (π/2) but varies the phase offset nonlinearly per channel, controlled by hyperparameter `β`. The final embedding is `γPM(p) = γ(p) + γ′(p)` (Eq. 2). Varying phases breaks periodic collisions so inner‑product similarities better reflect true spatial distance (Figure 4b).
- Result: Empirically improves reconstruction fidelity and reduces artifacts like disconnected components (end of Section 2.1).

3) Perceiver‑style encoder to produce continuous latents
- Architecture: A Perceiver transformer with learnable queries cross‑attends to the PMPE‑embedded point cloud and produces a set of continuous latent vectors (12 self‑attention blocks are shown in Figure 3; detailed layer sizes in Section 2.4). The Perceiver formulation is suited to very long inputs (many points) since only a fixed number of learnable queries are refined through cross‑attention.

4) Discretization via Vector Quantization with Optimal Transport (OptVQ)
- Continuous latents are quantized to a sequence of discrete code indices using an Optimal Transport VQ variant (Zhang et al., 2024), which mitigates local assignment pitfalls (Section 2.4). The codebook has 16,384 entries; each token is a 32‑d embedding; the sequence length is 512 tokens in the initial model (Section 2.4).
- Non‑differentiability problem: VQ layers can destabilize training because code assignment is discrete. The system adds two mechanisms to stabilize and structure the latent space (Sections 2.2–2.3).

5) Stochastic linear shortcut for gradient stabilization (Section 2.2)
- Mechanism: With 50% probability during training, bypass the quantization bottleneck entirely by projecting the continuous latents through a trainable linear layer and feeding them directly to the decoder (Figure 3).
- Why this helps: The linear path has well‑defined gradients and acts like a “teacher” for the quantized path. Unlike an identity shortcut (found ineffective in prior analyses), the linear layer learns a slightly different mapping, preventing the quantized path from getting stuck in poor local minima.
- Observed effect: Lower training/validation losses and more robust convergence across hyperparameters (Section 2.2).

6) Self‑supervised latent space regularization (Section 2.3; Figure 5)
- Goal: Encourage geometrically similar shapes to have similar latents.
- Mechanism: A DINOv2‑style student/teacher setup where the teacher is an EMA copy of the encoder. The student sees randomly masked queries; both produce “prototype scores” via MLP heads. A cross‑entropy loss aligns student and teacher distributions; this is added to the reconstruction loss with weight `λSSL = 0.0005` (Section 2.4).
- Effect: Latent cosine similarities better track true shape similarity (Figure 6 shows that, with the loss, a car is closer to another car than to an ice‑cream model).

7) Decoder and reconstruction target
- The decoder (24 transformer layers) maps discrete tokens to an implicit occupancy field from which a mesh is extracted (Figure 3). During training, a “Stochastic Gradient Shortcut” allows the decoder to directly use continuous latents (the same linear shortcut) half the time to stabilize learning (Section 2.2). Reconstruction supervision uses sampled inside/outside occupancy points (Section 2.4).
- Marching Cubes plus post‑processing (component removal, decimation) yields the final mesh (Section 3.1, Mesh extraction).

8) Architecture and training scale
- Model: ~273M parameters; 768‑wide, 12‑head transformers; 13 encoder and 24 decoder layers; 512 latent tokens; 16,384‑size codebook (Section 2.4).
- Data: ~1.5M 3D assets from licensed/public sources like Objaverse and opted‑in Roblox assets (Section 2.4). Each batch: 8,192 surface points for encoder input; two sets of 8,192 points for occupancy supervision—one uniform in volume, one near the surface (Section 2.4).

B. Generative and Reasoning Applications

1) Text‑to‑Shape (Section 3.1; Figure 8)
- Architecture: A decoder‑only GPT similar to GPT‑2 autoregressively generates shape tokens, conditioned on text features from a CLIP text encoder. Conditioning is injected via “dual‑stream attention” (Esser et al., 2024). Classifier‑Free Guidance is used by randomly dropping text 10% of the time during training (Section 3.1).
- Training pairs: For each training mesh, multiple rendered views are captioned by GPT‑4o to create diverse text prompts (Section 3.1).
- Mesh extraction: Marching Cubes on the predicted occupancy with in‑house decimation and floater removal (Section 3.1).

2) Shape‑to‑Text (Section 3.2; Figures 9–10)
- Architecture: Feed shape tokens into a pre‑trained decoder‑only LLM backbone (InternVL 2.5‑2B) through a two‑layer MLP projection. Two‑stage training: first train only the projection for alignment; then jointly fine‑tune the projection and LLM (shape tokenizer is frozen) using next‑token prediction on text outputs (Section 3.2).
- Controlling caption length: Append an instruction token—`caption short:`, `caption medium:`, or `caption long:`—to guide output length (Implementation details in Section 3.2).
- Cycle consistency demo: Caption a shape, then use the caption to regenerate the shape with the text‑to‑shape model (Figure 10). Overall geometry and key characteristics are preserved, though fine details can be lost.

3) Text‑to‑Scene with LLM collaboration (Section 3.3; Figures 11–12; Table 2)
- Scene graph format: JSON containing a flat list of objects. Each object has `object_category`, a natural‑language `object_caption`, and layout fields—`position`, `extent`, `rotation`. The example supports Y‑axis rotation only (Figures 11 and its caption; “should support rotation in all axis” is noted as a TODO).
- Pipeline:
  - Start from a user prompt (e.g., “Make a campsite”). An LLM generates an initial scene JSON. To help spatial reasoning, the system supplies in‑context exemplars consisting of prompt/JSON pairs built by processing real scenes through the shape‑to‑text model (Section 3.3).
  - For each object, text‑to‑shape generates geometry; a text‑to‑texture model (FlashTex‑based) produces textures; the scene is rendered (Figure 12).
  - Users can iteratively refine the scene (“Add another tent”) or manually adjust placements; the LLM updates the JSON accordingly (Section 3.3).
  - The same scene JSON supports analysis and suggestions: the LLM can summarize, propose object placements (e.g., condiments on a counter), recommend seating alternatives or background music (Table 2).

C. July 2025 Updates (Section 5)
- Data: +3M synthetic (text, shape) pairs created via a text‑to‑image and image‑to‑shape pipeline seeded by LLM‑expanded prompts. Result: better prompt adherence, especially for compositional prompts (Section 5.1).
- Improved VQ‑VAE tokenization (Section 5.2; Figure 13)
  - Latent length increased from 512 to 1024.
  - Two‑stage supervision: occupancy pre‑training, then TSDF fine‑tuning to provide richer surface gradients (Section 5.2.1).
  - Regularizers and inputs: Eikonal loss for valid SDFs; REPA‑style representation alignment between quantized latents and late‑decoder features to smooth quantized latents; increase surface point inputs from 8,192 to 32,768 to capture more detail (Section 5.2.2).
- 3D bounding‑box conditioning for text‑to‑shape (Section 5.3)
  - Append an MLP‑encoded vector of normalized box dimensions as an extra conditioning token. To prevent over‑reliance on the box (which drowned out text), randomly perturb box dimensions during training so the model balances both conditions.
- Accelerated shape extraction (Section 5.4)
  - Hierarchical SDF sampling: identify potentially occupied voxels on a coarse grid, then only refine those regions to the target resolution. This reduces complexity from O(N^3) toward O(N^2) by focusing computation near the surface.

## 4. Key Insights and Innovations
- Phase‑Modulated Positional Encoding (PMPE) for point clouds
  - What’s new: Adds a constant‑frequency, phase‑modulated sinusoidal component `γ′(p)` to the traditional encoding `γ(p)`, breaking periodic collisions in dot‑product attention (Eq. 2; Figure 4).
  - Why it matters: Cross‑attention can now better distinguish spatially distant points, improving reconstruction fidelity and reducing artifacts—crucial when many surface points are encoded (Section 2.1).
  - Innovation type: Fundamental improvement to geometric encoding for transformer cross‑attention.

- Stochastic linear shortcut across the quantization bottleneck
  - What’s new: During training, replace the discrete code path with a trainable linear projection of continuous latents 50% of the time (Section 2.2).
  - Why it matters: Stabilizes gradients, prevents local minima, and implicitly teaches the quantized path a better target distribution. Outperforms identity shortcuts documented in prior analyses (Section 2.2).
  - Innovation type: Training stabilization trick specific to VQ‑based tokenization.

- Self‑supervised latent regularization aligned with geometric similarity
  - What’s new: A DINOv2‑style student/teacher cross‑entropy on prototype scores with masked queries (Figure 5; Section 2.3).
  - Why it matters: Produces latents where cosine similarity correlates with geometric relatedness (Figure 6), enabling downstream retrieval, clustering, and more reliable conditioning for text generation.
  - Innovation type: Representation learning adapted to 3D latents.

- A complete 3D tokenization‑to‑LLM pipeline with measurable reconstruction wins
  - What’s new: Discrete shape tokens (512→1024 length) that integrate with GPT‑style decoders for text‑to‑shape, with shape‑to‑text enabling LLM‑based scene reasoning. Reconstruction accuracy (S‑IoU/V‑IoU) surpasses a strong continuous‑latent baseline trained on a smaller dataset (Table 1; Figure 7). Updates further improve reconstruction (Figure 13).
  - Why it matters: Establishes a practical 3D “language” that GPT‑style models can generate and reason over; this opens mixed‑modal authoring loops with LLMs (Figures 11–12; Table 2).
  - Innovation type: System‑level capability that bridges 3D geometry and autoregressive modeling.

- Practical engineering for control and speed
  - 3D bounding‑box conditioning with randomized perturbations keeps text conditioning effective (Section 5.3).
  - Hierarchical SDF sampling reduces extraction cost towards O(N^2) (Section 5.4).
  - Synthetic data generation improves prompt adherence (Section 5.1).
  - Innovation type: Incremental but impactful improvements to usability and scalability.

## 5. Experimental Analysis
- Evaluation setup
  - Tokenizer training: ~1.5M 3D assets normalized to [-1,1]^3; 8,192 input points; occupancy supervision with near‑surface and uniform volume samples (Section 2.4).
  - Comparisons: Two Cube variants—`Ours‑VQ` (discrete tokens via VQ‑VAE) and `Ours‑KL` (continuous latent with KL regularization, same architecture minus VQ)—are compared to `CraftsMan` (Li et al., 2024) trained on 170K Objaverse assets (Section 2.4).
  - Metrics: Surface‑IoU (S‑IoU) computed on near‑surface points; Volumetric IoU (V‑IoU) on uniformly sampled volume points. Evaluation is on Toys4K, which none of the models saw during training (Section 2.4).

- Main quantitative results
  - Table 1 reports:
    - CraftsMan: S‑IoU 68.8%, V‑IoU 83.6%
    - Ours‑VQ: S‑IoU 91.7%, V‑IoU 94.5%
    - Ours‑KL: S‑IoU 94.8%, V‑IoU 95.4%
  - Quote:
    > “Both our VQ‑VAE (Ours‑VQ) and the continuous variant (Ours‑KL) outperform CraftsMan... Our continuous variant still outperforms its discrete counterpart, highlighting that there remains some loss of geometry fidelity through the vector quantization process.” (Section 2.4; Table 1; Figure 7)

- Qualitative reconstructions and generation
  - Reconstruction: Figure 7 shows that both Cube variants preserve finer geometric details and reduce artifacts relative to CraftsMan.
  - Text‑to‑Shape: Figure 8 demonstrates diverse objects with sharp edges and smooth surfaces; the paper notes results are “approaching” the visual quality of continuous‑latent rectified‑flow models like Trellis/Hunyuan3D‑2/TripoSG (Section 3.1).
  - Shape‑to‑Text: Figure 9 shows controllable caption lengths; Figure 10 demonstrates cycle consistency—category and overall geometry persist when regenerating from the caption with some fine‑detail loss (Section 3.2).
  - Text‑to‑Scene: Figure 12 shows three generated scenes with plausible layout and stylistic consistency; Table 2 illustrates LLM‑based scene summarization, placement suggestions, and audio recommendations.

- Ablations and robustness
  - PMPE vs. traditional encoding: Figure 4 visualizes how PMPE improves dot‑product similarity structure; qualitative improvements are claimed in reconstruction fidelity and reduced artifacts (Section 2.1).
  - Self‑supervised loss: Figure 6 directly compares latent cosine similarity matrices with/without the SSL term; similarity aligns better with shape categories when the loss is used (Section 2.3).
  - Training stabilization: Section 2.2 reports improved stability and losses with the stochastic linear shortcut, noting identity shortcuts perform poorly.
  - July 2025 updates: Figure 13 compares the March’25 and Jul’25 VQ‑VAE reconstructions, showing visible quality gains from TSDF fine‑tuning, longer token sequences, and regularization (Section 5.2).

- How convincing are the experiments?
  - The reconstruction metrics (Table 1) are strong and clearly support the tokenizer’s quality. However, comparisons for text‑to‑shape generation are qualitative; the paper explicitly refrains from asserting superiority over rectified‑flow methods, stating only that results are “approaching” those approaches (Section 3.1).
  - For text‑to‑scene, there are no quantitative layout or affordance metrics; evidence is visual (Figure 12) and conversational (Table 2). The system’s effectiveness for precise spatial reasoning is partially mitigated by in‑context examples and interactive correction.

## 6. Limitations and Trade-offs
- Discretization fidelity gap
  - `Ours‑KL` (continuous) outperforms `Ours‑VQ` (discrete) on S‑IoU/V‑IoU (Table 1), indicating residual fidelity loss due to quantization. The July’25 improvements reduce but likely do not eliminate this gap.
- Scene layout precision and rotation constraints
  - The JSON scene graph supports only Y‑axis rotation in the example (Figure 11 caption), and spatial reasoning is largely delegated to the LLM aided by exemplars. Precise 3D relationships (e.g., avoidance of interpenetration, physical stability) are not guaranteed and may require user correction (Section 3.3).
- Texture and materials outside the shape tokens
  - Geometry is tokenized and generated; textures rely on a separate text‑to‑texture system (FlashTex‑based) and are not integrated into the token stream (Figure 12 caption). A truly unified 3D token would ideally capture both shape and appearance.
- Computational costs
  - The tokenizer uses a 273M‑parameter Perceiver encoder/decoder and processes tens of thousands of points per object (32,768 after the update; Section 5.2.2). Although hierarchical SDF sampling reduces extraction work toward O(N^2), high‑resolution extraction remains computationally expensive (Section 5.4).
- Data dependencies and domain bias
  - Training draws from ~1.5M assets plus ~3M synthetic (Section 5.1). Asset quality and domain distribution (e.g., Roblox‑style models) may bias generation and captioning.
- Not a unified 3D foundation model yet
  - The system demonstrates a crucial component (shape tokens) and applications, but it is still a collection of specialized modules (text encoder, GPT‑style decoders, LLM, texturing), not a single joint foundation model (Conclusion, Section 4).
- Limited evaluation coverage
  - No user studies for text‑to‑shape adherence or stylistic control; no standardized metrics for text‑to‑scene layout quality or inter‑object relations; few failure‑case analyses beyond noting fine‑detail loss in cycle consistency (Figure 10).

## 7. Implications and Future Directions
- How this changes the landscape
  - By turning 3D geometry into discrete tokens compatible with GPT‑style modeling, Cube bridges 3D generation with the broader mixed‑modal autoregressive ecosystem. This enables long‑context, multi‑turn 3D creation that can incorporate text, retrieved examples, and reasoning over structured scene graphs (Figures 11–12; Table 2).
- Immediate applications
  - Rapid prototyping of game assets and scenes; educational or creative tools where users draft scenes conversationally; asset cataloging via shape‑to‑text for search and retrieval; agentic assistants that analyze scenes and recommend improvements.
- Research directions enabled or suggested
  - Unifying modalities: Extend tokens to include textures, materials, and physics/rigging to move toward the paper’s vision of “4D behavior generation” (animations and scripted interactions) outlined in the Conclusion (Section 4).
  - Tighter integration with parts‑based modeling (CSG) for efficiency and “Roblox style,” including hybrid mesh‑CSG generation (Conclusion, “Mixed generation of meshes and parts”).
  - Reducing the VQ fidelity gap: better codebooks, hierarchical/multi‑scale tokenizers, learned decoders directly producing watertight SDFs with fewer extraction artifacts; the TSDF fine‑tuning and Eikonal loss (Section 5.2) are promising steps.
  - Stronger spatial reasoning: incorporate geometric constraints, differentiable collision, or learned layout priors; expand scene graph expressivity (full rotations, hierarchies, relations).
  - Data and evaluation: establish standardized compositional adherence metrics for text‑to‑shape; quantitative layout metrics for text‑to‑scene; curated benchmarks for 3D reasoning and prompt adherence.
  - Efficiency and interactivity: further accelerate extraction (e.g., adaptive meshing), streaming token decoders for progressive previews, and editable token spaces for localized shape edits.

Overall, Cube demonstrates that robust 3D shape tokenization—stabilized by PMPE, stochastic linear shortcuts, and self‑supervised latent regularization—can anchor a practical pipeline for multi‑modal 3D generation and reasoning. The reconstruction metrics (Table 1) and the qualitative applications (Figures 8–12) show the approach is viable today, while the update log (Section 5) maps out a credible path toward higher fidelity, better control, and faster inference.
