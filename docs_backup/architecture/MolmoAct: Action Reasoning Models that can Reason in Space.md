# MolmoAct: Action Reasoning Models that can Reason in Space

**ArXiv:** [2508.07917](https://arxiv.org/abs/2508.07917)
**Authors:** Jason Lee, Jiafei Duan, Haoquan Fang, Yuquan Deng, Shuo Liu, Boyang Li, Bohan Fang, Jieyu Zhang, Yi Ru Wang, Sangho Lee, Winson Han, Wilbert Pumacay, Angelica Wu, Rose Hendrix, Karen Farley, Eli VanderBilt, Ali Farhadi, Dieter Fox, Ranjay Krishna
**Institutions:** 

## 🎯 Pitch

MolmoAct pioneers a three-stage Action Reasoning Model that decouples spatial reasoning from action, using depth maps and 2D trajectories to enable transparent and steerable robot control. This framework significantly enhances adaptability and robustness in both simulated and real-world settings, offering a novel dataset to boost generalist performance and setting a new standard for explainability and precision in robotic manipulation.

---

## 1. Executive Summary (2-3 sentences)
MolmoAct introduces an Action Reasoning Model (ARM) that reasons in space before acting: it first predicts a compact depth representation of the scene, then sketches a 2D end‑effector trajectory, and finally emits low‑level control actions. This three‑stage, token‑based pipeline makes robot behavior explainable and steerable, and yields strong results in simulation and the real world, while the paper also releases an open 10k‑trajectory dataset that improves generalist performance.

## 2. Context and Motivation
- Gap addressed:
  - Most robot foundation models directly map images and language to control actions, with little or no interpretable intermediate reasoning. This limits adaptation, generalization, and transparency (Introduction, p.1–2).
  - Language-only “reasoning” is often too abstract for precise manipulation: it lacks depth understanding and loses geometric detail when trajectories are described in words (Sec. 2.3, p.5–6).

- Why important:
  - Real robots must make spatially consistent, physically grounded decisions (e.g., distance to objects, collision-free reaches). Without 3D awareness and explicit motion plans, policies are brittle and hard to steer (Fig. 1; Sec. 2.3).
  - Explainability matters for safety and user trust: the ability to inspect a depth map and a planned path clarifies why a robot acts a certain way (Fig. 1; Sec. 2.4).

- Prior approaches and shortcomings:
  - Vision-Language-Action (VLA) models (e.g., RT‑1/RT‑2, OpenVLA, π0/π0‑FAST, GR00T, Magma) excel at end‑to‑end action prediction but provide limited visibility into spatial reasoning and struggle with OOD generalization (Intro; Table 1; Sec. 5.3).
  - Works that add “reasoning” typically do so in language or latent form (e.g., ECoT, CoT‑VLA, ThinkAct). These are harder to ground precisely in 3D space (Sec. 6.3).

- Positioning:
  - MolmoAct reframes “think-before-you-act” as “reason-in-space”: predict depth tokens → sketch a 2D path (“visual reasoning trace”) → output actions (Sec. 2.3; Eq. 4). Each stage is decodable and editable, enabling explanation and direct trajectory steering (Fig. 1; Sec. 2.4).
  - The paper also contributes an open, mid‑training dataset (10,689 trajectories; 93 manipulation tasks) that measurably boosts general performance (Sec. 3.2; Fig. 4; Sec. 5.4).

## 3. Technical Approach
MolmoAct builds a three‑stage autoregressive pipeline on top of a vision‑language backbone (Molmo) and trains it end‑to‑end with next‑token prediction. The pipeline:

1) Vision‑Language Backbone (Sec. 2.1; Appendix A)
- Backbone choices:
  - “D” variant: SigLIP2 vision encoder + Qwen2.5‑7B LLM (MolmoAct‑7B‑D).
  - “O” variant: OpenAI CLIP ViT‑L/14 + OLMo2‑7B (MolmoAct‑7B‑O; the most open option).
- Images are turned into vision tokens via ViT + a connector (multi‑layer features + 2×2 attention pooling), then concatenated with text tokens for the LLM (Appx. A.3–A.4).
- Multi‑image inputs (e.g., front + wrist camera) are supported by tagging and concatenation (Appx. A.5).

2) Action tokenization that respects geometry (Sec. 2.2)
- Problem: A common trick maps continuous actions to 256 bins and reuses arbitrary “rare” text tokens, which ignores the fact that neighboring bins are similar.
- MolmoAct’s solution:
  - Normalize each action dimension by dataset quantiles, discretize into 256 bins.
  - Create an action vocabulary `Vaction` by monotonically assigning adjacent byte‑level BPE symbols (taken from the end of the Qwen2 tokenizer) to adjacent action bins, so neighboring bins start with nearby embeddings.
- Why this helps: embeddings for adjacent tokens begin close in representation space, giving a smoother learning signal and faster training convergence (Sec. 2.2). The paper contrasts its ~9–10k H100 GPU‑hours pre‑training with GR00T N1.5’s ~50k GPU‑hours (Sec. 2.2 vs. Sec. 4.1; minor discrepancy: 9,216 vs. 9,728 GPU‑hours).

3) Reason‑in‑space chain (Sec. 2.3; Eqs. 1–4)
- Stage A — Depth Perception Tokens:
  - Goal: give the policy an internal 2.5D scene sense without a depth sensor at test time.
  - How: A specialist depth estimator (Depth Anything v2) produces depth maps that are VQ‑encoded into a string of M=100 indices from a codebook of N=128 entries (VQVAE; Appx. C; Sec. 3.1).
  - The model learns to predict these tokens from the RGB image and instruction:
    - Vocabulary `Vdepth = {⟨DEPTH_START⟩, ⟨DEPTH_k⟩, ⟨DEPTH_END⟩}`, k=1..128 (Eq. 1–2).
    - Each depth string d is length 100 + start/end tokens.
  - Interpretation: the depth string is a compact, decodable summary of the scene’s depth map (Fig. 1; Sec. 2.3).

- Stage B — Visual Reasoning Trace:
  - A short polyline (L=1..5 points) on the image plane, with integer pixel coordinates normalized to [0..255], denotes the planned end‑effector path from the current gripper location to future waypoints (Eq. 3; Sec. 2.3).
  - Training targets come from automatically locating the gripper in each frame (using the Molmo VLM’s 2D pointing ability) and sub‑sampling the future trajectory between the current time and episode end (Sec. 3.1 “Visual Reasoning Trace”).
  - Why this matters: it aligns each low‑level action with a concrete spatial waypoint and retains geometric detail that language would lose.

- Stage C — Action Tokens:
  - With depth tokens d and trace τ as context, the model predicts the next control action a, one action‑dimension token at a time.
  - Full factorization (Eq. 4):
    - First predict all depth tokens: ∏ p(di | I, T, d<i)
    - Then trace tokens: ∏ p(τj | I, T, d, τ<j)
    - Finally action tokens: ∏ p(ak | I, T, d, τ, a<k)

4) Steerability at inference (Sec. 2.4; Eq. 5; Fig. 1 right)
- Users can draw an on‑image trace τ (1–5 points). The system overlays the sketch on the image to form I⁺ = I ⊕ τ, and then directly predicts actions conditioned on I⁺:
  - p(a | I⁺, T) = ∏ p(ak | I⁺, T, a<k) (Eq. 5).
- This gives precise, low‑friction control—often more reliable than rephrased language (Sec. 5.6; Fig. 9).

5) Data curation and training (Sec. 3–4; Fig. 2–3)
- Converting robot data into “action reasoning” format (Sec. 3.1):
  - For each timestep (I, T, a): produce ground‑truth depth tokens (from the VQVAE’d depth map) and a visual trace (from VLM‑based 2D gripper points concatenated over time).
  - Also build three auxiliaries: depth‑only prediction, trace‑only prediction, and trajectory‑conditioned action (I, T, τ → a), which help teach each subskill (Sec. 3.1).
- MolmoAct Dataset (mid‑training; Sec. 3.2; Fig. 4):
  - 10,689 trajectories, 93 tasks, 3 cameras (two side, one wrist), avg. length 112 steps, spanning realistic home + tabletop tasks.
- Training schedule (Sec. 4; Fig. 2–3; Appx. B):
  - Pre‑training (26.3M samples) on an Open‑X‑Embodiment subset (RT‑1, BridgeData V2, BC‑Z) + auxiliaries + 2M multimodal web samples (Fig. 3, right). 256 H100s, 100k steps, batch 512 (~9.7k GPU‑hours; Sec. 4.1).
  - Mid‑training on the MolmoAct Dataset (1M action‑reasoning + 1M trajectory‑conditioned samples). 128 H100s, 50k steps (~2.3k GPU‑hours; Sec. 4.2).
  - Post‑training (task adaptation): Low‑Rank Adaptation (LoRA rank=32, α=16) and “action chunking” (predict K=8 future action steps per inference cycle) for both simulation and real robots (Sec. 4.3; Appx. B).

## 4. Key Insights and Innovations
1) Spatial chain‑of‑thought that is fully decodable (fundamental innovation)
   - Instead of latent or linguistic reasoning, MolmoAct reasons via explicit spatial tokens: depth → 2D path → action (Fig. 1; Sec. 2.3). Each piece can be visualized (depth map, overlaid trace) and edited, which is rare among VLAs.
   - Significance: explainability, test‑time steering, and better grounding for manipulation.

2) Depth perception tokens distilled from a specialist model (not just RGB; substantial innovation)
   - A VQVAE compresses dense depth into a compact 100‑token string (Eq. 1–2; Sec. 3.1). The ARM learns to predict this string from RGB alone, internalizing 3D cues without a depth sensor.
   - Impact: improved spatial understanding supports more precise low‑level control (Sec. 2.3; Fig. 1). This mirrors recent “perception tokens” ideas in MLLMs but is operationalized for control.

3) Ordinal‑aware action tokenization (useful design advance)
   - Adjacent discrete bins are mapped to adjacent byte‑BPE symbols, providing a better embedding initialization than arbitrary rare tokens (Sec. 2.2).
   - Impact: smoother optimization and lower training time; the paper contrasts its ~9–10k GPU‑hours with a 5× larger training budget reported for GR00T N1.5 (Sec. 2.2).

4) A practical, precise steering interface (new capability)
   - Users draw a short polyline; the model follows it in closed loop (Sec. 2.4; Fig. 9). In tests, trace steering outperforms open‑ended language re‑prompting by 33% on the “pick up bowl” task (Sec. 5.6; Fig. 9 left; Table 23).

5) Releasing an open mid‑training dataset and full stack (community impact)
   - The MolmoAct Dataset (10k+ trajectories) and code enable reproducibility; mid‑training on it boosts performance by ≈5.5% on average in real‑world tasks (Sec. 5.4; Fig. 6b; Table 22).

## 5. Experimental Analysis
Setup and baselines
- Benchmarks and settings:
  - SimplerEnv (Google Robot): visual matching (in‑distribution) and “variant aggregation” (OOD visual perturbations). Zero‑shot after pre‑training and fine‑tuning variants are compared (Sec. 5.1; Table 1; Appx. D.1).
  - LIBERO (Franka sim): four suites (Spatial, Object, Goal, Long). Models are post‑trained with LoRA and evaluated with action chunking (Sec. 5.2; Table 2; Appx. D.2).
  - Real‑world single‑arm and bimanual Franka: 6 tasks; fine‑tune from 50 demos/task; measure task progression (0..1) over 25 trials each (Sec. 5.2; Fig. 5; Appx. D.3).
  - OOD generalization in the real world: multi‑task setting with language/spatial/distractor/novel‑object variations (Sec. 5.3; Fig. 6a; Table 21; Appx. D.4).
  - Ablation on mid‑training dataset: three real‑world tasks (close_lid, rotate_pot, pour_tea) with and without MolmoAct Dataset mid‑training (Sec. 5.4; Fig. 6b; Table 22; Appx. D.5).
  - Human preference for instruction following and trace generation: arena‑style pairwise ratings (Sec. 5.5; Fig. 7–8; Appx. D.6).
  - Steerability study: ambiguous “pick up bowl” with language vs. trace steering (Sec. 5.6; Fig. 9; Table 23; Appx. D.7).

Main quantitative results (selected)
- SimplerEnv (Table 1):
  - Zero‑shot visual matching: 
    > “MolmoAct (zero‑shot) … 70.5%”  
    This tops several strong systems (e.g., GR00T N1.5 fine‑tuned: 52.4% visual matching avg; Magma: 68.4%).
  - After fine‑tuning (RT‑1 subset): 
    > “MolmoAct (fine‑tuned) … 71.6% visual matching, 72.1% variant aggregation.”  
    The 72.1% in variant aggregation is +7.8 points over RT‑2‑X (64.3%), the next best listed in that column.

- LIBERO (Table 2):
  - Overall average:
    > “MolmoAct‑7B‑D … 86.6% avg”  
    Slightly higher than π0‑FAST (85.5%) and above strong “reasoning” baselines (e.g., ThinkAct 84.4%, CoT‑VLA 83.9%).
  - Long‑horizon suite:
    > “MolmoAct‑7B‑D … 77.2%”  
    +6.3 points over ThinkAct (70.9%), the second best in this column; this supports the value of explicit spatial plans (Table 2).

- Real‑world fine‑tuning (Fig. 5; Tables 15–20):
  - Single‑arm tasks (put bowl in sink, wipe table, table bussing):  
    > “+10% average task progression over π0‑FAST.”  
    For instance, Wipe Table averages 1.00 vs. π0‑FAST’s 0.817 (Table 19).
  - Bimanual tasks (set table, lift tray, fold towel):  
    > “+22.7% average improvement over π0‑FAST.”  
    E.g., Fold Towel averages 0.80 vs. 0.52 (Table 15).

- OOD generalization (Fig. 6a; Table 21):
  - In a multi‑task real‑world setting with language, spatial, distractor, and novel‑object perturbations,  
    > “MolmoAct surpasses π0‑FAST by +23.3% on average task progression.”  
    Table 21 details per‑task progression under each variant; MolmoAct consistently leads across categories.

- Benefit of mid‑training dataset (Sec. 5.4; Fig. 6b; Table 22):
  - On three real‑world tasks (close_lid, rotate_pot, pour_tea), mid‑training improves MolmoAct by ~5.5% on average.  
    > Example: pour_tea trials show higher mean scores with mid‑training (Table 22).

- Human preference and steerability (Sec. 5.5–5.6; Fig. 7–9; Table 23):
  - Open‑ended instruction following (Fig. 8 left):
    > “MolmoAct‑7B‑D‑Pretrain achieves the highest Elo, winning 58% vs. SpatialVLA and 81% vs. OpenVLA.”
  - Trace generation on internet images (Fig. 7 left):
    > “MolmoAct attains top Elo, surpassing Gemini‑2.5‑Flash, GPT‑4o, and HAMSTER, with non‑overlapping 95% CIs.”
  - Steerability (Fig. 9 left; Table 23):
    > “Trace steering succeeds 75% of the time, beating open‑instruction steering by 33% and outperforming π0‑FAST by 29% in the language setting.”

Assessment of evidence
- Breadth: The paper evaluates in sim (two benchmarks), on real hardware (single and bimanual), with human preferences, OOD shifts, and an ablation on the mid‑training dataset—good coverage (Sec. 5).
- Causality: The spatial chain’s benefits are supported by (i) long‑horizon LIBERO gains (Table 2), (ii) robust OOD performance (Fig. 6a; Table 21), and (iii) steerability advantages (Fig. 9).
- Transparency: Per‑trial tables in the appendix (Tables 15–23) and explicit training configs (Tables 7, 10–14) enhance reproducibility.

## 6. Limitations and Trade-offs
Assumptions and design choices
- 2D trace for 3D control (Sec. G): The steering cue is purely 2D. It helps in-plane guidance but can be imprecise along depth (out‑of‑plane) because no explicit 3D trace is provided at inference.
- Depth token resolution (Sec. G): The depth representation is limited to 100 tokens from a 128‑entry codebook. Fine manipulation might benefit from a higher‑resolution depth tokenization.
- Gripper visibility (Sec. G): Visual trace prediction relies on end‑effector visibility in the main camera. Occlusions degrade trace quality and, consequently, control.

Computational and data aspects
- Training cost: Although substantially less than some baselines, pre‑training still requires large compute (256 H100s for ~100k steps; Sec. 4.1). There is a minor inconsistency (9,216 vs. 9,728 GPU‑hours) across sections.
- External specialists: Depth tokens rely on a pre‑trained depth model (Depth Anything v2) and a VQVAE trained on 10M depth maps; gripper points come from a VLM (Molmo). Errors in these “teachers” can propagate into labels (Sec. 3.1).
- Partial openness: The strongest variant (7B‑D) uses SigLIP2 and Qwen2.5 backbones whose pre‑training data are not fully disclosed, though an “O” variant with more open components is provided (Sec. 2.1).

Scope limitations
- Control frequency and latency: Inference produces multiple reasoning tokens per step, and server‑to‑robot latency can limit control frequency (Sec. G).
- Task diversity: While the dataset spans many household tasks, results focus on manipulation with fixed arm embodiments. Mobile navigation or complex multi‑modal sensing (e.g., tactile) is out of scope.

## 7. Implications and Future Directions
Impact on the field
- A blueprint for “spatial chain‑of‑thought” in robotics: Decodable intermediate depth and trajectory tokens make policies both more transparent and easier to steer than language‑only or latent‑only reasoning (Fig. 1; Sec. 2.3–2.4).
- Practical, user‑friendly steering: Trajectory sketches are simple for non‑experts and empirically beat re‑prompting with language in precision and success rate (Fig. 9).

What this enables next
- 3D steering and richer spatial tokens:
  - Lift user sketches into 3D by conditioning on depth tokens at inference, or predict 3D keypoints/waypoint clouds; the paper explicitly calls this out as future work to solve out‑of‑plane issues (Sec. G).
  - Increase depth token resolution or make tokens region‑adaptive (foveate around gripper/contacts).
- Multi‑view, temporally aware spatial reasoning:
  - Fuse traces across multiple cameras or time (SLAM‑assisted traces mentioned in Sec. G) to handle occlusions and long‑horizon planning.
- Cross‑embodiment and modular control:
  - Use the spatial chain as a common interface across arms and hands (the paper shows single‑ and bimanual arms; Sec. 5.2). Extend to mobile manipulation by adding base trajectories.
- Data and training innovations:
  - Explore self‑training where the model’s own predicted traces and depth tokens are refined by execution outcomes.
  - Combine the ordinal‑aware action tokenization with other discretization schemes (e.g., adaptive bin widths) or hybrid continuous decoders.

Applications
- Household robotics: table setting, dish handling, wiping/cleaning—tasks directly tested in the paper (Fig. 5; Tables 15–20).
- Interactive tele‑assist: a human sketches a correction when the robot veers off course (Fig. 9).
- Safety‑critical settings: explainable depth maps and planned paths can be logged and audited.

> Bottom line: By turning spatial understanding and planning into explicit, tokenized steps that precede action, MolmoAct makes manipulation policies more interpretable, steerable, and robust—backed by strong sim/real performance (Tables 1–2; Fig. 5–6)—and provides an open, reproducible recipe and dataset for the community.
