# MolmoAct: Action Reasoning Models that can Reason in Space

**ArXiv:** [2508.07917](https://arxiv.org/abs/2508.07917)

## 🎯 Pitch

MolmoAct introduces a new paradigm for robotic foundation models by explicitly factoring perception, planning, and control into a three-stage action reasoning pipeline: first predicting depth-aware perception tokens, then sketching a visual trajectory trace, and finally outputting low-level actions. This structure enables robots to perform explainable and steerable behaviors, dramatically improving generalization, interpretability, and real-world robustness over prior end-to-end or language-only reasoning approaches. As an open, state-of-the-art Action Reasoning Model, MolmoAct sets a new foundation for building trustworthy, adaptive, and human-guidable robotic agents.

---

## 1. Executive Summary
MolmoAct introduces an “Action Reasoning Model” (ARM) that reasons about robot actions through an explicit, three-stage chain: predict 3D-aware depth tokens, sketch a 2D trajectory trace in the image, and then output low‑level action commands. This structure yields explainable, steerable behavior and strong performance: zero‑shot 70.5% on SimplerEnv Google Robot visual matching (Table 1), 86.6% average on LIBERO after post‑training (Table 2), and large gains in real‑world task progression and out‑of‑distribution generalization (Figures 5–6).

## 2. Context and Motivation
- Problem addressed
  - Most robot policies map images and language directly to control. These “end-to-end” policies often lack adaptability, generalization, and interpretability. The missing piece is explicit, spatial reasoning between perception and actuation (Introduction; Figure 1).
- Why this matters
  - Real robots operate in 3D, cluttered, dynamic spaces. Without understanding depth and planned motion, policies are brittle (e.g., to lighting, object layout) and hard to steer or debug. Providing interpretable intermediate representations improves trust and enables human guidance.
- Prior approaches and their limits
  - Vision-Language-Action (VLA) models (e.g., RT‑1, RT‑2, OpenVLA, GR00T N1.5, π0/π0‑FAST) improve generalization via large datasets and web pretraining, but remain opaque and sometimes brittle, with limited insight into why a given action was chosen (Introduction; §6.1).
  - Language-based “chain-of-thought” (CoT) for robotics decomposes tasks verbally (e.g., ECoT, CoT‑VLA, ThinkAct; §6.3), but textual steps do not capture precise 3D geometry or sub-centimeter motion constraints required for manipulation.
- Positioning of this work
  - MolmoAct moves from language-only reasoning to “reasoning in space”: it adds two spatial, non-linguistic intermediate representations—depth perception tokens and a drawable 2D trajectory trace—before predicting actions (Figure 1; §2.3–2.4). The model, code, and data are released openly.

## 3. Technical Approach
MolmoAct converts a standard vision-language backbone into an Action Reasoning Model (ARM) with three autoregressive stages. Below is the pipeline and the rationale behind each design.

- Backbone and variants (§2.1; Appendix A)
  - Start from Molmo, a vision-language model with a ViT image encoder, connector, and an LLM. Two 7B variants:
    - `MolmoAct-7B-D`: SigLIP2 ViT + Qwen2.5‑7B.
    - `MolmoAct-7B-O`: OpenAI CLIP ViT + OLMo2‑7B (most open).
  - Multi-image inputs (e.g., wrist + third-person views) are supported by concatenating image token streams with index markers (Appendix A.5).

- Stage 1: Depth Perception Tokens (§2.3 “Depth Perception Tokens”; Eq. 1–2)
  - Goal: internalize 3D geometry despite only having RGB input during inference.
  - Mechanism:
    - Train a specialist depth estimator (DepthAnything v2) and compress depth maps using a VQ‑VAE (“vector-quantized variational autoencoder”; trained on 10M depth maps for 20 epochs; §3.1).
    - The VQ‑VAE codebook has N=128 discrete codes. Each image is represented as a fixed-length string of M=100 depth tokens: `d = (<DEPTH_START>, DEPTH_z1, …, DEPTH_zM, <DEPTH_END>)` (Eq. 2).
    - During MolmoAct training, the model learns to predict this depth token string from RGB (“specialist-to-generalist distillation”), teaching it a depth-aware representation it can later condition on.

- Stage 2: Visual Reasoning Trace (§2.3 “Visual Reasoning Trace”; Eq. 3)
  - Goal: encode an explicit plan as a short, human-editable 2D path for the end-effector.
  - Representation:
    - A polyline `τ = (p1,…,pL)`, 1≤L≤5, points in image coordinates normalized to 0–255 (Eq. 3). `p1` is the current gripper location; later points come from evenly subsampling the future of the demo.
  - How labels are generated (§3.1):
    - Use a VLM (“Molmo”) trained on 2D pointing to localize the robot gripper in each frame via the prompt “point to the robot gripper” (or left/right for bimanual). This yields pixel coordinates for every timestep, which are linked into traces; for each step, select up to 5 points from now to the episode end.

- Stage 3: Action Tokens and factorization (§2.2; §2.3; Eq. 4)
  - Action space discretization (§2.2):
    - For each action dimension (e.g., 6‑DoF end-effector deltas and gripper), normalize by dataset quantiles and discretize into 256 bins.
    - Instead of mapping bins to arbitrary “rare” text tokens (which ignore ordinal structure), map them monotonically to 256 byte‑level BPE symbols taken from the end of the Qwen2 tokenizer so adjacent bins have adjacent symbols. This similarity‑preserving initialization makes learning smoother and faster.
  - Autoregressive factorization (§2.3; Eq. 4):
    - Generate depth tokens `d` first, then trace tokens `τ`, then action tokens `a`:
      - `p(d, τ, a | I, T) = ∏ p(di | I, T, d<i) × ∏ p(τj | I, T, d, τ<j) × ∏ p(ak | I, T, d, τ, a<k)`
    - This conditions actions on explicit depth and planned trajectory, grounding control in spatial reasoning.

- Test-time steerability via user sketches (§2.4; Eq. 5; Figure 1, right; Figure 9)
  - Users can draw a trace on the camera image, `I+ = I ⊕ τ`.
  - The model conditions on this overlaid image to produce actions:
    - `p(a | I+, T) = ∏ p(ak | I+, T, a<k)` (Eq. 5).
  - The trace is unambiguous, precise, and editable, often more reliable than language-only corrections.

- Training data and stages (§3–§4; Figure 2; Figure 3)
  - Data curation for “action reasoning data” (§3.1):
    - For any robot dataset with RGB, language, and actions, generate depth token strings and visual traces per frame using the specialist pipeline above.
  - Auxiliary robot data (§3.1):
    - Depth-only prediction; trace-only prediction; and trajectory‑conditioned actions (overlay trace on image and predict the next action)—the last is crucial for steerability.
  - Pre‑training (§4.1):
    - Mixture of OXE subset (RT‑1, BridgeData V2, BC‑Z) converted to action reasoning, plus auxiliary robot data and ~2M multimodal web samples (e.g., VQA, PixMo, LVIS pointing; §3.3). Total 26.3M samples; sample rates in Figure 3.
    - 100k steps on 256 H100s (9,728 GPU hours). Despite being much smaller than some competitors, achieves strong zero‑shot generalization (Table 1).
  - Mid‑training (§4.2; §3.2):
    - MolmoAct Dataset: 10,689 real‑world trajectories on 93 tasks; two cameras + wrist view; average 112 steps; mix of home and tabletop environments (Figure 4; Appendix E).
    - Converted into 1M action reasoning and 1M trajectory‑conditioned samples; trained 50k steps on 128 H100s.
  - Post‑training / adaptation (§4.3):
    - For new tasks or embodiments, collect 30–50 demos and LoRA‑tune only low‑rank adapters (rank 32, alpha 16). Use action chunking (N=8) to predict short open‑loop segments, then re‑plan.

- Implementation details supporting robustness (Appendix A–B)
  - High‑res tiling + attention pooling to keep fine details (A.2–A.3).
  - Multi‑image tokenization (A.5).
  - Stable distributed training, token reallocation for depth tokens (B.1).
  - Compute and cluster details (B.2).

## 4. Key Insights and Innovations
- Spatial chain-of-thought for control (§2.3; Figure 1)
  - Novelty: replaces text‑only CoT with two spatial, tokenized representations—depth and 2D trace—before predicting actions.
  - Why it matters: explicitly grounds actions in 3D perception and a visible plan. Each stage is decodable (depth map, overlayed trace, executed action), improving explainability and enabling human correction.

- Similarity-preserving action tokenization (§2.2)
  - Novelty: maps 256 discretization bins to adjacent byte‑level BPE symbols so neighboring bins start with similar embeddings.
  - Impact: better inductive bias for ordinal structure; faster training. The pre‑training budget is 9,728 GPU hours, over 5× less than a reported 50,000 GPU hours for GR00T N1.5 (§2.2).

- Steerability via visual traces (§2.4; Figure 9)
  - Novelty: a general, precise, test‑time control interface—draw the desired end‑effector path in the image, and the model follows it.
  - Impact: resolves ambiguity of language corrections, enables interactive refinement during execution, and yields higher success than language-only steering (Figure 9, left).

- Open mid‑training dataset and full release (§3.2; Abstract; Conclusion)
  - The MolmoAct Dataset: >10k high-quality robot trajectories across 93 tasks; used to bridge pre‑training to real‑world settings.
  - Impact: mid‑training on this dataset adds ~5.5% average real‑world performance (Figure 6b) and provides a blueprint for building ARMs.

## 5. Experimental Analysis
- Evaluation setup and baselines
  - Simulation zero‑shot and fine‑tuning on SimplerEnv Google Robot suite (visual matching and variant aggregation) with many competitive baselines (Table 1; §5.1, §5.3).
  - LIBERO post‑training on four suites (Spatial, Object, Goal, Long‑horizon), comparing against state‑of‑the‑art VLA policies (Table 2; §5.2).
  - Real‑world fine‑tuning on single‑arm and bimanual Franka tasks (Figure 5; Appendix D.3).
  - Out‑of‑distribution (OOD) generalization tests in SimplerEnv variant aggregation and real‑world multi‑task settings: language variation, spatial variation, distractors, novel objects (Figure 6a; §5.3; Appendix D.4).
  - Human evaluations for open‑ended instruction following and for line‑trace generation (Figures 7–8; §5.5).
  - Steerability study: ambiguous instruction corrected by language or a user‑drawn trace (Figure 9; §5.6; Appendix D.7).

- Main quantitative results
  - SimplerEnv (Google Robot; Table 1):
    - > “MolmoAct (zero-shot) 70.5% visual matching” and “71.6% after RT‑1 fine‑tuning,” outperforming closed or strong open baselines like π0/π0‑FAST, GR00T N1.5, and Magma in relevant regimes.
    - Variant aggregation (OOD) after fine‑tuning: > “72.1%,” exceeding RT‑2‑X by 7.8%; performance drop from visual matching to variant aggregation is <1%, indicating robustness (§5.3).
  - LIBERO (Table 2):
    - > “MolmoAct‑7B‑D average 86.6%,” the best among compared autoregressive policies; especially strong in Long‑horizon: > “77.2%,” a +6.3% gain over ThinkAct (§5.2).
  - Real‑world fine‑tuning (Figure 5; detailed per‑trial tables in Appendix D.3):
    - Single‑arm tasks: average task‑progression improvement of about +10% over π0‑FAST (e.g., Wipe Table average 1.00 vs. 0.817; Table 19).
    - Bimanual tasks: +22.7% over π0‑FAST on average (e.g., Set Table, Fold Towel; Tables 15–17).
  - OOD generalization (Figure 6a; Table 1):
    - Simulation: top performance on variant aggregation (72.1%).
    - Real world: average +23.3% task‑progress improvement over π0‑FAST across language, spatial, distractor, and novel‑object perturbations.
  - Effect of mid‑training (Figure 6b; §5.4):
    - Mid‑training on MolmoAct Dataset provides ~“+5.5%” average improvement across Close Lid, Rotate Pot, Pour Tea; even without mid‑training, MolmoAct outperforms π0‑FAST and OpenVLA by 14.8% and 10.9%.
  - Human preference studies
    - Open‑ended instruction following (Figure 8): MolmoAct receives the highest Elo rating; pairwise wins in 58% vs SpatialVLA and 81% vs OpenVLA.
    - Trace generation on Internet images (Figure 7): MolmoAct attains the top Elo, above GPT‑4o, Gemini‑2.5‑Flash, and a specialized trace VLM (“HAMSTER”).
  - Steerability (Figure 9; §5.6):
    - With ambiguous “pick up the bowl” scenarios, visual‑trace steering with MolmoAct achieves > “0.75 success,” beating MolmoAct with open‑ended language corrections (0.42) and π0‑FAST with language corrections (0.29 difference vs MolmoAct).

- Do the experiments support the claims?
  - Yes. The structured chain (depth→trace→actions) is validated across simulation and real robots. Robustness is evidenced by minimal drop from visual matching to variant aggregation (<1%), strong OOD improvements (Figure 6a), and human evaluations favoring MolmoAct’s instruction following and trace quality (Figures 7–8).
  - Ablations and diagnostics:
    - Impact of mid‑training dataset (Figure 6b).
    - Real‑world per‑trial tables (Appendix D) show consistency and failure cases (e.g., occasional partial progression on Set Table).
  - Conditions and trade‑offs:
    - Visual‑trace steering excels when users can specify precise paths; language-only steering remains weaker when instructions are ambiguous (Figure 9).

## 6. Limitations and Trade-offs
- Assumptions and data dependencies (Appendix G; §3.1)
  - Depth tokens rely on a specialist depth estimator and VQ‑VAE codebook learned from specific data (tabletop‑heavy). Transfer to markedly different camera setups or scenes might degrade without re‑distillation.
  - Visual traces come from gripper pointing via a VLM; accuracy depends on clear visibility. Occlusions or unusual embodiments may hurt trace quality (Limitations: “Camera Occlusion of End‑effector”).
- Steerability is 2D (§G)
  - User sketches are in image space; without explicit 3D lifting, the controller may follow the correct in‑plane path but drift along depth (out‑of‑plane). The paper suggests conditioning on predicted depth tokens to “lift” traces into 3D as future work.
- Control frequency and latency (§G)
  - Predicting depth/trace/action tokens adds latency; real robots may need higher control rates. The paper notes a mismatch between model inference and data collection frequency, suggesting model optimization or edge deployment as future directions.
- Resolution of depth tokens (§G)
  - Depth is compressed into 100 tokens per image; very fine manipulation might need more tokens or higher‑resolution depth encodings.
- Training and compute
  - Although efficient relative to some baselines, the full pipeline still uses substantial GPU hours and curated data generation (e.g., the 26.3M-sample pre‑training mixture; §4.1). High‑quality post‑training demos (30–50 per task) are still required for new tasks (§4.3).

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
