# Autoregressive Universal Video Segmentation Model

**ArXiv:** [2508.19242](https://arxiv.org/abs/2508.19242)
**Authors:** Miran Heo, Sukjun Hwang, Min‑Hung Chen, Yu‑Chiang Frank Wang, Albert Gu, Seon Joo Kim, Ryo Hachiuma
**Institutions:** NVIDIA, Yonsei University, Carnegie Mellon University, National Taiwan University

## 🎯 Pitch

AUSM revolutionizes video segmentation by introducing an autoregressive model that unifies prompted and unprompted segmentation through efficient next-frame mask prediction, akin to next-token prediction in language models. This innovation not only streamlines integration into diverse video systems but also significantly reduces training times and resource needs, offering robust accuracy across benchmarks and paving the way for future advancements in multi-task video corpora training.

---

## 1. Executive Summary (2-3 sentences)
This paper introduces AUSM, an autoregressive, single-architecture model that unifies prompted and unprompted video segmentation by predicting “the next frame’s masks” in the same spirit that language models predict the next token. It achieves constant-memory streaming inference via a state-space temporal module and, unlike most video segmentation systems, supports fully parallel training across frames, yielding up to 2.5× faster training on 16-frame clips (Fig. 4) while delivering state-of-the-art results among universal online methods across seven benchmarks (Table 1).

## 2. Context and Motivation
- Problem and gap
  - Video segmentation currently splits into two regimes:
    - `Prompted` segmentation (often called VOS): segment and track one or more user-specified targets given an initial cue (mask/box/point/text) in the first frame.
    - `Unprompted` segmentation (VIS/VPS): automatically detect, classify, segment, and track all instances throughout the video without user prompts.
  - Real-world systems need both: interactive editing requires prompted; autonomous perception (e.g., driving) requires unprompted. Today, these are handled by different models and pipelines, fragmenting supervision and engineering effort (Sec. 1).
- Why it matters
  - Unifying tasks allows training a single model on diverse supervision and deployment in multiple scenarios. Video data is expensive; a universal model amortizes annotation cost and simplifies productionization (Sec. 1).
- Shortcomings of prior approaches
  - Prompted VOS often uses explicit memory buffers that store several past frames with masks (e.g., STM/XMem/SAM2). These are powerful but not naturally designed for unprompted discovery of new objects and can be memory-heavy at long horizons (Sec. 1; Table 1).
  - Many unprompted VIS systems either:
    - Process each frame independently and then associate detections post hoc, losing fine temporal detail for detection (Sec. 1).
    - Or propagate compact object tokens (queries), compressing rich mask history to a few vectors, which hurts fine-grained mask quality—especially for VOS (Sec. 1).
  - Prior “universal” models (e.g., UNINEXT, UniVS, TarViS) typically retrofit one regime onto the other via heavy token compression or offline access to the whole video, leading to noticeable VOS drops and/or lack of online streaming capability (Sec. 1; Table 1).
  - Training scalability is another gap: many video models must train recurrently over frames; few support language-model-style parallel training (Sec. 1, 2.3).
- Positioning of this work
  - AUSM recasts streaming video segmentation as autoregressive mask prediction: predict `y_t` given past segmentations and frames—exactly analogous to next-token prediction (Sec. 2.1; Eq. (1)–(2)).
  - It unifies prompted and unprompted regimes through how the “initial prompt” `y_0` is set (empty vs. user masks), and it is designed for constant-memory streaming inference and fully parallel training (Sec. 2.2–2.3; Fig. 1).

## 3. Technical Approach
AUSM’s design mirrors decoder-only language models, replacing next-token prediction with next-frame mask prediction. It maintains a small set of vectors that represent tracked identities and uses a compressed temporal state to carry history forward.

- Core formulation (Sec. 2.1)
  - Language analogy: language models factorize sequence probability via `P(y_1:T) = Π_t P(y_t | y_<t)` (Eq. (1)).
  - AUSM applies the same idea to video segmentation with visuals and optional prompts:
    - `P(y_1:T | I_1:T) = Π_t P(y_t | y_0, y_<t, I_≤t)` (Eq. (2)),
    - where `y_0` is an initial prompt (e.g., a mask in the first frame for prompted VOS) or empty for unprompted VIS.
- Data structures and state (Sec. 2.2; Alg. 1)
  - `V ∈ R^{N_det × D}`: a fixed pool of object “detection queries” used to find new instances in each frame.
  - `B ∈ R^{N_id × D}`: a buffer of “ID vectors” available for assignment to newly discovered instances; each assigned ID vector maintains a persistent track.
  - `A_t`: the set (subset of `B`) of allocated ID vectors currently tracking objects.
  - `M_t`: the set of masks corresponding one-to-one with `A_t`.
  - `X_t`: frame features produced by a frame-independent image backbone; `X_0` is a learned “start” token tiled spatially.
- Inference loop (recurrent form, Alg. 1; Fig. 1-left)
  1. Initialization (Lines 2–9):
     - Unprompted: `A_0 = ∅`, `M_0 = ∅`, `B_0 = B`.
     - Prompted: sample `|y_0|` ID vectors from `B` to form `A_0 = Sampler(B, |y_0|)`; set `M_0 = m_0` (the given masks); remove those vectors from `B_0`.
  2. At each time `t` (Lines 10–20):
     - History marking (Line 11): use `HistoryMarker(A_{t-1}, M_{t-1})` to convert the set of tracked instances into a dense spatial feature map `S_t`. Mechanism: for each pixel, compute the (normalized) average of the ID vectors whose masks cover that pixel,
       > `S_t[h,w,:] = (Σ_i M_{t-1}^i[h,w] · A_{t-1}^i) / (ε + Σ_i M_{t-1}^i[h,w])` (Sec. 2.2, “History Marker”).
       - Intuition: rather than compressing each object into a single vector only, the ID vectors are “dissolved” into the pixels their masks occupy, preserving fine spatial details.
       - Set `E_t = X_{t-1} + S_t`.
     - History compression (Line 12; Fig. 2): `HistoryCompressor(E_t)` updates a single spatial state `F_t`. Each compressor layer has:
       - A temporal `Mamba` block (an SSM) that models dependencies across time efficiently with constant-memory recurrence.
       - A spatial self-attention block that captures relationships across pixels within a frame.
       - A feed-forward network (Sec. 2.2).
       - Why Mamba? Videos are long token streams; Mamba’s recurrent selective state spaces let AUSM keep only a fixed-size state instead of a growing key-value cache (Sec. 2.2, “History Compressor”).
     - History decoding (Line 13): `HistoryDecoder(Q=X_t, KV=F_t)` fuses current frame evidence with the compressed temporal state to produce a per-pixel feature map `G_t`. This is a stack of Transformer decoder layers (Sec. 2.2).
     - Pixel-level prediction (Line 14): `PixelDecoder(Q=concat(A_{t-1}, V), KV=G_t)` outputs two sets of predictions:
       - `ŷ_trk_t`: “tracking” predictions for each ID vector in `A_{t-1}` (one-to-one mapping to tracked objects).
       - `ŷ_det_t`: “detection” predictions for the fixed query pool `V` (to discover new objects). The Pixel Decoder uses masked attention and Hungarian-style detection heads akin to Mask2Former/DETR variants (Sec. 2.2).
     - Update sets (Lines 15–19):
       - `D = filter_fg(ŷ_det_t)`: keep only confident foreground detections.
       - Assign new IDs: sample `|D|` vectors `A′` from the remaining buffer `B_{t-1}` (Line 16); update `A_t = concat(A_{t-1}, A′)` and `B_t = B_{t-1} \ A′`.
       - Update masks: `M_t = concat( m̂_trk_t, D )` (tracked masks + masks for new detections).
- Parallel training (teacher forcing) across frames (Sec. 2.3; Alg. 2; Fig. 1-right)
  - Challenge: recurrent propagation makes many video models train sequentially. AUSM avoids this by ensuring every module accepts ground-truth conditioning in parallel (teacher forcing).
  - `Preprocess` builds per-frame targets for detection and tracking and the “teacher” history:
    - For each ground-truth instance `i`, sample a timestep `t_sample^i` to switch from being treated as a detection target (`y_det^i`) to a tracking target (`y_trk^i`). Masks `M^i_t` are set to ground-truth for all `t ≥ t_sample^i` (Fig. 3 and the formulas in Sec. 2.3).
    - Construct `A_{t-1}` to include exactly the ID vectors for instances present in `M_{t-1}` (one-to-one).
  - With these ground-truth histories, compute `E_{1:T}`, `F_{1:T}`, `G_{1:T}` and predictions for all `t` in parallel (Alg. 2, Lines 4–7).
  - Losses sum over frames (Alg. 2, Line 8):
    - Tracking loss `L_trk` uses the given one-to-one mapping between `y_trk_t` and `A_{t-1}`.
    - Detection loss `L_det` uses Hungarian matching between `y_det_t` and `V` (Sec. 2.3).
- Implementation details (Appendix A)
  - 6-layer `HistoryCompressor`, 6-layer `HistoryDecoder`, 100 detection queries `N_det`, and 100 ID vectors `N_id`; features at 1/8 resolution with 256 dims; Swin backbones (Swin-T/B) (Appendix A).
  - Training in three stages: COCO pseudo-videos (Stage 1), multi-source 5-frame clips (Stage 2), 16-frame “long-clip adaptation” with frozen backbone (Stage 3) (Sec. 3.2; Fig. 5).
  - Inference: shorter side 1024; in unprompted mode use a fixed foreground threshold 0.5 and keep top-10 (YTVIS) or top-20 (OVIS) instances per frame; no post-processing for prompted mode (Appendix A).
- Optional inference-time compute scaling (Sec. 3.4 “Scaling Inference Compute”; Table 3)
  - Repeat the input image or repeat the video forward–backward–forward to allow iterative refinement; take only the last pass’s outputs. This increases COCO AP from 34.2→35.0 and YTVIS19 AP from 62.6→63.5 at ×3 repetitions (Table 3). Appendix A adds a “spatial traversal” variant, improving COCO from 34.2→35.9.

## 4. Key Insights and Innovations
- Autoregressive unification of prompted and unprompted segmentation (Sec. 2.1; Eq. (2))
  - What’s new: a single probability factorization that conditions on past frames and past segmentations—with an optional initial prompt `y_0`—covers both settings. There’s no need for separate architectures.
  - Why it matters: enables one model to handle interactive (prompted) and fully automatic (unprompted) scenarios with shared weights (Table 1 reports both regimes from the same model).
- `History Marker`: dissolving masks into per-pixel identity features (Sec. 2.2 “History Marker”)
  - What’s different: instead of compressing each object’s history into a few vectors (common in token-propagation VIS), AUSM spreads each ID vector across all pixels it occupies, preserving spatial detail.
  - Impact: the paper attributes a “nearly 10% improvement in VOS” over previous unified online architectures by preserving fine-grained information (Sec. 2.2). This is a conceptual shift from vectorizing instances to distributing identity labels spatially.
- `History Compressor`: constant-memory temporal modeling with Mamba + spatial self-attention (Fig. 2; Sec. 2.2)
  - What’s different: a per-pixel temporal SSM (Mamba) recurrently updates one spatial state for the whole stream. This avoids storing frame-by-frame key/value caches or FIFO memory banks.
  - Why it matters: makes inference time per frame constant and enables arbitrarily long streams. It also removes explicit object-specific memory buffers typical in STM-like VOS systems (Sec. 2.2 and the concluding remarks in Sec. 3.3).
- Fully parallel training via teacher forcing across all modules (Sec. 2.3; Fig. 1-right; Alg. 2)
  - What’s different: many prior online methods require sequential training because they condition on their own previous predictions. AUSM’s modules accept ground-truth histories so the whole time dimension can be batched.
  - Impact: strong empirical speedup—up to 2.5× faster at 16 frames over an iterative baseline (Fig. 4)—with larger gains expected on longer sequences.

## 5. Experimental Analysis
- Evaluation methodology (Sec. 3.1–3.2; Table 1)
  - Benchmarks
    - Prompted: DAVIS 2017 (J&F), YouTube-VOS 2018 & 2019 (G = average J&F over seen/unseen classes), MOSE (J&F).
    - Unprompted: YouTube-VIS 2019 & 2021 and OVIS (AP).
  - Training data: COCO, DAVIS17, MOSE, SA-V, YTVIS19/21, OVIS; with dataset-specific classification heads where applicable (Sec. 3.1; Appendix A).
  - Setup: single universal model; Swin-T and Swin-B variants; 3-stage training including 16-frame long-clip adaptation (Sec. 3.2).
- Main results (Table 1)
  - Among universal online methods, AUSM delivers strong or best results across seven datasets using a single model:
    - AUSM (Swin-B) on prompted tasks:
      - DAVIS17 J&F: 81.6
      - MOSE J&F: 62.1
      - YTVOS18 G: 80.2
      - YTVOS19 G: 79.1
    - On unprompted tasks:
      - YTVIS19 AP: 62.6
      - YTVIS21 AP: 58.6
      - OVIS AP: 45.5 (highest among universal models in Table 1).
  - Comparisons
    - Against the recent universal streaming baseline UniVS (Swin-L), AUSM (Swin-B) improves:
      > YTVOS18: 80.2 vs 71.5 (+8.7), YTVIS19: 62.6 vs 60.0 (+2.6), YTVIS21: 58.6 vs 57.9 (+0.7), OVIS: 45.5 vs 41.7 (+3.8) (Table 1).
    - Against specialized prompted VOS models, memory-heavy systems still lead:
      > SAM2 (Hiera-L, trained with extra data) scores 90.7 on DAVIS17 J&F and 77.9 on MOSE, ahead of AUSM’s 81.6/62.1 (Table 1).
    - Against specialized VIS models on unprompted tasks, AUSM is competitive:
      > On YTVIS19, AUSM 62.6 vs DVIS 63.9 and VISAGE 64.2; on OVIS, AUSM 45.5 vs VISAGE 46.5 and DVIS 47.1 (Table 1).
- Efficiency and scalability evidence
  - Parallel vs iterative training time (Fig. 4):
    > At sequence length 16, iterative training takes 8.75s/iter vs 3.45s/iter for AUSM’s parallel pipeline (2.5× speedup). Both start at 1.47s/iter for length 1.
  - Long-clip adaptation improves quality (Fig. 5):
    > Moving from 5-frame (Stage 2) to 16-frame (Stage 3) training increases scores on MOSE (+4.5), YTVOS18 (+1.9), YTVOS19 (+3.1), and OVIS (+5.2).
- Robustness and ablations
  - Foreground threshold stability (Table 2):
    > AUSM’s unprompted AP is relatively stable for thresholds 0.3–0.7; YouTube-VIS peaks near 0.4–0.5; OVIS prefers higher thresholds (up to 46.5 AP at 0.7). The model uses 0.5 by default.
  - Inference compute scaling (Table 3):
    > Repeating the input 3 times improves COCO AP from 34.2→35.0 and YTVIS19 AP from 62.6→63.5. Appendix A’s spatial traversal variant further raises COCO to 35.9.
- Do the experiments support the claims?
  - Unification: A single model produces strong results on both prompted and unprompted tasks (Table 1), with qualitative examples of switching modes at inference (Figs. 6–7).
  - Parallel training: Clear time vs. sequence-length scaling advantage (Fig. 4).
  - Long-range streaming: AUSM removes FIFO memory buffers (Sec. 3.3 narrative; Sec. 2.2 design) and still performs well on OVIS (long, occluded videos), topping universal baselines (Table 1).

## 6. Limitations and Trade-offs
- Fine detail vs. efficiency (Sec. 5 “Limitations”)
  - AUSM primarily operates on 1/8-resolution features and relies on spatially distributed ID vectors rather than dense high-res memories. This yields a gap to specialized memory-heavy VOS on fine-grained mask accuracy (e.g., DAVIS, MOSE vs SAM2 in Table 1).
- Training–inference mismatch risks (Sec. 2.3)
  - Parallel training uses teacher forcing with ground-truth histories; at inference, AUSM must consume its own predictions. Although standard in sequence modeling, this can introduce error accumulation in challenging sequences.
- Very long sequences (Sec. 5 “Discussion”)
  - The model can run on arbitrarily long streams due to the Mamba-based state, but quality still degrades on extremely long contexts; adapting long-context LLM techniques (retrieval, length extrapolation) is proposed as future work.
- Capacity and instance count
  - AUSM fixes `N_id` (100 by default; Appendix A). Videos with more concurrent instances than `N_id` are not explicitly addressed. The model also caps top-k instances per frame at evaluation time (Appendix A).
- Category supervision
  - Classification is done with dataset-specific heads (Appendix A), not with a unified open-vocabulary text encoder. Cross-dataset generalization to unseen categories in unprompted mode may thus require retraining or extending the classification interface.
- Resource needs
  - Training uses 16 A100 GPUs; Stage 3 freezes the backbone to manage memory for 16-frame clips (Appendix A). Larger sequence training may still be constrained by hardware.

## 7. Implications and Future Directions
- How this changes the landscape
  - AUSM demonstrates that an LLM-like autoregressive interface—“next masks given past masks and frames”—can unify prompted and unprompted video segmentation with a single architecture and constant-memory streaming. This reframing opens the door to training on broad multi-task video corpora with a shared temporal state (Sec. 2.1–2.3; Table 1).
- Follow-up research enabled or suggested (Sec. 5 “Discussion”)
  - Long-context techniques from LLMs for video:
    - Retrieval over long histories (e.g., NIAH-style evaluations), length extrapolation mechanisms, or hybrid KV-SSM designs to sustain quality far beyond training horizons.
  - Task expansion under the same interface:
    - Multi-object tracking and single-object tracking by converting boxes/points/scribbles into mask prompts; referring VOS by initializing the state with text embeddings (Sec. 5).
  - Backbones specialized for video:
    - Reduce frame-independent layers and strengthen temporal modules (History Compressor/Decoder) to close the VOS quality gap without sacrificing unprompted performance (Sec. 5 “Limitations”).
  - Open-vocabulary and vision-language integration:
    - Replace dataset-specific classification heads with text-conditioned heads (e.g., CLIP-like) to generalize categories in unprompted mode.
  - Test-time compute policies:
    - Systematic inference-time repetition or spatial traversal (Sec. 3.4; Appendix A) as a controllable knob for quality vs. latency in deployment.
- Practical applications
  - Interactive video editing and content creation (prompted mode).
  - Autonomous perception (unprompted mode) in robotics, driving, or surveillance where constant-latency streaming without large memory buffers is valuable.
  - Edge deployments where constant memory and per-frame compute predictability are critical.

> Bottom line: AUSM provides a principled, efficient bridge between language-style autoregression and video segmentation, yielding a practical universal streaming system with competitive accuracy and significantly better training scalability.
