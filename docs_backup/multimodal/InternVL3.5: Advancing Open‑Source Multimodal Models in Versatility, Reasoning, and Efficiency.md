# InternVL3.5: Advancing Open‑Source Multimodal Models in Versatility, Reasoning, and Efficiency

**ArXiv:** [2508.18265](https://arxiv.org/abs/2508.18265)
**Authors:** Weiyun Wang, Zhangwei Gao, Lixin Gu, Hengjun Pu, Long Cui, Xingguang Wei, Zhaoyang Liu, Linglin Jing, Shenglong Ye, Jie Shao, Zhaokai Wang, Zhe Chen, Hongjie Zhang, Ganlin Yang, Haomin Wang, Qi Wei, Jinhui Yin, Wenhao Li, Erfei Cui, Guanzhou Chen, Zichen Ding, Changyao Tian, Zhenyu Wu, Jingjing Xie, Zehao Li, Bowen Yang, Yuchen Duan, Xuehui Wang, Songze Li, Xiangyu Zhao, Haodong Duan, Nianchen Deng, Bin Fu, Yinan He, Yi Wang, Conghui He, Botian Shi, Junjun He, Yingtong Xiong, Han Lv, Lijun Wu, Wenqi Shao, Kaipeng Zhang, Huipeng Deng, Biqing Qi, Jiaye Ge, Qipeng Guo, Wenwei Zhang, Wanli Ouyang, Limin Wang, Min Dou, Xizhou Zhu, Tong Lu, Dahua Lin, Jifeng Dai, Bowen Zhou, Weijie Su, Kai Chen, Yu Qiao, Wenhai Wang, Gen Luo
**Institutions:** Shanghai AI Laboratory, OpenGVLab, Other collaborating institutions (not fully specified)

## 🎯 Pitch

InternVL3.5 revolutionizes multimodal large language models by introducing a two-stage reinforcement learning pipeline, 'Cascade RL,' alongside innovative designs like the Visual Resolution Router (ViR) and Decoupled Vision-Language Deployment (DvD). These advancements not only enhance reasoning capabilities and computation efficiency—achieving up to 16% improvement in reasoning benchmarks and 4× speedups—but also expand real-world applications, making this a significant leap for open-source AI deployments.

---

## 1. Executive Summary (2-3 sentences)
InternVL3.5 is a family of open‑source multimodal large language models (MLLMs) that improve three hard problems at once: (1) robust multimodal reasoning, (2) inference efficiency for high‑resolution and multi‑image/video inputs, and (3) breadth of real‑world capabilities (GUI, embodied, SVG). It introduces a two‑stage reinforcement learning pipeline called `Cascade RL` and two efficiency designs—`Visual Resolution Router (ViR)` and `Decoupled Vision‑Language Deployment (DvD)`—that together deliver up to +16.0% average gains on reasoning benchmarks and up to 4.05× inference speedups over InternVL3 (Abstract; Sections 2.3, 2.5, 3.15; Table 18).

## 2. Context and Motivation
- Problem gap
  - Open‑source MLLMs trail closed‑source systems on complex reasoning, text‑centric, and agentic tasks (Table 2 shows large gaps vs GPT‑5 on Reasoning and Text). Existing RL approaches for MLLMs either:
    - Offline alignment only (e.g., DPO) — efficient but capped performance.
    - Online RL (e.g., PPO‑style) — better ceiling but unstable and compute‑heavy (Section 2.3).
  - Multimodal inputs are getting longer and higher‑resolution; naively pushing more image tokens through an LLM is a major latency and cost bottleneck that blocks real deployments (Abstract; Section 2.5).

- Why it matters
  - Real‑world uses (assistants on screens, robots, scientific agents) need accurate step‑by‑step reasoning and fast multimodal I/O. The paper targets both theoretical/raining stability (RL for reasoning) and systems efficiency (token routing and deployment architecture).

- Prior approaches and shortcomings
  - RL for MLLMs: many efforts report instability or scalability issues; offline RL is easier but limited; online RL is powerful but expensive (Section 2.3; references [104], [108], [109], [183]).
  - Efficiency: “Dynamic High Resolution” splits images into many patches but still grows cost linearly with resolution; prior works rarely decide compression per patch or decouple vision/LLM at serving time (Sections 2.1, 2.5).

- Positioning
  - InternVL3.5 extends the InternVL series with:
    - A cascaded offline→online RL training recipe (`Cascade RL`) to get both efficiency and a high ceiling (Section 2.3; Fig. 3).
    - A semantic `Visual Resolution Router (ViR)` plus consistency training (`ViCO`) to halve visual tokens with negligible quality loss (Sections 2.1, 2.3).
    - A serving architecture (`DvD`) that pipelines vision and language across different GPUs/servers to remove blocking (Section 2.5; Fig. 4).
  - It spans small to very large models (1B–241B) with both dense and MoE variants (Table 1), and demonstrates broad capability improvements across >35 benchmarks (Fig. 1; Table 2).

## 3. Technical Approach
This section explains how InternVL3.5 is built and trained, and how its efficiency features work.

- Model architecture (Section 2.1; Fig. 2; Table 1)
  - Base paradigm: `ViT–MLP–LLM`.
    - Vision encoder: `InternViT-300M` or `InternViT-6B` (Table 1).
    - Connector: an MLP projector aligns vision features to language space.
    - Language model: `Qwen3` series or `GPT-OSS` (Table 1).
  - Dynamic High Resolution (from InternVL1.5): images are tiled to preserve information without resizing excessively (Fig. 2a).
  - Visual tokens and compression:
    - Each image patch first produces 1024 visual tokens for the ViT, then a “pixel shuffle” compresses to 256 tokens before the LLM (Section 2.1).
    - InternVL3.5‑Flash adds an extra, stronger pixel shuffle (down to 64 tokens) and a `patch router` (ViR) that selects 256‑ or 64‑token paths per patch based on semantic richness (Fig. 2c).

- Training pipeline (Fig. 3; Sections 2.2–2.3)
  - Pre‑training (Section 2.2)
    - Objective: next‑token prediction on multimodal sequences with a reweighting scheme to avoid bias toward longer outputs (Eq. (1)–(2)); random JPEG compression augments robustness.
      - Intuition for Eq. (2): the “square averaging” scales losses so a sample’s contribution doesn’t grow just because it has more tokens.
    - Data: ~116M samples (~250B tokens). Text‑only : multimodal ≈ 1 : 2.5. Max context 32K (Section 2.2).
  - Supervised Fine‑Tuning (SFT) (Section 2.3)
    - Same loss as pre‑training but with high‑quality conversations and “Thinking‑mode” reasoning traces that are filtered for clarity and consistency, plus capability‑expansion data (GUI, embodied, SVG).
    - Context window stays at 32K tokens.
  - Reinforcement learning: `Cascade RL` (Section 2.3; Fig. 3)
    - Stage 1: Offline RL via `Mixed Preference Optimization (MPO)` minimizes a weighted sum of three losses (Eq. (3)):
      - Preference loss `Lp` (uses `DPO`), quality loss `Lq` (uses `BCO`), and LM loss `Lg`.
      - Rationale: learn from existing positive/negative rollouts efficiently to “warm‑up” the policy and remove low‑quality modes without sampling cost.
    - Stage 2: Online RL via `GSPO` (Group Sequence Policy Optimization) (Eq. (4)–(6)):
      - For each query, sample multiple responses, compute standardized rewards to form an “advantage” (Eq. (4)); optimize a PPO‑style clipped objective using a geometric mean of per‑token importance ratios (Eq. (6)).
      - Rationale: refine the policy on self‑generated rollouts to push the performance ceiling without a reference model constraint (Section 2.3).
    - Data for RL (Section 2.3):
      - Offline: MMPR‑v1.2 (~200K pairs).
      - Online: MMPR‑Tiny (~70K queries), selected to be neither too easy nor too hard (model accuracy in [0.2, 0.8]); reuses offline rollouts to avoid extra sampling cost.
  - Visual Consistency Learning (`ViCO`) to build InternVL3.5‑Flash (Section 2.3)
    - Goal: make outputs stable regardless of whether a patch is 256 or 64 tokens, then train the router to choose wisely.
    - Stage 1: Consistency training (Eq. (7)): minimize KL divergence between a frozen reference model (always using 256 tokens) and the trainable policy that randomly mixes 256 and 64 tokens. This ties compressed and uncompressed behaviors.
    - Stage 2: Router training (Eqs. (8)–(9)): compute, for each patch, the loss ratio `r_i` when compressing from 256→64 tokens. If `r_i` exceeds a dynamic threshold τ (k‑th percentile over a sliding window), label that patch “needs high resolution” (ξ=1/4→256 tokens); else label “safe to compress” (ξ=1/16→64 tokens). Train the router as a binary classifier with cross‑entropy while keeping the main MLLM frozen.
    - Data: reuse SFT data for consistency; focus on OCR and VQA subsets for router training (Section 2.3).
  - Test‑time scaling (Section 2.4)
    - “Deep Thinking”: prompt the model to reason step‑by‑step before answering.
    - “Parallel Thinking”: Best‑of‑N (BoN) sampling; a critic (`VisualPRM‑v1.1`) selects the best candidate. Applied only to reasoning benchmarks; all main results are without TTS unless noted (Section 2.4).

- System/serving design: `Decoupled Vision‑Language Deployment (DvD)` (Section 2.5; Fig. 4)
  - Separate servers/GPUs for vision and language. Vision server runs ViT+MLP(+ViR) and outputs BF16 features via TCP/RDMA to the language server, which runs only the LLM.
  - Pipeline the three stages—vision compute, feature transfer, LLM prefilling/decoding—so they overlap rather than block one another (Fig. 4b).

## 4. Key Insights and Innovations
- `Cascade RL` (fundamental training innovation; Section 2.3; Fig. 3; Table 15; Fig. 5)
  - What’s new: a deliberate two‑stage RL curriculum—Offline MPO for safe/cheap pruning of bad modes, then Online GSPO to refine the policy on its own samples.
  - Why it matters: combines stability and efficiency of offline RL with the performance ceiling of online RL, and scales from 1B to 241B (Fig. 5).
  - Evidence:
    - > “InternVL3.5‑8B: SFT 53.6 → MPO 56.3 → CascadeRL 60.3 (avg across reasoning benchmarks)” (Table 15).
    - > “CascadeRL achieves larger gains than MPO and similar or better than GSPO with roughly half the GPU hours” (Table 16: CascadeRL ~5.8K GPU‑h vs GSPO (2 ep) ~11K GPU‑h for 8B, with 60.3 vs 58.2 overall).

- `Visual Resolution Router (ViR)` + `ViCO` (semantic token compression; Sections 2.1, 2.3; Tables 17–18)
  - What’s new: choose compression per image patch based on sensitivity, not just image size; enforce output consistency between compressed/uncompressed with KL training.
  - Why it matters: reduces visual tokens by ~50% while maintaining nearly original accuracy (Section 2.1: “reduce tokens by 50% with nearly 100% performance”; Table 17 shows minimal deltas).
  - Evidence:
    - > “InternVL3.5‑38B: Overall 83.9 vs 83.4 after ‑Flash across DocVQA/ChartVQA/InfoVQA/TextVQA/OCRBench/AI2D/MMStar/MMMU/MathVista” (Table 17).
    - > “Throughput speedups up to 4.05× at 896px when combining DvD+ViR” (Table 18).

- `Decoupled Vision‑Language Deployment (DvD)` (systems contribution; Section 2.5; Table 18; Fig. 4)
  - What’s new: separate and pipeline vision and language servers; communicate compact BF16 features; overlap prefill/decoding with vision.
  - Why it matters: removes mutual blocking, improves GPU utilization, and speeds up both small and large models.
  - Evidence:
    - > “InternVL3.5‑38B: baseline 2.71 rps → +DvD 5.06 rps (1.87×) → +DvD+ViR 10.97 rps (4.05×) at 896px” (Table 18).
    - > “InternVL3.5‑241B‑A28B: baseline 2.54 rps → +DvD 4.73 rps (1.86×) → +DvD+ViR 8.81 rps (3.47×) at 896px” (Table 18).

- Breadth through “native” multimodal pretraining + high‑quality SFT (incremental but important; Sections 2.2–2.3; Table 14)
  - What’s new: joint text and multimodal pretraining at scale plus curated “Thinking‑mode” supervision and capability‑expansion (GUI, embodied, SVG).
  - Why it matters: preserves and even improves text benchmarks while adding new modalities.
  - Evidence:
    - > “InternVL3.5‑241B‑A28B overall text average 87.6 vs its LLM base Qwen3‑235B‑A22B at 85.3” (Table 14).

## 5. Experimental Analysis
- Evaluation design and breadth (Section 3; Fig. 1; Table 2)
  - Benchmarks span general multimodal (MMBench, MMStar, MMVet), reasoning (MMMU, MathVista, etc.), text (MMLU‑Pro, AIME24/25, GPQA, etc.), and agentic tasks (GUI grounding and agents, embodied, SVG).
  - Tools: VLMEvalKit (multiple sections), OpenCompass (Tables 2–6, 14), and official benchmark protocols.
  - Unless specified, results are without test‑time scaling; TTS is only used for some reasoning analyses (Section 2.4).

- Headline quantitative results
  - Overall capability vs leading models (Fig. 1; Table 2)
    - > “InternVL3.5‑241B‑A28B achieves 74.1 average on general multimodal suite; comparable to GPT‑5 at 74.0” (Fig. 1; Table 2, “General Overall”).
    - Open‑source leadership in many categories (e.g., OCRBench and LongVideoBench entries in Table 2).
  - Multimodal reasoning and math (Table 3)
    - > “InternVL3.5‑241B‑A28B: MMMU 77.7, MathVista 82.7, MathVision 63.9, MathVerse (vision‑only) 68.5; overall 66.9.”
    - Gains are consistent at all scales; e.g., 2B overall 50.7 vs prior InternVL3‑2B 32.4 (Table 3).
    - With “parallel thinking” (BoN), additional improvements up to +2–3 points on some models (Table 3, rows with “w/ Parallel Thinking”).
  - OCR, chart, and document understanding (Table 4)
    - Strong across AI2D, DocVQA, InfoVQA, OCRBench; e.g., > “InternVL3.5‑30B‑A3B overall 83.9” (Table 4).
  - Multi‑image & real‑world (Tables 5)
    - > “InternVL3.5‑38B overall 67.4 on multi‑image suite; InternVL3.5‑241B‑A28B 65.5.” Real‑world sets (RealWorldQA, MME‑RealWorld, WildVision, R‑Bench) also strong (right side of Table 5).
  - Comprehensive multimodal & hallucination (Table 6)
    - > “InternVL3.5‑38B: MMBench v1.1=87.3; MMVet=82.2; MMStar=75.3; HallBench=59.7; CRPE=77.7; POPE=90.4.”
    - Some hallucination metrics mixed at larger scale (see 14B, 241B “Overall” in hallucination columns).
  - Visual grounding saturation (Table 7)
    - High 90%+ across RefCOCO series; > “InternVL3.5‑241B‑A28B overall 92.4 (SoTA among reported)”—task seems near saturation.
  - Multilingual multimodal (Table 8)
    - Strong across 6 languages; > “InternVL3.5‑241B‑A28B: MTVQA overall 39.3 (higher is better, harder set), and top scores on MMMB/Multilingual MMBench columns.”
  - Video understanding (Table 9)
    - Competitive at scale; > “InternVL3.5‑38B: MVBench 75.0; MMBench‑Video 1.90; MLVU 77.0.”
  - GUI grounding and online agents (Table 10)
    - > “InternVL3.5‑241B‑A28B: ScreenSpot‑v2 92.9; OSWorld‑G 53.2; WindowsAgentArena 18.0; WebArena‑Lite‑v2 11.7” — far above many generalist models tested with the same 50‑step budget.
  - Embodied/spatial reasoning (Table 11)
    - > “InternVL3.5‑241B‑A28B overall 55.8; VSI‑Bench 69.5 (top among listed).”
  - SVG understanding/generation (Tables 12–13)
    - Understanding (SGP‑Bench): > “InternVL3.5‑241B‑A28B overall 70.7; InternVL3.5‑38B 69.5” — both state‑of‑the‑art among open models.
    - Generation (SArena‑Icon): > “InternVL3.5‑38B Text2SVG FID 14.56 (lower is better), better than GPT‑4o at 15.18; 241B improves to FID 11.27” (Table 13).
  - Text capability vs base LLMs (Table 14)
    - > “InternVL3.5‑241B‑A28B improves the overall text average from 85.3 (Qwen3‑235B‑A22B) to 87.6,” with large math gains on MATH/AIME.

- Ablations and efficiency (Section 3.15; Tables 15–18; Fig. 5)
  - Training stages: consistent benefits SFT→MPO→CascadeRL at all scales (Table 15; Fig. 5).
  - Training cost vs gain: CascadeRL delivers the largest gains per GPU hour compared to MPO alone or long GSPO runs (Table 16).
  - Efficiency: DvD yields ~1.2–2.0× rps; DvD+ViR yields up to 4.05× at 896px on 38B (Table 18) with minor performance deltas (Table 17).

- Do results support the claims?
  - Yes on three fronts:
    - Reasoning: Broad, consistent gains with scale and with ablations isolating RL stages (Table 15).
    - Efficiency: Clear throughput gains with DvD and ViR, and minimal quality loss (Tables 17–18).
    - Versatility: Substantial new strengths on GUI/embodied/SVG while preserving text capability (Tables 10–13, 14).

- Caveats and mixed results
  - Hallucination metrics improve overall but are inconsistent at some scales (Table 6, hallucination “Overall”).
  - Visual grounding is saturated; little headroom remains (Table 7).
  - Some large‑scale general understanding scores change marginally vs InternVL3 (Table 6, “Overall” rows), suggesting optimization focused more on reasoning and text than perception.

## 6. Limitations and Trade-offs
- Training/resource assumptions
  - Cascade RL still requires substantial compute (e.g., ~5.8K GPU‑hours for an 8B model in Table 16) even if cheaper than pure online RL; scaling to 241B/MoE adds engineering complexity (Table 1).
  - RL datasets depend on curated rollouts and reward models/filters (Section 2.3) which may encode biases or coverage gaps.

- Efficiency trade‑offs
  - ViR introduces a router and extra training (consistency + router phases in Section 2.3); slight performance drops remain in a few tasks (Table 17).
  - DvD needs network bandwidth and careful pipeline tuning; feature transfer could become a bottleneck without RDMA or with many concurrent requests (Section 2.5).

- Scope and scenarios not fully addressed
  - Hallucination robustness is improved but not uniformly across scales (Table 6).
  - Video performance is competitive but not dominant across all metrics; long video and multi‑video settings may still stress memory and throughput (Table 9).
  - Visual grounding is near saturation; future differentiation requires new task formulations (Section 3.8).

- Methodological constraints
  - Test‑time scaling is applied only to reasoning benchmarks; broader applicability or safety of BoN selection with VisualPRM is not explored (Section 2.4).
  - Router labels depend on KL‑based loss ratios against a frozen reference; this objective may not perfectly reflect downstream task utility for every patch type (Eqs. (7)–(9)).

## 7. Implications and Future Directions
- How this work shifts the field
  - Provides a practical recipe to make RL for MLLMs stable and scalable (`Cascade RL`), which others can adopt or extend.
  - Introduces a principled approach to per‑patch token budgeting (`ViR`+`ViCO`) that achieves big speedups without rewriting the core model.
  - Demonstrates a serving architecture (`DvD`) that better matches the heterogeneous compute patterns of vision vs language—important for production deployments.

- Follow‑up research enabled or suggested
  - RL methods:
    - Reward design for multimodal reasoning (richer process supervision; curriculum schedules during Cascade RL).
    - Adaptive per‑task switching between offline/online RL or automatic episode scheduling to maximize benefit vs cost (Table 16 hints at efficient operating points).
  - Token routing:
    - Extend `ViR` to multi‑image sequences and video (temporal‑aware routing of frames/patches).
    - Learn router signals from downstream task rewards instead of KL alone; explore multi‑level routing (region→patch→token).
  - Systems:
    - Joint co‑design of DvD with cache‑aware feature codecs; prioritize essential semantic channels in transmission.
    - Multi‑tenant scheduling across vision and language servers to guarantee QoS at scale.

- Practical applications and downstream use
  - GUI agents and RPA‑like flows on real desktops (Table 10); assistants that “use a computer.”
  - Embodied agents with spatial reasoning (Table 11); robotics/navigation/AR assistants.
  - Document/chart/OCR understanding at scale (Table 4); enterprise search and analytics.
  - SVG understanding and generation (Tables 12–13); UI design, data‑driven graphics, iconography.
  - Strong text and math reasoning (Table 14; Table 3), supporting educational tools and scientific assistants.

> Key takeaway: InternVL3.5 shows that careful combination of offline and online RL plus semantic‑aware visual token budgeting and decoupled serving can simultaneously improve reasoning quality and end‑to‑end efficiency—two axes that often trade off. The open‑source release across sizes (Table 1) makes these ideas broadly testable and extensible.
