# InternVL3: Exploring Advanced Training and Test‑Time Recipes for Open‑Source Multimodal Models

**ArXiv:** [2504.10479](https://arxiv.org/abs/2504.10479)
**Authors:** Jinguo Zhu, Weiyun Wang, Zhe Chen, Zhaoyang Liu, Shenglong Ye, Lixin Gu, Yuchen Duan, Hao Tian, Weijie Su, Jie Shao, Zhangwei Gao, Erfei Cui, Yue Cao, Yangzhou Liu, Weiye Xu, Hao Li, Jiahao Wang, Han Lv, Dengnian Chen, Songze Li, Yinan He, Tan Jiang, Jiapeng Luo, Yi Wang, Conghui He, Botian Shi, Xingcheng Zhang, Wenqi Shao, Junjun He, Yingtong Xiong, Wenwen Qu, Peng Sun, Penglong Jiao, Lijun Wu, Kaipeng Zhang, Huipeng Deng, Jiaye Ge, Kai Chen, Limin Wang, Min Dou, Lewei Lu, Xizhou Zhu, Tong Lu, Dahua Lin, Yu Qiao, Jifeng Dai, Wenhai Wang
**Institutions:** 

## 🎯 Pitch

InternVL3 revolutionizes multimodal AI by introducing a unified pre-training approach that integrates text and vision data from the start, eliminating complex post-hoc alignment stages. This innovation significantly enhances model efficiency and robustness, achieving state-of-the-art results across various benchmarks and proving vital for scaling practical applications and maintaining strong language capabilities.

---

## 1. Executive Summary (2-3 sentences)
InternVL3 introduces a “native multimodal pre‑training” recipe that jointly trains a vision–language model on both text-only and image/video–text data in one stage, rather than first training a text LLM and then retrofitting vision. Combined with a variable visual position encoding (`V2PE`), a two‑stage post‑training (SFT + Mixed Preference Optimization), test‑time scaling with a critic (VisualPRM), and a scalable training system, InternVL3 achieves state‑of‑the‑art open‑source results on many multimodal benchmarks (e.g., 72.2 on MMMU; Table 2) while preserving strong language ability (Table 11).

## 2. Context and Motivation
- Problem/gap:
  - Most strong multimodal LLMs (MLLMs) are built “post‑hoc”: pre-train a text‑only LLM, then add vision with extra modules, freezing strategies, and multi‑stage alignment (Section 1). This introduces alignment complexity, risks degrading language skill, and demands specialized data (e.g., OCR) and careful schedules [73, 7, 5, 18].
- Importance:
  - Multimodal models power real applications (document understanding, GUI agents, charts, video, spatial reasoning). Reducing the complexity and fragility of training pipelines matters for both practical scalability and research reproducibility (Figure 1; Sections 1 and 2).
- Prior approaches and limitations:
  - “Post‑hoc” adaptation pipelines commonly:
    - Freeze/partially tune the LLM or vision encoder to avoid catastrophic forgetting (Section 1).
    - Add modality bridges and staged fine‑tuning to align text and vision spaces.
    - Rely on curated data to patch modality gaps (e.g., OCR-heavy corpora).
  - These add engineering friction and make training brittle and costly.
- Positioning of this work:
  - InternVL3 unifies language and multimodal learning in a single pre‑training loop (“native multimodal pre‑training”), then applies modern post‑training and test-time scaling recipes (Sections 2.2–2.4). It also scales training efficiently with a new infrastructure (Section 2.5). The result is a simpler, more robust pipeline that is competitive with leading closed models across many tasks (Figures 1–2; Tables 2–10).

## 3. Technical Approach
InternVL3 keeps the familiar “ViT‑MLP‑LLM” architecture but changes how it is trained and scaled (Section 2).

- Architecture (Section 2.1; Table 1):
  - Vision encoder: `InternViT-300M` or `InternViT-6B`.
  - Language model: pre‑trained Qwen2.5 or InternLM3 base LLMs (base, not instruction‑tuned).
  - Multimodal adapter: a 2‑layer MLP to project visual tokens into the LLM embedding space.
  - Pixel unshuffle (as in InternVL2.5): reduces visual token count 4× so each 448×448 image tile is 256 visual tokens (scales to high resolutions).
  - Resulting model sizes: from `~0.9B` to `~78.4B` parameters (Table 1).

- Variable Visual Position Encoding (`V2PE`) (Section 2.1; Eqs. 1–4):
  - Goal: allow much longer mixed (text+vision) contexts without blowing up the positional index range.
  - Mechanism:
    - Each input is a sequence of tokens x = [x1, …, xL] (Eq. 1). For each token i, compute a position index `pi` recursively: `pi = fpos(pi-1, xi)` (Eq. 2).
    - Unlike standard encodings that add +1 per token, `V2PE` increments by:
      - +1 for textual tokens,
      - +`δ` for visual tokens, where `δ` is a small fraction (Eq. 3).
    - During training, `δ` is randomly sampled per image from {1, 1/2, 1/4, …, 1/256} (Eq. 4). During inference, `δ` can be chosen to keep the total context within range (Section 2.1).
  - Intuition: you “compress” the visual positions (smaller steps) so long visual sequences fit into the model’s effective positional window with less loss of relative order.

- Native multimodal pre‑training (Section 2.2):
  - One joint autoregressive objective on mixed text-only and multimodal corpora. The model predicts only text tokens while conditioning on everything (images/videos + preceding text):
    - Full autoregressive form (Eq. 5), restricted to text tokens (Eq. 6).
  - Loss weighting uses “square averaging” to balance gradients w.r.t. response length (Eq. 7).
  - All parameters (ViT, MLP, and LLM) are updated jointly—no freezing or staged alignment (Eq. 8).
  - Data mixture and sampling:
    - Combine large-scale multimodal data (image/video–text; domains include captioning, QA, math, charts, OCR, documents, GUI, tools, 3D, video) with pure text corpora (from InternLM2.5 and other open datasets) (Data paragraphs in Section 2.2 and 2.3).
    - Two‑stage sampling search yields a 1:3 ratio (language : multimodal) as best overall (Section 2.2).
    - Total tokens ≈ 200B (≈50B language + ≈150B multimodal).

- Post‑training (Section 2.3):
  - Stage 1: Supervised Fine‑Tuning (SFT)
    - Trains on higher‑quality, diverse instruction data (tool use, 3D scenes, GUI ops, long context, video, scientific diagrams, creative writing, multimodal reasoning).
    - Uses JPEG compression augmentation, square loss re-weighting, and multimodal data packing (from InternVL2.5).
  - Stage 2: Mixed Preference Optimization (`MPO`)
    - Addresses distribution shift between teacher-forced training and free‑generation at inference by aligning to both relative and absolute preferences (Eq. 9).
    - Three components:
      - Preference loss: DPO (Eq. 10) compares chosen vs. rejected responses.
      - Quality loss: BCO (Eqs. 11–13) learns absolute quality for chosen and rejected answers with a moving reward shift `δ` (not to be confused with V2PE’s `δ`).
      - Generation loss: standard LM loss on preferred responses (Eq. 6).
    - Training data: ~300K preference samples spanning many domains (Section 2.3).

- Test‑time scaling with a critic (Section 2.4):
  - Best‑of‑N sampling: generate N candidate solutions; select the best using a learned critic.
  - VisualPRM-8B (a Visual Process Reward Model):
    - Scores each intermediate step of a solution (“+” or “−”), then averages step scores into a final score (Eq. 14).
    - Trained on VisualPRM400K, expanded with rollouts from InternVL3-8B/38B.
  - In practice, Best‑of‑8 (“Bo8”) with VisualPRM improves reasoning substantially (Table 2, “w/ VisualPRM-Bo8”).

- Training infrastructure (Section 2.5):
  - Extends `InternEVO` to MLLMs:
    - Flexible sharding per module (ViT/MLP/LLM), overlapping communication and computation.
    - Supports data, tensor, sequence, and pipeline parallelism; head-parallel + sequence-parallel for up to 32K tokens.
    - Addresses compute imbalance between visual and text tokens by dynamic load balancing.
  - Reported speedups: 50%–200% vs InternVL2.5 training at similar model sizes.

## 4. Key Insights and Innovations
- Native multimodal pre‑training (fundamental):
  - What’s new: Jointly train on text-only and multimodal data from the start, predicting only text while conditioning on visual tokens (Eqs. 5–6).
  - Why it matters: Removes the fragile, manual alignment stages; all parameters co‑adapt, improving both language and vision without extra bridges or freezes (Sections 2.2 and 3.14, Figure 3).
  - Evidence: A controlled study replacing InternVL2‑8B’s MLP‑warmup with native multimodal pre‑training yields comparable or better performance even before instruction tuning (Figure 3), and better after SFT.

- Variable Visual Position Encoding, `V2PE` (incremental but impactful):
  - What’s new: Modality‑specific positional increments with smaller steps for vision tokens (Eqs. 2–4).
  - Why it matters: Extends usable mixed contexts without losing positional fidelity; ablation shows gains even on moderate‑length tasks when included during pre‑training (Table 12).

- Mixed Preference Optimization, `MPO` (algorithmic advance in post‑training):
  - What’s new: Mixes DPO (relative preference), BCO (absolute quality), and LM loss (Eqs. 9–13).
  - Why it matters: Better aligns the model’s free‑generation distribution with high‑quality responses, improving complex reasoning. Gains scale with model size (Table 13).

- Test‑time scaling with VisualPRM (practical recipe):
  - What’s new: A step‑wise critic for multimodal reasoning that selects the best of multiple candidates (Section 2.4).
  - Why it matters: Significant boosts on math/reasoning (e.g., InternVL3‑38B: Overall +3.8 with Bo8; Table 2).

- Scalable training system (`InternEVO` for vision–language) (engineering innovation):
  - What’s new: Decoupled sharding per module, parallelism mixes, dynamic balancing of visual/text compute (Section 2.5).
  - Why it matters: Enables training models up to ~78B parameters with long contexts and better utilization; 50–200% speedups over the previous generation.

## 5. Experimental Analysis
- Evaluation methodology:
  - Breadth: Over a dozen benchmark families covering multidisciplinary reasoning (MMMU), visual math (MathVista, MathVerse, etc.), OCR/doc/chart understanding (AI2D, ChartQA, DocVQA, InfoVQA, OCRBench), multi‑image reasoning (BLINK, Mantis‑Eval, etc.), real‑world photos (RealWorldQA, MME‑RW, WildVision, R‑Bench), comprehensive multimodal ability (MME, MMBench, MMVet, MMStar), multilingual multimodal (MMMB, Multilingual MMBench, MTVQA), visual grounding (RefCOCO family), video (Video‑MME, MVBench, MLVU, LongVideoBench, CG‑Bench), GUI grounding (ScreenSpot, ScreenSpot‑V2), and spatial reasoning (VSI‑Bench) (Sections 3.1–3.12; Tables 2–10; Figure 1).
  - Models: InternVL3 sizes from 1B to 78B; comparisons to open and closed baselines (Tables 2–10).
  - Test‑time scaling: Best‑of‑N with `VisualPRM‑8B` for reasoning/math (Table 2, “w/ VisualPRM‑Bo8”).
  - Language evaluation: extensive suite via OpenCompass (Table 11).

- Main quantitative results (selected highlights with sources):
  - Overall multimodal strength:
    - > “InternVL3‑78B achieves 72.2 on MMMU” (Table 2).
    - Figure 1 shows across tasks InternVL3 overtakes earlier open‑source MLLMs and is competitive with proprietary models.
    - Figure 2 places InternVL3 high on the OpenCompass multimodal leaderboard; 78B competes with Gemini‑2.5‑Pro.
  - Reasoning and math (Table 2):
    - `InternVL3‑78B`: MMMU 72.2; MathVista 79.0; MathVerse (Vision‑Only) 51.0; LogicVista 55.9.
    - Best‑of‑8 further lifts many scores (e.g., `InternVL3‑38B` Overall from 52.8 to 56.6).
  - OCR/doc/chart (Table 3):
    - `InternVL3‑78B`: OCRBench 906; AI2D 89.7/96.0 (with / without module? Table shows with/without “M”); ChartQA 89.7; DocVQA 95.4; VCR EN‑Easy 96.0/98.6.
  - Multi-image understanding and real‑world photos (Table 4):
    - `InternVL3‑78B`: BLINK 66.3; Mantis‑Eval 79.3; MMT 73.2; MIRB 64.3; RealWorldQA 78.0; WildVision 73.6 win‑rate.
  - Comprehensive multimodal capability and hallucination (Table 5):
    - `InternVL3‑78B`: MME 2549.8; MMBench EN/CN 89.0/88.7; MMVet 81.3; HallusionBench 59.1; CRPE 79.2; POPE 90.3.
  - Visual grounding (Table 6):
    - High but plateauing at the top end: `InternVL3‑78B` Overall 91.4 vs `InternVL2.5‑78B` 92.3; `Qwen2‑VL‑72B` 91.1.
  - Multilingual multimodal (Table 7):
    - `InternVL3‑78B`: best average 68.9 across MMMB, Multilingual MMBench, MTVQA subsets; exceeding `InternVL2.5‑78B` (68.0) and `Qwen2‑VL‑72B` average 67.2.
  - Video (Table 8):
    - `InternVL3‑78B`: MVBench 78.7, MLVU 79.5, LongVideoBench 65.7, CG‑Bench 48.4/65.3 (long/clue).
  - GUI grounding (Table 9):
    - `InternVL3‑72B`: ScreenSpot 88.7, ScreenSpot‑V2 90.9 (beats UI‑TARS‑72B: 90.3).
  - Spatial reasoning (VSI‑Bench, Table 10):
    - `InternVL3‑38B`: Overall 48.9 and strong sub-skills (e.g., Object Count 71.7; Abs. Dist. 50.2).
  - Language ability (Table 11):
    - With the same base LLM family (Qwen2.5), InternVL3 variants consistently surpass the corresponding Qwen2.5‑Chat models across many language tests; e.g., `InternVL3‑78B` vs `Qwen2.5‑72B‑Chat`: Overall 80.5 vs 78.9; MMLU 86.9 vs 84.4.

- Ablations and robustness:
  - Native multimodal pre‑training (Figure 3): replacing InternVL2‑8B’s MLP warm‑up with native multimodal pre‑training yields comparable or stronger scores pre‑SFT; and better after SFT.
  - `V2PE` (Table 12): adding `V2PE` during pre‑training improves a broad set of multimodal tasks; small `δ` values (e.g., 1/4–1/16) often work best even for shorter contexts.
  - `MPO` (Table 13): consistent reasoning gains across sizes; e.g., `+4.5` Overall for `InternVL3‑38B`.
  - Test‑time scaling (Table 2): Best‑of‑8 with VisualPRM notably boosts math/reasoning.

- Do the experiments support the claims?
  - Yes, breadth and consistency are strong:
    - SOTA or competitive on most benchmarks (Tables 2–10).
    - Clear scaling trends across sizes.
    - Controlled ablations isolate the effect of native pre‑training, `V2PE`, and `MPO` (Figure 3; Tables 12–13).
  - Nuances:
    - Visual grounding gains plateau at the very top (Table 6), with a plausible data‑coverage explanation provided in Section 3.8.
    - Some hallucination scores (e.g., MMHal, Table 5) show smaller gains or slight dips, indicating headroom for alignment.

## 6. Limitations and Trade-offs
- Assumptions and scope:
  - Modality set is vision (images, video) + text. Audio or 3D point‑cloud inputs are not primary training targets (though “3D scene understanding” appears as data domain; Sections 2.3 and 3.12).
  - The LLM and ViT are initialized from pre‑trained bases (Section 2.1). Training “entirely from scratch” is not demonstrated, though the paradigm could permit it.
- Data dependence:
  - Performance hinges on high‑quality, diverse multimodal and text corpora; the 1:3 sampling ratio (language:multimodal) and 200B token budget were tuned empirically (Section 2.2).
  - Certain domains (e.g., precise grounding) may need more targeted data; top‑end grounding slightly regresses vs. InternVL2.5‑78B (Table 6).
- Computational cost:
  - Training up to 78B parameters with long contexts is expensive. Although `InternEVO` reduces cost (50–200% speedups; Section 2.5), the total compute and memory requirements remain large.
  - Test‑time scaling (Best‑of‑N + VisualPRM) improves accuracy but increases inference compute (Table 2, “w/ VisualPRM‑Bo8”).
- Methodological trade‑offs:
  - `V2PE` introduces a new hyperparameter (`δ`) and per‑image randomness; picking `δ` at inference trades context range vs. positional resolution (Section 2.1; Table 12).
  - `MPO` requires preference/quality data and a careful balance of DPO/BCO/LM losses (Eq. 9). Gains depend on the quality of chosen/rejected samples.

## 7. Implications and Future Directions
- How this changes the landscape:
  - Demonstrates that a single‑stage, native multimodal pre‑training pipeline can replace multi‑stage post‑hoc alignment while preserving or improving language skill and multimodal performance (Figure 3; Tables 2–3, 11).
  - Establishes practical recipes—`V2PE`, `MPO`, VisualPRM‑based BoN, and `InternEVO`—that others can reuse to scale open models competitively (Sections 2.1–2.5; ablations).
- Follow‑up research enabled or suggested:
  - Extending native pre‑training to additional modalities (audio, time series, depth/LiDAR) and tasks (robotics, embodied agents).
  - Data curriculum design: automatic ratio scheduling between language and multimodal streams; targeted data to close remaining gaps (e.g., grounding).
  - Position encoding research: learnable modality‑adaptive encodings; adaptive `δ` selection at inference.
  - Preference optimization: richer process supervision across domains beyond math (e.g., spatial reasoning, GUI workflows).
  - Critic models: faster or learned‑to‑search critics that reduce Best‑of‑N compute cost; uncertainty‑aware selection.
- Practical applications:
  - Document AI and OCR pipelines (DOCVQA 95.4; OCRBench 906; Table 3).
  - Chart and diagram analytics (ChartQA 89.7; AI2D 89.7; Table 3).
  - Enterprise assistants handling multi‑image evidence and real‑world photos (BLINK 66.3; RealWorldQA 78.0; Table 4).
  - Video analytics and long video Q&A (MVBench 78.7; MLVU 79.5; LongVideoBench 65.7; Table 8).
  - GUI agents and automation (ScreenSpot‑V2 90.9; Table 9).
  - Spatial reasoning in 3D‑like scenes (VSI‑Bench Overall ~48–49; Table 10).
  - Strong pure‑language use while remaining multimodal (Table 11).

> Key headline results:
> - “InternVL3‑78B achieves 72.2 on MMMU” (Table 2).
> - “OCRBench 906; AI2D 89.7; ChartQA 89.7; DocVQA 95.4; VCR 96.0/98.6” (Table 3).
> - “MME 2549.8; MMBench EN/CN 89.0/88.7; MMVet 81.3; HallusionBench 59.1” (Table 5).
> - “MVBench 78.7; MLVU 79.5; LongVideoBench 65.7; CG‑Bench 48.4/65.3” (Table 8).
> - “ScreenSpot‑V2 90.9 at 72B; VSI‑Bench Overall up to 48.9” (Tables 9–10).
> - Language: “Overall 80.5 (InternVL3‑78B) vs 78.9 (Qwen2.5‑72B‑Chat)” with stronger MMLU, CMMLU, C‑Eval (Table 11).

In sum, InternVL3 provides a clear, reproducible blueprint—data mixture, loss design, position encoding, preference optimization, critic‑guided test‑time scaling, and scalable training—that materially advances open multimodal systems and narrows the gap to top proprietary models (Figures 1–2; Tables 2–10).
