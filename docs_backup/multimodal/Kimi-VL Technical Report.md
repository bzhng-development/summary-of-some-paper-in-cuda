# Kimi-VL Technical Report

**ArXiv:** [2504.07491](https://arxiv.org/abs/2504.07491)
**Authors:** Kimi Team, Angang Du, Bohong Yin, Bowei Xing, Bowen Qu, Bowen Wang, Cheng Chen, Chenlin Zhang, Chenzhuang Du, Chu Wei, Congcong Wang, Dehao Zhang, Dikang Du, Dongliang Wang, Enming Yuan, Enzhe Lu, Fang Li, Flood Sung, Guangda Wei, Guokun Lai, Han Zhu, Hao Ding, Hao Hu, Hao Yang, Hao Zhang, Haoning Wu, Haotian Yao, Haoyu Lu, Heng Wang, Hongcheng Gao, Huabin Zheng, Jiaming Li, Jianlin Su, Jianzhou Wang, Jiaqi Deng, Jiezhong Qiu, Jin Xie, Jinhong Wang, Jingyuan Liu, Junjie Yan, Kun Ouyang, Liang Chen, Lin Sui, Longhui Yu, Mengfan Dong, Mengnan Dong, Nuo Xu, Pengyu Cheng, Qizheng Gu, Runjie Zhou, Shaowei Liu, Sihan Cao, Tao Yu, Tianhui Song, Tongtong Bai, Wei Song, Weiran He, Weixiao Huang, Weixin Xu, Xiaokun Yuan, Xingcheng Yao, Xingzhe Wu, Xinxing Zu, Xinyu Zhou, Xinyuan Wang, Y. Charles, Yan Zhong, Yang Li, Yangyang Hu, Yanru Chen, Yejie Wang, Yibo Liu, Yibo Miao, Yidao Qin, Yimin Chen, Yiping Bao, Yiqin Wang, Yongsheng Kang, Yuanxin Liu, Yulun Du, Yuxin Wu, Yuzhi Wang, Yuzi Yan, Zaida Zhou, Zhaowei Li, Zhejun Jiang, Zheng Zhang, Zhilin Yang, Zhiqi Huang, Zihao Huang, Zijia Zhao, Ziwei Chen, Zongyu Lin
**Institutions:** Kimi Team

## 🎯 Pitch

Kimi-VL introduces a groundbreaking vision-language model utilizing a Mixture-of-Experts language decoder combined with a high-resolution vision encoder, enabling efficient multimodal reasoning and long-context understanding with minimal computational activation. This innovation significantly narrows the gap between open-source and proprietary models, offering cost-effective, accessible solutions for complex tasks like high-res UI understanding and long-document/video comprehension, driving advancements in areas such as enterprise document intelligence and video analytics.

---

## 1. Executive Summary
Kimi-VL is an efficient, open-source vision–language model (VLM) that combines a Mixture-of-Experts (MoE) language decoder with a native‑resolution vision encoder to deliver strong multimodal reasoning, long‑context understanding (up to 128K tokens), and GUI agent capabilities while activating only about 3B parameters at inference (Section 1; Figure 3). A long‑thinking variant, `Kimi‑VL‑A3B‑Thinking‑2506`, adds chain‑of‑thought (CoT) supervised fine‑tuning and reinforcement learning (RL) to reach state‑of‑the‑art performance among open‑source efficient VLMs on several reasoning and video benchmarks (Section 4.2; Table 4; Table 5).

## 2. Context and Motivation
- Problem gap:
  - Open-source VLMs have lagged behind proprietary models (e.g., GPT‑4o, Gemini) in three areas: efficiency at scale, long‑context processing, and advanced “long thinking” (multi‑step CoT) reasoning (Section 1).
  - Existing open VLMs:
    - Dense models (e.g., `Qwen2.5‑VL‑7B`, `Gemma‑3‑12B‑IT`) are competitive but compute‑heavier and generally lack long‑CoT reasoning out of the box (Section 1).
    - Early MoE VLMs (e.g., `DeepSeek‑VL2`, `Aria`) show promise but have key limits: fixed‑size vision encoders, short context (4K), weaker fine‑grained perception, and no long‑thinking support (Section 1).
- Why this matters:
  - Real use cases require: high‑resolution OCR and UI understanding, multi‑image/video reasoning, and long‑document/video comprehension (e.g., enterprise documents, software agents). Achieving this with small activated compute is crucial for accessibility, cost, and latency.
- Positioning:
  - Kimi‑VL integrates three fronts in one open model: an MoE text decoder for compute‑efficiency, a native‑resolution vision encoder for high‑fidelity perception, and a training recipe enabling 128K multimodal context plus long‑CoT reasoning (Figures 3–5; Figure 4 on pre‑training stages; Section 2.3–2.4).

## 3. Technical Approach
Step‑by‑step overview (with selective definitions of nonstandard terms):

- Core architecture (Figure 3; Section 2.1):
  - `MoonViT` (vision encoder):
    - Native‑resolution processing: images are patchified, flattened, and concatenated into variable‑length 1D sequences so the model can “pack” different sizes efficiently (NaViT‑style packing) without splitting images into tiles (Section 2.1, “MoonViT”).
    - Spatial encoding: combines interpolated absolute positional embeddings from SigLIP with 2D RoPE (rotary position embeddings) along height and width to preserve fine‑grained spatial relations at high resolution (Section 2.1).
      - Definition: `2D RoPE` rotates query/key vectors as a function of 2D position, improving generalization to large images compared with fixed embeddings.
    - In the `2506` thinking variant, MoonViT is continually trained to encode up to ~3.2M pixels per image (4× the prior limit), enabling ultra‑high‑resolution perception (Section 2.1).
  - `MLP projector`:
    - Uses a `pixel shuffle` step (2×2 spatial downsampling in exchange for more channels) followed by a two‑layer MLP to map vision features to the language model embedding space (Section 2.1).
  - `Moonlight` MoE language decoder:
    - An MoE LLM with 2.8B activated and 16B total parameters (Section 2.1). “Activated parameters” are the subset of expert weights used per token (typical in MoE routing), so inference cost follows ~3B parameters instead of the full 16B.
    - Architecture similar to `DeepSeek‑V3` (non‑shared and shared experts plus a router); initialized from a 5.2T‑token text‑pretrained checkpoint with 8K context (Section 2.1).

- Optimization and scaling (Section 2.2; 2.5):
  - `Enhanced Muon` optimizer with weight decay and per‑parameter update scaling, implemented in a memory‑efficient ZeRO‑1 style (Section 2.2).
  - 4D parallelism for throughput and long sequences: Data Parallelism, Expert Parallelism, Pipeline Parallelism, and Context Parallelism (the latter splits the sequence length across devices to train 128K efficiently with FlashAttention) (Section 2.5).
  - Additional memory tactics: Selective activation checkpointing and ZeRO‑1 optimizer state sharding; recomputation increases for ultra‑long sequences (Section 2.5).
  - Reported training throughput: ~60% higher than a 7B dense VLM baseline after parallelism optimization (Section 2.5).

- Multi‑stage pre‑training to preserve text ability while adding vision and long context (Figure 4; Table 1; Section 2.3):
  - Stage A: ViT training (2.0T tokens + 0.1T alignment)
    - CoCa‑like objective: contrastive `SigLIP` loss plus caption generation loss (weights L = L_siglip + 2·L_caption). A tiny text decoder is used here only; later alignment swaps to the MoE LLM (Section 2.3 “ViT Training Stages”).
    - Progressive image resolution sampling; observation that OCR skill emerges in the caption decoder as OCR data scales (Section 2.3).
    - Alignment step (0.1T) updates only MoonViT+projector to reduce perplexity when feeding vision tokens into the MoE LLM (Section 2.3).
  - Stage B: Joint pre‑training (1.4T tokens)
    - Mixes the original text corpus and diverse multimodal data; gradually increases the multimodal ratio to preserve language skill while learning vision grounding (Section 2.3 “Joint Pre‑training Stage”).
  - Stage C: Joint cooldown (0.6T tokens)
    - High‑quality text and multimodal data, augmented with verified synthetic QA pairs in math, knowledge, and code to sharpen capabilities (with rejection sampling) (Section 2.3 “Joint Cooldown Stage”).
    - Visual QA pairs are kept at a low ratio to avoid overfitting to QA patterns, serving to “activate” capabilities needed to learn from higher‑quality data (Section 2.3).
  - Stage D: Joint long‑context activation (0.3T tokens)
    - Extends context from 8K → 32K → 128K by resetting RoPE base from 50,000 to 800,000 and running two sub‑stages (each 4× length increase) (Section 2.3).
    - 25% of tokens are “long data” (long text, long interleaved image–text, long video, long document), 75% replay shorter data to retain short‑context skills (Section 2.3).
    - Validation: “Needle in a Haystack” (NIAH) recall is near 100% up to 64K for both text and video haystacks, and 87–92% at 128K (Table 2).

- Post‑training (Figure 5; Section 2.4):
  - Instruction SFT:
    - 1 epoch at 32K, then 1 epoch at 128K, mixing pure text and multimodal chat data in ChatML format; supervision on answers and special tokens only; format‑aware packing preserves dialogue structure and cross‑modal alignment (Section 2.4 “Joint SFT”).
  - Long‑CoT SFT (thinking warm‑start):
    - A small, high‑quality multimodal CoT set is created via prompt‑engineered sampling and verification, explicitly teaching “planning, evaluation, reflection, exploration” reasoning patterns (Section 2.4 “Long‑CoT SFT”).
  - RL for reasoning:
    - Online `policy mirror descent` with a 0/1 reward on answer correctness and a KL regularization term to stabilize updates (Equation (1); Section 2.4 “Reinforcement Learning”).
      - Plain‑language: the model generates answers, receives a binary correctness reward, and its policy is nudged toward outputs that earned higher reward while staying close to the current policy (the KL term).
    - Length penalty discourages unnecessary long CoTs (“overthinking”); difficulty‑aware curriculum and prioritized sampling focus compute where it teaches most (Section 2.4).

- Data construction highlights (Section 3):
  - Six multimodal categories curated at scale: `caption`, `interleaving`, `OCR`, `knowledge`, `video`, `agent` (Section 3.1).
  - OCR data includes multi‑page documents, figures/tables/diagrams, heavy augmentations; supports long‑document OCR and layout understanding (Section 3.1 “OCR Data”).
  - Agent data gathered from large‑scale virtualized environments with dense grounding labels and multi‑step trajectories (Section 3.1 “Agent Data”).

## 4. Key Insights and Innovations
- Efficient MoE VLM with only ~3B activated parameters for strong, general multimodal ability (Figure 3; Table 3).
  - Why it matters: Competes with or surpasses larger dense VLMs while lowering inference cost. For example, on InfoVQA (OCR), Kimi‑VL scores 83.2 vs GPT‑4o at 80.7 and Qwen2.5‑VL‑7B at 82.6, with a fraction of activated parameters (Table 3).
  - Distinction: Prior open VLMs either use dense decoders or earlier MoE designs with short context or weaker perception.

- Native‑resolution `MoonViT` with 2D RoPE + NaViT‑style packing for high‑res, variable‑aspect inputs (Section 2.1; Figure 3).
  - Why it matters: Avoids tiling, preserves fine spatial cues across huge images and UI screenshots. The `2506` variant expands to ~3.2M pixels per image and posts large gains in high‑res/OS UI tasks (Table 5: ScreenSpot‑Pro 52.8 and OSWorld‑G 52.5).

- Long‑context activation across modalities with demonstrated retrieval up to 128K (Section 2.3; Table 2).
  - Why it matters: Long PDFs/videos and interleaved sequences are first‑class citizens. The data mix and staged extension (replaying short data) maintain short‑context quality while enabling long‑form reasoning; this is nontrivial and often brittle without such recipes.

- Integrated long‑thinking via CoT SFT + online RL that both improves accuracy and reduces CoT length (Figure 5; Table 4).
  - Why it matters: `Kimi‑VL‑A3B‑Thinking‑2506` lifts reasoning benchmarks substantially (e.g., MathVision 56.9 and MathVista 80.1) while cutting average output tokens by ~20% on MMMU‑val and MathVision (Section 4.3), improving latency and cost. Figure 13 shows test‑time scaling: more thinking tokens → better accuracy up to a point.

These are fundamental advances in training recipe and system design rather than only incremental tuning.

## 5. Experimental Analysis
- Evaluation setup (Sections 4, B):
  - Breadth: college‑level (MMMU/MMVU), general VLM (MMBench, MMVet, MMStar, RealWorldQA, AI2D), math (MathVista, MathVision), multi‑image (BLINK), OCR (InfoVQA, OCRBench), long document (MMLongBench‑Doc), long video (Video‑MME, MLVU, LongVideoBench), video perception (EgoSchema, VSI‑Bench, TOMATO), and agent/GUI grounding (ScreenSpot‑V2/Pro, OSWorld, WindowsAgentArena).
  - Metrics: accuracy or Pass@1 for MCQ/VQA; OCRBench out of 1000; with/without subtitles for Video‑MME to isolate frame understanding from textual leakage.
  - Baselines: GPT‑4o/‑mini (numbers shown for context; GPT‑4o appears grayed in Table 3), Qwen2.5‑VL‑7B, Gemma‑3‑12B‑IT, Llama‑3.2‑11B‑Instruct, DeepSeek‑VL2. Some competitor entries are unavailable where models cannot handle task context lengths (Table 3).

- Main results for `Kimi‑VL‑A3B` (Instruct) (Table 3; Figure 2):
  - General:
    - “MMBench‑EN‑v1.1”: 83.1, matching GPT‑4o and ahead of other open baselines.
    - “RealWorldQA”: 68.1—near Qwen2.5‑VL‑7B (68.5), above Gemma‑3‑12B‑IT (59.1).
  - Math:
    - “MathVista”: 68.7, better than GPT‑4o (63.8), Qwen2.5‑VL‑7B (68.2).
    - “MathVision”: 21.4 (lower; later improved by thinking variants).
  - OCR:
    - “InfoVQA”: 83.2—tops GPT‑4o (80.7), DeepSeek‑VL2 (78.1); “OCRBench”: 867/1000 (Table 3).
  - Long context:
    - “MMLongBench‑Doc”: 35.1—above Qwen2.5‑VL‑7B (29.6) and GPT‑4o‑mini (29.0), though below GPT‑4o (42.8).
    - “LongVideoBench”: 64.5—near GPT‑4o (66.7).
  - Video:
    - “Video‑MME” w/o sub: 67.8 (strong without subtitles); with sub: 72.6 (Table 3).
    - MLVU MCQ: 74.2 (SoTA vs GPT‑4o 64.6; Qwen2.5‑VL‑7B 70.2).
    - EgoSchema: 78.5 vs GPT‑4o 72.2.
  - Multi‑image:
    - BLINK: 57.3—above Qwen2.5‑VL‑7B (56.4), GPT‑4o‑mini (53.6).
  - Agent:
    - “ScreenSpot‑V2”: 92.8; “ScreenSpot‑Pro”: 34.5; “OSWorld” Pass@1: 8.22 vs GPT‑4o 5.03; “WindowsAgentArena”: 10.4 vs GPT‑4o 9.4 (Table 3).

- Reasoning variants (Table 4; Figure 13):
  - `Kimi‑VL‑A3B‑Thinking`:
    - Inference‑time longer CoT boosts accuracy (Figure 13). At 16k thinking tokens, MathVision rises to 36.8, MMMU to 61.7, MathVista to 71.3.
  - `Kimi‑VL‑A3B‑Thinking‑2506` (integrated thinking model):
    - MathVision 56.9, MathVista 80.1, MMMU 64.0, MMMU‑Pro 46.3, VideoMMMU 65.2 (Table 4).
    - General/perception retained or improved (Table 5): MMBench 84.4; MMVet 78.1; RealWorldQA 70.0; MMStar 70.4; Video‑MME with sub 71.9; MMLongBench‑Doc 42.1 (matching GPT‑4o’s 42.8, and +10 points over the earlier thinking model).
    - High‑res agent grounding: ScreenSpot‑Pro 52.8; OSWorld‑G 52.5; ScreenSpot‑V2 91.4 (Table 5).

- Qualitative evidence:
  - Examples include multi‑image spatial reasoning, video game scene recognition, landmark identification, long‑document OCR (Figure 7; Figure 9), step‑by‑step GUI actions (Figure 10), long video scene segmentation (Figure 11).
  - There is also a demo of author inference from historical manuscripts (Figure 6). Note that identifying specific real persons from images can be sensitive; the figure is presented in the paper as a qualitative demonstration only.

- Do the experiments support the claims?
  - Breadth and consistency are strong: across OCR, multi‑image, long‑video/document, and agent tasks, `Kimi‑VL‑A3B` is competitive with models 2–4× larger; the `2506` variant demonstrably improves reasoning while retaining perception and long‑context performance (Table 3, Table 4, Table 5).
  - Particularly convincing:
    - Long‑video/document capability with 128K context (Table 2; Table 3; Table 5).
    - High‑resolution UI grounding and OS agents (Table 3; Table 5).
    - Test‑time CoT scaling behavior (Figure 13).
  - Less explored or missing:
    - Limited ablations isolating the effects of 2D RoPE, alignment stage, cooldown synthetic data, and long‑context composition.
    - No detailed failure analysis; qualitative examples are curated.

- Trade‑offs and conditional results:
  - The base instruct model underperforms proprietary larger models on some academic reasoning (e.g., GPT‑4o MMMU 69.1 vs Kimi‑VL 57.0; Table 3), while the thinking variant narrows or reverses gaps on reasoning‑heavy tests but increases generation cost unless prompt‑level controls are used (Figure 13, Section 4.3 on token reductions in `2506`).
  - With subtitles, video accuracy increases for most models; evaluating without subtitles (Table 3) is a fairer test of spatiotemporal perception.

## 6. Limitations and Trade-offs
- Capacity vs. breadth:
  - Only ~2.8B activated parameters in the decoder (plus ~0.4B in vision). While efficient, this caps raw language capacity, which may limit extremely specialized or knowledge‑heavy tasks (Section 5 “Conclusion, Limitation, and Future Work”, point 1).
- Long‑context constraints:
  - Although context is 128K, attention capacity still corresponds to a ~3B model, so performance on very long and complex sequences may lag larger decoders (Conclusion, point 3).
- Reasoning supervision and RL:
  - Rewards are binary (correct/incorrect), which may insufficiently capture reasoning quality; risk of spurious but correct answers being reinforced (Section 2.4).
  - CoT SFT and RL depend on synthetic and prompt‑engineered data; coverage and bias depend on generation quality and rejection sampling efficacy (Section 2.4; Section 3.3).
- Evaluation coverage:
  - Extensive, but with few ablations and limited error analysis. Some competitor numbers are absent due to context/ability limits, complicating perfect apples‑to‑apples comparisons (Table 3).
- Practical compute costs:
  - While more efficient than dense peers, long‑context training/inference and long CoTs still incur real costs. The `2506` token‑length reductions help (~20% shorter answers), but practitioners must tune thinking depth (Figure 13; Section 4.3).

## 7. Implications and Future Directions
- Field impact:
  - Establishes a strong template for efficient multimodal systems: MoE text decoding + native‑resolution vision + staged long‑context training. Demonstrates that a ~3B‑activated VLM can be competitive across OCR, high‑res UI grounding, long videos/docs, and—with CoT+RL—hard reasoning (Figures 2–5; Tables 3–5).
- Enabled research directions:
  - MoE‑centric VLM scaling: explore larger total parameters while keeping low activated compute; study expert specialization across modalities and context lengths.
  - Long‑context learning: ablations on RoPE scaling, data composition, and cross‑modal ordering to further improve 128K+ stability and retrieval.
  - Reasoning training: richer reward models (step‑level, structure‑aware), verifiable tool‑use, and hybrid search (the model already encodes planning/evaluation/reflection patterns; Figure 5).
  - High‑resolution perception: leverage the 3.2M‑pixel capability for CAD/medical/doc layouts and professional GUI agents (Table 5 improvements on ScreenSpot‑Pro and OSWorld‑G).
- Practical applications:
  - Enterprise document intelligence (OCR, table/form understanding, long‑document Q&A).
  - Video analytics at scale (surveillance summaries, sports highlights, instructional video comprehension).
  - Software agents for desktop/web/mobile automation with robust grounding and multi‑step plans (Figure 10; Table 3; Table 5).
  - Education and STEM assistance (math visual reasoning; Table 4).

> Representative results to remember:
> - Efficiency with reach: “MMBench‑EN‑v1.1 = 83.1” for Kimi‑VL (Table 3), matching GPT‑4o, and “InfoVQA = 83.2” (OCR).
> - Long‑form understanding: “MMLongBench‑Doc = 42.1” for `Thinking‑2506`, matching GPT‑4o’s 42.8 (Table 5).
> - Reasoning with small compute: `Thinking‑2506` hits “MathVista = 80.1”, “MathVision = 56.9”, and “VideoMMMU = 65.2” (Table 4), with only ~3B activated parameters.

Overall, Kimi‑VL and `Kimi‑VL‑A3B‑Thinking‑2506` show that careful architectural choices plus a multi‑stage multimodal training recipe can deliver a broadly capable, long‑context, and reasoning‑enabled VLM at low activated compute—pushing open‑source systems closer to proprietary leaders while making them more practical to deploy.
