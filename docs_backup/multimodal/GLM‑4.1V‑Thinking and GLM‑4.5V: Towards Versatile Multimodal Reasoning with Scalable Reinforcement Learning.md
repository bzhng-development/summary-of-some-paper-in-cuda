# GLM‑4.1V‑Thinking and GLM‑4.5V: Towards Versatile Multimodal Reasoning with Scalable Reinforcement Learning

**ArXiv:** [2507.01006](https://arxiv.org/abs/2507.01006)
**Authors:** GLM‑V Team, Wenyi Hong, Wenmeng Yu, Xiaotao Gu, Guo Wang, Guobing Gan, Haomiao Tang, Jiale Cheng, Ji Qi, Junhui Ji, Lihang Pan, Shuaiqi Duan, Weihan Wang, Yan Wang, Yean Cheng, Zehai He, Zhe Su, Zhen Yang, Ziyang Pan, Aohan Zeng, Baoxu Wang, Bin Chen, Boyan Shi, Changyu Pang, Chenhui Zhang, Da Yin, Fan Yang, Guoqing Chen, Jiazheng Xu, Jiale Zhu, Jiali Chen, Jing Chen, Jinhao Chen, Jinghao Lin, Jinjiang Wang, Junjie Chen, Leqi Lei, Letian Gong, Leyi Pan, Mingdao Liu, Mingzhi Zhang, Qinkai Zheng, Sheng Yang, Shi Zhong, Shiyu Huang, Shuyuan Zhao, Siyan Xue, Shangqin Tu, Shengbiao Meng, Tianshu Zhang, Tianwei Luo, Tianxiang Hao, Wenkai Li, Wei Jia, Xiao Liu, Xiaohan Zhang, Xin Lyu, Xuancheng Huang, Yanling Wang, Yadong Xue, Yanfeng Wang, Yanzi Wang, Yifan An, Yifan Du, Yiming Shi, Yiheng Huang, Yilin Niu, Yuan Wang, Yuanchang Yue, Yuchen Li, Yutao Zhang, Yuting Wang, Yu Wang, Yuxuan Zhang, Zhanxiao Du, Zhenyu Hou, Zhao Xue, Zhengxiao Du, Zihan Wang, Peng Zhang, Debing Liu, Bin Xu, Juanzi Li, Minlie Huang, Yuxiao Dong, Jie Tang
**Institutions:** 

## 🎯 Pitch

The paper presents "GLM-4.1V-Thinking" and "GLM-4.5V," a groundbreaking duo of vision-language models utilizing Reinforcement Learning with Curriculum Sampling (RLCS) and a domain-specific reward system to achieve state-of-the-art performance across 42 benchmarks. This innovation matters because it addresses the critical need for reliable, cross-domain multimodal reasoning in real-world applications, such as STEM problem-solving and GUI tasks, marking a significant leap in the capabilities of open-source models.

---

## 1. Executive Summary (2–3 sentences)
This work introduces `GLM-4.1V-Thinking` (9B dense) and `GLM-4.5V` (106B total, 12B activated MoE), a pair of vision–language models (VLMs) trained with a reasoning-centric pipeline that culminates in large-scale reinforcement learning. The core advance is Reinforcement Learning with Curriculum Sampling (RLCS) plus a domain-specific, hack-resistant reward system, which together produce state-of-the-art open-source performance across 42 benchmarks and strong generalization to diverse tasks such as STEM reasoning, GUI agents, coding from UI, grounding, long-document and video understanding (Table 2; Sections 5–6).

## 2. Context and Motivation
- Problem addressed
  - Modern VLMs must move beyond “captioning” to deep multimodal reasoning across domains (STEM, video, GUI, charts/docs, long sequences). Yet open-source models that explicitly “think” (emit chain-of-thought) have not consistently outperformed non-thinking peers of similar size across broad tasks (Introduction, p. 1–2).
  - Prior attempts to boost reasoning in VLMs using long-form reasoning and RL either focus on narrow domains or lack scalable, stable, multi-domain RL frameworks (Introduction, p. 1–2).

- Why it matters
  - Real-world agents (e.g., screen-driven assistants, web/mobile GUI agents) and high-stakes tasks (STEM problem solving, long-doc interpretation) require reliable multimodal reasoning and grounding, not just perception (Abstract; Introduction, p. 1–2). The paper targets this general-purpose capability.

- Prior approaches and gaps
  - Long-form reasoning and scalable RL improve LLM reasoning, but VLM transfer is incomplete—reward design is fragile, data coverage is uneven, and training often collapses when verifiers are hackable or noisy (Section 5.2; Figure 5).
  - SFT on short CoT can teach a style but may not scale reasoning quality; many pipelines lack standardized answer extraction, leading to noisy RL signals.

- Positioning
  - A unified pipeline: diverse multimodal pre-training, long-CoT SFT with standardized “thinking” format, and multi-domain RL using RLVR (verifiable rewards) and RLHF (preference/model rewards). The work foregrounds two mechanisms that make large-scale, cross-domain RL feasible: RLCS for difficulty-aware sampling and a robust, domain-specific reward system (Sections 4–5).
  - The models support both “thinking” and “non-thinking” modes (GLM-4.5V) for a performance/latency trade-off (Section 4.2).

## 3. Technical Approach
This section follows the training pipeline: architecture → data → pre-training/continual training → SFT → RL (algorithms, rewards, infrastructure).

- Architecture (Section 2; Figure 2)
  - Components: a ViT vision encoder (initialized from `AIMv2-Huge`), an MLP projector, and a GLM decoder LLM (`GLM-4-9B-0414` for `GLM-4.1V-Thinking`, `GLM-4.5-Air` for `GLM-4.5V`).
  - Video support:
    - Replace 2D conv with 3D conv in the ViT to enable 2× temporal downsampling (efficiency for videos). For a single image, duplicate the frame to keep shapes aligned.
    - Insert a literal “time index token” after each frame, encoding timestamps as strings. This explicitly signals frame ordering and temporal distances, improving temporal reasoning and grounding (Section 2).
  - Spatial positional encoding:
    - Add `2D-RoPE` in ViT attention to handle extreme aspect ratios and high resolutions, while preserving the ViT’s original learnable absolute positions by interpolating them to new grids with bicubic interpolation.
    - Equation (1): normalize patch grid coordinates g = (w, h) to g_norm in [−1, 1].
    - Equation (2): `Padapted(g) = Ibicubic(Porig, gnorm)`, i.e., sample from the original position table with bicubic interpolation to get per-patch embeddings.
    - Extend RoPE to `3D-RoPE` inside the LLM for stronger spatial understanding in the decoder without harming its text skills (Section 2).
  - Token-budget design: handles native image/video resolutions; examples report image token counts (e.g., 1,574; 5,187) and a 13,650-token video example (Figure 2).

- Data curation at scale (Section 3.1; Figure 4)
  - Image–text pairs (start with >10B pairs): multi-stage filtering (resolution, solid color, caption length, dedup), CLIP-similarity filter, concept-balanced resampling, then “factual-centered recaptioning” to denoise/enrich captions while retaining facts (Figure 4).
  - Interleaved web/book corpora:
    - Web (MINT, MMC4, OmniCorpus): remove ad/QR noise, enforce image–article relevance via CLIP-Score thresholding, and train a “high-knowledge-density” image classifier to prioritize charts, schematics, maps, etc. (Section 3.1).
    - Academic books: parse >100M digitized books (STEM focus), deeply extract image–text interleavings from PDFs.
  - OCR pre-training data (220M images):
    - Synthetic documents (rendered text over LAION backgrounds), natural-scene text with PaddleOCR boxes, and arXiv papers converted via LaTeXML → lightweight markup and paired with page rasterizations (Nougat-inspired) (Section 3.1).
  - Grounding data:
    - Natural images: LAION-115M with GLIP-v2 noun-phrase grounding; retain samples with ≥2 boxes, yielding 40M high-quality annotations.
    - GUI grounding: Crawl webpages, use Playwright to obtain screenshots plus visible DOM nodes with precise render-time boxes; synthesize >140M referring-expression QA pairs for GUIs (Section 3.1).
  - Video data: multi-source, human-in-the-loop annotations to capture actions and in-scene text; deduplicate by multimodal embeddings (Section 3.1).
  - Instruction tuning set (50M): covers general perception, STEM reasoning, documents, GUI agents, and UI coding; enforces taxonomy, constraint-guided synthesis for underrepresented areas, and contamination checks (Section 3.1).

- Training recipe (Section 3.2)
  - Multimodal pre-training:
    - `GLM-4.1V-Thinking`: tensor parallel size 2. `GLM-4.5V` MoE: expert parallel 8, pipeline parallel 4; “loss-free” routing with router-bias update 1e−3 and sequence-level balance loss 1e−4. Sequence length 8,192, global batch 1,536 for 120k steps. Data packing to fill sequences (Section 3.2).
  - Long-context continual training:
    - Extend to long sequences and video: increase sequence length to 32,768; add context-parallel size 4; run 10k steps at the same global batch (Section 3.2).

- Supervised Fine-Tuning (SFT) for long CoT (Section 4)
  - Purpose: align the model to produce standardized long-form reasoning and answers that are easy to evaluate in RL (not to insert new knowledge).
  - Format:
    - Use tags `<think>...</think>` for the chain-of-thought and `<answer>...</answer>` for the concise final. For verifiable tasks, require a single answer span inside `<|begin_of_box|>...<|end_of_box|>` to simplify extraction; these markers are added as special tokenizer tokens (Section 4.1).
    - `GLM-4.5V` supports a “non-thinking” mode by appending `/nothink` to the prompt; the model then returns an empty `<think>` and only the answer (Section 4.2).
  - Data quality controls: strict formatting checks, removal of mixed-language/redundant thought, and iterative inclusion of high-quality RL rollouts back into SFT to stabilize later RL (Section 4.1).

- Reinforcement Learning (RL) at scale (Section 5)
  - Terminology
    - `RLVR` (Reinforcement Learning with Verifiable Rewards): reward is computed by comparing a parsed final answer to a ground-truth answer with robust rules (binary or continuous).
    - `RLHF` (Reinforcement Learning from Human Feedback): reward from a learned reward model that scores answer quality.
    - `GRPO` objective: an RL objective similar in spirit to group relative preference optimization used in recent reasoning LLMs (Section 5.3).
  - Reward extraction and boxing (Section 5.2; Table 1)
    - To avoid errors in LLM-based answer extraction across open-domain tasks, the rollout must wrap the final answer in `<|begin_of_box|>...<|end_of_box|>`. The reward logic reads only that span.
    - “Reward hacking” is explicitly handled: e.g., vague answers like “a number between 0 and 10” were found to fool naïve verifiers; the system adds domain checks to prevent such exploits (Section 5.2).
    - Domain-specific verifiers (Table 1) with shared utilities (format checks, exact match) and per-domain logic:
      - Math/Physics/Chemistry: symbolic numeric matching via SymPy with tolerances; unit-aware LLM judging when needed.
      - Chart QA: numeric vs textual branches (tolerant numeric match vs exact/LLM-semantic).
      - OCR: edit-distance based continuous reward.
      - Grounding/GUI: IoU-based rewards; GUI action prediction includes action correctness plus IoU.
      - Long docs/Geo/spatial/video: exact or LLM semantic equivalence as appropriate.
    - Format/style rewards: penalize spurious boxing in non-verifiable data, mixed-language blocks, or repetitive text (Section 5.2).
  - RLCS (Reinforcement Learning with Curriculum Sampling) (Section 5.3)
    - Problem: as the model improves, many samples become trivial; GRPO yields no gradient when all rollouts in a batch are correct/incorrect.
    - Solution: combine offline difficulty labels (pass@k across models + human labels) with online difficulty estimates from current rollouts. Re-weight sampling per iteration to emphasize “mid-range” difficulty items that yield the most learning signal (Section 5.3).
    - Dynamic sampling expansion via ratio EMA (Section 5.3.1):
      - Compute `not_valid_sample_rate` in the last iteration (fraction of all-correct or all-incorrect prompts). Set `expansion_ratio = 1/(1 − not_valid_sample_rate)` and update `expansion_ratio_ema` (exponential moving average). Roll out more samples accordingly and then subsample to a balanced batch (close to half-correct/half-incorrect), restoring effective batch size and stability.
    - Other stabilizers and effectiveness tweaks (Section 5.3.1–5.3.2):
      - Larger batch sizes improve long-run performance ceilings.
      - “Force answering”: if thinking runs long and hits length limits, inject `</think>` to compel an answer so the sample can still receive a reward and teach “budgeted thinking” (Section 5.3.1; [64]).
      - Remove KL and entropy losses (both harmed stability/performance in multimodal RL); increase the PPO-style upper clipping bound (“clip-higher”) to help off-policy stability (Section 5.3.1–5.3.2).
      - Use `top-p = 1` in rollouts to avoid late-iteration “garbling,” likely by maintaining coverage over rare tokens (Section 5.3.2).
      - Prefer per-sample loss aggregation for steadier training (Section 5.3.2).
  - RL infrastructure (Section 5.4)
    - DP-rank length balancing (avoid stragglers when some ranks get many long sequences).
    - Sequence packing + gradient accumulation with fixed context length (32k); repack samples within ranks to minimize micro-steps (halves F/B time in practice).
    - Parallel-friendly implementation of the ratio-EMA oversampling.

## 4. Key Insights and Innovations
- RLCS: difficulty-aware online curriculum for RL across domains (Section 5.3)
  - What’s new: fuses offline difficulty labels with online pass@k to reweight sampling per iteration, plus a practical ratio-EMA oversampling scheme that anticipates how many rollouts to run in parallel before picking a balanced subset (Section 5.3.1).
  - Why it matters: raises sample efficiency where rollouts dominate cost; keeps training in the “learning sweet spot,” yielding consistent gains and faster improvements.

- Domain-specific, hack-resistant reward system with boxed answer spans (Section 5.2; Table 1)
  - What’s new: standardized `<|begin_of_box|>...<|end_of_box|>` spans; shared format checks; carefully engineered per-domain verifiers (e.g., tolerance-based numeric match, IoU for grounding/GUI, edit-distance for OCR).
  - Why it matters: Section 5.2 and Figure 5 show that a single weak verifier (e.g., multi-image QA) can collapse cross-domain RL—rewards must be precise across all domains.

- Architectural adaptations for robust spatial–temporal grounding (Section 2; Equations (1)–(2))
  - 2D-RoPE + preserved absolute positions via bicubic interpolation handle extreme aspect ratios and very high resolutions; 3D-RoPE in the LLM boosts spatial reasoning; temporal index tokens encode real-time distances between frames for video.

- “Thinking” and “non-thinking” dual-mode training (Section 4.2)
  - What’s new: `GLM-4.5V` can switch modes at inference by adding `/nothink` to the prompt; SFT trains both styles together, making it easy to trade off speed vs. peak performance.
  - Why it matters: in some tasks like OCR, non-thinking can be faster and even slightly better, while complex reasoning benefits from explicit thinking (Table 2 shows OCRBench: 87.2 non-thinking vs 86.5 thinking).

- Cross-domain RL generalization (Section 6.3; Figure 6)
  - Observation: training in one domain (e.g., STEM or GUI) often boosts others (e.g., grounding, VQA). Mixed-domain training (“Mix All”) yields the largest average gains in 3 of 5 categories, demonstrating mutual reinforcement.

## 5. Experimental Analysis
- Evaluation setup (Section 6.1; Appendix B–C)
  - 42 public benchmarks across 8 categories (General VQA, STEM, OCR/Chart/LongDoc, Visual Grounding, Spatial Reasoning, GUI Agents, Coding, Video).
  - Inference:
    - Max output length 8,192 tokens; image inputs capped at 6,144 tokens; video up to 48,000 tokens. For scoring/extraction tasks (e.g., judge models in coding/UI), GPT-4o (2024-11-20) is used consistently across all compared models. At least 95% successful request rate per benchmark is enforced.
  - Answer extraction:
    - For verifiable tasks, the final answer is the content inside `<|begin_of_box|>...<|end_of_box|>` (Section 6.1).

- Main quantitative results (Table 2)
  - Headline: `GLM-4.5V` (thinking mode) sets state-of-the-art among similarly sized open-source models in nearly all tasks; the compact `GLM-4.1V-9B-Thinking` beats much larger `Qwen2.5-VL-72B` on 29/42 benchmarks (Abstract; Section 6.2).
  - General VQA:
    > MMStar: 75.3 (GLM-4.5V thinking) vs 70.8 (Qwen2.5-VL-72B), 69.0 (Step-3 321B-A38B) (Table 2).  
    > MMBench-EN: 88.2 vs 88.0 (Qwen2.5-VL-72B) and 81.1 (Step-3) (Table 2).
  - STEM:
    > MMMU-Pro: 65.2 vs 51.1 (Qwen2.5-VL-72B) and 58.6 (Step-3) (Table 2).  
    > MathVista: 84.6 vs 74.8 (Qwen2.5-VL-72B) (Table 2).
  - Long documents, OCR & Charts:
    > MMLongBench-Doc: 44.7 vs 35.2 (Qwen2.5-VL-72B) and 31.8 (Step-3) (Table 2).  
    > ChartMuseum: 55.3 vs 39.6 (Qwen2.5-VL-72B) and 40.0 (Step-3) (Table 2).  
    > OCRBench: 86.5 (thinking) and 87.2 (non-thinking) vs 85.1 (Qwen2.5-VL-72B) (Table 2).
  - Visual Grounding and Spatial Reasoning:
    > RefCOCO-avg (val): 91.3 (GLM-4.5V thinking) vs 90.3 (Qwen2.5-VL-72B), while Step-3 reports 20.2 under their evaluation setting (Table 2).  
    > OminiSpatial: 51.0 (GLM-4.5V thinking) vs 47.9 (Qwen2.5-VL-72B) (Table 2).
  - GUI Agents (100-step budget for OSWorld; Section 6.1):
    > OSWorld: 35.8 vs 8.8 (Qwen2.5-VL-72B).  
    > AndroidWorld: 57.0 matches the best open-source baseline shown (Table 2).  
    > WebVoyager Some: 84.4 vs 40.4 (Qwen2.5-VL-72B) (Table 2).
  - Coding (VLM coding):
    > Design2Code: 82.2 vs 41.9 (Qwen2.5-VL-72B) and 34.1 (Step-3) (Table 2).  
    > Flame-React-Eval: 82.5 vs 46.3 (Qwen2.5-VL-72B) and 63.8 (Step-3) (Table 2).
  - Video:
    > VideoMMMU: 72.4 vs 60.2 (Qwen2.5-VL-72B).  
    > MMVU: 68.7 vs 62.9 (Qwen2.5-VL-72B) (Table 2).
  - RL gains:
    > Figure 1B reports that RL boosts performance by “up to +10.6%” over the SFT baseline when applied to GLM-4.5V.

- Ablations, diagnostics, and stability evidence
  - Reward quality matters: Figure 5 shows how a single flawed verifier (e.g., multi-image QA) can cause “reward noise” or “reward hacking” and then stall or regress other domains (MMMU, MathVista, AI2D). This directly supports the emphasis on hack-resistant verifiers (Section 5.2).
  - RLCS and cross-domain effects: Figure 6 demonstrates that single-domain RL often improves scores in other domains; “Mix All” further amplifies gains across STEM, OCR/Chart, and General VQA, with smaller or no gains in grounding and GUI in that specific study (Section 6.3).
  - Pre-training base quality: Figure 3 shows `GLM-4.1V-9B-Base` strongly outperforming a similar-scale base model (InternVL3-9B-Pretrain) in pass@k on non-MC MathVista samples—this “upper bound” perspective helps explain strong post-RL results (Section 3).

- Do the experiments support the claims?
  - Breadth and consistency: The 42-benchmark suite covers all claimed domains. Table 2 presents side-by-side comparisons with strong baselines (Step-3, Qwen2.5-VL-72B) and shows large margins in STEM, coding, GUI agents, charts/docs, and video. The mix of thinking/non-thinking results is transparent (e.g., OCRBench).
  - Mechanism–result links: Evidence in Figure 5 (reward failures) and Figure 6 (cross-domain transfer) connects the training design (reward rigor, RLCS) to observed stability and generalization.

## 6. Limitations and Trade-offs
- Reward scope and reasoning faithfulness (Section 7)
  - RL rewards mostly target final answers. Models sometimes reach correct answers with flawed intermediate reasoning; current verifiers do not check step-by-step chains. This risks reinforcing spurious chains if outcomes are correct.
- Stability sensitivity (Section 7; Sections 5.3.1–5.3.2)
  - Early versions were fragile; training remained sensitive to setup choices (e.g., entropy/KL terms, sampler decisions). The final recipe removes KL/entropy losses and sets `top-p=1` to avoid “garbling,” but this may introduce more sampling variance.
- Coverage vs. specialization (Figure 6)
  - Mixed-domain RL (“Mix All”) did not improve grounding or GUI in the reported cross-domain study—suggesting that some skills benefit from domain-specific emphasis even when broad gains occur elsewhere.
- Compute and engineering demands (Sections 3.2, 5.4)
  - Large-scale pre-training (120k steps, seq len 8,192, batch 1,536) and long-context continual training (32,768 tokens) are compute-heavy. The RL infrastructure (DP balancing, packing, ratio-EMA oversampling) is non-trivial to reproduce.
- Evaluation dependencies (Section 6.1)
  - For some tasks (e.g., Design2Code), external judge models (GPT-4o-mini in Appendix B.1) are used to assess similarity. While the protocol is the same across models, reliance on a particular judge can bias results if it systematically favors certain styles.

## 7. Implications and Future Directions
- Field impact
  - Demonstrates that multi-domain RL with robust, domain-specific verifiers and difficulty-aware sampling can lift VLM reasoning broadly, not just within a single silo. The open release of `GLM-4.1V-9B-Thinking`, `GLM-4.1V-9B-Base`, and `GLM-4.5V` plus reward modules offers a practical foundation for community research (Abstract; Section 1).
  - The dual-mode “thinking/non-thinking” interface in `GLM-4.5V` provides a pragmatic knob for latency/quality trade-offs in production systems (Section 4.2).

- Follow-up research enabled or suggested
  - Process-level rewards: build verifiers that check intermediate steps (mathematical derivations, logic chains, grounding traceability) to discourage shortcutting and hallucinated chains (Section 7).
  - Reward robustness: broaden anti-hacking defenses (e.g., adversarial prompting against verifiers; consistency checks across paraphrases; robust units/dimensions checking in STEM).
  - Curriculum learning: extend RLCS with more granular on-policy competence estimates (e.g., per-skill “mastery profiles”) and adaptive horizon/temperature control.
  - Cross-modal transfer: the paper hypothesizes that visual reasoning might improve text-only tasks (e.g., “reading code from images” aiding coding), worth systematic study (Section 7).

- Practical applications
  - GUI agents and RPA: Table 2’s large gains on OSWorld, WebVoyager, and WebQuest support practical agents for desktop/mobile/web automation and accessibility.
  - Enterprise analytics: strong Chart/Doc/OCR scores (ChartMuseum, MMLongBench-Doc, OCRBench) suggest applicability to reporting, document QA, and knowledge extraction.
  - STEM and education: high MMMU/MathVista/WeMath scores enable tutoring and reasoning assistants with visual diagrams.
  - Video understanding: improved scores on VideoMMMU/MMVU/MVBench enable video QA, surveillance summarization, and content moderation.

> Reinforcement-learning-centric training with precise verifiers and adaptive curricula—backed by robust engineering (Section 5.4)—is the mechanism that turns strong pre-training into broadly capable multimodal reasoning. The paper’s experiments (Table 2, Figures 5–6) show that when reward quality and sampling are handled carefully, cross-domain VLM reasoning can scale reliably rather than fracture across tasks.
