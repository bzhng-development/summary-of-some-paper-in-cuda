# Intern‑S1: A Scientific Multimodal Foundation Model

**ArXiv:** [2508.15763](https://arxiv.org/abs/2508.15763)
**Authors:** Lei Bai, Zhongrui Cai, Maosong Cao, Weihan Cao, Chiyu Chen, Haojiong Chen, Kai Chen, Pengcheng Chen, Ying Chen, Yongkang Chen, Yu Cheng, Yu Cheng, Pei Chu, Tao Chu, Erfei Cui, Ganqu Cui, Long Cui, Ziyun Cui, Nianchen Deng, Ning Ding, Nanqin Dong, Peijie Dong, Shihan Dou, Sinan Du, Haodong Duan, Caihua Fan, Ben Gao, Changjiang Gao, Jianfei Gao, Songyang Gao, Yang Gao, Zhangwei Gao, Jiaye Ge, Qiming Ge, Lixin Gu, Yuzhe Gu, Aijia Guo, Qipeng Guo, Xu Guo, Conghui He, Junjun He, Yili Hong, Siyuan Hou, Caiyu Hu, Hanglei Hu, Jucheng Hu, Ming Hu, Zhouqi Hua, Haian Huang, Junhao Huang, Xu Huang, Zixian Huang, Zhe Jiang, Lingkai Kong, Linyang Li, Peiji Li, Pengze Li, Shuaibin Li, Tianbin Li, Wei Li, Yuqiang Li, Dahua Lin, Junyao Lin, Tianyi Lin, Zhishan Lin, Hongwei Liu, Jiangning Liu, Jiyao Liu, Junnan Liu, Kai Liu, Kaiwen Liu, Kuikun Liu, Shichun Liu, Shudong Liu, Wei Liu, Xinyao Liu, Yuhong Liu, Zhan Liu, Yinquan Lu, Haijun Lv, Hongxia Lv, Huijie Lv, Qidang Lv, Ying Lv, Chengqi Lyu, Chenglong Ma, Jianpeng Ma, Ren Ma, Runmin Ma, Runyuan Ma, Xinzhu Ma, Yichuan Ma, Zihan Ma, Sixuan Mi, Junzhi Ning, Wenchang Ning, Xinle Pang, Jiahui Peng, Runyu Peng, Yu Qiao, Jiantao Qiu, Xiaoye Qu, Yuan Qu, Yuchen Ren, Fukai Shang, Wenqi Shao, Junhao Shen, Shuaike Shen, Chunfeng Song, Demin Song, Diping Song, Chenlin Su, Weijie Su, Weigao Sun, Yu Sun, Qian Tan, Cheng Tang, Huanze Tang, Kexian Tang, Shixiang Tang, Jian Tong, Aoran Wang, Bin Wang, Dong Wang, Lintao Wang, Rui Wang, Weiyun Wang, Wenhai Wang, Yi Wang, Ziyi Wang, Ling‑I Wu, Wen Wu, Yue Wu, Zijian Wu, Linchen Xiao, Shuhao Xing, Chao Xu, Huihui Xu, Jun Xu, Ruiliang Xu, Wanghan Xu, GanLin Yang, Yuming Yang, Haochen Ye, Jin Ye, Shenglong Ye, Jia Yu, Jiashuo Yu, Jing Yu, Fei Yuan, Bo Zhang, Chao Zhang, Chen Zhang, Hongjie Zhang, Jin Zhang, Qiaosheng Zhang, Qiuyinzhe Zhang, Songyang Zhang, Taolin Zhang, Wenlong Zhang, Wenwei Zhang, Yechen Zhang, Ziyang Zhang, Haiteng Zhao, Qian Zhao, Xiangyu Zhao, Xiangyu Zhao, Bowen Zhou, Dongzhan Zhou, Peiheng Zhou, Yuhao Zhou, Yunhua Zhou, Dongsheng Zhu, Lin Zhu, Yicheng Zou
**Institutions:** 

## 🎯 Pitch

Intern-S1 introduces a groundbreaking multimodal 'generalist-specialist' model using dynamic tokenization and a Mixture-of-Experts framework to achieve state-of-the-art performance in scientific reasoning across diverse tasks. By effectively handling complex scientific data like molecules and time-series with over 5 trillion pre-trained tokens, it significantly advances open-source capabilities in scientific domains, impacting fields from drug design to earth observation.

---

## 1. Executive Summary (2-3 sentences)
Intern-S1 is a scientific, multimodal “generalist–specialist” model that integrates a Mixture‑of‑Experts language backbone with a vision encoder, a time‑series encoder, and a new dynamic tokenizer to process scientific strings (e.g., molecules, proteins) efficiently (Figure 3; §2). It is continually pre-trained on 5T tokens—over half from science—then post-trained with a Mixture‑of‑Rewards reinforcement learning framework across 1,000+ tasks, achieving state-of-the-art open-source performance on scientific reasoning and competitive general reasoning (Tables 2–4; §5–6).

## 2. Context and Motivation
- Problem addressed
  - Open-source models have advanced rapidly on popular tasks (natural images, code, math), but lag on high‑value scientific domains where data are scarce and modalities are diverse (molecules, spectra, time series) and reasoning is long-horizon and verifiable (§1). Figure 2 shows progress on “popular” benchmarks (x‑axis) outpacing “science” benchmarks (y‑axis), indicating uneven capability growth.
- Why this matters
  - Scientific discovery requires models that can read papers and lab notes, parse figures/equations, manipulate structured scientific strings, reason over long chains, and verify outcomes. These abilities can impact drug design, materials discovery, earth observation, and physics problem solving (§1).
- Prior approaches and their gaps
  - General multimodal and reasoning models (open-source and closed-source) excel on vision‑language and general math/code tasks but underperform on scientific problem solving, where professional evaluation demands precision and verifiability (Figure 1; §1).
  - Tokenizers and data pipelines are not tailored to scientific strings (e.g., SMILES/FASTA), leading to poor compression and weak embeddings (§2.2).
  - Existing RL-for-reasoning pipelines struggle to scale across many task types and are unstable on MoE backbones (§5.2.3).
- Positioning
  - Intern-S1 targets the “low-resource, high-value” science regime by: (i) aggressively curating high‑purity science corpora from PDFs and web domains (Figures 6–9; §4.1); (ii) introducing a dynamic tokenizer for scientific strings (Figure 4; §2.2); (iii) adding a time‑series encoder (§2.3); and (iv) using a Mixture‑of‑Rewards framework to jointly optimize across 1,000+ verifiable and preference‑based tasks (§5.2.1).

## 3. Technical Approach
Step-by-step overview of the system, data, and learning pipeline.

- Model architecture (Figure 3; §2)
  - Backbone: `MoE LLM` based on Qwen3 (235B total, with 28B parameters activated per token in the large model; 8B single-expert for the mini variant). MoE (Mixture‑of‑Experts) activates only a subset (“experts”) per token, increasing capacity without proportional compute.
  - Vision: `InternViT‑6B` (or 300M for mini) provides dynamic‑resolution visual features. High‑resolution inputs are downsampled via `pixel unshuffle` to reduce tokens (a 448×448 image becomes 256 visual tokens) then projected into the LLM embedding space (§2.1).
  - Scientific strings: `Dynamic tokenizer` detects modalities (e.g., <SMILES>, <FASTA>) and applies modality‑specific subword rules; embeddings are kept in separate, orthogonal subspaces to avoid semantic collisions across modalities (§2.2; Figure 4).
    - Why: a single, static tokenizer biases frequent natural‑language tokens and yields poor compression for rare scientific tokens. The dynamic tokenizer improves compression by up to 70% on SMILES vs. strong baselines (Figure 4 right).
    - Compression ratio CR is measured as the total number of characters divided by total number of tokens for a dataset (Eq. “CR(τ,D)” in §2.2).
  - Time-series: `Adaptive downsampling + Transformer blocks` to handle signals with widely varying sampling rates and lengths (e.g., EEG, seismic, astronomical light curves) (§2.3). Outputs feed the LLM as conditioning context.
- Training system and kernels (Figure 5 stages; §3)
  - Parallelism: Fully-Sharded Data Parallel (`FSDP`) for pretraining/SFT; FSDP + 1‑way expert parallel for RL (§3.1–3.2). FSDP shards model states across GPUs to fit very large models.
  - Precision: FP8 for GEMMs (matrix multiplications) to boost throughput (DeepGEMM-style scaling), keeping the vision tower at BF16 for stability (§3.1).
  - Custom kernels: FP8 Grouped GEMM for MoE (§3.1), Liger fused cross‑entropy, FlashAttention‑3 with variable-length support.
  - Load balance: `Variable-Length Balanced Strategy (VLBS)` packs and sorts sequences to equalize per‑GPU work, delivering ~2× speedup in their setup (§3.1).
  - RL rollout/training: `colocated design` shares devices between inference and training, redistributing weights on-the-fly; inference uses LMDeploy with EP8 (8‑way expert parallel), FP8 weights, CPU offload, continuous batching, and dynamic slot rebalancing (§3.2).
- Data pipelines and continue pre‑training (CPT) (§4)
  - Scale: 5T tokens CPT, with >2.5T science tokens (Figure 6). Multimodal CPT stage adds ~250B tokens (≈180B interleaved image‑text, ≈30B of those scientific; §4.1.2).
  - Page‑level PDF parsing (Figure 7; §4.1.1): low‑cost OCR parses all pages; an equation/symbol detector routes only “hard” pages to a high‑cost VLM parser; separate post-processing steps clean typical failure modes. Page-level deduplication and quality filters remove noisy pages (≈20% tokens dropped in archives; ≈50% kept for web‑PDFs).
  - Domain‑centric web parsing (Figure 8; §4.1.1): group pages by domain and use an LLM agent to decide per‑domain actions (keep/discard/rewrite); complements page‑level rules by capturing site-specific quirks.
  - Scientific recall & filtering (Figure 9; §4.1.1): build a 3‑level taxonomy over six sciences; use a strong LLM to label a silver set, train lightweight classifiers (fastText/1.5B LLM) with in-domain and OOD validation, pushing science purity from ~2% to ~50%.
  - Multimodal scientific data (§4.1.2): enforce structure on exam-style problems (question/options/answer/explanation); validate LaTeX/Markdown rendering with VLM judgments; filter broken links and low-quality images.
- Optimization strategy (§4.2)
  - `Batch size warmup` (Figure 10): start with a small batch for better early optimization, then switch to a larger batch for efficiency after ≈400B tokens; scaling‑law analysis guides the switch point (§4.2.1).
  - `Starting point choice` (Figure 11; §4.2.2): comparing base vs instruct LLM initializations shows minimal differences except coding; instruct tends to have slightly narrower output diversity (initial entropy 0.15 vs 0.19 for base) but can be offset by RL hyperparameters.
  - `Learning‑rate setting via scaling laws` (§4.2.3): under a warmup‑stable‑decay schedule, batch size B and gradient noise must satisfy inequality (Eq. 1). They fit a loss–LR relationship and frame LR selection as a constrained optimization (Eq. 2), predicting final CPT loss ≈1.16; achieved 1.17–1.18.
  - `Multimodal CPT objective` (§4.2.4): standard autoregressive next-token loss on text tokens only (Eq. 3), with square‑averaged token weights to reduce gradient bias (Eq. 4).
- Post‑training pipeline (§5)
  - Stage 1: “offline RL” on curated instruction data (commonly called SFT), where responses are chosen by best‑of‑N sampling to approximate high reward (§5.1). Text data are filtered, labeled by category and difficulty, and balanced; multimodal instruction includes slow‑thinking reasoning data and scientific VL tasks (§5.1.1).
    - Mixture search: validate atomic datasets, then compose and tune ratios to mitigate conflicts (style alignment, curriculum) (§5.1.2).
  - Stage 2: `Online RL` with a `Mixture‑of‑Rewards (MoR)` (Figure 12; §5.2.1).
    - Verifiable tasks (math, algorithms, science QA): rule-based checks plus `CompassVerifier` (a lightweight generative verifier) provide binary outcome rewards (§5.2.2).
    - Open‑ended tasks (dialogue, creative writing): `POLAR‑7B` provides a relative preference score (policy discriminative learning), acting as a scalable, model‑based reward (§5.2.2).
    - Data mixing and filtering: pre‑measure per‑domain difficulty/convergence to set ratios, then perform online filtering during rollout; discard overly easy/hard or noisy items (§5.2.1, §5.2.4).
- Policy optimization on MoE (§5.2.3, §5.2.4)
  - Problem: GRPO‑style methods become unstable on MoE because small numerical discrepancies (different kernels, FP8, dynamic expert routing) cause inference/training expert mismatches, making token‑level importance‑sampling clipping unreliable (§5.2.3).
  - Choice: use `OREAL`—SFT loss on positives and policy‑gradient on negatives—avoiding token‑level clipping; remove token‑level reward model to reduce compute, but then entropy collapses (§5.2.3).
  - Fix: add `KL‑Cov` entropy control—apply a KL penalty only to tokens whose covariance ranks in the top fraction k (Eq. 5)—to keep exploration. Final loss mixes SFT, PG, and KL‑Cov (Eq. 6). With k=0.2 and β=0.01, entropy stays ~0.2 and validation accuracy rises (Figure 14).
  - Training details: FP8 rollouts and updates, 8 samples per prompt, batch 4096 with 8 mini‑batches, AdamW lr 5e‑7, grad‑norm filter (>0.3) drops ~3% batches, freeze ViT and MoE router, 600 steps, then checkpoint averaging (§5.2.4).

## 4. Key Insights and Innovations
- Dynamic tokenizer for scientific strings (Figure 4; §2.2)
  - Novelty: automatic modality detection (tags or heuristics) and per‑modality tokenization with disjoint embedding subspaces, preventing semantic interference between “C” in DNA, molecules, and multiple‑choice text.
  - Why it matters: up to 70% higher compression ratio on SMILES versus GPT‑OSS‑120B, DeepSeek‑R1, and Qwen3 tokenizers (Figure 4 right), meaning fewer tokens per example and more efficient, less biased learning of scientific string structures.
- Page‑level, cost‑aware PDF parsing and domain‑centric web parsing (Figures 7–9; §4.1.1)
  - Novelty: detect “hard” pages (equations/symbols) and route only those to high‑cost VLM parsing; domain-centric agents make per‑site decisions (discard/keep/rewrite) instead of using one-size‑fits‑all heuristics.
  - Impact: lifts science purity from ≈2% to ≈50% while controlling cost; removes ≈20% noisy tokens in archives and retains ≈50% of web‑PDF tokens; routes only 3–5% of pages to expensive parsing.
- Mixture‑of‑Rewards for 1,000+ tasks (Figure 12; §5.2.1–5.2.2)
  - Novelty: unify binary verifiable rewards (math/science) with model‑based relative preferences (open‑ended dialogue) inside one online RL loop; combine rule checks, `CompassVerifier`, and `POLAR`.
  - Significance: allows simultaneous growth of specialized scientific skills and general conversational alignment, with hybrid offline–online filtering to balance task difficulties at scale.
- Stable RL on MoE via OREAL + KL‑Cov (Eq. 5–6; §5.2.3–5.2.4)
  - Insight: on MoE, token‑level importance‑sampling clipping is brittle due to expert routing mismatch. Swapping to OREAL avoids this failure mode; adding KL‑Cov restores exploration after removing token‑level credit assignment.
  - Evidence: entropy held near 0.2 with rising validation accuracy (Figure 14); faster AIME2024 gains than DAPO under the same base model (Figure 13).
- Systems choices that trade performance for efficiency wisely (§3, §4.2)
  - FP8 training/inference, VLBS packing, and batch‑size warmup deliver throughput without derailing optimization; scaling‑law‑based LR setting predicted—and matched—the final pretrain loss (1.16 predicted vs 1.17–1.18 achieved; §4.2.3).

## 5. Experimental Analysis
- Evaluation protocol (§6.1)
  - Tooling: VLMEvalKit and OpenCompass with “thinking” enabled; sampling decoding (temperature 0.7 for Intern‑S1, 0.8 for mini). Table 1 lists decoding parameters.
- Benchmarks covered (§6.2)
  - General reasoning (text): MMLU‑Pro, GPQA (Diamond), AIME‑2025, IFEval.
  - General multimodal: MathVista, MMMU, MathVision, MMStar.
  - Scientific (text): SmolInstruct (chemistry instructions), ChemBench, MatBench (materials property prediction), ProteinLMBench.
  - Scientific (multimodal): SFE (Scientists’ First Exam), PHYSICS, MicroVQA (microscopy), MSEarth‑MCQ (earth science), XLRS‑Bench (ultra‑high‑res remote sensing).
- Main results (Intern‑S1, large; Tables 2–4)
  - General reasoning
    - Text-only: 
      - “MMLU-Pro 83.5; GPQA 77.3; AIME-2025 86.0; IFEval 86.7” (Table 2).
      - Outperforms open-source multimodal baselines on all four, e.g., vs InternVL3‑78B (73.0 MMLU‑Pro, 49.9 GPQA, 10.7 AIME, 75.6 IFEval).
    - Multimodal:
      - “MathVista 81.5; MMMU 77.7; MathVision 62.5; MMStar 74.9” (Table 2).
      - Strong absolute scores, notably on MathVista (best among all models listed, including APIs; Table 2).
  - Scientific reasoning (text)
    - “SmolInstruct 51.0; ChemBench 83.4; MatBench 75.0; ProteinLMBench 63.1” (Table 3).
    - Large margins over open‑source VLMs; e.g., MatBench +25.7 vs InternVL3‑78B (49.3).
  - Scientific reasoning (multimodal)
    - “SFE 44.3; PHYSICS 44.0; MicroVQA 63.9; MSEarth‑MCQ 65.7; XLRS‑Bench 55.0” (Table 4).
    - Best on 4/5 datasets; PHYSICS is second to OpenAI o3 (47.9).
  - Takeaway: Intern‑S1 markedly narrows the gap to strong proprietary APIs in scientific multimodal reasoning while staying competitive on general reasoning.
- Mini model (Intern‑S1‑mini; Tables 5–7)
  - Text-only general: best among open-source models listed on MMLU‑Pro 74.8, GPQA 65.2, AIME‑2025 80.0 (Table 5).
  - Multimodal general: top on MMMU 72.3; competitive on others (Table 5).
  - Text-only science: wins on all four—SmolInstruct 32.2; ChemBench 76.5; MatBench 61.6; ProteinLMBench 63.1 (Table 6).
  - Multimodal science: best on 4/5—Physics 28.8; MicroVQA 56.6; MSEarth‑MCQ 58.1; XLRS‑Bench 51.6 (Table 7).
- Ablations and diagnostics
  - Batch-size warmup: switching from small to large batch after ~400B tokens outperforms fixed‑batch alternatives early and maintains efficiency (Figure 10; §4.2.1).
  - Starting point: instruct vs base: similar final outcomes except coding; entropy differences are small and manageable via RL hyperparameters (Figure 11; §4.2.2).
  - RL data filtering: hybrid offline–online filtering produces faster AIME2024 accuracy gains than DAPO on Qwen2.5‑32B base (Figure 13; §5.2.4).
  - Entropy control: with KL‑Cov, entropy stays around 0.2 and validation accuracy steadily improves; without it, entropy collapses and accuracy stagnates (Figure 14; §5.2.4).
- Do the results support the claims?
  - Yes for the central claims: Intern‑S1 is top‑tier open-source for science (Tables 3–4) and competitive in general multimodal reasoning (Table 2). The ablation figures (10–14) give concrete evidence that their optimization choices (batch warmup, MoE‑stable RL with KL‑Cov, data filtering) are causally helpful.
  - Remaining gaps: still trails proprietary models on some text-only reasoning (e.g., GPQA, IFEval in Table 2) and on PHYSICS (Table 4), suggesting room for further scientific reasoning and instruction-following refinement.

## 6. Limitations and Trade-offs
- Data dependence and cost (§4.1.1)
  - The strong science capability rests on heavy data engineering: page‑level PDF routing to costly VLM parsers and domain‑level LLM agents. Although only 3–5% of pages are escalated, this still requires substantial infrastructure.
  - The recall/filtering pipeline depends on silver labels from a “strong LLM,” which could propagate its biases into the curated corpus (Figure 9).
- Tokenizer modality coverage (§2.2)
  - The dynamic tokenizer currently supports four modalities. Scientific practice spans many string formats (crystal graphs, reaction SMIRKS, etc.); adding them requires new detection rules and tokenization strategies.
- RL verification and reward modeling (§5.2.1–5.2.2)
  - For open‑ended tasks, `POLAR` is a learned preference model; its judgments may not always match human values or task-specific desiderata. For verifiable tasks, rule‑based and `CompassVerifier` checks can still produce false positives/negatives, and models may “reward hack” these signals.
- MoE RL constraints (§5.2.3–5.2.4)
  - To stabilize training, the vision encoder and MoE router are frozen during RL. This reduces instability but also limits adaptation of perception and routing to reward signals.
  - The fix for instability (OREAL + KL‑Cov, no token-level credit assignment) trades fine-grained credit assignment for stability and compute efficiency. Some tasks may benefit from token-level rewards that are currently removed.
- Compute and scalability
  - Even with FP8 and FSDP, continuing pre‑training on 5T tokens and multimodal CPT on 250B tokens implies large compute budgets.
  - RL steps are relatively few (600), which the paper positions as efficient, but it may cap the attainable improvements on some domains.

## 7. Implications and Future Directions
- Field impact
  - Intern‑S1 demonstrates that a carefully engineered, science‑first pipeline—dynamic tokenization, science‑purified corpora, and mixed verifiable/preference rewards—can close much of the gap between general LMMs and science experts while remaining open-source (Figures 1–2; Tables 3–4).
  - It also shows how to make MoE backbones workable for online RL at scale through losses that avoid token‑level clipping and through entropy control (Eq. 5–6).
- Follow‑up research
  - Expand the dynamic tokenizer to additional scientific languages (reactions, crystal encodings, gene annotations) and investigate learned dynamic tokenizers specialized for each modality with stronger robustness guarantees (§2.2).
  - Integrate graph and 3D molecular encoders; pair time‑series encoders with physics‑informed priors for signals (e.g., spectral constraints).
  - Develop unified verifiers with formal methods (symbolic checkers, simulations) to reduce reward hacking and increase coverage for “hard‑to‑verify” tasks (§5.2.2).
  - Revisit token‑level credit assignment in MoE with routing‑aware off‑policy corrections, to regain fine-grained signals without instability (§5.2.3).
  - Explore multilingual and cross‑domain generalization: current evaluations focus largely on English and specific sciences; many scientific corpora are multilingual and domain‑idiosyncratic.
- Practical applications
  - Chemistry: molecular synthesis planning and reaction condition prediction (highlighted in Abstract); protein sequence plausibility checks (ProteinLMBench).
  - Materials science: property prediction (MatBench), candidate filtering in high‑throughput discovery.
  - Earth observation: high‑resolution remote sensing analysis (XLRS‑Bench) and figure-grounded earth science QA (MSEarth‑MCQ).
  - Microscopy and biomedical imaging: multimodal reasoning over images and text in MicroVQA.
  - Scientific reading: robust parsing of equations/tables/figures from PDFs (Figures 7–9) to build machine‑readable corpora for downstream tools.

> “Intern‑S1 is a multimodal Mixture‑of‑Experts (MoE) model with 28 billion activated parameters and 241 billion total parameters, continually pre‑trained on 5T tokens, including over 2.5T tokens from scientific domains.” (Abstract)

> “Intern‑S1 ... significantly outperforms open‑source models in scientific domains, surpassing closed‑source state‑of‑the‑art models in professional tasks” (Abstract), reflected in Tables 3–4 where Intern‑S1 leads on most scientific text and multimodal benchmarks.

Overall, Intern‑S1 offers a concrete recipe—data, tokenization, architecture, and RL—for building science‑capable foundation models, and its public release (weights and toolchains) provides a strong platform for community progress (§1, §6).
