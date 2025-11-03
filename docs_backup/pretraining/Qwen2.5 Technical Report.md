# Qwen2.5 Technical Report

**ArXiv:** [2412.15115](https://arxiv.org/abs/2412.15115)
**Authors:** Qwen, An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan Lin, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jingren Zhou, Junyang Lin, Kai Dang, Keming Lu, Keqin Bao, Kexin Yang, Le Yu, Mei Li, Mingfeng Xue, Pei Zhang, Qin Zhu, Rui Men, Runji Lin, Tianhao Li, Tingyu Xia, Xingzhang Ren, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yu Wan, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, Zihan Qiu
**Institutions:** 

## 🎯 Pitch

Qwen2.5 introduces groundbreaking large language models, scaling up to 18 trillion tokens and integrating a unique two-stage reinforcement learning pipeline to deliver exceptional performance in math, coding, and long-context processing. By bridging the gap between model size, cost, and capability, Qwen2.5 addresses real-world demands for diverse, high-quality LLM applications, setting new benchmarks for both small edge devices and large-scale deployments.

---

## 1. Executive Summary (2-3 sentences)
Qwen2.5 is a family of large language models (LLMs) that scales high‑quality pre‑training to 18 trillion tokens and introduces a two‑stage reinforcement learning pipeline (offline DPO, online GRPO) on top of a million‑example supervised fine‑tuning set. It delivers strong general, math, coding, and long‑context performance across sizes from edge (0.5B) to large (72B) and API MoE variants, with `Qwen2.5-72B-Instruct` competing with much larger models and `Qwen2.5-Turbo` handling up to 1M‑token contexts (Sections 1–5; Tables 2–17; Figure 2).

## 2. Context and Motivation
- Problem gap:
  - Open‑weight LLMs have improved rapidly, but users face trade‑offs among size, cost, long‑context handling, and alignment (Section 1).
  - Prior Qwen2 used 7T tokens and had limited post‑training breadth; generation length was short and structured I/O support was weaker (Abstract; Section “Better in Use”).
- Importance:
  - Real‑world applications need models that are cost‑effective at multiple sizes, follow instructions reliably, reason about math/code, and process very long inputs (Sections 1, 4.1, 4.4).
- Prior approaches and limitations:
  - Existing open series (Llama, Mistral, Gemma, etc.) offer strong baselines but often have narrower context windows, less emphasis on structured data and long responses, or weaker small‑model performance (Section 1; evaluation in Tables 3–5, 7–10).
  - Long‑context handling typically relies on post‑hoc extrapolation with quality drop on short tasks; reward model (RM) evaluation is often Goodhart‑prone (Sections 3.3, 5.2.3; Tables 16–17).
- Positioning:
  - Qwen2.5 scales data and post‑training breadth, introduces staged long‑context training plus inference‑time attention improvements, and provides a wide size range with open weights and API MoE options (Sections 2–4; Table 1).

## 3. Technical Approach
This section explains how Qwen2.5 is built and aligned.

- Model family and architecture (Section 2; Table 1):
  - Dense decoder‑only Transformers at 0.5B, 1.5B, 3B, 7B, 14B, 32B, 72B parameters; API‑served MoE variants `Qwen2.5-Turbo` and `Qwen2.5-Plus`.
  - Key components:
    - `GQA` (Grouped Query Attention): reduces key‑value cache cost by sharing keys/values across groups of attention heads—keeps attention efficient at long contexts.
    - `SwiGLU` activation and `RMSNorm` with pre‑norm: stable and efficient training.
    - `RoPE` (rotary positional embeddings) with a learned bias term in attention (`QKV bias`), which improves length extrapolation.
  - Tokenizer: byte‑level BPE with 151,643 tokens and 22 control tokens, including new ones for tool use; a unified vocabulary across all models (Section 2).

- Pre‑training data and schedule (Section 3):
  - Scale and curation:
    - Expanded from 7T to 18T tokens, with better filtering and mixture balancing using Qwen2‑Instruct as a data quality filter; domains like science/technology are up‑sampled (Section 3.1).
    - Stronger math/code via integrating datasets from `Qwen2.5-Math` and `Qwen2.5-Coder`; additional high‑quality synthetic data filtered by general and math RMs (Section 3.1).
  - Hyperparameter scaling laws (Section 3.2):
    - Empirical laws relate optimal learning rate and batch size to model size (`N`) and data scale (`D`) for both dense and MoE models, targeting loss minimization across a grid of sizes (dense: 44M–14B; MoE: 44M–1B activated params).
    - Used to configure MoE to reach parity with selected dense models by tuning activated/total parameters.
  - Long‑context pre‑training (Section 3.3):
    - Two‑phase for dense models: start at 4,096 tokens then extend to 32,768; `RoPE` base raised from 10,000 to 1,000,000 via ABF (Attention Base Frequency) to preserve position encoding quality at long lengths.
    - `Qwen2.5-Turbo` uses progressive stages up to 262,144 tokens with RoPE base 10,000,000; each stage mixes 40% max‑length and 60% shorter sequences for smooth adaptation.

- Inference‑time long‑context upgrades (Section 3.3):
  - `YARN` and `Dual Chunk Attention (DCA)`:
    - YARN: a training‑free method to extend usable context windows of RoPE‑based models by rescaling positional encodings.
    - DCA: splits sequences into chunks and structures attention to preserve long‑range information with manageable compute.
    - Outcome: up to 4× sequence length capacity—Turbo handles up to 1M tokens; others up to 131,072 tokens—while keeping short‑sequence quality (Tables 16–17).

- Post‑training pipeline (Section 4):
  - Supervised Fine‑Tuning (`SFT`, Section 4.1):
    - >1M examples, emphasizing long responses (up to 8,192 generated tokens), math chain‑of‑thought, code with execution checks and unit tests, structured data reasoning (tables/JSON), logical reasoning (70k new queries), multilingual transfer, robust system prompts, and response filtering with critic and multi‑agent scoring.
    - Training details: 2 epochs, sequence length 32,768; LR decays 7e‑6 → 7e‑7; weight decay 0.1; gradient clipping 1.0.
  - Offline RL (`DPO`, Section 4.2):
    - DPO reframes preference learning as direct optimization over pairs (a “chosen” vs “rejected” answer). Qwen2.5 builds ~150k pairs by resampling with the SFT model and applying execution‑based checks or matching when objective scoring exists (math/coding/instruction logic).
    - Includes human + automated review; trained 1 epoch with Online Merging Optimizer at LR 7e‑7.
  - Online RL (`GRPO`, Section 4.3):
    - `GRPO` (Group Relative Policy Optimization): samples 8 responses per query and updates the policy relative to group‑wise baselines, improving signals like truthfulness/helpfulness/conciseness/relevance/harmlessness/debiasing defined by the RM’s labeling criteria.
    - Curriculum: prioritize queries with higher score variance under the RM to learn where quality differs most; global batch size 2048; each episode uses pairs of (query, response).
  - Long‑context post‑training (Section 4.4):
    - For `Qwen2.5-Turbo`, SFT is two‑stage: only short instructions (≤32k) first, then a mix of short and long (≤262k). RL is done on short instructions only, due to cost and lack of reliable long‑context RMs, yet still improves long‑context alignment.

- Implementation choices and why they matter:
  - Using verifiable domains (math, code) for offline RL ensures “learnable and reliable” signals before moving to harder‑to‑score human preference dimensions in online RL (Sections 4.2–4.3).
  - YARN + DCA avoids long‑context training from scratch for all models while preserving short‑length behavior (Tables 16–17 show “w/o DCA+YARN” ablation).
  - Size coverage (0.5B→72B + MoE) addresses practical deployment ranges and cost‑latency trade‑offs (Section 1; Table 1).

## 4. Key Insights and Innovations
- Scaling high‑quality data with smarter mixture control (Section 3.1):
  - Novelty: combines LLM‑assisted multi‑dimensional data filtering, domain balancing (down‑sample social/entertainment templates; up‑sample science/tech), and expert synthetic data with RM filtering.
  - Significance: directly ties to large gains in math/coding and general benchmarks at fixed model sizes (Tables 2–5, 7–10). This is more than just “more tokens”—it is “more useful tokens.”

- Two‑stage RL that separates verifiable skills from subjective alignment (Sections 4.2–4.3):
  - Novelty: a structured pipeline—Offline DPO on objectively checkable tasks, then Online GRPO on human‑preference dimensions with a strong RM.
  - Significance: improves instruction following and preference alignment without sacrificing reasoning quality; evidenced by large jumps on IFEval, Arena‑Hard, and MT‑Bench (Tables 6–8).

- Long‑context capability without sacrificing short‑context quality (Sections 3.3, 4.4; Tables 16–17):
  - Novelty: progressive long‑context training for Turbo plus YARN+DCA upgrades for all models; ablations show YARN+DCA preserves ≤32k behavior and substantially boosts ≥64k performance.
  - Significance: `Qwen2.5-Turbo` reaches 1M tokens and achieves 100% in a 1M‑token passkey retrieval test (Figure 2); `Qwen2.5-72B-Instruct` leads open‑weight long‑context scores (Tables 16–17).

- Reward‑model evaluation skepticism substantiated with multi‑benchmark evidence (Section 5.2.3; Table 15):
  - Insight: optimizing an RM for one benchmark (e.g., RewardBench) risks Goodhart’s law—improvements there may degrade on others and not predict downstream RL model quality.
  - Significance: pushes the community to adopt broader RM evaluation and to seek RM metrics that better predict RL outcomes.

## 5. Experimental Analysis
- Evaluation methodology (Section 5):
  - Contamination control: n‑gram de‑duplication with LCS thresholds ≥13 and ≥60% of the shorter sequence length to remove training samples overlapping with test items (Section 5).
  - Base model evaluation (Section 5.1; Tables 2–5): general (MMLU, MMLU‑Pro, MMLU‑redux, BBH, ARC‑C, TruthfulQA, Winogrande, HellaSwag), math/science (GPQA, TheoremQA, GSM8K, MATH), coding (HumanEval, HumanEval+, MBPP, MBPP+, MultiPL‑E), multilingual (exam, understanding, math, translation).
  - Instruction‑tuned evaluation (Section 5.2; Tables 6–10): general (MMLU‑Pro, MMLU‑redux, LiveBench 0831), math/science (GPQA, GSM8K, MATH), coding (HumanEval, MBPP, MultiPL‑E, LiveCodeBench 2305–2409), alignment (IFEval, MT‑Bench, Arena‑Hard).
  - In‑house automatic evaluations in English/Chinese and multilingual extensions (Tables 11–14).
  - Long‑context tests: RULER, LV‑Eval with keyword recall, LongBench‑Chat; passkey retrieval to 1M tokens for Turbo (Tables 16–17; Figure 2).
  - Long‑context speed: sparse attention based on Minference to accelerate prefill; TTFT speedups 3.2–4.3× at 1M tokens (Figure 3).

- Representative quantitative highlights (instruction‑tuned):
  - Large scale (Table 6):
    - “`Qwen2.5-72B-Instruct` achieves MATH 83.1, GSM8K 95.8, LiveCodeBench 55.5, Arena‑Hard 81.2, MT‑Bench 9.35.”
    - “`Qwen2.5-Plus` further improves MATH to 84.7, MultiPL‑E to 77.0, and Arena‑Hard to 81.4.”
    - On MMLU‑redux: 86.8 (`Qwen2.5-72B-Instruct`) vs 86.2 (Llama‑3.1‑405B‑Instruct).
  - Mid scale (Table 7):
    - “`Qwen2.5-32B-Instruct`: MMLU‑Pro 69.0, MATH 83.1, LiveCodeBench 51.2, Arena‑Hard 74.5.”
    - “`Qwen2.5-Turbo` (MoE): MATH 81.1, GSM8K 93.8, MultiPL‑E 73.7, IFEval 76.3—often matching or beating `Qwen2.5-14B-Instruct` despite lower cost.”
  - 7B scale (Table 8):
    - “`Qwen2.5-7B-Instruct` improves MATH to 75.5 and HumanEval to 84.8, beating Gemma2‑9B‑IT and Llama3.1‑8B‑Instruct on most metrics.”
  - Edge models (Table 9–10):
    - “`Qwen2.5-3B-Instruct` reaches MATH 65.9 and MultiPL‑E 60.2, competitive with larger 3.5–4B models.”
    - “`Qwen2.5-1.5B-Instruct` jumps to MATH 55.2 and HumanEval 61.6; `Qwen2.5-0.5B-Instruct` reaches MATH 34.4.”

- Long‑context results and ablations:
  - RULER (Table 16): at 128k tokens average, `Qwen2.5-72B-Instruct` scores 95.1 overall and 88.4 at 128k; without YARN+DCA the 128k score drops to 67.0.
  - LV‑Eval (Table 17): `Qwen2.5-72B-Instruct` averages 60.4 at 16k and remains 50.9 at 128k (45.2 at 256k); removing YARN+DCA substantially degrades ≥64k.
  - Passkey retrieval (Figure 2): Turbo achieves “100% accuracy at 1M tokens,” demonstrating precise recall across extreme lengths.

- Base models (Tables 2–5):
  - `Qwen2.5-72B` base surpasses 70B peers and is competitive with Llama‑3‑405B base on several tasks (e.g., MMLU 86.1 vs 85.2; GSM8K 91.5 vs 89.0; MATH 62.1 vs 53.8; Table 2).
  - `Qwen2.5-32B` base is particularly strong for math/coding (MATH 57.7, MBPP 84.5; Table 3).
  - 7B base improves markedly over Qwen2‑7B (MATH 49.8 vs 43.5; HumanEval 57.9 vs 51.2; Table 4).
  - Small bases (Table 5) show outsized gains; `Qwen2.5-0.5B` outperforms Gemma2‑2.6B on several math/coding tasks.

- RM evaluation (Table 15):
  - `Qwen2.5-RM-72B` is competitive across RewardBench, RMB, PPE, and a Chinese preference set; it leads on PPE’s objective average (69.85) and on the Chinese set (Accuracy 61.27), but it is not strictly dominant everywhere—underscoring the multi‑benchmark perspective.

- Do the experiments support the claims?
  - Breadth and depth: results cover multiple scales, domains, and public + internal benchmarks, with decontamination (Section 5). Long‑context ablations quantify the impact of YARN+DCA (Tables 16–17).
  - Alignment claims are supported by IFEval, MT‑Bench, Arena‑Hard, and in‑house preference results (Tables 6–8, 11–12). The RM section explicitly cautions against single‑metric over‑optimization (Section 5.2.3), which adds credibility.

## 6. Limitations and Trade-offs
- Training and compute:
  - 18T tokens and long‑context staging imply very high compute; reproducing training is out of reach for most groups (Sections 3.1–3.3).
- Openness and licensing:
  - Some open‑weight sizes carry non‑Apache licenses (`3B` is “Qwen Research”; `72B` is “Qwen”; Table 1). MoE variants with 1M context (`Turbo`, `Plus`) are proprietary API models (Abstract, Section 2).
- Generation length vs context length:
  - Open‑weight models support long context (up to 128k) but generation length is capped at 8,192 tokens (Table 1). Very long reasoning chains may still be truncated.
- RL and reward models:
  - Long‑context RL is avoided due to cost and lack of reliable long‑context RMs (Section 4.4). Alignment for ultra‑long inputs thus relies mainly on SFT.
  - RM quality does not straightforwardly predict RL model quality; this remains an open problem (Section 5.2.3).
- Data and synthetic content:
  - Heavy use of synthetic data and LLM‑based filtering could propagate upstream biases or errors despite multi‑stage filtering (Section 3.1, 4.1). The paper does not detail public release of datasets for external audit.
- Multilingual and cultural nuance:
  - While multilingual scores are strong, cultural nuance understanding (BLEnD) leaves room for improvement even at large scales (Tables 13–14).

## 7. Implications and Future Directions
- How this changes the field:
  - Demonstrates that careful data scaling and staged post‑training can let a 72B model compete with much larger ones on hard tasks (Table 6). It also shows small, well‑trained models (0.5B–3B) can be surprisingly capable, useful for on‑device or edge deployments (Tables 9–10).
  - Establishes a practical recipe for long‑context capability that preserves short‑context quality via YARN+DCA, plus progressive long‑context training for API models (Tables 16–17; Figure 2).
  - Encourages the community to move beyond single RM benchmarks to multi‑metric, predictive RM evaluations (Section 5.2.3; Table 15).

- Follow‑up research enabled/suggested:
  - Reward modeling:
    - Develop RM benchmarks that better predict downstream RL outcomes; investigate multi‑objective RMs and uncertainty‑aware RMs.
  - Long‑context alignment:
    - Build reliable long‑context reward signals and efficient long‑context RL training to go beyond SFT‑only alignment (Section 4.4).
  - Data governance and transparency:
    - Public audits of mixture composition, synthetic data filtering pipelines, and cross‑lingual quality checks.
  - Inference‑time scaling:
    - Combine long‑context with inference‑time reasoning methods (e.g., tool‑augmented reflection) under strict latency budgets.

- Practical applications:
  - Enterprise assistants needing long documents, contracts, or codebases; analytics over semi‑structured/structured data (tables/JSON) with verifiable outputs (Section 4.1).
  - Education and scientific domains requiring math/theorem problem solving (Tables 6–8).
  - Edge scenarios where `0.5B–3B` models deliver useful accuracy under tight resource constraints (Tables 9–10).
  - API deployments where `Qwen2.5-Turbo` offers 1M‑token context with improved TTFT using sparse attention (Figure 3).

> In short, Qwen2.5’s technical recipe—high‑quality 18T pre‑training, verifiable‑first offline RL + preference‑focused online RL, and training‑plus‑inference methods for long contexts—yields strong, scalable performance across sizes and tasks, while surfacing open challenges in reward modeling and long‑context alignment that invite further work.
