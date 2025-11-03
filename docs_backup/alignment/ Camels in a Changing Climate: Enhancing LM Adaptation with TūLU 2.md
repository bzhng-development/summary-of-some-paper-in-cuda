# Camels in a Changing Climate: Enhancing LM Adaptation with TūLU 2

**ArXiv:** [2311.10702](https://arxiv.org/abs/2311.10702)
**Authors:** Hamish Ivison, Yizhong Wang, Valentina Pyatkin, Nathan Lambert, Matthew Peters, Pradeep Dasigi, Joel Jang, David Wadden, Noah A. Smith, Iz Beltagy, Hannaneh Hajishirzi
**Institutions:** Allen Institute for AI (AI2)

## 🎯 Pitch

TÜLU 2 introduces a powerful suite of open instruction-tuned LLMs leveraging an advanced instruction dataset mixture, large-scale Direct Preference Optimization, and innovative parameter-efficient research with QLoRA. This work significantly enhances instruction-following capabilities, achieving competitive or superior performance to GPT-3.5-turbo across numerous tasks, thus accelerating the development of high-quality open models and reducing reliance on proprietary LLMs.

---

## 1. Executive Summary
TÜLU 2 is a suite of open instruction-tuned large language models (LLMs) built on `LLaMA‑2` (and a code-focused variant on `Code Llama`) that systematically integrates three advances: a higher-quality instruction dataset mixture (`TÜLU‑V2‑mix`), large-scale preference optimization with `DPO`, and an analysis of `QLoRA` for parameter‑efficient tuning. Across diverse benchmarks, the 70B models achieve state‑of‑the‑art results among open models and match or exceed `GPT‑3.5‑turbo‑0301` on several tasks (Table 1), while the work also releases all data, code, and checkpoints.

## 2. Context and Motivation
- Problem/gap:
  - Open models have rapidly improved, but best practices for adapting a pretrained LLM to follow instructions—data mixtures, alignment methods, and efficient finetuning—remain fragmented. Prior open recipes left open questions: how far do new distilled datasets help, does `DPO` scale stably to very large models, and how close can `QLoRA` get to full finetuning for long-form generation?
- Why it matters:
  - Instruction-following quality determines model usefulness for everyday, open-ended tasks. Clear, reproducible recipes and open checkpoints accelerate research and reduce reliance on closed models.
- Prior approaches and shortcomings:
  - `TÜLU 1` (Wang et al., 2023b) showed that mixtures of human- and LLM-generated instruction data help, but the space of better mixtures and alignment methods evolved quickly.
  - RLHF based on PPO works but is complex and resource-intensive; newer offline preference methods like `DPO` are simpler but had not been demonstrated stably at 70B scale.
  - `QLoRA` enables cheaper tuning but its impact on open-ended benchmarks (e.g., AlpacaEval) was unclear beyond MMLU.
- Positioning:
  - TÜLU 2 provides a controlled, end-to-end recipe: better base models (`LLaMA‑2`/`Code Llama`), a curated V2 data mixture, extended-context SFT, large-scale `DPO`, and a head‑to‑head comparison with `QLoRA`. It evaluates broadly (knowledge, reasoning, multilingual QA, coding, toxicity, truthfulness, and open-ended quality) using a unified framework.

## 3. Technical Approach
This is a full pipeline for post‑pretraining adaptation.

- Base models and sizes
  - General: `LLaMA‑2` at 7B, 13B, 70B.
  - Code-focused: `Code Llama` at 7B, 13B, 34B.
  - Design choice: `LLaMA‑2` is a stronger pretraining starting point than `LLaMA‑1` (more tokens; see Section 2), so it should benefit downstream SFT and alignment.

- `TÜLU‑V2‑mix` (the supervised instruction data)
  - Goal: improve instruction-following quality with compact, higher-quality data.
  - Composition (Section 2, V2 data mixture; datasets with an asterisk are new since V1):
    - FLAN v2 (50k), FLAN CoT subset (50k) to keep some explicit chain-of-thought.
    - OpenAssistant-1 (7,708 high‑quality paths).
    - ShareGPT (114,046).
    - GPT4‑Alpaca (20k), Code‑Alpaca (20,022).
    - New: `LIMA` (1,030), `WizardLM Evol‑Instruct V2` (30k), `Open‑Orca` (30k), a small `science literature` mix (7,544) across QA/fact‑checking/summarization/IE, and `hardcoded` safety/identity prompts (140) to set correct “who am I” answers.
  - Filtering: remove samples that refer to other systems (e.g., GPT‑4) to avoid contradicting the hardcoded identity responses.
  - Size and length: 326,154 samples (down from 490,445 in V1). Context during SFT is extended to 8,192 tokens (from 2,048) to leverage long conversations; only 20 samples are truncated vs 63,900 previously. Figure 1 shows the long‑tailed length distribution (median around the 25–75% range of 230–1,464 tokens; mean 1,097).
  - Design choice: fewer but higher-quality, diverse, distilled datasets; longer context to preserve instruction trajectories.

- Supervised finetuning (SFT) setup (Appendix B)
  - Precision: bfloat16; epochs: 2; LR: 2e‑5 (1e‑5 for 70B); warmup 0.03; batch size 128; max sequence length 8,192.
  - Rationale: conservative, stable hyperparameters to avoid overfitting and to exploit long contexts.

- Preference optimization with `DPO` (Direct Preference Optimization)
  - What is DPO? A simple offline alignment method using pairs of model responses `(y_w, y_l)` where `y_w` is preferred to `y_l`. Instead of learning a separate reward model and running PPO, DPO directly trains the policy so that the preferred response is more probable relative to a reference policy. Intuitively, the loss increases the log-probability margin between preferred vs. dispreferred responses; a temperature‑like parameter `β` (beta) controls aggressiveness.
  - Training details (Section 2, “RLHF training”; Appendix B):
    - Data: a filtered, binarized form of `UltraFeedback` (Cui et al., 2023) for three epochs.
    - Hyperparameters: LR 5e‑7, β=0.1, bfloat16, max length 8,192, effective batch size 32.
    - Choice rationale: DPO was chosen for simplicity and accessibility vs PPO; the very low LR is necessary for stable large‑scale training.
  - Scaling: performed at 7B, 13B, and notably 70B (first stable 70B DPO model released).

- Parameter‑efficient alternative: `QLoRA` SFT (Appendix B)
  - What is QLoRA? Train low‑rank adapters on top of a 4‑bit quantized base model, greatly reducing memory cost.
  - Training here: epochs 5, LR 1e‑4, max length 4,096, LoRA rank 64, alpha 16, dropout 0.1; wrap all attention and feedforward linear layers.
  - Scope: only used for SFT (not used for DPO here).

- Code specialization: `CODE TÜLU 2`
  - Approach: start from `Code Llama` (already code‑pretrained) and SFT on `TÜLU‑V2‑mix` to see if general instruction tuning improves code and non‑code tasks relative to `Code Llama` baselines.

- Implementation and compute
  - Training on TPU v3 pods (256 chips; 512 chips for 70B DPO). The 70B DPO run took ~7 days for 3 epochs (Section 3 “Training”).
  - Codebases: EasyLM for full SFT/DPO; Open‑Instruct for QLoRA.
  - All checkpoints, data, and evaluation code are released.

## 4. Key Insights and Innovations
- A stronger open instruction data mixture (`TÜLU‑V2‑mix`)
  - Novelty: curated, distilled, and filtered blend focused on quality and diversity, with long‑context SFT.
  - Evidence: across sizes, V2 beats V1 especially on open-ended generation (Table 2). Example at 7B: AlpacaEval win‑rate 73.9 (V2) vs 64.5 (V1); average score 54.2 (V2) vs 47.8 (V1).
  - Significance: compact, higher quality beats bigger, noisier mixtures for instruction following.

- Stable, large‑scale `DPO` at 70B
  - Novelty: first public demonstration of DPO training at 70B with stable gains (Section 3.3).
  - Evidence: `TÜLU 2+DPO 70B` reaches AlpacaEval 95.1% vs 86.6% without DPO (Table 3/4) and improves MT‑Bench from 7.49 to 7.89 (Table 4). It becomes the best open model on MT‑Bench (Table 4; also Appendix D category breakdown).
  - Significance: validates DPO as a practical alternative to PPO‑based RLHF at very large scales.

- Clear picture of `QLoRA` trade‑offs for open‑ended generation
  - Insight: QLoRA performs close to full finetuning on many capability tasks (e.g., MMLU) but lags on long‑form, open-ended generation such as AlpacaEval.
  - Evidence: at 13B, AlpacaEval drops from 78.9 (full) to 65.6 (QLoRA); at 70B, 86.6 (full) to 78.6 (QLoRA) (Table 5).
  - Significance: cost savings come with measurable quality loss on open-ended tasks, though the gap shrinks as model size grows.

- Domain specialization via `CODE TÜLU 2`
  - Insight: using `Code Llama` as a base and SFT on V2 mix yields large code gains but weaker general open-ended behavior.
  - Evidence: at 7B, CodexEval jump to 68.9 (`CODE TÜLU 2`) vs 36.9 (`TÜLU 2`), but AlpacaEval falls to 58.0 vs 73.9 (Table 6).
  - Significance: confirms the specialization/generalization trade‑off; smaller code‑specialized models can match or beat much larger general models on coding.

Incremental vs fundamental:
- Incremental: swapping base model to `LLaMA‑2`, longer context SFT, dataset curation.
- Fundamental/practice‑shifting: stable 70B `DPO` and the quantified, task‑level picture of `QLoRA` vs full finetuning.

## 5. Experimental Analysis
- Benchmarks and methodology (Section 3; Appendix A)
  - Knowledge: `MMLU` (0‑shot accuracy).
  - Math reasoning: `GSM8K` (8‑shot CoT accuracy).
  - Broad reasoning: `BBH` (3‑shot CoT accuracy).
  - Multilingual QA: `TyDiQA` Gold Passage (1‑shot F1).
  - Coding: `CodexEval` (HumanEval) Pass@10.
  - Open-ended helpfulness: `AlpacaEval` (% win vs davinci‑003 judged by GPT‑4‑0613).
  - Safety: `ToxiGen` (% toxic; lower is better). For averages, they transform to 100−x.
  - Truthfulness: `TruthfulQA` (% of outputs both truthful and informative). Caution: UltraFeedback includes TruthfulQA prompts; therefore, comparisons involving DPO models omit TruthfulQA due to contamination (Section 3 “Evaluation tools”).
  - Additional: `MT‑Bench` (GPT‑4‑0613 judge; single‑answer grading).

- Baselines (Table 1)
  - Proprietary: `GPT‑4‑0613`, `GPT‑3.5‑turbo‑0613`, `GPT‑3.5‑turbo‑0301`.
  - Open: `LLaMA‑2‑Chat` (7B/13B/70B), `Zephyr‑Beta 7B`, `Xwin‑LM 70B`.
  - TÜLU 2 suite: `TÜLU 2` and `TÜLU 2+DPO` at 7B/13B/70B.

- Headline results (Table 1; Table 4; Appendix D)
  - Average open‑model leader: 
    - “TÜLU 2 70B” has the highest average among open models across the 7 tasks in Table 1 (average 73.8). Where it is not best per task, the gap to the top open model is under 1% on average.
  - Competitive with GPT‑3.5‑0301:
    - `TÜLU 2 70B` matches or exceeds `GPT‑3.5‑turbo‑0301` on several tasks; e.g., AlpacaEval win‑rate 86.6 vs 83.6; ToxiGen 0.5% vs 27.7% (lower is better) (Table 1).
  - DPO boosts open‑ended quality substantially:
    - AlpacaEval: +11.2, +10.6, +8.5 points at 7B, 13B, 70B respectively (Table 3), reaching 95.1 at 70B (Table 4).
    - MT‑Bench: 70B improves from 7.49 to 7.89; among open models, `TÜLU 2+DPO 70B` is the best (Table 4; Appendix D category scores show gains particularly in Roleplay/Writing/Extraction).
    - Verbosity: average output length increases after DPO (e.g., 70B from 1,011 to 1,414 tokens on AlpacaEval; Table 4).
  - Multilingual regression with DPO:
    - TyDiQA drops notably after DPO (e.g., 70B: 53.6 → 35.8, −17.8 F1; Table 3). The SFT and DPO datasets are predominantly English, likely making non‑English out‑of‑distribution.
  - V2 vs V1 mixture (Table 2):
    - Clear gains on open‑ended tasks (e.g., AlpacaEval at 7B: 73.9 vs 64.5; BBH at 7B: 48.5 vs 44.2; CodexEval at 13B: 49.0 vs 38.9).
    - Slight regressions on GSM8K and TyDiQA in some sizes (likely fewer CoT examples and less multilingual exposure).
    - Improvements shrink with model size (average +13% at 7B, ~+1% at 70B), suggesting larger models rely less on instruction data quality.
  - QLoRA vs full finetuning (Table 5):
    - MMLU is close (e.g., 70B: 67.4 QLoRA vs 67.3 full), but AlpacaEval lags (70B: 78.6 vs 86.6).
    - The average gap shrinks with scale (7B average 47.7 vs 53.0; 70B average 71.0 vs 73.4).
  - Code specialization (Table 6):
    - `CODE TÜLU 2` dramatically boosts coding: at 7B, Pass@10 68.9 vs 36.9 for `TÜLU 2`; at 34B, Pass@10 82.5 (higher than Code Llama Instruct’s 76.5).
    - But general open‑ended generation degrades: AlpacaEval drops (e.g., 7B: 58.0 vs 73.9).

- Do the experiments support the claims?
  - The paper backs each claim with ablations (V1 vs V2; ShareGPT‑only vs V2), method comparisons (DPO vs no‑DPO; QLoRA vs full), and domain specialization tests (Code Llama vs LLaMA‑2 bases). Detailed metrics across multiple tasks (Tables 1–6) and judge‑based evaluations (Table 4; Appendix D) make the evidence convincing.
  - Caveats are clearly identified: contamination of TruthfulQA in UltraFeedback; non‑pinned evaluator versions on some leaderboards are normalized to GPT‑4‑0613.

- Example quoted results
  > Table 3: AlpacaEval improves by +11.2 (7B), +10.6 (13B), and +8.5 (70B) after DPO; TyDiQA drops by −1.9 (7B), −13.5 (13B), −17.8 (70B).

  > Table 4: `TÜLU 2+DPO 70B` achieves MT‑Bench 7.89 and AlpacaEval 95.1% (avg output length 1,414), outperforming its non‑DPO counterpart (7.49; 86.6%; 1,011).

  > Table 6: `CODE TÜLU 2` (7B) scores Pass@10 68.9 on HumanEval vs `TÜLU 2` (7B) at 36.9, but AlpacaEval decreases from 73.9 to 58.0.

## 6. Limitations and Trade-offs
- Data coverage and multilinguality
  - SFT and DPO datasets are predominantly English; performance on TyDiQA regresses after DPO (Table 3). Multilingual instruction and preference data are needed to avoid this trade-off.
- Evaluation dependence on LLM judges
  - AlpacaEval and MT‑Bench use GPT‑4 as the judge; while this setup is standard, it can introduce biases and version sensitivity. The paper fixes GPT‑4‑0613 for comparability, but broader generality remains a community caveat (Section 3 “Evaluation tools”).
- TruthfulQA contamination
  - The preference dataset (UltraFeedback) contains TruthfulQA prompts; results involving DPO models omit TruthfulQA to avoid unfairness (Section 3 “Evaluation tools”).
- QLoRA practicality vs quality
  - QLoRA substantially reduces compute but loses quality for long-form open-ended generation (Table 5). The gap narrows as models scale, yet remains material at 70B on AlpacaEval.
- Specialization vs generalization
  - Code specialization yields large coding gains but hurts open-ended helpfulness (Table 6). Choosing a code-pretrained base steers model behavior in ways that reduce general conversational quality.
- Compute requirements
  - Full 70B DPO training requires substantial resources (512 TPUv3 chips for ~7 days), limiting accessibility despite open artifacts (Section 3 “Training”).
- Safety calibration
  - While ToxiGen is measured (Table 1), detailed refusal behavior and broader safety trade-offs under DPO are not fully dissected; the paper flags this as future work.

## 7. Implications and Future Directions
- Impact on the field
  - Demonstrates a practical, reproducible, open recipe for high‑quality instruction tuning with long context, scaled DPO, and clear guidance on when `QLoRA` suffices. Establishes a new open baseline competitive with older GPT‑3.5 variants across many tasks (Table 1).
- What this enables
  - Researchers can:
    - Build on `TÜLU‑V2‑mix` and released code to explore mixture design (e.g., multilingual, safety‑focused, or domain‑targeted variants).
    - Study DPO dynamics at scale: e.g., how β, learning rate, and pair sampling affect verbosity, safety, or truthfulness.
    - Evaluate hybrid strategies: QLoRA for capability retention plus targeted full finetuning or DPO on open‑ended quality.
- Practical applications
  - Organizations needing open models for chat, reasoning, and coding can:
    - Use `TÜLU 2 70B` for general assistant tasks where open‑ended quality is critical.
    - Use `CODE TÜLU 2` for code generation/testing tasks where coding accuracy dominates.
    - Adopt `QLoRA` recipes for rapid adaptation under tight compute budgets, with awareness of open‑ended quality trade-offs.
- Recommended next steps (some echoed in Section 4 “Conclusion”)
  - Add multilingual data to both SFT and preference stages to recover TyDiQA performance.
  - Compare DPO to PPO and ReST at 70B+ under controlled data/eval regimes.
  - Analyze refusal behavior and verbosity control under DPO (Table 4 suggests increased verbosity).
  - Iterate mixture curation with principled ablations to pinpoint which sources drive open‑ended gains (e.g., WizardLM vs Open‑Orca vs ShareGPT).
  - Extend long‑context training and evaluation beyond 8k tokens to align with emerging long‑context use cases.

Overall, TÜLU 2 provides a carefully engineered and well‑measured update to open instruction tuning: a stronger data mixture, a validated large‑scale DPO recipe, and an honest assessment of QLoRA and code specialization trade-offs, all supported by detailed tables and released artifacts.
