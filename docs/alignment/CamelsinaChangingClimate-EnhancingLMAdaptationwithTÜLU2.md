# Camels in a Changing Climate: Enhancing LM Adaptation with TÜLU 2

**ArXiv:** [2311.10702](https://arxiv.org/abs/2311.10702)

## 🎯 Pitch

TÜLU 2 delivers an open, reproducible suite of large language models—spanning up to 70B parameters—finetuned on a rigorously curated instruction dataset and enhanced with scalable Direct Preference Optimization (DPO). This work not only sets a new open benchmark for instruction-following and coding ability, but also demystifies what matters in adapting LLMs to downstream tasks, providing the open-source community with all the models, data, and recipes needed for rapid and reliable language model alignment.

---

## 1. Executive Summary
TÜLU 2 is a publicly released suite of instruction‑tuned and preference‑optimized large language models (LLMs) based on LLAMA‑2 (and Code LLaMA) that systematizes recent advances in data curation, long‑context training, and human preference learning. It delivers strong open‑weight models up to 70B parameters, demonstrates stable Direct Preference Optimization (DPO) at that scale, and provides a carefully curated training mixture that improves open‑ended generation and coding ability (Tables 1–6, Figure 1).

## 2. Context and Motivation
- Problem addressed
  - How to reliably adapt open LLMs to follow user instructions and preferences across many tasks using openly available resources (datasets, code, and checkpoints).
  - Prior open instruction‑tuned models varied in data quality, lacked consistent evaluation, and did not demonstrate that newer preference‑learning techniques like `DPO` scale to very large models (70B).
- Importance
  - Practical: Industry‑grade instruction following and safety alignment typically depend on proprietary data and RLHF pipelines; open, reproducible recipes lower the barrier to strong assistants.
  - Scientific: Isolating the effects of data mixtures, finetuning recipes, and preference learning clarifies what actually drives performance across reasoning, knowledge, multilinguality, and open‑ended generation.
- Prior approaches and their gaps
  - Instruction tuning on mixes such as FLAN, Dolly, OpenAssistant, and ShareGPT improved helpfulness but varied in quality and length coverage. Earlier TÜLU (V1) used smaller context lengths and older base models (LLAMA‑1).
  - RLHF primarily used PPO or proprietary pipelines; DPO is simpler but was not shown to be stable and beneficial at 70B scale.
  - Parameter‑efficient finetuning (`QLoRA`) promised efficiency but its impact on open‑ended generation had not been deeply compared to full finetuning.
- Positioning of this work
  - Provides: (1) an improved instruction mixture `TÜLU‑V2‑mix`; (2) fully finetuned LLAMA‑2 models at 7B/13B/70B; (3) the largest DPO‑trained model to date (70B); (4) `CODE TÜLU 2` models from Code LLaMA showing strong coding performance; (5) a unified evaluation covering knowledge (MMLU), reasoning (GSM8k/BBH), multilingual QA (TyDiQA), coding (HumanEval/Codex‑Eval), open‑ended generation (AlpacaEval, MT‑Bench), toxicity (ToxiGen), and truthfulness (TruthfulQA) with pinned evaluation details (Appendix A; Tables 1–6, 8).

Definitions for uncommon terms:
- Instruction tuning: supervised finetuning on datasets of instruction–response pairs to make a model follow user prompts more reliably.
- RLHF (Reinforcement Learning from Human Feedback): training that uses human preferences to shape model behavior.
- DPO (Direct Preference Optimization): a simple, offline preference‑learning method that directly increases the log‑likelihood of “chosen” responses over “rejected” ones relative to the base policy; it avoids a learned reward model and PPO complexity (Section 2, “RLHF training”; Appendix B for hyperparameters).
- QLoRA: parameter‑efficient finetuning using 4‑bit base‑model quantization plus low‑rank adapters, enabling large‑model training on limited hardware (Section 2, “QLoRA training”; Appendix B).
- AlpacaEval and MT‑Bench: benchmarks that use GPT‑4 as a judge to compare model responses on open‑ended prompts (Section 3, “Evaluation tools”).

## 3. Technical Approach
The system is a staged adaptation pipeline with careful data choices and controlled training recipes.

1) Base models and sizes
- Use LLAMA‑2 base models at 7B, 13B, 70B and Code LLaMA at 7B, 13B, 34B (Section 2, “Improved base models”).
- Rationale: LLAMA‑2 is trained on ~2T tokens and outperforms LLAMA‑1. Code LLaMA adds code‑focused pretraining to boost coding tasks.

2) New instruction mixture: `TÜLU‑V2‑mix`
- Composition after filtering: 326,154 samples (Section 2, “V2 data mixture”).
- Sources (new additions marked with “*”; sample counts below are from Section 2):
  - FLAN v2 (50k) and its Chain‑of‑Thought subset (50k) to include step‑by‑step reasoning.
  - OpenAssistant‑1 high‑quality paths (7,708) to emphasize curated human interactions.
  - ShareGPT (114,046) for real‑world conversational patterns.
  - GPT4‑Alpaca (20k) and Code‑Alpaca (20,022) for distilled general and code instructions.
  - *LIMA (1,030) for careful, high‑quality instruction data.
  - *WizardLM Evol‑Instruct V2 (30k) to increase diversity and difficulty.
  - *Open‑Orca (30k) for GPT‑4 explanations augmenting FLAN‑style prompts.
  - *Science literature tasks (7,544) spanning QA, fact‑checking, summarization, and IE (Appendix C).
  - *Hardcoded identity prompts (140), plus filtering of references to other LLMs (e.g., “GPT‑4”) to keep self‑descriptions consistent.
- Long‑context training: maximum sequence length increased from 2,048 to 8,192 tokens. Only 20 samples are truncated in V2 vs 63,900 in V1, improving coverage of long ShareGPT/OpenAssistant conversations (Section 2, “Extended context length”; Figure 1 shows the token length histogram).

3) Supervised instruction finetuning (SFT)
- Training setup (Appendix B):
  - Precision bfloat16, 2 epochs, learning rate 2e‑5 (1e‑5 for 70B), warmup 0.03, no weight decay, effective batch size 128, max length 8,192.
- Purpose: align the base models to follow instructions with long‑context support using the V2 mixture.

4) Preference learning with DPO (optional second stage)
- Data: filtered, binarized UltraFeedback preference data (high‑quality GPT‑4‑graded comparisons) for 3 epochs (Section 2, “RLHF training”).
- Training hyperparameters (Appendix B): learning rate 5e‑7, bfloat16, batch size 32, beta 0.1, max length 8,192, warmup 0.1.
- Mechanism (intuition): for each prompt with two responses (chosen vs rejected), DPO increases the relative log‑likelihood of the chosen answer while regularizing against the base SFT policy; the `beta` parameter scales this relative preference pressure. It is an offline objective—no environment rollouts or learned reward model are needed.

5) QLoRA experiments (alternative to full SFT)
- Setup (Appendix B): 5 epochs, learning rate 1e‑4, max length 4,096, effective batch size 128; LoRA rank 64, alpha 16, dropout 0.1; adapters on attention and feedforward layers.
- Goal: estimate the compute–performance trade‑offs by replacing full finetuning with parameter‑efficient adaptation.

6) CODE TÜLU 2 (Code LLaMA + V2 mixture)
- Apply the same SFT recipe to Code LLaMA to assess coding vs general‑purpose trade‑offs (Section 3.5; Table 6).

7) Evaluation protocol
- Benchmarks (Appendix A):
  - MMLU (knowledge), GSM8k and BBH with Chain‑of‑Thought (reasoning), TyDiQA (multilingual QA, Gold Passage), HumanEval/Codex‑Eval (coding), AlpacaEval (open‑ended, GPT‑4 judge), MT‑Bench (open‑ended, GPT‑4 judge), ToxiGen (toxicity rate; lower is better), TruthfulQA (truthfulness/informativeness).
- Important controls:
  - AlpacaEval judged with GPT‑4‑0613 to ensure comparability (Section 3, “Evaluation tools”).
  - TruthfulQA is omitted for DPO‑trained models due to overlap with UltraFeedback (data contamination note in Section 3).
  - Possible training‑data overlap for proprietary models is acknowledged (Section 3).

## 4. Key Insights and Innovations
- Improved, smaller, long‑context instruction mixture (`TÜLU‑V2‑mix`)
  - Innovation: curated mix emphasizing high‑quality distilled data (e.g., LIMA, WizardLM Evol‑Instruct V2, Open‑Orca), long‑context coverage (8,192 tokens), and removal/downsampling of weaker or redundant sources (e.g., Dolly removed; FLAN downsampled) while keeping diverse conversation styles (Section 2).
  - Significance: V2 improves open‑ended generation and coding performance relative to V1 while reducing total size (326k vs 490k samples). Table 2 shows consistent gains in BBH, Codex‑Eval, AlpacaEval, and TruthfulQA across sizes, e.g., 7B AlpacaEval 73.9 vs 64.5 (+9.4), Codex‑Eval 36.9 vs 33.9 (+3.0).

- Stable DPO at 70B scale with large gains in open‑ended generation
  - Innovation: first public demonstration of DPO training that is stable and beneficial at 70B parameters, using a very low learning rate (5e‑7) and 3 epochs on UltraFeedback (Section 2; Table 3).
  - Significance: Table 3 shows large AlpacaEval gains at every scale (+11.2 at 7B, +10.6 at 13B, +8.5 at 70B), and Table 4 shows MT‑Bench improvements (e.g., 70B from 7.49 to 7.89). This establishes DPO as a simple, scalable alternative to PPO‑style RLHF.

- Clear empirical picture for QLoRA vs full finetuning on open‑ended tasks
  - Insight: QLoRA underperforms full finetuning on open‑ended generation (AlpacaEval) by large margins, though the gap narrows with model size (Table 5). Example: 7B AlpacaEval 56.1 (QLoRA) vs 73.9 (full), −17.8 points; 70B 78.6 vs 86.6, −8.0 points.
  - Significance: Parameter‑efficiency is attractive, but for open‑ended generation quality, full finetuning remains superior—an actionable guidance for practitioners.

- Leveraging code‑specialized pretraining (CODE TÜLU 2)
  - Innovation: Finetuning Code LLaMA on V2 yields substantial coding gains without proprietary data (Table 6).
  - Significance: At 7B, Codex‑Eval jumps to 68.9 vs 36.9 for the general LLAMA‑based TÜLU 2; at 13B, 76.2 vs 49.0. However, general open‑ended generation (AlpacaEval) drops notably (e.g., 7B: 58.0 vs 73.9), clarifying domain specialization trade‑offs.

Incremental vs fundamental:
- Incremental: switching to LLAMA‑2 base models, adding long‑context SFT.
- More fundamental: demonstrating scalable DPO at 70B and disentangling data/recipe choices that move open‑ended generation and coding ability in different directions.

## 5. Experimental Analysis
- Evaluation design
  - Benchmarks span knowledge (MMLU), reasoning (GSM8k/BBH), multilingual QA (TyDiQA GoldP), coding (HumanEval/Codex‑Eval), open‑ended dialogue (AlpacaEval, MT‑Bench), toxicity (ToxiGen), and truthfulness (TruthfulQA) (Appendix A).
  - Models compared include LLAMA‑2‑Chat, Xwin‑LM, Zephyr‑Beta, and proprietary GPT‑3.5/4 references (Table 1). MT‑Bench category scores are reported in detail (Table 8).

- Main quantitative results
  - Overall standing among open models
    - Quote (Table 1): “TÜLU 2 70B average 73.8,” best among open models on average; top in 3/7 tasks and within ~1% of the best open result on the rest.
  - Against GPT‑3.5 (older vs newer variants)
    - Quote (Table 1): GPT‑3.5‑turbo‑0301 average 72.3 vs TÜLU 2 70B 73.8 (TÜLU 2 slightly higher on average; better on AlpacaEval and ToxiGen).
    - Newer GPT‑3.5‑turbo‑0613 still leads across several metrics (average 77.6).
  - Effect of DPO
    - Quote (Table 3): “AlpacaEval +11.2 (7B), +10.6 (13B), +8.5 (70B).” Minimal changes on MMLU/BBH/Codex‑Eval; large multilingual drop on TyDiQA (−1.9, −13.5, −17.8).
    - MT‑Bench: 70B rises from 7.49 to 7.89, the top open‑weight result at reporting time (Table 4).
    - Verbosity: Average AlpacaEval output length increases post‑DPO (Table 4), consistent with known RLHF verbosity bias.
  - V2 mixture vs V1
    - Quote (Table 2): At 7B, average improves from 47.8 (V1) to 54.2 (V2); at 13B, 56.0 to 60.8; at 70B, 71.5 to 72.4. Improvements concentrate on BBH, Codex‑Eval, AlpacaEval, TruthfulQA; GSM8k and TyDiQA decline.
  - QLoRA vs full finetuning
    - Quote (Table 5): Average gap of −5.3 (7B), −3.6 (13B), −2.4 (70B). Largest deficits on AlpacaEval.
  - Code LLaMA finetuning
    - Quote (Table 6): Coding gains are large—e.g., at 7B, Codex‑Eval 68.9 (CODE TÜLU 2) vs 36.9 (TÜLU 2). But AlpacaEval drops—58.0 vs 73.9 at 7B.
  - MT‑Bench category breakdown (Table 8)
    - DPO at 70B especially boosts Roleplay and Writing (e.g., Roleplay rises from 8.30 to 9.25; Writing 9.15 to 9.25), with mixed changes in Coding/Math.

- Ablations, robustness, and caveats
  - Data mixture ablation is implicit in Table 2 (V1 vs V2) and ShareGPT‑only comparison at 7B; V2 surpasses ShareGPT‑only overall and even on AlpacaEval (Table 2).
  - Multilingual performance degrades with DPO; the training data are largely English (Section 3.3), so TyDiQA becomes out‑of‑distribution during preference optimization.
  - TruthfulQA contamination: UltraFeedback includes TruthfulQA prompts; results are omitted when comparing DPO‑trained models (Section 3).
  - External evaluators: AlpacaEval and MT‑Bench depend on GPT‑4; the paper locks AlpacaEval to GPT‑4‑0613 for fairness (Section 3), but notes GPT‑4 versions are not permanently pinned community‑wide.

- Do the experiments support the claims?
  - Yes for the key claims:
    - V2 mixture: clear gains on open‑ended generation and coding (Table 2).
    - DPO: consistent, large boosts in GPT‑4‑judged open‑ended metrics, stable at 70B (Tables 3–4), with known verbosity side‑effects (Table 4).
    - QLoRA trade‑offs: strong evidence of open‑ended degradation (Table 5).
    - Code specialization: marked coding gains but general chat trade‑offs (Table 6).
  - Mixed outcomes:
    - Multilingual QA (TyDiQA) worsens with DPO (Table 3), indicating alignment choices matter for multilingual capabilities.

## 6. Limitations and Trade-offs
- Data distribution and multilinguality
  - SFT and DPO data are predominantly English; multilingual QA performance drops substantially after DPO (TyDiQA −17.8 at 70B, Table 3). This limits applicability for multilingual assistants unless additional multilingual preference/SFT data are included.
- Open‑ended generation vs other skills
  - DPO strongly boosts open‑ended judged quality but may not improve factual or mathematical reasoning metrics (Table 3, small deltas on MMLU/BBH/GSM8k) and increases verbosity (Table 4).
- Parameter‑efficiency vs quality
  - QLoRA saves compute but underperforms full finetuning on open‑ended generation, especially at smaller scales (Table 5). Teams with constrained hardware face a quality trade‑off.
- Domain specialization
  - Using Code LLaMA raises coding performance but reduces open‑ended chat quality (Table 6). Specialization can harm general assistance.
- Evaluation and contamination concerns
  - GPT‑4‑based judges (AlpacaEval, MT‑Bench) may change over time; the paper fixed GPT‑4‑0613 for reported runs but broader comparisons can drift (Section 3).
  - TruthfulQA contamination in UltraFeedback means DPO‑trained models’ TruthfulQA results are not reported in cross‑model comparisons (Section 3).
  - For proprietary baselines, training‑set overlaps with benchmarks cannot be ruled out (Section 3).
- Compute requirements
  - Large‑scale training remains substantial; e.g., 70B DPO ran for ~7 days on a 512‑core TPUv3 pod (Section 3, “Training”).

## 7. Implications and Future Directions
- Field impact
  - Provides a reproducible, open pipeline showing that simple, offline preference optimization (DPO) scales to 70B and meaningfully improves GPT‑4‑judged open‑ended quality (Tables 3–4). This lowers the barrier to building strong open assistants without PPO‑style RL.
  - Offers an improved, public instruction mixture and long‑context training recipe, enabling the community to study how distilled data and context length drive performance (Section 2; Figure 1).
  - Clarifies that parameter‑efficient finetuning may not match full SFT on open‑ended tasks, guiding practitioners’ compute/budget decisions (Table 5).

- Suggested follow‑ups (some named in the Conclusion)
  - Multilingual alignment: Add multilingual SFT and multilingual preference data to recover TyDiQA and test DPO’s behavior in non‑English settings (Section 3.3).
  - RLHF method comparisons at scale: Head‑to‑head 70B‑scale comparisons of DPO vs PPO, rejection sampling (RS/ReST), and offline RL variants, including effects on refusals, verbosity, and factuality (Conclusion).
  - Data ablations: Systematically vary the proportions of distilled vs human‑written data, CoT density, and conversation length to quantify which ingredients drive which metrics (Sections 2–3.2).
  - Length/verbosity control: Incorporate length‑aware preference models or penalties to retain DPO gains while avoiding excessive verbosity (Table 4).
  - Domain‑adaptive assistants: Explore mixtures that retain general ability while selectively leveraging domain‑specialized bases (e.g., hybrid LLAMA‑2/Code LLaMA training or multi‑adapter routing; Table 6 trade‑offs).
  - Larger and newer bases: Apply the recipe to newer base models (e.g., Mistral‑family or successors) and extend beyond 70B with long‑context SFT/DPO.

- Practical applications
  - Open, high‑quality chat assistants for research and industry with strong open‑ended generation (TÜLU 2+DPO 70B, AlpacaEval 95.1 in Table 4).
  - Code‑focused copilots based on `CODE TÜLU 2`, which dramatically improves functional correctness on HumanEval (e.g., 82.5 at 34B; Table 6).
  - Educational and scientific assistants leveraging long‑context capabilities and a science‑task subset (Appendix C), within the multilingual limitations noted.

Block‑quoted highlights
- V2 mixture long‑context coverage gain: 
  > “Moving from 2,048 to 8,192 max length means we only truncate 20 (as opposed to 63,900) samples within our V2 mixture” (Section 2; Figure 1).
- DPO effect on open‑ended quality:
  > “TÜLU 2+DPO 70B … AlpacaEval 95.1 vs 86.6 without DPO; MT‑Bench 7.89 vs 7.49” (Tables 3–4).
- QLoRA trade‑off:
  > “7B AlpacaEval 56.1 (QLoRA) vs 73.9 (full); 70B 78.6 vs 86.6” (Table 5).
- Coding specialization:
  > “CODE TÜLU 2 (7B) Codex‑Eval 68.9 vs 36.9 for TÜLU 2; AlpacaEval drops to 58.0 vs 73.9” (Table 6).

Overall, TÜLU 2 provides a transparent, well‑controlled demonstration that careful data curation plus long‑context SFT and scalable DPO yields state‑of‑the‑art open‑weight instruction‑following models, with clear guidance on when parameter‑efficient methods and domain specialization help or hurt.
