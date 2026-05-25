# Tülu 3: Pushing Frontiers in Open Language Model Post-Training

**ArXiv:** [2411.15124](https://arxiv.org/abs/2411.15124)

## 🎯 Pitch

Tülu 3 introduces a fully open, state-of-the-art framework for post-training language models, releasing not only powerful models (8B, 70B, 405B parameters) but also the complete data, code, and a transparent, decontaminated evaluation suite. By pioneering a multi-stage approach—combining supervised finetuning, Direct Preference Optimization, and the novel Reinforcement Learning with Verifiable Rewards (RLVR)—Tülu 3 closes the performance gap with leading closed models and sets a new standard for transparency, reproducibility, and community-driven advancement in language model alignment and evaluation.

---

## 1. Executive Summary (2-3 sentences)
Tülu 3 is a fully open, state‑of‑the‑art post‑training recipe and model family (8B, 70B, 405B parameters) that combines supervised finetuning (SFT), Direct Preference Optimization (DPO), and a new reinforcement‑learning stage called Reinforcement Learning with Verifiable Rewards (`RLVR`). It closes much of the gap between open and closed instruction‑tuned models by releasing all data, code, infrastructure, and a rigorous evaluation suite with decontamination, and it achieves best‑in‑class open‑weight performance while rivaling small closed models (see Table 2, Table 4).

## 2. Context and Motivation
- Problem addressed
  - Open‑weight language models lack transparent, modern post‑training recipes (data + code + evaluation), leaving them behind closed models that use multi‑round instruction tuning, preference learning, and RL (Section 1; Section 2; Table 1).
  - The most important part of post‑training—data mixtures and recipes—is often undisclosed; open data also frequently contaminates evaluations (i.e., overlaps with test questions), leading to inflated results (Section 3.2; Table 37).
- Why it matters
  - Practical impact: reproducible, high‑performing open models enable research, productization, and safety auditing without vendor lock‑in.
  - Scientific impact: standardized, decontaminated evaluations and full recipes allow controlled study of how SFT, preference tuning, and RL interact.
- Prior approaches and gaps
  - Open recipes like Tülu 2 and Zephyr‑β showed strong chat quality but lagged on core skills such as math (MATH, GSM8K) and precise instruction following (IFEval) (Section 2; Table 3).
  - Closed systems (e.g., Llama 3.1 Instruct, GPT‑4o, Claude 3.5) use large‑scale data mixtures and multi‑stage training, but with limited transparency (Section 2; Table 2).
- This paper’s position
  - A comprehensive, open blueprint: new datasets (including synthetic, persona‑driven prompts), a decontaminated evaluation regime (Tülu 3 Eval + OLMES toolkit), an expanded preference pipeline with on‑policy data, and a new RL stage (`RLVR`) with verifiable rewards (Figure 1; Sections 3–7). Models, datasets, and code are released (Table 1).

## 3. Technical Approach
The recipe has four staged components, each with clear design choices and tooling (Figure 1; Section 2.3).

1) Data Curation and Decontamination (Section 3)
- What is curated:
  - A pool of millions of prompts combining high‑quality public data and persona‑driven synthetic data that target core skills: knowledge recall, reasoning, math, coding, precise instruction following, general chat, and safety (Table 3; Table 7).
  - Synthetic prompt generation is guided by “personas” to avoid mode collapse and increase diversity (Section 3.1.2): prompts for math, coding, and precise instruction following are generated with GPT‑4o, conditioned on ∼250k personas (Figures 30–36).
- Safety and non‑compliance data:
  - Curated and synthetic adversarial prompts, benign prompts, and contrastive prompts (CoCoNot) to avoid over‑refusal (Section 3.1.2).
- Decontamination (“removing training–test leakage”):
  - 8‑gram overlap with >50% token coverage on the same training instance indicates a match; datasets with >2% overlap with any evaluation are filtered or removed; specific decontaminated datasets are provided (Section 3.2; Table 8). A contamination survey of popular public sets is reported (Table 37).

2) Supervised Finetuning (SFT) (Section 4)
- From prompts to responses:
  - Keep high‑quality human/frontier‑model responses (e.g., GPT‑4o); generate new completions if originals come from weaker models; filter meta‑info and empty answers (Section 4.1.1).
- Mixture design:
  - Build skill‑specific mixes (e.g., math‑specialized) to set “upper bounds,” then combine and iterate with decontamination to form the final multi‑skill SFT mix (Section 4.1.2; Figure 3; Figure 2 shows length distribution).
- Key training choices:
  - Use “sum loss” instead of the standard “mean loss” to fix a known bug in loss aggregation with gradient accumulation and padding (Section 4.3.2; Equations (1)–(2)): this avoids weighting short sequences disproportionately.
  - 2 epochs, context length 4,096, effective batch 128, LR 5e‑6 (8B) / 2e‑6 (70B) (Table 11).
  - Compute: 8B on 32 H100s for ~6h; 70B on 64 H100s for ~50h (Section 4.3).

3) Preference Tuning (DPO) (Section 5)
- Data pipeline extending UltraFeedback (Figure 7; Section 5.2.1):
  - Prompt selection: reuse SFT prompts and add unused prompts from the same sources, plus new IF‑augmented prompts.
  - Response generation: sample 4 responses from a 22‑model pool (open + closed) and on‑policy completions from Tülu‑SFT to ensure the model learns from its own behavior distribution (on‑policy).
  - Preference labels: an LLM judge (primarily GPT‑4o‑2024‑08‑06) rates each response (helpfulness, instruction‑following, honesty, truthfulness); pairs are binarized by taking the highest‑rated as “chosen” and a lower‑rated as “rejected” (Appendix D).
- Algorithm:
  - Length‑normalized DPO (Section 5.1.2, Eq. (6)): standard DPO trains a policy to prefer chosen over rejected responses relative to a reference policy. Tülu 3 divides the log‑probabilities by sequence length to reduce length bias. This variant outperformed both vanilla DPO and SimPO in their setup (Table 18).
- Efficiency/infrastructure:
  - Cache reference log‑probs and compute chosen/rejected forward passes separately to cut GPU memory (Figure 17).
  - Hyperparameters: LR 5e‑7 (8B), 2e‑7 (70B); β=5 (KL penalty coefficient); batch 128; 1 epoch; max length 2,048 (Table 20).
  - Runtime: 8B ~10h on 8×H100; 70B ~19h on 64×H100 (Section 5.4.1).
- Preference mixes:
  - 8B best mix: 271k instances; 70B best mix: 334k instances, combining SFT‑reused on‑/off‑policy, WildChat (reused/unused), UltraFeedback, and persona IF data (Table 15).

4) Reinforcement Learning with Verifiable Rewards (`RLVR`) (Section 6)
- Idea (how it works):
  - Use a simple, task‑specific, deterministic verifier as the reward function `v(x,y)` instead of a learned reward model; give a fixed positive reward `α` (set to 10) if the model’s answer is correct, else 0 (Eq. (8)). Optimize the usual RLHF objective with KL penalty to a reference policy via PPO (Eq. (7)).
  - “Verifiable” means correctness can be programmatically checked (e.g., GSM8K/MATH answers, or whether output satisfies an instruction constraint in IFEval).
- RLVR training data:
  - 29,946 prompts: GSM8K train (8‑shot CoT prompting), MATH train (4‑shot CoT), and programmatically verifiable IFEval constraints (Table 22; Section 6.1).
- Stabilization details:
  - Initialize PPO’s value function from a general reward model; disable dropout; add a −10 penalty for responses without EOS; advantage whitening; shuffle across epochs (Section 6.2).
- Infrastructure and scale:
  - Asynchronous RLHF: dedicated inference GPUs via vLLM PagedAttention and dedicated learner GPUs; use Ray for allocation (Section 6.3).
  - 8B RL run ~65h on 8×H100; 70B ~60h on 48×H100; 405B ~46h on 256×H100 (Section 6.3; Section 8.1).
  - 405B: 16‑way TP inference for vLLM + training on remaining GPUs; checkpoints synchronized by NCCL broadcast (Section 8.1).

5) Evaluation Framework (Section 7)
- OLMES: open, reproducible evaluation toolkit with task configs and instance‑level outputs (Section 7.1).
- Split into development vs unseen suites (Table 24):
  - Development covers core skills (e.g., MMLU 0‑shot CoT with a “summarize” CoT prompt shown to help heterogeneous subjects; Table 46; Section 7.2).
  - Unseen suite tests generalization with different formulations: MMLU‑Pro, GPQA, AGIEval English (0‑shot CoT), DeepMind Mathematics (0‑shot CoT with answer‑format heuristics), BigCodeBench, and two new evaluations—IFEval‑OOD (52 novel constraints; Appendix F.3) and HREF (11 instruction‑following tasks, mixed LM‑judge and embedding‑based scoring; Section 7.3.2).

## 4. Key Insights and Innovations
- A. Fully open, modern post‑training recipe at scale with explicit decontamination
  - What’s new: a complete, reproducible pipeline—datasets, code, models, and evaluation tooling (Table 1). Decontamination is enforced with a transparent 8‑gram method and removal thresholds (Section 3.2; Table 8; Table 37).
  - Why it matters: enables fair benchmarking, avoids overstated gains, and allows others to adapt/extend the recipe.
- B. Persona‑driven synthetic data targeted at core skills
  - What’s new: scalable, persona‑conditioned generation for precise instruction following, math, and coding (Section 3.1.2).
  - Why it matters: improves targeted capabilities beyond generic chat (Table 10 shows removing persona data hurts IFEval, GSM8K, MATH; Section 4.2).
- C. Scaled on‑policy preference data and length‑normalized DPO
  - What’s new: preference pairs include on‑policy completions to reduce distribution shift (Figure 11); length‑normalized DPO outperforms DPO/SimPO in their setting (Table 18).
  - Why it matters: improved average performance and instruction following (e.g., IFEval gains from IF‑persona and IF‑augmented preferences; Figure 14; Table 15).
- D. RLVR: RL with verifiable rewards
  - What’s new: PPO on simple, binary, programmatic rewards for domains with clear correctness (GSM8K, MATH, IFEval) (Section 6).
  - Why it matters: consistently improves targeted tasks without training a reward model, and scales to 405B (Figures 19–23; Table 23; Table 4).
- E. Evaluation design with “unseen” suite and new benchmarks
  - What’s new: IFEval‑OOD (52 new constraints) and HREF (11 instruction‑following subtasks with human‑guided judge selection and design) (Sections 7.3.1–7.3.2; Table 48).
  - Why it matters: tests generalization beyond common benchmarks and guards against over‑fitting to specific constraint taxonomies.

## 5. Experimental Analysis
- Methodology and setup
  - Development and unseen suites cover knowledge, reasoning, math, coding, instruction following, and safety (Table 24).
  - Safety is a macro‑average across multiple benchmarks with automatic refusal/compliance classification (Section 7.2.1; Tables 25–26 provide breakdowns).
  - Extensive ablations on data (e.g., removing WildChat/persona/math data), algorithms (DPO variants vs PPO), hyperparameters (LR, β for PPO/DPO), and infrastructure choices (caching reference log‑probs) (Sections 4–6).
- Main quantitative results
  - Overall performance at 8B and 70B (development suite):
    > Table 2: “Tülu 3 8B” average 65.1 vs Llama‑3.1‑8B‑Instruct 62.9; “Tülu 3 70B” average 76.2 vs Llama‑3.1‑70B‑Instruct 74.1. Tülu 3 70B also outperforms Qwen‑2.5‑72B‑Instruct (72.8) and is competitive with small closed models (e.g., Claude 3.5 Haiku 75.3; GPT‑4o‑mini 69.6).
  - Stage‑wise gains at 8B (Table 6):
    > Average: 60.1 (SFT) → 64.7 (DPO) → 65.1 (RLVR).  
    > GSM8K: 76.2 → 84.3 → 87.6.  
    > IFEval: 72.8 → 81.1 → 82.4.  
    > MATH: 31.5 → 42.0 → 43.7.
  - Stage‑wise gains at 70B (Table 23):
    > Average: 73.4 (Llama‑3.1‑Inst.) → 75.9 (Tülu 3 DPO) → 76.0 (Tülu 3 RLVR).  
    > MATH: 56.4 (Llama‑3.1‑Inst.) → 62.3 (DPO) → 63.0 (RLVR).  
    > IFEval: 88.0 (Llama‑3.1‑Inst.) → 82.6 (DPO, formatting sensitivity) → 83.2 (RLVR).  
    > GSM8K stays high (93.5–93.7).
  - 405B model (Table 4):
    > Average with safety: 81.6 (GPT‑4o 11‑24), 79.0 (DeepSeek V3) vs 80.7 (Tülu 3 RLVR).  
    > MATH: 66.6 (Llama‑3.1‑Instruct) vs 67.3 (Tülu 3 RLVR).  
    > GSM8K: 95.4 (Llama‑3.1‑Inst.) vs 95.5 (Tülu 3 RLVR).  
    > IFEval: 88.4 (Llama‑3.1‑Inst.) vs 86.0 (Tülu 3 RLVR); DPO improves IFEval to 85.0 and RLVR further to 86.0 from SFT’s 82.4.
- Unseen suite and generalization
  - Pipeline progression generalizes (Table 31): final checkpoints typically best on both development and unseen for each skill, e.g., math improves in DeepMind Mathematics as well as MATH; instruction following improves on IFEval‑OOD as well as IFEval.
  - Cross‑model comparison on unseen tasks (Table 33): at both 8B and 70B, Tülu 3 generally sits between Llama‑3.1‑Instruct and Hermes‑3, winning in several subtasks; HREF breakdown shows mixed strengths across categories (Table 48).
- Safety
  - SFT strongly boosts safety:
    > Table 25 (8B): overall safety average 93.1 (Tülu 3 SFT) vs 75.2 (Llama‑3.1‑8B‑Instruct).  
    > Table 26 (70B): 94.4 (Tülu 3 SFT) vs 76.5 (Llama‑3.1‑70B‑Instruct).  
    DPO and RLVR maintain high safety with slight regressions.
- Ablations and diagnostics
  - Data ablations (Table 10):
    > Removing WildChat reduces average and notably hurts AlpacaEval (e.g., LC winrate 12.4 → 7.5), indicating “in‑the‑wild” chat data helps general chat quality.  
    > Removing persona data hurts IFEval, GSM8K, and HumanEval+.  
    > Removing math data drops MATH substantially (31.5 → 23.5), illustrating targeted SFT data is crucial for math.
  - Scaling SFT size (Figure 4): average performance increases with more SFT data; TruthfulQA shows a small decline, reflecting alignment trade‑offs across tasks.
  - Base model choice (Table 12): larger bases and ones with math pretraining (Qwen‑2.5‑Math) yield better math after SFT.
  - Chat template (Table 13): minor template changes affect results; the team chose a simple, consistent template to avoid generation inconsistencies downstream.
  - DPO design (Tables 18–20): length‑normalized DPO outperforms vanilla DPO/SimPO in their setup; LR 2e‑7 works best at 70B for their best mix; caching reference log‑probs reduces memory (Figure 17).
  - On‑policy preferences help (Figure 11); unique prompts matter more than duplicating prompts (Figures 8–9).
  - Judge choice (Table 17): GPT‑4o, Llama‑3.1‑405B, and GPT‑4‑Turbo perform similarly for annotation quality; GPT‑4o slightly leads in their setup.
  - RLVR dynamics:
    - Improves targeted tasks (GSM8K, MATH, IFEval) with rising train‑set verifiable rewards (Figure 19).
    - Initializing value function from a general reward model worked best (Figure 21).
    - Adding reward‑model scores to verifiable rewards adds noise; pure verifiable rewards work better (Figure 22).
    - Over‑optimization risks appear as KL grows; average scores can drop if divergence from reference becomes large (Figures 21–22; Appendix B.4 shows IFEval over‑optimization examples).
- Do the experiments support the claims?
  - Yes, on three fronts:
    - Performance: consistent stage‑wise gains and competitive results vs strong open and some closed baselines (Tables 2, 4, 5, 6, 23).
    - Method effectiveness: ablations isolate the impact of WildChat, persona, math SFT data, on‑policy preferences, DPO design, and RLVR choices (Tables 10, 12, 18; Figures 8–15, 19–22).
    - Generalization: unseen suite confirms improvements are not limited to development benchmarks (Table 31).

## 6. Limitations and Trade-offs
- Scope of verifiable RL:
  - `RLVR` applies to tasks with clean programmatic checks (math answers, constraint satisfaction). Many real tasks (open‑ended writing, multi‑hop QA without canonical answers) lack such verifiers.
  - Over‑optimization: aggressively increasing KL budget can degrade overall averages or yield “gaming” behavior on constraint checks (Figures 21–22; Appendix B.4).
- Evaluation sensitivities:
  - Some metrics depend on strict formatting and answer extraction heuristics (e.g., IFEval prompt strictness, MATH answer formatting; Section 7.2; “flex” extraction is needed).
  - Even with decontamination, n‑gram methods can miss paraphrased leakage (Section 3.2 acknowledges paraphrase ambiguity).
- Data and annotation sources:
  - Substantial reliance on synthetic data (GPT‑4o for generation; GPT‑4o or similar for judging) introduces model‑specific biases; off‑policy responses mostly from strong but finite model pools.
- Compute and engineering complexity:
  - Large training runs (e.g., 70B SFT ~50h on 64×H100; 70B RLVR ~60h on 48×H100; 405B RLVR 256×H100 with weight broadcasting and asynchronous inference/training) are non‑trivial to reproduce (Sections 4.3, 6.3, 8.1).
- Coverage:
  - The paper focuses on English, short to medium contexts, and single‑turn prompts; long‑context, multi‑turn, multilingual, tool‑use and agentic behaviors are explicitly left for future work (Section 8.3).

## 7. Implications and Future Directions
- How this work changes the landscape
  - Provides a transparent, modern, end‑to‑end post‑training recipe—including data, code, and evaluation—that achieves competitive performance with strong closed models at similar sizes (Tables 2 and 4). This sets a new bar for openness and rigor in the community.
  - Establishes good practice on decontamination and development vs unseen evaluation, with the OLMES toolkit enabling reproducibility (Section 7.1; Table 24).
  - Introduces `RLVR` as a practical, scalable alternative to reward‑model‑driven RL for verifiable tasks; shows it can be combined with SFT and DPO and scale to 405B (Sections 6 and 8.1).
- Follow‑up research enabled
  - Extending RLVR beyond math and constraint following—e.g., code execution feedback, tool use, program repair, and retrieval‑augmented tasks with automatic verifiers (Section 6.1; related work Section 9.2).
  - Richer preference signals: online DPO with better reward models, rejection sampling with stronger judges (Section 8.2 notes preliminary, inconclusive attempts), and step‑wise reasoning rewards.
  - Improved evaluation: broader unseen suites, fairness and robustness probing, multilingual versions, and long‑context, multi‑turn scenarios (Section 8.3).
- Practical applications
  - Organizations can adapt the released recipe to post‑train base LMs for domain‑specific assistants (e.g., scientific QA with SciRIFF, safety‑constrained customer support with WildGuardMix and CoCoNot, regulated industries requiring verifiable outputs).
  - Academics can use OLMES + decontaminated datasets to run fair head‑to‑head comparisons and study the effects of data and algorithmic design decisions at scale.

> In short: Tülu 3 demonstrates that with careful data curation (including persona‑driven synthesis and decontamination), scaled on‑policy preferences with length‑normalized DPO, and RLVR on verifiable signals—plus robust evaluation—open models can reach or surpass the best open baselines and rival small closed systems (Tables 2, 4, 5, 6, 23), while leaving a reproducible trail others can follow and extend.
