# TÜLU 3: Pushing Frontiers in Open Language Model Post‑Training

**ArXiv:** [2411.15124](https://arxiv.org/abs/2411.15124)
**Authors:** Nathan Lambert, Jacob Morrison, Valentina Pyatkin, Shengyi Huang, Hamish Ivison, Faeze Brahman, Lester James V. Miranda, Alisa Liu, Nouha Dziri, Shane Lyu, Yuling Gu, Saumya Malik, Victoria Graf, Jena D. Hwang, Jiangjiang Yang, Ronan Le Bras, Oyvind Tafjord, Chris Wilhelm, Luca Soldaini, Noah A. Smith, Yizhong Wang, Pradeep Dasigi, Hannaneh Hajishirzi
**Institutions:** Allen Institute for AI

## 🎯 Pitch

Tülu 3 redefines transparency in language model development by releasing an open post-training recipe, incorporating novel RL with Verifiable Rewards (RLVR) to enhance reproducibility and performance, surpassing some closed competitors. This innovation empowers researchers and practitioners with high-quality, trustworthy models and methodologies, paving the way for significant advancements in applications like safe chat systems, coding, and math reasoning without reliance on opaque, proprietary systems.

---

## 1. Executive Summary
Tülu 3 is a fully open post‑training recipe and model family (8B, 70B, 405B parameters) that combines supervised finetuning (SFT), preference optimization (DPO), and a new reinforcement‑learning stage called RL with Verifiable Rewards (RLVR). It closes a major transparency and performance gap by releasing models, code, datasets, training and evaluation infrastructure, and it achieves state‑of‑the‑art results among open models—reaching or surpassing several strong closed baselines (e.g., GPT‑4o‑mini, Claude 3.5‑Haiku) on a broad skill suite (Table 2).

## 2. Context and Motivation
- Problem addressed
  - Modern language models require “post‑training” (instruction tuning, preference learning, RLHF‑style optimization) to behave helpfully and follow instructions. However, the most successful recipes are opaque: data, code, and procedures are not openly released, which hinders reproducibility and progress (Section 1).
  - Open efforts have existed (e.g., Tülu 2, Zephyr‑β), but they typically rely on simpler pipelines, smaller or lower‑quality datasets, and limited evaluation rigor. They trail closed systems on core capabilities like math, precise instruction following, and safety (Section 2; Table 6 and Table 5 baselines).

- Why it matters
  - Practical impact: high‑quality, reproducible post‑training enables labs and practitioners to adapt base models for real applications—coding, math reasoning, safe chat—without relying on closed APIs.
  - Scientific impact: full release of datasets, decontamination tooling, training code, and an evaluation framework allows systematic comparison and ablation across methods and scales (Table 1; Section 7).

- Prior approaches and gaps
  - Typical recipe: SFT → RLHF (or DPO variants). Gaps: limited data transparency; uncertain contamination with test sets; narrow evaluations; few ablations on algorithmic and infrastructure decisions (Section 2).
  - Specific weaknesses the paper targets: weak math and instruction‑following performance in open recipes; limited scaling of preference data; little clarity on training‑time pitfalls (e.g., loss aggregation) (Sections 3–6).

- Positioning
  - Tülu 3 offers a complete, open pipeline spanning:
    - Curated and synthetic data targeting core skills with aggressive decontamination (Sections 3.1–3.2; Table 7; Table 8).
    - Multi‑stage training with extensive ablations (SFT in Section 4; DPO in Section 5).
    - A novel, generalist RL stage—RLVR—that uses task verifiers instead of a learned reward model (Section 6).
    - A standardized development vs. unseen evaluation suite and toolkit (Section 7; Table 24).

## 3. Technical Approach
High‑level recipe (Figure 1): curate prompts → SFT on prompt–completion pairs → DPO on preference pairs → RLVR on verifiable tasks → evaluate on development and unseen suites while decontaminating training data against them.

1) Data curation and decontamination (Section 3)
- Core skills targeted: knowledge recall, reasoning, math, coding, precise instruction following (IF), general chat, and safety (Table 3).
- Sources:
  - Public datasets with clear licenses (e.g., WildChat, OpenAssistant, FLAN v2, NuminaMath‑TIR, OpenMathInstruct2, Evol‑CodeAlpaca, Aya, SciRIFF, TableGPT) and AI2‑generated persona‑driven synthetic datasets (precise IF, math, coding) (Table 7; Sections 3.1.1–3.1.2).
  - Safety and non‑compliance prompts gathered and synthesized (CoCoNot, WildGuardMix, WildJailbreak) (Section 3.1.2).
- Persona‑driven synthesis: use ~250K personas from Persona Hub to generate diverse instructions for math, coding, and verifiable IF; completions produced by strong models (GPT‑4o, Claude‑3.5‑Sonnet) (Section 3.1.2; Figures 30–36).
- Decontamination: 8‑gram overlap at the prompt level; remove datasets with >2% overlap with an evaluation; remove overlapping instances when necessary (Section 3.2). Released decontaminated versions for several public sets (Table 8).

2) Supervised Finetuning (SFT) (Section 4)
- Data: ~0.94M curated prompt–completion pairs (Table 7; Figure 2 for length distribution).
- Training setup: Llama‑3.1 base models (8B, 70B); 2 epochs; effective batch 128; max length 4096; LR 5e‑6 (8B) and 2e‑6 (70B) (Table 11). Compute: 8B on 32 H100 GPUs for ~6h; 70B on 64 H100s for ~50h (Section 4.3).
- Critical engineering fix—loss aggregation: default “mean loss” across padded tokens interacts badly with gradient accumulation/distributed training. They switch to “sum loss,” re‑tuning LR, which yields better stability and performance (Section 4.3.2; Figures 5–6).
- Chat template choice matters: removing a trailing newline (their final “Tülu 3” template) avoids later inconsistencies while staying competitive with alternatives (Section 4.3.1; Table 13).

3) Preference Tuning (DPO) with scalable on‑policy data (Section 5)
- Data creation pipeline (Figure 7):
  - Stage 1: select prompts (some reused from SFT; some subsampled but unused; some new IF‑augmented prompts).
  - Stage 2: generate 4 responses per prompt from a model pool (22 models, including the on‑policy `Tülu 3 SFT` model and various external models; Appendix Table 38).
  - Stage 3: judge each response on helpfulness, instruction‑following, honesty, truthfulness using `GPT‑4o‑2024‑08‑06`; binarize to chosen vs rejected for DPO (Section 5.2.1).
- Final preference mix sizes: 8B uses 271K pairs; 70B uses 334K (Table 15).
- Algorithm: length‑normalized DPO (divide log‑likelihood by response length to reduce length bias), which consistently outperforms vanilla DPO and SimPO in their setting (Section 5.4.1; Table 18). Final hyperparameters: LR 5e‑7 (8B) / 2e‑7 (70B); β=5; 1 epoch; effective batch 128; max length 2048 (Table 20).
- Infrastructure optimizations for 70B: cache reference log‑probs; run chosen/rejected forwards separately to cut peak GPU memory—yields near‑identical losses with much lower memory (Section 5.4.2; Figure 17).

4) Reinforcement Learning with Verifiable Rewards (RLVR) (Section 6)
- Idea: use a deterministic verifier (`v(x,y)`) that gives a fixed positive reward `α` if the generated answer is correct/constraint‑satisfying, else 0. Optimize the standard KL‑regularized PPO objective with this reward, avoiding reward‑model pitfalls (Eq. 7–8; Section 6).
- Verifiable tasks and training prompts (Table 22):
  - GSM8K train (grade‑school math); extract final numeric answer with 8‑shot CoT prompt in the input.
  - MATH train (competition math); 3‑shot CoT; flexible answer extraction (“flex”) during evaluation.
  - IFEval constraints (precise IF); verifier functions for each constraint type.
- PPO setup highlights (Section 6.2):
  - Initialize the value function from a general reward model trained on UltraFeedback (Table 36), which performs best vs alternative initializations (Figure 21).
  - Disable dropout in policy and reference models to keep log‑probs consistent across rollout and learning phases.
  - Penalty when responses do not end with EOS; advantage normalization; shuffle prompts across epochs.
  - Asynchronous RL infrastructure: inference on dedicated GPUs via vLLM PagedAttention; learners run concurrently; scale to 405B with ZeRO‑3 + Ray orchestration (Section 6.3). Typical runtimes: 8B RLVR ~65h on 8 H100s; 70B ~60h on 48 GPUs (Section 6.3).
- Hyperparameters (Table 21): effective batch sizes up to 640 (70B); KL β sweeps; response length up to 2048; reward `α=10`.

5) Evaluation framework (Section 7)
- OLMES toolkit for reproducible runs and consistent prompting (Section 7.1).
- Two suites (Table 24):
  - Development: MMLU (zero‑shot CoT with “summarize reasoning” prompt), PopQA, TruthfulQA (MC2), BBH (3‑shot CoT), DROP (3‑shot), GSM8K (8‑shot CoT), MATH (4‑shot CoT; “flex” extraction), HumanEval/+, IFEval (prompt‑level accuracy), AlpacaEval 2 (length‑controlled win‑rate), safety suite (six datasets scored by WildGuard or refusal metrics; Section 7.2.1; Table 25–26).
  - Unseen: MMLU‑Pro, GPQA, AGIEval‑English, DeepMind Mathematics (zero‑shot “concise reasoning” prompt + SymPy equivalence), BigCodeBench‑Hard, new IFEval‑OOD (52 out‑of‑distribution constraints; Appendix F.3), new HREF (11 IF subtasks with human references; Section 7.3.2; Table 48).

## 4. Key Insights and Innovations
- RL with Verifiable Rewards (RLVR) for general post‑training (Section 6)
  - Novelty: replaces a learned reward model with explicit verifiers across multiple domains (math and constraint following), integrating into a general training pipeline beyond math‑only RL (contrast with VinePPO, STaR/Quiet‑STaR; Section 9.2).
  - Significance: targeted and reliable improvements on verifiable tasks without reward‑model brittleness—8B gains on GSM8K (+3.3 points over DPO to 87.6) and IFEval (+1.3 to 82.4) while improving MATH (+1.7 to 43.7) (Table 23).

- Scalable, on‑policy preference data generation (Section 5.2)
  - Novelty: large, mixed on‑policy/off‑policy preference sets created with a unified pipeline (Figure 7), at scale (>270K pairs per model; Table 15).
  - Significance: clear empirical gains from more unique prompts (Figure 8), from including on‑policy generations (Figure 11), and from targeted IF preference sets (Figure 14). This shows how to move beyond UltraFeedback while keeping costs manageable.

- Aggressive decontamination + development/unseen eval split (Sections 3.2 and 7)
  - Novelty: systematic 8‑gram prompt‑level matching with dataset removals where overlap exceeds 2%, plus decontaminated releases (Table 8), and explicit unseen evaluation suite (Table 24).
  - Significance: reduces overfitting risk and allows measurement of generalization. For instance, Tülu 3 improves on unseen DeepMind Math and AGIEval relative to its development gains (Table 31).

- Practical training/infrastructure guidance (Sections 4.3.2, 5.4.2, 6.3)
  - Loss‑aggregation fix (sum‑loss) prevents subtle weighting bugs (Section 4.3.2).
  - GPU memory reductions in DPO via cached reference log‑probs and split forwards (Figure 17).
  - Asynchronous RL layout with vLLM + ZeRO‑3 + Ray scales to 405B (Section 6.3; Section 8.1).

These go beyond incremental tuning—RLVR and the data scaling pipeline are conceptual advances; the decontamination/evaluation rigor and infrastructure lessons are broadly reusable.

## 5. Experimental Analysis
- Evaluation methodology
  - Core development suite and unseen suite (Table 24). Chain‑of‑thought prompting is used selectively (e.g., MMLU zero‑shot “summarize reasoning” prompt; Section 7.2) with robust answer extraction (e.g., MATH “flex”; Section 7.2).

- Main results (8B and 70B; Table 2)
  > Tülu 3‑70B average (development suite): `76.2`, beating Llama‑3.1‑70B‑Instruct (`74.1`) and Qwen‑2.5‑72B‑Instruct (`72.8`), and approaching Claude‑3.5‑Haiku (`75.3`) and GPT‑4o‑mini (`69.6`).  
  > Tülu 3‑8B average: `65.1`, above Llama‑3.1‑8B‑Instruct (`62.9`) and near Qwen‑2.5‑7B‑Instruct (`66.5`).

  Highlights by skill (70B; Table 2):
  - Math: MATH `63.0` vs `56.4` (Llama‑3.1‑70B‑Inst); GSM8K `93.5` (parity with Llama‑3.1‑70B‑Inst `93.7`).
  - Instruction following: IFEval `83.2` vs `88.0` (Llama‑3.1‑70B‑Inst), but higher AlpacaEval‑2 win‑rate `49.8` vs `33.4`.
  - Safety: strong overall safety (`88.3` vs `76.5`, Table 2; detailed breakdowns in Table 26).

- Stage‑wise gains (70B; Table 5) and (8B; Table 6)
  - SFT → DPO → RLVR tracks incremental improvements, especially in targeted domains. For 8B, GSM8K rises from `76.2` (SFT) → `84.3` (DPO) → `87.6` (RLVR), and IFEval from `72.8` → `81.1` → `82.4` (Table 6).
  - At 70B, DPO boosts MATH (`53.7`→`62.3`) and GSM8K stays saturated (`93.5`), RLVR yields modest further MATH/IF gains (`63.0`, `83.2`) (Table 23).

- 405B scaling (Table 4; Section 8.1)
  > Tülu 3‑405B (RLVR) achieves average with safety `80.7`, outperforming Llama‑3.1‑405B‑Instruct (`79.0`) and Nous‑Hermes‑3‑405B (`73.5`), and competitive with DeepSeek‑V3 (`75.9`) and GPT‑4o (11‑24) (`81.6`).

- Unseen‑suite generalization (Table 31)
  > The final 8B and 70B models improve the unseen averages relative to SFT and DPO stages. For 70B unseen, averages are `44.4` (DPO) and `44.4` (RLVR), beating SFT `41.0`.

  - HREF (instruction following with human references): at 70B, Tülu 3 scores `42.3`, below Llama‑3.1‑70B‑Instruct `45.6` but above Hermes‑3‑70B `36.8` (Table 33; subtask breakdown Table 48). Shows instruction‑following is multi‑faceted; distributional differences matter.
  - IFEval‑OOD: all models drop vs IFEval, indicating over‑specialization to known constraints. Tülu 3‑70B gets `27.8` vs Llama‑3.1‑70B‑Instruct `34.5` (Table 33).

- Ablations and diagnostics
  - SFT data ablations show:
    - Removing WildChat hurts AlpacaEval and degrades many skills (Table 10).
    - Removing safety data barely affects non‑safety metrics but reduces safety average substantially (Table 10).
    - Persona datasets materially help the skills they target: MATH/GSM8K/HumanEval(+)/IFEval drop when they are removed (Table 10).
  - Preference data ablations show:
    - More unique prompts → consistent gains (Figure 8); duplicating prompts without new content does not (Figure 9).
    - On‑policy data helps over off‑policy alone (Figure 11).
    - GPT‑4o is a slightly better judge, but several judges are similar (Table 17).
    - IF‑targeted preference sets (`Persona IF` + `IF‑augmented`) improve IFEval, with modest average trade‑offs; best mix balances both (Figure 14; Table 16).
  - RLVR ablations (Figures 19–22):
    - Verifiable rewards on GSM8K/MATH/IFEval yield higher training rewards and test gains.
    - Initialization of the value function from a general RM is best (Figure 21).
    - Adding RM scores to verifiable rewards is noisier and worse (Figure 22).
    - Larger KL divergence can reduce overall averages—“over‑optimization” trade‑off (Figures 19–22; Appendix B.4 shows pathological outputs for constraints).

- Safety evaluation
  - Strong refusal on harmful content while maintaining compliance on benign prompts (Tables 25–26; categories in Tables 39–40). E.g., 70B SFT reaches `94.4` safety average.

Overall, the experiments convincingly back the claims: staged training improves broad capabilities; RLVR is an effective, simple RL addition; data choices and infrastructure details measurably matter.

## 6. Limitations and Trade-offs
- Scope of verifiable RL
  - RLVR relies on tasks with reliable verifiers (math final answers, explicit constraints). It does not directly cover open‑ended dialog quality, nuanced safety, or multi‑step tool‑use without additional verifiers (Section 6.1). Over‑optimization can appear when KL is too low/high (Figures 19–22; Appendix B.4).

- Evaluation coverage and distribution shift
  - Despite an unseen suite, some generalization gaps remain: e.g., IFEval‑OOD drops across models (Table 33), and DeepMind Math formatting differences can interact with CoT behavior (Section 7.4.1).

- Judge and metric dependence
  - Preference data depends on LLM‑as‑a‑judge (mainly GPT‑4o). Although alternatives yield similar results (Table 17), this can encode judge biases. Some evaluations (AlpacaEval 2, HREF subsets) rely on LLM judges or embedding similarity (Sections 7.2, 7.3.2).

- Compute and engineering complexity
  - Training large models with DPO and RLVR requires substantial compute and careful engineering (Sections 4.3, 5.4.2, 6.3). The 70B and 405B runs needed dozens to hundreds of H100 GPUs with distributed inference/training orchestration and careful reliability handling (Section 8.1).

- Data constraints and contamination risk
  - Although decontamination is systematic (Section 3.2), paraphrase‑level contamination is difficult to detect perfectly; the method chooses precision over catching every paraphrase (embedding‑based checks were less reliable for this purpose).

- Mixed or conditional results
  - RLVR at 70B showed small average improvements and unusually low KL (Table 23; Figure 23), suggesting tuning sensitivity. Gains in one dimension (e.g., IFEval) can trade off with averages if KL is not balanced (Figure 21).

## 7. Implications and Future Directions
- Field impact
  - Tülu 3 sets a new reproducibility bar for post‑training: full data, code, decontamination tooling, and evaluation regime (Table 1). The release enables apples‑to‑apples comparisons and faster research progress on methods and data.

- Research directions
  - Richer verifiers and RL: extend RLVR to code execution feedback, tool‑use, or multi‑step verifiable workflows (Section 9.2; Section 8.3). Explore value‑model‑free variants or adaptive KL schedules to mitigate over‑optimization.
  - Preference data science: continue scaling unique, on‑policy prompts; study judge diversity; explore curriculum mixing by skill and difficulty (Figures 8–16).
  - Long‑context and multi‑turn post‑training: Tülu 3 focuses on shorter, mostly single‑turn interactions; extending to long documents and sustained dialog is a natural next step (Section 8.3).
  - Multilingual post‑training: beyond Aya inclusion, build full multilingual recipes with balanced data and cross‑lingual alignment (Section 8.3).
  - Safety beyond refusal: evaluate and train for calibrated helping, context‑aware compliance/non‑compliance, and robustness to jailbreaks with minimal over‑refusal (Section 7.2.1; Tables 25–26).

- Practical applications
  - Open, high‑quality assistants for research labs and startups; domain‑adapted assistants for math, coding, and instruction‑following tasks; safer chat systems with transparent refusal/compliance behavior; foundations for agentic systems that can be trained with verifiable sub‑goals.

In short, Tülu 3 is both a strong set of models and a methodologically complete blueprint. Its staged recipe, scalable data practices, RLVR innovation, and rigorous evaluation toolkit should shape how the community builds, measures, and iterates on open post‑training.
