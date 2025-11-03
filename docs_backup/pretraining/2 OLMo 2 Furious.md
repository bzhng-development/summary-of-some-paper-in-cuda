# 2 OLMo 2 Furious

**ArXiv:** [2501.00656](https://arxiv.org/abs/2501.00656)
**Authors:** Team OLMo, Pete Walsh, Luca Soldaini, Dirk Groeneveld, Kyle Lo, Shane Arora, Akshita Bhagia, Yuling Gu, Shengyi Huang, Matt Jordan, Nathan Lambert, Dustin Schwenk, Oyvind Tafjord, Taira Anderson, David Atkinson, Faeze Brahman, Christopher Clark, Pradeep Dasigi, Nouha Dziri, Michal Guerquin, Hamish Ivison, Pang Wei Koh, Jiacheng Liu, Saumya Malik, William Merrill, Lester James V. Miranda, Jacob Morrison, Tyler Murray, Crystal Nam, Valentina Pyatkin, Aman Rangapur, Michael Schmitz, Sam Skjonsberg, David Wadden, Christopher Wilhelm, Michael Wilson, Luke Zettlemoyer, Ali Farhadi, Noah A. Smith, Hannaneh Hajishirzi
**Institutions:** Allen Institute for AI (AI2)

## 🎯 Pitch

OLMo 2 revolutionizes open language models by providing not only the weights but also a complete reproducible recipe that includes training data and code, addressing major gaps in transparency and scientific study. Its stability-first design and innovative two-stage training place it at the forefront of performance for its compute level, making high-quality research more accessible and enabling deeper understanding of training dynamics and safety.

---

## 1. Executive Summary (2–3 sentences)
OLMo 2 is a fully open family of language models at 7B, 13B, and 32B parameters that releases everything needed to reproduce the models: weights, complete training data, training code/recipes, logs, and intermediate checkpoints. It introduces a stability‑first architecture and a two‑stage training recipe (“mid‑training” with a curated Dolmino Mix 1124) that together place the models on the Pareto frontier of performance vs. training compute while remaining fully transparent (Figure 1; Table 6).

## 2. Context and Motivation
- The problem/gap:
  - Many “open” models release weights, but not the full recipe or training data, limiting scientific study and reproducibility. Fully open efforts existed (e.g., OLMo 1, Pythia, Amber), but typically underperformed compared to recent open‑weights models (Introduction; Figure 1).
  - Large‑scale training is fragile: loss spikes and gradient explosions cause costly divergences, especially at larger scales (Section §3; Figure 2). This instability undermines performance and wastes compute.
  - Data quality and curriculum choices late in training can significantly shift downstream capabilities (e.g., math), but principled, cost‑efficient methods to compose such curricula are underdeveloped (Sections §4, §4.4).

- Why it matters:
  - Transparent, reproducible models enable research on training dynamics, memorization, scaling laws, and safety—areas that require access to pretraining data and logs (Introduction).
  - Training‑stability fixes and efficient curricula reduce cost and risk while improving capability, widening access to high‑quality open models.

- Prior approaches and their limits:
  - Prior fully open suites (OLMo‑0424, Pythia, Amber, DCLM) advanced openness but were either less competitive on benchmarks or did not resolve stability at larger scales (Figure 1; Table 6).
  - Contemporary open‑weights models (e.g., Llama 3.1, Gemma 2, Qwen 2.5) are strong, but their datasets and exact recipes are not fully disclosed, limiting reproducibility and study (Introduction; Table 6 notes on openness).

- Positioning:
  - OLMo 2 targets both stability and capability with:
    - Architectural and optimizer changes that measurably suppress loss/gradient spikes (Section §3; Figures 2–10).
    - A two‑stage training process with a late “mid‑training” curriculum (Dolmino Mix 1124) that patches specific deficits (especially math) and upgrades general skills (Sections §2.3, §4; Tables 5, 9, 11–13).
    - A post‑training pipeline (Tülu 3 + RLVR) using permissively licensed data only (Section §5; Tables 15–16).

## 3. Technical Approach
This section unpacks how OLMo 2 is built and trained, emphasizing the concrete mechanisms behind its stability and capability gains.

- Model architecture (Section §2.1; Table 1; Table 3):
  - Base: decoder‑only transformer with no biases and SwiGLU activation.
  - Stability‑oriented changes:
    - `RMSNorm` replaces the prior non‑parametric LayerNorm to normalize activations (Section §2.1; Table 1).
    - `Post‑norm` block layout: normalize the outputs of attention and MLP blocks (not the inputs). The block computes:
      - h := x + RMSNorm(Attention(x))
      - hout := h + RMSNorm(MLP(h))
      (Equations (1)–(2), Section §2.1). This change mitigates gradient amplification in deep stacks (Section §3.3.2; Figure 7).
    - `QK‑norm`: apply RMSNorm to queries and keys before dot‑product attention to prevent overly large attention logits (Section §3.3.2).
    - `z‑loss` regularization: add 10^-4 · log2 Z to the loss (Z is the softmax normalizer) to discourage very large logits (Section §3.3.3).
    - Larger `RoPE θ = 5e5` to increase positional encoding resolution (Section §2.1).
    - 32B uses `GQA` (grouped query attention) to reduce KV cache costs while retaining multi‑head queries (Section §2.3; Table 3).
  - Why these choices: Ablations and diagnostics show these normalizations and regularization reduce gradient/loss spikes (Figures 2, 7–9) and keep gradient/activation scales healthy through depth (Figures 5–6).

- Tokenizer (Section §2.2; Table 2):
  - Switch to `cl100k` vocabulary (used by GPT‑3.5/4) with a few legacy special tokens retained for backward compatibility.
  - Tested at 1B scale: small but consistent improvements across OLMES generative and MC tasks (Table 2).

- Two‑stage training recipe (Section §2.3; Table 3):
  - Stage 1: long pretraining (90–95% FLOPs) with cosine LR schedule after a 2,000‑step warmup. Tokens: 3.90T (7B), 5.0T (13B), 6.06T (32B) (Section §2.3 and §2.4; Table 4).
  - Stage 2 (“mid‑training”): short late stage (5–10% FLOPs) with linearly decaying LR to zero and a targeted, higher‑quality/specialized mixture, “Dolmino Mix 1124” (Sections §2.3, §4; Tables 5, 13).

- Data pipeline and mixes:
  - Pretraining mix “OLMo 2 Mix 1124” (Table 4):
    - ~95% web (DCLM baseline 1.0), plus permissive code (StarCoder subset), academic corpora (peS2o; arXiv), Wikipedia, and math web/proofs (OpenWebMath, Algebraic Stack).
    - Data cleaning for stability: filter documents with ≥32 repeated n‑gram sequences; additional training‑time masking for such spans (Section §3.1; Figure 3).
  - Mid‑training mix “Dolmino Mix 1124” (Table 5):
    - High‑quality web filtered by two classifiers (DCLM FastText; FineWeb Edu), plus encyclopedic/academic sources and StackExchange Q&A.
    - Math‑centric synthetic/filtered sets (TuluMath, DolminoSynthMath, `TinyGSM‑MIND`, MathCoder2‑synthetic, filtered Metamath and CodeSearchNet, GSM8K‑train) (Section §4.4.1; Table 5).
    - Composition varies by 50B / 100B / 300B token budgets, keeping relative proportions approximately constant by repeating sources as needed (Section §4.5; Table 13).

- Stability interventions (Section §3; Figures 2–10):
  - `Repeated n‑gram` removal/masking lowers the frequency of gradient spikes (Section §3.1; Figure 3).
  - `Initialization`: simple normal init with std=0.02 (no layer‑scaled init). This preserves gradient/activation norms across depth better (growth exponent near 0), enabling stable low‑precision training; spike score drops from 0.40→0.03 in tests (Section §3.2; Figures 4–6).
  - `AdamW ε`: reduce from 1e‑5 to 1e‑8 to allow larger early updates, stabilizing gradient norms sooner (Section §3.4.1; Figure 9).
  - `No weight decay on embeddings`: avoids vanishing embedding norms that otherwise induce large early gradients via normalization Jacobians (Section §3.4.2; Figure 10).

- Learning‑rate annealing behavior (Section §4.1; Figure 11; Table 8):
  - Higher peak LRs win early but are overtaken later; after a short mid‑training to LR=0, variants converge to nearly identical losses and similar downstream averages (Table 8). A higher LR can yield slightly better math (GSM8K +2.8) when the mid‑training data itself is math‑focused.

- “Micro‑anneals” to choose math data cheaply (Section §4.4.2; Table 12):
  - Method: brief 50/50 runs mixing a candidate math subset with a standard web mix; linearly anneal LR to zero to evaluate quality quickly (<10B tokens visible effects).
  - Findings:
    - Even a small fraction of domain data yields big gains (GSM* rises from 28.5→61 with only 10% math; Table 12).
    - Limited duplication of scarce math data (2×) can help (GSM* 61→66; Table 12).
    - Rewriting code‑style math (TinyGSM) into natural language (“MIND” prompts) dramatically improves outcomes (GSM* 25→65.5; 2× to 70.0), showing representation matters (Table 12).

- Model “soups” (weight averaging) (Section §4.5; Table 14):
  - Average multiple mid‑training runs with different data orders; across six mixes, a 3‑run soup matches or beats the best single run on both MC and generative averages and on GSM* (Table 14).

- Post‑training with Tülu 3 and RLVR (Section §5):
  - SFT: carefully curated permissive instruction data plus large‑scale persona‑driven synthetic questions; small variants of the Tülu 3 mix (Section §5; Table 17 notes).
  - DPO (preference tuning): on‑policy prompts from OLMo 2 SFT variants, plus responses from a pool of permissibly licensed models; GPT‑4o judges pairwise preferences on helpfulness/truthfulness/honesty/instruction‑following (Section §5; Table 16; Appendix D: Table 25 & Table 27).
  - RLVR (Reinforcement Learning with Verifiable Rewards): use PPO to reward only verifiably correct generations (e.g., exact numeric math answers). 7B/13B use reward models; 32B uses GRPO (no RM) (Section §5; Figures 13–15; Table 18).
  - Multi‑stage RLVR for 13B (GSM8K + MATH + constraints → GSM8K only → MATH only) steadily increases math and average scores (Figure 13).

- Infrastructure and efficiency (Section §6):
  - Training on two H100 clusters (Cirrascale “Jupiter” and Google Cloud “Augusta”) orchestrated by Ai2’s Beaker (Sections §6.1–6.2).
  - Practical speedups: PyTorch `torch.compile`, avoiding host–device syncs, asynchronous logging/checkpointing via a separate backend, and coordinated garbage collection—all to stabilize and speed large distributed jobs (Section §6.4; Figure 16).
  - Environmental accounting (energy, carbon, water) reported from logged telemetry and local grid intensities (Section §6.5; Table 19).

## 4. Key Insights and Innovations
- Stability‑first transformer stack that scales smoothly (Section §3; Figures 2–10; Table 1):
  - What’s new: the specific combination—post‑norm RMSNorm inside residual branches, RMSNorm on queries/keys, z‑loss, a simple unscaled initialization, reduced AdamW ε, and turning off embedding weight decay—systematically reduces gradient and loss spikes.
  - Why it matters: stable, low‑precision training at larger scales with fewer restarts and better final minima. Evidence: growth exponents near zero (Figure 5), improved gradient/activation scaling vs. width (Figure 6), significant drop in spike scores, and smoother losses (Figure 2).

- Cost‑effective “mid‑training” with Dolmino Mix 1124 and “micro‑anneals” (Sections §4.1–§4.5; Tables 5, 9, 11–14):
  - What’s new: treat late training as a targeted curriculum stage, and select math sources via extremely short “micro‑anneals” that are cheap yet predictive at full run scale (Table 12).
  - Why it matters: dramatic gains where the base model is weak—e.g., GSM8K jumps 24.1→67.5 (7B) and 37.3→75.1 (13B) after mid‑training (Table 9)—without expensive full‑run sweeps for each data variant.

- Fully open, compute‑efficient models on the Pareto frontier (Figure 1; Table 6):
  - What’s new: end‑to‑end transparency—weights, code, complete data, logs, and thousands of checkpoints—at performance competitive with popular open‑weights‑only models, often using fewer approximate training FLOPs (Figure 1; Table 6).
  - Why it matters: unlocks research into data/recipe effects, memorization, and scaling that closed or partially open projects cannot support.

- Multi‑stage RL with verifiable rewards at scale (Section §5; Table 16; Figures 13–15):
  - What’s new: integrate Tülu 3 SFT+DPO with RLVR in multiple stages (for 13B), initializing PPO’s value function from learned reward models; 32B uses GRPO to remove the need for an RM.
  - Why it matters: systematic, measurable gains from SFT→DPO→RLVR on reasoning/math while using permissive data (Table 16; Figures 13–15).

## 5. Experimental Analysis
- Evaluation methodology (Appendix A; Table 20; Section §2.5):
  - Base models: OLMES suite with standardized prompts and scoring across 5 multiple‑choice and 2 generative development tasks, plus 4 held‑out tasks (AGIEval, GSM8K, MMLU‑Pro, TriviaQA). OLMES uses two MC formats and reports the better (Multiple‑Choice vs. Cloze/Completion), consistent shots (often 5‑shot), and F1 for generative tasks (Appendix A; Table 20).
  - Instruct models: Tülu 3 evaluation settings on knowledge recall, reasoning, math, instruction following, and safety (Section §5; Table 15).

- Main quantitative results:
  - Base models (Table 6; Figure 1):
    - OLMo 2 7B achieves avg 62.9 on 10 dev benchmarks; OLMo 2 13B: 68.3; OLMo 2 32B: 73.3.
    - These are competitive with open‑weights models of similar size while requiring fewer training FLOPs; Figure 1 plots average performance vs. approximate FLOPs, placing OLMo 2 on the Pareto frontier among fully open models.
  - Gains from mid‑training (Table 9):
    - Average dev improvements: +10.6 (7B), +10.3 (13B), +7.0 (32B).
    - Math improves most: GSM8K +43.4 (7B: 24.1→67.5), +37.8 (13B: 37.3→75.1), +22.6 (32B: 56.2→78.8).
    - Reading comprehension also jumps (NQ, DROP), and general MC (MMLU, ARC‑C) improves.
  - Mid‑training mix comparisons (Table 11):
    - High‑quality web filters (DCLM FT top‑7% + FineWeb Edu ≥2) beat simple LR anneal (OLMES avg 75.2 vs. 74.0; MMLU 63.1 vs. 61.8).
    - Adding math boosts both math (GSM* 52.0) and generative tasks (OLMES‑Gen +6.2 over PT Mix).
    - Adding instruction data alongside math keeps broad gains (OLMES‑Gen best at 70.2; math slightly below pure math mix).
  - Micro‑anneals (Table 12):
    - Even 10% math materially helps (GSM* ~61). Doubling math subset helps (to ~66), with diminishing returns at 4×.
    - Rewriting code‑style TinyGSM to natural language yields the biggest jump (to ~65.5; 2× to 70.0).
  - Model soups (Table 14):
    - In 6 mixes, a 3‑run average consistently equals or beats the best single run across OLMES averages and frequently on GSM*.
  - Instruct models (Table 7; Table 16):
    - Stagewise gains are consistent. Example (13B): SFT avg 56.6 → DPO 62.0 → RLVR 63.4 (Table 16). GSM8K for 13B reaches 87.4; MATH 39.2.
    - Against open‑weights peers: OLMo 2‑13B‑Instruct approaches Qwen 2.5‑14B and surpasses Llama 3.1‑8B and Tülu 3‑8B on the averaged suite (Table 7).
    - 32B‑Instruct averages 68.8; with strong GSM8K (87.6) and MATH (49.7) (Table 7), aided by RLVR/GRPO (Figure 14).

- Stability evidence:
  - Training curves before vs. after interventions show many fewer spikes and steadier gradients (Figure 2).
  - Quantitatively, the initialization change reduces spike score of gradient norm from 0.40 to 0.03 in stress tests (Section §3.2). QK‑norm + post‑norm reduces gradient spike score from 0.108→0.069 (Figure 7). AdamW ε lowering smooths early gradients (Figure 9). Disabling embedding weight decay prevents embedding norm collapse and reduces spike frequency (Figure 10).

- Learning‑rate invariance:
  - Different peak LRs converge after annealing; downstream averages across nine tasks are within ~0.5 points (Table 8). Slight math advantage appears when high‑LR pretraining is followed by math‑focused mid‑training (+2.8 GSM8K).

- Robustness/held‑out:
  - The project maintains held‑out tasks (AGIEval, GSM8K test, MMLU‑Pro, TriviaQA) not used for development (Table 6; Appendix A.1). Mid‑training gains transfer to these held‑outs (Table 9, rightmost columns), indicating generalization of the approach.

- Do the experiments support the claims?
  - Yes, on two fronts:
    - Stability: multiple targeted ablations/diagnostics link each intervention to reduced spikes or healthier norms (Figures 3–10).
    - Capability: a clean, staged comparison (Table 9) ties mid‑training to large, broad improvements; micro‑anneals + soups show repeatable and composable data‑selection gains (Tables 12 & 14). Stagewise instruct training (Table 16; Figures 13–15) further adds consistent improvements.

## 6. Limitations and Trade-offs
- Assumptions and scope:
  - Mid‑training relies on the availability of “verifiable” domains (math, constrained tasks) for RLVR and on high‑quality filtered web/math corpora; transfer to other domains (e.g., multilingual, code) is not evaluated here (Tables 5, 15).
  - The tokenizer switch was validated at 1B scale with small gains (Table 2); effects at very large scales are inferred but not directly isolated.

- Scenarios not addressed:
  - Code specialization is not a target in this release; code datasets are a small fraction of pretraining and optional in mid‑training (Tables 4–5, 10).
  - The 1B model struggles to escape near‑random MC accuracy without mid‑training/post‑training (Appendix B; Table 22), pointing to capacity limits at very small sizes.

- Computational/data constraints:
  - Although compute‑efficient relative to peers, total tokens are still large (up to 6.6T; Section §2.3). Training requires H100‑class clusters with high‑speed interconnect (Section §6.1).
  - z‑loss implementation details matter: different fused vs. PyTorch implementations had mismatched backward behavior, forcing re‑training from a safe checkpoint (Section §3.3.3; Figure 8), suggesting fragility in some fused kernels.

- Evaluation trade‑offs:
  - OLMES chooses the better of two MC formats per task (Appendix A), which aligns with widely used practice but may slightly inflate “best‑of‑format” numbers compared to fixed‑format-only reports.
  - While FLAN was decontaminated from evaluation sets (Section §4.3), broader decontamination across all sources is intractable; some contamination risk remains typical of web‑scale pretraining.

- Environmental impact:
  - Despite efficiency, pretraining energy remains substantial (e.g., 7B ~131 MWh; 13B ~257 MWh; Table 19). Carbon and water use are reported but still non‑trivial.

## 7. Implications and Future Directions
- Field impact:
  - OLMo 2 demonstrates that fully open models can reach the performance/compute frontier (Figure 1; Table 6) when training stability and a principled late‑stage curriculum are prioritized. The complete release (data, code, logs, checkpoints) sets a high bar for transparency and will likely catalyze research on training dynamics, memorization, and data governance.

- Follow‑up research enabled/suggested:
  - Stability: Formalize spike‑score‑driven early‑warning systems and investigate theoretical links between post‑norm + QK‑norm + z‑loss and signal propagation (Sections §3.2–§3.3).
  - Data curricula: Extend “micro‑anneals” to other domains (code, multilingual, retrieval‑augmented recipes) and automate source selection with better proxies and learned scorers (Sections §4.3–§4.4.2).
  - RLVR: Build rewardable datasets for other verifiable tasks (unit tests for code, structured QA with validators); study GRPO vs. RM‑based PPO trade‑offs at scale (Section §5; Figure 14).
  - Small models: Address the 1B capacity cliff using distillation, mixture‑of‑experts, or task‑targeted SFT curricula (Appendix B).

- Practical applications:
  - Deployable “Instruct” models for general assistance with strong math and reasoning (Table 7), especially where transparent training data is required (regulated domains, research).
  - A reproducible blueprint for organizations to train specialized models with tighter budgets: adopt stability toolkit (Section §3), two‑stage mid‑training (Section §4), and soup averaging (Section §4.5).
  - Infrastructure practices (Section §6.4) are directly actionable for teams running large distributed training on PyTorch.

> Key quantitative takeaways:
> - “Mid‑training” raises GSM8K by +43.4 (7B) and +37.8 (13B) and improves general benchmarks (Table 9).
> - Final base model averages: 7B 62.9, 13B 68.3, 32B 73.3 (Table 6), with efficient FLOPs positioning on the Pareto frontier (Figure 1).
> - Instruct models gain consistently through SFT→DPO→RLVR (Table 16); 13B‑Instruct reaches GSM8K 87.4 and competitive averages vs. similarly sized peers (Table 7).

Overall, OLMo 2’s contribution is twofold: a stability‑first recipe that unlocks reliable, compute‑efficient scaling, and a transparent, reproducible end‑to‑end release that advances the science of language model training as much as it advances the models themselves.
