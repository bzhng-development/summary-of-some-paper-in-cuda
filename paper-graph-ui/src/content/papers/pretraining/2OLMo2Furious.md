# 2 OLMo 2 Furious

**ArXiv:** [2501.00656](https://arxiv.org/abs/2501.00656)

## 🎯 Pitch

OLMo 2 debuts a fully open suite of large language models (7B, 13B, 32B parameters) that not only outperform previous open models on the efficiency-performance frontier, but also release every artifact of their development—model weights, code, training data, recipes, and logs—for maximal transparency and reproducibility. By pioneering techniques for stable, efficient training and introducing a new two-stage 'mid-training' data curriculum, OLMo 2 demonstrates that state-of-the-art model performance can be achieved without sacrificing openness, enabling rigorous scientific research and empowering the community to build, analyze, and enhance advanced LLMs from the ground up.

---

## 1. Executive Summary
OLMo 2 introduces a fully open family of large language models (LLMs) at 7B, 13B, and 32B parameters, together with a training recipe that prioritizes stability and per-token efficiency. Beyond releasing weights, the work opens the entire development stack—training data, code, logs, and intermediate checkpoints—and demonstrates that, with careful architectural and training choices plus a targeted “mid‑training” data curriculum, open models can sit on the Pareto frontier of performance vs. compute (Figure 1) and yield competitive instruction‑tuned variants (Tables 6–7).

## 2. Context and Motivation
- Problem addressed
  - Training large, dense LLMs is commonly unstable (loss/gradient spikes) and opaque (limited release of data/recipes). Even successful open‑weights models rarely release full data and logs, limiting reproducibility and scientific study (Introduction, §1).
  - OLMo‑0424—an earlier open model—suffered frequent loss spikes and growing gradient norms, which harmed scaling to larger models (Figure 2).
- Why it matters
  - Stability and transparency are prerequisites for reliable scaling, scientific analysis of training dynamics, and community contributions. Fully open artifacts enable research on training dynamics, memorization, safety, and efficient transfer (Introduction; §6).
- What existed before and limitations
  - Open‑weights series (e.g., Llama 3.x, Qwen 2.5, Gemma 2) show strong capability but provide only partial openness (weights without full pretraining data/recipes).
  - Fully open projects (e.g., OLMo 1/0424, Pythia, Amber, DCLM) advanced openness but trailed the strongest open‑weights models on several benchmarks (Introduction; Figure 1).
- Positioning
  - OLMo 2 aims for both rigor and competitiveness: it replaces instability‑prone design choices, adds a two‑stage data curriculum (“mid‑training”) to efficiently inject missing skills, and adopts a modern post‑training pipeline (Tülu 3 + RLVR). It releases all artifacts—from data to logs—to be a platform for research (§2; §5; §6).

## 3. Technical Approach
This is a two‑stage base‑model training pipeline (pretraining → mid‑training) followed by post‑training (SFT → preference tuning → RLVR), supported by architectural and optimization choices aimed at stability and efficiency.

1) Model architecture and stability choices (§2.1; §3)
- Transformer backbone with several stability‑oriented changes relative to OLMo‑0424 (Table 1):
  - `RMSNorm` and post‑norm residual layout: normalize outputs of attention and MLP instead of inputs:
    - h := x + RMSNorm(Attention(x))
    - hout := h + RMSNorm(MLP(h))  (Equations (1)–(2), §2.1)
    - Rationale: mitigates gradient/activation growth across depth; improves stability (Figure 7).
  - `QK‑norm`: RMSNorm applied to queries and keys before attention score computation to prevent large logits and divergence (§2.1; §3.3.2).
  - `z‑loss`: a small penalty on the log‑partition Z of the softmax to keep logits in a healthy range (§3.3.3). The paper uses a small coefficient (Table 1 lists 1e‑5; §3.3.3 experiments discuss 1e‑4).
  - RoPE θ increased to `5e5` to improve positional resolution (§2.1).
  - `GQA` (grouped‑query attention) for the 32B variant to reduce KV cache and compute (Table 3).
  - Tokenizer: swaps to a larger, cl100k‑style tokenizer with added PII masking tokens; yields small but consistent gains at 1B scale (Table 2).
- Initialization and optimizer adjustments (§3.2; §3.4)
  - Initialization: simple truncated normal with std 0.02 for all parameters instead of depth‑scaled initialization; reduces gradient spikes and preserves gradient/activation scale across depth (Figures 4–6).
  - AdamW `ϵ` lowered from 1e‑5 to 1e‑8 for larger effective early updates and faster stabilization (Figure 9).
  - No weight decay on token embeddings to avoid shrinking embedding norms (which amplifies early gradients via normalization); improves stability (Figure 10).
- Data sanitization targeting loss spikes (§3.1; §2.4)
  - Detect and filter documents containing long repeated n‑gram runs (≥32 repeats, span 1–13 tokens). Also mask such regions during data loading to avoid spikes at run‑time (Figure 3).

2) Two‑stage base model training (§2.3–§2.4; §4)
- Stage 1—Pretraining with “OLMo 2 Mix 1124” (Table 4)
  - ~3.9T tokens, mostly high‑quality web from DCLM Baseline (3.71T), plus code (StarCoder, filtered), academic corpora (peS2o; arXiv), math‑heavy web/proofs (OpenWebMath, Algebraic Stack), and Wikipedia.
  - Sequence length 4,096; batch sizes 1,024 (7B) or 2,048 (13B/32B); cosine LR schedule with 2k warmup (Table 3).
  - Total tokens: 4.05T (7B), 5.6T (13B), 6.6T (32B) (§2.3).
- Stage 2—Mid‑training (“annealing”) with “Dolmino Mix 1124” (Tables 5, 10, 13; §4)
  - Purpose: in the schedule’s final phase, switch to up‑sampled high‑quality web + curated academic/encyclopedic content and math‑centric synthetic data to patch demonstrated capability gaps (esp. math).
  - High‑quality web is selected by combining a FastText quality filter with the FineWeb‑Edu classifier; plus decontaminated FLAN instructions, Wikipedia, peS2o, and a Q&A corpus from Stack Exchange (Table 5; §4.3).
  - Math sub‑mix combines multiple synthetic sources, including persona‑conditioned math problems with solutions (`TuluMath`), rewrites of TinyGSM in natural language (`TinyGSM‑MIND`), curated synthetic “textbooks” (`MathCoder2`‑style), filtered proofs (`Metamath`), and GSM8K train (Table 5; §4.4.1).
  - “Micro‑anneals”: short, 50/50 experiments mixing small math subsets with general web to quickly assess data quality and mixture ratios before committing full runs (§4.4.2; Table 12).
  - Learning‑rate study: higher peak LRs initially look better but converge to nearly the same final loss and downstream scores once linearly annealed to zero; i.e., performance is largely LR‑insensitive when you finish the schedule (Figure 11; Table 8).
  - “Checkpoint soups”: run the same mix multiple times with different data orderings and simply average the resulting checkpoints; this consistently beats the best single run (Table 14).

3) Post‑training (instruction tuning) based on Tülu 3 (§5)
- Supervised fine‑tuning (`SFT`): curated, permissively licensed instruction data; includes persona‑based synthetic prompts; careful filtering (e.g., removing “date cutoff” patterns) and majority voting on synthetic math SFT to avoid training on incorrect answers (§5 “SFT”; Table 17).
- Preference tuning (`DPO`): collect on‑policy generations from OLMo‑2 SFT checkpoints and off‑policy from a pool of permissively licensed models; score responses with GPT‑4o as an LM judge; construct (chosen, rejected) pairs; train with DPO (§5 “PreFT”; Table 27).
- Reinforcement Learning with Verifiable Rewards (`RLVR`): PPO on tasks with automatically checkable answers (e.g., GSM8K/MATH, constraint‑satisfaction prompts). Rewards are 1/0 based on verifiable correctness; value function initialized from a learned reward model (7B/13B). The 32B model uses `GRPO` (group relative policy optimization), which removes the reward model step (§5 “RLVR”; Figures 13–15; Table 18).

4) Infrastructure and implementation practices (§6)
- Training on two clusters (Cirrascale “Jupiter” and Google Cloud “Augusta”) with high‑bandwidth interconnect and storage; workload orchestration via Beaker.
- Throughput optimizations: Torch `compile`, detect/avoid host–device syncs, asynchronous checkpointing/logging on a separate backend, and synchronized Python GC—all to reduce stalls (Figure 16).

## 4. Key Insights and Innovations
- Stability recipe that scales
  - A specific combination—post‑norm `RMSNorm` + `QK‑norm` + simple 0.02 Gaussian initialization + lower AdamW `ϵ` + no embedding weight decay + repeated n‑gram filtering—turns a previously spiky run (OLMo‑0424) into a smooth training process (Figure 2), with each component justified by targeted analyses (Figures 3–10). This is more than a single trick; it’s a synergistic package that generalizes across sizes.
- Mid‑training (“annealing”) as curriculum injection
  - Rather than a single pretraining mix, OLMo 2 splits compute: finish the cosine schedule earlier, then linearly anneal to zero on carefully chosen high‑quality and domain‑specific sources (Tables 5, 9, 11). This raises math and reading‑comprehension scores dramatically without full retraining.
- “Micro‑anneals”: cheap, source‑level data selection
  - Small, controlled anneals diagnose which math sources and ratios help most—e.g., rewriting code‑style TinyGSM into natural language (`MIND`) flips GSM8K gains from negative to large positive (Table 12, Experiment 3). This is a practical methodology for data‑mix design at low cost.
- Learning‑rate invariance at scale
  - When the LR is annealed to zero during mid‑training, a wide range of peak LRs produce nearly the same end performance (Table 8), echoing smaller‑scale observations but now at trillion‑token regimes (Figure 11). This reduces hyperparameter sensitivity and simplifies scaling.
- Checkpoint soups as a default
  - Simple weight averaging across 3 runs with different data orders consistently equals or beats the best single run on multiple mixes (Table 14). This is a robust, cheap improvement.

## 5. Experimental Analysis
- Evaluation protocol (§2.5; Appendix A)
  - Base models: OLMES suite with multiple‑choice and generative tasks; explicit separation between “development” tasks and “held‑out” tasks to avoid overfitting (Appendix A.1; Table 20).
  - Instruct models: categories for knowledge recall, reasoning, math, instruction following, and safety, with standardized few‑shot and prompting settings as per the Tülu 3 evaluation regime (Table 15; Appendix A.2).
- Main quantitative results (all numbers trace to specific tables/figures)
  - Pareto frontier: performance vs. training FLOPs (Figure 1) shows OLMo 2 on the frontier among models of comparable size and openness levels.
  - Base‑model comparisons (Table 6):
    - 7B: “Avg” across dev tasks = 62.9 (FLOPs ~1.8×10^23), beating DCLM‑7B (56.9) and OLMo‑0424‑7B (50.7).
    - 13B: 68.3 (FLOPs ~4.6×10^23).
    - 32B: 73.3 (FLOPs ~13.0×10^23), competitive with Qwen 2.5‑32B (74.9) despite greater transparency and fewer assumed tokens.
  - Effect of mid‑training (Table 9):
    - 7B avg improves from 53.0 → 62.9 (+9.9), with GSM8K 24.1 → 67.5 (+43.4), NQ 29.0 → 36.9 (+7.9), DROP 40.7 → 60.8 (+20.1).
    - 13B avg improves 58.9 → 68.3 (+9.4); GSM8K 37.3 → 75.1 (+37.8).
    - 32B avg improves 66.3 → 73.3 (+7.0); GSM8K 56.2 → 78.8 (+22.6).
  - Instruct models (Table 7):
    - OLMo‑2‑7B‑Instruct average = 56.5; GSM8K = 85.1; BBH = 51.4; IFEval = 72.3.
    - OLMo‑2‑13B‑Instruct average = 63.5; GSM8K = 87.4; MATH = 82.6.
    - OLMo‑2‑32B‑Instruct average = 68.8; GSM8K = 87.6; BBH = 70.6; MATH = 85.6; competitive with Qwen‑2.5‑32B‑Instruct (avg 68.1).
  - Tokenizer swap ablation at 1B (Table 2): OLMES increases 59.8 → 60.6; MMLU 34.8 → 35.2.
  - LR sweeps (Table 8): across peak LRs and anneal lengths, OLMES dev average varies within ~2 points, confirming LR insensitivity once mid‑training anneals to zero.
  - Micro‑anneals (Table 12):
    - Even 10–35% math proportion yields large GSM* gains, with modest impact on MMLU.
    - Rewriting TinyGSM to natural language (“MIND”) turns GSM* from 25.0 (inline code) → 65.5; duplicating to 2× raises it to 70.0.
  - Checkpoint soups (Table 14): e.g., “Mix F” 7B, Best single vs. 3× soup—Avg 77.1 → 77.9, GSM* 73.5 → 74.5.
  - Training stability figures:
    - Loss and gradient spikes vanish with the revised stack (Figure 2).
    - Repeated n‑gram filtering reduces gradient spikes (Figure 3).
    - Initialization and normalization choices maintain near‑zero growth exponents across depth (Figures 5–6) and suppress spikes (Figure 4).
  - RLVR learning curves (Figures 13–15): reward increases correlate with higher GSM8K/MATH/IFEval and average scores; multi‑stage RLVR on 13B (GSM8K‑oriented pass followed by MATH‑oriented pass) raises both math metrics and overall average (§5; Figure 13).
- Environmental accounting (Table 19):
  - Estimated pretraining energy: 131 MWh (7B) and 257 MWh (13B); emissions ~52 and ~101 tCO2eq respectively, using measured power, site PUEs, and regional carbon intensities.

Assessment
- The experiments are thorough and aligned with the paper’s claims:
  - Stability: multiple targeted ablations/plots make the causal story plausible (Figures 3–10).
  - Curriculum: both aggregate improvements (Tables 9, 11) and fine‑grained data‑source diagnostics (Table 12) support mid‑training efficacy.
  - Competitiveness: standardized, open evaluation (OLMES) with held‑out tasks (Table 6, Appendix A) gives credibility. Instruction‑tuned models compare favorably against similarly sized open‑weights (Table 7).
- Areas where evidence is suggestive rather than definitive:
  - LR invariance is shown for several LRs and token budgets, but the paper notes cost limited a full sweep to map plateaus/boundaries (§4.1).
  - Safety metrics decrease modestly from SFT/DPO to RLVR in some cases (e.g., 32B safety: 93.8 → 85.9 in Table 16), suggesting a trade‑off that deserves deeper analysis.

## 6. Limitations and Trade-offs
- Scope and focus
  - English‑centric pretraining and evaluation; non‑English capability is not a goal here (§5 excludes multilingual SFT variants as non‑beneficial at this time).
  - Coding was not a primary optimization target in the Instruct suite (Table 15 excludes code), so code performance is not emphasized.
- Data and contamination
  - While decontamination is applied (e.g., FLAN; §4.3), full contamination auditing across all sources is intractable at this scale; held‑out tasks mitigate but cannot eliminate concerns (§2.5).
- Compute and design search constraints
  - LR sweeps and anneal lengths are explored but not exhaustively mapped due to cost (§4.1). Other knobs (e.g., proportion of math/web in Dolmino) are tuned via micro‑anneals but still heuristic.
- Safety and reward‑hacking risk in RLVR
  - Instruction models see slight safety metric drops at the final RLVR stage (Table 16); using verifiable rewards focuses the model on pass/fail signals, which can narrow general caution or calibration if not balanced with broader safety objectives.
- Small‑model scaling challenges
  - The 1B variant struggles to use trillions of tokens efficiently during base pretraining (Appendix B). It benefits greatly from post‑training, pointing to capacity limits at very small scales (Tables 22–23).
- Implementation subtleties
  - z‑loss coefficient differs across sections (Table 1 vs. §3.3.3), indicating that best values may be setup‑dependent; fused implementations can diverge in backward pass (Figure 8).
  - Post‑training tokenizer mismatch in early “preview” models required retraining to maintain consistency (Appendix C.3; Figure 17).

## 7. Implications and Future Directions
- What changes in the landscape
  - OLMo 2 demonstrates that fully open LLMs can be both competitive and reproducible when the entire pipeline—data, code, logs, intermediate checkpoints—is released. The stability recipe plus mid‑training curriculum give a replicable path to strong base models (Figure 1; Tables 6, 9).
- Practical applications
  - Researchers and practitioners can:
    - Reproduce large‑scale training runs and study dynamics (e.g., spike mitigation, initialization scaling).
    - Tailor mid‑training curricula for domain skills (e.g., biomedical, legal) using the micro‑anneal methodology (§4.4.2).
    - Apply RLVR and constraint‑based reward shaping to tasks with verifiable outcomes (e.g., program synthesis with unit tests, math, data cleaning).
- Follow‑up research enabled/suggested
  - Data curriculum design: extend micro‑anneals to multilingual or code‑first curricula; explore principled mixture optimization (beyond duplications/filters) using signals gathered during micro‑anneals (Table 12).
  - Stability at longer contexts: with RoPE θ = 5e5, study scaling to much longer sequence lengths with the same stability stack (§2.1).
  - Safety‑aware RL: integrate safety‑specific verifiable rewards or multi‑objective RL to avoid the observed safety dips at the final stage (Table 16).
  - Learning‑rate plateaus: systematically map the parameter/token regimes where LR invariance holds, and relate to theory (§4.1; Table 8).
  - Transparency benchmarks: OLMES’s dev/held‑out split is a concrete step; broader community adoption could standardize honest reporting and reduce overfitting (Appendix A).

> Overall, OLMo 2’s core message is methodological: training stability is a first‑class objective; data curricula can be engineered efficiently; and full openness is feasible at modern scales. The result is a family of base and instruction models that are compute‑efficient (Figure 1), show large capability gains from targeted mid‑training (Table 9), and offer a transparent foundation for research and deployment.
