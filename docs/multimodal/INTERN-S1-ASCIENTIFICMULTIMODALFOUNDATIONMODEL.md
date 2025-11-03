# INTERN-S1: A SCIENTIFIC MULTIMODAL FOUNDATION MODEL

**ArXiv:** [2508.15763](https://arxiv.org/abs/2508.15763)

## 🎯 Pitch

Intern-S1 defines a new open-source standard for scientific AI by unifying a massive Mixture-of-Experts language model with specialized vision and time-series encoders, plus a novel dynamic tokenizer tailored for complex scientific data. Trained on an unprecedented scale of scientific tokens and refined with a Mixture-of-Rewards reinforcement learning framework, Intern-S1 dramatically narrows the performance gap with closed models—delivering state-of-the-art reasoning and multimodal understanding across diverse scientific domains. This leap enables transformative applications in scientific discovery, from molecular design to materials science, democratizing advanced tools for researchers worldwide.

---

## 1. Executive Summary (2–3 sentences)
Intern‑S1 is an open‑source, scientific multimodal foundation model that integrates a large Mixture‑of‑Experts language model with vision and time‑series encoders, plus a new dynamic tokenizer for scientific strings. Trained on 5T tokens (over 2.5T from science) and post‑trained with a Mixture‑of‑Rewards online reinforcement learning framework, it achieves top open‑source performance on general reasoning and state‑of‑the‑art performance on many scientific text and image‑text benchmarks (Tables 2–4), substantially narrowing the gap with leading closed‑source systems.

## 2. Context and Motivation
- Problem addressed
  - Progress in open‑source models has been fast for popular domains (math, code, natural images), yet capability in scientific domains (chemistry, materials, life sciences, physics, earth science) lags behind and still often relies on expert systems or closed models (Introduction; Fig. 1–2).
  - Scientific data are low‑resource, diverse in modality (molecules, protein sequences, formulas, tables, figures, time series), and require long, rigorous reasoning (Introduction).

- Why it matters
  - Better scientific models can accelerate hypothesis testing, experimental design, and discovery across high‑value domains like drug design, materials discovery, and climate/earth observation (Introduction).

- Shortcomings of prior approaches
  - Open‑source multimodal models mainly target natural images and general VQA; they underperform on science‑specific content (e.g., chemistry strings, document equations) and low‑resource modalities (Fig. 2).
  - Static tokenizers treat scientific strings like ordinary text, leading to poor compression and ambiguous embeddings (Sec. 2.2). PDF parsing and web data pipelines are not optimized for scientific structure (Sec. 4.1.1).
  - RL for reasoning is largely validated on dense models; applying GRPO‑style methods to large MoE models is unstable due to expert routing mismatch between inference and training (Sec. 5.2.3).

- Positioning
  - Intern‑S1 tackles the full stack: data, architecture, training system, and RL, with a science‑first orientation:
    - 2.5T+ scientific tokens via specialized data pipelines (Sec. 4.1; Fig. 6–9).
    - A dynamic tokenizer that recognizes scientific substrings (SMILES, FASTA) and assigns modality‑specific embeddings (Sec. 2.2; Fig. 4).
    - A multimodal architecture that adds vision and time‑series encoders (Sec. 2; Fig. 3).
    - A scalable online RL setup using a Mixture‑of‑Rewards across 1000+ verifiable tasks, stabilized for MoE (Sec. 5.2; Fig. 12).

## 3. Technical Approach
This section walks through the model, data, training, and RL components and why each choice was made.

- Overall architecture (Sec. 2; Fig. 3)
  - Backbone LLM: `Qwen3‑235B` MoE (Intern‑S1) and `Qwen3‑8B` (Intern‑S1‑mini).
    - `Mixture‑of‑Experts (MoE)`: a routing mechanism that activates a subset of expert sub‑networks per token to increase capacity without proportional compute.
  - Vision encoder: `InternViT‑6B` (or `InternViT‑300M` for mini), trained from contrastive pretrain to LLM‑coupled next‑token prediction for stronger fine‑grained features (Sec. 2.1).
    - Uses dynamic resolution and `pixel unshuffle` to reduce visual tokens 4×; a 448×448 image becomes 256 visual tokens; an MLP projector aligns them to the LLM embedding space (Sec. 2.1).
  - Dynamic tokenizer for scientific strings (Sec. 2.2; Fig. 4).
  - Time‑series encoder with adaptive downsampling + Transformer blocks for long scientific signals (seismic, gravitational waves, EEG) (Sec. 2.3).

- Dynamic tokenizer: how it works and why it matters (Sec. 2.2; Fig. 4)
  - Problem: static tokenizers use one split strategy and one embedding set for all text. This:
    - Wastes tokens on rare formats (e.g., `SMILES` for molecules).
    - Forces the same symbol (e.g., “C”) in English, DNA, and molecules to share one embedding, biasing toward frequent usages.
  - Mechanism:
    1. Detect scientific substrings either by explicit tags (e.g., `<FASTA>`, `<SMILES>`) or rule/tool detectors (e.g., RDKit) (Fig. 4, left).
    2. Segment the input into modality spans (e.g., general text vs. SMILES vs. FASTA).
    3. Tokenize each span with a strategy tailored to that modality.
    4. Map each span into its own embedding subspace “orthogonal” to others (i.e., independent embeddings); concatenate into a single sequence for the Transformer (Fig. 4, left).
  - Outcome:
    - Much higher compression for scientific strings; `compression ratio` (characters per token) improves up to ~70% vs. OpenAI GPT‑OSS‑120B, DeepSeek‑R1, and Qwen3 tokenizers on SMILES (Fig. 4, right). The CR metric is formalized in Eq. CR(τ, D) (Sec. 2.2).
    - Reduces compute and avoids semantic interference across modalities.

- Time‑series encoder (Sec. 2.3)
  - Scientific signals vary in sampling rate and length; text tokenization is ill‑suited.
  - An adaptive downsampling module compresses long sequences, then Transformer blocks model temporal dependencies, producing representations that the LLM can reason over (Sec. 2.3).

- Data pipelines: scaling science data with quality control (Sec. 4.1; Fig. 6–9)
  - Scale and mix
    - Continued pretraining (CPT) on 5T text tokens; >2.5T are scientific (Fig. 6, left).
    - Image‑text CPT uses ~250B tokens: 70B text and 180B interleaved image‑text; ~30B tokens are multimodal scientific data (Sec. 4.1.2).
  - Page‑level PDF parsing (Sec. 4.1.1; Fig. 7)
    - PDFs are rich in equations/symbols. A two‑stage parser minimizes cost:
      - Low‑cost parser (MinerU) runs on all pages.
      - A detector flags pages with equations/symbolic markers for high‑cost VLM parsing (e.g., InternVL, Qwen‑VL), then post‑processing and page‑level deduplication.
    - Only 5% (archived) / 3% (web) pages go through high‑cost parsing, yet quality improves; 20–50% of low‑quality content is filtered (Sec. 4.1.1).
  - Domain‑centric web parsing (Sec. 4.1.1; Fig. 8)
    - Treat each domain (hostname) as a unit; sample pages and use an LLM agent to decide per‑domain actions (discard/retain/rewrite), capturing consistent parsing quirks at lower cost than page‑wise LLM parsing.
  - Scientific recall and filtering (Sec. 4.1.1; Fig. 9)
    - Build a taxonomy (six science domains: Math, Physics, Chemistry, Life, Earth, Materials).
    - Use a strong LLM to annotate a silver set → train lightweight classifiers (fastText, 1.5B LLMs).
    - Optimize prompts using in‑domain vs. out‑of‑domain validation sets.
    - Result: target‑domain purity rises from ~2% to ~50% (Sec. 4.1.1).

- Training system and optimization (Sec. 3, 4.2)
  - Systems (Sec. 3.1–3.2)
    - `FSDP` for parameter sharding; `FP8` matmuls (DeepGEMM) with dynamic scaling; BF16 for the vision tower for stability.
    - MoE kernels: TMA‑Adaptive FP8 Grouped GEMM for dynamic groups; fused loss kernels (Liger); FlashAttention‑3 for variable lengths.
    - Variable‑Length Balanced Strategy (VLBS): bucket + sliding‑window sort to equalize per‑rank lengths, giving ~2× speedup at scale (Sec. 3.1).
    - RL deployment: colocated training + inference meshes; FP8 inference; EP8 rollout via LMDeploy; continuous batching and on‑the‑fly slot rebalancing (Sec. 3.2).
  - Multi‑stage training (Fig. 5)
    1. Text CPT (unimodal).
    2. Image‑text CPT (joint).
    3. Image‑text SFT (offline RL with best‑of‑N).
    4. Image‑text Online RL (Mixture‑of‑Rewards).
  - Batch‑size warmup and LR via scaling laws
    - Observation: small batches train better early; large batches are more efficient later (Fig. 10).
    - Use WSD (Warmup‑Stable‑Decay) LR scheduler and connect batch size B to gradient noise `B_simple` (Eq. 1): as loss falls, the effective critical batch size rises (Sec. 4.2.3).
    - In Xtuner, batch grows from 66M to 132M tokens, with switch after ~400B tokens processed (Sec. 4.2.3).
    - Learning‑rate schedule is chosen by fitting loss‑vs‑LR scaling laws and solving a constrained optimization over LR per step Ω (Eq. 2), yielding accurate loss prediction: predicted ~1.16 vs. actual 1.17–1.18 (Sec. 4.2.3).
  - Start from base vs. instruction checkpoints (Sec. 4.2.2; Fig. 11)
    - Empirically similar final performance post SFT+RL; instruct has an edge where post‑training introduced genuinely new capability (coding), while elsewhere it mainly activates latent skills.
    - Base model shows slightly higher initial entropy (0.19 vs. 0.15 on a math subset), but this can be compensated by RL hyperparameters (Sec. 4.2.2).
  - Multimodal CPT loss (Sec. 4.2.4)
    - Standard causal objective on text tokens only (visual tokens are context), with square‑averaging token weights to reduce gradient bias (Eq. 3–4).

- Post‑training: Offline RL (SFT) then Online RL (Sec. 5)
  - Offline RL / SFT (Sec. 5.1)
    - Filtered, labeled, and enhanced instruction data across domains; best‑of‑N sampling ensures high‑reward responses.
    - For multimodal, augment with science diagrams, OCR, charts, and strengthened long‑thinking data (SOPHIA‑style with strict quality filters) (Sec. 5.1.1).
    - Mixture selection via stepwise ablations and composition validation (Sec. 5.1.2).
  - Online RL with Mixture‑of‑Rewards (Sec. 5.2; Fig. 12)
    - `Mixture‑of‑Rewards (MoR)`: unify verifiable rewards across >1000 task types (logic puzzles, algorithmic tasks, domain exams; InternBootCamp provides synthetic generators) and non‑verifiable open‑ended prompts via a learned preference model.
    - Verifiers (Sec. 5.2.2):
      - Easy‑to‑verify tasks: rule‑based checkers + `CompassVerifier` (a lightweight generative verifier) to reduce false negatives.
      - Open‑ended chat/writing: `POLAR‑7B` policy discriminator produces relative‑quality reward signals.
    - Hybrid data filtering (Sec. 5.2.4; Fig. 13)
      - Offline prune too‑easy (pass@8=1.0) and too‑hard/noisy (pass@8≤0.25) items using both a dense SFT and a MoE SFT model.
      - Online drop groups where all 8 rollouts are identical (all‑correct or all‑wrong), and remove garbled/infinite‑loop generations—empirically stabilizes training and speeds gains on AIME2024 (Fig. 13).
    - RL algorithm for MoE stability (Sec. 5.2.3; Eq. 6)
      - Direct GRPO‑style token‑ratio clipping is brittle for MoE due to expert routing divergence between inference and training.
      - Use `OREAL`: behavior cloning (SFT loss) on positive samples + policy gradient on negatives; avoid token‑level importance‑ratio clipping (Eq. 6).
      - Remove OREAL’s token‑level reward model for throughput, then prevent entropy collapse via a selective KL regularizer on high‑covariance tokens (`KL‑Cov`; Eq. 5). With k=0.2, β=0.01, entropy holds near ~0.2 and validation accuracy keeps rising (Fig. 14).
    - Training details (Sec. 5.2.4)
      - FP8 for rollout and training; 8 rollouts/prompt; batch 4096 (8 mini‑batches), AdamW lr=5e‑7, wd=0.1, β=(0.9, 0.95); ViT and router frozen; 600 steps; drop 3% batches with grad‑norm>0.3; final checkpoint averaging.

## 4. Key Insights and Innovations
- Dynamic, modality‑aware tokenization for science (Sec. 2.2; Fig. 4)
  - Novelty: per‑span tokenization and per‑modality embeddings prevent semantic interference; scientific strings get much better compression.
  - Significance: up to ~70% higher characters‑per‑token on SMILES (Fig. 4, right) reduces compute and lets the model attend across longer scientific context—this is a fundamental capability, not a small tweak.

- Page‑level, cost‑aware PDF parsing with VLM fallbacks (Sec. 4.1.1; Fig. 7)
  - Novelty: a hybrid low/high‑cost pipeline at page granularity, guided by equation/symbol detectors, plus page‑graph deduplication.
  - Significance: cheaply recovers high‑quality text/equations from PDFs, crucial for science where formulas and figures carry the core knowledge.

- Domain‑centric web parsing + recall/filtering (Sec. 4.1.1; Fig. 8–9)
  - Novelty: LLM agents make per‑domain decisions (discard/retain/rewrite) and a taxonomy‑guided recall/filter loop with in‑domain vs. OOD prompt optimization.
  - Significance: boosts science purity from ~2% to ~50% (Sec. 4.1.1), solving the low‑resource bottleneck at scale.

- MoR: a unified, scalable online RL framework for 1000+ tasks (Sec. 5.2; Fig. 12)
  - Novelty: mixes rule‑based verifiers (exactness) with learned verifiers (`CompassVerifier`) and preference reward (`POLAR`) in one training loop; hybrid offline/online filtering balances task difficulty and sample quality (Sec. 5.2.4).
  - Significance: enables sustained gains across heterogeneous tasks while keeping training stable and efficient—reported “10× less RL time” vs. comparable public work (Abstract; Sec. 5 overview).

- MoE‑stable RL via OREAL + KL‑Cov (Sec. 5.2.3; Eq. 5–6)
  - Novelty: avoids token‑ratio clipping instability in MoE; adds selective KL on high‑cov tokens to keep entropy healthy without collapsing exploration (Fig. 14).
  - Significance: a practical recipe to bring online RL to very large MoE VLMs.

## 5. Experimental Analysis
- Evaluation setup (Sec. 6.1; Table 1)
  - Tooling: VLMEvalKit and OpenCompass; “thinking mode” enabled; sampling with temperature 0.7 (Intern‑S1) / 0.8 (mini), top‑p 0.95, top‑k 50; max tokens 65,536 (Table 1).
  - Scope: text‑only and multimodal general reasoning; science‑specific text and image‑text.

- Benchmarks (Sec. 6.2)
  - General reasoning: MMLU‑Pro, GPQA (Diamond), AIME‑2025, IFEval; and multimodal MathVista, MMMU, MathVision, MMStar (Sec. 6.2.1).
  - Scientific reasoning (text): SmolInstruct (chemistry), ChemBench, MatBench (materials), ProteinLMBench (Sec. 6.2.2).
  - Scientific reasoning (multimodal): SFE, Physics (PhD qualifying problems), MicroVQA (microscopy), MSEarth‑MCQ, XLRS‑Bench (ultra‑high‑res remote sensing) (Sec. 6.2.2).

- Main quantitative results
  - General (Table 2)
    - Intern‑S1 leads among open‑source multimodal models on all eight tasks.
    - Examples:
      - “MathVista”: 81.5 vs. 79.0 (InternVL3‑78B) and 74.8 (Qwen2.5‑VL‑72B).
      - “MathVision”: 62.5 vs. 43.1 and 38.1 for the two open‑source baselines.
    - It remains competitive but not best vs. APIs on some text‑only tasks (e.g., GPQA: 77.3 vs. Grok‑4 at 87.5).
  - Science text‑only (Table 3)
    - Intern‑S1 tops 3/4 benchmarks:
      - “SmolInstruct”: 51.0 (best overall; APIs: 40.4–47.3).
      - “ChemBench”: 83.4 (tied with/better than APIs).
      - “MatBench”: 75.0 (far ahead of open‑source MLLMs/VLMs by +23–26 points).
    - “ProteinLMBench”: 63.1—strong but below o3 (67.7) and Kimi‑K2 (66.7).
  - Science multimodal (Table 4)
    - Intern‑S1 ranks first on 4/5:
      - “SFE”: 44.3 (best; Gemini‑2.5 Pro at 43.0).
      - “MicroVQA”: 63.9 (best).
      - “MSEarth‑MCQ”: 65.7 (best).
      - “XLRS‑Bench”: 55.0 (best).
    - “Physics” (qualifying exams): 44.0—second to o3 (47.9).
  - Intern‑S1‑mini (Tables 5–7)
    - Text‑only general: new open‑source SOTA on MMLU‑Pro 74.8, GPQA 65.2, AIME‑2025 80.0 (Table 5).
    - Science text‑only: leads on all four vs. similarly sized open‑source models (Table 6).
    - Science multimodal: best on 4/5, but behind on SFE (35.8 vs. ~43.5 for others) (Table 7).

- Ablations and diagnostics
  - Tokenization compression: Fig. 4 (right) quantifies the 70% CR improvement on SMILES.
  - Batch‑size warmup: Fig. 10 shows early advantages with small batches and overall benefits of switching to large batches mid‑training.
  - Start‑point choice: Fig. 11 shows marginal differences between base vs. instruct after CPT+SFT+RL, with instruct preferred when post‑training introduced new skills (coding).
  - RL data filtering: Fig. 13 shows faster AIME2024 accuracy gains vs. DAPO filtering on a 32B model.
  - Entropy control: Fig. 14 demonstrates stabilized entropy (~0.2) and rising validation accuracy with KL‑Cov vs. collapse without it.

- Do the experiments support the claims?
  - Yes for the core claims:
    - Strong general reasoning among open‑source VLMs (Table 2).
    - Significant gains on scientific text and multimodal tasks, including wins over APIs on several science benchmarks (Tables 3–4).
    - Tokenization, data, and RL ablations provide mechanistic evidence for why performance improves (Figs. 4, 10–14).
  - Mixed areas:
    - ProteinLMBench lags behind some closed‑source systems (Table 3).
    - Physics (multimodal) is close but behind o3 (Table 4).
    - Intern‑S1‑mini underperforms on SFE vs. other small VLMs (Table 7).

> “Intern‑S1 … outperforms both open‑source and close‑source models on image‑text or text‑only scientific tasks.” (Introduction; Fig. 1; detailed in Tables 3–4)

> “Intern‑S1 achieved top‑tier general reasoning capability among open‑source models” (Fig. 1; detailed in Table 2).

## 6. Limitations and Trade-offs
- Assumptions and dependencies
  - The dynamic tokenizer relies on correct detection/tags for scientific substrings; mis‑tagging could degrade compression or semantics (Sec. 2.2).
  - Verifiable RL hinges on the quality of rule‑based checkers and learned verifiers; for open‑ended tasks, POLAR provides relative rewards, not absolute correctness (Sec. 5.2.2).

- Scope not fully evaluated
  - A time‑series encoder is introduced (Sec. 2.3), but the benchmark suite does not directly evaluate time‑series tasks (e.g., seismology, EEG); real‑world performance on such data remains to be shown.

- Computational complexity
  - Training uses 5T tokens and a very large MoE LLM; even with FP8 and FSDP, compute and engineering complexity are high (Sec. 3–4). The RL stage uses sizeable batches and multi‑rollout sampling (Sec. 5.2.4).

- MoE RL stability and sensitivity
  - While OREAL + KL‑Cov improves stability, it required tuning (k=0.2, β=0.01) because Intern‑S1 started with low entropy (Sec. 5.2.4). Sensitivity to these hyperparameters and to the proportion of positive/negative examples may persist.

- Data transparency
  - Some multimodal RL data are from “private collections” and “anonymized, real‑world user queries” (Sec. 5.2.2), which can limit perfect reproducibility and external auditing.

- Frozen components during RL
  - The RL stage freezes the vision tower and router (Sec. 5.2.4), potentially capping multimodal adaptation during online learning.

## 7. Implications and Future Directions
- Field‑level impact
  - Demonstrates that science‑specialized, open‑source multimodal models can challenge or surpass closed systems on several domain benchmarks by investing in modality‑aware tokenization, science‑centric data pipelines, and verifier‑driven RL at scale (Tables 3–4).
  - Provides a concrete blueprint for stabilizing online RL in large MoE VLMs (OREAL + KL‑Cov) and for unifying heterogeneous tasks under a single reward framework (MoR).

- Research directions enabled
  - Tokenization: extend dynamic, per‑modality tokenization to other structured formats (e.g., crystallographic files, graph encodings, domain‑specific markup) and study its interaction with retrieval and long‑context memory.
  - Time‑series: design standardized, verifiable time‑series science benchmarks (astronomy, geophysics, biomed) to evaluate the dedicated encoder introduced in Sec. 2.3.
  - Verifiers: improve cross‑domain verifiers (e.g., programmatic checkers for chemistry/materials, automatic derivation checking in physics) and study how verifier bias propagates through RL.
  - MoE RL theory: formalize expert‑routing drift between inference and training, and design objectives explicitly robust to expert mismatches.

- Practical applications
  - Chemistry and materials: molecular synthesis planning, reaction condition prediction, and crystal stability estimation (claimed in Abstract), supported by strong ChemBench and MatBench results (Table 3).
  - Document‑heavy science: better parsing and reasoning over papers, lab protocols, and figure‑rich documents (PDF pipeline, Sec. 4.1.1).
  - Earth observation and microscopy: strong performance on XLRS‑Bench and MicroVQA (Table 4) suggests utility for remote sensing analysis and bioimaging assistants.

Overall, Intern‑S1 shows that systematic investment in science‑specific tokenization, data, and RL yields large, transferable gains across both text and multimodal scientific reasoning. The paper’s detailed pipelines (Fig. 6–9), stabilized MoE RL (Sec. 5.2.3–5.2.4), and comprehensive evaluations (Tables 2–7) make it a practical template for future open‑source, domain‑specialized generalist models.
