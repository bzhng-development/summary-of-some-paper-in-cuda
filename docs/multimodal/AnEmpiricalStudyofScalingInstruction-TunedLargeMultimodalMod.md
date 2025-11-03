# An Empirical Study of Scaling Instruction-Tuned Large Multimodal Models

**ArXiv:** [2309.09958](https://arxiv.org/abs/2309.09958)

## 🎯 Pitch

This paper delivers the first thorough empirical roadmap for scaling open-source instruction-tuned Large Multimodal Models (LMMs)—like LLaVA—up to 33B and 65B/70B parameters, dissecting the effects of model size, image resolution, data mixing, and parameter-efficient fine-tuning (LoRA/QLoRA). The findings reveal that simply increasing model scale, mixing language-only instructions, and using higher image resolutions consistently bolster both visual and language capabilities, while LoRA/QLoRA can match the performance of full fine-tuning at a fraction of the compute cost. This work sets accessible new baselines and practical recipes for researchers and practitioners eager to harness the power of very large, cost-efficient LMMs in real-world multimodal AI systems.

---

## 1. Executive Summary (2-3 sentences)
This paper empirically studies how to scale instruction-tuned Large Multimodal Models (LMMs) like `LLaVA` from 7B/13B parameters up to 33B and 65B/70B, and what design choices matter (model size, image resolution, data mixing, and parameter‑efficient fine‑tuning such as `LoRA/QLoRA`). It shows, with controlled experiments, that larger language backbones, higher image resolutions, and mixing in language-only instruction data consistently improve multimodal performance, and that `LoRA/QLoRA` can match full fine‑tuning while dramatically reducing compute (Tables 1–5).

## 2. Context and Motivation
- Problem addressed
  - How to scale open‑source LMMs effectively and affordably beyond 13B parameters, and what choices most impact performance when tuning for visual instruction following. Prior open‑source systems (e.g., LLaVA, MiniGPT‑4) mainly used 7B/13B backbones, leaving the behavior at 33B–70B underexplored (Section 1).
- Why this matters
  - Larger backbones promise better reasoning, knowledge, and language generation—capabilities crucial for real-world, open‑ended visual assistants. But training/serving costs scale steeply; practical recipes (e.g., `LoRA/QLoRA`, data mixtures, resolution choices) are needed to create strong, accessible baselines.
- Prior approaches and gaps
  - Earlier visual instruction-tuned models demonstrated feasibility at small scales (7B/13B) but did not systematically test:
    - The effect of model size scaling on multimodal vs. language abilities.
    - The role of image resolution beyond 224px/224px.
    - Whether mixing language‑only instruction data helps or hurts.
    - Whether parameter‑efficient fine‑tuning can replace full fine‑tuning at large scales.
- Positioning
  - This work provides an end‑to‑end scaling study with controlled ablations and cost analyses (Section 3; Tables 3–5), establishing practical, stronger baselines at 33B and 65B/70B.

## 3. Technical Approach
Step-by-step pipeline (Section 2):

1) Model backbones and checkpoints
- Language backbones:
  - `Vicuna-33B` (public checkpoint).
  - `Vicuna-65B` (trained by the authors on 159M tokens from ShareGPT; as context, Vicuna‑33B reportedly used 370M tokens).
  - Also a `LLaMA-2-70B-Chat` variant for 70B experiments (Table 5).
- Vision encoder:
  - A frozen CLIP ViT image encoder is used. Two input resolutions are evaluated: 224×224 and 336×336 (Table 3a).

2) Two-stage “LLaVA lightning” training
- Stage 1: Feature alignment pre‑training
  - Purpose: connect visual features to the language model so the LLM can “read” the image.
  - Mechanism: train a learned linear projection that maps CLIP visual features (dimension 1024) to the LLM’s word embedding space:
    - 1024 → 6656 for 33B, and 1024 → 8192 for 65B (Section 2).
  - Data: 558K “concept‑balanced” LAION‑CC‑SBU subset.
  - Optimization: learning rate 1e‑4; linear decay; 3% warmup; no weight decay; sequence length 2048; DeepSpeed `ZeRO3`.
- Stage 2: Visual instruction tuning
  - Purpose: teach the model to follow multimodal instructions in realistic dialogue tasks.
  - Data:
    - Base: `LLaVA-80K` multimodal instruction dataset.
    - Optional mixing: add ShareGPT language‑only instruction data to balance language and multimodal skills (Section 2).
  - Trainable modules:
    - Full-model fine‑tuning (all LLM parameters) vs. parameter‑efficient methods: `LoRA` and `QLoRA`.
      - `LoRA`: injects low-rank adapters into specific weight matrices; only these small adapters are trained.
      - `QLoRA`: like LoRA, but keeps the base model in 4‑bit quantized form to save memory, still training low‑rank adapters.
  - Hyperparameters:
    - Full fine‑tuning: LR 2e‑5; 1 epoch.
    - LoRA/QLoRA: LR 1e‑4 (larger than full FT); LoRA alpha set to 2×rank (empirically crucial).
    - Sequence length: 2048; DeepSpeed `ZeRO3` (full FT, LoRA) or `ZeRO2` (QLoRA).

3) Compute setup and practicalities (Section 2; Table 4)
- Full FT: total batch size 512 on 4 nodes, each node has 8×A100‑80G.
- LoRA/QLoRA: total batch size 64; 33B on 1 node; 65B on 2 nodes.
- Cost reporting: GPU hours per node (Table 4), convertible to dollar cost via Azure ND A100 v4 pricing ($13.63/hour).

4) Decoding
- Beam search (sizes 1 and 5) is used at inference for LLaVA‑Bench; higher beam sizes marginally increase latency but improve scores (Table 1).

Why these design choices?
- Two‑stage training isolates the hard “vision‑language interface” learning (Stage 1) from instruction following (Stage 2), stabilizing optimization and reducing data needs.
- Higher image resolution (336 vs. 224) likely captures more fine details (OCR, small objects), which matters for multimodal tasks (Table 3a).
- Language‑only data mixing counteracts a known side‑effect of multimodal tuning: degradation of pure language ability; the study tests whether mixing helps both (Tables 3–5).
- `LoRA/QLoRA` enable large‑scale experiments under limited memory/compute, and the paper measures their quality‑cost trade‑offs (Table 4).

## 4. Key Insights and Innovations
1) Scaling up the LLM backbone consistently boosts multimodal performance
- What’s new: The paper quantifies the effect from 13B → 33B → 65B/70B under a constant training recipe.
- Why it matters: Results show clear gains in reasoning, generation, knowledge, recognition/OCR—capabilities crucial for real‑world assistants.
- Evidence:
  - LLaVA‑Bench overall increases from 13B to 33B to 65B (Table 1 and Table 3a).
  - MM‑VET “Total” improves from 32.9 (33B) → 35.5 (65B) → 36.4 with data mixing (Table 2).

2) Simple but high‑impact levers: higher image resolution and language‑data mixing
- Image resolution:
  - Moving from 224×224 to 336×336 yields consistent +2–3 point gains across 7B–65B on LLaVA‑Bench (Table 3a).
- Data mixing (ShareGPT + LLaVA‑80K):
  - Adds about +2 points on LLaVA‑Bench for large models (e.g., 33B: 72.0 → 73.9; 65B: 72.3 → 74.2; Table 3a).
  - Improves MM‑VET totals (e.g., 33B: 32.9 → 34.1; 65B: 35.5 → 36.4; Table 2).
- Significance: These levers are easy to adopt and bring consistent, non‑trivial gains.

3) `LoRA/QLoRA` delivers near‑full‑tuning quality at a fraction of cost
- Novelty: A careful, large‑model comparison with compute accounting.
- Evidence (Table 4):
  - For 13B, LoRA rank 64 matches full FT performance (70.1 vs. 70.1 on LLaVA‑Bench) while training only adapters.
  - For 33B–65B, increasing LoRA rank improves performance toward full FT but at much lower incremental cost than scaling the entire model.
  - QLoRA reduces memory and running time vs. LoRA and is necessary to fit 65B with DeepSpeed `ZeRO2`.
- Practical tip surfaced:
  - Large LoRA learning rate and alpha matter more than very high ranks. Lowering LR to 2e‑5 and alpha to 16 (rank 64) drops LLaVA‑Bench from 71.8 → 65.5; increasing rank from 64 → 128 → 512 yields only modest gains (65.5 → 66.1 → 68.1) under the same low‑LR/alpha setting (Table 4 discussion).

4) Visual instruction tuning can preserve or even improve pure language ability
- Insight:
  - After training solely on multimodal instruction data, LLaVA retains language capability (Vicuna‑80, MMLU) comparable to its LLM initializer; with certain mixtures it can even improve MMLU at 70B scale (Table 5).
- Evidence:
  - Quote from Table 5:
    > LLaMA‑2‑70B‑Chat MMLU: 63.1 → LLaVA‑70B (with data mix) MMLU: 65.1
  - For 33B and 65B, mixing language data helps multimodal scores but doesn’t uniformly boost Vicuna‑80 (e.g., 33B drops from 85.3 → 80.3; Table 5), revealing a nuanced trade‑off.

## 5. Experimental Analysis
- Evaluation datasets and metrics
  - `LLaVA‑Bench (In‑the‑Wild)` [24 images, 60 questions; scored by `gpt4‑0314` against gold responses; tasks: Conversation, Detail (long description), Reasoning] (Section 3.1; Table 1).
  - `MM‑VET` [200 images, 218 Qs; evaluates Recognition, OCR, Knowledge, Generation, Spatial, Math; scored by `gpt4‑0613`] (Section 3.1; Table 2).
  - `MM‑Bench` [2,974 questions; 6 reasoning/perception categories: LR, AR, RR, FP‑S, FP‑C, CP] (Table 3b).
  - Language‑only: `Vicuna‑80` (instruction following quality) and `MMLU` (multi‑task knowledge) (Table 5).
- Baselines
  - Open‑source LMMs (e.g., LLaVA‑7B/13B, MiniGPT‑4, BLIP‑2, InstructBLIP, etc.) and proprietary systems (Bing Chat, Bard) on LLaVA‑Bench; chain‑of‑tools systems (MM‑ReAct with GPT‑3.5/4) on MM‑VET for context (Tables 1–2).
- Main quantitative results
  - LLaVA‑Bench (Table 1):
    - With beam=5, `LLaVA‑33B` reaches 74.8 overall; `LLaVA‑65B` gets 74.4; both exceed the `LLaVA‑13B` 73.5 and Bing Chat 71.5.
    - Gains are strongest on Reasoning and Detail, consistent with larger LLMs’ language strength.
  - MM‑VET (Table 2):
    - `LLaVA‑65B (Data Mixing)`: 36.4±0.2 Total, improving over 33B (34.1±0.3) and previous open‑source end‑to‑end LMMs. Category gains are notable in Knowledge and Generation.
    - Quote:
      > LLaVA‑65B: Knowledge 26.2 → LLaVA‑65B (Data Mixing): 30.4; Generation 28.3 → 32.3
  - MM‑Bench (Table 3b):
    - Baseline `LLaVA‑7B` Overall 36.2 → `LLaVA‑33B` 55.7 → `LLaVA‑65B` 56.0 when combining 336px images and data mixing.
    - Quote:
      > LLaVA‑65B (336×336, mixed): LR 24.4, AR 24.4, RR 72.3, FP‑S 49.3, FP‑C 50.5, CP 68.1, Overall 56.0
  - Language‑only (Table 5):
    - `Vicuna‑33B` vs. `LLaVA‑33B` (no mix): Vicuna‑80 85.6 vs. 85.3, MMLU 59.0 vs. 56.1 (small drop on MMLU, preserved instruction following).
    - `LLaVA‑70B (with mix)` surpasses `LLaMA‑2‑70B‑Chat` on MMLU: 65.1 vs. 63.1.
- Cost vs. quality (Table 4)
  - Quote:
    > 13B (Full FT) performance 70.1; 13B (LoRA rank 64) performance 70.1; time 2.3 vs. 2.1 GPU‑hours per node (for 1 epoch).
  - For 33B and 65B, higher LoRA ranks improve toward full FT, but cost climbs much less steeply than scaling model size.
- Ablations and robustness
  - Scaling factors (Table 3a): model size, image resolution, and data mixing are all beneficial; effects are additive (e.g., 65B + 336px + mix ≈ strongest).
  - LoRA hyperparameters (Section 3.2): LR and alpha are more critical than very large ranks; QLoRA solves OOM for 65B in `ZeRO2`.
- Do the experiments support the claims?
  - Yes, with important caveats:
    - Multiple benchmarks, including a generalist test (MM‑VET) and category‑wise MM‑Bench, show consistent scaling trends.
    - However, both LLaVA‑Bench and LLM‑as‑judge evaluations can be sample‑limited or biased; the paper acknowledges LLaVA‑Bench is small (Section 3.1) and that findings are preliminary given dataset sizes (Section 4).
- Mixed or conditional results
  - Language‑data mixing helps multimodal performance but sometimes reduces Vicuna‑80 at 33B/65B (Table 5).
  - Spatial and Math categories see weaker/flat gains on MM‑VET (Table 2), suggesting vision encoder or training data may bottleneck those skills.

## 6. Limitations and Trade-offs
- Data scale and composition
  - Training datasets are relatively small (e.g., 558K feature‑alignment pairs; 80K multimodal instructions; limited ShareGPT language data), leading the paper to call the findings “preliminary” and to plan larger‑scale data studies (Section 4).
- Evaluation methodology
  - Heavy reliance on LLM‑as‑judge (GPT‑4 variants) may introduce scoring bias; LLaVA‑Bench is acknowledged as small and may not yield statistically significant differences (Section 3.1).
- Visual encoder scaling not explored
  - The CLIP encoder remains frozen and fixed; the paper explicitly leaves scaling the vision side (and vision‑heavy tasks like spatial/maths) to future work (Section 4).
- Trade‑offs in data mixing
  - Mixing language‑only instructions boosts multimodal metrics but can reduce instruction‑following scores (Vicuna‑80) at 33B/65B (Table 5), indicating a tuning‑target trade‑off that needs finer control.
- Compute and engineering constraints
  - Full fine‑tuning at 33B/65B is expensive; while `LoRA/QLoRA` mitigate cost and memory, optimal hyperparameters are sensitive (large LR/alpha needed), and benefits plateau with ever‑higher ranks (Table 4 and discussion).

## 7. Implications and Future Directions
- How this changes the field
  - Establishes that simply scaling the LLM backbone, increasing input resolution, and modestly mixing language data can reliably strengthen open‑source LMMs. It also validates `LoRA/QLoRA` as practical defaults for large‑scale multimodal tuning, lowering the barrier to strong 33B–65B baselines.
- Follow‑up research enabled/suggested
  - Vision‑side scaling:
    - Train or adapt stronger/finer‑grained vision encoders; test resolutions beyond 336px; incorporate multi‑crop or multi‑scale features to target Spatial/Math/OCR gains (cf. MM‑VET categories in Table 2).
  - Data curriculum and mixture design:
    - Systematically vary language vs. multimodal ratios, complexity (reasoning depth), and long‑form requirements to jointly optimize Vicuna‑80, MMLU, and MM‑VET.
  - More robust evaluation:
    - Larger, human‑rated multimodal benchmarks; standardized, compute‑aware leaderboards to complement LLM‑as‑judge scores.
  - Efficient fine‑tuning strategies:
    - Explore adapter placement, parameter sharing, and scheduling; hybrid strategies combining `LoRA/QLoRA` with selective full FT on critical modules.
- Practical applications
  - Deployable assistants that can reason over complex images and produce detailed, grounded language outputs for:
    - Document understanding and OCR‑heavy workflows (improved OCR/Knowledge/Generation in Table 2).
    - Safety/quality inspection and UI understanding (benefits from higher resolution; Table 3a).
    - Multimodal chatbots with improved reasoning and long‑form description capabilities (Table 1, Reasoning/Detail gains at larger model sizes).
  - Cost‑effective fine‑tuning of large LMMs for enterprise domains (Table 4 provides a blueprint to balance quality vs. GPU budget).

Overall, this work is a practical, empirical roadmap for scaling open‑source LMMs. The most actionable takeaways are: scale the LLM backbone when possible, always use higher image resolution (≥336px), mix in some language‑only instruction data to stabilize language skills, and prefer `LoRA/QLoRA` with sufficiently large LR/alpha for cost‑effective, near‑full‑tuning performance (Tables 3–5).
