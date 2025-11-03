# SAIL‑VL2 Technical Report

**ArXiv:** [2509.14033](https://arxiv.org/abs/2509.14033)
**Authors:** Weijie Yin, Yongjie Ye, Fangxun Shu, Yue Liao, Zijian Kang, Hongyuan Dong, Haiyang Yu, Dingkang Yang, Jiacong Wang, Han Wang, Wenzhuo Liu, Xiao Liang, Shuicheng Yan, Chao Feng
**Institutions:** 

## 🎯 Pitch

SAIL-VL2 is a groundbreaking open-source vision-language model that achieves state-of-the-art performance with significantly smaller models by utilizing an advanced vision encoder, innovative training techniques, and efficient data curation. This approach not only reduces computational demands, enabling broader deployment, but also demonstrates that strategic data and model alignment can rival the capabilities of much larger, costlier models, transforming future practices in multimodal AI.

---

## 1. Executive Summary (2–3 sentences)
SAIL‑VL2 is an open‑source suite of vision‑language models (LVMs) that aims to deliver “small model, strong performance” by combining a progressively trained vision encoder (`SAIL‑ViT`), a lean vision‑to‑language adapter, and dense or mixture‑of‑experts (MoE) Qwen3 LLM backbones. Through large‑scale data curation (especially for captions, OCR, charts, and video), staged pre‑training, and a thinking‑fusion SFT–RL pipeline, SAIL‑VL2 achieves state‑of‑the‑art results among sub‑4B and ~8B models across 106 benchmarks, and strong reasoning scores in OpenCompass (Tables 8–10, Figure 1).

## 2. Context and Motivation
- Problem/gap addressed
  - Modern vision‑language models (LVMs) achieve strong results by scaling parameters and data, but this can be computationally expensive and inefficient to deploy. SAIL‑VL2 targets the question: how to inject multimodal knowledge efficiently so that smaller models can match or exceed larger alternatives (Intro; “small model, strong performance”).
  - Key weaknesses in prior efforts include noisy multimodal data (especially captions/OCR/video), shallow alignment between visual and language spaces, and limited reasoning ability without heavy reliance on very large LLMs.

- Importance
  - Practical: Smaller models cut inference cost and latency, enabling wider deployment (mobile, enterprise, edge).
  - Scientific: Demonstrates that careful data curation, progressive alignment, and targeted reasoning post‑training can close performance gaps without brute‑force scale.

- Prior approaches and their limits
  - “Bigger is better” scaling (e.g., InternVL, Qwen2.5‑VL) brings accuracy but at high compute and cost.
  - Many models treat vision encoders as fixed, leaving modality gaps; others use generic instruction data that underemphasize OCR, charts, and videos; most reasoning pipelines focus on text‑only or require very large backbones.

- Positioning
  - SAIL‑VL2 builds on SAIL‑VL (Dong et al., 2025a) and contributes three pillars: (1) a rigorous data pipeline (caption quality judges, chart synthesis, video selection), (2) a progressive training framework from `SAIL‑ViT` alignment to multi‑task pre‑training, then thinking‑fusion SFT–RL, and (3) architectural coverage of dense and sparse MoE LLMs with infrastructure for efficiency (Figures 2–3; Sections 2–5).

## 3. Technical Approach
This section unpacks “how it works,” from the model architecture to training recipes and infrastructure.

- Overall architecture (Figure 2; Table 1)
  - Vision encoder: `SAIL‑ViT` (Section 2.1) encodes images/videos into visual tokens.
  - Adapter: a lightweight 2‑layer MLP projects visual embeddings into the LLM token space (Section 2).
  - LLM backbones: dense Qwen3‑1.7B/8B and sparse MoE Qwen3‑30B‑A3B (activates ~3B experts per token) jointly process text and projected visual tokens (Section 2; Table 1).
  - Any‑resolution option: `SAIL‑ViT‑AnyRes` preserves native image resolutions via interpolated positional embeddings, improving fine‑grained grounding (Section 2.1.2).

- `SAIL‑ViT`: progressive alignment of vision and language (Section 2.1.1; Figure 2)
  - Key idea: don’t freeze the visual backbone; gradually align it to the LLM’s representation with increasing task complexity.
  - Three stages (instruction‑style training throughout):
    1) Warm‑up: freeze ViT and LLM, tune only the adapter on 8M simple samples (captioning + OCR), LR 2e‑4, batch 1920.
    2) Fine‑grained alignment: unfreeze ViT + adapter, expand data (more caption/OCR + video‑caption), LR 2e‑5, batch 512.
    3) World knowledge injection: unfreeze all (vision, adapter, LLM) on 36.5M diverse data covering captions, OCR, open‑ended QA, math, short QA, and pure text; LR 1e‑5, batch 512.
  - Result: a vision encoder that produces features closer to the LLM space, empirically validated by both classification benchmarks (Table 6) and feature‑space distance metrics (Table 7, Figure 6).

- `SAIL‑ViT` family (Section 2.1.2; 6.1 Model Zoo)
  - Base ViT: 448×448 input, 32×32 patch grid (patch size 14 → 1024 tokens). High‑res images are tiled into 448×448 crops.
  - AnyRes: interpolates positional embeddings to match arbitrary resolution; supports up to 1792×1792 within a 16,384 token budget (Section 6.1).

- Multi‑modal Mixture‑of‑Experts (Section 2.2)
  - `MoE` means replacing some dense MLP layers with many parallel “experts,” while a gating network activates only a few per token, scaling parameters without proportional compute.
  - SAIL‑VL2 uses Qwen3‑MoE with:
    - Load‑balancing auxiliary loss and averaged activation across ranks for stability.
    - “Distribution‑aware” calibration: probe data to adjust expert activation entropy so text and multimodal activation patterns remain healthy (prevents expert collapse).

- Data curation at scale (Sections 3.1, 4.1; Figure 3)
  - `SAIL‑Caption2`: 300M captions cleaned to 250M using automated quality judges trained on 500K labeled samples. Two dimensions:
    - `VIR` (Visual Information Richness) and `ITA` (Image‑Text Alignment), each scored 1–5. Judge models exceed 90% precision/recall (Table 2).
    - Outcome: retain >99% estimated high‑quality after filtering; add 1.69M chart captions (400K synthetically rendered from code + 1.29M open datasets) (Section 3.1.1).
  - Synthetic VQA (`Caption2QA`): transform ~80M captions into diverse QA pairs with an LLM; scaling to 180M improves smoothly in a log trend (Section 3.1.2; Figure 4).
  - Video (`SAIL‑Video`): 5.1M filtered QA samples using three metrics scored by an LVM: alignment (−1–10), content richness (−1–7), difficulty (−1–8). Keep items with alignment≥5, content≥5, difficulty≥3 (Section 4.1.1).
  - Instruction (`SAIL‑Instruction2`): 20M diverse, high‑quality visual instructions built via latent‑class bucketing (semantic clustering) and re‑annotation for accuracy; includes more long‑answer/reasoning items (Section 4.1.2; Figure 5).

- Pre‑training pipeline (Section 3.2; Table 3)
  - Two phases after `SAIL‑ViT`:
    1) Basic multimodal pre‑training on 64M samples (captions, chart captions, OCR). Uses `AdaLRS` (Adaptive Learning Rate Search) to automatically raise LR from 2e‑4 up to ~6.75e‑4, improving final loss by >0.06 (Section 3.2.1–3.2.3).
       - `AdaLRS` uses a backtracking line‑search on the loss descent slope; if increasing LR improves loss‑reduction velocity, keep it; otherwise roll back and decrease LR. Equation (1) formalizes the LR update with slope estimates v(·) and scaling factors α′, β′.
    2) Multi‑task pre‑training on 180M samples mixing visual understanding and instruction‑tuning data (no AdaLRS here due to weak loss–performance correlation). Resampling occurs at two levels:
       - Dataset‑level balancing to mix distributions (basic stage).
       - Linguistic n‑gram balancing to fight phrasing homogenization in synthetic data (multi‑task stage) (Section 3.2.2).
  - Scaling‑law: training up to 360B tokens shows monotonic gains on overall, natural VQA, and OCR VQA benchmarks (Figure 4).

- Post‑training for instruction following and reasoning (Section 4; Table 4)
  - Basic SFT: staged curriculum—world knowledge (Infinity‑MM Stage2) → `SAIL‑Instruction2` → harder reasoning subsets (LLaVA‑CoT, MMPR, Condor) → mixed 1:1 image:video phase (with `SAIL‑Video`). “Model soup” (merging homogeneous runs) yields reliable gains; mixing heterogeneous runs degrades performance (Table 5).
  - LongCoT SFT: build a 400K high‑quality multimodal Chain‑of‑Thought corpus with consistent formatting (`<think> ... </think>`; answer in `\boxed{}`), strict cleaning (redundancy filter by token overlap; answer distillation; CoT length balancing). Train for 1 epoch, batch 1024, cosine LR 1e‑6; objective is next‑token prediction over thought+answer, L_LongCoT in Equation (2).
  - RL with verifiable rewards: 70K challenging problems curated via pass@4 filters; two binary rewards—answer correctness (in `\boxed{}`) and format adherence (`<think> ... </think>`). PPO‑based optimizers differ by backbone: DAPO for dense, GSPO for MoE; context 16,384; max generation 4096; 2048 rollouts/episode; 8 PPO updates/episode; LR 1e‑6; dynamic clip 0.20–0.28 (Section 4.2.3).
  - Think‑Fusion SFT: 1M examples with 90% direct QA and 10% high‑quality CoT traces harvested from the RL stage via rejection sampling; train with a dual‑loss objective (Equation (3)) that conditions loss on different formats (Section 4.2.4).
  - RL with mixed rewards: curate “hard cases” (50K) + 50K general samples (LLaVA‑OneVision) to maintain breadth. Mixed reward = weighted combination of answer, thought‑quality (judge‑scored), and format (all binary); same PPO setup as before (Section 4.2.5). Note: the narrative mentions both 100K and 150K samples here—an inconsistency flagged in Limitations.

- Efficiency infrastructure (Section 5)
  - Stream packing: concatenate variable‑length sequences to minimize padding; maintain correct positions and masks; online packing from per‑node buffers. Visual packing additionally balances visual token counts across devices, which is critical for AnyRes inputs. Gains: nearly 2× SM utilization, ~50% faster training, +0.7% average accuracy on open‑ended QA; visual packing yields a further ~48% efficiency gain (Section 5.1).
  - MoE infra: kernel fusion for expert ops (up to 3× speedup), optimized attention/LayerNorm; distributed strategies differ by hardware (Megatron on NPUs with pipeline+expert parallelism; DeepSpeed ZeRO‑2 with CPU offload on NVIDIA GPUs) (Section 5.2).

## 4. Key Insights and Innovations
- Large‑scale, quality‑controlled multimodal data curation that targets hard modalities (Section 3.1; Table 2; Figure 3)
  - Novelty: automated `VIR`/`ITA` caption judges trained on balanced labels bring >90% precision/recall (Table 2), enabling economical filtering of 300M captions down to a high‑quality 250M. A code‑driven chart synthesis engine and consistent video filtering with alignment/content/difficulty scores create focused corpora where LVMs often struggle.
  - Significance: boosts pre‑training efficiency and downstream OCR/chart/video performance (Tables 8–9).

- Progressive alignment of the vision encoder to the language space (Section 2.1; Table 6; Table 7; Figure 6)
  - Novelty: three‑stage training that explicitly unfreezes the ViT and LLM at the right times, rather than “frozen vision encoder + adapter only.”
  - Significance: `SAIL‑ViT` features move measurably closer to text embeddings (lower nearest‑neighbor and Wasserstein distances across LLM sizes in Table 7; tighter overlap in Figure 6), and visual classification improves over AIMv2 baselines (average +2.11% for Huge, Table 6).

- AdaLRS: loss‑guided adaptive LR search during basic multimodal pre‑training (Section 3.2.3; Eq. 1)
  - Novelty: a simple line‑search‑style scheduler that probes LR increases and rolls back if the loss slope worsens.
  - Significance: automatically finds a better LR (from 2e‑4 to ~6.75e‑4), yielding >0.06 final‑loss improvement without manual sweeps (Section 3.2.1).

- Thinking‑fusion training (SFT–RL cycle) with format‑aware, partly verifiable rewards (Sections 4.2.2–4.2.5)
  - Novelty: staged LongCoT SFT → verifiable‑reward RL → Think‑Fusion SFT mixing 90% direct QA + 10% curated CoT → mixed‑reward RL. The use of `<think>` tags and `\boxed{}` answers standardizes supervision and reward parsing; rejection sampling harvests “best” CoTs from the model’s own RL rollouts.
  - Significance: strong reasoning at modest scales. The 8B‑Thinking model reaches 54.4 average on OpenCompass reasoning (Table 10)—competitive with GPT‑4o‑latest (54.8) and above many open‑source peers.

- Training‑efficiency engineering that also improves quality (Section 5.1)
  - Novelty: joint stream+visual packing explicitly balances both text and visual token loads across devices—rarely reported with quantified gains in LVMs.
  - Significance: up to ~1.5× faster training + ~0.7% accuracy gains on long‑context QA (Section 5.1).

## 5. Experimental Analysis
- Evaluation protocol and baselines (Section 6.1)
  - Benchmarks: 106 datasets spanning general multimodal understanding, math/reasoning, multi‑image/video, plus OpenCompass (8 datasets) and multiple video sets.
  - Judging and comparability:
    - For “basic” models: custom VLMEvalKit with Doubao‑1.5‑vision‑pro as judge; all baselines re‑evaluated in the same setting.
    - For “thinking” models: official OpenCompass leaderboard except two models (SAIL‑VL2‑A3B‑Thinking and Keye‑VL‑8B‑Thinking) evaluated with GPT‑4o‑Mini in OpenCompass‑aligned settings (Section 6.1). This is mostly fair but mixes judge models—see Limitations.

- Main quantitative results (Figures 1; Tables 8–10)
  - 2B scale (Table 8):
    - OpenCompassavg 70.31 vs Qwen2.5‑VL‑3B 65.36, InternVL3.5‑2B 66.64.
    - OCR/Docs: OCRBench 89.5, DocVQA 93.10 (leading among <4B).
    - Reasoning subsets: MathVista‑mini 71.10 (strong for size), MMMU‑val 47.67 (competitive).
    - AnyRes improves grounding: RefCOCOavg 57.82 vs 53.28 for fixed‑res 2B.
  - 8B scale (Table 9):
    - OpenCompassavg 75.07 vs InternVL3.5‑8B 73.49; OpenSourceavg 57.20.
    - OCR/Docs: DocVQA 95.28; OCRBench 91.30 (top tier).
    - Reasoning: MMMU‑val 55.44; MathVerse‑mini 43.17; MathVista‑mini 76.40.
    - Multi‑image/video: TempCompassavg 65.66; LongVideoBench‑val 58.34.
  - Thinking models (Table 10):
    - `SAIL‑VL2‑8B‑Thinking` average 54.4 across MathVista, MathVision, MathVerse, DynaMath, WeMath, LogicVista—best among open‑source models listed, close to GPT‑4o‑latest 54.8 and above Gemini‑2.0‑Flash 50.6.
    - MoE thinking with ~3B active parameters (`A3B‑Thinking`) averages 53.6, surpassing several larger closed‑source models and open‑source thinkers.

- Ablations and diagnostics
  - `SAIL‑ViT` vs AIMv2 in zero‑shot classification shows consistent improvements across ImageNet variants (Table 6).
  - Feature‑space alignment: distances to text embeddings reduced with `SAIL‑ViT` across LLM sizes (Table 7; Figure 6).
  - Scaling curves: larger multi‑task pre‑training budget monotonically improves metrics (Figure 4).
  - Model soup: merging homogeneous runs boosts performance (AVG 76.60 vs bases ~74.5) while heterogeneous merging can catastrophically fail (AVG 12.86) (Table 5)—a cautionary result.

- Convincingness
  - Breadth: 106 datasets and separate video evaluations provide wide coverage.
  - Depth: The reasoning leaderboard results (Table 10) support the efficacy of the SFT–RL pipeline.
  - Causality evidence: The paper ties specific design choices to measurable effects (e.g., AnyRes → better RefCOCO, AdaLRS → lower loss, packing → faster training + small accuracy gains, `SAIL‑ViT` → closer feature spaces and higher ImageNet averages).

- Representative quotes of outcomes
  > “SAIL‑VL2‑2B … achieves state‑of‑the‑art average performance on OpenCompass among officially released open‑source models under the 4B scale.” (Figure 1a; Table 8)

  > “SAIL‑VL2‑8B‑Thinking … establishes a new state‑of‑the‑art for open‑source models … 54.4 average” (Table 10)

  > “Data packing nearly doubles SM utilization and accelerates training by 50%, … visual packing … further 48% gain … +0.7% average improvement on open‑ended QA” (Section 5.1)

## 6. Limitations and Trade‑offs
- Mixed evaluation judges and potential comparability issues
  - Basic models are judged with Doubao‑1.5‑vision‑pro; thinking models mostly with OpenCompass, but two (including SAIL‑VL2‑A3B‑Thinking) are judged via GPT‑4o‑Mini (Section 6.1). While settings are “aligned,” cross‑judge variance may affect fine‑grained comparisons.

- Reward design and transparency
  - RL uses binary rewards for answer/format and judge‑based think quality for mixed‑reward RL (Sections 4.2.3, 4.2.5). Coefficients for the mixed reward are not disclosed; sensitivity analysis is absent.

- Data scale, compute, and reproducibility
  - The report mentions training on 776B tokens overall (Intro highlights), and tables detail large budgets (Table 3; Table 4), but compute hours, hardware counts, and per‑stage wall‑clock are not reported—important for practitioners planning replication.

- Minor inconsistencies
  - The mixed‑reward RL dataset is described as 100K (50K hard + 50K general) in “Data Curation,” yet “Training Recipe” refers to 150K samples (Section 4.2.5). Clarification is needed.

- Safety and bias
  - While hallucination is evaluated (HallusionBench, Tables 8–9), there is no targeted safety analysis (e.g., bias across languages/layouts in OCR, robustness to adversarial charts). The heavy use of synthetic/Q&A‑converted data may induce stylistic biases, though the authors add n‑gram resampling to mitigate this (Section 3.2.2).

- Scope and edge cases not fully explored
  - Long‑video comprehension is evaluated with 16 sampled frames (Tables 8–9 notes), which may underrepresent models optimized for dense temporal reasoning.
  - The AnyRes path improves RefCOCO, but the computational cost vs. benefit across tasks is not quantified; similarly, MoE training stability is discussed qualitatively with load‑balancing, but detailed failure rates/mitigations are not reported.

## 7. Implications and Future Directions
- Impact on the field
  - Demonstrates that careful data engineering and progressive alignment can push small‑to‑mid‑size LVMs to the top of open‑source leaderboards (Figure 1; Tables 8–10). This challenges the default “scale parameters first” strategy and provides a replicable recipe focused on data quality + staged training.

- What it enables next
  - Research:
    - Understanding which components drive reasoning gains: controlled ablations on 90/10 Think‑Fusion mixes, mixed‑reward coefficients, pass@k thresholds, and `<think>`/`\boxed{}` formatting.
    - More principled expert‑activation calibration for MoE under multimodal distributions (extend entropy probing to cross‑modal gates).
    - Robustness studies on OCR/chart/video under distribution shifts and low‑resource languages.
  - Engineering:
    - Wider adoption of stream+visual packing in multimodal training pipelines.
    - Reusable caption quality judges and chart synthesis tools for community datasets; the `Caption2QA` approach for scalable VQA pre‑training.

- Practical applications
  - High‑fidelity document OCR and layout understanding (DocVQA 95.28 at 8B) for enterprise automation.
  - Chart/table comprehension and data‑viz QA in BI tools (large chart corpus; strong ChartQA/DocVQA).
  - Efficient visual assistants on edge/cloud with good reasoning (OpenCompass reasoning 54.4 at 8B‑Thinking; MoE‑Thinking 53.6 with ~3B active parameters).
  - Video QA for surveillance, education, and media indexing, aided by curated video alignment/difficulty filtering and AnyRes support.

In sum, SAIL‑VL2 contributes a well‑engineered pathway to high‑accuracy, cost‑efficient multimodal models: clean and diverse data, progressively aligned `SAIL‑ViT`, adaptive pre‑training (AdaLRS, resampling), and a compact reasoning pipeline (LongCoT → PPO RL → Think‑Fusion → mixed‑reward RL). The results across 106 datasets and OpenCompass reasoning are strong, though future work should tighten evaluation comparability, reveal RL reward settings, and deepen robustness/safety analyses.
