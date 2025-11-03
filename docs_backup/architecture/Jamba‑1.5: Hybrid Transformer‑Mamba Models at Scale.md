# Jamba‑1.5: Hybrid Transformer‑Mamba Models at Scale

**ArXiv:** [2408.12570](https://arxiv.org/abs/2408.12570)
**Authors:** Jamba Team, Barak Lenz, Alan Arazi, Amir Bergman, Avshalom Manevich, Barak Peleg, Ben Aviram, Chen Almagor, Clara Fridman, Dan Padnos, Daniel Gissin, Daniel Jannai, Dor Muhlgay, Dor Zimberg, Edden M Gerber, Elad Dolev, Eran Krakovsky, Erez Safahi, Erez Schwartz, Gal Cohen, Gal Shachaf, Haim Rozenblum, Hofit Bata, Ido Blass, Inbal Magar, Itay Dalmedigos, Jhonathan Osin, Julie Fadlon, Maria Rozman, Matan Danos, Michael Gokhman, Mor Zusman, Naama Gidron, Nir Ratner, Noam Gat, Noam Rozen, Oded Fried, Ohad Leshno, Omer Antverg, Omri Abend, Opher Lieber, Or Dagan, Orit Cohavi, Raz Alon, Ro'i Belson, Roi Cohen, Rom Gilad, Roman Glozman, Shahar Lev, Shaked Meirom, Tal Delbari, Tal Ness, Tomer Asida, Tom Ben Gal, Tom Braude, Uriya Pumerantz, Yehoshua Cohen, Yonatan Belinkov, Yuval Globerson, Yuval Peleg Levy, Yoav Shoham
**Institutions:** AI21 Labs

## 🎯 Pitch

Jamba-1.5 revolutionizes long-context language modeling by integrating Transformer attention, Mamba state-space layers, and Mixture-of-Experts with an innovative INT8 quantization technique, enabling efficient processing of up to 256K tokens on standard 8×80GB GPU setups. This breakthrough makes high-quality, long-context inference not only feasible but cost-effective, opening new possibilities for real-world applications like legal analytics and conversational AI, while setting new expectations for large-scale, open-weight models.

---

## 1. Executive Summary (2-3 sentences)
Jamba-1.5 introduces large, open-weight language models that combine Transformer attention, Mamba state-space layers, and Mixture-of-Experts (MoE) to achieve extremely long effective context (256K tokens) with far lower memory and latency than similarly capable models. The work matters because it makes long-context, high-quality inference feasible on common 8×80GB GPU servers, through architectural choices and a practical, fast quantization method (ExpertsInt8) that preserves quality while enabling cost-effective serving (Sections 2, 3.1; Table 1; Figures 3–4).

## 2. Context and Motivation
- Problem addressed
  - Long-context inference is memory- and latency-intensive in standard Transformer models due to quadratic attention and large key-value (KV) caches. Most open-weight models either cannot use very long contexts in practice or degrade sharply when they do (Table 4, RULER results).
  - Serving very large models with long contexts on commodity enterprise hardware (e.g., 8×80GB GPUs) is challenging: KV caches dominate memory budgets and quantization options like FP8 are not universally available (Section 3.1).

- Importance
  - Real-world applications increasingly need long-context processing: multi-document reasoning, legal and financial due diligence, conversational memory, large tables, and long-form code or research review. Reducing the memory and latency costs while keeping quality makes these use cases practical at lower cost (Figures 3–4).

- Prior approaches and shortcomings
  - Pure Transformers scale poorly with context due to quadratic attention and KV cache growth.
  - State-space models (SSMs), like `Mamba`, scale linearly in sequence length but can be weaker alone on certain tasks that benefit from global pooling (full attention).
  - MoE increases parameter capacity with sparse activation, but serving large MoE models efficiently—especially with long contexts—requires careful memory management and quantization.
  - Existing quantization approaches either need calibration (e.g., GPTQ) or require hardware support (FP8 on H100), and may not preserve quality or speed at scale (Section 3.1; Figure 2).

- Positioning
  - Jamba-1.5 scales the previously introduced Jamba hybrid architecture (mixing attention and Mamba) to larger active capacity, and integrates MoE with a new quantization technique. Its goal is to retain or improve quality while dramatically reducing latency and memory, especially for long contexts (Sections 1–2, 3.1; Table 1).

## 3. Technical Approach
This section unpacks how the models are built and served.

- Model sizes and capacity
  - `Jamba-1.5-Large`: 398B total parameters, 94B “active” parameters (Section 2).
  - `Jamba-1.5-Mini`: 52B total parameters, 12B active parameters (Table 1).
  - Definition: `active parameters` are those actually used per token during inference. In MoE, only a subset of experts fire for a token, so the model’s total parameters exceed the per-token active count.

- Hybrid architecture: attention + Mamba + MoE (Section 2)
  - Structure: 9 “blocks,” each with 8 layers (`l=8`).
  - Attention-to-Mamba ratio: `a:m = 1:7`. That is, one full attention layer interleaved with seven Mamba layers per block (found optimal in prior Jamba work and validated by follow-ups; Section 2).
  - MoE in place of dense MLP every `e=2` layers, with `n=16` experts; the top `K=2` experts are selected for each token (top-2 gating).
  - Hidden dimension: 8192; attention heads: 64 query heads and 8 KV heads.
  - Why this matters:
    - Fewer attention layers and fewer KV heads directly reduce KV cache size.
    - Mamba layers provide linear-time sequence modeling, so most layers avoid quadratic attention costs but still receive global signal periodically from the full attention layers.
    - MoE increases capacity without proportionally increasing per-token computation.

- Why Mamba-1 with attention (not Mamba-2) (Figure 1; Section 2)
  - Finding: in hybrid settings, `Mamba-1 + Attention` outperforms `Mamba-2 + Attention` at scales tested (350M and 1.3B for 100B tokens of training).
  - Rationale offered: Mamba-2’s advantages (e.g., larger state size) are less crucial when global attention layers intermittently pool information across the entire context (Section 2, Figure 1). The hybrid model benefits more from strategically placed full attention than from a stronger SSM alone.

- KV cache reduction (Table 1)
  - KV cache stores key/value tensors for attention; its memory scales with context length and number of attention heads.
  - At 256K tokens, reported KV cache memory:
    - `Jamba-1.5-Large`: 9 GB
    - `Jamba-1.5-Mini`: 4 GB
    - Comparators: LLaMA-3.1-70B: 80 GB; Mistral-Large-2: 88 GB; Mixtral-8x22B: 56 GB; several 7–9B models still at ~32 GB (Table 1).
  - Mechanistically, the small KV cache reflects the hybrid design (few attention layers) and few KV heads (8), whereas query heads are larger in number (64) but do not increase KV cache.

- ExpertsInt8 quantization (Section 3.1; Figure 2)
  - What it does:
    - Observations: >85% of weights are in MoE layers; >90% are in MoE+MLP layers.
    - Step 1: Quantize MoE and MLP weights to INT8 and store them as INT8.
    - Step 2: Dequantize on-the-fly to BF16 immediately before computation—but crucially this dequantization happens inside the fused MoE kernel (`fused_moe`) in the vLLM serving runtime.
  - Why it’s fast and accurate:
    - Dequantization is fused with compute, so it adds negligible overhead. It reduces data movement time because weights move from HBM to SRAM in smaller INT8 form before compute (Section 3.1).
    - No calibration data needed (unlike GPTQ); quantization finishes within seconds at model load.
    - Works with A100 (unlike FP8 which requires H100). Maintains BF16 activations to avoid quality loss (Section 3.1).
  - Empirical comparisons (Figure 2):
    - On H100, ExpertsInt8 matches FP8 latency across batch sizes for Jamba-1.5 models and Mixtral models.
    - On A100, ExpertsInt8 significantly outperforms GPTQ and is available where FP8 is not.

- Activation Loss to control very large activations (Section 3.2)
  - Issue observed during pretraining: some expert outputs and last Mamba outputs grew to magnitudes up to 4×10^6 for certain tokens. This isn’t a training problem in BF16, but it creates risk for inference when activations are FP16-limited (max ~64K).
  - Solution: add an auxiliary “Activation Loss” proportional to the mean-square of activations with factor `α` to penalize large values.
    - For Jamba-1.5-Large, using `α = 1e-5` reduced max activations to ~2–3K.
    - The loss can be turned on late in training, shrinks activations quickly, and does not hurt evaluation quality; FP16-activation inference achieved the same scores as BF16 without NaNs/overflows (Section 3.2).

- Training and post-training (Sections 5.1–5.4)
  - Training infrastructure: H100 GPUs; in-house framework with FSDP, tensor/sequence/expert parallelism; MoE training adapted from MegaBlocks (Section 5.1).
  - Three stages (Section 5.2):
    - Pretraining: multilingual mixture of web, code, books, scientific articles (last updated March 2024), with parsing, filtering, deduplication.
    - Mid-training: emphasis on long documents to strengthen long-range abilities.
    - Post-training (SFT): targeted conversational skills and long-context retention via synthetic data pipelines (Section 5.3).
      - Synthetic data generators include: Table-based QA (textified tables), Document QA with embedded answer paragraphs, Tool use/function calling (based on and extending Glaive FC v2), and steerability scenarios validated by automatic checks and a reward model.
      - Both sizes share control tokens and a Hugging Face–compatible chat template.

- Serving practicality (Sections 2, 3.1; Figures 3–4)
  - With ExpertsInt8 and the hybrid architecture, `Jamba-1.5-Large` serves 256K-token contexts on a single 8×80GB GPU machine—where even LLaMA-3.1-405B cannot run long contexts on the same hardware (Figure 4).

## 4. Key Insights and Innovations
- Hybrid at scale that actually improves quality and efficiency (Sections 1–2; Figure 1)
  - Novelty: Scaling a `Transformer + Mamba + MoE` architecture to 94B active parameters while keeping memory for 256K contexts extremely low.
  - Significance: Demonstrates that a small fraction of full-attention layers interleaved with Mamba can outperform stronger SSMs (Mamba-2) when attention is present, validating the design choice for large models (Figure 1).

- ExpertsInt8: MoE-aware, fused dequantization quantization (Section 3.1; Figure 2)
  - Novelty: INT8 quantization of MoE/MLP weights with on-the-fly dequantization inside a fused MoE kernel, avoiding calibration and preserving BF16 activations.
  - Significance: Matches FP8 latency on H100, works on A100 where FP8 is unavailable, and surpasses GPTQ latency without quality loss—making large MoE models tractable for long-context serving.

- Activation Loss for safe FP16 activations at inference (Section 3.2)
  - Novelty: A simple, late-stage auxiliary loss that quickly suppresses outlier activations without hurting accuracy.
  - Significance: Expands inference options (FP16 activations) and stability, which is important for production serving stacks that default to FP16 activations.

- Extreme KV cache reduction without sacrificing quality (Section 2; Table 1; Figures 3–4)
  - Mechanism: Few attention layers, few KV heads (8), extensive Mamba usage, and MoE for capacity.
  - Significance: Enables 256K effective context with single-node serving for a 94B-active-parameter model—previously impractical for open-weight models at this scale.

## 5. Experimental Analysis
- Evaluation methodology overview
  - Standard academic benchmarks: MMLU, MMLU-Pro, GPQA, ARC-Challenge (ARC-C), BBH, HumanEval, GSM8K; instruction following (IFEval) and function calling (BFCL); safety metrics (RealToxicity, TruthfulQA) (Table 2).
  - Chatbot evaluations: Arena-Hard and WildBench (Table 3).
  - Long-context: RULER (13 synthetic tasks) and Infinite-BENCH (novel comprehension MC/QA with ~100K average lengths) (Tables 4–5).
  - Multilingual: mMMLU across seven languages (Table 6).
  - Throughput/latency: End-to-end latency and throughput measured on fixed hardware (Mini: 2×A100; Large: 8×A100) across context lengths up to 256K (Figures 3–4).
  - Quantization runtime: Latency comparisons across quantization settings and GPUs (Figure 2).

- Headline quantitative results (selected)
  - KV cache at 256K tokens (Table 1):
    - Jamba-1.5-Large: 9 GB; Jamba-1.5-Mini: 4 GB; LLaMA-3.1-70B: 80 GB; Mistral-Large-2: 88 GB; Mixtral-8x22B: 56 GB.
  - Standard academic (Table 2; selected):
    - MMLU (5-shot): Large 80.0 vs LLaMA-3.1-70B 83.6; Mistral-L2 82.5.
    - MMLU-Pro (5-shot): Large 48.3 vs LLaMA-3.1-70B 53.0; Mistral-L2 54.2.
    - ARC-C (0-shot): Large 93.0 vs LLaMA-3.1-70B 94.8.
    - HumanEval (pass@1): Large 71.3 vs LLaMA-3.1-70B 80.5; Mistral-L2 92.
    - IFEval (0-shot): Large 81.5 vs LLaMA-3.1-70B 87.5.
    - BFCL (function calling, 0-shot): Large 85.5 vs LLaMA-3.1-70B 84.8; Mistral-L2 85.1.
  - Chatbot (Table 3):
    - Arena-Hard: Large 65.4 vs LLaMA-3.1-70B 55.7; Mistral-L2 70.4.
    - WildBench: Large 48.5 vs LLaMA-3.1-70B 49.8; Mistral-L2 56.3.
  - Long-context (Table 4, RULER):
    - Effective length: both Jamba-1.5 models confirmed at 256K.
    - Average score: Large 95.7 (with 93.9 at 256K); Mini 92.6 (with 86.1 at 256K).
    - Competitors drop sharply at long contexts: e.g., Mistral-L2 average 80.5, 23.7 at 128K; LLaMA-3.1-70B average 89.6, 66.6 at 128K; GPT-4-1106-preview average 91.6, 81.2 at 128K.
  - Infinite-BENCH long novels (Table 5):
    - EN.MC: Large 80.4 vs LLaMA-3.1-70B 78.2; Mini 76.9 vs LLaMA-3.1-8B 65.1.
    - EN.QA: Large 34.9 vs LLaMA-3.1-70B 36.7; Mini 40.6 vs LLaMA-3.1-8B 27.1.
  - Multilingual mMMLU (Table 6; average across 7 languages):
    - Large: 73.94 vs LLaMA-3.1-70B 77.76; Mistral-L2 76.19.
    - Mini: 64.30 vs LLaMA-3.1-8B 56.83; Gemma-9B 63.34.
  - Throughput and latency (Figures 3–4):
    - Both Jamba-1.5 models exhibit notably lower end-to-end latency across context lengths, with a widening advantage at long contexts; throughput (tokens/sec) remains competitive. LLaMA-3.1-405B cannot run beyond ~64K–100K on 8×80GB GPUs, while Jamba-1.5-Large runs efficiently up to 256K (Figure 4).
  - Quantization runtime (Figure 2):
    - On H100, ExpertsInt8 ≈ FP8 latency.
    - On A100, ExpertsInt8 clearly faster than GPTQ and available where FP8 is not.

- Ablations and diagnostics
  - Mamba-1 vs Mamba-2 in hybrid: hybrid with Mamba-1 performs better than hybrid with Mamba-2 at comparable training scale (Figure 1).
  - Activation Loss: reduces extreme activations and enables FP16-activation inference with unchanged evaluation results (Section 3.2).

- Support for claims
  - The long-context capability is directly and comprehensively evidenced on RULER (Table 4) and Infinite-BENCH (Table 5).
  - Efficiency claims are supported by KV cache comparisons (Table 1), hardware-constrained latency/throughput curves (Figures 3–4), and quantization latency head-to-heads (Figure 2).
  - General capability parity is reflected in Table 2: Jamba-1.5-Large generally sits close to LLaMA-3.1-70B and Mistral-L2 across standard tasks, while not always leading.

- Where results are mixed or conditional
  - On some reasoning/coding metrics, LLaMA-3.1-70B or Mistral-L2 lead (e.g., HumanEval 92 for Mistral-L2; MMLU-Pro 54.2 for Mistral-L2).
  - On chat benchmarks, Jamba-1.5-Large beats LLaMA-3.1-70B on Arena-Hard but trails Mistral-L2 on both Arena-Hard and WildBench (Table 3).
  - The dominant strength is long-context performance and serving efficiency.

## 6. Limitations and Trade-offs
- Quality trade-offs vs top peers (Table 2)
  - Jamba-1.5-Large is competitive but not the absolute leader on several standard benchmarks (e.g., MMLU-Pro, HumanEval). The architecture trades some peak accuracy for major gains in context length and serving efficiency.

- Synthetic post-training and evaluation bias (Section 5.3; Table 3)
  - Post-training relies heavily on synthetic data pipelines. While carefully filtered and validated, synthesis may bias model behavior toward the training generators’ style or coverage.
  - Chatbot evaluations use GPT-4-Turbo judges (Arena-Hard, WildBench), which are widely used but may introduce judge-model bias.

- Reproducibility of data and recipes (Section 5.1–5.4)
  - The full pretraining corpus is proprietary, and many training recipe choices (data mixtures, detailed hyperparameters) are not fully enumerated, limiting full reproducibility despite open weights.

- Edge cases not deeply explored
  - Robustness to adversarial long-context perturbations, detailed failure modes in retrieval-heavy contexts, and catastrophic forgetting of long-range skills after aggressive instruction tuning are not deeply dissected beyond the reported benchmarks.

- Hardware-specific assumptions
  - While ExpertsInt8 is hardware-friendly (works on A100 and H100), the performance comparisons are reported on specific GPU configs (2×A100 for Mini; 8×A100 for Large; some H100 comparisons in Figure 2). Actual gains may vary with different interconnects, memory bandwidths, or inference engines.

## 7. Implications and Future Directions
- Field impact
  - Demonstrates that hybrid `Attention + SSM + MoE` architectures are practical at very large scales and can deliver state-of-the-art effective context length (256K) with striking efficiency. This resets expectations for what long-context open-weight models can do on mainstream 8×80GB servers (Table 1; Figures 3–4; Table 4).

- Practical applications
  - Long-document analytics (e.g., legal, finance, research review), multi-file code comprehension, retrieval-augmented generation with large evidence sets, complex tool-using agents that keep long conversational memory, and table-heavy enterprise workflows (Section 5.3 data pipelines hint at these).

- Research directions enabled
  - Architecture: Explore different attention-to-Mamba ratios, placement, and state sizes now that large-scale evidence suggests hybridization beats stronger SSMs when attention is present (Figure 1).
  - MoE serving: Extend ExpertsInt8-style fused quantization to other sparse layers and kernels; investigate accuracy-speed trade-offs for more aggressive quantization while keeping BF16 activations.
  - Long-context training: Systematize “mid-training” on long documents, and study instruction-tuning mixes that preserve long-range skills without regression—especially beyond synthetic tasks.
  - Safety and multilingual: Improve alignment without eroding long-context competency; scale non-English post-training beyond limited language coverage (Table 6).
  - Evaluation: Develop more naturalistic long-context benchmarks and analyze failure modes beyond RULER’s synthetic tasks and Infinite-BENCH’s literary QA.

Quote highlights that capture the core empirical contributions:
- “Jamba-1.5-Large can be served on a single machine with 8 80GB GPUs with context lengths up to 256K tokens” (Sections 1–2; Table 1; Figures 3–4).
- “Our quantization matches FP8 in latency … applicable to A100 GPUs, where FP8 is unavailable” (Section 3.1; Figure 2).
- “Jamba-1.5 models are the only ones with a confirmed effective length of 256K tokens [on RULER]” (Table 4).
- “Adding [Activation Loss] reduced the activations almost instantly … with FP16 activations we obtained the same results as the BF16 evaluations without any nans/overflows” (Section 3.2).

Overall, Jamba-1.5 offers a clear recipe—hybrid layers + MoE with fused INT8 dequantization and activation control—for building long-context LLMs that are not only capable but also economical to serve, shifting the practical frontier for enterprise-scale, long-input applications.
