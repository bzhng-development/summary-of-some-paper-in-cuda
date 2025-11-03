# The Llama 3 Herd of Models

**ArXiv:** [2407.21783](https://arxiv.org/abs/2407.21783)
**Authors:** Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al‑Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, Anirudh Goyal, Anthony Hartshorn, Aobo Yang, Archi Mitra, Archie Sravankumar, Artem Korenev, Arthur Hinsvark, Arun Rao, Aston Zhang, Aurelien Rodriguez, Austen Gregerson, Ava Spataru, + many others (total ~533 authors)
**Institutions:** Meta (Meta AI)

## 🎯 Pitch

Llama 3 introduces an open, frontier-scale language model with up to 405 billion parameters, excelling in multilinguality, long-context processing, and tool use through a simple dense-Transformer architecture. It democratizes research opportunities by providing a reproducible methodology for compute-optimal scaling and advanced safety measures, offering significant implications for AI development and deployment across diverse fields.

---

## 1. Executive Summary (2-3 sentences)
Llama 3 is a publicly released family of foundation models (8B, 70B, 405B parameters) trained at unprecedented open scale (3.8×10^25 FLOPs on 15.6T tokens) with a 128K-token context window, native tool use, and strong multilingual, coding, and reasoning abilities. It couples a deliberately simple dense-Transformer design with a data-centric training recipe, a compute-optimal scaling-law methodology, and a safety stack (including the released Llama Guard 3) to reach performance comparable to frontier models while remaining open (Sections 1, 3.2.1, 5.2; Table 2).

## 2. Context and Motivation
- Problem addressed
  - The field lacks an open, “flagship-grade” language model that is competitive with top closed systems, supports long context, tool use, multilinguality, and comes with a reproducible, scalable training and safety methodology (Section 1).
- Why it matters
  - Real-world impact: long-context assistants (contracts, codebases), safe tool-using agents (search, Python, computation engines), and multilingual access are practical requirements. Opening a 405B model lowers barriers for research and applied innovation (Sections 1, 4.3.5, 5.2, 5.4).
  - Theoretical significance: compute-optimal scaling guidance (how big to train for a given budget), and reliable large-scale training/inference techniques are foundational to the science of building foundation models (Section 3.2.1; Figures 2–4).
- Prior approaches and their gaps
  - Open models (e.g., LLaMA 2, Mistral/Mixtral, Gemma) either use smaller scales or different architectures (e.g., mixture-of-experts) and do not release a frontier-size dense model with long context and comprehensive safety stack. Closed models (e.g., GPT-4 class) show strong performance but are not open for reproducibility (Sections 1, 5; Tables 2, 9–14).
- Positioning
  - Llama 3 focuses on three levers—data, scale, and managing complexity—while keeping the model a standard dense Transformer (no mixture-of-experts), then achieving scale via sophisticated parallelism and an efficient data/finetuning pipeline. The work also experiments with a compositional route to multimodality (vision, video, speech) without altering the core language model (Sections 1–4, 7–8; Figures 1, 28–29).

## 3. Technical Approach
This section follows the training lifecycle: pre-training → post-training/alignment → capabilities → safety → inference → multimodality. Uncommon terms are defined on first use.

- Pre-training data and recipe (Section 3.1, 3.4)
  - Data pipeline (web and domain-specific):
    - Clean and deduplicate: URL/document/line-level dedup; heuristic filters (e.g., repeated n-grams, “dirty words”); model-based quality classifiers trained on Llama-2 annotations for general quality, math/reasoning, and code page detection (Section 3.1.1).
    - Multilingual: fastText for language ID across 176 languages, per-language de-dup, language-specific filters, and multilingual quality ranking using an Llama-2-based classifier (Section 3.1.1).
    - Data mix by “knowledge classification” plus small-scale “scaling-law” pilots to choose proportions; final mix ≈ 50% general knowledge, 25% math/reasoning, 17% code, 8% multilingual (Section 3.1.2).
  - Annealing (Section 3.1.3, 3.4.3): a short end-of-training phase where learning rate linearly decays to zero while upsampling select high-quality data. The team also uses annealing proactively to assess the value of small domain datasets by temporarily upweighting them at the end of training.
  - Training schedule (Section 3.4.1):
    - Optimizer/learning rate schedule: AdamW, cosine decay, warmup; progressively increase batch and sequence length (start at 4,096 tokens then 8,192).
    - Dynamic data mix adjustments mid-training to address observed weaknesses (e.g., increase non‑English data, upsample math data, add fresher web data).
  - Long-context pre-training: expand from 8K to 128K input tokens in 6 staged increments; a document-level attention mask prevents tokens from different documents from attending to each other; validate each stage via “needle-in-a-haystack” retrieval and recovery of short-context tasks (Section 3.4.2).

- Model architecture and compute-optimal sizing (Sections 3.2, 3.2.1)
  - Dense Transformer with targeted tweaks (Table 3): grouped-query attention (`GQA`, reduces key-value cache size), larger tokenizer (128K BPEs including 28K extra non-English tokens), RoPE base frequency increased to 500,000 for length extrapolation.
  - Scaling-law methodology (Figures 2–4; Section 3.2.1):
    - Build “IsoFLOPs” curves: train many small models under fixed compute budgets (6×10^18 to 10^22 FLOPs), fit a parabola of validation loss vs. number of tokens to find the compute-optimal tradeoff at each budget (Figure 2).
    - Fit the relation N*(C)=A·C^α (α≈0.53) to predict optimal tokens for any budget (Figure 3).
    - Predict downstream accuracy by (i) correlating normalized NLL on target tasks with compute, then (ii) correlating NLL with accuracy using LLaMA-2 models to anchor the curve (Figure 4). This drove the choice of ≈405B parameters on ≈15.6T tokens for 3.8×10^25 FLOPs.

- Large-scale training systems and parallelism (Sections 3.3–3.3.4; Figures 5–6; Table 4)
  - Hardware: up to 16K H100 GPUs (80GB HBM3), Meta’s Grand Teton servers, RoCE (RDMA over Converged Ethernet) at 400 Gbps, deep-buffer spines; storage via Tectonic at sustained 2 TB/s (Section 3.3.1).
  - Reliability: >90% effective training time despite 466 interruptions over 54 days (78% hardware-related); extensive instrumentation (NCCL flight recorder) and straggler detection (Section 3.3.4; Table 5).
  - 4D-parallelism (Figure 5): tensor parallelism (TP), context parallelism (CP), pipeline parallelism (PP), fully-sharded data parallelism (FSDP/DP) arranged as [TP, CP, PP, DP] to match network characteristics (Section 3.3.2).
  - New PP schedule (Figure 6): “N-tunable,” interleaved scheduling minimizes bubbles, balances memory (e.g., embeddings on first stage) and compute (loss on last), uses asynchronous point-to-point communications; proactive tensor deallocation; achieved 38–43% BF16 Model FLOPs Utilization (Table 4).
  - CP implementation: all-gather K/V before computing attention on local Q (made feasible by GQA and custom attention masks), trading a small exposed latency for flexibility (Section 3.3.2).

- Post-training/alignment pipeline (Section 4; Figure 7)
  - Reward model (`RM`): trained on human preferences with three-way ranking when available (“edited > chosen > rejected”), using an efficient concatenation trick to score pairs jointly (Section 4.1.2; Table 6).
  - Supervised finetuning (`SFT`): mix of human prompts with rejection-sampled outputs, synthetic data, and small curated sets; rejection sampling uses the RM to pick from K samples (10–30); PagedAttention speeds sampling by >2× (Section 4.1.3, 4.2.2).
  - Direct Preference Optimization (`DPO`): align to preferences; two stability changes—mask special formatting tokens in the DPO loss (prevents repetition/early termination) and add a small NLL term on the chosen response to preserve format likelihood (Section 4.1.4).
  - Iterative rounds: six cycles of RM→SFT→DPO with data upgrades; heavy quality control (topic/difficulty classifiers, RM- and Llama-based quality filters, semantic dedup) (Sections 4.1.6, 4.2.3; Tables 6–7).

- Capability-specific training (Section 4.3)
  - Code: continue pre-training a “code expert” on ~1T mostly code tokens; generate large synthetic datasets with execution feedback (linting, unit tests, iterative self-correction), language translation (e.g., Python→PHP; Figure 8), and backtranslation for documentation/explanations (Figure 9) (Section 4.3.1).
  - Multilingual: continue pre-training a multilingual expert (90% multilingual tokens) to improve non‑English annotation; careful use of translated math data for MGSM only (Section 4.3.2).
  - Reasoning: generate filtered chain-of-thought traces using outcome and step-wise reward models; interleave code execution for verification; “learn from mistakes” correction data (Section 4.3.3).
  - Long context: curate synthetic long-document QA, hierarchical summarization, and repo-level code reasoning; best trade-off occurred with ~0.1% long-context data mixed into SFT; DPO can remain short-context (Section 4.3.4).
  - Tool use: train agents to operate Search, Python, and Wolfram Alpha; message-level human preferences (because tool calls and follow-up reasoning often span multiple messages); bootstrap with synthetic single-step/multi-step/function-calling datasets; teach “zero-shot tools” by finetuning on (function definition, docstring, query, call) tuples (Section 4.3.5; Figures 10–11).
  - Factuality: “knowledge probing” pipeline—sample pretraining snippets → generate questions → score generations using the snippet as reference → add refusal data where the model is consistently confidently wrong to teach “know-what-you-know” behavior (Section 4.3.6).

- Safety (Section 5.4)
  - Pre-training safety: domain filtering for PII/unsafe content; low measured verbatim memorization rates (e.g., 405B: 1.13% for 50-gram; 3.91% for 1000-gram scenarios; Table 24).
  - Safety finetuning: carefully balanced adversarial and borderline prompts; multilingual safety sets; DPO mixes tuned per model size; refusal tone guidelines + tone classifier to improve user experience (Section 5.4.3).
  - System-level safety: released `Llama Guard 3` (8B classifier) for input/output moderation across 13 harm types; can be selectively enabled by harm; quantized int8 version available; additional guards—`Prompt Guard` (jailbreak/prompt-injection detector) and `Code Shield` (insecure code detector) (Sections 5.4.7; Tables 25–28).
  - Evaluations: violation rate (VR) vs false refusal rate (FRR) across languages, long-context, and tool use; Llama 3 + Llama Guard shows large VR reductions with moderate FRR increases (Figures 19–21).

- Inference (Section 6)
  - Pipeline-parallel inference across 16 GPUs (two nodes) with micro-batching improves throughput (Figure 24).
  - FP8 quantization for FFN GEMMs only (not attention) with row-wise dynamic scaling (clipped to 1200) and no quantization in first/last layers; validated with RM-score distribution (Figure 26) and improved throughput-latency curves (Figure 27) (Section 6.2).

- Multimodal (compositional) experiments (Sections 7–8; Figures 28–30)
  - Vision: a ViT-H/14 encoder (augmented to 40 layers, multi-layer features) feeds `cross-attention adapters` inserted every 4 LLM layers (≈100B adapter params in the 405B model). Train on ~6B image–text pairs at up to four 336×336 tiles, then anneal at higher resolution on ~500M examples; video uses a temporal aggregator and video-specific cross-attention layers (Section 7.2–7.4).
  - Multimodal post-training: SFT on academic/human/synthetic sets; preference learning (vision RM, DPO); rejection sampling to synthesize correct rationales; final “quality-tuning” pass on a small high-quality set (Section 7.5).
  - Speech (understanding): a 1B-parameter Conformer encoder + small adapter outputs token-level embeddings inserted directly into the language model; system prompts steer automatic speech recognition (ASR) vs. speech translation (AST) vs. chat; pre-train the encoder on ~15M hours via self-supervised BEST-RQ; finetune jointly with LLM frozen (Sections 8.1–8.3).
  - Speech (generation): streamable text normalization (TN) and prosody modeling (PM) conditioned on Llama-3 embeddings to reduce lookahead and improve naturalness (Section 8.2.2).

## 4. Key Insights and Innovations
- Compute-optimal scaling that predicts downstream accuracy (Section 3.2.1; Figures 2–4)
  - Novelty: moves beyond classic “loss vs. compute” by (i) building IsoFLOPs curves to identify compute-optimal tradeoffs and (ii) tying normalized NLL to task accuracies using previous model families (Figure 4).
  - Significance: de-risks billion-dollar-scale runs and justified selecting a ~405B dense model for a fixed budget—demonstrably close to optimal.
- High-throughput, network-aware 4D parallelism with a new PP schedule (Section 3.3.2; Figures 5–6; Table 4)
  - Novelty: unify TP/CP/PP/FSDP with explicit network topology ordering ([TP, CP, PP, DP]) and a tunable, interleaved pipeline schedule, enabling 8K→128K context scaling and 38–43% BF16 MFU at 8–16K GPUs.
  - Significance: shows dense Transformers can be trained stably at frontier scales without MoE complexity.
- Alignment stack that combines RM→SFT (with rejection sampling)→DPO plus two DPO stabilizers (Section 4.1)
  - Novelty: mask formatting tokens in DPO loss and add a small NLL term on the chosen response; extensive data quality controls (topic/difficulty/semantic dedup) (Sections 4.1.4, 4.2.3).
  - Significance: achieves on-par or better performance than closed models on several coding, math, and tool tasks (Table 2).
- Compositional multimodality via adapters that preserve the LLM (Sections 7–8; Figure 28)
  - Novelty: inject vision and video via cross-attention adapters (no LLM changes), and insert speech by converting audio to token-level embeddings; add video temporal aggregation; post-train with SFT→DPO→quality tuning.
  - Significance: reaches competitive image, video, and speech results without turning the language model into a multimodal monolith (Tables 29–32).
- Safety as a layered system (Section 5.4; Figures 19–21; Tables 25–28)
  - Novelty: combine data-level filtering, adversarial/borderline safety finetuning, a released safety classifier (`Llama Guard 3`) with category control and quantization, a jailbreak detector (`Prompt Guard`), and tooling-aware guards (`Code Shield`).
  - Significance: large reductions in violation rates with controlled false refusals across languages, long-context, and tools (Table 25; Figures 19–20).

## 5. Experimental Analysis
- Evaluation design
  - Pre-trained vs. post-trained: pre-trained models are tested on standard NLU/coding/reasoning sets (Tables 8–14). Post-trained models are tested on general knowledge, instruction following, math, coding, tool use, long context, multilingual, and human evaluations (Table 2, Table 16; Sections 5.1–5.3).
  - Metrics: accuracy for multiple choice/QA, pass@1 for code (HumanEval/MBPP), BLEU (speech translation), WER/CER (ASR), exact match/F1/ROUGE for long-context QA, and human win-rates.
  - Robustness and contamination: prompt-format robustness (labels/order; Figures 13–14), adversarial vs. non-adversarial gap (Figure 15), and contamination estimates using 8-gram overlap and “performance gain” selection (Table 15).
- Main post-trained results (Table 2)
  - General knowledge: `Llama 3 405B` achieves MMLU 87.3 (5-shot) vs GPT‑4 (0125) 85.1; MMLU-Pro 73.3 vs GPT‑4 64.8; IFEval 88.6 vs 84.3.
  - Coding: HumanEval pass@1 89.0 (405B) vs GPT‑4 86.6; MBPP EvalPlus 88.6 vs GPT‑4 83.6; Claude 3.5 Sonnet leads slightly on coding (92.0 HumanEval).
  - Math & reasoning: GSM8K 96.8 (8-shot CoT), MATH 73.8 (0-shot CoT), ARC-Challenge 96.9—405B is competitive or better than GPT‑4/4o; Claude 3.5 leads on GPQA (59.4).
  - Tool use: BFCL 88.5 (close to GPT‑4 88.3; Claude 90.2); Nexus 58.7 (best among open models).
  - Long context: QuALITY 95.2 EM; InfiniteBench En.MC 83.4; Multi-needle ~98 recall—close to GPT-4/4o’s perfect recall (Table 21).
  - Multilingual: MGSM 91.6 (405B) matches Claude 3.5 Sonnet and GPT-4o; on internal multilingual MMLU, 405B = 83.2 vs GPT‑4o 85.5 (Table 20).
- Proficiency exams (Table 17)
  - `Llama 3 405B` is highly competitive: LSAT 81.1, GMAT-Q 96.0, SAT-Math 94.9; many scores near GPT‑4o/Claude 3.5.
- Tool use human eval (Figure 16)
  - On code execution and plots, `Llama 3 405B` beats GPT‑4o; underperforms on file-upload tasks.
- Pre-trained baselines (Tables 9–14; Figure 12)
  - Even before alignment, `Llama 3 70B` and `8B` outperform similar-size open models across many categories; `405B` substantially improves general and math benchmarks (e.g., MMLU 85.2; MATH 53.8) (Tables 12–13).
- Robustness & adversarial gap
  - Label variant/order sensitivity is low, especially for `405B` (Figures 13–14).
  - Adversarial datasets (Adversarial SQuAD, GSM-Plus, PAWS) show gaps vs. non-adversarial; paraphrase detection parity is much improved (Figure 15).
- Contamination (Table 15)
  - Some benchmarks (e.g., HellaSwag, PiQA) show both higher contamination and meaningful performance gain estimates; others (e.g., SQuAD, MATH) show low/no gain despite overlap. MMLU-like sets required different detection strategies (not reliably capturable by 8-grams).
- Multimodal results
  - Vision (Table 29): `Llama 3-V 405B` beats GPT‑4V on all listed sets (e.g., DocVQA 92.6, ChartQA 85.8), and is close to Gemini 1.5 Pro/Claude 3.5.
  - Video (Table 30): `Llama 3-V 70B` reaches 60.8 on PerceptionTest (better than Gemini 1.0 Ultra 54.7), 87.9 on TVQA (≈ GPT‑4V 87.3).
  - Speech (Tables 31–33): `Llama 3 70B` ASR outperforms Whisper v2 / SeamlessM4T v2 on MLS/Libri/VoxPopuli; AST BLEU competitive; toxicity: very low “added toxicity” (e.g., English AT 0.68%).
  - Speech generation (Tables 34–35): using Llama embeddings improves TN accuracy (90.7% vs 88.0% with full right context) and human preference for prosody (≈60–64%).
- Safety outcomes
  - Across languages, `Llama 3 405B + Llama Guard 3` shows substantially lower violation rates with modest FRR increases (e.g., English: VR −86%, FRR +102%; Table 25; Figure 19). Long-context and tools show similar tradeoffs (Figure 20).
  - Prompt-injection susceptibility: lower than Mixtral, higher than GPT‑4 Turbo/Gemini Pro on this benchmark (Figure 22).
  - “Uplift” studies in cyber and chemical/biological risk show no significant increase over internet-only baselines within the tested protocols (Section 5.4.5).
- Inference efficiency
  - Micro-batching increases throughput (Figure 24). FP8 quantization improves prefill throughput up to ~50% while preserving response quality distribution (Figures 26–27).

Overall, the results convincingly support the central claims: competitive or superior benchmark performance for an open model family, strong tool and long-context abilities, and a measurable safety–helpfulness balance.

## 6. Limitations and Trade-offs
- Assumptions and data issues
  - Contamination estimation uses 8-gram overlap; several datasets (e.g., MMLU) are hard to assess with this method (Table 15). Some performance on high-overlap sets may be optimistic.
  - Heavy reliance on synthetic data (code/reasoning/tools) risks “model echoing” biases despite execution feedback and filters (Sections 4.2.2, 4.3.1–4.3.3).
- Scope and scenarios not fully addressed
  - Safety results rely on internal multilingual/long-context/tool benchmarks and cannot be externally reproduced as-is; competitor systems are anonymized (Sections 5.4.4, 5.4.8).
  - File-upload tool tasks lag behind GPT‑4o in human evals (Figure 16), suggesting remaining gaps in complex multi-file, multi-format orchestration.
- Computational constraints
  - Training requires up to 16K H100s with sophisticated RoCE fabrics; operations noted power-grid fluctuations and diurnal thermal effects (Section 3.3.4). This level of infra is not broadly accessible.
  - Long-context support required a separate “continued pre-training” stage (Section 3.4.2) and small long-context SFT fraction (~0.1%) to keep short-context quality (Section 4.3.4).
- Inference trade-offs
  - FP8 quantization is carefully limited (not applied to attention or first/last layers) with dynamic-scale clipping; while reward distributions match, rare edge failures may still surface (Section 6.2).
  - The fastest inference configuration still uses multi-GPU pipeline parallelism (Section 6.1), which complicates deployment compared to small models.
- Open questions
  - Adversarial robustness remains below non-adversarial performance in QA/mathematics (Figure 15). Long-horizon, multi-tool plans and safety under complex tool chains are still challenging (Sections 4.3.5, 5.4.4).

## 7. Implications and Future Directions
- Field impact
  - Open weights at the 405B scale, together with detailed training/inference/safety recipes, change the reproducibility and benchmarking landscape. It enables independent research on scaling laws, long-context behavior, safety alignment, and tool-using agents at frontier capacity (Sections 1, 3, 5.4, 6).
- Follow-up research
  - Scaling-law science: extend the two-stage NLL→accuracy mapping to more task families; study how IsoFLOPs “flat minima” near the optimum (Figure 2) interact with architecture choices.
  - Robustness and safety: public, standardized long-context and tool-use safety suites; stronger defenses against prompt injection and many-shot jailbreaking (Figures 20, 22).
  - Data quality: formalize “annealing for dataset valuation” (Section 3.1.3) as a general method; study how small curated sets shift capabilities late in training.
  - Multimodal composition: iterate on adapter design (e.g., parameter-efficient routing, better temporal aggregation) and long-form multimodal reasoning with minimal LLM changes (Sections 7.2–7.5, 8).
- Practical applications
  - Long-document assistants (legal, finance, scientific literature), code copilots with execution feedback, multilingual tutoring, retrieval-augmented QA with safe tool calls (Search/Python/Wolfram), speech chat and streaming TTS with improved prosody control (Sections 4.3.4–4.3.6, 5.2, 8).
  - System-level deployment with `Llama Guard 3` and `Prompt Guard`/`Code Shield` provides a concrete blueprint for configurable safety in production (Section 5.4.7).

> In short, the paper delivers a complete, open, frontier-scale stack: compute-optimal dense models; a robust data/finetuning pipeline; tooling for long context and agents; a measurable safety framework; and practical inference optimizations. The combination of strong results (Table 2), transparent methodology (Sections 3–4), and released safety/inference components makes it a significant step for both research and deployment.
