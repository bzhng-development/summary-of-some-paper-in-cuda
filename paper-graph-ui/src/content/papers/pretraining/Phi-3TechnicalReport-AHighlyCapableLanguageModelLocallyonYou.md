# Phi-3 Technical Report: A Highly Capable Language Model Locally on Your Phone

**ArXiv:** [2404.14219](https://arxiv.org/abs/2404.14219)

## 🎯 Pitch

This paper introduces the Phi-3 family of small language models—including phi-3-mini with just 3.8 billion parameters—that achieve performance rivaling models nearly 10 times larger, all while running efficiently and privately on a modern smartphone. By pioneering a data-centric training approach and targeted alignment, Phi-3 marks a leap in practical AI: it makes powerful, safe, and low-latency language capabilities available to everyone—without the need for internet connectivity or massive infrastructure.

---

## 1. Executive Summary (2-3 sentences)
This report introduces the `Phi-3` family of small language models (SLMs)—notably `phi-3-mini` (3.8B parameters)—that approach the capability of much larger models while running locally on a phone. The core advance is a data-first training strategy (“data‑optimal regime”) plus targeted post‑training that, together with several efficiency techniques, delivers competitive reasoning, coding, and safety performance at a fraction of the size and memory footprint (e.g., 4‑bit `phi-3-mini` runs at >12 tokens/s on an iPhone 14; Figure 2).

## 2. Context and Motivation
- Problem addressed
  - High-performing LLMs typically require hundreds of billions of parameters and vast compute, which precludes private, low-latency, on-device use and makes training/deployment costly. The paper targets a long-standing tension: can a small model be both fast and broadly capable?
- Why it matters
  - Real-world: On-device models reduce latency, preserve privacy, and enable offline use (e.g., medical, enterprise, travel). Cost-effective models widen access and make edge deployments feasible.
  - Theoretical: Classic scaling laws assume fixed data quality; the paper investigates how better data curation plus synthetic data can shift the size–performance trade-off.
- Prior approaches and gaps
  - Traditional “bigger is better” (compute-optimal) scaling [KMH+20, HBM+22] improves quality but scales cost and latency. Earlier small models often lag in reasoning and safety.
  - `phi-2` showed small models can punch above their size using curated and synthetic “textbook-like” data. However, it did not demonstrate strong on-device runtime or breadth across multilingual, long-context, and multimodal tasks.
- Positioning
  - This work expands the “Textbooks Are All You Need” recipe with a larger, more refined dataset and introduces architectural and systems-level optimizations. It also extends the series to `phi-3.5` for multilingual/long-context and `phi-3.5-Vision` for multimodal, benchmarking them against open and commercial models.

## 3. Technical Approach
The paper’s advances combine a data-centric training regime, efficient architectures, and post-training alignment.

- Model family and sizes
  - `phi-3-mini` (3.8B): Decoder-only transformer with 32 layers/heads, hidden size 3072; context 4K by default; 128K with LongRope; trained 3.3T tokens in bfloat16; chat-finetuned with a simple prompt template. Section “Technical Specifications.”
  - `phi-3-small` (7B) and `phi-3-medium` (14B): Same family, trained 4.8T tokens. `phi-3-small` incorporates several efficiency changes (below). Section “Technical Specifications.”
  - `phi-3.5-mini` and `phi-3.5-MoE` (language), and `phi-3.5-Vision` (multimodal): Add multilingual, long-context, and vision capabilities (Sections 4 and 7).

- Data pipeline and the “data‑optimal regime”
  - Two-phase pretraining (Section “Training Methodology”):
    - Phase 1: Heavily filtered public web data to impart general knowledge/language.
    - Phase 2: Mix of even more filtered web data (subset of Phase 1) and synthetic LLM-generated data to teach reasoning and niche skills.
  - Data-optimal regime (Section “Data Optimal Regime”):
    - Rather than merely scaling compute or epochs, the pipeline curates data to fit a model’s capacity—keeping material with educational/reasoning value while pruning trivia that would waste capacity. The paper illustrates the idea by removing ephemeral facts (e.g., sports results) so a small model can allocate capacity to reasoning.
    - Evidence that this regime changes scaling behavior appears in Figure 3, which contrasts `Phi` scaling vs. Llama-2 trained on fixed data.

- Post-training to become an assistant (Section “Post-training”)
  - SFT (Supervised Fine-Tuning): Curated, high-quality instruction data across math, coding, reasoning, safety, and model identity. Starts with English-only examples.
  - DPO (Direct Preference Optimization): Preference-based tuning on chat, reasoning, and safety data; “rejected” responses steer the model away from undesired behaviors.

- Efficiency choices (especially for `phi-3-small`)
  - `GEGLU` activation: A gated linear unit variant that tends to stabilize training.
  - `muP` (Maximal Update Parameterization): A method to tune hyperparameters on a proxy model and transfer to a larger target for stable scaling.
  - `Grouped-Query Attention (GQA)`: Multiple query heads share one key/value (K/V) set to reduce memory bandwidth during decoding.
  - `Blocksparse attention` (Figure 1): Each attention head attends to different blocks of the past context (local and “vertical/remote” blocks). This “divide-and-conquer” across heads reduces the amount of K/V state the model must store and fetch.
    - Definition: `KV cache` is the stored keys/values from prior tokens used during autoregressive decoding; reducing it lowers memory and latency.
    - Implementation: Custom high-efficiency kernels for training (Triton, based on FlashAttention) and inference (custom prefill kernel; extended paged attention in vLLM).
    - Alternating dense and blocksparse layers balances recall and memory savings.
  - Quantization: `phi-3-mini` can be quantized to 4-bit weights to fit ≈1.8GB RAM and still run >12 tokens/s on iPhone 14 A16, fully offline (Figure 2).
    - Definition: `Quantization` stores parameters at lower precision to reduce memory and speed up inference with minimal quality loss.
  - Long context via `LongRope` (Section 4): A technique to extend positional encoding (rope scaling) to 128K context without retraining from scratch.

- Mixture-of-Experts variant (`phi-3.5-MoE`)
  - Architecture (Section 2): 16 experts, top-2 routing per token; each expert is a GLU feed-forward network; 6.6B “active” parameters per token out of 42B total.
    - Definition: `MoE` activates a subset of expert sub-networks per token, increasing capacity without proportionally increasing compute per token.
  - Training uses `SparseMixer` to stabilize the sparse router.

- Multimodal model (`phi-3.5-Vision`, Section 7)
  - Components: CLIP ViT-L/14 image encoder + `phi-3.5-mini` text decoder.
  - Token interleaving: Visual tokens (from CLIP) are interleaved with text tokens; no special ordering required.
  - Dynamic cropping: Splits high-res images into blocks and concatenates their tokens to cover diverse aspect ratios and maintain detail (up to 1344×1344 during pretraining).
  - Training:
    - Pretraining: ~0.5T tokens across interleaved image-text documents (e.g., OBELICS), FLD-5B pairs, OCR-synthesized data, chart/table datasets, and text-only; apply next-token prediction on text tokens (no image-token loss).
    - Post-training: Multimodal SFT (~33B tokens) across natural images, charts, diagrams, presentations, videos, multi-image reasoning, plus safety; then DPO with text and smaller-scale multimodal prefs.

## 4. Key Insights and Innovations
- Data-first scaling for small models (“data‑optimal regime”) is powerful
  - What’s new: Rather than scaling model size, the team scales and curates data to match a small model’s capacity, blending filtered “educational” web data with synthetic reasoning data in two phases.
  - Why it matters: Figure 3 shows `Phi` models deviate favorably from Llama-2 scaling trained on fixed data; `phi-3-mini` reaches 68.8% MMLU (Table in Section 3) and MT-Bench 8.38 while being only 3.8B parameters.
- Practical on-device LLM at useful quality
  - What’s new: 4-bit `phi-3-mini` occupies ~1.8GB and runs >12 tokens/s fully offline on an iPhone 14 (Figure 2).
  - Why it matters: This demonstrates a step-change in deployability—capabilities close to GPT‑3.5-level benchmarks without a datacenter.
- Memory-efficient attention at scale (blocksparse + kernels)
  - What’s new: Per-head block patterns that collectively cover the full context, plus custom kernels for training and decoding (Figure 1).
  - Why it matters: Reduces KV cache and improves speed without giving up long-context retrieval; enables practical deployment with constrained memory.
- Compact MoE with high active capacity (`phi-3.5-MoE`)
  - What’s new: A 16×3.8B MoE (6.6B active) using top-2 routing and SparseMixer for stable training.
  - Why it matters: Table 3 shows it outperforms similarly sized open models (e.g., Llama‑3.1‑8B, Mixtral series) and approaches Gemini‑1.5‑Flash and GPT‑4o‑mini across many language benchmarks.
- Multimodal `phi-3.5‑Vision` tuned for real-user prompting
  - What’s new: High-resolution dynamic cropping, interleaved tokens, and a large SFT+DPO recipe tuned under simple, 0‑shot prompts (Section 7.2 setup).
  - Why it matters: Table 5 shows competitive or better results versus open multimodal baselines on diverse tasks (e.g., 91.3 on ScienceQA, 81.9 on MMBench), while being only 4.2B parameters.

## 5. Experimental Analysis
- Evaluation setup (Section 3 and Section 7.2)
  - Language benchmarks: MMLU, HellaSwag, ANLI, GSM‑8K (with Chain-of-Thought, CoT), MATH (CoT), MedQA, AGIEval, TriviaQA, ARC‑C/E, PIQA, Social IQA, BigBench-Hard (CoT), WinoGrande, OpenBookQA, BoolQ, CommonSenseQA, TruthfulQA, HumanEval, MBPP. Few-shot prompts; temperature 0; same pipeline for comparability.
  - Long-context: RULER and RepoQA (Section 4; Tables 1–2).
  - Multilingual: MMLU‑multilingual and MGSM (Figure 4; Table 3).
  - Safety: Internal multi-turn RAI benchmark (Table 4) and red-teaming (Figure 5).
  - Multimodal: MMMU, ScienceQA, MathVista, Inter-GPS, MMBench, POPE, AI2D, ChartQA, TextVQA (Table 5), plus multi-image/video BLINK and VideoMME (Table 6).
  - Note on fairness: The report emphasizes consistent prompts and that numbers may differ from other publications due to evaluation choices; e.g., “we did no optimization to the pipeline for the phi‑3 models” and even omitted known prompt tweaks that help (footnote in Section 3).

- Main quantitative results (selected)
  - Capability of small `phi-3` models (Section 3 table):
    - `phi-3-mini` (3.8B):
      - MMLU 68.8 vs GPT‑3.5 71.4, Mixtral‑8×7B 70.5, Llama‑3‑8B‑Instruct 66.5.
      - GSM‑8K (8‑shot CoT): 82.5, higher than GPT‑3.5 78.1 and Mixtral‑8×7B 64.7.
      - MATH (0‑shot CoT): 41.3 vs GPT‑3.5 45.3 (close for a much smaller model).
      - Average across tasks: 69.7 vs GPT‑3.5 72.8 and Mixtral‑8×7B 66.8.
    - Scaling within family:
      - `phi-3-small` (7B): MMLU 75.7; GSM‑8K 89.6; BigBench‑Hard (CoT) 79.1.
      - `phi-3-medium` (14B): MMLU 78.0; GSM‑8K 91.0; MATH 53.1.
      - These jumps support the “data-optimal regime” up to 7B; the paper notes smaller gains from 7B to 14B (Section “Data Optimal Regime”).
  - On-device feasibility
    - > “phi‑3‑mini can be quantized to 4‑bits so that it only occupies ≈ 1.8GB of memory… on iPhone 14… fully offline achieving more than 12 tokens per second.” (Figure 2 caption and preceding paragraph)
  - Long-context and multilingual (Section 4)
    - RepoQA (Table 1, code long-context QA): `phi-3.5-MoE` avg 85 vs Llama‑3.1‑8B 71 and Mixtral‑8×7B 68; GPT‑4o (May‑2024) 90.6.
    - RULER (Table 2): `phi-3.5-MoE` avg 87.1 vs Llama‑3.1‑8B 88.3. Performance drops sharply at 128K (64.2), attributed to limited high-quality long-context mid-training data.
    - Multilingual MMLU (Figure 4): `phi-3.5-mini` improves average from 47.3 (`phi‑3‑mini`) to 55.4; `phi-3.5-MoE` reaches 69.9.
  - Aggregate comparison vs strong baselines (Table 3)
    - Average across representative language benchmarks: `phi-3.5-MoE` 69.2, `phi-3.5-mini` 61.1, Gemini‑1.5 Flash 68.5, GPT‑4o‑mini 74.9.
    - Selected tasks:
      - BigBench‑Hard (0‑shot CoT): `phi-3.5-MoE` 79.1 vs GPT‑4o‑mini 80.4.
      - GSM‑8K (8‑shot CoT): `phi-3.5-MoE` 88.7 vs GPT‑4o‑mini 91.3.
      - HumanEval (code): `phi-3.5-MoE` 70.7 vs GPT‑4o‑mini 86.6; Gemini‑1.5 Flash 74.4.
  - Safety and robustness
    - Red-team results (Figure 5) show large reductions in harmful responses after safety alignment.
    - Internal RAI benchmark (Table 4): Lower is better. `phi-3.5-MoE` has ungroundedness 0.228 (better than Mistral‑7B’s 0.935 and Gemma‑7B’s 0.679), and competitive defect rates across categories.
  - Multimodal (`phi-3.5-Vision`, Table 5; Section 7.2)
    - ScienceQA: 91.3 (beats Gemini 1.0 Pro V 79.7; close to GPT‑4O 88.5).
    - MMBench (dev‑en): 81.9 (competitive with open baselines; below GPT‑4O 88.4).
    - ChartQA: 81.8 (substantially higher than many open baselines; GPT‑4O at 64.0 under this evaluation setup).
    - TextVQA: 72.0 (close to GPT‑4O 75.6 under identical pipeline).
  - Multi-image/video (Table 6)
    - BLINK: 57.0 (competitive with GPT‑4o‑mini 51.9; below GPT‑4O 63.2).
    - VideoMME: 50.8 (below Gemini‑1.5 Flash 62.3; the paper uses a uniform 16‑frame protocol for fairness across models).

- Robustness and qualitative checks
  - Search augmentation can fix factual gaps: Figure 6 shows a failure without search and a correct, more specific answer with search enabled.
  - The paper notes prompt sensitivity (e.g., adding “##” before questions improves scores) but doesn’t exploit such tweaks in reported numbers (footnote in Section 3).

- Overall assessment
  - The experimental suite is broad and carefully standardized. Results substantiate the central claims: (1) small models can reach strong general and reasoning performance with data‑optimal training; (2) the approach scales to multilingual, long-context, and multimodal settings; and (3) safety alignment substantially improves red-team outcomes. Some areas (extreme long context, code on HumanEval vs GPT‑4o‑mini, certain video tasks) still lag SOTA, clarifying the boundaries of the current recipe.

## 6. Limitations and Trade-offs
- Capacity constraints in factual knowledge (Section 6 “Weakness”)
  - Small models have limited memory for world facts; this surfaces in lower TriviaQA scores and other knowledge-heavy tasks. The report suggests search/RAG augmentation as a remedy (Figure 6).
- English-centric training at the `mini` scale
  - `phi-3-mini` is primarily English. Multilingual strength appears only after mid-training for `phi-3.5` series; `phi-3.5-mini` improves but still trails `phi-3.5-MoE` (Figure 4, Table 3).
- Long-context fragility at maximum window
  - RULER performance drops sharply at 128K (Table 2), attributed to insufficient high-quality long-context mid-training data. This indicates window extension via LongRope alone is not enough; high-quality long-context supervision is needed.
- Diminishing returns at larger sizes with the same data mixture
  - The jump from 3.8B → 7B brings large gains; 7B → 14B brings smaller gains (Section “Data Optimal Regime”), hinting the data mixture may be suboptimal for 14B scaling.
- Systems and portability trade-offs
  - Blocksparse attention depends on custom kernels and a particular sparsity pattern per head (Figure 1), which complicates portability across runtimes and hardware.
- MoE complexity
  - While `phi-3.5-MoE` delivers strong accuracy, MoE introduces routing complexity and potential load-balancing challenges. Training stability requires mechanisms like SparseMixer; inference may require specialized infrastructure for efficiency.
- Safety and hallucinations are mitigated, not solved
  - Despite improved RAI metrics and red-teaming (Figure 5, Table 4), the paper notes remaining issues: factual inaccuracies, bias reproduction, and occasional unsafe responses (Sections 5 and 7.4).
- Reproducibility of the “data-optimal” recipe
  - The precise filtering and synthetic data generation pipeline is not fully public. Reproducing the exact mixture and quality may be challenging for third parties.

## 7. Implications and Future Directions
- Field impact
  - This work shifts the default assumption from “capability requires scale” to “capability can be bought with better data and alignment,” at least up to mid-tier performance. It makes on-device assistants plausible and useful.
- Practical applications
  - Private mobile assistants; offline enterprise copilots; edge analytics in healthcare/finance where data egress is restricted; field operations (e.g., industrial inspection) needing multimodal understanding without connectivity.
  - Multilingual and long‑context variants broaden applicability to global use and large-document/code understanding (Tables 1–2).
- Research avenues
  - Data mixture optimization: Formalizing the “data‑optimal regime” (selection, scheduling, synthetic generation criteria) and making it adaptive to model size.
  - Long-context learning: Curate high-quality 128K+ training data; study retrieval and memory mechanisms that complement LongRope.
  - Efficient attention: Generalize blocksparse patterns, auto-learn sparsity, and standardize kernels for broader hardware support.
  - MoE for small models: Explore latency-aware expert routing and dynamic capacity control for mobile/edge inference.
  - Safety and grounding: Combine DPO with tool-use (retrieval, calculators, verifiers) and multimodal red-teaming; reduce hallucinations especially in finance and safety-critical domains (Section 7.4).
  - Multimodal scaling: Better video understanding through frame selection policies and temporal modeling beyond uniform 16-frame sampling (Table 6).

> “Highly capable language model running locally on a cell-phone… quantized to 4-bits so that it only occupies ≈ 1.8GB… achieving more than 12 tokens per second.” (Section 2; Figure 2)

> “Scaling law close to the ‘Data Optimal Regime’… phi‑1.5 → phi‑2 → phi‑3‑mini → phi‑3‑small vs Llama‑2 family trained on the same fixed data.” (Figure 3)

> “`phi-3.5-MoE`… achieves superior performance… on par with Gemini-1.5-Flash and GPT-4o-mini” in many language tasks (Table 3), while `phi-3.5-Vision` is competitive across single- and multi-image benchmarks (Tables 5–6).
