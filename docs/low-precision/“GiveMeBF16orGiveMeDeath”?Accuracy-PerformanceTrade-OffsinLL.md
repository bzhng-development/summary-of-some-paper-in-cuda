# “Give Me BF16 or Give Me Death”? Accuracy-Performance Trade-Offs in LLM Quantization

**ArXiv:** [2411.02355](https://arxiv.org/abs/2411.02355)

## 🎯 Pitch

This paper offers the most comprehensive empirical study to date on the quantization of large language models (LLMs), systematically evaluating FP8, INT8, and INT4 formats across a range of academic and real-world benchmarks for the Llama-3.1 family. By clarifying that FP8 is virtually lossless and well-tuned INT8/INT4 can match or rival full-precision baselines with only negligible accuracy loss—even at massive 405B scale—the work delivers actionable, data-driven recommendations for cost-efficient, high-performance LLM deployment across modern GPUs. These insights empower practitioners to confidently select quantization schemes that deliver scalable inference speeds without sacrificing output quality, addressing a critical barrier for real-world LLM serving at scale.

---

## 1. Executive Summary
This paper delivers a large-scale, end-to-end study of quantizing large language models (LLMs), comparing three widely deployable formats—`W8A8-FP` (FP8), `W8A8-INT` (INT8), and `W4A16-INT` (INT4 weights, FP16 activations)—across accuracy, speed, and cost. It finds FP8 to be essentially lossless, well-tuned INT8 to incur only minor accuracy drops, and INT4 weight-only to be more competitive than expected, and then translates these findings into concrete deployment guidance for synchronous vs. asynchronous serving on A6000, A100, and H100 GPUs (Sections 3–6; Tables 2–6; Figures 1–4).

## 2. Context and Motivation
- Problem addressed
  - Practitioners lack clear, data-driven guidance on the accuracy vs. performance trade-offs of commonly supported quantization formats when serving LLMs at scale. Prior reports gave mixed or pessimistic views of INT8 activation quantization, and many studies benchmarked only on academic tasks or tuned hyperparameters suboptimally (Sections 1–2.2).
- Why it matters
  - Serving LLMs is expensive and latency-sensitive. Quantization (reducing number precision) can cut memory, bandwidth, and compute cost—but if accuracy degrades, downstream applications suffer. Organizations need reliable recipes that balance speed, cost, and quality (Abstract; Section 5).
- Shortcomings of prior approaches
  - Narrow evaluations (mostly academic; limited real-world, open-ended tasks).
  - Missing strong full-precision baselines at very large scale (e.g., 405B).
  - Under-tuned quantization (e.g., poor calibration, suboptimal clipping) led to overstated accuracy loss (Section 2.2).
- How this paper positions itself
  - Comprehensive scope: the entire Llama‑3.1 Instruct family (8B, 70B, 405B), plus reasoning-focused DeepSeek-R1-Distill variants (Table 4), evaluated on Open LLM Leaderboards V1/V2, Arena-Hard-Auto, HumanEval/HumanEval+, RULER, and a text-similarity suite (Sections 3–4).
  - Deployment-grounded performance: vLLM 0.6.4.post1 on A6000/A100/H100, with synchronous and asynchronous scenarios that mirror real services (Sections 5.1–5.2; Tables 5–6; Figures 2–3).

## 3. Technical Approach
Key terms (defined only when uncommon or paper-specific):
- `Quantization`: representing numbers with fewer bits. Here, applied to model weights and/or activations.
- Format notation:
  - `W8A8-FP (FP8)`: 8-bit floating-point weights and activations.
  - `W8A8-INT (INT8)`: 8-bit integer weights and activations.
  - `W4A16-INT (INT4)`: 4-bit integer weights, 16-bit activations.
- `Per-output-channel` quantization: each output channel of a linear layer has its own scale (and, for symmetric schemes, zero-point 0).
- `Dynamic per-token activation quantization`: compute activation scales on the fly for each token, adapting to distribution changes at inference.
- `SmoothQuant`: an offline rescaling that shifts magnitude from activations into weights (using calibration data) to tame activation outliers, making activations easier to quantize.
- `GPTQ`: a post-training weight-quantization algorithm that uses second-order information (approximated Hessian statistics) to minimize the quantization error over calibration data.
- `AWQ`: a weight-only method that emphasizes preserving “activation-aware” important weights.

Step-by-step methodology (Sections 3.1–3.2, 5):
1. Models and formats
   - Models: Llama‑3.1‑8B/70B/405B Instruct for core study; DeepSeek-R1-Distill (Llama and Qwen families) to probe reasoning sensitivity (Table 4).
   - Formats: `W8A8-FP`, `W8A8-INT`, `W4A16-INT`, chosen for broad support in vLLM.
2. How each format is implemented
   - `W8A8-FP` (Section 3.2):
     - Weights: symmetric, per-output-channel, round-to-nearest (no calibration).
     - Activations: dynamic, per-token FP8 quantization (no calibration).
     - Rationale: Hopper/Ada support FP8 kernels; dynamic activations avoid fixed clipping errors.
   - `W8A8-INT` (Section 3.2):
     - Weights: GPTQ, symmetric per-output-channel INT8 (uses calibration).
     - Activations: dynamic per-token INT8. For `70B`, add SmoothQuant (calibration required) because that size shows more activation outliers.
     - Calibration: random tokens are sufficient for `8B`; larger models use higher-quality data (Platypus-style dataset; Lee et al., 2023).
   - `W4A16-INT` (Section 3.2):
     - Weights: GPTQ with MSE-optimal clipping, group size 128 (one scale per group of 128 consecutive weights).
     - Activations: left at FP16 (no activation quantization).
     - Calibration: random tokens hurt; use OpenPlatypus for better ranges.
3. Why these design choices?
   - INT4 is practical only as weight-only today (activation INT4 still degrades accuracy and lacks robust kernels).
   - FP8 activations are easier than INT8 because FP8 can represent a wider dynamic range; dynamic per-token scaling keeps activations in-range without precomputation.
   - For INT8 activations, SmoothQuant equalizes ranges across channels to combat outliers.
   - GPTQ with MSE-optimal clipping improves low-bit accuracy over absmax and, in this study, outperforms AWQ on real-world tasks (Table 1; Appendix Tables 12–13).
4. Evaluation pipeline (Sections 3–5)
   - Accuracy:
     - Academic: Open LLM Leaderboards V1 and V2; V2 uses normalized scores to equalize task difficulty (Tables 2–3).
     - Real-world/open-ended: Arena‑Hard‑Auto‑v0.1 (two runs, CI in Table 7), HumanEval/HumanEval+ pass@1, long-context RULER (Table 3, Appendix A.1).
     - Text similarity: ROUGE-1/L, BERTScore, and STS, using identical prompts and greedy decoding vs. BF16 outputs (Figure 1).
     - Reasoning: AIME’24, MATH‑500, GPQA‑Diamond with 20 samples per query to reduce variance (Table 4).
   - Performance:
     - vLLM 0.6.4.post1; A6000, A100, H100; synchronous (single-query, latency-focused) and asynchronous (multi-query, throughput-focused) modes.
     - Seven representative use cases: code completion, instruction following, multi-turn chat, RAG, summarization, docstring generation, code fixing, with specified prefill/decode token lengths (Section 5; Tables 5–6).
     - Cost efficiency computed with Lambda Labs on-demand pricing (Appendix Table 9).

Intuition for how quantization yields speedups (Section 5):
- Decode (generating tokens) is often memory-bandwidth bound; smaller weights (INT4/8) reduce memory traffic → faster decode.
- Prefill (processing the prompt) is often compute-bound; lower-precision activations (FP8/INT8) enable faster matmuls → faster prefill.
- Therefore, weight-only INT4 shines in decode-heavy, latency-sensitive synchronous serving; full W8A8 (FP or INT) shines in prefill-heavy or high-throughput asynchronous serving.

## 4. Key Insights and Innovations
1. FP8 is effectively lossless across scales and tasks when implemented with dynamic per-token activations and symmetric per-channel weights (Sections 3.2, 4; Tables 2–3).
   - Why it’s significant: it dispels concerns that FP8 harms quality and validates Hopper/Ada FP8 kernels for production. For example, on Llama‑3.1‑405B (Leaderboard V1):
     > Table 2 (405B, average): BF16 86.79 vs. W8A8-FP 86.89.
   - On the harder V2 suite at 405B:
     > Table 3 (V2 average): BF16 48.7 vs. W8A8-FP 48.7.

2. With proper tuning, INT8 achieves only small accuracy losses (Sections 3.2, 4; Tables 2–3).
   - What changed from prior reports: careful activation handling (dynamic per-token and SmoothQuant on 70B) and better calibration reduces the reported 10+ point drops to ~1–3% per task on average.
   - Evidence: Llama‑3.1‑405B (V2 average):
     > Table 3: W8A8-INT 47.9 vs. BF16 48.7 (−0.8 absolute, ~1.6% relative).

3. INT4 weight-only (with GPTQ and MSE-optimal clipping) is more competitive than expected and can rival INT8 (Sections 3.2, 4; Table 1).
   - Novelty: across open-ended tasks, a simple GPTQ variant surpassed AWQ, contradicting earlier claims favoring AWQ.
   - Evidence (real-world averages; Table 1):
     > Llama‑3.1‑8B: GPTQ 52.3 vs. AWQ 49.4; Llama‑3.1‑70B: GPTQ 73.1 vs. AWQ 72.3.

4. Deployment guidance based on serving mode (Section 5; Tables 5–6; Figures 2–3).
   - Synchronous (latency-first): `W4A16-INT` delivers the best latency and cost-per-query.
     > Table 5 (405B on A100, code completion): BF16 81.9s vs. INT4 48.9s latency; Cost Reduction 6.38×.
   - Asynchronous (throughput-first): `W8A8` (FP or INT) dominates peak throughput.
     > Table 6 (405B on 16×H100, instruction following): BF16 8.5 QPS vs. FP8 20.7 and INT4 24.7 QPS; FP8 and INT4 both >2.4× throughput.

5. Output similarity scales with model size (Figure 1).
   - Larger quantized models produce text much closer to BF16 outputs (high ROUGE/BERTScore/STS), suggesting quantization mainly perturbs token choices mildly without changing semantics.

## 5. Experimental Analysis
- Benchmarks and metrics (Sections 3.1, 4–5)
  - Academic: Open LLM Leaderboards V1/V2; V2 normalizes scores so that tasks with different base difficulties are comparable (Tables 2–3).
  - Real-world/open-ended: Arena‑Hard‑Auto‑v0.1 (win-rate style), HumanEval/HumanEval+ pass@1, RULER long-context (Table 3; Table 7 for CI).
  - Text similarity: ROUGE‑1/L, BERTScore, STS against BF16 references (Figure 1).
  - Reasoning: AIME’24, MATH‑500, GPQA‑Diamond with pass@1 from 20 samples (Table 4).
- Main accuracy results
  - Leaderboard V1 (easier; Table 2):
    > “All quantization schemes recover ~99% of BF16.” Example (8B averages): BF16 74.06; W8A8-FP 73.55; W8A8-INT 74.29; W4A16-INT 73.11.
  - Leaderboard V2 (harder; Table 3):
    > 405B averages: BF16 48.7; W8A8-FP 48.7; W8A8-INT 47.9; W4A16-INT 48.2.
    - At 70B, INT formats dip more, reinforcing that INT activation quantization is hardest at this scale (Figure 4).
  - Real-world/open-ended (Table 3 and Table 7):
    - Arena‑Hard‑Auto (405B): 
      > BF16 67.4; W8A8-FP 66.9; W8A8-INT 64.6; W4A16-INT 66.5. Two-run averages with 95% CI provided in Table 7.
    - HumanEval pass@1 (70B):
      > BF16 79.7; W8A8-FP 80.0; W8A8-INT 78.7; W4A16-INT 80.5.
    - RULER (long-context; 70B):
      > BF16 83.3; W8A8-FP 83.0; W8A8-INT 82.5; W4A16-INT 82.2.
  - Text similarity (Figure 1):
    > 405B: ROUGE‑1 ≈0.75, BERTScore ≈0.95, STS ≈0.97 for FP8/INT8/INT4 relative to BF16; 8B shows lower lexical overlap but maintains high semantic similarity (STS ≈0.94–0.96).
  - Reasoning (Table 4):
    > Llama‑70B DeepSeek‑R1‑Distill average: BF16 76.2 vs. W8A8-FP 76.5; W8A8-INT 76.0; W4A16-INT 75.0 (all within ~1–2 points).
- GPTQ vs. AWQ (INT4) (Table 1; Appendix Tables 12–13)
  - Academic benchmarks: small gaps, sometimes favoring AWQ by <0.5 points.
  - Real-world: GPTQ consistently higher, especially on coding (e.g., HumanEval MBPP; Table 1 right).
  - Important confounders controlled: MSE-optimal clipping for GPTQ (vs. absmax in the AWQ comparison), better calibration data, and inclusion of open-ended tasks.
- Performance and cost (Section 5; Tables 5–6; Figures 2–3)
  - Synchronous (Table 5):
    - `W4A16-INT` improves latency and cost-per-query the most. Example:
      > 8B on A6000, instruction following latency: BF16 3.1s vs. INT4 1.3s; Q/$ rises from 1,445 to 3,543.
      > 405B on A100, summarization latency: BF16 44.1s vs. INT4 29.4s; Cost Reduction 6.38× overall.
    - Notable deployment impact: 405B can meet targets on 4×A100 with INT4 where BF16 needed 16 GPUs (Section 5.1).
  - Asynchronous (Table 6):
    - `W8A8` formats deliver the highest peak throughput, especially on H100 with FP8 kernels.
      > 405B on 16×H100, multi-turn chat: BF16 5.3 QPS vs. FP8 10.4 QPS and INT4 11.6 QPS.
    - Latency–throughput trade-offs (Figures 2–3): INT4 wins at low latency; W8A8 wins at very high throughput.
- Do the experiments support the claims?
  - Yes: accuracy, similarity, and performance are measured across model sizes, formats, tasks, and hardware. Ablations on INT4 algorithms (GPTQ vs. AWQ) and calibration/clipping choices strengthen causal explanations (Table 1; Appendix A.2).

## 6. Limitations and Trade-offs
- Scope limitations acknowledged (Limitations section)
  - Components not quantified: KV-cache, input embeddings, and LM head quantization—these can materially affect memory and speed but were left for future work.
  - Language coverage: mostly English; multilingual effects untested.
  - Extremely long-context at 405B on RULER is omitted due to cost (Table 3 notes).
- Method-level constraints
  - `W8A8-INT` requires careful calibration and SmoothQuant at 70B; without tuning, accuracy can drop more (Sections 3.2, 4; Figure 4).
  - `W4A16-INT` relies on high-quality calibration data and MSE-optimal clipping; random-token calibration harms 4-bit results (Section 3.2).
  - Kernel availability matters: FP8 kernels are strongest on Hopper/Ada; INT8/INT4 performance depends on vLLM and GPU architecture (Sections 3.2, 5).
- Trade-offs to keep in mind
  - Activation INT quantization remains more brittle than FP activation quantization (Figure 4 trend; Section 4.1), especially at intermediate scales (70B).
  - Asynchronous throughput gains may increase per-request latency; synchronous low-latency serving sacrifices peak throughput (Figures 2–3).
  - Results center on Llama‑3.1 Instruct variants; other architectures or MoE models may behave differently.

## 7. Implications and Future Directions
- What changes after this work
  - Practitioners get actionable defaults:
    - Latency-first or small-batch online serving: choose `W4A16-INT` (Table 5; Figures 2–3).
    - Throughput-first or heavy batching: choose `W8A8` (FP8 on H100; INT8 on A100/Ampere) (Table 6).
    - If you have Hopper/Ada, FP8 is a near-zero-risk accuracy setting (Tables 2–3).
  - INT8 activation quantization, often viewed skeptically, is viable with dynamic per-token quantization and SmoothQuant at larger scales (Sections 3.2–4).
- Follow-up research enabled
  - Full-stack compression: extend these recipes to KV-cache, embeddings, LM head, and possibly attention-specific paths (Limitations).
  - Broader architectures and tasks: multilingual, domain-specific (e.g., biomed, law), and agentic workloads.
  - More aggressive mixed-precision: e.g., W4A8 with robust activation handling; selective per-layer formats guided by calibration sensitivity.
  - Scheduler–format co-design: adapt serving strategies dynamically (switch between INT4 and W8A8) based on observed prefill/decode ratios and SLA constraints.
- Practical applications
  - Cost-optimized deployment of chat assistants, code assistants, RAG systems, and batch summarization at varying scales (Sections 5.1–5.2; Tables 5–6).
  - Enabling very large models (e.g., 405B) to run with fewer GPUs by combining INT4 weights with careful calibration (Table 5, Section 5.1).

Key supporting quotes (with locations):
- Accuracy—V1, 405B:
  > Table 2: BF16 86.79; W8A8-FP 86.89; W8A8-INT 86.20; W4A16-INT 86.78.
- Accuracy—V2, 405B:
  > Table 3: BF16 48.7; W8A8-FP 48.7; W8A8-INT 47.9; W4A16-INT 48.2.
- Real-world—Arena-Hard, 405B:
  > Table 3: BF16 67.4; W8A8-FP 66.9; W8A8-INT 64.6; W4A16-INT 66.5.
- Performance—Synchronous, 405B on A100:
  > Table 5 (Code Completion): BF16 81.9s vs. INT4 48.9s; Cost Reduction 6.38×.
- Throughput—Asynchronous, 405B on 16×H100:
  > Table 6 (Instruction following): BF16 8.5 QPS; FP8 20.7; INT4 24.7.
- INT4 algorithms:
  > Table 1 (Real-world average, 8B): GPTQ 52.3 vs. AWQ 49.4; (70B): GPTQ 73.1 vs. AWQ 72.3.

In short, this study provides a clear, empirically grounded map for choosing between FP8, INT8, and INT4 in real deployments, shows how to tune INT8/INT4 to achieve high accuracy, and quantifies when each format is best for latency vs. throughput.
