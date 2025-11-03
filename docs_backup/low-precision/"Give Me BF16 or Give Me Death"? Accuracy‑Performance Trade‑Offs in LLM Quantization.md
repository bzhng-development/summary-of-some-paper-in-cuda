# "Give Me BF16 or Give Me Death"? Accuracy‑Performance Trade‑Offs in LLM Quantization

**ArXiv:** [2411.02355](https://arxiv.org/abs/2411.02355)
**Authors:** Eldar Kurtic, Alexandre Marques, Shubhra Pandit, Mark Kurtz, Dan Alistarh
**Institutions:** Not listed on arXiv

## 🎯 Pitch

This study provides a comprehensive evaluation of low-precision formats for large language model inference, identifying FP8 as nearly lossless in accuracy while unlocking significant speed and cost savings through tailored INT8 and INT4 quantization strategies. The findings offer concrete, evidence-based guidance, enabling organizations to optimize LLM deployments based on serving conditions, enhancing both cost-efficiency and performance without significant accuracy loss. This shifts the quantization dialogue from speculative to data-driven decision-making, opening new pathways for scalable, high-performance AI applications.

---

## 1. Executive Summary
This paper delivers a large-scale, end-to-end examination of how different low-precision formats for large language model (LLM) inference trade off accuracy, speed, and cost. It benchmarks FP8, INT8, and INT4 schemes across the full Llama‑3.1 family (8B, 70B, 405B) and multiple GPU generations, and ties accuracy findings to measured throughput/latency under realistic serving modes in vLLM. The central practical takeaways are: FP8 (W8A8-FP) is near-lossless in accuracy, well‑tuned INT8 (W8A8-INT) is much better than commonly believed, and 4‑bit weight‑only (W4A16-INT) often gives the best latency/cost in synchronous deployments while 8‑bit formats dominate throughput in asynchronous serving.

## 2. Context and Motivation
- Problem addressed
  - LLM quantization reduces numeric precision of model parameters and/or activations to lower memory footprint and improve inference speed. The community lacks a systematic, apples‑to‑apples picture of the accuracy versus performance trade‑offs for the formats that are actually supported and fast in modern inference stacks.
  - The gap spans two axes:
    - Accuracy across both academic and real‑world, open‑ended tasks at multiple model sizes.
    - Throughput/latency/cost on real hardware (A6000, A100, H100) and serving modes (synchronous and asynchronous continuous batching).

- Why it matters
  - Serving large models is expensive and often bottlenecked by memory bandwidth. Small degradations in accuracy can be acceptable if they yield large gains in cost and responsiveness. Organizations need reliable guidance on which quantization format fits their workload and hardware.

- Prior approaches and shortcomings
  - Many studies focus narrowly on accuracy on academic sets or only one model size, or report results without careful tuning, yielding pessimistic views of INT8 activation quantization (Section 2.2).
  - Some high‑compression formats (e.g., 2‑bit vector quantization) are not efficient beyond batch size 1, limiting practical value (Section 2.1).
  - Claims that W8A8‑INT suffers large losses (10%+) are shown to hinge on poor hyperparameters and weak calibration choices (Section 2.2, Appendix A.2).

- Positioning of this work
  - Comprehensive and deployment‑oriented: evaluates formats with production kernels in vLLM (version 0.6.4.post1) across three GPU generations and seven realistic use cases; links accuracy to end‑to‑end performance; covers the entire Llama‑3.1 range including 405B (Sections 3 and 5).
  - Methodologically careful: strong baselines, tuned hyperparameters, and large evaluation volume (>500k runs).

## 3. Technical Approach
This is an empirical study with targeted algorithmic choices that reflect what is both accurate and fast to serve today.

- Quantization formats studied (Section 3.2)
  - Naming shorthand used throughout figures/tables:
    - `W8A8-FP` (“FP8”): weights and activations quantized to 8‑bit floating point where hardware supports it (Hopper/Ada).
    - `W8A8-INT` (“INT8”): weights and activations quantized to 8‑bit integers (widely supported, including Ampere).
    - `W4A16-INT` (“INT4”): 4‑bit integer weights, 16‑bit activations (weight‑only low‑bit; often the fastest for decode‑bound workloads).
  - Why these: they have mature, efficient kernels in vLLM and map well to today’s GPUs. Ultra‑low‑bit vector formats are excluded because they are inefficient for batches >1 (Section 2.1).

- How each format is implemented (Section 3.2)
  - FP8 (W8A8-FP)
    - Weights: symmetric per‑output‑channel quantization using round‑to‑nearest assignment.
    - Activations: dynamic per‑token quantization (no calibration data required).
    - Design rationale: FP8 avoids many of INT8’s outlier issues for activations and has native hardware support on Hopper/Ada.
  - INT8 (W8A8-INT)
    - Weights: GPTQ, a post‑training quantization method that uses second‑order information (via calibration data) to minimize layer‑wise quantization error.
      - Intuition: for each weight group, GPTQ quantizes then applies a small, analytically computed correction that accounts for the local curvature of the loss, reducing the error introduced by rounding.
    - Activations: dynamic per‑token quantization; for the 70B model this is augmented with SmoothQuant.
      - SmoothQuant (activation‑to‑weight scaling) shifts amplitude from hard‑to‑quantize activation channels into the weights using a precomputed scaling from calibration data, reducing activation outliers while preserving function (Section 2.1).
    - Calibration data: random tokens are sufficient at 8B; higher‑quality calibration (Platypus/Lee et al., 2023) is used for 70B/405B (Section 3.2).
  - INT4 (W4A16-INT)
    - Weights: GPTQ with MSE‑optimal clipping, group size 128; activations remain at 16‑bit for robustness.
      - MSE‑optimal clipping squares the quantization error and selects a clipping threshold that minimizes mean squared error before rounding.
    - Calibration: random tokens hurt accuracy at 4‑bit, so OpenPlatypus data is used (Section 3.2).
    - Why not AWQ as default: head‑to‑head comparisons favor GPTQ on real‑world, open‑ended tasks (Table 1 and Appendix A.2).

- Core quantization primitive and why activations are hard (Section 2.1)
  - Many methods start from round‑to‑nearest (RTN) with min‑max scaling. Equation (1) shows:
    - Each group of `g` weights `x` is scaled by `s(x) = (max(x)−min(x))/(2^b−1)` and shifted by zero‑point `z(x)=min(x)`, then rounded to the nearest integer.
    - Outliers in activations (values far larger than average) inflate `max(x)−min(x)`, wasting dynamic range and hurting INT8 activation quantization; SmoothQuant counteracts this by rebalancing magnitudes.

- Models, datasets, and serving stack (Sections 3.1–3.2, 5)
  - Models: Llama‑3.1‑Instruct at 8B, 70B, and 405B; reasoning‑tuned `DeepSeek‑R1‑Distill` variants from Llama and Qwen families (Section 4.3, Table 4).
  - Benchmarks:
    - Academic: Open LLM Leaderboard V1 and V2 (Tables 2, 3, 10, 11).
    - Real‑world open‑ended: Arena‑Hard‑Auto‑v0.1 (Table 7), HumanEval and HumanEval+ (Table 3; Appendix Figures 5–6), long‑context RULER (Table 3).
    - Text similarity vs. full‑precision: ROUGE/BERTScore/STS on Arena‑Hard prompts under greedy decoding (Figure 1).
  - Performance evaluation in vLLM across seven representative tasks (code completion, docstrings, code fixing, RAG, instruction following, multi‑turn chat, summarization) with characteristic prefill/decode lengths (Section 5). Hardware: A6000, A100, H100; synchronous and asynchronous serving; cost computed from Lambda Labs pricing (Table 9).

- Serving concepts (Section 5)
  - Prefill vs. decode: prefill processes all input tokens in parallel (compute‑bound); decode generates tokens one by one (memory‑bandwidth‑bound).
  - Implication: weight‑only quantization (INT4) chiefly accelerates decode; weight+activation quantization (FP8/INT8) also speeds up prefill.

## 4. Key Insights and Innovations
1. FP8 is effectively lossless on accuracy across sizes
   - What’s new: end‑to‑end demonstration that `W8A8-FP` recovers ≈100% accuracy across both simple and challenging suites.
   - Evidence:
     - Leaderboard V1: 99.31–100.12% recovery for 8B–405B (Table 2).
     - Leaderboard V2: 99.9–101.2% recovery (Table 3).
     - Real‑world: Arena‑Hard and coding tasks match baselines within confidence intervals (Table 3; Table 7).
   - Why it matters: FP8 gives prefill speedups without sacrificing accuracy and is natively supported on modern GPUs.

2. Properly tuned INT8 is much stronger than commonly believed
   - Distinguishing factor: combines GPTQ weights with dynamic activations and, where needed (notably 70B), SmoothQuant with good calibration data.
   - Evidence:
     - Average loss is small (≈1–3 percentage points) rather than the 10%+ often reported without tuning (Section 2.2).
     - Leaderboard V2 average: 97.3% at 70B, 98.3% at 405B (Table 3).
   - Significance: expands the viable use of INT8 activation quantization to larger models when calibration and scaling are handled carefully.

3. 4‑bit weight‑only (W4A16-INT) is surprisingly competitive—and GPTQ > AWQ on real‑world tasks
   - What’s different: GPTQ with MSE‑optimal clipping and higher‑quality calibration, plus evaluation on open‑ended tasks.
   - Evidence:
     - Academic parity: AWQ vs GPTQ nearly tied (Table 1 left).
     - Real‑world advantage: GPTQ beats AWQ by noticeable margins, e.g., on 8B HumanEval 67.1 vs 63.0 pass@1 and MBPP 65.8 vs 62.8 (Table 1 right; Appendix Tables 12–13).
   - Why it matters: for latency‑sensitive serving, INT4 gives the best cost/latency in synchronous mode, and practitioners should prefer GPTQ in these conditions.

4. Deployment‑level guidance: INT4 for synchronous latency; W8A8 for asynchronous throughput
   - Evidence:
     - Synchronous: INT4 gives 2–3× cost reduction at 8B/70B and 5–7× at 405B, with lower latency (Table 5). Example: 8B code completion latency drops from 24.5s (BF16) to 9.7s (INT4), Q/$ rises from 183 to 462 (Table 5).
     - Asynchronous: W8A8 formats maximize QPS at higher latencies; INT4 remains competitive and sometimes wins but tends to lose at high‑throughput regimes (Table 6; Figures 2–3).
   - Practical value: concrete, data‑driven prescriptions for production choices based on workload and SLA.

5. Larger quantized models preserve semantics and even phrasing
   - Evidence from text similarity (Figure 1):
     - 70B/405B: ROUGE‑1 ≈0.7, BERTScore ≈0.93, STS ≈0.96 vs full‑precision—indicating close word and structure overlap.
     - 8B degrades somewhat in phrasing (ROUGE‑L ≈0.46–0.51) but keeps semantic fidelity (STS ≈0.94–0.95).

## 5. Experimental Analysis
- Evaluation methodology (Sections 3.1–3.2; 5; Appendices)
  - Benchmarks:
    - Academic: Open LLM Leaderboards V1 and V2, covering world knowledge, reasoning, math, instruction following (Tables 2–3, 10–11).
    - Real‑world: Arena‑Hard‑Auto‑v0.1 (two runs, 95% CI reported in Table 7), HumanEval/HumanEval+ coding (Table 3 and Appendix Figures 5–6), RULER for long‑context (Table 3).
    - Reasoning: AIME’24, MATH‑500, GPQA‑Diamond with pass@1 estimated from 20 samples/query using LightEval (Section 4.3; Table 4).
  - Serving performance:
    - vLLM 0.6.4.post1, three GPU types, seven use cases with task‑typical prefill/decode lengths (Section 5); synchronous and asynchronous settings; cost via Lambda pricing (Table 9).
  - Baselines: Full‑precision BF16 for all models; FP8, INT8, and INT4 variants as above.

- Headline accuracy numbers
  - Academic (Tables 2–3)
    - V1 (8 tasks): All formats recover ≈99% of BF16 overall. Sample 8B averages: BF16 74.06 vs FP8 73.55 (99.31%), INT8 74.29 (100.31%), INT4 73.11 (98.72%) in Table 2.
    - V2 (harder tasks): 8B shows more variance (INT4 96.1% recovery), but 70B and 405B remain strong: 70B FP8 100.0%, INT8 97.3%, INT4 97.4%; 405B FP8 99.9%, INT8 98.3%, INT4 98.9% (Table 3).
    - Hardest subtasks: integer activation quantization is the main challenge—e.g., MMLU‑Pro at 405B drops to 97.81% for INT8 (Table 11), while FP8 remains ≈99%.
  - Real‑world tasks (Table 3; Table 7)
    - Arena‑Hard: differences are within 95% CIs for most configs. 405B BF16 67.4 vs FP8 66.9 vs INT8 64.6 vs INT4 66.5 (Table 3 and Table 7).
    - Coding: HumanEval pass@1 at 70B is stable or slightly improved with quantization (BF16 79.7 vs FP8 80.0 vs INT4 80.5, Table 3). HumanEval+ similarly stable.
    - Long‑context: RULER at 8B/70B maintains ≥98% average score recovery; INT4 is slightly lower at 8B (81.1 vs 82.8, Table 3).
  - Reasoning‑tuned models (Table 4)
    - Across Llama‑70B and Qwen‑32B/14B/7B/1.5B, FP8/INT8/INT4 recover >99% average (except small models at INT4) on AIME’24, MATH‑500, and GPQA‑Diamond. Example: Llama‑70B FP8 averages 76.5 vs BF16 76.2; INT4 averages 75.0 (98.3%).

- GPTQ vs AWQ at INT4 (Table 1; Appendix A.2)
  - Academic: almost tied (e.g., 8B average 49.82 vs 50.05).
  - Real‑world: GPTQ clearly ahead—8B average 52.3 vs 49.4; large gaps on coding (HumanEval 67.1 vs 63.0; MBPP 65.8 vs 62.8).
  - Contributing factors: MSE‑optimal clipping for GPTQ (AWQ run used abs‑max), higher‑quality calibration, and inclusion of open‑ended tasks (Section “INT4 Quantization Algorithms” and Table 1).

- Performance and cost findings
  - Synchronous (Table 5)
    - INT4 dominates latency and cost per query.
      - 8B on A6000 (code completion): latency 24.5s (BF16) → 9.7s (INT4); Q/$ 183 → 462; cost reduction (CR) 2.39×.
      - 70B on A100 (docstrings): 2.9s (BF16) → 2.8s (INT4) but Q/$ 343 → 718 (2.09×); in several tasks both latency and cost improve.
      - 405B on H100: code completion Q/$ 1 → 8; CR 5.15×. Similar gains across tasks; INT4 enables using fewer GPUs with acceptable latency (Section 5.1).
  - Asynchronous (Table 6; Figures 2–3)
    - W8A8 (FP8/INT8) often yields the highest throughput (QPS) and cost‑efficiency at higher batching.
      - 16×H100, 405B summarization: BF16 8.5 QPS (Q/$ 638) → FP8 20.7 (1561) → INT4 24.7 (1856). Here INT4 also shines, but across many tasks FP8/INT8 take the QPS crown.
      - 4×H100, 70B summarization: BF16 1.7 QPS (0.5k Q/$) → FP8 2.6 (0.8k) → INT4 2.2 (0.6k).
    - Latency–throughput trade‑off:
      - Figure 2 (8B docstrings, 1×A6000): INT4 has lower inter‑token latency at low QPS; W8A8‑INT overtakes at higher QPS.
      - Figure 3 (70B code fixing, 2×A100): same crossover behavior—INT4 for low‑latency; W8A8 for high‑throughput.

- Do the experiments support the claims?
  - Yes, with breadth and detail:
    - Accuracy: dozens of tasks across three model sizes (Tables 2–4); text similarity corroborates qualitative parity (Figure 1).
    - Performance: end‑to‑end vLLM benchmarks across multiple GPUs and seven realistic workloads (Tables 5–6; Figures 2–3).
    - Robustness: 95% CIs for Arena‑Hard (Table 7); calibration choices explained; additional reasoning‑specific suite (Table 4).
  - Where results are conditional:
    - INT8 activation quantization requires good calibration and SmoothQuant for some sizes (Section 3.2).
    - INT4 degrades more on the smallest models and on a few hard V2 tasks but remains competitive overall (Tables 3–4).

## 6. Limitations and Trade-offs
- What the study assumes or leaves out (Limitations section)
  - KV‑cache, input embeddings, and LM head are not compressed here. Real deployments often quantize or pack KV cache; the accuracy/latency impact of KV quantization remains open.
  - Language coverage: primarily English, instruction‑tuned models; multilingual or domain‑specialized tasks may behave differently.
- Sensitivity to calibration and hyperparameters
  - INT8 accuracy hinges on high‑quality calibration and SmoothQuant choices, particularly at 70B (Section 3.2; Tables 3 and 11). Poor calibration can reproduce prior pessimistic results.
  - INT4 requires careful clipping and data; random‑token calibration hurts (Section 3.2).
- Hardware and software scope
  - Results are tied to vLLM 0.6.4.post1 kernels and three NVIDIA GPU families. Different runtimes/hardware may shift break‑even points.
- Cost modeling
  - Cost per query uses Lambda on‑demand rates (Table 9). Reserved/on‑prem pricing could change the relative economics.

## 7. Implications and Future Directions
- How this changes the landscape
  - Moves quantization guidance from “folk wisdom” to evidence‑based rules:
    - For latency‑sensitive, single‑query or small‑batch serving: prefer `W4A16-INT` (INT4) to cut latency and cost substantially (Table 5; Figures 2–3).
    - For high‑throughput, asynchronous serving: prefer `W8A8` (FP8 where supported, INT8 otherwise) to maximize QPS (Table 6).
    - When hardware supports FP8 (H100/Ada), it is the safest near‑lossless option (Tables 2–3).
  - Demonstrates that carefully tuned INT8 activation quantization is viable even for very large models (70B–405B), countering the narrative that INT8 is inherently unreliable (Tables 3 and 11).

- Follow‑up research enabled
  - Quantizing beyond weights/activations:
    - KV‑cache quantization and scheduling that co‑optimize memory bandwidth and cache reuse.
    - Quantization of input embeddings and output heads with guardrails for generation quality.
  - Mixed‑precision policies:
    - Layer‑ or block‑wise format selection (e.g., INT4 for attention/MLP weights, FP8 activations where prefill dominates) driven by per‑operator sensitivity.
  - Better calibration:
    - Automatic selection of calibration corpora per task/domain; on‑the‑fly calibration that adapts to workload drift.
  - System co‑design:
    - Kernel and scheduler improvements that exploit quantized formats more aggressively in continuous batching and context‑parallel setups.

- Practical applications
  - Chat assistants and code tools: pick INT4 for low‑latency completions, FP8/INT8 for bulk batch processing (Tables 5–6).
  - High‑end deployments of very large models (e.g., 405B): use INT4 to reduce GPU count/communication for latency‑sensitive endpoints; FP8/INT8 to maximize throughput in batch jobs.
  - Cost‑aware scaling: the cost‑reduction factors (up to ≈5–7× in synchronous mode at 405B, Table 5) make previously impractical deployments feasible.

> In short: with careful algorithmic choices (GPTQ with proper clipping, SmoothQuant with good calibration, dynamic activation quantization) and an awareness of serving mode, FP8/INT8/INT4 quantization can preserve ≥97–100% of accuracy on large, realistic benchmarks (Tables 2–4) while yielding large, measurable gains in latency, throughput, and cost (Tables 5–6).
