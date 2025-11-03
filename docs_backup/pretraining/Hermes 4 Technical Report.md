# Hermes 4 Technical Report

**ArXiv:** [2508.18255](https://arxiv.org/abs/2508.18255)
**Authors:** Ryan Teknium, Roger Jin, Jai Suphavadeeprasit, Dakota Mahan, Jeffrey Quesnelle, Joe Li, Chen Guang, Shannon Sands, Karan Malhotra
**Institutions:** 

## 🎯 Pitch

Hermes 4 introduces a novel open-weight reasoning LLM, using graph-based data synthesis and verifier-backed sampling to produce high-quality reasoning tracks, while incorporating a single-token supervision technique for efficient length control. This advancement not only democratizes complex reasoning capabilities for math, coding, and creative tasks but also ensures reproducibility and cost-effective deployment, setting a new standard for open AI development.

---

## 1. Executive Summary (2–3 sentences)
Hermes 4 is a family of open‑weight “reasoning” large language models (LLMs) that combine structured, multi‑turn chain‑of‑thought (CoT) with broad instruction following. The paper’s core advances are a scalable data generation + verification pipeline (DataForge + Atropos), a training recipe that keeps reasoning controllable in length via targeted loss‑masking on a `</think>` stop token, and a transparent evaluation stack; together these yield competitive math, coding, and creative performance while remaining reproducible (Sections 2–4, Tables 3–4).

## 2. Context and Motivation
- Problem addressed
  - Open models increasingly show strong general capabilities, but “reasoner” behavior (explicit, multi‑step thinking) is mostly proprietary. Two practical blockers for open efforts are: (a) how to synthesize and verify large quantities of high‑quality reasoning traces at scale, and (b) how to control the amount of thinking so models don’t spill into excessively long, expensive outputs (Sections 1–3).
- Why it matters
  - Real tasks (math proofs, coding with test suites, tool use, JSON schemas) benefit from step‑by‑step reasoning and from verifiable outputs. But without length control, reasoners can saturate context windows (e.g., Hermes 4–14B initially hit the 40,960‑token limit on 60% of LiveCodeBench prompts; Section 3.1, Fig. 3a). Open, reproducible evaluation is also needed because scores depend on inference engines and hardware (Section 4.1).
- Prior approaches and shortcomings
  - Proprietary reasoners scale “thinking at inference time,” but their data and training are closed. Recent open efforts exist (e.g., DeepSeek‑R1, Qwen3 Reasoning), yet papers often lack:
    - a general‑purpose synthetic data factory with verifiers across many task types,
    - a simple mechanism to halt CoT reliably without hurting quality,
    - consistent, engine‑controlled evaluation to avoid score drift (Sections 1–2, 4.1).
- Positioning
  - Hermes 4 contributes (1) a graph‑based data generator (`DataForge`) that can be nested and judged, (2) verifier‑backed rejection sampling via `Atropos` environments across formats, instructions, code, schemas, tools, and personas, and (3) a training/eval stack that improves training efficiency and enforces a predictable “reasoning budget” (Sections 2–4).

## 3. Technical Approach
Step‑by‑step, from data to training to evaluation.

- Data generation and verification (Section 2)
  - Corpus scale and composition
    - ≈5.1M samples, ≈19B tokens: 3.5M “reasoning” samples and 1.6M non‑reasoning. Reasoning traces average ≈5× as many tokens as non‑reasoning and can reach 16k+ tokens per example (Section 2; Fig. 3a). Hermes‑3 data is partially retained for continuity.
  - `DataForge`: a graph‑based synthetic data generator (Section 2.1; Fig. 1)
    - What it is: a directed acyclic graph (DAG) of “nodes,” each a structured transformation (`struct → struct`) with preconditions/postconditions defined using a PDDL‑style action interface (pre/post “what must hold/what is guaranteed”).
    - How it works: a datapoint is a random walk through the DAG:
      1) Start from a cleaned, deduplicated “seed” passage (from DCLM/FineWeb; semantic dedup at cosine ≥0.7 via ModernBERT, then LLM‑judge filters; Section 2.1.1).
      2) Transform the passage (e.g., convert a Wikipedia article into a rap or debate; Fig. 1a).
      3) Generate an instruction conditioned on the transformed passage—either contextual (passage is in the prompt) or standalone (passage inspires but isn’t included).
      4) Use a type‑specific answer generator guided by a system prompt to produce an answer.
      5) Send instruction+answer to a specialized LLM‑judge with a rubric (coherence, relevance, complexity, style, tone). If it fails, iterate up to a cap; otherwise accept.
      6) Train not only on the final QA, but also on the intermediate LLM calls (Hermes 4 acquires skill in instruction generation and judging; Section 2.1.2).
    - Higher‑order graphs: any graph with single source/target is itself a node; graphs can nest into larger graphs (Section 2.1.3; Fig. 1b).
  - Rejection sampling with `Atropos` environments (Section 2.2)
    - Definition (selective): “Rejection sampling” here means generating many candidate trajectories and keeping only those that pass programmatic or model‑based verifiers.
    - About `Atropos`: an async environment microservice manager that hosts ≈1000 task‑specific verifiers; stores multiple unique correct trajectories per prompt (OpenThoughts recipe; Section 2.2).
    - Key environments (Sections 2.2.1–2.2.5)
      - `Answer Format Training`: learns to produce correct output formats (e.g., LaTeX `\boxed{}`) with a binary reward and strict `<think>…</think>` delimiters.
      - `Instruction Following`: uses RLVR‑IFEval tasks (e.g., “every N‑th word in French”) to collect successful constraint‑satisfying traces.
      - `Internbootcamp`: 70k accepted trajectories from ≈1k reasoning tasks (solutions sampled from DeepHermes and others, filtered by correctness).
      - `Schema Adherence`: dynamic Pydantic schemas; models must either (a) generate valid JSON for a schema, or (b) edit malformed JSON to pass validation. Binary reward plus length penalty.
      - `Tool Use`: validates JSON tool‑call objects; rewards exact schema and value correctness.
  - Building a domain‑covering set of tasks (Section 2.3)
    - `Taxonomies`: recursively partition a domain into subdomains until leaves are concrete prompts (used, for example, to enumerate parseable output formats).
    - `PersonaHub`: synthesize persona‑grounded tasks (e.g., from FinePersonas) that seed diverse, realistic prompts, then backfill reasoning traces with stronger models (Appendix A).

- Training recipe (Section 3)
  - Base checkpoints and framework
    - Start from `Llama‑3.1 405B` and `70B` for larger models and from `Qwen3‑14B` for 14B; use a modified TorchTitan stack (footnote 5).
  - Efficiency for heterogeneous lengths (Section 3; Fig. 3a)
    - Pack samples in advance via `First‑Fit Decreasing` (FFD) bin‑packing so that batched sequences almost perfectly fill the context window; reported >99.9% batch efficiency.
    - Use `Flex Attention` to restrict attention within each packed sample.
    - Loss is computed only on tokens emitted in the `assistant` role (Fig. 2).
  - Optimization and hardware (Section 3; Table 1)
    - 192× NVIDIA B200 GPUs; mix of DDP/TP/FSDP depending on size.
    - Cosine LR with 300 warmup steps, 9000 total steps, global batch 384, training context 16,384 tokens.
    - Per‑model settings (Table 1):
      - `14B`: FSDP, 56B tokens, LR 5e‑5, ≈4,454 B200 hours
      - `70B`: FSDP+TP, 56B tokens, LR 1e‑5, ≈12,864 B200 hours
      - `405B`: FSDP+TP, 56B tokens, LR 5e‑6, ≈71,616 B200 hours

- Length‑control fine‑tuning via a single supervision token (Section 3.1; Fig. 3b; Table 2; Appendix B)
  - Issue observed: The 14B model in reasoning mode frequently ran past 40,960 tokens on coding tasks (60% overlong on LiveCodeBench; Section 3.1).
  - Mechanism: A second SFT stage teaches the model to emit `</think>` at a fixed budget (30k tokens):
    - Generate on‑policy long reasoning traces, forcibly insert `</think>` at exactly 30k tokens, then let the model produce the final answer (Fig. 3b).
    - Mask all tokens except the `</think>` and the following `<eos>`; this concentrates learning on “when to stop” without training on the reasoning content, thus avoiding synthetic‑data collapse from recursively training on self‑generated chains (Section 3.1; discussion cites [64, 19]).
    - Data preparation: 300k prompts across WebInstruct‑Verified, rSTAR‑Coder, DeepMath‑130k; filter for long generations; two cases handled depending on whether generation had already produced a `</think>` or not (Section 3.1.1).
    - Implementation using Axolotl’s character‑span masking interface (Section 3.1.2).
  - Effect (Table 2): massive reduction in “overlong@40960 toks”
    - Example for Hermes‑4‑Qwen3‑14B (reasoning mode):
      > “Overlong@40960 toks” drops from 60.0% to 0.1% on LiveCodeBench, and from 28.2%→0.1% (AIME’24), 25.9%→0.1% (AIME’25), 18.2%→0.2% (GPQA).
    - Accuracy mostly stable or improved on some tasks (e.g., LCB pass@1 improves from 28.6→42.5), with small regressions on others (AIME’25 48.7→46.8; Table 2).
  - Ablations (Appendix B; Table 5)
    - A stricter 20k budget with standard masking paradoxically increased overlong rates, likely because training on long chains selected for prefixes that trigger verbosity (Table 5a “Standard Masking”). Training on `</think>` only suppressed overlong but hurt AIME scores (Table 5b). The final choice of a 30k budget yields a better trade‑off (Table 2).

- Evaluation stack (Section 4)
  - Single, shared inference engine to avoid cross‑engine variance: an OpenAI‑compatible endpoint (SGLang 0.4.9.post3 with Triton backend on B200s; TP8 sharding; Section 4.5).
  - Benchmarks via `lighteval` (math/MCQ), `Atropos` (code and custom), and EQBench suites (Sections 4.2–4.3). All sampling parameters and contexts are standardized (Section 4.5).
  - LiveCodeBench engineering: overlap inference and verification using Modal; verify all test cases in sandbox; tuned for compute‑bound throughput (Section 4.3.3).
  - Elastic clusters: sglang‑router based worker add/remove with preemption and requeue, ensuring full utilization without blocking training jobs (Section 4.4).

## 4. Key Insights and Innovations
- Graph‑of‑graphs data synthesis with verified trajectories (Section 2; Fig. 1)
  - Novelty: `DataForge` composes declarative DAG nodes (PDDL‑style pre/post) into nested graphs; combined with `Atropos` verifiers this yields millions of accepted reasoning traces across diverse tasks (formatting, constraints, schemas, tools).
  - Why it matters: A general, reusable way to produce supervision for both “how to think” and “how to comply with formats/tools,” beyond narrow math/code corpora.
- Single‑token supervision for reasoning truncation (Section 3.1; Table 2; Appendix B)
  - Novelty: Rather than train on entire synthetic CoTs, Hermes 4 learns only when to emit `</think>`, focusing gradients on a single token. This sharply reduces over‑length outputs (>98.9% reduction across tested benchmarks; Table 2) while limiting accuracy regressions.
  - Significance: A simple, robust mechanism to budget “thinking compute” at inference time without collapsing reasoning diversity, addressing a major practical pain point for reasoner deployment.
- High‑efficiency batching for heterogeneous sequence lengths (Section 3; Fig. 3a)
  - Novelty: FFD packing with `Flex Attention` yields >99.9% batch efficiency despite long‑tail sample lengths; loss is restricted to `assistant` tokens.
  - Impact: Better throughput and training stability for mixed tasks (short instructions interleaved with long reasoning).
- Reproducible, engine‑controlled evaluation with detailed logging (Sections 4.1–4.3)
  - Novelty: A single OpenAI‑compatible endpoint for all benchmarks; Atropos provides per‑sample parsing/grading logs and overlapped verification for code. Internal checks found discrepancies between some third‑party parsers and LLM judges (Section 4.3.1).
  - Importance: Reduces “benchmark noise” and makes results easier to replicate.

## 5. Experimental Analysis
- Evaluation setup (Sections 4.2, 4.5)
  - Reasoning and code at 40,960 context; others at 32,768. Default sampling T=0.6, Top‑p=0.95, Top‑k=20 (Qwen3 recipe), with exceptions (e.g., DeepSeek‑V3 at T=0.3). pass@1 estimated with multiple samples depending on task: AIME (64), MATH‑500 (4), GPQA (8), LCB (16). Logged generations released (Section 4.6 link).
  - Benchmarks span reasoning (AIME’24/’25, MATH‑500, GPQA), code (LiveCodeBench v6 Aug2024+), logic (BBH), knowledge (MMLU, MMLU‑Pro, SimpleQA), constraint following (IFEval), general preference/“vibe” (Arena‑Hard v1 with LLM judge), reward modeling (RewardBench), reading comprehension (DROP, MuSR, OBQA), and creativity (EQBench3, CreativeWriting3) (Sections 4.2, 4.6; Tables 3–4).
- Main quantitative results
  - 405B model vs open baselines (Table 3)
    - Reasoning:
      > MATH‑500: 96.2 (R) vs DeepSeek‑R1 97.5; GPQA‑Diamond: 70.6 (R) vs R1 78.1.  
      > AIME’24/’25: 81.9/78.1 (R) vs R1 86.5/83.1.
    - Code:
      > LiveCodeBench v6 Aug2024+: 61.4 (R) vs R1 71.8; Qwen3‑235B 65.1.
    - Knowledge:
      > MMLU: 87.2 (R) vs Qwen3‑235B 89.3; MMLU‑Pro: 80.6 (R) vs Qwen3 83.1.
    - Alignment/style:
      > Arena‑Hard v1: 93.7 (R), comparable to Qwen3 93.9 and DeepSeek‑V3 92.6.  
      > RewardBench: 73.0 (R), near Qwen3 74.2.  
      > IFEval (Loose): 81.5 (R), lower than Qwen3 91.4.
    - Creativity:
      > EQBench3: 85.5 (R), close to R1 86.5.  
      > CreativeWriting3: 79.3 (R), near R1 80.3.
    - Refusals (internal metric):  
      > RefusalBench: 57.1 (R), substantially higher willingness to respond than DeepSeek‑V3 28.1 and Llama‑3.1‑405B 21.7 (Fig. 4).
  - 70B and 14B (Table 4)
    - 70B reasoning:
      > AIME’24/’25: 73.5/67.5; GPQA 66.1; LCB 50.5.  
      > MMLU 88.4; RewardBench 64.9; Arena‑Hard 90.1.
    - 14B reasoning (with 30k stop training):
      > AIME’24/’25: 55.4/46.8; MATH‑500 91.1; LCB 42.5.  
      > RewardBench 63.5; Arena‑Hard 83.0.
  - Length‑control effectiveness (Table 2)
    - On the 14B, “overlong@40960” falls by ≈99% across AIME/GPQA/LCB with minimal score drops; LiveCodeBench improves (28.6→42.5), AIME’25 decreases modestly (48.7→46.8).
- Do experiments support claims?
  - Reasoning strength: Competitive but generally behind the strongest closed/open reasoners on the most demanding math/GPQA (Table 3). Nevertheless, 405B is near‑frontier in creative and arena‑style preference tasks, and strong on general knowledge.
  - Length control: Convincingly demonstrated with both aggregate (Table 2) and ablation evidence (Appendix B, Table 5), including failure modes for naïve SFT on long traces.
  - Reproducibility: Unified engine, fixed hyperparameters, and released generations bolster credibility (Sections 4.1, 4.6).
- Ablations / robustness / failure cases
  - Appendix B shows that training on full long traces (even mixed with SFT) can worsen overlong rates, likely by learning verbosity‑inducing prefixes; `</think>`‑only masking avoids this but can depress AIME (Table 5b).
  - Qualitative behavior (Section 5) highlights “policy rigidity” differences and chat‑template sensitivity; useful but not a controlled quantitative study.

## 6. Limitations and Trade‑offs
- Assumptions and scope
  - Reliance on LLM judges and verifiers: Although different weights are used for generator vs judge to reduce bias (Section 2.1.2), judge‑based acceptance still risks subtle stylistic preferences.
  - Fixed reasoning budget: The 30k `</think>` stop is a coarse global limit; tasks that truly need longer chains may be truncated (Section 3.1). No adaptive per‑task budgeting is explored.
- Performance trade‑offs
  - Reasoning vs knowledge/code: The 405B trails DeepSeek‑R1 on AIME/GPQA and trails Qwen3‑235B and R1 on LiveCodeBench (Table 3), suggesting remaining gaps in deep math/coding.
  - IFEval: Constraint‑following “loose” score is below top baselines (Table 3–4), indicating room to improve instruction‑constraint adherence without harming reasoning.
- Computational cost and data
  - Training hours are substantial (e.g., ≈71k B200 hours for 405B; Table 1) despite only 56B tokens per size—this is a post‑training (SFT‑style) stage, not full pretraining.
  - Synthetic data risks: The team mitigates collapse by not training on model‑generated chains, but the dataset still leans heavily on synthetic generation and LLM judging (Sections 2–3).
- Behavior plasticity
  - The model is highly sensitive to chat templates and system prompts (Section 5.3). While powerful, this can make behavior less predictable across deployments without tight prompt control.
- Safety/Refusal profile
  - RefusalBench shows low refusal rates overall (Fig. 4). While three categories invert scoring for safety‑critical topics, the broader tendency “to answer” may conflict with conservative safety postures in some applications.

## 7. Implications and Future Directions
- How this changes the landscape
  - Hermes 4 demonstrates that open‑weight reasoners can be trained with verifiable, multi‑domain data, controlled reasoning length, and reproducible evaluation. The `</think>`‑only supervision is a simple, general mechanism other teams can adopt to bound inference cost without retraining full CoTs (Sections 3.1, Appendix B).
- Follow‑up research
  - Adaptive reasoning budgets: Learn a policy that predicts the needed “thinking tokens” per prompt (e.g., via a small controller head) rather than a fixed 30k cutoff; combine with a “thinking efficiency” metric (Section 3.1 cites concurrent work [33,34]).
  - Richer verifiers: Extend Atropos environments to more domains (e.g., theorem provers like Lean, more tool ecosystems), and to partial‑credit shaping rather than binary rewards (Sections 2.2, 4.3).
  - Better constraint‑following without regressions: Close the gap on IFEval while preserving CoT quality—e.g., staged training that alternates constraint‑only batches with reasoning batches, or multi‑objective optimization.
  - Robustness to template drift: Formalize and evaluate chat‑template sensitivity observed in Section 5.3; train invariance or controlled persona‑adoption strategies.
- Practical applications
  - Long‑form math/coding assistants with bounded cost: The `</think>` mechanism helps guarantee latency/compute budgets in production.
  - Enterprise agents: Schema adherence and tool‑use verifiers (Section 2.2.4–2.2.5) make the models suitable for structured outputs (APIs, JSON) and multi‑step tool calling.
  - Content creation and review: Strong performance on Arena‑Hard and creative benchmarks (Tables 3–4) supports use in drafting, rewriting, and LLM‑as‑a‑judge pipelines (RewardBench).

> Key takeaway: Hermes 4’s contribution is less a single best score and more a repeatable system—data graphs + verifier‑backed trajectories + targeted length supervision + controlled evaluation—that other open‑weight efforts can reuse to build practical, cost‑predictable reasoners.
