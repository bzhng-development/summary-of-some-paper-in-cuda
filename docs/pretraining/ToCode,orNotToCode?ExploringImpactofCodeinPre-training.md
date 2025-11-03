# To Code, or Not To Code? Exploring Impact of Code in Pre-training

**ArXiv:** [2408.10914](https://arxiv.org/abs/2408.10914)

## 🎯 Pitch

This paper delivers the first large-scale, controlled study of how programming code in pre-training data affects large language model (LLM) performance on both code and non-code tasks. By dissecting when, how much, and what type of code to include, the authors reveal that even models not explicitly trained for coding see substantial gains in generalization, reasoning, and world knowledge—up to a 12× improvement in code benchmarks and significant boosts in natural language tasks. These results reshape our understanding of optimal data mixtures, showing that preserving and improving code data in LLM pre-training yields consistent benefits far beyond software applications, influencing the fundamental design of next-generation AI systems.

---

## 1. Executive Summary
This paper asks a focused question: how much does including programming code in the pre‑training data of a large language model (LLM) help on non‑code tasks, and under what conditions? Through 64 controlled pre‑training runs (470M–2.8B parameters) that vary when and how code is used, the study shows that code is a critical ingredient for generalization: in the best setting, adding code yields +8.2% relative gain in natural language reasoning, +4.2% in world knowledge, +6.6% in open‑ended generation win‑rates, and a 12× boost in code tasks (Section 3.6, Table 2).

## 2. Context and Motivation
- Problem addressed
  - Many state-of-the-art LLMs include a non‑trivial share of code during pre‑training even if they are not “code models.” The field has anecdotal consensus that code helps general capabilities, but lacks a systematic analysis that isolates the effect of code on non‑code tasks (Section 1).
- Why it matters
  - Practical: Data mixture is one of the largest levers for LLM quality. Understanding whether, how much, and when to include code affects training cost, data curation, and final model behavior (Introduction; also Figure 1 for the “levers” studied).
  - Scientific: Code differs from web text in structure and semantics. If code improves reasoning or knowledge skills, that informs theories of how LLMs learn abstractions.
- Prior approaches and gaps
  - Prior models used code without isolating its effects (e.g., PaLM, Gopher, BLOOM; Introduction). Newer models increased code share (Llama 3 uses ~17% vs Llama 2’s ~4.5%; Section 1).
  - Topic-specific hints existed: improvements in data‑limited scaling, entity tracking, and math reasoning (Introduction), but no exhaustive, controlled study on broad non‑code performance.
- Positioning
  - This work supplies a systematic, multi‑axis ablation: proportion of code, where code appears in the training schedule, code quality/type, model scale, and the “cooldown” stage, evaluated on reasoning, world knowledge, code, and LLM‑as‑a‑judge win‑rates (Figure 1).

## 3. Technical Approach
This is an empirical study that varies the pre‑training recipe while keeping other factors controlled and then measures downstream effects.

- Core terminology (paper‑specific)
  - `continued pre‑training`: start from a pre‑trained model and continue training on a new data mix for a fixed token budget (Section 2.1).
  - `cooldown`: a short final training stage that up‑weights high‑quality datasets and linearly anneals the learning rate to a small value. It is known to boost instruction-following and overall quality (Section 2.1; 3.5).
  - `LLM‑as‑a‑judge win‑rate`: a pairwise evaluation where a strong LLM compares two model outputs and picks a preferred one. The score is the percentage of pairwise wins (Section 2.2).
  - `pass@1`: for code benchmarks, the fraction of problems correctly solved by a single generated attempt (Section 2.2).

- Experimental levers (Figure 1)
  1) Proportion of code in the pre‑training mix.
  2) Code quality and properties (web code; markup like Markdown/HTML; code‑adjacent data like commits and Q&A; small, high‑quality synthetic code).
  3) When code appears: from-scratch, during continued pre‑training, or only in cooldown.
  4) Model initialization: from random, from a code‑heavy LM, or from a balanced code+text LM.
  5) Model scale: 470M and 2.8B parameters.

- Data pipeline (Section 2.1; Appendices A.1–A.3)
  - Text: SlimPajama (627B tokens) with code sources removed (GitHub and StackExchange excluded), yielding 503B “text‑only” tokens.
  - Web code: The Stack with quality filters and top 25 languages, 139B tokens.
  - Markup: Markdown/CSS/HTML/etc. processed as a separate stream, 180B tokens.
  - Code‑adjacent: GitHub commits, Jupyter notebooks, StackExchange (21.4B tokens total).
  - Synthetic code: small, high‑quality, formally verified Python problems (3.2B tokens).
  - Cooldown mix: up‑weights “high‑quality” text, math, code, and instruction‑style datasets; learning rate linearly annealed to 1e‑6 (Section 3.5). Instructional data includes Dolly v2 (Section 3.5).

- Model/training setup (Section 2.3)
  - Decoder‑only Transformer; parallel attention; SwiGLU; 256k BPE; context length 8192.
  - Optimizer: AdamW; batch size 512; cosine LR with 1,325 warmup steps (pre‑training).
  - Scales: 470M and 2.8B parameters.
  - Infrastructure: TPU v5e; training framework FAX. Cost: 200B tokens take ~4,736 chip‑hours (470M) and ~13,824 (2.8B); 40B‑token cooldown ~1,024 chip‑hours (470M) (Section 2.3).

- Evaluation suite (Table 1; Section 2.2)
  - World knowledge: NQ Open, TriviaQA (0‑shot exact match).
  - Natural language (NL) reasoning: 11 tasks (e.g., BoolQ, HellaSwag, Winogrande; 0‑shot accuracy). Reported as an average across tasks.
  - Code: HumanEval and MBPP (0‑shot pass@1), averaged.
  - Open‑ended generation quality: Dolly‑200 prompts; judged pairwise by `Cohere Command‑R+`; win‑rate is “percent of wins” with instruction‑following and quality rubric (Appendix B, Section 2.2).

- Key recipes compared (Table 2; Section 3.1)
  - `text‑only`: train 400B tokens on text only.
  - `balanced‑only`: train 400B tokens on 50% code + 50% text.
  - `balanced → text`: initialize from 200B tokens of balanced training (100B code + 100B text), then continue 200B tokens mainly on text but keep ~10% code to avoid abrupt distribution shift (footnote 5), then (optionally) cooldown.
  - `code → text`: initialize from 200B tokens of code‑heavy training (80% code + 20% markup), then continue 200B tokens mainly on text with ~10% code, then (optionally) cooldown.

Why these choices?
- Isolating effects requires strict control: identical token budgets per phase, consistent architectures, and targeted changes (Figure 1). For example, separating markup or code‑adjacent data reveals which “kinds of code” matter (Section 3.4). Using an explicit cooldown test distinguishes “late‑stage” benefits (Section 3.5).

## 4. Key Insights and Innovations
1) Code helps non‑code tasks—consistently and substantially.
- What’s new: An exhaustive, controlled demonstration across multiple axes (initialization, proportion, quality, schedule, scale). Prior work was narrower (e.g., only data‑limited scaling or math).
- Why it matters: The best variant with code achieves +8.2% NL reasoning, +4.2% world knowledge, +6.6% win‑rate, and 12× code gains compared to text‑only (Section 3.6; Table 2).
- Evidence: Figure 2 (initialization), Figure 6–7 (cooldown), Table 2 (final comparisons).

2) There is an optimal code proportion for general tasks.
- Novelty: A clean proportion sweep shows NL reasoning peaks around 25% code; world knowledge degrades with high code shares; code performance increases almost linearly with code share (Section 3.3; Figure 4).
- Significance: This gives actionable guidance: for general models, ~25% code during from‑scratch pre‑training seems best; pushing code too high hurts knowledge tasks.

3) Code quality matters more than quantity; small, high‑quality synthetic code is especially potent.
- Novelty: Comparing web code vs. markup vs. code‑adjacent vs. synthetic code shows only the synthetic set improves both NL reasoning and code—despite being just 10% of the code stream (3.2B tokens) (Section 3.4; Figure 5).
- Significance: Better code beats more code. Synthetic code yields +9% NL reasoning and +44.9% code gains over web‑code‑only when training code‑only models (Figure 5a). Its benefits transfer to continued pre‑training too (+2% NL reasoning, +35% code vs. the same recipe without synthetic; Figure 5b).

4) Code in the cooldown stage boosts both generalization and judged quality.
- Novelty: Testing cooldown with and without code—holding the cooldown budget constant—shows code in cooldown improves NL reasoning (+3.6%), world knowledge (+10.1%), and code (+20%) vs. pre‑cooldown; cooldown without code helps only world knowledge (Section 3.5; Figure 6).
- Significance: Also delivers the best win‑rates in pairwise judged generation:
  > “Cooldown with code beats the baseline (model without cooldown) by 52.3% win‑rates, where win‑rates are 4.1% higher compared to cooldown without code.” (Section 3.5; Figure 7)

5) Trends hold when scaling from 470M to 2.8B, but trade‑offs intensify.
- Novelty: Replicates the main patterns at 2.8B (Figure 3).
- Significance: Scaling substantially boosts absolute performance—world knowledge and code nearly triple—yet the trade‑off between NL tasks and code generation gets sharper at larger scale (Section 3.2).

## 5. Experimental Analysis
- Evaluation setup (Section 2.2; Table 1)
  - NL reasoning: 11 tasks averaged (e.g., BoolQ, HellaSwag, Winogrande).
  - World knowledge: NQ Open, TriviaQA (exact match).
  - Code: HumanEval + MBPP (pass@1).
  - Open‑ended generations: Dolly‑200 prompts; `Command‑R+` LLM judge; pairwise win‑rates using a fixed rubric (Appendix B).

- Main results and comparisons
  - Initialization matters (Figure 2; Section 3.1)
    - `code → text` and `balanced → text` both outperform `text‑only` on NL reasoning by +8.8% and +8.2% relative gains, respectively. On world knowledge, `balanced → text` is best (+4.1% vs `text‑only`, and +21% vs `code → text`).
    - Code generation is best for `balanced‑only` (50% code throughout), but this model underperforms on NL tasks compared to `code → text`/`balanced → text`.
    - Open‑ended win‑rates: both code‑infused variants beat text‑only by ~+6.6% (Appendix C.1; Figure 9).

  - Scale up to 2.8B (Figure 3; Section 3.2)
    - Average improvements vs. 470M: ~+30–33% across categories; world knowledge and code nearly triple.
    - Relative patterns persist: `code → text` and `balanced → text` beat `balanced‑only` on NL tasks; but now they lag more on code generation (−43.1% and −46.3% relative drops vs `balanced‑only`).

  - Proportion sweep (Figure 4; Section 3.3)
    - NL reasoning: rises from 0%→25% code (+3.4% vs no code), stays flat until ~75%, then sharply drops at 100% code (−18.3% vs 0% code).
    - World knowledge: monotonically harmed by more code: −3.4% at 25%, −31% at 75%, −86% at 100% (vs 0% code).
    - Code performance: almost linear gains with code share; 100% vs 25% code gives +2.6× pass@1.

  - Code quality ablations (Figure 5; Section 3.4)
    - Web code + markup (+20% markup) or + code‑adjacent (+15% mixed) slightly help NL reasoning but hurt code pass@1 (−15.7% and −9.4% vs web‑code‑only).
    - Adding 10% synthetic code (small, verified Python set) during code‑only pre‑training yields +9% NL reasoning and +44.9% code vs web‑code‑only.
    - Transferring synthetic to continued pre‑training (`balanced+synth → text`) adds +2% NL reasoning and +35% code vs `balanced → text` (Figure 5b).

  - Cooldown (Figures 6–7; Section 3.5)
    - With code in cooldown vs no cooldown: +3.6% NL reasoning, +10.1% world knowledge, +20% code.
    - Without code in cooldown vs no cooldown: no gains in NL reasoning or code; +3.1% world knowledge.
    - LLM‑judge win‑rates: both cooldown variants strongly beat no‑cooldown; the best is cooldown with code:
      > “Win‑rates of 52.3% over a model with no cooldown,” and “4.1% higher compared to cooldown without code.” (Figure 7).

  - Full recipe comparison (Table 2; Section 3.6; Figure 8 for win‑rates with cooldown)
    - Best overall for NL tasks and judged quality: `balanced → text` + cooldown with code. It attains the highest judged win‑rate (37.7%) against `text‑only + cooldown` (35.7%) and `balanced‑only + cooldown` (32.7%) (Figure 8).
    - Best for code generation: `balanced‑only` (code‑heavy throughout). But it trails in NL tasks and judged quality versus the `balanced → text` recipe (Table 2; Figure 8).

- Do the experiments support the claims?
  - Strengths
    - Breadth and control: 64 training runs; consistent architectures; fixed token budgets per phase; orthogonal ablations (Figure 1). Clear treatment of when/what code is used.
    - Replication across scales: patterns persist at 470M and 2.8B (Figure 3).
    - Multiple evaluation modes: discriminative (accuracy/EM/pass@1) and generative LLM‑judge win‑rates (Table 1; Figures 7–9).
  - Caveats
    - Judge model and prompt design can bias win‑rates (Section 2.2; Appendix B). The judge is `Command‑R+` and prompts are Dolly‑200; results might vary with other judges/tasks.
    - The synthetic code set is proprietary and Python‑only (Section 2.1), so generality across languages is not directly tested.

- Trade‑offs and conditions
  - More code helps code tasks nearly linearly but hurts world knowledge beyond ~25% (Figure 4).
  - Initialization with code helps NL reasoning most; world knowledge prefers a balanced init then more text in continued pre‑training (Figure 2).
  - Cooldown with code improves everything—especially judged generation—while cooldown without code mainly helps world knowledge (Figure 6–7).

## 6. Limitations and Trade-offs
- Assumptions and design choices
  - The “best mix” is measured on specific benchmarks (Table 1) and a particular cooldown recipe (Section 3.5). Other domains (e.g., safety, multilinguality) are not studied (Section 6).
  - Synthetic code is limited to Python and proprietary; its exceptional gains may partly reflect alignment with the code evaluation tasks (HumanEval/MBPP are Python; Section 2.1, 3.4).
- Scope constraints
  - Model sizes stop at 2.8B for cost reasons (Section 6). Most frontier models are much larger; while trends replicated from 470M→2.8B (Figure 3), exact optima might shift at bigger scales.
  - LLM‑as‑a‑judge uses a single judge (Command‑R+) and 200 prompts (Dolly‑200). Though common, this proxy differs from diverse human ratings (Section 2.2).
- Computational cost
  - Each 200B‑token run is expensive (e.g., 13,824 TPU‑chip‑hours at 2.8B; Section 2.3), which limits broader sweeps (e.g., more code proportions at 2.8B, or different cooldown mixes).
- Open questions
  - Safety impacts of code during pre‑training are not assessed (Section 6).
  - Interaction between code and alignment stages beyond cooldown (e.g., RLHF) is not explored.

## 7. Implications and Future Directions
- How this changes practice
  - Include code—but not blindly. For general‑purpose models, target ~25% code in from‑scratch pre‑training (Figure 4), then reduce code share during continued pre‑training, and explicitly include some code in cooldown (Figures 2, 6–7; Table 2).
  - Invest in code quality. Small amounts of high‑quality synthetic code deliver outsized gains (+9% NL reasoning, +44.9% code in code‑only; +2% and +35% in continued pre‑training; Figure 5). Curation can trump sheer volume.
  - Expect trade‑offs. If code generation is the priority, sustain a high code share (e.g., `balanced‑only`). If broad NL tasks and judged quality matter, prefer `balanced → text` with code in cooldown (Table 2; Figure 8).

- Research directions
  - Scaling studies: verify the 25% “sweet spot” and synthetic code effect at 7B–70B scales; test whether trade‑offs sharpen with size (Section 3.2 suggests they do).
  - Smarter schedules: dynamic curricula that change the code share over time; optimize “when” code appears (early vs late) beyond the fixed phases tested here.
  - Broader code quality: expand synthetic sets beyond Python; test formal verification for other languages; quantify how markup, code‑adjacent data, and language diversity impact NL tasks vs. code.
  - Judge diversity: replicate win‑rates with multiple judges and larger prompt suites; evaluate human correlation for the specific tasks used here.
  - Safety and robustness: study how code pre‑training affects safety behaviors, hallucination under constrained reasoning, and robustness to adversarial prompts.

- Applications
  - Foundation model training: data‑mix design for companies/communities building general LLMs.
  - Education and tutoring agents: code‑infused pre‑training may boost step‑by‑step reasoning.
  - Software agents: for models that must reason and code, recipes that blend balanced pre‑training, text‑heavy continuation, and code in cooldown can deliver strong general performance without sacrificing judged quality (Table 2; Figure 8).

> Anchor findings to remember:
> - “Compared to text‑only pre‑training, for our best variant, the addition of code results in relative increase of 8.2% in natural language reasoning, 4.2% in world knowledge, 6.6% in generative win‑rates, and a 12× boost in code performance.” (Section 3.6; Table 2)
> - “Including code in cooldown leads to +3.6% NL reasoning, +10.1% world knowledge, +20% code vs. pre‑cooldown, and a +4.1% judged win‑rate increase over cooldown without code.” (Section 3.5; Figures 6–7)
> - “NL reasoning peaks around 25% code; world knowledge drops sharply as code rises; code metrics rise almost linearly with code proportion.” (Section 3.3; Figure 4)
> - “High‑quality synthetic code—even at 10% of the code stream—outperforms much larger web or code‑adjacent sources for both NL and code.” (Section 3.4; Figure 5)
