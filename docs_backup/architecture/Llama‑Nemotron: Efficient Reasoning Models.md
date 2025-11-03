# Llama‑Nemotron: Efficient Reasoning Models

**ArXiv:** [2505.00949](https://arxiv.org/abs/2505.00949)
**Authors:** Akhiad Bercovich, Itay Levy, Izik Golan, Mohammad Dabbah, Ran El‑Yaniv, Omri Puny, Ido Galil, Zach Moshe, Tomer Ronen, Najeeb Nabwani, Ido Shahaf, Oren Tropp, Ehud Karpas, Ran Zilberstein, Jiaqi Zeng, Soumye Singhal, Alexander Bukharin, Yian Zhang, Tugrul Konuk, Gerald Shen, Ameya Sunil Mahabaleshwarkar, Bilal Kartal, Yoshi Suhara, Olivier Delalleau, Zijia Chen, Zhilin Wang, David Mosallanezhad, Adi Renduchintala, Haifeng Qian, Dima Rekesh, Fei Jia, Somshubra Majumdar, Vahid Noroozi, Wasi Uddin Ahmad, Sean Narenthiran, Aleksander Ficek, Mehrzad Samadi, Jocelyn Huang, Siddhartha Jain, Igor Gitman, Ivan Moshkov, Wei Du, Shubham Toshniwal, George Armstrong, Branislav Kisacanin, Matvei Novikov, Daria Gitman, Evelina Bakhturina, Jane Polak Scowcroft, John Kamalu, Dan Su, Kezhi Kong, Markus Kliegl, Rabeeh Karimi, Ying Lin, Sanjeev Satheesh, Jupinder Parmar, Pritam Gundecha, Brandon Norick, Joseph Jennings, Shrimai Prabhumoye, Syeda Nahida Akter, Mostofa Patwary, Abhinav Khattar, Deepak Narayanan, Roger Waleffe, Jimmy Zhang, Bor‑Yiing Su, Guyue Huang, Terry Kong, Parth Chadha, Sahil Jain, Christine Harvey, Elad Segal, Jining Huang, Sergey Kashirsky, Robert McQueen, Izzy Putterman, George Lam, Arun Venkatesan, Sherry Wu, Vinh Nguyen, Manoj Kilaru, Andrew Wang, Anna Warno, Abhilash Somasamudramath, Sandip Bhaskar, Maka Dong, Nave Assaf, Shahar Mor, Omer Ullman Argov, Scot Junkin, Oleksandr Romanenko, Pedro Larroy, Monika Katariya, Marco Rovinelli, Viji Balas, Nicholas Edelman, Anahita Bhiwandiwalla, Muthu Subramaniam, Smita Ithape, Karthik Ramamoorthy, Yuting Wu, Suguna Varshini Velury, Omri Almog, Joyjit Daw, Denys Fridman, Erick Galinkin, Michael Evans, Katherine Luna, Leon Derczynski, Nikki Pope, Eileen Long, Seth Schneider, Guillermo Siman, Tomasz Grzegorzek, Pablo Ribalta, Monika Katariya, Joey Conway, Trisha Saar, Ann Guan, Krzysztof Pawelec, Shyamala Prayaga, Oleksii Kuchaiev, Boris Ginsburg, Oluwatobi Olabiyi, Kari Briski, Jonathan Cohen, Bryan Catanzaro, Jonah Alben, Yonatan Geifman, Eric Chung
**Institutions:** 

## 🎯 Pitch

Introducing the Llama-Nemotron series, this research establishes novel open-weight reasoning models with a unique toggle for user-controlled reasoning depth, maximizing inference efficiency and reasoning quality. These models, employing hardware-aware architecture search and vertical compression, significantly enhance real-world deployability and throughput, crucial for handling dynamic workloads in high-concurrency environments, thus paving the way for more accessible and efficient reasoning-driven applications.

---

## 1. Executive Summary
This paper introduces the Llama‑Nemotron family (`LN‑Nano` 8B, `LN‑Super` 49B, `LN‑Ultra` 253B), open‑weight reasoning models that pair state‑of‑the‑art reasoning quality with high inference efficiency and a user‑controlled reasoning toggle (`"detailed thinking on/off"`). The technical core is a hardware‑aware neural architecture search plus vertical compression that make very large models fast to serve, followed by supervised reasoning fine‑tuning and large‑scale reinforcement learning (RL) that lets the top model surpass its teacher on scientific reasoning benchmarks.

## 2. Context and Motivation
- Problem addressed
  - Reasoning‑optimized LLMs (e.g., OpenAI o1, DeepSeek‑R1) achieve strong results by generating long chains of thought, but they are expensive to serve and often require specific, high‑end hardware; they also lack simple user control over when to reason versus answer tersely. Section 1 frames inference efficiency as a new bottleneck for “overall model intelligence” because modern systems scale at inference time, not only at training time.
- Why it matters
  - Real deployments need to handle many concurrent users, tools, and agentic pipelines. Without high throughput and memory efficiency, multi‑step reasoning becomes impractical. Moreover, not every query benefits from long reasoning; wasted reasoning costs money and time and can reduce usability (Section 1).
- Where prior approaches fall short
  - State‑of‑the‑art open reasoning models like DeepSeek‑R1 run best on 8×H200 and do not give the end user a built‑in way to switch between terse and chain‑of‑thought modes. Traditional architecture compression is usually uniform and can degrade performance; prior NAS approaches rarely optimize for real‑world deployment constraints with per‑layer heterogeneity (Section 2).
- Positioning
  - This work combines: (i) deployment‑constrained NAS (“Puzzle”) that creates a heterogeneous transformer with block‑level attention removals and FFN compression, (ii) a new vertical compression (“FFN Fusion”), and (iii) a two‑stage post‑training program (reasoning SFT + large‑scale RL). It releases models, data, and code under permissive licenses (Abstract; release bullets).

## 3. Technical Approach
Step‑by‑step pipeline across five stages (Sections 2–6):

1) Making the base models inference‑efficient with Puzzle NAS (Section 2; Figure 3)
- What Puzzle is
  - `Puzzle` is a neural architecture search framework that builds a “library” of alternative transformer blocks and then assembles one variant per layer subject to deployment constraints.
  - Each candidate block is trained locally to mimic its parent block (block‑wise distillation) and is profiled for quality vs. cost.
- What block variants are used here
  - Attention removal in selected layers to reduce both compute and KV‑cache memory.
  - Variable FFN intermediate sizes (e.g., 87%, 75%, 50%, down to 10%) to trade accuracy for speed/memory (Section 2; bullet list).
- How the final architecture is chosen
  - A mixed‑integer programming (MIP) solver selects one variant per layer to optimize quality under constraints such as throughput, latency, memory, or batch×sequence (“cached tokens”) (Figure 3 and Section 2).
- Deployment targets and measured efficiency
  - `LN‑Super` (49B): optimized for a single H100 at tensor parallel 1 (TP1). It achieves “5× throughput speedup over Llama‑3.3‑70B‑Instruct at batch 256 and TP1,” and still ≥2.17× over Llama‑3.3‑70B run at its optimal TP4 (Section 2.1).
  - `LN‑Ultra` (253B): optimized for one 8×H100 node; NAS enforces at least 1.5× latency reduction vs. Llama‑3.1‑405B‑Instruct, and the final model realizes 1.71× after FFN Fusion (Section 2.1).

2) Vertical compression with FFN Fusion (Section 2; “Vertical Compression with FFN Fusion”)
- Idea
  - After some attention layers are removed by Puzzle, multiple FFN blocks can become consecutive. `FFN Fusion` replaces sequences of consecutive FFNs with fewer, wider FFNs that can be executed in parallel. This reduces the number of sequential steps (model depth along the compute graph), which lowers latency without reducing expressivity.
- Why it matters
  - Lower sequential depth improves utilization, especially in multi‑GPU pipelines where inter‑layer communication is costly.

3) Recovery training: knowledge distillation and continued pretraining (Section 2.2; Table 1)
- Purpose
  - NAS changes layer internals; this stage recovers quality and improves inter‑block compatibility.
- Details
  - `LN‑Super`: 40B tokens of distillation on the Distillation Mix dataset from Bercovich et al. (2024).
  - `LN‑Ultra`: 65B tokens of distillation + 88B tokens of continued pretraining (CPT) on Nemotron‑H Phase 4 (NVIDIA et al., 2025).
- Effect before SFT/RL (Table 1)
  - `LN‑Ultra‑CPT` exceeds Llama‑3.1‑405B‑Instruct on MATH500 (80.4 vs. 69.6) and RULER‑128K (83.2 vs. 73.7) and roughly ties on MMLU (88.1 vs. 88.6), showing that aggressive architecture changes can be reconciled with strong base quality via short CPT.

4) Reasoning‑focused supervised fine‑tuning (SFT) with a reasoning toggle (Sections 3–4)
- Reasoning toggle
  - A simple system instruction `“detailed thinking on/off”` teaches the model to emit chain‑of‑thought between `<think>...</think>` tags or to answer tersely. Paired data is created so every prompt has both a reasoning and a non‑reasoning response (Section 3.2).
- Data construction (Section 3; Table 2)
  - Math: Problems from AoPS; remove proofs/MCQ/binary/invalid; generate many candidate solutions (DeepSeek‑R1 for reasoning, Qwen2.5‑Math‑7B for non‑reasoning), filter by answer match using Qwen2.5‑32B judge; perform benchmark decontamination (Section 3.1.1).
  - Code: 28,904 competitive programming problems from TACO, APPS, CodeContests, CodeForces; decontamination and deduplication; multi‑sample solutions with explicit reasoning in `<think>`; syntax checks via Tree‑Sitter; ~488K Python samples; scaling experiments show more, harder data keeps improving results (Section 3.1.2).
  - Science and general: synthetic MCQs and open‑ended prompts with decontamination; responses by strong teachers (DeepSeek‑R1) plus rejection sampling with `Llama‑3.1‑Nemotron‑70B‑Reward`; also a Feedback‑Edit inference‑time scaling pipeline for high‑quality general responses (Section 3.1.3–3.1.4 and 3.2.1).
  - Overall size: 33,011,757 samples; 66.8% math, 30.6% code, 2.1% science, small chat/instruction/safety (Table 2).
- SFT training recipes (Section 4.2)
  - `LN‑Nano`: three stages; start only on reasoning to avoid degenerate repetition; later mix in non‑reasoning; final blend adds chat/instruction/tool use.
  - `LN‑Super`: one epoch over the full SFT set; sequence length 16k; fixed LR 5e‑6 (smaller runs suggest up to 3–4 epochs and higher LR can help).
  - `LN‑Ultra`: sequence packing to 24k effective length; larger LR helps but caused instabilities; they use linear warmup to 1e‑5 then cosine to 1e‑6; training suffered gradient explosions after the first epoch and required optimizer re‑init to continue.

5) Large‑scale RL to go beyond the teacher (Sections 5–6)
- Why RL is needed
  - Distillation is bounded by the teacher’s quality; to surpass DeepSeek‑R1, `LN‑Ultra` uses RL (Section 5).
- Algorithm and rollout setup (Section 5.1; Figure 5)
  - GRPO (`Group Relative Policy Optimization`): a policy‑gradient method using groupwise baselines.
  - Rollout prompt size 72, 16 samples per prompt at temperature=1, top_p=1; global batch 576; 2 gradient updates per rollout. Training consumes ~140k H100 hours.
- Rewards (Section 5.1)
  - Accuracy reward: a served `Llama‑3.3‑70B‑Instruct` judges whether the prediction matches the ground truth answer (numbers, sentences, or paragraphs).
  - Format reward: enforces `<think>` tags when reasoning is on and their absence when off.
- Data difficulty and curriculum (Section 5.1; Figure 6)
  - Pre‑filter prompts using `LN‑Super` pass‑rates; drop those with pass‑rate ≥0.75.
  - Curriculum: batches gradually shift from easy (high pass‑rate) to hard (low pass‑rate) using a Gaussian target distribution per batch.
- Infrastructure to make RL feasible (Section 5.2)
  - Co‑locate generation (vLLM) and training (Megatron‑LM) on the same GPUs; maintain separate weight copies and synchronize each step.
  - Parallelism: tensor=8 with sequence parallel, context=2, pipeline=18, data=2 for training; tensor=8, data=72 for generation across 72×(8×H100) nodes.
  - FP8 online generation path in vLLM with custom loaders and meta‑initialization to avoid materializing BF16 engines; delivers “32 tokens/s/GPU/prompt,” a “1.8× generation speedup” and enables cudagraph thanks to lower memory (Section 5.2.3).
  - Careful memory profiling for GPU/CPU and `/dev/shm` to avoid OOMs; identity layers inserted to balance heterogeneous pipelines (Section 5.2.2).

6) Final alignment via preference optimization (Section 6)
- Instruction following: short RL using `RLOO` (Leave‑One‑Out variant for RL from feedback) on synthetic multi‑constraint prompts, which boosts IFEval and also helps reasoning benchmarks (Section 6.1).
- RLHF with `RPO` (Reward‑aware Preference Optimization): iterative online RPO against `Llama‑3.1‑Nemotron‑70B‑Reward` on HelpSteer2. For `LN‑Super`, two iterations raise Arena‑Hard from 69.1 to 88.1 and also improve most other benchmarks (Section 6.2). `LN‑Ultra` uses GRPO for this stage with 8 samples per prompt for 30 steps.

Definitions of select terms used above
- `KV‑cache`: the saved key/value tensors used by attention to avoid recomputing past context.
- `Sequence packing`: packing multiple shorter training samples into contiguous segments of a long sequence to improve hardware utilization.
- `Context parallel`/`pipeline parallel`/`tensor parallel`: ways to split model computation across GPUs along sequence length, layers, and weight tensors, respectively.
- `FP8`: 8‑bit floating point precision; faster and lower memory than BF16/FP16 for GEMMs.

## 4. Key Insights and Innovations
- Hardware‑constrained, heterogeneous NAS for LLM inference
  - What’s new: Instead of uniformly shrinking the whole model, Puzzle builds a per‑layer menu with options like “remove attention” or “use a smaller FFN,” then solves a constrained selection problem with a MIP solver (Figure 3). This lets the final architecture sit precisely on a desired throughput/latency/memory point (Section 2).
  - Why it matters: Concrete efficiency wins under real serving constraints. `LN‑Super` yields up to 5× throughput on a single H100 vs. Llama‑3.3‑70B‑Instruct at TP1 (Section 2.1).
- FFN Fusion: vertical compression that speeds multi‑GPU pipelines
  - What’s new: Detect runs of FFN‑only layers that appear after some attention removal and fuse them into fewer, wider FFNs that execute in parallel (Section 2; “Vertical Compression with FFN Fusion”).
  - Why it matters: Lowers sequential critical path; the `LN‑Ultra` model achieves a 1.71× latency improvement vs. Llama‑3.1‑405B‑Instruct after applying Fusion (Section 2.1).
- A simple, effective reasoning toggle with format‑aware rewards
  - What’s new: The same model can switch between terse and chain‑of‑thought styles via a 1‑line system prompt. Training uses paired data and a format reward to make the control reliable (Sections 3.2 and 5.1).
  - Why it matters: Users spend compute only when they want reasoning; deployment teams can mix workloads without separate models.
- RL at scale with FP8 online generation to surpass the teacher
  - What’s new: A GRPO training pipeline that co‑locates vLLM generation and Megatron training, adds an FP8 decoding path with custom weight loaders and cudagraph support, and uses pass‑rate‑based curriculum (Sections 5.1–5.2).
  - Why it matters: Enables `LN‑Ultra` to exceed DeepSeek‑R1 on GPQA‑Diamond while running on 8×H100 instead of 8×H200 (Figure 4; Table 5).

## 5. Experimental Analysis
- Evaluation setup (Section 7.1)
  - Benchmarks
    - Reasoning: `AIME24`, `AIME25` (competition math), `GPQA‑Diamond` (graduate‑level science MCQ), `MATH500` (step‑by‑step math), `LiveCodeBench` (coding).
    - Non‑reasoning: `IFEval` (strict instruction following), `BFCL V2 Live` (tool/function calling), `Arena‑Hard` (pairwise conversational preference).
  - Decoding and context
    - All results use 32k context at eval time (even though SFT used 16k/24k), because longer context avoids truncating long reasoning (Section 7.1).
    - Reasoning‑on uses temperature 0.6, top‑p 0.95; reasoning‑off is greedy; up to 16 completions; report pass@1 (Section 7.1). AIME has high variance; numbers can vary with sampling.
  - Decontamination and data quality controls are described in Section 3 for math, code, and science.
- Main results
  - Top‑line accuracy and efficiency
    - Figure 4 plots GPQA‑Diamond accuracy vs. throughput (tokens/s) in two concurrency settings with FP8 serving; `LN‑Ultra` dominates both DeepSeek‑R1 and Llama‑3.1‑405B on the Pareto curve. Quoted points: improvements of “1.9×” and “4×” throughput depending on the setting.
  - `LN‑Ultra` vs open SOTA (Table 5, reasoning‑on)
    - GPQA‑Diamond: 76.0 vs DeepSeek‑R1 71.5; vs Llama‑4 Maverick 69.8; vs Llama‑3.1‑405B 43.4.
    - AIME24: 80.8 vs DeepSeek‑R1 79.8.
    - AIME25: 72.5 vs DeepSeek‑R1 70.0.
    - MATH500: 97.0 vs DeepSeek‑R1 97.3 (essentially tied).
    - LiveCodeBench (2408–2502): 66.3 vs DeepSeek‑R1 65.9.
    - IFEval: 88.9 vs DeepSeek‑R1 88.8 (parity).
    - Arena‑Hard: 87.0 (DeepSeek‑R1 at 92.0 is higher here).
  - Effect of RL (Table 5)
    - `LN‑Ultra‑SFT` scores 66.4 on GPQA‑D; RL lifts it to 76.0, crossing the teacher’s 71.5. This directly supports the claim that RL is necessary to surpass the teacher on scientific reasoning.
  - `LN‑Super` (49B) trade‑offs (Table 4)
    - Reasoning‑on GPQA‑D: 66.7 vs DeepSeek‑R1‑Distilled‑Llama‑70B at 65.2; AIME25: 60.0 vs 55.0; MATH500: 96.6 vs 94.5.
    - Instruction following and chat: After a dedicated IFEval RL run and subsequent preference optimization, IFEval reaches 89.2 (on/off similar); Arena‑Hard hits 88.3, beating several larger proprietary and open models listed in Section 6.2.
    - Coding: LCB (2408–2502) 45.5; the paper attributes underperformance to training on an earlier dataset version and plans a refresh (Section 7.3).
  - `LN‑Nano` (8B) (Table 3)
    - Outperforms comparable 7–8B baselines on many reasoning tasks, e.g., MATH500 95.4 vs Qwen‑7B 92.8, LiveCodeBench 46.6 vs Llama‑3.1‑8B‑Instruct 37.6; function calling (BFCL V2 Live) ~64, far ahead of Qwen‑7B’s 39.2.
- Additional evaluations: LLM‑as‑a‑judge (Table 6)
  - On JudgeBench, `LN‑Ultra` overall 79.14 surpasses DeepSeek‑R1 73.14 and trails only `o3‑mini(high)` 80.86. `LN‑Super` 69.71 exceeds `o1‑mini` 65.71. This suggests generalization to judgment tasks outside the training targets.
- Ablations and diagnostics
  - Curriculum helps: Figure 6 shows curriculum‑driven batching yields higher GPQA‑D than random sampling across training steps.
  - Training stability: Section 4.2 notes gradient explosions for `LN‑Ultra` during SFT; resuming with reinitialized optimizer states was necessary.
  - Data scaling for code: Section 3.1.2 reports continued benefits up to ~736k samples and especially from focusing first on harder CodeContests problems.
- Do the experiments support the claims?
  - Yes on three axes:
    - Accuracy: Tables 4–5 show strong or SOTA open‑model results in reasoning, with RL pushing `LN‑Ultra` beyond DeepSeek‑R1 on GPQA.
    - Efficiency: Section 2.1 and Figure 4 quantify large throughput/latency gains under realistic serving constraints on H100s.
    - Control: Tables split results by reasoning on/off, and Section 5.1 includes a format reward that ensures cleanly separated modes.

## 6. Limitations and Trade-offs
- Heavy reliance on synthetic/teacher‑generated data
  - Many SFT samples are distilled from strong closed/open models (DeepSeek‑R1, Qwen2.5). Although decontamination is performed (Section 3), this can import biases or errors from teachers. The accuracy reward during RL is judged by a `Llama‑3.3‑70B‑Instruct` model rather than solely by programmatic verification, which can be imperfect for open‑ended answers (Section 5.1).
- RL only for the largest model
  - The paper finds smaller models benefit less from RL and therefore applies reasoning RL only to `LN‑Ultra` (Section 5). This leaves open whether improved or cheaper RL variants could help `LN‑Super` or `LN‑Nano`.
- Compute and system complexity
  - The reasoning RL run consumed about 140k H100 hours (Section 5.1). The heterogeneous architecture, identity layers for balancing, and FP8 generation path add engineering complexity (Section 5.2).
- Hardware‑specific gains
  - Efficiency results are measured with FP8 serving on NVIDIA H100 nodes and specific parallelism settings (Figure 4; Section 5.2.3). Gains may not transfer directly to other accelerators or software stacks.
- Trade‑offs between skills
  - Section 7.3 reports a tension between instruction following (IFEval) and conversational preference (Arena‑Hard). Optimizing one can degrade the other; model merging was required to find a Pareto point for `LN‑Super`.
- Coding lag for `LN‑Super`
  - LiveCodeBench performance trails some contemporaries due to training on an earlier dataset version (Section 7.3), highlighting sensitivity to up‑to‑date code data.
- Mode control via prompting
  - The reasoning toggle depends on a system prompt string. Although a format reward reinforces behavior, mis‑prompting or adversarial inputs could still elicit unintended reasoning traces; the paper does not present a robustness audit of the toggle.

## 7. Implications and Future Directions
- How this changes the landscape
  - It demonstrates that open‑weight reasoning models can be both fast and strong, with a simple user‑visible switch for reasoning style. By releasing models, post‑training data, and training code (Abstract bullets), it lowers the barrier for research into efficiency‑aware reasoning systems and for enterprises that need commercial terms.
- Follow‑up research enabled
  - Extending GRPO‑based reasoning RL to smaller models or multi‑modal models; exploring cheaper verifiable rewards (e.g., programmatic checkers beyond science/math); improving reward models to reduce judge bias in open‑ended tasks; principled scheduling between SFT and RL checkpoints to maximize RL success (Section 7.4 hints earlier SFT checkpoints may be better RL initializations).
  - Automated or learned “reasoning policy”: rather than a manual `on/off` prompt, the model could decide when depth is warranted based on budget, latency targets, or uncertainty.
  - NAS beyond attention/FFN: integrate grouped‑query attention and linear attention blocks more aggressively, or search over routing and mixture‑of‑experts under deployment constraints (Section 2 notes Puzzle supports additional operations not used here).
  - Broader efficiency features: generalize the FP8 online generation path to other inference engines and accelerators; study accuracy‑latency trade‑offs of FP8 across domains.
- Practical applications
  - Production assistants that keep most turns terse but switch to chain‑of‑thought when asked to justify or solve hard problems; high‑concurrency agentic systems that need predictable latency and memory; enterprise function‑calling and workflow orchestration (`BFCL V2 Live` scores in Tables 3–5); education/tutoring where step‑by‑step thinking is valuable but controllable.

> Representative headline result: “LN‑Ultra ... 76.0% on GPQA‑Diamond while offering higher throughput than DeepSeek‑R1 and fitting on a single 8×H100 node” (Figure 4; Table 5).

Overall, the paper’s main technical message is that deployment‑aware architectural search plus targeted vertical compression, followed by carefully staged reasoning SFT and scalable RL with efficient FP8 generation, yields open models that are both strong reasoners and practical to serve—while giving users explicit control over when to spend the extra compute on long chains of thought.
