# NVIDIA Nemotron Nano 2: An Accurate and Efficient Hybrid Mamba‑Transformer Reasoning Model

**ArXiv:** [2508.14444](https://arxiv.org/abs/2508.14444)
**Authors:** Aarti Basant, Abhijit Khairnar, Abhijit Paithankar, Abhinav Khattar, Adi Renduchintala, Adithya Renduchintala, Aditya Malte, Akhiad Bercovich, Akshay Hazare, Alejandra Rico, Aleksander Ficek, Alex Kondratenko, Alex Shaposhnikov, Ali Taghibakhshi, Amelia Barton, Ameya Sunil Mahabaleshwarkar, Amy Shen, Andrew Tao, Ann Guan, Anna Shors, Anubhav Mandarwal, Arham Mehta, Arun Venkatesan, Ashton Sharabiani, Ashwath Aithal, Ashwin Poojary, Ayush Dattagupta, Balaram Buddharaju, Banghua Zhu, Barnaby Simkin, Bilal Kartal, Bita Darvish Rouhani, Bobby Chen, Boris Ginsburg, Brandon Norick, Brian Yu, Bryan Catanzaro, Charles Wang, Charlie Truong, Chetan Mungekar, Chintan Patel, Chris Alexiuk, Christian Munley, Christopher Parisien, Dan Su, Daniel Afrimi, Daniel Korzekwa, Daniel Rohrer, Daria Gitman, David Mosallanezhad, Deepak Narayanan, Dima Rekesh, Dina Yared, Dmytro Pykhtar, Dong Ahn, Duncan Riach, Eileen Long, Elliott Ning, Eric Chung, Erick Galinkin, Evelina Bakhturina, Gargi Prasad, Gerald Shen, Haim Elisha, Harsh Sharma, Hayley Ross, Helen Ngo, Herman Sahota, Hexin Wang, Hoo Chang Shin, Hua Huang, Iain Cunningham, Igor Gitman, Ivan Moshkov, Jaehun Jung, Jan Kautz, Jane Polak Scowcroft, Jared Casper, Jimmy Zhang, Jinze Xue, Jocelyn Huang, Joey Conway, John Kamalu, Jonathan Cohen, Joseph Jennings, Julien Veron Vialard, Junkeun Yi, Jupinder Parmar, Kari Briski, Katherine Cheung, Katherine Luna, Keith Wyss, Keshav Santhanam, Kezhi Kong, Krzysztof Pawelec, Kumar Anik, Kunlun Li, Kushan Ahmadian, Lawrence McAfee, Laya Sleiman, Leon Derczynski, Luis Vega, Maer Rodrigues de Melo, Makesh Narsimhan Sreedhar, Marcin Chochowski, Mark Cai, Markus Kliegl, Marta Stepniewska‑Dziubińska, Matvei Novikov, Mehrzad Samadi, Meredith Price, Meriem Boubdir, Michael Boone, Michael Evans, Michal Bien, Michal Zawalski, Miguel Martinez, Mike Chrzanowski, Mohammad Shoeybi, Mostofa Patwary, Namit Dhameja, Nave Assaf, Negar Habibi, Nidhi Bhatia, Nikki Pope, Nima Tajbakhsh, Nirmal Kumar Juluru, Oleg Rybakov, Oleksii Hrinchuk, Oleksii Kuchaiev, Oluwatobi Olabiyi, Pablo Ribalta, Padmavathy Subramanian, Parth Chadha, Pavlo Molchanov, Peter Dykas, Peter Jin, Piotr Bialecki, Piotr Januszewski, Pradeep Thalasta, Prashant Gaikwad, Prasoon Varshney, Pritam Gundecha, Przemek Tredak, Rabeeh Karimi Mahabadi, Rajen Patel, Ran El‑Yaniv, Ranjit Rajan, Ria Cheruvu, Rima Shahbazyan, Ritika Borkar, Ritu Gala, Roger Waleffe, Ruoxi Zhang, Russell J. Hewett, Ryan Prenger, Sahil Jain, Samuel Kriman, Sanjeev Satheesh, Saori Kaji, Sarah Yurick, Saurav Muralidharan, Sean Narenthiran, Seonmyeong Bak, Sepehr Sameni, Seungju Han, Shanmugam Ramasamy, Shaona Ghosh, Sharath Turuvekere Sreenivas, Shelby Thomas, Shizhe Diao, Shreya Gopal, Shrimai Prabhumoye, Shubham Toshniwal, Shuoyang Ding, Siddharth Singh, Siddhartha Jain, Somshubra Majumdar, Stefania Alborghetti, Syeda Nahida Akter, Terry Kong, Tim Moon, Tomasz Hliwiak, Tomer Asida, Tony Wang, Twinkle Vashishth, Tyler Poon, Udi Karpas, Vahid Noroozi, Venkat Srinivasan, Vijay Korthikanti, Vikram Fugro, Vineeth Kalluru, Vitaly Kurin, Vitaly Lavrukhin, Wasi Uddin Ahmad, Wei Du, Wonmin Byeon, Ximing Lu, Xin Dong, Yashaswi Karnati, Yejin Choi, Yian Zhang, Ying Lin, Yonggan Fu, Yoshi Suhara, Zhen Dong, Zhiyu Li, Zhongbo Zhu, Zijia Chen
**Institutions:** NVIDIA

## 🎯 Pitch

Nemotron Nano 2 introduces the `NVIDIA-Nemotron-Nano-9B-v2` model, a 9B-parameter hybrid Mamba-Transformer architecture that achieves state-of-the-art reasoning accuracy with significantly enhanced throughput—up to 6.3× faster—on commodity GPUs. This advancement transforms economic feasibility for long-form applications such as math tutoring and legal document analysis by combining efficient memory use with a streamlined distillation and compression strategy.

---

## 1. Executive Summary (2-3 sentences)
Nemotron Nano 2 introduces `NVIDIA-Nemotron-Nano-9B-v2`, a 9B-parameter hybrid Mamba–Transformer model distilled from a 12B base that keeps state‑of‑the‑art reasoning accuracy while delivering 3×–6× higher inference throughput for long “thinking trace” generations on a single 22 GiB A10G GPU (Figure 1, §4). It achieves this through an architecture with few attention layers, FP8 pretraining on 20T tokens, a 512k long‑context extension, a multi‑stage alignment pipeline (SFT → DPO/GRPO/RLHF), and a pruning‑plus‑distillation compression strategy adapted to memory/throughput constraints (§2–§4).

## 2. Context and Motivation
- Problem addressed
  - Reasoning‑style LLM use (math/code/tool use) often requires generating long chain‑of‑thought (“thinking traces”), which makes standard Transformers slow and memory‑intensive due to attention’s `KV cache` growth with sequence length. The goal is to keep or improve accuracy on difficult reasoning tasks while dramatically increasing throughput and fitting 128k context inference on a 22 GiB GPU (§1, §4.2).
  - `KV cache` (key–value cache) stores attention keys/values for each generated token so later tokens can attend to earlier ones; its memory grows linearly with sequence length and number of attention heads, which becomes a bottleneck for long contexts and long generations.

- Why it matters
  - Real‑world: Faster long‑form reasoning reduces latency and cost for math tutoring, coding assistants, and long‑document QA, enabling deployment on lower‑cost hardware (A10G) (§1, §4.2).
  - Scientific: Tests whether hybrid architectures that replace most attention with `Mamba‑2` (a structured state‑space model layer with linear‑time sequence processing) can keep accuracy while unlocking throughput and context length (§2.1).

- Prior approaches and limitations
  - Pure Transformers maintain strong accuracy but incur large KV caches and quadratic attention costs for long contexts.
  - Prior hybrids (e.g., Jamba; cited in §1) demonstrate feasibility but leave open how to: (a) sustain SoTA reasoning accuracy at small/medium scale, (b) deliver 128k context on 22 GiB, and (c) provide rigorous data/recipe releases.

- Positioning
  - Builds on `Nemotron‑H` hybrid design (few attention layers, many Mamba layers) but introduces new data, FP8 pretraining to 20T tokens, a 512k long‑context extension, alignment with budgeted thinking, and a compression pipeline targeted to the A10G memory budget (§1, §2, §3, §4).

## 3. Technical Approach
This section walks through the full pipeline: architecture → pretraining → long‑context extension → alignment → compression/distillation → budgeted thinking.

- Hybrid architecture (Figure 2, Table 1, §2.1)
  - Layering: 62 total layers with ≈8% attention (6 attention layers evenly dispersed), 28 Mamba‑2 layers, and 28 FFN layers. The small attention fraction preserves some capabilities that benefit from attention (e.g., exact token interactions), while Mamba‑2 handles most sequence modeling at lower memory/compute.
  - Key dims: model hidden 5120; FFN hidden 20480; grouped‑query attention (GQA) with 40 query heads and 8 key‑value heads; Mamba‑2 uses 8 groups, state dim 128, head dim 64, conv window 4. No positional embeddings; RMSNorm; squared ReLU activations (§2.1).
  - Why Mamba‑2? Mamba‑2 (an SSM‑based layer) processes sequences in linear time without caching past token keys/values, reducing memory pressure and improving throughput for long generations (§2.1). Attention is retained sparingly to preserve long‑range token‑wise interactions that SSMs may not fully replace.

- Pretraining data and curriculum (§2.2–§2.3)
  - Data scale and diversity: 20T tokens from curated web (Nemotron‑CC‑v2), code, math (Nemotron‑CC‑Math), multilingual, academic, and synthetic SFT‑style data targeting math/code/general reasoning. Three blend phases progressively emphasize higher‑quality data (Figure 3).
  - Two notable synthetic components:
    - `Fundamental Reasoning` SFT‑style data targeting logical/analytical reading comprehension (LSAT, LogiQA, AQuA‑RAT) improves MMLU‑Pro by +12.1 points in an 8B ablation (Table 3, §2.3.2).
    - Multilingual `DiverseQA` shows in ablation that translating English DiverseQA to many languages (“DiverseQA‑crawl”) yields the best Global‑MMLU scores (Table 2, §2.3.1).
  - Training numerics: FP8 (E4M3) for tensors with FP32 master weights; first/last four linears kept in BF16; optimizer state FP32. They keep weights in FP8 to do distributed all‑gathers in FP8 (§2.4). LR schedule: `Warmup‑Stable‑Decay`, stable LR 4.5e‑4, min 4.5e‑6; seq length 8192; global batch 768; Adam β1=0.9, β2=0.95; weight decay 0.1 (§2.5).

- Long‑context extension to 128k+ (§2.6)
  - After pretraining, continuous pretraining with sequence length 512k (not 128k/256k) using context‑parallelism (8‑way tensor parallel + 16‑way context parallel) and a small global batch to keep token count per batch unchanged. Added ≈18.9B tokens in this phase.
  - Synthetic long‑document QA: chunk academic documents >32k tokens and generate QA pairs, appending to the source document to teach long‑range dependencies. Ablation on an 8B model shows that training at 512k with synthetic long‑doc QA reaches the highest RULER‑128k (81.04) versus 128k/256k setups (Table 4).

- Alignment pipeline (Figure 4, §3)
  - Stage 1 SFT: ≈80B tokens of prompt–response with reasoning traces; 10% of prompts have responses with the trace removed to enable “reasoning‑off” direct‑answer mode. Samples are concatenated up to ~128k to maintain long‑range behaviors (§3.2).
  - Stage 2 SFT: focused on tool calling without concatenation (Stage 1 concatenation hurt learning tool‑calling patterns). Data comes from curated tool‑calling corpora and simulated multi‑turn/multi‑step calls with verification (§3.1, §3.2).
  - Stage 3 SFT: reinforces long‑context + introduces `truncated traces`—reasoning is cut after 1–2k tokens but the final answer remains. This teaches the model to finish cleanly when a thinking budget is exhausted (§3.2, §3.4).
  - Preference/RL phases:
    - IFEval RL: reward is how strictly instructions are followed; improves instruction following but may slightly move other metrics, so checkpoint selection matters (§3.2).
    - DPO on tool‑calling: uses the `WorkBench` environment to verify multi‑step calls against database state, generating on‑policy positive/negative trajectories (§3.2).
    - GRPO (a group‑relative policy method) and RLHF (chat helpfulness) on HelpSteer3‑style data (§3.2).
  - Checkpoint interpolation: weight‑space linear merge (α≈0.5) of a reasoning‑strong and a chat‑strong RL checkpoint recovers a balanced capability set (§3.2).

- Compression for 22 GiB and high throughput (§4)
  - Constraint: fit 128k context, batch≥1 in ≤19.66 GiB (22.06 GiB minus framework buffer and room for a vision encoder), and maximize throughput on 8k input/16k output in vLLM (§4.2).
  - Importance scoring (§4.1)
    - `Layer importance`: iteratively remove one candidate layer at a time, compute logits MSE vs original; prune the least impactful layer, repeat (§4.1).
    - `FFN/embedding importance`: activation‑based scoring (mean/L2 over outputs) to drop low‑importance FFN neurons and embedding channels (§4.1).
    - `Mamba head importance`: group‑aware head scoring following Taghibakhshi et al. 2025, but at the modest compression ratios used here, head pruning gave limited benefit (§4.1, §4.4).
  - Lightweight NAS (§4.2)
    - Step 1: pick depth. After 6B tokens of distillation, average reasoning accuracy improves markedly from 52→54→56 layers (44.92 → 47.35 → 51.48; Table 9); fix depth at 56 with 4 attention layers (≈7–8% attention).
    - Step 2: width search within the memory budget. Evaluate top candidates with short KD (19B tokens) and throughput measurement. The chosen `Candidate 2` uses hidden 4480, FFN 15680, 128 Mamba heads, totals 8.89B params with the best accuracy among top‑3 and competitive throughput (Table 10).
  - Distillation schedule (§4.3)
    - Loss: forward KL (teacher logits → student), i.e., match the teacher’s token‑level probability distribution.
    - Reasoning model: depth‑only KD (60B) @8k → width‑pruned KD (50B @8k, 25B @49k, 1B @262k) → DPO → GRPO → KD (0.4B @262k) to recover drops → RLHF → final checkpoint merge (Figure 6).
    - Dataset mix for KD: a 70% reasoning‑SFT + 30% pretraining blend maximizes math accuracy after ~6B KD (Table 11).
    - Base model KD: 120B (depth‑only) + 360B (width) @8k + 2.5B @524k with 100% pretraining data (§4.3).

- Budgeted thinking mechanism (§3.4)
  - Protocol: the model emits a `<think>` token to start its reasoning trace. The runtime counts “thinking tokens”; at the budget limit, it tries to inject a closing `</think>` after the current sentence (or forcibly by +500 tokens if no newline appears). Training with truncated traces makes outputs “well‑formed” (exactly one closing tag) and prevents the model from compensating by writing longer final answers (Figure 5a vs 5b).

## 4. Key Insights and Innovations
- Hybrid Mamba‑heavy depth with minimal attention for long generations (§2.1; Figure 2, Table 1)
  - Novelty: an explicit design where only ~8% of layers are attention while most are Mamba‑2. This slashes KV‑cache costs yet preserves some attention for token‑exact interactions.
  - Significance: enables 128k context and high throughput on 22 GiB hardware while maintaining accuracy on hard reasoning tasks (Figure 1, §4.4).

- Long‑context extension at 512k with synthetic long‑doc QA (§2.6; Table 4)
  - Novelty: train at 512k even though the target is 128k to reduce doc splitting during pretraining and attach generated QA to real long documents.
  - Significance: improves RULER‑128k substantially (up to 81.04 in ablation), without harming other benchmarks (§2.6).

- Alignment for “budgeted thinking” and tool‑calling reliability (§3.2–§3.4; Figure 5)
  - Novelty: mix concatenated 128k SFT, reasoning‑off samples (empty trace), and deliberately truncated traces; DPO in a verifiable multi‑step tool‑calling environment.
  - Significance: model obeys thinking budgets with well‑formed outputs and avoids compensating by bloating the final answer; tool‑calling strengthened through verified on‑policy preferences.

- Compression under explicit memory/throughput constraints using Minitron‑style NAS + KD (§4)
  - Novelty: importance‑guided pruning across layers/FFN/embeddings under a 19.66 GiB cap, with staged long‑sequence distillation and final RLHF/merging.
  - Significance: the final 9B student matches or beats similarly sized baselines while delivering up to 6.3× higher throughput for 8k/16k reasoning workloads (Figure 1; Table 10; §4.4).

- Data curation showing what matters for multilingual and high‑difficulty reasoning (§2.2–§2.3; Tables 2–3)
  - Finding: translated DiverseQA outperforms curated crawl on Global‑MMLU; specialized “Fundamental Reasoning” SFT boosts MMLU‑Pro by +12.1 on an 8B ablation.

## 5. Experimental Analysis
- Evaluation setup (§2.7, §3.3, §4.2)
  - Harness: based on lm‑evaluation‑harness with math grading via Math‑Verify; code via EvalPlus; ARC presented all options; RULER for long‑context (§2.7).
  - Throughput: vLLM measuring output tokens/s/GPU at ISL/OSL 8k/16k; single A10G GPU bfloat16; relative throughput reported (Figure 1, §4.2).
  - Metrics: task accuracies (or pass@k for coding/math), IFEval strictness, BFCL v3 tool‑calling score, Arena‑Hard win‑rate style metric (§3.3).

- Main quantitative results
  - Throughput vs Qwen3‑8B (reasoning workloads; A10G, BF16)
    > “up to 6.3× higher” at 8k input / 16k output; ≈3.3× at 1k/8k (Figure 1).
  - Aligned 12B reasoning model vs Qwen3‑8B/14B (Table 8)
    - Math/Science/Code: AIME‑24 85.42 vs 75.83; AIME‑25 76.25 vs 69.31; MATH‑500 97.75 vs 96.30; GPQA‑Diamond 64.48 vs 59.61; LiveCodeBench 70.79 vs 59.50.
    - Tool use and instruction: BFCL v3 66.98 vs 66.34; IFEval‑Strict 89.81 vs 89.39.
    - Long‑context: RULER‑128k 83.36 vs 74.13.
    - Chat: Arena‑Hard 74 vs 78.4 (Qwen3‑8B) and 87.7 (Qwen3‑14B)—Nano 2 focuses on reasoning/throughput, not topping chat.
    - Mixed: SciCode is lower (18.75 vs 24.65 for Qwen3‑8B), showing a domain where it lags.
  - Base model comparisons (Table 5)
    - `12B Base` vs Qwen3‑8B Base and Gemma3‑12B Base:
      > MMLU 78.24 (vs 76.44, 73.61); MMLU‑Pro‑5shot 63.98 (vs 56.27, 45.12); GSM8K CoT 91.66 (vs 84.00, 74.45); MATH 83.54 (vs 55.40, 42.40); AIME‑24 pass@32 56.67 (vs 20.00, 16.67); HumanEval+ avg@32 61.03 (vs 57.55, 36.68).  
      Commonsense is comparable or better; RULER‑128k 84.74 (Gemma3‑12B 80.70).
    - `9B Base` (the pruned student before alignment) retains strong scores: MMLU‑Pro 59.43, GSM8K 91.36, MATH 80.50, RULER‑128k 82.22.
  - Multilingual (Table 6)
    - `12B Base` averages: Global‑MMLU‑Lite 75.13 (vs Qwen3‑8B 72.81), MGSM 85.94 (vs 80.93).
    - `9B Base` remains competitive (Global‑MMLU‑Lite 69.94; MGSM 84.67).

- Ablations and pipeline diagnostics
  - Multilingual data choice: Within continuous pretraining of a 1B model, DiverseQA‑crawl leads (avg 47.0) vs curated crawl (37.0) and FineWeb‑2 (35.1) on Global‑MMLU (Table 2).
  - Fundamental Reasoning SFT: On an 8B model, MMLU‑Pro jumps 44.24 → 56.36 and average math rises by ≈+1.8 (Table 3).
  - Long‑context training length/synthetic data: RULER‑128k improves from 70.19 (256k, no synthetic) to 81.04 (512k, synthetic) (Table 4).
  - Depth effect under KD: 56 layers best (Table 9).
  - Architecture candidates post‑pruning: Candidate 2 achieves the best accuracy (63.02) with 8.89B params and strong throughput (156 toks/s) under the 8k/16k, batch‑8 test (Table 10).
  - Distillation/policy stages: DPO and GRPO boost tool‑calling (BFCL v3) and instruction following (IFEval), but temporarily depress MMLU‑Pro; a short KD recovers it. RLHF improves Arena‑Hard, then model‑merge balances trade‑offs (Figure 6).
  - Budget control: Truncation training eliminates “compensate by longer final answer” behavior and yields single‑closure well‑formedness across budgets (Figure 5b vs 5a).

- Do the experiments support the claims?
  - Yes for the central claims: (1) comparable or better accuracy vs similarly sized baselines across reasoning/math/code/multilingual (Tables 5–6, 8); (2) substantial throughput gains for long‑generation scenarios on A10G (Figure 1); (3) 128k context with strong RULER scores (Tables 5, 8).
  - Trade‑offs are candidly shown: SciCode lag; chat (Arena‑Hard) behind Qwen3‑8B/14B; temporary metric drops during RL phases and recovery via KD/merging (Figure 6).

## 6. Limitations and Trade-offs
- Scope and assumptions
  - Optimization target is specifically a 22 GiB A10G GPU with vLLM; memory budget and throughput measurements are calibrated to this setting (§4.2). Benefits transfer to other GPUs but exact ratios may differ.
  - The hybrid design fixes ≈7–8% attention layers; other ratios might work better for different tasks but are not exhaustively explored (§4.2.2).

- Data dependencies and potential biases
  - Heavy use of synthetic data (multilingual DiverseQA, STEM Q/A, reasoning SFT) generated by external large models (Qwen, DeepSeek) (§2.2). Quality and bias of these upstream models influence the final model.
  - Most post‑training data are single‑turn prompt‑response with reasoning traces; true multi‑turn, multi‑step interaction breadth may be narrower than in real deployments (§1).

- Performance trade‑offs
  - While reasoning and tool‑calling are strong, some coding‑research tasks (SciCode) and open‑ended chat (Arena‑Hard) trail state‑of‑the‑art chat‑optimized models (Table 8).
  - DPO/GRPO/RLHF phases can cause temporary regressions on knowledge/understanding metrics; extra KD or model merging is required to rebalance (Figure 6).

- Compression boundaries
  - Mamba head pruning brought limited benefits at the modest compression ratios here; larger reductions might require more sophisticated SSM‑aware pruning or retraining (§4.2.2).
  - Final model is 9B parameters in BF16; further memory reductions (e.g., INT8/FP8 inference quantization) are not studied.

- Budgeted thinking mechanism
  - Relies on special `<think>` tags and runtime heuristics for closing the trace; behavior with different decoders or third‑party stacks is untested (§3.4).

## 7. Implications and Future Directions
- Field impact
  - Demonstrates that Mamba‑heavy hybrids can deliver top‑tier reasoning accuracy with much higher throughput for long reasoning traces, even at small‑to‑mid scales and within tight memory budgets. This shifts deployment economics for long‑context, long‑generation applications (§1, Figure 1, Tables 5–6, 8).

- Practical applications
  - Cost‑efficient math/code tutors and graders; verifiable multi‑step tool workflows; legal/technical document analysis over 100k+ tokens; batch reasoning where output‑token throughput dominates cost.
  - On‑prem or edge deployments constrained to 22 GiB‑class GPUs (A10G) with 128k input contexts.

- Research directions
  - Architecture: explore dynamic attention–Mamba allocations (per layer or per token), and principled ways to place attention layers within Mamba stacks (§2.1).
  - Compression: stronger group‑aware SSM pruning at higher compression ratios; joint KD + RL objectives that avoid post‑RL regressions; quantization for inference.
  - Alignment: richer multi‑turn/multi‑tool curricula; improved reward modeling beyond IFEval strictness; more systematic budgeted thinking controls (e.g., token‑adaptive stopping).
  - Data: expand high‑fidelity math/code sources; controllable reasoning‑trace styles; multilingual long‑document QA beyond Wikipedia; better contamination checks at scale (§2.2–§2.3).
  - Evaluation: broader robustness checks (adversarial long‑context, tool‑calling under noisy APIs), and deeper failure analysis on tasks like SciCode.

Overall, Nemotron Nano 2 provides a complete, reproducible recipe—from FP8 pretraining and 512k long‑context extension to alignment and Minitron‑style compression—that yields an open, 9B‑parameter reasoning model with strong accuracy and dramatically higher throughput for long‑generation workloads on commodity datacenter GPUs (Figures 1, 5–6; Tables 4–6, 8, 10; §2–§4).
