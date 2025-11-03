# MobileLLM: Optimizing Sub-billion Parameter Language Models for On-Device Use Cases

**ArXiv:** [2402.14905](https://arxiv.org/abs/2402.14905)
**Authors:** Zechun Liu, Changsheng Zhao, Forrest Iandola, Chen Lai, Yuandong Tian, Igor Fedorov, Yunyang Xiong, Ernie Chang, Yangyang Shi, Raghuraman Krishnamoorthi, Liangzhen Lai, Vikas Chandra
**Institutions:** 

## 🎯 Pitch

MobileLLM introduces a hardware-aware architecture for sub-billion parameter language models that leverage deep-and-thin networks, embedding sharing, and grouped-query attention to operate efficiently on mobile devices without significant accuracy loss. This innovation bridges the gap between cloud-scale models and practical on-device intelligence, enabling privacy-preserving applications with sustained all-day use, making advanced AI accessible for real-world tasks on mobile hardware.

---

## 1. Executive Summary
MobileLLM designs and trains sub‑billion parameter language models (125M–350M, with extensions up to 1.5B) that run efficiently on mobile devices without major accuracy loss. The core contribution is a hardware‑aware architecture for small LLMs—deep‑and‑thin networks with embedding sharing, grouped‑query attention, and a new immediate block‑wise layer sharing—that delivers state‑of‑the‑art accuracy for models under 1B parameters and competitive performance on real on‑device tasks like chat and API calling.

## 2. Context and Motivation
- Problem addressed
  - Running today’s large LLMs on phones is impractical due to memory (DRAM) and energy constraints and unacceptable latency. Figure 2 shows typical device memory hierarchies: only 6–12 GB DRAM, of which one app should use ≈10% (Section 1).
  - Energy is also limiting: using the estimate “0.1 J per token per billion parameters,” a 7B model needs ~0.7 J/token; a phone battery (~50 kJ) would deplete quickly in interactive use (Section 1).
- Importance
  - On‑device models eliminate cloud cost and round‑trip latency, enabling privacy‑preserving assistants and interactive apps. The paper quantifies that a 350M 8‑bit model (~0.035 J/token) can sustain all‑day use on a phone, and a 125M model can decode at ~50 tokens/s vs. 3–6 tokens/s for 7B on current iPhone apps (Section 1).
- Shortcomings of prior approaches
  - Most open sub‑billion models (OPT‑125M/350M, Pythia, BLOOM‑560M, RWKV, etc.) were not designed for phone‑class constraints; they underutilize parameters at small scales (Tables 3–4).
  - The dominant “scaling law” perspective emphasized data and parameter count, treating architecture as secondary. Section 2.2.2 and Tables 11–12 show that, for small models, architecture choices (especially depth vs. width) matter substantially.
- Positioning
  - MobileLLM reframes small‑model design as a weight‑utilization problem under tight memory/latency budgets. It introduces a set of architectural choices and a layer‑sharing mechanism engineered to exploit the device memory hierarchy (Figure 6 + Section 2.3), then validates the approach across reasoning, QA, chat, and API calling.

## 3. Technical Approach
MobileLLM builds a strong sub‑billion baseline with four design choices, and then augments it with a layer‑sharing scheme that improves accuracy while keeping model size fixed.

A. Training setup (Section 2.1 / 3.1)
- Pretraining uses Adam with weight decay 0.1, initial LR 2e‑3, cosine decay.
- Exploratory runs: 120k iterations on 0.25T tokens. Final models: 480k iterations on 1T tokens on 32 A100 GPUs (batch size 32/GPU).

B. Strong baseline design (“MobileLLM”, Section 2.2; Figure 3; Table 10)
1) Use `SwiGLU` feed‑forward networks  
   - `SwiGLU` (a gated activation) replaces the standard ReLU MLP. This consistently increases average zero‑shot accuracy by ~1.3 points at both 125M and 350M in controlled ablations (Table 10).

2) Prefer “deep and thin” architectures  
   - “Depth” = number of transformer layers; “width” = embedding dimension and head count.  
   - For the same parameter budget, deeper models outperform wider ones on most tasks (Figures 4a–f; Tables 11–12). Example: at ~125M parameters, 30 layers (dim=512) average 44.8 vs. 12 layers (dim=768) averaging 43.9 on zero‑shot reasoning (Table 11).

3) `Embedding sharing`  
   - Definition: reuse the input token embedding matrix as the output classification matrix (logit projection), because both are `vocab_size × embedding_dim`. This reduces parameters without changing compute (Section 2.2.3).  
   - At 125M with 30 layers, sharing cuts ~16M parameters (11.8%) with only a 0.2‑point average drop; using those saved parameters to add two layers restores and slightly improves accuracy (from 44.8 to 45.0) while still being smaller (Table 1).

4) `Grouped‑Query Attention (GQA)`  
   - Definition: use more query heads than key/value heads, reusing each KV head across several query heads (Section 2.2.4). This reduces KV redundancy and KV‑cache size.  
   - Empirical choice: 16 query heads with 4 KV heads works best at these scales (Figure 5; Table 13). Keeping model size constant by slightly increasing embedding dimension after adopting GQA yields +0.4 points at 125M and +0.7 points at 350M (Table 10).

C. Immediate block‑wise `layer sharing` (“MobileLLM‑LS”, Section 2.3; Figure 6; Tables 2, 7, 14)
- Goal: increase effective depth without increasing the parameter count or DRAM traffic.  
- Mechanism: reuse the same block’s weights across two adjacent layers and compute that block twice in immediate succession. Because the shared weights stay in the on‑chip cache (SRAM), the model avoids reloading them from DRAM (Figure 2). That is why “immediate block‑wise” sharing (Figure 6b) is favored over “repeat‑all‑over” (Figure 6c), despite the latter’s slightly higher accuracy in one table, due to better cache locality and lower latency on mobile hardware (discussion under Figure 6 and Section 2.3).
- Effect: +0.4 to +0.6 points accuracy at constant parameter count when doubling layers via sharing (Table 14). On an iPhone 13, MobileLLM‑LS‑125M adds only ~2–3% execution time vs. the non‑shared 30‑layer baseline (Table 7), while a true 60‑layer non‑shared model would nearly double execution time and more than double load/init time.

D. Putting it together
- Final 125M config (Table 9): 30 layers, 9 heads, 3 KV heads, dim=576, hidden dim=1536 with embedding sharing and GQA; the LS variant doubles effective layers via immediate sharing.
- Final 350M config: 32 layers, 15 heads, 5 KV heads, dim=960, hidden dim=2560, with embedding sharing and GQA; LS doubles effective layers via sharing.

E. Fine‑tuning for downstream tasks (Section 3.3)
- Chat: fine‑tune MobileLLM and baselines identically; evaluate on AlpacaEval and MT‑Bench (Table 5).
- API calling: synthetic dataset of 5k train / 2.5k test conversations (8 turns on average; appendix H.5 shows examples). Fine‑tune for 4 epochs with LR 2e‑5 (Table 6).

F. Quantization and KD (Sections 3.4–3.5; Table 15–16; Figure 7)
- Post‑training 8‑bit weight/activation (W8A8) quantization with simple per‑token min‑max calibration drops <0.5 points and is compatible with layer sharing (Figure 7; Table 15).
- Knowledge Distillation from LLaMA‑v2‑7B slows training 2.6–3.2× and does not improve accuracy over label‑only training (Table 16).

## 4. Key Insights and Innovations
1) Depth beats width for small LLMs (fundamental insight)  
   - Evidence: a broad grid over layers/width at fixed parameter budgets shows deeper, thinner models consistently outperform shallower, wider ones on zero‑shot reasoning and on QA/reading comprehension (Figures 4a–f; Tables 11–12).  
   - Significance: challenges the common “architecture doesn’t matter” belief at small scales, and provides a design rule for sub‑billion models.

2) Squeeze more utility from parameters via three complementary sharing strategies (incremental to fundamental)
   - `Embedding sharing` (Table 1; Section 2.2.3): trims >10% parameters with negligible/recouped accuracy loss by reallocating to depth.
   - `Grouped‑query attention` (Figure 5; Table 13): reduces KV redundancy and KV cache with almost no accuracy loss; when re‑balancing dimensionality to keep total size, accuracy improves.
   - `Immediate block‑wise layer sharing` (Figure 6b; Tables 2, 7, 14): doubles effective depth without adding parameters and with minimal latency overhead by exploiting cache locality—this connects architecture to SoC memory hierarchy, which is novel in the small‑LLM context.

3) Hardware‑aware evaluation and profiling (practical innovation)  
   - Beyond offline metrics, the work quantifies mobile latency and memory implications (Figure 2; Table 7), showing the design is not only accurate but deployable on actual phones.

4) Small models can solve realistic on‑device tasks (application‑level insight)  
   - With appropriate pretraining and light fine‑tuning, 350M‑parameter models rival much larger models on structured tasks such as API calling intent/structure matching (Table 6), highlighting the practicality of on‑device LLM assistants.

## 5. Experimental Analysis
- Evaluation setup (Sections 2.1, 3.1)
  - Pretraining on 1T tokens for final models. Zero‑shot evaluations on eight commonsense tasks: ARC‑easy/challenge, BoolQ, PIQA, SIQA, HellaSwag, OBQA, WinoGrande (Tables 3, 10–15). QA/RC on TriviaQA (TQA) and RACE (Table 4). Chat benchmarks: AlpacaEval and MT‑Bench (Table 5). API calling: synthetic dataset with exact‑match and ROUGE metrics (Table 6).
- Main zero‑shot results (Table 3)
  - 125M scale:
    - `MobileLLM‑125M`: avg 46.3 vs. OPT‑125M 42.6, GPT‑Neo‑125M 42.9, Pythia‑160M 42.5, RWKV‑169M 43.6.  
    - `MobileLLM‑LS‑125M`: avg 47.0 (+0.7 over MobileLLM).  
    - Quote: “MobileLLM‑LS‑125M achieves comparable or even higher results than most previous 350M models.” (Table 3 values show it surpasses OPT‑350M at 43.9 and BLOOM‑560M at 44.2.)
  - 350M scale:
    - `MobileLLM‑350M`: avg 51.3; `MobileLLM‑LS‑350M`: 52.1.  
    - Both exceed prior models: Pythia‑410M (46.6), RWKV‑430M (47.0), BLOOM‑560M (44.2), OPT‑350M (43.9).
- QA/Reading comprehension (Table 4)
  - At 125M: `MobileLLM‑125M` reaches TQA F1 14.3 (5‑shot) vs. OPT‑125M 9.6; RACE middle/high 39.7/28.9 vs. 34.7/27.5 for OPT‑125M.  
  - At 350M: `MobileLLM‑350M` TQA F1 23.9 (5‑shot) vs. Pythia‑410M 13.8, BLOOM‑560M 8.9; RACE middle/high 45.6/33.8 vs. ~37–39/28–30 for other 350–560M baselines.
- Chat (Table 5)
  - Under 200M: `MobileLLM‑125M` MT‑Bench 2.33 vs. OPT‑125M 1.21; AlpacaEval win‑rate 24.07% vs. ≤4% for others.  
  - 200M–1B: `MobileLLM‑350M` MT‑Bench 3.28, AlpacaEval 47.08%; LS variant 3.16 / 48.20%. These values approach the “self‑win” of text‑davinci‑001 (~50%).
- API calling (Table 6)
  - `MobileLLM‑350M`: EMintent 65.3 and EMstructure 48.8; LLaMA‑v2‑7B: 62.8 and 50.9.  
  - Interpretation: the small model matches or slightly exceeds a 7B model on recognizing which API to call, with very close structure accuracy; ROUGE for “agent response” is lower, but API correctness dominates this task’s utility.
- Ablations and robustness
  - Depth vs. width sweeps (Figures 4, Tables 11–12) show consistent advantages from depth across many tasks, not just a single benchmark.  
  - Embedding sharing (Table 1) shows negligible loss that can be traded for depth gains.  
  - GQA head/KV‑head sweeps (Figure 5; Table 13) identify 16/4 as a robust setting.  
  - Layer‑sharing strategies (Figure 6; Table 2): “repeat‑all‑over” slightly best in average accuracy, but “immediate block‑wise” is adopted for device latency.  
  - Repetition count (Table 14): doubling gives gains; larger repetition shows diminishing returns.
- Quantization (Figure 7; Table 15)
  - W8A8 per‑token min‑max PTQ yields ≤0.5‑point average drop and preserves LS benefits across 125M and 350M.
- Knowledge distillation (Table 16)
  - KD from LLaMA‑v2‑7B does not improve accuracy for 125M/350M and slows training 2.6–3.2×; the label‑only regime is favored.
- On‑device profiling (Table 7)
  - iPhone 13 (MPS backend): `MobileLLM‑LS‑125M` has ~2.6% execution overhead vs. the baseline 30‑layer model. A non‑shared 60‑layer model would increase execution time by ~86% and load/init time by 143%.
- Scaling beyond sub‑billion (Appendix A; Table 8)
  - Extending the same design to 600M/1B/1.5B shows strong results. Example: `MobileLLM‑1.5B` averages 59.4 vs. Qwen1.5‑1.8B at 56.5.

Assessment: The experiments are extensive, compare to strong open baselines under consistent evaluation (Tables 3–6), and include thorough ablations (Tables 1–2, 10–16). The consistency across tasks and the device‑level profiling convincingly support the claims.

## 6. Limitations and Trade-offs
- Training compute and data
  - Final models train on 1T tokens with 32×A100 GPUs (Section 2.1/3.1). While smaller at inference, these models still require large‑scale pretraining resources that may limit reproducibility outside major labs.
- Latency vs. accuracy in layer sharing
  - “Repeat‑all‑over” sharing is sometimes slightly more accurate than immediate block‑wise (Table 2: e.g., 350M 50.7 vs. 50.2 average), but is not chosen due to hardware locality considerations. This is a deliberate latency/accuracy trade‑off.
- Task coverage
  - Evaluations emphasize commonsense/QA/chat and API calling. Long‑context reasoning, tool‑use beyond simple function calling, multilingual performance, and safety alignment are not deeply studied.
- Quantization scope
  - Only W8A8 PTQ with simple calibration is reported. More aggressive quantization (e.g., 4‑bit), sparsity, or structured pruning are not covered in the main results, though the paper argues compatibility (Section 3.4).
- Architecture generality
  - The “depth over width” insight is validated across many small‑scale tasks but may not universally transfer to domains with very long sequences or different training regimes. The paper does not analyze theoretical reasons beyond empirical evidence.

## 7. Implications and Future Directions
- Field impact
  - Establishes a practical recipe for sub‑billion LLMs: go deep‑and‑thin, share embeddings, adopt GQA, and exploit cache‑aware immediate layer sharing. This moves small‑model design from naive down‑scaling to hardware‑aware optimization with clear empirical rules (Figures 4–6; Table 10).
- Research directions
  - Memory‑aware training/inference: extend immediate sharing to dynamic or conditional sharing; combine with sparsity/pruning and low‑bit formats; co‑design with attention kernels like FlashAttention for edge hardware.  
  - Theoretical understanding: analyze why depth helps more than width at small scales; relate to optimization dynamics and representation hierarchy.  
  - Long‑context and multimodal: test whether these design choices hold under long‑sequence training or when adding vision/audio encoders for on‑device assistants.  
  - Better distillation: KD underperformed (Table 16). Investigate sequence‑level distillation, imitation learning from tool‑use traces, or curriculum teacher forcing specialized for small models.
- Practical applications
  - On‑device assistants, offline chat, and structured tasks like API/function calling (Table 6) where 350M models already approach 7B intent accuracy.  
  - Privacy‑sensitive domains (health, personal productivity) where cloud use is undesirable, and latency must be low.  
  - Edge deployments in wearables, cars, and IoT where DRAM and power are constrained; the immediate sharing technique is especially relevant to these memory hierarchies.

> Bottom line: By rethinking architecture for the edge—prioritizing depth, reusing weights smartly, and aligning with device memory constraints—MobileLLM makes small LLMs markedly more capable and deployable, narrowing the gap between cloud‑scale models and on‑device intelligence.
