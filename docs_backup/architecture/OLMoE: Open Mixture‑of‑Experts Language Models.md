# OLMoE: Open Mixture‑of‑Experts Language Models

**ArXiv:** [2409.02060](https://arxiv.org/abs/2409.02060)
**Authors:** Niklas Muennighoff, Luca Soldaini, Dirk Groeneveld, Kyle Lo, Jacob Morrison, Sewon Min, Weijia Shi, Pete Walsh, Oyvind Tafjord, Nathan Lambert, Yuling Gu, Shane Arora, Akshita Bhagia, Dustin Schwenk, David Wadden, Alexander Wettig, Binyuan Hui, Tim Dettmers, Douwe Kiela, Ali Farhadi, Noah A. Smith, Pang Wei Koh, Amanpreet Singh, Hannaneh Hajishirzi
**Institutions:** Allen Institute for AI, Contextual AI, University of Washington, Princeton University, other affiliated institutions as per author list

## 🎯 Pitch

OLMoE presents the 'OLMOE‑1B‑7B', a fully open Mixture-of-Experts language model innovating with only 1.3B active parameters per token, outperforming models with similar compute while offering unprecedented transparency. This breakthrough matters because it democratizes access to state-of-the-art language models, enabling cost-sensitive deployments in resource-constrained contexts and advancing our understanding of model specialization and efficiency.

---

## 1. Executive Summary (2-3 sentences)
OLMoE introduces a fully open Mixture‑of‑Experts (MoE) language model, `OLMOE‑1B‑7B`, that uses only 1.3B “active” parameters per token while storing 6.9B total parameters, and is pretrained on 5.1T tokens. It outperforms all publicly available models with similar active compute and is accompanied by unusually complete transparency (weights, code, data, logs), plus an extensive set of controlled experiments and analyses that clarify how to build stable, specialized MoE LMs (§§1–3, Fig. 1; Tables 4–5).

## 2. Context and Motivation
- Problem/gap
  - There is a persistent trade‑off between model performance and cost (both training and inference) for language models; high‑performers are expensive to train and use (§1). MoE models can be more compute‑efficient because only a subset of parameters is activated per token, but most MoEs are not fully open and leave critical design choices under‑documented (Fig. 1; §1).
  - Key unanswered MoE design questions include: how many total vs. active parameters to use; how granular to make experts; whether to include a shared expert; which routing algorithm to use; and what auxiliary objectives stabilize training (§1).
- Why this matters
  - Real‑world: Lower active compute means cheaper inference, enabling deployment in resource‑constrained settings while retaining strong accuracy (§1, Fig. 1).
  - Scientific: Lack of open MoEs with intermediate checkpoints, data, and code hinders systematic understanding of routing behavior, expert specialization, and stability (§1; §§4–5).
- Prior approaches and limitations
  - Open dense LMs (e.g., OLMo, Pythia) are transparent but compute‑inefficient at inference (§1).
  - Existing open‑weight MoEs (e.g., Mixtral‑8x7B, Qwen1.5‑MoE) provide weights but not the full recipe; others (OpenMoE, JetMoE) are more open but underperform in the same active‑compute regime (Fig. 1; Table 4; Appendix D).
- Positioning
  - OLMoE positions itself as the most open MoE to date that is also state‑of‑the‑art at its active‑compute level (Fig. 1). It releases model, code, data, logs, every 5k‑step checkpoint, and a large set of ablations, making MoE design choices reproducible and inspectable (§§1–4; Appendix A).

## 3. Technical Approach
This work answers “how to build a performant, stable, economical MoE LM” through a concrete model plus ablations.

- Core model architecture (Fig. 2; §2; Appendix B Table 10)
  - Base: decoder‑only Transformer, 16 layers, hidden size 2048, 16 attention heads, SwiGLU MLP, RoPE positions; sequence length 4096.
  - Replace each dense feed‑forward network (FFN) with an MoE module containing `NE=64` experts; for each input token, select `k=8` experts (“Top‑k” activation) via a learned `router`.
  - MoE computation (Eq. 1): for an input `x`, the router produces scores `r(x)` over experts; the Top‑k experts’ outputs `E_i(x)` are weighted by a softmax over `r(x)` and summed. Only these 8 experts run per token (“active parameters”).
  - Routing algorithm: “dropless token choice” (§4.1.4) — every token chooses experts (no token is dropped), contrasting with “expert choice” where experts pick tokens (Fig. 7).
  - Auxiliary losses added to the standard cross‑entropy (Eq. 2):
    - `Load balancing loss (LLB)` encourages even token distribution across experts (Eq. 3; §4.1.6). Weight α = 0.01.
    - `Router z‑loss (LRZ)` penalizes large router logits to improve stability (Eq. 4; §4.1.7). Weight β = 0.001.
  - Stabilization choices (§4.2): `RMSNorm` instead of non‑param LN (Fig. 14); `QK‑Norm` on queries/keys (Fig. 18); truncated normal initialization (std 0.02, ±3σ cutoffs; Fig. 13); AdamW with epsilon `1e‑8` (Fig. 19). All parameters, including RMSNorm and embeddings, receive weight decay (Figs. 15 & 17).
- Pretraining data and schedule (§2; Table 2; Appendix B)
  - Dataset `OLMOE‑MIX`: high‑quality Common Crawl subset (`DCLM‑Baseline`), plus StarCoder code, Algebraic Stack, arXiv, peS2o, and Wikipedia. Additional filtering removes documents with long repeated n‑grams and low‑quality code repos.
  - 5.133T tokens total (~1.3 epochs), with a final 100B‑token annealing phase where the LR linearly decays to 0 (Fig. 25 shows the annealing markers).
- Adaptation (Instruction + Preference tuning; §2; Table 3; §4.3)
  - SFT: Tulu 2 mix (326k) plus more code (CodeFeedback) and math (MetaMathQA) to boost GSM8K/HumanEval performance.
  - DPO: UltraFeedback pairs (60.8k; filtered) for preference optimization (§4.3).
  - Crucial choice: do not use load‑balancing during adaptation (Table 7 shows it degrades average performance).
  - Also explored `KTO` as a DPO alternative and found comparable averages (Table 7).
- Training and compute (Appendix B)
  - Pretraining: 256 H100 GPUs for ~10 days (FSDP ZeRO, BF16).
  - SFT: 32 H100 GPUs for 33 hours; DPO: 14 hours. KTO: 8 H100 GPUs for 30 hours.

Key unfamiliar terms, briefly:
- `MoE`: a neural layer with many “expert” FFNs; a router activates only a subset per token, reducing active compute.
- `Token choice` vs `Expert choice`: token choice lets each token pick experts; expert choice lets each expert pick tokens. Expert choice can “drop” tokens and poses issues for autoregressive decoding; token choice with load balancing avoids drops (Fig. 7; §4.1.4).
- `Dropless`: no tokens are dropped by the router.
- `Load balancing loss (LLB)`: auxiliary loss encouraging even expert usage (Eq. 3).
- `Router z‑loss`: auxiliary loss discouraging large router logits that cause numeric overflow (Eq. 4).
- `QK‑Norm`: layer normalization applied after attention query/key projections to stabilize attention scores (§4.2.5).

## 4. Key Insights and Innovations
1) A compute‑efficient, fully open MoE that leads its cost class
- What’s new: `OLMOE‑1B‑7B` uses 1.3B active params (6.9B total) and outperforms all available models with similar active compute (Table 4). Its instruction‑tuned variant also surpasses some much larger chat models on the chosen suite (Table 5).
- Why it matters: concrete evidence that small‑active, granular MoEs can match or exceed larger dense models in cost‑sensitive settings, with a fully reproducible recipe (Fig. 1; §§2–3).

2) Fine‑grained experts + token‑choice routing drive quality
- Novelty: Systematic ablations show that many small experts (`64` with `k=8`) beat fewer large ones at fixed FLOPs (Fig. 5), and “dropless token choice” routing beats expert choice despite lower throughput (Fig. 7).
- Significance: Clarifies a long‑standing ambiguity in MoE design—granularity and token‑choice routing are key levers for accuracy.

3) Stability “kit” for MoEs that scales to trillions of tokens
- Components: `RMSNorm` (over non‑param LN; Fig. 14), `QK‑Norm` (Fig. 18), truncated normal init (Fig. 13), `AdamW eps=1e‑8` (Fig. 19), and `router z‑loss` (Fig. 11). Together they suppress loss spikes across multi‑hundred‑billion‑token training.
- Importance: Enables over‑training at massive scale (5T tokens), yielding smooth losses across datasets (Appendix E, Fig. 24).

4) Training from scratch beats sparse upcycling under generous budgets
- Observation: An MoE trained from scratch catches and then surpasses a sparsely upcycled dense LM after ~500–600B additional tokens (Fig. 8), contrary to earlier upcycling claims at smaller budgets.
- Implication: For long training runs or when architectural/hyperparameter changes are desired (e.g., QK‑Norm), start from scratch.

5) New analyses of expert behavior: early saturation and clear specialization
- Early router saturation: Up to ~60% of Top‑8 routing choices already match the final checkpoint after only 1% of training; ~80% by 40% of training (Fig. 20, Eq. 5).
- Specialization with little redundancy: Experts show strong domain (Fig. 22, Eq. 7) and vocabulary specialization (Fig. 23; Table 8; Eq. 8), with limited co‑activation (Fig. 21, Eq. 6).
- Contrast: Mixtral‑8x7B shows much less domain/vocabulary specialization (Figs. 22 bottom, 34, 31), possibly because it was upcycled from a dense model (§5.3).

These are more than incremental tweaks—they codify a practical, stable MoE recipe and provide rare behavioral evidence (saturation/specialization) grounded in released checkpoints and code (§5; Appendix G).

## 5. Experimental Analysis
- Evaluation methodology
  - During pretraining: frequent in‑loop evaluations on multiple MC tasks and perplexity datasets (Appendix C, Table 11; Fig. 25).
  - After pretraining: standardized OLMES suite (ARC‑C/E, BoolQ, CSQA, HellaSwag, MMLU, OBQA, PIQA, SocialIQA, Winogrande) with prescribed prompting and normalization (Appendix C; Table 12 summarizes).
  - After adaptation: instruction‑following, reasoning, coding, and safety benchmarks (MMLU 0‑shot, GSM8K CoT, BBH 3‑shot CoT, HumanEval pass@10, AlpacaEval 1.0, XSTest, IFEval), matching open‑instruct practice (§§2, 4.3; Table 5).
- Main quantitative results
  - Efficiency vs dense (MoE vs Dense): At fixed active params (1.3B), the MoE reaches the dense model’s downstream performance with ~3× fewer tokens (hence ~3× fewer FLOPs), but trains only ~2× faster wall‑clock due to memory overhead (Fig. 4).
  - After pretraining (Table 4; OLMES setup):
    - `OLMOE‑1B‑7B` (1.3B active) achieves MMLU 54.1, HellaSwag 80.0, ARC‑C 62.1, ARC‑E 84.2, PIQA 79.8, Winogrande 70.2.
    - It outperforms dense `Llama2‑7B` (MMLU 46.2) and `OLMo‑1B` (MMLU 32.1), and is competitive with some 7–8B dense LMs while using ~6–7× less compute per forward pass (Table 4; §3).
  - After adaptation (Table 5):
    - `+SFT` boosts across the board; e.g., GSM8K from 3.0→40.5 EM (≈13.5×), consistent with added math data.
    - `+DPO` (final `OLMOE‑1B‑7B‑INSTRUCT`) yields the highest average (57.7) among compared models, including `Qwen1.5‑3B‑14B‑Chat` (57.3) and `DeepSeek‑3B‑16B‑Chat` (57.0). Notable metrics: AlpacaEval 84.0 %win; GSM8K 45.5 EM; BBH 37.0 EM; IFEval Loose Acc 48.1 (Table 5).
  - Adaptation ablations (Table 7):
    - No load‑balancing during SFT/DPO is better (Avg 57.7 vs 57.1 with LBL).
    - Using the post‑annealing checkpoint is preferable (Avg 57.7 vs 56.3).
    - `KTO` is roughly on par with `DPO` on average (both 57.7); DPO edges AlpacaEval.
- Ablation studies (selected highlights)
  - `Expert granularity`: 32 experts (k=4) > 8 experts (k=1); 64 experts (k=8) adds a smaller but consistent gain (Fig. 5).
  - `Shared expert`: slightly harms performance versus all routed experts (Fig. 6).
  - `Token choice` > `Expert choice` on accuracy (Fig. 7), though EC is ~20% faster throughput.
  - `Load balancing loss`: improves stability/quality; without it, experts collapse early to a few “hot” experts (Figs. 9–10). Weight α=0.01 used in pretraining.
  - `Router z‑loss`: fewer spikes; better validation and downstream metrics (Fig. 11).
  - `Dataset`: OLMOE‑MIX outperforms Dolma 1.7 given the same training setup (Fig. 12).
  - `Initialization`: truncated normal prevents divergence after ~450B tokens seen with normal init (Fig. 13).
  - `RMSNorm` and `QK‑Norm`: both improve stability/quality (Figs. 14, 18).
  - `AdamW eps=1e‑8` > `1e‑5` (Fig. 19).
  - `Sparse upcycling`: scratch MoE catches up ≈500B tokens and surpasses ~600B tokens (Fig. 8); adding noise to upcycled weights didn’t help (Appendix F, Fig. 28).
  - `Layer‑shared MoE`: similar or slightly worse than dense at matched compute (Appendix F, Fig. 29).
- Behavioral analyses (Section 5)
  - `Router saturation`: routing choices stabilize very early (Fig. 20), especially in later layers; the first layer is an outlier that saturates slower, explaining slower load‑balance convergence there (§5.1).
  - `Co‑activation`: few expert pairs fire together strongly; suggests limited redundancy (Fig. 21).
  - `Domain specialization`: clear specializations for arXiv/GitHub in OLMOE (Fig. 22 top), but near‑uniform usage in Mixtral (Fig. 22 bottom); top‑1 specializations confirm the contrast (Fig. 34).
  - `Vocabulary specialization`: later layers route by predicted next token and show strong specialization—e.g., Expert 27 in layer 7 focuses on non‑Latin scripts; Experts 48/23 focus on connectors; Expert 4 on measurement units (Fig. 23; Table 8). OLMOE shows stronger specialization than Mixtral (Appendix G, Figs. 30–31).

> “During pretraining OLMOE‑1B‑7B reaches better performance with less compute (FLOPs) than dense OLMo models… despite having used less than half as many FLOPs… with only 1B active parameters.” (Fig. 3)

> “OLMOE‑1B‑7B performs best among models that use <2B active parameters… and even outperforms some 7B dense LMs such as Llama2‑7B.” (Table 4; §3)

Assessment: The experimental program is unusually thorough—comparing dense vs MoE, multiple routing designs, init and normalization strategies, dataset choices, and adaptation settings. The results are consistent across training loss, validation loss, and diverse downstream tasks, and they are grounded in transparent configurations and released logs (links in §2, Appendix A). The specialization analyses further back the design choices by showing the MoE is indeed learning differentiated functions rather than redundant experts (§5).

## 6. Limitations and Trade-offs
- Scope and assumptions (Section H)
  - Size: With only 1.3B active parameters, there is an absolute ceiling on capability vs large dense models; bigger active compute or recursion/agentic scaffolding may be needed to match 8B+ dense models (§H).
  - Data scale: Trained for 5T tokens; more data (e.g., 15T as in Llama 3) could further improve results, but the effectiveness of over‑training for MoEs vs dense remains an open question (§H).
  - Modality and language: Text‑only and predominantly English; no multimodal inputs or multilingual evaluations (§H).
- Engineering trade‑offs
  - Memory and throughput: Although active compute is small, total parameters (6.9B) require more memory to store than a 1B dense model, and MoE training throughput is lower than dense at equal active params, yielding only ~2× time speedup even with 3× token efficiency (Fig. 4).
  - Routing constraints: The load‑balancing loss is needed for pretraining stability but may constrain extreme specialization (authors suggest future work on removing or relaxing it; §4.1.6).
  - EC vs TC trade‑off: Expert choice (EC) gives ~20% higher throughput but worse accuracy and practical issues for autoregressive decoding; token choice (TC) wins on quality (Fig. 7).
- Compute cost and accessibility
  - Although more economical at inference, pretraining still used 256 H100s for ~10 days; open‑source does not mean cheap to reproduce (§B).

## 7. Implications and Future Directions
- Field impact
  - OLMoE establishes a strong, reproducible baseline for sparse MoEs with end‑to‑end openness (weights, code, data, logs, checkpoints), enabling principled research on routing behavior and specialization—areas that were difficult to study without full artifacts (§1; Appendix A).
  - The ablations provide a practical “recipe” for others: fine‑grained experts (64, k=8), token‑choice routing with LBL and z‑loss, RMSNorm + QK‑Norm, truncated normal init, AdamW eps=1e‑8, and high‑quality data mixing (§§4.1–4.2).
- Research opportunities
  - Routing/objectives: Explore removal/relaxation of load‑balancing to allow uneven expert usage, or hybrid objectives that encourage both specialization and stability (§4.1.6).
  - First‑layer behavior: Investigate why layer 0 saturates much more slowly and how to stabilize it (Fig. 20).
  - Expert placement and inference: Use co‑activation statistics to co‑locate experts for lower communication overhead; dynamic capacity and adaptive routing (Figs. 21, 35).
  - Upcycling vs scratch: Characterize regimes where upcycling helps vs hurts, including noise schemes and architectural drift (Fig. 8; Appendix F).
  - Data mixture optimization: Automate data selection/mixing; the Reddit/FLAN additions were inconclusive (Appendix F, Fig. 26).
  - Multimodal/multilingual MoEs: Extend token‑choice design to vision/audio and multiple languages (§H).
- Practical applications
  - Cost‑sensitive deployment: 1.3B active parameters with strong accuracy makes for economical chat, coding assistance, and math tutoring—especially after SFT/DPO (Table 5).
  - Domain‑specialized inference: The demonstrated domain/vocabulary specialization suggests opportunities for domain‑aware routing, expert pruning, or fine‑tuning only certain experts for verticals (Figs. 22–23; Table 8).
- Ongoing improvements
  - A newer `OLMOE‑1B‑7B‑0125` annealed on a curated `DOLMINO` mix and post‑trained with an updated Tulu‑3 pipeline further improves OLMES average from 71.1→72.7 after pretraining (Table 16), and yields stronger post‑training averages (Table 17, Appendix I).

> “Router saturation reaches ~80% by 40% of pretraining, and later layers saturate earlier.” (Fig. 20) This enables earlier analyses/diagnostics during training and suggests that high‑level expert roles are decided early.

Overall, OLMoE changes the practical calculus for open MoEs: it provides a replicable, stable template that is compute‑efficient at inference, and it opens the black box of expert routing/specialization through full transparency and analysis.
