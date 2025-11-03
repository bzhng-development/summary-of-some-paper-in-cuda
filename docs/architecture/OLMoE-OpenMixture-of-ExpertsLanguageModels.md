# OLMoE: Open Mixture-of-Experts Language Models

**ArXiv:** [2409.02060](https://arxiv.org/abs/2409.02060)

## 🎯 Pitch

OLMoE introduces the first fully open, high-performing Mixture-of-Experts (MoE) language model suite, offering competitive accuracy at a fraction of the active parameter and compute cost, thanks to a design that activates only 1B out of 7B total parameters per token. By releasing not just model weights, but also the training data, code, logs, and a suite of transparent ablations, OLMoE empowers the community to rigorously study, reproduce, and improve cost-efficient sparse models—paving the way for more accessible, efficient, and scientifically grounded language model development.

---

## 1. Executive Summary (2–3 sentences)
OLMoE introduces a fully open Mixture‑of‑Experts (MoE) language model, `OLMOE‑1B‑7B`, that activates ~1B parameters per token while storing 6.9B total parameters. Trained from scratch on 5.1T tokens and then adapted, it achieves state‑of‑the‑art accuracy among models with similar active‑parameter cost and ships with model weights, training data, code, logs, and dense ablations that clarify how to train stable, high‑quality MoEs (§1, Table 1; Fig. 1; Tables 4–5; Appendix A).

## 2. Context and Motivation
- Problem addressed
  - Cost–performance gap in large language models (LLMs): strong models are expensive to train and run, limiting accessibility (§1). Sparse MoEs promise better cost/performance by activating only a small subset of parameters per token (Fig. 2), but best‑performing MoEs are mostly closed (lack data/code/logs) and their practical design choices remain unclear (Fig. 1; §1).
- Why it matters
  - Real‑world impact: lower inference cost per token enables broader deployment on limited hardware while keeping quality high. 
  - Scientific significance: MoEs introduce new degrees of freedom (number/size of experts, routing strategy, auxiliary losses, initialization) that have interacting effects on stability and quality; the field lacks transparent, controlled evidence (§1; Table 1).
- Prior approaches and gaps
  - Open‑weight MoEs exist (e.g., Mixtral‑8x7B, DeepSeekMoE, JetMoE, Qwen1.5‑MoE), but most do not release full recipes, data, or logs (Fig. 1; Appendix D).
  - Claims in prior work conflict on critical choices: expert granularity [39], shared experts [39], routing variants [219, 154, 58], and upcycling dense models to MoE [85]. Stability tricks are also underdocumented (e.g., router z‑loss [221]).
- Positioning of this work
  - A fully open MoE suite with: (i) a performant base model (`OLMOE‑1B‑7B`), (ii) an adapted chat model (`OLMOE‑1B‑7B‑INSTRUCT`), (iii) a transparent, controlled set of ablations on MoE‑specific and general training choices (§4), and (iv) analyses that reveal how MoEs route and specialize (§5).
  - An improved January 2025 iteration (`OLMOE‑1B‑7B‑0125`) shows further gains via a curated annealing mix and updated post‑training (Appendix I; Tables 16–17).

## 3. Technical Approach
This section explains how `OLMOE‑1B‑7B` is built and trained, and why particular design decisions were made.

- Core architecture (Fig. 2; §2)
  - Transformer decoder; the usual feed‑forward network (FFN) in each layer is replaced by an MoE module with `N_E` experts. For each token, the router selects the top‑`k` experts to process that token; outputs are weighted by routing probabilities and summed:
    - Equation (1): MoE output = sum over `i ∈ Top‑k(r(x))` of `softmax(r(x))_i * E_i(x)`, where `r(x)` is a learned linear router and `E_i` is expert `i`.
  - Final training loss adds two auxiliary terms to cross‑entropy (Eq. 2): `L = L_CE + α L_LB + β L_RZ`.
    - `L_LB` (Eq. 3, load‑balancing loss): encourages approximately equal token assignment across experts by multiplying, for each expert, the fraction of tokens it receives (`f_i`) with the sum of its routing probabilities (`P_i`), then summing across experts.
    - `L_RZ` (Eq. 4, router z‑loss): penalizes large pre‑softmax router logits to prevent numerical issues and improve quality.
- Final configuration chosen (Table 1; Appendix B)
  - Active vs total parameters: ~1.3B active, 6.9B total.
  - Experts: 64 experts per MoE layer; 8 are activated per token (`k = 8`). This fine granularity increases the number of expert combinations while keeping compute fixed (§4.1.2; Fig. 5).
  - Routing: dropless token‑choice routing [58, 154]; every token is assigned to exactly `k` experts (no token dropping), using a load‑balancing loss (§4.1.4; Fig. 7).
  - Loss weights: `α = 0.01` for load balancing (§4.1.6) and `β = 0.001` for z‑loss (§4.1.7).
  - Stability improvements (§4.2): truncated normal initialization (Fig. 13), RMSNorm instead of non‑parametric LN (Figs. 14 and 16), Query‑Key normalization (QK‑Norm; Fig. 18), and a smaller AdamW `epsilon` of 1e‑8 (Fig. 19). RMSNorm and embeddings are included in weight decay (Figs. 15 and 17).
- Data and training pipeline (§2; Table 2; Appendix B)
  - Pretraining corpus `OLMOE‑MIX` combines DCLM‑Baseline web pages with selected high‑quality sources (StarCoder code, peS2o and arXiv STEM, OpenWebMath, Algebraic Stack, Wikipedia). The mix outperforms Dolma 1.7 in ablations (Fig. 12).
  - Total training tokens: 5.133T (1.3 epochs over the dataset), with a final 100B‑token annealing phase during which the dataset is reshuffled and learning rate decays linearly to zero (§2).
  - Sequence length 4096; training with ZeRO/FSDP and BF16 (Appendix B).
- Adaptation (instruction and preference tuning; §2; Table 3; Appendix B; §4.3)
  - Supervised Fine‑Tuning (SFT): curated instruction data with extra code and math to boost GSM8k and coding (Table 3).
  - Preference tuning: DPO primarily; KTO is also tested and performs comparably on average (Table 7). Load balancing loss is turned off during adaptation (Table 6 and Table 7), as routing stays balanced and quality improves.
  - The final chat model is `OLMOE‑1B‑7B‑INSTRUCT` (Table 5).
- Why these design choices (supported by controlled ablations in §4)
  - MoE vs dense: MoE reaches the dense model’s accuracy using ~3× fewer tokens (compute), but because MoE stores more total parameters it trains ~2× faster in wall‑clock time in their setup (§4.1.1; Fig. 4). Hence, MoE is compute‑efficient even if some throughput is lost to communication/memory.
  - Fine‑grained experts: More, smaller experts (e.g., 64 with 8 active) yield better downstream accuracy than fewer, larger ones at the same compute, with diminishing returns after ~64 (§4.1.2; Fig. 5).
  - No shared expert: always‑active “shared” experts slightly hurt performance and drastically reduce expert‑combination flexibility (§4.1.3; Fig. 6).
  - Token‑choice routing (dropless) outperforms expert‑choice on quality, albeit with lower throughput (§4.1.4; Fig. 7).
  - Avoid sparse upcycling: starting from a dense checkpoint loses its early advantage after a few hundred billion tokens in their setting and constrains choices like initialization (§4.1.5; Fig. 8).

## 4. Key Insights and Innovations
- A fully open, competitive MoE that is cheap per token
  - Contribution: `OLMOE‑1B‑7B` uses ~1B active parameters, rivals or beats many larger dense models in its cost regime, and ships with pretrained, SFT, and DPO variants plus data, code, and logs (Fig. 1; Tables 4–5; Appendix A).
  - Significance: It sets a transparent baseline for MoE training and evaluation at low inference cost, closing a critical accessibility gap (§1).
- Evidence‑backed MoE recipe with stability fixes
  - Contribution: A carefully validated training recipe—truncated normal init (Fig. 13), RMSNorm (Figs. 14, 16), QK‑Norm (Fig. 18), z‑loss (Fig. 11), smaller AdamW epsilon (Fig. 19), and dropless token‑choice routing with load‑balancing (Figs. 7, 9–10).
  - Significance: Converts conflicting, scattered practices into a coherent, reproducible configuration that trains stably to 5T tokens (§4).
- Design clarifications that challenge prior heuristics
  - Fine‑grained experts help; shared experts don’t (Figs. 5–6). Token‑choice routing yields better accuracy than expert‑choice in this setup (Fig. 7). Sparse upcycling quickly loses its lead at this scale and constrains hyperparameters (Fig. 8).
  - Significance: These negative/positive results directly inform future MoE designs beyond this model.
- New analyses of routing behavior and specialization (§5)
  - Router saturation: routing decisions converge very early—up to ~60% of top‑8 expert choices are already stable after only 1% of pretraining, with later layers saturating earlier (Fig. 20).
  - Specialization: experts exhibit domain specialization (e.g., arXiv and GitHub) and vocabulary specialization (e.g., punctuation, units, names), with minimal co‑activation indicating low redundancy (Figs. 21–23; Table 8; §5.2–5.4). Mixtral shows less domain specialization in a like‑for‑like comparison (Fig. 22 bottom).

## 5. Experimental Analysis
- Evaluation methodology
  - During pretraining: a suite of multiple‑choice tasks (e.g., HellaSwag, MMLU, ARC‑Ch/E) tracked vs tokens/compute (Fig. 3, Fig. 25). Multiple evaluation formulations are used (e.g., CF/MCF; Appendix C, Table 11).
  - After pretraining: OLMES standard (§3; Table 4 and Table 12) with consistent prompt formatting and scoring; also DCLM’s Core/Extended evals (Table 13).
  - After adaptation: instruction‑following, math, coding, safety, and instruction‑following fidelity (Table 5); for the 0125 model, the Tulu‑3 eval suite (Table 17).
- Main quantitative results
  - Compute efficiency (MoE vs dense): 
    > “MoE reaches the dense model’s final performance with ~3× fewer tokens (compute), but only ~2× faster in time due to memory overhead” (Fig. 4).
  - After pretraining (OLMES; Table 4):
    - Among ~1B active‑parameter models, `OLMOE‑1B‑7B` is top on all listed tasks (e.g., MMLU 54.1 vs DCLM‑1B 48.5; HellaSwag 80.0 vs DCLM‑1B 75.1; WinoGrande 70.2 vs DCLM‑1B 68.1).
    - It also surpasses some 7–9B dense baselines (e.g., Llama‑2‑7B: MMLU 46.2 vs 54.1; Table 4).
  - After adaptation (Table 5):
    - SFT boosts GSM8k dramatically (from 3.0 EM to 40.5; “>10× gain” noted in §3), consistent with extra math data (Table 3).
    - DPO further improves AlpacaEval 1.0 (%win) from 69.2 to 84.0 and raises the overall average to 57.7, outperforming several larger or higher‑cost chat models (e.g., Qwen1.5‑3B‑14B Chat avg 57.3).
  - Improved 0125 release (Appendix I):
    - With curated annealing (DOLMINO; Table 15), the base model improves OLMES average by +1.6 and MMLU by +2.1 (Table 16).
    - With the Tulu‑3 post‑training pipeline, the adapted model gains ~10 points on the Tulu eval average (39.8 → 49.8) and markedly improves instruction‑following metrics like IFEval (45.3 → 66.4) and GSM8k CoT (47.4 → 72.4) (Table 17).
- Ablations, robustness, and negative results (Section 4)
  - Expert granularity, shared experts, routing variants, upcycling, auxiliary losses, initialization, layer norm, QK‑Norm, AdamW epsilon, dataset composition—all tested with controlled changes (Figs. 5–19; §4.1–4.2). 
  - Notable findings:
    - Load‑balancing loss is necessary in pretraining (Fig. 9–10) but can be dropped in SFT/DPO without harming routing balance (Table 6; Table 7).
    - Adding Reddit or FLAN to the corpus mix does not yield consistent gains (Appendix F, Fig. 26).
    - Precision of the load‑balancing computation (BF16 vs FP32) does not fix spikes (Fig. 27).
    - Layer‑shared MoE does not beat a dense model at equal compute and even reduces throughput (~20% lower) (Fig. 29).
- Do the experiments support the claims?
  - Yes. The paper pairs headline results (Tables 4–5; Fig. 1) with extensive ablations that justify each design choice (Section 4) and analyses that uncover MoE behavior (§5). The open logs and intermediate checkpoints further increase confidence (Appendix A).

## 6. Limitations and Trade-offs
- Assumptions and constraints (Section H; §4.1.6–4.1.7; §4.1.4)
  - Need for load‑balancing loss during pretraining constrains the router to use experts roughly equally, which may limit emergent specialization patterns (§4.1.6; authors suggest exploring removal or softening in future).
  - Token‑choice routing (quality‑oriented) is slower than expert‑choice (~24.4k vs ~29.4k tokens/s per device in their setup; §4.1.4).
- Scope not addressed (Section H)
  - Model is text‑only and primarily English; little coverage of multimodal or multilingual scenarios. 
  - Active parameters are limited (~1.3B); while cost‑efficient, this ceiling caps raw capability compared to larger active‑parameter models (Section H).
- Computational/memory trade‑offs
  - MoE has higher total parameter memory (6.9B) even though per‑token compute is small. This increases VRAM requirements and lowers training throughput compared to a dense model with the same active parameters (Fig. 4).
- Data and overtraining considerations
  - The model is substantially “overtrained” relative to classical compute‑optimal scaling (5T tokens) to maximize quality (§2; §4.1.2 notes granularity predictions may not transfer under overtraining). How far overtraining benefits MoEs vs dense remains open (Section H).
- Open questions
  - Can routers work without explicit load‑balancing losses at scale?
  - How do these findings extend to multimodal MoEs or very large expert counts (e.g., 256+)?
  - What is the best way to co‑locate frequently co‑activated experts across devices to cut communication (a deployment‑time optimization hinted at in §5.2)?

## 7. Implications and Future Directions
- Impact on the field
  - Establishes a transparent, strong MoE baseline in the low‑compute regime with complete artifacts (Fig. 1; Appendix A). The extensive ablations and routing analyses offer a de‑facto reference recipe for practitioners and a testbed for researchers.
- Research directions enabled
  - Routing without hard load balancing or with learned capacity; exploration of expert “always‑on” behavior without explicit shared experts (§4.1.3, §4.1.6).
  - Automated data‑mix tuning (given mixed results for adding Reddit/FLAN; Appendix F) and principled overtraining strategies for MoEs (§4.2.1; Section H).
  - Cross‑layer expert coordination and device placement based on co‑activation statistics to reduce communication (Fig. 21; Fig. 35).
  - Multimodal and multilingual MoEs building on this recipe (Section H; §4.1.4 notes expert‑choice might be preferable in some multimodal settings).
- Practical applications
  - Cost‑sensitive deployments (edge servers, limited GPUs) that need better quality than typical ~1B dense models at similar per‑token compute (Table 4).
  - Domains benefiting from expert specialization (code, math, scientific text), as evidenced by domain/vocabulary specializations (Figs. 22–23; Table 8).
  - Instruction‑following assistants where post‑training is crucial: the adapted 0125 model’s strong gains on GSM8k, IFEval, and safety suggest viability for tutoring, coding copilots, and enterprise assistants (Table 17).

> Bottom line: OLMoE contributes not only a strong, economical MoE model, but also a clear, reproducible path to train and analyze MoEs. Its ablations overturn some common intuitions (e.g., shared experts, upcycling) and its analyses reveal early router saturation and strong specialization, guiding both immediate engineering choices and longer‑term research into routing, specialization, and scaling.
