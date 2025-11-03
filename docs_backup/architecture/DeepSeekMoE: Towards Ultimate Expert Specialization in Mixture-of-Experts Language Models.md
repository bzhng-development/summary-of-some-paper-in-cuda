# DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models

**ArXiv:** [2401.06066](https://arxiv.org/abs/2401.06066)
**Authors:** Damai Dai, Chengqi Deng, Chenggang Zhao, R. X. Xu, Huazuo Gao, Deli Chen, Jiashi Li, Wangding Zeng, Xingkai Yu, Y. Wu, Zhenda Xie, Y. K. Li, Panpan Huang, Fuli Luo, Chong Ruan, Zhifang Sui, Wenfeng Liang
**Institutions:** 

## 🎯 Pitch

DeepSeekMoE revolutionizes the Mixture-of-Experts architecture by introducing fine-grained expert segmentation and shared expert isolation, allowing for maximized expert specialization without extra compute cost. This groundbreaking approach not only boosts parameter efficiency and reduces computational demand but also delivers superior task performance compared to larger dense models, marking a significant leap in scalable AI model design.

---

## 1. Executive Summary
DeepSeekMoE introduces a new Mixture‑of‑Experts (MoE) architecture that aims to maximize “expert specialization”—each expert learns distinct, non-overlapping knowledge—without increasing training compute. It achieves this with two mechanisms: fine‑grained expert segmentation (many small experts per layer) and shared expert isolation (a small set of always‑on experts for common knowledge). Experiments from 2B to 145B total parameters show strong accuracy–compute efficiency: at 16B, DeepSeekMoE matches or exceeds LLaMA2‑7B and DeepSeek‑7B on many tasks using roughly 40% of their FLOPs (Tables 3–4; Figure 1).

## 2. Context and Motivation
- Problem addressed
  - Conventional Transformer MoE layers replace each Feed‑Forward Network (FFN) with N experts, and route each token to the top‑K experts by a learned gate (Section 2; Equations 3–5). With typical N in [8,16] and K in {1,2}, two issues limit specialization (Section 1):
    - Knowledge hybridity: too few experts means each expert must model diverse, unrelated content; parameters memorize mixed concepts that are hard to use together.
    - Knowledge redundancy: different experts repeatedly relearn common knowledge, wasting parameters.
- Why it matters
  - Scaling LLMs with dense parameters is compute‑prohibitive; MoE offers parameter growth at fixed per‑token compute. But if experts do not specialize well, MoE underperforms its potential “upper bound” (the accuracy of a dense model with the same total FFN capacity). Improving specialization increases parameter efficiency, accuracy per FLOP, and practical deployability (Sections 1 and 4.3).
- Prior approaches and their gaps
  - Top‑1/Top‑2 routing: Switch Transformer (top‑1), GShard (top‑2) (Section 2). These increase total parameters but activate few experts per token; specialization suffers (knowledge hybridity, redundancy).
  - Fixed routing (Hash Layer) or expert‑choice routing improve stability or flexibility but do not directly target the two core issues at once (Related Work).
- Positioning
  - DeepSeekMoE pursues “ultimate expert specialization” by:
    1) splitting each FFN expert into many smaller experts to expand the combinatorial palette of experts per token without increasing per‑token compute (Section 3.1), and
    2) reserving several always‑on “shared experts” for common knowledge to de‑duplicate what routed experts must learn (Section 3.2).
  - It also adds expert- and device-level balance losses to prevent routing collapse and compute bottlenecks (Section 3.3).

## 3. Technical Approach
This section explains how the model computes, why the new mechanisms help, and how compute stays constant.

- Baseline MoE layer (Section 2)
  - Standard Transformer block: attention then FFN with residual connections (Equations 1–2).
  - Replace FFN with an MoE layer of N experts. For token t at layer l:
    - Compute gate scores `s_i,t = Softmax_i(u_t^lᵀ e_i^l)` (Equation 5), where `u_t^l` is the post‑attention token vector and `e_i^l` is a learned expert “centroid” vector.
    - Pick top‑K experts by `s_i,t`; their gates `g_i,t` keep the softmax value, others are zero (Equation 4).
    - Output sums the selected experts’ FFNs: `h_t^l = Σ_i g_i,t · FFN_i(u_t^l) + u_t^l` (Equation 3).
  - Activating only K experts keeps per‑token compute modest, but specialization is limited.

- Mechanism 1: Fine‑grained expert segmentation (Section 3.1; Equations 6–8)
  - Idea: split each expert into `m` smaller experts by shrinking each expert’s FFN intermediate dimension to `1/m` of the original. This increases the number of experts to `mN` but keeps total parameters unchanged.
  - To maintain per‑token compute, activate `mK` small experts instead of K big ones.
  - New output: `h_t^l = Σ_{i=1}^{mN} g_i,t · FFN_i(u_t^l) + u_t^l` (Equation 6), with `Topk` now selecting `mK` experts (Equation 7).
  - Why it helps:
    - Finer decomposition lets distinct knowledge shards live in different small experts, reducing hybridity.
    - Combinatorial flexibility skyrockets: with `N=16`, `K=2` gives 120 combinations; splitting `m=4` yields `mN=64`, `mK=8` and 4,426,165,368 combinations (Section 3.1). This richer palette increases the chance that a token’s activated set closely matches the knowledge it needs.

- Mechanism 2: Shared expert isolation (Section 3.2; Equations 9–11)
  - Idea: reserve `K_s` small experts per layer as shared, always activated for every token; route the remaining `mK − K_s` activations among the other `mN − K_s` “routed” experts.
  - Output now has two parts: sum of shared experts plus sum of routed experts selected by top‑`mK−K_s` gating (Equation 9).
  - Why it helps:
    - Shared experts concentrate ubiquitous knowledge (syntax, frequent facts, common patterns), reducing the pressure to relearn it across many routed experts (redundancy).
    - Routed experts can then specialize on distinctive knowledge, increasing parameter efficiency.

- Load balancing to avoid routing collapse and device hotspots (Section 3.3; Equations 12–17)
  - Expert‑level balance loss `LExpBal = α1 Σ_i' f_i P_i` (Equation 12):
    - `f_i` tracks how often Expert i is selected; `P_i` tracks average gate magnitude (Equations 13–14).
    - Encourages usage dispersion across experts, mitigating collapse to a few experts.
  - Device‑level balance loss `LDevBal = α2 Σ_d f'_d P'_d` (Equations 15–17):
    - Groups experts by device and balances aggregate usage, addressing pipeline bottlenecks when experts span devices (used at 145B scale; Section 7.1).
  - Practical settings:
    - Small `α1` prevents collapse without over‑constraining routing (e.g., 0.01 at 2B; 0.001 at 16B; 0.003 at 145B).
    - Larger `α2` (e.g., 0.05) at very large scales ensures device compute balance (Section 7.1).

- Putting it together (Figure 2)
  - (a) Conventional top‑2 routing.
  - (b) Add fine‑grained segmentation: more, smaller experts; more activations; same compute.
  - (c) Add shared expert isolation: some always‑on shared experts plus routed small experts; still same total expert parameters and activated compute per token.

- Implementation/configurations (Sections 4.1, 5.1, 7.1)
  - 2B scale (validation): 9 layers, hidden 1280; all FFNs are MoE; total expert params = 16× a standard FFN; activated expert params = 2× FFN; ≈2.0B total params, ≈0.3B activated (Section 4.1.3).
  - 16B scale: 28 layers, hidden 2048; all but the first layer use MoE (first converges slower for balance). Each MoE layer: 2 shared + 64 routed experts; each expert 0.25× FFN size; route to 2 shared + 6 routed (Section 5.1.2). ≈16.4B total; ≈2.8B activated.
  - 145B scale (preliminary): 62 layers, hidden 4096; 4 shared + 128 routed; each expert 0.125× FFN; route to 4 shared + 12 routed; ≈144.6B total; ≈22.2B activated (Section 7.1).

## 4. Key Insights and Innovations
- Fine‑grained expert segmentation is a structural change, not just a hyper‑parameter tweak.
  - Novelty: expands the activation combination space by orders of magnitude without increasing compute (Section 3.1). This is a fundamental innovation in how MoE capacity is arranged.
  - Impact: empirically improves accuracy across tasks (Table 1; Figure 3).
- Shared expert isolation targets redundancy directly.
  - Different from engineering‑motivated “shared experts” used to ease inference (cf. Rajbhandari et al., 2022). Here it is used algorithmically to compress common knowledge and free routed experts to specialize (Section 3.2).
  - Impact: replacing one routed activation with one shared activation tends to improve scores (Figure 3; see the “+ shared expert isolation” curve).
- Two‑level load balancing aligns accuracy and efficiency at scale.
  - Expert‑level balance prevents collapse; device‑level balance removes multi‑GPU bottlenecks (Section 3.3). This is an incremental but necessary ingredient for stable scaling to 145B.
- Near‑upper‑bound MoE performance at small scale.
  - At ≈2B total params, DeepSeekMoE’s performance is nearly that of a dense model with 16× FFN capacity (Table 2, “Dense×16” vs “DeepSeekMoE”). This suggests specialization is sufficiently high to approach the MoE capacity limit defined by the dense counterpart (Section 4.3).
- Activated‑parameter efficiency.
  - DeepSeekMoE often matches larger baselines with fewer activated parameters per token (e.g., at 16B vs LLaMA2‑7B; Table 4) and can keep accuracy even when halving activated routed experts (Figures 5–6). This reflects the practical benefit of better specialization.

## 5. Experimental Analysis
- Evaluation methodology
  - Datasets and metrics
    - Pretraining: bilingual (English/Chinese‑centric) multi‑domain corpus (web, math, code, literature) (Section 4.1.1; 5.1.1).
    - Language modeling: Pile loss or bits per byte (BPB) (Tables 1, 3–4).
    - Understanding/reasoning: HellaSwag, PIQA, ARC‑easy/challenge, RACE (Tables 1, 3–4).
    - Math: GSM8K, MATH (Tables 3–4).
    - Code: HumanEval, MBPP (Tables 1, 3–4).
    - Closed‑book QA: TriviaQA, NaturalQuestions (Tables 1, 3–4).
    - Multi‑subject: MMLU (Tables 3–4).
    - Disambiguation: WinoGrande; Chinese: CLUEWSC, CEval, CMMLU, CHID (Tables 3–4).
    - Open LLM Leaderboard: ARC, HellaSwag, MMLU, TruthfulQA, Winogrande, GSM8K (Figure 1).
  - Baselines
    - Dense Transformer; Hash Layer (top‑1 hash), Switch (top‑1 learned), GShard (top‑2 learned), and larger “×1.2/×1.5” GShard and dense models (Tables 1–2, Appendix B).
    - Strong external baselines: LLaMA2‑7B and DeepSeek‑7B dense models trained on 2T tokens (Tables 3–4).
  - Compute accounting
    - FLOPs per sequence and activated parameters are reported for fairness (Tables 1–4, 6). E.g., at 4K tokens, DeepSeekMoE‑16B uses 74.4T FLOPs vs LLaMA2‑7B’s 187.9T (Table 4).

- Main quantitative results
  - At ≈2B total parameters (Table 1):
    > DeepSeekMoE achieves Pile loss 1.808, HellaSwag 54.8, PIQA 72.3, ARC‑challenge 34.3, TriviaQA EM 16.6; all higher than GShard with the same compute (GShard Pile 1.867; HellaSwag 50.5; PIQA 70.6; ARC‑challenge 31.6; TriviaQA 10.2).
  - Near MoE upper bound (Table 2):
    > DeepSeekMoE matches a 1.5× larger GShard (e.g., Pile 1.808 vs 1.808; HellaSwag 54.8 vs 54.4) while using only 2/3 of its expert compute. It also nearly matches “Dense×16” (Pile 1.808 vs 1.806; HellaSwag 54.8 vs 55.1).
  - Ablations verify each mechanism (Figure 3):
    > Starting from GShard (0 shared + 2/16 routed), adding 1 shared expert improves normalized performance; further splitting experts (to 31 then 63 routed) keeps improving, with constant total/activated parameters.
  - Specialization diagnostics (Figures 4–6):
    > Disabling top routed experts harms DeepSeekMoE more than GShard×1.5 (Figure 4), indicating lower redundancy—each expert carries unique knowledge.
    > DeepSeekMoE attains GShard‑level Pile loss with only 4 activated routed experts (vs 7), demonstrating more accurate knowledge acquisition per activation (Figure 5). Training a model from scratch with half the activated routed experts still outperforms GShard on multiple benchmarks (Figure 6).
  - 16B scale: internal comparisons (Table 3)
    > With only 40.5% of DeepSeek‑7B’s compute (74.4T vs 183.5T FLOPs per 4K tokens), DeepSeekMoE‑16B has comparable or better performance on many tasks: Pile BPB 0.74 vs 0.75; HellaSwag 77.1 vs 75.4; PIQA 80.2 vs 79.2; TriviaQA 64.8 vs 59.7; NaturalQuestions 25.5 vs 22.2. It lags on MMLU (45.0 vs 48.2) and some multiple‑choice tasks—consistent with having fewer attention parameters (Section 5.2.1).
  - 16B scale: external comparisons (Table 4; Figure 1)
    > Versus LLaMA2‑7B (same 2T tokens), DeepSeekMoE‑16B uses 39.6% compute and achieves better or similar scores on most tasks: HellaSwag 77.1 vs 75.6; ARC‑challenge 49.8 vs 49.0; GSM8K 18.8 vs 15.5; HumanEval 26.8 vs 14.6; MBPP 39.2 vs 21.8; Chinese tasks are much higher due to bilingual pretraining (e.g., CHID 89.4 vs 37.9). The Open LLM Leaderboard plot (Figure 1) shows DeepSeekMoE‑16B as an outlier above the trend line for its activated parameter count.
  - Alignment (SFT) results (Table 5)
    > After 1.4M example SFT, DeepSeekMoE‑16B Chat matches the two 7B dense chat baselines on most tasks with ~40% compute, and substantially outperforms LLaMA2 SFT‑7B on code (HumanEval 45.7 vs 35.4; MBPP 46.2 vs 27.8).
  - 145B preliminary study (Table 6)
    > DeepSeekMoE‑145B clearly outperforms GShard‑137B at similar scale (e.g., Pile loss 1.876 vs 1.961; HellaSwag 75.8 vs 72.0; TriviaQA 61.1 vs 52.5), and reaches performance comparable to DeepSeek‑67B dense while using 28.5% of its compute. A half‑activated variant (DeepSeekMoE‑142B, 12.2B activated params) still matches or exceeds DeepSeek‑67B on multiple tasks with only 18.2% of its compute.

- Do the experiments support the claims?
  - Yes, across multiple scales, datasets, and baseline types:
    - Mechanism‑level ablations (Figure 3) isolate the effect of shared experts and segmentation.
    - Specialization diagnostics (Figures 4–6) probe redundancy and activation efficiency.
    - Compute‑normalized comparisons report FLOPs and activated parameters (Tables 1–4, 6).
  - Caveats:
    - Some benchmarks (MMLU‑like) favor higher attention capacity; MoE models with fewer attention parameters underperform there (Table 3 discussion).
    - External comparisons (e.g., to LLaMA2‑7B) control token count (2T) but training corpus composition differs; bilingual and code‑rich data likely boost DeepSeekMoE on Chinese and code tasks (Table 4, Section 5.2.2).

## 6. Limitations and Trade-offs
- Attention capacity vs FFN capacity
  - DeepSeekMoE allocates most capacity to FFNs (experts) and less to attention. This hurts tasks that rely on broad context integration and multiple‑choice reasoning (e.g., MMLU) (Table 3). The authors note similar struggles when attention is reduced (e.g., multi‑query attention variants; Section 5.2.1).
- Routing stability and balance
  - Although balance losses mitigate collapse, tuning `α1`/`α2` is delicate: too strong harms accuracy; too weak risks imbalance (Section 3.3). The first layer converged slowly for balance, so it remains dense at 16B (Section 5.1.2).
- Computational overheads not captured by FLOPs
  - Many tiny experts increase kernel launch/scheduling overhead. The 16B model avoided even finer segmentation because excessively small experts reduced efficiency (Section 5.1.2). At 145B, device‑level balancing and expert parallelism complicate training (Section 7.1).
- Data dependence
  - Gains on code/math and Chinese benchmarks stem partly from corpus composition (Sections 5.1.3 and 5.2.2). Results may shift with different domain mixes.
- Upper‑bound claim is scale‑ and setup‑dependent
  - The “near upper bound” observation (Table 2) is at ~2B parameters and 100B tokens; it is not theoretically proven to hold universally, and dense models with different attention/FFN ratios could change the bound (Section 4.3).

## 7. Implications and Future Directions
- How this changes the field
  - Demonstrates that architecting for specialization—via many small, combinable routed experts plus always‑on shared experts—can translate to consistent accuracy and compute savings across scales. It reframes MoE design from “more experts, same K” to “finer experts, larger K with shared cores.”
  - Shows that MoE can be a practical alternative to dense models for pretraining and instruction‑tuning, not just for scaling parameter counts on paper (Tables 3–5). The released 16B checkpoint runs on a single 40GB GPU (Abstract; Section 6, “Public Release”).
- Follow‑up research enabled
  - Adaptive activation budgets: dynamically choose `mK` per token based on uncertainty or difficulty.
  - Learned shared/routed partitioning: let the model decide which experts should be shared vs routed instead of fixing `K_s`.
  - Cross‑layer specialization: share or reuse expert identities across layers for hierarchical specialization.
  - Attention–FFN rebalancing: explore higher attention capacity in MoE settings to close the gap on MMLU‑like tasks while retaining FFN specialization.
  - Routing strategies: combine fine-grained segmentation with expert‑choice routing, token dropping/capacity control, or retrieval‑augmented memory.
- Practical applications
  - Cost‑effective language model pretraining/fine‑tuning when compute per token is limited.
  - Domains with heterogeneous skills (code, math, multilingual): routed experts can specialize by domain, with shared experts anchoring common language knowledge.
  - Deployment scenarios needing lower latency per token: fewer activated parameters and efficient kernels can yield faster inference than dense peers of similar quality (Section 5.2.1 notes ~2.5× speedup vs 7B dense with operator optimizations).

Overall, DeepSeekMoE provides concrete mechanisms that measurably increase expert specialization under fixed compute by (i) drastically growing the mix‑and‑match space of small experts and (ii) carving out shared capacity for ubiquitous knowledge. The combination is validated by ablations (Figure 3), specialization diagnostics (Figures 4–6), and strong compute‑normalized benchmarks from 2B to 145B parameters (Tables 1–2, 3–6; Figure 1).
