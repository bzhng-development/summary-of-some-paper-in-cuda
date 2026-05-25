# DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models

**ArXiv:** [2401.06066](https://arxiv.org/abs/2401.06066)

## 🎯 Pitch

DeepSeekMoE pioneers a novel Mixture-of-Experts (MoE) architecture by introducing fine-grained expert segmentation and a set of always-on shared experts to drive ultimate expert specialization. This design overcomes key limitations of previous MoE models by decomposing knowledge more precisely and reducing redundancy, enabling each expert to focus on distinct aspects of the data. As a result, DeepSeekMoE achieves near-dense model performance at a fraction of the computational cost, scaling to hundreds of billions of parameters while delivering superior efficiency and broader accessibility for large language models.

---

## 1. Executive Summary
DeepSeekMoE introduces a Mixture-of-Experts (MoE) architecture that explicitly pushes experts to specialize by combining two ideas: splitting experts into many small pieces (“fine‑grained expert segmentation”) and always enabling a small set of “shared” experts that hold common knowledge. Across scales (2B→16B→145B parameters), this design matches or surpasses prior MoE systems (e.g., GShard) at the same compute, approaches dense-model upper bounds, and delivers competitive performance to strong 7B dense baselines while using only ~40% of the compute (Tables 3–4, Figure 1).

## 2. Context and Motivation
- Problem addressed
  - In standard MoE Transformers, each token is sent (routed) to the top‑K of N experts. With few large experts, two issues arise (Section 1):
    - Knowledge hybridity: one expert must hold heterogeneous knowledge because it sees diverse tokens.
    - Knowledge redundancy: different experts repeatedly learn the same common knowledge because many tokens need it.
- Why it matters
  - Scaling LLMs improves capability but is compute‑expensive. MoE keeps compute nearly constant by only activating a subset of parameters per token, but only if experts specialize well so that activated compute is used efficiently (Section 1).
- Prior approaches and their limits
  - GShard (top‑2 routing), Switch Transformer (top‑1), Hash Layer (fixed hashing) increase parameter counts with sparse activation but still route to a small set of large experts, leaving hybridity/redundancy unresolved (Sections 1–2; Equations 3–5).
  - Shared experts have been used as an engineering optimization (DeepSpeed‑MoE; cited in Section 3.2), but not as an algorithmic means to reduce redundancy.
- Positioning
  - DeepSeekMoE is an architectural redesign specifically targeting “expert specialization” by:
    - Increasing the granularity and combinatorial flexibility of which experts can be activated for a token.
    - Removing common knowledge from routed experts via always‑on shared experts.
  - It demonstrates specialization gains through ablations and stress tests and scales effectively to large models (Sections 4–7).

## 3. Technical Approach
Key terms
- `Expert`: a module structurally identical to an FFN (feed‑forward network) inside a Transformer block.
- `Router`: the module that assigns each token to experts via a softmax over expert “affinity” scores and chooses the top‑K experts to activate (Equations 3–5).
- `Top‑K routing`: per token, activate only the K experts with highest affinity.

3.1 Preliminaries: standard MoE in Transformers (Section 2)
- A Transformer block applies self‑attention then an FFN. MoE replaces some FFNs with an MoE layer.
- For a token at layer l with hidden state u_l_t, the layer computes a weighted sum over experts:
  - h_l_t = sum over i of g_i,t · FFN_i(u_l_t) + u_l_t (Equation 3).
  - g_i,t is nonzero only for K experts with highest routing score s_i,t (Equations 4–5), ensuring sparse activation and thus compute efficiency.

3.2 Fine‑grained expert segmentation (Section 3.1; Figure 2b; Equations 6–8)
- Idea: Instead of N large experts, split each expert into m smaller experts by shrinking each FFN’s intermediate dimension by 1/m. Keep total parameters and total compute constant by:
  - Increasing the total number of experts to mN and the number of activated experts to mK.
  - Each token now selects mK smaller experts instead of K large experts.
- Why it helps:
  - More “slots” to place distinct knowledge reduces hybridity—specialized sub‑experts can focus on finer topics.
  - Combinatorial flexibility: with N=16 and K=2, standard routing yields C(16,2)=120 combinations. With m=4 (64 experts) and mK=8, combinations jump to C(64,8)=4,426,165,368 (Section 3.1). This lets the router assemble a tailored mixture for each token.
- Mechanics:
  - The MoE output becomes h_l_t = sum_{i=1..mN} g_i,t · FFN_i(u_l_t) + u_l_t (Equation 6) with g_i,t selecting top mK out of mN (Equation 7).

3.3 Shared expert isolation (Section 3.2; Figure 2c; Equations 9–11)
- Idea: Dedicate K_s experts that are always activated for every token (“shared experts”) to hold common knowledge and features (e.g., basic syntax, common phrases, arithmetic).
- Keep compute constant by reducing routed activations from mK to (mK−K_s). The final layer output:
  - h_l_t = sum_{i=1..K_s} FFN_i(u_l_t) + sum_{i=K_s+1..mN} g_i,t · FFN_i(u_l_t) + u_l_t (Equation 9).
- Why it helps:
  - Removes redundancy: routed experts no longer need to relearn fundamentals; they specialize on non‑overlapping, rarer knowledge.
  - The design is algorithmic: shared experts intentionally absorb ubiquitous patterns to free routed experts for specialization.

3.4 Load balancing for stability and efficiency (Section 3.3; Equations 12–17)
- Two balancing losses prevent routing collapse and device hot‑spots:
  - Expert‑level balance: encourages tokens to distribute across routed experts by penalizing the product of each expert’s load fraction f_i and average probability P_i (Equations 12–14). A small factor α1 avoids harming performance.
  - Device‑level balance: when experts are sharded across devices, group them per device (E_1..E_D) and balance average load across devices via f'_i and P'_i (Equations 15–17). A larger α2 reduces system bottlenecks.
- Practical tuning:
  - Small α1 to prevent collapse; larger α2 to balance compute across devices (Section 3.3).
  - For 2B and 16B setups without expert parallelism, device‑level loss is unnecessary (Sections 4.1.3, 5.1.2). For 145B with expert parallelism, α2=0.05 is used (Section 7.1).

3.5 Implementation and training (Sections 4.1–5.1, 7.1)
- Data: multilingual, English + Chinese with code and math; 100B tokens for 2B validation, 2T for 16B, 245B for 145B preliminary run (Sections 4.1.1, 5.1.1, 7.1).
- Systems: HAI‑LLM framework with tensor/data/pipeline/expert parallelism and custom CUDA/Triton kernels (Section 4.1.2).
- Representative model settings
  - 2B: 9 layers; 1 shared + 63 routed; activate 1 shared + 7 routed; expert size 0.25× FFN (Section 4.1.3).
  - 16B: 28 layers; 2 shared + 64 routed; activate 2 + 6; expert size 0.25×; sequence length 4K; 2T tokens (Sections 5.1.2–5.1.3).
  - 145B: 62 layers; 4 shared + 128 routed; activate 4 + 12; expert size 0.125×; 245B tokens preliminary (Section 7.1).

Analogy
- Think of shared experts as a “core curriculum” every student must take. Routed experts are electives. Fine‑grained segmentation makes electives smaller and more numerous, so each student can pick a tailored set of classes that fit their needs precisely.

## 4. Key Insights and Innovations
- Fine‑grained expert segmentation (Section 3.1; Figure 2b; Equations 6–8)
  - What’s new: increase expert granularity and the number of activated experts while holding total parameters and FLOPs constant.
  - Why it matters: massively increases the space of expert combinations, enabling more precise token‑to‑knowledge matching. This is a substantive architectural change rather than a training trick.
- Shared expert isolation (Section 3.2; Figure 2c; Equations 9–11)
  - What’s new: algorithmically reserve always‑on experts for common knowledge to reduce redundancy among routed experts.
  - Why it matters: boosts parameter efficiency and makes routed experts more specialized. Empirically crucial—disabling the shared expert increases Pile loss from 1.808 to 2.414 at 2B scale (Section 4.5).
- Explicit specialization evidence, not just performance (Section 4.5; Figures 4–6)
  - DeepSeekMoE is more sensitive when top routed experts are masked (Figure 4): Pile loss rises faster than GShard×1.5, meaning top experts are less interchangeable—an indicator of stronger specialization and lower redundancy.
  - With fewer activated experts (only 4 routed), DeepSeekMoE still matches GShard’s Pile loss (Figure 5), showing more “accurate” knowledge acquisition per activation.
  - A model trained from scratch with only half the activated routed experts still outperforms GShard on downstream tasks (Figure 6).
- Approaching dense upper bounds at small scale (Table 2)
  - With the same total parameters as GShard 2B, DeepSeekMoE achieves performance comparable to GShard×1.5 (1.5× expert parameters and compute) and nearly matches a dense model with 16× FFN parameters (“Dense×16”), which is an upper bound on MoE capacity at this depth/width.

## 5. Experimental Analysis
Evaluation setup
- Benchmarks span language modeling (Pile), understanding/reasoning (HellaSwag, PIQA, ARC‑Easy/Challenge), reading comprehension (RACE‑middle/high, DROP), code (HumanEval, MBPP), QA (TriviaQA, NaturalQuestions), math (GSM8K, MATH), multi‑subject MC (MMLU), disambiguation (WinoGrande), and Chinese benchmarks (CLUEWSC, CEval, CMMLU, CHID) (Sections 4.1.4, 5.1.3).
- Metrics: cross‑entropy or bits‑per‑byte (BPB) for language modeling; accuracy for multiple‑choice; EM for QA; Pass@1 for code (Sections 4.1.4, 5.1.3).

Main findings
- 2B validation scale (Table 1)
  - Quote: “DeepSeekMoE… has 2.0B total parameters… GShard has the same activated parameters” (Section 4.2).
  - Results vs GShard at equal compute (4.3T FLOPs/2K tokens):
    - Pile loss 1.808 vs 1.867.
    - HellaSwag 54.8 vs 50.5; PIQA 72.3 vs 70.6.
    - TriviaQA EM 16.6 vs 10.2; NQ EM 5.7 vs 3.2.
  - Takeaway: consistent gains across diverse tasks with the same total and activated parameters.
- Approaching larger baselines (Table 2)
  - Comparable to GShard×1.5 (higher expert size/compute) across most tasks:
    - HellaSwag 54.8 vs 54.4; PIQA 72.3 vs 71.1; HumanEval 4.9 vs 3.0; TriviaQA 16.6 vs 15.7.
  - Nearly matches Dense×16 (16× FFN parameters): Pile loss 1.808 vs 1.806; many task scores within noise.
- Ablations validate both components (Figure 3)
  - Starting from GShard (0 shared + 2/16 routed), adding shared expert improves performance; further splitting to 32 then 64 experts (with the same total/activated params) further improves normalized performance across six benchmarks.
  - Ratios between shared and routed experts (Section 4.4): 1, 2, or 4 shared experts give similar Pile losses (1.808, 1.806, 1.811); the paper later adopts a 1:3 ratio of shared:routed activations when scaling.
- Specialization diagnostics (Section 4.5; Figures 4–6)
  - Disable top routed experts: DeepSeekMoE’s Pile loss degrades faster than GShard×1.5 (Figure 4) → less redundancy/more specialization.
  - Fewer activated routed experts: at 4 routed experts, DeepSeekMoE ≈ GShard on Pile (Figure 5); a half‑activation model trained from scratch beats GShard on downstream tasks (Figure 6).
- 16B scale on 2T tokens (Section 5; Tables 3–4; Figure 1)
  - Architecture: 2 shared + 64 routed, activate 2+6; total params 16.4B; activated ~2.8B; 74.4T FLOPs per 4K tokens (Section 5.1.2).
  - Versus internal dense model DeepSeek 7B (same 2T data; 183.5T FLOPs):
    - Quote (Table 3): “With only 40.5% of computations, DeepSeekMoE 16B achieves comparable performance with DeepSeek 7B.”
    - Examples: HellaSwag 77.1 vs 75.4; PIQA 80.2 vs 79.2; TriviaQA 64.8 vs 59.7; NaturalQuestions 25.5 vs 22.2; HumanEval 26.8 vs 26.2.
    - Weakness: some multiple‑choice tasks (e.g., MMLU 45.0 vs 48.2), attributed to fewer attention parameters in the MoE model (Section 5.2.1).
    - Practical note: fits inference on a single 40GB GPU and runs ~2.5× faster than a 7B dense model with optimized kernels (Section 5.2.1).
  - Versus LLaMA2 7B (2T tokens; 187.9T FLOPs):
    - Quote (Table 4): “With only 39.6% of computations, DeepSeekMoE 16B outperforms LLaMA2 7B on the majority of benchmarks.”
    - Examples: HellaSwag 77.1 vs 75.6; ARC‑Challenge 49.8 vs 49.0; GSM8K 18.8 vs 15.5; HumanEval 26.8 vs 14.6; MBPP 39.2 vs 21.8.
    - Chinese benchmarks: very large gains (e.g., CHID 89.4 vs 37.9) because DeepSeekMoE is bilingual (Table 4).
    - Open LLM Leaderboard (Figure 1): strong average performance relative to models with similar activated parameter counts; comparable to LLaMA2‑7B with ~2.5× fewer activated parameters.
- Alignment via supervised fine‑tuning (SFT) (Section 6; Table 5)
  - Setup: 1.4M bilingual SFT examples; same SFT data for all models; 4K max length (Section 6.1).
  - Results: at ~40% compute, DeepSeekMoE Chat 16B is comparable to dense 7B models across reasoning, reading comprehension, math, and QA; strong on code (HumanEval 45.7, MBPP 46.2) and Chinese tasks; still behind on some multiple‑choice tasks (MMLU 47.2 vs DeepSeek Chat 7B’s 49.7).
- 145B preliminary scaling (Section 7; Table 6)
  - Model: 4 shared + 128 routed; activate 4+12; total 144.6B params; ~22.2B activated; trained 245B tokens.
  - Versus GShard 137B (similar total params/FLOPs): DeepSeekMoE 145B wins widely (e.g., Pile 1.876 vs 1.961; TriviaQA 61.1 vs 52.5).
  - Versus DeepSeek 67B Dense (2057.5T FLOPs) at only 28.5% compute (585.6T FLOPs): achieves comparable overall performance, stronger on LM and knowledge tasks, weaker on some MC tasks (Table 6).
  - A “Half Activated” variant (142B total; 12.2B activated; 374.6T FLOPs) still matches the 67B dense baseline on many tasks and beats GShard 137B (Table 6), echoing specialization efficiency.

Assessment
- The experiments are broad (English, Chinese, code, math), controlled (shared data for critical comparisons), and include ablations and diagnostics that directly test “specialization.” The evidence convincingly supports both the architectural claims and the efficiency claims, with clearly documented trade‑offs (multiple‑choice).

## 6. Limitations and Trade-offs
- Multiple‑choice weakness linked to attention capacity (Section 5.2.1)
  - DeepSeekMoE 16B uses fewer attention parameters (~0.5B) than comparable dense models (e.g., DeepSeek 7B has ~2.5B). This correlates with lower MMLU/CEval/CMMLU performance (Tables 3, 5).
- Efficiency limits of extreme segmentation (Section 5.1.2)
  - The paper avoids even finer segmentation at 16B “due to the potential reduction in computational efficiency associated with excessively small expert sizes.”
- Load‑balancing hyperparameters (Section 3.3, 5.1.2, 7.1)
  - Balance losses require careful tuning; too strong expert‑level balance can hurt model quality; too weak can cause routing collapse or device hot‑spots.
- Training scope for the 145B model (Section 7.1)
  - Preliminary run on 245B tokens without full convergence; results promising but not a finished model.
- Data distribution differences (Tables 3–4)
  - DeepSeekMoE uses a bilingual corpus with substantial math/code; comparisons to monolingual or differently curated datasets (e.g., LLaMA2’s) can favor certain tasks.
- Assumption that “shared knowledge” is universal across contexts
  - Shared experts are always on. If some “common” patterns vary by domain or language, this could waste activation budget or entangle language‑specific fundamentals; the ratio choice is empirical (1:3 shared:routed activations, Section 4.4).

## 7. Implications and Future Directions
- Field impact
  - DeepSeekMoE reframes MoE design around specialization rather than just parameter scaling. The combination of fine‑grained routing and shared experts demonstrates that sparse models can approach dense capacity at much lower compute (Table 2) and compete with strong 7B dense models at ~40% FLOPs (Tables 3–4).
  - Practical deployment matters: a 16B total‑parameter MoE that fits on a single 40GB GPU with faster inference than 7B dense is attractive for industry (Section 5.2.1).
- Research directions
  - Adaptive shared experts: learn which shared experts to enable per domain/language rather than “always on,” or dynamically vary the shared:routed ratio per layer.
  - Hierarchical or multi‑stage segmentation: split experts unevenly or conditionally to match knowledge granularity across layers or tasks.
  - Attention–FFN balance: increase attention capacity within the MoE budget to address multiple‑choice weaknesses, or explore attention‑side sparsity to keep compute low.
  - Better routing objectives: incorporate task signals or uncertainty to decide mK per token dynamically, extending Figure 5’s observation that fewer routed experts can suffice.
  - Specialization diagnostics: formalize redundancy/specialization measures beyond the masking tests (Figure 4), e.g., mutual information between expert activations and token attributes.
- Applications
  - Cost‑effective serving of bilingual assistants and code/maths copilots (Tables 3–5).
  - Domain‑specialized LLMs where shared experts capture universal linguistic patterns and routed experts focus on domain idiosyncrasies (medical, legal, scientific).
  - Edge/cloud hybrid inference: small activated footprints and single‑GPU feasibility enable latency‑sensitive applications.

Overall, DeepSeekMoE shows that careful architectural choices—more granular experts plus always‑on shared experts—can robustly improve expert specialization and deliver strong accuracy–compute trade‑offs, validated by targeted ablations and large‑scale comparisons (Figures 1–6; Tables 1–6, 8–10).
