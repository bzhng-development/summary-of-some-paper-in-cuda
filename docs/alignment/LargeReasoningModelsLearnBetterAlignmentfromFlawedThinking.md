# Large Reasoning Models Learn Better Alignment from Flawed Thinking

**ArXiv:** [2510.00938](https://arxiv.org/abs/2510.00938)

## 🎯 Pitch

This paper introduces RECAP, a reinforcement learning method that explicitly trains large reasoning models (LRMs) to override flawed or unsafe chain-of-thought (CoT) reasoning and reroute to safe, helpful responses. By systematically exposing models to counter-aligned, misleading reasoning during training, RECAP dramatically boosts robustness against prompt attacks and jailbreaks, reduces needless refusals, and even enhances core reasoning abilities—all without added training cost or inference-time overhead. This approach directly addresses a critical vulnerability in modern AI deployments, ensuring safer and more resilient real-world large language models.

---

## 1. Executive Summary
This paper introduces RECAP, a reinforcement learning (RL) training recipe that teaches large reasoning models (LRMs) to override misleading chain-of-thought (CoT) and produce safe, helpful answers. By deliberately prefilling models with counter-aligned reasoning during training—and optimizing only the continuation—RECAP substantially improves safety and jailbreak robustness, reduces overrefusal on benign prompts, preserves math capability, and keeps inference token budgets comparable (Sec. 1, Fig. 1; Sec. 4.2–4.3).

## 2. Context and Motivation
- Problem addressed
  - LRMs often “think out loud” through chain-of-thought (CoT) before answering. A simple “prefill” (injecting a reasoning prefix) can steer their internal reasoning and final output. The paper shows that starting from flawed CoT causes models to “follow without thinking,” forgetting safety and producing unsafe or incorrect completions (Sec. 2).
  - This brittleness is acute for safety alignment: injecting a short unsafe CoT fragment or a “confident” preface can bypass alignment and trigger harmful responses, even in models tuned with RLHF (reinforcement learning from human feedback) (Sec. 1; Sec. 2.2).

- Why it matters
  - Prefilling is widely exposed in APIs and open-source stacks, so adversaries can exploit it at inference time (Sec. 1). Robustness against CoT steering is crucial for real-world deployment where prompts, tools, or agents may introduce misleading intermediate steps.

- Prior approaches and gaps
  - SFT-based safety datasets and reasoning-aligned supervision improve average safety but still optimize the final answer rather than teaching recovery from flawed intermediate reasoning (Sec. 6, App. I).
  - RLHF variants (e.g., GRPO, DAPO) typically reward final outputs and do not expose models to counter-aligned CoT states they will face at inference time. Guardrail models or runtime reflection can help but rely on external components or on-the-fly interception (Sec. 6).
  - The paper documents the gap: when prefilling other models with unsafe CoT, safety drops dramatically across architectures. Example: taking the first ~200 words of unsafe CoT from `DSQwen-1.5B` reduces other DS models’ safety by an average of 36.4%; conversely, prefilling with the safest model’s CoT increases safety by 91.7% (Table 1, Sec. 2.2). The same brittleness appears for math and overrefusal (App. B, Tables 5–6).

- Positioning
  - RECAP integrates counter-aligned prefills directly into online RL training, forcing the policy to override flawed trajectories to earn high reward (Sec. 3.1). It needs no changes to the RLHF objective and introduces no extra sampling cost beyond standard RLHF (Sec. 1, 3.1).

## 3. Technical Approach
Key terms used once and then assumed:
- `LRM` (large reasoning model): an LLM that generates intermediate CoT before the final answer.
- `CoT prefilling` (“prefill”): injecting a pre-generated reasoning prefix into the model’s CoT before it continues.
- `Overrefusal`: the model wrongly refuses benign requests due to being overly cautious.
- `DAPO`: Dynamic-sampling policy optimization, an online RLHF variant improving stability and sample efficiency (Sec. 3.1, 4.1).
- `GRPO`: Group-relative policy optimization, another RLHF variant (Sec. 1).
- `RLVR`: Reinforcement Learning with Verifiable Rewards for math solutions (Sec. 4.1).

Step-by-step methodology
1. Construct counter-aligned prefills (Sec. 3.1; Fig. 1)
   - For a harmful prompt x, sample an unsafe reasoning prefix `y_pre_cot` from a weakly aligned or “helpful-only” model `π_harm`. This prefix is plausible text but semantically unsafe.
   - For a benign prompt, sample a refusal-oriented prefix from an overly conservative model `π_refuse` that tends to reject even benign inputs.
   - Intuition: blindly following these prefixes would earn low reward—unsafe on harmful prompts, unhelpful on benign ones—so high reward requires actively overriding them and steering back to safe/helpful reasoning.

2. Mix prefilled and standard prompts
   - Apply prefilling to a fraction `α` of the training examples; do not prefill the rest (Sec. 3.1). The paper’s default uses `α = 0.5` for safety prompts and a prefill length `ℓ_pre = 500` words (Sec. 5.1).

3. Rollouts and optimization only after the prefix
   - The model receives `(x, y_pre_cot)` and must generate the continuation `y_gen_cot` and final response `y_resp`.
   - In the DAPO objective, importance ratios and advantages are computed only for tokens after the prefill; i.e., the optimization starts at `t0(x) = ℓ_pre + 1` for prefilled samples and `t0(x)=1` otherwise (Sec. 3.1, objective J_RECAP). This prevents the learner from “rewriting” the prefix and focuses learning on recovery behaviors.

4. Rewards and multi-task RL setup (Sec. 4.1)
   - Safety rewards: logits from the `Granite-Guardian-3.1-8B` guardrail model provide dense signals (more informative than binary labels).
   - Overrefusal rewards: a rubric judged by `Llama-3.1-8B-Instruct` scores whether the response appropriately complies with a benign request (App. D shows the rubric).
   - Math rewards: RLVR provides verifiable correctness signals (e.g., checking final numeric answers).
   - The training corpus mixes domains (5K prompts total): ∼1K harmful (BeaverTails), ∼1K overrefusal (STAR‑1), and ∼3K math (GSM8K + MATH). Sixteen rollouts per prompt are sampled under DAPO (Sec. 4.2, 4.1).

5. Theoretical rationale (Sec. 3.2; App. C)
   - Define two evaluation distributions: `Jclean(π)` without any prefix and `Jpre(π)` when evaluation starts from (possibly adversarial) prefixes. The combined evaluation `J(π) = (1−β)Jclean + βJpre` reflects realistic conditions where some interactions include prefilling.
   - Theorem 1 (plain-English summary): if (i) policy updates are conservative (a standard assumption in clipped policy optimization), (ii) RECAP is not worse than DAPO on clean data up to a small slack, and (iii) DAPO makes little or no progress on prefilled states because it never trains on them, then after T steps the expected evaluation reward gap satisfies
     - Improvement ≥ β × (cumulative per-step gains on prefilled starts − bounded slacks).
   - Meaning: training on counter-aligned prefixes gives RECAP a systematic advantage whenever evaluation includes prefills (β > 0). The advantage scales with how much the policy learns to recover from unsafe prefixes (γ_pre) and persists even when there is no prefix at inference, given the clean-parity assumption (Sec. 3.2; App. C).

Design choices and why
- Counter-aligned (unsafe/refusal) prefixes rather than safe ones: ablations show that safe prefills encourage exploitation (the model “coasts” on the already-correct prefix), whereas counter-aligned prefills force corrective behavior and yield higher robustness (Sec. 5.1, Fig. 3c; Table 10).
- Intermediate prefill ratio and length: too little exposure undertrains recovery; too much makes the model rely on prefixes. The best trade-offs appear around `α=0.5` and `ℓ_pre≈500` words (Sec. 5.1, Fig. 3a–b; Table 10).
- Optimize only continuation tokens: this cleanly isolates the “recovery” policy from the injected prefix (Sec. 3.1).

## 4. Key Insights and Innovations
1. Training with counter-aligned prefills to teach recovery (fundamental)
   - Instead of trying to stop adversarial prefixes at runtime, RECAP systematically exposes the model to unsafe and over-conservative CoT during RL training and rewards it only when it overrides these trajectories (Sec. 3.1; Fig. 1). This reframes brittleness as a supervision signal.

2. A drop-in RLHF recipe with no extra cost (practical, widely adoptable)
   - RECAP keeps the standard DAPO objective and sampling budget; the only change is injecting prefixes on a subset of prompts and optimizing post-prefix tokens (Sec. 3.1). Fig. 2 shows similar inference costs (token budgets) to vanilla DAPO.

3. Theoretical guarantee on mixed evaluation distributions (conceptual advance)
   - Theorem 1 formalizes why exposure to prefilled starts yields higher expected reward when some evaluation interactions include prefixes (Sec. 3.2; App. C). The result isolates gains to improvements on prefilled states and bounds other differences as slacks.

4. Robustness that persists under adaptive attacks and induces self-reflection (behavioral)
   - RECAP increases the frequency of semantic self-reflection in CoT (e.g., revising unsafe reasoning mid-trajectory): 83.4% vs 59.7% on StrongREJECT with prefill; 74.2% vs 43.9% on WildJailbreak (Sec. 5.2).
   - It resists aggressive attacks including full CoT hijacking and iterative prefill resets (IPR) (Sec. 5.3; Tables 3–4).

## 5. Experimental Analysis
Evaluation setup (Sec. 4.1)
- Domains and metrics
  - Safety on direct harmful prompts: StrongREJECT (plus a prefilled variant injecting flawed CoT).
  - Jailbreaking robustness: WildJailbreak and Scale AI Fortress (500 expert adversarial prompts).
  - Overrefusal on benign prompts: XSTest and Fortress-OR (benign subset with rubric scoring).
  - Math: MATH500 and GSM8K (pass@1), AIME2024 (pass@16).
  - Most safety/helpfulness judgments use GPT‑4o; math uses verifiable reward checks.

- Models and baselines
  - Policies: `DSLlama-8B`, `DSQwen-14B` (DeepSeek-distilled LRMs with strong reasoning but limited safety).
  - RLHF: DAPO as the main optimizer; GRPO tested for robustness of conclusions (App. E; Table 8).
  - Baselines include STAR, SafeChain, standard SFT, and vanilla DAPO (Sec. 4.2).

Main results (Table 2; Sec. 4.2)
- Safety on direct harmful prompts
  - DSLlama-8B: RECAP 99.68 vs DAPO 96.81 on StrongREJECT; with unsafe CoT prefilling (harder), RECAP 98.70 vs DAPO 79.23.
  - DSQwen-14B: RECAP ties DAPO on StrongREJECT (99.04), but on the prefilled variant RECAP 98.08 vs DAPO 80.51.
- Jailbreak robustness
  - DSLlama-8B: WildJailbreak 88.75 vs 72.90; Fortress 86.84 vs 68.86 (RECAP vs DAPO).
  - DSQwen-14B: WildJailbreak 91.65 vs 77.60; Fortress 80.17 vs 67.85.
- Overrefusal (helpfulness on benign inputs)
  - DSLlama-8B: XSTest 91.87 vs 78.00; Fortress‑OR 91.80 vs 82.80.
  - DSQwen-14B: RECAP maintains high helpfulness (XSTest 96.80, equal to DAPO) and improves Fortress‑OR (97.60 vs 95.00).
- Math capability
  - DSLlama-8B: RECAP improves or matches DAPO (MATH500 83.60 vs 82.20; GSM8K 93.72 ≈ 93.71; AIME2024 70.00 vs 66.67).
  - DSQwen-14B: MATH500 90.00 vs 88.80; GSM8K 97.77 vs 97.19; AIME2024 unchanged at 86.67.
- Inference cost (Sec. 4.3; Fig. 2; Table 9)
  - RECAP’s total tokens per completion remain comparable to DAPO across domains; CoT is slightly longer for safety/overrefusal and shorter for math.
  - Qualitative traces are more structured and logically connected (Sec. 4.3, App. G).

Robustness checks and ablations
- Sensitivity to initial reasoning (Sec. 2.2; Table 1)
  - “Unsafe” prefixes from a weak model drastically reduce safety of stronger models; “safe” prefixes improve them. This effect also transfers across families (Qwen 3.0 vs DSQwen).
- Beyond safety: brittleness in math and helpfulness (App. B)
  > Table 5 shows MATH500 accuracy drops when prefilling with weak CoT (e.g., DSQwen‑14B: 86.40 → 82.60) and increases with strong CoT (→ 92.40).  
  > Table 6 shows helpfulness on XSTest for Qwen3‑4B jumps from 84.0 to 93.2 when prefilling with helpful CoT from DSQwen‑32B.
- Counter-aligned prefills matter (Sec. 5.1; Fig. 3c; Table 10)
  - Using unsafe/refusal prefills yields large safety gains; using already-safe prefills underperforms even vanilla DAPO on safety.
- Ratio and length trade-offs (Sec. 5.1; Fig. 3a–b; Table 10)
  - Best trade-offs around `α=0.5`, `ℓ_pre≈500`; too much or too long increases reliance on prefills and can reduce helpfulness or safety.
- Self-reflection frequency (Sec. 5.2)
  > On StrongREJECT with prefilling attacks: 83.4% (RECAP) vs 59.7% (DAPO) of CoT traces exhibit semantic self-reflection.  
  > On WildJailbreak: 74.2% vs 43.9%.
- Adaptive attacks (Sec. 5.3)
  - Full CoT hijacking (entire CoT replaced; model only produces the final response):
    > Table 3: DSLlama‑8B safety 98.08 (RECAP) vs 70.29 (DAPO); DSQwen‑14B 96.49 vs 73.48.
  - Iterative prefill reset (IPR; repeatedly re-inject flawed prefix after “ignore above and restart”):
    > Table 4 (DSLlama‑8B): k=1: 98.72 vs 79.23; k=2: 98.08 vs 70.29; k=3: 97.44 vs 69.65. The RECAP advantage widens as rounds increase.
- Generalization across reward designs and RL algorithms (App. E)
  > Table 7 (binary safety rewards): RECAP maintains gains, e.g., StrongREJ‑Prefill 96.49 vs 84.66; WildJailbreak 82.15 vs 72.85.  
  > Table 8 (GRPO optimizer): similar improvements, e.g., WildJailbreak 86.75 vs 71.30; Fortress 80.67 vs 64.33.

Overall assessment
- The experiments are broad (safety, jailbreak, overrefusal, math), include strong baselines, and use both clean and adversarial settings. The large margins under prefilling and adaptive attacks, together with ablations isolating key factors (α, ℓ_pre, prefill source), convincingly support the claim that counter-aligned prefill training improves robustness without sacrificing capability (Sec. 4–5, Figs. 1–3; Tables 1–4, 7–10).

## 6. Limitations and Trade-offs
- Reliance on external models for prefills
  - RECAP samples unsafe/refusal prefixes from other models (`π_harm`, `π_refuse`) (Sec. 3.1). Availability and quality of such sources matter; poor or distribution-mismatched prefills could reduce training efficacy.

- Hyperparameter sensitivity
  - The safety–helpfulness trade-off depends on the prefill ratio and length; extremes (`α→1`, `ℓ_pre` very long) can hurt performance (Sec. 5.1; Fig. 3a–b; Table 10).

- Reward-model and evaluator dependence
  - Safety rewards use Granite‑Guardian logits; overrefusal uses rubric judgments by Llama‑3.1; benchmark scoring often relies on GPT‑4o (Sec. 4.1). These choices may introduce biases or misclassifications; different evaluators might shift measured gains.

- Scope of attacks
  - RECAP is robust to full hijacking and iterative resets (Sec. 5.3) but adversaries might craft new strategies (e.g., multi-turn tool use, multimodal prefixes) not covered here.

- Computational cost of online RLHF
  - While RECAP adds no extra cost relative to DAPO, the base setup still requires substantial compute (e.g., 32–64 A100 GPUs; App. D). Smaller labs might find replication costly.

- Transfer to other modalities and languages
  - The paper’s results are text-only and primarily English; extension to multilingual/multimodal LRMs remains open (Sec. 7).

## 7. Implications and Future Directions
- Shift in alignment strategy
  - The work reframes safety alignment: instead of only teaching the “right answers,” teach models to recover from “wrong thinking.” This directly tackles the brittleness exposed by CoT prefilling and helps in settings where initial reasoning is noisy, adversarial, or inherited from other agents.

- Practical adoption
  - Because RECAP is a drop-in modification to standard RLHF pipelines (no new losses or sampling), it is practical for organizations already running online RL (Sec. 3.1, 4.3). Use cases include assistant models exposed to prefix control, agentic workflows where intermediate steps can be corrupted, and systems facing jailbreaking attempts.

- Research directions
  - Extend to multimodal and multilingual LRMs where prefills could be visual or code-like (Sec. 7).
  - Develop principled methods to generate informative flawed prefixes for other capabilities (e.g., mathematics), where multiple wrong trajectories exist and “useful” counter-aligned supervision is subtler (Sec. 7).
  - Explore curriculum and adaptive scheduling of `α` and `ℓ_pre` to optimize the safety–helpfulness trade-off.
  - Combine RECAP with verifiable rewards and programmatic constraints (e.g., tool-based safety checks) to further ground corrective reasoning.
  - Study interpretability links: RECAP increases self-reflection frequency (Sec. 5.2). Analyzing these trajectories could illuminate mechanisms of safety reasoning and inform new debugging tools.

In short, by deliberately training LRMs to override flawed CoT, RECAP offers a simple, theoretically grounded, and empirically strong route to robust safety alignment that does not sacrifice core reasoning capability.
