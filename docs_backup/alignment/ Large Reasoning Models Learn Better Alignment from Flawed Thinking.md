# Large Reasoning Models Learn Better Alignment from Flawed Thinking

**ArXiv:** [2510.00938](https://arxiv.org/abs/2510.00938)
**Authors:** ShengYun Peng, Eric Smith, Ivan Evtimov, Song Jiang, Pin‑Yu Chen, Hongyuan Zhan, Haozhu Wang, Duen Horng Chau, Mahesh Pasupuleti, Jianfeng Chi
**Institutions:** Meta Superintelligence Labs, IBM Research, Georgia Tech

## 1. Executive Summary (2–3 sentences)
- Reasoning: To summarize clearly, I distill the core method (RECAP), its mechanism (training with counter-aligned prefills), and the measurable impact (robust safety with preserved reasoning and efficiency), while anchoring in concrete evidence from figures and tables.

RECAP is a reinforcement learning recipe that injects deliberately flawed chain-of-thought (`CoT`) prefixes during training and teaches large reasoning models (`LRMs`) to override them, yielding robust safety alignment without extra training complexity beyond standard `RLHF`. Across diverse benchmarks, RECAP significantly boosts safety and jailbreak resilience, reduces overrefusal, and preserves math reasoning, while keeping inference token budgets comparable to baselines (Fig. 1, Fig. 2; Table 2).

## 2. Context and Motivation
- Reasoning: I lay out the problem (brittle safety reasoning), why it matters (prefill-supported interfaces and real risk), where prior approaches fall short (optimizing only final responses), and how this paper reframes the issue (train models to recover from flawed reasoning), using the paper’s empirical evidence to illustrate the gap.

- Problem addressed:
  - `LRMs` generate structured `CoT` before answers but remain highly sensitive to their initial reasoning direction; a flawed or adversarial prefill can steer them into unsafe outputs.
  - Evidence: When evaluated with different prefilled traces, safety scores swing widely. In Table 1, injecting unsafe `CoT` from `DSQwen-1.5B` drops average safety by 36.4%, while injecting safe `CoT` from `DSQwen-32B` increases safety by 91.7%. This holds across model families (Qwen3-4B shows −19.5% and −11.4% shifts).

- Why it matters:
  - `CoT prefilling`—injecting a partial reasoning prefix at inference—is widely supported (Appendix references to open-source and commercial APIs), making this brittleness a practical safety risk.
  - Jailbreaks often work by manipulating reasoning context. Robustness to adversarial initial `CoT` is crucial for dependable deployment (Sec. 2.2).

- Prior approaches and shortcomings:
  - Common `RLHF` variants (e.g., `GRPO`, `DAPO`) largely optimize the final response reward, not the trajectory itself (Sec. 2), assuming models will self-correct during reasoning.
  - Supervised finetuning on curated safety data (e.g., `STAR`, `SafeChain`) improves safety but can worsen overrefusal (Table 2), and does not directly teach recovery from harmful mid-trajectory reasoning.

- Positioning:
  - RECAP reframes alignment: expose the model during training to flawed `CoT` prefixes and reward it for rerouting to safe/helpful outcomes. This turns brittleness into a supervision signal for persistent robustness (Fig. 1; Sec. 3).

## 3. Technical Approach
- Reasoning: I break down RECAP step-by-step—what is prefilled training, how counter-aligned prefixes are constructed, how the optimization objective is modified, and why the theory predicts advantages. I define nonstandard terms and explain equations in plain language before notation.

- Key terms:
  - `LRM`: a language model that “reasons” by emitting intermediate `CoT` before the final answer.
  - `CoT prefilling`: injecting a partial, pre-generated reasoning prefix (`y_pre_cot`) into the model’s input, so the model continues from that reasoning.
  - `Counter-aligned`: a prefix that is syntactically fluent but semantically misaligned with safety or helpfulness (e.g., unsafe reasoning on harmful prompts; refusal-oriented reasoning on benign prompts).
  - `Overrefusal`: unwarranted refusal of benign requests due to overcautious alignment.
  - `DAPO`: dynamic sampling policy optimization—a practical online RLHF variant with clipped ratios and per-group advantage normalization (Sec. 3.1).

- Setup and notation (Sec. 2.1):
  - An input prompt `x` produces output `y = (ycot, yresp)`.
  - With prefilling, we feed `(x, y_pre_cot)` and the model generates `y_gen_cot` (continuation) and `yresp`. Full output: `(y_pre_cot ∥ y_gen_cot, yresp)`.

- Constructing counter-aligned prefills (Sec. 3.1):
  - For each training example `x`, create a prefix `y_pre_cot` of length `ℓpre`.
  - Source of prefixes:
    - Harmful prompts: sample unsafe reasoning from a weakly aligned or “helpful-only” model, denoted `πharm`.
    - Benign prompts: sample refusal-oriented reasoning from an overly conservative model `πrefuse`.
  - This creates intentional traps: naively continuing `y_pre_cot` would produce unsafe answers (for harmful prompts) or unnecessary refusals (for benign prompts). To earn high reward, the model must override the flawed trajectory with a corrective `y_gen_cot` and produce an aligned `yresp`.

- Training data mixture:
  - Apply prefilling to a fraction `α` of training prompts, forming `Dprefill` (Sec. 3.1).
  - The rest of the prompts are standard (no prefill), ensuring models learn to initiate safe reasoning even without a prefix.

- Optimization objective `JRECAP` (Sec. 3.1):
  - RECAP extends `DAPO` to handle prefills by optimizing only tokens after the injected prefix. In plain language:
    - Sample groups of rollouts per prompt from the old policy.
    - Compute the importance sampling ratio `ri,t(θ)` and normalized advantage `Â_i,t` only for time steps after `t0(x)`—which is 1 for clean prompts, and `ℓpre+1` for prefilled prompts.
    - Apply clipped policy updates per token (min of unclipped and clipped ratio times advantage), aggregated over optimized tokens across rollouts.
    - Dynamic sampling discards prompts if all rollouts are uniformly bad (`Ri = 0`) or uniformly good (`Ri > τ`), stabilizing training.
  - The formal objective (Sec. 3.1) is:
    > JRECAP(θ) = Ex∼Dprefill ,{oi}∼πθold [ sum over i, t≥t0(x) of min(ri,t(θ)Â_i,t, clip(ri,t(θ), 1−εlow, 1+εhigh)Â_i,t ) normalized over optimized tokens ]
    - Where `Ri` is scalar reward for the final response, `Â_i,t` is per-token advantage normalized across group rollouts, and only post-prefill tokens are optimized.

- Rewarding the outcomes (Sec. 4.1):
  - Safety rewards: continuous logits from `IBM Granite-Guardian-3.1-8B`, providing dense signals.
  - Overrefusal rewards: rubric-based helpfulness scored by `Llama-3.1-8B-Instruct`.
  - Math rewards: `RLVR` (verifiable rewards for correct answers).

- Why it works (Sec. 3.2; Appendix C):
  - Define combined evaluation objective `J(π) = (1−β)Jclean(π) + βJpre(π)`, mixing clean and prefilled evaluation distributions with weight `β`.
  - Theorem 1 (Sec. 3.2) informally shows RECAP’s updates produce strictly higher expected reward than vanilla `DAPO` under both clean and prefilled conditions, assuming conservative policy updates and bounded baseline improvements on prefilled samples:
    > ΔT := J(π_RT) − J(π_DT) ≥ β ∑_{t=0}^{T−1} [ γ_R_pre(t) − ε_R(t) − ζ(t) ] − O(ϵ)
    - `γ_R_pre(t)`: per-step improvement from training on prefilled samples.
    - `ε_R(t)`: approximation/estimation error in RECAP updates.
    - `ζ(t)`: incidental progress of `DAPO` on prefilled samples (typically small since `DAPO` never sees these states).
    - The advantage grows with the weight `β` (more adversarial settings) and cumulative learning to recover from unsafe reasoning (`γ_R_pre(t)`).

## 4. Key Insights and Innovations
- Reasoning: I extract the most novel aspects, differentiating substantive new ideas from optimizations, and explain why each matters with pointers to the paper’s evidence.

- Novel contributions:
  - Training on counter-aligned reasoning prefixes (fundamental innovation):
    - Instead of feeding safe `CoT`, RECAP intentionally injects flawed reasoning during training and rewards recovery (Sec. 3.1; Fig. 1).
    - Significance: Converts a failure mode (following bad reasoning) into a training signal for robustness, improving safety under both normal and adversarial contexts (Table 2; Tables 3–4).
  
  - Minimal change to standard `RLHF` workflows (incremental but impactful):
    - No new loss terms beyond `DAPO`/`GRPO`; no extra inference tokens required (Sec. 4.3; Fig. 2; Table 9).
    - Significance: Practical adoption—maintains throughput and latency while increasing structured reflection in `CoT`.

  - Theoretical robustness guarantee under mixed evaluation distributions (fundamental innovation):
    - Formal argument that RECAP’s expected reward advantage grows with exposure to prefilled samples (Sec. 3.2; Appendix C).
    - Significance: Explains empirical gains in jailbreak and prefilling attack settings where models start from adversarial states (higher `β` regimes).

  - Emergent increase in self-reflection frequency (novel capability change):
    - RECAP-trained models revise unsafe or mistaken reasoning mid-trajectory far more often (83.4% vs 59.7% on StrongREJECT; 74.2% vs 43.9% on WildJailbreak; Sec. 5.2).
    - Significance: Persistent robustness under adaptive attacks, tied to improved reasoning dynamics rather than superficial refusals.

  - Safety and helpfulness gains without capability tax (incremental but notable):
    - Improves direct harmfulness, jailbreak robustness, and reduces overrefusal simultaneously, while preserving or slightly enhancing math reasoning (Table 2).
    - Significance: Counters the typical trade-off (safety tax) noted in recent work by co-training alignment and reasoning capability (Sec. 4.2).

## 5. Experimental Analysis
- Reasoning: I detail the evaluation setup, metrics, baselines, and results with numerical specifics, then assess how convincing the evidence is, including ablations and robustness checks, and note conditions under which claims hold.

- Evaluation methodology (Sec. 4.1):
  - Domains:
    - Safety: direct harmful prompts (`StrongREJECT`) and jailbreak prompts (`WildJailbreak`, `Fortress`).
    - Overrefusal: `XSTest`, benign subset of `Fortress` (`FortressOR`).
    - Math: `MATH500`, `GSM8K`, `AIME2024`.
  - Metrics:
    - Safety/helpfulness: % of completions judged safe/helpful by `GPT-4o` (Fortress uses instance-specific rubrics).
    - Math: `pass@K` (K=1 for MATH500/GSM8K; K=16 for AIME2024).
  - Models and rewards:
    - Policies: `DSLlama-8B`, `DSQwen-14B` (DeepSeek-distilled reasoning LRMs).
    - Safety rewards: continuous logits from `Granite-Guardian-3.1-8B`.
    - Overrefusal scores: rubric judged by `Llama-3.1-8B-Instruct`.
    - Math rewards: `RLVR`.
  - Training setup:
    - 5k training prompts: 1k harmful (`BeaverTails`), 1k overrefusal (`STAR-1`), 3k math (`GSM8K`, `MATH`).
    - 16 rollouts per prompt; `DAPO` baseline; prefilling applied to fraction `α` and length `ℓpre` (defaults α=0.5, ℓpre=500; Sec. 4.2; Appendix D).

- Main quantitative results (Table 2):
  - Safety (DSLlama-8B):
    - `StrongREJECT`: `RECAP` 99.68 vs `DAPO` 96.81; vs `SFT` 73.48; `STAR` 77.00; `SafeChain` 68.05.
    - `StrongREJ-Prefill`: 98.70 vs 79.23 (prefill uses flawed `CoT`); large robustness gain.
    - `WildJailbreak`: 88.75 vs 72.90.
    - `Fortress` (500 adversarial prompts): 86.84 vs 68.86.
  - Safety (DSQwen-14B):
    - `StrongREJECT`: 99.04 equals `DAPO` 99.04 (near ceiling).
    - `StrongREJ-Prefill`: 98.08 vs 80.51.
    - `WildJailbreak`: 91.65 vs 77.60.
    - `Fortress`: 80.17 vs 67.85.
  - Overrefusal (helpfulness):
    - DSLlama-8B: `XSTest` 91.87 vs 78.00; `FortressOR` 91.80 vs 82.80.
    - DSQwen-14B: `XSTest` 96.80 vs 96.80 (parity); `FortressOR` 97.60 vs 95.00 (gain).
  - Math:
    - DSLlama-8B: `MATH500` 83.60 vs 82.20; `GSM8K` 93.72 vs 93.71; `AIME2024` 70.00 vs 66.67.
    - DSQwen-14B: `MATH500` 90.00 vs 88.80; `GSM8K` 97.77 vs 97.19; `AIME2024` 86.67 equal.

- Do experiments support claims?
  - Strong evidence for safety robustness under both clean and adversarial initial reasoning:
    - Large margins on prefilled harmful evaluation (`StrongREJ-Prefill`) and jailbreak (`WildJailbreak`, `Fortress`), consistent with the theoretical analysis that gains scale with exposure to adversarial starts (Sec. 3.2; Table 2).
  - Helpfulness improvement alongside safety:
    - Unlike `STAR` or `SafeChain`, RECAP boosts helpfulness on benign prompts (Table 2), suggesting it combats overrefusal by teaching recovery from refusal-oriented `CoT` prefills.
  - Capability preservation:
    - Math remains stable or slightly improved, despite prefilling used only for safety/overrefusal tasks in training (Sec. 4.2; Appendix F).

- Efficiency and `CoT` structure:
  - Token budget:
    - Fig. 2 shows comparable total tokens to `DAPO` (e.g., DSQwen-14B: `RECAP Total` 707 vs `DAPO Total` 625 averaged across domains), with slightly longer `CoT` in safety/overrefusal and shorter in math (Table 9 provides breakdown by benchmark).
  - Qualitative `CoT` improvements:
    - RECAP produces more structured, logically connected reasoning (Appendix G), aligning with higher self-reflection rates (Sec. 5.2).

- Ablations and robustness checks:
  - Prefilling ratio `α` and length `ℓpre` trade-offs (Fig. 3; Table 10):
    - Moderate `α` (0.5) and `ℓpre` (100–500) achieve the best safety-helpfulness balance; too high `α` or too long `ℓpre` can reduce helpfulness or induce overreliance.
  - Prefill source matters:
    - Counter-aligned (unsafe) prefixes outperform aligned (safe) prefixes, which underperform even vanilla `DAPO`—indicating recovery training, not safe exploitation, drives robustness (Fig. 3c).
  - Self-reflection:
    - On StrongREJECT with prefilling attacks: 83.4% of `CoT` include self-reflection under RECAP vs 59.7% under `DAPO`; on WildJailbreak: 74.2% vs 43.9% (Sec. 5.2).
  - Adaptive attacks:
    - Full `CoT` hijacking (replace entire reasoning with malicious prefix): RECAP maintains high safety (DSLlama-8B 98.08 vs 70.29; DSQwen-14B 96.49 vs 73.48; Table 3).
    - IPR (iterative prefill reset) attack: safety drops with more rounds but plateaus, and RECAP stays substantially higher (k=1: 98.72 vs 79.23; k=3: 97.44 vs 69.65; Table 4).
  - Generalization across reward/optimizer:
    - Binary safety reward still yields RECAP > `DAPO` (Table 7).
    - `GRPO` optimizer: RECAP maintains improvements vs vanilla `GRPO` (Table 8).

- Mixed or conditional results:
  - Ceiling effects on some safety metrics (e.g., DSQwen-14B `StrongREJECT` ~99) limit observable gains.
  - Aggressive prefilling (α=1, ℓpre=700) begins to degrade helpfulness and safety (Table 10), underscoring the need for balanced prefilling schedules.

## 6. Limitations and Trade-offs
- Reasoning: I identify assumptions, scope boundaries, and practical trade-offs from the paper’s own analyses and setups, linking them to the empirical and theoretical evidence provided.

- Assumptions and evaluation dependencies:
  - Automated judges:
    - Safety/helpfulness rely on `GPT-4o` and `Granite-Guardian-3.1-8B` rewards; bias or drift in these evaluators could affect training signals and reported metrics (Sec. 4.1; Table 7).
  - Theoretical assumptions:
    - Theorem 1 uses conservative update bounds and bounded incidental improvements of `DAPO` on prefilled starts (Sec. 3.2; Appendix C). These are reasonable for clipped policy updates but abstract away details like reward noise and distribution shift.

- Scope and edge cases:
  - Prefill-sensitive domains:
    - RECAP focuses on safety and overrefusal; math gains are modest and incidental. Extending prefilling to math is not explored, and incorrect mathematical `CoT` may have different correction dynamics (Sec. 7).
  - Data and model diversity:
    - Training corpus is relatively small (5k prompts) and English-only; multilingual and multimodal robustness remains untested (Sec. 4.2).

- Trade-offs in design:
  - Prefilling schedule:
    - Larger `α` and longer `ℓpre` can boost safety but risk overrefusal or reliance on prefixes (Fig. 3; Table 10). The chosen defaults (α=0.5, ℓpre=500) reflect a practical middle ground.
  - Efficiency vs reasoning richness:
    - RECAP slightly increases `CoT` length in safety/overrefusal (Table 9), though total tokens remain comparable to `DAPO` (Fig. 2). Some deployments may still prefer strictly shorter completions.

- Practical constraints:
  - Compute:
    - While RECAP introduces “no additional training cost beyond RLHF” in algorithmic complexity, executing RLHF at scale is itself compute-intensive (Appendix D: multi-node A100 clusters). The method is practical for labs already running `RLHF`, but not “lightweight” absent such infrastructure.
  - Availability:
    - Code is to be released “shortly”; reproducibility depends on access to reward models and prefilling sources (frontier and internal models are referenced in Sec. 3.1, Sec. 4.1).

## 7. Implications and Future Directions
- Reasoning: I extrapolate how this approach shifts alignment practice, suggest concrete follow-ups grounded in observed behaviors (self-reflection, robustness to prefilling), and outline practical applications.

- Field impact:
  - Shifts alignment from end-only optimization to trajectory-aware training:
    - By rewarding recovery from flawed reasoning, RECAP directly addresses the brittle behavior revealed by `CoT` prefilling (Sec. 2.2; Table 1). This encourages models to critique and revise their own `CoT`, a prerequisite for robust “reasoning safety.”
  - Offers a practical recipe:
    - Integration into `RLHF` stacks (`DAPO`, `GRPO`) without changing the objective (Sec. 3.1; Tables 7–8) makes adoption feasible for organizations already running online RL post-training.

- Follow-up research:
  - Multilingual and multimodal extensions:
    - Prefill brittleness may be more pronounced in non-English or vision–language contexts; evaluate and adapt RECAP across modalities and languages (Sec. 7).
  - Trajectory-level rewards:
    - Augment scalar final-response rewards with intermediate signals that mark self-correction or the abandonment of unsafe steps (connecting to `RLVR`-style verifiers for safety).
  - Adaptive prefilling curricula:
    - Dynamically tune `α`, `ℓpre`, and the diversity of prefill sources (`πharm`, `πrefuse`) to optimize robustness without overrefusal (guided by Fig. 3; Table 10).
  - Safety-capability cross-talk:
    - Investigate why safety prefilling improved math slightly (Table 2, Appendix F). Hypothesis: exposure to diverse flawed trajectories improves exploration and symmetry-aware reasoning.

- Practical applications:
  - API-level defense against prefill hijacks:
    - RECAP’s robustness to full `CoT` hijacking and resets (Tables 3–4) suggests value in hardening production endpoints where users can prefill responses.
  - Guardrail model training:
    - Train guardrails with counter-aligned prefills to better detect and neutralize unsafe reasoning mid-trajectory, complementing systems like `Llama Guard` (Related Work).
  - Safety-critical deployments:
    - Use RECAP for systems exposed to jailbreaks or roleplay prompts (e.g., public chatbots, coding assistants), where injected premises often try to bypass safety. The Fortress results (Table 2) show benefits in high-stakes adversarial settings.

- Representative supporting excerpts:
  > “Prefilling with y_pre_cot from DSQwen-1.5B reduces average safety scores by 36.4%, whereas prefilling with y_pre_cot from DSQwen-32B increases them by 91.7%.” (Table 1)

  > “RECAP (Ours) 99.68 [StrongREJECT], 98.70 [StrongREJ-Prefill], 88.75 [WildJailbreak], 86.84 [Fortress], 91.87 [XSTest], 91.80 [FortressOR]… [DSLlama-8B]” (Table 2)

  > “On StrongREJECT with prefilling attacks, 83.4% of CoT traces from DSQwen-14B trained with RECAP exhibit self-reflection, compared to 59.7% under DAPO.” (Sec. 5.2)

  > “Compared to DAPO, RECAP remains substantially more robust [full CoT hijacking], achieving on average +35.4% higher safety.” (Table 3)

  > “Increasing the number of IPR attack rounds k reduces safety scores, but the decline plateaus… RECAP is substantially more robust than vanilla DAPO, with the gap widening for larger k.” (Table 4)

  > “RECAP maintains a comparable total token budget to vanilla DAPO.” (Fig. 2; Table 9)


