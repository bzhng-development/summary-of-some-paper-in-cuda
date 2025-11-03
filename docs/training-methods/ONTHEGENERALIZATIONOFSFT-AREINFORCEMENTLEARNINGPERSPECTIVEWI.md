# ON THE GENERALIZATION OF SFT: A REINFORCEMENT LEARNING PERSPECTIVE WITH REWARD RECTIFICATION

**ArXiv:** [2508.05629](https://arxiv.org/abs/2508.05629)

## 🎯 Pitch

This paper delivers a breakthrough by rigorously showing that standard supervised fine-tuning (SFT) for large language models is equivalent to a reinforcement learning policy gradient update with a problematic, sparse implicit reward—explaining SFT’s poor generalization relative to RL. The authors introduce Dynamic Fine-Tuning (DFT), a remarkably simple, one-line change that rescales the SFT loss by each token’s probability, stabilizing training and dramatically boosting generalization on hard benchmarks (math, code, multimodal tasks) and even outperforming state-of-the-art RL methods in offline settings. This work not only deepens our theoretical understanding of LLM training but also provides a practical, efficient alternative to RL-based alignment—with major implications for improving LLM robustness and usefulness wherever standard SFT is used.

---

## 1. Executive Summary (2–3 sentences)
This paper reinterprets supervised fine-tuning (SFT) for large language models as a special case of reinforcement learning (RL), revealing that the usual SFT gradient implicitly corresponds to a sparse, ill‑posed reward that is inversely proportional to the model’s confidence (Equations 5–6). Building on this, it proposes Dynamic Fine-Tuning (DFT), a one-line modification that rescales each token’s loss by its own (stop‑gradient) probability (Equations 7–9), yielding more stable updates and markedly better generalization across math reasoning, code generation, and multimodal tasks (Tables 1–4), and even surpassing strong offline and online RL baselines in an offline RL setup (Table 2).

## 2. Context and Motivation
- Problem addressed:
  - SFT (training on “expert” demonstrations) is simple and effective for imitating desirable behavior, but it often generalizes poorly compared to RL on challenging reasoning tasks (Introduction; Related Work).
  - Many practical pipelines still rely on SFT because RL requires dense reward signals, heavy compute, and delicate hyperparameter tuning, which are not always feasible (Introduction).
  - The open question: Can SFT itself be fundamentally improved—especially in settings with only positive demonstrations and no reward model or preference data?

- Why this matters:
  - Practical: SFT remains the de facto post-training step for many LLMs because it is simple and data-efficient; improving its generalization without RL’s overhead would benefit a large portion of the ecosystem.
  - Theoretical: Clarifying the mathematical relationship between SFT and RL could expose failure modes of SFT and suggest principled fixes.

- Prior approaches and their shortcomings:
  - RL-based alignment methods (e.g., PPO/GRPO) and preference-based methods (e.g., DPO, RAFT/RFT) improve generalization but need explicit rewards, preference pairs, or online sampling (Related Work; Sections 4.2–4.4).
  - Theoretical connections between SFT and RL exist but typically rely on heuristic reweighting or do not show a precise gradient-level equivalence (Related Work).
  - Consequently, when only positive demonstrations are available, SFT is still used despite known generalization issues.

- Positioning:
  - This work offers (1) a precise derivation that equates the standard SFT gradient to a policy gradient with a sparse, inversely weighted reward (Equations 5–6), and (2) a minimal fix—DFT—that removes the harmful inverse-probability factor by multiplying the loss by the model’s own (stop‑grad) probability (Equations 7–9).
  - It deliberately avoids external rewards, reference models, or online sampling, targeting the native SFT setting.

## 3. Technical Approach
Step-by-step explanation, from baseline SFT to DFT.

1) Preliminaries: standard objectives
- SFT objective (sequence level): maximize the log-likelihood of the expert response `y*` given input `x`
  - Loss and gradient (Equations 1–2):
    - `LSFT(θ) = E_{(x,y*)∼D}[ - log πθ(y* | x) ]`
    - `∇θ LSFT(θ) = E_{(x,y*)∼D}[ - ∇θ log πθ(y* | x) ]`

- RL objective: maximize expected reward over outputs `y ∼ πθ(·|x)`
  - Objective and policy gradient (Equations 3–4):
    - `J(θ) = E_{x∼Dx, y∼πθ(·|x)}[ r(x, y) ]`
    - `∇θ J(θ) = E_{x∼Dx, y∼πθ(·|x)}[ ∇θ log πθ(y | x) · r(x, y) ]`

2) Rewriting SFT as a policy gradient with importance sampling
- Key derivation (Equation 5): rewrite the SFT gradient as an expectation under the current model policy by inserting an importance weight that compares the expert’s Dirac distribution with the model distribution:
  - `E_{(x,y*)∼D}[ -∇ log π(y*|x) ] = E_{x, y∼π(·|x)}[ 1[y=y*] / π(y|x) · ( -∇ log π(y|x) ) ]`
- Define weight and reward (before rectification) (Equation 6):
  - Importance weight `w(y|x) = 1 / πθ(y|x)`
  - Reward `r(x,y) = 1[y = y*]` (indicator: 1 only if the model output exactly matches the reference)
  - Then: `∇ LSFT = - E_{x,y∼π}[ w(y|x) ∇ log π(y|x) r(x,y) ]`
- Intuition:
  - Two issues emerge:
    - Sparsity: the reward `r(x,y)` is nonzero only at exact matches—extremely rare for long outputs.
    - Inverse-probability amplification: when the model assigns low probability to an expert token, the factor `1/π` becomes large, producing unstable, oversized gradients focused on rare tokens.

3) The proposed fix: Dynamic Fine-Tuning (DFT)
- Idea: cancel the harmful `1/π` factor by multiplying the SFT gradient by the model probability at the expert token, but using a stop‑gradient so gradients do not flow through the multiplier (Equation 7).
  - Sequence-level form (Equation 8):
    - `LDFT(θ) = E_{(x,y*)∼D}[ - sg(πθ(y* | x)) · log πθ(y* | x) ]`
    - `sg(·)` is the stop-gradient operator: it treats the value as a constant during backpropagation.
- Token-level version (practical and numerically stable) (Equation 9):
  - Sum over tokens of the expert sequence `y*_t`:
    - `LDFT = E[ - Σ_t sg(πθ(y*_t | y*_{<t}, x)) · log πθ(y*_t | y*_{<t}, x) ]`
  - Why per-token: full-sequence probabilities are products of many small numbers and can be numerically unstable; token-level reweighting mirrors common practice in PPO-style importance sampling.
- What does this change do?
  - From the RL view: after multiplying by `π`, the implicit reward becomes uniform (1) for expert tokens—removing the bias that over-emphasized low-probability tokens.
  - From the optimization view (Appendix A.4): the DFT gradient equals `-∇θ π(y*|x)`, i.e., it directly increases the model’s probability, whereas cross-entropy scales the gradient by `1/π` and thus disproportionately upweights unlikely tokens.
- Implementation detail:
  - This is a literal one-line change in most training loops: multiply each token’s negative log-likelihood by the stop‑gradient of its predicted probability.
  - No extra models, rewards, or sampling are needed.

4) A simple analogy
- Cross-entropy (SFT) behaves like a coach who yells the loudest when the student is least confident about a token, sometimes pushing too hard on rare, idiosyncratic tokens.
- DFT makes the coach’s voice proportional to the current confidence (but not trainable through that voice), so the student improves uniformly without letting rare tokens dominate the learning signal.

## 4. Key Insights and Innovations
- Precise gradient-level equivalence between SFT and RL with an implicit reward (Equations 5–6)
  - Novelty: It shows standard SFT is equivalent to a policy gradient with an indicator reward multiplied by `1/πθ(y|x)`. This clarifies why SFT overfits and destabilizes when probabilities are low: the update magnitude explodes.
  - Significance: This is a clean, actionable theoretical diagnosis that directly motivates a fix.

- Reward rectification through dynamic reweighting (Equations 7–9)
  - Novelty: Multiply the SFT loss by the stop‑gradient token probability to cancel the inverse-probability factor, turning the implicit reward into a uniform “1 for expert tokens.”
  - Significance: A one-line, reference-free change that improves stability and generalization without RL infrastructure.

- Token-level importance treatment for stability (Equation 9)
  - Novelty: Apply the reweighting per token (as in PPO’s practice) to avoid numerical issues with sequence-level probabilities.
  - Significance: Makes the method robust and easy to drop into existing SFT pipelines.

- Behavioral analysis of what changes in the model (Figure 2; Appendix A.4)
  - Insight: Unlike SFT, which pushes most token probabilities upward, DFT polarizes the token distribution—boosting a subset and suppressing others, especially non-semantic connector tokens like “the,” “let,” or punctuation.
  - Significance: Suggests DFT focuses capacity on semantically important tokens, which aligns with improved reasoning generalization.

## 5. Experimental Analysis
Evaluation setup, results, and whether they support the claims.

- Evaluation methodology
  - Math reasoning (Section 4.1):
    - Training data: 100k sampled problems from NuminaMath-CoT (Section 4.1.1).
    - Models: `Qwen2.5-Math-1.5B`, `Qwen2.5-Math-7B`, `LLaMA‑3.2‑3B`, `LLaMA‑3.1‑8B`, and `DeepSeekMath‑7B`.
    - Benchmarks: Math500, Minerva Math, OlympiadBench, AIME 2024, AMC 2023. Metric: Average@16 accuracy (16 decoding runs; temperature=1.0; max length=4096).
    - Optimizer and schedule: AdamW; LR mostly `5e-5` (lower for `LLaMA‑3.1‑8B`); batch size 256; cosine decay with 0.1 warmup.
  - Offline RL setting (Section 4.2):
    - Data generation: For 100k prompts, sample 4 responses each from the base model at temperature=1.0; filter correct ones via math verification to obtain ~140k positive examples; build 100k preference pairs for DPO.
    - Baselines: DPO (offline), RFT/RAFT (offline), PPO and GRPO (online RL).
  - Code generation (Section 4.3):
    - Training: 10k prompts from UltraFeedback; one epoch; LR `5e-5`; warmup 0.05; batch size 16.
    - Benchmarks: HumanEval, HumanEval+, MultiPL‑E.
  - Multimodal reasoning (Section 4.4):
    - Training with WeThink; one epoch; LR `5e-5`; evaluated with VLMEvalKit on MathVerse and MathVision; plus WeMath.

- Main quantitative results
  - Math reasoning (Table 1):
    - `Qwen2.5‑Math‑1.5B`: average accuracy improves from `15.92` (base) → `18.01` (SFT) → `31.58` (DFT). The DFT gain over base is `+15.66`, about 5.9× larger than SFT’s `+2.09`.
    - `Qwen2.5‑Math‑7B`: `21.25` → `23.62` (SFT) → `37.15` (DFT). DFT’s `+15.90` gain dwarfs SFT’s `+2.37`.
    - Strong gains on hard sets where SFT regresses:
      - OlympiadBench for `Qwen2.5‑Math‑1.5B`: `15.88` (base) → `12.63` (SFT) → `27.08` (DFT).
      - AIME24 for `Qwen2.5‑Math‑7B`: `6.68` → `2.48` (SFT) → `8.56` (DFT).
  - Learning dynamics (Figure 1):
    - Faster convergence: “peak performance within the first ~120 steps,” and “DFT already beats SFT’s best final accuracy within 10–20 steps.”
  - Offline RL comparison (Table 2):
    - Average across math benchmarks with `Qwen2.5‑Math‑1.5B`:
      - DFT (offline): `35.43`
      - GRPO (online RL): `32.00`
      - PPO (online RL): `28.66`
      - RFT (offline): `23.97`; DPO (offline): `23.20`
    - Highlights:
      - Math500: DFT `64.71` vs GRPO `62.86` vs PPO `56.10` vs RFT `48.23`.
      - AMC23: DFT `48.44` vs GRPO `41.25`.
      - AIME24 is one exception where GRPO `8.34` slightly exceeds DFT `7.93`.
  - Code generation (Table 3):
    - `Qwen2.5‑Coder‑7B`:
      - HumanEval: `62.2` (base) → `54.9` (SFT) → `67.7` (DFT).
      - HumanEval+: `53.0` → `48.8` → `59.8`.
      - MultiPL‑E avg: `57.76` → `57.62` → `62.30`.
    - `Qwen2.5‑Coder‑3B`: consistent improvements with DFT on HumanEval/HumanEval+ and MultiPL‑E.
  - Multimodal reasoning (Table 4):
    - `Qwen2.5‑VL‑3B` (MathVerse overall): `33.83` (base) → `35.66` (SFT) → `37.54` (DFT).
    - MathVision: `21.25` → `21.02` (SFT degradation) → `22.30` (DFT).
    - WeMath: `21.25` → `21.02` (SFT) → `22.30` (DFT).
  - Additional analyses and robustness:
    - Token probability distribution shifts (Figure 2): DFT increases the highest-probability mass while also increasing the lowest-probability mass (a more bimodal shape), unlike SFT which pushes most tokens upward uniformly.
    - Hyperparameter ablations (Appendix A.8, Figure 3):
      - DFT consistently outperforms SFT across learning rates (`2e-4` to `1e-5`) and batch sizes (32–256).
    - LoRA/PEFT setting (Appendix A.7, Table 8):
      - `Qwen2.5‑Math‑1.5B`: average `15.92` (base) → `16.87` (SFT) → `32.90` (DFT).
    - Higher-quality dataset (OpenR1‑Math‑220k; Appendix A.6, Table 7):
      - `Qwen2.5‑Math‑1.5B` average: `15.92` → `29.16` (SFT) → `38.19` (DFT).
    - Comparison with concurrent iw‑SFT (Appendix A.5, Table 5 and Table 6):
      - DFT achieves higher average accuracy than iw‑SFT in most model families on standard SFT and offline settings.

- Do the experiments support the claims?
  - Yes, across multiple models, datasets, and training regimes, DFT either substantially outperforms SFT or turns SFT regressions into improvements (Table 1; Table 4; Figure 1).
  - The offline RL results (Table 2) are particularly striking: a simple, reward-free reweighting matches or surpasses strong online RL baselines on most math benchmarks, though not all (AIME24 is a close second).
  - Analyses (Figure 2; Appendix A.4) connect the method to observed behavior changes (token distribution), consistent with the theoretical motivation.

- Notable caveats from results
  - Some per-benchmark exceptions exist (e.g., AIME24 in Table 2 where GRPO slightly leads), indicating DFT is not universally dominant but is consistently strong.
  - Gains are largest on challenging reasoning tasks; milder but still positive on many code/multimodal settings (Tables 3–4).

## 6. Limitations and Trade-offs
- Assumptions and scope
  - The theory assumes the SFT gradient can be reframed as an on-policy expectation with an indicator reward and importance weights (Equations 5–6); the fix targets this specific pathology.
  - DFT relies on the model’s current probabilities to weight training signals. Early miscalibration could, in principle, downweight rare but critical tokens (confirmation bias risk), though empirical results suggest this is not a practical issue here (Tables 1–4, A.6–A.8).

- Data and task coverage
  - Evaluations heavily emphasize math reasoning; while code and multimodal results are positive, the breadth of domains is still limited (Limitations, Appendix A.2).
  - Large-scale frontier models are not evaluated; generalization to much larger models is an open question (Appendix A.2).

- Objective trade-offs
  - DFT optimizes the probability directly (Appendix A.4), which removes the `1/π` amplification that makes CE aggressive on hard tokens. This stabilizes training but may reduce pressure on truly rare, correct tokens compared to CE.
  - Uniform reward across expert tokens means DFT treats all “correct” tokens equally; it does not explicitly prioritize semantically crucial tokens beyond what emerges from the model’s own probabilities (though Figure 2 suggests it implicitly focuses on semantically meaningful tokens).

- Practical constraints
  - While DFT is compute-light, results depend on thoughtful decoding and evaluation (e.g., 16-sample averaging for accuracy), which can be resource-intensive during evaluation (Section 4.1.1).
  - The offline RL comparison builds its dataset from the model’s own samples plus verification (Section 4.2.1). Performance might depend on verification quality and sample diversity.

## 7. Implications and Future Directions
- How this changes the landscape
  - Conceptual shift: SFT is RL with a bad implicit reward; fix the reward by canceling inverse-probability weighting, and you get stronger generalization with a trivial code change (Equations 5–9).
  - Practical impact: Many teams can instantly improve SFT pipelines by changing the token loss to `- sg(p) * log p`, without reward models, reference policies, or large batch on-policy rollouts.

- Follow-up research enabled/suggested
  - Alternative reweighting schemes: Explore other functions of `p` (e.g., temperature-scaled `p`, clipped weights, per-token importance estimates) to balance stability and rare-token learning.
  - Curriculum or schedule: Start with DFT and gradually anneal toward CE, or mix CE and DFT based on confidence or token type.
  - Token-type awareness: The analysis (Figure 2; Appendix A.4) shows function words are downweighted. Explicitly modeling token roles (semantic vs connective) could further improve reasoning.
  - Hybrid pipelines: Combine DFT with lightweight verification (as in the offline RL setup) for compounding gains without full online RL.
  - Theory: Provide generalization bounds for DFT vs CE under sequence modeling; analyze convergence under sparse/long-horizon supervision.

- Applications and use cases
  - Reasoning-heavy domains: math problem solving, code synthesis, scientific QA, where overfitting to demonstrations harms transfer.
  - Low-resource or compute-constrained settings: LoRA/PEFT scenarios (Appendix A.7) benefit substantially from DFT without added infrastructure.
  - Offline alignment with weak rewards: DFT can exploit verified positives and outperform preference- or reward-based methods at comparable scale (Table 2).

> Bottom line: By reframing SFT as a flawed policy gradient (Equations 5–6) and fixing it with a minimal, principled rescaling (Equations 7–9), DFT delivers large, robust gains across challenging benchmarks (Tables 1–4), often rivaling or beating far heavier RL pipelines (Table 2), and offers a drop-in improvement for standard fine-tuning workflows.
