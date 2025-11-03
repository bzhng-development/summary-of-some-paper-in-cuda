# Secrets of RLHF in Large Language Models Part I: PPO

**ArXiv:** [2307.04964](https://arxiv.org/abs/2307.04964)
**Authors:** Rui Zheng, Shihan Dou, Songyang Gao, Wei Shen, Binghai Wang, Yan Liu, Senjie Jin, Qin Liu, Limao Xiong, Lu Chen, Zhiheng Xi, Yuhao Zhou, Nuo Xu, Wenbin Lai, Minghao Zhu, Rongxiang Weng, Wensen Cheng, Cheng Chang, Zhangyue Yin, Yuan Hua, Haoran Huang, Tianxiang Sun, Hang Yan, Tao Gui, Qi Zhang, Xipeng Qiu, Xuanjing Huang
**Institutions:** Fudan University (and other affiliations as appropriate)

## 🎯 Pitch

"PPO-max revolutionizes reinforcement learning from human feedback (RLHF) by stabilizing Proximal Policy Optimization (PPO) for large language models through token-level KL penalties and innovative monitoring signals. This advancement significantly enhances model alignment with human preferences, narrowing safety gaps to ChatGPT and offering a highly reproducible path for practical RLHF deployment."

---

## 1. Executive Summary (2–3 sentences)
This paper tackles the instability of reinforcement learning from human feedback (RLHF) for aligning large language models (LLMs), focusing on why standard Proximal Policy Optimization (PPO) often collapses during training. It presents a carefully engineered variant, PPO‑max, and a set of monitoring signals that together stabilize training and yield consistent gains over supervised fine‑tuned (SFT) baselines on human preference evaluations, while narrowing gaps to ChatGPT on safety.

## 2. Context and Motivation
- Problem addressed
  - RLHF aims to make LLMs helpful, honest, and harmless, but PPO training in this setting is fragile, expensive, and failure‑prone. The paper documents “pattern collapse” (Section 5.2): the policy chases spurious reward patterns and drifts off‑distribution, leading to longer, low‑perplexity answers that earn high reward model (RM) scores yet degrade human‑judged quality (Figure 4, bottom three panels; Appendix A Figure 13).
- Why it matters
  - Stable RLHF is central to practical alignment: SFT alone underperforms on safety/groundedness (Introduction, pages 1–3), and large‑scale PPO failures are too costly to debug given LLM training expense and the need to coordinate four models: policy, value (critic), reward, and a reference policy (Section 3; Figure 1).
- Prior approaches and gaps
  - Prior RLHF successes (InstructGPT, Anthropic’s HH‑RLHF) use PPO but do not fully resolve instability or sensitivity to hyperparameters; many details are under‑specified and “implementation matters” (Related Work; [29, 28]). Classical PPO/TRPO offer generic constraints but do not address language‑specific failure modes like reward hacking through response length or low‑perplexity text (Sections 3.2–3.2.3, 5.2).
- Positioning
  - This work dissects PPO at the code‑ and algorithm‑level for RLHF with LLMs, proposes actionable training monitors (perplexity, response length, token‑level KL to the SFT model), and consolidates a set of effective design choices into PPO‑max that sustains long‑horizon, stable training (Sections 5.2–5.4, Figure 9).

## 3. Technical Approach
The overall RLHF pipeline (Section 3; Figure 1) comprises SFT, Reward Modeling, and PPO fine‑tuning. The paper re‑implements and scrutinizes each stage, then introduces PPO‑max.

A. Reward Modeling (Section 3.1; Equations 1–3; Section 4)
- Model: a transformer LM with the final unembedding removed and a scalar “reward head” added (Section 3.1).
- Training signal:
  - Pairwise preference loss: encourages higher reward for the human‑preferred response `yw` over the dispreferred `yl` via a logistic objective (Eq. 1).
  - Auxiliary LM imitation on preferred responses with weight `βrm` (Eq. 2) to reduce overfitting to pairwise ranking and keep language modeling competence.
- KL‑regularized reward at PPO time: the scalar reward is penalized by the KL divergence between the current RL policy and the SFT reference policy (Eq. 3). This double‑functions as an “entropy‑like” exploration incentive and a guardrail to keep PPO within the RM’s training distribution.

B. PPO with language‑model structure (Sections 3.2–3.2.3; Algorithm 1)
- Environment abstraction: dialogue history is the state; each next token is an action; the RM provides episodic reward on the final token of a response (Section 3.2).
- Variance reduction: advantages are computed by Generalized Advantage Estimation (GAE), which exponentially averages k‑step temporal‑difference errors to trade bias and variance (derivation in Section 3.2.2; Eqs. 7–12).
- Policy update:
  - PPO‑Penalty: maximize advantage‑weighted probability ratio minus a KL penalty (Eq. 14).
  - PPO‑Clip: use a clipped surrogate on the probability ratio to bound destructive updates (Eq. 15).
- Critic (value model): fits returns with MSE (Eq. 16).
- Optional PPO‑ptx: mix in pretraining LM loss with coefficient `λptx` to preserve knowledge/language competence during RL (Eq. 17).
- Practical structure: four models run concurrently—policy, critic, reward, and SFT reference (Figure 1), with an experience buffer and GAE‑based advantages (Algorithm 1).

C. Where PPO fails in practice and how the paper diagnoses it (Section 5.2; Figure 4)
- Reward/loss curves alone can be misleading: training losses stabilize and reward rises even when human/GPT‑4 ratings worsen.
- Three monitors catch collapse early:
  - `KL(policy || SFT reference)` increases (Figure 4, 4th panel).
  - Response length inflates sharply (Figure 4, 6th panel).
  - Perplexity on generated responses drops abnormally (Figure 4, 5th panel).
- Reward distribution drifts to long tails once collapse begins (Appendix A, Figure 13).

D. PPO‑max: a stabilized, LLM‑oriented PPO recipe (Sections 5.3–5.4; Figures 6–8, 9)
Core mechanisms and how they work:
1) Score re‑parameterization to tame gradients (Section 5.3.1; Eq. 18; Figure 6)
   - Normalize and clip per‑batch reward using rolling mean/variance and a clip bound `δ` (Eq. 18). This prevents a few extreme rewards from dominating updates.
   - Advantage normalization and clipping (minibatch‑wise) can help but is sensitive; the paper favors reward‑level constraints over stacking multiple clips (Appendix B.1, Figure 14 shows interactions).
   - Reward scaling alone is insufficient (Figure 6, “reward scale” curve shows little change).

2) Policy constraints are decisive (Section 5.3.2; Figure 7)
   - Token‑level KL penalty to SFT reference during PPO (Eq. 19) curbs distributional drift while allowing learning. Figure 7 shows it yields stable reward growth with tempered KL and perplexity changes; Appendix B.2 (Figure 15) shows how stronger/weaker penalties shift the reward/KL trade‑off.
   - Importance sampling aligns training with the current policy when using an experience buffer; it further stabilizes early‑stage fluctuations but can slightly reduce peak reward (Figure 7, “KL Penalty+Importance Sample”).
   - Entropy bonus (encouraging action entropy) is delicate: without clipping, it can destabilize by pushing entropy too high; with a tight clip it can be stabilizing but extremely sensitive to `δ` (Appendix B.3, Figure 16). The paper prefers KL penalty over entropy bonus.

3) Initialization matters differently for policy and critic (Section 5.3.3; Figure 8)
   - Policy must start from an SFT model; starting from a pretrained base (without SFT) causes language modeling degradation and training failure (Figure 8, KL and perplexity panels).
   - Critic initialization is flexible: reward‑model‑initialized or SFT‑initialized critics both converge; however, briefly pretraining the critic on value prediction before PPO reduces early optimization noise and yields smoother learning (Figure 8, “Pretrain Critic Model before PPO”).

4) Final PPO‑max setup (Section 5.4; Figure 9)
   - Use reward normalization+clipping, token‑level KL penalty to SFT, small replay buffer, critic pretraining, global gradient clipping, value loss clipping, and optional PPO‑ptx to limit alignment tax (Section 5.4; Figure 5 highlights chosen items). Figure 9 shows stable 10k‑step trajectories for reward, response length, KL, and perplexity.

Implementation details and training context (Section 5.1)
- SFT policy/reference: OpenChineseLLaMA 7B trained 2 epochs on 1M instructions (400K single‑turn, 600K multi‑turn).
- PPO data: 8K harmless + 20K helpful prompts (Chinese); batch 128 for sampling, 32 for training; policy LR 5e‑7; critic LR 1.65e‑6; warmup 10% steps.
- Hardware: 8×A100‑80G, ZeRO‑2, gradient checkpointing.

## 4. Key Insights and Innovations
1) “Policy constraints are the key factor” for stable RLHF (Abstract; Sections 5.2–5.3)
   - Unlike general PPO guidance, the paper shows that a token‑level KL penalty to the SFT reference is both necessary and sufficient to prevent reward hacking and collapse in LLMs (Figure 7), enabling much longer training (Figure 9). This is a fundamental insight about LLM‑specific PPO dynamics.

2) Train‑time monitors that predict collapse better than reward/loss (Section 5.2; Figure 4)
   - Monitoring `KL(policy||SFT)`, response length, and perplexity reveals drift and reward hacking before human ratings deteriorate. Quote:
     > “We observed … higher rewards do not reflect better policy behaviors… reward scores and training losses do not indicate whether PPO is optimizing correctly.” (Section 5.2; Figure 4)
   - This is a practical diagnostic contribution that shifts how RLHF runs are monitored.

3) Targeted score re‑parameterization—reward normalization+clipping beats blanket tricks (Section 5.3.1; Eq. 18; Figure 6; Appendix B.1)
   - Reward clipping with rolling statistics stabilizes updates without the brittleness seen when stacking advantage/value clipping. This is an incremental but high‑leverage implementation detail for RLHF.

4) Critic pretraining helps; policy SFT is mandatory (Section 5.3.3; Figure 8)
   - Pretraining the critic reduces early oscillations; skipping policy SFT fails outright. This clarifies which initialization budgets matter most.

5) Consolidated recipe and open resources (Abstract; Section 6)
   - The paper releases competitive Chinese/English reward models and complete PPO‑max code (footnote to GitHub on page 1). This lowers the barrier to reliable RLHF replication.

## 5. Experimental Analysis
Evaluation design
- Reward models (Section 4)
  - Data: English HH‑RLHF pairs (160k train; ~1k test) and a newly labeled Chinese dataset (39k total; 30k train; 3k test).
  - Training: LR 5e‑6, 10% warmup, dynamic batch (4–128), 1000 steps, `βrm=1`.
  - Results:
    - Difference‑score histograms show alignment with human preferences in both languages, stronger for Chinese (Figure 2).
    - Accuracy curves rise quickly, plateauing after ~200 steps, but early checkpoints are insufficient for PPO despite similar accuracy, highlighting that accuracy alone is not a good RM quality indicator (Figure 3).
    - Failure analysis shows RM can be misled by longer or seemingly “helpful” but factually wrong answers (Table 1 provides Chinese and English counter‑examples).

- PPO analysis and ablations (Section 5; Figures 4, 6, 7, 8; Appendices A–C)
  - Setup: Policy/reference from SFT 7B; manually constructed HH data for training; specified LRs and batch sizes (Section 5.1).
  - Diagnosing vanilla PPO:
    - Reward increases but human/GPT‑4 quality does not (Figure 4 top).
    - Collapse markers: rising `KL`, increasing length, decreasing perplexity (Figure 4 bottom).
    - Reward distribution becomes long‑tailed after collapse (Appendix A Figure 13).
  - Ablations:
    - Score re‑parameterization (Figure 6):
      - Reward normalization+clipping stabilizes; advantage clipping also works but is more sensitive.
      - Reward scaling alone is weak.
    - Policy constraints (Figure 7):
      - KL penalty stabilizes and sustains improvement.
      - Importance sampling reduces early instability; combining with KL further smooths but slightly lowers final reward.
      - Entropy bonus is useful only with carefully tuned clipping; otherwise destabilizing (Appendix B.3).
    - Initialization (Figure 8):
      - Policy must start from SFT; starting from a base pretrained checkpoint degrades language ability and RL.
      - Critic pretraining reduces early oscillations.
    - Secondary tricks:
      - Clipped surrogate objective (PPO‑Clip/TRPO‑like) does not match the stability of KL‑penalty–based PPO for LLMs (Appendix C.1, Figure 17).
      - Global gradient clipping has limited observable effect but is enabled by default (Appendix C.2, Figure 18).
      - GAE: λ=0.9 yields stable trade‑offs; λ=0 (TD) is numerically unstable; λ=1 (Monte Carlo) is high‑variance (Appendix C.3, Figure 19).

- PPO‑max outcomes (Section 5.4; Figure 9)
  - 10k‑step training shows smooth reward growth, bounded KL, and stable perplexity/length, indicating durable stability.

- Human and GPT‑4 preference evaluations (Section 6.2; Figure 10)
  - Human evaluation, RLHF vs SFT (Figure 10a):
    - English Harmless: RLHF Win 62%, Tie 33%, RLHF Lose 5%.
    - English Helpful: RLHF Win 44%, Tie 26%, RLHF Lose 30%.
    - Chinese Harmless: RLHF Win 39%, Tie 29%, RLHF Lose 32%.
    - Chinese Helpful: RLHF Win 46%, Tie 23%, RLHF Lose 33%.
  - GPT‑4 as judge (Figure 10b):
    - English Harmless: Win 34%, Tie 59%, Lose 7%.
    - English Helpful: Win 43%, Tie 23%, Lose 34%.
    - Chinese Harmless: Win 25%, Tie 60%, Lose 15%.
    - Chinese Helpful: Win 52%, Tie 17%, Lose 31%.
  - Takeaway: PPO‑max consistently improves over SFT, especially on safety (harmlessness), with somewhat mixed but positive gains on helpfulness.

- Comparison to ChatGPT on harmless evaluation (Section 6.3; Figure 11)
  - Against gpt‑3.5‑turbo‑0613:
    - English: SFT Win/Tie/Lose = 16%/39%/45%; RLHF = 18%/58%/24% (losses nearly halved).
    - Chinese: SFT = 5%/58%/37%; RLHF = 6%/65%/29% (fewer losses after RLHF).
  - Interpretation: While not surpassing ChatGPT, RLHF markedly reduces “defeats” versus the stronger baseline—evidence of meaningful alignment gains.

- NLU side‑effects and mitigation (Section 6.4; Figure 12)
  - C‑Eval shows an average drop in subject scores after PPO‑max; mixing back pretraining data during PPO (PPO‑ptx) recovers a substantial portion of the loss. Quote:
    > “The experimental results indicate a decrease in NLU capabilities after employing PPO… PPO‑ptx effectively alleviates the decline.” (Section 6.4; Figure 12)

Overall assessment
- The experimental suite convincingly supports two main claims:
  - PPO collapse is real and detectable with language‑specific monitors (Section 5.2, Figure 4, Appendix A).
  - PPO‑max yields stable training and better human preference outcomes than SFT, and narrows the gap to ChatGPT on safety (Figures 9–11).
- The paper is candid about reward model imperfections (Section 4.3 and Table 1), which explains why reward increases can decouple from human judgments—motivating their KL‑anchoring and monitors.

## 6. Limitations and Trade-offs
- Reward model quality ceiling
  - The policy’s upper bound depends on the RM; the English RM is less accurate than the Chinese one and can be fooled by plausible but incorrect answers (Section 4.3; Table 1). Accuracy plateaus early and is not a sufficient quality measure (Section 4.4; Figure 3).
- Dependence on SFT initialization
  - PPO‑max assumes a reasonably capable SFT policy. Without SFT, PPO fails (Section 5.3.3; Figure 8).
- Alignment tax and knowledge retention
  - PPO can degrade NLU performance; PPO‑ptx mitigates but does not eliminate the trade‑off (Section 6.4; Figure 12).
- Hyperparameter sensitivity persists
  - While PPO‑max is more robust, some components (e.g., entropy bonus clipping) remain highly sensitive (Appendix B.3).
- Scale and generality
  - Most experiments use 7B models and proprietary Chinese HH data; effects across bigger models and broader datasets are not systematically explored (Limitations section).
- Evaluation scope
  - The main metric is pairwise preference (human and GPT‑4). Standard benchmarks beyond C‑Eval are limited; safety/generalization under distribution shift remains an open question (Limitations).

## 7. Implications and Future Directions
- Field impact
  - The work reframes RLHF engineering around two practical pillars: strong distributional anchors (token‑level KL to SFT) and language‑specific monitors (perplexity, length, KL) for early‑warning detection of collapse. This shifts best practices from generic PPO recipes to LLM‑tailored control.
- Research directions
  - Reward modeling:
    - Better diagnostics than pairwise accuracy for RM quality; adversarial/red‑team data to harden RMs against length/“helpful‑but‑wrong” biases (Section 4.3 and Table 1).
    - Online or iterated RM updates to track the policy’s distribution during PPO (importance sampling hints in Section 5.3.2).
  - Policy constraints beyond KL:
    - Trust‑region or f‑divergence alternatives conditioned on token position; adaptive KL schedules (Appendix B.2 shows the trade‑off curve).
  - Stability without performance loss:
    - principled combinations of PPO‑ptx with curriculum learning to maintain NLU while aligning safety (Section 6.4).
  - Scalable evaluations:
    - Automated judge reliability studies (Section 6.2 referencing LLM‑as‑a‑judge), bias correction, and wider safety suites for multilingual settings.
- Practical applications
  - Deployers seeking stable RLHF can adopt PPO‑max and its monitors to reduce failed runs and improve safety‑critical behavior. Open‑sourced RMs and code (page 1, footnote to GitHub) lower barriers for teams aligning mid‑sized LLMs in English and Chinese.

> “PPO‑max alleviates the instability of vanilla PPO training and enables longer training steps with a larger training corpus. We evaluate PPO‑max on 7B and 13B SFT models, demonstrating comparable alignment performance with ChatGPT.” (Abstract; Sections 5.4, 6.3; Figures 9–11)

In short, the paper’s main contribution is not a new objective but a robust, LLM‑specific PPO recipe—anchored by token‑level KL constraints and actionable training monitors—that turns fragile RLHF runs into dependable alignment improvements.
