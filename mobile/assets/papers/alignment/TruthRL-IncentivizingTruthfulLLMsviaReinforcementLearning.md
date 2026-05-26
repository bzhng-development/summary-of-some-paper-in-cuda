# TruthRL: Incentivizing Truthful LLMs via Reinforcement Learning

**ArXiv:** [2509.25760](https://arxiv.org/abs/2509.25760)

## 🎯 Pitch

TruthRL introduces a new reinforcement learning framework for large language models that directly rewards not only accurate answers, but also proper abstention when uncertain, using a principled ternary reward system. By explicitly distinguishing between correct, incorrect, and uncertain responses, TruthRL dramatically reduces harmful hallucinations and improves model trustworthiness, especially in high-stakes domains. This shift from accuracy-only optimization to truthfulness-driven training sets a new standard for building reliable and responsible AI systems.

---

## 1. Executive Summary

This paper introduces **TruthRL**, a general reinforcement learning framework that directly optimizes the truthfulness of LLMs by explicitly rewarding correct answers, penalizing hallucinations, and treating abstentions as neutral through a ternary reward design (distinguishing correct, incorrect, and "I don't know" responses rather than the binary correct/incorrect signal used in vanilla RL). Evaluated across four knowledge-intensive benchmarks—CRAG, NaturalQuestions, HotpotQA, and MuSiQue—using Qwen2.5-7B-Instruct and Llama3.1-8B-Instruct backbones under both retrieval and non-retrieval setups, TruthRL reduces hallucinations by up to 28.9% and improves truthfulness by 21.1% compared to vanilla RL, establishing that a simple ternary reward structure is sufficient to induce calibrated uncertainty-awareness and that accuracy-driven training methods fundamentally conflate abstention with error, encouraging overconfident hallucination rather than honest acknowledgment of knowledge boundaries.

## 2. Context and Motivation

### The Core Problem: Accuracy-Driven Training Undermines Truthfulness

The fundamental tension this paper addresses is that **optimizing LLMs for factual accuracy alone actively teaches them to hallucinate**. This is not an incidental side effect—it is a direct consequence of the incentive structure embedded in standard training objectives. When a model is trained via supervised fine-tuning (SFT) or standard reinforcement learning with a binary correct/incorrect reward, every output is pushed toward one of two poles: correct answer or wrong answer. The model never learns that "I don't know" is an acceptable—let alone preferred—output when the correct answer lies beyond its knowledge boundary.

Why does this matter? Because in high-stakes domains like medicine, law, or financial advising, a confidently wrong answer inflicts far more damage than a candid abstention. The paper articulates this plainly in Section 1:

> "a model that answers fewer questions correctly while reliably abstaining when uncertain is far more trustworthy than a higher-accuracy model that frequently fabricates plausible but incorrect answers. In high-stakes domains, such misleading answers risk doing far more harm than abstention."

This reframes truthfulness as a **multi-dimensional objective** rather than a synonym for accuracy. As defined in Section 2.1, truthfulness involves three components: correctly answering what the model knows, abstaining when it is uncertain, and minimizing hallucinations. The problem is that existing training methods optimize only the first of these—and in doing so, they actively suppress the second and amplify the third.

### The Incentive Problem: Why Vanilla Training Encourages Guessing

The paper provides a clear incentive-based diagnosis of why standard training methods produce overconfident hallucination. Section 1 states:

> "accuracy-driven methods inherently motivate LLMs to guess rather than abstain from answering when unsure, since the expected incentive for guessing an answer is always higher than that from abstention by design (Kalai et al., 2025)."

The reasoning is straightforward but often overlooked: under a binary reward that awards +1 for correct and −1 for incorrect, an abstention has no defined value—but in practice it is treated as incorrect (receiving −1) because it does not match the ground-truth answer. Meanwhile, a guess has some non-zero probability of being correct, giving it a higher expected reward than abstention, which is guaranteed to be wrong under this scoring regime. The model therefore learns to always produce an answer, even when its internal confidence is effectively zero. This is not a failure of optimization—it is the rational response to the objective it was given.

Section 2.2 makes this explicit for SFT: "the model is trained to always provide an answer, even when unsure, which inevitably encourages hallucinations." For RL, the diagnosis is similar: "vanilla RL is not explicitly designed to recognize uncertainty or abstain when appropriate. As a result, it may substantially increase correctness but still fails to prevent hallucinations."

### The Empirical Evidence: Vanilla Training Destroys Uncertainty Awareness

The paper's preliminary findings (Section 2.3, Figure 2) provide stark evidence of this dynamic. Using Llama3.1-8B-Instruct on the CRAG benchmark, the authors examine **majority@k curves**—the relationship between the number of sampled responses and the rates of correct answers, abstentions, and hallucinations.

The untrained model (prompting baseline) shows a healthy pattern: as more responses are sampled, accuracy improves, hallucination decreases, and uncertainty remains present. This means the base model already possesses an **implicit capability** to gauge its own uncertainty—it sometimes voluntarily abstains, and when it does provide an answer, increased sampling diversity helps surface correct responses.

Vanilla SFT and vanilla RL dismantle this capability:

> "vanilla SFT and RL methods almost completely suppress abstention behavior (i.e., maintaining a near-zero uncertainty rate) and provide only a limited reduction in hallucinations—or even an increased hallucination rate compared to the baseline when k is large."

This is crucial: the problem is not that the base model lacks uncertainty awareness. The problem is that **standard training objectives actively destroy it**. The model enters training with some ability to say "I don't know," and exits training having been taught that saying anything—even something false—is better than admitting ignorance.

Figure 1 (Section 1) illustrates this with a concrete example. A question about ESTA visa requirements for a South Korean researcher receives two possible responses: a confident but factually incorrect hallucination (the model incorrectly asserts that presenting research always requires a B-1 visa, not ESTA), and an honest abstention that correctly identifies the ambiguity and refuses to give a definitive answer. Vanilla SFT/RL would reward the hallucination over the abstention because the hallucination is "closer" to producing an answer—even though the abstention is the correct behavior. TruthRL, by contrast, explicitly penalizes the hallucination while treating the abstention neutrally, aligning the training signal with the desired behavior.

### Prior Approaches and Their Limitations

The paper situates itself against three broad categories of prior work, each of which partially addresses the truthfulness problem but leaves a critical gap.

**Uncertainty-aware fine-tuning (R-Tuning and related methods).** The most directly relevant prior approach is R-Tuning (Zhang et al., 2024), which explicitly trains models to output "I don't know" on questions they cannot answer. The method identifies "unanswerable" questions (through various strategies, including the knowledge boundary probing the paper adopts as a baseline in Section 3.1) and replaces their ground-truth labels with abstention tokens during SFT. The paper acknowledges this as a step in the right direction: R-Tuning indeed reduces hallucination (Table 1 shows it achieves 11.3% hallucination on average without retrieval, compared to 78.7% for vanilla SFT using Qwen2.5-7B-Inst).

However, the paper identifies two specific failure modes that R-Tuning and similar methods exhibit:

1. **Over-conservatism**: The model abstains even on questions within its knowledge boundary. Table 1 shows R-Tuning's accuracy drops sharply—from 30.6% (prompting) to 14.5% on CRAG without retrieval for Qwen2.5-7B-Inst—because the model has been trained to associate uncertainty with abstention globally, rather than learning to discriminate between what it knows and what it doesn't. The paper characterizes this as a fundamental limitation: "such methods require non-trivial annotation on model-specific datasets, leading to limited generalization or overly conservative behavior (e.g., abstaining even when the model has sufficient knowledge)" (Section 1).

2. **Vulnerability to hallucination-baiting inputs**: Table 2 reveals that even R-Tuning—which shows promising overall truthfulness in aggregate metrics—suffers from high hallucination rates on comparison-style questions (e.g., "Which is larger, A or B?"), with a 43.7% hallucination rate versus TruthRL's 16.5%. This suggests that methods based on pre-labeled OOK classifications learn a brittle heuristic rather than genuine uncertainty-awareness.

**Retrieval-Augmented Generation (RAG).** The paper positions RAG not as a competing approach but as an orthogonal knowledge-expansion mechanism that does not independently solve the truthfulness problem. The key limitation noted in Section 1 is that "the retrieved documents in RAG can be noisy or even contain factually incorrect content, potentially misleading the model." Even when retrieval provides high-quality information, the model must still decide when to trust the retrieved content versus its parametric knowledge—a meta-cognitive skill that RAG alone does not instill. The paper's experimental design explicitly evaluates both retrieval and non-retrieval settings to disentangle these effects, with Table 1 showing that retrieval improves all methods but does not eliminate the relative advantage of TruthRL over accuracy-driven baselines.

**Reinforcement learning for LLMs (RLHF, RLVR).** Vanilla RL methods with outcome-based rewards (RLHF, RLVR) represent the dominant post-training paradigm that TruthRL seeks to reformulate. The paper's diagnosis is precise: these methods "conflate abstention with error, thereby discouraging models from producing calibrated 'I don't know' responses" (Section 5.2). This is not merely a conceptual claim—Table 1 provides direct evidence: TruthRL (Binary), which applies GRPO with a standard binary correct/incorrect reward, achieves the highest accuracy (60.3% on CRAG with retrieval using Llama3.1-8B-Instruct) but also produces near-zero uncertainty and substantially higher hallucination than TruthRL with the ternary reward. The binary reward models achieve high accuracy but fail on truthfulness because they optimize a single dimension of a fundamentally three-dimensional problem.

The paper also references several attempts to introduce richer reward structures—uncertainty-aware RL (Xu et al., 2024a; Xue et al., 2024; Lin et al., 2024; Wang et al., 2024c; Li et al., 2025a) and multi-objective optimization for factual faithfulness (Wang et al., 2024a). The paper positions these as incremental improvements that do not fully resolve the core tension: "designing scalable reward signals that reliably capture truthfulness while balancing accuracy and uncertainty remains an open challenge" (Section 5.2). TruthRL's contribution is not proposing a complex reward engineering solution but demonstrating that **a simple ternary reward—correct (+1), hallucination (−1), abstention (0)—is sufficient** when the underlying RL algorithm (GRPO) computes advantages relative to the group mean, which naturally distinguishes abstention from hallucination as having higher relative value.

### How TruthRL Positions Itself

The paper's positioning is grounded in a specific reframing of the objective function. Rather than asking "how can we make models more accurate?" or "how can we make models more conservative?", it asks: **"what reward structure would align the model's behavior with truthfulness as a multi-dimensional construct?"**

This is articulated through the problem formulation in Section 2.1. Truthfulness is defined as:

$$\text{Truthfulness} = w_1 \cdot \text{Acc} + w_2 \cdot \text{Unc} - w_3 \cdot \text{Hall}$$

where $\text{Acc}$ is accuracy, $\text{Unc}$ is abstention rate, and $\text{Hall}$ is hallucination rate. The paper sets $w_1 = 1, w_2 = 0, w_3 = 1$ for evaluation (Section 4.1), meaning truthfulness simplifies to **accuracy minus hallucination rate**. This choice is deliberate: abstention is not directly rewarded in the metric (because $w_2 = 0$, uncertainty has zero weight and is neither penalized nor rewarded), but it serves as a mechanism to avoid hallucinations. A model that correctly answers 40% of questions, abstains on 50%, and hallucinates on 10% scores a truthfulness of 30, while a model that answers 55% correctly and hallucinates on 45% (answering everything) scores only 10—despite having higher accuracy.

TruthRL's ternary reward implements this objective directly: correct answers receive +1, hallucinations receive −1, and abstentions receive 0. The critical insight is that **the advantage computation in GRPO** (Section 3.2, Equation for $\hat{A}_i$) naturally surfaces the relative value of these outcomes. When sampled responses within a group include both hallucinations and abstentions, the abstention's reward of 0 is higher than the hallucination's −1, giving it a positive advantage relative to the group mean. This subtle mechanism is what allows TruthRL to avoid the over-conservatism of R-Tuning (which hard-codes abstention as the correct answer for OOK questions) while still incentivizing abstention over hallucination when the model is uncertain.

The paper explicitly frames this as a shift in philosophy: from **accuracy-driven training** (where any non-correct output is penalized equally) to **truthfulness-driven training** (where the quality of being wrong is distinguished from the quality of admitting ignorance). This is not a modification of existing methods but a redefinition of what the training signal should optimize for. As the paper states in Section 1: "our findings advocate a shift from accuracy-driven to truthfulness-driven methods for developing LLMs."

## 3. Technical Approach

### 3.1 Reader Orientation

TruthRL is a training framework that teaches language models to be truthful—not just accurate—by using a three-way reward signal during reinforcement learning that says "correct answers are good, abstaining is neutral, and hallucinations are bad." The core problem it solves is that standard training objectives treat "I don't know" as just another wrong answer, which teaches models to confidently guess rather than honestly admit uncertainty; TruthRL fixes this by making the reward structure itself distinguish between the quality of being wrong and the quality of knowing you're wrong.

### 3.2 Big-Picture Architecture (Diagram in Words)

The system has five major components arranged in a training pipeline:

1. **Base LLM** — a pretrained instruction-tuned model (Llama3.1-8B-Instruct or Qwen2.5-7B-Instruct) that serves as the starting policy. It already has some implicit ability to express uncertainty when prompted.

2. **Knowledge Boundary Probe** — an optional preprocessing step that identifies which training questions the base model genuinely cannot answer (out-of-knowledge, or OOK, questions). This produces labeled data used for baseline methods (R-Tuning, RFT) and optionally informs enhanced reward designs.

3. **Rollout Generator (vLLM)** — during RL training, the current policy generates groups of candidate responses for each training question. For each question `$x$`, it samples `$G$` responses from the policy, where `$G$` is the group size used by GRPO.

4. **Verifier / Reward Function** — an LLM-based judge (Llama3.3-70B-Instruct) that evaluates each generated response against the reference answer and assigns one of three rewards: `$+1$` for correct, `$0$` for uncertain/abstention, `$-1$` for incorrect/hallucination. This is the ternary reward that defines TruthRL. A binary variant (`$+1$` correct, `$-1$` otherwise) is used for the vanilla RL baseline.

5. **GRPO Policy Optimizer** — the reinforcement learning algorithm that updates the model parameters. It takes the group of sampled responses and their rewards, computes per-response advantages by comparing each reward against the group mean, and then applies a clipped policy gradient update with a KL-divergence penalty to keep the policy from drifting too far from a reference model.

Information flows as follows: a training question enters the system → the current policy generates `$G$` responses → the LLM verifier scores each response as correct, uncertain, or hallucinated → the GRPO optimizer computes advantages and updates the policy → the updated policy generates higher-quality responses in the next iteration, learning to prefer abstention over hallucination when uncertain.

### 3.3 Roadmap for the Deep Dive

- **First**, the formal truthfulness objective and how it motivates the ternary reward design, since this is the conceptual foundation that distinguishes TruthRL from accuracy-driven methods.
- **Second**, the knowledge boundary probing mechanism and how it generates training data for baseline SFT methods (R-Tuning, RFT), since these baselines represent the closest prior approach and their limitations motivate TruthRL's RL-based alternative.
- **Third**, the GRPO algorithm and how its advantage computation interacts with the ternary reward to naturally surface abstention as preferable to hallucination without requiring hard-coded OOK labels, since this is the core technical mechanism that makes TruthRL work.
- **Fourth**, the reward design space—binary vs. ternary, and the knowledge-enhanced and reasoning-enhanced variants—since the paper's central empirical claim is that a simple ternary reward outperforms both simpler (binary) and more complex alternatives.
- **Fifth**, the training infrastructure and hyperparameters, since practical reproducibility depends on these details.
- **Sixth**, the verifier implementation and why an LLM-based judge is necessary rather than rule-based string matching, since the reliability of the reward signal is a prerequisite for the entire approach.

### 3.4 Detailed, Sentence-Based Technical Breakdown

This is primarily a **methodology paper** whose core idea is that replacing the binary correct/incorrect reward in reinforcement learning with a ternary reward that separately recognizes abstention is sufficient to produce models that balance factual accuracy with calibrated uncertainty expression.

---

#### The Truthfulness Objective and Its Implications for Reward Design

The paper defines truthfulness as a multi-dimensional objective that cannot be reduced to accuracy alone. Section 2.1 formalizes this:

$$\text{Truthfulness} = w_1 \cdot \text{Acc} + w_2 \cdot \text{Unc} - w_3 \cdot \text{Hall}$$

where `$\text{Acc}$` is the fraction of questions answered correctly, `$\text{Unc}$` is the fraction where the model abstains (e.g., answers "I don't know"), `$\text{Hall}$` is the fraction of responses that are factually incorrect, and `$w_1, w_2, w_3 \geq 0$` are weights controlling the desired balance among the three dimensions.

**What it computes:** a single scalar score where higher is better, with correctness contributing positively, hallucinations contributing negatively, and uncertainty optionally contributing positively (via `$w_2$`). In the paper's experimental setup (Section 4.1), the weights are set to `$w_1 = 1, w_2 = 0, w_3 = 1$`, meaning truthfulness simplifies to **accuracy minus hallucination rate**. This produces a score in the range `$[-100, 100]$` when expressed as percentages.

**Why this form:** setting `$w_2 = 0$` means abstention is not directly rewarded—the model doesn't get credit for refusing to answer. Instead, abstention is valuable **instrumentally**: it serves as a mechanism to avoid hallucinations, which are penalized. If a model correctly answers 40% of questions, abstains on 50%, and hallucinates on 10%, its truthfulness is `$40 - 10 = 30$`. If another model answers 55% correctly but hallucinates on the remaining 45% (answering everything), its truthfulness is `$55 - 45 = 10$`—despite having 15 percentage points higher accuracy. This weighting reflects the paper's central value judgment: in high-stakes domains, a confident hallucination is more damaging than a refusal to answer.

The key design insight is that the training reward function should be **aligned with this evaluation metric**. Under an accuracy-only objective, both abstention and hallucination are simply "not correct" and receive equivalent negative signals. Under the truthfulness objective, hallucination is explicitly worse than abstention. The ternary reward implements this directly in the RL training loop.

---

#### Knowledge Boundary Probing for Baseline Construction

Before describing TruthRL itself, the paper establishes strong baselines by probing the base model's knowledge boundary on the training set. This process identifies questions the base model genuinely cannot answer, enabling supervised training methods that teach explicit abstention behavior.

**Procedure.** For each training question `$x_i$`, the base model is prompted to generate 256 responses (sampled with temperature to produce diverse outputs). The question is marked as **out-of-knowledge (OOK)** if **none** of the 256 responses is correct according to the verifier. This operationalizes the definition of "outside the model's knowledge" empirically: if the model cannot produce a correct answer across 256 independent attempts, the question is considered fundamentally beyond its capability.

**What this computes:** a binary label for each training question—OOK or non-OOK—based on the model's empirical pass@256 rate. Questions with zero correct samples among 256 are OOK; all others are non-OOK. This is a model-specific notion of difficulty: the same question might be OOK for a 7B model but non-OOK for a 32B model.

**Why this form:** 256 is a sufficiently large sample count to provide reasonable confidence that a zero pass rate reflects genuine inability rather than sampling noise. The binary threshold (zero vs. any correct) is chosen for simplicity and because the downstream baselines use the OOK label categorically (either the ground-truth answer is replaced with "I don't know" in full, or it is not). A continuous difficulty estimate (e.g., pass@1 rate) could enable more nuanced strategies, but the baseline methods the paper compares against (R-Tuning, RFT) require binary OOK decisions.

**Usage in R-Tuning (Zhang et al., 2024).** OOK questions have their ground-truth answer replaced with "I don't know." The model is then trained with standard SFT on this modified dataset, learning to produce the abstention token when the question is outside its knowledge and the original answer when it is within its knowledge.

**Usage in Rejection Sampling Fine-Tuning (RFT).** Rather than directly learning ground-truth answers, RFT (Yuan et al., 2023) trains on **reasoning traces generated by the model itself**—a form of self-training. The paper extends this to incorporate uncertainty: for each training question, the model generates multiple reasoning traces. For OOK questions, the trace that concludes with "I don't know" is selected as the target response. For non-OOK questions, the trace that leads to the correct answer is selected. This provides richer supervision than R-Tuning because the model learns not just what to output but how to reason toward the correct decision (whether that decision is an answer or an abstention).

The knowledge boundary probing results are also optionally used in enhanced reward designs for TruthRL (Section 3.2, knowledge-enhanced variant), where OOK questions receive positive reward for abstention and negative reward for any answer, while non-OOK questions follow the standard ternary scheme.

---

#### TruthRL: The GRPO Training Framework

TruthRL is implemented using **Group Relative Policy Optimization (GRPO)** (Shao et al., 2024), an online reinforcement learning algorithm specifically designed for training language models with reward signals. The choice of GRPO over alternatives like PPO or DPO is consequential: GRPO's **group-based advantage computation** is what makes the ternary reward effective, because advantages are defined relative to the mean reward of a group of sampled responses, not against an absolute baseline.

**The GRPO objective.** The model parameters `$\theta$` are updated to maximize:

$$L_{\text{GRPO}}(\theta) = -\mathbb{E}_{x \sim \mathcal{D}, \{y_i\}_{i=1}^G \sim \pi_{\theta_{\text{old}}}(\cdot|x)} \left[ \frac{1}{G} \sum_{i=1}^G \frac{1}{|y_i|} \sum_{t=1}^{|y_i|} \min \left( w_{i,t}(\theta) \hat{A}_i, \text{clip}(w_{i,t}(\theta), 1 - \epsilon, 1 + \epsilon) \hat{A}_i \right) - \beta D_{\text{KL}} (\pi_\theta || \pi_{\text{ref}}) \right]$$

where `$x$` is a question sampled from the training distribution `$\mathcal{D}$`, `$\{y_i\}_{i=1}^G$` are `$G$` candidate responses sampled from the old policy `$\pi_{\theta_{\text{old}}}$`, `$|y_i|$` is the number of tokens in response `$y_i$`, `$w_{i,t}(\theta)$` is the importance ratio `$\frac{\pi_\theta(y_{i,t} | x, y_{i,<t})}{\pi_{\theta_{\text{old}}}(y_{i,t} | x, y_{i,<t})}$` measuring how much more likely the new policy makes each token compared to the old policy, `$\hat{A}_i$` is the estimated advantage for response `$y_i$` (computed from the group of rewards), `$\epsilon$` is the clipping threshold (set to `$\epsilon = 0.2$`), `$\text{clip}$` constrains the importance ratio to `$[1 - \epsilon, 1 + \epsilon]$` to prevent overly large policy updates, `$\beta$` is the KL-divergence regularization coefficient (set to `$\beta = 0.001$`), and `$D_{\text{KL}}(\pi_\theta || \pi_{\text{ref}})$` is the Kullback-Leibler divergence between the current policy and a reference policy `$\pi_{\text{ref}}$` (typically the starting checkpoint).

**What it computes:** the negative of the per-token clipped advantage, averaged over all tokens in all responses in a group, minus a KL penalty that prevents the policy from deviating too far from the reference model. The clipping mechanism (the `$\min$` of clipped and unclipped importance ratios) ensures that tokens whose probability changed too much relative to the old policy have their gradient contribution truncated, preventing destructive large updates. The empirical expectation is over the training questions and the sampled responses.

**Why this form:** GRPO inherits the clipped surrogate objective from PPO (Schulman et al., 2017), which has become the standard for stable policy gradient methods in language model training. The key difference from PPO is that GRPO computes advantages `$\hat{A}_i$` using **the group of responses** rather than learning a separate value function (critic). This eliminates the need to train and maintain a value network, reducing memory and computational overhead. The group-based normalization is what makes the ternary reward effective, as explained below.

**Advantage computation — the core mechanism.** For a group of `$G$` responses `$\{y_1, y_2, \ldots, y_G\}$` to question `$x$`, the reward function `$r(x, y)$` assigns a scalar to each response. The advantage for response `$y_i$` is computed as:

$$\hat{A}_i = \frac{r(x, y_i) - \text{mean}(\{r(x, y_j)\}_{j=1}^G)}{\text{std}(\{r(x, y_j)\}_{j=1}^G)}$$

where `$\text{mean}(\{r(x, y_j)\}_{j=1}^G)$` is the arithmetic mean of rewards across the group and `$\text{std}(\{r(x, y_j)\}_{j=1}^G)$` is the standard deviation of rewards across the group.

**What it computes:** a z-score normalized advantage: how many standard deviations above or below the group mean a response's reward lies. Responses with above-average rewards receive positive advantages (increase their probability); responses with below-average rewards receive negative advantages (decrease their probability).

**Why this form:** the group-relative normalization is the linchpin that makes the ternary reward work. Consider a simplified group of two responses `$y_1$` and `$y_2$` where `$y_1$` expresses abstention and `$y_2$` hallucinates.

- **Under binary reward:** both receive `$-1$`, the group mean is `$-1$`, both advantages are zero, and neither response is preferred over the other. The policy update cannot distinguish abstention from hallucination.
- **Under ternary reward:** `$y_1$` receives `$0$` and `$y_2$` receives `$-1$`, the group mean is `$-0.5$`, `$y_1$`'s advantage is positive (`$\hat{A}_1 = \frac{0 - (-0.5)}{\text{std}} > 0$`) and `$y_2$`'s advantage is negative (`$\hat{A}_2 = \frac{-1 - (-0.5)}{\text{std}} < 0$`). The policy update increases the probability of abstention and decreases the probability of hallucination.

Even when correct answers are present in the group, the relative ordering is preserved: correct (`$+1$`) > abstention (`$0$`) > hallucination (`$-1$`). The policy learns to prefer correctness when possible, to prefer abstention when correctness is unavailable, and to avoid hallucination in all cases. This implicit preference ordering emerges from the reward values themselves, without any additional rules or heuristics.

**The KL penalty.** The term `$\beta D_{\text{KL}}(\pi_\theta || \pi_{\text{ref}})$` with `$\beta = 0.001$` serves as a regularizer. The KL divergence `$D_{\text{KL}}(\pi_\theta || \pi_{\text{ref}}) = \mathbb{E}_{y \sim \pi_\theta} \left[ \log \frac{\pi_\theta(y)}{\pi_{\text{ref}}(y)} \right]$` measures how much the current policy distribution differs from the reference policy. By subtracting this from the objective (or equivalently, penalizing it in the loss), the optimizer keeps the policy near the reference model, preventing reward hacking (where the policy learns to exploit quirks of the reward function) and preserving the base model's general language capabilities. The choice of `$\beta = 0.001$` is relatively small, allowing substantial policy change under strong reward signals while providing a safety net against degenerate solutions.

**GRPO hyperparameters (Appendix A).** The training uses a constant learning rate of `$1 \times 10^{-6}$`, batch size of `$64$` (questions per update), KL coefficient `$\beta = 0.001$`, clip ratio `$\epsilon = 0.2$`, maximum context length of 16,384 tokens, and maximum generated tokens of 2,048. Rollout sampling during training uses temperature `$= 1.0$` and top-p `$= 1.0$` (no nucleus filtering), encouraging diverse response generation to populate the group with a mix of correct, uncertain, and hallucinated responses. The training is run on 8 NVIDIA H100 GPUs with 80GB memory using full-parameter fine-tuning with DeepSpeed ZeRO-3 offload, gradient checkpointing, and FlashAttention-2.

---

#### The Ternary Reward Design and Its Variants

The reward function is the central design element of TruthRL. The paper explores a design space ranging from simple (binary) to moderate (ternary) to complex (knowledge-enhanced, reasoning-enhanced), with the key empirical finding that the ternary reward offers the best balance of simplicity and effectiveness.

**Binary reward (vanilla RL baseline).** The simplest reward structure:

$$r_{\text{binary}}(x, y) = \begin{cases} +1, & \text{if } y \text{ is correct} \\ -1, & \text{otherwise} \end{cases}$$

where correctness is determined by an LLM judge (Llama3.3-70B-Instruct) comparing the generated answer against the reference answer.

**What it computes:** exactly the same signal as standard RLVR/RLHF: correct answers are good, everything else (hallucination, abstention, invalid formatting) is equally bad.

**Why this form (and its problems):** binary rewards are the default in RL for reasoning (Guo et al., 2025; Shao et al., 2024) because they are simple and effective for tasks where the goal is maximizing accuracy. However, as Section 2.3 demonstrates, this formulation "conflates abstention with error" and produces models with near-zero uncertainty rates (Table 1: TruthRL (Binary) shows uncertainty rates close to 0% across all settings). The model learns that producing any answer—even a guess—is better than admitting ignorance, because both abstention and hallucination receive `$-1$` but a guess has some non-zero probability of being correct.

**Ternary reward (TruthRL default).** The paper's core proposal:

$$r_{\text{ternary}}(x, y) = \begin{cases} +1, & \text{if } y \text{ is correct} \\ 0, & \text{if } y \text{ is uncertain (abstention)} \\ -1, & \text{if } y \text{ is incorrect (hallucination)} \end{cases}$$

**What it computes:** a three-level reward that explicitly distinguishes the quality of "giving a wrong answer" from the quality of "refusing to answer." The uncertainty level (`$0$`) is identified when the model's output contains an abstention signal such as "I don't know" (as specified in the inference prompt template, Appendix B, Table 9—the model is instructed to answer "I don't know" if uncertain and to enclose the final answer in `\boxed{}`).

**Why this form:** the three reward levels directly mirror the three components of the truthfulness metric (accuracy, uncertainty, hallucination). The key design decision is setting abstention's reward to `$0$` rather than, say, `$+0.5$` or `$-0.5$`. A value of `$0$` makes abstention **neutral**—it neither contributes positively nor negatively to the raw reward, but gains positive advantage whenever the group contains hallucinations (which shift the mean downward). If abstention were rewarded positively (e.g., `$+0.5$`), the model would learn to abstain even when it has knowledge, leading to over-conservatism (the R-Tuning failure mode). If abstention were penalized (e.g., `$-0.5$`), it would be closer to the binary reward and the model would avoid it. The `$0$` value achieves the desired behavior: the model abstains when it cannot be correct, but does not abstain when correctness is achievable.

**Knowledge-enhanced variant.** This variant modifies the ternary reward by incorporating the knowledge boundary probing results (Section 3.1) to provide different reward structures depending on whether a question is out-of-knowledge (OOK):

$$r_{\text{knowledge}}(x, y) = \begin{cases} +1, & \text{if } x \text{ is OOK and } y \text{ is uncertain} \\ -1, & \text{if } x \text{ is OOK and } y \text{ is not uncertain} \\ \text{standard ternary}, & \text{if } x \text{ is not OOK} \end{cases}$$

For non-OOK questions, the standard ternary reward applies. For OOK questions, abstention is explicitly rewarded (`$+1$`) and any attempt to answer (whether hallucinated or coincidentally correct) is penalized (`$-1$`).

**What it computes:** a context-sensitive reward where the desirability of abstention depends on whether the question is fundamentally answerable by this model. For questions where the model truly lacks knowledge, abstention is treated as the correct behavior; for questions within the model's capability, the standard ternary distinction (correct preferred over abstention) applies.

**Why this form:** this is an attempt to address a potential weakness of the standard ternary reward: what if the model learns to abstain on questions it could answer correctly? By explicitly marking OOK questions and rewarding abstention only on those, the knowledge-enhanced variant aims to provide a more precise training signal. However, Table 3 shows that this variant **underperforms the standard ternary reward** on average truthfulness (23.2 vs. 25.6) while achieving slightly lower hallucination (18.9 vs. 18.8). The paper does not deeply analyze why, but a plausible explanation is that the binary OOK labeling is noisy (256 samples may miss low-probability correct responses) and that the hard boundary between OOK and non-OOK creates a cliff in the reward landscape that is difficult for the policy to navigate smoothly.

A knowledge-enhanced variant is also applied on top of the binary reward, where OOK questions assign `$+1$` to uncertain responses and `$-1$` to others. Table 3 shows this improves over the standard binary reward (average truthfulness 11.7 vs. 4.5) but still significantly underperforms the ternary reward, because for non-OOK questions the binary structure still conflates abstention with hallucination.

**Reasoning-enhanced variant.** This variant adds a reward component that evaluates the **quality of the reasoning process**, not just the final outcome. Three heuristics are explored (Section 4.6), all built on top of the outcome-based ternary reward `$r_{\text{outcome}}$`:

1. **Multiplicative:** `$r_{\text{final}} = r_{\text{outcome}} \cdot (1 + r_{\text{reason}})$` — the outcome reward is scaled by reasoning quality plus one. For correct outcomes (`$r_{\text{outcome}} = 1$`), the final reward ranges from `$1$` to `$2$` depending on reasoning quality; for incorrect (`$-1$`), it ranges from `$-1$` to `$-2$`; for uncertain (`$0$`), it remains `$0$` regardless of reasoning. This disproportionately encourages good reasoning when the outcome is correct.

2. **Additive:** `$r_{\text{final}} = r_{\text{outcome}} + \lambda \cdot r_{\text{reason}}$` with `$\lambda = 0.5$` — the reasoning reward is a complementary signal added to the outcome reward. Even when the outcome is neutral (abstention, `$r_{\text{outcome}} = 0$`), good reasoning can earn positive reward. This encourages the model to reason well even when it ultimately abstains.

3. **Conditional:** `$r_{\text{final}} = r_{\text{outcome}} \cdot r_{\text{reason}}$` when `$r_{\text{outcome}} = 1$`, and `$r_{\text{final}} = r_{\text{outcome}}$` otherwise — reasoning quality only matters when the outcome is correct. This enforces strict alignment: good reasoning is only rewarded when it leads to correct answers, avoiding the risk of rewarding eloquent but incorrect reasoning.

The reasoning quality score `$r_{\text{reason}} \in \{0, 1\}$` is produced by a second LLM judge (also Llama3.3-70B-Instruct) using a prompt that evaluates whether the reasoning "provides precise and relevant information" (Appendix B, Table 12).

**Results for reasoning-enhanced variants (Table 8).** The outcome-only ternary reward achieves a truthfulness of 37.2 and a reasoning score of 56.6. The multiplicative variant achieves similar truthfulness (37.0) but lower reasoning score (54.7)—surprisingly, emphasizing reasoning in the reward actually reduces reasoning quality. The additive variant achieves slightly lower truthfulness (36.1) but the highest reasoning score (59.1), suggesting that decoupling the reasoning signal from the outcome reward helps the model learn to reason well independently of whether it gets the right answer. The conditional variant shows lower values on both metrics (35.6 truthfulness, 55.1 reasoning). The paper interprets these mixed results as evidence that "explicitly optimizing reasoning quality requires non-trivial design to balance multiple objectives" and that the simple outcome-based ternary reward already implicitly improves reasoning ability (the reasoning score jumps from 50.2 for prompting to 56.6 for TruthRL without explicit reasoning rewards).

---

#### The LLM Verifier: Why Rule-Based Matching Fails

A critical infrastructure component is the **verifier** that assigns rewards during training and evaluates correctness during testing. The paper uses an LLM-based judge (Llama3.3-70B-Instruct) rather than rule-based string matching, and Section 4.5 demonstrates that this choice is not incidental—it is essential for training stability.

**LLM judge protocol (Appendix B, Table 11).** The judge receives the question, the reference answer, and the model's predicted answer. It outputs a JSON object with an "explanation" field and a "score" field (`$1$` or `$0$`). The judge prompt includes specific instructions for handling edge cases: numeric answers must "almost exactly match" the ground truth; self-contradictory predictions score `$0$`; predictions that don't answer the question score `$0$`; concise summaries of the ground truth score `$1$`; set-valued answers must contain exactly the same items.

**Rule-based verifier failure (Table 5).** When the LLM verifier is replaced with exact string matching, the model "collapses into overly conservative behavior, abstaining on the vast majority of queries." The truthfulness score becomes `$-3.6$` (negative, indicating poor performance) with a hallucination rate of only `$3.6\%$`. This happens because "the predicted answer rarely matches the reference answer in exact string form, causing rule-based verifiers to misclassify many correct responses." The model learns that producing any answer is likely to be scored as incorrect, so it converges on always abstaining—which is the only output that reliably avoids negative reward.

**Why the LLM verifier works.** The LLM judge can handle semantic equivalence (e.g., "New York City" vs. "NYC"), partial correctness, formatting variations, and nuanced errors. This provides a more accurate reward signal that correctly identifies genuine correct answers even when they don't match the reference string exactly. The result is a stable training process where the model can learn to prefer correct answers over abstention without being penalized for formatting differences.

**Robustness across judges (Table 6).** An important robustness check: the trained model is evaluated under three different judge models—Llama3.3-70B-Instruct, Qwen2.5-72B-Instruct, and Gemma3-27B-Instruct. TruthRL achieves the highest truthfulness and lowest hallucination under all three judges (average truthfulness 37.5, hallucination 19.3), with the relative ranking of methods preserved across judges. This rules out the possibility that TruthRL learned to exploit idiosyncrasies of a specific judge model.

---

#### Training Pipeline: From Data to Trained Model

The training pipeline integrates the components described above into a concrete procedure.

**Step 1: Training data.** The model is trained on the **CRAG** benchmark (Yang et al., 2024a) and evaluated across all four benchmarks (CRAG, NaturalQuestions, HotpotQA, MuSiQue). This is an important design choice: training on a single dataset and evaluating on multiple datasets tests whether the learned truthfulness behavior generalizes to unseen question distributions. For the retrieval setup, up to 50 web pages are retrieved per question using the question text as a search query; for other datasets, the 2018 Wikipedia dump with the E5 retriever (Wang et al., 2024b) is used, following the Search-R1 setup (Jin et al., 2025).

**Step 2: Base model initialization.** The base model is an instruction-tuned checkpoint (Llama3.1-8B-Instruct or Qwen2.5-7B-Instruct). No additional SFT initialization is performed before RL training for TruthRL—the model starts from the instruction-tuned checkpoint and learns the truthfulness behavior entirely through RL. For baseline SFT methods (R-Tuning, RFT), the OOK-labeled training data is used for supervised fine-tuning before evaluation.

**Step 3: GRPO training loop.** The RL training proceeds in online iterations:
- For each training question, the current policy generates `$G$` responses (where `$G$` is the group size, implicitly determined by the batch configuration).
- Each response is scored by the LLM judge as correct (`$+1$`), uncertain (`$0$`), or hallucinated (`$-1$`).
- Advantages are computed using the group mean and standard deviation.
- The policy is updated using the clipped GRPO objective with KL regularization.
- The reference policy `$\pi_{\text{ref}}$` is typically the initial checkpoint, kept frozen throughout training.

**Step 4: Inference and evaluation.** For evaluation, the trained model uses **greedy decoding** (temperature = 0) to ensure deterministic, reproducible outputs. The inference prompts (Appendix B, Tables 9 and 10) explicitly instruct the model to enclose its final answer in `\boxed{}` and to answer "I don't know" if uncertain. The LLM judge (separate from training) scores each response against the reference answer. The truthfulness score is computed as accuracy minus hallucination rate.

**RL vs. offline/semi-online alternatives (Table 4).** The paper compares online GRPO against offline DPO and iterative DPO. Offline DPO uses fixed preference pairs (constructed from OOK/non-OOK data) and shows limited gains (average truthfulness `$-10.1$`, actually worse than prompting). Iterative DPO improves over several iterations (reaching 12.6 truthfulness at iteration 3) but then regresses at iteration 4 (`$-2.0$`), suggesting that repeated offline fine-tuning cannot stably balance exploration and exploitation. Online GRPO achieves far superior results (25.6 truthfulness) because the policy continuously generates new on-policy responses, preventing the distribution shift that degrades offline methods.

---

#### Summary of Design Choices and Their Justifications

- **Ternary reward with abstention at 0** (not positive, not negative): avoids both over-conservatism (if abstention were rewarded) and hallucination encouragement (if abstention were penalized). The `$0$` value gains positive advantage only when hallucinations are present in the group, creating an implicit preference ordering through relative comparison.
- **GRPO with group-based advantage** (not PPO with learned value function): the group normalization is what activates the ternary reward's ability to distinguish abstention from hallucination. A learned value function would need to predict the expected reward for each state, introducing complexity and potential inaccuracy.
- **GRPO hyperparameters** (`$\beta = 0.001$`, `$\epsilon = 0.2$`, constant LR `$1 \times 10^{-6}$`): small KL penalty allows substantial policy change; standard clip ratio prevents destructive updates; constant learning rate avoids the complexity of scheduling.
- **LLM judge for verification** (not rule-based string matching): handles semantic equivalence, preventing reward signal corruption that would occur if correct answers were misclassified as incorrect due to formatting differences.
- **Training on CRAG, evaluating on four benchmarks**: tests generalization of truthfulness behavior beyond the training distribution.
- **No SFT warmup before RL**: the instruction-tuned base model already follows the output format (using `\boxed{}` and expressing uncertainty when prompted), allowing RL to directly optimize for truthfulness without needing supervised initialization.
- **Temperature = 1.0 during training, temperature = 0 during evaluation**: high temperature during training ensures diverse group composition (mix of correct, uncertain, hallucinated responses), which is necessary for the ternary advantage to provide meaningful signal. Greedy decoding during evaluation ensures reproducibility.

## 4. Key Insights and Innovations

### Innovation 1: Reframing Truthfulness as a Multi-Dimensional Objective That Inverts the Priority Between Accuracy and Calibration

The paper's most fundamental conceptual contribution is not the ternary reward itself—it is the **reframing of what "truthfulness" means for LLMs** and the corresponding inversion of priorities between accuracy and calibrated abstention. Before this work, the dominant paradigm treated truthfulness as essentially a synonym for factual accuracy: a model is truthful if it produces correct answers. Methods for improving truthfulness focused on expanding the model's knowledge (via retrieval or fine-tuning on factual data) or improving its reasoning (via chain-of-thought or search), both of which aim to increase the probability of generating correct answers. Uncertainty expression—saying "I don't know"—was treated as a secondary concern at best and as a failure mode at worst.

TruthRL fundamentally redefines the problem. The truthfulness metric itself (accuracy minus hallucination) encodes a specific value judgment: **a hallucination is worse than an abstention, even though neither is a correct answer**. This is not a marginal preference—it is a categorical distinction. Under the paper's formulation, a model that answers 40% of questions correctly, abstains on 50%, and hallucinates on 10% achieves a truthfulness score of 30, while a model that answers 55% correctly and hallucinates on the remaining 45% scores only 10. The first model is three times more truthful despite being 15 percentage points less accurate.

Why is this a genuine intellectual shift rather than an obvious observation? Because it inverts the implicit weighting that governs nearly all existing LLM training. SFT, RLHF, and RLVR all optimize objectives where any non-correct output receives equivalent—and equivalently negative—supervision. This creates what the paper, citing Kalai et al. (2025), identifies as the **incentive structure for guessing**: "the expected incentive for guessing an answer is always higher than that from abstention by design." The paper's reframing recognizes that this is not a bug in the optimization—it is the rational response to an objective that fails to distinguish between the *type* of error being made. A model that has learned that "abstention = incorrect = hallucination" will always prefer to produce an answer, because even a random guess has a non-zero probability of being correct, while admitting ignorance guarantees punishment.

Prior work on uncertainty-aware training (Cheng et al., 2024; Yang et al., 2024b; Zhang et al., 2024) implicitly recognized this problem but addressed it by **hard-coding abstention behavior** through supervised fine-tuning on OOK-labeled data. R-Tuning trains models to output "I don't know" on questions they cannot answer—but this treats abstention as a learned response to a specific category of question rather than as a **calibrated expression of the model's own internal uncertainty**. The paper's diagnosis of why this fails is precise: R-Tuning's models "achieve much lower hallucination with little to no compromise in accuracy" in aggregate (Table 1), but they exhibit over-conservatism by abstaining even when they have sufficient knowledge, and they remain vulnerable to hallucination-baiting inputs (Table 2, 43.7% hallucination for R-Tuning on comparison questions vs. 16.5% for TruthRL). This indicates that R-Tuning learns a **brittle heuristic** ("when the question looks unfamiliar, say I don't know") rather than genuine knowledge-boundary recognition.

TruthRL's reframing solves this differently: instead of teaching the model *which* questions to abstain on through labeled examples, it teaches the model that abstention is preferable to hallucination *in general*, while correctness is preferable to both. The model learns to calibrate its own uncertainty because the reward structure makes calibration instrumentally valuable—abstaining on a question you would otherwise get wrong improves your expected reward. This is a fundamentally different learning dynamic than supervised approaches, and it explains why TruthRL achieves both higher accuracy and lower hallucination than R-Tuning (Table 1: on CRAG with retrieval using Llama3.1-8B-Instruct, TruthRL achieves 56.6% accuracy vs. R-Tuning's 48.4%, with 19.4% hallucination vs. 33.1%).

The significance of this reframing extends beyond the specific method. It establishes that **truthfulness is not a property that emerges from accuracy-maximizing training**—it requires explicit optimization as a distinct objective with its own structure. This has implications for how the field should think about model evaluation (accuracy alone is insufficient), reward design (binary signals are fundamentally misaligned with truthfulness), and the goals of post-training (producing trustworthy models requires optimizing for more than correctness). The paper's central advocacy—"a shift from accuracy-driven to truthfulness-driven methods for developing LLMs" (Section 1)—is not incremental; it is a call to redefine what post-training optimization should target.

---

### Innovation 2: Demonstrating That Standard Training Objectives Actively Destroy Existing Uncertainty Awareness

One of the paper's most striking empirical findings is not that TruthRL improves truthfulness—it is that **vanilla SFT and RL actively degrade** a capability the base model already possesses. Section 2.3's preliminary findings (Figure 2) reveal that the untrained Llama3.1-8B-Instruct model, under simple prompting, exhibits a healthy uncertainty-awareness profile: as the number of sampled responses increases, hallucination decreases, accuracy improves, and a meaningful abstention rate is maintained. This means the base model already has some implicit ability to recognize when it doesn't know something and to occasionally express that uncertainty.

Vanilla SFT and RL **dismantle this capability**. The majority@k curves in Figure 2 show that after SFT or RL training with binary accuracy rewards, the uncertainty rate collapses to near zero at all values of k, while hallucination rates either stay flat or increase. The model enters training with the ability to say "I don't know" and exits training having been taught that saying anything—even fabricating an answer—is preferable to admitting ignorance.

Why is this finding intellectually significant? Because it reframes the hallucination problem in a way that contradicts the implicit assumption behind most mitigation work. The dominant narrative is that LLMs hallucinate because they lack knowledge, or because their training data contains inaccuracies, or because their reasoning is imperfect—and therefore the solution is to add knowledge (RAG), clean the data, or improve reasoning (chain-of-thought, search). The paper's finding suggests a different diagnosis: **hallucination is not primarily a knowledge problem; it is an incentive problem**. The base model *already knows* how to express uncertainty to some degree, but standard training objectives systematically punish that behavior.

This is a more fundamental critique than it first appears. It implies that many widely-used training techniques—SFT on QA datasets, RLHF with binary preference signals, RLVR with accuracy rewards—are actively making models less truthful, even as they improve accuracy. The accuracy gains are real, but they come at the cost of destroying the model's uncertainty calibration, producing a system that is simultaneously more accurate and less trustworthy—exactly the dangerous combination the paper identifies for high-stakes applications.

The finding also provides a unified explanation for a pattern of conflicting results in the literature. Studies that report SFT improving factuality are typically measuring accuracy on in-distribution questions; studies that report SFT increasing hallucination are typically measuring behavior on out-of-distribution or difficult questions. Both can be simultaneously true: SFT improves accuracy where the model has knowledge (by reinforcing correct answers) while increasing hallucination where it doesn't (by punishing abstention). The paper resolves this apparent contradiction by showing that both effects are consequences of the same incentive structure.

This insight is supported quantitatively throughout the paper's experiments. Table 1 shows that vanilla SFT reduces accuracy compared to prompting on out-of-distribution benchmarks (e.g., Qwen2.5-7B-Inst drops from 22.7% average accuracy with prompting to 21.3% with SFT in the non-retrieval setting) while dramatically increasing hallucination (from 61.4% to 78.7%). Figure 3b shows that on difficult CRAG questions where almost no method produces correct answers, SFT achieves 0% uncertainty—meaning it produces confident hallucinations on every single question—while TruthRL abstains on 84.5% of them. The paper's conceptual contribution is recognizing that these are not separate phenomena requiring separate solutions; they are two manifestations of the same underlying incentive misalignment.

---

### Innovation 3: The Ternary Reward as a Minimal Sufficient Intervention—and Why More Complex Rewards Underperform

After establishing that binary rewards destroy uncertainty awareness, the natural question is: what alternative reward structure would prevent this? One might expect that solving a multi-dimensional optimization problem (maximize accuracy, minimize hallucination, calibrate abstention) would require a complex reward function with carefully tuned weights, per-question difficulty estimates, or auxiliary losses. The paper's most surprising empirical finding is that **a simple ternary reward—with abstention placed at exactly zero, not a positive or negative value—is sufficient**, and that more complex reward designs (knowledge-enhanced, reasoning-enhanced) generally **perform worse**.

The sufficiency of the ternary reward is not obvious a priori. Placing abstention at zero means the model receives no direct positive reinforcement for saying "I don't know"—it only learns that abstention is better than hallucination through relative comparison, as the GRPO advantage computation surfaces the difference. This is a subtle mechanism: the model never sees an explicit "abstention is good" signal; it sees that when a group contains both abstentions and hallucinations, the abstentions have higher relative advantage. The behavior that emerges is calibrated—TruthRL abstains more than prompting but less than R-Tuning, and its abstentions are concentrated on questions the model genuinely cannot answer (Figure 3b)—despite the reward never explicitly teaching calibration.

The paper's ablation study (Table 3, Figure 4) demonstrates the practical consequences of this design space. The binary reward achieves the highest accuracy (60.3% on CRAG with retrieval for Llama3.1-8B-Instruct) but produces near-zero uncertainty and high hallucination. Augmenting the binary reward with knowledge-enhanced signals (explicitly rewarding abstention on OOK questions) partially restores uncertainty but at the cost of accuracy, without approaching the ternary reward's truthfulness. The ternary reward achieves the best balance, with substantially lower hallucination than binary (19.4% vs. 39.5%) and higher accuracy than knowledge-enhanced SFT baselines (56.6% vs. 48.4% for R-Tuning).

Even more telling is that augmenting the ternary reward with knowledge-enhanced signals (which was expected to help by providing more precise guidance) **reduces** truthfulness from 37.2 to 32.7 on CRAG (Table 3). This is a classic case of "more information hurts": the OOK labels, derived from the base model's pass@256 rate, are noisy and create a hard boundary in the reward landscape that the gradient-based optimizer struggles to navigate. The standard ternary reward, which never explicitly tells the model which questions are OOK, produces better calibration because the model learns to rely on its own internal uncertainty signals rather than external labels.

The reasoning-enhanced variants (Table 8) tell a similar story. Explicitly rewarding reasoning quality through additive or multiplicative schemes can improve the reasoning score (from 56.6 to 59.1 for the additive variant) but generally reduces truthfulness (from 37.2 to 36.1). The outcome-only ternary reward already implicitly improves reasoning quality (from 50.2 for prompting to 56.6), suggesting that the reasoning improvement is a byproduct of the model learning to be more careful about when it answers—good reasoning correlates with correct answers, and the ternary reward already incentivizes correctness.

Why is this finding conceptually significant? Because it establishes a **sufficiency result**: a simple ternary reward is enough to produce calibrated truthfulness in online RL, and additional complexity tends to hurt. This is not obvious in a field where reward engineering is increasingly elaborate (multi-objective RLHF, process reward models, learned verifiers). The paper's insight is that the **structure** of the reward—specifically, the ordinal relationship correct > abstain > hallucinate—matters far more than the precision of the reward values or the incorporation of auxiliary signals. This is a form of **reward minimalism** that has practical implications for practitioners: don't over-engineer the reward function; just ensure it encodes the right preference ordering, and the RL algorithm will do the rest.

---

### Innovation 4: The GRPO Advantage Mechanism as an Implicit Preference Learner

While the ternary reward is the *what* of TruthRL, the GRPO advantage computation is the *how* that makes it work—and the paper's analysis of this interaction constitutes a conceptual contribution about **why group-based advantage estimation is particularly well-suited to uncertainty-aware training**. This is not a claim about GRPO being new (it isn't; Shao et al., 2024 introduced it), but rather a claim about what happens when GRPO meets a ternary reward, and why this combination successfully avoids the pitfalls of both accuracy-driven training and supervised abstention training.

The key insight is that GRPO's group-relative advantage computation **automatically surfaces the correct preference ordering without requiring explicit pairwise comparisons or labeled preference data**. When the policy generates a group of responses that includes correct answers, abstentions, and hallucinations, the advantage for each response is determined by its reward relative to the group mean. Because correct > abstention > hallucination in the reward values, the advantages preserve this ordering regardless of the specific composition of the group. If the group consists entirely of hallucinations and one abstention, the abstention's advantage is positive. If the group consists of abstentions and one correct answer, the correct answer's advantage is positive. The policy learns the full ordering: prefer correctness when possible, prefer abstention when correctness is unavailable, avoid hallucination always.

This mechanism contrasts with alternative approaches that the paper compares against (Table 4). DPO, which requires pre-constructed preference pairs, performs poorly (average truthfulness −10.1) because the fixed preference dataset cannot capture the nuanced trade-offs between accuracy and abstention that vary per question. Iterative DPO improves somewhat (reaching 12.6 truthfulness at iteration 3) but is unstable (regressing to −2.0 at iteration 4) because it keeps relearning from the same type of constructed pairs. Online GRPO with ternary reward achieves 25.6 truthfulness because the group-based advantages are computed on-policy from the model's own current response distribution, creating a natural curriculum: as the policy improves, the group composition shifts (fewer hallucinations, more abstentions and correct answers), and the advantages adapt accordingly.

The paper's ablation comparing online GRPO to offline and semi-online alternatives (Table 4) is not framed as a major contribution, but it contains an important conceptual message: **for training uncertainty-awareness, online interaction with the reward signal may be necessary, not just beneficial**. Offline methods can teach a model to prefer correct over incorrect when correct examples exist in the data, but they cannot teach the model to calibrate its own uncertainty because the preference pairs are constructed by an external process (OOK labeling) rather than arising from the model's own output distribution. The gap between iterative DPO (12.6 at best) and online GRPO (25.6) is large enough to suggest a qualitative difference in what can be learned, not just an efficiency difference.

This insight connects to a broader theme in recent RL for LLMs work: the distinction between what can be learned from fixed data versus online interaction. Chu et al. (2025) argue that "SFT memorizes, RL generalizes"; TruthRL extends this to suggest that **online RL generalizes differently than offline RL**—specifically, it can learn calibration behaviors that require experiencing the consequences of one's own uncertainty in context, rather than being told about uncertainty through labeled examples.

## 5. Experimental Analysis

### Evaluation Methodology

- **Dataset.** The paper evaluates on four knowledge-intensive question-answering benchmarks: CRAG (Yang et al., 2024a), NaturalQuestions (NQ; Kwiatkowski et al., 2019), HotpotQA (Yang et al., 2018), and MuSiQue (Trivedi et al., 2022). Models are trained exclusively on CRAG and evaluated across all four datasets to test generalization of truthfulness behavior. For the retrieval setup, CRAG uses up to 50 web pages per question as retrieval documents; NQ, HotpotQA, and MuSiQue use the 2018 Wikipedia dump with the E5 retriever (Wang et al., 2024b), following Search-R1 (Jin et al., 2025). The paper does not explicitly report dataset sizes or train/validation/test splits; these are inherited from the original benchmark releases.

- **Base model(s).** Two backbone models are used: Llama3.1-8B-Instruct (Dubey et al., 2024) and Qwen2.5-7B-Instruct (Qwen et al., 2025), with additional scalability experiments on Llama3.2-3B-Instruct, Qwen2.5-3B-Instruct, and Qwen2.5-32B-Instruct (Table 7). The models are instruction-tuned checkpoints, selected as representative open-weight models at varying scales. The paper argues these models are "representative of the capabilities of many contemporary LLMs" and sit in a regime where non-trivial accuracy coexists with significant hallucination—leaving room for truthfulness-oriented training to make a measurable difference.

- **Metrics.** The primary metric is **truthfulness score**, defined as `Truthfulness = w1 · Acc + w2 · Unc − w3 · Hall`, with weights set to `w1 = 1, w2 = 0, w3 = 1` for evaluation (Section 4.1). This simplifies to **accuracy minus hallucination rate**, producing a score in the range [−100, 100] when expressed as percentages. Hallucination rate (Hall) is the fraction of responses that are factually incorrect; uncertainty rate (Unc) is the fraction where the model abstains (e.g., outputs "I don't know"). Accuracy (Acc) is reported as an auxiliary metric. Correctness is determined by an LLM judge (Llama3.3-70B-Instruct) comparing predictions against reference answers, following a detailed grading prompt (Appendix B, Table 11) that handles semantic equivalence, numeric tolerance, self-contradiction, and set-valued answers.

- **Baselines.** Five baselines are compared: **(1) Prompting** — the base model with no fine-tuning, using the inference prompt that instructs the model to enclose answers in `\boxed{}` and say "I don't know" if uncertain. **(2) Vanilla SFT** — standard supervised fine-tuning on the training dataset with ground-truth answers, without any uncertainty modeling. **(3) RFT (Rejection Sampling Fine-Tuning)** (Yuan et al., 2023) — the base model generates multiple reasoning traces per question; for OOK questions the trace concluding with "I don't know" is selected as the target, while for non-OOK questions the correct trace is selected. **(4) R-Tuning** (Zhang et al., 2024) — OOK questions have their ground-truth answer replaced with "I don't know" and the model is fine-tuned on this modified dataset. **(5) TruthRL (Binary)** — GRPO training with binary reward (+1 correct, −1 otherwise), which recovers the vanilla RL baseline. TruthRL with ternary reward is the proposed method.

- **Generation budget / compute accounting.** The paper does not use a fixed "generation budget" as a primary axis of comparison. Instead, all methods are trained to convergence under their respective objectives (SFT for 1 epoch, RL for a fixed number of training steps) and evaluated with greedy decoding (temperature = 0). The RL methods use a batch size of 64 questions per update with a constant learning rate of 1e-6. The paper does not report total training FLOPs, wall-clock time, or inference cost comparisons between methods. The only compute-normalized comparison is the online RL training itself, where all RL variants (binary, ternary, knowledge-enhanced) receive comparable training compute.

- **Cross-validation / statistical protocol.** No cross-validation or statistical significance testing is reported. The paper does not mention multiple random seeds, confidence intervals, or standard deviations for any results. Results are reported as point estimates from single training runs, evaluated on the full test sets of each benchmark. For the iterative DPO experiments (Table 4), each iteration builds on the previous checkpoint, creating a sequential dependence between data points. This absence of statistical rigor is a notable limitation: with test set sizes inherited from the benchmarks (likely hundreds to low thousands of questions), the stability of the reported improvements—particularly for per-difficulty breakdowns (Figure 3, Table 2)—cannot be assessed from the reported data.

---

### Main Quantitative Results

#### Aggregate Comparison Across Benchmarks (Table 1)

The central result table (Table 1) compares all methods across four benchmarks, two backbone models, and two retrieval settings, reporting truthfulness score (T), hallucination rate (H), and accuracy (A). The headline finding: **TruthRL achieves the highest truthfulness and lowest hallucination rates in nearly every configuration, with particularly strong results under retrieval.**

**Without retrieval, Llama3.1-8B-Instruct backbone.** TruthRL achieves an average truthfulness of 10.5 across the four benchmarks, compared to −20.9 for prompting, −50.3 for vanilla SFT, −25.1 for RFT, −13.6 for R-Tuning, and −26.7 for TruthRL (Binary). The hallucination rate for TruthRL is 20.5%, compared to 53.1% (prompting), 75.2% (SFT), 56.8% (RFT), 33.5% (R-Tuning), and 63.3% (TruthRL Binary). Accuracy for TruthRL (31.0%) is competitive with prompting (32.1%) but lower than TruthRL Binary (36.7%), reflecting the accuracy-abstention tradeoff: TruthRL sacrifices some correct answers to avoid hallucinations, while TruthRL Binary maximizes accuracy at the cost of hallucinating on most of its remaining outputs.

The per-benchmark breakdown reveals substantial variation. On CRAG, TruthRL achieves 22.4 truthfulness (38.7% accuracy, 16.3% hallucination)—dramatically better than prompting's −4.4 (40.1% accuracy, 44.5% hallucination). On MuSiQue, TruthRL's truthfulness is −7.7 (8.2% accuracy, 16.0% hallucination), compared to prompting's −54.2 (10.5% accuracy, 64.7% hallucination). The MuSiQue result is particularly instructive: TruthRL achieves slightly lower accuracy than prompting (8.2% vs. 10.5%) but far lower hallucination (16.0% vs. 64.7%), demonstrating that the ternary reward teaches the model to abstain on extremely difficult multi-hop questions rather than fabricate answers.

**Without retrieval, Qwen2.5-7B-Instruct backbone.** The pattern is consistent. TruthRL achieves average truthfulness of 5.7 (18.6% accuracy, 12.9% hallucination) versus prompting's −38.7 (22.7% accuracy, 61.4% hallucination). R-Tuning achieves the lowest hallucination (11.3%) but at the cost of collapsed accuracy (9.8%), resulting in negative average truthfulness (−1.5). TruthRL Binary achieves the highest accuracy (28.6%) but also the second-highest hallucination (70.6%), illustrating the binary reward's systematic bias toward answering at all costs.

**With retrieval, Llama3.1-8B-Instruct backbone.** This is where TruthRL shines most. Average truthfulness rises to 25.6 (44.4% accuracy, 18.8% hallucination), compared to prompting's −16.4 (37.7% accuracy, 54.1% hallucination) and TruthRL Binary's 4.5 (52.2% accuracy, 47.7% hallucination). On CRAG specifically, TruthRL achieves 37.2 truthfulness (56.6% accuracy, 19.4% hallucination)—a 31.9-point improvement over prompting (5.3 truthfulness, 48.8% accuracy, 43.5% hallucination). On HotpotQA, TruthRL achieves 37.4 truthfulness (52.3% accuracy, 14.9% hallucination), compared to prompting's −4.4 (44.6% accuracy, 49.0% hallucination). The retrieval setting amplifies TruthRL's advantage because external knowledge increases the proportion of questions the model can answer correctly, and the ternary reward ensures that when the retrieved information is insufficient or conflicting, the model abstains rather than hallucinating.

R-Tuning with retrieval shows mixed results: on CRAG it achieves 15.2 truthfulness (48.4% accuracy, 33.1% hallucination)—better than prompting but far behind TruthRL. On MuSiQue it achieves −53.9 truthfulness (14.4% accuracy, 68.3% hallucination), actually worse than prompting (−60.5 truthfulness, 12.5% accuracy, 73.0% hallucination). This per-benchmark inconsistency underscores R-Tuning's brittleness: its abstention behavior is tied to OOK labels derived from the training distribution and does not transfer reliably to out-of-distribution questions.

**With retrieval, Qwen2.5-7B-Instruct backbone.** TruthRL achieves average truthfulness of 23.1 (37.6% accuracy, 14.6% hallucination) versus prompting's −7.9 (38.6% accuracy, 46.5% hallucination). TruthRL Binary achieves higher accuracy (47.9%) but negative average truthfulness (−2.3) due to a 50.1% hallucination rate. RFT achieves 11.0 truthfulness (41.1% accuracy, 30.2% hallucination)—competitive with prompting on accuracy but with substantially lower hallucination, though still well behind TruthRL.

**The binary reward accuracy-hallucination tradeoff.** Across all configurations, TruthRL Binary consistently achieves the highest or near-highest accuracy—but also high hallucination and near-zero uncertainty. This is the empirical signature of the incentive problem diagnosed in Sections 1 and 2: a binary reward that treats "not correct" as uniformly bad will produce models that always generate an answer, maximizing expected accuracy at the cost of confident hallucination when the model is uncertain. TruthRL's ternary reward resolves this by making abstention preferable to hallucination in the advantage computation, even though abstention receives no positive reward.

#### Knowledge Boundary Recognition Analysis (Figure 3, Table 2)

Figure 3 decomposes model behavior on the CRAG benchmark under retrieval (Llama3.1-8B-Instruct backbone) into three categories: accuracy, hallucination, and uncertainty. Figure 3a shows the full test set; Figure 3b shows a "challenging subset" where almost no method produces correct answers.

**Full test set (Figure 3a).** TruthRL achieves 56.6% accuracy, 19.4% hallucination, and 24.0% uncertainty. This is the only configuration where all three categories have substantial non-zero values—the model correctly answers a majority of questions, abstains on a sizable minority, and hallucinates relatively rarely. In contrast:
- SFT achieves 50.7% accuracy, 49.3% hallucination, 0.0% uncertainty—the model answers everything, getting roughly half right and half wrong.
- TruthRL Binary achieves 60.3% accuracy, 39.5% hallucination, 0.2% uncertainty—highest accuracy but answering nearly everything, with the hallucination rate only slightly below 50%.
- R-Tuning achieves 48.4% accuracy, 33.1% hallucination, 18.5% uncertainty—substantial uncertainty but also the lowest accuracy among non-prompting methods, reflecting over-conservatism.

**Difficult subset (Figure 3b).** On questions where almost no method produces correct answers, the contrast is stark:
- TruthRL achieves 0.0% accuracy (as expected), 15.5% hallucination, and 84.5% uncertainty—the model correctly identifies its knowledge boundary on the vast majority of these questions.
- SFT achieves 0.9% accuracy, 99.1% hallucination, 0.0% uncertainty—near-universal confident hallucination.
- TruthRL Binary achieves 0.0% accuracy, 100.0% hallucination, 0.0% uncertainty—the binary reward model fabricates answers on literally every single difficult question.
- R-Tuning achieves 0.0% accuracy, 29.3% hallucination, 70.7% uncertainty—better than accuracy-driven methods but still hallucinating on nearly a third of these questions, compared to TruthRL's 15.5%.

This figure is the paper's most direct evidence for the claim that TruthRL improves "the capability of LLMs to recognize their knowledge boundary" (Section 4.3 title). The 84.5% abstention rate on questions where the model cannot produce correct answers—compared to 0% for SFT and TruthRL Binary—demonstrates that the ternary reward has taught calibration that generalizes beyond the training distribution.

**Hallucination-baiting questions (Table 2).** The paper evaluates on comparison-type questions from CRAG (e.g., "Which is larger, A or B?"), which are "known to be prone to inducing hallucinations" (Section 4.3, citing Kang et al., 2025). TruthRL achieves 52.4 truthfulness (16.5% hallucination, 14.6% uncertainty), compared to:
- Prompting: 9.7 truthfulness, 39.8% hallucination, 10.7% uncertainty
- SFT: 3.0 truthfulness, 48.5% hallucination, 0.0% uncertainty
- RFT: 12.7 truthfulness, 38.8% hallucination, 9.7% uncertainty
- R-Tuning: 6.8 truthfulness, 43.7% hallucination, 5.8% uncertainty

The critical finding is that R-Tuning—which performs reasonably well on aggregate metrics (Table 1)—shows a 43.7% hallucination rate on these baiting questions, nearly triple TruthRL's 16.5%. This demonstrates that R-Tuning's abstention behavior is cued by surface-level features that correlate with OOK questions in the training data, rather than by a genuine assessment of whether it knows the correct answer. When the question format changes to a multiple-choice style, the OOK signal is no longer reliable, and R-Tuning reverts to hallucination. TruthRL, which learns to calibrate uncertainty from its own output distribution through RL, transfers this calibration to the comparison format.

#### Reward Design Ablation (Table 3, Figure 4)

Table 3 ablates the reward design, comparing binary, ternary, and knowledge-enhanced variants of each, with Llama3.1-8B-Instruct under retrieval. The average truthfulness and hallucination across the four benchmarks:
- Binary reward: 4.5 truthfulness, 47.7% hallucination
- Binary + knowledge-enhanced: 11.7 truthfulness, 38.3% hallucination
- Ternary reward (TruthRL): 25.6 truthfulness, 18.8% hallucination
- Ternary + knowledge-enhanced: 23.2 truthfulness, 18.9% hallucination

The key finding: **the ternary reward substantially outperforms the binary reward, and adding knowledge-enhanced signals to the ternary reward slightly reduces truthfulness.** The binary→ternary jump is large (4.5→25.6 average truthfulness, 47.7%→18.8% hallucination), confirming that the reward structure itself—not training duration, model size, or other factors—is the primary driver of truthfulness improvement. The ternary→ternary+knowledge drop (25.6→23.2) is a notable negative result: providing explicit OOK labels as additional reward signals degrades performance, likely because the labels are noisy (derived from finite sampling) and create a hard boundary in the reward landscape that interferes with the smooth gradient signal from the group-based advantage.

Figure 4 tracks the learning dynamics of hallucination rate (4a), uncertainty rate (4b), and accuracy (4c) over training steps for the three reward variants. The binary reward drives hallucination down from roughly 45% to roughly 39%—a marginal improvement—while uncertainty stays near zero and accuracy rises to roughly 60%. The ternary reward simultaneously reduces hallucination (from roughly 38% to 19%), increases uncertainty (from roughly 0% to 24%), and maintains or slightly increases accuracy (roughly 55–57%). The enhanced reward (ternary + knowledge) shows similar trajectories to the standard ternary but with more variance and slightly worse final hallucination and accuracy. The dynamic trends confirm that the ternary reward produces a qualitatively different learning trajectory than the binary reward—one where uncertainty emerges naturally rather than being suppressed.

#### Online RL vs. Offline and Semi-Online Alternatives (Table 4)

Table 4 compares online GRPO (TruthRL) against offline DPO and iterative DPO (1–4 iterations), using Llama3.1-8B-Instruct under retrieval:
- DPO (offline): −10.1 average truthfulness, 51.1% hallucination
- Iterative DPO, Iter 1: −4.4 truthfulness, 48.7% hallucination
- Iterative DPO, Iter 2: −0.1 truthfulness, 45.7% hallucination
- Iterative DPO, Iter 3: 12.6 truthfulness, 31.7% hallucination
- Iterative DPO, Iter 4: −2.0 truthfulness, 42.8% hallucination
- TruthRL (online GRPO): 25.6 truthfulness, 18.8% hallucination

**Iterative DPO shows an inverted-U pattern.** Performance improves through Iter 3 (truthfulness rising from −10.1 to 12.6), then regresses sharply at Iter 4 (back to −2.0). This pattern—improvement followed by collapse—is characteristic of distribution shift in offline RL: as the policy moves away from the data-generating distribution through repeated fine-tuning, the fixed preference pairs become increasingly off-policy, and the DPO objective provides misleading gradients. The paper notes that "repeated offline fine-tuning cannot effectively balance exploration and exploitation" (Section 4.4).

**Online GRPO substantially outperforms even the best iterative DPO checkpoint** (25.6 vs. 12.6 truthfulness), demonstrating that on-policy interaction with the reward signal provides benefits that cannot be recovered by repeatedly refreshing preference data from a fixed policy. This gap is not just a matter of efficiency—it suggests that certain calibration behaviors require the model to experience the consequences of its own uncertainty during training, which offline preference pairs cannot simulate.

Per-benchmark results show that iterative DPO improves most on the training distribution (CRAG: −10.1 offline → 28.0 at Iter 3) but the improvement is less consistent on transfer benchmarks (HotpotQA: −50.5 → −19.0 at Iter 3; MuSiQue: regression from −50.5 to −19.0 at Iter 3, then back to −39.5 at Iter 4). TruthRL outperforms Iter 3 DPO on every benchmark, with the largest gaps on HotpotQA (37.4 vs. 26.5) and MuSiQue (−0.9 vs. −19.0).

#### Scalability Across Model Sizes (Table 7)

Table 7 reports TruthRL performance on CRAG with retrieval across five backbone models spanning 3B to 32B parameters. Paired prompting-vs-TruthRL comparisons:

| Model | Prompting T | Prompting H | TruthRL T | TruthRL H |
|---|---|---|---|---|
| Llama3.2-3B-Inst | 1.9 | 45.1 | 27.4 | 21.5 |
| Qwen2.5-3B-Inst | −0.3 | 45.4 | 21.9 | 16.2 |
| Qwen2.5-7B-Inst | 10.6 | 38.4 | 33.1 | 17.3 |
| Llama3.1-8B-Inst | 5.3 | 43.5 | 37.2 | 19.4 |
| Qwen2.5-32B-Inst | 29.1 | 27.1 | 40.0 | 18.2 |

**Truthfulness improvement is larger for smaller models.** The 3B models gain roughly 22–25 points in truthfulness, while the 32B model gains only 10.9 points. This diminishing relative gain at larger scales is expected: larger models already have lower hallucination rates (27.1% for 32B prompting vs. 45.1% for 3B prompting), so there is less room for TruthRL to reduce hallucination further. However, even the 32B model benefits: hallucination drops from 27.1% to 18.2% while accuracy necessarily increases (since truthfulness = accuracy − hallucination, and truthfulness rose from 29.1 to 40.0—this implies accuracy rose from roughly 56.2% to roughly 58.2%).

**The consistency of improvement across architectures and scales** (Llama and Qwen, 3B to 32B) provides the paper's strongest evidence that TruthRL is not model-specific or scale-dependent. The ternary reward mechanism operates on the output distribution regardless of the underlying model's capacity, and the benefits generalize across model families.

#### LLM Judge Robustness (Table 6)

Table 6 evaluates all methods on CRAG under three different judge models: Llama3.3-70B-Instruct (the training judge), Qwen2.5-72B-Instruct, and Gemma3-27B-Instruct (both unseen during training). The average across judges:
- Prompting: 4.6 truthfulness, 43.9% hallucination
- SFT: 3.3 truthfulness, 48.4% hallucination
- RFT: −3.9 truthfulness, 48.9% hallucination
- R-Tuning: 16.0 truthfulness, 32.7% hallucination
- TruthRL: 37.5 truthfulness, 19.3% hallucination

**Relative rankings are preserved across judges.** TruthRL achieves the highest truthfulness and lowest hallucination under all three evaluators, with a maximum spread of 4.1 points in truthfulness between the most favorable (Gemma3-27B: 39.7) and least favorable (Qwen2.5-72B: 35.6). This rules out the concern that TruthRL is overfitting to the idiosyncrasies of the training judge. The consistency across judges with different architectures, scales, and training procedures suggests the learned truthfulness behavior reflects genuine output quality differences that are detectable by any reasonable evaluator, not just reward hacking against a specific judge.

#### Rule-Based Verifier Failure (Table 5)

Training TruthRL with a rule-based verifier (exact string matching) instead of an LLM judge produces truthfulness of −3.6 and hallucination of 3.6%—the model collapses into near-universal abstention. This occurs because "the predicted answer rarely matches the reference answer in exact string form, causing rule-based verifiers to misclassify many correct responses" (Section 4.5). The model learns that producing any answer is likely to be scored as incorrect (−1), so it converges on always abstaining (0 reward), which is the only output that reliably avoids negative reward.

This negative result demonstrates that **a high-quality verifier is a prerequisite for the ternary reward approach**, not an optional enhancement. If the verifier cannot accurately distinguish correct from incorrect answers, the signal that correctness > abstention is corrupted—many genuine correct answers are penalized—and the model's rational response is to abstain on everything. The paper does not explore intermediate verifier quality levels, so it remains unclear how robust TruthRL is to verifier noise (e.g., what accuracy does the verifier need for TruthRL to outperform baselines?).

#### Reasoning-Enhanced Reward Experiments (Table 8)

Table 8 reports results from incorporating reasoning quality rewards on top of the outcome-based ternary reward, evaluated on CRAG with Llama3.1-8B-Instruct under retrieval. The base model (prompting) achieves 5.3 truthfulness, 43.5% hallucination, and a reasoning score of 50.2. TruthRL with outcome-only ternary reward achieves 37.2 truthfulness, 19.4% hallucination, and 56.6 reasoning score. The three reasoning-enhanced variants:
- Multiplicative: 37.0 truthfulness, 19.4% hallucination, 54.7 reasoning score
- Additive: 36.1 truthfulness, 19.1% hallucination, 59.1 reasoning score
- Conditional: 35.6 truthfulness, 19.3% hallucination, 55.1 reasoning score

**The outcome-only ternary reward already improves reasoning quality** (from 50.2 to 56.6) without explicit reasoning supervision. This is an emergent benefit: when the model learns to be more careful about when it answers, the reasoning that precedes correct answers tends to be higher quality. The additive reasoning reward further improves reasoning score to 59.1 but reduces truthfulness to 36.1, indicating a tradeoff between optimizing for reasoning quality and optimizing for outcome truthfulness. The multiplicative and conditional variants reduce both truthfulness and reasoning score compared to the outcome-only baseline.

The paper interprets these mixed results as evidence that "explicitly optimizing reasoning quality requires non-trivial design to balance multiple objectives" and that "heuristic designs like additive reasoning rewards can boost reasoning scores but may compromise the outcome." This is a measured negative result: the simple ternary outcome reward turns out to be hard to beat, and attempts to incorporate more sophisticated reward components tend to create unintended tradeoffs that degrade the primary metric.

---

### Ablation Studies and Robustness Checks

- **Binary vs. ternary reward (Table 3, Figure 4):** The binary reward achieves the highest accuracy (60.3% on CRAG) but produces near-zero uncertainty and 39.5% hallucination; the ternary reward achieves slightly lower accuracy (56.6%) but dramatically lower hallucination (19.4%) and 24.0% uncertainty, resulting in 16.4 points higher truthfulness (37.2 vs. 20.8). The learning curves (Figure 4) show qualitatively different dynamics: binary reward suppresses uncertainty throughout training, while ternary reward allows uncertainty to emerge and stabilize over time. This is the central ablation confirming that reward structure, not training algorithm or model architecture, drives the truthfulness improvement.

- **Knowledge-enhanced reward vs. standard ternary (Table 3, Figure 4):** Adding explicit OOK-based reward signals reduces average truthfulness from 25.6 to 23.2, despite slightly lower hallucination (18.9% vs. 18.8%). The per-benchmark breakdown shows the knowledge-enhanced variant underperforms on CRAG (32.7 vs. 37.2) and NQ (27.2 vs. 28.8) but matches or slightly exceeds on HotpotQA (35.1 vs. 37.4) and MuSiQue (−2.3 vs. −0.9). This negative result suggests that OOK labels—derived from the base model's pass@256 rate—introduce noise and hard decision boundaries that disrupt the smoother gradient signal from the standard ternary advantage computation.

- **Online RL vs. offline/semi-online (Table 4):** Offline DPO achieves −10.1 average truthfulness (worse than prompting). Iterative DPO improves to 12.6 at Iter 3 then regresses to −2.0 at Iter 4, demonstrating instability characteristic of off-policy training. Online GRPO achieves 25.6, nearly double the best iterative DPO checkpoint. This ablation establishes that on-policy interaction with the reward signal is necessary for learning calibrated truthfulness behaviors, not merely beneficial.

- **Model scale (Table 7):** TruthRL improves truthfulness across all tested scales (3B to 32B), with larger relative gains for smaller models (22–25 points for 3B models vs. 10.9 points for 32B). The hallucination reduction is consistent (16–27 percentage points) across scales. This robustness check confirms that the ternary reward mechanism does not depend on model capacity thresholds and benefits both weak and strong models.

- **LLM judge consistency (Table 6):** The relative ranking of methods (TruthRL > R-Tuning > prompting ≈ SFT > RFT) is preserved across three different judge models with varying architectures and scales. TruthRL's absolute truthfulness varies from 35.6 to 39.7 across judges, but remains the highest in all cases. This rules out judge-specific overfitting as an explanation for TruthRL's gains.

- **Verifier quality (Table 5):** Replacing the LLM judge with rule-based string matching causes catastrophic collapse: truthfulness drops to −3.6 with 3.6% hallucination as the model learns to abstain on nearly everything. This negative result establishes that a semantic verifier capable of recognizing correct answers despite formatting variations is a hard requirement for TruthRL—the ternary reward cannot function if the correctness signal is unreliable.

- **Reasoning reward integration (Table 8):** Adding explicit reasoning quality signals to the ternary outcome reward does not improve truthfulness. The additive variant improves reasoning score (59.1 vs. 56.6) but reduces truthfulness (36.1 vs. 37.2). The multiplicative and conditional variants reduce both. This ablation suggests that the outcome-based ternary reward already captures most of the available reasoning quality improvement, and that multi-objective reward design for this setting introduces tradeoffs that are difficult to tune.

- **Retrieval vs. non-retrieval (Table 1):** Retrieval improves all methods but the relative gains are larger for TruthRL. For Llama3.1-8B-Instruct, TruthRL's truthfulness improves from 10.5 (non-retrieval) to 25.6 (retrieval), while prompting improves from −20.9 to −16.4. This interaction effect suggests that TruthRL is particularly effective at leveraging external knowledge—the model learns when to trust retrieved information versus when to abstain—while accuracy-driven methods continue to hallucinate even with access to supporting documents.

- **Cross-benchmark generalization (Table 1):** TruthRL is trained exclusively on CRAG but improves truthfulness on all four benchmarks, with the largest absolute improvements on CRAG (37.2 vs. 5.3) and the smallest on MuSiQue (−0.9 vs. −60.5). The generalization to HotpotQA and NQ is substantial (37.4 and 28.8 vs. −4.4 and −5.8 for prompting), indicating that the learned truthfulness behavior is not specific to the CRAG question distribution. The limited improvement on MuSiQue reflects the fundamental difficulty of multi-hop reasoning questions that require synthesizing information across multiple documents—even with retrieval, the base model's accuracy is low (12.5%), and TruthRL's primary mechanism is to abstain rather than hallucinate, which reduces hallucination dramatically (15.9% vs. 73.0%) but cannot create correct answers where the base model lacks the underlying reasoning capability.

---

### Critical Assessment

**Claim 1: TruthRL reduces hallucinations and improves truthfulness compared to vanilla RL and SFT.** This claim is well-supported by Table 1, which shows consistent improvements across 2 backbone models, 4 benchmarks, and 2 retrieval settings—16 independent comparisons in total. TruthRL achieves the lowest hallucination in 30 of 32 metric-by-setting-by-backbone cells (all except two R-Tuning cells in the non-retrieval Qwen setting, where R-Tuning's hallucination is lower but its accuracy is catastrophically low). The magnitude is substantial: 28.9% reduction in hallucination and 21.1% improvement in truthfulness compared to vanilla RL, as quoted in the abstract. However, the paper does not clarify whether these percentages are relative (hallucination dropped by 28.9% of its original value) or absolute (hallucination dropped by 28.9 percentage points). Calculating from Table 1 average values for Llama3.1-8B-Instruct without retrieval: hallucination for binary RL is 63.3%, for TruthRL is 20.5%—an absolute reduction of 42.8 percentage points, or a 67.6% relative reduction. With retrieval: 47.7% vs. 18.8%—an absolute reduction of 28.9 percentage points, or a 60.6% relative reduction. The 28.9% figure matches the absolute reduction in the retrieval setting, suggesting the abstract's "up to 28.9%" refers to absolute percentage point reduction. More precise language would avoid this ambiguity.

**Claim 2: TruthRL teaches models to recognize their knowledge boundaries rather than being overly conservative.** This claim is supported by Figure 3b and the combination of Table 1 metrics. Figure 3b shows that on questions where the model cannot produce correct answers, TruthRL abstains 84.5% of the time while SFT and binary RL hallucinate 99–100% of the time. This demonstrates that TruthRL's abstention is concentrated on questions the model genuinely cannot answer, not spread uniformly. Table 1 shows that TruthRL maintains competitive accuracy while achieving much lower hallucination—if the model were being "overly conservative," its accuracy would drop substantially as it abstained on questions it could answer, which does not occur. However, the evidence for "knowledge boundary recognition" is behavioral (the model abstains on hard questions and answers easy ones correctly), not mechanistic—there is no analysis of the model's internal representations or confidence calibration to demonstrate that it "knows what it doesn't know" in a causal sense. The behavioral evidence is strong, but the claim of "recognizing knowledge boundaries" implies a cognitive capability that the experiments do not directly probe.

**Claim 3: A simple ternary reward outperforms more complex reward designs.** Supported by Tables 3 and 8. The ternary reward achieves higher truthfulness than the knowledge-enhanced variant (25.6 vs. 23.2 average) and all reasoning-enhanced variants. The negative result on knowledge-enhanced rewards is particularly informative: providing additional information (OOK labels) that one would expect to help actually hurts. Two caveats apply. First, the "more complex" designs explored are still relatively simple—the paper does not test learned reward models, process-based rewards, or multi-objective RL formulations with tunable weights, which limits the strength of the "simple outperforms complex" claim. Second, the ternary reward itself has an implicit complexity: distinguishing abstention from hallucination requires the verifier (LLM judge) to correctly classify the "I don't know" output category, which is a non-trivial capability that the paper does not analyze in isolation.

**Weakness: No statistical significance or variance reporting.** The paper reports all results as point estimates without confidence intervals, standard deviations, or multiple random seeds. Given the test set sizes (likely 500–2000 questions depending on the benchmark), the observed differences between methods may or may not be statistically significant. The difficulty-bin analysis (Figure 3) splits an already-modest test set into subsets, and the "challenging subset" in Figure 3b is described qualitatively ("where almost no method provides correct answers") without specifying its size or how questions are selected. The iterative DPO results (Table 4) show clear instability (performance rises then falls), but without error bars it is impossible to determine whether this pattern is systematic or noise. This is a significant omission for a paper making quantitative claims about percentage-point improvements.

**Weakness: Training on CRAG, evaluating on CRAG.** CRAG is used as both the training dataset and one of the four evaluation benchmarks. The paper does not specify whether the CRAG test results are on a held-out split or on the training set itself. If the CRAG results are in-distribution, the generalization claims rest entirely on the three transfer benchmarks (NQ, HotpotQA, MuSiQue), where improvements are real but smaller in absolute terms. For Llama3.1-8B-Instruct with retrieval, the truthfulness improvement over prompting is 31.9 on CRAG but only 34.6 on HotpotQA, 34.6 on NQ, and 59.6 on MuSiQue (though the last starts from a very low baseline). The paper would benefit from a clear holdout specification and from reporting CRAG results separately from transfer benchmark averages to avoid in-distribution results inflating the aggregate.

**Weakness: The "difficult subset" is not defined programmatically.** Figure 3b evaluates on "difficult questions where almost no method can provide correct answers" but does not provide an operational definition—what accuracy threshold qualifies as "almost no method"? How many questions are in this subset? Is it the same subset for all methods, or defined per-method based on that method's accuracy? Without these details, the dramatic result (SFT hallucinates 99.1% of the time on these questions) cannot be replicated or properly interpreted. If the subset is defined as "questions where SFT achieves near-zero accuracy," then the finding that SFT hallucinates on them is close to tautological.

**Weakness: No exploration of the ternary reward's sensitivity to the abstention reward value.** The paper places abstention at exactly 0 and argues that this is crucial—but never tests alternative values. What happens if abstention is +0.5? −0.5? +0.1? The claim that "0 is the right value" is supported only by the failure mode of binary reward (−1 for abstention) and the over-conservatism of R-Tuning (+1 for abstention on OOK questions), but these are confounded with other design differences (binary vs. ternary, SFT vs. RL). An ablation testing ternary rewards with abstention at −0.5, 0, +0.5 under identical GRPO training would strengthen the central design claim considerably.

**Weakness: Single training dataset limits generality claims.** All training is on CRAG. While the transfer results to NQ, HotpotQA, and MuSiQue are positive, they are still all factoid QA benchmarks. The paper's claims about "high-stakes domains" (medicine, law) and the importance of truthfulness in those settings are not validated on any domain-specific datasets. A medical QA benchmark or a legal QA benchmark would provide much stronger evidence for the practical applicability of TruthRL to the high-stakes settings the paper motivates with.

**Missing experiment: What happens if you combine R-Tuning's OOK SFT with TruthRL's RL?** The paper treats R-Tuning and TruthRL as separate approaches (SFT vs. RL), but a natural combination would be to use R-Tuning as a warmup before TruthRL RL training, or to use R-Tuning's OOK labels as an auxiliary signal during RL. This is not explored, leaving unclear whether the approaches are complementary or redundant.

**Missing experiment: The effect of verifier accuracy on TruthRL performance.** Table 5 shows catastrophic failure with rule-based verification, but what about intermediate verifier quality? A synthetic experiment degrading the LLM judge's accuracy (e.g., by introducing controlled noise) would reveal how robust TruthRL is to verifier errors—a practically important question since perfect verification is unavailable in most real-world settings.

**Missing experiment: Comparison to calibration-based methods.** The paper compares against accuracy-driven methods and OOK-based SFT methods, but does not compare against post-hoc calibration techniques (e.g., using the model's token probabilities as a confidence score and thresholding). A simple baseline where the prompted model's output confidence determines whether to answer or abstain would contextualize TruthRL's gains relative to much simpler, training-free approaches to the same problem.

## 6. Limitations and Trade-offs

### The Difficulty Estimation Cost Is Unaccounted For and Prohibitively Expensive

The knowledge boundary probing procedure—which identifies out-of-knowledge (OOK) questions for the R-Tuning and RFT baselines and optionally informs the knowledge-enhanced reward variant—requires generating **256 responses per training question** and verifying each against the ground-truth answer to determine whether the base model can produce a correct answer. The paper acknowledges this in Section 3.1:

> "For each training question, we sample 256 responses, and the question is marked as OOK if none of the responses is correct."

**Consequence.** This is an enormous computational cost that the paper does not quantify or amortize into any reported efficiency metric. Assuming CRAG's training split is on the order of a few thousand questions (the paper does not state the exact size, but typical QA benchmarks have 1,000–10,000 training examples), the probing step consumes 256 × N generations just to construct labels—potentially exceeding the cost of the downstream RL training itself. In a deployment setting where a practitioner wants to apply TruthRL to a new dataset or domain, this probing step must be repeated from scratch, making the approach far less practical than the headline results suggest. The cost is particularly problematic for the knowledge-enhanced reward variant (Section 3.2), which requires OOK labels as input to the reward function—yet this variant is shown to **underperform** the standard ternary reward (Table 3: 23.2 vs. 25.6 average truthfulness). The probing step is therefore both expensive and, for the best-performing version of TruthRL, unnecessary.

**What evidence exists.** Table 3 implicitly demonstrates that the probing step is not needed for TruthRL's primary method: the standard ternary reward (which does not use OOK labels) outperforms the knowledge-enhanced variant (which does). However, the paper does not report the probing cost in FLOPs, wall-clock time, or GPU hours, and does not factor it into any efficiency comparison. The probing step is described as a data preprocessing step for baselines (Section 3.1) and its cost is not discussed in any limitation context.

**Mitigation status.** The probing cost is not a limitation of TruthRL per se (the standard ternary reward does not require it), but it is a limitation of the baseline construction and the broader experimental pipeline. A practitioner who wants to replicate the full set of baselines (R-Tuning, RFT) or experiment with the knowledge-enhanced reward incurs this cost. The paper does not suggest cheaper alternatives for OOK detection, such as using a smaller number of samples, employing confidence-based heuristics, or training a lightweight OOK classifier amortized across questions.

---

### The Method Is Validated on a Single Training Dataset and a Single Task Family

All training is performed on the **CRAG benchmark** (Yang et al., 2024a) only. Evaluation is conducted on CRAG plus three additional benchmarks (NaturalQuestions, HotpotQA, MuSiQue), but the model never sees training data from those benchmarks. The paper states in Section 4.1: "Models are trained on CRAG and evaluated across all four datasets."

**Consequence.** While the transfer results are positive (Table 1 shows TruthRL improves over prompting on all four benchmarks), all four benchmarks are fundamentally **factoid question answering** tasks—short-form, knowledge-intensive questions with verifiable reference answers. The paper's motivating examples (Section 1) invoke high-stakes domains like law and medicine, but no experiments are conducted on legal QA, medical QA, or any domain-specific benchmark where the consequences of hallucination are genuinely severe. The method's effectiveness on tasks with different output structures (e.g., long-form generation, multi-turn dialogue, summarization, code generation) is unknown. The ternary reward relies on an LLM judge that can reliably classify outputs as correct, uncertain, or hallucinated—this becomes substantially harder for open-ended generation where correctness is multi-dimensional, subjective, or context-dependent. Furthermore, CRAG is both the training dataset and one of the four evaluation datasets (Table 1 includes CRAG in the average). The paper does not clarify whether CRAG evaluation uses a held-out split, raising the possibility that in-distribution evaluation inflates the aggregate metrics relative to what a practitioner would see on a truly new domain.

**What evidence exists.** Table 1 shows the transfer results to NQ, HotpotQA, and MuSiQue. The improvements are real: for Llama3.1-8B-Instruct with retrieval, TruthRL improves truthfulness from −5.8 (prompting) to 28.8 on NQ and from −4.4 to 37.4 on HotpotQA. However, the improvement on MuSiQue is from −60.5 to −0.9—the model still achieves negative truthfulness, indicating it cannot produce net-correct answers on this multi-hop reasoning benchmark. The paper does not report performance on any non-QA task, any domain-specific benchmark, or any open-ended generation task. Section 1 provides a motivating example about ESTA visa requirements (a legal/immigration question), but this example is illustrative only and not drawn from a benchmark used in experiments.

**Mitigation status.** The paper does not claim generalization beyond knowledge-intensive QA, and the focus on factoid QA with verifiable answers is a reasonable scope for a method that relies on an outcome-based verifier. However, the gap between the motivating high-stakes scenarios and the experimental validation is not acknowledged. Future work on extending the ternary reward to tasks without clean correctness signals (e.g., using process-based or learned reward models for open-ended generation) is implicitly suggested by the paper's broader advocacy for truthfulness-driven training but is not explicitly outlined as a limitation to address.

---

### No Statistical Significance or Variance Reporting Undermines Quantitative Claims

The paper reports all results as **point estimates** from single training runs. No confidence intervals, standard deviations, or error bars appear anywhere in the paper—not in Table 1, not in Figure 3, not in any ablation. There is no mention of multiple random seeds or repeated trials.

**Consequence.** The test sets of the four benchmarks (CRAG, NQ, HotpotQA, MuSiQue) contain on the order of hundreds to low thousands of questions each. While the paper does not report exact test set sizes, typical sizes for these benchmarks are: CRAG ~1,400, NQ ~3,600, HotpotQA ~7,400, MuSiQue ~2,400 (test or validation splits). With these sample sizes, a difference of a few percentage points in accuracy or hallucination rate can easily fall within the margin of sampling error. For example, in Table 1 (Llama3.1-8B-Instruct with retrieval), the truthfulness difference between TruthRL and prompting on NQ is 28.8 − (−5.8) = 34.6 points—likely robust even without formal tests. But finer comparisons, such as TruthRL vs. R-Tuning on HotpotQA (37.4 vs. 1.7 truthfulness), involve a baseline with high variance (R-Tuning's performance fluctuates dramatically across benchmarks: 15.2 on CRAG, 2.1 on NQ, 1.7 on HotpotQA, −53.9 on MuSiQue in the Llama3.1 retrieval setting). Without variance estimates, it is impossible to determine whether R-Tuning's inconsistency reflects genuine brittleness or sampling noise from small per-benchmark test sets.

The iterative DPO results (Table 4) show a clear inverted-U pattern: truthfulness rises from −10.1 (offline) to 12.6 (Iter 3) then drops to −2.0 (Iter 4). The paper interprets the Iter 4 regression as evidence that "repeated offline fine-tuning cannot effectively balance exploration and exploitation" (Section 4.4). This is a strong claim based on a single trajectory. Without error bars or replication across seeds, the regression could be a random fluctuation rather than a systematic failure mode. The paper's analysis of difficulty-dependent behavior (Section 4.3, Figure 3) divides an already-modest test set into a "challenging subset" of unspecified size; variance on this subset could be large enough to affect the reported 84.5% vs. 70.7% abstention comparison between TruthRL and R-Tuning.

**What evidence exists.** Nowhere in the paper are confidence intervals, standard deviations, or significance tests reported. The word "seed" does not appear in the main text or appendices. The training details in Appendix A specify hyperparameters but do not mention replication. The absence of any variance reporting is the most significant methodological omission in the paper, because it prevents readers from assessing whether the reported improvements are reliable or within the noise floor of the evaluation.

**Mitigation status.** The paper does not acknowledge this limitation. The consistency of TruthRL's advantage across 16 independent comparisons (2 backbones × 4 benchmarks × 2 retrieval settings) provides informal evidence that the effect is real—the probability of TruthRL outperforming prompting on all 16 by chance alone is low. However, this informal robustness does not substitute for proper statistical reporting, particularly for the finer-grained claims about reward design variants (Table 3), iterative DPO trajectories (Table 4), and per-difficulty breakdowns (Figure 3).

---

### The Ternary Reward's Sensitivity to the Abstention Value Is Unexplored

The paper places abstention at exactly **0** in the ternary reward—correct is +1, hallucination is −1, abstention is 0. This value is asserted to be correct based on conceptual reasoning (Section 3.2: "The model learns that producing any answer is likely to be scored as incorrect, so it converges on always abstaining") and the comparison to binary reward (where abstention is effectively −1, leading to near-zero uncertainty). However, the paper **never tests any alternative abstention value**. What happens if abstention is +0.3? −0.3? +0.5? The claim that 0 is the optimal value is supported only by the observation that binary reward (−1 for abstention) suppresses uncertainty (Figure 2) and that R-Tuning (which explicitly rewards abstention via SFT on OOK questions) is overly conservative (Table 1). These comparisons confound the reward value with other design differences: binary vs. ternary structure, SFT vs. RL training algorithm.

**Consequence.** A practitioner implementing TruthRL on a new domain with a different base model, different verifier quality, or a different balance of easy-to-hard questions may find that the 0 value produces inappropriate calibration—too much abstention (if the verifier is noisy, causing correct answers to be misclassified and making abstention relatively more attractive) or too little (if the base model rarely generates uncertain outputs, limiting the group diversity needed for the advantage mechanism to surface the abstention preference). Without knowing the sensitivity of the method to this parameter, deployment requires guesswork or expensive hyperparameter sweeps.

The paper's own evidence contains a hint that the abstention value matters more than the paper acknowledges: the knowledge-enhanced ternary reward (which assigns +1 to abstention on OOK questions) achieves slightly lower hallucination than the standard ternary reward (18.9% vs. 18.8% average) but substantially lower truthfulness (23.2 vs. 25.6). This means a targeted increase in the abstention reward changes behavior in ways that harm the overall metric—suggesting that the 0 value sits in a sensitive region of the reward space.

**What evidence exists.** The paper's ablation of reward designs (Table 3) tests binary (abstention effectively −1), ternary (abstention 0), and knowledge-enhanced ternary (abstention +1 on OOK questions, 0 on non-OOK). This spans the range from "abstention is penalized" to "abstention is rewarded" but only at three discrete, confounded points. There is no sweep of abstention values in the ternary reward (e.g., −0.5, 0, +0.3, +0.5) holding the reward structure constant. The learning curves in Figure 4 show only the three confounded variants, not a parametric sweep.

**Mitigation status.** The paper does not acknowledge the unexplored sensitivity to the abstention value as a limitation. The strong claim that "a simple ternary reward scheme generally works better than the binary scheme and more complicated designs" (Section 4.4) implicitly asserts that the exact value of 0 is not critical—but this assertion is untested. A one-paragraph ablation in Appendix or a footnote acknowledging that the 0 value was chosen for conceptual reasons and not empirically optimized would clarify the status of this design choice.

---

### Hard Multi-Hop and Out-of-Distribution Questions Remain Unsolved

On **MuSiQue**, a multi-hop reasoning benchmark that requires synthesizing information across multiple documents, TruthRL achieves a truthfulness score of **−0.9** with retrieval using Llama3.1-8B-Instruct (Table 1)—meaning accuracy (15.0%) minus hallucination (15.9%) is still negative. On the non-retrieval setting, MuSiQue truthfulness drops to **−7.7** (8.2% accuracy, 16.0% hallucination). These are improvements over prompting (−60.5 with retrieval, −54.2 without), but they represent a floor: the model cannot produce net-positive truthfulness on this benchmark.

**Consequence.** TruthRL improves truthfulness primarily by converting hallucinations into abstentions, not by converting hallucinations into correct answers. On MuSiQue, the hallucination rate drops dramatically (from 73.0% to 15.9% with retrieval), but accuracy barely moves (12.5% → 15.0%). The model learns to say "I don't know" rather than fabricate multi-hop reasoning chains—which is the correct behavior given its limited multi-hop capability, but it means that TruthRL does not address the underlying **capability gap**. For applications where the task inherently requires complex multi-step reasoning (legal analysis, medical diagnosis, financial modeling), TruthRL will produce a truthful but largely unhelpful model that abstains on most queries. The truthfulness metric, which rewards abstention only instrumentally (by reducing hallucination), gives a score near zero for such a model—which is better than a negative score, but far from what a useful system requires.

The paper's own difficulty analysis (Figure 3) supports this interpretation. On the "challenging subset" of CRAG (Figure 3b), TruthRL achieves 0% accuracy, 15.5% hallucination, and 84.5% uncertainty. The model correctly identifies that it cannot answer these questions—which is the intended behavior—but the absolute performance on these questions remains at zero. The improvement over baselines is entirely in the hallucination→abstention conversion, not in solving previously unsolvable problems.

**What evidence exists.** Table 1 provides the MuSiQue results. The paper acknowledges this limitation implicitly in Section 4.5: "the predicted answer rarely matches the reference answer in exact string form," and more broadly in how the ternary reward is designed—it never rewards the model for answering correctly on questions it currently gets wrong unless the model spontaneously generates correct answers during RL training. There is no mechanism for the model to acquire new knowledge or reasoning capabilities through TruthRL; the RL process only reshapes how the model deploys its existing capabilities.

**Mitigation status.** The paper does not frame this as a limitation—it is arguably a feature of the approach (the model should not hallucinate on questions it cannot answer). However, the paper's motivating examples (Section 1: ESTA visa question, medical/law domains) imply that TruthRL helps in high-stakes settings where the model's knowledge is incomplete, which is exactly the MuSiQue regime. The gap between "the model truthfully admits it doesn't know" and "the model is useful" is not discussed. Combining TruthRL with methods that expand the model's knowledge boundary (better retrieval, continued pretraining, or reasoning-enhancing techniques) is left entirely to future work.

---

### The Method Depends Critically on a High-Quality LLM Verifier, and Verifier Failure Is Catastrophic

The ternary reward requires the verifier to accurately classify model outputs into three categories: correct (+1), uncertain/abstention (0), and incorrect/hallucination (−1). The paper uses **Llama3.3-70B-Instruct** as the verifier—a model substantially larger and more capable than the 7B–8B models being trained. This is a form of **supervision from a stronger model**, not a self-supervised or self-improvement setup.

**Consequence.** When the verifier is replaced with rule-based string matching, the method **collapses catastrophically**: truthfulness drops to −3.6 and the model learns to abstain on nearly everything (Table 5). The paper explains: "the predicted answer rarely matches the reference answer in exact string form, causing rule-based verifiers to misclassify many correct responses" (Section 4.5). This means TruthRL is only viable when a high-quality semantic verifier is available—and that verifier must be accurate enough that correct answers are reliably rewarded, or the model's rational strategy is universal abstention (since abstaining gives a guaranteed 0 reward while attempting an answer risks −1 from verifier error).

The paper does not explore **how good the verifier needs to be**. The Llama3.3-70B judge is very strong, but what if only a 7B judge is available? What if the judge's accuracy on the target domain is 90%? 80%? The binary choice between a 70B LLM judge and exact string matching leaves a vast unexplored territory of intermediate verifier quality where TruthRL's behavior is unknown. In many practical settings—particularly the high-stakes domains the paper motivates with—reference answers may not exist, and verifier quality may be substantially lower than what the paper assumes.

This dependence on a stronger verifier also raises a **scalability concern**: to train a truthful N-parameter model, the paper requires a >10× larger verifier model. If one wanted to apply TruthRL to a 70B model, would a verifier need to be even larger? The paper's scalability experiments (Table 7) show TruthRL working across model sizes from 3B to 32B, but all use the same 70B verifier. The limit of this approach—how large the trainee can be relative to the verifier—is unexplored.

**What evidence exists.** Table 5 provides the catastrophic failure case with rule-based verification. Table 6 demonstrates robustness across different LLM judges (Llama3.3-70B, Qwen2.5-72B, Gemma3-27B), showing that the method does not overfit to a specific judge model. However, all three judges are high-capacity models (27B–72B) evaluated in a regime where their agreement is likely high. The paper does not test degraded verifiers (e.g., a 7B judge, a judge with artificially introduced noise, or a judge trained on a different domain). Section 4.5 notes: "a high-quality verifier is as important as the reward design itself in reinforcement learning for truthfulness"—this is stated as a finding, but its implications as a limitation (the method cannot be applied in domains without access to a verifier at least as strong as the trainee) are not discussed.

**Mitigation status.** The paper acknowledges the verifier quality requirement in Section 4.5 but does not frame it as a limitation of the method's applicability. No experiments explore the verifier quality threshold, and no suggestions are offered for practitioners who lack access to a 70B-class verifier. The observation that TruthRL is robust across different high-capacity judges (Table 6) is encouraging but does not address the more fundamental dependency on having a judge that is substantially stronger than the model being trained.

## 7. Implications and Future Directions
- Field-level impact
  - TruthRL shifts post-training from accuracy-only to truthfulness-oriented objectives, showing that reward structure (ternary vs binary) fundamentally changes LLM behavior. It offers a practical recipe—simple reward, online GRPO, LLM judge—that improves reliability across datasets, models, and retrieval settings.

- Practical applications
  - High-stakes QA, customer support, medical/legal triage, enterprise search assistants, and any RAG system where calibrated abstention reduces risk. TruthRL’s abstain-over-guess behavior is particularly valuable for safety-critical deployments.

- Follow-up research
  - Better verifiers: More reliable, cheaper, perhaps hybrid (LLM + rules) judges to maintain semantic awareness without collapse (Table 5).
  - Multi-objective RL: Explicitly incorporate uncertainty into the evaluation objective (nonzero `w2`) to reflect real-world preferences for abstention in risky contexts.
  - Reasoning rewards: The initial explorations (Table 8) show outcome-only rewards already improve reasoning scores; principled multi-signal designs could further enhance faithful reasoning without hurting outcomes.
  - Data-aware retrieval: Couple TruthRL with retrieval re-ranking and evidence quality estimation to further suppress hallucinations from noisy context.
  - Calibration metrics: Beyond Acc/Hall/Unc, evaluate and optimize explicit calibration (e.g., expected calibration error) to align confidence with correctness (Figure 5 hints this is promising).

> Bottom line: A minimal change—treating abstention distinctly from error in an online RL setup—alters incentives so the model stops guessing when unsure, sharply reducing hallucinations and improving trustworthiness (Table 1, Figure 3).
