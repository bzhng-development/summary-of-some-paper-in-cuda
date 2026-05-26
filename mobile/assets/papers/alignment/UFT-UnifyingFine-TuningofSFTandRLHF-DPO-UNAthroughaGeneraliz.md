# UFT: Unifying Fine-Tuning of SFT and RLHF/DPO/UNA through a Generalized Implicit Reward Function

**ArXiv:** [2410.21438](https://arxiv.org/abs/2410.21438)

## 🎯 Pitch

This paper introduces Unified Fine-Tuning (UFT), a novel single-stage framework that merges supervised fine-tuning (SFT) and alignment approaches (such as RLHF, DPO, and UNA) using a shared, generalized implicit reward function. By eliminating the traditional multi-stage pipeline that often leads to degraded model capabilities ('alignment tax'), UFT significantly enhances both instruction-following and factual accuracy in large language models—streamlining alignment and enabling LLMs to retain and extend their acquired skills in a more robust, efficient, and practical way.

---

## 1. Executive Summary

This paper proposes **Unified Fine-Tuning (UFT)**, a framework that integrates SFT and alignment into a single training stage using the same objective and loss functions through a generalized implicit reward function adapted from UNA (converting SFT data into positive-reward alignment data, then training with a unified squared-error loss against explicit reward scores). Experiments on Mistral 7B and Qwen 32B using UltraChat (instruction-tuning) and HelpSteer2 (alignment) demonstrate that UFT outperforms SFT alone on downstream tasks and, when mixing both data types, surpasses sequential SFT+alignment pipelines while preventing performance degradation — most notably on ifeval (instruction-following) and truthful (factuality), where Mistral+UFT reaches 46.03% and 54.05% respectively versus sequential methods that fall below 27% and 50% on those same metrics. The framework establishes that merging instruction-tuning and alignment data under a single implicit reward objective preserves capabilities across both stages, though the benefit diminishes with larger model size and depends critically on the ratio of instruction-tuning to alignment data in the training mixture.

## 2. Context and Motivation

### The Core Problem: Sequential Fine-Tuning Causes Performance Degradation

The fundamental issue this paper tackles is a well-documented but incompletely solved pathology in the standard LLM post-training pipeline. The prevailing recipe across the industry — OpenAI's GPT family, Anthropic's Claude, Meta's Llama series — applies **Supervised Fine-Tuning (SFT)** followed by **alignment** (typically RLHF with PPO, DPO, or related variants) as separate, sequential stages. SFT teaches the pretrained model to follow instructions and generate useful responses; alignment teaches it to reject harmful requests and adhere to human preferences around helpfulness, honesty, and harmlessness.

The problem is that **these two stages pull the model in different directions**, and the second stage often erases or degrades capabilities acquired during the first. The paper refers to this broadly as "performance degradation" (Section 1), which the field also calls "alignment tax." Concretely, the authors observe that when they take a base Mistral 7B model, apply SFT, and then apply alignment (whether DPO, KTO, or UNA), the final aligned model performs **worse** on several benchmarks than the SFT-only model (Tables 1–3). This is not a subtle effect — across the 12 tasks measured on both the old and new HuggingFace Open LLM Leaderboards, the sequential methods collectively underperform SFT-only on the majority of tasks. The degradation is visible in both closed-form tasks (multiple-choice QA, math reasoning) and open-ended generation (MT-Bench, Alpaca-eval).

This degradation matters for several practical reasons the paper implies but does not exhaustively enumerate:

- **Deployment quality:** An aligned model that scores well on safety metrics but has lost instruction-following capability or factual accuracy is not practically useful — it is safe but unhelpful. The paper highlights this tension explicitly by tracking `ifeval` (instruction-following) and `truthful` (factuality/hallucination reduction) alongside traditional reasoning benchmarks.
- **Iterative development friction:** In practice, model developers tune SFT and alignment separately, often with different teams, data mixtures, and hyperparameter budgets. The sequential coupling means improvements to SFT do not cleanly transfer to the aligned model, and vice versa, creating a whack-a-mole dynamic where fixing one capability breaks another.
- **Theoretical incompleteness:** The fact that sequential training degrades previously acquired capabilities hints that the underlying optimization objectives — cross-entropy next-token prediction for SFT versus preference-based reward maximization for alignment — are not naturally compatible, and that naïvely applying them in sequence creates interference in the model's parameter space.

### Where Existing Approaches Fall Short

The paper situates itself within a landscape where prior work has attempted to address the SFT-alignment gap through several distinct strategies, each with significant limitations.

**RLHF/PPO (the incumbent).** The dominant alignment paradigm, introduced by Christiano et al. and scaled by InstructGPT/ChatGPT (Ouyang et al., 2022; Bai et al., 2022), trains an explicit reward model on human preference data and then optimizes the policy with Proximal Policy Optimization (PPO) using a KL penalty to prevent divergence from the reference (usually the SFT model). This approach has three well-known practical drawbacks that the paper recapitulates (Section 2.2):

> "The training of RLHF is memory-intensive because it requires maintaining both the explicit reward model, the value model and the policy model. Additionally, reinforcement learning is notorious for its instability."

These are not merely inconvenient — they impose real barriers. Training with PPO requires simultaneously hosting four models (policy, reference, reward/critic, value) in GPU memory, limiting the feasible model size given hardware constraints. The instability of RL training on language tasks (reward hacking, policy collapse, sensitivity to hyperparameters) makes results brittle and hard to reproduce.

But the deeper limitation for the problem this paper addresses is that **RLHF/PPO compounds the sequential degradation issue rather than solving it**. PPO's KL penalty is designed to prevent the policy from straying too far from the SFT model, but it is a soft constraint applied at the level of output distributions, not a structural integration of the two objectives. The SFT stage optimizes for one thing (response likelihood); the RL stage optimizes for another (reward maximization with a KL budget); and the model's capabilities can still shift in non-obvious ways during the transition.

**DPO (the simplification).** Direct Preference Optimization (Rafailov et al., 2023) addressed the practical complexity of RLHF by deriving an implicit mapping between the policy and the reward model, eliminating the need for explicit reward model training and online RL. DPO's insight — that the optimal policy under a KL-constrained reward maximization objective can be expressed directly in terms of the policy ratio with respect to a reference model — is elegant and forms the conceptual foundation on which UNA and UFT both build.

However, DPO has structural limitations that the paper identifies:

- **Restricted to pairwise data.** DPO's loss requires subtracting implicit rewards between preferred ($y_w$) and dispreferred ($y_l$) responses, which cancels the intractable partition function $Z(x)$. This means DPO can only use datasets of the form $(x, y_w, y_l)$ — preference pairs. It cannot natively ingest binary feedback ("thumbs up/down") or scalar scores, which are far cheaper to collect and more expressive.
- **No integration with SFT.** DPO implicitly assumes you already have an SFT-trained model serving as the reference ($\pi_{\text{ref}}$). It does not provide a mechanism for jointly optimizing instruction-following and preference satisfaction — it purely replaces the RL stage of the RLHF pipeline.

**KTO (the binary extension).** Kahneman-Tversky Optimization (Ethayarajh et al., 2023) extended DPO to handle binary feedback by estimating $Z(x)$ from multiple responses to the same prompt. This relaxes the pairwise constraint but introduces its own complication: the partition function estimation requires multiple samples per prompt, and the resulting objective is approximate. More fundamentally, KTO still treats SFT as a separate, prerequisite stage and does not unify it with alignment.

**ORPO (monolithic preference optimization).** Hong et al. (2024) proposed ORPO as a method that combines SFT and alignment into a single objective without a reference model. ORPO augments the standard SFT cross-entropy loss with an odds-ratio term that penalizes dispreferred responses relative to preferred ones. The paper acknowledges this as prior art attempting the same unification goal (Section 5) but identifies two critical weaknesses:

> "ORPO's reliance on pairwise datasets and its deteriorating performance compared to other SFT and alignment methods pose challenges."

The first issue — reliance on pairwise data — limits ORPO's applicability in settings where only scores or binary labels are available. The second — deteriorating performance — is a direct quality concern: a unified method that underperforms sequential baselines on standard benchmarks defeats the purpose.

**PAFT (parallel adapter training).** Pentyala et al. (2024) proposed running SFT and alignment in parallel via separate LoRA adapters and merging them post-hoc using sparsity-based fusion. The paper notes this is "inefficient" because it requires training two separate adapters, introduces a separate merging step with its own hyperparameters, and does not truly unify the objectives — it aggregates models trained independently under different criteria, which is a fundamentally different approach from joint optimization.

**UNA (the direct precursor).** The paper's intellectual lineage runs directly through UNA (Wang et al., 2024), which the current authors also contributed to. UNA demonstrated that by using a generalized implicit reward function — $r_\theta(x, y) = \beta \log(\pi_\theta(y|x) / \pi_{\text{ref}}(y|x))$ — without the $Z(x)$ term that DPO requires, one could handle pairwise, binary, *and* score-based feedback under a single objective by minimizing the distance between this implicit reward and the explicit reward signal $r_\phi(x, y)$. This eliminated the partition function problem and unified previously disparate alignment methods (RLHF, DPO, KTO) under one framework.

UNA's key theoretical move — dropping the $Z(x)$ term and directly regressing the implicit reward against explicit reward scores — is what makes UFT possible. But UNA itself did not address the SFT-alignment unification problem. It was purely an alignment method, assuming the model had already undergone SFT. UFT's contribution is to realize that **UNA's framework can absorb SFT by simply treating instruction-tuning data as alignment data with a maximum reward score of 1**. This insight — that SFT responses are, in effect, "perfectly preferred" responses — is simple in retrospect but non-obvious enough that prior work did not make the connection.

### How This Paper Positions Itself

The paper's positioning is best understood as **extending UNA from an alignment-only framework to a unified post-training framework**. The authors make this ambition explicit (Section 2.4) with the analogy:

> "Drawing inspiration from the old Chinese proverb 'Read ten thousand books, travel ten thousand miles', we can liken the pretraining stage to 'Read ten thousand books'... Conversely, the fine-tuning stage can be likened to 'Travel ten thousand miles,' where the model is exposed to a wide array of prompts. It generates diverse responses and learns from the feedback it receives, thereby refining and enhancing its capabilities."

This framing positions UFT not merely as a technical trick for combining two loss functions, but as a **conceptual re-unification** of the post-training process. In the paper's vision, just as pretraining has a single objective (next-token prediction on a massive corpus), post-training should have a single objective (maximizing implicit reward relative to a reference model, where reward signals come from different labelers with different data formats). The distinction between "instruction data" and "preference data" becomes a matter of the reward label's value — 1.0 for expert demonstrations, and some explicit score (or comparison) for preference data — rather than a distinction of training paradigm.

The key theoretical justification that makes this unification credible is the **heuristic proof in Section 2.3** showing that when UFT's sigmoid-MSE loss is applied to data with $r_\phi(x, y) = 1$, it drives the implicit reward $r_\theta(x, y)$ toward $+\infty$, which — since $\pi_{\text{ref}}$ and $\beta$ are fixed — maximizes $\pi_\theta(y|x)$. In other words, UFT on SFT data is equivalent to SFT, but with an additional KL regularization toward the pretrained base model (since the loss involves $\pi_{\text{ref}}$). This provides a coherent explanation for why UFT *should* work: it preserves the SFT objective while adding a stabilizer that SFT alone lacks.

The paper positions UFT as solving three problems simultaneously:
1. **UFT outperforms SFT** when trained on instruction-tuning data alone, because the implicit KL penalty prevents overfitting and preserves pretrained knowledge (Tables 1–2, comparing Mistral+UFT vs. Mistral+SFT).
2. **UFT prevents sequential degradation** by co-training on instruction and alignment data, so the model never has to "forget" SFT capabilities to learn alignment (Tables 1–3, comparing Mistral+UFT vs. Mistral+SFT+DPO/KTO/UNA).
3. **UFT handles heterogeneous data formats** natively — instruction-tuning data (score=1), pairwise preferences, binary labels, and scalar scores can all coexist in the same batch under the same loss function (Equation 8), which is a capability that ORPO, DPO, and KTO individually lack.

The paper also positions itself as empirically practical. Unlike the theoretical density of the DPO/UNA derivations, the UFT recipe is straightforward to implement: (1) take any instruction-tuning dataset, assign reward $r=1$ to every $(x, y)$ pair; (2) merge it with alignment data that already has reward labels; (3) train with the UNA loss function (MSE between sigmoid of implicit reward and explicit reward). The code is open-sourced, the hyperparameter sweeps are documented (Appendix B), and the experiments use standard model sizes (7B, 32B) and datasets (UltraChat, HelpSteer2), making the approach directly reproducible.

A subtle but important aspect of the positioning: the paper does **not** claim that UFT solves *all* degradation or that it is universally superior. The data distribution experiment (Section 4.3, Tables 7–9) shows that the ratio of instruction-to-alignment data matters — too much instruction data degrades `truthful` performance, while too little reduces instruction-following capability — and that the optimal ratio is task-dependent. Similarly, the Qwen 32B results (Tables 4–5) show that the degradation problem diminishes with larger model size, suggesting UFT's benefits are most pronounced in the smaller-model regime where sequential training's interference effects are strongest. This nuanced positioning — UFT as an effective but ratio-sensitive solution whose advantages scale inversely with model size — is more credible than a claim of universal dominance.

## 3. Technical Approach

### 3.1 Reader Orientation (Approachable Technical Breakdown)

UFT is a training recipe — not a new model architecture or loss function from scratch, but a way of *repackaging* data and reusing an existing objective (UNA's implicit reward regression) so that instruction-tuning and alignment happen in one pass instead of two. The system solves the problem of **sequential fine-tuning degradation** by converting instruction-tuning examples into a format that alignment training already knows how to consume, then training everything jointly under a single loss function that maximises response quality while staying close to the pretrained model.

### 3.2 Big-Picture Architecture (Diagram in Words)

The system has four major components, arranged in a simple data-flow pipeline:

1. **Data Converter** — takes instruction-tuning pairs `$(x, y)$` and wraps them with a synthetic reward label `$r_\phi = 1$`, producing the same `$(x, y, r)$` triple format that alignment data already uses. This is a preprocessing step, not a trained component.
2. **Merged Training Set** — the union of the converted instruction data (now reward-labelled) and the original alignment data (pairwise, binary, or score-based). Both data types coexist in the same minibatches.
3. **Generalized Implicit Reward Model** — the core mathematical mechanism, inherited from UNA: for any prompt-response pair, the model computes its own implicit reward `$r_\theta(x, y) = \beta \log(\pi_\theta(y|x) / \pi_{\text{ref}}(y|x))$`, which measures how much more (or less) likely the current policy is to produce response `$y$` compared to the frozen reference model.
4. **Unified Loss Function** — a single regression loss (Sigmoid + MSE) that pushes the implicit reward toward the explicit reward label for every example in the merged dataset, regardless of whether that example came from instruction-tuning or alignment. This is the only training signal; there is no separate SFT loss term.

Information flows in a single training loop: sample a minibatch from the merged dataset → for each `$(x, y, r_\phi)$` triple, compute the policy's log-probability of `$y$` given `$x$` and the reference model's log-probability → compute the implicit reward `$r_\theta$` from the log-ratio → apply Sigmoid → compute MSE against `$r_\phi$` → backpropagate through the policy parameters `$\theta$` only (the reference model is frozen).

### 3.3 Roadmap for the Deep Dive

- **First**, the generalized implicit reward function (UNA's core contribution, which UFT inherits) — what it is, why it lacks the partition function `$Z(x)$` that constrained DPO, and how this enables handling arbitrary reward scores, not just pairwise preferences. This is the mathematical engine that makes everything else possible.
- **Second**, the UNA loss function — how the implicit reward is transformed through a Sigmoid and regressed against explicit reward scores via MSE. This covers the `$g(\cdot)$` function in Equation 8 and explains why this particular form (Sigmoid + MSE) works for SFT data with `$r_\phi = 1$`.
- **Third**, the data conversion step — how instruction-tuning pairs become reward-labelled triples, why a score of 1 is the natural choice, and what assumptions this encodes about the quality of SFT data.
- **Fourth**, the heuristic proof that UFT on SFT data is equivalent to SFT — the mathematical argument that maximising the implicit reward `$r_\theta(x, y)$` toward `$+\infty$` (by regressing the Sigmoid toward 1) maximises `$\pi_\theta(y|x)$`, but with an implicit KL regularisation toward `$\pi_{\text{ref}}$` that SFT lacks. This is the key theoretical justification for the entire framework.
- **Fifth**, the full UFT training procedure — how the merged dataset is constructed, what hyperparameters are swept, which base models and LoRA configurations are used, and what design choices (like keeping the alignment data fixed at 20k while varying instruction data) were made for the data distribution experiments.
- **Sixth**, the relationship between UFT and the pretraining vs. fine-tuning analogy — how the paper conceptualises the unified post-training framework as parallel to pretraining, with a single objective applied to heterogeneous data sources.

### 3.4 Detailed, Sentence-Based Technical Breakdown

This is primarily a **method unification paper** whose core idea is that SFT data can be recast as alignment data with a maximum reward label, enabling a single implicit-reward-based training stage to replace the sequential SFT-then-alignment pipeline without any new loss functions or architectural changes.

---

#### The Generalized Implicit Reward Function (UNA's Foundation)

The starting point for UFT is UNA's generalized implicit reward function, which defines a scalar reward for any response `$y$` given a prompt `$x$` purely in terms of the policy model `$\pi_\theta$` and a frozen reference model `$\pi_{\text{ref}}$`:

$$r_\theta(x, y) = \beta \log \left( \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)} \right)$$

where `$\pi_\theta(y|x)$` is the probability the current (trainable) model assigns to response `$y$` given prompt `$x$`, `$\pi_{\text{ref}}(y|x)$` is the probability the frozen reference model assigns to the same response, and `$\beta > 0$` is a temperature-like hyperparameter that controls the scale of the reward signal.

**What it computes:** the log-ratio of two probabilities, scaled by `$\beta$`. If the current model assigns higher probability to `$y$` than the reference model does, the reward is positive (the response is "rewarded" relative to the baseline). If the current model assigns lower probability, the reward is negative. The magnitude depends on how much the two distributions differ and on `$\beta$`: smaller `$\beta$` compresses the reward range, larger `$\beta$` amplifies it.

**Why this form:** this is derived from the optimal policy under a KL-constrained reward-maximisation objective (the same RLHF objective in Equation 3). The standard RLHF objective is:

$$\pi^*_\theta(y|x) = \max_{\pi_\theta} \mathbb{E}_{x \sim \mathcal{D}} \left[ \mathbb{E}_{y \sim \pi_\theta(y|x)} [r_\phi(x, y)] - \beta D_{\text{KL}} (\pi_\theta(y|x) \| \pi_{\text{ref}}(y|x)) \right]$$

Solving this (by setting the functional derivative to zero) yields the relationship that an optimal policy satisfies `$r_\phi(x, y) = \beta \log(\pi^*_\theta(y|x) / \pi_{\text{ref}}(y|x)) + \beta \log Z(x)$`, where `$Z(x)$` is a prompt-dependent partition function that ensures the policy normalises to a valid probability distribution. DPO (Equation 4) retains this `$Z(x)$` term and cancels it by subtracting rewards between paired responses, which is why DPO is restricted to pairwise data — the `$Z(x)$` term is intractable but common to both responses for the same prompt, so subtraction eliminates it.

UNA's critical departure from DPO is **dropping the `$Z(x)$` term entirely**. Equation 5 defines the generalized implicit reward as simply `$\beta \log(\pi_\theta / \pi_{\text{ref}})$` without the partition function. This is not an approximation of the optimal policy relationship — it is a *definition* of a reward function that the model computes internally. The argument is: if we train the model to make this internally-computed reward match an external explicit reward signal `$r_\phi(x, y)$` (from human labelers, reward models, or data construction), then we are effectively optimising the policy without needing to estimate `$Z(x)$` at all. The partition function is absorbed into the regression target: if the explicit reward `$r_\phi$` is systematically offset by some `$Z(x)$`-like constant, the regression will learn to compensate for it (or, more precisely, the constant cancels out when comparing distributions over responses from the same prompt, which is what downstream sampling does).

This move from *derived relationship* (DPO) to *defined reward* (UNA) is what gives UNA — and hence UFT — its generality. Since the implicit reward is just a scalar computed from the policy ratio, any explicit reward signal — pairwise comparisons, binary labels, continuous scores — can be used as a regression target. The constraint that the data must allow `$Z(x)$` cancellation disappears because `$Z(x)$` was never part of the reward definition to begin with.

The `$\beta$` hyperparameter plays a crucial role: it controls the sensitivity of the implicit reward to changes in the policy. When `$\beta$` is small, even large shifts in `$\pi_\theta$` produce small reward changes, meaning the policy has more freedom to move without incurring large implicit reward discrepancies (effectively, a weaker regularisation toward `$\pi_{\text{ref}}$`). When `$\beta$` is large, small policy shifts produce large reward changes, tightening the coupling to the reference model. The paper sweeps `$\beta \in \{0.01, 0.03, 0.1, 0.3\}$` (Tables 12–13) and finds `$\beta = 0.01$` optimal at the best learning rate of `$3 \times 10^{-5}$` for Mistral 7B. This relatively small `$\beta$` means the model is given substantial room to move away from the pretrained distribution — consistent with the fact that instruction-tuning requires significant behavioural change.

---

#### The UNA / UFT Loss Function

UFT inherits the UNA loss function in Equation 6, which minimises the discrepancy between the implicit reward `$r_\theta(x, y)$` and the explicit reward `$r_\phi(x, y)$` through a general function `$g(\cdot, \cdot)$`:

$$\mathcal{L}_{\text{UFT}}(\pi_\theta) = \mathbb{E}_{(x,y) \sim \mathcal{D}} \left[ g(r_\theta(x, y), r_\phi(x, y)) \right]$$

where `$\mathcal{D}$` is the merged training dataset containing both converted instruction-tuning data and alignment data, `$r_\theta(x, y) = \beta \log(\pi_\theta(y|x) / \pi_{\text{ref}}(y|x))$` is the implicit reward from Equation 5, `$r_\phi(x, y)$` is the explicit reward label associated with the example, and `$g(\cdot, \cdot)$` is a discrepancy function.

**What it computes:** for each example in the dataset, compute the model's internal reward estimate and compare it to the target reward label via some distance metric. The expectation averages this distance over the training distribution, producing a scalar training signal.

**Why this form:** the generality of `$g$` is intentional — different choices of discrepancy function correspond to different assumptions about the reward distribution. The paper does not explore this generality deeply (it fixes `$g$` to be Sigmoid + MSE for all experiments), but the framework allows `$g$` to be, for example, cross-entropy (for binary labels), hinge loss (for pairwise data), or Huber loss (for robustness to outliers in continuous scores).

The specific instantiation used throughout the paper applies a Sigmoid transformation to the implicit reward and then computes MSE against the explicit reward:

$$\mathcal{L}_{\text{UFT-SFT}}(\pi_\theta) = \mathbb{E}_{(x,y) \sim \mathcal{D}} \left[ \left( \sigma(r_\theta(x, y)) - r_\phi(x, y) \right)^2 \right]$$

where `$\sigma(z) = 1 / (1 + e^{-z})$` is the logistic Sigmoid function. Substituting the implicit reward definition gives the full form for SFT data (Equation 7):

$$\mathcal{L}_{\text{UFT-SFT}}(\pi_\theta) = \mathbb{E}_{(x,y) \sim \mathcal{D}} \left[ \left( \sigma\left( \beta \log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)} \right) - 1 \right)^2 \right]$$

**What this expanded form computes:** for each SFT training pair `$(x, y)$`, the model computes the log-ratio of the current policy probability to the reference policy probability for generating `$y$`, scales it by `$\beta$`, squashes it through the Sigmoid to the interval `$[0, 1]$`, and then computes the squared difference from the target value `$1$`. If the current model assigns much higher probability to `$y$` than the reference model, the log-ratio is large and positive, `$\sigma(\cdot)$` is close to 1, and the loss is near 0. If the current model assigns similar or lower probability than the reference, `$\sigma(\cdot)$` is near or below 0.5, and the loss is substantial.

**Why Sigmoid + MSE:** the Sigmoid maps the unbounded implicit reward `$r_\theta \in (-\infty, +\infty)$` to a bounded interval `$[0, 1]$` that matches the bounded explicit reward labels (which the paper assumes lie in `$[0, 1]$`, with 1 being maximum quality). This avoids the scaling mismatch that would occur if MSE were applied directly to the unbounded `$r_\theta$` — the model could trivially reduce loss by making `$r_\theta$` arbitrarily large for all examples, regardless of their relative quality. The Sigmoid saturates, so pushing `$r_\theta$` from 5 to 10 produces almost no further reduction in the loss, while pushing it from 0 to 2 produces a large reduction. This creates a natural "diminishing returns" pressure that prevents unbounded policy drift.

The choice of MSE rather than binary cross-entropy for the SFT case (where `$r_\phi = 1$` is a hard 0/1 label) deserves comment. If the labels were treated as binary, cross-entropy `$-[1 \cdot \log(\sigma(r_\theta)) + 0 \cdot \log(1 - \sigma(r_\theta))] = -\log(\sigma(r_\theta))$` would be the maximum-likelihood objective. The paper uses MSE instead, which is equivalent to treating the regression target as a continuous value and penalising squared error. The authors do not explicitly justify MSE over BCE for SFT data, but the practical advantage is that MSE provides a smoother gradient landscape — as `$\sigma(r_\theta) \to 1$`, the gradient of MSE `$2(\sigma(r_\theta) - 1) \cdot \sigma'(r_\theta)$` decays to zero smoothly, whereas the gradient of BCE `$-\sigma'(r_\theta) / \sigma(r_\theta)$` can be numerically unstable as `$\sigma(r_\theta) \to 1$` due to the division by a quantity approaching 1 (though this is typically well-behaved in practice with proper numerics). More importantly, using the same loss function (MSE) for both SFT data (with `$r_\phi = 1$`) and continuous-score alignment data (with `$r_\phi \in [0, 1]$`) means there is no loss-function switch when merging datasets — everything flows through the same `$g$` function, which is the architectural simplicity the paper is aiming for.

For alignment data with general explicit rewards `$r_\phi \in [0, 1]$`, the loss is identical in form:

$$\mathcal{L}_{\text{UFT-align}}(\pi_\theta) = \mathbb{E}_{(x,y) \sim \mathcal{D}_{\text{align}}} \left[ \left( \sigma\left( \beta \log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)} \right) - r_\phi(x, y) \right)^2 \right]$$

The only difference is that `$r_\phi(x, y)$` can take any value in `$[0, 1]$` rather than being fixed at 1. This means the model learns to modulate its policy ratio to produce implicit rewards that match the continuous quality score: high-quality responses get pushed to have large positive implicit rewards, low-quality responses get pushed toward negative implicit rewards, and intermediate responses settle at intermediate values. The policy is thus trained to assign probability mass in proportion to the explicit reward scores, with the reference model serving as the anchor point.

When the dataset contains pairwise preference data (the HelpSteer2 dataset used in experiments is of this type — it provides `$y_w$` and `$y_l$` with associated scores), UFT converts each pair into two separate training examples: `$(x, y_w, r_\phi(x, y_w))$` and `$(x, y_l, r_\phi(x, y_l))$`, where the scores come from the dataset's reward annotations. The model does not see the pairwise structure — it just sees individual scored responses and learns to assign implicit rewards that match the scores. This is a departure from DPO, which explicitly uses the *difference* between preferred and dispreferred response rewards in its loss. UFT's approach is simpler but discards the relative information; whether this matters empirically depends on whether the absolute scores are well-calibrated.

---

#### Converting SFT Data to Reward-Labelled Triples

The central insight that enables UFT is deceptively simple: **instruction-tuning data can be treated as alignment data where the response has a perfect reward score**. The paper states (Section 2.3):

> "Due to the high quality of instruction-tuning data, they can be regarded as data with a score of 1, i.e., positive feedback. With this consideration, the instruction-tuning dataset can be transformed into alignment data in the format of prompt `$x$`, response `$y$` and feedback `$r = 1$`, which is the highest reward in consideration."

The conversion procedure operates at the data preprocessing level, not during training. Given an instruction-tuning dataset `$\mathcal{D}_{\text{SFT}} = \{(x_i, y_i)\}_{i=1}^{N_{\text{SFT}}}$` where each `$y_i$` is an expert-written (or otherwise high-quality) response to prompt `$x_i$`, the converter produces:

$$\mathcal{D}_{\text{SFT-converted}} = \{(x_i, y_i, r_\phi = 1)\}_{i=1}^{N_{\text{SFT}}}$$

This is then merged with the existing alignment dataset `$\mathcal{D}_{\text{align}} = \{(x_j, y_j, r_\phi(x_j, y_j))\}_{j=1}^{N_{\text{align}}}$` to form the unified training set `$\mathcal{D}_{\text{UFT}} = \mathcal{D}_{\text{SFT-converted}} \cup \mathcal{D}_{\text{align}}$`.

**Why `$r_\phi = 1$` is the right choice.** The assignment of a score of 1 encodes the assumption that instruction-tuning responses are *maximally good* — they represent the desired behaviour that the model should learn to reproduce. In the implicit reward framework, a target of 1 (after Sigmoid) means the model must learn to assign `$\pi_\theta(y|x) \gg \pi_{\text{ref}}(y|x)$` for these responses — the policy must become much more likely to generate these responses than the base pretrained model would. This is exactly what SFT does: it increases the probability of the demonstrated responses. The difference, as discussed below, is that UFT does this through the KL-regularised reward-maximisation pathway rather than through direct likelihood maximisation.

The choice of 1 as the maximum also creates a natural compatibility with alignment data. If the alignment dataset provides explicit reward scores in `$[0, 1]$` (which HelpSteer2 does, since it provides scalar helpfulness/harmlessness scores that can be normalised), then `$r_\phi = 1$` for SFT data means "these responses are at least as good as the best alignment responses, and probably better, since they were written by experts." This creates a coherent ordering: SFT responses ≥ best alignment responses > mediocre alignment responses > worst alignment responses. The model learns to assign the highest probability boosts to SFT-quality responses, moderate boosts to good alignment responses, and probability suppression to poor alignment responses.

**What assumptions this encodes.** The `$r_\phi = 1$` assignment assumes that *all* instruction-tuning responses are of uniformly perfect quality. In real datasets, this is rarely true — SFT data often contains suboptimal, inconsistent, or even contradictory examples. The paper does not address this; it treats the SFT data as a monolith of quality. A more sophisticated version of UFT could assign continuous scores to SFT data based on some quality metric (e.g., reward model scores, length, diversity), but this is left to future work. The practical consequences of the uniform-1 assumption are that the model may overfit to low-quality SFT examples by learning to assign them maximal implicit reward, which could manifest as degraded performance on metrics that penalise low-quality generations.

The paper also implicitly assumes that the SFT and alignment data distributions are compatible — that the prompts in UltraChat and HelpSteer2 are drawn from similar enough domains that training on both simultaneously does not create conflicting objectives. The data distribution experiment (Section 4.3) partially validates this by showing that mixing does not catastrophically degrade performance, but the assumption is not rigorously tested on out-of-distribution prompts.

---

#### The Heuristic Proof: Why UFT on SFT Data Is Equivalent to SFT (But Better)

The paper provides a heuristic argument (Section 2.3) that UFT trained on SFT data with `$r_\phi = 1$` achieves the same goal as SFT — maximising `$\pi_\theta(y|x)$` for the demonstrated responses — but with an additional benefit that explains its empirical superiority. The argument proceeds as follows:

**Step 1: The loss drives the implicit reward to infinity.** With `$r_\phi = 1$` and MSE loss after Sigmoid, the loss for an SFT example is:

$$\mathcal{L} = \left( \sigma\left( \beta \log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)} \right) - 1 \right)^2$$

Minimising this loss requires `$\sigma(r_\theta(x, y)) \to 1$`, which in turn requires `$r_\theta(x, y) \to +\infty$`. Since `$\sigma(z) = 1/(1 + e^{-z})$`, we have `$\sigma(z) = 1$` only in the limit `$z \to \infty$`. The loss can never reach exactly zero, but it can become arbitrarily small as the implicit reward grows large.

**Step 2: The implicit reward going to infinity forces `$\pi_\theta(y|x)$` up.** Recall that:

$$r_\theta(x, y) = \beta \log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}$$

For this to go to `$+\infty$`, the log-ratio must go to `$+\infty$`. Since `$\beta > 0$` is fixed and `$\pi_{\text{ref}}(y|x)$` is fixed (the reference model is frozen), the only way for the log-ratio to diverge is for `$\pi_\theta(y|x) \to 1$` (since a probability cannot exceed 1). If `$\pi_{\text{ref}}(y|x)$` were 0 for some token in `$y$`, the ratio would be undefined; but in practice, pretrained language models assign non-zero probability to all token sequences (the distribution has full support due to softmax), so `$\pi_{\text{ref}}(y|x) > 0$` and the ratio is well-defined.

Thus, UFT on SFT data maximises `$\pi_\theta(y|x)$` — exactly what SFT's cross-entropy loss does.

**Step 3: The critical difference — KL regularisation.** The difference between UFT and SFT is *how* `$\pi_\theta(y|x)$` is maximised. SFT directly pushes `$\pi_\theta(y|x)$` toward 1 via gradient descent on `$-\log \pi_\theta(y|x)$`, with no regard for what happens to the probabilities of other responses. The model can (and often does) achieve this by radically reshaping its output distribution, potentially overwriting pretrained knowledge for prompts that are similar to the training examples.

UFT maximises `$\pi_\theta(y|x)$` by maximising the ratio `$\pi_\theta(y|x) / \pi_{\text{ref}}(y|x)$`. This means that to increase the loss-minimising quantity `$\sigma(\beta \log(\pi_\theta / \pi_{\text{ref}}))$`, the model must increase `$\pi_\theta(y|x)$` *relative to* the reference model's probability. If `$\pi_{\text{ref}}(y|x)$` is already high (i.e., the pretrained model already assigns substantial probability to the correct response), the model does not need to change much — a small increase in `$\pi_\theta(y|x)$` yields a large enough ratio. If `$\pi_{\text{ref}}(y|x)$` is low, the model must work harder, but it is penalised for changing the distribution *more than necessary* because the ratio-based objective rewards efficient increases.

This implicit KL regularisation is why the paper claims UFT outperforms SFT: it prevents the model from "forgetting" pretrained knowledge by anchoring the optimisation to `$\pi_{\text{ref}}$`. This is analogous to the KL penalty in RLHF (Equation 3), but instead of being an explicit added term with a tunable coefficient, it emerges from the structure of the ratio-based reward definition. The `$\beta$` hyperparameter controls the strength of this regularisation — smaller `$\beta$` weakens it, larger `$\beta$` strengthens it — mirroring the role of the KL coefficient in standard RLHF.

**Step 4: Where the argument is heuristic, not rigorous.** The argument says that the loss drives `$r_\theta \to +\infty$`, which drives `$\pi_\theta(y|x) \to 1$`. But in practice, with finite training steps and a finite learning rate, `$\pi_\theta(y|x)$` never reaches 1. The model settles at some intermediary point where the gradient of the Sigmoid-MSE loss balances against the regularising effect of the reference model (and the implicit regularisation from limited model capacity and finite data). The heuristic argument establishes *directional correctness* — UFT pushes `$\pi_\theta(y|x)$` upward, just like SFT — but does not characterise *where* it stops.

Furthermore, the argument applies to a single SFT example in isolation. In a dataset with many examples, the model must balance conflicting demands: maximise `$\pi_\theta(y_1|x_1)$` for example 1, `$\pi_\theta(y_2|x_2)$` for example 2, and so on. Since the `$\pi_\theta$` distribution must sum to 1 over all possible responses for each prompt, the model cannot simultaneously push all SFT responses to probability 1 — it must allocate probability mass across them. The ratio-based objective changes *how* this allocation happens relative to SFT, but the heuristic argument does not characterise the multi-example equilibrium.

The paper also does not prove that the ratio-based regularisation is *optimal* — it demonstrates empirically that UFT outperforms SFT on the tested benchmarks (Tables 1–2), but the claim that this is "because of KL regularisation" is an interpretation, not a proven causal mechanism. Ablations that vary `$\beta$` (Tables 12–13) provide supporting evidence: at `$\beta = 0.01$` with `$\text{lr} = 3\times 10^{-5}$`, UFT achieves its best average performance (30.09 on the new leaderboard, 64.25 on the old leaderboard for Mistral 7B), suggesting an intermediate regularisation strength is optimal. But the paper does not compare directly against SFT with explicit KL regularisation added (which would be SFT + `$\lambda \cdot D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})`$), which would test whether the ratio-based mechanism provides benefits beyond a simple additive KL penalty.

---

#### Training Procedure: Models, Data, Hyperparameters, and Design Choices

The experimental setup for UFT is documented across Sections 3 and 4 and Appendices A and B.

**Base models.** Two model families are used:
- **Mistral 7B-v0.1** (Jiang et al., 2023) — a 7-billion-parameter dense transformer, representative of the "small but capable" model class that is most susceptible to sequential degradation (the paper's primary target).
- **Qwen 2.5 32B** (Qwen et al., 2025) — a 32-billion-parameter model that tests whether UFT's benefits persist at larger scales where degradation is naturally less severe.

For both models, the pretrained base checkpoint is used as the starting point for all experiments (including the sequential SFT baselines). There is no intermediate SFT checkpoint — the base model goes directly into UFT training.

**Parameter-efficient fine-tuning with LoRA.** All training uses Low-Rank Adaptation (LoRA; Hu et al., 2021) with rank `$r = 16$`. The paper does not specify the LoRA alpha, dropout, or target modules (e.g., whether LoRA is applied only to attention weights or also to feed-forward layers). This is a notable omission — LoRA configuration can significantly impact training dynamics, and the interaction between LoRA's low-rank constraint and the ratio-based objective is not discussed. The choice of `$r = 16$` is standard for 7B models but relatively small for 32B, which may partially explain why the Qwen results show smaller gains from UFT.

**Reference model.** The reference model `$\pi_{\text{ref}}$` is the frozen pretrained base model (Mistral 7B-v0.1 or Qwen 32B). It never receives gradient updates. This is a critical design choice: using the pretrained model rather than an SFT-intermediate model as the reference means that UFT computes its implicit rewards relative to the *original* language model distribution, not relative to any fine-tuned state. This is different from standard RLHF and DPO pipelines, where the reference model is typically the SFT model — the model that has already been trained to follow instructions. By anchoring to the pretrained model, UFT ensures that both the instruction-tuning signal and the alignment signal operate relative to the same baseline, preventing the "reference model shift" that occurs in sequential pipelines.

**Training data.** Two datasets are used:
- **UltraChat** (Ding et al., 2023) — a large-scale multi-turn dialogue dataset generated by ChatGPT. The paper "unfolds" each conversation into multiple training examples by treating each exchange as a separate `$(x, y)$` pair (where `$x$` is the conversation history up to that point and `$y$` is the response). From the full UltraChat dataset, 20,000 samples are selected for the main experiments (Sections 4.1–4.2), and additional samples (16k, 32k, 65k, 130k, 260k) are drawn for the data distribution experiments (Section 4.3).
- **HelpSteer2** (Wang et al., 2024) — a preference dataset of 20,000 examples containing prompts, responses, and scalar scores for helpfulness, correctness, coherence, complexity, and verbosity. The paper uses 20,000 examples from this dataset for all alignment experiments. The exact mapping from HelpSteer2's multi-dimensional scores to the single scalar `$r_\phi \in [0, 1]$` used in UFT is not described — the paper does not specify whether it uses a single dimension (e.g., helpfulness), an average, or some other aggregation.

**Dataset construction for UFT.** For the main UFT experiments, the training set is constructed by:
1. Converting the 20k UltraChat samples to `$(x, y, r_\phi = 1)$` triples.
2. Merging these with the 20k HelpSteer2 samples (which already have reward labels).
3. The resulting dataset has 40k examples — a 1:1 ratio of instruction-tuning to alignment data.

For the data distribution experiments (Section 4.3), the 20k alignment examples are fixed, while the number of instruction-tuning examples is varied across {16k, 20k, 32k, 65k, 130k, 260k}, producing total dataset sizes from 36k to 280k and instruction-to-alignment ratios from 0.8:1 to 13:1.

**Hyperparameter sweeps.** The paper conducts thorough sweeps for both SFT and UFT (documented in Appendices A and B).

For SFT (Tables 10–11):
- Learning rates: `$3 \times 10^{-6}, 1 \times 10^{-5}, 3 \times 10^{-5}, 1 \times 10^{-4}, 3 \times 10^{-4}$`
- Best learning rate: `$1 \times 10^{-4}$` achieves the highest average on the new leaderboard (29.87), though `$3 \times 10^{-5}$` achieves the highest average on the old leaderboard (63.72). The paper selects `$1 \times 10^{-4}$` for the main comparison, presumably prioritising the new leaderboard.
- At `$3 \times 10^{-4}$`, performance collapses (average 26.98 on the new leaderboard, 55.18 on the old), indicating a narrow usable learning rate range for SFT with LoRA.

For UFT (Tables 12–13):
- Learning rates: `$3 \times 10^{-6}, 1 \times 10^{-5}, 3 \times 10^{-5}, 1 \times 10^{-4}$`
- `$\beta$` values: `$0.01, 0.03, 0.1, 0.3$`
- Best combination: `$\text{lr} = 3 \times 10^{-5}$`, `$\beta = 0.01$` achieves the highest or near-highest average on both leaderboards (30.09 new, 64.25 old).
- The `$\text{lr} = 1 \times 10^{-4}, \beta = 0.3$` configuration — the highest learning rate with the strongest regularisation — performs worst (61.07 on the old leaderboard), consistent with the interpretation that strong KL regularisation at high learning rates prevents the model from learning the instruction-following task effectively.

The paper does not specify other hyperparameters (optimizer, batch size, sequence length, training epochs, learning rate schedule, warmup steps). Given that it uses LoRA with standard HuggingFace PEFT integration (implied by the GitHub repository link), the defaults are likely AdamW with a linear or cosine schedule, but this is not stated. The omission of batch size and training epochs makes it impossible to assess whether the models were trained to convergence or for a fixed number of steps — a relevant detail since the ratio-based UFT objective may converge at different rates than SFT's cross-entropy.

**Training hardware and duration.** Not specified. Given that the experiments use LoRA on 7B and 32B models with 20k–280k examples, the compute requirements are modest by modern standards — likely a few GPU-hours on A100-class hardware for the 7B experiments.

**Baseline sequential methods.** For the SFT+Alignment baselines, the best SFT model (`$\text{lr} = 1 \times 10^{-4}$`) is taken and further fine-tuned with DPO, KTO, and UNA on the 20k HelpSteer2 alignment examples. The hyperparameters for these alignment stages are not detailed — the paper does not specify DPO's `$\beta$`, KTO's hyperparameters, or UNA's `$\beta$` and learning rate for the sequential runs. This is a significant gap because the alignment stage hyperparameters strongly influence the degree of degradation; suboptimal alignment tuning could exaggerate UFT's apparent advantage.

**Evaluation protocol.** All models are evaluated on:
- The new HuggingFace Open LLM Leaderboard v2 (6 tasks: bbh, gpqa, mmlu-pro, musr, ifeval, math-hard), reporting average scores.
- The old HuggingFace Open LLM Leaderboard (6 tasks: gsm8k, truthful, winograde, arc, hellaswag, mmlu), reporting standard metrics per task.
- MT-Bench (multi-turn dialogue quality, judged by GPT-4).
- Alpaca-eval (Length-Controlled Win Rate against reference responses).

The evaluation uses pretrained, off-the-shelf evaluation harnesses (the HuggingFace leaderboard infrastructure), meaning the evaluation protocol is standardised and reproducible. No task-specific fine-tuning or prompting tricks are applied beyond what the leaderboard defaults provide.

**Notable design choices and their justifications.**
- **LoRA rather than full fine-tuning:** the paper justifies this implicitly by the need to run many hyperparameter configurations and dataset size ablations. LoRA's parameter efficiency makes sweeping practical. The tradeoff is that LoRA may not capture all the benefits of full-weight training — the paper does not verify whether UFT's advantages over SFT persist under full fine-tuning.
- **Frozen pretrained reference rather than SFT reference:** discussed above; this is the design choice that structurally prevents the sequential degradation problem by giving both SFT and alignment data the same anchor.
- **Equal-weight merging of SFT and alignment data (20k + 20k):** this produces a 1:1 ratio that the paper uses for the main comparison. The data distribution experiment shows this is near-optimal but not necessarily the best — slightly more instruction data (32k) improves `ifeval` further (46.76 vs. 46.03) at a small cost to `truthful` (54.38 vs. 54.05). The paper does not claim 1:1 is universally optimal; it presents it as a reasonable default that works across the tested range.
- **Sigmoid + MSE as the discrepancy function `$g$`:** the paper inherits this from UNA without ablating alternatives (e.g., BCE, Huber, L1). The choice is justified by UNA's prior results showing it works for score-based alignment, but its specific suitability for SFT data (where labels are binary 1's) is not compared against other options like directly using BCE for the SFT portion.

---

#### The Unified Post-Training Framework: Pretraining and Fine-Tuning as Parallels

Section 2.4 of the paper articulates a conceptual vision that goes beyond the technical mechanics of UFT. The authors propose that with UFT, the post-training process (SFT + alignment) becomes a **single stage with a single objective**, structurally parallel to pretraining:

> "By integrating SFT and alignment through UFT, we can establish a unified fine-tuning framework that runs parallel to the pretraining phase."

The analogy, drawn from the Chinese proverb "Read ten thousand books, travel ten thousand miles," maps as follows:
- **Pretraining = "Read ten thousand books":** the model passively consumes trillions of tokens, learning linguistic patterns, factual knowledge, and reasoning capabilities through next-token prediction with cross-entropy loss. The data is undifferentiated — all tokens from all sources are treated identically, and the objective is universal.
- **Fine-tuning (UFT) = "Travel ten thousand miles":** the model actively engages with specific prompts, generates responses, and receives feedback on the quality of those responses. The data is differentiated by reward labels, but the objective — maximise implicit reward relative to the pretrained baseline — is again universal.

The structural parallel is that just as pretraining has exactly one loss function (cross-entropy) applied to one data format (token sequences), UFT-based post-training has exactly one loss function (Sigmoid-MSE on implicit rewards) applied to one data format `$(x, y, r_\phi)$` triples. The heterogeneity of post-training data — instruction demonstrations, preference pairs, binary feedback, scalar scores — is handled not by switching objectives but by encoding the heterogeneity into the reward label `$r_\phi$`, which the unified loss consumes without discrimination.

This framing is more than philosophical. It has a concrete implication for data acquisition and training pipeline design: under UFT, there is no need to separate "SFT data collection" from "alignment data collection." Any source of feedback on response quality — expert demonstrations (score=1), user thumbs-up/down, reward model scores, LLM-as-judge evaluations — can be added to the training mix at any time, in any proportion, without changing the training code. The model's behaviour is controlled by the distribution of reward labels in the training data, not by the sequence of training stages.

The paper does not fully realise this vision — its experiments use only two datasets (UltraChat and HelpSteer2) in a single training run — but the architectural foundation is in place. The data distribution experiment (Section 4.3) is a step toward demonstrating that the ratio of data types matters and can be tuned, much as pretraining data mixtures are tuned. Future work could explore dynamic mixing, curriculum learning (start with high-reward SFT data, gradually introduce alignment data), or active data acquisition guided by the model's current implicit reward distribution.

---

#### Summary of the Technical Architecture

UFT can be understood as a **data reformulation layer plus a generalised reward regression engine**. The reformulation layer converts SFT pairs to reward triples by assigning `$r_\phi = 1$`. The regression engine — inherited entirely from UNA — defines a scalar implicit reward `$r_\theta = \beta \log(\pi_\theta / \pi_{\text{ref}})$`, squashes it through a Sigmoid to `$[0, 1]$`, and minimises the MSE against the explicit reward label. When trained on merged data, the engine simultaneously maximises `$\pi_\theta(y|x)$` for expert demonstrations (like SFT) and calibrates the policy ratio to match preference scores (like alignment), with both objectives operating relative to the same frozen pretrained reference model. The KL regularisation that emerges from the ratio-based formulation prevents the overfitting and catastrophic forgetting that plague sequential pipelines, while the unified loss function eliminates the need for stage-specific hyperparameters or loss-function switching.

The framework's primary limitation from a technical standpoint is that it inherits UNA's assumption that the implicit reward — a scalar derived from a single log-ratio — is a sufficient representation of response quality. It cannot capture multi-dimensional tradeoffs (e.g., "this response is more helpful but less concise") in a principled way, and it provides no mechanism for handling cases where the SFT and alignment objectives genuinely conflict (e.g., an SFT response that is high-quality but harmful — should it get `$r_\phi = 1$` or `$r_\phi = 0$`?). These are fundamental limitations of scalar reward modelling, not specific to UFT, but they become more pressing in a unified framework where the scalar must encode everything the model should learn about response quality.

## 4. Key Insights and Innovations

### Innovation 1: Reframing SFT as a Special Case of Alignment — Not a Separate Training Paradigm

The paper's most conceptually distinctive move is the recognition that **SFT data is structurally identical to alignment data with a maximum reward label**, and that this re-labelling alone is sufficient to collapse two training stages into one. This is not a new loss function or a clever optimisation trick — it is a **reframing of what SFT data represents** within the reward-modelling worldview that alignment methods already inhabit.

Before this work, the field treated SFT and alignment as fundamentally different animals. SFT was framed as *behavioural cloning* — imitate the expert — operating through token-level cross-entropy on demonstrations. Alignment was framed as *preference learning* — satisfy human judgments — operating through reward maximisation (explicit or implicit) on comparative feedback. They used different loss functions, different data formats, different reference models, and different training philosophies. The fact that sequential application caused degradation was treated as an unfortunate side effect to be mitigated (through KL penalties, learning rate annealing, or adapter merging in PAFT), not as evidence that the separation itself was artificial.

UFT challenges this dichotomy at its root. The core insight, appearing in Section 2.3, is deceptively simple:

> "Due to the high quality of instruction-tuning data, they can be regarded as data with a score of 1, i.e., positive feedback."

This reframes expert demonstrations not as targets to imitate but as **responses that received the highest possible reward from an implicit labeler** — the human expert who wrote them. In this view, SFT data is just a particular kind of preference data where the preference is absolute (score = 1) rather than relative. The expert didn't just prefer their response over some alternative; they produced a response they consider maximally good, and the model should learn to assign it maximal implicit reward.

**Why this is non-obvious.** Prior unification attempts (ORPO, PAFT) tried to *combine* SFT and alignment objectives — adding an odds-ratio penalty to the cross-entropy loss, or training separate adapters and merging them. These approaches implicitly accepted that SFT and alignment are different things that need to be glued together. UFT's move is more radical: it asserts they are *the same thing*, just with different reward labels. The SFT cross-entropy loss is not supplemented or regularised with an alignment term — it is **replaced entirely** by the same reward-regression loss that handles preference data. This is a category shift from "two things we combine" to "one thing we parametrise differently."

**The significance is methodological, not just empirical.** If SFT is a special case of alignment, then:
- **Data acquisition can be unified.** Any source of quality signal — demonstrations, preferences, ratings, binary feedback — feeds into the same training pipeline with the same code. There is no need to maintain separate SFT and alignment data collection workflows.
- **The reference model question is resolved.** In sequential pipelines, the alignment stage typically uses the SFT model as reference, creating a moving baseline. UFT uses the pretrained model as reference for *everything*, because both instruction data and preference data are evaluated relative to the same starting point. This eliminates the "reference model shift" that likely contributes to sequential degradation.
- **Hyperparameter tuning is simplified.** Instead of tuning SFT hyperparameters (learning rate, epochs) and alignment hyperparameters (β, KL coefficient, PPO clipping) separately and worrying about their interaction, there is a single set of hyperparameters (learning rate, β) applied to a single training run.

**The evidence for this reframing's validity** comes from Tables 1–2, where UFT trained *only* on the converted UltraChat data (no alignment data at all) outperforms SFT trained on the same data. If UFT were merely an awkward way to approximate SFT, it should underperform — the Sigmoid-MSE loss on a ratio is a less direct way to maximise likelihood than cross-entropy. The fact that it *outperforms* SFT (30.09 vs. 29.87 on the new leaderboard average for Mistral 7B; 64.25 vs. 63.17 on the old leaderboard) suggests that the reframing does more than replicate SFT — it improves upon it, likely through the implicit KL regularisation discussed in Section 3.

**This is a fundamental reframing, not an incremental improvement.** It changes how one thinks about what post-training *is*. Before UFT: post-training is a sequence of distinct operations applied to a pretrained model. After UFT: post-training is a single operation — reward-conditioned policy optimisation — where the reward labels encode everything the model should learn, from "this is a perfect response" to "this response is slightly better than that one." The distinction between instruction-tuning and alignment becomes a matter of data labelling, not training methodology.

---

### Innovation 2: The Implicit KL Regularisation Mechanism as a Self-Stabilising Alternative to Explicit Penalties

The paper's second distinctive contribution is an empirical demonstration — supported by a heuristic argument — that the ratio-based implicit reward formulation provides a **built-in regularisation toward the pretrained model** that conventional SFT lacks, and that this regularisation is sufficient to prevent the overfitting and capability loss that plague standard fine-tuning. This is not a new theoretical result (the KL interpretation of policy ratios is standard in the RLHF literature), but **using it as SFT's primary stabiliser — without any explicit KL penalty term — is novel**.

**What the field did before.** Standard SFT applies cross-entropy loss to maximise `π_θ(y|x)` for demonstration responses. There is no mechanism preventing the model from radically reshaping its output distribution to achieve this — it can suppress pretrained knowledge for any prompt that shares statistical patterns with the training data. This manifests as the well-known phenomenon where SFT models lose general reasoning capability, become less calibrated, or forget rare knowledge, even as their likelihood on training responses improves. The standard mitigations are early stopping (stop before overfitting), small learning rates, or mixing in pretraining data during SFT ("data rehearsal") — all external interventions, not properties of the objective itself.

In the alignment literature, KL regularisation is standard and explicit. RLHF adds `−β D_KL(π_θ || π_ref)` directly to the reward-maximisation objective (Equation 3). DPO inherits this implicitly through the `π_ref` in the loss denominator. But in both cases, `π_ref` is typically the SFT model, not the pretrained model — the regularisation prevents the aligned model from diverging from the *instruction-tuned* model, not from the original pretrained distribution. The SFT stage itself remains unregularised.

**What UFT does differently.** By defining the implicit reward as `r_θ(x, y) = β log(π_θ(y|x) / π_ref(y|x))` with `π_ref` set to the *pretrained* model, UFT builds KL regularisation into the very definition of what the model is optimising. To increase the implicit reward for a demonstration response, the model must increase `π_θ(y|x)` **relative to** `π_ref(y|x)`. If the pretrained model already assigns high probability to the correct response, the model barely needs to change — the ratio is already large. If the pretrained model assigns low probability, the model must work to increase it, but the ratio-based signal ensures that every unit of probability mass reallocated to the target response comes with a "cost" proportional to how much the pretrained model disfavoured that response.

This is **self-stabilising** in a way that explicit KL penalties are not. An explicit KL penalty adds a term `−β D_KL(π_θ || π_ref)` to the loss, which penalises *any* deviation from `π_ref` regardless of whether that deviation serves a useful purpose. The practitioner must choose `β` to balance the two terms — too large and the model learns nothing, too small and it overfits. The ratio-based formulation penalises deviations *selectively*: the model is penalised for changing `π_θ(y|x)` in ways that don't increase the ratio, but is actively encouraged to change it in ways that do. The regularisation is not a separate term fighting the objective — it is the objective.

**The evidence.** Tables 12–13 (Appendix B) show that UFT performance is sensitive to `β`, with `β = 0.01` optimal for Mistral 7B at learning rate `3 × 10⁻⁵`. At `β = 0.3` (stronger regularisation), performance degrades (61.07 vs. 64.25 on the old leaderboard average), confirming that over-regularisation hurts. But the key comparison is UFT vs. SFT at their respective best hyperparameters (Tables 1–2): UFT wins in 8 of 12 tasks and on both leaderboard averages, despite using an objective that is less direct than cross-entropy for the SFT task. The most natural explanation — the one the paper advances — is that the built-in KL regularisation prevents the SFT-like overfitting that degrades performance on out-of-distribution evaluation tasks like `bbh`, `gpqa`, and `musr`.

The Qwen 32B results (Tables 4–5) provide additional suggestive evidence. At larger model scales, pretrained models are more robust and less prone to overfitting during SFT, so the regularisation advantage of UFT should diminish. This is exactly what happens: Qwen+UFT (48.71) and Qwen+SFT (48.88) are nearly identical on the new leaderboard average, and Qwen+UFT's advantage on the old leaderboard (78.19 vs. 77.91) is smaller than Mistral's (64.25 vs. 63.17). The regularisation benefit matters most where overfitting risk is highest — smaller models with less pretrained knowledge to anchor them.

**This is a conceptual advance in understanding SFT stability, not a new regularisation technique.** KL regularisation of fine-tuning is not new. What's new is the recognition that a ratio-based implicit reward objective can *replace* SFT entirely and provide regularisation as a structural property rather than an added term — and that this structural regularisation is empirically sufficient to outperform explicit SFT on downstream benchmarks. The paper doesn't ablate this mechanism against SFT + explicit KL penalty (which would nail down causality), but the combination of the heuristic proof and the consistent empirical advantage across model sizes makes a compelling case that something about the ratio-based formulation is doing useful regularisation work.

---

### Innovation 3: The Sequential Degradation Problem as an Artifact of Stage Separation, Not Objective Incompatibility

Perhaps the paper's most actionable diagnostic contribution is the empirical demonstration that **performance degradation in sequential SFT+alignment pipelines is not an inevitable consequence of the alignment objective conflicting with the SFT objective — it is a consequence of applying them in sequence rather than jointly.** The evidence for this comes from the direct comparison in Tables 1–3: Mistral+UFT (trained jointly) consistently outperforms Mistral+SFT+DPO, Mistral+SFT+KTO, and Mistral+SFT+UNA (trained sequentially) on both leaderboards and on generation benchmarks.

**Why this is a meaningful distinction.** The "alignment tax" literature has generally framed the problem as one of **objective conflict** — alignment objectives (reward maximisation, preference satisfaction) pull the model away from the capabilities it acquired during SFT (instruction-following, factual accuracy), and the right response is to find better trade-offs between these competing demands. This framing motivates approaches like:
- Tuning the KL penalty coefficient to limit how far alignment can move the model.
- Designing better preference data that doesn't conflict with SFT capabilities.
- Using multi-objective RL to track helpfulness and harmlessness separately.
- Applying alignment to only a subset of model layers.

All of these approaches accept that alignment and SFT are in tension and try to manage that tension.

UFT suggests a different diagnosis: the degradation arises not because the objectives are inherently conflicting, but because **sequential training allows the optimisation dynamics of the second stage to overwrite the parameter configurations learned in the first stage, even when those configurations are compatible with both objectives.** This is a subtle but important distinction. Gradient-based optimisation is path-dependent — the order in which data is presented and objectives are applied determines which local minimum the model converges to. Even if there exists a parameter configuration that satisfies both the SFT and alignment objectives well, gradient descent applied sequentially may miss it because the alignment gradients, applied on top of an already-optimised SFT model, push the parameters in directions that destroy SFT capabilities without a corresponding mechanism to restore them.

Joint training on merged data changes the optimisation dynamics fundamentally. The model receives gradients from both SFT-like and alignment-like examples in each minibatch, so it never settles into a configuration that satisfies only one objective at the expense of the other — it must find a configuration that works for both simultaneously. This is the standard argument for multi-task learning, but its application to the SFT-alignment pipeline is novel because the field had accepted the sequential structure as necessary (DPO and KTO assume an SFT-trained starting point; RLHF uses the SFT model as both initialisation and reference).

**The evidence in the specific task improvements.** The most dramatic evidence for this diagnosis is in the `ifeval` and `truthful` columns of Table 1. For Mistral 7B:

| Method | ifeval | truthful |
|---|---|---|
| Mistral+SFT | 29.50 | 51.06 |
| Mistral+SFT+DPO | 26.64 | 47.83 |
| Mistral+SFT+KTO | 25.17 | 49.67 |
| Mistral+SFT+UNA | 26.82 | 49.54 |
| Mistral+UFT | **46.03** | **54.05** |

The sequential methods all *degrade* ifeval relative to SFT-only (from 29.50 down to 25–27), while UFT *improves* it dramatically (to 46.03, a 56% relative gain over SFT and a 72% gain over the best sequential method). This is not a case of finding a better trade-off — UFT simultaneously improves both instruction-following and truthfulness beyond what either SFT-only or sequential methods achieve. This strongly suggests the sequential methods were not facing an inherent conflict between instruction-following and alignment; they were simply failing to preserve instruction-following capability because the sequential training dynamics overwrote it.

The Qwen 32B results (Table 4) show the same pattern but attenuated: Qwen+UFT achieves 64.05 on ifeval vs. 47.48 for SFT-only and a range of 47.48–57.33 for sequential methods. The pattern holds, but the magnitude of UFT's advantage shrinks as model size increases, consistent with the idea that larger models are more resistant to sequential overwriting (their loss landscapes are flatter and their pretrained knowledge is more robustly encoded).

**This is a decisive empirical finding, not just a new method.** The paper demonstrates that joint training can reverse the degradation that sequential training causes, which implies that the degradation was never about the objectives being incompatible — it was about the optimisation procedure being suboptimal. This reframes the research question from "how do we trade off SFT capability against alignment safety?" to "how do we design training procedures that find parameter configurations satisfying both objectives simultaneously?" — a shift from a multi-objective optimisation framing to a data-mixing and training-dynamics framing.

---

### Innovation 4: Data Distribution as a Training-Time Knob for Capability Control — Not Just a Pretraining Concern

The data distribution experiment (Section 4.3, Tables 7–9) introduces a concept that is standard in pretraining but under-explored in post-training: **the ratio of different data types in the training mixture directly controls the model's capability profile, and this ratio can be tuned on held-out validation data to optimise for specific downstream metrics.** This is not a theoretical innovation — it's a **practical methodology** that the post-training literature has largely ignored in favour of sequential pipelines where each stage uses a fixed dataset.

**What the field did before.** In sequential pipelines, the data for each stage is determined by the stage's purpose: all SFT data in the SFT stage, all alignment data in the alignment stage. There is no concept of "mixing ratios" between these data types because they are never in the same training run. The only tuning knob for data composition is within-stage: how many SFT examples to use, what alignment dataset to use, whether to filter or augment the data. The interaction between SFT and alignment data quantities — whether more SFT data makes the alignment stage harder, or whether more alignment data erodes SFT capabilities more severely — is a property of the sequencing, not a tunable parameter.

In joint-training approaches like ORPO, the training data is pairwise preference data that implicitly contains both signals — the preferred responses serve as implicit SFT targets, while the preference comparison serves as the alignment signal. But the ratio of "SFT signal" to "alignment signal" is fixed by the dataset (each pair provides both), and there's no way to independently scale one relative to the other without changing the dataset itself.

**What UFT enables.** Because UFT treats all data as `(x, y, r)` triples in a unified format, the training set is just a collection of examples with different reward labels. The data mixture is controlled by how many examples of each type are included, and the ratio of instruction-tuning examples (reward = 1) to alignment examples (reward ∈ [0, 1]) becomes a **continuous hyperparameter** that can be swept. The paper demonstrates this by fixing the alignment data at 20k and varying the instruction-tuning data from 16k to 260k (Tables 7–9), effectively sweeping the instruction-to-alignment ratio from 0.8:1 to 13:1.

**The key findings from this sweep:**
- **Some tasks are ratio-insensitive.** `gpqa`, `musr`, `winograde`, and `arc` show small, statistically noisy variations across ratios — they are unaffected by whether the model sees more instruction or alignment data within the tested range.
- **Some tasks are ratio-sensitive in opposite directions.** `truthful` peaks at the lowest instruction data ratio (16k instruction + 20k alignment → 56.69 truthful) and declines monotonically as more instruction data is added (down to 50.17 at 260k instruction). This makes intuitive sense: `truthful` measures factual accuracy and resistance to generating falsehoods, and alignment data (which often contains rejections of incorrect statements and corrections of misconceptions) directly targets this. Diluting the alignment data with more instruction-following examples weakens this signal. Conversely, `ifeval` generally improves with more instruction data (44.21 at 16k → 46.76 at 32k, with some non-monotonicity at higher ratios), consistent with instruction-following being the primary skill taught by instruction-tuning examples.
- **There is a "sweet spot" for overall performance.** The 20k+20k and 32k+20k mixtures produce the best all-around results, with 32k giving a slight edge on `ifeval` and the new leaderboard average (32.52 vs. 32.81). Beyond 65k instruction examples, the degradation in `truthful` begins to dominate, pulling down average scores.

**Why this is a significant methodological contribution.** The ability to tune the instruction-to-alignment ratio means that UFT users can **target specific capability profiles** without changing the training algorithm. If deploying in a context where factuality is paramount (medical, legal), use more alignment data. If deploying in a context where instruction-following breadth matters more (creative writing, open-ended assistant), use more instruction data. This is what pretraining practitioners already do with domain mixtures (code vs. web text vs. scientific articles), and UFT brings the same capability to post-training.

Moreover, this finding implies that the "optimal" post-training data mixture is task-dependent and should be tuned on validation data representing the target deployment distribution — the same principle that guides pretraining data curation. The paper demonstrates this at a small scale (sweeping one dataset's proportion), but the principle generalises: any labelled feedback source can be added to the mix and its proportion tuned.

**This is an incremental innovation in methodology but a potentially large one in practice.** The concept of data mixing ratios is not new to machine learning, but its systematic application to the SFT/alignment interface — where the data types were previously treated as stage-separated rather than mixable — opens a new axis for post-training optimisation that the sequential paradigm obscured.

## 5. Experimental Analysis

### Evaluation Methodology

- **Dataset.** The paper uses **UltraChat** for instruction-tuning data and **HelpSteer2** for alignment data. UltraChat is a large-scale multi-turn dialogue dataset generated by ChatGPT; the paper unfolds conversations into individual (prompt, response) pairs and selects 20,000 samples for the main experiments (with additional subsets of 16k, 32k, 65k, 130k, and 260k for the data distribution study in Section 4.3). HelpSteer2 (Wang et al., 2024) provides 20,000 examples with prompts, responses, and scalar scores across multiple quality dimensions (helpfulness, correctness, coherence, complexity, verbosity); the paper uses all 20k for alignment experiments but does not specify how the multi-dimensional scores are aggregated into the single scalar reward `$r_\phi \in [0, 1]$` used in training. For UFT, the two datasets are merged to form a unified training set of (prompt, response, reward) triples, with SFT data receiving `$r_\phi = 1$` and alignment data retaining its original reward labels.

- **Base model(s).** Two model families are tested: **Mistral 7B-v0.1** (Jiang et al., 2023), a 7-billion-parameter dense transformer, and **Qwen 2.5 32B** (Qwen et al., 2025), a 32-billion-parameter model. The Mistral 7B experiments serve as the primary testbed because smaller models are more susceptible to the sequential degradation problem that UFT aims to solve; the Qwen 32B experiments test whether UFT's benefits persist at larger scales where degradation is naturally less severe. Both models are used in their pretrained base form as the starting point for all training, including the sequential SFT baselines — there is no intermediate SFT checkpoint anchoring.

- **Metrics.** All models are evaluated on a broad suite spanning 14 tasks across four evaluation frameworks. The **new HuggingFace Open LLM Leaderboard v2** (6 tasks: bbh, gpqa, mmlu-pro, musr, ifeval, math-hard) reports average scores computed by the leaderboard's standard evaluation harness. The **old HuggingFace Open LLM Leaderboard** (6 tasks: gsm8k, truthful, winograde, arc, hellaswag, mmlu) reports the standard per-task metrics used by that leaderboard: exact-match accuracy for gsm8k, mc2 score for truthful, accuracy for winograde, acc-norm for arc and hellaswag, and accuracy for mmlu. **MT-Bench** (Zheng et al., 2023) evaluates multi-turn dialogue quality through GPT-4 judgment, producing a scalar score on a 1–10 scale. **Alpaca-eval** (Li et al., 2023) reports Length-Controlled Win Rate (LC WR) against reference responses, along with average output length for comprehensive analysis. All evaluation uses the standard, off-the-shelf leaderboard infrastructure with no task-specific prompting or fine-tuning.

- **Baselines.** The paper compares against six baselines organized into two groups:
  - **SFT-only baselines:** `Mistral+SFT` and `Qwen+SFT`, trained exclusively on the 20k UltraChat instruction-tuning examples using standard cross-entropy loss (Equation 1) at the best-found learning rate of `$1 \times 10^{-4}$`.
  - **Sequential SFT+Alignment baselines:** `Mistral+SFT+DPO` (Rafailov et al., 2023), `Mistral+SFT+KTO` (Ethayarajh et al., 2024), and `Mistral+SFT+UNA` (Wang et al., 2024), each constructed by taking the best SFT model and further fine-tuning it on the 20k HelpSteer2 examples using the respective alignment method. DPO uses pairwise preference structure from HelpSteer2; KTO handles binary feedback by estimating the partition function; UNA regresses the implicit reward against explicit reward scores. The same baselines are constructed for Qwen 32B. Hyperparameters for these alignment stages are not reported, which is a notable omission — suboptimal tuning could exaggerate UFT's apparent advantage.
  - **Base model (no fine-tuning):** The pretrained `Mistral` and `Qwen` checkpoints evaluated directly, providing a lower bound on what fine-tuning contributes.

- **Generation budget / compute accounting.** The paper measures compute indirectly through dataset sizes and training configurations rather than through generation budgets or FLOP counts. All SFT and UFT experiments use comparable dataset sizes (20k instruction examples for the SFT comparison; 40k total examples — 20k instruction + 20k alignment — for the sequential vs. unified comparison). Training uses LoRA with rank `$r = 16$` across all methods, making the per-step compute roughly comparable. The data distribution experiments (Section 4.3) vary the instruction dataset size from 16k to 260k while keeping alignment data fixed at 20k, making total training cost proportional to dataset size. The paper does not report wall-clock training time, GPU hours, or total FLOPs, so compute comparisons are approximate. Notably, the difficulty estimation cost that burdens many test-time compute papers is absent here — UFT requires no additional inference-time computation beyond standard generation.

- **Cross-validation / statistical protocol.** The paper does not employ explicit cross-validation or report confidence intervals. Strategy selection — choosing the best learning rate for SFT and the best (learning rate, β) combination for UFT — is based on computing per-task scores on the full test sets of both leaderboards and selecting the configuration that maximises average performance. This means the test sets are used for both hyperparameter selection and final evaluation, introducing potential overfitting to the specific benchmarks. The paper does not discuss this risk or employ held-out validation sets. For the data distribution experiments (Section 4.3), the six different instruction-to-alignment ratios are compared directly on the same evaluation sets, again without cross-validation. The absence of statistical testing means that differences between methods — particularly the small gaps in some tasks — cannot be assessed for significance, and the reported averages may be sensitive to the specific 500–1000 questions per benchmark. The paper labels some differences as "statistically significant" and others as "statistically small" in the text (Section 4.3) without defining these terms or providing p-values, confidence intervals, or standard errors.

---

### Main Quantitative Results

#### UFT vs. SFT on Instruction-Tuning Data Alone

**Headline finding:** When trained exclusively on 20k UltraChat instruction-tuning examples, UFT outperforms standard SFT on aggregate metrics for both Mistral 7B and Qwen 32B, though the advantage is modest and task-dependent.

For Mistral 7B, comparing `Mistral+UFT` against `Mistral+SFT` across Tables 1–3:

- **New leaderboard average (Table 1):** UFT achieves 30.09 vs. SFT's 29.87 — a 0.22 percentage-point advantage. UFT wins on 4 of 6 tasks (bbh: 46.55 vs. 46.04, gpqa: 29.24 vs. 28.72, mmlu-pro: 30.25 vs. 29.35, math-hard: 3.87 vs. 2.66), loses on 2 (musr: 41.73 vs. 42.94, ifeval: 28.89 vs. 29.50). The gaps are small across all tasks — the largest absolute difference is 1.21 points on math-hard — making it unclear whether these differences are statistically meaningful given the absence of error bars.
- **Old leaderboard average (Table 2):** UFT achieves 64.25 vs. SFT's 63.17 — a 1.08 point advantage. UFT leads on 5 of 6 tasks (gsm8k: 45.57 vs. 39.65, truthful: 51.18 vs. 51.06, winograde: 78.93 vs. 78.53, mmlu: 62.44 vs. 61.99), trails by a fraction on arc (63.82 vs. 63.99) and hellaswag (83.54 vs. 83.78). The gsm8k improvement (+5.92 points, nearly 15% relative gain over SFT) is the largest single-task difference in this comparison.
- **Generation benchmarks (Table 3):** UFT achieves MT-Bench 6.55 vs. SFT's 6.33 (+0.22), but trails on Alpaca-eval LC WR (7.27 vs. 8.07, a 0.80 point disadvantage). UFT's average generation length is slightly higher (974 vs. 908 tokens), which may partially explain the MT-Bench gain but not the Alpaca-eval loss (since LC WR controls for length).

For Qwen 32B, comparing `Qwen+UFT` against `Qwen+SFT` across Tables 4–6:

- **New leaderboard average (Table 4):** UFT achieves 48.71 vs. SFT's 48.88 — a 0.17 point disadvantage, unlike Mistral where UFT led. Qwen+UFT trails on 4 of 6 tasks (bbh, mmlu-pro, musr, ifeval) and leads on only 2 (gpqa tied at 40.42, math-hard: 31.23 vs. 30.77). This is a weaker showing than Mistral and consistent with the paper's stated observation that "the performance degradation caused by SFT and alignment is diminished with larger model size" — UFT's regularisation advantage matters less when SFT alone is already stable.
- **Old leaderboard average (Table 5):** UFT achieves 78.19 vs. SFT's 77.91 — a narrow 0.28 point advantage. UFT leads on 4 of 6 tasks, but the gaps are consistently small (gsm8k: 88.29 vs. 87.38, truthful: 58.96 vs. 58.81, winograde: 82.64 vs. 82.00, mmlu: 83.34 vs. 83.39), with no difference exceeding 1.5 points.
- **Generation benchmarks (Table 6):** UFT achieves MT-Bench 7.95 vs. SFT's 7.85 and Alpaca-eval 9.96 vs. 8.34, winning on both. Qwen+UFT's generation length (1292 tokens) is substantially shorter than Qwen+SFT's (2800 tokens), meaning the Alpaca-eval LC WR advantage is not attributable to length inflation — UFT generates more concise outputs that are nonetheless preferred.

**Interpretation:** UFT's advantage over SFT on instruction-tuning data alone is real but modest, and it shrinks with model size. For the 7B model, UFT provides a consistent 1–2% relative improvement on aggregate benchmarks, with the largest gains on math reasoning (gsm8k, math-hard). For the 32B model, UFT and SFT are essentially tied on knowledge/reasoning benchmarks, with UFT showing a slight edge only on generation quality. The paper attributes UFT's advantage to the implicit KL regularisation preventing overfitting (Section 2.3), and the size-dependent results support this: smaller models benefit more from regularisation because they overfit more readily.

---

#### UFT vs. Sequential SFT+Alignment on Combined Data

**Headline finding:** When trained on both instruction-tuning (20k UltraChat) and alignment (20k HelpSteer2) data, UFT substantially outperforms all three sequential methods (SFT+DPO, SFT+KTO, SFT+UNA), with the largest and most significant gains concentrated on ifeval (instruction-following) and truthful (factuality).

For Mistral 7B across Tables 1–3:

**Table 1 (new leaderboard):**
- Mistral+UFT average: **32.81**
- Best sequential method (SFT+UNA): 29.16
- SFT+DPO: 29.09, SFT+KTO: 28.85
- UFT outperforms the best sequential method by 3.65 points (12.5% relative improvement over SFT+UNA).
- **ifeval:** UFT achieves **46.03** vs. 26.82 (SFT+UNA), 26.64 (SFT+DPO), 25.17 (SFT+KTO). This is a 19.21-point (71.6%) improvement over SFT+UNA. ifeval measures the model's ability to follow specific formatting and content instructions in its responses — a direct test of instruction-following capability. All sequential methods perform *worse* than SFT-only (29.50) on this metric, confirming the degradation problem. UFT not only prevents this degradation but more than doubles the SFT-only score.
- **bbh:** UFT achieves **45.46** vs. 43.74 (SFT+UNA), 44.52 (SFT+DPO), 42.89 (SFT+KTO).
- **gpqa:** UFT achieves **31.15** vs. 30.78 (SFT+UNA), 29.98 (SFT+DPO), 31.00 (SFT+KTO). Small advantage.
- On `mmlu-pro` and `math-hard`, UFT is roughly comparable to sequential methods (30.05 vs. 29.95–30.48; 3.13 vs. 2.94–3.13). On `musr`, UFT (41.06) slightly edges the sequential methods (40.31–40.59).

**Table 2 (old leaderboard):**
- Mistral+UFT average: **64.34**
- Best sequential method (SFT+KTO): 63.23
- SFT+DPO: 62.84, SFT+UNA: 63.02
- UFT outperforms the best sequential method by 1.11 points.
- **truthful:** UFT achieves **54.05** vs. 47.83 (SFT+DPO), 49.67 (SFT+KTO), 49.54 (SFT+UNA). This is a 4.38-point (8.8%) improvement over SFT+KTO. truthful measures the model's tendency to reproduce common misconceptions vs. provide accurate information — an alignment-relevant metric. All sequential methods degrade relative to SFT-only (51.06); UFT improves by 3 points over SFT-only. This is the cleanest demonstration of the paper's central claim: sequential alignment hurts factuality, joint training improves it.
- **gsm8k:** UFT (41.59) is comparable to SFT+DPO (42.19) and SFT+KTO (42.57), and slightly ahead of SFT+UNA (39.99).
- On `winograde`, `arc`, `hellaswag`, and `mmlu`, differences are small (≤1 point) with no clear winner.

**Table 3 (generation benchmarks):**
- Mistral+UFT achieves MT-Bench **6.78** vs. 4.81 (SFT+DPO), 4.76 (SFT+KTO), 5.24 (SFT+UNA). UFT's score is substantially higher — a 1.54-point (29.4%) advantage over SFT+UNA.
- Mistral+UFT achieves Alpaca-eval LC WR **8.28** vs. 1.05 (SFT+DPO), 0.64 (SFT+KTO), 1.34 (SFT+UNA). This is a 6.94-point (518%) advantage over SFT+UNA — an enormous gap.
- Notably, the sequential methods all produce much longer outputs than UFT (4945–6215 tokens vs. 1317 for UFT), yet achieve *lower* length-controlled win rates, meaning their outputs are not only longer but also lower quality. The paper notes this explicitly: "Mistral+UFT does not bias towards long generation like the other three sequential methods, which is another advantage of UFT."

For Qwen 32B across Tables 4–6:

**Table 4 (new leaderboard):**
- Qwen+UFT average: **52.39**
- Best sequential method (SFT+UNA): 51.13
- SFT+DPO: 49.24, SFT+KTO: 50.27
- UFT outperforms SFT+UNA by 1.26 points (2.5% relative improvement). The gap is smaller than with Mistral (3.65 points), confirming that degradation diminishes with model size.
- **ifeval:** UFT achieves **64.05** vs. 57.33 (SFT+UNA), 47.48 (SFT+DPO), 53.03 (SFT+KTO). The 6.72-point advantage over SFT+UNA is substantial but proportionally smaller than Mistral's 19.21-point gap.
- **math-hard:** UFT achieves **37.80** vs. 34.70 (SFT+UNA), 33.67 (SFT+DPO), 35.07 (SFT+KTO). This is a 3.10-point (8.9%) advantage — notable because math reasoning showed smaller UFT benefits with Mistral.
- Other tasks show small, mixed differences.

**Table 5 (old leaderboard):**
- Qwen+UFT average: **80.29**
- Best sequential method (SFT+UNA): 79.58
- UFT leads by 0.71 points.
- **truthful:** UFT achieves **66.70** vs. 58.54 (SFT+DPO), 59.77 (SFT+KTO), 62.74 (SFT+UNA). A 3.96-point (6.3%) advantage over SFT+UNA.
- **gsm8k:** UFT (90.68) trails SFT+UNA (92.08) by 1.4 points — the only task where UFT loses to a sequential method by a non-trivial margin. This is a notable exception to the general pattern of UFT superiority.
- Other tasks show differences ≤0.5 points.

**Table 6 (generation benchmarks):**
- Qwen+UFT achieves MT-Bench **8.67** vs. 8.64 (SFT+DPO), 8.58 (SFT+KTO), 8.57 (SFT+UNA). The gaps are negligible — essentially tied.
- Qwen+UFT achieves Alpaca-eval LC WR **13.79** vs. 13.75 (SFT+UNA), 9.83 (SFT+DPO), 10.17 (SFT+KTO). UFT and SFT+UNA are tied; both substantially outperform DPO and KTO.
- Qwen+UFT's generation length (1307 tokens) is the lowest among all methods, shorter than SFT+UNA (1328), KTO (2149), and DPO (2664). Despite shorter outputs, its win rate is competitive, indicating higher per-token quality.

**Synthesis of sequential comparison:** The pattern is remarkably consistent. For the smaller model (Mistral 7B), UFT provides large, unambiguous gains: +3.65 points on the new leaderboard average, +1.11 on the old, dramatic improvements on ifeval (+19.21 over best sequential) and truthful (+4.38 over best sequential), and massive generation quality improvements. All three sequential methods degrade relative to SFT-only on ifeval and truthful — the alignment tax is clearly visible. UFT not only avoids this tax but produces *gains* beyond SFT-only. For the larger model (Qwen 32B), UFT still leads but the margins shrink: +1.26 on the new leaderboard, +0.71 on the old, +6.72 on ifeval, +3.96 on truthful. The sequential methods still degrade on ifeval and truthful relative to SFT-only (except SFT+UNA on ifeval, which improves slightly), but the degradation is less severe. This size-dependent pattern is internally consistent with the paper's theory: larger models are more robust to sequential overwriting, so UFT's joint-training advantage is proportionally smaller.

The two standout results — ifeval and truthful — deserve special emphasis because they directly measure the capabilities most vulnerable to sequential degradation. On Mistral 7B, UFT's ifeval score of 46.03 is not merely better than sequential methods; it is in a different performance class entirely — nearly double the SFT-only score and 2.8× higher than SFT+DPO. The truthful score of 54.05, while less dramatic proportionally, represents a meaningful improvement in factual reliability that sequential methods actively undermine. These two metrics capture the paper's core value proposition: UFT simultaneously improves instruction-following and factuality, two capabilities that sequential pipelines force into a zero-sum tradeoff.

---

#### Data Distribution: Impact of Instruction-to-Alignment Ratio

**Headline finding:** The ratio of instruction-tuning to alignment data in the UFT training mixture has task-dependent effects, with truthful declining as instruction data increases and ifeval showing a non-monotonic response with a sweet spot around 20k–32k instruction examples. No single ratio is optimal across all tasks.

The experiment (Section 4.3, Tables 7–9) fixes the alignment data at 20k HelpSteer2 examples and varies the UltraChat instruction data across {16k, 20k, 32k, 65k, 130k, 260k}. For clarity, the configurations can be expressed as instruction:alignment ratios of 0.8:1, 1:1, 1.6:1, 3.25:1, 6.5:1, and 13:1, respectively. All training uses UNA (the UFT loss function) with consistent hyperparameters, though the paper does not specify which (learning rate, β) combination from Appendix B was used — presumably the best configuration identified earlier (lr = 3×10⁻⁵, β = 0.01).

**Table 7 (new leaderboard):**

- **Average score:** Peaks at 32.81 for 20k instruction (1:1 ratio) and 32.52 for 32k (1.6:1), then declines to 31.28 at 65k (3.25:1) and 31.66 at 130k (6.5:1), recovering slightly to 32.39 at 260k (13:1). The relationship is non-monotonic but generally declines after the 32k peak.
- **ifeval:** 44.21 at 16k → 46.03 at 20k → 46.76 at 32k (peak) → 41.91 at 65k → 44.20 at 130k → 45.80 at 260k. The non-monotonic pattern is puzzling: performance drops sharply at 65k (losing 4.85 points from the 32k peak) then partially recovers at higher instruction data volumes. The paper does not explain this fluctuation. The overall range (41.91–46.76) spans 4.85 points — substantial on this metric where base Mistral scores 23.22 — indicating ifeval is genuinely sensitive to data ratio.
- **musr:** Shows a clear downward trend as instruction data increases: 39.19 (16k) → 41.06 (20k) → 39.06 (32k) → 36.80 (65k) → 38.66 (130k) → 39.85 (260k). The 65k configuration loses 4.26 points from the 20k peak. musr measures multi-step reasoning, and the degradation at high instruction data ratios suggests that alignment data contributes reasoning capability that instruction-following data does not fully replicate.
- **math-hard:** Shows small variations (2.38–3.51) with no clear trend.
- **bbh, gpqa, mmlu-pro:** Roughly flat across all ratios (bbh: 44.31–45.46, gpqa: 29.75–31.38, mmlu-pro: 29.66–30.10), suggesting these knowledge-intensive benchmarks are insensitive to the instruction-to-alignment ratio within the tested range.

**Table 8 (old leaderboard):**

- **Average score:** Peaks at 64.78 for 16k instruction (0.8:1 ratio), declines to 64.45 at 32k, then further to 63.62 at 130k — a generally downward trend as instruction data increases, driven primarily by truthful.
- **truthful:** **56.69** at 16k → 54.05 at 20k → 54.38 at 32k → 52.35 at 65k → 50.17 at 130k → 50.17 at 260k. This is a monotonic decline of 6.52 points (11.5% relative) from the lowest to highest instruction data ratio. The paper interprets this as evidence that "a larger proportion of alignment data benefits the bias and ethical performance of the LLM." truthful is the metric most directly tied to alignment — it measures whether the model reproduces common misconceptions — so this sensitivity is expected: diluting alignment-specific data reduces the model's ability to distinguish truth from plausible falsehood.
- **gsm8k:** Relatively flat (40.83–41.59). No ratio sensitivity.
- **winograde, arc, hellaswag, mmlu:** Differences ≤1.5 points with no clear trend. The one exception is arc at 65k (65.27), which is 1.45 points above 20k (63.82) — possibly noise.

**Table 9 (generation benchmarks):**

- **MT-Bench:** 6.25 at 16k → 6.78 at 20k (peak) → 6.54 at 32k → 6.30 at 65k → 6.83 at 130k (second peak) → 6.45 at 260k. The pattern is noisy — no monotonic trend, though 20k and 130k are the two highest points. The range (6.25–6.83) spans 0.58 points.
- **Alpaca-eval LC WR:** Ranges from 6.85 (260k) to 9.92 (65k), with 8.28 at 20k. No clear relationship to ratio. The 65k spike at 9.92 is unexplained — it comes alongside the worst musr score (Table 7) and a mediocre truthful (52.35), suggesting a tradeoff where high Alpaca-eval performance may come at the expense of reasoning capability.
- **Average generation length:** Consistently in the 1317–1378 token range across all ratios — surprisingly stable. Unlike the sequential methods (Table 3) which produced 4945–6215 tokens, UFT's output length is ratio-invariant and substantially shorter, confirming that UFT does not learn a length-inflation strategy regardless of data mixture.

**Synthesis of data distribution experiment:** The most instructive finding is the opposite-direction sensitivity of ifeval and truthful. truthful shows a clean monotonic relationship: more alignment data → higher truthful. ifeval shows a messy non-monotonic relationship with a peak at 32k instruction data and a puzzling dip at 65k. This means there is no universally optimal ratio — the choice depends on whether the target application prioritises factuality (favour more alignment data, e.g., 16k instruction) or instruction-following breadth (favour a balanced mix, e.g., 20k–32k instruction). For general-purpose deployment, the 20k+20k (1:1) and 32k+20k (1.6:1) configurations represent the best compromise, performing well on both sensitive metrics without severely sacrificing either.

The paper notes that "simply adding more instruction-tuning data does not improve the performance of ifeval, MT-Bench, and Alpaca-eval," which is an important negative result: beyond a certain point, additional SFT examples provide diminishing or negative returns under UFT. This aligns with the broader observation in the fine-tuning literature that data quality and diversity matter more than quantity for instruction-following, and that simply scaling up SFT data can lead to overfitting or dilution of alignment signals. The UFT framework makes this tradeoff explicit and tunable.

The non-monotonicity in ifeval (dip at 65k, recovery at 130k and 260k) is concerning. The paper does not explain this pattern, and without error bars or multiple seeds, it is impossible to determine whether it reflects a genuine underlying phenomenon or random variation. It could be a real effect — perhaps certain instruction-following skills are temporarily suppressed as the model adjusts its internal representation to accommodate a large volume of new instruction data before eventually recovering as it learns to organise this information — but the paper presents no evidence for such a mechanism. This fluctuation is a reminder that all results in this paper are single-point estimates without statistical characterisation.

---

### Ablation Studies and Robustness Checks

**Ablation of SFT learning rate (Appendix A, Tables 10–11):** Five learning rates are tested for SFT on Mistral 7B: 3×10⁻⁶, 1×10⁻⁵, 3×10⁻⁵, 1×10⁻⁴, and 3×10⁻⁴. Performance on the new leaderboard (Table 10) is relatively flat from 3×10⁻⁶ (avg 29.45) through 1×10⁻⁴ (avg 29.87), with 3×10⁻⁴ collapsing to 26.98 — a classical learning rate sensitivity pattern where the model tolerates a factor-of-30 range but fails catastrophically at the highest tested rate. On the old leaderboard (Table 11), the pattern is similar: 3×10⁻⁵ achieves the best average (63.72), 1×10⁻⁴ is slightly lower (63.17), and 3×10⁻⁴ collapses to 55.18. The paper selects 1×10⁻⁴ for the main comparison based on new leaderboard performance. This choice is consequential because UFT beats SFT by only 0.22 points on the new leaderboard average (30.09 vs. 29.87); if SFT had been tuned against the old leaderboard instead and 3×10⁻⁵ used, the gap would shift. This kind of hyperparameter sensitivity — where the "best" configuration depends on which benchmark is used for selection — is not discussed.

**Ablation of UFT learning rate and β (Appendix B, Tables 12–13):** A 4×4 grid of learning rates (3×10⁻⁶, 1×10⁻⁵, 3×10⁻⁵, 1×10⁻⁴) and β values (0.01, 0.03, 0.1, 0.3) is tested. Key findings:
- **Optimal configuration:** lr = 3×10⁻⁵, β = 0.01 achieves the best or near-best averages on both leaderboards (30.09 new, 64.25 old).
- **β sensitivity:** At the best learning rate (3×10⁻⁵), increasing β from 0.01 to 0.3 reduces old leaderboard average from 64.25 to 61.35 (-2.90 points) and new leaderboard average from 30.09 to 29.45 (-0.64 points). Stronger regularisation consistently hurts — the model needs sufficient freedom (low β) to adapt to the instruction-following task. This is the opposite of what one might expect if the KL regularisation were the sole source of UFT's benefit; it suggests that the regularisation must be light enough to allow substantial policy change, and the structural properties of the ratio-based objective (not the regularisation strength per se) provide the advantage.
- **Learning rate sensitivity:** At the best β (0.01), increasing learning rate from 3×10⁻⁵ to 1×10⁻⁴ reduces old leaderboard average from 64.25 to 63.71 (-0.54 points) and new leaderboard from 30.09 to 30.08 (flat). The model is somewhat robust to learning rate at this β.
- **Interaction effects:** The worst configuration is not simply the highest β or highest learning rate. At lr = 1×10⁻⁴, β = 0.3, the old leaderboard average is 61.07 — the lowest in the grid. High learning rate + strong regularisation is particularly damaging, likely because the model takes large steps toward a reference distribution it is heavily penalised for leaving, creating oscillatory training dynamics.
- The paper presents these sweeps as tables but does not plot learning curves, making it impossible to assess whether configurations converged or whether some combinations were under-trained relative to others. If the number of training steps was held constant across configurations rather than tuned per-setting, the sweeps conflate learning rate and effective training duration.

**Model size robustness (Tables 1–6, implicit ablation):** The paper tests UFT on two model sizes (7B, 32B) from different families, which serves as a robustness check on the claim that UFT prevents sequential degradation. As discussed in the main results, UFT's advantage is larger on the 7B model — where sequential degradation is most severe — and smaller on the 32B model, where degradation is naturally attenuated. This pattern is internally consistent and strengthens the claim: UFT matters most where the problem it solves is worst. However, testing only two model sizes from two families leaves open the question of whether the effect generalises to other architectures (e.g., non-dense models like MoE), other scales (e.g., 1B, 70B), or other pretraining data distributions.

**Negative result: alignment tax in sequential methods (Tables 1–3, 4–6):** The consistent underperformance of SFT+DPO, SFT+KTO, and SFT+UNA relative to SFT-only on ifeval and truthful (Tables 1–2 for Mistral, Tables 4–5 for Qwen) serves as a validation that the performance degradation problem is real and not an artifact of the specific alignment hyperparameters or datasets used here. For Mistral, SFT+UNA loses 2.68 points on ifeval (29.50 → 26.82) and 1.52 points on truthful (51.06 → 49.54) relative to SFT-only; for Qwen, the pattern holds but is smaller in magnitude. This confirms that the paper is addressing a genuine phenomenon, not a straw-man.

**Length analysis (Tables 3, 6, 9):** The paper tracks average generation length across all configurations. The sequential alignment methods (DPO, KTO, UNA) all produce dramatically longer outputs than UFT — for Mistral, 4945–6215 tokens vs. 1317 for UFT (Table 3) — yet achieve *lower* length-controlled win rates. This is an implicit ablation confirming that UFT's generation quality advantage is not a length artifact. UFT produces shorter, higher-quality outputs, while sequential methods learn to inflate length without commensurate quality improvement — a known failure mode of RL-based alignment where the model learns that longer responses tend to score higher under reward models, independent of content quality. The paper does not dig into why UFT avoids this — it may be because joint training with high-quality SFT examples (which are typically concise, expert-written responses) anchors the model's output length distribution, or because the Sigmoid-MSE objective saturates differently than the DPO/KTO loss functions, reducing the marginal benefit of length inflation.

---

### Critical Assessment

**Claim 1: UFT outperforms SFT on instruction-tuning data alone.**

What the experiments demonstrate: On Mistral 7B, UFT achieves a 0.22-point higher average on the new leaderboard and a 1.08-point higher average on the old leaderboard (Tables 1–2). On Qwen 32B, UFT trails by 0.17 points on the new leaderboard and leads by 0.28 points on the old (Tables 4–5). These differences are extremely small — at or below what would typically be considered noise in leaderboard evaluations, especially given the absence of error bars, multiple seeds, or statistical tests. The largest single-task gain is gsm8k on Mistral (+5.92 points, Table 2), which is substantial, but it is offset by losses on other tasks (e.g., ifeval loses 0.61 points, arc loses 0.17 points).

The paper's theoretical argument — that UFT's ratio-based objective provides implicit KL regularisation that SFT lacks — is plausible and internally consistent with the β-sensitivity results (higher β hurts, Tables 12–13), but the paper provides no direct evidence that regularisation is the causal mechanism. A critical missing experiment is SFT with an explicit KL penalty added (`SFT + λ·D_KL(π_θ || π_ref)`), which would test whether the ratio-based formulation provides benefits beyond what a simple additive regularisation can achieve. Without this, the claim that UFT "outperforms SFT by minimizing divergence from the pretrained model" (Section 2.3) is an interpretation, not a demonstrated fact.

Additionally, the SFT hyperparameter sweep (Tables 10–11) shows that SFT performance is sensitive to learning rate — the difference between the best (29.87 at lr = 1×10⁻⁴) and the second-best (29.75 at lr = 3×10⁻⁵) on the new leaderboard is only 0.12 points. If UFT had been compared against the second-best SFT configuration instead, the gap would be 0.34 points (30.09 vs. 29.75) — still small. The claim that UFT outperforms SFT is marginal and would benefit from replication across multiple seeds to establish robustness.

**Verdict:** The claim is **weakly supported** — the direction is correct but the magnitude is tiny, and the experiments lack the statistical rigour to distinguish a real effect from noise. The claim would be stronger with multi-seed experiments reporting means and standard deviations, a direct ablation against SFT+KL, and testing at a wider range of model scales.

**Claim 2: UFT prevents the performance degradation caused by sequential SFT+alignment.**

What the experiments demonstrate: On Mistral 7B, UFT dramatically outperforms all three sequential methods on ifeval (46.03 vs. 25.17–26.82) and truthful (54.05 vs. 47.83–49.67) — the two tasks where degradation is most visible (Tables 1–2). It also outperforms on aggregate benchmarks (32.81 vs. 28.85–29.16 on the new leaderboard, 64.34 vs. 62.84–63.23 on the old) and on generation quality (MT-Bench 6.78 vs. 4.76–5.24, Alpaca-eval 8.28 vs. 0.64–1.34, Table 3). On Qwen 32B, the same pattern holds but with reduced magnitude (ifeval: 64.05 vs. 47.48–57.33; truthful: 66.70 vs. 58.54–62.74, Tables 4–5).

This is the paper's strongest claim, and the evidence is compelling for the specific models and datasets tested. The performance gap on ifeval and truthful is large enough to be practically meaningful even without formal significance testing — a 19-point ifeval improvement and a 4-point truthful improvement on Mistral represent capability differences that users would notice in deployment.

However, several caveats weaken the generality of this claim:
- **Alignment hyperparameters are not reported or tuned.** The sequential methods (SFT+DPO, SFT+KTO, SFT+UNA) use fixed hyperparameters that are not swept. If the alignment stage were extensively tuned — as UFT's hyperparameters were (Appendix B, 16 configurations) — the gap might narrow. The paper effectively compares a carefully tuned UFT against untuned sequential baselines. A fairer comparison would sweep DPO's β, KTO's hyperparameters, and UNA's β and learning rate with the same thoroughness as UFT's sweep.
- **The sequential methods use the same 20k alignment data.** In practice, sequential alignment often uses larger or different alignment datasets than SFT data. The 20k HelpSteer2 examples might be insufficient for DPO/KTO/UNA to perform well — these methods may benefit from more preference data. UFT has the advantage of also seeing the 20k instruction examples during its "alignment" phase.
- **Only one alignment dataset is tested.** HelpSteer2 provides scalar scores, which map naturally to UFT's reward regression framework. It is unclear whether UFT's advantage would persist with purely pairwise datasets (e.g., Anthropic HH-RLHF) or binary feedback datasets where absolute scores are unavailable.
- **The degradation problem diminishes at larger scales.** The Qwen 32B results show that the sequential methods are more competitive, particularly SFT+UNA, which nearly matches UFT on aggregate (51.13 vs. 52.39, Table 4) and ties on generation quality (Alpaca-eval 13.75 vs. 13.79, Table 6). For very large models (70B+), the advantage of UFT over sequential training might disappear entirely.

**Verdict:** The claim is **well-supported for the tested regime** (7B–32B models, UltraChat+HelpSteer2, tasks where degradation is severe) but the generality to other datasets, scales, and alignment methods is unproven. The hyperparameter imbalance between UFT (swept) and sequential baselines (unswept) is a significant fairness concern.

**Claim 3: UFT establishes an effective and efficient unified post-training framework.**

What the experiments demonstrate: UFT successfully trains a model on merged instruction+alignment data in a single stage, producing performance that dominates sequential pipelines. The framework is flexible — the data distribution experiment (Tables 7–9) shows that the instruction-to-alignment ratio can be tuned to prioritise different capabilities, and the method accommodates both instruction data (r = 1) and alignment data (r ∈ [0, 1]) in the same loss function.

The claim of "efficiency" warrants scrutiny. UFT replaces two training stages (SFT + alignment) with one — this is conceptually simpler and requires fewer hyperparameter decisions. But the total training data is the same (20k + 20k = 40k examples), and the per-step compute is comparable (same LoRA rank, same model architecture). UFT does not reduce total FLOPs; it re-organises them into a single stage. The efficiency gain is engineering simplicity, not compute savings. The paper does not claim otherwise in the main text, but the abstract's phrasing — "an effective and efficient paradigm" — could mislead readers into expecting computational efficiency improvements.

The claim of a "unified" framework is accurate in the specific sense that UFT provides a single loss function (UNA's reward regression) for all post-training data. However, the paper only tests two data types — instruction demonstrations and scored alignment data — from two datasets. A truly unified framework would handle the full diversity of post-training data: multi-turn dialogues, tool-use trajectories, code execution feedback, human preference comparisons, AI feedback ratings, and safety red-teaming corrections. The paper does not test whether UFT's reward-regression approach scales to this variety, or whether assigning r = 1 to all SFT data is appropriate when SFT data contains varying quality levels.

**Verdict:** The claim is **supported in its narrow formulation** (UFT unifies SFT and the specific alignment methods tested) but the paper's framing — "a unified fine-tuning framework that runs parallel to the pretraining phase" (Section 2.4) — implies a generality that the experiments do not establish. Testing on a wider range of data types, quality levels, and task categories would be needed to justify the "unified post-training framework" label.

**Claim 4: The data distribution between instruction-tuning and alignment data impacts performance.**

What the experiments demonstrate: The ratio sweep (Tables 7–9) convincingly shows that truthful degrades monotonically as the proportion of instruction data increases (56.69 at 16k → 50.17 at 260k), while ifeval shows a non-monotonic pattern with a peak at 32k (46.76) and fluctuations at higher ratios. Other tasks are largely ratio-insensitive. This directly supports the claim that data distribution matters and can be tuned.

The experiment is limited to varying instruction data quantity while holding alignment data fixed. This sweeps only one degree of freedom in a two-dimensional space. The complementary experiment — varying alignment data while holding instruction data fixed — is not performed. It is possible that the observed effects are driven by *total dataset size* rather than the ratio per se — at 260k instruction examples, the total dataset is 14× larger than at 20k, and the model may be overfitting to the instruction data not because of the ratio but because of the absolute volume. An experiment that held total dataset size constant while varying the ratio (e.g., 20k instruction + 20k alignment vs. 30k instruction + 10k alignment vs. 10k instruction + 30k alignment) would disentangle ratio effects from scale effects. The paper does not do this.

Additionally, the non-monotonic ifeval pattern — a 4.85-point drop from 32k to 65k followed by partial recovery — is unexplained and undermines confidence in the ratio-tuning methodology. If the relationship between data ratio and task performance is not smooth or predictable, then tuning on a validation set (as the paper implicitly advocates) becomes unreliable unless the validation set is large and representative.

**Verdict:** The claim is **supported with significant caveats.** Data distribution clearly matters, but the paper only sweeps one variable (instruction data quantity) and the non-monotonicity in ifeval is unexplained, limiting the practical utility of the ratio-tuning approach.

**Missing experiments that would have strengthened the paper:**

1. **Multi-seed experiments with error bars.** Every table reports single-point estimates without any measure of variance. Given the small performance gaps in many comparisons (UFT vs. SFT, Qwen+UFT vs. Qwen+SFT+UNA), knowing whether these differences are within or outside the range of random seed variation is essential for interpreting the results.

2. **SFT + explicit KL penalty baseline.** The paper's theoretical argument is that UFT's ratio-based formulation provides beneficial KL regularisation. Comparing UFT against SFT with an explicit KL term added (at various λ values) would test whether the ratio-based mechanism is necessary or whether any KL-regularised SFT would achieve similar results. This is the most direct ablation of the paper's core mechanistic claim.

3. **Full fine-tuning experiments.** All results use LoRA with r = 16. LoRA constrains the model's parameter updates to low-rank subspaces, which may interact with the ratio-based objective in ways that full-weight training would not — for example, LoRA might naturally regularise the model and reduce the advantage UFT gains from KL regularisation, or it might restrict the model's ability to simultaneously satisfy instruction and alignment objectives. Full fine-tuning experiments would establish whether UFT's benefits are specific to parameter-efficient training or generalise to the full-weight regime used in production LLM training.

4. **Tuned sequential baselines.** Sweep DPO's β, KTO's hyperparameters, and UNA's β and learning rate for the alignment stage with the same granularity as UFT's sweep. The current comparison between a 16-configuration-swept UFT and unswept sequential methods is biased in UFT's favour.

5. **Ratio sweep at constant total dataset size.** Vary instruction:alignment ratio while keeping total examples constant to separate ratio effects from scale effects.

6. **Diverse alignment data types.** Test UFT with pairwise-only data (converting pairwise preferences to reward scores via some heuristic), binary feedback data, and mixed data types to validate the claim that UFT handles arbitrary reward formats.

7. **Larger model scales.** Test at 1B (to see if UFT's advantage is even larger where overfitting is most severe) and at 70B+ (to test whether the advantage disappears entirely, as the 32B results hint it might).

8. **Training dynamics analysis.** Plot learning curves for UFT vs. SFT vs. sequential methods, tracking both training loss and downstream task performance over time. This would reveal whether UFT's benefit comes from faster convergence, better final performance, or resistance to overfitting later in training.

**Summary assessment:** The experiments convincingly demonstrate that UFT prevents the specific degradation pattern observed when sequentially applying SFT and alignment to Mistral 7B and Qwen 32B using UltraChat and HelpSteer2. The gains on ifeval and truthful are substantial and practically meaningful. The claim that UFT outperforms SFT alone is directionally correct but the margins are small and fragile. The framework's generality to other data types, scales, and alignment methods is asserted rather than demonstrated. The absence of statistical characterisation, the hyperparameter imbalance in baseline comparisons, and the LoRA-only experiments are the most significant limitations. The paper succeeds as a proof of concept for unified fine-tuning but overstates the breadth of its empirical validation relative to the scope of its claims.

## 6. Limitations and Trade-offs

### The Qwen 32B Results Partially Undermine the Claim That UFT Is Decisively Superior at Larger Scales

UFT’s strongest results appear on the 7B model, where degradation from sequential methods is most severe. On Qwen 32B, the advantage shrinks considerably. On the new leaderboard (Table 4), Qwen+UFT (52.39) leads Qwen+SFT+UNA (51.13) by only 1.26 points — a 2.5% relative improvement, compared to 12.5% on Mistral. On generation benchmarks (Table 6), UFT and sequential SFT+UNA are essentially tied on MT-Bench (8.67 vs. 8.57) and Alpaca-eval (13.79 vs. 13.75). The paper acknowledges this explicitly in Section 4.2: “the performance degradation caused by SFT and alignment is diminished with larger model size, and this is consistent the conclusions of previous works on RLHF.”

The practical consequence is that UFT’s value proposition depends on model scale in a way the paper does not systematically characterise. A practitioner training a 70B or 140B model — the regime where sequential pipelines are most expensive and a unified alternative would be most welcome — cannot confidently extrapolate from these results. At some model size, the advantage of joint training may become negligible, and the cost of switching from a well-understood sequential pipeline (with separate SFT and alignment stages, each with established hyperparameter heuristics) to a unified framework may not be justified by the diminishing returns.

The paper tests only two model sizes (7B, 32B), leaving a gap between 32B and the scales where production alignment typically operates (70B–405B). It does not offer scaling projections, theoretical arguments about asymptotic behaviour, or even speculation about where the crossover point might lie. This is a case where the empirical support weakens precisely in the regime most relevant to practitioners.

**Mitigation status:** Not addressed. The paper notes the diminishing trend but treats it as an observation, not as a limitation requiring further investigation. No experiments at intermediate scales (e.g., 13B, 70B) are proposed.

---

### UFT’s Reported Gains Over Sequential Baselines Are Inflated by the Absence of Hyperparameter Tuning for the Alignment Stage

The paper conducts an extensive hyperparameter sweep for UFT: a 4×4 grid of learning rates and β values (16 configurations, Tables 12–13), selecting the best combination (lr = 3×10⁻⁵, β = 0.01) for the final comparison. The sequential baselines — Mistral+SFT+DPO, Mistral+SFT+KTO, and Mistral+SFT+UNA — receive no such treatment. Section 3 states that “the best performing SFT model of learning rate 1e−4 in the previous experiment is utilized for further fine-tuning using DPO, KTO and UNA,” but does not report whether DPO’s β, KTO’s target ratio, or UNA’s β and learning rate were swept. Absent evidence to the contrary, a reader must assume these alignment-stage hyperparameters were either set to defaults or chosen without systematic exploration.

This asymmetry matters because alignment methods are known to be hyperparameter-sensitive. DPO’s β controls the strength of regularisation toward the reference model — too high and the model learns nothing from preferences, too low and it overfits. UNA’s β plays an analogous role in the implicit reward formulation. The sequential SFT+UNA baseline is particularly important: it is the direct sequential analog of UFT, since both use the same implicit reward mechanism (UNA). If UFT-UNA with sweeping outperforms sequential-UNA without sweeping, the comparison conflates the benefit of joint training with the benefit of hyperparameter optimisation.

The consequence is that UFT’s 3.65-point advantage over the best sequential method on Mistral’s new leaderboard average (Table 1) and the 19-point ifeval gap cannot be cleanly attributed to unified training alone. A practitioner who invests equivalent tuning effort into a sequential pipeline may close much of the gap — though likely not all, given the internal consistency of UFT’s advantage across tasks.

**Mitigation status:** Not addressed. The paper does not mention hyperparameter tuning for sequential baselines, report the configurations used, or discuss this as a fairness concern. A simple sensitivity check — testing DPO and UNA at 2–3 β values — would have substantially strengthened the claim.

---

### The Difficulty Estimation Cost Analogy: UFT’s Fixed Score Assignment for SFT Data Is Unrealistic and Unverified

UFT’s central design choice — assigning r_φ = 1 to every instruction-tuning example — encodes the assumption that all SFT responses are of uniformly maximal quality. The paper states this explicitly in Section 2.3: “Due to the high quality of instruction-tuning data, they can be regarded as data with a score of 1, i.e., positive feedback.”

This is a strong assumption that rarely holds in practice. Real instruction-tuning datasets (including UltraChat, which is ChatGPT-generated and contains varying quality) exhibit substantial heterogeneity: some responses are concise, accurate, and helpful; others are verbose, partially incorrect, or stylistically inconsistent. Assigning r_φ = 1 to all of them trains the model to assign maximal implicit reward — and thus maximal probability boost relative to the pretrained distribution — to low-quality examples alongside high-quality ones. This can cause the model to learn and reproduce undesirable patterns present in the SFT data.

The paper provides no analysis of whether this assumption causes harm. There is no experiment where SFT data is scored by a reward model or human annotator and assigned variable r_φ values rather than a uniform 1. There is no comparison showing whether UFT with variable SFT scores outperforms UFT with uniform r_φ = 1. Without such evidence, a practitioner cannot know whether UFT’s gains over SFT come from the ratio-based objective (the claimed mechanism) or simply from having both data types in the same batch, and whether further improvements are possible by relaxing the uniform-score assumption.

A related concern: the paper never specifies how the multi-dimensional HelpSteer2 scores (helpfulness, correctness, coherence, complexity, verbosity) are aggregated into the scalar r_φ used for alignment data. If the aggregation is flawed — e.g., averaging incommensurate dimensions, or using a dimension that correlates with length — the alignment signal the model receives may be systematically distorted. The absence of this detail makes UFT harder to reproduce and its alignment behaviour harder to predict.

**Mitigation status:** Not addressed. The paper treats the r_φ = 1 assignment as definitional and does not ablate it, discuss its potential failure modes, or propose variable-scoring extensions. The HelpSteer2 score aggregation is not described.

---

### All Experiments Operate Under LoRA With a Fixed Rank; Generalisation to Full Fine-Tuning Is Unverified

Every experiment in the paper uses Low-Rank Adaptation with rank r = 16 (Section 3). LoRA constrains parameter updates to low-dimensional subspaces, which fundamentally changes the optimisation dynamics relative to full-weight training. The ratio-based implicit reward objective in UFT — r_θ = β log(π_θ / π_ref) — depends on the log-probability ratio between the trainable policy and the frozen reference model. Under LoRA, π_θ is computed from a rank-constrained perturbation of the base weights, which limits how much the log-ratio can change for any given input and may interact with the KL regularisation effect in ways that full-weight training would not.

Two specific failure modes are plausible:
- **Regularisation redundancy:** LoRA itself acts as a regulariser by restricting the model’s capacity to change. If UFT’s benefit derives from implicit KL regularisation (as the heuristic proof in Section 2.3 argues), and LoRA provides independent regularisation through its rank constraint, then UFT’s advantage over SFT may be partially an artifact of LoRA — full fine-tuning might show a smaller or absent benefit.
- **Insufficient capacity for joint objectives:** LoRA with r = 16 may not have enough degrees of freedom to simultaneously satisfy the instruction-tuning objective (maximising π_θ for 20k+ diverse responses) and the alignment objective (calibrating the policy ratio to match preference scores). The model might settle at a compromise that works under LoRA’s capacity constraints but would not be optimal in the full-weight regime, or conversely, LoRA might prevent the interference that causes degradation in full-weight sequential training, making UFT’s benefits appear larger than they would be in practice.

The paper does not discuss the LoRA interaction at all — it treats LoRA as a neutral efficiency tool rather than a component that could systematically affect the results. Given that production LLM fine-tuning (especially for mid-to-large models) increasingly uses full-weight training or higher-rank adapters, this is a significant generalisation gap.

**Mitigation status:** Not addressed. No full fine-tuning experiments, no LoRA rank ablation (e.g., r = 8, 32, 64), and no discussion of how the rank constraint might interact with the ratio-based objective.

---

### The Statistical Reliability of All Reported Results Is Unknown

The paper reports single-point estimates for every metric in every table, without standard deviations, confidence intervals, standard errors, or multi-seed averages. The test sets from the HuggingFace leaderboards contain 500–1,000 questions per task (typical for mmlu, arc, hellaswag), meaning differences of 0.5–1.0 percentage points on individual tasks are often within the range of sampling variation. The paper’s headline comparisons involve differences of this magnitude: UFT vs. SFT on Mistral’s new leaderboard is a 0.22-point gap (30.09 vs. 29.87, Table 1); UFT vs. SFT+UNA on Qwen’s old leaderboard is 0.71 points (80.29 vs. 79.58, Table 5).

The paper uses the test sets for both hyperparameter selection (choosing the best learning rate and β based on average leaderboard scores; Appendix B) and final evaluation, which inflates the apparent performance because the “best” configuration is selected precisely because it scored well on these same tasks. Without a held-out validation set or cross-validation, the reported numbers are optimistically biased — they represent the upper tail of performance that would be observed across multiple random seeds and evaluation splits.

The non-monotonic patterns in the data distribution experiment (Section 4.3) — particularly ifeval fluctuating by 4.85 points between 32k and 65k instruction examples (46.76 → 41.91, Table 7) — are flagged by the paper but treated as genuine phenomena rather than potential noise. Without error bars, a reader cannot distinguish a real training-dynamics effect from random variation due to small test sets and single-seed training.

A practitioner using these results to make deployment decisions needs to know whether UFT’s 1–2 point average advantage over SFT is reliable or within the noise. The paper provides no basis for answering this.

This limitation is related to but distinct from the hyperparameter-tuning imbalance (Limitation 2). Even if the sequential baselines were perfectly tuned, the single-seed, no-error-bar reporting would still prevent rigorous comparison. The two issues compound: we cannot know whether UFT is genuinely better, and we cannot measure the uncertainty in that judgment.

**Mitigation status:** Not addressed. The paper does not mention multiple random seeds, report any variance estimates, or discuss the statistical significance of its comparisons. The word “significant” appears in Section 4.3 to describe observed differences (“statistically significant improvements”) without any supporting statistical test, p-value, or confidence interval.

## 7. Implications and Future Directions
- Field impact
  - By demonstrating that SFT and alignment can be trained together under one objective (Eq. 6–7), UFT reframes post-training as a single-stage “learn-from-feedback” process. This simplifies pipelines and reduces capability regressions commonly seen after alignment.

- Practical applications
  - Building instruction-following assistants that also maintain safety and factuality without multi-stage tuning.
  - Continual post-training: add new alignment data or new instruction data and keep optimizing the same objective.
  - Deployment-time adaptation: online UFT with LLM-as-judge or reward models (Section 2.2) for rapid feedback incorporation.

- Research directions
  - Reward modeling: richer score schemas (calibrated 0–1 scales, multi-dimensional rewards for helpfulness, harmlessness, faithfulness) plugged into Eq. 6.
  - Objective design: explore alternative `g` functions and schedules for `β`, and token- or step-level variants compatible with Eq. 5.
  - Data curation: principled methods to balance instruction and alignment proportions per target KPI (e.g., optimize mixture for `truthful` vs `ifeval`).
  - Broader settings: multilingual/post-training in specialized domains; full-parameter vs adapter-based tuning; larger model scales; online learning stability.
  - Theoretical analysis: tighter guarantees for the heuristic equivalence to SFT (Eq. 7), calibration of implicit reward to explicit reward scales, and convergence properties under mixed feedback.

> Core takeaway: Figure 2 shows UFT’s single pipeline; Eqs. 5–7 formalize the unifying loss; Tables 1–6 demonstrate better average performance and reduced alignment tax; Tables 7–9 reveal how instruction/alignment ratios shape outcomes. UFT offers a practical and theoretically grounded path to combine utility and safety training in one stage.
