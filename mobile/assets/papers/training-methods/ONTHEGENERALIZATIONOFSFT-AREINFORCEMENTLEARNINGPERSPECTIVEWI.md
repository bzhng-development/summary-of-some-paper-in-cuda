# ON THE GENERALIZATION OF SFT: A REINFORCEMENT LEARNING PERSPECTIVE WITH REWARD RECTIFICATION

**ArXiv:** [2508.05629](https://arxiv.org/abs/2508.05629)

## 🎯 Pitch

This paper delivers a breakthrough by rigorously showing that standard supervised fine-tuning (SFT) for large language models is equivalent to a reinforcement learning policy gradient update with a problematic, sparse implicit reward—explaining SFT’s poor generalization relative to RL. The authors introduce Dynamic Fine-Tuning (DFT), a remarkably simple, one-line change that rescales the SFT loss by each token’s probability, stabilizing training and dramatically boosting generalization on hard benchmarks (math, code, multimodal tasks) and even outperforming state-of-the-art RL methods in offline settings. This work not only deepens our theoretical understanding of LLM training but also provides a practical, efficient alternative to RL-based alignment—with major implications for improving LLM robustness and usefulness wherever standard SFT is used.

---

## 1. Executive Summary

This paper introduces **Dynamic Fine-Tuning (DFT)**, a theoretically motivated improvement to Supervised Fine-Tuning that addresses SFT's limited generalization compared to reinforcement learning by rectifying an implicit, ill-posed reward structure in the standard cross-entropy gradient. Through mathematical analysis connecting the SFT gradient to a policy gradient with importance-weighted sparse rewards, the authors show that SFT implicitly penalizes low-probability expert tokens with disproportionately large gradients, and DFT corrects this by dynamically rescaling the objective at each token by the model's own probability for that token—a single-line code change from `−log π` to `−sg(π) log π`. Evaluated on mathematical reasoning benchmarks (MATH500, Minerva Math, OlympiadBench, AIME 2024, AMC 2023) using Qwen2.5-Math-1.5B, DFT achieves an average gain of +15.66 points over the base model—over 5.9× larger than SFT's +2.09 point improvement—and in offline RL settings with rejection-sampled data, DFT outperforms both offline methods (DPO, RFT) and online methods (PPO, GRPO) on the same model, establishing that confidence-based reweighting alone can match or exceed explicit reward-based training when the model's prior competence aligns with the task.

## 2. Context and Motivation

### The Core Problem: SFT Generalizes Poorly Compared to RL, and Nobody Knows Exactly Why

The fundamental tension this paper addresses is one of the most practically significant yet theoretically murky questions in LLM post-training: **why does Supervised Fine-Tuning (SFT) on expert demonstrations consistently underperform reinforcement learning (RL) on generalization, even when both methods see the same underlying data?** The empirical pattern is well-documented but poorly understood. Chu et al. (2024) provide the pithiest summary: "SFT memorizes while RL generalizes." The paper opens by citing this finding directly (Section 1), establishing that the phenomenon is robust across both textual and visual domains.

This gap matters enormously for several reasons the paper makes explicit:

- **SFT is the most widely used post-training paradigm.** Chung et al. (2024), Zhang et al. (2024c), Sanh et al. (2022), and Ouyang et al. (2022) all rely on SFT as a core component. Its simplicity—minimize cross-entropy on expert outputs—makes it the default choice for task adaptation, instruction following, and capability enhancement. If SFT has a fundamental limitation baked into its objective function, that limitation propagates across a vast fraction of deployed LLM systems.

- **RL is often impractical.** As the paper notes (Section 1), RL methods like PPO (Schulman et al., 2017) and GRPO (Shao et al., 2024) require substantial computation, careful hyperparameter tuning, and—critically—explicit reward signals. These conditions are frequently absent in real-world settings. When a practitioner has only positive demonstrations (no negative examples, no reward model, no preference pairs), SFT is the *only* viable option. Improving SFT in this "native setting"—the paper's term—therefore directly benefits the most resource-constrained and common deployment scenarios.

- **The SFT-RL gap is a barrier to understanding optimization in LLMs.** The fact that two methods trained on the same expert data produce qualitatively different behaviors (memorization vs. generalization) suggests something fundamental about the optimization landscape that current theory doesn't capture. Closing this gap isn't just an engineering problem—it's a scientific one about how language models internalize knowledge from demonstrations.

### Conflicting Demands: SFT Is Indispensable but Insufficient

The paper carefully situates itself within a literature that reveals a paradox. On one hand, SFT is universally acknowledged as essential for certain functions. Chu et al. (2024) show that SFT remains "indispensable as an initialization step, stabilizing output formatting prior to effective RL training" (Section 2). Mandlekar et al. (2022) similarly find in robotics that RL struggles to recover the expert-like behaviors that SFT—analogous to behavioral cloning—captures efficiently.

On the other hand, the generalization penalty is severe. The paper cites a systematic comparison showing that across textual and visual domains, SFT-trained models overfit to the surface statistics of demonstrations while RL-trained models discover more robust policies (Section 2). This isn't merely an academic observation—it manifests as SFT models *degrading* on challenging benchmarks even as their training loss decreases. Section 4.1 provides concrete examples: on OlympiadBench, SFT drops Qwen2.5-Math-1.5B accuracy from 15.88 (base model) to 12.63; on AIME 2024, SFT reduces Qwen2.5-Math-7B from 6.68 to 2.48. The model is learning, but it's learning the wrong thing for generalization.

### Where Prior Approaches Fall Short

The paper identifies specific limitations in three categories of prior work:

**1. Hybrid SFT+RL methods don't fix SFT itself.** The dominant approach—exemplified by InstructGPT (Ouyang et al., 2022)—is to use SFT for initialization, then apply RL with a learned reward model for refinement. More recent variants interleave SFT and RL updates (Sheng et al., 2025; Liu et al., 2025; Qiu et al., 2025). DPO (Rafailov et al., 2023) bypasses explicit reward modeling by optimizing directly on preference data. NFT (Chen et al., 2025b) models incorrect generations via an implicit negative policy. All of these, the paper argues, "rely on reward signals, preference pairs, or negative samples" and "do not fundamentally improve SFT in its native setting, where only positive demonstrations are available" (Section 2). They enrich the pipeline without addressing the core limitation of the objective function itself.

**2. Theoretical unification attempts don't produce actionable fixes.** Several prior works have recognized the conceptual connection between SFT and RL. Du et al. (2025) reinterpret RLHF as reward-weighted SFT but preserve reliance on explicit rewards. Wang et al. (2025a) cast SFT as RL with an implicit reward and propose smaller learning rates to manage a vanishing KL constraint. Abdolmaleki et al. (2025) analyze learning from positive and negative feedback. These works establish conceptual bridges but, the paper contends, "do not provide a precise mathematical equivalence between the SFT gradient and the offline policy gradient" (Section 2). The connections are qualitative rather than quantitative, making it hard to derive precise corrections.

**3. Loss-reweighting methods make assumptions or miss the mechanism.** Some prior work has independently arrived at the idea of reweighting training losses to improve SFT. MixCE (Zhang et al., 2023) combines forward and reverse KL divergences. GOLD (Pang & He, 2021) adopts offline RL with demonstrations but introduces "reliance on an unknown demonstration distribution $\pi_b$ and a restrictive $1/N$ assumption" (Section 2). Qin & Springenberg (2025) introduce importance weighting based on the data-generating policy, which the paper treats as concurrent work and compares against in Appendix A.4, finding DFT more consistent and computationally simpler (no separate reference model required). The paper explicitly notes that its weighting philosophy inverts that of Focal Loss (Lin et al., 2017): Focal Loss downweights well-classified examples (`−(1 − p)^γ log p`) to emphasize hard cases, while DFT downweights poorly-classified examples (`−p log p`) to encourage generalization. This inversion, the paper argues, reflects "a fundamental shift in the LLM era: while underfitting was once a central challenge, overfitting and memorization now dominate" (Section 2).

### How This Paper Positions Itself

The paper's positioning is threefold, each building on the executive summary's core insight but now fleshed out in context:

**First, it offers a precise mathematical derivation rather than a qualitative analogy.** The key move in Section 3.2 is rewriting the SFT gradient (an expectation under the fixed demonstration distribution) as an on-policy RL gradient via importance sampling. The derivation (Equation 5, expanded in Appendix A.2) shows:

$$E_{(x,y^\star)\sim D}[-\nabla_\theta \log \pi_\theta(y^\star|x)] = E_{x\sim D_x} E_{y\sim\pi_\theta(\cdot|x)}\left[\frac{\mathbb{1}[y = y^\star]}{\pi_\theta(y|x)}(-\nabla_\theta \log \pi_\theta(y|x))\right]$$

This isn't just "SFT is like RL"—it's an exact algebraic equivalence. Under this formulation, the implicitly defined reward is $r(x,y) = \mathbb{1}[y = y^\star]$ (sparse: 1 for exact expert match, 0 otherwise), and the importance weight is $w(y|x) = 1/\pi_\theta(y|x)$. The paper's central theoretical claim is that this $1/\pi_\theta$ factor is the culprit: "when the model assigns low probability to expert actions, the gradient becomes excessively large, yielding an ill-posed reward structure and unstable optimization" (Section 1, Section 3.2).

**Second, it derives a principled correction rather than an ad-hoc fix.** The "reward rectification" in Section 3.3 follows directly from the analysis: multiply by the inverse of the problematic weight factor, i.e., by $\pi_\theta(y^\star|x)$, to cancel the $1/\pi_\theta$ distortion. The stop-gradient operator `sg(·)` is critical—without it, the correction would alter gradient flow through the probability term itself, creating a different optimization problem. With the stop-gradient, the weight is treated as a constant scaling factor, yielding this simplified gradient (Appendix A.3):

$$\nabla_\theta L_{DFT} = -\nabla_\theta \pi_\theta(y^\star|x)$$

compared to standard cross-entropy's:

$$\nabla_\theta L_{CE} = -\frac{1}{\pi_\theta(y^\star|x)} \nabla_\theta \pi_\theta(y^\star|x)$$

Both share the same direction (toward higher probability on the target), but CE amplifies updates for low-probability tokens by a factor of $1/\pi$, while DFT applies uniform scaling. The paper argues this avoids "over-concentration on specific low-probability reference tokens" (Section 3.3). From the RL perspective, the rectified reward becomes uniformly 1 for all expert trajectories—the paper explicitly connects this to "contemporary verification based reward approach RLVR (DeepSeek-AI et al., 2025) that assigns uniform reward to all correct samples" (Section 3.3).

**Third, it demonstrates that this single-line change produces gains competitive with full RL pipelines.** The paper doesn't just claim theoretical elegance—it shows that DFT in an offline RL setting (Section 4.2), using only rejection-sampled correct responses as training data with no negative examples, no reward model, and no online interaction, outperforms both offline methods (DPO, RFT) and online methods (PPO, GRPO) on Qwen2.5-Math-1.5B. On the average across five math benchmarks, DFT achieves 35.43 vs. GRPO's 32.00 and PPO's 28.66 (Table 2). This is striking because DFT requires "neither a reference model nor large batch sizes" (Section 1)—GRPO and PPO need both.

**A key boundary condition the paper is explicit about:** DFT is not a universal replacement for SFT. Section 4.5 presents a case study on the Natural Questions dataset (factual knowledge), where SFT improves performance from 31.24% to 36.62% but DFT *reduces* it to 30.14%. The paper's explanation: "because it reweights samples based on the model's own confidence, it tends to reinforce the model's existing beliefs. When the model lacks sufficient factual knowledge, such reinforcement may hinder effective learning instead of facilitating it" (Section 4.5). This establishes DFT's domain of applicability: reasoning tasks where the model has non-trivial prior competence, not factual knowledge acquisition where the model starts from ignorance. The paper's aim, it states explicitly, is "not to assert that DFT universally outperforms SFT, but rather to offer a new perspective on objective design" (Section 5).

## 3. Technical Approach

### 3.1 Reader Orientation

This paper proposes a **single-line modification to the standard supervised fine-tuning loss** for large language models—changing `−log π` to `−sg(π) log π`—that is derived from a mathematical equivalence showing that the standard SFT gradient implicitly encodes an unstable, inverse-probability-weighted policy gradient. The core problem it solves is that standard SFT overfits to low-probability tokens in expert demonstrations, producing disproportionately large gradients that destabilize training and harm generalization; the solution "rectifies" this implicit reward structure by cancelling the inverse-probability weight with the model's own token probability, producing a uniformly weighted update that behaves more like reinforcement learning without requiring any reward model, preference data, or online sampling.

### 3.2 Big-Picture Architecture (Diagram in Words)

The system has three conceptual layers, though the practical implementation reduces to a single loss function change:

1. **Base Language Model (policy `$\pi_\theta$`)** — a pretrained LLM (e.g., Qwen2.5-Math-1.5B, LLaMA-3.2-3B) that defines a probability distribution over token sequences given a prompt. This is the same model used in standard SFT; DFT does not modify the architecture.

2. **Standard SFT Loss (cross-entropy)** — the conventional objective `$-\log \pi_\theta(y^\star|x)$` that maximizes the log-probability of expert response `$y^\star$` given prompt `$x$`. This is the baseline that DFT modifies.

3. **Dynamic Reweighting Mechanism (the DFT modification)** — a stop-gradient scaling factor `$\text{sg}(\pi_\theta(y^\star_t | y^\star_{<t}, x))$` multiplied into the per-token cross-entropy, producing `$-\text{sg}(\pi_\theta(y^\star_t | y^\star_{<t}, x)) \log \pi_\theta(y^\star_t | y^\star_{<t}, x)$`. Because the stop-gradient treats the scaling factor as a constant during backpropagation, the effective gradient becomes `$-\nabla_\theta \pi_\theta(y^\star_t | y^\star_{<t}, x)$` rather than standard cross-entropy's `$-(1/\pi_\theta(y^\star_t | y^\star_{<t}, x)) \nabla_\theta \pi_\theta(y^\star_t | y^\star_{<t}, x)$`.

Information flows as follows: a training batch of (prompt, expert response) pairs enters → the model computes token-level probabilities `$\pi_\theta(y^\star_t | y^\star_{<t}, x)$` for each expert token → these probabilities are detached from the computation graph via `sg(·)` → the detached probabilities are used as multiplicative weights on the standard per-token negative log-likelihood → the sum of weighted per-token losses is averaged to produce a scalar loss → backpropagation proceeds through the `$\log \pi_\theta$` term only, with the weight treated as a constant.

### 3.3 Roadmap for the Deep Dive

- **First**, the mathematical derivation that establishes SFT as a special case of RL policy gradient with an implicitly defined, ill-posed reward (Section 3.2 of the paper), since this derivation is the entire theoretical motivation for DFT.
- **Second**, the specific correction DFT applies—the "reward rectification" step—and why it takes the form it does, including the critical role of the stop-gradient operator and the transition from sequence-level to token-level weighting (Section 3.3).
- **Third**, the gradient analysis showing what DFT's effective gradient actually computes and how it differs from standard cross-entropy (Appendix A.3), since understanding the gradient-level difference is essential for understanding why the method works.
- **Fourth**, the relationship to RLVR and why the rectified reward becomes uniformly 1, connecting the theoretical motivation to contemporary RL practices.
- **Fifth**, the practical implementation details—the single-line code change, hyperparameter configurations, and the training setup used across all experiments (Section 4.1 and Appendix).

### 3.4 Detailed, Sentence-Based Technical Breakdown

This is primarily a **theoretically motivated optimization paper** whose core idea is that the standard SFT gradient contains a hidden inverse-probability weight that can be algebraically cancelled by multiplying the per-token loss by the model's own probability for that token, yielding a more stable and better-generalizing update. The method itself is trivial to implement but non-obvious without the mathematical derivation.

---

#### The SFT Gradient as an Ill-Posed Policy Gradient

The paper's central theoretical move is rewriting the SFT gradient—normally computed as an expectation over a fixed dataset of expert demonstrations—as an on-policy RL gradient through importance sampling. The derivation (Section 3.2, with full details in Appendix A.2) starts from the standard SFT objective.

**Standard SFT loss.** The sentence-level cross-entropy minimized in SFT is:

$$L_{SFT}(\theta) = \mathbb{E}_{(x,y^\star)\sim\mathcal{D}}\left[-\log \pi_\theta(y^\star | x)\right]$$

where `$\mathcal{D}$` is the dataset of prompt-response pairs, `$x$` is a prompt, `$y^\star$` is the expert (ground-truth) response, and `$\pi_\theta(y^\star|x)$` is the model's probability of generating `$y^\star$` given `$x$` under parameters `$\theta$`.

**What it computes:** the negative log-likelihood of the expert response under the model's distribution, averaged over the training set. Minimizing this loss maximizes the model's probability of exactly reproducing the expert output for each training prompt.

**Why this form:** cross-entropy is the maximum-likelihood objective for categorical distributions, making it the standard choice for supervised sequence learning. It has the property that the gradient points in the direction that increases probability mass on the correct token at each position.

**Standard SFT gradient.** Taking the gradient with respect to `$\theta$` yields:

$$\nabla_\theta L_{SFT}(\theta) = \mathbb{E}_{(x,y^\star)\sim\mathcal{D}}\left[-\nabla_\theta \log \pi_\theta(y^\star | x)\right]$$

where `$\nabla_\theta \log \pi_\theta(y^\star|x)$` is the score function—the gradient of the log-probability of the expert sequence with respect to the model parameters.

**What it computes:** the average direction in parameter space that increases the log-probability of expert responses. Each training example contributes a `$-\nabla_\theta \log \pi_\theta(y^\star|x)$` vector.

**The importance sampling rewrite.** The critical step is converting this expectation under the fixed dataset distribution into an expectation under the model's own current distribution. The paper does this by introducing an importance weight:

$$\mathbb{E}_{(x,y^\star)\sim\mathcal{D}}\left[-\nabla_\theta \log \pi_\theta(y^\star | x)\right] = \mathbb{E}_{x\sim\mathcal{D}_x} \mathbb{E}_{y\sim\pi_\theta(\cdot|x)}\left[\frac{\mathbb{1}[y = y^\star]}{\pi_\theta(y|x)}\left(-\nabla_\theta \log \pi_\theta(y | x)\right)\right]$$

where `$\mathcal{D}_x$` is the marginal distribution of prompts in the training set, `$y \sim \pi_\theta(\cdot|x)$` denotes sampling a response from the model's current policy, `$\mathbb{1}[y = y^\star]$` is the indicator function that is `1` when the sampled response exactly matches the expert response and `0` otherwise, and `$\pi_\theta(y|x)$` in the denominator is the model's probability of the sampled response (which equals `$\pi_\theta(y^\star|x)$` when the indicator fires, and is irrelevant—multiplied by zero—otherwise).

**What it computes:** mathematically identical quantity to the original SFT gradient, but expressed as an expectation under the model's own sampling distribution `$\pi_\theta(\cdot|x)$` rather than under the fixed dataset distribution. The indicator `$\mathbb{1}[y=y^\star]$` acts as a sparse reward signal that is non-zero only when the model's sample exactly matches the expert; the `$1/\pi_\theta(y|x)$` term reweights these rare-match events to recover the correct expectation.

**Why this form:** this rewrite exposes the structural similarity to the policy gradient in reinforcement learning. The standard policy gradient for maximizing expected reward `$r(x,y)$` is:

$$\nabla_\theta J(\theta) = \mathbb{E}_{x\sim\mathcal{D}_x, y\sim\pi_\theta(\cdot|x)}\left[\nabla_\theta \log \pi_\theta(y|x) \cdot r(x,y)\right]$$

Comparing the two, we can identify:

$$r_{SFT}(x,y) = \mathbb{1}[y = y^\star] \quad \text{(the implicit reward: 1 for exact expert match, 0 otherwise)}$$
$$w(y|x) = \frac{1}{\pi_\theta(y|x)} \quad \text{(the importance weight)}$$

The SFT gradient can then be written compactly as:

$$\nabla_\theta L_{SFT}(\theta) = -\mathbb{E}_{x\sim\mathcal{D}_x, y\sim\pi_\theta(\cdot|x)}\left[w(y|x) \cdot \nabla_\theta \log \pi_\theta(y|x) \cdot r_{SFT}(x,y)\right]$$

This reveals that standard SFT is equivalent to a policy gradient method with **two pathological properties**:

1. **Reward sparsity:** the implicit reward `$r_{SFT}(x,y)$` is non-zero only for exact sequence match to the expert output. For a vocabulary of size `$V$` and sequence length `$L$`, the probability of sampling an exact match is astronomically small for any reasonable model, meaning almost all samples contribute zero to the expectation. The training signal comes exclusively from the importance weight correcting for this sparsity.

2. **Inverse-probability weighting:** the importance weight `$w(y|x) = 1/\pi_\theta(y|x)$` grows without bound as the model assigns low probability to the expert response. When the model is uncertain about the correct output (which is common early in training or on difficult examples), `$\pi_\theta(y^\star|x)$` is very small—potentially `$10^{-6}$` or lower for a long sequence—making the weight `$1/\pi_\theta$` extremely large (`$10^6$` or higher). This amplifies the gradient by that same factor, creating "disproportionately large gradients and training instability" (Section 3.2).

The paper argues that this second property—the inverse-probability weight—is the "key contributor to SFT's generalization limitations compared to RL" (Section 3.2). It effectively creates an ill-posed reward landscape where low-probability expert tokens receive massively amplified training signals, causing the model to overfit to rare exact-match samples rather than learning the underlying reasoning patterns.

---

#### Reward Rectification via Dynamic Reweighting

The correction DFT applies follows directly and algebraically from the diagnosis. If the problem is the `$1/\pi_\theta$` weight distorting the gradient, the solution is to multiply by `$\pi_\theta$` to cancel it.

**The core idea.** Starting from the SFT gradient expressed as a policy gradient with importance weight `$w = 1/\pi_\theta$`:

$$\nabla_\theta L_{SFT} = -\mathbb{E}\left[w \cdot \nabla_\theta \log \pi_\theta \cdot r\right]$$

The paper proposes multiplying the reward by a corrective factor `$1/w = \pi_\theta$`:

$$\nabla_\theta L_{DFT} = -\mathbb{E}_{x\sim\mathcal{D}_x, y\sim\pi_\theta(\cdot|x)}\left[\text{sg}\left(\frac{1}{w}\right) \cdot w(y|x) \cdot \nabla_\theta \log \pi_\theta(y|x) \cdot r(x,y)\right]$$

where `$\text{sg}(\cdot)$` is the **stop-gradient operator**—it treats its argument as a constant during backpropagation, so gradients do not flow through the corrective weight.

**Why the stop-gradient is critical.** Without `$\text{sg}(\cdot)$`, the factor `$\pi_\theta$` would appear inside the gradient computation, creating a `$\nabla_\theta(\pi_\theta \log \pi_\theta)$` term that is a fundamentally different optimization problem. The stop-gradient ensures that the weight is computed using the current model probabilities but treated as a fixed scalar for the backward pass. This means the corrective factor `$\pi_\theta$` **cancels the inverse-probability weight `$1/\pi_\theta$` numerically in the forward pass** (since `$\text{sg}(\pi_\theta) \cdot (1/\pi_\theta) = 1$`), while **not introducing any additional gradient terms from the weight itself**.

**Transition to the loss formulation.** Because the indicator function `$\mathbb{1}[y=y^\star]$` is zero except when the sample exactly matches the expert, the expectation over `$y \sim \pi_\theta$` is non-zero only at `$y = y^\star$`. At that point, `$\pi_\theta(y|x) = \pi_\theta(y^\star|x)$`. The corrective factor becomes `$\text{sg}(\pi_\theta(y^\star|x))$`. Since the expectation over samples now collapses to the single expert sequence (all other samples contribute zero regardless of the weight), the DFT objective simplifies to a reweighted version of the standard SFT loss:

$$L_{DFT}(\theta) = \mathbb{E}_{(x,y^\star)\sim\mathcal{D}}\left[-\text{sg}(\pi_\theta(y^\star|x)) \cdot \log \pi_\theta(y^\star|x)\right]$$

where `$\text{sg}(\pi_\theta(y^\star|x))$` is the model's probability of the entire expert sequence `$y^\star$` given prompt `$x$`, detached from the computation graph, and `$\log \pi_\theta(y^\star|x)$` is the standard log-probability receiving gradients.

**What it computes:** a scaled cross-entropy loss where each training example is weighted by the model's own current confidence (probability) in the correct answer. When the model is already confident (`$\pi_\theta(y^\star|x)$` close to 1), the weight is near 1 and the loss behaves like standard cross-entropy. When the model is uncertain (`$\pi_\theta(y^\star|x)$` close to 0), the weight is near 0 and the loss is almost completely suppressed, preventing the large gradient that standard SFT would produce for that example.

**Why this form:** it directly cancels the `$1/\pi_\theta$` distortion identified in the theoretical analysis. From the RL perspective, the rectified effective reward becomes `$\text{sg}(\pi_\theta(y^\star|x)) \cdot r_{SFT}(x,y) = \text{sg}(\pi_\theta(y^\star|x)) \cdot \mathbb{1}[y=y^\star]$`, which evaluates to `$\pi_\theta(y^\star|x)$` at the expert trajectory rather than `1`. The paper notes (Section 3.3) that this is "akin to contemporary verification based reward approach RLVR (DeepSeek-AI et al., 2025) that assigns uniform reward to all correct samples"—both methods treat all correct outputs equally rather than differentially weighting them by model confidence.

---

#### From Sequence-Level to Token-Level Weighting

The sequence-level formulation `$\pi_\theta(y^\star|x)$` computes the probability of the entire response as a product of conditional token probabilities:

$$\pi_\theta(y^\star|x) = \prod_{t=1}^{|y^\star|} \pi_\theta(y^\star_t | y^\star_{<t}, x)$$

For long sequences (math solutions can be hundreds or thousands of tokens), this product becomes vanishingly small—easily `$10^{-100}$` or smaller—making the sequence-level weight numerically unstable. The paper acknowledges this directly: "in practice, computing importance weights over the entire trajectory can induce numerical instability" (Section 3.3).

**The token-level decomposition.** Following the same approach used in PPO (Schulman et al., 2017), the paper decomposes the weighting to the token level:

$$L_{DFT}(\theta) = \mathbb{E}_{(x,y^\star)\sim\mathcal{D}}\left[-\sum_{t=1}^{|y^\star|} \text{sg}(\pi_\theta(y^\star_t | y^\star_{<t}, x)) \cdot \log \pi_\theta(y^\star_t | y^\star_{<t}, x)\right]$$

where `$|y^\star|$` is the length of the expert response in tokens, `$y^\star_t$` is the `$t$`-th token of the expert response, `$y^\star_{<t}$` are all preceding tokens (the context), `$\pi_\theta(y^\star_t | y^\star_{<t}, x)$` is the model's probability of the correct token at position `$t$` given the prefix and prompt, and `$\text{sg}(\cdot)$` detaches this probability from gradient computation.

**What it computes:** the sum of per-token cross-entropy losses, each scaled by the model's own probability for that specific token in context. A token the model is already likely to produce (high `$\pi_\theta$`) receives a weight near 1 and is trained normally; a token the model finds surprising (low `$\pi_\theta$`) receives a weight near 0 and contributes almost nothing to the loss. The average is then taken over all tokens and all training examples.

**Why token-level:** the per-token probabilities are typically in the range `$[10^{-3}, 0.99]$` rather than the astronomically small sequence-level products, making the weights numerically stable and providing more granular control. A long sequence can now have some tokens weighted heavily (those the model is already confident about) and others weighted lightly (those the model finds difficult), rather than the entire sequence receiving a single near-zero weight.

**Comparison to sequence-level and geometric-mean variants.** The paper empirically compares against two sequence-level alternatives in Section 4.6 (Table 5):

- **Full sequence-level weighting:** using `$\text{sg}(\pi_\theta(y^\star|x))$` as a single weight for the entire sequence loss. The authors report this "makes the loss nearly uninformative and produces a highly skewed weight distribution that is difficult to tune," with average accuracy remaining at 15.75 (essentially unchanged from the base model's 15.92).

- **Geometric-mean weighting:** inspired by GSPO (Zheng et al., 2025), this rescales the sequence probability by taking the `$|y^\star|$`-th root (the geometric mean) to avoid numerical collapse. Even with this stabilization, the training signal is "weak" and produces only marginal gains (17.21 average accuracy vs. 15.92 base).

The token-level formulation achieves 31.58 average accuracy on the same model and data, demonstrating that the per-token granularity is essential for the method to work effectively.

---

#### Gradient Analysis: What DFT Actually Optimizes

Appendix A.3 provides a concise gradient-level analysis that reveals exactly how DFT differs from standard cross-entropy.

**DFT gradient derivation.** Starting from the sequence-level DFT loss (for clarity; the token-level version follows identically):

$$L_{DFT}(\theta) = -\text{sg}(\pi_\theta(y^\star|x)) \cdot \log \pi_\theta(y^\star|x)$$

Since `$\text{sg}(\cdot)$` blocks backpropagation through its argument, `$\text{sg}(\pi_\theta(y^\star|x))$` is treated as a constant `$c$` during differentiation:

$$\nabla_\theta L_{DFT} = -c \cdot \nabla_\theta \log \pi_\theta(y^\star|x) = -c \cdot \frac{1}{\pi_\theta(y^\star|x)} \nabla_\theta \pi_\theta(y^\star|x)$$

In the forward pass, `$c = \pi_\theta(y^\star|x)$`, so the `$\pi_\theta$` and `$1/\pi_\theta$` terms cancel:

$$\nabla_\theta L_{DFT} = -\nabla_\theta \pi_\theta(y^\star|x)$$

**What it computes:** the gradient of DFT is simply the negative gradient of the model's probability of the target token (or sequence). It directly maximizes `$\pi_\theta(y^\star|x)$` rather than maximizing `$\log \pi_\theta(y^\star|x)$`.

**Comparison to standard cross-entropy.** For standard SFT with `$L_{CE} = -\log \pi_\theta(y^\star|x)$`:

$$\nabla_\theta L_{CE} = -\frac{1}{\pi_\theta(y^\star|x)} \nabla_\theta \pi_\theta(y^\star|x)$$

Both DFT and cross-entropy share the **same gradient direction**: both point in the direction that increases `$\pi_\theta(y^\star|x)$`. The difference is entirely in the **scaling factor**:

- **Cross-entropy scaling:** `$1/\pi_\theta(y^\star|x)$` — the gradient magnitude is inversely proportional to the model's current probability. Low-probability tokens receive enormously amplified gradients; high-probability tokens receive modest gradients.

- **DFT scaling:** `$1$` (uniform) — every token contributes a gradient proportional to `$\nabla_\theta \pi_\theta$`, with no amplification based on current model confidence.

**Why uniform scaling matters.** The paper argues that the inverse-probability amplification in cross-entropy is harmful for generalization because it forces the model to concentrate excessive optimization effort on tokens it currently finds surprising. For a mathematical reasoning task, these low-probability tokens might include:

- **Rare substantive terms** (specialized mathematical notation, theorem names) that genuinely need to be learned.
- **Noise or stylistic idiosyncrasies** in the training data (particular phrasing choices, formatting artifacts) that don't generalize.
- **Grammatical function words** (conjunctions, punctuation) that the model misassigns probability to due to unusual sentence structures in mathematical text.

Cross-entropy treats all three categories identically, amplifying gradients for any low-probability token regardless of its importance for the task. DFT, by applying uniform scaling, allows the model to fit confident predictions (high-probability tokens) more aggressively while being conservative about uncertain predictions—which may be uncertain precisely because they don't represent learnable patterns.

The paper's probability distribution analysis (Figure 2, Section 4.7) provides empirical support: after DFT training, the token probability distribution becomes **bimodal** rather than uniformly shifted rightward. SFT uniformly increases probabilities across the board, mainly targeting the lowest-probability tokens. DFT actively suppresses some token probabilities while boosting others, creating more tokens in both the highest and lowest probability bins. The tokens relegated to the lowest bin tend to be "conjunctive words or punctuations such as 'the', 'let', ',', '.' etc." (Section 4.7), which suggests DFT is implicitly learning that function words don't need to be fitted with perfect confidence—a form of implicit regularization.

---

#### Connection to RLVR and the Rectified Reward

From the RL perspective established in Section 3.2, DFT can be understood as modifying the implicit reward function. In the original SFT-as-RL formulation:

$$r_{SFT}(x,y) = \mathbb{1}[y = y^\star]$$

After DFT's correction:

$$r_{DFT}(x,y) = \text{sg}(\pi_\theta(y^\star|x)) \cdot \mathbb{1}[y = y^\star]$$

Since the expectation over `$y\sim\pi_\theta$` is non-zero only at `$y=y^\star$` (where `$\pi_\theta(y|x) = \pi_\theta(y^\star|x)$`), and the corrective factor `$\text{sg}(\pi_\theta)$` cancels the importance weight `$1/\pi_\theta$`, the effective reward in the DFT objective becomes uniformly 1 for all expert trajectories.

The paper explicitly connects this to **RLVR (Reinforcement Learning with Verification Rewards)** from DeepSeek-R1 (DeepSeek-AI et al., 2025): "this is akin to contemporary verification based reward approach RLVR that assigns uniform reward to all correct samples" (Section 3.3). In RLVR, the reward function is a binary verifier: correct answers receive a fixed positive reward (typically +1), incorrect answers receive 0 or a penalty. This uniform reward structure encourages the policy to find *any* path to the correct answer rather than overfitting to the specific path in the demonstration.

DFT achieves the same uniform-reward property without requiring a separate verifier, reward model, or online sampling. The mathematical equivalence established in Section 3.2 shows that standard SFT already *has* an implicit reward structure—it's just that the reward is distorted by inverse-probability weighting. DFT rectifies this distortion, converting the implicit reward from `$\mathbb{1}[y=y^\star]/\pi_\theta$` to a uniform 1, purely through the objective function modification.

**Why this connection matters practically.** It explains why DFT can compete with full RL pipelines in offline settings (Section 4.2, Table 2): the uniform-reward property that RLVR achieves through explicit verification, DFT achieves through algebraic manipulation of the loss function. The key practical difference is that RLVR requires sampling multiple responses and checking them against ground truth (or a verifier) to identify correct answers, while DFT only requires the single expert demonstration already present in the SFT dataset.

---

#### Practical Implementation Details

**The single-line code change.** The paper's method amounts to modifying the standard cross-entropy loss computation. In standard SFT, the per-token loss for a target token `$y_t$` with model-predicted probability `$p_t = \pi_\theta(y_t | y_{<t}, x)$` is:

```
loss = -log(p_t)
```

In DFT, it becomes:

```
loss = -p_t.detach() * log(p_t)
```

where `.detach()` is the PyTorch stop-gradient operation (equivalent to `tf.stop_gradient()` in TensorFlow or `jax.lax.stop_gradient()` in JAX). The detached probability `p_t.detach()` has the same numerical value as `p_t` in the forward pass but contributes zero gradient during backpropagation.

**Training configuration (Section 4.1).** The paper uses the following hyperparameters across all math reasoning experiments:

- **Optimizer:** AdamW
- **Learning rates:**
  - `$5 \times 10^{-5}$` for Qwen2.5-Math-1.5B, Qwen2.5-Math-7B, LLaMA-3.2-3B, DeepSeekMath-7B
  - `$2 \times 10^{-5}$` for LLaMA-3.1-8B-Base (lower rate for the larger base model)
- **Mini-batch size:** 256
- **Maximum input length:** 2048 tokens
- **Learning rate schedule:** cosine decay with a warm-up ratio of 0.1
- **Training data:** 100,000 randomly sampled instances from NuminaMath-CoT (LI et al., 2024) for the main experiments; OpenR1-Math-220k (Hugging Face, 2025) for the higher-quality data experiments
- **Framework:** built on the verl framework (Sheng et al., 2025)
- **Training epochs:** 1 for NuminaMath experiments (as shown in Figure 1, performance peaks and then plateaus or declines within ~400 steps, which is less than one epoch on 100k examples at batch size 256); 3 epochs for OpenR1-Math experiments (Appendix A.5)

**Evaluation setup.** All evaluations use:
- **Temperature:** 1.0 for sampling
- **Maximum generation length:** 4096 tokens
- **Decoding runs:** 16 per benchmark, averaged (denoted "Average@16" in tables)
- **Prompting:** Chain-of-Thought (CoT) prompting with each model's default chat template
- **Benchmarks:** MATH500 (Hendrycks et al., 2021), Minerva Math (Lewkowycz et al., 2022), OlympiadBench (He et al., 2024), AIME 2024, AMC 2023
- **Evaluation pipeline:** Official Qwen2.5-Math evaluation pipeline (Qwen Team et al., 2024a)

**Offline RL setup (Section 4.2).** For the experiments comparing DFT against DPO, RFT, PPO, and GRPO:
- **Data generation:** 100,000 math questions, each with 4 sampled responses from the base Qwen2.5-Math-1.5B model at temperature 1.0
- **Filtering:** Only correct responses (verified by Math-Verify) are retained, yielding approximately 140,000 examples
- **DFT training:** same configuration as the standard SFT setting, trained on the filtered correct responses
- **DPO training:** 100,000 positive-negative preference pairs constructed from the generated responses; learning rate `$1 \times 10^{-6}$`, batch size 128, warmup ratio 0.05, using ms-swift (Zhao et al., 2024)
- **PPO/GRPO training:** learning rate `$1 \times 10^{-6}$`, batch size 256, warmup ratio 0.1; GRPO uses `$n=4$` responses per prompt
- **RFT (Rejection Fine-Tuning):** standard SFT on the filtered correct responses, using the same configuration as DFT

**Code generation setup (Section 4.3).** Uses UltraFeedback (Cui et al., 2024) dataset, sampling 10,000 prompts and selecting the highest-scoring response per prompt for SFT/DFT training. Learning rate `$5 \times 10^{-5}$`, warmup ratio 0.05, batch size 16, one epoch. Evaluated on HumanEval, HumanEval+, and MultiPL-E (8 languages).

**Multi-modal setup (Section 4.4).** Uses WeThink dataset (Yang et al., 2025), fine-tuned with LLaMA-Factory (Zheng et al., 2024), learning rate `$5 \times 10^{-5}$`, one epoch. Evaluated on MathVerse, MathVision, WeMath using VLMEvalKit.

**Parameter-efficient fine-tuning (Appendix A.6).** LoRA with rank=8 and alpha=16, applied to LLaMA-3.2-3B and Qwen2.5-Math-1.5B. All other hyperparameters identical to full-parameter training.

---

#### Relationship to Prior Weighting Methods: Why DFT Inverts Focal Loss

The paper explicitly contrasts its weighting philosophy with Focal Loss (Lin et al., 2017), which is widely used in computer vision for dense object detection.

**Focal Loss:** `$-(1-p)^\gamma \log p$`, where `$\gamma > 0$` is a focusing parameter. When the model is confident (`$p$` close to 1), the weight `$(1-p)^\gamma$` is near 0, downweighting well-classified examples. When the model is uncertain (`$p$` close to 0), the weight is near 1, preserving the full loss. This emphasizes "hard" (low-probability) examples.

**DFT:** `$-p \log p$` (with the `$p$` detached). When the model is confident (`$p$` close to 1), the weight is near 1. When the model is uncertain (`$p$` close to 0), the weight is near 0. This emphasizes "easy" (high-probability) examples.

**Why the inversion?** The paper's explanation (Section 2): "This inversion reflects a fundamental shift in the LLM era: while underfitting was once a central challenge, overfitting and memorization now dominate, demanding a rethinking of objective design." In classical computer vision, the primary challenge was that easy negative examples dominated the loss for dense predictors, drowning out the signal from rare positive examples. Focal Loss addressed this by upweighting hard examples. In LLM fine-tuning, the primary challenge is that rare token patterns in demonstrations—which may be noise, stylistic quirks, or specific phrasings that don't generalize—receive massively amplified gradients through the `$1/\pi_\theta$` factor in cross-entropy. DFT addresses this by downweighting surprising tokens, essentially trusting the model's own uncertainty as a signal that certain predictions shouldn't be forced.

**Connection to learning from noisy demonstrations.** The paper also draws a connection to work on behavioral cloning from noisy data (Sasaki & Yamashina, 2020), noting in Appendix A.3: "their method introduces a weighted behavioral cloning objective, where the weights are derived from a previously trained policy's confidence in each action. Similarly, the weighting mechanism in DFT shares the same intuition, but instead of relying on a fixed old policy model to compute confidence scores, it uses a single policy model to perform confidence-based weighting on-the-fly during training." This is a crucial distinction: DFT doesn't require a separate reference model or a two-stage training process. The model's own evolving confidence serves as the reliability signal, making the method self-contained.

---

#### Summary of Design Choices and Their Justifications

- **Stop-gradient on the weight:** prevents the corrective factor from introducing additional gradient terms that would create a different optimization problem. Without `sg`, the gradient would be `$\nabla_\theta(\pi \log \pi) = (\log \pi + 1)\nabla_\theta \pi$`, which is not simply a rescaled version of the cross-entropy gradient and has different convergence properties.

- **Token-level rather than sequence-level weighting:** avoids numerical underflow from sequence-length probability products. The paper empirically validates this choice in Section 4.6 (Table 5), where both full-sequence and geometric-mean weighting fail to produce meaningful improvements.

- **No additional models or sampling:** DFT modifies only the loss function. It requires no reference model (unlike DPO), no reward model (unlike PPO), no online sampling (unlike GRPO), and no importance weights from a separate policy (unlike iw-SFT). The model's own probabilities serve as the reweighting signal.

- **Uniform effective reward:** the algebraic cancellation of `$1/\pi_\theta$` by `$\pi_\theta$` means that in the RL interpretation, the rectified reward becomes uniformly 1 for all correct trajectories. This connects DFT to RLVR and explains its competitive performance against explicit RL methods in offline settings.

- **Implicit regularization via confidence gating:** by downweighting low-probability tokens, DFT prevents the model from overfitting to rare patterns in the training data. The bimodal probability distribution observed after DFT training (Figure 2) suggests the model learns to confidently predict semantically important tokens while allowing grammatical function words to remain at lower probability—a form of learned attention to content over form.

## 4. Key Insights and Innovations

### Innovation 1: The SFT Gradient IS a Policy Gradient—with a Broken Reward

This paper makes a claim that is simultaneously obvious in retrospect and transformative in implication: **the standard SFT gradient is not "like" reinforcement learning—it mathematically IS a policy gradient with an implicitly defined, fundamentally pathological reward structure.** This is a conceptual contribution, not an engineering one. The paper doesn't invent a new connection between SFT and RL; it provides the precise algebraic equivalence that turns a vague analogy into a diagnostic tool.

Prior work recognized that SFT and RL share conceptual similarities. Du et al. (2025) reinterpret RLHF as reward-weighted SFT. Wang et al. (2025a) cast SFT as RL with an implicit reward. Abdolmaleki et al. (2025) analyzed learning from positive and negative feedback. But all of these connections were qualitative—"SFT is like RL with X property"—rather than quantitative equivalences from which corrective modifications could be derived algebraically. The paper explicitly distinguishes itself on this point (Section 2): prior approaches "do not provide a precise mathematical equivalence between the SFT gradient and the offline policy gradient."

The precise equivalence (Equation 5, derived in Appendix A.2) exposes two specific pathologies that are invisible from the standard cross-entropy formulation but immediately apparent in the RL framing:

1. **The implicit reward is a Dirac delta at the expert trajectory:** `r(x,y) = 1[y = y*]`. This means the reward signal is non-zero only when the model samples the *exact* expert output—an event whose probability, for any non-trivial sequence length, is vanishingly small. The training signal doesn't come from the reward itself; it comes entirely from the importance weight correcting for this sparsity.

2. **The importance weight is `1/π_θ`:** This means that when the model assigns low probability to the expert output—exactly the situation where cross-entropy produces the largest gradients—the importance weight amplifies those gradients further. A token with probability `10^{-6}` receives a gradient scaled by `10^6`, creating the "disproportionately large gradients and training instability" the paper identifies (Section 3.2).

This framing is significant because it **explains, rather than merely documents, the SFT-RL generalization gap.** The field knew from Chu et al. (2024) that "SFT memorizes while RL generalizes," but there was no first-principles explanation for *why* the cross-entropy objective—the workhorse of supervised learning for decades—fails to produce generalizing solutions in the LLM fine-tuning regime. The inverse-probability weighting diagnosis provides that explanation: cross-entropy over-optimizes for exact reproduction of rare token patterns in the training data, many of which are noise, stylistic idiosyncrasies, or surface-form artifacts rather than recoverable reasoning patterns. RL methods, by contrast, operate with rewards that (when properly designed) don't inversely scale with model confidence, allowing them to avoid this overfitting trap.

The paper's contribution here is not the mathematics of importance sampling—which is standard—but rather the **application of this lens to diagnose a specific, previously unexplained failure mode of SFT.** It transforms "SFT generalizes poorly" from an empirical lament into a theoretically grounded statement about an ill-posed reward landscape, which in turn enables a principled correction rather than an ad-hoc fix.

---

### Innovation 2: Confidence-Weighting as a Unified Principle for Objective Design

The second contribution is a **shift in perspective on what loss functions should optimize in the LLM era.** The paper frames this explicitly through the contrast with Focal Loss (Section 2): Focal Loss downweights well-classified examples (`-(1-p)^γ log p`) to combat underfitting; DFT downweights poorly-classified examples (`-p log p`) to combat overfitting. The paper argues this inversion "reflects a fundamental shift in the LLM era: while underfitting was once a central challenge, overfitting and memorization now dominate, demanding a rethinking of objective design."

This isn't just a clever rhetorical move—it captures a genuine conceptual advance. The machine learning community has spent decades designing loss functions that combat underfitting: class balancing, hard example mining, focal loss, triplet loss, contrastive loss. All of these strategies share the assumption that the model needs *more* signal from the examples it finds difficult. DFT's core insight is that in the LLM fine-tuning regime, the opposite is often true: the model needs *less* signal from surprising tokens because surprise often indicates noise, stylistic variation, or patterns that don't generalize rather than genuinely important learning opportunities.

The paper makes this concrete through two empirical findings that support the conceptual claim:

**First, the probability distribution analysis (Figure 2) reveals that DFT produces a bimodal token distribution**—some tokens become very high-probability, others become very low-probability—rather than the uniform rightward shift produced by SFT. The tokens that end up in the low-probability bin are "conjunctive words or punctuations such as 'the', 'let', ',', '.' etc." (Section 4.7). This suggests DFT has implicitly learned that grammatical function words don't need to be predicted with high confidence—a form of **learned regularization** that SFT, with its uniform pressure to increase all token probabilities, cannot achieve. This finding provides a mechanistic explanation for DFT's better generalization: by not forcing the model to fit function words perfectly, DFT preserves capacity for the semantically meaningful tokens that actually matter for reasoning.

**Second, the Natural Questions case study (Section 4.5) reveals the boundary condition.** When the model lacks prior competence—as in factual knowledge tasks where the correct answer is genuinely novel information—DFT's confidence-weighting backfires (30.14% vs. SFT's 36.62%). The model's low confidence on unknown facts is *warranted*, but DFT interprets it as a signal to downweight, preventing the model from learning what it genuinely needs to learn. This negative result is as important as the positive ones because it defines DFT's domain of applicability: **confidence-weighting helps when the model has non-trivial prior competence and overfitting is the risk; it hurts when the model starts from ignorance and needs to absorb new information.** This boundary condition is not obvious from the mathematical derivation alone—it emerges from the empirical interaction between the weighting mechanism and the model's prior knowledge state.

This contribution is **fundamental rather than incremental** because it challenges a core assumption embedded in almost all loss function design from the pre-LLM era—that difficult examples deserve more optimization effort—and provides both theoretical motivation and empirical evidence for the opposite approach in the LLM fine-tuning setting. The fact that this insight manifests as a single-line code change doesn't diminish its conceptual significance; the simplicity is a consequence of the clarity of the diagnosis, not a sign of triviality.

---

### Innovation 3: Algebraic Reward Rectification as an Alternative to Explicit RL

The third innovation is a **practical demonstration that reward-aware training behavior can be achieved without any of the machinery of reinforcement learning**—no reward model, no preference pairs, no online sampling, no reference model, no value function, and no policy gradient estimation. DFT achieves this through pure algebraic manipulation of the SFT objective, exploiting the mathematical equivalence established in Innovation 1.

This matters because of the substantial gap between RL's theoretical advantages and its practical deployment costs. As the paper documents (Section 2), online RL methods like PPO and GRPO require "substantial computation, careful hyperparameter tuning, and explicit reward signals—conditions often impractical in real-world settings" (Section 1). Offline RL methods like DPO and RFT reduce some of these costs but still require preference pairs or rejection sampling. The paper's experimental results in Section 4.2 show DFT in an offline RL setting achieving 35.43 average accuracy across five math benchmarks, compared to 32.00 for GRPO and 28.66 for PPO on the same base model (Table 2). This is striking because DFT achieves this with:

- **No separate reward model:** the reward rectification is baked into the loss function
- **No preference pairs:** only positive demonstrations are needed
- **No reference model:** the model's own probabilities serve as the weighting signal
- **No online sampling:** training is entirely offline
- **No KL constraint or clipping:** the uniform scaling naturally prevents gradient explosions
- **No large batch sizes:** standard SFT batch sizes (256) are used

The significance here isn't that DFT achieves a new state-of-the-art on math reasoning (it doesn't; these are modest-scale models on standard benchmarks). Rather, the significance is that it **establishes a new point on the Pareto frontier of simplicity vs. performance**, demonstrating that reward-aware optimization—traditionally the exclusive domain of RL—can be achieved through objective function design alone, provided the design is grounded in a correct understanding of what the standard objective is actually computing.

This has direct implications for practitioners: it suggests that many of the benefits attributed to RL in the literature may be achievable through simpler means if the SFT objective is properly understood and corrected. It doesn't render RL obsolete—the paper is careful to note DFT's limitations on factual knowledge tasks and hard out-of-distribution problems—but it does narrow the gap and provide a much lower-cost option for the common case where only positive demonstrations are available.

The paper positions this as a bridge between the SFT and RL paradigms: "DFT achieves the same uniform-reward property [as RLVR, DeepSeek-AI et al., 2025] without requiring a separate verifier, reward model, or online sampling" (Section 3.3 and Appendix discussion). This is a **conceptual reframing**: it suggests that the SFT-RL dichotomy is partly an artifact of how the objectives are implemented rather than a fundamental divide, and that smarter objective design can capture RL-like properties within the supervised learning framework.

---

### Innovation 4: The Bimodal Probability Signature as a Diagnostic for Generalization

The fourth contribution is more subtle but methodologically important: the paper identifies a **specific, measurable signature in the model's output probability distribution that distinguishes generalizing from memorizing training**—what might be called a "generalization phenotype" at the token-probability level.

Figure 2 (Section 4.7) shows the token probability distributions on the training set after fine-tuning with different methods. SFT produces a uniform rightward shift: all tokens become more probable, with the largest gains in the lowest-probability bins. DFT produces a qualitatively different pattern: a **bimodal distribution** where some tokens become much more probable (pushing into the highest-probability bins) and others become less probable (pushing into the lowest-probability bins). The same bimodal pattern appears in the RL-trained models (DPO, GRPO, PPO), though at a smaller scale than DFT.

The insight here is that **generalization may require selective fitting rather than uniform fitting.** The model should become very confident about tokens that carry semantic content essential to the reasoning task, while allowing tokens that serve primarily grammatical or stylistic functions to remain at lower probability. SFT's uniform pressure to increase all probabilities prevents this selectivity, forcing the model to "waste" capacity and gradient signal on function words that don't generalize. RL methods achieve selectivity through their reward structure (correct answers are rewarded regardless of phrasing), and DFT achieves it through confidence-weighting (tokens the model is already uncertain about are downweighted, preventing overfitting to them).

This finding is significant beyond DFT itself because it provides a **diagnostic tool for evaluating fine-tuning methods**: methods that produce bimodal output distributions (like RL and DFT) may generalize better than methods that produce uniform distributions (like SFT), even when both achieve similar training loss. The paper doesn't develop this into a formal metric, but the observation opens a line of inquiry into whether probability distribution shape can serve as a proxy for generalization quality without requiring held-out evaluation sets—which is particularly valuable in domains where ground-truth evaluation is expensive or unavailable.

This contribution is **incremental in its current form** (it's an observation, not a fully developed diagnostic framework) but **potentially fundamental** if it generalizes beyond the specific models and tasks studied. The fact that the same bimodal pattern emerges across RL methods (DPO, GRPO, PPO) and DFT—despite their radically different training procedures—suggests it may reflect a general property of generalization in autoregressive language models, not an artifact of any particular algorithm. The paper's identification of the specific tokens relegated to the low-probability bin (conjunctions, punctuation, common function words) provides a concrete, interpretable mechanism for *why* selective fitting helps: by not forcing the model to perfectly predict "the" and "of" and ",", DFT preserves optimization capacity for the mathematical reasoning tokens that actually determine answer correctness.

---

### Innovation 5: The Implicit Reward Diagnosis as a Unifying Framework

The paper's most architecturally significant contribution is the **framework itself**: the demonstration that the SFT gradient can be exactly expressed as a policy gradient with an implicit reward, and that differences between SFT and RL can be understood as differences in the structure of this implicit reward. This framework unifies several previously disconnected observations under a single theoretical lens.

Prior to this work, the SFT-RL tradeoff was understood through disparate empirical findings: "SFT memorizes" (Chu et al., 2024), "RL generalizes" (Swamy et al., 2025), "SFT is needed as initialization" (Ouyang et al., 2022), "LLMs cannot self-correct through prompting alone" (Huang et al., 2023, cited in Section 2). Each finding made sense in isolation, but there was no framework explaining *why* these patterns hold across different tasks and models.

The implicit-reward framework provides this explanation by characterizing the optimization dynamics directly:

- **SFT's implicit reward** is `1[y=y*]/π_θ(y|x)`—sparse, inversely proportional to model confidence, and concentrated on exact expert reproductions. This explains both SFT's strength (it efficiently captures expert-like behavior by directly targeting the demonstration) and its weakness (it overfits to exact sequence matches, failing to generalize to novel phrasings or reasoning paths).

- **RL's explicit reward** (in typical RLVR/verification-based setups) is `1[answer is correct]`—still sparse but uniform across all correct outputs, without inverse-probability weighting. This explains RL's strength (it explores diverse strategies for reaching correct answers) and its cost (it requires sampling many trajectories and verifying correctness, which is expensive).

- **DFT's rectified implicit reward** is `π_θ(y*|x) · 1[y=y*]` with the `π_θ` factor cancelling the `1/π_θ` importance weight in the gradient, yielding an effective reward of 1 for all expert tokens regardless of model confidence. This positions DFT as an algebraic shortcut to RL-like behavior without RL's computational overhead.

This framework is **fundamental** because it doesn't just propose a new method—it provides a **language for reasoning about fine-tuning objectives** that is more precise than the existing SFT-vs-RL dichotomy. Under this framework, methods can be characterized by their implicit reward structure (sparse vs. dense, confidence-dependent vs. uniform, exact-match vs. correctness-based), and improvements to SFT can be designed by modifying this reward structure rather than by adding auxiliary losses or switching to explicit RL. The framework also generates testable predictions: methods that make the implicit reward more uniform and less confidence-dependent should generalize better, a hypothesis that DFT confirms and that future work can test on other objective modifications.

The paper doesn't fully exploit this framework's potential—it applies it to derive one correction (DFT) but doesn't systematically explore the space of possible reward structures—but the framework itself is a conceptual contribution that enables future research to reason about objective design in a more principled way.

## 5. Experimental Analysis

### Evaluation Methodology

- **Dataset.** The primary training dataset for mathematical reasoning is a random sample of 100,000 instances from NuminaMath-CoT (LI et al., 2024). For higher-quality data experiments (Appendix A.5), the paper uses OpenR1-Math-220k (Hugging Face, 2025), filtered by LUFFY (Yan et al., 2025) to approximately 45k prompts with verified correct reasoning traces from DeepSeek-R1. For offline RL experiments (Section 4.2), training data is generated by sampling 4 responses per question from the base model on 100,000 math questions and retaining only correct ones (approximately 140,000 examples). Code generation experiments (Section 4.3) use 10,000 prompts from UltraFeedback (Cui et al., 2024) with the highest-scoring response per prompt selected for training. Multi-modal experiments (Section 4.4) use the WeThink dataset (Yang et al., 2025). The factual knowledge case study (Section 4.5) uses the Natural Questions dataset (Kwiatkowski et al., 2019). Evaluation for math is on five held-out benchmarks: MATH500 (Hendrycks et al., 2021), Minerva Math (Lewkowycz et al., 2022), OlympiadBench (He et al., 2024), AIME 2024 (American Institute of Mathematics, 2024), and AMC 2023 (Mathematical Association of America, 2023). Code evaluation uses HumanEval (Chen et al., 2021), HumanEval+ (Liu et al., 2023), and MultiPL-E (Cassano et al., 2023) across 8 programming languages. Multi-modal evaluation uses MathVerse (Zhang et al., 2024b), MathVision (Wang et al., 2024), and WeMath (Qiao et al., 2024).

- **Base model(s).** The paper evaluates DFT on five model families spanning multiple scales: Qwen2.5-Math-1.5B and Qwen2.5-Math-7B (Qwen Team et al., 2024a), LLaMA-3.2-3B and LLaMA-3.1-8B (Dubey et al., 2024), and DeepSeekMath-7B (Shao et al., 2024). For code generation, the paper additionally uses Qwen2.5-3B, Qwen2.5-Coder-3B, and Qwen2.5-Coder-7B. For multi-modal reasoning, Qwen2.5-VL-3B is used. The choice of Qwen2.5-Math-1.5B as the primary model for detailed analysis (Figures 1-3, Tables 1-2) reflects its position as the smallest model with non-trivial math reasoning capability (base accuracy averaging 15.92 across five benchmarks), providing room for improvement while being computationally manageable for extensive ablation and comparison experiments. The LLaMA models are included to demonstrate cross-family generalization, and DeepSeekMath-7B tests DFT on a math-specialized architecture. The Qwen2.5-Math-7B experiments test scaling to larger models within the same family.

- **Metrics.** The primary metric throughout is **Average@16 accuracy**: the model generates 16 responses per question at temperature 1.0 (maximum 4096 tokens), and the reported number is the average correctness rate across these 16 runs. This is not pass@16 (the probability that at least one of 16 samples is correct) but rather the mean per-sample accuracy, which directly measures the model's expected performance when sampling a single answer. For math benchmarks, correctness is determined by the official Qwen2.5-Math evaluation pipeline's answer verification, which extracts and compares final answers. For code benchmarks, pass@1 is reported using standard evaluation harnesses. All tables report accuracy as percentages. The "Avg." column in Tables 1-2 is the arithmetic mean across the five math benchmarks.

- **Baselines.** The paper compares DFT against: **(1) Base model** (no fine-tuning), performance of the pretrained model with CoT prompting; **(2) Standard SFT** — supervised fine-tuning with cross-entropy loss on the same training data, using identical hyperparameters except for the loss function itself; **(3) Majority voting** — not explicitly tabled but mentioned in the text as trailing verifier-based methods; **(4) DPO** (Rafailov et al., 2023) — direct preference optimization, trained on 100,000 preference pairs in the offline RL setting (Section 4.2); **(5) RFT/RAFT** (Dong et al., 2023; Ahn et al., 2024) — rejection fine-tuning, standard SFT on filtered correct responses only; **(6) PPO** (Schulman et al., 2017) — online RL with a learned reward model, using verl framework; **(7) GRPO** (Shao et al., 2024) — group relative policy optimization, online RL with n=4 responses per prompt; **(8) iw-SFT** (Qin & Springenberg, 2025) — importance-weighted SFT, compared in Appendix A.4 as a concurrent method. For code and multi-modal experiments, only the base model and standard SFT are compared against DFT.

- **Generation budget / compute accounting.** The paper does not use a formalized "generation budget" or FLOPs comparison as a primary axis of evaluation. Instead, all methods are compared at equal training data and equal number of training steps. For the offline RL experiments (Table 2), DFT, DPO, and RFT all train on the same corpus of approximately 140,000 correct responses (or 100,000 preference pairs for DPO), while PPO and GRPO additionally sample responses online during training—this means the online methods consume more total compute, making DFT's advantage in accuracy a conservative estimate of its efficiency. Training hyperparameters are matched across SFT, DFT, and RFT: learning rates of 5 × 10⁻⁵ (or 2 × 10⁻⁵ for LLaMA-3.1-8B), batch size 256, cosine decay with 0.1 warmup ratio, AdamW optimizer. DPO uses a lower learning rate of 1 × 10⁻⁶ and batch size 128. PPO and GRPO use learning rate 1 × 10⁻⁶ and batch size 256. Experiments are run for 1 epoch on NuminaMath (100k examples) and 3 epochs on OpenR1-Math (45k examples).

- **Cross-validation / statistical protocol.** There is no cross-validation or statistical significance testing reported in the paper. All results are single-run final evaluations. The paper reports average accuracy across 16 decoding runs (temperature 1.0) to reduce sampling variance in the evaluation, but does not report standard deviations, confidence intervals, or standard errors for any metric. The absence of error bars in the convergence curves (Figure 1) and the lack of statistical testing for the differences between methods—particularly for small benchmarks like AIME 2024 (30 questions) and AMC 2023 (40 questions), where differences of a few percentage points can reflect a single additional correct answer—means that some of the smaller reported gains (e.g., Qwen2.5-Math-7B w/DFT vs. w/SFT on MATH500: 68.20 vs. 53.96) should be interpreted with appropriate caution regarding their statistical reliability. The paper does use a consistent evaluation framework (Qwen2.5-Math pipeline) across all math experiments, which ensures comparability but does not substitute for statistical testing.

---

### Main Quantitative Results

#### Mathematical Reasoning: DFT vs. Standard SFT Across Models and Benchmarks

Table 1 presents the core set of results: DFT consistently yields larger average improvements over base models than standard SFT across all five model families and all five math benchmarks. The headline numbers are the average accuracy gains in the rightmost column:

- **Qwen2.5-Math-1.5B:** DFT achieves 31.58 average accuracy, a +15.66 point gain over the base model's 15.92, while SFT achieves only 18.01 (+2.09 gain). DFT's improvement is approximately 7.5× larger than SFT's.
- **Qwen2.5-Math-7B:** DFT reaches 37.15 average (+15.90 over 21.25 base), while SFT reaches only 23.62 (+2.37 gain). DFT's improvement is approximately 6.7× larger.
- **LLaMA-3.2-3B:** DFT reaches 4.65 average (+3.46 over 1.19 base), SFT reaches 3.24 (+2.05 gain). DFT's improvement is approximately 1.7× larger.
- **LLaMA-3.1-8B:** DFT reaches 11.02 average (+10.02 over 1.00 base), SFT reaches 6.33 (+5.33 gain). DFT's improvement is approximately 1.9× larger.
- **DeepSeekMath-7B:** DFT reaches 18.15 average (+15.51 over 2.64 base), SFT reaches 9.82 (+7.18 gain). DFT's improvement is approximately 2.2× larger.

The most striking patterns emerge on the challenging benchmarks where standard SFT **degrades** performance compared to the base model. On OlympiadBench, Qwen2.5-Math-1.5B drops from 15.88 to 12.63 with SFT but rises to 27.08 with DFT—a swing of +11.20 over base where SFT is −3.25. On AIME 2024, Qwen2.5-Math-7B drops from 6.68 to 2.48 with SFT (−4.20) but improves to 8.56 with DFT (+1.88). On AMC 2023, Qwen2.5-Math-1.5B shows minimal change with SFT (19.38 to 18.75, −0.63) but nearly doubles with DFT (38.13, +18.75). These negative SFT results on difficult benchmarks are not isolated—they occur across multiple models and benchmarks (LLaMA-3.2-3B: AIME24 0.41 to 0.00 with SFT; LLaMA-3.1-8B: AIME24 0.21 to 0.00 with SFT), establishing a pattern of SFT-induced degradation on out-of-distribution reasoning tasks that DFT consistently reverses.

The variation in the relative advantage of DFT over SFT across model families reveals an interesting relationship with base model capability. DFT provides the largest gains over SFT on the most capable math models (Qwen2.5-Math series, DeepSeekMath-7B) and more modest gains on the general-purpose LLaMA models. This is consistent with the paper's hypothesis that DFT works by reinforcing existing competence: models with stronger math priors benefit more from confidence-based reweighting because their uncertainty signals are more informative about which tokens represent noise vs. genuine learning opportunities.

Figure 1 provides convergence curves for Qwen2.5-Math-1.5B on all five benchmarks, showing that DFT's advantage is not merely a final-performance effect but manifests in learning dynamics: **(1)** DFT reaches peak performance within approximately 120 training steps on most benchmarks, while SFT continues to fluctuate or slowly improve over 400 steps; **(2)** DFT's performance in the first 10-20 steps already exceeds SFT's best final accuracy on MATH500, Minerva Math, and AMC 2023; **(3)** DFT curves show a clear peak-then-plateau pattern, while SFT curves oscillate more, consistent with the hypothesis that DFT produces more stable gradient updates. The sharp early rise and plateau of DFT suggests that the method very rapidly identifies and reinforces the reasoning patterns the model already partially knows, after which additional training provides diminishing returns.

#### Offline RL Setting: DFT vs. Explicit RL Methods

Table 2 presents results in a setting designed to mimic RL data conditions—the training data consists of rejection-sampled correct responses from the base model itself, providing denser positive signal than the original demonstrations alone. The headline numbers:

- **DFT (offline):** 35.43 average accuracy across five benchmarks
- **GRPO (online):** 32.00 (DFT outperforms by +3.43)
- **PPO (online):** 28.66 (DFT outperforms by +6.77)
- **RFT (offline):** 23.97 (DFT outperforms by +11.46)
- **DPO (offline):** 23.20 (DFT outperforms by +12.23)
- **DFT in standard SFT setting** (Table 1, same model): 31.58

Several patterns are noteworthy. First, DFT in the offline RL setting (35.43) substantially outperforms DFT in the standard SFT setting (31.58), a gain of +3.85 points from using rejection-sampled correct responses rather than human demonstrations. This confirms that data quality matters for DFT as it does for any method, but the improvement is additive with DFT's algorithmic advantage.

Second, DFT outperforms both offline RL methods (RFT, DPO) by large margins (+11–12 points) despite all three methods training on the same static dataset. RFT trains with standard cross-entropy on the same filtered correct responses; its poor performance (23.97) compared to DFT demonstrates that simply providing higher-quality data is not sufficient—the loss function itself must be corrected to avoid the inverse-probability overfitting problem. DPO's even lower performance (23.20) despite access to both positive and negative examples is notable; the paper doesn't analyze this failure in detail, but it's consistent with recent findings that DPO can struggle with mathematical reasoning tasks where the preference signal is weak or where negative examples are not sufficiently informative.

Third, DFT outperforms online RL methods (GRPO, PPO) that have access to substantially more compute through online sampling. GRPO, the strongest online baseline, achieves 32.00 vs. DFT's 35.43. This is the paper's most surprising empirical claim: that a purely algebraic modification to the SFT loss—with no reward model, no online interaction, no preference pairs—can outperform the current standard approach to RL for language model math reasoning. The gap is largest on Minerva Math (25.16 vs. 18.93, +6.23) and AMC 2023 (48.44 vs. 41.25, +7.19), suggesting DFT's advantage is most pronounced on benchmarks that require generalization beyond the training distribution.

However, note that on AIME 2024, DFT (7.93) is slightly below GRPO (8.34), indicating that the advantage is not universal. The paper does not discuss this specific comparison or analyze what properties of AIME problems might favor GRPO's exploration over DFT's confidence-weighted fitting.

#### Code Generation: DFT Across Languages and Model Variants

Table 3 shows DFT's performance on code generation benchmarks. For **Qwen2.5-3B** (the general-purpose base), DFT raises HumanEval from 43.3 to 45.7 (+2.4) and HumanEval+ from 36.0 to 39.0 (+3.0), while SFT slightly degrades performance to 41.5 and 34.8 respectively. This is the same pattern seen in math: SFT degrades, DFT improves. On MultiPL-E averaged across 8 languages, DFT reaches 41.84 vs. 40.05 base and 39.10 SFT.

For **Qwen2.5-Coder-3B** (code-specialized), DFT provides larger gains: HumanEval 52.4 → 56.7 (+4.3 vs. SFT's −0.6), HumanEval+ 42.7 → 50.0 (+7.3 vs. SFT's +1.2). The MultiPL-E average improves to 53.16 from 48.39 base, with SFT at 50.52.

For **Qwen2.5-Coder-7B** (larger code-specialized), DFT achieves the largest absolute improvements: HumanEval 62.2 → 67.7 (+5.5 vs. SFT's −7.3 degradation), HumanEval+ 53.0 → 59.8 (+6.8 vs. SFT's −4.2). On MultiPL-E, DFT reaches 62.30 vs. 57.76 base, with SFT marginally at 57.62.

A striking pattern is that **SFT degrades performance on the most capable code models** (Qwen2.5-Coder-7B drops 7.3 points on HumanEval), while DFT consistently improves. This mirrors the math results where SFT degraded Qwen2.5-Math-7B on AIME24 by 4.20 points. The paper doesn't analyze this per-language breakdown in detail, but Table 3 shows that DFT's improvements are consistent across most programming languages, with particularly large gains in C# (Qwen2.5-Coder-3B: 47.20 → 58.39, +11.19) and Bash (Qwen2.5-Coder-7B: 39.24 → 48.73, +9.49).

#### Multi-Modal Reasoning: DFT Extends to Vision-Language Tasks

Table 4 shows results on multi-modal math reasoning benchmarks with Qwen2.5-VL-3B. The results are positive but more modest than the text-only math results, which is expected given that multi-modal reasoning involves visual perception capabilities that may not benefit from token-level confidence reweighting in the same way:

- **MathVerse Overall:** DFT reaches 37.54 vs. 33.83 base and 35.66 SFT (+3.71 over base, +1.88 over SFT)
- **MathVision:** DFT reaches 22.30 vs. 21.25 base and 21.02 SFT (+1.05 over base, +1.28 over SFT). Notably, SFT slightly degrades MathVision performance (21.25 → 21.02).
- **WeMath:** DFT reaches 23.71 vs. 4.10 base and 23.33 SFT (+19.61 over base, +0.38 over SFT). The near-identical performance to SFT on WeMath suggests that for this benchmark, the gains come primarily from exposure to training data rather than from DFT's reweighting mechanism.

The subcategory breakdown on MathVerse shows consistent DFT advantages across all three vision-reliance levels (Vision Only, Vision Intensive, Vision Dominant), with the largest relative gain on Vision Dominant problems (32.49 vs. 30.96 base, +1.53)—suggesting DFT helps even when visual reasoning is the primary bottleneck.

#### Learning Efficiency and Convergence

Figure 1 provides detailed convergence trajectories that support the claim of "faster convergence" and "better early-stage performance." Across all five benchmarks, DFT's curve is consistently above SFT's at every training step after the first few iterations, and DFT reaches its peak accuracy substantially earlier. On MATH500, DFT peaks around step 50-75 at approximately 64% and maintains that level, while SFT climbs slowly to about 44% at step 400 and is still rising. On AMC 2023, DFT reaches approximately 38% by step 25-50, while SFT oscillates between 15-20% throughout training. The AIME 2024 curve is the noisiest (as expected for a 30-question benchmark), but DFT maintains a consistent advantage over SFT throughout.

The paper does not report training loss curves, only evaluation accuracy. This means we cannot assess whether DFT's faster convergence in accuracy comes with faster convergence in loss, or whether DFT's loss landscape is fundamentally different (the weighted loss is not directly comparable to the unweighted loss, so training loss curves would need careful interpretation).

---

### Ablation Studies and Robustness Checks

**Token-level vs. sequence-level weighting (Section 4.6, Table 5):** The paper compares three variants of DFT on Qwen2.5-Math-1.5B: (1) **Full sentence-level weighting** using `sg(π_θ(y⋆|x))` as a single weight for the entire sequence—this produces average accuracy of 15.75, essentially unchanged from the base model's 15.92, and the authors report it "makes the loss nearly uninformative and produces a highly skewed weight distribution"; (2) **Geometric-mean weighting**, which rescales the sequence probability via its |y⋆|-th root (inspired by GSPO, Zheng et al., 2025) to avoid numerical underflow—this reaches 17.21, a marginal improvement over base but far below the token-level variant; and (3) **Token-level weighting** (the final DFT design)—this achieves 31.58, more than doubling the base accuracy. This is a decisive ablation: the token-level granularity is not a minor implementation detail but is essential for the method to function. The failure of the geometric-mean variant is particularly informative because it shows that numerical stability alone is not sufficient—the per-token confidence signal carries information that is lost when aggregated to a single sequence-level weight.

**Comparison with iw-SFT (Appendix A.4, Table 6):** DFT is compared against Importance-Weighted SFT (Qin & Springenberg, 2025) across all five model families. DFT outperforms iw-SFT on four of five models: LLaMA-3.2-3B (4.65 vs. 2.26, +2.39), LLaMA-3.1-8B (11.02 vs. 6.87, +4.15), DeepSeekMath-7B (18.15 vs. 14.81, +3.34), and Qwen2.5-Math-1.5B (31.58 vs. 30.28, +1.30). iw-SFT outperforms DFT on Qwen2.5-Math-7B (39.60 vs. 37.15, +2.45). The paper notes that iw-SFT incurs additional computational overhead by requiring a separate reference model to compute importance weights, "whereas DFT dynamically derives its own weighting directly from the token probabilities of model." The per-benchmark breakdown reveals inconsistency in iw-SFT: for LLaMA-3.2-3B, iw-SFT underperforms standard SFT on MATH500 (5.13 vs. 8.65) and AMC23 (2.03 vs. 3.13), while DFT consistently improves on both. Similarly, for LLaMA-3.1-8B, iw-SFT produces worse results than SFT on Minerva Math (4.31 vs. 5.78), while DFT reaches 8.26.

**DFT vs. iw-SFT in offline RL setting (Appendix A.4, Table 7):** In the offline RL setting on Qwen2.5-Math-1.5B, DFT achieves 35.43 vs. iw-SFT's 31.86, a gap of +3.57. The paper notes that iw-SFT's improvement from standard SFT setting (30.28) to offline RL setting (31.86) is only +1.58 points, while DFT's improvement from standard (30.67 in this comparison, though note this differs from the 31.58 in Table 1—likely a different run or data split) to offline (35.43) is +4.76 points. This suggests DFT more effectively leverages the additional signal from rejection-sampled correct responses.

**Training with higher-quality data (Appendix A.5, Table 8):** Using the filtered OpenR1-Math-220k dataset (approximately 45k prompts with DeepSeek-R1 reasoning traces), DFT on Qwen2.5-Math-1.5B achieves 38.19 average accuracy vs. SFT's 29.16 and the base model's 15.92. DFT provides an additional +9.03 gain over SFT, similar in magnitude to the +15.66 gain over base in the NuminaMath experiments (Table 1). This demonstrates DFT's effectiveness compounds with data quality: better training data improves both SFT and DFT, but DFT consistently extracts more value from the same data. The gap between DFT (38.19) and SFT (29.16) on OpenR1-Math is actually smaller in absolute terms than on NuminaMath (+9.03 vs. +13.57), which might suggest that higher-quality data partially mitigates the overfitting problem that DFT addresses, but DFT still provides substantial additional benefit.

**Parameter-efficient fine-tuning with LoRA (Appendix A.6, Table 9):** DFT with LoRA adapters (rank=8, alpha=16) on Qwen2.5-Math-1.5B achieves 32.90 average accuracy vs. 16.87 for LoRA SFT and 15.92 for the base model. On LLaMA-3.2-3B, LoRA DFT achieves 4.63 vs. 2.56 for LoRA SFT and 1.19 for the base model. The gains are comparable in magnitude to full-parameter training (Table 1: Qwen2.5-Math-1.5B full-parameter DFT reaches 31.58; LoRA DFT reaches 32.90—slightly higher, though this may be within sampling variance given the lack of error bars). This demonstrates DFT's compatibility with resource-constrained training settings.

**Learning rate and batch size sensitivity (Appendix A.7, Figure 3):** The paper sweeps learning rates {2e-4, 1e-4, 5e-5, 1e-5} and batch sizes {32, 64, 128, 256} on Qwen2.5-Math-1.5B. DFT consistently outperforms SFT across all configurations, ruling out the possibility that DFT's advantage is due to SFT having suboptimal hyperparameters. Both methods are somewhat sensitive to learning rate, with intermediate values (1e-4, 5e-5) performing best and extreme values (2e-4, 1e-5) degrading noticeably. Both methods are relatively insensitive to batch size across the range 32-256. The paper does not report optimal hyperparameters separately for DFT vs. SFT—both use the same learning rate in the main experiments—so there may be additional gains from DFT-specific tuning.

**Number of training epochs (Section 4.1 and Appendix A.5):** All main experiments train for 1 epoch on NuminaMath (100k examples at batch size 256 = approximately 390 steps). Figure 1 shows DFT plateauing at roughly 120 steps on most benchmarks, suggesting 1 epoch is sufficient and additional epochs would likely not improve DFT further and might cause overfitting. For OpenR1-Math (45k examples), training uses 3 epochs, which corresponds to a similar total number of optimizer steps. The paper does not present an epoch ablation, so we cannot determine whether DFT is more or less prone to overfitting with extended training compared to SFT—though the early plateau in Figure 1 suggests DFT would be relatively robust to additional epochs.

**Probability distribution analysis (Section 4.7, Figure 2):** The paper analyzes token probability distributions on the training set after fine-tuning with different methods. The base model's distribution is a reference point (not trained on this data). SFT produces a uniform rightward shift, with the largest increases in the lowest-probability bins (0.0-0.001 and 0.001-0.01), while the highest-probability bin (0.9-1.0) remains relatively unchanged. DFT produces a qualitatively different distribution: it significantly increases the proportion of tokens in both the highest (0.9-1.0) and lowest (0.0-0.0001) probability bins, creating a bimodal pattern. The RL methods (DPO, GRPO, PPO) show the same bimodal tendency as DFT but at a milder scale. The paper notes that tokens in the lowest-probability bin after DFT training are "generally the conjunctive words or punctuations such as 'the', 'let', ',', '.' etc."—this is not a quantitative analysis but an anecdotal observation, and the paper does not provide a systematic breakdown of which token categories end up in which bins.

**Factual knowledge limitation (Section 4.5):** On the Natural Questions dataset (open-domain factual QA), SFT improves Qwen2.5-Math-1.5B from 31.24% to 36.62%, while DFT degrades it to 30.14%—below the base model. This is the paper's key negative result and serves to establish a clear boundary condition: DFT helps on reasoning tasks where the model has substantial prior competence, but hurts on factual knowledge tasks where the model starts from ignorance and needs to absorb genuinely new information. The paper's explanation is that DFT's confidence-weighting "tends to reinforce the model's existing beliefs," which is counterproductive when those existing beliefs are wrong. This negative result appears in only a single paragraph with no table, no error bars, and no analysis of which subcategories of factual questions are most affected—a more thorough treatment would strengthen this important finding.

---

### Critical Assessment

#### What the Experiments Genuinely Demonstrate

The experiments convincingly demonstrate that **on mathematical reasoning benchmarks using 1.5B-7B parameter Qwen and LLaMA models fine-tuned on 100k NuminaMath examples, DFT consistently outperforms standard SFT**, often by large margins (3-5× larger improvement over base). This result is robust across five model families, five evaluation benchmarks, two data sources (NuminaMath and OpenR1-Math), and two training regimes (full-parameter and LoRA). Figure 1 shows the advantage is not a fluke of final checkpoint selection—DFT outperforms SFT at every training step on every benchmark. The SFT degradation results on difficult benchmarks (OlympiadBench, AIME 2024) are particularly compelling because they demonstrate a failure mode (overfitting causes worse generalization than no training at all) that DFT consistently reverses.

The experiments also demonstrate that **DFT can outperform explicit RL methods in an offline setting**, but this result carries important qualifications. Table 2 shows DFT (35.43) beating GRPO (32.00) and PPO (28.66) on Qwen2.5-Math-1.5B with 140k rejection-sampled training examples. However, this is a single model at a single scale, and the RL baselines use default hyperparameters that may not be optimal. The paper does not sweep GRPO's n parameter (number of responses per prompt), PPO's clipping threshold, or either method's KL penalty coefficient. Given RL's documented sensitivity to hyperparameters (Schulman et al., 2017; Sheng et al., 2025), it's possible that better-tuned RL baselines would close or eliminate the gap. The stronger claim—that DFT is categorically better than online RL for math reasoning—is not supported by this single experiment, and the paper does not make this claim in those terms, though the abstract's phrasing ("DFT achieves competitive results in offline RL settings") is appropriately measured.

The cross-domain experiments (code generation in Table 3, multi-modal reasoning in Table 4) demonstrate that DFT's benefits extend beyond pure math reasoning, but the gains are smaller and less consistent. On code generation, DFT helps most on the strongest models (Qwen2.5-Coder-7B) and can hurt slightly on weaker ones (Qwen2.5-3B's MultiPL-E average shows a small gain, but individual language results vary). On multi-modal benchmarks, DFT's edge over SFT is modest (MathVerse: +1.88, MathVision: +1.28, WeMath: +0.38), suggesting the token-level reweighting mechanism may be less impactful when the primary challenge is visual perception rather than linguistic reasoning. These results establish that DFT is broadly applicable, not domain-specific, but they also show that the magnitude of benefit is task-dependent in ways the paper does not fully characterize.

#### Where the Evidence Is Weaker or Incomplete

**Single-run evaluations without statistical testing.** Every number in every table is a single evaluation run (averaged over 16 decoding samples, but from a single training run). For the smaller benchmarks—AIME 2024 (30 questions), AMC 2023 (40 questions)—a difference of one correct answer corresponds to 3.3 and 2.5 percentage points respectively. The paper reports AIME 2024 differences like Qwen2.5-Math-7B w/DFT 8.56 vs. w/SFT 2.48, a difference of 6.08 points that could reflect as few as 2 additional correct answers. Without any measure of variance (error bars on Figure 1, standard deviations in Table 1, or multiple training seeds), we cannot assess whether these differences are statistically reliable or within the noise floor of training stochasticity.

**The 16-sample evaluation metric conflates accuracy and consistency.** Average@16 is an unusual metric. It measures the expected per-sample accuracy of the model, which is neither pass@k (the standard for reasoning benchmarks) nor maj@k (majority voting accuracy). A model that is consistently correct 60% of the time and a model that is correct on 100% of queries 60% of the time and 0% the other 40% will have the same Average@16 score, but they represent very different generalization behaviors. The paper's choice of this metric is not explained or justified. Since DFT is hypothesized to improve generalization by reducing overfitting to surface patterns, one might expect it to particularly improve pass@k (the model generates at least one correct solution more often) or maj@k (consensus is more often correct). The paper's metric choice may actually understate DFT's advantage if DFT's primary benefit is increasing the model's ability to generate correct solutions on problems it previously got entirely wrong.

**The offline RL comparison is narrow.** Table 2 compares DFT against DPO, RFT, PPO, and GRPO on a single model (Qwen2.5-Math-1.5B) with a single data configuration (4 samples per question, 100k questions). We don't know whether the advantage holds at other scales, with different data quantities, or with different sampling temperatures for response generation. The RL baselines use fixed hyperparameters without tuning. Additionally, the paper reports DFT in the offline RL setting as training on ~140k correct responses, but standard RFT also trains on this data and achieves only 23.97—why does the same data produce such different results under DFT vs. SFT? The explanation that DFT's loss function correction matters more than data quality is plausible, but the paper does not disentangle the effects of data filtering (only correct responses) from loss function modification.

**The Natural Questions negative result is underdeveloped.** Section 4.5 reports a single number (DFT: 30.14%, SFT: 36.62%, Base: 31.24%) with no table, no error characterization, and no analysis of what types of factual questions are most harmed. This is a missed opportunity, because understanding *when* DFT fails is as important as understanding when it succeeds. Does DFT degrade uniformly across all factual questions, or does it selectively harm questions where the model's prior is wrong? Does the degradation increase with training duration? Does a smaller learning rate mitigate the problem? The paper's one-paragraph treatment leaves these questions unanswered, and the finding's credibility is weakened by the absence of even basic experimental details (dataset split, evaluation protocol, number of examples).

**No investigation of training dynamics under DFT.** The paper provides convergence curves (Figure 1) but does not analyze *why* DFT converges faster and plateaus earlier. Is it because the effective learning rate is lower (since low-probability tokens receive smaller gradients)? Is it because the optimization landscape is fundamentally different (the gradient of `π` vs. the gradient of `log π`)? Is it because DFT avoids the gradient spikes that cause SFT's oscillations? The gradient analysis in Appendix A.3 provides the mathematical equivalence `∇L_DFT = −∇π` vs. `∇L_CE = −(1/π)∇π`, but this analysis is not connected to the observed convergence behavior. Without gradient norm statistics, loss landscape visualizations, or analysis of which parameters change most, we cannot distinguish between DFT genuinely finding better minima vs. simply taking smaller steps.

**No ablation on the stop-gradient operator.** The `sg(·)` is identified as "critical" in the paper's technical description (Section 3.3), but there is no empirical ablation showing what happens without it. The paper states that without stop-gradient, the loss becomes `−(π log π)`, which is a different optimization objective, but we don't know whether this variant performs worse than standard SFT, comparably, or differently. This is a glaring omission for a method whose entire novelty is the insertion of a stop-gradient-wrapped multiplicative factor.

**Limited scale of investigation.** All experiments use models in the 1.5B–8B parameter range. We don't know whether DFT's advantages persist, diminish, or reverse at larger scales (13B, 70B, 405B). The extrapolation from 1.5B to deployment-scale models is precarious because the relationship between model confidence and token informativeness likely changes with scale—larger models may have better-calibrated uncertainties, making the inverse-probability distortion less severe, or they may have more severe overfitting because their larger capacity allows them to memorize rare patterns more effectively.

**What would strengthen the paper.** The following experiments would substantially increase confidence in the claims:

- **Multiple training seeds** with standard deviations or confidence intervals on all tabled results, particularly for the small benchmarks (AIME, AMC).
- **Pass@k and maj@k metrics** alongside Average@16 to characterize whether DFT improves the model's ability to generate correct solutions (pass@k) or its consistency (maj@k) vs. its per-sample accuracy.
- **Gradient norm statistics during training** comparing DFT and SFT to directly test the claim that DFT produces more stable, less spike-prone gradients.
- **Ablation of the stop-gradient operator** comparing `−sg(π) log π` vs. `−π log π` (no stop-gradient) vs. standard `−log π` to establish that the stop-gradient is necessary.
- **Scaling experiments** at 13B or larger to test whether DFT's advantage grows, shrinks, or holds constant with model scale.
- **Hyperparameter sweeps for RL baselines** in Table 2 to ensure the comparison is between well-tuned versions of each method.
- **A more thorough factual knowledge study** with per-category breakdown, training duration sweeps, and learning rate sensitivity to map the boundary conditions of DFT's applicability.

#### Do the Experiments Support the Paper's Central Claims?

The paper's central claim is that standard SFT's cross-entropy loss encodes an ill-posed reward structure (the `1/π` inverse-probability weight) that causes overfitting and poor generalization, and that DFT corrects this via `−sg(π) log π`, producing more stable training and better generalization.

**The mathematical equivalence is established analytically, not empirically.** The derivation in Sections 3.2 and Appendix A.2 is algebraic—it does not depend on experimental validation. The claim that the SFT gradient equals a policy gradient with importance weight `1/π` is mathematically true under the definitions given. However, whether this mathematical equivalence explains the observed generalization gap is a causal claim that the experiments do not directly test. The experiments show that DFT (which was designed based on this analysis) outperforms SFT, but this is consistent with multiple possible explanations: the `1/π` diagnosis might be correct, or DFT might work for unrelated reasons (e.g., it effectively reduces the learning rate on uncertain tokens, which acts as a regularizer independently of any RL interpretation), or SFT might have hyperparameters that are suboptimal for its own objective. The convergence curves (Figure 1) and hyperparameter sweeps (Figure 3) partially address the third possibility but cannot distinguish between the first two. A more direct test of the theoretical claim would involve manipulating the `1/π` term independently of DFT—for instance, by clipping gradient norms per-token based on `1/π`, or by adding an explicit inverse-probability penalty to an otherwise well-behaved objective—and showing that this recovers SFT-like degradation.

**The generalization claim is supported with appropriate benchmarks but the definition of "generalization" is narrow.** The paper's benchmarks (MATH500, Minerva Math, OlympiadBench, AIME 2024, AMC 2023) all measure mathematical reasoning, which is a specific type of generalization—the ability to solve new math problems that require similar reasoning patterns to the training data. The paper does not test whether DFT improves generalization in the sense of transfer to entirely different task families (e.g., from math to code, from English to other languages, from reasoning to factual recall). The code and multi-modal experiments (Tables 3 and 4) extend the domain slightly but still test the same type of generalization (within-domain reasoning on held-out problems). This is not a weakness per se—most fine-tuning papers evaluate on held-out benchmarks within the same domain—but it means the claim "DFT improves generalization" should be understood as "DFT improves within-domain generalization on reasoning tasks," not "DFT produces more general-purpose language models."

**The efficiency claims are well-supported but context-dependent.** DFT achieves its gains with essentially no additional computational cost (a single `.detach()` call per token). The comparisons in Table 2 against online RL methods that require substantially more compute (sampling, reward evaluation, multiple training phases) make DFT's efficiency advantage unambiguous. However, the claim that DFT's gains are "several times larger than standard SFT" (e.g., 7.5× for Qwen2.5-Math-1.5B) should be understood as a ratio of improvement-over-base, not a ratio of absolute accuracy. Both methods start from the same base accuracy; DFT adds more points, but the "7.5×" figure compares the magnitude of improvement, not the final performance. On Qwen2.5-Math-1.5B, DFT achieves 31.58% vs. SFT's 18.01%—a meaningful but not transformative absolute difference on a task where ceiling performance is much higher.

**The RLVR connection is conceptually elegant but empirically untested.** The paper argues that DFT's rectified reward becomes uniformly 1 for all expert trajectories, making it "akin to" RLVR (DeepSeek-AI et al., 2025). This is an interesting conceptual parallel, but the paper does not compare DFT against an actual RLVR implementation (e.g., training with GRPO and verification rewards on the same data). Table 2 compares DFT against GRPO, but GRPO in that experiment uses the standard setup, not an RLVR-specific configuration. The claim that DFT captures RLVR-like properties through algebraic manipulation is a theoretical connection that would be strengthened by direct empirical comparison.

#### Summary Assessment

The experiments robustly demonstrate that DFT outperforms standard SFT on mathematical reasoning fine-tuning across multiple models and benchmarks, that it converges faster, and that it can outperform both offline and online RL methods in specific settings. The results are consistent enough across models and benchmarks to be convincing about DFT's practical value for reasoning tasks. However, the paper's deeper theoretical claim—that the `1/π` inverse-probability weighting is the causal mechanism behind SFT's generalization failures—is not directly tested by the experiments, and the boundary conditions established by the Natural Questions negative result are thin (a single number in a paragraph). The paper succeeds as a practical contribution (here is a simple, effective modification to SFT that consistently helps on reasoning tasks) but only partially succeeds as a theoretical contribution (here is *why* SFT fails and *why* DFT fixes it), because the experiments demonstrate that DFT works without fully discriminating between the proposed mechanism and alternative explanations.

## 6. Limitations and Trade-offs

### 6.1 Sharp Boundary at Problem Difficulty: No Improvement on Hard Problems

**The assumption or constraint.** DFT—and indeed all test-time and fine-tuning strategies studied—fundamentally depends on the base model possessing non-trivial prior competence on the target task. The paper's mathematical diagnosis (Section 3.2) identifies the `1/π_θ` distortion as the core pathology, and DFT's correction works by suppressing gradients for tokens where the model is uncertain. This mechanism is beneficial when uncertainty signals noise or non-generalizable patterns, but it becomes counterproductive when the uncertainty reflects genuine ignorance—the model *should* receive strong gradients on concepts it doesn't yet know.

The paper is explicit about this boundary condition in Section 5:

> "DFT can not offer universal benefits across all scenarios. In domains that primarily involve the acquisition of factual knowledge, conventional SFT still remains the most efficient approach. DFT may also not be an ideal choice for hard examples or domains under-represented in the training data, since it assigns low initial probabilities to such samples, reducing their learning weight."

**The consequence.** On problems that lie substantially beyond the base model's current capabilities, DFT's confidence-weighting suppresses precisely the training signal the model needs to learn. The paper does not characterize this as a gradual degradation—it appears to be a sharp boundary. For math reasoning (where models start with non-trivial competence—even Qwen2.5-Math-1.5B scores 15.92 average), DFT helps substantially. For factual knowledge on Natural Questions (Section 4.5), DFT **degrades performance below the untrained base model** (30.14% vs. 31.24% base), while SFT improves it (36.62%). This implies that practitioners deploying DFT must know in advance whether their task falls on the "reasoning/competence" side or the "factual knowledge/ignorance" side of this boundary—and misclassification means DFT performs worse than no fine-tuning at all.

Even within the reasoning domain, the difficulty gradient matters. On the hardest math problems (AIME 2024, AMC 2023 for smaller models; implicitly difficulty bin 5 if the paper had used the binning approach from the reference example), DFT provides smaller absolute gains or is outperformed by other methods. In the offline RL comparison (Table 2), GRPO slightly outperforms DFT on AIME 2024 (8.34 vs. 7.93), suggesting that on the hardest problems, explicit exploration via RL may be more effective than confidence-weighted supervised learning.

**What evidence exists in the paper.** Section 4.5 provides a single-paragraph case study on Natural Questions, reporting only aggregate accuracy with no error characterization, no breakdown by question type, no sweep of training duration or learning rate, and no analysis of *which* factual questions DFT most harms. The paper does not investigate whether the degradation can be mitigated by adjusting the weighting scheme (e.g., a temperature parameter on the confidence weight, or a lower bound to prevent complete suppression of low-probability tokens). The pattern across math benchmarks (Table 1) also shows that DFT's relative advantage over SFT varies with base model capability—larger gains on Qwen2.5-Math series (strong math priors) than on LLaMA series (general-purpose models)—consistent with the competence-dependence hypothesis, but the paper does not systematically analyze this relationship.

**Mitigation status.** The paper does not propose any mitigation for this limitation. It identifies the boundary condition as a limitation in Section 5 and leaves it as a constraint on DFT's applicability. The paper states its aim is "not to assert that DFT universally outperforms SFT, but rather to offer a new perspective on objective design" (Section 5). A natural mitigation—not explored—would be to make the confidence weight tunable via a temperature parameter `τ` (using `π^{1/τ}` rather than `π`) to control the degree of suppression, or to set a minimum weight floor to ensure even very surprising tokens receive some gradient signal. These are obvious extensions that the paper does not investigate.

---

### 6.2 No Direct Empirical Validation of the Causal Mechanism

**The assumption or constraint.** The paper's theoretical contribution rests on a specific causal claim: standard SFT generalizes poorly *because* the implicit `1/π_θ` inverse-probability weight creates an ill-posed reward landscape that causes overfitting to rare exact-match token patterns (Section 3.2). DFT's design follows algebraically from this diagnosis—cancel the `1/π_θ` by multiplying by `π_θ`—and the experiments demonstrate that DFT outperforms SFT. However, the experiments are **efficacy demonstrations**, not **mechanism tests**. They show that DFT works, but they do not discriminate between the proposed mechanism and alternative explanations that would also predict DFT's superiority.

**The consequence.** Multiple alternative mechanisms could explain DFT's empirical advantage, and the paper provides no evidence to distinguish among them:

1. **Effective learning rate reduction:** DFT's gradient `−∇π` is always smaller in magnitude than cross-entropy's `−(1/π)∇π` for `π < 1`. In practice, DFT's average gradient norm across a batch will be substantially smaller than SFT's. This alone could explain faster convergence (no gradient spikes to recover from) and better generalization (implicit regularization from smaller effective step sizes), with no need to invoke the RL reward-structure interpretation. The paper's learning rate sweep (Figure 3, left) shows DFT and SFT both peak at similar learning rates and degrade at extremes, but this doesn't isolate the gradient norm effect from the weighting effect.

2. **Noise suppression:** Low-probability tokens may genuinely be noise (formatting artifacts, stylistic quirks, annotation errors) rather than informative learning signals. DFT downweights them, which acts as automatic noise filtering. This is consistent with the bimodal probability distribution in Figure 2 (function words relegated to low probability) but doesn't require the `1/π_θ` distortion diagnosis—any method that downweights uncertain tokens would have a similar effect.

3. **Implicit curriculum learning:** DFT effectively creates an automatic curriculum where the model first masters high-confidence tokens and gradually addresses harder ones as confidence grows. This is a known beneficial property of adaptive weighting schemes independent of any RL interpretation.

The paper's theoretical framework is elegant, but **elegance is not evidence**. Without experiments that manipulate the `1/π_θ` term independently of DFT—for instance, by manually injecting `1/π` amplification into an otherwise well-behaved objective and showing it degrades performance to SFT-like levels, or by testing a method that achieves uniform gradient scaling through a different mechanism (e.g., gradient clipping per-token based on `1/π`) and showing similar benefits—the causal claim remains an interpretation, not an established fact.

**What evidence exists in the paper.** No experiment directly tests the causal mechanism. The probability distribution analysis (Figure 2) shows DFT produces a bimodal pattern similar to RL methods, which is consistent with the reward-rectification interpretation but also consistent with alternative explanations. The gradient analysis in Appendix A.3 derives `∇L_DFT = −∇π` vs. `∇L_CE = −(1/π)∇π`, establishing the mathematical difference, but the paper does not measure gradient norms during training or correlate gradient magnitude with per-token probability to verify that the `1/π` amplification is indeed the source of SFT's instability. The comparison with iw-SFT (Appendix A.4) partially addresses this—iw-SFT uses importance weights from a reference model rather than from the current model, and DFT outperforms it on 4 of 5 models—but this comparison confounds multiple differences (weight source, weight computation, stop-gradient usage) and cannot isolate the mechanism.

**Mitigation status.** The paper does not acknowledge this as a limitation. The theoretical derivation is presented as sufficient justification for DFT's design, and the experiments are presented as validation of the theory, without discussion of alternative explanations. This is a significant gap because the paper's primary claimed contribution is theoretical ("we mathematically establish LLM SFT as a special RL in policy gradient space"—Section 1); if the theoretical mechanism is not empirically validated, the contribution reduces to "we found a loss function modification that works well empirically," which is a useful engineering finding but not the fundamental advance the paper claims. Future work that ablates the mechanism—e.g., by showing that gradient clipping based on `1/π` recovers DFT-like behavior without the RL interpretation—would substantially strengthen or refute the theoretical claims.

---

### 6.3 Single Training Run, No Statistical Characterization of Results

**The assumption or constraint.** All experimental results in the paper come from **single training runs** evaluated once (with 16 decoding samples per benchmark question to reduce sampling variance at evaluation time). The paper reports no standard deviations, confidence intervals, standard errors, or any other measure of statistical reliability for any number in any table or figure. The convergence curves in Figure 1 show no error bars or shaded regions, making it impossible to assess whether DFT's advantage over SFT at any given training step is within or beyond typical run-to-run variance.

**The consequence.** For the smaller benchmarks, this is particularly acute. AIME 2024 has 30 questions and AMC 2023 has 40 questions (the paper does not explicitly state these sizes, but they are standard for these benchmarks). On AIME 2024, a single additional correct answer changes accuracy by 3.33 percentage points; on AMC 2023, by 2.5 points. Many of the paper's reported differences fall in this range or close to it, particularly for the harder benchmarks where absolute accuracies are low:

- **Qwen2.5-Math-7B on AIME 2024:** DFT 8.56 vs. SFT 2.48 (difference: 6.08 points, ~2 additional correct answers)
- **LLaMA-3.1-8B on AIME 2024:** DFT 0.41 vs. SFT 0.00 (difference: 0.41 points, may be 0 or 1 additional correct answer depending on rounding)
- **DeepSeekMath-7B on AIME 2024:** DFT 1.24 vs. SFT 0.41 (difference: 0.83 points)

The absence of error characterization means readers cannot distinguish between genuine improvements and run-to-run noise for these small-benchmark comparisons. The aggregate "Avg." column partially mitigates this by averaging across five benchmarks, but the individual benchmark results are the basis for the paper's claims about generalization to challenging problems, and those claims carry substantial uncertainty that is not quantified.

The absence of multiple training seeds also means we cannot assess whether DFT's faster convergence (Figure 1) is robust or whether SFT sometimes catches up given a different random initialization. The paper trains only one seed per configuration per model, which is standard practice in LLM fine-tuning research (due to computational cost) but is a real limitation when claiming that one method systematically outperforms another.

**What evidence exists in the paper.** None. The paper never mentions this limitation, never reports error bars, and never discusses statistical significance. The reproducibility statement (after Section 5) describes how to reproduce the experiments but does not address variance. Given the computational cost of training these models (even 1.5B parameters on 100k examples), running multiple seeds for every configuration would be expensive, but even 3 seeds for the primary comparison (Table 1, Qwen2.5-Math-1.5B) would allow basic characterization of variance and substantially increase confidence in the headline results.

**Mitigation status.** Not addressed. The paper provides no estimates of variance, no discussion of the issue, and no suggestions for how future work might address it (e.g., reporting min/max across seeds, using larger evaluation sets, or applying statistical tests that account for small benchmark sizes). A reader deciding whether to adopt DFT based on these results must simply trust that the reported differences are reliable, which is particularly problematic for the AIME and AMC results given their small sizes.

---

### 6.4 Evaluation Metric (Average@16) Obfuscates Generalization Behavior

**The assumption or constraint.** The paper evaluates all math reasoning experiments using **Average@16 accuracy**—the mean correctness rate across 16 independently sampled answers per question at temperature 1.0. This metric measures the model's expected per-sample accuracy but conflates two distinct aspects of model capability: **(1) coverage**—what fraction of problems can the model solve at least once in k attempts (measured by pass@k), and **(2) consistency**—how reliably does the model produce correct answers on problems it knows (measured by the ratio of pass@1 to pass@k).

DFT's hypothesized mechanism—reducing overfitting to rare token patterns in training data—should primarily improve **coverage**: by not forcing the model to memorize surface-form statistics, DFT should enable it to solve problems it would otherwise fail entirely, even if it only gets them right occasionally (low consistency but high coverage). Standard SFT, by contrast, may produce high consistency on a narrow set of problems it has effectively memorized, but low coverage on genuinely novel problems. Average@16 cannot distinguish these scenarios. A model that scores 100% on half the benchmark and 0% on the other half has the same Average@16 as a model that scores 50% uniformly across all problems, but they represent fundamentally different capabilities.

**The consequence.** The paper's metric choice may **understate DFT's advantage** if DFT primarily improves coverage (solving previously unsolvable problems, even inconsistently) rather than consistency. Alternatively, it may **overstate DFT's advantage** if DFT produces less consistent but more broadly capable behavior—a practitioner who values reliability (high pass@1 on the problems the model attempts) might prefer SFT's narrower but more consistent performance profile. Without pass@k and consistency metrics, readers cannot assess which aspect of model behavior DFT improves, making it difficult to predict whether DFT will be beneficial for their specific deployment scenario.

This is particularly relevant for the paper's comparison with RL methods (Table 2). RL is known to improve coverage at the expense of consistency (it explores diverse strategies, some of which work), while SFT is known to produce consistent but narrow behavior. If DFT produces RL-like coverage but SFT-like consistency, it would represent a genuinely novel combination, but Average@16 cannot reveal this.

**What evidence exists in the paper.** None. The paper reports only Average@16 for math and pass@1 for code (Table 3 uses standard pass@1, which is the correct metric for code benchmarks). The paper does not discuss the choice of Average@16, justify it relative to standard alternatives (pass@k, maj@k), or report alternative metrics that would characterize coverage vs. consistency. Figure 2 (probability distributions) provides indirect evidence that DFT produces qualitatively different output behavior than SFT (bimodal vs. uniform distributions), which is consistent with DFT affecting coverage more than consistency, but this is not quantified.

**Mitigation status.** Not addressed. The paper does not acknowledge this as a limitation or discuss the metric's implications for interpreting results. Reporting pass@k for k ∈ {1, 4, 16, 64} alongside Average@16, or reporting consistency as pass@1 / pass@16, would provide a much clearer picture of DFT's effect on model behavior without requiring additional experiments (the same 16 samples can be used to compute all these metrics). This is a low-cost improvement that would substantially increase the informativeness of the experimental results.

---

### 6.5 Zero Investigation of the Stop-Gradient Operator's Necessity

**The assumption or constraint.** The `sg(·)` (stop-gradient) operator is identified as theoretically essential to DFT's mechanism (Sections 3.3, Appendix A.3). The paper's derivation shows that without `sg`, the loss becomes `−π log π` rather than `−sg(π) log π`, and the gradient becomes `−(log π + 1)∇π` rather than `−∇π`. These are fundamentally different optimization objectives: `−sg(π) log π` maximizes `π` directly; `−π log π` maximizes `π log π`, which has a maximum at `π = 1/e ≈ 0.37` and penalizes probabilities *above* this value—meaning a model trained without stop-gradient would be actively discouraged from becoming too confident, even on tokens it should learn perfectly.

The paper states in Section 3.3: "The stop-gradient operator sg(·) denotes the stop gradient operator, ensuring that gradients do not flow through the reward scaling term w." And in Appendix A.3: "the detached probability sg(π_θ(y⋆| x)) is treated as a constant during differentiation."

**The consequence.** We have no empirical evidence that the stop-gradient is actually necessary for DFT's performance. It is possible that `−π log π` (no stop-gradient) performs similarly to `−sg(π) log π` in practice—for instance, if the `(log π + 1)` factor provides a beneficial regularizing effect, or if the difference is small for the probability ranges encountered during training. Alternatively, it is possible that `−π log π` performs substantially worse (as the paper's theory predicts) because the `π > 1/e` penalty would prevent the model from learning high-confidence predictions. Without this ablation, we cannot assess whether the stop-gradient—which is the defining implementation detail distinguishing DFT from a naive "multiply by probability" approach—is actually load-bearing.

This matters for two reasons. First, **implementation correctness**: practitioners implementing DFT must know whether `.detach()` is essential or optional. If it's essential (as the paper's theory claims), omitting it would produce a different and potentially harmful method; if it's optional, the method is even simpler than advertised. Second, **theoretical validation**: if `−π log π` without stop-gradient performs comparably to `−sg(π) log π`, the paper's entire theoretical framework (which motivates the stop-gradient as critical for cancelling the `1/π` distortion without introducing additional gradient terms) is called into question, because the framework would predict a clear performance difference that fails to materialize.

**What evidence exists in the paper.** None. The paper provides no ablation comparing `−sg(π) log π` against `−π log π` (no stop-gradient) or against any intermediate variant (e.g., a convex combination of the two). The theoretical analysis in Appendix A.3 derives the different gradient forms but does not empirically test their consequences. This is arguably the single most important ablation for a method whose entire novelty is the insertion of a stop-gradient-wrapped multiplicative factor, and its absence is a significant gap.

**Mitigation status.** Not addressed. The paper does not acknowledge the need for this ablation, does not report it, and does not list it as future work. This is a self-contained experiment requiring no new data, models, or infrastructure—simply removing the `.detach()` call and rerunning the training—which makes its absence particularly notable.

---

### 6.6 Unquantified Practical Overhead for the Practitioners Who Most Need DFT

**The assumption or constraint.** DFT is positioned as a solution for practitioners who lack access to reward models, preference data, or the computational resources for online RL. The paper emphasizes this repeatedly: "SFT remains the only viable option when datasets contain only positive demonstrations, with no negative samples or reward model available" (Section 1), and DFT "requires neither a reference model nor large batch sizes" (Section 1). These practitioners are implicitly assumed to have the resources for full-parameter fine-tuning on 100k+ examples with standard optimizer configurations.

**The consequence.** The paper does not quantify DFT's actual computational cost relative to SFT, nor does it investigate whether DFT's benefits survive in the resource-constrained settings its target users face. DFT adds a `.detach()` call per token, which is negligible in compute—but the paper reports no wall-clock timing, memory usage, or throughput comparisons. More importantly, DFT's primary empirical results (Table 1) use full-parameter fine-tuning on 100k NuminaMath examples, which requires substantial GPU memory and compute (fine-tuning a 7B model on 100k sequences is non-trivial for individual practitioners or small labs). The LoRA experiments (Appendix A.6, Table 9) partially address this, showing DFT works with parameter-efficient fine-tuning, but they use the same 100k-example dataset as full-parameter training.

The paper also does not investigate whether DFT's benefits depend on dataset size. The convergence curves (Figure 1) show DFT plateauing at ~120 steps on 100k examples (batch size 256, so ~30k examples seen), suggesting DFT may work well with smaller datasets, but this is not tested. The OpenR1-Math experiments (Appendix A.5) use only 45k examples, but this dataset has substantially higher quality (DeepSeek-R1 reasoning traces), confounding dataset size with data quality. A practitioner with 1,000 high-quality demonstrations—a common scenario in specialized domains—has no guidance on whether DFT will help.

**What evidence exists in the paper.** The LoRA experiments (Appendix A.6, Table 9) show DFT works under parameter-efficient fine-tuning with rank=8 and alpha=16, achieving results comparable to full-parameter training (32.90 vs. 31.58 for Qwen2.5-Math-1.5B). This is encouraging but limited to a single LoRA configuration on a single model. The paper does not sweep LoRA rank, test other PEFT methods (e.g., prompt tuning, adapters), or vary dataset size. No timing or memory measurements are reported anywhere in the paper. The convergence speed advantage (DFT peaks at ~120 steps vs. SFT still improving at 400 steps) suggests DFT could be trained for fewer steps, reducing cost, but the paper does not explicitly recommend or evaluate this.

**Mitigation status.** The LoRA experiments partially address the resource constraint concern, but the paper does not systematically investigate DFT's performance as a function of dataset size, model size, or training budget. The limitation section (Section 5) acknowledges that "the evaluation scope remains limited" and that the paper has "not yet assessed its performance on broader task categories or with larger-scale LLM," but does not specifically address the question of whether DFT helps at the small-data, small-compute scales most relevant to its target users. A practitioner with a 1B model and 500 examples cannot determine from this paper whether DFT is likely to help, hurt, or have no effect.

Additionally, the Natural Questions result (Section 4.5) establishes that DFT can actively harm performance on some tasks. This creates a **vetting burden**: practitioners must determine whether their task falls on the "reasoning" or "factual knowledge" side of DFT's competence boundary before adopting it, but the paper provides no diagnostic for making this determination beyond the broad task category. A mixed-domain dataset (e.g., an instruction-tuning corpus containing both reasoning and factual queries) might benefit from DFT on some examples and be harmed on others—the paper provides no guidance for this common scenario.

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
