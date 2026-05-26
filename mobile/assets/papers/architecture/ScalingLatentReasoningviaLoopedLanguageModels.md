# Scaling Latent Reasoning via Looped Language Models

**ArXiv:** [2510.25741](https://arxiv.org/abs/2510.25741)

## 🎯 Pitch

Ouro introduces Looped Language Models (LoopLMs), a breakthrough architecture that weaves iterative, non-textual reasoning directly into pre-training by repeatedly applying the same stack of Transformer layers. This enables models with just 1.4B–2.6B parameters to match or outperform much larger (4B–8B+) LLMs on challenging reasoning and math tasks, drastically improving parameter efficiency while enabling adaptive compute at inference. By providing deep, faithful internal reasoning with lower resource demands and built-in safety gains, LoopLMs mark a new direction for scaling large language models beyond sheer parameter counts—unlocking powerful AI under practical constraints.

---

## 1. Executive Summary

This paper introduces **Ouro**, a family of pre-trained **Looped Language Models (LoopLM)** that build iterative latent computation directly into pre-training rather than deferring reasoning to post-training chain-of-thought, demonstrating that a 1.4B and 2.6B parameter model—trained on 7.7T tokens with shared-weight transformer blocks applied recurrently—can match or exceed the performance of standard transformers up to 8–12B parameters across reasoning benchmarks. The architecture incorporates an **entropy-regularized objective** for learned depth allocation (preventing collapse to always using the maximum number of recurrent steps via a KL penalty toward a uniform prior over exit steps) and an **adaptive exit gate** (a per-step halting probability that enables early termination on simple inputs while allocating more iterations to complex ones). The headline result is 2–3× parameter efficiency: the 1.4B Ouro model with 4 recurrent steps achieves 67.4 on MMLU and 82.4 on MATH500, matching Qwen3-4B's 73.2 and 59.6 respectively, while the 2.6B variant reaches 90.9 on MATH500—surpassing Qwen3-8B's 62.3—establishing that looped recurrence amplifies knowledge manipulation capability without increasing raw knowledge storage capacity, as confirmed by controlled synthetic experiments showing looped and non-looped models both saturate at approximately 2 bits per parameter.

## 2. Context and Motivation

### The Core Problem: Decoupling Compute Depth from Parameter Count

The central tension this paper addresses is structural: for decades, the primary path to better language model performance has been **adding more parameters**—stacking more transformer layers, widening hidden dimensions, increasing attention heads—which directly multiplies both training cost and deployment footprint. A model with 70B parameters costs roughly 10× more to serve than a 7B model, requires proportionally more memory, and demands more expensive hardware. This parameter-to-capability coupling creates an uncomfortable tradeoff: the models that perform best on complex reasoning tasks are also the least accessible for real-world deployment.

The paper articulates this directly in Section 1:

> "deploying models with hundreds of billions of parameters requires extensive infrastructure, increasing latency and cost while limiting accessibility. These factors make parameter efficiency critical: achieving better model capability within a fixed parameter budget."

This framing is important because it distinguishes parameter efficiency from mere compression. The goal is not to make a small model that's "good enough" by sacrificing capability—it's to make a small model that genuinely competes with much larger ones on the same tasks. The paper claims a 2–3× parameter efficiency gain, meaning a 1.4B model performing at 4B-level and a 2.6B model at 8B-level. This is a fundamentally different claim from prior work on distillation or pruning, which typically accept some performance degradation in exchange for size reduction.

### The Two Existing Paths and Their Limitations

The paper identifies two established strategies for improving model capability without increasing parameter count, and argues both face structural ceilings:

**Path 1: Scaling training data without scaling parameters.** The observation that smaller models trained on more tokens can outperform larger models trained on fewer tokens is well-established (Hoffmann et al., 2022; the Chinchilla scaling laws). However, the paper argues this path is increasingly constrained by data scarcity:

> "data scarcity increasingly limits this path"

This is a specific claim worth unpacking. The total stock of high-quality, publicly available text on the internet is finite and, by some estimates, already substantially consumed by existing training runs. While synthetic data generation and data recycling can extend this boundary, they introduce distributional concerns and diminishing returns. The paper positions data scaling as a necessary but insufficient lever—pushing more tokens through a fixed-parameter model eventually hits a knowledge capacity ceiling that the paper itself quantifies at approximately 2 bits per parameter (Section 6.1).

**Path 2: Inference-time compute via Chain-of-Thought.** The second established strategy is to spend more computation at inference time through explicit reasoning traces—generating intermediate steps (Chain-of-Thought, or CoT) that decompose complex problems into simpler sub-problems. This defers reasoning to post-training and scales compute at deployment rather than at training time. The paper identifies two specific problems with this approach:

First, **context-length bloat**. CoT reasoning extends the output sequence, consuming precious context window space and increasing the quadratic attention cost. For problems requiring dozens of reasoning steps, the generated trace can easily exceed thousands of tokens—this is not just a computational cost but a practical limitation when context windows are finite and shared across input, reasoning, and output.

Second, and more subtly, the paper identifies a **faithfulness problem** with explicit CoT. In Section 7.2, they cite a growing body of evidence ([73–76]) that "standard LLMs often appear to decide on an answer before generating chain-of-thought text and then use that text to rationalize the already-formed decision." This is the post-hoc rationalization critique: the model's reasoning trace may be causally decoupled from its actual decision process, making it unreliable for interpretability, safety verification, and debugging. If you intervene on the reasoning and the answer doesn't change, the reasoning wasn't doing the work it appeared to be doing.

The paper proposes a third path: **architectural innovation** that builds iterative computation into the model structure itself, so that reasoning happens in latent space rather than in emitted tokens, decoupling compute depth from both parameter count and sequence length.

### The Intellectual Lineage: From Universal Transformers to Latent Reasoning

The paper does not claim to invent the idea of recurrent transformer blocks. It explicitly traces the lineage from the Universal Transformer (Dehghani et al., 2018) through recursive transformers (Bae et al., 2024), looped transformers (Saunshi et al., 2025), and latent reasoning approaches (Geiping et al., 2025; Hao et al., 2024). The Related Work section (Section 2) organizes this prior work into two complementary perspectives:

**Perspective 1: Parameter sharing for model efficiency** views looped models as a compression technique—reusing transformer blocks to reduce total parameters while maintaining computational depth. The canonical example is ALBERT (Lan et al., 2019), which combined cross-layer parameter sharing with embedding factorization. This work was largely pre-LLM era and focused on encoder-only models for classification tasks. More recent work like Megrez2 (Li et al., 2025) reuses experts across layers in Mixture-of-Experts models for edge deployment. The paper positions itself as extending this perspective into the decoder-only, multi-trillion-token pretraining regime where it had not been demonstrated at frontier scale.

**Perspective 2: Latent reasoning and iterative refinement** views looped computation as a form of non-verbal "thinking." Each recurrent step refines the model's internal representation without emitting tokens—what Saunshi et al. (2025) called "latent thoughts" and Hao et al. (2024) called "continuous thought" tokens. These approaches show promising results on reasoning benchmarks but at scales of 100M–1B parameters and billions of tokens, leaving open the question of whether the benefits persist or amplify at frontier scales (billions of parameters, trillions of tokens).

The paper's framing of these two perspectives is deliberate: they are not competing interpretations but complementary ones. Parameter sharing provides the efficiency mechanism; latent reasoning explains why the efficiency doesn't come at the cost of capability on reasoning tasks. The paper aims to unify these perspectives and demonstrate their combined power at scale.

### What Makes This Different from Prior LoopLM Work

The paper identifies a specific gap that prior work left open, stated explicitly as a research question:

> "Does LoopLM exhibit more favorable scaling behavior (in capabilities, efficiency and safety), compared to non-recursive transformer models?"

The key word is **scaling behavior**. Prior work demonstrated that looped transformers can work at modest scales—matching deeper non-looped models on specific tasks, showing theoretical advantages in expressiveness. But the critical unanswered question was whether these advantages compound or diminish as models, data, and training budgets scale to the frontier regime (multi-trillion tokens, competitive with production models like Qwen3, Gemma3, and Llama3).

This matters because architectural innovations that work at 100M parameters often fail to translate to 1B+ parameters. Training dynamics change—gradient flow through recurrent paths can amplify instabilities, the interaction between shared weights and optimizer states becomes more complex, and the relationship between parameter count and knowledge capacity can shift. The paper explicitly documents several scale-dependent challenges:

- **Training instability at 8 recurrent steps** in Stage 1a, requiring reduction to 4 steps (Section 4.3)
- **Batch size scaling** from 4M to 8M tokens to stabilize gradient estimates through recurrent iterations
- **Learning rate sensitivity** of recurrent architectures requiring more conservative rates than parameter-matched transformers
- **The need for progressive sequence length increases** (4K → 16K → 64K → 32K) to manage the interaction between recurrence depth and context length

These are not minor implementation details—they are evidence that scaling LoopLM to the multi-trillion-token regime required solving non-trivial optimization challenges that prior small-scale work never encountered.

### The Knowledge Capacity vs. Manipulation Distinction

Perhaps the most conceptually important framing in the paper is the distinction between **knowledge capacity** (how many facts a model can store in its parameters) and **knowledge manipulation** (how well a model can compose and reason over stored facts). This distinction is motivated by the Physics of Language Models framework (Allen-Zhu and Li, 2025; Allen-Zhu, 2025) and serves as the paper's explanatory mechanism for *why* LoopLM works.

The hypothesis is clean: parameter sharing cannot increase the total information stored in the model's weights because the total number of trainable parameters is fixed. What it can do is improve how efficiently those parameters are used for multi-step reasoning—retrieving facts, composing them, applying logical operations, and resolving ambiguities.

The paper tests this hypothesis through controlled synthetic experiments (Section 6):

1. **Capo task** (knowledge capacity): Train models to memorize synthetic biographies and measure bits of stored knowledge per parameter. Result: looped and non-looped models both saturate at ~2 bits/parameter regardless of recurrent depth—"looping does not increase knowledge capacity nor improve capacity scaling" (Section 6.1).

2. **Mano task** (knowledge manipulation): Train models to solve modular arithmetic expressions requiring composition of learned operations. Result: looped models consistently outperform iso-parameter non-looped models, and often match or exceed iso-FLOP non-looped models—"LoopLM has a better inductive bias towards knowledge manipulation" (Section 6.2).

3. **Multi-hop QA** (natural language manipulation): Train models on synthetic multi-hop questions requiring composition of learned facts. Result: looped models require fewer training examples to achieve the same accuracy and learn faster at equal data budgets—"models with more loops learns faster and achieve better performance" (Section 6.2).

This set of experiments is the paper's strongest mechanistic argument. It shows not just *that* LoopLM works, but *why*: it's not storing more knowledge—it's better at using the knowledge it has. This is fundamentally different from claims made by prior parameter-sharing work, which typically framed the benefit purely in terms of compression ratios without explaining what capability was being preserved or enhanced.

### Positioning Against Inference-Time Compute Methods

The paper positions LoopLM as complementary to—and in some ways preferable to—inference-time compute scaling through CoT. The key comparative claims:

**Latency and context efficiency.** CoT reasoning consumes output tokens, extending generation time and context usage. LoopLM's iterative computation happens entirely in latent space—the output sequence length is unchanged regardless of how many recurrent steps are used internally. For a user waiting for a response, this means the visible output length doesn't balloon with problem difficulty.

**Improved faithfulness.** Section 7.2 provides evidence that LoopLM's intermediate states genuinely revise the model's predictions rather than rationalizing pre-committed answers. When linear probes are trained on intermediate hidden states to predict the step-level answer, they show systematic disagreement across steps (only 36.1% of step-2 answers match step-4 answers on Quora Question Pairs). This pattern of revision is precisely what faithful reasoning should exhibit—each recurrent pass performs non-trivial computation that can change the outcome.

**Safety alignment improves with depth.** Section 7.1 shows that safety (measured by harmfulness on HEx-PHI) *improves* as recurrent steps increase, including when extrapolating beyond the trained depth of 4 steps to 8 steps. This is an unexpected finding: the model becomes safer with more computation even though it wasn't explicitly trained for safety at those depths. The paper's PCA analysis (Figure 8b) suggests this works because deeper recurrence better separates harmful from benign prompt representations, making the model more capable of distinguishing between them.

## 3. Technical Approach

### 3.1 Reader Orientation

This paper develops and scales a **Looped Language Model (LoopLM)**—a transformer where a stack of weight-tied layers is applied recurrently, with an adaptive exit mechanism that lets the model spend more computation on hard problems and less on easy ones, all while keeping the visible output sequence short. The core problem it solves is: how do you get the reasoning capability of a deep model without paying the parameter cost, and how do you make that capability adaptive so that simple inputs don't waste computation? The shape of the solution is: share parameters across computational depth, train those parameters so that each iteration refines the output, and train a separate gate to decide when to stop, all within a single unified pre-training pipeline.

### 3.2 Big-Picture Architecture (Diagram in Words)

The Ouro LoopLM has five major components:

**1. A stack of weight-tied transformer blocks.** The Ouro 1.4B model uses 24 transformer layers; the 2.6B model uses 48 layers. These layers are not applied once but are reused `t` times, where `t` ranges from 1 up to `T_max` (set to 4 in the final training, initially attempted at 8). Each of the `t` iterations is called a **recurrent step** or **loop**. The same weights process the hidden states repeatedly, meaning the total number of trainable parameters equals the stack depth times the per-layer parameter count—but the *computational* depth is `t × stack_depth`.

**2. An LM head at every recurrent step.** Unlike a standard transformer that only produces output at the final layer, LoopLM attaches a language modeling head at *each* recurrent step. At step `t`, the hidden state `h^{(t)}` is projected through `lmhead` to produce logits over the vocabulary. This means the model has a sequence of predictions—one per loop—each progressively refined. The training loss is a weighted sum over these per-step losses.

**3. An exit gate running parallel to the LM head at each step.** At each recurrent step `t`, a small neural network (a linear layer followed by a sigmoid) takes the current hidden state as input and outputs a scalar `λ_t ∈ (0, 1)`, which is the probability of *exiting* (halting) at that step. Crucially, the gate does not just fire once—it produces a probability at *every* step, and these are combined across steps to form a proper categorical distribution over exit steps.

**4. Entropy regularization over the exit distribution.** The model is trained with a KL-divergence penalty toward a uniform prior over the `T_max` exit steps. Without this, gradient descent on the task loss collapses all probability mass onto the final step (because deeper steps produce lower loss, so the gate learns to always continue, receiving more training signal and reinforcing the behavior). The entropy term forces the model to explore different depths, which is essential for learning *when* to stop based on input difficulty rather than always going to the maximum.

**5. A Stage II focused gate training procedure.** After pre-training, the transformer parameters are frozen, and the gate is trained separately with a greedy signal: should the model continue to the next step or exit now, based on whether an additional loop actually reduces the loss? This produces a gate that makes compute-aware stopping decisions.

Information flows as follows: a tokenized sequence enters the model → the embedding layer maps tokens to vectors → the 24 (or 48) shared transformer layers process the hidden states once (step `t=1`) → the LM head produces first-pass logits and the exit gate produces an exit probability `λ_1` → if not exiting, the same transformer stack processes the updated hidden states again (step `t=2`) → the LM head produces refined logits, the gate produces `λ_2` → this repeats up to `T_max` times → at inference, a threshold `q` on the cumulative exit distribution determines the final exit step, and the LM head output at that step becomes the model's prediction.

### 3.3 Roadmap for the Deep Dive

- **First**, the formal definition of the LoopLM architecture—what exactly is being looped, how the forward pass differs from a standard transformer, and how per-step losses are computed. This establishes the base computational graph that everything else builds on.
- **Second**, the adaptive computation mechanism via exit gates—how the gate is parameterized, how its per-step probabilities combine into a valid distribution over exit steps, and how inference uses a quantile-based rule for early exiting. Understanding this is prerequisite to the training objectives.
- **Third**, the Stage I training objective—the entropy-regularized expected task loss, including its formulation, its variational interpretation as an ELBO with a uniform prior, and the mathematical reason it prevents collapse to `T_max`. This is the core learning algorithm.
- **Fourth**, the Stage II focused gate training—how the gate is fine-tuned with a greedy signal derived from per-step loss improvements, and why this produces better compute-accuracy trade-offs than Stage I alone.
- **Fifth**, the full training pipeline—how stability considerations forced architectural adjustments (reducing recurrent steps from 8 to 4, progressive batch size scaling, KL coefficient reduction), and the sequencing of data across four stages plus SFT. This integrates the technical design with practical engineering constraints.

### 3.4 Detailed, Sentence-Based Technical Breakdown

This is an **architectural innovation and scaling paper** whose core idea is that recurrent application of weight-tied transformer blocks, combined with a learned adaptive-depth mechanism, can achieve parameter efficiency gains of 2–3× over standard transformers at multi-trillion-token scales, with the gains stemming from enhanced knowledge manipulation capability rather than increased knowledge storage.

---

#### 3.4.1 The LoopLM Architecture: Formal Definition

**What is being looped.** The LoopLM reuses a stack of `L` causal transformer layers `T_θ`, where `θ` denotes the parameters of those layers and `L` equals 24 for Ouro 1.4B or 48 for Ouro 2.6B (Table 2). A standard non-looped transformer with `L` layers computes:

$$F(\cdot) = \text{lmhead} \circ M_L \circ \text{emb}(\cdot)$$

where

$$M_L(\cdot) = T_{\theta_L} \circ \cdots \circ T_{\theta_1}(\cdot)$$

and `emb(·)` maps tokens to `ℝ^d` (embedding), each `T_{θ_i}` is a causal transformer layer (self-attention + SwiGLU feed-forward with sandwich RMSNorm), and `lmhead` maps back to vocabulary logits. The looper version `F^{(t)}` applies this same `L`-layer stack `t` times:

$$F^{(t)}(\cdot) = \text{lmhead} \circ M_L \circ M_L \circ \cdots \circ M_L \circ \text{emb}(\cdot)$$

with exactly `t` copies of `M_L`. When `t=1`, this recovers the standard non-looped model `F^{(1)} ≡ F`.

**A crucial detail:** the LM head is attached at *every* recurrent step, not just the final one. At step `t`, the model computes `p^{(t)}_θ(x_{ℓ+1} | x_{1:ℓ}) = \text{softmax}(\text{lmhead}(h^{(t)}_ℓ))`, where `h^{(t)}_ℓ` is the hidden state at position `ℓ` after `t` loops. This means we have `T_max` separate predictions for each token position, one per recurrent step, all sharing the same `lmhead` parameters.

**Per-step loss computation.** The cross-entropy loss at a single recurrent step `t` is:

$$L^{(t)} = \mathbb{E}_{x_{1:M}} \left[ \frac{1}{M-1} \sum_{\ell=1}^{M-1} -\log p^{(t)}_\theta(x_{\ell+1} | x_{1:\ell}) \right]$$

where `M` is the sequence length and the expectation is over training sequences. This is the standard next-token prediction loss, computed separately at each recurrent depth.

**Why per-step heads?** This design serves two purposes. First, it allows the model to produce usable predictions at any depth—important for early exiting. Second, it provides a training signal at every recurrent step, ensuring that intermediate states are directly optimized toward the task objective rather than being purely intermediate representations with no direct supervision. Without per-step LM heads, earlier recurrent steps would only receive gradient through the subsequent steps, making their optimization more difficult and less reliable.

**Transformer block details.** Each `T_θ` block uses Multi-Head Attention (MHA) with Rotary Position Embeddings (RoPE, base frequency 10K in early stages, increased to 40K and then 1M in later stages), a SwiGLU-activated feed-forward network, and a sandwich normalization structure (RMSNorm before both the attention and FFN sub-layers, following Geiping et al., 2025). The model uses a vocabulary of 49,152 tokens from SmolLM2. The Ouro 1.4B uses hidden size `d_model = 2048` with 24 layers; the Ouro 2.6B uses the same hidden size with 48 layers (Table 2).

**The upcycling procedure.** Stage 1b introduces a key architectural fork: the 1.4B model retains the original 24-layer stack, while the 2.6B model is created by duplicating the 24 layers to 48 and continuing training. The paper notes that the recurrent architecture makes this upcycling "particularly smooth, as the shared weights across iterations naturally facilitates layer duplication without the typical instabilities seen in standard transformer upcycling" (Section 4.3.1). This is because the model is already trained to process its own outputs repeatedly—adding more layers in the stack is a less disruptive change than in a non-looped architecture.

---

#### 3.4.2 Adaptive Computation via Exit Gates

**Gate parameterization.** At each recurrent step `t ≤ T_max`, an exit gate runs in parallel with the LM head. The gate takes the final-layer hidden state `h^{(t)}` at step `t` and produces an instantaneous exit probability:

$$\lambda_t(x) = \sigma \left( \text{Linear}_\phi(h^{(t)}) \right) \in (0, 1)$$

where `σ` is the sigmoid function, `Linear_φ` is a learned linear transformation with parameters `φ`, and `h^{(t)}` is the hidden state. The `Linear` layer projects from the model's hidden dimension `d_model` to a single scalar logit, which the sigmoid squeezes into `(0, 1)`. At each step, the gate answers the question: "conditional on having reached this step, what is the probability of stopping here?"

**Constructing a valid distribution over exit steps.** A single `λ_t` is not enough—we need a proper probability distribution over which step to exit at. The paper constructs this distribution sequentially. First, define the **survival probability** `S_t(x)`, which is the probability of *not* having exited in the first `t` steps:

$$S_t(x) = \prod_{j=1}^t (1 - \lambda_j(x))$$

with `S_0(x) ≡ 1` (before any steps, we haven't exited with probability 1). Then, the **unnormalized probability of exiting first at step `t`** is the probability of surviving through steps 1 to `t-1`, then exiting at `t`:

$$\tilde{p}_t(x) = \lambda_t(x) \cdot S_{t-1}(x), \quad t = 1, \ldots, T_{\text{max}} - 1$$

For the final step `T_max`, there is no subsequent step to continue to, so we assign all remaining survival mass:

$$\tilde{p}_{T_{\text{max}}}(x) = S_{T_{\text{max}}-1}(x)$$

This construction ensures that:

$$p_\phi(t | x) = \tilde{p}_t(x) \text{ for } t = 1, \ldots, T_{\text{max}}$$

defines a valid distribution over `{1, ..., T_max}`, meaning:

$$\sum_{t=1}^{T_{\text{max}}} p_\phi(t | x) = 1$$

which the paper explicitly verifies in Equation (3).

**Why this construction?** Alternative approaches might use a softmax over `T_max` independent logits to directly produce a categorical distribution. The sequential construction is preferred because it respects the temporal structure of the loop: the decision at step `t` is made *given* that we haven't exited earlier. The softmax approach would require the model to decide the exit step all at once from the first hidden state, losing the ability to condition on intermediate computation. The sequential construction also enables the cumulative distribution function (CDF) to have a simple closed form, which is important for inference.

**Inference with quantile-based early exit.** At inference time, we need to decide when to stop. The paper uses a deterministic quantile-based policy. The cumulative exit probability up to step `n` is:

$$\text{CDF}(n | x) = \sum_{t=1}^n p_\phi(t | x) = 1 - \prod_{j=1}^n (1 - \lambda_j(x)), \quad n < T_{\text{max}}$$

with `CDF(T_max | x) = 1`. Given a threshold `q ∈ [0, 1]`, we exit at the first step where the CDF exceeds `q`:

$$t_{\text{exit}}(x) = \min\{ m \in \{1, \ldots, T_{\text{max}}\} : \text{CDF}(m | x) \geq q \}$$

**What `q` controls.** The threshold `q` is a deployment-time knob that trades compute for accuracy:
- `q ≈ 0`: always exit at step 1 (minimal compute, potentially lower accuracy)
- `q ≈ 1`: always run to `T_max` (maximal compute, potentially best accuracy)
- Intermediate `q`: adaptive depth, with the model deciding which inputs need more computation

Smaller `q` makes the model more eager to exit—it exits as soon as the CDF, which accumulates probability mass from the gate's per-step exit predictions, crosses the threshold. The paper evaluates different `q` values indirectly through the average exit round experiments in Section 5.4.1, where varying `q` produces different operating points on the compute-accuracy curve.

**Why CDF-based and not sampling-based?** A sampling-based policy (sample `t` from `p_φ(t|x)` and stop there) would introduce variance and could exit very early or very late purely by chance. The CDF-based policy is deterministic given `q` and the gate predictions, making it reproducible and controllable. It's also equivalent to a quantile of the learned distribution: `q = 0.5` exits at the median of `p_φ`, `q = 0.9` exits at the 90th percentile.

---

#### 3.4.3 Stage I: Entropy-Regularized Training Objective

**The core tension.** Naively training with gradient descent on the next-token prediction loss creates a self-reinforcing collapse. Deeper recurrent steps typically achieve lower per-step loss `L^{(t)}` (the model gets better with more iterations). As gradient descent pushes probability mass toward steps with lower loss, `p_φ(t|x)` concentrates on `t = T_max`. Those later steps then receive more training signal, their loss drops further, and the gate shifts even more mass toward them. The result: the gate collapses to always using `T_max`, defeating the purpose of adaptive computation.

**The Stage I objective.** The solution is to combine the expected task loss with entropy regularization:

$$\mathcal{L} = \underbrace{\sum_{t=1}^{T_{\text{max}}} p_\phi(t | x) \, L^{(t)}}_{\text{expected task loss}} - \underbrace{\beta \, H(p_\phi(\cdot | x))}_{\text{entropy regularization}}$$

where the entropy of the exit distribution is:

$$H(p_\phi(\cdot | x)) = -\sum_{t=1}^{T_{\text{max}}} p_\phi(t | x) \log p_\phi(t | x)$$

**What it computes operationally.** For each token in each training sequence: (1) compute the per-step loss `L^{(t)}` at every recurrent step `t = 1, ..., T_max`; (2) compute the gate's exit distribution `p_φ(t|x)` from the per-step `λ_t` values using the sequential construction; (3) compute the weighted sum `Σ p_φ(t|x) L^{(t)}`, which is the expected next-token loss under the gate's exit distribution; (4) compute the entropy of `p_φ`, which is high when probability is spread across many steps and low when concentrated on few steps; (5) subtract `β · H` from the expected loss. The total loss is backpropagated through both the transformer parameters `θ` and the gate parameters `φ`.

**The coefficient `β`.** The hyperparameter `β` controls the exploration-exploitation trade-off:
- Large `β` (0.1 in early training): strong pressure toward high entropy (uniform-like distribution), forcing the model to explore all depths and preventing premature collapse
- Small `β` (0.05 in later stages): weaker entropy pressure, allowing the gate to concentrate mass on specific depths when the model has learned which depths are beneficial

The paper also explicitly reduces `β` from 0.1 to 0.05 for two reasons: (1) it decreases conflicting gradients between the task loss and KL penalty, leading to more stable optimization, and (2) it reduces the "pull" from the uniform prior, giving the model more freedom to learn useful depth patterns (Section 4.3).

**Variational interpretation as an ELBO.** The objective can be reframed as variational inference over a latent variable `z ∈ {1, ..., T_max}` representing the exit step. The negative ELBO with a uniform prior `π(t) = 1/T_max` is:

$$\mathcal{L}_{\text{ELBO}} = \sum_{t=1}^{T_{\text{max}}} p_\phi(t | x) \, L^{(t)} + \beta \, \text{KL}(p_\phi(\cdot | x) \| \pi(\cdot))$$

The KL divergence to a uniform prior is:

$$\text{KL}(p_\phi(\cdot | x) \| \pi) = -H(p_\phi(\cdot | x)) + \log T_{\text{max}}$$

Since `log T_max` is constant with respect to the parameters, minimizing the ELBO is equivalent to minimizing the Stage I objective (up to this constant). This connects the training objective to the PonderNet framework (Banino et al., 2021), which also uses an ELBO for dynamic halting.

**Why a uniform prior?** The paper explicitly contrasts their uniform prior with alternative depth-biasing priors. A geometric prior (as in PonderNet) assigns `π(t) = η^{t-1}(1-η)`, which biases toward early exits. A Poisson or lognormal prior also favors shallower computation. The uniform prior is "depth-unbiased"—it imposes no structural preference for early or late exits. The rationale (Section 3.3): the paper wants exit decisions to be driven entirely by input difficulty, not by a built-in bias toward shallow steps. The entropy term then prevents collapse to `T_max` without pushing toward early exits. Empirical comparison in Appendix A (Figure 10) confirms that the uniform prior achieves lower training loss and cleaner convergence compared to geometric priors, especially at larger `η` values where the geometric prior's bias toward early steps starves deeper iterations of training signal.

**Training `θ` and `φ` jointly.** In Stage I, both the transformer parameters `θ` and the gate parameters `φ` are learned simultaneously with the objective above. The transformer learns to produce better hidden states (lower `L^{(t)}`) and to do so in ways that differentiate depth utility across inputs; the gate learns to map those hidden states to halting decisions. This joint optimization, combined with the entropy regularizer, is what creates adaptive depth allocation without explicit per-input depth supervision.

---

#### 3.4.4 Stage II: Focused Adaptive Gate Training

**Why Stage II is needed.** After Stage I, the gate has learned a distribution over exit steps that avoids collapse, but it was trained only with entropy regularization—it doesn't have explicit supervision for *when* to exit versus continue. The gate may produce reasonable-looking distributions without knowing whether continuing actually helps. Stage II provides this missing signal: train the gate to base its decisions on realized performance improvements from additional loops.

**Freezing the transformer.** Stage II freezes all transformer parameters `θ` and trains only the gate parameters `φ`. This ensures that the gate learns to make decisions based on the existing model's behavior without changing the model's representations to accommodate the gate.

**Computing ideal continuation labels.** The core idea is to compare the loss at step `t-1` with the loss at step `t` and ask: did looping again help? For each token position `i` in each sequence, the per-step loss is computed with gradient detached (`L^{(t)}_{i, \text{stop}}`) so the gate training doesn't backpropagate through the loss computation into the transformer. The loss improvement from step `t-1` to `t` is:

$$I^{(t)}_i = \max\left(0, L^{(t-1)}_{i, \text{stop}} - L^{(t)}_{i, \text{stop}}\right)$$

If `I^{(t)}_i` is large (loss dropped substantially from the previous step), the loop was helpful, and the ideal behavior is to continue. If `I^{(t)}_i ≈ 0` (loss didn't improve), the loop wasn't helpful, and the ideal behavior is to exit.

**Converting improvement to a soft label.** The ideal continuation probability `w^{(t)}_i` is a soft binary label constructed from `I^{(t)}_i`:

$$w^{(t)}_i = \sigma\left(k \cdot (I^{(t)}_i - \gamma)\right)$$

with slope `k = 50.0` and threshold `γ = 0.005`. The steep slope (`k = 50`) acts like a smooth threshold: when `I^{(t)}_i` is significantly above `γ`, `w^{(t)}_i ≈ 1` (recommend continuing); when `I^{(t)}_i` is significantly below `γ`, `w^{(t)}_i ≈ 0` (recommend exiting). The threshold `γ = 0.005` sets the minimum improvement considered meaningful—improvements below this are treated as noise.

**Why sigmoid with a threshold?** A hard threshold (continue if `I > γ`, else exit) would create discontinuous labels, making optimization difficult. The sigmoid with large `k` produces a smooth, differentiable version of this hard decision. As `k → ∞`, this converges to a step function; `k = 50` provides a practical compromise between sharpness and smoothness.

**The adaptive exit loss.** At each step `t ≥ 2`, the gate's predicted continuation probability (the probability of *not* exiting, i.e., `1 - λ^{(t)}_i`) is compared to the ideal label `w^{(t)}_i` via binary cross-entropy, averaged over the sequence:

$$\mathcal{L}^{(t)}_{\text{adaptive}} = -\frac{1}{M} \sum_{i=1}^M \left[ w^{(t)}_i \log\left(1 - \lambda^{(t)}_i\right) + (1 - w^{(t)}_i) \log\left(\lambda^{(t)}_i\right) \right]$$

The total adaptive loss averages across all recurrent steps from `t = 2` to `T_max`:

$$\mathcal{L}_{\text{adaptive}} = \frac{1}{T_{\text{max}}} \sum_{t=2}^{T_{\text{max}}} \mathcal{L}^{(t)}_{\text{adaptive}}$$

Step `t = 1` is excluded because there's no previous step to compute an improvement from.

**What this loss penalizes.** The first term `w^{(t)}_i log(1 - λ^{(t)}_i)` penalizes **underthinking**: the improvement signal says "continue" (`w ≈ 1`), but the gate predicts a high exit probability (`λ` large, so `1-λ` small), meaning the model is exiting too early. The second term `(1 - w^{(t)}_i) log(λ^{(t)}_i)` penalizes **overthinking**: the improvement signal says "exit" (`w ≈ 0`), but the gate predicts a low exit probability (`λ` small), meaning the model is continuing when additional computation yields negligible improvement. This is symmetric—it targets both failure modes simultaneously without requiring hand-crafted penalties.

**Why this works as a greedy signal.** The loss improvement `I^{(t)}_i` is a greedy metric: it looks only at the marginal benefit of one additional step, not the global optimal depth. This is appropriate because the gate makes sequential decisions—at step `t`, it only needs to decide whether to take one more step, not whether the final optimal depth is step 3 versus step 7. The threshold `γ` encodes the acceptable cost-benefit trade-off: a step must improve the loss by at least `γ` to be worth taking.

**Empirical validation.** Section 5.4.1 (Figure 5) shows that Stage II training consistently shifts the accuracy-compute curve upward compared to the untrained (Stage I only) gate. At an average exit round of 2.5 on MMLU, the trained gate achieves ~66% accuracy versus ~64% for the untrained gate—a systematic 2–3% accuracy improvement at the same compute budget across most operating points.

---

#### 3.4.5 Training Pipeline: The Full Seven-Stage Recipe

**Why this isn't just "train a looped transformer."** The paper details a complex multi-stage pipeline because recurrent architectures exhibit different optimization characteristics than standard transformers. The training isn't a single run with fixed hyperparameters—it's a carefully sequenced progression that manages stability, adapts the data distribution, and gradually shifts the model from exploration to depth-specialization.

**Stage overview (Figure 4).** The total training processes 7.7T tokens across:

| Stage | Name | Tokens | Key Features |
|---|---|---|---|
| 1a | Pre-training I | 3T | 8 recurrent steps, `β=0.1`, 4K sequence length |
| 1b | Pre-training II | 3T | Reduced to 4 steps, upcycling fork (1.4B/2.6B), batch scaling 4M→8M |
| 2 | CT Annealing | 1.4T | LR annealed to 3e-5, 16K sequence length, `β=0.05`, high-quality data |
| 3 | LongCT | 20B | 64K sequence length, ProLong data |
| 4 | Mid-training | 300B | 32K sequence length, SFT-quality data, LR 1e-5 |

**Stage 1a: Exploration and instability.** The initial training used 8 recurrent steps with a constant learning rate of `3 × 10^{-4}`, sequence length 4K, batch size starting at 4M tokens (increasing to 8M), and `β = 0.1`. This stage "led to loss spikes and gradient oscillations" (Section 4.3). The paper hypothesizes this is due to "compounded gradient flow through multiple recurrent iterations, which can amplify small perturbations." The KL coefficient of 0.1 was chosen to provide strong entropy regularization during early exploration, ensuring the gate didn't prematurely collapse before the transformer had learned useful depth-varying representations.

**Stage 1b: Stability-driven reduction and upcycling.** The recurrent steps were reduced from 8 to 4. This is a significant design decision: the model's maximum computational depth was halved to maintain stability. The paper implies this trade-off was necessary—8-step recurrence produced better per-step losses but the optimization was unreliable. With 4 steps, the model could be trained stably at scale. At this point, the 1.4B path retains the existing 24 layers; the 2.6B path duplicates them to 48 layers and continues training. The upcycling works well because "the shared weights across iterations naturally facilitates layer duplication"—the model already knows how to process its own outputs, so having more layers in the shared stack is a natural extension.

**Stage 2: Continual Training (CT) Annealing.** The learning rate is annealed to `3 × 10^{-5}` (10× lower than Stage 1) following a cosine decay schedule. This stage introduces higher-quality data: the corpus shifts from web-heavy to include HQ MegaMath, Nemotron-CC-Math, code data, and SFT-style data (Table 5). Sequence length extends to 16K. The KL coefficient `β` is reduced from 0.1 to 0.05, allowing the gate to begin specializing without being strongly pulled toward uniformity. RoPE base frequency increases from 10K to 40K to support the longer sequences.

**Stage 3: Long Context Training (LongCT).** A short but focused stage: only 20B tokens from ProLong-64K, but at sequence length 64K. This extends the model's context window with minimal token budget. RoPE base increases to 1M.

**Stage 4: Mid-training.** The final pre-training stage uses a diverse set of high-quality supervised data (ChatML format), consisting of both `⟨Question, Answer⟩` and `⟨Question, CoT, Answer⟩` pairs from 20+ open-source SFT datasets. The effective volume is 300B tokens: 90B sampled from the SFT mix, with 30B tokens replayed from Stage 1 data and 180B from Stage 2 data to stabilize the training distribution. Learning rate drops to `1 × 10^{-5}` with a cosine schedule, RoPE base stays at 1M, sequence length at 32K. This stage "consolidates and extends capabilities acquired during pre-training under diverse supervised signals."

**SFT Stage (Reasoning).** The paper performs supervised fine-tuning on approximately 8.3M examples (Table 6) emphasizing mathematical reasoning (3.5M), code generation (3.2M), scientific reasoning (808K), and conversational abilities (767K). Training uses 2 epochs, maximum sequence length 32K, Adam optimizer with `lr = 2 × 10^{-5}`, `β = (0.9, 0.95)`, cosine decay schedule, using the LlamaFactory codebase. The SFT checkpoint produces "Ouro-Thinking" models (1.4B-Thinking-R4 and 2.6B-Thinking-R4).

**RL attempts (Section 4.5).** The paper documents unsuccessful attempts at RLVR (Reinforcement Learning with Verifiable Rewards) alignment using DAPO and GRPO. The core issue: vLLM and SGLang, used for fast rollouts, assume a fixed execution path, but LoopLM's dynamic early-exit mechanism breaks this assumption. Two attempted workarounds failed: (1) off-policy rollouts with simulated early exit—the mismatch between full-depth generation and early-depth loss computation didn't improve performance; (2) fixed 4-round training—training proceeded normally but performance didn't surpass the SFT checkpoint. The paper speculates this may be due to limited headroom after extensive SFT at these model scales. This is an open problem the paper flags for future work.

**Stability considerations summarized.** The paper makes several architectural and optimization choices specifically for recurrent architectures: (1) sandwich normalization (RMSNorm before both attention and FFN) following Geiping et al. (2025) to stabilize deep recurrent computation; (2) conservative optimizer settings (AdamW with weight decay 0.1, `β_1 = 0.9`, `β_2 = 0.95`, gradient clipping at 1.0) chosen "specifically to maintain stability with recurrent architectures"; (3) smaller learning rates than parameter-matched transformers because "recurrent architectures require smaller learning rates"; (4) progressive sequence length increases (4K → 16K → 64K → 32K) to "stabilize optimization while expanding context capacity with training throughput."

---

#### 3.4.6 KV Cache Sharing for Inference Efficiency

**The memory problem.** A naive implementation of LoopLM would maintain separate KV caches for each recurrent step. For a 4-step model, this means 4× the KV cache memory of an equivalent standard transformer. Since KV cache memory is often the bottleneck for serving (it scales with batch size × sequence length × layers × hidden size), this 4× overhead could negate the parameter efficiency advantage.

**Prefilling vs. decoding asymmetry.** The paper makes a critical empirical distinction: during prefilling (processing the input prompt), each recurrent step genuinely needs its own KV cache—"attempting to reuse KV caches during prefilling leads to performance degradation (>10 points on GSM8K)" (Section 5.4.2). This makes sense: during prefilling, each recurrent pass fundamentally transforms the input representations, so cached keys and values from step `t` are not valid for step `t+1`.

During decoding (auto-regressive generation), however, the situation changes. The paper explores three reuse strategies: (1) **first-step reuse**: only maintain the KV cache from step 1; (2) **last-step reuse**: only maintain the KV cache from the final step (step 4); (3) **averaged reuse**: maintain an average of KV caches across all steps.

**Results (Table 14).** First-step reuse catastrophically fails: GSMK8 accuracy drops from 78.92 to 18.73, MATH-500 from 82.40 to 8.43. This suggests the initial-step representations are not informative for subsequent decoding—the model needs the refined representations from later steps. In contrast, last-step reuse is nearly lossless: GSMK8 78.85 (vs. 78.92 full) and MATH-500 80.40 (vs. 82.40 full). Averaged reuse performs slightly worse on MATH-500 (78.52) but nearly identically on GSMK8 (78.73). Both achieve 4× memory reduction, making LoopLM deployment practical with memory footprints comparable to standard transformers of similar parameter count.

**What this means.** The final recurrent step's representations contain the most refined information and serve as an excellent approximation for all subsequent decoding steps. This finding "enables practical deployment of LoopLM models"—the architecture's computational cost at inference (FLOPs) is higher than a standard transformer (4 passes through a shallower stack vs. 1 pass through a deeper stack), but the memory cost can be matched through KV cache reuse.

## 4. Key Insights and Innovations

### Innovation 1: Parameter Count Decouples Knowledge Capacity from Knowledge Manipulation — and LoopLM Only Improves the Latter

This is the paper's most conceptually significant finding, and it reframes what we mean when we say a model "benefits from depth." The dominant assumption in the scaling literature (Hoffmann et al., 2022; Kaplan et al., 2020) is that making a model deeper—whether through more layers, more parameters, or more computation—improves *everything* in some correlated way: more parameters means more memorized facts, better reasoning, higher accuracy across the board. The paper shows this is wrong, or at least incomplete, through a clean set of controlled synthetic experiments.

The key diagnostic move is the distinction between **knowledge capacity** (raw storage of facts in parameters, measured in bits per parameter) and **knowledge manipulation** (composing stored facts to answer multi-step questions). Prior work on looped transformers (Saunshi et al., 2025; Geiping et al., 2025; Dehghani et al., 2018) had demonstrated performance parity with deeper models, but never explained *what* about the architecture produced the gain. The dominant implicit assumption was that looping effectively increased capacity—that reusing layers somehow packed more knowledge into the same parameter budget.

The paper disproves this. The Capo experiment (Section 6.1, Figure 6 left) shows that looped and non-looped models both saturate at approximately **2 bits per parameter** of knowledge storage regardless of recurrent depth. A 4-loop model stores the same amount of factual information as a 1-loop model with identical parameter count. The knowledge capacity scaling law—bits vs. parameters—is almost perfectly overlapping. This is a *null result* that is genuinely informative: it tells us looping does not create more parameter-efficient knowledge encoding.

Where the gain actually comes from is the Mano and Multi-hop QA experiments (Section 6.2, Figures 6 right and 7). On tasks requiring composition of learned operations—parsing arithmetic expression trees, chaining multiple facts in natural language—looped models consistently outperform iso-parameter non-looped baselines, and often match or exceed iso-FLOP non-looped models. The Mano task shows 2-loop and 4-loop models matching a 12-layer non-looped baseline on complex arithmetic. The Multi-hop QA task shows looped models learning 3-hop reasoning with fewer unique training examples and faster convergence.

This is a fundamental reframing, not an incremental result, because it changes the design rationale for recurrent architectures. You don't build a LoopLM to store more facts in fewer parameters—you build it to get more reasoning per stored fact. The implication is that knowledge capacity is a function of total unique parameter count (as the ~2 bits/parameter scaling suggests), while manipulation capability is a function of computational depth, which can be achieved through recurrence without adding parameters. This is conceptually analogous to the separation of memory and compute in classical computer architecture, but now applied to the parameter space of neural networks.

The evidence is anchored in Figures 6 and 7, the Capo/Mano/Multi-hop QA results, and the per-category MMLU analysis in Appendix B.4 (Table 15), which shows that looped models improve most dramatically on reasoning-heavy categories (Elementary Mathematics: +155.6%, Formal Logic: +143.3%) and least on knowledge-heavy categories (Global Facts: +8.3%, Moral Scenarios: +7.8%). This real-world benchmark pattern independently confirms the synthetic finding.

### Innovation 2: Entropy-Regularized Depth Allocation as a Pre-Training Primitive, Not a Post-Training Add-On

Adaptive computation—letting a model spend different amounts of compute on different inputs—has a long history (Graves, 2016; Banino et al., 2021; Dehghani et al., 2018). But prior approaches to adaptive depth in transformers have almost universally treated the decision mechanism as something you add *after* the base model is designed or trained: either as a separate routing network (Mixture-of-Experts), as a post-hoc early-exit classifier trained on top of frozen representations, or as a reinforcement learning objective layered onto an existing architecture.

This paper makes a fundamentally different choice: **depth allocation is learned jointly with the language modeling objective during pre-training, regularized by an entropy term that prevents collapse.** The exit gate and the transformer parameters are trained simultaneously from scratch on the same data, with the same loss, in the same backward pass. This means the representations that determine when to exit are the same representations used for next-token prediction—there is no separate pathway, no post-hoc calibration, and no architectural discontinuity between the "thinking" and "deciding to stop thinking" processes.

Why does this matter? Two reasons, both going beyond what prior adaptive-computation work achieved.

First, **joint training prevents the off-policy problem**. Post-hoc exit classifiers (trained on frozen model representations) suffer from distribution shift: the representations they were trained on were produced by a model that always ran to full depth, so the classifier never sees what happens when the model actually exits early and the subsequent computation is *not performed*. The LoopLM gate is trained on-policy: during training, the model actually exits at different steps (because the entropy regularizer forces exploration), and the per-step losses are computed from whichever step the model was at. There is no mismatch between training behavior and deployment behavior.

Second, **the entropy regularizer is not just a training trick—it's what makes the gate a learned function of input difficulty rather than a learned function of training dynamics.** Without regularization, gradient descent naturally concentrates probability mass on later steps (because they produce lower loss, and lower loss produces gradients that further lower loss at those steps—a self-reinforcing loop). The entropy penalty breaks this cycle by penalizing concentration, forcing the gate to maintain a spread of probabilities. But crucially, this is not a fixed, non-adaptive pressure: as training progresses and the transformer learns to produce better representations, the *differences* in per-step loss between easy and hard inputs become more pronounced, and the gate can selectively allocate depth based on these differences while still maintaining adequate entropy. At convergence, the entropy term prevents collapse but doesn't prevent specialization—the gate can still learn that certain inputs should exit early and others should continue, as long as the *average* entropy across the batch stays above the threshold.

The empirical comparison in Appendix A (Figure 10) makes this concrete. Geometric priors (which bias toward early exit) produce higher training loss and more late-training oscillations than the uniform prior, because they starve deeper steps of training signal. The uniform prior imposes no structural depth preference, allowing all depths to receive comparable signal during early training before the gate learns to specialize. The Stage I → Stage II progression (Sections 3.3 and 3.4) then refines this: Stage I ensures the gate doesn't collapse; Stage II teaches it to make *good* decisions (correlating exit with marginal improvement).

This is a fundamental contribution because it establishes adaptive depth as a **pre-training primitive**—something you design into the architecture and training objective from the start, not something you bolt on later. It changes the design space for future architectures: rather than asking "how do we add adaptive computation to a trained model?", we can ask "what architectures and training objectives produce adaptive computation as an emergent property of pre-training?"

### Innovation 3: Latent Recurrence as a Faithfulness Mechanism — and the Empirical Case That It Works Differently from CoT

The faithfulness of chain-of-thought reasoning has become a contentious topic. Multiple recent papers (Arcuschin et al., 2025; Barez et al., 2025; Korbak et al., 2025) have shown that standard LLMs often appear to reach a decision *before* generating their reasoning trace, using the subsequent tokens to rationalize rather than to compute. If you intervene on the intermediate reasoning (changing a step, deleting a premise) and the final answer remains unchanged, the reasoning was causally decoupled from the output—it was post-hoc rationalization, not genuine computation.

The paper makes a distinctive claim: **LoopLM's latent recurrence produces a more faithful reasoning process than explicit CoT generation, and provides a protocol for measuring this faithfulness.** This is significant because it addresses a critique leveled at CoT that has no obvious fix within the CoT paradigm. If the problem is that explicit text generation allows (or encourages) the model to commit to an answer before reasoning, one solution is to move the reasoning into latent space, where there is no discrete token commitment and each recurrent pass can genuinely revise the model's internal state.

The empirical evidence comes from the Quora Question Pairs experiment (Section 7.2, Figure 9). The protocol is: train linear probes on intermediate hidden states at each recurrent step to predict that step's answer; compute agreement matrices between steps; and compare to non-looped baselines. Three findings emerge:

1. **Within a recurrent step, the step's answer is well-predicted by probes on representations within that step** (ROC AUC rising quickly), but **not well-predicted by probes on the preceding step's final representation**. This shows that each recurrent pass performs new computation that can revise the provisional answer, rather than simply refining a pre-committed decision.

2. **Systematic disagreement across steps.** The agreement matrix (Figure 9 right) shows that adjacent steps are far from full consensus: only 55.1% of step-2 answers match step-3 answers, and only 36.1% of step-2 answers match step-4 answers. This is precisely what faithful reasoning should produce—the model is genuinely updating its decision as it computes more. If the reasoning were post-hoc rationalization, we would expect near-perfect agreement from the earliest step onward.

3. **Qwen3-4B-Thinking, a CoT-based model, shows the opposite pattern.** A linear probe on the final-token logits achieves 0.99 ROC AUC for predicting the eventual answer, meaning the CoT trace adds almost no information to what was already present before reasoning began. This reproduces the post-hoc rationalization finding from prior work, establishing that the LoopLM results are not an artifact of the task or the probing methodology.

This is not merely a "LoopLM is more faithful" claim—it's a **diagnostic framework for reasoning faithfulness** that can be applied to any architecture. The agreement matrix and probe-based step-wise prediction protocol provide a quantitative measure of whether intermediate computation genuinely contributes to the final output. This matters beyond this paper because it gives the field a way to evaluate future reasoning architectures on a dimension (faithfulness) that is orthogonal to task accuracy and often in tension with it.

The conceptual contribution here is distinguishing between two types of iterative refinement: **convergent refinement** (each step brings the output closer to a pre-determined target, with early steps already strongly correlated with the final answer) versus **revision-based refinement** (each step can substantively change the output, with early steps potentially pointing in different directions). The evidence suggests LoopLM exhibits the latter on ambiguous inputs, while standard CoT exhibits the former. This distinction hasn't been clearly articulated in prior work on latent reasoning (Saunshi et al., 2025; Hao et al., 2024), which focused on accuracy rather than the internal dynamics of how predictions evolve.

### Innovation 4: The Upcycling Finding — Recurrent Architectures Enable Smooth Parameter Scaling Mid-Training

This is a more engineering-oriented innovation than the previous three, but it has significant practical implications for how model families are developed. The standard approach to creating models at multiple scales is to train each size independently from scratch. This is expensive and wasteful: a 1B model and an 8B model trained on the same data learn similar low-level features; the smaller model's training doesn't inform the larger one's.

The paper documents a finding that is likely to influence future model development pipelines: **LoopLM's weight-tied recurrent structure makes layer upcycling unusually effective.** In Stage 1b, the 1.4B model (24 layers, 4 recurrent steps) is expanded to 2.6B by simply duplicating the 24-layer stack to 48 layers and continuing training (Section 4.3.1). The paper notes this process is "particularly smooth, as the shared weights across iterations naturally facilitates layer duplication without the typical instabilities seen in standard transformer upcycling."

The significance of this claim depends on what "typical instabilities" refers to. In standard (non-looped) transformer upcycling, duplicating layers creates a depth mismatch: the model has been trained to map input → output through exactly L layers of computation, and suddenly it has 2L layers. The expanded model must learn that layers L+1 through 2L should approximate the identity function at initialization while gradually taking on more computation during continued training. This often causes loss spikes and requires careful learning rate annealing.

In LoopLM, by contrast, the model has already been trained to process its own output—the 24-layer stack is designed to be applied repeatedly. Adding more layers to the shared stack is a more natural extension because the model already knows how to handle its own hidden states as input. There is less of a conceptual gap between "apply 24 layers twice" and "apply 48 layers once."

This is an incremental but practically significant innovation because it enables a more efficient model development pipeline: train a smaller LoopLM first, then upcycle to larger sizes with continued training, rather than training each size independently. The paper doesn't ablate this claim (we don't see a control experiment where non-looped upcycling is attempted on the same data), so the strength of the evidence is moderate, but the finding is consistent with the architecture's design principles.

### Innovation 5: Safety Improves with Recurrent Depth — Including Beyond the Trained Regime

This finding is unexpected and, if it generalizes, has implications for AI safety that go beyond the LoopLM architecture itself. The standard intuition about model safety and compute is that safer behavior comes from explicit alignment training (RLHF, constitutional AI, safety SFT), not from deeper computation. If anything, more compute at inference time might be expected to *increase* harmful capabilities by enabling more sophisticated reasoning about how to bypass safeguards.

The paper shows the opposite (Section 7.1, Figure 8a): on the HEx-PHI benchmark of harmful prompts, harmfulness scores **decrease** (improve) as recurrent steps increase, including when extrapolating from the trained depth of 4 steps to 5–8 steps. The Ouro 2.6B model's harmful rate drops from a higher value at 1–2 steps to 0.003 at 4 steps, competitive with Qwen3-4B-Thinking (0.009). More striking, this trend continues into the extrapolated regime (steps 5–8) where task-specific benchmark performance degrades (Tables 10–13 show accuracy drops at T > 4). The safety improvement and the benchmark performance degradation are *decoupled*—the model gets safer even as its fine-grained knowledge recall gets worse.

The PCA analysis (Figure 8b) provides a mechanistic hint: as recurrent steps increase, the model's hidden representations of harmful and benign prompts become more separable in the principal component space. Points associated with unsafe responses (harmfulness scores 4–5) appear near the boundary between the benign and harmful clusters, suggesting that ambiguity in representation space leads to unsafe outputs, and deeper recurrence resolves this ambiguity.

This is a novel empirical finding with conceptual weight. The dominant paradigm in AI safety treats safety alignment as a property you *train into* the model's weights through curated data and reward signals. This paper suggests that safety can also be a property of *computation depth*—that the same weights, applied more times, produce safer outputs. If this generalizes to other architectures and tasks, it implies a new axis for safety interventions: rather than (or in addition to) training models to be safe, design architectures where safety emerges from deeper computation during inference.

The finding is still provisional—it's demonstrated on one benchmark (HEx-PHI), on one model family (Ouro), and the mechanism is not fully explained (the PCA shows correlation, not causation). But it's the kind of result that, if replicated and understood, could change how we think about the relationship between inference compute and safety.

## 5. Experimental Analysis

### Evaluation Methodology

- **Dataset.** The primary evaluations span standard benchmarks for base models (MMLU, MMLU-Pro, BBH, ARC-C, HellaSwag, Winogrande, GSM8K, MATH500, HumanEval, HumanEval+, MBPP, MBPP+) and advanced reasoning benchmarks for thinking models (AIME 2024/2025, OlympiadBench, GPQA, SuperGPQA, BeyondAIME, HLE). Base model evaluations use lm-eval-harness and evalplus frameworks with settings detailed in Appendix C.1 (Tables 16–17). The Quora Question Pairs dataset is used for faithfulness analysis (Section 7.2), and HEx-PHI is used for safety evaluation (Section 7.1). Synthetic experiments use the Capo biography dataset, the Mano modular arithmetic task, and a multi-hop QA task derived from Yao et al. (2025) (Section 6).

- **Base model(s).** The Ouro models are decoder-only transformers with 1.4B parameters (24 layers, `d_model = 2048`) and 2.6B parameters (48 layers, `d_model = 2048`), both using MHA with RoPE, SwiGLU FFN, sandwich RMSNorm, and a 49,152-token SmolLM2 vocabulary (Table 2). These are trained on 7.7T tokens across four stages plus SFT (Figure 4). The models use 4 recurrent steps at inference unless otherwise noted. The choice of two scales (1.4B and 2.6B) tests whether performance gains persist as the shared layer stack doubles.

- **Metrics.** For base models: accuracy on MMLU (5-shot, logprobs), MMLU-Pro (5-shot CoT, strict match), BBH (3-shot CoT, strict match), ARC-C (25-shot, logprobs), HellaSwag (10-shot, logprobs), Winogrande (5-shot, logprobs), GSM8K (3-shot CoT, strict match), MATH500 (5-shot CoT, strict match), HumanEval and MBPP (pass@1 via evalplus). For reasoning models: pass@1 and pass@10 on AIME, with LLM-as-judge protocol on OlympiadBench, GPQA, SuperGPQA, BeyondAIME, and HLE (temperature = 1.0, top_p = 0.7). For safety: harmfulness scores (1–5) and harmful rate on HEx-PHI. For faithfulness: ROC AUC of linear probes and agreement matrices across recurrent steps on Quora Question Pairs. For knowledge capacity: bits per parameter on the Capo task.

- **Baselines.** For base model comparison (Tables 7–8): Gemma3 (1B, 4B, 12B), Llama3.2 (1.2B, 3B), Llama3.1 (8B), Qwen2.5 (1.5B, 3B, 7B), Qwen3 (1.7B, 4B, 8B). For reasoning model comparison (Table 9): Qwen3-1.7B, Qwen3-4B, Qwen3-8B, DeepSeek-Distill-Qwen-1.5B, DeepSeek-Distill-Qwen-7B. For faithfulness analysis (Section 7.2): Qwen3-4B-Instruct and Qwen3-4B-Thinking serve as CoT baselines. For synthetic experiments (Section 6): iso-parameter and iso-FLOP non-looped transformers of matching configurations. For safety (Section 7.1): Qwen3-4B-Thinking is referenced for comparison on harmful rate.

- **Generation budget / compute accounting.** For comparing looped and non-looped models on standard benchmarks, the primary axis is parameter count: the 1.4B Ouro is compared to models up to 4B, and the 2.6B Ouro to models up to 12B (Tables 7–8). For within-architecture analysis (Section 5.3), compute is measured in recurrent steps (T = 1 through T = 8), with T = 4 being the trained maximum. For early exit experiments (Section 5.4.1), compute is measured as the average exit round across the evaluation set. In synthetic experiments (Section 6), compute is controlled by matching FLOPs across looped and non-looped configurations where noted. KV cache memory is measured in multiples of the standard transformer's cache size (1.00× baseline vs. 4.00× for full LoopLM cache vs. 4.00× memory reduction for reuse strategies, Table 14).

- **Cross-validation / statistical protocol.** No cross-validation is reported for the main benchmark results—all comparisons are single-run evaluations using the specified evaluation frameworks. For the Mano task (Section 6.2), experiments are run across 3 random seeds with the best performance reported. For the multi-hop QA task (Section 6.2), experiments use 4 random seeds with average performance reported. For the entropy-regularization ablation (Appendix A), experiments are repeated with multiple random seeds and variability is shown as shaded regions (Figure 10). For the scaling law analysis (Appendix D), the coefficient of determination R² is used to measure fit quality; generalizability is assessed by fitting on subsets of model sizes/training data/recurrent steps and evaluating on held-out configurations.

### Main Quantitative Results

#### Base Model Benchmark Performance

The headline result for base model evaluation (Tables 7–8): Ouro 1.4B R4 achieves performance comparable to or exceeding 4B-class models, and Ouro 2.6B R4 achieves performance comparable to or exceeding 8B-class models, with particularly strong results on reasoning-intensive tasks.

**Ouro 1.4B R4 vs. baselines (Table 7).** On general knowledge tasks, Ouro 1.4B R4 scores 67.35 on MMLU (vs. Qwen3-4B at 73.19, Qwen3-1.7B at 62.46), 48.62 on MMLU-Pro (vs. 51.40 for Qwen3-4B and 37.27 for Qwen3-1.7B), and 71.02 on BBH (vs. 70.95 for Qwen3-4B—essentially tied—and 53.51 for Qwen3-1.7B). On math and coding tasks, the gaps are larger: GSM8K at 78.92 (vs. 72.86 for Qwen3-4B and 70.28 for Qwen3-1.7B), MATH500 at 82.40 (vs. 59.60 for Qwen3-4B—a 22.8-point advantage—and 25.80 for Qwen3-1.7B), HumanEval at 74.40 (vs. 77.40 for Qwen3-4B—slightly behind—and 66.50 for Qwen3-1.7B). The pattern is clear: LoopLM's advantage is most pronounced on multi-step reasoning (MATH500, GSM8K, BBH) and more modest on knowledge-recall tasks (MMLU, HumanEval).

**Ouro 2.6B R4 vs. baselines (Table 8).** This model pushes into 8B+ territory. On MMLU: 74.60 (vs. Qwen3-8B at 76.63 and Gemma3-12B at 72.14). On MMLU-Pro: 55.73 (vs. 53.72 for Qwen3-8B and 49.21 for Gemma3-12B). On BBH: 80.46 (vs. 77.65 for Qwen3-8B and 78.41 for Gemma3-12B). On MATH500: 90.85, which dramatically exceeds Qwen3-8B (62.30) and Gemma3-12B (83.20)—a 28.5-point gap over the 8B model. On GSM8K: 81.58 (vs. 83.09 for Qwen3-8B, slightly behind, and 77.18 for Gemma3-12B). On HumanEval: 78.70 (vs. 84.80 for Qwen3-8B—behind—and 46.30 for Gemma3-12B). The same pattern recurs: reasoning tasks show the largest advantages, while coding tasks show smaller or negative gaps.

**Interpretation of the pattern.** The paper's knowledge capacity hypothesis (Section 6) predicts exactly this: LoopLM should excel where the bottleneck is composing and manipulating stored knowledge (math reasoning, multi-hop logic) rather than where it is retrieving specific facts (some MMLU subcategories) or generating syntactically complex outputs (code). The MATH500 result (82.40 and 90.85 for 1.4B and 2.6B respectively) is the strongest single piece of evidence—these are competition-level math problems requiring multi-step symbolic reasoning, and the looped models surpass much larger baselines by wide margins. Conversely, coding tasks (HumanEval, MBPP) show smaller or reversed gaps, consistent with the idea that code generation requires syntactic knowledge and pattern matching that may benefit more from parameter count than from iterative latent reasoning.

#### Reasoning Model (Ouro-Thinking) Performance

The SFT variants (Ouro-Thinking) are evaluated on advanced reasoning benchmarks that require deep multi-step problem solving (Table 9).

**Ouro-1.4B-Thinking-R4.** On AIME 2024: 65.0 pass@1 / 83.3 pass@10 (vs. Qwen3-4B at 61.3/75.0—surpassing at both metrics). On AIME 2025: 46.3/73.3 (vs. 51.3/63.3 for Qwen3-4B—higher pass@10 but lower pass@1). On OlympiadBench: 71.55 (vs. 73.18 for Qwen3-4B—slightly behind). On BeyondAIME: 34.0 (vs. 31.0 for Qwen3-4B—ahead). On HLE: 5.21 (vs. 5.21 for Qwen3-4B—tied). On SuperGPQA: 47.37 (vs. 51.89 for Qwen3-4B—behind). On GPQA: 45.45 (vs. 54.54 for Qwen3-4B—behind).

**Ouro-2.6B-Thinking-R4.** On AIME 2024: 64.7/90.0 (vs. Qwen3-8B at 73.0/86.7—higher pass@10 but lower pass@1). On AIME 2025: 50.3/76.7 (vs. 66.7/81.3 for Qwen3-8B—behind on both). On OlympiadBench: 76.44 (vs. 75.25 for Qwen3-8B—slightly ahead). On BeyondAIME: 39.0 (vs. 38.0 for Qwen3-8B—slightly ahead). On HLE: 5.58 (vs. 2.22 for Qwen3-8B—substantially ahead). On SuperGPQA: 53.68 (vs. 48.0 for Qwen3-8B—ahead). On GPQA: 52.69 (vs. 59.10 for Qwen3-8B—behind).

**Key pattern.** The reasoning model results are notably less one-sided than the base model results. While the base models show dramatic advantages on MATH500 and GSM8K, the thinking models show mixed results: sometimes ahead (OlympiadBench, BeyondAIME, HLE), sometimes behind (GPQA, AIME pass@1 for 2.6B). The comparison to DeepSeek-Distill models (which are themselves reasoning-specialized via distillation) shows consistent advantage for Ouro: 1.4B Ouro-Thinking scores 65.0/83.3 on AIME24 vs. DeepSeek-Distill-Qwen-1.5B at 29.6/66.7; 2.6B Ouro-Thinking scores 76.44 on OlympiadBench vs. DeepSeek-Distill-Qwen-7B at 72.0. This suggests LoopLM's advantage is most visible against dense baselines rather than against other reasoning-specialized architectures.

#### Performance Scaling with Recurrent Depth

Section 5.3 (Tables 10–13) examines how performance varies as the number of recurrent steps changes, including extrapolation beyond the trained depth of T = 4.

**Base model results (Tables 10–11).** For both 1.4B and 2.6B base models, performance on standard benchmarks generally improves from T = 1 to T = 4. For Ouro 1.4B: MMLU increases from 41.21 (T = 1) to 67.45 (T = 4), a 26.2-point gain; ARC-C from 37.63 to 60.92, a 23.3-point gain; HellaSwag from 55.24 to 74.29, a 19.1-point gain; Winogrande from 56.99 to 72.30, a 15.3-point gain. For Ouro 2.6B: MMLU goes from 51.55 (T = 1) to 74.60 (T = 4), a 23.1-point gain; ARC-C from 47.95 to 66.38; HellaSwag from 68.94 to 79.56. The improvements are monotonic (or nearly so) through T = 4.

**Extrapolation behavior (T = 5–8).** When recurrent steps exceed the trained maximum of 4, performance degrades. For Ouro 1.4B at T = 8: MMLU drops to 64.49 (from 67.45 at T = 4), ARC-C to 58.19 (from 60.92), HellaSwag to 71.60 (from 74.29). The degradation is gradual, not catastrophic—T = 8 scores are still substantially higher than T = 1 scores. For Ouro 2.6B at T = 8: MMLU drops to 72.24 (from 74.60), ARC-C to 64.76 (from 66.38). The paper notes this contrasts with safety, which continues to improve into the extrapolated regime (Section 7.1, Figure 8a).

**Reasoning model results (Tables 12–13).** For the SFT models on advanced reasoning tasks, the pattern differs from base models. Ouro-1.4B-Thinking performance peaks at T = 4 or T = 5 depending on the task: OlympiadBench peaks at T = 5 (72.30 vs. 71.55 at T = 4), AIME 2024 peaks at T = 4 (65.00), AIME 2025 peaks at T = 5 (47.00 vs. 46.30 at T = 4). Ouro-2.6B-Thinking peaks slightly earlier: OlympiadBench at T = 4 (76.44), AIME 2024 at T = 3 (70.33 vs. 64.70 at T = 4), AIME 2025 at T = 3 (50.67 vs. 50.30 at T = 4). The authors note that "neither model peaks strictly at T = 4 across all tasks, unlike the base model evaluations," speculating that "the longer decoding required for these reasoning tasks allows for a more active exploration of capabilities at different recurrent depths." Extrapolation beyond T = 4 on reasoning tasks shows steeper degradation than on base model benchmarks, with AIME 2024 for the 2.6B model dropping from 64.70 at T = 4 to 39.00 at T = 8.

#### Adaptive Computation Efficiency

Section 5.4.1 (Figure 5) compares four early exit strategies on MMLU: static exit (fixed depth), hidden state difference threshold, untrained Ponder gate (Stage I only), and trained Ponder gate (Stages I + II). The trained Ponder gate achieves the best accuracy at every computational budget. At an average exit round of ~2.5, it reaches approximately 66% accuracy, vs. ~64% for the untrained gate, ~63% for hidden state difference, and ~60% for static exit at depth 2. The gap between trained and untrained gates (~2 percentage points) represents the benefit of the Stage II adaptive exit loss, which teaches the gate to base decisions on observed loss improvements rather than entropy regularization alone.

The hidden state difference strategy (exiting when the L2-norm of representational change between steps falls below a threshold) performs competitively, tracking within 1–2% of the trained gate at moderate budgets. This suggests representation stability serves as a reasonable proxy for computational convergence, though it is consistently outperformed by the learned gate.

The static baseline shows the "deeper is better" property (monotonic improvement from 1 to 4 rounds) with diminishing returns: T = 1 achieves ~40%, T = 2 jumps to ~60%, T = 4 reaches 67.35%. The large gap from T = 1 to T = 2 explains why adaptive methods are effective—most examples achieve near-maximal performance at intermediate depths, with only a minority requiring full depth.

#### KV Cache Sharing

Table 14 evaluates three KV cache reuse strategies during decoding. First-step reuse causes catastrophic collapse: GSM8K drops from 78.92 (full 4× cache) to 18.73, MATH500 from 82.40 to 8.43. Last-step reuse preserves performance: GSM8K at 78.85 (down 0.07), MATH500 at 80.40 (down 2.00). Averaged reuse: GSM8K at 78.73 (down 0.19), MATH500 at 78.52 (down 3.88). All reuse strategies achieve 4× memory reduction. The conclusion: the final recurrent step's representations are sufficiently informative for subsequent decoding, while initial-step representations are not.

### Ablation Studies and Robustness Checks

**Entropy regularization prior choice (Appendix A, Figure 10):** The uniform prior over exit steps produces lower training loss and cleaner convergence than geometric priors on a 776M-parameter LoopLM with T_max = 4 trained on 20B tokens of FineWeb-Edu. Geometric priors with parameter η ∈ {0.1, ..., 0.9} (assigning probability η^{t-1}(1-η) to step t) plateau at higher training loss, with the gap widening as η grows—stronger bias toward early exit reduces training signal for deeper steps. The uniform prior also exhibits smaller late-training oscillations, consistent with maintained entropy preventing premature collapse.

**Progressive training stage effects:** The paper's multi-stage pipeline itself is an implicit ablation of training stability interventions. Stage 1a (8 recurrent steps) was unstable with loss spikes; Stage 1b (reduced to 4 steps) converged stably. The KL coefficient β was reduced from 0.1 (Stage 1) to 0.05 (Stage 2+), which "decreases the conflicting gradients between task loss and the KL penalty, leading to more stable optimization" (Section 4.3). Batch size was scaled from 4M to 8M tokens to provide more stable gradient estimates for the recurrent architecture.

**KV cache sharing strategies (Table 14):** Ablation of which intermediate step's KV cache to retain finds that first-step caches are insufficient (GSM8K drops to 18.73), while last-step caches are nearly equivalent to full caches (78.85 vs. 78.92). Averaged caches are intermediate (78.73 on GSM8K, but 78.52 vs. 82.40 on MATH500). This is a non-obvious asymmetry: the model can decode using only the final step's representations but not the initial step's.

**Per-category MMLU analysis (Appendix B.4, Table 15):** Breaking MMLU into 57 sub-categories reveals that LoopLM's gains are concentrated in reasoning-heavy categories. The largest relative improvements (Loop 4 vs. Loop 1) are: Elementary Mathematics (+155.6%), Formal Logic (+143.3%), Logical Fallacies (+127.8%), High School Statistics (+126.9%). The smallest improvements are: Moral Scenarios (+7.8%), Global Facts (+8.3%), Virology (+13.7%), Anatomy (+21.4%). This pattern directly supports the knowledge manipulation hypothesis—improvements are proportional to the reasoning complexity of the task.

**Negative result: RLVR alignment (Section 4.5):** Attempts to apply RLVR (DAPO and GRPO) after SFT did not improve performance over the SFT checkpoint. Two approaches were attempted: (1) off-policy rollouts in vLLM with simulated early exit—the mismatch between full-depth generation and early-depth loss computation did not improve performance; (2) fixed 4-round RL training—proceeded normally but did not surpass SFT. The paper attributes this to vLLM/SGLang's assumption of fixed execution paths, which breaks under LoopLM's dynamic computation, and speculates that limited headroom after extensive SFT at these model scales may also be a factor. This is a significant negative result because it shows the dynamic early-exit mechanism creates infrastructure challenges for standard RL alignment pipelines.

### Critical Assessment

**Claim: "2–3× parameter efficiency gains" (from Executive Summary).** The evidence for this claim varies substantially by task. On MATH500, the efficiency gain is dramatic: Ouro 1.4B (82.40) outperforms Qwen3-4B (59.60) by 22.8 points—this is far more than 2–3× efficiency, since the 1.4B model is beating baseline models with nearly 3× more parameters by a large margin. On GSM8K, Ouro 1.4B (78.92) beats Qwen3-4B (72.86) by 6 points. On MMLU, the 1.4B model (67.35) sits between the 1.7B (62.46) and 4B (73.19) baselines—roughly a 2× efficiency gain. On coding tasks, the claim has weaker support: Ouro 1.4B (74.40 on HumanEval) trails Qwen3-4B (77.40), and Ouro 2.6B (78.70) trails Qwen3-8B (84.80). The efficiency gain is thus highly task-dependent, concentrated on reasoning benchmarks, and much less evident on coding. The paper's abstract claim of matching "up to 12B SOTA LLMs" is accurate for specific benchmarks (the 2.6B model reaches 90.85 on MATH500, beating Gemma3-12B at 83.20) but overstates the generality of the result. A fairer characterization would be that LoopLM achieves 2–4× parameter efficiency on mathematical and logical reasoning, with more modest gains (or slight losses) on knowledge-intensive and code-generation tasks.

**Claim: "Gains stem from knowledge manipulation, not knowledge capacity" (from Section 6).** This is the most rigorously tested claim in the paper, and the evidence is strong within the scope of the synthetic experiments. The Capo task (Figure 6 left) cleanly shows that looping does not increase bits per parameter of stored factual knowledge. The Mano task (Figure 6 right) shows that looped models systematically outperform iso-parameter non-looped baselines on arithmetic reasoning. The multi-hop QA task (Figure 7) shows that looped models require fewer training examples and converge faster on compositional questions. The MMLU per-category analysis (Table 15) independently confirms the pattern on real-world benchmarks. However, the synthetic experiments use small models (GPT-2 scale, 1M–40M parameters), and the knowledge capacity measurement is specific to the biography memorization task. It is possible—though the paper provides evidence against this—that at larger scales, looping interacts with knowledge capacity differently. The ~2 bits/parameter saturation is an empirical regularity that may shift with model scale, architecture, or data distribution. The MMLU analysis partially addresses this concern by showing the same reasoning-vs-knowledge gradient at the 1.4B scale, but a 2.6B per-category breakdown would strengthen the claim.

**Claim: "Safety improves with recurrent depth, even beyond trained regime" (Section 7.1).** The safety result (Figure 8a) is genuinely surprising and well-documented, but it has important caveats. The evaluation uses a single benchmark (HEx-PHI, 330 examples across 11 categories), one judge model (GPT-4o), and one model family (Ouro). The sample size is small—330 examples split across conditions—which limits statistical reliability. The PCA analysis (Figure 8b) shows correlation between depth and representation separability but does not establish causation. Critically, the paper does not compare to a baseline that would rule out a simpler explanation: perhaps *any* form of deeper computation (not specifically LoopLM recurrence) improves safety alignment, and this is a property of iterative refinement rather than the looped architecture. A comparison with a standard transformer using CoT at varying reasoning depths would help isolate whether the safety improvement is a LoopLM-specific phenomenon or a general effect of spending more compute on harmfulness assessment.

**Claim: "LoopLM produces causally faithful reasoning traces" (Section 7.2).** The faithfulness analysis is creative and makes a compelling case, but it is limited to one dataset (Quora Question Pairs, a semantic equivalence task) and one comparison point (Qwen3-4B-Thinking). The 0.99 ROC AUC for Qwen3-4B-Thinking (indicating near-perfect predictability of final answer from pre-reasoning logits) is a striking baseline, but it's not clear whether this is a general property of CoT models or specific to this model on this task. The agreement matrix (Figure 9 right) showing 36.1% step-2/step-4 agreement is compelling evidence of internal revision, but the analysis would benefit from a causal intervention experiment (e.g., perturbing early-step hidden states and measuring the impact on later-step outputs) rather than purely observational probes. The paper acknowledges this limitation ("we cannot manipulate the latent reasoning process") but does not propose alternative causal validation.

**Weaknesses in the experimental design:**

The baseline models are not matched for training data quantity or quality. Ouro models are trained on 7.7T tokens of the paper's curated data mixture; the comparisons in Tables 7–8 include models trained on anywhere from 2T (Gemma3-1B) to 36T (Qwen3-1.7B, Qwen3-4B, Qwen3-8B) tokens, with different data compositions. The Qwen3 models, for instance, were trained on 36T tokens—nearly 5× more data than Ouro—which makes Ouro's performance on reasoning tasks even more notable but complicates the parameter efficiency claim. If Ouro had been trained on 36T tokens like Qwen3, would the gap shrink or widen? We don't know.

The reasoning model comparisons (Table 9) use an in-house harness and LLM-as-judge protocol, while the baselines (Qwen3, DeepSeek-Distill) may have been evaluated under different conditions in their original papers. The paper states "All systems are evaluated with a single in-house harness and identical prompting," which addresses this concern for the reported numbers, but the DeepSeek-Distill models in particular are designed for reasoning tasks and would be expected to perform well—the fact that Ouro often beats them is meaningful.

Missing experiments that would strengthen the paper:
- **Training data quantity ablation.** Train a 1.4B standard transformer on the same 7.7T-token Ouro data mixture and compare to the looped 1.4B model. This would isolate the architectural effect from the data effect.
- **Equivalent-depth non-looped baseline.** Compare Ouro 1.4B with 4 recurrent steps (effective depth: 24 layers × 4 = 96 layers of computation, but only 24 layers of parameters) against a 96-layer non-looped model with the same parameter count as Ouro 1.4B. This would test whether the recurrent structure is better than simply having more shallow, unshared layers.
- **Scaling behavior beyond 2.6B parameters.** The paper studies two scales (1.4B, 2.6B). Whether the 2–3× efficiency gain persists, shrinks, or grows at larger scales (e.g., 7B, 13B) is unknown. The scaling law analysis in Appendix D is limited to models under 1.4B and 20B tokens, which doesn't predict frontier-scale behavior.
- **Direct comparison to inference-time compute methods.** The paper argues that LoopLM is more efficient than CoT for reasoning, but never directly compares Ouro with 4 recurrent steps to a standard transformer using 4× output tokens of CoT reasoning under matched total FLOPs. The faithfulness comparison (Figure 9) is the closest this comes, but it's a quality comparison, not a compute-matched accuracy comparison.
- **Safety comparison with CoT-based model at matched depth.** To establish that the safety-depth relationship is LoopLM-specific, compare HEx-PHI scores for Ouro at varying T to a CoT model prompted with varying reasoning budgets.

Despite these limitations, the experimental section provides substantial evidence for the paper's core contributions. The benchmark results convincingly demonstrate that LoopLM achieves strong performance relative to much larger standard transformers, particularly on reasoning tasks. The synthetic experiments provide a mechanistic explanation for why. The depth-scaling and adaptive-computation results validate that the architecture works as designed. And the faithfulness and safety analyses, while preliminary, open interesting directions for future investigation.

## 6. Limitations and Trade-offs

### 6.1 Difficulty Estimation Cost Is Not Accounted for in the Efficiency Claims

**The assumption or constraint.** The adaptive computation mechanism—the exit gate that learns to terminate early on simple inputs and allocate more iterations to complex ones—requires that the model learn to differentiate inputs by difficulty during training. The paper's entropy-regularized objective (Section 3.3) and Stage II focused gate training (Section 3.4) jointly teach this discrimination. However, the paper never quantifies the *training-time cost* of learning adaptive depth allocation. Every training token must be processed through all `T_max` recurrent steps to compute the per-step losses `L^{(t)}` and the gate's exit distribution `p_φ(t|x)`, even though at inference the model will often exit early. The training FLOPs for a LoopLM with `T_max = 4` are approximately 4× those of a non-looped transformer with the same parameter count (since each token passes through the shared layer stack 4 times during training).

**The consequence.** The headline efficiency claim—2–3× parameter efficiency over standard transformers—compares Ouro to baselines purely on *parameter count and benchmark performance*, without accounting for training cost. A practitioner deciding whether to adopt LoopLM must consider total training FLOPs, not just final parameter count. If Ouro 1.4B requires 4× the training FLOPs per token of a standard 1.4B transformer, then from a training-budget perspective, it is closer to a 5.6B standard model (1.4B × 4 passes). The paper's claim of matching 4B models would then represent a more modest parameter-efficiency gain when measured against training FLOPs rather than parameter count. This matters because training compute is the dominant cost for foundation model development, and a method that reduces deployment parameters at the expense of increased training FLOPs may not be a net win in all regimes.

**What evidence exists in the paper.** The paper is largely silent on this trade-off. The training configuration (Section 4) specifies that the model is trained with 4 recurrent steps throughout Stages 1b–4, and the total training budget is 7.7T tokens, but there is no comparison between the FLOPs consumed to train Ouro 1.4B and the FLOPs that would have been consumed to train a baseline standard transformer of equivalent final performance. The scaling law analysis in Appendix D compares LoopLM and standard models at matched *model sizes* (e.g., Figure 13) but does not control for total training FLOPs. Section 4.3 mentions that "recurrent architectures require smaller learning rates than parameter-matched Transformers" and that the paper adopted conservative rates for stability, which suggests additional training cost beyond the per-step FLOPs.

**Mitigation status.** Not addressed. The paper does not report training FLOPs for any experiment, does not compare Ouro to a standard transformer trained with an equivalent FLOPs budget (only to models with larger parameter counts, often trained on more tokens), and does not discuss this as a limitation. The upcycling procedure (Section 4.3.1), where the 1.4B model is expanded to 2.6B via layer duplication and continued training, partially mitigates this by amortizing training cost across model scales, but the base training cost of the recurrent architecture remains unaccounted for in the efficiency claims.

### 6.2 Performance Gains Are Concentrated on Reasoning Tasks; the Method Shows Weak or Negative Gains on Knowledge-Recall and Code Generation Benchmarks

**The assumption or constraint.** The paper's knowledge capacity vs. manipulation hypothesis (Section 6) predicts that LoopLM should improve performance primarily on tasks requiring multi-step reasoning and knowledge composition, while providing little or no benefit on tasks that depend on factual recall or syntactic pattern reproduction. The paper frames this as a feature—an architectural bias toward better knowledge manipulation—but in practice it means that LoopLM's efficiency gains are *non-uniform across task types*, and the 2–3× headline number averages over substantial variance.

**The consequence.** A practitioner deploying LoopLM for a general-purpose application cannot expect uniform 2–3× parameter efficiency. On coding tasks, the gains are modest at best and sometimes negative: Ouro 1.4B R4 scores 74.40 on HumanEval vs. Qwen3-4B at 77.40 (a deficit), and 67.40 on HumanEval+ vs. 70.70 (a deficit); Ouro 2.6B R4 scores 78.70 on HumanEval vs. Qwen3-8B at 84.80 (a 6.1-point deficit) and 70.70 on HumanEval+ vs. 75.30 (a 4.6-point deficit) (Table 8). On knowledge-heavy MMLU subcategories, the gains are small: Global Facts improves by only 8.3% from Loop 1 to Loop 4, and Moral Scenarios by 7.8% (Table 15, Appendix B.4). On GPQA (a knowledge-intensive graduate-level science benchmark), Ouro-1.4B-Thinking-R4 scores 45.45 vs. Qwen3-4B at 54.54, and Ouro-2.6B-Thinking-R4 scores 52.69 vs. Qwen3-8B at 59.10—both deficits (Table 9). For applications where the task distribution includes substantial code generation, factual QA, or domain-specific knowledge retrieval, LoopLM may underperform a parameter-matched or even smaller standard transformer.

**What evidence exists in the paper.** The per-category MMLU analysis (Table 15) is the most direct evidence: reasoning-heavy categories (Elementary Mathematics +155.6%, Formal Logic +143.3%) show dramatically larger relative improvements from recurrence than knowledge-heavy categories (Global Facts +8.3%, Moral Scenarios +7.8%). The benchmark tables (Tables 7–9) consistently show larger gaps on math and logic tasks than on coding and general knowledge tasks. The synthetic Capo experiment (Section 6.1, Figure 6 left) provides mechanistic evidence: looping does not increase knowledge capacity in bits per parameter at all. The Mano and Multi-hop QA experiments (Section 6.2) confirm that looping helps with knowledge *manipulation* but not *storage*. The paper does not, however, provide a systematic task-level breakdown that would let a practitioner estimate expected gains for a specific application domain.

**Mitigation status.** The paper acknowledges this pattern implicitly through the knowledge capacity vs. manipulation framing (Section 6), which explains *why* the gains are task-dependent, but does not frame it as a limitation. The abstract and introduction emphasize the positive results (reasoning benchmarks) without caveating that the efficiency gains do not apply uniformly. Section 8 (Conclusion) does not mention this task-dependence as a limitation or a direction for future work.

### 6.3 The Hardest Problems Remain Unsolved — LoopLM Cannot Compensate for Fundamental Capability Gaps

**The assumption or constraint.** LoopLM improves performance by iteratively refining internal representations, which helps the model better utilize knowledge it already possesses. But it cannot generate correct solutions to problems that are fundamentally beyond the base model's knowledge or capability. This is the same fundamental limitation that the test-time compute scaling paper (Snell et al., 2024, summarized in the reference example) identified: inference-time computation amplifies existing capability but does not create it from nothing.

**The consequence.** On tasks where the base model's single-pass accuracy is near zero, additional recurrent steps provide negligible benefit. This is most visible in the difficulty-quintile analysis of Section 5.3: at T = 1, benchmarks like AIME 2024 show Ouro-1.4B-Thinking at 0.00 pass@1 and Ouro-2.6B-Thinking at 3.00 pass@1 (Tables 12–13). While performance improves substantially with recurrence (to 65.0 and 64.7 respectively at T = 4), the T = 1 numbers reveal that the model lacks even a rudimentary grasp of these problems without iterative refinement. More critically, on benchmarks where even the fully-looped model struggles, there is evidence of a ceiling: extrapolation to T = 5–8 degrades performance (Section 5.3, Tables 10–13), suggesting that the benefits of additional recurrent steps saturate and then reverse. The degradation at T > 4 indicates that the model was not trained to use more than 4 steps effectively, meaning the computational depth is bounded by the training configuration. There is no mechanism for the model to dynamically exceed `T_max` when it encounters an input that would genuinely benefit from more computation.

For the hardest reasoning tasks (HLE, BeyondAIME), the absolute performance remains low even for Ouro-2.6B-Thinking-R4: 5.58 on HLE and 39.0 on BeyondAIME (Table 9). These are the kinds of problems where the base model's pass@1 is near zero, and recurrence pushes the model from "cannot solve at all" to "solves some fraction," but the absolute performance ceiling is substantially below what much larger models or specialized reasoning systems (e.g., o1-style inference-time scaling) can achieve. The paper does not compare Ouro to inference-time compute baselines like best-of-N or beam search applied to standard transformers, so we cannot assess whether the recurrent architecture is more or less effective at pushing this capability frontier than alternative ways of spending a given FLOPs budget.

**What evidence exists in the paper.** The extrapolation experiments (Tables 10–13) show performance degradation beyond T = 4, directly demonstrating that the benefits of recurrence are bounded by training configuration. The absolute scores on HLE (5.58 for 2.6B) and BeyondAIME (39.0 for 2.6B) show that performance on frontier reasoning tasks remains low in absolute terms, even if it is competitive with baseline models. The T = 1 to T = 4 improvements on AIME (0.00 → 65.0 for 1.4B; 3.00 → 64.7 for 2.6B) illustrate the range of the effect: it can take the model from near-zero to competitive, but it cannot push it significantly beyond what a standard model of equivalent effective depth could achieve.

**Mitigation status.** The paper does not directly address the extrapolation ceiling as a limitation. Section 8 suggests that "future research should focus on enhancing performance extrapolation at greater depths." The RLVR attempts (Section 4.5) failed to improve performance over SFT, suggesting that post-training optimization may not be a reliable path to pushing the capability ceiling either. The lack of comparison to inference-time compute methods (best-of-N, beam search, revision models) leaves open the question of whether LoopLM's bounded computational depth is a fundamental architectural limitation or simply a training artifact that could be overcome with more sophisticated depth-extension training.

### 6.4 The Faithfulness Analysis Is Observational, Not Causal, and Is Limited to a Single Task and Comparison Point

**The assumption or constraint.** The faithfulness argument in Section 7.2 rests on two pieces of evidence: (1) linear probes trained on intermediate hidden states show that the step-`i` answer is not well-predicted by the step-(`i-1`) final representation, and (2) the agreement matrix across steps shows systematic disagreement (only 36.1% of step-2 answers match step-4 answers on Quora Question Pairs). Both are **observational** measures—they show that the model's predictions change across steps, but they do not establish that these changes are causally responsible for the final output in the way that faithfulness requires. The paper acknowledges this: "In our case, we cannot manipulate the latent reasoning process" (Section 7.2).

**The consequence.** The faithfulness claim—that LoopLM's latent recurrence "mitigates the post-hoc rationalization issues seen in standard CoT" (Section 8)—is supported by correlational evidence but not by the kind of causal intervention that would definitively establish faithfulness. A faithful reasoning process should satisfy a counterfactual criterion: if an intermediate state is perturbed to represent a different line of reasoning, the final output should change accordingly. The paper's probes show that intermediate states *do* change across steps and that these changes *are correlated with* output changes, but they do not show that intervening on an intermediate state *causes* the output to change in the predicted direction. It is possible—though the paper's evidence makes it seem less likely—that the observed step-to-step disagreement reflects noise or instability in the intermediate predictions rather than genuine, causally-effective reasoning revision.

Furthermore, the analysis is conducted on a single dataset—Quora Question Pairs, a semantic equivalence task with ambiguous boundaries—and compared to a single baseline model (Qwen3-4B-Thinking). We do not know whether the faithfulness pattern generalizes to other task types (mathematical reasoning, logical deduction, factual QA), other model scales, or other CoT-trained models. The 0.99 ROC AUC finding for Qwen3-4B-Thinking (indicating near-perfect predictability of the final answer from pre-reasoning logits) is striking but may be specific to that model's training or to semantic equivalence tasks, where the model can determine the answer largely from lexical overlap before engaging in deeper reasoning.

**What evidence exists in the paper.** Figure 9 provides the main evidence: the ROC AUC curves (left) showing rising predictability within each recurrent step and resets between steps, and the agreement matrix (right) showing systematic cross-step disagreement. The paper also cites prior work (Arcuschin et al., 2025; Barez et al., 2025; Korbak et al., 2025) establishing that standard CoT models exhibit post-hoc rationalization, and reproduces this finding for Qwen3-4B-Thinking (0.99 ROC AUC). The evidence that LoopLM differs from this pattern is reasonably strong within the scope of the Quora task, but the evidence that this difference constitutes *causal faithfulness* (as opposed to, e.g., step-by-step refinement of a noisy initial estimate) is not provided.

**Mitigation status.** The paper does not attempt to mitigate this limitation—it acknowledges the inability to perform causal interventions and relies on the observational proxy instead. Section 7.2 frames the findings carefully ("this systematic disagreement across steps... is precisely what a faithful latent process should exhibit"), which is a reasonable interpretation, but the paper does not propose alternative validation methods (e.g., counterfactual token editing, gradient-based attribution of output changes to specific recurrent steps, or training interventions that would break the faithfulness property if it were not causal).

### 6.5 Reinforcement Learning Alignment Infrastructure Does Not Support Dynamic Computation, Leaving a Gap in the Post-Training Pipeline

**The assumption or constraint.** Modern LLM training pipelines typically include a reinforcement learning stage (RLHF, RLVR) after supervised fine-tuning to align model behavior with human preferences or verifiable reward signals. The paper attempted to apply RLVR to Ouro using DAPO and GRPO on the DAPO-17K dataset (Section 4.5) but encountered a fundamental infrastructure problem: standard rollout engines (vLLM, SGLang) assume a fixed execution path through the model, which breaks under LoopLM's variable-depth computation. The paper documents two attempted workarounds, neither successful.

**The consequence.** The Ouro-Thinking models are produced via SFT only, without the RL alignment stage that is standard practice for state-of-the-art reasoning models. The paper states that the RL attempts "did not yield significant performance gains over the final SFT checkpoint" (Section 4.5). This means either: (1) the SFT checkpoint is already near the performance ceiling for these model scales, and RL would not have helped regardless of infrastructure; or (2) the attempted RL approaches were suboptimal due to the infrastructure mismatch, and a properly implemented on-policy RLVR pipeline could have produced further gains. The paper leans toward explanation (1), speculating that "after having already undergone extensive SFT, these smaller models may have limited headroom for RL gains." But the infrastructure barrier prevents testing this hypothesis definitively. Furthermore, the inability to run standard RL pipelines means that any future improvements to RL algorithms for reasoning (which is an active and rapidly advancing area) cannot be easily applied to LoopLM without specialized infrastructure development.

This is a practical deployment concern: a team adopting LoopLM would need to either (a) develop custom RL infrastructure that handles dynamic computation graphs, or (b) accept that their post-training pipeline stops at SFT. The paper's discussion of the vLLM/SGLang incompatibility (Section 4.5) suggests that this is a non-trivial engineering challenge—the attempted workaround of fixed 4-round rollouts with dynamic early-exit inference did not work despite training proceeding normally.

**What evidence exists in the paper.** Section 4.5 provides a detailed account of the two failed RLVR attempts and the infrastructure incompatibility. The paper does not report quantitative results from these experiments (e.g., "performance did not surpass the SFT checkpoint")—presumably because they did not improve and there are no positive results to report. The authors note that "the model still used fewer rounds at inference when beneficial despite being trained at four rounds. The mechanism behind this generalization remains unclear," which suggests that the dynamic exit behavior is at least partially preserved during fixed-depth RL training, but the paper does not evaluate whether this preserved behavior produces correct or optimal exit decisions.

**Mitigation status.** Not solved. The paper states: "We will further explore RL alignment for this architecture as we continue to develop infrastructure that can fully support LoopLM's dynamic computation" (Section 4.5). This is explicitly flagged as future work, and no timeline or approach is proposed. The three deployment advantages proposed in Section 7.3 (speculative decoding via draft-verify using intermediate heads, joint acceleration and pre-emptive safety screening, anytime generation with monotone refinement) may partially compensate for the lack of RL alignment by providing alternative mechanisms for controlling output quality and safety, but they do not replace the reward-based optimization that RL provides over the output distribution.

### 6.6 Comparative Baselines Are Not Controlled for Training Data Quantity or Quality

**The assumption or constraint.** The paper compares Ouro models against a range of baseline models (Qwen3, Gemma3, Llama3.1/3.2, DeepSeek-Distill) that were trained on different data quantities (ranging from 2T to 36T tokens), different data compositions, and different training recipes. Ouro is trained on 7.7T tokens of the authors' curated data mixture (Tables 3–5). The baseline models were trained by their respective organizations with varying levels of data curation, quality filtering, and domain-specific data augmentation. The paper does not train a matched standard transformer baseline on the same 7.7T-token data mixture.

**The consequence.** The headline comparison—Ouro 1.4B matching or exceeding 4B models, Ouro 2.6B matching or exceeding 8B models—confounds the effect of the LoopLM architecture with the effect of the training data. If the Ouro data mixture is higher quality or more reasoning-focused than the data used to train, say, Qwen3-4B (which was trained on 36T tokens of largely web-derived data), then some fraction of Ouro's performance advantage may be attributable to data rather than architecture. Conversely, Qwen3-4B was trained on approximately 4.7× more tokens than Ouro 1.4B (36T vs. 7.7T), which would be expected to improve its performance—so Ouro's ability to match or exceed it on reasoning tasks is actually *more* impressive given the token disadvantage. But without a matched-training ablation, we cannot quantify either effect independently.

The paper's control is incomplete in a specific way: the existing baselines were trained under different philosophies. Gemma3-4B was trained on only 4T tokens (less than Ouro's 7.7T), so its underperformance is partly expected. Qwen3-4B was trained on 36T tokens (much more than Ouro), making Ouro's competitive performance more notable. But Qwen3-4B's data composition is unknown and may have emphasized different capabilities. The only way to cleanly isolate the architectural effect would be to train a standard 1.4B transformer on the exact same 7.7T-token Ouro data mixture and compare. This experiment is not performed.

**What evidence exists in the paper.** Tables 7–8 report the total training tokens for each baseline model (Ouro 1.4B: 7.7T; Gemma3-1B: 2T; Llama3.2-1.2B: 9T; Qwen2.5-1.5B: 18T; Qwen3-1.7B: 36T; etc.). The large variance in these numbers (2T to 36T) makes clear that training data quantity is not controlled. Appendix Tables 3–5 describe the Ouro data mixture in detail, but no equivalent description is provided for the baseline models. The paper does not discuss this as a confounding factor or limitation, and does not attempt to correct for it (e.g., by estimating what fraction of performance gaps are attributable to data vs. architecture).

**Mitigation status.** Not addressed. The paper does not train a matched-data baseline, does not perform a data ablation, and does not discuss this as a threat to validity. The scaling law analysis in Appendix D compares LoopLM and standard models trained on the same data (FineWeb-Edu, 20B tokens), but at scales (53M–1.4B parameters, 20B tokens) that are far below the main experiments (1.4B–2.6B parameters, 7.7T tokens). The small-scale scaling curves (Figure 13) show that standard models outperform LoopLM at matched model sizes, consistent with the idea that the architectural advantage emerges at larger scales—but this finding doesn't resolve the data-confounding issue at the 1.4B/2.6B scale.

## 7. Implications and Future Directions
- Field‑level impact
  - Establishes recurrent depth as a practical, scalable axis of LLM capability. For reasoning‑centric tasks, looping can substitute for parameters—e.g., a 2.6B LoopLM competes with 8B dense models (Table 8)—and enables adaptive computation via early exit.
- Practical applications
  - Latency/compute‑aware deployment via Q‑exit thresholding (Algorithm 1), anytime generation, and built‑in draft‑and‑verify speculative decoding using intermediate heads (Section 7.3).
  - Memory‑efficient decoding with last‑step cache reuse enables deployment on constrained hardware (Section 5.4.2; Table 14).
  - Safer outputs through deeper latent refinement, even at the same parameter count (Figure 8).
- Research directions
  - Train at deeper loop counts and study methods to improve extrapolation beyond the trained depth (Section 8).
  - Better theoretical underpinnings of latent reasoning with tied weights; the paper provides an `O(log D)` construction for graph reachability with loops (Appendix B.5), suggesting efficiency advantages versus discrete/continuous CoT.
  - Improved RL infrastructure for dynamic‑depth models and principled safety/faithfulness evaluation that goes beyond proxy graders.
  - Multilingual tokenization and specialized vocabularies (e.g., math/code symbols) to lift limits acknowledged in Section 4.1.

> Primary takeaway: Looping the same layers turns depth from a static architectural choice into an input‑adaptive computation budget, yielding strong parameter efficiency and emergent benefits in safety and faithfulness when trained at scale with a uniform‑prior, entropy‑regularized halting objective and a loss‑improvement‑aligned gate.
