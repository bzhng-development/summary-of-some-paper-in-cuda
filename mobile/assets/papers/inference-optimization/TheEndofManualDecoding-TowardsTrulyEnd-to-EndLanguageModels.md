# The End of Manual Decoding: Towards Truly End-to-End Language Models

**ArXiv:** [2510.26697](https://arxiv.org/abs/2510.26697)

## 🎯 Pitch

AutoDeco is a novel, lightweight enhancement to transformer-based language models that enables them to dynamically predict their own decoding parameters—temperature and top-p—at each generation step, thus realizing truly end-to-end and context-adaptive text generation. By replacing static, hand-tuned decoding hyperparameters with differentiable, learned predictions (via a new soft top-p mechanism), AutoDeco eliminates manual decoding, consistently boosts accuracy across tasks, and introduces the unprecedented ability for models to interpret and execute natural language decoding instructions—significantly advancing both the usability and control of LLMs.

---

## 1. Executive Summary

This paper introduces **AutoDeco**, a lightweight architecture that enables truly end-to-end language model generation by learning to predict its own decoding parameters dynamically. Augmenting standard transformers with small MLP heads that predict token-level temperature and top-p values from each step's hidden state, AutoDeco eliminates the laborious manual tuning of static decoding hyperparameters without adding meaningful latency (only 1–2% overhead). Across eight benchmarks spanning math reasoning, QA, code generation, and instruction following—and integrated into Llama-Nemotron-8B, R1-Distill-Qwen-7B, Qwen3-30B-A3B, and GPT-OSS-20B—AutoDeco consistently outperforms both greedy search and default sampling baselines while matching the performance of expert-guided oracle tuning derived from hacking the test set, establishing that dynamic, self-regulated decoding can substitute for exhaustive manual hyperparameter sweeps with near-zero computational cost. Most strikingly, the authors uncover an emergent capability for **instruction-based decoding control**: when prompted with natural language commands like "generate with low randomness," the model spontaneously adjusts its predicted temperature down by 0.11 and top-p down by 0.06 on a token-by-token basis, achieving 95%+ consistency after targeted training—demonstrating that the model learns not merely what to generate but the meta-skill of how to modulate its own stochasticity in response to user intent.

## 2. Context and Motivation

### The Core Problem: Your State-of-the-Art Language Model Isn't Actually End-to-End

The paper opens with a provocative claim: the "end-to-end" label routinely applied to modern large language models is, in practice, a misnomer. While the neural architecture itself is trained end-to-end—input tokens flow through transformer layers to produce output logits—the final step that converts those logits into actual text relies on a **non-differentiable, manually configured decoding process** that sits outside the model's learned parameters. This is the gap AutoDeco aims to close.

The specific problem is deceptively simple. When you prompt an LLM like GPT-4 or Llama, the model produces a vector of logits over its vocabulary. To actually generate text, you must choose:

- **Temperature ($T$)**: A parameter that divides the logits before the softmax, controlling the sharpness of the probability distribution. Low temperature ($T \rightarrow 0$) makes the model nearly deterministic (it almost always picks the highest-probability token), while high temperature ($T > 1$) flattens the distribution, increasing diversity but also the risk of nonsense.
- **Top-p (nucleus sampling threshold)**: After sorting tokens by probability, the model retains only the smallest set whose cumulative probability exceeds $p$, truncating the long tail of low-probability tokens. This prevents the model from sampling from the vast pool of extremely unlikely tokens that collectively introduce noise.
- **Top-k**: An alternative truncation method that simply keeps the $k$ most probable tokens.

These are not learned parameters. They are **external knobs** that a human must set before generation begins—and critically, the same knob settings apply to **every single token** in the entire output sequence. The paper's key insight is that this static, one-size-fits-all approach is fundamentally misaligned with how language generation actually works: different tokens, different positions, and different contexts within a single generation demand different levels of stochasticity.

> "a single static configuration is inherently suboptimal because the ideal level of stochasticity varies dramatically within a single generation. For instance, a model might need high creativity to explore initial reasoning paths but high precision to deliver the final answer."

This is not merely a philosophical objection. It means that regardless of how carefully you tune your temperature and top-p, you are guaranteed to be using suboptimal settings for a significant fraction of the tokens in every generation. The model might need high temperature when brainstorming premises but low temperature when computing the final arithmetic—and a static configuration cannot satisfy both simultaneously.

### Why This Matters: The Hidden Cost of "Manual Decoding"

The practical consequences of this gap are substantial and multi-dimensional.

**First, the tuning burden is enormous.** The paper cites commercial API providers like DeepSeek, who "explicitly recommend different temperature settings for distinct application scenarios." Developers building on top of LLMs must conduct laborious grid searches over hyperparameter space to find settings that work for their specific task. This process is:

- **Computationally expensive**: Each candidate configuration requires generating complete outputs and evaluating them, often on large validation sets.
- **Task-dependent**: The optimal temperature for math reasoning (where precision matters) is typically much lower than for creative writing. This means the tuning process must be repeated for every new task or deployment context.
- **Inherently imperfect**: Even the "optimal" static setting found through exhaustive search represents a compromise—it's the best single setting averaged over an entire task distribution, not the best setting for each individual token or even each individual prompt.

The paper quantifies this dependency starkly in Figure 3: for Llama-Nemotron-8B, the optimal static temperature on BRUMO25 is $T = 0.8$ with $p = 0.9$, while on GPQA-Diamond the optimal drops to $T = 0.3$ with $p = 0.6$. A developer deploying a single model to handle both types of queries faces an impossible choice: optimize for one task and underperform on the other, or accept mediocrity on both.

**Second, the problem runs deeper than task-level variation.** Even within a single task, the optimal decoding strategy varies by **difficulty**. The paper observes in Table 1 that the performance gain from AutoDeco is more pronounced on certain models and benchmarks than others. For instance, Qwen3-30B-A3B-Instruct-2507 shows a smaller absolute improvement from AutoDeco on math reasoning (average gain of ~0.5 points over Default Sampling) compared to R1-Distill-Qwen-7B (gain of ~2.6 points). The authors hypothesize this stems from output length: Qwen3's answers are shorter, so "the sensitivity of task accuracy to variations in sampling parameters is substantially lower." This reveals an important subtlety: the value of dynamic decoding is itself context-dependent, and a static approach cannot adapt to this meta-level variation either.

**Third, there is a conceptual gap between model capabilities and decoding strategy.** Modern LLMs are trained on enormous corpora and develop sophisticated internal representations of language, reasoning, and uncertainty. Yet at generation time, we ignore all of that learned knowledge about *when to be confident* versus *when to explore* and instead impose a single, externally chosen stochasticity level. The model might internally know that it is highly uncertain about the next token (the logit distribution is flat) or highly confident (one token dominates), but under a static temperature setting, it has no mechanism to act on that knowledge. The temperature is applied uniformly regardless of the model's internal uncertainty state.

### Where Existing Approaches Fall Short

The paper situates its contribution within a well-established taxonomy of decoding methods, identifying specific limitations in each category.

#### Deterministic Decoding: Safe but Sterile

Deterministic methods produce a single, reproducible output for a given input. The most fundamental is **greedy search**, which simply selects the highest-probability token at each step. **Beam search** extends this by maintaining $k$ parallel candidate sequences, selecting the overall most probable complete sequence rather than making irrevocable greedy choices at each step.

These methods work well for tasks where correctness and reproducibility are paramount—machine translation, factual question answering, mathematical computation—because they consistently select high-probability outputs. However, the paper notes they are "known to favor dull, high-frequency phrases" (citing Vijayakumar et al., 2016), producing repetitive, generic text that lacks the variability of natural human language. For open-ended generation (creative writing, dialogue, brainstorming), deterministic decoding is fundamentally unsuitable because it collapses the rich probability distribution learned by the model into a single point.

Crucially, deterministic methods represent an **extreme point** in the stochasticity spectrum: zero randomness. They provide no mechanism for the calibrated injection of diversity that many tasks require. The paper's contribution here is not to replace deterministic decoding but to enable the model to learn *when* deterministic behavior is appropriate and *when* stochasticity is needed—a decision that deterministic methods cannot make because they have no notion of a stochasticity knob.

#### Stochastic Sampling: Powerful but Manual

Stochastic methods sample from the model's output distribution rather than selecting the argmax. This introduces diversity but creates a new problem: unrestricted sampling from the full vocabulary produces incoherent text because the long tail of low-probability tokens collectively has significant probability mass and introduces noise.

Truncation methods address this by restricting the sampling pool:

- **Top-k sampling** (Fan et al., 2018): Only the $k$ most likely tokens are considered.
- **Nucleus sampling / top-p** (Holtzman et al., 2019): The smallest set of tokens whose cumulative probability exceeds $p$ is retained. This is more adaptive than top-k because it automatically adjusts the truncation size based on the shape of the distribution—a peaked distribution will have a small nucleus, while a flat distribution will have a larger one.

These methods, combined with temperature scaling, form the dominant decoding paradigm in deployed LLMs. The paper acknowledges their power but identifies a critical limitation:

> "Despite their power, as our introduction highlights, finding the optimal configuration for these hyperparameters is a non-trivial, task-dependent manual process."

The paper cites Shi et al. (2024), who conducted "a thorough examination of decoding methods in the era of LLMs" and documented the extent of the tuning burden. The fundamental issue is that temperature, top-p, and top-k are **static hyperparameters** set before generation and fixed for the entire output sequence. They cannot respond to:

- **Within-sequence dynamics**: The need for high creativity during brainstorming followed by high precision during final answer delivery.
- **Token-level uncertainty**: The model's own confidence about the next token, which varies dramatically from position to position.
- **Contextual demands**: The fact that function words (prepositions, articles) require very different stochasticity than content words (nouns, verbs) in open-ended generation.

#### Model-Based Decoding: Sophisticated but Still Static

A more recent class of methods modifies the model's output distribution using external signals or auxiliary models rather than fixed hyperparameters. The paper highlights several approaches:

**Contrastive decoding** (Li et al., 2023; Chuang et al., 2023) uses a smaller "amateur" language model to identify and penalize generic, high-probability tokens, steering the larger "expert" model toward more interesting outputs. The intuition is that tokens with high probability under both the expert and the amateur are likely generic filler, while tokens with high probability only under the expert are likely more substantive and interesting. By subtracting the amateur's log-probabilities from the expert's, contrastive decoding surfaces the expert's unique knowledge.

**Speculative decoding** (Leviathan et al., 2023; Chen et al., 2023) uses a faster "draft" model to generate candidate token sequences, which are then verified in parallel by the larger model. This dramatically accelerates inference without changing the output distribution.

**Plug-and-play language models** (Dathathri et al., 2019) use attribute classifiers to steer generation toward desired topics or styles by modifying the model's hidden states.

These methods are effective for their intended purposes, but the paper identifies a shared limitation that has been overlooked:

> "While they are effective, they still operate under a fixed algorithmic framework: the choice of the 'guidance model' itself acts as another form of hyperparameter. For example, in contrastive decoding and speculative decoding, the authors suggest that using a smaller LM of the same architecture as the guidance model yields the best results."

In other words, model-based methods shift the tuning burden from traditional hyperparameters (temperature, top-p) to **architectural and model choices** (which amateur model to use, what weighting scheme, what verification strategy). The decoding process becomes more sophisticated but no more adaptive: it still applies a single, fixed strategy to every token in every generation, regardless of context.

More fundamentally, none of these methods address the **dynamic control** problem. They provide sophisticated ways to modify the output distribution, but they do not enable the model itself to learn when and how to modulate its own stochasticity based on its internal state and the demands of the current context.

### The Missing Piece: Learned, Token-Level Self-Regulation

The paper's central insight is that the field has been asking the wrong question. Rather than "what is the best static decoding strategy?" the question should be: **can the model learn its own decoding strategy as part of the end-to-end training objective?**

This reframing connects to broader themes in machine learning where hand-designed components are progressively replaced by learned ones. In computer vision, hand-crafted features (SIFT, HOG) gave way to learned convolutional filters. In NLP, discrete symbolic representations gave way to learned embeddings. In reinforcement learning, hand-designed exploration schedules gave way to learned exploration policies. The paper positions decoding hyperparameters as the next hand-designed component ripe for learned replacement.

The specific gap AutoDeco fills is:

1. **Granularity**: Existing methods set decoding parameters at the task level or sequence level. AutoDeco predicts them at the **token level**, allowing the model to modulate its stochasticity independently for each token it generates.

2. **Adaptivity**: Existing methods use the same strategy regardless of the model's internal state. AutoDeco derives its predictions from the **model's own hidden state**, meaning the decoding strategy is informed by the model's learned representations of context, uncertainty, and task demands.

3. **End-to-end learning**: AutoDeco's parameters are learned by directly optimizing the final cross-entropy loss—the same objective used to train the base language model. This means the decoding strategy is optimized for what actually matters (predicting the correct next token) rather than being tuned via proxy metrics on downstream task performance.

4. **Zero additional latency**: Unlike model-based methods that require running auxiliary models or additional forward passes, AutoDeco's predictions are generated in parallel with the standard language modeling head using the same hidden state, adding only 1-2% overhead.

### How This Paper Positions Itself

The paper explicitly frames AutoDeco not as an incremental improvement to existing decoding methods but as a **paradigm shift**:

> "Instead of relying on fixed hyperparameters or predefined heuristics, we empower the model to dynamically control its own stochasticity at each generation step."

This is a stronger claim than simply proposing a better decoding algorithm. The authors are arguing that the **architectural boundary** between the language model and the decoding process should be eliminated entirely. Just as modern LLMs learned to perform attention, representation, and prediction within a unified architecture, they should also learn to control *how* those predictions are converted into text.

The paper's positioning is substantiated through several design choices:

- **Simplicity**: The AutoDeco heads are 2-layer MLPs—trivially small compared to the transformer layers they augment. This demonstrates that the benefit comes from the architectural principle (learned, dynamic, token-level control) rather than from a complex new mechanism.
- **Generalizability**: The method is validated across four model families (Llama, Qwen, GPT-OSS, and DeepSeek distilled models) spanning dense and mixture-of-experts architectures. Training on math reasoning generalizes zero-shot to QA, code, and instruction following—suggesting the model learns a fundamental meta-skill rather than task-specific heuristics.
- **The emergent ability as validation**: The most compelling evidence for the paradigm shift is the emergent instruction-based control described in Section 3.3. The fact that the model spontaneously learns to interpret abstract commands like "generate with low randomness" and translate them into concrete adjustments to its temperature and top-p predictions demonstrates that the AutoDeco architecture creates a genuine **interface between language understanding and generation control** that does not exist in standard architectures. This capability is not explicitly trained—it emerges from the architecture's design—which is strong evidence that the paradigm shift is real rather than merely aspirational.

The paper also positions itself relative to the practical reality of LLM deployment. It does not claim that AutoDeco makes all existing decoding research obsolete. Deterministic methods, contrastive decoding, and speculative decoding address different problems (reproducibility, factuality, latency) that are orthogonal to the dynamic stochasticity problem AutoDeco solves. The contribution is additive: AutoDeco can be integrated with these methods because it operates at the level of logit manipulation before the final sampling step, making it compatible with any downstream decoding algorithm.

### Why This Problem Hasn't Been Solved Before

The paper implicitly addresses why learned, dynamic decoding has not been proposed earlier by tackling the two obstacles that make it technically challenging:

**First, the non-differentiability barrier.** The standard top-p sampling algorithm involves a hard cutoff—tokens whose cumulative probability exceeds $p$ are retained with their original probabilities, and all other tokens are assigned zero probability. This step function is non-differentiable, meaning gradients cannot flow from the final loss back to any parameter that influences the top-p threshold. Without a differentiable top-p, there is no way to train a top-p prediction head using standard gradient-based optimization.

The paper's solution—a differentiable soft top-p using a smooth exponential decay mask—is the technical innovation that makes end-to-end training possible. The key insight in Equation 2 is that the mask function $m = \exp(-\alpha \cdot \text{ReLU}(c - \hat{P}))$ is everywhere differentiable with respect to $\hat{P}$, the predicted top-p threshold. The ReLU ensures tokens inside the nucleus ($c \leq \hat{P}$) get a mask of 1.0 (since $\text{ReLU}(0) = 0$ and $\exp(0) = 1$). Tokens outside the nucleus get exponentially decaying masks as their cumulative probability exceeds $\hat{P}$. The steepness parameter $\alpha$ controls the trade-off between accurately approximating the hard cutoff (large $\alpha$) and maintaining smooth gradients (small $\alpha$). The paper uses $\alpha = 30$, which produces a relatively sharp transition while remaining differentiable.

**Second, the absence of ground-truth labels.** There are no "correct" temperature and top-p values for any given token in any training dataset. Supervised datasets contain input-output pairs, not optimal decoding trajectories. The paper's response is that explicit labels are unnecessary: the temperature and top-p heads can be trained by backpropagating through the entire differentiable pipeline, optimizing them directly for the final cross-entropy loss on the next-token prediction task. This is the "end-to-end" claim in its strongest form—the model learns to set its own decoding parameters because doing so demonstrably reduces prediction error, without any external guidance about what those parameters should be.

The training refinements—Easy-Token Masking and Dynamic Fine-Tuning—address specific pathologies that arise from this purely loss-driven approach. Without them, the temperature head learns to predicts near-zero temperature for "easy" tokens where the base model already assigns high probability to the correct answer. While locally optimal, this biases the head toward excessive determinism. Easy-Token Masking randomly masks the loss on these positions (60% masking rate), forcing the head to learn from the minority of challenging tokens where calibrated stochasticity is beneficial. Dynamic Fine-Tuning re-weights the training loss to focus on tokens where the model has a "reasonable prior," preventing outlier high-uncertainty tokens from distorting the learned temperature scale.

### Summary of the Motivational Landscape

The paper addresses a recognized but unsolved problem: **LLM decoding quality depends critically on hyperparameters that are set statically and tuned manually, despite the fact that optimal stochasticity varies per-token, per-context, and per-task.** Existing solutions either (a) accept the manual tuning burden and live with static suboptimality, (b) use deterministic methods that sacrifice diversity, or (c) employ sophisticated model-based methods that shift the tuning burden to different hyperparameters without achieving dynamic adaptation.

AutoDeco's contribution is to demonstrate that **the model can learn to set its own decoding parameters** by making the sampling process differentiable and training lightweight prediction heads end-to-end. The technical challenge was non-differentiability of top-p sampling; the solution is a smooth approximation. The training challenge was the absence of supervision; the solution is to let the final task loss provide the signal. The result is not just better performance—it is a fundamentally different relationship between the model and its generation process, one where the model has learned to regulate its own stochasticity in response to both its internal state and, remarkably, to explicit natural language commands from the user.

## 3. Technical Approach

### 3.1 Reader Orientation

AutoDeco is a lightweight augmentation to existing transformer language models that adds two small neural network heads which predict, for each token the model generates, the optimal temperature and top-p values to use when sampling that token from the model's output distribution. The system solves the problem that optimal decoding hyperparameters vary dramatically across tokens, contexts, and tasks—yet all existing methods force a single static configuration on every token of every generation—by making the sampling process itself differentiable and training the prediction heads end-to-end on the same next-token prediction objective that trained the base model, allowing the model to learn its own context-specific decoding strategy without any external supervision signals for what the "correct" temperature or top-p should be.

### 3.2 Big-Picture Architecture (Diagram in Words)

The AutoDeco system has four major components operating in a single forward pass per token:

1. **Base LLM (frozen)** — any pre-trained transformer model (Llama, Qwen, GPT-OSS) that computes hidden states at each generation step. Its parameters remain unchanged; it serves as the feature extractor and language model.

2. **Temperature Prediction Head** — a 2-layer MLP that takes the current hidden state as input and outputs a scalar `$\hat{T}$`, the predicted optimal temperature for the current token. This value will be used to scale the logits before softmax.

3. **Top-p Prediction Head** — a second 2-layer MLP that takes *both* the hidden state and the just-predicted `$\hat{T}$` as input and outputs a scalar `$\hat{P}$` in `$[0, 1]$`, the predicted optimal nucleus sampling threshold. This head conditions on temperature because temperature scaling changes the shape of the probability distribution, which affects what top-p threshold is appropriate.

4. **Differentiable Soft Top-p Module (training only)** — during training, replaces the standard hard-cutoff top-p sampling with a smooth, differentiable approximation that enables gradient flow from the final cross-entropy loss back through both the `$\hat{P}$` prediction and the `$\hat{T}$` prediction. During inference, standard hard top-p sampling is used (since gradients are no longer needed).

The information flow per token is: the base LLM computes a hidden state → the temperature head predicts `$\hat{T}$` from that hidden state → the top-p head predicts `$\hat{P}$` from the hidden state and `$\hat{T}$` jointly → the standard language modeling head produces logits → the logits are divided by `$\hat{T}$` → softmax produces probabilities → the soft top-p mask (training) or hard top-p filter (inference) is applied using `$\hat{P}$` → the resulting distribution is used to sample or compute loss for the next token.

### 3.3 Roadmap for the Deep Dive

- **First**, the formal training problem and why it is difficult—the non-differentiability of standard top-p sampling that prevents naive end-to-end training, establishing the core technical obstacle that motivates the entire approach.
- **Second**, the differentiable soft top-p mechanism—the mathematical transformation that replaces the hard cutoff with a smooth approximation, enabling gradient flow from the final loss back to both prediction heads.
- **Third**, the training procedure and loss function—how the temperature and top-p heads are optimized jointly with the base language model's objective without any token-level ground-truth labels for the optimal parameter values, including the two data debiasing techniques (Easy-Token Masking and Dynamic Fine-Tuning) that prevent degenerate learned behaviors.
- **Fourth**, the inference procedure—how the predicted `$\hat{T}$` and `$\hat{P}$` values are used at test time to dynamically modulate the sampling distribution with near-zero additional latency, and the design choice to use standard hard top-p during inference despite having trained with soft top-p.
- **Fifth**, the training data and hyperparameter configuration—the dataset, optimization settings, and architectural choices that enable efficient training of the AutoDeco heads in only 400 steps.

### 3.4 Detailed, Sentence-Based Technical Breakdown

This is primarily a **systems and methods paper** whose core idea is that the decoding hyperparameter selection process—traditionally an external, manual, and static procedure—can be internalized into the model's own learned parameters by making the sampling operations differentiable and training lightweight prediction heads end-to-end on the standard language modeling objective.

---

#### The Core Training Problem: Non-Differentiability of Standard Top-p Sampling

The paper's central technical challenge is straightforward to state but required a non-obvious solution: we want to train two small neural networks to predict temperature `$\hat{T}$` and top-p threshold `$\hat{P}$` for each token, using only the final cross-entropy loss as the training signal, but top-p sampling as it is normally implemented contains a step function that is non-differentiable, which means gradients cannot flow from the loss back to the top-p head.

To understand why this is, consider the standard top-p (nucleus) sampling algorithm. Given a probability distribution `$p$` over the vocabulary and a threshold `$P \in [0, 1]$`, the algorithm:

1. Sorts the tokens by probability in descending order.
2. Computes the cumulative sum of these sorted probabilities.
3. Finds the smallest index `$k$` such that the cumulative probability up to and including token `$k$` exceeds `$P$`.
4. Sets the probability of all tokens after index `$k$` to exactly zero.
5. Renormalizes the remaining probabilities to sum to 1.

The problematic operation is step 4: the hard cutoff. For any token `$i$`, the decision of whether to keep it or zero it out is a discrete function of `$P$`—if the cumulative probability at `$i$` is less than or equal to `$P$`, the token survives; otherwise, it is eliminated. The gradient of this operation with respect to `$P$` is zero almost everywhere (the probability of each token changes discretely, not smoothly, as `$P$` varies) and undefined at the cutoff boundary. This means that if `$\hat{P}$` is the output of a neural network, the chain rule `$\partial \mathcal{L} / \partial \hat{P} = (\partial \mathcal{L} / \partial \tilde{p}) \cdot (\partial \tilde{p} / \partial \hat{P})$` fails because `$\partial \tilde{p} / \partial \hat{P}$` does not exist or is zero.

The temperature parameter does not suffer from this problem. Temperature scaling is applied as:

$$p = \text{softmax}\left(\frac{l}{\hat{T}}\right)$$

where `$l$` is the vector of logits from the language model head. The softmax function is everywhere differentiable with respect to its input, and the input is differentiable with respect to `$\hat{T}$` (the derivative of `$l / \hat{T}$` with respect to `$\hat{T}$` is `$-l / \hat{T}^2$`, which exists and is non-zero for any finite `$\hat{T} > 0$`). This means a temperature prediction head could be trained end-to-end with standard backpropagation—no special machinery needed. The non-differentiability is exclusively in the top-p operation.

The paper could have sidestepped this problem by only predicting temperature and leaving top-p as a fixed hyperparameter. Indeed, the ablation study (Figure 4) shows that a temperature-only head already provides substantial gains over default sampling. However, the full AutoDeco with both heads outperforms either head alone, indicating that the two parameters provide complementary control—temperature modulates the overall entropy of the distribution, while top-p truncates the low-probability tail that temperature alone cannot cleanly eliminate. Solving the non-differentiability problem for top-p is therefore necessary to achieve the full benefit of dynamic, multi-parameter decoding control.

---

#### The Differentiable Soft Top-p Mechanism

The paper's solution is to replace the hard-cutoff top-p operation with a smooth, differentiable approximation during training. This approximation is used only during training to enable gradient flow; at inference time, standard hard top-p sampling is used because gradients are no longer needed and the hard cutoff provides cleaner truncation.

The differentiable soft top-p mechanism works in three stages, applied sequentially to the probability distribution at each training step.

**Stage 1: Temperature-scaled probabilities.**

The raw logits `$l$` produced by the language model head are first scaled by the predicted temperature `$\hat{T}$` and passed through softmax to produce an initial probability distribution:

$$p = \text{softmax}\left(\frac{l}{\hat{T}}\right)$$

where `$l \in \mathbb{R}^{|V|}$` is the vector of logits over the vocabulary `$V$`, `$\hat{T} \in \mathbb{R}^+$` is the scalar temperature predicted by the temperature head, and `$p \in \mathbb{R}^{|V|}$` is the resulting probability distribution satisfying `$\sum_i p_i = 1$` and `$p_i \geq 0$` for all `$i$`.

**What it computes:** each logit `$l_i$` is divided by `$\hat{T}$`, shrinking the dynamic range of the logits when `$\hat{T} > 1$` (making the distribution more uniform) and expanding it when `$\hat{T} < 1$` (making the distribution more peaked around the maximum). The softmax then exponentiates and normalizes. The output `$p$` is a proper probability distribution whose entropy is controlled by `$\hat{T}$`.

**Why this form:** temperature scaling multiplies the logits by a constant factor before the exponentiation in softmax. Because `$\text{softmax}(l / T)_i = \exp(l_i / T) / \sum_j \exp(l_j / T)$`, dividing by a larger `$T$` reduces the differences between exponentiated values, flattening the distribution. This is the standard temperature scaling used in all LLM decoding; the innovation is that `$\hat{T}$` is not a fixed constant but a learned function of the hidden state.

**Stage 2: Differentiable soft mask generation.**

After obtaining the temperature-scaled probabilities `$p$`, the system sorts them in descending order and computes their cumulative sum `$c$`. The soft mask is then computed as:

$$m^{\text{(sorted)}} = \exp\left(-\alpha \cdot \text{ReLU}(c - \hat{P})\right)$$

where `$c \in \mathbb{R}^{|V|}$` is the vector of cumulative probabilities after sorting (each element `$c_k$` is the sum of the `$k$` largest probabilities in `$p$`), `$\hat{P} \in [0, 1]$` is the scalar top-p threshold predicted by the top-p head, `$\alpha \in \mathbb{R}^+$` is a steepness hyperparameter set to 30, and `$m^{\text{(sorted)}} \in \mathbb{R}^{|V|}$` is the resulting mask vector in sorted order (which will be unsorted to match the original vocabulary order before application).

**What it computes:** for each position `$k$` in the sorted probability distribution, the mask value `$m^{\text{(sorted)}}_k$` is:
- 1.0 if `$c_k \leq \hat{P}$` (the token is "inside the nucleus"), because then `$\text{ReLU}(c_k - \hat{P}) = 0$` and `$\exp(0) = 1$`.
- A value between 0 and 1 that decays exponentially toward zero as `$c_k$` exceeds `$\hat{P}$` (the token is "outside the nucleus"), because `$\text{ReLU}(c_k - \hat{P}) = c_k - \hat{P}$`, which grows positive, and `$\exp(-\alpha \cdot (c_k - \hat{P}))$` produces exponential decay.

The parameter `$\alpha$` controls the steepness of this decay. With `$\alpha = 30$`, a token whose cumulative probability exceeds `$\hat{P}$` by 0.1 receives a mask value of `$\exp(-30 \cdot 0.1) = \exp(-3) \approx 0.05$`, meaning its probability is scaled down by 95%. Tokens just barely beyond the threshold (exceeding by 0.01) receive a mask of `$\exp(-30 \cdot 0.01) \approx 0.74$`, a more modest reduction. This creates a smooth transition region rather than an instantaneous zeroing.

**Why this form:** the combination of ReLU and exponential decay is deliberately chosen to be everywhere differentiable with respect to `$\hat{P}$`. The derivative `$\partial m_k / \partial \hat{P}$` exists for all `$k$` because: (a) when `$c_k \leq \hat{P}$`, the ReLU outputs zero, the exponential outputs 1, and the gradient with respect to `$\hat{P}$` is zero (the mask doesn't change if `$\hat{P}$` increases further while still exceeding `$c_k$`); (b) when `$c_k > \hat{P}$`, the ReLU outputs `$c_k - \hat{P}$`, and the chain rule gives `$\partial m_k / \partial \hat{P} = \exp(-\alpha \cdot (c_k - \hat{P})) \cdot (-\alpha) \cdot (-1) = \alpha \cdot \exp(-\alpha \cdot (c_k - \hat{P})) > 0$`—a positive, non-zero gradient that flows from the mask value back to `$\hat{P}$`. This gradient is always well-defined and non-vanishing for tokens near the boundary, enabling the top-p head to learn.

The choice of exponential decay (rather than linear or polynomial) is important because it ensures the mask values for tokens far outside the nucleus approach zero rapidly enough to approximate a hard cutoff, while maintaining non-zero gradients everywhere. A linear decay would either be too gentle (tokens far outside still have non-negligible probability) or too steep (approaching a step function, with near-zero gradients). The exponential form with `$\alpha = 30$` balances these concerns: it produces a sharp transition (as shown in Figure 2a, the mask drops from 1.0 to near 0.0 within a cumulative probability interval of roughly 0.1–0.2) while maintaining meaningful gradients throughout the transition region.

**Stage 3: Final renormalized distribution.**

The soft mask `$m$` (after unsorting to match the original vocabulary order) is applied element-wise to the initial probabilities `$p$`, and the result is renormalized to sum to 1:

$$\tilde{p} = \frac{p \odot m}{\sum(p \odot m) + \epsilon}$$

where `$p \in \mathbb{R}^{|V|}$` is the original temperature-scaled probability distribution from Stage 1, `$m \in \mathbb{R}^{|V|}$` is the soft mask from Stage 2 (unsorted to the original vocabulary indexing), `$\odot$` denotes element-wise multiplication, `$\epsilon$` is a small constant for numerical stability (preventing division by zero if all mask values collapse to zero), and `$\tilde{p} \in \mathbb{R}^{|V|}$` is the final differentiable probability distribution used for computing the cross-entropy loss.

**What it computes:** each vocabulary token's temperature-scaled probability is multiplied by its corresponding mask value (close to 1.0 for tokens inside the nucleus, decaying to near-zero for tokens outside), and then the entire vector is renormalized. The effect is that tokens outside the nucleus have their relative probability mass suppressed—not eliminated entirely (which would be non-differentiable) but scaled down by a factor that depends continuously on how far outside the nucleus they are. The renormalization then redistributes this suppressed mass proportionally among the surviving tokens.

**Why this form:** element-wise multiplication by the mask followed by renormalization preserves the relative ordering of probabilities within the nucleus (all those tokens are multiplied by approximately 1.0) while smoothly attenuating the tail. An alternative approach—setting tail token probabilities exactly to zero and renormalizing—would produce the standard hard top-p but break differentiability. Another alternative—not renormalizing—would produce a distribution that does not sum to 1, which would corrupt the cross-entropy loss. The renormalization step ensures `$\tilde{p}$` is always a valid probability distribution regardless of the mask values, which is necessary for the cross-entropy loss to be well-defined.

**Figure 2b provides a concrete illustration** with a vocabulary size of 50. The original probability distribution (shown in blue) has a relatively flat tail. When the soft mask is applied (shown in red), the bulk of the probability mass within the nucleus is preserved, while the tail probabilities are visibly suppressed but not zeroed out. This produces a distribution that closely approximates hard top-p sampling while maintaining smooth dependence on `$\hat{P}$`.

**Critical design choice: soft top-p during training, hard top-p during inference.** The soft top-p mechanism is used exclusively during training to enable gradient flow. At inference time, the predicted `$\hat{P}$` is used with standard hard top-p sampling—the exact, non-differentiable cutoff algorithm described at the beginning of this section. This is valid because there is no training signal at inference; the model simply needs to produce the best possible distribution. Hard top-p produces cleaner truncation than soft top-p (no residual probability mass on tail tokens), and the model has learned through the soft approximation what `$\hat{P}$` values lead to good performance. The inference-time hard top-p is therefore a deployment optimization, not a departure from the training objective—the soft approximation served its purpose of creating a differentiable training signal, and the learned mapping from hidden state to `$\hat{P}$` transfers to the hard cutoff.

---

#### The End-to-End Training Procedure

With the differentiable soft top-p mechanism in place, training AutoDeco becomes a straightforward end-to-end optimization: the entire pipeline (base LLM hidden states → temperature head → top-p head → temperature scaling → softmax → soft top-p mask → renormalization → cross-entropy loss) is a single differentiable computation graph.

**The loss function.** The standard cross-entropy loss is computed between the final distribution `$\tilde{p}$` from the soft top-p pipeline and the ground-truth next token `$y^*$`:

$$\mathcal{L} = -\log \tilde{p}_{y^*}$$

where `$\tilde{p}_{y^*}$` is the probability assigned to the correct token in the final, dynamically-modulated distribution. This is exactly the same loss function used to train the base language model—there is no auxiliary loss, no regularization term, and no separate objective for the temperature or top-p heads.

**What it computes:** the negative log probability of the correct token under the dynamic distribution. If `$\tilde{p}_{y^*}$` is high (the dynamic sampling process assigns high probability to the correct token), the loss is low. The gradient of this loss with respect to `$\tilde{p}$` is `$-1/\tilde{p}_{y^*}$` at the index of the correct token and zero elsewhere.

**Why this form:** using the standard next-token prediction loss means the temperature and top-p heads are trained to optimize exactly what matters—the model's ability to assign high probability to the correct continuation. There is no need for "ground-truth" temperature or top-p values because the training signal comes entirely from whether the dynamically-modulated distribution succeeds at the fundamental language modeling task. If predicting a high temperature helps (because the raw logits are overconfident and need flattening to give the correct token more mass), the gradient will push `$\hat{T}$` upward. If predicting a low top-p helps (because the correct token is in the high-probability nucleus and tail tokens are distracting), the gradient will push `$\hat{P}$` downward.

**The gradient flow.** Because every operation in the pipeline is differentiable:
- `$\partial \mathcal{L} / \partial \tilde{p}$` propagates through the renormalization to `$\partial \mathcal{L} / \partial m$` and `$\partial \mathcal{L} / \partial p$`.
- `$\partial \mathcal{L} / \partial m$` propagates through the exponential and ReLU to `$\partial \mathcal{L} / \partial \hat{P}$` (updating the top-p head) and `$\partial \mathcal{L} / \partial p$` (through the element-wise multiplication).
- `$\partial \mathcal{L} / \partial p$` propagates through the softmax and the division by `$\hat{T}$` to `$\partial \mathcal{L} / \partial \hat{T}$` (updating the temperature head).
- Both heads also receive gradients through the base LLM's hidden states (since both take the hidden state as input), but the base model parameters are frozen, so these gradients are not used to update the transformer weights.

This means a single backward pass simultaneously trains both heads to make predictions that, in combination, maximize the probability of the correct next token.

**Training data: reject sampling trajectories from DeepMath-103K.** The AutoDeco heads are trained on a specialized dataset generated through reject sampling. The base models (Llama-Nemotron-8B, R1-Distill-Qwen-7B, Qwen3-30B-A3B, GPT-OSS-20B) are used to sample multiple solutions to problems from the DeepMath-103K dataset (He et al., 2025). Reject sampling means that only trajectories where the final answer is correct are retained for training. This filtering is crucial because it ensures the training data consists of high-quality reasoning chains that successfully reach the correct answer. When the AutoDeco heads are trained on these trajectories, they learn to predict temperature and top-p values that are consistent with producing correct answers—the model learns that successful reasoning is associated with particular patterns of stochasticity modulation.

The choice of mathematics as the training domain is deliberate but not restrictive. The paper demonstrates in Section 3.2.1 that heads trained exclusively on math generalize zero-shot to QA, code generation, and instruction following. This suggests that the skill of modulating decoding stochasticity is largely domain-independent: the patterns of when to be deterministic versus creative in mathematical reasoning transfer to other types of text generation.

**The base model is frozen.** Throughout training, all parameters of the base language model remain unchanged. Only the temperature head and top-p head parameters are updated. This is a critical design choice with several motivations:

- **Efficiency:** Training only the heads (which are 2-layer MLPs with negligible parameter count compared to the transformer) requires minimal computation. The paper reports that training converges in approximately 400 steps and 6,000 training samples (Figure 6 in the Appendix).
- **Preservation of base capabilities:** Freezing the base model ensures that the language model's fundamental knowledge, reasoning ability, and generation quality are not altered. The AutoDeco heads learn to modulate the *expression* of existing capabilities, not to change the capabilities themselves.
- **Modularity and portability:** Because the base model is untouched, AutoDeco can be added to any pre-trained model without re-training or fine-tuning the core transformer. This is what enables the drop-in integration across four model families.

The paper acknowledges in Section 5 that this frozen-backbone approach has limitations—specifically, the imprecise control observed in the instruction-based decoding experiments (Section 3.3) is hypothesized to stem from the base model being unable to adjust its internal representations to the AutoDeco heads. Joint training of the base model and the heads is identified as an important direction for future work.

---

#### Data Debiasing Techniques: Easy-Token Masking and Dynamic Fine-Tuning

Training the heads purely on the cross-entropy loss over reject-sampled trajectories produces reasonable results, but the paper identifies two specific failure modes that degrade performance. Two debiasing techniques are introduced to address them.

**Easy-Token Masking.**

The problem: for many tokens in the training data, the base model's greedy prediction (the argmax of the logits) already matches the ground-truth correct token. For these "easy" tokens, the optimal temperature is near zero—if the model is already confident and correct, any stochasticity would risk sampling a wrong token. The cross-entropy loss naturally rewards predicting low temperature for these positions because low temperature sharpens the distribution, making the correct token even more probable.

The consequence is that, without intervention, the temperature head learns a strong bias toward predicting very low temperatures everywhere, because the majority of tokens in any training corpus are "easy" in this sense (function words, common syntactic structures, predictable continuations). This bias makes the head overly conservative, predicting low temperatures even on tokens where calibrated stochasticity would be beneficial (creative choices, reasoning exploration, diverse outputs).

**The mitigation:** Easy-Token Masking randomly masks (sets to zero) the training loss on a large fraction of positions where the base model's greedy prediction is correct. The paper uses a masking rate of 60%. Concretely, for each training example, the system identifies tokens where `$\arg\max_i l_i = y^*$` (the greedy prediction matches the ground truth). For 60% of these positions, selected at random, the loss contribution is zeroed out. The loss is computed and backpropagated only on: (a) the 40% of easy tokens that were not masked, and (b) all tokens where the greedy prediction is incorrect.

**Why this works:** by removing the training signal from the majority of easy tokens, the temperature head is forced to learn from the minority of challenging tokens. On these tokens, the correct behavior is often to predict a moderate or high temperature because the base model's raw logits are poorly calibrated—the correct token might not have the highest logit, and some stochasticity is needed to give it a chance of being sampled. The head thus learns to associate high uncertainty contexts with higher temperature predictions, while retaining the ability to predict low temperature on easy tokens (through the 40% of easy tokens that remain in the training signal). The paper reports that this masking is critical for preventing the temperature head from collapsing to a near-deterministic mode.

**Dynamic Fine-Tuning.**

The problem: even with Easy-Token Masking, the training data contains outlier tokens where the model's logits are extremely poorly calibrated—the correct token has very low probability under the base model. On these tokens, the cross-entropy loss would push the temperature head to predict extremely high values to flatten the distribution enough to give the correct token non-trivial mass. However, these outliers are often noise (tokens that are genuinely unpredictable from context) rather than genuine opportunities for beneficial stochasticity. If the head learns to predict very high temperature on these outliers, it overgeneralizes and predicts inappropriately high temperature on other challenging-but-not-impossible tokens.

The mitigation: Dynamic Fine-Tuning (Wu et al., 2025) re-weights the training loss to focus on tokens where the model has a "reasonable prior"—meaning the correct token has non-negligible probability under the base model's raw logits, even if it's not the top prediction. The paper does not specify the exact re-weighting formula, but the conceptual effect is: tokens where the base model assigns the correct token a probability above some threshold get higher weight in the loss, while tokens where the correct token's probability is near-zero get lower weight.

**Why this works:** Dynamic Fine-Tuning prevents the temperature head from being skewed by tokens that are essentially unpredictable—situations where no amount of temperature adjustment would make the correct token likely. It teaches the head to apply high temperatures "judiciously in situations of calibrated uncertainty"—when the model has several plausible options and needs stochasticity to explore among them—rather than in situations of fundamental ignorance. This produces more moderate, useful temperature predictions that genuinely improve generation quality without introducing gratuitous randomness.

**Interaction between the two techniques.** Easy-Token Masking and Dynamic Fine-Tuning address complementary problems. Easy-Token Masking prevents the head from being overly conservative (predicting temperature near zero everywhere) by removing the easy-token signal that dominates the training data. Dynamic Fine-Tuning prevents the head from being overly aggressive (predicting very high temperature on outliers) by downweighting the influence of tokens that are genuinely impossible to predict. Together, they shape the training signal so that the head learns to predict low temperature when the model is confident and correct, moderate temperature when the model faces genuine uncertainty among plausible alternatives, and avoids both the degenerate extremes of perpetual determinism and perpetual randomness.

**What happens without these techniques is not extensively ablated**, but the paper's strong emphasis on their necessity suggests that naïve end-to-end training without debiasing produces heads that are either trivially deterministic (collapsing to temperature near zero because most tokens are easy) or erratically high-temperature (overfitting to impossible tokens). The success of AutoDeco depends on these carefully designed data interventions, not just on the differentiable sampling mechanism.

---

#### Training Hyperparameters and Configuration

The paper provides specific training configuration details in Section 6 (Appendix). The setup is designed for efficiency and accessibility, with training completing in approximately 400 steps.

**Optimization:**
- **Optimizer:** AdamW (the standard adaptive optimizer with decoupled weight decay).
- **Learning rate:** `$5 \times 10^{-6}$`, a relatively low learning rate appropriate for fine-tuning small additional heads on top of a frozen pre-trained model.
- **Batch size:** 1 per device with 4 gradient accumulation steps, yielding an effective global batch size of 32. The small per-device batch size is necessary because the base model (up to 30B parameters for Qwen3-30B-A3B) consumes substantial GPU memory even in inference mode.
- **Max token length:** 16,384 tokens. This is the maximum sequence length the model processes; longer sequences would be truncated.

**Infrastructure:**
- **Hardware:** 8 GPUs (type unspecified, but the batch size and model sizes suggest high-memory GPUs such as A100-80GB or H100).
- **DeepSpeed strategy:** ZeRO Stage 3 for the dense models (DeepSeek-R1-Distill-Qwen-7B and Llama-Nemotron-8B); ZeRO Stage 2 for the mixture-of-experts models (Qwen3-30B-A3B and GPT-OSS-20B). ZeRO Stage 3 shards optimizer states, gradients, and parameters across GPUs, enabling training of larger models. Stage 2 shards only optimizer states and gradients.

**Architecture:**
- **Temperature head:** A 2-layer MLP (exact hidden dimension not specified, but described as "lightweight" and adding negligible parameters relative to the base model). Takes the final hidden state `$h_t \in \mathbb{R}^{d_{\text{model}}}$` as input, outputs a scalar `$\hat{T}_t$`.
- **Top-p head:** A 2-layer MLP. Takes both the hidden state `$h_t$` and the predicted `$\hat{T}_t$` as input (the paper emphasizes this micro-dependency with a dashed arrow in Figure 1), outputs a scalar `$\hat{P}_t \in [0, 1]$`. Feeding `$\hat{T}_t$` as input allows the top-p head to condition on the temperature, since the optimal nucleus size depends on how sharply peaked the temperature-scaled distribution is.
- **Soft top-p steepness:** `$\alpha = 30$`, controlling the sharpness of the differentiable soft mask transition. This value was determined empirically to provide a good balance between approximating the hard cutoff and maintaining informative gradients.

**Training data:**
- **Dataset:** DeepMath-103K (He et al., 2025), a large-scale dataset of challenging mathematical problems with verifiable answers.
- **Data generation:** For each base model, solutions are sampled and filtered via reject sampling to retain only correct trajectories.
- **Data quantity:** The paper reports that strong performance is achieved with approximately 6,000 training samples and 400 training steps (Figure 6 in Appendix), indicating remarkable data efficiency. This is a key practical advantage: expensive reject sampling need only be done on a modest dataset, not on millions of examples.

**Training objective:** The standard next-token prediction cross-entropy loss on the final `$\tilde{p}$` distribution, with Easy-Token Masking (60% of easy tokens masked) and Dynamic Fine-Tuning re-weighting applied.

**Training curves (Figure 6 in Appendix)** show that the loss converges effectively across all four model families, with the major decrease occurring in the first 100–200 steps and stabilizing thereafter. The fast convergence is expected because only the small MLP heads are being trained, not the full transformer.

---

#### Inference: Dynamic Decoding with Negligible Overhead

At inference time, the AutoDeco system operates as a drop-in replacement for standard generation with no changes to the user's generation logic. The process for each token generation step is:

**Step 1: Compute hidden state.** The base LLM processes the input sequence (prompt plus previously generated tokens) and produces the final hidden state `$h_t$` at the current position `$t$`. This is identical to standard transformer inference.

**Step 2: Predict decoding parameters.** In parallel:
- The standard language modeling head computes logits `$l_t \in \mathbb{R}^{|V|}$` from `$h_t$`.
- The temperature head computes `$\hat{T}_t = \text{temp\_head}(h_t)$`.
- The top-p head computes `$\hat{P}_t = \text{top-p\_head}(h_t, \hat{T}_t)$`.

The temperature head and LM head operate strictly in parallel (both taking only `$h_t$` as input). The top-p head has a micro-dependency on `$\hat{T}_t$`, meaning it must wait for the temperature head's output before computing its own. However, since both heads are tiny MLPs (a few thousand parameters each, versus millions or billions in the transformer), this sequential dependency adds negligible wall-clock time.

**Step 3: Internal probability modification.** Unlike standard decoding where temperature and top-p are applied externally by the user's generation code, AutoDeco applies them internally within the model:

1. The logits are divided by `$\hat{T}_t$`: `$l'_t = l_t / \hat{T}_t$`.
2. Softmax is applied: `$p_t = \text{softmax}(l'_t)$`.
3. Standard hard top-p sampling is performed using `$\hat{P}_t$`: tokens are sorted by probability, the smallest set with cumulative probability exceeding `$\hat{P}_t$` is retained, all other token probabilities are set to zero, and the distribution is renormalized.
4. A token is sampled from the resulting distribution (or the argmax is taken if greedy decoding is desired).

The critical point is that steps 1–3 happen inside the model's forward pass. The user's generation code receives a standard probability distribution and samples from it normally—it does not need to know about temperature or top-p at all. As the paper emphasizes, AutoDeco requires "a '1-line-change' in a user's code" because the dynamic behavior is encapsulated within the model.

**Latency analysis (Table 3).** The paper provides detailed latency measurements for R1-Distill-Qwen-7B across various prompt lengths (1k to 24k input tokens). The AutoDeco heads add:
- **FLOPs:** essentially zero overhead. For a 1k-token prompt, both Default Sampling and AutoDeco require `$2.89 \times 10^{13}$` FLOPs; at 24k tokens, AutoDeco uses `$36.20 \times 10^{13}$` vs. `$36.19 \times 10^{13}$` for Default Sampling—a difference in the fourth significant digit.
- **Memory:** an increase of 4 MB (e.g., 15,546 MB → 15,550 MB for a 1k prompt). This is the memory needed to store the MLP parameters and their intermediate activations, which is negligible compared to the base model's memory footprint.
- **Wall-clock latency:** an increase of 0.29–0.6 seconds per 1,000 generated tokens, representing an average relative increase of 1.7%. For a 1k-token prompt generating 1k output tokens, Default Sampling takes 18.23 seconds while AutoDeco takes 18.84 seconds.

These measurements confirm that the overhead is negligible and scales proportionally with the base model's inference cost, meaning the relative overhead remains constant regardless of sequence length.

**Why the overhead is so small.** The temperature and top-p heads are 2-layer MLPs with a hidden dimension that is tiny compared to the transformer's hidden dimension (which is typically 4096 for 7B models and larger for 30B+ models). The FLOPs consumed by these heads are dominated by matrix multiplications of size `$d_{\text{model}} \times d_{\text{head}}$` and `$d_{\text{head}} \times 1$`, where `$d_{\text{head}}$` is the MLP hidden dimension. Even if `$d_{\text{head}} = 256$`, this is approximately `$256 \times d_{\text{model}}$` FLOPs, whereas each transformer layer consumes approximately `$12 \times d_{\text{model}}^2$` FLOPs (for the attention and feed-forward sublayers). With `$d_{\text{model}} = 4096$` and 32 layers, the base model consumes roughly `$32 \times 12 \times 4096^2 \approx 6.4 \times 10^{12}$` FLOPs per token, while the AutoDeco heads consume roughly `$2 \times 256 \times 4096 \approx 2 \times 10^6$` FLOPs—six orders of magnitude less. The overhead is literally a rounding error.

**Design choice: hard top-p at inference despite soft top-p at training.** The paper trained the top-p head using the soft differentiable approximation because hard top-p blocked gradient flow. At inference, there is no training signal, so the non-differentiability of hard top-p is irrelevant. Using hard top-p at inference provides cleaner truncation—the tail tokens are truly eliminated rather than merely attenuated, which produces more focused distributions. The assumption is that the mapping from hidden state to predicted `$\hat{P}$` learned during training (where `$\hat{P}$` influenced the loss through the soft approximation) generalizes to the hard cutoff at inference. The strong empirical results (Tables 1 and 2) validate this assumption: the learned `$\hat{P}$` values are meaningful even when applied with the exact, non-smooth truncation algorithm.

---

#### What AutoDeco Does NOT Change

It is important to clarify what components AutoDeco leaves untouched to understand its role in the generation pipeline:

- **The base language model parameters are frozen.** AutoDeco does not modify the model's internal representations, knowledge, or reasoning capabilities. It only changes how the model's existing logits are converted into a sampling distribution.

- **The logits themselves are unchanged.** The standard LM head still produces the same raw logits `$l_t$`. AutoDeco only modifies how those logits are processed after they are produced—temperature scaling and top-p truncation are applied to the logits, but the logits themselves are not altered.

- **The token sampling process is downstream of AutoDeco.** After AutoDeco produces the modulated probability distribution, any sampling strategy can be used: multinomial sampling, greedy selection, beam search, etc. AutoDeco is a preprocessing step on the logits/probabilities, not a replacement for the sampling algorithm.

- **AutoDeco does not change the training objective of the base model.** The heads are trained with the standard next-token prediction loss applied to the dynamically-modulated distribution `$\tilde{p}$`. There is no reinforcement learning, no adversarial training, and no auxiliary task. The base model was pre-trained with standard cross-entropy loss; the heads are fine-tuned with standard cross-entropy loss on a different (dynamically created) distribution.

This modularity is a key strength: AutoDeco can be added to any existing pre-trained model without modifying any existing component, and it does not constrain the choice of downstream generation algorithm. It is a self-contained module that transforms static, externally-configured decoding into dynamic, internally-controlled decoding, and can be integrated into any transformer-based LLM with minimal engineering effort.

## 4. Key Insights and Innovations

### Innovation 1: The Decoding Problem Is Not About Finding Better Static Parameters—It's About Eliminating the Concept of Static Parameters Entirely

The dominant assumption in the decoding literature—implicit in every paper that proposes top-k, top-p, beam search, contrastive decoding, or speculative decoding—is that the decoding strategy is an **external configuration** to be optimized. The field's energy has been directed at two questions: (1) what is the best decoding algorithm (greedy? beam? nucleus? contrastive?), and (2) what are the best hyperparameters for that algorithm on a given task? The search space is well-defined: pick an algorithm, then sweep temperature, top-p, top-k, beam width, amateur model choice, etc. until validation performance peaks.

AutoDeco's first fundamental move is to **reject the premise of that search space**. The paper argues, implicitly but forcefully, that the very concept of a "decoding hyperparameter" is the problem. Temperature and top-p are not knobs to be tuned offline; they are **control signals that should be generated by the model itself at each token**. This reframes decoding from a configuration problem (find the best external setting) to an **architectural problem** (give the model the ability to set its own settings). The move is analogous to what convolutional neural networks did to hand-crafted features—not "find better SIFT parameters" but "eliminate SIFT and let the model learn features from data."

What makes this more than a rhetorical reframing is the technical mechanism that makes it possible (the differentiable soft top-p, covered in Section 3), but the conceptual contribution is separable from the mechanism. The paper demonstrates that when a model is given the *capacity* to control its own decoding, it learns to do so in ways that no static configuration could replicate—responding to token-level context, internal uncertainty, and even natural language instructions. This is not a better knob; it's the abolition of knobs.

**Comparison to prior work:** Every decoding method in the paper's related work section (Section 4) operates within the external-configuration paradigm. Deterministic methods fix the algorithm (greedy, beam search) and have no stochasticity knobs at all. Stochastic methods (Fan et al., 2018; Holtzman et al., 2019) fix the algorithm and let the user choose knobs, then apply them uniformly to every token. Model-based methods (Li et al., 2023; Chuang et al., 2023; Leviathan et al., 2023) shift the configuration burden from numerical parameters to architectural choices (which amateur model, what weighting scheme), but the fundamental external-configuration paradigm remains. The paper's claim is that AutoDeco is the first method to **internalize decoding control into the model's learned parameters**, making the distinction between "model" and "decoding strategy" disappear.

**Evidence:** The most compelling evidence for this reframing is not the performance gains (which could be achieved by better static tuning in principle) but the **emergent instruction-based control** in Section 3.3. The fact that the model spontaneously translates "generate with low randomness" into concrete adjustments to its internal temperature and top-p predictions—without being explicitly trained on such commands—demonstrates that AutoDeco creates a genuine **interface between language understanding and generation control** that does not exist in the external-configuration paradigm. This capability is definitionally impossible for any static decoding method, no matter how well-tuned.

**Significance:** This is a **fundamental shift** rather than an incremental improvement. It does not make existing decoding research obsolete (deterministic methods, contrastive decoding, and speculative decoding address orthogonal problems), but it changes the question from "how should we configure decoding?" to "what should the model be allowed to control about its own generation process?" The answer suggested by this paper—temperature and top-p—is likely just the first step. The architectural pattern (predict control signals from hidden states, make the controlled operation differentiable, train end-to-end) could extend to other decoding parameters, other generation modalities, and other forms of model self-regulation.

---

### Innovation 2: The Non-Differentiability of Sampling Operations Is Not a Dead End—It's a Surmountable Engineering Obstacle

A trained machine learning practitioner, asked whether you could train a neural network to predict its own top-p threshold by backpropagating through top-p sampling, would answer "no"—correctly pointing out that the hard cutoff in nucleus sampling is a non-differentiable operation that blocks gradient flow. This is not a subtle theoretical limitation; it's a first-day-of-neural-nets kind of obstacle. The implicit conclusion most researchers draw is that decoding parameter prediction must be approached through reinforcement learning, bilevel optimization, or some other gradient-free method.

AutoDeco's second key insight is that **this obstacle is an engineering problem with a differentiable approximation solution, not a fundamental barrier**. The paper's solution—replacing the hard cutoff with a smooth exponential decay mask during training, then reverting to hard top-p at inference—is elegant in its minimality. It does not require a new training paradigm, a new loss function, or a new optimization algorithm. It requires only a 10-line modification to the sampling code that makes an existing operation differentiable.

**Why this is intellectually significant beyond AutoDeco:** The differentiable soft top-p is not just a trick for this specific problem. It demonstrates a general pattern: non-differentiable sampling operations that are traditionally considered "outside the model" can be brought inside the differentiable computation graph by replacing them with smooth approximations during training. This pattern could apply to:

- **Top-k sampling:** replace the hard retention of exactly k tokens with a soft weighting that smoothly decays as rank decreases, enabling a trainable k-prediction head.
- **Beam search branching decisions:** replace the hard selection of top-b beams with a soft attention over candidates, enabling the model to learn when to expand the beam.
- **Speculative decoding acceptance criteria:** replace the hard accept/reject decision with a soft probability of acceptance, enabling the draft model to learn to predict tokens the target model will accept.
- **Any discrete sampling operation** where the "right" parameter (threshold, beam width, acceptance rate) varies by context and could be predicted from hidden states.

The paper does not explore these extensions, but the conceptual move is clear: the boundary between "differentiable model internals" and "non-differentiable post-processing" is not fixed. It can be shifted by replacing hard operations with soft approximations during training, creating gradient pathways to components that were previously unreachable by backpropagation.

**Comparison to prior work:** Prior approaches to learned decoding have generally avoided the non-differentiability problem rather than solving it. Reinforcement learning methods (e.g., training a policy to select decoding parameters and rewarding it based on downstream task performance) do not require differentiable sampling operations but introduce high-variance gradient estimates, credit assignment over long sequences, and sample inefficiency. Bilevel optimization (treat decoding parameters as hyperparameters and optimize them via grid search or Bayesian optimization) is computationally prohibitive at the token level. The differentiable soft top-p approach is distinct in that it **preserves the simplicity and efficiency of standard supervised learning** (single forward pass, single backward pass, standard cross-entropy loss) while enabling gradient-based optimization of components previously considered out of reach.

**Evidence:** The ablation study (Figure 4) provides indirect but strong evidence for this insight. The top-p head alone (without the temperature head) achieves a ~3-3.5 absolute point improvement over Default Sampling on AIME. This means the differentiable approximation—despite being a "soft" version of the true hard top-p used at inference—produces meaningful learning signals that transfer to the hard operation at test time. If the approximation introduced a substantial gap between training and inference behavior, the top-p-only head would not work. Its success validates that the soft approximation is "close enough" to the hard operation for the learned mapping to transfer.

**Significance:** This is an **incremental technical contribution** with **fundamental implications**. The specific soft top-p formulation is a clever piece of engineering, but the deeper significance is the demonstration that gradient-based optimization can reach into components of the generation pipeline previously walled off by non-differentiability. The paper opens a design space: which other "external" operations can be internalized by finding smooth approximations?

---

### Innovation 3: The Model Learns a Domain-Independent "Meta-Skill of How to Generate"—Not Task-Specific Decoding Heuristics

A natural expectation when training AutoDeco heads exclusively on mathematical reasoning data would be that the heads learn math-specific decoding patterns: low temperature for arithmetic steps, moderate temperature for theorem selection, etc. The startling result in Table 2 is that these heads **generalize zero-shot to qualitatively different domains**—QA (GPQA-Diamond, MMLU-Pro), code generation (LiveCodeBench), and instruction following (IFEval)—with improvements of comparable magnitude to those seen on in-domain math tasks.

This is not merely "good transfer learning." It is evidence that the model has learned something more fundamental: **a domain-independent skill for modulating its own stochasticity based on internal uncertainty signals**, not task-specific heuristics. The paper calls this a "meta-skill of how to generate text effectively," and the framing is precise. The heads are not learning "math problems need temperature 0.7"—they're learning "when the hidden state exhibits pattern X, predict temperature Y," where pattern X encodes something like "the model is uncertain among several plausible continuations" or "the next token is a function word where determinism is safe."

**Why this is surprising:** The training data consists entirely of correct mathematical reasoning chains. These chains have specific structural properties—step-by-step deduction, numeric computation, symbolic manipulation—that are absent from general QA or code. If the heads were learning math-specific patterns, they would fail on out-of-domain tasks (predicting inappropriate temperatures for natural language explanation or code syntax). The fact that they succeed suggests that the **features in the hidden state that predict good decoding parameters are largely domain-invariant**. Uncertainty looks like uncertainty whether you're doing math or answering a biology question; syntactic predictability looks similar across domains; the need for creative exploration versus precise execution is a universal property of language generation, not a quirk of mathematical reasoning.

**Comparison to prior work:** The dominant paradigm for adapting LLMs to specific tasks involves either fine-tuning on task-specific data (which changes the model's knowledge and behavior) or tuning task-specific hyperparameters (which changes how the model's outputs are sampled). Both approaches assume a tight coupling between the training domain and the deployment domain. AutoDeco breaks this coupling: train the decoding heads on one domain (math), deploy on any domain. This is reminiscent of how pre-trained language models themselves generalize across tasks after being trained on a general corpus, but applied at the meta-level of *how to decode* rather than *what to know*.

The paper's finding also challenges the conventional wisdom that adaptive decoding requires broad, task-matched supervision. The literature on controlled generation (Dathathri et al., 2019; plug-and-play LMs) typically assumes that the control signal must be trained on data from the target domain or using attribute classifiers specific to the desired control dimension. AutoDeco suggests that for certain fundamental control dimensions—stochasticity, exploration vs. exploitation—the training signal from a single high-quality domain (where correctness is well-defined and verifiable) is sufficient to learn transferable control policies.

**Evidence:** The quantitative evidence is in Tables 1 and 2. On R1-Distill-Qwen-7B, AutoDeco improves the average out-of-domain score by 4.4 points (39.32 → 46.88 average across GPQA-Diamond, MMLU-Pro, LiveCodeBench, IFEval), compared to a 2.6-point improvement on in-domain math tasks (34.76 → 37.37). The out-of-domain gain is actually *larger*, which is the opposite of what you'd expect if the heads were learning math-specific heuristics. On Llama-Nemotron-8B, the pattern is similar: 3.4-point gain out-of-domain vs. 3.5-point gain in-domain—essentially identical, confirming domain-agnostic transfer.

**Significance:** This is a **fundamental empirical finding** with significant practical implications. It means that AutoDeco heads can be trained once on a high-signal domain (where correctness is easy to verify, enabling the reject sampling that produces clean training data) and then deployed across arbitrary tasks without per-task tuning. For practitioners, this eliminates the chicken-and-egg problem of needing good decoding parameters to generate training data for learning decoding parameters. For researchers, it suggests that the model's internal representations encode a rich signal about appropriate generation stochasticity that is largely task-independent—opening questions about what other "meta-skills" might be extractable from hidden states and transferable across domains.

---

### Innovation 4: Decoding Parameters Are a Natural Language Interface—Not Just Numerical Knobs

The paper's most striking finding—and the one with the broadest implications for how we think about LLM controllability—is the emergent ability of AutoDeco to respond to natural language commands about generation style. When prompted with "I hope the answers can be more innovative and diverse," the model spontaneously raises its predicted temperature and top-p values. When prompted with "I hope the answers can be as certain as possible," it lowers them. After targeted training with a ranking loss, this behavior becomes robust (95%+ consistency in the correct directional adjustment, as shown in Table 4).

This is not a feature that was designed or trained for in the original AutoDeco formulation. The heads are trained purely on the cross-entropy loss over correct mathematical reasoning chains—there is no diversity command in the training data, no explicit objective to make temperature respond to instructions. The behavior **emerges** from the architecture: because the temperature and top-p heads take the model's hidden state as input, and the hidden state encodes the model's understanding of the prompt (including any meta-instructions about desired output style), the heads learn to associate certain linguistic patterns in the prompt with certain decoding parameter values. If the training data happens to contain examples where "diverse" prompts correlate with higher-entropy outputs (which is plausible, since creative prompts in the math dataset might have multiple valid reasoning paths), the heads will pick up this correlation without explicit supervision.

**Why this is a conceptual breakthrough, not just a cool demo:** The finding reveals that **decoding parameters sit at the intersection of language understanding and generation control**. In a standard LLM, there is a hard wall between these two functions: the model understands the prompt and produces logits, but how those logits are converted into text is controlled by an external, non-linguistic mechanism (the temperature slider in the API call). AutoDeco tears down this wall. Because the decoding parameters are now part of the model's own forward pass, they become subject to the same linguistic influences as any other model output. The model's understanding of a user's intent—expressed in natural language—can directly modulate its generation behavior.

This has profound implications for the future of LLM interfaces. Current approaches to controlling generation style rely on either (a) external parameter settings (temperature, top-p) that users must understand and configure numerically, or (b) prompt engineering tricks ("be creative," "think step by step") that influence the model's output through its language understanding alone, without changing the underlying sampling distribution. AutoDeco suggests a third path: **natural language as a direct control interface for the model's internal generation parameters**, where the model translates linguistic intent into concrete stochasticity adjustments on a token-by-token basis. "Be creative here but precise here" could become a literal instruction that the model executes internally, rather than a vague prompt prefix.

**Comparison to prior work:** The concept of controlling generation through natural language exists in the prompt engineering literature (e.g., system prompts that specify "you are a creative assistant" vs. "you are a precise assistant"), but this approach influences the model's *what to generate* (the content and style of the output text) without changing *how to generate* (the sampling distribution over tokens). The model might produce more creative *content* when told to be creative, but it's still sampling from the same temperature-scaled distribution. AutoDeco is the first demonstration of natural language directly controlling the sampling parameters themselves—a fundamentally different mechanism that could enable more fine-grained and reliable style control than prompt engineering alone.

The paper's targeted training with a ranking loss (Section 3.3) is also notable as a methodological contribution. The observation that the emergent behavior is initially inconsistent and can be made robust through lightweight training with a ranking objective suggests a general recipe for "solidifying" emergent capabilities: observe the nascent behavior, design a loss function that reinforces the desired directional relationship, and fine-tune for a small number of steps. This is distinct from training the capability from scratch (which would require explicit labels or rewards) and from relying on unreliable emergence in deployment.

**Evidence:** Table 4 provides the quantitative evidence. After targeted training, the "low diversity" command drops average temperature from 0.72 to 0.61 (a 0.11 decrease) with 99% consistency over 100 test questions, and top-p from 0.79 to 0.73 (a 0.06 decrease) with 97% consistency. The "high diversity" command raises average temperature from 0.72 to 0.82 (a 0.10 increase) with 96% consistency. Figure 5 provides qualitative visualization of the token-level adjustments, showing that the predicted `$\hat{T}$` and `$\hat{P}$` values (dotted lines) consistently sit above or below the baseline (solid lines) depending on the command.

The paper is candid about limitations: the control is directional but not absolute (the model cannot achieve near-zero temperature when asked for "no randomness"), and the released models do not include this feature because the investigation is preliminary. This honesty strengthens rather than weakens the contribution—it acknowledges that the finding opens a research program rather than providing a finished product.

**Significance:** This is a **fundamental conceptual advance** that changes what we think LLMs can do. It demonstrates that the boundary between "model capabilities" (what the model knows and can express) and "decoding configuration" (how the model's knowledge is converted into text) is artificial and can be dissolved. The model can learn to control its own generation parameters in response to linguistic input, creating a unified interface where user intent about both *content* and *style* is expressed in natural language and executed through the same forward pass. This points toward a future where LLMs are not just language generation systems with externally configured sampling, but truly end-to-end systems that understand and act on instructions about *how* to generate, not just *what* to generate.

## 5. Experimental Analysis

### Evaluation Methodology

- **Dataset.** The AutoDeco heads are trained on a specialized dataset of reject sampling trajectories generated from the DeepMath-103K dataset (He et al., 2025), a large-scale collection of challenging mathematical problems with verifiable answers. Evaluation is conducted across eight diverse benchmarks split into two categories: in-domain math reasoning (AIME 2024+2025, BRUMO25, HMMT25 from Balunović et al., 2025, and BeyondAIME from ByteDance-Seed, 2025) and out-of-domain general tasks (GPQA-Diamond from Rein et al., 2024, MMLU-Pro from Wang et al., 2024, LiveCodeBenchV6 from Jain et al., 2024, and IFEval from Zhou et al., 2023). For MMLU-Pro, a "lite" subset is used for balanced assessment across subject areas; LiveCodeBench uses the V6 version with an evaluation window starting September 1, 2023; all other benchmarks use their full sets.

- **Base model(s).** The paper integrates AutoDeco into four representative open-source model families: Llama-3.1-Nemotron-Nano-8B-v1 (Nvidia, a general-purpose 8B dense model), R1-Distill-Qwen-7B (DeepSeek, a 7B distilled reasoning model known for strong math capabilities), Qwen3-30B-A3B-Instruct-2507 (a 30B-parameter mixture-of-experts instruct model with 3B active parameters), and OpenAI-GPT-OSS-20B (a 20B MoE model released by OpenAI, with reasoning effort set to medium by default). These models span dense and MoE architectures, general-purpose and reasoning-specialized training, and scales from 7B to 30B total parameters—providing broad coverage of the contemporary open-source LLM landscape.

- **Metrics.** The primary metric is Pass@1 accuracy, estimating the probability that a single generated answer is correct. Pass@1 is computed via oversampling with 128 samples per problem (8 random seeds, 16 samples per seed), with the reported value representing the mean across seeds and the standard deviation across seeds provided as error bounds (e.g., `55.43 ± 1.22`). For mathematical reasoning benchmarks, correctness is determined by exact answer matching; for general-domain tasks, evaluation follows each benchmark's standard protocol (exact match for QA, pass@1 for code generation based on test case execution, instruction-level accuracy for IFEval). Additional Pass@k metrics (k = 16, 32, 64) are reported in Appendix Tables 5–7 using the standard unbiased estimator.

- **Baselines.** The paper evaluates against two standard, non-expert decoding strategies: **Greedy Search** (selecting the highest-probability token at each step, equivalent to temperature → 0) and **Default Sampling** (temperature `$\hat{T} = 1.0$`, top-p `$\hat{P} = 1.0$`, meaning no temperature scaling and no nucleus truncation—effectively sampling from the raw softmax distribution). Additionally, the paper compares against an **Expert-Guided Tuning** baseline, which performs a fine-grained grid search over temperature (swept first with top-p fixed at 1.0) and then top-p (swept with the optimal temperature fixed) on the test set itself—an oracle setting infeasible in practice since it requires access to ground-truth answers on the evaluation data. No other learned decoding methods (contrastive decoding, speculative decoding) are used as baselines, as the paper positions AutoDeco as orthogonal to these approaches rather than competing with them.

- **Generation budget / compute accounting.** The paper does not constrain or compare methods by a fixed generation budget in its main results—each method (Greedy, Default Sampling, AutoDeco, Expert-Guided) is evaluated by generating a fixed number of samples per problem (128 total across 8 seeds) and computing Pass@1 as if a single sample were drawn. The efficiency comparison (Section 3.2.2) measures FLOPs, memory, and wall-clock latency for generating 1,000 tokens across varying prompt lengths, comparing Default Sampling and AutoDeco with identical generation parameters except for the dynamic vs. static temperature/top-p. The Expert-Guided Tuning incurs the cost of sweeping 10 temperature values and 10 top-p values (100 total configurations) on the test set, but this cost is not quantified since it is presented as an oracle upper bound rather than a practical method.

- **Cross-validation / statistical protocol.** No cross-validation is used. The AutoDeco heads are trained once on the DeepMath-103K reject sampling trajectories (a fixed training set separate from all evaluation benchmarks) and then evaluated directly on the eight test benchmarks. Statistical reliability is assessed through multiple random seeds: 8 independent runs with different seeds, each generating 16 samples per problem, yielding 128 total samples per problem. The mean and standard deviation across the 8 seeds are reported for each Pass@1 estimate. For the emergent instruction-based control experiments (Section 3.3), consistency is measured across 100 test prompts, with the fraction of prompts showing the correct directional adjustment reported as "Consistency." The Expert-Guided Tuning sweeps hyperparameters at a granularity of 0.1 for both temperature and top-p, with the optimal setting selected based on the highest test-set accuracy—introducing an unknown degree of overfitting to the specific 128 samples generated per problem.

### Main Quantitative Results

#### In-Domain Mathematical Reasoning (Table 1)

AutoDeco demonstrates consistent and substantial improvements over both Greedy Search and Default Sampling across all four model families and four mathematical reasoning benchmarks. The headline numbers from Table 1:

**Llama-Nemotron-8B:** AutoDeco achieves an average score of 46.05 across AIME, BRUMO25, HMMT25, and BeyondAIME, compared to 42.50 for Greedy Search and 42.59 for Default Sampling—an improvement of approximately 3.5 absolute points. The gains are distributed across all benchmarks: AIME improves from 51.67 (Greedy) / 50.84 (Default) to 55.43 (±1.22); BRUMO25 from 56.67 / 57.89 to 60.60 (±0.98); HMMT25 shows the largest relative gain, from 26.67 / 29.82 to 33.98 (±1.14); and BeyondAIME from 35.00 / 31.82 to 34.19 (±0.65).

**R1-Distill-Qwen-7B:** AutoDeco achieves 37.37 average vs. 30.58 (Greedy) and 34.76 (Default)—a gain of 6.8 and 2.6 points respectively. The largest absolute gain is on AIME: 47.03 (±1.12) vs. 38.33 (Greedy) and 43.49 (Default), representing an 8.7-point improvement over greedy and 3.5 points over default sampling.

**Qwen3-30B-A3B-Instruct-2507:** AutoDeco achieves 56.54 average vs. 52.25 (Greedy) and 56.05 (Default)—a more modest gain of 4.3 points over greedy but only 0.5 points over default sampling. The paper attributes this smaller margin to Qwen3 generating significantly shorter answers, reducing sensitivity to sampling parameter variations. Even here, AIME shows a notable improvement: 68.46 (±0.76) vs. 67.92 (±1.36) under Default Sampling.

**OpenAI-GPT-OSS-20B:** AutoDeco achieves 58.13 average vs. 51.50 (Greedy) and 56.64 (Default)—gains of 6.6 and 1.5 points respectively. AIME shows the strongest absolute improvement: 72.33 (±1.20) vs. 69.61 (±1.32) under Default Sampling.

Two patterns are notable. First, AutoDeco never underperforms either baseline on any benchmark-model pair—the improvement is monotonic across all 16 conditions (4 models × 4 benchmarks). Second, the standard deviations under AutoDeco are consistently smaller than or equal to those under Default Sampling (e.g., on Llama-Nemotron-8B for AIME: ±1.22 vs. ±1.44; on R1-Distill-Qwen-7B for HMMT25: ±0.39 vs. ±0.83), suggesting that dynamic decoding not only improves average performance but also reduces variance across random seeds—a sign that the model is making more consistent, well-calibrated decisions about when to be stochastic.

#### Out-of-Domain Generalization (Table 2)

The most striking finding in Table 2 is not just that AutoDeco generalizes to out-of-domain tasks (despite being trained exclusively on mathematical reasoning), but that the magnitude of improvement often matches or exceeds in-domain gains.

**R1-Distill-Qwen-7B:** On the four general-domain benchmarks, AutoDeco improves the average score from 39.32 (Greedy) / 42.47 (Default) to 46.88—a gain of 7.6 and 4.4 points respectively, which is larger than the in-domain gain over Default Sampling (2.6 points on math). The breakdown: GPQA-Diamond improves from 37.87 (Greedy) / 47.41 (Default) to 48.91; MMLU-Pro from 47.20 / 47.65 to 50.75; LiveCodeBenchV6 from 49.13 / 53.00 to 53.14; IFEval from 32.90 / 32.35 to 33.90. The IFEval gain is notably small (1.0–1.6 points), perhaps because instruction-following accuracy depends more on content correctness than on sampling stochasticity.

**Llama-Nemotron-8B:** AutoDeco achieves 49.72 average vs. 48.43 (Greedy) and 46.35 (Default). Notably, Default Sampling underperforms Greedy Search on two benchmarks—GPQA-Diamond (44.93 vs. 51.01) and IFEval (65.25 vs. 71.53)—indicating that the static temperature of 1.0 introduces harmful stochasticity on these tasks. AutoDeco recovers most of the greedy performance on GPQA-Diamond (50.52) and IFEval (71.02), demonstrating that it learns to predict lower temperatures on tasks where determinism is beneficial. On MMLU-Pro, AutoDeco (55.64) outperforms both Greedy (52.00) and Default (54.00), suggesting nuanced temperature predictions that improve over even the best static baseline.

**Qwen3-30B-A3B-Instruct-2507:** AutoDeco achieves 70.24 average vs. 68.84 (Greedy) and 69.03 (Default)—a 1.4-point gain over greedy but only 1.2 points over default sampling, consistent with the smaller in-domain gains for this model.

**OpenAI-GPT-OSS-20B:** AutoDeco achieves 59.42 average vs. 56.56 (Greedy) and 58.63 (Default)—gains of 2.9 and 0.8 points respectively. On LiveCodeBenchV6, AutoDeco (71.25) sets the highest score among all methods and models in the table.

A critical pattern emerges when comparing AutoDeco's behavior across tasks where Default Sampling beats Greedy Search versus the reverse. On GPQA-Diamond with Llama-Nemotron-8B, Default Sampling (44.93) is substantially worse than Greedy Search (51.01). AutoDeco (50.52) nearly closes this gap, predicting more deterministic parameters when stochasticity harms performance. Conversely, on R1-Distill-Qwen-7B for GPQA-Diamond, Default Sampling (47.41) substantially outperforms Greedy Search (37.87), and AutoDeco (48.91) further improves on Default Sampling by raising temperature adaptively. This demonstrates that AutoDeco is not simply learning a fixed preference for higher or lower stochasticity—it is learning to match its stochasticity level to the demands of the specific task, model, and context.

#### Pass@k Performance (Appendix Tables 5–7)

The paper addresses a concern raised by recent work (Yue et al., 2025; Chen et al., 2025) that optimizing for Pass@1 can harm Pass@k performance (for k > 1), potentially creating a brittleness where the model produces a single good answer but lacks diversity for best-of-N selection. The results in Tables 5, 6, and 7 show that this concern does not materialize for AutoDeco.

At Pass@16 (Table 5), AutoDeco improves over Default Sampling on all four model families: Llama-Nemotron-8B average improves from 81.21 to 83.21 (+2.0 points); R1-Distill-Qwen-7B from 62.24 to 66.23 (+4.0 points); Qwen3-30B-A3B from 77.30 to 77.78 (+0.5 points); OpenAI-GPT-OSS-20B from 83.91 to 85.45 (+1.5 points). The absolute gains are comparable to or slightly larger than the Pass@1 gains.

At Pass@32 (Table 6): Llama-Nemotron-8B improves from 84.47 to 86.44 (+2.0 points); R1-Distill-Qwen-7B from 66.86 to 70.79 (+3.9 points); Qwen3-30B-A3B from 80.27 to 80.20 (−0.07 points, essentially flat); OpenAI-GPT-OSS-20B from 87.08 to 89.23 (+2.2 points).

At Pass@64 (Table 7): Llama-Nemotron-8B improves from 87.35 to 89.75 (+2.4 points); R1-Distill-Qwen-7B from 70.92 to 74.78 (+3.9 points); Qwen3-30B-A3B from 82.94 to 83.61 (+0.7 points); OpenAI-GPT-OSS-20B from 89.68 to 91.55 (+1.9 points).

The paper highlights that while absolute gains remain consistent or slightly grow with k, the **relative error reduction** becomes dramatically larger at high k because the baseline error rate shrinks. For OpenAI-GPT-OSS-20B, the error rate at Pass@1 under Default Sampling is 100% − 56.64% = 43.36%, and AutoDeco's improvement of 1.49 points represents a 3.5% relative error reduction. At Pass@64, the error rate is 100% − 89.68% = 10.32%, and AutoDeco's improvement of 1.87 points represents an **18.1% relative error reduction**—a more than 5× larger relative impact. The paper argues this demonstrates that "as the task becomes easier for the baseline model (i.e., the error rate decreases at high k), the performance gains from our method become even more significant."

This finding has an important practical implication: AutoDeco is not merely a Pass@1 optimizer that sacrifices diversity. It improves the quality of the entire sampling distribution, making it suitable both for single-sample generation and for best-of-N selection pipelines where multiple candidates are generated and the best is chosen via verifier or majority voting. The consistent gains across k-values suggest that AutoDeco is learning to produce better-calibrated probability distributions overall, not just to sharpen the distribution around a single high-probability answer.

#### Comparison with Expert-Guided Oracle Tuning (Figure 3)

Figure 3 presents a side-by-side comparison of AutoDeco against Expert-Guided Tuning on two models (Llama-Nemotron-8B and R1-Distill-Qwen-7B) across four benchmarks each. The Expert-Guided Tuning performs a grid search over temperature (0.1 to 1.0 in steps of 0.1, with top-p initially fixed at 1.0 to isolate the temperature effect) and then over top-p (with the optimal temperature fixed), selecting the combination that maximizes accuracy on the test set. This represents a **practical upper bound** for any static decoding strategy—it is an oracle that "hacks the test set" by using ground-truth answers to find the optimal settings.

The results are remarkably tight. For Llama-Nemotron-8B on AIME, AutoDeco's single-pass performance is essentially identical to the oracle-tuned baseline (both near ~55%, read from the figure). On BRUMO25, AutoDeco slightly exceeds the oracle (~61% vs. ~60%). On GPQA-Diamond and HMMT25, the two methods are within approximately one point of each other. For R1-Distill-Qwen-7B, the same pattern holds: AutoDeco matches or slightly exceeds the oracle-tuned baseline across all four benchmarks, with differences consistently under one absolute point.

The figure also reveals the extreme task-dependence of optimal static hyperparameters, which is the fundamental problem AutoDeco solves. For Llama-Nemotron-8B, the oracle-tuned temperature varies from `$\hat{T} = 0.8$` (BRUMO25, with `$\hat{P} = 0.9$`) to `$\hat{T} = 0.3$` (GPQA-Diamond, with `$\hat{P} = 0.6$`). Even within the math domain, AIME requires `$\hat{T} = 0.4$` while BRUMO25 needs `$\hat{T} = 0.8$`. A single static configuration deployed across all tasks would necessarily be suboptimal for most of them. AutoDeco achieves near-oracle performance on each task without any per-task tuning, because it predicts context-appropriate parameters on the fly.

The paper argues—persuasively—that this makes AutoDeco **effectively superior to any feasible expert-tuning strategy in practice**. The Expert-Guided Tuning requires access to test-set answers, which is impossible in real-world deployments where the test data is unknown. Any practical tuning process (using a held-out validation set, cross-validation, or manual heuristics) would be strictly worse than the oracle baseline, and AutoDeco already matches the oracle. The conclusion is that AutoDeco provides "the optimal and, frankly, only practical solution for developers seeking robust, high-performance generation across diverse user inputs."

#### Efficiency Analysis (Table 3)

Table 3 provides a detailed computational cost comparison between Default Sampling and AutoDeco on R1-Distill-Qwen-7B, measuring FLOPs, memory usage, and wall-clock latency for generating 1,000 output tokens across prompt lengths from 1k to 24k input tokens.

**FLOPs:** The difference is negligible at all prompt lengths. At 1k input tokens, both methods consume `$2.89 \times 10^{13}$` FLOPs. At 24k tokens, AutoDeco consumes `$36.20 \times 10^{13}$` FLOPs vs. `$36.19 \times 10^{13}$` for Default Sampling—a difference of `$0.01 \times 10^{13} = 1 \times 10^{11}$` FLOPs, or approximately 0.03% overhead. The FLOPs are dominated by the base transformer layers; the 2-layer MLP heads contribute a vanishingly small fraction of total computation.

**Memory:** AutoDeco adds 4 MB of GPU memory usage at all prompt lengths (e.g., 15,546 MB → 15,550 MB at 1k tokens; 27,262 MB → 27,266 MB at 24k tokens). This represents the storage for the additional MLP parameters and their intermediate activations, which is < 0.03% of the base model's memory footprint.

**Latency:** Wall-clock time increases by 0.29–0.6 seconds per 1,000 generated tokens across prompt lengths, translating to a relative increase of 1.0–3.2% depending on prompt length, with an average of approximately 1.7%. At 1k input tokens, Default Sampling takes 18.23 seconds and AutoDeco takes 18.84 seconds (+0.61s, +3.3%). At 24k input tokens, Default Sampling takes 25.76 seconds and AutoDeco takes 26.05 seconds (+0.29s, +1.1%). The absolute overhead decreases with longer prompts because the time spent in the transformer layers (which grows with sequence length) increasingly dominates the fixed cost of the AutoDeco heads.

The paper does not provide training efficiency metrics in the main text, but Appendix Section 7 and Figure 6 report that training converges in approximately 400 steps and 6,000 training samples. Training is conducted on 8 GPUs with DeepSpeed ZeRO-3 (for dense models) or ZeRO-2 (for MoE models). The per-device batch size is 1 with 4 gradient accumulation steps, yielding an effective batch size of 32. The training time is not quantified in hours or GPU-hours, but given the small number of steps and frozen base model, it is likely on the order of a few hours on 8 GPUs—substantially less than the computational cost of a full fine-tuning run or even a typical hyperparameter sweep.

#### Ablation: Temperature-Only and Top-p-Only Heads (Figure 4)

Figure 4 presents an ablation study on AIME using R1-Distill-Qwen-7B, isolating the contributions of the temperature head and top-p head. The key findings:

- **Default Sampling baseline:** approximately 43–44% on AIME.
- **Temperature head alone:** approximately 46–47%. A gain of roughly 3–3.5 absolute points over Default Sampling.
- **Top-p head alone:** approximately 46–47%. A gain nearly identical to the temperature head alone.
- **Full AutoDeco (both heads):** approximately 48–49%. A further gain of approximately 1.5–2 points over either head alone.

The most striking result is that **each head is remarkably effective in isolation**. Either a learned temperature head or a learned top-p head alone—without the other—provides a substantial improvement over static default parameters. This has two implications. First, the benefits of dynamic decoding are robust: even a minimal implementation (a single learned parameter) captures significant gains. Second, the two heads provide complementary benefits. Temperature controls the overall entropy of the distribution (how peaked or flat it is), while top-p controls tail truncation (how much of the low-probability mass to eliminate). The full AutoDeco with both heads achieves the best performance because it can independently modulate these two axes, enabling finer-grained control than either alone.

The paper does not ablate the design choices within each head—for instance, whether the top-p head benefits from receiving the predicted temperature as input (the micro-dependency shown as a dashed arrow in Figure 1), or whether a single combined head predicting both parameters jointly would match the two-head architecture. It also does not ablate the steepness parameter `$\alpha = 30$` in the soft top-p mask, which controls the trade-off between approximating the hard cutoff and maintaining informative gradients. These are non-trivial missing ablations that would strengthen the architectural design claims.

#### Emergent Instruction-Based Decoding Control (Figure 5 and Table 4)

The qualitative demonstration in Figure 5 shows the token-level temperature and top-p predictions for a single creative prompt under three conditions: baseline (no meta-instruction), "I hope the answers can be more innovative and diverse," and "I hope the answers can be as certain as possible." The model was not trained on any diversity-control commands—this behavior emerged spontaneously from the standard AutoDeco training on math trajectories.

In the baseline condition (left panel), the predicted temperature and top-p values vary across tokens, typically oscillating between approximately 0.5 and 0.9 for temperature and between 0.6 and 0.9 for top-p. In the high-diversity condition (middle panel), the predicted values (shown as dotted lines) are consistently elevated above the baseline (solid lines), with temperature climbing to approximately 0.8–1.0 and top-p rising to 0.8–1.0. In the low-diversity condition (right panel), the predicted values are consistently suppressed below the baseline, with temperature dropping toward 0.4–0.6 and top-p toward 0.5–0.7.

The paper notes that this emergent capability was initially inconsistent—appearing only "occasionally across different prompts." This motivated a targeted training procedure to solidify the behavior. Training prompts were augmented with diverse "decoding-control commands" (specific examples not provided in the paper), and a ranking loss was applied that encourages the model to predict higher `$\hat{T}$` and `$\hat{P}$` for high-diversity commands relative to baseline, and lower values for low-diversity commands.

After this targeted training (Table 4), evaluated on 100 test prompts:
- **Low Diversity command:** Average temperature drops from 0.72 (baseline, no command) to 0.61 (a decrease of 0.11), with 99% consistency (99 out of 100 prompts show the correct directional adjustment). Average top-p drops from 0.79 to 0.73 (decrease of 0.06), with 97% consistency.
- **High Diversity command:** Average temperature rises from 0.72 to 0.82 (increase of 0.10), with 96% consistency. Average top-p rises from 0.79 to 0.83 (increase of 0.04), with 85% consistency.

The consistently lower reliability of top-p adjustments compared to temperature adjustments (85% vs. 96% for high diversity, 97% vs. 99% for low diversity) may reflect the more complex relationship between top-p predictions and the diversity of outputs: top-p depends on the shape of the probability distribution, not just its scale, making it harder to control precisely through a ranking loss.

The paper is explicit about limitations: the control is directional (up or down) but not absolute. When prompted to "ensure your generation has no randomness," the model produces a "modest but directionally correct drop" rather than near-zero temperature. The paper hypothesizes that achieving precise absolute control may require joint training of the base LLM and the AutoDeco heads, since the frozen base model cannot adjust its internal representations to produce the exact hidden states that would map to a specific target temperature.

The models released with the paper do **not** include this instruction-based control feature—the investigation is described as preliminary, and the authors state they will "continue to advance this line of research and release updated models as soon as we reach more definitive conclusions."

### Ablation Studies and Robustness Checks

- **Training data domain and out-of-domain generalization (Tables 1 vs. 2):** The most important ablation—though not framed as one—is the fact that AutoDeco heads trained exclusively on mathematical reasoning data generalize to QA, code generation, and instruction following. This is tested across four model families and four out-of-domain benchmarks, and the gains are consistent and often larger than in-domain gains (e.g., R1-Distill-Qwen-7B gains 4.4 points out-of-domain vs. 2.6 points in-domain over Default Sampling). This demonstrates that the heads are not learning math-specific heuristics but rather a domain-independent skill for modulating stochasticity.

- **Temperature-only vs. top-p-only vs. full AutoDeco (Figure 4):** Each head alone provides substantial gains over Default Sampling (~3–3.5 points on AIME for R1-Distill-Qwen-7B), and the combination provides an additional ~1.5–2 points. This confirms that both parameters contribute complementary control and that even a minimal single-parameter dynamic decoding system is beneficial. Not tested, however, is whether other parameter combinations (e.g., temperature + top-k, or top-p alone with a learned dynamic top-k) would perform differently, or whether the specific choice of temperature and top-p is optimal among possible decoding parameter pairs.

- **Pass@1 vs. Pass@k scaling (Tables 5–7 vs. Table 1):** The consistent absolute gains across k = 1, 16, 32, 64 demonstrate that AutoDeco does not create a tradeoff between single-sample quality and sample diversity. This is a robustness check against the concern that dynamic decoding might over-optimize the distribution for a single draw at the expense of the distribution's overall calibration. The fact that relative error reduction grows with k (3.5% at Pass@1 → 18.1% at Pass@64 for GPT-OSS-20B) suggests the opposite: AutoDeco improves the entire sampling distribution in a way that compounds when multiple samples are drawn.

- **Efficiency across prompt lengths (Table 3):** The latency overhead of AutoDeco is tested at prompt lengths from 1k to 24k input tokens. Overhead remains consistently small (1–3%) and does not grow with sequence length, confirming that the cost is a fixed per-token addition from the MLP heads rather than a multiplier on the transformer computation. Memory overhead is constant at 4 MB regardless of prompt length, consistent with the fixed parameter count of the AutoDeco heads.

- **Training efficiency and data requirements (Figure 6):** Training curves show convergence within approximately 400 steps and 6,000 training samples across all four model families. This demonstrates that AutoDeco does not require large-scale training data or extended fine-tuning, making it practical to integrate into existing pre-trained models with minimal computational investment. However, the paper does not report how performance varies with training data quantity—would 3,000 samples achieve 90% of the gain? Would 12,000 samples provide further improvement? The data efficiency claim of "6K samples and 400 steps" establishes feasibility but does not characterize the scaling behavior.

- **No explicit ablation of training techniques:** The paper does not ablate Easy-Token Masking (60% masking rate) or Dynamic Fine-Tuning (re-weighting strategy). These techniques are described as important for preventing degenerate behaviors (the temperature head collapsing to near-zero or predicting inappropriately large values), but their individual contributions are not quantified. A comparison of AutoDeco trained with and without these techniques—ideally showing that naïve end-to-end training without debiasing performs substantially worse—would strengthen the methodological contribution. This is a notable gap, since the paper emphasizes these techniques in Section 2.1 as critical to successful training.

- **No ablation of the temperature-to-top-p micro-dependency:** The design choice to feed the predicted temperature as input to the top-p head (shown as the dashed arrow in Figure 1) is not ablated. An alternative architecture where both heads operate independently from the hidden state alone (without the top-p head seeing the temperature prediction) would test whether the micro-dependency provides meaningful benefit or is an unnecessary complication.

- **No ablation of the soft top-p steepness parameter** `$\alpha$`: The paper uses `$\alpha = 30$` throughout but does not explore how sensitive the results are to this choice. Smaller `$\alpha$` values would produce a smoother, less accurate approximation of the hard cutoff (potentially introducing a larger train-test mismatch), while larger values would produce a sharper approximation but with vanishing gradients in the transition region (potentially making the top-p head harder to train). The choice of 30 is presented as a fixed hyperparameter; its robustness is untested.

- **No comparison to alternative differentiable approximations:** The paper proposes a specific formulation for differentiable top-p (exponential decay mask with ReLU thresholding). Alternative approaches—sigmoid-based masks, Gumbel-softmax reparameterization, straight-through estimators—are not discussed or compared. This is not necessarily a weakness (the paper's contribution is the overall architecture and findings, not the specific mask function), but it means the differentiable top-p formulation is presented as a solution without evidence that it is superior to alternative approaches to the same problem.

- **No evaluation of dynamic decoding on deterministic tasks:** All benchmarks in the paper are tasks where some degree of stochasticity is beneficial (math reasoning with multiple valid approaches, creative QA, code generation with multiple implementations). The paper does not evaluate on tasks where deterministic decoding is known to be optimal (e.g., extractive QA, factual knowledge retrieval, translation), which would test whether AutoDeco correctly learns to predict near-zero temperature in those contexts. This is a missing negative control: can AutoDeco "do no harm" on tasks where any stochasticity degrades performance?

- **Negative result: Qwen3-30B-A3B shows smaller gains (Tables 1–2):** The paper attributes the smaller improvements on Qwen3 to its shorter output lengths, which reduce sensitivity to sampling parameters. This is a useful finding—it establishes a boundary condition on when dynamic decoding provides value. Tasks or models that produce very short, deterministic outputs (e.g., classification, short-form QA, models fine-tuned for concise answers) may not benefit substantially from AutoDeco. The paper does not systematically test this hypothesis (e.g., by varying output length constraints or comparing long-form vs. short-form generation on the same task), leaving it as an observation rather than a verified boundary condition.

### Critical Assessment

**Do the experiments demonstrate that AutoDeco "consistently matches or exceeds the performance of expert-guided tuning"?**

Yes, with qualifications. Figure 3 shows that AutoDeco achieves accuracy within approximately one point of the oracle-tuned baseline on all eight test conditions (2 models × 4 benchmarks). This is strong evidence that dynamic, token-level decoding parameter prediction can substitute for exhaustive per-task grid search—with the crucial advantage that AutoDeco requires zero per-task tuning. The qualification is that the "expert-guided tuning" sweeps temperature and top-p at a granularity of 0.1, which may miss optimal settings between grid points. A finer-grained sweep might find slightly better static configurations, widening the gap. Additionally, the oracle tuning has access to the test set, which means it can overfit to the specific 128 samples per problem used for evaluation—a properly conducted validation-set-based tuning might perform worse than AutoDeco even if the oracle appears to match it. The paper's claim that AutoDeco is "effectively superior to any feasible expert-tuning strategy in practice" is justified, since any real-world tuning process must use a validation set, introducing generalization error that AutoDeco avoids entirely.

**Do the experiments demonstrate that AutoDeco "consistently outperforms standard default decoding settings"?**

Yes, very strongly. Across 4 models × 8 benchmarks = 32 evaluation settings, AutoDeco achieves higher Pass@1 than Default Sampling in 31 cases (the single exception being MMLU-Pro on Default Sampling vs. AutoDeco for Qwen3-30B-A3B, where the difference is within statistical noise: 76.25 vs. 78.38 with AutoDeco actually higher). AutoDeco outperforms Greedy Search in all 32 cases. The consistency of this result—across model families, model scales, dense and MoE architectures, math and non-math tasks—provides strong evidence that dynamic decoding is a robust improvement over static defaults.

**Do the experiments demonstrate that AutoDeco generalizes zero-shot to out-of-domain tasks?**

Yes, and this is one of the paper's strongest empirical contributions. The training data (DeepMath-103K reject sampling trajectories) shares essentially no surface-level features with GPQA-Diamond (graduate-level science QA), LiveCodeBench (code generation), or IFEval (instruction following with verifiable constraints). The consistent gains on these tasks (Table 2) demonstrate genuine zero-shot transfer. However, the paper does not test the limits of this transfer. It is unknown whether AutoDeco would generalize to fundamentally different modalities (e.g., structured data generation, multilingual tasks, long-form creative writing) or to tasks with very different output distributions (e.g., extremely short outputs, extremely long outputs, non-text outputs). The generalization claim is well-supported for the tested domains but should not be assumed universal.

**Do the experiments demonstrate emergent instruction-based decoding control?**

Yes, with important caveats about robustness and precision. Figure 5 shows a compelling qualitative example, and Table 4 quantifies the directional consistency (85–99%) after targeted training. The claim that this capability is "emergent" (arising from standard AutoDeco training without any diversity-control commands) is supported by the observation that the behavior appeared before targeted training, albeit inconsistently. The "emergent" label is appropriate—the model was not trained to respond to diversity commands, yet the architecture creates a pathway for linguistic understanding (encoded in hidden states) to influence decoding parameter predictions.

The caveats: (1) the emergent behavior was inconsistent before targeted training, so the out-of-the-box system does not provide reliable instruction-based control; (2) the control remains directional, not absolute (the model cannot hit a target temperature on command); (3) the evaluation uses only 100 test prompts, and the specific commands tested ("more innovative and diverse," "as certain as possible") are not systematically varied to test generalization across phrasings; (4) the targeted training procedure (ranking loss on augmented prompts) is described at a high level without sufficient detail for replication. The finding is genuinely novel and important, but the experiments supporting it are preliminary—as the authors themselves acknowledge by not including this feature in the released models.

**What is the strongest evidence that AutoDeco learns a genuine capability rather than memorizing dataset-specific patterns?**

The out-of-domain generalization results (Table 2) are the strongest evidence. If the heads were memorizing math-specific patterns ("on problems of type X, use temperature Y"), they would fail on GPQA-Diamond, LiveCodeBench, and IFEval. The fact that they succeed—and often with larger gains than in-domain—strongly suggests learning of transferable principles. The pattern where AutoDeco correctly predicts lower temperatures on tasks where Default Sampling underperforms Greedy Search (e.g., Llama-Nemotron-8B on GPQA-Diamond) is particularly telling, because it shows the model responding to task-level demands that it could not have memorized from math training data. The pass@k results (Tables 5–7) provide additional evidence: if AutoDeco were merely sharpening distributions around a memorized correct answer, pass@k performance would degrade (reduced diversity means fewer distinct correct answers to find via best-of-N). Instead, pass@k improves, indicating that the entire sampling distribution is better calibrated.

**What are the genuine weaknesses in the experimental design?**

1. **No baseline from the learned-decoding or controlled-generation literature.** The paper's comparison is exclusively against static decoding baselines (Greedy, Default Sampling, oracle-tuned static). No comparison is made to contrastive decoding (Li et al., 2023; Chuang et al., 2023), which also modifies the sampling distribution using learned or heuristic signals. A comparison to contrastive decoding would test whether AutoDeco's dynamic control provides benefits beyond what a well-tuned static modification can achieve. The paper argues that AutoDeco is orthogonal to these methods, which is true architecturally, but a head-to-head comparison would strengthen the claim that AutoDeco is the best available approach.

2. **No evaluation on deterministic-optimal tasks.** All benchmarks are tasks where some stochasticity is beneficial. The paper does not test whether AutoDeco correctly predicts near-zero temperature on tasks where any randomness harms performance—extractive QA, factual knowledge retrieval, closed-book factual probing. Without this negative control, it is unknown whether AutoDeco is safe to deploy as a default decoding strategy or whether it should be restricted to tasks where diversity is desired.

3. **Training data from the same base models used for evaluation.** The reject sampling trajectories are generated by the same models that are evaluated. This is necessary (the heads need to learn from the model's own output distribution), but it means the training data is model-specific. The paper demonstrates that AutoDeco transfers across tasks but does not test whether AutoDeco heads trained on one model transfer to a different model—a more challenging test of the "meta-skill" hypothesis.

4. **Small number of runs for statistical significance testing.** Pass@1 is estimated with 8 random seeds, 16 samples per seed, yielding 128 total samples per problem. For the 500-question AIME benchmark, the standard error of the mean Pass@1 is approximately `$\sqrt{p(1-p)/500}$`, which is roughly 2.2 percentage points at p = 0.5. The observed improvements of 2–4 points for some model-benchmark pairs are within or near this margin. The standard deviations across seeds are reported (±1–2 points), which provides some confidence, but formal hypothesis tests (paired t-test across seeds, bootstrap confidence intervals for the difference) are not reported.

5. **Undisclosed computational cost of the Expert-Guided Tuning baseline.** The grid search over 10 temperature × 10 top-p values requires evaluating 100 configurations on the full test set, generating 128 samples per problem per configuration—a total of 12,800 samples per problem. For AIME with 500 problems, this is 6.4 million samples for a single model-benchmark pair. The paper does not quantify this cost or compare it to AutoDeco's training cost (400 steps, 6,000 samples). Such a comparison would strengthen the practical argument for AutoDeco.

6. **No ablation of key design choices.** As noted above, Easy-Token Masking, Dynamic Fine-Tuning, the temperature-to-top-p micro-dependency, and the soft top-p steepness `$\alpha$` are not ablated. These are described as important but their individual contributions are unquantified. A failure mode analysis—what happens if you remove each component—would substantially strengthen the methodological contribution and provide practical guidance for implementation.

7. **The instruction-based control experiments are preliminary.** The 100-prompt test set, the unspecified diversity-control commands, the unspecified ranking loss formulation, and the acknowledgment that precise absolute control has not been achieved all indicate that this finding, while exciting, is at an early stage. The paper appropriately labels it as preliminary and excludes it from the released models, but the prominence given to this finding in the abstract and introduction could mislead readers about its maturity.

8. **Single-pass evaluation without comparison to best-of-N with static parameters.** The paper demonstrates that AutoDeco improves Pass@1 and maintains or improves Pass@k, but does not compare AutoDeco's Pass@1 against best-of-N with static parameters. For instance, does AutoDeco with 1 sample outperform Default Sampling with 4 samples? This "compute-matched" comparison would test whether AutoDeco's benefits are equivalent to simply generating more samples with a static strategy—a practically important question for deployment where sampling budget is flexible.

**Which experiments would have strengthened the paper?**

- **Ablation of Easy-Token Masking rate and Dynamic Fine-Tuning:** Training AutoDeco with masking rates of 0%, 30%, 60%, 90% would characterize the sensitivity to this hyperparameter and validate the paper's claim that masking prevents degenerate conservative behavior.

- **Ablation of soft top-p steepness `$\alpha$`:** Testing `$\alpha \in \{10, 20, 30, 50, 100\}$` would reveal whether the choice of 30 is near-optimal or whether results are robust across a wide range.

- **Comparison to contrastive decoding:** Evaluating contrastive decoding with the same base model (using a smaller model from the same family as the amateur) on the same benchmarks would contextualize AutoDeco's gains relative to the most prominent alternative approach to modifying sampling distributions.

- **Deterministic-task evaluation:** Testing on SQuAD, Natural Questions, or a factual knowledge probe would establish whether AutoDeco is safe for tasks where stochasticity is harmful.

- **Cross-model transfer:** Training AutoDeco heads on one model (e.g., Llama-Nemotron-8B) and evaluating on a different model from the same family (e.g., a smaller Llama variant) would test the generality of the learned decoding policy.

- **Compute-matched comparison:** Comparing Pass@1 of AutoDeco (1 sample) against Pass@1 of Default Sampling with best-of-N where N is chosen to equalize total FLOPs (accounting for AutoDeco's ~1.7% overhead) would address whether dynamic decoding provides benefits beyond simply drawing more samples.

- **Confidence interval or hypothesis test for the main results:** Reporting 95% confidence intervals or paired t-test results for the difference between AutoDeco and baselines would quantify the statistical reliability of the claimed improvements, particularly for the smaller gains (1–2 points) where sampling error could account for the difference.

**Do the claims hold conditionally?**

The central claim—that AutoDeco consistently outperforms static decoding baselines—holds unconditionally across all tested models, tasks, and benchmarks. There is no evidence of tasks or conditions where AutoDeco underperforms Default Sampling. The documented exception (smaller gains on Qwen3-30B-A3B) is a matter of magnitude, not direction: AutoDeco still improves performance, just by less.

The claim of expert-tuning-competitive performance holds with the qualification that the expert tuning was at a granularity of 0.1. A finer grid search (0.05 or 0.01) might find slightly better static configurations, potentially creating a small gap with AutoDeco. However, even if such a gap exists, the practical argument stands: AutoDeco achieves its performance without any tuning, while the expert baseline is an oracle that cannot exist in practice.

The claim of zero-shot generalization holds for the specific out-of-domain tasks tested (GPQA-Diamond, MMLU-Pro, LiveCodeBench, IFEval) but should not be assumed to extend to all possible tasks. The paper does not characterize the "generalization envelope"—what properties of a task predict whether AutoDeco will transfer effectively.

The claim of emergent instruction-based control holds as a demonstration of a new capability but not as a robust, deployable feature. The paper's caveats (directional only, not absolute; inconsistency before targeted training; exclusion from released models) bound the claim appropriately, but the prominence of this finding in marketing the paper's contributions could mislead readers who do not carefully read Section 3.3's limitations.

## 6. Limitations and Trade-offs

### The Difficulty Estimation Problem Remains Unsolved — For an Entirely Different Reason Than the Paper Acknowledges

The paper's central architectural innovation is making decoding parameters predicted rather than configured. But this raises a deep question the paper does not address: **how do we know whether the learned predictions are any good in deployment?** The temperature and top-p heads are trained end-to-end to minimize next-token cross-entropy loss on correct training trajectories. At inference time, the model predicts parameters for novel prompts. But there is no mechanism to detect when those predictions are **wrong** — when the head predicts an inappropriately high temperature on a factuality-critical token, or a near-zero top-p that eliminates the correct answer from consideration.

This is not the standard "no ground-truth labels" problem that the paper explicitly solves via the differentiable pipeline. The paper's training approach works precisely because ground-truth labels are unnecessary — the loss signal flows through the sampling operations. The unsolved problem is different: **at deployment time, there is no feedback signal, no confidence estimate, and no fallback mechanism for the predicted parameters.** If the temperature head overestimates uncertainty on a particular token and flattens the distribution excessively, causing the model to sample a wrong answer, the system has no way to detect this failure, let alone correct it. The model cannot "know what it doesn't know" about its own decoding parameter predictions — it cannot ask for help, route to a human, or fall back to conservative static parameters when its predictions are unreliable.

**Consequence**: The AutoDeco architecture replaces one source of brittleness (static hyperparameters that are suboptimal for most tokens) with another (learned predictions that may be catastrophically wrong on out-of-distribution inputs without any detection mechanism). On the benchmarks tested, the predictions are good enough to improve performance. But nothing in the architecture guarantees this will hold on arbitrary user inputs. A practitioner deploying AutoDeco would want to know: under what conditions do the predicted parameters become harmful rather than helpful? The paper provides no characterization of failure modes, no confidence intervals on predicted parameters, and no safe-fallback behavior.

**Evidence in the paper**: The paper provides no analysis of when or why the predicted parameters might be wrong. There is no evaluation of the calibration or reliability of the temperature and top-p predictions independently of downstream task accuracy. The only hint of this issue is the Qwen3-30B-A3B result (Tables 1–2), where gains are smaller — the authors hypothesize this is due to shorter output lengths making the task less sensitive to sampling parameters, but an alternative interpretation is that the AutoDeco head predictions are less reliable on this model for unspecified reasons. Without prediction-level diagnostics, these explanations are indistinguishable.

**Mitigation status**: Not addressed. The paper does not propose any mechanism for detecting or recovering from incorrect parameter predictions at inference time. The frozen base model architecture prevents any form of joint calibration between the language model's uncertainty and the heads' parameter predictions. This is a fundamental architectural limitation, not a training procedure issue — the heads are structurally unable to express uncertainty about their own predictions.

---

### The Training Data Construction — Via Reject Sampling from the Same Models Being Evaluated — Creates a Circularity That Makes It Unclear What AutoDeco Actually Learns

The training procedure for AutoDeco uses **reject sampling trajectories** generated by the same base models that will later be evaluated. Specifically: each base model generates multiple solutions to DeepMath-103K problems, and only trajectories producing correct final answers are retained. The AutoDeco heads are then trained to predict temperature and top-p values that maximize the probability of these correct trajectories.

This creates a circularity that the paper does not acknowledge. The training data consists of (prompt, correct continuation) pairs where the continuation was generated under the base model's **default** sampling distribution (the reject sampling uses whatever temperature and top-p were set during data generation — likely default values, though the paper does not specify). The heads are then trained to predict parameters that make these continuations more probable under a **different** (dynamically modulated) sampling distribution. But the heads are never trained on data generated under their own dynamic distribution — they learn to optimize a distribution they never sampled from.

This is not inherently invalid (off-policy learning is common), but it has a specific consequence: **the heads learn to predict parameters that would have been good for trajectories sampled under default parameters, not parameters that are optimal for trajectories sampled under dynamic parameters.** There is no guarantee that the optimal temperature for a token in a default-sampled trajectory is the same as the optimal temperature for that token in a dynamically-sampled trajectory, because the dynamic parameters change the entire future of the generation — earlier tokens sampled with different stochasticity lead to different contexts for later tokens.

**Consequence**: The heads may learn a distributional "sweet spot" that works well when applied to the base model's default distribution, but that does not necessarily correspond to the optimal dynamic policy. This could explain the paper's finding that AutoDeco gains are larger on some models and tasks than others — the mismatch between the training distribution (default-sampled trajectories) and the deployment distribution (dynamically-sampled trajectories) may vary in severity. More importantly, it means that AutoDeco is not learning in the fully "end-to-end" sense the paper claims — the training signal is mediated by a data collection process that is itself not end-to-end.

This issue is structurally similar to the well-known problem in imitation learning where a policy trained on expert demonstrations fails when deployed because the distribution of states it encounters differs from the distribution in the training data. In AutoDeco's case, the "expert demonstrations" are correct trajectories sampled under default parameters, and the "policy" is the dynamic temperature/top-p schedule. When deployed, the dynamic schedule produces different trajectories, leading to a distribution shift.

**Evidence in the paper**: The training procedure is described in Section 2.1 and Appendix Section 6. The paper states that heads are "trained on a specialized dataset of reject sampling trajectories" generated "from our four base models on problems from the DeepMath-103K dataset." It does not specify what decoding parameters were used to generate these trajectories. This is a significant omission — if the trajectories were generated with temperature = 1.0 and top-p = 1.0 (Default Sampling), then the heads are essentially learning to predict parameters that would have improved those specific trajectories, which is a weaker claim than learning optimal parameters for dynamic generation.

**Mitigation status**: Not addressed. The paper does not acknowledge this circularity or discuss alternatives. A possible mitigation would be iterative training: use the trained AutoDeco heads to generate new trajectories, filter for correctness, and retrain the heads on this on-policy data. This would align the training and deployment distributions at the cost of additional computation. The paper's ReST^EM experiment is not attempted for AutoDeco heads (the experiment is about revision models). The authors identify joint training of the base model and AutoDeco heads as future work (Section 5), which would address this circularity by allowing the base model's representations to adapt to the dynamic sampling distribution, but this is presented as an aspiration rather than a concrete plan.

---

### The "Drop-in" Claim Overstates Practical Deployability — AutoDeco Requires Per-Model Training and There Is No Evidence of Cross-Model Transfer

The paper presents AutoDeco as a "drop-in enhancement for any transformer-based model" (Section 5) and emphasizes the simplicity of the integration: "a '1-line-change' in a user's code" (Section 2.2). This is accurate for the **inference interface** — once trained, an AutoDeco-augmented model is used exactly like its base counterpart. But it omits a critical deployment prerequisite: **the AutoDeco heads must be trained separately for each base model**, and this training requires model-specific reject sampling data.

The paper trains separate AutoDeco heads for each of the four evaluation models (Llama-Nemotron-8B, R1-Distill-Qwen-7B, Qwen3-30B-A3B, GPT-OSS-20B). Each training run requires generating solutions from that specific model on DeepMath-103K, filtering for correctness, and running 400 optimization steps on 8 GPUs. The paper emphasizes that this is "resource-friendly" and "effortlessly integrated" (Appendix Section 7), but "resource-friendly" for an organization with 8 GPUs and the expertise to set up DeepSpeed ZeRO-3 training is not the same as "drop-in" for the typical practitioner who downloads a model from Hugging Face and expects it to work immediately.

**Consequence**: For AutoDeco to benefit a new model, someone must perform the training pipeline. This is not a one-time cost — it must be repeated for every base model variant, every fine-tuned checkpoint, and potentially every significant model update. The paper provides no evidence that AutoDeco heads trained on one model transfer to a different model (cross-model transfer), or even to a different fine-tuned version of the same model. If cross-model transfer fails, the cost of deploying AutoDeco at scale — across dozens or hundreds of model variants in a production ecosystem — could be substantial.

The paper's claim that training converges in "about only 6K training samples and 400 steps" (Appendix Section 7) is presented as evidence of efficiency, but it also reveals a dependency: **you need 6,000 correct, model-specific trajectories before you can train the heads.** Generating these trajectories via reject sampling requires running the base model on at least that many problems (likely many more, since only correct trajectories are retained). For a 30B-parameter MoE model like Qwen3-30B-A3B, generating 6,000+ solutions on challenging math problems is not trivial — it requires substantial GPU-hours before AutoDeco training even begins.

**Evidence in the paper**: The training setup is described in Section 3.1 and Appendix Section 6. Each model's heads are trained independently on model-specific reject sampling data. There is no experiment testing whether heads trained on one model can be applied to another (e.g., training on Llama-Nemotron-8B and evaluating on R1-Distill-Qwen-7B, or training on the 7B model and evaluating on the 30B model). The paper releases pre-trained AutoDeco heads for three additional large-scale models (Deepseek-V3.1-Terminus, Qwen3-235B-A22B-Thinking-2507, GPT-OSS-120B) but explicitly notes these were not comprehensively benchmarked "due to the substantial computational cost of evaluating these large-scale models" (Section 1), which further illustrates the resource barrier.

**Mitigation status**: Not addressed as a limitation. The paper frames the training cost as minimal and the method as widely applicable, but does not grapple with the per-model training requirement. Cross-model transfer is not suggested as future work. A possible mitigation would be to train a single set of AutoDeco heads on a diverse set of models and demonstrate that they transfer to unseen architectures — this would transform the method from "per-model fine-tuning" to "universal decoding module." The paper does not pursue this direction.

---

### The Paper's Core Claims Rest on a Single Training Domain (Mathematics) and the Generalization Results, While Impressive, Leave Critical Boundary Conditions Uncharacterized

AutoDeco is trained exclusively on mathematical reasoning data (DeepMath-103K with reject sampling) and evaluated on math, QA, code, and instruction following. The generalization results are genuinely impressive — the out-of-domain gains in Table 2 often match or exceed in-domain gains. But the paper provides no systematic characterization of **when this transfer succeeds and when it fails**, which is essential information for a practitioner deciding whether to deploy AutoDeco on their specific task.

Several important task categories are entirely untested. First, **tasks where any stochasticity is harmful**: extractive question answering (where the answer is a span in a provided document), factual knowledge probing (where the model either knows the fact or doesn't), closed-book translation, and classification tasks. In these settings, the optimal decoding strategy is deterministic — temperature → 0. Does AutoDeco learn to predict near-zero temperature on such tasks, or does it inappropriately inject stochasticity learned from math training (where multiple valid reasoning paths exist)? The paper's inability to achieve near-zero temperature even when explicitly commanded (Section 3.3: "a modest but directionally correct drop in the average predicted temperature, rather than the ideal of near-zero") suggests the latter may be the case.

Second, **highly open-ended creative tasks**: storytelling, poetry, dialogue generation, brainstorming. In these settings, the relationship between decoding parameters and output quality is fundamentally different from math — there is no single "correct" answer, and the optimal stochasticity may be much higher than anything encountered in math training. The paper's training data (correct mathematical reasoning chains) may bias the heads toward lower temperatures than are optimal for creative generation.

Third, **tasks with extreme output lengths**: very short answers (a few tokens, as in classification or factual QA) and very long generations (thousands of tokens, as in story generation or long-form reasoning). The paper observes that Qwen3-30B-A3B, which produces shorter outputs, shows smaller gains — but does not systematically test whether output length is a moderating variable. In very long generations, the compounding effect of per-token parameter predictions could amplify errors (a single poorly-predicted temperature early in the generation could derail the entire trajectory).

**Consequence**: A practitioner considering AutoDeco for a task that differs substantially from mathematical reasoning — particularly deterministic tasks or highly creative tasks — has no evidence to predict whether it will help, be neutral, or harm performance. The paper's claim that AutoDeco is "a practical, drop-in enhancement for any transformer-based model" (Abstract) is by construction true only for tasks resembling the evaluation benchmarks. Whether it extends to the full diversity of real-world LLM use cases is unknown.

**Evidence in the paper**: The generalization results (Tables 1–2) cover eight benchmarks spanning math, QA, code, and instruction following. This is a respectable diversity, but all benchmarks share the property that there is a verifiable correct answer (math solution, QA answer choice, code that passes tests, instruction verified by a protocol). None of the benchmarks are purely deterministic (where greedy decoding is optimal), purely creative (where diversity is the primary metric), or extremely long-form. The paper provides no negative results on any task category, which is suspicious — either the method genuinely never underperforms (unlikely), or tasks where it underperforms were not tested.

**Mitigation status**: Not addressed. The paper does not delineate the expected scope of applicability or identify task categories where dynamic decoding might be harmful. The future work section (Section 5) focuses on joint training and more precise instruction-based control, not on characterizing generalization boundaries. A responsible deployment would require validation on the specific target task — but if such validation is necessary anyway, one of AutoDeco's key selling points (eliminating per-task tuning) is partially undermined.

---

### The Emergent Instruction-Based Control Finding — the Paper's Most Exciting Result — Is Preliminary to the Point of Being Unsubstantiated

The paper's most attention-grabbing claim is that AutoDeco enables natural language control over decoding parameters. The abstract states: "we uncover an emergent capability for instruction-based decoding control: the model learns to interpret natural language commands... and adjusts its predicted temperature and top-p on a token-by-token basis." The Introduction describes this as "perhaps most excitingly" and frames it as a major contribution.

However, the actual experimental support for this claim is thin. The capability is demonstrated qualitatively on a single prompt in Figure 5. The quantitative results in Table 4 use only 100 test prompts, evaluate only directional consistency (did the temperature move in the right direction?), and do not report task performance (did the adjusted parameters actually improve output quality for the intended goal?). The targeted training procedure that makes the behavior reliable (resulting in 85–99% consistency) is described in a single paragraph without sufficient detail for replication: the diversity-control commands are not listed, the ranking loss is not formally specified, and the training data augmentation process is not described.

Most importantly, **the paper itself acknowledges these limitations and excludes the feature from released models**:

> "We do not yet have a conclusive understanding of this phenomenon... The models released with this paper do not include this experimental control feature. We will continue to advance this line of research and release updated models as soon as we reach more definitive conclusions." (Section 3.3)

This is honest and appropriate for a research paper. But the prominence given to this finding in the abstract and introduction — where it is presented as a core contribution alongside the performance and efficiency results — creates a mismatch between the claimed contribution and the evidence provided. The claim that AutoDeco "endows the model with a new, intuitive way to interpret and act on user intent" (Introduction) is aspirational given the current evidence.

**Consequence**: Readers who skim the abstract and introduction may come away believing that AutoDeco provides reliable, deployable instruction-based decoding control. In reality, this capability is at the stage of an interesting observation that has been partially validated and requires substantial further research. A practitioner attempting to use instruction-based control in a production system would find no released model supporting it, no specification of which commands work, no guarantee of behavior on out-of-distribution phrasings, and no associated task performance data.

**Evidence in the paper**: Section 3.3, Figure 5, and Table 4 contain all the evidence for this claim. The section is notably shorter and less detailed than the performance and efficiency sections. The paper's cautionary language ("we do not yet have a conclusive understanding," "we hypothesize," "given the preliminary nature of this investigation") appears only in Section 3.3 itself, not in the abstract or introduction where the strongest claims are made. This is a structural problem in the paper's framing rather than a factual error.

**Mitigation status**: Partially addressed through honesty about the limitations, but the paper's framing oversells the finding. A more appropriate presentation would label this as "preliminary evidence of an emergent capability" in the abstract and introduction, and clearly separate it from the well-supported performance and efficiency claims. The future work section (Section 5) identifies joint training of the base model and AutoDeco heads as a path toward more precise control, but this is a significant research undertaking, not a minor refinement.

---

### The Expert-Guided Tuning Baseline — the Paper's Primary Comparative Upper Bound — Is a Weak Oracle That Overstates AutoDeco's Practical Advantage

The paper's headline comparison (Figure 3) shows that AutoDeco matches or slightly exceeds "Expert-Guided Tuning" — a grid search over temperature (0.1 to 1.0, step 0.1) and top-p (0.1 to 1.0, step 0.1) performed directly on the test set. The paper correctly notes this is an oracle setting (it "hacks the test set") and uses it to establish that AutoDeco achieves the practical upper bound for static decoding.

But there are two problems with this baseline that make the comparison less informative than it appears.

**First, the grid granularity is coarse.** Sweeping temperature at 0.1 intervals means the expert tests only 10 temperature values. If the true optimal temperature is 0.65, the grid search finds either 0.6 or 0.7 — suboptimal by up to 0.1. A finer grid (0.05 or 0.01) would find better static configurations. AutoDeco's predicted temperatures are continuous — they could, in principle, hit 0.65. So part of AutoDeco's apparent advantage over the "expert" may simply be the granularity of the search, not any fundamental superiority of dynamic over static decoding. The expert could be made stronger without violating any practical constraint (a 0.05-step grid is 400 configurations — computationally expensive but not infeasible for production deployment).

**Second, the sequential search procedure (temperature first, then top-p) is known to be suboptimal** because temperature and top-p interact. The optimal top-p at temperature = 0.3 may be different from the optimal top-p at temperature = 0.8. By fixing temperature first and then searching top-p, the expert misses configurations where a slightly suboptimal temperature combined with a better top-p would yield higher overall accuracy. A full grid search over all (temperature, top-p) pairs — 100 configurations at 0.1 granularity — would be a stronger oracle. The paper's sequential procedure reduces this to 20 evaluations (10 temperature + 10 top-p) but at the cost of missing interaction effects.

**Consequence**: The Expert-Guided Tuning baseline is weaker than it could be, even within the constraints of static decoding. AutoDeco's ability to match this baseline is less impressive if the baseline itself leaves performance on the table due to search procedure limitations. More importantly, the comparison does not establish that dynamic decoding is **necessary** — it is possible that a better static tuning procedure (finer grid, full grid search, or Bayesian optimization) would match or exceed AutoDeco's performance on these benchmarks. The paper's argument that AutoDeco is "effectively superior to any feasible expert-tuning strategy in practice" (Section 3.2.1) assumes that the tested expert tuning procedure is representative of the best possible static tuning, which is not established.

**Evidence in the paper**: Figure 3 and the description in Section 3.2.1 specify the search procedure: "Temperature is adjusted first (setting top-p to 1.0), and the selection is made based on the best performance of temperature to conduct the search for top-p." This is a sequential coordinate-ascent search, which is known to converge to suboptimal points for non-separable objective functions. The paper does not compare against a full grid search or any other optimization procedure.

**Mitigation status**: Not addressed. The paper does not discuss the limitations of sequential search or the granularity of the grid. A stronger comparison would use Bayesian optimization (which can find better configurations with fewer evaluations) or, for the oracle setting, a full grid search at finer granularity to establish a true upper bound. The fact that the oracle baseline is meant to represent the best possible static performance makes these limitations particularly consequential — if the "best possible" is underestimated, the paper's central claim about matching it is less meaningful.

## 7. Implications and Future Directions
- Impact on practice:
  - AutoDeco can eliminate manual decoding sweeps. For production systems, this means higher, more stable performance across diverse inputs with zero per-task retuning and ~1–2% extra latency (Sections 3.2.1–3.2.2; Table 3).
  - The per-token `T̂/P̂` traces can serve as an introspective signal of uncertainty or creativity demand, aiding downstream systems (e.g., when to verify, when to sample more).
- Impact on research:
  - Decoding becomes part of the learnable model. This reframes generation quality as an optimization problem over both “what” to say and “how” to explore while saying it.
  - The differentiable soft top‑p is a tool that can enable training other decoding controls end-to-end (e.g., repetition penalties, length normalization).
  - Natural-language control of decoding opens a new interface—users express style constraints in words rather than API knobs—and suggests work on more precise, quantitative control (Section 3.3).
- Concrete next steps (some noted in Section 5 “Future Work” and Section 3.3):
  - Jointly train the backbone with AutoDeco heads to achieve precise, absolute control (e.g., truly near‑zero temperature on “no randomness” prompts).
  - Expand controllable parameters (top‑k, penalties, contrastive or speculative decoding weights) and study interactions among them.
  - RL or bandit-style objectives that directly reward task outcomes while penalizing unnecessary randomness or verbosity.
  - Task-specific curricula (reasoning vs. creative writing) to learn richer, context-aware decoding policies.
  - System integration: couple `T̂/P̂` with self-verification or tool-use modules to condition verification or drafting depth on predicted uncertainty.

> Bottom line: By turning decoding from a static, manual post-process into a learned, token-level component of the model, AutoDeco delivers practical gains today (accuracy, simplicity, negligible overhead) and opens a rich design space for controllable, instruction-steerable generation tomorrow.
