# A Fast Learning Algorithm for Deep Belief Nets

**URL:** [https://www.cs.toronto.edu/~hinton/absps/ncfast.pdf](https://www.cs.toronto.edu/~hinton/absps/ncfast.pdf)

## 🎯 Pitch

This paper introduces a fast, greedy learning algorithm for deep, densely connected belief networks by using **complementary priors**—priors whose correlations exactly cancel the "explaining away" effects in the likelihood term, making the true posterior distribution over hidden variables factorial (e.g., an infinite directed net with tied weights that is equivalent to a restricted Boltzmann machi

---

## 1. Executive Summary

This paper introduces a fast, greedy learning algorithm for deep, densely connected belief networks by using **complementary priors**—priors whose correlations exactly cancel the "explaining away" effects in the likelihood term, making the true posterior distribution over hidden variables factorial (e.g., an infinite directed net with tied weights that is equivalent to a restricted Boltzmann machine). The method learns deep networks one layer at a time on the MNIST handwritten digit database by treating each pair of adjacent layers as a restricted Boltzmann machine trained with contrastive divergence, then uses a contrastive version of the wake-sleep algorithm called the **up-down algorithm** to fine-tune all weights jointly. After fine-tuning, a network with three hidden layers (~1.7 million parameters) achieves 1.25% error on the permutation-invariant MNIST test set, outperforming the best discriminative methods of the time (1.4% for support vector machines, 1.5% for backpropagation nets), establishing that generative models can surpass discriminative ones on classification only when a good generative model of the input distribution is learnable.

## 2. Context and Motivation

### The Core Problem: Deep Belief Nets Are Intractable to Learn

The paper addresses a fundamental barrier that, in 2006, had prevented neural network researchers from training networks with many layers of hidden units. The problem is not that deep networks lack representational power — they are exponentially more expressive than shallow ones — but that **learning the weights of a deep directed belief network is computationally intractable** because inference of the hidden states given data is itself intractable.

To understand why this matters, we need to be precise about the model class. The paper works with **directed belief nets** — specifically, logistic belief nets composed of stochastic binary units, where the probability that a unit turns on is a logistic (sigmoid) function of its parents' states and the weights on the directed connections:

$$p(s_i = 1) = \frac{1}{1 + \exp\left(-b_i - \sum_j s_j w_{ij}\right)}$$

When generating data, these nets operate by ancestral sampling: start at the deepest hidden layer (where the prior is factorial — each unit's state is drawn independently), then sample each layer in turn going downward. The prior being factorial during *generation* is a consequence of the directed structure. But once a data vector is observed, the posterior distribution over hidden variables — which we need to compute for learning — is emphatically **not** factorial. The observed data create statistical dependencies among the hidden variables, a phenomenon called **explaining away** (illustrated in Figure 2), which makes exact inference exponentially expensive in the number of hidden units.

This is not a minor inconvenience. For a network with three hidden layers of 500 units each, the joint posterior distribution over 1,500 binary variables has $2^{1500}$ possible configurations. No amount of computation can enumerate this. Prior approaches to learning such networks therefore fell into two camps, each with crippling limitations that the paper explicitly identifies.

**The first camp: variational methods.** Variational inference (Neal & Hinton, 1998) approximates the true intractable posterior with a simpler, tractable distribution — typically a factorial one — and optimizes a lower bound on the log probability of the data. The guarantee that learning improves the lower bound is reassuring, but the paper identifies two critical weaknesses:

> "the approximations may be poor, especially at the deepest hidden layer, where the prior assumes independence"

The factorial approximation becomes increasingly inaccurate as we go deeper because the true posterior correlations compound across layers. More importantly, variational methods still require **all parameters to be learned simultaneously**. The paper is explicit about the consequence:

> "variational learning still requires all of the parameters to be learned together and this makes the learning time scale poorly as the number of parameters increases"

For networks with millions of parameters, joint optimization of all weights was simply infeasible with the computational resources of the time.

**The second camp: Markov Chain Monte Carlo (MCMC).** MCMC methods (Neal, 1992) can, in principle, give unbiased samples from the true posterior. But they are "typically very time-consuming" because the Markov chain must run long enough to reach equilibrium for every single training example at every iteration of learning. For deeply connected networks, mixing times can be prohibitively slow.

Both approaches share a deeper conceptual limitation: they treat the explaining-away phenomenon as something to be *approximated around*. The paper's key conceptual move is to ask a different question: **can we eliminate explaining away entirely by designing the model's prior correctly?**

---

### Why This Matters: The Pre-2006 Landscape of Deep Learning

To appreciate why this paper is important, we must understand what neural network research looked like in 2006 and, critically, what the paper explicitly frames as the status quo limitation. The paper does not merely state that deep networks were hard to train — it provides a specific picture of *why that mattered*:

**Generative vs. discriminative learning.** The paper draws a sharp contrast between discriminative and generative models. In the category of approaches that dominated at the time:

> "In discriminative learning, each training case constrains the parameters only by as many bits of information as are required to specify the label"

This is stated as a *disadvantage* of discriminative models: they are sample-inefficient in the sense that a labeled example provides only $\log_2(10) \approx 3.3$ bits of constraint on millions of parameters. For a generative model:

> "each training case constrains the parameters by the number of bits required to specify the input"

An MNIST digit image is $28 \times 28 = 784$ pixels, providing orders of magnitude more information per example than a label alone. Generative models can therefore learn from unlabeled data and can learn many more parameters without overfitting. This matters because:

> "The superior classification performance of discriminative learning methods holds only for domains in which it is not possible to learn a good generative model. This set of domains is being eroded by Moore's law."

This is a forward-looking argument: as computation becomes cheaper, building good generative models of high-dimensional data becomes feasible, and when it is feasible, generative models should outperform discriminative ones on downstream tasks like classification. The paper explicitly positions its approach as evidence for this thesis.

**Shallow architectures and handcrafted features.** In the mid-2000s, state-of-the-art MNIST performance was achieved by either:

1. **Shallow backpropagation networks** with one or two hidden layers — typically achieving 1.5–3% error, with the best results requiring careful architecture design (cross-entropy loss, weight decay, specific hidden layer sizes). Table 1 shows these in detail: a net with 500 hidden units in layer 1 and 300 in layer 2 achieves 1.51% error; a single 800-unit hidden layer achieves 1.53%.

2. **Support vector machines** with polynomial kernels — achieving 1.4% error (Decoste & Schoelkopf, 2002), which was the best published result for the permutation-invariant version of the task at the time.

3. **Convolutional neural networks** (LeCun et al., 1998) achieving 0.95% on unpermuted images by exploiting spatial information through weight sharing and subsampling.

All of these approaches had clear limitations. Backpropagation in deep networks was known to work poorly — gradients vanish or explode as they propagate through many nonlinear layers, a phenomenon that would later be studied extensively but was already a practical barrier. The paper's own Table 1 implicitly shows this: the reported backpropagation results are for nets with *at most* two hidden layers. No one was training three-hidden-layer networks with backpropagation on MNIST and getting competitive results.

**Table 1 as a pre-2006 benchmark.** The paper's Table 1 (reproduced in Section 6) serves as a carefully constructed picture of the competitive landscape. On the permutation-invariant task (the "basic" version where no spatial knowledge is provided), the error rates are:

- 1.25% — this paper's generative model (new)
- 1.4% — SVM with degree-9 polynomial kernel (Decoste & Schoelkopf, 2002)
- 1.51% — Backprop: 784→500→300→10, cross-entropy and weight-decay
- 1.53% — Backprop: 784→800→10, cross-entropy and early stopping
- 2.95% — Backprop: 784→500→150→10, squared error and on-line updates
- 2.8–4.4% — Nearest neighbor (various configurations)

The gap between 1.25% and 1.4% is significant at this task — and more importantly, the generative model achieves this while also being able to generate digit images, interpret its hidden representations, and learn from unlabeled data. The paper is not merely claiming to match discriminative methods; it claims to surpass them on a task where the prevailing wisdom expected discriminative methods to dominate.

**The promise of deep architectures was unrealized.** The theoretical motivation for deep architectures was well understood: hierarchical representations can model complex functions with exponentially fewer parameters than shallow ones. The visual cortex is deep (Lee & Mumford, 2003, cited by the paper), suggesting that depth is important for perception. But a practical learning algorithm for deep, densely connected networks with many hidden layers did not exist.

---

### Where Prior Approaches Fall Short: A Systematic Breakdown

The paper identifies specific failure modes of existing methods, each of which its approach directly addresses:

#### 1. Explaining Away Makes Exact Inference Exponential

This is the central technical obstacle. The "explaining away" phenomenon (Figure 2) is not merely a minor correlation — it means that the posterior over hidden variables cannot be factorized without losing information. In a densely connected belief net, every pair of hidden variables becomes dependent when data is observed, and the joint posterior must be represented as a full probability table over all $2^H$ configurations.

Prior work attempted to handle this with variational approximations that assume a factorial posterior. But as the paper notes, these approximations are "especially poor at the deepest hidden layer, where the prior assumes independence." The deepest hidden layer is precisely where we want the most abstract, useful representations — and it is where the approximation is worst.

#### 2. Variational Methods Require Joint Learning of All Parameters

Even with a factorial approximation, variational learning (Neal & Hinton, 1998) optimizes a lower bound that depends on all parameters simultaneously. There is no mechanism for learning the weights one layer at a time — changing the weights of one layer affects the optimal settings for all other layers. The learning time "scales poorly as the number of parameters increases," making deep networks with millions of weights impractical.

This is a crucial point: the paper's core algorithmic contribution is not just that it makes inference tractable, but that it enables **greedy, layer-by-layer learning** with a guarantee that adding layers never decreases a variational bound on the data probability.

#### 3. The Wake-Sleep Algorithm Has Mode-Averaging Problems

The wake-sleep algorithm (Hinton et al., 1995) was a prior attempt at learning deep directed networks with separate recognition (inference) and generative weights. It alternates between a "wake" phase (data-driven, updating generative weights) and a "sleep" phase (model-driven, updating recognition weights). The paper identifies a specific pathology:

> "the 'mode-averaging' problems that can cause the wake-sleep algorithm to learn poor recognition weights"

Mode averaging occurs when the recognition model learns to map a data point to a weighted average of multiple valid hidden representations rather than committing to one particular mode. This degrades the quality of the learned representations because the recognition distribution becomes diffuse, and the generative model receives ambiguous targets during learning.

The paper's contrastive version of wake-sleep (the "up-down" algorithm) addresses this by using a top-level associative memory that can settle to a particular mode before the down-pass, and by initializing the associative memory from the data-driven up-pass rather than from equilibrium.

#### 4. Contrastive Divergence Was Limited to Shallow RBMs

Contrastive divergence learning (Hinton, 2002) had already been shown to be efficient for training restricted Boltzmann machines — single-layer undirected models. The paper explicitly notes:

> "it appears that the efficiency has been bought at a high price: When applied in the obvious way, contrastive divergence learning fails for deep, multilayer networks with different weights at each layer because these networks take far too long even to reach conditional equilibrium with a clamped data vector"

The existing success of contrastive divergence was confined to shallow architectures. The paper's insight — that an infinite directed net with tied weights is equivalent to an RBM — provides a bridge from contrastive divergence in a single RBM to greedy layer-by-layer learning of a deep architecture.

#### 5. No Practical Method Existed for Generative Deep Learning

The paper's framing in Section 8 is explicit about the gap:

> "We have shown that it is possible to learn a deep, densely connected belief network one layer at a time."

The emphasis is on "it is possible" — prior to this work, this was an open question. The approach of assuming that higher layers do not exist when learning the lower layers (i.e., training a shallow model and then deepening it) "is not compatible with the use of simple factorial approximations to replace the intractable posterior distribution." The paper's complementary prior framework provides the theoretical justification for why the layer-by-layer approach is valid: assuming the higher layers have tied weights that implement a complementary prior makes the true posterior factorial at each layer, so the factorial approximation used during greedy learning is exact, not approximate.

---

### How This Paper Positions Itself

The paper positions itself not as an incremental improvement on existing training methods but as a **fundamentally different approach** that changes the nature of the inference problem rather than approximating around it.

**The core conceptual innovation: complementary priors.** Instead of accepting that explaining away must be approximated, the paper asks whether the prior over hidden variables can be designed to *exactly cancel* the correlations introduced by the likelihood. If such a "complementary prior" exists, the posterior becomes factorial — not approximately, but exactly — and inference becomes trivial:

> "it would be much better to find a way of eliminating explaining away altogether, even in models whose hidden variables have highly correlated effects on the visible variables. It is widely assumed that this is impossible."

The paper explicitly challenges this assumption. Section 2 demonstrates, through the infinite directed network with tied weights (Figure 3), that complementary priors do exist if the weights between layers are constrained to be equal. This is the conceptual bridge between directed belief nets and undirected RBMs: the infinite directed net with tied weights is exactly equivalent to an RBM, and the inference procedure — applying the transposed weight matrix layer by layer — gives unbiased samples from the true posterior.

**The algorithmic consequence: greedy layer-by-layer pretraining.** Once the equivalence is established, the learning algorithm follows naturally. Learning the tied weight matrix $W_0$ for the bottom layer under the assumption that all higher weights are tied to it is equivalent to training an RBM on the visible data. After $W_0$ is learned, the data can be transformed through $W_0^T$ to produce "data" for the next layer, and the process repeats. The paper provides a variational argument (Section 4) that this greedy procedure is guaranteed not to decrease a lower bound on the log probability of the data under the full model.

Crucially, this greedy learning is **unsupervised** — it does not require labels — and it produces initial weights that capture the statistical structure of the input distribution. The labels are incorporated only at the top layer, in the associative memory, by treating them as additional visible units in the top-level RBM. This means the model learns a joint distribution $P(\text{image}, \text{label})$ rather than a conditional $P(\text{label}|\text{image})$, which is why it can both classify and generate.

**The fine-tuning phase as a refinement, not a core dependency.** The paper is careful to note that the greedy algorithm provides "a fairly good set of parameters quickly" (Section 1), but the up-down fine-tuning is what produces the excellent generative model that achieves 1.25% error. However, the paper also speculates (Section 8) that the greedy algorithm alone might be sufficient if used to train "an ensemble of larger, deeper networks" — the fine-tuning is presented as one way to use the fast initializations, not the only way.

**Positioning relative to discriminative methods.** The paper's abstract and introduction make a bold claim: generative models can outperform discriminative ones on classification tasks. This was counter to the prevailing wisdom, which held that discriminative methods are superior for prediction because they focus on the decision boundary rather than modeling the input distribution. The paper's position is that this wisdom is contingent on computational constraints: when a good generative model *can* be learned (which requires enough computation and enough unlabeled data), it will outperform discriminative models because it can extract more information per training example and learn more parameters without overfitting.

**Positioning in the broader landscape of representation learning.** The paper frames its layer-by-layer learning as a form of representation transformation, drawing explicit analogies to boosting (Freund, 1995), projection pursuit (Friedman & Stuetzle, 1981), and Sanger's PCA algorithm (1989). The common thread is that a sequence of simpler models is learned, with each model modifying its input in a way that forces the next model to learn something new:

> "The idea behind our greedy algorithm is to allow each model in the sequence to receive a different representation of the data."

Unlike boosting (which reweights data points) or projection pursuit (which removes non-gaussianity in one direction), this approach re-represents the data through a nonlinear transformation. Each layer's RBM learns a new set of features that capture higher-order structure in the previous layer's feature activations.

---

### Summary: The Gap This Paper Fills

Before this work, the field faced a trilemma for deep belief nets:

1. **Use exact inference (MCMC)**: unbiased but computationally infeasible for deep nets.
2. **Use variational inference**: tractable but requires joint learning of all parameters and gives poor approximations at deeper layers.
3. **Use shallow models**: practical but cannot exploit the representational power of depth.

The paper's complementary prior framework resolves this trilemma by showing that a specific architectural constraint — tying the weights between layers to form an infinite directed model — makes the true posterior factorial, eliminating the need for approximation during the greedy pretraining phase. This enables:

- **Layer-by-layer learning** with contrastive divergence, which is fast and scales well.
- **Unsupervised pretraining** that learns statistical structure from unlabeled data.
- **A variational guarantee** that adding layers improves the generative model under maximum likelihood learning.
- **An initialization for fine-tuning** that enables the full model to reach a better solution than either purely discriminative methods or purely shallow approaches.

The paper does not claim to have solved deep learning entirely — the limitations section acknowledges that the model is designed for images where pixel intensities can be treated as probabilities, that top-down feedback is limited to the top two layers, and that the fine-tuning algorithm "is currently too slow." But it provides the first demonstration that deep, densely connected belief networks can be trained at all, and that doing so yields practical benefits on a standard benchmark.

## 3. Technical Approach

### 3.1 Reader Orientation

This paper presents a **training system** for deep belief networks — neural networks with many layers of stochastic binary neurons — that learns a generative model of handwritten digits and their labels. The core problem it solves is that deep directed networks are impossible to train directly because computing what each hidden neuron "should have done" given an observed image requires averaging over exponentially many possible hidden configurations; the solution is to design the network's prior over hidden variables to exactly cancel the troublesome correlations, making inference trivial, which enables a fast layer-by-layer training procedure followed by joint fine-tuning.

---

### 3.2 Big-Picture Architecture (Diagram in Words)

The system has five major components connected in a pipeline:

1. **Data layer (visible units):** 784 units (28×28 pixel intensities), plus 10 label units organized as a softmax group.
2. **Three hidden layers:** 500 units in the first hidden layer, 500 in the second, 2000 in the third (the top two layers form an undirected associative memory).
3. **Bottom-up recognition weights:** separate directed connections that flow upward, used to infer hidden states from visible data; initially tied to the transpose of the generative weights, later untied and refined.
4. **Top-down generative weights:** directed connections flowing downward (2000→500→500→784+10), defining the probability of lower layers given upper layers.
5. **Top-level associative memory:** the top two layers (500 units ↔ 2000 units) connected by undirected, symmetric weights, forming a restricted Boltzmann machine that can settle to equilibrium via Gibbs sampling.

Information flows in three distinct phases:
- **Greedy pretraining:** visible data → bottom RBM trained with contrastive divergence → learned features become "visible" data for the next RBM → repeat for three layers.
- **Up-pass (inference):** image clamped on visible units → stochastic binary states sampled layer-by-layer upward using recognition weights → terminates in the top-level associative memory.
- **Down-pass (generation):** starting from the associative memory state (possibly after Gibbs sampling) → stochastic binary states sampled layer-by-layer downward using generative weights → produces reconstructed image and label.

---

### 3.3 Roadmap for the Deep Dive

- **First, the complementary prior framework:** why explaining away makes inference intractable, what a complementary prior must do to cancel it, and how an infinite directed network with tied weights achieves this — this is the theoretical foundation that makes everything else possible.
- **Second, the equivalence between infinite directed nets and restricted Boltzmann machines:** how Gibbs sampling in an RBM corresponds exactly to inference in the infinite directed net, establishing that training an RBM with contrastive divergence trains the bottom layer of a deep architecture.
- **Third, contrastive divergence learning:** the specific algorithm for training each RBM, including the energy function, the weight update rule, and why the difference of two correlations provides a useful learning signal.
- **Fourth, the greedy layer-by-layer pretraining algorithm:** how weights are learned one layer at a time (bottom-up), how data is transformed through each learned layer to create input for the next, and the variational argument that guarantees this greedy procedure improves the overall generative model.
- **Fifth, the up-down fine-tuning algorithm:** how the weights from greedy pretraining are refined jointly using a contrastive version of the wake-sleep algorithm, including the specific updates for generative weights, recognition weights, and the top-level associative memory.
- **Sixth, the complete training protocol:** the exact architecture (784→500→500→2000↔10), hyperparameters (learning rates, momentum, weight decay, minibatch composition, epochs per phase), and testing procedures (deterministic vs. stochastic up-pass, free-energy computation for labels, the final 1.25% error configuration).

---

### 3.4 Detailed, Sentence-Based Technical Breakdown

This is a **theoretical framework and algorithm paper** whose core idea is that deep directed belief networks can be trained efficiently if the prior over hidden variables is designed to make the posterior factorial, and that this can be achieved by an architecture with tied weights between layers — an architecture equivalent to a restricted Boltzmann machine, enabling greedy layer-by-layer learning with contrastive divergence followed by joint fine-tuning.

---

#### The Explaining-Away Problem and Why It Matters

The central technical obstacle that the paper overcomes is called **explaining away**, illustrated in Figure 2 with a concrete example.

Consider two independent rare causes (an earthquake and a truck hitting the house) that can each make the house jump. In the prior (before observing anything), the earthquake node has a bias of −10, meaning $P(\text{earthquake}=1) \approx e^{-10}$; similarly for the truck. These two causes are **independent in the prior** — knowing about the earthquake tells you nothing about the truck.

When we observe that the house jumped, the posterior changes dramatically. The jump could be explained by either cause. If we infer that the earthquake happened, this "explains away" the evidence for the truck — the truck is no longer needed to explain the jumping, and since trucks are rare, its posterior probability drops. The two hidden causes become **strongly anti-correlated in the posterior** even though they were independent in the prior.

Formally, for a logistic belief net (Neal, 1992) with binary units `$s_i \in \{0,1\}$`, the probability that unit `$i$` turns on given its parent states is:

$$p(s_i = 1) = \frac{1}{1 + \exp\left(-b_i - \sum_j s_j w_{ij}\right)}$$

where `$b_i$` is the bias of unit `$i$` and `$w_{ij}$` is the weight from parent `$j$` to child `$i$`.

**What it computes:** this is the standard logistic (sigmoid) activation: the total input `$b_i + \sum_j s_j w_{ij}$` is passed through the logistic function to produce a probability between 0 and 1. A binary state is then sampled from a Bernoulli distribution with this probability. The probability is high when the weighted sum of active parents plus the bias is large and positive; it is low when this sum is large and negative. At zero total input, the probability is exactly 0.5.

**Why this form:** the logistic function is the canonical link function for binary variables in an exponential family. It makes the log-probability linear in the parent states: `$\log p(s_i=1) - \log p(s_i=0) = b_i + \sum_j s_j w_{ij}$`. This additive property in the log domain is what makes the energy-based formulation work — the log-probability of a joint configuration decomposes into a sum of pairwise interactions, which is essential for both the complementary prior construction and the connection to RBMs.

The consequence of explaining away is that the posterior distribution `$P(\mathbf{h}|\mathbf{v})$` over all hidden variables given an observed visible vector is a full joint distribution with complex dependencies — it does not factorize into a product of independent distributions per hidden variable. For `$H$` binary hidden units, representing this posterior requires `$2^H$` numbers in the worst case. Computing expectations under this posterior (needed for the learning gradient) is therefore exponential in `$H$`.

---

#### The Complementary Prior: Making the Posterior Factorial by Design

The paper's key insight is that rather than approximating the intractable posterior, we can design the prior `$P(\mathbf{h})$` so that when multiplied by the likelihood `$P(\mathbf{v}|\mathbf{h})$`, the resulting posterior `$P(\mathbf{h}|\mathbf{v}) \propto P(\mathbf{v}|\mathbf{h})P(\mathbf{h})$` is exactly factorial — that is, `$P(\mathbf{h}|\mathbf{v}) = \prod_j P(h_j|\mathbf{v})$` with each hidden unit conditionally independent given the data.

The paper calls such a prior a **complementary prior**: it has exactly the opposite correlations to those introduced by the likelihood, so that when they combine, the correlations cancel.

Appendix A formalizes the conditions under which complementary priors exist. The general result is:

**Likelihood form that admits a complementary prior.** The likelihood must be expressible as:

$$P(\mathbf{x}|\mathbf{y}) = \frac{1}{\Omega(\mathbf{y})} \exp\left(\sum_j \Phi_j(\mathbf{x}, y_j) + \beta(\mathbf{x})\right)$$

where `$\Omega(\mathbf{y})$` is a normalizing term that depends on `$\mathbf{y}$`, each `$\Phi_j(\mathbf{x}, y_j)$` is a function that couples the entire observable vector `$\mathbf{x}$` with a single hidden variable `$y_j$`, and `$\beta(\mathbf{x})$` is a function of only the observables.

**What this means:** the likelihood can depend on all hidden variables, but the interaction between the data and each hidden variable must be additive and separable — the total log-likelihood is a sum of terms, each involving a single hidden variable `$y_j$` interacting with all of `$\mathbf{x}$`, plus terms depending only on `$\mathbf{x}$`. There are no direct interaction terms between `$y_j$` and `$y_k$` in the likelihood. The normalization `$\Omega(\mathbf{y})$` is the only thing that can couple the hidden variables together.

**Corresponding complementary prior.** The prior must take the form:

$$P(\mathbf{y}) = \frac{1}{C} \exp\left(\log \Omega(\mathbf{y}) + \sum_j \alpha_j(y_j)\right)$$

where `$C$` is a normalization constant and `$\alpha_j(y_j)$` are arbitrary functions of individual hidden variables.

**Why this form:** when the likelihood (with normalization `$\Omega(\mathbf{y})$` in the denominator) is multiplied by the prior (with `$\log \Omega(\mathbf{y})$` in the exponent), the `$\Omega(\mathbf{y})$` terms cancel:

$$P(\mathbf{x}, \mathbf{y}) = P(\mathbf{x}|\mathbf{y})P(\mathbf{y})$$
$$= \frac{1}{\Omega(\mathbf{y})} e^{\sum_j \Phi_j(\mathbf{x}, y_j) + \beta(\mathbf{x})} \cdot \frac{1}{C} e^{\log \Omega(\mathbf{y}) + \sum_j \alpha_j(y_j)}$$
$$= \frac{1}{C} \exp\left(\sum_j [\Phi_j(\mathbf{x}, y_j) + \alpha_j(y_j)] + \beta(\mathbf{x})\right)$$

The joint distribution is now a product of factors, each involving only one `$y_j$` (since `$\Phi_j$` and `$\alpha_j$` each depend on a single `$y_j$`). Because the joint factorizes over `$j$`, the posterior `$P(\mathbf{y}|\mathbf{x})$` also factorizes — each `$y_j$` is conditionally independent of all other hidden variables given `$\mathbf{x}$`. The normalization term `$\Omega(\mathbf{y})$`, which was the source of the intractable coupling between hidden variables in the likelihood, has been exactly canceled by the complementary prior.

**General factorization result.** For the specific case where the likelihood also factorizes over visible variables — meaning `$P(\mathbf{x}|\mathbf{y}) = \prod_i P(x_i|\mathbf{y})$` — the joint distribution can be written as:

$$P(\mathbf{x}, \mathbf{y}) = \frac{1}{Z} \exp\left(\sum_{i,j} \Psi_{i,j}(x_i, y_j) + \sum_i \gamma_i(x_i) + \sum_j \alpha_j(y_j)\right)$$

This is the form of a **complete bipartite undirected graphical model** — an RBM — where every visible unit is connected to every hidden unit with no visible-visible or hidden-hidden connections. The next section shows how this connects to the infinite directed network.

---

#### The Infinite Directed Network with Tied Weights

The paper's central construction is an infinite logistic belief net where the weight matrices between consecutive layers are **tied** (i.e., constrained to be transposes of each other). This is illustrated in Figure 3.

The architecture, moving upward from the data:

- Layer `$\mathbf{v}^0$`: visible units (the data)
- Layer `$\mathbf{h}^0$`: first hidden layer, generated from `$\mathbf{v}^0$` using `$p(\mathbf{h}^0|\mathbf{v}^0)$` with weight matrix `$\mathbf{W}_0$`
- Layer `$\mathbf{v}^1$`: generated from `$\mathbf{h}^0$` using `$p(\mathbf{v}^1|\mathbf{h}^0)$` with weight matrix `$\mathbf{W}_0^T$`
- Layer `$\mathbf{h}^1$`: generated from `$\mathbf{v}^1$` using `$p(\mathbf{h}^1|\mathbf{v}^1)$` with weight matrix `$\mathbf{W}_0$`
- And so on, alternating between `$\mathbf{W}_0$` and `$\mathbf{W}_0^T$` ad infinitum

**Why the weights are transposes:** the key constraint is that the weight matrix going from `$\mathbf{v}^\ell$` to `$\mathbf{h}^\ell$` (upward/generative to hidden) is `$\mathbf{W}_0$`, and the weight matrix going from `$\mathbf{h}^\ell$` to `$\mathbf{v}^{\ell+1}$` (downward/hidden to next visible-like layer) is `$\mathbf{W}_0^T$`. This tied-transpose structure is what creates the complementary prior. The upward direction uses `$\mathbf{W}_0$` (the generative weights from the RBM perspective), and the downward direction uses `$\mathbf{W}_0^T$` (the recognition weights).

The remarkable property of this infinite network is that **inference is exact and trivial**. To sample from the true posterior `$P(\mathbf{h}^0|\mathbf{v}^0)$`, we simply:

1. Clamp `$\mathbf{v}^0$` to the data.
2. Compute the probability of each hidden unit in `$\mathbf{h}^0$` turning on as a logistic function of the input from `$\mathbf{v}^0$` through weights `$\mathbf{W}_0$`.
3. Sample binary states from these independent Bernoulli distributions.
4. This sample is an **unbiased draw from the true posterior** — no approximation is involved.

**Why this works.** The infinite stack with tied weights implements a complementary prior at every layer. Let us trace the inference procedure upward:

- Start: `$\mathbf{v}^0$` is observed.
- Sample `$\mathbf{h}^0 \sim P(\mathbf{h}^0|\mathbf{v}^0)$`. Because of the tied weights extending infinitely upward, the prior `$P(\mathbf{h}^0)$` in this infinite network is the complementary prior for the likelihood `$P(\mathbf{v}^0|\mathbf{h}^0)$`. Therefore `$P(\mathbf{h}^0|\mathbf{v}^0) \propto P(\mathbf{v}^0|\mathbf{h}^0)P(\mathbf{h}^0)$` is factorial — each `$h^0_j$` is independent given `$\mathbf{v}^0$`. Sampling from independent Bernoullis is correct.
- Now, having sampled `$\mathbf{h}^0$`, treat it as "data" for the next pair of layers. The same argument applies recursively: `$\mathbf{h}^0$` is analogous to `$\mathbf{v}^1$` in the next RBM upward, and `$\mathbf{v}^1$` is generated from `$\mathbf{h}^0$` using `$\mathbf{W}_0^T$`. So `$P(\mathbf{v}^1|\mathbf{h}^0)$` is the distribution of `$\mathbf{v}^1$` given `$\mathbf{h}^0$`, and the infinite stack above `$\mathbf{v}^1$` provides a complementary prior for it. Sampling `$\mathbf{h}^1 \sim P(\mathbf{h}^1|\mathbf{v}^1)$` is therefore also exact.
- This continues upward indefinitely.

**Formal demonstration via unrolling Gibbs sampling.** Appendix A provides a rigorous proof. Consider an undirected bipartite model (RBM) with joint distribution:

$$P(\mathbf{x}, \mathbf{y}) = \frac{1}{Z} \exp\left(\sum_{i,j} \Psi_{i,j}(x_i, y_j) + \sum_i \gamma_i(x_i) + \sum_j \alpha_j(y_j)\right)$$

Gibbs sampling in this model alternates between sampling `$\mathbf{y}|\mathbf{x}$` and `$\mathbf{x}|\mathbf{y}$`. If we "unroll" this Gibbs chain in space — treating each Gibbs update as a separate layer — we get an infinite sequence `$\mathbf{x}^{(0)}, \mathbf{y}^{(0)}, \mathbf{x}^{(1)}, \mathbf{y}^{(1)}, \ldots$` where:

$$P(\mathbf{x}^{(\ell)}|\mathbf{y}^{(\ell-1)}) = g_x(\mathbf{x}^{(\ell)}|\mathbf{y}^{(\ell-1)}) \quad \text{(factorial in } \mathbf{x})$$
$$P(\mathbf{y}^{(\ell)}|\mathbf{x}^{(\ell)}) = g_y(\mathbf{y}^{(\ell)}|\mathbf{x}^{(\ell)}) \quad \text{(factorial in } \mathbf{y})$$

By induction (equations A.13–A.20), the authors show that the marginal distribution `$P(\mathbf{x}^{(0)})$` under this infinite directed construction is exactly the same as the marginal under the original undirected RBM, and that the conditional distributions going "downward" (in the generative direction from `$\mathbf{y}^{(\ell)}$` to `$\mathbf{x}^{(\ell)}$`) are correct. This proves that inference going upward (computing `$P(\mathbf{y}^{(0)}|\mathbf{x}^{(0)})$`, `$P(\mathbf{x}^{(1)}|\mathbf{y}^{(0)})$`, etc.) via factorial sampling gives unbiased samples from the true posterior of the infinite directed model, which is equivalent to the undirected RBM.

---

#### Equivalence to Restricted Boltzmann Machines

The equivalence between the infinite directed net with tied weights and a restricted Boltzmann machine (RBM) is the bridge that enables practical learning. An RBM is an undirected graphical model with:

- A layer of visible units `$\mathbf{v}$`
- A layer of hidden units `$\mathbf{h}$`
- Symmetric weights `$w_{ij}$` connecting each visible unit `$i$` to each hidden unit `$j$`
- **No** connections within the visible layer or within the hidden layer (hence "restricted")

The energy of a joint configuration `$(\mathbf{v}, \mathbf{h})$` is:

$$E(\mathbf{v}, \mathbf{h}) = -\sum_i b_i v_i - \sum_j c_j h_j - \sum_{i,j} v_i h_j w_{ij}$$

where `$b_i$` are visible biases, `$c_j$` are hidden biases, and `$w_{ij}$` are the symmetric connection weights.

The probability of a joint configuration under the Boltzmann distribution is:

$$P(\mathbf{v}, \mathbf{h}) = \frac{1}{Z} \exp(-E(\mathbf{v}, \mathbf{h}))$$

where `$Z = \sum_{\mathbf{v},\mathbf{h}} \exp(-E(\mathbf{v}, \mathbf{h}))$` is the partition function (sum over all possible configurations).

**What the energy function represents:** lower energy means higher probability. The visible biases make certain pixel patterns more likely regardless of hidden state; the hidden biases make certain feature detectors more likely to be active regardless of the image; the weighted interaction term `$-v_i h_j w_{ij}$` makes configurations where `$v_i$` and `$h_j$` are both active more probable when `$w_{ij}$` is positive (they "agree"), and less probable when `$w_{ij}$` is negative (they "disagree"). This is the classic formulation of a Hopfield-like energy that defines a probability via the Boltzmann distribution.

**Conditional distributions are factorial.** Because there are no hidden-hidden or visible-visible connections, the conditional distribution of hidden units given visible units factorizes:

$$P(\mathbf{h}|\mathbf{v}) = \prod_j P(h_j|\mathbf{v})$$
$$P(h_j = 1|\mathbf{v}) = \frac{1}{1 + \exp\left(-c_j - \sum_i v_i w_{ij}\right)}$$

Similarly, the conditional distribution of visible units given hidden units factorizes:

$$P(\mathbf{v}|\mathbf{h}) = \prod_i P(v_i|\mathbf{h})$$
$$P(v_i = 1|\mathbf{h}) = \frac{1}{1 + \exp\left(-b_i - \sum_j h_j w_{ij}\right)}$$

These are exactly logistic functions of the form in equation 2.1, with the bias `$b_i$` or `$c_j$` playing the role of `$b_i$` in the directed model and the weight sums playing the role of `$\sum_j s_j w_{ij}$`.

**How Gibbs sampling works in an RBM.** To generate samples from an RBM, we alternate between sampling `$\mathbf{h}$` given `$\mathbf{v}$` and sampling `$\mathbf{v}$` given `$\mathbf{h}$`:

1. Start with a random visible vector `$\mathbf{v}^{(0)}$`.
2. Sample `$\mathbf{h}^{(0)} \sim P(\mathbf{h}|\mathbf{v}^{(0)})$`: for each hidden unit `$j$`, compute its activation probability as `$\sigma(c_j + \sum_i v_i^{(0)} w_{ij})$` and draw a binary sample. This is parallel because the hidden units are conditionally independent.
3. Sample `$\mathbf{v}^{(1)} \sim P(\mathbf{v}|\mathbf{h}^{(0)})$`: for each visible unit `$i$`, compute its activation probability as `$\sigma(b_i + \sum_j h_j^{(0)} w_{ij})$` and draw a binary sample.
4. Sample `$\mathbf{h}^{(1)} \sim P(\mathbf{h}|\mathbf{v}^{(1)})$`, and so on.

This alternating Gibbs sampling is **exactly the same process** as the inference/generation alternation in the infinite directed net with tied weights:
- Sampling `$\mathbf{h}$` from `$\mathbf{v}$` in the RBM corresponds to one step of inference upward in the directed net (using weight matrix `$\mathbf{W}$`).
- Sampling `$\mathbf{v}$` from `$\mathbf{h}$` in the RBM corresponds to one step of generation downward in the directed net (using weight matrix `$\mathbf{W}^T$`).

The stationary distribution of this Gibbs chain is the model distribution `$P(\mathbf{v}, \mathbf{h})$` defined by the energy. After sufficiently many iterations, the samples are unbiased draws from the RBM's equilibrium distribution.

**The learning gradient for an RBM.** For maximum likelihood learning in an RBM, the derivative of the log probability of a data vector `$\mathbf{v}^0$` with respect to a weight `$w_{ij}$` is:

$$\frac{\partial \log P(\mathbf{v}^0)}{\partial w_{ij}} = \langle v_i^0 h_j^0 \rangle - \langle v_i^\infty h_j^\infty \rangle$$

where `$\langle v_i^0 h_j^0 \rangle$` is the expected product of unit `$i$` and unit `$j$` when the visible vector is clamped to the data `$\mathbf{v}^0$` and hidden states are sampled from `$P(\mathbf{h}|\mathbf{v}^0)$`, and `$\langle v_i^\infty h_j^\infty \rangle$` is the expected product under the model's equilibrium distribution (after running Gibbs sampling to convergence starting from any state).

**What this gradient computes:** the first term (`$\langle v_i^0 h_j^0 \rangle$`) is the "data-driven" correlation — it measures how often visible unit `$i$` and hidden unit `$j$` fire together when the network sees real data. The second term (`$\langle v_i^\infty h_j^\infty \rangle$`) is the "model-driven" correlation — how often they fire together under the network's own generative distribution, independent of data. The difference tells us how to adjust `$w_{ij}$`: increase the weight if the pair co-activates more under data than under the model (making the model assign higher probability to data-like configurations), decrease it otherwise.

**Why this form:** this is derived from the log-likelihood gradient of the Boltzmann distribution. The gradient is the difference between the sufficient statistics (pairwise products) under the data distribution and under the model distribution. This difference-of-correlations form is the standard learning rule for Boltzmann machines and is the foundation for both maximum likelihood and contrastive divergence learning.

**Equivalence of gradients.** As shown in equation 2.4, the gradient for the infinite directed net with tied weights is:

$$\frac{\partial \log P(\mathbf{v}^0)}{\partial w_{ij}} = \langle h_j^0 (v_i^0 - v_i^1) \rangle + \langle v_i^1 (h_j^0 - h_j^1) \rangle + \langle h_j^1 (v_i^1 - v_i^2) \rangle + \cdots$$

All intermediate terms cancel telescopically, leaving:

$$\frac{\partial \log P(\mathbf{v}^0)}{\partial w_{ij}} = \langle v_i^0 h_j^0 \rangle - \langle v_i^\infty h_j^\infty \rangle$$

which is exactly the RBM learning rule of equation 3.1. The infinite network and the RBM are equivalent in both their generative process and their learning gradient.

---

#### Contrastive Divergence Learning

Exact maximum likelihood learning in an RBM requires running the Gibbs chain to equilibrium to compute `$\langle v_i^\infty h_j^\infty \rangle$`, which is computationally expensive. The paper uses **contrastive divergence** (CD) learning (Hinton, 2002), which approximates the equilibrium expectation by running the Gibbs chain for only `$n$` full steps (where `$n$` is small — typically `$n=1$`).

The CD-`$n$` learning rule replaces the equilibrium expectation with an expectation after `$n$` steps of Gibbs sampling starting from the data:

$$\Delta w_{ij} \propto \langle v_i^0 h_j^0 \rangle - \langle v_i^n h_j^n \rangle$$

**What it computes:** the learning signal is still a difference of correlations, but the "negative phase" correlation is computed after only `$n$` steps of Gibbs sampling rather than at equilibrium. A full step consists of updating `$\mathbf{h}$` given `$\mathbf{v}$`, then updating `$\mathbf{v}$` given `$\mathbf{h}$`. After `$n$` steps, we measure how often `$v_i^n$` and `$h_j^n$` co-activate. The weight is then incremented by `$\epsilon(\langle v_i^0 h_j^0 \rangle - \langle v_i^n h_j^n \rangle)$`, where `$\epsilon$` is a learning rate.

**Why this works: contrastive divergence minimizes a difference of KL divergences.** The objective being minimized is:

$$KL(P^0 \| P_\theta^\infty) - KL(P_\theta^n \| P_\theta^\infty)$$

where `$P^0$` is the empirical data distribution, `$P_\theta^\infty$` is the model's equilibrium distribution, and `$P_\theta^n$` is the distribution after `$n$` steps of Gibbs sampling starting from `$P^0$`. The first term `$KL(P^0 \| P_\theta^\infty)$` is what maximum likelihood minimizes. The second term is the KL divergence between the `$n$`-step distribution and equilibrium.

**Why this difference is non-negative and informative.** Gibbs sampling starting from the data distribution always moves the distribution closer to the model's equilibrium (it reduces KL divergence with `$P_\theta^\infty$`). Therefore `$KL(P^0 \| P_\theta^\infty) \geq KL(P_\theta^n \| P_\theta^\infty)$`, and their difference is never negative (ignoring sampling noise). Minimizing this difference has the effect of pulling the equilibrium distribution toward the data while simultaneously pushing the `$n$`-step reconstruction away from equilibrium — it makes the model's generative process closer to the data in `$n$` steps.

**What is being ignored.** The term `$KL(P_\theta^n \| P_\theta^\infty)$` depends on the parameters `$\theta$` both through `$P_\theta^n$` (because `$P_\theta^n$` is produced by `$n$` steps of a Gibbs chain that depends on `$\theta$`) and through `$P_\theta^\infty$`. The derivative of `$KL(P_\theta^n \| P_\theta^\infty)$` with respect to `$\theta$` includes terms accounting for how `$P_\theta^n$` changes as `$\theta$` changes. CD learning **ignores these terms** — it treats `$P_\theta^n$` as fixed when computing the gradient of the subtracted KL divergence. This is the approximation: we assume the change in `$P_\theta^n$` with respect to `$\theta$` is dominated by the change in `$P_\theta^\infty$`.

**Practical CD-1 procedure.** Throughout the paper, the greedy pretraining uses CD with a very small number of steps (implicitly $n=1$ based on the "contrastive divergence learning" description in Section 3). The procedure for training one RBM on a batch of data is:

1. Clamp the visible units to a data vector (or batch of data vectors).
2. Compute hidden unit probabilities `$P(h_j = 1|\mathbf{v}) = \sigma(c_j + \sum_i v_i w_{ij})$` and sample binary hidden states `$h_j^0$`.
3. Compute the positive phase correlation: `$\langle v_i^0 h_j^0 \rangle$` (outer product of data and sampled hidden states).
4. Compute visible unit reconstruction probabilities `$P(v_i = 1|\mathbf{h}^0) = \sigma(b_i + \sum_j h_j^0 w_{ij})$` and sample binary visible states `$v_i^1$`.
5. Compute hidden unit probabilities given the reconstructions `$P(h_j = 1|\mathbf{v}^1)$` and sample `$h_j^1$` (for CD-1; for CD-$n$, repeat steps 4–5 `$n$` times).
6. Compute the negative phase correlation: `$\langle v_i^1 h_j^1 \rangle$`.
7. Update weights: `$\Delta w_{ij} = \epsilon(\langle v_i^0 h_j^0 \rangle - \langle v_i^1 h_j^1 \rangle)$`.
8. Update biases: `$\Delta b_i = \epsilon(v_i^0 - v_i^1)$`, `$\Delta c_j = \epsilon(h_j^0 - h_j^1)$`.

**Real-valued visible units.** During the greedy pretraining, the visible units of each RBM are **not strictly binary**. For the bottom RBM (pixels), the "visible" units have real-valued activities between 0 and 1 — these are the normalized pixel intensities, treated as probabilities. For higher-layer RBMs, the visible units are the activation probabilities of the hidden units from the lower RBM. The hidden layer of each RBM uses stochastic binary states during training. This use of real-valued probabilities in the visible layer is a standard extension of the RBM to handle continuous-valued (or rather, [0,1]-bounded) inputs, treating them as the parameters of independent Bernoulli distributions.

---

#### The Greedy Layer-by-Layer Pretraining Algorithm

With the RBM-to-infinite-net equivalence and CD learning established, the greedy pretraining algorithm (Section 4) learns a deep network one layer at a time, starting from the bottom (closest to the data) and moving upward.

**Step 1: Learn the bottom-layer weights.** Assume that all weight matrices in the infinite directed network are tied together — that is, `$\mathbf{W}_0 = \mathbf{W}_1 = \mathbf{W}_2 = \cdots$`. Under this assumption, learning the bottom weight matrix `$\mathbf{W}_0$` is equivalent to training an RBM on the visible data using CD. This RBM has:
- Visible units: 784 pixel intensities (plus 10 label units, treated as additional visible units in the top RBM; during bottom-layer training, the labels are not yet incorporated — the bottom RBM learns purely from images)
- Hidden units: 500 binary units
- Weights `$\mathbf{W}_0$` (784×500) connecting every visible unit to every hidden unit

The RBM is trained for 30 epochs (full passes through the training set) using CD learning. After training, `$\mathbf{W}_0$` contains features that capture correlations in the pixel data.

**Why the tied-weights assumption is valid for this phase.** During step 1, we assume higher layers exist with weights tied to `$\mathbf{W}_0$`. This assumption provides a complementary prior that makes the posterior over the first hidden layer exactly factorial, so that applying `$\mathbf{W}_0^T$` to the data and sampling independently per hidden unit is exact inference — not an approximation. Even if the actual higher-layer weights will later differ from `$\mathbf{W}_0$`, the assumption is what makes the RBM learning objective the correct objective for the bottom layer under the full infinite model.

**Step 2: Freeze `$\mathbf{W}_0$` and transform the data.** Commit to using `$\mathbf{W}_0^T$` to infer the states of the first hidden layer. Specifically, for each training image `$\mathbf{v}$`:
- Compute activation probabilities: `$\mathbf{p} = \sigma(\mathbf{c} + \mathbf{v}\mathbf{W}_0)$` (where `$\mathbf{c}$` are the hidden biases learned in step 1).
- These probabilities become the "data" for the next RBM — they are the representation of the original image in terms of the learned features.

Freeze `$\mathbf{W}_0$` and `$\mathbf{c}$` — they will not be changed during subsequent greedy learning steps (though they may be updated later during fine-tuning).

**Why freeze `$\mathbf{W}_0$`.** The variational argument in equations 4.1–4.3 shows that after step 1, the bound on `$\log P(\mathbf{v}^0)$` is tight (because the factorial posterior was exact under the tied-weights assumption). Once `$Q(\mathbf{h}^0|\mathbf{v}^0)$` (the inference distribution) and `$P(\mathbf{v}^0|\mathbf{h}^0)$` (the generative distribution from `$\mathbf{h}^0$` to `$\mathbf{v}^0$`) are frozen, maximizing the bound with respect to higher-level weights reduces to maximizing the expected log-likelihood of the inferred `$\mathbf{h}^0$` configurations under the higher-level model. Freezing the bottom weights ensures that the greedy procedure never decreases a lower bound on the data log-probability.

**Step 3: Learn the next RBM on the transformed data.** Train a second RBM where:
- The "visible" units are the 500-dimensional activation probability vectors from step 2.
- The hidden units are another 500 binary units.
- All weight matrices from this layer upward are assumed to be tied (`$\mathbf{W}_1 = \mathbf{W}_2 = \cdots$`).

Train this RBM for 30 epochs using CD, learning weights `$\mathbf{W}_1$` (500×500). After training, `$\mathbf{W}_1^T$` maps the inferred `$\mathbf{h}^0$` representations to `$\mathbf{v}^1$` representations, and the features in `$\mathbf{W}_1$` capture higher-order correlations among the first-layer features.

**Step 4: Recursively apply the transformation.** Freeze `$\mathbf{W}_1$` and transform the data again: each training case's `$\mathbf{h}^0$` probabilities are mapped through `$\mathbf{W}_1^T$` to produce 500-dimensional "data" for the next level.

**Step 5: Learn the top-level associative memory.** The top two layers form the undirected associative memory (Figure 1). This is an RBM with:
- 500 units in the lower layer (the penultimate hidden layer, analogous to `$\mathbf{h}^0$` in the RBM notation)
- 2000 units in the top layer (the top hidden layer)
- **Including the labels**: the 10 label units are added as additional visible units in this top RBM, forming a softmax group.

At this stage, the labels are provided as part of the input during training. When the label units are reconstructed from the top layer, exactly one label unit is allowed to be active, with the probability of picking label `$i$` given by:

$$p_i = \frac{\exp(x_i)}{\sum_j \exp(x_j)}$$

where `$x_i$` is the total input received by label unit `$i$` from the 2000 top-level units. This is the standard softmax function — it produces a proper probability distribution over the 10 digit classes by normalizing the exponentials of the inputs.

**Why the softmax does not affect learning.** The paper makes a surprising observation: "the learning rules are unaffected by the competition between units in a softmax group." The competition among label units changes only the probability of each label turning on, but the weight update rules depend only on these probabilities, not on the fact that they were produced by a competitive softmax rather than independent logistics. The synapses do not need to know which label units are competing — they only see the final activity probabilities.

**Variational guarantee for greedy learning.** The paper provides a theoretical guarantee (Section 4) that the greedy procedure never decreases a lower bound on the log probability of the data under the full generative model, assuming maximum likelihood learning is used for each RBM. The derivation uses the variational free energy bound:

$$\log P(\mathbf{v}^0) \geq \sum_{\mathbf{h}^0} Q(\mathbf{h}^0|\mathbf{v}^0) \left[\log P(\mathbf{h}^0) + \log P(\mathbf{v}^0|\mathbf{h}^0)\right] - \sum_{\mathbf{h}^0} Q(\mathbf{h}^0|\mathbf{v}^0) \log Q(\mathbf{h}^0|\mathbf{v}^0)$$

where `$Q(\mathbf{h}^0|\mathbf{v}^0)$` is the factorial distribution produced by applying `$\mathbf{W}_0^T$`, `$P(\mathbf{v}^0|\mathbf{h}^0)$` is the generative distribution defined by `$\mathbf{W}_0$`, and `$P(\mathbf{h}^0)$` is the prior over the first hidden layer defined by the higher-layer weights.

When all weights are tied (step 1), `$Q(\mathbf{h}^0|\mathbf{v}^0)$` is the true posterior, so the bound is tight: `$\log P(\mathbf{v}^0) = \text{bound}$`. After `$\mathbf{W}_0$` is frozen (step 2), `$Q(\mathbf{h}^0|\mathbf{v}^0)$` and `$P(\mathbf{v}^0|\mathbf{h}^0)$` are fixed. The bound's derivative with respect to higher-layer weights is exactly the derivative of `$\sum_{\mathbf{h}^0} Q(\mathbf{h}^0|\mathbf{v}^0) \log P(\mathbf{h}^0)$` — the expected log-likelihood of the inferred `$\mathbf{h}^0$` under the higher-level model. Maximizing this is exactly what the next RBM does. Therefore:

> "if we use the full maximum likelihood Boltzmann machine learning algorithm to learn each set of tied weights and then we untie the bottom layer of the set from the weights above, we can learn the weights one layer at a time with a guarantee that we will never decrease the bound on the log probability of the data under the model"

**The CD approximation voids the strict guarantee.** The paper acknowledges that because CD learning (rather than exact maximum likelihood) is used for efficiency, the guarantee is no longer rigorous. However, CD is a good enough approximation that the procedure works well in practice. The properties of CD — particularly that it moves the model distribution toward the data distribution — mean that the greedy procedure still tends to improve the model.

**Analogy to boosting and representation learning.** The paper explicitly frames the greedy algorithm as analogous to boosting (Freund, 1995) and projection pursuit (Friedman & Stuetzle, 1981), but with a crucial difference:

> "instead of reweighting each data vector to ensure that the next step learns something new, it re-represents it"

In boosting, each subsequent weak learner focuses on examples that previous learners got wrong by reweighting the data. In the greedy RBM algorithm, each subsequent layer sees a different representation of the data — the activation patterns of the previous layer's features — which naturally emphasizes different statistical structure. The first layer captures local pixel correlations, the second captures correlations among those local features, and the top layer captures abstract structure that relates features to digit classes.

---

#### The Up-Down Fine-Tuning Algorithm

After greedy pretraining, the network has good initial weights in every layer, but they are not jointly optimal. The greedy procedure learned each layer in isolation (or with the tied-weights assumption), and once higher layers are trained independently, the bottom-up inference using `$\mathbf{W}_0^T$`, `$\mathbf{W}_1^T$`, etc. is no longer exact — the true posterior is no longer factorial because the higher-layer weights no longer implement a complementary prior.

The **up-down algorithm** (Section 5) is a contrastive version of the wake-sleep algorithm (Hinton et al., 1995) that fine-tunes all weights jointly. It has three phases per training iteration.

**Untying recognition from generative weights.** Before fine-tuning, the recognition weights (used for bottom-up inference) are untied from the generative weights. Initially, the recognition weight from layer `$\ell$` to layer `$\ell+1$` is `$\mathbf{W}_\ell^T$` (the transpose of the generative weight `$\mathbf{W}_\ell$` from layer `$\ell+1$` to layer `$\ell$`). After untying, the recognition weights become independent parameters that can be learned separately, though they retain the constraint that the posterior at each layer is approximated by a factorial distribution.

**Phase 1: The Up-Pass (Wake Phase).** This is a bottom-up stochastic pass driven by the data:

1. Clamp the visible units to a training image.
2. Compute hidden unit activation probabilities for the first hidden layer using the (untied) recognition weights `$\mathbf{R}_0$`: `$\mathbf{p}^1 = \sigma(\mathbf{b}^1_{\text{rec}} + \mathbf{v}\mathbf{R}_0)$`, and sample binary states `$\mathbf{h}^1$` from these probabilities.
3. Compute activation probabilities for the second hidden layer using recognition weights `$\mathbf{R}_1$`: `$\mathbf{p}^2 = \sigma(\mathbf{b}^2_{\text{rec}} + \mathbf{h}^1\mathbf{R}_1)$`, and sample binary states `$\mathbf{h}^2$`.
4. Compute activation probabilities for the top layer (in the associative memory) using recognition weights from the penultimate layer, plus label input: `$\mathbf{p}^{\text{top}} = \sigma(\mathbf{b}^{\text{top}} + \mathbf{h}^2\mathbf{R}_2 + \text{labels} \cdot \mathbf{W}_{\text{lab}\rightarrow\text{top}})$`, and sample binary top-level states.
5. For each directed generative connection (top-down), compute the positive phase statistics: the outer product of the states in the layer above (pre-synaptic) and the reconstruction probabilities in the layer below (post-synaptic). The generative weights are updated using the maximum likelihood rule of equation 2.2:

$$\Delta W^{\text{gen}}_{\ell, ij} = \langle h^{\ell+1}_j (v^\ell_i - \hat{v}^\ell_i) \rangle$$

where `$v^\ell_i$` is the actual state of unit `$i$` in layer `$\ell$` from the up-pass, `$\hat{v}^\ell_i$` is the probability that unit `$i$` would be turned on if reconstructed from the layer above using the generative weights, and `$h^{\ell+1}_j$` is the state of unit `$j$` in layer `$\ell+1$`.

**Why the reconstruction probability is used.** Unlike during greedy pretraining where the reconstruction was computed using the tied transpose weights, the up-pass now uses the frozen generative weights to compute `$\hat{v}^\ell_i$`. This is because the generative and recognition weights are no longer tied — the generative model is defined by the generative weights, and the learning signal should reflect how well those generative weights can reconstruct the lower layer from the upper layer.

6. The top-level undirected weights (the associative memory) are updated using CD learning, exactly as in an RBM, with the states from step 4 serving as the positive phase. Several iterations of alternating Gibbs sampling (typically 3, 6, or 10, increasing over epochs) are run to get negative phase statistics (see Phase 2).

**Why use Gibbs sampling at the top.** The top two layers form an undirected associative memory. During the up-pass, the top layer state was inferred from below. To get a proper negative phase for CD learning of the top-level weights, we need to run the chain away from the data-driven initialization. The number of Gibbs steps controls how thoroughly the chain mixes — more steps give a better approximation to the equilibrium distribution.

**Phase 2: Gibbs Sampling in the Associative Memory (Transition).** Between the up-pass and down-pass, the top-level associative memory runs for `$K$` iterations of alternating Gibbs sampling:

1. Starting from the up-pass top-level state `$\mathbf{h}^{\text{top},0}$` and penultimate state `$\mathbf{h}^2$`.
2. For `$k = 1$` to `$K$`:
   - Update penultimate layer probabilities given top layer: `$\mathbf{p}^2_{\text{new}} = \sigma(\mathbf{b}^2_{\text{gen}} + \mathbf{h}^{\text{top},k-1}\mathbf{W}_{\text{pen}\rightarrow\text{top}}^T)$`, sample binary states.
   - Update label probabilities given top layer (using softmax): `$\mathbf{p}^{\text{lab}} = \text{softmax}(\mathbf{b}^{\text{lab}} + \mathbf{h}^{\text{top},k-1}\mathbf{W}_{\text{lab}\rightarrow\text{top}}^T)$`.
   - Update top layer probabilities given penultimate and labels: `$\mathbf{p}^{\text{top},k} = \sigma(\mathbf{b}^{\text{top}} + \mathbf{h}^{2,k}\mathbf{W}_{\text{pen}\rightarrow\text{top}} + \mathbf{p}^{\text{lab}}\mathbf{W}_{\text{lab}\rightarrow\text{top}})$`, sample binary states.
3. The final penultimate state `$\mathbf{h}^{2,K}$` and top state `$\mathbf{h}^{\text{top},K}$` are the negative phase samples for CD learning of the top-level weights.
4. The negative phase statistics (outer products of penultimate and top states, and labels and top states) are collected and used to update the top-level weights via the standard CD update rule.

**Why use the data-driven up-pass as initialization for Gibbs sampling rather than starting from equilibrium.** This is the "contrastive" aspect of the algorithm. In the original wake-sleep algorithm's sleep phase, the top-level units were sampled from their independent priors and then an ancestral pass generated downward. But the independent prior is a poor model for the top layer. By initializing the Gibbs chain from the data-driven up-pass, the associative memory starts in a state that is relevant to the current data. Running a few Gibbs steps then moves it away from the data but not all the way to equilibrium, creating a contrastive signal. This:

1. Ensures recognition weights are learned for representations that actually occur when processing real data.
2. Helps eliminate the mode-averaging problem: if the recognition weights tend to pick one particular mode of the posterior (ignoring other equally good modes), the Gibbs sampling near that mode will not encourage the recognition weights to spread out to cover the other modes (as would happen with a pure ancestral sleep phase starting from independent priors).

**Phase 3: The Down-Pass (Sleep Phase).** Starting from the state of the associative memory after Gibbs sampling (or after the up-pass, depending on the variant — the pseudocode in Appendix B starts from the end of the Gibbs run):

1. The penultimate layer state `$\mathbf{h}^2$` is the final state from the Gibbs sampling.
2. Using generative weights, compute activation probabilities for the first hidden layer: `$\mathbf{p}^1_{\text{gen}} = \sigma(\mathbf{b}^1_{\text{gen}} + \mathbf{h}^2\mathbf{W}_1)$`, and sample binary states `$\mathbf{h}^1_{\text{gen}}$`.
3. Using generative weights, compute activation probabilities for the visible layer: `$\mathbf{p}^0_{\text{gen}} = \sigma(\mathbf{b}^0_{\text{gen}} + \mathbf{h}^1_{\text{gen}}\mathbf{W}_0)$`. For pixel units, these are probabilities; for label units, a softmax is used.
4. Now, using the generated states, compute what the recognition weights **would predict** in the upward direction:
   - Predict the first hidden layer from the generated visible states: `$\hat{\mathbf{h}}^1 = \sigma(\mathbf{b}^1_{\text{rec}} + \mathbf{p}^0_{\text{gen}}\mathbf{R}_0)$`.
   - Predict the second hidden layer from the generated first hidden states: `$\hat{\mathbf{h}}^2 = \sigma(\mathbf{b}^2_{\text{rec}} + \mathbf{h}^1_{\text{gen}}\mathbf{R}_1)$`.
5. Update the recognition weights to minimize the difference between the actual generated states and what the recognition model predicted:

$$\Delta R_{0,ij} = \langle v^{\text{gen}}_i (h^1_{\text{gen},j} - \hat{h}^1_j) \rangle$$
$$\Delta R_{1,ij} = \langle h^1_{\text{gen},i} (h^2_{\text{gen},j} - \hat{h}^2_j) \rangle$$

**What this does.** The down-pass trains the recognition weights to correctly infer hidden states from visible states, but using data generated by the model's own generative process rather than real data. This ensures that the recognition model learns to invert the generative model everywhere the generative model has probability mass, not just near the training data.

**Why term this "contrastive wake-sleep."** In the original wake-sleep algorithm:
- The wake phase is the same: data-driven up-pass, update generative weights.
- The sleep phase starts from independent top-level priors, generates downward (ancestral pass), and updates recognition weights to recover the generated hidden states from the generated visible states.

The paper's version is "contrastive" because:
1. The sleep phase starts from the data-driven state (after Gibbs sampling) rather than independent priors, making it more relevant to the data distribution.
2. The top-level weights are learned with CD rather than maximum likelihood.

**Parameter updates from the pseudocode (Appendix B).** The complete set of weight updates per training case (shown for batch size 1; in practice, averaged over minibatches of 100 images: 10 per digit class) is:

*Generative weights (updated during up-pass):*
- `$\text{hidvis} \mathrel{+}= r \cdot \text{hidstates}^T \cdot (\text{data} - \text{pvisprobs})$`
- `$\text{visgenbiases} \mathrel{+}= r \cdot (\text{data} - \text{pvisprobs})$`
- `$\text{penhid} \mathrel{+}= r \cdot \text{penstates}^T \cdot (\text{hidstates} - \text{phidprobs})$`
- `$\text{hidgenbiases} \mathrel{+}= r \cdot (\text{hidstates} - \text{phidprobs})$`

*Top-level associative memory weights (updated with CD):*
- `$\text{labtop} \mathrel{+}= r \cdot (\text{poslabtopstatistics} - \text{neglabtopstatistics})$`
- `$\text{labgenbiases} \mathrel{+}= r \cdot (\text{targets} - \text{neglabprobs})$`
- `$\text{pentop} \mathrel{+}= r \cdot (\text{pospentopstatistics} - \text{negpentopstatistics})$`
- `$\text{pengenbiases} \mathrel{+}= r \cdot (\text{wakepenstates} - \text{negpenstates})$`
- `$\text{topbiases} \mathrel{+}= r \cdot (\text{waketopstates} - \text{negtopstates})$`

*Recognition weights (updated during down-pass):*
- `$\text{hidpen} \mathrel{+}= r \cdot \text{sleephidstates}^T \cdot (\text{sleeppenstates} - \text{psleeppenstates})$`
- `$\text{penrecbiases} \mathrel{+}= r \cdot (\text{sleeppenstates} - \text{psleeppenstates})$`
- `$\text{vishid} \mathrel{+}= r \cdot \text{sleepvisprobs}^T \cdot (\text{sleephidstates} - \text{psleephidstates})$`
- `$\text{hidrecbiases} \mathrel{+}= r \cdot (\text{sleephidstates} - \text{psleephidstates})$`

where `$r$` is a global learning rate (the same for all layers).

---

#### Complete Training Protocol and Hyperparameters

The paper provides a detailed account of the entire training pipeline for the MNIST experiments.

**Architecture (Figure 1):**
- Input: 784 pixel units (28×28) with real-valued intensities normalized to [0,1], plus 10 label units (one-hot encoding)
- Hidden layer 1: 500 units
- Hidden layer 2: 500 units
- Top layers (associative memory): 500 units (penultimate) ↔ 2000 units (topmost), plus the 10 softmax label units connected to the 2000 topmost units
- Total parameters: approximately 1.7 million weights
- Connections: full connectivity between adjacent layers, no intra-layer connections

**Greedy pretraining details:**
- **Training set:** 44,000 images from the MNIST training set (60,000 available), divided into 440 balanced mini-batches of 100 images each (10 of each digit class per mini-batch).
- **Weights updated:** after each mini-batch.
- **Each RBM trained for:** 30 epochs (full sweeps through the 44,000-image training set).
- **Visible units in RBMs:** real-valued activities between 0 and 1. For the bottom RBM, these are normalized pixel intensities. For higher RBMs, these are the activation probabilities of the hidden units from the lower RBM.
- **Hidden units in RBMs:** stochastic binary states during training.
- **Training time:** "a few hours per layer in MATLAB on a 3 GHz Xeon processor."
- **After greedy pretraining:** error rate on test set is 2.49% (using deterministic up-pass for classification).

**Fine-tuning details (up-down algorithm):**
- **Training set:** same 44,000 images in 440 balanced mini-batches of 100.
- **Total epochs:** 300 epochs of up-down fine-tuning.
- **Learning rate, momentum, weight decay:** chosen by training the network several times and observing performance on a separate validation set of 10,000 images (taken from the remainder of the full 60,000-image training set). Exact values are not specified beyond being "conservative to avoid oscillations."
- **Gibbs steps in associative memory:** 3 full iterations of alternating Gibbs sampling for the first 100 epochs, 6 iterations for the second 100 epochs, 10 iterations for the last 100 epochs.
- **Effect of increasing Gibbs steps:** "Each time the number of iterations of Gibbs sampling was raised, the error on the validation set decreased noticeably."

**Final training on full dataset:**
- The network that performed best on the validation set had an error rate of 1.39% on the test set.
- This network was then trained on all 60,000 training images until its error rate on the full training set matched its previous final error rate on the 44,000-image subset. This took a further 59 epochs.
- Total learning time: "about a week" (including both greedy pretraining and up-down fine-tuning).
- Final test error rate: 1.25%.

**Testing procedures:**
Three methods were evaluated:

1. **Stochastic up-pass + Gibbs sampling for labels:** Fix the binary states of the 500 units in the lower layer of the associative memory via a stochastic up-pass, then run a few iterations of alternating Gibbs sampling in the associative memory with the label units initialized to 0.1 and allowed to compete via softmax. This method gave error rates "almost 1% higher" than the reported 1.25%.

2. **Exact free-energy computation:** Fix the binary states of the 500 penultimate units via an up-pass. For each of the 10 possible label configurations (one label on, others off), compute the exact free energy of the resulting 510-component binary vector (500 penultimate + 10 labels) in the top-level RBM. The label with the lowest free energy is selected. Because "almost all the computation required is independent of which label unit is turned on" (Teh & Hinton, 2001), this is computationally efficient. However, this gave error rates "about 0.5% higher" than reported because of noise from the stochastic up-pass.

3. **Deterministic up-pass + exact free energy (reported method):** Use the recognition weights to compute activation probabilities during the up-pass (without sampling), making the up-pass deterministic. Then compute the exact free energy for each label as in method 2. Pick the label with the lowest free energy. The paper also validated that averaging the label probabilities or log-probabilities over 20 stochastic up-passes gave "almost identical results" and were "very similar to using a single deterministic up-pass."

**Further training test:** To check if the 1.25% result could be improved with more training, the network was left running with a very small learning rate. After six additional weeks, the test error fluctuated between 1.12% and 1.31%, and was 1.18% for the epoch with the smallest number of training errors. This confirms that the 1.25% result is near the asymptote of what this architecture can achieve, and that the up-down algorithm had essentially converged.

**Design choice: balanced mini-batches.** The training set (either 44,000 or 60,000 images) was divided into mini-batches of 100 images, each containing exactly 10 examples of each digit class. This balanced composition ensures that each weight update sees equal representation of all classes, preventing the model from developing biases toward frequently occurring digits. For the 60,000-image final training, the "unequal numbers of each class" in the raw MNIST training set were addressed by "randomly assigning images to each of the 600 mini-batches," which approximately achieves balance.

**Design choice: increasing Gibbs steps during fine-tuning.** Starting with 3 Gibbs steps and increasing to 10 over the course of 300 epochs is a curriculum strategy. Early in fine-tuning, the weights are close to their greedy initialization and the model is rough — a few Gibbs steps provide enough contrastive signal. As the model improves, more Gibbs steps are needed to get good negative phase samples because the chain mixes more slowly near a good model's equilibrium. The observation that validation error drops when Gibbs steps are increased validates this strategy.

**Design choice: separate validation set.** Holding out 10,000 images from the 60,000 training set (while using only 44,000 for most training) provides an unbiased estimate for hyperparameter selection (learning rate, momentum, weight decay). The final test uses the official 10,000-image test set, which is never touched during training or validation.

## 4. Key Insights and Innovations

### Innovation 1: Eliminating Explaining Away Through Model Design Rather Than Approximating Around It

The dominant assumption in the field prior to this work — which the paper explicitly calls out as "widely assumed to be impossible" — was that explaining away in densely connected belief nets is an unavoidable fact of inference that must be *approximated*, not eliminated. Variational methods (Neal & Hinton, 1998) accept that the posterior is intractable and substitute a simpler factorial distribution, optimizing a bound that may be loose at deeper layers. MCMC methods (Neal, 1992) accept the intractability and pay the computational cost of sampling from the true posterior. Both camps treat the posterior's non-factorial nature as a given constraint to work around.

The paper's foundational conceptual move is to reject this framing entirely. Rather than asking "how can we approximate the intractable posterior," it asks "can we design the model so the posterior becomes factorial by construction?" The answer — complementary priors — is not an incremental improvement on variational bounds or sampling schemes. It is a **fundamentally different kind of solution**: a structural constraint on the model architecture (tied weights extending infinitely upward, equivalent to an undirected RBM) that makes the inference problem trivial rather than merely tractable.

The significance of this move extends beyond the specific solution. It introduces a design principle: **when inference is hard, change the model to make inference easy, rather than fighting the model with better approximations**. This principle would later echo through the deep learning literature — variational autoencoders (Kingma & Welling, 2014) can be seen as learning an inference network that amortizes the cost of posterior approximation, and normalizing flows (Rezende & Mohamed, 2015) design transformations that make distributions tractable. But in 2006, this was a genuinely new way of thinking about the problem. The paper does not merely propose a better approximation; it changes the terms of the debate from "how good is your approximation" to "can you make approximation unnecessary."

The theoretical rigor in Appendix A — which characterizes exactly which likelihood functions admit complementary priors and proves that the infinite tied-weight network implements them — elevates this from a trick to a principled framework. The result that the joint distribution must take the form of a complete bipartite undirected graphical model (equation A.6) under the complementary prior conditions is a constructive characterization: it tells you exactly what model class to use. The equivalence to RBMs then provides the practical learning algorithm as a direct consequence, rather than as an ad-hoc procedure. This is theoretical work that directly prescribes practice, which is rare and valuable.

**Evidence anchoring:** The entire greedy pretraining algorithm (Section 4) and its variational guarantee depend on this insight. Figure 3 and equations 2.3–2.4 demonstrate the equivalence concretely: the learning gradient for the infinite directed net is exactly the RBM gradient.

---

### Innovation 2: Greedy Layer-by-Layer Unsupervised Pretraining as a General Strategy for Deep Architectures

Before this paper, training deep neural networks meant initializing all weights randomly and attempting to optimize them jointly with backpropagation — an approach that empirically failed beyond one or two hidden layers. The problem was understood as an optimization difficulty (vanishing gradients, poor local minima), but the solution space was confined to better optimization algorithms within the same joint-training paradigm.

The paper introduces a completely different training paradigm: **learn the network one layer at a time, bottom-up, with each layer trained to model the statistical structure of the previous layer's representations, without any label information**. This is not merely a better initialization scheme for backpropagation (though it functions as one); it is a different philosophy of how deep representations should be learned. The contrast drawn in Section 4 to boosting, projection pursuit, and Sanger's PCA is instructive: all of these methods modify the data or its weighting to force subsequent models to learn something new. The greedy RBM algorithm does something qualitatively different — it **re-represents** the data through a nonlinear transformation, such that each layer learns the statistical structure *at its own level of abstraction*.

What makes this a genuine innovation rather than an obvious extension of RBM training is the **theoretical justification for why layer-by-layer learning should work at all**. The naive approach — train a shallow model, fix it, and train another model on its outputs — has no guarantee of improving the full generative model. In fact, it risks making things worse because the fixed lower layers may produce representations that are suboptimal for the higher layers. The paper's variational argument (equations 4.1–4.3) shows that, under the tied-weights assumption (which makes the complementary prior exact), the greedy procedure is **guaranteed not to decrease a lower bound on the data log-probability**. Even though this guarantee is technically voided by the use of contrastive divergence rather than exact maximum likelihood, the conceptual framework it provides — that each layer is learning a model of the data as transformed by all previous layers — is what justifies the approach.

This innovation is **fundamental** rather than incremental because it opens an entire research program. The idea that deep networks should be *pretrained* in an unsupervised fashion before being *fine-tuned* on a supervised task became the dominant paradigm in deep learning for roughly the next decade (until advances in batch normalization, residual connections, and large labeled datasets made end-to-end supervised training competitive again). The paper's specific instantiation (stacked RBMs with CD learning) is one member of a much broader class of unsupervised pretraining methods — denoising autoencoders, variational autoencoders, and contrastive predictive coding all follow the same high-level recipe: learn representations layer by layer or through self-supervision, then fine-tune.

**Evidence anchoring:** Section 6 reports that after greedy pretraining alone, the network achieves 2.49% error on MNIST — already competitive with best backpropagation results of the time (1.5–3%). The up-down fine-tuning further reduces this to 1.25%, but the 2.49% result demonstrates that the greedy pretraining by itself produces useful representations without any label information at the lower layers.

---

### Innovation 3: The Up-Down Algorithm as a Contrastive Solution to the Mode-Averaging Problem in Wake-Sleep

The wake-sleep algorithm (Hinton et al., 1995) was a prior attempt at learning deep directed networks with separate recognition and generative weights. It suffered from a specific pathology that limited its effectiveness: **mode averaging**. When the true posterior over hidden states is multimodal (multiple different hidden configurations could plausibly explain the same observation), the recognition model trained in the sleep phase tends to map visible inputs to a weighted average of these modes rather than committing to one. This produces a diffuse, imprecise recognition distribution — the recognition weights learn to point to the "center of mass" of the posterior modes, which may correspond to no actual valid hidden representation. The sleep phase compounds this because it uses independent top-level priors to generate fantasies, which explore regions of hidden state space far from where real data lives.

The paper's up-down algorithm is not a minor tweak to wake-sleep but a **structural re-engineering** that addresses mode averaging through two mechanisms. First, it replaces the independent top-level prior with an **undirected associative memory** that can settle to a coherent multimodal state through Gibbs sampling. This is crucial: the independent prior assumed in standard wake-sleep is a poor model for the top-level representations, precisely because it assumes away all the correlations that make deep representations useful. The RBM at the top learns those correlations, so the "fantasies" generated during the down-pass come from a realistic joint distribution over high-level features.

Second, and more subtly, the **contrastive initialization** of the down-pass — starting Gibbs sampling from the data-driven up-pass state rather than from equilibrium or from independent priors — ensures that the recognition weights are trained on hidden representations that are **relevant to actual data**. If the recognition model tends to pick one mode of the posterior for a given data point, the down-pass will explore the neighborhood of that mode (via a few Gibbs steps) but will not be forced to cover other distant modes. This means the recognition weights are not penalized for ignoring alternative explanations — a sharp, mode-committing recognition model is actually rewarded. The paper states this directly:

> "It ensures that the recognition weights are being learned for representations that resemble those used for real data, and it also helps to eliminate the problem of mode averaging."

This is a **diagnostic innovation** as much as an algorithmic one: it identifies *why* wake-sleep fails (mode averaging from independent priors) and designs the architecture (top-level RBM + contrastive initialization) to specifically counteract that failure mode. The increasing Gibbs steps during training (3 → 6 → 10 over 300 epochs) is a curriculum that gradually demands better mixing as the model improves, showing a nuanced understanding of the interplay between learning dynamics and MCMC mixing.

**Evidence anchoring:** The observation that "each time the number of iterations of Gibbs sampling was raised, the error on the validation set decreased noticeably" (Section 6.1) validates that better mixing in the associative memory directly improves the quality of the learned generative model. The final 1.25% error rate — surpassing the best discriminative methods — is the ultimate evidence that the up-down algorithm successfully learns excellent recognition weights without mode averaging.

---

### Innovation 4: Demonstrating That Generative Models Can Outperform Discriminative Ones on Classification — With a Boundary Condition

The paper's most empirically striking claim is not just that deep belief nets can be trained, but that a generatively trained model achieves **better classification accuracy** than the best discriminative methods on the permutation-invariant MNIST benchmark. Table 1 shows 1.25% error versus 1.4% for SVMs (Decoste & Schoelkopf, 2002) and 1.5% for the best backpropagation networks. This was counter to the prevailing wisdom of the time, which held that discriminative methods should dominate on prediction tasks because they focus on the decision boundary rather than "wasting" capacity on modeling the input distribution.

What elevates this from a benchmark result to an intellectual contribution is the paper's explicit framing of **when and why** generative models should win. The abstract and Section 8 do not claim universal superiority. They articulate a specific condition:

> "The superior classification performance of discriminative learning methods holds only for domains in which it is not possible to learn a good generative model. This set of domains is being eroded by Moore's law."

This is a **thesis about the relationship between computation, data, and model class**. Discriminative models are sample-efficient in terms of labels (each label provides `$\log_2(10) \approx 3.3$` bits of constraint) but sample-inefficient in terms of data (they cannot use unlabeled examples during training). Generative models are the opposite: each unlabeled image provides 784 dimensions of constraint, orders of magnitude more information per example than a label. When computation is cheap enough and data is plentiful enough to learn a good generative model of the input distribution, that model will extract more statistical structure and ultimately generalize better, even on discriminative tasks.

This argument reframes the generative-vs-discriminative debate from an architectural preference to an **empirical claim about scaling**. It predicts that as computational resources grow, the set of domains where generative models dominate will expand — a prediction that has aged remarkably well given the subsequent rise of self-supervised pretraining and generative models (BERT, GPT, diffusion models) that achieve state-of-the-art on discriminative benchmarks.

The paper also implicitly makes a **methodological point**: learning a joint model `$P(\text{image}, \text{label})$` rather than a conditional `$P(\text{label}|\text{image})$` provides capabilities beyond classification — generation, interpretation of hidden representations, and exploration of the model's "mental states" (Section 7). These are not merely side benefits; they are evidence that the model has learned something genuine about the data structure. The ability in Section 7 to generate coherent digit images, explore the free-energy landscape, and visualize what the associative memory "has in mind" demonstrates a level of interpretability absent from SVMs or backpropagation networks.

**Evidence anchoring:** Table 1 provides the classification numbers. Figure 8 (class-conditional generation) and Figure 9 (evolution of mental states in the associative memory) provide the qualitative evidence that the generative model has captured real structure, not just achieved a benchmark number. The fact that further training for six weeks only reduced error to 1.18% suggests the 1.25% result is near the architectural limit, not an artifact of early stopping.

---

### Innovation 5: The Infinite Directed Network as a Conceptual Bridge Between Directed and Undirected Models

This innovation is more abstract than the others but is the intellectual linchpin that makes the entire paper coherent. The equivalence between an **infinite directed belief net with tied weights** and a **single restricted Boltzmann machine** is not merely a mathematical curiosity — it is a **conceptual unification** of two seemingly different model classes that enables transfer of learning algorithms from one to the other.

Before this paper, directed belief nets and undirected Boltzmann machines were studied as separate formalisms with different properties and different learning algorithms. Directed models have simple ancestral sampling for generation but intractable inference. Undirected models have simple conditional distributions (due to the bipartite structure of RBMs) but require expensive Gibbs sampling for both generation and learning. The infinite directed net construction shows that these are **two views of the same underlying object**: the RBM is the finite "summary" of an infinite directed stack, and Gibbs sampling in the RBM corresponds exactly to the alternation of inference and generation in the directed stack.

The practical consequence — which Section 3.4 explains in detail — is that **contrastive divergence learning in an RBM simultaneously learns the bottom layer of the infinite directed net**. This is what makes the greedy layer-by-layer algorithm possible: training an RBM on data is equivalent to training the first layer of a deep directed network under the assumption that the higher layers have tied weights providing a complementary prior. Untying the weights after training is equivalent to saying "the higher layers will eventually learn a better prior, but for now, the tied-weight assumption was good enough to learn a useful first layer."

This insight also explains **why CD learning works at all for deep networks**. As the paper notes, CD "fails for deep, multilayer networks with different weights at each layer because these networks take far too long even to reach conditional equilibrium." The equivalence reveals that the failure is not with CD per se but with applying it to the wrong architecture. When the weights are tied, the directed network's layers are coupled in a way that makes the RBM equivalence hold, and CD efficiently trains the whole stack simultaneously (because training the RBM trains the equivalent infinite directed net). When the weights are untied, this coupling is broken, and CD can no longer efficiently propagate learning signals through many layers — hence the need for the greedy layer-by-layer procedure followed by separate fine-tuning.

This is a **theoretical bridge** that connects directed and undirected graphical models, inference and generation, and shallow and deep learning in a single construction. It is not an algorithm that could be replaced by a different optimization method; it is a **structural insight** about how model architecture, inference, and learning are fundamentally linked.

**Evidence anchoring:** The equivalence is established formally in Appendix A (equations A.13–A.20) and computationally in equations 2.2–2.4, which show that the gradient of the infinite directed net telescopes to the RBM gradient. The practical consequence is that Section 4's greedy algorithm works — without this equivalence, training the first layer with an RBM would have no principled connection to the full deep architecture.

## 5. Experimental Analysis

### Evaluation Methodology

- **Dataset.** The MNIST database of handwritten digits, consisting of 60,000 training images and 10,000 test images (28×28 grayscale pixels). The paper uses the "permutation-invariant" version of the task where no knowledge of spatial geometry is provided — a random permutation of the pixels would not affect the learning algorithm. From the 60,000 training images, 44,000 are used for most of the training, with 10,000 held out as a validation set for hyperparameter selection, and the remaining 6,000 apparently unused. The official 10,000-image test set is used only for final evaluation.

- **Base model.** A custom-designed deep belief network with architecture 784 → 500 → 500 → 2000 ↔ 10 (input pixels, two hidden layers of 500 units each, an associative memory with 500 penultimate units and 2000 top-level units, plus 10 softmax label units connected to the 2000-unit top layer). All units are stochastic binary except the visible units during RBM training, which use real-valued activities between 0 and 1. The network contains approximately 1.7 million parameters. There is no pretrained base model — the network is trained from scratch using the proposed greedy layer-by-layer algorithm followed by up-down fine-tuning.

- **Metrics.** Classification error rate on the 10,000-image test set — the fraction of images for which the model's predicted digit label does not match the ground truth. Predictions are made by a deterministic bottom-up pass (using activation probabilities rather than stochastic binary states) followed by exact free-energy computation over the 10 possible label configurations in the top-level associative memory. The paper also reports error rates after greedy pretraining alone (2.49%) and at various stages of fine-tuning, using a deterministic up-pass.

- **Baselines.** The paper compares against published results from prior work, not against reimplemented baselines. The key baselines on the permutation-invariant version of MNIST are:
  - **Support vector machine** with degree-9 polynomial kernel: 1.4% error (Decoste & Schoelkopf, 2002)
  - **Backpropagation network** (784→500→300→10) with cross-entropy loss and weight decay: 1.51% error (John Platt, personal communication, 2005)
  - **Backpropagation network** (784→800→10) with cross-entropy loss and early stopping: 1.53% error
  - **Backpropagation network** (784→500→150→10) with squared error and online updates: 2.95% error
  - **Nearest neighbor**: 2.8% error (L3 norm, all 60,000 training examples), 3.1% (L2 norm, all 60,000), 4.0% (L3 norm, 20,000 examples), and 4.4% (L2 norm, 20,000 examples)
  - **Convolutional neural network** (LeNet5) on unpermuted images without data augmentation: 0.95% error (LeCun et al., 1998) — not directly comparable since it exploits spatial structure, but included in Table 1 for context

- **Generation budget / compute accounting.** Training time is the relevant compute measure rather than a per-example generation budget. Greedy layer-by-layer pretraining takes "a few hours per layer in MATLAB on a 3 GHz Xeon processor." The up-down fine-tuning runs for 300 epochs on 44,000 images, then a further 59 epochs on all 60,000 images, totaling "about a week" of training. There is no formal FLOPs accounting of the kind seen in modern scaling studies — the compute measurement is wall-clock time on a specific processor.

- **Cross-validation / statistical protocol.** The 60,000 training images are split: 44,000 for training, 10,000 as a validation set for selecting learning rate, momentum, and weight decay. The test set is the standard 10,000-image MNIST test set, used only once for final evaluation. Training uses 440 balanced mini-batches of 100 images each, with exactly 10 examples per digit class per mini-batch. For the final training run on all 60,000 images, the unequal class distribution is handled by randomly assigning images to 600 mini-batches. There is no k-fold cross-validation — the validation set serves as a single holdout for hyperparameter selection, and the test set provides a single final performance number.

### Main Quantitative Results

#### Greedy Pretraining Performance

After the greedy layer-by-layer training alone (three RBMs trained for 30 epochs each, bottom-up), the network achieves **2.49% error** on the test set using a deterministic up-pass for classification. This result is reported in Section 6.1 as an intermediate milestone.

This number is significant because it already places the purely unsupervised, greedily trained network in competitive territory with the best backpropagation results of the era (1.5–3.0% error, depending on architecture and training details, as shown in Table 1). The greedy pretraining does not use any label information at the lower layers — the labels are incorporated only at the top-level associative memory — yet the resulting representations are sufficiently informative to support classification accuracy within striking distance of fully supervised discriminative methods. This validates the core premise that unsupervised layer-by-layer learning captures meaningful statistical structure.

#### Up-Down Fine-Tuning Progression

The up-down fine-tuning progressively reduces error, with a clear dependence on the number of Gibbs steps used in the associative memory during the negative phase. Section 6.1 reports:

- After the first 100 epochs (using 3 full iterations of alternating Gibbs sampling in the associative memory before each down-pass): error decreased from the 2.49% greedy baseline, though the exact number is not reported.
- After the second 100 epochs (6 Gibbs iterations): "the error on the validation set decreased noticeably."
- After the third 100 epochs (10 Gibbs iterations): again "the error on the validation set decreased noticeably."

This progressive improvement with better mixing in the associative memory is a key finding: it demonstrates that the quality of the top-level undirected model directly affects the quality of the entire generative model, and that the coarse negative phase from few Gibbs steps is a bottleneck that more thorough equilibration can alleviate. The fact that performance improves each time the number of Gibbs steps is increased — rather than plateauing — suggests that the model's learning is limited by the quality of the negative phase and that even more Gibbs steps might have yielded further improvements, though this was not tested beyond 10.

The network that performed best on the validation set after 300 epochs achieved **1.39% error** on the test set. This network was selected as the best checkpoint from the hyperparameter sweep (learning rate, momentum, weight decay chosen using the separate 10,000-image validation set).

#### Final Performance After Full Training

The 1.39% network was then trained on all 60,000 training images (rather than the 44,000 subset) "until its error rate on the full training set was as low as its final error rate had been on the initial training set of 44,000 images." This took a further 59 epochs, bringing the total training time to "about a week." The final network achieves **1.25% error** on the official 10,000-image test set.

This 1.25% figure is the paper's headline result and is reported in the abstract, Section 1, and Section 6.1. It is placed in direct comparison with prior work in Table 1:

> "This beats the 1.5% achieved by the best backpropagation nets when they are not handcrafted for this particular application. It is also slightly better than the 1.4% errors reported by Decoste and Schoelkopf (2002) for support vector machines on the same task."

The improvement from 1.4% (best prior permutation-invariant result) to 1.25% represents a relative error reduction of approximately 10.7% — meaningful on this benchmark but not a dramatic leap. The paper acknowledges that "substantial reductions in the error rate can be achieved by supplementing the data set with slightly transformed versions of the training data" (Decoste & Schoelkopf achieve 0.56% with pixel translations; Simard et al. achieve 0.4% with elastic deformations in a convolutional network), and notes that "we have not yet explored the use of distorted data for learning generative models."

#### Convergence and Asymptotic Performance

To verify that the 1.25% result represents genuine convergence rather than premature stopping, the network was left running with a very small learning rate after the 1.25% checkpoint:

> "After six weeks, the test error was fluctuating between 1.12% and 1.31% and was 1.18% for the epoch on which number of training errors was smallest."

This is an important methodological detail. The 1.12–1.31% fluctuation range suggests that the 1.25% result is near the effective asymptote of what this specific architecture (784→500→500→2000↔10, ~1.7M parameters) can achieve on permutation-invariant MNIST. The best observed point (1.18% on the epoch with fewest training errors) is better than 1.25% but was selected post-hoc, so the 1.25% figure is the more conservative and reliable one. The six-week extended run provides confidence that the up-down algorithm had not left significant performance on the table — further training yielded marginal and noisy improvements rather than a clear downward trend.

#### Testing Methodology Ablation: Stochastic vs. Deterministic Inference

Section 6.2 reports a comparison of three testing methods, which is effectively an ablation on the inference procedure:

1. **Stochastic up-pass + Gibbs sampling for labels:** Binary states are sampled during the up-pass, then the label is determined by running a few iterations of Gibbs sampling in the associative memory with label units initialized to 0.1. This method gives error rates "almost 1% higher than the rates reported above" — meaning approximately 2.25% rather than 1.25%.

2. **Exact free-energy computation with stochastic up-pass:** The binary states from the stochastic up-pass are fixed, and the exact free energy is computed for each of the 10 possible label configurations by exploiting the fact that "almost all the computation required is independent of which label unit is turned on" (Teh & Hinton, 2001). This gives error rates "about 0.5% higher than the ones quoted" — approximately 1.75%. The gap between methods 1 and 2 (~0.5%) isolates the benefit of exact free-energy computation over approximate Gibbs sampling for label selection.

3. **Deterministic up-pass + exact free-energy computation (reported method):** Using activation probabilities instead of sampled binary states during the up-pass makes the entire inference deterministic. The exact free energy is then computed as in method 2. This gives the reported 1.25%.

The paper also validates that averaging label probabilities or log-probabilities over 20 stochastic up-passes gives "almost identical results" and are "very similar to using a single deterministic up-pass." This confirms that the 0.5% gap between methods 2 and 3 is due to sampling noise in the hidden states, which can be removed either by determinism or by averaging, and that the stochasticity is zero-mean noise rather than systematic bias.

These results collectively demonstrate that the learned generative model supports multiple inference procedures with different accuracy-efficiency tradeoffs, and that the reported 1.25% represents the optimal inference method rather than an artifact of a particular testing protocol.

#### Error Analysis: Qualitative Patterns

Figure 6 shows the 125 test cases that the network got wrong (out of 10,000, consistent with 1.25% error). Each misclassified image is labeled with the network's incorrect guess. The paper does not provide a quantitative breakdown of error types, but the figure allows qualitative inspection. Figure 7 shows the 49 cases where "the network guessed right but had a second guess whose probability was within 0.3 of the probability of the best guess" — these are cases where the classification was correct but uncertain, representing 0.49% of the test set. The existence of these near-ties suggests that a nontrivial fraction of the remaining errors may be similarly ambiguous — cases where the model's top two predictions are close and the correct answer happens to be second.

#### Generative Quality: Class-Conditional Sampling and Mental State Evolution

Section 7 provides qualitative evidence that the model has learned a meaningful generative distribution, not merely a decision boundary. Figure 8 shows 10 samples from the model's class-conditional distribution for each digit, generated by clamping the label and running the top-level associative memory for 1000 iterations of alternating Gibbs sampling between samples. The generated digits are recognizable and exhibit natural within-class variation — different writing styles, slants, and stroke weights — indicating that the model has captured the manifold of digit appearance rather than memorizing templates.

Figure 9 explores the dynamics of the associative memory. Starting from a random binary image (each pixel on with probability 0.5), an up-pass initializes the top-level state. The figure then shows a down-pass from this initial state (first column) followed by down-passes after every 20 iterations of Gibbs sampling in the associative memory (subsequent columns), with the label clamped to a particular digit class. The sequence shows the associative memory evolving from a noisy, unrecognizable initial state toward a coherent digit image of the specified class over approximately 100–200 Gibbs iterations. This is not presented as a quantitative result but as qualitative evidence that the associative memory's free-energy landscape contains basins of attraction corresponding to digit classes — the Gibbs sampling gradually descends into the appropriate basin.

### Ablation Studies and Robustness Checks

The paper does not contain formal ablation studies in the modern sense (systematic removal of components with quantitative comparison). However, several implicit ablations and sensitivity analyses are present:

**Effect of Gibbs steps in associative memory during fine-tuning**: Increasing the number of alternating Gibbs sampling iterations from 3 to 6 to 10 over successive 100-epoch phases produces "noticeable" decreases in validation error at each increase (Section 6.1). This demonstrates that the quality of the negative phase matters and that 3 steps are insufficient for the model to reach its best performance. No experiment tests whether further increases (e.g., 20 or 50 steps) would continue to yield improvements, though the later 6-week extended run with 10 steps showed only marginal further gains (1.25% → 1.18% at best), suggesting diminishing returns.

**Stochastic vs. deterministic inference**: As described in the testing methodology comparison, removing sampling noise from the up-pass improves accuracy by approximately 0.5 percentage points (Section 6.2). This quantifies the inference noise penalty and validates the use of deterministic probabilities rather than stochastic binary states for classification.

**Gibbs sampling vs. exact free energy for label selection**: Using approximate Gibbs sampling for label inference degrades accuracy by approximately 1 percentage point compared to exact free-energy computation (Section 6.2, methods 1 vs. 2). This establishes that the exact computation is worth the additional cost, and that the Gibbs approximation has not converged sufficiently in the few iterations used.

**Training set size**: The network achieves 1.39% error when trained on 44,000 images and 1.25% when trained on all 60,000, a relative improvement of approximately 10%. This confirms that the model benefits from additional training data, consistent with the paper's thesis that generative models can use unlabeled data effectively. No experiment tests performance with smaller subsets to establish a data scaling curve.

**Extended training asymptotic behavior**: The six-week extended run with a very small learning rate shows test error fluctuating between 1.12% and 1.31% (Section 6.1). This bounds the best achievable performance for this architecture on this task and confirms that the 300+59 epoch training protocol had not prematurely converged to a poor local optimum.

**Notable missing ablations**: The paper does not report:
- Performance of the same architecture trained with random initialization and backpropagation (to isolate the benefit of greedy pretraining vs. the architecture itself).
- Performance with fewer hidden layers (e.g., 784→500→10 or 784→500→500→10 without the 2000-unit associative memory) to quantify the benefit of depth.
- Performance with different numbers of units per layer to establish sensitivity to width.
- Performance of the greedy pretraining alone at each layer (1-layer, 2-layer, 3-layer) to show how performance accumulates with depth before fine-tuning.
- Comparison against an RBM-only model (single hidden layer) with the same total number of parameters.
- CD-$n$ with $n > 1$ during greedy pretraining to test whether more Gibbs steps in the lower layers improves the initialization.

### Critical Assessment

The experiments demonstrate that the proposed training procedure — greedy layer-by-layer RBM pretraining followed by contrastive wake-sleep fine-tuning — can train a deep belief network with three hidden layers to achieve 1.25% error on permutation-invariant MNIST. This result is genuine and was state-of-the-art at publication. However, the experimental evidence supports a narrower conclusion than some of the paper's broader claims.

**Does the paper demonstrate that deep belief nets can be trained layer by layer?** Yes. The 2.49% greedy-only error rate — competitive with fully supervised backpropagation networks — directly demonstrates that unsupervised, layer-by-layer pretraining produces useful representations. The further improvement to 1.25% with joint fine-tuning confirms that the pretrained initialization enables the full model to reach a solution that outperforms the best prior methods. This is the paper's core algorithmic claim and it is well-supported.

**Does the paper demonstrate that generative models outperform discriminative ones on classification?** Yes, but with important scope limitations that the paper itself acknowledges. The 1.25% error beats the 1.4% SVM (Decoste & Schoelkopf, 2002) and the 1.5% best backpropagation result. However:

- The margin is narrow (0.15–0.25 percentage points), and the paper does not report confidence intervals on its own error rate. The six-week extended run shows fluctuations of ±0.1 percentage points around 1.22%, so the difference from 1.4% is only approximately 1–2 standard deviations of the model's own run-to-run variability — statistically suggestive but not overwhelming.
- The discriminative baselines are from prior publications, not reimplemented and re-optimized under identical conditions. It is possible that a more exhaustive hyperparameter search for backpropagation or a different SVM kernel could close the gap.
- The paper's broader claim — that "the superior classification performance of discriminative learning methods holds only for domains in which it is not possible to learn a good generative model" — is a theory about scaling, not a finding that 1.25% < 1.4% on MNIST constitutes proof. The MNIST result is presented as one data point supporting this theory, not as definitive evidence.

**Does the paper demonstrate that the up-down algorithm solves the mode-averaging problem?** The evidence is indirect. The 1.25% result demonstrates that the algorithm works well, and the observation that validation error decreases when Gibbs steps are increased is consistent with the claim that better mixing improves recognition weight quality. But there is no direct comparison to standard wake-sleep on the same architecture (which would likely perform poorly, motivating the contrastive approach) and no quantitative measurement of mode-averaging behavior. The claim about mode averaging is a theoretical argument supported by the algorithm's design and its strong empirical performance, not by a controlled ablation.

**Critical weaknesses:**

The paper operates with a single architecture (784→500→500→2000↔10) on a single dataset (MNIST) using a single training protocol. There is no evidence that the method generalizes to other datasets, other input dimensionalities, or other problem types. The architecture hyperparameters (500, 500, 2000 units) appear to have been chosen based on "preliminary experiments with 16×16 images of handwritten digits from the USPS database" (Section 6.1 footnote), but no systematic architecture sweep is reported. This is understandable for a paper introducing a fundamentally new training paradigm — establishing that it works at all is the primary contribution — but it means the reader cannot know whether the specific architecture was critical to the result or whether a wide range of configurations would work.

The training protocol involves multiple stages with different hyperparameters (RBM training epochs, up-down learning rate, momentum, weight decay, Gibbs steps schedule) that were tuned on a validation set. The paper does not report the sensitivity of the final result to these choices. If the hyperparameters were extensively optimized, the reported 1.25% may partly reflect tuning effort rather than inherent superiority of the method.

The 44,000 / 10,000 / 6,000 split of the 60,000 training images means the model was selected based on validation performance, but the final 1.25% result comes from a network retrained on all 60,000 images for a further 59 epochs — a different training protocol than the one used for validation-based model selection. The paper does not explain how the hyperparameters chosen on the 44,000-image validation run were adapted for the 60,000-image run, or whether the 59-epoch stopping point was chosen using test set feedback (which would constitute a form of test set leakage). The description "until its error rate on the full training set was as low as its final error rate had been on the initial training set of 44,000 images" suggests that training-set error (not test-set error) was used to determine stopping, which avoids direct leakage but assumes that training error on 60,000 images is a good proxy for when to stop — an assumption that may not hold if overfitting differs between the 44,000 and 60,000 regimes.

**Missing experiments that would have strengthened the paper:**

A direct head-to-head comparison against an identical architecture trained with standard backpropagation from random initialization would isolate the benefit of the greedy pretraining + up-down approach from the benefit of the architecture itself. If backpropagation on the 784→500→500→2000↔10 architecture achieves, say, 3% error, then the pretraining provides a large benefit. If it achieves 1.5%, the benefit is modest. If it fails entirely (as the paper implies by noting that CD "fails for deep, multilayer networks with different weights at each layer"), then the pretraining is essential. The paper provides the last argument implicitly but does not run the experiment.

Ablations on depth — removing one hidden layer, or removing the 2000-unit associative memory — would reveal whether the depth enabled by the training method is actually responsible for the performance gain, or whether a shallower model with similar total parameters could match the result.

The paper does not report training-set error, only test-set error. The generalization gap (difference between training and test error) would indicate whether the 1.25% result is limited by underfitting (large training error, small generalization gap) or by the model's capacity (small training error, larger generalization gap). This information is absent.

**Conditional nature of the claims:**

The paper's generative-vs-discriminative claim explicitly depends on a domain where "it is possible to learn a good generative model." MNIST satisfies this condition — the digit images are low-dimensional, highly structured, and amenable to generative modeling — so the result does not demonstrate the claim's generality. The prediction that this "set of domains is being eroded by Moore's law" is forward-looking and unverifiable within the paper's scope.

The paper's claim that the algorithm can learn "deep, densely connected belief nets" is demonstrated for three hidden layers. Whether the method scales to five, ten, or twenty layers — where the variational approximation might degrade, the cumulative errors from CD learning might compound, and the up-down fine-tuning might become unstable — is untested. The paper's own caveat about bigger networks being needed "to compete with human shape recognition abilities" acknowledges this limitation implicitly.

**Summary of evidential strength:**

The paper convincingly demonstrates a working training algorithm for a three-hidden-layer belief network on MNIST, achieving state-of-the-art classification performance. The core algorithmic contributions — complementary priors enabling greedy layer-by-layer pretraining, the RBM equivalence, and the up-down fine-tuning — are validated by the empirical result. The broader claims about generative models surpassing discriminative ones, and about the method scaling to much larger networks, are plausible and forward-looking but not directly tested. The experimental section is best read as a proof of concept for a new training paradigm rather than a comprehensive empirical characterization of that paradigm's properties and limits.

## 6. Limitations and Trade-offs

### The Infinite Directed Model Construction Requires Tied Weights That Are Untied During Learning

**The assumption or constraint.** The entire theoretical framework — complementary priors, exact factorial posteriors, the equivalence to RBMs — depends on the assumption that weight matrices between consecutive layers are tied (specifically, transposes of each other) and that this tying extends infinitely upward. The paper states this directly in Section 2:

> "The use of tied weights to construct complementary priors may seem like a mere trick for making directed models equivalent to undirected ones. As we shall see, however, it leads to a novel and very efficient learning algorithm that works by progressively untying the weights in each layer from the weights in higher layers."

The greedy pretraining algorithm exploits this tied-weight assumption to make the posterior factorial at each layer, then **violates the assumption** by freezing the bottom weights and learning the next layer's weights independently. The up-down fine-tuning goes further by untying the recognition weights from the generative weights entirely, so that the bottom-up inference pathway uses different parameters than the top-down generative pathway.

**The consequence.** Once the weights are untied, the complementary prior property is destroyed. The true posterior over hidden variables is no longer factorial. The paper is explicit about this (Section 4):

> "As these higher-level weights change, the priors for lower layers cease to be complementary, so the true posterior distributions in lower layers are no longer factorial, and the use of the transpose of the generative weights for inference is no longer correct."

The practical consequence is subtle but important: the inference procedure used throughout (bottom-up factorial sampling via the recognition weights) is **an approximation** after the first layer of greedy training, and the paper provides no bound on how good or bad this approximation becomes as higher layers are progressively untied and retrained. The variational guarantee (equations 4.1–4.3) shows that the bound on the data log-probability is tight at the moment each layer is frozen (because the tied-weight assumption makes the posterior exactly factorial), and that subsequent higher-layer learning can only increase this bound. But the bound itself becomes looser as the approximation degrades — we are optimizing an increasingly poor surrogate for the true log-likelihood. There is no guarantee that maximizing this loose bound actually improves the true model quality, only that it does not decrease the bound below its previous tight value.

This is not a mere theoretical quibble. The quality of the factorial approximation at deeper layers determines whether the up-down fine-tuning receives useful gradient signals. If the recognition model (which assumes factorial posteriors) systematically ignores important posterior correlations, the generative weights will be updated based on misleading hidden-state samples, potentially steering the model toward a worse generative distribution even as the variational bound improves.

**What evidence exists in the paper.** The paper provides no direct measurement of how the factorial approximation degrades. There is no experiment comparing the true posterior (e.g., via long-run Gibbs sampling in the full model, if feasible) to the factorial approximation at each layer after greedy training or during fine-tuning. The empirical evidence that the approach works — 2.49% error after greedy pretraining, 1.25% after fine-tuning — demonstrates that the approximation is "good enough" for MNIST, but does not characterize when or why it might fail on harder problems with more complex posterior dependencies. The paper's own caveat about models of natural images (Section 8) is telling: the current model "is designed for images in which nonbinary values can be treated as probabilities (which is not the case for natural images)." This suggests the authors recognize that the modeling assumptions may not generalize.

**Mitigation status.** The paper does not attempt to mitigate this limitation beyond the pragmatic observation that the procedure works empirically. The up-down fine-tuning phase could be viewed as partially compensating: by jointly adjusting all weights, the model can reshape the generative distribution to better match the data even with an imperfect inference procedure. But the paper does not frame this as a solution to the approximation problem, nor does it provide a theoretical analysis of why fine-tuning helps. The limitation is fundamental to the approach: the factorial approximation is what makes learning tractable, but its degradation is what makes the learning suboptimal. Any deep model trained with this method operates under a tension between tractability (factorial assumption during greedy learning) and accuracy (non-factorial true posterior), and the paper provides no tools for diagnosing or controlling this tension.

---

### The Method Is Demonstrated on a Single Dataset with a Single Architecture, With No Sensitivity Analysis

**The assumption or constraint.** All experiments use the MNIST handwritten digit database (10-digit classification, 28×28 grayscale images) with a single architecture: 784 → 500 → 500 → 2000 ↔ 10. The architecture hyperparameters (two 500-unit hidden layers, a 2000-unit top layer in the associative memory) were chosen based on preliminary experiments with 16×16 USPS digit images (Section 6.1 footnote), but no systematic architecture sweep is reported. There is no evidence that the method works on any other dataset, any other input modality (natural images, speech, text), or any other task type (regression, multi-label classification, structured prediction).

The architecture size is not arbitrary — the 2000-unit top layer is large relative to the 500-unit hidden layers, and the paper provides no justification for this asymmetry. A practitioner wanting to apply the method to a new domain has no guidance on how to choose the number of layers, the width of each layer, the size of the associative memory, or the number of training epochs for the greedy and fine-tuning phases beyond what worked for MNIST.

**The consequence.** The paper's headline claim — that generative deep belief nets can outperform discriminative methods — is supported by a single data point. MNIST is an unusually favorable testbed for generative models: the digit classes are well-separated, the images are low-dimensional and highly structured, the background is uniform, and the within-class variation is largely continuous (different writing styles) rather than discrete (different object identities). The method's success on MNIST does not imply success on more challenging datasets like CIFAR-10 (natural images with complex backgrounds), ImageNet (thousands of classes with high intra-class variability), or tasks requiring long-range dependencies (language, speech).

The absence of architecture sensitivity analysis means a practitioner cannot know whether the specific 784→500→500→2000↔10 configuration was critical to achieving 1.25% error, or whether a wide range of configurations would work similarly well. If the result is brittle — for example, if reducing the top layer from 2000 to 1000 units causes error to jump to 2% — then the method is less practically useful than the paper suggests, because extensive architecture tuning would be required for each new application. Conversely, if a 784→200→200→500↔10 network also achieves ~1.5%, the method is robust and the specific hyperparameters are incidental.

The paper acknowledges the scale limitation in its conclusion (Section 8):

> "The network in Figure 1 has about as many parameters as 0.002 cubic millimeters of mouse cortex... and several hundred networks of this complexity could fit within a single voxel of a high-resolution fMRI scan. This suggests that much bigger networks may be required to compete with human shape recognition abilities."

This is a forward-looking statement, not a demonstration. The paper provides no evidence that the training algorithm scales to networks orders of magnitude larger, or that the greedy layer-by-layer approach remains effective when the number of layers increases from three to ten or twenty. The fact that backpropagation through many layers was already known to suffer from vanishing gradients suggests that the up-down fine-tuning — which is closely related to backpropagation through the directed generative connections — might encounter the same difficulty in deeper architectures.

**What evidence exists in the paper.** Only MNIST results are reported. The preliminary USPS experiments (referenced in a footnote) are not described in sufficient detail to serve as a second data point. There is no experiment varying the number of hidden layers (e.g., comparing 1-layer, 2-layer, and 3-layer versions of the greedy pretraining) to establish a depth-ablation curve. There is no experiment varying layer widths or the size of the associative memory. There is no training-set error reported alongside test-set error, so the generalization gap cannot be assessed.

The paper's Table 1 compares against published results from other methods on the same MNIST benchmark, which is standard practice for introducing a new method. But the comparison is to prior work's best published numbers, not to reimplemented baselines under identical experimental conditions. The backpropagation baselines span a range of 1.51–2.95% depending on architecture and training choices, but the paper does not report how well backpropagation would perform on the identical 784→500→500→2000↔10 architecture with the same computational budget (a week of training). Without this comparison, it is impossible to attribute the 1.25% result to the training algorithm rather than to the large architecture or the extensive hyperparameter search.

**Mitigation status.** The paper does not address this limitation beyond the general statement about bigger networks in the conclusion. The authors do not claim that the method is general or that the specific architecture is optimal. But the lack of any evidence beyond MNIST means the paper's contributions should be understood as a **proof of concept** for a new training paradigm, not as a validated general-purpose method. A practitioner considering this approach for a different domain would need to run their own feasibility experiments from scratch, with no guidance from the paper on transferability.

---

### Difficulty Estimation and Cost of the Full Training Pipeline Are Not Characterized

**The assumption or constraint.** The paper reports a total training time of "about a week" on a 3 GHz Xeon processor in MATLAB (Section 6.1). This is the only compute measurement provided. There is no formal accounting of how this time breaks down across the training phases (30 epochs per RBM layer × 3 layers for greedy pretraining, vs. 300 + 59 epochs for up-down fine-tuning), no reporting of FLOPs or parameter updates, and no discussion of memory requirements.

More importantly, paper undertakes substantial hyperparameter optimization — learning rate, momentum, and weight decay were "chosen by training the network several times and observing its performance on a separate validation set of 10,000 images" (Section 6.1). The number of "several times" is not specified, nor is the total compute invested in these preliminary runs. The scheduling of Gibbs steps (3 → 6 → 10) was determined by observing validation performance, which implies additional exploratory runs.

**The consequence.** The reported "about a week" of training refers to the final successful training run, not the total compute required to develop the method. A practitioner attempting to apply this approach to a new dataset would need to invest substantial additional computation in hyperparameter search (learning rate, momentum, weight decay, number of RBM training epochs, number of fine-tuning epochs, Gibbs step schedule, layer sizes), and the paper provides no guidance on how costly this search will be or how sensitive the result is to each hyperparameter.

The asymmetric information about training phases is particularly problematic for assessing the method's efficiency claims. The greedy pretraining ("a few hours per layer") produces a model with 2.49% error — already competitive with some backpropagation results. The up-down fine-tuning takes approximately a week to reduce this to 1.25%. A practitioner might reasonably ask: is the week of fine-tuning worth 1.24 percentage points of error reduction? Could the same week of computation be better spent on a larger ensemble of greedily trained models (as the paper itself speculates in Section 8: "It might be better to omit the fine-tuning and use the speed of the greedy algorithm to learn an ensemble of larger, deeper networks")? The paper raises this question but does not answer it empirically.

The dependence on extensive hyperparameter tuning also weakens the comparison to discriminative baselines. The SVM and backpropagation results in Table 1 presumably also involved hyperparameter optimization, but the degree of tuning is not controlled for. If the 1.25% result required searching over more hyperparameter configurations than the 1.4% SVM result, the fair comparison would account for total computation, not just final model quality.

**What evidence exists in the paper.** The paper reports that the final 1.25% model was obtained after a network achieving 1.39% on the validation set was retrained on all 60,000 images for 59 additional epochs. The six-week extended run (Section 6.1) demonstrates that further training with a small learning rate yields noisy performance between 1.12% and 1.31%, with 1.18% at the best epoch (selected post-hoc). This provides some evidence about the asymptotic behavior but does not address the cost of getting to the 1.39% validation point in the first place.

The paper provides no learning curves (error vs. epoch) for either the greedy pretraining or the up-down fine-tuning. Without these, the reader cannot assess whether the 30-epoch RBM training and 300-epoch fine-tuning are near convergence or whether significantly more or less training would yield similar results.

**Mitigation status.** The paper does not characterize the computational cost of its hyperparameter search or provide sensitivity analyses. The extended training run (six weeks to observe fluctuations around 1.2%) is the closest the paper comes to a convergence analysis, but this is presented as a verification that the 1.25% result is near-asymptotic rather than as a cost-benefit analysis. The speculation about using greedy training alone for ensembles (Section 8) is an acknowledgement that the cost-benefit tradeoff of fine-tuning is unknown.

---

### The Testing Methodology Uses Deterministic Inference That Is Not Part of the Generative Model

**The assumption or constraint.** The reported 1.25% error rate uses a specific testing protocol (Section 6.2): a deterministic bottom-up pass where activation probabilities replace stochastic binary states, followed by exact free-energy computation over the 10 label configurations in the top-level associative memory. The paper reports that alternative testing methods degrade performance significantly:

> "This method of testing gives error rates that are almost 1% higher than the rates reported above" (stochastic up-pass + Gibbs sampling for labels)
> "This method gives error rates that are about 0.5% higher than the ones quoted" (stochastic up-pass + exact free energy)

The deterministic up-pass and exact free-energy computation are engineering choices to maximize classification accuracy, not components of the learned generative model. The generative model is defined by stochastic binary units — during generation, units are sampled from Bernoulli distributions, and the model's joint distribution `$P(\text{image}, \text{label})$` is defined over binary hidden states. The deterministic inference protocol instead computes expectations under the factorial approximate posterior.

**The consequence.** The 1.25% error rate does not reflect the performance of the generative model as a classifier under its own native inference procedure. The native classifier — the one consistent with the model's definition — would involve a stochastic up-pass and either Gibbs sampling or exact free energy for label selection, yielding error rates between approximately 1.75% and 2.25% (the paper's reported degradations of 0.5% and ~1% relative to deterministic inference).

This matters for two reasons. First, the paper's central claim that generative models can outperform discriminative ones on classification rests on the 1.25% vs. 1.4% comparison. If the generative model's native inference achieves ~1.75%, the claim weakens substantially — the model would still be competitive but no longer state-of-the-art. The deterministic inference procedure is essentially extracting a discriminative classifier from the generative model by replacing sampling with expectations, which is a common and valid technique, but it means the 1.25% number reflects a hybrid approach (generative training + discriminative inference) rather than a pure generative model's classification ability.

Second, for a practitioner wanting to use the trained model for both generation and classification, the deterministic inference protocol is the right choice for classification, and the 1.25% number is the practically relevant one. But the paper should be clearer that this number is achieved by modifying the inference procedure away from the model's native stochastic dynamics. The gap between 1.25% (deterministic) and ~1.75% (stochastic up-pass + exact free energy) quantifies how much the factorial recognition model's sampling noise hurts classification — a measure of inference quality that the paper does not discuss in these terms.

**What evidence exists in the paper.** Section 6.2 provides the three-way comparison of testing methods, explicitly quantifying the degradation from using stochastic inference. The paper states that averaging over 20 stochastic up-passes gives "almost identical results" to a single deterministic up-pass, confirming that the noise is zero-mean and can be eliminated by averaging. This is a partial defense — averaging 20 up-passes incurs a 20× computational cost at test time, so it is not equivalent to the deterministic method in practice.

The paper also reports that after greedy pretraining alone, the deterministic up-pass achieves 2.49% error (Section 6.1), but does not report what the stochastic up-pass error would be at that stage. It is possible that the gap between deterministic and stochastic inference changes during training, and that the fine-tuning phase specifically improves the deterministic pathway more than the stochastic one.

**Mitigation status.** The paper does not frame the deterministic-vs-stochastic inference gap as a limitation. The deterministic up-pass is presented as the natural testing method, and the stochastic alternatives are described as inferior. The observation that "averaging either the label probabilities or the label log probabilities over the 20 repetitions before picking the best one" gives results "very similar to using a single deterministic up-pass" provides a principled justification (the deterministic pass approximates the expectation over stochastic passes) but does not address the conceptual point that the model's native inference is stochastic and less accurate.

A practitioner should understand that the reported 1.25% error rate uses a testing protocol that is not equivalent to running the trained generative model as-is. The model must be "determinized" at test time to achieve this number. This is a standard technique (even today, dropout is disabled at test time, batch normalization uses running averages, and stochastic models are evaluated deterministically), but the paper's specific framing — generative models beating discriminative ones at classification — makes the distinction between model definition and evaluation protocol particularly important.

---

### The Generalization Claims Rest on a Single Test Set With No Replication Across Random Seeds

**The assumption or constraint.** The paper reports a single final result: 1.25% error on the MNIST test set from one training run (the greedy pretraining followed by 300 + 59 epochs of up-down fine-tuning). There is no replication across random initializations, no cross-validation across different training/test splits, and no reporting of variance or confidence intervals. The stochastic elements of the training procedure — random initialization of weights, stochastic binary sampling during RBM training and up-passes, random composition of mini-batches — all contribute to run-to-run variability that is not characterized.

The six-week extended run provides some evidence about variability near convergence (test error fluctuates between 1.12% and 1.31%), but this is variability of a single model across epochs, not variability across independent training runs from different initial conditions. A single model fluctuating within a range does not tell us whether a different random initialization would lead to a different attractor with systematically different performance. The gap between the 1.39% validation-selected network and the 1.25% final network (trained on more data for more epochs) could partly reflect randomness rather than genuine improvement — the paper cannot distinguish these because only one trajectory was run.

**The consequence.** The headline comparison — 1.25% vs. 1.4% for SVM — cannot be assessed for statistical significance without knowing the variance of the 1.25% estimate. If the standard deviation across independent training runs is, say, 0.1 percentage points, then the difference from 1.4% is approximately 1.5 standard deviations — suggestive but not statistically significant by conventional standards. If the standard deviation is 0.05, the difference is more convincing. Neither value is provided.

This is not merely a statistical formality. Neural network training in 2006 (and still today) exhibits substantial run-to-run variation due to random initialization and stochastic optimization. The paper's own backpropagation baselines in Table 1 span 1.51% to 2.95% across different architectures and training protocols, suggesting that the "best backpropagation result" is sensitive to many choices. Without replication, the 1.25% number could be a particularly fortunate training run rather than a reliable estimate of the method's expected performance.

The single-test-set evaluation also means the paper provides no evidence about how the method's performance varies with different random splits of the training data. MNIST has a fixed train/test split (60,000/10,000), which is standard and appropriate. But the paper further splits the 60,000 training images into 44,000 training, 10,000 validation, and 6,000 apparently unused — and this split is not described as being done randomly or with stratification. If the 44,000/10,000 split happened to favor the method (e.g., if the validation set was easier than average), the reported performance could be optimistic.

**What evidence exists in the paper.** There is no replication, no standard deviation reporting, and no discussion of variability across runs. The paper acknowledges that the learning rate, momentum, and weight decay were chosen by "training the network several times and observing its performance on a separate validation set," which implies that multiple runs were conducted but that only the best-performing configuration was carried forward to the final test-set evaluation. This is a form of implicit hyperparameter optimization that can inflate the apparent performance if the validation set is finite and the hyperparameter space is large.

The six-week extended run provides a within-run variability estimate (1.12–1.31%) but does not address between-run variability. A reader cannot know whether training the same architecture with the same hyperparameters from a different random seed would yield 1.20%, 1.30%, or 1.50%.

**Mitigation status.** The paper does not address this limitation. The era's publication standards did not require replication or confidence intervals — reporting a single test-set result was standard practice. The 1.25% number should be interpreted as a point estimate from one training run, not as the expected performance of the method. For a practitioner considering this approach, the takeaway is that the method can achieve approximately 1.25–1.4% error on MNIST, with the exact number depending on random variation and hyperparameter choices that are not fully characterized.

---

### The Hardest Problems: The Model Cannot Generate or Recognize Meaningfully Difficult or Ambiguous Cases

**The assumption or constraint.** The paper frames MNIST as a benchmark for evaluating the method, but MNIST digits span a limited range of difficulty. The "hardest" test cases — those that the model misclassifies — are not analyzed in terms of why the model fails or whether they represent fundamentally different kinds of difficulty that the model's architecture cannot handle.

The model's generative capacity, while visually impressive for 2006, produces samples that are recognizable but often distorted. Figure 8 (class-conditional generation) and Figure 9 (evolution from random initialization) demonstrate that the associative memory converges to digit-like patterns, but the generated images lack the crispness and consistency of real MNIST digits. Some generated samples have fragmented strokes, unnatural proportions, or ambiguous class membership. The paper does not quantify generation quality — there is no measure of how often generated samples are recognizable as the intended digit class, no comparison to the real data distribution via likelihood or visual Turing tests, and no analysis of failure modes in generation.

**The consequence.** If the model is used as a generative model of handwritten digits — for example, to synthesize training data or to complete partially occluded images — the practical utility depends on generation quality that the paper does not measure. The 1.25% classification error is a discriminative measure; it does not guarantee that samples from `$P(\text{image}|\text{label})$` are realistic or useful.

More subtly, the model's architecture may impose a structural limitation on what kinds of difficulty it can represent. The factorial approximate posterior assumes that features in each hidden layer are statistically independent given the layer below. For digits, this is a reasonable approximation — the presence of a horizontal stroke (detected by one hidden unit) is largely independent of the presence of a vertical stroke (detected by another), conditional on the pixel image. For objects with strong part-whole dependencies (e.g., a face where the presence of two eyes implies the presence of a nose in a specific spatial relationship), the factorial approximation is much less appropriate. The paper's demonstrations on MNIST do not reveal how severely this structural limitation affects generation or recognition quality on data with stronger feature dependencies.

This connects to the wider limitation of the complementary prior framework. The infinite directed net with tied weights delivers a factorial posterior only when the likelihood function satisfies the separability condition in Appendix A (equation A.1). For general data distributions, this condition may not hold, and the method's success depends on how well the data can be approximated by a model in this class. MNIST is well-approximated; it is unclear what other domains share this property.

**What evidence exists in the paper.** The paper provides the generated samples in Figures 8 and 9 but does not quantify their quality. Figure 6 (the 125 misclassified test images) is presented without analysis of whether these errors are on genuinely ambiguous cases (digits that humans would also find difficult) or on clear cases that the model should have gotten right. Figure 7 (the 49 cases where the second-best guess is within 0.3 of the best) suggests that some errors reflect genuine ambiguity, but the paper does not compare the model's uncertainty to human uncertainty on the same images.

The paper does not report the model's log-likelihood on the test set, which is the standard quantitative measure of generative model quality. Classification error measures only how well the model discriminates among classes, not how well it models the data distribution. A model that assigns high probability to unrealistic images could still achieve low classification error if the unrealistic images are assigned to the correct class.

**Mitigation status.** The paper acknowledges that the model "is limited in many ways" (Section 8), citing Lee & Mumford (2003) and listing specific limitations: design for images where nonbinary values can be treated as probabilities, limited top-down feedback, no handling of perceptual invariances, no learned attention. These are presented as directions for future work rather than as measured deficiencies. The lack of quantitative generation quality metrics is not addressed.

For a practitioner interested in generative modeling (not just classification), the paper provides qualitative evidence that the model captures digit structure but no quantitative basis for comparing it to alternative generative models. The classification result demonstrates that the learned representations are informative, but the generative quality remains an open question beyond the visual inspection of Figures 8 and 9.
