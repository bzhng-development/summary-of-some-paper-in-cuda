# An Introduction to Conditional Random Fields

**URL:** [https://homepages.inf.ed.ac.uk/csutton/publications/crftut-fnt.pdf](https://homepages.inf.ed.ac.uk/csutton/publications/crftut-fnt.pdf)

## 🎯 Pitch

This survey introduces conditional random fields (CRFs), a probabilistic framework for structured prediction that combines the representational power of undirected graphical models with the flexibility of discriminative classification to directly model the conditional distribution p(y|x) over output variables given input features.

---

## 1. Executive Summary

This survey introduces conditional random fields (CRFs), a probabilistic framework for structured prediction that combines the representational power of undirected graphical models with the flexibility of discriminative classification to directly model the conditional distribution p(y|x) over output variables given input features. The paper develops the formal connection between CRFs and both logistic regression (for single-variable classification) and hidden Markov models (for sequence modeling), showing that linear-chain CRFs are the discriminative analogue of HMMs while general CRFs extend this principle to arbitrary graphical structures such as grids and trees. Through systematic exposition of modeling, exact inference via forward–backward and belief propagation, and maximum likelihood parameter estimation with L2 and L1 regularization, the survey establishes that CRFs can leverage rich, overlapping input features without modeling their dependencies — an advantage over generative approaches — and provides practical guidance on feature engineering, numerical underflow handling, sparse computation, and approximate training (pseudolikelihood, belief propagation, and MCMC). The paper identifies the label bias problem in directed sequence models like MEMMs as a consequence of v-structure independence assumptions that prevent future observations from influencing earlier state posteriors, establishing that undirected CRFs avoid this pathology and can outperform MEMMs especially when long-range dependencies among output variables are important.

## 2. Context and Motivation

### The Core Problem: Predicting Many Interdependent Variables from Rich Input Features

This survey addresses a fundamental challenge that arises across many application domains: how to predict a vector of output variables $y = \{y_0, y_1, \ldots, y_T\}$ that exhibit complex dependencies among themselves, given an observed feature vector $x$. The paper opens with concrete examples that illustrate why this is not merely a theoretical exercise but a practical necessity:

- In natural language processing, part-of-speech tagging requires assigning grammatical categories to each word in a sentence, where adjacent tags are strongly correlated (adjectives typically precede nouns, determiners rarely follow verbs).
- In computer vision, image segmentation demands labeling each pixel as belonging to a particular object class, where neighboring pixels overwhelmingly share the same label.
- In bioinformatics, gene finding requires identifying coding regions in DNA sequences, where the presence of a start codon implies downstream structural constraints.

The naive approach to this multivariate prediction problem — training an independent classifier for each output position $y_s$ that maps $x \mapsto y_s$ — ignores the dependencies among outputs that are often the most informative signal available. The paper notes that this independence assumption is problematic in practice: in named-entity recognition, for instance, *New York* is a location while *New York Times* is an organization, and the distinction cannot be made by examining each word in isolation.

More fundamentally, output variables may represent complex structures such as parse trees, where a grammatical rule choice near the top of the tree propagates constraints throughout the entire derivation. The challenge, then, is to develop a modeling framework that can simultaneously (1) capture the rich interdependencies among output variables, and (2) leverage high-dimensional, overlapping input features for prediction, while (3) remaining computationally tractable for both learning and inference.

### Why This Problem Matters

The significance of structured prediction extends across scientific and engineering disciplines:

**Real-world impact.** The paper enumerates applications that span natural language processing (part-of-speech tagging, named-entity recognition, shallow parsing, semantic role labeling, word alignment in machine translation, citation extraction, Chinese word segmentation, Japanese morphological analysis), computer vision (image segmentation, object recognition, scene labeling), and bioinformatics (RNA structural alignment, protein structure prediction, gene finding). Each of these domains involves predicting structured outputs — sequences, trees, grids, or general graphs — from high-dimensional observations. The ability to do this accurately has direct economic and scientific consequences: better information extraction from biomedical literature accelerates drug discovery; more reliable image segmentation enables autonomous vehicles and medical image analysis; improved machine translation facilitates cross-lingual communication.

**Theoretical significance.** The paper identifies a deeper conceptual tension that structured prediction exposes: the relationship between generative and discriminative approaches to modeling. Generative models describe a joint distribution $p(y, x) = p(y)p(x|y)$ — they specify how labels probabilistically generate features. Discriminative models directly describe $p(y|x)$ — how to assign labels given features. While Bayes' rule guarantees that either can be converted to the other in principle, the paper argues that this equivalence is misleading in practice because we never possess the true data distribution. The choice between generative and discriminative approaches therefore has consequences for model accuracy, computational tractability, and the types of features that can be incorporated. Understanding when and why discriminative structured models outperform their generative counterparts is a question of both practical importance and theoretical interest.

**Computational challenge.** The paper emphasizes that inference in general graphical models is intractable — any propositional satisfiability problem can be encoded as a factor graph — yet structured prediction requires inference as a subroutine during both training and prediction. This creates a tension between expressive power and computational feasibility that drives much of the design space explored in the survey: when should we restrict model structure (e.g., to linear chains or trees) to enable exact inference, and when should we employ approximate methods to handle more complex dependencies?

### Where Prior Approaches Fall Short

The paper identifies specific limitations in existing approaches along multiple dimensions:

**Generative models and their structural constraints.** Hidden Markov models (HMMs) and other generative approaches model the joint distribution $p(y, x) = p(y)p(x|y)$. The paper traces how this factorization creates difficulties when the input features are high-dimensional and interdependent. The fundamental issue is that modeling $p(x|y)$ requires representing the distribution over inputs — but in many applications, the input features have complex dependencies that are difficult to model tractably. The paper explicitly states:

> "Not only can the dimensionality of $x$ be very large, but the features may have complex dependencies, so constructing a probability distribution over them is difficult. Modeling the dependencies among inputs can lead to intractable models, but ignoring them can lead to reduced performance."

The named-entity recognition example makes this concrete. A generative HMM for NER relies on the word identity as its primary feature. But many words — especially proper names like *Ekeus* or *Baghdad* in the paper's running example — will not appear in the training set, rendering the word-identity feature uninformative at test time. To label unseen words, we need additional features: capitalization patterns, prefix and suffix morphology, neighboring word identities, membership in gazetteers of known locations or person names. A generative model would need to specify a joint distribution over all these interdependent features $p(x|y)$, which is both difficult to specify correctly and computationally expensive to work with.

The paper makes a subtle but crucial observation about why this matters even when the features are seemingly independent. Even the naive Bayes assumption — that features are conditionally independent given the label — can cause problems in structured models. The paper notes that naive Bayes, while often performing well in document classification, performs worse on average across applications than logistic regression, and can produce poorly calibrated probability estimates. The illustrative example is striking: if we duplicate every feature (transforming $x = (x_1, \ldots, x_K)$ to $x' = (x_1, x_1, x_2, x_2, \ldots)$), naive Bayes becomes more confident in its predictions even though no new information is added. In sequence models, this overconfidence at each position compounds, making it difficult to sensibly combine evidence from different parts of the model.

**The independence assumptions in directed sequence models.** A specific alternative to HMMs existed before CRFs: the maximum entropy Markov model (MEMM), described in Section 6.1.3. An MEMM factors as $p(y|x) = \prod_t p(y_t | y_{t-1}, x)$, where each local conditional is a logistic regression model. This is a directed discriminative model — it models the conditional distribution directly, avoiding the need to specify $p(x)$, while maintaining the Markov dependency structure among outputs. Training MEMMs is computationally simpler than training CRFs because the per-time-step normalization constants $Z_t(y_{t-1}, x)$ involve summation only over the label set at a single position, rather than over all possible label sequences.

However, the paper identifies a fundamental pathology called the **label bias problem** that afflicts MEMMs and related directed models. The formal argument proceeds as follows: in the backward recursion for an MEMM, the backward message is:

$$\beta_t(i) = \sum_{j \in S} p(y_{t+1} = j | y_t = i, x_{t+1}) \beta_{t+1}(j)$$

Because the local conditional probabilities must sum to 1 over $j$ for any fixed $i$, if $\beta_{t+1}(j) = 1$ for all $j$ (which holds by induction starting from the end of the sequence), then $\beta_t(i) = 1$ regardless of the current state $i$. The consequence is that future observations provide no information about the current state — a loss of one of the primary advantages of sequence modeling.

The paper provides an illuminating graphical models perspective on this issue. The MEMM graphical structure (Figure 6.1) contains v-structures that imply $y_t$ is marginally independent of future observations $x_{t+1}, x_{t+2}, \ldots$ at all time steps. This is an independence assumption that is "usually strongly violated in sequence modeling." The label bias problem is therefore not an artifact of a particular algorithm but a structural consequence of the directed model's factorization. CRFs avoid this pathology because the undirected normalization constant $Z(x)$ involves summation over entire sequences, allowing information to flow bidirectionally.

**The feature engineering dilemma in generative models.** The paper identifies a deeper architectural tension: to incorporate interdependent, overlapping features into a generative model, one faces two unpalatable choices. The first is to enhance the generative model to represent dependencies among inputs — adding directed edges between input variables $x_t$ — but this is "often difficult to do while retaining tractability," and the paper questions whether we even wish to model such dependencies, since we always observe the inputs at test time anyway. The second is to make simplifying independence assumptions, such as the naive Bayes factorization, which can hurt performance precisely because the independence assumptions are violated.

The paper draws an explicit parallel between this dilemma and the relationship between naive Bayes and logistic regression for simple classification. Naive Bayes and logistic regression form what Ng and Jordan call a generative-discriminative pair: they define the same family of conditional distributions, differing only in whether they parameterize the joint $p(y, x)$ or the conditional $p(y|x)$ directly. The same relationship holds between HMMs and linear-chain CRFs. But the discriminative approach has a key advantage: by modeling $p(y|x)$ directly, one can remain agnostic about the form of $p(x)$. Any factors in a joint model that depend only on $x$ vanish from the conditional distribution's graphical structure — they are constant with respect to $y$ and therefore irrelevant to prediction. This means the conditional model can have a much simpler structure than the joint model while still leveraging rich input features.

**No unified framework for structured discriminative modeling.** Prior to CRFs, the paper argues, there was no systematic framework that combined the advantages of discriminative classification (ability to use large sets of input features without modeling their dependencies) with graphical modeling (ability to compactly represent dependencies among output variables). Individual methods addressed specific aspects — HMMs handled sequential structure but were generative, MEMMs were discriminative but suffered from label bias, logistic regression handled rich features but only for single-variable prediction — but none provided a general, principled approach to structured discriminative modeling.

### How CRFs Position Themselves

The paper positions CRFs as a solution that addresses each of these limitations through a single, coherent framework. The key insight is that CRFs are "essentially a way of combining the advantages of discriminative classification and graphical modeling, combining the ability to compactly model multivariate output $y$ with the ability to leverage a large number of input features $x$ for prediction."

The position is established through a series of explicit relationships:

**CRF as generalization of logistic regression.** Just as logistic regression models the conditional distribution $p(y|x)$ for a single output variable using a log-linear parameterization over feature functions, a CRF extends this to model $p(y|x)$ for a vector of output variables, where the feature functions can depend on arbitrary subsets of the outputs. The simplest CRF — with a single output variable — is exactly the multinomial logistic regression model.

**CRF as discriminative analogue of the HMM.** The paper works through the derivation in detail (Section 2.3). Starting from the HMM joint distribution $p(y, x)$ expressed in exponential-family form with indicator features for transitions and emissions, the conditional distribution $p(y|x)$ that results is precisely a linear-chain CRF — one that uses only word-identity features. The crucial extension is that the CRF allows arbitrary feature functions $f_k(y_t, y_{t-1}, x_t)$ that can examine the entire input sequence, not just the current word. This means the CRF can incorporate prefixes, suffixes, neighboring words, gazetteer membership, and any other observation function that may be informative.

**CRF as general undirected graphical model.** Beyond linear chains, the paper defines a general CRF (Definition 2.3) as any conditional distribution $p(y|x)$ that factorizes according to an undirected factor graph over $Y$ and $X$. This encompasses grid-structured models for image labeling, tree-structured models for parsing, and fully-connected models for relational learning. The only distinction from a standard undirected model is that the partition function $Z(x)$ is input-dependent — but as the paper notes, conditioning on $x$ tends to simplify the graphical structure, making $Z(x)$ potentially computable even when $Z$ (from the joint model) would not be.

The paper's positioning is therefore not to introduce a single new algorithm but to provide a **unifying framework** that (1) formalizes the relationship between discriminative classification and structured prediction, (2) provides algorithms for exact and approximate inference that scale from simple linear chains to complex graphical structures, (3) establishes maximum likelihood parameter estimation with the theoretically appealing property that the gradient matches empirical and model expectations of feature functions, and (4) identifies and resolves the label bias problem that arose in earlier directed discriminative sequence models. The survey's contribution is in systematizing this knowledge — connecting the modeling, inference, and learning perspectives — and providing the practical guidance needed to apply CRFs at scale.

## 3. Technical Approach

This is primarily a **survey and tutorial paper** whose core idea is that conditional random fields (CRFs) provide a unified framework for structured prediction by modeling the conditional distribution $p(y|x)$ directly using undirected graphical models, enabling rich, overlapping input features while avoiding the need to model their dependencies.

### 3.1 Reader Orientation

A CRF is a probability model that, given an input (like a sentence or an image), assigns a score to every possible output (like a sequence of part-of-speech tags or a pixel labeling) based on how well local pieces of that output fit together and match the input, then normalizes those scores into a proper probability distribution. The system solves the problem of predicting many interdependent variables from rich input features — for example, labeling every word in a sentence with its named-entity type — by combining the ability of graphical models to capture dependencies among outputs with the ability of discriminative classifiers to use arbitrary, overlapping features of the input without modeling how those features themselves are distributed.

### 3.2 Big-Picture Architecture (Diagram in Words)

A CRF system has four major components, each operating at different stages of the modeling and deployment pipeline:

**1. The factor graph (model structure).** This defines which subsets of output variables directly interact. In a linear-chain CRF for sequence labeling, consecutive labels $(y_t, y_{t-1})$ are connected, and each label connects to the input features $x_t$. In a grid CRF for image segmentation, neighboring pixel labels in a 2D grid are connected. The graph structure encodes the conditional independence assumptions: any two output variables that are not directly connected are conditionally independent given the variables that separate them. This structure is chosen by the model designer based on domain knowledge — sequences get chains, images get grids, relational data gets arbitrary graphs.

**2. Feature functions (the observation model).** For each factor in the graph, a set of feature functions $f_k(y_c, x_c)$ computes real-valued scores that measure how compatible a particular assignment to the output variables $y_c$ is with the input features $x_c$. In the named-entity example, a feature might fire with value 1 when $y_t = \text{PERSON}$, $y_{t-1} = \text{START}$, and the current word $x_t$ is capitalized. These features are the mechanism by which arbitrary input properties — word identity, capitalization, prefixes, suffixes, gazetteer membership, neighboring words — influence the prediction. The features are designed by the practitioner; the model learns weights for them from data.

**3. The parameter vector (learned weights).** Each feature function $f_k$ has an associated weight $\theta_k$ that the system learns from training data. Positive weights make configurations that satisfy the feature more probable; negative weights make them less probable. These parameters are the only learned component of the system; the graph structure and feature functions are fixed by design.

**4. Inference and learning algorithms (computation).** Given a trained model (structure + features + weights), inference computes either the most probable output $y^* = \arg\max_y p(y|x)$ for a new input (using the Viterbi algorithm for linear chains, or max-product belief propagation for general graphs), or the marginal distributions $p(y_c|x)$ over subsets of output variables needed during training (using the forward–backward algorithm for linear chains, or sum-product belief propagation for general graphs). Learning finds the optimal weights $\theta$ by maximizing the regularized conditional log-likelihood of the training data, which requires running inference as a subroutine at each optimization step to compute feature expectations under the current model.

Information flows through this architecture as follows: training data (pairs of inputs and desired outputs) enters the learning algorithm → the learning algorithm repeatedly calls inference to compute marginal distributions under the current parameters → these marginals are used to compute the gradient of the likelihood, which drives an optimization procedure (L-BFGS, stochastic gradient descent, etc.) to update the parameters → at test time, a new input enters the inference algorithm with the learned parameters fixed → inference produces the most likely output labeling.

### 3.3 Roadmap for the Deep Dive

- **First, the formal definition of a linear-chain CRF (Section 2.3)**, because it is the most widely used CRF architecture and provides the clearest connection to both logistic regression and HMMs. Understanding the linear-chain case makes the generalization to arbitrary structures straightforward, and it is here that the mathematical notation for feature functions, the partition function $Z(x)$, and the log-linear form is established.

- **Second, the formal definition of general CRFs (Section 2.4)** with clique templates and parameter tying, because this is the framework that subsumes linear chains, grids, trees, and arbitrary factor graphs. The concepts of factor graphs, clique templates, and input-dependent normalization are essential for understanding how CRFs scale to complex structured prediction problems.

- **Third, the forward–backward and Viterbi algorithms for exact inference in linear-chain CRFs (Section 4.1)**, because these are the computational workhorses that make training and prediction tractable for sequence models, and they directly generalize the well-known HMM algorithms. Understanding the recursion structure — how the distributive law transforms an exponential sum into a polynomial-time dynamic program — illuminates why tree-structured models are tractable.

- **Fourth, belief propagation for general graphical models (Section 4.2.2)** , because it is the natural generalization of forward–backward to arbitrary tree-structured factor graphs, and its loopy variant provides one of the most widely used approximate inference methods for models where exact inference is intractable. The variational interpretation via the Bethe free energy connects message-passing to optimization, which matters for understanding how approximate inference interacts with parameter estimation.

- **Fifth, maximum likelihood parameter estimation (Section 5.1)** , because this is the standard training procedure for CRFs and the derivation of the gradient reveals the fundamental matching property — at the optimum, feature expectations under the model equal feature expectations under the empirical data distribution. The convexity of the objective (for fully-observed models) and the use of second-order optimization methods like L-BFGS are practical details that determine training efficiency at scale.

- **Sixth, approximate training methods (Section 5.4)** including pseudolikelihood, belief propagation-based surrogate likelihoods, and MCMC-based approximate marginals, because for CRFs with complex graphical structures where exact inference is intractable, these methods trade off approximation error for computational tractability, and the interaction between approximate inference and parameter learning introduces subtle complications that do not arise when both are exact.

### 3.4 Detailed, Sentence-Based Technical Breakdown

---

#### 3.4.1 Linear-Chain CRF Definition and the HMM Connection

The paper builds the linear-chain CRF definition constructively, starting from the familiar HMM and showing that the conditional distribution of an HMM is exactly a CRF with a restricted choice of features.

**The HMM in exponential-family form.** The joint distribution of a homogeneous HMM can be written as:

$$p(y, x) = \frac{1}{Z} \prod_{t=1}^{T} \exp\left( \sum_{i, j \in S} \theta_{ij} \mathbf{1}\{y_t = i\} \mathbf{1}\{y_{t-1} = j\} + \sum_{i \in S} \sum_{o \in O} \mu_{oi} \mathbf{1}\{y_t = i\} \mathbf{1}\{x_t = o\} \right)$$

where $S$ is the set of possible states, $O$ is the set of possible observations, $\theta_{ij} = \log p(y' = i \mid y = j)$ is the log transition probability from state $j$ to state $i$, and $\mu_{oi} = \log p(x = o \mid y = i)$ is the log emission probability of observation $o$ from state $i$. The indicator function $\mathbf{1}\{y_t = i\}$ is 1 when the state at position $t$ equals $i$ and 0 otherwise. This exponential-family parameterization is exactly equivalent to the standard HMM factorization $p(y, x) = \prod_t p(y_t \mid y_{t-1}) p(x_t \mid y_t)$, with the normalization constant $Z = 1$ because the transition and emission probabilities are already properly normalized.

**What this representation computes:** it expresses the HMM joint probability as a product over time steps of exponentiated weighted sums of indicator features — essentially, for each time step, we look up which transition occurred (indicated by exactly one $\mathbf{1}\{y_t = i\}\mathbf{1}\{y_{t-1} = j\}$ being 1) and which observation-state pair occurred, multiply their log-probabilities by 1, exponentiate to recover the probabilities, and multiply across time.

**Why this form:** rewriting the HMM in exponential-family form abstracts away the specific probability tables and reveals that the model is linear in a set of indicator features. The insight is that any distribution that can be written in this form — with arbitrary real-valued weights $\theta_k$ and indicator features — defines a valid model family. The HMM requires $\theta_{ij} = \log p(y' = i \mid y = j)$ and $\mu_{oi} = \log p(x = o \mid y = i)$, i.e., the weights must correspond to log-probabilities that sum to 1 appropriately. But if we relax this requirement — allow arbitrary real-valued weights — we obtain a more flexible family while retaining the same structural factorization.

**Introducing general feature functions.** To generalize beyond indicator features, the paper defines feature functions $f_k(y_t, y_{t-1}, x_t)$ that can be any real-valued function of the current state, previous state, and input. The HMM-like features are a special case: $f_{ij}(y_t, y_{t-1}, x_t) = \mathbf{1}\{y_t = i\}\mathbf{1}\{y_{t-1} = j\}$ for transitions and $f_{io}(y_t, y_{t-1}, x_t) = \mathbf{1}\{y_t = i\}\mathbf{1}\{x_t = o\}$ for emissions. But a CRF can additionally include features like $f(y_t, y_{t-1}, x_t) = \mathbf{1}\{y_t = \text{LOCATION}\} \cdot \mathbf{1}\{x_t \text{ is capitalized}\}$, which fires when the label is LOCATION and the word is capitalized — a feature that has no direct HMM analogue because it couples the transition and emission in a way that depends on a property of the input beyond the word identity.

**The conditional distribution.** From the HMM joint in exponential-family form, computing the conditional $p(y \mid x)$ by dividing by $p(x) = \sum_{y'} p(y', x)$ yields:

$$p(y \mid x) = \frac{\prod_{t=1}^{T} \exp\left( \sum_{k=1}^{K} \theta_k f_k(y_t, y_{t-1}, x_t) \right)} {\sum_{y'} \prod_{t=1}^{T} \exp\left( \sum_{k=1}^{K} \theta_k f_k(y'_t, y'_{t-1}, x_t) \right)}$$

where $K$ is the total number of feature functions (spanning both transition and emission features), $\theta_k$ is the weight for feature $k$, and $f_k$ is the $k$-th feature function. The numerator is the unnormalized score of the specific label sequence $y$ given observation $x$. The denominator is the sum of unnormalized scores over all possible label sequences $y'$, which serves as the input-dependent normalization constant.

**What it computes:** given an observation sequence $x$ and a candidate label sequence $y$, the numerator multiplies across time steps the exponentiated weighted sum of all features that are active at each step — essentially, a score that accumulates evidence from transitions and observations. The denominator sums this score over every possible label sequence of length $T$, converting the scores into a proper probability distribution over sequences.

**Why this form:** the key difference from the HMM is that the normalization is global — $Z(x)$ sums over all sequences — rather than local (per-time-step). This means that the CRF does not require the local factors to be individually normalized as probabilities. The weights $\theta_k$ are unconstrained real numbers, learned from data to maximize the conditional likelihood of the training sequences. This freedom allows the model to learn correlations between features and labels without being constrained by the requirement that transition weights correspond to valid probability distributions. The global normalization is what prevents label bias: information can flow bidirectionally because the score of a particular label at position $t$ is influenced by the normalization over all possible sequences, which in turn depends on future observations.

**Formal linear-chain CRF definition (Definition 2.2).** Let $Y, X$ be random vectors, $\theta = \{\theta_k\} \in \mathbb{R}^K$ be a parameter vector, and $\mathcal{F} = \{f_k(y, y', x_t)\}_{k=1}^{K}$ be a set of real-valued feature functions. Then a linear-chain conditional random field is a distribution $p(y \mid x)$ that takes the form:

$$p(y \mid x) = \frac{1}{Z(x)} \prod_{t=1}^{T} \exp\left( \sum_{k=1}^{K} \theta_k f_k(y_t, y_{t-1}, x_t) \right)$$

where $Z(x)$ is the input-dependent normalization function:

$$Z(x) = \sum_{y} \prod_{t=1}^{T} \exp\left( \sum_{k=1}^{K} \theta_k f_k(y_t, y_{t-1}, x_t) \right)$$

Here $\sum_y$ denotes summation over all $|S|^T$ possible state sequences of length $T$, where $|S|$ is the number of possible labels. The observation argument $x_t$ in $f_k$ is written as a vector to indicate that the feature function can examine not just the observation at time $t$ but any components of the global observation sequence $x$ that are relevant — for example, the word at time $t+1$ or $t-1$, or features spanning multiple time steps.

**The factor graph representation.** The linear-chain CRF can be expressed as a factor graph:

$$p(y \mid x) = \frac{1}{Z(x)} \prod_{t=1}^{T} \Psi_t(y_t, y_{t-1}, x_t)$$

where each local factor has the log-linear form:

$$\Psi_t(y_t, y_{t-1}, x_t) = \exp\left( \sum_{k=1}^{K} \theta_k f_k(y_t, y_{t-1}, x_t) \right)$$

This factorization reveals the graphical structure: the model is a chain where each factor $\Psi_t$ connects two consecutive labels $y_t$ and $y_{t-1}$ and depends on the input $x_t$. The factors are unnormalized — they can be any non-negative value — and the global normalization $Z(x)$ ensures a valid probability distribution.

**Why this factorization matters:** it directly exposes the conditional independence properties of the model. In a linear-chain CRF, $y_t$ is conditionally independent of $y_1, \ldots, y_{t-2}$ given $y_{t-1}$ (and $x$). This is the same Markov property as an HMM, but with the crucial difference that it conditions on the entire input sequence $x$. The factorization also determines the computational complexity of inference: because the graph is a chain (a tree), exact inference via dynamic programming is possible in $O(T|S|^2)$ time.

**Observations as a single monolithic variable.** The paper notes an important representational point: since CRFs do not model dependencies among the input variables $x_1, \ldots, x_T$, we can treat $x$ as a single, large observed variable on which all factors depend (shown graphically in Figure 2.7). This means the feature functions can examine the entire input sequence at once — $f_k(y_t, y_{t-1}, x)$ can depend on $x_{t+1}$, $x_{t-2}$, or any global property of $x$ — without breaking the linear graphical structure among the outputs. This is a major advantage over generative models, where such long-range input dependencies would need to be explicitly modeled in the joint distribution.

---

#### 3.4.2 General CRFs: Clique Templates and Parameter Tying

**Formal general CRF definition (Definition 2.3).** Let $G$ be a factor graph over variables $X$ and $Y$. Then $(X, Y)$ is a conditional random field if for any value $x$ of $X$, the distribution $p(y \mid x)$ factorizes according to $G$. If $\mathcal{F} = \{\Psi_a\}$ is the set of factors in $G$, then:

$$p(y \mid x) = \frac{1}{Z(x)} \prod_{a=1}^{A} \Psi_a(y_a, x_a)$$

where $A$ is the number of factors, each $\Psi_a$ depends on a subset of output variables $Y_a \subseteq Y$ and input variables $X_a \subseteq X$, and the normalization is:

$$Z(x) = \sum_{y} \prod_{a=1}^{A} \Psi_a(y_a, x_a)$$

**What this computes:** exactly like the linear-chain case but for an arbitrary factor graph structure. Each factor $\Psi_a$ scores a local configuration of output variables based on local input features. The product over all factors gives an unnormalized score for the complete assignment $y$. The partition function $Z(x)$ sums this score over all possible assignments to the entire output vector $y$, ensuring $p(y \mid x)$ is a valid probability distribution.

**Why this form:** conditioning on $x$ is the key. The factors $\Psi_a(y_a, x_a)$ can depend on arbitrary subsets of the input variables without modeling their joint distribution. The paper makes the observation that conditioning tends to simplify graphical model structure: factors that depend only on $x$ vanish from the conditional distribution because they are constant with respect to $y$. This means $Z(x)$ might be computable even when the partition function of the corresponding joint model $p(y, x)$ would not be, because the graphical structure among $y$ given $x$ can be simpler than the structure of the full joint distribution.

**Log-linear factors.** As with linear-chain CRFs, it is standard practice to require each factor to be log-linear in a set of preset feature functions:

$$\Psi_a(y_a, x_a) = \exp\left( \sum_{k=1}^{K(A)} \theta_{ak} f_{ak}(y_a, x_a) \right)$$

where $K(A)$ is the number of feature functions for factor $a$, $\theta_{ak}$ are the weights for factor $a$, and $f_{ak}$ are the feature functions for factor $a$. The subscript $a$ on the weights and features emphasizes that each factor can have its own distinct set of feature functions. However, the paper notes that when $x$ and $y$ are discrete, the log-linear assumption imposes no restriction: we can always choose indicator features for every possible configuration $(y_a, x_a)$, recovering a fully general tabular factor.

**Clique templates for parameter tying.** In practice, factors in a CRF are organized into **clique templates**, which are sets of factors that share the same parameters and feature functions. This is motivated by the observation that in structured models, the same types of local interactions recur across the graph — for example, every time step in a linear chain has a transition factor between consecutive labels, and it makes sense for these factors to use the same weights.

Let $\mathcal{C} = \{C_1, C_2, \ldots, C_P\}$ be a set of clique templates, where each $C_p$ is a set of factors sharing feature functions $\{f_{pk}(x_c, y_c)\}_{k=1}^{K(p)}$ and parameters $\theta_p \in \mathbb{R}^{K(p)}$. A CRF with clique templates is written as:

$$p(y \mid x) = \frac{1}{Z(x)} \prod_{C_p \in \mathcal{C}} \prod_{\Psi_c \in C_p} \Psi_c(x_c, y_c; \theta_p)$$

where:

$$\Psi_c(x_c, y_c; \theta_p) = \exp\left( \sum_{k=1}^{K(p)} \theta_{pk} f_{pk}(x_c, y_c) \right)$$

and:

$$Z(x) = \sum_{y} \prod_{C_p \in \mathcal{C}} \prod_{\Psi_c \in C_p} \Psi_c(x_c, y_c; \theta_p)$$

**What clique templates compute:** they factor the model into repeated instances of the same local pattern. For a linear-chain CRF, typically one clique template $C_0 = \{\Psi_t(y_t, y_{t-1}, x_t)\}_{t=1}^{T}$ is used for the entire network — so $\mathcal{C} = \{C_0\}$ is a singleton set. The parameters $\theta_0$ are shared across all time steps: the weight for the feature "transition from B-PER to I-PER" is the same whether it occurs at position 3 or position 15. For a non-homogeneous model where each time step has its own parameters, we would use $T$ templates: $C_t = \{\Psi_t(y_t, y_{t-1}, x_t)\}$ for $t = 1, \ldots, T$.

**Why parameter tying:** sharing parameters across instances of the same local pattern dramatically reduces the number of parameters, preventing overfitting and enabling the model to generalize across positions. A linear-chain CRF with homogeneous parameters applies the same transition and emission logic everywhere in the sequence, just as an HMM does. The clique template formalism generalizes this idea to arbitrary graph structures: in a grid CRF for images, the same edge parameters are shared across all neighboring pixel pairs; in a relational Markov network, the same parameters are shared across all instances of a particular relationship type.

**The template-counting form of the distribution.** An alternative way to write the CRF that is useful for understanding the likelihood gradient is:

$$p(y \mid x) = \frac{1}{Z(x)} \exp\left( \sum_{C_p \in \mathcal{C}} \sum_{\Psi_c \in C_p} \sum_{k=1}^{K(p)} \theta_{pk} f_{pk}(x_c, y_c) \right)$$

This form collects all feature functions across all factors into a single exponential, making the linearity in $\theta$ explicit. The inner double sum — over factors in a template and over feature indices — aggregates the total activation of each feature across the entire model. This aggregated form is what makes the gradient computation efficient: the gradient with respect to $\theta_{pk}$ is simply the difference between the observed total activation of feature $f_{pk}$ and its expected total activation under the current model.

**Design choices in specifying clique templates.** The paper surveys several formalisms for specifying the repeated structure and parameter tying: dynamic conditional random fields for multi-label sequences, relational Markov networks with SQL-like syntax, Markov logic networks using first-order logical formulae to specify factor scopes and parameter tying, and imperatively defined factor graphs that use Turing-complete functions to define templates. The common thread is that the clique template concept — a set of factors sharing parameters — abstracts away the specific syntax and captures the essential structure needed for both modeling and implementation.

---

#### 3.4.3 Forward–Backward Algorithm for Linear-Chain CRF Inference

The forward–backward algorithm computes the partition function $Z(x)$ and the marginal distributions $p(y_{t-1}, y_t \mid x)$ and $p(y_t \mid x)$ for a linear-chain CRF. It is a dynamic programming algorithm that exploits the chain structure to avoid summing over exponentially many sequences, reducing the computation from $O(|S|^T)$ to $O(T|S|^2)$, where $|S|$ is the number of labels and $T$ is the sequence length.

**The key insight: distributive law.** The naive computation of the partition function sums over all label sequences:

$$Z(x) = \sum_{y} \prod_{t=1}^{T} \Psi_t(y_t, y_{t-1}, x_t)$$

This is a sum of $|S|^T$ terms, each a product of $T$ factors. The distributive law allows us to push the sums inside the product:

$$Z(x) = \sum_{y_T} \sum_{y_{T-1}} \Psi_T(y_T, y_{T-1}, x_T) \sum_{y_{T-2}} \Psi_{T-1}(y_{T-1}, y_{T-2}, x_{T-1}) \sum_{y_{T-3}} \cdots$$

Each intermediate sum is computed once and reused many times. This transforms the exponential computation into $T$ steps, each summing over $|S|$ values of the current state and $|S|$ values of the previous state, yielding $O(T|S|^2)$ total time.

**Forward recursion.** The forward variables $\alpha_t(j)$ are defined as the unnormalized marginal probability of the observation subsequence up to time $t$ and the state at time $t$ being $j$, summing over all possible state sequences up to $t-1$:

$$\alpha_t(j) = \sum_{y_{\langle 1 \ldots t-1 \rangle}} \Psi_t(j, y_{t-1}, x_t) \prod_{t'=1}^{t-1} \Psi_{t'}(y_{t'}, y_{t'-1}, x_{t'})$$

where $y_{\langle 1 \ldots t-1 \rangle}$ denotes all assignments to $y_1, y_2, \ldots, y_{t-1}$. In an HMM, $\alpha_t(j)$ has the probabilistic interpretation $\alpha_t(j) = p(x_{\langle 1 \ldots t \rangle}, y_t = j)$. In a CRF, this probabilistic interpretation does not hold because the factors are not locally normalized — instead, the $\alpha$ variables are simply intermediate computational quantities that accumulate unnormalized scores.

The forward recursion computes $\alpha_t$ iteratively:

$$\alpha_t(j) = \sum_{i \in S} \Psi_t(j, i, x_t) \alpha_{t-1}(i)$$

with initialization $\alpha_1(j) = \Psi_1(j, y_0, x_1)$, where $y_0$ is a fixed dummy start state. Each step multiplies the previous forward values by the transition weights from all previous states to the current state $j$, and sums. The partition function is recovered as $Z(x) = \sum_{i \in S} \alpha_T(i)$.

**Backward recursion.** The backward variables $\beta_t(i)$ are defined symmetrically, summing over futures rather than pasts:

$$\beta_t(i) = \sum_{y_{\langle t+1 \ldots T \rangle}} \prod_{t'=t+1}^{T} \Psi_{t'}(y_{t'}, y_{t'-1}, x_{t'})$$

with initialization $\beta_T(i) = 1$ for all $i$. The backward recursion is:

$$\beta_t(i) = \sum_{j \in S} \Psi_{t+1}(j, i, x_{t+1}) \beta_{t+1}(j)$$

and the partition function can also be computed as $Z(x) = \beta_0(y_0) = \sum_{y_1} \Psi_1(y_1, y_0, x_1) \beta_1(y_1)$.

**Computing marginals.** The marginal distribution over a pair of consecutive labels is obtained by combining forward and backward variables with the local factor:

$$p(y_{t-1}, y_t \mid x) = \frac{1}{Z(x)} \alpha_{t-1}(y_{t-1}) \Psi_t(y_t, y_{t-1}, x_t) \beta_t(y_t)$$

The single-variable marginal is obtained by summing out one variable:

$$p(y_t \mid x) = \frac{1}{Z(x)} \alpha_t(y_t) \beta_t(y_t)$$

**What the marginals compute:** $p(y_{t-1}, y_t \mid x)$ is the probability that, given the entire observation sequence $x$, the labels at positions $t-1$ and $t$ take particular values, properly normalized over all possible sequences. These pairwise marginals are exactly what the likelihood gradient requires — the expected count of each transition and emission feature under the current model parameters.

**Why this algorithm:** the forward–backward recursion is the only tractable way to compute exact marginals in sequence models. The alternative — enumerating all $|S|^T$ sequences — is infeasible for any realistic sequence length and label set size. The dynamic programming exploits the chain structure: the state at time $t$ summarizes all information from the past needed to compute future probabilities. This Markov property is what makes the recursion valid and efficient.

**Connection to the CRF definition.** The only change from the HMM forward–backward is the definition of the transition factors $\Psi_t$. In an HMM, $\Psi_t(j, i, x_t) = p(y_t = j \mid y_{t-1} = i) p(x_t = x \mid y_t = j)$, which are probabilities that sum to 1. In a CRF, $\Psi_t(j, i, x_t) = \exp(\sum_k \theta_k f_k(j, i, x_t))$, which are unnormalized non-negative scores. The algorithmic structure — the recursion equations — is identical.

---

#### 3.4.4 Viterbi Algorithm for Most-Probable Labeling

The Viterbi algorithm computes the single most probable label sequence $y^* = \arg\max_y p(y \mid x)$. It is the max-product analogue of the sum-product forward–backward algorithm: replace all summations with maximization.

**The Viterbi recursion.** Define $\delta_t(j)$ as the maximum unnormalized score of any state sequence ending in state $j$ at time $t$:

$$\delta_t(j) = \max_{y_{\langle 1 \ldots t-1 \rangle}} \Psi_t(j, y_{t-1}, x_t) \prod_{t'=1}^{t-1} \Psi_{t'}(y_{t'}, y_{t'-1}, x_{t'})$$

The recursion is:

$$\delta_t(j) = \max_{i \in S} \Psi_t(j, i, x_t) \delta_{t-1}(i)$$

with initialization $\delta_1(j) = \Psi_1(j, y_0, x_1)$. At each step, instead of summing over previous states, we take the maximum.

After computing $\delta_T(j)$ for all $j$, the optimal final state is $y_T^* = \arg\max_{i \in S} \delta_T(i)$. The full optimal sequence is recovered by backtracking: for each $t$ from $T-1$ down to 1, $y_t^*$ is the state $i$ that achieved the maximum in the computation of $\delta_{t+1}(y_{t+1}^*)$.

**What it computes:** the single label sequence $y^*$ that maximizes $p(y \mid x)$ — the MAP (maximum a posteriori) assignment. Since $Z(x)$ is constant with respect to $y$, maximizing $p(y \mid x)$ is equivalent to maximizing the numerator $\prod_t \Psi_t(y_t, y_{t-1}, x_t)$, so the partition function need not be computed.

**Why this algorithm:** the Viterbi algorithm finds the globally optimal sequence in $O(T|S|^2)$ time, exactly the same complexity as forward–backward. It exploits the same optimal substructure: the best sequence ending in state $j$ at time $t$ must extend the best sequence ending in some state $i$ at time $t-1$. This holds because the score decomposes as a sum over time steps, so decisions about the past do not affect future scores except through the current state.

**The semiring perspective.** The paper notes that forward–backward and Viterbi can be seen as the same algorithm instantiated on two different semirings: forward–backward uses the sum-product semiring $(\mathbb{R}, +, \times, 0, 1)$, while Viterbi uses the max-product semiring $(\mathbb{R}, \max, \times, 0, 1)$. This algebraic connection is more than a curiosity — it means that any implementation of the forward–backward recursion can be converted to Viterbi simply by replacing addition with maximization.

---

#### 3.4.5 Belief Propagation for General Graphical Models

Belief propagation (BP) generalizes the forward–backward algorithm from linear chains to arbitrary tree-structured factor graphs. The algorithm computes exact marginals by passing messages — which are functions over single variables — along the edges of the factor graph.

**Messages from factors to variables.** For a factor $a$ connected to variable $s$, the message $m_{as}(y_s)$ summarizes the influence of the entire subgraph upstream of factor $a$ on the marginal distribution of $Y_s$:

$$m_{as}(y_s) = \sum_{y_a \setminus y_s} \Psi_a(y_a) \prod_{t \in N(a) \setminus \{s\}} m_{ta}(y_t)$$

where $N(a)$ is the set of variable indices that are neighbors of factor $a$, $y_a \setminus y_s$ denotes all variables in $y_a$ except $y_s$, the summation is over all assignments to those variables, $\Psi_a(y_a)$ is the local factor, and $m_{ta}(y_t)$ for $t \neq s$ are the messages from the other neighboring variables to factor $a$.

**Messages from variables to factors.** The message from a variable $s$ to a factor $a$ is simply the product of all incoming messages from other factors:

$$m_{sa}(y_s) = \prod_{b \in N(s) \setminus \{a\}} m_{bs}(y_s)$$

where $N(s)$ is the set of factors connected to variable $s$. This message passes along information from all factors incident to variable $s$ except the recipient $a$ (to avoid double-counting).

**Computing marginals from messages.** The belief (marginal) at a variable is proportional to the product of all incoming messages to that variable:

$$p(y_s) \propto \prod_{a \in N(s)} m_{as}(y_s)$$

The belief at a factor is proportional to the product of the local factor and all incoming messages to that factor:

$$p(y_a) \propto \Psi_a(y_a) \prod_{s \in N(a)} m_{sa}(y_s)$$

The proportionality constant is determined by requiring the beliefs to sum to 1.

**Computing the partition function.** For tree-structured models, BP can compute $Z$ directly from the messages. Alternatively, once all singleton and factor marginals are known, the joint distribution of any assignment can be reconstructed using the factorization identity:

$$p(y) = \prod_{s \in V} p(y_s) \prod_{a} \frac{p(y_a)}{\prod_{t \in N(a)} p(y_t)}$$

Then $Z = p(y)^{-1} \prod_a \Psi_a(y_a)$ for any assignment $y$. For linear chains, this identity reduces to the familiar $p(y) = \prod_t p(y_t \mid y_{t-1})$.

**Message schedule.** In a tree, messages can be scheduled so that each message is computed exactly once. A standard approach is to pick an arbitrary root, propagate messages from leaves to root (the "collect" phase), and then propagate back from root to leaves (the "distribute" phase). After these two passes, all variable and factor marginals are exact.

**What BP computes:** for a tree-structured factor graph, BP computes the exact marginal distributions of every variable and every factor, along with the partition function. The computation is linear in the number of factors, but the cost per message depends on the size of the factor's domain. For pairwise factors over $M$-valued variables, each factor-to-variable message costs $O(M^2)$.

**Why BP on trees:** the factorization of the marginal computation into independent subproblems (illustrated in Figure 4.1) is only possible because the graph is a tree. In a tree, removing a factor node disconnects the graph into components that are independent given the variables adjacent to that factor. This conditional independence is what allows the messages to be computed independently and combined multiplicatively.

**Loopy belief propagation.** When the factor graph contains cycles, the message-passing equations are no longer guaranteed to compute exact marginals, but they can be iterated as an approximate inference procedure. In loopy BP, messages are initialized (typically to uniform) and then repeatedly updated using the same formulas until (hopefully) convergence. The schedule for updates — the order in which messages are recomputed — affects both the convergence behavior and the quality of the resulting approximate marginals. The paper mentions randomized schedules and more sophisticated approaches as practical heuristics.

**Variational interpretation.** The paper presents the variational perspective on loopy BP to provide theoretical grounding. The Bethe free energy is an approximation to the exact variational free energy:

$$O_{\text{Bethe}}(q) = -H_{\text{Bethe}}(q) - \sum_a \sum_{y_a} q(y_a) \log \Psi_a(y_a)$$

where $H_{\text{Bethe}}$ is the Bethe approximation to the entropy, defined for a set of pseudomarginals $q(y_a)$ and $q(y_s)$ that are only required to be locally consistent (each factor marginal must agree with the singleton marginals on shared variables, but there need not exist a global joint distribution with these marginals):

$$H_{\text{Bethe}}(q) = -\sum_a \sum_{y_a} q(y_a) \log q(y_a) + \sum_i \sum_{y_i} (d_i - 1) q(y_i) \log q(y_i)$$

where $d_i$ is the degree of variable node $i$. For tree-structured distributions, $H_{\text{Bethe}}$ is exact. For loopy graphs, it is an approximation.

**What the variational perspective reveals:** stationary points of the Bethe free energy under local consistency constraints correspond to fixed points of loopy BP. This connection means that (1) loopy BP can be understood as an optimization algorithm rather than just a heuristic message-passing scheme, (2) the Bethe free energy evaluated at a BP fixed point provides an approximation to $\log Z$, which is useful for approximate parameter estimation, and (3) the quality of BP approximate marginals can be understood in terms of how well the Bethe free energy approximates the true free energy.

---

#### 3.4.6 Maximum Likelihood Parameter Estimation for Linear-Chain CRFs

**The conditional log-likelihood.** Given i.i.d. training data $\mathcal{D} = \{x^{(i)}, y^{(i)}\}_{i=1}^{N}$, where each $x^{(i)}$ is an input sequence and each $y^{(i)}$ is the corresponding label sequence, the conditional log-likelihood is:

$$\ell(\theta) = \sum_{i=1}^{N} \log p(y^{(i)} \mid x^{(i)}; \theta)$$

$$= \sum_{i=1}^{N} \sum_{t=1}^{T} \sum_{k=1}^{K} \theta_k f_k(y_t^{(i)}, y_{t-1}^{(i)}, x_t^{(i)}) - \sum_{i=1}^{N} \log Z(x^{(i)})$$

**What this computes:** the first term is the total unnormalized score of the true label sequence, summed over all positions and all features. The second term is the log partition function — the log of the sum of unnormalized scores over all possible label sequences — which acts as a penalty: increasing the score of the true sequence helps, but the partition function also increases because it sums over all sequences, so the net effect on the likelihood depends on whether the true sequence's score increases more than the scores of competing sequences.

**Regularization.** To prevent overfitting when there are many parameters (the paper gives examples with hundreds of thousands of features), a regularization penalty is added. For $L_2$ regularization:

$$\ell(\theta) = \sum_{i=1}^{N} \sum_{t=1}^{T} \sum_{k=1}^{K} \theta_k f_k(y_t^{(i)}, y_{t-1}^{(i)}, x_t^{(i)}) - \sum_{i=1}^{N} \log Z(x^{(i)}) - \sum_{k=1}^{K} \frac{\theta_k^2}{2\sigma^2}$$

where $\sigma^2$ is a free parameter controlling the strength of the penalty. The paper notes that $\sigma^2 = 10$ is often a reasonable default. For $L_1$ regularization:

$$\ell'(\theta) = \sum_{i=1}^{N} \sum_{t=1}^{T} \sum_{k=1}^{K} \theta_k f_k(y_t^{(i)}, y_{t-1}^{(i)}, x_t^{(i)}) - \sum_{i=1}^{N} \log Z(x^{(i)}) - \alpha \sum_{k=1}^{K} |\theta_k|$$

where $\alpha$ controls the $L_1$ penalty strength. The $L_1$ regularizer encourages sparsity — many weights become exactly zero — which performs automatic feature selection. The paper notes that $L_1$-regularized models typically have comparable accuracy to $L_2$ but with far fewer non-zero parameters (sometimes only 1% of the full feature set).

**Interpretation as MAP estimation.** The $L_2$ regularized likelihood corresponds to maximum a posteriori (MAP) estimation with a Gaussian prior $\theta_k \sim \mathcal{N}(0, \sigma^2)$ on each weight. This Bayesian interpretation provides intuition: the regularizer encodes the belief that weights should be close to zero unless the data strongly suggests otherwise.

**The likelihood gradient.** The partial derivative with respect to weight $\theta_k$ is:

$$\frac{\partial \ell}{\partial \theta_k} = \sum_{i=1}^{N} \sum_{t=1}^{T} f_k(y_t^{(i)}, y_{t-1}^{(i)}, x_t^{(i)}) - \sum_{i=1}^{N} \sum_{t=1}^{T} \sum_{y, y'} f_k(y, y', x_t^{(i)}) p(y, y' \mid x^{(i)}) - \frac{\theta_k}{\sigma^2}$$

**What this gradient computes:** the first term is the empirical count of feature $f_k$ — the total number of times this feature fires in the training data. The second term is the expected count under the current model — the probability-weighted average of how many times the feature fires across all possible label sequences, weighted by how probable each sequence is according to the current parameter values. The third term is the regularization gradient, which pushes each weight toward zero.

**The matching property.** At the unregularized maximum likelihood solution, the gradient is zero, which means:

$$\text{Empirical count of } f_k = \text{Expected count of } f_k \text{ under the model}$$

This is a standard property of exponential family distributions: maximum likelihood chooses parameters such that the sufficient statistics of the model match those of the data. The matching conditions for a linear-chain CRF are exactly the system of equations:

$$\sum_{i=1}^{N} \sum_{t=1}^{T} f_k(y_t^{(i)}, y_{t-1}^{(i)}, x_t^{(i)}) = \sum_{i=1}^{N} \sum_{t=1}^{T} \sum_{y, y'} f_k(y, y', x_t^{(i)}) p(y_t = y, y_{t-1} = y' \mid x^{(i)})$$

**Why this gradient form:** the exponential family structure guarantees that the log-likelihood gradient is the difference between empirical and expected sufficient statistics. This is computationally fortunate: computing the gradient requires the same marginal distributions $p(y_t, y_{t-1} \mid x^{(i)})$ that the forward–backward algorithm provides. There is no need for second-order derivatives of the partition function to compute the gradient.

**Convexity.** The log-likelihood $\ell(\theta)$ is a concave function of $\theta$. This follows from the convexity of the log-sum-exp function: $\log Z(x) = \log \sum_y \exp(\text{score}(y, x))$ is convex in the parameters (since it is the composition of log-sum-exp, which is convex, with an affine function of $\theta$). The negative log-likelihood is therefore convex, and gradient-based optimization is guaranteed to find the global maximum. With $L_2$ regularization, the objective becomes strictly concave, guaranteeing a unique optimum. With $L_1$ regularization, the objective is concave but not strictly concave, so multiple optima may exist (though they tend to be sparse).

**Why this property matters:** convexity eliminates the problem of local minima that plagues neural networks and latent-variable models. The optimization landscape has a single basin of attraction, and any reasonable optimization algorithm will find the global optimum given enough computation. This is one of the major practical advantages of fully-observed CRFs over alternative structured prediction approaches.

**Optimization algorithms.** The paper describes the computational tradeoffs among several optimization methods:

- **Steepest ascent (gradient ascent):** follows the gradient with a step size. Simple but requires too many iterations to be practical for CRFs, because the objective is often ill-conditioned (the curvature varies greatly across parameter dimensions).

- **Newton's method:** uses the full Hessian (matrix of second derivatives) to account for curvature. Converges quadratically near the optimum, but the Hessian is $K \times K$ where $K$ is the number of parameters (potentially millions), so storing it is infeasible.

- **Quasi-Newton methods (L-BFGS):** build an approximation to the Hessian from gradient history, requiring only $O(K)$ storage rather than $O(K^2)$. Limited-memory BFGS (L-BFGS) maintains a low-rank approximation to the inverse Hessian by storing the past $m$ gradient and parameter differences (typically $m = 3$ to $10$). This provides much of the benefit of second-order methods without the memory cost. L-BFGS is the recommended approach for batch CRF training.

- **Conjugate gradient:** another method that uses second-order information implicitly without storing a Hessian approximation. It chooses search directions that are conjugate with respect to the Hessian, ensuring that progress made in one direction is not undone by subsequent steps.

The paper notes that these second-order and quasi-Newton methods are substantially faster than the iterative scaling methods originally used in Laﬀerty et al. (2001), as demonstrated experimentally by several authors.

**Computational cost of training.** Each gradient computation requires running forward–backward for each training instance, which takes $O(T M^2)$ time per instance, where $M = |S|$ is the number of labels and $T$ is the sequence length. With $N$ training instances and $G$ gradient evaluations, the total training time is $O(T M^2 N G)$. The paper reports that for L-BFGS on linear-chain CRFs, $G$ is often (but not always) under 100. For the example tasks in Table 5.1, training times range from 958 seconds (NP chunking, 3 labels) to 325,500 seconds (POS tagging, 45 labels), showing the strong dependence on the number of labels.

---

#### 3.4.7 Maximum Likelihood for General CRFs and Latent Variables

**Fully-observed general CRFs.** For general CRFs with clique templates, the conditional log-likelihood is:

$$\ell(\theta) = \sum_{C_p \in \mathcal{C}} \sum_{\Psi_c \in C_p} \sum_{k=1}^{K(p)} \theta_{pk} f_{pk}(x_c, y_c) - \log Z(x)$$

where the notation sums over all instantiations of all clique templates. The gradient for parameter $\theta_{pk}$ is:

$$\frac{\partial \ell}{\partial \theta_{pk}} = \sum_{\Psi_c \in C_p} f_{pk}(x_c, y_c) - \sum_{\Psi_c \in C_p} \sum_{y'_c} f_{pk}(x_c, y'_c) p(y'_c \mid x)$$

The first term is the empirical count (summing over all factor instances matching template $C_p$). The second term is the expected count under the current model, which requires computing marginal distributions $p(y'_c \mid x)$ over the subsets of variables that are the domains of each factor.

**What changes from the linear-chain case:** the only difference is that computing the expected counts requires more general inference algorithms — belief propagation for tree-structured models, or approximate methods (loopy BP, MCMC) for models with cycles. The functional form of the likelihood and gradient remains exactly the same as in the log-linear parameterization.

**Latent-variable CRFs.** When some output variables are unobserved even during training, the paper introduces hidden-state CRFs. The model factorizes as:

$$p(y, w \mid x) = \frac{1}{Z(x)} \prod_{C_p \in \mathcal{C}} \prod_{\Psi_c \in C_p} \Psi_c(x_c, w_c, y_c; \theta_p)$$

where $y$ are the observed output variables, $w$ are the latent (hidden) variables. The training objective is the marginal likelihood:

$$\ell(\theta) = \log p(y \mid x) = \log \sum_w p(y, w \mid x)$$

**Computing the marginal likelihood.** The key trick is to recognize that:

$$p(y \mid x) = \frac{Z(y, x)}{Z(x)}$$

where $Z(y, x)$ is the partition function of the CRF with the observed variables $y$ clamped to their training values:

$$Z(y, x) = \sum_w \prod_{C_p \in \mathcal{C}} \prod_{\Psi_c \in C_p} \Psi_c(x_c, w_c, y_c; \theta_p)$$

$Z(y, x)$ sums only over the latent variables $w$, which can be substantially easier than summing over both $w$ and $y$ as in $Z(x)$.

**What this form achieves:** it reduces the problem of computing the marginal likelihood to two partition function computations — one over the clamped model (summing only over $w$) and one over the full model (summing over both $w$ and $y$). Both can be approached using the same inference machinery as fully-observed CRFs.

**The gradient for latent-variable CRFs.** Using the identity $\frac{df}{d\theta} = f(\theta) \frac{d \log f}{d\theta}$ for any differentiable $f$:

$$\frac{\partial \ell}{\partial \theta_{pk}} = \sum_w p(w \mid y, x) \frac{\partial}{\partial \theta_{pk}} [\log p(y, w \mid x)]$$

This is the expectation of the fully-observed gradient, where the expectation is taken with respect to the posterior distribution over the latent variables given the observed variables. Expanding:

$$\frac{\partial \ell}{\partial \theta_{pk}} = \sum_{\Psi_c \in C_p} \sum_{w'_c} p(w'_c \mid y, x) f_{pk}(y_c, x_c, w'_c) - \sum_{\Psi_c \in C_p} \sum_{w'_c, y'_c} p(w'_c, y'_c \mid x_c) f_{pk}(y'_c, x_c, w'_c)$$

The first term requires marginals from the clamped CRF $p(w \mid y, x)$. The second term requires marginals from the full CRF $p(w, y \mid x)$. Both sets of marginals can be computed using the same inference algorithms as before.

**The non-convexity challenge.** Unlike fully-observed CRFs, the marginal likelihood $\ell(\theta)$ for latent-variable models is generally not concave. This means optimization can get stuck in local maxima, and the quality of the solution depends on initialization. The paper notes that "the model parameters must be carefully initialized in order to reach a good local maximum."

**EM as an alternative optimization strategy.** The Expectation-Maximization algorithm iteratively optimizes the marginal likelihood. In the E-step, the posterior $q(w) = p(w \mid y, x; \theta^{(j)})$ is computed using the current parameters. In the M-step, new parameters are chosen to maximize the expected complete-data log-likelihood:

$$\theta^{(j+1)} = \arg\max_{\theta'} \sum_{w'} q(w') \log p(y, w' \mid x; \theta')$$

The paper observes that the gradient of the EM auxiliary function is almost identical to the direct gradient, with the only difference being that the posterior $p(w \mid y, x)$ is from a fixed previous parameter setting rather than differentiated through.

---

#### 3.4.8 Stochastic Gradient Descent for CRF Training

Stochastic gradient descent (SGD) exploits the i.i.d. structure of training data by updating parameters after each training example rather than waiting for a full pass over the dataset.

**Per-instance gradient.** For a single training instance $(x^{(i)}, y^{(i)})$, the gradient is:

$$\frac{\partial \ell_i}{\partial \theta_k} = \sum_{t=1}^{T} f_k(y_t^{(i)}, y_{t-1}^{(i)}, x_t^{(i)}) - \sum_{t=1}^{T} \sum_{y, y'} f_k(y, y', x_t^{(i)}) p(y, y' \mid x^{(i)}) - \frac{\theta_k}{N \sigma^2}$$

This is identical to the batch gradient but without the sum over training instances, and with the regularization term divided by $N$ so that the sum of per-instance gradients equals the batch gradient.

**SGD update rule.** At iteration $m$, a random training instance is selected, and:

$$\theta^{(m)} = \theta^{(m-1)} + \alpha_m \nabla \ell_i(\theta^{(m-1)})$$

where $\alpha_m$ is a step size that must decrease over time to ensure convergence.

**Step size schedule.** The paper recommends a schedule of the form:

$$\alpha_m = \frac{1}{\sigma^2(m_0 + m)}$$

where $m_0$ is chosen based on the performance of fixed step sizes on a small validation subset. The classic conditions for convergence are $\sum_m \alpha_m = \infty$ (the steps must not decay too fast, so the algorithm can reach any point in parameter space) and $\sum_m \alpha_m^2 < \infty$ (the steps must decay fast enough that the variance of the parameter estimates goes to zero).

**Tradeoffs.** SGD requires many more iterations than batch methods, but each iteration is $O(TM^2)$ rather than $O(TM^2 N)$. For large $N$, the per-iteration speedup can be substantial. The paper notes that SGD requires more tuning than off-the-shelf L-BFGS, is not applicable to non-i.i.d. data, and may not be beneficial on small datasets.

---

#### 3.4.9 Approximate Training Methods for Intractable CRFs

When the CRF has a graphical structure too complex for exact inference, the paper describes two general strategies for approximate training: surrogate likelihood methods (modify the objective function) and approximate marginal methods (compute approximate gradients).

**Pseudolikelihood.** Instead of the full joint likelihood, pseudolikelihood maximizes the product of local conditional distributions:

$$\ell_{\text{pl}}(\theta) = \sum_{s \in V} \log p(y_s \mid y_{N(s)}, x; \theta)$$

where $N(s)$ are the neighbors of variable $Y_s$ in the Markov network. Each term $p(y_s \mid y_{N(s)}, x)$ conditions on the true values of the neighboring variables (available during training), and its normalization involves summation only over the values of $Y_s$, not over all joint configurations. This makes each term $O(M)$ to compute rather than $O(M^{|N(s)|+1})$.

**What pseudolikelihood computes:** it trains the model to predict each variable correctly given its neighbors' true values. The idea is that if all the local conditional distributions match the data, the joint distribution should be well-approximated, because the local conditionals determine the joint under the model's conditional independence assumptions.

**Why pseudolikelihood can fail:** at test time, the true neighboring labels are not available, so the model must use its own predictions for them. If the model learns to rely heavily on the ground-truth neighbor values during training, it may perform poorly at test time when those neighbors are predicted with errors. The paper reports that pseudolikelihood "has sometimes proved effective in NLP, [but] more commonly the performance of pseudolikelihood is poor."

**Blockwise pseudolikelihood and composite likelihood.** A natural improvement is to condition on larger blocks of variables rather than single variables. For a linear-chain CRF, per-edge pseudolikelihood is:

$$\ell_{\text{epl}}(\theta) = \sum_{t=1}^{T-1} \log p(y_t, y_{t+1} \mid y_{t-1}, y_{t+2}, \theta)$$

This conditions each edge on its immediate neighbors, providing more context during training. Composite likelihood generalizes this to arbitrary blocks: the user selects which subsets of variables to include as individual terms in the surrogate likelihood, with larger blocks typically yielding better parameter estimates at higher computational cost.

**Belief propagation-based approximate training.** For a fixed belief vector $q$ (set of approximate marginal distributions), the Bethe approximation to the likelihood is:

$$\ell_{\text{Bethe}}(\theta, q) = \sum_{C_p \in \mathcal{C}} \sum_{\Psi_c \in C_p} \log \Psi_c(x_c, y_c) - \sum_{C_p \in \mathcal{C}} \sum_{\Psi_c \in C_p} \sum_{y_c} q(y_c) \log \frac{q(y_c)}{\Psi_c(x_c, y_c)} + \sum_{s \in Y} (1 - d_s) \sum_{y_s} q(y_s) \log q(y_s)$$

where $d_s$ is the degree of variable node $s$. The first term is the unnormalized score of the true labels. The second term involves the factor beliefs (pseudomarginals). The third term compensates for overcounting in the entropy approximation.

**Saddlepoint optimization.** Training with $\ell_{\text{Bethe}}$ is a saddlepoint problem: $\max_\theta \min_q \ell_{\text{Bethe}}(\theta, q)$. The minimization over $q$ (finding the best belief vector for fixed parameters) is exactly what loopy BP does. The maximization over $\theta$ uses gradients that are computed from the BP beliefs. The algorithm alternates: run loopy BP to convergence (or for a fixed number of iterations) to obtain $q$, then take a gradient step in $\theta$ using the approximate marginals from $q$. The gradient of $\ell_{\text{Bethe}}$ with respect to $\theta_{pk}$ when $q$ is fixed is:

$$\frac{\partial \tilde{\ell}}{\partial \theta_{pk}} = \sum_{\Psi_c \in C_p} f_{pk}(x_c, y_c) - \sum_{\Psi_c \in C_p} \sum_{y'_c} f_{pk}(x_c, y'_c) q(y'_c)$$

**What this achieves:** the approximate gradient is simply the empirical count minus the expected count under the BP beliefs, exactly analogous to the exact gradient but using approximate marginals.

**The alternative surrogate likelihood.** An alternative formulation that is equivalent at BP fixed points is:

$$\hat{\ell}(\theta; q) = \log \left( \frac{\prod_{C_p \in \mathcal{C}} \prod_{\Psi_c \in C_p} q(y_c)}{\prod_{s \in Y} q(y_s)^{d_s - 1}} \right)$$

This is a direct generalization of the tree-structured factorization identity $p(y) = \prod_s p(y_s) \prod_a p(y_a) / \prod_{t \in a} p(y_t)$, applied to the pseudomarginals from BP.

**MCMC-based approximate training.** In the approximate marginals framework, MCMC methods (like Gibbs sampling) can be used to obtain approximate marginals $\hat{p}(y \mid x; \theta)$ that are substituted into the exact gradient formula. The paper notes that this is computationally demanding because MCMC chains need many iterations to converge, and inference must be run for many different parameter settings during training.

**Contrastive divergence.** To reduce the computational burden, contrastive divergence initializes the MCMC chain at the training label sequence and runs it for only a few iterations (often just one). The resulting approximate marginals are used in place of the true model expectations. The paper notes that CD has been applied mostly to restricted Boltzmann machines and latent-variable models, with limited work on CRFs.

**SampleRank.** A more recent approach where the objective is for learned parameters to correctly rank pairs of output configurations according to a supervised scoring function. Parameter updates are computed from differences between successive MCMC states, without waiting for convergence. The paper reports that SampleRank-trained models can substantially outperform CD.

**Why approximate training methods are necessary:** the paper provides Table 5.1 showing that practical CRF applications involve hundreds of thousands of parameters and training sequences. For general CRFs with loopy structure, exact inference is exponentially expensive. Approximate training methods make it feasible to learn models with complex dependencies, at the cost of introducing approximation error that can interact with parameter estimation in subtle and not-fully-understood ways.

---

#### 3.4.10 Feature Engineering for CRFs in Practice

The paper devotes substantial attention to feature engineering because "the accuracy of a CRF is strongly dependent on the features that are used," and the choice of features has more impact on performance than the choice of structured prediction algorithm.

**Label-observation features.** When output variables are discrete, features are typically structured as:

$$f_{pk}(y_c, x_c) = \mathbf{1}\{y_c = \tilde{y}_c\} q_{pk}(x_c)$$

where $q_{pk}(x_c)$ is an observation function that depends only on the input (e.g., "the current word is capitalized," "the current word ends in '-ing'"). This means each feature fires only for a specific output configuration $\tilde{y}_c$, but as long as that configuration matches, the feature value depends on arbitrary properties of the input. The computational advantage is that each observation function $q_{pk}$ needs to be evaluated only once per input, and the result is shared across all label configurations that reference it.

**Edge-observation vs. node-observation features.** For linear-chain CRFs, edge-observation features allow transition factors to depend on the input:

$$f(y_t, y_{t-1}, x_t) = q_m(x_t) \mathbf{1}\{y_t = y\} \mathbf{1}\{y_{t-1} = y'\} \quad \forall y, y' \in \mathcal{Y}, \forall m$$

This creates features like "$x_t$ is 'New', $y_t$ is LOCATION, $y_{t-1}$ is LOCATION." Node-observation features restrict transition factors to be purely label–label, with input dependence only on individual variables:

$$f(y_t, y_{t-1}, x_t) = \mathbf{1}\{y_t = y\} \mathbf{1}\{y_{t-1} = y'\} \quad \forall y, y' \in \mathcal{Y}$$

with separate label-observation features $f(y_t, x_t) = q_m(x_t) \mathbf{1}\{y_t = y\}$. Node-observation features reduce the parameter count significantly (from $M^2 \times (\text{number of observation functions})$ to $M^2 + M \times (\text{number of observation functions})$), which can help avoid overfitting when data is limited.

**Unsupported features.** Features that never fire in the training data can still be useful because they can be assigned negative weights, preventing the model from assigning high probability to configurations that are plausible a priori but never observed. The paper's example: the feature "current word is 'with' and label is City-Name" is unlikely to fire, but assigning it a negative weight reduces the probability of incorrectly labeling 'with' as a city. The heuristic for selecting which unsupported features to include: first train a CRF without unsupported features for a few iterations, then add unsupported features only for those cliques where the current model assigns non-negligible probability to an incorrect configuration.

**Redundant features and backoff.** In NLP applications, it can be helpful to include both edge factors $\Psi_t(y_t, y_{t-1}, x_t)$ and node factors $\Psi_t(y_t, x_t)$ in the same model. Although the edge factors alone can represent the same family of distributions, the redundant node factors provide a smoothing effect analogous to backoff in language modeling — useful when the number of features is large relative to the amount of training data. Regularization is essential in this case because it encourages the weight to be distributed across the overlapping features rather than concentrated in a few.

**Features as model combination.** The output of simpler models can be used as observation functions for a CRF. Examples include gazetteer features ("word appears in a list of city names from Wikipedia"), the marginal probabilities from an HMM trained on similar data, or cluster labels from unsupervised word clustering. The paper cautions against training the auxiliary model and the CRF on the same data, because the auxiliary model may perform unrealistically well on its training set, causing the CRF to over-rely on it.

**Input-dependent structure.** In skip-chain CRFs, edges are added between occurrences of the same word in a sentence, encouraging them to share the same label. This means the graphical structure of $p(y \mid x)$ depends on the input $x$ — a generalization beyond the fixed-structure models discussed earlier. This is useful when long-range dependencies are triggered by specific input properties rather than being uniformly present.

**Boundary handling.** A special START label at the beginning of each sequence allows the model to learn that features at sequence boundaries behave differently — for example, capitalization is a weaker indicator of proper nouns at the beginning of a sentence. This is implemented by prepending a dummy label $y_0 = \text{START}$ to each training sequence.

**Feature induction.** The paper mentions McCallum's (2003) method that begins with base features and iteratively adds conjunctions of existing features, and notes that $L_1$ regularization provides an alternative approach to automatic feature selection, with Lavergne et al. (2010) finding that $L_1$-regularized models can achieve comparable performance while keeping only 1% of features nonzero.

---

#### 3.4.11 Practical Implementation Concerns

**Exploiting sparsity.** Two types of sparsity improve computational efficiency. First, sparsity in factor values: if many factor values $\Psi_t(y_t, y_{t-1}, x_t)$ are known to be zero (infeasible transitions), the message-passing iterations can be implemented with sparse matrix operations, reducing the $O(M^2)$ per-step cost. Second, sparsity in feature vectors: in text applications, most features are binary indicators that are zero for any given input. Using sparse vector representations avoids computing dot products with mostly-zero vectors.

**Parameter tying for transitions.** Tying parameters for certain subsets of transitions reduces the effective size of the transition matrix, lessening the $O(M^2)$ dependence. For example, if states are organized into types, transitions between types can share parameters.

**Numerical underflow prevention.** The forward and backward variables decay exponentially toward zero. Two standard solutions:

**Scaling:** normalize each $\alpha_t$ and $\beta_t$ vector to sum to 1 at each step, saving the scaling factors. The partition function can be recovered from the product of scaling factors and the marginals. This is the standard approach from Rabiner's HMM tutorial.

**Logarithmic domain:** perform all computations in log-space using the operator $a \oplus b = \log(e^a + e^b)$, computed stably as:

$$a \oplus b = a + \log(1 + e^{b-a}) = b + \log(1 + e^{a-b})$$

where the version with the smaller exponent is chosen. The forward recursion becomes:

$$\log \alpha_t(j) = \bigoplus_{i \in S} \left( \log \Psi_t(j, i, x_t) + \log \alpha_{t-1}(i) \right)$$

The paper notes that in CRFs, the logarithmic approach does not impose additional overhead compared to scaling, because computing $\Psi_t(j, i, x_t) = \exp(\sum_k \theta_k f_k(j, i, x_t))$ already requires calling $\exp$. Both approaches require $O(T M^2)$ calls to transcendental functions.

**Gradient computation efficiency.** The gradient can be expressed in terms of the marginal distributions, which are computed during the forward–backward pass. Specifically, the expected feature count for feature $f_k$ at time $t$ is:

$$\sum_{y, y'} f_k(y, y', x_t) p(y_{t-1} = y', y_t = y \mid x)$$

By precomputing the pairwise marginals $p(y_{t-1}, y_t \mid x)$ for all $t$, the gradient for all features can be accumulated in a single pass. The sparsity of feature vectors again matters: many $f_k(y, y', x_t)$ are zero, so only non-zero features need to be accumulated.

**Training at scale.** Table 5.1 provides concrete scale examples from NLP:
- NP chunking: 248,471 parameters, 8,936 training sequences (211,727 tokens), 3 labels, training time 958 seconds.
- NER: 187,540 parameters, 946 sequences (204,567 tokens), 9 labels, training time 4,866 seconds.
- POS tagging: 509,951 parameters, 38,219 sequences (912,344 tokens), 45 labels, training time 325,500 seconds (~3.8 days).

The dominant factor in training time is the number of labels $M$ (due to the $O(M^2)$ cost per position), followed by the total number of tokens $N \times T$ (which determines the number of forward–backward passes per gradient evaluation).

**Parallelism.** The gradient is a sum over training instances, so it can be computed in parallel by dividing the training set across multiple threads or machines. Each thread computes the gradient on its subset, and the results are summed. For multicore machines, this provides near-linear speedup. For distributed settings, the cost of communicating parameter vectors across the network can be a bottleneck, and asynchronous stochastic gradient methods are a potential solution noted by the paper.

## 4. Key Insights and Innovations

### Innovation 1: The Generative-Discriminative Decomposition as a Unifying Framework for Structured Prediction

The paper's most fundamental conceptual move is to reframe the entire structured prediction problem through the lens of a single tradeoff: modeling the joint distribution $p(y, x) = p(y)p(x|y)$ versus modeling the conditional distribution $p(y|x)$ directly. While this distinction was well-established for simple classification through the Ng and Jordan (2002) analysis of naive Bayes versus logistic regression, the paper systematically extends this generative-discriminative pair concept to structured models, identifying linear-chain CRFs as the discriminative analogue of HMMs and general CRFs as the discriminative analogue of arbitrary undirected graphical models.

What makes this reframing intellectually distinctive is not the mathematical connection itself — the derivation that an HMM's conditional distribution is a restricted CRF is straightforward algebra — but rather the **architectural insight** about what the shift from joint to conditional modeling enables. The paper argues that the critical bottleneck in generative structured models is not the output dependencies (which are handled well by graphical models of either type) but the input model $p(x|y)$ or $p(x)$. By conditioning on $x$, the CRF simply does not need to represent how input features depend on each other — factors that depend only on $x$ vanish from the conditional distribution's graphical structure because they are constant with respect to $y$. This has the profound practical consequence that the conditional model's graph can be dramatically simpler than the joint model's graph while still leveraging arbitrarily complex, overlapping input features.

The paper makes this concrete through the contrast with HMMs and MEMMs. An HMM must either model input dependencies (difficult and often intractable) or make naive independence assumptions (which produce overconfident, miscalibrated probability estimates that compound errors in sequence models). MEMMs attempt to be discriminative but remain directed, paying for this with the label bias pathology — a structural flaw that the paper diagnoses as arising from v-structure independence assumptions, not from any contingent algorithmic choice. The CRF's undirected, globally-normalized form is presented not as an arbitrary alternative but as the **minimal structural change** needed to escape the label bias problem while retaining discriminative training. This is a genuine conceptual contribution: the paper identifies a design space (directed vs. undirected, locally vs. globally normalized, generative vs. discriminative) and maps out which combinations produce which pathologies.

The significance extends beyond the specific models discussed. By providing the diagram in Figure 2.4 — showing the relationships among naive Bayes, logistic regression, HMMs, linear-chain CRFs, generative models, and general CRFs — the paper essentially **defines the taxonomy of structured prediction approaches** that much subsequent work would adopt. The observation that "any logistic regression classifier can be converted into a naive Bayes classifier with the same decision boundary, and vice versa" (Section 2.2.3) becomes, when extended to sequences, the claim that CRFs and HMMs share the same hypothesis space but differ in how they are trained — a statement with precise theoretical meaning through Minka's (2005) analysis of parameter sharing between the input model and the conditional model. This framing makes it possible to understand exactly when and why discriminative training should outperform generative training (when modeling $p(x)$ requires trading off accuracy on $p(y|x)$, the distribution we care about) and when it should not (when limited data makes the generative model's smoother estimates beneficial).

### Innovation 2: The Diagnosis of Label Bias as a Structural Property of Directed Discriminative Models

The paper's analysis of the label bias problem in MEMMs (Section 6.1.3) represents a diagnostic achievement that goes beyond simply identifying a flaw in a competitor model. By connecting the label bias phenomenon to the graphical structure of the MEMM — specifically, to the v-structure independence assumptions that make $y_t$ marginally independent of future observations $x_{t+1}, x_{t+2}, \ldots$ — the paper transforms what could have been an empirical complaint into a **structural theorem**: any directed model with the MEMM's factorization will exhibit label bias, regardless of how its local conditional distributions are parameterized or trained.

The formal argument is clean and general. In the backward recursion for an MEMM, the backward message $\beta_t(i) = \sum_j p(y_{t+1}=j \mid y_t=i, x_{t+1}) \beta_{t+1}(j)$ must equal 1 for all $i$ regardless of future observations, because the local conditional probabilities sum to 1 and $\beta_{t+1}(j) = 1$ by backward induction. This means future observations provide literally zero information about the current state in the posterior distribution — a catastrophic loss of the primary motivation for sequence modeling. The paper's insight is that this is not an algorithmic failure but a **necessary consequence** of the directed factorization: the MEMM's graph encodes the independence assumption $y_t \perp\!\!\!\perp x_{t+1}, x_{t+2}, \ldots \mid y_{t-1}$, which is typically false in sequence labeling problems.

This diagnosis has several layers of significance. First, it provides a **principled criterion** for when directed discriminative models will fail: whenever the true data distribution violates the independence assumptions encoded in the graph. Second, it explains **why** the undirected CRF avoids the problem: the global normalization $Z(x)$ sums over entire sequences, coupling all positions and allowing bidirectional information flow without violating any independence constraints. Third, it clarifies that label bias is **not inherent to directed models in general** — as the paper notes, Berg-Kirkpatrick et al. present directed generative models with log-linear local conditionals that avoid label bias because their graphical structure is different — but is specific to the v-structure topology of the MEMM. This nuance is important because it prevents the field from drawing the wrong lesson (that directed models are always problematic) and instead points to the correct lesson (that the conditional independence assumptions encoded in the graph structure must match the domain).

The broader intellectual contribution is to demonstrate **graphical model literacy** as a tool for understanding and designing structured predictors. Rather than treating inference algorithms and learning procedures as independent components, the paper shows how the graphical structure determines both the statistical properties (which independencies are assumed, and whether they hold) and the algorithmic properties (whether inference is tractable, whether label bias occurs). This unified perspective — in which a single factor graph simultaneously encodes modeling assumptions, dictates inference complexity, and determines susceptibility to pathologies — is perhaps the deepest lesson the survey imparts, and the label bias analysis is its clearest illustration.

### Innovation 3: The Variational Interpretation of Belief Propagation as a Bridge Between Inference and Learning

While belief propagation was well-established as an inference algorithm for graphical models at the time of the survey, the paper's contribution is to articulate **why the variational interpretation matters specifically for CRF training**, and to use that interpretation to derive principled approximate training procedures. This is not a new algorithm but a conceptual synthesis that connects previously separate ideas — message-passing, Bethe free energy minimization, and maximum likelihood parameter estimation — into a coherent framework.

The key move is recognizing that if loopy BP can be understood as minimizing the Bethe free energy $O_{\text{Bethe}}(q)$ over pseudomarginals $q$, then the value $\min_q O_{\text{Bethe}}(q)$ provides an approximation to $\log Z(x)$, and the minimizing pseudomarginals $q^*$ provide approximations to the true marginals. This yields not one but two strategies for approximate CRF training. The **approximate marginals strategy** simply substitutes BP beliefs for the true marginals in the exact gradient formula (Equation 5.25). The **surrogate likelihood strategy** optimizes $\ell_{\text{Bethe}}(\theta, q)$ as a saddlepoint problem, alternating between BP inference (minimizing over $q$) and gradient steps in $\theta$ (maximizing over $\theta$).

What makes this framing significant is that it resolves an ambiguity about whether approximate inference and exact learning can be combined safely. The paper notes that Kulesza and Pereira (2008) found "a situation in which the perceptron algorithm interacts in a pathological fashion with max-product belief propagation," but that "surrogate likelihood methods, by contrast, do not seem to display this sort of pathology, as Wainwright (2006) points out for the case of convex surrogate likelihoods." The variational perspective explains this difference: when an approximate inference procedure corresponds to minimizing a well-defined objective function (like the Bethe free energy), using it within learning is optimizing a coherent (if approximate) objective. When it does not — when, for example, early-stopped BP does not correspond to any stationary point — the interaction between inference approximation and parameter updates can be arbitrary and pathological.

The paper extends this logic to piecewise training (Sutton and McCallum, 2009) and to the relationship between BP fixed points and the alternative surrogate likelihood $\hat{\ell}(\theta; q) = \log(\prod_{C_p} \prod_{\Psi_c} q(y_c) / \prod_s q(y_s)^{d_s-1})$, showing that these are not ad hoc heuristics but instances of the same underlying variational principle. This **unification of approximate training methods** under the Bethe free energy is a genuine conceptual contribution: it gives practitioners a principled way to reason about which approximations are safe (those that correspond to well-defined objective functions) and which are risky (those that do not), and it provides a framework for developing new approximate training methods by designing new variational objectives.

The survey also provides one of the earliest and most accessible explanations of how the Bethe approximation works — defining $H_{\text{Bethe}}$ as exact for trees and approximate for loopy graphs, explaining local consistency constraints, and connecting the BP fixed-point equations to stationarity conditions. For many readers, this would have been their first exposure to variational inference in the context of structured prediction, and the clarity of the exposition has made this section of the survey a standard reference.

### Innovation 4: The Systematic Empirical Demonstration That Feature Engineering Dominates Algorithm Choice

The paper makes an understated but practically crucial empirical contribution through Table 5.1 and the surrounding discussion in Section 5.5: by reporting concrete numbers for model size, training data scale, and training time across three NLP tasks, it provides the first **systematic characterization of the computational economics of CRF training** at realistic scales. The 45-label POS tagging task takes 325,500 seconds (~3.8 days) to train, while the 3-label NP chunking task takes 958 seconds — a 340× difference driven primarily by the $O(M^2)$ dependence on the number of labels. This quantitative grounding transforms the preceding algorithmic discussion from theory to engineering practice.

But the deeper insight is methodological rather than engineering. The paper states explicitly that "careful selection of features has more effect on performance than the choice of structured prediction algorithm," and that "the similarities between various structured prediction methods are more important than the differences." This is a striking claim to make in a survey about a specific model class — essentially arguing that the choice of CRF versus SVM-struct versus perceptron matters less than the feature engineering that all these methods share. The paper supports this through the running NER example (Section 2.6.1), which devotes considerable space to the specific observation functions used (Table 2.2), and through the discussion of unsupported features, edge-observation versus node-observation features, and feature induction.

This insight is genuinely significant because it redirects research attention: if feature engineering dominates, then progress in structured prediction comes less from inventing new model families and more from developing better methods for constructing, selecting, and regularizing features. The subsequent decade of NLP research — in which neural network approaches largely replaced manual feature engineering by learning feature representations automatically — can be seen as a validation of this insight from the opposite direction: the breakthrough of deep learning for NLP was precisely that it automated the feature engineering that the CRF literature had identified as the key bottleneck.

The paper also makes a more subtle methodological point through its feature engineering discussion: the distinction between **model structure** (the factor graph over outputs) and **feature design** (the observation functions that couple inputs to outputs) is not always clean, and the most powerful CRF applications blur this boundary. Input-dependent structure (skip-chain CRFs), features that encode the output of simpler models (HMM marginals as CRF features), and redundant factors that provide backoff-like smoothing all represent ways in which "features" can compensate for limitations in "structure." This observation — that the feature function interface is the primary site of flexibility in CRF design — explains why the log-linear parameterization with arbitrary real-valued feature functions (Equation 2.18) is so powerful: it makes no restrictions on what information can influence the prediction, deferring all domain-specific knowledge to the feature engineering process.

## 5. Experimental Analysis

### Evaluation Methodology

- **Dataset.** The paper uses the MATH benchmark (Hendrycks et al., 2021), consisting of high-school competition-level mathematics problems. The specific split from Lightman et al. (2022) is used: 12,000 training questions and 500 test questions. The MATH dataset is chosen because test-time compute is expected to be most beneficial when the model already possesses the necessary knowledge — mathematical reasoning requires multi-step logical deduction rather than novel factual recall, fitting this profile.

- **Base model(s).** All main experiments use PaLM 2-S* (Codey) (Anil et al., 2023). The authors argue this model is "representative of the capabilities of many contemporary LLMs" and sits in a useful regime: non-trivial performance on MATH (roughly 10–19% pass@1 depending on configuration) but far from saturation, leaving room for test-time compute to make a measurable difference. For the FLOPs-matched comparison in Section 7, a second model with approximately 14× more parameters is used as the pretraining-scaled baseline. This larger model uses greedy decoding with no additional test-time compute.

- **Metrics.** The primary metric throughout is MATH test accuracy (%) — the fraction of the 500 test questions for which the selected final answer matches the ground truth. Answers are graded using the grading function released by Lightman et al. (2022) (Appendix G). When analyzing difficulty-dependent behavior, accuracy is reported within each of five difficulty quintiles separately. The difficulty bins are computed based on the base model's pass@1 rate on each question (estimated from 2048 samples), not the MATH dataset's hand-labeled difficulty levels, because model-specific difficulty is more predictive of test-time compute efficacy.

- **Baselines.** The paper uses several baselines:
  - **Majority voting**: select the most common final answer among N sampled solutions, with no learned verifier.
  - **ORM best-of-N weighted**: score N solutions with an outcome reward model and apply best-of-N weighted selection (following Li et al., 2023).
  - **PRM best-of-N weighted**: score N solutions with the process reward model and apply best-of-N weighted selection.
  - **Parallel sampling** (for revisions): generate N independent solutions from the revision model and select the best via verifier or majority voting.
  - **Greedy decoding from the 14× larger model** (for the FLOPs-matched comparison).

- **Generation budget / compute accounting.** The universal unit of test-time compute is one "generation" — one complete sampled answer from the base LLM. For beam search and best-of-N, the budget equals the number of beams or samples N. For lookahead search with k lookahead steps, the cost is N × (k+1) to account for the additional rollout computation (Section 5.3). Budgets are swept across powers of 2, typically from 2⁰ to 2⁹ (1 to 512 generations). For the FLOPs-matched comparison (Section 7), FLOPs are accounted using standard scaling law approximations: pretraining FLOPs X = 6ND_pretrain and inference FLOPs Y = 2ND_inference, where N is the number of model parameters. The ratio R = D_inference / D_pretrain determines how many extra inference generations the smaller model can afford while matching the total FLOPs of the larger model.

- **Cross-validation / statistical protocol.** To avoid contaminating strategy selection with test-set performance, the authors use two-fold cross-validation within each difficulty bin on the 500-question test set. The best-performing strategy is selected on one fold and evaluated on the other, with results averaged (Section 3.2). Difficulty bins are computed once per question using 2048 samples; the same bin assignments are used across all experiments for consistency.

### Main Quantitative Results

#### Search Against PRM Verifiers (Section 5)

The headline finding is that the optimal search algorithm depends on both the generation budget and question difficulty, with beam search dominating at low budgets on medium problems but best-of-N weighted pulling ahead at high budgets on easy problems — a pattern driven by PRM over-optimization.

**Aggregate search algorithm comparison (Figure 3, left).** Across all 500 test questions with a maximum budget of 256 generations:

- At low budgets (2–8 generations), beam search with M = 4 significantly outperforms best-of-N weighted. At 4 generations, beam search (M = 4) achieves roughly 27% accuracy versus roughly 16% for best-of-N weighted — an approximately 11 percentage point gap.
- At high budgets (64–256), beam search performance flattens and falls slightly below best-of-N weighted. Best-of-N weighted reaches approximately 38% at 512 generations; beam search (M = 4) plateaus around 34%.
- Lookahead search (both k = 1 and k = 3) generally underperforms at the same generation budget due to its higher per-step cost. The 3-step lookahead variants converge to similar performance as other methods at very high budgets but never surpass them. This is a notable negative result: the most sophisticated optimizer performs worst overall.
- Majority voting trails all verifier-based methods substantially, reaching only about 29% at 512 generations.

**Difficulty-bin analysis for search (Figure 3, right).** When results are broken out by difficulty quintile (beam search M = 4 vs. best-of-N weighted, shown at four budget levels: 4, 16, 64, 256 generations):

- **Bin 1 (easiest):** Beam search accuracy *decreases* from roughly 78% to 77% as the budget goes from 4 to 256, while best-of-N weighted increases from 68% to 88%. This is the clearest evidence of PRM over-optimization — beam search finds solutions that exploit the verifier signal rather than genuinely improving correctness.
- **Bin 2:** Beam search improves modestly (roughly 14% → 32%) but best-of-N weighted improves faster (roughly 14% → 60%), maintaining a clear advantage at high budgets.
- **Bin 3:** Beam search consistently outperforms best-of-N weighted across all budgets, reaching roughly 34% vs. 23% at 256 generations. This is where the PRM's guidance genuinely helps navigate toward correct solutions.
- **Bin 4:** Beam search shows the strongest relative advantage, reaching roughly 17% vs. 10% for best-of-N at 256 generations.
- **Bin 5 (hardest):** Both methods hover near 1–3% regardless of budget. No method makes meaningful progress on problems fundamentally beyond the base model's capability.

**Compute-optimal search (Figure 4).** By selecting the best search strategy per difficulty bin at each budget level (using two-fold cross-validation):

- At 16 generations, compute-optimal (oracle bins) achieves approximately 27% accuracy, roughly matching PRM best-of-N weighted at 64 generations — a 4× compute reduction.
- At 256 generations, compute-optimal oracle reaches approximately 39.5%, surpassing PRM best-of-N weighted at the same budget (roughly 37%).
- Compute-optimal with predicted difficulty bins (using PRM score rather than ground-truth correctness to estimate difficulty) tracks the oracle version closely, particularly at lower budgets. The two curves "largely overlap" (Figure 4), with the predicted version reaching approximately 37% at 256 generations.
- Both compute-optimal variants consistently outperform ORM best-of-N weighted (which peaks around 34% at 512 generations) and majority voting (around 29%).

**PRM vs. ORM (Figure 14, Appendix F).** At 2048 samples, PRM best-of-N weighted achieves approximately 40% accuracy versus roughly 35% for ORM best-of-N weighted and roughly 30% for majority voting. The gap between PRM and ORM widens with the number of samples, confirming the PRM's superior scaling properties. This is attributed to step-level PRM training acting as beneficial representation learning even when last-step aggregation effectively reduces the PRM to ORM-like behavior at selection time.

#### Revision Model Results (Section 6)

The headline finding is that sequential revisions modestly outperform parallel sampling in aggregate, but the optimal sequential-to-parallel ratio is strongly difficulty-dependent: easy problems favor purely sequential revision, while hard problems benefit from a balanced mix.

**Revision model pass@1 trajectory (Figure 6, left).** Starting from approximately 18.2% pass@1 at step 1, the revision model's per-step accuracy improves to roughly 24–25% by steps 15–20, and remains in the 23–25% range out to 64 steps. The model generalizes beyond its 4-step training horizon — it was trained only on sequences with up to 4 previous incorrect answers, yet continues improving and then maintaining accuracy at much longer horizons.

**Sequential vs. parallel (Figure 6, right).** At 64 generations:
- Sequential + best-of-N weighted: approximately 41.5%
- Parallel + best-of-N weighted: approximately 39%
- Sequential + majority: approximately 38%
- Parallel + majority: approximately 35%

Sequential outperforms parallel under both selection mechanisms, with the verifier-based gap (roughly 2.5 percentage points) being slightly narrower than the majority-based gap (roughly 3 points). The verifier-based advantage over majority voting is roughly 3.5–4 points for both sequential and parallel configurations.

**Sequential-to-parallel ratio sweep (Figure 7, left).** For fixed generation budgets, varying the ratio of sequential revisions to parallel chains reveals:
- At 256 generations, the optimal ratio is around 2¹ to 2³ (2:1 to 8:1 sequential-to-parallel), achieving approximately 43–44% accuracy.
- Fully parallel (leftmost point) yields approximately 40%.
- Fully sequential (rightmost point) yields approximately 42%.
- At lower budgets (8–32 generations), fully sequential is optimal — the curves are monotonically increasing with the sequential-to-parallel ratio. This makes intuitive sense: with a small budget, the diversity from parallel sampling is insufficient to outweigh the benefits of chain-based refinement.

**Difficulty-dependent ratio (Figure 7, right).** At a fixed budget of 128 generations:
- **Bin 1:** Performance is essentially flat across all ratios, around 90–92%. Easy questions are insensitive to the allocation strategy — the model gets them right regardless.
- **Bin 2:** Slight advantage for higher sequential ratios, approximately 63% at fully sequential vs. 58% at fully parallel.
- **Bin 3:** A clear optimal ratio emerges at moderate sequential-to-parallel values (around 2¹ to 2³), reaching approximately 42% vs. 35% at the extremes. This mirrors the search finding: medium problems benefit from a balance of exploration (parallel diversity) and exploitation (sequential refinement).
- **Bin 4:** Similar pattern, with the peak at a moderate ratio achieving roughly 18% vs. 14% at fully parallel.
- **Bin 5:** All ratios produce roughly 2–3% accuracy. No allocation strategy helps on fundamentally hard problems.

**Compute-optimal revisions (Figure 8).** Selecting the optimal sequential-to-parallel ratio per difficulty bin:
- At 64 generations, compute-optimal oracle achieves approximately 40%, matching parallel best-of-N weighted at 256 generations — a 4× improvement.
- At 256 generations, compute-optimal oracle reaches approximately 44%, compared to roughly 41% for best-of-N weighted and 37% for parallel-only.
- Compute-optimal predicted bins perform slightly below oracle bins at high budgets (approximately 41% at 256 generations) but still substantially outperform the parallel baseline.
- Importantly, the parallel baseline appears to **plateau** around 36–37% at high budgets, while compute-optimal scaling continues to improve. This suggests the gains from adaptive allocation compound at higher budgets rather than saturating.

#### FLOPs-Matched Comparison: Test-Time vs. Pretraining Compute (Section 7)

The headline finding is that test-time compute with a smaller model can outperform a ~14× larger model — but only on problems within the base model's capability range and when the inference-to-pretraining token ratio R is favorable.

**Revisions (Figure 9, left; Figure 1, top-right bar chart).** Comparing PaLM 2-S* with compute-optimal revisions against the ~14× larger model:

| Difficulty | R ≪ 1 (0.16) | R ≈ 1 (0.79) | R ≫ 1 (22) |
|---|---|---|---|
| Easy (bin 1) | +11.8% | +3.5% | −11.9% |
| Medium (bin 2–3) | +27.8% | +16.7% | +5.4% |
| Hard (bins 4–5) | +21.6% | negative (implied) | −37.2% |

At R ≪ 1, test-time compute with the smaller model outperforms the larger model across **all** difficulty levels. At R ≫ 1, it only remains preferable on easy-to-medium questions, with hard questions showing a −37.2% relative disadvantage. This pattern is clear: when inference tokens are a small fraction of total compute, the pretraining savings from using a smaller model translate into a large test-time budget, which more than compensates on solvable problems. When inference dominates (R ≫ 1), the larger model's per-token cost eats into the test-time budget, and the advantage disappears or reverses.

**PRM search (Figure 9, right; Figure 1, bottom-right bar chart).** The pattern is starker:

| Difficulty | R ≪ 1 (0.16) | R ≈ 1 (0.79) | R ≫ 1 (22) |
|---|---|---|---|
| Easy | +19.1% | +2.2% | +2.0% |
| Medium | 0.0% | −35.3% | −30.8% |
| Hard | −3.6% | −35.3% | −52.9% |

PRM search shows substantially weaker benefits than revisions in the FLOPs-matched comparison. On medium and hard questions, test-time compute with PRM search is dramatically worse than simply using the larger model, even at moderate R values. Only on easy questions does test-time compute remain preferable across all R regimes. This asymmetry between search and revisions — revisions perform better in the FLOPs-matched comparison — is a noteworthy empirical finding that the paper does not extensively analyze.

**Figure 9 detail.** The line plots show accuracy per difficulty bin as test-time compute scales for the smaller model. The 14× larger model's greedy performance is plotted as a horizontal star at three x-axis positions corresponding to the three R values. Where the scaling line is above the star, test-time compute wins. On bin 1 (easiest), the scaling line is above all three stars for revisions. On bin 5 (hardest), the line is below all three stars and essentially flat near 0–5%, confirming that no amount of test-time compute helps on the hardest problems — these require capabilities that can only be acquired through pretraining.

### Ablation Studies and Robustness Checks

**PRM aggregation strategy (Appendix E, Figure 13):** Comparing "min," "prod," and "last" step-wise aggregation for converting per-step PRM scores into a single solution score reveals that "last" performs best — achieving roughly 37% at 256 samples, compared to roughly 35% for "min" and 27% for "prod." ORM achieves roughly 34%. This contradicts prior work (Lightman et al., 2023; Wang et al., 2023) which found "min" to be best. The authors attribute this to their use of soft Monte Carlo labels rather than binary correctness labels, which changes how per-step scores distribute. Notably, "last" aggregation effectively reduces the PRM to ORM-like behavior at selection time, yet the PRM still outperforms a separately trained ORM — evidence that step-level PRM training provides beneficial representation learning.

**PRM vs. ORM scaling (Appendix F, Figure 14):** The PRM consistently outperforms the ORM across all sample counts, with the gap widening as samples increase. At 2048 samples, PRM best-of-N weighted reaches approximately 40% vs. ORM's 35% vs. majority voting's 30%. This confirms that the PRM's advantage is not merely a fixed offset but compounds with more samples — better verifiers enable more effective scaling of test-time compute.

**Revision model verifier choice (Appendix J, Figure 15a):** The base-LM PRM underperforms the revision-specific ORM when scoring revision model outputs — sequential + base-LM PRM achieves roughly 40% at 64 generations vs. sequential + revision ORM at roughly 42%. This confirms distribution shift as a practical concern: the PRM was trained on base model outputs, and the revision model produces systematically different solutions that the PRM is not calibrated for. The revision-specific ORM, trained on revision model outputs, closes this gap.

**Revision history in verifier context (Appendix J, Figure 15b):** Including previous revisions in the ORM's context provides a small improvement over the no-history ablation — approximately 1–2 percentage points at 64 generations — but both variants outperform the parallel baseline. This indicates that the sequential sampling benefit is not solely attributable to the verifier seeing more context (though that helps marginally); the revision process itself generates genuinely better candidates.

**Oracle vs. predicted difficulty bins (Figures 4, 8, and Appendix C, Figures 11–12):** Both oracle and predicted bins yield qualitatively similar trends across difficulty levels. For search (Figure 4), predicted bins track oracle bins closely, with curves "largely overlapping." For revisions (Figure 8), predicted bins show slightly lower performance at high budgets (approximately 41% vs. 44% at 256 generations) but still substantially outperform the parallel baseline. This is the critical robustness check: the compute-optimal strategy works without access to ground-truth labels, using only the PRM's output distribution as a difficulty proxy.

**Majority voting for revisions (Appendix B, Figure 10):** The sequential-to-parallel ratio trends observed with verifier-based selection are replicated with majority voting: easy questions are insensitive to ratio, hard questions show an optimal intermediate ratio, and fully sequential marginally outperforms fully parallel in aggregate. This robustness to the selection mechanism suggests that the revision model's sequential sampling benefit is not an artifact of the verifier but reflects genuine improvement in the proposal distribution.

**ReST^EM revision model (Appendix K, Figure 16):** An attempt to further optimize the revision model using ReST^EM (Singh et al., 2024) backfires: additional sequential revisions **substantially hurt** performance with this model. At 256 generations, fully sequential performance drops to approximately 33.5% compared to roughly 38.5% at the optimal ratio. The authors hypothesize that on-policy data collection in ReST^EM exacerbates spurious correlations in revision data, causing the model to fail to learn the revision task properly. This is a notable negative result that highlights the sensitivity of revision training to the data generation procedure: the offline, edit-distance-based pairing method (Section 6.1) works, but online RL-style optimization of the same model degrades it.

**Beam width sweeps for search (implicit in Figure 3):** The paper sweeps two beam width settings — M = √N (growing with budget) and M = 4 (fixed). The fixed beam width of 4 performs better at low budgets, while the growing beam width catches up at higher budgets. This suggests that the optimal exploration breadth depends on the total budget: when resources are scarce, narrow deep search is better; when resources are abundant, broader search helps.

### Critical Assessment

**Claim 1: "Compute-optimal scaling improves efficiency by more than 4× over best-of-N."** This claim is supported for specific budget regimes but requires careful qualification. For search (Figure 4), compute-optimal at 16 generations (~27%) matches best-of-N at 64 generations (~27%) — a 4× reduction. For revisions (Figure 8), compute-optimal at 64 generations (~40%) matches best-of-N at 256 generations (~40%) — also a 4× reduction. However, the 4× figure applies to the lower-to-moderate budget regime (16–64 vs. 64–256). At the highest tested budgets (256–512), the gap between compute-optimal and best-of-N narrows. For search, compute-optimal at 256 generations achieves ~39.5% vs. best-of-N at ~37% — a meaningful gap but not 4×. The 4× figure is best understood as the maximum observed efficiency gain, not a uniform property across all budgets.

Crucially, the difficulty estimation cost is **not included** in the budget accounting. Generating 2048 samples per question to estimate difficulty is more expensive than the largest test-time budgets studied (256–512 generations). The paper acknowledges this explicitly (Section 3.2) but the reported 4× gains are computed as if difficulty were known for free. In a deployment setting where difficulty estimation must be amortized across queries, the effective gains would be smaller. The paper does not provide an analysis of how these gains change when difficulty estimation cost is included, which is a significant gap.

**Claim 2: "Test-time compute with a smaller model can outperform a ~14× larger model."** This claim is supported with sharp boundary conditions that the paper is admirably transparent about. The claim holds convincingly for easy-to-medium problems at R ≪ 1 (0.16) and R ≈ 1 (0.79) with revisions: +27.8% on medium problems at R ≪ 1, +16.7% at R ≈ 1. However, the claim fails for hard problems at R ≫ 1: −37.2% for revisions, −52.9% for PRM search. This conditional pattern is clearly documented in the paper's Figure 1 bar charts.

A significant weakness is that the 14× larger model uses **only greedy decoding** — no majority voting, no best-of-N, no test-time compute at all. This makes the baseline artificially weak. A fairer comparison would give the larger model some test-time compute budget, since the claim is about the optimal allocation of total compute between training and inference. If the larger model were given even a modest best-of-8 or best-of-16 budget, the crossover points would shift, potentially making test-time compute with the smaller model less favorable. The paper does not include this ablation.

Additionally, the 14× larger model scales parameters while holding training data fixed (following the LLaMA paradigm), which is not compute-optimal pretraining (Hoffmann et al., 2022). A Chinchilla-optimal large model (scaling both parameters and data) would likely be a stronger baseline, potentially changing the FLOPs-matched results. The paper acknowledges this limitation in Section 7 but does not explore it experimentally.

**Claim 3: "Efficacy depends critically on prompt difficulty."** This is the most robust and well-supported claim in the paper. The difficulty-bin analyses show qualitatively different — and sometimes opposite — effects of the same strategy at different difficulty levels. Beam search *hurts* easy-problem performance at high budgets (Figure 3, right: bin 1 accuracy decreases from ~78% to ~77% as budget increases from 4 to 256) while *helping* on medium-hard problems (bin 3: ~34% vs. ~23% at 256 generations). Fully sequential revisions dominate on easy problems but a balanced ratio is optimal on hard problems (Figure 7, right). These non-monotonic patterns are replicated across search methods, revision strategies, and selection mechanisms (majority and verifier-based). However, the findings are demonstrated on a single benchmark (MATH) with a single model family (PaLM 2-S*). Whether the specific difficulty thresholds and strategy rankings generalize to other reasoning domains (code generation, logical reasoning) or other model families is untested.

**Experimental design weaknesses that limit the strength of conclusions:**

- **Single benchmark, single model family:** All results are on MATH with PaLM 2-S*. The authors argue the model is "representative" but provide no evidence. The PRM's over-optimization behavior, the revision model's sequential-to-parallel optimal ratio, and the specific difficulty bin thresholds could all be model-specific or task-specific.

- **Small test set for strategy selection:** 500 questions split into five difficulty quintiles of ~100 each, then further split by two-fold cross-validation, means the compute-optimal policy is selected based on ~50 questions per fold per bin. With small sample sizes, the selected strategies may be noisy — a different random split could select different optimal strategies. The paper does not report confidence intervals on the compute-optimal scaling curves, and the two-fold cross-validation (rather than, say, five-fold) provides only a single train/test split per fold, limiting the assessment of variance.

- **PRM search and revisions studied in isolation:** The paper studies search and revisions independently but never combines PRM tree-search with the revision model as the proposal distribution. Section 8 acknowledges this gap explicitly. Since the two mechanisms have complementary strengths (revisions improve candidate quality on easy problems, search helps select among diverse candidates on medium problems), the current results likely represent a lower bound on what combined approaches could achieve. The absence of this natural combination is a notable omission.

- **No dynamic or online difficulty estimation:** The difficulty bins are static and precomputed from 2048 samples. In a practical system, one might estimate difficulty adaptively — start with a few samples, assess the verifier's score distribution, and allocate the remaining budget accordingly. Such an approach could subsume the difficulty estimation cost into the problem-solving process, potentially closing the gap between the reported 4× gains and what is achievable in deployment. The paper does not explore this direction.

- **Revision model's correct-to-incorrect reversion is addressed only with post-hoc selection:** Approximately 38% of correct answers in a revision chain get revised to incorrect answers in the next step. The paper mitigates this with majority voting or verifier-based selection across the chain, but these are heuristics that select the best answer post-hoc rather than preventing the model from making the error in the first place. A more principled solution — such as training the model with explicit "stop revising" signals or incorporating correctness detection — is not explored. The ReST^EM negative result (Appendix K) suggests that naive attempts to improve the revision model can backfire, making this a significant open problem.

- **The "unsupported features trick" is described but not experimentally validated:** The paper describes a heuristic for selecting which zero-count features to include (train without them first, then add features for cliques where the model assigns non-negligible probability to incorrect configurations). However, the survey provides no experimental comparison of this heuristic against alternative feature selection methods or against including all unsupported features. Given the practical importance of controlling model size, this is a missed opportunity for quantitative guidance.

**Experiments that would have strengthened the paper:**

1. **Combined PRM search + revisions:** Apply beam search using the revision model as the proposal distribution, or use the PRM to guide which revision branches to pursue. This would test whether the complementary strengths identified in the separate analyses can be combined for additional gains.

2. **Ablation of the edit-distance-based pairing for revision training:** The paper uses the incorrect answer with smallest character-level edit distance to the correct answer as the last incorrect example in the training sequence (Section 6.1). Comparing this against random pairing would quantify the contribution of this specific design choice, which the paper highlights as important but does not experimentally isolate.

3. **Adaptive difficulty estimation:** Compare static binning (2048 pre-samples) against dynamic estimation (start with 4–8 samples, assess score distribution, allocate remaining budget). This would address the paper's own acknowledged limitation about difficulty estimation cost and bridge the gap between the reported gains and practical deployment.

4. **Larger model with modest test-time compute as baseline:** In the FLOPs-matched comparison, give the 14× larger model a best-of-8 or best-of-16 budget (adjusting the FLOPs accounting accordingly) rather than greedy decoding only. This would test whether test-time compute with a small model truly dominates pretraining, or whether the finding is partly an artifact of the weak baseline.

5. **Confidence intervals on compute-optimal scaling curves:** Report variance across cross-validation folds or bootstrap resamples to quantify the uncertainty in the reported accuracy numbers, especially given the small per-bin sample sizes (~100 questions, further halved by cross-validation).

## 6. Limitations and Trade-offs

### 6.1 Difficulty Estimation Cost Is Not Accounted For in Headline Efficiency Gains

**The assumption or constraint.** The entire compute-optimal framework depends on estimating each prompt's difficulty before deciding how to allocate the inference budget. The paper's method requires generating 2048 samples per question and averaging either ground-truth correctness (oracle bins) or PRM final-answer scores (predicted bins) to assign a question to one of five difficulty quintiles. The authors acknowledge this directly in Section 3.2:

> "estimating difficulty in this way still incurs additional computation cost during inference... our experiments do not account for this cost largely for simplicity"

Despite this transparency, all reported efficiency gains — the 4× reductions in generation budget for equivalent accuracy — are computed *after* difficulty is known, without amortizing the cost of learning it.

**The consequence.** The 2048-sample difficulty estimation step consumes **more compute than the largest test-time budgets studied** (256–512 generations). In a realistic deployment, the total cost would be difficulty estimation + strategy execution, and the former could dominate the latter. The 4× figure (e.g., compute-optimal at 16 generations matching best-of-N at 64 generations for search, as shown in Figure 4) should therefore be understood as an **upper bound on achievable efficiency** in a setting where difficulty is known for free, not as a realized deployment gain. A practitioner implementing this system would face a difficult exploration–exploitation tradeoff: spend a large fraction of the total compute budget just to decide *how* to spend the remainder, potentially negating the adaptive strategy's advantage.

**What evidence exists in the paper.** The paper does not report any experiment that includes difficulty estimation cost in the budget accounting. Figure 4 and Figure 8, which show the main 4× efficiency claims, treat difficulty bins as a precomputed input. The paper does not provide an analysis of how the gains change when difficulty estimation is amortized across queries (e.g., by sharing difficulty estimates across similar prompts or by caching estimates for repeated queries). The predicted difficulty curve in Figure 4 (using PRM scores rather than ground truth) tracks the oracle curve closely, confirming that ground-truth labels are not needed, but the 2048-sample cost remains for the PRM-based method.

**Mitigation status.** The paper acknowledges this limitation in Section 3.2 and suggests future work on "pretraining or finetuning models to directly predict difficulty of a question" from the question text alone, bypassing the need for 2048 samples. However, no such model is developed or evaluated. The paper also does not explore adaptive estimation strategies — for instance, starting with a small number of samples (4–8), assessing the PRM's score distribution as a quick difficulty signal, and allocating the remaining budget accordingly. An adaptive approach could potentially subsume the estimation cost into the problem-solving process, but the paper provides no evidence for or against this approach. This limitation remains the single largest barrier to practical deployment of the compute-optimal framework as described.

---

### 6.2 Hard Problems Remain Fundamentally Unsolved — Test-Time Compute Cannot Create Capability

**The assumption or constraint.** The paper implicitly assumes that the base model possesses some non-trivial capability on the target problems — that there exist correct solutions in the proposal distribution to find or refine. This is captured in the definition of difficulty bins: a question is in bin 5 (hardest) if the base model's pass@1 rate is essentially zero. The compute-optimal framework does not claim to help on such problems, but the boundaries of this limitation are worth characterizing precisely because they delineate where the method is applicable.

**The consequence.** Across all methods — search, revisions, and their compute-optimal combinations — the hardest questions (difficulty bin 5) show **near-zero improvement regardless of compute budget**. The evidence is consistent across every experiment:

- In Figure 3 (right), bin 5 accuracy hovers at 1–3% for all search methods and all budgets from 4 to 256 generations.
- In Figure 7 (right), bin 5 shows roughly 2–3% accuracy irrespective of the sequential-to-parallel ratio at 128 generations.
- In the FLOPs-matched comparison (Figure 9), the bin 5 scaling line is essentially flat near 0–5% for both revisions and PRM search, and the 14× larger model (greedy decoding) also performs poorly, though somewhat better at the higher R values.

This means test-time compute amplifies existing capability but **cannot create it**. If the base model cannot generate a correct solution even once in 2048 attempts, no amount of clever search or iterative revision will find one. For problems that require novel reasoning, out-of-distribution generalization, or capabilities not present in the base model's training distribution, scaling inference compute offers no path forward. The paper is candid about this in the Section 7 takeaway box, but the implication is profound: for the hardest problems, pretraining remains the **only** viable path to improvement.

**What evidence exists in the paper.** The bin 5 results, consistently near floor across all experiments, provide strong negative evidence. However, the paper does not characterize *what makes a problem bin 5* for this model family beyond the pass@1 operationalization. Are these problems that require mathematical techniques not represented in the training data? Problems with complex multi-step dependencies where the base model makes early errors that compound? Problems requiring external knowledge? Understanding the failure modes would help practitioners predict which problem distributions are suitable for test-time compute strategies. The paper also does not explore whether the bin 5 boundary shifts with model scale — perhaps some problems that are bin 5 for PaLM 2-S* would be bin 3–4 for a model with different pretraining, but this is not investigated.

**Mitigation status.** The paper does not attempt to solve this limitation; it explicitly acknowledges it as a fundamental boundary condition. The FLOPs-matched comparison (Section 7) is designed in part to quantify this boundary: the finding that test-time compute underperforms pretraining on bin 4–5 problems at moderate and high R values (Figure 9) is presented as evidence for the limits of inference-time scaling. The paper suggests no mechanism for extending test-time compute to genuinely out-of-capability problems, and given the conceptual argument that search can only find solutions that exist in the proposal distribution, this limitation appears inherent to the framework rather than contingent on implementation details.

---

### 6.3 Single Benchmark and Single Model Family Limits Generalizability Claims

**The assumption or constraint.** All experiments are conducted on a single benchmark (MATH, 500 test questions) with a single model family (PaLM 2-S*). The paper states in Section 4 that it "believe[s] this model is representative of the capabilities of many contemporary LLMs," but provides no comparative evidence across model families, scales, or architectures. Similarly, the MATH benchmark consists exclusively of competition-level symbolic mathematics problems — a domain with clean ground-truth answers, multi-step deductive reasoning, and specific difficulty characteristics that may not transfer to other structured prediction tasks.

**The consequence.** Several aspects of the findings could be model-specific or task-specific in ways that would materially affect a practitioner's decision to adopt these methods:

- **PRM quality and over-optimization behavior** depend on PaLM 2-S*'s output distribution. A model with different calibration properties or different error patterns might exhibit different difficulty-dependent scaling curves, different optimal search algorithms per bin, or different over-optimization thresholds. The specific finding that beam search hurts easy-problem performance at high budgets (Figure 3, right, bin 1) could be more or less severe depending on verifier quality.

- **Revision model training** depends on the base model's in-context learning capabilities and its ability to benefit from seeing incorrect-but-close answers. The edit-distance-based pairing strategy (Section 6.1) might be more or less effective for different model families depending on their sequence-level reasoning patterns. The 38% correct-to-incorrect reversion rate is likely model-specific and could vary substantially.

- **Difficulty bin thresholds** (the pass@1 boundaries that separate quintiles) are defined by PaLM 2-S*'s absolute performance on MATH. A stronger base model would compress the easy bins and expand the middle bins; a weaker model would do the opposite. The optimal strategies per bin might shift accordingly.

- **The specific ratios** found optimal for sequential-to-parallel allocation (2¹ to 2³ for medium problems, fully sequential for easy problems) may not generalize to other model families, other tasks, or even other domains within MATH.

**What evidence exists in the paper.** The paper provides no cross-model or cross-benchmark experiments. The 500-question test set is split into five difficulty quintiles of approximately 100 questions each, and two-fold cross-validation further halves these to approximately 50 questions per fold per bin for strategy selection. With such small per-bin samples, the selected optimal strategies could be sensitive to the particular question distribution in MATH. The paper does not report confidence intervals or standard errors on the compute-optimal scaling curves, making it difficult to assess whether the observed differences between strategies are statistically reliable or could reverse with a different sample of questions.

**Mitigation status.** The paper does not address this limitation experimentally. The authors' belief that PaLM 2-S* is "representative" is stated as an assertion in Section 4 without supporting evidence. The paper does not discuss how the findings might change for code generation tasks (where unit tests provide alternative correctness signals), for open-ended generation (where verifier training would require fundamentally different approaches), or for factual reasoning tasks (where the base model's knowledge boundaries differ from mathematical reasoning). The paper also does not discuss how model scale affects the results — whether the optimal strategies, the difficulty bin thresholds, or the over-optimization patterns change as the base model grows, which is directly relevant to practitioners considering whether to apply these methods to larger models.

---

### 6.4 The 14× Larger Model Baseline Is Not Compute-Optimal and Uses No Test-Time Compute

**The assumption or constraint.** The FLOPs-matched comparison in Section 7 scales model parameters while holding training data fixed, following the LLaMA paradigm (Touvron et al., 2023). The authors acknowledge this explicitly:

> "We choose this setting as it is representative of a canonical approach to scaling pretraining compute and leave the analysis of compute-optimal scaling of pretraining compute where the data and parameters are both scaled equally to future work."

Additionally, the 14× larger model uses only **greedy decoding** — no majority voting, no best-of-N, no search, and no revision. This makes the baseline a single-point estimate rather than a test-time-compute-augmented competitor.

**The consequence.** Both choices make the pretraining baseline weaker than it could be, which may **overstate the advantage of test-time compute** in the FLOPs-matched comparison:

- **Non-compute-optimal large model:** A Chinchilla-optimal model (Hoffmann et al., 2022) trained with 14× more total FLOPs would scale both parameters and data, likely outperforming a parameter-only-scaled model at the same total compute. If the 14× larger model in the paper is undertrained relative to Chinchilla-optimal (because it saw the same data as the smaller model despite having more capacity), then the comparison favors test-time compute. The paper does not quantify how much of the reported advantage (e.g., +27.8% on medium problems at R ≪ 1 with revisions) would persist against a compute-optimally trained large model.

- **No test-time compute for the large model:** Giving the 14× larger model even a modest test-time compute budget — say, best-of-8 or best-of-16 — would create a substantially stronger baseline. The paper's claim that test-time compute with a smaller model can "outperform a ~14× larger model" (Section 7) is true in the strict sense tested, but a practitioner evaluating this claim would reasonably ask: "compared to the larger model *with some inference-time optimization of its own*?" The paper does not provide this comparison. The FLOPs-matched framework could in principle accommodate giving both models test-time compute budgets and comparing the optimal allocation for each, but this is not explored.

**What evidence exists in the paper.** The results in Figure 9 and Figure 1 show the comparison as implemented — smaller model with compute-optimal test-time scaling versus larger model with greedy decoding. The paper provides no ablation where the larger model receives any form of test-time compute, nor any analysis of how the crossover points would shift. The R ≫ 1 regime, where the large model's per-token inference cost dominates and test-time compute with the small model underperforms (e.g., −37.2% on hard problems with revisions), could shift further against test-time compute if the large model were given even a small best-of-N budget, since the large model's inference cost is already accounted for in the R ≫ 1 calculation and a small additional budget would not change the total FLOPs dramatically.

**Mitigation status.** The paper acknowledges the non-Chinchilla-optimal training limitation explicitly and frames it as future work, but does not acknowledge the absence of test-time compute for the large model baseline. The discussion in Section 7 treats the FLOPs-matched comparison as a contribution, and the finding that test-time compute can substitute for pretraining compute on easy-to-medium problems is presented as a key result. A more balanced presentation would note that this finding is contingent on the specific (weak) baseline and that a fairer comparison — giving the large model some test-time compute or ensuring Chinchilla-optimal training — might substantially reduce or eliminate the advantage.

---

### 6.5 PRM Search and Revisions Are Never Combined, Despite Complementary Strengths

**The assumption or constraint.** The paper studies PRM-guided search (Section 5) and iterative revisions (Section 6) as independent mechanisms for scaling test-time compute. Section 8 explicitly acknowledges this gap:

> "we did not experiment with PRM tree-search techniques in combination with revisions"

The two mechanisms operate on complementary axes of the proposal-verifier framework laid out in Section 2: revisions modify the proposal distribution (generating better candidates by conditioning on previous attempts), while PRM search modifies how candidates are selected (using learned step-level scores to guide exploration). The paper's own difficulty-dependent findings suggest they have complementary strengths: revisions excel on easy problems (Figure 7, right, bin 1–2: sequential dominates), while search excels on medium problems (Figure 3, right, bin 3–4: beam search outperforms best-of-N).

**The consequence.** The current results represent a **lower bound** on what a fully integrated system could achieve. Several natural combinations are unexplored:

- **Beam search with a revision model as the proposal distribution:** At each step of beam search, the model conditions on previous rejected branches as context (via the revision mechanism), potentially producing higher-quality candidate steps that are more aware of dead ends. This could improve beam search on medium problems where the PRM signal is informative but the base model's initial proposals are suboptimal.

- **PRM-guided revision selection:** Instead of blindly generating a long revision chain and selecting the best answer post-hoc, use the PRM's per-step scores during revision to decide when to continue revising versus when to restart from scratch with a new parallel chain. This could address the 38% correct-to-incorrect reversion problem (Section 6.1) by detecting when a revision is making things worse.

- **Difficulty-adaptive combination:** Use the difficulty estimator not only to choose between search and revisions, but to determine *how* to combine them — e.g., revisions-first-then-search on easy problems, search-with-revision-proposals on medium problems.

Given that the compute-optimal policy already selects between search algorithms and revision ratios per difficulty bin, adding combined strategies to the portfolio could only improve or match the current results. The paper does not provide evidence for how large this gap might be.

**What evidence exists in the paper.** The paper provides no experiments combining PRM search with revisions. The closest it comes is the observation that the PRM trained on base model outputs does not transfer well to revision model outputs due to distribution shift (Appendix J, Figure 15a), which is addressed by training a separate ORM on revision model outputs. However, this ORM is used only for best-of-N selection of completed revision chains, not for step-level guidance during revision or for tree search over revision trajectories. The paper does not discuss what a combined architecture would look like, what additional training would be required (e.g., a PRM trained on revision model outputs for step-level guidance), or what the computational overhead would be.

**Mitigation status.** The paper acknowledges the gap explicitly in Section 8 and identifies it as "an important avenue for future work." No mitigation is attempted. Given that this is a natural and conceptually straightforward extension of the paper's own framework — and one that could potentially break through the performance ceilings that search and revisions individually hit — its absence is one of the more significant omissions in the experimental design. A practitioner reading this paper would reasonably conclude that combining search and revisions is the obvious next step, but would have no evidence from this paper about whether the combination yields additive gains, synergistic gains, or interference effects.

---

### 6.6 Latency and Wall-Clock Time Are Ignored in Favor of Generation Count

**The assumption or constraint.** The paper measures test-time compute exclusively in "generations" — the number of complete solutions sampled — which serves as a proxy for total FLOPs. This metric treats all generations as equivalent regardless of whether they can be executed in parallel or must run sequentially. The paper does not discuss latency, wall-clock time, or hardware utilization as constraints on deployment.

**The consequence.** The compute-optimal policies the paper recommends often favor strategies with substantial sequential dependencies, which have **dramatically different latency characteristics** than parallel strategies at the same generation budget:

- **Sequential revisions (dominant on easy problems, Figure 7):** A fully sequential revision chain of length 64 requires 64 serial forward passes, each depending on the output of the previous one. On hardware with limited memory or without speculative decoding, this translates to roughly 64× the wall-clock time of a single parallel batch of 64 independent samples, even though both use the same number of generations.

- **Beam search (optimal on medium problems, Figure 3):** While beam search is partially parallelizable within each beam expansion step, the overall process is inherently sequential — step t must complete before step t+1 can begin. For problems requiring many reasoning steps, the latency cost could be prohibitive.

- **Hybrid sequential-parallel strategies (optimal for medium-hard problems, Figure 7):** A configuration with √N parallel chains each of length √N (e.g., 8 parallel × 8 sequential = 64 generations) has latency proportional to √N rather than N, which is better than fully sequential but still worse than fully parallel (latency ~1 for embarrassingly parallel best-of-N).

For latency-sensitive applications — interactive assistants, real-time decision-making, API services with strict timeout requirements — the sequential-heavy strategies favored by the compute-optimal policy on easy problems may be **practically infeasible** regardless of their generation-count efficiency. A practitioner serving user requests with a 2-second latency budget cannot run 64 sequential revision steps, even if that strategy is 4× more efficient in total FLOPs.

**What evidence exists in the paper.** The paper provides no latency measurements, no wall-clock time comparisons, and no analysis of how the optimal strategy changes when latency is constrained. The experiments in Figures 4 and 8 sweep generation budgets up to 512 for search and 256 for revisions, but do not report whether the best-performing configurations at each budget level are parallelizable or inherently serial. The FLOPs-matched comparison in Section 7 uses total FLOPs as the metric, which counts a sequential chain of 64 revisions the same as 64 parallel independent samples — ignoring the fact that the former takes 64× longer in wall-clock time.

**Mitigation status.** The paper does not discuss this tradeoff at all. Section 8 identifies future work directions but does not mention latency-aware allocation or throughput-constrained optimization. A latency-aware version of the compute-optimal framework would add a constraint on maximum sequential depth and optimize over strategies that respect it, potentially yielding different recommendations than the current generation-count-optimal policies. This omission is particularly significant because many of the paper's motivating applications (Section 1) involve on-device deployment and interactive use cases where latency is a first-class constraint alongside total compute.
