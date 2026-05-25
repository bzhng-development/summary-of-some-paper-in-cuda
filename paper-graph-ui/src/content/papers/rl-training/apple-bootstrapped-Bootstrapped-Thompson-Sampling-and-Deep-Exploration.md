# Bootstrapped Thompson Sampling and Deep Exploration

**URL:** [https://arxiv.org/pdf/1507.00300](https://arxiv.org/pdf/1507.00300)

## 🎯 Pitch

This technical note introduces a new approach to achieving the exploration behavior of Thompson sampling **without explicitly maintaining or sampling from posterior distributions**, instead using a **bootstrap technique** that augments observed data with artificially generated samples to induce a prior distribution critical for effective exploration.

---

## 1. Executive Summary

This technical note introduces a new approach to achieving the exploration behavior of Thompson sampling **without explicitly maintaining or sampling from posterior distributions**, instead using a **bootstrap technique** that augments observed data with artificially generated samples to induce a prior distribution critical for effective exploration. The method is analyzed on multi-armed bandit problems and extended to episodic reinforcement learning, where the key mechanism — **Bootstrapped Thompson Sampling** (randomized value functions fit to bootstrap resamples of combined real and artificial data) — enables deep exploration with nonlinearly parameterized models such as deep neural networks. The paper demonstrates through simulation that without artificial data, bootstrap-based Thompson sampling fails catastrophically (failing to identify the optimal arm with probability 1 − 2ϵ on a simple two-arm Bernoulli bandit with ϵ = 0.01), but adding as few as two artificially generated observations drives efficient exploration, establishing that prior-inducing synthetic data is essential for bootstrap approximations to match the performance guarantees of true posterior sampling — and that this approach is the only known computationally efficient means of achieving deep exploration with nonlinear function approximators.

## 2. Context and Motivation

### The Core Problem: Exploration Requires Posterior Sampling, But Posterior Sampling Is Intractable for Modern Models

The fundamental tension this paper addresses is deceptively simple: **effective exploration in sequential decision-making requires maintaining and sampling from posterior distributions over models of the world, but the function approximators that power modern AI systems — deep neural networks — make posterior sampling computationally infeasible.** This is not merely an implementation inconvenience; it represents a structural barrier that prevents reinforcement learning systems from tackling problems where efficient exploration is essential.

To understand why this barrier matters, we need to step back and consider what makes exploration principled in the first place. In a sequential decision task — whether a multi-armed bandit where you choose which ad to display, or a reinforcement learning problem where an agent navigates a maze — the agent must constantly decide between two competing objectives: **exploitation** (choosing actions that appear best given current knowledge) and **exploration** (choosing actions that reduce uncertainty and may reveal better alternatives). The Bayes-optimal solution — the one that optimally balances these objectives to maximize long-run expected reward — requires the agent to maintain a posterior distribution over possible environments and to reason about how its beliefs will evolve under different action sequences. This is an intractable computation for all but the simplest problems.

The paper frames its contribution within this broader landscape by identifying two classes of heuristic exploration strategies that have dominated practical and theoretical work, each with severe limitations that the proposed method overcomes.

### The Two Dominant Heuristic Approaches and Their Failures

**Upper Confidence Bound (UCB) methods** represent one popular class of exploration heuristics. The core idea is to assign an "optimism bonus" to actions whose outcomes are poorly understood — essentially, to act as if uncertain actions might yield high rewards, thereby incentivizing the agent to try them. For a broad class of problems, well-designed UCB algorithms enjoy optimal learning rates and strong theoretical guarantees. The paper acknowledges this strength but then identifies a critical practical deficiency:

> "designing, tuning, and applying such algorithms can be challenging or intractable, and as such, upper-conﬁdence bound algorithms applied in practice often suﬀer poor empirical performance."

This is not a throwaway critique. The phrase "can be challenging or intractable" points to a fundamental limitation: UCB methods require constructing confidence sets around value estimates, which demands understanding the statistical properties of the estimator. For a tabular representation (where each state-action pair has its own independent estimate), constructing confidence bounds is conceptually straightforward — you can use Hoeffding's inequality or similar concentration results. But for a deep neural network with millions of parameters processing raw pixels, what does a confidence set even mean? The statistical properties of the jointly learned representation across states are not analytically tractable. One can apply heuristics — adding noise to Q-values, maintaining ensembles and using their variance as a proxy for uncertainty — but these are not the well-designed optimism bonuses that theory requires, and as the paper notes, they "often suffer poor empirical performance."

**Thompson sampling (probability matching)** represents the second major approach. Rather than adding an explicit optimism bonus, Thompson sampling takes a more elegant route: at each decision point, the agent samples one model from its posterior distribution over environments, then acts greedily with respect to that sampled model. The randomness in the sampling procedure provides the exploration signal — if the posterior is uncertain about an action's value, that action will sometimes be sampled as optimal and thus be taken. This mechanism has deep theoretical foundations: the paper cites Russo and Van Roy (2014), which established that Thompson sampling can be viewed as a randomized approximation to a well-designed UCB algorithm, inheriting its optimality properties while being simpler to implement.

Thompson sampling has several attractive properties that have driven its recent resurgence:

- **Empirical performance**: The paper notes that Thompson sampling "received relatively little attention until recently when its strong empirical performance was noted, and a host of analytic guarantees followed." This is a telling historical observation — an algorithm known since the 1930s was largely ignored because it seemed computationally demanding, but gained traction when empirical results demonstrated its effectiveness.

- **Algorithmic simplicity**: The core loop — maintain a posterior, sample from it, act greedily — is conceptually straightforward and requires no explicit tuning of exploration bonuses.

- **Theoretical guarantees**: The connection to UCB algorithms means that Thompson sampling inherits optimal regret bounds for a range of problem classes.

However, the paper identifies the central bottleneck that motivates its entire contribution:

> "Almost all of the literature on Thompson sampling takes the ability to sample from a posterior distribution as given. For many commonly used distributions, this is served through conjugate updates or Markov chain Monte Carlo methods. However, such methods do not adequately accommodate contexts in which models are nonlinearly parameterized in potentially complex ways, as is the case in deep learning."

This paragraph is the intellectual hinge of the paper. Let me unpack what "taking the ability to sample from a posterior as given" actually means and why it fails for deep learning:

**Conjugate updates** work when the prior and likelihood belong to the same exponential family. For example, if you model reward probabilities as Beta-distributed and observe binary outcomes, the posterior is also Beta-distributed, and you can update its parameters by simply adding counts. This is computationally trivial and forms the backbone of Thompson sampling tutorials and textbook implementations. But it requires that each arm or state-action pair have its own independent parameters — there is no sharing of statistical strength across arms, no generalization.

**Markov chain Monte Carlo (MCMC)** methods can handle more complex posteriors by generating samples that asymptotically converge to the target distribution. But MCMC on the weight space of a deep neural network — where the posterior is a distribution over millions of highly correlated parameters shaped by nonlinear transformations — is computationally prohibitive. Each MCMC sample might require thousands of gradient steps over the entire dataset; doing this at every timestep to select an action is simply infeasible at the scale of modern deep RL applications.

**Bootstrapped DQN and related ensemble approaches** existed as heuristics before this work — maintaining an ensemble of neural networks trained on different subsets of data and using the ensemble variance as a proxy for uncertainty. But the paper will argue (and demonstrate empirically in Section 3.1) that these methods, without the prior-inducing mechanism of artificial data, can fail catastrophically at exploration because they lack what the paper calls "deep exploration."

### The Concept of Deep Exploration

The paper uses the term "deep exploration" in Section 4 when discussing reinforcement learning, but the concept pervades the entire work. Deep exploration refers to the agent's ability to take actions that are neither immediately rewarding nor immediately informative, but that position the agent to gain valuable information later. In a maze, this might mean going down a corridor that appears uninteresting to discover whether it leads to a high-reward region that can be exploited in future episodes. In a bandit problem, this means occasionally pulling an arm that has performed poorly in the past to maintain a realistic assessment of whether it might actually be good.

Why is deep exploration hard? Because the agent must reason about the value of information — it must recognize that an action's value includes not just the expected immediate reward, but also the expected improvement in future decision quality that results from the data that action generates. This information value propagates backward through time: to know whether going left is valuable, the agent must know whether the data gained by going left will help it make better decisions at the subsequent choice point, which in turn depends on what it might learn there, and so on. This recursive reasoning is computationally demanding and requires representing uncertainty about the environment in a way that can be propagated forward through planning.

The bootstrap approaches that existed before this paper — including the online bootstrap of Eckles and Kaptein (2014) and the sub-sampling approach of Baransi et al. (2014) — fail at deep exploration because **their posteriors are restricted to the support of the observed data**. If an agent has only experienced low rewards from an action, a bootstrap resample of that experience will consist entirely of low rewards, leading the agent to conclude with certainty that the action is bad. True posterior sampling, by contrast, maintains a prior belief that assigns positive probability to the action being good, and as long as the data hasn't ruled that possibility out, the posterior will occasionally sample optimistic parameter values. The bootstrap without artificial data lacks this prior, and therefore cannot sustain the uncertainty that deep exploration requires.

The paper makes this failure concrete in Section 3.1 with a simple but devastating example: a two-arm Bernoulli bandit where arm 1 deterministically yields reward ϵ (a small value like 0.01) and arm 2 yields reward 1 with probability 2ϵ and 0 otherwise. The optimal action is arm 2 (expected reward 2ϵ vs. ϵ), but with probability at least 1 − 2ϵ, bootstrap-based Thompson sampling without artificial data will observe arm 1 yielding ϵ and arm 2 yielding 0 on the first two pulls, then commit permanently to arm 1 — accumulating linear regret forever. The problem is not that the agent fails to explore sufficiently long; it's that after a single unlucky observation from arm 2, the bootstrap posterior collapses to certainty that arm 2 is worthless. A true Bayesian posterior with an appropriate prior would retain uncertainty, occasionally sampling parameter values where arm 2 is good, and thus continue exploring until it discovers the truth.

This failure mode is not a minor edge case. It illustrates a structural deficiency: **without a prior, the bootstrap cannot represent optimism in the face of negative data**. The paper's contribution is to show that adding artificially generated data — data sampled from a prior distribution over possible observations — fixes this deficiency by giving the bootstrap a notion of what the world might look like beyond what has been observed.

### The Gap in Reinforcement Learning with Generalization

The paper identifies an additional layer of motivation specific to reinforcement learning (Section 4). The vast majority of RL algorithms with theoretical exploration guarantees operate in the "tabula rasa" setting — each state-action pair is treated independently, with no generalization between them. This is the setting of R-MAX (Brafman and Tennenholtz, 2003), UCRL2 (Jaksch et al., 2010), and posterior sampling for RL (Osband et al., 2013). These algorithms maintain explicit counts of visits and rewards for every state-action pair and construct confidence bounds or posterior distributions accordingly. They work beautifully in theory — proving near-optimal regret bounds for finite MDPs — but collapse when faced with state spaces like "all possible configurations of pieces on a Go board" or "all possible raw pixel inputs from an Atari game."

The paper is explicit about this limitation:

> "For most practical systems where the numbers of states and actions is very large or even inﬁnite the ability to generalize is crucial for good performance. Of those algorithms which do combine generalization with exploration, many require an intractable model-based planning step, or are restricted to unrealistic parametric domains."

The reference [15] (Van Roy and Wen, 2014) is particularly important here. That work showed how to combine efficient generalization and exploration via **randomized linearly parameterized value functions** — essentially Thompson sampling where the value function is linear in hand-crafted features, and the posterior over weight vectors can be maintained and sampled analytically. This is a significant advance: it handles generalization (by sharing statistical strength across states through the feature representation) and exploration (through posterior sampling over linear weights). But — and this is the crucial limitation the present paper addresses — linear value functions with fixed features are insufficient for the kinds of perceptual problems where deep learning excels. Playing Atari from pixels requires learning the feature representation itself, which is precisely what deep neural networks do and what makes posterior sampling over their parameters intractable.

The paper's position is that the deep RL revolution — exemplified by the DQN work (Mnih et al., 2015) that achieved superhuman performance on Atari games — has succeeded at generalization but failed at exploration:

> "These algorithms have attained superhuman performance and generated excitement for a new wave of artiﬁcial intelligence, but still fail at simple tasks that require eﬃcient exploration since they use simple exploration schemes that do not adequately account for the possibility of delayed consequences."

The exploration scheme used in DQN is ϵ-greedy: with probability ϵ, take a random action; otherwise, take the action that maximizes the current Q-network's prediction. This is about as simple as exploration gets. It does not maintain uncertainty estimates, does not reason about the value of information, and certainly does not engage in deep exploration. The paper's diagnosis is that this is not a failure of the deep learning community's imagination — it's a consequence of the computational intractability of posterior sampling with deep neural networks. The gap the paper fills is providing a computationally feasible mechanism (the bootstrap with artificial data) that approximates the behavior of Thompson sampling while scaling to deep architectures.

### Prior Work on Bootstrap Approximations and Why It Is Insufficient

The paper explicitly positions itself against existing attempts to use the bootstrap for Thompson-style exploration. The idea of using the bootstrap to approximate a posterior distribution is not new — the paper notes it "has been noted from inception of the bootstrap concept" (Efron, 1979), and the Bayesian bootstrap of Rubin (1981) formalized the connection to Dirichlet process priors. Similarly, Eckles and Kaptein (2014) proposed Thompson sampling with the online bootstrap, and Baransi et al. (2014) proposed a sub-sampling approach for multi-armed bandits.

The paper's contribution is not proposing a bootstrap approximation per se, but rather identifying and fixing a critical flaw in existing bootstrap-based approaches:

> "However, we show that these existing approaches fail to ensure suﬃcient exploration for eﬀective performance in sequential decision problems. As we will demonstrate, the way in which we generate and use artiﬁcial data is critical."

The distinction between "using the bootstrap" and "using the bootstrap with appropriately generated artificial data" is the paper's core technical insight. Without artificial data, the bootstrap's support is restricted to the observed data — a property that causes the catastrophic failure demonstrated in Section 3.1. With artificial data sampled from an appropriate prior distribution, the bootstrap approximates a true Bayesian posterior (the paper establishes equivalence for the Bernoulli bandit case in Section 3.2) and thus inherits the exploration properties of Thompson sampling.

### Parallelizability as a Practical Motivation

A final motivation that the paper emphasizes is computational scalability. Thompson sampling with true posterior sampling is not just intractable — for deep models, even approximate methods like MCMC are sequential by nature (each sample depends on the previous one) and cannot easily leverage parallel computation. The bootstrap approach, by contrast, is **embarrassingly parallelizable**: each bootstrap sample involves training a model on an independently resampled dataset, and these can be done simultaneously on separate hardware. The paper makes this point explicitly in Section 2:

> "our approach is parallelizable and as such scales well to massive complex problems."

And in Section 3, discussing online implementation:

> "In its most naive implementation this parallel bootstrap will have a computational cost per timestep D times larger than a greedy algorithm. However, for speciﬁc function classes such as neural networks it may be possible to share some computation between models and provide signiﬁcant savings."

This is not just a matter of engineering convenience. The ability to distribute the computation across multiple processors or machines makes the difference between an algorithm that can be deployed on large-scale problems and one that is confined to academic benchmarks. The paper even speculates about sharing lower-level features between bootstrap-sampled networks or using dropout masks as implicit bootstrap samples — ideas that later work on Bayesian neural networks and MC Dropout would develop further.

### How the Paper Positions Itself

The paper positions itself at the intersection of two research streams that had been largely separate: the theoretical literature on Thompson sampling (which assumed tractable posterior sampling) and the practical literature on deep reinforcement learning (which achieved impressive results with function approximation but used rudimentary exploration). The contribution is a bridge: a method that approximates Thompson sampling using the bootstrap, supplemented with artificial data to provide the prior that makes exploration work, and that scales to the nonlinear function approximators (deep neural networks) that modern RL systems rely on.

This positioning is reflected in the paper's structure. Sections 3 and 4 develop the method for bandits and RL, respectively, but both are grounded in the same core idea: the bootstrap with artificial data. The bandit setting (Section 3) serves as a proof of concept — demonstrating the failure mode of prior-free bootstraps and the equivalence to Thompson sampling under appropriate artificial data. The RL setting (Section 4) shows how the idea extends to sequential decision problems with delayed consequences, where the bootstrap operates on entire episodes rather than individual observations, and where the "deep exploration" that Thompson sampling enables becomes critical.

The paper's authors — Ian Osband and Benjamin Van Roy — bring substantial theoretical credentials to this work (Van Roy co-authored the foundational paper establishing Thompson sampling's connection to UCB algorithms [1]), and the paper reflects their background: the motivation is rooted in theoretical guarantees, the analysis establishes formal equivalences, but the proposed solution is pragmatic and computationally minded. The paper is essentially arguing: *Thompson sampling has the right theoretical properties, but we can't implement it for deep models. Here's a bootstrap-based approximation that preserves those properties, scales to deep models, and — crucially — requires artificial data to work correctly.*

## 3. Technical Approach

### 3.1 Reader Orientation

This is primarily a **methodology paper** that proposes a practical approximation to Thompson sampling for sequential decision-making, where the core idea is to replace computationally intractable posterior sampling with a bootstrap procedure that augments observed data with artificially generated samples to induce a prior distribution critical for exploration. The system being built is a **randomized decision policy**: at each decision point, the agent fits a model (e.g., a value function) to a bootstrap resample of its experience history combined with synthetic prior data, then acts greedily with respect to that fitted model — the randomness in which data points are included provides the exploration signal, while the synthetic data ensures the agent maintains sufficient uncertainty to continue exploring even after initially unfavorable observations.

### 3.2 Big-Picture Architecture (Diagram in Words)

The system has five major components that interact in a loop:

1. **History Buffer (`$H$`)** — stores all observed data from the agent's interaction with the environment so far. In bandit problems, this is a sequence of (action, observation) pairs. In reinforcement learning, this is a sequence of episodes, each containing (state, action, reward, next-state) trajectories.

2. **Artificial Data Generator (`$\tilde{P}$`)** — a prior distribution over possible observations that generates synthetic data points `$\tilde{H}$` (of size `$M$`) representing what the agent might have seen before any real interaction. This is the crucial component that distinguishes the paper's method from prior bootstrap approaches.

3. **Bootstrap Resampling Engine (`$\mathcal{B}$`)** — the core computational subroutine that takes the combined dataset `$\tilde{H} \cup H$` and produces a single fitted model by resampling from this dataset (either with uniform weights as in Algorithm 1, or with exponential weights as in the Bayesian bootstrap of Algorithm 2) and applying a model-fitting function `$\phi$` to the resampled empirical distribution.

4. **Model Fitting Function (`$\phi$`)** — a problem-specific subroutine that maps a dataset to a model. For bandits, this estimates expected rewards for each arm. For reinforcement learning, this fits a state-action value function `$Q$` (e.g., a deep neural network trained via temporal-difference learning).

5. **Greedy Action Selector** — takes the single sampled model `$\hat{p}$` (bandits) or `$Q$` (RL) and selects the action that maximizes expected reward or value according to that model. This is the exploitation step; exploration emerges from the randomization in which model was sampled.

Information flows as follows: at each timestep, the artificial data generator produces `$M$` synthetic observations from `$\tilde{P}$` → the bootstrap engine combines these with the history `$H$` and produces a single sample model by resampling and fitting → the greedy action selector picks the action that maximizes reward/value under that sampled model → the agent executes the action, observes the outcome, and appends it to `$H$` → the cycle repeats. For the online/parallel variant (Algorithms 5 and 6), `$K$` bootstrap samples are maintained in parallel (each with its own history and weights), updated incrementally as new data arrives, and one is selected uniformly at random at each decision point.

### 3.3 Roadmap for the Deep Dive

- **First**, the standard bootstrap (Algorithm 1) and Bayesian bootstrap (Algorithm 2), since these are the computational primitives that produce the randomized models at the heart of the method — understanding their mechanics and what distributions they approximate is foundational.
- **Second**, the augmentation procedure: how artificial data is generated from `$\tilde{P}$` and combined with observed data, why the ratio `$M/N$` controls the strength of the induced prior, and the connection to Dirichlet process priors that formally justifies this approach.
- **Third**, Bootstrapped Thompson Sampling for multi-armed bandits (Algorithm 3), which instantiates the general framework in the simplest sequential decision setting — this is where the failure mode of prior-free bootstraps is most clearly visible and where the equivalence to Thompson sampling can be established.
- **Fourth**, the extension to episodic reinforcement learning (Algorithm 4), where the data units become entire episodes rather than individual observations, the model is a state-action value function, and the exploration benefits of artificial data enable deep exploration — taking actions not immediately rewarding or informative but that position the agent for future learning.
- **Fifth**, the parallel and incremental variants (Algorithms 5 and 6), which address the scalability concerns that would otherwise make bootstrap-based Thompson sampling computationally prohibitive for online learning with large models.

### 3.4 Detailed, Sentence-Based Technical Breakdown

#### The Standard Bootstrap (Algorithm 1)

The standard bootstrap, introduced by Efron (1979), is a nonparametric method for estimating the sampling distribution of any statistic by resampling from the observed data. The paper presents it as Algorithm 1 with precise pseudocode that makes the computational procedure unambiguous.

The algorithm takes as input: a dataset `$\{x_1, \ldots, x_N\} \subseteq \mathcal{X}$` of `$N$` observations, a function `$\phi: \mathcal{P}(\mathcal{X}) \to \mathcal{Y}$` that maps a probability measure over the data space to some output space (e.g., the expectation operator, a model fitting procedure, or any statistic of interest), and an integer `$K$` specifying the number of bootstrap replicates.

The procedure operates as follows:

1. **Resampling step:** For each replicate `$k = 1, \ldots, K$`, sample `$N$` data points `$x_1^k, \ldots, x_N^k$` from the original dataset `$\{x_1, \ldots, x_N\}$` **uniformly with replacement**. This means each `$x_n^k$` is drawn independently with probability `$1/N$` of selecting any particular original observation. Some original points will appear multiple times in the resample; others will not appear at all.

2. **Empirical distribution construction:** For each resample `$k$`, construct the empirical distribution:
   $$\hat{P}_k(dx) = \sum_{n=1}^N \mathbb{1}(x_n^k \in dx) / N$$
   where `$\mathbb{1}(\cdot)$` is the indicator function (1 if the condition is true, 0 otherwise), and `$dx \subseteq \mathcal{X}$` denotes a measurable subset of the data space. This equation says: for any subset of the data space, the probability mass assigned by `$\hat{P}_k$` is the fraction of the resampled points that fall in that subset. In practice, this means `$\hat{P}_k$` is simply the categorical distribution that puts probability proportional to count on each unique value in the resample.

3. **Statistic computation:** Compute `$y_k = \phi(\hat{P}_k)$` — apply the function of interest to the empirical distribution from the `$k$`-th resample. For example, if `$\phi$` computes the mean, then `$y_k$` is the sample mean of the `$k$`-th bootstrap resample.

4. **Output distribution:** The final output `$\hat{P} \in \mathcal{P}(\mathcal{Y})$` is the empirical distribution of the computed statistics:
   $$\hat{P}(dy) = \sum_{k=1}^K \mathbb{1}(y_k \in dy) / K$$
   This is a distribution over `$\mathcal{Y}$` that approximates the sampling distribution of `$\phi$` applied to the unknown true data-generating distribution.

**What this accomplishes:** The bootstrap provides a nonparametric estimate of the distribution of any statistic `$\phi$` without requiring parametric assumptions about the data-generating process. By treating the observed data as if it were the population and repeatedly resampling from it, the bootstrap simulates the variability that would arise from drawing new datasets from the true distribution. The output `$\hat{P}$` answers questions like "how much would my estimate vary if I collected new data?" or "what is a confidence interval for my parameter?"

**Why this form (sampling with replacement from the empirical distribution):** The key insight is that the empirical distribution of the observed data is the nonparametric maximum likelihood estimate of the true population distribution. By sampling from it, we are sampling from our best estimate of the data-generating process. The "with replacement" aspect is essential — sampling without replacement would simply return the original dataset deterministically (up to permutation), providing no variability. The with-replacement sampling introduces the randomness that captures sampling uncertainty.

**Connection to posteriors:** The paper notes that the output `$\hat{P}$` is "reminiscent of a Bayesian posterior with an extremely weak prior." The "extremely weak" characterization comes from the fact that the support of `$\hat{P}$` is restricted to values that can be formed from the observed data points alone — there is no mechanism for the bootstrap to represent uncertainty about outcomes that have never been observed. This restriction to the observed support is precisely the deficiency that the paper will later address with artificial data.

**Critical limitation for sequential decisions:** Because the empirical distribution `$\hat{P}_k$` puts all its mass on the observed data points `$\{x_1, \ldots, x_N\}$`, any statistic `$\phi(\hat{P}_k)$` can only take values that are possible given the patterns present in the observed data. If an action has only produced reward 0 so far, every bootstrap resample will consist entirely of reward 0, and any estimate of that action's value will be exactly 0 with no uncertainty. There is no mechanism for the bootstrap to say "I've only seen zeros, but the true mean might be positive."

#### The Bayesian Bootstrap (Algorithm 2)

The Bayesian bootstrap, due to Rubin (1981), modifies the standard bootstrap by replacing the uniform resampling with a Dirichlet-weighted scheme that has a formal Bayesian interpretation. The paper presents it as Algorithm 2.

The algorithm takes the same inputs as Algorithm 1 — data `$\{x_1, \ldots, x_N\}$`, function `$\phi$`, and number of replicates `$K$` — but replaces the resampling step with a weight-sampling step.

The procedure:

1. **Weight sampling:** For each replicate `$k = 1, \ldots, K$`, sample `$N$` independent weights `$w_1^k, \ldots, w_N^k \sim \text{Exp}(1)$`, where `$\text{Exp}(1)$` denotes the exponential distribution with rate parameter 1 (mean 1). These are non-negative random variables whose sum is random.

2. **Weighted empirical distribution:** Construct the empirical distribution as:
   $$\hat{P}_k(dx) = \frac{\sum_{n=1}^N w_n^k \mathbb{1}(x_n \in dx)}{\sum_{n=1}^N w_n^k}$$
   This is a probability measure that puts mass proportional to the random weight `$w_n^k$` on each observed data point `$x_n$`. The denominator normalizes so that total mass is 1.

3. **Statistic computation and output:** As in Algorithm 1, compute `$y_k = \phi(\hat{P}_k)$` and output the empirical distribution of the `$y_k$` values.

**What this accomplishes:** The Bayesian bootstrap produces a distribution that can be interpreted as an approximate posterior under a specific nonparametric prior. Unlike the standard bootstrap, which assigns each observation either multiplicity 0, 1, 2, or more in each resample (a discrete set of possibilities), the Bayesian bootstrap assigns continuous weights, yielding a smoother distribution over statistics. This continuous weighting is what creates the formal connection to Dirichlet process priors.

**The Dirichlet process connection:** The paper states that the Bayesian bootstrap's output can be interpreted as a posterior "based on the data and a degenerate Dirichlet prior." A Dirichlet process is a distribution over probability measures — essentially, a prior over possible data-generating distributions. The "degenerate" qualifier means the prior is improper (puts mass on measures that are not absolutely continuous), which is why the posterior support is still restricted to the observed data points. However — and this is the crucial bridge to the augmentation step — if the data space `$\mathcal{X}$` is finite and the artificial dataset `$\{x_{N+1}, \ldots, x_{N+M}\}$` equals the entire space `$\mathcal{X}$`, then as `$K \to \infty$`, the distribution produced by the Bayesian bootstrap **converges to the true Bayesian posterior** conditioned on `$\{x_1, \ldots, x_N\}$` under a uniform Dirichlet prior. This convergence result is what licenses using the bootstrap as a computationally tractable approximation to posterior sampling.

**Why this form (exponential weights vs. uniform resampling):** The exponential weights arise from the normalization of Gamma-distributed random variables to produce Dirichlet-distributed probabilities. Specifically, if `$G_1, \ldots, G_N \sim \text{Gamma}(\alpha, 1)$` are independent Gamma random variables with shape `$\alpha$` and rate 1, then the normalized vector `$(G_1/\sum G_i, \ldots, G_N/\sum G_i)$` follows a Dirichlet distribution with concentration parameters all equal to `$\alpha$`. The Exponential(1) distribution is Gamma(1, 1), corresponding to a Dirichlet posterior with concentration parameter 1 on each observation — which is the posterior under a limiting Dirichlet process prior. This probabilistic structure gives the Bayesian bootstrap its Bayesian interpretation, whereas the standard bootstrap's uniform resampling has no such formal posterior interpretation.

#### The Augmented Bootstrap: Inducing a Prior via Artificial Data

This is the paper's central technical contribution — a modification to either bootstrap algorithm that addresses the critical limitation of restricting the posterior to observed data.

**The augmentation procedure:**

Given observed data `$\{x_1, \ldots, x_N\}$` and a chosen prior strength parameter `$M$` (the number of artificial data points), the agent samples `$M$` artificial data points `$x_{N+1}, \ldots, x_{N+M}$` from a "prior" distribution `$P_0$` over the data space. These are then **appended** to the observed data to form an augmented dataset of size `$N + M$`, and the bootstrap algorithm (either standard or Bayesian) is applied to this combined dataset.

The paper states this in plain terms:

> "In particular, we augment the dataset `$\{x_1, \ldots, x_N\}$` with artiﬁcially generated samples `$\{x_{N+1}, \ldots, x_{N+M}\}$` and apply the bootstrap to the combined dataset. The artiﬁcially generated data can be viewed as inducing a prior distribution."

**What this accomplishes:** The artificial data serves as a mechanism to encode prior beliefs about what observations are possible, even if they have not yet been observed. If the prior distribution `$P_0$` assigns positive probability to an outcome (e.g., an action yielding high reward), then the augmented dataset will contain examples of that outcome, and the bootstrap distribution will retain some probability mass on models that predict that outcome. This prevents the collapse of uncertainty that occurs when only unfavorable outcomes have been observed, enabling the sustained exploration that Thompson sampling provides.

**The ratio `$M/N$` controls prior strength:** The relative number of artificial to real data points determines how strongly the prior influences the bootstrap distribution. When `$N$` (observed data) is small, the artificial data of size `$M$` dominates, and the bootstrap behaves like sampling from the prior. As `$N$` grows, the observed data gradually overwhelms the artificial data, and the bootstrap converges to the true data-generating distribution. The paper is explicit about this controllability:

> "The important thing here is that the relative strength `$M/N$` of the induced prior can be controlled in an explicit manner."

Why this matters: in a Thompson sampling context, the exploration-exploitation tradeoff is implicitly determined by the prior's concentration. A strong prior (large `$M$` relative to `$N$`) encourages more exploration initially; a weak prior (small `$M$`) leads to faster convergence to exploitation but risks premature commitment to suboptimal actions. The artificial data mechanism provides an explicit knob to tune this tradeoff.

**Finite vs. infinite data spaces:** The paper carefully distinguishes two regimes:

- **Finite `$\mathcal{X}$`:** If the data space is finite and the artificial dataset equals the entire space (i.e., one observation per possible value), then the Bayesian bootstrap converges to the exact posterior under a uniform Dirichlet prior. This establishes formal equivalence to Thompson sampling.

- **Large or infinite `$\mathcal{X}$`:** The strategy of generating one artificial observation per possible value "does not scale gracefully" — if `$M$` must be huge to cover the space, the prior overwhelms the observed data. Instead, for large spaces, the paper samples `$M$` artificial data points from a "prior sampling distribution `$P_0$`." This is conceptually equivalent to a Dirichlet process prior with base measure `$P_0$` and concentration parameter `$M$`. The prior `$P_0$` encodes the agent's beliefs about what kinds of observations are plausible, and `$M$` controls the overall weight of the prior relative to the data.

**The Dirichlet process interpretation:** The paper states that this augmentation "corresponds to using a Dirichlet process prior with generator `$P_0$`." A Dirichlet process is a distribution over distributions — it encodes a prior belief about the data-generating process as a whole. The base measure `$P_0$` is the expected distribution (the "mean" of the Dirichlet process), and the concentration parameter (here, `$M$`) controls how tightly the posterior concentrates around `$P_0$`. The augmented bootstrap samples from this posterior by: (1) drawing `$M$` points from `$P_0$` to represent the prior, (2) combining with `$N$` observed points, and (3) bootstrapping the combined dataset. This is computationally tractable where explicit Dirichlet process inference would not be.

**Why this approach for deep learning:** The paper emphasizes that for nonlinear function approximators like deep neural networks, this augmented bootstrap is "especially promising" because it provides a posterior approximation without requiring any explicit representation of the posterior over network weights. The procedure is simply: train the neural network on the augmented dataset. By doing this `$K$` times with different bootstrap resamples (which, in the standard bootstrap case, means different subsets of the augmented data, since each resample selects `$N+M$` points with replacement), we obtain `$K$` different neural networks that represent `$K$` samples from an approximate posterior over functions. The training procedure itself does not need to know about the Bayesian interpretation — it is just standard supervised learning on a resampled dataset.

**Parallelizability:** A key practical advantage the paper highlights: each bootstrap sample is independent, so the `$K$` networks can be trained in parallel on `$K$` separate machines. The computational cost is `$K$` times that of training a single network, but wall-clock time need not increase if `$K$` processors are available. The paper even speculates that computation could be shared: lower-level features extracted by early layers might be similar across bootstrap samples, and weight sharing or tree-structured architectures could reduce the total computational burden. A particularly intriguing suggestion is that dropout — a regularization technique that randomly masks neurons during training — could be used as an implicit bootstrap: "a specially constructed dropout mask for each bootstrap sample" would allow a single network to emulate an ensemble of bootstrapped networks with minimal overhead.

#### Bootstrapped Thompson Sampling for Multi-Armed Bandits (Algorithm 3)

Algorithm 3 instantiates the augmented bootstrap framework in the simplest sequential decision setting: the stochastic multi-armed bandit. This is where the method's mechanics are clearest, and where the paper demonstrates both the failure of prior-free bootstraps and the corrective effect of artificial data.

**Problem formalization:**

The agent faces a finite set of actions `$\mathcal{A}$`. At each time `$t$`, the agent selects an action `$A_t$` and observes an outcome `$Y_{t, A_t} \in \mathcal{Y}$`, from which it receives a known reward `$R(Y_{t, A_t})$`. The "true outcome distribution" `$p^*$` is itself drawn from a family of distributions `$\mathcal{P}$` according to a prior. Conditioned on `$p^*$`, the outcome vectors `$Y_t = (Y_{t,a})_{a \in \mathcal{A}}$` are i.i.d. across time, with each `$Y_t$` distributed according to `$p^*$`. The marginal distribution for action `$a$` is denoted `$p^*_a$`.

The agent's objective is to minimize **Bayesian regret**: the expected difference between the reward of the optimal action (known only in hindsight) and the reward of the actions actually taken, where the expectation is over both the prior over `$p^*$` and the randomness in outcomes and the agent's policy:

$$\text{BayesRegret}(T) = \mathbb{E}\left[\sum_{t=1}^T \mathbb{E}\left[\max_a R(Y_{t,a}) - R(Y_{t, A_t}) \mid p^*\right]\right]$$

where `$T$` is the time horizon, the outer expectation is over the prior on `$p^*$`, and the inner expectation conditions on the true distribution and accounts for outcome randomness.

**What this equation computes:** The Bayesian regret measures how much total reward the agent loses, on average, relative to an oracle that knows `$p^*$` and always picks the best action. The inner expectation computes the per-timestep gap between the optimal action's expected reward (`$\max_a \mathbb{E}[R(Y_{t,a}) \mid p^*]$`) and the chosen action's expected reward (`$\mathbb{E}[R(Y_{t, A_t}) \mid p^*]$`). The sum over `$t$` accumulates this gap across the entire interaction. The outer expectation averages over the prior distribution of environments — the agent doesn't know which `$p^*$` it faces, so its average performance across possible environments is what matters.

**Why this form (Bayesian rather than frequentist regret):** Bayesian regret is the natural objective when the environment is drawn from a known prior. It aligns with the Thompson sampling framework because Thompson sampling is a Bayesian algorithm (it maintains and samples from a posterior) and its theoretical guarantees are typically stated in terms of Bayesian regret. The alternative — frequentist regret, which considers worst-case `$p^*$` — would require a different algorithmic approach (like UCB) and would not directly connect to the posterior sampling mechanism.

**The history `$H_t$`:** At time `$t$`, the agent has observed a history `$H_t = (A_1, Y_{1, A_1}, \ldots, A_{t-1}, Y_{t-1, A_{t-1}})$` of past actions and their outcomes. All decisions at time `$t$` can depend on `$H_t$` and possibly on external randomness `$U_t$` (a sequence of i.i.d. random variables independent of everything else). The agent's policy is a sequence of action distributions `$\pi_t(\cdot) = \mathbb{P}(A_t \in \cdot \mid H_t)$` that are measurable with respect to the history.

**Algorithm 3 procedure:**

The algorithm maintains a history `$H_t$` initialized as empty. At each timestep `$t$`, it performs the following steps:

1. **Sample artificial history:** Generate `$M$` artificial action-observation pairs:
   $$\tilde{H} = ((\tilde{A}_1, \tilde{Y}_1), \ldots, (\tilde{A}_M, \tilde{Y}_M)) \sim \tilde{P}$$
   where `$\tilde{P}$` is the prior sampling distribution over action-observation pairs. The paper notes that `$\tilde{P}$` can be stochastic or deterministic: "As a special case, this subroutine could generate `$M$` deterministic pairs. There can be advantages, though, to using a stochastic sampling routine, especially when the space of action-observation pairs is large and we do not want to impose too strong a prior."

2. **Bootstrap sample:** Apply the bootstrap algorithm `$\mathcal{B}$` to the combined dataset `$\tilde{H} \cup H_t$`, using the function `$\phi = \mathbb{E}[p^* \mid H_{t+M} = \cdot]$` and requesting `$K = 1$` sample:
   $$\hat{P} = \mathcal{B}(\tilde{H} \cup H_t, \mathbb{E}[p^* \mid H_{t+M} = \cdot], K = 1)$$
   The function `$\mathbb{E}[p^* \mid H_{t+M} = \cdot]$` maps a specified history of action-observation pairs to a probability distribution over possible reward distributions — it is the model-fitting step that, given a dataset, produces an estimate of what the true distribution `$p^*$` might be.

3. **Sample a model:** Draw `$\hat{p} \sim \hat{P}$`. Since `$K=1$`, `$\hat{P}$` is a point mass at a single fitted model, so this step simply retrieves that model.

4. **Greedy action selection:** Select the action that maximizes expected reward under the sampled model:
   $$A_t \in \arg\max_a \mathbb{E}[R(Y_{t,a}) \mid \hat{p}]$$
   where the expectation is with respect to the outcome distribution specified by `$\hat{p}$`.

5. **Observe and update:** Observe the actual outcome `$Y_{t, A_t}$` for the chosen action, and append `$(A_t, Y_{t, A_t})$` to the history: `$H_{t+1} = H_t \cup \{(A_t, Y_{t, A_t})\}$`.

**What this accomplishes:** At each timestep, the agent pretends the world is described by a single model `$\hat{p}$` drawn from the bootstrap distribution over possible worlds, then acts optimally for that pretend world. The randomness in which bootstrap resample is drawn — which observations are included with what weights — creates the exploration: if the posterior is uncertain about action `$a$`, then some bootstrap resamples will include observations that make `$a$` look good (especially if the artificial data encodes optimistic prior beliefs), and the agent will occasionally select `$a$` as a result.

**Why this form (single bootstrap sample per step):** The `$K = 1$` setting is the direct analog of Thompson sampling's "sample one parameter from the posterior" step. Thompson sampling does not average over the posterior; it samples once and commits to the greedy action for that sample. Using `$K=1$` in the bootstrap replicates this behavior exactly. Using `$K > 1$` and averaging would produce a different (and typically less exploratory) algorithm — the exploration benefit comes from the randomization, and averaging would dilute it.

**The constant-cost online variant:** The paper acknowledges a practical problem: "One drawback of Algorithm 3 is that the computational cost per timestep grows with the amount of data `$H_t$`." Running a full bootstrap on the entire history at every timestep is `$O(t)$` per step, which becomes prohibitive for long horizons. The solution proposed is to maintain `$D$` online bootstrap models in parallel and sample uniformly among them. Instead of generating a new bootstrap sample at each timestep from scratch, each of the `$D$` models is trained incrementally (e.g., via stochastic gradient descent) on a different bootstrap weighting of the data, and the agent randomly selects one model to act greedily with respect to at each step. This reduces the per-step cost to `$O(D)$` independent of `$t$`. The paper notes that for neural networks, the `$D$` models might share lower-level features, further reducing the effective cost.

#### Simulation Demonstration and the Failure of Prior-Free Bootstraps (Section 3.1)

The paper provides a minimal, deliberately simple example that exposes the structural failure of bootstrap-based Thompson sampling without artificial data. The experimental design is as follows:

**Setup:** Two actions (`$\mathcal{A} = \{1, 2\}$`), continuous outcomes in `$[0, 1]$`, and the reward is the outcome itself (`$R(y) = y$`). The true distribution `$p^*$` is parameterized by a small constant `$0 < \epsilon \ll 1$` (in the simulations, `$\epsilon = 0.01$`):

- **Action 1:** Deterministic reward `$\epsilon$` — `$p^*_1(y) = \delta_\epsilon(y)$`, where `$\delta_x$` is the Dirac delta function putting all probability mass on `$x$`. This arm always yields reward exactly `$\epsilon$`.
- **Action 2:** Stochastic reward — `$p^*_2(y) = (1 - 2\epsilon)\delta_0(y) + 2\epsilon\delta_1(y)$`. With probability `$1 - 2\epsilon$`, it yields 0; with probability `$2\epsilon$`, it yields 1. The expected reward is `$2\epsilon$`.

The optimal action is arm 2 (expected reward `$2\epsilon$` vs. `$\epsilon$` for arm 1), but arm 2 yields zero reward `$1 - 2\epsilon$` of the time, while arm 1 always yields positive (though small) reward.

**Failure mode with `$M = 0$` (no artificial data):** Without artificial data, the agent must begin by sampling each arm at least once (the bootstrap has no information and must try actions to get data). With probability `$1 - 2\epsilon$` — which is approximately 0.98 when `$\epsilon = 0.01$` — the first pull of arm 2 yields reward 0. The agent's history after one pull of each arm is then: arm 1 → reward `$\epsilon$`, arm 2 → reward 0. Under any bootstrap resample of this history, every observation from arm 2 is 0 (since that is the only observation available), and every observation from arm 1 is `$\epsilon$`. Thus:

- For arm 1: the bootstrap estimate of expected reward is always `$\epsilon$`.
- For arm 2: the bootstrap estimate of expected reward is always 0.

The agent will therefore prefer arm 1 at every subsequent timestep, never pulling arm 2 again, and thus never discovering that arm 2 can yield reward 1. It commits to a suboptimal action forever, accumulating linear regret. This is not a matter of insufficient exploration budget — the agent simply has no mechanism to represent the possibility that arm 2 might be good after observing a single zero. The support of the bootstrap distribution is restricted to `$\{0\}$` for arm 2, and no amount of additional computation will change that.

**Why true Thompson sampling does not fail here:** True Thompson sampling with a Beta prior would place a Beta(1,1) (uniform) prior on each arm's success probability. After observing one zero from arm 2, the posterior would be Beta(1,2) — still assigning substantial probability mass to high success probabilities. The agent would occasionally sample `$\theta_2 > \epsilon$` from this posterior, pull arm 2, and if lucky, observe a 1, updating the posterior to Beta(2,2), which is even more favorable. This self-correcting behavior is what makes Thompson sampling explore efficiently. The bootstrap without artificial data lacks this prior-induced optimism.

**Corrective effect of artificial data with `$M = 2$`:** The paper's simulation adds `$M = 2$` artificial data points generated by a distribution `$\tilde{P}$` that selects each action once and samples an observation uniformly from `$[0, 1]$` for each. This means the augmented dataset, after one real pull of each arm, contains:

- Arm 1: real observation `$\epsilon$` + artificial observation Uniform(0,1)
- Arm 2: real observation 0 + artificial observation Uniform(0,1)

When the bootstrap resamples from this combined dataset, the artificial observations provide variability: sometimes arm 2's artificial observation is high, making arm 2 look better than arm 1 in that resample. The agent will occasionally pull arm 2 as a result. If it pulls arm 2 and observes a 0 again, there are now two zeros for arm 2, but the artificial observation still provides some probability of optimistic resamples. Eventually, if it pulls arm 2 and observes a 1, the balance shifts, and arm 2 starts being selected more often. The artificial data provides the "optimism in the face of uncertainty" that the prior provides in true Thompson sampling.

**Figure 1 results (described in prose):** The paper presents six subplots of cumulative regret over time for three bootstrap variants (standard Bootstrap, Bayesian BayesBootstrap, and BESA from Baransi et al., 2014), each with and without artificial data (`$M=0$` top row, `$M=2$` bottom row). Across 20 Monte Carlo simulations:

- **Top row (`$M=0$`):** All three bootstrap variants show rapidly growing (linear-looking) cumulative regret, indicating they have committed to the suboptimal arm and never recovered. The Bayesian bootstrap and standard bootstrap perform similarly poorly; BESA performs marginally better but still shows substantial regret.
- **Bottom row (`$M=2$`):** All three variants show sublinear regret — the regret curves flatten over time, indicating that the per-step regret goes to zero as the agent learns to prefer arm 2. The standard bootstrap and Bayesian bootstrap with artificial data both effectively learn the optimal action; BESA shows higher variance but still substantially lower regret than without artificial data.

The paper draws two conclusions: (1) "the choice of bootstrap method makes little difference," and (2) "injecting artiﬁcial data is crucial to incentivizing eﬃcient exploration." The first point is practically significant — it means the simpler standard bootstrap (which requires no weight sampling) works as well as the more theoretically grounded Bayesian bootstrap for exploration purposes. The second point is the central thesis of the paper.

**A note on BESA:** The paper briefly describes BESA (Baransi et al., 2014) as a subsampling variant "that applies to two armed bandit problems" where "the algorithm estimates the reward of each arm by drawing a sample average (with replacement) with sample size equal to the number of times the other arm has been played." The paper critiques BESA as not generalizing "gracefully to settings with dependent arms" and notes that it still fails without artificial data, though perhaps less catastrophically than the other methods. This positions the augmented bootstrap as a more general and principled solution than ad-hoc subsampling schemes.

#### Formal Equivalence to Thompson Sampling (Section 3.2)

The paper establishes that, for certain choices of bootstrap algorithm and artificial data, Bootstrapped Thompson Sampling is **exactly equivalent** to traditional Thompson sampling with a conjugate prior. This equivalence is important because it means the theoretical regret bounds developed for Thompson sampling (e.g., the `$O(\sqrt{T})$` Bayesian regret bounds of Russo and Van Roy, 2014; Agrawal and Goyal, 2013) transfer directly to the bootstrap algorithm.

**The Bernoulli bandit example:**

Consider a multi-armed bandit with independent arms, where each arm `$a$` generates binary rewards (0 or 1) from a Bernoulli distribution with unknown mean `$\theta_a$`. Suppose the agent's prior for each `$\theta_a$` is `$\text{Beta}(\alpha_a, \beta_a)$` — a Beta distribution with shape parameters `$\alpha_a$` (pseudocount of successes) and `$\beta_a$` (pseudocount of failures).

Let the agent have observed `$n_a^0$` rewards of zero and `$n_a^1$` rewards of one for arm `$a$`. Under the Beta-Bernoulli conjugate model, the posterior for `$\theta_a$` is:

$$\theta_a \mid \text{data} \sim \text{Beta}(\alpha_a + n_a^1, \beta_a + n_a^0)$$

A sample `$\hat{\theta}_a$` from this posterior can be generated via the following procedure (which is a standard property of Gamma-distributed normalization):

1. Sample independent exponential random variables: `$x_1, \ldots, x_{\alpha_a + n_a^1} \sim \text{Exp}(1)$` and `$y_1, \ldots, y_{\beta_a + n_a^0} \sim \text{Exp}(1)$`.
2. Compute:
   $$\hat{\theta}_a = \frac{\sum_{i=1}^{\alpha_a + n_a^1} x_i}{\sum_{i=1}^{\alpha_a + n_a^1} x_i + \sum_{j=1}^{\beta_a + n_a^0} y_j}$$

**What this equation computes:** The ratio of the sum of `$\alpha_a + n_a^1$` exponential random variables (representing successes) to the total sum of all `$\alpha_a + \beta_a + n_a^0 + n_a^1$` exponential random variables (representing all observations). This ratio follows a `$\text{Beta}(\alpha_a + n_a^1, \beta_a + n_a^0)$` distribution. Operationally, it provides a single sample from the posterior over arm `$a$`'s success probability: you draw exponential noise, sum the "success" and "failure" terms separately, and take the success proportion.

**Why this form (exponential sampling for Beta generation):** The Beta distribution can be expressed as `$\text{Gamma}(a,1) / (\text{Gamma}(a,1) + \text{Gamma}(b,1))$` where the two Gamma random variables are independent. A `$\text{Gamma}(k, 1)$` random variable with integer shape `$k$` is the sum of `$k$` independent `$\text{Exp}(1)$` random variables. So the posterior sampling procedure is: generate `$\alpha_a + n_a^1$` exponential RVs (one per prior + observed success), generate `$\beta_a + n_a^0$` exponential RVs (one per prior + observed failure), sum each group, and normalize. This decomposition is the key to the bootstrap connection.

**Equivalence to the Bayesian bootstrap with artificial data:**

The paper observes that this sampling procedure is **identical** to applying the Bayesian bootstrap (Algorithm 2) to an augmented dataset that contains, for each arm `$a$`:

- `$\alpha_a$` artificial observations of reward 1 (representing prior successes)
- `$\beta_a$` artificial observations of reward 0 (representing prior failures)
- `$n_a^1$` real observations of reward 1
- `$n_a^0$` real observations of reward 0

The Bayesian bootstrap assigns an `$\text{Exp}(1)$` weight to each observation (both real and artificial), sums the weights for each outcome category, and computes the proportion for reward 1. This is mathematically identical to the exponential sampling procedure above.

Therefore, if the artificial data generator `$\tilde{P}$` is configured to produce `$\alpha_a + \beta_a$` total artificial observations for arm `$a$`, with `$\alpha_a$` of them being reward 1 and `$\beta_a$` being reward 0, then Bootstrapped Thompson Sampling with the Bayesian bootstrap is **exactly** Thompson sampling with a `$\text{Beta}(\alpha_a, \beta_a)$` prior.

**What this equivalence means:** Any theoretical regret bound proven for Thompson sampling with Beta priors applies directly to Bootstrapped Thompson Sampling with appropriately configured artificial data. The paper cites two key results:

- Russo and Van Roy (2014): Thompson sampling approaches the performance of a well-tuned UCB algorithm, establishing a connection to optimal learning rates.
- Agrawal and Goyal (2013): Thompson sampling achieves `$O(\sqrt{T})$` Bayesian regret for Bernoulli bandits, matching the information-theoretic lower bound up to constants.

The paper states these equivalences "imply that theoretical regret bounds previously developed for Thompson sampling apply to the BootstrapThompson algorithm with the Bayesian bootstrap and appropriately generated artiﬁcial data."

**Generalization to Dirichlet-multinomial models:** The paper notes that the equivalence "can easily be generalized to the case where each arm generates rewards from among a ﬁnite set of possibilities with probabilities distributed according to a Dirichlet prior." If outcomes can take `$K$` possible values and the prior is Dirichlet with concentration parameters `$\alpha_1, \ldots, \alpha_K$`, then the Bayesian bootstrap with `$\alpha_1 + \cdots + \alpha_K$` artificial observations (drawn according to those concentrations) reproduces Thompson sampling exactly. This covers categorical rewards, multinomial outcomes, and any setting with finite outcome spaces and Dirichlet priors.

**The broader conjecture:** The paper goes further, stating: "We expect that, with appropriately designed schemes for generating artiﬁcial data, such equivalences can also be established for a far broader range of problems." This is a forward-looking claim rather than a proven result — it suggests that the augmented bootstrap framework can approximate Thompson sampling for non-conjugate, continuous, and nonlinearly parameterized models, even though exact equivalence may not hold. The intuition is that the bootstrap approximates the posterior (as has been known since Efron, 1979), and the artificial data provides the prior, so the combined procedure approximates posterior sampling with that prior. The quality of the approximation depends on how well the bootstrap captures posterior uncertainty for the specific model class, but the bandit equivalence provides a proof of concept.

**Why this is important for the RL extension:** The equivalence in the bandit setting provides the theoretical foundation for extending the method to reinforcement learning. In RL, the model is a value function rather than a simple reward distribution, and the "observations" are entire trajectories, but the same principle applies: the bootstrap with artificial data approximates posterior sampling over value functions, and the artificial data encodes a prior that maintains exploration. The bandit analysis demonstrates that the core mechanism (bootstrap + artificial data = approximate posterior sampling) works, and the RL extension applies this mechanism to the more complex sequential setting.

#### Extension to Episodic Reinforcement Learning (Algorithm 4)

Algorithm 4 extends the bootstrapped Thompson sampling framework from bandits to episodic reinforcement learning with delayed consequences. This is the setting where the "deep exploration" property of Thompson sampling becomes critical, and where the method's compatibility with deep neural network function approximators is most valuable.

**Episodic MDP setting:**

The agent interacts with an environment over repeated episodes of fixed length `$\tau$`. In each episode `$l = 1, 2, \ldots$`, the agent observes a sequence of states `$s_l^1, \ldots, s_l^\tau$`, selects actions `$a_l^1, \ldots, a_l^\tau$` according to a policy `$\pi$`, and receives rewards `$r_l^1, \ldots, r_l^\tau$`. After each action, the environment transitions to the next state `$s_l^{t+1}$` according to unknown dynamics. The agent's objective is to maximize the expected sum of rewards over episodes, despite initial uncertainty about the environment's dynamics and reward structure.

**State-action value functions:** A central concept is the state-action value function `$Q$`, which estimates the expected cumulative reward from taking action `$a$` in state `$s$` at time `$t$` and following the optimal policy thereafter:

$$Q_t(s, a) \approx \mathbb{E}\left[\sum_{k=t}^\tau r_l^k \mid s_l^t = s, a_l^t = a, \pi^*\right]$$

where `$\pi^*$` is the optimal policy. Given `$Q_t$`, the greedy action at state `$s$` and time `$t$` is `$\arg\max_a Q_t(s, a)$`.

**Algorithm 4 procedure:**

The algorithm maintains a history `$H$` of episodes, initially empty. Each element of `$H$` is a complete episode trajectory: a sequence of states, actions, and rewards of length `$\tau$`. At the start of each episode:

1. **Sample artificial history:** Generate `$M$` artificial episodes from a prior distribution `$\tilde{P}$` over episode trajectories:
   $$\tilde{H} = (\tilde{s}_1^1, \tilde{a}_1^1, \tilde{r}_1^1, \ldots, \tilde{s}_1^\tau, \tilde{a}_1^\tau, \tilde{r}_1^\tau, \ldots, \tilde{s}_M^1, \tilde{a}_M^1, \tilde{r}_M^1, \ldots, \tilde{s}_M^\tau, \tilde{a}_M^\tau, \tilde{r}_M^\tau) \sim \tilde{P}$$

   The paper suggests a concrete approach to generating this artificial data: "sample state-action pairs from a diﬀusely mixed generative model and assign them stochastically optimistic rewards and random state transitions." A "diffusely mixed generative model" means a distribution over states and actions that covers the state-action space broadly, ensuring the prior is not concentrated in a narrow region. "Stochastically optimistic rewards" means rewards that are high on average, encoding the prior belief that unexplored regions of the state space might be valuable — this is the RL analog of the optimistic prior that made bandit exploration work in Section 3.1.

2. **Bootstrap sample a Q-function:** Apply the bootstrap algorithm `$\mathcal{B}$` to the combined dataset `$\tilde{H} \cup H$`, using a value function fitting procedure `$\phi$` as the statistic and requesting `$K = 1$` sample:
   $$\hat{P} \leftarrow \mathcal{B}(\tilde{H} \cup H, \phi, K = 1)$$
   The function `$\phi$` takes a dataset of episodes and produces a state-action value function `$Q$`. This could be, for example, fitted Q-iteration: repeatedly applying the Bellman backup to the observed transitions and regressing a function approximator (like a deep neural network) to the targets. The paper is generic about the specific `$\phi$`, noting only that it "could output a deep neural network trained to ﬁt a state-action value function via least-squares value iteration."

3. **Sample a Q-function:** Draw `$Q \sim \hat{P}$`. Since `$K=1$`, this retrieves the single fitted Q-function from the bootstrap resample.

4. **Execute episode greedily:** For `$t = 1, \ldots, \tau$`, at state `$s_l^t$`, select action `$a_l^t \in \arg\max_\alpha Q_t(s_l^t, \alpha)$`, observe reward `$r_l^t$` and transition to `$s_l^{t+1}$`.

5. **Update history:** Append the completed episode trajectory to `$H$`:
   $$H \leftarrow H \cup \{(s_l^1, a_l^1, r_l^1, \ldots, s_l^\tau, a_l^\tau, r_l^\tau)\}$$

**What this accomplishes:** Before each episode, the agent generates a single randomized Q-function by fitting a value function approximator to a bootstrap resample of combined real and artificial episode data. It then follows the greedy policy with respect to that Q-function for the entire episode — no further randomization within the episode. The exploration comes from the variability across episodes: different bootstrap resamples produce different Q-functions, which induce different behaviors. Some of these behaviors will be exploratory, visiting states and trying actions that the agent's current point estimate might avoid, because the bootstrap resample happened to include data (especially artificial data) that makes those actions look promising.

**Why whole episodes as data units:** The bootstrap operates on entire episodes, not individual transitions, because the value function `$\phi$` needs coherent trajectories to learn about delayed consequences. A single transition `$(s, a, r, s')$` provides information about immediate reward and next state, but fitting a Q-function via temporal-difference learning requires sequences of transitions to propagate value information backward in time. Treating episodes as atomic data units preserves the temporal structure within each episode, allowing `$\phi$` to capture long-term dependencies.

**Deep exploration:** The paper claims that Algorithm 4 "enjoys the beneﬁts of what we call deep exploration in that it sometimes selects actions which are neither exploitative nor informative in themselves, but that are oriented toward positioning the agent to gain useful information downstream in the episode." This is the key distinction from simpler exploration schemes like `$\epsilon$`-greedy or Boltzmann exploration, which randomize at the level of individual actions without considering the informational consequences. Deep exploration emerges from the posterior sampling mechanism: the sampled Q-function represents a coherent hypothesis about the entire environment (not just local values), and acting greedily with respect to that hypothesis naturally leads to trajectories that test the hypothesis. If the hypothesis says "there's a high-reward region behind that door," the agent will go through the door, not because going through the door is intrinsically exploratory, but because the sampled model says it's optimal.

**Why this is novel:** The paper states that "the general approach represented by this algorithm may be the only known computationally eﬃcient means of achieving deep exploration with nonlinearly parameterized representations such as deep neural networks." Prior approaches to deep exploration either required tractable posterior sampling (which doesn't scale to deep networks), explicit model-based planning with uncertainty propagation (computationally prohibitive for large state spaces), or linear parameterizations (which don't capture the representational power of deep learning). The bootstrap approach sidesteps all of these: it requires only the ability to train a model on a weighted/resampled dataset, which for neural networks is standard supervised learning.

**Incorporating prior experience:** The paper notes that when prior data is available — e.g., "episodes of experience with actions selected by an expert agent" — it can be included in the artificial data `$\tilde{H}$`. This "oﬀers a means of incorporating apprenticeship learning as a springboard for the learning process." The expert demonstrations seed the prior with good behaviors, and the bootstrap exploration refines and improves upon them. This is a natural way to combine imitation learning with reinforcement learning: the artificial data provides an initial policy, and the bootstrap resampling maintains uncertainty about whether deviations from that policy might be even better.

#### Parallel and Incremental Variants (Algorithms 5 and 6)

The paper recognizes that refitting a model from scratch on the entire history at every episode is computationally prohibitive for large-scale applications. Algorithms 5 and 6 address this by introducing incremental, parallelized versions.

**Algorithm 5: Incremental Bayesian Bootstrap Sample**

This is a subroutine that performs a single Bayesian bootstrap update incrementally. It takes as input the dataset `$x_1, \ldots, x_N$`, previously sampled weights `$w_1, \ldots, w_{N-1}$` (one per existing data point), the model-fitting function `$\phi$`, and the current model (implicitly).

1. **Sample new weight:** Generate `$w_N \sim \text{Exp}(1)$` for the new data point. The weights for existing points are already sampled and stored.
2. **Construct weighted empirical distribution:** 
   $$\hat{P}(dx) = \frac{\sum_{n=1}^N w_n \mathbb{1}(x_n \in dx)}{\sum_{n=1}^N w_n}$$
3. **Update model:** Apply `$\phi$` to the entire reweighted dataset. In practice, this would be an incremental update to an existing model — for neural networks, this might mean a few gradient steps with the new data point and with all data points weighted by their respective `$w_n$`.

**Algorithm 6: Incremental RL with Bootstrapped Value Function Randomization**

This is the scalable version of Algorithm 4. It maintains `$K$` parallel bootstrap models (e.g., `$K$` deep neural networks), each with its own weight vector and history. The two-phase structure:

**Initialization (executed once):**

- For each bootstrap model `$k = 1, \ldots, K$` (in parallel), initialize an empty history `$H_k$` and sample `$M$` artificial episodes `$\tilde{H}_k \sim \tilde{P}$` independently. Each model gets its own independent set of artificial data.

**Per-episode loop (executed for `$l = 1, 2, \ldots$`):**

1. **Parallel model update:** For each `$k = 1, \ldots, K$` (in parallel), apply the incremental bootstrap to update model `$Q_k$` using the artificial data and the model's private history. This is the step where new data from the previous episode is incorporated with its `$\text{Exp}(1)$` weight.
2. **Model selection:** Sample `$k \sim \text{Uniform}(1, \ldots, K)$` — uniformly at random among the `$K$` models.
3. **Episode execution:** For `$t = 1, \ldots, \tau$`, act greedily with respect to `$Q_k$`: `$a_l^t \in \arg\max_\alpha Q_k^t(s_l^t, \alpha)$`, observe reward and transition.
4. **Parallel history update:** For each `$k = 1, \ldots, K$` (in parallel), append the completed episode to `$H_k$`. Crucially, **all** `$K$` models observe the episode, not just the one that was selected for action. This means all models are trained on all data, but with different random weights, creating diversity in their predictions.

**What this accomplishes:** The `$K$` models encode `$K$` independent bootstrap samples from the (approximate) posterior over Q-functions. The uniform selection at each episode implements the Thompson sampling behavior: with probability `$1/K$`, the agent acts according to each possible world model. The exploration comes from the diversity across models — some will be optimistic about unexplored regions, leading to exploratory behavior when selected. The exploitation comes from the fact that all models are trained on all real data, so as data accumulates, they converge to similar predictions in well-explored regions.

**Why this form (uniform selection among `$K$` models):** The uniform selection approximates drawing a single bootstrap sample per episode (as in Algorithm 4's `$K=1$`), but amortizes the computational cost. Rather than generating a fresh bootstrap sample from scratch each episode (which would require `$O(K)$` model training per episode if you wanted `$K$`-sample diversity), Algorithm 6 maintains `$K$` persistent models and randomly selects one per episode. The total computational cost per episode is `$O(K)$` model updates (one per model), but these are parallelizable, and each update is incremental (not from scratch). This makes the approach feasible for online learning with large models.

**The experience replay connection:** The paper notes that this approach "is akin to training each model using experience replay, but with past experiences weighted randomly to induce exploration." Experience replay is the standard technique in deep RL (popularized by DQN) where transitions are stored in a buffer and sampled randomly for training. The key difference here is the weighting: each model sees all past experiences (not a random subset), but with different `$\text{Exp}(1)$` weights, so some experiences influence certain models more than others. These weight differences are what create the diversity across models that drives exploration.

**Why `$K$` separate histories:** Each model maintains its own history `$H_k$` because the weights `$w_n^k$` are history-specific — model `$k$`'s weight for a particular observation depends on the random exponential draw for that model. The data (the episode trajectories) is shared across all models, but the weights differ, so each model effectively sees a different "perspective" on the same data. This is the bootstrap principle in an online, incremental form.

**Practical considerations for neural networks:** The paper suggests that with neural network function approximators, the `$K$` models need not be fully independent. Lower-level features (early layers of the network) might be shared across models, with only the higher-level layers differing. This could be implemented via a multi-headed architecture: one shared "body" network that processes raw inputs into a feature representation, and `$K$` separate "head" networks that map features to Q-values, each trained with different bootstrap weights. The paper also speculates that dropout masks could serve as implicit bootstrap samples, an idea that anticipates the later work on MC Dropout as approximate Bayesian inference (Gal and Ghahramani, 2016), though the paper does not develop this idea beyond a brief mention.

#### Summary of Design Choices and Their Justifications

- **Bootstrap over MCMC or conjugate updates:** The bootstrap requires only the ability to fit a model to a weighted/resampled dataset, which for neural networks is standard supervised learning. MCMC in weight space is computationally prohibitive; conjugate updates don't exist for deep architectures. The bootstrap scales naturally with the model fitting procedure already in use.

- **Artificial data over prior-free bootstrap:** Without artificial data, the bootstrap distribution's support is restricted to observed outcomes, causing catastrophic collapse of uncertainty when only unfavorable observations exist (the two-arm bandit example). Artificial data sampled from a prior distribution provides the "optimism in the face of uncertainty" that makes Thompson sampling explore effectively.

- **Exponential weights (Bayesian bootstrap) over uniform resampling:** The Bayesian bootstrap has a formal interpretation as a Dirichlet process posterior, which enables establishing exact equivalence to Thompson sampling for conjugate models and provides a principled framework for prior specification. However, the paper's empirical results (Section 3.1) show that the standard bootstrap works equally well, suggesting the choice may not matter much in practice.

- **Episodes as data units in RL:** Preserving temporal coherence within episodes is necessary for value function fitting to capture delayed consequences. Bootstrapping at the episode level (rather than the transition level) mirrors how Thompson sampling over MDPs would resample entire models of the environment.

- **Parallel `$K$` models with shared data but different weights:** This amortizes the cost of bootstrap resampling over time, making the approach feasible for online learning. The uniform selection among models approximates the per-episode resampling of Algorithm 4 without the need to refit from scratch.

- **Stochastically optimistic artificial rewards in RL:** Encoding the prior belief that unexplored regions might yield high rewards is the RL analog of the Beta prior in bandits. It ensures that bootstrap Q-functions remain optimistic about unseen state-action pairs, driving deep exploration.

## 4. Key Insights and Innovations

### Innovation 1: Artificial Data Is Not an Implementation Detail — It Is the Mechanism That Makes Bootstrap-Based Exploration Work

The paper's most distinctive conceptual contribution is a diagnostic insight rather than a new algorithm: **existing bootstrap-based approaches to Thompson sampling fail not because the bootstrap is a poor posterior approximation in general, but because they lack a mechanism to represent beliefs about outcomes that have never been observed.** The simulation in Section 3.1 makes this point with surgical clarity: with probability approximately 0.98 on a simple two-arm bandit, three different bootstrap variants without artificial data commit to a suboptimal action permanently and never recover. Add two artificial observations, and all three variants learn the optimal action efficiently.

What makes this insight fundamental rather than incremental is that it **reconceptualizes the role of the prior in sequential decision-making.** Before this work, one could reasonably have believed that the bootstrap — by resampling observed data to capture estimation uncertainty — would naturally produce the kind of posterior uncertainty that drives exploration. After all, the bootstrap is a standard tool for constructing confidence intervals; if it captures uncertainty about parameter estimates, shouldn't it capture the uncertainty that drives exploration? The paper's answer is a definitive **no**, and the reason is subtle: exploration requires uncertainty not just about the *value* of observed outcomes, but about the *possibility* of unobserved outcomes. The bootstrap without a prior has no mechanism to say "I've only seen zeros from this arm, but it might actually produce ones." Its support is the support of the data. This is perfectly adequate for i.i.d. uncertainty quantification, but catastrophic for sequential decision-making where the data you collect depends on the actions you take, and ceasing to take an action means you will never see evidence that might change your mind.

The paper frames this as a distinction between two kinds of uncertainty that the bootstrap captures or fails to capture:

- **Within-support uncertainty:** Given the outcomes observed so far, how much would my estimate vary if I collected more data from the same distribution? The standard bootstrap captures this well — resampling simulates alternative datasets and produces variation in estimates.
  
- **Out-of-support uncertainty:** Could there be outcomes I haven't seen yet? Could this action that has always produced zero actually produce a one? The standard bootstrap cannot represent this — if an outcome has never appeared in the data, it has zero probability in every resample.

True Thompson sampling with a Bayesian posterior handles both kinds of uncertainty because the prior places positive probability on all outcomes in its support, and the posterior — while concentrating toward the data — never entirely rules out possibilities that haven't been observed. The artificial data mechanism restores this property to the bootstrap by explicitly injecting prior observations into the dataset before resampling.

This insight has implications beyond the specific algorithms proposed. It **diagnoses a structural failure mode** that applies to any exploration method based on resampling observed data without a prior-inducing mechanism. Ensemble methods that train models on disjoint subsets of real data, subsampling approaches like BESA, and any technique that estimates uncertainty purely from the empirical distribution of past observations will suffer from the same collapse of uncertainty when faced with initially unfavorable outcomes. The paper thus establishes a **requirement** for bootstrap-based exploration — the inclusion of prior-inducing synthetic data — that is non-obvious from first principles and that prior work (Eckles and Kaptein, 2014; Baransi et al., 2014) had missed.

The evidence for this insight is not merely the simulation in Figure 1, though that provides a clean demonstration. It is also the **formal equivalence established in Section 3.2**: for the Bernoulli bandit with Beta priors, the Bayesian bootstrap with appropriately configured artificial data reproduces Thompson sampling *exactly*. This equivalence proves that the artificial data is not a hack or a heuristic — it is the precise mechanism that bridges nonparametric resampling and parametric Bayesian inference in the sequential decision setting. The artificial data plays the role of the prior pseudocounts in the conjugate update, and its omission would be as fatal as omitting the prior in a Bayesian analysis.

This is a fundamental advance in understanding, not an incremental improvement. It transforms the question from "can we use the bootstrap for exploration?" (to which prior work gave a weakly positive but empirically unreliable answer) to "how do we configure the artificial data to encode the right prior for exploration?" The latter question has a principled answer (match the artificial data to a Bayesian prior for which Thompson sampling has guarantees), and the paper provides a constructive demonstration for the Beta-Bernoulli and Dirichlet-multinomial cases.

### Innovation 2: Bootstrapping Entire Value Functions Provides the Only Known Tractable Mechanism for Deep Exploration with Nonlinear Function Approximators

The extension from bandits to reinforcement learning (Algorithm 4) represents a conceptual leap that the paper presents almost casually but that carries significant intellectual weight. The key move is treating **entire Q-functions as the objects of posterior sampling**, with the bootstrap operating at the level of episodes rather than individual transitions. This is not a straightforward generalization of the bandit algorithm — it embeds a specific hypothesis about what it means to "explore" in a sequential setting with function approximation.

To appreciate what's novel here, consider the default approach to exploration in deep RL at the time (and to a large extent still today): **action-level randomization.** In ε-greedy, you flip a coin at each timestep and take a random action with probability ε. In Boltzmann exploration, you sample actions according to a softmax over Q-values. In both cases, the exploration decision is local and myopic — you randomize at *this* timestep, observe the result, update your model, and move on. There is no coherent plan to test a hypothesis about the environment. If the optimal strategy requires going through a low-reward region to reach a high-reward region, action-level randomization will rarely discover it because the random walk through the low-reward region is exponentially unlikely.

Posterior sampling for RL (Osband et al., 2013) solves this by sampling an entire MDP from the posterior and then acting optimally with respect to that sampled MDP. If the sampled MDP places high reward behind a door, the agent will go through the door — not because it's randomizing, but because it's following an optimal policy for a coherent world model that happens to be optimistic about what's behind the door. This is deep exploration: the agent takes a sequence of coherent, goal-directed actions that collectively test a hypothesis about the environment's structure.

The problem, of course, is that sampling an entire MDP from a posterior is intractable for all but the smallest tabular problems. You would need a posterior over transition dynamics and reward functions that covers the state space, and for deep RL with raw pixel inputs, this is computationally inconceivable.

The paper's innovation is to observe that **you don't need to sample an explicit MDP to get deep exploration — you need to sample a value function that is consistent with some plausible MDP.** The bootstrap provides a way to do this that scales with the function approximator you're already using. By training a Q-network on a bootstrap resample of the combined real and artificial episode data, you obtain a randomized value function that, when followed greedily, produces behavior that *looks like* deep exploration. The artificial data — with its stochastically optimistic rewards — ensures that some bootstrap Q-functions will assign high values to unexplored regions, and when those Q-functions are selected, the agent will navigate toward those regions in a coherent, goal-directed way.

What makes this a fundamental contribution is that it **decouples deep exploration from explicit model-based planning.** The agent does not need to maintain a posterior over environment dynamics, does not need to simulate "what would I learn if I went left?", and does not need to solve a partially observable Markov decision process for information gain. It simply needs to maintain multiple value function approximators trained on differently weighted data, with optimistic prior data injected, and alternate between them. The computational requirements are training K neural networks in parallel — substantial, but feasible with modern hardware. The conceptual requirements are dramatically simpler than any alternative approach to deep exploration.

The paper's claim that this "may be the only known computationally eﬃcient means of achieving deep exploration with nonlinearly parameterized representations such as deep neural networks" is bold but defensible. At the time of writing, alternative approaches to deep exploration included:

- **Bayesian neural networks with MCMC:** Intractable for the network sizes used in deep RL. Even approximate methods like stochastic gradient Langevin dynamics were computationally demanding and sensitive to hyperparameters.

- **Explicit uncertainty propagation through planning (e.g., R-MAX with function approximation):** Requires maintaining confidence sets around value estimates, which is analytically intractable for deep networks. The optimism bonuses used in tabular R-MAX have no natural extension to shared representations.

- **Information-directed sampling:** Requires estimating the information gain of candidate actions, which is itself a computationally intensive estimation problem, especially with deep function approximators.

- **Linear value function randomization (Van Roy and Wen, 2014):** Provides deep exploration but with fixed linear features — precisely the limitation the present work overcomes by substituting the bootstrap for analytic posterior sampling.

The bootstrap approach sidesteps all of these difficulties by reducing posterior sampling to a supervised learning problem on reweighted data. The significance of this move extends beyond the specific algorithm: it suggests that **the exploration problem in deep RL might be solvable through the same supervised learning machinery that solved the generalization problem.** You don't need specialized uncertainty representations or planning algorithms — you need diversity in your training data weights and some optimistic prior data.

The evidence for this claim in the paper is necessarily limited — there are no large-scale deep RL experiments in this technical note. The contribution is conceptual and methodological rather than empirical. However, the formal connection to Thompson sampling (which has known deep exploration properties) and the bandit simulation demonstrating the necessity of artificial data provide a strong theoretical foundation. The paper essentially says: "We have a mechanism that (a) provably reproduces Thompson sampling in the cases where we can check, (b) scales to any function approximator that can be trained on weighted data, and (c) therefore provides the first practical path to deep exploration with deep networks." This is a claim about architecture and scalability rather than about benchmark performance, and it has proven prescient — subsequent work on bootstrapped DQN (which the authors of this paper would go on to develop) validated the practical viability of these ideas.

### Innovation 3: The Bootstrap Is Reframed as a General-Purpose Posterior Approximation Engine, Not Just a Confidence Interval Tool

While the use of the bootstrap for posterior approximation predates this paper (the Bayesian bootstrap of Rubin, 1981, explicitly connects to Dirichlet process priors), the paper makes a distinctive reframing move: **it treats the bootstrap not as a statistical procedure applied to a specific estimator, but as a general-purpose computational engine that can be plugged into any sequential decision algorithm that requires posterior samples.** This reframing shifts the bootstrap from a tool for *analyzing* uncertainty (constructing confidence intervals, estimating standard errors) to a tool for *acting under* uncertainty (generating the randomized policies that drive exploration).

The standard use of the bootstrap in statistics is retrospective: you've collected a dataset, you've computed an estimate, and now you want to know how precise that estimate is. You bootstrap to get a distribution over possible estimates, from which you construct confidence intervals or standard errors. This is an offline, analytical use case.

The paper's use of the bootstrap is prospective and operational: at each timestep, you generate a single bootstrap sample (a single resampled dataset with a single fitted model), use it to select an action, observe the outcome, and then update the dataset on which future bootstrap samples will be based. The bootstrap distribution is never explicitly computed or summarized — it is sampled from on-the-fly, and each sample influences the data that will be used for future samples. This is a closed-loop use of the bootstrap that is conceptually distinct from its statistical origins.

This reframing matters because it **positions the bootstrap as a computational primitive** — a building block for exploration algorithms — rather than an analytical technique. The paper's Algorithms 3 and 4 can be read as templates: wherever a Bayesian decision-making algorithm says "sample from the posterior," you can substitute "apply the bootstrap with artificial data to a function that fits a model on weighted data." The specific bootstrap variant (standard vs. Bayesian), the specific function φ (expectation, Q-function fitting, policy network), and the specific artificial data distribution can all be swapped out depending on the problem. The architecture — bootstrap resampling of augmented data followed by greedy action with respect to the fitted model — remains constant.

This is a design pattern, not just an algorithm, and its generality is what makes the paper influential beyond the specific bandit and RL settings it analyzes. Subsequent work has applied this pattern to contextual bandits, offline RL, recommender systems, and active learning — any domain where you need to maintain and sample from an approximate posterior over a complex model class, you can reach for the augmented bootstrap.

The evidence for the power of this reframing is partly in the formal equivalences (Section 3.2) and partly in the scalability arguments. The paper demonstrates that the bootstrap can exactly reproduce conjugate Bayesian inference for simple models (Bernoulli bandits with Beta priors), which provides a proof of concept: the bootstrap is not an ad-hoc approximation but a principled mechanism that recovers known-correct behavior in the special cases where the correct behavior can be computed analytically. For more complex models where analytic posteriors are unavailable, the paper argues by extrapolation: the bootstrap approximates the posterior (as established in the nonparametric Bayes literature), and the artificial data provides the prior, so the combined procedure approximates posterior sampling with that prior. The quality of this approximation depends on the bootstrap's fidelity for the specific model class, but the architecture is sound.

This is an incremental contribution relative to the bootstrap literature (the Bayesian bootstrap existed since 1981), but fundamental relative to the exploration literature — it provides a unified framework for implementing Thompson sampling that works across model classes, scales with computational resources, and requires no analytic derivations. The "general-purpose posterior approximation engine" reframing makes the bootstrap a first-class tool for sequential decision-making, not just a diagnostic for static analyses.

## 5. Experimental Analysis

### Evaluation Methodology

- **Dataset.** The paper does not use an external benchmark dataset in the conventional sense. The multi-armed bandit experiments in Section 3.1 are conducted on a **synthetic two-arm Bernoulli-style bandit problem** designed by the authors specifically to demonstrate the failure mode of prior-free bootstrap methods. The problem is parameterized by a single value `$\epsilon$` (set to 0.01 in the simulations) with deterministic and stochastic arms as described in Section 3.1. The reinforcement learning extension (Section 4) is presented as an algorithmic framework without accompanying simulation results — there is no dataset, no environment (e.g., no Atari game, no MuJoCo task), and no empirical RL evaluation in this paper. The only quantitative results in the entire paper are the bandit simulations in Section 3.1.

- **Base model(s).** There is no base model in the sense of a pretrained neural network. The bandit experiments use **three bootstrap variants** as the mechanism for generating randomized reward estimates: the standard bootstrap (Algorithm 1, Efron 1979), the Bayesian bootstrap (Algorithm 2, Rubin 1981), and BESA (a subsampling approach from Baransi et al., 2014). These are not "models" in the deep learning sense — they are statistical resampling procedures applied to small histories of (action, reward) pairs. The paper does not test any deep neural network, any function approximator, or any reinforcement learning environment. The phrase "particularly well-suited for contexts in which exploration is coupled with deep learning" in the abstract is a forward-looking claim about the method's applicability, not a description of experiments conducted.

- **Metrics.** The sole reported metric is **cumulative regret** over time, plotted in Figure 1 (six subplots). Regret at time `$T$` is defined as the sum from `$t=1$` to `$T$` of the expected difference between the reward of the optimal action (arm 2, with expected reward `$2\epsilon$`) and the reward of the chosen action, where the expectation is conditioned on the true underlying distribution `$p^*$`. Lower cumulative regret indicates better performance — a flat or slowly growing curve means the agent has learned to select the optimal action and incurs near-zero additional regret per step; a linearly growing curve means the agent is persistently selecting a suboptimal action. The Bayesian regret (expectation over the prior on `$p^*$`) is not computed explicitly; the reported results are Monte Carlo estimates of cumulative regret averaged over 20 independent simulation runs for each algorithm variant.

- **Baselines.** The evaluation compares **six algorithm configurations in a 3×2 factorial design**: three bootstrap methods (standard Bootstrap from Algorithm 1, Bayesian BayesBootstrap from Algorithm 2, and BESA from Baransi et al., 2014) each tested with and without artificial data (`$M=0$` top row of Figure 1, `$M=2$` bottom row). There is no comparison to:
  - **True Thompson sampling** with a Beta prior (which would be the gold-standard baseline for demonstrating that the bootstrap approximates posterior sampling behavior)
  - **Any UCB algorithm** (which the paper discusses extensively in the introduction as a competing approach to exploration)
  - **ϵ-greedy or Boltzmann exploration** (standard simple baselines for bandit problems)
  - **Any other exploration heuristic**
  
  This is a conspicuous absence. The paper's central claim is that bootstrap-based Thompson sampling with artificial data enables effective exploration where prior-free methods fail, but the natural baselines for this claim would be: (a) true Thompson sampling with a correctly specified prior (to show the bootstrap approximates it), and (b) simple exploration heuristics like ϵ-greedy (to show the approach is competitive with or superior to standard alternatives). Neither is tested.

- **Generation budget / compute accounting.** The paper does not use "generations" or any deep learning compute metric. For the bandit experiments, the relevant resource is the number of timesteps (the horizon over which regret is accumulated), shown on the x-axis of Figure 1 up to approximately 10,000 steps. The paper does not account for the computational cost of running the bootstrap — how many resamples per step, how expensive the model-fitting function `$\phi$` is, or how training time scales with history size. The artificial data generation cost (`$M=2$` data points per step, sampled from a uniform distribution) is trivial. No FLOPs counting, no wall-clock time measurement, and no scalability analysis is presented.

- **Cross-validation / statistical protocol.** The only statistical protocol mentioned is that the results in Figure 1 are averaged over **20 Monte Carlo simulations** for each algorithm variant. Each simulation is an independent run of the agent on the same bandit problem with the same `$\epsilon = 0.01$`, but with different random seeds for the agent's internal randomness (bootstrap resampling, artificial data generation) and the environment's reward realizations. The paper does not report confidence intervals, standard errors, or any measure of variability across the 20 runs — the plotted curves appear to be point estimates of mean cumulative regret. No cross-validation is used (there is no train/test split on a synthetic bandit with known parameters). The `$\epsilon$` parameter is fixed at 0.01; no sensitivity analysis across different `$\epsilon$` values (which would vary the difficulty of the exploration problem) is presented.

---

### Main Quantitative Results

The entire empirical contribution of the paper consists of Figure 1 and the accompanying discussion in Section 3.1. I'll describe what Figure 1 shows in detail, since the paper itself only provides a high-level interpretation without specific numerical values at particular timesteps.

#### Bandit Simulation: Artificial Data Enables Exploration, Prior-Free Bootstraps Fail

**Headline result:** Without artificial data (`$M=0$`), all three bootstrap variants incur **linearly growing cumulative regret** on the two-arm bandit with `$\epsilon = 0.01$`, indicating they commit to the suboptimal arm (arm 1, expected reward `$\epsilon$`) and never recover. With only `$M=2$` artificial data points per step (one per arm with uniform random reward), all three bootstrap variants learn to prefer the optimal arm (arm 2, expected reward `$2\epsilon$`) and exhibit **sublinear cumulative regret** — the curves flatten over time, indicating per-step regret approaching zero.

**Figure 1 subplot structure:** The figure is organized as a 2×3 grid. The rows distinguish artificial data configuration (`$M=0$` top, `$M=2$` bottom). The columns distinguish the bootstrap method: left column is standard Bootstrap (Algorithm 1), middle column is Bayesian BayesBootstrap (Algorithm 2), right column is BESA (Baransi et al., 2014). Each subplot shows cumulative regret on the y-axis versus timesteps on the x-axis, with a single curve representing the mean over 20 Monte Carlo runs. The y-axis scales and x-axis ranges appear consistent across subplots for comparability.

**Top row (`$M=0$`, no artificial data):**

The paper states: "with probability at least `$1 - 2\epsilon$`, BootstrapThompson without artiﬁcial history (`$M=0$`) will never learn the optimal policy." For `$\epsilon = 0.01$`, this means with probability approximately 0.98, the agent's first pull of arm 2 yields reward 0 (since `$p^*_2(0) = 0.98$`), after which the bootstrap distribution for arm 2 puts all mass on 0, and the agent selects arm 1 (which always yields `$\epsilon$`) at every subsequent timestep. The cumulative regret therefore grows as `$(2\epsilon - \epsilon) \cdot (T - 2) \approx \epsilon T$` for large `$T$`.

The curves in the top row are consistent with this analysis: all three subplots show **roughly linearly increasing cumulative regret** across the 10,000-step horizon. The standard Bootstrap and Bayesian BayesBootstrap curves (top-left and top-center) are visually similar in slope and appearance. The BESA curve (top-right) appears to have a slightly lower slope but is still clearly growing — the paper comments that BESA "performs well in some settings [3], but the approach does not generalize gracefully to settings with dependent arms" and notes it "seems to outperform" the others on this example only weakly and still fails without artificial data.

**Bottom row (`$M=2$`, with artificial data):**

The artificial data is generated as described: "each of the two actions once and samples an observation uniformly from [0, 1] for each." This means the augmented history at each step contains the real observations plus two artificial observations: one for arm 1 with a random reward in [0,1], and one for arm 2 with a random reward in [0,1]. These artificial observations provide the prior-induced optimism: when the uniform draw for arm 2 happens to be high (e.g., 0.8), the bootstrap resample that includes that observation will estimate arm 2's value as high, and the agent will select it. If arm 2 yields reward 1 on that pull, the real data now favors arm 2, and learning accelerates.

The curves in the bottom row all show **sublinear cumulative regret** — the regret grows initially (while the agent is still uncertain) and then flattens out, indicating the agent has converged to selecting arm 2 most of the time and incurs near-zero per-step regret. The standard Bootstrap and Bayesian BayesBootstrap curves (bottom-left and bottom-center) appear visually very similar, consistent with the paper's statement that "the choice of bootstrap method makes little difference." Both appear to reach a stable low-regret regime well before the 10,000-step horizon. The BESA curve (bottom-right) also shows substantially lower regret than its `$M=0$` counterpart but exhibits **higher variance** and potentially a slower convergence — the paper notes "BESA seems to outperform" the other methods only weakly and emphasizes that the artificial data, not the bootstrap variant, is the critical factor.

**Specific numerical values:** The paper does not provide exact cumulative regret values at any specific timestep for any method. The analysis is entirely qualitative, based on visual inspection of the curve shapes (linear vs. sublinear). This is a significant limitation for a paper that makes strong claims about exploration efficiency — there is no "method A achieves cumulative regret of X at timestep 10,000 while method B achieves Y," no learning rate quantification, and no formal statistical comparison between methods.

**What the figure does not show:**

- **True Thompson sampling performance.** There is no curve for Thompson sampling with a Beta(1,1) prior, which would provide a direct comparison to the behavior the bootstrap is supposed to approximate. We cannot assess how closely BootstrapThompson with artificial data matches the exploration efficiency of true posterior sampling — we can only see that it learns where the prior-free version does not.

- **Sensitivity to M.** Only `$M=0$` and `$M=2$` are tested. How does performance vary with `$M$`? Does `$M=1$` suffice? Does `$M=10$` cause the artificial data to overwhelm the real data and delay convergence? The framework's key claim — that "the relative strength `$M/N$` of the induced prior can be controlled" — is not empirically validated.

- **Sensitivity to the artificial data distribution `$\tilde{P}$`.** The artificial data for `$M=2$` uses a specific generator: each action once, observation Uniform(0,1). What if the observations were uniform but with a different range? What if the distribution were Beta(1,1) instead of uniform? What if observations were generated from an alternative generative model? The paper's claim about Dirichlet process priors suggests that the choice of `$\tilde{P}$` matters, but no sensitivity analysis is presented.

- **Sensitivity to `$\epsilon$`.** The exploration difficulty is controlled by `$\epsilon$`: smaller `$\epsilon$` means arm 2 is rarer (2ϵ probability of reward 1) and the gap between arms is smaller (2ϵ − ϵ = ϵ). As `$\epsilon \to 0$`, the problem becomes arbitrarily hard — the optimal arm almost never yields reward, and the suboptimal arm yields negligible but positive reward. As `$\epsilon \to 0.5$`, the problem becomes trivial — arm 2 yields reward 1 frequently. Testing across a range of `$\epsilon$` values would establish the difficulty range over which artificial data provides benefit, but only `$\epsilon = 0.01$` is tested.

- **Multi-arm and dependent-arm generalizations.** The two-arm bandit is the simplest possible setting. The paper discusses Dirichlet-multinomial generalizations theoretically but provides no simulation results for K > 2 arms, for dependent arms (where observations from one arm inform beliefs about others), or for contextual bandits. The claim that the approach generalizes is purely analytical.

---

### Ablation Studies and Robustness Checks

The paper **does not contain ablation studies** in the modern machine learning sense. There is no systematic investigation of how varying individual components of the algorithm affects performance. What the paper does provide is a **three-way comparison of bootstrap methods** (standard, Bayesian, BESA) at two levels of artificial data (`$M=0$` and `$M=2$`), which can be interpreted as a limited ablation:

- **Bootstrap method choice (standard vs. Bayesian vs. BESA):** The paper reports that "the choice of bootstrap method makes little difference" based on the visual similarity of the three curves within each row of Figure 1. This is a robustness finding — the standard bootstrap (which requires no weight sampling and is simpler to implement) performs comparably to the Bayesian bootstrap (which has formal posterior interpretation) on this problem. BESA is reported to underperform the others but not dramatically. The paper does not provide quantitative support for this claim (no final regret values, no statistical test of difference) — it is a qualitative assessment from visual inspection.

- **Artificial data presence (M=0 vs. M=2):** This is the key comparison, and the paper's central empirical claim. The finding that `$M=2$` dramatically improves over `$M=0$` is unambiguous from Figure 1 (linear vs. sublinear regret). However, the choice of `$M=2$` is not ablated — we don't know if `$M=1$`, `$M=4`, or `$M=10$` would perform differently.

- **Artificial data distribution:** Not ablated. The paper uses a specific `$\tilde{P}$` (one observation per action, Uniform(0,1) reward). Alternative priors (e.g., Beta(1,1) per arm, which would correspond exactly to a uniform prior on success probability for the Bernoulli interpretation) are not tested, even though the paper argues for formal equivalence to Beta priors in Section 3.2.

- **Number of Monte Carlo simulations (20):** Not ablated. The paper reports results over 20 runs but does not examine whether this is sufficient for stable estimates of mean cumulative regret. No variance estimates are reported.

- **Problem difficulty (`$\epsilon$`):** Not ablated. The single value `$\epsilon = 0.01$` is used throughout. The failure mode described (arm 2 yields 0 on first pull with probability 1 − 2ϵ) becomes less likely as `$\epsilon$` increases, so the benefit of artificial data should be most pronounced at small `$\epsilon$` — but this relationship is not quantified.

**What would constitute meaningful ablations in this framework:**

- **Artificial data strength `$M$`:** Sweeping `$M$` from 0 to, say, 20, and measuring the resulting Bayesian regret (or cumulative regret at a fixed horizon) would characterize the exploration-exploitation tradeoff as a function of prior strength. The paper's claim that `$M$` provides "explicit control" over the induced prior is not empirically validated.

- **Artificial data generation strategy:** Comparing "optimistic" priors (artificial observations skewed toward high rewards) against "neutral" priors (uniform) and "pessimistic" priors would test whether the prior's optimism is necessary for exploration or whether any nonzero prior mass on unobserved outcomes suffices. The bandit simulation uses uniform rewards which are stochastically higher than the suboptimal arm's reward (`$\epsilon = 0.01$`) and thus implicitly optimistic relative to arm 1. Would a prior that generates mostly low rewards also drive exploration? The paper does not address this.

- **Comparison to exact Thompson sampling:** Running Thompson sampling with Beta(1,1) priors on the same problem would provide a gold-standard reference for what optimal exploration looks like. The gap between BootstrapThompson and true Thompson sampling would reveal how well the bootstrap approximates the posterior for this problem.

- **Scaling with number of arms:** Testing on bandits with K=5, 10, 50 arms would reveal whether the artificial data mechanism scales gracefully or whether the prior becomes diluted across many arms.

---

### Critical Assessment

#### What the experiments demonstrate — and what they do not

**Claim from Executive Summary: "Without artificial data, bootstrap-based Thompson sampling fails catastrophically (failing to identify the optimal arm with probability 1 − 2ϵ)."**

The experiments **support this claim for the specific case tested**: a two-arm bandit with `$\epsilon = 0.01$` and `$M=0$` artificial data. All three bootstrap variants show linearly growing regret, consistent with permanent commitment to the suboptimal arm after an unlucky initial observation. The probability analysis (failure with probability 1 − 2ϵ) is analytically correct given the bootstrap's restriction to observed support — it is not an empirical estimate but a logical consequence of the algorithm's design, and the simulation curves are visually consistent with this analysis.

However, the experiments demonstrate this failure for **exactly one problem instance** (one value of `$\epsilon$`, two arms, one reward structure). The failure mode is analytically predicted for any problem where (a) the optimal arm can produce unfavorable initial outcomes, and (b) those outcomes are the only observations for that arm. The experiment confirms the analysis but does not test boundary conditions: does the failure still occur if both arms are stochastic? If arm 1 is also stochastic and early observations happen to favor it? If there are more than two arms? These would all be instances of the same structural issue, but they are not tested.

**Claim from Executive Summary: "Adding as few as two artificially generated observations drives efficient exploration."**

The experiments **support this claim for M=2 with the specific artificial data generator tested.** The bottom row of Figure 1 shows sublinear regret for all three bootstrap variants, indicating the agent learns to prefer the optimal arm. However, the claim "as few as two" is not empirically validated against other values of M — the paper does not show that M=1 fails, or that M=1 suffices, or that M=2 is the minimal effective amount. The number 2 is a design choice in the experiment, not an empirically established minimum. The claim that M=2 is "as few as" anything is an interpretation unsupported by parameter sweeps.

Additionally, "efficient exploration" is not defined quantitatively. Without a comparison to the Bayes-optimal policy or to Thompson sampling with a correct prior, we cannot assess whether the exploration is *efficient* in any formal sense, only that it avoids the catastrophic failure of the M=0 case. The curves look reasonable (sublinear), but there are no regret bounds, no comparison to optimal rates, and no quantification of how much regret is incurred relative to what is achievable.

**Claim from Executive Summary: "This establishes that prior-inducing synthetic data is essential for bootstrap approximations to match the performance guarantees of true posterior sampling."**

The experiments **do not establish this claim** as stated. What the experiments establish is a narrower result: that on one specific bandit problem, artificial data prevents a catastrophic failure mode that occurs without it. This is not the same as establishing that bootstrap-with-artificial-data matches the performance guarantees of true posterior sampling. To establish that, the paper would need to:

1. Demonstrate that the bootstrap-with-artificial-data achieves regret bounds comparable to those proven for Thompson sampling (e.g., O(√T) Bayesian regret). The simulations provide no regret quantification, no comparison to theoretical rates, and no asymptotic analysis.

2. Compare directly to true posterior sampling on the same problem and show that the regret curves are similar (or that the gap shrinks with more data). This comparison is entirely absent.

3. Show that the formal equivalence established for Beta-Bernoulli models in Section 3.2 translates into matched empirical performance — i.e., run Thompson sampling with Beta(1,1) and BootstrapThompson with M appropriately configured, and demonstrate they behave identically. This is not done.

The paper establishes a formal equivalence for a specific case (Section 3.2) and demonstrates a qualitative improvement from adding artificial data (Figure 1). The leap from these to "matches the performance guarantees of true posterior sampling" is a theoretical argument, not an empirical finding. The experiments support the *necessity* of artificial data for avoiding catastrophic failure; they do not empirically address the *sufficiency* of artificial data for recovering Thompson sampling's theoretical properties.

**Claim from paper: "The choice of bootstrap method makes little difference."**

This claim is **supported qualitatively** by Figure 1: the standard bootstrap and Bayesian bootstrap curves are visually similar in both the M=0 and M=2 conditions. The BESA method appears to differ somewhat (higher variance in the M=2 case, possibly different slope in the M=0 case), but the broad pattern (linear regret without artificial data, sublinear with) holds across all three. However, without quantitative comparisons (final regret values, statistical tests, confidence intervals), the claim remains an informal visual assessment. For a paper whose title is "Bootstrapped Thompson Sampling," the precise choice of bootstrap method would seem to be a central question — yet it is dismissed in a single sentence with minimal empirical support.

**Claim from paper (Section 4): "The general approach represented by this algorithm may be the only known computationally eﬃcient means of achieving deep exploration with nonlinearly parameterized representations such as deep neural networks."**

This claim receives **zero empirical support** in this paper. There are no experiments with deep neural networks, no reinforcement learning environments, no demonstration of deep exploration, and no comparison to alternative methods for deep exploration with nonlinear function approximators. The claim is entirely conceptual — an argument about what the algorithm architecture *enables* — and should be understood as a forward-looking statement or a research agenda, not as an experimental finding. This is a significant gap for a paper that motivates its contribution largely through the limitations of deep RL exploration.

#### Genuine weaknesses in the experimental design

**Single problem instance.** All empirical results are on one two-arm bandit with one parameter setting (`$\epsilon = 0.01$`). This is an extraordinarily narrow empirical basis for claims about exploration in sequential decision-making. The failure mode that the paper identifies is analytically clear and the simulation serves as a sanity check, but three variants × two conditions × one problem = six curves is minimal empirical evidence. There is no demonstration that the method works across different bandit structures (different numbers of arms, different reward distributions, contextual bandits) or that the artificial data approach transfers to problems where the optimal prior is less obvious.

**No comparison to standard baselines.** The absence of true Thompson sampling, UCB, ϵ-greedy, or any other exploration algorithm as a baseline is a serious omission. The paper's narrative is that BootstrapThompson approximates Thompson sampling, but we never see Thompson sampling's performance on the same problem. The paper discusses UCB's limitations in the introduction but never compares to it empirically. The result is that we can see BootstrapThompson works with artificial data but not without — but we cannot assess whether it works *as well as* alternatives, which is the relevant practical question.

**No quantitative metrics beyond regret curves.** The paper relies entirely on visual inspection of cumulative regret plots. There are no tables of final regret, no learning rate metrics (e.g., time to reach 95% optimal action selection), no statistical comparisons, and no confidence intervals. In modern empirical machine learning, this level of quantitative reporting would be considered insufficient for a paper making strong empirical claims.

**No hyperparameter sensitivity analysis.** The only hyperparameter varied is M (0 vs. 2). The artificial data distribution `$\tilde{P}$`, the value of `$\epsilon$`, the number of bootstrap replicates, and the initial conditions are all held fixed. The paper's claim that M/N controls prior strength is not empirically characterized.

**No RL experiments whatsoever.** Section 4 presents Algorithm 4 (and the incremental variants in Algorithms 5 and 6) as a major contribution — a method for deep exploration in reinforcement learning with nonlinear function approximators. Yet there is not a single RL simulation in the paper. The claims about computational efficiency, deep exploration, and scalability are entirely theoretical. A skeptical reader would note that the paper's title promises "Deep Exploration" but the only experiments are on a two-arm bandit with no function approximation and no sequential state.

**Artificial data cost is not analyzed.** The bandit experiments use `$M=2$` artificial data points, which is trivially cheap. But the RL extension proposes generating `$M$` artificial episodes with "diffusely mixed" state-action sampling and "stochastically optimistic rewards." The cost of generating realistic artificial episodes for complex environments (e.g., how do you sample "diffusely mixed" states in an Atari game from pixels?) is not addressed, and the sensitivity of the method to the quality of the artificial data generator is not studied.

#### Experiments that would have strengthened the paper

- **Direct comparison to Thompson sampling with Beta priors** on the same two-arm bandit, showing that BootstrapThompson with appropriately configured artificial data produces regret curves indistinguishable from true posterior sampling. This would directly validate the equivalence argument of Section 3.2.

- **A sweep over `$\epsilon$` values** (e.g., 0.001, 0.005, 0.01, 0.05, 0.1) to characterize how the exploration difficulty affects the relative benefit of artificial data. At larger `$\epsilon$`, the failure mode (observing zero from arm 2 on the first pull) becomes less likely, so the benefit of artificial data should diminish — does it? This would test the boundary of the paper's claims.

- **A sweep over `$M$`** (e.g., 0, 1, 2, 5, 10, 20) to characterize the exploration-exploitation tradeoff as a function of prior strength. The optimal `$M$` should balance sufficient exploration to find the optimal arm against excessive prior influence that delays convergence. This would operationalize the paper's claim that `$M$` provides explicit control.

- **A multi-arm extension** (e.g., 10 arms, one optimal with low probability of high reward, others with varying means) to test whether the artificial data mechanism scales to larger action spaces where the bootstrap must maintain uncertainty across many arms simultaneously.

- **At minimum, one RL experiment** — perhaps a simple gridworld or chain MDP where deep exploration is required (e.g., a long corridor with a high-reward state at the end that requires sustained exploration to discover) — demonstrating that Algorithm 4 with neural network function approximation and artificial data actually achieves deep exploration where prior-free bootstraps or ϵ-greedy fail. The complete absence of RL experiments makes the paper's title ("Deep Exploration") and much of its motivation (deep RL, deep neural networks) aspirational rather than demonstrated.

In summary, the experimental section of this paper is **minimal to the point of being a proof-of-concept illustration rather than a systematic empirical evaluation.** The single simulation convincingly demonstrates the necessity of artificial data for avoiding a specific failure mode on a specific problem, and this is a genuine contribution — it isolates and visualizes a structural issue that prior work missed. But the paper's broader claims about Thompson sampling approximation, deep exploration, scalability to deep learning, and advantages over alternative methods are not empirically tested. The experiments should be understood as a diagnostic demonstration that supports the paper's analytical arguments, not as a comprehensive evaluation of the proposed methods.

## 6. Limitations and Trade-offs

### 6.1 The Artificial Data Design Problem Is Delegated to the Practitioner Without Guidance

**The assumption or constraint:** The entire Bootstrapped Thompson Sampling framework depends on the practitioner specifying a prior distribution `$\tilde{P}$` from which artificial data is sampled, and a prior strength parameter `$M$` that controls how many artificial observations are mixed with real data. The paper acknowledges that these choices matter — "the way in which we generate and use artiﬁcial data is critical" (Section 1) — but provides minimal guidance on how to make them in practice. For the bandit simulation, the artificial data generator is hand-designed (each action once, observation Uniform(0,1)), and the paper simply declares that `$M=2$` works without exploring alternatives. For the RL setting, the suggestion is even more abstract: "sample state-action pairs from a diffusely mixed generative model and assign them stochastically optimistic rewards and random state transitions" (Section 4). What constitutes "diffusely mixed" for a specific environment? How optimistic should the rewards be? How many artificial episodes `$M$` are needed relative to the complexity of the state space? The paper provides no answers.

**The consequence:** The method offloads to the practitioner a design problem that is, in many ways, equivalent to the original difficulty of specifying a prior for Thompson sampling. In conjugate Bayesian models, specifying a Beta(`$\alpha, \beta$`) prior is straightforward — the parameters have clear interpretations as pseudocounts. But in the bootstrap framework, the prior is encoded implicitly through synthetic data generation, and the mapping from prior beliefs about the environment to an artificial data generator is non-obvious. A practitioner who generates artificial data that is insufficiently optimistic will see exploration fail (as in the `$M=0$` case). One who generates data that is too optimistic will see the agent persist in exploring unpromising regions long after sufficient evidence has accumulated. The paper offers no systematic way to tune these choices, no diagnostic for whether the artificial data is well-calibrated, and no theoretical guidance beyond the special case of conjugate models where the equivalence to Bayesian posteriors can be established (Section 3.2). For the deep RL setting that motivates the paper, the conjugate case provides no practical guidance — there is no analog of Beta pseudocounts for a Q-network trained on Atari frames.

**What evidence exists:** The paper provides none. The sensitivity of the method to artificial data design is not studied empirically. Only one artificial data generator (uniform rewards, one observation per action) and one value of `$M$` (2) are tested in the bandit experiment. The RL extension (Section 4) includes no experiments at all, so the practical feasibility of designing artificial episode generators for realistic environments is entirely unexamined. The paper's silence on this point is conspicuous given that the artificial data is the central mechanism that enables exploration — the paper devotes extensive theoretical discussion to why it is necessary (Section 3.1) and how it connects to Dirichlet process priors (Section 2), but no treatment to how it should be designed in practice.

**Mitigation status:** Not addressed. The paper does not propose a method for automatically designing or tuning artificial data generators, does not provide sensitivity analyses, and does not discuss the practical engineering considerations involved. The formal equivalence established for conjugate models (Section 3.2) partially mitigates this limitation for those special cases, since it tells the practitioner exactly what artificial data to generate to match a desired Beta or Dirichlet prior. But for the non-conjugate, nonlinearly parameterized settings that are the paper's stated motivation (deep learning, Section 1), no such guidance exists. The paper's suggestion that "appropriately designed schemes for generating artiﬁcial data" can establish equivalences for broader problem classes is a conjecture, not a result, and the paper does not pursue it further.

---

### 6.2 No Evidence That the Method Works with Deep Neural Networks or in Reinforcement Learning Environments

**The assumption or constraint:** The paper's title promises "Deep Exploration," and its introduction and abstract repeatedly position the method as a solution for contexts "coupled with deep learning" where "maintaining or generating samples from a posterior distribution becomes computationally infeasible" (Abstract). Section 4 presents detailed algorithms for episodic RL with value function approximation (Algorithms 4, 5, 6), makes the strong claim that this approach "may be the only known computationally eﬃcient means of achieving deep exploration with nonlinearly parameterized representations such as deep neural networks," and discusses practical considerations like weight sharing, dropout masks, and parallel training. However, **not a single experiment in the paper uses a neural network, a deep learning framework, or a reinforcement learning environment.** The sole empirical result (Figure 1, Section 3.1) is on a two-arm Bernoulli-style bandit with no function approximation — the action values are estimated directly from the resampled data without any learned representation.

**The consequence:** Every claim about deep learning and RL is strictly aspirational. The paper provides no evidence that:

- The bootstrap produces meaningful posterior approximations when applied to deep neural networks (where the mapping from data weights to function outputs is highly complex and non-convex).
- The artificial data mechanism scales to high-dimensional state spaces (where generating "diffusely mixed" artificial states is itself a hard generative modeling problem).
- The parallel training scheme of Algorithm 6 is computationally feasible for the network sizes used in deep RL (DQN, A3C, etc.) without prohibitive overhead.
- Deep exploration actually occurs with nonlinear function approximators — that the agent exhibits coherent exploratory trajectories that test hypotheses about the environment, rather than simply oscillating due to function approximation noise.
- The method outperforms or even matches simpler exploration heuristics (ϵ-greedy, entropy regularization, parameter noise) on standard deep RL benchmarks.

The gap between the paper's motivation and its empirical validation is extreme. The paper argues that prior bootstrap approaches for Thompson sampling (Eckles and Kaptein, 2014; Baransi et al., 2014) are insufficient — but those papers at least included experiments. This paper's proposed remedy (adding artificial data) is demonstrated only on a toy problem that could be solved by dozens of alternative methods, and the extension to the setting that motivates the entire work (deep RL) is entirely speculative.

**What evidence exists:** None. The paper contains no deep learning experiments and no RL experiments. The claim about computational efficiency ("this approach is parallelizable and as such scales well to massive complex problems," Section 2) is an architectural observation, not an empirical finding. The discussion of weight sharing and dropout masks (Section 2, Section 4) is speculative. A reader looking for evidence that Bootstrapped Thompson Sampling actually works with deep neural networks will find none.

**Mitigation status:** Not addressed within the paper. The authors would go on to develop Bootstrapped DQN in subsequent work, which did demonstrate the method with deep networks on Atari benchmarks. But within this paper, the RL and deep learning claims are unsupported. The paper's transparency about this is limited — the abstract and introduction create a strong impression that the method is demonstrated for deep learning, but the only quantitative evidence is a two-arm bandit. The paper does not explicitly acknowledge this as a limitation.

---

### 6.3 The Computational Cost of Maintaining K Parallel Models Is Not Analyzed or Compared to Alternatives

**The assumption or constraint:** The online, incremental variants of the algorithm (Algorithms 5 and 6) maintain `$K$` parallel models — for deep RL, `$K$` separate neural networks — each trained on all observed data but with different exponential weights. The paper states that "in its most naive implementation this parallel bootstrap will have a computational cost per timestep D times larger than a greedy algorithm" (Section 3) and speculates that "it may be possible to share some computation between models and provide significant savings" (Section 3) or that "a specially constructed dropout mask for each bootstrap sample" could reduce overhead (Section 2). However, the paper makes **no attempt to quantify this cost** in any setting — no FLOPs analysis, no wall-clock measurements, no comparison of the total computational budget required to achieve a given level of performance relative to alternative exploration methods.

**The consequence:** The claimed scalability advantage of the bootstrap over MCMC or conjugate methods rests on the assumption that training `$K$` models in parallel is practically feasible. But for deep RL with networks like those in DQN (hundreds of thousands to millions of parameters), `$K$` might need to be substantial — the paper provides no guidance on how large `$K$` must be for the bootstrap distribution to adequately approximate the posterior. If `$K=10$` is needed, the computational cost is 10× that of a standard DQN agent. An alternative exploration method — say, ϵ-greedy with `$\epsilon = 0.1$`, which adds zero computational overhead — might achieve comparable or better exploration at dramatically lower cost. Without a cost-benefit analysis, the practitioner cannot assess whether the bootstrap approach is worth its computational premium.

Furthermore, the parallelizability argument (each model can be trained on a separate processor) applies only to the training phase. During action selection, the agent must evaluate the selected Q-network on the current state, which is a single forward pass — but maintaining `$K$` networks in memory still requires `$K \times$` the GPU memory of a single network, which may be prohibitive for large models. The paper's speculation about weight sharing and dropout masks might reduce memory overhead, but these are not implemented or tested.

**What evidence exists:** None. The paper includes no computational cost analysis. The `$D$` models mentioned in Section 3 are never instantiated. The parallel Algorithm 6 is presented as pseudocode but never run. The claim that computation can be shared via dropout masks or tree-structured architectures is purely speculative. The paper does not compare the total compute (training time × number of models) required for BootstrapThompson to achieve a given regret level against the compute required by ϵ-greedy, UCB, or true Thompson sampling (where the latter is feasible).

**Mitigation status:** Not addressed. The paper acknowledges the cost qualitatively ("one drawback of Algorithm 3 is that the computational cost per timestep grows with the amount of data H_t," Section 3) and proposes the parallel variant as a solution, but the parallel variant's cost is itself not analyzed. The speculation about reducing cost through weight sharing and dropout is identified as future work. The paper provides no empirical evidence that the approach is more computationally efficient than alternatives when total computational expenditure is accounted for, which is particularly problematic given that the paper's central motivation is computational tractability relative to true posterior sampling.

---

### 6.4 The Formal Equivalence to Thompson Sampling Is Established Only for Conjugate Models with Finite Outcome Spaces

**The assumption or constraint:** Section 3.2 establishes that Bootstrapped Thompson Sampling with the Bayesian bootstrap and appropriately configured artificial data is **exactly equivalent** to traditional Thompson sampling for Bernoulli bandits with Beta priors and for multinomial bandits with Dirichlet priors. The paper states this equivalence clearly and extends it: "We expect that, with appropriately designed schemes for generating artiﬁcial data, such equivalences can also be established for a far broader range of problems." However, the equivalence proof relies on two properties that are specific to these models: (1) the outcome space is finite (binary for Bernoulli, categorical for multinomial), and (2) the prior is conjugate (Beta for Bernoulli, Dirichlet for multinomial). These properties are what allow the posterior to be expressed as a normalized sum of exponential random variables, which is in turn identical to the Bayesian bootstrap with artificial pseudocounts.

**The consequence:** For the settings that motivate the paper — deep neural networks, continuous state and action spaces, non-conjugate priors, complex likelihoods — the formal equivalence does not hold. The paper provides no theoretical guarantee that the bootstrap approximates the posterior for these settings. The bootstrap is an established tool for uncertainty quantification in i.i.d. settings (confidence intervals for means, regression coefficients), but its behavior as a posterior approximation for the parameters of a deep neural network trained on temporally dependent RL data is not characterized by any result in this paper. The argument for using the bootstrap in these settings is therefore an **extrapolation** from the conjugate case, not a deduction from established theory. The paper acknowledges this implicitly ("We expect... such equivalences can also be established") but does not prove it, and the quality of the approximation in practice is unknown from the paper's evidence.

This is particularly concerning for the RL extension, where the object being bootstrapped is a Q-function fitted via temporal-difference learning. TD learning involves bootstrapping in the RL sense (using current value estimates as targets for future value estimates), which introduces additional sources of variance and potential instability. The paper's bootstrap operates on episodes — resampling entire trajectories with replacement — but the value function fitting step inside `$\phi$` uses TD methods that are not themselves bootstrapped in the statistical sense. The interaction between these two levels of resampling/approximation is not analyzed.

**What evidence exists:** The equivalence is proven constructively for the Beta-Bernoulli and Dirichlet-multinomial cases (Section 3.2). Beyond these, the paper offers no theoretical analysis, no empirical comparisons between BootstrapThompson and true Thompson sampling on non-conjugate models, and no characterization of when the bootstrap approximation might fail. The bandit simulation (Section 3.1) does not include true Thompson sampling as a baseline, so even for the simple two-arm case, we cannot assess how close the bootstrap approximation is to the real thing.

**Mitigation status:** Partially addressed through the formal equivalence proofs, which establish that the method recovers the correct behavior in the special cases where correctness is well-defined. This is a necessary condition for the method to be considered a principled approximation, but it is not sufficient to guarantee good performance in the settings of interest. The paper's conjecture that equivalences extend to broader problem classes is flagged as an expectation, not a result, and the paper does not pursue it further.

---

### 6.5 The Difficulty Estimation Problem for Artificial Data Design Is Unbounded for Complex Environments

**The assumption or constraint:** The artificial data must be generated from a prior distribution `$\tilde{P}$` that encodes the agent's beliefs about what observations are possible. For the bandit setting, this is straightforward: the prior is a distribution over (action, reward) pairs where the actions are discrete and few, and the rewards are scalar values. But the RL extension (Algorithm 4) requires generating **entire artificial episodes** — sequences of states, actions, rewards, and next-states of length `$\tau$`. The paper's suggestion for doing this is: "sample state-action pairs from a diffusely mixed generative model and assign them stochastically optimistic rewards and random state transitions" (Section 4).

**The consequence:** This prescription is **not implementable** for most realistic RL environments without solving a problem as hard as the original exploration challenge. Consider an agent learning to play an Atari game from pixels. What is a "diffusely mixed generative model" over Atari frames? The space of possible 84×84×4 pixel observations is astronomically large, and the vast majority of configurations correspond to noise, not valid game states. Generating artificial episodes that are **plausible enough to serve as a useful prior** — states that look like real game frames, transitions that respect (or reasonably approximate) game dynamics, rewards that are optimistic but not absurd — requires either: (a) a pretrained generative model of the environment (which presupposes substantial knowledge of the environment, undermining the exploration motivation), (b) hand-designed artificial data (feasible for toy problems but not for complex environments), or (c) ignoring realism and using random noise as artificial data (which would likely provide a uselessly weak prior).

The paper's mention of using "prior data... from episodes of experience with actions selected by an expert agent" (Section 4) as artificial data partially mitigates this — expert demonstrations can serve as a strong prior. But this presupposes access to an expert, which is not available in the pure exploration setting the paper primarily addresses. More fundamentally, the whole point of exploration is to discover behaviors that the agent (or any expert) has not yet demonstrated; a prior built solely from expert data cannot encourage exploration beyond the expert's demonstrated behaviors.

**What evidence exists:** None. The RL setting is not tested, so the feasibility of designing artificial data generators for any specific RL environment is unexamined. The paper does not discuss the engineering challenges of generating realistic artificial episodes, does not propose any concrete generative models, and does not analyze how the quality of the artificial data affects the exploration behavior of the resulting agent. For the bandit case, the artificial data generator is trivial (uniform rewards over [0,1]), and the method works — but the bandit's observation space is a single scalar, making the design problem vastly simpler than in RL.

**Mitigation status:** Not addressed. The paper treats the artificial data generator as an input to the algorithm and does not discuss how to construct it for specific environments. The suggestion of using expert data partially addresses the issue for settings where demonstrations are available, but this is not a general solution. The core challenge — how to encode a useful prior over complex observation spaces without effectively solving the exploration problem in the process — is left entirely to the practitioner.

---

### 6.6 The Method Offers No Mechanism for Adapting the Prior Online or Across Episodes

**The assumption or constraint:** In both the bandit algorithm (Algorithm 3) and the RL algorithm (Algorithm 4), the artificial data generation process is **fixed** for the duration of learning. The prior strength `$M$` and the distribution `$\tilde{P}$` are chosen at the outset and do not change as the agent accumulates real experience. This is in contrast to true Bayesian inference, where the prior's influence naturally diminishes as data accumulates — the posterior concentrates toward the data-generating distribution, and the agent's behavior transitions smoothly from exploration (driven by prior uncertainty) to exploitation (driven by data). In the bootstrap framework, the ratio `$M/N$` (artificial to real data) does shrink as `$N$` grows, which provides a natural decay of prior influence. However, the **content** of the artificial data — what observations it contains — remains static.

**The consequence:** The prior encoded in the artificial data may be well-suited to early exploration (when the agent knows little) but **actively harmful later** (when the agent has learned the environment's structure and the prior now encodes misleading information). For example, if the artificial data for an RL agent includes transitions that are physically impossible in the true environment (because the generative model was "diffusely mixed" and didn't respect environment constraints), those impossible transitions will continue to influence the bootstrap distribution even after the agent has observed thousands of real episodes that never exhibit such transitions. The agent may persistently explore toward states that cannot actually be reached, wasting computation and potentially degrading policy quality.

More subtly, the fixed prior cannot adapt to what the agent has learned about **which regions of the state space are worth exploring.** An ideal exploration strategy should shift its optimism from regions that have been thoroughly explored (and found wanting) to regions that remain uncertain. A fixed artificial data generator cannot do this — it sprinkles optimism uniformly according to `$\tilde{P}$`, regardless of where the agent has already been. This is a structural limitation of the approach: the prior is encoded as data, and data, once generated, does not adapt. True Thompson sampling with a dynamically updated posterior naturally reallocates uncertainty to regions that remain unexplored because the posterior's variance is data-dependent. The bootstrap with static artificial data cannot fully replicate this dynamic reallocation, because the prior component of the dataset never changes.

**What evidence exists:** None. The paper does not study whether static artificial data causes persistent exploratory inefficiency in later stages of learning. The bandit simulation (Figure 1) shows that the method with `$M=2$` eventually converges to the optimal action, suggesting the static prior does not prevent convergence in that simple case. But the horizon is only 10,000 steps, and the optimal arm's advantage is unambiguous once discovered. In more complex environments where the agent must continually decide which of many unexplored regions to probe, the inability to adapt the prior could matter more — and this is never tested.

**Mitigation status:** Not addressed. The paper does not discuss adaptive artificial data generation, does not propose mechanisms for modifying `$\tilde{P}$` or `$M$` as learning progresses, and does not analyze whether the natural decay of the ratio `$M/N$` is sufficient to prevent prior-induced inefficiencies at later stages. The parallel variant (Algorithm 6) gives each bootstrap model its own independent set of artificial data — which adds diversity but still does not adapt over time. This limitation is structural: the method gains its simplicity and parallelizability from treating the prior as static data, and making it adaptive would require redesigning the bootstrap procedure to incorporate dynamic prior information, potentially reintroducing the computational complexity the approach was designed to avoid.
