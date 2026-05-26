# Rainbow: Combining Improvements in Deep Reinforcement Learning

**URL:** [https://ojs.aaai.org/index.php/AAAI/article/view/11796/11655](https://ojs.aaai.org/index.php/AAAI/article/view/11796/11655)

## 🎯 Pitch

A single agent stacking six separate DQN improvements matches the original DQN's final performance using only 3% of the training data, and ultimately more than triples the median human-normalized score across 57 Atari games—proving these innovations are not just individually useful but strikingly complementary. Surprisingly, the ablation shows that removing prioritized replay or multi-step learning alone causes crippling median drops of over 100 percentage points, while some components like dueling are almost game-dependent in their benefit.

---

## 1. Executive Summary

This paper examines whether six independently developed extensions to the DQN algorithm—Double Q-learning (decoupling action selection from evaluation to reduce overestimation bias), prioritized experience replay (sampling transitions proportionally to TD-error magnitude), dueling network architecture (factorizing Q-values into separate value and advantage streams), multi-step learning (using truncated n-step returns instead of 1-step bootstrap targets), distributional Q-learning (modeling the full distribution of returns rather than just the mean), and Noisy Nets (stochastic network layers for state-conditional exploration)—are complementary and can be fruitfully combined into a single integrated agent the authors call **Rainbow**. Evaluated across the full suite of 57 Atari 2600 games from the Arcade Learning Environment, Rainbow achieves a median human-normalized score of 231% in the no-ops regime, matching DQN's final performance after only 7M frames and surpassing any published baseline within 44M frames. Ablation studies reveal that prioritized replay and multi-step learning are the two most crucial components (each helping almost uniformly across 53 out of 57 games), while the contributions of dueling networks and double Q-learning are more game-dependent, establishing that most—but not all—of these orthogonal improvements compound beneficially when integrated.

## 2. Context and Motivation

### The Core Problem: Complementary or Redundant? Nobody Knew

By 2017, deep reinforcement learning had undergone a Cambrian explosion of algorithmic improvements to the DQN algorithm. DQN itself was only two years old (Mnih et al., 2015), yet the community had already produced a proliferation of extensions: Double DQN, prioritized replay, dueling networks, distributional RL, multi-step learning, Noisy Nets, and many more. Each of these was published as a standalone paper demonstrating substantial performance gains over DQN in isolation. But a fundamental question remained completely unanswered: **are these improvements orthogonal and additive, or do they overlap in ways that make combining them redundant or even harmful?**

This is not a trivial question. In machine learning, stacking improvements often yields diminishing returns—or worse, negative interactions. Two techniques that each independently improve performance might address the same underlying bottleneck, so combining them buys you nothing. Or they might interact destructively: one technique's assumptions could be violated by another's modifications to the learning dynamics. The community had no systematic evidence either way for the growing catalog of DQN extensions.

The paper frames this as the central motivating gap:

> "Each of these algorithms enables substantial performance improvements in isolation. Since they address radically different issues, and since they build on a shared framework, they could plausibly be combined. In some cases this has been done: Prioritized DDQN and Dueling DDQN both use double Q-learning, and Dueling DDQN was also combined with prioritized replay. In this paper we propose to study an agent that combines all the aforementioned ingredients."

The key phrase is "could plausibly be combined." Plausibility is not evidence. The paper sets out to convert plausibility into empirical fact—or to discover that some combinations are not, in fact, compatible.

### Why This Problem Matters

The significance of this question extends beyond the specific algorithms studied. There are at least three reasons why systematically testing whether independent improvements compound is important for the field.

**First, scientific: understanding what matters.** If you have six techniques that each improve performance in isolation, you don't know which ones are addressing fundamental limitations versus which ones are merely papering over symptoms. By studying their interactions—noting, for example, that double Q-learning's effect largely disappears when combined with distributional RL's constrained value support—you learn something about *why* each technique works and what the true bottlenecks in deep Q-learning actually are. The ablation analysis in this paper serves this exact purpose: it reveals that some "improvements" (like dueling networks) are largely redundant in the presence of the full combination, while others (like prioritized replay and multi-step learning) remain critical regardless of what else is included.

**Second, practical: what should practitioners implement?** For anyone building a deep RL system, the question is not "does technique X improve DQN?" but rather "given that I'm already using techniques A, B, and C, should I also add X?" The marginal benefit of a technique depends on what else is in the system. A practitioner reading six separate papers—each showing improvements over vanilla DQN—cannot determine the marginal value of each addition from those papers alone. This paper provides exactly that evidence through its ablation study, which measures the *drop* in performance when each component is removed from the full Rainbow agent. These drops represent the marginal contribution of each component *in the context of all others being present*—which is precisely what a practitioner needs to know when deciding where to invest implementation effort.

**Third, methodological: establishing a pattern for the field.** The DQN extension literature was following a predictable pattern: new paper, new technique, improvement over DQN baseline, repeat. This paper established a different template: systematically combine, then ablate. This matters because the same dynamic recurs throughout deep learning—in computer vision (combining architectural innovations), in NLP (combining attention mechanisms, pretraining objectives, and decoding strategies), and in generative modeling. The Rainbow paper provided a clear demonstration that combination-plus-ablation studies are feasible, informative, and can produce state-of-the-art results, encouraging the field to move beyond isolated comparisons against weak baselines.

### Prior Approaches and Where They Fell Short

To understand what this paper contributes, we need to examine what each of the six extensions addresses and why they might—or might not—be complementary.

**DQN (Mnih et al., 2015): the starting point.** The baseline DQN algorithm combines Q-learning with deep convolutional neural networks, using experience replay (storing transitions in a buffer and sampling uniformly for training) and target networks (a periodically updated copy of the online network used for computing bootstrap targets) to stabilize learning. The loss function minimized at each update is:

$$(R_{t+1} + \gamma_{t+1} \max_{a'} q_{\bar{\theta}}(S_{t+1}, a') - q_{\theta}(S_t, A_t))^2$$

This is a 1-step temporal difference update: the target is the immediate reward plus the discounted maximum Q-value at the next state. DQN achieved superhuman performance on several Atari games, but its limitations were quickly recognized:

- **Overestimation bias**: the max operator in the target systematically overestimates Q-values, because it selects the action with the highest *estimated* value, and estimation errors are asymmetric (positive errors are more likely to be selected than negative ones). This can lead to unstable learning and poor policies.
- **Uniform replay**: all transitions in the buffer are sampled with equal probability, regardless of their learning value. Transitions with surprising outcomes (large TD errors) are exactly those from which the agent has the most to learn, but DQN treats them identically to transitions where the agent's predictions are already accurate.
- **Inefficient value representation**: DQN's network architecture outputs Q-values directly, without distinguishing between the inherent value of a state and the relative advantage of each action. In many states, the action choice barely matters (e.g., when no enemies are nearby), and learning accurate state values could generalize better than learning action-specific Q-values for every action in every state.
- **1-step bootstrap targets**: The target uses only the immediate reward plus the next-step value estimate. This means reward signals propagate backward through the Bellman update one step at a time—slowly, especially when rewards are sparse and delayed.
- **Expected return only**: DQN learns the *expectation* of the discounted return, discarding all information about the distribution of possible outcomes. In stochastic environments, knowing whether returns are typically concentrated around the mean or widely dispersed (high variance) could inform risk-sensitive decision-making and lead to more robust policies.
- **Undirected exploration**: The $\epsilon$-greedy exploration strategy acts uniformly at random with probability $\epsilon$, regardless of the agent's uncertainty about different parts of the state space. This is particularly ineffective in games like Montezuma's Revenge, where long sequences of specific actions are required to encounter the first reward—random exploration has an astronomically low probability of discovering these sequences.

**Double DQN (van Hasselt, Guez, and Silver, 2016)** addresses the overestimation bias by decoupling action selection from action evaluation in the bootstrap target:

$$(R_{t+1} + \gamma_{t+1} q_{\bar{\theta}}(S_{t+1}, \arg\max_{a'} q_{\theta}(S_{t+1}, a')) - q_{\theta}(S_t, A_t))^2$$

The online network $\theta$ selects the best action, but the target network $\bar{\theta}$ evaluates it. This reduces the correlation between selection and evaluation errors that causes overestimation. **Limitation when considered alone**: Double DQN only addresses the max-bias problem. It doesn't help with data efficiency, exploration, representation learning, or reward propagation speed. It is a surgical fix for one specific pathology.

**Prioritized experience replay (Schaul et al., 2015)** replaces uniform sampling from the replay buffer with sampling proportional to the TD error magnitude:

$$p_t \propto \left| R_{t+1} + \gamma_{t+1} \max_{a'} q_{\bar{\theta}}(S_{t+1}, a') - q_{\theta}(S_t, A_t) \right|^\omega$$

where $\omega$ controls how aggressively the distribution skews toward high-error transitions ($\omega = 0$ recovers uniform sampling). Transitions with large TD errors—where the agent's prediction was most wrong—are replayed more frequently, focusing learning on the most informative experiences. Importance sampling corrections compensate for the non-uniform sampling distribution. **Limitation when considered alone**: Prioritized replay changes *which* transitions are learned from, but not *what* is learned from them (still 1-step Q-learning targets), *how* values are represented (still a standard network architecture), or *how* the agent explores. It also introduces a risk: stochastic transitions with inherently unpredictable rewards may maintain high priority even after the agent has learned all it can about them, wasting capacity.

**Dueling networks (Wang et al., 2016)** factor the Q-value into a state value $v(s)$ and action advantages $a(s, a)$, combined via:

$$q_{\theta}(s, a) = v_{\eta}(f_{\xi}(s)) + a_{\psi}(f_{\xi}(s), a) - \frac{\sum_{a'} a_{\psi}(f_{\xi}(s), a')}{N_{\text{actions}}}$$

The shared encoder $f_{\xi}$ processes the input, then two separate streams estimate the value and advantages. Subtracting the mean advantage ensures identifiability (otherwise, adding a constant to all advantages and subtracting it from the value would produce the same Q-values). **Why this helps**: In states where the action choice doesn't matter much (e.g., no threats, no opportunities), the advantages are all near zero, and the network learns primarily about state values—which generalize across actions. In states where actions matter a lot, the advantage stream captures action-specific differences. **Limitation when considered alone**: This is purely an architectural change—it changes how Q-values are computed from the neural network's internal representations, but doesn't affect the learning objective, the replay strategy, the exploration mechanism, or the temporal difference target.

**Multi-step learning** replaces the 1-step bootstrap target with an n-step truncated return:

$$R^{(n)}_t = \sum_{k=0}^{n-1} \gamma^{(k)}_t R_{t+k+1}$$

The loss becomes:

$$(R^{(n)}_t + \gamma^{(n)}_t \max_{a'} q_{\bar{\theta}}(S_{t+n}, a') - q_{\theta}(S_t, A_t))^2$$

**How this helps**: Multi-step returns propagate observed rewards backward n steps per update instead of just 1, dramatically accelerating learning when rewards are sparse or delayed. They also reduce bias from function approximation error in the bootstrap step (because the bootstrap contributes less to the target when n is larger), at the cost of increased variance (because n actual rewards are summed, each with their own noise). **Limitation when considered alone**: Multi-step returns change the target but don't help with exploration, representation, or which experiences are sampled. They also introduce a bias-variance tradeoff that requires tuning n; the optimal n may vary across environments.

**Distributional Q-learning (Bellemare, Dabney, and Munos, 2017)** learns the full probability distribution of returns rather than just the mean. The return distribution is represented as a categorical distribution over a discrete support $z$ with $N_{\text{atoms}}$ atoms spanning $[v_{\text{min}}, v_{\text{max}}]$. For each state-action pair, the network outputs a probability mass $p^i_{\theta}(s, a)$ on each atom $z^i$. The learning objective is the Kullback-Leibler divergence between the current estimate $d_t$ and a projected target distribution:

$$d'_t \equiv (R_{t+1} + \gamma_{t+1} z, \; p_{\bar{\theta}}(S_{t+1}, a^*_{t+1}))$$

$$D_{\text{KL}}(\Phi_z d'_t \| d_t)$$

where $\Phi_z$ projects the target distribution (which may fall between atoms after scaling and shifting) back onto the fixed support $z$. **Why this helps**: Learning the full distribution provides a richer learning signal than the mean alone—the shape of the return distribution encodes information about environmental stochasticity that can help the agent learn more robust policies. It also provides an auxiliary learning objective that may serve as a form of representation learning. **Limitation when considered alone**: Distributional RL changes the loss function and the network output structure but doesn't address overestimation, replay prioritization, exploration, or the architecture of the value function approximator. It also constrains returns to a fixed range $[v_{\text{min}}, v_{\text{max}}]$, which implicitly clips values.

**Noisy Nets (Fortunato et al., 2017)** replace standard linear layers with noisy layers:

$$y = (b + Wx) + (b_{\text{noisy}} \odot \epsilon_b + (W_{\text{noisy}} \odot \epsilon_w)x)$$

where $\epsilon_b$ and $\epsilon_w$ are random variables (factorised Gaussian noise, in the variant used by Rainbow). The network learns the parameters of both the deterministic and noisy streams. **How this helps**: The noise is part of the network's weights, not added to the action selection process externally. This means the network can learn to modulate its own exploration: it can reduce noise in parts of the state space where it's confident and maintain noise where it's uncertain, implementing a form of state-conditional exploration that is far more efficient than $\epsilon$-greedy's undirected random actions. Over training, the network can "self-anneal" its exploration by learning to ignore the noisy stream. **Limitation when considered alone**: Noisy Nets change how actions are selected (the exploration mechanism), but don't affect the learning objective, the replay strategy, the network architecture for representing values, or the temporal difference target. They also introduce additional parameters and noise variables that must be managed.

### Partial Combinations and the Missing System-Level View

The paper notes that some pairwise combinations had been tried before—Prioritized DDQN, Dueling DDQN, Dueling DDQN with prioritized replay—but these were ad hoc. No one had attempted a systematic integration of all major extensions, and critically, **no one had studied the marginal contribution of each component in the presence of the others**. This is the key distinction: prior work showed "A + B is better than A alone," but couldn't answer "if I already have B, C, D, and E, does adding A still help?"

The very existence of partial combinations raises the question that motivates this paper: if you can successfully combine two or three extensions, can you combine all six? And if you can, do they all remain useful, or do some become redundant? The dueling network architecture, for instance, helps by separating value and advantage learning—but does this matter as much when distributional RL is already providing a richer learning signal? Double Q-learning addresses overestimation bias—but does distributional RL's constrained value support $[v_{\text{min}}, v_{\text{max}}]$ implicitly mitigate overestimation as well, by capping values? These are exactly the kinds of questions that can only be answered by building the full system and then systematically removing components, which is precisely the experimental design of this paper.

### How This Paper Positions Itself

The paper positions itself as an **integration and analysis study**, not as proposing a fundamentally new algorithm. This is worth emphasizing: Rainbow as an algorithm is "just" the combination of six existing ideas, plus the engineering work required to make them interoperate. The contribution is not the individual pieces but rather:

1. **Demonstrating feasibility**: showing that all six extensions can be made to work together in a single agent, which required resolving non-obvious integration challenges (e.g., adapting the dueling architecture to output distributions rather than scalar Q-values, using the KL loss rather than TD error for prioritized replay in the distributional setting, replacing the 1-step distributional loss with an n-step variant).

2. **Quantifying complementarity**: through the ablation study, establishing which components provide marginal benefits in the full combination and which have their contributions subsumed by others.

3. **Achieving state-of-the-art results**: Rainbow's median human-normalized score of 231% in the no-ops regime substantially exceeded all published baselines at the time, setting a new performance standard on the Atari 2600 benchmark.

4. **Providing a new baseline**: by releasing a single agent that integrates six major ideas, the paper created a stronger foundation that subsequent research could build upon (rather than each new paper comparing only against vanilla DQN, which had long since been superseded).

The paper is explicit about what it is *not*: it doesn't claim to be exhaustive. The Discussion section notes many promising directions that were not included—policy gradient methods, episodic control, alternative exploration strategies like Bootstrapped DQN or count-based exploration, auxiliary tasks, recurrent architectures, hierarchical RL—and frames Rainbow as a starting point for even more comprehensive integrated agents. This is the key positioning: **Rainbow is not the end of the integration story, but the first systematic attempt to show that integration is both possible and highly productive.** Rather than proposing one more independent improvement, the paper argues for a different kind of research contribution: demonstrating that existing improvements are complementary and that the field should think in terms of integrated systems, not isolated algorithmic tweaks.

## 3. Technical Approach

### 3.1 Reader Orientation

This is an **integration and empirical analysis paper** whose core idea is that six independently developed extensions to DQN—each addressing a fundamentally different limitation—can be combined into a single algorithm (called Rainbow) that compounds their benefits rather than suffering from destructive interference, and that systematically removing each component reveals which improvements remain essential in the presence of the others. The paper builds a complete Atari-playing agent by resolving the non-obvious engineering challenges of making these extensions interoperate (e.g., adapting dueling architectures for distributional outputs, using KL loss instead of TD error for prioritization), then validates the combination by measuring both aggregate performance and per-component marginal contributions across all 57 Atari 2600 games.

### 3.2 Big-Picture Architecture (Diagram in Words)

The Rainbow agent has five major interconnected components, which process experience through a pipeline:

1. **Noisy Network Architecture** (exploration mechanism): All linear layers in the Q-network are replaced with noisy layers that inject learned, state-conditional noise into the weights. This provides exploration without an external $\epsilon$-greedy schedule—the network learns when to explore and when to act deterministically.

2. **Dueling Network Architecture** (value representation): The convolutional encoder $f_{\xi}$ processes the input frames into a shared representation $\phi$, which feeds into two separate streams: a value stream $v_{\eta}(\phi)$ with $N_{\text{atoms}}$ outputs (one per distributional atom) and an advantage stream $a_{\psi}(\phi, a)$ with $N_{\text{atoms}} \times N_{\text{actions}}$ outputs. These are combined via the standard dueling aggregation to produce per-action probability distributions over returns.

3. **Distributional Multi-Step Loss** (learning objective): For each sampled transition, the agent computes an n-step truncated return $R^{(n)}_t$ by summing $n$ actual observed rewards (with $n = 3$). The target distribution is constructed by taking the online network's predicted distribution at state $S_{t+n}$, shifting it by the n-step return, contracting it by the n-step discount $\gamma^{(n)}_t$, and projecting it onto the fixed support $z$. The loss is the Kullback-Leibler divergence between this target distribution and the current network's prediction, with double Q-learning used for action selection in the bootstrap step.

4. **Prioritized Experience Replay** (data selection): Transitions are sampled from the replay buffer with probability proportional to $(\text{KL loss})^{\omega}$, where $\omega = 0.5$. Importance sampling weights correct for the non-uniform sampling distribution, with the correction strength $\beta$ linearly annealed from 0.4 to 1.0 over training.

5. **Environment Interaction** (data generation): The agent acts fully greedily with respect to the mean of its predicted Q-value distributions (since exploration is handled internally by the noisy networks). Transitions $(S_t, A_t, R_{t+1}, \gamma_{t+1}, S_{t+1})$ are stored in a replay buffer, and training begins after 80K frames are collected.

Information flows as follows: raw pixel frames enter the convolutional encoder → shared representation feeds value and advantage streams → noisy layers inject exploration noise into all linear transformations → advantage and value streams are aggregated via the dueling combination → softmax produces per-action probability distributions over return atoms → the mean of each distribution gives Q-values for action selection → transitions are stored in the replay buffer → prioritized sampling selects transitions based on KL loss magnitude → multi-step distributional loss is computed using double Q-learning → gradients update the network parameters (including the noise parameters).

### 3.3 Roadmap for the Deep Dive

- **First**, I'll explain the base DQN algorithm and its loss function in detail, since all six extensions modify different parts of this foundation and understanding the base is essential for understanding what each extension changes.
- **Second**, I'll walk through each of the six extensions individually—their motivation, their mathematical formulation, and their implementation details—building understanding of what each component does before seeing how they interact.
- **Third**, I'll explain the integration challenges: how the components are made to interoperate, including the key design decisions that differ from the standalone versions (e.g., using KL loss for prioritization instead of TD error, adapting the dueling architecture for distributional outputs, replacing the 1-step distributional loss with an n-step variant).
- **Fourth**, I'll detail the full Rainbow loss function and how all components fit together in a single update.
- **Fifth**, I'll cover the hyperparameter configuration and experimental setup, since Rainbow's performance depends on specific choices (e.g., $n = 3$, $\omega = 0.5$, 80K frame warmup, Adam optimizer with reduced learning rate) that were empirically determined through limited manual coordinate descent.

### 3.4 Detailed, Sentence-Based Technical Breakdown

This is primarily an **integration and empirical analysis paper** whose core idea is that six independently developed improvements to DQN address orthogonal limitations and can be successfully combined into a single agent whose performance exceeds any individual baseline, and that ablation studies can quantify the marginal contribution of each component in the context of the full combination.

---

#### Base DQN: The Foundation All Extensions Modify

To understand what Rainbow integrates, we must first understand the baseline DQN algorithm (Mnih et al., 2015) that all six extensions modify. DQN combines Q-learning—a classic reinforcement learning algorithm for learning action-value functions—with deep convolutional neural networks and two stabilization techniques: experience replay and target networks.

**The Q-learning update.** In standard Q-learning, the agent maintains an estimate $q(s, a)$ of the expected discounted return when taking action $a$ in state $s$ and following the optimal policy thereafter. After observing a transition $(S_t, A_t, R_{t+1}, S_{t+1})$, the Q-value for the taken action is updated toward a bootstrap target:

$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \left[ R_{t+1} + \gamma \max_{a'} Q(S_{t+1}, a') - Q(S_t, A_t) \right]$$

where $\alpha$ is the learning rate, $\gamma$ is the discount factor, and $\max_{a'} Q(S_{t+1}, a')$ is the estimated value of the best action at the next state. The term in brackets is the **temporal difference (TD) error**: the difference between the bootstrap target (immediate reward plus discounted best future value) and the current estimate.

**How DQN adapts this for deep networks.** DQN replaces the tabular Q-values with a convolutional neural network $q_{\theta}$ parameterised by $\theta$. The network takes as input a stack of raw pixel frames (providing temporal context) and outputs Q-values for all actions. Rather than updating after every transition, DQN uses **experience replay**: transitions $(S_t, A_t, R_{t+1}, \gamma_{t+1}, S_{t+1})$ are stored in a circular buffer holding the last 1 million transitions, and mini-batches are sampled uniformly at random for training. This breaks temporal correlations between consecutive updates and allows each transition to be reused multiple times.

The **target network** is a second copy of the Q-network, parameterised by $\bar{\theta}$, which is periodically copied from the online network $\theta$ (every 32K frames in Rainbow's configuration). The target network is used for computing the bootstrap target, while gradients are only back-propagated through the online network. This prevents the target from moving during optimization, stabilising what would otherwise be a moving-target regression problem.

**The DQN loss function.** For a mini-batch of transitions sampled uniformly from the replay buffer, DQN minimises:

$$\mathcal{L}_{\text{DQN}}(\theta) = \left( R_{t+1} + \gamma_{t+1} \max_{a'} q_{\bar{\theta}}(S_{t+1}, a') - q_{\theta}(S_t, A_t) \right)^2$$

Relating this to the Q-learning update above: the squared error replaces the incremental update (since we're doing gradient descent on a loss function rather than tabular updates), the target network $\bar{\theta}$ provides the bootstrap Q-values, and the online network $\theta$ provides the current Q-values for the taken action. The max over actions selects the estimated best action at the next state—this is the **greedy policy** with respect to the target network's Q-values.

**Why DQN works despite its simplicity.** The combination of experience replay and target networks addresses two fundamental challenges of combining neural networks with TD learning. First, neural networks assume i.i.d. data, but consecutive transitions in an MDP are highly correlated—experience replay approximates i.i.d. sampling by drawing randomly from a large buffer. Second, the TD target depends on the same network parameters being optimized—the target network breaks this dependency by providing a (temporarily) fixed target. RMSprop, a variant of stochastic gradient descent that normalises gradients by a running average of their recent magnitude, is used as the optimizer.

**DQN's known limitations (the motivation for each extension).** The six extensions each target a specific weakness of this baseline:

- The **max operator** in the target selects the action with the highest *estimated* value; since estimates contain errors, this systematically selects actions with positive errors, causing overestimation bias (→ Double Q-learning).
- **Uniform replay** wastes capacity on transitions where the agent's predictions are already accurate (→ Prioritized replay).
- **Single Q-value output** conflates the inherent value of a state with the advantage of each action, making it harder to learn in states where actions don't matter (→ Dueling networks).
- **1-step bootstrap targets** propagate reward information backward one step per update, which is slow when rewards are sparse (→ Multi-step learning).
- **Learning only the mean** of the return distribution discards information about outcome variability that could inform better policies (→ Distributional RL).
- **$\epsilon$-greedy exploration** acts randomly regardless of the agent's uncertainty about different states, failing to direct exploration toward informative regions (→ Noisy Nets).

With this foundation established, I now walk through each extension in detail, explaining its mechanism, its mathematical formulation, and what limitation it addresses.

---

#### Extension 1: Double Q-Learning (Addressing Overestimation Bias)

**The problem: maximization bias in Q-learning.** The DQN target $R_{t+1} + \gamma_{t+1} \max_{a'} q_{\bar{\theta}}(S_{t+1}, a')$ uses the same network (the target network $\bar{\theta}$) for both selecting the best action (via the max operator) and evaluating that action's value. When Q-value estimates contain noise—which they always do during learning, due to finite samples, function approximation error, and environmental stochasticity—the max operator introduces an upward bias. This is because the action with the highest *estimated* value is likely to be one whose estimation error happens to be positive, even if its true value is not the highest. Mathematically, if $E[\epsilon_i] = 0$ (unbiased estimates with symmetric errors), then $E[\max_i (Q_i + \epsilon_i)] \geq \max_i Q_i$, with strict inequality whenever there is uncertainty about which action is best. This overestimation can lead to unstable learning and suboptimal policies, especially in environments with many actions or high stochasticity.

**The solution: decouple selection from evaluation.** Double Q-learning (van Hasselt, 2010) eliminates this bias by using two independent value functions: one to select the action, and the other to evaluate it. In the deep RL context (van Hasselt, Guez, and Silver, 2016), DQN already maintains two networks—the online network $\theta$ and the target network $\bar{\theta}$—which are natural candidates for this decoupling because the target network is a delayed copy of the online network, providing approximate independence.

**The Double DQN loss function.** The Double DQN target replaces the max operator with a two-step procedure: the online network selects the best action, and the target network evaluates that action:

$$\mathcal{L}_{\text{DDQN}}(\theta) = \left( R_{t+1} + \gamma_{t+1} q_{\bar{\theta}}\left(S_{t+1}, \arg\max_{a'} q_{\theta}(S_{t+1}, a')\right) - q_{\theta}(S_t, A_t) \right)^2$$

where $\arg\max_{a'} q_{\theta}(S_{t+1}, a')$ selects the action with the highest value according to the online network, and $q_{\bar{\theta}}(S_{t+1}, \cdot)$ evaluates that action using the target network.

**Operational meaning.** When the online and target networks disagree about which action is best (which happens when Q-value estimates are uncertain), Double DQN uses the online network's opinion for selection but the target network's opinion for evaluation. This means that if the online network overestimates some action's value, it might select that action—but the target network's (independent) estimate provides a more realistic evaluation, preventing the overestimation from feeding back into the bootstrap target. If both networks overestimate the same action, some bias remains, but because the target network's estimates are a delayed (and thus usually more conservative) copy, the bias is substantially reduced.

**Why this form rather than alternatives.** An alternative approach would be to simply use a single network but apply a bias correction term, as in some tabular Q-learning variants. However, such corrections are difficult to estimate accurately with function approximation. Using the online/target network split—which already exists in DQN for stability—is elegant because it requires no additional networks, no additional hyperparameters, and adds minimal computational cost (just an extra argmax and forward pass through the target network for the selected action). The key insight is that the online and target networks are sufficiently decorrelated (due to the target network update period of 32K frames) to provide meaningful bias reduction, even though they are not fully independent.

**Important subtlety: why Double DQN may matter less in Rainbow.** As the paper notes in its analysis (Section "Analysis"), when distributions are constrained to a fixed support $[v_{\text{min}}, v_{\text{max}}] = [-10, 10]$, the values are implicitly clipped—any return above 10 gets mapped to the highest atom, and any return below -10 gets mapped to the lowest. This clipping counteracts overestimation independently of double Q-learning: even if the Q-network overestimates some action's value, the distributional support prevents it from growing arbitrarily large. This is why, in the ablation study, removing double Q-learning from Rainbow had relatively small aggregate impact—the distributional value constraints were already providing implicit overestimation protection. (I will return to this in the integration section.)

---

#### Extension 2: Prioritized Experience Replay (Addressing Uniform Sampling)

**The problem: not all transitions are equally informative.** DQN samples transitions uniformly from the replay buffer, treating a correctly predicted transition identically to a surprising one. However, from a learning perspective, transitions with large TD errors—where the agent's current predictions are far from the bootstrap targets—are exactly those from which the agent has the most to learn. Uniform sampling wastes training compute on transitions where learning has already converged.

**The solution: sample proportionally to TD error magnitude.** Prioritized experience replay (Schaul et al., 2015) assigns each transition a priority based on its last-encountered absolute TD error, and samples transitions with probability proportional to this priority:

$$P(i) = \frac{p_i^{\omega}}{\sum_k p_k^{\omega}}$$

where $p_i$ is the priority of transition $i$, and $\omega$ is a hyperparameter controlling how aggressively the distribution skews toward high-priority transitions. When $\omega = 0$, this recovers uniform sampling; when $\omega = 1$, sampling is directly proportional to priority.

**The priority metric in standard (non-distributional) prioritized replay.** In the original formulation, the priority is the absolute TD error:

$$p_t = \left| R_{t+1} + \gamma_{t+1} \max_{a'} q_{\bar{\theta}}(S_{t+1}, a') - q_{\theta}(S_t, A_t) \right|$$

A small constant $\epsilon$ is added to ensure non-zero probability for all transitions. New transitions are inserted with maximum priority (to ensure they are sampled at least once), and priorities are updated whenever a transition is replayed.

**Importance sampling correction.** Sampling non-uniformly changes the distribution of updates, introducing bias in the stochastic gradient estimates (the expectation of the sampled gradient no longer matches the expectation under the data distribution). To correct for this, prioritized replay applies importance sampling weights:

$$w_i = \left( \frac{1}{N} \cdot \frac{1}{P(i)} \right)^{\beta}$$

where $N$ is the buffer size, $P(i)$ is the sampling probability of transition $i$, and $\beta$ controls the degree of correction. When $\beta = 1$, the correction fully compensates for the non-uniform sampling, yielding unbiased gradient estimates (at the cost of increased variance). When $\beta = 0$, no correction is applied. In practice, $\beta$ is linearly annealed from an initial value $\beta_0$ to 1 over the course of training—this is because the biased gradient estimates are most harmful near convergence, while early in training the bias is less problematic. Rainbow uses $\beta$ annealed from 0.4 to 1.0.

The importance sampling weights are multiplied into the loss for each transition, so the effective update is $w_i \cdot \mathcal{L}_i$ rather than $\mathcal{L}_i$.

**Rainbow's modification: KL loss as priority metric.** In the distributional setting, the TD error (based on mean Q-values) is no longer the natural priority metric because the algorithm minimises KL divergence, not squared error. Rainbow therefore uses the KL loss itself as the priority:

$$p_t \propto \left( D_{\text{KL}}(\Phi_z d^{(n)}_t \| d_t) \right)^{\omega}$$

where $d^{(n)}_t$ is the multi-step distributional target and $d_t$ is the current distribution estimate. The authors argue that "the KL loss as priority might be more robust to noisy stochastic environments because the loss can continue to decrease even when the returns are not deterministic."

**Why KL-based priority might be more robust.** Consider a stochastic transition where the reward is randomly 0 or 1 with equal probability, and the agent has learned this perfectly (its distribution correctly assigns 50% mass to each outcome). The TD error—computed from the means—will remain large because the mean (0.5) is never exactly equal to any single sampled reward (always 0 or 1). The KL loss, however, will be low because the predicted distribution matches the true distribution. Using TD error as priority would cause this transition to be replayed frequently despite the agent having nothing left to learn from it; using KL loss correctly deprioritises it.

**Why this form rather than alternatives.** A natural alternative would be to use the absolute TD error based on mean Q-values, as in the original prioritized replay paper. However, in the distributional setting, the mean is a derived quantity (computed as $z^{\top} p_{\theta}(s, a)$), and large distributional changes can occur without large changes in the mean (e.g., shifting probability mass between atoms while keeping the mean constant). The KL loss captures changes in the full distribution, not just the mean, providing a more faithful signal of learning progress. The authors report that "performance is very robust to the choice of $\omega$" when using KL-based priorities, suggesting that the KL loss provides a more reliable importance signal than the TD error.

---

#### Extension 3: Dueling Network Architecture (Addressing Value-Advantage Conflation)

**The problem: Q-values conflate state value with action advantage.** A Q-value $q(s, a)$ represents the expected return from taking action $a$ in state $s$ and following the optimal policy thereafter. This single number conflates two distinct concepts: the inherent value of being in state $s$ (regardless of which action is taken), and the additional benefit (or penalty) of choosing action $a$ specifically. In many states—for example, when no enemies are nearby in an Atari game—the action choice is near-irrelevant, and all Q-values are approximately equal to the state value. In such states, the network must learn that all actions have similar values, but it must do so by making the same "discovery" independently for each action, which is inefficient.

**The solution: factor Q-values into value and advantage streams.** The dueling network architecture (Wang et al., 2016) introduces a neural network structure with two separate computational streams that share a common convolutional encoder. The encoder $f_{\xi}$ (parameterised by $\xi$) processes the input state into a shared representation $\phi = f_{\xi}(s)$. This representation feeds into two branches:

1. **The value stream** $v_{\eta}(\phi)$, parameterised by $\eta$, which outputs a single scalar: the estimated value of state $s$, representing how good it is to be in this state regardless of action.

2. **The advantage stream** $a_{\psi}(\phi, a)$, parameterised by $\psi$, which outputs a vector of $N_{\text{actions}}$ scalars: the estimated advantage of each action—how much better or worse each action is compared to the average action in this state.

The Q-value is then reconstructed as:

$$q_{\theta}(s, a) = v_{\eta}(\phi) + a_{\psi}(\phi, a) - \frac{\sum_{a'} a_{\psi}(\phi, a')}{N_{\text{actions}}}$$

where $\theta = \{\xi, \eta, \psi\}$ is the concatenation of all parameters.

**Why subtract the mean advantage?** Without the subtraction, the value-advantage decomposition would be unidentifiable: adding a constant $c$ to all advantages and subtracting $c$ from the value produces identical Q-values. The mean subtraction forces the advantages to have zero mean for each state, which identifies the decomposition: $v(s)$ now represents the average Q-value across actions, and $a(s, a)$ represents the deviation of action $a$ from that average. This constraint also provides a useful inductive bias: when actions have similar values (the common case), advantages are near zero, and the value stream captures most of the learning signal, generalizing across actions. When actions differ substantially, the advantage stream captures the action-specific differences.

**Adaptation for distributional outputs.** In Rainbow, the dueling architecture must be adapted for distributional RL, where each action outputs a probability distribution over $N_{\text{atoms}}$ atoms rather than a single scalar. The modification is straightforward but worth detailing:

- The shared encoder $f_{\xi}(s)$ remains unchanged, producing the same representation $\phi$.
- The value stream $v_{\eta}(\phi)$ now outputs a vector of length $N_{\text{atoms}}$, representing the distribution of state values: $v^i_{\eta}(\phi)$ is the logit for atom $z^i$ in the value distribution.
- The advantage stream $a_{\psi}(\phi, a)$ now outputs a matrix of size $N_{\text{atoms}} \times N_{\text{actions}}$: $a^i_{\psi}(\phi, a)$ is the logit for atom $z^i$ in the advantage for action $a$.
- The aggregation is performed per-atom before applying softmax, ensuring the resulting per-action distributions are properly normalised:

$$p^i_{\theta}(s, a) = \frac{\exp\left( v^i_{\eta}(\phi) + a^i_{\psi}(\phi, a) - \bar{a}^i_{\psi}(\phi) \right)}{\sum_j \exp\left( v^j_{\eta}(\phi) + a^j_{\psi}(\phi, a) - \bar{a}^j_{\psi}(\phi) \right)}$$

where $\bar{a}^i_{\psi}(\phi) = \frac{1}{N_{\text{actions}}} \sum_{a'} a^i_{\psi}(\phi, a')$ is the mean advantage for atom $i$ across all actions.

**What this computes operationally.** For each atom $z^i$, the value stream provides a baseline logit, and the advantage stream provides an action-specific adjustment relative to the mean advantage. The adjusted logits are then softmax-normalised across atoms for each action independently, producing a valid probability distribution $p_{\theta}(s, a)$ for each action. The Q-value for action $a$ is the mean of this distribution: $q_{\theta}(s, a) = \sum_i z^i \cdot p^i_{\theta}(s, a)$.

**Why this architecture helps.** In states where the action choice doesn't matter (e.g., open space with no threats), the advantage stream learns to output near-zero logits for all actions, and the value stream alone determines the output distribution. The value stream receives gradient signals from all actions in these states, since all actions share the same value estimate—this is $N_{\text{actions}}$ times more data for learning the state value than a standard architecture would receive (which learns action values independently). In states where actions matter a lot, the advantage stream provides the action-specific differentiation. The dueling architecture thus adaptively allocates representational capacity: most parameters focus on learning general state properties (via the shared encoder and value stream), while a smaller capacity is reserved for action-specific differentiation (via the advantage stream).

---

#### Extension 4: Multi-Step Learning (Addressing Slow Reward Propagation)

**The problem: 1-step targets propagate reward information slowly.** The DQN target $R_{t+1} + \gamma \max_{a'} q(S_{t+1}, a')$ uses only the immediate reward, bootstrapping on the Q-value at the next state for all subsequent rewards. This means that when the agent encounters a reward at time $t + k$, the information about that reward must propagate backward through $k$ separate Bellman updates (each one step at a time) before it affects the Q-value at time $t$. In games with sparse or delayed rewards, this propagation is extremely slow.

**The solution: use n-step truncated returns.** Multi-step learning replaces the 1-step bootstrap target with an n-step truncated return, which combines $n$ observed rewards with the Q-value at the $n$-th future state:

$$R^{(n)}_t = \sum_{k=0}^{n-1} \gamma^{(k)}_t R_{t+k+1}$$

$$\mathcal{L}_{\text{n-step}}(\theta) = \left( R^{(n)}_t + \gamma^{(n)}_t \max_{a'} q_{\bar{\theta}}(S_{t+n}, a') - q_{\theta}(S_t, A_t) \right)^2$$

where $\gamma^{(k)}_t = \prod_{i=1}^k \gamma_{t+i}$ is the cumulative discount for $k$ steps (in episodic tasks, $\gamma_{t+i} = 0$ if $S_{t+i}$ is terminal, so the return truncates naturally at episode boundaries).

**What this computes operationally.** The agent sums the next $n$ actual rewards (discounted appropriately), then adds the discounted Q-value estimate at step $t+n$ as a proxy for all remaining future rewards. For $n = 1$, this recovers the standard 1-step DQN target. For $n = \infty$, this becomes a Monte Carlo return (summing rewards until episode termination, with no bootstrapping). Intermediate values of $n$ interpolate between these extremes.

**The bias-variance tradeoff.** Increasing $n$ reduces bias (because the bootstrap estimate contributes proportionally less to the target) but increases variance (because more actual stochastic rewards are summed). The optimal $n$ depends on the environment's stochasticity and the quality of the Q-value estimates.

**Rainbow's configuration.** Rainbow uses $n = 3$, selected by comparing values of 1, 3, and 5. The paper reports: "We observed that both $n = 3$ and 5 did well initially, but overall $n = 3$ performed the best by the end." This is a typical finding: moderate $n$ provides faster initial learning (reward propagation is 3× faster than 1-step), while very large $n$ introduces too much variance or becomes mismatched to the distributional support's fixed range.

**Why not eligibility traces?** Eligibility traces (TD($\lambda$)) provide a softer combination over all n-step returns with exponentially decaying weights. The paper notes this alternative in the Discussion (under the name "optimality tightening" and eligibility traces) but chose n-step returns as a simpler, well-understood baseline with a single tunable hyperparameter. The choice represents a pragmatic tradeoff between complexity and benefit.

---

#### Extension 5: Distributional Q-Learning (Learning Returns Distributions)

**The problem: the mean discards distributional information.** DQN learns $q(s, a) = \mathbb{E}[G_t | S_t = s, A_t = a]$, the expected discounted return. In deterministic environments, this is sufficient—the return is a single value. In stochastic environments, however, the return is a random variable with a potentially complex distribution (e.g., bimodal if an action sometimes succeeds brilliantly and sometimes fails catastrophically). The mean collapses this distribution to a single number, discarding information about variance, multimodality, and tail risk. This information could be useful for more robust decision-making and potentially provides a richer learning signal.

**The solution: learn the full return distribution.** Distributional RL (Bellemare, Dabney, and Munos, 2017) models the probability distribution of returns using a categorical distribution over a discrete support. The support $z$ is a fixed vector of $N_{\text{atoms}}$ evenly spaced values:

$$z^i = v_{\text{min}} + (i - 1) \frac{v_{\text{max}} - v_{\text{min}}}{N_{\text{atoms}} - 1}, \quad i \in \{1, \ldots, N_{\text{atoms}}\}$$

For Rainbow, $N_{\text{atoms}} = 51$, $v_{\text{min}} = -10$, and $v_{\text{max}} = 10$, meaning the support consists of 51 equally spaced points from -10 to 10 (inclusive).

For each state-action pair $(s, a)$, the network outputs a vector of $N_{\text{atoms}}$ logits, which are passed through a softmax (independently for each action) to produce a probability distribution:

$$p^i_{\theta}(s, a) = \frac{\exp(\text{logit}^i_{\theta}(s, a))}{\sum_j \exp(\text{logit}^j_{\theta}(s, a))}$$

The estimated distribution over returns for $(s, a)$ is then $d_t = (z, p_{\theta}(S_t, A_t))$, meaning: with probability $p^i_{\theta}(S_t, A_t)$, the return is $z^i$. The mean Q-value is recovered as $q_{\theta}(s, a) = z^{\top} p_{\theta}(s, a) = \sum_i z^i \cdot p^i_{\theta}(s, a)$.

**The distributional Bellman target.** The key insight is that return distributions satisfy a distributional variant of Bellman's equation. If the agent follows an optimal policy, the distribution of returns at $(s, a)$ should match the distribution obtained by: (1) taking the distribution of returns at the next state under the optimal action, (2) contracting it by $\gamma$ (multiplying all possible returns by the discount factor, which compresses the distribution toward zero), and (3) shifting it by the reward (adding the constant $R_{t+1}$, or in stochastic environments, adding the reward distribution).

For a given transition, the target distribution is constructed as:

$$d'_t \equiv (R_{t+1} + \gamma_{t+1} z, \; p_{\bar{\theta}}(S_{t+1}, a^*_{t+1}))$$

**What this notation means operationally.** Take the target network's predicted probability masses $p_{\bar{\theta}}(S_{t+1}, a^*_{t+1})$ over the support atoms for the optimal next action $a^*_{t+1}$. These masses are now associated not with the original atoms $z$, but with transformed atoms $R_{t+1} + \gamma_{t+1} z$. For example, if $\gamma_{t+1} = 0.99$ and $R_{t+1} = 1$, the atom that was at support value $z^i = 5$ is now at $1 + 0.99 \times 5 = 5.95$. This transformation shifts the support (by the reward) and contracts it (by the discount).

**The projection step.** The transformed support atoms $R_{t+1} + \gamma_{t+1} z$ generally fall between the fixed atoms of the original support $z$. The projection operator $\Phi_z$ maps each transformed atom onto the nearest original atoms, distributing its probability mass proportionally to the distances. Concretely, for a transformed atom at value $x$ that falls between original atoms $z^i$ and $z^{i+1}$, the probability mass $p$ on $x$ is split: $(z^{i+1} - x) / (z^{i+1} - z^i) \cdot p$ goes to atom $z^i$, and $(x - z^i) / (z^{i+1} - z^i) \cdot p$ goes to atom $z^{i+1}$. This ensures the target distribution is represented on the same fixed support as the prediction.

**The loss function.** The learning objective is the Kullback-Leibler divergence between the projected target distribution and the current prediction:

$$\mathcal{L}_{\text{dist}}(\theta) = D_{\text{KL}}(\Phi_z d'_t \| d_t) = \sum_i \Phi_z d'_t(i) \log \frac{\Phi_z d'_t(i)}{p^i_{\theta}(S_t, A_t)}$$

where the gradient is taken with respect to $\theta$ (the prediction parameters) only, treating $\Phi_z d'_t$ as a fixed target.

**What this computes operationally.** For each atom $i$, the KL divergence penalises the prediction $p^i_{\theta}$ for being away from the target probability $\Phi_z d'_t(i)$. Unlike the squared error in DQN (which only compares means), the KL divergence compares entire distributions, providing a richer learning signal. If the target distribution is highly concentrated (low variance) but the prediction is diffuse (high variance), the KL penalty is large even if the means match—the agent learns to represent uncertainty accurately.

**Why this form rather than alternatives.** Why categorical distributions with a fixed support rather than, say, quantile regression (Dabney et al., 2018, a later paper) or Gaussian approximations? The categorical representation is simple to implement (just softmax over $N_{\text{atoms}}$ outputs), compatible with standard neural network architectures, and the projection operator provides a principled way to handle the transformation of the support. The fixed support $[-10, 10]$ was chosen because Atari game returns, when using reward clipping (standard in DQN, where all rewards are clipped to $[-1, 1]$) and a discount of $\gamma = 0.99$, are bounded: the maximum possible return is approximately $1 / (1 - 0.99) = 100$, but with reward clipping and finite episodes, values rarely exceed 10 in practice. An alternative would be to learn the support parameters as well, but the fixed support simplifies optimization.

**Important design choice: why the constrained support matters for overestimation.** With $v_{\text{max}} = 10$ and $v_{\text{min}} = -10$, any return above 10 gets projected entirely onto the maximum atom $z^{51} = 10$, and any return below -10 gets projected onto the minimum atom $z^1 = -10$. This means the distributional agent's Q-values (the means of these distributions) can never exceed 10 or go below -10, regardless of how optimistic or pessimistic the network becomes. This implicit clipping provides a form of overestimation protection that is independent of double Q-learning—the network simply cannot output Q-values above 10. The paper explicitly connects this observation to the limited marginal benefit of double Q-learning in the ablation study: "We hypothesize that clipping the values to this constrained range counteracts the overestimation bias of Q-learning." I will return to this in the integration section.

---

#### Extension 6: Noisy Nets (Addressing Undirected Exploration)

**The problem: $\epsilon$-greedy exploration is state-independent and undirected.** DQN (and its variants) typically explore by selecting a random action with probability $\epsilon$, and the greedy action otherwise. The $\epsilon$ is annealed from 1.0 to some small final value (0.1 or 0.01) over the first few million frames. This exploration strategy has two major limitations. First, it is **state-independent**: the agent explores with the same probability regardless of how uncertain it is about the current state. In well-explored regions, random actions waste time; in novel regions, the fixed exploration rate might be insufficient. Second, it is **undirected**: random actions are uniformly distributed across all possible actions, making it astronomically unlikely to discover reward sequences that require specific, coordinated action sequences (e.g., retrieving a key to open a door in Montezuma's Revenge).

**The solution: learned, state-conditional exploration through noisy networks.** Noisy Nets (Fortunato et al., 2017) replace standard linear layers with noisy layers that inject learnable stochasticity directly into the network weights:

$$y = (b + Wx) + (b_{\text{noisy}} \odot \epsilon^b + (W_{\text{noisy}} \odot \epsilon^w)x)$$

where:
- $b$ and $W$ are the standard (deterministic) bias and weight parameters,
- $b_{\text{noisy}}$ and $W_{\text{noisy}}$ are learnable parameters controlling the scale of the noise,
- $\epsilon^b$ and $\epsilon^w$ are zero-mean random variables with fixed statistics (factorised Gaussian noise in Rainbow's implementation),
- $\odot$ denotes element-wise (Hadamard) product.

This is a noisy linear transformation: the output $y$ combines a deterministic component $b + Wx$ (which processes the input normally) with a noisy component $b_{\text{noisy}} \odot \epsilon^b + (W_{\text{noisy}} \odot \epsilon^w)x$ (which adds state-dependent randomness). The noise variables $\epsilon^b$ and $\epsilon^w$ are sampled independently for each forward pass, so the network produces different outputs for the same input on different passes.

**Factorised Gaussian noise.** Rainbow uses factorised Gaussian noise to reduce the number of independent noise variables. For a layer with $p$ inputs and $q$ outputs, independent noise would require $p \times q$ noise variables for the weights plus $q$ for the biases—which is expensive. Factorised noise instead generates $p + q$ independent Gaussian variables and constructs the weight noise matrix as the outer product:

$$\epsilon^w_{i,j} = f(\epsilon_i) f(\epsilon_j)$$

where $f(x) = \text{sgn}(x) \sqrt{|x|}$ is a function that preserves the sign and square-roots the magnitude. This reduces the number of independent noise variables from $O(pq)$ to $O(p + q)$ while maintaining reasonable expressivity for exploration.

**Learning to modulate exploration.** The key property of Noisy Nets is that the noise scale parameters $b_{\text{noisy}}$ and $W_{\text{noisy}}$ are learned by gradient descent along with all other network parameters. The agent can therefore learn to reduce the noise in parts of the state space where exploration is no longer beneficial (by driving these parameters toward zero) and maintain noise where uncertainty remains. This implements **state-conditional exploration**: the same network processes the state and determines both what action to take (through the deterministic stream) and how much to explore (through the noisy stream). Over training, the network typically learns to "self-anneal" its exploration by reducing the noise scale, without requiring an explicit $\epsilon$ annealing schedule.

**Interaction with action selection.** When using Noisy Nets, the agent acts **fully greedily** with respect to the network's (stochastic) output: $A_t = \arg\max_a q_{\theta}(S_t, a)$, where $q_{\theta}(S_t, a) = z^{\top} p_{\theta}(S_t, a)$ is the mean Q-value computed from the current noisy forward pass. There is no external $\epsilon$—all exploration comes from the noise injected into the weights. During evaluation, the noise is disabled (or averaged over multiple forward passes), and the agent acts deterministically.

**Rainbow's configuration.** Rainbow uses $\sigma_0 = 0.5$ as the initialisation parameter for the noisy stream weights, and acts with $\epsilon = 0$ (pure greedy action selection with respect to the noisy Q-values). The paper notes: "The noise was generated on the GPU. Tensorflow noise generation can be unreliable on GPU. If generating the noise on the CPU, lowering $\sigma_0$ to 0.1 may be helpful."

**Why Noisy Nets over alternative exploration methods.** Compared to $\epsilon$-greedy, Noisy Nets provide directed, state-conditional exploration: the network learns where to explore rather than exploring uniformly. Compared to other sophisticated exploration methods like Bootstrapped DQN (Osband et al., 2016) or count-based exploration (Bellemare et al., 2016), Noisy Nets are simpler to implement (just replace linear layers with noisy variants), add minimal computational overhead, and integrate naturally with any value-based architecture. The paper acknowledges that combining Noisy Nets with other exploration methods is "fruitful subject for further research."

---

#### Integration Challenges: Making the Six Components Work Together

The individual extensions were designed for (and tested with) standard scalar-output DQN. Integrating all six into a single agent requires resolving several non-trivial design decisions where the components' interfaces must be adapted.

**Integration choice 1: Multi-step distributional loss.** The distributional RL loss (Equation 3 in the original formulation) uses a 1-step target $d'_t = (R_{t+1} + \gamma_{t+1} z, p_{\bar{\theta}}(S_{t+1}, a^*_{t+1}))$. Rainbow extends this to an n-step target. The target distribution is constructed by contracting the value distribution at $S_{t+n}$ according to the cumulative n-step discount and shifting it by the truncated n-step return:

$$d^{(n)}_t = \left( R^{(n)}_t + \gamma^{(n)}_t z, \; p_{\bar{\theta}}(S_{t+n}, a^*_{t+n}) \right)$$

where $R^{(n)}_t = \sum_{k=0}^{n-1} \gamma^{(k)}_t R_{t+k+1}$ is the truncated n-step return, $\gamma^{(n)}_t = \prod_{i=1}^n \gamma_{t+i}$ is the cumulative discount over $n$ steps, and $a^*_{t+n}$ is the optimal action at step $t+n$ (defined below with double Q-learning).

The loss is then:

$$\mathcal{L}_{\text{Rainbow}}(\theta) = D_{\text{KL}}\left( \Phi_z d^{(n)}_t \| d_t \right)$$

where $\Phi_z$ projects the shifted-and-scaled target distribution back onto the fixed support $z$, exactly as in the 1-step case.

**Operational meaning.** The agent computes $R^{(n)}_t$ by summing the next $n$ observed rewards. It then takes the target network's predicted distribution at $S_{t+n}$, shifts each support atom by $R^{(n)}_t$ (adding the summed rewards), contracts each shifted atom by $\gamma^{(n)}_t$ (applying the cumulative discount), projects the resulting distribution onto the fixed support, and minimises KL divergence from this target. This propagates reward information $n$ steps backward per update—the distributional analog of n-step TD learning.

**Integration choice 2: Double Q-learning for the bootstrap action.** In the distributional n-step target, which action $a^*_{t+n}$ should be used at the bootstrap state $S_{t+n}$? Rainbow applies double Q-learning: the online network $\theta$ selects the greedy action based on mean Q-values, and the target network $\bar{\theta}$ provides the distribution for that action:

$$a^*_{t+n} = \arg\max_a q_{\theta}(S_{t+n}, a) = \arg\max_a \left( z^{\top} p_{\theta}(S_{t+n}, a) \right)$$

$$d^{(n)}_t = \left( R^{(n)}_t + \gamma^{(n)}_t z, \; p_{\bar{\theta}}(S_{t+n}, a^*_{t+n}) \right)$$

This decouples action selection (by the online network) from distribution evaluation (by the target network), reducing the overestimation bias that would arise if the same network performed both operations.

**Integration choice 3: KL loss as priority for experience replay.** In standard prioritized replay, the priority is the absolute TD error. In the distributional multi-step setting, Rainbow uses the KL loss:

$$p_t \propto \left( D_{\text{KL}}(\Phi_z d^{(n)}_t \| d_t) \right)^{\omega}$$

with $\omega = 0.5$. The authors chose KL loss because "this is what the algorithm is minimizing," and note that "the KL loss as priority might be more robust to noisy stochastic environments because the loss can continue to decrease even when the returns are not deterministic." When using non-distributional Rainbow variants (in the ablation study), the priority falls back to the absolute TD error.

**Integration choice 4: Dueling architecture for distributional outputs.** As detailed in the Dueling networks section above, Rainbow adapts the dueling architecture to output per-atom logits rather than scalar Q-values. The value stream outputs $N_{\text{atoms}}$ values, the advantage stream outputs $N_{\text{atoms}} \times N_{\text{actions}}$ values, and the per-atom aggregation (value plus centered advantage) is followed by per-action softmax normalization. This ensures each action's output is a valid probability distribution over the support atoms.

**Integration choice 5: Noisy Nets applied to all linear layers.** Rainbow replaces every linear layer in the dueling distributional network with its noisy equivalent (Equation 4). This includes the linear layers in the shared encoder, the value stream, and the advantage stream. All noisy layers use factorised Gaussian noise. The agent acts greedily ($\epsilon = 0$) because exploration is handled internally by the noise.

**Integration choice 6: Replay buffer warmup period.** DQN typically waits 200K frames before starting learning, to populate the replay buffer with sufficiently diverse transitions. Rainbow reduces this to 80K frames: "We have found that, with prioritized replay, it is possible to start learning sooner, after only 80K frames." This is because prioritized replay biases sampling toward recent transitions (new transitions are inserted with maximum priority), so the effective buffer diversity is achieved faster.

**Integration choice 7: Optimizer and learning rate.** DQN uses RMSprop with learning rate $\alpha = 0.00025$. Rainbow switches to Adam (Kingma and Ba, 2014), with a reduced learning rate of $\alpha/4 = 0.0000625$, selected among $\{\alpha/2, \alpha/4, \alpha/6\}$. The authors note that "we found [Adam] less sensitive to the choice of the learning rate than RMSProp." Adam's $\epsilon$ hyperparameter is set to $1.5 \times 10^{-4}$.

**Summary of integration.** A single Rainbow update proceeds as follows:

1. Sample a mini-batch of transitions from the replay buffer, with sampling probability proportional to $(\text{KL loss})^{\omega}$ for each transition.
2. For each transition, compute the n-step return $R^{(n)}_t$ by summing the next $n$ rewards (using stored future transitions).
3. Select the bootstrap action using the online network: $a^*_{t+n} = \arg\max_a (z^{\top} p_{\theta}(S_{t+n}, a))$.
4. Construct the target distribution using the target network: $d^{(n)}_t = (R^{(n)}_t + \gamma^{(n)}_t z, p_{\bar{\theta}}(S_{t+n}, a^*_{t+n}))$.
5. Project the target distribution onto the fixed support: $\Phi_z d^{(n)}_t$.
6. Compute the per-transition loss: $D_{\text{KL}}(\Phi_z d^{(n)}_t \| d_t)$, where $d_t = (z, p_{\theta}(S_t, A_t))$ is the online network's prediction.
7. Multiply each transition's loss by its importance sampling weight $w_i = (1/N \cdot 1/P(i))^{\beta}$.
8. Average the weighted losses over the mini-batch and backpropagate gradients through the online network (including through the noisy layer parameters $b_{\text{noisy}}$ and $W_{\text{noisy}}$).
9. Update the priorities of the sampled transitions to the new KL loss values.
10. Every 32K frames, copy the online network parameters $\theta$ to the target network $\bar{\theta}$.

---

#### Hyperparameter Configuration and Experimental Setup

**Evaluation methodology.** All agents are evaluated on 57 Atari 2600 games from the Arcade Learning Environment (Bellemare et al., 2013). Performance is measured every 1M environment steps by suspending learning and running the latest agent for 500K evaluation frames. Episodes are truncated at 108K frames (30 minutes of simulated play). Scores are normalised per game such that 0% corresponds to a random agent and 100% to an average human expert. The primary aggregate metric is the **median human-normalised score** across all 57 games; the mean is less informative because it is dominated by a few games (e.g., Atlantis) where agents achieve scores orders of magnitude above human level.

**Two testing regimes.** At the end of training, the best agent snapshot is re-evaluated under two conditions:

- **No-ops starts:** A random number (up to 30) of no-op actions are inserted at the beginning of each episode. This tests robustness to initial state variation and is also the regime used during training.
- **Human starts:** Episodes are initialised with state sequences randomly sampled from the initial portion of human expert trajectories (Nair et al., 2015). This tests whether the agent has overfit to its own trajectories by measuring performance on state distributions it has not encountered during training.

The gap between no-ops and human starts scores indicates the degree of overfitting.

**Hyperparameter summary (Table 1).** The paper provides a complete hyperparameter table, reproduced here with values verbatim:

| Parameter | Value |
|---|---|
| Min history to start learning | 80K frames |
| Adam learning rate | 0.0000625 |
| Exploration $\epsilon$ | 0.0 |
| Noisy Nets $\sigma_0$ | 0.5 |
| Target Network Period | 32K frames |
| Adam $\epsilon$ | $1.5 \times 10^{-4}$ |
| Prioritization type | proportional |
| Prioritization exponent $\omega$ | 0.5 |
| Prioritization importance sampling $\beta$ | 0.4 → 1.0 |
| Multi-step returns $n$ | 3 |
| Distributional atoms | 51 |
| Distributional min/max values | $[-10, 10]$ |

These hyperparameters are identical across all 57 games—Rainbow is a single agent configuration with no per-game tuning.

**Hyperparameter tuning process.** The combinatorial space of all six components' hyperparameters is too large for exhaustive search. The authors performed "limited tuning" using manual coordinate descent: starting from the values used in each component's original paper, they tuned "the most sensitive among hyper-parameters" one at a time.

Specific tuning decisions documented in the paper:

- **Learning rate:** Tested $\{\alpha/2, \alpha/4, \alpha/6\}$ relative to DQN's $\alpha = 0.00025$, selected $\alpha/4 = 0.0000625$.
- **Multi-step $n$:** Tested $\{1, 3, 5\}$. $n = 3$ performed best overall, though $n = 3$ and $n = 5$ "both did well initially."
- **Prioritization exponent $\omega$:** Tested $\{0.4, 0.5, 0.7\}$. When using KL loss as priority, "performance is very robust to the choice of $\omega$."
- **Replay warmup:** Reduced from DQN's 200K to 80K frames, enabled by prioritized replay's bias toward recent transitions.
- **Exploration schedule (for non-Noisy Nets variants):** When Noisy Nets are removed in ablations, $\epsilon$ is annealed to 0.01 in the first 250K frames (much faster than DQN's anneal to 0.1 over 4M frames).

**Why Adam over RMSprop?** The authors report that Adam was "less sensitive to the choice of the learning rate," which is practically important when combining many components: each component might shift the effective loss landscape, making a fixed learning rate that works for one combination fail for another. Adam's adaptive per-parameter learning rates provide more robustness to such shifts.

**Why $\beta$ annealing from 0.4 to 1.0?** The importance sampling correction trades off bias correction against variance: full correction ($\beta = 1$) yields unbiased gradient estimates but can introduce high variance in the importance weights. Early in training, the biased estimates (from $\beta < 1$) are acceptable because the optimal policy is far away and precise gradients matter less; near convergence, unbiased updates become more important. The linear anneal from 0.4 to 1.0 over the full 200M frame training run balances this tradeoff, following the original prioritized replay paper's recommendation.

**Computational cost.** Rainbow runs on a single GPU. The 7M frames needed to match DQN's final performance correspond to "less than 10 hours of wall-clock time." A full 200M frame run takes "approximately 10 days," with "less than 20%" variation between all discussed variants. The paper explicitly focuses on algorithmic variations rather than parallelisation, leaving scalability questions to future work.

---

#### Design Rationale: Why This Approach Over Alternatives

**Why combine rather than propose a new method?** The paper could have introduced a new algorithm that incorporates insights from all six extensions in a more tightly integrated way. Instead, it chose to explicitly combine existing methods, making the provenance of each component clear and enabling ablation studies that isolate each component's contribution. This is scientifically valuable because it answers a question the community had ("are these complementary?") rather than introducing yet another point solution. It is practically valuable because practitioners can understand exactly what each component does and make informed decisions about which to include in their own implementations.

**Why these six extensions and not others?** The paper acknowledges the selection is not exhaustive ("This list is, of course, far from exhaustive"). The criteria appear to have been: (1) each addresses a fundamentally different limitation of DQN (overestimation, data efficiency, representation, reward propagation, return modeling, exploration), reducing overlap; (2) each had demonstrated clear standalone improvements; (3) the set was "manageable" in size. The Discussion section catalogs numerous omitted extensions (Bootstrapped DQN, count-based exploration, episodic control, auxiliary tasks, recurrent architectures, hierarchical RL, policy gradient hybrids) and frames Rainbow as a foundation for even more comprehensive integration.

**Why manual coordinate descent rather than systematic hyperparameter search?** The combinatorial hyperparameter space is large (each component has 1–3 sensitive parameters), and full Atari training runs take 10 days on a single GPU. An exhaustive grid search would be computationally infeasible. Manual coordinate descent—tuning one hyperparameter at a time while holding others fixed—is a pragmatic compromise that likely finds reasonable (if not globally optimal) hyperparameter settings. The authors acknowledge this limitation implicitly by providing the complete hyperparameter table, enabling others to reproduce and potentially improve upon their tuning.

## 4. Key Insights and Innovations

### Innovation 1: The Integration-Ablation Paradigm as a Scientific Method for Compound Systems

The Rainbow paper's most distinctive intellectual contribution is not the specific algorithm it produces but the **research methodology it establishes**: systematically combine independently developed improvements, then measure each component's marginal contribution through ablation in the full combined context. This represents a fundamental shift in how the field evaluates algorithmic progress in compound systems.

**What the field did before Rainbow.** The dominant pattern in deep RL research circa 2017 was the isolated-baseline comparison: new technique X is developed, compared against a standard baseline (usually vanilla DQN), and shown to improve performance. Each technique's paper demonstrated gains in isolation, but these gains were measured against a weak, increasingly obsolete baseline. A practitioner wanting to build the best possible system had no principled way to determine whether technique A (shown to help over DQN) still provides benefit when techniques B through F are already in place. The field was accumulating improvements without understanding their interactions—a classic case of **compositional uncertainty**.

Partial combinations existed (Prioritized DDQN, Dueling DDQN with prioritized replay), but these were ad hoc. No one had attempted a full integration or, critically, measured what happens when you *remove* a component from the full system. The difference between "does A help over DQN?" and "does A help when B through F are already present?" is not subtle—it is the difference between knowing a drug works in isolation and knowing it works in combination with other drugs, where interactions can be synergistic, redundant, or antagonistic.

**Why this is a methodological innovation, not just an engineering feat.** The paper converts a question of engineering ("can we make these things work together?") into a question of scientific diagnosis ("what does each component contribute, and what does that tell us about the underlying learning dynamics?"). The ablation study (Figures 2–4) is not merely a report card—it is a **differential diagnosis** of the DQN algorithm's remaining bottlenecks. The finding that prioritized replay and multi-step learning remain crucial in the full combination while dueling networks and double Q-learning become marginal is not just a performance ranking; it reveals something about *why* the original DQN struggled and which fixes are fundamental versus symptomatic.

Consider: double Q-learning was introduced to fix overestimation bias, a well-documented pathology of Q-learning (van Hasselt, 2010; van Hasselt, Guez, and Silver, 2016). Its marginal contribution in Rainbow is small because the distributional support $[-10, 10]$ implicitly clips Q-values, providing independent overestimation protection. This tells us that overestimation bias was a *symptom* of unbounded value estimates, not a fundamental flaw requiring a dedicated algorithmic fix—a different mechanism (value clipping via distributional constraints) addresses the same root cause more effectively. This kind of mechanistic insight only emerges when you can observe a component becoming redundant in the presence of others, which requires the full integration-plus-ablation design.

**The paradigm's lasting impact.** Subsequent work in deep RL has adopted this pattern extensively. The R2D2 paper (Kapturowski et al., 2019) integrated recurrent architectures with distributed prioritized replay; the Agent57 paper (Badia et al., 2020) combined an even larger set of components (including episodic memory, intrinsic motivation, and a meta-controller) with systematic ablation. Both explicitly follow the Rainbow template: combine, then remove to diagnose. This paradigm has arguably been more influential than any single algorithmic component Rainbow introduced, because it changed *how the field evaluates progress*—from isolated comparisons against weak baselines toward understanding interactions in compound systems.

**Tying to evidence.** The ablation results in Figure 3 (median performance curves for Rainbow minus each component) and Figure 4 (per-game performance drops) are the empirical manifestation of this methodology. Figure 4 is particularly informative: it shows that prioritized replay and multi-step learning help almost uniformly (53 out of 57 games each), while dueling and double Q-learning show mixed effects (helping on some games, hurting on others). This per-game breakdown would be impossible to obtain without the full integration as a baseline—you cannot measure the marginal value of dueling networks on Frostbite unless you have a working Rainbow agent to remove them from. The methodology enables a granularity of diagnosis that isolated comparisons cannot provide.

**A nuance on significance.** This is fundamentally a **methodological innovation**, not a theoretical one. It doesn't introduce new mathematics, new proof techniques, or new formal frameworks. However, for a field where empirical progress often outpaces theoretical understanding, methodological innovations that improve our ability to *diagnose why things work* can be as impactful as theoretical ones—they guide future research toward the most productive directions. The finding that prioritized replay and multi-step learning dominate suggests future work should focus on data efficiency and reward propagation; the finding that dueling networks contribute little in the full combination suggests this architectural innovation has been largely superseded.

---

### Innovation 2: The Empirical Discovery That Addressed "Limitations" Are Not Independent—Some Are Fundamental, Others Incidental

The paper's second major contribution is an **empirical taxonomy of DQN's limitations**, revealed by which fixes remain essential in the full combination and which become redundant. This moves beyond the individual papers' claims ("X improves DQN") to a systems-level understanding of which bottlenecks are fundamental constraints on deep Q-learning and which are merely symptoms of a particular (now-replaced) configuration.

**The taxonomy that emerges.** The ablation results partition the six components into three tiers, each with a different mechanistic interpretation:

**Tier 1: Fundamental bottlenecks—prioritized replay and multi-step learning.** These components address limitations that no other component compensates for. Prioritized replay governs *which data the agent learns from*—replacing uniform sampling with error-proportional sampling. Multi-step learning governs *how quickly reward information propagates*—replacing 1-step bootstrap targets with n-step returns. No other Rainbow component touches either of these functions: distributional RL changes the learning objective but not the sampling distribution; dueling networks change the architecture but not the temporal difference target's horizon; Noisy Nets change exploration but not how past experience is consumed. The near-uniform game-level benefit (53/57 games, Figure 4) confirms that these are **independent, non-substitutable improvements**—there is no redundancy because no other mechanism addresses data selection or reward propagation speed.

**Tier 2: Partially substitutable improvements—distributional RL and Noisy Nets.** These components provide clear aggregate benefit (Figures 2–3) but their contributions are partially overlapping with other mechanisms. Distributional RL provides the richest learning signal (the full return distribution versus just the mean), which helps particularly at later stages of training—the distributional ablation matches Rainbow for the first 40M frames before diverging (Figure 3). This suggests distributional RL primarily aids *asymptotic performance*, not early learning speed—it helps the agent squeeze out remaining improvements after the easy gains are exhausted. The breakdown by human-performance thresholds (Figure 2) supports this: the distributional ablation lags primarily on games where performance is above 200% of human level, i.e., games where the agent has already learned competent policies and is refining them.

Noisy Nets provide state-conditional exploration versus $\epsilon$-greedy's undirected random actions. The aggregate benefit is clear (Figure 3, red dashed line below Rainbow), but the per-game pattern (Figure 4) shows both large improvements on some games and small regressions on others. This mixed pattern is characteristic of exploration mechanisms: the right exploration strategy depends on the game's structure, and no single method dominates universally.

**Tier 3: Components subsumed by other mechanisms—dueling networks and double Q-learning.** These components show the most interesting pattern: small or negligible aggregate benefit, with game-dependent effects that can be positive or negative. The interpretation is not that these techniques are useless—they demonstrably help in isolation—but that their function is **largely provided by other components in the full Rainbow**.

For double Q-learning, the paper explicitly identifies the substitution mechanism (Section "Analysis"): "the actual returns are often higher than 10 and therefore fall outside the support of the distribution, spanning from -10 to +10. This leads to underestimated returns, rather than overestimations. We hypothesize that clipping the values to this constrained range counteracts the overestimation bias of Q-learning." In other words, distributional RL's fixed support $[v_{\text{min}}, v_{\text{max}}] = [-10, 10]$ implicitly prevents unbounded overestimation—Q-values literally cannot exceed 10, even if the network's logits would otherwise produce higher values. Double Q-learning's decoupled action selection and evaluation addresses the same overestimation problem but does so through a different mechanism (reducing correlation between selection and evaluation errors). When the distributional constraint is already capping values, the additional benefit of decoupling is marginal.

For dueling networks, the substitution is less clearly identified but plausibly involves distributional RL providing a richer representation that reduces the need for explicit value-advantage factorization. Distributional RL already forces the network to represent the full return distribution (51 atoms), which implicitly encodes information about state quality and action differentiation in a distributed way. The explicit separation into value and advantage streams may add less when the network is already learning a rich distributional representation.

**Why this taxonomy matters beyond Rainbow.** This finding provides a **scalpel for future research**: it tells the community which aspects of deep Q-learning are genuinely unsolved problems versus which have been solved by multiple, partially redundant approaches. Research effort should focus on Tier 1 problems (data efficiency, reward propagation) where no existing mechanism provides relief, rather than Tier 3 problems (overestimation bias, value-advantage representation) where existing solutions already work well in combination. This is a form of **research prioritisation**—the paper doesn't just say "Rainbow works well," it says "here's what still matters and what doesn't," guiding the next generation of work toward the highest-impact directions.

**The distinction between fundamental and incidental.** A limitation of DQN is **fundamental** if it persists in the optimal combination of known fixes; it is **incidental** if it is resolved (perhaps unintentionally) by improvements targeting other limitations. Overestimation bias turns out to be largely incidental—it disappears when you constrain value estimates, which distributional RL does for a different reason (to keep the categorical distribution's support manageable). Data inefficiency, in contrast, is fundamental—even with all other improvements, prioritized replay provides a unique and non-substitutable benefit. This distinction is practically actionable: a researcher wanting to improve deep Q-learning further should invest in better replay prioritisation or faster reward propagation, not in better overestimation correction.

**Tying to evidence.** The per-game ablation heatmap (Figure 4) provides the most granular evidence for this taxonomy. The columns for prioritization and multi-step are overwhelmingly blue (Rainbow outperforms the ablation), while the columns for dueling and double Q-learning show mixed colors (sometimes blue, sometimes red). The aggregate learning curves (Figure 3) show the temporal dimension: prioritization and multi-step ablations lag from the beginning, while the distributional ablation only diverges after 40M frames, and the dueling/double ablations are nearly indistinguishable from full Rainbow at many points.

---

### Innovation 3: Reframing Exploration from an External Schedule to a Learned, State-Conditional Process

Noisy Nets represent a conceptual shift in how exploration is framed in value-based deep RL, and Rainbow's integration of Noisy Nets with five other components validates that **learned, state-conditional exploration is compatible with (and preferable to) the dominant $\epsilon$-greedy paradigm** in a full-scale system.

**The prior paradigm: exploration as an external perturbation.** DQN and its early variants treated exploration as an **external mechanism** added to an otherwise deterministic policy: with probability $\epsilon$, act randomly; otherwise, act greedily. The $\epsilon$ schedule is a global hyperparameter, annealed from 1.0 to some final value over a fixed number of frames, identical across all states and all games. This framing has deep roots—$\epsilon$-greedy exploration dates back to the earliest RL work (Watkins, 1989)—but it treats exploration as a **nuisance to be reduced over time** rather than a **skill to be learned**. The agent doesn't learn to explore; it merely explores less as training progresses, according to a predetermined schedule that knows nothing about the agent's actual uncertainty.

**The Noisy Nets reframing: exploration as learned network stochasticity.** Noisy Nets make exploration an **internal, learned property of the network**: the noise is injected into the weights, and the noise scale parameters are trained by gradient descent alongside the value function parameters. This means the network can learn to modulate its own exploration—reducing noise in well-understood parts of the state space while maintaining it in novel or uncertain regions. There is no external $\epsilon$ parameter and no annealing schedule; the network "self-anneals" by learning to drive the noisy stream parameters toward zero where exploration is no longer beneficial.

This reframes exploration from a **scheduling problem** (when and how fast should $\epsilon$ decay?) to a **representation learning problem** (can the network learn state representations that encode uncertainty?). The network's ability to reduce noise in familiar states and maintain it in unfamiliar ones means exploration becomes **state-conditional** by construction, without requiring explicit uncertainty estimates, visitation counts, or exploration bonuses.

**Rainbow's validation of this framing at scale.** Prior to Rainbow, Noisy Nets had been demonstrated as a standalone improvement over DQN (Fortunato et al., 2017). But it was unknown whether learned stochastic exploration would interact gracefully with distributional RL (which changes the value representation), prioritized replay (which changes the data distribution), and multi-step returns (which change the reward signal). The fact that Noisy Nets provide clear benefit *in the presence of all five other components* (Figure 3, red vs. rainbow curves) validates that learned exploration is not made redundant by other improvements—it addresses a genuinely orthogonal aspect of agent behavior that no other component touches.

**The deeper implication: exploration and exploitation are not separate phases.** The standard DQN recipe separates exploration and exploitation temporally: first explore (large $\epsilon$, decaying over time), then exploit (small fixed $\epsilon$). This corresponds to thinking of exploration as something you *stop doing* once you've learned enough. Noisy Nets, by contrast, maintain exploration throughout training but let the network decide *where* to explore. Some parts of the state space may be fully exploited (near-zero noise) even early in training, while others may remain exploratory late in training. This aligns more naturally with how exploration works in biological learning—you remain curious about uncertain things regardless of how much total experience you've accumulated.

**The mixed per-game results as a feature, not a bug.** The fact that Noisy Nets help substantially on some games but marginally hurt others (Figure 4) is informative: it suggests that $\epsilon$-greedy is actually well-suited to some game structures (perhaps those with dense rewards where random actions quickly encounter learning signals) and poorly suited to others (sparse-reward games where directed exploration matters). The mixed pattern is what you'd expect from an exploration mechanism that *adapts to the environment*—it can't be simultaneously optimal for all environments because optimal exploration strategies are environment-dependent. The fact that the aggregate benefit is positive (Figure 3) while per-game effects vary (Figure 4) is consistent with learned exploration finding better strategies on average but occasionally being outperformed by simpler baselines on specific games.

**Tying to evidence.** The aggregate comparison in Figure 3 (Noisy Nets removed, red dashed line) shows a consistent gap below full Rainbow, confirming the positive contribution. The per-game breakdown in Figure 4 shows the variance: Noisy Nets removal causes large drops on some games and small gains on others, with the gains concentrated in a minority of titles. The overall benefit in median human-normalised performance (visible in Figure 1's final scores) validates that state-conditional learned exploration scales to the full Atari benchmark, which was not guaranteed when Noisy Nets were first introduced.

---

### Innovation 4: The Diagnostic Value of Difficulty-Stratified Performance Analysis

Although the paper does not frame it as a primary contribution, its use of **human-performance thresholds** to stratify results (Figure 2) represents a significant methodological advance in how Atari benchmark results are interpreted. This innovation has been underappreciated in subsequent literature but is arguably one of the paper's most practically useful analytic tools.

**The prior approach: aggregate curves hide distributional shifts.** The standard Atari benchmark report—both before and after Rainbow—relies on median human-normalised performance curves (like Figure 1). The median tells you whether the "typical" game improves, but it cannot distinguish between two very different scenarios: (A) the agent is getting much better on games where it was already competent, pushing already-superhuman performance even higher, versus (B) the agent is making progress on previously unsolved games, expanding the set of games where it achieves reasonable performance. For a practitioner interested in whether a new technique helps on hard games or merely inflates scores on easy ones, the median is uninformative.

**The Rainbow stratification: counting games above performance thresholds.** Figure 2 plots, for each agent, the number of games where human-normalised performance exceeds a given threshold (20%, 50%, 100%, 200%, 500%) as a function of training frames. This decomposes the aggregate improvement into its distributional components:

- At the **20% threshold** (leftmost panel): how many games have the agent made *any* meaningful progress on? This captures progress on the hardest games, where even reaching 20% of human performance represents a significant achievement.
- At the **100% threshold** (center panel): how many games reach human-level performance? This is a natural milestone for "solved."
- At the **500% threshold** (rightmost panel): how many games achieve superhuman performance by a wide margin? This captures progress on games where the agent was already strong, now pushing to extreme levels.

**What this reveals about Rainbow that the median hides.** The top row of Figure 2 shows that Rainbow's improvement over baselines is visible at *all* thresholds—it is not merely inflating already-high scores on easy games. The gap between Rainbow and the best baseline is present at the 20% level (more games with any meaningful performance), at the 100% level (more games reaching human parity), and at the 500% level (more games achieving extreme superhuman scores). This is a much stronger claim than "the median improved": it says Rainbow is simultaneously making hard games less hard and easy games even better, which implies the improvements are not merely shifting probability mass within an existing capability envelope but genuinely expanding that envelope in multiple directions.

The bottom row of Figure 2 applies the same analysis to the ablations, revealing *where* each component contributes. The distributional ablation's deficit is most visible at the 200% and 500% thresholds (rightmost panels)—confirming that distributional RL primarily helps on games where the agent is already strong, refining already-competent policies to superhuman levels. The multi-step ablation shows deficits even at the 20% threshold—confirming that multi-step returns help on the hardest games, where reward propagation speed is a bottleneck. This differential diagnosis would be invisible in a median-only analysis.

**Why this is an innovation in benchmark analysis, not just a visualization.** The threshold-counting approach transforms the Atari benchmark from a **single-dimensional performance metric** (median score) into a **multi-dimensional capability profile**. It acknowledges that "performance on Atari" is not a single quantity—it's a distribution over 57 diverse environments with radically different reward structures, difficulty levels, and required skills. A technique that helps on 40 easy games but hurts on 17 hard ones might improve the median while being net-harmful for the games most in need of progress. Conversely, a technique that makes modest gains on 10 hard games while slightly regressing on 47 easy ones might lower the median while being more scientifically important (showing progress on previously unsolved challenges). The threshold analysis makes these distinctions visible.

**A note on adoption.** This specific innovation has been less widely adopted than the integration-ablation methodology, perhaps because counting above thresholds is less visually compact than a single learning curve. However, it is conceptually important: it demonstrates that aggregate metrics in heterogeneous benchmarks can be misleading and that distributional analysis of performance (not to be confused with distributional RL—here I mean analysing the *distribution of scores across tasks*) is essential for understanding *how* an algorithm improves, not just *how much*.

**Tying to evidence.** Figure 2 is the sole locus of this analysis in the paper, but its impact on interpreting the results is substantial. Without it, the claim that Rainbow "improves on games where baseline agents were already good, as well as improving in games where baseline agents are still far from human performance" (quoted from the Analysis section) would be an assertion backed only by the median curve. With Figure 2, it becomes a directly verifiable empirical statement: the rainbow-colored curve is above the baseline curves at every threshold.

## 5. Experimental Analysis

### Evaluation Methodology

- **Dataset.** All experiments use the full suite of 57 Atari 2600 games from the Arcade Learning Environment (Bellemare et al., 2013), following the standard training and evaluation procedures established by Mnih et al. (2015) and van Hasselt et al. (2016). This is the canonical benchmark for deep RL at the time, spanning diverse game mechanics, reward structures, and difficulty levels.

- **Base model(s).** The base architecture is a convolutional neural network processing stacks of raw pixel frames, identical in structure to DQN (Mnih et al., 2015). The paper does not introduce a new base model—Rainbow is an algorithmic combination applied to the same underlying network architecture used by all six extensions it integrates. The key architectural modification is replacing all linear layers with noisy equivalents and restructuring the output head for distributional dueling outputs, but the convolutional encoder and overall depth remain consistent with prior work.

- **Metrics.** The primary metric is **median human-normalised score** across all 57 games. Per-game scores are normalised such that 0% corresponds to a random agent and 100% corresponds to an average human expert: $\text{normalised} = 100 \times (\text{score}_{\text{agent}} - \text{score}_{\text{random}}) / (\text{score}_{\text{human}} - \text{score}_{\text{random}})$. The median is preferred over the mean because the mean is "dominated by a few games (e.g., Atlantis) where agents achieve scores orders of magnitude higher than humans do." Additionally, the paper tracks the **number of games above human-performance thresholds** (20%, 50%, 100%, 200%, 500%) to decompose where improvements originate. Final evaluation also reports **median scores under two testing regimes** (no-ops starts and human starts) after training concludes.

- **Baselines.** The paper compares Rainbow against seven published agents, representing both the original DQN and each extension in its standalone (or partially combined) form: **DQN** (Mnih et al., 2015), **Double DQN** (van Hasselt, Guez, and Silver, 2016), **Prioritized DDQN** (Schaul et al., 2015—prioritized replay combined with double Q-learning), **Dueling DDQN** (Wang et al., 2016—dueling architecture combined with double Q-learning), **Distributional DQN** (Bellemare, Dabney, and Munos, 2017), **Noisy DQN** (Fortunato et al., 2017), and **A3C** (Mnih et al., 2016—an actor-critic method included for broader algorithmic context). Learning curves for Dueling DDQN and Prioritized DDQN were provided by their original authors; DQN, DDQN, Distributional DQN, Noisy DQN, and A3C were re-run by the Rainbow authors.

- **Generation budget / compute accounting.** All comparisons are on a single GPU with "less than 20%" variation in wall-clock time between variants. The paper explicitly focuses on **algorithmic data efficiency** (performance as a function of environment frames) rather than wall-clock speed, noting that parallelisation methods like Gorila (Nair et al., 2015) or A3C can trade data efficiency for wall-clock time. A full 200M frame run takes approximately 10 days. The 7M frames required to match DQN's final performance correspond to less than 10 hours of wall-clock time.

- **Cross-validation / statistical protocol.** There is no cross-validation or statistical significance testing reported. The standard Atari evaluation protocol is deterministic: agents are evaluated every 1M training frames by suspending learning and running the current policy for 500K evaluation frames. Final performance is assessed by taking the best agent snapshot (selected by evaluation scores during training) and running two testing regimes: no-ops starts (random number up to 30 no-op actions at episode start, consistent with training) and human starts (episodes initialised from human expert trajectory starting points, per Nair et al., 2015). Episodes are truncated at 108K frames (30 minutes of simulated play). All hyperparameters are identical across all 57 games—no per-game tuning.

### Main Quantitative Results

#### Aggregate Performance Against Published Baselines

The headline result appears in Figure 1 and Table 2: **Rainbow achieves a median human-normalised score of 231% in the no-ops regime and 153% in the human starts regime**, substantially exceeding all published baselines. The best individual baseline is Distributional DQN at 185% (no-ops) and 125% (human starts). DQN itself achieves only 79% and 68% respectively, meaning Rainbow improves over DQN by approximately 2.9× in no-ops and 2.25× in human starts.

The learning curves in Figure 1 establish two temporal milestones:

- **Data efficiency**: Rainbow matches DQN's final performance (79% median) after only 7M frames—compared to DQN's 200M frames to reach that level. This represents roughly a 28.6× improvement in sample efficiency to reach DQN-level performance.
- **Final performance**: Rainbow surpasses the best final performance of all baselines within 44M frames and continues improving substantially through 200M frames.

The paper reports this as: "we match DQN's best performance after 7M frames, surpass any baseline in 44M frames, reaching substantially improved final performance."

Table 2 provides the exact final median scores for all agents in both testing regimes. Key comparisons (no-ops):

| Agent | Median Score |
|---|---|
| DQN | 79% |
| DDQN (*) | 117% |
| Prioritized DDQN (*) | 140% |
| Dueling DDQN (*) | 151% |
| A3C (*) | — (116% human starts only) |
| Noisy DQN | 118% |
| Distributional DQN | 185% |
| **Rainbow** | **231%** |

Asterisks indicate scores taken from the corresponding publications; others are from the Rainbow authors' own re-implementations.

#### Threshold-Stratified Performance Analysis (Figure 2, Top Row)

The top row of Figure 2 decomposes the aggregate improvement by counting how many games achieve at least 20%, 50%, 100%, 200%, and 500% of human-normalised performance, plotted as a function of training frames. This analysis reveals that Rainbow's superiority is present at **all performance levels**:

- At the **20% threshold** (games where the agent has made any meaningful progress), Rainbow's curve is above all baselines from early in training and maintains or widens the gap throughout.
- At the **100% threshold** (games reaching human-level performance), Rainbow consistently leads, reaching more games at human-level than any baseline.
- At the **200% and 500% thresholds** (games where the agent substantially exceeds human performance), Rainbow shows progressively larger advantages—the gap between Rainbow and baselines widens at higher thresholds, indicating that Rainbow's improvements compound most strongly on games where the agent is already competent.

The paper summarises: "the Rainbow agent is improving scores on games where the baseline agents were already good, as well as improving in games where baseline agents are still far from human performance." This rejects the hypothesis that Rainbow merely inflates already-high scores on easy games; it simultaneously pushes the frontier on hard games (raising the floor) and achieves extreme superhuman performance on games where baselines were already strong (raising the ceiling).

#### Ablation Study: Aggregate Impact of Removing Each Component (Figure 3)

Figure 3 plots median human-normalised performance for Rainbow versus six ablated variants, each removing exactly one component from the full combination. The results establish a clear importance ranking:

**Tier 1 (largest drops when removed): Prioritized replay and multi-step learning.** Removing either component causes "a large drop in median performance" (quoted directly from the Analysis section). The multi-step ablation (dashed orange line) lags Rainbow from the very beginning of training and the gap persists through 200M frames, indicating multi-step returns help both early and late learning. Removing multi-step learning "also hurt final performance"—unexpectedly, since multi-step returns are primarily motivated as a data-efficiency improvement for early learning. The prioritized replay ablation (dashed dark blue) shows a similar large and persistent gap.

**Tier 2 (moderate drops): Distributional RL and Noisy Nets.** The distributional ablation (dashed yellow line) shows a distinctive pattern: for the first ~40M frames, it performs as well as full Rainbow, but then begins to lag and the gap widens through 200M frames. This temporal pattern suggests distributional RL primarily aids **asymptotic performance** rather than early learning speed. The Noisy Nets ablation (dashed red line) shows a consistent but more modest gap throughout training, indicating learned exploration provides steady benefit but is not as critical as data selection or reward propagation.

**Tier 3 (minimal aggregate impact): Dueling networks and Double Q-learning.** The dueling ablation (dashed green line) and double Q-learning ablation (dashed purple line) show "limited" difference in aggregate median performance, with curves that largely overlap full Rainbow. The paper notes: "in the case of double Q-learning, the observed difference in median performance (Figure 3) is limited, with the component sometimes harming or helping depending on the game."

#### Per-Game Ablation Analysis (Figure 4 and Figure 2, Bottom Row)

Figure 4 provides the most granular diagnostic: a game-by-game breakdown of the performance difference between Rainbow and each ablation, averaged over the full learning run. The results reveal that Tier 3 components show **game-dependent effects**—sometimes helping, sometimes hurting—while Tier 1 components help almost uniformly:

- **Prioritized replay** causes a performance drop in 53 of 57 games when removed. Only 4 games show Rainbow performing worse than the prioritized-ablated variant.
- **Multi-step learning** similarly causes a drop in 53 of 57 games when removed.
- **Distributional RL** and **Noisy Nets** show larger but still predominantly positive effects, with some games showing small regressions.
- **Dueling networks** and **Double Q-learning** show the most mixed patterns—roughly balanced between games where they help and games where they hurt, with the magnitude of effects being smaller than for Tier 1 components.

The bottom row of Figure 2 stratifies the ablation results by human-performance thresholds, revealing *where* each component's contribution manifests:

- The **multi-step ablation** shows deficits at the 20% and 50% thresholds—multi-step returns help on the hardest games where reward propagation speed is a bottleneck for even basic progress.
- The **distributional ablation** shows its largest deficits at the 200% and 500% thresholds—distributional RL primarily helps on games where the agent is already competent, refining policies to extreme superhuman levels.
- The **dueling ablation** shows possible improvement on games above 200% human performance but possible degradation on games below 100%, though the effects are small and "the median score hides the fact that the impact of Dueling differed between games."

#### Final Evaluation: No-Ops vs. Human Starts

The final evaluation after training (Table 2) compares the two testing regimes. Rainbow achieves 231% median in no-ops starts versus 153% in human starts—a gap of 78 percentage points. This gap is consistent with the pattern seen across all baselines: DQN drops from 79% to 68% (11 point gap), Distributional DQN drops from 185% to 125% (60 point gap). The human-starts regime is more challenging because it tests the agent on state distributions generated by human play rather than the agent's own trajectories. The paper does not comment extensively on the no-ops/human-starts gap, noting only that it "indicates the extent to which the agent has over-fit to its own trajectories."

#### Double Q-Learning Diagnosis: Implicit Overestimation Protection from Distributional Constraints

In the Analysis section, the paper reports a specific diagnostic experiment to understand why double Q-learning provides limited benefit:

> "To further investigate the role of double Q-learning, we compared the predictions of our trained agents to the actual discounted returns computed from clipped rewards. Comparing Rainbow to the agent where double Q-learning was ablated, we observed that the actual returns are often higher than 10 and therefore fall outside the support of the distribution, spanning from -10 to +10. This leads to underestimated returns, rather than overestimations."

The interpretation is that the distributional support $[-10, 10]$ imposes a hard cap on Q-values: when actual returns exceed 10, the agent's predictions are forced to underestimate them (since the maximum atom is at 10). This underestimation counteracts the typical overestimation bias of Q-learning, making double Q-learning's decoupled action selection and evaluation partially redundant. The paper adds: "We hypothesize that clipping the values to this constrained range counteracts the overestimation bias of Q-learning. Note, however, that the importance of double Q-learning may increase if the support of the distributions is expanded."

#### Wall-Clock Time and Computational Cost

The paper reports that the 7M frames required to match DQN's final performance correspond to "less than 10 hours of wall-clock time" on a single GPU. A full 200M frame run takes "approximately 10 days." The variation in wall-clock time between all discussed variants is "less than 20%," confirming that the algorithmic modifications (noisy layers, dueling architecture, distributional outputs) do not substantially increase per-step computation. This is noteworthy because some of these modifications add parameters (dueling's two-stream architecture, distributional RL's 51-atom output per action, Noisy Nets' additional noise parameters), but the computational cost of the forward and backward passes remains dominated by the convolutional encoder.

### Ablation Studies and Robustness Checks

**Removal of prioritized replay (Figure 3, dashed dark blue; Figure 4, "Prioritization" column):** Removing prioritized replay causes a large drop in median performance, with the ablation lagging Rainbow from the start and the gap persisting through 200M frames. Per Figure 4, the drop occurs in 53 out of 57 games, making it one of the two most uniformly beneficial components. This ablation uses uniform sampling from the replay buffer instead of KL-loss-proportional sampling with importance weights.

**Removal of multi-step learning (Figure 3, dashed orange; Figure 4, "Multi-step" column):** Removing multi-step returns (reverting to $n = 1$) causes a large and persistent drop, also in 53 of 57 games per Figure 4. The paper explicitly notes that this "also hurt final performance," which is unexpected given that multi-step returns are primarily justified as accelerating early learning through faster reward propagation. The continued benefit at 200M frames suggests that n-step returns provide a favourable bias-variance tradeoff that helps even near convergence.

**Removal of distributional RL (Figure 3, dashed yellow; Figure 4, "Distributional" column):** This ablation replaces the categorical distribution over 51 atoms with a standard scalar Q-value output (mean squared error loss). The distinctive finding is temporal: for the first ~40M frames, the distributional ablation performs as well as full Rainbow—the curves overlap almost exactly. After ~40M frames, the ablation begins to lag, and the gap widens through 200M frames. The threshold-stratified analysis (Figure 2, bottom row) shows the deficit is concentrated at the 200% and 500% thresholds, indicating distributional RL primarily helps on games where the agent has already learned competent policies and is refining them to superhuman levels. The paper notes this pattern but does not provide a mechanistic explanation beyond the general benefit of learning richer distributional representations.

**Removal of Noisy Nets (Figure 3, dashed red; Figure 4, "Noisy Nets" column):** Removing Noisy Nets and reverting to $\epsilon$-greedy exploration (with $\epsilon$ annealed to 0.01 in the first 250K frames) causes a moderate drop in aggregate performance. The per-game pattern (Figure 4) is mixed: Noisy Nets provide large improvements on some games but small regressions on others. This is consistent with exploration mechanisms being environment-dependent—state-conditional learned exploration helps on average but can underperform $\epsilon$-greedy on specific games where random exploration is sufficient.

**Removal of dueling networks (Figure 3, dashed green; Figure 4, "Dueling" column):** Removing the dueling architecture (reverting to a standard Q-network head without value-advantage factorization) shows "no significant difference" in aggregate median performance. However, Figure 4 reveals game-dependent effects: dueling helps on some games (particularly those with above-human performance levels, per Figure 2 bottom row) and hurts on others. The paper's interpretation is that dueling's benefit is partially subsumed by other components, possibly because distributional RL's richer output representation reduces the need for explicit value-advantage factorization.

**Removal of double Q-learning (Figure 3, dashed purple; Figure 4, "Double" column):** Removing double Q-learning (reverting to standard Q-learning where the same target network both selects and evaluates the bootstrap action) shows "limited" aggregate impact, with the ablation curve largely overlapping Rainbow. The dedicated diagnostic analysis (comparing predicted returns to actual discounted returns) reveals the mechanism: the distributional support ceiling at $v_{\text{max}} = 10$ causes underestimation of returns that exceed 10, which counteracts the overestimation bias that double Q-learning is designed to address. The paper explicitly notes this as a hypothesis: "clipping the values to this constrained range counteracts the overestimation bias of Q-learning."

**KL loss as priority metric (discussed in "The Integrated Agent" section, not plotted separately):** The paper reports that using KL loss rather than absolute TD error as the priority for experience replay makes performance "very robust to the choice of $\omega$," the prioritization exponent. Values of $\omega \in \{0.4, 0.5, 0.7\}$ were tested, with 0.5 selected. The authors argue KL-based priority is "more robust to noisy stochastic environments because the loss can continue to decrease even when the returns are not deterministic"—a claim based on the intuition that stochastic transitions can maintain high TD error even after the agent has learned the correct distribution, while KL loss would correctly decrease. No controlled experiment comparing KL-based vs. TD-error-based prioritization in the full Rainbow is reported.

**Multi-step n comparison (discussed in "Experimental Methods"):** The paper compared $n = 1$, $3$, and $5$ for the multi-step return horizon. The finding: "$n = 3$ and $5$ did well initially, but overall $n = 3$ performed the best by the end." This suggests that $n = 5$ may provide too much variance or become mismatched to the distributional support's fixed range as training progresses, while $n = 1$ propagates rewards too slowly. No learning curves for different $n$ values are shown, limiting the reader's ability to assess how sensitive performance is to this hyperparameter.

**Reduced replay warmup (80K vs. 200K frames):** The paper reduces DQN's standard 200K-frame warmup (during which no learning occurs) to 80K frames, noting: "We have found that, with prioritized replay, it is possible to start learning sooner, after only 80K frames." The justification is that prioritized replay biases sampling toward recent transitions (new transitions get maximum priority), so the effective diversity required for stable learning is achieved faster. No ablation comparing different warmup durations is shown.

**Adam vs. RMSprop optimizer:** The paper switches from DQN's RMSprop to Adam, with a reduced learning rate of $0.0000625$ ($\alpha/4$ relative to DQN's $\alpha = 0.00025$). The learning rate was selected among $\{\alpha/2, \alpha/4, \alpha/6\}$. The justification: "we found [Adam] less sensitive to the choice of the learning rate than RMSProp." No direct comparison of Adam vs. RMSprop performance in the full Rainbow is shown.

**Factorised Gaussian noise (implicit design choice):** Rainbow uses factorised Gaussian noise (reducing independent noise variables from $O(pq)$ to $O(p+q)$ per layer) rather than the independent Gaussian noise variant also described in Fortunato et al. (2017). No comparison between factorised and independent noise is reported for Rainbow, though the choice is noted as reducing computational overhead.

**$\beta$ annealing from 0.4 to 1.0:** The importance sampling correction exponent is linearly annealed from 0.4 to 1.0 over the full 200M training frames, following the original prioritized replay paper. No ablation on alternative annealing schedules or fixed $\beta$ values is reported.

### Critical Assessment

**Central claim: Six independently developed DQN extensions are complementary and can be fruitfully combined into a single agent that achieves state-of-the-art performance.**

The experimental evidence supports this claim convincingly for the aggregate performance metric. The median human-normalised score of 231% (Table 2) substantially exceeds every published baseline, and the learning curves (Figure 1) show Rainbow maintaining or widening its advantage throughout training. The "fruitfully combined" claim is supported by the ablation study (Figure 3), which shows that removing any single component generally reduces performance—if the components were redundant or antagonistic, we would expect some ablations to match or exceed full Rainbow. The fact that no ablation outperforms Rainbow in aggregate (all dashed lines in Figure 3 are at or below the solid rainbow line for most of training) demonstrates that the combination is not merely the sum of its parts but genuinely compounds benefits.

However, the "complementary" claim requires nuance that the aggregate results partially obscure. The ablation study reveals a spectrum of complementarity, not a uniform pattern. Prioritized replay and multi-step learning are strongly complementary—they address genuinely orthogonal bottlenecks (data selection and reward propagation speed) that no other component touches, and their removal causes large, uniform drops across games. Distributional RL and Noisy Nets are moderately complementary—they provide clear aggregate benefit but their contributions overlap partially with other mechanisms or are environment-dependent. Dueling networks and double Q-learning are weakly complementary at best—their aggregate contribution is minimal, and their per-game effects are mixed (helping on some games, hurting on others). Calling all six "complementary" without this stratification overstates the uniformity of the findings.

The paper's own language is appropriately cautious: the abstract says the ablation study "shows the contribution of each component," not that all components are equally important. The Analysis section acknowledges differential impact, noting that double Q-learning and dueling had "limited" aggregate effect while prioritization and multi-step were "most crucial." This honesty strengthens the paper's credibility but means the headline message—"combining six improvements works"—should be understood as "combining six improvements works, and the combination reveals that two of them dominate while two are largely redundant in the full context."

**Claim: Rainbow matches DQN's final performance in 7M frames (data efficiency) and surpasses all baselines in 44M frames.**

The 7M-frame claim is supported by Figure 1, where the rainbow curve crosses DQN's final performance level (horizontal dashed line at ~79% median) at approximately 7M frames. However, this comparison is against DQN's *final* performance after 200M frames of training—not against a DQN variant trained with Rainbow's hyperparameters (Adam optimizer, reduced learning rate, 80K warmup, KL-based priorities). Some of Rainbow's data efficiency advantage may come from these hyperparameter improvements rather than the six algorithmic extensions. A fairer comparison would be: DQN with Rainbow's optimizer, learning rate, and replay configuration (but without any of the six extensions). This ablation is not reported.

The 44M-frame claim is supported by Figure 1, where Rainbow's curve rises above the best baseline (Distributional DQN, reaching ~185% final) at approximately 44M frames. This is a clean comparison because Distributional DQN represents the strongest individual extension. The paper could have strengthened this claim by reporting the exact median scores at 7M and 44M frames for both Rainbow and the relevant baselines in a table, rather than requiring readers to estimate from the learning curves.

**Claim: The contribution of each component is quantified through ablation.**

This claim is well-supported by Figures 2, 3, and 4, which provide three complementary views: aggregate learning curves (temporal dynamics), per-game performance drops (distributional analysis across environments), and threshold-stratified analysis (where each component matters most). The methodology is sound—removing one component at a time from the full combination measures marginal contribution in context—and the results are informative.

The primary limitation is the **absence of interaction ablations**. Removing one component at a time measures the marginal value of that component *given all others present*, but it does not reveal whether synergies or redundancies exist between specific pairs. For example: does double Q-learning help more when distributional RL is absent (consistent with the hypothesis that distributional support clipping provides implicit overestimation protection)? Would dueling networks matter more without multi-step learning (since both affect the structure of the value representation)? These questions require removing *pairs* of components, which was not done. The paper's conclusion that "the importance of double Q-learning may increase if the support of the distributions is expanded" is speculative without an experiment testing double Q-learning in a non-distributional variant of Rainbow.

A second limitation: the ablation study is conducted by **removing components from Rainbow**, not by **adding components to a simpler baseline**. This means we never see, for instance, DQN + prioritized replay + multi-step learning (the two most crucial components) and how much the remaining four components add on top of that. This baseline would be informative because it would quantify the marginal benefit of the Tier 2 and Tier 3 components *given the Tier 1 components alone*. If DQN + prioritization + multi-step already achieves 90% of Rainbow's final performance, the practical message ("implement these two things first, the rest are optional") would be stronger and more actionable than the current presentation.

**Missing experiments and baseline comparisons.**

Several experiments would have strengthened the paper considerably:

1. **DQN with Rainbow's hyperparameters and infrastructure.** Some of Rainbow's gains may come from the switch to Adam, the reduced learning rate, the shorter warmup period, or the faster $\epsilon$ annealing schedule (for non-Noisy Nets variants). An apples-to-apples hyperparameter comparison would isolate the algorithmic contributions from engineering improvements. This is particularly relevant for the 7M-frame data efficiency claim.

2. **Ablation of the distributional support range.** The finding that double Q-learning is partially redundant because the support $[-10, 10]$ clips values suggests a direct experiment: what happens to the double Q-learning ablation's performance if the support is expanded (e.g., to $[-20, 20]$ or $[-50, 50]$)? The paper predicts that "the importance of double Q-learning may increase"—testing this would convert a hypothesis into evidence.

3. **Scalar vs. distributional prioritization comparison.** The paper reports that KL-based prioritization makes performance "very robust to the choice of $\omega$," but never directly compares TD-error-based vs. KL-based prioritization in the full Rainbow setting. This comparison would validate the claim about robustness and resolve whether the priority metric choice matters independently of the distributional RL component.

4. **Interaction between dueling and distributional RL.** The hypothesis that distributional RL reduces the need for dueling networks (by providing richer representations) suggests a direct test: does the dueling ablation show a larger drop in the *non-distributional* variant of Rainbow? If so, this would confirm that these two architectural innovations partially substitute for each other.

5. **Noisy Nets with expanded support.** The mixed per-game results for Noisy Nets suggest that state-conditional exploration is beneficial on average but not universally. Reporting which game categories benefit most (e.g., sparse-reward exploration-heavy games vs. dense-reward games) would provide actionable guidance for practitioners. A correlation analysis between Noisy Nets' per-game benefit and game characteristics (sparsity, stochasticity, action space size) is absent.

6. **Confidence intervals or statistical measures.** The paper reports no confidence intervals, standard errors, or significance tests for any result. For the per-game analysis in Figure 4—where the claim that some components help on 53/57 games depends on counting—the variability of these counts across random seeds or evaluation runs is unknown. The 500K-frame evaluation runs provide substantial data per game, but without error bars on the median curves or per-game differences, readers cannot assess whether small differences (e.g., between the dueling ablation and full Rainbow) are statistically reliable or within noise.

**Generalization limitations.**

All results are on Atari 2600 games with a single network architecture (convolutional encoder with noisy dueling head), a single optimizer (Adam), and a single hyperparameter set applied uniformly across all 57 games. The paper makes no claim about generalization to other domains (continuous control, robotics, non-vision environments), other network architectures (recurrent, transformer-based), or other base algorithms (policy gradient, actor-critic). The Discussion section explicitly lists many promising directions not included in Rainbow, framing the paper as a foundation rather than a universal solution. This scoping is honest and appropriate, but it means the findings should be understood as specific to value-based deep RL on discrete-action vision-based tasks.

A subtle but important limitation: **the ablation results are contingent on the specific hyperparameters of the full Rainbow**. If the full Rainbow's hyperparameters were jointly optimised for the six-component combination, removing a component might hurt not because the component is inherently valuable, but because the remaining hyperparameters are suboptimal for the five-component variant. For example, if the prioritization exponent $\omega = 0.5$ was chosen because it works well with KL-based priorities, the prioritized replay ablation (which falls back to TD-error-based priorities) might perform worse partly because $\omega = 0.5$ is suboptimal for non-distributional prioritization. The manual coordinate descent tuning process likely mitigates this to some extent, but the problem is inherent to ablation studies in compound systems: you cannot hold all else equal when the "all else" was optimised for the full system.

**Summary of evidence sufficiency.**

The central claim—that the six extensions can be successfully combined and that the combination achieves state-of-the-art performance—is well-supported by the aggregate results in Figure 1 and Table 2. The ablation study (Figures 2–4) provides credible evidence that most components contribute positively in aggregate, with the important caveat that two components (dueling networks and double Q-learning) show minimal aggregate benefit and game-dependent effects. The data-efficiency claims are supported but confounded with hyperparameter improvements. The paper successfully demonstrates *that* the combination works; it provides partial but incomplete evidence for *why* each component matters and *how* they interact. The absence of pairwise interaction ablations, confidence intervals, and alternative hyperparameter configurations leaves room for future work to refine the understanding of which components are genuinely complementary versus merely not-harmful in the full combination.

## 6. Limitations and Trade-offs

### 6.1 The Distributional Value Support Creates an Implicit Value Clipping That Suppresses Overestimation — But Arbitrarily

Rainbow inherits from the distributional RL component a fixed support for return distributions spanning $[-10, 10]$, discretised into 51 atoms. This range is not a principled choice tied to Atari game properties; it is a pragmatic default that "happens to work" across the benchmark. The paper discovers—but does not design—a critical side effect: because the maximum possible Q-value output by the network is 10 (the mean of a distribution whose maximum atom is at 10), any true return exceeding 10 is systematically underestimated.

**The consequence.** The paper's own diagnostic analysis (Section "Analysis") reveals:

> "the actual returns are often higher than 10 and therefore fall outside the support of the distribution... This leads to underestimated returns, rather than overestimations."

This underestimation partially counteracts the overestimation bias that double Q-learning was designed to fix, which explains why removing double Q-learning from Rainbow has "limited" aggregate impact (Figure 3). But this is an accidental interaction, not a designed solution. It means Rainbow's performance depends on an implicit value cap that truncates genuinely achievable returns. On games where optimal play yields returns substantially above 10—for instance, games with dense positive rewards and long episodes—the agent's Q-values will saturate at the support ceiling, potentially distorting policy rankings between actions whose true values differ above 10 but appear identical when both are capped. The paper does not explore whether this ceiling limits asymptotic performance on high-scoring games.

**What evidence exists in the paper.** The paper reports the comparison between predicted returns and actual discounted returns in the Analysis section (not in a dedicated figure or table), noting that actual returns "often" exceed 10 and that this "leads to underestimated returns." Figure 4 shows that double Q-learning's removal has mixed per-game effects, consistent with the hypothesis that the distributional support provides partial overestimation protection on some games but not others. However, there is no ablation varying the support range to directly test this mechanism—no experiment showing what happens to Rainbow's performance (or to the double Q-learning ablation gap) when $[v_{\text{min}}, v_{\text{max}}]$ is expanded to $[-20, 20]$ or $[-50, 50]$.

**Mitigation status.** The paper partially acknowledges this limitation by noting that "the importance of double Q-learning may increase if the support of the distributions is expanded" (Analysis section). This is a forward-looking suggestion but not a demonstrated fix. No expanded-support experiment is conducted, leaving unanswered whether the current support range is near-optimal, unnecessarily restrictive, or a serendipitous sweet spot. The choice of $[-10, 10]$ thus remains an unexplained and untested hyperparameter that influences both the value representation and the effective overestimation characteristics of the algorithm—without the paper quantifying how sensitive final performance is to this choice.

---

### 6.2 The Ablation Study Measures Marginal Contributions in the Full Combination Only — Not How Much Each Component Adds to a Minimal Baseline

The paper's ablation methodology removes exactly one component at a time from the full six-component Rainbow agent. This measures each component's marginal contribution *conditional on the other five being present*. It does not measure what would happen if you started from a simpler baseline and *added* components incrementally—which is the decision most practitioners face.

**The consequence.** A reader wanting to build a competitive Atari agent learns that prioritized replay and multi-step learning are "the two most crucial components" and that removing either causes large drops in performance (Figure 3). But they do not learn the answer to a more actionable question: *how much of Rainbow's total improvement over DQN is captured by just these two components alone?* It is possible that a DQN variant with only prioritized replay and multi-step learning (plus the improved optimizer and hyperparameters) achieves, say, 90% of Rainbow's final median score—in which case the remaining four components collectively contribute only 10% on top, and a practitioner could stop implementing after the first two. Conversely, it might be that prioritized replay and multi-step learning provide substantial gains but leave large room for the other four components to add complementary value. The paper's ablation design cannot distinguish between these scenarios.

This limitation matters for research prioritisation as well. The finding that dueling networks and double Q-learning have "limited" aggregate impact in the full Rainbow (Figure 3) could mean either: (a) these components are genuinely unimportant—their function is subsumed by the other four components—or (b) they provide the same benefit as the other components but with a different mechanism, so their marginal value is small *only when the other four are already present*, and they would be valuable additions to a simpler baseline. The paper's conclusion that dueling and double Q-learning are less important is valid only under interpretation (a); the experiments do not rule out interpretation (b).

**What evidence exists in the paper.** None. The ablation study starts from Rainbow and removes components; it never starts from a simpler baseline and adds components. Figure 3 shows the gap between Rainbow and each ablation—not the gap between, say, DQN + prioritization + multi-step and full Rainbow. The paper does not report any incremental addition experiments.

**Mitigation status.** The paper does not acknowledge this as a limitation, and proposes no mitigation. The methodology is standard for combination-ablation studies but leaves the incremental-contribution question unanswered. A follow-up study could address this by constructing a "minimal Rainbow" (DQN with the improved optimizer and hyperparameters, plus prioritized replay and multi-step learning) and measuring how much each of the remaining four components adds incrementally. Until such experiments exist, the practical implication "implement prioritized replay and multi-step learning first, the rest are optional" remains plausible but unverified.

---

### 6.3 All Results Are on a Single Benchmark (Atari 2600) with a Single Model Architecture and a Single Base Algorithm Family

Rainbow is evaluated exclusively on the 57 Atari 2600 games from the Arcade Learning Environment, using a convolutional neural network architecture processing pixel inputs, built on the DQN (Q-learning) algorithm family. The paper makes no claims about generalization to other problem domains, input modalities, network architectures, or base RL algorithms—and provides no evidence that the complementarity patterns observed on Atari would transfer elsewhere.

**The consequence.** The paper's headline finding—that six DQN extensions are complementary and their combination yields state-of-the-art results—is established only within the narrow envelope of discrete-action vision-based game playing using Q-learning with function approximation. The following questions are left completely unanswered:

- Would the same six components compound similarly for policy-gradient methods (A3C, TRPO, PPO) or actor-critic architectures? The Discussion section mentions that "similar ideas may benefit also policy-based RL algorithms," but this is speculation, not evidence.
- Would the relative importance ranking change if the agent processed non-visual state representations (e.g., proprioceptive inputs in robotics, token embeddings in text-based environments)? The dueling architecture's benefit, for instance, depends on having many actions where the choice doesn't matter in most states—a property of Atari games that may not hold in other domains.
- Would the distributional range $[-10, 10]$ remain appropriate for environments with different reward scales, or would it need careful per-domain tuning—potentially requiring the kind of reward normalisation (e.g., Pop-Art) that the paper mentions in the Discussion but does not incorporate?

This matters for deployment decisions. A team working on continuous control or robotic manipulation cannot assume that Rainbow's component ranking applies to their domain; they would need to replicate a substantial portion of the ablation study—running dozens of 10-day experiments—to determine which components transfer and which are Atari-specific.

**What evidence exists in the paper.** None outside Atari. The Discussion section lists numerous extensions not included in Rainbow (policy gradient hybrids, hierarchical RL, auxiliary tasks, recurrent architectures) and frames them as "promising candidates for further experiments on integrated agents," implicitly acknowledging that the current study is limited to a specific algorithmic ecosystem. But no cross-domain experiments are reported or suggested.

**Mitigation status.** The paper acknowledges the scope limitation implicitly through its framing as a study of "extensions to the DQN algorithm" (abstract) and through the Discussion section's list of omitted directions. It does not claim Atari-specific findings generalise. However, it also does not explicitly flag the single-domain evaluation as a limitation that users should consider before applying the approach elsewhere. Given the paper's influence—it became a standard baseline for deep RL across domains—the absence of any note about domain transfer is a notable gap.

---

### 6.4 The Hyperparameter Tuning Budget Is Limited and the Sensitivity of the Ablation Results to Hyperparameter Choices Is Unknown

The paper uses manual coordinate descent (tuning one hyperparameter at a time while holding others fixed, starting from each component's original paper defaults) to configure Rainbow. The authors acknowledge:

> "The combinatorial space of hyper-parameters is too large for an exhaustive search, therefore we have performed limited tuning."

Specific values were selected from small candidate sets: learning rate from $\{\alpha/2, \alpha/4, \alpha/6\}$, multi-step $n$ from $\{1, 3, 5\}$, prioritization exponent $\omega$ from $\{0.4, 0.5, 0.7\}$. These choices were evaluated on the full 200M-frame Atari benchmark—meaning only a handful of full training runs informed the final configuration.

**The consequence.** The ablation study's conclusions depend critically on the assumption that the hyperparameters of the full Rainbow are reasonably optimised, and that removing a component does not change which hyperparameter values would be optimal for the remaining components. Both assumptions are questionable.

First, the full Rainbow may be suboptimally tuned. Coordinate descent finds local optima conditional on a particular ordering of tuning decisions and starting points; it does not explore interactions between hyperparameters. If some hyperparameter interacts strongly with a particular component—for example, if the optimal prioritization exponent $\omega$ depends on whether Noisy Nets are present, or if the optimal multi-step $n$ varies with the distributional support range—then coordinate descent could miss configurations that would substantially change the ablation results.

Second and more importantly: when a component is removed in the ablation study, the remaining hyperparameters are held at their Rainbow-optimised values. But the optimal values for a five-component variant may differ from those of the six-component full agent. For example, the learning rate 0.0000625 was chosen for the full Rainbow's loss landscape (which involves distributional KL loss, Noisy Nets' gradient contributions, and dueling architecture gradients). Removing distributional RL changes the loss to an MSE-based TD error; the optimal learning rate for this loss landscape might differ. If the ablated variant underperforms partly because it is stuck with suboptimal hyperparameters (tuned for the full system), the measured performance drop overstates the removed component's true marginal contribution.

**What evidence exists in the paper.** The paper reports robustness to some hyperparameter choices—for instance, "when using the KL loss of distributional DQN as priority, we have observed that performance is very robust to the choice of $\omega$." But robustness of individual hyperparameters does not address sensitivity of the *ablation conclusions* to joint hyperparameter shifts. The paper provides no experiment where an ablated variant is re-tuned—for instance, re-optimising the learning rate and exploration schedule for the dueling-ablated variant to see whether the gap closes.

**Mitigation status.** The paper does not discuss this limitation or propose mitigation. The standard defence for this type of study is that re-tuning every ablation would be computationally prohibitive (each Atari run takes ~10 GPU-days, and re-tuning six ablations on even a modest hyperparameter grid would require hundreds of runs). This is a genuine practical constraint, but it means the quantitative claims about ablation gaps should be interpreted as upper bounds on each component's marginal value—the true contribution could be smaller (if the ablation is mis-tuned) or larger (if full Rainbow is mis-tuned for that component's absence). The paper does not provide the caveat that ablation-inferred importance rankings are conditional on a single hyperparameter configuration.

---

### 6.5 The "Human Starts" Generalisation Gap Is Large and Its Composition Is Not Analysed

Rainbow achieves a median human-normalised score of 231% in the no-ops starts regime (where test episodes begin with a random number of no-op actions, consistent with training) but only 153% in the human starts regime (where episodes are initialised from human expert trajectory starting points, per Nair et al., 2015). This 78-percentage-point gap is the largest absolute drop among all baselines in Table 2—Distributional DQN drops by 60 points, Dueling DDQN by 34 points, and DQN by only 11 points. While Rainbow outperforms all baselines in both regimes, the *magnitude* of the gap raises questions about the nature of the performance being measured.

**The consequence.** The human starts regime tests whether the agent has learned policies that generalise to state distributions it did not generate during training. A large gap between no-ops and human starts scores indicates that the agent's performance is partially specific to its own trajectory distribution—it has learned to handle the states it tends to visit under its own exploratory policy, but stumbles on states arising from human play patterns. This is a form of **distributional overfitting**: the policy is specialised to the on-policy state distribution rather than being a generally competent game player.

The paper frames this gap as standard practice ("the difference between the two regimes indicates the extent to which the agent has over-fit to its own trajectories"), but the fact that Rainbow's absolute gap is larger than any baseline's raises the possibility that combining multiple components amplifies overfitting. Noisy Nets, for instance, learns state-conditional exploration that adapts to the agent's own state visitation distribution—potentially specialising the exploration (and thus the training distribution) even more tightly to the agent's own behaviour. Prioritized replay similarly skews the training distribution toward high-KL-loss transitions, which are a function of the agent's current policy and value estimates. Together, these mechanisms may create a self-reinforcing cycle where the agent trains on an increasingly policy-specific data distribution, excelling in no-ops evaluation (which matches its training distribution) but struggling with novel starting states.

**What evidence exists in the paper.** Table 2 reports the final scores in both regimes. The paper notes that the gap "indicates the extent to which the agent has over-fit to its own trajectories" but provides no further analysis—no per-game breakdown of the human-starts gap (analogous to the per-game ablation analysis in Figure 4), no comparison of which games show the largest drops between regimes, and no investigation of whether specific components (particularly Noisy Nets or prioritized replay) disproportionately contribute to the gap. Figure 2's threshold analysis is only shown for no-ops; an equivalent analysis for human starts would reveal whether the overfitting is concentrated in specific performance bands.

**Mitigation status.** The paper acknowledges the gap exists by reporting both scores but does not analyse or attempt to mitigate it. The Discussion section mentions that "exposing the real game to agents is a promising direction for future research" in the context of removing domain modifications like frame-stacking, reward clipping, and action repetition—but this is about environment fidelity, not about closing the no-ops-to-human-starts generalization gap. The gap remains an observed but unexplained phenomenon, and a practitioner deploying Rainbow-like agents should be aware that the headline 231% no-ops score substantially overstates performance under more realistic (human-like) starting conditions.

---

### 6.6 The Interaction Between Prioritized Replay and Multi-Step Learning Is Not Analysed — the Two "Most Crucial" Components May Overlap Significantly

The paper identifies prioritized replay and multi-step learning as the "two most crucial components" based on the large median-performance drops when either is removed (Figure 3). Both are described as addressing fundamentally different limitations: prioritized replay governs *which* data is learned from (data selection), while multi-step learning governs *how much* reward information is propagated per update (reward propagation speed). However, these mechanisms interact in non-obvious ways that the paper does not investigate—and their individual ablation drops may partially reflect the same underlying benefit.

**The consequence.** Multi-step returns make the TD target incorporate $n$ observed rewards, reducing reliance on bootstrapping from (potentially inaccurate) Q-value estimates. This changes the TD error distribution: multi-step targets typically have lower variance in the bootstrap component but higher variance in the reward sum, shifting the pattern of which transitions generate large errors. Prioritized replay, in turn, samples based on these errors. The two components may therefore interact: multi-step returns might make the priority signal more reliable (because the bootstrap error component is smaller), or conversely might concentrate priority on stochastic reward transitions (because multi-step returns amplify reward variance). Without analysing this interaction, the paper's claim that both are independently "crucial" rests on the assumption that their ablation drops are additive and reflect distinct mechanisms—an assumption that pairwise interaction ablations would be needed to verify.

A concrete scenario: suppose that adding multi-step returns improves the quality of the prioritization signal, making prioritized replay more effective. In this scenario, removing *either* component causes a large drop because the remaining component works less well without the other. Their large individual ablation drops would reflect a synergistic dependency, not independent importance. The paper provides no evidence to distinguish this from the interpretation that both are independently powerful.

**What evidence exists in the paper.** The per-game ablation analysis (Figure 4) shows that both components help on 53 of 57 games when removed individually—but this does not reveal whether they help on the *same* 53 games (because they address the same underlying bottleneck) or on *different* subsets of games (because they are genuinely orthogonal). The paper does not report a "remove both" ablation that would quantify their joint contribution, nor any correlation analysis between the per-game drops from removing prioritization versus removing multi-step learning.

**Mitigation status.** The paper does not acknowledge this as a limitation. The ablation methodology—removing one component at a time—is standard but cannot detect pairwise interactions or reveal whether the two most beneficial components are synergistic, additive, or partially redundant. A practitioner wondering whether to implement both or just one (if the other is difficult to implement in their infrastructure) gets no guidance from the current results beyond the observation that removing either individually hurts. The practical answer—whether both are worth the implementation cost versus picking the single best one—requires pairwise ablation data that the paper does not provide.

## 7. Implications and Future Directions

### How This Work Changes the Landscape

This paper changes how the deep reinforcement learning field evaluates algorithmic progress by establishing the **integration-ablation paradigm** as a standard methodology. Before Rainbow, the dominant pattern was isolated-baseline comparisons: a new technique is developed, compared against vanilla DQN, and shown to improve performance. Each technique's paper demonstrated gains in isolation, but against a weak, increasingly obsolete baseline. This created a landscape where practitioners accumulated improvements without knowing whether they were additive, redundant, or antagonistic—a classic case of **compositional uncertainty** where the field knew that A helps over DQN, B helps over DQN, and C helps over DQN, but had no principled way to determine whether A + B + C still helps or whether B's benefit is entirely subsumed by A.

Rainbow reframes the central question from "does technique X improve DQN?" to **"does technique X provide marginal benefit in the context of all other known improvements?"** This is a fundamentally different standard of evidence. It requires building the complete integrated system first, then demonstrating that removing X hurts—reversing the burden of proof from additive (start simple, add one thing) to subtractive (build the best system you can, then remove one thing at a time). The paper shows that this is computationally feasible even for 10-day Atari training runs, and that the resulting ablation analysis yields insights—about which components are fundamental versus incidental, which address symptoms versus root causes—that additive comparisons cannot provide.

The magnitude of this shift is **methodological rather than theoretical**: the paper does not introduce new mathematics, new proof techniques, or new formal frameworks. It changes *how the field evaluates empirical progress in compound systems*. This matters for a field where empirical results often outpace theoretical understanding: improving the quality of empirical diagnosis—knowing not just that something works but *why* it works in the context of everything else—is as practically important as introducing new algorithms. The lasting impact is visible in subsequent work: the Agent57 paper (Badia et al., 2020) explicitly follows the Rainbow template (integrate many components, ablate to diagnose), as does R2D2 (Kapturowski et al., 2019) and much of the distributed RL literature. Rainbow established that integration-plus-ablation is not merely possible but produces more informative science than isolated comparisons.

Beyond methodology, the paper provides a **conceptual reframing of DQN's limitations** by revealing which fixes are fundamental (their benefit persists in the full combination) versus incidental (their benefit is subsumed by other improvements targeting the same underlying bottleneck through different mechanisms). The finding that double Q-learning—the canonical fix for overestimation bias—becomes largely redundant when distributional RL constrains values to $[-10, 10]$ is not just a performance ranking; it tells us something mechanistic: overestimation bias was a *symptom* of unbounded value estimates, not a fundamental flaw requiring a dedicated algorithmic surgery. A different mechanism (value clipping via the distributional support) addresses the same root cause more effectively. Conversely, the finding that prioritized replay and multi-step learning remain crucial regardless of what else is present tells us that data selection efficiency and reward propagation speed are **irreducibly fundamental bottlenecks**—no other existing mechanism compensates for them.

This reframing has practical implications for research prioritisation. Before Rainbow, a researcher wanting to improve deep Q-learning faced a menu of six extensions, each backed by a paper showing improvement over DQN, with no guidance on which represented the highest-leverage investment. After Rainbow, the answer is clear: invest in better data selection (beyond proportional TD-error prioritization) and faster reward propagation (beyond fixed-n returns), rather than in better overestimation correction or architectural value-advantage factorization. The paper's ablation results serve as a **research roadmap**—not just a report card on Rainbow, but a diagnosis of where deep Q-learning's true bottlenecks remain.

The paper also **reconciles a tension** in how exploration is framed in value-based deep RL. Before Rainbow, $\epsilon$-greedy (with hand-tuned annealing schedules) and more sophisticated exploration mechanisms (Noisy Nets, Bootstrapped DQN, count-based exploration) coexisted without clarity on whether learned, state-conditional exploration was genuinely superior in full-scale systems or merely a boutique improvement on selected games. Rainbow demonstrates that Noisy Nets provide clear aggregate benefit in the presence of five other major improvements—learned exploration is not made redundant by better value representations, smarter replay, or faster reward propagation. This validates the conceptual shift from exploration as an **external nuisance to be scheduled away** (the $\epsilon$-greedy paradigm) to exploration as a **learned, state-conditional skill** (the Noisy Nets paradigm). The mixed per-game results (Figure 4) further refine this picture: they show that no single exploration strategy dominates universally, which is exactly what you would expect from a mechanism that adapts to environment structure—it can't be simultaneously optimal for all environments, but it improves the average case.

Finally, the paper introduces **difficulty-stratified benchmark analysis** (the human-performance threshold counting in Figure 2) that moves beyond the median-centric reporting standard for Atari. This is a quieter contribution but an important one: it demonstrates that aggregate metrics on heterogeneous benchmarks can hide distributional shifts in *where* improvements come from. A technique that inflates already-superhuman scores on easy games while leaving hard games untouched might improve the median without making scientific progress on the frontiers of what agents can learn. Rainbow's threshold analysis makes these distributional shifts visible and shows that Rainbow genuinely improves both the floor (more games reaching minimal competence) and the ceiling (more games achieving extreme superhuman performance). This analytic approach has been partially adopted in subsequent work but remains underutilised—the paper provides a template that more benchmarking efforts should follow.

---

### Follow-Up Research This Work Enables

**Directly testing the substitution hypothesis: distributional support as implicit overestimation protection.** The paper's most intriguing mechanistic finding is that the distributional support $[-10, 10]$ implicitly suppresses Q-value overestimation, making double Q-learning partially redundant. But this is a *discovery*, not a *designed experiment*—the paper observes the effect post-hoc and hypothesises the mechanism. A direct test would systematically vary $v_{\text{max}}$ (e.g., values of $\{5, 10, 20, 50, 100\}$) and measure: (a) the degree of overestimation (predicted Q-values minus true discounted returns) with and without double Q-learning at each support ceiling; (b) the performance gap between Rainbow and the double-Q-ablated variant at each ceiling. The paper predicts that "the importance of double Q-learning may increase if the support of the distributions is expanded." If confirmed, this would resolve whether the value clipping is a serendipitous hack or a genuinely desirable property—and would provide guidance for setting the distributional support in new domains. If the gap does *not* close with expanded support (i.e., double Q-learning remains unimportant even when values can exceed 10), that would suggest a different mechanism (perhaps distributional RL's richer learning signal inherently suppresses overestimation, independent of support range), which would be an even more interesting finding.

**Building a minimal Rainbow: what is the simplest agent that achieves 90% of Rainbow's performance?** The ablation study measures marginal contributions by removing one component from the full six-component system. The more actionable question for practitioners is: *starting from DQN, what is the minimum set of additions needed to capture most of Rainbow's gain?* A "minimal Rainbow" study would construct a sequence of agents—DQN → DQN + prioritized replay → + multi-step → + distributional → + Noisy Nets → + dueling → + double Q—and measure the cumulative performance at each step, using Rainbow's hyperparameters throughout (Adam optimizer, reduced learning rate, 80K warmup). This would reveal whether prioritized replay + multi-step alone captures, say, 80% of Rainbow's final median improvement over DQN (making the remaining four components collectively worth only 20% on top) or whether each addition provides substantial marginal gain. The current ablation results cannot answer this because removing one component from a six-component system is not the inverse of adding it to a simpler baseline—the marginal value of a component depends on what else is present. This experiment would directly inform implementation priorities: if two components capture most of the gain, practitioners can deploy simpler, more maintainable agents with well-understood failure modes rather than wrestling with the full Rainbow's complexity. The experiment is computationally feasible: each incremental variant is one 10-day Atari run, and the sequence would require 7 runs (the 6 incremental agents plus DQN, reusing Rainbow's DQN baseline), plus the full Rainbow for comparison.

**Pairwise component interaction ablations: which components are synergistic versus redundant?** The current ablation study removes one component at a time, which cannot detect whether two components are synergistic (each helps more when the other is present), additive (their benefits are independent and sum), or partially redundant (each provides a similar benefit through different mechanisms, so removing one has small effect because the other compensates). The most interesting pairwise questions raised by the paper: (1) Does double Q-learning matter more when distributional RL is absent? (The support-clipping hypothesis predicts yes—removing distributional RL should increase the double-Q ablation gap.) (2) Does dueling matter more when distributional RL is absent? (The richer-representation hypothesis predicts yes—if distributional RL provides implicit value-advantage factorization through its 51-atom output, dueling's explicit factorization should be more valuable in the scalar-output case.) (3) Do multi-step returns and prioritized replay interact synergistically? (Multi-step returns change the TD error distribution; prioritized replay samples based on those errors; the two components might be more effective together than the sum of their individual contributions.) Answering these requires a factorial ablation: for each pair of interest, run Rainbow minus neither (full Rainbow), minus A only, minus B only, and minus both. With three pairs of interest, this adds 9 experimental conditions (each pair requires 4 variants, and the full Rainbow and single ablations are shared). At ~10 GPU-days per Atari run, this is ~90 GPU-days—substantial but feasible with modest parallelism. The results would transform the ablation analysis from a component ranking into a mechanistic interaction map, revealing which combinations are genuinely more than the sum of their parts.

**Extension to policy-gradient and actor-critic methods: testing whether the Rainbow components transfer across algorithmic families.** All Rainbow results are on Q-learning-based agents. The Discussion section notes that "similar ideas may benefit also policy-based RL algorithms such as TRPO, or actor-critic methods," but provides no evidence. A systematic transfer study would take a strong policy-gradient baseline (e.g., PPO on Atari, or A3C as already included in Figure 1) and incrementally add Rainbow-adapted components: distributional value functions (distributional critic), multi-step returns (GAE already partially captures this, but explicit n-step vs. λ-return comparisons would be informative), prioritized experience replay (the PPO/A3C literature uses on-policy data, but a replay buffer with off-policy corrections is possible), and Noisy Nets for exploration (replacing entropy bonuses or ε-greedy). The key question: do the components that dominate in Rainbow (prioritization, multi-step) remain dominant for policy-gradient agents, or does the ranking shift because policy-gradient methods already address some of DQN's bottlenecks through different mechanisms? For instance, A3C's parallel actor-learners provide a form of diversity that might partially substitute for prioritized replay's focus on high-error transitions. Distributional critics might matter more for policy-gradient methods (where accurate value estimates directly affect policy updates via advantage estimation) than for Q-learning (where the value is used only for bootstrapping). This study would establish whether Rainbow's findings are about deep RL in general or about Q-learning specifically—a distinction with major practical implications for algorithm selection.

**Diagnosing and closing the human-starts generalization gap.** Rainbow shows a 78-percentage-point gap between no-ops starts (231% median) and human starts (153% median)—the largest absolute drop among all baselines in Table 2. This gap is reported but never analysed. A dedicated investigation would: (1) compute the per-game human-starts gap (analogous to Figure 4's per-game ablation analysis) to identify which games drive the aggregate drop; (2) compare the state visitation distributions induced by the agent's own policy versus human play for those high-gap games, to characterise *how* the distributions differ (does the agent avoid certain regions that humans visit? Does it linger in safe states that humans quickly leave?); (3) ablate Rainbow components specifically for their contribution to the human-starts gap—Noisy Nets' learned exploration and prioritized replay's error-skewed sampling distribution are prime suspects for causing distributional specialisation to the agent's own trajectories; (4) test mitigations, including training with a mix of agent-generated and human-demonstration starting states (a form of domain randomisation for initial conditions), or adding an auxiliary loss that encourages the agent's state representation to be invariant to whether a state was reached via the agent's policy or a human trajectory. The paper's identification of this gap—and the fact that Rainbow's gap is larger than any baseline's—raises the possibility that combining multiple improvements amplifies overfitting to the agent's own trajectory distribution. Understanding this would inform deployment of integrated agents in settings where the initial state distribution at test time differs from training.

**Adaptive distributional support: learning the value range rather than fixing it.** Rainbow inherits a fixed $[v_{\text{min}}, v_{\text{max}}] = [-10, 10]$ from the distributional RL component, set once across all 57 games. The paper discovers that this range causes implicit value underestimation when true returns exceed 10, and that this has the side effect of suppressing overestimation bias. A principled approach would *learn* the support range, either per-game or adaptively during training. One concrete design: initialise $v_{\text{min}}$ and $v_{\text{max}}$ to conservative values (e.g., $[-1, 1]$) and expand the support whenever the projected target distribution places non-negligible mass at the boundary atoms, indicating that the true return distribution extends beyond the current support. This would eliminate the need to pre-specify a magic range while preserving the overestimation-suppression property (since the support expands only as needed, rather than being arbitrarily wide). A second design: learn $v_{\text{min}}$ and $v_{\text{max}}$ as trainable parameters (or as outputs of a small network conditioned on game-initial observations), optimised to minimise the projection error (mass lost at the boundaries). This would make the distributional representation scale-adaptive, removing a hyperparameter that the paper shows has non-trivial interactions with other components. The experiment would compare adaptive-support Rainbow against the fixed-support baseline on games with known extreme returns (e.g., Atlantis, where agents achieve scores orders of magnitude above human level) to test whether the adaptive support improves performance on games where the current $[-10, 10]$ range is most likely to clip values.

---

### Practical Applications and Downstream Use Cases

**Standardised strong baseline for deep RL research.** Rainbow provides a single, well-documented, open-source agent that integrates six major DQN improvements into one tested configuration. Before Rainbow, a researcher developing a new deep RL technique had to choose among several incompatible baselines (DQN, DDQN, Dueling DDQN, etc.) and could game their comparison by picking whichever baseline made their improvement look largest. Rainbow establishes a new floor: any claimed improvement to value-based deep RL should be demonstrated against Rainbow (or its closest modern equivalent), not against vanilla DQN. The paper's complete hyperparameter table (Table 1) and the fact that a full training run takes ~10 days on a single GPU make this practical for academic labs with modest compute. The 231% median score sets a concrete performance target—subsequent work can measure both whether they improve the median and, using the threshold-stratified analysis (Figure 2), *where* those improvements manifest. This standardisation reduces the noise in published comparisons and makes it harder to claim progress through hyperparameter tuning or baseline selection rather than genuine algorithmic advance.

**Incremental implementation roadmap for production RL systems.** For teams building real-world RL systems (game AI, robotics simulation, industrial control), implementing all six Rainbow components simultaneously is a substantial engineering investment, and the failure modes of the combined system are harder to diagnose than those of individual components. The ablation study provides a prioritised deployment roadmap: implement prioritized replay and multi-step returns first—these two components account for the largest performance drops when removed (Figure 3), help almost uniformly across environments (53/57 games each, Figure 4), and address genuinely distinct bottlenecks (data selection and reward propagation speed). The marginal cost of these additions is low (prioritized replay is a sampling change, not an architectural one; multi-step returns are a target computation change). Distributional RL and Noisy Nets are second-priority additions—they provide clear aggregate benefit but the distributional component primarily helps asymptotic performance (after 40M frames, Figure 3) and Noisy Nets show game-dependent effects. Dueling networks and double Q-learning are optional—their aggregate marginal contribution is small in the full combination, and a team could omit them without substantial performance loss while gaining simpler architecture and fewer hyperparameters. This prioritised roadmap converts the paper's scientific findings into an engineering decision framework, letting teams allocate implementation effort where the evidence shows it matters most.

**Difficulty-stratified evaluation for heterogeneous benchmarks.** The threshold-counting analysis in Figure 2—plotting the number of tasks where performance exceeds 20%, 50%, 100%, 200%, and 500% of a human baseline—is a lightweight evaluation methodology applicable to any heterogeneous benchmark. For practitioners deploying RL in multi-task settings (e.g., a warehouse robot that must handle picking, placing, sorting, and navigating), aggregate average success rate can hide regressions on specific task categories. Adopting a threshold-stratified evaluation reveals whether improvements are concentrated on already-solved tasks or are genuinely expanding the set of tasks where the system achieves minimal competence. A team could set domain-specific thresholds (e.g., "tasks where success rate exceeds 80%," "tasks where success rate exceeds 95%") and track these alongside the aggregate metric to ensure that algorithmic improvements benefit the hardest tasks rather than merely inflating scores on easy ones. The methodology requires no new infrastructure—just binning per-task scores by threshold and plotting counts over time—and provides a more honest picture of progress than a single-number summary.

---

### When to Prefer This Method

The Rainbow paper itself makes no explicit "prefer X over Y" decision framework—it demonstrates that combining six components works better than any individual baseline, but does not position Rainbow against qualitatively different algorithmic families (policy gradient, model-based RL, evolutionary methods) as a prescriptive choice. The Discussion section notes that many directions (policy gradient hybrids, hierarchical RL, episodic control, auxiliary tasks) are "promising candidates for further experiments on integrated agents" and frames Rainbow as a foundation, not as the final answer to algorithm selection. As such, a formulaic decision matrix would impose a structure the paper does not itself provide. The practical takeaway is not "prefer Rainbow under conditions X, Y, Z" but rather: (a) if you are building a value-based deep RL agent for discrete-action vision-based tasks, the six components can be combined successfully and the ablation results tell you which ones to prioritise; (b) the integration-ablation methodology itself is the transferable contribution and should be applied whenever multiple independent improvements accumulate in a research area, regardless of domain.
