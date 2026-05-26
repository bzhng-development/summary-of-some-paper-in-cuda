# MULTIPLAYER NASH PREFERENCE OPTIMIZATION

**ArXiv:** [2509.23102](https://arxiv.org/abs/2509.23102)

## 🎯 Pitch

This paper introduces Multiplayer Nash Preference Optimization (MNPO), a breakthrough framework that extends Nash learning from human feedback (NLHF) to the multiplayer setting, capturing the true diversity and complexity of human preferences. By formulating large language model alignment as an n-player game, MNPO enables more robust policy optimization that accounts for heterogeneous, possibly non-transitive preferences—demonstrating both theoretical rigor and empirical gains over state-of-the-art two-player baselines. This approach addresses the critical limitations of single-opponent algorithms and moves the field closer to genuinely human-centric, scalable, and reliable AI alignment.

---

## 1. Executive Summary

This work introduces **Multiplayer Nash Preference Optimization (MNPO)**, a framework that generalizes Nash learning from human feedback to n-player games by formulating preference optimization as competition against a population of opponents—either time-indexed historical policies (**TD-MNPO**) or policies paired with heterogeneous preference oracles (**HT-MNPO**)—rather than a single opponent. Evaluated on instruction-following benchmarks including AlpacaEval 2.0, Arena-Hard, and MT-Bench using Gemma-2-9B-it as the base model, TD-MNPO achieves a 4.23-point improvement on Arena-Hard over the strongest two-player Nash baseline (52.26 vs. 48.03 for INPO) and a 2.92-point improvement over DPO on AlpacaEval 2.0, while preserving reasoning and knowledge capabilities across 11 academic benchmarks. The framework subsumes many existing preference optimization algorithms as special cases—including DPO, SimPO, SPPO, INPO, and SPIN—by varying the number of players, opponent selection, distance metric, and target reward gap, establishing that multiplayer competitive dynamics yield more robust and effective alignment only when the homogeneous preference oracle restriction holds for formal equilibrium guarantees.

## 2. Context and Motivation

### The Core Problem: Real Preferences Don't Fit the Two-Player Mold

The fundamental question this paper tackles is: **if human preferences are inherently diverse, non-transitive, and multi-sourced, why do our alignment frameworks force them into a two-player game?** This matters because the dominant paradigm for aligning large language models—Reinforcement Learning from Human Feedback (RLHF)—rests on assumptions that empirical evidence increasingly shows are violated in practice.

Traditional RLHF, as deployed in systems like InstructGPT (Ouyang et al., 2022), Claude (Bai et al., 2022), and Gemini (Team et al., 2023), builds on the **Bradley–Terry model** (Bradley & Terry, 1952). This model assumes that human preferences are transitive: if a human prefers response A over B, and B over C, then they must prefer A over C. It further assumes that these preferences can be captured by a single scalar reward function $r^*(x, y)$, with the probability of preferring $y_1$ over $y_2$ given by:

$$P(y_1 \succ y_2 \mid x) = \frac{\exp(r^*(x, y_1))}{\exp(r^*(x, y_1)) + \exp(r^*(x, y_2))} = \sigma\big(r^*(x, y_1) - r^*(x, y_2)\big)$$

Under these assumptions, learning a reward model from pairwise comparisons and optimizing against it is mathematically well-behaved—it reduces to maximizing expected reward with a KL-divergence penalty toward a reference policy (Equation 1 in the paper):

$$J(\pi) = \mathbb{E}_{x \sim d_0}\Big[\mathbb{E}_{y \sim \pi(\cdot|x)}[R(x, y)] - \tau\,\text{KL}(\pi(\cdot|x) \parallel \pi_{\text{ref}}(\cdot|x))\Big]$$

However, recent empirical studies (Ethayarajh et al., 2024; Wu et al., 2024) reveal that **real human preferences are often non-transitive**—you can have cycles where A is preferred to B, B is preferred to C, but C is preferred to A. This can happen because different annotators value different dimensions (helpfulness vs. safety vs. conciseness), because the same annotator applies different criteria across contexts, or because preferences genuinely exhibit intransitive structures that cannot be reduced to a single scalar ranking.

### The Nash Learning from Human Feedback Response—and Its Limitations

To address the Bradley–Terry bottleneck, a line of work beginning with Munos et al. (2023) reframed alignment as a **two-player Nash game**. Instead of assuming transitive preferences encoded in a scalar reward, these methods assume only the existence of a **general preference oracle** $P: \mathcal{X} \times \mathcal{Y} \times \mathcal{Y} \to [0, 1]$ that can be queried for binary preference signals. The game objective becomes (Equation 2):

$$J(\pi_1, \pi_2) = \mathbb{E}_{x \sim d_0}\Big[\mathbb{E}_{y_1 \sim \pi_1, y_2 \sim \pi_2}[P(y_1 \succ y_2 \mid x)] - \tau\,\text{KL}(\pi_1 \parallel \pi_{\text{ref}}) + \tau\,\text{KL}(\pi_2 \parallel \pi_{\text{ref}})\Big]$$

Here, the max-player $\pi_1$ maximizes win probability against $\pi_2$ while staying close to $\pi_{\text{ref}}$, and the min-player $\pi_2$ minimizes $\pi_1$'s win probability while also staying close to $\pi_{\text{ref}}$. Due to symmetry, the Nash equilibrium is unique and both players converge to the same optimal policy $\pi^*$, which is the best response against itself—meaning $J(\pi^*, \pi^*) = 0.5$ and no alternative policy can achieve a win rate above 50% against it.

This game-theoretic formulation spawned several algorithms with strong theoretical guarantees:

- **INPO** (Iterative Nash Policy Optimization, Zhang et al., 2025b): Uses no-regret learning with a multiplicative weights update that competes against both the reference policy and the previous iteration's policy simultaneously.
- **ONPO** (Optimistic Nash Policy Optimization, Zhang et al., 2025a): Incorporates optimistic mirror descent to accelerate convergence.
- **EGPO** (Extragradient Preference Optimization, Zhou et al., 2025): Uses extragradient techniques to ensure last-iterate convergence under noisy preferences.
- **SPPO** (Self-Play Preference Optimization, Wu et al., 2024): Approximates the Nash equilibrium through self-play with win-rate estimates.
- **DNO** (Direct Nash Optimization, Rosset et al., 2024): Uses the current policy as the opponent in an iterative self-improvement loop.

These methods collectively demonstrate that game-theoretic approaches can outperform reward-based RLHF, particularly when preference structures deviate from Bradley–Terry assumptions.

**However, all of these methods share a critical limitation: they are fundamentally restricted to two-player interactions.** In every case, the policy being trained competes against exactly one opponent—whether it's a fixed reference model, a previous checkpoint, or a single synthetic adversary. The paper identifies this as a **single-opponent bias** that creates specific failure modes:

1. **Oscillatory behavior**: When optimized against one opponent at a time, the policy can overfit to that opponent's weaknesses, then swing wildly when the opponent changes. This is visible in the training dynamics of iterative methods where performance can degrade between iterations.

2. **Narrow exploration**: A single opponent provides only one dimension of competitive pressure. The policy never learns to handle the full diversity of possible counter-strategies, leading to brittle solutions.

3. **Brittle approximation of preference populations**: Real alignment scenarios rarely involve a single preference source. They involve mixtures of annotators with different criteria, multiple reward models trained for different quality dimensions (helpfulness, safety, truthfulness), or sequences of historical model checkpoints—all creating inherently multi-source, sometimes conflicting, preference signals (Freund & Schapire, 1999).

### Where Existing Approaches Fall Short: The Multiplayer Reality

The paper identifies several concrete scenarios where the two-player restriction becomes particularly problematic:

**Heterogeneous annotator populations.** When alignment data comes from multiple annotators with different evaluation criteria—for instance, some prioritizing factual accuracy while others prioritize engaging style—reducing this to a single preference oracle flattens important distinctions. The policy cannot learn to balance competing desiderata because the training signal forces compromise into a single dimension.

**Multiple reward models.** Modern RLHF pipelines often train separate reward models for different quality axes (helpfulness, safety, truthfulness, conciseness). Each of these reward models induces its own preference oracle. A two-player framework must either train separate policies for each dimension (losing cross-dimension tradeoff optimization) or somehow aggregate the reward models into one (reintroducing the scalarization problem).

**Historical policy trajectories.** Iterative methods like INPO and DNO already maintain sequences of past policies. But in the two-player formulation, only the most recent policy (or at most two—current and reference) participates in the game. The paper argues that earlier checkpoints contain useful information: they represent different stages of capability development and different strategies that the current policy should be robust against.

**Preference non-transitivity at scale.** While the two-player Nash formulation can handle non-transitive preferences in principle (since it doesn't assume a scalar reward), it does so within the constrained geometry of a single opponent. In an $n$-player game, the policy must simultaneously maintain a favorable win rate against an entire population, which imposes a stronger and more realistic robustness requirement.

The paper's central insight is that **extending to the multiplayer setting provides a principled mean-field approximation** that reduces gradient variance, stabilizes optimization, and more precisely captures diverse preference structures. When all players share the same preference oracle—as naturally occurs when competing against historical versions of a single policy trajectory—the resulting symmetric game admits strong theoretical guarantees via the multiplicative weights update framework of Freund & Schapire (1999).

### How This Paper Positions Itself

The paper positions MNPO as a **unifying generalization** of existing Nash preference optimization methods. Rather than proposing an entirely new algorithm, it shows that the multiplayer formulation:

1. **Subsumes existing methods as special cases.** Table 1 demonstrates this explicitly: DPO is recovered with $n=2$, opponent $= \pi_{\text{ref}}$, and backward Bernoulli KL divergence; INPO is recovered with $n=3$, opponents $= \{\pi_t, \pi_{\text{ref}}\}$, and specific weighting coefficients; SPPO, SPIN, IPO, and others all emerge from Equation 17 with appropriate parameter choices. This is not merely taxonomic—it means the multiplayer framework provides a conceptual toolkit for understanding when and why different methods work, and for designing new variants by adjusting the opponent set and distance metric.

2. **Provides stronger theoretical guarantees in the homogeneous case.** When all players share the same preference oracle, MNPO inherits the convergence properties of two-player methods while enabling richer equilibrium dynamics. The paper proves (via Lemma 1 and Proposition 1) that the time-dependent update rule produces a unique minimizer at each iteration, and that the average policy over $T$ iterations converges to an $\epsilon$-approximate Nash equilibrium with $\epsilon = O(1/\sqrt{T})$.

3. **Extends naturally to heterogeneous oracles without sacrificing empirical performance.** The HT-MNPO variant (Equation 18) replaces the historical mixture over past policies with a mixture over opponent policies paired with distinct reward models. While this sacrifices formal Nash equilibrium guarantees (since the game becomes general-sum when $P_i \neq P_j$), the paper argues that the algorithmic structure remains principled—each policy optimizes against the current opponent distribution using its own oracle—and the empirical results bear this out.

4. **Addresses a practical gap in the RLHF-to-NLHF transition.** The shift from reward-based RLHF to Nash-based NLHF solved the Bradley–Terry transitivity problem but introduced a new bottleneck: the single-opponent bias. MNPO resolves this by generalizing the game structure while preserving the non-transitive preference handling that motivated NLHF in the first place. It is, in effect, a second-order correction: NLHF fixed the *preference model*, and MNPO fixes the *interaction model*.

The paper's framing in Section 3.1 explicitly builds on the Plackett–Luce model (Debreu, 1960; Plackett, 1975) to generalize from pairwise to listwise comparisons, replacing the logistic function $\sigma(r(x, y_1) - r(x, y_2))$ with a softmax over multiple alternatives:

$$P\!\left(y_i \succ \{y_j\}_{j \neq i} \mid x\right) = \frac{\exp(R(x, y_i))}{\exp(R(x, y_i)) + \sum_{j \neq i} \exp(R(x, y_j))}$$

This is not just a technical convenience—it provides the mathematical bridge between pairwise preference optimization (DPO, INPO) and the multiplayer objective in Equation 7:

$$J(\pi_i, \{\pi_j\}_{j \neq i}) = \mathbb{E}_{x \sim d_0}\!\left[\mathbb{E}_{y_i \sim \pi_i, \{y_j \mid y_j \sim \pi_j\}_{j \neq i}}\!\left[P\!\left(y_i \succ \{y_j\}_{j \neq i} \mid x\right)\right] - \tau\,\text{KL}(\pi_i(\cdot \mid x) \parallel \pi_{\text{ref}}(\cdot \mid x))\right]$$

When $k=2$ (one-vs-one comparison), this reduces exactly to the Bradley–Terry objective, ensuring backward compatibility with the entire RLHF literature. The key difference is that the softmax now penalizes cases where $y_i$ fails to dominate the collective strength of *all* alternatives simultaneously, not just a single opponent—a much stronger robustness requirement that the paper argues better reflects real-world alignment demands.

## 3. Technical Approach

### 3.1 Reader Orientation

MNPO is a training framework that teaches a language model to produce responses preferred by humans by having it compete simultaneously against a *population* of other policies—rather than just one opponent—in a mathematical game where the winner is the policy whose outputs are most consistently preferred. The core problem it solves is that real-world human preferences come from many sources (different annotators, different evaluation criteria, different reward models), and reducing this diversity to a single opponent in traditional Nash learning creates brittle policies that exploit that one opponent's weaknesses instead of learning broadly robust behavior. The "shape" of the solution is to generalize the two-player Nash game to an $n$-player game, where each policy maximizes its average win rate against *all* other policies while staying close to a trusted reference model, yielding a competitive equilibrium that balances performance against the entire population with adherence to the baseline.

### 3.2 Big-Picture Architecture (Diagram in Words)

The MNPO training system has five major components:

1. **Base Policy Model ($\pi_\theta$)** — the large language model being aligned (Gemma-2-9B-it). It generates candidate responses to prompts and is iteratively updated through preference optimization.

2. **Preference Oracle(s)** — one or more mechanisms that compare responses and output binary preference signals (which response is better). In the homogeneous case (TD-MNPO), a single reward model (ArmoRM-Llama3-8B-v0.1) serves as the universal preference oracle. In the heterogeneous case (HT-MNPO), multiple reward models (Skywork-Reward-V2-Llama-3.1-8B, Athene-RM-8B) provide distinct preference signals, each paired with a specific policy in the population.

3. **Opponent Population** — a set of $n$ policies that the current policy competes against. In TD-MNPO, opponents are time-indexed historical checkpoints of the same policy (e.g., $\pi, \pi_{t-1}, \pi_{t-2}$), weighted by recency. In HT-MNPO, opponents are distinct policies each paired with their own preference oracle.

4. **Multiplayer Game Objective** — the mathematical criterion that defines what "good" means: maximize the probability of being preferred over all opponents simultaneously while staying close to a reference policy $\pi_{\text{ref}}$ through a KL-divergence penalty with coefficient $\tau$.

5. **Iterative Update Rule (MNPO Loss)** — the practical training loss derived from the theoretical multiplicative weights update that avoids computing an intractable normalization factor. It compares the log-ratio of the current policy's responses to a weighted combination of opponent policies' log-ratios, aligned with a target reward gap.

Information flows through the system in an iterative loop (Algorithm 1 in Appendix B): a prompt is sampled → the current policy $\pi_t$ generates two candidate responses → the preference oracle compares them and outputs which is preferred → the MNPO loss is computed using the preferred/dispreferred pair, the current policy's log-probabilities, and the weighted log-probabilities from the opponent population → the policy parameters are updated via gradient descent → the updated policy becomes $\pi_{t+1}$ and the oldest opponent may be dropped as newer checkpoints are added → repeat for $T = 3$ iterations.

### 3.3 Roadmap for the Deep Dive

- **First**, the formal multiplayer game objective (Equation 7) and its Plackett–Luce foundation, since this defines what equilibrium means in the $n$-player setting and why the structure admits theoretical guarantees.

- **Second**, the homogeneous preference oracle setting and the theoretical multiplicative weights update (Equation 10), because this is the idealized algorithm that the practical loss function approximates—understanding it reveals why the loss has its particular form.

- **Third**, the practical MNPO loss derivation (Equations 12–15), covering how the intractable normalization factor is eliminated, how the log-ratio formulation emerges, and how the loss connects to reward-aware preference optimization (RPO).

- **Fourth**, the time-dependent opponent selection mechanism (TD-MNPO, Equation 17), because this is the concrete algorithm used in experiments—how historical policies are selected, weighted, and combined into the opponent population.

- **Fifth**, the heterogeneous extension (HT-MNPO, Equation 18) and its game-theoretic properties, covering when and why it works empirically despite lacking formal equilibrium guarantees.

- **Sixth**, the unified connections to existing RLHF methods (Table 1), showing precisely how DPO, INPO, SPPO, SPIN, and others are recovered as special cases by varying the number of players, opponent selection, distance metric, and target reward gap.

### 3.4 Detailed, Sentence-Based Technical Breakdown

This is primarily a **theoretical framework paper** whose core idea is that extending Nash preference optimization from two-player to $n$-player games yields more robust alignment by forcing each policy to simultaneously satisfy a population of diverse preference signals rather than exploiting a single opponent.

---

#### The Plackett–Luce Foundation: Generalizing from Pairwise to Listwise Comparisons

Before the multiplayer game can be defined, the preference comparison itself must be generalized. In two-player NLHF, preferences are pairwise: given two responses $y_1$ and $y_2$, the preference oracle outputs which one is preferred. In the multiplayer setting, each policy must be compared against *all* opponents at once—we need a way to say "$y_i$ is preferred over the entire set $\{y_j\}_{j \neq i}$."

**The Bradley–Terry model** handles pairwise comparisons by assuming a latent reward $R(x, y)$ and modeling the probability that $y_1$ is preferred over $y_2$ as:

$$P(y_1 \succ y_2 \mid x) = \frac{\exp(R(x, y_1))}{\exp(R(x, y_1)) + \exp(R(x, y_2))} = \sigma\big(R(x, y_1) - R(x, y_2)\big)$$

where $\sigma(\cdot)$ is the logistic sigmoid function, $R(x, y)$ represents the scalar "quality" of response $y$ for prompt $x$, and the denominator sums over both alternatives being compared.

**What this computes:** the probability that response $y_1$ beats $y_2$ as a function of the difference in their latent rewards. If $R(x, y_1)$ is much larger than $R(x, y_2)$, the probability approaches 1; if they are equal, it is exactly 0.5.

**Why this form (as background):** The Bradley–Terry model reduces a complex preference to a single scalar comparison. This is mathematically convenient—the sigmoid function maps any real-valued difference to a valid probability in $(0, 1)$—but it assumes transitivity, which empirical evidence contradicts (Ethayarajh et al., 2024; Wu et al., 2024).

**The paper generalizes this to the Plackett–Luce model** (Debreu, 1960; Plackett, 1975), which extends the softmax from two items to $k$ items. Given a pool of $k$ to-be-ranked responses $\{y_1, y_2, \ldots, y_k\}$ and a learned reward function $R(x, y)$, the probability that $y_i$ is preferred over all the others simultaneously is:

$$P\!\left(y_i \succ \{y_j\}_{j \neq i} \mid x\right) = \frac{\exp(R(x, y_i))}{\exp(R(x, y_i)) + \sum_{j \neq i} \exp(R(x, y_j))}$$

where the numerator is the exponentiated reward of the target response $y_i$, and the denominator sums the exponentiated rewards of *all* responses in the comparison pool (including $y_i$ itself).

**What this computes:** the probability that $y_i$ is the top-ranked choice among the entire set of $k$ alternatives, assuming that preferences follow a random utility model where each response's utility is its reward plus Gumbel noise. The softmax structure means that $y_i$ must dominate not just any single opponent but the *collective strength* of all alternatives.

**Why this form:** the Plackett–Luce model is the natural $k$-ary generalization of the Bradley–Terry model. Critically, when $k = 2$ (a pairwise comparison), this reduces exactly to the Bradley–Terry sigmoid form:

$$P\!\left(y_i \succ \{y_j\}_{j \neq i} \mid x\right) \Big|_{k=2} = \frac{\exp(R(x, y_i))}{\exp(R(x, y_i)) + \exp(R(x, y_j))} = \sigma(R(x, y_i) - R(x, y_j))$$

This backward compatibility is essential: it means the entire RLHF literature—DPO, PPO-based RLHF, all Bradley–Terry methods—can be expressed as special cases of the Plackett–Luce framework with $k=2$. The generalization to $k > 2$ is what enables the multiplayer game structure, where each policy competes against $n-1$ opponents at once.

**The corresponding negative log-likelihood** for reward learning under the Plackett–Luce model (Section 3.1, Equation 6) is derived from the probability above. For a single comparison where $y_i$ is the preferred response among the pool $\{y_1, \ldots, y_k\}$:

$$-\log P\!\left(y_i \succ \{y_j\}_{j \neq i} \mid x\right) = \log\!\left(\exp(R(x, y_i)) + \sum_{j \neq i} \exp(R(x, y_j))\right) - R(x, y_i)$$

where the log-sum-exp (LSE) term aggregates the exponentiated rewards of all alternatives, and the $-R(x, y_i)$ term penalizes the model when the preferred response's reward is not sufficiently larger than the collective LSE of the competitors.

**What this computes:** the cross-entropy loss for teaching a reward model to rank $y_i$ above all other responses in the pool. The LSE acts as a "soft maximum" over the dispreferred responses—it is always at least as large as any individual $\exp(R(x, y_j))$, meaning the model is penalized proportionally to how far $R(x, y_i)$ falls below the soft ceiling created by the strongest dispreferred responses.

**Why this form:** the LSE function has the mathematical property that $\text{LSE}(a_1, \ldots, a_k) \approx \max(a_1, \ldots, a_k)$ when one element dominates, but it is smooth and differentiable everywhere. This makes it suitable for gradient-based optimization while capturing the intuition that $y_i$ should beat *all* competitors, not just a single randomly selected one. When $k=2$, this loss reduces to $-\log \sigma(R(x, y_i) - R(x, y_j))$, which is exactly the standard Bradley–Terry reward learning objective used in classical RLHF pipelines.

---

#### The Homogeneous Multiplayer Game Objective

With the Plackett–Luce comparison model established, the paper defines the objective for the homogeneous multiplayer setting—the case where all $n$ players share exactly the same preference oracle $P$. This symmetry is what enables the formal equilibrium guarantees.

**The homogeneous preference oracle** is defined as $P: \mathcal{X} \times \mathcal{Y} \times \{\mathcal{Y}\}^{n-1} \to [0, 1]$. It takes a prompt $x$, a candidate response $y_i$ from player $i$, and a set of $n-1$ responses $\{y_j\}_{j \neq i}$ from the other players, and outputs the probability that $y_i$ is preferred over all opponents. Queries to this oracle return binary preference signals:

$$z \sim \text{Ber}\!\left(P\!\left(y_i \succ \{y_j\}_{j \neq i} \mid x\right)\right)$$

where $z = 1$ means "$y_i$ wins against the group" and $z = 0$ means "the group as a whole is preferred over $y_i$."

**The multiplayer game objective** (Section 3.1, Equation 7) for each policy $\pi_i$ competing against the other $n-1$ policies is:

$$J(\pi_i, \{\pi_j\}_{j \neq i}) = \mathbb{E}_{x \sim d_0}\!\left[\mathbb{E}_{y_i \sim \pi_i, \{y_j \mid y_j \sim \pi_j\}_{j \neq i}}\!\left[P\!\left(y_i \succ \{y_j\}_{j \neq i} \mid x\right)\right] - \tau\,\text{KL}(\pi_i(\cdot \mid x) \parallel \pi_{\text{ref}}(\cdot \mid x))\right]$$

where $d_0$ is the fixed but unknown prompt distribution, $\pi_i(\cdot \mid x)$ is the response distribution of player $i$, $\{\pi_j\}_{j \neq i}$ are the response distributions of all other players, $P(y_i \succ \{y_j\}_{j \neq i} \mid x)$ is the probability that $y_i$ is preferred over the collective of opponent responses, $\tau > 0$ is the KL regularization coefficient, and $\pi_{\text{ref}}$ is the reference policy (typically the supervised fine-tuned model before any RLHF).

**What this computes:** the expected value for player $i$—how often its responses beat the collective of opponents, minus a penalty for deviating too far from the reference policy. The outer expectation $\mathbb{E}_{x \sim d_0}$ averages over the prompt distribution. The inner expectation $\mathbb{E}_{y_i \sim \pi_i, \{y_j \mid y_j \sim \pi_j\}}$ samples one response from player $i$ and one response from each opponent, then evaluates whether $y_i$ wins against the entire opponent set. The KL term $-\tau\,\text{KL}(\pi_i \parallel \pi_{\text{ref}})$ penalizes player $i$ for producing responses that the reference policy would assign low probability to, preventing the policy from drifting into degenerate regions of the output space where it might exploit quirks of the preference oracle without actually producing good responses.

**Why this form:** the objective generalizes the two-player Nash objective (Equation 2 in the paper) in three important ways. First, the preference comparison is now listwise ($y_i$ vs. all $\{y_j\}$) rather than pairwise ($y_1$ vs. $y_2$), which means the policy must be robust against a whole distribution of opponent strategies simultaneously. Second, the game is fully symmetric—every player's objective has the identical structure with the same $P$ and the same $\pi_{\text{ref}}$—which is what guarantees $\pi^*_1 = \pi^*_2 = \cdots = \pi^*_n$ at equilibrium (all players converge to the same optimal policy). Third, the optimization is decentralized: each player's update depends only on its own actions and the aggregate behavior of opponents, without requiring coordination or joint optimization. This decentralized structure is what makes the iterative, parallel training procedure computationally tractable.

**Three properties that the paper explicitly highlights** (Section 3.1, after Equation 7):

- **(i) Symmetric treatment:** All policies compete in a symmetric manner. No player has a privileged position—each sees the same objective structure relative to the same reference policy. At equilibrium, $\pi^*_1 = \pi^*_2 = \cdots = \pi^*_n$, meaning the game identifies a single robust policy that is the best response against itself in a population sense.

- **(ii) Decentralized optimization:** Each policy's update depends only on its own parameters and the fixed outputs of its opponents. There are no complex interdependencies where player $i$'s gradient depends on player $j$'s gradient. This makes the training loop parallelizable—all players can be updated concurrently given the current opponent outputs.

- **(iii) Generalization of the two-player case:** When $n = 2$, the objective reduces to the standard two-player Nash objective from Equation 2, where each policy maximizes its pairwise preference probability subject to its own KL penalty. This means MNPO is strictly more general than existing NLHF methods—any two-player method is a special case of the $n$-player formulation with $n=2$.

---

#### Nash Equilibrium and Duality Gap in the Multiplayer Setting

The concept of optimality in MNPO is defined through game-theoretic equilibrium rather than a single maximization criterion. Since multiple players are optimizing simultaneously, "optimal" means a state where no player can improve by unilaterally changing their strategy.

**The Nash equilibrium** in this $n$-player game (Section 3.1, Equation 8) is defined as a policy profile $(\pi^*_1, \pi^*_2, \ldots, \pi^*_n)$ satisfying:

$$J(\pi^*_i, \{\pi^*_j\}_{j \neq i}) \geq J(\pi_i, \{\pi^*_j\}_{j \neq i}), \quad \forall \pi_i \in \Pi, \; \forall i \in \{1, \ldots, n\}$$

where $\Pi$ is the policy class containing all policies with the same support set as $\pi_{\text{ref}}$, and $J(\cdot, \cdot)$ is the multiplayer objective from Equation 7.

**What this means operationally:** player $i$ evaluates its current value $J(\pi^*_i, \{\pi^*_j\})$ against the population of other Nash policies. If player $i$ were to deviate to any alternative policy $\pi_i$, its value could not increase—it would either stay the same or decrease. Since this holds for *all* players simultaneously, no one has an incentive to change, making the profile stable. Due to the symmetry of the game (all players share the same $P$ and $\pi_{\text{ref}}$), the equilibrium is unique and all players converge to the identical policy $\pi^*_1 = \cdots = \pi^*_n = \pi^*$.

**Why this equilibrium concept matters:** in the two-player case, $J(\pi^*, \pi^*) = 0.5$, meaning the Nash policy achieves exactly a 50% win rate against itself—it cannot beat its own clone. The key property is that $J(\pi^*, \pi) \geq 0.5$ for *any* alternative policy $\pi$, meaning no competitor can achieve a win rate above 50% against the Nash policy. In the multiplayer setting, the equilibrium has a similar property: each Nash policy achieves a win rate against the population of other Nash policies that cannot be exceeded by any unilateral deviation.

**The duality gap** (Section 3.1, Equation 9) quantifies how far a given policy is from the Nash equilibrium:

$$\text{DualGap}(\pi) = \max_{\pi' \in \Pi} J(\pi', \mathcal{O}_\pi) - J(\pi, \mathcal{O}_\pi)$$

where $\mathcal{O}_\pi = \{\pi_j\}^{n-1}_{j=1}$ is the fixed set of opponent policies, $\max_{\pi' \in \Pi} J(\pi', \mathcal{O}_\pi)$ is the best possible value any alternative policy could achieve against the current opponents, and $J(\pi, \mathcal{O}_\pi)$ is the value achieved by $\pi$ itself.

**What this computes:** the maximum improvement player $i$ could achieve by unilaterally switching to an optimal alternative strategy while all opponents remain fixed. It is always non-negative (since you can always choose $\pi' = \pi$ to get zero). A Nash equilibrium is characterized by $\text{DualGap}(\pi^*) = 0$—no alternative strategy can improve the value. A policy with $\text{DualGap}(\pi) \leq \epsilon$ is called an $\epsilon$-approximate Nash policy, meaning any alternative could improve the value by at most $\epsilon$.

**Why this metric:** the duality gap provides a single scalar summary of how close a policy is to equilibrium without requiring knowledge of the true Nash policy $\pi^*$. It is computable given only the current policy and its opponents. This is the standard metric in game-theoretic optimization for quantifying convergence quality when the equilibrium itself is unknown.

---

#### The Theoretical Multiplicative Weights Update

The paper derives an idealized iterative algorithm (Section 3.1, Equation 10) that would solve the homogeneous multiplayer game if it could be computed exactly. This is the theoretical foundation that the practical loss function approximates.

**The multiplicative weights update** for player $i$ at iteration $t+1$, given opponents $\{\pi^{(t)}_j\}_{j \neq i}$, is:

$$\pi^{(t+1)}_i(y \mid x) \propto \left(\prod_{j \neq i} \pi^{(t)}_j(y \mid x)\right)^{\frac{1}{n-1}} \exp\!\left(\frac{\eta}{n-1} \sum_{j \neq i} P\!\left(y \succ \pi^{(t)}_j \mid x\right)\right)$$

where $\pi^{(t)}_j(y \mid x)$ is the probability that opponent $j$ assigns to response $y$ for prompt $x$, $n$ is the total number of players, $P(y \succ \pi^{(t)}_j \mid x)$ is the probability that response $y$ is preferred over opponent $j$'s distribution (i.e., the expected win rate of the fixed response $y$ against random draws from $\pi^{(t)}_j$), and $\eta$ is the learning rate controlling the step size of the update.

**What this computes:** an updated probability distribution over responses for player $i$. For any specific response $y$, the update multiplies two factors: (1) the **geometric mean** of all opponents' current probabilities for $y$, raised to the power $1/(n-1)$—this is the "population belief" about how likely $y$ is—and (2) an **advantage exponentiation** $\exp(\frac{\eta}{n-1} \sum_{j \neq i} P(y \succ \pi^{(t)}_j \mid x))$ that amplifies responses with high average win rates against the opponent population. The proportionality symbol $\propto$ means the result must be normalized so that probabilities sum to 1 over all possible responses $y$.

**Why this form:** this update is an instance of **online mirror descent** with the KL divergence as the Bregman potential, following the framework of Freund & Schapire (1999). The theoretical analysis in Appendix F.3 shows that it can be derived as the solution to:

$$\pi^{(t+1)}_i = \arg\max_{\pi \in \Pi} \frac{1}{n-1} \sum_{j \neq i} \langle \pi, P(\cdot \succ \pi^{(t)}_j \mid x) \rangle - \frac{1}{\eta} \text{KL}\!\left(\pi \,\Big\|\, \Big(\prod_{j \neq i} \pi^{(t)}_j\Big)^{\frac{1}{n-1}}\right)$$

which maximizes the expected win rate against opponents while regularizing toward the geometric mean of opponent distributions. The regret bound guarantees that the average policy $\bar{\pi}^{(T)} = \frac{1}{T} \sum_{t=1}^T \pi^{(t)}$ converges to an $\epsilon$-approximate Nash equilibrium with $\epsilon = O(1/\sqrt{T})$ (Hart & Mas-Colell, 2000).

**The geometric mean interpretation:** the term $(\prod_{j \neq i} \pi^{(t)}_j(y \mid x))^{1/(n-1)}$ aggregates the beliefs of all opponents into a single distribution. If all opponents agree that response $y$ is likely, the geometric mean will be large; if any opponent assigns near-zero probability to $y$, the geometric mean collapses to near-zero. This provides stability: a response cannot gain high probability in the update unless it is considered plausible by *all* opponents, preventing the policy from exploiting narrow weaknesses of a single opponent.

**The advantage weighting interpretation:** the exponential term $\exp(\frac{\eta}{n-1} \sum_j P(y \succ \pi^{(t)}_j \mid x))$ amplifies responses that outperform opponents. The sum $\sum_j P(y \succ \pi^{(t)}_j)$ is the total win probability of $y$ against the population—responses that consistently defeat many opponents get exponentiated to higher values. The learning rate $\eta$ controls the exploration-exploitation tradeoff: small $\eta$ keeps the update close to the geometric mean (conservative, exploration-heavy), while large $\eta$ aggressively amplifies high-advantage responses (aggressive, exploitation-heavy).

**The critical problem:** Equation 10 cannot be computed directly because it involves a normalization factor (the partition function $Z_{\pi^{(t)}}(x) = \sum_y (\prod_j \pi^{(t)}_j(y \mid x))^{1/(n-1)} \exp(\frac{\eta}{n-1} \sum_j P(y \succ \pi^{(t)}_j \mid x))$) that sums over the exponentially large response space $\mathcal{Y}$. This is intractable for language models where the response space is combinatorially vast.

---

#### Eliminating the Intractable Normalization: The Log-Ratio Formulation

To avoid computing the partition function, the paper works with **log-ratios of probabilities between pairs of responses**. For any two responses $y$ and $y'$, the ratio $\pi^{(t+1)}_i(y \mid x) / \pi^{(t+1)}_i(y' \mid x)$ does not depend on the normalization factor, because the $Z_{\pi^{(t)}}(x)$ term cancels:

$$\frac{\pi^{(t+1)}_i(y \mid x)}{\pi^{(t+1)}_i(y' \mid x)} = \frac{(\prod_{j \neq i} \pi^{(t)}_j(y \mid x))^{1/(n-1)}}{(\prod_{j \neq i} \pi^{(t)}_j(y' \mid x))^{1/(n-1)}} \cdot \frac{\exp\!\left(\frac{\eta}{n-1} \sum_{j \neq i} P(y \succ \pi^{(t)}_j \mid x)\right)}{\exp\!\left(\frac{\eta}{n-1} \sum_{j \neq i} P(y' \succ \pi^{(t)}_j \mid x)\right)}$$

Taking the logarithm of both sides yields a linear relationship that the ideal updated policy must satisfy (Section 3.1, Equation 11):

$$\frac{1}{n-1} \sum_{j \neq i} \log \frac{\pi^{(t+1)}_i(y \mid x)}{\pi^{(t)}_j(y \mid x)} = \frac{\eta}{n-1} \sum_{j \neq i} P\!\left(y \succ \pi^{(t)}_j \mid x\right) - \log Z_{\pi^{(t)}}(x)$$

**To eliminate the remaining dependence on $Z_{\pi^{(t)}}(x)$**, the paper subtracts the same equation evaluated at $y'$ from the equation evaluated at $y$, which cancels the $\log Z_{\pi^{(t)}}(x)$ term. This motivates defining the **log-ratio function** $h_t(\pi, y, y')$ (Section 3.1, Equation 12):

$$h_t(\pi, y, y') = \log \frac{\pi(y \mid x)}{\pi(y' \mid x)} - \frac{1}{n-1} \sum_{j \neq i} \log \frac{\pi^{(t)}_j(y \mid x)}{\pi^{(t)}_j(y' \mid x)}$$

where $\log \frac{\pi(y \mid x)}{\pi(y' \mid x)}$ is the log-ratio assigned by the candidate policy $\pi$ to the response pair $(y, y')$, and $\frac{1}{n-1} \sum_{j \neq i} \log \frac{\pi^{(t)}_j(y \mid x)}{\pi^{(t)}_j(y' \mid x)}$ is the average log-ratio assigned by the opponent population to the same pair.

**What $h_t$ computes:** the difference between how the candidate policy $\pi$ ranks $y$ relative to $y'$ and how the opponent population collectively ranks them. If $\pi$ assigns a higher relative probability to $y$ than the opponents do (on average), $h_t(\pi, y, y')$ will be positive; if $\pi$ assigns a lower relative probability, it will be negative. This is essentially a "relative advantage" measure—how much more $\pi$ favors $y$ over $y'$ compared to the population consensus.

**The key equality** (Section 3.1, Equation 13): for the ideal updated policy $\pi^{(t+1)}$, the function $h_t$ must satisfy:

$$h_t(\pi^{(t+1)}, y, y') = \frac{\eta}{n-1} \sum_{j \neq i} \left[P\!\left(y \succ \pi^{(t)}_j \mid x\right) - P\!\left(y' \succ \pi^{(t)}_j \mid x\right)\right]$$

where the right-hand side is the learning-rate-scaled difference in win probabilities of $y$ versus $y'$ against the opponent population.

**What this equality means:** the ideal updated policy should adjust its log-ratios so that the relative preference for $y$ over $y'$ (compared to the opponent baseline) exactly matches the relative advantage of $y$ over $y'$ in terms of win probabilities. If $y$ has a much higher win rate against opponents than $y'$ does, the policy should increase its relative probability for $y$ proportionally. If they have equal win rates, the policy's relative probabilities should match the opponent population's relative probabilities.

**Why this equality holds** (from the derivation of Equation 11): it follows directly from the multiplicative weights update structure. When you take the log-ratio of Equation 10 evaluated at $y$ and $y'$, the normalization constant cancels, and you're left with exactly this relationship. The equation is not an approximation—it is an exact consequence of the ideal update rule, provided the partition function is consistent.

---

#### The Practical MNPO Loss Function

Since we cannot compute $\pi^{(t+1)}$ directly (due to the intractable normalization), the paper instead defines a loss function whose unique minimizer is $\pi^{(t+1)}$ (Section 3.1, Equation 14):

$$\pi^{(t+1)} \leftarrow \arg\min_{\pi} \underbrace{\mathbb{E}_{y_w, y_l \sim \mathcal{D}_t}\!\left[\left(h_t(\pi, y_w, y_l) - \frac{\eta}{n-1} \sum_{j \neq i} \left[P(y_w \succ \pi^{(t)}_j) - P(y_l \succ \pi^{(t)}_j)\right]\right)^2\right]}_{L_t(\pi)}$$

where $\mathcal{D}_t$ is a dataset of preference pairs $(y_w, y_l)$ where $y_w$ (the "winner") is preferred over $y_l$ (the "loser") according to the preference oracle, $h_t(\pi, y_w, y_l)$ is the log-ratio function from Equation 12 evaluated on the winning and losing responses, and the bracketed difference is the target advantage that the ideal updated policy would achieve.

**What this loss computes:** a squared error between the policy's actual log-ratio adjustment $h_t(\pi, y_w, y_l)$ and the target adjustment $\frac{\eta}{n-1} \sum_j [P(y_w \succ \pi^{(t)}_j) - P(y_l \succ \pi^{(t)}_j)]$. When $\pi = \pi^{(t+1)}$ (the ideal update), these two quantities are equal (by Equation 13), so $L_t(\pi^{(t+1)}) = 0$. For any other policy, they differ, producing a positive loss. Minimizing this loss through gradient descent finds the policy that satisfies the log-ratio condition without ever computing the partition function.

**Why this is valid:** Lemma 1 (proved in Appendix F.1) establishes that $\pi^{(t+1)}$ is the *unique* minimizer of $L_t(\pi)$ within the policy class $\Pi$. The proof works by contradiction: if there were a second minimizer $\tilde{\pi}$ with $L_t(\tilde{\pi}) = 0$, then $h_t(\tilde{\pi}, y, y')$ would match the target for all pairs $(y, y')$, which would force $\tilde{\pi}$ to have identical pairwise ratios to $\pi^{(t+1)}$, which would force $\tilde{\pi} = \pi^{(t+1)}$ (since the ratios uniquely determine a probability distribution on the support set). Therefore, gradient descent on $L_t(\pi)$ converges to the same policy that the intractable Equation 10 would produce.

**The simplification via preference sampling:** the term $P(y \succ \pi^{(t)}_j)$ in the loss is itself an expectation over opponent responses—it requires computing the win rate of a specific response $y$ against the entire distribution of responses from opponent $j$. This is also expensive. The paper replaces it with an equivalent formulation using direct preference sampling (Section 3.1, Equation 15 and Proposition 1):

$$L'_t(\pi) = \mathbb{E}_{y, y' \sim \pi^{(t)},\; y_w, y_l \sim \lambda_P(y, y')}\!\left[\left(h_t(\pi, y_w, y_l) - \frac{1}{2\eta}\right)^2\right]$$

where $y, y'$ are two responses sampled independently from the current policy $\pi^{(t)}$, $\lambda_P(y, y')$ is the preference distribution that assigns the pair $(y_w, y_l)$ as $(y, y')$ if $y$ is preferred over $y'$ according to the oracle $P$, and as $(y', y)$ otherwise, and $\frac{1}{2\eta}$ is a constant replacing the win-rate difference term.

**What Proposition 1 proves:** $L'_t(\pi)$ is equivalent to $L_t(\pi)$ up to an additive constant that does not depend on $\pi$. The equivalence works because: (1) when responses are sampled from the current policy $\pi^{(t)}$ and labeled by the preference oracle, the expected win-rate difference $\mathbb{E}_{y, y' \sim \pi^{(t)}}[P(y \succ \pi^{(t)}_j) - P(y' \succ \pi^{(t)}_j)]$ equals exactly $1$ (by symmetry—the win rate is 0.5 against itself, and the difference integrates to 1), and (2) the squared loss with target $\frac{1}{2\eta}$ has the same minimizer as the squared loss with the original target, because adding a constant to the target shifts the loss by a $\pi$-independent amount.

**Why this simplification matters:** the original loss $L_t(\pi)$ requires knowing the win-rate difference $P(y_w \succ \pi^{(t)}_j) - P(y_l \succ \pi^{(t)}_j)$, which itself requires expensive Monte Carlo estimation against each opponent. The simplified loss $L'_t(\pi)$ only requires: sample two responses from $\pi^{(t)}$, ask the oracle which one it prefers, compute $h_t(\pi, y_w, y_l)$, and take the squared difference from $\frac{1}{2\eta}$. The constant target $\frac{1}{2\eta}$ absorbs all the opponent-specific win-rate information because the sampling procedure (drawing from $\pi^{(t)}$ and labeling with $\lambda_P$) implicitly captures the correct expected advantage. This is a crucial computational simplification that makes the algorithm practical at scale.

---

#### Reward-Aware Connection to RPO

The paper observes that the loss function $L'_t(\pi)$ can be interpreted through the lens of **Reward-aware Preference Optimization** (RPO, Sun et al., 2025), which provides a unifying mathematical perspective (Section 3.2).

**The RPO framework** defines the loss over preference pairs as:

$$L^{\text{RPO}}_D(\pi_\theta, (x, y_1, y_2) \mid r^\star, \pi_{\text{ref}}, \beta, \eta) := D\!\left(r_{\pi_\theta}(x, y_1) - r_{\pi_\theta}(x, y_2) \;\big\|\; \eta r^\star(x, y_1) - \eta r^\star(x, y_2)\right)$$

where $r_{\pi_\theta}(x, y) = \beta \log \frac{\pi_\theta(y \mid x)}{\pi_{\text{ref}}(y \mid x)}$ is the implicit reward induced by the policy $\pi_\theta$ (the log-ratio of policy to reference probabilities, scaled by $\beta$), $r^\star(x, y)$ is a target reward model, $D: \mathbb{R} \times \mathbb{R} \to \mathbb{R}^*$ is a distance metric, and $\eta$ scales the target reward gap.

**What RPO captures:** the intuition that preference optimization should align the policy's implicit reward differences with explicit reward model differences. If the reward model says response $y_1$ is substantially better than $y_2$, the policy should assign a correspondingly higher log-probability to $y_1$ relative to $y_2$ (compared to the reference). The distance metric $D$ measures the discrepancy, and the specific choice of $D$ determines the optimization behavior.

**The MNPO loss as RPO with squared distance:** the paper shows that $L'_t(\pi)$ is exactly an instance of RPO with:

- Implicit reward: $r_{\pi_\theta}(x, y) = \mathbb{E}_{\pi_j}\!\left[\log \frac{\pi(y \mid x)}{\pi_j(y \mid x)}\right]$, where the reference is now the average of opponent policies rather than a single $\pi_{\text{ref}}$
- Target reward gap: $\eta r^\star(x, y_1) - \eta r^\star(x, y_2) = \frac{1}{2\eta}$, a constant independent of the specific responses
- Distance metric: $D_{\text{sq}}(a, b) = (a - b)^2$, the squared distance

**What this connection means:** the MNPO loss can be understood as encouraging the policy's implicit reward (log-ratio against opponents) to match a constant target gap. The constant $\frac{1}{2\eta}$ encodes the ideal win-rate advantage—when the policy's log-ratio adjustment exactly matches this constant, it achieves the equilibrium properties described by the multiplicative weights theory. The squared distance metric penalizes both under-adjustment (not favoring winners enough) and over-adjustment (favoring winners too aggressively, which could lead to over-optimization).

**Why this connection is valuable:** it bridges the game-theoretic MNPO framework with the broader literature on reward-aware preference optimization. It shows that what MNPO is doing—encouraging the policy to maintain a consistent relative advantage over opponents—is algebraically equivalent to reward-model distillation with a constant target, but where the "reference reward model" is implicitly defined by the opponent population rather than a separately trained model. This perspective also explains why the method is robust: by averaging over multiple opponents, the implicit reward signal is smoother and less prone to the kind of exploitation that occurs when optimizing against a single imperfect reward model.

---

#### Time-Dependent MNPO (TD-MNPO): The Practical Algorithm

The theoretical framework assumes a fixed set of $n$ opponents. In practice, we don't have $n$ separate policies available at the start of training—the policy is being trained iteratively, and opponents must be constructed from what is available. The paper adopts a **time-dependent opponent selection** mechanism inspired by recent iterative preference optimization methods like DNO (Rosset et al., 2024), SPIN (Chen et al., 2024), and INPO (Zhang et al., 2025b).

**Opponent construction** (Section 3.2): at iteration $t$, the opponent set is constructed as a mixture of recent time-indexed historical policies $\{\pi_{t-j}\}_{j=0}^{t}$ (where $n \leq t+1$ ensures we have enough history), with each opponent $\pi_{t-j}$ weighted by a coefficient $\lambda_j \in [0, 1]$. The weights optionally satisfy $\sum_j \lambda_j \leq 1$, allowing for a "reference policy" component that soaks up any remaining weight.

**The TD-MNPO loss** (Section 3.2, Equation 17) is:

$$L^{\text{TD}}_{t, D}(\pi \mid \beta, \{\lambda_j\}, \eta) = \mathbb{E}_{y, y' \sim \pi,\; y_w, y_l \sim \lambda_P(y, y')}\; D\!\left[\log \frac{\pi(y_w \mid x)}{\pi(y_l \mid x)} - \sum_{j=0}^{n-2} \lambda_j \log \frac{\pi_{t-j}(y_w \mid x)}{\pi_{t-j}(y_l \mid x)} \;\Big\|\; \eta \delta^\star\right]$$

where $\pi$ is the current policy being optimized, $\{\pi_{t-j}\}_{j=0}^{n-2}$ are the historical opponent policies (with $\pi_{t-0} = \pi_t$ being the most recent, $\pi_{t-1}$ the previous iteration, etc.), $\lambda_j$ are the importance weights for each opponent, $\log \frac{\pi(y_w \mid x)}{\pi(y_l \mid x)}$ is the log-ratio assigned by the current policy to the winning vs. losing response, $\sum_j \lambda_j \log \frac{\pi_{t-j}(y_w \mid x)}{\pi_{t-j}(y_l \mid x)}$ is the weighted average log-ratio assigned by the opponent population, $D$ is a distance metric (squared distance $D_{\text{sq}}$ in the paper's implementation), $\eta$ is the reward scaling parameter, and $\delta^\star$ encodes the target reward gap (set to $\frac{1}{2\eta}$ following Equation 15).

**What this loss computes operationally:** for each training example, we have a prompt $x$, a winning response $y_w$, and a losing response $y_l$. The policy's current log-ratio $\log \frac{\pi(y_w \mid x)}{\pi(y_l \mid x)}$ (how much more likely the policy makes the winner than the loser) is compared against the opponent-weighted log-ratio $\sum_j \lambda_j \log \frac{\pi_{t-j}(y_w \mid x)}{\pi_{t-j}(y_l \mid x)}$ (how much more likely the opponent population collectively makes the winner). The difference between these—the policy's "excess preference" for the winner over the opponent baseline—is compared to the target reward gap $\eta \delta^\star$ via the distance metric $D$. The loss penalizes deviations from the target gap.

**How the training loop works** (Algorithm 1 in Appendix B, with $T = 3$ iterations):

1. **Iteration 0:** Start with the base model (Gemma-2-9B-it) as $\pi_0$. The reference policy $\pi_{\text{ref}}$ is also set to this initial model.

2. **Iteration 1:** Generate response pairs from $\pi_0$ on training prompts. Query the preference oracle (ArmoRM-Llama3-8B-v0.1) to get preference labels. Construct opponent set from available historical policies—at this point, only $\pi_0$ exists, so the "population" may reduce to a two-player or degenerate case depending on the number of players $n$. Update $\pi_0$ to $\pi_1$ by minimizing the TD-MNPO loss.

3. **Iteration 2:** Generate response pairs from $\pi_1$ on fresh prompts. With at least two historical policies now ($\pi_0, \pi_1$), construct the opponent set using weighted combinations—for example, with $n=3$, opponents might be $\pi_1$ (weight $\lambda_0$) and $\pi_0$ (weight $\lambda_1$). Query the oracle, compute the loss, update to $\pi_2$.

4. **Iteration 3:** Repeat with $\pi_2$ generating responses. With three historical policies ($\pi_0, \pi_1, \pi_2$), richer opponent mixtures are possible. Update to the final policy $\pi_3$.

**Hyperparameter settings** (Appendix C): The paper uses a cosine learning-rate scheduler with a peak learning rate of $5 \times 10^{-7}$, a warmup ratio of 0.1, and a global batch size of 128. The optimizer is AdamW (Loshchilov & Hutter, 2017) without weight decay. For TD-MNPO, the weighting coefficients $\lambda_j$ are selected via grid search from $\{0, 0.1, 0.333, 0.5, 0.667, 0.9\}$, and $\eta$ is chosen from $\{0.1, 0.01, 0.0075, 0.005, 0.002\}$ with the final value set to $\eta = 0.0075$. The regularization parameter $\beta$ is maintained within the range $[0.01, 10]$, and the paper notes that gradually increasing $\beta$ throughout training effectively mitigates training degradation while enabling continued model improvement.

**The four claimed benefits of time-dependent opponent selection** (Section 3.2):

- **(i) Smoother policy evolution:** By blending multiple past policies rather than relying solely on the most recent one, the update avoids abrupt shifts. The penalty term involves a weighted average, which is more stable than a single-policy comparison. If one iteration produces a poor policy (due to noise or a bad batch), its influence is diluted by the other opponents.

- **(ii) Greater robustness:** Transient fluctuations in recent iterations are mitigated. If the policy accidentally develops a narrow strength (e.g., producing verbose responses that fool the oracle on one batch), the presence of older checkpoints—which don't share that narrow strength—provides a correcting signal. The policy must maintain advantage against the *entire history*, not just the latest snapshot.

- **(iii) Unified interpretation:** As summarized in Table 1, the loss recovers DPO, INPO, SPIN, SPPO, IPO, and others as special cases by varying $n$, opponent selection, and distance metric. This is not just a taxonomy—it means practitioners can understand existing methods through a single conceptual lens and design new variants by adjusting the opponent mixture.

- **(iv) Stable convergence:** Recent policies exert greater influence (via higher $\lambda_j$ on recent indices) while older policies preserve the broader learning trajectory. This recency-weighted averaging is similar to momentum in optimization—it accelerates convergence in consistent directions while smoothing out oscillatory components.

---

#### Heterogeneous MNPO (HT-MNPO): Multiple Preference Oracles

The homogeneous setting assumes all players share the same preference oracle. This is natural when opponents are historical checkpoints of the same model (they were all trained with the same reward model). But many real-world scenarios involve preference signals from heterogeneous sources—different annotators, different reward models trained for different quality dimensions.

**The heterogeneous formulation** (Section 3.3): each player $\pi_i$ is now paired with a distinct reward model $r_i(x, y)$, which induces a player-specific preference oracle $P_i: \mathcal{X} \times \mathcal{Y} \times \{\mathcal{Y}\}^{n-1} \to [0, 1]$. The objective for player $i$ becomes:

$$J_i(\pi_i, \{\pi_j\}_{j \neq i}) = \mathbb{E}_{x \sim d_0}\!\left[\mathbb{E}_{y_i \sim \pi_i, y_j \sim \pi_j}\!\left[P_i\!\left(y_i \succ \{y_j\}_{j \neq i} \mid x\right)\right] - \tau\,\text{KL}(\pi_i(\cdot \mid x) \parallel \pi_{\text{ref}}(\cdot \mid x))\right]$$

where the key difference from Equation 7 is the subscript on $P_i$—each player evaluates preferences according to its own oracle.

**The game-theoretic consequences** (Section 3.3): when $P_i \neq P_j$, the resulting game is **general-sum** rather than constant-sum. In a constant-sum game (like the homogeneous case), one player's gain is exactly another's loss—the total value is conserved. The multiplicative weights update of Freund & Schapire (1999) has formal convergence guarantees to Nash equilibria in constant-sum games. In a general-sum game, these guarantees do not apply: different players may have different notions of what constitutes a "win," and there may be no strategy profile that simultaneously satisfies all players' equilibrium conditions. The paper explicitly acknowledges this (Section 3.3):

> "When $P_i \neq P_j$, the resulting game is general-sum and lacks the symmetry needed for formal Nash equilibrium guarantees... Consequently, the iterative framework in Eq. 10 does not have formal convergence to the Nash equilibrium in the heterogeneous case (Daskalakis et al., 2009; Hart & Mas-Colell, 2000)."

**The heterogeneous duality gap** is defined per-player rather than as a unified quantity. For player $i$ with opponents $\mathcal{O}_\pi = \{\pi_j\}_{j \neq i}$:

$$\text{DualGap}_i(\pi_i) = \max_{\pi'_i \in \Pi} J_i(\pi'_i, \mathcal{O}_\pi) - J_i(\pi_i, \mathcal{O}_\pi)$$

A policy profile is considered near a stationary point when $\max_i \text{DualGap}_i(\pi_i) \leq \epsilon$—no player has a strong unilateral incentive to deviate according to its own objective.

**The HT-MNPO loss** (Section 3.3, Equation 18) for player $i$ is:

$$L^{\text{HT}}_{i, D}(\pi_i \mid \beta, \{\lambda_j\}, \eta) = \mathbb{E}_{y, y' \sim \pi_i,\; y_w, y_l \sim \lambda_{P_i}(y, y')}\; D\!\left[\log \frac{\pi_i(y_w \mid x)}{\pi_i(y_l \mid x)} - \sum_{j \neq i} \lambda_j \log \frac{\pi_j(y_w \mid x)}{\pi_j(y_l \mid x)} \;\Big\|\; \eta \delta^\star_i\right]$$

where $\delta^\star_i$ is the target reward gap induced by player $i$'s specific reward model $r_i$, the preference sampling uses $P_i$ (player $i$'s own oracle), and the opponent mixture $\sum_{j \neq i} \lambda_j \log \frac{\pi_j}{\pi_j}$ is over the other players' policies (not historical time indices).

**Key structural differences from TD-MNPO:**

1. **Opponents are co-evolving policies, not historical snapshots.** In each iteration, all $n$ players update simultaneously. Player $i$'s opponents are the current versions of players $j \neq i$, not past versions of player $i$ itself.

2. **Each player has its own preference signal.** Player $i$'s responses are evaluated by $P_i$ (which may be, for example, a "helpfulness" reward model), while player $j$'s responses are evaluated by $P_j$ (which may be a "safety" reward model). The loss for player $i$ uses $P_i$ exclusively.

3. **The loss encourages player $i$ to satisfy its *own* oracle** while maintaining reasonable log-ratios relative to the opponent population. A player optimizing for helpfulness will produce responses that its helpfulness oracle prefers, but the presence of safety-optimizing opponents in the population prevents it from completely ignoring safety concerns—its log-ratios are still compared against opponents that encode safety preferences.

**The paper's experimental implementation uses three reward models** to simulate heterogeneous oracles (Section 4): ArmoRM-Llama3-8B-v0.1 (Wang et al., 2024a), Skywork-Reward-V2-Llama-3.1-8B (Liu et al., 2025), and Athene-RM-8B (Frick et al., 2024). Each reward model is paired with one of three co-evolving policies, and all policies are updated in parallel using Algorithm 2 (Appendix B).

**Why HT-MNPO still works empirically** (despite lacking formal guarantees): the paper argues that the algorithmic structure—each policy performing gradient descent on its own loss relative to the current opponent distribution—is "natural and principled" even in the absence of convergence proofs. The empirical results (Tables 2–4) show that HT-MNPO variants achieve strong performance, often outperforming TD-MNPO. The paper's hypothesis (implicit in the problem framing) is that the general-sum game admits good stationary points even without formal Nash equilibria, and that the iterative procedure reliably finds them in practice, though this remains an empirical claim rather than a theoretical result.

---

#### Connections to Existing RLHF Methods: A Unified Framework

One of the paper's major technical contributions is demonstrating that the TD-MNPO loss (Equation 17) recovers many existing preference optimization algorithms as special cases (Section 3.2, Table 1). This is achieved by varying four degrees of freedom: the number of players $n$, the choice of opponents, the importance weights $\lambda_j$, the distance metric $D$, and the target reward gap $\delta^\star$. The paper uses two distance metrics in its unification:

- **$D_{\text{sq}}$ (squared distance):** $D_{\text{sq}}(a, b) = (a - b)^2$, which produces a squared-error loss. This is used by methods derived from the logarithmic barrier or mirror descent perspective, including MNPO itself, SPPO, INPO, and IPO.

- **$D_{\text{bwd}}$ (backward Bernoulli KL divergence):** used by methods derived from the logistic regression perspective, including DPO, SimPO, and SPIN. This metric is not explicitly defined in the paper but is understood as the negative log-likelihood under the Bradley–Terry model: $-\log \sigma(a - b)$ for the preference direction.

**Recovering DPO** (Rafailov et al., 2023): Set $n = 2$ (two-player game), opponent $= \pi_{\text{ref}}$ (the reference policy), $\lambda_j = 1$ (full weight on the reference), distance metric $D = D_{\text{bwd}}$, and target reward gap $\delta^\star \to \infty$ (encoded as "$\infty$" in Table 1). With these choices, the TD-MNPO loss reduces to the DPO loss: the current policy's log-ratio is compared against the reference policy's log-ratio, and the backward KL divergence produces the logistic regression form $-\log \sigma(\beta \log \frac{\pi_\theta(y_w)}{\pi_{\text{ref}}(y_w)} - \beta \log \frac{\pi_\theta(y_l)}{\pi_{\text{ref}}(y_l)})$.

**Recovering INPO** (Zhang et al., 2025b): Set $n = 3$ (three-player game), opponents $= \{\pi_t, \pi_{\text{ref}}\}$ (the current iteration's policy and the reference), importance weights $\lambda_j$ set to $\tau/\eta$ for $\pi_t$ and $(\eta - \tau)/\eta$ for $\pi_{\text{ref}}$, distance metric $D = D_{\text{sq}}$, and target reward gap $\delta^\star = 1/(2\tau)$. This recovers exactly the INPO loss, which uses a weighted combination of the current policy and reference policy as opponents. The paper notes that INPO is the two-player predecessor closest to MNPO—MNPO generalizes it by expanding the opponent set beyond $\{\pi_t, \pi_{\text{ref}}\}$ to arbitrary mixtures of historical policies.

**Recovering SPPO** (Wu et al., 2024): Set $n = 2$, opponent $= \pi_t$, $\lambda_j = 1$, $D = D_{\text{sq}}$, and target reward gap $\delta^\star = \eta(\hat{P}(y \succ \pi_t \mid x) - 1/2)$ where $\hat{P}(y \succ \pi_t \mid x)$ is the estimated win rate. This shows that SPPO is MNPO with a single opponent (the previous iteration) and a win-rate-dependent target rather than a constant.

**Recovering SPIN** (Chen et al., 2024): Set $n = 2$, opponent $= \pi_t$, $\lambda_j = \beta$ (the regularization coefficient serves as the opponent weight), $D = D_{\text{bwd}}$, and $\delta^\star \to \infty$. SPIN uses the self-play structure with a single opponent from the previous iteration, framed as logistic regression.

**Recovering IPO** (Azar et al., 2024): Set $n = 2$, opponent $= \pi_{\text{ref}}$, $\lambda_j = 1$, $D = D_{\text{sq}}$, and target reward gap $\delta^\star = 1/(2\tau)$. This is the squared-loss analog of DPO—instead of logistic regression, it uses squared error to match a constant target gap.

**What this unification achieves:** it demonstrates that the distinction between "offline" and "online" preference optimization, between "reference-based" and "self-play" methods, and between different loss function forms (logistic vs. squared-error) are all special cases of choices within the MNPO framework. The multiplayer generalization is not an alternative to existing methods—it is a superset that contains them. This means that improvements to MNPO (better opponent selection, better weighting schemes, mixed distance metrics) can potentially improve all of these methods simultaneously.

**A practical implication:** the paper observes that methods using $D_{\text{sq}}$ (including MNPO) tend to be more amenable to $n > 2$ extensions because the squared error decomposes additively across opponents—each additional opponent adds a term to the sum in Equation 17 without changing the loss structure. In contrast, methods using $D_{\text{bwd}}$ (like DPO) have a logistic form $\log(1 + \exp(\cdot))$ that does not decompose cleanly across multiple reference signals. This explains why the game-theoretic Nash methods (INPO, SPPO, MNPO) all use squared-error formulations—it is the mathematical structure that supports generalization beyond pairwise comparisons.

---

#### Training and Evaluation Configuration

Although not part of the core technical innovation, the concrete training setup matters for understanding the scale and reproducibility of the results.

**Base model:** All experiments use Gemma-2-9B-it (Team et al., 2024) as the starting point. This is a 9-billion-parameter instruction-tuned model. The reference policy $\pi_{\text{ref}}$ is set to this initial model and remains frozen throughout training.

**Online RLHF framework:** The training follows the online RLHF paradigm (Dong et al., 2024), where each iteration generates fresh responses from the current policy on new prompts, rather than reusing a fixed offline dataset. This is important because it means the training data distribution shifts as the policy improves—responses from iteration $t+1$ are generated by a different policy than responses from iteration $t$, and the preference oracle must evaluate these on-policy samples.

**Training data:** The paper uses Gemma2-Ultrafeedback-Armorm (Cui et al., 2023), containing approximately 60K training samples and 2K test samples. In iteration 1, both the prompts and responses from the dataset are used. In iterations 2 and 3, only the prompts are retained—fresh responses are generated from the current policy and labeled by the preference oracle.

**Preference oracles:** For TD-MNPO, ArmoRM-Llama3-8B-v0.1 serves as the universal preference oracle. For HT-MNPO, three reward models are used simultaneously: ArmoRM-Llama3-8B-v0.1, Skywork-Reward-V2-Llama-3.1-8B, and Athene-RM-8B. All are 8-billion-parameter reward models, but they are trained with different data and architectures, inducing distinct preference oracles.

**Evaluation:** The primary benchmarks are AlpacaEval 2.0 (length-controlled win rate, %), Arena-Hard v0.1 (win rate, %), and MT-Bench (score out of 10), all evaluated using GPT-5-mini as the judge. Additionally, 11 academic benchmarks (IFEval, GPQA, MMLU, ARC, HellaSwag, TruthfulQA, Winogrande, GSM8K, Minerva-Math, AIME-24, HumanEval) assess knowledge, reasoning, math, and coding capabilities. The evaluation framework is EvalScope (Team, 2024) version 1.0.2.

**Ablation on number of players** (Appendix D, Table 5): The paper ablates $n \in \{1, 2, 3, 4\}$ in TD-MNPO. AlpacaEval 2.0 performance improves from 53.32% ($n=1$, degenerate single-player case) to 54.34% ($n=2$, +1.02), to 57.27% ($n=3$, +3.93), to 57.42% ($n=4$, +4.10). The diminishing returns beyond $n=3$ (only +0.15 from $n=3$ to $n=4$) suggest that three players capture most of the benefit of the multiplayer formulation, which is consistent with the paper's focus on the $n=3$ setting for the main TD-MNPO results.

**Ablation on 2-player vs. 3-player HT-MNPO** (Appendix D, Table 8): For each reward model in the heterogeneous setting, adding a third player (the full HT-MNPO configuration) outperforms the average over all pairwise 2-player configurations. The improvement ranges from +1.76 (Skywork-Reward-V2) to +3.93 (Athene-RM-8B), confirming that the multiplayer structure provides benefits beyond what can be achieved by running multiple independent 2-player games and averaging the results.

## 4. Key Insights and Innovations

### Innovation 1: The Single-Opponent Bias as a Diagnosable Bottleneck in Nash Learning

The paper's most conceptually distinctive move is not proposing a new algorithm, but *diagnosing* a previously unnamed structural limitation of the entire NLHF paradigm: the **single-opponent bias**. Prior to MNPO, the field's progression from RLHF to NLHF was framed as solving the Bradley–Terry transitivity problem—Munos et al. (2023) showed that by abandoning scalar rewards in favor of general preference oracles, alignment could handle non-transitive preferences that violate the standard RLHF assumptions. Subsequent methods (INPO, SPPO, ONPO, EGPO) refined the optimization dynamics, convergence guarantees, and empirical stability of this two-player game formulation. The implicit assumption across all this work was that *two players are sufficient*—if you can find a policy that is the best response to a single opponent, you have solved alignment.

MNPO identifies why this assumption breaks. The paper argues that real preference alignment involves a **mixture of annotators, heterogeneous evaluation criteria, multiple reward models, or sequences of historical checkpoints** that "cannot be reduced to a single synthetic opponent" without introducing specific pathologies. This is not merely an observation that "more opponents might help"—it is a claim that the two-player restriction actively *creates* failure modes: oscillatory behavior from overfitting to one opponent's transient weaknesses, narrow exploration from single-dimensional competitive pressure, and brittle approximations of diverse preference populations. By naming and analyzing this bottleneck, the paper reframes the NLHF research agenda: the question is no longer just "how do we solve the two-player Nash game efficiently?" but "how do we construct opponent populations that faithfully represent the preference diversity we care about?"

The significance of this diagnostic move extends beyond the specific MNPO algorithm. It provides a lens for understanding *why* iterative methods sometimes degrade between iterations (the opponent changes, and the policy was overfit to the previous one) and *why* reward model ensembling helps (it implicitly creates a population of preference signals, approximating what MNPO does explicitly). The paper's ablation on the number of players (Appendix D, Table 5) provides concrete evidence: moving from $n=1$ (degenerate) to $n=2$ (standard NLHF) to $n=3$ yields a 3.93-point improvement on AlpacaEval 2.0, while $n=4$ adds only 0.15 points more. This diminishing-returns curve is exactly what you would expect if the core problem is escaping the single-opponent bottleneck—three players capture most of the benefit because the jump from one opponent to a *population* (any $n>2$) is the qualitative change, while adding further opponents provides only incremental smoothing.

This is a **fundamental conceptual shift**, not an incremental improvement. It changes the framing of alignment from "find a policy that beats one adversary" to "find a policy that is robust to a distribution of preference signals," which connects NLHF to mean-field game theory, multi-agent robustness, and distributional robustness in ways the two-player formulation obscured.

### Innovation 2: Multiplayer Preference Optimization as a Unifying Framework for RLHF Methods

Prior to MNPO, the landscape of preference optimization methods was fragmented along several axes: offline vs. online (DPO vs. INPO/SPIN), reference-based vs. self-play (DPO vs. SPPO), logistic regression vs. squared-error loss (DPO vs. IPO), and single-iteration vs. iterative (DPO vs. DNO). Each method was justified by its own theoretical derivation—DPO from the Bradley–Terry optimal policy, INPO from no-regret game dynamics, SPPO from self-play with win-rate estimates—and the relationships between them were understood piecemeal. For instance, it was known that DPO and IPO differ primarily in their loss function (logistic vs. squared-error), and that INPO generalizes IPO by adding a second opponent, but there was no single mathematical structure that encompassed all of them.

The TD-MNPO loss (Equation 17) provides exactly this structure. By exposing four degrees of freedom—number of players $n$, opponent selection and weighting $\{\lambda_j\}$, distance metric $D$, and target reward gap $\delta^\star$—the paper shows that DPO, Distill-DPO, DNO, SPIN, SPPO, IPO, INPO, SimPO, and CPO are all special cases. Table 1 is not merely taxonomic; it is an **explanatory device** that reveals the design decisions implicit in each method. For example, DPO emerges as a two-player game with the reference policy as the sole opponent and backward Bernoulli KL divergence—it is MNPO with $n=2$ and a specific distance metric chosen for Bradley–Terry compatibility. INPO emerges as a three-player game ($\pi_t, \pi_{\text{ref}}$, and the policy itself) with squared distance—it is MNPO with $n=3$ and a different distance metric chosen for game-theoretic convergence. The fact that both are special cases of the same framework clarifies that their differences are not fundamental philosophical disagreements but rather different choices along shared axes.

This unification is a **theoretical advance of independent value**, separate from MNPO's empirical performance. It means that improvements to one axis of the MNPO framework—better opponent selection strategies, adaptive weighting schemes, principled choices of distance metric—can potentially benefit *all* methods simultaneously, because they are all instances of the same underlying structure. It also provides a design space for new methods: practitioners can now systematically explore choices of $n$, opponent mixtures, and distance metrics rather than inventing entirely new loss functions from scratch. The framework reveals, for instance, that the distinction between offline and online methods is primarily about whether the opponent set includes the current policy ($\pi_t$) or only a fixed reference ($\pi_{\text{ref}}$)—a continuum that methods like INPO already straddle by including both. This is a **fundamental reframing** that converts a fragmented literature into a coherent design space.

### Innovation 3: Reward-Aware Preference Optimization as the Mathematical Bridge Between Game-Theoretic and Reward-Based Alignment

The connection between MNPO's squared-error loss and the Reward-aware Preference Optimization (RPO) framework of Sun et al. (2025) is presented in Section 3.2 as a brief equivalence, but it represents a deeper conceptual insight: **game-theoretic Nash optimization and reward-model distillation are not competing paradigms—they are algebraically equivalent under the right loss function and opponent structure.**

Prior work largely treated these as separate approaches. Reward-based RLHF (Christiano et al., 2017; Ouyang et al., 2022) trains an explicit reward model and optimizes against it. Nash-based NLHF (Munos et al., 2023; Zhang et al., 2025b) abandons explicit rewards in favor of preference oracles and game dynamics. The two were seen as addressing different problems: reward methods work when Bradley–Terry assumptions hold, while game methods work when preferences are non-transitive. The paper's RPO connection reveals that MNPO's loss is mathematically identical to distilling a reward model whose "reward" is defined by the aggregate preferences of the opponent population—specifically, the implicit reward $r_{\pi_\theta}(x, y) = \mathbb{E}_{\pi_j}[\log \frac{\pi(y \mid x)}{\pi_j(y \mid x)}]$ with a constant target gap $\delta^\star = 1/(2\eta)$. The distance metric $D_{\text{sq}}$ used by MNPO is exactly the squared-error distillation loss from RPO.

What makes this significant is that it **dissolves the apparent tension between reward-based and game-theoretic alignment**. MNPO is simultaneously a multiplayer Nash method (by derivation from Freund & Schapire, 1999) and a reward distillation method (by equivalence to RPO with a population-defined implicit reward). The population of opponents serves as the "reward model" through their collective log-ratios. This means the convergence guarantees from the game-theoretic derivation (the $O(1/\sqrt{T})$ regret bound) apply to a reward distillation process—and conversely, the stability properties of reward distillation (smooth gradients, no adversarial dynamics) provide intuition for why the Nash game converges stably in practice. The insight is that **the opponent population is the reward model**, and improving opponent diversity is mathematically equivalent to improving reward model robustness.

This is a **fundamental conceptual bridge** between two literatures that were developing independently. It does not produce a new empirical gain by itself, but it changes how researchers should think about designing alignment methods: rather than choosing between game-theoretic and reward-based approaches, one should design opponent populations (or reward ensembles) that capture the desired preference diversity, and then the mathematical equivalence guarantees that both perspectives lead to the same update.

### Innovation 4: Homogeneous vs. Heterogeneous Preference Oracles as a Critical Boundary for Theoretical Guarantees

The paper makes an unusually honest theoretical move: it explicitly identifies and names a boundary beyond which its formal guarantees do not apply. When all players share the same preference oracle (the **homogeneous** setting, as in TD-MNPO where opponents are historical checkpoints of the same model evaluated by the same reward model), the game is constant-sum and multiplicative weights updates converge to a Nash equilibrium with $O(1/\sqrt{T})$ regret. When players have distinct preference oracles (the **heterogeneous** setting, as in HT-MNPO where three different reward models evaluate their respective policies), the game becomes general-sum and "the iterative framework in Eq. 10 does not have formal convergence to the Nash equilibrium" (Section 3.3).

Prior work in NLHF largely avoided this distinction. Methods like INPO and SPPO implicitly assumed homogeneous oracles (since there is only one preference signal in a two-player self-play loop). Extensions that considered multiple reward models typically either aggregated them into a single oracle (reintroducing scalarization) or trained separate policies per reward model (losing cross-dimension tradeoff optimization). MNPO's contribution is to **name the boundary explicitly** and then **show empirically that the heterogeneous case works well despite lacking guarantees**.

This is significant for two reasons. First, it identifies a **genuine theoretical gap**—the most practically relevant setting (multiple reward models for different quality dimensions) is precisely the one where the clean theory breaks down. This directs future theoretical work: can we prove convergence to some weaker equilibrium concept (coarse correlated equilibrium? stationary points of the gradient dynamics?) in the heterogeneous multiplayer game? Second, it provides a **pragmatic license**: the empirical results in Tables 2–4 show that HT-MNPO variants often outperform TD-MNPO (e.g., HT-MNPO with Athene-RM-8B achieves 59.64 on AlpacaEval 2.0 vs. 57.27 for TD-MNPO), even though HT-MNPO lacks the formal guarantees that TD-MNPO enjoys. This is not a failure of the theory—it is evidence that the practical benefits of heterogeneous preference signals (capturing diverse quality dimensions) outweigh the theoretical cost of operating in a general-sum game, at least for the model scales and training horizons tested.

This boundary-drawing is an **intellectually distinctive contribution** because it resists the temptation to claim universal guarantees. It tells practitioners: "If you use a single reward model with historical opponents, we can prove this converges. If you use multiple reward models, we can't prove it converges, but here's extensive evidence that it works well." This is a more useful contribution than a weaker theoretical claim that papered over the distinction, and it opens a clear research direction: developing equilibrium concepts and convergence proofs for the heterogeneous multiplayer setting.

## 5. Experimental Analysis

### Evaluation Methodology

- **Dataset.** The paper draws training data from the Gemma2-Ultrafeedback-Armorm dataset (Cui et al., 2023), containing approximately 60K training samples and 2K test samples. In iteration 1, both prompts and responses from this dataset are used. In subsequent iterations (2 and 3), only the prompts are retained—fresh responses are generated from the current policy and labeled by the preference oracle, following the online RLHF paradigm (Dong et al., 2024).

- **Base model(s).** All experiments use **Gemma-2-9B-it** (Team et al., 2024), a 9-billion-parameter instruction-tuned model. This model serves both as the initial policy to be optimized and as the frozen reference policy π_ref throughout training. The paper argues this model is representative of contemporary open-source instruction-tuned LLMs, and its 9B scale makes full online RLHF training loops computationally feasible while leaving substantial room for improvement through preference optimization.

- **Metrics.** Evaluation spans two categories. For instruction-following and preference alignment, the primary metrics are **AlpacaEval 2.0** (length-controlled win rate, %), **Arena-Hard v0.1** (win rate, %), and **MT-Bench** (score out of 10), all judged by **GPT-5-mini** (configured with reasoning effort set to "minimal"). For knowledge, reasoning, math, and coding capabilities, the paper reports standard accuracy metrics on 11 academic benchmarks: IFEval (instruction following), GPQA (graduate-level QA), MMLU (knowledge), ARC, HellaSwag, TruthfulQA, Winogrande (commonsense reasoning), GSM8K, Minerva-Math, AIME-24 (mathematical reasoning), and HumanEval (coding). GPT-5-mini is used as the judge rather than the older GPT-4 Turbo (Preview-1106), following Dubois et al. (2024)'s recommendation. The evaluation framework is EvalScope (Team, 2024) version 1.0.2.

- **Baselines.** The paper compares against several preference optimization methods trained under the same online RLHF setup with Gemma-2-9B-it as the base:
  - **SFT Model**: the initial supervised fine-tuned model without any preference optimization.
  - **DPO** (Rafailov et al., 2023): trained using the official Hugging Face DPO Trainer.
  - **SimPO** (Meng et al., 2024): following the official GitHub implementation, using a reference-free reward with length normalization.
  - **SPPO** (Wu et al., 2024): following the official GitHub implementation, using self-play with win-rate estimates.
  - **INPO** (Zhang et al., 2025b): reproduced according to the settings described in the original paper.

  Additionally, for external comparison, the paper evaluates a range of open-source models (SmolLM3-3B, Llama-3.1-8B-it, Olmo-2-32B-Instruct, Tulu-2-DPO-70B, Llama-3.1-Tulu-3-70B-DPO, Llama-3.3-70B-it, Mixtral-8x22B-it, Qwen3-235B-it) and closed-source models (Gemini-2.5-Pro, GPT-5, Claude-Sonnet-4). These serve as reference points for the absolute performance level of the 9B MNPO model relative to much larger or more capable systems.

- **Generation budget / compute accounting.** All baselines and MNPO variants are trained for T = 3 iterations using the same online RLHF framework. In each iteration, responses are generated from the current policy on a fresh prompt set, preference pairs are collected via the respective oracle(s), and the policy is updated. The generation budget is thus naturally matched across methods at the level of training iterations and response samples per iteration—all methods see the same number of preference-labeled pairs. Hyperparameter optimization is conducted separately per method, with the paper noting that optimal configurations vary across base models and across iterations of the same model. For MNPO specifically, β is maintained within [0.01, 10] and gradually increased throughout training to mitigate degradation.

- **Cross-validation / statistical protocol.** The paper does not report a formal cross-validation procedure for strategy selection. However, hyperparameters are selected through grid search: for TD-MNPO, history weights λ_j at timesteps 1 and 2 are searched over {0, 0.1, 0.333, 0.5, 0.667, 0.9}, and η is searched over {0.1, 0.01, 0.0075, 0.005, 0.002} with the final value set to η = 0.0075. For stability analysis, Appendix D (Table 7) reports mean and standard deviation of AlpacaEval 2.0 performance for several methods across three runs under three different LLM judges (GPT-4-1106-preview, GPT-4.1, GPT-5-mini), showing that MNPO achieves both higher mean scores and competitive or lower variance compared to DPO and INPO across judges.

### Main Quantitative Results

#### Instruction-Following and Preference Alignment

**Table 2** reports performance on AlpacaEval 2.0, Arena-Hard, and MT-Bench. The headline result is that **TD-MNPO achieves 57.27 on AlpacaEval 2.0 (LC), 52.26 on Arena-Hard (WR), and 7.03 on MT-Bench (score/10)**, outperforming all baseline preference optimization methods on all three benchmarks.

On AlpacaEval 2.0, the improvements over baselines are: +2.92 over DPO (54.35), +2.11 over SimPO (55.16), +1.30 over SPPO (55.97), and +1.18 over INPO (56.09). The relative ordering of baselines (INPO > SPPO > SimPO > DPO > SFT) is consistent with the expectation that game-theoretic methods outperform reward-based methods, and MNPO extends that trend further.

On Arena-Hard, the advantage is more pronounced: **TD-MNPO at 52.26 represents a 4.23-point improvement over INPO at 48.03**, the next-best preference optimization method. This gap is substantially larger than the 1.18-point advantage on AlpacaEval 2.0, suggesting that the multiplayer formulation provides particular benefit on more challenging, diverse evaluation prompts. Notably, TD-MNPO at 9B parameters surpasses much larger open-source fine-tuned models, including Tulu-2-DPO-70B (27.88) and Mixtral-8x22B-it (40.98), and approaches the performance range of Llama-3.1-Tulu-3-70B-DPO (71.34) despite an 8× parameter disadvantage. It does not match the largest models (Qwen3-235B-it at 88.71) or closed-source systems (GPT-5 at 98.11, Gemini-2.5-Pro at 86.98, Claude-Sonnet-4 at 77.58).

On MT-Bench, TD-MNPO scores 7.03, improving over INPO's 6.95 and DPO's 6.87. The absolute improvements are modest (within 0.16 points of INPO), and all preference optimization methods cluster between 6.86 and 7.03—substantially above the SFT baseline at 6.49 but with relatively small differentiation among methods. This ceiling effect on MT-Bench is consistent with the benchmark's limited score range (1-10), and the fact that even the SFT model (6.49) and Qwen3-235B-it (8.27) are separated by only 1.78 points.

**Heterogeneous MNPO variants** (HT-MNPO) are reported in the lower portion of Table 2, with three configurations corresponding to the three reward models used as heterogeneous oracles. HT-MNPO (Athene-RM-8B) achieves the highest AlpacaEval 2.0 score among all 9B models at **59.64**, outperforming TD-MNPO by 2.37 points. HT-MNPO (ArmoRM-Llama3) achieves 57.63 (+0.36 over TD-MNPO), and HT-MNPO (Skywork-Reward-V2) achieves 56.01 (−1.26 below TD-MNPO). On Arena-Hard, all three HT-MNPO variants score between 50.34 and 51.17, which is slightly below TD-MNPO's 52.26, suggesting that the heterogeneous formulation's benefits may be more benchmark-dependent or that the specific reward models chosen favor AlpacaEval 2.0's evaluation criteria more than Arena-Hard's. On MT-Bench, HT-MNPO (ArmoRM-Llama3) achieves the highest score at 7.52, substantially exceeding TD-MNPO's 7.03 and all baselines.

**Table 7** (Appendix D) provides variance estimates across three runs under three judges. MNPO's AlpacaEval 2.0 scores with GPT-5-mini show a mean of 57.20 ± 0.09, compared to INPO at 56.16 ± 0.06 and DPO at 53.98 ± 0.34. The low variance for MNPO (standard deviation 0.09) suggests the training is stable across runs, and the gap to INPO (approximately 1 point) exceeds the combined standard deviations, indicating statistical reliability at this sample size.

#### Knowledge and Reasoning Capabilities

**Table 3** reports performance on instruction following, knowledge, and commonsense reasoning benchmarks. **TD-MNPO achieves the highest average score of 71.08** across all seven benchmarks, compared to 70.68 for DPO, 70.25 for INPO, and 69.60 for SimPO.

The most notable individual result is on **GPQA**, where TD-MNPO scores **33.33**, representing a 5.55-point improvement over the next-best method (SimPO at 32.32 has an apparent typographical or reporting issue), and a 5.05-point improvement over the SFT baseline (28.28). GPQA is a graduate-level reasoning benchmark, and this substantial gain suggests that the multiplayer preference optimization does not degrade—and may actively improve—complex reasoning capabilities. INPO (27.78) and DPO (29.29) show more modest gains over the SFT baseline on this benchmark.

On IFEval (instruction following), TD-MNPO scores 73.94, which is 0.74 points above DPO (72.96), roughly comparable to SimPO (73.79), and 0.74 above INPO (73.20). On MMLU (knowledge), scores cluster tightly: SFT at 75.35, DPO at 75.77, TD-MNPO at 75.63—differences within 1 point across all methods, suggesting the knowledge benchmark is near saturation for these 9B-scale models or that preference optimization has minimal impact on factual knowledge retention.

On commonsense reasoning benchmarks, the pattern varies:
- **ARC**: all methods score between 91.07 and 91.29, with negligible differences (≤0.22 points).
- **HellaSwag**: all methods score between 80.10 and 80.44, with TD-MNPO at 80.18—no meaningful degradation or improvement from any preference optimization method.
- **TruthfulQA**: SimPO shows a notable degradation to 63.40, substantially below the SFT baseline at 70.75. TD-MNPO scores 70.26, close to the SFT baseline and substantially better than SimPO, suggesting the multiplayer formulation helps preserve truthfulness that single-opponent methods may compromise.
- **Winogrande**: all methods score between 72.93 and 73.88, with TD-MNPO at 73.09—within the cluster and close to the SFT baseline of 73.72.

The **HT-MNPO variants** show generally competitive or superior performance. HT-MNPO (Skywork-Reward-V2) achieves the highest average at 71.80, driven by strong GPQA performance (36.36, the highest across all methods including TD-MNPO) and IFEval (75.26). HT-MNPO (ArmoRM-Llama3) scores 70.83, and HT-MNPO (Athene-RM-8B) scores 70.99. All three heterogeneous variants exceed the SFT baseline average of 70.28.

#### Mathematical and Coding Performance

**Table 4** reports results on mathematical reasoning (GSM8K, Minerva-Math, AIME-24) and coding (HumanEval). **TD-MNPO achieves the highest average of 48.10** across the four benchmarks, compared to 47.33 for SPPO, 47.10 for INPO, and 46.94 for DPO.

**The most striking result is on AIME-24**, an extremely challenging mathematics benchmark: **TD-MNPO scores 3.33, while all other preference optimization methods and the SFT baseline score exactly 0.** This is the only non-zero AIME-24 score among the 9B models in Table 4. The fact that TD-MNPO alone achieves any correct answers on AIME-24—while DPO, SimPO, SPPO, and INPO all score 0—is strong evidence that the multiplayer formulation enables capabilities that two-player Nash methods cannot unlock. However, the absolute score of 3.33 is very low, and the paper does not report the number of AIME-24 problems attempted or provide confidence intervals, so this result should be interpreted as a qualitative demonstration of capability emergence rather than a reliable quantitative benchmark.

On HumanEval (coding), TD-MNPO achieves **61.59**, the highest among all 9B models, exceeding INPO (59.15), DPO (59.76), and the SFT baseline (60.37). The gap over the SFT baseline is 1.22 points, which is modest, but the consistency of improvement across the other coding/math benchmarks suggests the multiplayer formulation provides genuine benefits for structured reasoning tasks.

On GSM8K, all methods cluster between 81.96 (SFT) and 82.94 (INPO), with TD-MNPO at 82.64. The 0.68-point range across methods suggests near-saturation on this grade-school math benchmark. On Minerva-Math, scores range more widely: SFT at 44.12, SPPO at 47.43, INPO at 46.32, TD-MNPO at 44.85. TD-MNPO does not achieve the highest score on this benchmark—SPPO (47.43) and several HT-MNPO variants (47.79 for ArmoRM-Llama3, 49.63 for Skywork-Reward-V2) outperform it.

**HT-MNPO variants** on math/coding: HT-MNPO (ArmoRM-Llama3) achieves the highest average at 48.68, driven by strong Minerva-Math (47.79) and a matching AIME-24 score of 3.33. HT-MNPO (Skywork-Reward-V2) achieves the highest Minerva-Math score (49.63) but scores 0 on AIME-24. The AIME-24 performance is inconsistent across heterogeneous variants—only ArmoRM-Llama3 matches TD-MNPO's 3.33, while the other two heterogeneous reward models produce 0—suggesting that the AIME-24 capability is sensitive to the specific reward model used as the preference oracle.

### Ablation Studies and Robustness Checks

- **Number of players (n) in TD-MNPO** (Appendix D, Table 5): Ablating n ∈ {1, 2, 3, 4} on AlpacaEval 2.0 yields scores of 53.32 (n=1), 54.34 (n=2, +1.02), 57.27 (n=3, +3.93), and 57.42 (n=4, +4.10). The diminishing returns from n=3 to n=4 (+0.15) indicate that three players capture most of the benefit, with the qualitative jump occurring between n=2 (standard two-player NLHF) and n=3 (first true multiplayer setting). This supports the paper's central claim that the single-opponent bias is a real bottleneck—moving from one opponent to a population yields a large gain, while adding further opponents to an already-population setting provides only incremental smoothing.

- **2-player vs. 3-player HT-MNPO** (Appendix D, Table 8): For each reward model used as a heterogeneous oracle, the 3-player HT-MNPO setting (full configuration from Table 2) is compared against the average of all 2-player configurations involving that reward model (i.e., running MNPO as a two-player game against each of the other two reward models separately and averaging). The improvements from moving to 3-player are: +2.20 for ArmoRM-Llama3 (55.43 → 57.63), +1.76 for Skywork-Reward-V2 (54.25 → 56.01), and +3.93 for Athene-RM-8B (55.71 → 59.64). The consistent positive delta shows that the multiplayer structure provides benefits beyond simply running multiple independent 2-player games. The particularly large delta for Athene-RM-8B (almost 4 points) suggests that this reward model benefits disproportionately from having a richer opponent population, though the paper does not analyze why.

- **Different base models** (Appendix D, Table 6): On Llama-3-8B-it, the AlpacaEval 2.0 score improves from 24.80 (base) to 41.48 (INPO, +16.64) to 42.94 (TD-MNPO, +18.14). The absolute scores are lower than with Gemma-2-9B-it (53.32 n=1 and 57.27 n=3 respectively), but the relative pattern—MNPO outperforms INPO, which substantially outperforms the base model—replicates. The +1.46 gap between TD-MNPO and INPO on Llama-3-8B-it is comparable to the +1.18 gap on AlpacaEval 2.0 with Gemma-2-9B-it (Table 2), suggesting the multiplayer advantage is not specific to a single base model family.

- **Variance across runs and judges** (Appendix D, Table 7): The standard deviations for AlpacaEval 2.0 scores under GPT-5-mini are 0.09 (MNPO), 0.06 (INPO), 0.34 (DPO), and 0.18 (SFT). MNPO's low variance (0.09) indicates stable training despite the more complex multiplayer dynamics—there is no evidence that the n-player game introduces instability relative to two-player methods. Under GPT-4-1106-preview, MNPO's mean is 54.05 ± 1.58, with higher variance than under GPT-5-mini but still competitive with or better than baselines (DPO at 48.32 ± 1.26, INPO at 49.30 ± 0.95). The across-judge consistency of MNPO's advantage (4.75 points over INPO under GPT-4-1106-preview, 1.43 under GPT-4.1, 1.04 under GPT-5-mini) shows that the relative ranking is robust to judge selection, though the magnitude of the advantage varies.

- **β scheduling**: The paper notes that "gradually increasing β throughout training effectively mitigates training degradation while enabling continued model improvement" (Section 4). No explicit ablation table is provided for this claim—it appears as an empirical observation from hyperparameter tuning rather than a controlled comparison. The range [0.01, 10] is wide, and the schedule is not specified in detail, making this aspect difficult to reproduce or evaluate independently.

### Critical Assessment

#### Claim 1: "MNPO consistently outperforms existing NLHF baselines" (Section 1, abstract)

**Supported with qualifications.** The claim holds clearly for TD-MNPO on AlpacaEval 2.0, Arena-Hard, and MT-Bench (Table 2), where it outperforms DPO, SimPO, SPPO, and INPO. It also holds on the overall average across academic benchmarks (Table 3: 71.08 vs. 70.68 for DPO, 70.25 for INPO; Table 4: 48.10 vs. 47.33 for SPPO, 47.10 for INPO). However, the "consistently" qualifier requires scrutiny:

- On individual benchmarks, TD-MNPO is NOT always best: SPPO outperforms it on Minerva-Math (47.43 vs. 44.85), INPO matches or exceeds it on GSM8K (82.94 vs. 82.64), and several methods outperform it on Winogrande (73.88 for DPO vs. 73.09 for TD-MNPO). The consistency is at the aggregate level (average scores) but not at the per-benchmark level.

- On MT-Bench, the absolute improvement over INPO is 0.08 points (7.03 vs. 6.95), which is within the range of noise given that SFT itself scores 6.49 and the total scale is only 1-10. Whether this represents a genuine alignment improvement versus a near-identical score is debatable.

- The "consistently" claim for HT-MNPO is more qualified: HT-MNPO (Skywork-Reward-V2) scores 56.01 on AlpacaEval 2.0, which is below TD-MNPO's 57.27, and HT-MNPO (ArmoRM-Llama3) scores 50.93 on Arena-Hard, below TD-MNPO's 52.26. So HT-MNPO outperforms baselines but does not consistently outperform TD-MNPO itself—it depends on the specific reward model and benchmark.

#### Claim 2: "The multiplayer formulation provides significant advantages" and "extends NLHF to n-player games" (Section 1, abstract)

**Supported with evidence, but the causal mechanism is underdetermined.** The ablation on number of players (Table 5) cleanly shows that increasing n from 2 to 3 yields a 3.93-point improvement on AlpacaEval 2.0, and from 3 to 4 yields only +0.15. This demonstrates that the multiplayer formulation (n ≥ 3) outperforms the two-player formulation (n = 2) in a controlled comparison where opponent construction varies only by the number of historical checkpoints included. However, what the experiment shows is that "using more historical policies as opponents helps," not necessarily that "multiplayer game-theoretic dynamics" are the mechanism. It could equally be that averaging over more policy snapshots provides better variance reduction in the gradient estimate (a statistical benefit), or that including older checkpoints prevents overfitting to the most recent iteration's artifacts (a regularization benefit). The paper does not include an ablation that disentangles these explanations—for example, comparing an n=3 game against a two-player game with the same amount of historical averaging applied differently.

The claim that this "extends NLHF to n-player games" is true by construction—the method generalizes the game from 2 to n players—but the theoretical extension (Appendix F.3) only provides Nash equilibrium guarantees in the homogeneous case with a shared preference oracle. The practical algorithm (TD-MNPO) uses historical checkpoints that were all trained with the same oracle, so this condition holds. The theoretical contribution is valid but narrow: it extends the known convergence guarantees for multiplicative weights in constant-sum games to the specific case where the game is symmetric and all players share the same preference oracle. The paper acknowledges that the heterogeneous case (HT-MNPO) lacks these guarantees, which is honest but also means the "extension" is only partially grounded theoretically.

#### Claim 3: "MNPO subsumes many existing preference optimization algorithms as special cases" (Table 1, Section 3.2)

**Strongly supported as a conceptual contribution, weakly tested empirically.** Table 1 convincingly shows that DPO, INPO, SPPO, SPIN, IPO, DNO, SimPO, CPO, and others are recovered by specific parameter settings within the TD-MNPO loss (Equation 17). This is a mathematical derivation, not an empirical claim—it follows from the algebraic form of the loss function and does not require experimental validation. However, the paper does not empirically verify that the "special case" parameter settings actually reproduce the original algorithms' performance. For example, setting n=2, opponent=π_ref, D=D_bwd in the MNPO framework should produce something equivalent to DPO, but the paper does not run this configuration and compare it to the standard DPO implementation. Such a sanity check would strengthen confidence that the theoretical unification is not merely formal but also practically accurate.

Additionally, the unification covers only the *loss function form*, not the full training pipeline. Methods like SPIN and DNO involve specific data generation and filtering procedures that are not captured by the loss function alone. The claim that MNPO "subsumes" them should be understood as "subsumes the mathematical form of their update rule when abstracted to a common framework," not "fully reproduces each method's training dynamics."

#### Claim 4: "MNPO preserves reasoning, knowledge, and factual accuracy while achieving preference alignment" (Section 5, Tables 3-4)

**Supported with minor qualifications.** The average scores across academic benchmarks (Table 3: 71.08 TD-MNPO vs. 70.28 SFT; Table 4: 48.10 vs. 46.61) show improvements rather than degradation. This is a meaningful finding because RLHF alignment is known to sometimes degrade reasoning and factual accuracy (Ouyang et al., 2022; Dong et al., 2024). TD-MNPO not only avoids degradation but shows modest improvements, particularly on GPQA (+5.05 over SFT) and AIME-24 (+3.33 over SFT's 0). The one exception is TruthfulQA, where TD-MNPO (70.26) is slightly below SFT (70.75), though the 0.49-point difference may not be significant.

However, the claim of "preserving" capabilities should be understood in the context of a 9B model that is far from state-of-the-art on these benchmarks. A 0.80-point improvement on the average of seven benchmarks (70.28 → 71.08) is a very modest absolute gain. It is more accurate to say that MNPO does not cause the catastrophic forgetting observed with some RLHF methods (e.g., SimPO's drop to 63.40 on TruthfulQA), and that it enables small improvements on most reasoning benchmarks. Whether these improvements would persist at larger scales or with more aggressive optimization is untested.

#### Genuine Weaknesses and Missing Experiments

**Single base model scale (9B).** All experiments use Gemma-2-9B-it. The paper does not test whether the benefits of the multiplayer formulation scale with model size—would an MNPO-trained 70B model show the same relative improvements over INPO? The one cross-model-family test (Llama-3-8B-it, Table 6) is also at the 8-9B scale. The paper's claims about "scalable framework for next-generation alignment techniques" (Section 1) are aspirational rather than empirically validated at larger scales.

**Single preference oracle type (reward model scoring).** All preference signals come from 8B reward models (ArmoRM, Skywork, Athene). The paper does not test with actual human preference data, which is the setting where non-transitivity and heterogeneity are most pronounced. The reward models themselves may impose Bradley–Terry-like structures that make the preference signals more transitive than real human preferences, potentially reducing the practical benefit of the Nash formulation compared to standard RLHF.

**No comparison to reward model ensembling.** A natural baseline for the heterogeneous setting would be to ensemble the three reward models (average their scores) and train with a standard two-player Nash method. This would test whether the multiplayer game structure provides benefits beyond simply aggregating reward signals. The paper compares HT-MNPO against homogeneous TD-MNPO (using a single reward model) but not against a "single policy trained with an ensemble of reward models" baseline, which would isolate the effect of the game structure from the effect of using multiple reward signals.

**No latency or computational cost analysis.** The paper trains all methods for T=3 iterations, but multiplayer MNPO requires maintaining and computing log-probabilities against multiple opponent policies during each update. The computational overhead relative to two-player INPO or DPO is not reported. For HT-MNPO with n=3, three separate policies must be maintained and updated, which at least triples the memory and (for sequential updates) training time. The paper does not discuss whether the performance improvements justify this additional cost.

**AIME-24 result is fragile.** The 3.33 score on AIME-24 is the most dramatic result in Table 4, but it appears in only two configurations (TD-MNPO and HT-MNPO ArmoRM-Llama3) and is absent from all baselines including SFT. The paper does not report the number of AIME-24 problems, the specific problems solved, or any measure of statistical reliability (e.g., pass@k or confidence intervals). With such a low absolute score, the result could easily be due to lucky sampling rather than genuine capability improvement. The fact that two of three HT-MNPO variants score 0 on the same benchmark further suggests this result is noisy and should not be overinterpreted.

**No ablation on the distance metric D.** The paper uses squared distance D_sq for MNPO but notes that backward Bernoulli KL divergence D_bwd recovers DPO-style methods (Table 1). There is no experiment testing whether D_sq or D_bwd performs better in the multiplayer setting, or whether the choice of distance metric interacts with the number of players. This is a significant missing ablation because the paper's theoretical unification implies that the distance metric is a key design degree of freedom.

**Limited evaluation of heterogeneous game properties.** The paper claims that HT-MNPO "can find effective stationary points even without formal equilibrium guarantees" (Section 3.3), but reports no analysis of whether the training actually converges, whether the duality gap decreases over iterations, or whether the policies reach a stable profile. The only evidence is final benchmark performance, which could be achieved without any game-theoretic convergence properties—the training might simply be benefiting from multi-task learning across reward models rather than from equilibrium-seeking dynamics.

## 6. Limitations and Trade-offs

### 6.1 Heterogeneous Multiplayer Setting Lacks Formal Equilibrium Guarantees

**The assumption or constraint.** The homogeneous setting (TD-MNPO)—where all players share the same preference oracle and opponents are historical checkpoints of a single policy—admits formal Nash equilibrium convergence guarantees via the multiplicative weights update framework of Freund & Schapire (1999), with the average policy over T iterations converging to an ε-approximate Nash equilibrium with ε = O(1/√T). The paper is explicit that the heterogeneous setting (HT-MNPO) abandons these guarantees. Section 3.3 states:

> "When P_i ≠ P_j, the resulting game is general-sum and lacks the symmetry needed for formal Nash equilibrium guarantees... Consequently, the iterative framework in Eq. 10 does not have formal convergence to the Nash equilibrium in the heterogeneous case (Daskalakis et al., 2009; Hart & Mas-Colell, 2000)."

The practical algorithm (HT-MNPO, Equation 18) is described as "natural and principled" but the theoretical motivation—multiplicative weights yielding equilibrium convergence—no longer applies. The paper defines a per-player duality gap and considers a profile "near a stationary point" when max_i DualGap_i(π_i) ≤ ε, but provides no proof that the iterative procedure reaches such points, nor any bound on the rate at which it would do so if it did.

**The consequence.** The heterogeneous setting is arguably the most practically relevant configuration MNPO enables. Real-world alignment scenarios involve mixtures of annotators with different criteria, multiple reward models trained for separate quality dimensions (helpfulness, safety, truthfulness, conciseness), or distinct evaluation rubrics—precisely the "heterogeneous or even conflicting evaluators" the paper invokes to motivate HT-MNPO (Section 3.3). Yet this is the setting where the method's behavior is least understood theoretically.

A practitioner deploying HT-MNPO cannot rely on formal guarantees about where training will converge, whether it will cycle, or whether the final policy profile represents any well-defined game-theoretic solution concept. The paper offers only the empirical observation that HT-MNPO "achieves strong performance in multi-reward-model scenarios, suggesting that it can find effective stationary points" (Section 3.3). But "effective stationary points" is not a rigorous criterion: a policy could be at a poor local stationary point, or the training could drift slowly without converging, and the practitioner would have no theoretical basis for diagnosing or predicting this. In the worst case, the lack of constant-sum structure means different players could pull the joint policy distribution in incompatible directions, producing oscillatory or divergent training dynamics that only manifest at larger scales or longer training horizons than the T = 3 iterations tested in the paper.

**What evidence exists in the paper.** The paper provides no theoretical analysis of HT-MNPO convergence. The empirical results in Tables 2–4 show that HT-MNPO variants achieve strong final benchmark scores—HT-MNPO (Athene-RM-8B) reaches 59.64 on AlpacaEval 2.0, the highest among all 9B models, and HT-MNPO (ArmoRM-Llama3) reaches 7.52 on MT-Bench—but these are endpoint measurements after T = 3 iterations. There is no reporting of per-iteration duality gaps, no analysis of whether the policy profile stabilized between iterations, and no experiment testing longer training horizons to check for eventual divergence. The paper does not compare HT-MNPO against a simple baseline that trains separate policies on each reward model and averages their outputs (or ensembles the reward models into a single oracle and runs TD-MNPO), which would help isolate whether the heterogeneous game structure provides benefits beyond access to multiple reward signals.

**Mitigation status.** The paper partially acknowledges the gap by distinguishing the homogeneous and heterogeneous settings explicitly and noting that the latter lacks guarantees. Section 8 (Limitations and Future Work, Appendix G) states that "future work could explore alternative equilibrium concepts (e.g., coarse correlated equilibrium) or game structures that provide theoretical grounding for heterogeneous preference optimization." This is a clear pointer to future research but provides no immediate mitigation for practitioners. The paper also mentions an "External Opponent" variant (EO-MNPO, Appendix G.1) that uses external LLM policies as opponents, which could be interpreted as an alternative approach to heterogeneity with a knowledge-distillation interpretation, but this variant is not empirically tested and its relationship to the convergence problem is not analyzed.

### 6.2 Single Base Model Scale and Architecture Limits Generalizability Claims

**The assumption or constraint.** All experiments use exactly one base model: Gemma-2-9B-it (Team et al., 2024), a 9-billion-parameter instruction-tuned model. The one cross-model check uses Llama-3-8B-it (Appendix D, Table 6), also an 8B model, showing that MNPO outperforms INPO on that architecture as well (42.94 vs. 41.48 on AlpacaEval 2.0). But all experimental evidence for MNPO's benefits—the 4.23-point Arena-Hard improvement, the AIME-24 capability emergence, the heterogeneous formulation's strong MT-Bench scores—comes exclusively from models at the ~8-9B parameter scale. The paper's abstract claims MNPO "establishes a scalable foundation for next-generation alignment techniques" and Section 1 describes it as "a principled and scalable framework for aligning LLMs with complex, non-transitive human preferences," but "scalable" is not tested beyond a single model scale.

**The consequence.** There are several reasons why MNPO's benefits might not generalize to larger models or might change character at scale:

- **Preference oracle quality saturation.** The paper uses 8B reward models (ArmoRM, Skywork, Athene) to provide preference signals. At larger policy model scales (70B, 405B, or beyond), the policy's outputs may exceed the reward model's reliable evaluation range—the reward model, being smaller and less capable, may not discriminate effectively among high-quality responses. The paper acknowledges this dynamic in Appendix G: "As the policy model improves and its generations become consistently high-quality, distinguishing between chosen and rejected responses becomes increasingly difficult for the preference oracle." This saturation could disproportionately affect MNPO because its multiplayer structure depends on preference signals to differentiate strategies across the opponent population.

- **Computational scaling of multi-policy maintenance.** MNPO requires computing and storing log-probabilities from multiple opponent policies during each update. For HT-MNPO with n = 3 heterogeneous players, three separate policy networks must be maintained and forward-propagated. At 9B parameters, this is manageable on 8× H100 GPUs. At 70B or 405B parameters, maintaining n = 3 full policy copies (or even n = 2 for TD-MNPO with opponent history) could become prohibitively expensive—potentially requiring model parallelism across more devices or introducing unacceptable training latency. The paper reports no computational cost analysis even at the 9B scale.

- **Interaction between model scale and optimal n.** The paper's ablation (Appendix D, Table 5) shows diminishing returns from n = 3 to n = 4 at 9B scale. At larger scales, where the policy already produces more diverse and higher-quality outputs, the optimal number of opponents might differ—perhaps n = 2 suffices, or perhaps n > 3 becomes beneficial because the richer strategy space requires a larger population to cover. Without scale experiments, practitioners cannot determine whether increasing n is a worthwhile investment for their specific model size.

**What evidence exists in the paper.** The cross-model check (Table 6) shows that MNPO's advantage over INPO on Llama-3-8B-it (+1.46 on AlpacaEval 2.0) is comparable to its advantage on Gemma-2-9B-it (+1.18, Table 2). This suggests architectural robustness across two model families at similar scale, but both are 8-9B parameters. The paper provides no experiments at 1B, 70B, or any other scale. There is no analysis of how the optimal n, β schedule, or opponent weighting λ_j might vary with model size.

**Mitigation status.** Not addressed. The paper does not acknowledge model-scale generalizability as a limitation. The "scalable" characterization appears in the abstract and introduction without qualification. The absence of scale experiments is untreated in the limitations section (Appendix G), which focuses on preference oracle fidelity and binary preference signal informativeness rather than model size.

### 6.3 Preference Signals Come Exclusively from Automated Reward Models, Not Human Annotations

**The assumption or constraint.** Every experiment in the paper uses reward models (8B parameter neural networks) as preference oracles: ArmoRM-Llama3-8B-v0.1 for TD-MNPO, and Skywork-Reward-V2-Llama-3.1-8B and Athene-RM-8B as additional oracles for HT-MNPO. No experiment involves actual human preference judgments. The entire motivation for NLHF and MNPO—capturing non-transitive, heterogeneous, multi-sourced human preferences—is evaluated exclusively on preferences *simulated by reward models*.

This matters because the paper's central critique of prior work is that real human preferences violate Bradley–Terry assumptions, motivating the shift to game-theoretic formulations. But reward models, being trained to predict human preference labels via Bradley–Terry or similar objectives, may systematically differ from actual human preferences in ways that reduce the relevance of the experimental results. Specifically:

- **Reward models impose scalar structure.** A reward model outputs a single score R(x, y). Even when multiple reward models are used (the heterogeneous setting), each one is scalar-valued. The non-transitivity that MNPO is designed to handle arises from preference *cycles* that cannot be reduced to a scalar. If the reward models themselves largely respect transitivity (because they are trained to predict aggregate human judgments that smooth over individual annotator inconsistencies), the preference oracles may exhibit much less non-transitivity than real human preferences. In that case, the experiments may not test MNPO's ability to handle the very phenomenon that motivated it.

- **Distribution shift between reward model training and MNPO training.** The reward models are trained on human preference data from potentially different prompt distributions, model outputs, and annotator populations than those encountered during MNPO training. The paper uses Gemma2-Ultrafeedback-Armorm (Cui et al., 2023) as the training dataset, which contains preference labels generated by ArmoRM. This means the preference oracle and the training data are aligned by construction—a best-case scenario that may not reflect deployment with independently trained reward models or actual human annotators.

**The consequence.** A practitioner deploying MNPO with real human preference feedback—the use case the paper is ultimately targeting—faces several unknowns that the experiments do not resolve:

- **Human preference noise.** Real human annotations contain significant noise: annotator disagreement, inconsistent standards, attention lapses, and ordering effects. The multiplayer formulation's variance reduction properties (which the paper claims reduce gradient variance through opponent population averaging, Section 3) might help with noisy oracles, or might amplify noise by requiring preferences to be consistent across a population. The reward model experiments provide no signal about which direction this goes.

- **Preference sparsity and cost.** Reward models provide cheap, unlimited preference queries. Human annotations are expensive and limited. MNPO's online RLHF framework (Dong et al., 2024) generates fresh responses and queries the oracle at each iteration—T = 3 iterations with 60K training samples represents hundreds of thousands of preference queries. Replicating this with human annotators would be cost-prohibitive. The paper does not discuss how MNPO would be adapted to a limited human preference budget, or whether the benefits persist when the number of preference queries is constrained.

- **Oracle miscalibration.** Reward models have systematic biases (length bias, verbosity bias, stylistic preferences) that may differ from human biases in deployment. MNPO's population-based optimization could amplify these biases if they are consistent across opponents (e.g., if all historical checkpoints share the same length bias inherited from the reward model). The paper notes that training can suffer from "diminishing discriminative capability" of the oracle as policy quality improves (Appendix G), suggesting that oracle quality is a recognized bottleneck, but tests this only with reward models.

**What evidence exists in the paper.** The paper's experiments are entirely confined to reward-model-based oracles. There is no human evaluation, no comparison of reward model preferences to human preferences on the same prompts, and no analysis of whether the observed benefits (the 4.23-point Arena-Hard improvement, the AIME-24 capability) would replicate with human annotators. The paper acknowledges the preference oracle dependency in Appendix G: "MNPO's performance is fundamentally linked to the quality of its preference data. Three primary limitations warrant consideration..." but frames these as general RLHF limitations rather than a gap specific to the experimental validation.

**Mitigation status.** Minimally addressed. Appendix G notes that "when rejected responses are themselves of high quality, the binary preference signal becomes less informative, potentially slowing down or stalling the convergence" and suggests "more nuanced feedback mechanisms" as future work. But this is framed as a future challenge for the field, not as a limitation of the present experimental evidence. The paper does not discuss the gap between reward-model-based evaluation and the human-preference motivation, nor does it propose experiments to bridge this gap.

### 6.4 Computational Cost of Multi-Policy Training Is Not Reported or Analyzed

**The assumption or constraint.** MNPO's headline results—the 4.23-point Arena-Hard improvement, the 59.64 AlpacaEval 2.0 for HT-MNPO, the unified framework—are presented without any analysis of the computational cost required to achieve them relative to baselines. The paper reports that all experiments use 8× NVIDIA H100 GPUs (Appendix C) and that training runs for T = 3 iterations, but provides no comparison of wall-clock time, GPU-hours, memory requirements, or FLOPs between MNPO and DPO, INPO, or SPPO.

The multiplayer structure introduces several sources of additional computational overhead that are not accounted for:

- **Opponent log-probability computation.** For each training batch, the TD-MNPO loss (Equation 17) requires computing log-probabilities from up to n−1 opponent policies for each response pair. If n = 3, this means two additional forward passes per training sample beyond the forward pass for the current policy. For HT-MNPO with n = 3, three separate policies must each compute their own forward passes plus the opponent log-probabilities—a minimum of three policy evaluations per training sample for each of three players, or 9 total evaluations per sample if computed naively.

- **Opponent policy storage.** Both TD-MNPO and HT-MNPO require maintaining multiple policy copies in memory. At 9B parameters (approximately 18GB in FP16), storing n = 3 policy copies requires ~54GB just for model weights, plus optimizer states, activations, and training data. The paper uses 8× H100 GPUs with 96GB each (Appendix C), so this fits comfortably at 9B scale, but the memory multiplier scales linearly with n and with model size—a 70B model with n = 3 would require on the order of 420GB for weights alone.

- **Multiple policy updates in HT-MNPO.** Algorithm 2 (Appendix B) shows that HT-MNPO requires updating each of the n policies in sequence (or in parallel with sufficient hardware). The paper does not specify whether the three HT-MNPO policies are updated sequentially (tripling training time) or in parallel on separate GPU groups (tripling hardware requirements). Either way, the cost in GPU-hours is substantially higher than training a single policy with DPO or INPO.

**The consequence.** Without cost analysis, a practitioner cannot determine whether MNPO's performance improvements represent genuine algorithmic efficiency gains or are simply the result of spending more compute. This matters for several practical decisions:

- **Budget allocation.** If HT-MNPO with n = 3 requires 3× the GPU-hours of INPO to achieve a 2.37-point improvement on AlpacaEval 2.0 (59.64 vs. 57.27 for TD-MNPO, or ~3.5 points over INPO's 56.09), a practitioner might prefer to spend that compute budget on more INPO iterations, larger batch sizes, or a larger base model. Without cost numbers, this tradeoff cannot be evaluated.

- **Scaling to larger models.** The absence of cost analysis at 9B scale means there is no basis for projecting costs at 70B or 405B scale, where the memory and computational multipliers of multi-policy training would be much more consequential. A method that is cost-effective at 9B might be prohibitively expensive at 70B if the overhead scales poorly.

- **Fairness of comparisons.** All baselines (DPO, SimPO, SPPO, INPO) are trained for T = 3 iterations with the same online RLHF framework. If MNPO requires significantly more computation per iteration due to opponent log-probability computation, then the comparison is not FLOPs-matched—MNPO may be winning because it uses more compute, not because the multiplayer formulation is inherently better. The paper does not control for total computational cost across methods.

**What evidence exists in the paper.** Essentially none. The paper reports hyperparameter settings (Appendix C: cosine learning rate schedule, peak LR 5×10⁻⁷, batch size 128, AdamW optimizer) and hardware (8× H100 GPUs), but not training time, memory usage, or total FLOPs. There is no ablation comparing MNPO to baselines under a fixed compute budget—for instance, giving DPO or INPO additional iterations to match MNPO's total GPU-hours and measuring whether they close the performance gap.

**Mitigation status.** Not addressed. The paper does not mention computational cost as a consideration or limitation. The "Limitations and Future Work" section (Appendix G) focuses entirely on preference oracle quality and heterogeneous game theory, with no discussion of practical deployment costs. This is a significant omission for a paper that claims to provide a "practical and scalable framework."

### 6.5 HT-MNPO Benefits Are Not Disentangled from Multi-Reward Access

**The assumption or constraint.** The HT-MNPO experiments (Tables 2–4) compare three heterogeneous configurations—each using a different reward model (ArmoRM, Skywork, Athene) as the preference oracle for one player—against homogeneous TD-MNPO (which uses only ArmoRM) and against single-reward-model baselines (DPO, INPO, SPPO). The HT-MNPO configurations consistently achieve strong performance: HT-MNPO (Athene-RM-8B) reaches 59.64 on AlpacaEval 2.0 vs. 57.27 for TD-MNPO; HT-MNPO (ArmoRM-Llama3) reaches 7.52 on MT-Bench vs. 7.03 for TD-MNPO.

However, the experimental design confounds two distinct factors: (1) the **multiplayer game structure** (policies competing against multiple opponents with different oracles), and (2) **access to multiple reward models** (having preference signals from three different reward models available during training). A natural alternative baseline would be to **ensemble the three reward models** (average their scores, or take a majority vote on preferences) and use this ensembled signal to train a single policy with a standard two-player Nash method like INPO. This baseline would isolate the effect of the game structure: if HT-MNPO outperforms the ensemble baseline, the multiplayer dynamics are contributing beyond simply having richer reward information. If the ensemble baseline matches HT-MNPO, the benefit comes from multi-reward access, not from the game formulation.

**The consequence.** Without this baseline (or a similar disentanglement experiment), the paper cannot support the claim that HT-MNPO's strong performance is due to the heterogeneous game-theoretic formulation rather than simply having access to more diverse preference signals during training. The Appendix D ablation on 2-player vs. 3-player HT-MNPO (Table 8) partially addresses this—it shows that the 3-player HT-MNPO configuration outperforms the average of 2-player HT-MNPO runs—but does not compare against a single-policy ensemble-of-reward-models baseline. The 2-player HT-MNPO runs still use the heterogeneous game structure (just with n=2 instead of n=3), so they don't isolate the game structure from the reward access.

A practitioner considering HT-MNPO faces an unclear cost-benefit analysis. If the same performance can be achieved by simply averaging the scores from three reward models and training with INPO—a simpler, cheaper, and better-understood procedure—then the complexity of maintaining multiple co-evolving policies, the lack of convergence guarantees, and the increased computational cost of HT-MNPO are unjustified. The paper provides no evidence to resolve this question.

**What evidence exists in the paper.** The multi-reward comparison is implicit: HT-MNPO (ArmoRM-Llama3) uses ArmoRM as its oracle but competes against policies with Skywork and Athene oracles, and achieves 57.63 on AlpacaEval 2.0 vs. 57.27 for TD-MNPO (which uses only ArmoRM). The +0.36 improvement could be due to the heterogeneous game structure, or it could be due to the ArmoRM policy benefiting from seeing outputs that the other reward models prefer (a form of cross-reward-model knowledge transfer). The paper does not test the simpler hypothesis: just train with all three reward models' signals aggregated into a single preference oracle.

**Mitigation status.** Not addressed. The paper does not discuss ensemble-of-reward-models baselines or the confound between game structure and reward diversity. The limitations section (Appendix G) discusses preference oracle quality and heterogeneous game theory but not the experimental design gap. This is a notable omission because the paper's theoretical framework (the RPO connection, Section 3.2) suggests a natural alternative interpretation: the opponent population implicitly defines an ensembled reward signal through the weighted log-ratios. Testing whether this implicit ensemble outperforms an explicit ensemble would directly test whether the game structure provides value beyond reward aggregation.

### 6.6 Evaluation Scope Is Limited to Single-Turn Instruction-Following with Automated Judges

**The assumption or constraint.** All evaluation benchmarks (AlpacaEval 2.0, Arena-Hard, MT-Bench) and all academic capability benchmarks (IFEval, GPQA, MMLU, ARC, HellaSwag, TruthfulQA, Winogrande, GSM8K, Minerva-Math, AIME-24, HumanEval) are single-turn, English-language tasks with closed-form or reference-based evaluation. MT-Bench and AlpacaEval 2.0 involve multi-turn conversations, but the evaluation is still based on single-turn quality as judged by GPT-5-mini. The paper does not test on:

- **Multi-turn dialogue or interactive settings**, where preference alignment involves maintaining coherence, memory, and consistency across turns. The multiplayer dynamics might interact differently with multi-turn generation, where earlier-turn errors compound.

- **Open-ended generation tasks** without clear correctness criteria, such as creative writing, summarization quality, or explanation quality. These domains are where human preferences are most likely to exhibit the non-transitivity and heterogeneity that motivates MNPO, yet they are entirely absent from the evaluation.

- **Safety-critical or harm-reduction scenarios.** The paper uses reward models that may not encode safety constraints, and the benchmarks do not measure harmful outputs, refusal quality, or robustness to adversarial prompts. A method that optimizes aggressively against opponent populations could potentially find adversarial strategies that exploit weaknesses in the preference oracle without producing genuinely safer outputs.

- **Non-English languages or multilingual settings**, where preference structures may differ culturally and where reward models may be less reliable.

**The consequence.** The paper's central motivation—that real-world preferences are diverse, non-transitive, and multi-sourced—implies that the benefits of the multiplayer formulation should be most pronounced in settings with genuine preference diversity and ambiguity. Single-turn instruction-following benchmarks with automated LLM judges may be the setting where preferences are *most* transitive and *least* heterogeneous, since LLM judges tend to impose consistent, scalar-like evaluation criteria. If this is the case, the experimental results may represent a lower bound on MNPO's benefits (the method helps even when preferences are relatively well-behaved) or an upper bound (the method's advantages are overstated because the benchmarks don't expose its limitations).

Additionally, the reliance on GPT-5-mini as the sole judge for the main instruction-following benchmarks creates a potential confound: GPT-5-mini's preferences may correlate with the reward model preferences used during training (since both are derived from similar training paradigms and human preference data), making the evaluation less independent than it appears. The multi-judge analysis in Appendix D (Table 7) shows that MNPO's advantage varies substantially across judges—4.75 points over INPO under GPT-4-1106-preview vs. 1.04 points under GPT-5-mini—indicating that the magnitude of the measured improvement is judge-dependent. The paper's argument that GPT-5-mini is superior for evaluation (following Dubois et al., 2024) is reasonable, but using the same model family for evaluation that dominates the RLHF ecosystem creates a circularity risk that is not discussed.

**What evidence exists in the paper.** The evaluation is comprehensive within its scope—three instruction-following benchmarks plus 11 academic benchmarks, with multiple judge models tested in Appendix D—but the scope itself is narrow. There are no experiments on multi-turn dialogue, safety, creative generation, or non-English tasks. The paper does not claim results beyond the evaluated domains, so this is an omission rather than an overclaim, but it means the paper's title claim of aligning "LLMs with complex, non-transitive human preferences" is tested only on tasks where the complexity and non-transitivity of preferences is simulated by reward models and evaluated by LLM judges.

**Mitigation status.** Not addressed as a limitation. The paper's limitations section (Appendix G) focuses on preference oracle fidelity and heterogeneous game theory, with no discussion of evaluation scope. The paper acknowledges in Appendix G that "future research could explore more nuanced feedback mechanisms to address learning in this high-performance regime," which partially gestures toward more complex evaluation, but does not specifically identify the gap between the evaluated domains and the motivating use cases.

## 7. Implications and Future Directions
- Field impact
  - Recasting preference alignment as a multiplayer game broadens the alignment toolkit beyond two‑player dynamics. It encourages modeling populations of preferences—annotators, domains, or teacher models—as explicit opponents.
  - The TD‑MNPO lens provides a unifying view of the preference‑optimization landscape (Table 1), likely simplifying comparison, transfer of techniques, and hybrid designs.

- What this enables
  - Multi‑annotator alignment: Simultaneously align to diverse preference clusters by treating each as an opponent population.
  - Multi‑domain or multi‑skill training: Use external domain experts (Appendix F.1; Eq. 20) as opponents to distill strengths from multiple specialized models (Eq. 21).
  - Stability improvements: Historical‑mixture opponents can yield smoother and safer online preference optimization.

- Practical applications
  - Instruction‑tuned assistants evaluated by diverse users (robustness to style/length preferences).
  - Systems requiring balanced performance across reasoning, knowledge, and coding, where single‑opponent tuning can overfit.

- Research directions
  - Opponent selection and weighting: Learn `{λ_j}` and the opponent set adaptively (who to play, how often).
  - Human‑in‑the‑loop MNPO: Replace or complement reward models with stratified human preference panels to better capture heterogeneity.
  - Convergence and stability: Extend theory toward last‑iterate convergence in stochastic multiplayer settings, and analyze the effect of KL strength and support constraints.
  - Safety and value pluralism: Encode safety reviewers and value groups as dedicated opponents to shape safer, more pluralistic behaviors.

In sum, MNPO contributes a principled, extensible framework for aligning large language models under complex, heterogeneous preferences, with both theoretical foundations (Eqs. 8–11; Lemma 1; Proposition 1; Appendix E.3) and empirical gains across instruction following and reasoning tasks (Tables 2–4).
