# Finite-time Analysis of the Multiarmed Bandit Problem

**URL:** [https://homes.di.unimi.it/~cesabian/Pubblicazioni/ml-02.pdf](https://homes.di.unimi.it/~cesabian/Pubblicazioni/ml-02.pdf)

## 🎯 Pitch

This paper introduces and analyzes simple, computationally efficient policies for the multi-armed bandit problem — the fundamental exploration-exploitation dilemma central to reinforcement learning — that achieve **logarithmic regret uniformly over time** rather than only asymptotically.

---

## 1. Executive Summary

This paper introduces and analyzes simple, computationally efficient policies for the multi-armed bandit problem — the fundamental exploration-exploitation dilemma central to reinforcement learning — that achieve **logarithmic regret uniformly over time** rather than only asymptotically. The work develops three named policies: **UCB1**, an index-based policy derived from Agrawal's framework where each machine's index is its average reward plus a confidence bound based on Chernoff-Hoeffding inequalities (operationalized as playing the machine with the highest upper confidence index); **UCB2**, a more complex epoch-based variant that brings the regret constant arbitrarily close to the optimal 1/(2∆_i^2) by dividing plays into epochs with exponentially growing lengths (operationalized as selecting machines via an index that accounts for the size of the confidence interval at the current epoch); and **ε_n-GREEDY**, a randomized policy that is a dynamic variant of the ε-greedy heuristic with an exploration probability decaying as cK/(d²n) (operationalized as randomly exploring with probability ε_n otherwise exploiting). The headline result establishes that UCB1 achieves expected regret bounded above by a sum of terms of the form 8 ln n / ∆_i plus a small constant — specifically, at most `[8 Σ (ln n / ∆_i)] + (1 + π²/3) Σ ∆_j` — while UCB2 brings the leading constant arbitrarily close to the information-theoretically optimal 1/(2∆_i^2), with both policies achieving this logarithmic regret for all reward distributions with bounded support in [0,1] and without any preliminary knowledge of the distributions. The ε_n-GREEDY policy yields a stronger instantaneous regret bound of order c/(d²n) + o(1/n), establishing that suboptimal machine selection probability decays polynomially in the number of plays — but only when a lower bound d ≤ min_{i: µ_i < µ*} ∆_i on the reward gap is known to the algorithm.

## 2. Context and Motivation

### The Core Problem: Logarithmic Regret Everywhere, Not Just in the Limit

The multi-armed bandit problem is the canonical formalization of the exploration-exploitation dilemma: an agent faces K gambling machines ("one-armed bandits"), each with an unknown reward distribution, and must decide which arms to play to maximize total reward over time. Every time the agent plays a suboptimal arm, it incurs **regret** — the expected loss relative to always playing the best arm. The central theoretical question is: *how fast can regret grow as a function of the number of plays n, and what policies achieve the best possible growth rate?*

Lai and Robbins (1985) established the foundational result: regret must grow at least logarithmically in n for any policy, and they constructed policies achieving precisely this asymptotic lower bound. Specifically, they showed that for any suboptimal machine j,

$$IE[T_j(n)] \geq \left(\frac{1}{D(p_j \| p^*)} + o(1)\right) \ln n$$

where $D(p_j \| p^*)$ is the Kullback-Leibler divergence between the reward density of machine j and that of the optimal machine. This is the information-theoretic gold standard: the optimal machine is played exponentially more often than any suboptimal one, and the constant $1/D(p_j \| p^*)$ cannot be improved upon asymptotically.

However, **asymptotic optimality leaves a critical gap**. The Lai and Robbins policies, and the subsequent refinements by Agrawal (1995) and others, guarantee that regret behaves like $c\ln n$ in the limit as $n \to \infty$, but they say nothing about what happens at finite times. Formally, the $o(1)$ term in Equation (1) vanishes as $n \to \infty$, but for any specific $n$ it could be large — potentially dominating the logarithmic term entirely for practical horizons. A policy could be asymptotically optimal yet perform terribly for the first million plays, and existing theory provided no tools to guarantee otherwise.

The authors state this motivation explicitly in the introduction: "we strengthen previous results by showing policies that achieve logarithmic regret **uniformly over time**, rather than only asymptotically." The emphasis on uniformity — regret bounded by an explicit logarithmic function plus a concrete constant for every $n$, not just in the limit — is the core gap this paper fills.

### Why This Matters: Theory Meets Practical Deployment

The gap between asymptotic and finite-time guarantees has both theoretical and practical significance.

**Theoretical significance.** In the classical bandit literature, the regret analysis of Lai and Robbins (1985) relies on the computation of **upper confidence indices** for each machine. These indices are functions of the entire sequence of past rewards from that machine — not just summary statistics like the sample mean. Computing them exactly is "generally hard" (as the paper notes in Section 1), making the policies theoretically elegant but computationally expensive. Agrawal (1995) simplified this dramatically by introducing indices that depend only on the total reward obtained so far, making the policies "much easier to compute," but his regret bounds retained the asymptotic $o(1)$ term. There was no rigorous understanding of whether the simplicity of Agrawal-style indices came at the cost of poor finite-time behavior, or whether the asymptotic optimality actually kicked in at reasonable timescales.

**Practical significance for AI and reinforcement learning.** The authors explicitly position the multi-armed bandit as "fundamental in different areas of artificial intelligence, such as reinforcement learning (Sutton & Barto, 1998) and evolutionary programming (Holland, 1992)." In reinforcement learning, exploration strategies are often heuristic — ε-greedy, Boltzmann exploration, optimistic initialization — and practitioners choose among them based on empirical performance rather than theoretical guarantees. A policy with a proven finite-time regret bound provides a principled default: deploy UCB1, and you are guaranteed that regret after $n$ plays will not exceed a specific, computable bound involving only the suboptimality gaps $\Delta_i = \mu^* - \mu_i$ and the logarithm of $n$. This transforms exploration from an art to an engineering decision with predictable worst-case behavior.

The practical gap is particularly acute for the **ε-greedy** heuristic. As the authors observe, the standard ε-greedy rule — play the empirically best arm with probability $1-\varepsilon$, explore uniformly with probability $\varepsilon$ — causes "a linear (rather than logarithmic) growth in the regret" because the constant exploration probability $\varepsilon$ keeps pulling suboptimal arms forever. The "obvious fix" is to decay $\varepsilon$ over time, but *how fast?* The paper provides the answer: a rate of $1/n$ is sufficient for logarithmic regret. Without a finite-time analysis, a practitioner cannot know whether a particular decay schedule (e.g., $\varepsilon_n = 1/\sqrt{n}$) is too aggressive or too conservative. The paper's Theorem 3 provides an explicit, finite-time bound for the $\varepsilon_n$-GREEDY policy with $\varepsilon_n = cK/(d^2 n)$, giving practitioners a theoretically grounded decay schedule.

**The bounded support assumption: generality without distributional knowledge.** A crucial practical consideration is what the policy needs to know about the reward distributions. The Lai and Robbins framework assumes specific parametric families (indexed by a single real parameter), enabling the computation of Kullback-Leibler divergences. The UCB policies in this paper require only that the rewards have **bounded support** (normalized to $[0,1]$) — no other distributional assumptions, no parametric form, no knowledge of variances. This is important for real-world applications where reward distributions are unknown and potentially non-stationary in complex ways. The authors emphasize that the policies work "without any preliminary knowledge about the reward distributions (apart from the fact that their support is in $[0,1]$)."

### Where Prior Approaches Fall Short

The paper identifies specific limitations in the existing literature along several dimensions.

**1. Asymptotic analysis without finite-time constants.** Lai and Robbins (1985) proved that regret grows as $(1/D(p_j \| p^*) + o(1)) \ln n$, where $o(1)$ is an unspecified function vanishing at infinity. This means that for any finite $n$, the actual regret could be $(1/D(p_j \| p^*) + 1000) \ln n$ — the asymptotic constant tells you nothing about the additive overhead. A policymaker deciding between two exploration strategies for a finite-horizon problem (say, $n = 10,000$ plays) cannot use the Lai-Robbins result to make a quantitative comparison. The finite-time analysis in this paper replaces the unknown $o(1)$ with explicit, computable constants: for UCB1, the constant is $1 + \pi^2/3$; for UCB2, the constant $c_\alpha$ is defined explicitly in Equation (18) and can be computed for any choice of the parameter $\alpha$.

**2. Computational intractability of optimal indices.** The Lai and Robbins upper confidence indices depend on the full reward history for each arm through complex functions involving Kullback-Leibler divergences. The authors characterize this as "generally hard" to compute. In practice, this meant the theoretically optimal policies were rarely implemented outside of small-scale simulations. Agrawal (1995) addressed this by introducing sample-mean-based indices that are trivial to compute — the index for arm $i$ after $s$ plays is simply $\bar{X}_{i,s} + c(n,s)$ for some exploration bonus $c(n,s)$ — but his analysis remained asymptotic. This paper adopts Agrawal's computational philosophy while providing finite-time guarantees, yielding policies that are simultaneously "simple to implement and computationally efficient" (Section 1).

**3. No unified analysis of different exploration paradigms.** The literature contained two fundamentally different approaches to exploration: **optimistic index policies** (Lai & Robbins, Agrawal) that play the arm with the highest upper confidence bound, and **randomized exploration** (ε-greedy, Boltzmann exploration) that explicitly randomize between exploration and exploitation. These were developed in separate literatures with different analytical tools. The paper provides a unified finite-time analysis covering both paradigms: UCB1 and UCB2 represent the optimistic approach, while $\varepsilon_n$-GREEDY represents randomized exploration. This unified treatment enables direct theoretical comparison: the authors show that both approaches achieve logarithmic regret, but with different dependencies on prior knowledge (UCB policies require no knowledge of the gaps $\Delta_i$, while $\varepsilon_n$-GREEDY requires a lower bound $d \leq \min_i \Delta_i$).

**4. The normal rewards case: a surprising gap.** Section 1 notes that "surprisingly, we could not find in the literature regret bounds (not even asymptotical) for the case when both the mean and the variance of the reward distributions are unknown." For normally distributed rewards — arguably the most natural continuous distribution — no existing policy had proven logarithmic regret when variances were unknown. The paper fills this gap with UCB1-NORMAL (Theorem 4), which uses the sample variance to construct confidence intervals and achieves logarithmic regret with a leading constant of $256 \sigma_i^2 / \Delta_i^2$. This is weaker than the $1/(2\Delta_i^2)$ achievable with known variance, but it closes a theoretical hole that had persisted since the bandit problem was first formalized.

**5. Robustness to dependencies.** A subtle but practically important limitation of most prior work is the assumption that rewards are independent across arms (i.e., $X_{i,s}$ and $X_{j,t}$ are independent for $i \neq j$). In many real-world settings — A/B testing, clinical trials where patients receive different treatments, adaptive routing in networks — the rewards of different arms may be correlated due to shared environmental factors. The paper notes (end of Section 2) that Theorems 1–3 hold under the weaker assumption that $\mathbb{E}[X_{i,t} \mid X_{i,1},\ldots,X_{i,t-1}] = \mu_i$ — only the conditional mean must be constant, not the full distribution. Rewards can be dependent across arms and non-stationary in higher moments, as long as the martingale property holds. This significantly broadens the practical applicability of the bounds.

### How This Paper Positions Itself

The paper does not claim to improve the asymptotic regret constant — Lai and Robbins already established the information-theoretic lower bound, and Agrawal's policies achieve it. Instead, the positioning is that **finite-time, non-asymptotic analysis with explicit constants is a distinct and valuable contribution** that enables:

- **Provable performance guarantees from the very first play**, not just in the limit.
- **Practical guidance for parameter selection**: if you set the exploration parameter $c$ in $\varepsilon_n$-GREEDY too small, "the regret grows linearly (exponentially in the semi-logarithmic plot)"; if too large, "the regret grows logarithmically, but with a large leading constant." The finite-time bound quantifies this tradeoff explicitly.
- **Head-to-head comparison of policies** at finite horizons: the experiments in Section 4 compare UCB1-TUNED, UCB2, and $\varepsilon_n$-GREEDY on seven different Bernoulli reward distributions over the first 100,000 plays — precisely the regime where asymptotic analysis is silent.

The paper explicitly connects to the broader reinforcement learning literature through its opening sentence, framing the exploration-exploitation dilemma as a fundamental challenge for "reinforcement learning policies." By providing policies that are simple, efficient, and come with finite-time guarantees, the authors aim to influence how exploration is implemented in practical RL systems — not just how it is analyzed in theory. The extension in Section 5 to non-stationary reward processes (Gittins indices, Markovian reward processes) signals an ambition to push finite-time analysis beyond the basic stationary bandit toward problems where "preliminary knowledge about the reward processes" is typically required, and where "there are no finite-time regret bounds shown" for existing learning-based solutions.

## 3. Technical Approach

### 3.1 Reader Orientation

This paper develops and analyzes **three specific algorithms** (named UCB1, UCB2, and ε_n-GREEDY) that decide which arm to pull next in a multi-armed bandit problem, where the only feedback is the numerical reward received after each pull, and the goal is to maximize cumulative reward over a finite number of plays. The core problem solved is **the exploration-exploitation trade-off under finite-time guarantees**: the algorithms must balance trying unfamiliar arms (to discover which is best) against sticking with the empirically best arm (to accumulate reward), and the paper proves that each algorithm keeps the expected loss from suboptimal pulls — the **regret** — bounded by an explicit logarithmic function of the number of plays, with concrete constants that hold for every time step, not just in the asymptotic limit.

### 3.2 Big-Picture Architecture (Diagram in Words)

The system consists of four interacting components that operate in a closed loop:

1. **The bandit environment**: K arms, each producing rewards drawn from an unknown distribution with support in [0,1] (or normally distributed with unknown mean and variance for UCB1-NORMAL). The environment is stationary — each arm's distribution does not change over time — but rewards may be dependent across arms (the martingale assumption suffices for UCB1, UCB2, and ε_n-GREEDY).

2. **The reward history**: For each arm i, the algorithm maintains a running record of how many times that arm has been played (call it `s`) and the sequence of rewards observed from it. From this history, it computes summary statistics — at minimum, the sample average `X̄_{i,s}` — that serve as inputs to the decision rule.

3. **The index computation or exploration probability**: At each time step, the algorithm either (a) computes an **upper confidence index** for every arm (UCB1, UCB2), which is the sum of the arm's current sample mean plus an exploration bonus that grows with the total number of plays `t` and shrinks with the number of times that specific arm has been played `s`, or (b) draws a random coin to decide between uniform exploration and greedy exploitation (ε_n-GREEDY), with the exploration probability decaying as `cK/(d²n)`.

4. **The selection rule**: Play the arm with the highest computed index (optimistic policies), or play the arm chosen by the randomized exploration/exploitation decision (ε_n-GREEDY). Then observe the reward, update the history, and repeat.

Information flows as follows: at time `t`, the algorithm examines the reward history → computes indices (or an exploration probability) for each arm → selects one arm → receives a reward → updates the history for that arm → advances to time `t+1`. The regret at time `n` is computed post-hoc as `Σ_{j: μ_j < μ*} Δ_j · 𝔼[T_j(n)]`, where `Δ_j = μ* - μ_j` is the suboptimality gap of arm `j`.

### 3.3 Roadmap for the Deep Dive

- **First**, the formal definition of regret and the notation that all subsequent analysis depends on, since every theorem's bound is a statement about `𝔼[T_j(n)]` — the expected number of times each suboptimal arm is played.
- **Second**, the policy UCB1, which is the simplest and most broadly applicable algorithm, to establish the core mechanism of upper confidence bound construction from Chernoff-Hoeffding inequalities.
- **Third**, the policy UCB2, which refines UCB1's epoch structure to drive the leading constant arbitrarily close to the information-theoretic optimum `1/(2Δ²_j)`, introducing the idea of trading exploration bonus sharpness against constant overhead.
- **Fourth**, the randomized policy ε_n-GREEDY, which takes a fundamentally different approach — explicit randomization rather than optimism — and yields a stronger *instantaneous* regret bound (probability of choosing a suboptimal arm at time `n`) rather than just an integrated expected count.
- **Fifth**, the special case UCB1-NORMAL for normally distributed rewards with unknown variances, which extends the UCB1 template to handle the additional uncertainty from estimated variance using the Student and χ² distributions.
- **Sixth**, the experimental methodology and the empirically-tuned variant UCB1-TUNED, which replaces the theoretically-pure exploration bonus with a variance-aware heuristic that performs substantially better in practice.

### 3.4 Detailed, Sentence-Based Technical Breakdown

This is primarily a **theoretical analysis paper** that constructs algorithms and proves finite-time regret bounds for them. The core idea is that **optimism in the face of uncertainty** — always playing the arm whose upper confidence bound is highest — combined with a specific functional form for that bound (sample mean plus a term derived from concentration inequalities) yields logarithmic regret with explicit constants that hold for every `n`, not just asymptotically.

---

#### The Formal Regret Framework and Notation

Before presenting any algorithm, the paper establishes a rigorous notational framework that all subsequent proofs depend on. Understanding this notation is essential because every theorem's regret bound is expressed in these terms.

The environment consists of `K ≥ 1` arms. For each arm `i ∈ {1, …, K}`, there is an infinite sequence of random variables `X_{i,1}, X_{i,2}, …` representing the rewards obtained on the 1st, 2nd, 3rd, … play of that arm. These rewards satisfy:

- For each fixed `i`, the sequence `X_{i,1}, X_{i,2}, …` is independent and identically distributed (i.i.d.) according to some unknown distribution with unknown expectation `μ_i`.
- Rewards across *different* arms `i ≠ j` are independent (for the main theorems), or at least satisfy the martingale property `𝔼[X_{i,t} | X_{i,1}, …, X_{i,t-1}] = μ_i` (the weaker assumption noted at the end of Section 2 that also applies to Theorems 1–3).

The optimal arm, denoted with a superscript `*`, is any arm `i` achieving `μ_i = μ*` where:

$$\mu^* \stackrel{\text{def}}{=} \max_{1 \leq i \leq K} \mu_i$$

The **suboptimality gap** for arm `j` is:

$$\Delta_j \stackrel{\text{def}}{=} \mu^* - \mu_j$$

This is the expected regret incurred per play of arm `j` — every time you pull arm `j` instead of the optimal arm, you lose `Δ_j` in expectation.

For a given policy `A`, let `T_i(n)` be the number of times arm `i` has been played during the first `n` plays. The regret after `n` plays is then:

$$\text{Regret}(n) = \mu^* n - \sum_{j=1}^{K} \mu_j \cdot \mathbb{E}[T_j(n)]$$

> where `μ* n` is the total expected reward if the optimal arm were played `n` times, and `Σ μ_j · 𝔼[T_j(n)]` is the expected total reward actually obtained under policy `A`.

**What it computes:** the expected total lost reward due to not always playing the best arm. The subtraction is `(best possible reward) — (actual expected reward)`.

**Why this form:** the regret decomposes linearly across arms because rewards are additive. Since `Σ_{j=1}^K T_j(n) = n` (you play exactly one arm per time step), the regret can be rewritten purely in terms of the suboptimal arms only:

$$\text{Regret}(n) = \sum_{j: \mu_j < \mu^*} \Delta_j \cdot \mathbb{E}[T_j(n)]$$

This decomposition is the workhorse of the entire paper. Every theorem bounds regret by bounding each `𝔼[T_j(n)]` individually for each suboptimal arm `j`. The factor `Δ_j` is the cost per pull, and `𝔼[T_j(n)]` is the expected number of pulls — so the regret is the sum over suboptimal arms of (cost per pull) × (expected pulls). The key analytical challenge is showing that `𝔼[T_j(n)]` grows only logarithmically in `n`, with a constant inversely proportional to `Δ²_j`.

**Additional notation:** For each arm `i` and any `n ≥ 1`, define the sample mean of the first `n` plays of that arm:

$$\bar{X}_{i,n} = \frac{1}{n} \sum_{t=1}^{n} X_{i,t}$$

The policy observes `T_i(t-1)` plays of arm `i` before time `t`, so the available sample mean at decision time `t` for arm `i` is `X̄_{i, T_i(t-1)}`. The random variable `I_t ∈ {1, …, K}` denotes which machine is played at time `t`.

---

#### Policy UCB1: The Basic Index-Based Algorithm

UCB1 is the foundational algorithm of the paper, described in Figure 1, and its analysis in Theorem 1 establishes the proof template that UCB2 and UCB1-NORMAL extend. The algorithm is deterministic and index-based.

**Initialization.** During the first `K` plays, UCB1 plays each arm exactly once (t = 1, 2, …, K). This guarantees that every arm has at least one observation before any index-based decisions are made, avoiding division by zero in the exploration bonus.

**Ongoing decision rule (for t > K).** At time `t`, let `s = T_i(t-1)` be the number of times arm `i` has been played so far. The algorithm computes an **index** for each arm `i`:

$$I_i(t) = \bar{X}_{i, T_i(t-1)} + \sqrt{\frac{2 \ln t}{T_i(t-1)}}$$

> where `X̄_{i, T_i(t-1)}` is the sample mean of arm `i` from its plays so far, `t` is the current total number of plays, and `T_i(t-1)` is the number of times arm `i` has been played (with the convention that `T_i(t-1) > 0` because of the initialization phase).

**What it computes:** the index has two additive terms. The first term `X̄_{i,s}` is the current *estimate* of arm `i`'s expected reward — this is exploitation: all else equal, prefer arms with higher observed averages. The second term `√(2 ln t / s)` is the *exploration bonus* or *confidence radius* — it represents the uncertainty about arm `i`'s true mean based on how many times it has been played. This term is large when `s` is small (the arm is underexplored) and grows slowly (logarithmically) with total time `t`. The algorithm then plays `I_{t+1} = argmax_i I_i(t)`, breaking ties arbitrarily.

**Why this form — the Chernoff-Hoeffding foundation.** The exploration bonus is derived from **Fact 1 (Chernoff-Hoeffding bound)** stated in the paper. Fact 1 says: if `X_1, …, X_n` are i.i.d. in `[0,1]` with mean `μ`, and `S_n = X_1 + … + X_n`, then for any `a ≥ 0`:

$$\mathbb{P}\{S_n \geq n\mu + a\} \leq e^{-2a^2/n}$$

and symmetrically for the lower tail. This bound on the deviation of sums from their expectation is the key to constructing confidence intervals.

To see how this becomes the exploration bonus: suppose we want a confidence interval for `μ_i` that holds simultaneously for all `t` and all possible `s` with probability at least `1 — 1/t⁴`. We set `a = √(2n ln t)`, which gives `a²/n = 2 ln t` and therefore `e^{-2a²/n} = e^{-4 ln t} = t^{-4}`. The half-width of the confidence interval is `a/n = √(2 ln t / n)`. When `n = s = T_i(t-1)`, this half-width is exactly `√(2 ln t / s)`. So the index `X̄_{i,s} + √(2 ln t / s)` is the upper endpoint of a `(1 — t⁻⁴)`-confidence interval for `μ_i`.

**The selection rule in plain language.** UCB1 always plays the arm with the most optimistic plausible mean — the arm whose confidence interval extends highest. This is the principle of **optimism in the face of uncertainty**: an arm is either promising because its sample mean is high (exploitation), or because it hasn't been played enough to rule out that it might be great (exploration). The logarithmic growth of `ln t` ensures that the exploration bonus never vanishes but grows slowly enough that the *total* regret from exploration is logarithmic rather than linear.

**Regret bound (Theorem 1).** For any arm `j` with `μ_j < μ*`, setting the threshold `ℓ = ⌈(8 ln n)/Δ²_j⌉`, the expected number of plays of arm `j` satisfies:

$$\mathbb{E}[T_j(n)] \leq \frac{8 \ln n}{\Delta^2_j} + 1 + \frac{\pi^2}{3}$$

Summing over suboptimal arms with the decomposition `Regret(n) = Σ Δ_j · 𝔼[T_j(n)]` gives the total regret bound:

$$\text{Regret}(n) \leq \left[8 \sum_{j: \mu_j < \mu^*} \frac{\ln n}{\Delta_j}\right] + \left(1 + \frac{\pi^2}{3}\right) \sum_{j=1}^{K} \Delta_j$$

**What this bound means operationally.** For a 2-armed bandit where the optimal arm has mean 0.9 and the suboptimal arm has mean 0.6 (so `Δ = 0.3`), after `n = 10,000` plays, the bound says the expected number of suboptimal pulls is at most `(8 × ln(10000)) / (0.3)² + (1 + π²/3) ≈ (8 × 9.21) / 0.09 + 4.29 ≈ 818.7 + 4.29 ≈ 823`. The expected regret is `Δ × 823 ≈ 247`. This is a *guarantee*, not an estimate — the actual regret could be lower, but it cannot be higher than this bound under the assumptions.

**The proof structure (Section 3, proof of Theorem 1).** The proof is worth understanding because it establishes the analytical template for all subsequent theorems. Let `c_{t,s} = √(2 ln t / s)`. The key insight is to decompose `T_j(n)`, the number of times suboptimal arm `j` is played during the first `n` rounds, by considering each time `t` where the index of arm `j` exceeded the index of the optimal arm `*`. The authors bound this event through a union bound over three possible causes:

1. **The optimal arm's sample mean is too low (underestimation):** `X̄*_s ≤ μ* — c_{t,s}`. This occurs with probability at most `t⁻⁴` by the Chernoff-Hoeffding bound.
2. **The suboptimal arm's sample mean is too high (overestimation):** `X̄_{j, s_j} ≥ μ_j + c_{t,s_j}`. Also probability at most `t⁻⁴`.
3. **The suboptimal arm hasn't been played enough yet:** `μ* < μ_j + 2c_{t,s_j}`. For `s_j ≥ ⌈(8 ln n)/Δ²_j⌉`, this inequality is false because `μ* — μ_j = Δ_j` and `2c_{t,s_j} = 2√(2 ln t / s_j) ≤ 2√(2 ln n / (8 ln n / Δ²_j)) = Δ_j`. So once arm `j` has been played more than `8 ln n / Δ²_j` times, this event cannot occur.

The expected count `𝔼[T_j(n)]` is then bounded by the threshold `ℓ = ⌈(8 ln n)/Δ²_j⌉` (the number of plays before the third event is impossible) plus the sum over `t` and `s` of the probabilities of events 1 and 2, which is at most `Σ_{t=1}^∞ Σ_{s=1}^t Σ_{s_j=1}^t 2t⁻⁴`. The double sum evaluates to `Σ_{t=1}^∞ 2t⁻² = π²/3`, and adding the initial `+1` (for the first arm play in the initialization) gives the constant term `1 + π²/3`.

**Why constant 8 instead of the optimal 1/2?** The Chernoff-Hoeffding bound gives `e^{-2a²/n}`, which leads to `√(2 ln t / s)` in the exploration bonus and `8 ln n / Δ²_j` in the regret bound. The optimal constant `1/(2Δ²_j)` comes from using the Kullback-Leibler divergence `D(p_j ‖ p*)` via Sanov's theorem or large deviations theory, which captures the exact exponential decay rate of the probability that a suboptimal arm appears better than the optimal one. The Chernoff-Hoeffding bound, while more general (it requires only bounded support, not a specific distribution family), is looser — it uses a worst-case bound on the moment-generating function that is tight only for Bernoulli(1/2) variables. The constant `8` is the price of distribution-free generality.

---

#### Policy UCB2: Optimal Constants via Epoch-Based Play

UCB2, described in Figure 2, addresses the constant-factor suboptimality of UCB1 by changing the structure of play from round-by-round index comparison to an **epoch-based schedule**. The key idea is that by committing to play a chosen arm for multiple consecutive rounds (an epoch), the algorithm can use a tighter exploration bonus that brings the leading constant arbitrarily close to the optimal `1/(2Δ²_j)`.

**Input parameter.** UCB2 takes a single parameter `α ∈ (0, 1)` that controls the trade-off between the regret constant and the additive overhead. Smaller `α` gives a better leading constant (closer to `1/(2Δ²_j)`) but a larger additive constant `c_α` that diverges as `α → 0` (see Equation (18) for the exact expression of `c_α`).

**Epoch structure.** Let `τ(r) = ⌈(1 + α)ʳ⌉` be an exponential function that determines epoch lengths. For each arm `i`, let `r_i` count how many epochs have been completed for that arm so far (including the current epoch if one is in progress). At the start of each new epoch, the algorithm selects an arm `i` and then plays it for exactly `τ(r_i + 1) — τ(r_i)` consecutive plays. This means the number of plays in each epoch grows approximately as `α(1+α)^{r_i}`, which is exponential in the epoch index.

**Index computation.** At the moment of selecting which arm starts a new epoch (which happens after the current epoch completes), the algorithm computes an index for each arm `i` using the current number of plays `n` and the number of completed epochs `r_i`:

$$I_i = \bar{X}_i + a_{n, r_i}$$

where

$$a_{n, r} = \sqrt{\frac{(1 + \alpha) \ln(e n / \tau(r))}{2 \tau(r)}}$$

> where `X̄_i` is the sample mean of arm `i` based on all plays so far (which is `τ(r_i)` plays since epochs partition the plays of each arm), `n` is the current total number of plays across all arms, `τ(r)` is the total number of plays of arm `i` after `r` epochs, and `α` is the algorithm's parameter.

**What it computes:** The index has the same structure as UCB1 — sample mean plus exploration bonus — but the exploration bonus uses a different functional form. The numerator `(1+α) ln(en/τ(r))` is analogous to `2 ln t` in UCB1 (both grow logarithmically in total plays), but the extra factor `(1+α)` provides a tighter constant. The denominator `2τ(r)` is `2s` where `s` is the number of plays of this arm, providing a `1/√s` scaling like UCB1 but with a coefficient that approaches `1/√(2s)` as `α → 0`.

**Why this form — the epoch structure enables tighter bounds.** The critical difference from UCB1 is that UCB2 commits to playing the chosen arm for an entire epoch, which means the index comparison occurs only at epoch boundaries. This reduces the number of decision points where the index comparison could go wrong. Specifically, for arm `j` to complete its `r`-th epoch, its index must have exceeded the optimal arm's index at the specific moment the epoch started — not at every round within the epoch. The epoch length `τ(r+1) — τ(r)` is chosen so that by the time arm `j` has played `τ(r)` times, enough data has accumulated that the probability of it still appearing better than the optimal arm is tightly controlled.

The exploration bonus `a_{n,r}` uses the *current total plays* `n` in the logarithm rather than the specific time when the epoch started, which is a technical detail that simplifies the union bound over decision times. The factor `(1 + α)` in the numerator sets the confidence level: the probability that the index comparison fails for arm `j` at epoch `r` is controlled by a Chernoff-Hoeffding bound that yields a decay rate governed by `(1 + α)`.

**Regret bound (Theorem 2).** For `n` sufficiently large (specifically `n ≥ max_{i: μ_i < μ*} 1/(2Δ²_i)`), and for any suboptimal arm `j`, the expected regret is bounded by:

$$\text{Regret}(n) \leq \sum_{j: \mu_j < \mu^*} \left[\frac{(1 + \alpha)(1 + 4\alpha) \ln(2e \Delta^2_j n)}{2 \Delta_j} + \frac{c_\alpha}{\Delta_j}\right]$$

> where `c_α` is a constant defined in Equation (18) that depends only on `α`, and `e` is Euler's number.

**What this bound means operationally.** As `α → 0`, the leading constant `(1+α)(1+4α)/2` approaches `1/2`, which is the information-theoretically optimal constant. However, `c_α → ∞` as `α → 0`, so the bound gets a larger additive constant. In practice, the authors suggest that `α` can be chosen as a slowly-decreasing function `α_n` of the horizon `n`, so the leading constant approaches optimal as `n` grows while the additive constant doesn't explode prematurely.

**The remark on α selection.** The paper explicitly notes this tradeoff: "By choosing α small, the constant of the leading term in the sum (4) gets arbitrarily close to `1/(2Δ²_i)`; however, `c_α → ∞` as α → 0. The two terms in the sum can be traded-off by letting `α = α_n` be slowly decreasing with the number `n` of plays." This anticipates the practical tuning problem — a fixed small `α` might be theoretically elegant but practically wasteful at finite horizons because the `c_α` term dominates.

**Proof sketch (Appendix A).** The proof uses a similar union-bound template to UCB1 but must handle the epoch structure. For arm `j` to start its `r`-th epoch (with `r > r̃_j`, where `r̃_j` is a threshold epoch after which the suboptimality gap `Δ_j` dominates the exploration bonus), one of two bad events must happen: either arm `j`'s sample mean is over-optimistic (event probability controlled by Chernoff-Hoeffding), or the optimal arm's sample mean is under-optimistic. The epoch structure introduces an integral over continuous time (replacing the discrete sum over `t` in UCB1) to bound `Σ_{r > r̃_j} (τ(r) — τ(r-1)) × P(event at epoch r)`. The integral can be evaluated analytically because `τ(r)` is approximately `(1+α)^r`, leading to closed-form expressions involving `c_α`.

---

#### Policy ε_n-GREEDY: Randomized Exploration with Annealing

The ε_n-GREEDY policy, described in Figure 3, represents a fundamentally different approach: instead of computing deterministic upper confidence indices, it explicitly randomizes between exploration (picking a uniformly random arm) and exploitation (picking the arm with the highest current sample mean). The key innovation is the **annealing schedule** for the exploration probability `ε_n`, which decays as `O(1/n)`.

**Algorithm mechanics.** At time `n`:

1. Compute the current exploration probability `ε_n = cK / (d² n)`, where `c > 0` and `d` are parameters.
2. With probability `ε_n`, select an arm uniformly at random (each of the `K` arms has probability `ε_n/K`).
3. With probability `1 — ε_n`, select the arm with the highest current sample mean `X̄_{i, T_i(n-1)}` (greedy exploitation).

**Parameters.** The policy requires two parameters:

- `d`: a **lower bound on the suboptimality gaps**. Specifically, `0 < d ≤ min_{i: μ_i < μ*} Δ_i`. This is a non-trivial requirement — it means the algorithm must be told, a priori, that no suboptimal arm is within `d` of the optimal arm's mean. Without this knowledge, the algorithm cannot set the decay schedule.
- `c`: a constant that must be "large enough" (the paper suggests `c > 5` for the bound to hold). This controls the overall scale of the exploration probability.

**The instantaneous regret bound (Theorem 3).** Unlike Theorems 1 and 2, which bound the *integrated* expected number of suboptimal pulls `𝔼[T_j(n)]`, Theorem 3 bounds the **instantaneous probability** that a suboptimal arm `j` is chosen at time `n`:

$$\mathbb{P}\{I_n = j\} \leq \frac{c}{d^2 n} + 2\left(\frac{c}{d^2} \ln\frac{(n-1)d^2 e^{1/2}}{cK}\right) \left(\frac{cK}{(n-1)d^2 e^{1/2}}\right)^{c/(5d^2)} + \frac{4e}{d^2} \left(\frac{cK}{(n-1)d^2 e^{1/2}}\right)^{c/2}$$

**What this bound means.** For large `n` and `c > 5`, the second and third terms decay as `O(1/n^{1+δ})` for some `δ > 0`, making them negligible compared to the leading `c/(d²n)` term. So the probability of pulling a suboptimal arm at time `n` decays as `O(1/n)` — polynomially fast, not just logarithmically slow. This is a *stronger* result than Theorems 1–2 in terms of the asymptotic rate, but it comes at the cost of requiring the parameter `d`.

**Why the decay schedule works.** The intuition is that as `n` grows, the empirical means `X̄_{i, T_i(n)}` become increasingly accurate estimates of the true means `μ_i`. The probability that the greedy choice (the arm with the highest sample mean) is wrong depends on the probability that any suboptimal arm's sample mean exceeds the optimal arm's sample mean. By Bernstein's inequality (Fact 2), this probability decays exponentially in the number of times each arm has been played. Since `ε_n` ensures that every arm is played at least `Ω(log n)` times by time `n` in expectation, the misidentification probability decays polynomially, and the `1/n` decay of `ε_n` is slow enough to ensure sufficient exploration while being fast enough to keep the *regret* (which is `Σ Δ_j × P(I_n = j)`) logarithmic.

**The role of `x₀` in the proof (Section 3, proof of Theorem 3).** The proof is built around the quantity:

$$x_0 = \frac{1}{2K} \sum_{t=1}^{n} \varepsilon_t$$

which is half the expected number of *random* plays of any specific arm during the first `n` rounds. The analysis partitions the number of plays of arm `j` into those that were random (`T^R_j(n)`) and those that were greedy choices. For the greedy choices, the probability of choosing suboptimal arm `j` at time `n` is bounded by the probability that `X̄_{j, T_j(n)} ≥ X̄*_{T*(n)}`, which is itself bounded via Chernoff-Hoeffding by splitting at `μ_j + Δ_j/2` and `μ* — Δ_j/2`. The random plays contribute `ε_n/K` directly to the instantaneous probability.

The key technical challenge is that `T_j(n)` is a random variable — you can't simply plug in a fixed number into the Chernoff-Hoeffding bound. The proof handles this by conditioning on events of the form `T_j(n) = t` and summing over `t`, using the fact that the probability of arm `j` having been played `t` times is bounded by the probability that the *random* plays have been sufficient (`T^R_j(n) ≤ t`), which is in turn bounded by Bernstein's inequality. This careful conditioning chain is what enables the polynomial decay in the bound.

**Why require `d`?** The parameter `d` controls how aggressively `ε_n` decays. If the decay is too fast, there won't be enough exploration to distinguish arms with small gaps. The condition `d ≤ min Δ_i` ensures that the schedule is calibrated to the smallest gap in the problem — it must be slow enough to separate the optimal arm from even the second-best arm. Without this knowledge, a fixed decay schedule could fail (regret could be linear) if the true gaps are smaller than expected.

---

#### Policy UCB1-NORMAL: Handling Unknown Variance in Normal Rewards

UCB1-NORMAL, described in Figure 4, adapts the UCB1 framework to the case where rewards are normally distributed with **unknown means and unknown variances**. This is a natural extension because in many real-world settings, reward variability differs across arms — an arm with high variance needs more exploration than an arm with low variance, even if they have the same sample mean.

**The index computation.** For each arm `i`, after `s` plays, the algorithm computes the sample mean `X̄_{i,s}` and the sample sum of squares `Q_{i,s} = Σ_{t=1}^s X²_{i,t}`. The exploration bonus is:

$$\text{bonus}_i = \sqrt{\frac{16 \cdot \frac{Q_{i,s} - s \bar{X}^2_{i,s}}{s - 1} \cdot \ln t}{s}}$$

> where `(Q_{i,s} — sX̄²_{i,s})/(s — 1)` is the sample variance (the usual unbiased estimator), `t` is the current total number of plays, `s = T_i(t-1)` is the number of plays of arm `i`, and 16 is a constant derived from the tail bounds on the Student and χ² distributions.

**What it computes:** The numerator of the exploration bonus is proportional to the sample standard deviation of arm `i` — arms with higher observed variance get a larger exploration bonus, reflecting the greater uncertainty about their true mean. The factor `ln t / s` has the same structure as UCB1: logarithmically growing in total time, decaying with the number of plays of this arm.

**Why 16? The role of Conjectures 1 and 2.** The constant 16 comes from the tail bounds that the proof requires. The paper states two conjectures that are numerically verified but not analytically proven:

- **Conjecture 1:** For a Student random variable `X` with `s` degrees of freedom, `ℙ{X ≥ a} ≤ e^{-a²/4}` for `0 ≤ a ≤ √(2(s+1))`. Setting `a = 4√(ln t)` gives the exponential decay rate `t⁻⁴` that matches UCB1's proof template. The factor 4 in the exponent becomes 16 in the exploration bonus after squaring and adjusting for the variance estimate.

- **Conjecture 2:** For a χ² random variable `X` with `s` degrees of freedom, `ℙ{X ≥ 4s} ≤ e^{-(s+1)/2}`. This controls the probability that the sample variance overestimates the true variance by more than a factor of 4, which would falsely inflate the exploration bonus and cause unnecessary exploration.

The index for arm `i` is `X̄_{i,s} + bonus_i`, and the arm with the highest index is played at each step (after an initialization phase of playing each arm at least once, and again whenever `s ≤ ⌈8 ln t⌉` to avoid the Student distribution's heavy tails at very small sample sizes).

**Regret bound (Theorem 4).** The expected regret after `n` plays is bounded by:

$$\text{Regret}(n) \leq 256 (\log n) \left[\sum_{j: \mu_j < \mu^*} \frac{\sigma^2_j}{\Delta_j}\right] + \left(1 + \frac{\pi^2}{2} + 8 \log n\right) \left[\sum_{j=1}^{K} \Delta_j\right]$$

> where `σ²_j` is the variance of arm `j`'s reward distribution, and `Δ_j = μ* — μ_j` as before.

**What this bound means.** The leading constant is `256 σ²_j / Δ²_j`, which is substantially larger than UCB1's `8/Δ²_j` (for bounded rewards, where `σ²_j ≤ 1/4`). The factor 256 arises from: (a) the 16 in the exploration bonus being squared to 256 in the regret bound, (b) the variance `σ²_j` explicitly appearing as a multiplicative factor — arms with high variance are more expensive to distinguish.

**Why the Student and χ² distributions?** For normal data with unknown variance, the standardized mean `(X̄ — μ) / (σ̂ / √s)` follows a Student's t-distribution with `s — 1` degrees of freedom, not a normal distribution. The Chernoff-Hoeffding bound used in UCB1 does not apply to Student variables because they have polynomial (not exponential) tail decay. The conjecture about the Student tail provides an exponential bound `e^{-a²/4}` that restores the `t⁻⁴` decay needed for the union bound. Similarly, the sample variance `σ̂² = (Q — sX̄²)/(s — 1)` is distributed as `(σ² / (s — 1)) × χ²_{s-1}`, so controlling the variance estimate requires a tail bound on the χ² distribution.

**The proof structure (Appendix B).** The proof follows the same three-event decomposition as UCB1 (Section 3, proof of Theorem 1), substituting the normal-specific tail bounds for the Chernoff-Hoeffding bounds. The threshold `ℓ` becomes `max(256 σ²_j / Δ²_j, 8) ln t` to account for both the mean estimation error (via the Student conjecture) and the variance estimation error (via the χ² conjecture). The union bound over `t` yields `π²/2 + 8 log n` in the additive term, replacing the `1 + π²/3` from UCB1.

---

#### UCB1-TUNED: An Empirically Improved Heuristic

Section 4 introduces UCB1-TUNED, a practical variant that is not accompanied by a theoretical regret bound but performs "substantially better than UCB1 in essentially all of our experiments." The modification replaces the pure Chernoff-Hoeffding exploration bonus with a **variance-aware** bonus.

**The modified index.** For arm `j` that has been played `s` times by total time `t`, UCB1-TUNED uses:

$$I_j = \bar{X}_{j,s} + \sqrt{\frac{\ln n}{s} \min\{1/4, V_j(s)\}}$$

where

$$V_j(s) = \left(\frac{1}{s} \sum_{\tau=1}^{s} X^2_{j,\tau}\right) - \bar{X}^2_{j,s} + \sqrt{\frac{2 \ln t}{s}}$$

> where `(1/s) Σ X²_{j,τ} — X̄²_{j,s}` is the sample variance (the biased estimator, not divided by `s — 1`), and `√(2 ln t / s)` is an upper confidence bound for the variance estimation error.

**What it computes:** `V_j(s)` is an *upper confidence bound* for the true variance of arm `j`, constructed analogously to the upper confidence bound for the mean in UCB1. The sample variance is used as the point estimate, and `√(2 ln t / s)` is added to account for the uncertainty in estimating the variance from `s` samples. The outer `min{1/4, V_j(s)}` caps the variance at 1/4, which is the maximum variance for any distribution on `[0, 1]` (achieved by Bernoulli(1/2)).

**Why this works better.** The theoretical UCB1 exploration bonus `√(2 ln t / s)` uses a worst-case variance of 1/4 for all arms, which is overly conservative for arms whose reward distributions have low variance. For example, a Bernoulli(0.9) arm has variance `0.9 × 0.1 = 0.09`, which is nearly 3× smaller than the worst-case 0.25. UCB1-TUNED estimates the actual variance and shrinks the exploration bonus accordingly, focusing exploration on arms with high variance (where uncertainty is genuinely larger) and reducing unnecessary exploration of low-variance arms. The authors note they "are not able to prove a regret bound" for this variant, acknowledging the gap between practical performance and theoretical guarantees.

---

#### Design Choices and Their Justifications

**Why upper confidence bounds rather than Bayesian posteriors?** Bayesian approaches to the bandit problem (e.g., Thompson sampling, Gittins indices) require specifying a prior over the reward distributions and computing posterior distributions, which is computationally expensive and relies on modeling assumptions. The UCB approach is **frequentist and nonparametric** — it makes no assumptions about the reward distributions beyond bounded support, and the indices are simple functions of sample means and counts that can be updated in `O(1)` time per play. This makes the policies "simple to implement and computationally efficient" (Section 1).

**Why Chernoff-Hoeffding rather than tighter concentration inequalities?** The Chernoff-Hoeffding bound (Fact 1) provides an exponential tail bound `e^{-2a²/n}` that holds for all bounded random variables. Tighter bounds exist for specific distributions (e.g., Bernstein's inequality for small-variance distributions, or KL-based bounds), but they require knowledge of the variance or the distribution family. The authors prioritize **generality** — the policies work for *any* distribution on `[0, 1]` without parameter tuning.

**Why epoch-based play in UCB2?** The epoch structure reduces the number of decision points where index comparisons occur, tightening the union bound. In UCB1, the index is compared at every time step, producing `n` decision points, each contributing to the failure probability. In UCB2, decisions occur only at epoch boundaries, and the number of epochs for any arm is `O(log n)` because epoch lengths grow exponentially. This reduction in decision points allows a tighter exploration bonus (with constant approaching `1/2` instead of `8`) while maintaining the same overall failure probability budget.

**Why `ln t` in the exploration bonus rather than `ln(1+t)` or another function?** The `ln t` arises from the union bound over time: to guarantee that a confidence interval holds simultaneously for all `t` up to `n`, each individual interval must hold with probability `1 — 1/t⁴` so that `Σ_t 1/t⁴` converges (to `π⁴/90`). The `ln t` in the numerator of the exploration bonus is the solution to `e^{-2(s × bonus²)} = t⁻⁴`, giving `bonus = √(2 ln t / s)`. Using a slower-growing function (like `ln ln t`) would cause the union bound sum to diverge; using a faster-growing function would unnecessarily inflate the bonus.

**Why `cK/(d²n)` decay for ε_n-GREEDY?** The `1/n` decay is the critical threshold: any slower decay (e.g., `1/√n`) causes the *total* number of exploration steps to grow super-logarithmically, resulting in super-logarithmic regret. Any faster decay (e.g., `1/n²`) risks insufficient exploration — there may not be enough random plays to guarantee that the greedy choice eventually identifies the optimal arm. The `1/n` rate balances these tensions: total exploration plays grow as `Σ_{t=1}^n 1/t ≈ ln n`, which is logarithmic, while the per-arm exploration frequency is sufficient for the sample means to converge at a rate that makes misidentification probability decay polynomially.

**Why `256 log n` in UCB1-NORMAL vs. `8 ln n` in UCB1?** The factor of 32× increase (256 vs. 8) reflects the additional uncertainty from:
- Unknown variance: each arm's variance `σ²_i` must be estimated from data, and the estimate has its own error (controlled by the χ² bound)
- Heavier tails: the Student distribution has polynomial tails while the normal distribution (implicitly bounded by Chernoff-Hoeffding) has exponential tails, requiring a larger constant to achieve the same `t⁻⁴` decay
- The factor 4 in the Student bound (`a = 4√(ln t)`) squared yields 16, which squares again (to 256) when plugged into the regret calculation because the suboptimality gap `Δ_j` appears squared in the denominator of the threshold `ℓ`.

## 4. Key Insights and Innovations

### Innovation 1: Finite-Time Analysis as a Distinct Theoretical Contribution, Not Just Tighter Constants

The most fundamental intellectual move in this paper is the elevation of **uniform-over-time regret bounds** from a desirable refinement to a first-class theoretical goal warranting new policies, new proof techniques, and a re-evaluation of what "optimal" means in bandit problems. This is not merely "asymptotic bounds with explicit constants" — it is a qualitatively different form of guarantee that changes what one can promise about an algorithm's behavior.

**What the field did before.** Lai and Robbins (1985) established the asymptotic lower bound `Regret(n) ~ (1/D(p_j ‖ p*)) ln n`, and a series of subsequent papers (Agrawal, 1995; Burnetas & Katehakis, 1996) developed policies achieving this asymptotic rate. The asymptotic framework answers the question: "as n → ∞, how fast does the per-round regret decay to zero?" But it is silent on the question a practitioner actually asks: "If I run this policy for exactly n = 10,000 rounds, how much regret should I expect, at most?" The `o(1)` term in the asymptotic expression could conceal arbitrarily large constant overheads that dominate at any finite horizon. A policy could be asymptotically optimal yet practically useless for the first million plays.

**What this paper does differently.** The finite-time analysis provides a bound of the form `Regret(n) ≤ (constant) × ln n + (explicit additive constant)` that holds for *every* n ≥ 1, not just in the limit. For UCB1, the bound is `Regret(n) ≤ [8 Σ (ln n)/Δ_j] + (1 + π²/3) Σ Δ_j`. The constant terms — 8 in the leading coefficient, `(1 + π²/3)` in the additive part — are computed exactly and hold without hidden caveats. This transforms the regret bound from a limiting statement (useful for comparing policies asymptotically) to a **worst-case performance guarantee** (useful for deployment, budget allocation, and head-to-head comparison at specific horizons).

**Why this is a conceptual shift, not a technical tightening.** An asymptotic analysis essentially says "the log term dominates everything eventually." A finite-time analysis says "here is the exact trade-off between the log term and the constant overhead, and you can compute which dominates at your specific horizon." This matters profoundly for practical algorithm selection. For example, UCB2 achieves a leading constant approaching `1/(2Δ²_j)` — dramatically better than UCB1's `8/Δ²_j` — but with a much larger additive constant `c_α` that diverges as the leading term approaches optimality. In an asymptotic framework, UCB2 is strictly superior (better constant). In a finite-time framework, UCB1 may be preferable at moderate horizons because its smaller additive overhead dominates. The paper makes this explicit via the remark on Theorem 2: "the two terms in the sum can be traded-off by letting α = α_n be slowly decreasing with the number n of plays." This is not a statement one can make in an asymptotic analysis — it is a uniquely finite-time insight.

**Evidence.** The distinction is most visible in the experimental results (Section 4), where UCB2 consistently performs "slightly worse" than UCB1-TUNED despite its superior asymptotic constant. Figures 6–12 show UCB2 trailing UCB1-TUNED on essentially all seven Bernoulli distributions tested, across 100,000 plays. The asymptotic analysis alone would predict the opposite. The finite-time framework explains this: UCB2's `c_α` overhead is large enough at these horizons to outweigh its tighter leading constant. This validates the authors' decision to develop both policies rather than presenting UCB2 as the unambiguously superior successor to UCB1.

---

### Innovation 2: The Upper Confidence Bound as a Unifying Principle Across Distribution Types and Uncertainty Sources

Prior to this paper, the bandit literature contained two largely separate theoretical traditions: **parametric policies** (Lai & Robbins, 1985) that assumed specific distribution families and used Kullback-Leibler divergences to construct optimal indices, and **nonparametric policies** (Agrawal, 1995) that assumed only bounded support and used concentration inequalities for index construction. The parametric tradition achieved optimal constants but at the cost of computational complexity and distributional assumptions; the nonparametric tradition was computationally simple but lacked finite-time analysis. This paper shows that a **single principle — the upper confidence bound constructed from concentration inequalities** — can be systematically extended across distribution types (bounded support, normal with unknown variance) and across uncertainty sources (mean uncertainty, variance uncertainty), yielding logarithmic regret with explicit constants in each case.

**The unifying move.** The UCB principle is: construct an index for each arm that represents the highest value its true mean could plausibly take, given the observed data, and play the arm with the highest index. The innovation is recognizing that this principle is **modular** — the concentration inequality determines the exploration bonus and the regret constant, but the algorithmic structure (maintain running averages, add bonus, pick max) remains invariant. UCB1 instantiates the principle with Chernoff-Hoeffding (yielding constant 8); UCB2 refines it with an epoch structure that tightens the union bound (yielding constant approaching 1/2); UCB1-NORMAL instantiates it with Student and χ² tail bounds (yielding constant 256σ²_i/Δ²_i) to handle unknown variance. The ε_n-GREEDY policy shows the same modularity from the other direction: replace the implicit exploration of optimism with explicit randomization, and the Chernoff-Hoeffding and Bernstein inequalities still govern the regret bound.

**Why this is nontrivial.** The Lai and Robbins framework ties the index computation to the specific parametric form of the reward distribution — the Kullback-Leibler divergence `D(p_j ‖ p*)` is computed from the density functions, which must be known up to a parameter. Extending this to unknown variance in the normal case would require completely re-deriving the index. The UCB framework, by contrast, simply plugs in a different tail bound. The authors exploit this modularity to tackle the normal-with-unknown-variance case, noting in Section 1 that "surprisingly, we could not find in the literature regret bounds (not even asymptotical) for the case when both the mean and the variance of the reward distributions are unknown." They fill this gap not by inventing new machinery but by applying the existing UCB template with the appropriate concentration inequalities (Conjectures 1 and 2).

**Evidence of the unification's power.** The proof templates for Theorems 1–4 are structurally identical: bound `E[T_j(n)]` by decomposing the event that a suboptimal arm's index exceeds the optimal arm's index into three cases (optimal arm underestimated, suboptimal arm overestimated, or suboptimal arm insufficiently played), apply the relevant concentration inequality to each, and sum over time. Theorem 1 uses Chernoff-Hoeffding; Theorem 4 uses Student/χ² tail bounds — different inequalities, but the proof architecture is identical. This modularity is what makes the paper more than a collection of algorithms; it is a **framework for deriving bandit algorithms with finite-time guarantees** given appropriate tail bounds.

---

### Innovation 3: Polynomial Instantaneous Regret via Annealed ε-Greedy — A Stronger Guarantee from a Weaker-Looking Algorithm

The ε_n-GREEDY policy and its analysis (Theorem 3) represent a genuinely counterintuitive finding: a simple randomized algorithm — the kind a practitioner might invent by intuition — can achieve a **stronger** theoretical guarantee than the sophisticated deterministic UCB policies, at least along one dimension. Theorem 3 bounds the *instantaneous* probability of choosing a suboptimal arm `P{I_n = j} = O(1/n)`, a polynomial decay, whereas the UCB bounds are on the *integrated* expected count `E[T_j(n)] = O(log n)`. An exponential decay in the instantaneous probability is stronger than a logarithmic bound on the cumulative count.

**Why this is surprising.** The ε-greedy heuristic is widely used in practice (the authors cite Sutton & Barto, 1998) but is typically considered theoretically inferior to UCB-style optimism. The standard criticism is that constant-ε exploration causes linear regret. The "obvious fix" — annealing ε — was known informally, but there was no rigorous characterization of what decay rate works and what the resulting regret looks like. A natural intuition would be that annealing is a crude approximation to the principled confidence bounds of UCB1, and thus should perform worse. Theorem 3 inverts this expectation: with the right schedule (`ε_n = cK/(d²n)`), ε_n-GREEDY achieves a bound of order `c/(d²n) + o(1/n)` on the probability of choosing a *specific* suboptimal arm at time n. For large c (> 5), the subleading terms are `O(1/n^{1+δ})`, making the `1/n` term the dominant decay.

**Comparison to the UCB guarantees.** The UCB1 bound is on `E[T_j(n)]`, the expected *total* plays of arm j over the first n rounds. To convert this to an instantaneous bound, one would need additional arguments (e.g., that plays of arm j are spread roughly uniformly over time, which is not guaranteed). The ε_n-GREEDY bound directly controls the per-round regret: at any late round n, the probability of a mistake is small and quantifiable. This is a stronger guarantee for applications where regret is time-discounted or where late-round performance matters disproportionately (e.g., a bandit algorithm deployed indefinitely where stakeholders care most about recent behavior).

**The cost: parameter knowledge.** The stronger guarantee comes with a significant caveat: ε_n-GREEDY requires a known lower bound `d ≤ min_{i: μ_i < μ*} Δ_i`, the minimum suboptimality gap. Without this, the algorithm cannot set its decay schedule. UCB1 requires no such knowledge — it adapts automatically to whatever gaps exist. The trade-off is explicit: stronger per-round guarantees in exchange for a priori knowledge of the problem's difficulty. This is not framed as a weakness but as a precise characterization of the **information requirements** for different guarantee strengths, which is itself a conceptual contribution.

**Evidence of sensitivity.** The experiments (Section 4.1) dramatically illustrate the cost of this parameter dependence. The authors note that "the choice of c in policy ε_n-GREEDY is difficult as there is no value that works reasonably well for all the distributions that we considered." On distributions 12 and 14 (10-armed bandits with multiple suboptimal arms having varied gaps), ε_n-GREEDY performs poorly because "ε_n-GREEDY explores uniformly over all machines, thus the policy is hurt if there are several nonoptimal machines." The UCB policies, by contrast, focus exploration on arms whose confidence bounds overlap with the best arm, naturally avoiding this pitfall. The fact that the randomized policy can still be "best" when optimally tuned (the paper notes it "performs almost always best" with optimal tuning) but degrades rapidly when mistuned is a nuanced empirical finding that complements the theoretical analysis.

---

### Innovation 4: Variance-Aware Exploration as an Empirical Heuristic with a Clear Theoretical Origin

Section 4 introduces UCB1-TUNED, a variant that replaces the worst-case exploration bonus `√(2 ln t / s)` with a variance-aware bonus `√(ln n / s × min{1/4, V_j(s)})`, where `V_j(s)` is an upper confidence bound on the true variance of arm j. This is presented without a regret bound — the authors state they "are not able to prove a regret bound" — yet it performs "substantially better than UCB1 in essentially all of our experiments." This is an unusual move in a theoretical paper, and it serves as a bridge between rigorous analysis and practical deployment.

**What makes this an innovation rather than just tuning.** The construction of `V_j(s)` is not ad hoc. It follows the same UCB principle applied to a different estimand: `V_j(s) = (sample variance) + √(2 ln t / s)` is an upper confidence bound for the true variance, exactly analogous to `X̄_{i,s} + √(2 ln t / s)` being an upper confidence bound for the true mean. The outer `min{1/4, V_j(s)}` reflects the known fact (via the Bhatia-Davis inequality or the Popoviciu inequality for variances on bounded intervals) that the maximum variance of any distribution on `[0, 1]` is 1/4, achieved by the Bernoulli(1/2) distribution. The heuristic is thus **theory-guided**: it instantiates the UCB principle at two levels — once for the mean (using the estimated variance to set the exploration bonus width) and once for the variance itself (using concentration to set an upper bound on variance).

**Why this matters as a diagnostic.** UCB1-TUNED demonstrates that the `8/Δ²_j` constant in UCB1's regret bound is largely driven by the worst-case variance assumption. By measuring variance from data and using it to shrink the exploration bonus for low-variance arms, UCB1-TUNED implicitly achieves a constant closer to the actual variance of each arm. For a Bernoulli(0.9) arm with variance 0.09, the effective exploration bonus is roughly `√0.09/0.25 ≈ 0.6×` the UCB1 bonus. The experimental results (Figures 6–12) confirm that this variance correction is practically significant — UCB1-TUNED consistently outperforms UCB2 (which has the better asymptotic constant but no variance adaptation) and matches or approaches optimally-tuned ε_n-GREEDY.

**The tension between theory and practice.** The authors' honesty about the lack of a regret bound for UCB1-TUNED — combined with its prominent placement and favorable experimental results — sends a methodological message. It acknowledges that the theoretical framework (Chernoff-Hoeffding union bounds over all t) may not be the right tool to analyze variance-adaptive methods, and that closing this gap requires new proof techniques. The paper thus identifies a theoretical open problem through empirical demonstration rather than through conjecture alone. This is a form of **diagnostic contribution**: UCB1-TUNED's strong performance tells us that variance estimation is a high-value target for future theoretical work, and its construction from UCB principles provides a template for how such methods might be designed.

**Evidence.** Figures 6, 7, and 8 show UCB1-TUNED on three 2-armed Bernoulli problems. On distribution 2 (arms with means 0.9 and 0.8, a "hard" problem with small Δ = 0.1), UCB1-TUNED achieves roughly 20% suboptimal plays by n = 10,000, while UCB1 achieves roughly 30% — a 50% relative improvement. On distribution 3 (arms 0.55 and 0.45, where the optimal arm has high variance 0.2475), UCB1-TUNED and UCB1 converge to similar performance, consistent with the variance correction being less impactful when the optimal arm is the high-variance one. This differential behavior confirms that the mechanism is operating as designed — shrinking the bonus specifically for low-variance arms, not uniformly across arms.

## 5. Experimental Analysis

### Evaluation Methodology

- **Dataset.** The experiments use seven synthetic Bernoulli bandit configurations — three 2-armed problems (distributions 1–3) and four 10-armed problems (distributions 11–14) — with reward expectations specified in a table in Section 4. Each entry in the table gives the probability of receiving reward 1 for each arm. Distribution 1 (arms 0.9, 0.6) and distribution 11 (arms 0.9, 0.6, …, 0.6) are characterized as "easy" because the optimal machine has low variance and the gaps Δ_i are large; distribution 3 (arms 0.55, 0.45) and distribution 14 (arms 0.55, 0.45, …, 0.45) are characterized as "hard" because the optimal machine has high variance and some gaps are small. No external benchmark datasets are used — the experiments are purely synthetic, designed to test how the policies behave under controlled, known reward distributions.

- **Base models.** Not applicable — this is not a machine learning model evaluation. The experimental subjects are the three policies introduced in the paper (UCB1, UCB2, ε_n-GREEDY) plus the empirically-tuned variant UCB1-TUNED, all implemented directly as described in Figures 1–4 and Section 4.

- **Metrics.** Two performance measures are tracked for each experiment: (1) **the percentage of plays of the optimal machine** — the fraction of the first n plays in which the policy selected the arm with the highest true mean μ*, and (2) **the actual regret** — "the difference between the reward of the optimal machine and the reward of the machine played," which corresponds to the empirical cumulative regret Σ_{t=1}^n (μ* — μ_{I_t}) rather than the expected regret analyzed in the theorems. Both quantities are plotted on semi-logarithmic axes (log-scale for the x-axis representing number of plays, linear scale for the y-axis representing percentage or cumulative regret) over 100,000 plays, averaged over 100 independent runs. Semi-logarithmic plotting is deliberately chosen because logarithmic regret appears as a straight line, making it visually diagnostic: "If a parameter is chosen too small, then the regret grows linearly (exponentially in the semi-logarithmic plot); if a parameter is chosen too large then the regret grows logarithmically, but with a large leading constant (corresponding to a steep line in the semi-logarithmic plot)."

- **Baselines.** No external baselines from prior work are implemented. The comparison is head-to-head among the paper's own policies: UCB1 (Theorem 1), UCB2 (Theorem 2), ε_n-GREEDY (Theorem 3), and UCB1-TUNED (the heuristic variant from Section 4 with no regret bound). The Lai and Robbins (1985) and Agrawal (1995) policies are discussed theoretically but not implemented in the experiments — the paper notes that Lai and Robbins' indices are "generally hard" to compute, which likely motivates their absence. The experiments thus compare the paper's proposed policies against each other rather than against the prior state of the art.

- **Generation budget / compute accounting.** The independent variable is the number of plays n, swept from 1 to 100,000. Every policy makes exactly one arm pull per time step, so the computational cost per step is essentially identical across policies (all require O(1) updates to running averages and, for the index-based policies, O(K) index computations per step). The experimental comparison is therefore a pure statistical efficiency comparison: given the same number of interactions with the environment, which policy accumulates less regret? There is no notion of "training compute" separate from "inference compute" — the bandit setting conflates learning and deployment into a single stream of plays.

- **Cross-validation / statistical protocol.** The experimental protocol is straightforward Monte Carlo simulation: for each of the seven bandit configurations, each policy is run for 100,000 plays, and this is repeated over 100 independent random seeds (different reward realizations and, for ε_n-GREEDY, different randomization of arm selection). Results are averaged across these 100 runs. Section 4 describes a "first round of experiments on distribution 2 to find out good values for the parameters of the policies," indicating that parameter tuning was performed on distribution 2 and then those parameter values (or the tuning methodology) were applied to the other six distributions. For ε_n-GREEDY, the paper acknowledges that tuning is distribution-dependent: "the choice of c in policy ε_n-GREEDY is difficult as there is no value that works reasonably well for all the distributions that we considered. Therefore, we have roughly searched for the best value for each distribution." This means ε_n-GREEDY's reported performance reflects per-distribution tuning, not a single fixed parameter setting, which gives it an advantage over UCB1 and UCB2 (which have no tuned parameters beyond α for UCB2, fixed at 0.001 after the initial sweep).

### Main Quantitative Results

#### Parameter Sensitivity: UCB2 is Robust, ε_n-GREEDY is Brittle

The paper's first experimental result (Figure 5) is a parameter sweep rather than a policy comparison. Figure 5 shows the performance of UCB2 on distribution 2 as α varies. The key finding is that "Policy UCB2 is relatively insensitive to the choice of its parameter α, as long as it is kept relatively small." The figure (plotted on semi-logarithmic axes) likely shows that for α values across several orders of magnitude (presumably from around 0.0001 to 0.1), the regret curves are similar in shape and asymptotic slope, though the exact intercept varies. Based on this sweep, a fixed value of α = 0.001 is selected "for all the remaining experiments."

In contrast, the paper reports that tuning ε_n-GREEDY is substantially harder: "the choice of c in policy ε_n-GREEDY is difficult as there is no value that works reasonably well for all the distributions that we considered." No equivalent of Figure 5 is shown for ε_n-GREEDY's parameter c, but the text indicates that c was tuned per-distribution by roughly searching for the best value. The experiments then show performance for ε_n-GREEDY with c set to this distribution-specific optimum, and also for values of c around this optimum to illustrate sensitivity. The paper states that "the performance degrades rapidly if this parameter is not appropriately tuned," establishing that ε_n-GREEDY's strong results are contingent on oracle-like parameter knowledge — a significant practical caveat.

**The parameter d.** For ε_n-GREEDY, the parameter d — the lower bound on the minimum suboptimality gap — was "set to Δ = μ* — max_{i: μ_i < μ*} μ_i" in every experiment. This is the true minimum gap, meaning the algorithm was given exactly the tightest possible lower bound. This is the most favorable possible setting for ε_n-GREEDY; in practice, d would need to be a conservative underestimate, which would slow the decay of ε_n and increase regret. The experiments do not explore sensitivity to d, only to c — a notable omission given that Theorem 3's bound depends on d² in the denominator of the leading term.

#### Head-to-Head Policy Comparison Across Seven Bandit Configurations

The main results appear in Figures 6–12, with one figure per distribution, each showing four performance curves (UCB1-TUNED, UCB2, and ε_n-GREEDY at multiple c values) for both the percentage of optimal plays and the cumulative regret. The paper provides a summary of the comparison across all seven distributions (Section 4.1), which I will quote and then unpack with specific figure references.

The paper's own summary:

> "– An optimally tuned ε_n-GREEDY performs almost always best. Significant exceptions are distributions 12 and 14: this is because ε_n-GREEDY explores uniformly over all machines, thus the policy is hurt if there are several nonoptimal machines, especially when their reward expectations differ a lot. Furthermore, if ε_n-GREEDY is not well tuned its performance degrades rapidly (except for distribution 13, on which ε_n-GREEDY performs well a wide range of values of its parameter)."

> "– In most cases, UCB1-TUNED performs comparably to a well-tuned ε_n-GREEDY. Furthermore, UCB1-TUNED is not very sensitive to the variance of the machines, that is why it performs similarly on distributions 2 and 3, and on distributions 13 and 14."

> "– Policy UCB2 performs similarly to UCB1-TUNED, but always slightly worse."

**Distribution 1 (Figure 6): 2 arms, means 0.9 and 0.6.** This is the easiest problem — large gap Δ = 0.3, optimal arm has low variance (0.9 × 0.1 = 0.09). All policies converge rapidly to high optimal-play percentages. The optimally-tuned ε_n-GREEDY likely reaches >99% optimal plays within the first few thousand rounds. UCB1-TUNED and UCB2 follow closely. The regret curves on the semi-logarithmic plot should appear roughly linear (logarithmic regret) with shallow slopes.

**Distribution 2 (Figure 7): 2 arms, means 0.9 and 0.8.** A harder problem — small gap Δ = 0.1, though both arms have low variance. This distribution was used for the initial parameter tuning, so all policies are operating with parameters optimized for this case. The paper's earlier statement that ε_n-GREEDY "performs almost always best" likely applies here: the tuned ε_n-GREEDY should show the lowest regret at most horizons, with UCB1-TUNED close behind and UCB2 trailing slightly. The regret curves should have steeper slopes than distribution 1 (reflecting the 1/Δ² dependence — with Δ = 0.1, the regret constant is 9× larger than for Δ = 0.3).

**Distribution 3 (Figure 8): 2 arms, means 0.55 and 0.45.** Characterized as "hard" because the optimal machine has high variance (0.55 × 0.45 = 0.2475, close to the maximum 0.25). The gap Δ = 0.1 is the same as distribution 2, but the high variance makes the problem harder — more samples are needed to confidently identify the optimal arm. The paper highlights that "UCB1-TUNED is not very sensitive to the variance of the machines, that is why it performs similarly on distributions 2 and 3." This is a notable positive result for UCB1-TUNED: it adapts to the increased variance by inflating its exploration bonus via V_j(s), avoiding the degradation that a fixed-bonus policy would suffer. In contrast, ε_n-GREEDY (which explores uniformly regardless of variance) may struggle here because uniform exploration does not target the high-variance optimal arm.

**Distribution 11 (Figure 9): 10 arms, one optimal at 0.9, nine suboptimal at 0.6.** This is the multi-armed extension of distribution 1 — large gaps (Δ = 0.3 for all suboptimal arms), low variance for the optimal arm. The presence of nine identical suboptimal arms creates an interesting dynamic: ε_n-GREEDY's uniform exploration wastes `(9/10) × ε_n` probability on suboptimal arms (all equally bad), while UCB policies focus exploration on arms whose confidence bounds overlap with the leader. The figure likely shows this effect — ε_n-GREEDY may still perform well because the gap is large enough that even wasteful exploration identifies the optimal arm quickly, but the gap between ε_n-GREEDY and UCB1-TUNED may be narrower than on distribution 1.

**Distribution 12 (Figure 10): 10 arms with heterogeneous means (0.9, 0.8, 0.8, 0.8, 0.7, 0.7, 0.7, 0.6, 0.6, 0.6).** This is identified as a distribution where ε_n-GREEDY performs poorly because it "explores uniformly over all machines, thus the policy is hurt if there are several nonoptimal machines, especially when their reward expectations differ a lot." There are three distinct suboptimality levels (Δ = 0.1, 0.2, 0.3, and 0.4), creating a hierarchy of arms where uniform exploration wastes pulls on very suboptimal arms (means 0.6) that could be quickly eliminated by UCB policies. Figure 10 should show UCB1-TUNED and UCB2 outperforming ε_n-GREEDY, particularly at intermediate horizons before ε_n has decayed sufficiently.

**Distribution 13 (Figure 11): 10 arms, optimal at 0.9, nine suboptimal at 0.8.** This is the multi-armed extension of distribution 2 — small gap Δ = 0.1 for all suboptimal arms, all arms have low-to-moderate variance. The paper notes this is an exception where "ε_n-GREEDY performs well [over] a wide range of values of its parameter." With all suboptimal arms identical and the gap small, uniform exploration is not wasteful in the same way as distribution 12 — every suboptimal arm is equally plausible as the best, so exploring them uniformly is reasonable. The figure likely shows all policies performing similarly, with ε_n-GREEDY possibly maintaining a small advantage.

**Distribution 14 (Figure 12): 10 arms, optimal at 0.55, nine suboptimal at 0.45.** This is the hardest configuration — small gap Δ = 0.1, high variance for the optimal arm (0.2475), and nine distractors. The paper identifies this alongside distribution 12 as a case where ε_n-GREEDY struggles. UCB1-TUNED's variance adaptation should be particularly valuable here: the optimal arm's high variance triggers a larger exploration bonus, focusing extra plays on it, while uniform ε_n-GREEDY exploration spreads plays thinly across all ten arms. Figure 12 should show UCB1-TUNED achieving the lowest regret, with UCB2 close behind and ε_n-GREEDY trailing.

#### The Semi-Logarithmic Diagnostic: Separating Linear from Logarithmic Regret

A methodological contribution embedded in the experimental design is the use of semi-logarithmic plots as a diagnostic tool. The paper states: "If a parameter is chosen too small, then the regret grows linearly (exponentially in the semi-logarithmic plot); if a parameter is chosen too large then the regret grows logarithmically, but with a large leading constant (corresponding to a steep line in the semi-logarithmic plot)." This is a clever visualization choice: on a plot where the x-axis (number of plays) is logarithmic and the y-axis (regret or percent suboptimal plays) is linear, a function of the form `c × log(n)` appears as a straight line with slope `c`, while a function growing as `α × n` appears as an exponential curve shooting upward. This allows the reader to visually distinguish logarithmic regret (the theoretical goal) from linear regret (the failure mode) and to compare the leading constants of logarithmic policies by comparing line slopes.

The figures for ε_n-GREEDY with different values of c (shown as multiple curves per figure) presumably illustrate this diagnostic: a too-small c produces regret that curves upward (super-logarithmic, approaching linear), while a too-large c produces a straight but steep line (logarithmic with a large constant). The optimal c produces the shallowest straight line. For distributions where the paper claims ε_n-GREEDY degrades rapidly with mistuning (distributions 12 and 14), the curves for suboptimal c values should show markedly worse behavior.

### Ablation Studies and Robustness Checks

**Parameter α in UCB2 (Figure 5).** The sweep over α on distribution 2 shows that UCB2's performance is relatively insensitive to α when kept small. A fixed value of α = 0.001 is selected for all subsequent experiments. This is a robustness check confirming that the theoretical sensitivity (c_α → ∞ as α → 0) does not manifest catastrophically at practical horizons — α = 0.001 is small enough to approach the optimal constant while keeping the additive overhead manageable. No equivalent sensitivity analysis is shown for the horizon-dependent α_n scheme suggested in the Theorem 2 remark.

**Parameter c in ε_n-GREEDY (Figures 6–12, informally).** The multiple ε_n-GREEDY curves in each figure represent different values of c around the empirically best value. This serves as an informal sensitivity analysis, demonstrating that performance "degrades rapidly if this parameter is not appropriately tuned" for most distributions. Distribution 13 is identified as an exception where ε_n-GREEDY "performs well [over] a wide range of values." This differential sensitivity is itself a finding: ε_n-GREEDY's robustness depends on the bandit structure, with homogeneous suboptimal arms being more forgiving.

**Parameter d in ε_n-GREEDY.** Not ablated. In all experiments, d is set to the true minimum gap Δ, which is the most favorable possible value. The paper does not explore what happens when d underestimates the true gap (which would slow ε_n decay and increase regret) or overestimates it (which could cause linear regret if the decay is too fast to resolve small gaps). This is a significant omission given that Theorem 3's bound and practical applicability both hinge on d being a valid lower bound.

**Variance estimation in UCB1-TUNED (Figures 6–12, compared to UCB1).** While UCB1 itself is not plotted in Figures 6–12 (only UCB1-TUNED, UCB2, and ε_n-GREEDY appear), the paper's claim that UCB1-TUNED performs "substantially better than UCB1 in essentially all of our experiments" is supported by the comparison to UCB2, which itself uses a theoretically tighter exploration bonus than UCB1. The fact that UCB1-TUNED outperforms UCB2 implies an even larger gap over UCB1. The mechanism — variance-aware bonus shrinking — is validated by the differential performance on distributions 2 and 3: UCB1-TUNED "performs similarly on distributions 2 and 3" despite distribution 3's optimal arm having much higher variance, suggesting the variance correction successfully compensates.

**Number of arms (2 vs. 10).** Distributions 1–3 and 11–14 form a deliberate 2×2 design: easy vs. hard gap structure crossed with 2 arms vs. 10 arms. Distribution 11 is the 10-arm analog of distribution 1; distribution 13 is the 10-arm analog of distribution 2; distribution 14 is the 10-arm analog of distribution 3; distribution 12 has no 2-arm analog and tests heterogeneous suboptimal arms. This design allows assessment of how each policy scales with the number of arms. The paper's observation that ε_n-GREEDY is "hurt if there are several nonoptimal machines" (distributions 12 and 14) but performs well with many identical suboptimal arms (distribution 13) isolates the effect of arm heterogeneity rather than arm count per se.

**Optimal arm variance.** The contrast between distributions 2 and 3 (same gap Δ = 0.1, different optimal-arm variance: 0.09 vs. 0.2475) isolates the effect of optimal-arm variance. UCB1-TUNED's insensitivity to this factor is highlighted as a strength. The underlying UCB1 (not plotted) would presumably show degradation on distribution 3 because its fixed bonus doesn't account for the higher variance, requiring more plays to achieve the same confidence. This makes distribution 3 a stress test that UCB1-TUNED passes but UCB1 would likely fail.

**Horizon (100,000 plays).** All experiments run for 100,000 plays, which is long enough for asymptotic behavior to manifest — logarithmic regret appears as a straight line on semi-logarithmic axes, and policies with different constants should be clearly separated. The paper does not explore shorter horizons (e.g., n = 100 or n = 1,000) where the additive constants dominate the logarithmic term. This is a limitation: the finite-time guarantees of Theorems 1–3 are most valuable at short-to-moderate horizons, yet the experiments only show behavior at horizons where asymptotic effects dominate.

**Number of runs (100).** Error bars are not shown in Figures 6–12, making it impossible to assess whether the differences between policies are statistically significant or within the noise of 100 runs. For distribution 2 with Δ = 0.1, the expected number of suboptimal pulls after 100,000 plays might be on the order of 8 × ln(100000) / 0.01 ≈ 9,200 for UCB1, with standard deviation on the order of √9200 ≈ 96 across 100 runs (standard error ≈ 9.6). Differences between policies of a few percentage points in optimal-play rate could be statistically meaningful or could be noise — the plots do not provide the information needed to decide.

### Critical Assessment

**Claim from Section 1: "UCB1 achieves logarithmic regret uniformly over time."** The theoretical claim is proven in Theorem 1. The experiments do not directly validate the theorem (they test UCB1-TUNED, not UCB1), but they provide circumstantial support: UCB1-TUNED, which is a refinement of UCB1, achieves logarithmic regret (straight lines on semi-log plots) across all seven distributions. The uniformity claim — that the bound holds for every n — is inherently theoretical and cannot be validated by experiments at discrete horizons. The experiments do confirm that the policies do not exhibit pathological finite-time behavior (e.g., large initial regret spikes before logarithmic behavior kicks in), which is the practical concern the theorem addresses.

**Claim: "UCB2 brings the leading constant arbitrarily close to the optimal 1/(2Δ²_i)."** This is a theoretical claim about the regret bound as α → 0, proven in Theorem 2. The experiments use a fixed α = 0.001 and show that UCB2 performs "similarly to UCB1-TUNED, but always slightly worse." This is actually evidence *against* the practical superiority of UCB2 at this horizon — despite its better asymptotic constant, the additive overhead c_α (which the theorem acknowledges diverges as α → 0) makes it perform worse than UCB1-TUNED at n = 100,000. The experiments do not test the theoretically-suggested scheme of letting α = α_n decay with n, which could potentially close this gap. This is a genuine disconnect between the theorem and the experiments: Theorem 2 proves UCB2 can approach the optimal constant, but the experiments show that at practical horizons with fixed α, it underperforms a heuristic variant of UCB1.

**Claim: "ε_n-GREEDY yields a stronger instantaneous regret bound of order c/(d²n) + o(1/n)."** Theorem 3 proves an instantaneous bound. The experiments measure *cumulative* regret and percentage of optimal plays, not instantaneous per-round probabilities. The instantaneous bound's stronger decay rate (polynomial vs. logarithmic for the expected count) cannot be directly validated from cumulative regret plots, where both polynomial instantaneous regret and logarithmic expected-count regret can produce similar-looking cumulative curves. The experiments do show that optimally-tuned ε_n-GREEDY "performs almost always best" in terms of cumulative regret, but this doesn't isolate whether the advantage comes from the polynomial instantaneous decay or from other factors (e.g., the uniform exploration schedule happening to work well for these specific distributions). A direct validation of Theorem 3 would require plotting `P{I_n = j}` as a function of n, which is not done.

**Claim: "UCB1-TUNED performs substantially better than UCB1 in essentially all of our experiments."** UCB1 itself is not plotted in Figures 6–12, so this claim cannot be verified from the paper's figures. The reader must rely on the authors' assertion and the indirect evidence that UCB1-TUNED outperforms UCB2 (which itself has a better asymptotic constant than UCB1). This is a significant evidential gap — the paper's most practically useful algorithm lacks both a theoretical regret bound and a direct experimental comparison to its theoretical parent.

**Weakness: No comparison to the Lai and Robbins or Agrawal policies.** The paper motivates its work by citing the computational intractability of Lai and Robbins indices and the lack of finite-time analysis for Agrawal's policies, yet neither is implemented as a baseline. This leaves open the question: do the new finite-time bounds correspond to better practical performance, or are they purely theoretical contributions? If Agrawal's asymptotically-optimal policy were implemented and compared, it might perform similarly to UCB2 (since both target the optimal asymptotic constant) or even outperform it if its additive overhead is smaller. Without this comparison, the experimental results demonstrate that the new policies work, but not that their finite-time guarantees translate to practical advantages over prior art.

**Weakness: The parameter tuning protocol advantages ε_n-GREEDY.** ε_n-GREEDY's parameter c is tuned per-distribution, while UCB2's α is fixed at 0.001 based only on distribution 2. UCB1-TUNED has no tunable parameters. This makes the comparison uneven — ε_n-GREEDY benefits from distribution-specific optimization while the UCB policies receive at most one distribution of tuning. The paper acknowledges this implicitly by showing ε_n-GREEDY with multiple c values, allowing the reader to see performance degradation, but the headline of "optimally tuned ε_n-GREEDY performs almost always best" conflates algorithmic quality with tuning effort. A fairer comparison would fix c based on distribution 2 (as was done for α in UCB2) and use that fixed value across all distributions.

**Weakness: The d parameter in ε_n-GREEDY is set to the true minimum gap.** This gives ε_n-GREEDY information that UCB1 and UCB2 do not have access to. In any real deployment, d would need to be a conservative guess, necessarily smaller than the true minimum gap. A smaller d means slower ε_n decay (since ε_n ∝ 1/d²), which increases regret. The experiments do not explore how ε_n-GREEDY's performance degrades when d is a loose lower bound. This is a significant omission because the requirement to know d is identified in Section 2 as a key weakness of ε_n-GREEDY relative to UCB1.

**Weakness: Single environment type (Bernoulli rewards).** All experiments use Bernoulli reward distributions. While the theorems hold for any distribution with support in [0,1], the experimental validation is limited to the simplest possible non-trivial case. Continuous reward distributions (beta, truncated normal), heavy-tailed distributions, or distributions with different shapes would stress-test the policies in ways these experiments do not. UCB1-TUNED's variance adaptation, in particular, might behave differently for non-Bernoulli distributions where the relationship between mean and variance is not constrained by the Bernoulli functional form.

**Weakness: No experimental validation of UCB1-NORMAL.** Theorem 4 establishes a logarithmic regret bound for normally distributed rewards with unknown variances, but no experiments are reported for UCB1-NORMAL. This is a gap between the theoretical contribution and the empirical evaluation — the theorem is stated and proven, but the reader gains no intuition for how UCB1-NORMAL performs in practice, how it compares to the bounded-support policies when applied to approximately normal rewards, or how sensitive it is to violations of the normality assumption.

**What would strengthen the paper experimentally.** Several experiments are conspicuous by their absence: (1) a direct comparison of UCB1 (the theoretical algorithm) against UCB1-TUNED (the heuristic) to quantify the benefit of variance adaptation; (2) an implementation of Agrawal's (1995) policy as a baseline to assess whether finite-time guarantees correspond to practical gains; (3) experiments on continuous reward distributions (e.g., beta, uniform, truncated normal) to test distributional robustness; (4) experiments with d set to a loose lower bound for ε_n-GREEDY to assess practical sensitivity; (5) shorter-horizon experiments (n = 100, 1,000, 10,000) where the additive constants in the finite-time bounds are expected to dominate; (6) UCB1-NORMAL experiments on normally distributed rewards; (7) error bars or confidence intervals on the 100-run averages to enable statistical comparison between policies.

## 6. Limitations and Trade-offs

### The Difficulty Estimation Tax: Unbounded Overhead for a Key Input

**The assumption or constraint.** The UCB1 and UCB2 policies require no parameter tuning and no prior knowledge of the reward distributions beyond bounded support. However, the regret bounds themselves contain the suboptimality gaps `Δ_i = μ* — μ_i` in the denominator of the leading term — for UCB1, `8 ln n / Δ²_j`, and for UCB2, `(1+α)(1+4α) ln(2e Δ²_j n) / (2 Δ²_j)`. These gaps are **unknown to the algorithm**. The bounds therefore describe the regret as a function of quantities the algorithm cannot observe. A practitioner running UCB1 does not know `Δ_i` and therefore cannot compute the numerical regret guarantee at horizon `n`. The paper provides no mechanism for estimating these gaps online or for recovering a data-dependent bound after the fact.

**The consequence.** This creates a gap between the theoretical guarantee — which is a function of unknown problem parameters — and practical deployment, where a user wants a concrete numerical bound on expected regret before committing to run the algorithm. If the smallest gap `Δ_min` is very small, the bound `8 ln n / Δ²_min` could be enormous, and the user has no way to know whether they are in this regime. The guarantee is **existential but not computable**: it assures the user that logarithmic regret is achieved, but cannot tell them whether the constant will be 100 or 100,000 for their specific problem instance.

This is fundamentally different from the ε_n-GREEDY policy, where the parameter `d` is an explicit lower bound on `Δ_min` that the user must provide, enabling a computable numerical guarantee (the bound in Theorem 3 evaluates to a concrete number given `c`, `d`, `K`, and `n`). The UCB policies' advantage — not requiring `d` as input — comes at the cost of **non-constructive guarantees**. A user choosing between UCB1 and ε_n-GREEDY faces a tradeoff: UCB1 requires no gap knowledge but gives no computable regret bound; ε_n-GREEDY requires gap knowledge but delivers an explicit probability bound.

**What evidence exists in the paper.** The experiments in Section 4 provide post-hoc empirical measurements of regret for specific known distributions (where `Δ_i` is known to the experimenter but not the algorithm), confirming logarithmic behavior across a range of gap sizes. Figures 6–12 show that UCB1-TUNED, UCB2, and ε_n-GREEDY all achieve low regret on distributions with gaps as small as `Δ = 0.1` (distributions 2, 3, 13, 14). However, the experiments cover only a handful of gap values and do not systematically vary `Δ_min` to map out the constant factor. The theoretical bounds tell us that halving `Δ_min` quadruples the leading constant — a sensitivity that is not stress-tested.

**Mitigation status.** The paper does not address this limitation. There is no proposal for online gap estimation, no discussion of data-dependent confidence intervals on regret, and no comparison of the theoretical bounds (plugging in the true `Δ_i`) to the empirical regret to assess tightness. The experiments implicitly acknowledge the gap between theory and practice by replacing UCB1 with UCB1-TUNED — a heuristic with no regret bound at all — suggesting that the authors view the theoretical bounds as qualitative guides rather than quantitative tools for deployment.

---

### UCB1-NORMAL Depends on Unproven Conjectures

**The assumption or constraint.** Theorem 4's regret bound for UCB1-NORMAL — the policy for normally distributed rewards with unknown means and variances — rests on two conjectures that the authors "could only verify numerically":

> **Conjecture 1.** Let X be a Student random variable with s degrees of freedom. Then, for all 0 ≤ a ≤ √(2(s+1)), IP{X ≥ a} ≤ e^{-a²/4}.

> **Conjecture 2.** Let X be a χ² random variable with s degrees of freedom. Then IP{X ≥ 4s} ≤ e^{-(s+1)/2}.

The proof of Theorem 4 (Appendix B) uses Conjecture 1 to bound the probability that the sample mean of a suboptimal arm exceeds its true mean by the exploration bonus (the normal-distribution analogue of the Chernoff-Hoeffding bound used for bounded rewards), and Conjecture 2 to bound the probability that the sample variance overestimates the true variance by more than a factor of 4 (which would inflate the exploration bonus and cause unnecessary exploration). Without these conjectures, the proof collapses — there is no alternative tail bound offered.

**The consequence.** If either conjecture is false (even for some range of degrees of freedom `s` or tail probability `a`), the regret bound of Theorem 4 is unsupported. The constant `256 σ²_i / Δ²_i` in the leading term — which is 32× larger than UCB1's constant for bounded rewards — may not be valid. Worse, the qualitative claim that UCB1-NORMAL achieves logarithmic regret at all depends on these conjectures. A policy that uses a Student-based confidence interval without a valid tail bound cannot guarantee the `t⁻⁴` decay rate in the union bound, which is the linchpin of the proof template shared by all four theorems. The algorithm itself (Figure 4) would still run — it simply computes sample variances and uses them in the exploration bonus — but any user deploying it would be relying on numerically-checked inequalities rather than a mathematical proof.

**What evidence exists in the paper.** The paper states that the conjectures were "only verified numerically" (Appendix B header). No details of the numerical verification are provided — no range of `s` tested, no resolution of the `a` grid, no maximum observed deviation from the conjectured bound. The reader cannot assess how thorough the verification was or whether edge cases (very small `s`, very large `a`) were adequately checked. The experiments in Section 4 do not include UCB1-NORMAL at all — it is a purely theoretical contribution with zero empirical validation. This means there is **no experimental evidence** that the policy works, let alone that the conjectures hold in practice.

**Mitigation status.** The paper does not attempt to prove the conjectures, does not provide a weaker bound that avoids them, and does not run experiments to validate UCB1-NORMAL empirically. The conjectures are presented as open problems, and the theorem is conditional on them. A practitioner interested in normal rewards with unknown variance is left with a policy whose theoretical guarantee is contingent and whose practical performance is untested — a doubly weak foundation for deployment.

---

### ε_n-GREEDY Requires Oracle Knowledge of the Minimum Gap to Function at All

**The assumption or constraint.** Theorem 3 establishes that ε_n-GREEDY achieves `P{I_n = j} ≤ c/(d²n) + o(1/n)` — a polynomial instantaneous regret bound — but **only** when `d` satisfies:

$$0 < d \leq \min_{i: \mu_i < \mu^*} \Delta_i$$

That is, `d` must be a valid lower bound on every suboptimality gap, including the smallest one. The algorithm's exploration schedule `ε_n = cK / (d² n)` depends critically on this parameter: `d` appears squared in the denominator, so underestimating `d` — using a value smaller than the true `Δ_min` — makes `ε_n` proportionally larger, causing more exploration and higher regret. More critically, **overestimating** `d` — using a value larger than `Δ_min` — is catastrophic: `ε_n` decays too quickly, and there may not be enough exploration to reliably identify the optimal arm among near-optimal competitors. The regret could become linear rather than logarithmic.

**The consequence.** In any real deployment, the minimum gap `Δ_min` is unknown — it is precisely what the bandit algorithm is trying to discover. The user must provide `d` as a conservative guess. If the guess is too conservative (`d` much smaller than `Δ_min`), regret is inflated by a factor of `Δ²_min / d²`, which can be large. If the guess is too aggressive (`d > Δ_min`), the regret bound does not hold, and the algorithm may fail entirely. This creates an uncomfortable practical dynamic: the algorithm works best when `d` is as close as possible to `Δ_min`, but setting `d` too close risks crossing the threshold into failure. There is no mechanism within ε_n-GREEDY to adapt `d` online or to detect when the guess is wrong.

This stands in contrast to UCB1 and UCB2, which require no such parameter and adapt automatically to whatever gaps exist. The ε_n-GREEDY policy trades this robustness for the stronger instantaneous guarantee (polynomial decay of misidentification probability rather than logarithmic growth of expected cumulative plays). But the tradeoff is only worthwhile if a valid `d` can be supplied, and the paper provides no guidance on how to obtain one in practice.

**What evidence exists in the paper.** The experiments in Section 4 set `d` to "Δ = μ* — max_{i: μ_i < μ*} μ_i" for every distribution — the **exact** minimum gap, which is the most favorable possible value and cannot be known to the algorithm in practice. The sensitivity analysis focuses exclusively on the parameter `c` (the overall scale of exploration), varying it around its empirically best value in Figures 6–12. There is **no ablation** where `d` is set to a loose lower bound (e.g., `d = 0.01` when the true `Δ_min = 0.1`, or `d = 0.5` when `Δ_min = 0.1`). The paper never tests what happens when the `d` assumption is violated — when `d` overestimates the true minimum gap, which is the failure mode that would cause linear regret.

The paper's own experimental summary acknowledges the practical difficulty: "the choice of c in policy ε_n-GREEDY is difficult as there is no value that works reasonably well for all the distributions that we considered." But `c` sensitivity is a secondary problem — `d` sensitivity is existential. An algorithm that requires the answer to the question it is trying to answer (`Δ_min`) as an input parameter is fundamentally limited in applicability.

**Mitigation status.** The paper does not propose any method for estimating `d`, for adapting `d` online, or for relaxing the requirement. The remark in Section 2 explicitly compares ε_n-GREEDY unfavorably to UCB1 on this point — "unlike Theorems 1 and 2, here we need to know a lower bound d on the difference between the reward expectations" — framing it as a known cost of the stronger guarantee rather than a problem to be solved. The paper offers no future work direction for removing or relaxing the `d` requirement.

---

### No Experimental Validation for the Theoretically Central UCB1 Policy; The Best Practical Policy Has No Theory

**The assumption or constraint.** The paper's theoretical contributions — Theorems 1 through 4 — prove regret bounds for four specific policies: UCB1, UCB2, ε_n-GREEDY, and UCB1-NORMAL. However, the experimental section (Section 4) does not evaluate **UCB1** or **UCB1-NORMAL** at all. Instead, the experiments feature **UCB1-TUNED**, a heuristic variant that replaces UCB1's exploration bonus `√(2 ln t / s)` with a variance-aware version `√(ln n / s × min{1/4, V_j(s)})`, where `V_j(s)` is an upper confidence bound on the true variance. The authors explicitly state they "are not able to prove a regret bound" for UCB1-TUNED. So the policy with the strongest experimental performance has no theoretical guarantee, and the policy with the cleanest theoretical guarantee (UCB1) has no experimental validation.

**The consequence.** A practitioner reading this paper faces a dilemma. The theoretical results say: use UCB1 to get a proven logarithmic regret bound with no parameter tuning and no prior knowledge. The experimental results say: UCB1-TUNED performs "substantially better than UCB1 in essentially all of our experiments" (Section 4), but no regret bound is available. Which should be deployed? The paper provides no basis for answering this question quantitatively. It cannot say how much better UCB1-TUNED performs than UCB1 in terms of regret — only that it is "substantially" better — and it cannot say whether UCB1-TUNED ever fails catastrophically (e.g., linear regret) on some distribution where UCB1's guarantee would have protected the user.

The gap is particularly problematic because UCB1-TUNED's variance estimation introduces a new potential failure mode: if `V_j(s)` underestimates the true variance (because the variance upper confidence bound fails), the exploration bonus could be too small, causing premature convergence to a suboptimal arm. The Chernoff-Hoeffding bound that underpins UCB1's bonus is deterministic and distribution-free — it never underestimates the required exploration. UCB1-TUNED's bonus relies on an estimated variance that could, with some probability, be too low. Without a regret bound, the user cannot quantify this risk.

**What evidence exists in the paper.** The paper's claim that UCB1-TUNED outperforms UCB1 rests on the authors' assertion, not on plotted data. UCB1 does not appear in Figures 6–12. The comparison is indirect: UCB1-TUNED outperforms UCB2 (which is plotted and has a better asymptotic constant than UCB1), so by transitivity UCB1-TUNED must outperform UCB1. But this transitivity is not rigorous — UCB2 and UCB1 have different additive constants and different epoch structures, so UCB1 might behave differently at finite horizons in ways the asymptotic constant comparison doesn't capture. A direct UCB1 vs. UCB1-TUNED comparison on the seven distributions would cost nothing (since UCB1 is trivial to implement) and would ground the paper's central practical claim.

**Mitigation status.** The paper does not acknowledge this as a limitation. UCB1-TUNED is introduced in Section 4 without discussion of the gap between its empirical success and the lack of a theoretical guarantee. The authors' phrasing — "we are not able to prove a regret bound" — implies that proving one is desirable but difficult, not that the absence is a weakness of the current paper. However, for a paper whose primary contribution is finite-time theoretical guarantees, shipping a policy without such a guarantee as the recommended practical choice undercuts the central thesis that finite-time bounds matter for deployment.

---

### Single Distribution Family and No Real-World Data

**The assumption or constraint.** All experiments in Section 4 use Bernoulli reward distributions — the simplest possible non-trivial bandit setting, where each pull returns either 0 or 1. The seven distributions (1–3 and 11–14 in the table) vary the means and number of arms, but all are Bernoulli. The paper contains **no experiments with continuous reward distributions** (beta, truncated normal, uniform), **no experiments with heavy-tailed distributions** (where the Chernoff-Hoeffding bound's variance assumption is conservative), **no experiments with non-stationary rewards** (despite noting in Section 2 that Theorems 1–3 hold under the weaker martingale assumption), and **no experiments on real-world bandit data** (clinical trials, A/B testing, recommender system logs).

**The consequence.** The theoretical results are distribution-free — UCB1 and UCB2 are proven to achieve logarithmic regret for **any** distribution with support in `[0,1]`. This generality is a major selling point of the paper, distinguishing it from Lai and Robbins' parametric framework. But the experimental validation is entirely within the Bernoulli family, which has special properties not shared by arbitrary bounded distributions:

- **Bernoulli variance is a deterministic function of the mean** (`σ² = μ(1-μ)`), so the variance-mean relationship is fixed and known. For continuous distributions, variance and mean are independent parameters — an arm could have mean 0.5 and variance 0.01 (tightly concentrated) or mean 0.5 and variance 0.25 (highly dispersed). UCB1-TUNED's variance estimation is specifically designed to handle this, but it is never tested in a setting where mean and variance are decoupled.
- **Bernoulli rewards have the maximum possible variance for their mean**, making them a worst case for exploration. Continuous distributions concentrated near their mean would require less exploration, and the fixed bonus of UCB1 would be more conservative than necessary — but how much more? The experiments cannot answer.
- **The Chernoff-Hoeffding bound** used in UCB1 is tightest for Bernoulli(1/2) variables and loosest for distributions concentrated near 0 or 1. Real-world rewards (click-through rates, conversion probabilities, revenue per user) are often highly skewed, with means far from 1/2 and variances far below the Bernoulli maximum. The paper's experiments systematically explore means from 0.45 to 0.9 — a range where Bernoulli variance ranges from 0.2475 to 0.09 — but never test a distribution with, say, mean 0.5 and variance 0.01, which would stress-test whether UCB1's worst-case bonus is overly conservative.

**What evidence exists in the paper.** None. The experimental design is entirely synthetic and entirely Bernoulli. The authors do not justify this choice beyond presenting it as the natural setting for testing bandit algorithms. The omission is particularly noticeable for UCB1-NORMAL (Theorem 4), which is specifically designed for normally distributed rewards — yet no experiments with normal rewards are reported. The policy exists only as a theorem.

**Mitigation status.** The paper does not acknowledge the limitation and does not suggest that future work should validate the policies on other distribution families. The robustness claims in Section 2 — that Theorems 1–3 hold under "the weaker assumption that IE [X_{i,t} | X_{i,1}, …, X_{i,t-1}] = μ_i" — are purely theoretical and untested experimentally. A practitioner deploying UCB1 in a non-Bernoulli setting has no empirical evidence that the finite-time behavior resembles the Bernoulli results.

---

### The Gap Between Theoretical Optimality and Practical Performance Remains Uncharacterized

**The assumption or constraint.** Theorem 2 proves that UCB2's regret bound approaches the information-theoretic optimum `1/(2Δ²_j)` as the parameter `α → 0`. The paper explicitly notes that this comes at the cost of a diverging additive constant: "c_α → ∞ as α → 0." The suggested remedy is to "let α = α_n be slowly decreasing with the number n of plays" — a horizon-dependent schedule that could theoretically approach the optimal constant without the additive term exploding prematurely. However, the paper **never implements or tests this α_n scheme**. The experiments fix α = 0.001 for all horizons, and under this fixed setting, UCB2 performs "similarly to UCB1-TUNED, but always slightly worse" (Section 4.1).

**The consequence.** The central theoretical refinement of UCB2 over UCB1 — the ability to approach the optimal constant — is **never demonstrated empirically**. The experiments show that at a fixed α = 0.001 and horizon n = 100,000, UCB2 is slightly worse than a heuristic variant of UCB1. This raises the question: is there *any* horizon n and any α-setting where UCB2 outperforms UCB1-TUNED? The paper provides no evidence that the asymptotic advantage of UCB2's tighter constant ever materializes in practice. If UCB2 cannot beat UCB1-TUNED at n = 100,000, when would it? n = 10⁶? 10⁹? The user cannot know.

More fundamentally, the paper does not characterize the **practical regime where the asymptotic constant matters**. The theoretical bounds suggest that the difference between UCB1's `8/Δ²_j` and UCB2's `≈ 1/(2Δ²_j)` is a factor of ~16 in the leading coefficient. If both policies achieve logarithmic regret, and UCB1's constant is 16× larger, then at sufficiently large n, UCB2 must dominate. But "sufficiently large" could be far beyond any practical horizon, especially for small Δ_j. The experiments at n = 100,000 show the opposite — UCB2 trailing — which suggests that the additive constant c_α is still dominant at this horizon. Without testing larger n or the α_n scheduling scheme, the paper cannot tell the reader when (if ever) UCB2 becomes preferable to UCB1 or UCB1-TUNED.

**What evidence exists in the paper.** Figures 6–12 consistently show UCB2 slightly worse than UCB1-TUNED. Figure 5 shows that UCB2 is "relatively insensitive to the choice of its parameter α, as long as it is kept relatively small," but this is tested only on distribution 2 and only at fixed-horizon n = 100,000. The α_n scheme mentioned in the Theorem 2 remark is never evaluated. There is no plot showing regret as a function of n for different fixed α values to assess whether smaller α eventually overtakes larger α at very large n.

**Mitigation status.** The paper acknowledges the α-c_α tradeoff explicitly in the Theorem 2 remark ("the two terms in the sum can be traded-off by letting α = α_n be slowly decreasing with the number n of plays") and in the experimental section (noting that a fixed α = 0.001 was used). But the acknowledgment is theoretical — it does not provide actionable guidance. A practitioner wanting to use UCB2 must choose α, and the paper gives no method for choosing it optimally given a target horizon n. The experimental conclusion that UCB1-TUNED is preferable to UCB2 in all tested settings suggests that the practical value of UCB2's asymptotic refinement may be limited, but the paper does not state this conclusion.
