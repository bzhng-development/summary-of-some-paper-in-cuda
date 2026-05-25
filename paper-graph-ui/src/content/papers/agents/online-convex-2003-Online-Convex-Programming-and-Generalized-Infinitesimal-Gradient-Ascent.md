# Online Convex Programming and Generalized Infinitesimal Gradient Ascent

**URL:** [https://www.cs.cmu.edu/~maz/publications/techconvex.pdf](https://www.cs.cmu.edu/~maz/publications/techconvex.pdf)

## 🎯 Pitch

This paper introduces and formally defines **online convex programming**—a repeated optimization setting where a convex feasible set is known but the convex cost function at each step is revealed only after a decision is made—and proposes the **Greedy Projection** algorithm (projected gradient descent with a decreasing learning rate $\eta_t = t^{-1/2}$) to solve it.

---

## 1. Executive Summary

This paper introduces and formally defines **online convex programming**—a repeated optimization setting where a convex feasible set is known but the convex cost function at each step is revealed only after a decision is made—and proposes the **Greedy Projection** algorithm (projected gradient descent with a decreasing learning rate $\eta_t = t^{-1/2}$) to solve it. The paper establishes a regret bound of $\|\mathcal{F}\|^2 \frac{\sqrt{T}}{2} + (\sqrt{T} - \frac{1}{2}) \|\nabla c\|^2$, proving that the average regret approaches zero, and demonstrates that the algorithm is a generalization of infinitesimal gradient ascent for repeated games. Applying this framework yields **Generalized Infinitesimal Gradient Ascent (GIGA)**—an extension of two-action gradient ascent to games with arbitrarily many actions—and the paper proves GIGA is **universally consistent**, establishing that gradient-based learning converges to no-regret behavior against any environment, uniformly over all adversaries.

## 2. Context and Motivation

### The Core Problem: Decisions Before Information

The fundamental question this paper tackles is deceptively simple: **how should you make decisions over time when you don't know the costs of your choices until after you've committed to them?** This problem is ubiquitous. The paper opens with a concrete example that captures its essence:

> "Imagine a farmer who decides what to plant each year. She has certain restrictions on her resources, both land and labour, as well as restrictions on the output she is allowed to produce. How can she select which crops to grow without knowing in advance what the prices will be?"

The farmer knows her constraints (the land she has, the labor available, quotas she must respect), but the prices—her ultimate payoff—are revealed only after planting and harvest. She cannot wait to observe prices and then retroactively plant the optimal crop mix. She must commit first, observe the outcome, and then use that information to plan the next year.

This is not an isolated agricultural example. The same structure appears across an enormous range of practical domains that the paper highlights: factory production planning (where you build goods before knowing demand), portfolio allocation (where you invest before observing returns), and many other industrial optimization problems "where one is unaware of the value of the items produced until they have already been constructed."

At its core, this represents a fundamental departure from classical optimization. In classical **convex programming**, you know everything in advance: the feasible set (what you're allowed to do) and the cost function (what it costs to do each thing). You compute the optimal point once and you're done. In **online convex programming**—the problem this paper formalizes—the feasible set is known, but the cost functions arrive sequentially and are revealed only after each decision. You cannot simply "solve" the problem; you must interact with it over time, adapting as information arrives.

### Formalizing the Gap: From Experts to Convex Functions

The paper positions itself within a well-studied lineage of online learning problems, but identifies a critical generalization gap.

#### The Experts Problem: The Known Foundation

The **experts problem** (attributed to Littlestone and Warmuth, 1989; Freund and Schapire, 1999) had been extensively studied. In this setting, you have a fixed set of $n$ experts, each recommending a course of action at each step. At each round, you select a probability distribution over the experts, observe the cost incurred by each expert, and suffer the expected cost of your distribution. The goal is to achieve regret that is sublinear in $T$—meaning your average cost approaches the average cost of the best single expert in hindsight.

The experts problem is, in fact, a special case of what this paper studies. If you treat $x \in \mathbb{R}^n$ as a probability distribution (each component $x_i$ is the probability of following expert $i$), then the set of all such distributions is the $n$-dimensional simplex, which is convex. The cost function on this set is linear—the expected cost is simply the dot product of your distribution with the vector of expert costs—and linear functions are convex. So the experts problem is an **online linear programming** problem on the simplex.

But the experts framework has hard limitations that motivated the search for a more general theory.

#### Where the Experts Framework Falls Short

**1. The feasible set is restricted to the simplex.** In the farmer example, constraints involve land allocation (a bounded polytope, but not a simplex), labor hours, and output quotas. These define a convex feasible region that is not a probability simplex. You cannot represent "plant 40% wheat, 30% corn, 30% soybeans" combined with "total land = 100 acres, labor ≤ 500 hours" as a distribution over a finite set of experts.

**2. The cost functions are restricted to be linear.** The experts framework assumes each expert incurs a known cost, and your cost is a linear combination of expert costs. But real optimization problems often involve **truly nonlinear convex costs**. The paper gives a concrete example: $F = \{x : x \cdot x \leq 1\}$ (the unit ball) with cost $c(x) = x \cdot x$ (the squared distance from the origin). The minimum of this cost is at the center of the feasible set, not at any vertex. Linear cost functions on a convex polytope always have minima at vertices, so an experts algorithm that only considers vertices would miss the optimal interior point.

**3. Experts algorithms produce bounds that depend on the number of experts.** As the paper notes in Section 4, if you naïvely translate a convex problem into an experts problem by enumerating vertices, the number of vertices can be enormous—unrelated to the diameter of the feasible set. The bounds from experts algorithms would scale with the number of vertices, whereas the bound this paper proves scales with the diameter $\|\mathcal{F}\|$ and the maximum gradient norm $\|\nabla c\|$, both of which can be small even for polytopes with exponentially many vertices.

**4. Convex regions can be curved.** The feasible set might be a sphere or an ellipsoid, which cannot be represented as the convex hull of any finite number of points. Discretizing such a set with a sequence of increasingly fine polytopes is possible in principle, but the paper calls this approach "very undesirable" because it introduces complexity that the direct convex treatment avoids entirely.

### The Need for a General Online Convex Programming Theory

The paper articulates the need for an algorithm that operates directly on **arbitrary convex sets** with **arbitrary convex cost functions**, without reduction to discrete experts or enforced linearity. The problem was not merely of theoretical interest; it was the natural mathematical structure underlying a broad class of practical sequential decision problems.

However, at the time of writing (2003), no such theory existed. Online learning research had developed sophisticated algorithms for the experts problem and for various prediction settings, but no one had formulated the general convex programming case or provided an algorithm with provable regret bounds for it.

The paper identifies three specific advantages of its approach over the state of the art (Section 1):

> "The first is that gradient descent is a simple, natural algorithm that is widely used, and studying its behavior is of intrinsic value. Secondly, this algorithm is more general than the experts setting, in that it can handle an arbitrary sequence of convex functions, which has yet to be solved. Finally, in online linear programs this algorithm can in some circumstances perform better than an experts algorithm."

### Conflicting Pressures in Algorithm Design

A subtle tension in prior work that the paper identifies is between **static** and **dynamic** algorithm designs. The authors explicitly contrast their approach with that of Kalai and Vempala (2002):

> "They are attempting to make the algorithm behave in a lazy fashion, changing its vector slowly, whereas here we are attempting to be more dynamic, as is highlighted in sections 2.2 and 3.4."

This tension is important because it reflects different assumptions about the environment. If the optimal decision changes over time (as in the dynamic regret analysis of Section 2.2), a lazy algorithm that changes slowly might fall behind. Conversely, a highly dynamic algorithm might overreact to noise. The paper explores both: it provides bounds for static regret (competing against a single fixed point) in the main Theorem 1, bounds for dynamic regret (competing against sequences of bounded path length) in Theorem 2, and an alternative "lazy projection" variant in Theorem 3 that achieves better static regret at the cost of being less responsive to changes.

### The Repeated Games Connection: A Separate Lineage

The paper is motivated not only by the gap in online optimization theory, but also by a specific algorithmic puzzle in game theory: **infinitesimal gradient ascent**.

Singh, Kearns, and Mansour (2000) had proposed infinitesimal gradient ascent as an algorithm for two-player, two-action repeated games. The algorithm performs gradient ascent in the space of mixed strategies: at each round, compute the gradient of the utility with respect to your mixed strategy (which is simply the vector of payoffs against the opponent's observed action), take a step in that direction, and project back onto the simplex (since mixed strategies must be valid probability distributions).

This algorithm worked well in practice and had some theoretical properties, but it was limited in two critical ways:

1. **It only applied to two-action games.** The projection onto the simplex for $n$ actions is more complex than for two actions, and it was not obvious how to extend the algorithm or the analysis.

2. **It had not been proven universally consistent.** **Universal consistency** is the gold-standard property in repeated games (formalized by Fudenberg and Levine, 1995, 1998). A behavior is universally consistent if, for any $\epsilon > 0$, there exists a time $T$ such that, against *any* environment, after time $T$ the average regret never again exceeds $\epsilon$ with probability at least $1 - \epsilon$. The key phrase is "for any environment"—this means the algorithm's convergence to no-regret behavior is uniform over all adversaries, including those that adapt to your past play.

Proving universal consistency is hard precisely because adaptive adversaries can exploit patterns in your behavior. If your algorithm depends on its own past actions in a way that an adversary can learn, the adversary might manipulate you into making suboptimal choices indefinitely. The paper notes that "not all algorithms are self-oblivious" and gives the example of Kalai and Vempala's algorithm, which "uses a 'random seed' at the beginning that an adaptive adversary could learn over time and then use in some settings."

Infinitesimal gradient ascent was known to converge to Nash equilibrium in certain classes of games, but whether it was universally consistent—whether gradient-based learning in games guarantees no-regret behavior against any opponent—was an open question.

### Bridging Two Communities

The paper's central intellectual move is to recognize that these two motivations—the gap in online optimization theory and the open problem in repeated games—are actually the same problem viewed from different angles.

A repeated game, from the perspective of one player, is precisely an online linear programming problem. The player's feasible set is the simplex of mixed strategies. The cost (or negative utility) at each round is a linear function: when the opponent plays action $h_{t,2}$, the utility of playing mixed strategy $x$ is $\sum_{i} x_i \cdot u(i, h_{t,2})$, which is a linear function of $x$.

Therefore, if one can develop a general algorithm for online convex programming with provable regret bounds, applying it to the simplex with linear cost functions immediately yields a learning algorithm for repeated games. The regret bound from the online convex programming analysis translates directly into a bound on the expected regret against the best fixed action in hindsight, which is the core building block for proving universal consistency.

This is exactly the path the paper follows: develop the Greedy Projection algorithm and Theorem 1 for general online convex programming, specialize it to the simplex with linear utilities to obtain **Generalized Infinitesimal Gradient Ascent (GIGA)** (Section 3.3), prove a bound on expected regret against oblivious deterministic environments (Theorem 4), and then leverage the concept of **self-oblivious behavior** to extend the bound to all environments, establishing universal consistency (Lemma 1, Section 3.4).

### The Significance of Gradient Descent

The paper emphasizes that gradient descent is "a simple, natural algorithm that is widely used" in artificial intelligence and machine learning. Proving that it has strong theoretical guarantees in adversarial online settings—and, in particular, that it yields universally consistent behavior in games—connects practical AI techniques with theoretical game theory in a way that was not previously established.

The result matters because it says something non-obvious: following the gradient—a local, myopic update that only considers the most recent cost function—is sufficient to guarantee that, over time, you will not regret not having played a single fixed action all along. This is not obvious because gradient descent can, in principle, be led astray by an adversary that constructs cost functions designed to push the algorithm toward suboptimal regions. The theorem shows that if the adversary plays such a game, the gradient steps are small enough (due to the decreasing learning rate $\eta_t = t^{-1/2}$) that the algorithm cannot be jerked around too aggressively, and the projection step ensures it never leaves the feasible set.

### How This Paper Positions Itself

The paper positions itself at the intersection of three research communities and claims contributions to each:

1. **Online convex optimization (new problem):** It defines the problem class, proposes algorithms (Greedy Projection and Lazy Projection), and proves regret bounds that depend on the diameter of the feasible set and the gradient bound, not on the dimensionality or number of vertices.

2. **Online linear optimization (better bounds):** It shows that, for convex polytopes, the Greedy Projection bound can be incomparable to experts-based bounds—and potentially much smaller—because it depends on geometric properties rather than vertex count.

3. **Repeated games and multiagent learning (new algorithm + proof):** It extends infinitesimal gradient ascent from two-action to arbitrary-action games (GIGA) and proves universal consistency, resolving an open question about gradient-based learning in games.

The paper explicitly acknowledges related but distinct lines of work and distinguishes its contributions. It notes that Cesa-Bianchi, Long, and Warmuth (1994), Kivinen and Warmuth (1997, 2001), and Herbster and Warmuth (2001) studied gradient descent for **online prediction** problems where the loss functions are convex Bregman divergences—a specific, structured class of convex functions. In contrast, "in this paper, we are considering arbitrary convex functions, in problems that may or may not involve prediction." The offline case of gradient descent with Bregman divergences was studied by Della Pietra, Della Pietra, and Lafferty (1999), but the paper's focus is entirely on the adversarial online setting.

The work of Kalai and Vempala (2002) on online linear programming is the closest prior work, but the paper distinguishes its approach along the lazy-versus-dynamic dimension discussed above, and generalizes from linear to arbitrary convex costs.

## 3. Technical Approach

### 3.1 Reader Orientation

This is a **theoretical algorithm design paper** that develops a concrete procedure—projected gradient descent with a decreasing learning rate—for making sequential decisions from a convex set when convex cost functions are revealed only after each decision, and proves mathematically that this procedure achieves vanishing average regret compared to the best fixed decision in hindsight.

The paper builds a **unified algorithmic framework** that takes any online convex programming problem (a bounded, closed, nonempty convex set plus an adversarial sequence of convex cost functions) and transforms it into a sequence of decisions whose cumulative cost approaches the minimum possible, without requiring any statistical assumptions about how the cost functions are generated.

### 3.2 Big-Picture Architecture (Diagram in Words)

The Greedy Projection algorithm consists of three simple components that interact in a tight loop:

1. **Gradient Oracle**: Given the current decision point `$x_t \in \mathcal{F}$` and the just-revealed cost function `$c_t$`, this component computes the gradient `$\nabla c_t(x_t)$`—the direction of steepest increase in cost at the point that was actually played.

2. **Gradient Step**: The algorithm takes a step opposite the gradient (since we want to minimize cost, not maximize it), scaling the step by a time-decreasing learning rate `$\eta_t = t^{-1/2}$`. This produces a raw candidate point `$y_{t+1} = x_t - \eta_t \nabla c_t(x_t)$` in `$\mathbb{R}^n$` that may lie outside the feasible set.

3. **Projection Operator**: The candidate point `$y_{t+1}$` is projected back to the closest point in the feasible set: `$x_{t+1} = P(y_{t+1}) = \arg\min_{x \in \mathcal{F}} d(x, y_{t+1})$`. This ensures the next decision is always feasible.

Information flows in a strict cycle: the current feasible point `$x_t$` is played → the adversary reveals `$c_t$` → the gradient oracle computes `$\nabla c_t(x_t)$` → the point is updated to `$y_{t+1}$` via gradient step → `$y_{t+1}$` is projected to `$x_{t+1}$` inside `$\mathcal{F}$` → the cycle repeats. There is no memory beyond the current point; the algorithm is Markovian with respect to its position in the feasible set.

The paper also presents a variant, **Lazy Projection**, that separates the gradient accumulation from the projection: it accumulates all gradients in an unconstrained vector `$y_t$` and only projects to `$\mathcal{F}$` when needed to produce the actual decision `$x_t$`. This design trades off responsiveness for tighter static regret bounds.

### 3.3 Roadmap for the Deep Dive

- **First**, the paper's formal model of online convex programming—the feasible set assumptions, the cost function assumptions, and the precise definition of regret—since the regret bound is meaningless without these foundations.
- **Second**, the Greedy Projection algorithm itself, including the learning rate schedule `$\eta_t = t^{-1/2}$` and the projection operator, because the entire subsequent analysis revolves around this procedure.
- **Third**, the proof structure of Theorem 1 (the static regret bound), as the proof reveals the core analytical technique—linearization of convex functions, telescoping sum of squared distances, and the critical inequality `$(P(y) - x)^2 \leq (y - x)^2$`.
- **Fourth**, the Lazy Projection variant and its separate two-potential analysis, since it illustrates a fundamentally different way to think about the algorithm's behavior (ideal potential vs. projection potential).
- **Fifth**, the reduction from online convex programming with arbitrary convex functions to online linear programming, which is the key simplification that makes the analysis tractable and connects to the repeated games application.
- **Sixth**, the dynamic regret extension (competing against sequences with bounded total variation), as it represents the "dynamic" design philosophy the paper advocates.

### 3.4 Detailed, Sentence-Based Technical Breakdown

This is primarily a **theoretical analysis paper** whose core idea is that projected gradient descent with an appropriately decaying learning rate achieves sublinear regret against an arbitrary sequence of convex cost functions, and that this abstract result, when specialized to the simplex, proves that gradient-based learning in repeated games is universally consistent.

---

#### The Online Convex Programming Model

The paper defines online convex programming through a set of seven explicit assumptions that together specify what the algorithm knows, what it can compute, and what it is up against. Understanding these assumptions is essential because every part of the regret analysis depends on them.

**The feasible set `$\mathcal{F}$`** is a subset of `$\mathbb{R}^n$` that satisfies three structural conditions (Assumptions 1-3): it is **bounded** (there exists a finite diameter `$N$` such that all pairwise distances are at most `$N$`), **closed** (all limit points of sequences in `$\mathcal{F}$` are themselves in `$\mathcal{F}$`), and **nonempty**. These conditions ensure that the projection operator—which the algorithm relies on at every step—is always well-defined and unique. A closed, bounded, nonempty convex set in `$\mathbb{R}^n$` guarantees that for any point `$y$` in `$\mathbb{R}^n$`, there exists exactly one point `$x$` in `$\mathcal{F}$` that minimizes the Euclidean distance to `$y$`. The paper explicitly defines the diameter:

$$\|\mathcal{F}\| = \max_{x, y \in \mathcal{F}} d(x, y)$$

where `$d(x, y) = \|x - y\| = \sqrt{(x-y) \cdot (x-y)}$` is the Euclidean distance. This quantity `$\|\mathcal{F}\|$` appears in the leading term of the regret bound and measures the "size" of the decision space—intuitively, how far the algorithm might start from the optimal point.

**The cost functions `$c_t : \mathcal{F} \to \mathbb{R}$`** satisfy three operational conditions (Assumptions 4-6): each `$c_t$` is **convex** (Definition 2) and **differentiable** (Assumption 4), there exists a **uniform bound on gradient norms** (Assumption 5: `$\|\nabla c_t(x)\| \leq \|\nabla c\|$` for all `$t$` and all `$x \in \mathcal{F}$`), and there exists an **algorithm to compute the gradient** at any queried point (Assumption 6). The convexity assumption means that for any two points `$x, y \in \mathcal{F}$` and any `$\lambda \in [0,1]$`:

$$\lambda c_t(x) + (1-\lambda) c_t(y) \geq c_t(\lambda x + (1-\lambda) y)$$

Geometrically, the function lies below its chords; equivalently, the gradient provides a linear lower bound:

$$c_t(y) \geq c_t(x) + \nabla c_t(x) \cdot (y - x) \quad \text{for all } x, y \in \mathcal{F}$$

This inequality is the **single most important property** in the entire paper. It allows the analysis to replace an arbitrary convex function with its first-order Taylor approximation at the played point, with the guarantee that the actual cost is at least as large as the linear approximation. This linearization step (which reduces arbitrary convex costs to linear costs in the analysis) is what makes the regret bound tractable.

The uniform gradient bound `$\|\nabla c\|$` acts as a Lipschitz constant for the cost functions. It limits how much the cost can change as the decision point moves, which in turn limits how much damage the adversary can inflict in a single round. The quantity `$\|\nabla c\|$` appears squared in the second term of the regret bound.

**The projection oracle (Assumption 7)** is the computational primitive the algorithm requires: given any point `$y \in \mathbb{R}^n$`, we must be able to compute `$P(y) = \arg\min_{x \in \mathcal{F}} d(x, y)$`, the unique closest point to `$y$` in `$\mathcal{F}$`. For many convex sets of practical interest (simplices, Euclidean balls, boxes, halfspaces), this projection can be computed efficiently. The paper takes this as a black-box primitive; the regret analysis only uses the property that projection is non-expansive: `$\|P(a) - P(b)\| \leq \|a - b\|$` for any `$a, b \in \mathbb{R}^n$`, and more specifically that for any `$x \in \mathcal{F}$`:

$$(P(y) - x)^2 \leq (y - x)^2$$

This inequality (proven in Gentile and Warmuth, 2000) says that projection never pushes a point farther from any feasible target `$x$` than the original unprojected point was. It is the geometric fact that makes the regret analysis possible: after gradient step and projection, the algorithm's squared distance to any fixed reference point has not increased beyond what the gradient step alone would have caused.

---

#### The Greedy Projection Algorithm

The algorithm itself is remarkably simple, consisting of exactly two lines of pseudocode (Algorithm 1):

**Initialization:** Select an arbitrary starting point `$x_1 \in \mathcal{F}$` and a sequence of learning rates `$\eta_1, \eta_2, \ldots \in \mathbb{R}^+$`.

**Update rule (for each time step `$t$`):**

$$x_{t+1} = P\left(x_t - \eta_t \nabla c_t(x_t)\right)$$

where `$P(\cdot)$` is the Euclidean projection onto `$\mathcal{F}$`.

**What this does operationally:** At step `$t$`, the algorithm receives the cost function `$c_t$` (which was unknown when `$x_t$` was chosen), computes the gradient of `$c_t$` at the point that was just played, takes a step of size `$\eta_t$` in the negative gradient direction (descending the cost landscape), and projects the result back into the feasible set if the step carried it outside. The next decision `$x_{t+1}$` becomes the starting point for the subsequent round.

**The learning rate schedule `$\eta_t = t^{-1/2}$`** is the specific choice that yields the tightest bound in Theorem 1. The decreasing schedule is essential because it balances two competing forces:

- **Large learning rates early** allow the algorithm to move quickly toward good regions when it is likely far from the optimum. If the initial point `$x_1$` is on the opposite side of `$\mathcal{F}$` from the best fixed point, large steps help close the gap.

- **Small learning rates later** prevent the algorithm from overreacting to individual cost functions and being jerked around by an adversary. Since the regret sums over `$T$` rounds, and the cumulative effect of gradient steps grows with `$\sum \eta_t^2$`, a decaying learning rate ensures this sum is sublinear in `$T$`.

The `$t^{-1/2}$` decay rate is the natural choice that emerges from the analysis: it makes `$\sum_{t=1}^T \eta_t \approx 2\sqrt{T}$` (the beneficial term from moving toward the optimum) while keeping `$\sum_{t=1}^T \eta_t^2 \approx \log T$`. However, the paper's proof actually works for any non-increasing sequence `$\eta_t$`, with the final bound depending on `$1/\eta_T$` and `$\sum \eta_t$`. The `$t^{-1/2}$` choice optimizes this tradeoff.

**Design choice: Why projected gradient descent?** The paper identifies three motivations. First, gradient descent is "a simple, natural algorithm that is widely used, and studying its behavior is of intrinsic value." Second, it handles arbitrary convex functions directly, without the reduction to experts that would be required by prior approaches. Third, the regret bounds scale with geometric quantities (diameter and gradient bound) rather than combinatorial quantities (number of vertices), which can be substantially smaller.

**The geometric intuition** behind the algorithm is described through the valley metaphor. If we imagine the feasible set `$\mathcal{F}$` as the floor of a valley and the cost function `$c_t$` as the altitude, the gradient points uphill. Moving opposite the gradient is walking downhill. The projection step corresponds to "skirting the edges of the valley"—if the downhill step would take you outside the feasible floor, you slide along the boundary instead.

---

#### The Static Regret Bound (Theorem 1)

Theorem 1 is the paper's central technical result. With learning rate `$\eta_t = t^{-1/2}$`, the regret of Greedy Projection satisfies:

$$R_G(T) \leq \|\mathcal{F}\|^2 \frac{\sqrt{T}}{2} + \left(\sqrt{T} - \frac{1}{2}\right) \|\nabla c\|^2$$

where `$R_G(T) = \sum_{t=1}^T c_t(x_t) - \min_{x \in \mathcal{F}} \sum_{t=1}^T c_t(x)$` is the cumulative regret after `$T$` rounds.

**Meaning of the bound:** Both terms grow as `$\sqrt{T}$`, so the average regret `$R_G(T)/T$` decays as `$1/\sqrt{T}$` and approaches zero as `$T \to \infty$`. The first term `$\|\mathcal{F}\|^2 \sqrt{T}/2$` depends on the diameter of the feasible set and represents the cost of possibly starting far from the optimal point. The second term `$(\sqrt{T} - 1/2) \|\nabla c\|^2$` depends on the maximum gradient magnitude and represents the cumulative cost of responding to cost functions after seeing them rather than knowing them in advance.

**The proof structure** is the intellectual core of the paper. It proceeds in four conceptual steps, each reducing the problem to a simpler one:

**Step 1: Linearization.** The analysis first shows that, without loss of generality, we can assume all cost functions are linear. For any sequence of convex cost functions `$\{c_1, c_2, \ldots\}$` and the points `$\{x_1, x_2, \ldots\}$` generated by the algorithm, define `$g_t = \nabla c_t(x_t)$`. Consider replacing each `$c_t$` with the linear function `$\tilde{c}_t(x) = g_t \cdot x$`. The algorithm's behavior would be identical because it only ever evaluates `$\nabla c_t(x_t)$`, and the gradient of `$\tilde{c}_t$` at `$x_t$` is also `$g_t$`. Moreover, the actual regret is at most the regret under the linearized functions, because for any fixed `$x^* \in \mathcal{F}$`, convexity gives:

$$c_t(x_t) - c_t(x^*) \leq g_t \cdot x_t - g_t \cdot x^*$$

So if we prove a bound for linear cost functions, the same bound holds for arbitrary convex functions. This reduction is non-obvious and powerful: the adversary's only power against this algorithm is to choose the gradient vectors `$g_t$`; the nonlinear shape of the cost function beyond its first-order behavior at `$x_t$` is irrelevant.

**Step 2: Distance tracking.** Define `$y_{t+1} = x_t - \eta_t g_t$` (the point before projection). For any fixed comparison point `$x^* \in \mathcal{F}$`, we track the squared distance from `$x^*$` to the algorithm's points. Starting from the identity:

$$y_{t+1} - x^* = (x_t - x^*) - \eta_t g_t$$

we expand the squared norm:

$$(y_{t+1} - x^*)^2 = (x_t - x^*)^2 - 2\eta_t (x_t - x^*) \cdot g_t + \eta_t^2 \|g_t\|^2$$

The paper observes a structural pattern: `$(x_t - x^*)^2$` acts as a potential, `$2\eta_t (x_t - x^*) \cdot g_t$` is the immediate per-round regret (scaled by `$2\eta_t$`), and `$\eta_t^2 \|g_t\|^2$` is the error term.

Rearranging to isolate the per-round cost:

$$(x_t - x^*) \cdot g_t = \frac{1}{2\eta_t} \left((x_t - x^*)^2 - (y_{t+1} - x^*)^2\right) + \frac{\eta_t}{2} \|g_t\|^2$$

**Step 3: Projection inequality.** The projection step replaces `$y_{t+1}$` with `$x_{t+1} = P(y_{t+1})$`. The critical geometric fact (from Gentile and Warmuth, 2000) is:

$$(x_{t+1} - x^*)^2 \leq (y_{t+1} - x^*)^2$$

This means that projection can only help—it never pushes the algorithm farther from the optimal point. Substituting this into the per-round bound:

$$(x_t - x^*) \cdot g_t \leq \frac{1}{2\eta_t} \left((x_t - x^*)^2 - (x_{t+1} - x^*)^2\right) + \frac{\eta_t}{2} \|g_t\|^2$$

**Step 4: Telescoping sum with learning rate adjustment.** Summing over `$t = 1$` to `$T$`, the distance terms telescope, but with different weights `$1/\eta_t$` at each step. The total regret is:

$$R_G(T) = \sum_{t=1}^T (x_t - x^*) \cdot g_t \leq \sum_{t=1}^T \left[\frac{1}{2\eta_t} \left((x_t - x^*)^2 - (x_{t+1} - x^*)^2\right) + \frac{\eta_t}{2} \|g_t\|^2\right]$$

The telescoping sum with varying weights requires careful handling. The paper expands it as:

$$\frac{1}{2\eta_1}(x_1 - x^*)^2 - \frac{1}{2\eta_T}(x_{T+1} - x^*)^2 + \frac{1}{2} \sum_{t=2}^T \left(\frac{1}{\eta_t} - \frac{1}{\eta_{t-1}}\right)(x_t - x^*)^2 + \frac{\|\nabla c\|^2}{2} \sum_{t=1}^T \eta_t$$

Since the learning rates are non-increasing (`$\eta_t \leq \eta_{t-1}$`), we have `$\frac{1}{\eta_t} - \frac{1}{\eta_{t-1}} \geq 0$`, so the middle sum is non-negative. Using the diameter bound `$(x_t - x^*)^2 \leq \|\mathcal{F}\|^2$` and the gradient bound `$\|g_t\| \leq \|\nabla c\|$`:

$$R_G(T) \leq \|\mathcal{F}\|^2 \left[\frac{1}{2\eta_1} + \frac{1}{2} \sum_{t=2}^T \left(\frac{1}{\eta_t} - \frac{1}{\eta_{t-1}}\right)\right] + \frac{\|\nabla c\|^2}{2} \sum_{t=1}^T \eta_t$$

The bracket telescopes to `$1/(2\eta_T)$`. Finally, with `$\eta_t = 1/\sqrt{t}$`, we compute `$\sum_{t=1}^T 1/\sqrt{t} \leq 2\sqrt{T} - 1$` (via integral approximation), yielding the stated bound.

**Why the proof works (the core insight):** The algorithm's squared distance to any fixed reference point changes by at most the sum of (a) a term proportional to the negative of the per-round regret, and (b) a term proportional to `$\eta_t^2$`. Since the adversary controls the gradient, they could try to push the algorithm away from good regions, but the projection step ensures the algorithm never leaves `$\mathcal{F}$`, and the decreasing learning rate ensures the adversary cannot push it around arbitrarily fast. The `$\sqrt{T}$` bound emerges because the algorithm's total motion is limited: if you sum squared step sizes `$\sum \eta_t^2 \|\nabla c\|^2$`, you get `$\|\nabla c\|^2 \sum 1/t \approx \|\nabla c\|^2 \log T$`, but the bound cleverly converts this to a `$\sqrt{T}$` dependence through the telescoping manipulation.

---

#### The Lazy Projection Variant

Lazy Projection (Algorithm 2) is a structurally different algorithm that the paper analyzes separately because it achieves a better constant in the static regret bound (no `$\sqrt{T} - 1/2$` coefficient on `$\|\nabla c\|^2$`):

**Update rule:**

$$y_{t+1} = y_t - \eta_t \nabla c_t(x_t)$$
$$x_{t+1} = P(y_{t+1})$$

with initial condition `$y_1 = x_1 \in \mathcal{F}$`.

**The critical difference from Greedy Projection:** In Greedy Projection, the gradient step is taken from the previous feasible point `$x_t$`, so `$y_{t+1} = x_t - \eta_t \nabla c_t(x_t)$`. In Lazy Projection, the gradient step is taken from the previous unprojected point `$y_t$`, so `$y_{t+1} = y_t - \eta_t \nabla c_t(x_t)$`. This means `$y_t$` accumulates all past gradients without ever being projected; the projection only happens when producing the actual decision `$x_t$`.

**Two-potential analysis:** The analysis of Lazy Projection decomposes the regret into two competing potentials:

1. **Ideal potential (Lemma 2):** The squared distance from the unprojected point `$y_t$` to the optimal reference `$x^*$`. This potential grows when the algorithm is making progress (moving toward regions that perform well), and the regret is bounded by:

$$\sum_{t=1}^T c_t(y_t) - c_t(x^*) \leq \frac{\|\mathcal{F}\|^2}{2\eta} - \frac{d(y_{T+1}, x^*)^2}{2\eta} + \frac{T\eta \|\nabla c\|^2}{2}$$

where `$d(y_{T+1}, x^*)^2$` is the final squared distance. This is essentially the same analysis as Greedy Projection but applied to the `$y_t$` sequence, which by construction never undergoes projection.

2. **Projection potential (Lemma 3):** The squared distance from `$y_t$` to the feasible set, defined as `$d(y_t, \mathcal{F}) = \min_{x \in \mathcal{F}} d(y_t, x) = d(y_t, P(y_t))$`. This potential measures the "cost" of being far from the feasible set and having to use `$x_t = P(y_t)$` instead of `$y_t$` directly:

$$\sum_{t=1}^T c_t(x_t) - c_t(y_t) \leq \frac{d(y_{T+1}, \mathcal{F})^2}{2\eta}$$

**The cancellation:** When combining the two lemmas, the negative `$-d(y_{T+1}, \mathcal{F})^2/(2\eta)$` from Lemma 2 cancels with the positive `$+d(y_{T+1}, \mathcal{F})^2/(2\eta)$` from Lemma 3, yielding the clean bound of Theorem 3:

$$R_L(T) \leq \frac{\|\mathcal{F}\|^2}{2\eta} + \frac{\eta \|\nabla c\|^2 T}{2}$$

With a fixed learning rate `$\eta$`, this is a `$\|\mathcal{F}\|^2/(2\eta) + \eta \|\nabla c\|^2 T/2$` bound. Optimizing over `$\eta$` yields `$\eta = \|\mathcal{F}\|/(\|\nabla c\|\sqrt{T})$` and a regret of `$\|\mathcal{F}\| \|\nabla c\| \sqrt{T}$`.

**The geometric lemma underlying the projection potential (Lemma 7 and Corollary 1):** The analysis of the projection potential relies on a nontrivial geometric fact about convex projections. For any sequence of points `$y_t$` and their projections `$x_t = P(y_t)$`, if we define `$z_t = d(y_t, \mathcal{F})$`, the distance from `$y_t$` to the feasible set, then:

$$(y_t - P(y_t)) \cdot (y_{t+1} - y_t) \leq z_t(z_{t+1} - z_t) \leq \frac{z_{t+1}^2 - z_t^2}{2}$$

The first inequality (Lemma 7) bounds the dot product between the projection error vector and the step vector by the product of the current distance to `$\mathcal{F}$` and the change in that distance. The second inequality (Lemma 4) is the algebraic fact that `$(a-b)b \leq (a^2 - b^2)/2$` for any real numbers, applied with `$a = z_{t+1}$` and `$b = z_t$`.

Since `$y_{t+1} - y_t = -\eta \nabla c_t(x_t) = -\eta g_t$`, we have `$g_t \cdot (P(y_t) - y_t) = -(y_t - P(y_t)) \cdot (y_{t+1} - y_t)/\eta$`, and the per-round projection cost `$c_t(x_t) - c_t(y_t) = g_t \cdot (x_t - y_t)$` telescopes to `$d(y_{T+1}, \mathcal{F})^2/(2\eta)$`.

**Why the name "Lazy Projection":** The algorithm is "lazy" because it defers projection—it lets `$y_t$` wander freely in `$\mathbb{R}^n$`, accumulating the full effect of all past gradients, and only projects when forced to produce a decision. In contrast, Greedy Projection "proactively" projects after every single gradient step. The lazy approach achieves a tighter static regret bound (the constant factor is better) but may be less responsive to dynamic environments because the accumulated `$y_t$` can drift far from `$\mathcal{F}$`, and the projection `$P(y_t)$` can change abruptly when the accumulated vector crosses certain boundaries.

---

#### Dynamic Regret (Theorem 2)

The static regret analysis compares against a single fixed point `$x^*$` that is optimal in hindsight. However, in non-stationary environments, the best decision might drift over time. The paper formalizes this via **dynamic regret**, where the comparison class `$\mathcal{A}(T, L)$` consists of sequences `$(x_1, \ldots, x_T)$` with total path length at most `$L$`:

$$\sum_{t=1}^{T-1} d(x_t, x_{t+1}) \leq L$$

Dynamic regret is defined as:

$$R_A(T, L) = C_A(T) - \min_{A' \in \mathcal{A}(T, L)} C_{A'}(T)$$

Theorem 2 states that with a **fixed** learning rate `$\eta$` (not the decaying schedule used for static regret), the dynamic regret of Greedy Projection satisfies:

$$R_G(T, L) \leq \frac{7\|\mathcal{F}\|^2}{4\eta} + \frac{L\|\mathcal{F}\|}{\eta} + \frac{T\eta \|\nabla c\|^2}{2}$$

**The proof** (deferred to Appendix A) follows a similar telescoping structure but replaces the fixed `$x^*$` with a sequence `$z_1, \ldots, z_T$` that can change over time. The additional term `$L\|\mathcal{F}\|/\eta$` accounts for the cost of tracking a moving target: each time the optimal point shifts, the algorithm's distance-based potential is reset, and the accumulated resets sum to the total path length `$L$` times the diameter `$\|\mathcal{F}\|$` divided by the step size `$\eta$`.

The key design choice is using a **fixed** learning rate for the dynamic case. With a decaying learning rate, the algorithm becomes increasingly sluggish and cannot track changes. A fixed `$\eta$` keeps the algorithm responsive, but incurs a linear-in-`$T$` penalty `$T\eta \|\nabla c\|^2/2$`. Balancing the tracking term `$L\|\mathcal{F}\|/\eta$` against the responsiveness penalty `$T\eta \|\nabla c\|^2$` yields an optimal fixed `$\eta$` that depends on the path length `$L$`.

---

#### Linearization as the Analytical Bridge

The reduction from arbitrary convex costs to linear costs (Step 1 in the proof of Theorem 1) deserves separate emphasis because it is the analytical engine that makes everything else work. The paper states this reduction explicitly:

> "First, begin with arbitrary `$\{c_1, c_2, \ldots\}$`, run the algorithm and compute `$\{x_1, x_2, \ldots\}$`. Then define `$g_t = \nabla c_t(x_t)$`. If we were to change `$c_t$` such that for all `$x$`, `$c_t(x) = g_t \cdot x$`, the behavior of the algorithm would be the same."

The two key facts are:

1. **Algorithm invariance:** The algorithm only queries the gradient at the point it plays. If two cost functions have the same gradient at `$x_t$`, they produce the same next point `$x_{t+1}$`. The linear function `$g_t \cdot x$` has gradient `$g_t$` everywhere, so it matches the original `$c_t$` at `$x_t$` in both value and gradient.

2. **Regret dominance:** For any comparison point `$x^*$`, convexity gives `$c_t(x^*) \geq c_t(x_t) + g_t \cdot (x^* - x_t)$`, which rearranges to `$c_t(x_t) - c_t(x^*) \leq g_t \cdot x_t - g_t \cdot x^*$`. The actual per-round regret is bounded above by the regret against the linearized function.

Together, these facts mean that any regret bound proven for linear cost functions applies unchanged to arbitrary convex functions. The adversary's power reduces to selecting a sequence of vectors `$g_t$` with bounded norm.

---

#### The Repeated Games Connection: GIGA

The paper's third major algorithm, **Generalized Infinitesimal Gradient Ascent (GIGA)** (Algorithm 3), is Greedy Projection specialized to the setting of repeated games. The feasible set is the `$(|A|-1)$`-dimensional simplex:

$$\mathcal{F} = \left\{x \in \mathbb{R}^{|A|} : x_i \geq 0 \text{ for all } i, \sum_{i=1}^{|A|} x_i = 1\right\}$$

The cost functions are linear and derived from observed utilities. When the opponent plays action `$h_{t,2} \in Y$`, the utility of playing mixed strategy `$x$` is `$\sum_i x_i \cdot u(i, h_{t,2})$`. Converting to a cost (since the algorithm minimizes, not maximizes), the gradient at round `$t$` is the vector of action utilities `$g_t = [-u(1, h_{t,2}), \ldots, -u(|A|, h_{t,2})]$`, and the update becomes:

$$y_{t+1}^i = x_t^i + \eta_t u(i, h_{t,2})$$
$$x_{t+1} = P(y_{t+1})$$

For the simplex, the quantities in the regret bound specialize to `$\|\mathcal{F}\| \leq \sqrt{2}$` (the diameter of the simplex in Euclidean norm) and `$\|\nabla c\| \leq \sqrt{|A|} |u|$`, where:

$$|u| = \max_{(a,y) \in A \times Y} u(a,y) - \min_{(a,y) \in A \times Y} u(a,y)$$

is the range of possible utilities. Theorem 4 then states that for all oblivious deterministic environments, the expected regret satisfies:

$$\mathbb{E}_{h \in \mathcal{F}_{\sigma,\rho}} [R_{* \to a}(h|_T)] \leq \sqrt{T} + \left(\frac{\sqrt{T} - 1}{2}\right) |A| |u|^2$$

**The connection to infinitesimal gradient ascent (Singh, Kearns, and Mansour, 2000):** The original infinitesimal gradient ascent applied only to two-action games. The projection onto the 1-dimensional simplex (a line segment from `$(1,0)$` to `$(0,1)$`) has a simple closed form: clip each coordinate to `$[0,1]$` and renormalize. GIGA extends this to `$|A|$` actions by using the general Euclidean projection onto the `$(|A|-1)$`-dimensional simplex, which can be computed efficiently (for instance, by the algorithm of Duchi et al., 2008, though not cited here as the paper predates that reference). The paper's Theorem 4 thus establishes that GIGA is **exactly** what you get when you run Greedy Projection on the simplex with linear utility functions—it is both a generalization of infinitesimal gradient ascent and a special case of the online convex programming framework.

---

#### Self-Oblivious Behavior and Universal Consistency

The final piece of the technical architecture connects the regret bound against oblivious deterministic environments to **universal consistency** against arbitrary adaptive environments. This connection is not an algorithm but a **proof technique** that leverages the structure of GIGA.

**Self-oblivious behavior (Definition 9):** A behavior `$\sigma$` is self-oblivious if its action distribution at time `$t$` depends only on the past actions of the **environment**, not on its own past actions. Formally, there exists a function `$f : Y^* \to \Delta(A)$` such that for all histories `$h$`, `$\sigma(h) = f(\Pi_2(h))$`, where `$\Pi_2(h)$` extracts only the environment's actions from the history.

GIGA is self-oblivious because its state (the current mixed strategy `$x_t$`) is deterministically updated based on `$x_1$` (a constant) and the observed environment actions `$h_{1,2}, \ldots, h_{t-1,2}$`. The player's own past actions do not feed into the update.

**Why self-obliviousness matters (Lemma 9):** If a behavior is self-oblivious, then for any time `$T$`, the expected regret against an arbitrary adaptive environment `$\rho$` can be bounded by the expected regret against the **worst-case oblivious deterministic environment** that plays the same fixed sequence of actions. The proof constructs this worst-case environment by enumerating all finite histories of length `$T$`, finding the one that maximizes a certain "expected regret" potential `$V_\sigma(h)$`, and having the oblivious adversary play exactly the sequence of environment actions from that history. Because the behavior's response depends only on the environment's actions (self-obliviousness), the distribution over player actions when facing this fixed sequence is identical to the distribution when those same environment actions arose from an adaptive process.

**From expected regret to high-probability regret (Lemma 10):** The actual regret `$R_{* \to a}(h)$` is decomposed via Doob's decomposition into:

$$R_{* \to a}(h) = V_\sigma(h) + V_\sigma^{\text{rem}}(h)$$

where `$V_\sigma(h)$` is the "expected" part (the sum of conditional expectations of per-round regret given the history so far) and `$V_\sigma^{\text{rem}}(h)$` is the "random" part (a martingale difference sequence with bounded increments). If the expected part is bounded by `$T\epsilon$` for all oblivious deterministic environments (which Theorem 4 guarantees for GIGA), then Lemma 9 implies it is bounded by `$T\epsilon$` for all adaptive environments as well. Azuma's inequality (Lemma 8) then bounds the probability that the random part exceeds `$T\epsilon$`:

$$\Pr\left[\sum_{i=1}^T Y_i > T\epsilon\right] \leq \exp\left(-\frac{T\epsilon^2}{8|u|^2}\right)$$

where `$Y_i$` are the martingale differences with `$|Y_i| \leq 2|u|$`.

**Universal consistency (Lemma 1):** Combining these pieces, for any `$\epsilon > 0$`, there exists a time `$T$` such that for all environments `$\rho$` (adaptive or otherwise), the probability that the average regret ever exceeds `$\epsilon$` after time `$T$` is less than `$\epsilon$`. The proof takes a union bound over all `$T > t$` (a geometric series with ratio `$r = \exp(-\epsilon^2/(32|u|^2)) < 1$`) and over all actions `$a \in A$` (multiplying by `$|A|$`). The resulting bound holds simultaneously for all times beyond the threshold, establishing the uniform convergence property that defines universal consistency.

This proof architecture is the paper's final technical contribution: it shows that the abstract online convex programming analysis (Theorem 1) directly implies universal consistency for gradient-based learning in games (GIGA), connecting two previously separate research threads through the concept of self-oblivious behavior.

## 4. Key Insights and Innovations

### Innovation 1: Formalizing "Online Convex Programming" as a Distinct Problem Class That Unifies and Generalizes Prior Settings

The paper's most fundamental contribution is not an algorithm but a **problem definition** — the very act of naming and axiomatizing online convex programming as a distinct object of study. Before this work, three separate research communities studied structurally related problems under different names, with different assumptions, and with no unifying language. The experts community (Littlestone and Warmuth, 1989; Freund and Schapire, 1999) worked on the simplex with linear costs. The online prediction community (Cesa-Bianchi et al., 1994; Kivinen and Warmuth, 1997) studied convex Bregman divergences in regression settings. The repeated games community (Fudenberg and Levine, 1995; Singh et al., 2000) analyzed gradient dynamics on the simplex with linear utilities. Each community developed its own algorithms and its own style of analysis, but no one had articulated the common mathematical structure underlying all three.

The paper's Definition 4 — "a feasible set F ⊆ ℝⁿ and an infinite sequence {c₁, c₂, …} where each cₜ : F → ℝ is a convex function" — is deceptively simple. Its power lies in what it **omits**: no requirement that costs be linear, no requirement that F be a simplex or polytope, no requirement that costs arise from a prediction task or a game. The seven explicit assumptions (boundedness, closedness, nonemptiness, differentiability, gradient bound, gradient oracle, projection oracle) carve out precisely the conditions under which a single algorithm — projected gradient descent — can be analyzed, and no more. This is the kind of definition that seems obvious in retrospect but was not obvious at the time, because it required recognizing that the experts problem, linear optimization on polytopes, and gradient ascent in games are all instances of the same abstract interaction pattern: choose a point from a convex set, incur a convex cost revealed after the choice, repeat.

What makes this a conceptual innovation rather than a mere taxonomy is that the definition **enables new analysis**. The paper doesn't just say "these problems are related"; it proves a regret bound (Theorem 1) that depends on geometric quantities (diameter ∥F∥ and gradient bound ∥∇c∥) rather than combinatorial quantities (number of vertices or experts). This means the bound applies uniformly to curved feasible sets (balls, ellipsoids) that have no finite vertex representation, and to polytopes with exponentially many vertices where experts-based bounds would be vacuous. The bound says something nontrivial about a class of problems that had no prior unified theory.

The significance is that this definition opened up a new research program. By isolating the essential structure — convex set, convex costs, gradient access, projection — the paper made it possible for subsequent work to develop algorithms, prove lower bounds, and study variants (bandit feedback, strongly convex costs, adaptive adversaries) within a single coherent framework. The field of online convex optimization as it exists today traces its lineage directly to this definitional move.

### Innovation 2: The Linearization Principle — Reducing Arbitrary Convex Costs to Linear Costs Without Loss of Generality

The paper's second major conceptual contribution is a **proof technique** that has become so standard in online learning that its origin is easy to overlook: the observation that any adversarial sequence of convex cost functions can be replaced, for analysis purposes, by a sequence of linear functions, with the regret only increasing. This is not a statement about the algorithm's behavior (which depends only on gradients regardless) but about the **adversary's power** — the worst-case regret against convex functions is achieved when the adversary plays linear functions.

Why is this a conceptual innovation rather than a routine technical step? Because it dramatically simplifies the problem while preserving its essential difficulty. Without this reduction, analyzing regret against arbitrary convex functions would require tracking second-order effects: curvature, Hessians, the fact that the cost at the comparison point x* might be nonlinear. The linearization principle says: none of that matters. The adversary's only real weapon is choosing the gradient vector gₜ at the point the algorithm plays; the nonlinear shape of cₜ away from xₜ is irrelevant because convexity guarantees that cₜ(x*) ≥ cₜ(xₜ) + gₜ · (x* − xₜ), so the linear lower bound is all that constrains the adversary's ability to make x* look good in hindsight.

This is a subtle but profound shift in how to think about adversarial online learning. It says that the "convex" in online convex programming is not what makes the problem hard — linear costs are already as hard as anything. The convexity assumption is a **restriction on the adversary** (preventing pathological non-convex landscapes where gradient information is misleading), not a source of additional difficulty for the algorithm.

The paper makes this point explicitly in Section 4.2 when discussing how to convert an online linear programming algorithm into an online convex programming algorithm: "we find that the worst case is when the cost function is linear. This assumption depends on two properties of the algorithm; the algorithm is deterministic, and the only property of the cost function cₜ that is observed is ∇cₜ(xₜ)." The insight is that any deterministic algorithm that only queries gradients is, from the adversary's perspective, playing against linear costs — so the analysis might as well assume linearity from the start.

This idea has had enormous downstream influence. It is the reason that nearly all subsequent work in online convex optimization analyzes regret against linear functions and then invokes convexity to extend the bound. It is one of those rare proof techniques that simultaneously simplifies the analysis and clarifies the conceptual structure of the problem.

### Innovation 3: Projection as an Analytical Tool — The Inequality (P(y) − x)² ≤ (y − x)² as the Keystone of Regret Analysis

The paper identifies and exploits a geometric fact about Euclidean projection onto convex sets that is simple to state but whose consequences for online learning are far-reaching: for any point y ∈ ℝⁿ, its projection P(y) onto a closed convex set F, and any x ∈ F, the squared Euclidean distance satisfies (P(y) − x)² ≤ (y − x)². In words: projection never pushes you farther from any feasible target than you already were.

This inequality (attributed to Gentile and Warmuth, 2000) is the **single structural property** that makes the entire regret analysis of Greedy Projection go through. Without it, the gradient step yₜ₊₁ = xₜ − ηₜgₜ could easily leave the feasible set, and the analysis would have to account for the difference between the point the algorithm wanted to play (yₜ₊₁) and the point it actually played (xₜ₊₁ = P(yₜ₊₁)). With it, the projection step can only help: it reduces or maintains the squared distance to any comparison point x* while simultaneously ensuring feasibility.

What makes this an innovation rather than a mere citation of a known fact is the paper's recognition of its **architectural role** in regret analysis. The Greedy Projection proof (Theorem 1) uses this inequality to convert the distance-tracking sum from a form involving the unprojected points yₜ₊₁ (which have a clean recurrence) to a form involving the projected points xₜ₊₁ (which are the actual decisions). The inequality acts as a bridge between the "analysis-friendly" unprojected sequence and the "reality-bound" projected sequence.

The Lazy Projection analysis (Theorem 3) reveals the deeper structure: the inequality's role is to create a **cancellation** between two competing potentials. The ideal potential (distance from unprojected point yₜ to x*) benefits from the projection inequality in one direction (the term −d(yₜ₊₁, x*)²/(2η) in Lemma 2), while the projection potential (distance from yₜ to the feasible set F) benefits in the opposite direction (the term +d(yₜ₊₁, F)²/(2η) in Lemma 3). The two terms cancel exactly when combined, leaving only the diameter of F and the gradient bound. This cancellation is not an accident; it is the mathematical expression of the fact that Lazy Projection's two sources of error — being far from the optimum and being far from the feasible set — offset each other in a way that projection algorithms can exploit.

The paper's geometric lemmas (Lemmas 5-7 and Corollary 1 in Appendix B) show that this cancellation is rooted in a deeper fact about the geometry of convex sets: the dot product between the projection error vector (y − P(y)) and the step vector (y' − y) is bounded by the product of distances to the set, which telescopes. The paper proves these lemmas from first principles using only the definition of convexity and the properties of Euclidean distance, making the entire argument self-contained.

### Innovation 4: Self-Obliviousness as the Bridge from Oblivious Adversaries to Universal Consistency

The paper's final conceptual contribution is the identification of **self-obliviousness** — the property that a behavior's decisions depend only on the opponent's past actions, not on its own — as the structural condition that allows regret bounds against oblivious deterministic environments to imply universal consistency against arbitrary adaptive environments.

This is a subtle and non-obvious connection. In repeated games, an adaptive adversary can condition its play on the entire history of joint actions, including the player's own past moves. This creates the possibility of a vicious circle: the adversary learns the player's strategy, exploits it, the player adapts, the adversary re-adapts, and so on. Proving that a learning algorithm converges to no-regret behavior against such an adversary is substantially harder than proving convergence against an environment that plays a fixed (but unknown) sequence of actions regardless of what the player does.

The paper's key insight (Lemma 9) is that if the player's behavior is self-oblivious — if it ignores its own past actions when deciding what to do next — then the adversary's adaptivity is **powerless to affect the player's action distribution** at any given round, once we condition on the adversary's past actions. The player's response to the sequence (y₁, y₂, …, yₜ₋₁) of environment actions is the same whether those actions were generated by an oblivious process that decided them in advance or by an adaptive process that reacted to the player's moves. This means that for any adaptive environment, there exists an oblivious deterministic environment that produces the same distribution over player actions up to any finite time T — namely, the environment that simply plays the sequence of actions that maximizes the player's expected regret.

This insight is not algorithmic but **structural**: it identifies a property of the learning algorithm (self-obliviousness) that decouples the adversarial adaptivity problem from the regret analysis. Once you prove that (a) your algorithm is self-oblivious and (b) it achieves low expected regret against all oblivious deterministic environments, you get universal consistency against all environments "for free" via Azuma's inequality and a union bound over time. The adaptivity of the adversary is handled entirely by the probabilistic tail bound; the expected-regret analysis only needs to consider the simpler oblivious case.

GIGA happens to be self-oblivious because its state (the current mixed strategy xₜ) is updated using only the environment's observed actions, not the player's own realized actions. But the conceptual contribution is broader: the paper provides a **template** for proving universal consistency of any self-oblivious algorithm, and explicitly contrasts this with algorithms that are not self-oblivious (citing Kalai and Vempala's algorithm, which uses a random seed that an adaptive adversary could learn). This distinction between self-oblivious and non-self-oblivious algorithms is a diagnostic concept that clarifies why some learning algorithms are robust to adaptivity and others are not — it is a property of the algorithm's information dependence, not of its regret bound per se.

The significance of this insight is that it opened a path for proving strong guarantees for gradient-based learning in games without requiring game-specific analysis. The proof that GIGA is universally consistent (Lemma 1, Theorem 4) follows directly from plugging the online convex programming regret bound into the self-obliviousness template. This is a fundamentally different approach from the game-theoretic analyses that preceded it (Fudenberg and Levine's proofs for fictitious play and related dynamics, or the specific two-action analysis of Singh et al.), and it demonstrates the power of the online convex programming abstraction: game-theoretic guarantees emerge as corollaries of a more general optimization theory.

## 5. Experimental Analysis

### Evaluation Methodology

**Important caveat:** This paper is a **theoretical work** — a technical report from February 2003 — and does not contain empirical experiments in the modern machine learning sense. There are no datasets, no training runs, no ablation studies over hyperparameters, and no tables of accuracy numbers comparing against baselines. The paper's contribution is a mathematical framework and a set of regret bounds proved analytically. The "results" are theorems, not experimental measurements.

However, the paper does contain a specific kind of quantitative evaluation that can be analyzed as one would analyze experimental results: the **numerical regret bounds** themselves, the comparison of these bounds across algorithm variants (Greedy Projection vs. Lazy Projection), the comparison across settings (static regret vs. dynamic regret), and the reduction of these bounds to concrete numbers in the game-theoretic application (GIGA). The following analysis treats these analytical results as the paper's "experimental" content.

- **Dataset.** Not applicable. The paper's results are analytical and apply to any sequence of convex cost functions satisfying the assumptions (bounded gradients, differentiable, etc.). There is no empirical dataset.

- **Base model(s).** Not applicable in the machine learning sense. The "model" is the Greedy Projection algorithm (Algorithm 1) with learning rate schedule $\eta_t = t^{-1/2}$, and its two variants: Lazy Projection (Algorithm 2) with fixed learning rate $\eta$, and Generalized Infinitesimal Gradient Ascent (Algorithm 3) specialized to the simplex.

- **Metrics.** The primary metric throughout is **regret** $R(T)$, defined as the cumulative cost incurred by the algorithm minus the cumulative cost of the best fixed feasible point in hindsight:
  $$R_A(T) = \sum_{t=1}^T c_t(x_t) - \min_{x \in \mathcal{F}} \sum_{t=1}^T c_t(x)$$
  For the dynamic regret analysis (Theorem 2), the metric is **dynamic regret** $R_A(T, L)$, where the comparison class is sequences of points with total path length at most $L$. For the game-theoretic application, the metric is **expected regret against the best fixed action**: $\mathbb{E}_{h \in \mathcal{F}_{\sigma,\rho}}[R_{* \to a}(h|_T)]$, and ultimately **universal consistency** (probability that average regret ever exceeds $\epsilon$ after time $T$ is less than $\epsilon$).

- **Baselines.** The paper does not implement empirical baselines because there are no experiments. However, it positions its bounds against two conceptual baselines:
  - **Experts algorithms** (Littlestone and Warmuth, 1989; Freund and Schapire, 1999): The paper argues (Section 4) that experts algorithms applied to online linear programming would produce bounds that depend on the number of vertices of the polytope, which can be exponentially large and "totally unrelated to the diameter," making them "incomparable" to the Greedy Projection bound. No specific expert algorithm's bound is computed for direct comparison.
  - **Kalai and Vempala (2002):** An alternative algorithm for online linear programming that is described as "lazy" (changing its vector slowly), contrasted with the paper's more "dynamic" approach. No numerical comparison of bounds is provided.

- **Generation budget / compute accounting.** All bounds are expressed as functions of the number of rounds $T$ (the horizon). The "cost" of the algorithm per round is one gradient evaluation and one projection. The analysis does not count FLOPs or wall-clock time; it counts iterations and expresses regret as a function of $T$, the diameter $\|\mathcal{F}\|$, and the gradient bound $\|\nabla c\|$. The learning rate schedule $\eta_t = t^{-1/2}$ is the "compute allocation" strategy.

- **Cross-validation / statistical protocol.** Not applicable. The results are worst-case bounds that hold for all sequences of cost functions satisfying the assumptions, with no probabilistic qualification beyond what is introduced in the game-theoretic setting (where Azuma's inequality is applied to the martingale difference sequence to get high-probability bounds). There is no train/test split, no hyperparameter tuning on held-out data, and no averaging over random seeds. The analysis is purely deductive from the stated assumptions.

---

### Main Quantitative Results

The paper's quantitative contributions are organized around three algorithm variants and two regret notions. I'll present each as a separate analytical "result group."

#### Static Regret of Greedy Projection (Theorem 1)

**Headline bound:** With learning rate $\eta_t = t^{-1/2}$, the cumulative regret after $T$ rounds satisfies:

$$R_G(T) \leq \|\mathcal{F}\|^2 \frac{\sqrt{T}}{2} + \left(\sqrt{T} - \frac{1}{2}\right) \|\nabla c\|^2$$

**Interpretation as a scaling law:** Both terms scale as $\Theta(\sqrt{T})$, meaning the average regret $R_G(T)/T$ decays as $\Theta(1/\sqrt{T})$ and approaches zero as $T \to \infty$. The first term $\|\mathcal{F}\|^2 \sqrt{T}/2$ is the "initialization penalty": if the algorithm starts at the opposite side of the feasible set from the optimal fixed point, it incurs regret proportional to the squared diameter. The second term $(\sqrt{T} - 1/2) \|\nabla c\|^2$ is the "reactivity penalty": the algorithm always responds after seeing the cost function, so it incurs regret proportional to the squared maximum gradient norm.

**Key constants:** The bound does not depend on the dimension $n$ of the ambient space, only on the geometry of the feasible set (through its diameter $\|\mathcal{F}\|$) and the smoothness of the cost functions (through $\|\nabla c\|$). This is the central selling point over experts-based approaches, which would scale with the number of vertices or the dimension.

**The bound in the proof's telescoping form (before plugging in $\eta_t$):**

$$R_G(T) \leq \|\mathcal{F}\|^2 \frac{1}{2\eta_T} + \frac{\|\nabla c\|^2}{2} \sum_{t=1}^T \eta_t$$

This form reveals the learning rate tradeoff explicitly: a smaller $\eta_T$ (faster decay) reduces the initialization penalty but increases the cumulative $\sum \eta_t$ if the early rates are large. The $t^{-1/2}$ schedule balances these to achieve the $\sqrt{T}$ rate. The paper does not claim optimality of this schedule; it is the specific choice that yields the clean bound.

**Comparison to Lazy Projection (Theorem 3):** With a **fixed** learning rate $\eta$, Lazy Projection achieves:

$$R_L(T) \leq \frac{\|\mathcal{F}\|^2}{2\eta} + \frac{\eta \|\nabla c\|^2 T}{2}$$

Optimizing over $\eta$ gives $\eta = \|\mathcal{F}\|/(\|\nabla c\|\sqrt{T})$, yielding regret $\|\mathcal{F}\| \|\nabla c\| \sqrt{T}$. The Greedy Projection bound (with its $t^{-1/2}$ schedule) has a $\|\nabla c\|^2 \sqrt{T}$ dependence rather than $\|\mathcal{F}\| \|\nabla c\| \sqrt{T}$. These are incomparable: when $\|\mathcal{F}\|$ is large and $\|\nabla c\|$ is small, Lazy Projection's bound is better; when $\|\mathcal{F}\|$ is small and $\|\nabla c\|$ is large, Greedy Projection's constant is better. Neither dominates the other in all regimes.

**What is NOT provided:** The paper does not give a lower bound showing that $\Omega(\sqrt{T})$ regret is unavoidable, nor does it compare its $\sqrt{T}$ rate to the optimal achievable rate for online convex programming. (Subsequent literature established $\Theta(\sqrt{T})$ as the minimax optimal rate for this setting.)

---

#### Dynamic Regret of Greedy Projection (Theorem 2)

**Headline bound:** With a **fixed** learning rate $\eta$, the dynamic regret against a sequence of comparison points with total path length $L$ satisfies:

$$R_G(T, L) \leq \frac{7\|\mathcal{F}\|^2}{4\eta} + \frac{L\|\mathcal{F}\|}{\eta} + \frac{T\eta \|\nabla c\|^2}{2}$$

**Interpretation:** The three terms capture distinct sources of dynamic regret. The first term $7\|\mathcal{F}\|^2/(4\eta)$ is the initialization cost (a constant independent of $T$ and $L$). The second term $L\|\mathcal{F}\|/\eta$ is the **tracking cost**: each unit of path length the optimal sequence moves requires the algorithm to "pay" $\|\mathcal{F}\|/\eta$ in additional regret to catch up. The third term $T\eta \|\nabla c\|^2/2$ is the **reactivity penalty** from the static case, now linear in $T$ because the learning rate is fixed.

**The $T$ and $L$ tradeoff:** Notice that the tracking cost scales as $L/\eta$ while the reactivity penalty scales as $T\eta$. If $L$ is small (the optimal sequence is nearly static), a small $\eta$ optimizes the bound, and the dynamic regret approaches the static regret. If $L$ is large (the environment is highly non-stationary), a larger $\eta$ is needed to keep the tracking cost manageable, at the expense of a larger reactivity penalty. The paper does not optimize $\eta$ over $L$ explicitly, but the structure of the bound implies the optimal fixed $\eta \propto \sqrt{(7\|\mathcal{F}\|^2/2 + L\|\mathcal{F}\|)/(T\|\nabla c\|^2)}$.

**Key difference from static case:** The fixed learning rate is essential for dynamic tracking. With the decaying schedule $\eta_t = t^{-1/2}$, the algorithm becomes increasingly sluggish — after many rounds, $\eta_t \approx 0$ and the algorithm essentially stops moving, making it incapable of tracking a moving optimum. The paper's explicit contrast between its "dynamic" design philosophy and Kalai and Vempala's "lazy" approach (Section 5) is embodied in this choice.

**What is NOT provided:** The dynamic regret bound is proved only for Greedy Projection with a fixed learning rate, not for Lazy Projection. The paper does not discuss whether Lazy Projection can achieve dynamic regret bounds, what learning rate schedule would be optimal for the dynamic case, or whether any algorithm can achieve sublinear dynamic regret when $L$ grows with $T$ (e.g., $L = \Theta(T)$). These are left to future work.

---

#### GIGA and Universal Consistency in Repeated Games (Theorem 4 and Lemma 1)

**Headline bound for GIGA (Theorem 4):** With learning rate $\eta_t = t^{-1/2}$, for any oblivious deterministic environment, the expected regret of GIGA against any fixed action $a \in A$ satisfies:

$$\mathbb{E}_{h \in \mathcal{F}_{\sigma,\rho}}[R_{* \to a}(h|_T)] \leq \sqrt{T} + \left(\frac{\sqrt{T} - 1}{2}\right) |A| |u|^2$$

where $|u| = \max_{(a,y) \in A \times Y} u(a,y) - \min_{(a,y) \in A \times Y} u(a,y)$ is the utility range.

**Plugging in the constants:** The bound follows from Theorem 1 by substituting the specific values for the simplex: $\|\mathcal{F}\| \leq \sqrt{2}$ (the diameter of the probability simplex in Euclidean norm), and $\|\nabla c\| \leq \sqrt{|A|} |u|$ (the gradient of the linear utility function has components bounded by $|u|$, and there are $|A|$ of them, so the Euclidean norm is at most $\sqrt{|A|} |u|$). The paper computes:

- $\|\mathcal{F}\|^2 \leq 2$, so the first term becomes $2 \cdot \sqrt{T}/2 = \sqrt{T}$.
- $\|\nabla c\|^2 \leq |A| |u|^2$, so the second term becomes $(\sqrt{T} - 1/2) |A| |u|^2$.

**The dependence on $|A|$:** The regret scales linearly with the number of actions $|A|$. This is worse than the logarithmic dependence $\ln |A|$ achieved by the multiplicative weights / Hedge algorithm (Freund and Schapire, 1999) for the experts problem on the simplex. However, the paper's contribution is not to beat Hedge at its own game — it is to establish that gradient ascent, a fundamentally different update rule, also achieves no-regret and carries the additional property of universal consistency. The paper explicitly acknowledges that experts algorithms may have better dependence on $|A|$ in some cases (Section 4).

**Universal consistency (Lemma 1 and Lemma 11):** For any $\epsilon > 0$, there exists a time $t$ such that for all environments $\rho$ (adaptive or otherwise) and all times $T > t$:

$$\Pr_{h \in \mathcal{F}_{\sigma,\rho}}[R(h|_T) > 2T\epsilon] < |A| \exp\left(-\frac{T\epsilon^2}{8|u|^2}\right)$$

and consequently, for all sufficiently large $t$:

$$\Pr_{h \in \mathcal{F}_{\sigma,\rho}}[\exists T > t, R(h|_T) > T\epsilon] < \epsilon$$

**The convergence rate:** The probability that the average regret exceeds $2\epsilon$ at time $T$ decays exponentially in $T$ (specifically, as $\exp(-T\epsilon^2/(8|u|^2))$). The threshold time $t$ after which the regret never again exceeds $\epsilon$ with probability $1 - \epsilon$ is:

$$t \approx \frac{32|u|^2}{\epsilon^2} \ln\left(\frac{|A|}{(1 - r^{-1})\epsilon}\right)$$

where $r = \exp(-\epsilon^2/(32|u|^2))$. This threshold depends logarithmically on $|A|$ and quadratically on $1/\epsilon$ and $|u|$. The paper does not compute this threshold explicitly in the main text; the explicit form appears in the proof of Lemma 1 (Appendix C).

**Comparison to two-action infinitesimal gradient ascent:** Singh, Kearns, and Mansour (2000) studied two-action games (where $|A| = 2$). GIGA extends the algorithm to arbitrary $|A|$ and proves universal consistency — a property that Singh et al. did not establish even for the two-action case. The bound in Theorem 4 shows that the regret scales polynomially in $|A|$ (specifically linearly), which is a concrete quantification of the cost of having more actions. The exponential convergence rate in $T$ (from Azuma's inequality) means that, although the bound's constant depends on $|A|$, the algorithm still converges rapidly in practice for moderate numbers of actions.

**What is NOT provided:** The paper does not provide a lower bound showing that the $|A|$ dependence in the regret bound is necessary for gradient-based methods, nor does it compare the convergence rate to other universally consistent algorithms from the game theory literature (Fudenberg and Levine's conditional universal consistency results, or Hart and Mas-Colell's regret-matching). The relationship between the $|A| |u|^2$ constant in the regret bound and the $|u|^2$ constant in the exponential tail bound is stated but not optimized.

---

#### Reduction from Convex to Linear: Exact vs. Approximate Conversion (Section 4.2)

The paper analyzes the cost of converting an online linear programming algorithm (OLPA) to an online convex programming algorithm using sampling. The **Exact** algorithm (Algorithm 6) plays the expectation $\mathbb{E}_{X \sim D_t}[X]$ of the OLPA's distribution. The **Approx** algorithm (Algorithm 7) plays the empirical average of $s_t$ samples from $D_t$.

**Headline bound:** If $s_t = t$ samples are used at round $t$, the expected regret of Approx exceeds the expected regret of Exact by at most:

$$\mathbb{E}[R_{\text{Approx}}(T)] \leq R_{\text{Exact}}(T) + \|\nabla c\| \|\mathcal{F}\| (2\sqrt{T} - 1)$$

**Interpretation:** The sampling error introduces an additional $\sqrt{T}$ term in the regret, with constant $\|\nabla c\| \|\mathcal{F}\|$. The $2\sqrt{T} - 1$ factor comes from $\sum_{t=1}^T 1/\sqrt{s_t} = \sum_{t=1}^T 1/\sqrt{t} \leq 2\sqrt{T} - 1$. This means that the asymptotic rate remains $\Theta(\sqrt{T})$ — the sampling overhead does not change the regret's scaling with $T$, only the constant. To make the sampling overhead negligible compared to the base regret (which is also $\Theta(\sqrt{T})$), one would need $s_t$ to grow faster than $t$ (e.g., $s_t = t^2$) to make $\sum 1/\sqrt{s_t}$ sublinear in $\sqrt{T}$.

**The "super regret" construction:** The proof introduces an intermediate game where the adversary knows both the empirical average $z_t$ and the true expectation $x_t$, and can choose separate gradient vectors $g_t$ (sent to the OLPA) and $h_t$ (used to compute the actual cost difference between $z_t$ and $x_t$). The adversary's optimal choice for $h_t$ is a vector of length $\|\nabla c\|$ in the direction of $z_t - x_t$, which makes $h_t \cdot (z_t - x_t) = \|\nabla c\| d(z_t, x_t)$. This construction decomposes the Approx regret into the Exact regret plus a sampling error term that depends only on the expected distance $\mathbb{E}[d(z_t, x_t)]$, which is bounded by $\|\mathcal{F}\|/\sqrt{s_t}$ via a variance argument.

**What is NOT provided:** The bound on $\mathbb{E}[d(z_t, x_t)]$ uses a coarse inequality: $\mathbb{E}[d(0, z_t)] \leq \|\mathcal{F}\|/\sqrt{s_t}$. This is tight only for distributions with maximum variance (e.g., all mass on opposite vertices of the feasible set). For distributions concentrated near their mean, the distance would be much smaller. The paper does not provide a data-dependent bound or an adaptive sampling scheme that adjusts $s_t$ based on the observed variance.

---

### Ablation Studies and Robustness Checks

Since this is a theoretical paper, "ablations" correspond to variations of the assumptions, algorithm design choices, and proof techniques. I'll structure these as conceptual ablations.

**Learning rate schedule ablation: decaying vs. fixed:** The paper analyzes two regimes. For **static regret** (Theorem 1), the decaying schedule $\eta_t = t^{-1/2}$ is used, and the analysis in the proof of Theorem 1 works for any non-increasing sequence $\eta_t$, with the final bound depending on $1/\eta_T$ and $\sum \eta_t$. The $t^{-1/2}$ choice is then plugged in as the specific schedule that balances these terms. For **dynamic regret** (Theorem 2), a **fixed** $\eta$ is used instead. The paper does not explicitly compare the performance of the decaying vs. fixed schedule for intermediate cases (e.g., $L = \Theta(T^{1/2})$), nor does it explore whether an adaptive schedule (e.g., resetting $\eta_t$ when a change in the environment is detected) would improve dynamic regret. The choice of schedule is presented as a binary switch between the two theorems rather than a tunable tradeoff.

**Projection variant ablation: Greedy vs. Lazy:** The paper compares two projection strategies. **Greedy Projection** (project after every gradient step) achieves a regret bound with $\|\nabla c\|^2 \sqrt{T}$ dependence and $-\frac{1}{2}\|\nabla c\|^2$ constant adjustment. **Lazy Projection** (accumulate all gradients, project only when producing the decision) achieves $\|\mathcal{F}\| \|\nabla c\| \sqrt{T}$ with no $-\frac{1}{2}$ adjustment. These bounds are presented as incomparable — one is better when $\|\nabla c\|$ is small relative to $\|\mathcal{F}\|$, the other when $\|\nabla c\|$ is large. However, the paper does not provide an analysis of **intermediate** strategies (e.g., project every $k$ steps) or prove that these two extremes bracket the achievable performance.

**Cost function linearization ablation:** The proof of Theorem 1 introduces a critical reduction: the analysis assumes linear cost functions $g_t \cdot x$ without loss of generality. The paper verifies that this does not loosen the bound by showing that for any comparison point $x^*$, convexity gives $c_t(x_t) - c_t(x^*) \leq g_t \cdot x_t - g_t \cdot x^*$. This is a one-sided bound: the actual regret is never larger than the linearized regret. The implication is that any worst-case regret bound for linear functions applies unchanged to convex functions. The paper does **not** explore whether there exist convex functions for which the adversary could achieve **higher** regret than the linear worst-case — the linearization principle shows that the answer is no, because the adversary can always choose to play linear functions and the algorithm would not know the difference.

**Gradient oracle abstraction:** Assumption 6 requires that the algorithm can compute $\nabla c_t(x_t)$ given $x_t$. The paper briefly notes (footnote 1 on page 2) that differentiability can be relaxed: "the algorithm can also work if there exists an algorithm that, given $x$, can produce a vector $g$ such that for all $y$, $g \cdot (y - x) \leq c_t(y) - c_t(x)$." This $g$ is a **subgradient**, and the paper's entire analysis extends to convex but non-differentiable functions using subgradients instead of gradients, because only the linear lower bound property is used in the proof. This is a significant robustness check — the algorithm does not actually need differentiability, only subgradient access. The paper does not rename itself as "subgradient descent" but the mathematical content supports it.

**Projection oracle cost assumption:** Assumption 7 requires exact Euclidean projection onto $\mathcal{F}$. The paper does not ablate this assumption — it does not discuss what happens if projection is approximate (e.g., solved via iterative optimization with finite precision) or if a different metric (non-Euclidean) is used for projection. The Lazy Projection analysis (Appendix B) relies heavily on Lemma 5 (the obtuse angle property of Euclidean projection) and Lemma 6-7 (the dot-product-to-distance relationship), both of which are specific to Euclidean geometry. The paper explicitly flags this as a limitation for future work in Section 6: "here we deal with a Euclidean geometry: what if one considered gradient descent on a noneuclidean geometry, like [1, 24]?"

**Self-obliviousness as the adaptivity bridge (Appendix C):** The proof of universal consistency hinges on self-obliviousness. The paper does not ablate this assumption by analyzing whether GIGA would remain universally consistent without it (the answer is implicit: the proof would fail because Lemma 9 requires self-obliviousness to construct the worst-case oblivious environment). The paper does provide a **negative example** of a non-self-oblivious algorithm: Kalai and Vempala (2002) use a random seed, and "an adaptive adversary could learn over time and then use in some settings." This is an implicit ablation: the property matters, and not all no-regret algorithms possess it.

**Martingale decomposition in universal consistency:** The decomposition $R_{* \to a}(h) = V_\sigma(h) + V_\sigma^{\text{rem}}(h)$ into expected and random components (Doob's decomposition) is applied with Azuma's inequality to get the exponential tail bound. The paper does not explore whether alternative concentration inequalities (e.g., Bernstein's inequality for martingales with variance bounds) would yield tighter constants, nor whether the $2|u|$ bound on the martingale differences $|Y_i|$ is tight for GIGA specifically (as opposed to being a worst-case bound over all possible histories).

---

### Critical Assessment

This section evaluates whether the paper's analytical results genuinely support its stated claims, identifies gaps and limitations, and discusses what additional analysis would have strengthened the contribution.

#### Claim 1: Greedy Projection Achieves $O(\sqrt{T})$ Average Regret

**What the paper demonstrates:** Theorem 1 proves that with learning rate $\eta_t = t^{-1/2}$, the cumulative regret is bounded by $A\sqrt{T} + B$ for constants $A$ and $B$ depending on $\|\mathcal{F}\|$ and $\|\nabla c\|$. This directly implies $\limsup_{T \to \infty} R_G(T)/T \leq 0$.

**What the paper does NOT demonstrate:** The bound does not establish whether $\sqrt{T}$ is the **optimal** rate. Is there any algorithm that achieves $o(\sqrt{T})$ regret for online convex programming under the same assumptions? If not, is there a matching lower bound showing that $\Omega(\sqrt{T})$ regret is unavoidable? The paper does not address this question at all. (Subsequent work by Abernethy et al., 2008, and others established that $\Theta(\sqrt{T})$ is indeed minimax optimal for general convex functions with bounded gradients, so the paper's bound is order-optimal, but this is not established in the paper itself.)

The constants in the bound — specifically the $\|\nabla c\|^2$ dependence in the second term — are not necessarily tight. The Lazy Projection bound achieves $\|\mathcal{F}\| \|\nabla c\| \sqrt{T}$ (product of norms rather than squared norm of gradient), which can be substantially smaller. The paper does not claim optimality of constants, but the reader should understand that the specific constants in Theorem 1 are an artifact of the proof technique (and the specific $t^{-1/2}$ schedule), not a fundamental limit of the algorithm.

#### Claim 2: The Algorithm Generalizes Prior Settings (Experts, Repeated Games) to Arbitrary Convex Sets and Arbitrary Convex Functions

**What the paper demonstrates:** The connection to repeated games is made explicit through GIGA (Section 3.3), where the simplex is a special case of a convex set and linear utility functions are a special case of convex functions. Theorem 4 follows by plugging the simplex-specific constants into the general bound. The connection to the experts problem is discussed in Section 4.1, where the polytope of distributions is a convex set and linear costs are convex.

**What the paper does NOT demonstrate (but claims):** The paper asserts (Section 4.1) that the bounds from experts algorithms "depend on the number of experts" and are "incomparable" to the Greedy Projection bounds, which depend on the diameter. This claim is **not substantiated with a concrete comparison**. The paper does not instantiate a specific experts algorithm (e.g., Hedge / multiplicative weights), compute its regret bound on a specific polytope with exponentially many vertices, and show that the gradient-based bound is smaller. The claim about incomparability is qualitative and rests on the observation that the number of vertices can be "totally unrelated to the diameter" — which is true but leaves open the question of whether there exist polytopes where the diameter-based bound **is** smaller, and whether those polytopes arise in practice.

A concrete example would have strengthened this claim. For instance: consider the $n$-dimensional $\ell_1$ ball $\{x : \|x\|_1 \leq 1\}$. This has $2n$ vertices but diameter $\sqrt{2}$ (far less than $n$). An experts-based approach enumerating vertices would have regret scaling with $\ln(2n)$, while the gradient-based bound would scale with $\sqrt{2} \cdot \|\nabla c\| \sqrt{T}$, independent of $n$. Such an example would make the comparison concrete, but the paper does not provide one.

#### Claim 3: GIGA Is Universally Consistent

**What the paper demonstrates:** The chain of reasoning from Theorem 4 (regret bound against oblivious deterministic environments) through Lemmas 9, 10, 11, and 1 (extension to arbitrary adaptive environments via self-obliviousness and Azuma's inequality) proves that for any $\epsilon > 0$, there exists a $t$ such that $\Pr[\exists T > t, R(h|_T) > T\epsilon] < \epsilon$ for all environments $\rho$.

**Assessment:** This proof is **complete and rigorous**. It establishes exactly the definition of universal consistency as formalized by Fudenberg and Levine (1995, 1998). The proof technique (self-obliviousness + exponential tail bound via Azuma) is elegant and general. The result resolves the open question of whether gradient-based learning in multi-action games achieves no-regret behavior against arbitrary adversaries.

**What the paper does NOT demonstrate:** The bound's dependence on $|A|$ (the number of actions) is linear through the $|A| |u|^2$ term in Theorem 4. This means that while GIGA is universally consistent, its convergence rate deteriorates as the action space grows. The paper does not discuss whether this linear dependence is inherent to gradient-based methods (could a different learning rate schedule improve it?) or whether it can be reduced. The experts literature had already achieved $O(\sqrt{T \ln |A|})$ regret via multiplicative weights, which is exponentially better in $|A|$. The paper's contribution is not to compete with that bound, but to establish universal consistency for gradient ascent specifically. However, the practical implication is that GIGA would be a poor choice for games with very large action spaces, which the paper does not explicitly state.

#### Claim 4: Dynamic Regret Can Be Bounded When the Comparison Sequence Has Limited Path Length

**What the paper demonstrates:** Theorem 2 provides a bound that separates into an initialization cost, a tracking cost scaling with $L/\eta$, and a reactivity cost scaling with $T\eta$. For fixed $L$, the dynamic regret is $O(\sqrt{T})$ with an appropriately chosen $\eta \propto 1/\sqrt{T}$, recovering the static rate. When $L$ grows with $T$, the bound degrades gracefully.

**Assessment:** This is a strong and forward-looking result, but it is **incomplete** in several ways. First, the theorem assumes a **fixed** learning rate $\eta$ and does not explore whether a time-varying schedule could improve the bound (analogous to the $t^{-1/2}$ schedule for static regret). Second, the bound is proved only for Greedy Projection, not for Lazy Projection — it is unclear whether Lazy Projection, with its tighter static regret constant, also enjoys dynamic regret guarantees. Third, the proof (Appendix A) is presented as a sketch and contains less detail than the static regret proof. A reader attempting to verify the dynamic regret bound must fill in several telescoping-sum steps themselves, and the derivation of the $7/4$ constant is not fully explained in the text.

#### Structural Weaknesses in the Theoretical Evaluation

**No lower bounds:** The paper does not provide any lower bounds on regret — no demonstration that $\Omega(\sqrt{T})$ regret is unavoidable, no lower bound showing that the dependence on $\|\mathcal{F}\|$ or $\|\nabla c\|$ is tight, and no lower bound for the dynamic regret setting. This is not necessarily a flaw for a paper that is defining a new problem and providing the first algorithm, but it does mean that the reader cannot assess the optimality of the proposed method without consulting subsequent literature.

**No explicit comparisons to alternative algorithms:** The paper mentions experts algorithms (Freund and Schapire, 1999) and Kalai and Vempala (2002) as related work, but never instantiates their bounds on a common problem to provide a head-to-head comparison. The claim that the bounds are "incomparable" is true in the mathematical sense (neither bound uniformly dominates the other), but it would be more informative to characterize **when** each approach is preferable. A simple example problem with a polytope having many vertices but small diameter could have made this vivid.

**The constant in Theorem 1 is not optimized:** The constraint $\|\nabla c\|^2$ appears multiplied by $\sqrt{T} - 1/2$, but the $-1/2$ likely arises from a specific integral approximation ($\sum_{t=1}^T 1/\sqrt{t} \leq 2\sqrt{T} - 1$) rather than being fundamental. A sharper integral bound ($\sum_{t=1}^T 1/\sqrt{t} \leq 2\sqrt{T} - 1 + 1/\sqrt{T}$ or similar) would change the constant slightly. The paper does not discuss constant-factor optimality.

**The sampling-to-expectation gap in Section 4.2 is analyzed with a crude variance bound:** The bound $\mathbb{E}[d(z_t, x_t)] \leq \|\mathcal{F}\|/\sqrt{s_t}$ uses only the fact that the distribution is supported on the feasible set of diameter $\|\mathcal{F}\|$. For distributions concentrated near their mean (as would often occur in practice), the actual expected distance could be much smaller. The paper does not provide a data-dependent or variance-dependent bound, nor a high-probability bound on the sampling error (only an expectation bound).

**No regret bound for the actual algorithm used in repeated games (GIGA) against adaptive adversaries:** Theorem 4 provides expected regret against oblivious deterministic environments. Lemma 11 extends this to a high-probability bound against arbitrary adaptive environments, but this bound controls $R(h) > 2T\epsilon$ with probability $\exp(-T\epsilon^2/(8|u|^2))$ — it does not directly bound expected regret against adaptive environments. The expectation of regret against an adaptive adversary could, in principle, be higher than against the worst-case oblivious adversary (though the paper's proof architecture shows it isn't, via Lemma 9). The paper does not provide an explicit expected regret bound for the adaptive case, only the high-probability tail bound.

#### What Would Have Strengthened the Paper

1. **A concrete numerical example** illustrating the bound for a specific online convex programming problem, even a toy one (e.g., tracking the minimum of a moving quadratic on a line segment), would have grounded the theoretical analysis and shown the constants in action.

2. **A lower bound** showing $\Omega(\sqrt{T})$ regret for some online convex programming instance would have established that the algorithm's rate is optimal, strengthening the paper's contribution from "an algorithm" to "the right algorithm."

3. **An explicit comparison** between the Greedy Projection bound and the Hedge algorithm's bound on a specific polytope (e.g., the $n$-dimensional $\ell_1$ ball or the hypercube) would have made the claimed advantage over experts algorithms concrete and verifiable.

4. **An analysis of the dynamic regret of Lazy Projection** would have completed the picture, since Lazy Projection achieves a better static regret constant and it is natural to ask whether it also handles non-stationary environments.

5. **A discussion of the $|A|$ dependence** in GIGA's regret bound — specifically, whether it is fundamental or an artifact of the analysis — would have contextualized the result within the broader game-theoretic literature, where exponential weighting achieves logarithmic dependence on $|A|$.

## 6. Limitations and Trade-offs

### 6.1 No Lower Bounds Establish the Tightness of the $\sqrt{T}$ Regret Rate

The paper proves an upper bound of $O(\sqrt{T})$ on the average regret of Greedy Projection (Theorem 1), but **never establishes whether this rate is optimal**. There is no lower bound showing that any algorithm for online convex programming must incur $\Omega(\sqrt{T})$ regret, nor any demonstration that the dependence on $\|\mathcal{F}\|$ or $\|\nabla c\|$ is tight.

The consequence is that a reader cannot assess whether Greedy Projection is fundamentally efficient or merely the first algorithm analyzed. If the optimal rate were $O(\log T)$ or $O(T^{1/4})$, then the $O(\sqrt{T})$ bound would represent a significant gap between what the algorithm achieves and what is possible. Conversely, if $\Omega(\sqrt{T})$ is the lower bound (as subsequent literature established), then the algorithm's rate is order-optimal, but this is not supported by evidence in the paper itself.

The paper provides **no measurement or analysis** of this limitation — it is entirely absent from the text. The authors do not discuss the question of optimality, do not reference any existing lower bounds from related settings (e.g., the $\Omega(\sqrt{T})$ lower bound for online linear optimization was known from the experts literature), and do not conjecture whether the $\sqrt{T}$ rate is improvable.

**Mitigation status:** None within the paper. The authors make no attempt to address this gap, nor do they flag it as a limitation. The result is presented as a positive guarantee without any discussion of whether it is best-possible. A reader must consult external literature to determine whether the algorithm's performance is near-optimal or substantially suboptimal.

---

### 6.2 Difficulty Estimation: The $|A|$ Dependence in GIGA Makes Large Action Spaces Impractical

When the general regret bound of Theorem 1 is specialized to repeated games via GIGA (Theorem 4), the regret scales as $|A| |u|^2 \sqrt{T}$, where $|A|$ is the number of actions available to the player. This linear dependence on the action space size is **exponentially worse** than the $O(\sqrt{T \ln |A|})$ regret achieved by multiplicative-weight / Hedge algorithms for the same setting.

The consequence is a sharp practical tradeoff. For games with small action spaces (e.g., $|A| = 2$ or $3$), the constants are small and GIGA is competitive with or better than alternatives. But as the number of actions grows — for example, in a game with $|A| = 100$ — the regret bound includes a factor of $100 \cdot |u|^2$, which may render the guarantee vacuous for any reasonable time horizon. A practitioner choosing between GIGA and Hedge for a 100-action game would want to know that the theoretical guarantee for gradient ascent degrades linearly in $|A|$ while Hedge degrades only logarithmically.

The paper **does not measure this empirically** (there are no experiments) but the scalars appear directly in the plug-in calculation of Theorem 4. The paper acknowledges this only implicitly in Section 4:

> "While the bounds on the performance of most experts algorithms depends on the number of experts, these bounds are based on other criterion which may sometimes be lower."

This framing presents the difference as a potential advantage for gradient methods (because diameter-based bounds can be smaller than vertex-count-based bounds), but does not address the **reverse case**: when $|A|$ is large and the diameter is small (e.g., a high-dimensional simplex), the gradient bound is substantially worse than the experts bound.

**Mitigation status:** The paper acknowledges the tradeoff only obliquely, and never explicitly calculates the GIGA bound for a concrete large-action game to demonstrate the $|A|$ penalty. No attempt is made to improve the dependence (e.g., through a different learning rate schedule, a different projection geometry, or incorporation of entropy regularization). The limitation is not flagged as an open problem.

---

### 6.3 The Euclidean Projection Assumption Restricts the Algorithm's Geometry

The entire regret analysis depends on the Euclidean projection operator $P(y) = \arg\min_{x \in \mathcal{F}} d(x, y)$ and the geometric inequality $(P(y) - x)^2 \leq (y - x)^2$ that it satisfies. The Lazy Projection analysis (Appendix B) goes further, relying on the specific Euclidean relationships in Lemmas 5-7 (the obtuse angle property, the dot-product-to-distance bounds, and Corollary 1's telescoping inequality). These are **fundamentally Euclidean results** that do not hold under general Bregman divergences or other non-Euclidean geometries.

The consequence is that the algorithm and its analysis do not extend to settings where a different metric is natural. For instance, if the feasible set is the probability simplex, the Euclidean projection is computationally tractable (e.g., via Duchi et al.'s algorithm), but it does not capture the information geometry of the simplex — an entropic projection (using KL divergence rather than Euclidean distance) would be more natural and might yield tighter bounds. More practically, for feasible sets where Euclidean projection is expensive but an alternative projection operator is cheap, the algorithm's requirement of exact Euclidean projection may make it computationally infeasible.

The paper does not **measure** this limitation, but it acknowledges it explicitly in the conclusions (Section 6):

> "here we deal with a Euclidean geometry: what if one considered gradient descent on a noneuclidean geometry, like [1, 24]?"

This is a transparent admission, but it is placed as future work rather than a limitation of the current results.

**Mitigation status:** The authors explicitly flag this as an open direction and cite Amari (1998) on natural gradient and Mahony and Williamson (2001) on non-Euclidean gradient descent as relevant prior art. No attempt is made within the paper to extend the analysis to Bregman divergences or mirror descent — the paper stays entirely within the Euclidean framework. A practitioner working with distributions (where KL divergence is natural) or with positive definite matrices (where Riemannian metrics apply) would need to verify whether the Euclidean projection onto their feasible set is even the right operation, let alone computationally feasible.

---

### 6.4 The Dynamic Regret Analysis Assumes a Known Path Length $L$

Theorem 2 bounds the dynamic regret against comparison sequences of path length at most $L$, but the bound depends on a **fixed learning rate $\eta$** that must be chosen in advance. The optimal choice of $\eta$ depends on $L$ (and $T$ and $\|\mathcal{F}\|$ and $\|\nabla c\|$), but $L$ is a property of the environment that is **not known to the algorithm at the start**. The paper does not provide an adaptive method for tuning $\eta$ online as the environment's degree of non-stationarity is revealed.

The consequence is that the dynamic regret guarantee is **not implementable as stated** in unknown environments. A practitioner must guess $L$ (the total amount the optimal decision will move over $T$ rounds) before seeing any data, and the quality of the bound depends on this guess. If $L$ is underestimated, the chosen $\eta$ will be too small, the algorithm will be too sluggish, and the tracking cost $L\|\mathcal{F}\|/\eta$ will dominate the regret. If $L$ is overestimated, $\eta$ will be too large, and the reactivity penalty $T\eta\|\nabla c\|^2/2$ will dominate.

The paper provides **no experimental or analytical** treatment of this issue. Theorem 2 is stated as a bound that holds for any fixed $\eta$, with no discussion of how to set $\eta$ without oracle knowledge of $L$. The static regret analysis (Theorem 1) uses a predetermined schedule $\eta_t = t^{-1/2}$ that requires no oracle knowledge, but the dynamic case offers no such schedule.

**Mitigation status:** None. The paper does not propose a doubling trick, an adaptive restart scheme, or a meta-algorithm that tunes $\eta$ from observed data. The dynamic regret result is presented as an "if you know $L$, this bound holds" guarantee rather than a "here is an algorithm that achieves this bound without knowing $L$" result. The gap between the oracle-tuned and practical performance is unaddressed.

---

### 6.5 The Online Convex Programming Framework Requires Exact Gradient Access at the Played Point

Assumption 6 of the paper's model requires that for each cost function $c_t$, there exists an algorithm that, given the current point $x_t$, produces the exact gradient $\nabla c_t(x_t)$. Footnote 1 relaxes this to requiring a subgradient $g$ satisfying $g \cdot (y - x) \leq c_t(y) - c_t(x)$ for all $y$, but in either case, the algorithm must query the cost function's derivative **at exactly the point it played**.

The consequence is that the framework does not handle **bandit feedback**, where the algorithm only observes the scalar cost $c_t(x_t)$ and must estimate the gradient from this single number. The gradient-based update $x_{t+1} = P(x_t - \eta_t \nabla c_t(x_t))$ is undefined without gradient access, and the entire linearization proof technique collapses because the algorithm cannot compute $g_t = \nabla c_t(x_t)$. All three algorithms (Greedy Projection, Lazy Projection, GIGA) require this oracle, and no variant of the framework is proposed to handle gradient-free feedback.

The paper **does not discuss this limitation** at all. The assumption is stated (Assumption 6, page 2) but never problematized. The repeated games application (Section 3.3) satisfies gradient access because the utility function is known and the opponent's action is observed, so $g_t = [-u(1, h_{t,2}), \ldots, -u(|A|, h_{t,2})]$ can be computed exactly. But the general online convex programming model is introduced as a framework for "factory production, farm production, and many other industrial optimization problems," where gradient access may not be available — the farmer observes profit after harvest but cannot compute the gradient of the profit function with respect to her planting decisions without knowing what the profit would have been for every alternative planting plan.

**Mitigation status:** None. The paper neither provides a bandit variant of the algorithm nor discusses whether the framework can be extended to gradient-free settings. A practitioner deploying this algorithm in a setting where gradient queries are expensive or impossible (as in many black-box optimization scenarios) would find no guidance in the paper.

---

### 6.6 The Analysis Does Not Account for Projection Complexity or Approximation Error

Assumption 7 requires an algorithm that can compute the exact Euclidean projection $P(y) = \arg\min_{x \in \mathcal{F}} d(x, y)$ for any $y \in \mathbb{R}^n$. The paper treats this as a black-box primitive and does not count its computational cost in the regret analysis, nor does it analyze what happens when the projection is solved only approximately.

The consequence is that the algorithm may be **far more expensive per iteration than the headline $\sqrt{T}$ bounds suggest**. For complex feasible sets (e.g., intersections of many halfspaces, or the convex hull of a large point set), exact Euclidean projection is itself a convex optimization problem that may require an iterative solver with its own convergence guarantees and computational cost. The paper's analysis implicitly assumes that projection costs $O(1)$ per round and that the results are exact, but in high dimensions or with complex constraints, projection can dominate the per-iteration runtime. Furthermore, if projection is computed with finite precision, the geometric inequality $(P(y) - x)^2 \leq (y - x)^2$ that underpins the entire regret analysis may be violated, and the regret bound does not account for the accumulated approximation error.

The paper provides **no analysis** of this issue. Projection is treated as an oracle, and the seven assumptions (page 2) list it as a requirement without any discussion of its cost. The paper does not provide algorithms for computing $P(y)$ for any specific feasible sets, does not bound the complexity of projection, and does not discuss what happens if an $\epsilon$-approximate projection is used instead.

**Mitigation status:** None. The paper assumes the existence of an exact projection oracle and builds the entire theory on that foundation. For a practitioner choosing whether to implement Greedy Projection, the question "how expensive is it to project onto my feasible set at each iteration?" is a critical practical concern that the paper does not address. The framework is sufficiently abstract that it applies to any closed convex set for which projection is tractable (simplices, balls, boxes, halfspaces), but it provides no tools for determining whether a given feasible set meets this criterion, and no guarantees for the case where it does not.
