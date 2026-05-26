# Least Angle Regression

**URL:** [https://hastie.su.domains/Papers/LARS/LeastAngle_2002.pdf](https://hastie.su.domains/Papers/LARS/LeastAngle_2002.pdf)

## 🎯 Pitch

This paper introduces **Least Angle Regression (LARS)**, a new model selection algorithm for linear regression that builds up coefficient estimates in a sequence of equiangular steps—each step moving along a direction that makes equal angles with the currently most correlated predictors.

---

## 1. Executive Summary

This paper introduces **Least Angle Regression (LARS)**, a new model selection algorithm for linear regression that builds up coefficient estimates in a sequence of equiangular steps—each step moving along a direction that makes equal angles with the currently most correlated predictors. Tested on a diabetes dataset (10 covariates, 442 patients) and a 64-predictor quadratic expansion, LARS provides a computationally efficient backbone from which two simple modifications produce the full solution paths for both the Lasso (a constraint on the sum of absolute coefficients) and Forward Stagewise Linear Regression (an iterative, small-step procedure that moves toward the most correlated residual direction). The paper establishes that (1) the **LARS/Lasso relationship** reduces Lasso computation by roughly an order of magnitude relative to prior quadratic programming methods, (2) the **LARS/Stagewise relationship** explains the previously observed near-identity of Lasso and Stagewise coefficient paths by showing both are constrained versions of the equiangular LARS strategy, and (3) a **simple degrees-of-freedom approximation** (`df ≈ k` for the k-step LARS estimate) enables a Mallows Cp criterion that selects among the sequence of LARS estimates with no additional computation beyond the original fit. In a FLOPs-matched sense, LARS computes the full regularization path at a cost comparable to a single ordinary least-squares fit on all covariates, establishing that principled model selection along the Lasso/Stagewise spectrum need not carry a prohibitive computational burden.

## 2. Context and Motivation

### The Core Problem: Model Selection at Scale

The fundamental problem this paper tackles is deceptively simple: **given a large set of possible predictor variables, how do you select a parsimonious subset that yields accurate predictions of a response variable?** This is the canonical model selection problem in linear regression, and it matters enormously. The paper's opening example—the diabetes study with 442 patients and 10 baseline variables (age, sex, BMI, blood pressure, and six blood serum measurements)—captures the dual demands that drive model selection in practice:

- **Prediction accuracy**: the model should produce reliable baseline predictions of disease progression for future patients.
- **Scientific interpretability**: the form of the model should suggest which covariates are actually important factors in disease progression.

These two goals are often in tension. Including all 10 variables (the full OLS model) maximizes in-sample fit but risks overfitting, producing poor predictions on new patients and obscuring which variables genuinely matter. A model with only 3 or 4 carefully chosen variables can be more interpretable and may predict better, but which 3 or 4? This is the selection problem, and it grows combinatorially: with $m$ predictors there are $2^m$ possible subsets to consider. For the 64-predictor quadratic model the paper later examines (10 main effects + 45 interactions + 9 squares from the diabetes data), that's $2^{64} \approx 1.8 \times 10^{19}$ possible models—impossible to exhaustively evaluate.

The gap the paper addresses is not that model selection algorithms don't exist, but that the **dominant existing methods are either too greedy (making impulsive, irreversible decisions that can eliminate useful predictors), too computationally expensive (requiring quadratic programming solvers for each regularization parameter value), or both.** The paper positions LARS as a principled middle ground: more cautious than Forward Selection, yet requiring only the same order of computational effort as a single full OLS fit.

### The Landscape of Existing Methods and Their Shortcomings

Before LARS, practitioners faced a tradeoff between computational efficiency and statistical prudence. The paper identifies four families of existing approaches, each with specific limitations that motivate the LARS development.

#### Forward Selection: Fast but Dangerously Greedy

Forward Selection is perhaps the most intuitive model-building algorithm and serves as the paper's primary foil. As described in Section 1 (citing Weisberg, 1980), the procedure works as follows:

1. Start with no predictors in the model.
2. Find the predictor $x_{j_1}$ having the largest absolute correlation with the response $y$.
3. Perform simple linear regression of $y$ on $x_{j_1}$, producing a residual vector orthogonal to $x_{j_1}$.
4. Project all other predictors orthogonally to $x_{j_1}$ (to remove the component already explained).
5. Repeat the selection process on the residual, adding a second predictor $x_{j_2}$, then a third, etc.

After $k$ steps, you have a $k$-predictor linear model. The procedure is computationally cheap—it makes exactly one "greedy" choice per step—but this greediness is precisely the problem. The paper characterizes it bluntly:

> "Forward Selection is an aggressive fitting technique that can be overly greedy, perhaps eliminating at the second step useful predictors that happen to be correlated with $x_{j1}$."

The issue is irreversibility. Once a predictor is not chosen at a given step, it is effectively removed from consideration in subsequent steps because the residual has been orthogonalized with respect to the chosen variable. If $x_{j_2}$ is moderately correlated with $x_{j_1}$ and also strongly predictive of $y$, Forward Selection might skip it at step 2 (because $x_{j_1}$ already explains some of its signal), and $x_{j_2}$ may never be reconsidered. This is the "overly greedy" critique that motivates the entire paper: **a good model selection procedure should allow compromise among correlated predictors rather than making winner-take-all decisions at each step.**

The simulation study in Section 3.3 (Figure 5) provides empirical evidence of this flaw. In a 100-replication study using the diabetes quadratic model (64 predictors), Forward Selection's proportion explained peaks at 0.950 after only 3 steps, then declines more steeply than the LARS/Lasso/Stagewise curves. The rapid rise and subsequent fall are characteristic of a method that commits too early to individual predictors.

#### All Subsets and Backward Elimination: Principled but Impractical

At the opposite extreme, **All Subsets regression** evaluates every possible subset of predictors to find the one that optimizes some criterion (e.g., minimum residual sum of squares for a given model size, or minimum AIC/BIC). This is exhaustive and, in principle, optimal—but it is computationally infeasible for $m$ beyond about 30–40 due to the combinatorial explosion. For the diabetes quadratic model with 64 predictors, All Subsets is impossible.

**Backward Elimination** starts with all $m$ predictors and sequentially removes the least significant one (by some criterion, typically partial F-test or AIC), refitting after each deletion. While less greedy than Forward Selection (since it starts from the full model and can "see" joint effects), it is computationally expensive when $m$ is large because the initial fit involves all predictors, and it cannot be applied at all when $m > n$ (more predictors than observations)—the full model is not identifiable. The paper does not dwell on Backward Elimination, but it sits in the same "expensive or limited" category.

#### The Lasso: Principled and Attractive, but Computationally Expensive

The Lasso (Tibshirani, 1996) represents a major conceptual advance over Forward Selection. Rather than making discrete inclusion/exclusion decisions, the Lasso frames model selection as a **continuous optimization problem**:

$$\text{Lasso: minimize } S(\hat{\beta}) = \|y - X\hat{\beta}\|^2 \quad \text{ subject to } \quad T(\hat{\beta}) = \sum_{j=1}^m |\hat{\beta}_j| \leq t$$

Here $t$ is a regularization parameter that controls the total absolute coefficient mass. As $t$ varies from 0 to $\sum |\hat{\beta}_j^{\text{OLS}}|$ (the unconstrained OLS solution), the Lasso produces a continuous family of solutions $\hat{\beta}(t)$. This approach has two enormously attractive properties that Forward Selection lacks:

1. **Shrinkage**: coefficients are pulled toward zero, trading off decreased variance for increased bias—a mechanism that often improves prediction accuracy, particularly when predictors are correlated (discussed further in Hastie, Tibshirani & Friedman, 2001).

2. **Parsimony through sparsity**: for any given $t$, only a subset of the covariates have non-zero $\hat{\beta}_j$. At $t = 1000$ in the diabetes example (left panel of Figure 1), only variables 3, 9, 4, and 7 have non-zero coefficients. **Variable selection emerges naturally from the $\ell_1$ constraint geometry** rather than from ad-hoc significance testing.

The left panel of Figure 1—which shows all 10 coefficient paths $\hat{\beta}_j(t)$ as $t$ increases—illustrates both the Lasso's elegance and, implicitly, the problem that LARS solves. The paths are piecewise linear, with variables entering the model sequentially as $t$ increases (order: 3, 9, 4, 7, ..., 1). This piecewise linear structure suggests that a more direct computational strategy should exist.

Prior to LARS, computing the Lasso required solving a quadratic programming problem. Osborne, Presnell, and Turlach (2000a, 2000b) had developed a "homotopy method" that tracks the solution path by following the active set of non-zero coefficients, but this approach was not widely known in the statistical community and still involved the machinery of convex optimization. The paper's key insight is that **the Lasso's piecewise linear path can be generated by a simple modification of a forward stepwise-style algorithm**—specifically, by enforcing a sign consistency constraint (Equation 3.1) on the equiangular steps that LARS naturally takes. This connection reduces Lasso computation by roughly an order of magnitude.

#### Forward Stagewise: Promising but Painfully Slow

Forward Stagewise Linear Regression (henceforth "Stagewise") represents yet another point in the design space—and its unexpected similarity to the Lasso is one of the paper's two major motivational puzzles. The Stagewise algorithm, described in Section 1, is a cautious cousin of Forward Selection:

1. Start with $\hat{\mu} = 0$ (all coefficients zero).
2. Compute the current correlation vector $\hat{c} = X'(y - \hat{\mu})$, where $\hat{c}_j$ is proportional to the correlation between predictor $x_j$ and the current residual.
3. Find the predictor with the greatest absolute current correlation: $\hat{j} = \arg\max_j |\hat{c}_j|$.
4. Take a **small step** in that direction: $\hat{\mu} \rightarrow \hat{\mu} + \epsilon \cdot \text{sign}(\hat{c}_{\hat{j}}) \cdot x_{\hat{j}}$, where $\epsilon$ is some small constant.

The critical word here is **"small."** If $\epsilon$ were set to $|\hat{c}_{\hat{j}}|$ (making the residual orthogonal to $x_{\hat{j}}$), Stagewise would reduce to Forward Selection. By taking tiny steps, Stagewise avoids the irreversibility problem: if predictor $j$ is chosen at one step but predictor $k$ becomes more correlated with the residual at the next step, Stagewise can immediately switch directions. Over thousands of such infinitesimal steps, the algorithm navigates a cautious path through predictor space, never committing too heavily to any single direction.

The right panel of Figure 1 shows the Stagewise coefficient paths for the diabetes data, computed using 6,000 steps (with $\epsilon$ small enough to hide the discrete staircase structure). The **striking empirical observation** that motivates the entire paper is this:

> "The striking fact is the similarity between the Lasso and Stagewise estimates. Although their definitions look completely different, the results are nearly, but not exactly, identical."

This similarity had been noted empirically by Hastie et al. (2001), but no one had explained **why** two procedures with such different definitions—one a convex optimization problem with an $\ell_1$ constraint, the other an iterative greedy algorithm with infinitesimal steps—should produce nearly identical coefficient paths. The paper's resolution of this puzzle, through the unifying LARS geometry, is one of its central contributions.

The Stagewise procedure has an additional attraction beyond its statistical properties: unlike the Lasso, it generalizes naturally to non-linear settings. As the paper discusses in Section 8, Stagewise ideas underlie **boosting** (Freund & Schapire, 1997), one of the most effective prediction methods in machine learning. In least-squares boosting, one repeatedly fits regression trees to the current residual and takes small steps in the direction of the fitted tree—a procedure that is structurally identical to Forward Stagewise regression with an infinite set of tree predictors. Understanding why Stagewise works well in the linear case thus has direct implications for understanding boosting, a connection the paper explores in its final section.

But the computational cost of Stagewise is prohibitive: 6,000 steps for 10 predictors, and for the 64-predictor quadratic model, even more. The practical utility of Stagewise—both for linear regression and as a conceptual bridge to boosting—demands a faster implementation. The paper's discovery that **an equiangular strategy with step sizes computed analytically can short-circuit the thousands of tiny Stagewise steps** is the second major computational contribution, reducing the full Stagewise path to a sequence of at most $m$ (or slightly more, with modifications) analytically computed steps.

### The Unifying Gap: No Principled, Computationally Efficient Framework

The landscape in 2003 was thus fragmented. Practitioners could choose:

- **Forward Selection**: fast ($O(m^2)$ operations) but dangerously greedy and statistically unreliable.
- **All Subsets**: statistically principled but computationally impossible beyond small $m$.
- **The Lasso**: statistically attractive with its shrinkage and sparsity properties, but requiring quadratic programming solvers—slow and not widely accessible to statisticians working in standard computing environments.
- **Forward Stagewise**: cautious, empirically effective, and theoretically promising (especially for boosting connections), but excruciatingly slow due to the requirement of thousands of tiny steps.

The paper identifies a **missing middle ground**: a computationally efficient algorithm that is statistically more cautious than Forward Selection, that produces the Lasso and Stagewise solution paths as special cases, and that can be implemented with standard linear algebra tools at a cost comparable to a single OLS fit. LARS fills precisely this gap.

### How the Paper Positions Itself

The paper's intellectual positioning is three-fold, corresponding to its three main contributions:

**First, LARS as a standalone method with its own inferential machinery.** The paper does not frame LARS merely as a computational shortcut for existing methods. Section 4 develops a degrees-of-freedom analysis (the "simple approximation" $df(\hat{\mu}_k) \doteq k$) and a corresponding $C_p$ criterion that applies specifically to LARS estimates—**not** to Lasso or Stagewise estimates. This gives LARS an independent identity as a model selection tool: a practitioner can run LARS (always exactly $m$ steps), use the $C_p$ formula to select the optimal step $k$, and report both the selected model and an estimate of its prediction error—all without additional computation. The $C_p$ formula, Equation 4.10, is remarkably simple:

$$C_p(\hat{\mu}_k) \doteq \|y - \hat{\mu}_k\|^2 / \bar{\sigma}^2 - n + 2k$$

This is **the same formula** as the $C_p$ estimate for OLS based on $k$ preselected predictors, but it applies to the adaptively selected LARS model. The paper is careful to note that this formula holds exactly under orthogonal designs (Theorem 3) and under the more general Positive Cone Condition (Theorem 4, Equation 4.11), and provides both bootstrap and delta-method evidence that it is a good approximation in practice even when these conditions are violated (Figure 6). This inferential contribution distinguishes LARS from being "just" a faster way to compute Lasso solutions.

**Second, the LARS/Lasso connection as a computational breakthrough with theoretical depth.** The paper shows that a minor modification of the LARS algorithm—enforcing the sign constraint $\text{sign}(\hat{\beta}_j) = \text{sign}(\hat{c}_j)$ for active predictors (Equation 3.1)—causes LARS to trace exactly the Lasso solution path. This is formalized in Theorem 1 and proven through a series of lemmas (Lemmas 7–10) that characterize the Lasso path geometrically: the estimates move linearly along equiangular directions, with active sets that can grow and shrink according to the Lasso modification rule (Equation 3.6). The paper positions this not as a mere computational trick but as a **revelation of the geometric structure underlying the Lasso**: the Lasso is a constrained version of LARS that cannot take equiangular steps which would cause coefficient sign reversals. This insight makes the Lasso's behavior more interpretable to a statistical audience familiar with Forward Selection concepts.

**Third, the LARS/Stagewise relationship as a unification of apparently disparate methods.** The near-identity of Lasso and Stagewise coefficient paths had been an empirical puzzle. The paper resolves it by showing that both are constrained versions of the same equiangular strategy: Lasso constrains the **signs of coefficients** at each step (coefficient signs must match current correlation signs), while Stagewise constrains the **signs of coefficient changes** at each step (successive differences must match current correlation signs). Section 3.2 makes this comparison explicit:

- **Stagewise**: successive differences of $\hat{\beta}_j$ agree in sign with the current correlation $\hat{c}_j$.
- **Lasso**: $\hat{\beta}_j$ agrees in sign with $\hat{c}_j$.
- **LARS**: no sign restrictions (unconstrained equiangular steps).

From this perspective, Lasso is "intermediate" between LARS and Stagewise in terms of constraint severity. When the equiangular direction $u_{\mathcal{A}}$ naturally lies within the convex cone $\mathcal{C}_{\mathcal{A}}$ generated by the active predictors (Equation 3.12), all three methods coincide. When it does not—as happens at the arrowed point in the right panel of Figure 1, where the active set $\mathcal{A} = \{3,9,4,7,2,10,5,8\}$ was reduced to $\hat{\mathcal{B}} = \mathcal{A} - \{3,7\}$—the Stagewise modification projects $u_{\mathcal{A}}$ onto the nearest face of the convex cone, producing a different (and more cautious) direction of progress. Theorem 2 formalizes this connection, and Lemma 12 in Section 6 proves uniqueness: the Stagewise direction of advance is the unique vector satisfying the three geometric constraints (I: non-negative simplex, II: equiangular for a subset, III: maximal correlation decline rate) that characterize idealized Stagewise behavior.

### The Boosting Connection: A Forward-Looking Motivation

Section 8 reveals an additional motivation that goes beyond linear regression. Forward Stagewise regression is structurally identical to **least-squares boosting** with regression trees (Friedman, 2001; Hastie et al., 2001, Chapter 10), where the "predictors" are not the original $m$ covariates but the infinite set of all possible regression trees that could be fit to the data. In boosting, one repeatedly fits a tree to the current residual and takes a small step in that direction—exactly the Stagewise algorithm with trees as the base learner.

The paper notes two important consequences of the LARS/Lasso/Stagewise connection for boosting research:

> "Hastie et al. (2001) noted the striking similarity between Forward Stagewise regression and the Lasso, and conjectured that this may help explain the success of the Forward Stagewise process used in least-squares boosting. That is, in some sense least squares boosting may be carrying out a Lasso fit on the infinite set of tree predictors."

LARS itself cannot be directly applied to boosting since computing the optimal equiangular step among infinitely many tree predictors is infeasible. But the connection suggests a "modified form of Forward Stagewise" that, instead of taking a small step in only the most correlated tree, takes a small least-squares step in **all trees currently in the model**—a procedure that approximates LARS and can be implemented with the infinite predictor set. This forward-looking motivation positions LARS not as an end in itself but as a conceptual bridge that clarifies how greedy forward procedures (Forward Selection, Stagewise, boosting) relate to regularized optimization (the Lasso), with implications for designing better boosting algorithms.

### Summary: Why This Paper Matters

The paper addresses a gap that is simultaneously computational, statistical, and conceptual:

- **Computationally**, it reduces Lasso and Stagewise computation from quadratic programming or thousands of small steps to roughly the cost of a single OLS fit (order $O(m^3 + nm^2)$ for the full $m$-step sequence).
- **Statistically**, it provides a simple degrees-of-freedom formula ($df \approx k$) and $C_p$ criterion for the LARS estimates themselves, giving the method standalone value beyond being a computational shortcut.
- **Conceptually**, it reveals the geometric unity underlying Forward Selection, the Lasso, and Stagewise—all are greedy forward procedures that differ only in the constraints imposed on their equiangular progress—and connects this unity to the puzzle of why boosting works.

## 3. Technical Approach

### 3.1 Reader Orientation

This paper develops **Least Angle Regression (LARS)**, an algorithm for building linear regression models that constructs coefficient estimates through a sequence of analytically computed "equiangular" steps—each step moving the prediction vector in a direction that bisects the angle between the currently most correlated predictors. The problem it solves is the computational-statistical gap in model selection: how to efficiently trace out a full spectrum of parsimonious linear models—from the null model to the full OLS fit—at roughly the cost of a single least-squares computation, while naturally recovering the solution paths of both the Lasso and Forward Stagewise regression as constrained special cases.

### 3.2 Big-Picture Architecture (Diagram in Words)

The LARS framework has four interconnected components:

1. **The core LARS algorithm** — a forward stagewise procedure that, unlike classic Forward Selection, does not commit fully to one predictor at a time. Instead, it moves the prediction vector in the "least angle direction"—a unit vector that makes equal angles with all currently active predictors—and adjusts the step size analytically so that a new predictor joins the active set exactly when its correlation with the residual matches the current maximum.

2. **The Lasso modification** — a constraint mechanism layered on top of LARS that enforces $\text{sign}(\hat{\beta}_j) = \text{sign}(\hat{c}_j)$ for active predictors. When a coefficient would cross zero during an equiangular move, this modification halts the step early, drops the offending variable from the active set, and recomputes the equiangular direction. This produces the exact Lasso solution path.

3. **The Stagewise modification** — an alternative constraint mechanism that projects the LARS equiangular direction onto the convex cone generated by the active predictors. When the unconstrained equiangular vector has negative components in the cone's coordinate system (meaning it would require moving *against* the current correlation signs), the modification replaces it with the nearest direction that stays within the cone, producing the idealized Forward Stagewise path.

4. **The inferential wrapper** — a degrees-of-freedom approximation ($df(\hat{\mu}_k) \doteq k$) and corresponding $C_p$ formula that enables model selection along the LARS path without additional computation.

Information flows as follows: the data $(X, y)$ enters the system → LARS initializes at $\hat{\mu}_0 = 0$ with an empty active set → at each step $k$, the algorithm computes current correlations $\hat{c}_j = x_j'(y - \hat{\mu}_{k-1})$ for all predictors, identifies the active set $\mathcal{A}$ of maximally correlated predictors, computes the equiangular vector $u_{\mathcal{A}}$ (Equation 2.6), calculates the step size $\hat{\gamma}$ (Equation 2.13) that will bring a new predictor into the active set, and updates $\hat{\mu}_k = \hat{\mu}_{k-1} + \hat{\gamma} u_{\mathcal{A}}$ → if running with the Lasso or Stagewise modifications, constraint checks at each step may trigger variable removals and direction recomputation → after $m$ steps (pure LARS) or potentially more (modified versions), the algorithm reaches the full OLS solution → the $C_p$ criterion (Equation 4.10) can be evaluated at each step to select the optimal model size, requiring no extra computation beyond the forward pass.

### 3.3 Roadmap for the Deep Dive

- **First, the equiangular geometry** (Equations 2.4–2.7), which defines the core mathematical object that LARS moves along. This is the foundation that everything else builds on, and understanding it is essential before the algorithm's mechanics make sense.

- **Second, the core LARS step** (Equations 2.8–2.13), which shows how step sizes are computed analytically to bring exactly one new predictor into the active set. This is where the computational efficiency comes from: rather than taking thousands of tiny Stagewise steps, LARS jumps directly to the next "event" in correlation space.

- **Third, the Lasso modification** (Equations 3.1–3.6), which adds the sign-consistency constraint and the variable-dropping mechanism. This builds naturally on the core step because the constraint only matters at points where an equiangular move would violate it.

- **Fourth, the Stagewise modification** (Equations 3.10–3.14), which interprets the Stagewise infinitesimal limit as a projection problem onto a convex cone. This is the most geometrically subtle of the three variants and reveals why Lasso and Stagewise are nearly identical in practice.

- **Fifth, the degrees-of-freedom and $C_p$ analysis** (Section 4), which gives LARS standalone value as an inferential tool. This depends on understanding the LARS estimate structure (particularly the geometric relationship to OLS projections shown in Figure 4).

- **Sixth, the computational organization** (Section 7), which explains how Cholesky factorization updates keep the full $m$-step cost at $O(m^3 + nm^2)$.

### 3.4 Detailed, Sentence-Based Technical Breakdown

This is primarily a **methodological paper with formal proofs** whose core idea is that model selection along the Lasso/Stagewise spectrum can be reduced to computing a sequence of equiangular directions and analytically determined step sizes, with constraint modifications that handle coefficient sign restrictions.

---

#### The Equiangular Vector: The Central Geometric Object

The entire LARS algorithm rests on one geometric construction: given a set of active predictors, find the unit vector that makes equal angles with all of them. This vector defines the direction of LARS progress, and its properties determine how correlations evolve, when new variables enter, and how the Lasso and Stagewise modifications constrain the path.

**Definition.** Let $\mathcal{A}$ be a subset of the predictor indices $\{1, 2, \ldots, m\}$. For each $j \in \mathcal{A}$, we associate a sign $s_j = \pm 1$ (the sign of the current correlation between $x_j$ and the residual—whether the predictor is positively or negatively correlated with what remains to be explained). Define the signed design matrix for the active set:

$$X_{\mathcal{A}} = (\cdots \; s_j x_j \; \cdots)_{j \in \mathcal{A}}$$

where $X_{\mathcal{A}}$ is an $n \times |\mathcal{A}|$ matrix whose columns are the original predictor vectors $x_j$, each multiplied by its current sign $s_j$. This sign convention is crucial: it means we always think of active predictors as "positively oriented" relative to the current residual, so that increasing any signed predictor reduces the residual equally.

**The Gram matrix and the normalizing constant.** From $X_{\mathcal{A}}$, we compute the $|\mathcal{A}| \times |\mathcal{A}|$ Gram matrix:

$$\mathcal{G}_{\mathcal{A}} = X_{\mathcal{A}}' X_{\mathcal{A}}$$

where each entry $[\mathcal{G}_{\mathcal{A}}]_{jk} = s_j s_k \cdot x_j' x_k$ is the signed inner product between predictors $j$ and $k$. Since the original $x_j$ have unit length ($\|x_j\|^2 = 1$ after standardization per Equation 1.1), the diagonal entries are $1$ and off-diagonals are signed correlations.

The next quantity is the **scalar normalizer** $\mathcal{A}_{\mathcal{A}}$:

$$A_{\mathcal{A}} = (1_{\mathcal{A}}' \mathcal{G}_{\mathcal{A}}^{-1} 1_{\mathcal{A}})^{-1/2}$$

where $1_{\mathcal{A}}$ is a column vector of $|\mathcal{A}|$ ones, and $\mathcal{G}_{\mathcal{A}}^{-1}$ is the inverse of the Gram matrix.

**What it computes:** $A_{\mathcal{A}}$ is a single positive number. Operationally, take the sum of all entries of the inverse Gram matrix (that is, $1_{\mathcal{A}}' \mathcal{G}_{\mathcal{A}}^{-1} 1_{\mathcal{A}}$—a quadratic form in the vector of ones), then take the reciprocal square root. The result is a number between 0 and 1: if $\mathcal{A}$ is a singleton, $A_{\mathcal{A}} = 1$; as more predictors join, $A_{\mathcal{A}}$ decreases.

**Why this form:** Lemma 5 proves that $A_{\mathcal{A}}$ is the length of the point in the extended simplex $\mathcal{S}_{\mathcal{A}}$ (the set of all linear combinations $\sum_{j \in \mathcal{A}} s_j x_j P_j$ with $\sum P_j = 1$, where $P_j$ can be negative) that is nearest to the origin. It is the **minimum possible length** of any convex combination of the signed predictors. This geometric interpretation is essential for the Lasso modification (Constraint IV in Lemma 10) because the fastest-descent direction for the residual sum of squares at a given total coefficient norm is the equiangular vector, and its effectiveness is measured by $A_{\mathcal{A}}$.

**The equiangular vector.** With $A_{\mathcal{A}}$ defined, we compute two intermediate objects:

$$w_{\mathcal{A}} = A_{\mathcal{A}} \mathcal{G}_{\mathcal{A}}^{-1} 1_{\mathcal{A}}$$

where $w_{\mathcal{A}}$ is a vector of length $|\mathcal{A}|$ (one weight per active predictor), and

$$u_{\mathcal{A}} = X_{\mathcal{A}} w_{\mathcal{A}}$$

where $u_{\mathcal{A}}$ is an $n$-vector (living in the same space as the response $y$ and the predictions $\hat{\mu}$). This $u_{\mathcal{A}}$ is the **equiangular vector**.

**What it computes:** $u_{\mathcal{A}}$ is the unit-length vector that makes equal angles (all less than $90^\circ$) with every column of $X_{\mathcal{A}}$. Concretely, the inner product between $u_{\mathcal{A}}$ and each signed predictor $s_j x_j$ is exactly $A_{\mathcal{A}}$:

$$X_{\mathcal{A}}' u_{\mathcal{A}} = A_{\mathcal{A}} 1_{\mathcal{A}}$$

and the squared length is $\|u_{\mathcal{A}}\|^2 = 1$.

**Why this form:** If the active predictors were orthogonal, $u_{\mathcal{A}}$ would simply be the normalized sum $\frac{1}{\sqrt{|\mathcal{A}|}} \sum_{j \in \mathcal{A}} s_j x_j$—the direction that points "straight down the middle" of all active predictors. With correlated predictors, the Gram matrix inverse $\mathcal{G}_{\mathcal{A}}^{-1}$ corrects for redundancy: predictors that are highly correlated with others already in $\mathcal{A}$ receive smaller weights $w_j$, because moving in their direction would partially duplicate the contribution of other active predictors. The equiangular property $X_{\mathcal{A}}' u_{\mathcal{A}} = A_{\mathcal{A}} 1_{\mathcal{A}}$ is the defining condition that makes LARS work: when moving along $u_{\mathcal{A}}$, the absolute correlations of all active predictors with the residual **decline at exactly the same rate** (Equation 2.16), maintaining the "most correlated" status of the active set until a new predictor catches up.

**The inner product with non-active predictors.** A final preparatory computation is:

$$a \equiv X' u_{\mathcal{A}}$$

where $a$ is an $m$-vector whose $j$th component $a_j = x_j' u_{\mathcal{A}}$ is the inner product between the equiangular direction and each predictor $x_j$ (in its original, unsigned orientation).

**What it computes:** For active predictors $j \in \mathcal{A}$, $a_j = s_j A_{\mathcal{A}}$, reflecting the equal-angle property. For inactive predictors $j \notin \mathcal{A}$, $a_j$ measures how quickly the correlation $c_j(\gamma)$ with the residual changes as we move along $u_{\mathcal{A}}$. This governs when an inactive predictor will "catch up" to the active set.

---

#### The Core LARS Step: Equiangular Progress with Analytical Step Size

The LARS algorithm proceeds iteratively. Suppose we are at step $k-1$, having built up the prediction vector $\hat{\mu}_{k-1}$ and identified an active set $\mathcal{A}$ of predictors that are all equally correlated with the current residual. The goal of step $k$ is to determine how far we can move along the equiangular direction $u_{\mathcal{A}}$ before some new predictor achieves the same correlation and must join the active set.

**Current correlations.** At the start of step $k$, we compute:

$$\hat{c} = X'(y - \hat{\mu}_{\mathcal{A}})$$

where $\hat{c}$ is the $m$-vector of current correlations between each predictor $x_j$ and the residual $y - \hat{\mu}_{\mathcal{A}}$. Each $\hat{c}_j$ is proportional to the sample correlation because the $x_j$ have unit length.

**The active set and maximum correlation.** We identify:

$$\hat{C} = \max_j \{|\hat{c}_j|\}$$

the maximum absolute correlation, and

$$\mathcal{A} = \{j : |\hat{c}_j| = \hat{C}\}$$

the set of indices achieving this maximum. By construction, all active predictors have $|\hat{c}_j| = \hat{C}$.

**Signs.** For each $j \in \mathcal{A}$, we record:

$$s_j = \text{sign}\{\hat{c}_j\}$$

This $s_j$ is $+1$ if predictor $j$ is positively correlated with the residual, $-1$ if negatively. These signs feed into the construction of $X_{\mathcal{A}}$ (Equation 2.4) and remain constant within a single LARS step.

**The equiangular direction.** We compute $X_{\mathcal{A}}$, $A_{\mathcal{A}}$, and $u_{\mathcal{A}}$ exactly as in Equations 2.4–2.6, and the inner product vector $a = X' u_{\mathcal{A}}$ (Equation 2.11).

**The trajectory parametrization.** LARS now considers a one-parameter family of predictions moving from the current estimate in the equiangular direction:

$$\mu(\gamma) = \hat{\mu}_{\mathcal{A}} + \gamma u_{\mathcal{A}}$$

for $\gamma > 0$. As $\gamma$ increases, the current correlations evolve linearly:

$$c_j(\gamma) = x_j'(y - \mu(\gamma)) = \hat{c}_j - \gamma a_j$$

**What this means:** each correlation starts at $\hat{c}_j$ and decreases (if $a_j > 0$) or increases (if $a_j < 0$) at rate $a_j$ as we move along $u_{\mathcal{A}}$. For active predictors $j \in \mathcal{A}$, we have $a_j = s_j A_{\mathcal{A}}$, so:

$$|c_j(\gamma)| = \hat{C} - \gamma A_{\mathcal{A}} \quad \text{for } j \in \mathcal{A}$$

This is the crucial equiangular property: **all active correlations decline at the same rate $A_{\mathcal{A}}$**, maintaining equality among themselves while gradually decreasing in magnitude. This is what allows a new predictor to "catch up."

**The step size computation.** A new predictor $j \notin \mathcal{A}$ joins the active set when its absolute correlation $|c_j(\gamma)|$ reaches the declining maximum $\hat{C} - \gamma A_{\mathcal{A}}$. There are two ways this can happen:

- The correlation $\hat{c}_j(\gamma)$ itself catches the declining maximum from below. This occurs when $\hat{c}_j(\gamma) = \hat{C} - \gamma A_{\mathcal{A}}$, which solves to $\gamma = (\hat{C} - \hat{c}_j) / (A_{\mathcal{A}} - a_j)$.

- The negative correlation $-\hat{c}_j(\gamma)$ (corresponding to the reversed predictor $-x_j$) catches the maximum. This occurs when $-\hat{c}_j(\gamma) = \hat{C} - \gamma A_{\mathcal{A}}$, which solves to $\gamma = (\hat{C} + \hat{c}_j) / (A_{\mathcal{A}} + a_j)$.

The step size $\hat{\gamma}$ is the smallest positive value among all these candidates:

$$\hat{\gamma} = \min^+_{j \in \mathcal{A}^c} \left\{ \frac{\hat{C} - \hat{c}_j}{A_{\mathcal{A}} - a_j}, \frac{\hat{C} + \hat{c}_j}{A_{\mathcal{A}} + a_j} \right\}$$

**What "$\min^+$" means:** the minimum is taken over only those fractions that are positive. If a denominator is zero or negative, or if the resulting fraction is non-positive, that candidate is excluded. This ensures $\hat{\gamma} > 0$ and that at least one new predictor genuinely enters.

**What this computes:** $\hat{\gamma}$ is the exact distance along $u_{\mathcal{A}}$ at which the first inactive predictor achieves correlation equal to the (declining) active correlation. It is the largest step we can take before the "most correlated" set expands.

**Why this form:** The formula is the analytical solution to a geometric intersection problem. The two fractions correspond to the two possible ways a predictor can enter: in its original orientation (numerator $\hat{C} - \hat{c}_j$, tracking convergence of $\hat{c}_j(\gamma)$ upward to $\hat{C} - \gamma A_{\mathcal{A}}$) or in reversed orientation (numerator $\hat{C} + \hat{c}_j$, tracking convergence of $-\hat{c}_j(\gamma)$ upward to the same target). The denominators $A_{\mathcal{A}} \pm a_j$ are the differences in decline rates. This direct computation replaces thousands of small Stagewise steps with a single analytical jump to the next "event" in correlation space.

**The update.** The next LARS estimate is:

$$\hat{\mu}_{\mathcal{A}+} = \hat{\mu}_{\mathcal{A}} + \hat{\gamma} u_{\mathcal{A}}$$

with the new active set $\mathcal{A}+ = \mathcal{A} \cup \{\hat{j}\}$ where $\hat{j}$ is the minimizing index in Equation 2.13. The new maximum absolute correlation is:

$$\hat{C}+ = \hat{C} - \hat{\gamma} A_{\mathcal{A}}$$

**The termination convention.** When $\mathcal{A}$ contains all $m$ covariates, Equation 2.13 is undefined because $\mathcal{A}^c$ is empty. By convention, the algorithm takes $\hat{\gamma}_m = \hat{\gamma}_m = \hat{C}_m / A_m$, which makes $\hat{\mu}_m = \bar{y}_m$ (the full OLS projection of $y$ onto all $m$ predictors) and $\hat{\beta}_m$ equal to the full OLS estimate. This final step completes the path from $\hat{\mu}_0 = 0$ to the unconstrained OLS solution in exactly $m$ equiangular steps.

---

#### The Geometric Relationship to OLS (Why LARS "Approaches but Never Reaches")

Figure 4 illustrates a critical geometric fact about LARS that underpins both the degrees-of-freedom analysis (Section 4) and the intuition for why LARS is less greedy than Forward Selection. At each step $k$, let $\bar{y}_k$ be the OLS projection of $y$ onto the linear space $L(X_k)$ spanned by the $k$ currently active predictors. Equation 2.19 shows:

$$\bar{y}_k = \hat{\mu}_{k-1} + X_k \mathcal{G}_k^{-1} X_k'(y - \hat{\mu}_{k-1}) = \hat{\mu}_{k-1} + \frac{\hat{C}_k}{A_k} u_k$$

where the last equality uses the fact that $X_k'(y - \hat{\mu}_{k-1}) = \hat{C}_k 1_{\mathcal{A}}$ (all active correlations equal $\hat{C}_k$) and the definition $u_k = X_k w_k = X_k A_k \mathcal{G}_k^{-1} 1_{\mathcal{A}}$.

**What this says:** Starting from $\hat{\mu}_{k-1}$, the full OLS fit $\bar{y}_k$ lies in exactly the same direction as the LARS equiangular vector $u_k$, but further out. The distance from $\hat{\mu}_{k-1}$ to $\bar{y}_k$ is $\bar{\gamma}_k = \hat{C}_k / A_k$, while the LARS step size is $\hat{\gamma}_k$. The ratio:

$$\rho_k = \frac{\hat{\gamma}_k}{\bar{\gamma}_k} < 1$$

measures how much of the way to OLS the LARS step goes.

**Why this matters:** LARS always moves **toward** the OLS projection but stops short because a new predictor becomes equally correlated before the active-set residual is fully explained. Forward Selection, by contrast, goes all the way to $\bar{y}_k$ at each step (taking $\rho_k = 1$), which is why it can be overly greedy—it completely orthogonalizes the residual with respect to the chosen predictor before considering others, potentially locking out correlated alternatives. LARS's partial step ($\rho_k < 1$) leaves residual correlation with all active predictors, allowing new predictors to "catch up" and join the model. This is the mechanism that makes LARS less greedy.

The relationship $\hat{\mu}_k - \hat{\mu}_{k-1} = \frac{\hat{\gamma}_k}{\bar{\gamma}_k} (\bar{y}_k - \hat{\mu}_{k-1})$ (Equation 2.22) also yields the delta-method justification for the degrees-of-freedom approximation in Section 4: locally, $\hat{\mu}_k$ behaves like a linear smoother with an effective trace of $k$.

---

#### The Lasso Modification: Enforcing Sign Consistency

The unmodified LARS algorithm does not constrain the signs of the regression coefficients. The Lasso, by contrast, requires that the sign of any non-zero coefficient $\hat{\beta}_j$ match the sign of the current correlation $\hat{c}_j = x_j'(y - \hat{\mu})$ (Equation 3.1, proved in Lemma 8). This constraint follows from the Karush-Kuhn-Tucker conditions for the Lasso optimization problem: the gradient of the squared error must point opposite to the subgradient of the $\ell_1$ penalty, which forces coefficient signs to align with correlation signs for active variables.

**The potential violation.** As LARS moves along $\mu(\gamma) = \hat{\mu}_{\mathcal{A}} + \gamma u_{\mathcal{A}}$, the coefficient vector evolves as:

$$\beta_j(\gamma) = \hat{\beta}_j + \gamma \hat{d}_j$$

for $j \in \mathcal{A}$, where $\hat{d}_j = s_j w_{\mathcal{A},j}$ is the $j$th component of the weight vector times the sign, and $\hat{d}_j = 0$ for $j \notin \mathcal{A}$ (Equation 3.3). As $\gamma$ increases, a coefficient $\beta_j(\gamma)$ might cross zero if $\hat{\beta}_j$ and $\hat{d}_j$ have opposite signs. When this happens, the Lasso sign condition is violated because $\beta_j(\gamma)$ changes sign while $c_j(\gamma)$ has not—correlations for active variables all remain positive (in the signed sense) and decline together as $|c_j(\gamma)| = \hat{C} - \gamma A_{\mathcal{A}} > 0$.

**The detection mechanism.** For each active predictor $j \in \mathcal{A}$, the value of $\gamma$ at which its coefficient would cross zero is:

$$\gamma_j = -\hat{\beta}_j / \hat{d}_j$$

This is defined only when $\hat{d}_j \neq 0$ and $\hat{\beta}_j$ and $\hat{d}_j$ have opposite signs (making $\gamma_j > 0$). The earliest such crossing is:

$$\tilde{\gamma} = \min_{\gamma_j > 0} \{\gamma_j\}$$

with the convention $\tilde{\gamma} = \infty$ if no $\gamma_j > 0$ exists (Equation 3.5). Let $\tilde{j}$ be the index achieving this minimum.

**The Lasso modification rule.** At each LARS step, we compare $\tilde{\gamma}$ with the equiangular step size $\hat{\gamma}$ from Equation 2.13:

- **If $\tilde{\gamma} < \hat{\gamma}$:** The coefficient $\beta_{\tilde{j}}(\gamma)$ would cross zero before any new predictor enters the active set. We stop the current LARS step at $\gamma = \tilde{\gamma}$, update $\hat{\mu}_{\mathcal{A}+} = \hat{\mu}_{\mathcal{A}} + \tilde{\gamma} u_{\mathcal{A}}$, and **remove** $\tilde{j}$ from the active set: $\mathcal{A}+ = \mathcal{A} - \{\tilde{j}\}$. The algorithm then recomputes the equiangular direction with the reduced active set and continues.

- **If $\tilde{\gamma} \geq \hat{\gamma}$:** No coefficient crosses zero before a new predictor enters. The step proceeds as in standard LARS: $\mathcal{A}+ = \mathcal{A} \cup \{\hat{j}\}$ where $\hat{j}$ is the new predictor from Equation 2.13.

**What this accomplishes:** The modification ensures that the sequence of estimates satisfies the Lasso sign condition (Equation 3.1) at every point. This is exactly the condition that characterizes Lasso solutions (Lemmas 7–10 prove that the Lasso path is uniquely determined by the requirement that active coefficients maintain sign consistency while equiangular progress minimizes the residual sum of squares for a given $\ell_1$ norm).

**Why this works (proof sketch):** Lemma 7 shows that between breakpoints (where the active set is constant), the Lasso solution moves linearly along the equiangular direction $u_{\mathcal{A}}$ determined by the active set. Lemma 8 establishes the sign condition. Lemma 9 shows that the active set must also be the set of maximally correlated predictors. Lemma 10 then proves that at a breakpoint where $\mathcal{A}$ could change, the new active set must be a subset of $\mathcal{A}_{10} = \mathcal{A}_1 \cup \mathcal{A}_0$ (where $\mathcal{A}_1$ has non-zero coefficients and $\mathcal{A}_0$ has zero coefficients but maximal correlation), and that the equiangular direction for the new set must minimize $A$ subject to sign constraints. Theorem 1 follows by induction: starting from $\hat{\beta}_0 = 0$, the Lasso modification produces the unique sequence of active sets and moves that satisfy all Lasso necessary conditions.

**The one-at-a-time condition:** Theorem 1 assumes that at each breakpoint, only one variable is added to or removed from the active set. This is "the usual case for quantitative data, and can always be realized by adding a little jitter to the $y$ values." If multiple variables would enter or leave simultaneously (a tie in the $\gamma$ calculations), the correct Lasso active set must be found by checking all subsets, which is computationally more expensive. The authors note that their implementation (Section 7) does not handle many-at-a-time cases, though the underlying theory extends.

**Computational cost:** The Lasso modification adds steps relative to pure LARS because variables can leave and later re-enter the active set. In the diabetes example (Figure 1, left panel), LARS took exactly $m = 10$ steps while Lasso took 12 steps—variable 7 was briefly removed at the arrowed point, then restored one step later. For the 64-predictor quadratic model, Lasso required 103 steps versus 64 for pure LARS. Each additional step requires recomputing the equiangular direction with a modified active set, but the per-step cost remains low due to Cholesky updating/downdating (Section 7).

**Backward Lasso:** Section 3.4 notes that the Lasso can also be run backwards from the full OLS solution $\bar{\beta}_m$, using the sign condition (3.1) to identify the correct equiangular direction at each step and tracking coefficients backward until they hit zero. LARS and Stagewise do not share this property because they lack the sign restriction that makes the backward direction identifiable.

---

#### The Stagewise Modification: Projection onto the Convex Cone

The Forward Stagewise algorithm (Equations 1.6–1.7) takes infinitesimal steps: at each iteration, it finds the predictor most correlated with the current residual and moves a tiny amount $\epsilon$ in that signed direction. As $\epsilon \to 0$, the discrete staircase of small steps converges to a continuous path. The question is: what direction does this continuous path take when multiple predictors are tied for maximum correlation?

The answer is not the unconstrained equiangular direction $u_{\mathcal{A}}$ except in special cases. The reason is that Stagewise can only move in **non-negative** combinations of the signed active predictors.

**The convex cone constraint.** After $N$ infinitesimal steps from some estimate $\hat{\mu}$, let $N_j$ be the number of steps taken along signed predictor $s_j x_j$, and let $P_j = N_j / N$ be the proportion of steps in that direction. The net movement is proportional to:

$$v = \sum_{j \in \mathcal{A}} s_j x_j P_j$$

where $P_j \geq 0$ and $\sum_{j \in \mathcal{A}} P_j = 1$ (Lemma 11 proves that only maximally correlated predictors can receive non-zero steps). The set of all such vectors is the **convex cone** generated by the signed active predictors:

$$\mathcal{C}_{\mathcal{A}} = \left\{ v : v = \sum_{j \in \mathcal{A}} s_j x_j P_j, \; P_j \geq 0 \right\}$$

**The LARS direction versus the cone.** The LARS equiangular vector $u_{\mathcal{A}}$ corresponds to $P_j \propto w_{\mathcal{A},j}$ where $w_{\mathcal{A}} = A_{\mathcal{A}} \mathcal{G}_{\mathcal{A}}^{-1} 1_{\mathcal{A}}$. If all components of $w_{\mathcal{A}}$ are non-negative, then $u_{\mathcal{A}}$ lies within $\mathcal{C}_{\mathcal{A}}$ and Stagewise can follow it directly—the infinitesimal step proportions $P_j = w_{\mathcal{A},j} / \sum_{k \in \mathcal{A}} w_{\mathcal{A},k}$ are all valid (non-negative). In this case, LARS, Lasso, and Stagewise all coincide (this happens under the Positive Cone Condition of Equation 4.11).

If some $w_{\mathcal{A},j}$ is negative, then $u_{\mathcal{A}}$ lies **outside** $\mathcal{C}_{\mathcal{A}}$. Moving along $u_{\mathcal{A}}$ would require taking negative proportions of some signed predictors—that is, moving **against** their current correlation signs. Stagewise cannot do this because its steps only go in the direction of maximum positive correlation.

**The projection solution.** The Stagewise modification replaces $u_{\mathcal{A}}$ with its orthogonal projection onto the convex cone $\mathcal{C}_{\mathcal{A}}$. Geometrically (Figure 9, Lemma 12), this projection lands on a face of the cone corresponding to a subset $\hat{\mathcal{B}} \subsetneq \mathcal{A}$ where the projection has strictly positive weights. Let $u_{\hat{\mathcal{B}}}$ be the unit vector along this projection (the equiangular vector for $\hat{\mathcal{B}}$, scaled appropriately). Then the Stagewise-modified LARS proceeds using $u_{\hat{\mathcal{B}}}$ instead of $u_{\mathcal{A}}$.

**What this computes operationally:** Given the active set $\mathcal{A}$ and the LARS weight vector $w_{\mathcal{A}}$, we check whether any $w_{\mathcal{A},j} < 0$. If so, we drop the most negative component(s) from $\mathcal{A}$, recompute the equiangular vector for the reduced set, and check again. This is essentially the inner loop of the Non-Negative Least Squares (NNLS) algorithm (Lawson & Hanson, 1974), which finds the non-negative weight vector whose corresponding direction is nearest to the unconstrained equiangular vector. The resulting direction $u_{\hat{\mathcal{B}}}$ satisfies three constraints (Lemmas 11–12):

**Constraint I:** $u_{\hat{\mathcal{B}}}$ lies in the convex cone $\mathcal{C}_{\hat{\mathcal{B}}}$ (weights are non-negative). This ensures the direction is reachable by Stagewise infinitesimal steps.

**Constraint II:** $u_{\hat{\mathcal{B}}}$ is equiangular for $\hat{\mathcal{B}}$, meaning all predictors in $\hat{\mathcal{B}}$ have their correlations decline at the same rate $A_{\hat{\mathcal{B}}}$.

**Constraint III:** For predictors $j \in \mathcal{A} - \hat{\mathcal{B}}$ (those dropped from the active set), their correlations decline **faster** than those in $\hat{\mathcal{B}}$: $x_j' u_{\hat{\mathcal{B}}} > A_{\hat{\mathcal{B}}}$. This ensures the dropped predictors remain strictly less correlated than the active ones and do not immediately re-enter.

**The successive difference property.** A key consequence of the Stagewise modification is that the coefficient vector evolves with **monotone sign consistency of successive differences**:

$$\text{sign}(\hat{\beta}_j^+ - \hat{\beta}_j) = s_j$$

whenever predictor $j$ is in the current active subset $\hat{\mathcal{B}}$. That is, coefficients always move **away from zero** along the direction of their current correlation sign. This is a stronger constraint than the Lasso's condition (coefficient sign equals correlation sign) because it governs changes, not just current values.

**Comparison of the three methods (Section 3.2):**

- **Stagewise:** $\text{sign}(\Delta \hat{\beta}_j) = \text{sign}(\hat{c}_j)$ — successive differences agree with current correlations. Coefficients move monotonically away from zero while active; reversals only possible when a predictor is "resting" between periods of activity (as variable 7 did in the right panel of Figure 1 between the 8th and 10th Stagewise-modified LARS steps).

- **Lasso:** $\text{sign}(\hat{\beta}_j) = \text{sign}(\hat{c}_j)$ — current coefficient signs match current correlation signs. Coefficients can change sign only after passing through zero, at which point the variable must be dropped from the active set (the Lasso modification rule).

- **LARS:** No sign restrictions. Coefficients can take any values consistent with the equiangular geometry.

From this perspective, the methods form a hierarchy of increasing constraint: LARS (unconstrained equiangular steps), Lasso (coefficient sign constraint), Stagewise (coefficient change sign constraint). This explains both why Lasso and Stagewise produce nearly identical paths in practice (the constraints differ only in their temporal scope) and why they can diverge (when the projection step forces Stagewise to drop variables that the Lasso would retain).

**Computational cost:** The Stagewise modification requires checking the sign of $w_{\mathcal{A}}$ components at each step and potentially dropping variables (using NNLS-type downdating). This can substantially increase the number of steps relative to pure LARS: for the 64-predictor quadratic model, Stagewise took 255 steps versus 64 for LARS and 103 for Lasso. The reason is that correlated predictors can enter and leave the active set multiple times as the algorithm navigates the faces of the convex cone.

---

#### The Degrees of Freedom and $C_p$ Criterion: Inference Without Extra Computation

Section 4 develops inferential tools specifically for LARS estimates (not for Lasso or Stagewise). The foundation is the degrees of freedom definition for a general estimator $\hat{\mu} = g(y)$ under the homoskedastic model $y \sim (\mu, \sigma^2 I)$:

$$\text{df}_{\mu,\sigma^2} = \sum_{i=1}^n \text{cov}(\hat{\mu}_i, y_i) / \sigma^2$$

This generalizes the linear-model definition $\text{df} = \text{trace}(M)$ for $\hat{\mu} = M y$, and leads to the $C_p$-type unbiased risk estimator (Equation 4.5):

$$C_p(\hat{\mu}) = \frac{\|y - \hat{\mu}\|^2}{\sigma^2} - n + 2 \text{df}_{\mu,\sigma^2}$$

**The simple approximation.** The paper's striking empirical finding (Figure 6, both panels) is that for the $k$-step LARS estimate $\hat{\mu}_k$:

$$\text{df}(\hat{\mu}_k) \doteq k$$

with the approximation holding within bootstrap confidence limits for both the 10-predictor and 64-predictor diabetes models. This leads to the practical $C_p$ formula (Equation 4.10):

$$C_p(\hat{\mu}_k) \doteq \|y - \hat{\mu}_k\|^2 / \bar{\sigma}^2 - n + 2k$$

**What this means operationally:** After running the full $m$-step LARS sequence, one can compute $C_p(\hat{\mu}_k)$ for each $k = 0, 1, \ldots, m$ using only quantities already computed during the forward pass ($\|y - \hat{\mu}_k\|^2$, the residual sum of squares at step $k$) plus an estimate $\bar{\sigma}^2$ of the error variance (typically from the full OLS model). The step $k$ with minimum $C_p$ is selected as the "best" model. This requires **no additional computation** beyond the original LARS fit—no cross-validation, no bootstrap, no separate model fitting.

**Why $k$ and not something else:** Theorem 3 proves that $\text{df}(\hat{\mu}_k) = k$ exactly when the predictors are mutually orthogonal. In this case, Lemma 1 (Section 4.1) shows that LARS reduces to soft-thresholding at the order statistics of the data: $\hat{\mu}_{k,i}(y) = \eta(y_i; |y|_{(k+1)})$ where $\eta$ is the soft-thresholding function. The divergence $\nabla \cdot \hat{\mu}_k = \sum_i \partial \hat{\mu}_{k,i} / \partial y_i = k$ almost everywhere, and Stein's formula (Equation 4.12) then yields $\text{df} = k$.

Theorem 4 extends this to the **Positive Cone Condition**: $X$ satisfies $\mathcal{G}_{\mathcal{A}}^{-1} 1_{\mathcal{A}} > 0$ (element-wise) for all subsets $\mathcal{A}$. Under this condition, the equiangular direction always lies within the convex cone, LARS/Lasso/Stagewise all coincide, and $\hat{\mu}_k$ is continuous and almost differentiable, allowing Stein's formula to apply. Lemma 2 proves the divergence formula $\nabla \cdot \hat{\mu}_k(y) = k$ on a set of full measure by showing that $u_k$ lies in a subspace $L_k$ of dimension $n - k + 1$, and that $\langle \nabla \gamma_l, u_l \rangle = 1$ and $\langle \nabla \gamma_l, u \rangle = 0$ for $u \in L_{l+1}$, so the total divergence sums to $k$.

**When the simple approximation is not exact:** For general $X$ not satisfying the Positive Cone Condition, Stein's formula may not be directly applicable because $\hat{\mu}_k$ can fail to be almost differentiable at points where multiple variables enter or leave simultaneously (the continuity assumptions underlying Equation 4.12 break down). However, the bootstrap evidence in Figure 6 and the delta-method argument (Equation 4.15: $\hat{\mu}_k$ is locally linear with matrix $M_k = P_k - \cot_k \cdot u_k v_k'$ having trace $k$) suggest the approximation remains good in practice. The paper notes that "it requires concerted effort at pathology to make $\text{df}(\hat{\mu}_k)$ much different than $k$."

**The Lasso degrees of freedom:** The simple approximation $df \doteq k$ does **not** apply to Lasso because the number of steps can exceed $m$ (12 for diabetes, 103 for the quadratic model) while the full model still has $m$ degrees of freedom. However, the paper reports an empirical finding: if $\ell(k)$ is the index of the last Lasso model containing exactly $k$ non-zero predictors, then $df(\hat{\mu}_{\ell(k)}) \doteq k$. That is, **degrees of freedom for Lasso is approximately the number of non-zero coefficients in the model**, regardless of how many steps were taken to get there. The paper states this without proof, noting "we do not yet have any mathematical support for this claim."

---

#### Computational Organization: Cholesky Updates and Overall Cost

Section 7 describes the computational strategy that achieves the paper's efficiency claims. The key observation is that the LARS algorithm can be organized around a **guided Cholesky factorization** of the Gram matrix $X'X$.

**Per-step operations.** At step $k$, the algorithm must:

1. Compute $m - k$ inner products $\hat{c}_j = x_j'(y - \hat{\mu}_{k-1})$ for inactive predictors $j \notin \mathcal{A}_{k-1}$ to identify the next active variable.
2. Invert the $k \times k$ Gram matrix $\mathcal{G}_k = X_k' X_k$ to compute the equiangular weights $w_k$ and direction $u_k$.
3. Compute the step size $\hat{\gamma}_k$ via Equation 2.13.

**Cholesky updating.** Rather than recomputing $\mathcal{G}_k^{-1}$ from scratch at each step, the algorithm maintains the Cholesky factorization $R_{k-1}$ of $\mathcal{G}_{k-1}$ (where $R_{k-1}$ is upper triangular and $\mathcal{G}_{k-1} = R_{k-1}' R_{k-1}$). When a new variable $x_{k}$ joins the active set, the Cholesky factor is **updated** by adding one row and column—a standard operation in numerical linear algebra (Golub & Van Loan, 1983) costing $O(k^2)$ operations. The total cost over $m$ steps is:

$$O(m^3 + n m^2)$$

**What this means:** The $O(m^3)$ term comes from the cumulative cost of Cholesky updates; the $O(n m^2)$ term comes from computing initial inner products and updating correlations. This is **the same order as a single least-squares fit** on all $m$ predictors, which requires $O(m^3 + n m^2)$ for the Cholesky decomposition of the full $m \times m$ Gram matrix plus computing $X'y$. At the final step $m$, the algorithm has computed $R = R_m$, the Cholesky factor for the full cross-product matrix—exactly what a standard OLS computation would produce.

**The Lasso modification cost.** When a variable is dropped (the $\tilde{\gamma} < \hat{\gamma}$ case), the Cholesky factor must be **downdated**—removing a row and column. This costs $O(m^2)$ per downdate. Since the Lasso modification typically adds only a few extra steps relative to pure LARS (12 vs. 10 for diabetes, 103 vs. 64 for the quadratic model), the total cost remains $O(m^3 + n m^2)$ with a modest constant-factor increase.

**The Stagewise modification cost.** The Stagewise modification can be more expensive because the NNLS-type inner loop (checking and enforcing non-negativity of $w_{\mathcal{A}}$) may drop multiple variables at once and cause more frequent active set changes. For the 64-predictor quadratic model, Stagewise required 255 steps—roughly 4 times the pure LARS count and 2.5 times the Lasso count. The paper notes that "with many correlated variables, the stagewise version can take many more steps... increasing the computations by factors up to 5 or more in extreme cases." The per-step cost remains low (Cholesky update/downdate), but the increased number of steps drives up the total.

**The $m \gg n$ case.** When there are more predictors than observations ($m > n$), LARS terminates at the saturated least-squares fit after at most $n - 1$ variables have entered the active set. (The rank is $n - 1$ rather than $n$ because the predictors are mean-centered, removing one degree of freedom.) The total cost is $O(n^3)$. The paper notes that the simple $C_p$ approximation has not been investigated for $m > n$, and that near the saturated end, the model sequence "tends to be quite variable with respect to small changes in $y$."

**Inner product updates.** For efficiency, the inner products $\hat{c}_j$ can be updated at each step using the cross-product matrix $X'X$ rather than recomputed from scratch. However, this strategy is only beneficial when $m \ll n$; for $m \gg n$, maintaining the full $m \times m$ cross-product matrix would be more expensive than direct computation, so the algorithm switches to working directly with the $n \times m$ design matrix.

---

#### The Homotopy Connection (Noting Prior Work)

The paper acknowledges (Section 3.1, 5) that the LARS/Lasso connection "closely parallels the homotopy method in the papers by Osborne, Presnell, and Turlach (2000a, 2000b)." The homotopy method tracks the Lasso solution path by following the active set of non-zero coefficients and solving a sequence of linear systems, exploiting the piecewise linearity of the path. The paper's contribution is to recast this optimization-theoretic approach in the language of regression and correlation ("we will stick to the language of regression and correlation rather than convex optimization"), making it accessible to a statistical audience while providing new geometric insights (the convex cone projection for Stagewise, the degrees-of-freedom analysis, the Cp criterion) that go beyond the homotopy literature.

## 4. Key Insights and Innovations

### Innovation 1: The Equiangular Strategy as a New Middle Ground Between Greedy Selection and Full Regularization

The central conceptual move of this paper is introducing the **least angle direction** as a principled compromise between the extremes of Forward Selection (which commits fully to one predictor at each step) and methods like the Lasso or ridge regression (which solve a global optimization problem). Before LARS, the field understood forward stepwise procedures as inherently discrete and potentially reckless—they make irreversible "winner-take-all" decisions that can lock out correlated predictors prematurely. The Lasso, by contrast, was understood as a convex optimization problem solved by quadratic programming, with no obvious connection to forward greedy search.

LARS reinterprets model building as a **continuous, piecewise-linear navigation** through predictor space governed by a single geometric principle: at each step, move in the direction that makes equal angles with all currently active predictors, and stop exactly when a new predictor achieves equal correlation with the residual. This is neither the "all-in" commitment of Forward Selection (which would move all the way to the OLS projection, making $\rho_k = 1$) nor the infinitesimal caution of Stagewise (which takes tiny steps requiring thousands of iterations). It is an exact analytical middle ground—the step size computation in Equation 2.13 solves for the precise moment when the correlation equilibrium breaks.

What makes this distinctive is that **the equiangular principle is not derived from an optimization criterion**—it is not minimizing any global objective function. It is a geometric rule for navigating predictor space that happens to coincide with the Lasso path under a sign constraint and with the Stagewise limit under a convex cone projection. The paper is essentially saying: if you want to build a model cautiously but efficiently, the natural thing to do is follow the bisector of the active predictors and adjust step sizes to maintain correlation balance. This reframes model selection from an optimization problem (minimize error subject to constraint) to a **mechanical procedure** with intelligible geometry—a conceptual shift that makes the entire Lasso/Stagewise spectrum accessible to practitioners who think in terms of correlations and forward search rather than Lagrange multipliers and KKT conditions.

The significance of this reframing is evident in the simulation study (Figure 5): LARS achieves a proportion explained of 0.963 at $k = 10$, nearly matching the ideal value of 1.0 for the true model, while Forward Selection peaks lower (0.950) and declines more steeply. The equiangular strategy is not just faster than the Lasso—it is **statistically more prudent than Forward Selection** because it never orthogonalizes the residual completely with respect to any single predictor, leaving room for correlated alternatives to enter later.

---

### Innovation 2: The Geometric Unification of Three Seemingly Unrelated Methods

Prior to this paper, the Lasso (Tibshirani, 1996) and Forward Stagewise regression were understood as entirely different approaches to model selection—one a convex optimization with an $\ell_1$ penalty, the other an iterative greedy algorithm with infinitesimal steps. The empirical observation that their coefficient paths were nearly identical (right panel of Figure 1 versus left panel) was a puzzle without a theoretical explanation. Hastie et al. (2001) had noted the similarity and conjectured a connection, but no one had demonstrated **why** two procedures with such different definitions should produce essentially the same answer.

The paper's resolution of this puzzle is a genuine theoretical advance. By showing that both the Lasso and Stagewise are **constrained versions of the same equiangular LARS strategy**, the paper reveals a hierarchy of constraint severity that was completely invisible before:

- **LARS**: unconstrained equiangular steps. Any predictor weights are allowed in the equiangular direction.
- **Lasso**: the equiangular step is halted if it would cause a coefficient sign reversal (Equation 3.6). The constraint is on $\text{sign}(\hat{\beta}_j)$ matching $\text{sign}(\hat{c}_j)$—a "state" constraint on the current coefficient values.
- **Stagewise**: the equiangular direction itself is modified if it would require moving against the current correlation signs. The constraint is on $\text{sign}(\Delta \hat{\beta}_j)$ matching $\text{sign}(\hat{c}_j)$—a "flow" constraint on the direction of coefficient changes.

These three constraints form a natural progression: Stagewise is the most cautious (it won't even *move* in a direction that reduces any active coefficient), Lasso is intermediate (it will move but must *stop* before a coefficient crosses zero), and LARS is the most aggressive (no sign restrictions at all). This ordering explains both why Lasso and Stagewise are nearly identical in practice—their constraints differ only in temporal scope, and violation of the Lasso constraint typically implies violation of the Stagewise constraint a short time later—and why they can diverge when the geometry forces it, as at the arrowed point in Figure 1 where the Stagewise modification drops variables 3 and 7 from the active set while the Lasso drops only variable 7.

This unification is more than a taxonomic exercise. It provides a **generative explanation** for the empirical success of both methods: they work because they approximate the equiangular ideal, not because of any special property of $\ell_1$ penalties or infinitesimal step sizes. The $\ell_1$ penalty matters only insofar as it enforces a sign-consistency condition that keeps the Lasso close to the LARS path. The Stagewise infinitesimal limit matters only insofar as it converges to a convex cone projection that, in most cases, stays close to the unconstrained equiangular direction. The fundamental driver of statistical performance is the **equiangular compromise among correlated predictors**—everything else is constraint details.

This reframing has implications beyond linear regression. As the paper notes in Section 8, least-squares boosting with trees is structurally identical to Forward Stagewise regression with an infinite predictor set. If Stagewise approximates LARS in the linear case, and LARS approximates the Lasso, then **boosting may be implicitly performing a Lasso-like regularization on the space of all possible trees**—a conjecture that the paper explicitly states and that has influenced subsequent research on the statistical properties of boosting.

---

### Innovation 3: Degrees of Freedom as a Diagnostic Tool for Adaptive Model Selection (Not Just a Computational Shortcut)

The degrees-of-freedom analysis in Section 4 is easy to overlook as "just" a model selection criterion, but it represents a subtle and important conceptual contribution that distinguishes LARS from being merely a faster way to compute Lasso solutions. The paper does not simply provide a $C_p$ formula—it develops a **framework for thinking about the complexity of adaptively selected models** that connects to Stein's unbiased risk estimation (SURE) theory and the geometry of the LARS path.

The key move is defining degrees of freedom through the covariance-based formula (Equation 4.4), $\text{df}_{\mu,\sigma^2} = \sum_{i=1}^n \text{cov}(\hat{\mu}_i, y_i) / \sigma^2$, rather than through model size or number of parameters. This definition, drawn from Efron (1986) and Efron & Tibshirani (1997), captures the **effective complexity** of an estimator—how much the fitted values $\hat{\mu}_i$ adapt to the specific realization of $y_i$. For a linear estimator $\hat{\mu} = M y$, this reduces to $\text{trace}(M)$, which for OLS with $k$ preselected predictors is exactly $k$. But for an adaptive procedure like LARS—where the selected predictors at step $k$ depend on which correlations happen to be largest in this particular dataset—the effective degrees of freedom could be larger than $k$ because the selection step itself consumes degrees of freedom.

The paper's finding that $\text{df}(\hat{\mu}_k) \doteq k$ is therefore **not obvious**. It says that LARS's adaptive model selection is, in terms of overfitting cost, roughly equivalent to having prespecified which $k$ predictors to use. This is a strong statement about the statistical efficiency of the equiangular strategy: the data-dependent choice of which predictors enter at steps 1 through $k$ does not, on average, inflate the model's complexity beyond $k$.

The theoretical support for this claim is carefully bounded. Theorem 3 proves exact equality $\text{df} = k$ for orthogonal designs—a case where LARS reduces to soft-thresholding at order statistics, and the divergence $\nabla \cdot \hat{\mu}_k = k$ can be verified directly. Theorem 4 extends this to designs satisfying the Positive Cone Condition, using Stein's lemma to convert the covariance sum to an expected divergence and Lemma 2 to show the divergence equals $k$ almost everywhere. These are non-trivial applications of Stein's SURE theory to a non-smooth estimator, requiring careful treatment of differentiability at points where the active set changes (Lemmas 13–17 in the Appendix). The paper acknowledges that for general $X$, Stein's formula may not apply because $\hat{\mu}_k$ can fail to be almost differentiable at multiple points, but the bootstrap evidence (Figure 6) and the delta-method argument (Equation 4.15) suggest the approximation is robust.

The practical consequence—the $C_p$ formula $C_p(\hat{\mu}_k) \doteq \|y - \hat{\mu}_k\|^2 / \bar{\sigma}^2 - n + 2k$ (Equation 4.10)—gives LARS standalone value as a model selection tool independent of the Lasso or Stagewise. A practitioner can run LARS (always exactly $m$ steps), compute $C_p$ at each step using quantities already available from the forward pass, and select the optimal model size with **no additional computation**. This is a significant practical advantage over cross-validation or bootstrap methods, and it does not carry over to the Lasso or Stagewise—their $C_p$ formulas would be different because their paths involve more steps than there are predictors, and the simple $k$ approximation no longer holds (though the paper reports empirically that Lasso degrees of freedom approximately equals the number of non-zero coefficients).

Figure 7 demonstrates the criterion in action: for the 10-predictor diabetes model, minimum $C_p$ occurs at $k = 7$, and for the 64-predictor quadratic model, at $k = 16$. Both selections "looked sensible, their first several selections of 'important' covariates agreeing with an earlier model based on a detailed inspection of the data assisted by medical expertise." This validation against domain knowledge—not just against held-out prediction error—strengthens the case that LARS with $C_p$ selects scientifically meaningful models, not just statistically predictive ones.

---

### Innovation 4: The Convex Cone Projection as a Bridge Between Discrete and Continuous Model Search

The Stagewise modification of LARS (Section 3.2, formalized in Section 6) introduces a geometric operation—projection of the equiangular vector onto the convex cone generated by the active predictors—that has no obvious precedent in the model selection literature. This operation resolves a subtle but important conceptual tension: what does it mean for a forward greedy algorithm to take the limit as step size goes to zero when multiple predictors are tied for maximum correlation?

Before LARS, the answer was not clear. The Stagewise algorithm (Equation 1.7) is defined operationally: at each iteration, take a small step toward the single most correlated predictor. When multiple predictors are exactly tied, the algorithm must break ties arbitrarily (or randomly), producing a staircase path whose limiting behavior depends on the tie-breaking sequence. The paper's insight is that the **limit of infinitesimal Stagewise steps is not the equiangular LARS direction in general**, but rather the projection of that direction onto the convex cone $\mathcal{C}_{\mathcal{A}}$ (Equation 3.12) of non-negative combinations of the signed active predictors.

This is a fundamentally new way to think about forward greedy search in the continuous limit. It reveals that the Stagewise procedure has an implicit constraint—it can only move in directions that increase (or at least do not decrease) the contribution of each active predictor—and that this constraint survives the infinitesimal limit. When the unconstrained equiangular direction $u_{\mathcal{A}}$ lies within $\mathcal{C}_{\mathcal{A}}$, all three methods (LARS, Lasso, Stagewise) coincide. When it does not, Stagewise "projects" onto the nearest feasible direction, which necessarily lies on a lower-dimensional face of the cone—meaning Stagewise drops some predictors from the active set that LARS would retain.

Lemma 12 proves that this projection is unique and satisfies three geometric constraints that characterize the Stagewise limit: the direction must be a non-negative combination of active predictors (Constraint I), must maintain equal correlation decline rates for the retained predictors (Constraint II), and must reduce correlations for dropped predictors faster than for retained ones (Constraint III). These constraints are not imposed by fiat—they emerge from the requirement that the continuous limit of the discrete Stagewise procedure be well-defined.

The conceptual significance of this insight extends beyond linear regression to boosting (Section 8). In least-squares boosting with trees, the "predictors" are an infinite set of regression trees, and the convex cone $\mathcal{C}_{\mathcal{A}}$ becomes the cone of non-negative combinations of the currently active trees. The projection operation suggests that the boosting path may not follow the "full" equiangular direction (which would require weights on all trees, possibly negative) but instead restricts to non-negative tree combinations—a constraint that is naturally satisfied by standard boosting algorithms. The paper's closing suggestion of a modified boosting procedure that "take[s] a small least squares step in all trees currently in our model" rather than only the most correlated tree is a direct consequence of this geometric understanding. Such a procedure would approximate LARS in the tree space and, by the LARS/Lasso connection, might inherit Lasso-like regularization properties—a hypothesis that connects the paper's linear-model geometry to the practical performance of one of machine learning's most successful methods.

## 5. Experimental Analysis

### Evaluation Methodology

- **Dataset.** The paper uses two primary datasets, both derived from the diabetes study introduced in Table 1 (Efron et al., 2003). The first is the original 10-predictor dataset: $n = 442$ diabetes patients measured on $m = 10$ baseline variables (age, sex, BMI, average blood pressure, and six blood serum measurements), with response variable $y$ being a quantitative measure of disease progression one year after baseline. All analyses use the same 442 observations. The second is a **quadratic model** expansion of the same data (Equation 3.15): 10 main effects, all 45 two-way interactions, and 9 squares (all covariates except the dichotomous variable $x_2$, sex), yielding $m = 64$ predictors from the same $n = 442$ observations. Both datasets are standardized per Equation 1.1: each predictor is centered to mean 0 and scaled to unit length; the response is centered to mean 0.

- **Base model(s).** The "base model" in every experiment is ordinary least squares (OLS) applied to subsets of the standardized predictors. No pretrained neural models are involved—this is a classical linear regression paper. The LARS algorithm itself takes as input the $n \times m$ design matrix $X$ and response vector $y$, and produces a sequence of coefficient estimates $\hat{\beta}_k$ for $k = 0, 1, \ldots, m$. The full OLS model on all $m$ predictors ($m = 10$ or $m = 64$) serves as the terminal point of every algorithm path ($\hat{\beta}_m = \hat{\beta}^{\text{OLS}}$).

- **Metrics.** Three distinct metrics are used across different experimental sections:
  - **Residual sum of squares** $S(\hat{\beta}) = \|y - X\hat{\beta}\|^2$: the standard OLS error criterion, used to track how fit improves as more predictors enter.
  - **Proportion explained** (Equation 3.17): $\text{pe}(\hat{\mu}) = 1 - \|\hat{\mu} - \mu\|^2 / \|\mu\|^2$, where $\mu$ is the true mean vector in the simulation study. $\text{pe}(0) = 0$ and $\text{pe}(\mu) = 1$, so values near 1 indicate the estimate recovers the true signal. This is the primary metric in the simulation comparison (Section 3.3, Figure 5).
  - **$C_p$ statistic** (Equation 4.10): $C_p(\hat{\mu}_k) \doteq \|y - \hat{\mu}_k\|^2 / \bar{\sigma}^2 - n + 2k$, used as an unbiased estimator of prediction risk for selecting the optimal LARS step $k$. Lower $C_p$ indicates better expected prediction performance.
  - **Degrees of freedom** (Equation 4.4): $\text{df}_{\mu,\sigma^2} = \sum_{i=1}^n \text{cov}(\hat{\mu}_i, y_i) / \sigma^2$, estimated via parametric bootstrap (Equations 4.6–4.8) with $B = 500$ replications, using $\bar{\mu}$ and $\bar{\sigma}^2$ from the full OLS model as the data-generating parameters. Bootstrap samples are drawn as $y^* \sim N(\bar{\mu}, \bar{\sigma}^2)$.

- **Baselines.** The paper compares against four established methods:
  - **Classic Forward Selection** (Weisberg, 1980; Section 1): at each step, selects the predictor with largest absolute correlation with the current residual, then performs full OLS regression on all selected predictors, orthogonalizing the residual completely with respect to the chosen variable. This serves as the "overly greedy" baseline throughout.
  - **Full OLS** (all $m$ predictors): the unregularized least-squares fit, representing the maximum-complexity endpoint of every algorithm path. Provides $\bar{\mu}$ and $\bar{\sigma}^2$ for bootstrap and $C_p$ computations.
  - **The Lasso** (Tibshirani, 1996): computed via the LARS modification (Section 3.1, Theorem 1), producing the full regularization path. In the simulation study, Lasso is run as a separate algorithm for comparison with LARS and Stagewise.
  - **Forward Stagewise Linear Regression** (Section 1, Equations 1.6–1.7): the iterative small-step procedure with 6,000 steps for the 10-predictor case (Figure 1, right panel). In the simulation, Stagewise is run via the LARS-Stagewise modification (Section 3.2, Theorem 2).

- **Generation budget / compute accounting.** Compute is measured in two complementary ways:
  - **Number of algorithm steps** ($k$ for LARS, variable for Lasso and Stagewise): LARS always takes exactly $m$ steps to reach the full OLS solution. Lasso takes more ($m = 10$ requires 12 steps; $m = 64$ requires 103 steps) because of variable removals. Stagewise takes the most ($m = 10$ requires 13 modified LARS steps; $m = 64$ requires 255 steps). Step count is reported explicitly for every experiment.
  - **Total floating-point operations** (Section 7): the asymptotic cost of LARS is $O(m^3 + n m^2)$, the same order as a single full OLS fit. The Lasso modification adds $O(m^2)$ per variable downdate; Stagewise adds more due to increased step count (up to 5× or more in extreme cases). The paper does not report wall-clock times or FLOP counts for individual experiments, relying on the asymptotic analysis in Section 7.

- **Cross-validation / statistical protocol.**
  - **Simulation study (Section 3.3)**: 100 simulated response vectors $y^*$ are generated from the model $y^* = \mu + \epsilon^*$, where $\mu = X\beta$ is the true mean (obtained by running LARS for 10 steps on the original diabetes data) and $\epsilon^*$ is a bootstrap sample (with replacement) from the residuals $\epsilon = y - \mu$ of the original fit. The "true $R^2$" for this model is $\|\mu\|^2 / (\|\mu\|^2 + \|\epsilon\|^2) = 0.416$. For each simulated dataset, LARS, Lasso, and Stagewise are run, and proportion explained $\text{pe}(\hat{\mu}_k^*)$ is computed at each step. Averages and standard deviations over the 100 replications are reported in Figure 5.
  - **Bootstrap degrees of freedom (Section 4, Figure 6)**: $B = 500$ parametric bootstrap replications, with $y^* \sim N(\bar{\mu}, \bar{\sigma}^2)$ where $\bar{\mu}$ and $\bar{\sigma}^2$ are from the full OLS model. The 500 replications are divided into 10 groups of 50 to compute student-t confidence intervals for $\hat{\text{df}}_k$. The covariance $\widehat{\text{cov}}_i$ is estimated via Equation 4.7; $\hat{\text{df}}$ is computed via Equation 4.8.
  - **$C_p$ model selection (Section 4, Figure 7)**: $C_p(\hat{\mu}_k)$ is computed at each step $k$ using Equation 4.10, with $\bar{\sigma}^2$ from the full OLS model. The step achieving minimum $C_p$ is selected; no separate train/test split or cross-validation is used—$C_p$ itself is the selection criterion.

### Main Quantitative Results

#### Coefficient Paths: The Near-Identity of Lasso, Stagewise, and LARS

Figure 1 (left panel) displays the full Lasso solution path $\hat{\beta}_j(t)$ for the 10-predictor diabetes data, plotted against $t = \sum |\hat{\beta}_j|$. As $t$ increases from 0 to 3460.00 (where the constraint ceases to bind and the solution equals full OLS), the coefficients evolve piecewise linearly. Variables enter the model sequentially in order: 3, 9, 4, 7, 2, 10, 5, 8, 6, 1. At $t = 1000$, only variables 3, 9, 4, and 7 have non-zero coefficients. Shrinkage toward zero is evident for all coefficients at small $t$, with some coefficients (e.g., variable 8) showing non-monotonic paths that increase after initially being suppressed.

Figure 1 (right panel) displays the Forward Stagewise path for the same data, computed from 6,000 Stagewise steps (Equation 1.7) with $\epsilon$ small enough to conceal the discrete staircase. The paths are "nearly, but not exactly, identical" to the Lasso paths. Variable 8's track differs noticeably at large $t$, as highlighted by the authors.

Figure 3 (left panel) shows the pure LARS coefficient paths for the same data, computed in exactly $m = 10$ steps. The tracks are "nearly but not exactly the same as either the Lasso or Stagewise tracks." Variables enter in the identical order: 3, 9, 4, 7, 2, 10, 5, 8, 6, 1.

The step-efficiency comparison is stark: LARS reaches the full solution in 10 steps, Lasso in 12 (due to one variable removal at the arrowed point in Figure 1, left panel, where variable 7 was briefly removed and then restored), and Stagewise requires 6,000 original steps (reducible to 13 modified LARS steps via the Stagewise modification of Section 3.2).

Figure 3 (right panel) plots the absolute current correlations $|\hat{c}_{kj}| = |x_j'(y - \hat{\mu}_{k-1})|$ for each variable $j = 1, 2, \ldots, 10$ as a function of the LARS step $k$. The heavy curve shows the maximum absolute correlation $\hat{C}_k$, which declines monotonically from approximately 20,000 at $k = 1$ to near 0 at $k = 10$. At each step, a new variable joins the active set and its correlation merges with the declining maximum curve. Variables that have not yet entered (e.g., variable 1, which enters last) have correlations that initially trail below the maximum and then catch up when it is their turn.

#### Simulation Comparison: LARS, Lasso, and Stagewise Perform Nearly Identically

Figure 5 presents the central simulation result comparing LARS, Lasso, Stagewise, and Forward Selection on the 64-predictor quadratic model, averaged over 100 replications.

**Headline result.** All three methods—LARS, Lasso, and Stagewise—perform "almost identically." The proportion explained $\text{pe}(\hat{\mu})$ (Equation 3.17) rises quickly, reaching a maximum of **0.963 at $k = 10$** for LARS (the solid curve), and then declines slowly as $k$ grows to 40. The Lasso and Stagewise curves (dotted and dashed) are nearly superimposed on the LARS curve when plotted against the average number of non-zero $\hat{\beta}_j^*$ terms. At the 40th step, Stagewise averages 33.23 non-zero terms, Lasso averages 35.83, and LARS (which always keeps all previously entered variables) averages 40. Small dots indicate the standard deviation over the 100 simulations as roughly ±0.02.

**Comparison with Forward Selection.** The dashed curve for classic Forward Selection rises very quickly—reaching a maximum of **0.950 after only $k = 3$ steps**—and then "falls back more abruptly than the LARS/Lasso/Stagewise curves." The rapid initial rise confirms Forward Selection's greediness: it achieves good fit with very few predictors by committing fully to the strongest signals. The steeper subsequent decline confirms the paper's characterization of Forward Selection as "a dangerously greedy algorithm": its early irreversible choices lock out useful predictors and cause worse performance at moderate model sizes compared to the more cautious equiangular methods.

**Interpretation of the peak-and-decline pattern.** All methods exhibit a rise to a maximum followed by decline as model complexity increases. This is the classic bias-variance tradeoff: early steps add genuine signal (increasing proportion explained), while later steps add noise variables that inflate variance without improving the fit to the true mean $\mu$. The fact that LARS/Lasso/Stagewise peak higher (0.963 vs. 0.950) and decline more slowly than Forward Selection indicates that the equiangular strategy makes better choices about which predictors to add in the critical intermediate steps (roughly $k = 3$ to $k = 15$) where Forward Selection has already committed to potentially suboptimal predictors.

**Stopping rule implications.** The simulation shows that any stopping point between $k = 5$ and 25 "typically gave a $\hat{\mu}(k)^*$ with true predictive $R^2$ about 0.40, compared to the ideal value 0.416 for $\mu$." This flat region around the optimum means that the exact choice of $k$ is not critical—a range of model sizes produce nearly equivalent prediction accuracy—which bodes well for the practical use of $C_p$ or other selection criteria that might not pinpoint the exact optimum.

#### Degrees of Freedom: The Simple Approximation Holds Empirically

Figure 6 presents bootstrap estimates of $\text{df}(\hat{\mu}_k)$ for LARS estimates, with $B = 500$ replications and 95% confidence intervals (dashed lines) computed from 10 groups of 50 replications each.

**Left panel: 10-predictor diabetes model.** The bootstrap estimates track the line $df = k$ (solid line) almost perfectly for $k = 1, 2, \ldots, 10$. The confidence intervals are narrow (roughly ±0.5 at intermediate $k$) and consistently contain the value $k$. This provides strong empirical support for the simple approximation (Equation 4.9) in a realistic non-orthogonal setting.

**Right panel: 64-predictor quadratic model.** Despite the much larger predictor set, the bootstrap estimates again closely follow $df = k$ for $k = 1, 2, \ldots, 64$. The confidence intervals widen at larger $k$ (roughly ±2 at $k = 60$) but consistently contain the identity line. This is the more surprising result, because the 64-predictor quadratic model with interactions and squares introduces substantial multicollinearity—exactly the conditions under which one might expect the Positive Cone Condition (Equation 4.11) to fail and the simple approximation to break down. Yet the bootstrap evidence suggests $df \doteq k$ remains accurate.

The paper notes that "it requires concerted effort at pathology to make $\text{df}(\hat{\mu}_k)$ much different than $k$," though it does not construct or test such pathological cases. The theoretical results (Theorems 3 and 4) cover orthogonal designs and the Positive Cone Condition, but the empirical evidence suggests the approximation extends well beyond these cases.

#### $C_p$ Model Selection: Sensible Models with No Extra Computation

Figure 7 applies the $C_p$ formula (Equation 4.10) to select the optimal LARS step for both models.

**Left panel: 10-predictor diabetes model.** $C_p(\hat{\mu}_k)$ achieves its minimum at **$k = 7$**. The curve is U-shaped: $C_p$ starts high at $k = 4$ (approximately 25), drops sharply to its minimum at $k = 7$ (approximately 5), and then rises gradually to roughly 10 at $k = 10$. The minimum at 7 predictors is noteworthy because it excludes 3 of the 10 available variables while retaining those that the paper notes "agreed with an earlier model based on a detailed inspection of the data assisted by medical expertise."

**Right panel: 64-predictor quadratic model.** $C_p(\hat{\mu}_k)$ achieves its minimum at **$k = 16$**. The curve declines from roughly 50 at $k = 0$ to a minimum near 20 at $k = 16$, then rises gradually to approximately 40 at $k = 64$. The model with 16 predictors represents substantial parsimony relative to the 64 available, and again the selected variables' "first several selections of 'important' covariates" matched domain-expert models.

**Computational cost of model selection.** The $C_p$ values at all $k$ are computed from quantities already available from the LARS forward pass: $\|y - \hat{\mu}_k\|^2$ is the residual sum of squares at step $k$, which must be tracked anyway, and $\bar{\sigma}^2$ is a single scalar from the full OLS fit. No additional model fitting, cross-validation, or bootstrap is required. This is a genuine practical advance over Lasso model selection, which would require additional computation because the $C_p$ formula does not simplify to $k$ degrees of freedom.

#### The $(T, S)$ Curve for Lasso: Convex Quadratic Spline

Figure 8 plots the residual sum of squares $S = \|y - \hat{\mu}\|^2$ versus $T = \sum |\hat{\beta}_j|$ for the Lasso path applied to the diabetes data. The 12 modified LARS steps are indicated as points along the curve. The triangle marks the boundary point at $t = 1000$ (corresponding to the model with variables 3, 9, 4, and 7 non-zero). The dashed arrow indicates the tangent at $t = 1000$, with negative slope $R_t$ (Equation 5.31).

**Key properties.** The curve is decreasing (adding predictors always reduces residual error), convex (the marginal reduction in error per unit increase in $T$ diminishes), and piecewise quadratic (it is a quadratic spline with $\dot{S}(T) = -2\hat{C}(T)$ and $\ddot{S}(T) = 2A_{\mathcal{A}}^2$, where $\mathcal{A}$ is the current active set). The convexity confirms that the Lasso path does not exhibit the kind of "overshooting" that Forward Selection can produce—each step optimally trades off bias reduction against coefficient inflation.

### Ablation Studies and Robustness Checks

The paper's "ablation" structure differs from modern ML papers—rather than systematically removing components and measuring performance degradation, the paper proves formal theorems and provides empirical confirmation through separate analyses. The closest analogues to ablation studies are:

**LARS vs. Lasso modification (Figures 1, 3, and the arrowed point in Figure 1, left panel).** The difference between pure LARS (Figure 3) and Lasso-modified LARS (Figure 1, left) is entirely attributable to the single Lasso modification rule (Equation 3.6): stopping an equiangular step when a coefficient would cross zero and removing that variable from the active set. In the diabetes example, this modification activated exactly once—at the arrowed point in Figure 1 (left panel), where variable 7 was dropped from the active set (which then contained all 10 indices) and restored one step later. The consequence: LARS took 10 steps; Lasso took 12. The coefficient tracks are nearly identical except for variable 8, whose path is noticeably affected by variable 7's temporary absence. This demonstrates that the Lasso modification is a minor perturbation of the LARS path in practice—it matters only when the equiangular geometry forces a coefficient sign violation, which is rare for this dataset.

**LARS vs. Stagewise modification (Figure 1, right panel, arrowed point).** The Stagewise modification (Section 3.2) was activated at the arrowed point in the right panel of Figure 1: the active set $\mathcal{A} = \{3, 9, 4, 7, 2, 10, 5, 8\}$ was reduced to $\hat{\mathcal{B}} = \mathcal{A} - \{3, 7\}$ because the equiangular weight vector $w_{\mathcal{A}}$ had negative components for variables 3 and 7, forcing a projection onto the convex cone $\mathcal{C}_{\mathcal{A}}$. This is a more aggressive intervention than the Lasso modification: Stagewise drops two variables where Lasso dropped one, and the removed variables are different (Stagewise drops 3 and 7; Lasso drops only 7). The total steps: LARS 10, Lasso 12, Stagewise 13. This confirms that the constraint hierarchy (LARS: no restrictions, Lasso: coefficient sign, Stagewise: coefficient change sign) produces increasingly cautious behavior with more frequent active set reductions.

**Forward Selection as an extreme ablation.** The simulation study (Figure 5) effectively treats Forward Selection as an "ablated" version of LARS where the equiangular compromise is replaced by full commitment ($\rho_k = 1$ instead of $\rho_k < 1$). The result—Forward Selection peaks lower (0.950 vs. 0.963) and declines faster—quantifies the cost of greediness. The equiangular strategy's 1.3 percentage point advantage in proportion explained at the peak, combined with substantially better performance at moderate model sizes (the region where Forward Selection has already declined but LARS remains near its peak), demonstrates that the partial-step mechanism (Equation 2.22: $\hat{\mu}_k - \hat{\mu}_{k-1} = \frac{\hat{\gamma}_k}{\bar{\gamma}_k} (\bar{y}_k - \hat{\mu}_{k-1})$) is not merely a computational convenience but a statistically meaningful improvement.

**Orthogonal design: exact solution (Lemma 1, Section 4.1).** In the orthogonal case ($x_j = e_j$, standard basis vectors), LARS reduces exactly to soft-thresholding at the order statistics of $|y_j|$: $\hat{\mu}_{k,i}(y) = \eta(y_i; |y|_{(k+1)})$ where $\eta$ is the soft-threshold operator. This special case provides both an existence proof that $df = k$ can hold exactly (Theorem 3) and a sanity check that LARS reduces to a known optimal procedure (soft-thresholding) when predictors are uncorrelated. The proof of Lemma 1 walks through the LARS mechanics explicitly, showing that $\hat{\gamma}_k = |y|_{(k)} - |y|_{(k+1)}$ and $u_k = |\mathcal{A}_k|^{-1/2} 1_{\mathcal{A}_k}$, confirming that the equiangular machinery produces the expected soft-thresholding solution.

**Positive Cone Condition: theoretical sufficient condition for $df = k$ (Theorem 4, Lemmas 15–17).** Theorem 4 proves that $\text{df}(\hat{\mu}_k) = k$ holds exactly when the Positive Cone Condition (Equation 4.11: $\mathcal{G}_{\mathcal{A}}^{-1} 1_{\mathcal{A}} > 0$ element-wise for all subsets $\mathcal{A}$) is satisfied. Lemma 15 derives the condition $x_+' u_{\mathcal{A}} < A_{\mathcal{A}}$ for a new predictor $x_+$ joining the active set under the Positive Cone Condition—a geometric inequality that ensures the equiangular direction always lies within the convex cone. Lemmas 16–17 establish that $\hat{\mu}_k$ is continuous and almost differentiable under this condition, validating the use of Stein's formula. The diabetes data does **not** satisfy the Positive Cone Condition (as evidenced by the Lasso and Stagewise modifications being triggered), yet the bootstrap evidence (Figure 6) still supports $df \doteq k$, suggesting the condition is sufficient but far from necessary.

**Bootstrap vs. delta-method for degrees of freedom (Equations 4.6–4.8 vs. 4.13–4.15).** The paper provides two independent justifications for $df \doteq k$. The bootstrap (Equations 4.6–4.8) is the empirical gold standard, directly estimating $\sum \text{cov}(\hat{\mu}_i, y_i)$ from parametric resampling. The delta method (Equation 4.15) provides an asymptotic justification: locally, $\hat{\mu}_k$ behaves as a linear estimator with matrix $M_k = P_k - \cot_k \cdot u_k v_k'$, whose trace is $\text{trace}(P_k) = k$. That both approaches agree (the delta method predicts $df = k$; the bootstrap confirms it) strengthens the result. The paper also notes that nearly identical results were obtained using residual resampling ($y^* = \bar{\mu} + e^*$ with $e^*$ resampled from $e = y - \bar{\mu}$) rather than the normal model, confirming robustness to the distributional assumption.

### Critical Assessment

The experiments in this paper serve a fundamentally different purpose from those in modern ML papers. There is no held-out test set, no cross-validation for hyperparameter selection, and no comparison against a large suite of competing algorithms. The experiments are **demonstrative rather than evaluative**: they illustrate the geometric and statistical properties derived in the theorems, using a single dataset (the diabetes study) as a running example. This reflects the paper's nature as a **methodological contribution with proofs**, where the primary claims are mathematical (Theorems 1–4) and the experiments provide empirical confirmation and practical guidance. This assessment must therefore evaluate the experiments on their own terms: do they convincingly support the paper's stated contributions?

#### On the Claim That LARS Reduces Lasso Computation by an Order of Magnitude

The claim is that the LARS modification "calculates all possible Lasso estimates for a given problem, using an order of magnitude less computer time than previous methods" (Abstract). The experimental evidence for this is **indirect**. The paper shows that the Lasso solution path for the diabetes data requires 12 modified LARS steps (Figure 1, left panel) compared to 6,000 Stagewise steps for the unmodified Stagewise procedure. Section 7 states the asymptotic cost: $O(m^3 + n m^2)$ for the full LARS path, with Lasso modifications adding only $O(m^2)$ per variable downdate. This is "the same order of magnitude of computational effort as ordinary least squares applied to the full set of covariates."

**What is demonstrated:** The step-count reduction relative to Stagewise (12 vs. 6,000) and Forward Selection (not directly compared) is shown. The $O(m^3 + nm^2)$ complexity bound is stated and justified.

**What is not demonstrated:** No wall-clock timing comparisons are provided. No comparison is made against Osborne et al.'s (2000a) homotopy method, which the paper acknowledges as a closely related prior approach. The claim of "an order of magnitude less computer time than previous methods" is not backed by runtime measurements against any specific prior Lasso implementation (e.g., quadratic programming solvers, or the Osborne et al. homotopy code). The asymptotic analysis is credible—a full quadratic programming solution for each $t$ would cost $O(m^3)$ per $t$ value, while LARS computes the entire path in one $O(m^3 + nm^2)$ pass—but the paper provides no empirical timing data to quantify the constant-factor improvement.

**Significance:** The computational claim is central to the paper's practical value proposition, and the asymptotic argument is convincing. However, the absence of timing comparisons against Osborne et al. (2000a) is a genuine gap, since that paper had already developed a homotopy method for the Lasso path that is structurally similar to LARS (the authors acknowledge this in Sections 3.1 and 5). A runtime comparison against the homotopy method would have clarified whether LARS offers additional computational advantages beyond conceptual accessibility.

#### On the Claim That LARS Explains the Lasso/Stagewise Similarity

The experiments **strongly support** this claim, but the support is visual and qualitative rather than quantitative. Figure 1 demonstrates the near-identity of Lasso and Stagewise paths; Figures 1 and 3 together show that both are close to the LARS path. The arrowed points in both panels of Figure 1 identify the specific locations where the three methods diverge, and the paper's geometric analysis (Sections 5 and 6) explains exactly why: the Lasso modification drops variable 7 at the point where its coefficient would cross zero; the Stagewise modification drops variables 3 and 7 at the point where the equiangular vector leaves the convex cone.

**What is demonstrated:** The geometric mechanism of divergence is precisely characterized and visually confirmed.

**What is not demonstrated:** There is no quantitative measure of similarity (e.g., integrated squared difference between paths, correlation between coefficient vectors at corresponding complexity levels, or overlap in selected variables at each model size). The claim of "nearly identical" paths is supported by visual inspection of Figure 1, which is convincing for the diabetes data but leaves open the question of whether the similarity would persist under different data configurations (different correlation structures, different signal-to-noise ratios, different numbers of true non-zero coefficients).

**A missing experiment:** The simulation study (Figure 5) aggregates over 100 replications but reports only proportion explained, not the similarity between Lasso and Stagewise paths across replications. Showing the distribution of, say, the $\ell_2$ distance between Lasso and Stagewise coefficient vectors at each step would have strengthened the unification claim with quantitative evidence. The proportion-explained curves being nearly identical (Figure 5) is consistent with the paths being similar but does not directly measure path similarity—different coefficient vectors can produce similar prediction accuracy.

#### On the Claim That $df(\hat{\mu}_k) \doteq k$ and the $C_p$ Criterion

The evidence is **moderately strong but limited in scope**. Figure 6 demonstrates the simple approximation holds within bootstrap confidence intervals for two specific design matrices (10-predictor and 64-predictor diabetes data). The $C_p$ minima (Figure 7) select models that "looked sensible" and agreed with domain expertise.

**What is demonstrated:** The approximation works for the diabetes data in both its original and quadratically expanded forms. The bootstrap methodology (Equation 4.6–4.8) is clearly described and produces tight confidence intervals.

**What is not demonstrated:** The paper does not test the approximation on:
- Synthetic data with controlled correlation structures designed to violate the Positive Cone Condition. The authors state that "it requires concerted effort at pathology to make $\text{df}(\hat{\mu}_k)$ much different than $k$," but they do not construct or test such pathological cases. This leaves the boundary of applicability unknown.
- The $m > n$ case, which is explicitly flagged as uninvestigated ("We have not investigated the accuracy of the simple approximation formula (4.12) for the case $m > n$"). Given the practical importance of high-dimensional regression, this is a significant gap.
- Data with heavy-tailed errors or heteroskedasticity, where the homoskedastic model (Equation 4.1) that underlies the $C_p$ derivation would be violated.

**The $C_p$ selection is validated only by domain expertise, not by held-out prediction error.** The paper states that the selected models "agreed with an earlier model based on a detailed inspection of the data assisted by medical expertise," but no quantitative prediction evaluation (e.g., cross-validated mean squared error) is reported. The $C_p$ statistic is an unbiased estimator of prediction risk under the linear model assumptions, so it should correlate with true prediction error, but this correlation is assumed rather than demonstrated for the diabetes data.

**A missing experiment (acknowledged in the paper):** The empirical claim that Lasso degrees of freedom approximately equals the number of non-zero coefficients ("$\text{df}(\hat{\mu}_{\ell(k)}) \doteq k$") is stated without any mathematical support and without bootstrap verification. Given that the simple $C_p$ formula (Equation 4.10) applies only to LARS and not to Lasso or Stagewise, providing a comparable formula for Lasso would have been practically valuable. The paper leaves this as an empirical observation with no experimental backing.

#### On the Claim of Better Statistical Performance Than Forward Selection

The simulation study (Figure 5) provides the paper's only quantitative performance comparison, and it **convincingly demonstrates** LARS's superiority over Forward Selection on one specific data-generating configuration.

**Strengths:** The simulation design is careful: it uses the actual diabetes predictor matrix (preserving the real correlation structure), constructs a realistic true model ($\mu$ from 10-step LARS on the original data), generates 100 independent response vectors via residual bootstrap, and reports both means and standard deviations. The Forward Selection comparison is fair—both methods are run on the same 100 datasets, and Forward Selection's performance is tracked at each step count.

**Weaknesses and limitations:**

- **Single data-generating configuration.** The true model ($\mu$ from 10-step LARS) is constructed from the same predictor matrix used in the simulation. This means the simulation tests how well each method recovers a signal that LARS itself would produce—potentially favoring LARS and its close relatives (Lasso, Stagewise). A simulation with a different true coefficient structure (e.g., a sparse model with only 3–5 non-zero coefficients, or a dense model where all predictors contribute) would test whether the equiangular strategy's advantage over Forward Selection generalizes.

- **No comparison against All Subsets or other exhaustive methods.** The 64-predictor case makes All Subsets computationally infeasible, but a smaller simulation (e.g., $m = 15$–$20$ predictors) could have compared LARS against the gold standard of exhaustive search. This would have quantified how much optimality is lost by the greedy-but-cautious LARS strategy relative to the globally optimal subset selection.

- **The proportion explained metric (Equation 3.17) measures recovery of the true mean $\mu$ but not variable selection accuracy.** A model could achieve high $\text{pe}(\hat{\mu})$ by including many small coefficients that collectively approximate $\mu$ well, without correctly identifying which predictors are truly non-zero. The paper does not report metrics like true positive rate, false discovery rate, or $\ell_2$ error in $\hat{\beta}$, which would be relevant to the scientific interpretability goal stated in Section 1 ("suggest which covariates were important factors in disease progression").

- **No comparison against ridge regression.** Ridge regression ($\ell_2$ penalty) is the other major regularization method and would provide a natural baseline for prediction accuracy. Its absence is notable, especially since the paper's introduction discusses shrinkage and the bias-variance tradeoff (citing Hastie et al., 2001) in terms that apply equally to ridge and Lasso.

#### Overall Assessment

The experiments successfully serve their primary purpose: to **illustrate and confirm the geometric and computational properties** derived in the paper's theorems. The coefficient path figures (1, 3) provide visual confirmation of the equiangular mechanism, the Lasso modification's single activation, and the Stagewise modification's more aggressive variable dropping. The simulation study (Figure 5) provides the paper's only head-to-head performance comparison and convincingly shows that the equiangular strategy avoids Forward Selection's worst excesses. The bootstrap degrees-of-freedom analysis (Figure 6) and $C_p$ selection (Figure 7) demonstrate that the simple approximation $df \doteq k$ is empirically viable and leads to sensible model choices.

Where the experimental evaluation falls short of modern standards is in **breadth of conditions tested, quantitative comparison metrics, and held-out validation**. The reliance on a single dataset (with one expanded variant) means the generalizability of the findings—particularly the near-identity of Lasso and Stagewise paths, and the accuracy of the $df \doteq k$ approximation—is assumed rather than demonstrated. The absence of timing comparisons, prediction error evaluations on held-out data, variable selection accuracy metrics, and comparisons against ridge regression or Osborne et al.'s homotopy method represent genuine gaps. These gaps are understandable given the paper's primary contribution as a methodological and theoretical advance, and several are explicitly acknowledged (the $m > n$ case for $df$, the many-at-a-time generalization for Theorem 1). The experiments do support the central claims, but the support is narrower than a dedicated empirical evaluation paper would provide.

## 6. Limitations and Trade-offs

### Limitation 1: The Degrees-of-Freedom Approximation Is Not Proven for General Design Matrices

**The assumption or constraint.** The simple approximation $df(\hat{\mu}_k) \doteq k$—which underpins the $C_p$ criterion (Equation 4.10) and thus the entire inferential framework that gives LARS standalone value—is proven exactly only under two restrictive conditions: mutually orthogonal predictors (Theorem 3) and the Positive Cone Condition (Theorem 4, Equation 4.11: $\mathcal{G}_{\mathcal{A}}^{-1} 1_{\mathcal{A}} > 0$ element-wise for all subsets $\mathcal{A}$). For general design matrices, the paper explicitly acknowledges that the continuity and differentiability assumptions required by Stein's formula (Equation 4.12) can fail:

> "While for the most general design matrices $X$, it can happen that $\hat{\mu}_k$ fails to be almost differentiable, we will see that the divergence formula $\nabla \cdot \hat{\mu}_k(y) = k$ does hold almost everywhere." (Section 4.2)

The distinction between the divergence holding almost everywhere and Stein's formula being applicable is subtle but critical. Stein's SURE theory requires almost differentiability—a stronger condition than divergence existing almost everywhere—and Lemma 3 proves this only under the Positive Cone Condition. The Appendix (Lemmas 13–17) provides a careful treatment of continuity and local linearity at points where the active set changes, but the proof of almost differentiability for general $X$ is not provided.

**The consequence.** For a design matrix that violates the Positive Cone Condition, the $C_p$ formula $C_p(\hat{\mu}_k) \doteq \|y - \hat{\mu}_k\|^2 / \bar{\sigma}^2 - n + 2k$ is an **unproven heuristic** rather than a theoretically justified unbiased risk estimator. A practitioner using $C_p$ to select the LARS model size on arbitrary data has no guarantee that the selected model minimizes prediction error—the $C_p$ values could be systematically biased if the true degrees of freedom deviate substantially from $k$. This matters because the positive cone condition is not a mild technicality; it requires that for every possible active set $\mathcal{A}$, the inverse Gram matrix $\mathcal{G}_{\mathcal{A}}^{-1}$ times the all-ones vector yields strictly positive entries. This condition fails whenever there exist subsets of predictors whose partial correlations induce the equiangular direction to lie outside their convex cone—exactly the circumstance that triggers the Lasso and Stagewise modifications. The diabetes data itself violates the condition, as evidenced by the Lasso modification being triggered (the arrowed point in Figure 1, left panel) and the Stagewise modification requiring projection onto the convex cone (arrowed point in Figure 1, right panel). The fact that the bootstrap estimates (Figure 6) still support $df \doteq k$ for this dataset is encouraging but does not constitute a proof, and the paper does not characterize the class of design matrices for which the approximation holds.

**What evidence exists in the paper.** The bootstrap evidence in Figure 6 demonstrates the approximation holds for two specific design matrices—the 10-predictor and 64-predictor diabetes data—within the resolution of 500 bootstrap replications and their associated confidence intervals. The delta-method argument (Equation 4.15) provides an asymptotic justification: locally, $\hat{\mu}_k$ is linear with matrix $M_k = P_k - \cot_k \cdot u_k v_k'$ having trace $k$. The paper also notes that "it requires concerted effort at pathology to make $df(\hat{\mu}_k)$ much different than $k$" (Section 4.2). However, no pathological counterexamples are constructed or tested, and the boundary of applicability is not characterized. The paper explicitly leaves the $m > n$ case uninvestigated (Section 7), which is an important practical regime where the approximation might be expected to break down due to saturation effects.

**Mitigation status.** The paper offers the bootstrap procedure (Equations 4.6–4.8) as a fallback: if the simple approximation is suspected to be inaccurate, one can estimate $df(\hat{\mu}_k)$ directly via parametric resampling. However, this defeats the paper's primary practical selling point—that $C_p$ model selection requires "no additional computation beyond that for the original LARS estimates" (Section 4). Running $B = 500$ bootstrap replications multiplies the computational cost by roughly 500. The paper provides no guidance on when the simple approximation can be trusted without bootstrap verification, leaving the practitioner to either assume the approximation holds (with unknown risk) or incur the very computational cost that LARS was designed to avoid.

---

### Limitation 2: The $C_p$ Criterion Requires an Independent Estimate of $\sigma^2$, Which May Not Be Available

**The assumption or constraint.** The $C_p$ formula (Equation 4.10) depends on $\bar{\sigma}^2$, an estimate of the error variance. The paper uses the full OLS model on all $m$ predictors to obtain $\bar{\sigma}^2$, which is reasonable when $n > m$ and the full model is not severely overfit. However, this strategy fails in two important practical regimes:

1. **When $m \gg n$ (more predictors than observations):** The full OLS model is not identifiable—there is no unique $\hat{\beta}$, and the residual variance estimate from a saturated model (all $n$ degrees of freedom consumed) is identically zero, making $\bar{\sigma}^2$ unavailable. The paper acknowledges this in Section 7: "The estimation of $\sigma^2$ may have to depend on an auxiliary method such as nearest neighbors (since the final model is saturated)." No such method is developed or evaluated.

2. **When the full OLS model is severely overfit even with $n > m$:** The $C_p$ formula is an unbiased estimator of prediction risk only if $\bar{\sigma}^2$ is an unbiased estimator of the true error variance. If the full OLS model is overfit, $\bar{\sigma}^2$ will be biased downward, making $C_p$ underestimate the true risk and potentially select models that are too complex. The diabetes data with $n = 442$ and $m = 10$ ($n/m = 44.2$) is a relatively benign regime, but with noisier data or larger $m/n$ ratios, the full-model variance estimate becomes increasingly unreliable.

**The consequence.** In the high-dimensional setting ($m \gg n$), which is arguably the most important practical use case for model selection algorithms, the $C_p$ criterion is **not directly applicable** without an auxiliary variance estimation procedure that the paper does not provide. A practitioner facing $m = 1000$ predictors and $n = 100$ observations—a common scenario in genomics, finance, or text analysis—cannot use Equation 4.10 as stated. They must either adopt a different variance estimation method (with unknown impact on $C_p$'s unbiasedness) or abandon $C_p$ entirely in favor of cross-validation, which reintroduces the computational burden that LARS was designed to avoid.

Even in the $n > m$ regime, the dependence on $\bar{\sigma}^2$ from the full model creates a circularity: LARS is promoted as a method to select a parsimonious model because the full OLS model may be overfit, yet the criterion for selecting that parsimonious model requires an estimate of $\sigma^2$ from the very overfit model we are trying to avoid. The paper's simulation study (Section 3.3) sidesteps this issue by using the known true $\sigma^2$ implicit in the residual bootstrap design, but this is not available in practice.

**What evidence exists in the paper.** None. The paper does not test $C_p$ with alternative variance estimators, does not evaluate $C_p$ performance under different $n/m$ ratios, and does not compare $C_p$-selected models against cross-validation-selected models on held-out prediction error. The statement that the $C_p$-selected models "looked sensible" and agreed with domain expertise (Section 4) is qualitative validation, not quantitative evidence of predictive performance. The $m > n$ case is flagged as an open problem.

**Mitigation status.** The paper acknowledges the issue in a single sentence in Section 7 and suggests nearest neighbors as a possible auxiliary method, but provides no development, evaluation, or even a sketch of how this would work. The limitation is effectively unaddressed; the practical value of the $C_p$ criterion is contingent on having access to a reliable $\bar{\sigma}^2$, which the paper assumes rather than provides.

---

### Limitation 3: The Lasso and Stagewise Modifications Can Incur Substantially More Steps Than Pure LARS, and the Worst-Case Computational Advantage Is Not Characterized

**The assumption or constraint.** The headline computational claim—"LARS and its variants are computationally efficient: the paper describes a publicly available algorithm that requires only the same order of magnitude of computational effort as Ordinary Least Squares applied to the full set of covariates" (Abstract)—applies strictly to pure LARS, which always takes exactly $m$ steps. The Lasso and Stagewise modifications can require substantially more steps, and the paper provides only point estimates of this increase rather than a general characterization.

For the diabetes data:
- Pure LARS: 10 steps (always $m$)
- Lasso modification: 12 steps (1.2×)
- Stagewise modification: 13 steps (1.3×)

For the 64-predictor quadratic model:
- Pure LARS: 64 steps
- Lasso modification: 103 steps (1.6×)
- Stagewise modification: 255 steps (4.0×)

The paper acknowledges this variability: "with many correlated variables, the stagewise version can take many more steps than LARS because of frequent dropping and adding of variables, increasing the computations by a factors up to 5 or more in extreme cases" (Section 7). However, no bound is derived on the maximum number of steps as a function of $m$, $n$, or the correlation structure of $X$.

**The consequence.** The computational advantage of LARS over prior Lasso and Stagewise implementations is **variable and data-dependent**, not uniform. For a practitioner with highly correlated predictors, the Stagewise-modified LARS might require 5× or more steps than pure LARS, each step involving Cholesky downdating at $O(m^2)$ cost. While the asymptotic $O(m^3 + n m^2)$ bound still holds in the sense that the total cost is polynomial in $m$, the constant factor can be large enough to matter in practice. The paper provides no timing comparisons against quadratic programming solvers or the Osborne et al. (2000a) homotopy method, so a practitioner cannot assess whether the LARS approach offers a meaningful wall-clock improvement for their specific problem, especially in the Stagewise case where step counts can balloon.

Moreover, the $C_p$ criterion's computational advantage (no extra computation needed) applies only to pure LARS, not to Lasso or Stagewise. A practitioner who wants Lasso estimates for their sparsity properties must run the modified algorithm (potentially 1.6× more steps for the quadratic model) and then **cannot** use the simple $C_p$ formula for model selection—the paper explicitly notes that the $df \doteq k$ approximation "cannot hold for the Lasso, since the degrees of freedom is $m$ for the full model but the total number of steps taken can exceed $m$" (Section 4). The empirical claim that Lasso degrees of freedom approximately equals the number of non-zero coefficients is stated without proof or bootstrap verification. This means that for Lasso—arguably the most practically relevant variant given its popularity—the paper provides no computationally cheap model selection criterion at all.

**What evidence exists in the paper.** The step-count data for the two datasets is reported explicitly. The simulation study (Figure 5) shows that at the 40th step, Stagewise averages 33.23 non-zero terms versus 35.83 for Lasso and 40 for LARS, providing some evidence of the extra steps required. The paper's Section 7 states the $O(m^3 + n m^2)$ bound and notes the potential for "factors up to 5 or more" but provides no systematic study of how step counts scale with correlation strength, number of predictors, or signal-to-noise ratio.

**Mitigation status.** The paper provides no mitigation. It does not derive a bound on maximum steps, does not propose heuristics to reduce step count, and does not compare against alternative implementations. The public S-plus implementation is provided, so practitioners can benchmark on their own data, but the paper offers no guidance on when to expect the modifications to be cheap versus expensive.

---

### Limitation 4: All Experiments Use a Single Dataset with No Independent Test Evaluation

**The assumption or constraint.** Every empirical result in the paper—the coefficient path visualizations (Figures 1, 3), the simulation study (Figure 5), the degrees-of-freedom bootstrap (Figure 6), the $C_p$ model selection (Figure 7), and the $(T, S)$ curve (Figure 8)—uses the diabetes data or a quadratic expansion of the same data. The simulation study (Section 3.3) constructs a true model from the same predictor matrix (10-step LARS on the original diabetes $X, y$) and generates 100 response vectors by residual bootstrap from the same data. There is no second dataset, no external validation, and no held-out test set in any experiment.

**The consequence.** All empirical claims about LARS's behavior are **conditioned on the specific correlation structure of the diabetes predictor matrix**. This is a particularly severe limitation because the paper's central theoretical contributions concern how LARS behaves under different predictor correlation structures:

- The near-identity of Lasso and Stagewise paths (Figure 1) might hold only for predictor matrices where the equiangular direction rarely leaves the convex cone. With different correlation structures—for example, predictors organized in blocks with high within-block and low between-block correlation—the frequency of Lasso and Stagewise modifications could be much higher, producing visibly different paths.

- The degrees-of-freedom approximation (Figure 6) is validated only for the diabetes correlation structure. The paper acknowledges that the diabetes data does not satisfy the Positive Cone Condition (since the modifications were triggered), yet $df \doteq k$ still holds. Whether this generalizes to other non-positive-cone designs is unknown. A practitioner with, say, financial data (where predictors often exhibit factor structure with strong multicollinearity) or genomic data (where $m \gg n$ is the norm) has no evidence that the $C_p$ criterion will select reasonable models.

- The simulation study's demonstration that LARS outperforms Forward Selection (Figure 5) uses a true model $\mu$ constructed from 10-step LARS on the original diabetes data. This means the true model lives in the span of the first 10 LARS-selected predictors for this specific $X$, which almost certainly biases the comparison in favor of LARS (and its close relatives Lasso and Stagewise). A true model with a different sparsity pattern—for example, one where the truly non-zero coefficients correspond to predictors that LARS would not select early—might show Forward Selection outperforming LARS.

- The $C_p$-selected models are validated only qualitatively ("looked sensible," "agreed with an earlier model based on a detailed inspection of the data assisted by medical expertise"). No quantitative prediction error on held-out data is reported. In the 64-predictor simulation, the true $\mu$ is known, so held-out prediction error could have been computed directly from the simulated $y^*$ and the known $\mu$, but this was not done.

**What evidence exists in the paper.** None beyond the single dataset. The paper provides no replication on other datasets, no sensitivity analysis to correlation structure, and no comparison of $C_p$-selected models against cross-validation or held-out prediction error.

**Mitigation status.** The paper provides no mitigation. The limitation is structural—the paper is a methodological contribution with a demonstrative empirical component, not a comprehensive empirical evaluation. The mathematical results (Theorems 1–4) apply to any design matrix, so the theoretical contributions are not dataset-dependent. But the practical guidance (the $C_p$ approximation works, Lasso and Stagewise paths are nearly identical, LARS outperforms Forward Selection) is entirely conditioned on the diabetes data. A practitioner cannot assess, from this paper alone, whether these empirical patterns will hold for their application.

---

### Limitation 5: No Comparison Against Ridge Regression or Other Continuous Regularization Methods

**The assumption or constraint.** The paper's stated goal is model selection—choosing a parsimonious subset of predictors for scientific interpretability and efficient prediction. The Lasso is motivated as "an attractive version of Ordinary Least Squares that constrains the sum of the absolute regression coefficients" (Abstract), and the simulation compares LARS/Lasso/Stagewise against Forward Selection and against each other. However, the paper never compares against ridge regression ($\ell_2$ penalty), which is the other dominant regularization method and the natural baseline for prediction accuracy.

Ridge regression solves: minimize $\|y - X\hat{\beta}\|^2$ subject to $\sum \hat{\beta}_j^2 \leq t$. Like the Lasso, it shrinks coefficients toward zero and trades variance for bias. Unlike the Lasso, it does not produce sparse solutions (coefficients are never exactly zero), so it is not a model selection method in the sense of choosing a subset of predictors. However, it is widely used for prediction, and the paper's introduction cites Hastie et al. (2001) on shrinkage and the bias-variance tradeoff—concepts that apply equally to ridge and Lasso.

**The consequence.** The paper provides no evidence about whether LARS's equiangular strategy produces **better predictions** than ridge regression at comparable model complexity. This matters because a practitioner choosing between methods cares about prediction accuracy, not just about the elegance of the coefficient paths. The simulation study (Figure 5) compares methods on proportion explained $\text{pe}(\hat{\mu})$, which measures recovery of the true mean $\mu$. Ridge regression could plausibly achieve higher $\text{pe}(\hat{\mu})$ than LARS at some model complexities, especially when the true coefficient vector is dense (many small non-zero coefficients rather than a few large ones)—a scenario where the Lasso's sparsity is a disadvantage. The paper's silence on this comparison leaves open the question of whether LARS should be preferred over ridge for prediction tasks, or only when sparsity and interpretability are primary goals.

The omission is particularly notable because the paper's own degrees-of-freedom framework (Section 4) provides a natural way to compare methods on an equal footing: compute $C_p$ (or an equivalent criterion) for ridge regression along its regularization path and compare against LARS's $C_p$ curve. This would directly address the question: for a given degrees of freedom, which method achieves lower prediction risk? The paper's failure to perform this comparison, even on the diabetes data, is a missed opportunity.

**What evidence exists in the paper.** Zero. Ridge regression is mentioned only in passing in Section 8, in the context of boosting connections ("Hastie et al. (2001) noted the striking similarity between Forward Stagewise regression and the Lasso"). The word "ridge" does not appear in the paper.

**Mitigation status.** The paper provides no mitigation. It could be argued that ridge regression is not a model selection method (it does not produce sparse solutions or select predictor subsets), so it falls outside the paper's scope. However, the paper's own stated goals include prediction accuracy ("the model would produce accurate baseline predictions of response for future patients," Section 1), and the $C_p$ criterion is explicitly a prediction risk estimator. Omitting the most natural prediction-focused baseline weakens the case that LARS should be adopted for prediction tasks.

---

### Limitation 6: The "One-at-a-Time" Condition Is Required for the Lasso Modification to Be Computationally Simple, and the General Case Is Not Implemented

**The assumption or constraint.** Theorem 1, which establishes that the LARS modification yields all Lasso solutions, explicitly assumes the "one-at-a-time" condition:

> "Under the Lasso modification, and assuming the 'one at a time' condition discussed below, the LARS algorithm yields all Lasso solutions." (Theorem 1)

The "one-at-a-time" condition means that at each step, at most one variable is added to or removed from the active set. The paper acknowledges that this condition can fail:

> "'One at a time' means that the increases and decreases never involve more than a single index $j$. This is the usual case for quantitative data, and can always be realized by adding a little jitter to the $y$ values. Section 5 discusses tied situations." (Section 3.1)

The suggested fix—adding jitter to $y$—is an ad-hoc perturbation that resolves ties but introduces an arbitrary element into the solution: different jitter realizations produce different Lasso paths. The paper's Section 5 sketches how the many-at-a-time case could be handled in principle (by checking all subsets of the tied variables to determine which yields the correct active set), but explicitly states this is not implemented:

> "A LARS-Lasso algorithm is available even if the one-at-a-time condition does not hold, but at the expense of additional computation... Since one-at-a-time computations, perhaps with some added $y$ jitter, apply to all practical situations, the LARS algorithm described in Section 7 is not equipped to handle many-at-a-time problems." (Section 5)

**The consequence.** The publicly available LARS implementation (Section 7) does **not** guarantee correct Lasso solutions when ties occur—for example, when two predictors have identical absolute correlations with the residual at a step boundary, or when two coefficients would cross zero simultaneously. Adding jitter to $y$ breaks the ties but changes the problem: the algorithm now computes the Lasso path for $y + \delta$ rather than for the actual data $y$. If the jitter is small, the paths will be similar, but there is no formal guarantee, and the practitioner has no way to verify correctness without implementing the subset-checking procedure themselves.

This limitation is more than a theoretical curiosity. Ties can occur naturally when:
- Predictors are exactly collinear or nearly collinear (common in designed experiments and in high-dimensional data with more predictors than observations).
- The response $y$ has symmetries that cause multiple predictors to have equal correlations at a step boundary.
- The design matrix has a block structure where predictors within a block are exchangeable.

In such cases, the jitter fix is a hack that sacrifices reproducibility (different random seeds produce different paths) and potentially accuracy (the jittered path may differ meaningfully from the true Lasso path if the tie structure is extensive).

**What evidence exists in the paper.** None. The paper does not demonstrate the jitter fix on any dataset, does not compare jittered paths against true many-at-a-time paths, and does not characterize the sensitivity of Lasso solutions to the jitter magnitude. The diabetes data apparently contains no ties, so the issue does not arise in any of the paper's empirical examples. The paper's statement that one-at-a-time "is the usual case for quantitative data" is an empirical claim with no supporting evidence—no survey of typical design matrices, no analysis of how often ties occur in practice.

**Mitigation status.** The paper acknowledges the limitation explicitly and sketches a theoretical solution (subset checking), but the implementation does not support it. The jitter suggestion is offered as a practical workaround but is not evaluated. The paper provides no guidance on choosing the jitter magnitude—too small and ties may persist due to floating-point precision; too large and the path may be distorted. This leaves the practitioner with an implementation that works correctly "in the usual case" but has undefined behavior in edge cases that are not characterized.
