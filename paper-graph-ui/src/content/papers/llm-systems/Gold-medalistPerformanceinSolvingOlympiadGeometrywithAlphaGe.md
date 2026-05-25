# Gold-medalist Performance in Solving Olympiad Geometry with AlphaGeometry2

**ArXiv:** [2502.03544](https://arxiv.org/abs/2502.03544)

## 🎯 Pitch

AlphaGeometry2 sets a new benchmark for automated reasoning by outperforming average IMO gold medalists in solving Olympiad geometry problems, raising the solve rate from 54% to 84% on problems from 2000 to 2024. By substantially expanding its geometric language, automating problem formalization and diagram generation, strengthening its symbolic engine, and leveraging large language models in a novel multi-tree search framework, AlphaGeometry2 demonstrates that AI can achieve creative, human-style mathematical reasoning—paving the way for advanced AI tutors, automated proof assistants, and formal math understanding.

---

## 1. Executive Summary

This paper introduces **AlphaGeometry2 (AG2)**, a substantially upgraded neuro-symbolic system that solves Olympiad-level Euclidean geometry problems by pairing a Gemini-based language model with an enhanced symbolic engine (DDAR, the Deductive Database Arithmetic Reasoning). AG2 extends the domain language to cover locus problems involving moving objects and linear equations of angles, ratios, and distances, while introducing the **Shared Knowledge Ensemble of Search Trees (SKEST)** — a novel search algorithm in which multiple differently-configured beam searches run in parallel and exchange discovered facts through a shared database — enabling broader exploration of auxiliary construction strategies. On all IMO geometry problems from 2000–2024, AG2 achieves an 84% solve rate (42 out of 50 translated problems), surpassing an average IMO gold medalist for the first time and substantially improving on AG1's previous 54%, establishing that a language-model-guided symbolic search framework can match elite human performance on Olympiad geometry when the domain language, verifier speed, search algorithm, and model capacity are jointly scaled.

## 2. Context and Motivation

### The Core Problem: Olympiad Geometry as a Reasoning Frontier

The fundamental problem AlphaGeometry2 addresses is deceptively simple to state: **how can we build an AI system that reliably solves Euclidean geometry problems at the level of the world's best high-school mathematicians?** The International Mathematical Olympiad (IMO) represents perhaps the most prestigious benchmark of pre-college mathematical reasoning, and its geometry problems demand a distinctive combination of skills — creative diagrammatic insight, precise algebraic manipulation, and the ability to synthesize non-obvious auxiliary constructions — that has resisted purely neural, purely symbolic, and prior neuro-symbolic approaches alike.

This matters for reasons beyond competition math. Geometry theorem proving is a microcosm of the broader reasoning challenges facing AI: it requires both the creative, pattern-matching intuition that large language models (LLMs) have recently demonstrated, and the rigorous, verifiable deductive chains that symbolic systems guarantee. Mastering this domain forces a reconciliation of these two paradigms — neural "guessing" and symbolic "checking" — in a context where errors are unambiguous and standards are absolute. The architecture that succeeds here yields lessons transferable to other domains demanding creative reasoning under hard constraints, from program synthesis to scientific discovery to automated planning.

### Why Geometry Specifically, and Why Now

The paper is motivated by a specific historical moment. When AG1 (Trinh et al., 2024) was published, it achieved a 54% solve rate on IMO 2000–2024 geometry problems by combining a custom transformer language model (trained on 100M synthetic diagrams) with a symbolic engine (DDAR) through beam search. This was a breakthrough — the first system to solve Olympiad geometry at a competitive level — but it also exposed clear limitations that define AG2's research agenda:

**The domain language was too narrow.** AG1 could only express problems where every point was defined as the intersection of at most two objects (line or circle) — so-called "constructive" problems. Many real IMO problems involve locus statements ("as point X moves on line L, point Y traces a fixed circle"), linear equations of angles and distances (e.g., "∠A + 2∠B = 180°"), or non-constructive definitions where a point satisfies three or more simultaneous constraints. AG1 simply could not represent these problems, leaving 34% of recent IMO geometry problems formally out of reach regardless of how much compute was applied.

**The symbolic engine was slow and lacked double-point reasoning.** DDAR1 (the original symbolic engine) computed deduction closures — the set of all facts derivable from a set of premises — but its time complexity scaled polynomially in the number of points, with bottlenecks at $O(N^8)$ for similar-triangle searches. This limited both training data generation (restricting dataset size) and test-time search (restricting the depth and breadth of exploration). Furthermore, DDAR1 could not handle "double points" — situations where a proof is easier if one reformulates the goal by introducing a new point that happens to coincide with an existing one. This reformulation trick, trading a difficult membership goal for an easier collinearity proof (see Figure 1), is a standard human technique that DDAR1 could not execute because it lacked the machinery to assert that two differently-named points share the same coordinates.

**The language model was capacity-limited.** AG1 used a custom transformer trained from scratch on a domain-specific tokenizer. While effective, this left open questions: would a larger, pre-trained model generalize better? Would fine-tuning a math-specialized Gemini model on AG data outperform training from scratch? Could multimodal information (diagrams as images) augment the purely text-based reasoning? AG1 lacked the infrastructure to even ask these questions.

**The search algorithm was monolithic and isolated.** AG1 ran a single beam search with one language model. If that model produced a poor auxiliary construction early in the tree, the entire search was wasted. There was no mechanism for multiple search strategies to collaborate, no way to share intermediate discoveries between search trees, and no ensemble diversity to hedge against individual model failures.

### Conflicting Tensions in Prior Approaches

The paper positions itself at the intersection of two research traditions that have historically been in tension:

**Pure symbolic methods.** Algebraic approaches — Wu's method (Chou, 1985; Wu, 2008), the Area method (Chou et al., 1993, 1994), and Gröbner bases (Kapur, 1986a,b) — can solve many geometry problems by converting them into systems of polynomial equations and grinding through algebraic elimination. These methods are sound and complete within their domains, but they produce opaque, non-human-readable proofs and scale poorly to problems requiring creative auxiliary constructions. Synthetic approaches — Deduction Databases (Chou et al., 2000) and the Full Angle method (Chou et al., 1996) — produce more human-like proofs but are limited by the deduction rules they implement. AG1 and AG2 use synthetic methods because, as the paper notes, they are "a more human-like approach suitable for transferring the research knowledge to other domains."

**Pure neural methods (LLMs).** In the era of large language models, a natural alternative is to simply ask an LLM to solve the problem end-to-end. But as the paper documents in Table 4, both OpenAI o1 and Gemini Thinking score **zero** on IMO-AG-50. The reason is not mysterious: solving Olympiad geometry requires performing dozens of precise algebraic manipulations without error throughout a potentially long proof, and LLMs "are notoriously unreliable even for basic arithmetics" (Yan et al., 2025, cited in Appendix A). The paper explicitly stakes out a position against LLM-only verification: while some work uses LLMs as verifiers for self-correction (Chen et al., 2024; Lightman et al., 2024; Madaan et al., 2023), this "lies on the assumption that LLMs are reliable verifiers and this notion has been challenged" (Huang et al., 2024; Stechly et al., 2025, cited in Appendix A). For AlphaGeometry, the symbolic engine provides guaranteed-correct verification — a design choice the paper defends as essential.

**Prior neuro-symbolic systems.** AG1 demonstrated that a language model could suggest auxiliary constructions while a symbolic engine verified and extended them, achieving 25/30 on a subset of IMO problems. TongGeometry (Zhang et al., 2024) showed that a refined search procedure could push this to 30/30 on the same subset (Table 4). But both systems were evaluated on IMO-AG-30, a subset of only problems expressible in AG1's restricted language. When evaluated on the full IMO-AG-50, AG1's solve rate drops to 54%, and neither system could even attempt problems involving loci, linear equations, or non-constructive definitions. The gap between IMO-AG-30 and IMO-AG-50 performance was not just a matter of optimization — it was a *representational* gap that no amount of better search could close.

**Wu's method hybrids.** Sinha et al. (2024) combined Wu's algebraic method with AG1's DDAR to achieve a reported solve rate rivaling or exceeding gold medalists on subsets of IMO problems. But this approach inherits the opacity of algebraic methods and does not address the domain language limitations — it can only operate on problems AG1 could already formalize.

### The Central Gap This Paper Fills

None of the prior work addresses what the paper identifies as the bottleneck: **scaling all components of a neuro-symbolic system simultaneously**. AG1 proved the basic architecture works but left performance on the table in four independent dimensions — expressiveness (domain language), speed (symbolic engine), intelligence (language model), and search strategy. Prior improvements (TongGeometry, Sinha et al.) optimized within one or two of these dimensions while leaving the others fixed. AG2's central hypothesis is that these dimensions are not independent — a more expressive language enables generating richer training data, which enables training better language models, which enables more productive search strategies, which in turn demand a faster symbolic engine to evaluate. The whole is greater than the sum of its parts, but only if *all* parts are upgraded together.

### How This Paper Positions Itself

The paper explicitly structures itself as an **engineering paper with scientific contributions at each layer of the stack**:

1. **At the representation layer**: the extended domain language (Section 2) is not just "adding more predicates" — it is a principled expansion to capture locus theorems, linear equations, and non-constructive problem statements, moving from 66% to 88% coverage of IMO geometry problems. This is the foundation on which all other improvements depend, because it determines what problems can be attempted and what training data can be generated.

2. **At the reasoning layer**: the faster DDAR2 symbolic engine (Section 3) is not just "rewriting in C++" — it includes algorithmic improvements (hard-coded search patterns reducing $O(N^8)$ to cubic, hash-based similar-triangle detection) and the conceptually novel double-point handling mechanism, which enables a "reformulation" proof strategy that was impossible in DDAR1.

3. **At the data layer**: the improved synthetic data generation (Section 4) is not just "10× more data" — it includes a new greedy pruning algorithm for finding minimal problem statements, explicit generation of locus-type theorems using movement-dependency tracking ($P(\cdot)$ functions), and a deliberate rebalancing toward harder problems (2× larger diagrams, 10× longer proofs) and a 50:50 ratio of problems with and without auxiliary constructions (vs. 9:91 in AG1). This data determines what the language model learns.

4. **At the learning layer**: the Gemini-based language model (Section 6) is not just "bigger transformer" — it includes controlled experiments comparing training from scratch vs. fine-tuning pre-trained math models, custom tokenizers vs. standard LLM tokenizers, and domain-specific language vs. natural language translation, all evaluated on downstream IMO solve rate rather than proxy metrics.

5. **At the search layer**: the SKEST algorithm (Section 5) is not just "ensemble of beam searches" — it introduces a knowledge-sharing mechanism where individual nodes in any search tree can write discovered facts (filtered to be problem-relevant, not auxiliary-point-specific) into a shared database accessible to all other nodes in all other search trees, effectively turning parallel search into a collaborative rather than competitive process. The search trees themselves are deliberately diverse: some predict single aux points while others predict multiple; some force uniform distribution over aux point types; some are deep-and-narrow (beam size 64, depth 10) while others are shallow-and-wide (beam size 512, depth 4).

The paper also explicitly connects to the broader trajectory toward end-to-end automated systems. Appendix G describes progress on automated formalization (using Gemini's few-shot translation from natural language to AG domain language, achieving 33/44 formalizable problems) and automated diagram generation (a three-stage optimization combining Adam, Gaussian elimination, and Gauss-Newton-Levenberg methods to construct diagrams for non-constructive problem statements). These components move AlphaGeometry from a system that requires human translation effort toward a fully automated pipeline that "takes inputs in natural language and reliably outputs corresponding diagrams and full solutions without any hallucinations" (Section 8).

Finally, the paper positions its unsolved problems as explicit research challenges rather than failures. The remaining 12% of unformalizable IMO problems — involving 3D geometry, inequalities, non-linear equations, and countably many points — define the boundaries of the current approach. The two attempted-but-unsolved problems (IMO 2018 P6 and IMO 2023 P6) are attributed to missing deduction machinery (inversion, projective geometry, radical axis), with the paper hypothesizing that "breaking problems into subproblems and applying Reinforcement learning approaches could close this gap" (Section 8). This framing — clear about achievements, transparent about limitations, and explicit about next steps — positions AG2 not as an endpoint but as the current state of a trajectory the authors clearly intend to continue.

## 3. Technical Approach

### 3.1 Reader Orientation

AlphaGeometry2 is a neuro-symbolic theorem prover: a search algorithm that uses a large language model (Gemini) to propose creative geometric constructions, and a fast symbolic engine (DDAR2) to verify whether those constructions actually prove the goal. The system solves the problem of Olympiad geometry by splitting the task into two parts — a "creative" part (suggesting auxiliary points, lines, or circles) handled by a learned neural model, and a "rigorous" part (deducing all logical consequences of those suggestions and checking whether the goal follows) handled by a guaranteed-correct symbolic engine — with a novel ensemble search strategy that lets multiple search trees running different strategies share their intermediate discoveries.

### 3.2 Big-Picture Architecture (Diagram in Words)

The system operates as an asynchronous pipeline with seven major components:

1. **Problem Input:** A geometry problem expressed in a domain-specific formal language (e.g., `triangle a b c; a b = a c ? eqangle b a b c c b c a`). This arrives either from automated formalization of natural language (Appendix G) or from human translation.

2. **Analysis String Generator:** Before search begins, the symbolic engine (DDAR2) computes three sets of facts — (a) facts deducible from the premises alone ($S_1$), (b) facts deducible if we additionally assume the goal is true ($S_2$), and (c) facts that are numerically true by inspecting the diagram ($S_3$). These are serialized into the "analysis string" as `S₁ (S₂ − S₁) (S₃ − S₂)` and prepended to the problem statement.

3. **Language Model (LM) Workers (Gemini):** Multiple replicas of one or more Gemini-based models run asynchronously on TPUv4 hardware. Each worker receives the problem statement + analysis string and is prompted to generate auxiliary constructions (new points with geometric constraints, e.g., "construct point X such that X lies on line AB and X is equidistant from C and D"). Different search trees call these workers with different prompting strategies and decoding parameters.

4. **DDAR2 Workers (Symbolic Engine):** A pool of C++-accelerated symbolic engines run asynchronously, reading candidate auxiliary constructions from a database and attempting to compute the deduction closure from the premises plus each construction. If the closure contains the goal fact, the problem is solved. If not, the worker records what facts *were* deduced and writes them (filtered to remove construction-specific facts) to the shared facts database.

5. **Shared Facts Database (Knowledge-Sharing Mechanism):** A global repository of "interesting facts" — facts about the original problem that DDAR2 could *not* prove without auxiliary constructions, but *did* prove once a particular auxiliary construction was added. These facts are made available to all nodes in all search trees, enabling one search to benefit from the partial progress of another even if neither individually solved the problem.

6. **Search Controllers (Multiple Strategies):** Each search tree runs under its own controller (a beam search or variant) that queries LM workers for auxiliary constructions, feeds candidate constructions to DDAR2 workers, monitors results, and populates the shared facts database. The controllers are deliberately diverse: some predict one aux point per node ("classic" search), others predict multiple aux points in one generation; some force uniform sampling across different types of aux point predicates; some are deep-and-narrow, others shallow-and-wide.

7. **Termination:** As soon as *any* DDAR2 worker reports that the goal is in the deduction closure (given the original premises plus some sequence of auxiliary constructions), all search trees terminate. The proof is the sequence of auxiliary constructions that led to the goal.

Information flows asynchronously: LM workers write candidate nodes to a database; DDAR2 workers pick up nodes from that database, attempt them, and write results (success or newly-discovered facts) to the shared database; search controllers read results and decide which nodes to expand next based on their beam search logic. The system can solve multiple problems simultaneously, with idle DDAR2 workers from solved problems reallocated to unsolved ones.

### 3.3 Roadmap for the Deep Dive

- **First**, the expanded domain language (Section 2 of the paper), because it determines which problems are even *representable* and constrains what the symbolic engine and language model can operate on.
- **Second**, the DDAR2 symbolic engine — its algorithms, the double-point handling mechanism, and the performance improvements — because DDAR2 is the "verifier" that all search relies on, and its speed and capabilities determine the feasible scale of both training data generation and test-time search.
- **Third**, the synthetic training data generation pipeline, because it determines what the language model learns about auxiliary constructions, and the improvements in data quality and quantity are what make the Gemini models effective.
- **Fourth**, the language model itself — training setup, inference setup, the analysis string interface, and the surprising findings about tokenizers and domain-specific languages — because the LM is the "creative" component that proposes constructions.
- **Fifth**, the SKEST search algorithm, because it orchestrates all other components at test time and its knowledge-sharing mechanism is the system-level innovation that ties improvements together.

### 3.4 Detailed, Sentence-Based Technical Breakdown

This is primarily a **systems engineering paper** whose core idea is that achieving gold-medalist-level geometry theorem proving requires simultaneously scaling four independent components — domain expressiveness, symbolic engine speed, language model capacity, and search diversity — and introducing a knowledge-sharing mechanism that lets parallel search trees collaborate rather than compete.

---

#### The Expanded Domain Language (Section 2 of the Paper)

The domain-specific language is the foundation of everything AG2 does, because it determines which problems can be expressed, which training data can be generated, and which deduction rules the symbolic engine can apply. AG1's language consisted of nine basic predicates (Table 1): `cong` (segment equality), `perp` (perpendicular), `para` (parallel), `coll` (collinearity), `cyclic` (concyclicity), `eqangle` (directed angle equality), `eqratio` (ratio equality), `aconst` (constant angle value), and `rconst` (constant ratio value). Points were defined as the intersection of at most two objects (lines or circles), which restricted AG1 to **constructive** problems only — problems where every point can be constructed step-by-step by following their definition order.

**Problem types AG1 could not represent.** The paper identifies four categories of problems that were formally unreachable for AG1:

- **"Find X" problems** (e.g., IMO 2009 P4: "Find the angle between..."). AG1 could only prove relationships between known quantities, not compute unknown values.
- **Linear equation problems** (e.g., IMO 2024): problems where geometric quantities are related by linear equations like $t_1 \cdot \log(AB) + t_2 \cdot \log(CD) + y = 0$ or $t_1 \cdot AB + t_2 \cdot CD = 0$. AG1 had no syntax for expressing weighted sums of distances or angles.
- **Locus problems**: problems about how one geometric object moves as another moves — "as point X moves on line L, point Y traces a fixed circle." These require representing movement dependency and fixed-invariant properties, which AG1's static language could not express.
- **Non-constructive problems**: problems where a point is defined by three or more simultaneous constraints rather than being constructible as the intersection of exactly two objects.

**New predicates for computation and linear equations.** AG2 adds five new predicates to address the first two categories:

- `acompute a b c d`: Find the angle between lines $AB$ and $CD$. This transforms "compute" problems into proof problems — the system must derive the correct angle value.
- `rcompute a b c d`: Find the ratio $AB/CD$, similarly converting a computation goal into a derivation task.
- `distmeq a1 b1 a2 b2 ... an bn t1 t2 ... tn y`: Represents the equation $t_1 \log(A_1B_1) + t_2 \log(A_2B_2) + \cdots + t_n \log(A_nB_n) + y = 0$. The coefficients $t_i$ are numbers and $y$ is a constant. The log-distance representation is algebraically convenient because products of distances become linear sums.
- `distseq a1 b1 a2 b2 ... an bn t1 t2 ... tn`: Represents $t_1 \cdot A_1B_1 + t_2 \cdot A_2B_2 + \cdots + t_n \cdot A_nB_n = 0$, a linear equation in distances (not log-distances).
- `angeq a1 b1 a2 b2 ... an bn t1 t2 ... tn y`: Represents $t_1 d(A_1B_1) + t_2 d(A_2B_2) + \cdots + t_n d(A_nB_n) + y = 0$, where $d(AB)$ is the angle between the undirected line $AB$ and a fixed reference (the horizontal line). This expresses linear equations in angles.

**Locus problem syntax.** For locus problems, AG2 introduces a convention for representing moving points using a wildcard token `*`. The paper lists 11 locus cases in Table 2, each with a corresponding predicate pattern. For example:

- **Case 3 (line through fixed points):** When the problem says "line through A and B moves through a fixed point," the AG2 question syntax is `? coll a b * : X`, meaning "there exists a point X (the fixed point) such that X is collinear with A and B." The `*` is a placeholder for the moving argument, and `: X` names the unknown fixed entity.
- **Case 8 (point on fixed circle):** `? cyclic a * * * : X` means "there exists a fixed circle X such that moving point a always lies on it." The three `*` arguments represent the three points needed to define a circle, all of which are unconstrained because the circle is fixed in the statement.
- **Case 1 (circle through fixed points):** `? cyclic a b c * : X` means "the circumcircle of a, b, c always passes through a fixed point X." Here `*` is the unknown fixed point on the circle.

The key insight is that locus statements are expressed as **existence problems** — "there exists some fixed geometric object (point, line, circle) with property P" — and the `*` syntax marks which argument positions are the unknown moving elements versus the unknown fixed elements.

**Non-constructive problems and diagram support.** AG1 required every point to be defined by at most two predicates, making the problem statement a constructive recipe for drawing the diagram. AG2 relaxes this: points can be defined by three or more predicates, so that a diagram is not constructible by following a simple recipe. To support this during formalization, the paper introduces automated diagram generation (Appendix G), which initializes points via three alternating strategies (random normal distribution, in-order construction, heuristically-ordered construction) and then optimizes coordinates to satisfy all constraints using a three-stage pipeline: Adam gradient descent, Gaussian elimination, and Gauss-Newton-Levenberg optimization. This decouples problem definition from diagram construction.

**Additional diagram-check predicates.** AG2 introduces explicit predicates for representing topological and non-degeneracy conditions that the symbolic engine uses during deduction:

- `sameclock a b c d e f`: The directed triplet $A \to B \to C$ has the same clockwise orientation as $D \to E \to F$. This expresses ordering relations on a line or around a circle.
- `noverlap a b`: Points $A$ and $B$ are distinct. This is a non-degeneracy condition.
- `lessthan a b c d`: Segment $AB$ is shorter than segment $CD$, used in the SSA (Side-Side-Angle) triangle congruence theorem.
- `overlap a b`: Points $A$ and $B$ are actually the same point (different names for the same coordinates). This is the mechanism for handling double points (see DDAR2 section below).
- `cyclic_with_center a1 a2 ... an x`: The first $x$ points ($a_1$ through $a_x$) are all the *center* of the circle that passes through the remaining $n-x$ points. When $x=0$, this reduces to the ordinary `cyclic` predicate. This predicate is introduced because overlapping points can be defined by being the center of the same circle.

**Coverage improvement.** With these extensions, the domain language coverage on IMO 2000–2024 geometry problems rises from 66% (AG1) to 88% (AG2). The remaining 12% are problems involving 3D geometry, inequalities, non-linear equations, and countably many points (problems parameterized by an arbitrary integer $n$, meaning the configuration has $n$ points where $n$ can be any positive integer). These are marked as "Not attempted" on Figure 8. Within the 88% that can be formalized, not all are solved — the coverage increase means more problems are *attempted*, and the solving rate within the attempted set determines the final 84% figure.

---

#### The DDAR2 Symbolic Engine (Section 3 of the Paper)

The symbolic engine, DDAR (Deductive Database Arithmetic Reasoning), is the component that performs guaranteed-correct deductive reasoning. Its job is to compute the **deduction closure**: given a set of initial facts (the premises of the problem plus any auxiliary constructions), iteratively apply a fixed set of geometric deduction rules until no new facts can be derived. If the goal fact appears in the closure, the problem is proven. DDAR2 is used in two contexts: (1) during synthetic data generation, where it computes closures on millions of random diagrams to produce training examples, and (2) during test-time proof search, where it evaluates candidate auxiliary constructions proposed by the language model. Speed in both contexts directly determines how much data can be generated and how many auxiliary construction attempts can be evaluated within a fixed time budget.

##### The Core Deduction Mechanism

DDAR2 operates over a set of points and maintains a database of known facts (predicates involving those points). The algorithm iterates through a fixed list of geometric deduction rules — such as "if A, B, C are collinear and A, B, D are collinear, then A, C, D are collinear," or "if quadrilateral ABCD is cyclic then ∠ABC = ∠ADC," or "if AB/CD = EF/GH and AB/CD = IJ/KL then EF/GH = IJ/KL" — and attempts to apply each rule to all combinations of points that satisfy its premises. When a rule fires, the conclusion is added to the database, potentially enabling new rule applications. This process repeats until a fixed point is reached (no new facts can be derived) or a computational limit is exceeded.

The key distinguishing feature of DDAR over a pure logical deduction engine is the **AR (Arithmetic Reasoning) submodule**. This submodule handles linear equations involving angles, distances, and log-distances. Instead of reasoning about these as discrete facts (like "∠ABC = 60° or ∠DEF = 60°"), the AR submodule maintains a system of linear equations and can reduce any linear expression to a normal form through Gaussian elimination. For example, if the database contains "∠ABC = ∠DEF + 30°" and "∠DEF = 2∠GHI," the AR submodule can combine these to derive "∠ABC = 2∠GHI + 30°" without a separate deduction rule for each algebraic operation. This makes the engine exponentially more efficient for angle and ratio reasoning than a purely rule-based approach.

##### DDAR1's Performance Bottlenecks

The paper identifies two main computational bottlenecks in the original DDAR1:

- **Similar triangle search:** For every triple of points in the diagram, DDAR must check whether they form a triangle similar to any other triangle formed by three points. The naive search time complexity is $O(N^8)$ in the number of points $N$ — a triple-of-triples search — which becomes infeasible for diagrams with more than a dozen points. This was the single most expensive step in DDAR1.
- **Cyclic quadrilateral search:** Similar to the triangle search, detecting all sets of four concyclic points from angle equalities requires examining combinations of angle facts, which is exponential in the number of clauses per premise in the worst case.
- **Rule-based angle/distance reasoning:** DDAR1 used explicit deduction rules for relationships like perpendicular lines, parallel lines, and angle additions. Each rule application required pattern-matching against all combinations of known facts, and the interaction between angle rules and ratio rules required many intermediate steps.

##### DDAR2 Algorithmic Improvements

Rather than relying on general pattern-matching for these expensive operations, DDAR2 implements specialized, hard-coded search procedures that exploit the structure of geometry:

**Similar triangle detection via shape hashing.** For each triple of points $(A, B, C)$, DDAR2 computes a "shape" signature — a normalized representation of the triangle's angles and side ratios computed by the AR submodule's normal form. It then hashes all triples by their shape signature. If any two distinct triples $(A, B, C)$ and $(D, E, F)$ produce the same hash (meaning the AR submodule has determined they have the same angle or ratio configuration, up to known equations), they are flagged as similar. This reduces the search from $O(N^8)$ to $O(N^3)$, since only one pass over all triples is needed. The hashing relies on the AR submodule's ability to reduce any angle or ratio expression to a canonical normal form, so triangles that are "provably similar given the current knowledge" will hash identically.

**Cyclic quadrilateral detection via angle hashing.** For each pair (point $X$, segment $AB$), DDAR2 hashes the value of $(A, B, \angle AXB)$, where the angle value is again the AR submodule's normal form. If two pairs $(X, AB)$ and $(Y, AB)$ produce the same angle hash, then $A, B, X, Y$ form a cyclic quadrilateral (by the inscribed angle theorem: points subtending the same angle from the same chord lie on the same circle). This reduces the search from exponential clause matching to $O(N^3)$.

**Elimination of explicit angle/distance rules.** DDAR2 removes the separate rule base for angles and distances and instead delegates all such reasoning to the AR submodule. Deductions about perpendicular lines, parallel lines, angle sums, and ratio equalities are computed automatically by the Gaussian elimination engine rather than by discrete rule applications. This both simplifies the rule base and accelerates the deduction closure because multiple algebraic steps can be collapsed into a single Gaussian elimination operation.

##### Double-Point Handling (Section 3.1)

This is perhaps the conceptually most novel improvement in DDAR2. Many geometry proofs require a "reformulation" trick: instead of proving a difficult statement directly (e.g., "point X, defined as the intersection of lines a and b, lies on circle ω"), one constructs a new point X' as the intersection of one of those objects with the target object (e.g., X' as the intersection of line a and circle ω), proves that X' also lies on the other object (line b), and then concludes that X and X' must be the same point (since two lines intersect in at most one point), hence X (which is X') lies on ω.

DDAR1 could not execute this strategy because it could not represent the fact that two differently-named points have the same coordinates. The new `overlap` predicate solves this. The four-step procedure for double-point reasoning is:

1. **Construction:** The language model proposes an auxiliary point $X'$ defined as the intersection of one known object (e.g., line a) with the target object (e.g., circle ω). At this point, $X'$ is a distinct point from $X$ — the system does not yet know they coincide.

2. **Proof of collinearity/coincidence:** DDAR2 attempts to prove that $X'$ lies on the remaining object (line b). If successful, $X'$ lies on both a and b.

3. **Overlap deduction:** Since $X$ is defined as the intersection of a and b, and now $X'$ is also proven to lie on both a and b, DDAR2 can deduce `overlap X X'` — they are the same point. (The justification: two distinct lines intersect in at most one point, so any point on both lines must be that unique intersection point; the `overlap` rule captures this.)

4. **Goal transfer:** Once $X$ and $X'$ are known to overlap, any predicate true of $X'$ is automatically true of $X$. In particular, since $X'$ is on ω by construction, $X$ is on ω, which proves the goal.

This mechanism is what gives DDAR2 the ability to "reformulate" proofs in the way human geometers do, and it is critical for hard problems where the direct proof path involves difficult angle or ratio chasing but the indirect path is simpler. The paper notes that this is a capability DDAR1 was "missing" and which is "crucial for tackling hard problems."

##### The C++ Implementation (Section 3.3)

The core arithmetic reasoning in DDAR (the Gaussian elimination engine that normalizes linear expressions of angles, distances, and log-distances) was reimplemented in C++ and exported to Python via pybind11. The paper reports a speed improvement of **over 300 times** compared to the original Python implementation. On a benchmark of 25 IMO problems that DDAR cannot solve (selected from the IMO-AG-50 set), DDAR1 averaged $1179.57 \pm 8.055$ seconds (about 19.7 minutes), while DDAR2 averaged $3.44711 \pm 0.05476$ seconds (about 3.4 seconds). This two-order-of-magnitude speedup is what makes the larger-scale training data generation (Section 4) and the more aggressive test-time search (Section 5) computationally feasible.

---

#### Synthetic Training Data Generation (Section 4 of the Paper)

The language model's ability to suggest useful auxiliary constructions comes entirely from training on algorithmically generated synthetic data. No human-written proofs, human-annotated diagrams, or human-crafted problem statements are used — a deliberate design choice that "eliminates the risk of data contamination and allows for the exploration of theorem distributions that may extend beyond established human knowledge." The data generation pipeline mirrors AG1's but with crucial improvements in scale, quality, and scope.

##### The Basic Pipeline

The generation procedure for a single training example works as follows:

1. **Sample a random diagram:** Generate a random geometric configuration by choosing random coordinates for some initial points and constructing additional points through random geometric operations (midpoint, foot of perpendicular, intersection of lines/circles, etc.). The construction follows a random sequence, creating a well-defined constructive diagram.

2. **Run DDAR2 deduction closure:** On this diagram, run DDAR2 to derive all facts that logically follow from the construction. This produces a large set of facts — typically thousands for a medium-complexity diagram.

3. **Select a target fact:** For each fact in the closure, the system will attempt to create a "problem" that asks for that fact as the goal and a "proof" that shows how to derive it. Not all facts are equally useful; the system balances across different predicate types (collinear, cyclic, equal angles, equal ratios, etc.) to ensure the training distribution covers all reasoning patterns.

4. **Traceback to find a minimal proof:** Starting from the target fact, trace backwards through the deduction closure to identify which premises and which deduction rules were used to prove it. This produces a proof — a sequence of deduction steps — but that proof uses *all* points in the original diagram, many of which are unnecessary for the specific target.

5. **Prune to find a minimal problem:** Remove points from the diagram and check whether the target fact is still provable (by re-running DDAR2 on the reduced set). The goal is to find the smallest set of initial premises (problem statement) that still implies the target. This is the step where AG2 introduces a significant algorithmic improvement (see below).

6. **Identify required auxiliary points:** Among the points in the minimal diagram, determine which are "auxiliary" — points that must be constructed to prove the goal but are not mentioned in the goal or the problem premises. These become the auxiliary construction targets that the language model must learn to predict.

7. **Serialize as a training example:** The problem statement (minimal premises), the goal fact, the auxiliary constructions, and the proof steps are serialized into the AG domain language as a single training instance.

##### The Greedy Pruning Algorithm

AG1's pruning step used an exhaustive search: it tried removing every possible subset of points and re-ran DDAR to check provability, selecting the subset with smallest cardinality. This is exponential in the number of points and becomes completely infeasible for larger diagrams (the diagrams AG2 uses are up to 2× the size of AG1's, with more points). AG2 switches to a **greedy algorithm** (Figure 3):

```
def prune_points(points, check_provable):
    pruned = set(points)
    for p in reverse_topological(points):
        if check_provable(pruned - {p}):
            pruned = pruned - {p}
    return pruned
```

**What it computes:** For a set of points and a predicate `check_provable` (which returns `True` if the target fact is still derivable from the given set), the algorithm iterates through all points in reverse topological order (starting from the "most derived" points — those that depend on other points for their construction — and ending with the initial seed points). For each point $p$, it tentatively removes $p$ from the set and checks if the goal is still provable. If so, $p$ stays removed; if not, $p$ is kept. This requires only a linear number of checks (one per point) rather than exponential.

**Why this form works:** The algorithm is guaranteed to find a minimal set with respect to inclusion (no point can be removed without breaking provability) *provided* the `check_provable` predicate is monotonic — meaning that if a set $A$ is sufficient to prove the goal, any superset $B \supseteq A$ is also sufficient. The paper notes that incorporating construction dependency closure into `check_provable` (i.e., requiring the pruned set to remain closed under construction dependencies so a random construction is still possible) breaks monotonicity, but this can be fixed by processing points in reverse topological order — removing the most dependent points first ensures that by the time a point is considered, all points depending on it have already been kept or discarded, so the dependency closure condition is automatically satisfied.

##### Data Scale and Distribution Improvements

Compared to AG1, AG2's training data has four major improvements (Figure 2):

1. **Larger diagrams:** The random diagrams are explored at up to 2× the size in terms of number of points, enabling generation of more complex problems that require deeper reasoning chains. The problem size distribution (Figure 2a) is shifted toward larger numbers of points.

2. **More complex theorems:** Problems are generated with up to 2× more points and premises, and proofs have up to 10× more deduction steps. This exposes the language model to the kind of long reasoning chains that appear in hard IMO problems.

3. **Balanced predicate distribution:** The distribution of question types (collinear, equal distance, concyclic, parallel, perpendicular, equal ratio, equal angles, compute value) is made much more uniform (Figure 2b). In AG1, some predicate types dominated heavily; in AG2, the distribution is deliberately flattened so the model sees all reasoning patterns equally.

4. **Balanced auxiliary/no-auxiliary ratio:** In AG1, 91% of training examples had no auxiliary points and only 9% required auxiliary constructions. In AG2, this is rebalanced to a 50:50 split (Figure 2c). This is critical because the language model's main job at test time is to predict auxiliary constructions — training primarily on problems that need no aux points left the model underexposed to the very skill it most needed.

##### Locus-Type Theorem Generation

AG2's data generator can also produce problems involving locus statements, which AG1 could not. The mechanism relies on tracking **movement dependency**: for each point $X$, the function $P(X)$ records the set of points that control $X$'s movements. Table 3 gives two examples:

- If `a = midpoint(b, c)` and `d = midpoint(a, c)`, then $P(d) = \{b, c\}$ because $d$ is uniquely determined by $b$ and $c$ through a chain of constructions (the midpoint of $a$ and $c$, where $a$ itself is the midpoint of $b$ and $c$).
- If `a = on_line(b, c)`, meaning $a$ can be anywhere on line $bc$, then $P(a) = \{a, b, c\}$ — $a$ is self-dependent because its position along the line is unconstrained.

During random diagram generation, the system records $P(\cdot)$ for every point. It then examines each fact in the deduction closure and checks whether the dependence structure matches one of 17 templates (Table 5). For example:

- If DDAR proves `cong a b c d`, and $P(b, c, d) - P(a)$ is nonempty (meaning $b, c, d$ are controlled by a set of points that does not fully control $a$), then the system generates a locus theorem: "the circle with center $b$ and radius $cd$ passes through a fixed point $a$" (Case 2 in Table 2). The auxiliary points in the synthetic proof will be the elements of $P(b, c, d) - P(a)$ and everything they depend on.

- If DDAR proves `coll a b c` and $P(c) - P(a, b)$ is nonempty, then "point $c$ moves on a fixed line $ab$" (Case 7).

These 17 detection patterns cover the 11 locus types listed in Table 2. The generator records the movement dependency structure alongside the synthetic proof, so the training data includes not just the construction but the *justification* for why a locus statement holds — the dependency information that the symbolic engine uses at test time to verify locus claims.

---

#### The Language Model (Section 6 of the Paper)

The language model is the "creative" component that proposes auxiliary constructions. It is trained to generate sequences of geometric constructions followed by deduction steps, but at test time, only the auxiliary construction part is used — the symbolic engine takes over once constructions are proposed.

##### Model Architecture and Training

AG2's language model is a **sparse mixture-of-experts Transformer** based on the Gemini architecture (Gemini Team, 2024). Multiple model sizes are trained, with parameter counts mentioned as $51$ million, $176$ million, and $3.3$ billion (denoted "3p3B" in Figure 5). The paper states that "the exact number of TPUs depends on the model size," indicating that training and inference are conducted on TPUv4 hardware, but does not provide explicit TPU counts.

Training uses a single phase of unsupervised learning on the full AG2 dataset (approximately 300 million theorems, an "order of magnitude larger" than AG1's 100M). This is simplified from AG1's two-phase training (first on all data, then on only auxiliary-point problems). The optimization uses the largest possible batch size allowed by hardware, with "no training issues compared to smaller batches." The learning rate schedule is a linear warm-up followed by cosine annealing, with hyperparameters "determined from scaling laws" (details not provided).

Three training setups are evaluated:

1. **Training from scratch** with a custom tokenizer in the AG domain-specific language (the AG1 setup). This tokenizer operates at the word level, where each token is a complete predicate name, point name, number, or special symbol. The vocabulary size is a few thousand tokens.

2. **Fine-tuning pre-trained math-specialized Gemini models** in natural language (Appendix B). These models were pre-trained on public math datasets and then fine-tuned on AG data translated into natural language (e.g., `coll a b c` becomes "a, b, c are collinear").

3. **Multimodal training from scratch** with an additional image input — a diagram of the geometry problem rendered as an image (Appendix C). This model receives both the problem text and the diagram as pixel input.

The paper reports an interesting finding: despite significant differences in tokenizers (custom word-level with few thousand tokens vs. standard Gemini subword tokenizer with 300k tokens), language (domain-specific AG language vs. natural language), and training paradigm (from scratch vs. fine-tuning pre-trained models), **all setups achieve comparable downstream IMO solve rates**. The learning curves in Figure 5 show that larger models achieve lower perplexity on train, eval, and the special IMO evaluation set, with the 3.3B model reaching the lowest loss. Figure 10 shows that a model pre-trained on math data has initially lower loss but converges to the same point as a from-scratch model after training on 200B tokens.

##### The Analysis String Interface (Section 6.2)

A key improvement over AG1 is how the language model receives information about what the symbolic engine has already deduced. In AG1, the LM input was simply `<problem_statement>`. In AG2, the system computes three sets of facts and formats them into an **analysis string**:

- **$S_1$:** The set of all facts that DDAR2 can deduce from the original problem premises alone, without any auxiliary constructions. This is the "trivial" deduction closure — what follows immediately from the given information.

- **$S_2$:** The set of all facts that DDAR2 can deduce if we assume the goal predicate is also true (in addition to the premises). This captures what would follow if the theorem were correct — it provides the model with "forward from premises plus goal" context, which can suggest proof strategies by showing intermediate facts that connect premises to conclusion.

- **$S_3$:** The set of all facts that are numerically correct when verified against the diagram coordinates. These are facts that are *true in the particular diagram* but may not be logically implied by the premises — they represent "diagram observations" that a human would see by looking at the figure but that need formal proof.

By definition, $S_1 \subset S_2 \subset S_3$ (facts deducible from premises are also deducible with the goal assumed, which are a subset of all numerically true facts). The analysis string is constructed as:
```
<problem_statement> serialized(S₁) serialized(S₂ − S₁) serialized(S₃ − S₂)
```

The subtraction ensures facts are not repeated. The order presents the model first with what is strictly known ($S_1$), then with what follows from assuming the goal ($S_2 - S_1$, which hints at proof strategies without giving the answer), then with diagram observations that need proof ($S_3 - S_2$). This enriched context helps the language model propose auxiliary constructions that connect known facts to the goal rather than generating random constructions.

##### Inference Setup (Section 6.2)

At inference time, the language model uses **top-k sampling with temperature $t = 1.0$ and $k = 32$**. The paper provides ablation evidence that this is necessary: with greedy decoding ($t = 0.0, k = 1$) and no tree search, the models solve only 2 out of 26 IMO problems requiring auxiliary constructions. With temperature 1.0 and 32 samples (but still no tree search), this rises to 9 out of 26. Lower temperatures ($t < 1.0$) "do not produce diverse enough auxiliary constructions" (Figure 6 shows the ratio of unique samples drops sharply below $t = 1.0$), while higher temperatures "result in an increasing number LM outputs with a wrong domain language syntax" — the model starts generating malformed constructions that DDAR2 cannot parse.

The LM's output format is a sequence of constructions. A typical auxiliary point construction looks like:
```
x00 a : cong a b c d (000) coll a e f (001)
```
meaning "construct a new point `a` such that `a b = c d` (constraint 000) and `a`, `e`, `f` are collinear (constraint 001)." The `x00` token marks the start of an auxiliary construction. Multiple constructions can be generated in sequence, each prefixed by `x00` or subsequent numbers. At test time, the system extracts the complete set of generated aux points and passes them to DDAR2 for verification.

##### Tokenizer and Language Independence (Appendix B)

The paper reports two surprising findings that have implications beyond geometry:

**Tokenizer independence:** Models trained with a custom word-level tokenizer (vocabulary of a few thousand tokens, where each token has full meaning — e.g., "coll" is one token, "a" is another, "cyclic" is another) perform identically on IMO problems to models trained with the standard Gemini subword tokenizer (vocabulary of 300k tokens). This challenges the common belief (cited from Singh and Strouse, 2024) that tokenizers are a major bottleneck in mathematical reasoning. The paper attributes this to the AG language's regularity: the "word-level" tokens correspond precisely to the semantic units (predicates, points, numbers), and modern LLM subword tokenizers appear flexible enough to learn equivalent representations.

**Domain-language independence:** When the entire AG2 training dataset is translated from the domain-specific language into natural language (e.g., `coll a b c` becomes "a, b, c are collinear"), and a model is trained on the natural-language version, it achieves the same downstream IMO solve rate as the model trained on the domain-specific language. This suggests that the formal language is not providing a special inductive bias for learning — the model can learn the geometric reasoning patterns equally well from natural-language descriptions. This finding "opens a path for fine-tuning large language models pre-trained in natural language on math data," which is exactly what the fine-tuning setup does.

**Fine-tuning math models:** A Gemini model with 3.3B parameters pre-trained on public math datasets and then fine-tuned on AG data performs "on par with smaller models and the 3.3B model trained from scratch." However, the paper notes that even though performance is comparable, "they do produce slightly different auxiliary points proposals" — the from-scratch model and the fine-tuned model, despite training on the same data, develop slightly different patterns of which constructions they suggest for which problems. This diversity is exploited by the SKEST search algorithm, where multiple models with different training histories contribute to the ensemble.

##### Multimodal Training (Appendix C)

The multimodal model receives both the problem text and a diagram image as input. Despite "promising results during the training" (lower loss), **no improvement in IMO solve rate** is observed when using the multimodal model alone. The paper hypothesizes three reasons:

- **Diagram crowding:** IMO problem diagrams are very complicated and become "very crowded," making the visual signal noisy.
- **Image tokenization artifacts:** The process of splitting the diagram into "independent sequential patches" for transformer processing leads to "loss of some spatial information." Fine-grained spatial relationships that matter for geometry (e.g., whether a point lies inside or outside a circle, or which side of a line a point is on) may be scrambled by the patch-based tokenization.
- **Text already encodes visual information:** The analysis string already provides topological information through predicates like `sameclock`, and the DDAR engine has access to diagram coordinates. So the image may be redundant rather than complementary.

Despite this, the multimodal model produces "slightly different auxiliary point proposals" and contributes to the SKEST ensemble, where its different behavior adds diversity.

##### Full Proof Generation Capacity (Appendix F)

Although the inference system only uses the LM for auxiliary constructions, the model is trained on complete proofs. The paper tests whether it can generate full proofs without the symbolic engine. A step-verification system is built: for each generated proof step, the system extracts the premises, runs a focused DDAR with only the relevant deduction rule, and checks whether the conclusion follows. Steps are classified as verified (all checks passed), unverified (premises don't imply conclusion under the stated rule), or having various errors (wrong grammar, invalid aux point, numerical error, etc.).

The result (Figure 16): the models "almost do not make any syntax errors," and the majority of generated proof steps are either verified or correct-but-unverified (meaning the reasoning is geometrically sound but the specific rule citation is wrong). Small and larger models perform similarly on this metric. However, "without any further tuning, the model cannot generate complete full proofs" for IMO problems — getting every step correct throughout a 20–50 step proof remains beyond the current models. The paper frames this as "support[ing] the idea that large language models can be self-sufficient without depending on external tools, but until inference speed is improved and hallucinations are completely resolved, the tools will stay essential for math applications."

---

#### The SKEST Search Algorithm (Section 5 of the Paper)

SKEST (Shared Knowledge Ensemble of Search Trees) is the test-time search algorithm that orchestrates all components. It is not a single search but a collection of parallel, differently-configured beam searches that communicate through a shared database of discovered facts.

##### The Basic Search Tree (Classic Search)

Each search tree is a beam search over auxiliary constructions. A node corresponds to one attempt at an auxiliary construction followed by one DDAR2 run. The procedure at each node:

1. **Query the LM:** The current problem state (original premises + any auxiliary constructions from ancestor nodes + the analysis string) is presented to the language model, which samples one or more candidate auxiliary constructions.
2. **Run DDAR2:** Each candidate construction is added to the premises, and DDAR2 computes the deduction closure.
3. **Check success:** If the goal fact appears in the closure, the problem is solved and all search trees terminate. The proof is the sequence of auxiliary constructions from root to this node.
4. **Score and rank:** If the goal is not reached, the node is scored based on some heuristic (e.g., how many new facts were deduced, how close the new facts are to the goal). The top-scoring nodes are kept for the next level of the beam.
5. **Populate shared database:** Facts deduced at this node that are about the *original problem* (not specific to the auxiliary construction) are written to the shared facts database. This is the knowledge-sharing mechanism.

##### Diversity of Search Strategies

The paper deliberately runs multiple search trees with different configurations to explore different parts of the search space:

- **"Classic" search tree:** The same beam search used in AG1. The LM proposes one auxiliary point per node. This tree progresses incrementally, one aux point at a time.

- **Multiple auxiliary points per node:** The LM is allowed (and prompted) to generate as many auxiliary points as it wants in a single generation. Recall that the LM is trained on full proofs which include multiple aux points sequentially, so generating multiple points at once is within its training distribution. The paper notes that "even though we want our models to generate all necessary auxiliary points in one query, in practice, we observe the need to call the model multiple times given previously produced auxiliary points." Allowing multiple aux points per node "accelerates finding a solution and effectively increases the tree search depth."

- **Uniform aux point type sampling:** Normally, the LM is prompted with the token `x00` and generates the rest of the aux point specification (e.g., `x00 a : cong a b c d (000) coll a e f (001)`). In this variant, the system explicitly forces the LM to generate different types of initial predicates: `x00 a : cong`, `x00 a : coll`, `x00 a : cyclic`, `x00 a : perp`, etc., ensuring uniform coverage across the different geometric construction types rather than letting the LM's learned prior dominate.

- **Deep-but-narrow tree:** Beam size 64 and depth 10. This configuration explores long chains of carefully-selected auxiliary constructions, betting that a deep sequence of moderately-good steps leads to a solution.

- **Shallow-but-wide tree:** Beam size 512 and depth 4. This configuration explores many different initial constructions but does not go deep, betting that a single creative initial construction can solve the problem quickly.

##### The Knowledge-Sharing Mechanism

This is the key innovation that distinguishes SKEST from a simple ensemble. When a node in any search tree runs DDAR2 and fails to prove the goal, it may still have derived new facts. Some of these facts are **auxiliary-point-specific** — they depend on the particular constructed point and are useless to other searches exploring different constructions. But others are **problem-relevant** — they are facts about the original points and predicates that DDAR2 could *not* prove without auxiliary constructions, but *did* prove once a particular aux point was added.

The filtering criterion is: a fact is "interesting" and worth sharing if it involves only points from the original problem statement (not the auxiliary constructions) AND it was not in the original $S_1$ (the facts DDAR2 could deduce without any aux points). These are precisely the facts that required auxiliary constructions to prove, and they represent genuine progress on the problem.

These filtered facts are written to the **shared facts database**, which is readable by all nodes in all search trees. When a search tree expands a node, it includes these shared facts in the analysis string passed to the LM and in the premises passed to DDAR2. This means:

- **LM benefits:** The language model sees what other search trees have discovered and can use that information to propose better auxiliary constructions.
- **DDAR2 benefits:** The symbolic engine starts from a richer set of premises (original + all shared facts), so its deduction closure is larger and may reach the goal without needing additional aux points.
- **Search efficiency:** Multiple search trees are not redundant — they collaborate. A discovery made by a shallow-wide search (which explores many different initial constructions) can be exploited by a deep-narrow search (which builds long chains from promising starting points), combining their complementary strengths.

##### Asynchronous Execution Architecture

The system is designed for maximum parallelism and resource utilization:

- **LM workers** run asynchronously on TPUv4, with multiple replicas per model serving different search trees under their own search strategies. Each LM worker writes the content of the nodes it explores to a database.
- **DDAR2 workers** run asynchronously (likely on CPU, since DDAR2 is C++ and "finishes in 3.44711 seconds" on the benchmark), continuously picking up nodes from the database and attempting them. The DDAR2 workers coordinate among themselves to divide work equally.
- **A single DDAR2 worker pool is shared across different problems** when multiple problems are being solved simultaneously. If a problem is solved early, its DDAR2 resources are freed and reallocated to the remaining unsolved problems. This is an important practical optimization for batch evaluation.
- **Search controllers** monitor results asynchronously. When a DDAR2 worker reports success for any node in any search tree, all controllers receive the termination signal and all search stops.

##### Performance Scaling and Hyperparameters (Figure 9)

For a single search tree (without the full SKEST ensemble), the paper ablates beam size, beam depth, and number of samples:

- **Beam size:** Sweeping from 20 to 512, the optimal is 128 (solving the most IMO problems). Larger beams (512) do not help — the additional nodes explored are not productive enough to justify the compute.
- **Beam depth:** Sweeping from 20 to 24, the optimal is 4. Deeper trees (more sequential aux points) do not improve solve rate, suggesting that most problems solvable by AG2 require at most 4 auxiliary constructions in sequence.
- **Number of samples:** Sweeping from 20 to 32, the optimal is 32. More samples (higher diversity of aux point proposals per node) do not help.

These numbers are for a single search tree; the full SKEST ensemble combines multiple trees with different configurations, with the specific configurations chosen to cover complementary regions of the search space (shallow-wide vs. deep-narrow, single-aux vs. multi-aux, uniform vs. unforced sampling).

---

#### Summary of Design Choices and Their Justifications

- **Synthetic data only, no human demonstrations:** Eliminates data contamination risk, allows arbitrary scaling, and enables discovering solution strategies beyond human-taught ones. Contrasts with TongGeometry which uses human-crafted diagrams to guide generation.
- **Greedy pruning over exhaustive search:** Makes minimal-problem extraction feasible for 2× larger diagrams; exploits monotonicity of DDAR provability with reverse-topological ordering to handle dependency closure.
- **50:50 auxiliary/no-auxiliary data balance (vs. 9:91 in AG1):** Directly addresses the LM's primary test-time task (predicting aux points) by ensuring half the training examples require this skill rather than being drowned in trivial no-aux proofs.
- **C++ Gaussian elimination over Python:** The 300× speedup makes 300M-example training and aggressive test-time search computationally feasible; the elimination engine is the inner loop of both data generation and proof search.
- **Shape hashing over combinatorial search:** Reducing $O(N^8)$ triangle matching to $O(N^3)$ transforms DDAR from a research prototype that strains on >15-point diagrams to a production system that handles 30+ point IMO diagrams in seconds.
- **Double-point handling via `overlap`:** Enables the reformulation proof strategy that is essential for hard IMO problems; without it, many proofs are simply unreachable regardless of search budget.
- **Analysis string over raw problem statement:** Giving the LM context about what is known ($S_1$), what follows from assuming the goal ($S_2$), and what is numerically true ($S_3$) provides problem-specific guidance that turns random auxiliary construction into informed conjecture.
- **Top-k sampling at $t=1.0, k=32$ over greedy decoding:** Required for diversity — greedy decoding solves only 2/26 aux-requiring problems; this is evidence that auxiliary construction prediction is inherently multi-modal (multiple different constructions can solve the same problem and the model must explore them).
- **Ensemble of diverse search strategies over single beam search:** Different search trees (classic, multi-aux, forced-uniform, deep-narrow, shallow-wide) cover complementary regions of the search space, and the knowledge-sharing mechanism ensures they don't simply duplicate work but rather amplify each other's partial progress.
- **Asynchronous LM and DDAR workers over synchronous execution:** Decouples the neural component (which benefits from batch inference on TPUs) from the symbolic component (which is CPU-bound and benefits from parallel independent execution), maximizing hardware utilization.

## 4. Key Insights and Innovations

### Innovation 1: Neuro-Symbolic Theorem Proving Scales Gold-Medalist When All Components Scale Together — The Architecture *Is* the Contribution

The deepest intellectual move in AG2 is not any single algorithm but the demonstration that Olympiad-level geometry theorem proving yields to a **joint scaling attack** on every bottleneck simultaneously: domain expressiveness, verifier speed, model capacity, and search diversity. Each of these improvements individually is incremental engineering. Their combination — and the paper's argument that they *must* be combined to cross the gold-medalist threshold — constitutes a methodological claim with implications beyond geometry.

**What the field assumed before.** The dominant assumption, crystallized in AG1's architecture (Trinh et al., 2024), was that neuro-symbolic geometry solving followed a **diminishing-returns curve**: initial gains from a language model suggesting auxiliary constructions were large (AG1 jumped from 14 to 25 out of 30 on IMO-AG-30), but further improvements would come from refining the search strategy (TongGeometry; Zhang et al., 2024) or augmenting the symbolic engine with algebraic methods (Sinha et al., 2024). Each subsequent system optimized *within* its representational envelope — the set of problems expressible in AG1's nine-predicate, constructive-points-only language. No prior work questioned whether the envelope itself was the binding constraint.

**What AG2 reframes.** The paper restructures the problem from "optimize search for a fixed solver" to **"scale four independent dimensions until none is the bottleneck."** The evidence for this reframing is the solve-rate trajectory: AG1 + AG1 domain language = 54% on IMO-AG-50; AG2 DDAR alone = 32% (16/50); AG2 DDAR + AG1-level language model + single search tree = 76% (38/50); AG2 full ensemble = 84% (42/50). Each component contributes non-redundantly. The 88% domain language coverage (up from 66%) is not a performance gain but a **capability expansion** — problems that were literally invisible to prior systems now become attempted. The 300× DDAR speedup is not just "faster" but **causal** for both the larger training dataset (300M theorems, an order of magnitude more than AG1's 100M) and the aggressive multi-tree search that the SKEST algorithm orchestrates. The 3.3B-parameter Gemini model is not just "bigger" but a fundamentally different training paradigm (sparse mixture-of-experts, math pre-training, natural language compatibility) that the paper shows is *interchangeable* with the custom-transformer approach — meaning the capability comes from the training data, not the model architecture.

This makes AG2 a **systems argument**: the neuro-symbolic architecture itself — not any one component — is what crosses the gold-medalist threshold. The paper's title ("Gold-medalist Performance in Solving Olympiad Geometry with AlphaGeometry2") is deliberately not "Gemini Solves Olympiad Geometry" or "SKEST: A Novel Search Algorithm for Geometry." It attributes the achievement to the integrated *system*, and the paper's structure — domain language → symbolic engine → training data → language model → search algorithm — reflects this thesis by walking through each layer as a necessary contributor. This is significant because it suggests that future progress on hard reasoning domains will come not from a single breakthrough but from identifying and jointly attacking all binding constraints, a lesson that transfers to program synthesis, formal verification, and scientific discovery.

---

### Innovation 2: The "Verifier-as-Bottleneck" Principle — Search Strategy Is Constrained by Deduction Closure, Not Creativity

Hidden within the paper's engineering narrative is a sharp diagnostic claim about *where* the difficulty in neuro-symbolic reasoning actually lies. The language model is not the bottleneck. The search algorithm is not the bottleneck. The bottleneck is **how much the symbolic engine can deduce from a given set of premises and auxiliary constructions** — because the LM's job is only to propose constructions that *expand* the deduction closure until it contains the goal, and if DDAR cannot close the gap even with good constructions, no amount of LM creativity helps.

**What the field assumed before.** A natural intuition — reinforced by the success of LLMs in creative tasks — is that the "hard part" of geometry is the creative leap: envisioning the auxiliary point, line, or circle that unlocks the proof. AG1's framing leaned into this, positioning the language model as the creative component and the symbolic engine as the verifier. TongGeometry improved search for these creative leaps. The implicit assumption was that if you could just generate diverse enough constructions, the symbolic engine would eventually find a proof.

**What AG2 reframes.** The paper provides systematic evidence that the binding constraint is the *deductive power of the symbolic engine*, not the creativity of the LM. Consider the evidence structure:

- **DDAR alone solves 16/50 IMO problems** (Figure 8, Table 4). These are problems solvable with zero auxiliary constructions — pure deduction closure from the premises. Any LM proposal on these problems is unnecessary.
- **The remaining problems are partitioned by what DDAR lacks.** The two unsolved problems that AG2 attempted but failed (IMO 2018 P6, IMO 2023 P6) are explicitly attributed to missing deduction machinery: "inversion, projective geometry or radical axis, which are not implemented in our current DDAR." The paper is not saying "the LM couldn't think of the right construction" — it is saying that *even if the LM proposed the correct construction*, DDAR cannot verify the proof because the necessary deduction rules don't exist. The bottleneck is verifier capability, not proposer creativity.
- **The analysis string interface** (Section 6.2) is designed around this insight. The three sets $S_1$, $S_2$, $S_3$ are DDAR-computed facts, and the entire purpose of feeding them to the LM is to narrow the search for constructions that will expand the closure toward the goal. This only works because DDAR's deduction is the limiting factor — if the LM could hallucinate correct proofs without DDAR, the analysis string would be unnecessary.
- **Full proof generation fails** (Appendix F). The LM, trained on complete proofs, cannot generate correct 20–50 step proofs without DDAR verification at each step. The majority of individual steps are verifiable, but the chain breaks. The conclusion: "until inference speed is improved and hallucinations are completely resolved, the tools [symbolic engines] will stay essential for math applications."

This reframes the neuro-symbolic architecture from "LM proposes, symbolic engine verifies" to **"symbolic engine's deductive reach determines what the LM needs to propose."** The creative burden on the LM is only to bridge the gap between DDAR's unaided deduction closure and the goal — and the size of that gap is determined by DDAR's rule set, not by the problem's intrinsic difficulty. As DDAR improves (through rules for inversion, projective geometry, etc.), the gap shrinks, the burden on the LM decreases, and unsolved problems become solvable. This principle is transferable: in any neuro-symbolic system, **invest in the verifier first**, because the proposer is only as useful as the verification envelope allows.

---

### Innovation 3: SKEST as Collaborative Search — Knowledge Sharing Makes Parallel Search Superlinear

The SKEST algorithm (Shared Knowledge Ensemble of Search Trees) is not just an ensemble — it is a **collaborative search architecture** where multiple search trees running different strategies share intermediate discoveries through a filtered facts database, enabling each tree to build on partial progress made by others. This transforms search from a race between independent explorers into a cooperative process that can solve problems unreachable by any single strategy alone.

**What the field assumed before.** Standard approaches to parallelizing neuro-symbolic search fall into two categories. The first is **ensemble diversity**: run multiple independent searches with different random seeds or models and hope one finds a solution. This improves robustness (if one search fails, another might succeed) but offers no mechanism for one search to benefit from another's intermediate work — each search starts from scratch and either succeeds or fails independently. The second is **larger single search**: increase beam width or sample count within one search tree. This explores more candidates but is limited by the strategy's inherent bias — a deep-narrow beam search can never recover from a poor early choice, no matter how many parallel beams it runs.

**What SKEST reframes.** The key conceptual move is the **filtered facts database** (Section 5). When a DDAR worker evaluates a candidate auxiliary construction and fails to prove the goal, it still produces a deduction closure. Some facts in that closure are *auxiliary-point-specific* (they depend on the particular constructed point), but others are *problem-relevant* — they involve only original problem points and were not derivable by DDAR without an auxiliary construction. These are exactly the facts that represent genuine progress on the problem, and they're useful to any other search tree regardless of what construction produced them. By extracting and sharing these facts, SKEST decouples **discovery** from **exploitation**: a shallow-wide search that explores many diverse initial constructions can discover useful intermediate facts (e.g., "points A, B, C are concyclic"), and a deep-narrow search that builds long chains from promising starting points can exploit those facts as premises, potentially reaching the goal in fewer steps than either strategy could alone.

The paper's evidence for the value of diversity is the finding that models trained differently (from-scratch vs. fine-tuned on math data, domain-language vs. natural-language, text-only vs. multimodal) produce "slightly different auxiliary point proposals" despite comparable aggregate performance (Appendix B, Appendix C). Individually, they're equivalent. Combined through SKEST, their differences create exploration diversity without sacrificing the ability to share progress — a property that is *superlinear* in the number of search trees because discoveries compound rather than competing.

This is a fundamental advance in search architecture for neuro-symbolic systems because it addresses the **credit assignment problem** that makes beam search fragile: a partial exploration that doesn't reach the goal is normally wasted compute. SKEST salvages that compute by extracting and propagating intermediate results, effectively amortizing failed searches into progress for the ensemble. The implication is that future neuro-symbolic systems should not merely parallelize search but should design explicit knowledge-sharing protocols — the filtered facts database is one such protocol, but the principle generalizes to any domain where intermediate deductions are reusable across search paths.

---

### Innovation 4: Surprising Negative Results — Tokenizer Independence, Domain-Language Independence, and the Failure of Multimodal Vision

The paper contains three carefully-controlled negative results that challenge common assumptions in the field and have implications beyond geometry theorem proving. These are not failures of AG2 — they are **diagnostic findings** that clarify what matters and what doesn't in neuro-symbolic reasoning systems, and they're arguably as valuable as the performance gains.

**Tokenizer independence (Appendix B).** The paper finds that AG language models perform identically on IMO problems whether trained with a custom word-level tokenizer (vocabulary of a few thousand tokens, where each token is a complete semantic unit like `coll` or `cyclic`) or with the standard Gemini subword tokenizer (vocabulary of 300k tokens). This challenges the influential view (Singh and Strouse, 2024) that tokenizers are a major bottleneck in mathematical reasoning — the argument that subword tokenization fragments numbers and symbolic expressions in ways that prevent models from learning algebraic structure. AG2's finding suggests that when the training data is sufficiently large and structured (300M synthetic theorems), the tokenizer choice washes out — the model learns the relevant representations regardless of tokenization granularity. This doesn't refute the tokenizer-bottleneck hypothesis for general mathematical reasoning (where training data is far messier), but it establishes a boundary condition: for formal, domain-specific languages with high data volume, tokenizer design is not a critical hyperparameter.

**Domain-language independence (Appendix B).** When AG2 training data is translated from the formal domain-specific language into natural language ("a, b, c are collinear" instead of `coll a b c`), and a model is trained on the natural-language version, it achieves the same IMO solve rate. This is striking because the formal language was deliberately designed to be precise and unambiguous — no parsing ambiguity, no anaphora resolution, no natural language variation. The finding that natural language works equally well suggests that the geometric reasoning patterns themselves, not the formalism encoding them, are what the model learns. This has direct implications for autoformalization research (Appendix G): if models can learn geometry reasoning from either formal or natural language, then the translation step from natural language to formal language may be eliminable — a model could, in principle, reason directly in natural language and call a symbolic verifier that translates internally, or a future system could train end-to-end on natural-language geometry data.

**Multimodal vision doesn't help (Appendix C).** Despite "promising results during training," the multimodal Gemini model that receives both problem text and diagram image achieves **no improvement** in IMO solve rate when used alone. The paper attributes this to diagram crowding (IMO diagrams are complex and information-dense), patch-based tokenization artifacts (spatial relationships critical for geometry may be lost in the tokenization process), and redundancy (the analysis string already encodes topological information via `sameclock` predicates, and DDAR has access to diagram coordinates). This is a significant negative result because there is widespread intuition — and evidence from human cognition (Tversky and Suwa, 2009, cited in Appendix A) — that visual reasoning should aid geometry. The paper cites Chae et al. (2024) showing that vision-language models have "poor atomic visual skills," and the AG2 result provides a concrete instance: even a state-of-the-art multimodal model, when given diagrams for problems it *can* solve from text alone, gains no benefit. The paper is careful: the multimodal model does contribute diversity to the SKEST ensemble, so it's not useless. But as a standalone capability, vision doesn't help — geometry reasoning is primarily algebraic, not visual, and diagrams may be more useful for human intuition than for machine deduction.

These three findings together constitute a **principled narrowing of the design space** for future neuro-symbolic systems: invest effort in data scale and quality, not tokenizer design; formal languages are a convenience for verifiers, not a necessity for learners; and visual input is unlikely to be a high-leverage addition for tasks whose core reasoning is algebraic. These are not obvious a priori, and the paper's documentation of them — in appendices, with controlled comparisons — elevates them from engineering anecdotes to scientific contributions.

---

### Innovation 5: Difficulty Through a Constructive Lens — The Remaining 12% of Problems Defines a Research Program

The paper's treatment of unsolved and unformalizable problems is not a concession but a **constructive diagnostic**: by precisely categorizing *why* problems are out of reach, the paper converts the remaining 16% (8 out of 50 IMO problems) into a structured research agenda with clear technical targets. This is a methodological contribution — using the system's architecture to map the boundaries of current capability — that is more valuable than simply reporting an 84% solve rate.

**The taxonomy of failures.** Of the 8 unsolved problems:
- **6 are unformalizable** in the AG2 domain language. These involve 3D geometry, inequalities, non-linear equations, and countably many points (problems parameterized by an arbitrary integer $n$). These are *representational* gaps — the language cannot express the problem statement, so no amount of search or model improvement can help.
- **2 are formalizable but unsolved** (IMO 2018 P6 and IMO 2023 P6). These are *deductive* gaps — the problems require advanced geometry techniques (inversion, projective geometry, radical axis) that DDAR2 does not implement. The paper explicitly states: "while such problems in theory can be solved without these techniques, such solutions would require longer inference time, longer proofs and more auxiliary constructions to make up for the lack of the aforementioned machinery."

**What this tells us about the system's limits.** The taxonomy reveals a clean separation: representational gaps cannot be addressed by scaling existing components; they require expanding the domain language. Deductive gaps *might* be addressed by scaling (longer search, more aux points) but are better addressed by adding deduction rules. The paper chooses the latter framing: "breaking problems into subproblems and applying Reinforcement learning approaches could close this gap." This is precise: RL could learn when to deploy missing techniques as learned strategies, even if DDAR doesn't implement them natively.

For the inequality problems (the largest category of unformalizable problems), Appendix H provides a detailed list of inequality deduction rules that would need to be incorporated — essentially, a specification for the next version of DDAR. This is unusual in ML papers, which typically hand-wave about future work. AG2 instead treats its failure modes as a requirements document for AG3, making the research trajectory explicit and falsifiable.

**Why this framing is innovative.** Most AI systems papers report performance on a benchmark and gesture vaguely at "future work to address remaining failures." AG2's approach is different: the system's modular architecture (domain language ↔ symbolic engine ↔ language model ↔ search) means that each failure can be traced to a specific component, and the fix for each failure has a specific technical form (extend the language, add deduction rules, scale training data, diversify search). This transforms the 84% solve rate from a final score into a **progress report on a well-defined engineering roadmap** — and the roadmap itself is part of the contribution. For a field that often celebrates benchmark saturation without understanding *why* the remaining problems resist solution, AG2's failure analysis is a methodological standard worth emulating.

## 5. Experimental Analysis

### Evaluation Methodology

- **Dataset.** The primary benchmark is IMO-AG-50, a set of 50 geometry problems translated from all IMO geometry problems appearing between 2000 and 2024 (45 original problems; some are split into two due to formalization specifics). A secondary evaluation set is IMOSL-AG-30, consisting of 30 problems drawn from the hardest end of IMO Shortlist geometry problems (2002–2022) that never appeared at the IMO but are formalizable in the AG2 language. All problems are manually translated from natural language into the AG domain-specific formal language. The paper also uses three held-out evaluation sets during LM training: "eval" (synthetic problems with and without aux points), "eval_aux" (synthetic problems with aux points only), and "imo_eval" (IMO problems previously solved by AlphaGeometry), on which perplexity loss is computed as a proxy metric.

- **Base model(s).** The language model component uses sparse mixture-of-experts Transformers based on the Gemini architecture (Gemini Team, 2024), trained at multiple scales: 51M, 176M, and 3.3B parameters ("3p3B" in Figure 5). Models are trained in three configurations: (1) from scratch with a custom word-level tokenizer in the AG domain-specific language; (2) fine-tuned from math-specialized Gemini models pre-trained on public math datasets, using natural-language translations of AG data; (3) multimodal training from scratch with an additional diagram image input. The full AG2 system combines multiple models of different sizes and training paradigms through the SKEST ensemble. The symbolic engine (DDAR2) runs on CPU (C++ implementation with pybind11), while LM inference runs on TPUv4.

- **Metrics.** The primary metric is **solve rate** — the fraction of problems for which the system finds a valid proof within the allocated compute budget. A proof is valid if DDAR2 can derive the goal predicate from the problem premises and the auxiliary constructions proposed by the language model. No partial credit is given; a problem is either solved (full proof found) or unsolved. During language model training, perplexity loss on the three evaluation sets is reported as a proxy metric, but the paper explicitly notes that "these are only proxy metrics" because perplexity is computed on entire proofs (including deduction steps the LM generates at training time but never uses at test time), and there may be multiple valid proofs for a given problem while perplexity is computed for only one specific solution. Results are reported as raw counts (e.g., "42 out of 50") and percentages.

- **Baselines.** The paper compares against multiple systems spanning neuro-symbolic, purely neural, and purely symbolic approaches:
  - **AG1** (Trinh et al., 2024): the predecessor neuro-symbolic system, evaluated on both IMO-AG-50 (54% solve rate) and IMO-AG-30 (25/30 solved).
  - **AG1 DDAR / AG2 DDAR**: the symbolic engine alone without any language model, solving 14/50 and 16/50 respectively.
  - **TongGeometry** (Zhang et al., 2024): a neuro-symbolic system with guided tree search, evaluated on IMO-AG-30 (30/30 solved) but not on the full IMO-AG-50; TongGeometry DD (symbolic only) solves 18/30 on IMO-AG-30.
  - **Wu's method hybrids** (Sinha et al., 2024): combining Wu's algebraic method with AG1 components, evaluated on IMO-AG-30 (27/30 with Wu + AG1) but not on IMO-AG-50.
  - **OpenAI o1** and **Gemini thinking**: purely neural LLM approaches, each scoring 0/50 on IMO-AG-50 (Table 4).
  - **Human medalist benchmarks**: average bronze (27.1/50), silver (33.9/50), and gold (40.9/50) medalist performances on IMO-AG-50, provided for calibration. On IMO-AG-30, the corresponding averages are 19.3, 22.9, and 25.9.
  - **AG2 with AG1 setup**: a single search tree configuration (38/50 on IMO-AG-50, 28/30 on IMO-AG-30), serving as an ablation isolating the contribution of the SKEST ensemble.

- **Generation budget / compute accounting.** The paper does not use a unified "generation budget" metric across all components. Language model inference cost is measured implicitly through beam search hyperparameters (beam size, beam depth, number of samples) and the number of search trees in the SKEST ensemble. Symbolic engine cost is measured through DDAR2 runtime (averaging ~3.45 seconds per problem on a benchmark of 25 unsolvable IMO problems, compared to ~1179.57 seconds for DDAR1). The paper does not provide total FLOPs or wall-clock time for full-system runs on IMO-AG-50, nor does it report the number of LM queries per problem. The only explicit compute budget reported is for the FLOPs-matched comparison in Section 7, which uses formulas for pretraining and inference FLOPs to compare test-time compute against pretraining compute at different ratios of inference-to-pretraining tokens (R = 0.16, 0.79, 22). The paper also reports that AG2 solved IMO 2024 P4 "within 30 seconds at IMO 2024," but this is a single-problem anecdote, not a systematic benchmark.

- **Cross-validation / statistical protocol.** The paper does not report cross-validation or statistical significance testing for the main IMO-AG-50 results. The 50-problem test set is treated as a fixed evaluation benchmark; the solve rate is reported as a count (42/50). No confidence intervals, error bars, or significance tests are provided for the solve rate. For language model training, the three evaluation sets (eval, eval_aux, imo_eval) are used to monitor perplexity during training, but these are not used for hyperparameter selection in a cross-validated manner. The ablation of inference hyperparameters (Figure 9) varies one parameter at a time while holding others fixed at default values (beam size 512, beam depth 4, 32 samples), reporting raw solve counts. The paper does not describe any hold-out procedure for selecting the SKEST ensemble configuration — the specific choice of search tree types appears to be designed based on the authors' understanding of complementary search strategies rather than a systematic hyperparameter optimization.

---

### Main Quantitative Results

#### Overall Solve Rate on IMO-AG-50 (Figure 8, Table 4)

The headline result: **AG2 full setting (multiple search trees with SKEST) solves 42 out of 50 IMO geometry problems from 2000–2024**, corresponding to an 84% solve rate. This surpasses the average IMO gold medalist benchmark of 40.9/50 (Figure 8, Table 4) and substantially improves on AG1's 54% (27/50). The 42 solved problems are broken down in Figure 8, which groups problems by status: "Solved by DDAR" (problems requiring zero auxiliary constructions — 16 problems), "Solved" (problems requiring auxiliary constructions from the LM — 26 problems), "Not solved" (2 problems: IMO 2018 P6 and IMO 2023 P6), and "Not attempted" (6 problems that are unformalizable in the AG2 language). The reduction from 14 (AG1 DDAR) to 16 (AG2 DDAR) purely-symbolic solves reflects the improved DDAR2 rule set and double-point handling.

The single-search-tree ablation ("AG2 with AG1 setup") solves 38/50 (76%), meaning the full SKEST ensemble contributes an additional 4 problems (8 percentage points) over the best single-tree configuration. This establishes that ensemble diversity and knowledge sharing provide gains beyond what a monolithic beam search can achieve.

On the harder IMOSL-AG-30 benchmark (Appendix E, Figure 15), AG2 solves **20 out of 30 problems**, with DDAR2 alone solving a subset. This set consists of problems nominated by IMO experts from the hard end of each year's shortlist but never selected for the actual IMO, making them systematically more difficult than the IMO-AG-50 set. The paper does not provide human baselines for this set.

Table 4 provides the side-by-side comparison with all baselines on IMO-AG-50 and IMO-AG-30:

| System | IMO-AG-50 solved | IMO-AG-30 solved |
|---|---|---|
| OpenAI o1 | 0 | 0 |
| Gemini thinking | 0 | 0 |
| AG1 DDAR (Trinh et al., 2024) | 14 | 14 |
| AG2 DDAR | 16 | 15 |
| TongGeometry DD (Zhang et al., 2024) | — | 18 |
| Average bronze medalist | 27.1 | 19.3 |
| Wu with AG1 DDAR (Sinha et al., 2024) | — | 21 |
| Average silver medalist | 33.9 | 22.9 |
| AG1 (Trinh et al., 2024) | 27 | 25 |
| Average gold medalist | 40.9 | 25.9 |
| Wu + AG1 (Sinha et al., 2024) | — | 27 |
| TongGeometry w/o value (Zhang et al., 2024) | — | 28 |
| AG2 with AG1 setup (single search tree) | 38 | 28 |
| TongGeometry full setting (Zhang et al., 2024) | — | 30 |
| **AG2 full setting (multiple search trees)** | **42** | **30** |

Two observations from this table: (1) TongGeometry achieves 30/30 on IMO-AG-30, matching AG2's perfect score on that subset, but TongGeometry was not evaluated on the full IMO-AG-50, making it impossible to compare coverage on the broader problem set; (2) the human gold medalist average drops from 40.9/50 on IMO-AG-50 to 25.9/30 on IMO-AG-30, while AG2 drops from 42/50 to 30/30 — both are at ceiling on the easier subset, with the discrimination coming from the full problem set.

#### Language Model Scaling and Training Dynamics (Figure 5, Figure 7, Figure 10)

**Model size scaling (Figure 5).** The learning curves for 51M, 176M, and 3.3B parameter models show that larger models achieve consistently lower perplexity on train, eval, and imo_eval sets across all training token counts. The 3.3B model reaches the lowest loss on all three evaluation sets. However, the paper does not provide a direct comparison of IMO solve rate vs. model size for a fixed search configuration — the relationship between perplexity improvement and downstream solve rate is not quantified.

**Training token scaling (Figure 7).** When a single language model is coupled with DDAR via the "classic" tree search (beam size 512, beam depth 4, 32 samples), the number of IMO-AG-50 problems solved grows from 25 at roughly $10^9$ tokens to 38 at roughly $10^{12}$ tokens. Interestingly, "AlphaGeometry2 can already solve 27 out of 50 problems after only 250 training steps with a batch size of 256, or around 200 million tokens." This indicates that a substantial fraction of the model's problem-solving capability is acquired very early in training, with the remaining gains requiring two orders of magnitude more tokens to achieve a modest additional improvement (27 → 38 problems). The paper does not report the solve rate for the larger token counts beyond 38, nor does it show whether the curve would continue to improve with further training.

**Pre-training vs. fine-tuning (Figure 10).** The learning curves comparing a 3.3B model trained from scratch on AG data versus a 3.3B model pre-trained on math data and then fine-tuned on AG data show that the pre-trained model has initially lower loss but "both converge to the same point after training for 200B tokens." This is the basis for the claim that fine-tuning and from-scratch training are equivalent in the limit, though the paper notes that the models "do produce slightly different auxiliary points proposals," providing diversity for the SKEST ensemble.

#### Inference Hyperparameter Ablation (Figure 6, Figure 9)

**Temperature and diversity (Figure 6).** The ratio of unique auxiliary construction samples as a function of sampling temperature shows a sharp transition: at $t = 0.0$ (greedy decoding), essentially zero diversity (all samples are identical). At $t = 1.0$, the ratio of unique samples is approximately 0.92 (nearly all samples are distinct). At $t = 2.0$, the ratio approaches 1.0, but the paper notes that higher temperatures "result in an increasing number LM outputs with a wrong domain language syntax." The chosen operating point ($t = 1.0$, $k = 32$) balances diversity against syntactic validity.

**Beam size, depth, and samples (Figure 9).** Starting from a default configuration (beam size 512, beam depth 4, 32 samples), the paper ablates each parameter:

- **Beam size:** Varying from 20 to 512, the optimal is 128, solving 29 problems. The default 512 solves 27 problems — larger beams actually *reduce* performance, likely because they force the search to allocate budget to low-quality beams that would have been pruned at smaller beam sizes.

- **Beam depth:** Varying from 20 to 24, the optimal is 4 at 24 problems solved. Depths of 20–21 solve 22 problems; depths of 22+ show a decline to 23 problems. The optimal depth of 4 suggests most AG2-solvable problems require ≤4 sequential auxiliary constructions.

- **Number of samples:** Varying from 20 to 32, the optimal is 32 at 24 problems solved. Lower sample counts (20) solve only 20 problems.

These numbers are notably lower than the 38 problems solved by "AG2 with AG1 setup" in Table 4, indicating that the single-tree configurations in Figure 9 use different (unspecified) default settings from the final "AG2 with AG1 setup" baseline. The paper does not explain this discrepancy — the Figure 9 ablation appears to use a weaker base configuration than the final single-tree results.

#### DDAR2 Speed Benchmark (Section 3.3)

On a benchmark of 25 IMO problems that DDAR cannot solve (selected from the IMO-AG-50 set), run 50 times on an AMD EPYC 7B13 64-core CPU:

- **DDAR1:** average runtime 1179.57 ± 8.055 seconds (~19.7 minutes)
- **DDAR2:** average runtime 3.44711 ± 0.05476 seconds (~3.4 seconds)

This represents a speedup factor of approximately 342×. The paper notes that "the average running time may vary depending on the machine status at different times." The benchmark selection (problems DDAR cannot solve) means these are hard problems requiring full deduction closure computation, making them a conservative test of the engine's worst-case performance rather than average-case.

#### Automated Formalization and Diagram Generation (Appendix G)

**Formalization coverage.** Using Gemini with a few-shot prompt combining manual translations of "several dozens" of geometry problems, followed by a second Gemini call to merge five sampled translations into a final answer, the automated formalization pipeline correctly translates 33 out of 44 formalizable IMO 2000–2024 geometry problems. "For easier geometry problems, it is very consistent and makes almost no mistakes." This 75% formalization success rate (33/44) is the rate at which the pipeline produces syntactically valid and semantically correct AG language problem statements — it is distinct from the 88% domain language coverage, which measures what fraction of problems *can be expressed* in the language regardless of whether an automated system can produce the correct expression.

**Diagram generation.** On the 44 IMO problems formalizable in AG language, the three-stage optimization pipeline (Adam gradient descent → Gaussian elimination → Gauss-Newton-Levenberg) successfully finds diagrams for 43 problems (97.7% success rate) when allowed to restart with new random initial configurations after failure. The paper notes that "43 / 44 problems got their diagram sequentially generated within 1 hour."

#### Domain Language Coverage (Section 2)

The expanded AG2 domain language covers 88% (44/50) of IMO-AG-50 problems, up from 66% (33/50) in AG1. The remaining 12% (6 problems) are unformalizable due to 3D geometry, inequalities, non-linear equations, and countably-many-point problems. This coverage increase means AG2 *attempts* more problems than AG1 — the solving rate on the formalizable subset is 42/44 = 95.5%, compared to AG1's 27/33 = 81.8%. The overall solve rate improvement from 54% to 84% thus reflects both the coverage increase (more problems attempted) and the solving-rate improvement within the attempted set.

#### Proof Step Validity (Appendix F, Figure 16)

When the language model is asked to generate complete proofs (not just auxiliary constructions) for IMO-AG-50 problems using 32 samples at temperature 1.0, the step verification system finds that:

- **Syntax errors are rare:** the models "almost do not make any syntax errors."
- **The majority of generated steps are valid:** either fully verified (premises imply conclusion under the stated rule with numerical check passing) or correct but "unverified" (DDAR cannot confirm the rule application but the step is geometrically sound).
- **Small and larger models perform similarly** on this metric.
- **Full proofs are not achieved:** "without any further tuning, the model cannot generate complete full proofs" — getting every step in a 20–50 step proof chain correct remains beyond the current models.

---

### Ablation Studies and Robustness Checks

- **Single search tree vs. SKEST ensemble (Table 4).** The full AG2 system (multiple search trees with knowledge sharing) solves 42/50, while the single-search-tree configuration ("AG2 with AG1 setup") solves 38/50. The 4-problem gap (8 percentage points) represents the contribution of ensemble diversity and the knowledge-sharing mechanism. On IMO-AG-30, the gap is 2 problems (30 vs. 28). The paper does not ablate the knowledge-sharing mechanism separately from the ensemble diversity — it is not possible to determine from the reported results whether the gain comes from having multiple diverse search trees (any one of which might independently find the solution) or specifically from the fact-sharing protocol.

- **Greedy vs. temperature sampling (Section 6.2).** With greedy decoding ($t = 0.0$, $k = 1$) and no tree search, the models solve 2/26 IMO problems requiring auxiliary constructions. With $t = 1.0$ and $k = 32$ (but still no tree search), this rises to 9/26. This establishes that stochastic sampling is essential for exploring the space of auxiliary constructions, but even with high diversity, a flat sampling approach (no tree search) leaves most problems unsolved. The tree search provides the structure needed to build on promising constructions.

- **Temperature sweep (Figure 6).** The ratio of unique samples as a function of temperature shows that $t = 1.0$ provides near-maximal diversity without excessive syntax errors. Lower temperatures ($t < 1.0$) produce insufficiently diverse auxiliary constructions. This justifies the chosen operating point but does not explore whether slightly higher temperatures with syntax filtering could achieve even better diversity-error tradeoffs.

- **Beam size, beam depth, sample count (Figure 9).** Each parameter is ablated independently while holding others fixed. The optimal configurations are beam size 128, beam depth 4, and 32 samples. Notably, larger beam sizes (512) and deeper trees (depth > 4) *reduce* performance, indicating over-optimization — expanding the search budget beyond these thresholds explores unproductive regions of the construction space. The paper does not perform a joint hyperparameter sweep (all three parameters optimized simultaneously), so the reported optima may not represent the true global optimum for a single search tree.

- **Custom tokenizer vs. standard Gemini tokenizer (Appendix B).** Models trained with a custom word-level tokenizer (vocabulary of a few thousand tokens) achieve the same IMO-AG-50 solve rate as models trained with the standard Gemini subword tokenizer (vocabulary of 300k tokens). This finding is described as "somewhat surprising" and challenges the hypothesis that tokenization is a bottleneck for mathematical reasoning in this domain.

- **Domain-specific language vs. natural language (Appendix B).** Models trained on AG data translated into natural language achieve the same IMO solve rate as models trained on the original domain-specific language. This is described as "somewhat surprising again" and suggests that the geometric reasoning patterns, not the formalism encoding them, are what the model learns.

- **Training from scratch vs. fine-tuning pre-trained math models (Appendix B, Figure 10).** A 3.3B Gemini model pre-trained on public math datasets and fine-tuned on AG data converges to the same perplexity as a from-scratch model after 200B tokens of AG training. Descriptively, both approaches yield comparable performance, but the models "do produce slightly different auxiliary points proposals," providing the diversity that the SKEST ensemble exploits.

- **Multimodal model standalone performance (Appendix C).** The multimodal model that receives both problem text and diagram image achieves "no improvements in the solve rate on the downstream IMO problems when using this model alone" despite lower training loss. However, the multimodal model contributes to the SKEST ensemble by producing "slightly different auxiliary point proposals." This is a negative result: adding visual input does not improve standalone reasoning capability, though it does increase ensemble diversity.

- **Full proof generation step validity (Appendix F, Figure 16).** When the LM generates complete proofs, step-level verification reveals that proof steps are mostly valid or geometrically sound, but chaining them into complete 20–50 step proofs fails. The models "almost do not make any syntax errors" — the failure mode is logical consistency across long chains, not grammatical competence. Small and larger models perform similarly.

- **Automated formalization coverage (Appendix G).** The Gemini-based formalization pipeline correctly translates 33/44 formalizable problems. This is not an ablation of AG2 itself but a robustness check on the feasibility of a fully-automated system. The 75% success rate is a lower bound — the remaining 11 problems may require better prompting, more few-shot examples, or iterative refinement.

- **Diagram generation robustness (Appendix G).** The three-stage optimization pipeline with random restarts finds valid diagrams for 43/44 formalized problems within 1 hour. The single failure case is not analyzed, leaving open whether it represents a fundamental limitation of the optimization approach or a hyperparameter issue.

- **AG2 vs. AG2 DDAR only (Table 4).** AG2 DDAR alone solves 16/50 problems (32%), while the full AG2 system solves 42/50 (84%). The 26-problem gap is attributable to the language model + search components. This establishes a lower bound on what the symbolic engine alone can achieve and quantifies the marginal contribution of the neural and search components.

- **DDAR1 vs. DDAR2 speed (Section 3.3).** The 342× speedup is measured on a specific benchmark of 25 hard IMO problems. The paper does not report speedup on easier problems, synthetic data generation workloads, or end-to-end search scenarios, so the real-world impact on total system throughput is not directly quantified — only the per-problem deduction closure time is compared.

---

### Critical Assessment

#### Central Claim: AG2 achieves an 84% solve rate on IMO-AG-50, surpassing an average gold medalist.

**What the experiments demonstrate:** The 42/50 solve rate is directly reported in Figure 8 and Table 4, and the gold medalist benchmark of 40.9 is also reported. The experiments clearly demonstrate that AG2 solves more IMO geometry problems from 2000–2024 than the average gold medalist, at least as measured by the formalized versions of those problems.

**What needs qualification:**

- **The 50-problem test set is small and fixed.** With only 50 problems, the difference between 42/50 (AG2) and 41/50 (approximately gold medalist average) is fragile — solving or failing any single borderline problem shifts the conclusion. No confidence intervals are provided, making it impossible to assess whether AG2's 42/50 is statistically distinguishable from the gold medalist average of 40.9/50. On a problem-by-problem basis, AG2 may solve problems that gold medalists miss and vice versa; the aggregate count masks the distribution of individual problem difficulty.

- **The gold medalist benchmark is an average, not a threshold.** The paper states that AG2 "surpasses an average IMO gold medalist for the first time" and that Sinha et al. (2024) "previously claimed to achieve a gold medalist performance, but it was done on a subset of IMO problems." This framing conflates "surpassing the average gold medalist solve rate" with "achieving gold-medalist-level performance." The average gold medalist solves 40.9/50 geometry problems, but an individual gold medalist might solve anywhere from 35 to 50 depending on the specific problems and the individual's strengths. AG2's 42/50 falls within the plausible range of gold medalist variance, not unambiguously above it.

- **Problem formalization may simplify or alter the task.** All problems are manually translated into the AG domain language, and "some problems are split into two due to the specifics of our formalization" (Section 7). The formalization process may strip away problem nuances (diagrams, natural language assumptions) that human contestants must interpret, potentially making the formalized version easier than the original. Conversely, some problems become two separate AG problems, inflating the denominator. The paper does not analyze whether formalization introduces systematic bias in problem difficulty.

- **The IMO-AG-50 set is curated.** It consists of geometry problems from 2000–2024, a period during which the nature of IMO geometry problems may have shifted (e.g., toward problems requiring computational methods that AG2 handles well, or away from 3D geometry that AG2 cannot represent). This temporal window was presumably chosen to match the period for which AG1 data was available, but it means the benchmark is not a random sample of Olympiad geometry.

#### Central Claim: The combination of domain language expansion, faster symbolic engine, better training data, stronger language model, and novel search algorithm enables the improvement from 54% to 84%.

**What the experiments demonstrate:** The paper provides ablation evidence for several individual components. The domain language expansion from 66% to 88% coverage is quantified (Section 2). The DDAR2 speedup of 342× is benchmarked (Section 3.3). The training data improvements are characterized through distribution comparisons (Figure 2). The SKEST ensemble vs. single-tree comparison shows a 4-problem gain (Table 4). The language model improvements are shown through perplexity curves (Figure 5) and the training token scaling curve (Figure 7).

**What needs qualification:**

- **Component contributions are not isolated.** The paper does not perform a systematic ablation where each component is removed or downgraded while holding others fixed, with the IMO-AG-50 solve rate as the dependent variable. We do not know, for example, what solve rate a system with AG2's domain language and DDAR2 but AG1's language model and search would achieve, or how much of the improvement comes from DDAR2 speed vs. DDAR2 algorithmic improvements vs. C++ implementation. The aggregate improvement from 27/50 (AG1) to 42/50 (AG2 full) is clear, but the marginal contribution of each component is not.

- **The training data improvement is correlational, not causal.** Figure 2 shows that AG2 training data has larger diagrams, longer proofs, more balanced predicate distributions, and a 50:50 aux/no-aux ratio compared to AG1's 9:91. However, there is no experiment showing that *these specific distributional changes* cause the solve-rate improvement. The data improvements are confounded with the language model improvements (Gemini vs. custom transformer, 300M vs. 100M examples) and the search improvements (SKEST vs. single beam search).

- **The "single search tree" ablation is not cleanly controlled.** The "AG2 with AG1 setup" baseline in Table 4 (38/50) uses a single search tree, but it is unclear which language model, which domain language, and which DDAR version it uses. Presumably it uses the AG2 language model and DDAR2 but AG1-style search, but this is not stated. The comparison therefore conflates search strategy with other potential differences.

- **Figure 7 shows a curve, not a controlled experiment.** The training token scaling curve shows solve rate improving with more training, but this is observed for one specific model and search configuration. It does not demonstrate that *more training tokens cause higher solve rate* versus other factors that might correlate with token count (e.g., later checkpoints might benefit from better optimization dynamics).

#### Central Claim: The SKEST knowledge-sharing mechanism enables collaborative search where trees benefit from each other's partial progress.

**What the experiments demonstrate:** The 4-problem improvement from single-tree (38/50) to multi-tree SKEST (42/50) is evidence that the ensemble provides gains beyond what a single search tree can achieve.

**What needs qualification:**

- **Knowledge sharing vs. ensemble diversity is not disentangled.** The 4-problem gain could arise from (a) the knowledge-sharing mechanism specifically, (b) simple ensemble diversity (different trees explore different regions and one happens to find the solution), or (c) increased total compute (more trees = more aux construction attempts). The paper does not run a control where multiple search trees run independently *without* knowledge sharing to isolate the contribution of the sharing mechanism from the contribution of simply running more diverse searches.

- **No analysis of which shared facts matter.** The paper does not provide examples of specific facts that were shared between trees and enabled a solution that would otherwise not be found. Without such examples or statistics (e.g., "in X of the 42 solved problems, facts discovered by one search tree were critical to another tree's success"), the claim that knowledge sharing is the key mechanism remains hypothetical.

- **The ensemble configuration is hand-designed, not optimized.** The choice of search tree types (classic, multi-aux, forced-uniform, deep-narrow, shallow-wide) and the number of trees is based on the authors' intuitions about complementary search strategies. There is no systematic exploration of which combinations of tree types are most effective or how many trees are optimal. The ensemble may be suboptimal, or the gains may come from a subset of the tree types with others contributing little.

#### Central Claim: Language model improvements (Gemini architecture, larger dataset, enriched inference context) are essential to the performance gain.

**What the experiments demonstrate:** The learning curves (Figure 5) show that larger models achieve lower perplexity. The training token curve (Figure 7) shows solve rate improving with more training. The analysis string interface is described as an improvement over AG1's minimal input.

**What needs qualification:**

- **Perplexity and solve rate are only loosely coupled.** The paper explicitly acknowledges that perplexity is a "proxy metric" because it is computed on full proofs (which the LM never generates at test time) and uses one specific solution path when multiple exist. Figure 5 shows perplexity improvements; Figure 7 shows solve-rate improvements; but the relationship between them is not quantified. A model with better perplexity might generate more accurate auxiliary constructions, or it might simply memorize proof patterns better without improving construction quality.

- **The analysis string contribution is not ablated.** There is no experiment showing IMO solve rate with and without the analysis string context. We cannot determine from the reported results whether the $S_1, S_2, S_3$ enrichment actually improves construction quality or whether the model would perform similarly with just the problem statement.

- **Model size is confounded with training data scale.** The larger models (3.3B) are trained on the full 300M-example dataset, while the smaller models (51M, 176M) may be trained on the same data but for the same number of tokens, meaning they see fewer unique examples. The learning curves (Figure 5) show loss vs. tokens, not vs. unique examples, so the benefit of larger models may partially reflect their ability to absorb more data rather than their inherent capacity advantage.

#### Additional Weaknesses

- **No wall-clock or FLOPs budget for main results.** The paper reports that AG2 solved IMO 2024 P4 "within 30 seconds at IMO 2024," but this is a single-problem anecdote. No systematic wall-clock time, FLOPs count, or LM query count is reported for the IMO-AG-50 evaluation. Without this, it is impossible to assess the computational cost of the 42/50 solve rate or to compare efficiency with other systems (TongGeometry, Wu's method hybrids) or with human contestants (who have a 4.5-hour time limit for the entire IMO, not just geometry).

- **The 30-second IMO 2024 P4 result is ambiguous.** The paper states the solution "was obtained within 30 seconds at IMO 2024" — it is unclear whether this refers to wall-clock time on the competition hardware, time for the LM to generate auxiliary constructions, or total end-to-end time including formalization and DDAR. The context (IMO 2024) suggests this was the live competition setting, but the hardware and parallelism are not specified.

- **No comparison with IMO contest time limits.** Human contestants solve geometry problems as part of a 4.5-hour exam covering all topics (algebra, combinatorics, number theory, geometry), typically allocating ~1–1.5 hours to geometry. Without knowing AG2's per-problem compute budget, we cannot assess whether AG2's performance is competitive in terms of time efficiency, not just accuracy.

- **Formalization is manual for the main results.** The 42/50 solve rate is reported using manually formalized problems. The automated formalization pipeline (33/44) is a separate evaluation (Appendix G). An important open question is what the end-to-end solve rate would be for a fully automated system that formalizes natural-language problems and then solves them — the 75% formalization success rate would reduce the effective coverage substantially, though formalization errors (producing a valid but semantically wrong problem statement) are arguably more concerning than formalization failures (producing nothing).

- **The IMOSL-AG-30 evaluation lacks baselines.** The 20/30 solve rate on hard shortlist problems is reported without human baselines, without AG1 baselines, and without comparison system baselines (TongGeometry, Wu's method). We don't know whether 20/30 is impressive or expected relative to the difficulty of these problems, or how much of the shortlist difficulty distribution AG2 captures.

- **No evaluation on non-competition geometry.** All results are on IMO and IMO Shortlist problems. The system's ability to generalize to textbook geometry, construction problems, or geometric reasoning in non-competition contexts is untested.

- **The 4-problem SKEST gain may be fragile.** With only 50 test problems, the 4-problem difference between single-tree (38) and ensemble (42) could reflect a small number of problems that happen to benefit from a specific ensemble search tree configuration, rather than a general advantage of knowledge sharing. A larger test set would be needed to establish robustness.

- **No diversity analysis of the ensemble.** The paper states that models trained differently produce "slightly different auxiliary point proposals," but does not quantify this diversity (e.g., overlap in generated constructions, correlation in success/failure patterns across models). Without such analysis, it's unclear whether the ensemble gain comes from genuinely complementary behaviors or simply from increased total compute budget.

In summary, the experiments convincingly demonstrate that AG2 achieves a substantially higher solve rate on IMO geometry problems than its predecessor and that this improvement places it in the gold-medalist performance range. However, the experimental design prioritizes demonstrating the aggregate capability of the fully-integrated system over isolating the causal contribution of individual components. The most significant unaddressed questions are: (1) whether knowledge sharing specifically (vs. ensemble diversity or increased compute) drives the SKEST gain, (2) what the computational cost of achieving 84% actually is, and (3) how robust the results are to the specific 50-problem test set composition. These gaps are characteristic of large-scale systems papers where comprehensive ablation is prohibitively expensive, but they leave room for skepticism about the paper's causal claims regarding which specific innovations are most responsible for the performance improvement.

## 6. Limitations and Trade-offs

### 6.1 Unformalizable Problems Define an Absolute Capability Ceiling

**The assumption or constraint.** AG2's domain language, despite expanding coverage from 66% to 88% of IMO 2000–2024 geometry problems, cannot represent problems involving 3D geometry, inequalities, non-linear equations, or countably many points (problems parameterized by an arbitrary integer $n$). The paper explicitly acknowledges this: "our domain language does not allow talking about variable number of points, non-linear equations, and problems involving inequalities, which must be addressed in order to fully 'solve Euclidean geometry'" (Section 8). These 6 unformalizable problems are simply marked "Not attempted" in Figure 8 — they are not failures of the search algorithm or the language model, but **absolute representational boundaries** that no amount of compute or model improvement can cross.

**The consequence.** The headline 84% solve rate is computed over *formalizable* problems. On the full set of IMO geometry problems as they appear in natural language, the effective ceiling is 88% even if AG2 solved every formalizable problem perfectly. The 12% of problems that are invisible to the system include entire classes of geometric reasoning — inequalities (which often appear in the hardest IMO problems), 3D geometry (which has appeared in multiple IMOs), and problems with parametric generality (where the answer must hold for *any* number of points). These are not edge cases; they represent substantively different reasoning capabilities than the angle-and-ratio chasing that DDAR2 implements. A practitioner hoping to deploy AG2 on arbitrary geometry problems would encounter a hard failure (not a "try harder" failure) on any problem outside the language's expressiveness envelope, with no graceful degradation. Appendix H specifies the inequality rules DDAR2 would need to incorporate to handle these problems, underscoring that the gap is not small — it requires a substantial expansion of the deduction rule base and corresponding training data.

**What evidence exists in the paper.** Figure 8 directly labels 6 problems as "Not attempted" out of 50. Section 2 states the coverage increases from 66% to 88%. Section 8 explicitly lists 3D geometry, inequalities, non-linear equations, and countably many points as the remaining gaps. Appendix H provides a detailed specification of inequality rules (definitions of $\omega$, $\eta$, polygon membership, angle types, etc.) that would be required, effectively serving as a requirements document for the next version.

**Mitigation status.** The paper treats this as future work: "Extending AlphaGeometry to encompass these topics is a substantial undertaking that falls beyond the scope of this work" (Section 8). No partial mitigation (e.g., a fallback strategy or approximate reasoning for these problem types) is attempted. The limitation is architectural — the domain language must be redesigned to support new predicates and the symbolic engine must implement corresponding deduction rules — so incremental scaling of existing components cannot address it.

---

### 6.2 Missing Deduction Machinery Leaves Formalizable Problems Unsolved

**The assumption or constraint.** Even within the formalizable subset, AG2 cannot solve problems requiring advanced geometric techniques that DDAR2 does not implement: inversion, projective geometry, and the radical axis theorem. The two attempted-but-unsolved IMO problems (2018 P6 and 2023 P6) are attributed specifically to this gap. The paper states: "Two of the remaining unsolved IMO problems (IMO 2018 P6, IMO 2023 P6) involve advanced geometry problem-solving techniques such as inversion, projective geometry or radical axis, which are not implemented in our current DDAR" (Section 7). It further acknowledges that "while such problems in theory can be solved without these techniques, such solutions would require longer inference time, longer proofs and more auxiliary constructions to make up for the lack of the aforementioned machinery in our current DDAR, which hinders AlphaGeometry's current problem-solving capabilities" (Section 7).

**The consequence.** This limitation reveals a fundamental dependency structure in the neuro-symbolic architecture: **the language model cannot compensate for missing deduction rules**. The LM's job is to propose auxiliary constructions that expand DDAR's deduction closure until it contains the goal. If DDAR lacks the inference rules to close the gap even with perfect auxiliary constructions — because the gap requires reasoning steps that DDAR simply cannot execute — then the LM is powerless regardless of its creativity. The paper implicitly endorses this interpretation by attributing the failures to DDAR's missing machinery rather than to the LM's failure to propose the right constructions. This has a direct practical consequence: improving the system on these problems requires *symbolic engineering* (implementing new deduction rules and proving their correctness), not *neural scaling* (more training data or larger models). For a practitioner, this means that AG2's solve rate on a new distribution of problems depends critically on whether those problems rely on deduction techniques that DDAR2 implements — and there is no way around this dependency short of rewriting DDAR's rule base.

**What evidence exists in the paper.** Figure 8 labels both IMO 2018 P6 and IMO 2023 P6 as "Not solved." Table 4 shows the full AG2 system solving 42/50, with 2 problems attempted but unsolved. Figure 15 (IMOSL-AG-30) shows that out of 30 hard shortlist problems, 10 are unsolved, several of which likely involve similar missing techniques (though the paper does not provide a per-problem diagnosis for the shortlist set). The discussion in Section 7 provides qualitative attribution.

**Mitigation status.** The paper hypothesizes that "breaking problems into subproblems and applying Reinforcement learning approaches could close this gap" (Section 8), suggesting a future direction where RL could learn to deploy missing techniques as learned strategies even without native DDAR support. However, no experiments in this direction are reported. The mitigation is entirely speculative.

---

### 6.3 Computational Cost Is Not Characterized, Making Deployment Feasibility Unknown

**The assumption or constraint.** The paper reports zero systematic measurements of the compute budget required to achieve the 84% solve rate. No wall-clock time, no FLOPs count, no number of language model queries, and no number of DDAR2 invocations are provided for the IMO-AG-50 evaluation. The only concrete timing number is the anecdotal "AG2 solved IMO 2024 P4 within 30 seconds at IMO 2024" (Section 7 and Appendix D), with no hardware specification or breakdown of what "within 30 seconds" includes (formalization? LM inference? DDAR? all of the above?). Section 3.3 reports that DDAR2 averages 3.45 seconds per problem on a 25-problem benchmark, but this is only the symbolic engine component, not the full search. The SKEST ensemble uses "TPUv4 to serve multiple replicas per model" (Section 5) with "different search trees within the same model querying the same server under their own search strategy," but the number of replicas, TPU-hours, or total queries is never stated.

**The consequence.** A practitioner cannot answer the most basic deployment question: **how much does it cost to run?** The 84% solve rate is a capability claim divorced from a cost claim. It is entirely possible that AG2 achieves this performance using computing resources that are orders of magnitude beyond what a human contestant has available (a human has roughly 1–1.5 hours for a geometry problem, with pencil and paper). The paper notes that AG2 with a single search tree solves 38/50 problems (Table 4), suggesting that the ensemble's 4 additional problems come at the cost of running multiple parallel search trees with multiple language model replicas — but the cost multiple (how many more FLOPs or TPU-hours the ensemble requires vs. the single tree) is unknown. Without cost data, comparisons with TongGeometry, Wu's method hybrids, or even human contestants are incomplete — we don't know whether AG2 is competitive on efficiency or achieves its win through brute-force compute scaling.

**What evidence exists in the paper.** The paper provides component-level speed data: DDAR2 is 342× faster than DDAR1 on a benchmark of 25 hard problems (Section 3.3), and DDAR2 averages 3.45 seconds per problem. Section 5 describes the asynchronous architecture (LM workers on TPUv4, DDAR workers on CPU, shared worker pools across problems) but never quantifies total cost. Section 6.1 notes "the largest possible batch size allowed by the hardware using TPUv4" for training but does not specify the batch size or hardware count. Figure 7 shows solve rate vs. training tokens for one model, which is a *training* cost proxy but does not capture inference cost. The 30-second IMO 2024 P4 anecdote (Section 7) is the only inference timing datum.

**Mitigation status.** Not addressed. The paper provides no framework for reasoning about inference cost, no normalization of solve rate by compute budget, and no guidance on how to select the ensemble configuration to balance cost and performance. The fact that multiple search trees, multiple LM replicas, and an asynchronous DDAR worker pool are all described qualitatively but never costed quantitatively is a significant gap for any practitioner considering deployment.

---

### 6.4 The 50-Problem Test Set Makes Performance Estimates Fragile

**The assumption or constraint.** All main results are evaluated on IMO-AG-50, a set of exactly 50 formalized geometry problems. The secondary evaluation, IMOSL-AG-30, adds 30 problems. With 80 total test problems and a headline solve rate of 84% (42/50), each individual problem contributes 2 percentage points to the solve rate. The paper provides no confidence intervals, no significance tests, and no cross-validation on the test set (the solve rate is reported as a raw count). The difficulty estimation section from other neuro-symbolic work — where the test set is split into folds to validate strategy selection — is entirely absent here. The specific problems that AG2 solves vs. fails may be highly sensitive to the chosen formalization (some problems are split into two, per Section 7), the particular Gemini model checkpoint, or the exact ensemble configuration.

**The consequence.** The claim "surpasses an average IMO gold medalist" rests on a 1.1-problem margin (42 vs. 40.9). A gold medalist's performance on IMO-AG-50 is itself an average, not a fixed threshold — individual gold medalists vary, and problem difficulty varies year to year. On a different 50-problem sample from the same distribution (say, 2025–2045 IMO problems), AG2's solve rate could easily be 38/50 or 44/50 without any change in the system. The lack of statistical characterization means we cannot distinguish the signal (AG2 is genuinely better than the gold medalist average) from noise (this particular 50-problem draw happens to favor AG2's capabilities). For a practitioner evaluating whether AG2 will work on *their* geometry problems, the absence of any uncertainty quantification means there is no basis for predicting performance on out-of-sample problems beyond the point estimate.

**What evidence exists in the paper.** Section 7 reports the result as raw counts (42/50). Table 4 provides comparison counts for all systems. Figure 8 is a problem-by-problem visualization but is used descriptively, not for statistical analysis. No standard errors, bootstrap intervals, or sensitivity analyses are reported. The paper does not discuss test set size as a limitation. The IMOSL-AG-30 evaluation (Appendix E, Figure 15) uses a different set of 30 problems, providing a second point estimate (20/30), but again without error characterization.

**Mitigation status.** Not acknowledged as a limitation. The paper treats the solve rate as a fixed quantity determined by the system's capabilities rather than as an estimate with sampling error. In fairness, this is standard practice in the theorem-proving literature (AG1, TongGeometry, and Sinha et al. all report solve rates as raw counts on fixed benchmarks), but the gold-medalist comparison makes the fragility particularly salient because the margin is small.

---

### 6.5 The Domain Language Formalization Is Manual for the Main Results

**The assumption or constraint.** The 42/50 solve rate is achieved on problems that were **manually translated** from natural language IMO statements into the AG domain-specific formal language. The paper states that "there are a total of 45 geometry problems in 2000-2024 IMO, which we translate into 50 AlphaGeometry problems" and notes that "some problems are split into two due to the specifics of our formalization" (Section 7). The automated formalization pipeline using Gemini, described in Appendix G, is evaluated separately and achieves only 33/44 (75%) on formalizable IMO problems — 11 problems cannot be automatically formalized. The main AG2 results bypass this bottleneck entirely by using human translators.

**The consequence.** The 84% solve rate is **not the performance that an end-to-end automated system would achieve**. In a fully-automated pipeline (natural language input → Gemini formalization → AG2 solving → proof output), the formalization step introduces a failure mode that compounds with solving failure. Even if AG2 achieved 100% solve rate on formalized problems, the automated system would be capped at the 75% formalization success rate. More subtle are formalization *errors* — Gemini may produce a syntactically valid AG language statement that does not accurately capture the original problem semantics. Such errors are harder to detect than outright formalization failures and could cause AG2 to "solve" a different problem from the one asked, producing a proof that is formally correct but answers the wrong question. The paper does not report whether any of the 33 successfully formalized problems contain semantic errors, nor does it verify that AG2's proofs on these machine-formalized statements correspond to correct solutions to the original natural-language problems. This is a critical gap for any claim of building a "fully automated system that reliably solves geometry problems from natural language input" (Abstract). The reliability of the automated pipeline is significantly lower than the curated manual-formalization results suggest.

**What evidence exists in the paper.** Appendix G reports the 33/44 automated formalization success rate and describes the approach (Gemini few-shot with merge step). The paper notes that "for easier geometry problems, it is very consistent and makes almost no mistakes," implying that harder problems are where formalization errors concentrate — exactly the regime where AG2's solving ability is most uncertain. The 43/44 diagram generation success rate (also Appendix G) is a separate metric for a different component. Section 8 frames the automated system as "progress towards" a fully automated pipeline, but the main results in Section 7 and Table 4 are all based on manual formalization.

**Mitigation status.** The paper treats automated formalization as ongoing work and is transparent that the main results use manual translation: "We start by manually translating several dozens of geometry problems into the AG language" (Appendix G). The separate evaluation of automated formalization (33/44) provides a realistic lower bound on end-to-end performance, but the paper does not report the end-to-end solve rate (formalization + solving) for the fully automated pipeline, which would presumably be lower than 42/44 formalizable problems × 33/44 formalized ≈ 70% on the formalizable subset, and lower still on all 50 problems. This transparency is commendable, but the Abstract's claim of "progress towards using AG2 as a part of a fully automated system" should be read with the 75% formalization ceiling in mind.

---

### 6.6 No Evidence That the Knowledge-Sharing Mechanism Specifically Drives the Ensemble Gain

**The assumption or constraint.** The SKEST algorithm's defining innovation is the **knowledge-sharing mechanism**: when a node in any search tree runs DDAR2 and fails to prove the goal, it extracts facts about the original problem (not specific to its auxiliary construction) that DDAR2 could not prove without aux points, and writes them to a shared database accessible to all other nodes in all other search trees. The paper claims this enables trees to "help each other" (Section 5) and that models with different training produce "slightly different auxiliary point proposals" that "do help each other via the knowledge sharing mechanism" (Appendix B). However, the paper never isolates this mechanism from the alternative explanation: **running more diverse search trees in parallel, even without sharing, increases the probability that at least one finds a solution simply through independent exploration.** The 4-problem gain from single-tree (38/50) to ensemble (42/50) could arise from knowledge sharing, from ensemble diversity alone, from increased total compute budget, or from any combination of these.

**The consequence.** The paper's most architecturally novel contribution — collaborative search through shared intermediate deductions — is supported only by correlational evidence. Without an ablation where multiple diverse search trees run in parallel *without* knowledge sharing, it is impossible to determine whether the sharing protocol adds value beyond what independent parallel search would achieve. If the gain is primarily from ensemble diversity (different trees explore different regions and one wins), then the knowledge-sharing database is an implementation detail rather than a conceptual advance. If the gain is primarily from increased compute (ensemble runs more total LM queries and DDAR2 evaluations), then the same solve rate could be achieved by simply running the single best tree for longer. The paper's argument that diversity matters is supported by the finding that differently-trained models produce different auxiliary constructions, but this diversity could be exploited by a flat ensemble without any communication between trees. The specific claim that knowledge *sharing* is the key mechanism remains unverified.

**What evidence exists in the paper.** Table 4 shows single-tree (38/50) vs. ensemble (42/50). Appendix B and C report that fine-tuned models, from-scratch models, and multimodal models produce "slightly different auxiliary point proposals" and "do help each other via the knowledge sharing mechanism." No ablation removes knowledge sharing while preserving ensemble diversity. No examples are provided of specific facts shared between trees that enabled a solution. No statistics are reported on how often shared facts are used by recipient trees. The paper provides no diagnostic analysis of the knowledge-sharing mechanism at all — how many facts are shared, what types of facts are most valuable, or whether trees that receive shared facts solve problems faster than those that don't.

**Mitigation status.** Not acknowledged as an open question. The paper treats the ensemble gain as evidence for the value of knowledge sharing, but the causal chain from sharing to gain is assumed rather than demonstrated. For a practitioner, this means the paper provides no guidance on whether implementing a shared facts database is worth the engineering complexity — it might be that a simpler approach (just run multiple diverse trees independently and stop when any one succeeds) achieves the same result with less infrastructure. This is a significant gap for a paper whose title-level contribution includes "a novel knowledge-sharing mechanism."

## 7. Implications and Future Directions
- Field impact
  - Demonstrates that a carefully engineered neuro‑symbolic stack—expanded DSL, very fast symbolic engine, and knowledge‑sharing LM search—can surpass expert human benchmarks on a long‑standing challenge set (Figure 8, Table 4). This strengthens the case for hybrid systems in formal reasoning tasks beyond geometry.

- Enabled research directions
  - Extend DSL to inequalities, non‑linear relations, variable‑size constructions, and 3D geometry to close the remaining 12% coverage.
  - Incorporate projective/inversion/radical‑axis modules and/or hybridize with algebraic methods (e.g., Wu’s method) to attack the hardest problems identified in Section 8.
  - Reinforcement learning over tree‑of‑subproblems: Section 9 suggests breaking problems into subproblems could boost success on complex cases.
  - Improve auto‑formalization with supervised fine‑tuning and larger exemplars, moving toward reliable natural‑language‑to‑proof pipelines (Section 9).
  - Develop faster, more faithful visual tokenization and spatial reasoning if multi‑modal gains are to materialize (Appendix B).

- Applications
  - Intelligent tutoring: step‑by‑step, verifiable geometric proofs and auxiliary constructions.
  - Proof assistants and graders for geometry competitions or coursework.
  - Theorem discovery and diagram synthesis from free‑form text (Sections 3 and 5).
  - Methodological transfer: the SKEST paradigm and “analysis string” interface (Section 7.2) could guide LMs with symbolic context in other domains (algebra, program verification).

> Headline result: “AG2 full setting solves 42/50 of all IMO 2000–2024 geometry problems” (Table 4; Figure 8), with engine runtime reduced by >300× (Section 4.3) and DSL coverage expanded to 88% (end of Section 2). These advances collectively shift the frontier of machine mathematical reasoning toward robust, automated understanding and solution of complex geometric problems.
