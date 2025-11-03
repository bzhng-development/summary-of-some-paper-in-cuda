# SUM OF THE GL(3) FOURIER COEFFICIENTS OVER QUADRATICS AND MIXED POWERS

**ArXiv:** [2310.11408](https://arxiv.org/abs/2310.11408)

## 🎯 Pitch

This paper breaks new ground in analytic number theory by establishing sharp, nontrivial upper bounds for short averages of GL(3) arithmetic functions—specifically, the Fourier coefficients of SL(3, Z) Hecke–Maass cusp forms and the triple divisor function—over highly sparse, structured algebraic sets such as mixed-power polynomials and binary quadratic forms. By leveraging an advanced arsenal of analytic techniques, including the DFI δ-method, GL(3) and d3 Voronoi summation, and stationary phase, the authors extract cancellation far beyond previously known results, yielding significant power savings in these challenging settings. These results push the frontier of our understanding of how arithmetic objects behave on thin sets, with implications for prime distribution in polynomial sequences and the theory of automorphic forms.

---

## 1. Executive Summary (2-3 sentences)
This paper develops new upper bounds for short averages of GL(3) arithmetic sequences—either the Fourier coefficients `Λ(1,n)` of an SL(3, Z) Hecke–Maass cusp form or the triple divisor function `d3(n)`—evaluated on sparse algebraic sets: mixed-power sums `n1^2 + n2^2 + n3^k` and binary quadratic forms `Q(n1, n2)`. The key advance is a careful use of the Duke–Friedlander–Iwaniec (DFI) δ-method, GL(3) and `d3` Voronoi summation, Poisson summation, and stationary phase analysis to extract cancellation beyond trivial bounds, yielding in particular (i) for `k=3`, a power-saving in `X` of size `X^{1/8}` over the trivial bound in the mixed-power case, and (ii) `X^{7/4+ε}` for the quadratic-form case when the second variable has length `Y = X^θ` with `θ > 3/4`, improving earlier bounds at equal lengths.

## 2. Context and Motivation
- Problem addressed:
  - Understand how arithmetic functions of GL(3) type—specifically the `(1,n)` Fourier coefficients `Λ(1,n)` of an SL(3, Z) Hecke–Maass cusp form, or the triple divisor function `d3(n)`—behave on sparse algebraic sequences, notably:
    - Mixed powers: sums of the form (eq. (1.5))
      - `Sk(X) = Σ_{1≤n1,n2≤X^{1/2}} Σ_{1≤n3≤Y} A(n1^2 + n2^2 + n3^k) a(n3)` with `Y = X^{1/k}`, `k ≥ 3`, `A(n)` = `Λ(1,n)` or `d3(n)`, and `a(n)` square-summable as in (1.4).
    - Binary quadratic forms: (eq. (1.7))
      - `S = Σ_{1≤n1≤X} Σ_{1≤n2≤Y} Λ(1, Q(n1, n2))` where `Q(x,y)` is positive definite, symmetric, integral.

- Importance:
  - Sparse averages of arithmetic functions over structured sets are central in analytic number theory, with applications to prime distribution in polynomial sequences, understanding cancellation in automorphic coefficients, and bounding moments of L-functions. Mixed-power and quadratic-form values are prototypical sparse sets with rich arithmetic structure.

- Prior work and gaps:
  - For divisor-type functions over polynomial sequences, there is substantial progress (Introduction, pp. 1–3). For example:
    - Over quadratics, Hooley obtained asymptotics for `d(n^2+a)`; over higher-degree sparse sequences, Friedlander–Iwaniec and Heath-Brown proved infinitude of primes in specialized polynomial families.
    - For `d3`, Zhou–Hu (AIMS Math. 2022) obtained (eq. (1.3))
      > `Σ_{1≤n1,n2≤X^{1/2}} Σ_{1≤n3≤X^{1/k}} d3(n1^2 + n2^2 + n3^k) = O(X^{1 + 1/k − δ(k) + ε})`
      with explicit `δ(k)`.
  - For GL(3) Fourier coefficients `Λ(1,n)`, far fewer results are known on such sparse sets; even linear polynomials are nontrivial, and quadratic or higher-degree polynomials are considerably harder.
  - The same authors previously obtained for the quadratic-form case with equal lengths (eq. (1.6)):
    > `Σ_{1≤n1,n2≤X} Λ(1, Q(n1,n2)) ≪ X^{2 − 1/68 + ε}`.
    That leaves room for improvement, particularly when the two variables have unequal lengths.

- Positioning:
  - This paper targets two regimes with unified analytic techniques:
    - Mixed powers with one small variable weighted by an `L^2`-bounded sequence `a(n)` (Theorem 1).
    - Binary quadratic forms with unequal lengths `X` and `Y = X^θ` (Theorem 2).
  - Methodologically, it uses the DFI δ-method, GL(3) and `d3` Voronoi summation, Poisson summation, and stationary phase to balance oscillation and arithmetic structure (Sections 3–5). Two notable deviations from standard δ-method treatments are highlighted in the Remarks after Theorem 2:
    - In Theorem 1, no summation formula is applied in the smallest variable.
    - In Theorem 2, the Cauchy–Schwarz inequality is avoided at a key step, enabling extra savings (Remark (i)).

## 3. Technical Approach
The paper analyzes two sums with a common toolkit. Below is a step-by-step outline of how the machinery is deployed and why each step matters.

A. Common ingredients (Section 2, Preliminaries):
- GL(3) Fourier coefficients `Λ(m,n)`:
  - Appear in the Fourier–Whittaker expansion (eq. (2.1)) of a Maass cusp form for SL(3, Z).
  - Known “Rankin–Selberg-type” second moment bound (Lemma 1, eq. (2.3)):
    > `Σ_{m^2 n ≲ X} |Λ(m,n)|^2 ≪ X^{1+ε}`.
- GL(3) Voronoi summation (Lemma 2, eq. (2.6)):
  - Transforms a sum `Σ_n Λ(m,n) e(an/q) g(n)` into dual sums over divisors `n1 | mq` and `n2 ≥ 1` with Kloosterman sums `S(…)` and integral transforms `G±(…)`.
  - The oscillatory behavior of `G±` is controlled by Lemma 4 (eq. (2.4)–(2.5)), giving a cosine-sine expansion with phase `± 3 (yz)^{1/3}`.
- Voronoi for `d3(n)` (Lemma 3; eqs. (2.7)–(2.9)):
  - Similar structure; this lets the same strategy handle `A(n) = d3(n)` when needed.
- Poisson summation (Lemma 5, eq. (2.10)):
  - Transfers sums over `Z^n` to dual sums, exposing additive characters that can be analyzed with Gauss/Kloosterman techniques.
- Exponential sums for quadratic forms (Lemmas 6–7):
  - Closed forms and bounds for Gauss/Kloosterman sums associated with quadratic forms and squares, enabling explicit evaluation/simplification of character sums coming from Poisson.

B. The DFI δ-method to detect polynomial identities (Section 3):
- Goal: enforce the algebraic constraint `r = P(n)` (here `P(n)` is a polynomial, such as `n1^2 + n2^2 + n3^k` or `Q(n1, n2)`) inside a multi-variable sum.
- Mechanism (Lemma 8, eq. (3.2)):
  - Replace the Kronecker delta `δ(r − P(n))` by a doubly smoothed exponential average:
    > `δ(r − P(n)) = (1/Q) Σ_{q=1}^∞ (1/q) Σ_{a mod q}^* e((r−P(n)) a/q) ∫ ψ(q,x) e((r−P(n)) x/(qQ)) dx`
  - Here `Q` is an optimization parameter (taken as `Q = X^{1/2}` in Theorem 1 and `Q = X` in Theorem 2), and `ψ(q,x)` is a controlled weight with decay and derivative bounds (eqs. (3.3)–(3.5)).

C. Theorem 1: Mixed powers (Section 4)
- Object: `Sk(X)` in (1.5) with `A(n) = Λ(1,n)` (the `d3` case is similar).
- Step 1 (δ-method injection; eq. (4.1)):
  - Insert the δ-expansion for `r = n1^2 + n2^2 + n3^k` and smooth functions `W1, W2, W3, V, U` to localize ranges; this yields an integral over `u` and sums over `q, a`.
- Step 2 (GL(3) Voronoi in the `r`-sum; §4.1, eqs. (4.2)–(4.5)):
  - Apply Lemma 2 to the `r`-sum; invoke Lemma 4 to approximate the transforms `G±` by oscillatory integrals with phase `± 3 ( (X z n^2 m)/q^3 )^{1/3}` after rescaling (eq. (4.3)).
  - A crucial support condition (by repeated integration by parts) emerges (eq. (4.4)):
    > The new sum over `m` is negligible unless `n^2 m ≲ K`, where `K := q^3/X + X^{1/2} u^3`.
    This reduces the range of `m`, a key saving.
- Step 3 (Poisson in the variables `n1, n2`; §4.2, eqs. (4.6)–(4.8)):
  - Write `n1 = α1 + l1 q`, `n2 = α2 + l2 q`, apply Poisson in `(l1,l2)` to get dual variables `(m1, m2)` and a character sum
    > `C(m1, m2, a; q) = Σ_{α1,α2 mod q} e( (−a(α1^2+α2^2) + m1 α1 + m2 α2)/q )` (eq. (4.9)).
  - The resulting integral transform `J(m1,m2,u,q)` is localized so that the `m1, m2` sums are effectively supported on `|m1|, |m2| ≲ X^ε` (by integration by parts).
- Step 4 (Evaluate the quadratic character sum explicitly; §4.4, eqs. (4.20)–(4.22)):
  - Using Lemma 6 (Gauss sums for quadratic forms) and Lemma 7 (1D Gauss sums), the double sum in `α1, α2` collapses to a simple explicit factor:
    > `C(m1, m2, a; q) = ε_q^2 q e( −4a(m1^2 + m2^2)/q )`, up to unit factors (eqs. (4.20)–(4.21)).
  - This removes dependence on `α1, α2` and simplifies the arithmetic.
- Step 5 (Cauchy–Schwarz in the `m`-sum; §4.4, eqs. (4.16)–(4.19)):
  - To manage `Λ(n,m)`, introduce:
    - `Θ = Σ_{m≤K/n^2} |Λ(n,m)|^2 m^{−2/3}` (eq. (4.18)),
    - `Ω = Σ_{m≤K/n^2} |Σ_{n3≤Y} a(n3) C_1(…)\, L_±(…)|^2` (eq. (4.19)),
    where `C_1` absorbs Kloosterman and additive phases (eq. (4.10)) and `L_±` are the oscillatory integrals over `u` (eq. (4.11)).
  - Lemma 9 bounds these integrals as `L_± ≪ q^{3/2}/Q^{3/2}`.
- Step 6 (Poisson in the `m`-sum and frequency split; §4.5–4.6):
  - Poisson-transform the `m`-sum inside `Ω`, producing dual frequencies `m` and a new character sum `S` (eqs. (4.24)–(4.26)).
  - Two cases:
    - Zero frequency (`m=0`): handled in Lemma 11 with `S0 ≪ q^4/n` under a congruence restriction `n3'^k ≡ n3^k (mod q)`.
    - Non-zero frequencies (`m≠0`): factor `q = q1 q2 q3` by localizing the `n`-part and squarefree/squarefull parts of `q3`. Use Weil bounds and—crucially—the Dąbrowski–Fisher stationary phase method for sums of products of Kloosterman sums (Section 4.6, culminating in Lemma 12), achieving
      > `S_{≠0}(q) ≪ q^{7/2} (q1 q2 q3'')^{1/2}/n` in the coprime case, and `≪ q^4/n` otherwise (eq. (4.37)).
  - These are the core arithmetic cancellations.
- Step 7 (Assembling bounds; §4.7–4.10):
  - Control the contribution of zero frequencies (Lemma 13) and non-zero frequencies (Lemma 15), plus the small-`n^2 m` “error” range (Section 4.9).
  - Optimize parameters:
    - Take `Q = X^{1/2}`; in the main range, `K ≍ X^{1/2}`.
  - Final bound (Section 4.10):
    > For `k=3`: `Sk(X) ≪ X^{7/8+ε} Y`.  
    > For `k≥4`: `Sk(X) ≪ X^{1+ε} Y^{1/2}`.
  - Both improve over the trivial bound `Sk(X) ≪ X^{1+1/k+ε}` (just summing absolute values via `|A(n)| ≪ n^ε` and (1.4)).

D. Theorem 2: Binary quadratic forms (Section 5)
- Object: `S = Σ_{n1, n2} Λ(1, Q(n1,n2)) W1(n1/X) W2(n2/Y)` with `Y = X^θ`, `0<θ≤1` (eq. (1.7)).
- High-level differences from Theorem 1:
  - The variables `n1` and `n2` have different lengths (`X` vs. `Y`), and `Q` has cross terms (`2C n1 n2`), so the Poisson step produces different integral scales and character sums.
  - The authors deliberately avoid a Cauchy–Schwarz step in a place where it typically appears; instead, they extract an extra (small) saving via partial summation (eq. (5.12)–(5.13)), which turns a boundary-case estimate into a power saving (Remarks after Theorem 2).
- Steps:
  1) δ-method, Voronoi on the `r`-sum, as in §5.1, producing oscillatory integrals `J_±` with an effective `m`-range `n^2 m ≲ K' := q^3/X^2 + X u^3` (eq. (5.1)).
  2) Poisson in the `(n1, n2)`-sum with general `Q`; the dual variables `(m1, m2)` now satisfy `|m1| ≲ X^ε`, `|m2| ≲ X^{1−θ+ε}` (eq. (5.3), §5.2).
  3) Bundle transforms into `W_±` (eq. (5.6)). Stationary phase yields (Lemma 16):
     > `W_±(…) ≪ q^{3/2}/Q^{3/2}` (uniformly) and a derivative bound (eq. (5.12)).
  4) Character sum over `a` and the extra δ-phase over `β mod q/n`:
     - Denote `S1(m1, m2, m, n; q)` (eq. (5.9)).
     - Factor `q = q1 q2` with `q1 | (2n|A|)^∞` and `(q2, 2n|A| q1)=1` (where `|A|` is the determinant of the quadratic form matrix).
     - Use Lemma 6 to collapse the `α`-sums to Gauss sums and Ramanujan sums; after routine divisor-sum manipulations,
       > `S1 ≪ (q1^3 / n) · q2^2 d(q1) d(q2)` (Lemma 17, eq. (5.10)).
  5) Avoid Cauchy–Schwarz; instead, apply partial summation to the `m`-sum (eq. (5.12)) to gain an extra factor `X^{1/12}` (eq. (5.13)). This is the small but critical saving that drops the final exponent below the trivial threshold.
  6) Summing over dyadic `q`, `n|q`, `m1, m2` and collecting bounds (Section 5.5–5.6) yields:
     > `S ≪ X^{7/4+ε}` provided `Y = X^θ` with `3/4 < θ ≤ 1` (eq. (5.14)).

Why these design choices?
- Not applying Voronoi/Poisson in the smallest variable in Theorem 1 prevents losses from an unfavorable conductor growth that would outweigh gains.
- In Theorem 2, evading a Cauchy–Schwarz step preserves arithmetic structure needed to apply partial summation plus derivative bounds (eq. (5.12)), which produce a strict power saving.

## 4. Key Insights and Innovations
- DFI δ-method tailored for GL(3) sums on sparse algebraic sets (Sections 3–4):
  - The δ-method is used with bespoke parameter choices (`Q = X^{1/2}` or `X`) and weight management (`ψ(q,x)`, `U(x)`, `V(x)`) to balance oscillation from Voronoi and Poisson with controlled ranges (eqs. (3.3)–(3.5), (4.4), (5.1)).
  - Significance: It allows converting an intractable combinatorial constraint `r = P(n)` into structured exponential sums where analytic tools (Voronoi, Poisson, stationary phase) can extract cancellation.

- Explicit evaluation of key quadratic character sums after Poisson (§4.4):
  - The exact formula `C(m1, m2, a; q) = ε_q^2 q e(−4a(m1^2+m2^2)/q)` (eqs. (4.20)–(4.21)) eliminates noisy dependence on residue classes.
  - Significance: This dramatically simplifies the arithmetic and is a principal step enabling the later Poisson in `m` and the analysis of zero/non-zero frequencies.

- Nonstandard handling of frequency sums with Dąbrowski–Fisher stationary-phase bounds (§4.6):
  - After Poisson in `m`, the character sums become products of Kloosterman sums twisted by linear fractional transformations (eq. (4.34)).
  - Using tools from [5] (Dąbrowski–Fisher) and related l-adic techniques (Fouvry–Kowalski–Michel [6]) yields near square-root cancellations in specific cases:
    > `S_{≠0}(q') ≪ q'^{7/2}` (eq. (4.35)) in the favorable coprime regime.
  - Significance: This is a deep input that prevents the non-zero frequency from dominating, a common hurdle in δ-method analyses.

- A targeted departure from the standard Cauchy–Schwarz step in Theorem 2 (§5.6):
  - By leveraging a derivative bound on `W_±` (eq. (5.12)) and partial summation (eq. (5.13)), the method gains an `X^{1/12}` saving that turns a boundary estimate into a true power saving, culminating in `X^{7/4+ε}` instead of `X^{2−o(1)}`.
  - Significance: This is an instructive procedural innovation—sometimes avoiding a standard inequality preserves structure that yields better cancellation.

## 5. Experimental Analysis (Interpretation of Theoretical Results)
Because this is a theoretical paper, “experiments” are mathematical derivations and bounds. We evaluate the results by comparing exponents to trivial bounds and prior work.

- Setups, baselines, and metrics:
  - Mixed-power sum (eq. (1.5)): baseline trivial bound `Sk(X) ≪ X^{1+1/k+ε}` (just sum sizes).
  - Quadratic-form sum (eq. (1.7)): with equal lengths `X = Y`, earlier bound (eq. (1.6)):
    > `Σ_{1≤n1,n2≤X} Λ(1, Q(n1,n2)) ≪ X^{2 − 1/68 + ε}`.
  - The metric is the exponent of `X` (powers saved vs. trivial baselines).

- Main quantitative results:
  - Theorem 1 (Section 4.10):
    > “`Sk(X) ≪ X^{7/8+ε} Y` for `k=3`;  
    > `Sk(X) ≪ X^{1+ε} Y^{1/2}` for `k ≥ 4`.”
    - Translating exponents:
      - For `k=3` (`Y = X^{1/3}`): `Sk(X) ≪ X^{7/8 + 1/3 + ε} = X^{1.208…+ε}` vs. trivial `X^{1+1/3} = X^{4/3}` → saves `X^{1/8}` over the trivial bound.
      - For `k ≥ 4` (`Y = X^{1/k}`): `Sk(X) ≪ X^{1 + 1/(2k) + ε}` vs. trivial `X^{1 + 1/k}` → saves a factor `X^{1/(2k)}` in the exponent.
    - Scope: Works for `A(n) = Λ(1,n)` (GL(3) coefficients) and also for `A(n) = d3(n)`, with any `a(n)` satisfying `Σ_{n≤X} |a(n)|^2 ≪ X^{1+ε}` (eq. (1.4)).
  - Theorem 2 (Section 5.6, eq. (5.14)):
    > “`Σ Λ(1, Q(n1,n2)) W1(n1/X) W2(n2/Y) ≪ X^{7/4+ε}` for `Y = X^θ`, `3/4 < θ ≤ 1`.”
    - For equal lengths `Y = X`, this is `≪ X^{7/4+ε}`, improving the earlier `X^{2 − 1/68 + ε}` (eq. (1.6)).
    - The lower bound on `θ` (`θ > 3/4`) arises from technical constraints ensuring the Poisson and stationary-phase steps yield sufficient decay and that dual sums stay short (Section 5.2–5.3).

- Do the derivations convincingly support the claims?
  - The pipeline is standard in modern analytic number theory but carefully executed:
    - δ-method expansion (Lemma 8) with well-controlled weights (eqs. (3.3)–(3.5)).
    - GL(3) Voronoi (Lemma 2) plus asymptotics (Lemma 4) to extract oscillation and get range restrictions like `n^2 m ≲ K` (eqs. (4.4), (5.1)).
    - Poisson in the large variables yields manageable dual sums and explicit character sums; stationary phase estimates (e.g., eq. (4.15)) are applied in regimes justifying non-negligible contribution.
    - Arithmetic sums are bounded using exact Gauss-sum evaluations (Lemmas 6–7), Weil bounds, and advanced cancellation results [5,6], giving Lemmas 11–12 and 17.
    - The final optimization `Q = X^{1/2}` (Theorem 1) and `Q = X` (Theorem 2) are standard and explained by the structure of the transforms (Sections 4.10 and 5.1).

- Robustness and sensitivity:
  - Smoothing: The results are for smoothed sums (weights `W1, W2, W3, V, U`), which is standard. Removing smooth weights usually requires additional technical work.
  - The dependence on `θ`: Theorem 2 requires `θ > 3/4`; pushing below `3/4` would need finer control of dual sums and/or stronger cancellation in character sums.

- Comparisons with prior results:
  - For `d3` in the mixed-power sum:
    - Zhou–Hu (eq. (1.3)) showed `O(X^{1 + 1/k − δ(k) + ε})` with `δ(k) = 1/15` for `k=3`, `δ(k) = (k−1)/k^2` for `4 ≤ k ≤ 7`, and `δ(k) = 1/(2k^2(k−1))` for `k ≥ 8`.
    - This paper’s Theorem 1 exponent is:
      - Better for `k=3` (1.208… vs 1 + 1/3 − 1/15 = 1.266…).
      - Better for large `k ≥ 8` (1 + 1/(2k) vs ≈ 1 + 1/k), since `1/(2k) < 1/k`.
      - Weaker for `4 ≤ k ≤ 7` (e.g., for `k=4`, 1.125 vs 1 + 1/16 = 1.0625).
    - So the improvement is conditional on `k`; the paper’s claim that it “gets a stronger bound for each k ≥ 3 than (1.3)” (below eq. (1.5)) holds for `Λ(1,n)` (which [27] does not cover) and for `d3` when `k=3` or `k ≥ 8`, but not for all `k ∈ [4,7]`.
  - For `Λ(1, ·)` over quadratic forms:
    - At equal lengths, `X^{7/4+ε}` improves the previous `X^{2 − 1/68 + ε}` (eq. (1.6)), a substantial gain.

## 6. Limitations and Trade-offs
- Smoothing and weights:
  - All main sums use smooth bump functions `W1, W2, W3, V, U` to localize and enable integration by parts and stationary phase (Sections 4–5). Extending to sharp cutoffs typically requires further technical effort.

- Range of parameters:
  - Theorem 2 requires `Y = X^θ` with `θ > 3/4`. The method, as implemented, does not cover shorter second variables; pushing below `3/4` likely needs stronger control of dual sums or new ideas.

- Scope of arithmetic functions:
  - The analysis treats `A(n) = Λ(1,n)` and `A(n) = d3(n)`; extension to more general sequences or higher-rank coefficients would require adapting Voronoi input and second-moment bounds analogous to Lemma 1.

- Comparisons with best-known results:
  - For the `d3` mixed-power problem, the exponent here is not uniformly better than Zhou–Hu (eq. (1.3)); it improves at `k=3` and `k ≥ 8` but is weaker for `4 ≤ k ≤ 7`. The paper’s general statement about “stronger bounds” (after eq. (1.5)) should be read with this nuance.

- No main terms:
  - The paper proves upper bounds; it does not derive asymptotic main terms. For some divisor-type problems, main terms with secondary terms are known (e.g., eq. (1.2) for `d` over three squares), but such precision is far harder in the GL(3) setting.

## 7. Implications and Future Directions
- Broader impact on GL(3) analytic theory:
  - This work adds to a growing toolkit for handling GL(3) arithmetic functions on sparse algebraic sets, bringing techniques (DFI δ, GL(3)/`d3` Voronoi, Poisson, stationary phase, structured character-sum analysis) into a coherent pipeline.

- Methodological lessons:
  - The explicit quadratic character-sum evaluation after Poisson (eqs. (4.20)–(4.21)) and the Dąbrowski–Fisher-type handling of products of Kloosterman sums (Section 4.6) are broadly applicable techniques.
  - The strategic avoidance of Cauchy–Schwarz in Theorem 2 (Section 5.6) shows that preserving arithmetic structure plus partial summation can yield savings that a generic inequality would forfeit.

- Potential extensions:
  - Other sparse polynomial sets: sums over `x^2 + y^2 + z^k` with different ranges or additional congruence conditions; ternary/quaternary quadratic forms beyond symmetric positive-definite; non-diagonal mixed powers.
  - Other GL(3) families: varying the `(m,n)`-index in `Λ(m,n)` or incorporating twists (e.g., characters) could probe hybrid bounds related to GL(3) L-functions.
  - Toward asymptotics: With stronger control of character sums and oscillatory integrals, one could hope for asymptotic formulae in special cases (as in GL(2) divisor problems), or at least sharper exponents.
  - Lowering the `θ` threshold in Theorem 2: Developing finer estimates for `W_±` and `S1`, perhaps using deeper algebraic geometry inputs, might push the range to `θ ≤ 3/4`.

- Practical applications:
  - While primarily theoretical, improved bounds for sums of automorphic coefficients over structured sets inform subconvexity problems, moments of L-functions, and distribution questions (e.g., primes represented by polynomials) through the common analytic toolkit.

---

Selected quotes for quick reference:
- Theorem 1 (Section 4.10):
  > “`Sk(X) ≪ X^{7/8+ε}Y` for `k = 3` and `Sk(X) ≪ X^{1+ε} Y^{1/2}` for `k ≥ 4`.”
- Theorem 2 (eq. (5.14)):
  > “`Σ Λ(1, Q(n1, n2)) W1(n1/X) W2(n2/Y) ≪ X^{7/4+ε}` for `Y = X^θ`, `3/4 < θ ≤ 1`.”
- Prior `d3` result (eq. (1.3)):
  > “`Σ d3(n1^2 + n2^2 + n3^k) = O(X^{1 + 1/k − δ(k) + ε})` with explicit `δ(k)`.”
- Prior GL(3) over quadratics (eq. (1.6)):
  > “`Σ_{1≤n1,n2≤X} Λ(1, Q(n1, n2)) ≪ X^{2−1/68 + ε}`.”
