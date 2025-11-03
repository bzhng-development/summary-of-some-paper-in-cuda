# Carleman Estimates and Controllability of Forward Stochastic Parabolic Equations with General Dynamic Boundary Conditions

**ArXiv:** [2510.12345](https://arxiv.org/abs/2510.12345)

## 🎯 Pitch

This paper pioneers a new Carleman estimate tailored for backward stochastic parabolic equations with general second-order operators and complex dynamic boundary conditions—a setting that models coupled bulk-surface dynamics under uncertainty. Leveraging this sharp analytical tool, the authors establish both approximate and null controllability for the corresponding forward stochastic problems, even with reaction, convection, and non-trivial coupling between domain and boundary. These results not only advance the theoretical understanding of controllability in stochastic PDEs with dynamic boundaries, but directly impact applications where controlling diffusive processes under uncertainty at evolving interfaces is vital—such as in material science, fluid dynamics, or reactive surface engineering.

---

## 1. Executive Summary (2-3 sentences)
This paper develops a new Carleman estimate—a powerful weighted energy inequality—for backward stochastic parabolic equations with dynamic boundary conditions and general second-order diffusion operators. Using this estimate, it establishes both approximate and null controllability (with explicit control-cost bounds) for forward stochastic parabolic equations that include reaction and convection terms, under dynamic boundary conditions that couple the bulk and the boundary through a time-evolving boundary PDE.

## 2. Context and Motivation
- Problem addressed:
  - Control and observation of stochastic diffusion processes when the boundary itself evolves dynamically in time (via a boundary PDE), rather than imposing static boundary conditions (e.g., Dirichlet or Neumann). The bulk and boundary dynamics are coupled through the conormal derivative `∂_ν^A y` (Section 1, Eq. (1.7)).
  - The forward system includes both reaction terms (`a_i y`, i=1,2) and convection terms (`B_i·∇ y`), and both bulk and surface diffusion are governed by general symmetric, uniformly elliptic matrices `A(t,x,ω)` and `A_Γ(t,x,ω)` (Assumption (1.1)).
  - The adjoint/dual backward system (Eq. (1.2)) contains bulky “divergence source terms” (`∇·F` in the bulk and `∇_Γ·F_Γ` on the boundary) that complicate classical Carleman approaches.

- Why this matters:
  - Dynamic boundary conditions (DBCs) model important physical situations where the boundary has its own dynamics—e.g., heat transfer across an interface with surface diffusion, reactive surfaces, or thin films (Section 1; discussion under Eq. (1.7)).
  - Stochasticity captures uncertainty due to random forcing or noisy environments, which are ubiquitous in realistic models (e.g., random fluctuations in material properties or forcing).
  - Controllability (steering a system to a desired terminal state) under such complex boundary dynamics and noise is crucial for design and regulation in engineering and scientific systems. Theoretical advances here broaden the set of systems amenable to rigorous control design.

- Prior approaches and gaps:
  - Earlier SPDE controllability results covered Dirichlet/Neumann/Robin boundary conditions (see references [2], [3], [7], [32], [34]), or dynamic boundary conditions but typically with simpler Laplacian operators and no divergence sources (e.g., [6]; Section 1).
  - The needed Carleman estimates for the backward problem either did not handle divergence-form source terms or did not extend to general elliptic operators `A`, `A_Γ`, and did not keep explicit dependence on the final time `T`, which is essential for control-cost bounds (Theorem 1.1; Appendix A).

- Positioning:
  - This work extends the Carleman framework to:
    - general second-order operators in divergence form in both bulk and boundary;
    - dynamic boundary coupling via the conormal derivative;
    - divergence-form source terms handled by a duality technique in the stochastic setting;
    - explicit parameter dependence on `T`, enabling explicit cost estimates for null controllability (Eq. (1.8)).
  - It claims (Section 1, paragraph after Theorem 1.3) to be the first to prove both null and approximate controllability for such a general linear forward SPDE with DBCs including reaction and convection.

## 3. Technical Approach
At a high level, the paper follows the classical control-theoretic chain:
Carleman estimate for backward adjoint -> observability inequality -> controllability of forward system via duality.
The difficulty lies in doing this for stochastic systems with dynamic boundary conditions and general operators.

Step-by-step:

1. Problem setup and notation (Section 1)
   - Domain: a bounded smooth domain `G ⊂ R^N (N ≥ 2)` with boundary `Γ = ∂G`; time interval `(0,T)`. Define `Q=(0,T)×G`, `Σ=(0,T)×Γ`, and a control subdomain `G0 ⊂⊂ G`.
   - Probabilistic environment: a filtered probability space carrying a 1D Brownian motion `W(t)`; processes are adapted to the filtration (Section 1).
   - State spaces: `L^2`-type and Sobolev spaces for the bulk and the surface; tangential calculus on `Γ` with tangential gradient `∇_Γ` and Laplace–Beltrami `Δ_Γ`. The conormal derivative with respect to `A` is `∂_ν^A z = (A ∇z · ν)|_Γ` (Eq. (1.2); below Eq. (1.2)).
   - Dynamic boundary conditions:
     - The boundary PDE includes a time derivative term `dz_Γ`, a surface diffusion term `∇_Γ·(A_Γ ∇_Γ z_Γ)`, and is coupled to the bulk via `-∂_ν^A z` (Eq. (1.2), second line).
     - This models a boundary evolving in time, with its own diffusion and interaction with the interior (discussion after Eq. (1.2)).

2. Backward adjoint equation with divergence sources (Eq. (1.2))
   - Backward in time with terminal data `(z_T, z_{Γ,T})` at `t=T`.
   - Bulk: `dz + ∇·(A ∇z) dt = (F1 + ∇·F) dt + Z dW(t)`.
   - Boundary: `dz_Γ + ∇_Γ·(A_Γ ∇_Γ z_Γ) dt - ∂_ν^A z dt = (F2 - F·ν + ∇_Γ·F_Γ) dt + Z̄ dW(t)`.
   - The divergence-form sources `∇·F` (bulk) and `∇_Γ·F_Γ` (boundary) are the main new obstacle for a Carleman estimate.

3. Carleman weights and geometry (Eqs. (1.3)-(1.5), (1.4))
   - A “level set” function `ψ(x)` is chosen using Lemma 1.1 so that `ψ>0` in `G`, `ψ=0` on `Γ`, and `|∇ψ|>0` away from a small interior set `G1 ⊂⊂ G0`. On `Γ`, `∂_ν ψ < 0` (Eq. (1.3)).
   - Time–space weights:
     - `φ(t,x) = e^{µ ψ(x)} / [t (T-t)]` (blows up near `t=0, T`; amplifies information there).
     - `α(t,x) = [e^{µ ψ(x)} - e^{2µ ||ψ||_∞}] / [t (T-t)]`.
     - `ℓ = λ α`, `θ = e^ℓ` (Eq. (1.4)).
   - These weights shape the inequality: left side focuses on weighted energy of `z`, `z_Γ`, and gradients; right side keeps “undesirable” terms under control (Eq. (1.6)).

4. Carleman estimate without divergence sources (Lemma 2.1; Appendix A)
   - Starting point: prove the Carleman estimate for Eq. (1.2) with `F ≡ 0` and `F_Γ ≡ 0` (Lemma 2.1, Eq. (2.1)).
   - Technique:
     - Use global weighted identities tailored to stochastic parabolic operators in the bulk (Theorem A.1) and on the boundary (Theorem A.2). These are derived via Itô calculus and PDE identities (Appendix A, Eqs. (A.2), (A.3)).
     - Choose a boundary weight `Φ` and the boundary parameter `A^e` to control the dynamic boundary terms (Lemma A.2; Eqs. for `A^e`, `B^e`), exploiting `∂_ν ψ < 0` on `Γ` (Eq. (1.3)). This is key to handling boundary integrals that otherwise prevent closing the estimate (Appendix A, Step 2).
     - Use ellipticity (Assumption (1.1)) and carefully scale parameters `λ, µ` with `T` to absorb remainder terms (Lemma A.1; bounds in Appendix A).
     - Control localized terms near `G1` by a cutoff argument (Eq. (A.38)) to translate weighted `|∇z|^2` terms into localized `z^2` observation on `Q0=(0,T)×G0` (final step of Appendix A; Eq. (A.37)-(A.38)).

5. Handling divergence sources via duality (Section 2; Proposition 2.1)
   - Directly, `∇·F` and `∇_Γ·F_Γ` do not fit easily in the Carleman inequality. The workaround is a duality argument:
     - Introduce a forward controlled system (Eq. (2.2)) with controls `(u, v1, v2)` acting in the bulk drift (localized in `G0`) and in the stochastic terms (globally in bulk and boundary).
     - Set up an optimal control problem with a weighted quadratic cost using a regularized weight `θ_ε` (Eq. (2.4)) and penalty on terminal energy (Eq. (2.5)).
     - Derive the optimality system (Eq. (2.6)-(2.7)) where the adjoint is precisely the backward system without divergence sources (the one handled by Lemma 2.1).
     - Prove a uniform estimate on the optimal controls and states (Eq. (2.3)), pass to the limit `ε→0`, and fold these bounds back into the backward equation’s energy identity (Eqs. (2.23)-(2.25)).
   - In short: the “bad” divergence sources are paired against the forward controlled variables and estimated through the controllability analysis of the auxiliary forward system.

6. The full Carleman estimate (Theorem 1.1)
   - Combine the base Carleman (Lemma 2.1), the duality machinery (Proposition 2.1), and additional weighted energy manipulations (Eqs. (2.26)-(2.34)) to obtain Eq. (1.6):
   > For large enough `µ, λ` (with explicit `T`-dependence),  
   > the weighted integrals of `z, z_Γ, ∇z, ∇_Γ z_Γ` over `Q` and `Σ` are bounded by a localized observation of `z` on `Q0`, source terms `(F1, F2)`, divergence sources `(F, F_Γ)`, and noise processes `(Z, Z̄)` (Eq. (1.6)).

7. Observability and controllability (Sections 3–4)
   - Apply Theorem 1.1 to the specific adjoint of the target forward system (Eq. (3.1)) by substituting `F1=-a1 z`, `F=z B1`, `F2=-a2 z_Γ`, `F_Γ = z_Γ B2` (Eq. (3.4)), leading to a specialized Carleman estimate (Corollary 3.1, Eq. (3.3)).
   - Deduce:
     - Unique continuation: if `(z, Z, Z̄)` vanish in `Q0×Q×Σ`, then the terminal data vanish (Eq. (3.7)).
     - Observability inequality: initial energy is controlled by the localized observation of `z` on `Q0` plus `Z, Z̄` energies (Eq. (3.8)).
   - Use standard duality (Lemma 4.1) to convert observability into:
     - Approximate controllability (Theorem 1.2): range-density via injectivity (Section 4, Eq. (4.4)-(4.5)).
     - Null controllability with an explicit cost bound (Theorem 1.3; Eq. (1.8), derived from Eq. (4.7)).

Key design choices:
- Weighted identities in both the bulk and the boundary operator (Appendix A) tailored to dynamic boundary coupling.
- A boundary weight `Φ` that leverages `∂_ν ψ<0` to sign boundary terms correctly (Lemma A.2).
- Duality to remove divergence-form sources that resist direct Carleman integration by parts (Section 2).

## 4. Key Insights and Innovations
- New Carleman estimate for stochastic parabolic equations with dynamic boundary conditions and general operators (Theorem 1.1):
  - Differs from earlier results (e.g., [6]) by:
    - Allowing general symmetric, uniformly elliptic `A, A_Γ` instead of the identity (Laplacian);
    - Including divergence-form sources in both bulk and surface equations;
    - Making the dependence of Carleman parameters `(λ, µ)` on final time `T` explicit (critical for control-cost bounds).
  - Significance: This is the cornerstone inequality enabling all subsequent observability and controllability conclusions.

- Duality-based treatment of divergence sources in the stochastic setting (Section 2, Proposition 2.1):
  - Rather than force-fitting `∇·F` terms into Carleman integration, the paper uses an auxiliary forward controlled system and optimality conditions that “transfer” the burden of these terms to controlled variables that can be estimated (Eqs. (2.2)–(2.7), inequality (2.3)).
  - Significance: This expands the class of admissible source terms and is robust under stochasticity.

- Controllability of a general forward stochastic parabolic system with DBCs and convection (Theorems 1.2 and 1.3):
  - Prior work with DBCs often omitted convection terms or restricted to Laplacians (Section 1). Here, controllability is established with convection and general operators, plus explicit control-cost dependence on `T` and coefficient norms (Eq. (1.8)).
  - Significance: Broadens the practical and theoretical reach of SPDE control under realistic boundary behavior.

- Boundary-weight design to absorb dynamic boundary terms (Appendix A; Lemma A.2 and Step 2):
  - The choice `Φ ∝ (Aν·ν)|∇ℓ|` and the geometric facts about `ψ` on `Γ` (Eq. (1.3)) tame problematic surface integrals, especially those involving `∂_ν^A z` and tangential derivatives, and exploit surface diffusion `∇_Γ·(A_Γ ∇_Γ z_Γ)` to absorb “bad” boundary energies (Remark 1.4).
  - Significance: Shows how geometry and boundary calculus can be engineered into the Carleman framework for DBCs.

These are primarily fundamental innovations (new estimate + technique) rather than incremental tweaks, enabling new controllability results under broader assumptions.

## 5. Experimental Analysis
Since this is a theoretical paper, “experiments” are proof-driven evaluations. The “metrics” are rigorous inequalities with explicit constants and parameter dependencies. The “baselines” are earlier Carleman/controllability results for simpler boundary conditions or operators.

- Evaluation methodology:
  - Establish the main inequality (Carleman) with weighted integrals that serve as proxies for system observability (Theorem 1.1, Eq. (1.6)).
  - Specialize to the adjoint of the forward system (Eq. (3.1)) to obtain a usable observability bound (Corollary 3.1, Eq. (3.3)); then derive unique continuation (Eq. (3.7)) and observability (Eq. (3.8)).
  - Use duality (Lemma 4.1) to translate observability into approximate and null controllability, with an explicit exponential cost bound (Theorem 1.3, Eq. (1.8)).

- Main quantitative results:
  - Carleman estimate (Eq. (1.6)):
    > The weighted bulk and boundary energies of `z` and its gradients are bounded by: a localized observation on `Q0`, the source terms `F1, F2`, the divergence sources `F, F_Γ`, and the noise energies `Z, Z̄`, with explicit powers of `λ, µ` and `φ` (weights).
  - Observability inequality (Eq. (3.8)):
    > Initial energy `∥z(0)∥^2 + ∥z_Γ(0)∥^2` ≤ exp(C K) × [localized observation of `z` on `Q0` + total noise energy],  
    > with `K = 1 + 1/T + ∥a_i∥^{2/3}_∞ + T(∥a_i∥_∞) + (1+T)(∥B_i∥^2_∞)` aggregating coefficient sizes and time scales.
  - Null controllability cost (Eq. (1.8)):
    > `∥u∥^2 + ∥v1∥^2 + ∥v2∥^2 ≤ exp(C K) ∥(y0, y_{Γ,0})∥^2`,  
    > with the same `K` as above, making explicit how short times (`T→0`) or large coefficients inflate cost.

- Support for claims:
  - The proofs carefully track how `λ, µ` must scale with `T` and coefficients to absorb terms (e.g., Corollary 3.1, condition on `λ ≥ λ0[T + T^2(1 + norms)]`; Appendix A for detailed estimates).
  - The duality step (Section 2) is nontrivial and central to handling divergence sources; its conclusion (Eq. (2.3)) is fed into the final Carleman bound (Eq. (2.25)) and gradient control (Eqs. (2.31)-(2.34)), closing the loop.

- Ablation-like insights and robustness:
  - Role of surface diffusion in DBCs: Remark 1.4 highlights that the boundary surface diffusion term `∇_Γ·(A_Γ∇_Γ y_Γ)` is crucial to absorb terms like  
    > `λ µ ∫_Σ θ^2 φ |∂_ν^A ψ| |∇_Γ z_Γ|^2`,  
    and removing it makes the problem open.
  - Controls: adding `v1, v2` (noise-level controls on bulk and surface) enables the weaker observability inequality used here (Remark 1.6; Section 3), while the “drift-only” single localized control remains open.

- Conditions and trade-offs:
  - Large-parameter regime: The estimates require `µ` large, and `λ` even larger with explicit `T`-dependence (Theorem 1.1; Corollary 3.1).
  - Cost growth: The control cost grows exponentially with a factor `K` that includes `1/T` and coefficient norms (Eq. (1.8)), reflecting the intrinsic difficulty of fast-time or strongly convective/reactive systems.

Overall, the “experimental” (proof) evidence convincingly supports the main claims under the stated assumptions. The paper is thorough in tracking constants and parameter dependencies, which is critical for control-cost statements.

## 6. Limitations and Trade-offs
- Structural assumptions:
  - Diffusion matrices `A, A_Γ` must be symmetric, uniformly elliptic, and sufficiently regular in time and space (Assumption (1.1); beginning of Section 1). This excludes degenerate or highly irregular media.
  - Coefficients `a_i, B_i` are bounded adapted processes (Section 1, after Eq. (1.7)). Unbounded or rough coefficients are out of scope.

- Boundary dynamics:
  - The method critically uses the presence of surface diffusion `∇_Γ·(A_Γ ∇_Γ y_Γ)` to absorb boundary terms (Remark 1.4; Appendix A, Step 2). DBCs lacking this term (pure dynamic coupling without surface diffusion) remain open in higher dimensions.

- Controls:
  - The controllability results use three controls: `u` localized in the drift (bulk), and `v1, v2` acting on the stochastic terms in the entire bulk and boundary (Section 1, after Eq. (1.7)). Achieving controllability with just the localized drift control `u` is a challenging open problem (Remark 1.6).

- Nonlinear extensions:
  - While linear systems with reaction and convection are covered, extending to semilinear systems with gradient nonlinearities on both bulk and boundary faces challenges due to lack of compactness in the stochastic setting (Remark 1.5). Some special semilinear cases (Dirichlet) exist in the literature, but the dynamic-boundary, gradient-dependent case is open.

- Cost and scalability:
  - The control cost bound is exponential in `K` (Eq. (1.8)), which includes `1/T` and coefficient norms. Thus, small control times or large coefficients can make costs very large—this is typical but still a practical constraint.

## 7. Implications and Future Directions
- Field impact:
  - Establishes a general-purpose Carleman framework for stochastic parabolic equations with dynamic boundary conditions and general operators. This broadens the toolkit for SPDE controllability in settings closer to real interfaces and surfaces.
  - The explicit `T`-dependence in the Carleman parameters and cost bound bridges theory and practice by quantifying temporal trade-offs.

- Practical applications:
  - Control of diffusion-driven processes with active interfaces: e.g., heat or mass transfer with reactive/porous boundaries; catalysis and surface chemistry; biological membranes with surface transport; microfluidic devices where both bulk and boundary layers matter.
  - Regulation under uncertainty: stochasticity in materials or forcing is explicitly modeled and controlled.

- Research directions:
  - DBCs without surface diffusion (Remark 1.4): develop new boundary-weight techniques to absorb the problematic `|∇_Γ z_Γ|^2` terms without relying on `∇_Γ·(A_Γ ∇_Γ y_Γ)`.
  - Single-control schemes (Remark 1.6): aim for observability that relies only on localized bulk observations (`z` on `Q0`) without noise controls `v1, v2`. This may require new unique continuation results for the adjoint (beyond Eq. (3.7)).
  - Semilinear and quasilinear extensions with gradient dependencies (Remark 1.5): overcome the lack of compactness in SPDEs to handle nonlinear boundary and bulk terms, possibly via monotonicity methods, random compactness alternatives, or pathwise techniques.
  - Relaxed coefficient assumptions: investigate robustness to lower regularity or anisotropic degeneracy in `A, A_Γ`.
  - Numerical aspects: leverage the explicit `T`-dependent scaling to design discretizations and regularizations that preserve observability and provide computable control-cost estimates.

Block-quoted key results for quick reference:
- Carleman estimate (general form): Eq. (1.6), Theorem 1.1.
- Observability for adjoint: Eq. (3.3) (Carleman), Eq. (3.8) (observability inequality).
- Unique continuation: Eq. (3.7).
- Null controllability cost: Eq. (1.8) with `K` detailed below Eq. (1.8).

In essence, the paper shows how to control noisy diffusion systems with evolving boundaries under very general diffusion/convection structures, by crafting a boundary-aware Carleman estimate and a duality mechanism that neutralizes divergence-form sources—opening paths to richer boundary-controlled SPDEs.
