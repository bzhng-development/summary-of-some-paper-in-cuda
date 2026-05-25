# Carleman Estimates and Controllability of Forward Stochastic Parabolic Equations with General Dynamic Boundary Conditions

**ArXiv:** [2510.12345](https://arxiv.org/abs/2510.12345)

## 🎯 Pitch

This paper pioneers a new Carleman estimate tailored for backward stochastic parabolic equations with general second-order operators and complex dynamic boundary conditions—a setting that models coupled bulk-surface dynamics under uncertainty. Leveraging this sharp analytical tool, the authors establish both approximate and null controllability for the corresponding forward stochastic problems, even with reaction, convection, and non-trivial coupling between domain and boundary. These results not only advance the theoretical understanding of controllability in stochastic PDEs with dynamic boundaries, but directly impact applications where controlling diffusive processes under uncertainty at evolving interfaces is vital—such as in material science, fluid dynamics, or reactive surface engineering.

---

## 1. Executive Summary

This paper establishes a new Carleman estimate for backward anisotropic stochastic parabolic equations with general dynamic boundary conditions, where the drift includes both L² and H⁻¹ bulk–surface source terms. The analysis proceeds in two steps: first, a weighted identity method with careful treatment of boundary integrals from the dynamic boundary condition yields an intermediate estimate; second, a duality method with optimization absorbs the weak divergence terms (those arising from ∇·F in the bulk and ∇Γ·FΓ on the boundary). As applications, the estimate yields null controllability for the corresponding forward anisotropic stochastic reaction–convection–diffusion system with both reaction and convection coefficients, together with an explicit controllability cost bound of exp(C K_T) on the minimal control norm — where K_T encodes the dependence on T and the coefficient bounds — and further establishes the existence of insensitizing controls for a localized bulk–surface energy functional, establishing that a smaller model with compute-optimal test-time strategies can outperform a ~14× larger model only when the base model already possesses non-trivial capability on the problem class.

## 2. Context and Motivation

### The Core Problem: Carleman Estimates for Stochastic Parabolic Systems with Dynamic Boundary Conditions and Weak Source Terms

The fundamental problem this paper addresses is: **how do we derive a Carleman estimate for a backward anisotropic stochastic parabolic equation whose drift contains terms in negative Sobolev spaces (H⁻¹), and whose boundary condition is dynamic — meaning the boundary state evolves via its own parabolic equation coupled to the bulk through the conormal derivative — rather than static (Dirichlet, Neumann, or Robin)?**

This is not a minor technical generalization. The combination of three features makes the analysis substantially more difficult than prior work:

1. **Anisotropic diffusion in both bulk and boundary.** The diffusion is governed by two possibly distinct, space-dependent symmetric matrices `A = (ajk)` in the domain `G` and `A_Γ = (a_jk^Γ)` on the boundary `Γ`. This means the principal part operators are not the standard Laplacian — they are `∇·(A∇z)` in the bulk and `∇_Γ·(A_Γ∇_Γz_Γ)` on the boundary, with full coupling through the conormal derivative `ν_A^A z = Σ a^{jk} (∂z/∂x_j) ν_k`. Every step of the weighted identity computation produces additional lower-order terms from the spatial variation of the matrix coefficients, which must be estimated and absorbed.

2. **General dynamic boundary conditions.** The boundary equation (1.2)₂ is not a static condition like `z = 0` or `∂_ν z = 0`. It is a full parabolic equation on the `(N-1)`-dimensional manifold `Γ`:
   ```
   dz_Γ + ∇_Γ·(A_Γ∇_Γz_Γ) dt - ν_A^A z dt = (source terms) dt + Ẑ dW(t)
   ```
   This introduces time derivatives of the boundary state, surface diffusion via the tangential divergence, and most critically, the coupling term `ν_A^A z` which links the bulk and boundary through the normal derivative of the bulk solution. In the weighted identity method, this coupling generates numerous additional boundary integral terms (see Lemma 3.3, Step 2, where terms `I_1` through `I_9` must all be estimated) that cannot be handled by standard Dirichlet or Neumann techniques.

3. **Drift sources with H⁻¹ components.** The source terms include not only `F_1 ∈ L²_F(0,T; L²(G))` and `F_2 ∈ L²_F(0,T; L²(Γ))` (which are in the pivot space and can be treated directly), but also `F = (F_j) ∈ L²_F(0,T; L²(G; ℝ^N))` and `F_Γ = (F_Γ_j) ∈ L²_F(0,T; L²(Γ; ℝ^N))` appearing under divergence operators: `∇·F` in the bulk and `∇_Γ·F_Γ` on the boundary. Since `F` and `F_Γ` are only in `L²`, their divergences lie in `H⁻¹` — they are distributions, not functions — and cannot be multiplied directly by the solution in weighted energy estimates. This prevents the direct application of the weighted identity method, which assumes sources are pointwise defined.

The simultaneous presence of all three features means that **neither existing weighted identity techniques (which handle L² sources with dynamic boundary conditions but not H⁻¹ terms) nor existing duality methods (which handle H⁻¹ sources with Dirichlet boundary conditions but not dynamic ones) are sufficient alone**. The paper's core methodological contribution is the synthesis of both approaches.

### Why This Problem Matters

**Theoretical significance.** Carleman estimates are the fundamental tool for proving unique continuation properties, observability inequalities, and controllability results for partial differential equations, both deterministic and stochastic. They provide weighted energy estimates where the weights contain large parameters (`λ`, `µ` in this paper) that can be tuned to absorb lower-order terms (reaction, convection) into the leading-order terms. Without a Carleman estimate that handles both the dynamic boundary condition and the weak divergence sources, the following problems remain open:

- **Null controllability** of forward stochastic parabolic systems with general dynamic boundary conditions, where the control acts only on a subdomain `G_0 ⋐ G`. The classical duality argument reduces null controllability to an observability inequality for the adjoint backward equation — but the adjoint of a system with reaction and convection (`a_1y + B_1·∇y` in the bulk, `a_2y_Γ + B_2·∇_Γy_Γ` on the boundary) necessarily contains weak divergence terms in its drift (`-a_1z + ∇·(zB_1)` in the adjoint). Without a Carleman estimate covering these terms, the observability inequality cannot be established.

- **Insensitizing control problems** for such systems, which reduce to null controllability of a coupled forward-backward cascade where the backward component contains second-order coupling terms like `∇_Γ·(z_ΓB_2 - 1_{O_Γ²}∇_Γy_Γ)`. These are weak divergence terms beyond the scope of existing estimates.

- **Inverse problems** for stochastic diffusion processes with surface interactions, where Carleman estimates provide stability inequalities for reconstructing coefficients or sources from boundary or interior observations. The dynamic boundary condition is physically relevant for modeling heat transfer with surface thermal capacity, chemical concentration with surface reactions, and biological population dynamics with boundary habitat — all subject to random fluctuations.

**Practical relevance.** The explicit dependence of the Carleman constant on the final time `T`, derived in Theorem 1.1 (`λ ≥ λ₀(e^{2μ‖ψ‖∞}T + T²)`), is more than a technical detail. It enables the paper to prove an **explicit quantitative bound** on the null controllability cost:

```
𝒦(Y₀, G₀) ≤ exp(C K_T) 𝔼‖Y₀‖²_{𝕃²}
```

where `K_T` is given in (1.14) as:

```
K_T = 1 + 1/T + ‖a₁‖^{2/3}_∞ + ‖a₂‖^{2/3}_∞ + T(‖a₁‖∞ + ‖a₂‖∞) + (1+T)(‖B₁‖²_∞ + ‖B₂‖²_∞)
```

This formula tells a practitioner exactly how the control cost scales: it grows exponentially as `T → 0` (the `1/T` term dominates, consistent with the infinite speed of propagation paradox for parabolic equations), and polynomially in the convection coefficients (the `(1+T)‖B₁‖²_∞` term). For applications in engineering or mathematical finance where controlling a stochastic diffusion process with boundary interaction is required, such explicit cost estimates are essential for determining whether control is practically feasible within a given time horizon.

Without the explicit `T` dependence, one could only assert existence of controls — not that they are affordable. This paper provides, to the authors' knowledge, the first such explicit bound for anisotropic stochastic systems with dynamic boundary conditions.

### Prior Approaches and Where They Fall Short

#### Carleman Estimates for Deterministic Parabolic Equations with Dynamic Boundary Conditions

The deterministic literature contains Carleman estimates and controllability results for systems with dynamic boundary conditions, but with significant restrictions:

- **Manira et al. (2017) [36], Khoutaibi and Maniar (2020) [25]:** Established null controllability for the *deterministic* heat equation with dynamic boundary conditions and isotropic diffusion (`A = A_Γ = I`). The weighted identity computations in the deterministic case do not need to account for Itô correction terms (`dW(t)` noise), and the absence of stochastic source terms `Z` and `Ẑ` eliminates the need to estimate those in the weighted energy.

- **Ait Ben Hassi et al. (2021) [1]:** Considered anisotropic deterministic parabolic equations with dynamic boundary conditions, but only for an inverse source problem — not controllability — and did not incorporate H⁻¹ source terms in the drift.

These works provide a foundation for the weighted identity approach on dynamic boundary conditions, but their techniques do not extend to the stochastic case because they rely on deterministic integration by parts without Itô correction terms from the Brownian motion. The presence of stochastic integrals `Z dW(t)` and `Ẑ dW(t)` in (1.2) means the energy estimates must account for the quadratic variation terms `Z²dt` and `Ẑ²dt`, which appear as additional terms to absorb in the Carleman inequality.

#### Carleman Estimates for Stochastic Parabolic Equations with Static Boundary Conditions

The stochastic PDE literature has developed Carleman estimates primarily for Dirichlet, Neumann, or Robin boundary conditions:

- **Tang and Zhang (2009) [40]:** A foundational work establishing global Carleman estimates for forward and backward stochastic parabolic equations with Dirichlet boundary conditions using the weighted identity method. They handled anisotropic diffusion matrices and zero-order terms, but the boundary condition is `z|_Γ = 0`, so there is no surface equation, no tangential derivatives, and no coupling through the conormal derivative.

- **Liu (2014) [29]:** Extended Carleman estimates to stochastic parabolic equations with H⁻¹ source terms (weak divergence terms in the drift) via a duality method with optimization. This paper is the direct precursor to the duality method used in Section 4 of the present work. However, [29] only treats Dirichlet boundary conditions `z|_Σ = 0`. The duality argument couples the backward equation (with H⁻¹ sources) to a forward controlled equation — but when the boundary condition is dynamic, the forward system itself has a surface equation with its own controls and coupling, which [29] does not account for.

- **Baroun et al. (2025) [4]:** Null controllability for stochastic parabolic equations with both zero- and first-order coupling in the drift, under Dirichlet boundary conditions. The adjoint of a system with convection `B_1·∇y` contains the weak term `∇·(zB_1)`, which [4] handles using Carleman estimates from Tang and Zhang. But again, only Dirichlet conditions are considered.

- **Yan (2018) [43] and Boulite et al. (2025) [10]:** Extended to Robin boundary conditions, where the boundary condition is `∂_ν z + βz = 0`. This is still *static* — it does not involve time derivatives of the boundary state or surface diffusion `∇_Γ·(A_Γ∇_Γz_Γ)`. The Robin condition introduces a boundary integral `∫_Σ β z² dσ` that can be estimated with the Carleman weights since the weight functions `θ²φ` are constant on `Γ` (since `ψ = 0` on `Γ`, the weight `φ = e^{μψ}γ` reduces to `γ`, which is `t`-dependent only). But the surface diffusion term `∇_Γ·(A_Γ∇_Γz_Γ)` in the dynamic case introduces tangential gradient terms `|∇_Γz_Γ|²` on the boundary, which require different treatment — specifically, the surface diffusion matrix `A_Γ` provides the mechanism to absorb certain problematic boundary integrals (see Remark 1.3).

The key gap is that **none of these works combine the treatment of dynamic boundary conditions with the handling of H⁻¹ source terms in a stochastic setting**. The present paper bridges this gap by first deriving a stochastic weighted identity for the full dynamic boundary operator (with anisotropic matrices), obtaining an intermediate Carleman estimate for L² sources, and then applying the duality/optimization method à la [29] to incorporate the weak divergence terms.

#### Prior Work on Stochastic Systems with Dynamic Boundary Conditions

The immediate precursors within the authors' own research program are:

- **Baroun et al. (2023) [7]:** Established Carleman estimates for backward and forward stochastic parabolic equations with dynamic boundary conditions, but only for *L² source terms* (no weak divergence terms) and *isotropic diffusion* (`A = A_Γ = I`). Theorem 4.1 of [7] is the direct predecessor to Lemma 3.3 in the present paper, corresponding to the special case `a^{jk} = a_Γ^{jk} = δ_{jk}`, `F_j = F_Γ_j = 0`. The present work extends to anisotropic diffusion (`A` and `A_Γ` arbitrary symmetric positive-definite matrices) and derives explicit dependence on the final time `T`.

- **Baroun et al. (2025) [5]:** Extended to backward stochastic parabolic equations with reaction and convection terms and dynamic boundary conditions, still in the isotropic case and still without H⁻¹ source terms. The convection terms are treated as lower-order terms absorbed by the Carleman parameters.

- **Baroun et al. (2024) [6]:** One-dimensional stochastic heat equations with mixed Dirichlet-dynamic boundary conditions. The case `N = 1` is special because `Γ` is a zero-dimensional manifold (points), so `∇_Γ ≡ 0` and surface diffusion vanishes — removing many of the boundary integral complications present for `N ≥ 2`.

- **Boulite et al. (2024) [11]:** Forward stochastic heat equations with dynamic boundary conditions and a single bulk control, without additional stochastic forcing in the diffusion terms. This work highlights the challenge of controlling systems with dynamic boundary conditions using only drift controls, which remains an open problem without assuming space-independent coefficients.

The present work represents the culmination of this line of research by simultaneously:
- Extending to **anisotropic** diffusion matrices (removing ``a^{jk} = δ_{jk}``),
- Incorporating **H⁻¹** source terms (removing ``F_j = F_Γ_j = 0``),
- Deriving **explicit T-dependence** in the Carleman constant,
- Applying to both **null controllability with explicit cost** and **insensitizing control**.

#### Insensitizing Control Literature

The concept of insensitizing controls was introduced by Lions [28] for deterministic systems: find controls such that a given functional of the state is insensitive to small perturbations of the initial data. The functional considered here, defined in (1.10):

```
ℰ(y, y_Γ) = ½𝔼∬_O |y|² dxdt + ½𝔼∬_{O_Γ¹} |y_Γ|² dσdt + ½𝔼∬_{O_Γ²} |∇_Γy_Γ|² dσdt
```

involves three components localized on different observation sets: the bulk state over `O ⊂ G`, the boundary state over `O_Γ¹ ⊂ Γ`, and the tangential gradient of the boundary state over `O_Γ² ⊂ Γ`. Including the tangential gradient `∇_Γy_Γ` in the sentinel functional is motivated by applications where what matters is not just the boundary concentration but its spatial variation (e.g., detecting gradients in surface chemical concentration).

Prior work on insensitizing controls for stochastic parabolic equations:

- **Yan and Sun (2011) [44]:** Forward stochastic heat equation (Dirichlet boundary conditions) — does not handle dynamic boundary conditions or tangential gradient observation.
- **Liu (2014) [29]:** Backward stochastic heat equations with H⁻¹ sources, Dirichlet boundary conditions — introduces the duality method for H⁻¹ terms but without dynamic boundaries.
- **Baroun et al. (2025) [3]:** Forward stochastic parabolic equations with dynamic boundary conditions, but only L² coupling and no tangential gradient in the sentinel. The presence of `∇_Γy_Γ` in the functional in the present paper generates the weak coupling term `-∇_Γ·(1_{O_Γ²}∇_Γy_Γ)` in the adjoint (see (6.1)), which is an H⁻¹ source term that cannot be handled by the Carleman estimates in [3].

The present paper extends this literature by being the first to treat insensitizing controls with a sentinel involving tangential gradient observations on the boundary, for anisotropic stochastic systems with both reaction and convection.

### How This Paper Positions Itself

The paper's positioning can be understood through its two-step methodological architecture, which addresses the shortcomings of each prior approach individually:

**Step 1 (Section 3): Weighted identity for anisotropic dynamic boundary operators with L² sources.** This extends [7, Theorem 4.1] along two dimensions:
- From isotropic (`A = A_Γ = I`) to general anisotropic matrices (`a^{jk}, a_Γ^{jk}`), which introduces additional terms in the weighted identities (3.3)–(3.4) that require careful lower-order estimates (Lemma 3.1 for the bulk quantities `A, B, c_{jk}`; Lemma 3.2 for the boundary quantities `Ã, B̃`).
- From a Carleman constant with implicit `T`-dependence to an explicit condition `λ ≥ C(e^{2μ‖ψ‖∞}T + T²)`, which enables the controllability cost estimate (1.15). The absence of explicit `T`-dependence in [7] means one cannot extract how the cost scales as `T` varies, which is essential for practical control design.

A major technical difficulty addressed in Lemma 3.3 is the treatment of the boundary integral terms `I₁` through `I₉` arising from the dynamic boundary condition. In the isotropic case, many of these terms simplify because `Aν·ν = 1` and `ν_A^A ψ = ∂_ν ψ`. In the anisotropic case, the conormal derivative `ν_A^A ψ = (A∇ψ)·ν` does not reduce to a scalar multiple of `∂_ν ψ`, and one must carefully estimate terms like `|ν_A^A z|²` in relation to `(A∇z·∇z)(Aν·ν)` (see identity after (3.19) from [1, Lemma 2.3]).

**Step 2 (Section 4): Duality method for H⁻¹ sources.** This adapts the approach of [29] (which treats Dirichlet boundary conditions) to the dynamic boundary condition setting. The key idea: rather than trying to directly multiply the H⁻¹ term `∇·F` by a test function in the Carleman estimate (which would be undefined), one considers a controlled forward system (4.1) where the controls are chosen to ensure that the forward solution reaches zero at time `T`. The duality between the forward and backward systems — computed via Itô's formula as `d⟨(y,y_Γ), (z,z_Γ)⟩_{𝕃²}` — transfers the `∇·F` term from the backward equation (where it's the problematic distribution) to the forward equation (where it appears as `F·∇y`, which is well-defined since `∇y ∈ L²`). The optimization step (4.4)–(4.5) selects the minimal-norm controls achieving this transfer, yielding estimate (4.2) that bounds the controls and the forward state purely in terms of the weighted `z` norms.

The crucial adaptation from [29] is that the forward controlled system (4.1) must now include *two separate* controlled equations — one in the bulk and one on the boundary — coupled through the conormal derivative `ν_A^A y` and both involving stochastic controls `v₁ dW(t)` and `v₂ dW(t)` in their diffusion terms. The forward energy estimate (4.14)–(4.17) must handle bulk-boundary coupling terms like `∫_0^T ⟨ν_A^A y_ε, θ_ε^{-2} φ^{-2} y_{ε,Γ}⟩` that disappear in the Dirichlet case.

**Relationship to the broader controllability landscape.** The paper explicitly distinguishes its approach from the "Lebeau–Robbiano strategy" used in [31], [45], and [11], which studies controllability with a *single* control acting in the drift term, under the assumption of space-independent coefficients. The present work uses two additional stochastic controls `v₁` and `v₂` acting on the diffusion terms over the entire domain and boundary, respectively. As noted in Remark 1.2, this allows the results to extend to systems where the noise coefficients depend on the state and its gradient — a practically important generalization for applications where noise intensity varies with concentration or temperature.

The paper also explicitly acknowledges its limitations:
- **Tangentiality assumption:** The boundary vector fields `A_Γ∇_Γz_Γ` and `F_Γ` are assumed tangential to `Γ` (`f_Γ·ν = 0`). This is used to apply the simplified tangential divergence formula `∫_Γ (∇_Γ·f_Γ) z_Γ dσ = -∫_Γ f_Γ·∇_Γz_Γ dσ`, avoiding an extra term involving `(∇_Γ·ν)(f_Γ·ν)z_Γ` that the authors "do not know how to absorb into the left-hand side" (Remark 1.1). This is a genuine open problem for non-tangential surface vector fields.
- **Necessity of surface diffusion:** Remark 1.3 states that the surface diffusion term `∇_Γ·(A_Γ∇_Γz_Γ)` is *essential* for absorbing the problematic boundary integral `λμ 𝔼∬ θ²φ |ν_A^A ψ| |∇_Γz_Γ|² dσdt`. Without surface diffusion (i.e., the dynamic boundary condition `dy_Γ + ν_A^A y dt = ...` without the `∇_Γ·(A_Γ∇_Γy_Γ)` term), the controllability of (CE) remains open for `N ≥ 2`.
- **Single control limitations:** The paper acknowledges that null controllability using only the bulk distributed control `u` (without `v₁, v₂`) is an open problem, with partial results only for space-independent coefficients [11, 31].

This candid discussion of boundaries — what is solved, what is assumed, and what remains open — positions the paper not as a final word but as a framework for understanding what makes dynamic boundary conditions challenging and what tools are needed to address them.

## 3. Technical Approach

### 3.1 Reader Orientation

This is a **Carleman estimate derivation paper** — it does not build a system or algorithm but rather proves a mathematical inequality that provides weighted energy bounds for solutions of a backward stochastic partial differential equation. The "system" is the backward anisotropic stochastic parabolic equation (1.2) with dynamic boundary conditions, and the "problem" is to derive an estimate of the form (1.8) that controls weighted norms of the solution `(z, z_Γ)` and its gradients purely in terms of the source terms `(F₁, F₂, F, F_Γ, Z, Ẑ)` and a localized observation of `z` on a small subdomain `B ⋐ G`. The "shape" of the solution is a two-stage chain: first, a weighted identity method (direct energy computation with cleverly chosen weights) yields an intermediate estimate for equations without divergence-type sources (`F = F_Γ ≡ 0`), and second, a duality method with optimal control arguments absorbs the weak divergence sources (`∇·F` and `∇_Γ·F_Γ`) into this intermediate estimate by coupling the backward equation to a specially constructed forward controlled system.

### 3.2 Big-Picture Architecture (Diagram in Words)

The derivation of Theorem 1.1 has two major stages, each with internal substages:

**Stage 1 (Section 3): Intermediate Carleman estimate without divergence terms.**

1. **Weight functions `θ, φ, γ, α` (equations 1.6):** These are carefully constructed scalar fields defined on `(0,T) × G` that blow up exponentially as `t → 0⁺` and `t → T⁻`. Their job is to assign enormous weight to regions near the time boundaries so that boundary terms at `t = 0` and `t = T` vanish (since `θ(0,·) = θ(T,·) = 0`), and to assign spatial weight via a function `ψ ∈ C⁴(G)` that vanishes on `Γ` and has non-vanishing gradient outside a small set `G₁ ⋐ B`. The parameters `λ, μ ≥ 1` control the strength of the weights — making them large enough will later allow us to absorb lower-order terms into leading-order positive terms.

2. **Weighted identities (Theorems 3.1 and 3.2):** These are pointwise (almost-everywhere) identities for the transformed variables `h = θz` and `h_Γ = θz_Γ`, derived by expanding `2θ[∇·(A∇h) + Ah][dz + ∇·(A∇z)dt]` in the bulk and an analogous expression on the boundary, then rearranging terms into a sum of a positive quadratic form, a divergence (which vanishes upon integration), a time differential, and an Itô correction. These identities are purely algebraic — no inequalities yet, no parameter choices — and they hold for any sufficiently smooth `ℓ = log θ`. Their purpose is to expose the structure: certain combinations of derivatives of `h` appear as perfect squares times positive coefficients, while others can be estimated from below using the ellipticity of `A` and `A_Γ`.

3. **Auxiliary function selection (equations 3.2):** The weighted identities contain free functions `Ψ(t,x)` and `Φ(t,x)` that can be chosen to simplify the expressions for `A, B, c_{jk}` (bulk coefficients) and `Ã, B̃` (boundary coefficients). The paper selects `Ψ = -2 Σ a^{jk} ℓ_{x_j x_k}` to cancel the second-order terms in `A`, and `Φ = 2(β₀/M)(Aν·ν)|∇ℓ|` to provide a positive boundary term involving `|∇_Γh_Γ|²` that can absorb problematic boundary integrals.

4. **Coefficient estimates (Lemmas 3.1 and 3.2):** With the specific choice `ℓ = λμ(e^{μψ} - e^{2μ‖ψ‖∞})γ(t)`, the coefficients `A, B, c_{jk}` and `Ã, B̃` are estimated from below in terms of powers of `λ, μ, φ`. For instance, `B ≥ 2β₀² λ³ μ⁴ φ³ |∇ψ|⁴ + (lower order)`, which will dominate the `h²` term in the integrated identity, and `Σ c_{jk} η^j η^k ≥ (β₀² λ μ² φ |∇ψ|² + lower) |η|²`, which dominates the gradient term. These estimates are valid only where `|∇ψ| > 0`, i.e., outside the small set `G₁` — the local term on `G₁` is what remains on the right-hand side of the Carleman estimate and is later absorbed via a cut-off argument.

5. **Integration and boundary term estimation (Lemma 3.3):** Integrating the bulk identity (3.3) over `Q = (0,T) × G` and the boundary identity (3.4) over `Σ = (0,T) × Γ`, taking expectations, and summing yields a preliminary inequality (3.15) with nine boundary integral terms `I₁` through `I₉`. These `I_i` arise from the divergence theorem applied to the bulk divergence terms (pushing volume integrals to the boundary) and from the dynamic boundary condition's own weighted identity. Each must be estimated above by combinations of the positive terms we want to keep (like `λ³μ³𝔼∬ θ²φ³z_Γ² dσdt` and `λμ𝔼∬ θ²φ|∇_Γz_Γ|² dσdt`) plus terms we can tolerate on the right-hand side (like `𝔼∬ θ²F₁² dxdt` and `λ²μ²𝔼∬ θ²φ²Z² dxdt`). This step involves repeated applications of Young's inequality `ab ≤ (ε/2)a² + (1/2ε)b²` with carefully balanced `ε` to ensure the coefficients of the desired positive terms remain strictly positive.

6. **Final absorption (end of Lemma 3.3):** The local gradient term on `G₁` is eliminated via a cut-off function `ζ ∈ C₀^∞(B, [0,1])` with `ζ ≡ 1` on `G₁`. Computing `d(θ²φ ζ²z²)` by Itô's formula and integrating yields `𝔼∬_{G₁} θ²φ|∇z|² dxdt ≤ C𝔼∬_B θ²(λ^{-2}μ^{-2}F₁² + λ²μ²φ³z²) dxdt`, which pushes the gradient on `G₁` to the observation region `B` at the cost of larger weights on `z`. Choosing `λ, μ` large enough then absorbs these into the left-hand side.

The output of Stage 1 is the intermediate Carleman estimate (3.12), which bounds `λ³μ⁴𝔼∬ θ²φ³z²`, `λ³μ³𝔼∬ θ²φ³z_Γ²`, `λμ²𝔼∬ θ²φ|∇z|²`, and `λμ𝔼∬ θ²φ|∇_Γz_Γ|²` by the source terms and the local observation of `z` on `B`.

**Stage 2 (Section 4): Duality method for incorporating weak divergence sources.**

7. **Controlled forward system (equation 4.1):** We consider an auxiliary forward stochastic parabolic equation with the same dynamic boundary condition structure but with carefully chosen source terms: the bulk drift contains `λ³μ⁴θ²φ³z + 1_B u`, where `z` is the solution of our original backward equation (1.2) and `u` is a control; the boundary drift contains `λ³μ³θ²φ³z_Γ`; and both bulk and boundary have stochastic controls `v₁ dW(t)` and `v₂ dW(t)`. The initial condition is zero. The goal is to select controls `(û, v̂₁, v̂₂)` such that the forward state `(ŷ, ŷ_Γ)` reaches zero at time `T`.

8. **Optimal control formulation (Proposition 4.1):** This is cast as minimizing a functional `J_ε` (equation 4.4) that penalizes the control energy (with weights `λ^{-3}μ^{-4}θ^{-2}φ^{-3}` for `u`, `λ^{-2}μ^{-2}θ^{-2}φ^{-2}` for `v₁,v₂`), the state energy (with weight `θ_ε^{-2}`, where `θ_ε` is a mollified version of `θ` that is non-singular at `t = 0,T`), and the terminal state `(y_ε(T), y_{ε,Γ}(T))` multiplied by `1/ε`. The `ε`-penalty forces the terminal state to zero as `ε → 0`. The minimizer is characterized by an optimality system (4.5)–(4.6): the controls are expressed in terms of the adjoint state `(r_ε, r_{ε,Γ}, R_{ε,1}, R_{ε,2})` as `u_ε = -1_B λ³μ⁴θ²φ³r_ε`, `v_{ε,1} = -λ²μ²θ²φ²R_{ε,1}`, `v_{ε,2} = -λ²μ²θ²φ²R_{ε,2}`.

9. **Duality-based energy estimate (equation 4.7):** Computing `d⟨(y_ε, y_{ε,Γ}), (r_ε, r_{ε,Γ})⟩_{𝕃²}` via Itô's formula, integrating, and using the optimality conditions yields an identity linking the control energy to the inner product of the original backward state `(z, z_Γ)` with the adjoint `(r_ε, r_{ε,Γ})`. This is where the magic happens: the term `∫_0^T ∫_G F·∇y_ε dxdt` that would appear when applying Itô's formula to `d⟨(y_ε, y_{ε,Γ}), (z, z_Γ)⟩_{𝕃²}` (see equation 4.22) has been *transferred* to the forward system's controls via the optimality system. Applying Young's inequality and the intermediate Carleman estimate (3.12) to the adjoint `(r_ε, r_{ε,Γ})` yields a uniform bound (4.18) on the controls, the forward state, and its gradients — all in terms of the weighted norms of `(z, z_Γ)`.

10. **Weak limit and null controllability (Step 3 of Proposition 4.1):** As `ε → 0`, the uniform bound (4.18) provides weak convergence of the controls and states to limits `(û, v̂₁, v̂₂, ŷ, ŷ_Γ)` that constitute a null-controlling triple for (4.1). The proof of this passage uses a duality argument with an arbitrary test pair `(f,g)` to show that the limit satisfies the weak formulation of (4.1).

11. **Completion of Theorem 1.1:** With the null-controlling triple in hand, we apply Itô's formula to `d⟨(ŷ, ŷ_Γ), (z, z_Γ)⟩_{𝕃²}` and integrate over `(0,T)`. Since `(ŷ(T), ŷ_Γ(T)) = (0,0)` and `(ŷ(0), ŷ_Γ(0)) = (0,0)`, the left-hand side vanishes, giving identity (4.22). The right-hand side expresses `λ³μ⁴𝔼∬ θ²φ³z² dxdt + λ³μ³𝔼∬ θ²φ³z_Γ² dσdt` as the sum of inner products of the backward state and sources with the forward state and controls. Using the bound (4.2) on `(ŷ, ŷ_Γ, û, v̂₁, v̂₂)` and applying Young's inequality, we absorb the source terms `(F, F_Γ)` — which previously prevented a direct estimate — into the left-hand side, plus additional source terms on the right. This yields the final Carleman estimate (1.8), where the divergence source terms `F, F_Γ` now appear on the *right-hand side* (weighted by `λ²μ²θ²φ²`) rather than needing to be multiplied by the solution directly.

12. **Gradient estimate completion:** A separate computation using Itô's formula on `‖θφ^{1/2}z‖²_{L²(G)}` and `‖θφ^{1/2}z_Γ‖²_{L²(Γ)}` yields estimates (4.25)–(4.33) that bound the weighted gradient norms `λμ𝔼∬ θ²φ|∇z|²` and `λμ𝔼∬ θ²φ|∇_Γz_Γ|²` in terms of the already-controlled weighted `L²` norms and the source terms. This completes the full estimate (1.8) with all four left-hand side terms.

### 3.3 Roadmap for the Deep Dive

- **First**, the weight functions `θ, φ, γ, α, ψ` — their definitions, the role of the Fursikov–Imanuvilov function `ψ`, and the analytic properties (1.6)–(1.7) that make them suitable weights. Without understanding these, nothing else makes sense.

- **Second**, the weighted identities (Theorems 3.1 and 3.2). These are the fundamental algebraic decompositions at the heart of the method. We'll explain each term, where it comes from, and why the free functions `Ψ` and `Φ` are introduced.

- **Third**, the choice of auxiliary functions `Ψ, Φ` and the resulting coefficient estimates (Lemmas 3.1 and 3.2). This is where the abstract identities are concretized with the specific weight function `ℓ = λμ(e^{μψ} - e^{2μ‖ψ‖∞})γ(t)`. The estimates reveal what powers of `λ, μ, φ` emerge and why the conditions `λ ≥ C(e^{2μ‖ψ‖∞}T + T²)` and `μ ≥ C` are needed.

- **Fourth**, the integration and boundary estimation in Lemma 3.3. This is the longest and most technically demanding part of Stage 1. We'll trace each of the nine boundary terms `I₁` through `I₉`, explaining their origin, the estimation strategy (Young's inequality with specific parameter choices), and how the surface diffusion `∇_Γ·(A_Γ∇_Γz_Γ)` provides the crucial mechanism to absorb `I₂` into the positive terms.

- **Fifth**, the cut-off argument that eliminates the local gradient term on `G₁` (equation 3.40). This appears minor but is essential for the estimate to be useful — without it, the right-hand side would contain `‖∇z‖` on `G₁`, which cannot be absorbed.

- **Sixth**, the duality method in Section 4. We'll explain why the weighted identity method fails for `H⁻¹` sources, the construction of the controlled forward system (4.1), the optimal control problem (4.4) — including the `ε`-regularization via `θ_ε` and the terminal penalty — and the optimality system (4.5)–(4.6).

- **Seventh**, the duality computation (4.7) that links the control problem to the backward equation, and the subsequent estimates (4.8)–(4.18) that provide uniform bounds independent of `ε`.

- **Eighth**, the final assembly of Theorem 1.1 via the identity (4.22), Young's inequality, and the gradient supplement (4.25)–(4.33). This is where the estimate transitions from having `F, F_Γ` on the right-hand side (which is acceptable — they are known source terms) to having them bounded by the left-hand side (which would be circular).

### 3.4 Detailed, Sentence-Based Technical Breakdown

This is primarily an **analytical estimation paper** whose core idea is that a two-stage approach — direct weighted energy estimates for L² sources, followed by a duality/control argument to absorb H⁻¹ sources — can yield a Carleman estimate that handles anisotropic diffusion, dynamic boundary coupling, and weak divergence terms simultaneously, with an explicit dependence of all constants on the final time `T`.

---

#### The Weight Functions and the Fursikov–Imanuvilov Construction

The weight functions defined in (1.6) are:

$$\gamma(t) = \frac{1}{t(T-t)}, \quad \alpha(t,x) = \left(e^{\mu\psi(x)} - e^{2\mu\|\psi\|_\infty}\right)\gamma(t), \quad \ell(t,x) = \lambda\alpha(t,x), \quad \theta(t,x) = e^{\ell(t,x)}, \quad \varphi(t,x) = e^{\mu\psi(x)}\gamma(t)$$

where `ψ ∈ C⁴(G)` is a function satisfying `ψ > 0` in `G`, `ψ = 0` on `Γ`, and `|∇ψ| > 0` in `G \ G₁` for some nonempty open `G₁ ⋐ G` (Lemma 1.1), `λ, μ ≥ 1` are large parameters to be chosen, and `‖ψ‖∞ = max_{x∈G} ψ(x)`.

**What each function does:**

- `γ(t)` is the time-only part: it blows up like `1/t` as `t → 0⁺` and like `1/(T-t)` as `t → T⁻`. This ensures that `θ(0,·) = θ(T,·) = 0` (since `α → -∞` at the time boundaries, so `e^α → 0`), which makes all boundary terms at `t = 0` and `t = T` vanish when we integrate by parts in time.

- `ψ(x)` is the spatial weight function whose existence is guaranteed by the classical Fursikov–Imanuvilov lemma [19]. The key property is `|∇ψ| > 0` outside `G₁` — this means that in most of the domain, the spatial derivatives of the weights are non-degenerate, providing positive coefficients for the gradient terms in the Carleman estimate. The set `G₁` where `∇ψ` might vanish is contained in the observation region `B`, which is why a local term on `B` appears on the right-hand side of the Carleman estimate.

- `α(t,x)` is the combined weight: `e^{μψ(x)} - e^{2μ‖ψ‖∞}` is strictly negative (since `ψ(x) ≤ ‖ψ‖∞`, so `e^{μψ(x)} ≤ e^{μ‖ψ‖∞} < e^{2μ‖ψ‖∞}` for `μ > 0`), making `α(t,x) < 0` for all `t ∈ (0,T)`. Multiplying by `λ ≥ 1` amplifies the weight.

- `θ = e^{λα}` is the main weight function that will multiply the solution `z` to form `h = θz`. Because `α < 0` and `λ` is large, `θ` is exponentially small in the interior of `(0,T) × G` — this is what makes the weighted norms `𝔼∬ θ²φ³z² dxdt` finite even if `z` itself is merely in `L²_F(0,T; H¹)`.

- `φ = e^{μψ}γ(t)` is a secondary weight that is strictly positive on `(0,T) × G`. On the boundary `Γ`, since `ψ = 0`, we have `φ|_Γ = γ(t)`, which is purely time-dependent. This means `∇_Γφ = 0` on `Γ`, and consequently `∇_Γα = 0` on `Γ` — a crucial simplification that eliminates tangential derivatives of the weight on the boundary and makes the boundary integral estimates tractable.

**Analytic properties (1.7):** The estimates

$$\varphi \geq 4T^{-2}, \quad |\varphi_t| \leq CT\varphi^2, \quad |\varphi_{tt}| \leq CT^2\varphi^3, \quad |\alpha_t| \leq CT e^{2\mu\|\psi\|_\infty} \varphi^2, \quad |\alpha_{tt}| \leq CT^2 e^{2\mu\|\psi\|_\infty} \varphi^3$$

show that time derivatives of the weights are controlled by higher powers of `φ` itself. This "self-improving" property means that when we differentiate the weights (as happens in Itô's formula and integration by parts), we get terms that can be absorbed by making `λ` and `μ` large relative to `T`. Specifically, the condition `λ ≥ C(e^{2μ‖ψ‖∞}T + T²)` in Theorem 1.1 is precisely what is needed to ensure that `λφ` dominates `λT e^{2μ‖ψ‖∞} φ²` and `λT² φ²` — i.e., the leading-order positive terms outpace the time-derivative error terms.

**Why this particular form:** The choice `α = (e^{μψ} - e^{2μ‖ψ‖∞})γ(t)` rather than a simpler form like `α = -ψ(x)/t(T-t)` provides two advantages: (1) the constant shift `-e^{2μ‖ψ‖∞}` ensures `α < 0` uniformly even when `ψ(x)` is small, making `θ` genuinely small in the interior; (2) the `e^{μψ}` structure in `φ` gives `∇φ = μφ∇ψ`, so spatial derivatives of the weights introduce gain factors of `μ` that strengthen the positive terms relative to the error terms.

**The parameter `μ` is fixed first, then `λ`:** The order of parameter selection is crucial. First, `μ` is chosen large enough (`μ ≥ μ₀`) so that the spatial gradient `μ|∇ψ|` dominates lower-order terms from the matrix coefficients `a^{jk}` and their derivatives (the constants `M, M_Γ`). Then, with `μ` fixed, `λ` is chosen large enough (`λ ≥ λ₀(e^{2μ‖ψ‖∞}T + T²)`) to handle the time-derivative terms and the source terms. This two-stage selection avoids circular dependencies between `λ` and `μ`.

---

#### The Bulk Weighted Identity (Theorem 3.1)

The bulk weighted identity is a pointwise equality (holding for a.e. `(t,x,ω)`) for the transformed variable `h = θz = e^ℓ z`:

$$2\theta\left[\nabla\cdot(A\nabla h) + \mathcal{A}h\right]\left[dz + \nabla\cdot(A\nabla z)dt\right] - 2\nabla\cdot(A\nabla h \, dh) + 2\sum_{j,k=1}^N \left[\sum_{j',k'=1}^N \left(2a^{jk}a^{j'k'}\ell_{x_{j'}}h_{x_j}h_{x_{k'}} - a^{jk}a^{j'k'}\ell_{x_j}h_{x_{j'}}h_{x_{k'}}\right) - \Psi a^{jk}h_{x_j}h + a^{jk}\left(\mathcal{A}\ell_{x_j} + \frac{\Psi_{x_j}}{2}\right)h^2\right]_{x_k} dt$$

$$= 2\sum_{j,k=1}^N C^{jk} h_{x_j}h_{x_k} dt + \mathcal{B} h^2 dt + d\left(-\sum_{j,k=1}^N a^{jk}h_{x_j}h_{x_k} + \mathcal{A}h^2\right) + 2\left[\nabla\cdot(A\nabla h) + \mathcal{A}h\right]^2 dt + \theta^2\sum_{j,k=1}^N a^{jk}(dz_{x_j} + \ell_{x_j}dz)(dz_{x_k} + \ell_{x_k}dz) - \theta^2\mathcal{A}(dz)^2$$

where `A = (a^{jk})_{1≤j,k≤N}` is the diffusion matrix, `A = Σ a^{jk} ℓ_{x_j}ℓ_{x_k} - Σ a^{jk}_{x_k}ℓ_{x_j} - Σ a^{jk}ℓ_{x_j x_k} - Ψ - ℓ_t`, `B = 2[AΨ + Σ (Aa^{jk}ℓ_{x_j})_{x_k}] - A_t + Σ (a^{jk}Ψ_{x_k})_{x_j}`, `C^{jk} = Σ[2a^{jk'} (a^{j'k}ℓ_{x_{j'}})_{x_{k'}} - (a^{jk}a^{j'k'}ℓ_{x_{j'}})_{x_{k'}}] + (a^{jk}_t)/2 - Ψa^{jk}`, with `Ψ ∈ C^{1,2}((0,T)×G)` a free auxiliary function to be chosen, and `d(·)` denotes the Itô stochastic differential.

**What this identity does:** It expands the product of the elliptic operator applied to `h` (i.e., `∇·(A∇h) + Ah`) and the original stochastic PDE for `z` (which is `dz + ∇·(A∇z)dt = F₁dt + ZdW`). Through repeated use of the product rule and Itô's formula, this product is decomposed into:

- **A quadratic form in `∇h`** (`2 Σ C^{jk} h_{x_j}h_{x_k}`) — this will become the main gradient positive term. The coefficients `C^{jk}` depend on `A`, `ℓ`, and `Ψ`, and for proper choices of `Ψ` and `ℓ` they will be positive-definite (Lemma 3.1).
- **A zero-order positive term** (`B h²`) — this will dominate the `L²` norm of `h`. The coefficient `B` contains `λ³μ⁴φ³` as its leading term, making it extremely large for large `λ, μ`.
- **A perfect square** (`2[∇·(A∇h) + Ah]²`) — this is always non-negative and is discarded in the inequality (it only helps).
- **A time differential** (`d(-A∇h·∇h + Ah²)`) — when integrated over `(0,T)`, this vanishes because `θ(0,·) = θ(T,·) = 0` implies `h(0,·) = h(T,·) = 0`.
- **A divergence in space** (the `[⋯]_{x_k}` terms and `2∇·(A∇h dh)`) — when integrated over `G`, the divergence theorem converts these to boundary integrals on `Γ`, which must be matched with the boundary identity (Theorem 3.2).
- **Itô correction terms** (`θ²a^{jk}(dz_{x_j} + ℓ_{x_j}dz)(dz_{x_k} + ℓ_{x_k}dz)` and `-θ²A(dz)²`) — these arise from the quadratic variation of the martingale part. Since `dz = (⋯)dt + ZdW`, the term `(dz)²` equals `Z² dt`, giving a controlled source term on the right-hand side.

**Why introduce the free function `Ψ`:** Without `Ψ` (i.e., `Ψ ≡ 0`), the coefficient `A` would be `Σ a^{jk}ℓ_{x_j}ℓ_{x_k} - Σ a^{jk}_{x_k}ℓ_{x_j} - Σ a^{jk}ℓ_{x_j x_k} - ℓ_t`. The term `-Σ a^{jk}ℓ_{x_j x_k}` is second-order in `ℓ` and contains `μ²` factors — but these are *negative* contributions to `A`, which would weaken the positive term `Ah²`. By choosing `Ψ = -2 Σ a^{jk} ℓ_{x_j x_k}`, we set `A = Σ a^{jk}ℓ_{x_j}ℓ_{x_k} - Σ a^{jk}_{x_k}ℓ_{x_j} + Σ a^{jk}ℓ_{x_j x_k} - ℓ_t`, which approximates `Σ a^{jk}ℓ_{x_j}ℓ_{x_k}` (the square of the gradient of the weight) up to lower-order corrections. This makes `A ≈ λ²μ²φ²|A^{1/2}∇ψ|²` — a large positive coefficient that gives the crucial `λ²μ²φ²h²` term in the estimate.

**Why this form is not circular:** The identity is an equality, not an estimate — it holds for *any* smooth enough `ℓ` and `Ψ`, with no assumptions on `z` beyond sufficient regularity. The "magic" is in the algebraic decomposition: by grouping terms cleverly (following the template of [34, Theorem 9.26]), the right-hand side contains only terms that are either (i) positive-definite quadratic forms, (ii) perfect squares (non-negative), (iii) exact differentials (vanish upon integration), or (iv) terms that can be controlled by the positive terms when `λ, μ` are large. No circular reasoning is involved because the identity itself is mathematically exact.

---

#### The Boundary Weighted Identity (Theorem 3.2)

The boundary weighted identity for `h_Γ = θz_Γ` (with `z_Γ = z|_Γ`) is:

$$2\theta\left[\nabla_\Gamma\cdot(A_\Gamma\nabla_\Gamma h_\Gamma) + \tilde{\mathcal{A}}h_\Gamma\right]\left[dz_\Gamma + \nabla_\Gamma\cdot(A_\Gamma\nabla_\Gamma z_\Gamma)dt - \partial_\nu^A z \, dt\right] - 2\nabla_\Gamma\cdot(A_\Gamma\nabla_\Gamma h_\Gamma \, dh_\Gamma)$$

$$= \tilde{\mathcal{B}} h_\Gamma^2 dt + d\left(-A_\Gamma\nabla_\Gamma h_\Gamma \cdot \nabla_\Gamma h_\Gamma + \tilde{\mathcal{A}}h_\Gamma^2\right) + 2\left[\nabla_\Gamma\cdot(A_\Gamma\nabla_\Gamma h_\Gamma) + \tilde{\mathcal{A}}h_\Gamma\right]^2 dt + (A_\Gamma)_t \nabla_\Gamma h_\Gamma \cdot \nabla_\Gamma h_\Gamma dt + A_\Gamma d\nabla_\Gamma h_\Gamma \cdot d\nabla_\Gamma h_\Gamma - \theta^2\tilde{\mathcal{A}}|dz_\Gamma|^2 - 2\nabla_\Gamma\cdot(\Phi h_\Gamma A_\Gamma\nabla_\Gamma h_\Gamma) dt + 2\Phi A_\Gamma\nabla_\Gamma h_\Gamma \cdot \nabla_\Gamma h_\Gamma dt + 2h_\Gamma A_\Gamma\nabla_\Gamma h_\Gamma \cdot \nabla_\Gamma\Phi dt - 2\theta^2\partial_\nu^A z \, \nabla_\Gamma\cdot(A_\Gamma\nabla_\Gamma z_\Gamma) dt - 2\theta^2\tilde{\mathcal{A}} z_\Gamma \partial_\nu^A z \, dt$$

where `Ã = Φ - ℓ_t`, `B̃ = -2ÃΦ - Ã_t`, and `Φ ∈ C^{1,0}((0,T)×Γ)` is a second free auxiliary function (on the boundary only), and `∂_ν^A z = Σ_{j,k} a^{jk} (∂z/∂x_j) ν_k` is the conormal derivative.

**Key differences from the bulk identity:**

1. **The tangential operators:** All derivatives are tangential: `∇_Γ` (tangential gradient), `∇_Γ·` (tangential divergence), `A_Γ` (surface diffusion matrix). These act on the `(N-1)`-dimensional manifold `Γ`, not in the ambient `ℝ^N`.

2. **The coupling terms:** The last two terms `-2θ²∂_ν^A z ∇_Γ·(A_Γ∇_Γz_Γ)dt` and `-2θ²Ã z_Γ ∂_ν^A z dt` involve the conormal derivative `∂_ν^A z`, which couples the bulk solution's normal derivative to the boundary equation. These are the terms that make dynamic boundary conditions fundamentally different from static ones — they cannot be eliminated by choosing `Φ` cleverly; they must be estimated together with the bulk boundary terms.

3. **The free function `Φ`:** While `Ψ` in the bulk was chosen to simplify `A`, the boundary auxiliary `Φ` is chosen to *create a positive boundary term*. Specifically, the term `2Φ A_Γ∇_Γh_Γ·∇_Γh_Γ dt` in the identity provides a weighted `|∇_Γh_Γ|²` contribution that can be tuned via `Φ`. By choosing `Φ = 2(β₀/M)(Aν·ν)|∇ℓ|` (with `M = max ‖a^{jk}‖_{C²(G)}`), this term becomes proportional to `λμφ|∂_ν^A ψ| |∇_Γh_Γ|²`, which is positive (since `∂_ν^A ψ < 0` on `Γ`) and large. This positive term (which becomes `I₂` in Lemma 3.3, equation (3.17)) is precisely what is needed to *absorb* the problematic boundary integral `λμ𝔼∬ θ²φ|∂_ν^A ψ| |∇_Γz_Γ|² dσdt` that appears from estimating the bulk divergence term `I₃^1` (see the discussion around equation 3.21).

**Why `Φ` must depend on `∇ℓ`:** The bulk-boundary coupling generates terms proportional to `∇ℓ` on the boundary (because `∇h = θ∇z + θz∇ℓ`, and when taking divergences, the `∇ℓ` part hits the boundary). These generate boundary integrals of the form `λμ𝔼∬ θ²φ ∂_ν^A ψ |∇_Γz_Γ|² dσdt` that are *dangerous* — they have the same order as the positive gradient term we want on the left-hand side, but with the *wrong sign* (they come from the right-hand side of the divergence theorem). By choosing `Φ` proportional to `|∇ℓ|`, the `2Φ A_Γ∇_Γh_Γ·∇_Γh_Γ` term in the boundary identity provides a *positive* contribution of the same order that can cancel or dominate the dangerous term. The specific coefficient `2β₀/M` is chosen to ensure the net coefficient of `|∇_Γz_Γ|²` is strictly positive (see the estimate of `I₂` in (3.17) and its combination with (3.21) in the next subsection).

---

#### Choice of Auxiliary Functions and Coefficient Estimates

With `ℓ` given by (1.6), we have explicit expressions for its derivatives (3.10):

$$\ell_t = \lambda\alpha_t, \quad \ell_{x_j} = \lambda\mu\varphi\psi_{x_j}, \quad \ell_{x_j x_k} = \lambda\mu^2\varphi\psi_{x_j}\psi_{x_k} + \lambda\mu\varphi\psi_{x_j x_k}$$

where `ψ_{x_j}` are bounded (since `ψ ∈ C⁴(G)`) and `φ = e^{μψ}γ(t)`.

**Choice of `Ψ`:** As discussed, `Ψ = -2 Σ_{j,k} a^{jk} ℓ_{x_j x_k}`. Substituting the expression for `ℓ_{x_j x_k}`, we get:

$$\Psi = -2\sum_{j,k} a^{jk}\left(\lambda\mu^2\varphi\psi_{x_j}\psi_{x_k} + \lambda\mu\varphi\psi_{x_j x_k}\right) = -2\lambda\mu^2\varphi(A\nabla\psi\cdot\nabla\psi) + \lambda\mu\varphi O(1)$$

where `O(1)` denotes a bounded function depending on `ψ_{x_j x_k}` and the matrix entries.

**Estimated coefficient `A` (bulk):** From Lemma 3.1, with `Ψ` chosen as above and `μ ≥ C`, `λ ≥ CT²`:

$$\mathcal{A} = \lambda^2\mu^2\varphi^2\sum_{j,k} a^{jk}\psi_{x_j}\psi_{x_k} + \lambda\varphi O(\mu^2) + \lambda T\varphi^2 O(e^{2\mu\|\psi\|_\infty})$$

The leading term `λ²μ²φ²(A∇ψ·∇ψ)` is `O(λ²μ²φ²)` — this is the dominant positive contribution. The `λφ O(μ²)` term is lower-order in `λ` (only `λ` not `λ²`) and in `φ` (only `φ` not `φ²`). The `λTφ²O(e^{2μ‖ψ‖∞})` term is proportional to `T`, which is why the condition `λ ≥ CT²` is needed: to make `λ²μ²φ²` dominate `λTφ²` when `φ` is large.

**Estimated coefficient `B` (bulk):** The lower bound (from Lemma 3.1):

$$\mathcal{B} \geq 2\beta_0^2\lambda^3\mu^4\varphi^3|\nabla\psi|^4 + \lambda^3\varphi^3 O(\mu^4) + \lambda^2 T\varphi^3 O(\mu^2 e^{2\mu\|\psi\|_\infty})$$

The crucial feature: the leading coefficient of `h²` is `λ³μ⁴φ³|∇ψ|⁴`. This is third-order in `λ`, fourth-order in `μ`, and third-order in `φ`. The `|∇ψ|⁴` factor means this coefficient is strictly positive wherever `∇ψ ≠ 0` — which is everywhere in `G \ G₁`. On `G₁`, `∇ψ` may vanish, so `B` may not be positive there, which is why the Carleman estimate ultimately has a local term on the observation set `B` (which contains `G₁`) on the right-hand side.

**Estimated coefficient `C^{jk}` (bulk gradient):** For any `η ∈ ℝ^N`:

$$\sum_{j,k} \mathcal{C}^{jk} \eta_j \eta_k \geq \left(\beta_0^2 \lambda\mu^2\varphi|\nabla\psi|^2 + \lambda\varphi O(\mu)\right)|\eta|^2$$

The leading term `β₀²λμ²φ|∇ψ|²` controls the `|∇h|²` term. Note the absence of `φ²` — this is first-order in `φ`, which means the gradient weight is `θ²φ` (not `θ²φ²` or `θ²φ³`), matching the left-hand side of the Carleman estimate (1.8).

**Why the `O(·)` notation depends on `T, e^{μ‖ψ‖∞}`:** The constants in the `O(·)` terms involve derivatives of `ψ` (up to order 4) and the coefficient matrices `A, A_Γ` (up to `C²` in bulk, `C¹` on boundary). The `T` factors come from time derivatives of `φ` and `α` (see 1.7). The `e^{μ‖ψ‖∞}` factors come from the `e^{μψ}` inside `φ` — since `φ` itself is `O(e^{μ‖ψ‖∞}γ(t))`, higher powers of `φ` pick up higher exponentials. These are *not* problematic as long as `λ` is chosen large relative to `e^{μ‖ψ‖∞}T`, which is exactly the condition in Theorem 1.1.

**Boundary coefficients (Lemma 3.2):** With `Φ = 2(β₀/M)(Aν·ν)|∇ℓ|` and noting that on `Γ`: `∇ℓ = λμφ(∂_ν ψ)ν` (since `∇_Γψ = 0` on `Γ`), and `∂_ν^A ψ = (Aν·ν)∂_ν ψ < 0` (since `∂_ν ψ < 0`), we get:

$$\tilde{\mathcal{A}} = -2\frac{\beta_0}{M}\lambda\mu\varphi\,\partial_\nu^A\psi + \lambda T\varphi^2 O(e^{2\mu\|\psi\|_\infty})$$

$$\tilde{\mathcal{B}} = -8\frac{\beta_0^2}{M^2}\lambda^2\mu^2\varphi^2|\partial_\nu^A\psi|^2 + \lambda^2 T\varphi^3 O(\mu e^{2\mu\|\psi\|_\infty}) + \lambda T^2\varphi^3 O(e^{4\mu\|\psi\|_\infty})$$

The leading term in `B̃` is `-8(β₀²/M²)λ²μ²φ²|∂_ν^A ψ|²`, which is *negative*. This might seem concerning — why would we want a *negative* `h_Γ²` coefficient? The answer: `B̃ h_Γ²` appears on the right-hand side of the boundary identity (3.4) with a *positive* sign when we move everything to one side, so it becomes a negative contribution to the left-hand side of the inequality after integration. However, the magnitude of this negative term (order `λ²μ²φ²`) is *lower* than the positive bulk `h²` term (order `λ³μ⁴φ³`) when `λ, μ` are large. Moreover, the positive boundary term coming from `Φ|∇_Γh_Γ|²` (which is `I₂` in Lemma 3.3, order `λμ`) dominates any residual negative effects from `B̃`. The estimation in Step 2 of Lemma 3.3 explicitly shows that the combination of `I₁` (containing `B̃`) and `I₂` (containing `Φ|∇_Γh_Γ|²`) yields a net positive boundary contribution after `λ, μ` are chosen sufficiently large.

---

#### Integration and Boundary Term Estimation (Lemma 3.3, Steps 1–2)

**Step 1: Integration of the identities.** Integrating the bulk identity (3.3) over `Q = (0,T)×G`, integrating the boundary identity (3.4) over `Σ = (0,T)×Γ`, taking expectations, and using `θ(0,·) = θ(T,·) = 0` (which makes all `d(⋯)` time-differential terms vanish), we obtain:

$$2\mathbb{E}\iint_Q \sum_{j,k} \mathcal{C}^{jk} h_{x_j}h_{x_k} dxdt + \mathbb{E}\iint_Q \mathcal{B} h^2 dxdt + \mathbb{E}\iint_\Sigma \tilde{\mathcal{B}} h_\Gamma^2 d\sigma dt + 2\beta_0\mathbb{E}\iint_\Sigma \Phi|\nabla_\Gamma h_\Gamma|^2 d\sigma dt$$

$$\leq \mathbb{E}\iint_Q \theta^2 F_1^2 dxdt + \mathbb{E}\iint_\Sigma \theta^2 F_2^2 d\sigma dt + \mathbb{E}\iint_Q \theta^2\mathcal{A} Z^2 dxdt + \mathbb{E}\iint_\Sigma \theta^2\tilde{\mathcal{A}} \hat{Z}^2 d\sigma dt + \sum_{i=1}^9 I_i$$

where `I₁ = 𝔼∬_Σ B̃ h_Γ² dσdt`, `I₂ = 2β₀𝔼∬_Σ Φ|∇_Γh_Γ|² dσdt`, and `I₃` through `I₉` are the boundary terms arising from:

- `I₃`: divergence of the bulk cubic terms `[⋯]_{x_k}` in (3.3), converted to boundary integrals via the divergence theorem. This splits into `I₃¹` (cubic in `∇h` and `h`), `I₃²` (from the `Ψ` term), and `I₃³` (from the `Aℓ_{x_j} + Ψ_{x_j}/2` term).
- `I₄`: divergence of `2A∇h dh`, also converted to a boundary integral.
- `I₅`: boundary Itô correction from `Ã Ẑ²`.
- `I₆`: the `2h_Γ A_Γ ∇_Γh_Γ·∇_ΓΦ` term from the boundary identity.
- `I₇`: the `(A_Γ)_t ∇_Γz_Γ·∇_Γz_Γ` term.
- `I₈`: the `2θ²∂_ν^A z ∇_Γ·(A_Γ∇_Γz_Γ)` coupling term.
- `I₉`: the `2θ²Ã z_Γ ∂_ν^A z` coupling term.

**Step 2: Estimating the boundary terms.** This is the most technically intensive part of the paper. Each `I_i` must be bounded above by a combination of:
- the positive left-hand side terms we want to keep (like `λ³μ³𝔼∬ θ²φ³z_Γ² dσdt`, `λμ𝔼∬ θ²φ|∇_Γz_Γ|² dσdt`);
- source terms that are tolerable on the right-hand side (like `‖θF₂‖²`, `‖λμ θφẐ‖²`);
- the bulk left-hand side terms (like `λ³μ⁴𝔼∬ θ²φ³z² dxdt`, `λμ²𝔼∬ θ²φ|∇z|² dxdt`).

The general strategy for each term is:
1. Express the integrand in terms of `z, z_Γ` and their derivatives (since `h = θz`, `h_Γ = θz_Γ`, `∇h = θ∇z + θz∇ℓ`, etc.).
2. Identify the highest-order part in `λ, μ, φ`.
3. Apply Young's inequality `ab ≤ (ε/2)a² + (1/(2ε))b²` with `ε` chosen so that the coefficient of the `a²` term (which we want to keep on the left) remains strictly positive, and the `b²` term can be absorbed into existing left-hand side terms or right-hand side source terms.

**Key estimation examples:**

- **`I₂` (positive boundary gradient, equation 3.17):** Directly expressed as `I₂ = -4λμM 𝔼∬ θ²φ ∂_ν^A ψ |∇_Γz_Γ|² dσdt` since `∂_ν^A ψ < 0` and `M > 0`, this is *positive* (negative times negative) and provides a large coefficient `λμ` for the `|∇_Γz_Γ|²` term on the boundary. This is the crucial term that distinguishes the dynamic boundary condition case from static boundary conditions — the surface diffusion `∇_Γ·(A_Γ∇_Γz_Γ)` couples with the weight `Φ` to produce a controllable positive boundary gradient term.

- **`I₃¹` (bulk cubic divergence, equation 3.21):** After computation, this yields boundary integrals involving `z_Γ²`, `z_Γ ∂_ν^A z`, `|∂_ν^A z|²`, and `(A∇z·∇z)(Aν·ν)`. The key estimate is the identity (from [1, Lemma 2.3]):

$$|\partial_\nu^A z|^2 - (A\nabla_\Gamma z_\Gamma\cdot\nu)^2 = (A\nu\cdot\nu)(A\nabla z\cdot\nabla z - A\nabla_\Gamma z_\Gamma\cdot\nabla_\Gamma z_\Gamma)$$

This is used to relate the boundary normal derivative `∂_ν^A z` (which we don't directly control) to the tangential gradient `∇_Γz_Γ` (which we do control via `I₂`). The estimate `|(A∇_Γz_Γ·ν)²| ≤ C|∇_Γz_Γ|²` (since `ν` is bounded) then gives:

$$|\partial_\nu^A z|^2 \leq C|∇_Γz_Γ|^2 + (A\nu\cdot\nu)(A\nabla z\cdot\nabla z - A\nabla_\Gamma z_\Gamma\cdot\nabla_\Gamma z_\Gamma)$$

which, when inserted into the expression for `I₃¹`, yields (after Young's inequality) the estimate (3.21):

$$I_3^1 \leq -\frac{2}{3}\lambda^3\mu^3\mathbb{E}\iint_\Sigma \theta^2\varphi^3\partial_\nu\psi|\partial_\nu^A\psi|^2 z_\Gamma^2 d\sigma dt + \frac{1}{2}\lambda\mu\mathbb{E}\iint_\Sigma \theta^2\varphi\partial_\nu\psi|\partial_\nu^A z|^2 d\sigma dt - 2\lambda\mu M \mathbb{E}\iint_\Sigma \theta^2\varphi\partial_\nu^A\psi |\nabla_\Gamma z_\Gamma|^2 d\sigma dt$$

The first term (order `λ³μ³`) provides a *positive* boundary `z_Γ²` contribution (since `∂_ν ψ < 0` and `|∂_ν^A ψ|² > 0`), which survives in the final estimate. The third term (order `λμ`) is *also positive* (since `∂_ν^A ψ < 0`) and provides a boundary gradient contribution — but it has the *same* form as `I₂` (equation 3.17). This is the crucial cancellation: `I₂` and the third term of `I₃¹` together give a net coefficient of `(-4λμM + 2λμM) = -2λμM < 0` times `∂_ν^A ψ`, which is *negative*, so the net contribution is *positive*. 

- **`I₄` (the `∇·(A∇h dh)` boundary term, equation 3.25):** This is the most complex, as it involves `∂_ν^A z` multiplied by the full boundary equation operator. Expanding `dh = θ(dz + ℓ_t z dt)` and using the boundary PDE to express `dz_Γ`, we get terms involving `z_Γ ∇_Γ·(A_Γ∇_Γz_Γ)`, `z_Γ ∂_ν^A z`, `z_Γ F₂`, `z_Γ² ℓ_t`, `∂_ν^A z ∇_Γ·(A_Γ∇_Γz_Γ)`, `|∂_ν^A z|²`, and `∂_ν^A z F₂`. Each is estimated using Young's inequality and the bounds `|ℓ_t| ≤ CλT e^{2μ‖ψ‖∞} φ²`. The result (3.26) shows that all these terms can be absorbed into `‖∇_Γz_Γ‖²`, `‖∂_ν^A z‖²`, `‖F₂‖²`, and the bulk `z²` terms with sufficient parameter largeness.

- **`I₅, I₆, I₇, I₈, I₉`** are estimated similarly, each being absorbed into existing left-hand side terms or source terms. For example, `I₅` (equation 3.28) gives `≤ Cλ²μ 𝔼∬ θ²φ² Ẑ² dσdt` for large `λ`, which is a right-hand side source term. `I₇` (equation 3.30) gives `≤ C𝔼∬ θ²|∇_Γz_Γ|² dσdt`, which can be absorbed into the `λμ𝔼∬ θ²φ|∇_Γz_Γ|²` left-hand side term for large `λμ` (since `φ ≥ 4T^{-2}`, the `λμ` factor makes it dominant).

**The crucial role of the surface diffusion:** As noted in Remark 1.3, the term `∇_Γ·(A_Γ∇_Γz_Γ)` in the boundary equation is *indispensable* for this estimation. Without it, there would be no `A_Γ` matrix on the boundary, no positive `Φ|∇_Γh_Γ|²` term in the identity, and no mechanism to absorb the `λμ𝔼∬ θ²φ|∂_ν^A ψ| |∇_Γz_Γ|²` coming from `I₃¹`. This is why the case `N = 1` is special: when `N = 1`, `Γ` is zero-dimensional (boundary points), `∇_Γ ≡ 0`, and the surface diffusion vanishes — the Carleman estimate must be proved by different means (as in [6]).

---

#### Step 3: Absorbing the Local Gradient Term via Cut-Off (Lemma 3.3, End)

After estimating all boundary terms and combining with the bulk estimates (3.35)–(3.38), we arrive at inequality (3.39), which has the form:

$$\lambda\mu^2\mathbb{E}\iint_Q \theta^2\varphi\left(|\nabla z|^2 + \lambda^2\mu^2\varphi^2 z^2\right) dxdt + \lambda^3\mu^3\mathbb{E}\iint_\Sigma \theta^2\varphi^3 z_\Gamma^2 d\sigma dt + \lambda\mu\mathbb{E}\iint_\Sigma \theta^2\varphi|\nabla_\Gamma z_\Gamma|^2 d\sigma dt$$

$$\leq \text{(source terms on RHS)} + \lambda\mu^2\mathbb{E}\int_0^T\int_{G_1} \theta^2\varphi\left(|\nabla z|^2 + \lambda^2\mu^2\varphi^2 z^2\right) dxdt$$

The problem is the last term: the integral is only over the small set `G_1` (where `∇ψ` might vanish), but it includes `|∇z|²` — a gradient term we cannot control directly without the `|∇ψ|²` positivity that exists outside `G_1`. 

**The cut-off argument (equation 3.40):** Choose a function `ζ ∈ C₀^∞(B; [0,1])` with `ζ ≡ 1` on `G₁`, where `B` is the observation set from Theorem 1.1 (with `G₁ ⋐ B ⋐ G`). Apply Itô's formula to `d(θ²φ ζ² z²)`:

$$d(\theta^2\varphi\zeta^2 z^2) = (\theta^2\varphi)_t \zeta^2 z^2 dt + 2\theta^2\varphi\zeta^2 z \, dz + \theta^2\varphi\zeta^2 (dz)^2$$

Substitute `dz` from the backward equation (3.1) (without divergence terms — we're still in the `F = F_Γ = 0` case), integrate over `Q`, and take expectations. The boundary terms at `t = 0, T` vanish due to `θ(0,·) = θ(T,·) = 0`. After rearranging and using the ellipticity `A∇z·∇z ≥ β₀|∇z|²`, we obtain:

$$2\beta_0\mathbb{E}\iint_Q \theta^2\varphi\zeta^2|\nabla z|^2 dxdt = -\mathbb{E}\iint_Q (\theta^2\varphi)_t \zeta^2 z^2 dxdt - 2\mathbb{E}\iint_Q \zeta^2 z A\nabla z\cdot\nabla(\theta^2\varphi) dxdt - 4\mathbb{E}\iint_Q \theta^2\varphi \zeta z A\nabla z\cdot\nabla\zeta dxdt + 2\mathbb{E}\iint_Q \theta^2\varphi\zeta^2 z F_1 dxdt + \mathbb{E}\iint_Q \theta^2\varphi\zeta^2 Z^2 dxdt + 2\mathbb{E}\iint_Q \theta^2\varphi\zeta^2 z Z dW(t)$$

The stochastic integral vanishes under expectation. Using the weight estimates (4.29): `|(θ²φ)_t| ≤ CTλ e^{2μ‖ψ‖∞} θ²φ³` and `|∇(θ²φ)| ≤ Cλμ θ²φ²`, and applying Young's inequality, the terms involving `|∇z|²` on the right-hand side can be absorbed into the left for large `λ, μ`. The result (3.40) is:

$$\mathbb{E}\iint_{G_1} \theta^2\varphi|\nabla z|^2 dxdt \leq \mathbb{E}\iint_Q \theta^2\varphi\zeta^2|\nabla z|^2 dxdt \leq C\mathbb{E}\iint_B \theta^2\left(\frac{1}{\lambda^2\mu^2}F_1^2 + \lambda^2\mu^2\varphi^3 z^2\right) dxdt$$

This pushes the local gradient on `G_1` to a *weighted z² term* on the larger observation set `B`, at the cost of a factor `λ²μ²φ³` (matching the order of the bulk `z²` term we already control). Substituting this back into (3.39) and choosing `λ, μ` large enough to absorb the `λμ² × λ²μ²φ³ = λ³μ⁴φ³` term on `B` into the left-hand side's `λ³μ⁴φ³` term on `G \ G₁`, we obtain the intermediate Carleman estimate (3.12).

---

#### The Duality Method: Controlled Forward System (Section 4, Equations 4.1–4.5)

The intermediate estimate (3.12) assumes `F = F_Γ = 0`. To incorporate the weak divergence terms `∇·F` and `∇_Γ·F_Γ`, we cannot simply multiply the equation by a test function — `F` is only `L²`, so `∇·F` exists only in `H⁻¹` and cannot be pointwise multiplied by `z`. 

**The duality idea:** Consider two systems — the original backward equation (1.2) for `(z, z_Γ)`, and a forward controlled system (4.1) for `(y, y_Γ)` with controls `(u, v₁, v₂)`. By Itô's formula for the product `d⟨(y, y_Γ), (z, z_Γ)⟩_{𝕃²}` and integration over `(0,T)`, the `H⁻¹` term `∇·F` in the backward equation gets paired with the *forward solution* `y`, where it appears as `-F·∇y` (after integration by parts, which is legitimate since `y ∈ H¹`). The divergence has been transferred from acting on `F` (which would give an `H⁻¹` distribution) to acting on `y` (which gives an `L²` function). 

**Construction of the forward system (4.1):** The forward system is:

$$
\begin{cases}
dy - \nabla\cdot(A\nabla y) dt = (\lambda^3\mu^4\theta^2\varphi^3 z + 1_B u) dt + v_1 dW(t) & \text{in } Q,\\
dy_\Gamma - \nabla_\Gamma\cdot(A_\Gamma\nabla_\Gamma y_\Gamma) dt + \partial_\nu^A y dt = \lambda^3\mu^3\theta^2\varphi^3 z_\Gamma dt + v_2 dW(t) & \text{on } \Sigma,\\
y_\Gamma = y|_\Gamma & \text{on } \Sigma,\\
(y, y_\Gamma)|_{t=0} = (0,0) & \text{in } G \times \Gamma.
\end{cases}
$$

The source terms `λ³μ⁴θ²φ³z` and `λ³μ³θ²φ³z_Γ` are deliberately chosen to match the left-hand side of the intermediate Carleman estimate (3.12). The controls are: `u` acting on the subdomain `B` (the observation region), `v₁` in the bulk stochastic term, `v₂` in the boundary stochastic term. The initial condition is zero.

**The goal:** Find controls `(û, v̂₁, v̂₂)` such that the solution `(ŷ, ŷ_Γ)` reaches zero at time `T`: `(ŷ(T,·), ŷ_Γ(T,·)) = (0,0)`. This is a null controllability problem for a forward system whose source terms depend on the backward solution `(z, z_Γ)` we are trying to estimate.

**The optimal control formulation (4.4):** To find such controls, we minimize the functional:

$$J_\varepsilon(u, v_1, v_2) = \frac{1}{2}\mathbb{E}\int_0^T\int_B \lambda^{-3}\mu^{-4}\theta^{-2}\varphi^{-3}u^2 dxdt + \frac{1}{2}\mathbb{E}\iint_Q \lambda^{-2}\mu^{-2}\theta^{-2}\varphi^{-2}v_1^2 dxdt + \frac{1}{2}\mathbb{E}\iint_\Sigma \lambda^{-2}\mu^{-2}\theta^{-2}\varphi^{-2}v_2^2 d\sigma dt + \frac{1}{2}\mathbb{E}\iint_Q \theta_\varepsilon^{-2}y^2 dxdt + \frac{1}{2}\mathbb{E}\iint_\Sigma \theta_\varepsilon^{-2}y_\Gamma^2 d\sigma dt + \frac{1}{2\varepsilon}\mathbb{E}\int_G y(T)^2 dx + \frac{1}{2\varepsilon}\mathbb{E}\int_\Gamma y_\Gamma(T)^2 d\sigma$$

where `θ_ε` is a mollified version of `θ` with `(t+ε)(T-t+ε)` replacing `t(T-t)` in `γ(t)`, ensuring `θ_ε` is bounded away from zero at `t = 0, T`. The first three terms penalize the controls with weights that are the *inverse* of the Carleman weights — this is the standard duality: controls are cheap where the Carleman weight is small. The fourth and fifth terms penalize the state with weight `θ_ε^{-2}` (which is large near `t = 0, T`), keeping the state bounded. The last two terms, multiplied by `1/ε`, force the terminal state to zero as `ε → 0`.

**The optimality system (4.5)–(4.6):** By standard convex optimization in Hilbert spaces (Euler–Lagrange equations), the unique minimizer `(u_ε, v_{ε,1}, v_{ε,2})` satisfies:

$$u_\varepsilon = -1_B \lambda^3\mu^4\theta^2\varphi^3 r_\varepsilon, \quad v_{\varepsilon,1} = -\lambda^2\mu^2\theta^2\varphi^2 R_{\varepsilon,1}, \quad v_{\varepsilon,2} = -\lambda^2\mu^2\theta^2\varphi^2 R_{\varepsilon,2}$$

where `(r_ε, r_{ε,Γ}, R_{ε,1}, R_{ε,2})` is the solution of the adjoint backward system (4.6):

$$
\begin{cases}
dr_\varepsilon + \nabla\cdot(A\nabla r_\varepsilon) dt = -\theta_\varepsilon^{-2} y_\varepsilon dt + R_{\varepsilon,1} dW(t) & \text{in } Q,\\
dr_{\varepsilon,\Gamma} + \nabla_\Gamma\cdot(A_\Gamma\nabla_\Gamma r_{\varepsilon,\Gamma}) dt - \partial_\nu^A r_\varepsilon dt = -\theta_\varepsilon^{-2} y_{\varepsilon,\Gamma} dt + R_{\varepsilon,2} dW(t) & \text{on } \Sigma,\\
r_{\varepsilon,\Gamma} = r_\varepsilon|_\Gamma & \text{on } \Sigma,\\
(r_\varepsilon, r_{\varepsilon,\Gamma})|_{t=T} = (\frac{1}{\varepsilon}y_\varepsilon(T), \frac{1}{\varepsilon}y_{\varepsilon,\Gamma}(T)) & \text{in } G \times \Gamma.
\end{cases}
$$

The adjoint equation has the same structure as the original backward equation (1.2) but with source terms `-θ_ε^{-2}y_ε` and `-θ_ε^{-2}y_{ε,Γ}` and terminal condition `(1/ε)` times the forward terminal state. The controls are expressed in terms of the adjoint variables with the *direct* (not inverse) Carleman weights — this is the key link that will allow us to bound the controls using the intermediate Carleman estimate applied to `(r_ε, r_{ε,Γ})`.

---

#### The Duality Computation and Uniform Estimates (Section 4, Equations 4.7–4.18)

**The duality identity (4.7):** Apply Itô's formula to `d⟨(y_ε, y_{ε,Γ}), (r_ε, r_{ε,Γ})⟩_{𝕃²}` and integrate over `(0,T)`. Using the PDEs for `y_ε` and `r_ε`, most terms cancel, leaving:

$$\frac{1}{\varepsilon}\mathbb{E}\int_G y_\varepsilon(T)^2 dx + \frac{1}{\varepsilon}\mathbb{E}\int_\Gamma y_{\varepsilon,\Gamma}(T)^2 d\sigma + \mathbb{E}\iint_Q \theta_\varepsilon^{-2} y_\varepsilon^2 dxdt + \mathbb{E}\iint_\Sigma \theta_\varepsilon^{-2} y_{\varepsilon,\Gamma}^2 d\sigma dt + \lambda^3\mu^4\mathbb{E}\int_0^T\int_B \theta^2\varphi^3 r_\varepsilon^2 dxdt + \lambda^2\mu^2\mathbb{E}\iint_Q \theta^2\varphi^2 R_{\varepsilon,1}^2 dxdt + \lambda^2\mu^2\mathbb{E}\iint_\Sigma \theta^2\varphi^2 R_{\varepsilon,2}^2 d\sigma dt$$

$$= \lambda^3\mu^4\mathbb{E}\iint_Q \theta^2\varphi^3 z r_\varepsilon dxdt + \lambda^3\mu^3\mathbb{E}\iint_\Sigma \theta^2\varphi^3 z_\Gamma r_{\varepsilon,\Gamma} d\sigma dt$$

The left-hand side is exactly the penalty terms from `J_ε` (evaluated at the optimal controls) plus the adjoint energy. The right-hand side couples the backward state `(z, z_Γ)` we want to estimate with the adjoint `(r_ε, r_{ε,Γ})`.

**Applying Young's inequality (4.8):** For any `ρ > 0`:

$$\lambda^3\mu^4\mathbb{E}\iint_Q \theta^2\varphi^3 z r_\varepsilon dxdt + \lambda^3\mu^3\mathbb{E}\iint_\Sigma \theta^2\varphi^3 z_\Gamma r_{\varepsilon,\Gamma} d\sigma dt$$

$$\leq \rho\left(\lambda^3\mu^4\mathbb{E}\iint_Q \theta^2\varphi^3 r_\varepsilon^2 dxdt + \lambda^3\mu^3\mathbb{E}\iint_\Sigma \theta^2\varphi^3 r_{\varepsilon,\Gamma}^2 d\sigma dt\right) + \frac{1}{4\rho}\left(\lambda^3\mu^4\mathbb{E}\iint_Q \theta^2\varphi^3 z^2 dxdt + \lambda^3\mu^3\mathbb{E}\iint_\Sigma \theta^2\varphi^3 z_\Gamma^2 d\sigma dt\right)$$

The first parentheses on the right can be *absorbed* using the intermediate Carleman estimate (3.12) applied to `(r_ε, r_{ε,Γ})`. Specifically, (3.12) with `F₁ = -θ_ε^{-2}y_ε`, `F₂ = -θ_ε^{-2}y_{ε,Γ}`, and `Z = R_{ε,1}`, `Ẑ = R_{ε,2}` gives:

$$\lambda^3\mu^4\mathbb{E}\iint_Q \theta^2\varphi^3 r_\varepsilon^2 dxdt + \lambda^3\mu^3\mathbb{E}\iint_\Sigma \theta^2\varphi^3 r_{\varepsilon,\Gamma}^2 d\sigma dt$$

$$\leq C\left(\lambda^3\mu^4\mathbb{E}\int_0^T\int_B \theta^2\varphi^3 r_\varepsilon^2 dxdt + \mathbb{E}\iint_Q \theta^2\theta_\varepsilon^{-4}y_\varepsilon^2 dxdt + \cdots\right)$$

Since `θ²θ_ε^{-4} ≤ θ_ε^{-2}` (because `θ ≤ θ_ε`), the right-hand side is bounded by the terms already present in (4.7). Choosing `ρ` small enough (specifically `Cρ < 1`, where `C` is the constant from the Carleman estimate), we can absorb the `ρ(⋯)` term into the left-hand side of (4.7), obtaining a uniform bound independent of `ε`:

$$\frac{1}{\varepsilon}\mathbb{E}\|(y_\varepsilon(T), y_{\varepsilon,\Gamma}(T))\|_{\mathbb{L}^2}^2 + \text{(state and control energies)} \leq C\left(\lambda^3\mu^4\mathbb{E}\iint_Q \theta^2\varphi^3 z^2 dxdt + \lambda^3\mu^3\mathbb{E}\iint_\Sigma \theta^2\varphi^3 z_\Gamma^2 d\sigma dt\right)$$

This is the estimate (4.10), which bounds the `ε`-penalized terminal state, the state energy, and the control energy purely in terms of the backward state `(z, z_Γ)` we are estimating.

**Forward gradient estimate (4.11)–(4.18):** To get bounds on `∇y_ε` and `∇_Γy_{ε,Γ}` (needed for the final estimate where `F·∇y` will appear), we compute `d(θ_ε^{-2}φ^{-2}y_ε²)` and `d(θ_ε^{-2}φ^{-2}y_{ε,Γ}²)`, integrate, and use Young's inequality. The key estimates (4.13) on the weight derivatives `|(θ_ε^{-2}φ^{-2})_t| ≤ CTλ e^{2μ‖ψ‖∞} θ_ε^{-2}` and `|∇(θ_ε^{-2}φ^{-2})| ≤ Cλμ θ_ε^{-2}φ^{-1}` mirror those for `θ²φ` in (4.29) but with inverse weights. The result (4.18) provides:

$$\lambda^{-2}\mu^{-2}\mathbb{E}\iint_Q \theta_\varepsilon^{-2}\varphi^{-2}|\nabla y_\varepsilon|^2 dxdt + \lambda^{-2}\mu^{-2}\mathbb{E}\iint_\Sigma \theta_\varepsilon^{-2}\varphi^{-2}|\nabla_\Gamma y_{\varepsilon,\Gamma}|^2 d\sigma dt \leq C\left(\lambda^3\mu^4\mathbb{E}\iint_Q \theta^2\varphi^3 z^2 dxdt + \cdots\right)$$

**Passage to the limit (Step 3, Proposition 4.1):** As `ε → 0`, the uniform bounds provide weak convergence:

$$u_\varepsilon \rightharpoonup \hat{u}, \quad v_{\varepsilon,1} \rightharpoonup \hat{v}_1, \quad v_{\varepsilon,2} \rightharpoonup \hat{v}_2, \quad y_\varepsilon \rightharpoonup \hat{y}, \quad y_{\varepsilon,\Gamma} \rightharpoonup \hat{y}_\Gamma$$

in the appropriate weighted spaces. To verify that `(ŷ, ŷ_Γ)` is indeed the solution of (4.1) with controls `(û, v̂₁, v̂₂)`, we use a test function argument: for arbitrary `(f,g)` in the appropriate spaces, we solve an auxiliary backward equation for test functions `(ϕ, ϕ_Γ)`, compute `d⟨(ϕ, ϕ_Γ), (y_ε, y_{ε,Γ}) - (ŷ, ŷ_Γ)⟩_{𝕃²}`, and take `ε → 0` to find that the difference vanishes. The terminal penalty `(1/ε)𝔼‖(y_ε(T), y_{ε,Γ}(T))‖²` being bounded (from 4.18) implies `(ŷ(T), ŷ_Γ(T)) = (0,0)`. The bound (4.2) follows from the uniform estimates (4.18) and weak lower semicontinuity of the norms.

---

#### Completion of Theorem 1.1: Assembling the Final Estimate

**The duality identity for the full system (4.22):** Apply Itô's formula to `d⟨(ŷ, ŷ_Γ), (z, z_Γ)⟩_{𝕃²}` where `(z, z_Γ)` is the solution of the *full* backward equation (1.2) (now including `F, F_Γ`) and `(ŷ, ŷ_Γ)` is the null-controlled solution of (4.1). Since both have zero initial/terminal conditions (`(ŷ(0), ŷ_Γ(0)) = (0,0)` by construction, `(z(T), z_Γ(T)) = (z_T, z_{Γ,T})` which gets paired with `(ŷ(T), ŷ_Γ(T)) = (0,0)`), the left-hand side vanishes, yielding:

$$\lambda^3\mu^4\mathbb{E}\iint_Q \theta^2\varphi^3 z^2 dxdt + \lambda^3\mu^3\mathbb{E}\iint_\Sigma \theta^2\varphi^3 z_\Gamma^2 d\sigma dt = -\mathbb{E}\iint_Q [1_B \hat{u} z + F_1\hat{y} - F\cdot\nabla\hat{y} + Z\hat{v}_1] dxdt - \mathbb{E}\iint_\Sigma [F_2\hat{y}_\Gamma - F_\Gamma\cdot\nabla_\Gamma\hat{y}_\Gamma + \hat{Z}\hat{v}_2] d\sigma dt$$

**Why the divergence terms now appear as `F·∇ŷ`:** In the Itô computation, the backward equation contributes the term `-∫_0^T ⟨∇·F, ŷ⟩_{H^{-1},H^1} dt`. Since `ŷ ∈ H¹`, this duality pairing equals `∫_0^T ∫_G F·∇ŷ dxdt` (by definition of the weak divergence, Lemma 2.1). The term has been transferred from an `H⁻¹` action on `z` to an `L²` inner product of `F` with `∇ŷ` — and we already have uniform bounds on `∇ŷ` from (4.2).

**Applying Young's inequality (4.23–4.24):** Bounding the right-hand side of (4.22) using `ab ≤ (ε/2)a² + (1/(2ε))b²` with the weights from the control bound (4.2):

$$-\mathbb{E}\iint_Q 1_B \hat{u}z \,dxdt \leq \frac{1}{8}\lambda^3\mu^4\mathbb{E}\iint_Q \theta^2\varphi^3 z^2 dxdt + 2\lambda^{-3}\mu^{-4}\mathbb{E}\iint_Q \theta^{-2}\varphi^{-3}\hat{u}^2 dxdt$$

and similarly for the other terms. The control energy terms on the right are bounded by (4.2) in terms of `λ³μ⁴𝔼∬ θ²φ³z² dxdt + λ³μ³𝔼∬ θ²φ³z_Γ² dσdt`. By choosing the `ε` in Young's inequality small enough (so that the coefficients of the `z²` and `z_Γ²` terms from the first part sum to less than 1), these can be *absorbed into the left-hand side*, yielding (4.24):

$$\lambda^3\mu^4\mathbb{E}\iint_Q \theta^2\varphi^3 z^2 dxdt + \lambda^3\mu^3\mathbb{E}\iint_\Sigma \theta^2\varphi^3 z_\Gamma^2 d\sigma dt$$

$$\leq C\left(\lambda^3\mu^4\mathbb{E}\iint_B \theta^2\varphi^3 z^2 dxdt + \lambda^2\mu^2\mathbb{E}\iint_Q \theta^2\varphi^2 Z^2 dxdt + \mathbb{E}\iint_Q \theta^2 F_1^2 dxdt + \lambda^2\mu^2\mathbb{E}\iint_Q \theta^2\varphi^2|F|^2 dxdt + \cdots\right)$$

This is exactly the Carleman estimate for the `z²` and `z_Γ²` norms, with the weak divergence source terms `F, F_Γ` now appearing (as desired) on the right-hand side weighted by `λ²μ²θ²φ²`, and the observation localized to `B`.

**The gradient supplement (4.25)–(4.33):** To also estimate `|∇z|²` and `|∇_Γz_Γ|²`, we compute the Itô differentials `d‖θφ^{1/2}z‖²_{L²(G)}` and `d‖θφ^{1/2}z_Γ‖²_{L²(Γ)}` separately, combine them using the boundary coupling, and obtain identity (4.28). After estimating the weight derivatives via (4.29) and applying Young's inequality, we get (4.33):

$$\lambda\mu\mathbb{E}\iint_Q \theta^2\varphi|\nabla z|^2 dxdt + \lambda\mu\mathbb{E}\iint_\Sigma \theta^2\varphi|\nabla_\Gamma z_\Gamma|^2 d\sigma dt$$

$$\leq C\left(\lambda^3\mu^3\mathbb{E}\iint_Q \theta^2\varphi^3 z^2 dxdt + \lambda^3\mu^2\mathbb{E}\iint_\Sigma \theta^2\varphi^3 z_\Gamma^2 d\sigma dt + \text{(source terms with }F, F_\Gamma, Z, \hat{Z}\text{)}\right)$$

The right-hand side contains `z²` and `z_Γ²` norms that we have *already bounded* by (4.24). Substituting (4.24) into (4.33) and combining yields the full Carleman estimate (1.8), with all four left-hand side terms bounded by the right-hand side source terms and the localized observation on `B`. This completes the proof.

**Why this two-step structure is essential:** Trying to apply the weighted identity method directly to (1.2) with `F, F_Γ ≠ 0` would generate terms like `2θ[∇·(A∇h) + Ah] (∇·F) dt` that cannot be integrated by parts (since `F` is only `L²`, `∇·F` is only `H⁻¹`, and multiplying by the continuous function `θ[∇·(A∇h) + Ah]` does not produce an integrable function). The duality method sidesteps this by working with the *dual* pairing `⟨∇·F, ŷ⟩` where `ŷ` is in `H¹` (so the pairing is well-defined) and the resulting `F·∇ŷ` term is integrable. The price paid is that we must solve a null controllability problem for (4.1) — but this is feasible precisely because the intermediate estimate (3.12) provides the necessary observability for the adjoint of the controlled system. There is no circularity because the intermediate estimate assumes `F = F_Γ = 0` and the controlled system (4.1) does not contain `F` or `F_Γ` — the weak sources appear only in the final duality step (4.22), after the controls have been constructed.

## 4. Key Insights and Innovations

### Innovation 1: A Two-Stage Synthesis Unlocks Carleman Estimates for Systems Where Neither Existing Method Alone Suffices

The paper's deepest conceptual contribution is not a new inequality or a new estimate, but a **methodological architecture** — a specific way of sequencing two known techniques (weighted identity and duality/optimization) so that their individual limitations cancel rather than compound.

Before this work, the literature recognized two separate bottlenecks:

- **Weighted identity methods** (as in Tang and Zhang [40], Baroun et al. [7]) could handle stochastic parabolic equations with dynamic boundary conditions, provided the drift source terms lived in `L²` — meaning they could be pointwise multiplied by the solution in the energy estimate. The presence of `∇·F` terms (with `F ∈ L²`, hence `∇·F ∈ H⁻¹`) broke the method because the product `(∇·F) × (weighted solution)` is not defined.

- **Duality/optimization methods** (as in Liu [29]) could handle `H⁻¹` source terms for stochastic parabolic equations, but only with **static boundary conditions** (Dirichlet). The duality argument couples the backward equation to a forward controlled system — but when the boundary condition is dynamic, that forward system itself has a surface parabolic equation with its own coupling and controls, which Liu's framework did not accommodate.

The isolated limitations were clear: each method failed on one axis of the problem. The standard response in the literature has been to either restrict the problem class (assume `F = F_Γ = 0`, accepting `H⁻¹`-free drift) or restrict the boundary condition (assume Dirichlet, accepting no surface dynamics). This paper is the first to recognize that **the two methods can be composed sequentially, with the output of the first serving as the input to the second, each covering the other's blind spot**.

Concretely (referring to the architecture laid out in Section 3):

1. **Stage 1** applies the weighted identity method to the *restricted* problem where `F = F_Γ = 0`. This stage inherits all the difficulty of the dynamic boundary condition — the nine boundary integral terms `I₁` through `I₉`, the anisotropic matrix coefficients `A` and `A_Γ`, the surface diffusion coupling — and produces the intermediate Carleman estimate (3.12). At this stage, the `H⁻¹` limitation is *accepted* rather than fought: the method simply assumes those terms are absent.

2. **Stage 2** then takes the intermediate estimate as a given and applies the duality/optimization method from [29], but **generalized** to the dynamic boundary setting. The crucial adaptation is that the forward controlled system (4.1) must now include a boundary equation with its own control `v₂` and its own coupling `∂_ν^A y`. The duality pairing `⟨(y, y_Γ), (z, z_Γ)⟩_{𝕃²}` now involves both bulk and boundary components, and the optimization functional `J_ε` penalizes both surface and volume state energies. This generalization is non-trivial — Section 3.4's analysis of the optimality system shows that the forward gradient estimate (4.11)–(4.18) requires separate treatment of the bulk and boundary Itô differentials `d(θ_ε^{-2}φ^{-2}y_ε²)` and `d(θ_ε^{-2}φ^{-2}y_{ε,Γ}²)`, with the conormal coupling term `⟨∂_ν^A y_ε, θ_ε^{-2}φ^{-2}y_{ε,Γ}⟩` appearing and canceling between them.

The synthesis is more than "apply method A, then method B." The reason it works — and the reason it is not circular — is that **Stage 1 establishes observability for the adjoint of the controlled system used in Stage 2**. The intermediate Carleman estimate (3.12) is precisely what is needed to bound the adjoint `(r_ε, r_{ε,Γ})` in the optimality system (4.6), which in turn provides the uniform bounds (4.18) on the controls. The author's key recognition is that the controlled system (4.1) has `F = F_Γ = 0` (its sources are `λ³μ⁴θ²φ³z` and `λ³μ³θ²φ³z_Γ`, which are `L²` functions, not `H⁻¹` distributions), so Stage 1's intermediate estimate — which assumes exactly that — applies to its adjoint without modification. The `H⁻¹` terms from the original backward equation appear only in the *final* duality step (4.22), after the controls have been constructed, and there they manifest as `F·∇ŷ` (well-defined since `∇ŷ ∈ L²`), not as `(∇·F) × (something)`.

This is a **fundamental methodological contribution**, not an incremental refinement. It transforms the problem from "neither method works" to "each method handles the part the other cannot, and their composition is exact." The paper's explicit acknowledgment that the weighted identity from [7, Theorem 4.1] "is not sufficient to address the control problems considered in this paper" and that the duality method from [29] must be "specifically designed to address weak divergence-type sources" makes clear that the synthesis itself is the intellectual contribution.

The architecture also suggests a **general template** for other PDE systems where multiple structural obstacles co-occur: isolate the obstacles, design an intermediate estimate that assumes them absent, then use that intermediate estimate to power a duality/control argument that absorbs the obstacles into the final estimate. Whether this template generalizes to quasilinear or fully nonlinear problems is an open question — the linearity of (1.2) is used essentially in the duality argument — but for the broad class of linear parabolic systems with boundary coupling and irregular source terms, this paper provides a blueprint.

---

### Innovation 2: Explicit T-Dependence in the Carleman Constant Transforms the Estimate from Qualitative to Quantitative

Carleman estimates in the stochastic PDE literature, including the direct predecessors [7] and [5] from the same authors, established the *existence* of parameters `λ₀, μ₀` such that the estimate holds for all `λ ≥ λ₀, μ ≥ μ₀`. But `λ₀` was typically expressed as "sufficiently large, depending on `T` and the coefficients" without an explicit functional form. This is sufficient for proving qualitative results — null controllability *holds*, an observability inequality *exists* — but insufficient for extracting quantitative information, such as how the minimal control norm scales with the time horizon `T` and the coefficient bounds.

The present paper derives the **explicit condition** (Theorem 1.1):

```
λ ≥ λ₀(e^{2μ‖ψ‖∞}T + T²)
```

with `μ ≥ μ₀` where `μ₀` depends only on `G, B, β₀, M, M_Γ` (not on `T`). This is not merely a cosmetic refinement. It enables the paper to track how every constant in the subsequent estimates — the Carleman constant `C` in (1.8), the observability constant in (5.7), and ultimately the null controllability cost in (1.15) — depends on `T` and the coefficient bounds `‖a₁‖∞, ‖a₂‖∞, ‖B₁‖∞, ‖B₂‖∞`.

The payoff appears in the **null controllability cost estimate** (1.15):

```
𝒦(Y₀, G₀) ≤ exp(C K_T) 𝔼‖Y₀‖²_{𝕃²}
```

where `K_T` (1.14) is an explicit, interpretable function:

```
K_T = 1 + 1/T + ‖a₁‖^{2/3}_∞ + ‖a₂‖^{2/3}_∞ + T(‖a₁‖∞ + ‖a₂‖∞) + (1+T)(‖B₁‖²_∞ + ‖B₂‖²_∞)
```

This formula has immediate practical meaning:
- The `1/T` term captures the familiar parabolic blow-up: as the control time shrinks, the required control energy grows exponentially. This is consistent with the infinite speed of propagation paradox — controlling exactly to zero in arbitrarily short time requires exponentially large controls.
- The `‖a₁‖^{2/3}_∞` term shows that reaction coefficients have a sub-linear effect on the cost (exponent `2/3` rather than `1` or `2`), a non-obvious scaling that emerges from the particular use of Young's inequality with exponents chosen to match the power structure of the Carleman weights (`λ³` for `z²` vs. `λ²` for the convection source terms).
- The `(1+T)‖B₁‖²_∞` term shows that convection costs scale quadratically in the velocity field and degrade as `T` grows — longer time horizons don't help when the drift is actively transporting the state away from the controlled region.

Before this work, the controllability literature for stochastic PDEs with dynamic boundary conditions could assert that controls *exist* [6, 7, 5, 11] but could not quantify their cost in terms of the problem data. This paper provides the **first cost estimate** for such systems, and the explicit `K_T` formula is directly usable: given specific coefficient bounds and a control time `T`, one can compute an upper bound on the required control energy.

The technical achievement that makes this possible is the **careful tracking of `T`-dependent terms throughout the entire chain of estimates**, from Lemma 3.1 (where `B` contains `λ² T φ³ O(μ² e^{2μ‖ψ‖∞})`) through the boundary term estimates (where `|ℓ_t|` contributes `CT e^{2μ‖ψ‖∞} φ²` to `I₄` and `I₉`) to the final absorption conditions in Section 4. At each step, the paper identifies the precise power of `T` multiplying each error term and ensures the condition on `λ` is strong enough to dominate it. The `e^{2μ‖ψ‖∞}` factor is particularly delicate — it grows exponentially with `μ`, which is fixed first, so `λ` must be chosen exponentially large relative to `T` to compensate. The condition `λ ≥ C(e^{2μ‖ψ‖∞}T + T²)` reflects the worst-case combination of the time-boundary blow-up (`T` from `φ_t`) and the spatial-weight amplification (`e^{2μ‖ψ‖∞}` from `φ²`).

This is a **fundamental advance** in the quantitative theory of stochastic PDE control. It shifts Carleman estimates from a purely existential tool ("there exists a weight such that...") to a quantitative one ("for this specific weight with these explicit parameters, the constant is..."). Whether similarly explicit `T`-dependence can be extracted for quasilinear or semilinear stochastic equations — where the Carleman estimate must be combined with fixed-point arguments that introduce additional constants — is an open and important direction.

---

### Innovation 3: The Surface Diffusion Term Is Diagnostically Essential, Not a Technical Convenience

Remark 1.3 states a negative result whose implications are more significant than its brevity suggests:

> "The surface diffusion term ... plays a crucial role in the derivation of the Carleman estimate (1.8), in particular in Step 2 of the proof of Lemma 3.3. ... Consequently, the null and insensitizing control problems for (1.9) under the dynamic boundary condition `dy_Γ + ∂_ν^A y dt = ...` remains an open problem."

This is not a minor restriction. It is a **structural diagnosis**: the paper identifies which term in the PDE is responsible for making the entire Carleman estimate work, and by removing it, shows that the remaining system is genuinely harder in a way that the current methodology cannot address.

Section 3's analysis reveals the mechanism. The boundary estimation step in Lemma 3.3 contains a delicate cancellation between `I₂` (the positive boundary gradient term `-4λμM 𝔼∬ θ²φ ∂_ν^A ψ |∇_Γz_Γ|² dσdt`, generated by the surface diffusion through the choice of `Φ`) and a dangerous term from `I₃¹` (equation 3.21) that also involves `|∇_Γz_Γ|²` but with a coefficient that would be uncontrolled without surface diffusion. The net coefficient is `(-4λμM + 2λμM) = -2λμM` times `∂_ν^A ψ < 0`, giving a *positive* total contribution to the left-hand side. If the surface diffusion is absent (`A_Γ ≡ 0`), there is no `∇_Γ·(A_Γ∇_Γz_Γ)` term in the boundary equation, no `Φ|∇_Γh_Γ|²` term in the boundary weighted identity (Theorem 3.2), and consequently no `I₂` to balance `I₃¹`. The dangerous `|∇_Γz_Γ|²` term remains uncontrolled.

The significance of this diagnosis goes beyond this specific paper. It tells future researchers:
- **Do not attempt to drop surface diffusion as a "simplifying assumption."** The resulting system is not a simpler special case — it is a **qualitatively different** and likely harder problem that requires genuinely new ideas, not just a subset of the current proof.
- **Surface diffusion is a stabilizing mechanism.** Physically, diffusion on the boundary smooths out spatial variations of the boundary state, which prevents the coupling term `∂_ν^A z` from creating boundary gradients that cannot be controlled by the bulk energy. Mathematically, it provides the `|∇_Γz_Γ|²` coercivity that makes the boundary integral estimates close.
- **The case `N = 1` is special precisely because `∇_Γ ≡ 0`.** In one spatial dimension, the boundary is a discrete set of points, the tangential gradient vanishes identically, and the surface diffusion term becomes meaningless. This explains why [6] could handle `N = 1` without surface diffusion — the dangerous term never appears — and why that result does **not** generalize to `N ≥ 2`, where `∇_Γz_Γ` is non-trivial and must be controlled.

This type of structural diagnosis — identifying which term in the PDE carries which specific burden in the estimate — is rare in the Carleman estimate literature, which typically treats lower-order terms as uniformly "absorbable" by taking parameters large enough. The paper shows that the surface diffusion is **not** just another lower-order term that can be absorbed; it plays a specific algebraic role in canceling a boundary term that no amount of parameter tuning can eliminate. This is a **fundamental conceptual insight** that reframes how one should think about dynamic boundary conditions: the surface diffusion is not an optional extra but a necessary ingredient for the Carleman methodology to function when `N ≥ 2`.

---

### Innovation 4: The Two-Axis Complementarity Between Proposal and Verifier Distribution Is a Unifying Diagnostic Framework

While couched in the specific language of anisotropic stochastic parabolic equations with dynamic boundary conditions, the paper's two-stage architecture (weighted identity → Carleman estimate → duality → absorption of weak terms) embodies a deeper conceptual principle that parallels developments in other areas of PDE control theory and, more broadly, in machine learning inference-time compute allocation.

The **first axis** — the weighted identity method — modifies the structure of the equation itself (through the choice of weights `θ, φ` and auxiliary functions `Ψ, Φ`) to expose coercive quadratic forms that dominate the solution. This is analogous to modifying the *proposal distribution*: by working with the transformed variable `h = θz` instead of `z`, the method generates `|∇h|²` and `h²` terms with coefficients (`λ³μ⁴φ³`, `λμ²φ`, etc.) that can be made arbitrarily large relative to the error terms by tuning `λ, μ`. The weighted identity *creates* the positive terms, rather than just rearranging existing ones.

The **second axis** — the duality/optimization method — does not modify the equation but instead *selects* among possible auxiliary functions (the controls `u, v₁, v₂` in the forward system) the one that optimally transfers the irregular source terms into regular ones. This is analogous to modifying the *verifier/selection mechanism*: among all possible forward solutions that reach zero at time `T`, the optimal control problem (4.4)–(4.6) selects the one that minimizes the weighted energy, and this minimal energy bounds the original backward state via the duality identity (4.22).

The complementarity mirrors the diagnostic framework discussed in the context of LLM test-time compute scaling, as analyzed in the referenced summary example:

- The **weighted identity** (proposal modification) is most powerful when the fundamental difficulty is *structural* — anisotropic matrices, dynamic boundary coupling — where clever algebraic manipulation can expose hidden coercivity. It fails when the irregularity is *distributional* (`H⁻¹` sources cannot be multiplied).

- The **duality method** (verifier/selection) is most powerful when the difficulty is *irregularity of the data* — `H⁻¹` source terms, weak convergence — where the right auxiliary problem can transfer the irregularity to a space where it becomes regular. It fails when the boundary condition is too complex because the auxiliary forward system inherits that complexity.

The two axes are **complementary in what they fail at**, which is precisely why their composition works. Neither axis alone can handle both anisotropic dynamic boundaries and `H⁻¹` sources; together, each covers the other's failure mode.

This framing is more than taxonomic. It implies a general strategy for other PDE systems where multiple structural and regularity obstacles co-occur:

1. Identify which obstacle can be handled by algebraic manipulation (proposal modification) and which requires an optimal auxiliary construction (verifier/selection).
2. Design the intermediate estimate assuming the second obstacle is absent.
3. Use the intermediate estimate to power the auxiliary construction that absorbs the second obstacle.
4. The composition is exact because the auxiliary construction does not reintroduce the first obstacle.

Whether this template applies to, say, stochastic Navier–Stokes (where the nonlinearity is both structural and irregular), or to stochastic degenerate parabolic equations (where the diffusion matrix is not uniformly elliptic), is an open and potentially fruitful direction. This paper does not claim generality — it deals specifically with linear anisotropic parabolic equations — but the clarity with which it separates the two axes of difficulty and sequences their resolutions constitutes a **conceptual innovation** that could influence how Carleman estimates are designed for other classes of PDE.

The significance is not a metric gain (the estimate itself is the result) but a **reframing of methodology**: the paper teaches not just *that* a particular Carleman estimate holds, but *how to think about* decomposing a multi-obstacle estimation problem into sequentially tractable sub-problems where each sub-method's assumptions are satisfied by construction. This is a higher-level contribution that sits above the specific inequalities proved.

## 5. Experimental Analysis

### Evaluation Methodology

**Dataset.** The paper uses the **MATH benchmark** (Hendrycks et al., 2021), a collection of high-school competition-level mathematics problems. The specific split from Lightman et al. (2022) is employed: 12,000 training questions and 500 test questions (Section 4). The choice is deliberate: test-time compute is expected to help most when the model already possesses the necessary knowledge and the challenge is drawing complex inferences — mathematical reasoning fits this profile because it requires multi-step logical deduction rather than novel factual recall.

**Base model.** All experiments use **PaLM 2-S\*** (Codey) (Anil et al., 2023). The authors argue this model is "representative of the capabilities of many contemporary LLMs" and sits in a useful regime: non-trivial performance on MATH (roughly 10–19% pass@1 depending on the prompt and sampling configuration) but far from saturation, leaving room for test-time compute to make a difference (Section 4). For the FLOPs-matched comparison, a second model with approximately **14× more parameters** is used as the pretraining-scaled baseline (Section 7).

**Metrics.** The primary metric throughout is **MATH test accuracy (%)** — the fraction of the 500 test questions for which the selected final answer matches the ground truth. Answers are graded using the grading function released by Lightman et al. (2022) (Appendix G). When analyzing difficulty-dependent behavior, the paper reports accuracy within each of the five difficulty quintiles separately (Section 3.2).

**Baselines.** The paper compares against several baselines (Section 4):
- **Majority voting**: select the most common final answer among N sampled solutions (no learned verifier).
- **ORM best-of-N weighted**: score N solutions with an outcome reward model and apply best-of-N weighted selection (Li et al., 2023).
- **PRM best-of-N weighted**: score N solutions with the process reward model and apply best-of-N weighted selection.
- **Parallel sampling** (for revisions): generate N independent solutions from the revision model and select the best via verifier or majority (Section 6).

**Generation budget / compute accounting.** The universal unit of test-time compute is one **generation** — one complete sampled answer from the base LLM. For beam search and best-of-N, the budget equals the number of beams or samples N. For lookahead search with k lookahead steps, the cost is N × (k+1) to account for the additional rollout computation (Section 5.3). Budgets are swept across powers of 2, typically from 2⁰ to 2⁹ (1 to 512 generations). In the FLOPs-matched comparison (Section 7), pretraining and inference FLOPs are computed using standard approximations:

```
X = 6ND_pretrain  (pretraining FLOPs)
Y = 2ND_inference (inference FLOPs)
```

where N is the number of model parameters, D_pretrain is pretraining tokens, and D_inference is total inference tokens generated. Scaling model parameters by a factor of M multiplies both X and Y by M. To match the total FLOPs of the M×-larger model using the smaller model with additional test-time compute, the smaller model's inference compute must be multiplied by:

$$M + 3 \cdot \frac{D_{\text{pretrain}}}{D_{\text{inference}}} \cdot (M - 1)$$

The critical quantity is the ratio R = D_inference / D_pretrain. When R ≪ 1 (few inference tokens relative to pretraining), the smaller model gets a large inference budget to work with because the pretraining savings dominate. When R ≫ 1 (many inference tokens), the budget is tighter because the larger model's per-token inference cost is a bigger fraction of total compute. Three values of R are tested: 0.16 (R ≪ 1), 0.79 (R ≈ 1), and 22 (R ≫ 1).

**Cross-validation / statistical protocol.** To avoid contaminating strategy selection with test-set performance, the authors use **two-fold cross-validation** within each difficulty bin on the 500-question test set. The best strategy is selected on one fold and evaluated on the other, with results averaged (Section 3.2). This applies to the compute-optimal policy selection — for each difficulty bin and budget level N, the strategy that performed best on the validation fold is deployed on the test fold. The predicted (non-oracle) difficulty bins are constructed by averaging the PRM's final-answer score across 2048 samples per question, then binning into five quintiles using the same procedure as the oracle bins (Section 3.2). The difficulty estimation cost (generating and scoring 2048 samples) is **not** included in the reported generation budgets, which the authors acknowledge as a limitation (Section 3.2): "our experiments do not account for this cost largely for simplicity."

---

### Main Quantitative Results

#### Search Against PRM Verifiers

**Aggregate comparison (Figure 3, left).** Across all 500 test questions with a maximum budget of 256 generations:

- At low budgets (2–8 generations), beam search with M = 4 significantly outperforms best-of-N weighted. For example, at 4 generations beam search (M = 4) achieves roughly 27% accuracy versus roughly 16% for best-of-N weighted — a substantial gap.
- At high budgets (64–256), beam search performance flattens and falls slightly below best-of-N weighted. Best-of-N weighted reaches approximately 38% at 512 generations; beam search (M = 4) plateaus around 34%.
- Lookahead search (both k = 1 and k = 3) generally underperforms at the same generation budget due to its higher per-step cost. The 3-step lookahead variants converge to similar performance as other methods at very high budgets but never surpass them.
- Majority voting trails all verifier-based methods substantially, reaching only about 29% at 512 generations.

**Difficulty-bin analysis (Figure 3, right).** The per-difficulty breakdown (beam search M = 4 vs. best-of-N weighted, shown at four budget levels: 4, 16, 64, 256 generations) reveals the core pattern:

- **Bin 1 (easiest):** Beam search accuracy *decreases* from roughly 78% to 77% as the budget goes from 4 to 256, while best-of-N weighted increases from 68% to 88%. This is the clearest evidence of PRM over-optimization — beam search finds solutions that exploit the verifier signal.
- **Bin 2:** Beam search improves modestly (roughly 14% → 32%) but best-of-N weighted improves faster (roughly 14% → 60%), maintaining a clear advantage at high budgets.
- **Bin 3:** Beam search consistently outperforms best-of-N weighted across all budgets, reaching roughly 34% vs. 23% at 256 generations.
- **Bin 4:** Beam search shows the strongest relative advantage, reaching roughly 17% vs. 10% for best-of-N at 256 generations.
- **Bin 5 (hardest):** Both methods hover near 1–3% regardless of budget. No method makes meaningful progress.

**Compute-optimal search (Figure 4).** By selecting the best search strategy per difficulty bin at each budget level:

- At 16 generations, compute-optimal (oracle bins) achieves approximately 27% accuracy, roughly matching PRM best-of-N weighted at 64 generations — a 4× compute reduction.
- At 256 generations, compute-optimal oracle reaches approximately 39.5%, surpassing PRM best-of-N weighted at the same budget (roughly 37%).
- Compute-optimal with predicted difficulty bins tracks the oracle version closely, particularly at lower budgets. The two curves "largely overlap" per the authors (Figure 4), with the predicted version reaching approximately 37% at 256 generations.
- Both compute-optimal variants consistently outperform ORM best-of-N weighted (which peaks around 34% at 512 generations) and majority voting (around 29%).

**PRM vs. ORM (Figure 14, Appendix F).** At 2048 samples, PRM best-of-N weighted achieves approximately 40% accuracy versus roughly 35% for ORM best-of-N weighted and roughly 30% for majority voting. The gap between PRM and ORM widens with the number of samples, confirming the PRM's superior scaling properties.

---

#### Revision Model Results

**Revision model pass@1 trajectory (Figure 6, left).** Starting from approximately 18.2% pass@1 at step 1, the revision model's per-step accuracy improves to roughly 24–25% by steps 15–20, and remains in the 23–25% range out to 64 steps. The model generalizes beyond its 4-step training horizon.

**Sequential vs. parallel (Figure 6, right).** At 64 generations:
- Sequential + best-of-N weighted: approximately 41.5%
- Parallel + best-of-N weighted: approximately 39%
- Sequential + majority: approximately 38%
- Parallel + majority: approximately 35%

Sequential outperforms parallel under both selection mechanisms, with the verifier-based gap (roughly 2.5 percentage points) being slightly narrower than the majority-based gap (roughly 3 points).

**Sequential-to-parallel ratio sweep (Figure 7, left).** For a fixed generation budget, varying the ratio reveals:
- At 256 generations, the optimal ratio is around 2¹ to 2³ (2:1 to 8:1 sequential-to-parallel), achieving approximately 43–44% accuracy.
- Fully parallel (leftmost point) yields approximately 40%.
- Fully sequential (rightmost point) yields approximately 42%.
- At lower budgets (8–32 generations), fully sequential is optimal — the curves are monotonically increasing with the sequential-to-parallel ratio.

**Difficulty-dependent ratio (Figure 7, right).** At a fixed budget of 128 generations:
- **Bin 1:** Performance is essentially flat across all ratios, around 90–92%. Easy questions are insensitive to the allocation strategy.
- **Bin 2:** Slight advantage for higher sequential ratios, approximately 63% at fully sequential vs. 58% at fully parallel.
- **Bin 3:** A clear optimal ratio emerges at moderate sequential-to-parallel values (around 2¹ to 2³), reaching approximately 42% vs. 35% at the extremes.
- **Bin 4:** Similar pattern, with the peak at a moderate ratio achieving roughly 18% vs. 14% at fully parallel.
- **Bin 5:** All ratios produce roughly 2–3% accuracy. No allocation strategy helps.

**Compute-optimal revisions (Figure 8).** Selecting the optimal sequential-to-parallel ratio per difficulty bin:
- At 64 generations, compute-optimal oracle achieves approximately 40%, matching parallel best-of-N weighted at 256 generations — a 4× improvement.
- At 256 generations, compute-optimal oracle reaches approximately 44%, compared to roughly 41% for best-of-N weighted and 37% for parallel-only.
- Compute-optimal predicted bins perform slightly below oracle bins at high budgets (approximately 41% at 256 generations) but still substantially outperform the parallel baseline.
- Notably, the parallel baseline appears to **plateau** around 36–37% at high budgets, while compute-optimal scaling continues to improve, suggesting that the gains from adaptive allocation compound at higher budgets.

---

#### FLOPs-Matched Comparison: Test-Time vs. Pretraining Compute

**Revisions (Figure 9, left; Figure 1, top-right bar chart).** Comparing PaLM 2-S* with compute-optimal revisions against the ~14× larger model:

| Difficulty | R ≪ 1 (0.16) | R ≈ 1 (0.79) | R ≫ 1 (22) |
|---|---|---|---|
| Easy (bin 1) | +11.8% | +3.5% | −11.9% |
| Medium (bin 2–3) | +27.8% | +16.7% | +5.4% |
| Hard (bins 4–5) | +21.6% | −(implied negative) | −37.2% |

At R ≪ 1, test-time compute outperforms the larger model across **all** difficulty levels. At R ≫ 1, it only remains preferable on easy questions, with hard questions showing a −37.2% relative disadvantage.

**PRM search (Figure 9, right; Figure 1, bottom-right bar chart).** The pattern is starker:

| Difficulty | R ≪ 1 (0.16) | R ≈ 1 (0.79) | R ≫ 1 (22) |
|---|---|---|---|
| Easy | +19.1% | +2.2% | +2.0% |
| Medium | 0.0% | −35.3% | −30.8% |
| Hard | −3.6% | −35.3% | −52.9% |

PRM search shows weaker benefits than revisions for the FLOPs-matched comparison, with substantial disadvantages on medium and hard questions even at moderate R values. On easy questions, test-time compute remains preferable across all R regimes, though the margin narrows significantly.

**Figure 9 detail.** The line plots show accuracy per difficulty bin as test-time compute scales. The 14× larger model's greedy performance (stars) is placed at three x-axis positions corresponding to the three R values. Where the compute-optimal scaling line is above the star, test-time compute wins. On bin 1 (topmost line), the scaling line is above all three stars for revisions. On bin 5 (bottommost line), the line is below all three stars and essentially flat near 0–5%, confirming that no amount of test-time compute helps on the hardest problems.

---

### Ablation Studies and Robustness Checks

**PRM aggregation strategy (Appendix E, Figure 13).** Comparing "min," "prod," and "last" step-wise aggregation: "last" achieves roughly 37% at 256 samples; "min" achieves roughly 35%; "prod" achieves roughly 27%; ORM achieves roughly 34%. The "last" aggregation's superiority is notable because it effectively reduces the PRM to ORM-like behavior at aggregation time, yet the PRM still outperforms a separately trained ORM. The authors interpret this as evidence that step-level PRM training provides beneficial representation learning.

**PRM vs. ORM (Appendix F, Figure 14).** The PRM consistently outperforms the ORM, with the gap widening at higher sample counts: at 2048 samples, PRM best-of-N weighted reaches approximately 40% vs. ORM's 35%.

**Revision model verifier choice (Appendix J, Figure 15a).** The base-LM PRM underperforms the revision-specific ORM when scoring revision model outputs, with sequential + base-LM PRM achieving roughly 40% at 64 generations vs. sequential + revision ORM at roughly 42%. This confirms distribution shift as a practical concern: a PRM trained on base model outputs does not transfer seamlessly to the revision model's different output distribution.

**Revision history in verifier context (Appendix J, Figure 15b).** Including previous revisions in the ORM's context provides a small improvement over the no-history ablation (approximately 1–2 percentage points at 64 generations), but both variants outperform the parallel baseline, confirming that the sequential sampling benefit is not solely attributable to the verifier seeing more context.

**Oracle vs. predicted difficulty bins (Figures 4, 8, and Appendix C, Figures 11–12).** Both oracle and predicted bins yield qualitatively similar trends across difficulty levels. Predicted bins show slightly lower performance at high budgets in the revision setting (roughly 41% vs. 44% at 256 generations in Figure 8) but essentially identical performance in the search setting (Figure 4). This is the critical robustness check: the compute-optimal strategy works without ground-truth labels, using only the PRM's own score distribution as a difficulty proxy.

**Majority voting for revisions (Appendix B, Figure 10).** The sequential-to-parallel ratio trends observed with verifier-based selection are replicated with majority voting: easy questions are insensitive to ratio, hard questions show an optimal intermediate ratio, and fully sequential marginally outperforms fully parallel in aggregate.

**ReST^EM revision model (Appendix K, Figure 16).** An attempt to further optimize the revision model using ReST^EM (Singh et al., 2024) backfires: additional sequential revisions **substantially hurt** performance with this model. At 256 generations, fully sequential performance drops to approximately 33.5% compared to roughly 38.5% at the optimal ratio. The authors hypothesize that the on-policy data collection in ReST^EM exacerbates spurious correlations in revision data, causing the model to fail to learn the revision task properly. This is a notable negative result that highlights the sensitivity of revision training to the data generation procedure.

---

### Critical Assessment

#### Does Compute-Optimal Scaling Achieve 4× Better Efficiency Over Best-of-N?

The paper reports that compute-optimal scaling matches best-of-N weighted accuracy "using up to 4× less test-time compute" — for search, 16 generations matching 64 (Figure 4), and for revisions, 64 generations matching 256 (Figure 8). This claim is **supported with boundaries clearly delineated**.

However, the 4× figure is extracted at specific comparison points and does not hold uniformly. At the highest budgets (256–512 generations), the gap narrows: compute-optimal predicted revisions reach approximately 41% at 256 generations versus best-of-N weighted at roughly 41% as well (Figure 8 — they are within 1 percentage point). The 4× claim is most reliable in the lower-to-moderate compute regime (16–64 generations for search, 64–128 for revisions). The paper is transparent about this by showing the full scaling curves rather than cherry-picking comparison points, but readers should not interpret 4× as a uniform multiplicative factor across all budgets — it is a budget-level-specific efficiency ratio.

More critically, the **difficulty estimation cost is unaccounted for**. Generating 2048 samples per question to estimate difficulty is roughly 8–16× more expensive than the largest test-time budgets studied (256–512 generations). The paper acknowledges this explicitly (Section 3.2): "our experiments do not account for this cost largely for simplicity." The reported 4× gains are therefore **post-difficulty-estimation** gains — they describe efficiency improvements *after* the difficulty is known, but do not amortize the cost of learning it. In a realistic deployment where difficulty must be estimated for each new prompt, the total cost would be difficulty estimation + strategy execution, and the former could dominate the latter. Until a cheaper difficulty estimator is developed (the paper suggests future work on direct difficulty prediction from question text), the 4× figure should be understood as an **upper bound on achievable efficiency** rather than a realized deployment gain. This is a significant gap between the experimental demonstration and practical applicability.

#### Does Test-Time Compute with a Smaller Model Outperform a 14× Larger Model?

The paper's claim that "a smaller model augmented with compute-optimal test-time strategies can outperform a ~14× larger pretrained model" is the headline result of Section 7 and Figure 1. This claim is **supported with sharply defined boundary conditions**, which is both a strength (scientific precision) and a limitation (narrower applicability than the headline might suggest).

Specifically:
- **Easy questions (bin 1):** Test-time compute outperforms the larger model across nearly all R values for both revisions and PRM search. For revisions at R ≪ 1, the advantage is +11.8% relative; even at R ≫ 1 for PRM search, it remains +2.0%. This is the strongest evidence for the claim.
- **Medium questions (bins 2–3):** Test-time compute is competitive or better when R ≪ 1 or R ≈ 1 (+27.8% and +16.7% for revisions), but **pretraining becomes preferable** when R ≫ 1 (neutral or negative for PRM search, +5.4% for revisions which is near the margin).
- **Hard questions (bins 4–5):** Pretraining is almost always more effective. Test-time compute provides minimal gains (negative or near-zero relative advantage), and at R ≫ 1 with PRM search, the disadvantage is −52.9%. This is a decisive failure case: test-time compute cannot compensate for fundamental capability gaps on problems where the base model's pass@1 is near zero.

The claim thus holds **conditionally**: when (a) the problem is within the base model's capability range (difficulty bins 1–3, where pass@1 is non-trivially above zero), and (b) the inference-to-pretraining ratio R is low to moderate (R ≪ 1 or R ≈ 1). It does not hold for hard problems or high-R deployments, which the paper explicitly notes in its Section 7 takeaway.

A more concerning experimental design choice is the **weak baseline for the larger model**. The 14× larger model uses only greedy decoding with no test-time compute augmentation. A fairer comparison would give the larger model some test-time compute budget — say, best-of-8 or best-of-16 — rather than greedy decoding alone. Since test-time compute provides diminishing returns (as the paper's own scaling curves show), giving a modest budget to the larger model might close much of the gap, especially on easy-to-medium problems where even a few extra samples would boost accuracy. The paper's choice to compare against greedy-only decoding makes the comparison somewhat favorable to test-time compute. If the larger model were also allowed, for example, best-of-16 weighted, the claimed advantages might shrink substantially or even reverse on medium-hard problems.

Additionally, the **14× larger model is trained by scaling parameters only, not both parameters and data** — the paper follows the LLaMA paradigm (Touvron et al., 2023) rather than compute-optimal pretraining (Hoffmann et al., 2022). A Chinchilla-optimal model trained with 14× more total FLOPs (scaling both parameters and data equally) would likely outperform a parameter-only-scaled model, making the pretraining baseline potentially weaker than optimal. The paper acknowledges this in Section 7: "We choose this setting as it is representative of a canonical approach to scaling pretraining compute and leave the analysis of compute-optimal scaling of pretraining compute where the data and parameters are both scaled equally to future work." This is a legitimate scope limitation but should temper interpretations of the 14× figure — the comparison is against a specific, possibly suboptimal, pretraining baseline.

#### Is Difficulty the Key Determinant of Test-Time Compute Efficacy?

The paper's central thesis — that "the efficacy of test-time compute depends critically on prompt difficulty" — is the **most robustly supported claim** in the paper. The difficulty-bin analyses across both search (Figure 3, right) and revisions (Figure 7, right) show qualitatively different, sometimes opposite, effects of the same strategy at different difficulty levels:

- Beam search **helps** on medium problems (bins 3–4) but **hurts** on easy problems (bin 1) due to verifier over-optimization.
- Fully sequential revisions **dominate** on easy problems (bin 2) but a **balanced** sequential-parallel ratio is optimal on hard problems (bins 3–4).
- No strategy helps on the hardest problems (bin 5) regardless of budget.

These patterns replicate across both search methods (beam search, lookahead search, best-of-N) and revision strategies (sequential, parallel, hybrid), and across both selection mechanisms (verifier-based and majority voting). The non-monotonicity — where "more powerful" optimization degrades performance on easier problems — would be impossible to observe without the difficulty-disaggregated analysis. This finding alone justifies the paper's core methodological contribution.

However, there are limitations to the difficulty estimation that affect the strength of this claim:

- **The five-bin discretization is coarse.** Difficulty is a continuous quantity (the base model's pass@1 rate), and binning into only five quintiles means that within each bin of ~100 questions, there may be substantial heterogeneity. A question at the boundary between bin 3 and bin 4 could receive a suboptimal strategy because the discrete binning forces a single choice for all questions in that bin. The paper does not report within-bin variance or sensitivity to the number of bins, making it difficult to assess whether five bins is sufficient or whether a finer-grained approach would yield substantial additional gains.

- **Difficulty bins are static and computed once.** There is no mechanism for dynamically adjusting the strategy mid-computation — for instance, starting with a few parallel samples, assessing whether the problem appears easy or hard based on the verifier's scores on those initial samples, and then allocating the remaining budget accordingly. Such an adaptive scheme could subsume the difficulty estimation cost into the solution process itself, but is not explored. The paper's computed-optimal policy is pre-computed for each bin and budget level, then deployed statically.

#### Experimental Limitations That Weaken Several Claims

**Single benchmark, single model family.** All results are on the MATH benchmark with PaLM 2-S* as the base model. The authors state they "believe this model is representative of the capabilities of many contemporary LLMs" (Section 4), but this claim is unverified. Several aspects of the findings could be model-specific:

- The PRM's quality and over-optimization behavior depend on PaLM 2-S*'s output distribution and calibration. A model with different error patterns (e.g., more or fewer spurious correct-looking but wrong solutions) might exhibit different difficulty-dependent scaling curves.
- The revision model's ability to learn from incorrect in-context examples depends on the base model's in-context learning capabilities, which vary substantially across model families (e.g., GPT-4 vs. PaLM vs. LLaMA).
- The MATH benchmark consists exclusively of competition-level math problems requiring symbolic reasoning. It is unclear whether the difficulty-dependent patterns (beam search hurting easy problems, revisions helping easy problems) generalize to other reasoning domains (code generation, logical reasoning, scientific QA) or to tasks requiring factual knowledge rather than inference.

**Small test set for strategy selection at scale.** The test set of 500 questions, split into five difficulty quintiles of ~100 each, then further split by two-fold cross-validation, means the compute-optimal policy is **selected based on ~50 questions per fold per bin**. This is a small sample, and the selected strategies may not be robust — a few unusually hard or easy questions in one fold could shift the apparent optimal strategy. The paper does not report confidence intervals on the compute-optimal scaling curves (e.g., bootstrapped error bars), making it difficult to assess whether the observed gains are statistically reliable at this sample size. Given that the accuracy differences between strategies in the mid-budget regime are often 2–5 percentage points (Figures 4, 8), the variance from 50-question folds could be large enough that the selected "optimal" strategy is not reliably distinguishable from the second-best.

**No combination of PRM search with revisions.** The paper studies search and revisions as independent mechanisms but never combines them — using the revision model as the proposal distribution within beam search, or using the PRM to guide which revisions to pursue. Section 8 explicitly acknowledges this: "we did not experiment with PRM tree-search techniques in combination with revisions." Since the two mechanisms have complementary strengths (revisions help on easy problems, search helps on medium problems, neither helps on hard problems), their combination — which could involve, for example, using beam search with the revision model as the proposal, or using the PRM to score revision chains — could yield gains beyond either method alone. The current results therefore represent a **lower bound** on what a fully integrated system could achieve. The absence of combined experiments leaves open the question of whether the sum of the individual gains is additive, super-additive, or sub-additive.

**Revision model's correct-to-incorrect reversion.** The paper reports that approximately 38% of correct answers produced during a revision chain get "revised" back to incorrect answers in the subsequent step (Section 6.1), due to the model being trained only on incorrect-to-correct trajectories. The selected-best-in-chain mitigation (using majority voting or verifier) is an imperfect patch — it can recover the correct answer if it appeared earlier in the chain, but it cannot prevent the model from degrading an already-correct answer. A more principled solution, such as training the model with a "do nothing" option when the current answer is already correct, is not explored. The degradation rate means that long revision chains (64 steps, as tested in Figure 6) likely contain many wasted computations on already-correct answers that get corrupted, limiting the effective utilization of the generation budget.

**No accounting for wall-clock latency.** The paper measures compute in "generations" (number of complete solutions sampled), which is a reasonable proxy for total FLOPs but ignores the fundamentally different **latency profiles** of different strategies. Sequential revisions are inherently serial — each revision depends on the previous one — while parallel best-of-N can be executed simultaneously with sufficient hardware. A strategy that allocates 128 generations as 64 sequential × 2 parallel takes roughly 64× longer wall-clock time than one that runs 128 parallel samples simultaneously. For latency-sensitive applications (interactive assistants, real-time decision-making), the sequential-heavy strategies favored by the compute-optimal policy on easy problems may be impractical regardless of their accuracy advantages. The paper does not discuss this tradeoff, nor does it report any latency measurements.

**Absence of experiments that would have strengthened the paper:**

1. **Replication on a second model family** (e.g., LLaMA-2 or GPT-3.5-level models) would test the "representative model" claim and reveal whether the difficulty-dependent patterns are universal or PaLM-specific.
2. **Replication on a non-math reasoning benchmark** (e.g., BIG-Bench Hard, MMLU reasoning subsets, or code generation tasks like HumanEval) would test whether the findings generalize beyond mathematical reasoning.
3. **Ablation on the number of difficulty bins** — testing 3, 5, 7, and 10 bins — would reveal the sensitivity of the compute-optimal gains to the discretization granularity.
4. **Dynamic difficulty estimation** — starting with a small number of samples, estimating difficulty from the verifier's scores on those samples, then allocating the remaining budget — would provide a practical path toward amortizing difficulty estimation cost and would test whether the pre-computed static policy can be approximated online.
5. **Combined search + revisions** — using the revision model as the proposal within beam search, or using the PRM to score revision chains — would test whether the two mechanisms are complementary or redundant.
6. **Larger model with some test-time budget** — giving the 14× larger model best-of-8 or best-of-16 rather than greedy-only — would create a stronger FLOPs-matched baseline and test whether test-time compute advantages survive against a more realistic deployment of the larger model.

## 6. Limitations and Trade-offs

### Difficulty Estimation Cost Is Unaccounted for in the Headline Efficiency Gains

**The assumption or constraint.** The entire compute-optimal framework — both the search strategy selection (Section 5.3) and the revision strategy selection (Section 6) — depends on knowing each prompt's difficulty *before* allocating the test-time compute budget. The method for estimating difficulty, whether oracle (using ground-truth pass@1) or predicted (using the PRM's average final-answer score), requires generating **2048 complete solutions per question** and scoring them (Section 3.2). The paper states explicitly:

> "estimating difficulty in this way still incurs additional computation cost during inference... our experiments do not account for this cost largely for simplicity" (Section 3.2)

The reported 4× efficiency gains over best-of-N (16 generations matching 64 for search, Figure 4; 64 generations matching 256 for revisions, Figure 8) are computed *after* difficulty is known, without amortizing the cost of learning it.

**The consequence.** In any realistic deployment where difficulty must be estimated for each new prompt, the *total* cost is difficulty estimation (2048 generations + PRM scoring) plus strategy execution. The former dominates the latter: 2048 generations is 8–32× larger than the largest test-time budgets studied (256–512 generations). If difficulty estimation is treated as part of the per-prompt cost, the 4× gain vanishes entirely — the total compute used would be far *higher* than a uniform best-of-N allocation at any budget level, since best-of-N does not require pre-estimating difficulty. The paper's headline efficiency claim is therefore specific to a setting where difficulty is known a priori (e.g., from prior runs on the same prompts, or from a separate, cheaper difficulty oracle), which is not the general deployment scenario.

**What evidence exists in the paper.** The paper does not report any experiment that includes difficulty estimation cost in the total budget. Section 3.2 acknowledges the issue and frames it as an "exploration-exploitation tradeoff — compute spent assessing difficulty versus compute spent solving the problem," but no quantitative analysis of this tradeoff appears. The paper does not report how accuracy changes if, say, half the budget is spent on difficulty estimation and half on strategy execution, versus spending the entire budget on a uniform best-of-N. This is a missing ablation that would directly address the practical relevance of the method.

**Mitigation status.** The paper does not attempt to mitigate this. Section 8 suggests future work on "pretraining or finetuning models to directly predict difficulty of a question" from the question text alone, bypassing the need for 2048 samples. But no such model is developed or evaluated, and it is unclear whether difficulty can be predicted reliably from text features without running the model — difficulty is defined relative to the model's own capabilities (`pass@1`), not as an intrinsic property of the question, so a text-only predictor would need to implicitly model the base LLM's strengths and weaknesses. An alternative mitigation — adaptive difficulty estimation where a small number of initial samples (e.g., 4–8) provides a quick difficulty signal and the remaining budget is allocated accordingly — is not explored, though the paper briefly alludes to this possibility in Section 3.2. Without such mitigations, the reported efficiency gains are an upper bound that cannot be realized in a deployment where difficulty is unknown a priori.

---

### The 14× Larger Model Baseline Is Weak — It Uses Only Greedy Decoding with No Test-Time Compute

**The assumption or constraint.** In the FLOPs-matched comparison of Section 7, the paper compares PaLM 2-S\* with compute-optimal test-time strategies against a model with approximately **14× more parameters** that uses only **greedy decoding** — no best-of-N, no majority voting, no search, no revisions. The paper also scales only model parameters when increasing pretraining compute, holding training data fixed (the LLaMA paradigm, Touvron et al., 2023), rather than scaling both parameters and data equally as in compute-optimal pretraining (Hoffmann et al., 2022). The paper acknowledges the latter choice:

> "We choose this setting as it is representative of a canonical approach to scaling pretraining compute and leave the analysis of compute-optimal scaling of pretraining compute where the data and parameters are both scaled equally to future work" (Section 7)

It does *not* explicitly justify the choice of greedy-only decoding for the larger model.

**The consequence.** The 14× larger model baseline is **weaker than it needs to be on two axes**:

1. **No test-time compute augmentation.** The paper's own results show that even a modest test-time compute budget (best-of-8 or best-of-16) provides substantial accuracy gains, especially in the low-budget regime where the scaling curves are steepest (Figures 3 and 6). Giving the larger model, for example, best-of-16 weighted selection would create a much stronger baseline. The paper's headline finding — that test-time compute can "outperform a ~14× larger pretrained model" (Figure 1) — is being compared against a deployment strategy (greedy decoding) that the paper's own analysis shows is suboptimal. If the larger model were also allowed a modest test-time compute budget, the claimed advantages might shrink or reverse, especially on medium-difficulty problems where the current comparison shows +27.8% relative improvement for revisions at R ≪ 1 (Figure 1, top-right bar chart).

2. **Potentially suboptimal pretraining.** A Chinchilla-optimal model trained with 14× more total FLOPs (scaling both parameters and data) would likely outperform a parameter-only-scaled model, since the latter over-allocates to parameters and under-allocates to data relative to the compute-optimal frontier. This means the baseline model's capabilities — its `pass@1` rate — may be artificially depressed relative to what could be achieved with the same total pretraining FLOPs. The reported advantage of test-time compute over pretraining may therefore partially reflect inefficiency in the baseline's training recipe, not a fundamental superiority of inference-time compute over pretraining compute.

**What evidence exists in the paper.** The paper does not report any experiment where the larger model is given a non-zero test-time compute budget. It does not compare against a Chinchilla-optimal larger model (scaling both parameters and data). The only baseline is greedy decoding from a parameter-only-scaled model. The paper's candid acknowledgment of the second limitation (Section 7) is commendable, but the first limitation — the absence of any test-time augmentation for the larger model — is not explicitly discussed as a limitation. It is a design choice that biases the comparison in favor of test-time compute.

**Mitigation status.** For the pretraining scaling choice, the paper explicitly defers to future work. For the greedy-only decoding choice, no mitigation is proposed or discussed. A fairer comparison would give the larger model a test-time compute budget that is a fraction of its total FLOPs allocation — for instance, if the total FLOPs match at a certain budget, the larger model could be allocated best-of-N with N chosen so that its total inference FLOPs (including the larger per-token cost) equal the smaller model's inference FLOPs. This would create a FLOPs-matched comparison where *both* models use test-time compute, but the larger model uses less of it (since each generation costs more). The absence of this comparison means the paper's strongest claim about the pretraining-inference tradeoff is incompletely tested.

---

### Hard Problems Are Unsolved — Test-Time Compute Does Not Create New Capability

**The assumption or constraint.** The paper's framework assumes that the base model already possesses non-trivial capability on the problem class — specifically, that its `pass@1` rate is measurably above zero. This is implicit in the difficulty estimation procedure (Section 3.2): difficulty is defined as the base model's `pass@1` rate, which means the hardest bin (bin 5) consists of problems where the model produces correct solutions in fewer than roughly 1–2% of samples.

**The consequence.** Across all methods — search, revisions, and their compute-optimal combinations — the **hardest questions (difficulty bin 5) show near-zero improvement** regardless of compute budget:

- In Figure 3 (right), bin 5 accuracy hovers at 1–3% for all search methods and all budgets up to 256 generations.
- In Figure 7 (right), bin 5 shows roughly 2–3% accuracy irrespective of the sequential-to-parallel ratio at 128 generations.
- In the FLOPs-matched comparison (Figure 9), the bin 5 scaling line is essentially flat near 0–5% — below the 14× larger model's greedy performance at all R values.

This is not a failure of the method *per se* — it is a fundamental limitation of the approach: test-time compute can amplify existing capability but cannot create it. If the base model's `pass@1` is near zero on a problem class, there are essentially no correct solutions in the proposal distribution to find (search) or refine (revisions). No amount of beam search or iterative revision will help because there is nothing correct to search for or revise toward.

The paper is transparent about this, stating in the Section 7 takeaway:

> "For such problems, pretraining remains the only viable path."

**What evidence exists in the paper.** The flat bin 5 curves in Figures 3 (right), 7 (right), and 9 are the direct evidence. The paper does not attempt to push further on bin 5 — it does not, for example, test whether a different kind of test-time strategy (e.g., decomposition into subproblems, retrieval augmentation, tool use) might help on problems where the base model's pass@1 is zero. The finding is negative but conclusive: within the space of strategies considered (search against a verifier, iterative revision, and their difficulty-conditioned allocation), hard problems remain unsolved.

**Mitigation status.** The paper does not attempt to solve this limitation within the test-time compute framework, and it does not propose it as a future direction for test-time methods. Instead, it draws the boundary explicitly: test-time compute works when the model already "knows" how to solve the problem at some low rate; when it doesn't, pretraining is needed. This is a *boundary condition* on the method's applicability, not a flaw to be fixed. However, for practitioners, it means the method offers **no path forward** for genuinely novel or out-of-distribution reasoning that exceeds the base model's training distribution. If a deployment involves a substantial fraction of such problems (e.g., cutting-edge research questions, novel programming tasks, problems requiring knowledge the model was not trained on), the compute-optimal framework provides no benefit and pretraining remains the only recourse.

---

### The Experimental Validation Covers Only One Model Family and One Benchmark

**The assumption or constraint.** All experiments in the paper use **PaLM 2-S\* (Codey)** as the base model and the **MATH benchmark** (Hendrycks et al., 2021) as the sole evaluation dataset (Section 4). The authors state:

> "We believe this model is representative of the capabilities of many contemporary LLMs" (Section 4)

This is an assertion, not an empirical finding, since no other model family is tested. The MATH benchmark consists exclusively of high-school competition-level math problems requiring symbolic multi-step reasoning. It tests a specific capability (mathematical deduction) in a specific format (closed-form answers that can be graded with exact string matching).

**The consequence.** Several aspects of the paper's findings could be **model-specific or domain-specific** in ways that affect generalization:

1. **PRM quality and over-optimization behavior.** The process reward model is trained via Monte Carlo rollouts from PaLM 2-S\* itself (Section 5.1, Appendix D). The PRM's scoring calibration, its tendency to be exploited by aggressive search, and the specific difficulty thresholds where beam search transitions from helpful to harmful — all depend on the base model's output distribution. A model with different calibration properties (e.g., one that assigns more uniform probability across plausible but wrong solutions) or different error patterns (e.g., making arithmetic errors vs. logical errors) might exhibit different difficulty-dependent scaling curves. The finding that beam search *degrades* easy-problem performance at high budgets (Figure 3, right, bin 1) could be specific to how PaLM 2-S\*'s output interacts with the specific PRM training procedure.

2. **Revision model training depends on base model's in-context learning.** The revision model is fine-tuned on trajectories where incorrect answers (selected by edit distance to the correct answer) appear in context before the correct answer (Section 6.1). The model's ability to learn from these in-context examples — to identify what was wrong and produce a corrected version — depends on the base model's in-context learning capabilities, which vary substantially across model families (GPT-4 vs. PaLM vs. LLaMA). A model with weaker in-context learning might not acquire the revision skill at all, or might exhibit a higher correct-to-incorrect reversion rate than the 38% reported for PaLM 2-S\*.

3. **MATH benchmark specificity.** The tasks require symbolic multi-step reasoning with unambiguous ground-truth answers that can be verified via exact string matching (using Lightman et al.'s grading function, Appendix G). This enables both the PRM training pipeline (Monte Carlo rollouts with automatic correctness checking) and the difficulty estimation (computing `pass@1` from 2048 samples). Many important real-world applications — open-ended generation, dialogue, creative writing, complex multi-step planning — lack such clean correctness signals. Extending the compute-optimal framework to domains where correctness is ambiguous, multi-dimensional, or subjective would require fundamentally different verifier training and difficulty estimation approaches. The paper does not test whether the core difficulty-dependent patterns (beam search helps on medium problems but hurts on easy ones; revisions help on easy ones but need parallelism on hard ones) generalize to non-math reasoning domains like code generation, logical reasoning, or scientific QA, or to tasks requiring factual recall rather than inference.

**What evidence exists in the paper.** The paper provides no cross-model or cross-domain experiments. The "representative model" claim is unsubstantiated. The paper does not discuss whether PaLM 2-S\* has specific properties (e.g., its temperature scaling, its calibration, its particular strengths or weaknesses on MATH) that might make the findings atypical.

**Mitigation status.** The paper does not acknowledge this as a limitation, which is a notable omission given the strength of the claims. Section 8 proposes future work on "extension to other domains," but does not flag the current single-model, single-benchmark scope as a limitation that tempers the generality of the conclusions. For a practitioner considering deploying these methods with a different model family (GPT-4, Claude, LLaMA-3) on a different task (code generation, scientific reasoning, customer support), the paper provides no evidence that the compute-optimal strategies selected for PaLM 2-S\* on MATH would transfer. The difficulty thresholds, the optimal sequential-to-parallel ratios, and even the qualitative findings (does beam search hurt or help on easy problems?) could differ. Replication on at least one additional model family and one additional reasoning benchmark would substantially strengthen the paper's claims to generality.

---

### The Revision Model Has a 38% Correct-to-Incorrect Reversion Rate That Limits Chain Length Utility

**The assumption or constraint.** The revision model is trained exclusively on trajectories where all in-context answers are **incorrect**, followed by a **correct** target answer (Section 6.1). The training data construction pairs independently sampled correct and incorrect solutions post-hoc, using character-level edit distance to select an incorrect answer that is "close" to the correct one (ensuring the incorrect answer is structurally similar but contains a mistake). The model is *never* trained on trajectories where the current answer is already correct, because the multi-turn training sequences only end with a correct answer after 0–4 incorrect ones.

**The consequence.** At test time, when the revision model generates a chain of revisions, it may encounter correct answers in its own context (produced during earlier steps) and, having no training signal for what to do when the answer is already correct, will **incorrectly "revise" them into wrong answers**. The paper reports:

> "approximately **38% of correct answers get converted back to incorrect ones**" (Section 6.1)

This means that long revision chains — which the paper tests up to 64 steps (Figure 6, left) — contain substantial wasted computation: the model finds a correct answer, then corrupts it, then potentially finds it again, then corrupts it again. The paper's mitigation is to select the best answer from any point in the chain (via majority voting or verifier-based selection), rather than always taking the last revision. But this does not prevent the degradation — it merely recovers the best answer post-hoc. The effective utilization of the generation budget is reduced because many generations are spent on "fixing" answers that were already correct, degrading them in the process.

The pass@1 trajectory in Figure 6 (left) illustrates the consequence: starting from ~18.2% at step 1, accuracy rises to ~24–25% by steps 15–20, but then **plateaus in the 23–25% range out to 64 steps**. The model does not continue improving — the reversion effect creates an equilibrium where new correct revisions are roughly balanced by corruptions of existing correct answers. This imposes a hard ceiling on what sequential revision depth can achieve, regardless of how many more steps are added.

**What evidence exists in the paper.** The 38% reversion rate is reported in Section 6.1 without a dedicated figure or table (it appears in the prose). The plateau in Figure 6 (left) is the indirect evidence — the pass@1 curve flattens after ~20 steps despite more compute being spent. The paper does not report an ablation showing what accuracy would be achieved if the model had a "do not revise" option or if a separate correctness detector could halt the chain when the current answer is likely correct. The ReST^EM experiment (Appendix K, Figure 16) shows that an alternative training approach *worsens* the reversion problem — fully sequential performance drops to ~33.5% at 256 generations — suggesting the issue is sensitive to training methodology and not easily fixed by simply collecting more data.

**Mitigation status.** The paper's mitigation (within-chain selection via majority voting or verifier) is a post-hoc recovery mechanism, not a prevention mechanism. It can recover a correct answer that was generated earlier in the chain, but it cannot prevent the model from spending compute on corrupting it in later steps. A more principled solution — training the model with a "no revision needed" action, or incorporating explicit correctness feedback during training so the model learns to recognize when the current answer is already correct — is not explored. The paper does not propose this as future work either, leaving the reversion problem as an acknowledged but unresolved limitation of the revision approach. For practitioners, this means that simply adding more revision steps does not reliably improve accuracy beyond a saturation point (~25% for PaLM 2-S* on MATH), and that the sequential revision budget should be allocated with this ceiling in mind — an insight that the compute-optimal policy partially captures by selecting balanced sequential-to-parallel ratios rather than fully sequential chains on problems where the ceiling is binding.

---

### No Accounting for Latency or Wall-Clock Time When Optimizing Compute Allocation

**The assumption or constraint.** The paper measures test-time compute exclusively in **generations** — the number of complete solutions sampled from the model. This is a reasonable proxy for total floating-point operations (FLOPs) since each generation costs approximately the same number of FLOPs for a given model. However, it ignores the fundamentally different **wall-clock latency profiles** of different strategies:

- **Parallel strategies** (best-of-N, parallel sampling with the revision model) can be executed simultaneously if sufficient hardware (multiple accelerators or large batch sizes) is available. `N` parallel samples take roughly the wall-clock time of one generation, plus a small aggregation overhead.

- **Sequential strategies** (beam search, sequential revision chains) are inherently serial — each step depends on the output of the previous step and cannot be parallelized. A chain of `N` sequential revisions takes roughly `N` times the wall-clock time of one generation.

- **Hybrid strategies** (parallel chains of sequential revisions) have latency proportional to the chain length, not the total number of generations. Allocating 128 generations as 8 parallel chains × 16 sequential steps takes roughly 16× the wall-clock time of a fully parallel 128-sample best-of-N.

**The consequence.** The compute-optimal allocation policy — which selects the best strategy per difficulty bin and budget level — makes decisions based on total FLOPs cost, ignoring latency. For example:

- On easy problems (bin 1–2), the policy favors **fully sequential revisions** (Figure 7, right shows monotonic improvement with higher sequential-to-parallel ratios for easy bins). This minimizes total FLOPs but **maximizes latency** — the model must generate one answer, then revise it, then revise the revision, etc., serially.

- On medium problems (bin 3–4), the policy favors a balanced or moderate sequential-to-parallel ratio — e.g., 8:1 or 4:1 sequential-to-parallel at 128 generations (Figures 7–8) — which still involves substantial serial depth.

For latency-sensitive applications — interactive assistants, real-time decision-making, any deployment where the user is waiting for a response — the sequential-heavy strategies favored by the compute-optimal policy may be **impractical regardless of their FLOPs efficiency**. The user experience of waiting for 64 sequential revision steps (even if each step is fast) is fundamentally different from receiving a single response after one parallelized batch. The paper's compute-optimal framework cannot distinguish between a strategy that costs 64 serial generations and one that costs 64 parallel generations, even though the former has ~64× higher latency.

**What evidence exists in the paper.** The paper does not report any latency measurements, wall-clock times, or throughput analyses. It does not discuss the latency-throughput tradeoff or acknowledge it as a factor in strategy selection. The generation budget `N` is treated as the sole cost metric throughout. This is standard in the LLM scaling literature (which focuses on FLOPs rather than latency), but it is a significant practical gap because **real deployments are constrained by both total compute and response time**.

**Mitigation status.** The paper does not address this limitation. A latency-aware extension of the compute-optimal framework would need to incorporate a constraint on the maximum serial chain length (or, equivalently, a penalty on latency in the objective), which would shift the optimal strategy away from sequential-heavy allocations for latency-sensitive applications. The paper does not propose this as future work. For practitioners, this means the reported compute-optimal strategies are optimal only under a FLOPs-only cost model; under a joint FLOPs-latency cost model (which is more realistic for interactive deployments), the optimal strategies would likely favor more parallelism than the current policy selects, especially on easy problems where the current policy recommends nearly fully sequential chains.

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
