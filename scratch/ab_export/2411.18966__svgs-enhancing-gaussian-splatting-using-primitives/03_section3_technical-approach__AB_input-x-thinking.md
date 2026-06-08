# Section 3 — Technical Approach — A/B (the discriminator)

**2411.18966 — SVGS: Enhancing Gaussian Splatting Using Primitives with Spatially Varying Colors**



## PyMuPDF input · THINK-HIGH

## 3. Technical Approach

### 3.1 Reader Orientation

This paper introduces **SVGS (Spatially Varying Gaussian Splatting)**, a system that upgrades each Gaussian primitive in a splatting-based radiance field so its colour and opacity can change *across the surface of the primitive*, not just with viewing direction. The core problem is that standard Gaussian primitives (3DGS ellipsoids or 2DGS surfels) use a single colour and a single opacity per primitive, which forces the scene to be represented by an enormous number of tiny, redundant primitives when the surface carries complex texture or geometry. The solution is to equip each primitive with a **spatially varying function** that maps a 2D surface coordinate `$(u,v)$` to a colour and opacity, so that a single primitive can locally model texture and shape far more compactly.

### 3.2 Big-Picture Architecture (diagram in words)

The SVGS pipeline works as follows:

- **Input**: a set of multi-view images with known camera poses.
- **Primitive set**: the scene is represented by a collection of **2D Gaussian surfels** (flat ellipses in 3D, each with a local 2D coordinate frame). Each surfel carries, in addition to its usual geometry and view-dependent colour, a **spatially varying function** `$F_c(p)$` for colour and `$F_\alpha(p)$` for opacity.
- **Rendering a pixel**: for a given ray, the intersection point `$p = (u,v)$` on the surfel is computed in the surfel’s local coordinate space. The colour and opacity at that point are obtained by evaluating the spatially varying function at `$(u,v)$` and adding the result to the view-dependent spherical harmonics colour. These per-point values are then alpha-blended along the ray as in standard Gaussian splatting.
- **Optimisation**: the parameters of all surfels (including those of the spatially varying functions) are trained by rendering training views, comparing against ground truth with an L1 + D-SSIM loss, and backpropagating gradients through the rasteriser.
- **Three alternative functions** are explored for `$F_c$` and `$F_\alpha$`: (1) **bilinear interpolation** across four colour/opacity anchors, (2) **movable kernels** that weight and sum a set of learnable kernel centres, and (3) **a tiny per-surfel MLP**.

The resulting system can represent complex textures with significantly fewer primitives than vanilla 2DGS, while still delivering real-time rendering (≥30 FPS).

### 3.3 Roadmap for the Deep Dive

- **First**, we’ll unpack the **spatially varying primitive formulation** — exactly how colour and opacity are redefined and how the intersection point `$p$` is computed, because this is the foundation that distinguishes SVGS from all prior Gaussian splatting work.
- **Second**, we’ll examine the **bilinear interpolation** design, which partitions each surfel into four quadrants and interpolates; this is the simplest but also reveals a gradient-vanishing limitation.
- **Third**, we’ll cover the **movable kernels** design, the paper’s best-performing variant, which places learnable kernel centres on the surfel and blends their colours via distance-based weights.
- **Fourth**, we’ll describe the **tiny MLP** design, which uses a small neural network per surfel to map `$(u,v)$` to colour and opacity; this is the most expressive but also the hardest to optimise and most parameter-heavy.
- **Fifth**, we’ll summarise the **training and optimisation** setup, including the baseline 2DGS framework, the hyperparameters, and how the three functions are integrated into the same rasterisation and backpropagation pipeline.

This order moves from the overall framework through the three concrete function designs in increasing complexity, and ends with the practical training details that tie everything together.

### 3.4 Detailed, Sentence-Based Technical Breakdown

#### 3.4.1 Spatially Varying Gaussian Primitives and Baseline

The paper is fundamentally an **extension of the Gaussian splatting representation**: instead of representing a scene with many small, single-colour primitives, it gives each primitive a **spatially varying colour and opacity** so that a single primitive can model a textured patch of surface. This is achieved by replacing the constant-colour, constant-opacity primitive with one whose appearance depends on the precise spot where the ray from the camera hits the primitive.

**Starting point: 2DGS surfels.** The paper builds on **2D Gaussian Splatting (2DGS)** [3], which compresses the 3D ellipsoids of 3DGS [2] into **planar Gaussian surfels** — each surfel is an elliptical disk with a local 2D coordinate system. In 2DGS, every surfel has:
- A **view-dependent colour** represented by spherical harmonics `$SH(\mathbf{d})$`, where `$\mathbf{d}$` is the viewing ray direction from the current pixel.
- A single **scalar opacity** `$\alpha$` that is constant across the whole surfel.

These two quantities are independent of the **exact intersection point** on the surfel: two different rays from the same viewing direction that hit the same surfel at different surface positions will receive the identical colour and opacity. This forces the model to represent any surface texture as a dense patchwork of many tiny primitives, each covering a small, near-homogeneous region.

**SVGS redefinition.** To break this limitation, SVGS redefines colour and opacity as **spatially varying functions** of the intersection point `$p$` on the surfel:

$$
c(p, \mathbf{d}) = SH(\mathbf{d}) + F_c(p),
$$

$$
\alpha(p) = F_\alpha(p),
$$

where `$p = (u,v)$` is the intersection point expressed in the local 2D coordinate system of the Gaussian surfel (the surfel origin is `$(0,0)$` and the ellipse axes form the coordinate axes). `$F_c(p)$` and `$F_\alpha(p)$` are the spatially varying functions for colour and opacity, respectively.

**What these equations compute:** For a given ray, the final colour is the sum of the standard view-dependent spherical harmonics term (which captures global lighting effects) and a **spatially-dependent offset** `$F_c(p)$` that depends on where the ray hits the surfel. The opacity becomes a pure function of that in-surface location. All these values are then passed through the differentiable alpha-blending rasteriser just like in standard 2DGS.

**Why this form:** The additive split keeps the existing view-dependent modelling (SH) while making the local texture a separate, learnable, spatially-varying correction. The paper explicitly does not constrain `$F_c$` and `$F_\alpha$` to non-negative ranges during optimisation; instead, they allow the functions to output any real values, because the final colour and opacity are normalised into `$[0,1]$` via a sigmoid activation during the actual splatting render. This avoids the need for expensive, potentially unstable value-clamping inside the gradient computation, and lets the optimiser freely explore the parameter space. (Models like 3D-HGS [38] and NegGS [48] that explicitly use negative Gaussians become special cases of this unconstrained formulation.)

**Computation of the intersection point.** The choice of **2D surfels** instead of 3D ellipsoids is crucial: for a flat disk, the intersection between a ray and the surface is a single point whose local `$(u,v)$` coordinates can be computed in closed form. For 3D ellipsoids, the intersection point would be on a curved surface and the coordinate mapping would be more complex. The paper therefore adopts the 2DGS local coordinate system by default, but notes that the same idea can be extended to 3D Gaussians with extra care.

**Baseline 2DGS colour and opacity.** By contrast, 2DGS would simply use:

$$c(\mathbf{d}) = SH(\mathbf{d}), \quad \alpha = \text{constant},$$

i.e., no spatial dependence inside the primitive. This is the “single view-dependent colour + one opacity” model that SVGS replaces.

#### 3.4.2 Bilinear Interpolation Function

The first concrete spatial variation design is **bilinear interpolation**. The idea is to divide each elliptical Gaussian surfel into four quadrants, assign a learnable colour and opacity to each quadrant, and then use standard bilinear interpolation to compute the value at any continuous `$(u,v)$` position.

**Formulation.** The colour offset `$F_c(p)$` and opacity `$F_\alpha(p)$` are defined as:

$$
F_c(p) = (1 - u')(1 - v') \, c_0 + (1 - u') v' \, c_1 + u' (1 - v') \, c_2 + u' v' \, c_3,
\label{eq:bi_c}
$$

$$
F_\alpha(p) = (1 - u')(1 - v') \, \alpha_0 + (1 - u') v' \, \alpha_1 + u' (1 - v') \, \alpha_2 + u' v' \, \alpha_3,
\label{eq:bi_a}
$$

where `$c_i$` and `$\alpha_i$` for `$i = 0, 1, 2, 3$` are the four new learnable RGB-colour vectors and opacity scalars associated with the four quadrants, respectively. The coordinates `$u', v'$` are a **rescaled** version of the raw object-space coordinates `$(u,v)$`:

$$
u' = \frac{1}{1 + e^{-\lambda_s u}}, \qquad
v' = \frac{1}{1 + e^{-\lambda_s v}},
\label{eq:sigmoid_rescale}
$$

with `$\lambda_s = 5.0$` by default.

**What it computes:** For a given intersection point `$(u,v)$`, the function first squeezes the raw coordinates through a sigmoid with steepness `$\lambda_s$` to map them into the range `$(0,1)$`. Then it performs a standard four-corner bilinear blend: the four anchor colours `$c_0,\dots,c_3$` (and opacities) are weighted by products of the complementary fractional coordinates. This produces a smooth, piecewise-bilinear colour/opacity field across the surfel.

**Why this form:** The sigmoid rescale is necessary to keep the coordinates within a well-behaved range before interpolation, because the raw `$(u,v)$` coordinates can theoretically span the whole ellipse and might otherwise cause extreme or irregular values. The bilinear form itself is a classic, efficient way to define a continuous function over a quadrilateral from corner values.

**Limitation — gradient vanishing.** The paper observes a practical problem with this design: near the centre of each quadrant, the sigmoid derivative becomes extremely small, which causes the gradient signal for those pixels to effectively vanish. This means bilinear interpolation can struggle to fit sharp colour transitions when the pattern does not naturally align with the quadrant boundaries. Fig. 12 (top row) and the discussion in Sec. IV-E explicitly highlight this: bilinear interpolation can capture abrupt colour boundaries but fails in other regions due to gradient decay.

#### 3.4.3 Movable Kernels Function

The second, and best-performing, design is **movable kernels**. This formulation generalises the idea of fixed quadrants by allowing the “anchors” to **move** freely on the surfel surface and by using a **soft, distance-based weighting** rather than a hard bilinear interpolation.

**Formulation.** A set of `$k$` learnable **kernel centres** `$K_i = (K^x_i, K^y_i)$` are placed on each surfel (by default `$k = 4$`). The colour offset and opacity are then computed as a weighted sum of per-kernel colour/opacity values:

$$
F_c(p) = \sum_{i=0}^{k-1} F_{K_i}(p) \, c_i,
\qquad
F_\alpha(p) = \sum_{i=0}^{k-1} F_{K_i}(p) \, \alpha_i,
\label{eq:mk_sum}
$$

where the weight for kernel `$i$` is given by an **exponential decay** with the distance from the intersection point to the kernel centre:

$$
F_{K_i}(p) = e^{-\lambda_e \, \|p - K_i\|^2}.
\label{eq:exp_kernel}
$$

Here `$\lambda_e = 0.1$` controls the sharpness of the kernel, and the default number of kernels is `$k = 4$`.

**What it computes:** For an intersection point `$p = (u,v)$`, the function computes the squared Euclidean distance to each learnable kernel centre `$K_i$`, applies an exponential decay, and uses the resulting weights to blend the per-kernel colours `$c_i$` and opacities `$\alpha_i$`. The result is a smooth, spatially varying, mixture-of-Gaussians on the surfel.

**Why this form:** The exponential (effectively an unnormalised Gaussian) provides a naturally smooth, localised receptive field around each kernel centre. Unlike the bilinear interpolation, the weights are not restricted to a strict four-corner partition and the kernel centres can move during training, giving the model much more flexibility to place its “sub-primitives” where the texture actually demands them. The fixed number `$k=4$` keeps the per-primitive parameter count low (roughly 1.4× that of 2DGS, see Fig. 5), while still offering strong expressive power.

**Kernel movement and degeneracy.** The kernel centres are regular learnable parameters and are updated by gradient descent during optimisation. In rare cases a kernel centre may drift outside the ellipse boundary; when this happens, the primitive locally degenerates to the behaviour of vanilla 2DGS (the contribution of that kernel becomes negligible). The paper does not enforce any hard boundary constraint because such events are extremely infrequent — Table X reports that across all tested datasets, over 99.5% of kernels remain within the Gaussian.

**Alternative kernel functions.** The paper also experiments with a **sigmoid kernel** instead of the exponential:

$$
F_{S_i}(p) = 1 - \tanh(\|p - K_i\|^2),
\label{eq:sigmoid_kernel}
$$

but finds that the exponential form (Eq. 7) yields slightly better results (Table VII). The movable kernel with exponential decay is therefore the default.

#### 3.4.4 Tiny Neural Network (MLP) Function

The third design replaces the hand-crafted kernel or interpolation functions with a **learned, per-surfel tiny multilayer perceptron (MLP)**. This is the most direct — and most parameter-intensive — way to obtain a general spatially varying function.

**Formulation.** A separate small neural network is instantiated for each Gaussian surfel:

$$
(F_c(p),\, F_\alpha(p)) = \text{MLP}(p),
\label{eq:mlp}
$$

where the MLP takes the 2D local coordinate `$p = (u,v)$` as input and outputs both the colour offset and the opacity at that location. The paper uses a **three-layer** network with a sigmoid activation function, as illustrated in Fig. 6. The architecture is deliberately kept shallow: an ablation in Table VIII shows that increasing the number of layers beyond one does not improve performance and can even slightly degrade it (on the Lego scene, from 35.66 PSNR for 1 layer down to 35.38 PSNR for 4 layers).

**What it computes:** The MLP freely maps every `$(u,v)$` to an output, with no imposition of a particular functional form — it is a universal function approximator on the 2D domain. This gives the surfel the maximal flexibility to represent complex, irregular texture patterns.

**Why this form — and its trade-offs:** A per-primitive neural network can, in principle, fit any spatial variation, including patterns that do not conform to a quad-partition or a kernel-mixture. However, this flexibility comes at a cost: the parameter count per primitive is **1.88×** that of 2DGS (Fig. 5), and the optimisation of many independent tiny MLPs is less stable than that of the purely analytic kernel functions. Consequently, the MLP variant performs best when the total number of primitives is strictly limited (strong representational power per primitive), but under the “unlimited” primitive regime it tends to underperform the movable kernel (Table I). The paper therefore treats the MLP as a useful but not dominant alternative.

#### 3.4.5 Training and Optimisation Pipeline

The three spatially varying functions are all trained within the **same underlying 2DGS framework**, with the same overall loss and the same hyperparameter schedule.

**Loss and metrics.** The training objective follows the standard Gaussian Splatting recipe: an L1 loss combined with a D-SSIM term on the rendered images. For the normal-consistency loss (used in 2DGS to improve geometry), the paper **discards it by default** because the primary goal is novel-view synthesis quality, and Table VI shows that even without that loss SVGS significantly outperforms 2DGS.

**Training hyperparameters.** All experiments use the hyperparameters from the 2DGS and 3DGS codebase (Section IV-A):

- **Training iterations**: 30 K
- **Gradient splitting threshold**: 0.0002
- **Opacity reset**: set opacity to 0.01 every 3000 iterations
- **Splitting, cloning, and removal** of Gaussians: stopped after 15 K iterations
- **Optimiser**: not explicitly stated in this paper, but the 2DGS/3DGS code uses the Adam optimiser with the standard settings; the paper does note that all experiments are run on an NVIDIA A100 80 GB GPU.

**Back-propagation through the spatially varying functions.** The forward and backward passes for each of the three functions are implemented in custom CUDA kernels. The authors modified the existing 2DGS CUDA code to handle the additional per-pixel evaluations of `$F_c$` and `$F_\alpha$` and their gradients with respect to both the kernel parameters and the intersection-point coordinates. This is the main source of the increased training and inference time (Table XII: 1083 s vs. 635 s for 2DGS on Blender).

**No explicit value constraints.** As noted in Sec. 3.4.1, there is no clipping or projection of `$F_c$` or `$F_\alpha$` values during optimisation; the functions are free to produce negative or arbitrarily large outputs, and the final rasterisation pipeline applies a sigmoid to clamp colours and opacity into `$[0,1]$`. This keeps the optimisation simple and unconstrained, and the opacities naturally tend to correct values; Gaussians with opacity below a threshold are automatically pruned by the standard densification control.

#### 3.4.6 Integration with the 2DGS Rendering and the Overall Parameter Budget

The paper’s 2DGS baseline already uses a standard **alpha-blending** rendering equation along each ray, accumulating colour and opacity from front to back. The only change in SVGS is that for each surfel hit, instead of using a constant opacity and a purely view-dependent colour, the system computes `$c(p,\mathbf{d})$` and `$\alpha(p)$` as in Eqs. (1)–(2), and then blends them identically. This means the rendering and optimisation are a **drop-in replacement** of the primitive attributes.

**Parameter count per primitive.** Fig. 5 provides the exact comparison: relative to a 2DGS primitive (which stores roughly 58 parameters for geometry, SH colours, and opacity), the three SVGS variants add:

- **Bilinear interpolation**: 4 extra colours + 4 extra opacities → 1.28× the parameters of 2DGS.
- **Movable kernels** (k=4): 4 kernel centres + 4 colours + 4 opacities → 1.40×.
- **Tiny MLP** (3-layer, hidden size ≈8): → 1.88×.

**Fairness of comparisons.** The paper carefully controls for parameter count when comparing against 2DGS. In Table II, they create **2DGS\*** by proportionally increasing the number of Gaussians so that the total parameter count matches that of the SVGS-MK model; the 2DGS\* still underperforms SVGS, showing that the gain is not simply from more parameters but from the **spatially varying structure**. Similarly, in Table IX they force 2DGS to have twice the number of Gaussians (and therefore ~1.43× the parameters) — it still loses to SVGS with fewer Gaussians, confirming the superior expressiveness per primitive.

**Training-time and rendering-time overhead.** Table XII reports that while SVGS-MK training takes about 1.7× the time of 2DGS, and rendering runs at 133 FPS vs. 211 FPS, it still comfortably exceeds real-time (30 FPS). Furthermore, because SVGS can represent the same scene with fewer primitives, the total number of primitives and total training iterations can be reduced; an ablation in the right half of Table XII shows that when training time is restricted to match 2DGS (by reducing iterations), SVGS still produces higher quality than 2DGS.

Thus the overall technical approach is: take the 2DGS pipeline, replace the per-primitive constant colour and opacity with one of three analytically simple or MLP-based spatially varying functions, and let the same differentiable rasteriser optimise both the original parameters and the new spatial variation parameters.


## GLM-OCR input · THINK-NONE

## 3. Technical Approach

### 3.1 Reader orientation (approachable technical breakdown)

This paper presents a method for enhancing the representational power of individual Gaussian primitives in Gaussian Splatting-based scene reconstruction. The system replaces the traditional per-primitive uniform color and opacity with **spatially varying functions** — meaning that different rays intersecting the same Gaussian primitive at different surface locations can receive different colors and opacities, rather than all rays receiving the same values. The core problem it solves is that standard Gaussian Splatting primitives are "non-compact" — they waste many small primitives to approximate what is fundamentally a single surface with varying texture, and SVGS addresses this by making each primitive capable of representing spatially varying appearance on its own, thereby achieving better rendering quality with fewer primitives.

### 3.2 Big-picture architecture (diagram in words)

The SVGS system consists of five major components that interact in a standard Gaussian Splatting pipeline:

1. **2D Gaussian surfel primitives** — the scene is represented as a collection of elliptical 2D surfels (flat disks in 3D), each defined by a center position, rotation, scale, and a local 2D coordinate system on the ellipse surface.

2. **Spatially varying color/opacity functions** — attached to each primitive, these functions compute color and opacity as a function of the local intersection point `$(u,v)$` on the surfel, replacing the traditional uniform per-primitive color and opacity.

3. **Ray-surfel intersection computation** — for each pixel, a ray is cast from the camera. The intersection point with each surfel is computed in the surfel's local 2D coordinates, producing a `$(u,v)$` coordinate pair.

4. **Alpha-blending rendering** — the spatially varying colors and opacities from all primitives are alpha-composited along each ray, producing the final pixel color (identical to standard Gaussian Splatting's rendering equation).

5. **Gradient-based optimization** — the parameters of all primitives (including the spatially varying function parameters) are optimized via back-propagation through the rendering pipeline against multi-view image supervision.

Information flows as follows: input multi-view images → initialize Gaussian surfel primitives → for each training view ray, compute ray-surfel intersections → evaluate spatially varying color/opacity at each intersection → alpha-blend into pixel colors → compute loss against ground truth → back-propagate gradients to update all parameters, including the spatially varying function parameters.

### 3.3 Roadmap for the deep dive

- **First**, the core formulation of spatially varying colors and opacity (Equations 1-2), which defines what "spatially varying" means and how it generalizes standard Gaussian Splatting.
- **Second**, the three concrete implementations of the spatially varying function — bilinear interpolation, movable kernels, and tiny MLPs — since these are the three distinct design options that the paper evaluates.
- **Third**, the bilinear interpolation method (Section III-B), as it is the simplest and builds intuition for local spatial variation.
- **Fourth**, the movable kernel method (Section III-C), which is the best-performing design and introduces learnable kernel positions.
- **Fifth**, the tiny MLP variant (Section III-D), which uses per-primitive neural networks for maximum expressiveness.
- **Sixth**, the rendering and optimization integration, explaining how these functions plug into the standard Gaussian Splatting pipeline and how gradients flow.

### 3.4 Detailed, sentence-based technical breakdown

This is primarily a **method paper** whose core idea is that endowing individual Gaussian primitives with spatially varying color and opacity functions — rather than treating them as having a single uniform color and opacity — dramatically increases each primitive's ability to represent complex textures and geometry, enabling more compact scene representations and higher rendering quality.

---

#### The Core Formulation: Spatially Varying Colors and Opacity

Standard Gaussian Splatting methods (3DGS, 2DGS) represent each primitive with a color that depends only on the viewing direction and a single opacity value. For a given primitive, all rays that intersect it at different locations receive exactly the same color (for the same viewing direction) and exactly the same opacity. This is the fundamental limitation the paper addresses: to represent a surface with varying color or varying opacity across its extent, existing methods must create many tiny primitives, each covering a small patch where the color is approximately uniform.

SVGS replaces this uniform-per-primitive model with a **location-dependent model**. The paper defines the color function `$c(\mathbf{p}, \mathbf{d})$` and opacity function `$\alpha(\mathbf{p})$` as:

$$c(\mathbf{p}, \mathbf{d}) = SH(\mathbf{d}) + F_c(\mathbf{p})$$

$$\alpha(\mathbf{p}) = F_\alpha(\mathbf{p})$$

where `$\mathbf{p}$` is the intersection point between the ray and the Gaussian primitive, expressed in the primitive's local 2D coordinate system; `$\mathbf{d}$` is the viewing ray direction; `$SH(\mathbf{d})$` is the standard view-dependent spherical harmonic color component (inherited from 3DGS/2DGS); `$F_c(\mathbf{p})$` is the spatially varying color contribution (a function of local position `$\mathbf{p}$` only); and `$F_\alpha(\mathbf{p})$` is the spatially varying opacity contribution (a function of local position `$\mathbf{p}$` only).

**What these equations compute:** The final color at a ray-primitive intersection is the sum of the standard view-dependent spherical harmonic color and a position-dependent offset. The final opacity is purely a function of the local intersection position. This means that for a fixed viewing direction, two rays that hit the same primitive at different `$(u,v)$` locations will receive different colors because `$F_c$` differs. The opacity can also vary across the primitive's surface.

**Why this form:** The additive decomposition `$SH(\mathbf{d}) + F_c(\mathbf{p})$` is a design choice that preserves the existing view-dependent appearance modeling (which handles specular effects like highlights that change with viewing angle) while adding spatial variability. The paper explicitly states that no constraints are imposed on `$F_c$` and `$F_\alpha$` — they can take negative values — because the final colors and opacities are normalized to valid ranges via sigmoid activation in the rendering pipeline. Enforcing explicit value constraints during optimization would be computationally expensive and non-trivial; instead, the system relies on the downstream sigmoid to produce physically meaningful values. This makes methods like 3D-HGS and NegGS (which use half-Gaussians or negative Gaussians) special cases of the SVGS formulation.

The local intersection point `$\mathbf{p} = (u, v)$` is defined as the 2D coordinates on the Gaussian surfel's local coordinate plane, where the surfel's center is `$(0,0)$` and the axes of the elliptical surfel form the coordinate axes. The paper adopts 2D Gaussian surfels (from 2DGS) as primitives by default because computing the ray-surfel intersection in local coordinates is straightforward — the ray intersects the surfel plane at a single point. Extending this to 3D ellipsoids would require computing the intersection with the ellipsoid surface, which is more involved but conceptually similar.

---

#### Spatially Varying Function 1: Bilinear Interpolation

**What it is.** The bilinear interpolation method divides each elliptical Gaussian surfel into four quadrants and assigns a separate learnable color and opacity to each quadrant. For any intersection point `$\mathbf{p} = (u, v)$` on the surfel, the color and opacity are computed by bilinearly interpolating between the four quadrant values based on the point's position.

**How it works.** Four learnable color parameters `$c_0, c_1, c_2, c_3$` and four learnable opacity parameters `$\alpha_0, \alpha_1, \alpha_2, \alpha_3$` are associated with each Gaussian primitive, corresponding to the four corners/quadrants of the bilinear interpolation grid. The spatially varying functions are:

$$F_c(\mathbf{p}) = (1 - u')(1 - v')c_0 + (1 - u')v'c_1 + u'(1 - v')c_2 + u'v'c_3$$

$$F_\alpha(\mathbf{p}) = (1 - u')(1 - v')\alpha_0 + (1 - u')v'\alpha_1 + u'(1 - v')\alpha_2 + u'v'\alpha_3$$

where `$u', v'$` are rescaled coordinates in `$[0,1]$` obtained by passing the raw local coordinates `$(u,v)$` through a sigmoid function:

$$u' = \frac{1}{1 + e^{-\lambda_s u}}, \quad v' = \frac{1}{1 + e^{-\lambda_s v}}$$

with `$\lambda_s$` being a parameter controlling the sigmoid's transition rate, set to 5.0 by default.

**What this computes:** Given a local intersection point `$(u,v)$` on the Gaussian surfel, the sigmoid rescaling maps it to `$(u',v')$` in the range `$[0,1]$`. Then bilinear interpolation weights the four corner values by how close the point is to each corner, producing a smoothly varying color and opacity across the primitive. The four corners correspond conceptually to four quadrants of the elliptical surfel, each with its own color/opacity.

**Why this form:** Bilinear interpolation is the simplest form of spatial variation — it allows a primitive to have up to four distinct colors at its extremes while smoothly blending between them. The sigmoid rescaling (rather than a simple linear mapping) is necessary because the raw local coordinates `$(u,v)$` can be arbitrarily large (the Gaussian extends infinitely in its local coordinate system). The sigmoid compresses the coordinates to `$[0,1]$` so that the bilinear interpolation weights `$(1-u')(1-v')$` etc. are well-defined and sum to 1.

However, the paper identifies a **gradient vanishing problem** with this approach (Fig. 4(a) and the discussion in Section IV-E). Because the sigmoid function saturates (its derivative approaches zero for large inputs), when a pixel falls near the center of a quadrant (i.e., far from the edges), the gradient of the bilinear interpolation weights with respect to `$(u,v)$` becomes nearly zero. This makes it difficult for the optimization to move the colors around and adjust to the scene. The bilinear interpolation therefore works best when the spatial variation pattern naturally aligns with a four-quadrant distribution.

---

#### Spatially Varying Function 2: Movable Kernels

**What it is.** The movable kernel method places `$k$` learnable kernel centers `$K_i = (K_i^x, K_i^y)$` on each Gaussian surfel. The color and opacity at an intersection point `$\mathbf{p} = (u,v)$` are computed as a weighted sum over these kernels, where each kernel's weight decays exponentially with distance from the point to the kernel center. The kernels are "movable" because their positions `$(K_i^x, K_i^y)$` are learnable parameters that can shift during optimization, unlike the fixed quadrant centers in bilinear interpolation.

**How it works.** For `$k$` kernels (default `$k=4$`), each kernel `$i$` has a center position `$(K_i^x, K_i^y)$`, a learnable color contribution `$c_i$`, and a learnable opacity contribution `$\alpha_i$`. The spatially varying functions are:

$$F_c(\mathbf{p}) = \sum_{i=0}^{k-1} F_{K_i}(\mathbf{p}) c_i$$

$$F_\alpha(\mathbf{p}) = \sum_{i=0}^{k-1} F_{K_i}(\mathbf{p}) \alpha_i$$

where the kernel weight function `$F_{K_i}(\mathbf{p})$` is an exponential decay based on the Euclidean distance from the point `$\mathbf{p}$` to the kernel center `$K_i$`:

$$F_{K_i}(\mathbf{p}) = e^{-\lambda_e \| \mathbf{p} - K_i \|^2}$$

with `$\lambda_e$` controlling the decay rate (set to 0.1 by default) and `$\|\mathbf{p} - K_i\|$` being the L2 distance in the local 2D coordinate space.

**What this computes:** For a given intersection point `$\mathbf{p}$` on the surfel, the system computes the squared distance from `$\mathbf{p}$` to each kernel center `$K_i$`. Each kernel's weight is `$e^{-\lambda_e \cdot \text{distance}^2}$` — close to 1 when `$\mathbf{p}$` is near the kernel center and approaching 0 when far away. The final color offset `$F_c$` is the weighted sum of the kernel colors `$c_i$`, and the opacity `$F_\alpha$` is the weighted sum of the kernel opacities `$\alpha_i$`. The kernels effectively act as "local color patches" that can move to where they are needed on the surfel surface.

**Why this form:** The exponential decay function is a natural choice for a local kernel — it creates a smooth, differentiable bump around each kernel center. Compare this to bilinear interpolation: the four quadrants in bilinear interpolation are effectively "fixed kernels" at the four corners (or, more precisely, four color anchors at `$(0,0), (0,1), (1,0), (1,1)$`). By making the kernel centers learnable, the movable kernel design gains flexibility — the kernels can move to where the texture variation actually occurs rather than being locked to the corners. This is why the paper finds movable kernels outperform bilinear interpolation (Table I).

The choice of `$k=4$` kernels is a design decision balancing expressiveness and parameter count. Ablation in Table VII shows that increasing to `$k=8$` does not significantly improve results, suggesting 4 movable kernels already provide sufficient spatial degrees of freedom for typical texture patterns. The decay parameter `$\lambda_e = 0.1$` controls how "sharp" or "blurry" each kernel is — smaller values make kernels broader and more overlapping; larger values make them sharper and more localized.

An alternative kernel function (sigmoid-based) is also evaluated:

$$F_{S_i}(\mathbf{p}) = 1 - \tanh(\|\mathbf{p} - K_i\|^2)$$

This sigmoid form provides a different decay profile but the exponential version achieves slightly better results (Table VII).

**Kernel movement behavior.** The kernel centers move via gradient descent during optimization. The paper reports (Table X) that across all tested datasets, kernels almost never move outside the Gaussian surfel boundary — the probability of a kernel staying inside is essentially 1. If a kernel were to move outside, SVGS would effectively degenerate to 2DGS at that primitive (since all intersection points would be far from the kernel, giving near-zero weights), but the paper does not impose any special constraint to prevent this because it is so rare.

---

#### Spatially Varying Function 3: Tiny MLPs

**What it is.** Instead of defining a parametric function like interpolation or kernels, this variant allocates a small multi-layer perceptron (MLP) to **each** Gaussian surfel. The MLP takes the local coordinates `$(u,v)$` as input and directly outputs the color offset `$F_c$` and opacity `$F_\alpha$` for any intersection point on that surfel.

**How it works.** For each Gaussian primitive, a tiny neural network is instantiated:

$$F_c, F_\alpha = \text{MLP}(\mathbf{p})$$

where `$\mathbf{p} = (u,v)$` is the 2D local coordinate. The MLP uses a three-layer architecture with sigmoid activation functions internally. The input is 2-dimensional `$(u,v)$`, and the output includes both the RGB color offset and the opacity value.

**Design choices and parameter count.** The paper deliberately keeps the MLP shallow (three layers) to limit parameter count, but per-primitive parameters still far exceed the other two methods. Fig. 5 shows the parameter count comparison: a single movable kernel primitive has about 1.4× the parameters of a standard 2DGS primitive, while the tiny MLP version has substantially more. The three-layer default is a design choice — ablation (Table VIII) shows that increasing layers from 1 to 4 has negligible impact on reconstruction quality, with deeper networks introducing training instability without improving expressiveness. This suggests that even a shallow MLP provides sufficient representational capacity for per-primitive spatial variation.

**Training behavior.** The paper notes that optimizing per-primitive neural networks is "usually difficult with unstable convergence" (Section IV-B). This is a well-known challenge in hybrid explicit-implicit representations: the MLP parameters for each primitive interact through the rendering equation, and the optimization landscape can be more complex than for simple kernel-based methods. When the number of primitives is limited, the MLP variant shows strong representation ability (Table I, limited Gaussian columns), but with unlimited primitives, it performs worse than movable kernels due to optimization difficulties.

---

#### Integration with the Gaussian Splatting Rendering Pipeline

The spatially varying functions are integrated into the standard Gaussian Splatting alpha-blending framework. The rendering equation for a pixel remains:

$$C = \sum_{i=1}^{N} c_i \alpha_i \prod_{j=1}^{i-1} (1 - \alpha_j)$$

where `$c_i$` and `$\alpha_i$` are now the spatially varying color and opacity from each primitive (rather than constants), computed as `$c(\mathbf{p}_i, \mathbf{d}) = SH(\mathbf{d}) + F_c(\mathbf{p}_i)$` and `$\alpha(\mathbf{p}_i) = F_\alpha(\mathbf{p}_i)$`, with `$\mathbf{p}_i$` being the intersection point of the ray with the `$i$`-th surfel.

**Gradient flow.** The gradients of the photometric loss (L1 + D-SSIM, following 3DGS) back-propagate through the alpha-blending equation into the spatially varying function parameters — the quadrant colors/opacities in bilinear interpolation, the kernel centers and color/opacity contributions in movable kernels, and the MLP weights in the tiny MLP variant. The paper implements custom CUDA kernels for the forward and backward passes of each spatially varying function.

**Optimization settings.** Training follows the standard 3DGS/2DGS protocol: 30K iterations, gradient splitting threshold of 0.0002, opacity reset to 0.01 every 3000 iterations, and stopping of Gaussian splitting/cloning/removal after 15K iterations. The normal consistency loss from 2DGS is **disabled by default** because the paper's primary goal is novel view synthesis quality, not geometric reconstruction. An ablation (Table VI) confirms that SVGS outperforms 2DGS both with and without normal consistency loss.

**Primitive initialization.** Gaussian surfels are initialized from the sparse SfM point cloud (as in 3DGS/2DGS). The spatially varying function parameters are initialized randomly: for bilinear interpolation, the four quadrant colors start at zero (so `$F_c$` starts at zero, and only `$SH(\mathbf{d})$` contributes initially); for movable kernels, kernel centers are initialized at random positions on the surfel; for MLPs, network weights are randomly initialized. The `$SH(\mathbf{d})$` component provides the initial view-dependent color, and `$F_c$` gradually learns the spatial variation pattern.

**Parameter count efficiency.** A critical design aspect is that SVGS enhances per-primitive expressiveness while using **fewer total primitives**. The movable kernel variant uses approximately 1.4× the parameters per primitive compared to 2DGS (Fig. 5), but Table IX shows that even when 2DGS is given 2× the number of primitives (resulting in ~1.43× the total parameters of SVGS), SVGS still achieves better rendering quality. This is the "compactness" argument: each SVGS primitive does more work, so fewer are needed overall.