# Full GLM-OCR pipeline — sections 1-4 (think-high)

**2411.18966 — SVGS: Enhancing Gaussian Splatting Using Primitives with Spatially Varying Colors**



## 1. Executive Summary

The paper introduces **SVGS (Spatially Varying Gaussian Splatting)**, a method that equips 2D Gaussian surfels with spatially varying colors and opacity through bilinear interpolation, movable kernels, and tiny neural networks, enabling a single primitive to capture complex textures and geometry more compactly than the single-color baseline. Evaluated on Blender, DTU, Mip-NeRF360, and Tanks&Temples, the movable kernel variant achieves a PSNR of 35.08 on Blender—surpassing all prior Gaussian splatting methods—and matches the rendering fidelity of a parameter-scaled 2DGS while using only ~70% of the primitives (10k vs. 14k). The gain is largest when geometry is simple but textures are complex, establishing that spatially varying primitives can dramatically improve compactness and novel-view synthesis in such texture-dominated, geometrically simple regimes.


## 2. Context and Motivation

### The Core Problem: Non-Compact Representation in Gaussian Splatting

Gaussian splatting methods (3DGS [2], 2DGS [3], and their derivatives) represent a 3D scene as a collection of explicit primitives — ellipsoids or surfels — that are rasterized via alpha blending. Each primitive carries a center position, a covariance matrix defining its spatial extent, and a view-dependent color model (typically spherical harmonics). It also holds a single scalar opacity $\alpha$ that controls its contribution to the final image.

The fundamental bottleneck is that **the color and opacity are uniform across the surface of the primitive for a given viewing direction**. If two camera rays strike the same Gaussian primitive at different points on its surface, they receive exactly the same color and opacity (apart from the view-dependent effect, which varies with ray direction but not with the intersection location *on* the primitive). Figure 3(c) illustrates this: in the vanilla formulation, the color depends only on the viewing direction $\mathbf{d}$, and the opacity is a single global scalar $\alpha$, completely ignoring *where* on the primitive the ray lands.

Because of this, a scene with spatially varying textures — a checkerboard wall, a painted toy, a finely detailed billboard — must be approximated by **creating many tiny Gaussian primitives**, each covering a small patch and each carrying its own color and opacity. As the paper states:

> “when the scene has complex geometry and appearance, these methods have to create a large number of these simple Gaussians to approximate the spatially varying opacity and textures on the scene, which leads to a huge waste of Gaussians.”

This is the **non-compact representation** problem. The representation is inefficient: it uses an enormous number of primitives to capture what could, in principle, be represented by a single, more expressive primitive whose color and opacity vary across its surface. The consequences are:

- **High memory footprint**: storing many primitives consumes GPU memory.
- **Longer training time and rendering cost**: more primitives mean more parameters to optimize and more primitives to blend during rendering.
- **Rendering artifacts**: a dense cloud of tiny primitives can cause “smearing” or blur (Figure 8 shows wires and brackets that are blurred in 2DGS, but sharp in SVGS).
- **Degraded geometry**: when the primitives are too numerous and small, it becomes harder to extract a clean, coherent surface (as in the DTU geometry comparisons, where 2DGS with many primitives performs worse under a primitive budget).

The central gap the paper addresses is therefore: **can we make each Gaussian primitive more expressive by letting its color and opacity vary with the location where a ray hits it, thereby reducing the total number of primitives needed to represent a scene?**

### Why Is This Problem Important?

The problem is particularly acute for **real-world scenes that combine complex textures with relatively simple geometry**. Think of a textured wall, a printed poster, a painted tabletop, or the synthetic objects in the Blender [1] dataset — they often have smooth or planar surfaces but rich, detailed textures. Current Gaussian splatting methods are forced to use a very large number of small primitives to capture this texture detail, even though the underlying geometry could be represented by a few larger primitives. This is exactly the scenario the paper highlights: “scenes combining complex textures with relatively simple geometry occur frequently in real-world environments,” and on such scenes the inefficiency of uniform-color primitives is most severe.

Making primitives more spatially expressive yields several practical benefits:

- **Compactness**: the same scene can be represented with far fewer primitives. Table I shows that SVGS with 10k primitives outperforms 2DGS even when 2DGS is allowed many more primitives, and Table II demonstrates that when 2DGS is scaled to match the parameter count of SVGS (14k primitives vs. 10k), SVGS still achieves better quality — the gain comes from improved per-primitive expressiveness, not just more parameters.
- **Higher novel-view synthesis quality**: the gain is especially large on texture-dominated scenes with simple geometry. On the Blender dataset, SVGS with movable kernels reaches a PSNR of 35.08, surpassing all prior Gaussian splatting methods (Table III), because each primitive can now capture the detailed texture of a smooth surface without introducing a dense cloud of micro-primitives.
- **A balanced trade-off between geometry and appearance**: because SVGS builds on 2D surfels (2DGS), it inherits the strong geometric reconstruction of 2DGS while significantly improving NVS. The paper explicitly positions SVGS as a middle ground — “methods achieving higher NVS precision often exhibit inferior reconstruction quality (like 3DGS and NeRF), whereas those with superior geometric fidelity tend to underperform in NVS (NeuS and PGSR). SVGS thus occupies a balanced middle ground, delivering strong results in both aspects.” This is critical for applications like 3D asset creation, AR/VR, and robotics where both a clean surface and photorealistic rendering are desired.

Beyond the practical impact, the work has theoretical significance: it **redefines the design space of Gaussian primitives**. Instead of treating each primitive as a uniform “splat,” SVGS introduces the idea that a primitive can carry an internal function that varies spatially. This opens a new research axis — what kinds of functions, how many sub-components, what level of expressiveness — and shows that even simple, explicit functions (like movable kernels) can dramatically boost representation power without sacrificing real-time performance.

### Prior Approaches and Their Shortcomings

**Standard Gaussian Splatting and 2DGS.** 3DGS [2] and its surface-aligned descendant 2DGS [3] use a color model based purely on view-dependent spherical harmonics: $c(\mathbf{d}) = SH(\mathbf{d})$, and a scalar opacity $\alpha$. No matter where the ray hits the primitive, the color and opacity are the same (up to view-direction effects). This is the baseline that all prior methods inherit. Many subsequent improvements — Scaffold-GS [35] (which distributes Gaussians via anchor points), Mip-Splatting [36] (which filters primitives to prevent aliasing), MCMC-3DGS [41] (which reformulates the optimization as a sampling process) — all **keep the same per-primitive appearance model**. They improve optimization, densification, or anti-aliasing, but they do not make a single primitive capable of representing a textured, spatially varying pattern.

**Methods that Attempt to Add Spatial Variation.** A few recent works have recognized the need for more expressive primitives and tried to add texture or spatial variation:

- **Texture-GS [4]** disentangles geometry from appearance by learning a UV mapping per Gaussian and applying a 2D texture. This enables appearance editing, but it still relies on many primitives for the geometry; the texture is a separate, global (or per-primitive) map, and the color does not vary naturally with the intersection point in a simple, unified function.
- **Textured-GS [5]** (Huang and Gong) and **Textured-Gaus [6]** (Chao et al.) propose to equip each Gaussian with a texture map — through extra spherical harmonics degrees or explicit alpha/RGB maps — that can vary with the surface position. However, as Figure 12 (second row) shows, even when constrained to a single Gaussian, Textured-GS [5] suffers from “evident color and transparency attenuation near the center” and shows “pronounced multi-view inconsistency, with its color patterns rapidly deteriorating under even slight viewpoint changes.” Textured-Gaus [6] (third row) “fails entirely when restricted to a single Gaussian,” and even with 100 or 1000 primitives, the reconstruction remains coarse and unstable. Learning a texture map on a primitive is hard: the mapping from 3D points to a consistent 2D texture coordinate across views is ill-posed without a well-defined surface parametrization.
- **Splat-the-Net [39]** represents each primitive as a bounded neural density field (a shallow MLP) and derives an analytical line integral for perspective-accurate splatting. This is a more radical, volumetric approach that increases expressiveness, but it is fundamentally different from the simple, explicit 2D coordinate functions of SVGS and requires a line integral that is more costly than standard alpha blending. It also does not exploit the natural 2D coordinate system of a surfel.
- **3D-HGS [38] and NegGS [48]** allow negative opacity values (e.g., via half-Gaussian functions) to model sharp discontinuities. While these are, as the paper notes, “special cases of our SVGS” because SVGS permits negative outputs, they do not provide a mechanism for the opacity or color to vary as a function of *the surface position on the primitive* — they are still a single, fixed function per primitive that just happens to go negative.

In short, prior efforts either add a complex, global texture mapping that struggles with single-primitive fitting, or rely on learned neural functions that are less interpretable and harder to optimize, or they do not fully exploit the simple, explicit 2D coordinate system that a surfel offers. None of them provide a simple, lightweight, and effective way to let the color and opacity vary continuously across the surface of a single Gaussian primitive.

### How SVGS Positions Itself

SVGS takes a direct and principled route: it **embeds a spatially varying function into each 2D Gaussian surfel**, using the local $(u,v)$ coordinates of the ray-primitive intersection point (Figure 3(c)). Instead of the uniform color $SH(\mathbf{d})$ and opacity $\alpha$, the primitive now outputs:

$$c(\mathbf{p}, \mathbf{d}) = SH(\mathbf{d}) + F_c(\mathbf{p}), \quad \alpha(\mathbf{p}) = F_\alpha(\mathbf{p})$$

where $\mathbf{p} = (u,v)$ is the 2D intersection point on the surfel’s local coordinate system, and $F_c$, $F_\alpha$ are user-chosen functions that vary across the primitive’s surface. This is the “fundamental distinction” illustrated in Figure 3: the color and opacity become **spatially varying attributes** — different rays hitting the same primitive at different locations can get different colors.

The paper proposes and compares three concrete designs for $F$:

1. **Bilinear interpolation**: partition the surfel into four quadrants, each with a learnable color and opacity, and blend via bilinear interpolation (Figure 4(a)). This is like a simple, fixed texture grid.
2. **Movable kernels**: place $k=4$ learnable sub-kernels on the surfel, each with a center $K_i$ and an exponential radial basis function. The kernels can move during optimization (hence “movable”), providing high flexibility and a smooth, blending-based representation (Figure 4(b)).
3. **Tiny MLP**: a shallow 3-layer neural network that maps $(u,v)$ directly to an output color and opacity. This is the most expressive but also the most parameter-heavy and unstable to train (Figure 6, Table VIII).

The key design choice is that **all three are local, explicit functions** that operate on the 2D coordinate of the surfel. They do not require learning a global UV mapping or complex neural fields; they are simple to implement inside the existing CUDA splatting pipeline and can be optimized together with the other Gaussian parameters. The movable kernels, in particular, act like a set of “sub-primitives” that can shift to cover the most important parts of the primitive’s surface, giving a great deal of expressiveness per parameter.

By building on top of 2D Gaussian surfels, SVGS inherits the strong geometric reconstruction of 2DGS — the normal vector, the well-defined local frame — and adds spatial variation on top. This makes it especially effective in the **texture-dominated, geometrically simple** regime: on the Blender dataset, where objects are smooth but have detailed textures, SVGS with movable kernels achieves a PSNR of 35.08, surpassing all prior Gaussian splatting methods (Table III). On more complex geometry (Mip-NeRF360, Tanks&Temples), it still outperforms 2DGS and remains competitive with 3DGS-based methods, while using fewer primitives (Table I, IX).

Crucially, SVGS does **not** claim to be a universal replacement for all splatting pipelines. The paper explicitly states that “our method and MCMC-3DGS address Gaussian splatting from two distinct yet complementary perspectives. In theory, they are not mutually exclusive and could be integrated, which presents a promising direction for future research.” This positions SVGS as a **new primitive design** that can be combined with other advances in optimization, anti-aliasing, and densification. The paper’s goal is not merely to chase PSNR but to “seek a balanced trade-off between geometry reconstruction fidelity and novel-view synthesis accuracy”—a trade-off that is particularly valuable in practical, real-world applications where both a clean surface and a high-quality render are needed.


## 3. Technical Approach

### 3.1 Reader Orientation

SVGS is a method that takes the standard 2D Gaussian surfel primitive – which normally has a single view-dependent color and a single opacity – and gives it a spatially varying internal function so that different rays striking different points on the primitive can receive different colors and opacities. The core problem it solves is the non-compact representation in Gaussian splatting: the tendency to waste thousands of tiny primitives to model a single textured surface when each primitive’s appearance is uniform across its surface. The solution’s shape is a **per-primitive, locally defined function** that maps the 2D intersection point on a surfel to a color offset and an opacity, allowing the same scene to be represented with far fewer, more expressive primitives.

### 3.2 Big-Picture Architecture (Diagram in Words)

The system has five major components:

1. **Base 2D Gaussian primitives (surfels)** – Each primitive is a flat elliptical disc defined by a center position, two tangent axes that span its local 2D plane, and a scaling matrix controlling its size. It carries the conventional spherical harmonic (SH) view-dependent color and an initial opacity, plus the new spatially varying function parameters.
2. **Spatially varying functions $F_c$ and $F_\alpha$** – For each primitive, one of three designs (bilinear interpolation, movable kernels, or a tiny MLP) computes a color offset and an opacity value from the local intersection coordinate $(u,v)$. These functions are the core novel addition.
3. **Ray-surfel intersection computation** – When rendering a pixel, a ray is cast through the pixel into the scene. For each visible surfel, the intersection of the ray with the 2D plane is computed and expressed in the local $(u,v)$ coordinate system of that primitive (Fig. 3(c)).
4. **Alpha-blended splatting with modified CUDA kernels** – The spatially varying functions are evaluated at the intersection point, the final color and opacity are produced, and the standard splatting pipeline blends them onto the image. The forward and backward passes are implemented in custom CUDA kernels.
5. **Training loop** – The whole set of primitives (all parameters: positions, covariances, SH coefficients, and the spatially varying function parameters) is optimized via gradient descent against a photometric loss on the training views, using the same densification and pruning schedule as 2DGS [3].

Information flows as: input images & camera poses → initialise primitives → for each training iteration: sample a camera, project primitives, compute per-pixel ray-surfel intersections in local $(u,v)$ coordinates, evaluate $F_c(u,v)$ and $F_\alpha(u,v)$, blend into image via differentiable alpha blending, compute loss against ground truth, back-propagate to update all parameters. At test time, the same forward pass is used for novel views.

### 3.3 Roadmap for the Deep Dive

1. **The overall spatially varying formulation** – how color and opacity are redefined as functions of the local intersection point, and why this is a drop-in replacement for the uniform model.
2. **The local coordinate system on a 2D Gaussian surfel** – how the intersection point is parameterised and why surfels simplify the extension to 3D ellipsoids.
3. **Bilinear interpolation with quadrant colours** – the first concrete $F$, its parametric simplicity, and why it suffers from gradient vanishing.
4. **Movable kernels** – the second and best-performing $F$, its learnable sub-kernel positions, and the exponential basis that enables smooth, flexible spatial variation.
5. **Tiny MLPs** – the third, most expressive $F$, its per-primitive neural network, and the trade-off between capacity and optimisation stability.
6. **Training and optimisation integration** – how the three functions are welded into the 2DGS codebase, the CUDA modifications, and the specific hyperparameters that make the system work.

### 3.4 Detailed, Sentence-Based Technical Breakdown

This is an empirical methods paper whose core idea is to **replace the uniform color and opacity of a Gaussian primitive with a function that varies across the primitive’s surface**, using the 2D local coordinate of the intersection point as input. It is implemented on top of 2D Gaussian surfels, and the three candidate functions are compared on multiple datasets.

---

#### 3.4.1 Spatially Varying Gaussian Primitives: The Overall Formulation

Every existing Gaussian splatting method (3DGS [2], 2DGS [3], and their derivatives) defines the colour of a primitive through a view-dependent spherical harmonic function $SH(\mathbf{d})$ and a single scalar opacity $\alpha$, both of which are **independent of where on the primitive a viewing ray lands**. The paper’s first and most fundamental change is to make the colour and opacity **spatially varying attributes**:

$$c(\mathbf{p}, \mathbf{d}) = SH(\mathbf{d}) + F_c(\mathbf{p})$$

$$\alpha(\mathbf{p}) = F_\alpha(\mathbf{p})$$

where $\mathbf{p} = (u,v)$ is the local 2D coordinate of the intersection point between the ray and the Gaussian surfel (Fig. 3(c)), $\mathbf{d}$ is the viewing direction, $SH(\mathbf{d})$ is the ordinary spherical harmonic function that captures view-dependent effects, $F_c$ is a spatially varying function that adds a colour offset, and $F_\alpha$ is a spatially varying function that produces the opacity directly.

**What these equations compute:** For a given pixel and a given primitive, the colour is the standard view-dependent spherical harmonic colour **plus** a correction that depends on which part of the primitive’s surface the ray hit. The opacity is entirely determined by the spatial location on the primitive, not by the viewing direction. If two different rays hit the same primitive at different surface points, they will receive different colours and opacities – this is the “spatially varying” property.

**Why this form:** The additive form preserves the well-established view-dependent spherical harmonic model, which is good for modelling specular highlights and view-dependent reflections, while layering on a **spatially varying offset** that can capture surface texture. Separating the opacity from view dependence is natural: opacity is a surface property and should not change when you look from a different angle. Moreover, the paper imposes **no explicit constraints** on the values of $F_c$ and $F_\alpha$ – they can output negative numbers. This is permissible because the outputs are added to the SH colour (which is already unconstrained in value) and because the renderer applies a sigmoid activation to both colour and opacity before blending them into the final image. Enforcing value constraints during optimisation would be computationally expensive; the unconstrained formulation lets the primitive automatically learn to turn off parts of itself by driving opacity below the pruning threshold, or to create negative colour offsets that can model sharp discontinuities. The authors note that 3D-HGS [38] and NegGS [48] are therefore special cases of this more general formulation.

The design of $F_c$ and $F_\alpha$ is the main subject of the paper, and three concrete realisations are explored. All three operate on the local intersection coordinate $(u,v)$ in the **2D coordinate system of the primitive**, which is explained next.

---

#### 3.4.2 Computing the Intersection Point on 2D Gaussian Surfels

To evaluate the spatially varying function, we must know exactly where a given ray strikes the primitive. The paper builds on **2D Gaussian surfels** [3], which are flat ellipses embedded in 3D space. Each surfel is defined by a centre point $\mu$, two orthogonal tangent vectors $\mathbf{t}_u$ and $\mathbf{t}_v$ that span its 2D local plane, and associated scaling factors along those axes. The 2D Gaussian’s influence is a function of the distance from a point to the surfel’s centre in this 2D manifold.

When rendering a pixel, a ray is cast from the camera centre through the pixel. For each surfel, the renderer computes the **intersection of the ray with the surfel’s plane**. This intersection point is then expressed in the local coordinate system of that surfel:

- The origin $(0,0)$ is the surfel’s centre point $\mu$.
- The coordinate axes are the tangent vectors $\mathbf{t}_u$ and $\mathbf{t}_v$, so that a point’s local coordinates are obtained by projecting the 3D offset vector $(\mathbf{p} - \mu)$ onto $\mathbf{t}_u$ and $\mathbf{t}_v$, and scaling appropriately by the inverse of the primitive’s size.

The resulting $(u,v)$ is a continuous 2D parameter that tells us where on the primitive’s surface the ray landed. This is exactly the input to $F_c$ and $F_\alpha$.

The choice of 2D surfels (rather than 3D ellipsoids) is important for simplicity: an ellipsoid would require a ray-ellipsoid intersection that yields a 3D point on the curved surface, and then a mapping from that 3D point to a 2D local coordinate system, which is more complex and less numerically stable. The paper acknowledges this limitation and states that “our discussion can also be extended to 3D Gaussians … while the calculation of intersection points requires careful consideration,” but for the current work the 2D surfel is the natural and simpler substrate.

---

#### 3.4.3 Bilinear Interpolation with Quadrant Colours

The first concrete $F$ divides each surfel into four quadrants and uses bilinear interpolation to blend the colours and opacities of those quadrants. Each quadrant has its own learnable colour vector $c_i$ and opacity scalar $\alpha_i$, for $i = 0,1,2,3$, corresponding to the four corners of the interpolation grid.

To define the interpolation, the unbounded local coordinates $(u,v)$ are first mapped into the $(0,1)$ range via a sigmoid rescaling:

$$u' = \frac{1}{1 + e^{-\lambda_s u}}, \quad v' = \frac{1}{1 + e^{-\lambda_s v}}$$

where $\lambda_s = 5.0$ controls the steepness of the transition. The spatially varying colour offset and opacity are then:

$$F_c(p) = (1 - u')(1 - v') c_0 + (1 - u') v' c_1 + u' (1 - v') c_2 + u' v' c_3$$

$$F_\alpha(p) = (1 - u')(1 - v') \alpha_0 + (1 - u') v' \alpha_1 + u' (1 - v') \alpha_2 + u' v' \alpha_3$$

where $c_i$ and $\alpha_i$ are the new learnable parameters associated with each quadrant.

**What these equations compute:** The colour and opacity at a given $(u,v)$ are a linear combination of the four corner values, weighted by the fractional position inside the unit square. The sigmoid mapping compresses the unbounded $(u,v)$ into the range $(0,1)$, ensuring that the interpolation always operates within a well-defined domain. This effectively partitions each primitive into four regions, each with an independent colour and opacity, and blends them smoothly across the boundaries.

**Why this form:** Bilinear interpolation is the simplest possible spatial variation function – it is a fixed, piecewise-linear blend of four corner values. It adds only 8 additional parameters per primitive (4 colour vectors + 4 opacity scalars). However, the sigmoid rescaling creates a **gradient vanishing problem** near the centre of each quadrant. When $u$ or $v$ is large in magnitude, the sigmoid saturates at 0 or 1, and its derivative becomes very small, meaning that the gradient signal from a pixel in the centre of a quadrant cannot effectively reach the corner parameters. This is why the method is only effective when the natural texture pattern conforms to a four-quadrant distribution, and why it underperforms the movable kernels on complex, real-world scenes (Fig. 12, top row). The authors note this in Sec. IV-E: “this scaling causes the gradient of a pixel to be almost zero when it falls near the center of a quadrant, making it difficult to perfectly fit the scene.”

---

#### 3.4.4 Movable Kernels: Learnable Sub-Primitives

The second design generalises the fixed quadrants into **movable sub-kernels** that are not constrained to any predetermined position on the primitive. Each surfel is equipped with $k = 4$ tiny kernels, each with a learnable 2D centre $K_i = (K_i^x, K_i^y)$ in the same local $(u,v)$ coordinate system, a learnable colour $c_i$, and a learnable opacity $\alpha_i$. The colour and opacity at a point $p = (u,v)$ are computed as a weighted sum of these kernel outputs:

$$F_c(p) = \sum_{i=0}^{k-1} F_{K_i}(p) \, c_i, \qquad F_\alpha(p) = \sum_{i=0}^{k-1} F_{K_i}(p) \, \alpha_i$$

where the weight of the $i$-th kernel is an exponential radial basis function:

$$F_{K_i}(p) = e^{-\lambda_e \|p - K_i\|^2}, \quad \lambda_e = 0.1.$$

**What these equations compute:** For every point $(u,v)$ on the primitive, the system computes the Euclidean distance to each kernel’s centre $K_i$, converts that distance into a weight via a decaying exponential, and then blends the kernel’s own colour $c_i$ and opacity $\alpha_i$ with those weights. The kernel centres are **learnable parameters** – they move during optimisation, so the primitive can reallocate its expressive power to the regions that need it most. If a kernel moves outside the Gaussian’s effective region, the exponential decay rapidly suppresses its contribution, and the primitive effectively reverts to the ordinary 2DGS behaviour.

**Why this form:** The movable kernel function is a **smooth, continuous, and fully differentiable** alternative to bilinear interpolation. It avoids the gradient vanishing problem because the exponential kernel function has a non-zero derivative everywhere (except exactly at the centre, which is a single point of measure zero). The kernels can shift to concentrate on high-frequency detail, and the sum-of-exponentials produces a soft blending that is much more flexible than a rigid four-quadrant grid. The exponential decay rate $\lambda_e = 0.1$ is chosen to make each kernel’s influence fairly localised, so that the optimisation of one kernel does not corrupt another, and the total number of kernels $k = 4$ is a trade-off between expressiveness and parameter count (each kernel adds 2 position coordinates + 3 colour channels + 1 opacity = 6 parameters; total 24 extra parameters per primitive, which is about 1.4× the 2DGS base parameter count, as shown in Fig. 5). The paper also experiments with an alternative sigmoid-based kernel $F_{S_i}(p) = 1 - \tanh(\|p - K_i\|^2)$ (Table VII) and finds the exponential kernel slightly superior, which is consistent with the fast-decay, localised nature of the exponential being more stable during gradient-based optimisation.

The key design insight is that the kernels are **movable**. During training, the gradient pulls the kernel centres towards areas of high texture error, and pushes them away from areas that are already well-captured by other primitives. This creates a dynamic, adaptive partition of the primitive’s surface that is not tied to a fixed grid. The authors report that in practice, “the kernel position moves with the gradient… there is almost no instance where the kernel moves outside the Gaussian” (Table X), so the learned kernels reliably stay within the primitive’s bounds.

---

#### 3.4.5 Tiny MLP: Per-Primitive Neural Network

The third design is to replace the hand-crafted function with a small, **per-primitive neural network** – a tiny multilayer perceptron (MLP) that is attached to each Gaussian surfel.

$$F_c(p), F_\alpha(p) = \text{MLP}(p)$$

where $\text{MLP}$ is a 3-layer fully connected network that takes the 2D local coordinate $(u,v)$ as input and outputs a 4-dimensional vector (RGB colour offset + opacity). The paper uses a sigmoid activation function inside the MLP, and the output is used directly as $F_c$ and $F_\alpha$.

**What this equation computes:** The MLP learns an arbitrary continuous function from the 2D surface coordinates to colour and opacity. Every primitive has its own independent set of network weights, so no two primitives are forced to share the same spatial pattern.

**Why this form:** An MLP is the most general function approximator considered in the paper – it can represent any continuous spatial pattern, including sharp transitions, nonlinear blends, and complex colour variations that are not well captured by the simpler parametric forms. However, this generality comes at a steep cost: the per-primitive MLP adds a large number of parameters (Fig. 5 shows it is far greater than the other two methods), and training hundreds or thousands of tiny neural networks simultaneously is **unstable**. The optimisation of the MLP is difficult because the gradients are less structured than those of the explicit kernels, and the network’s capacity can overfit to the current training view rather than generalising across views. Table VIII shows that increasing the number of layers from 1 to 4 barely changes the PSNR, and the 3-layer version actually slightly underperforms the movable kernels. This is a classic trade-off: more expressive, but harder to optimise, and the paper’s results demonstrate that the simple, explicit functions already provide enough expressiveness for the tested scenes.

The paper also notes that the tiny MLP’s performance is particularly good when the number of Gaussian primitives is severely limited (Table I, “w/ limited number”), because with very few primitives the MLP’s high capacity can compensate for the restricted primitive count. But in the normal regime with many primitives, the explicit formulations are easier to train and achieve higher quality.

---

#### 3.4.6 Training and Optimisation Integration

All three spatially varying functions are implemented inside a modified version of the 2DGS [3] codebase. The standard 2DGS already renders images by projecting 2D Gaussian surfels onto the image plane, evaluating the Gaussian weight at each pixel, and performing alpha blending. The paper’s main CUDA modification is to **replace the uniform colour and opacity with a call to the spatially varying function** at the intersection point.

**Forward pass:** For each primitive that influences a pixel, the renderer computes the 3D intersection of the pixel’s ray with the primitive’s plane, expresses that point in the local $(u,v)$ coordinates, evaluates $F_c(u,v)$ and $F_\alpha(u,v)$ using the chosen function, adds the result to the SH colour, and then blends the resulting colour and opacity with the other primitives in the sorted depth order using the standard alpha-blending equation:

$$C = \sum_{j} c_j \alpha_j \prod_{k=1}^{j-1} (1 - \alpha_k)$$

where $c_j$ and $\alpha_j$ are now the spatially varying outputs.

**Backward pass:** The custom CUDA code also computes the gradients of the photometric loss with respect to all the new parameters: the 4 corner colours and opacities for bilinear interpolation; the 4 kernel centres, colours, and opacities for movable kernels; and the MLP weights for the tiny MLP. These gradients are then used by the Adam optimiser alongside the standard 2DGS gradients.

**Training hyperparameters:** The paper follows the 2DGS/3DGS protocol exactly:
- **Iterations:** 30,000.
- **Gradient threshold for splitting:** 0.0002 (positional gradient), with splitting and cloning of Gaussians stopped after 15,000 iterations.
- **Opacity reset:** Every 3,000 iterations, all opacities are reset to 0.01 to allow dead primitives to be pruned.
- **Optimiser:** AdamW (the original 3DGS paper uses Adam; 2DGS likely uses the same, though not re-specified in SVGS).
- **Normal consistency loss:** Disabled for main NVS experiments; enabled only for geometry experiments on DTU.
- **Hardware:** Single NVIDIA A100 80GB GPU.

The paper does not report the learning rate or other Adam hyperparameters, as they are presumably the same as in the 2DGS reference implementation.

**Design choice – no normal loss for NVS:** The authors deliberately disable the normal consistency loss for novel-view synthesis because it can hurt the photometric quality; they only enable it when evaluating geometric reconstruction on DTU. Table VI shows that omitting normal loss improves PSNR for both SVGS and 2DGS.

**Additional ablation on training time:** Because the per-primitive function evaluation is slower than the baseline (see Table XII), the paper also tests a **reduced-iteration** setting where SVGS is trained for fewer steps to match the 2DGS wall-clock time. Even with a time-equivalent budget, SVGS outperforms 2DGS (Table XII, right half), demonstrating that the per-primitive expressiveness is not just a matter of more compute – it is a genuine capacity improvement.

**Geometry reconstruction:** For surface evaluation (Chamfer Distance on DTU), the normal consistency loss is re-enabled, and the paper shows that SVGS with movable kernels achieves geometric quality comparable to 2DGS when the number of primitives is unrestricted, and **significantly better** when the primitive count is limited (Tables IV, V; Figs. 10, 11). This is because the spatially varying functions allow each primitive to cover more area with a coherent colour and opacity, which in turn gives the surface normal optimisation a stronger, less fragmented signal.

---

#### Summary of Design Choices and Their Justifications

- **2D surfel substrate over 3D ellipsoid**: Simpler ray-primitive intersection that yields a clean 2D local coordinate, with the most important axis aligned with the surface normal; 2DGS already provides strong geometric priors.
- **Additive colour formulation ($SH + F_c$)**: Preserves the well-tested view-dependent SH model while enabling spatial variation; the unconstrained output range allows negative contributions that the sigmoid activation later clamps.
- **Opacity as pure spatial function ($F_\alpha$)**: Opacity is a surface property that should not depend on viewing direction; this also generalises prior work on negative opacities.
- **Sigmoid rescaling for bilinear coordinates**: Maps unbounded coordinates into $(0,1)$, but the steepness $\lambda_s = 5.0$ is a compromise that creates a bottleneck where gradients vanish – a known limitation.
- **Exponential kernel for movable kernels ($\lambda_e = 0.1$, $k=4$)**: Provides localised, smooth, and gradient-friendly weights; $k=4$ is a sweet spot between expressiveness and parameter count (24 extra parameters per primitive).
- **Tiny 3-layer MLP with sigmoid activation**: Keeps the per-primitive network as small as possible to avoid explosion of parameters, but still underperforms explicit functions due to optimisation instability.


## 4. Key Insights and Innovations

### Innovation 1: Spatially Varying Gaussian Primitives as a New Primitive Design Space

Before SVGS, every Gaussian splatting method—3DGS, 2DGS, and the wave of densification, anti-aliasing, and optimization variants that followed—treated the primitive as having a single, uniform color (modulated only by view direction through spherical harmonics) and a single scalar opacity. The cost of a complex texture was a dense, redundant cloud of tiny primitives, each covering a small, uniformly colored patch. SVGS breaks this by introducing **spatially varying color and opacity** directly inside a single primitive, using the local intersection coordinate $(u,v)$ as the input. This is not a modest parameter tweak; it redefines what a Gaussian primitive *is*. A primitive becomes a small, textured patch that can internally vary its appearance, rather than a monochromatic blob.

This shift from “uniform splat” to “textured splat” opens a new design space: what kind of spatially varying function, how many sub‑components, and how to balance expressiveness against parameter count. Prior works that attempted to add spatial variation—such as Textured‑GS [5] (Huang and Gong, 2024) which encodes spatial variation through higher‑degree spherical harmonics, or Textured‑Gaus [6] (Chao et al., 2024) which attaches explicit alpha/RGB texture maps—either struggled with single‑primitive fitting (Figure 12, second and third rows), suffered from multi‑view inconsistency, or needed many primitives to achieve a coarse result. SVGS’s simple, explicit, and **per‑primitive local** formulation avoids these pitfalls entirely: Figure 1 shows a single SVGS primitive can fit a round plane with four distinct colors, a task that neither 2DGS nor 3DGS can accomplish.

The consequence is that the representation becomes fundamentally more compact. Table I shows that with only 10k primitives, SVGS with movable kernels reaches a Blender PSNR of 35.08; when 2DGS is scaled to match the total parameter count (14k primitives, Table II), it still underperforms—the gain comes from per‑primitive capacity, not from more overall parameters. This establishes spatially varying primitives as a **new design axis** orthogonal to earlier advances in densification or anti‑aliasing, and one that can be combined with them.

---

### Innovation 2: Movable Kernels as an Adaptive, Explicit Sub‑Primitive Mechanism

The bilinear interpolation and the tiny MLP are straightforward attempts at spatial variation, but the **movable kernel design** is the paper’s most distinctive technical contribution—not because it is a new function form, but because it combines the simplicity of an explicit basis with the flexibility of learnable, shifting sub‑components. Prior to SVGS, methods that tried to give a Gaussian primitive spatial variation either used a fixed, regular grid (e.g., a texture map with static coordinates) or a black‑box neural network, both of which suffer from gradient vanishing or optimization instability. The movable kernel approach places $k=4$ learnable 2D centers on the primitive, each with a local color and opacity, and blends them via a distance‑weighted exponential function. Crucially, these kernels **move** during optimization—they are not tied to fixed quadrants. They can migrate toward the regions of highest texture error, adaptively allocating the primitive’s limited representational budget to where it matters most.

This is a novel, lightweight, and gradient‑friendly way to achieve spatially varying appearance *without* a global UV parameterization, and without the high parameter count and training fragility of an MLP. As the ablation in Table VII confirms, 4 exponential kernels are the sweet spot; 8 kernels add parameters without improving quality. The exponential kernel also avoids the gradient vanishing that plagues bilinear interpolation (Figure 12, top row), making it more robust in complex scenes. This design is not a minor refinement of a fixed‑grid texture; it is a new conceptual building block—a “sub‑primitive kernel” that can be used as a drop‑in to enhance any Gaussian‑based pipeline, and the results in Table I show it consistently outperforms the other spatial functions across all tested datasets.

---

### Innovation 3: 2D Surfels as the Enabling Substrate for Compact Spatial Variation—and the Recognition of a Texture‑Dominated, Geometry‑Simple Regime

An often‑overlooked but crucial conceptual move in SVGS is the choice to build on **2D Gaussian surfels** rather than 3D ellipsoids. The paper argues that the flat, well‑defined local frame of a 2D surfel provides a natural and simple parameterization for the intersection point $(u,v)$, making the spatially varying function a clean, direct, and computationally light addition. Extending the same idea to 3D ellipsoids would require a ray‑ellipsoid intersection and a 3D‑to‑2D mapping that is “more complex and less numerically stable” (Section III‑A). This is not a mere implementation detail; it is a design insight: **the 2D primitive is a more suitable canvas for texture than the 3D primitive**, precisely because its geometry is simpler.

At the same time, the paper explicitly identifies a regime—**scenes with complex textures but relatively simple geometry**—where this advantage is most pronounced. In such scenes (predominantly the Blender dataset), 3DGS was forced to use a large number of small, uniform primitives to approximate the texture, while SVGS can represent the same surface with far fewer, more expressive primitives. This reframes the non‑compact representation problem: it is not that Gaussian splatting is inherently inefficient; it is that the primitive design was mismatched to the texture complexity of the surface.

By adopting 2D surfels and adding spatial variation, SVGS achieves a **balanced trade‑off** between high‑fidelity novel‑view synthesis (where 3DGS and NeRF excel) and high‑quality geometric reconstruction (where NeuS and 2DGS are strong). The paper explicitly positions SVGS as a middle ground: “methods achieving higher NVS precision often exhibit inferior reconstruction quality… whereas those with superior geometric fidelity tend to underperform in NVS… SVGS thus occupies a balanced middle ground.” This is a valuable, practice‑oriented innovation: it shows that for many real‑world applications—where a clean surface and photorealistic rendering are both required—the 2D surfel with spatially varying colors is a uniquely suitable representation, and that the same idea can be integrated with other advances (e.g., anti‑aliasing, MCMC optimization) as a modular primitive upgrade.