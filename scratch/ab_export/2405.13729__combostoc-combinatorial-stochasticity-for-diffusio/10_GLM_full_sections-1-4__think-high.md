# Full GLM-OCR pipeline — sections 1-4 (think-high)

**2405.13729 — ComboStoc: Combinatorial Stochasticity for Diffusion Generative Models**



## 1. Executive Summary

This paper identifies that standard diffusion training undersamples the **combinatorial complexity** of high-dimensional, multi-attribute data—and proposes **ComboStoc** (Combinatorial Stochasticity), a training scheme that vectorizes the interpolation schedule $t$ into a tensor of the same shape as the data, applying independently sampled timestep values per dimension to achieve uniform sampling density across the full combinatorial space (e.g., asynchronous timesteps across image patch tokens and feature channels, or across 3D shape parts, existence indicators, bounding boxes, and shape codes). Training with ComboStoc accelerates convergence and yields systematic FID-50K improvements over SiT and DiT baselines on ImageNet (Tab. 1), while on PartNet structured 3D shapes it is indispensable for generating valid outputs—establishing that the approach is most effective when data exhibits strong combinatorial structure and diminishes when dimensions are nearly independent. ComboStoc also enables a novel test-time generation mode: asynchronous timesteps allow different dimensions to be generated with varying degrees of preservation, supporting graded control such as soft inpainting and part-based shape completion.


## 2. Context and Motivation

### The Core Problem: Diffusion Models Undersample the Combinatorial Complexity of Structured Data

The central issue this paper identifies is that standard training of diffusion generative models systematically **undersamples the combinatorial space** formed by the many dimensions and attributes that constitute a high‑dimensional data point. In the prevailing formulation — summarized by the one‑sided stochastic interpolant \(x_t = (1 - t)z + tx_1\) — every feature, patch, or attribute of a sample follows the **same scalar interpolation schedule** \(t\). This means that all training samples lie exactly on the diagonal line connecting a pure noise sample \(z\) and a target data point \(x_1\); the model never sees samples where different dimensions are at *different* stages of the diffusion process.

The paper shows that this synchronized training leads to a **non‑uniform sampling density** over the path space. In particular, the probability density of sampling a point \(x\) in the rectangular region \(\mathcal{R} = \{x \mid z \preceq x \preceq x_1\}\) is

\[
\rho(x) = \int_0^1 G_{p_t}(x)\,dt,
\qquad G_{p_t} = \mathcal{N}\!\left(p_t;\;(1-t)^2 \mathbf{1}\right),
\]

and a straightforward gradient calculation (Section 3, Eq. 4) yields

\[
(x_1 - x) \cdot \nabla\rho(x) > 0,
\]

i.e., \(\rho(x)\) **grows monotonically as we move toward the target data point** \(x_1\). As visualized in Figure 2(d), this means that the standard training distribution collapses toward the data points, leaving large portions of the path space — especially regions far from the data manifold — with very low training coverage. In a high‑dimensional space where data points are few and the “curse of dimensionality” makes the manifold sparsely sampled, this undercoverage becomes severe.

Why does this matter? Diffusion models are evaluated by **solving a stochastic differential equation (SDE) from noise to data**, a process that can visit *any* point in the path space, including those far from the training distribution. When the network encounters such under‑covered regions at test time (e.g., due to a perturbed or asynchronous integration trajectory), it produces unreliable predictions, resulting in **poor generation quality, slow convergence, or outright failure** on small datasets. The paper demonstrates exactly this: on the PartNet structured‑3D dataset (≈18 K shapes), the baseline synchronized model (`unsync_none`) fails to produce valid shapes (Figure 8), and on ImageNet (1.3 M images), it converges substantially more slowly than the ComboStoc variants (Figure 6). The problem is therefore *not just theoretical* — it is a practical bottleneck that limits the performance and data efficiency of modern diffusion models.

### Why Combinatorial Complexity Is Important

The significance of this gap grows with the **structural richness** of the data. Most interesting generative tasks involve data that are not just a bag of independent coordinates but exhibit strong internal structure:

- **Images** are represented as collections of patch tokens (spatial) and feature channels (color/texture). A state‑of‑the‑art model like SiT treats an image as a grid of latent patches, each encoding a high‑dimensional vector; the interplay between patches and their feature dimensions creates a combinatorial space that the model must navigate to produce coherent, globally consistent images.
- **Structured 3D shapes** (e.g., PartNet objects) push this even further: a shape is a set of semantic parts, each described by an existence indicator, a 6‑dimensional bounding box, and a 512‑dimensional latent shape code. The number of parts varies across objects, and the parts are permutation‑invariant — the same 3D shape can be represented by many different orderings of its parts. The resulting combinatorial complexity is far larger than that of an image with a fixed grid of patches.

In both cases, the model must learn to **correlate different dimensions and attributes** — patches must agree on the overall structure, part shapes must be consistent with part poses, and so on. If training only covers the diagonal of the path space, the model never experiences the full range of “intermediate” combinations where some dimensions are more denoised than others, missing the opportunity to learn these correlations robustly. Furthermore, small datasets exacerbate the problem: with few data points, the already sparse path space becomes even harder to cover, making the model’s failure on structured 3D shapes (Figure 8) almost inevitable.

### Where Existing Approaches Fall Short

The paper situates its contribution against several lines of prior work, each of which addresses only a slice of the problem or requires substantial architectural changes.

**Standard diffusion training (DDPM, score‑based, flow matching, stochastic interpolants).** Virtually all mainstream diffusion frameworks — whether they predict noise, score, or velocity — apply a single scalar time \(t\) simultaneously to all dimensions of a sample. This is the fundamental design that creates the biased sampling density shown in Figure 2. While the training objective is often well‑behaved, the *distribution of training inputs* is not; the model is never forced to reason about off‑diagonal states where, say, the left half of an image is nearly fully denoised while the right half is still pure noise. As a result, when the SDE solver does encounter such states (through stochastic perturbations or during user‑controlled generation), the model’s outputs can be of poor quality, and on structured or small data, the model may not learn at all.

**Spatially or temporally varying noise schedules for specific tasks.** Several recent works have explored assigning different noise levels to different regions or time steps, but **exclusively at inference time**. SDEdit [Meng et al. 2021] injects a global noise level for stroke‑based editing; SVNR [Pearl et al. 2023] uses spatially variant diffusion for denoising; video diffusion models such as FIFO‑Diffusion [Kim et al. 2024] and Diffusion Forcing [Chen et al. 2024] assign stronger noise to later frames to reflect temporal uncertainty. These methods are **task‑specific** and do not alter the training distribution to cover combinatorial space uniformly; they only modify how a pre‑trained model is sampled. The paper explicitly notes that these approaches “do not address the training‑time combinatorial coverage perspective studied here” (Section 2.1).

**Masked patch training for images.** The work of Gao et al. [2023] (Masked Diffusion Transformer) is the closest prior attempt to force a model to learn inter‑patch dependencies during diffusion training. It randomly masks portions of a diffused image and asks the model to reconstruct the masked regions, which implicitly creates off‑diagonal states. However, this approach relies on a **complex encoder–decoder architecture with side‑interpolation modules**, making it far from a minimal, general modification. In contrast, ComboStoc requires only a vectorized timestep tensor and a simple drift compensation — no additional network components.

**Hierarchical and task‑specific 3D shape generation.** For structured 3D shapes, prior methods such as StructureNet [Mo et al. 2019a] and StructRe [Wang et al. 2025] exploit hierarchical part decompositions (coarse‑to‑fine) to regularize generation. While effective for consistency, they **do not directly model the full combinatorial space of leaf‑level parts and attributes** in a unified diffusion framework. ComboStoc instead works on flatly structured leaf‑part ensembles, showing that by simply covering the combinatorial space during training, a single generative model can outperform or match these specialized, hierarchy‑aware baselines (Table 5), while also enabling new test‑time capabilities like part‑level assembly and graded completion.

**Accelerating diffusion training via representation learning.** Orthogonal to the combinatorial coverage problem, techniques like REPA [Yu et al. 2024] distill pre‑trained visual representations into the denoiser to speed up convergence. ComboStoc’s approach is **complementary**: it addresses the *sampling of inputs* rather than the *learning objective*, and the paper suggests that combining both could yield further gains.

### How ComboStoc Positions Itself

The paper frames its contribution as a **simple, principled fix** that converts the one‑dimensional interpolation schedule \(t\) into a **tensor of the same shape as the data**, with each entry independently sampled from \([0, 1]\) (Equation 5). This single change:

1. **Uniformly covers the combinatorial subspace** spanned by each pair of source noise and target data point (Figure 2(b,e) vs. (a,d)), guaranteeing that the model sees all possible mixtures of partially denoised dimensions and attributes during training.  
2. **Eliminates the sampling bias** that the paper analytically exposes in the standard linear interpolant (Section 3). The training distribution becomes uniform within the hyper‑rectangle \(\mathcal{R}\), so no region is systematically under‑covered.  
3. **Forces the model to learn inter‑dimension correlations** because it must predict how to move *any* off‑diagonal state toward the data point — the model must learn to “synchronize” the different dimensions to reach a coherent final sample.  
4. **Enables a new test‑time paradigm** where different dimensions can be given different degrees of preservation (asynchronous inference), allowing graded control that ranges continuously from pure generation to full preservation (Section 5.3). This unified framework subsumes many specialized editing tasks without retraining.

Unlike the prior “inference‑only” asynchronous methods, ComboStoc **modifies training** to match the full combinatorial space, which directly improves standard synchronized generation (all FID gains in Section 5.2 are obtained with synchronized inference). The paper also provides a careful theoretical analysis — showing that with a simple drift‑compensation term (Eq. 10), the off‑diagonal velocity field still integrates correctly to the target data point, so the scheme is not merely an ad‑hoc augmentation but a **proper generative flow model** (Section 4.1).

Crucially, ComboStoc is **not specific to any architecture**. While the paper demonstrates it on transformer‑based SiT and a custom 3D shape network, the same principle can be applied to U‑Net diffusers by simply tensorizing the timestep and modulating feature maps accordingly (as noted in the conclusion). The paper’s empirical results across images and 3D shapes — with consistent improvements in FID, training speed, and data efficiency — argue that **the combinatorial perspective is a fundamental, under‑explored dimension of diffusion model design**, and that fully sampling it during training is a low‑cost, high‑impact strategy.


## 3. Technical Approach

### 3.1 Reader orientation (approachable technical breakdown)

The paper constructs a **training and inference framework** that converts the scalar diffusion timestep into a tensor of the same shape as the data, so that during training the model is exposed to every possible off‑diagonal mixture of partially denoised dimensions. This solves the problem of **path‑space undersampling**—where standard diffusion formulations leave large regions of the interpolation space unvisited—by guaranteeing uniform coverage of the combinatorial subspace, and in doing so also enables a new test‑time mode where different dimensions can be generated with different degrees of finalisation.

### 3.2 Big-picture architecture (diagram in words)

The system has five major interacting components:

1. **Data representation** – for images, a VAE‑encoded latent tensor `$x_1 \in \mathbb{R}^{C\times H\times W}$`; for structured 3D shapes, a set of part tokens, each containing an existence indicator, a 6‑dimensional bounding box, and a 512‑dimensional shape code.
2. **Timestep tensorizer** – produces a tensor `$t$` of the same shape as `$x_1$`, where every entry is sampled independently from `$[0,1]$`. Different dimensions (patches, channels, parts, attributes) can therefore receive different timestep values.
3. **Interpolation module** – blends source noise `$z \sim \mathcal{N}(0,\mathbf{1})$` and target data `$x_1$` element‑wise via `$x_t = (1 - t) \odot z + t \odot x_1$`. This is the core modification of the linear stochastic interpolant.
4. **Denoising network** – a transformer (SiT‑style for images, a custom structured‑shape transformer for 3D) that takes `$x_t$`, a class label `$c$`, and the tensor‑shaped timestep `$t$` (after embedding), and predicts either the velocity `$x_1 - z$` (images) or the target data point `$x_1$` (3D shapes).
5. **Velocity compensation** – when asynchronous timesteps are used at test time, a correction term `$v_{\text{cmpn}}$` is added to the predicted velocity during integration to pull off‑diagonal trajectories back to the target data point.
6. **Inference scheduler** – a procedure (Algorithm 1) that can run either a **synchronized** schedule (all entries of `$t$` equal, producing the usual high‑quality generation) or an **asynchronous** schedule where different dimensions follow different timestep trajectories, enabling graded control.

Information flow during training: a target data point `$x_1$` and a noise sample `$z$` are drawn → a tensorized timestep `$t$` is generated → the interpolated sample `$x_t$` is formed → the network (conditioned on `$c$` and the embedded `$t$`) predicts the target quantity → a regression loss is minimised. At test time, either noise `$z$` is integrated to a final sample (synchronous), or a user‑specified mask `$m$` determines the initial state and timestep, and the model generates the remaining dimensions (asynchronous).

### 3.3 Roadmap for the deep dive

- **First**, we examine the **sampling bias** built into the standard one‑sided linear interpolant—this is the mathematical motivation for why simply desynchronising the timestep matters.
- **Second**, we introduce the core **ComboStoc interpolation** (Equation 5) that replaces the scalar `$t$` with a tensor of independent timestep values, and show how it uniformly covers the combinatorial subspace.
- **Third**, we present the **theoretical justification** that this off‑diagonal scheme still defines a proper generative flow model (the continuity equation and marginalised velocity field), so that training on off‑diagonal samples is consistent with a valid probability path.
- **Fourth**, we detail the **off‑diagonal drift problem** that arises when the standard velocity prediction is used with asynchronous timesteps at test time, and the two **compensation strategies** (off‑diagonal drift minimisation and cone‑shaped velocity).
- **Fifth**, we explain the **architectural adaptation**—how the timestep embedding module is modified to accept a tensor‑shaped timestep input for both images and structured 3D shapes.
- **Sixth**, we describe the **training configurations**, enumerating the different levels of combinatorial flexibility (`unsync_patch`, `unsync_vec`, `unsync_all`, etc.) and the corresponding training strategies (mixed vs. full asynchronous batches).
- **Finally**, we present the **inference procedures** for both standard synchronous generation and the novel asynchronous “graded control” mode, including the integration algorithms.

### 3.4 Detailed, sentence‑based technical breakdown

This is primarily an **empirical analysis paper** whose central idea is that **replacing the scalar diffusion timestep with a per‑dimension tensor during training** fully covers the combinatorial space of data dimensions and attributes, thereby eliminating a systematic sampling bias and yielding both faster convergence and a new test‑time control interface.

---

#### 3.4.1 The Sampling Bias in Standard Diffusion Training

Standard diffusion generative models—whether DDPM, score‑based, flow matching, or stochastic interpolants—all train on samples that lie on a **single interpolation path** connecting a noise source to a data point. The most basic such formulation is the linear one‑sided stochastic interpolant (Equation 1):

$$
x_t = (1 - t) \, z \;+\; t \, x_1,
\qquad t \in [0,1],
$$

where `$z \sim \mathcal{N}(0,\mathbf{1})$` is a draw from the source distribution, `$x_1 \sim \mathcal{D}$` is a draw from the target data distribution, and `$t$` is a **scalar** interpolation schedule. Every training sample therefore lies exactly on the diagonal line segment connecting `$z$` to `$x_1$`; no sample ever sees a state where, for instance, one spatial patch is nearly fully denoised while another is still pure noise.

The paper shows that this training scheme creates a **non‑uniform sampling density** over the path space. For a fixed pair `$(z, x_1)$`, the subspace of all possible intermediate points is the hyper‑rectangle

$$
\mathcal{R} = \{x \mid z \preceq x \preceq x_1\},
$$

where `$\preceq$` denotes element‑wise inequality. The probability density of sampling a point `$x \in \mathcal{R}$` during training is obtained by integrating, over all `$t$`, the Gaussian distribution centred at the moving point `$p_t = (1-t)z + t x_1$` with variance scaled by the interpolation coefficient `$1-t$`:

$$
\rho(x) = \int_0^1 G_{p_t}(x) \, dt,
\qquad
G_{p_t} = \mathcal{N}\!\big(p_t;\; (1-t)^2 \mathbf{1}\big).
$$

Substituting `$p_t$` explicitly gives

$$
\rho(x)
= \frac{1}{\sqrt{2\pi}} \int_0^1
\frac{1}{1-t}
e^{-\frac{\|t(z - x_1) + x - z\|^2}{2(1-t)^2}}
\, dt.
$$

This integral has no closed form, but a direct computation of the gradient reveals an informative property:

$$
(x_1 - x) \cdot \nabla\rho(x)
= \frac{1}{\sqrt{2\pi}}
e^{-\frac{\|x - z\|^2}{2}}
\;>\; 0.
$$

**What this gradient equation shows:** The directional derivative of `$\rho(x)$` in the direction `$x_1 - x$` (i.e., moving from an arbitrary point `$x$` toward the target data point) is **strictly positive**. Consequently, `$\rho(x)$` **monotonically increases** as one approaches `$x_1$`. Figure 2(d) visualises this with a clear shrinking of density toward the data point. In high dimensions with few data points, this means that the vast majority of the path space—especially regions far from the target manifold—receives very little training coverage. When the SDE solver later visits such under‑covered regions during test‑time integration, the network’s predictions become unreliable, causing slow convergence or outright failure on small, structured datasets.

**Why this form matters:** This analysis is not dependent on the specific interpolant; any formulation that moves along a single diagonal path `$x_t = \sigma_t z + \alpha_t x_1$` (with different coefficients) still suffers from the same **undersampling of off‑diagonal states**, because the training distribution is always concentrated along a one‑dimensional curve in the high‑dimensional combinatorial space. The paper’s insight is that to fix this, one must **break the synchronisation** of the timestep across dimensions.

---

#### 3.4.2 The ComboStoc Interpolation: Uniform Sampling of the Combinatorial Space

The core of ComboStoc is a minimal but decisive modification: instead of a scalar `$t$`, we use a **tensor of timesteps** with the same shape as the data. Every entry of this tensor is an **independent uniform draw** from `$[0,1]$`. The blended sample is then obtained through an element‑wise operation:

$$
x_t = (1 - t) \odot z \;+\; t \odot x_1,
$$

where `$\odot$` denotes element‑wise (Hadamard) product. **`$t$` is now a tensor of shape matching `$x_1$` and `$z$`, with `$t_{ijk} \sim \mathcal{U}(0,1)$` independently.** For a latent image of shape `$(C, H, W)$`, this gives each spatial location and each feature channel its own randomly sampled diffusion stage. For a structured 3D shape, the tensor `$t$` spans part indices, existence indicators, bounding‑box dimensions, and per‑part shape codes.

**What this equation computes:** Given a noise sample `$z$` and a data sample `$x_1$`, it produces an interpolated point `$x_t$` where each dimension `$d$` is placed at a stage `$t_d$` along the line from `$z_d$` to `$(x_1)_d$`. Because every dimension’s `$t_d$` is independent, the resulting point can lie **anywhere inside the hyper‑rectangle** `$\mathcal{R} = \{x \mid z \preceq x \preceq x_1\}$`, not just on the diagonal. The sampling density over `$\mathcal{R}$` is therefore **uniform by construction**: every sub‑region is equally likely to be visited during training.

**Why this form:** The alternative—keeping `$t$` a scalar—confines all training samples to the diagonal and creates the density collapse analysed in Section 3.4.1. By vectorising `$t$`, we **fully sample the combinatorial complexity** of the data: the model must learn to handle states where, e.g., the left half of an image is nearly clean while the right half is still heavily noisy, or where the existence of some parts is already determined while their shape codes remain undetermined. This forces the network to learn the **correlations among dimensions and attributes**—to synchronise them—in order to reach a coherent final sample.

Crucially, the scheme is **not architecture‑specific**. Applying it requires only the ability to produce a tensor `$t$` of the appropriate shape and to condition the network on it. In the following sections we detail how this conditioning is implemented for transformer‑based image models and for structured 3D shape models.

---

#### 3.4.3 Ensuring a Proper Generative Flow: Marginalised Velocity Field and Continuity Equation

Simply training on off‑diagonal samples with the standard target `$x_1 - z$` raises a question: does the resulting model still define a valid generative process that transports the source distribution `$p_0$` to the target distribution `$p_1$`? The paper answers this with a **formal proof** showing that the ComboStoc scheme, when the conditional vector field is properly defined, satisfies the **continuity equation**.

The key is to define a **conditional vector field** `$u(x \mid x_0, x_1)$` that generates a point‑wise probability path `$p_t(x \mid x_0, x_1)$` moving from `$x_0$` to `$x_1$` inside the **rectangular region** `$\text{span}(x_0, x_1)$`. Unlike standard flow matching, where the conditioning is only on the data point `$x_1$` and the vector field is confined to the diagonal `$\text{diag}(x_0, x_1)$`, here the conditioning is on **both the source and target points** `$(x_0, x_1)$`. The conditional vector field `$u(x \mid x_0, x_1)$` is time‑invariant and, by construction, moves a point mass from `$x_0$` to `$x_1$` in the rectangular subspace.

The **marginalised velocity field** `$u_t(x)$` is then defined as the expectation of this conditional field under the source and target distributions:

$$
u_t(x)
= \frac{1}{p_t(x)}
\iint
u(x \mid x_0, x_1)\;
p_t(x \mid x_0, x_1)\;
q(x_0)\; r(x_1)
\; dx_0\, dx_1,
$$

where `$q(x_0)$` is the source noise distribution, `$r(x_1)$` the target data distribution, and `$p_t(x)$` is the marginal probability density at time `$t$`.

The paper then shows that this `$u_t(x)$` and the corresponding probability path `$p_t(x)$` satisfy the **continuity equation**:

$$
\frac{d}{dt} p_t(x)
= \iint \Big( \frac{d}{dt} p_t(x \mid x_0, x_1) \Big) q(x_0)\, r(x_1)\, dx_0\, dx_1
= - \iint \operatorname{div}\!\big( u(x \mid x_0, x_1)\, p_t(x \mid x_0, x_1) \big) q(x_0)\, r(x_1)\, dx_0\, dx_1
= -\,\operatorname{div}\!\left( \iint u(x \mid x_0, x_1)\, p_t(x \mid x_0, x_1)\, q(x_0)\, r(x_1)\, dx_0\, dx_1 \right).
$$

**What this chain of equalities establishes:** The left‑hand side is the time derivative of the marginal probability density. The first equality expands it into an integral over the conditioning pair `$(x_0, x_1)$`. The second equality uses the fact that `$u(x \mid x_0, x_1)$` generates the conditional path `$p_t(\cdot \mid x_0, x_1)$` that moves from `$x_0$` to `$x_1$`, so the time derivative of the conditional density equals the negative divergence of the flux produced by that conditional vector field. The third equality switches the order of integration and differentiation (permitted by the regularity of the integrands). The final equality is simply the definition of the marginalised vector field as the expected flux.

**Why this form matters:** It proves that the ComboStoc conditional vector field, when marginalised over all source‑target pairs, produces exactly the **same probability path** `$p_t$` as the original flow matching formulation—but now the path is supported on the **full rectangular region**, not just the diagonal. The model trained to predict this marginalised velocity therefore generates a valid flow from noise to data. Note that `$u(x \mid x_0, x_1)$` is time‑invariant; any apparent time dependence in `$u_t(x)$` arises only after marginalisation over `$p(x_0)$`.

In practice, for **images** the training target remains the constant velocity `$x_1 - z$` (as in SiT), and for **3D shapes** it is the data point `$x_1$` (an `$x$`‑prediction). The off‑diagonal drift that would otherwise occur during test‑time integration with asynchronous timesteps is handled by a **compensation strategy**, analysed next.

---

#### 3.4.4 The Off‑Diagonal Drift Problem and Velocity Compensation

When the model is evaluated with asynchronous timesteps—i.e., different dimensions start from different initial times `$t_0$`—the simple integration of the constant velocity `$x_1 - z$` no longer reaches the target data point. This is the **off‑diagonal drift problem**.

Let `$x_{t_0}$` be a sample at a tensorised timestep `$t_0$` (entries may differ), and let `$\underline{t}_0 = \min(t_0)$` be the smallest entry. If we integrate the constant velocity `$x_1 - z$` from `$t_0$` to `$1$` (using the same step count for all dimensions), the resulting point is:

$$
x_{t_0} + \int_{\underline{t}_0}^{1} (x_1 - z)\, dt
= z + t_0 \odot (x_1 - z) + (1 - \underline{t}_0)(x_1 - z)
= x_1 + (t_0 - \underline{t}_0) \odot (x_1 - z),
$$

where the integral is taken with a uniform step size for all dimensions, so dimensions with larger `$t_0$` effectively “stop early” and accumulate a smaller displacement. **The outcome is `$x_1$` plus an offset proportional to the asynchrony `$t_0 - \underline{t}_0$`**, hence the trajectory drifts away from the target.

To neutralise this drift, the paper introduces a **compensation velocity** `$v_{\text{cmpn}}$` defined as the negative of the **off‑diagonal offset vector**:

$$
\delta(x_t)
= x_t - x_1
- \frac{(x_t - x_1) \cdot (x_1 - z)}{\|x_1 - z\|^2}
\,(x_1 - z),
\qquad
v_{\text{cmpn}} = -\delta(x_t).
$$

**What this computes:** `$\delta(x_t)$` is the component of the vector `$x_t - x_1$` that is **orthogonal** to the diagonal direction `$x_1 - z$`. It measures how far the current point deviates from the diagonal line connecting `$z$` to `$x_1$`. By subtracting this orthogonal component from the (constant) velocity, we steer the trajectory back onto the diagonal and ensure convergence to `$x_1$`. The operation is equivalent to performing gradient descent on a drift potential `$\Phi(\delta) = \tfrac{1}{2}\|\delta\|^2$`.

**Why this form:** Simply following the raw velocity `$x_1 - z$` from an off‑diagonal start would leave the orthogonal offset unchanged (the velocity has no component orthogonal to the diagonal). The compensation term adds that missing orthogonal component, pulling the sample toward the diagonal without altering the along‑diagonal progress. Compared with the alternative **cone‑shaped velocity field** (Equation 11):

$$
v_{t_0} = \frac{x_1 - x_{t_0}}{1 - \underline{t}_0},
$$

which directly points from `$x_{t_0}$` to `$x_1$` but scales inversely with the remaining time `$1 - \underline{t}_0$`, the drift‑minimisation approach is **better behaved**: the cone velocity can grow arbitrarily large as `$\underline{t}_0 \to 1$`, introducing numerical instability and degrading regression quality (Table 6 shows cone velocity yields higher FID and lower SSIM than drift minimisation). The paper therefore adopts **off‑diagonal drift minimisation** as the default compensation strategy for all experiments.

For **synchronous generation** (all entries of `$t$` equal), `$\underline{t}_0 = t_0$` and the offset `$\delta$` is identically zero; no compensation is needed. This is why the FID gains reported in Section 5.2 are obtained purely from the **training** benefit of ComboStoc, without any test‑time compensation.

---

#### 3.4.5 Network Adaptation: Timestep Embedding for Tensorised Inputs

The standard SiT model [Ma et al. 2024] conditions the transformer on a scalar timestep `$t$` via a two‑stage process: first, `$t$` is encoded into a sinusoidal frequency vector of length `$C_F$`; then an MLP projects this vector to a conditioning tensor of size `$C_H$` (the hidden dimension), which is used to modulate the transformer layers through adaptive layer norm (adaLN) operations. This design assumes `$t$` is a **single scalar** shared across all patches.

To accept the tensorised `$t \in \mathbb{R}^{C \times H \times W}$`, the paper modifies the embedding module as follows (Figure 5(b)):

1. **Per‑entry frequency encoding.** Each scalar entry of the tensor `$t$` is independently passed through the same sinusoidal encoding as in the original SiT, producing a vector of length `$C_C$` (set to `$4$` to keep the embedding compact). The result is a tensor of shape `$(N, C, H, W, C_C)$`, where `$N$` is the batch size.

2. **Patch‑wise embedding.** The channel dimension of this feature tensor is merged with the spatial dimensions (`$H, W$`) to form a sequence of local patch vectors. Specifically, the tensor is reshaped to `$(N, T, C \cdot C_C)$` where `$T = H \times W / L^2$` is the number of patches (with patch size `$L$`). A **learned linear projection** (the same patch‑embedding architecture as ViT, but with separate parameters) maps each of these `$T$` tokens to a vector of dimension `$C_H$`. This produces a **patch‑wise timestep conditioning tensor** of shape `$(N, T, C_H)$`, which is then added to the image patch tokens or used in the modulation layers.

**Why this design:** The patch‑wise embedding ensures that **different spatial locations and different feature channels receive their own timestep signal** at the same granularity as the image tokens. By keeping `$C_C = 4$` small, the total number of extra parameters is modest (the overall model has slightly fewer parameters than SiT, Table 8). However, the smaller timestep encoding capacity also means that the `unsync_none` configuration—which uses this modified module but with all entries of `$t$` equal—has a slightly weaker timestep representation than the original SiT, which may explain why `unsync_none` performs marginally worse than baseline SiT in Figure 6(a).

For **structured 3D shapes**, the data point `$x_1$` is a collection of part tokens. Each part is represented as `$p = (s, b, e)$`, where `$s \in [0,1]$` indicates existence, `$b \in \mathbb{R}^6$` encodes the bounding box, and `$e \in \mathbb{R}^{512}$` is a latent shape code. The timestep `$t$` is therefore a tensor with different entries for **part index**, **attribute type** (existence, bounding‑box dimension, shape‑code channel), and **feature dimension** (individual coordinates of the shape code). The embedding for each scalar attribute (existence, bounding‑box elements) follows the same two‑stage scheme—sinusoidal encoding to a small `$C_C$`‑dimensional vector, then a learned linear layer to `$C_H$`—before being combined into a part‑level token. This per‑attribute conditioning allows the model to distinguish, for instance, between a part whose existence is already determined (near `$t=1$`) and one whose shape code is still largely noise (near `$t=0$`).

---

#### 3.4.6 Training Configurations: Levels of Combinatorial Flexibility

The paper explores a **spectrum of combinatorial flexibility** by varying which dimensions receive independent timestep values. For **images**, four configurations are defined (summarised in Table 2(a)):

| Setting | Timestep splitting rule |
| :--- | :--- |
| `unsync_none` | No split; a single scalar `$t$` for all dimensions. |
| `unsync_patch` | Different `$t$` per **spatial patch** (i.e., per `$(h,w)$` location); same across channels. |
| `unsync_vec` | Different `$t$` per **feature channel**; same across patches. |
| `unsync_all` | Different `$t$` per **patch and per channel** simultaneously. |

`unsync_all` represents the fullest exploitation of combinatorial complexity: within a single training sample, the left and right halves of the image can be at different denoising stages, and the early (structure‑focused) and late (colour‑focused) VAE latent channels can also be at different stages (cf. Figure 13).

For **structured 3D shapes**, the combinatorial axes are **parts**, **attributes** (existence, bounding box, shape code), and **feature dimensions** (individual coordinates of the shape code). This yields six configurations (Table 2(b)):

| Setting | Splitting rule |
| :--- | :--- |
| `unsync_none` | No split. |
| `unsync_part` | Different `$t$` per **part index**. |
| `unsync_att` | Different `$t$` per **attribute type** (existence vs. bbox vs. shape code). |
| `unsync_att_part` | Different `$t$` per **attribute type and per part**. |
| `unsync_vec` | Different `$t$` per **feature vector dimension** (i.e., per channel of the 512‑dim shape code). |
| `unsync_all` | Different `$t$` per **part, per attribute, and per feature dimension**. |

**Training strategy.** For ImageNet (1.3 M images), the paper employs a **mixed‑batch strategy**: in each training batch, **half of the samples** use the fully asynchronous tensor `$t$` (according to the chosen `unsync_*` setting), and **the other half** use a standard synchronised scalar `$t$` (identical to baseline SiT). This acts as a form of curriculum: pure asynchronous training on a massive dataset can make early optimisation more difficult because the model struggles to learn a coherent velocity field across all off‑diagonal states simultaneously; retaining half of the samples on the diagonal provides an “anchor” that stabilises convergence. For the relatively small **PartNet** dataset (≈18 K shapes), this stabilisation is unnecessary, and **all samples** use the asynchronous schedule without mixing.

**Hyperparameters.** All image models are trained from scratch using the **AdamW** optimiser with a fixed learning rate of `$10^{-4}$` and a batch size of `$256$` on 4 NVIDIA H100 GPUs (800 K iterations, ≈7.5 days). The structured 3D shape model uses the **same optimiser and learning rate**, with a batch size of `$16$` on 4 NVIDIA A100 GPUs (1.5 K epochs, ≈3 days). Image generation uses the standard SiT **velocity prediction** loss (`$v$`‑prediction) to isolate the effect of ComboStoc; structured 3D shape generation uses **`$x$`‑prediction** (predicting the clean data point directly) as a more robust choice for the heterogeneous attribute mixture that includes both continuous coordinates and discrete existence indicators. The evaluation of image quality uses the **SDE integrator** with 250 steps; 3D shapes use **500 sampling iterations** with per‑step binarisation of part existence at a 0.5 threshold.

---

#### 3.4.7 Inference: Synchronous Evaluation and Asynchronous Control (Algorithm 1)

The paper provides two distinct inference modes, both using the same trained network `$f_\theta$`.

**(A) Synchronised inference (standard generation).** This is the mode used for all FID and quantitative evaluations in the paper. The procedure (Algorithm 1(A)) is:

1. Sample source noise `$z \sim \mathcal{N}(0, \mathbf{1})$`.
2. Initialise `$x^{(0)} \leftarrow z$`.
3. For `$k = 0$` to `$K-1$` (with `$K=250$` for images), set a uniform scalar `$t_k = k/K$` and construct a **synchronised time tensor** `$t^{(k)} = t_k \cdot \mathbf{1}$` (all entries equal).
4. Feed this `$t^{(k)}$` and the class label `$c$` to the model to obtain the predicted velocity `$\hat{v}^{(k)} = f_\theta(x^{(k)}; c, t^{(k)})$`.
5. Update the sample via the **SiT‑style integrator** (a standard Euler or Heun step): `$x^{(k+1)} \leftarrow \text{SITSTEP}(x^{(k)}, \hat{v}^{(k)}, t_k, t_{k+1})$`.
6. Return the final `$x^{(K)}$`.

Because all dimensions share the same `$t_k$`, the integration trajectory stays on the diagonal, and **no drift compensation is required**—the standard velocity `$x_1 - z$` is sufficient. This is exactly the inference of the baseline SiT, but the model weights have been trained with the combinatorial stochasticity.

**(B) Asynchronous inference with graded control.** This mode exploits the per‑dimension timestep to allow **flexible, continuous specification of how much of a reference sample is preserved**. The procedure (Algorithm 1(B)) is:

1. Given a reference sample `$x_1$` and a **mask tensor** `$\mathbf{m} \in [0,1]^{\text{shape}(x)}$`, where each entry `$m_i$` specifies the **degree of preservation** for that dimension (`$m_i = 1$` means fully preserved, `$m_i = 0$` means generated from scratch).
2. Construct the initial state as `$x^{(0)} = (1 - \mathbf{m}) \odot z + \mathbf{m} \odot x_1$`. The asynchronous initial timestep tensor is set to `$t^{(0)} = \mathbf{m}$`.
3. Set a step size `$\Delta t = (1 - \mathbf{m}) / K$` (element‑wise). Entries with larger `$m_i$` start closer to `$x_1$` and therefore have a smaller effective step size, evolving more slowly.
4. For `$k = 0$` to `$K-1$`:
   - Feed the **fully asynchronous time field** `$t^{(k)}$` to the model to obtain the predicted velocity `$\hat{v}^{(k)}$`.
   - **Apply velocity compensation** using the off‑diagonal drift minimisation (Equation 10): the effective velocity becomes `$v = \hat{v}^{(k)} - \delta(x^{(k)})$`, where `$\hat{v}^{(k)}$` is the model’s raw prediction (typically `$x_1 - z$`).
   - Update the sample with the SiT integrator: `$x^{(k+1)} \leftarrow \text{SITSTEP}(x^{(k)}, v, t^{(k)}, t_{k+1})$`.
   - Update the timestep: `$t^{(k+1)} \leftarrow \min(1, \, t^{(k)} + \Delta t)$`.
5. Return the final `$x^{(K)}$`.

This unified framework allows a single model to perform **soft inpainting** (Figure 12, where the mask varies smoothly across space), **graded control over image structure vs. colour** (Figure 13, where different VAE latent channels are assigned different preservation weights), and **part‑level shape completion/assembly** (Figures 14–15, where some parts are fixed while others are generated).

**Design choice on step sizing.** The paper compares two strategies: **uniform step number** (all dimensions take the same `$K$` steps, as above) and **uniform step size** (dimensions with larger `$m_i$` take fewer steps because their total time interval is shorter). An ablation (Table 7) shows that the results are similar, and the uniform‑step‑number scheme is adopted for simplicity.

---

#### 3.4.8 Putting It All Together: Training Procedure and Loss

**Training objective.** For images, the model is trained to predict the **velocity** `$x_1 - z$` (the `$v$`‑prediction of SiT). The loss is the standard mean‑squared error between the predicted velocity and the target velocity, evaluated at all sampled points `$x_t$` (both on‑diagonal and off‑diagonal, depending on the `unsync_*` setting). For structured 3D shapes, the model predicts the **clean data point** `$x_1$` directly (`$x$`‑prediction), with an `$\ell_2$` loss on all attributes (existence, bounding‑box coordinates, shape code).

**Batching and mixing.** At each training iteration, a batch of `$B$` data‑noise pairs is drawn. For ImageNet, `$B/2$` of these pairs use the full tensorised `$t$` (with independent entries per dimension), and the remaining `$B/2$` use a scalar `$t$` (synchronised). For PartNet, all `$B$` pairs use the asynchronous `$t$`. The timestep tensor is then embedded as described in Section 3.4.5, and the model forward pass produces the prediction. The gradient is computed and the optimiser steps.

**Drift compensation during training?** The off‑diagonal drift minimisation is **not used during training**; the training target is simply the standard velocity `$x_1 - z$` (or `$x_1$`). The compensation is only applied at **inference time**, specifically for the asynchronous grading mode. This is because during synchronous evaluation the integration path is diagonal, so the standard velocity is exact. During asynchronous inference, the compensation corrects the trajectory that would otherwise drift due to the asynchrony of the timestep.

**Why this split works:** Training on off‑diagonal samples with the standard target forces the network to learn a **consistent velocity field** that, when integrated along the diagonal, recovers `$x_1$`. The off‑diagonal drift is a **test‑time integration artefact**, not a training signal mismatch; the model learns to output the correct velocity from any state, and the compensation merely adjusts the integration to account for the asynchrony of the time steps. This is analogous to how a flow matching model trained on all `$t$` learns a time‑dependent velocity field that is integrated with a specific solver; here, the solver is modified to include a corrective force.


## 4. Key Insights and Innovations

### Innovation 1: The Sampling Bias Diagnosis — Exposing Why Standard Diffusion Training Leaves the Path Space Under‑Covered

The paper’s first intellectual move is to **identify and mathematically characterize a systematic sampling bias** that pervades virtually all standard diffusion generative models. Prior work had implicitly accepted that training on the diagonal interpolation path `$x_t = (1-t)z + t x_1$` provides a sufficient distribution of training inputs. The ComboStoc analysis shows otherwise: the probability density `$\rho(x)$` of sampling a point `$x$` in the hyper‑rectangle between a noise source and a data point is **not uniform** — it collapses toward the target data point `$x_1$`. Formally, the gradient `$(x_1 - x) \cdot \nabla\rho(x) > 0$` (Section 3, Eq. 4) reveals that density **monotonically increases** as one moves toward the data manifold. This is not a minor technicality; it means that large swathes of the path space — especially regions far from the data points — receive **vanishingly little training coverage**, and the model is never forced to learn how to handle them.

Why is this insight significant? It is a **diagnostic concept** that reframes the well‑known difficulties of training diffusion models on small, highly structured datasets (e.g., the failure of `unsync_none` on PartNet 3D shapes, Figure 8) not as a problem of insufficient capacity or poor optimization, but as a **sampling deficiency** in the input distribution. Prior approaches either assumed the diagonal path was adequate (standard DDPM, flow matching, stochastic interpolants) or attempted to mitigate the resulting poor performance with task‑specific heuristics (masking in Gao et al. [2023], hierarchical decomposition in StructureNet [Mo et al. 2019a]) or inference‑time noise schedules (SDEdit, SVNR, video diffusion). ComboStoc’s diagnosis shows that the root cause is **uniform**: every method that confines training to a single interpolation path suffers from this density collapse, because the path space is inherently high‑dimensional and the data points are sparse.

This is a **fundamental shift** in understanding: rather than treating the interpolation path as a convenient, neutral training support, the paper exposes it as a **biased sampling distribution** that systematically starves the model of experience in the vast off‑diagonal regions of the combinatorial space. The visual and analytical evidence in Figure 2 (d vs. e) and the toy example in Figure 3 (first row) make this concrete — standard flow matching produces non‑converging outliers in the integrated trajectories, while uniform coverage eliminates them. This diagnostic is the intellectual engine that motivates the entire ComboStoc framework and stands as a stand‑alone contribution to the theory of diffusion generative models.

### Innovation 2: The Tensorized Timestep as a Principled, Architecture‑Agnostic Fix — Replacing a Scalar with a Tensor to Cover the Full Combinatorial Space

Having diagnosed a sampling bias that is both pervasive and previously ignored, the paper proposes a **fix of striking minimality and generality**: instead of a scalar timestep `$t$`, use a **timestep tensor** of the same shape as the data, with independent entries sampled uniformly from `$[0,1]$` (Eq. 5). This one change — applied during training — transforms the sampling distribution from a collapsed diagonal to a **uniform density over the entire combinatorial hyper‑rectangle** spanned by each source‑target pair (Figure 2(e)). It is, as the paper emphasises, “extremely simple,” requiring no additional network components, no complex side‑modules, and no task‑specific engineering.

The novelty of this move lies in its **conceptual leap beyond the scalar‑time paradigm**. For decades, the default in diffusion models, score‑based models, stochastic interpolants, and even the broader family of continuous normalizing flows has been to treat `$t$` as a global, one‑dimensional variable that applies identically to every degree of freedom. ComboStoc **breaks this homogeneity** by recognising that the data’s dimensions and attributes are not independent identical copies of a scalar; they are **structured, correlated, and combinatorially rich**. The vectorised timestep is therefore not merely a schedule tweak — it is a **change in the very definition of the interpolation process**, from a line to a volume.

Crucially, this is not an ad‑hoc augmentation. The paper provides a **mathematical proof** (Section 4.1, Eqs. 6–7) that the conditional vector field defined over the rectangular subspace still yields a marginalised velocity field that satisfies the continuity equation and generates the correct probability path from noise to data. This establishes that ComboStoc is a **proper generative flow model** — training on off‑diagonal samples is not just a heuristic that happens to work, but a principled, theoretically grounded way to expand the support of the training distribution without breaking the transport guarantees of the original formulation.

Compared to prior attempts to expose models to off‑diagonal states — notably the masked diffusion training of Gao et al. [2023], which requires a separate encoder‑decoder and side‑interpolation, or the inference‑only spatially varying noise schedules of SVNR and video diffusion — ComboStoc is **architecture‑agnostic and minimally invasive**. As the paper notes, the same principle can be applied to U‑Net diffusers by simply tensorising the timestep and modulating feature maps accordingly. This makes it a **broadly applicable and easy‑to‑adopt strategy** that does not tie itself to any particular network. The consistent FID‑50K improvements over both SiT and DiT baselines (Figure 6, Table 1), achieved with a model that actually has **fewer parameters** than SiT (Table 8), underscore that the gains come from the **training distribution**, not from increased capacity.

Finally, the innovation is not just the vectorised timestep itself, but the **training strategy** that balances on‑diagonal and off‑diagonal samples (mixed‑batch for large data, full‑async for small data) and the **velocity compensation** that makes asynchronous inference possible. Together, they form a complete, principled, and practical framework.

### Innovation 3: Combinatorial Complexity as an Essential Axis — Structured Data Demands Asynchronous Training

The paper makes a second conceptual contribution by **demonstrating that the value of ComboStoc scales with the combinatorial complexity of the data**, and that for data with strong internal structure, covering the combinatorial space is not just a helpful augmentation but a **prerequisite for a working generative model**. This is an empirical insight that reframes the problem of training diffusion models on structured data: it is not simply a matter of “more data” or “larger models,” but of **adequately sampling the combinatorial space of dimensions and attributes**.

The evidence is starkest in the 3D structured shape generation task on PartNet. The baseline `unsync_none` — which uses the standard synchronised scalar timestep — **fails entirely** to produce valid shapes (Figure 8). In contrast, as the combinatorial flexibility is progressively increased (from `unsync_none` through `unsync_part`, `unsync_att`, `unsync_vec`, to `unsync_all`), the quality of generated shapes dramatically improves. The best setting, `unsync_all`, achieves FPD, COV, and MMD scores competitive with or exceeding state‑of‑the‑art hierarchical methods like StructRe and StructureNet (Table 5), even though ComboStoc works directly on flat leaf‑level parts without exploiting the tree hierarchy. This shows that **flatly covering the full combinatorial space can compensate for, and even surpass, the regularisation provided by hierarchical priors**.

On images (ImageNet), the effect is more subtle but equally instructive: the training acceleration is proportional to how finely the timestep is split. `unsync_all` (splitting per patch and per channel) consistently outperforms `unsync_patch` and `unsync_vec`, which in turn outperform `unsync_none` (Figure 6(b)). This reveals that the **combinatorial richness of the data** — the interplay between spatial patches and feature channels — is what the model struggles to learn from diagonal training alone. The wall‑clock time comparison (Figure 6(c)) confirms that the quality gains more than compensate for the slight per‑step overhead.

This is a **fundamental shift in the perspective on data efficiency for diffusion models**. Prior work had largely attributed the failure of diffusion on small datasets (e.g., PartNet) to the “curse of dimensionality” in the data space itself. ComboStoc shows that the real bottleneck is the **training distribution’s coverage of the path space**, which is even more severely affected by dimensionality than the data distribution alone. By uniformly covering the rectangular region, ComboStoc effectively **amplifies the effective training signal** from a limited number of data points, making it possible to train a generative model where standard approaches collapse (as in the 1,000‑image ablation, Figure 17). The paper thus establishes combinatorial complexity as a **first‑class design consideration** for diffusion training, and provides a simple, quantifiable axis — the degree of timestep asynchrony — along which to tune.

### Innovation 4: Unified Graded Control — Asynchronous Inference Turns One Model into a Continuum of Generators

Beyond improving standard generation quality, ComboStoc’s tensorised timestep enables a **new test‑time capability that unifies a wide range of previously task‑specific editing and generation operations under a single, general interface**: **graded control**. Given a reference sample `$x_1$` and a mask tensor `$\mathbf{m} \in [0,1]^{\text{shape}(x)}$`, the model can generate a new sample where each dimension is preserved to the degree specified by `$m_i$` — from fully preserved (`$m_i = 1$`) to completely regenerated (`$m_i = 0$`). This is not a discrete binary mask; it is a **continuous, per‑dimension knob** that can vary smoothly across space, channels, and attributes.

This capability is conceptually distinct from the standard diffusion inpainting, where a region is either replaced or kept, and from the task‑specific noise schedules in SDEdit, FIFO‑Diffusion, or video diffusion models. Those methods were designed as **inference‑time add‑ons** to a synchronously trained model, and they often require careful tuning of global noise levels or are limited to specific masking patterns. ComboStoc’s graded control emerges **naturally from the training scheme**: because the model was trained on off‑diagonal states with different degrees of denoising, it has learned to handle the full continuum of preservation levels. No retraining, no additional modules, no special inpainting fine‑tuning are required.

The paper demonstrates this across both image and 3D domains. For images:
- Spatially varying `$m_i$` allows **soft inpainting** with smooth, continuous transitions (Figure 12), where the model faithfully preserves a central subject while generating diverse surroundings that blend seamlessly.
- Per‑channel preservation reveals a **latent structure–color decomposition**: preserving early VAE latent channels preserves image structure, while preserving later channels retains colour distributions (Figure 13). This is a striking, almost unsupervised discovery about the VAE latent space that is directly exploitable for controlled editing.
- The drift compensation (off‑diagonal drift minimisation, Eq. 10) ensures that the generated image remains coherent even when the timestep asynchrony is large (Figure 16).

For structured 3D shapes:
- Assigning high `$t_0$` to part shape codes and bounding boxes while leaving other parts free enables **part‑level completion** (Figure 14) and **assembly** (Figure 15), where the model arranges and refines given parts into a coherent whole. These tasks were previously handled by specialised, task‑specific models; here they are **unified** under a single generative framework.

This innovation is significant because it **recasts the generative model not as a one‑shot sampler, but as a flexible, graded interpolator** between the noise distribution and a reference sample. It transforms the model from a “black‑box” generator into a **controllable, inspectable, and composable tool** that can be used for a spectrum of applications without any additional training. The paper’s simple demonstration that the same model can simultaneously perform high‑quality standard generation (synchronised inference) and these graded operations (asynchronous inference) underscores that **the training and control aspects are two sides of the same combinatorial coin**.