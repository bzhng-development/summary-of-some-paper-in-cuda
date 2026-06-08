# Section 3 — Technical Approach — A/B (the discriminator)

**2405.13729 — ComboStoc: Combinatorial Stochasticity for Diffusion Generative Models**



## PyMuPDF input · THINK-HIGH

## 3. Technical Approach

### 3.1 Reader orientation (approachable technical breakdown)
ComboStoc is a training and inference strategy for diffusion generative models that replaces the usual scalar diffusion timestep with a tensor of independent, asynchronous timesteps, assigning a different “age” to every dimension and attribute of a data sample, so that the model sees and learns from a vastly broader, uniformly covered set of corrupted samples during training. It solves the problem that standard diffusion models (e.g. flow matching) only train on the single diagonal path connecting a source noise point and a target data point, leaving large subregions of the path space under‑explored and causing the model to perform poorly when those regions are encountered at test time.

### 3.2 Big-picture architecture (diagram in words)
1. **Data sample** – for images: a latent tensor `x1` of shape `C × H × W` (patch channels × height × width); for structured 3D shapes: an ensemble of up to `L=256` parts, each described by an existence indicator, a 6‑dimensional bounding box, and a 512‑dimensional shape code.
2. **Noise source** – `z ∼ N(0,1)` of the same shape as the data.
3. **Interpolation schedule tensor** `t` – a tensor of the **same shape** as the data, with every entry independently drawn from `U[0,1]`. This is the core of ComboStoc: the diffusion sample is computed element‑wise as `xt = (1 − t) ⊙ z + t ⊙ x1`.
4. **Network backbone** – a transformer (SiT‑XL/2 for images; a smaller SiT for 3D) that takes the noisy sample `xt`, a class label `c`, and the tensorized timestep `t` and predicts either the velocity `v = x1 − z` (images) or the clean target `x1` (3D shapes). The timestep tensor is encoded via a per‑element sine‑cosine expansion followed by a patch‑wise / part‑wise projection to match the transformer’s hidden dimension.
5. **Training** – the standard regression loss (velocity or `x1` prediction) on the asynchronously generated samples. For images, half of each batch uses asynchronous `t` and half uses a synchronized scalar `t`; for 3D shapes the whole batch is asynchronous.
6. **Asynchronous inference (graded control)** – an optional test‑time mode where the user supplies an initial timestep tensor `t0 ∈ [0,1]^shape` that encodes the “preservation degree” of each dimension/attribute; the denoising integration proceeds with different step sizes per dimension, enabling continuous inpainting, part‑level assembly, etc.

### 3.3 Roadmap for the deep dive
- First we quantify the **sampling bias** of the standard linear interpolant: why the path‑space density shrinks near the data point and how that hurts performance on structured data.
- Second we introduce the **ComboStoc vectorized timestep** – the simple desynchronisation that makes the sampling density uniform inside each source–target hyper‑rectangle.
- Third we prove that the resulting **marginalised vector field still satisfies the continuity equation**, so the training scheme defines a proper generative flow model even though it samples off‑diagonal points.
- Fourth we discuss the **off‑diagonal drift** that appears at test time when starting from an asynchronous point, and the two compensation strategies (off‑diagonal drift minimisation and the cone‑shaped velocity).
- Fifth we describe the **image‑generation adaptation**: how the SiT time‑embedding module is changed to consume a full tensor, the four combinatorial configurations, and the mixed‑batch training.
- Finally we cover the **structured 3D shape adaptation**, where the representation of parts and attributes makes the combinatorial complexity even stronger and the asynchronous treatment is indispensable.

---

### 3.4 Detailed, sentence‑based technical breakdown

**What type of paper this is and the core idea.**  
This paper is both an empirical study and a training‑methodology paper. Its central idea is that standard diffusion generative models (regardless of whether they are DDPM‑style, score‑based, or flow‑matching) implicitly follow a single, synchronised transport path per noise–data pair, thereby under‑sampling the vast combinatorial space that the high‑dimensional dimensions and attributes of the data can span. By replacing the scalar `t` with a tensor `t` of independent, per‑dimension/attribute timesteps, ComboStoc turns the one‑dimensional diagonal path into a full hyper‑rectangle (Fig. 2), making the sampling density uniform and giving the model a much richer, less biased view of the path space.

---

#### 4.4.1 Sampling Bias in Standard Diffusion Models

**The one‑sided linear interpolant.**  
All of the modern diffusion formulations can be unified through the stochastic interpolants framework. The simplest (and often most performant) is the **linear one‑sided interpolant**:

$$
x_t = (1 - t) \, z + t \, x_1, \qquad t \in [0,1]
$$

where `$z \sim \mathcal{N}(0, \mathbf{I})$` is a sample from the source (pure noise) distribution, `$x_1 \sim D$` is a sample from the target data distribution, and `$t$` is a scalar interpolation schedule.

**What this equation does operationally:** for every training step a pair `$(z, x_1)$` is drawn and a scalar `$t$` is sampled uniformly; the model receives the resulting point `$x_t$` that lies **exactly on the straight line** connecting `$z$` and `$x_1$`. The model is then trained to predict the velocity `$x_1 - z$` (or the clean sample) everywhere on that line.

**The sampling density is not uniform.**  
The set of all such lines, weighted by the distributions of `$z$` and `$x_1$`, forms a high‑dimensional path space. The probability that a given point `$x$` is visited during training can be written as the integral of the Gaussian densities that are “attached” to each line. Concretely, the authors model the density of a sample point `$x$` by integrating over `$t$` the Gaussian density `$G_{p_t}$` centred at `$p_t = (1-t)z + t x_1$` with variance scaled by the interpolation coefficient:

$$
\rho(x) = \int_{0}^{1} \frac{1}{\sqrt{2\pi}\,(1-t)} \exp\!\left(-\frac{\|x - p_t\|^2}{2\,(1-t)^2}\right) dt.
$$

Substituting `$p_t = (1-t)z + t x_1$` yields:

$$
\rho(x) = \frac{1}{\sqrt{2\pi}} \int_{0}^{1} \frac{1}{1-t} \exp\!\left(-\frac{\|\,t(z - x_1) + x - z \,\|^2}{2\,(1-t)^2}\right) dt.
$$

**What this integral computes:** the (unnormalised) probability of encountering the point `$x$` anywhere along the continuous set of lines that the diffusion process follows. It is, in effect, the **training‑time sampling density** – the model “sees” different regions of the path space with different frequencies.

**The density shrinks toward data points.**  
By differentiating `$\rho$` with respect to `$x$`, the authors obtain a simple, closed‑form expression for the gradient:

$$
(x_1 - x) \cdot \nabla \rho(x) = \frac{1}{\sqrt{2\pi}} \; e^{-\frac{\|x - z\|^2}{2}} \; > \; 0.
$$

Because the dot product `$(x_1 - x) \cdot \nabla \rho$` is **strictly positive**, the gradient always points in the direction from `$x$` toward `$x_1$`. This means that `$\rho$` **increases monotonically** as one moves from the source noise toward the target data point – the path space is more densely covered near the data samples and becomes sparser farther away.

**Why this form matters:** the gradient in Eq. (4) is a direct, analytical proof that the standard interpolant creates a non‑uniform, shrinking coverage. Without this, one might suspect the problem only empirically; the derivation shows it is a fundamental property of the one‑sided linear scheme that **any** similar interpolant (`$\sigma_t z + \alpha_t x_1$`) will share.

**Consequences for training.**  
Regions far from the target data points are, by construction, less frequently visited during training. When the stochastic differential equation sampler later draws a test‑time trajectory that passes through such a low‑density region, the model has little to no training signal for that part of the space and therefore produces a poor prediction. This is particularly acute for **small datasets** (e.g. 3D shapes with only 18 k samples) and for data with many attributes, where the curse of dimensionality further thins the coverage.

---

#### 3.4.2 The ComboStoc Vectorized Timestep and Uniform Coverage

**Desynchronising the schedule.**  
The key idea of ComboStoc is to **stop treating all dimensions and attributes identically**. Instead of a scalar `$t$`, the interpolation schedule becomes a tensor `$\mathbf{t}$` of the same shape as the data, where each entry is independently sampled from `$\mathcal{U}[0,1]$`. The sample point is then constructed element‑wise:

$$
\mathbf{x}_{\mathbf{t}} = (1 - \mathbf{t}) \odot \mathbf{z} + \mathbf{t} \odot \mathbf{x}_1,
$$

where `$\odot$` denotes elementwise product.

**What this equation computes operationally:** for a given noise source `$\mathbf{z}$` and target `$\mathbf{x}_1$`, each entry (pixel, patch, channel, part attribute) is now noised to a *different* degree. The set of all possible sample points for one `$(\mathbf{z}, \mathbf{x}_1)$` pair is no longer a one‑dimensional line but the whole **hyper‑rectangle** `$R = \{\mathbf{x} \;|\; \mathbf{z} \preceq \mathbf{x} \preceq \mathbf{x}_1\}$`, where the inequality is interpreted element‑wise.

**Why this gives uniform density.**  
By construction, when `$\mathbf{t}$` is a tensor of independent uniforms, every point inside the hyper‑rectangle `$R$` is equally likely to be sampled. There is no “diagonal” concentration; the density is **flat** over the entire combinatorial subspace. This directly addresses the shrinking‑density problem: the model now trains on all on‑diagonal **and** off‑diagonal points, meaning that every combination of per‑dimension/attribute noise levels is well represented.

**Visual intuition.**  
Figure 2 (b) illustrates the difference: the standard model (Fig. 2 a) only visits the diagonal, while ComboStoc (Fig. 2 b) fills the whole rectangle. The resulting sampling density `$\rho(\mathbf{x})$` computed by numerical integration (Fig. 2 d vs. e) shows that the standard flow matching density shrinks toward the target data point, while the ComboStoc density is broad and uniform.

**How this affects the learned velocity field.**  
Because the model now sees points where different dimensions are at very different stages of denoising, it is forced to learn the **correlations among dimensions and attributes** – to “synchronise” them in order to reach the target data point. This is the second, more subtle benefit: the model does not just see more data; it learns a stronger, multi‑faceted signal about how the dimensions must cooperate.

---

#### 3.4.3 Proof of Validity: Marginalised Vector Field Still Generates the Correct Probability Path

**The concern.**  
A natural worry is that if training uses off‑diagonal points that are not on the line `$\mathbf{z} \rightarrow \mathbf{x}_1$`, the learned velocity field might not, when integrated, actually transport the source distribution to the target distribution. The paper provides a formal proof that this is not the case: the vector field obtained by marginalising over all `$(\mathbf{z}, \mathbf{x}_1)$` pairs still satisfies the **continuity equation** and therefore defines a valid generative flow model.

**Conditional probability path and vector field.**  
For each pair `$(\mathbf{x}_0, \mathbf{x}_1)$` (where `$\mathbf{x}_0 = \mathbf{z}$`), the authors define a conditional probability path `$p_t(\mathbf{x} \mid \mathbf{x}_0, \mathbf{x}_1)$` that is a **point mass** moving from `$\mathbf{x}_0$` to `$\mathbf{x}_1$`. The conditional vector field that generates this path is (by construction) the constant velocity:

$$
\mathbf{u}(\mathbf{x} \mid \mathbf{x}_0, \mathbf{x}_1) = \mathbf{x}_1 - \mathbf{x}_0.
$$

**What this vector field does:** it tells any particle that happens to be at `$\mathbf{x}$` (inside the rectangle) how to move in order to eventually reach `$\mathbf{x}_1$`, when the point mass is at the correct time. Note that this is a **time‑invariant** vector field; the time dependence only appears after marginalisation.

**Marginalisation and the continuity equation.**  
The full, marginalised probability path `$p_t(\mathbf{x})$` and its associated vector field `$\mathbf{u}_t(\mathbf{x})$` must obey the continuity equation:

$$
\frac{d}{dt} p_t(\mathbf{x}) = -\operatorname{div}\!\big(\mathbf{u}_t(\mathbf{x}) \, p_t(\mathbf{x})\big).
$$

The authors show that if one expands `$p_t(\mathbf{x})$` as an integral over the joint distribution of `$(\mathbf{x}_0, \mathbf{x}_1)$` and uses the fact that the conditional pair `$(p_t(\mathbf{x} \mid \mathbf{x}_0, \mathbf{x}_1), \mathbf{u}(\mathbf{x} \mid \mathbf{x}_0, \mathbf{x}_1))$` exactly satisfies the continuity equation for each fixed pair, then the **same** equality holds for the marginalised quantities:

$$
\frac{d}{dt} p_t(\mathbf{x}) = \iint \frac{d}{dt} p_t(\mathbf{x} \mid \mathbf{x}_0, \mathbf{x}_1) \, q(\mathbf{x}_0) \, r(\mathbf{x}_1) \, d\mathbf{x}_0 d\mathbf{x}_1
$$
$$
= - \iint \operatorname{div}\!\big( \mathbf{u}(\mathbf{x} \mid \mathbf{x}_0, \mathbf{x}_1) \, p_t(\mathbf{x} \mid \mathbf{x}_0, \mathbf{x}_1) \big) \, q(\mathbf{x}_0) \, r(\mathbf{x}_1) \, d\mathbf{x}_0 d\mathbf{x}_1
$$
$$
= - \operatorname{div}\! \left( \iint \mathbf{u}(\mathbf{x} \mid \mathbf{x}_0, \mathbf{x}_1) \, p_t(\mathbf{x} \mid \mathbf{x}_0, \mathbf{x}_1) \, q(\mathbf{x}_0) \, r(\mathbf{x}_1) \, d\mathbf{x}_0 d\mathbf{x}_1 \right)
$$
$$
= - \operatorname{div}\!\big( \mathbf{u}_t(\mathbf{x}) \, p_t(\mathbf{x}) \big),
$$

where in the first equality we expand probability, in the second we use the conditional continuity property, in the third we swap integration and differentiation (regularity assumption), and in the fourth we apply the definition of the marginalised vector field:

$$
\mathbf{u}_t(\mathbf{x}) = \frac{1}{p_t(\mathbf{x})} \iint \mathbf{u}(\mathbf{x} \mid \mathbf{x}_0, \mathbf{x}_1) \, p_t(\mathbf{x} \mid \mathbf{x}_0, \mathbf{x}_1) \, q(\mathbf{x}_0) \, r(\mathbf{x}_1) \, d\mathbf{x}_0 d\mathbf{x}_1.
$$

**What this chain of equalities says operationally:**  
- The first term `$\frac{d}{dt} p_t(\mathbf{x})$` is the time‑derivative of the marginal probability that a point is at `$\mathbf{x}$`.  
- By writing it as the integral over all `$(\mathbf{x}_0, \mathbf{x}_1)$` of the conditional time‑derivative, we see that **if each conditional pair obeys the continuity equation, the marginal does too**.  
- The conditional pair obeys it trivially because `$p_t(\mathbf{x} \mid \mathbf{x}_0, \mathbf{x}_1)$` is a deterministic path (Dirac delta moving from `$\mathbf{x}_0$` to `$\mathbf{x}_1$`) and `$\mathbf{u}(\mathbf{x} \mid \mathbf{x}_0, \mathbf{x}_1)$` is the constant velocity that transports it.  
- Therefore the marginal `$\mathbf{u}_t(\mathbf{x})$`, defined by the weighted average in Eq. (7), **must** generate the correct `$p_t$`.

**Why this form matters:** the proof is **not** that we can arbitrarily sample off‑diagonal points and still use the same loss. It is that the **same prediction target** `$\mathbf{x}_1 - \mathbf{z}$` (velocity) or `$\mathbf{x}_1$` (target), when learned on the off‑diagonal points, still produces a vector field whose marginal is consistent with the desired transport. The only difference is that the training data now covers a much larger, uniformly dense set of points, giving the model a more comprehensive, less biased view of the path space. This is the theoretical justification for why the simple desynchronisation is both **valid** and **beneficial**.

---

#### 3.4.4 Off‑Diagonal Drift and Its Mitigation During Test‑Time Integration

**Why off‑diagonal drift occurs.**  
Even though the marginalised vector field is correct, during test‑time integration one often starts from a **specific** off‑diagonal point `$\mathbf{x}_{t_0}$` where different dimensions have different initial timesteps (e.g. one half of an image is at `$t=0.9$` while the other half is at `$t=0$`). If one simply integrates the **local** velocity `$\mathbf{x}_1 - \mathbf{z}$` (which is constant and points from `$\mathbf{z}$` to `$\mathbf{x}_1$`) without any correction, the final point will miss `$\mathbf{x}_1$` by an offset equal to:

$$
\mathbf{x}_{t_0} + \int_{t_0}^{1} (\mathbf{x}_1 - \mathbf{z}) \, dt = \mathbf{z} + \mathbf{t}_0 \odot (\mathbf{x}_1 - \mathbf{z}) + (1 - t_0) (\mathbf{x}_1 - \mathbf{z}) 
= \mathbf{x}_1 + (t_0 - \bar{t}_0) \odot (\mathbf{x}_1 - \mathbf{z}),
$$

where `$\bar{t}_0 = \min(\mathbf{t}_0)$` is the smallest timestep among the dimensions. The residual vector `$(\mathbf{t}_0 - \bar{t}_0) \odot (\mathbf{x}_1 - \mathbf{z})$` is exactly the **off‑diagonal drift** – it is pure noise that has not been fully denoised because the faster dimensions have already completed their trajectory while the slower ones are still on the way.

**Off‑diagonal drift minimisation.**  
To pull the trajectory back toward the diagonal, the authors propose a **compensation velocity** `$\mathbf{v}_{\text{cmpn}}$` defined as the negative gradient of a drift potential. The off‑diagonal offset vector is first computed as the component orthogonal to the diagonal direction:

$$
\boldsymbol{\delta}(\mathbf{x}_t) = \mathbf{x}_t - \mathbf{x}_1 - \frac{(\mathbf{x}_t - \mathbf{x}_1) \cdot (\mathbf{x}_1 - \mathbf{z})}{\|\mathbf{x}_1 - \mathbf{z}\|^2} \, (\mathbf{x}_1 - \mathbf{z}),
$$

$$
\mathbf{v}_{\text{cmpn}} = -\boldsymbol{\delta}(\mathbf{x}_t).
$$

**What this does:** `$\boldsymbol{\delta}$` measures how far the current point `$\mathbf{x}_t$` has strayed from the ideal diagonal line connecting `$\mathbf{z}$` and `$\mathbf{x}_1$`. The compensation velocity `$\mathbf{v}_{\text{cmpn}}$` is exactly the negative of that offset, so following it (in addition to the nominal velocity) **pushes the trajectory back onto the diagonal**. This is equivalent to performing gradient descent on the potential `$\Phi(\boldsymbol{\delta}) = \tfrac{1}{2} \|\boldsymbol{\delta}\|^2$`, which is why the method is called “off‑diagonal drift minimisation”.

**Why this form:** the diagonal direction is `$\mathbf{x}_1 - \mathbf{z}$`; the orthogonal projection ensures that only the off‑diagonal component is penalised, and the compensation does not interfere with the correct in‑line motion. The alternative of simply adding a constant that pulls toward `$\mathbf{x}_1$` would not be as targeted.

**Cone‑shaped velocity field.**  
As a second, more intuitive but less regular option, the authors also consider a **cone‑shaped velocity**:

$$
\mathbf{v}_{t_0} = \frac{\mathbf{x}_1 - \mathbf{x}_{t_0}}{1 - \bar{t}_0},
$$

where again `$\bar{t}_0 = \min(\mathbf{t}_0)$`.

**What this computes:** if one uses this velocity for the entire integration (i.e. the model predicts this and the integration follows it), then starting from `$\mathbf{x}_{t_0}$` and integrating with step sizes adjusted to the remaining time of the slowest dimension, the final point is **guaranteed** to be `$\mathbf{x}_1$`:

$$
\mathbf{x}_{t_0} + \int_{t_0}^{1} \frac{\mathbf{x}_1 - \mathbf{x}_{t_0}}{1 - \bar{t}_0} \, dt = \mathbf{x}_{t_0} + \mathbf{x}_1 - \mathbf{x}_{t_0} = \mathbf{x}_1.
$$

**Why this is inferior in practice:** the cone velocity normalises by `$1 - \bar{t}_0$`, which can be very small if the slowest dimension is close to 1, resulting in **huge** velocity magnitudes and numerical instability. The off‑diagonal drift minimisation, by contrast, only adds a small correction proportional to the offset, leading to much smoother integration.

**Ablation evidence (Tab. 6, Fig. 16).**  
The paper compares the three strategies – no compensation, off‑diagonal drift minimisation, and the cone‑shaped velocity – on a 50 k‑step training run using the first 100 ImageNet categories. The off‑diagonal drift minimisation achieves the best FID (`103.01` vs. `103.75` without compensation) and the highest SSIM (`0.262` vs. `0.255`) when generating images with a preservation weight `$\lambda = 0.75$` on one half. The cone‑shaped velocity actually **degrades** performance (FID `113.59`, SSIM `0.224`), because its large magnitude breaks the smoothness of the learned field. The visual comparison (Fig. 16) shows that without compensation a clear discontinuity appears along the midline between differently preserved halves, whereas the off‑diagonal drift minimisation produces a seamless transition.

**How this is used in the whole pipeline:** for all the main experiments (image generation FIDs, 3D shape generation, inpainting, etc.) the off‑diagonal drift minimisation is **applied at test time** when asynchronous starting points are used. During standard synchronous training and inference, no compensation is needed because all dimensions share the same scalar `$t$` and the points are always on the diagonal.

---

#### 3.4.5 Adapting ComboStoc to Image Generation

**Baseline: SiT‑XL/2.**  
The image experiments build on the **SiT** model (Scalable Interpolant Transformers), a state‑of‑the‑art transformer‑based diffusion model that encodes images via a VAE into a latent tensor `$x_1$` of shape `$(C, H, W)$` and uses the standard ViT architecture with adaptive layer‑norm modulation. The model is trained to predict the velocity `$x_1 - z$` given the diffused latent `$x_t$`, the class label `$c$`, and the scalar timestep `$t$`.

**Tensorising the timestep.**  
In ComboStoc, the scalar `$t$` is replaced by a tensor `$\mathbf{t}$` of shape `$(C, H, W)$` (matching the latent image), where each entry is an independent random draw from `$\mathcal{U}[0,1]$`. To feed this tensor to the transformer, the **time‑embedding module** is redesigned (Fig. 5):

1. **Per‑entry sine‑cosine encoding.** Each scalar value in `$\mathbf{t}$` is first mapped to a 4‑dimensional frequency code vector using the standard sine‑cosine transform (as in Vaswani et al.). This produces a tensor of shape `$(N, C, H, W, 4)$`.

2. **Patch‑wise projection.** To turn these 4‑D per‑pixel codes into a sequence of tokens matching the image patches, the authors **transpose** the channel and code dimensions to treat the 4‑D vector as a “feature” per pixel, then apply the **same patch‑embedding layer** as the ViT encoder (but with separate parameters) – a linear layer that maps small spatial patches (`$2 \times 2$`) plus the 4 code dimensions into the transformer’s hidden dimension `$C_H = 1152$`. With a patch size of `$p$`, this produces `$T = H \times W / p^2$` timestep tokens.

3. **Integration into the SiT block.** These timestep tokens are then added to the image patch tokens and combined with the class label `$c$` through the standard adaptive‑layer‑norm modulation operators, exactly as in SiT.

**Why `$C_C = 4$` is used:** the 4‑dimensional code is deliberately very small compared to the hidden dimension `$C_H = 1152$` to avoid blowing up the embedding module. This design choice is responsible for the slight parameter count reduction (`$\approx 673$` M vs. `$675$` M) and the slightly slower per‑step training speed (Sec. 5.5).

**Combinatorial configurations.**  
The paper enumerates four configurations that progressively unlock more combinatorial complexity (Tab. 3a):

- **unsync_none:** a scalar `$t$` broadcast to all dimensions – effectively the baseline SiT.
- **unsync_patch:** `$\mathbf{t}$` is `$(N, 1, H, W)$` – different timesteps per spatial pixel, but all channels of a pixel share the same value.
- **unsync_vec:** `$\mathbf{t}$` is `$(N, C, 1, 1)$` – different timesteps per feature channel, but all spatial pixels of a channel share the same value.
- **unsync_all:** `$\mathbf{t}$` is `$(N, C, H, W)$` – every spatial pixel **and** every channel gets an independent timestep; this is the full ComboStoc.

The results (Fig. 6b) show that **unsync_all consistently achieves the lowest FID**, followed by unsync_vec and unsync_patch (which are nearly indistinguishable), with unsync_none trailing far behind. This demonstrates that fully exploiting the combinatorial richness of both spatial and channel axes pays off.

**Mixed batch training.**  
For ImageNet (1.28 M images), the authors found that fully asynchronous training on every sample can be slightly too difficult early on. They therefore apply the asynchronous timestep to **only half the samples in each batch**; the other half receive the standard synchronised scalar `$t$`. This mixed strategy acts as a curriculum, ensuring that the model still sees many “easy” on‑diagonal points while also being exposed to the off‑diagonal ones. For the much smaller PartNet dataset (18 k shapes), no mixing is needed – all samples use asynchronous `$\mathbf{t}$`.

---

#### 3.4.6 Adapting ComboStoc to Structured 3D Shape Generation

**Representation.**  
A structured 3D shape is represented as a flat collection of **leaf‑level semantic parts** (no hierarchy is used). Specifically, `$\mathbf{x} = \{ \mathbf{p}_i \}, i \in [1, L]$` with `$L = 256$` covering the maximum number of parts. Each part `$\mathbf{p}$` is a tuple:

$$
\mathbf{p} = (s, \mathbf{b}, \mathbf{e}),
$$

where `$s \in [0,1]$` is an existence indicator, `$\mathbf{b} = (x, y, z, l, w, h)$` is a 6‑dimensional bounding box (centre and extents), and `$\mathbf{e} \in \mathbb{R}^{512}$` is a latent shape code encoding the part’s geometry in normalised coordinates.

**Why this representation is particularly challenging for standard diffusion:** the data is a concatenated tensor of shape `$(N, L, V_{\text{cat}})$` where `$V_{\text{cat}} = 1 + 6 + 512 = 519$` elements. These 519 elements are not homogeneous; the first is a Bernoulli score, the next six are continuous and often small, and the last 512 are a high‑dimensional, highly structured latent code. The combinatorial complexity is enormous, and the standard synchronised interpolant fails to cover the space adequately.

**Tensorised timestep for parts and attributes.**  
ComboStoc constructs a timestep tensor `$\mathbf{t}$` that matches the shape of the part‑feature tensor, using broadcast semantics to control which dimensions are synchronised. The six configurations (Tab. 3b) are:

- **unsync_none:** `$\mathbf{t}$` is shape `$(N, 1, 1)$` – all parts share the same scalar `$t$`.
- **unsync_part:** `$\mathbf{t}$` is `$(N, L, 1)$` – different timesteps per part, but within a part all attributes share the same value.
- **unsync_att:** `$\mathbf{t}$` is `$(N, 1, [1,1,1])$` – different timesteps for the **categories** of attributes (existence, bounding box, shape code) but all parts share the same per‑category value.
- **unsync_att_part:** `$\mathbf{t}$` is `$(N, L, [1,1,1])$` – different timesteps per part **and** per attribute category.
- **unsync_vec:** `$\mathbf{t}$` is `$(N, 1, V_{\text{cat}})$` – different timesteps for every individual scalar in the concatenated feature vector, but all parts share the same per‑element values.
- **unsync_all:** `$\mathbf{t}$` is `$(N, L, V_{\text{cat}})$` – every part **and** every scalar gets an independent timestep; the fully asynchronous setting.

**Embedding of timesteps for each attribute type.**  
The sine‑cosine encoding is applied to each scalar, producing a 4‑D code. Then, for each of the three attribute blocks (existence, bbox, shape code), a separate learnable fully‑connected layer projects the 4‑D code to the transformer’s hidden dimension `$C_H = 384$`. The resulting embeddings are then summed with the part‑token embeddings before the self‑attention blocks.

**Training and inference specifics.**  
- The network is a **SiT‑small** architecture (12 layers, 384 hidden dim, 256 tokens for parts, 6 attention heads).  
- **`$x$`‑prediction** is used instead of velocity prediction because the heterogeneous representation (binary existence, continuous bbox, and high‑dimensional latent code) makes `$x$`‑prediction more robust.  
- All training samples use the fully asynchronous scheme (no mixing).  
- During **standard generation**, the timesteps are synchronised (Algorithm 1(A)); during **graded control**, the asynchronous starting timesteps `$\mathbf{t}_0$` are specified by the user (Algorithm 1(B)).

**Why this is indispensable for 3D shapes.**  
Fig. 8 and Tab. 4 show that **unsync_none almost completely fails** to produce meaningful shapes, while progressively richer combinatorics (unsync_part, unsync_att, unsync_att_part, etc.) yield dramatic improvements. The part‑level synchronisation (`$L$` dimension) is particularly important, as it teaches the model to reason about the **arrangement and correlation of parts**. The full `unsync_all` setting achieves the best FPD, COV, and MMD, and in Tab. 5 it matches or is competitive with hierarchy‑based baselines without using any tree structure.


## GLM-OCR input · THINK-HIGH

## 3. Technical Approach

### 3.1 Reader Orientation

ComboStoc is a training and inference framework for diffusion generative models that replaces the standard globally synchronised noise schedule with a per-dimension, per-attribute *asynchronous* schedule, implemented by converting the scalar interpolation time `$t$` into a tensor of the same shape as the data sample. It solves the problem that standard one-sided stochastic interpolants (and related flow/score formulations) produce a **non-uniform sampling density** along the diagonal path connecting a pure noise source to a target data point — regions farther from the target are under-trained, and when encountered at test time the model can generate poor results. The solution is remarkably simple: **sample the whole combinatorial subspace** uniformly by independently drawing a `$[0,1]$` time value for every entry of the data tensor, exposing the model to all possible asynchronous mixtures of noise and signal during training, which in turn forces the network to learn the correlations among dimensions and attributes.

### 3.2 Big-Picture Architecture

The system consists of five major pieces, all built on top of a scalable transformer diffusion backbone (SiT [Ma et al. 2024]):

1. **A vectorised timestep schedule** — a tensor of shape `(N, C, H, W)` (images) or per-part stacks (3D shapes) whose entries are independently drawn from `$[0,1]$`; this replaces the single scalar `$t$` of standard interpolants.
2. **A modified timestep embedding module** — the transformer receives this tensor as conditioning, embedding each entry via a small frequency/MLP block and then re-patchifying it analogous to the image tokens so that each spatial/attribute location sees its own timestep.
3. **The base stochastic interpolant model** — a transformer that predicts either the velocity `$x_1 - z$` (images) or the clean target `$x_1$` (3D shapes) from the elementwise-mixed intermediate sample `$x_t = (1 - t) \odot z + t \odot x_1$`.
4. **A compensation mechanism for off-diagonal drift** — an extra term (or an alternative velocity field) that corrects the integration path when test-time sampling stumbles onto points that are not on the diagonal, ensuring convergence to the target.
5. **Dual inference modes** — (A) *synchronised inference* for standard generation, where all dimensions use the same scalar step, and (B) *asynchronous inference* for graded control, where a user-provided preservation mask `$\mathbf{m}$` sets the initial timestep per dimension and the model generates with different degrees of finalisation in different parts.

Information flows as: a data sample `$x_1$` and a source noise sample `$z$` enter → the vectorised timestep tensor `$t$` is sampled → an intermediate `$x_t$` is constructed via elementwise interpolation → the network `$f_\theta(x_t; c, t)$` predicts either the velocity or the target → during training, a compensation loss or a modified prediction target may be used to handle off-diagonal samples; during inference, either a synchronised or an asynchronous integration loop runs.

### 3.3 Roadmap for the Deep Dive

- **First**, the core **combinatorial stochastic process** — what it means to vectorise the timestep, why it produces uniform sampling density where the standard interpolant does not, and the elementwise interpolation Equation (5) that defines the process.
- **Second**, the **theoretical justification** that this process still defines a proper generative flow model (the continuity equation proof), so that the model is not simply a heuristic.
- **Third**, the **off-diagonal drift problem** that arises specifically for velocity-prediction models when trained with vectorised timesteps, and the two families of compensation (off-diagonal drift minimisation and cone-shaped velocity) that fix it.
- **Fourth**, the **architectural modifications** to the SiT transformer — how the scalar timestep embedding is upgraded to a patch-wise embedding of a full tensor without introducing a parameter explosion.
- **Fifth**, the **training configuration space** — the different levels of combinatorial flexibility (`unsync_none`, `unsync_patch`, `unsync_vec`, `unsync_all` for images; analogous splits for 3D structured shapes) and how they relate to the degrees of structure in the data.
- **Sixth**, the **inference procedures** — the synchronised mode for standard sampling and the asynchronous mode that enables graded control across patches, parts, and attributes.

### 3.4 Detailed, Sentence-Based Technical Breakdown

This is primarily a **training-dynamics paper** whose core idea is that by desynchronising the diffusion time schedule across all dimensions and attributes of a data point, one can uniformly cover the combinatorial path space, eliminating a sampling bias that limits both convergence speed (for large image datasets) and basic model viability (for small structured 3D datasets).

---

#### 3.4.1 The Combinatorial Stochastic Process: From Scalar to Vectorised Timesteps

The standard linear one-sided stochastic interpolant [Albergo et al. 2023] — which underlies both flow matching and many diffusion formulations — defines a trajectory from a source noise sample `$z \sim \mathcal{N}(0,1)$` to a data sample `$x_1 \sim \mathcal{D}$` via

$$x_t = (1 - t) z + t x_1, \quad t \in [0,1]$$

where `$t$` is a scalar interpolation schedule shared by **every dimension** of `$z$` and `$x_1$`.

> **What it computes:** a single point on the straight line connecting `$z$` and `$x_1$`. As `$t$` sweeps from 0 to 1, the point moves from pure noise to pure data. The training objective (regressing the velocity `$\partial_t x_t = x_1 - z$` or the clean sample) is applied to all such points uniformly in `$t$`.
>
> **Why this form:** linear interpolation is simple, yields a constant velocity field that is easy to learn, and — in the stochastic interpolants framework — is one member of a broad family of paths `$x_t = \sigma_t z + \alpha_t x_1$`, all of which share the same diagonal structure.

The problem, as the paper analyses, is that **the sampling density along this diagonal is not uniform**. Conditioning on a specific data point `$x_1$` and a source noise `$z$`, the density of points `$x$` in the rectangular subspace `$\mathcal{R} = \{x \mid z \preceq x \preceq x_1\}$` visited during training is

$$\rho(x) = \int_0^1 G_{p_t}(x) \, dt = \frac{1}{\sqrt{2\pi}} \int_0^1 \frac{1}{1-t} e^{-\frac{\|t(z - x_1) + x - z\|^2}{2(1-t)^2}} dt$$

where `$G_{p_t} = \mathcal{N}(p_t; (1-t)^2 \mathbf{1})$` is the Gaussian centred at the diagonal point `$p_t = (1-t)z + t x_1$` with variance shrinking as `$t$` approaches 1.

> **What it computes:** the marginal probability density of a point `$x$` being sampled during training, integrated over all possible `$t$`. The shrinking variance `$(1-t)^2$` means that points near the noise end (`$t \approx 0$`) are sampled under a wide Gaussian, while points near the data end (`$t \approx 1$`) are sampled under a very tight Gaussian.
>
> **Why this matters:** the gradient of this density has a positive projection along `$x_1 - x$` (the direction from any `$x$` toward the data point), as shown by
> $$(x_1 - x) \cdot \nabla \rho(x) = \frac{1}{\sqrt{2\pi}} e^{-\frac{\|x - z\|^2}{2}} > 0,$$
> meaning **density grows monotonically as one approaches the data point**. Regions far from the diagonal — points where different dimensions have different `$t$` values — are under-sampled, and when the stochastic integrator encounters them at test time the model may produce unreliable outputs.

**The ComboStoc fix** is to **break the synchronisation** and turn the scalar `$t$` into a tensor of the same shape as `$x$`, with each entry independently sampled uniformly from `$[0,1]$`:

$$x_t = (1 - \mathbf{t}) \odot z + \mathbf{t} \odot x_1$$

where `$\mathbf{t} \in [0,1]^{\text{shape}(x)}$` is the vectorised timestep tensor, and `$\odot$` denotes elementwise (Hadamard) product.

> **What it computes:** a point in the rectangular subspace `$\mathcal{R}$` where different dimensions (e.g., different image patches, different feature channels, different shape attributes) are at **different** stages of the noise-to-data transition. For example, one patch may be almost pure noise (`$t_i \approx 0$`) while another is nearly the final data (`$t_j \approx 0.9$`).
>
> **Why this form:** by independently choosing each `$t$`-entry, the training distribution becomes **uniform over the whole rectangular subspace** spanned by each `$(z, x_1)$` pair (by construction), instead of being concentrated along the diagonal. This forces the network to learn how to *synchronise* the different dimensions — i.e., to pull them together toward a coherent data point — rather than simply following a fixed diagonal path. The sampling density is now constant within the subregion, eliminating the shrinking-coverage bias illustrated in Fig. 2(d) of the paper.

The process is visualised in Fig. 2(b): instead of the single diagonal line of Fig. 2(a), the training now covers the entire 2D rectangle for each pair of dimensions, with each point sampled via its own independent `$t$`-value.

---

#### 3.4.2 Theoretical Justification: The ComboStoc Process as a Proper Flow Model

A critical question is whether training on these *off-diagonal* samples still defines a valid generative model — i.e., whether the marginalised velocity field actually generates the correct probability path from the source distribution to the target distribution. The paper provides a derivation (Equations (6)–(7)) that shows the ComboStoc scheme fits within the general framework of conditional flow matching, simply by replacing the `$x_1$`-conditioned vector field with an `$(x_0, x_1)$`-conditioned one.

Specifically, they define a conditional vector field `$u(x \mid x_0, x_1)$` that holds whenever `$x \in \text{span}(x_0, x_1)$` — the full rectangular subspace — and show that the marginalised velocity

$$u_t(x) = \frac{1}{p_t(x)} \iint u(x \mid x_0, x_1) \, p_t(x \mid x_0, x_1) \, q(x_0) \, r(x_1) \, dx_0 \, dx_1$$

satisfies the continuity equation

$$\frac{d}{dt} p_t(x) = - \text{div}\left( u_t(x) \, p_t(x) \right)$$

where `$q(x_0) = p_0$` is the source (noise) distribution, `$r(x_1) = p_1$` is the target (data) distribution, and `$p_t(x \mid x_0, x_1)$` is the probability path of a point distribution moving from `$x_0$` to `$x_1$`.

> **What these equations compute:** The first equality (derived via expanding the time derivative of `$p_t(x)$` into an integral over `$x_0, x_1$`, applying the fact that `$u(x \mid x_0, x_1)$` generates `$p_t(x \mid x_0, x_1)$`, and swapping differentiation with integration) shows that **the aggregate velocity field `$u_t(x)$` is exactly the one that transports the marginal probability `$p_t(x)$` according to the continuity equation**. In other words, the model trained on ComboStoc samples, when integrated at test time, will correctly move samples from the noise distribution to the data distribution.
>
> **Why this form is necessary:** it closes the gap between the *training* procedure (which samples from the `$\text{span}(x_0, x_1)$` subspace) and the *standard test-time integration* (which operates with a scalar time variable). The apparent time dependence in the marginalised `$u_t(x)$` emerges from the integration over `$x_0, x_1$`, but the underlying conditional field `$u(x \mid x_0, x_1)$` is time-invariant — it simply moves a point from `$x_0$` to `$x_1$` by construction. This matches the Flow Matching recipe, with the only difference being that the conditional support is the rectangular span rather than the diagonal.

The paper emphasises that the issue of off-diagonal drift (Fig. 4) is **not** a violation of this theoretical guarantee; rather, it is a *practical integration problem* that occurs when the test-time solver steps outside the diagonal and the model's predicted velocity does not alone steer the point back to the target. This is addressed in the next section.

---

#### 3.4.3 Addressing Off-Diagonal Drift: Velocity Compensation for Training and Inference

For **velocity prediction** (`$v$`-prediction, where the network outputs `$x_1 - z$`), the ComboStoc training generates samples with asynchronous `$\mathbf{t}$`. If at test time one simply integrates the predicted constant velocity `$x_1 - z$` from such an off-diagonal point, the trajectory will **miss the target data point**:

$$\mathbf{x}_{t_0} + \int_{t_0}^{1} (x_1 - z) \, dt = x_1 + \left( \mathbf{t}_0 - \min(\mathbf{t}_0) \right) \odot (x_1 - z)$$

where `$t_0 = \min(\mathbf{t}_0)$` is the scalar minimum across the dimensions, and the integral is assumed to run until all dimensions reach 1 (the slowest one finishes). The residual term `$(\mathbf{t}_0 - \min(\mathbf{t}_0)) \odot (x_1 - z)$` is an offset that leaves the point off the target.

> **What it computes:** the error when blindly following the constant diagonal velocity from an off-diagonal starting point. Dimensions that were *ahead* (larger `$t$`) would overshoot if the integration used their individual times; using the minimum time ensures no overshoot but leaves a gap equal to the difference in `$t$` scaled by the velocity.
>
> **Why this happens:** the standard velocity `$x_1 - z$` is only the correct direction when all dimensions are at the same `$t$`. An off-diagonal point requires a **corrective component** that pushes it back toward the diagonal before or during the final approach to the target.

The paper introduces two strategies to fix this (Eq. (10)–(12)):

**Off-diagonal Drift Minimisation.** The offset vector `$\delta(\mathbf{x}_t)$` (the per-dimension deviation from the diagonal) is defined as

$$\delta(\mathbf{x}_t) = \mathbf{x}_t - x_1 - \frac{(\mathbf{x}_t - x_1) \cdot (x_1 - z)}{\|x_1 - z\|^2} (x_1 - z)$$

and the compensation velocity is simply its negation:

$$v_{\text{cmpn}} = -\delta(\mathbf{x}_t)$$

> **What it computes:** `$\delta$` projects the current point onto the direction `$(x_1 - z)$` and subtracts; what remains is the component **orthogonal** to the target direction — the off-diagonal drift. The compensation velocity `$-\delta$` is the direction that **pushes the point back onto the diagonal**. Adding this to the original velocity during integration (equivalent to gradient descent on a drift potential `$\Phi(\delta) = \frac{1}{2}\|\delta\|^2$`) pulls the trajectory toward the data point.
>
> **Why this form:** it directly follows from the geometry of the problem — the orthogonal projection of the off-diagonal error onto the line `$z \to x_1$` is the part that needs to be eliminated. Minimising the squared norm of this error is a natural, smooth, and convex objective that does not require rescaling the velocity by (potentially very small) time differences.

**Cone-shaped Velocity Field.** An alternative is to re-define the velocity at off-diagonal points as

$$v_{t_0} = \frac{x_1 - \mathbf{x}_{t_0}}{1 - \min(\mathbf{t}_0)}$$

where `$\min(\mathbf{t}_0)$` is the smallest entry in the tensor `$\mathbf{t}_0$`. This generalises the constant velocity: at synchronised points it recovers `$x_1 - z$`, and at off-diagonal points it points **directly toward the target** with a magnitude scaled by the inverse of the remaining time for the slowest dimension.

> **What it computes:** a velocity field that forms a "cone" — all points share the same direction toward `$x_1$`, but with magnitude inversely proportional to the time remaining for the slowest dimension.
>
> **Why this form:** the integration
> $$\mathbf{x}_{t_0} + \int_{t_0}^{1} \frac{x_1 - \mathbf{x}_{t_0}}{1 - t_0} \, dt = x_1$$
> exactly reaches the target because the velocity is constant along the line connecting the off-diagonal point to the target. However, the scaling by `$1/(1 - \min(\mathbf{t}_0))$` can be numerically violent when one dimension is very close to 1 (making the denominator tiny), which degrades training stability.

The paper adopts the **off-diagonal drift minimisation** approach for all experiments, citing better FID and visual seamlessness in graded generation (Tab. 6, Fig. 16). The compensation is **applied only at inference** (the training target remains the standard `$x_1 - z$` or `$x_1$`), because the model is already trained on the off-diagonal samples and learns to expect them; the compensation corrects the test-time integrator.

---

#### 3.4.4 Model Architecture Modifications for Tensorised Timesteps

**Images (SiT backbone).** The baseline SiT-XL/2 transformer receives a scalar timestep `$t$`, encodes it into a frequency vector of length `$C_F$` (sine/cosine embedding), passes it through an MLP to produce a conditioning vector of channel dimension `$C_H$` (the hidden size, 1152), and then **modulates** the transformer blocks via an adaptive layer norm (adaLN) mechanism. The conditioning is a single vector shared by all tokens.

For ComboStoc (Fig. 5), the timestep becomes a tensor `$\mathbf{t}$` of shape `$(N, C, H, W)$`. The embedding module is modified in two stages:

1. **Per-entry frequency encoding** — the same positional encoding (sine/cosine) is applied *elementwise* to every entry of `$\mathbf{t}$`, producing a tensor of shape `$(N, C, H, W, C_F)$`. Then a small MLP (with output dimension `$C_C = 4$`, much smaller than `$C_H$`) compresses each entry into a 4-dimensional vector. This yields a tensor of shape `$(N, C, H, W, 4)$`.
2. **Patch-wise re-embedding** — the channel and spatial dimensions are reshaped and then processed by a **patch embedding layer** (identical to the ViT image patch embedding, but with separate parameters) that treats the 4-channel timestep map as an image and produces a sequence of tokens of dimension `$C_H = 1152$`, one per patch (patch size `$2 \times 2$`). The resulting token sequence is then used as the conditioning signal in the transformer's adaLN modulation, so that each patch sees the timestep of its own spatial location and its own feature channels.

> **Why this design:** using a small per-entry encoding (dimension 4) keeps the timestep embedding compact, avoiding a blow-up in parameters. The patch-wise re-embedding re-uses the same inductive bias as the image tokenisation — the model already understands spatial and channel structure. This is why `unsync_none` (which uses the same architecture but with a scalar timestep broadcast to all entries) has slightly worse performance than baseline SiT: the smaller embedding dimension (4 vs. `$C_H$`-sized) is a bottleneck that the tensorised version overcomes by re-patchifying.

**Structured 3D shapes.** A structured 3D object is represented as a collection of up to `$L = 256$` part slots, each encoded as `$p = (s, b, e)$` where:
- `$s \in [0,1]$` — part existence indicator,
- `$b = (x, y, z, l, w, h)$` — bounding box centre and sizes,
- `$e \in \mathbb{R}^{512}$` — latent shape code (from Wang et al. [2025]'s point cloud VAE).

These attributes are of different nature and dimensionality. The timestep embedding for each scalar attribute (e.g., existence `$s$`, each bounding box coordinate) follows the same two-stage scheme as images: frequency-encode each scalar to a 4-dimensional vector, then apply a **part-level** FC layer to embed the collection of attributes for a part into the hidden dimension (384 for the SiT-small model). The latent shape code `$e$` and the bounding box vector are also separately embedded via their own FC layers, and the per-part outputs are summed to form the final conditioning token.

The key additional combinatorial axes here are:
- **Parts** — different part slots can receive different timesteps.
- **Attributes** — existence, bounding box, and shape code can be at different `$t$` values within the same part.
- **Feature vector dimensions** — within a shape code, different coordinates can be desynchronised.

This gives the `$3 \times 2 = 6$` configurations enumerated in Tab. 2(b).

---

#### 3.4.5 Training Configurations and Combinatorial Flexibility Levels

The paper systematically probes how much desynchronisation helps, by defining **levels of combinatorial flexibility** (Tab. 2):

**For images (latent shape `$C \times H \times W$`):**
- `unsync_none` — `$t$` is a scalar, broadcast. This is the baseline close to standard SiT (but with the modified, slightly smaller embedding, so it underperforms SiT slightly).
- `unsync_patch` — `$t$` varies across spatial locations (`$H \times W$` independent values) but is shared across channels.
- `unsync_vec` — `$t$` varies across feature channels (`$C$` independent values) but is shared across space.
- `unsync_all` — `$t$` varies across both channels and space — fully elementwise independent.

**For structured 3D shapes (part slots with attributes):**
- `unsync_none` — all parts and attributes share one `$t$`.
- `unsync_part` — each part slot has its own `$t$` (all attributes within a part are synchronised).
- `unsync_att` — within each part, the existence `$s$`, bounding box `$b$`, and shape code `$e$` get independent `$t$` values, but all parts share the same per-attribute schedule.
- `unsync_att_part` — both attributes are desynchronised **and** different parts get different `$t$`.
- `unsync_vec` — within the shape code `$e$`, individual feature dimensions are desynchronised (but parts and other attributes share schedules).
- `unsync_all` — all three axes (part, attribute, feature dimension) are fully independent — the maximal combinatorial split.

**Training protocol.** For ImageNet (1.3M images), the paper applies a **mixed strategy**: in each batch, half the samples use the fully asynchronous `$\mathbf{t}$` (the chosen configuration) and the other half use synchronised `$t$` (all entries equal). This acts as a curriculum — the synchronised samples prevent early training instability, and the asynchronous samples force the model to learn inter-dimension correlations. For PartNet (18K shapes), the dataset is so small and the structure so strong that **no mixing is needed** — all samples in every batch are asynchronous.

**Why this matters:** The mixed strategy on large datasets is a design choice that balances between learning from uniform subspace coverage (which is harder early on) and from the simpler diagonal paths (which stabilise training). For small datasets, the uniform coverage is essential to avoid overfitting to the diagonal; without it, the baseline (`unsync_none`) entirely fails to produce valid manifold shapes (Fig. 8).

---

#### 3.4.6 Inference: Synchronous Generation and Asynchronous Graded Control

The paper provides two inference procedures (Alg. 1), both using the standard SDE integrator (250 steps for images, 500 for 3D shapes).

**Synchronised inference (standard generation).** All dimensions share a single scalar timestep schedule `$\{t_k = k/K\}_{k=0}^K$`, producing a tensor `$\mathbf{t}^{(k)} = t_k \cdot \mathbf{1}$`. The model is called with this synchronised time tensor, and the integration follows the standard SiT step. This mode is used for all FID/quality evaluations in Sec. 5.2, confirming that the performance gains come purely from the asynchronous **training**, not from any test-time trick.

**Asynchronous inference (graded control).** A user provides a **mask** `$\mathbf{m} \in [0,1]^{\text{shape}(\mathbf{x})}$` and a reference sample `$\mathbf{x}_1$`. The process initialises the state as
$$\mathbf{x}^{(0)} = (1 - \mathbf{m}) \odot \mathbf{z} + \mathbf{m} \odot \mathbf{x}_1$$
and the timestep tensor as `$\mathbf{t}^{(0)} = \mathbf{m}$`. The step size per dimension is `$\Delta t = \frac{1 - \mathbf{m}}{K}$` (uniform step number across dimensions; the paper also tests uniform step size in an ablation). At each iteration, the model receives the fully asynchronous time field `$\mathbf{t}^{(k)}$`, predicts the velocity, and the integrator updates the sample and advances `$\mathbf{t}^{(k)}$` by `$\Delta t$`. Dimensions with larger `$m_i$` start closer to the target and evolve with smaller per-step changes, effectively being more “preserved.” All dimensions complete within the same `$K$` steps.

> **What this enables:** A continuous, spatially-varying inpainting where a region is not just binary “keep” or “generate” but is smoothly blended. For images, this yields soft transitions between preserved and generated areas (Fig. 11, 12). For 3D shapes, fixing `$t_0 = 0.9$` on the bases of chairs produces diverse completions that adapt the preserved bases (Fig. 14). For channel-varying control, assigning a high `$t_0$` to early channels preserves structure while later channels control colour (Fig. 13).

---

#### 3.4.7 Implementation Details and Hyperparameters

**Image model:** SiT-XL/2 — 28 layers, hidden dim 1152, patch size `$2 \times 2$`, 16 attention heads. Training: AdamW with fixed learning rate `$10^{-4}$`, batch size 256, 4 NVidia H100 GPUs, 800K iterations (7.5 days). Evaluation: SDE integrator with 250 steps. Classifier-free guidance (CFG) scale 4.0 used for gallery figures but **not** for FID-50K reporting unless noted. The `$C_C = 4$` per-entry timestep embedding dimension is used to keep the added parameter count small.

**Structured 3D model:** SiT-small — 12 layers, hidden dim 384, 256 part tokens, 6 attention heads. Training: AdamW lr `$10^{-4}$`, batch size 16, 4 NVidia A100 GPUs, 1.5K epochs (3 days). Sampling: 500 iterations, part existence binarised at threshold 0.5 before being diffused back for the next iteration. Class-conditional sampling without CFG.

**Compensation:** For all experiments, the off-diagonal drift minimisation scheme is used. The compensation velocity `$v_{\text{cmpn}} = -\delta$` is added to the predicted velocity during test-time integration. No training objective modification is needed; the model is trained to predict `$x_1 - z$` (images) or `$x_1$` (3D shapes) from the ComboStoc-sampled `$\mathbf{x}_t$`.

**Mixed training:** For ImageNet, each batch contains 50% samples with fully asynchronous `$\mathbf{t}$` and 50% with synchronised `$t$`. For PartNet, 100% of samples are asynchronous.

**Key design choices justified:**
- *Velocity prediction with compensation* rather than switching to `$x$`-prediction: maintains compatibility with the state-of-the-art SiT configuration and isolates the effect of ComboStoc.
- *Uniform step number* (rather than uniform step size) for graded control: produces more stable results when `$\mathbf{t}_0$` has large variations (Tab. 7 ablation).
- *Full asynchronous training on small datasets*: the combinatorial coverage is essential to learn valid shapes with 18K samples; the baseline without it fails completely.


## GLM-OCR input · THINK-NONE

## 3. Technical Approach

### 3.1 Reader orientation (approachable technical breakdown)

This paper builds a **training and inference framework for diffusion generative models** that explicitly accounts for the combinatorial structure of high-dimensional data. The core problem it solves is that standard diffusion training—which moves all dimensions of a data sample along a synchronized interpolation schedule from noise to data—systematically *undersamples* regions of the path space where different dimensions or attributes are at different stages of the diffusion process, leading to poor performance when those regions are encountered at test time. The solution is strikingly simple: **vectorize the diffusion time schedule** so that each dimension (patch, feature channel, part attribute) receives its own independently sampled interpolation time, thereby covering the full combinatorial space of intermediate states uniformly.

### 3.2 Big-picture architecture (diagram in words)

The system has three major components that interact during training and can be reconfigured for inference:

1. **Base generative model (a stochastic interpolant / flow-matching transformer)** — takes a data sample `$x_1$` and a source noise `$z$` and learns to predict either the velocity `$x_1 - z$` (for images) or the clean data `$x_1$` (for structured 3D shapes) given a diffused intermediate `$x_t$`. Implemented as a SiT-style transformer [Ma et al. 2024] or DiT-style transformer [Peebles and Xie 2023].

2. **Tensorized timestep embedding module** — replaces the scalar time `$t$` with a tensor `$t$` of the same shape as the data sample (e.g., `$C \times H \times W$` for images; per-part/per-attribute for shapes). Each entry is independently sampled from `$[0, 1]$`. A modified embedding pipeline encodes this tensor into per-dimension conditioning signals.

3. **Off-diagonal drift compensation mechanism** — during test-time integration, when the solver encounters sample points with asynchronous (vectorized) timesteps, the raw velocity `$x_1 - z$` no longer integrates to the correct target. An additional compensation velocity `$v_{\text{cmpn}}$` pulls trajectories back toward the target.

**Information flow during training:** A data sample and noise sample are drawn → a tensorized timestep `$t$` is constructed with independent entries for each dimension/attribute → the diffused sample `$x_t = (1 - t) \odot z + t \odot x_1$` is formed → the network predicts velocity (or clean data) conditioned on `$x_t$`, class label, and the tensorized `$t$` → loss is computed against the target.

**During standard inference:** Synchronized timesteps are used (all entries of `$t$` equal) → standard SDE/ODE integration recovers the clean sample. **During graded-control inference:** An initial mask `$m$` specifies per-dimension preservation weights → the process starts from `$(1 - m) \odot z + m \odot x_1$` → asynchronous timesteps evolve each dimension at its own pace → the model generates a result with spatially varying degrees of preservation.

### 3.3 Roadmap for the deep dive

- **First**, the training-time **ComboStoc scheme** itself — how the tensorized timestep is constructed, what the diffused sample looks like, and why this uniformizes sampling density (Section 4, Eq. 5).
- **Second**, the **sampling bias analysis** that motivates the scheme — why standard synchronized interpolants create non-uniform coverage of the path space (Section 3, Eq. 2–4).
- **Third**, the **image-specific adaptation** — how the SiT timestep embedding module is modified to accept tensorized time, the configurations (`unsync_none`, `unsync_patch`, `unsync_vec`, `unsync_all`), and the half-batch mixing strategy (Section 4.1, 5.1–5.2).
- **Fourth**, the **structured 3D shape adaptation** — the part/attribute/feature-vector hierarchy, the six combinatorial configurations, and why `$x$`-prediction is used instead of `$v$`-prediction (Section 4.2, 5.2).
- **Fifth**, the **off-diagonal drift problem and compensation** — why following only `$x_1 - z$` fails at asynchronous sample points, the two compensation strategies (drift minimization vs. cone velocity), and the ablation justifying the chosen approach (Section 5.4.1).
- **Sixth**, the **asynchronous inference procedure** for graded control — the mask-based initialization, the uniform-step-number vs. uniform-step-size integration schemes, and how this enables soft inpainting and part-level assembly (Section 5.3, Algorithm 1).

### 3.4 Detailed, sentence-based technical breakdown

This is primarily a **methodology and empirical analysis paper** whose core idea is that diffusion generative models can be substantially improved by desynchronizing the interpolation schedule across dimensions and attributes, thereby covering the combinatorial path space uniformly rather than with the shrinking density characteristic of standard approaches.

---

#### 3.4.1 The Sampling Bias in Standard Diffusion Models

The paper identifies a fundamental and previously under-explored weakness in standard diffusion/flow-matching training: the **sampling density along the transport path is non-uniform**, with coverage shrinking as one moves away from the target data points. This is not a model-specific artifact but a direct consequence of the one-sided stochastic interpolant formulation.

**The standard interpolant.** In the unified framework of stochastic interpolants [Albergo et al. 2023], the diffused sample at time `$t$` is:

$$x_t = (1 - t) z + t x_1, \quad t \in [0, 1]$$

where `$z \sim N(0, 1)$` is the source noise, and `$x_1 \sim D$` is a target data sample. This defines a straight-line path from noise to data in the high-dimensional space of all dimensions concatenated.

**What this equation computes:** For a given noise point `$z$` and data point `$x_1$`, the intermediate `$x_t$` is a convex combination that starts at pure noise (`$t=0$`) and ends at the clean data (`$t=1$`). Every dimension of every token/patch/attribute follows the **same** scalar `$t$` — hence "synchronized."

**Why this form is used:** It is the simplest one-sided interpolant, conceptually straightforward, and the SiT paper [Ma et al. 2024] showed it achieves strong practical performance (secondary only to more complex schedules). All alternative interpolants follow `$x_t = \sigma_t z + \alpha_t x_1$` and therefore all share the same fundamental path-space undersampling problem since they are one-dimensional in `$t$`.

**The sampling density analysis.** The paper derives the probability density `$\rho(x)$` for sampling a point `$x$` in the path space — the subspace `$\mathcal{R}$` bounded by `$z$` as the minimum corner and `$x_1$` as the maximum corner. This density comes from integrating over all `$t$` the Gaussian distributions centered at `$p_t = (1-t)z + tx_1$` with variance scaled by `$(1-t)^2$`:

$$\rho(x) = \int_0^1 G_{p_t}(x) \, dt = \int_0^1 \frac{1}{\sqrt{2\pi}(1-t)} e^{-\frac{\|x - p_t\|^2}{2(1-t)^2}} \, dt.$$

**What this computes:** The probability that a particular point `$x$` in the subspace `$\mathcal{R}$` is visited during training, averaged over all interpolation times `$t \in [0, 1]$`. Each `$t$` contributes a Gaussian centered on the diagonal path with standard deviation `$1-t$`. As `$t$` approaches 1, the Gaussian narrows toward the data point.

**Why this form matters:** There is no closed-form solution, but the paper analyzes the gradient `$\nabla \rho$` to characterize the *direction* of density change. Substituting `$p_t = (1-t)z + tx_1$` yields:

$$(x_1 - x) \cdot \nabla \rho(x) = \frac{1}{\sqrt{2\pi}} e^{-\frac{\|x-z\|^2}{2}} > 0.$$

**What this gradient equation means:** The dot product of the gradient `$\nabla \rho(x)$` with the vector pointing *toward* the data point `$x_1$` from `$x$` is strictly positive. Therefore, `$\nabla \rho(x)$` always has a positive projection along the `$x_1 - x$` direction — meaning **the sampling density increases monotonically as one approaches the target data points**. The density is not uniform: it shrinks in regions farther from `$x_1$`.

**Why this is a problem:** During test-time stochastic evaluation (SDE or ODE integration), the solver may visit regions of the path space where the density `$\rho$` is low. The network `$f_\theta$` has been trained primarily on high-density regions near the data points and along the diagonal. When the solver stumbles into low-density off-diagonal regions, the network's predictions are less reliable, leading to poor sample quality or, in the low-data regime, complete failure (as shown for structured 3D shapes with `unsync_none` in Fig. 8).

**Visualization evidence.** Figure 2(d) visualizes the sampling density of the standard flow matching formulation, showing the clear shrinking tendency. Figure 3 (first row) shows a toy example where the velocity field `$u_t(x \mid x_1)$` and probability density `$p_t(x \mid x_1)$` exhibit non-converging outliers in integrated trajectories — particles that fail to reach the data manifold.

**Remark on variance-preserving interpolants.** The paper notes that when considering a full data distribution with variance `$\Sigma$`, the Gaussian in the integrand has variance `$(1-t)^2 \mathbb{1} + t^2 \Sigma$`, which motivates variance-preserving interpolants like cosine/sine schedules. However, for data with many attributes and high dimensions, **dataset sparsity** (the curse of dimensionality) means the target distribution's own variance cannot fill the path space adequately. For small-scale datasets like structured 3D shapes (18K samples in PartNet), this sparsity makes the sampling bias **indispensable** to address; for large-scale datasets like ImageNet (1.3M samples), mitigating it mainly improves convergence speed.

---

#### 3.4.2 The ComboStoc Training Scheme: Tensorized Timesteps

The paper's central technical contribution is a simple modification to the standard training procedure: **replace the scalar interpolation time `$t$` with a tensor `$t$` of the same shape as the data sample, where each entry is independently and uniformly sampled from `$[0, 1]$`.**

**The ComboStoc interpolation equation:**

$$x_t = (1 - t) \odot z + t \odot x_1$$

where `$\odot$` denotes elementwise (Hadamard) product.

**What this computes:** For a data sample of shape `$C \times H \times W$` (images) or an ensemble of parts with per-part attributes (structured shapes), each entry of `$t$` is independently drawn from `$U[0,1]$`. The diffused sample `$x_t$` is therefore an elementwise blend: for dimensions where `$t_i$` is close to 0, the sample is mostly noise; for dimensions where `$t_i$` is close to 1, the sample is mostly data. Different dimensions can be at *different stages* of the diffusion process simultaneously.

**Why this form works — uniform sampling of the combinatorial subspace:**

The key insight is that the subspace `$\mathcal{R} = \{x \mid z \preceq x \preceq x_1\}$` (elementwise) is a **hyper-rectangle** with `$z$` and `$x_1$` at opposite corners. Standard synchronized sampling with scalar `$t$` only ever visits points on the **diagonal** of this hyper-rectangle. The ComboStoc scheme with independently sampled `$t_i$` for each dimension `$i$` visits **every point in the hyper-rectangle** and, by construction, does so with **uniform density**. This is because the joint distribution of `$t$` is a product of independent uniforms over `$[0,1]$` for each dimension, and the mapping `$t \mapsto x_t$` is a linear bijection onto `$\mathcal{R}$`.

**Contrast with standard training:** In standard training (Eq. 1), the network only sees points on the line segment connecting `$z$` and `$x_1$` — the 1-dimensional diagonal through a `$D$`-dimensional space. At test time, the SDE solver performs a random walk that can (and does) visit off-diagonal points. The network has never been trained on these, creating a train-test mismatch. ComboStoc eliminates this mismatch by **training on the full hyper-rectangle**.

**The three-fold benefits (from the paper):**

1. **Broader network coverage:** Training on the full combinatorial space means the network sees and learns to handle states where different dimensions are at different noise levels. At test time, when the stochastic solver inevitably visits such states, the network's predictions are more robust.

2. **Learning inter-dimension correlations:** Because the training samples mix noise and data differently across dimensions, the network must learn how to *synchronize* them to reach a coherent data point. This implicitly encourages learning the correlations among patches, feature channels, and attributes — the very structure that makes the data interesting.

3. **Enabling graded control at inference:** Since the network has been trained on samples with arbitrary per-dimension `$t$` values, at inference time one can initialize the process with a mask `$m$` specifying different preservation levels for different dimensions. The network can then generate around these constraints natively, without task-specific fine-tuning.

**The formal proof that ComboStoc defines a proper generative flow (Section 4.1):**

The authors show that the marginalized vector field `$u_t(x)$` induced by the `$x_0, x_1$`-conditioned vector field `$u(x \mid x_0, x_1)$` satisfies the continuity equation and generates the correct probability path `$p_t$` from source to target distribution. The derivation (Eqs. 6–7) generalizes the Flow Matching framework [Lipman et al. 2023] by replacing `$\text{diag}(x_0, x_1)$` (the line connecting `$x_0$` and `$x_1$`) with `$\text{span}(x_0, x_1)$` (the full rectangular subspace). The key steps are:

$$\frac{d}{dt} p_t(x) = \iint \left( \frac{d}{dt} p_t(x \mid x_0, x_1) \right) q(x_0) r(x_1) \, dx_0 \, dx_1$$

$$= - \iint \text{div}\left( u(x \mid x_0, x_1) \, p_t(x \mid x_0, x_1) \right) q(x_0) r(x_1) \, dx_0 \, dx_1$$

$$= - \text{div}\left( \iint u(x \mid x_0, x_1) \, p_t(x \mid x_0, x_1) \, q(x_0) r(x_1) \, dx_0 \, dx_1 \right)$$

**What this proves:** The first equality expands the marginal probability into an integral over all source-target pairs. The second uses the fact that the conditional vector field `$u(x \mid x_0, x_1)$` generates the conditional probability path `$p_t(x \mid x_0, x_1)$` — a point distribution moving from `$x_0$` to `$x_1$`. The third equality swaps differentiation and integration. The result defines the marginalized velocity field `$u_t(x) = \frac{1}{p_t(x)} \iint u(x \mid x_0, x_1) p_t(x \mid x_0, x_1) q(x_0) r(x_1) dx_0 dx_1$`. Crucially, `$u(x \mid x_0, x_1)$` is time-invariant — the apparent time dependence in Flow Matching arises only after marginalization over `$p(x_0)$`.

**Why this form is a valid generalization:** In standard Flow Matching, the conditional vector field is defined only on the diagonal connecting `$x_0$` and `$x_1$`. Here, it is defined on the entire span. The continuity equation still holds because any point `$x$` in the span moves according to a valid (time-invariant) vector field that pushes it toward `$x_1$`. The paper notes that the timestep `$t$` in the continuity equation is a **scalar integration variable** — even though training uses vectorized timesteps (Eq. 5), the analysis in the standard scalar-time framework shows the marginalized velocity field `$u_t(x)$` still generates the correct probability path.

---

#### 3.4.3 Image-Specific Adaptation: Modifying the SiT Timestep Embedding

For image generation, the paper builds on top of **SiT-XL/2** [Ma et al. 2024], a state-of-the-art scalable transformer for ImageNet-class generation. The model encodes an image via a VAE encoder [Rombach et al. 2022] into a latent `$x_1$` of shape `$C \times H \times W$` (where `$C = 4$` for the SD-VAE latent channels, `$H = W = 32$` for 256×256 images after 8× downsampling). The network is trained to predict the **velocity** `$v = x_1 - z$` given the diffused latent `$x_t$`, the class label `$c$`, and the interpolation time `$t$`.

**Modification 1: Tensorizing the timestep.** The scalar `$t$` is replaced with a tensor of shape `$(N, C, H, W)$` (batch size `$N$`). For each sample, entries are independently drawn from `$U[0,1]$`. The diffused sample becomes:

$$x_t = (1 - t) \odot z + t \odot x_1$$

where `$\odot$` is elementwise multiplication.

**Modification 2: Adapted timestep embedding module (Figure 5).** The original SiT module (Fig. 5a) takes a scalar `$t$` of shape `$(N,)$`, applies sine/cosine frequency encoding to produce a vector of length `$C_F$` (the frequency embedding dimension), then passes it through two MLP layers to produce a conditioning vector of shape `$(N, C_H)$` where `$C_H = 1152$` is the hidden dimension of the SiT transformer. This conditioning vector is then used to modulate the transformer layers via adaLN (adaptive layer norm) operations.

The ComboStoc adaptation (Fig. 5b) works in two stages:

- **Stage 1 — Per-entry encoding:** The tensor `$t$` of shape `$(N, C, H, W)$` has each of its entries independently passed through the same sine/cosine frequency encoding and a small MLP to produce a compressed encoding of dimension `$C_C = 4$`. This yields a tensor of shape `$(N, C, H, W, C_C)$`. Using `$C_C = 4$` (instead of `$C_H = 1152$`) keeps the parameter count manageable.

- **Stage 2 — Patch-wise embedding:** The result tensor is transposed to combine the channel dimensions `$C \times C_C$`, and then processed by a **patch embedding layer** (identical in structure to the ViT patch embedding [Dosovitskiy et al. 2021], but with separate parameters). With patch size `$L \times L = 2 \times 2$`, the spatial grid `$H \times W$` is divided into `$T = HW/L^2$` patches. Each patch of the timestep tensor is embedded into a vector of dimension `$C_H = 1152$`, matching the hidden dimension of the main transformer. The result is a sequence of `$T$` timestep tokens that are added to the corresponding patch tokens of the diffused image latent.

**Why this two-stage design?** The SiT architecture conditions on class labels and timesteps via **modulation** (adaLN), not via concatenation. The modulation operates on vectors of channel dimension `$C_H$`. To provide per-patch, per-channel conditioning, the timestep information must be transformed into a sequence of vectors matching the patch sequence length. The first stage compresses each scalar entry to `$C_C = 4$` to avoid an explosion in parameters (a naive full-dimensional encoding would require a timestep embedding dimension of `$C_H = 1152$` for each of `$C \times H \times W$` entries, which is prohibitive). The second stage uses a standard patch embedding to aggregate local timestep information.

**The four combinatorial configurations (Tab. 2):**

The paper enumerates four settings with increasing degrees of desynchronization:

- **`unsync_none`** — no splitting: a single scalar `$t$` broadcast to all dimensions (identical to baseline SiT, but with the smaller ComboStoc embedding module). Shape: `$(N,)$`.

- **`unsync_patch`** — spatial desynchronization: different `$t$` values for different spatial locations (pixels/patches), but all feature channels share the same `$t$` at each location. Shape: `$(N, 1, H, W)$`.

- **`unsync_vec`** — feature channel desynchronization: different `$t$` values for different channels of the latent encoding, but all spatial locations share the same per-channel `$t$`. Shape: `$(N, C, 1, 1)$`.

- **`unsync_all`** — full desynchronization: independent `$t$` values for every `$(C, H, W)$` entry. Shape: `$(N, C, H, W)$`.

**Half-batch mixing strategy.** For large-scale datasets like ImageNet (1.3M images), each training batch applies the split timesteps only to **half** the samples; the other half uses synchronized timesteps. This balances between samples on the diagonal paths (standard) and samples off the diagonal (ComboStoc). The paper describes this as a form of "curriculum learning": fully asynchronous training can increase early optimization difficulty; retaining a portion of synchronized samples helps stabilize convergence. For the smaller PartNet dataset (18K shapes), asynchronous schedules are applied to **all** samples with no mixing.

**Velocity prediction target.** The paper follows the SiT convention of `$v$`-prediction: the network predicts `$x_1 - z$`. The loss is computed as the mean squared error between the predicted and true velocities. This is in contrast to the structured 3D shape model, which uses `$x$`-prediction (predicting `$x_1$` directly), as noted to be a "more robust design choice for the heterogeneous representation that mixes existence indicators, bounding boxes, and shape codes."

---

#### 3.4.4 Structured 3D Shape Generation: Part-Level Combinatorial Complexity

Structured 3D shapes have significantly stronger combinatorial complexity than images. The representation (from Wang et al. [2025]) encodes a shape as a collection of **up to `$L = 256$` parts**, where each part `$p_i$` is a tuple:

$$p_i = (s_i, b_i, e_i)$$

where:
- `$s_i \in [0, 1]$` — the **existence score** indicating whether this part slot is occupied;
- `$b_i = (x, y, z, l, w, h)$` — the **bounding box** (center coordinates `$x, y, z$` and lengths `$l, w, h$`);
- `$e_i \in \mathbb{R}^{512}$` — a **latent shape code** encoding the part geometry in normalized coordinates (obtained from a pretrained point-cloud VAE).

Additionally, the part ordering is **permutation-invariant**: shuffling the indices does not change the represented 3D shape. This property is fundamentally different from images, which have a fixed spatial grid.

**The six combinatorial configurations (Tab. 2b):**

The paper identifies three axes of combinatorial complexity for structured shapes:

- **Spatial parts** (`part`): treating different part slots as different "spatial" positions.
- **Attributes** (`att`): treating the different attributes (existence `$s$`, bounding box `$b$`, shape code `$e$`) of each part independently.
- **Feature vectors** (`vec`): treating the individual dimensions within the bounding box (6 scalars) or shape code (512 scalars) independently.

Combining these axes yields six settings:

| Setting | Splits parts? | Splits attributes? | Splits feature dims? |
|:---|:---|:---|:---|
| `unsync_none` | ✗ | ✗ | ✗ |
| `unsync_part` | ✓ (per part) | ✗ | ✗ |
| `unsync_att` | ✗ | ✓ (per attribute) | ✗ |
| `unsync_att_part` | ✓ | ✓ | ✗ |
| `unsync_vec` | ✗ | ✗ | ✓ (per feat dim) |
| `unsync_all` | ✓ | ✓ | ✓ |

**Embedding design.** The embedding for part existence and bounding box attributes follows the same pattern as the timestep embedding (Fig. 5): each scalar dimension is turned into a sine/cosine frequency code, then embedded into a vector of dim 4 (matching the `$C_C = 4$` of the image model), and finally each collective attribute (e.g., the full bounding box of 6 scalars) is embedded as a whole into a vector of hidden dim 384 via a dedicated FC layer. The shape code `$e \in \mathbb{R}^{512}$` is already a learned latent from the pretrained VAE.

**Network architecture.** The model uses a **SiT small** configuration: 12 layers, hidden dimension 384, 256 tokens for the parts (matching `$L = 256$`), and 6 attention heads. The model predicts `$x_1$` directly (`$x$`-prediction) rather than velocity. The paper describes `$x$`-prediction as "a more robust design choice for the heterogeneous representation that mixes existence indicators, bounding boxes, and shape codes." Both velocity and `$x$`-prediction are viable targets in flow-based models [Li and He 2025], but for this mixed representation, regressing the clean data directly is more stable.

**Training and inference details.** Trained on PartNet [Mo et al. 2019b] (18K shapes, mostly chairs and tables) with:
- AdamW optimizer, fixed learning rate `$10^{-4}$`, batch size 16
- 4 Nvidia A100 GPUs, 3 days for 1.5K epochs
- Class-conditional sampling without CFG (classifier-free guidance)
- 500 sampling iterations; at each iteration, part existence is binarized via threshold 0.5 before being re-diffused
- Synchronous timesteps during standard inference (all entries of `$t$` equal); asynchronous timesteps for the graded control applications

---

#### 3.4.5 The Off-Diagonal Drift Problem and Compensation

A critical issue arises when training with ComboStoc's vectorized timesteps for velocity prediction: **the network trained to predict `$v = x_1 - z$` at arbitrary off-diagonal sample points will, during test-time integration with asynchronous timesteps, fail to converge to the target data point `$x_1$`**.

**The problem formalized.** Consider an ODE integration starting from an asynchronous sample point `$x_{t_0}$` with tensorized initial time `$t_0$` (different values across dimensions). If the solver follows only the velocity `$x_1 - z$`, the terminal point is:

$$x_{t_0} + \int_{t_0}^{1} (x_1 - z) \, dt = z + t_0 \odot (x_1 - z) + (1 - t_0)(x_1 - z)$$

where `$t_0 = \min(t_0)$` is the scalar minimum across all dimensions of `$t_0$`. This simplifies to:

$$= x_1 + (t_0 - t_0) \odot (x_1 - z)$$

**What this equation means:** If `$t_0$` were a uniform scalar (all entries equal), the final point would be exactly `$x_1$` — the integration works perfectly. But when `$t_0$` has different entries, the final point is `$x_1$` *plus* an offset vector `$(t_0 - t_0) \odot (x_1 - z)$`. The dimensions with larger `$t_0$` than the minimum receive a correction proportional to `$(t_0 - t_0)(x_1 - z)$`, pushing them **past** `$x_1$`. The result is a drift away from the target — visible in Fig. 4 as the dotted line.

**Why this happens:** During training with ComboStoc, the network is shown sample points `$x_t$` where individual entries `$t_i$` can differ, and the target velocity is still `$x_1 - z$`. However, this target velocity is only correct for points **on the diagonal** connecting `$z$` to `$x_1$`. For an off-diagonal point, the vector directly pointing to `$x_1$` is not `$x_1 - z$` but rather `$x_1 - x_t$`. The training implicitly learns to compensate, but at test time with asynchronous schedules, the integration needs explicit correction.

**Compensation method 1: Off-diagonal drift minimization (chosen approach).**

The paper defines the off-diagonal offset vector `$\delta(x_t)$`:

$$\delta(x_t) = x_t - x_1 - \frac{(x_t - x_1) \cdot (x_1 - z)}{\|x_1 - z\|^2} (x_1 - z).$$

**What this computes:** `$\delta(x_t)$` is the component of the vector from `$x_t$` to `$x_1$` that is **orthogonal** to the target velocity direction `$x_1 - z$`. The first term `$x_t - x_1$` is the direct vector to the target. The second term projects this vector onto the velocity direction `$x_1 - z$` and subtracts that projection. The remainder `$\delta(x_t)$` is the *off-diagonal* component — the part of the displacement that the raw velocity does not cover.

The compensation velocity is:

$$v_{\text{cmpn}} = -\delta(x_t)$$

**What this does during integration:** Following `$v_{\text{cmpn}}$` in addition to the original `$x_1 - z$` is equivalent to performing gradient descent on a drift potential `$\Phi(\delta(x_t)) = \frac{1}{2} \|\delta_t\|^2$`. This pulls the trajectory orthogonally back toward the diagonal connecting `$z$` and `$x_1$`, promoting convergence to `$x_1$` (Fig. 4, dashed arrow).

**Compensation method 2: Cone-shaped velocity field (alternative, rejected).**

An alternative velocity field generalizes the constant velocity to a cone covering the expanded region `$\mathcal{R}$`:

$$v_{t_0} = \frac{x_1 - x_{t_0}}{1 - t_0}$$

where `$t_0 = \min(t_0)$`. For synchronized schedules, `$v = \frac{x_1 - x_{t_0}}{1 - t_0} = \frac{x_1 - z - t_0(x_1 - z)}{1 - t_0} = x_1 - z$`, recovering the original formulation. This is a "cone" because for any timestep `$t_\lambda = \lambda t_0 + (1 - \lambda) 1$` along the line from `$t_0$` to 1, the velocities are equal — thus the field is constant along rays emanating from `$x_1$`.

**Why this is rejected:** The cone velocity introduces a division by `$1 - t_0$` where `$t_0$` is the **slowest**-evolving dimension. When dimensions are scheduled very differently (some near 0, some near 1), `$1 - t_0$` can be extremely small, causing the velocity magnitude to blow up. This "lack of regularity" leads to degraded regression of the velocity field. Empirically (Tab. 6), the cone velocity field achieves **poorer** FID (113.59 vs. 103.01) and **lower** SSIM (0.224 vs. 0.262) compared to the drift minimization approach.

**Ablation results (Tab. 6):** Trained for 50K steps on the first 100 ImageNet categories on 8 Nvidia H20 GPUs.

- **w/o Cmpn** (no compensation): FID 103.75, SSIM 0.255
- **Off-diagonal Drift** minimization: FID 103.01, **SSIM 0.262** (best)
- **Cone Velocity**: FID 113.59, SSIM 0.224 (worst)

The visual difference is more pronounced than the scalar metrics suggest: Fig. 16 shows that without compensation, a **noticeable discontinuity appears along the midline** of images where the two halves have different preservation weights. The off-diagonal drift minimization produces a seamless transition.

**How the compensation is used:** The off-diagonal drift minimization is applied during **training** (to compute the correct training targets at off-diagonal sample points) and during **asynchronous inference** (Alg. 1B). In standard synchronous inference (Alg. 1A), no compensation is needed because all dimensions share the same scalar `$t$`.

---

#### 3.4.6 Asynchronous Inference for Graded Control

The ComboStoc training scheme enables a new inference paradigm: **graded control** where different dimensions and attributes of the generated sample can be given different degrees of preservation relative to a reference sample. This is formalized in Algorithm 1B.

**Initialization with a mask.** Given a reference sample `$x_1$` and a mask `$m \in [0, 1]^{\text{shape}(x)}$` where each entry `$m_i$` specifies the degree of preservation for that dimension:

$$x^{(0)} = (1 - m) \odot z + m \odot x_1, \quad t^{(0)} = m$$

**What this means:** For `$m_i = 0$`, that dimension starts from pure noise and evolves fully. For `$m_i = 1$`, that dimension starts from the reference data and stays (nearly) fixed. For `$m_i = 0.5$`, the dimension starts halfway between noise and data.

**The integration procedure (uniform step number `$N$`).** The remaining trajectory from `$t^{(0)}_i$` to 1 is divided into `$N = 250$` equal steps for **all** dimensions:

$$\Delta t = \frac{1 - m}{N}$$

Each dimension increments by `$\Delta t_i = \frac{1 - m_i}{N}$`. Dimensions with larger `$m_i$` have fewer remaining steps (since they start closer to 1) but the same **step count** `$N$` — meaning they evolve more slowly per step. The network consumes the fully asynchronous time field `$t^{(k)}$` at each iteration and predicts the velocity at that state.

**Why this works:** The network was trained on samples with arbitrary per-dimension `$t$` values. At inference, it sees the same kind of asynchronous state. The network has learned to "synchronize" the different dimensions — to move them coherently toward a data-consistent configuration despite their different starting points.

**Comparison with uniform step size scheme (Tab. 7).** An alternative scheme uses a uniform step size `$\Delta t = 1/N$` for all dimensions, with different dimensions having different numbers of steps (stopping when they reach 1). The paper finds that both schemes produce similar results for the fully trained 800K-step model, indicating that ComboStoc is robust to the specific integration strategy.

**Applications enabled:**

- **Soft inpainting with spatially continuous masks (Fig. 12):** Unlike binary-mask inpainting [Lugmayr et al. 2022], ComboStoc supports a smoothly varying `$m$` map. Pixels near the center of the subject have `$m \approx 0.85$` and are strongly preserved; pixels at the periphery have `$m = 0$` and are freely generated. The model produces coherent transitions between preserved and generated regions.

- **Channel-varying control (Fig. 13):** Assigning different `$m$` values to different VAE latent channels reveals that earlier channels (`$C = 0$`) encode spatial structure while later channels (`$C = 1, 2, 3$`) encode color distributions. Preserving only channel 0 preserves structure but loses color; preserving channel 3 preserves color but loses spatial structure.

- **Quadrant-based control (Fig. 11):** Different `$m$` values for the four quadrants of an image enable a smooth blend between reference preservation and novel generation.

- **Part-level assembly for 3D shapes (Fig. 15):** Giving the part shape codes `$e$` and bounding box sizes high `$m = 0.9$` (largely preserved) while letting the part existence `$s$`, positions, and other attributes be freely generated enables the model to arrange pre-specified parts into coherent assemblies.

---

#### 3.4.7 Training and Inference Pseudocode (Algorithm 1)

The paper provides a complete pseudocode specifying both the **synchronized** (standard generation) and **asynchronous** (graded control) inference procedures.

**Synchronized inference (Alg. 1A):** This is the standard generation mode. `$t$` is a scalar broadcast to all dimensions as a uniform tensor. The process starts from pure noise `$x^{(0)} = z$` and follows the SDE integrator for `$K = 250$` steps. The model `$f_\theta$` predicts velocity at each step. All FID results in Section 5.2 use this inference mode — **the training gains come purely from the asynchronous training scheme, not from changed inference**.

**Asynchronous inference with graded control (Alg. 1B):** The process starts from `$x^{(0)} = (1 - m) \odot z + m \odot x_1$`. The timestep tensor `$t^{(0)} = m$` is initialized with the mask values. At each of `$K = 250$` iterations, the model consumes the fully asynchronous time field `$t^{(k)}$` of the same shape as the data, predicts velocity, and updates via the standard SiT integrator. The timestep is then incremented: `$t^{(k+1)} \leftarrow \min(1, t^{(k)} + \Delta t)$` where `$\Delta t = \frac{1 - m}{K}$`. This uniform-step-number scheme means all dimensions complete their evolution within the same 250 iterations, but dimensions with larger initial `$m_i$` evolve more slowly per iteration.

**Why the uniform-step-number scheme is the default:** It ensures that the entire process completes in a fixed number of steps, which is convenient for batching and does not leave some dimensions with extremely few integration steps. The paper's ablation (Tab. 7) shows the uniform-step-size alternative produces similar results, so the choice is not critical.

---

#### 3.4.8 Summary of Design Choices and Their Justifications

- **Tensorized timesteps with independent entries per dimension** rather than a single scalar: this is the core idea — it uniformizes sampling density across the full combinatorial subspace spanned by each source-target pair, eliminating the path-space coverage bias identified in Section 3.

- **`$C_C = 4$` for the compressed timestep encoding** rather than `$C_H = 1152$`: keeps the parameter count of the timestep embedding module bounded. The paper explicitly notes this is likely why `unsync_none` slightly underperforms baseline SiT (both have identical architectures elsewhere, but the smaller embedding is less expressive).

- **Half-batch mixing for ImageNet, full-batch for PartNet:** ImageNet's large scale (1.3M images) requires stabilization; fully asynchronous training on a small dataset like PartNet (18K shapes) does not need such stabilization and benefits from full coverage of the combinatorial space.

- **`$x$`-prediction for structured shapes, `$v$`-prediction for images:** The heterogeneous representation of shape parts (existence indicators, bounding boxes, shape codes) benefits from the stability of predicting the clean data directly. For images, `$v$`-prediction is used to isolate the effect of ComboStoc and remain consistent with the SiT baseline.

- **Off-diagonal drift minimization over cone velocity:** The drift potential approach has better regularity (no division by `$1 - \min(t_0)$`) and empirically achieves better FID, better SSIM, and visually seamless transitions in graded control (Figs. 16, Tab. 6).

- **Uniform step number `$N = 250$` for asynchronous inference:** All dimensions complete in the same number of iterations, simplifying implementation. The uniform-step-size alternative is shown to be equally valid.