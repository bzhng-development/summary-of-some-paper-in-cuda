# GLM-OCR markdown input — 2405.13729

# ComboStoc: Combinatorial Stochasticity for Diffusion Generative Models

_This is the GLM-OCR output (the LLM input). 79,441 chars._

---

ComboStoc: Combinatorial Stochasticity for Diffusion Generative Models

RUI XU, The University of Hong Kong, China
JIEPENG WANG, The University of Hong Kong, China
HAO PAN†, Tsinghua University, China
YANG LIU, Microsoft Research Asia, China
XIN TONG, Microsoft Research Asia, China
SHIQING XIN, Shandong University, China
CHANGHE TU, Shandong University, China
TAKU KOMURA, The University of Hong Kong, China
WENPING WANG†, Texas A&M University, USA

Fig. 1. ComboStoc improves diffusion generative models across data modalities of images and structured 3D shapes. Left: structured 3D shapes where semantic parts are colored randomly. Right: images with consistently lower Frechet Inception Distance (FID) than baseline results. Middle: core idea of ComboStoc is a simple conversion of the interpolation schedule $t$ of the diffusion models into a tensor of the same shape as the data point $x_1$ and noise point $z$, and applying different values within [0, 1] for different dimensions or attributes to fully sample the combinatorial complexity of the dimensions and attributes.

In this paper, we study an under-explored but important factor of diffusion generative models, i.e., the combinatorial complexity. Data samples are generally high-dimensional, and for various structured generation tasks, additional attributes are combined to associate with data samples. We show that the space spanned by the combination of dimensions and attributes can be insufficiently covered by existing training schemes of diffusion generative models, potentially limiting test time performance. We present a simple fix to this problem by constructing stochastic processes that fully exploit the combinatorial structures, hence the name ComboStoc. Using this simple strategy, we show that network training is significantly accelerated across diverse data modalities, including images and 3D structured shapes. Moreover, ComboStoc enables a new way of test time generation which uses asynchronous time steps for different dimensions and attributes, thus allowing for varying degrees of control over them. Our code is available at: https://github.com/Xrvitd/ComboStoc.

CCS Concepts: • Computing methodologies → Machine learning algorithms; Computer vision representations; Shape modeling.

Additional Key Words and Phrases: Diffusion Generative Model, Combinatorial Stochastic, Image, Structured 3D Shape, Graded Control

ACM Reference Format:
Rui Xu, Jiepeng Wang, Hao Pan, Yang Liu, Xin Tong, Shiqing Xin, Changhe Tu, Taku Komura, and Wenping Wang. 2026. ComboStoc: Combinatorial Stochasticity for Diffusion Generative Models. ACM Trans. Graph. 45, 4, Article 94 (July 2026), 17 pages. https://doi.org/10.1145/3811285

1 Introduction

Diffusion models are the state-of-the-art generative models across many domains and applications [Yang et al. 2023]. Diffusion generative models rely heavily on modeling the desired behavior over

an extended space of noise corrupted data samples, so that they can cover the target data distributions systematically. However, the current training schemes generally focus on a single transport path from the source pure noise distribution to the target data distribution [Albergo et al. 2023; Albergo and Vanden-Eijnden 2023; Lipman et al. 2023; Liu et al. 2023]. The training therefore can lead to biased sampling density across the space of corrupted samples, where certain regions may be insufficiently covered and, when encountered during stochastic evaluation, produce less accurate behavior.

To address the mismatch between the training scheme and test-time evaluation, we propose fully sampling the space of combinatorial complexity. The combinatorial nature of this space arises because data samples typically reside in high-dimensional spaces with distinct combinatorial structures. For instance, the most powerful generative models to date utilize transformers as the network architecture [Ma et al. 2024; Peebles and Xie 2023]. These models treat an image sample as a collection of patch tokens, which are generated in parallel. Furthermore, each patch token is encoded as a high-dimensional vector. The combination of patches and their feature vectors presents highly complex spaces, over which the diffusion generative models must learn to evolve toward data samples where patches and feature vectors are correlated nontrivially. In addition, for generative tasks in more structured domains like 3D shapes with semantic parts [Liu et al. 2024; Wang et al. 2025], the combinatorial complexity is even more pronounced: each part has numerous attributes encoding different properties like its existence, bounding box and part shape, in addition to the part/patch decomposition and multiple feature channels analogous to images.

We sample the spaces of such combinatorial complexity by a simple modification of typical transport plans. In particular, instead of using a synchronized time schedule for each data sample, we apply asynchronous time steps for each of the patches/parts, attributes and feature vector dimensions, which allows for full sampling of a subspace spanning the various combinations of each pair of source and target data points.

We show that by simply enhancing the training scheme to incorporate the combinatorial sampling, the generative models for images and 3D structured shapes can be significantly improved. In particular, for images from ImageNet [Deng et al. 2009], we obtain systematic FID-50k improvements along different training iterations than baseline SiT [Ma et al. 2024] and DiT [Peebles and Xie 2023] (Tab. 1). For 3D structured shapes which have stronger combinatorial complexity, we show our training scheme is indispensable for obtaining a working generative model (Fig. 1 left).

In addition to the improved performances, the training scheme exploiting combinatorial stochasticity enables new modes of using the trained generative models. Specifically, we can now generate different patches/parts/attributes in asynchronous time schedules. This means that for example we can condition the final sample on flexible partial observations of a reference sample beyond binary masks. Instead, for images we can apply graded control across patches and channels. For structured shapes we can also specify the shapes of some parts only, and let the model generate the remaining parts and attributes. These new modes of generation have the potential to unify specialized image and shape editing solutions.

In summary, we make the following contributions:

- We propose ComboStoc, an improved diffusion-based training framework that enhances generative modeling by vectorizing the diffusion time steps during training, enabling the model to better capture and reason about structured and combinatorial data.
- ComboStoc performs consistently well across both image and 3D structured shape domains. ComboStoc significantly accelerates training and achieves lower FID scores on ImageNet [Deng et al. 2009], while on 3D structured shapes it produces substantially better generation quality on generation tasks.
- While maintaining high generation quality, ComboStoc supports a rich set of timestep-controlled inference applications within shorter training times, including image inpainting, as well as controllable generation and part-level assembly for 3D structured shapes.

2 Related Works

2.1 Image Generation.

For image generation, a large body of work has focused on improving diffusion training schemes [Go et al. 2024; Hang et al. 2023a; Wang et al. 2024; Zheng et al. 2025], including refined loss weighting and time-step schedules [Hang et al. 2023b], training acceleration via distillation [Meng et al. 2023], and enforcing sampling-path consistency [Song et al. 2023]. Despite these advances, the role of combinatorial complexity in diffusion training has received relatively little attention. A notable exception is Gao et al. [2023], which attributes the slow convergence of DDPM-based DiT models [Peebles and Xie 2023] to pixel-wise regression losses that fail to sufficiently emphasize structural correlations across image patches. To address this issue, Gao et al. [2023] propose a mask-and-diffusion scheme that randomly masks portions of diffused images during training, encouraging the model to learn inter-patch dependencies. This approach, however, relies on a relatively complex encoder-decoder architecture with additional side-interpolation modules. In contrast, our method introduces a substantially simpler training scheme that requires only minimal modifications to baseline architectures, yet already yields significant training improvements for SiT [Ma et al. 2024] models.

Beyond training efficiency, several works explore spatially or temporally varying noise schedules to enable finer-grained control during diffusion inference. For image editing, SDEdit [Meng et al. 2021] adapts standard diffusion models for stroke-based editing by injecting a global noise level across the entire image, allowing user-provided strokes to be coherently blended into the generated output. Sahoo et al. [2024] improves likelihood estimation by introducing spatially adaptive noise conditioned on the input signal. SVNR [Pearl et al. 2023] further generalizes this idea with a spatially variant diffusion formulation that initializes denoising directly from the noisy input and assigns each pixel an individual time embedding, enabling more realistic noise modeling and achieving state-of-the-art performance in real-world image denoising. Soft inpainting is explored in Levin and Fried [2025] by incorporating carefully designed blending masks into the iterative denoising process. Related trends also appear in video diffusion models, methods such as Ruhe et al. [2024] and Kim et al. [2024] assign stronger noise to later frames to reflect higher temporal uncertainty, while Chen et al. [2024]; Song et al. [2025] randomize frame-wise noise strengths during training to simulate diverse prefix-mask conditions. At inference time, these models again exploit asymmetric noise schedules to better leverage temporal reasoning. Concurrently, Hu et al. [2026] explicitly assign pixel-level timesteps and studies asynchronous denoising for text-to-image alignment, and AR-Diffusion [Sun et al. 2025] introduces frame-specific timesteps with a scheduler balancing timestep compositions for auto-regressive video generation. Both works share the spirit of per-dimension asynchronous scheduling with our approach, but they focus on different application domains and do not address the training-time combinatorial coverage perspective studied here. Different from these approaches, we apply fully unsynchronized noise during training, which allows for graded control in the inference stage of greater flexibility than specific schedules, demonstrating the synergy of efficient training and adaptive inference.

2.2 Structured 3D Shape Generation.

While diffusion-based generative models for 3D data have rapidly expanded [Xiang et al. 2024; Zhang et al. 2023, 2024; Zhao et al. 2025; Zheng et al. 2023], relatively few works explicitly target structured shape generation. Early efforts such as Mo et al. [2019a] focus on learning hierarchical shape representations and generating structured variations using a VAE framework. Building on this line of work, Wang et al. [2025] propose a rewriting-based model that enables more generalizable cross-category generation. In contrast to hierarchical representations, we focus on generating flatly structured 3D shapes composed of leaf-level semantic parts. By independently specifying parts and attributes, our model naturally supports a wide range of tasks, including shape completion and part-based assembly. Previously, these applications were typically addressed using specialized, task-specific solutions [Huang et al. 2020; Sung et al. 2015]. Our results suggest that such diverse tasks can instead be unified under a single generative framework that explicitly models highly structured data.

Recent part-level generative approaches further highlight the importance of structured 3D representations. BANG [Zhang et al. 2025] introduces a diffusion-based framework for part-level decomposition via generative exploded dynamics, producing temporally coherent exploded states that enable controllable and semantically consistent part separation. X-Part [Yan et al. 2025] proposes a controllable generative model for high-fidelity, structure-coherent part-level decomposition, leveraging bounding-box prompts and point-wise semantic features to produce editable and production-ready 3D assets. These applications in part-level generation can potentially benefit from a more robust and data-efficient generative model that our approach demonstrates.

2.3 Diffusion Acceleration and Representation Learning.

REPA [Yu et al. 2024] was proposed to accelerate diffusion training by distilling pre-trained, self-supervised visual representations of clean images into intermediate latent representations of noisy inputs, resulting in substantial speedups. Beyond REPA [Yu et al. 2024], several recent studies explore accelerating or stabilizing diffusion and flow-based generative training from other orthogonal perspectives. DeepFlow [Shin et al. 2025] introduces deep velocity supervision with inter-layer velocity alignment, achieving up to an 8× speedup in convergence for flow-based models. Representation Autoencoders (RAE) [Yu et al. 2024] replace VAEs with pretrained representation encoders to provide semantically rich, high-dimensional latent spaces, improving reconstruction quality and convergence speed for DiT-style architectures. Our approach focusing on handling combinatorial complexity is orthogonal to these works, suggesting potential gains from combining these strategies and our approach.

3 Background on Diffusion Models

The problem of generative modeling aims at capturing the complete distribution of a set of data samples. Its state-of-the-art solutions include denoising diffusion probabilistic models [Ho et al. 2020], score-based models [Song et al. 2021] and flow matching [Lipman et al. 2023; Liu et al. 2023], all of which transform a simple source distribution (e.g., the unit normal distribution) into the target distribution following the dynamics specified by variations of stochastic differential equations. Remarkably, the different formulations can be unified through the framework of stochastic interpolants [Albergo et al. 2023; Albergo and Vanden-Eijnden 2023]. In particular, the stochastic interpolants framework defines the process of turning data samples into source distributions and vice versa as a simple interpolation between the two distributions, augmented with random perturbations during the processes.

We reproduce the formulation of a simple linear one-sided interpolant process below (illustrated in Fig. 2 (a)):

$$x_t = (1 - t)z + tx_1, \quad t \in [0, 1]$$

where $z \sim N(0, 1)$ samples the source distribution, $x_1 \sim D$ samples the target data distribution, $t \in [0, 1]$ is the interpolation schedule. A network model $f_\theta(x_t)$ can be trained to recover the interpolation velocity $\frac{\partial x_t}{\partial t} = x_1 - z$, the target data sample $x_1$, or the noise $z$ [Albergo et al. 2023]. To generate data samples, one starts from random samples $z$, follows the velocity field and integrates them numerically to reach the final samples. Remarkably, on modeling large scale image datasets like ImageNet [Deng et al. 2009], a scalable

transformer architecture implementing the above process [Ma et al. 2024] shows state-of-the-art performance and outperforms alternative formulations, including DDPM [Ho et al. 2020] implemented via the same network [Peebles and Xie 2023].

Note that we use the linear interpolant model for its conceptual simplicity. Nonetheless, its practical performance is also strong, being secondary among alternative interpolants according to SiT [Ma et al. 2024]. Importantly, the variety of interpolants all follow an interpolation path $x_t = \sigma_t z + \alpha_t x_1$, differing only in the specific shape of the path specified by coefficients $\sigma_t, \alpha_t$, which means the problem of undersampling in the path space always exists.

While diffusion generative models are widely recognized for their robustness in modeling diverse distributions by transforming random noise, we note their lack of performance on structured data with insufficient samples (e.g. 3D shapes, 18K in PartNet [Mo et al. 2019b]) and slow convergence on unstructured data with large scale samples (e.g. images, 1.3M in ImageNet [Deng et al. 2009]).

Such deficiencies are shown extensively in Figs. 6, 7, 8 and Tab. 4 where unsync_none represents the linear one-sided interpolant model in Eq. (1), presenting failures in structured 3D shape generation and slow convergence in image generation. We hypothesize that these difficulties are related to the non-uniform sampling density of the path space, as analyzed next.

Sampling bias. We note that although diffusion models are trained to generate from noise, their inputs are more precisely samples on the path space from source to target distributions, which is much larger compared with fixed interpolation paths Eq. (1), and insufficient data samples tend to worsen the problem. Specifically, we show that the model by Eq. (1) creates non-uniform sampling density, where regions of the path space farther from the target data points are less densely covered during training.

Without loss of generality, suppose that the subspace $\mathcal{R}$ spanned by $z$ and $x_1$ has $z$ as the minimum corner and $x_1$ as the maximum corner, i.e., $\mathcal{R} = \{x|z \preceq x \preceq x_1\}$. As shown in Fig. 2 (a), we denote an arbitrary point $x \in \mathcal{R}$, and a moving point $p_t = (1-t)z + tx_1$ along the diagonal connecting $z$ and $x_1$.

We denote by $\rho(x)$ the probability density for sampling $x$, given by integrating all evaluations at $x$ of Gaussian distributions produced by interpolating the source noise and the target data point, i.e., $G_{p_t} = N(p_t; (1-t)^2 1)$ centered at $p_t$ and with variance scaled by the interpolation coefficient $1-t$. Therefore, we have

$$\rho(x) = \int_0^1 G_{p_t}(x) dt = \int_0^1 \frac{1}{\sqrt{2\pi}(1-t)} e^{-\frac{\|x-p_t\|^2}{2(1-t)^2}} dt.$$

Substituting $p_t$ with the parameterized equation $p_t = (1-t)z + tx_1$, we have

$$\rho(x) = \frac{1}{\sqrt{2\pi}} \int_0^1 \frac{1}{1-t} e^{-\frac{\|t(z-x_1)+x-z\|^2}{2(1-t)^2}} dt.$$

The above integration does not have closed-form solution, so we visualize $\rho$ by numerical integration in Fig. 2 (c,d,e), where we see the clear tendency of shrinking coverage toward data points. Moreover, we find the gradient $\nabla \rho$ is more friendly to work with. In particular, by straightforward calculation we have

$$(x_1 - x) \cdot \nabla \rho(x) = \frac{1}{\sqrt{2\pi}} e^{-\frac{\|x-z\|^2}{2}} > 0,$$

which means that $\nabla \rho(x)$ has a positive projection along the direction $x_1 - x$. Therefore, we can conclude that the sampling density $\rho(x)$ is not uniform and grows when approaching the target data points.

In Fig. 2(d) we visualize the sampling density of the standard flow matching formulation, where the shrinking density is clear. Additionally, in Fig. 3 the first row, we visualize the velocity field $u_t(x | x_1)$ and the probability density $p_t(x | x_1)$ in a toy example setting, where we observe non-converging outliers in the integrated trajectories. This toy example illustrates the potential effects of non-uniform sampling density in flow matching, where regions farther from target data points receive less training coverage.

Remark. The above analysis assumes a source unit normal distribution and a single target data point. When considering the target data distribution as a set of points with variance $\Sigma$, the parameterized Gaussian distribution $G_{p_t}$ has variance $(1-t)^2 1 + t^2 \Sigma$, which motivates the use of variance-preserving interpolants, such as the cosine/sine interpolant [Ma et al. 2024]. Nevertheless, the distribution of target dataset may not be large enough to cover the path space well. For data with many attributes and of high dimensions, the dataset sparsity problem is worsened by the curse of dimensionality. Indeed, we observe that while for images with sufficiently large datasets, mitigating the sampling bias mainly improves convergence, for small scale datasets like structured 3D shapes and sparse images, addressing the sampling bias is essential for training a working generative model (Sec. 5.2, Sec. 5.4.3).

To address this biased sampling problem, in the next section we show that by simply desynchronizing the interpolation schedule and sampling all possible combinations of attributes and feature

dimensions, the sampling density becomes uniform within the sub-regions $\mathcal{R}$, thus enabling robust generation in the low-data regime and faster convergence in the rich-data domain.

4 Combinatorial Stochastic Process

Most interesting data samples are high-dimensional. For example, state-of-the-art generative models encode images as latent patches with both spatial and feature dimensions [Ma et al. 2024; Peebles and Xie 2023]. 3D shapes structured as part ensembles include even more attributes in addition to spatial and feature dimensions, such as the varying numbers of parts, their bounding boxes, and their positions [Mo et al. 2019b]. Generating such data requires handling more flexible dimensions.

Regardless of the number of dimensions and attributes a data sample has, standard diffusion generative models treat them homogeneously and synchronously. For instance, in the case of stochastic interpolants, the generative model is trained on samples distributed according to densities with shrinking coverage along the transport paths connecting the source distribution to each target data sample, as illustrated in Fig. 2 (a), (d) and analyzed in Sec. 3. This design leaves the low density regions insufficiently trained, and once they are sampled in test stage by solving stochastic differential equations, the network can produce poor results.

To address the aforementioned problem, we emphasize the combinatorial complexity of individual dimensions and attributes of data samples. Specifically, we purposely sample points with asynchronous diffusion schedules for dimensions and attributes, as illustrated in Fig. 2 (b). Implementing these asynchronous schedules is relatively straightforward. We transform the interpolation schedule $t$ from Eq. (1) into a tensor $t$ of the same shape as $x$, using different values independently and uniformly sampled within $[0, 1]$ for the dimensions and attributes to obtain sample points:

$$x_t = (1 - t) \odot z + t \odot x_1$$

where $\odot$ denotes the elementwise product. We note that in contrast to the biased sampling of the standard model (Fig. 2), the sampling density of ComboStoc is uniform within the subregions spanned by each pair of source and target data points by construction.

The benefits of using these augmented samples from combinatorial stochasticity are threefold:

1. Ensuring broader network coverage compared to the synchronized schedule, resulting in more robust and higher quality performance during the testing stage.
2. Encouraging the network to learn the correlations among different dimensions and attributes, as it is trained to synchronize them to reach the final data points.
3. Enabling more flexible control over the generation process, allowing different dimensions and attributes to be given varying degrees of finalization in the synthesized final result.

In Fig. 3, we visualize the velocity field $u_t(x \mid x_1)$ and the probability density $p_t(x \mid x_1)$ in a toy example setting. The velocity field $u_t(x \mid x_1)$ is obtained by summing velocities (Sec. 5.4.1) over all $x_0$-derived spans from Fig. 2 (c,d,e), which does not admit a closed-form expression in this case. Therefore, we additionally perform particle-based simulations to illustrate how noisy samples move toward the data points as the induced velocity field evolves. As shown, our velocity field exhibits broader spatial coverage, resulting in fewer outliers and more concentrated particle trajectories throughout the convergence process.

Next, we discuss the detailed adaptations and achieved effects through generative tasks from two different domains, namely images and structured 3D shapes.

4.1 Images

For image generation, we take on the baseline of SiT [Ma et al. 2024] which applies highly scalable transformer networks and achieves state-of-the-art performance on ImageNet scale generation. In particular, a given image is encoded via the VAE encoder from Rombach et al. [2022] as a latent image $x_1$ of shape $C \times H \times W$, and the network is trained to predict velocity given the diffused latent image $x_t$ (Eq. (1)) and optionally conditioned on the image class $c$ and interpolation schedule $t$, i.e., $f_0(x_t; c, t) = x_1 - z$.

![Fig. 4. Illustration of compensation drift. When the network is trained to predict velocity $x_1 - z$ at an off-diagonal sample point $x_0$, a compensation drift ($v_{cmpn}$ in green) can be applied to pull the trajectory back to the diagonal.](image)

Correspondingly, we make several simple adaptations to implement the ComboStoc scheme. First, we construct $t$ with the same shape of $C \times H \times W$, and update the timestep embedding module of SiT to accommodate this change (see Sec. 5.2 and Fig. 5). Note that the conditioning on class labels and timesteps are mixed and implemented as modulation operations in SiT [Ma et al. 2024], and therefore are not symmetric to the data samples in importance$^1$. Second, importantly, we note that for velocity prediction, the samples with asynchronous $t$ should not predict the original velocity $x_1 - z$ only; otherwise there will be drift off the target data points during test stage integration, as illustrated by Fig. 4 the dotted line. Then we provide a formal analysis of this problem.

Proof that the ComboStoc Scheme Defines a Proper Generative Flow Model. Our scheme defines a conditional vector field $u(x|x_0, x_1)$ whenever $x \in \text{span}(x_0, x_1)$, where $\text{span}(x_0, x_1)$ is the rectangular subspace spanned by $x_0, x_1$. In comparison, FlowMatching [Lipman et al. 2023] defines the conditional vector field $u'(x|x_0, x_1)$ whenever $x \in \text{diag}(x_0, x_1)$, where $\text{diag}(x_0, x_1)$ is the line connecting $x_0, x_1$ and the diagonal of $\text{span}(x_0, x_1)$. Following FlowMatching, we show that $u(x|x_0, x_1)$ marginalized over $x_0, x_1$ generates the probability path $p_t$, that connects $x_0 \sim p_0 = q(x_0)$ and $x_1 \sim p_1 = r(x_1)$.

Precisely, we show that $p_t$ and $u_t$ satisfy the continuity equation:

$$\frac{d}{dt} p_t(x) = \iint \left( \frac{d}{dt} p_t(x|x_0, x_1) \right) q(x_0) r(x_1) dx_0 dx_1$$

$$= - \iint \text{div} \left( u(x|x_0, x_1) p_t(x|x_0, x_1) \right) q(x_0) r(x_1) dx_0 dx_1$$

$$= - \text{div} \left( \iint u(x|x_0, x_1) p_t(x|x_0, x_1) q(x_0) r(x_1) dx_0 dx_1 \right)$$

(6)

where in the first equality we expand the probability into integration over $x_0, x_1$; in the second equality we use the fact that $u(x|x_0, x_1)$ generates $p_t(x|x_0, x_1)$, a point distribution that moves from $x_0$ to $x_1$; in the third equality we switch the order of integration and differentiation based on the regularity of integrands; and in the last equality we simply apply the definition of marginalized vector field, i.e.

$$u_t(x) = \frac{1}{p_t(x)} \iint u(x|x_0, x_1) p_t(x|x_0, x_1) q(x_0) r(x_1) dx_0 dx_1.$$

In the above derivations, the major difference from FlowMatching is that we replace the $x_1$-conditioned vector field and probability density from FlowMatching with the $x_0, x_1$-conditioned vector field and probability density, where $u(x|x_0, x_1)$ is a time-invariant vector field that moves the point distribution from $x_0$ to $x_1$ by construction and thus generates $p_t(x|x_0, x_1)$ (Fig. 2). Note that $u(x|x_0, x_1)$ is time-invariant for both FlowMatching and our analysis; the apparent time dependence in FlowMatching arises only after marginalization over $p(x_0)$. We also note that the timestep $t$ in the continuity equation above is a scalar integration variable: while during training the path space is sampled via vectorized timestep interpolation (Eq. 5), the analysis operating in the standard scalar-time framework shows that the resulting marginalized velocity field $u_t(x)$ still generates the correct probability path $p_t$. The issue of off-diagonal drift observed in Fig. 4 can also be understood from the perspective of test-time integration. A detailed discussion is provided in Sec. 5.4.1.

4.2 Structured 3D shapes

We use the generative modeling of structured 3D shapes [Wang et al. 2025] as a new task to further demonstrate the importance of exploiting combinatorial complexity. Indeed, structured 3D shapes have even stronger combinatorial complexity than images, as shown in its varying numbers of parts, their positions and bounding boxes, as well as the detailed shape variations for each part. Precisely, we denote a structured 3D object as a collection of object parts, i.e., $x = \{p_i\}, i \in [L]$, where we set $L = 256$ to cover the maximum number of parts in a dataset. An object part is further encoded as $p = (s, b, e)$, where $s \in [0, 1]$ indicates the existence of this part, $b = (x, y, z, l, w, h)$ denotes the bounding box center $(x, y, z)$ and length $l$, width $w$ and height $h$, and $e \in \mathbb{R}^{512}$ is a latent shape code encoding the part shape in normalized coordinates. Note that under this representation, a permutation of the part indices does not change the 3D shape, which is quite different from images represented as a feature grid of fixed order and size.

To generate structured 3D shapes with semantic parts, we train a stochastic interpolant model. In particular, given a structured 3D shape $x_1 = \{p_i\}$ and its diffused sample $x_t$ (Eq. (5)), we make the

network predict the target data sample directly, i.e., $f_{\theta}(\mathbf{x}_t; c, t) = \mathbf{x}_1$, where $c$ is the optional class label of the 3D shape. We note that for image generation we follow the standard SiT configuration of velocity prediction ($v$-prediction) to isolate the effect of ComboStoc, while for structured 3D shapes we adopt $x$-prediction as a more robust design choice for the heterogeneous representation that mixes existence indicators, bounding boxes, and shape codes. As noted by Li and He [2025], both velocity and $x$-prediction are viable prediction targets in flow-based generative models. Note that here $t$ assigns different time steps for all the different attributes and dimensions of each object part. We validate the generative model for structured 3D shapes by training on the PartNet [Mo et al. 2019b] dataset, as discussed in Sec. 5.

5 Results and Discussion

In this part, we show that ComboStoc improves the training convergence of diffusion generative models for both images and structured 3D shapes (Sec. 5.2). We also demonstrate the novel applications enabled by the asynchronous time steps of ComboStoc (Sec. 5.3). We emphasize the distinction between the training and inference contributions of ComboStoc: the standard generation quality improvements reported in Sec. 5.2 (Figs. 6–7, Tab. 4) are obtained using synchronized timesteps during inference, identical to the baseline SiT inference procedure. Gains arise purely from the asynchronous training scheme. The asynchronous inference mode is only employed in Sec. 5.3 and Sec. 5.4.2, where it provides the graded control interface for downstream applications such as soft inpainting and part-level assembly.

5.1 Implementation Details

The image generation model is modified from SiT-XL/2, i.e., the large model with 28 layers, 1152 hidden dimension, $2 \times 2$ patch size, and 16 attention heads. We trained the model using the default settings of SiT, with AdamW solver and fixed learning rate $10^{-4}$, and batch size 256, on 4 Nvidia H100 gpus. The training takes 7.5 days for 800K iterations. Evaluating the models uses the SDE integrator with 250 steps. The use of classifier-free guidance (CFG) or not is specified at corresponding results. For comparison with baselines in terms of FID-50K, CFG is not used unless otherwise specified. In the result gallery Fig. 1, CFG is used with guidance strength 4.0.

The structured 3D shape generation model uses a network of SiT small model, i.e., the model has 12 layers, 384 hidden dimension, 256 tokens for parts and 6 attention heads. We trained the model using the AdamW solver with a fixed learning rate of $10^{-4}$ and batch size 16. We trained the model on 4 Nvidia A100 gpus, which takes 3 days for 1.5K epochs. Evaluating the models uses iterative sampling with 500 iterations; in each iteration, the predicted part existence is binarized via threshold 0.5 before being diffused back for the next iteration. Class conditional sampling without CFG is always applied. During typical inferences, we use synchronous timesteps (i.e., the tensorized timesteps have an identical value); in applications introduced in Sec. 5.3, we employ asynchronous timesteps for different applications.

Algorithm 1 Inference with Synchronous and Asynchronous Timesteps

Require: Trained velocity model $f_{\theta}(\mathbf{x}_t; c, t)$; scalar time schedule $\{t_k\}_{k=0}^K$ with $t_0 = 0, t_K = 1$; number of steps $K$ (e.g., $K=250$).

(A) Synchronized Inference

1. Sample source noise $z \sim N(0, 1)$.
2. Initialize $\mathbf{x}^{(0)} \leftarrow \mathbf{z}$.
3. for $k = 0$ to $K-1$ do
4. $t_k \leftarrow \frac{k}{K}$.
5. Construct synchronized time tensor:
$t^{(k)} \leftarrow t_k \cdot 1$
6. Entries $t^{(k)}$ share the same scalar $t_k$.
7. Predict velocity at this synchronized time:
$\hat{v}^{(k)} \leftarrow f_{\theta}(\mathbf{x}^{(k)}; c, t^{(k)})$
8. Standard SiT-style numerical integration:
$\mathbf{x}^{(k+1)} \leftarrow \text{SITSTEP}(\mathbf{x}^{(k)}, \hat{v}^{(k)}, t_k, t_{k+1})$
9. end for
10. return $\mathbf{x}^{(K)}$ as the generated sample.

(B) Asynchronous Inference with Graded Control

Require: Optional observed sample $\mathbf{x}_1$; mask $\mathbf{m} \in [0, 1]^{\text{shape}(\mathbf{x})}$ ($m_i$ specifies the degree of preservation for entry $i$, i.e., the initial timestep $\mathbf{t}^{(0)} = \mathbf{m}$).
1. Sample source noise $z \sim N(0, 1)$.
2. Initialize the starting point:
$\mathbf{x}^{(0)} = (1 - \mathbf{m}) \odot \mathbf{z} + \mathbf{m} \odot \mathbf{x}_1$.
3. Initialize the asynchronous timestep tensor:
$\mathbf{t}^{(0)} = \mathbf{m}$, $\Delta t = \frac{1 - \mathbf{m}}{K}$.
4. Entries with larger $m_i$ start closer to $\mathbf{x}_1$ and have smaller step size $\Delta t$.
5. for $k = 0$ to $K-1$ do
6. Model consumes a fully asynchronous time field $\mathbf{t}^{(k)}$:
$\hat{v}^{(k)} \leftarrow f_{\theta}(\mathbf{x}^{(k)}; c, t^{(k)})$
7. Update sample via the standard SiT integrator:
$\mathbf{x}^{(k+1)} \leftarrow \text{SITSTEP}(\mathbf{x}^{(k)}, \hat{v}^{(k)}, t^{(k)}, t_{k+1})$
8. Update the vectorized timestep:
$\mathbf{t}^{(k+1)} \leftarrow \min(1, \mathbf{t}^{(k)} + \Delta t)$
9. end for
10. return $\mathbf{x}^{(K)}$ as the graded control result.

We provide the pseudocode of the ComboStoc inference procedure in Alg. 1. For standard generation tasks such as unconditional image synthesis or 3D shape generation, we employ the synchronized schedule in Alg. 1 (A), where a single scalar timestep is uniformly interpolated into 250 values and broadcast to all dimensions during the denoising process.

For hierarchical control and inpainting applications in Sec. 5.3, we use the asynchronous schedule in Alg. 1 (B). In this case, the process does not start from pure noise. Instead, we construct the initial state via $\mathbf{x}^{(0)} = (1 - \mathbf{m}) \odot \mathbf{z} + \mathbf{m} \odot \mathbf{x}_1$, where the mask $\mathbf{m} \in [0, 1]^{\text{shape}(\mathbf{x})}$ encodes the desired degree of preservation for each dimension. Each entry of $\mathbf{m}$ simultaneously determines the starting point $\mathbf{x}^{(0)}$ and its

corresponding initial timestep $t_i^{(0)} = m_i$. The remaining trajectory from $t_i^{(0)}$ to 1 is then uniformly interpolated into 250 steps.

As a result, different dimensions follow different effective timestep schedules, larger $m_i$ values evolve more slowly, while smaller ones evolve more freely, yet all dimensions complete their evolution within the same 250 integration steps. This unified framework enables flexible graded control and smooth spatially varying inpainting within a single generative process. We present the results of synchronous inference in Sec. 5.2, explore various applications enabled by asynchronous inference in Sec. 5.3, and report ablation studies comparing the uniform step size and uniform step number strategies in Sec. 5.4.2.

5.2 Improved Training of Diffusion Models

We explore the combinatorial complexities for both images and structured 3D shapes, and build corresponding configurations which exploit these complexities to compare with baseline configurations that do not apply asynchronous time schedules. We show that the different configurations improve over baseline configurations universally; in addition, the stronger is the combinatorial complexity, the more important our scheme is for training a working model.

Images. Following SiT [Ma et al. 2024], we train on ImageNet [Deng et al. 2009] for class-conditioned image generation. To fully explore the effects of combinatorial stochasticity, we enumerate four settings with different levels of combinatorial flexibility in diffusing the data samples (see also Tab. 2 (a)). In particular, we use unsync_none, unsync_patch, unsync_vec, and unsync_all to denote no splitting of time steps, using different time steps for latent image pixels, for latent image channels, and for both image pixels and channels. We run the different settings on top of the SiT-XL/2 baseline model. Considering the difficulty posed by ImageNet [Deng et al. 2009] data size, in each batch we apply the split time steps only to half of the samples and leave the other half unchanged with synchronized time steps, which balances between samples along and off diagonal paths (Fig. 2). This mixed strategy serves as a form of curriculum learning: for large-scale datasets, fully asynchronous training can increase early optimization difficulty, so retaining a portion of synchronized samples helps stabilize convergence. This is consistent with other diffusion model strategies that employ non-uniform timestep schedules [Li and He 2025]. In contrast, for the relatively small PartNet dataset (Sec. 5), we apply asynchronous schedules to all samples without mixing, as the smaller data scale does not require such stabilization. Plots of FID-50K [Heusel et al. 2017] with respect to training steps are shown in Fig. 6 (a, b), where classifier-free guidance is not used and all images are sampled by SDE sampling (velocity field can be used for SDE sampling, via Eq. (4) in the SiT [Ma et al. 2024] paper).

Given a tensorized timestep $t$ of shape $(N, C, H, W)$ that is the same as the latent encoding of input images, we not only encode each of the timesteps for different dimensions as done before via frequency transform [Vaswani et al. 2017], but also embed the result feature map of timesteps in the same way as image embedding, i.e., the patch-wise embedding originally from ViT [Dosovitskiy et al. 2021]. This design ensures that the different dimensions are conditioned on their corresponding timesteps in addition to the shared class label. In particular, Fig. 5 (a) shows the original SiT timestep embedding module, where $N$ is the batch size, $C_F$ is the length of the sine/cosine frequency embedding, and $C_H$ is the hidden dimension of SiT transformer. Fig. 5 (b) shows the adapted timestep embedding module for ComboStoc. $t$ is now of shape $(N, C, H, W)$. The first two layers remain the same as the original module, applying to each entry of tensor $t$ and producing a compressed timestep encoding of dim $C_C$. Given the result tensor of shape $(N, C, H, W, C_C)$, we further transpose it to combine the channel dimensions and use the same patchwise embedding layer (but different parameters) as SiT and ViT to embed the local patches into vectors of dim $C_H$. Assuming the patch size is $L \times L$, then $T = H \times W/L^2$. Note that to avoid...

introducing large embedding layers, we have used $C_C = 4$ to encode a timestep scalar, which is significantly smaller than the $C_H = 1152$ of SiT. This can be the reason why unsync_none performs slightly worse than the baseline SiT, both of which have exactly the same architecture elsewhere. In Tab. 3 we give the details of split timestep specifications for all configurations, across images and structured 3D shapes. We rely on the broadcast semantics of Numpy [Harris et al. 2020] and Pytorch [Paszke et al. 2019] to assign synchronized timesteps to multiple dimensions.

As shown by the quantitative results in Fig. 6 (a), our scheme (using unsync_all) shows consistent improvement of the baseline SiT model, and significant improvement over the reference DiT model. Since ComboStoc has a slightly slower per-step training speed than SiT due to the tensorized timestep embedding (see Sec. 5.5), we also provide a wall-clock time comparison in Fig. 6 (c), which confirms that the improvement holds under a fair time budget. Second, as shown in Fig. 6 (b), the different settings of time step unsynchronization behave differently. Overall, the finest split by unsync_all obtains the best performances consistently, followed by unsync_vec and unsync_patch which split along feature and spatial dimensions and have almost indistinguishable performances. The worst performance is obtained by unsync_none, i.e., the setting using no combinatorial stochasticity. Fig. 7 visualizes the results of different settings along training steps, where we see better structured images emerge earlier for settings using stronger combinatorial complexity. Specifically, at 200K iterations, the unsync_all setting already produces stable and coherent image structures, whereas the other configurations still exhibit notable fluctuations as training progresses. For instance, under the unsync_vec and unsync_patch settings, the facial features of Siamese cats undergo substantial changes between 200K and 600K iterations. Even in the unsync_none setting, the eyes and facial regions of the Siamese cats remain noticeably less stable compared with the unsync_all configuration. The comparison among these four settings shows that fully utilizing the combinatorial complexity indeed helps network training. We reiterate that all FID results in this subsection are obtained with synchronized inference, confirming that the quality gains stem from the asynchronous training scheme alone, without requiring asynchronous inference.

Due to the smaller timestep embedding module, unsync_none has slightly worse performance than the baseline SiT. While it may be possible to align unsync_none with baseline SiT by introducing more capable embedding layers, unsync_all already outperforms the baseline with significant margins (Fig. 6 (a)). In addition to the result quality, in Sec. 5.5 we provide detailed analysis of the computational complexity of our model in comparison with baseline SiT and DiT models, and find that our model is as efficient as the baselines in actual runtime despite a moderate increase in GFlops.

Structured 3D shapes. We show that for the task of structured 3D shape generation, which has even stronger combinatorial complexity due to the flexible parts and their multiple attributes, our scheme becomes more important to the extent of being indispensable.

We have adopted the pretrained part shape encoding network from Wang et al. [2025]. In particular, Wang et al. [2025] design a point cloud VAE to encode 3D shapes into a sparse set of latent codes, and on top of the latent set, they train another transformer VAE to compress them into a single latent code. Therefore, each part shape from the PartNet dataset [Mo et al. 2019b] is normalized into unit size and encoded into a single code, which allows us to represent structured 3D shapes as a collection of parts. The embedding modules for part existence and bounding box follow the same design as timestep embedding. That is, we first turn each of the scalar dimensions into frequency codes using the sine/cosine embedding, and then embed them into vectors of dim 4 (cf. Fig. 5), before finally embedding each of the collective attributes as a whole into vectors of hidden dim 384, through respective FC layers.

For structured 3D shape generation, we identify combinatorial complexity in the following axes: attributes/feature vectors, and spatial parts. Therefore, we obtain $3 \times 2 = 6$ settings, i.e., unsync_none and unsync_part which apply the same or different time schedules to parts respectively, unsync_att and unsync_att_part which use attribute level schedules, and unsync_vec and unsync_all which use the most finely divided feature vector level schedules. See Tab. 2 (b) for a summary of the 6 configurations. Because of the relatively small size of the PartNet dataset (18K shapes in total, mostly in chair and table classes), we deem it easier to learn and simply apply the corresponding asynchronous time steps to all samples in each batch, in contrast to the mixing scheme of ImageNet training. We report results at 1.5K epochs, since earlier results cannot be decoded into valid manifold shapes for evaluation in settings like unsync_none.

As shown in Fig. 8, the more combinatorial complexity we exploit, the better the performance of the trained network. In comparison, the baseline setting without combinatorial stochasticity, unsync_none, almost entirely fails to produce meaningful shapes. Moreover, since this task models the highly flexible composition of various numbers of parts, applying the spatial part unsynchronization (Tab. 2 (b)) helps obviously, as shown through the three pairs.

of columns in Fig. 8 (e.g., part vs none, att_part vs att, and all vs vec.).

We report quantitative results in Tab. 4 using the chair category. Following Wang et al. [2025] we use three metrics, including Frechet Point Distance (FPD) that measures the FID on sampled point clouds, coverage (COV) that measures how well each ground truth sample is covered by the closest generated sample, and minimum matching distance (MMD) that measures how well each generated sample resembles the closest GT sample. The numerical results again show that the part level combinatorial stochasticity enhances generative

performance significantly, and unsync_all shows the best overall result.

Tab. 5 gives the comparison between our structured 3D shape generation model and two baselines, i.e., StructRe [Wang et al. 2025] and StructureNet [Mo et al. 2019a], in terms of FPD, COV and MMD. Shapes in PartNet [Mo et al. 2019b] are labeled into semantic parts that are organized into trees, i.e., coarse parts can be decomposed into fine parts by following the tree. Exploiting this hierarchical data, the two baselines expand coarse parts into fine parts progressively, which helps constrain the generated shapes toward better regularity. In comparison, our network does not use this hierarchical information and directly generates the leaf level parts. Nevertheless, the results by unsync_all show performances within the baseline results. Visually, we find our results generally show stronger diversity than the shapes by Wang et al. [2025] and Mo et al. [2019a]. Finally, it is an interesting topic to study how to combine the approaches of hierarchical generation and diffusion generative models, which have differing advantages in aspects of structure regularity and diversity. In Fig. 9 we show more random samples generated by the unsync_all setting.

5.3 Applications

The asynchronous timesteps for different dimensions and attributes of ComboStoc enables a novel test-time application, namely the ability to specify different degrees of preservation of a data sample to its dimensions and attributes. Specifically, given $t_0$ specifying the weights in [0, 1] to preserve the data of $x$, we sample the generative process starting from $x_0 = (1 - t_0) \odot z + t_0 \odot x$, and increase the time steps for individual dimensions and attributes via $\frac{1 - t_0}{N}$ for $N$ steps. Examples of such asynchronous generative processes are shown in Figs. 10, 11, 12, 13 for images and Figs. 14, 15 for structured 3D shapes.

Images. In Fig. 10 we show that giving different $t_0$ to a half of a reference image while leaving the other half to generate from scratch, we can achieve different degrees of preservation of the reference images. In particular, as the preservation weight increases from 0 to 1, the preservation of reference content is strengthened. The first example of the red-panda perfectly illustrates the difference in constraint capability caused by the size of $t_0$. As $t_0$ increases, the area on the left becomes closer and closer to $x_1$, especially the background area in the zoom-in window. We note that at [0.4, 0.7] the weight is good enough to preserve most of the reference content, and if the $t_0$ is too big like 0.9, the smooth transition between the two parts will be degraded, since the time steps on the left are too concentrated (only from 0.9 to 1.0).

In Fig. 11, we use different preservation weights encoded by $t_0$ for the four quadrants of each image, and show intermediate results along the iterative SDE integration process. From the three examples we can see that stronger weights cause better preservation of reference regions, and the different regions are filled with coherent content despite the spatially varying time schedules. This mode of controlled generation is novel, compared with the binary inpainting mode proposed for standard diffusion models [Lugmayr et al. 2022], where regions of an image are divided into two discrete types, i.e., those to preserve and those to generate from scratch. Fig. 12 further demonstrates this capability with smooth, spatially continuous $t_0$ maps. Rather than the piecewise-constant weights in Figs. 10 and 11, here we use a smoothly varying $t_0$ map centered on each subject, with preservation strength ranging continuously from $\approx 0.85$ at the center to 0 at the periphery. Unlike binary-mask inpainting [Lugmayr et al. 2022], ComboStoc supports such continuous spatial control natively, without any task-specific training. The generated results show that the model faithfully preserves the central subject while coherently generating diverse surroundings and backgrounds, with natural transitions between preserved and generated regions. This smooth, continuous control is a unique advantage of ComboStoc’s asynchronous timestep formulation.

We also find that channel-varying $t_0$ reveals interesting observations about the different contents of latent image encoding [Rombach et al. 2022]. Fig. 13 shows another example of image generation where we use varying degrees of data preservation across both spatial dimensions (the four quadrants) and feature channel dimensions. In particular, we assign spatial preservation weights according to the left column in the figure, and additionally assign 0.5 to the specified channel index $C$ and 0 to other channels, as shown in the middle four columns. Interestingly, we see that the different channels of the stable-diffusion VAE latent space [Rombach et al. 2022] have very different content. For $C = 0$ the first channel, the generated results mostly preserve the spatial structures of the reference images, and the color cues are largely lost. From $C = 1$ to $C = 3$, the generated results increasingly preserve the color cues of the reference images.

but lose more of the structures. The findings suggest that earlier channels of the VAE latent space emphasize on structures and later ones on image-level color distributions.

Structured 3D shapes. By controlling different parts and attributes of structured 3D shapes, we can achieve diverse effects, including shape completion and part assembly. In Fig. 14, we fix the bases of chairs by giving them $t_0 = 0.9$, and complete them with meaningful but diverse structures that satisfy the class condition. The given bases have the chance of being slightly updated to adapt to the completed shapes. In Fig. 15, we randomly position a set of parts, and let the network arrange them into proper shapes, by giving the part

Fig. 10. Image generation using different weights of preservation. Each reference image $x_1$ (right) is split into two vertical halves (left), and the left half is given the preservation weights while the right region starts from scratch.

Fig. 11. Image generation with spatially different preservation weights. As shown in the left column, the four quadrants use $t_0 = 0, 0.25, 0.5, 0.75$, respectively. The sampling iterations converge to results that preserve the corresponding quadrants from the reference images (right) differently.

Fig. 12. Soft inpainting with spatially continuous $t_0$ maps. The first column shows the source image and the second column shows the $t_0$ map (each pixel corresponds to an $8 \times 8$ latent patch), which varies smoothly from $\approx 0.85$ (bright, strongly preserved) to 0 (black, freely generated). The remaining columns show diverse inpainting results: the model preserves the subject according to the graded mask while coherently generating diverse surroundings.

Fig. 13. Spatially and channel varying $t_0$. For the spatial dimensions $t_0[;; i, j]$, the assignment is specified in the left column. For the feature channel dimensions $t_0[;; C, ;;]$, the $C$-dim is given 0.5 and the rest given 0. Therefore, we obtain results that correspond to the reference images $(x_1)$ in complex ways. Notably, earlier channels correspond more to image structures and later channels to image colors.

shape codes $e$ and bounding box sizes large preservation weights ($t_0 = 0.9$) and making the rest attributes free to be generated. Here we have considered a simplified setting where the part rotations are given, and leave the more challenging case of rotating shape parts as future work.

5.4 More Discussions

5.4.1 Minimizing the Off-diagonal Drift. The problem of off-diagonal drift can be revealed from the perspective of test-time integration. For ease of analysis we assume an ODE is solved to follow the prescribed velocity field to move from the source noise point to the target data point.

If the integration never stumbles on sample points off diagonal line connecting $z$ and $x_1$, the integration starting from a point $x_{t_0} = z + t_0(x_1 - z)$ and following the velocity field $x_1 - z$ would always end at the target point, i.e.,

$$x_{t_0} + \int_{t_0}^{1} (x_1 - z) dt = z + t_0(x_1 - z) + (1 - t_0)(x_1 - z) = x_1.$$ (8)

Now suppose the integration stumbles upon an off-diagonal sample point $x_{t_0}$ at a tensorized interpolation schedule $t_0$ with different values for its various entries. The integration by following only the velocity $x_1 - z$ would miss the target data point $x_1$ (Fig. 4, dotted arrow). Precisely,

$$x_{t_0} + \int_{t_0}^{1} (x_1 - z) dt$$

$$= z + t_0 \odot (x_1 - z) + (1 - t_0)(x_1 - z)$$

$$= x_1 + (t_0 - t_0) \odot (x_1 - z),$$

where $t_0 = \min(t_0)$ denotes the minimum interpolation schedule across the dimensions of $t_0$. Here we assume the test-time setting that for asynchronous timesteps we integrate until the slowest one finishes.

Table 6. Ablation study on velocity adaptation methods. We compare no compensation, off diagonal drift minimization, and cone velocity on the first 100 ImageNet [Deng et al. 2009] categories. FID evaluates generation quality, while SSIM measures structural preservation under asynchronous generation with $t_0 = 0.75$ in Fig. 16.

| Approaches | w/o Cmpn | Off-diagonal Drift | Cone Velocity |
| :--- | :--- | :--- | :--- |
| FID $\downarrow$ | 103.75 | 103.01 | 113.59 |
| SSIM $\uparrow$ in Fig. 16 | 0.255 | 0.262 | 0.224 |

To address this divergence problem, we propose a velocity compensation approach for mitigation, where the compensated component minimizes the off-diagonal drift by following the negative gradient of a drift potential, as discussed next.

Off-diagonal Drift Minimization. The off-diagonal offset vector $\delta(x_t)$ can be defined as

$$\delta(x_t) = -v_{cmpn} = x_t - x_1 - \frac{(x_t - x_1) \cdot (x_1 - z)}{||x_1 - z||^2}(x_1 - z).$$ (10)

To minimize the drift, we can simply follow its negation $v_{cmpn}$ in addition to the original velocity during integration, which is equivalent to minimizing a drift potential $\Phi(\delta(x_t)) = \frac{1}{2} \| \delta_t \|^2$ by gradient descent, and promotes the convergence to target data points (Fig. 4, dashed arrow).

To verify the effectiveness of our proposed method, we also tried another intuitive velocity adaptation method, where we design a cone-shaped velocity field such that its integration converges to target data points, as discussed next.

Cone-shaped Velocity Field. Different from the off-diagonal drift minimization, we can also design a cone-shaped velocity field that generalizes the simple constant velocity $x_1 - z$ to a cone of velocities covering the expanded region $R$. In particular, we can use the following velocity

$$v_{t_0} = \frac{x_1 - x_{t_0}}{1 - t_0},$$ (11)

where again $t_0 = \min(t_0)$ denotes the minimum interpolation schedule across the dimensions of $t_0$. Note that for synchronized schedule, $v = \frac{x_1 - x_{t_0}}{1 - t_0} = \frac{x_1 - z - t_0(x_1 - z)}{1 - t_0} = x_1 - z$, i.e., the original velocity is a special case of this velocity field. To see why this is a cone shaped velocity field, note that for a timestep $t_\lambda = \lambda t_0 + (1 - \lambda)1$ along the line of $t_0$ and 1, their velocities are equal. Therefore, it is easy to see such a constant velocity along the line connecting an off-diagonal point and the target point would lead to convergence to the target data point. Precisely,

$$x_{t_0} + \int_{t_0}^{1} \frac{x_1 - x_{t_0}}{1 - t_0} dt = x_{t_0} + x_1 - x_{t_0} = x_1,$$ (12)

due to the cone-shaped velocity field. On the other hand, we note that the cone-shaped velocity introduces significant scaling by normalizing against the slowest time dimension, especially when the dimensions are scheduled very differently. This lack of regularity may cause the degraded regression of the velocity field, as demonstrated in later experiments.

Fig. 16. Comparison of velocity adaptation methods in graded image generation using different preservation weights. Each reference image $x_1$ is split into two vertical halves, with the right half assigned preservation weights while the left half starts from scratch. Rows (a), (b), and (c) correspond to no compensation, the off-diagonal drift minimization method, and the cone-shaped velocity field method, respectively.

Fig. 17. Training Loss and FID on Insufficient Data. Training curves of SiT-S/2 (32.58M parameters) and ComboStoc-S/2 (32.36M parameters) on a small-scale dataset consisting of 1,000 images sampled from ImageNet [Deng et al. 2009]. Despite using slightly fewer parameters, ComboStoc consistently achieves a lower training loss and a better FID throughout training. FIDs are computed on images generated using ODE sampling with a classifier-free guidance scale of 1.0.

Ablation Study. We conducted ablation experiments on the two different velocity adaptation methods using a subset of ImageNet (the first 100 categories). The models were trained for 50,000 steps on 8 Nvidia H20 graphics cards. The first row in Tab. 6 presents the FID values obtained with these two methods, as well as without any compensation. The results indicate that, compared to not using any compensation, our proposed off-diagonal drift minimization method achieves better convergence. In contrast, the cone-shaped velocity field results in poorer performance, which may be attributed to its scaled magnitude. While the FID improvement from drift compensation appears modest in Tab. 6, the visual impact is more pronounced: as shown in Fig. 16, without compensation, images exhibit obvious background discontinuities along the boundary between differently preserved regions, whereas the off-diagonal drift minimization produces a seamless transition. These artifacts are visually salient but difficult to capture with scalar metrics alone.

Additionally, similar to Fig. 10, we examined the differences among three methods in asynchronous generation, as illustrated in Fig. 16. We used different $t_0$ values for each side of the image. The results show that the off-diagonal drift minimization method produces a more seamless transition. In contrast, without any compensation, a noticeable discontinuity appears along the midline of the image. The cone-shaped velocity field method, which converges more slowly, yields poorer results. It is important to note that these methods were trained for only 50,000 steps on a subset of ImageNet, so the overall image quality is relatively low.

To obtain a quantitative evaluation metric for the graded generation, we computed the structural similarity (SSIM) between each pair of reference image $x_1$ and generated image using $t_0 = 0.75$, for the three configurations. We have randomly selected 5K reference images for evaluation. The second row of Tab. 6 presents SSIM scores, indicating that the off-diagonal drift minimization method better preserved the structural similarity and quality to reference images, in comparison to the other settings.

Based on the above observation, we have used the off-diagonal drift minimization approach to mitigate the divergence issue in training and all experiments.

5.4.2 Async timestep at inference stage. For the graded control applications on both image and 3D shape, currently we use different stepsizes but the same step number $N$ for individual dimensions. Therefore, regions of larger $t_0$ will go through the same $N$ iterations as those of smaller $t_0$ in all of the applications. This scheme differs from the formula in Sec. 4.1, where a uniform stepsize is applied to analyze the drift problem during training.

Meanwhile, we tested the uniform stepsize (different step number, US) scheme for graded control in Tab. 7 under the initial timestep setting in Fig. 16 but using our fully trained 800K steps model (not the 50,000 steps subset used in Fig. 16), and found the results are similar to the uniform step number (UN) scheme.

5.4.3 Ablation on Insufficient Data. To further evaluate the effectiveness of ComboStoc under data-scarce regimes, we conduct an ablation study on insufficient training data. We adopt the smallest SiT-S/2 model (32.58M parameters) as the baseline and construct

ComboStoc-S/2 on top of it, which has slightly fewer parameters (32.36M) due to different time embedding module. Both models are trained from scratch on a small subset of ImageNet [Deng et al. 2009] containing only 1,000 images in one category.

Fig. 17 presents the quantitative comparison between SiT-S/2 and ComboStoc-S/2 on the insufficient dataset. Although ComboStoc-S/2 contains slightly fewer parameters, it consistently achieves a lower training loss than SiT-S/2 throughout almost the entire training process. In terms of FID, SiT-S/2 performs slightly better at early stages, when the learned representations remain coarse. As training proceeds and more meaningful structure emerges, ComboStoc-S/2 progressively closes the gap and ultimately outperforms SiT-S/2, leading to a clear advantage in the later stages. This behavior indicates that once meaningful structure begins to emerge, ComboStoc is able to exploit structured information more effectively, leading to richer representations and improved generation quality under limited data.

Fig. 18 further illustrates this advantage through visual comparisons across training iterations. Using identical random seeds for both training and inference, we observe that ComboStoc produces higher-quality and more stable images at significantly earlier training stages. In contrast, SiT often remains blurry even after 200K steps in several cases, or exhibits chaotic outputs in early training and only converges to clear images after extensive optimization. Overall, ComboStoc demonstrates faster convergence and stronger robustness in low-data regimes, consistently yielding clear and semantically coherent samples midway through training.

5.5 Computational Complexity Analysis

We provide a comparison with SiT and DiT in Tab. 8, in terms of parameter count, training stage speed, memory usage, inference stage speed, and GFlops. All tests are done on a single Nvidia A100-80G GPU at the XL/2 model configuration, with an input image of size 256×256 and training batch size 256. The GFlops are calculated by DeepSpeed [Rasley et al. 2020].

From Tab. 8 we can see that compared with DiT and SiT, our model has a smaller number of parameters as we use a smaller timestep.

embedding module (see Fig. 5). Therefore, our GPU memory cost during training is slightly smaller than SiT. On the other hand, for the conditioning by class label and timestep implemented as a modulation operator (see Fig. 3 of the DiT paper for illustration [Peebles and Xie 2023]), our conditioning is a tensor of the same shape as the image tensor, in contrast to DiT/SiT’s conditioning by a vector only of the channel size of the image tensor; to produce the conditioning tensor involves more computation than the conditioning vector, so our model leads to more flops and slightly increased training cost per step. Nevertheless, the production of the conditioning tensor is a standard MLP feature transformation and fits nicely into GPU parallel computation, so the inference speed is not sacrificed in comparison with DiT/SiT. In summary, the per-step training slowdown (0.15 vs. 0.19 steps/sec for SiT) is a direct consequence of the tensorized timestep conditioning; however, as shown in Fig. 6 (c), ComboStoc still achieves better FID at the same wall-clock training time, indicating that the quality gains more than compensate for the per-step overhead.

6 Conclusion

We have proposed to focus on the problem of combinatorial complexity of high-dimensional and multi-attribute data samples (like 3D shapes and images) for diffusion generative models. In particular, we note that for one-sided stochastic interpolants that model many variants of diffusion and flow-based models, there exists the problem of under-sampling regions of the path space where the dimensions/attributes are off-diagonal or asynchronous. We propose to fix this issue by sampling the whole space spanned by combinatorial complexity uniformly. Experiments across two data modalities show that indeed by utilizing the combinatorial complexity, performances can be enhanced, and new generation paradigms can be enabled where different attributes of a data sample are generated in asynchronous schedules to achieve varying degrees of control simultaneously. We hope that our work can inspire future works that look through the combinatorial perspective of generative models.

Limitations and future work. Our ComboStoc scheme will only have significant effects when the data has combinatorial and structural information, such as different patches for images and different parts for 3D shapes. When the data resides in a vector space whose dimensions are nearly independent, it is hard to exploit the correlation of dimensions and train a model that works well under the combinatorial schedule of different dimensions. Indeed, in such a case the individual dimensions may ideally be generated separately by different models. However, we note that many data types in real life contain strong structural and combinatorial information; particularly eminent are tasks within scientific domains, including molecule docking and protein folding [Corso et al. 2022; Wu et al. 2024; Yim et al. 2024], where diffusion models that better handle their combinatorial structures can be desirable. Additionally, ComboStoc is not specifically designed for diffusion transformers. Instead, it can be easily applied to other networks like U-Net diffusers; we only need to tensorize the timestep and apply it to corresponding feature maps through elementwise modulation, where the modulation can be the standard conditioning (e.g. the affine transformation).

References

Michael S. Albergo, Nicholas M. Boffi, and Eric Vanden-Eijnden. 2023. Stochastic Interpolants: A Unifying Framework for Flows and Diffusions.

Michael Samuel Albergo and Eric Vanden-Eijnden. 2023. Building Normalizing Flows with Stochastic Interpolants. In International Conference on Learning Representations.

Boyuan Chen, Diego Marti Monsó, Yilun Du, Max Simchowitz, Russ Tedrake, and Vincent Sitzmann. 2024. Diffusion forcing: Next-token prediction meets full-sequence diffusion. Advances in Neural Information Processing Systems 37 (2024), 24081–24125.

Gabriele Corso, Hannes Stark, Bowen Jing, Regina Barzilay, and Tommi Jaakkola. 2022. Diffdock: Diffusion steps, twists, and turns for molecular docking. arXiv preprint arXiv:2210.01776 (2022).

Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. 2009. ImageNet: A large-scale hierarchical image database. In Computer Vision and Pattern Recognition (CVPR).

Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Housby. 2021. An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. International Conference on Learning Representations (2021).

Shanghua Gao, Pan Zhou, Ming-Ming Cheng, and Shuicheng Yan. 2023. Masked Diffusion Transformer is a Strong Image Synthesizer. In International Conference on Computer Vision (ICCV).

Hyojun Go, Yunsung Lee, Seunghyun Lee, Shinhyeok Oh, Hyeongdon Moon, and Seungtaek Choi. 2024. Addressing negative transfer in diffusion models. Advances in Neural Information Processing Systems 36 (2024).

Tiankai Hang, Shuyang Gu, Chen Li, Jianmin Bao, Dong Chen, Han Hu, Xin Geng, and Baining Gu. 2023a. Efficient diffusion training via min-snr weighting strategy. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 7441–7451.

Tiankai Hang, Shuyang Gu, Chen Li, Jianmin Bao, Dong Chen, Han Hu, Xin Geng, and Baining Gu. 2023b. Efficient Diffusion Training via Min-SNR Weighting Strategy. In International Conference on Computer Vision (ICCV).

Charles R Harris, K Jarrod Millman, Stefian J Van Der Walt, Ralf Gommers, Pauli Virtanen, David Cournaueau, Eric Wieser, Julian Tierney, Sebastian Berg, Nathaniel Smith, et al. 2020. Array programming with NumPy. nature 585, 7825 (2020), 357–362.

Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. 2017. GANs trained by a two time-scale update rule converge to a local nash equilibrium. In Neural Information Processing Systems. Curran Associates Inc.

Jonathan Ho, Ajay Jain, and Pieter Abbeel. 2020. Denoising Diffusion Probabilistic Models. In Neural Information Processing Systems, Vol. 33. Curran Associates, Inc.

Zijing Hu, Yunze Tong, Fengda Zhang, Junkun Yuan, Jun Xiao, and Kun Kuang. 2026. Asynchronous Denoising Diffusion Models for Aligning Text-to-Image Generation. In The Fourteenth International Conference on Learning Representations.

Jialei Huang, Guanqi Zhan, Qingna Zhan, Kaichun Mo, Lin Shao, Baoquan Chen, Leonidas Guibas, and Hao Dong. 2020. Generative 3D Part Assembly via Dynamic Graph Learning. In Neural Information Processing Systems (NeurIPS).

Jihwan Kim, Junoh Kang, Jinyoung Choi, and Bohyung Han. 2024. Fifo-diffusion: Generating infinite videos from text without training. Advances in Neural Information Processing Systems 37 (2024), 89834–89868.

Eran Levin and Ohad Fried. 2025. Differential diffusion: Giving each pixel its strength. In Computer Graphics Forum. Wiley Online Library, e70040.

Tianhong Li and Kaiming He. 2025. Back to basics: Let denoising generative models denoise. arXiv preprint arXiv:2511.13720 (2025).

Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matthew Le. 2023. Flow Matching for Generative Modeling. In International Conference on Learning Representations.

Anran Liu, Cheng Lin, Yuan Liu, Xiaoxiao Long, Zhiyang Dou, Hao-Xiang Guo, Ping Luo, and Wenping Wang. 2024. Part123: part-aware 3d reconstruction from a single-view image. In ACM SIGGRAPH 2024 Conference Papers. 1–12.

Xingchao Liu, Chengyue Gong, and Qiang Liu. 2023. Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow. In International Conference on Learning Representations.

Andreas Lugmayr, Martin Danelljan, Andres Romero, Fisher Yu, Radu Timofte, and Luc Van Gool. 2022. RePaint: Inpainting Using Denoising Diffusion Probabilistic Models. In Computer Vision and Pattern Recognition (CVPR).

Nanye Ma, Mark Goldstein, Michael S Albergo, Nicholas M Boffi, Eric Vanden-Eijnden, and Saining Xie. 2024. Sit: Exploring flow and diffusion-based generative models with scalable interpolant transformers. In European Conference on Computer Vision. Springer, 23–40.

Chenlin Meng, Yutong He, Yang Song, Jiaming Song, Jiajun Wu, Jun-Yan Zhu, and Stefano Ermon. 2021. Sdedit: Guided image synthesis and editing with stochastic differential equations. arXiv preprint arXiv:2108.01073 (2021).

Chenlin Meng, Robin Rombach, Ruiqi Gao, Diederik Kingma, Stefano Ermon. Jonathan Ho, and Tim Salimans. 2023. On Distillation of Guided Diffusion Models. In Computer Vision and Pattern Recognition (CVPR).

Kaichun Mo, Paul Guerrero, Li Yi, Hao Su, Peter Wonka, Niloy Mitra, and Leonidas Guibas. 2019a. StructureNet: Hierarchical Graph Networks for 3D Shape Generation.

ACM Trans. Graph. 38, 6 (2019).

Kaichun Mo, Shilin Zhu, Angel X. Chang, Li Yi, Subarna Tripathi, Leonidas J. Guibas, and Hao Su. 2019b. PartNet: A Large-Scale Benchmark for Fine-Grained and Hierarchical Part-Level 3D Object Understanding. In Computer Vision and Pattern Recognition (CVPR).

Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, et al. 2019. Pytorch: An imperative style, high-performance deep learning library. Advances in neural information processing systems 32 (2019).

Naama Pearl, Yaron Brodsky, Dana Berman, Assaf Zomet, Alex Rav Acha, Daniel Cohen-Or, and Dani Lischinski. 2023. Svrn: Spatially-variant noise removal with denoising diffusion. arXiv preprint arXiv:2306.16052 (2023).

William Peebles and Saining Xie. 2023. Scalable Diffusion Models with Transformers. In International Conference on Computer Vision (ICCV).

Jeff Rasley, Samyam Rajbhandari, Olatunji Ruwase, and Yuxiong He. 2020. DeepSpeed: System Optimizations Enable Training Deep Learning Models with Over 100 Billion Parameters (KDD ’20). Association for Computing Machinery, New York, NY, USA, 3505–3506.

Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. 2022. High-Resolution Image Synthesis With Latent Diffusion Models. In Computer Vision and Pattern Recognition (CVPR).

David Ruhe, Jonathan Heek, Tim Salimans, and Emiel Hoogeboom. 2024. Rolling Diffusion Models. arXiv:2402.09470 [cs.LG] https://arxiv.org/abs/2402.09470

Subham Sahoo, Aaron Gokaslan, Christopher M De Sa, and Volodymyr Kuleshov. 2024. Diffusion models with learned adaptive noise. Advances in Neural Information Processing Systems 37 (2024), 105730–105779.

Inkyu Shin, Chenglin Yang, and Liang-Chieh Chen. 2025. Deeply supervised flow-based generative models. arXiv preprint arXiv:2503.14494 (2025).

Kiwhan Song, Boyuan Chen, Max Simchowitz, Yilun Du, Russ Tedrake, and Vincent Sitzmann. 2025. History-guided video diffusion. arXiv preprint arXiv:2502.06764 (2025).

Yang Song, Prafulla Dhariwal, Mark Chen, and Ilya Sutskever. 2023. Consistency Models. In International Conference on Machine Learning.

Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. 2021. Score-Based Generative Modeling through Stochastic Differential Equations. In International Conference on Learning Representations.

Mingzhen Sun, Weinung Wang, Gen Li, Jiawei Liu, Jiahui Sun, Wanquan Feng, Shanshan Lao, SiYu Zhou, Qian He, and Jing Liu. 2025. Ar-diffusion: Asynchronous video generation with auto-regressive diffusion. In Proceedings of the Computer Vision and Pattern Recognition Conference. 7364–7373.

Minhyuk Sung, Vladimir G. Kim, Roland Angst, and Leonidas Guibas. 2015. Data-driven structural priors for shape completion. ACM Trans. Graph. 34, 6, Article 175 (nov 2015).

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Proceedings of the 31st International Conference on Neural Information Processing Systems (Long Beach, California, USA) (NIPS ’17). Curran Associates Inc., Red Hook, NY, USA, 6000–6010.

Jiepeng Wang, Hao Pan, Yang Liu, Xin Tong, Taku Komura, and Wenping Wang. 2025. StructRe: Rewriting for Structured Shape Modeling. ACM Trans. Graph. (April 2025). doi:10.1145/3732934

Kai Wang, Mingjia Shi, Yukun Zhou, Zekai Li, Zihang Yuan, Yuzhang Shang, Xiaojiang Peng, Hanwang Zhang, and Yang You. 2024. A Closer Look at Time Steps is Worthy of Triple Speed-Up for Diffusion Model Training. arXiv preprint arXiv:2405.17403 (2024).

Kevin E Wu, Kevin K Yang, Rianne van den Berg, Sarah Alamdari, James Y Zou, Alex X Lu, and Ava P Amini. 2024. Protein structure generation via folding diffusion. Nature communications 15, 1 (2024), 1059.

Jianfeng Xiang, Zelong Lv, Sicheng Xu, Yu Deng, Ruicheng Wang, Bowen Zhang, Dong Chen, Xin Tong, and Jiaolong Yang. 2024. Structured 3d latents for scalable and versatile 3d generation. arXiv preprint arXiv:2412.01506 (2024).

Xinhao Yan, Jiachen Xu, Yang Li, Changfeng Ma, Yunhan Yang, Chunshi Wang, Zibo Zhao, Zeqiang Lai, Yunfei Zhao, Zhuo Chen, et al. 2025. X-part: high fidelity and structure coherent shape decomposition. arXiv preprint arXiv:2509.08643 (2025).

Ling Yang, Zhilong Zhang, Yang Song, Shenda Hong, Runsheng Xu, Yue Zhao, Wentao Zhang, Bin Cui, and Ming-Hsuan Yang. 2023. Diffusion Models: A Comprehensive Survey of Methods and Applications. ACM Comput. Surv. 56, 4, Article 105 (Nov. 2023), 39 pages.

Jason Yim, Hannes Stark, Gabriele Corso, Bowen Jing, Regina Barzilay, and Tommi S Jaakkola. 2024. Diffusion models in protein structure and docking. Wiley Interdisciplinary Reviews: Computational Molecular Science 14, 2 (2024), e1711.

Sihyun Yu, Sangkyung Kwak, Huiwon Jang, Jongheon Jeong, Jonathan Huang, Jinwoo Shin, and Saining Xie. 2024. Representation alignment for generation: Training diffusion transformers is easier than you think. arXiv preprint arXiv:2410.06940 (2024).