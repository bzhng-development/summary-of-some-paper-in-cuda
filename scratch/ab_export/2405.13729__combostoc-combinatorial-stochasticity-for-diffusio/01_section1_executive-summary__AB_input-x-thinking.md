# Section 1 — Executive Summary — A/B

**2405.13729 — ComboStoc: Combinatorial Stochasticity for Diffusion Generative Models**


_input chars: PyMuPDF=90,744 · GLM-OCR=79,441_



## PyMuPDF input · THINK-HIGH

## 1. Executive Summary

This paper proposes **ComboStoc**, a training strategy that transforms the diffusion timestep into a tensor of the same shape as the data and applies asynchronous values across dimensions and attributes to fully sample the combinatorial complexity of the data (e.g., per-patch and per-channel timesteps for images—denoted `unsync_all`—and per-part and per-attribute timesteps for structured 3D shapes—also `unsync_all`; Table 2). On ImageNet class-conditional generation, ComboStoc reduces FID from 17.2 to 15.69 at 400K training steps and from 12.6 to 11.41 at 800K steps compared to SiT-XL (Table 1), while on PartNet 3D shape generation the baseline `unsync_none` fails entirely and `unsync_all` achieves the best FPD (4.04), COV (0.86), and MMD (0.68) (Table 4). The method further enables graded test-time control by using asynchronous timesteps at inference, establishing that combinatorial stochasticity is a critical training ingredient when data possess strong structural correlations.


## GLM-OCR input · THINK-HIGH

## 1. Executive Summary

This paper studies an under-explored but important factor in diffusion generative models—**combinatorial complexity**—and proposes a simple fix, **ComboStoc**, that constructs stochastic processes fully exploiting combinatorial structures by converting the interpolation schedule $t$ into a tensor of the same shape as the data point and applying independently sampled, asynchronous values for different dimensions and attributes (e.g., different time steps for image patches and feature channels in latent space, or for 3D shape parts and their existence, bounding box, and shape code attributes). Across images on ImageNet and structured 3D shapes on PartNet, ComboStoc consistently accelerates training convergence—achieving systematically lower FID scores than baseline SiT and DiT (e.g., the `unsync_all` configuration significantly outperforms all other settings in Fig. 6a)—while on small-scale 3D data with stronger combinatorial structure the scheme becomes **indispensable** for obtaining a working generative model where the baseline `unsync_none` fails entirely (Fig. 8). Beyond training improvements, ComboStoc enables a novel inference-time paradigm of **graded control**, where different dimensions and attributes follow asynchronous time schedules, allowing flexible partial preservation of reference samples across patches, channels, and parts (Figs. 10–15), establishing that the gains from fully exploiting combinatorial stochasticity manifest when data exhibits strong structural correlations across dimensions and attributes—becoming essential for small datasets where baseline models collapse—while for data with nearly independent dimensions the scheme offers limited benefit.


## PyMuPDF input · THINK-NONE

## 1. Executive Summary

This paper analyzes an under-explored factor in diffusion generative models—the **combinatorial complexity** of high-dimensional, multi-attribute data—and introduces **ComboStoc**, a training and inference framework that desynchronizes the interpolation schedule across dimensions and attributes to uniformly sample the entire space spanned by source–target pairs. On ImageNet, ComboStoc reduces FID-50K from 17.2 (baseline SiT-XL) to 15.69 at 400K training steps and from 12.6 to 11.41 at 800K steps without classifier-free guidance, establishing a consistent improvement over the state-of-the-art SiT transformer (Tab. 1). For structured 3D shapes on PartNet, where combinatorial complexity is even stronger—spanning variable numbers of parts, existence indicators, bounding boxes, and latent shape codes—the unsynchronized training scheme proves indispensable: the baseline setting without combinatorial stochasticity almost entirely fails to produce meaningful shapes, while the finest split (`unsync_all`) achieves the best overall quantitative metrics (FPD 4.04, COV 0.86, MMD 0.68; Tab. 4). Beyond training acceleration, ComboStoc enables a novel test-time capability—**graded control**—where different dimensions and attributes receive asynchronous time steps, allowing varying degrees of preservation (e.g., spatially continuous soft inpainting with preservation weights smoothly decaying from ~0.85 at the subject center to 0 at the periphery; Fig. 12), establishing that combinatorial stochasticity is a simple yet powerful design principle for diffusion generative models whenever data exhibits structured, non-independent dimensions.



## GLM-OCR input · THINK-NONE

## 1. Executive Summary

This paper studies an under-explored factor in diffusion generative models—the **combinatorial complexity** arising from high-dimensional data samples where dimensions, patches, and attributes combine in ways that standard training schedules insufficiently cover. The authors propose **ComboStoc** (**Combinatorial Stochasticity**), a training framework that replaces the single scalar interpolation schedule $t$ with a tensor of asynchronous time values (e.g., assigning independent $t$ values per latent patch and per feature channel for images, or per part, per attribute, and per feature dimension for structured 3D shapes), thereby uniformly sampling the full rectangular subspace between noise and data points. Across ImageNet image generation and PartNet structured 3D shape generation, ComboStoc achieves systematically lower FID scores than baseline SiT and DiT models—matching SiT performance with roughly 4× fewer training iterations in some regimes—and for 3D shapes with strong combinatorial complexity proves **indispensable** for obtaining a working generative model, establishing that the benefits of combinatorial stochasticity scale with the structural richness of the data modality.