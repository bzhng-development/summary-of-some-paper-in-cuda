# Section 1 — Executive Summary — A/B

**2411.18966 — SVGS: Enhancing Gaussian Splatting Using Primitives with Spatially Varying Colors**


_input chars: PyMuPDF=68,718 · GLM-OCR=59,002_



## PyMuPDF input · THINK-HIGH

## 1. Executive Summary

This paper introduces **SVGS (Spatially Varying Gaussian Splatting)**, a method that replaces the uniform, single-color appearance of standard 2D Gaussian surfels with **spatially varying colors and opacity** inside each primitive—so that different rays hitting the same surface at different points can see different RGB and alpha values. Three concrete instantiations of this spatially varying function are proposed: bilinear interpolation that partitions a Gaussian into four quadrants with learnable vertex colors/opacities, movable kernels that place four learnable exponential kernels on the surfel (weighted by distance to the hit point), and tiny three-layer MLPs that map local $(u,v)$ coordinates to color and opacity. On the Blender dataset, where scenes have rich textures but relatively flat geometry, SVGS equipped with the movable-kernel design attains 34.10 PSNR / 0.970 SSIM / 0.030 LPIPS—outperforming all prior Gaussian- and NeRF-based methods—and when 2DGS is forced to use twice as many Gaussians (1.43× more parameters), SVGS still yields higher fidelity, establishing that a single spatially varying Gaussian can replace numerous uniform counterparts, especially when complex textures are paired with simple underlying geometry.


## GLM-OCR input · THINK-HIGH

## 1. Executive Summary

This paper introduces **Spatially Varying Gaussian Splatting (SVGS)**, a method that equips each Gaussian surfel primitive with spatially varying colors and opacity via three functions—bilinear interpolation, movable kernels, and tiny MLPs—so that a single primitive can express different colors at different ray-intersection points. Evaluated on Synthetic Blender, DTU, Mip-NeRF360, and Tanks&Temples, SVGS with **movable kernels** achieves the best novel-view synthesis, surpassing all other Gaussian Splatting methods on the Blender dataset (Table III) while matching the quality of 2DGS with twice as many primitives (Table IX) and still outperforming 2DGS when training time is reduced by one third (Table XII), establishing that spatially varying primitives enable more compact, efficient scene representation, especially in the common real-world regime of complex textures on relatively simple geometry.


## PyMuPDF input · THINK-NONE

## 1. Executive Summary

This paper introduces **SVGS (Spatially Varying Gaussian Splatting)**, a method that augments individual 2D Gaussian surfel primitives with spatially varying color and opacity functions—rather than a single view-dependent color and scalar opacity—to improve their representational capacity per primitive. The authors evaluate three instantiations of this spatially varying function: bilinear interpolation (dividing each surfel into four quadrants with learnable corner values), movable kernels (four learnable sub-kernel centers that weight contributions via an exponential decay), and a tiny MLP (a three-layer network mapping local $(u,v)$ coordinates to color and opacity). Across the Blender, DTU, Mip-NeRF360, and Tanks&Temples datasets, the movable-kernel variant achieves the best novel-view synthesis results, reaching **34.10 PSNR** on the Blender dataset—surpassing all prior Gaussian-splatting and NeRF-based methods on that benchmark—while using **~1.4× the per-primitive parameter count** of 2DGS yet **fewer total primitives** (e.g., 205K vs. 384K–446K Gaussians in parameter-matched comparisons), establishing that spatially varying primitives are especially effective when textures are complex but geometry is relatively simple.


## GLM-OCR input · THINK-NONE

## 1. Executive Summary

This paper introduces **SVGS (Spatially Varying Gaussian Splatting)**, a method that augments individual Gaussian primitives with **spatially varying colors and opacity**—meaning different rays intersecting the same primitive at different surface locations see different appearance values rather than a single view-dependent color shared across the entire primitive—to improve the compactness and expressiveness of Gaussian-based scene representations. The authors implement three concrete spatially varying function designs on 2D Gaussian surfels: bilinear interpolation that partitions each surfel into four quadrants with per-quadrant learnable colors, movable kernels that place four exponential-decay sub-functions whose centers shift during optimization, and tiny three-layer MLPs that map local surfel coordinates to color and opacity. Evaluated on the Synthetic Blender, Mip-NeRF360, Tanks&Temples, and DTU datasets against the 2DGS baseline, the movable-kernel variant achieves the best novel-view synthesis performance—surpassing all state-of-the-art methods on the Blender dataset—while the bilinear interpolation and MLP designs also outperform 2DGS, establishing that **a single Gaussian primitive with spatially varying attributes** can represent complex textures on geometrically simple surfaces more effectively than the uniform-color primitives of vanilla Gaussian Splatting.