# Mixture-of-Transformers: A Sparse and Scalable Architecture for Multi-Modal Foundation Models

**ArXiv:** [2411.04996](https://arxiv.org/abs/2411.04996)

## 🎯 Pitch

This paper introduces Mixture-of-Transformers (MoT), a novel architecture that decouples all non-embedding transformer weights by modality while maintaining global self-attention across interleaved sequences of text, image, and speech tokens. MoT dramatically reduces pretraining computational cost—cutting FLOPs and wall-clock time by up to half—without sacrificing performance, enabling the practical and scalable training of powerful multi-modal foundation models essential for unified AI systems. By addressing core inefficiencies and instability in prior approaches like dense early-fusion models and Mixture-of-Experts, MoT paves the way for faster, more efficient, and accessible deployment of multi-modal large models.

---

## 1. Executive Summary (2–3 sentences)
This paper introduces Mixture-of-Transformers (`MoT`), a sparse multi-modal Transformer in which all non-embedding weights are “untied” by modality (text, image, speech) while keeping a single global self-attention over the entire interleaved sequence (Section 2.2; Eq. 2–3; Algorithm 1). Across three training regimes—autoregressive text+image (“Chameleon”), text+image+speech, and the Transfusion regime with diffusion for images—`MoT` reaches the dense baseline’s quality with substantially fewer training FLOPs and wall‑clock time (e.g., at 7B parameters: 55.8% FLOPs for text+image, 37.2% FLOPs for speech, and 47.2% of wall‑clock time for image; Figures 5, 8, 19).

## 2. Context and Motivation
- Problem addressed
  - Early‑fusion multi‑modal LLMs interleave tokens from different modalities into a single sequence and train one dense Transformer on all of them. This is expensive and hard to optimize at scale: Chameleon needs 9.2T training tokens to match LLaMA‑2’s 2T text performance (Section 1), and different modalities interact suboptimally during joint training.
  - Empirical evidence shows tokens from text, images, and speech occupy distinct regions of the model’s latent space even without modality priors (PCA in Figure 2b and Appendix Figure 23), suggesting that a single shared parameterization may be suboptimal.

- Why this matters
  - Real‑world systems increasingly require unified models that can read and generate text, images, and speech. Reducing training cost and improving stability in such models has direct impact on feasibility, iteration speed, and deployment cost.

- Prior approaches and shortcomings
  - Mixture‑of‑Experts (`MoE`) sparsifies computation by routing tokens to expert MLPs, but learned routers create load‑balancing issues and training instabilities, especially early in training and at scale (Section 1). In multi‑modal setups, simple rule‑based routing by modality often outperforms learned routing but is typically limited to the MLP (FFN) only.
  - Prior multi‑modal models often use cross‑attention between separate encoders/decoders, adding architectural complexity and layers (Section 2.2, footnote 2).

- Positioning
  - `MoT` generalizes “modality‑aware sparsity” from only the FFN to the entire Transformer (FFN, Q/K/V/O projections, LayerNorm), yet retains a single global self‑attention so tokens still attend across modalities (Figure 3a). It preserves the dense model’s FLOP profile per step but converges in far fewer steps, thus saving total training FLOPs.

## 3. Technical Approach
`MoT` makes one architectural change to a standard Transformer layer: it retains joint self‑attention but uses modality‑specific parameters everywhere else.

- Baseline Transformer (Eq. 1; Section 2.2)
  - For an input sequence `x = (x1…xn)`, a standard layer computes an attention block followed by an FFN, with LayerNorm and residuals.

- `MoT` layer (Eq. 2–3; Algorithm 1)
  - Idea in plain language:
    - Group tokens by modality (text, image, speech).
    - Project each group with its own Q/K/V weights and later its own output projection `W_O` (lines 3–11 in Algorithm 1).
    - Reassemble the sequence and run one global self‑attention over all tokens (lines 8–9), so cross‑modal interactions are preserved.
    - Apply modality‑specific LayerNorms and a modality‑specific FFN to each token (lines 12–15).
  - Notation (Eq. 3): for token `i` with modality `m_i`, use `W_Q^m_i, W_K^m_i, W_V^m_i, W_O^m_i` and `LayerNorm_attn^m_i, LayerNorm_ffn^m_i`. Attention weights are still normalized jointly over the entire sequence.

- Why keep global self‑attention? (Section 2.2, footnote 2)
  - It allows early‑fusion models to share a single stack and capture cross‑modal context, while avoiding the extra layers/latency of separate encoders bridged by cross‑attention.

- Why not learned routing (MoE)?
  - Deterministic routing by a token’s modality is simple and stable. It avoids router training pathologies and imbalanced expert use (Section 1). Unlike MoE, `MoT`’s “routing” does not add inference-time complexity.

- FLOP profile
  - Per training step, `MoT` keeps the same FLOPs as the dense model (Section 2.2). Savings come from reaching a target loss in fewer steps (quantified via “step matching” plots in Figures 5–6, 8–12).

- Implementation details and data flow (Figure 3a; Algorithm 1)
  - “Modality indexing logic” selects per‑modality weights.
  - “Sequence re-ordering buffer” groups tokens by modality for projections, and restores order before attention.
  - Works with discrete tokens trained autoregressively (Figure 3b; Chameleon) and with continuous latent image patches trained by diffusion (Figure 3c; Transfusion). In the diffusion case, images are 256 latent patches from a VAE (Appendix A.2).

- Training regimes evaluated
  - Chameleon setting (Section 3.2; Figure 4): text and images both trained autoregressively; images tokenized with a pretrained VQ‑VAE/VQGAN into 1,024 discrete tokens.
  - Chameleon + Speech (Section 3.3; Figure 7): add speech tokens (25Hz semantic tokens; Table 2) with an autoregressive objective.
  - Transfusion setting (Section 3.4; Figure 10a; Appendix A): text is autoregressive; images are trained with a diffusion loss on VAE latents; a single Transformer handles both via a combined objective.

## 4. Key Insights and Innovations
- Modality‑specific parameter decoupling across the entire Transformer (fundamental)
  - Novelty: Prior work typically untied only FFNs (or used MoE in FFNs). `MoT` untyes FFN, all attention projections (Q/K/V/O), and LayerNorms by modality (Section 2.2; Figure 3a; Algorithm 1).
  - Why it matters: lets each modality use its own “memory” (FFN) and projections, matching the observed modality clusters in feature space (Figure 2b; Appendix Figure 23). Empirically yields large training step reductions, especially for non‑text modalities.

- Global self‑attention with sparse per‑modality parameters (fundamental)
  - Different from using cross‑attention between separate encoders/decoders, `MoT` keeps a single attention over all tokens (Eq. 3). This preserves cross‑modal interactions while reaping the benefits of modality‑tailored processing.

- IsoFLOP per step but faster convergence (practical + empirical)
  - `MoT` does not increase per‑step compute—savings come from requiring fewer steps for the same loss. This is rigorously quantified with step‑matching analyses across scales and settings (Figures 5–6, 8–12).

- Systems perspective: favorable parameter‑to‑FLOP ratio vs MoE and better wall‑clock (practical)
  - Analytic comparison shows MoT scales parameters with number of modalities `K` rather than number of experts `E` (Section 6.1), giving a lower parameter‑to‑FLOP ratio than typical MoEs. On AWS A100s, `MoT` achieves the dense model’s image quality in 47.2% of the time and text in 75.6% (Figure 19).

- Evidence for modality interference and benefits of separation (insight)
  - Leave‑One‑Out (LOO) analysis (Figure 15) shows combining two modalities in one tower degrades them, while separating improves losses in their isolated towers, supporting the design choice of modality‑specific weights.

## 5. Experimental Analysis
- Evaluation setup
  - Models and scales
    - Up to 7B parameters with sequence length 4,096 tokens; training runs are FLOP‑controlled and from scratch. Table 1 (Chameleon), Table 3 (Chameleon+Speech), and Table 4 (Transfusion) list the model hyperparameters and training tokens per setup.
  - Baselines
    - Dense Transformer and a 4‑expert MoE (`MoE‑4x`) with Expert Choice routing (Section 3.2.1). To ensure isoFLOP at inference for MoE in auto‑regressive settings, validation losses are computed using the same EC routing used at training; the paper discusses caveats (router can “peek ahead” and distribution shifts can misbalance experts).
  - Datasets and metrics
    - Chameleon: evaluation on Obelisc, COCO, Flickr30k, Shutterstock (Section 3.2.1).
    - Speech: People’s Speech, VoxPopuli (EN), LibriLight, MLS English, Spotify (Table 2); validation reported on LibriLight (LL60K) and People’s Speech 30K (PPL30K).
    - Transfusion: text PPL on C4 and Wikipedia; image generation quality via diffusion validation loss on CC12M, CLIP score and FID on COCO‑30k, and captioning CIDEr on COCO (Section 3.4.1).

- Main quantitative results
  - Chameleon 7B (autoregressive text+image)
    - Training speed: 
      > MoT reaches the dense model’s final training loss at 120k steps in 60k steps—45.5% of steps (Figure 5a–b).
      > By modality: image needs 34.8% of steps; text 55.8% (Figure 5c–f).
    - Validation at 55.8% training steps matches or beats dense final validation losses across all held‑out datasets (Figure 5g–n).
  - Chameleon across scales (37M → 7B)
    - Image: consistent, large speedups for MoT vs dense and MoE; at 7B, MoE’s gains vanish while MoT remains strong (Figure 6a–f, i–r).
    - Text: both MoT and MoE beat dense, with MoT comparable or slightly better (Figure 6g–h, o–t). Appendix Figure 24 shows matching validation trends.
  - Chameleon + Speech 7B (text+image+speech)
    - Speech training speed:
      > MoT matches dense speech loss in 22.9% of training steps (Figure 8a–b).
      > On LL60K and PPL30K, MoT reaches dense speech validation quality with 31–37% of dense steps (Figure 8c–f), summarized as ~37.2% of FLOPs.
    - Image and text performance stay strong even with speech added; at 55.8% training steps MoT matches/exceeds dense final validation on Obelisc, COCO, Flickr, SSTK (Figure 8g–n).
    - Across scales (443M, 880M, 1.5B): speech speedups remain large (15.1–33.6% of steps), while MoE shows instability on speech validation despite lower training loss (Figure 9).
  - Transfusion (text autoregressive + image diffusion)
    - 7B image modality:
      > MoT achieves the dense image training loss in roughly 30–37% of steps (Figure 10b–e).
      > Better image quality: higher CLIP (+0.005 to +0.01) and lower FID (e.g., FID 8.14 with guidance 1.6 vs a denser model’s 9.22; Section 3.4.2).
    - 760M vs 1.4B dense:
      > Despite using half the FLOPs, 760M MoT outperforms 1.4B dense on all image metrics: CLIP 0.214 vs 0.206; FID 21.145 vs 24.688; CIDEr 0.320 vs 0.286; and lower image training loss (Figure 11c–e).
    - Across scales (163M, 760M, 1.4B):
      > MoT consistently improves FID and CLIP (e.g., FID 21.58 vs 27.42 at 163M; 15.75 vs 25.58 at 760M; 15.85 vs 19.32 at 1.4B; Figure 12(6),(15),(24)), and improves captioning CIDEr (Figure 12(9),(18),(27)).
      > Text perplexity improvements are small in Transfusion (Appendix Figure 26), likely because decoupled objectives already make text training easy (Section 3.4.2).

- Ablations and analyses
  - Component ablations (Figure 14; Section 3.5)
    - Untying only FFN helps significantly (especially image). Adding Q/K/V untying further improves. Untying LayerNorm adds negligible benefit on top of those.
  - Modality Leave‑One‑Out (Figure 15; Section 4)
    - When two modalities are forced to share one tower, their losses worsen; when isolated, each modality improves the most in its own tower—evidence of modality interference and benefit of separation.
  - System profiling (Section 6)
    - Scaling GPUs from 16→256 increases MoT’s step advantage (Figure 18). On 256 A100s, MoT reaches dense image training loss in 47.2% of wall‑clock time and text in 75.6% (Figure 19b,d). MoE shows no time advantage and can be slower on image.

- Do the experiments support the claims?
  - Yes, across 13 pretraining runs over three regimes and multiple scales, `MoT` shows consistent speedups and validation parity or gains, strongest for image and speech (Sections 3.2–3.4; Figures 5–12). The wall‑clock profiling (Section 6.2.2; Figure 19) shows that FLOP savings translate to real training time reductions.

## 6. Limitations and Trade-offs
- Increased parameter memory (but same FLOPs per step)
  - Untying parameters by modality replicates non‑embedding weights for each modality (`K`-fold), increasing memory footprint even if per‑step compute remains isoFLOP. This affects optimizer state and checkpoint size.
- Fixed routing by modality
  - Deterministic routing is stable but coarse. It cannot adapt to sub‑domains (e.g., “diagrams” vs “photos”) or dynamically allocate capacity based on difficulty, as MoE can in principle.
- Text gains are modest in the Transfusion setup
  - In the diffusion+LM regime, text perplexity gains are small (Appendix Figure 26), possibly because the decoupled losses already make text training close to optimal (Section 3.4.2). This suggests `MoT`’s benefits concentrate where modality differences and compute are largest (images, speech).
- Evaluation caveats for MoE baseline
  - To ensure isoFLOP at validation, Expert Choice routing is used at eval time, which can “peek ahead” and inflate validation quality, or conversely misroute under distribution shift (Section 3.2.1). This makes MoE comparisons conservative in some plots and still MoT wins, especially in non‑text modalities.
- Engineering considerations
  - MoT introduces overheads for grouping/reordering tokens by modality and per‑modality GEMMs; while mitigable (Section 6.1), production inference may need dynamic batching across requests to keep specialized kernels efficient (Section 6.3).
- Scope
  - Results are shown for three modalities (text, image, speech) and two training paradigms (autoregressive and diffusion). Further modalities (video, structured data) and tasks remain to be validated.

## 7. Implications and Future Directions
- How this changes the landscape
  - `MoT` shows a simple, deterministic, and stable alternative to MoE for multi‑modal scaling: keep global attention, but give each modality its own projections, FFN, and norms. This directly addresses observed modality separation in feature space (Figure 2/Appendix Figure 23) and delivers large step/time savings without complex routers.
- Practical applications
  - Faster pretraining of unified generative models for assistants that read/write text, render images, and handle speech. Reduced wall‑clock time and FLOPs lower cost barriers for labs and companies training multi‑modal models.
- Research avenues
  - Finer‑grained routing: combine `MoT` with MoE inside a modality (Section 5; Figures 16–17), where “MoT + Text MoE‑4x” improves text while preserving image quality—suggesting hybrid designs.
  - Beyond three modalities and beyond images/speech: test with video, 3D, sensory streams; study whether modality granularity (e.g., splitting images into “photographs” vs “diagrams”) yields further gains.
  - Better systems kernels: group GEMMs and block‑sparse operations to remove indexing overhead (Section 6.1), plus dynamic/continuous batching for inference (Section 6.3).
  - Understanding interference: LOO results (Figure 15) show non‑reciprocal competition across modalities. A theoretical model of interference in shared attention with modality‑specific towers could guide automated capacity allocation.
  - Long‑context multi‑modal generation: global attention remains O(n²). Exploring efficient attention variants within `MoT` while preserving cross‑modal alignment is a natural next step.

> Representative headline results
> - Chameleon 7B: match dense in 45.5% of steps overall; 34.8% steps for image; 55.8% training FLOPs to match validation (Figure 5a–n).
> - Chameleon+Speech 7B: match dense speech in 22.9% steps (Figure 8a–b); reach dense speech validation in ~37% of FLOPs (Figure 8c–f) while preserving text+image quality (Figure 8g–n).
> - Transfusion 760M vs 1.4B dense: better image CLIP (0.214 vs 0.206), FID (21.145 vs 24.688), CIDEr (0.320 vs 0.286) at half the FLOPs (Figure 11c–e).
> - Wall‑clock: on 256 A100s, MoT reaches dense image quality in 47.2% of the time and text in 75.6% (Figure 19b,d).
