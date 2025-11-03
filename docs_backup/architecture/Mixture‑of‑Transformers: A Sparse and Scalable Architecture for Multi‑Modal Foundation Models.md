# Mixture‑of‑Transformers: A Sparse and Scalable Architecture for Multi‑Modal Foundation Models

**ArXiv:** [2411.04996](https://arxiv.org/abs/2411.04996)
**Authors:** Weixin Liang, Lili Yu, Liang Luo, Srinivasan Iyer, Ning Dong, Chunting Zhou, Gargi Ghosh, Mike Lewis, Wen‑tau Yih, Luke Zettlemoyer, Xi Victoria Lin
**Institutions:** Meta FAIR (Facebook AI Research), Stanford University (co‑authors affiliated)

## 🎯 Pitch

Mixture-of-Transformers (MoT) revolutionizes multi-modal processing by assigning modality-specific weights within a transformer while maintaining a unified global attention, significantly reducing training overhead without compromising performance. This approach enhances scalability and efficiency, particularly impacting non-text modalities like images and speech, making large multi-modal systems more accessible and cost-effective to deploy and train.

---

## 1. Executive Summary (2-3 sentences)
Mixture‑of‑Transformers (MoT) is a sparse multi‑modal transformer that assigns modality‑specific weights (for text, image, speech) to every non‑embedding component, while keeping one shared, global self‑attention over the full interleaved sequence. Across three regimes—autoregressive text+image (Chameleon), text+image+speech, and a mixed objective setting with text autoregression and image diffusion (Transfusion)—MoT matches or surpasses dense baselines with far fewer training FLOPs and lower wall‑clock time (e.g., Figure 5, Figure 8, Figure 10, Figure 11, Figure 19).

## 2. Context and Motivation
- Problem addressed
  - Early‑fusion multi‑modal foundation models process interleaved sequences of text, images, and speech in one transformer. Training them is very compute‑intensive and data‑hungry; e.g., Chameleon uses 9.2T tokens to match text LLMs trained on 2T tokens (Section 1).
  - Dense transformers show “modality conflict”: even though all tokens are handled uniformly, learned representations cluster by modality (PCA in Figure 2b and Appendix Figure 23), and training dynamics between modalities can interfere (Section 1, Figure 15).

- Why it matters
  - Reducing compute without sacrificing quality determines whether large multi‑modal systems are trainable at scale and deployable cost‑effectively.
  - Better modularity across modalities can lead to more stable training and faster convergence.

- Prior approaches and shortcomings
  - Mixture‑of‑Experts (MoE) sparsifies compute by activating a subset of experts per token, routed by a learned gate. In practice, MoE needs load‑balancing, can be unstable at scale, and adds router overhead (Section 1; Related Work §7.2).
  - Modality‑aware variants typically only sparsify MLPs (FFNs) or add modality‑specific modules during post‑training; they still share many parameters and/or rely on learned routing (Related Work §7.2).

- Positioning
  - MoT removes learned routing entirely and deterministically “routes” by the token’s modality, but not just in FFNs: it un‑ties all non‑embedding weights (FFN, attention projection matrices W_Q, W_K, W_V, W_O, and layer norms) while retaining a single global attention to preserve cross‑modal interactions (Section 2.2, Figure 3a).

## 3. Technical Approach
MoT modifies each transformer layer to use modality‑specific parameters around a shared attention computation.

- Key idea in plain language
  - For each token, use weights specialized for that token’s modality (text/image/speech) everywhere except embeddings, but let all tokens still attend to each other in one shared attention operation so cross‑modal information flows (Figure 3a).
  - This yields a sparse parameterization (different weights per modality), yet the computation per token remains equivalent to a dense transformer—hence “isoFLOP” during training/inference at matched activations (Section 2.2).

- How a standard transformer layer looks vs. MoT
  - Dense layer (Equation 1): compute attention, add & layer‑norm, compute FFN, add & layer‑norm.
  - MoT layer (Equations 2–3; Algorithm 1):
    1. Group input tokens by modality `m` (Algorithm 1, lines 3–5).
    2. Apply modality‑specific projections to get `Q^m, K^m, V^m` (line 6; modality‑specific `W_Q^m, W_K^m, W_V^m`).
    3. Concatenate results back to original sequence order (line 8).
    4. Run a single global self‑attention over all tokens (line 9).
    5. Apply modality‑specific output projection `W_O^m`, residual, and modality‑specific layer norm (line 11–12).
    6. Apply modality‑specific FFN and layer norm (lines 13–14).
  - The attention softmax is computed over the full mixed sequence—this is the “global self‑attention” that keeps cross‑modal interaction intact (Equation 3; Figure 3a).

- Why this design over alternatives
  - Versus MoE: MoT avoids learned routing and the associated instability and load balancing (Section 1, §6.1), while still specializing parameters by modality.
  - Versus cross‑attention fusion stacks: MoT needs fewer layers and keeps one stack with normalized attention across all tokens (Section 2.2, footnote 2), simplifying architecture and preserving consistent token‑level interactions.

- Modality representations and training objectives
  - Discrete images: tokenized into 1,024 discrete tokens (VQ‑VAE/VQGAN‑style) and trained autoregressively (Chameleon setting; Figure 3b, Figure 4).
  - Discrete speech: semantic tokens (~25 Hz) via a DinoSR‑style tokenizer; also trained autoregressively (Section 3.3.1; Table 2).
  - Continuous images: latent patches from a VAE and trained with a diffusion objective while text remains autoregressive (Transfusion setting; Appendix A, Figure 3c).
  - MoT supports all three within the same transformer (Sections 3.2–3.4).

- Step‑matching metric
  - To compare convergence speeds fairly, “step matching” plots map the training step of a sparse model to the dense model’s step at equal loss. The slope `s` is the fraction of dense steps required; smaller is faster (Figures 5b, 6, 8b, 11b, 12).

- Systems considerations (Section 6)
  - Computation per step matches dense (isoFLOP), but parameter count is higher since each modality has its own weights. This lowers the Parameter‑to‑FLOPs (PpF) ratio versus MoE (no large pool of unused experts and router params), which can improve distributed throughput (Section 6.1).
  - Overheads from grouping tokens by modality and re‑ordering can be reduced via caching indices and using grouped GEMMs (Section 6.1).

## 4. Key Insights and Innovations
1) Full‑stack modality‑specific decoupling with shared global attention
- What’s new: All non‑embedding parameters are modality‑specific (FFN, Q/K/V/O, layer norms) with one global attention over the interleaved sequence (Section 2.2; Figure 3a; Algorithm 1).
- Why it matters: It keeps cross‑modal information flow while letting each modality learn its own computations, reducing interference (Figure 15) and speeding convergence (Figures 5–6, 8, 10–12).

2) IsoFLOP sparsity that is simple and stable (no learned router)
- What’s new: Deterministic routing by modality avoids MoE’s router imbalance and bi‑level optimization issues, yet doesn’t increase training/inference FLOPs versus dense at matched activations (Section 2.2; §6.1).
- Why it matters: Delivers consistent speedups in steps and wall‑clock time, especially for non‑text modalities (image, speech), where MoE benefits were smaller or unstable (Figures 5, 8, 11–12, 19).

3) Evidence that most of the gains come from untying FFN and attention projections
- Ablation (Section 3.5; Figure 14): untying only FFN gives strong gains; adding Q/K/V untying further improves; untying layer norms has negligible effect on eval.
- Significance: Guides practical designs—prioritize FFN and attention projections for modality separation.

4) Modality Leave‑One‑Out (LOO) analysis revealing competitive interactions
- LOO study (Section 4; Figure 15): merging any two modalities into one tower hurts those modalities; isolating a modality in its own tower reduces its loss the most.
- Significance: Confirms that modality‑specific parameters reduce modality competition and improve optimization.

5) Practical systems wins and horizontal scaling
- Wall‑clock improvements on A100s (Section 6.2.2, Figure 19): image quality matched in 47.2% of dense time; text in 75.6%.
- Scaling (Section 6.2.1, Figure 18): as GPU count increases 16→256, MoT’s fraction of steps to match dense improves (image 42.1%→21.6%, text 75.7%→50.9%).
- Significance: The method’s benefits grow with scale and are observable in real training environments.

## 5. Experimental Analysis
- Evaluation setups
  - Chameleon (Section 3.2, Figure 4): text and discrete image tokens, both autoregressive. Validation on Obelisc, COCO, Flickr30k, Shutterstock; scales 37M→7B; hyperparams in Table 1.
  - Chameleon+Speech (Section 3.3, Figure 7): adds discrete speech tokens; speech validation on LibriLight (LL60K) and People’s Speech (PPL30K); scales 443M→7B; hyperparams in Table 3.
  - Transfusion (Section 3.4, Figure 10; Appendix A): text autoregression + image diffusion; eval by diffusion val loss, COCO‑30k CLIP and FID, and captioning CIDEr; scales 163M→7B; hyperparams in Table 4.

- Baselines
  - Dense transformer with same architecture and FLOPs.
  - MoE‑4x (Expert‑Choice routing) as a strong sparse baseline; used for validation perplexity with caveats about EC access to future tokens during validation and distribution sensitivity (Section 3.2.1 “Mixture‑of‑Experts Implementation”).

- Main quantitative results
  - Chameleon, 7B
    - Faster convergence overall: 
      > “MoT requires 45.5% of dense steps to match total training loss” (Figure 5b, slope s=0.455).
    - Image modality:
      > “MoT matches dense image loss in 34.8% of steps” (Figure 5d).
    - Validation at 55.8% steps:
      > “MoT at 55.8% steps achieves comparable or lower val losses than dense final” on multiple datasets (Figures 5g–5n), implying ~44.2% FLOPs savings.
  - Chameleon across scales (37M→7B)
    - Image: consistent acceleration; e.g., at 1.5B, s=0.350 (Figure 6n); at 7B, s=0.348 (Figure 6r).
    - Text: moderate gains; e.g., 7B s=0.558 (Figure 6t). Full curves in Figure 6; matching validation in Appendix Figure 24.
  - Chameleon+Speech, 7B
    - Speech training: 
      > “MoT achieves equivalent speech loss in 22.9% of steps vs dense” (Figure 8b, s=0.229).
    - Speech validation:
      > LL60K matching s=0.313 and PPL30K s=0.372 (Figures 8d, 8f), i.e., “37.2% of FLOPs” to match baseline.
    - Text and image validation remain at or better than dense at 55.8% steps (Figures 8g–8n).
  - Chameleon+Speech across scales (443M, 880M, 1.5B)
    - Speech modality: MoT matches dense in 15.1%–33.6% of steps (Figure 9f,n,v).
    - MoE‑4x: improves speech training loss but often underperforms on speech validation (Figures 9g–9h, 9o–9p, 9w–9x), consistent with Expert‑Choice sensitivity to distribution shift (Section 3.3.3).
  - Transfusion (text AR + image diffusion)
    - 7B image training:
      > “MoT reaches dense’s image loss in ~30% of steps” (Figure 10c, s≈0.309) and matches validation loss at ~37% (Figure 10e, s=0.374).
    - 760M vs 1.4B dense:
      > CLIP 0.214 vs 0.206 (higher is better), FID 21.145 vs 24.688 (lower is better), CIDEr 0.320 vs 0.286 (Figures 11c–11e), while using half the training/inference FLOPs.
    - Across scales (163M, 760M, 1.4B): image FID and CLIP consistently improve over dense (Figure 12 panels (5–6), (14–15), (23–24)). Text validation on C4/Wikipedia is roughly on par, but captioning CIDEr consistently higher for MoT (Figure 12 panels (9), (18), (27)); full text curves in Appendix Figure 26.

- Ablations and analyses
  - Ablation on where to untie (Figure 14, Section 3.5):
    > FFN untying accounts for the largest gains; adding Q/K/V untying brings further improvement; LayerNorm untying adds little.
  - Leave‑One‑Out (LOO) (Section 4, Figure 15):
    > Isolating a modality in its own tower lowers its loss most; merging hurts both merged modalities, evidencing competition for shared parameters in dense models.
  - Hybrid MoT + text‑MoE (Section 5; Figures 16–17):
    > Replacing only the text FFN with MoE‑4x further accelerates text without hurting image, both in Chameleon and Transfusion.

- Systems profiling and scaling
  - Wall‑clock (A100s on AWS): 
    > “MoT matches dense image quality in 47.2% of time; text in 75.6%” (Figure 19b, 19d).
    - MoE‑4x showed 1.7× slowdown for image and no speed advantage for text in this setup (Figure 19b,d).
  - Horizontal scaling: 
    > As GPU count increases, MoT’s step fraction to match dense improves (Figure 18).

- Do the experiments support the claims?
  - Yes, across three regimes and multiple scales from 37M to 7B, results are consistent that MoT accelerates especially the non‑text modalities while maintaining or improving quality under matched FLOPs (Figures 5–12, 19). The ablations (Figure 14) and LOO (Figure 15) give mechanistic evidence for why modality‑specific untying helps.

- Notable caveats called out in the paper
  - MoE‑4x validation uses the Expert‑Choice router, which can access future tokens during validation and is sensitive to distribution shift; this can over‑ or under‑estimate MoE quality (Section 3.2.1). The paper discusses these limitations when interpreting MoE comparisons.

## 6. Limitations and Trade-offs
- Assumptions and scope
  - Requires clear modality labeling per token (“modality indexing logic,” Figure 3a); performance assumes that tokens can be reliably assigned to modalities.
  - Benefits are strongest when modalities differ substantially (speech/image vs text). Text‑only gains are smaller in mixed‑objective Transfusion where dense already separates objectives (Section 3.4.2, Appendix Figure 26).

- Compute/memory considerations
  - IsoFLOP per step is preserved, but parameter count increases because each modality owns a full copy of non‑embedding weights. This can raise memory footprint even if compute is unchanged (Section 2.2; §6.1).
  - Some overhead exists for grouping tokens, reordering, and masking. The paper suggests engineering remedies (caching indices, grouped GEMMs) but does not report micro‑benchmarks of each overhead (Section 6.1).

- Evaluation caveats
  - MoE‑4x comparisons may be confounded by Expert‑Choice validation behavior (Section 3.2.1).
  - Text improvements in Transfusion are marginal (Section 3.4.2); additional techniques (e.g., hybrid text MoE) may be required.

- Data and training scale
  - Major image‑generation improvements are shown at 0.5T tokens in Transfusion; the paper notes models are not yet saturated and expects further gains with more data (Section 3.4.4, Appendix B).

- Generality to more modalities or objectives
  - While the method extends to speech and diffusion, the paper does not explore audio generation, video, or more than three modalities; nor adaptive routing within a modality.

## 7. Implications and Future Directions
- How this changes the landscape
  - Demonstrates a practical middle ground between dense models and MoE: deterministic, modality‑aware sparsity that is easy to train, isoFLOP, and yields proportional wall‑clock wins at scale (Figure 19), especially for images and speech.
  - Provides architectural evidence that “full‑stack” modality separation (FFN + attention projections) is more impactful than partial separation (Figure 14).

- Follow‑up research enabled/suggested
  - Hybridization: Combine MoT with MoE in a targeted way (e.g., MoE for text only) to maximize strengths of each (Section 5; Figures 16–17).
  - Engineering for inference/training
    - Grouped/Block‑sparse GEMMs and cached modality indices to reduce overhead (Section 6.1).
    - Dynamic/continuous batching across requests by modality for higher inference throughput (Section 6.3).
  - Extending modalities and objectives
    - Video tokens, high‑rate audio, or other continuous modalities with matching objectives.
    - Investigate adaptive sub‑modality specialization (e.g., different image domains) without learned routers.
  - Theoretical understanding
    - Why FFN untying contributes most; formal analysis of modality competition and representation geometry (see modality clustering in Figure 2b and Appendix Figure 23).

- Practical applications
  - Multi‑modal assistants that generate and understand text, images, and speech more efficiently.
  - Content creation systems (image generation/editing; Figure 13 and Appendix B) with improved training efficiency and cross‑modal capabilities.
  - Large‑scale pretraining pipelines: MoT’s lower PpF ratio and stable scaling (Section 6.1, Figure 18) can reduce cost on cloud clusters.

---

Definitions of less common terms used above
- `MoT` (Mixture‑of‑Transformers): an architecture that assigns separate weights per modality to FFN, attention projections, and layer norms, while keeping a shared global attention over all tokens (Section 2.2; Algorithm 1).
- `isoFLOP`: comparison where training/inference FLOPs per step are matched across models.
- `Expert‑Choice (EC) routing`: an MoE mechanism where experts pick top‑k tokens; strong for load balancing but can complicate autoregressive evaluation (Section 3.2.1).
- `Step matching`: a diagnostic where the number of steps a model needs to reach the dense model’s loss is reported as a fraction `s`; lower is better (Figures 5b, 6, 8b, 11b, 12).
- `CLIP score` (↑ better): semantic alignment between generated images and prompts on COCO‑30k (Section 3.4.1).
- `FID` (↓ better): Fréchet Inception Distance; lower indicates more realistic images (Section 3.4.1).
- `CIDEr` (↑ better): captioning metric for image‑to‑text quality on COCO (Section 3.4.1).
- `PpF ratio`: Parameter‑to‑FLOPs ratio; lower can improve distributed training throughput (Section 6.1).
