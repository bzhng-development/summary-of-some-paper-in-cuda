# MM1.5: Methods, Analysis & Insights from Multimodal LLM Fine-tuning

**ArXiv:** [2409.20566](https://arxiv.org/abs/2409.20566)

## 🎯 Pitch

MM1.5 introduces a new family of multimodal large language models (MLLMs) that achieve breakthroughs in text-rich image understanding, visual referring and grounding, and multi-image reasoning—three historically challenging areas for vision-language models. The paper's core innovation is a rigorous, data-centric three-stage training regimen, including high-resolution continual pre-training and dynamic image splitting, which allows even small models (1B–3B) to excel at fine-grained, grounded, and multi-image tasks. These advances unlock new practical capabilities such as robust document and UI comprehension, precise visual grounding, and scalable video and multi-image analysis, positioning MM1.5 as a versatile foundation for the next generation of multimodal AI applications.

---

## 1. Executive Summary (2–3 sentences)
MM1.5 is a family of multimodal large language models (MLLMs) that significantly improves three hard capabilities in vision–language systems: understanding text‑rich images (OCR‑heavy), visual referring/grounding (point/box-conditioned reasoning and box‑grounded output), and multi‑image reasoning. It achieves these through a data‑centric three‑stage training recipe (large‑scale pre‑training, high‑resolution continual pre‑training, and carefully balanced supervised fine‑tuning), plus a dynamic high‑resolution image encoding method (“AnyRes”) and explicit coordinate handling; the approach scales from 1B to 30B parameters, with dense and mixture‑of‑experts (MoE) variants, and is further specialized into video and UI models (Sections 1, 3–4; Figs. 1–2).

## 2. Context and Motivation
- Problem/gap addressed
  - Existing MLLMs often struggle with three underdeveloped but practically crucial competencies:
    - Reading and reasoning over high‑resolution, text‑rich images (documents, charts, UI) where small text and layout matter.
    - Referring to specific regions in an image and grounding responses with coordinates (boxes/points) instead of loose natural language references.
    - Multi‑image reasoning and multimodal in‑context learning (ICL) at inference time.
  - Many open models focus mainly on supervised instruction tuning (SFT) and single images, which limits generalization and fine‑grained grounding (Section 2).
- Why it matters
  - Real applications like document understanding, mobile UI agents, and multi‑image reasoning (e.g., comparison, retrieval‑augmented vision, or video via frame sampling) need precise reading, fine‑grained spatial reasoning, and the ability to handle more than one image (Sections 1–2).
- Prior approaches and their limits
  - High‑resolution comprehension: previous “static tiling” (e.g., fixed 2×2 grid) wastes tokens or pads empty areas for unusual aspect ratios; other works do not fully study design choices (Section 3.5).
  - Visual referring/grounding: strong proprietary systems (e.g., GPT‑4o) often rely on “set‑of‑mark (SoM) prompting” (a separate markup to denote regions) rather than native coordinate tokens; most open models lack robust, integrated grounding (Section 2).
  - Training recipes: many open approaches emphasize SFT alone; the impact of pre‑training data mix, high‑resolution continual pre‑training, or SFT category composition on cross‑capabilities remains under‑explored (Sections 1, 3.2–3.4).
- Positioning relative to the field
  - MM1.5 keeps the MM1 architecture (Section 3.1) to isolate the benefits of data and training strategies. It contributes a thorough empirical study and a matured recipe that delivers strong performance at small scales (1B/3B) and scales to 30B, with new, specialized video and UI variants (Sections 1, 4–6).

## 3. Technical Approach
This section decodes the full pipeline from model design to training and inference.

- Base architecture (Section 3.1; Fig. 1)
  - Vision encoder: a CLIP‑style image encoder.
  - LLM: the same decoder‑only language backbone as MM1.
  - Connector: `C-Abstractor` to map visual features to the LLM space.
  - Inputs/outputs:
    - Supports single or multiple images.
    - Accepts visual prompts (points and bounding boxes) through “coordinate tokens.”
    - Can output grounded responses by emitting bounding boxes inside text.
  - For all ablations, the 3B dense model is used unless noted (Section 3.1).

- Dynamic high‑resolution image encoding (“AnyRes”) (Sections 3.5–3.5.1; Fig. 11; Tables 1–3)
  - Problem: fixed 2×2 tiling (static splitting) is inefficient—small images are oversplit and long/tall images waste tiles with padding.
  - Solution: dynamic image splitting selects a grid `(n_h, n_w)` on a per‑image basis:
    - Consider all grids whose number of tiles lies in `[n_min, n_max]` (e.g., up to 10 tiles).
    - If a grid covers the image without downscaling, choose the one that minimizes padding after longer‑side resizing.
    - Otherwise choose the grid that minimizes resolution loss due to downscaling (Eq. (1), Fig. 11).
  - Global‑local format: always include a low‑resolution overview image in addition to tiles (“global view”), placed after sub‑images so that, under the autoregressive mask, the overview can attend to all tiles (Table 3 row 4 vs. row 1).
  - Sub‑image position indicators (Table 3): tested two schemes—`index` (triplet `(k, i, j)` for image ID and tile row/col) and `seps` (text separators like “:”, “,”, “<n>” between image tokens). These are optional; average gains are small, but index tokens help grounding slightly.

- Three‑stage training recipe (Fig. 2; Sections 3.2–3.4 and 4)
  1) Large‑scale pre‑training
     - Data (Section 4): 
       - 2B image–text pairs (captioning‑like).
       - 600M interleaved image–text documents (1B images total; sequences that mix images and text).
       - 2T text‑only tokens (“HQ‑Text,” a curated higher‑quality mix emphasizing general knowledge, math, and code; Section 3.4).
     - Crucial change vs. MM1: adjust the data ratio from `45:45:10` (image:interleaved:text) to `50:10:40` (Section 3.4; Fig. 10). This substantially downweights interleaved and upweights text, improving downstream knowledge and text‑rich tasks after SFT.
     - Optimization (Section 4): 200k steps, sequence length 4096, same schedule as MM1.
  2) High‑resolution continual pre‑training (Section 3.3; Fig. 9)
     - Goal: before SFT, inject strong OCR/text‑rich skills at high pixel density.
     - Data: 45M OCR‑style examples (PDFA, IDL, RenderedText, DocStruct‑4M), sampled equally per batch.
     - Input resolution: high‑res is critical—`1344×1344` performs best; using `378×378` can underperform skipping this stage entirely (Fig. 9a).
     - Synthetic captions: public synthetic caption sets (ShareGPT4V‑PT, LLaVA‑Recap‑3M) did not beat OCR‑only continual pre‑training in this setup (Fig. 9b). However, an in‑house 7M self‑training caption set shows consistent gains (Appendix A.1, Fig. 13).
     - Optimization: batch size 256, AdaFactor, peak LR `1e-5`, cosine decay, 30k steps.
  3) Supervised fine‑tuning (SFT) with balanced mixtures (Section 3.2; Fig. 4)
     - Data is categorized to target capabilities: `general`, `text-rich`, `refer&ground`, `science`, `math`, `code`, plus `multi-image` and `text-only` (Section 3.1 “SFT data categorization”, Fig. 4; Appendix A.2 Table 13).
     - Mixing ratios are chosen via extensive ablations (Sections 3.2.1–3.2.2; Figs. 5–8):
       - Within single‑image data, use the `general + text‑rich` mix as a strong base, then add others using per‑category ratios α relative to `general` per training batch:
         - `α_science = 0.1`, `α_math = 0.5`, `α_code = 0.2`.
         - `α_ref&ground = 2.0` to substantially boost grounding accuracy, accepting a small drop to other averages (Fig. 6d).
       - Across groups: set sampling weights `(w_single, w_multi, w_text) = (0.8, 0.1, 0.1)` (Fig. 7).
       - The final “All Mixture” balances best overall capability (Fig. 8).
     - Optimization: batch size 256, LR `2e-5`, 23k steps, 1 epoch.

- Final dynamic resolution settings (Section 4; Section 3.5.1)
  - Training: `(n_min, n_max) = (4, 9)`, encoder resolution `672×672`, `144` tokens per sub‑image, overview after tiles; dynamic splitting enabled only if a sample has fewer than 3 images.
  - Inference: can increase `n_max` (e.g., to 16) for higher effective resolution without retraining (Table 2 rows 1→3, 5–6).

- Mixture‑of‑Experts language backbones (Section 4 “Mixture‑of‑Experts (MoE)”)
  - Replace dense FFN layers in the LLM with `64` experts (every two layers), `top‑2` gating, balance loss `0.01`, router z‑loss `0.001`. Vision stack unchanged. MoE yields stronger multi‑capability integration at 1B/3B parameter scales (Tables 4–9).

- Specialized variants (Sections 5–6)
  - `MM1.5-Video`: treat videos as multi‑image inputs by uniformly sampling `N=24` frames, `144` tokens per frame, dynamic splitting disabled per frame due to token budget. Both training‑free (reuse MM1.5 image model) and SFT versions (mix of ShareGPTVideo, VideoChat2, ActivityNet‑QA) are built (Section 5; Table 10–11).
  - `MM1.5-UI`: further SFT on Ferret‑UI mixture (801k samples) to target mobile UI understanding and interaction; retains coordinate grounding (Section 6; Fig. 12; Table 12).

## 4. Key Insights and Innovations
- Data‑centric, stage‑wise recipe that measurably transfers to downstream capabilities
  - Innovation: explicit ablations that link data choices to specific capability gains across stages. Examples:
    - High‑res continual pre‑training on OCR data is pivotal—`1344×1344` outperforms lower resolutions and even beats skipping the stage (Fig. 9a).
    - SFT category ratios (`α` for science/math/code/ref&ground) and group weights `(w_single, w_multi, w_text)` are tuned to balance capabilities (Figs. 6–8).
    - Changing pre‑training mix from `45:45:10` to `50:10:40` and swapping to `HQ‑Text` improves text‑rich (+0.85) and knowledge (+0.99) averages with a minor multi‑image trade‑off (Fig. 10).
  - Significance: turns a common intuition—“data quality and balance matter”—into a clear, reproducible recipe with quantified trade‑offs.

- Dynamic high‑resolution image splitting with global‑local fusion
  - What’s new: a principled grid selection (Eq. (1), Fig. 11) that minimizes padding or downscale loss, plus an overview image placed after tiles for better attention flow (Table 3).
  - Why it matters: improves text‑rich benchmarks (DocVQA, InfoVQA) and adapts to unusual aspect ratios (Table 2). With 10 tiles at `672×672` and `144` tokens per tile, text‑rich and general averages improve (Table 1, row 7).

- Native visual referring and grounding with coordinate tokens
  - Difference from prior models: avoids SoM prompting and instead encodes/decodes boxes/points directly in text (Section 3.1; Fig. 1). This yields strong grounding scores (Table 7), while still preserving general abilities.

- Strong small‑scale models and MoE integration
  - Contribution: 1B/3B dense and MoE variants that, under the above recipe, outperform or match larger open baselines across many benchmarks (Tables 4–9). The 3B‑MoE rivals or surpasses 7B dense on several categories, showing MoE is an effective way to pack diverse capabilities at constant activated parameters.

- Unified path to video/UI without architecture changes
  - Insight: treating video frames as “multi‑image” is effective even training‑free; further SFT yields SOTA‑level results on public video QA and state‑of‑the‑art on UI elementary tasks (Tables 10–12; Fig. 12).

## 5. Experimental Analysis
- Evaluation setup (Sections 4.1 and Appendix A.4; Table 14)
  - Benchmarks grouped by capability: `general`, `text-rich`, `knowledge`, `refer&ground`, `multi-image`, and `VL-ICL`.
  - Metrics vary by benchmark (e.g., accuracy, ANLS for document QA, Recall@IoU>0.5, GPT‑assisted scores); zero‑shot and greedy decoding by default. Some competitors use beam search (Table 4 note).

- Main quantitative results
  - Across sizes, MM1.5 improves substantially over MM1 in nearly all categories:
    - Example at 30B (Tables 5–9):
      - MathVista +16.2 points (39.4→55.6), DocVQA +15.6 (75.8→91.4), InfoVQA +20.0 (47.3→67.3), MuirBench +21.5 (36.7→58.2).
  - Small‑scale leadership (Table 4):
    - At 1B, `MM1.5-1B` surpasses contemporaries like SPHINX‑Tiny, DeepSeek‑VL, TinyLLaVA on most reported benchmarks. Quote:
      > On DocVQA(test), `MM1.5-1B` reaches 81.0% vs. 70.0% for LLaVAOneVision‑0.5B and 53.0% for SPHINX‑Tiny (Table 6).
  - 3B dense vs. popular 3–4B models (Tables 4–6, 8–9):
    - `MM1.5-3B` beats MiniCPM‑V2 on many tasks, e.g., MathVista 44.4 vs. 38.7, DocVQA 87.7 vs. 71.9 (Table 6), and offers native grounding (Table 7).
    - Against Phi‑3‑Vision‑4B, `MM1.5-3B` lags on some knowledge tasks (AI2D/MMMU) but wins on text‑rich (DocVQA 87.7 vs. 83.3; InfoVQA 58.5 vs. 49.0), grounding (Table 7), and multimodal ICL (56.3 vs. 19.5; Table 8).  
  - MoE gains (Tables 4–9):
    - `MM1.5-3B-MoE` often surpasses `MM1.5-7B` in knowledge/general/grounding/multi‑image, trading a slight drop on text‑rich.  
      > For example, VL‑ICL average is 59.6 for 3B‑MoE vs. 56.0 for 7B dense (Table 8).
  - Referring & Grounding (Table 7):
    - `MM1.5-3B` achieves RefCOCO average 85.6, Flickr30k 85.9, and LVIS‑Ref 67.9; comparable to or better than many larger models; `MM1.5-30B` reaches LVIS‑Ref 84.9/61.4 (box/point) and Ferret‑Bench 77.1.
  - Multi‑image & ICL (Tables 8–9):
    - `MM1.5-30B` attains NLVR2 90.6 and MuirBench 58.2; in VL‑ICL, `MM1.5-30B` reaches 77.6, exceeding several open competitors and approaching GPT‑4V on some subtasks (Table 8).

- Ablations and what they prove
  - SFT category effects (Section 3.2.1; Fig. 5):
    - Adding `text-rich` boosts both text‑rich and knowledge averages.
    - `Science` helps knowledge; `code` mildly helps text‑rich; `refer&ground` is necessary for grounding but slightly regresses other categories—hence the tuned ratio `α_ref&ground = 2.0`.
  - SFT mixing ratios (Section 3.2.2; Figs. 6–8):
    - `w_text=0.1` has minor effect on core capabilities but helps language generalization.
    - `w_multi=0.1` meaningfully lifts multi‑image scores while slightly reducing base capabilities. The “All Mixture” maximizes overall average (Fig. 8).
  - Continual pre‑training resolution (Section 3.3; Fig. 9a):
    - `1344×1344` clearly best; `378×378` can be worse than no continual pre‑training.
  - Continual pre‑training data (Fig. 9b; Appendix A.1):
    - OCR‑only is strong; public synthetic captions did not add gains in this setup, but 7M high‑quality self‑generated captions do scale improvements (Fig. 13).
  - Pre‑training mix (Section 3.4; Fig. 10):
    - Switching to HQ‑Text and `50:10:40` further increases text‑rich (+0.85), knowledge (+0.99), and grounding (+~1.4), with a small multi‑image decrease (−0.05).
  - Dynamic splitting (Section 3.5.1; Tables 1–3):
    - More tiles and higher per‑tile resolution help text‑rich (Table 1).
    - Increasing `n_max` particularly helps DocVQA/InfoVQA (Table 2). Training with higher `n_max` is better than only increasing it at inference.
    - Placing overview after tiles yields a small but consistent gain (Table 3).

- Specialized variants
  - Video (Section 5; Tables 10–11):
    - Training‑free: `MM1.5-Video-3B` already surpasses several 7B training‑free baselines on multiple‑choice QA (e.g., NExTQA 72.8 vs. SlowFast‑LLaVA‑7B at 64.2; Table 10).
    - With video SFT: `MM1.5-Video-7B` achieves ActivityNet‑QA 60.9 and top/runner‑up results across diverse datasets; on LLaVA‑Hound, it is SOTA among reported entries (Table 11).
  - UI (Section 6; Table 12):
    - Even `MM1.5-UI-1B` surpasses the 13B Ferret‑UI baseline on all four elementary UI tasks (Ref‑i/A and Grd‑i/A), e.g., Ref‑i 90.0 vs. 80.5 and Grd‑i 86.5 vs. 79.4.

- Do the experiments support the claims?
  - Yes: the combination of broad SOTA‑level scores (Tables 4–9), targeted ablations (Figs. 5–10; Tables 1–3), and transfer to video/UI (Tables 10–12) directly tie the proposed recipe and AnyRes design to the reported capability gains. The paper also reports when trade‑offs occur (e.g., refer&ground slightly lowering other averages, reduced interleaved pre‑training lowering multi‑image a bit).

## 6. Limitations and Trade-offs
- Data and compute intensity
  - Large pre‑training corpora (2B image–text, 600M interleaved, 2T text) and high‑resolution continual pre‑training (45M OCR) are expensive (Sections 3.3–4). Dynamic splitting increases vision tokens for text‑rich inputs (Table 1).
- Mixture sensitivity and tuning overhead
  - Capability balance depends on careful ratio tuning. For instance, raising multi‑image SFT weight improves multi‑image benchmarks but reduces the “base” capability average (Fig. 7 right).
- Interleaved pre‑training trade‑off
  - Lowering interleaved proportion from 45% to 10% slightly drops multi‑image performance (−0.05 average) even though text‑rich and knowledge improve (Fig. 10).
- Grounding and coordinate transformations
  - Inference‑time changes to `n_min` can hurt grounding due to mismatched local→global coordinate conversion (Table 2 row 7), which constrains “just scale up at inference” tactics.
- Video framing constraints
  - Video variant disables dynamic splitting per frame and uses a fixed 24‑frame, 144‑token budget to fit context (Section 5). This may limit very long videos or fine temporal granularity.
- Generalization scope
  - Despite breadth, there remain untested areas (e.g., 3D, depth, fine‑grained pixel‑level segmentation). Some reported benchmarks are GPT‑assessed (e.g., MM‑Vet, LLaVA‑Hound), which can introduce evaluation variance.

## 7. Implications and Future Directions
- Field impact
  - The work shifts attention from architecture novelty to a reproducible, data‑centric training recipe that demonstrably transfers to core multimodal abilities and small‑scale models. It elevates integrated grounding and multi‑image reasoning to first‑class citizens in generalist MLLMs (Sections 1, 4).
- Follow‑up research enabled
  - Systematic study of high‑quality synthetic captions at scale (Appendix A.1 shows early promise).
  - Joint optimization of interleaved pre‑training and dynamic splitting for long‑context, multi‑image chains (e.g., videos or multi‑page documents).
  - Unifying image, video, and UI capabilities under one training schedule, possibly with curriculum over `n_max`, frame counts, and UI‑specific tasks (Section 7, conclusion).
  - More robust grounding with consistent coordinate systems under varying tiling, and extensions to pixel‑level grounding.
- Practical applications
  - Document understanding and enterprise OCR (charts, forms, infographics; Table 6).
  - Grounded assistants that can refer to and act on UI elements (Section 6; Fig. 12; Table 12).
  - Multi‑image analytics (e.g., product comparisons, surveillance snapshots) and video QA/analysis pipelines that can start training‑free, then gain with targeted SFT (Section 5; Tables 10–11).
  - Edge deployment scenarios: 1B/3B dense and MoE variants provide competitive capability for mobile and embedded settings (Tables 4–9).

In short, MM1.5 demonstrates that careful, high‑resolution continual pre‑training plus principled SFT mixture design—and a well‑engineered AnyRes pipeline—unlock substantial, verifiable gains in OCR‑heavy understanding, native grounding, and multi‑image reasoning, from mobile scales to 30B models, with straightforward extension to video and UI domains.
