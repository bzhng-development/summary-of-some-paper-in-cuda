# MM1.5: Methods, Analysis & Insights from Multimodal LLM Fine‑tuning

**ArXiv:** [2409.20566](https://arxiv.org/abs/2409.20566)
**Authors:** Haotian Zhang, Mingfei Gao, Zhe Gan, Philipp Dufter, Nina Wenzel, Forrest Huang, Dhruti Shah, Xianzhi Du, Bowen Zhang, Yanghao Li, Sam Dodge, Keen You, Zhen Yang, Aleksei Timofeev, Mingze Xu, Hong‑You Chen, Jean‑Philippe Fauconnier, Zhengfeng Lai, Haoxuan You, Zirui Wang, Afshin Dehghan, Peter Grasch, Yinfei Yang
**Institutions:** Apple AI Research

## 🎯 Pitch

The MM1.5 family of multimodal large language models revolutionizes image processing by enhancing text-heavy visual understanding, precise visual grounding, and multi-image reasoning through an innovative data-centric training approach. By refining high-resolution handling and strategic data combination, MM1.5 outperforms peers without new architectures, making it pivotal for document AI and grounded assistant applications, especially for resource-constrained environments.

---

## 1. Executive Summary
`MM1.5` is a family of multimodal large language models (MLLMs) that substantially improves three hard capabilities for images: understanding text-heavy visuals (OCR-like tasks), visual referring and grounding (point/box-based localization), and reasoning across multiple images. The work’s significance is not a new architecture but a data‑centric training recipe—spanning pre-training, high‑resolution continual pre-training, and supervised fine‑tuning (SFT)—plus a robust high‑resolution image handling method, all validated by extensive ablations and strong results from 1B to 30B parameters (Figures 1–2; Sections 3–4; Tables 4–9).

## 2. Context and Motivation
- Problem/gap
  - Many open and proprietary MLLMs handle general vision–language tasks but struggle with:
    - Reading small text in images at diverse aspect ratios and resolutions (“text‑rich image understanding”).
    - Precise visual grounding and referring (identifying and reasoning over specific regions via coordinates).
    - Multi-image reasoning and in-context learning with interleaved image–text inputs.
  - The “how” of building such capabilities—especially data composition across the full training lifecycle—has been under-explored. Prior open models often focus mainly on SFT with limited study of pre-training data or high-resolution strategies (Section 2).
- Importance
  - Text-rich understanding powers document analysis, charts, UI comprehension, receipts, and infographics.
  - Referring/grounding is crucial for controllable agents, robotics, and UI automation.
  - Multi-image reasoning supports tasks like change detection, comparison, and episodic reasoning.
- Prior approaches and their limits
  - Closed models (GPT-4V/4o, Gemini 1.5, Claude 3.5) show strong multimodal competence but rely on proprietary training; GPT‑4o often needs “set‑of‑mark” prompting to reference regions instead of native grounded outputs (Section 2).
  - Open models (LLaVA, InternVL2, Qwen2‑VL, Cambrian‑1) narrow the gap but often:
    - Underperform on fine-grained grounding (Section 2).
    - Use static image tiles or low token budgets, limiting small text recognition (Sections 2, 3.5).
    - Provide limited analysis of data category mixing across pre-training/continual‑pre-training/SFT.
- Positioning
  - `MM1.5` keeps the MM1 architecture to isolate and study training recipes and high‑resolution processing. It offers detailed, reproducible ablations on:
    - SFT mixtures by capability (general, text‑rich, science/math/code, refer&ground, multi-image, text‑only).
    - High‑resolution continual pre-training data and resolution.
    - Pre-training data ratios and improved text-only corpora.
    - Dynamic image splitting to reach up to 4 MP effective resolution with minimal padding (Sections 3.2–3.5; Figure 11; Tables 1–3).

## 3. Technical Approach
Step-by-step overview of how the system is built and trained.

- Overall architecture (Figure 1; Section 3.1)
  - Vision encoder: CLIP-like image encoder.
  - Language backbone: decoder-only LLM.
  - Connector: `C‑Abstractor` projects visual tokens to the LLM (Section 3.1).
  - Capabilities:
    - Accepts multiple images interleaved with text.
    - Emits grounded outputs natively: bounding boxes embedded in text (e.g., “<x1,y1,x2,y2>”) and interprets point/box inputs (“coordinate tokens”).
    - Handles high resolution via dynamic image splitting.

- Three-stage training pipeline (Figure 2; Section 4)
  1) Pre-training (same data sources as MM1 for image portions; updated text-only set)
     - Data composition changed from MM1’s 45:45:10 to `50:10:40` for image‑caption : interleaved image‑text : text‑only (Section 3.4; Figure 10; Section 4).
       - Image‑caption: 2B pairs.
       - Interleaved documents: 600M docs (1B images).
       - Text‑only: 2T tokens from a higher‑quality “HQ‑Text” collection (Section 3.4).
     - Motivation: Improve language and knowledge-heavy benchmarks after SFT; large down-weighting of interleaved data yielded better downstream performance even if pre-training few-shot metrics did not always predict this (Section 3.4).
     - Training: 200k steps, sequence length 4096 (Section 4).
  2) High‑resolution continual pre-training
     - Target: strengthen text-rich (OCR-like) understanding before SFT.
     - Data: 45M document-centric images from PDFA, IDL, RenderedText, DocStruct‑4M (Section 3.3).
     - Resolution: best setup is 1344×1344 with splitting (Figure 9a).
     - Synthetic captions: public sets (LLaVA‑Recap‑3M, ShareGPT4V‑PT) did not clearly help beyond the OCR mixture at this stage; a separate self‑trained captioner generating 7M high-quality captions showed promise (Appendix A.1; Figure 13), but OCR-only remained the default (Section 3.3).
  3) Supervised fine-tuning (SFT)
     - Carefully balanced multi-capability mixture (Figure 4; Section 3.2). Final macro-ratios:
       - `wsingle=0.8`, `wmulti=0.1`, `wtext=0.1` (Section 3.2.2).
       - Within single-image 80%: roughly 37.2% text‑rich, 22.5% refer&ground, 11.3% general, 5.6% math, 2.3% code, 1.1% science (Section 4).
     - Key mixing choices from ablations:
       - When blending with general data, best α (target:general) for science=0.1, math=0.5, code=0.2 (Figure 6a–c).
       - For refer&ground, α=2.0 trades a small drop in base scores for a large grounding gain (Figure 6d).
       - Multi-image `wmulti=0.1` boosts multi-image metrics while limiting regressions on single-image averages; text-only `wtext=0.1` changes little but reserves capacity for images (Figure 7).

- Dynamic high‑resolution image splitting (“AnyRes”) (Section 3.5; Figure 11; Tables 1–3)
  - Problem: Fixed 2×2 tiling wastes tokens on small or elongated images (padding) and misses detail on long documents.
  - Method (Equation 1):
    - Predefine allowed grid shapes by `nmin ≤ nh·nw ≤ nmax`.
    - For a given input image size (h,w) and encoder resolution r:
      - If some grid can cover the image without downscaling below r, choose the grid that minimizes padding.
      - Otherwise choose the grid that minimizes resolution loss due to downscaling.
  - “Global–Local” format: besides the sub-images, also pass a downscaled “overview” image; the paper places the overview after the tiles so it can attend to all tiles (Table 3, row 4 vs. row 1).
  - Sub-image position indicators:
    - `index`: tuples `(k,i,j)` describing image k, row i, column j.
    - `seps`: special tokens between tiles to recover 2D layout.
    - Averages show small differences; indicators help some DocVQA/InfoVQA and grounding, but are not strictly necessary (Table 3).
  - Training vs. inference: train with `(nmin,nmax)=(4,9)` but can infer at higher grids like `(4,16)` for more effective resolution (Table 2).

- Model scales and MoE (Section 4)
  - Dense: 1B, 3B, 7B, 30B.
  - MoE: 1B‑MoE and 3B‑MoE with 64 experts (top‑2 routing; Section 4).
  - Same image encoder and connector; only LLM FFN layers become experts.

- Specializations
  - `MM1.5-Video` (Section 5):
    - Training-free mode: treat a video as 24 uniformly sampled frames, each encoded into 144 visual tokens; dynamic splitting disabled to fit token budgets.
    - SFT mode: fine-tune with ShareGPTVideo (556k), VideoChat2 (225k), ActivityNet‑QA (31.5k).
  - `MM1.5-UI` (Section 6):
    - Fine-tune general models on the Ferret‑UI mixture (801k samples) for mobile UI tasks that require OCR + grounding + commonsense about GUI widgets.

## 4. Key Insights and Innovations
- A. Data-centric SFT design that trades and balances capabilities (Sections 3.2.1–3.2.2; Figures 5–8)
  - Novelty: Treat SFT not as a single “big bag of data” but as capability-aligned categories with explicit ratios.
  - Why it matters: Enables small models (1B/3B) to achieve balanced, strong performance. Example: adding refer&ground data improves grounding “a lot” while slightly hurting other averages; the paper picks α=2.0 to optimize overall (Figure 6d). The final “All Mixture” yields the best cross-category average (Figure 8, rightmost bar).

- B. High‑resolution continual pre‑training with document‑style data is crucial (Section 3.3; Figure 9a)
  - Novelty: An additional stage between pre-training and SFT at high resolution (1344×1344) on 45M OCR-rich images.
  - Why it matters: It raises text‑rich and knowledge performance beyond what SFT alone can deliver. Using 1344×1344 beats 756×756 and 378×378; training at 378×378 can even underperform “no continual pre-training” (Figure 9a).

- C. Dynamic image splitting with global–local and flexible grids (Section 3.5; Tables 1–3; Figure 11)
  - Novelty: A compute‑aware grid selection that minimizes padding or resolution loss, plus an overview frame and optional position indicators.
  - Why it matters: Improves text-rich benchmarks substantially and is especially effective for non‑square documents and infographics; e.g., on 3B, raising `nmax` from 4 to 16 improves DocVQA by +3.1 and InfoVQA by +6.9 points (Table 2, rows 1→3). Training for the larger grid is better than only changing grid at inference (rows 2 vs. 5).

- D. Rethinking pre‑training mix: less interleaved images, more high‑quality text (Section 3.4; Figure 10)
  - Novelty: Move from 45:45:10 to `50:10:40` and upgrade text-only corpus (HQ‑Text).
  - Why it matters: After SFT, performance jumps across text‑rich (+0.85), knowledge (+0.99), and refer&ground (+~1.4). There is a small multi-image drop (−0.05) due to less interleaved data, a trade-off the paper explicitly accepts (Figure 10).

- E. Capability‑focused specializations with minimal additional machinery (Sections 5–6; Tables 10–12)
  - MM1.5 is reused “as is” for videos (training‑free) or slightly fine‑tuned for video/UI:
    - Training‑free VideoQA already beats many 7B training‑free baselines at 3B size (Table 10).
    - UI variant sets new SOTA on multiple Ferret‑UI elementary tasks, even at 1B (Table 12).

## 5. Experimental Analysis
- Evaluation Design (Sections 3.1, A.4; Tables 4–9; Figure 4)
  - Benchmarks grouped by capability:
    - General: MME, SEED‑IMG, POPE, LLaVA‑Bench (Wild), MM‑Vet, RealWorldQA.
    - Text‑rich: WTQ, TabFact, OCRBench, ChartQA, TextVQA, DocVQA, InfoVQA.
    - Knowledge: AI2D, ScienceQA, MathVista, MMMU.
    - Referring & Grounding: RefCOCO family, Flickr30k Entities, LVIS‑Ref, Ferret‑Bench.
    - Multi-image: QBench2, Mantis, NLVR2, BLINK, MVBench, MuirBench.
    - In‑context learning: VL‑ICL (6 subtasks).
  - Metrics follow each benchmark’s standard (Table 14). “Category Average Score” is the unweighted average across metrics within a category; “MMBase score” averages general + text‑rich + knowledge (Section 3.1).

- Main quantitative results
  - Small scales lead among peers (Table 4):
    - At 1B, `MM1.5‑1B` tops peers across many benchmarks. Example: TextVQA 72.5 vs LLaVA‑0.5B  — (not reported), vs InternVL2‑2B 73.4; DocVQA 81.0 vs InternVL2‑2B 86.9; RefCOCO avg 81.4 with native grounding (Table 7).
    - At 3B, `MM1.5‑3B` is competitive with or better than MiniCPM‑V2‑3B, InternVL2‑2B, and Phi‑3‑Vision‑4B on many axes (Table 4). For text‑rich specifically (Table 6):
      > “DocVQA 87.7, InfoVQA 58.5, ChartQA 74.2, TextVQA 76.5”  
      These are strong for a 3B model and close to or exceeding larger peers.
  - Scaling and MoE help (Tables 4–9):
    - Dense 1B→30B steadily improves (e.g., AI2D: 59.3→77.2; Table 5).
    - `3B‑MoE` often surpasses dense `7B` on knowledge, general, grounding, and multi‑image (Table 4), showing MoE’s parameter‑efficient scaling.
  - Referring & Grounding excellence (Table 7):
    - `MM1.5‑3B`: RefCOCO avg 85.6, Flickr30k 85.9, LVIS‑Ref avg 67.9—on par with or better than larger grounding-specialized models (e.g., Ferret‑7B). GPT‑4o relies on prompting tricks, whereas MM1.5 generates pointers natively.
  - Multi‑image and ICL (Tables 8–9):
    - VL‑ICL: `MM1.5‑30B` achieves a 77.6 average vs GPT‑4V’s 65.8 on this suite (Table 8), indicating strong multimodal in‑context learning.
    - Multi‑image: `MM1.5‑30B` yields 79.3 on QBench2, 64.6 Mantis, 90.6 NLVR2, 54.0 MVBench, 58.2 MuirBench (Table 9).
  - Text‑rich improvements at 30B (Table 6):
    - `MM1.5‑30B`: DocVQA 91.4, InfoVQA 67.3, ChartQA 83.6, WTQ 54.1, TabFact 84.0.
    - These gains align with the continual pre‑training and dynamic splitting ablations.

- Ablations that justify design choices
  - SFT category impact (Figure 5): text‑rich data boosts both text‑rich and knowledge averages; science data boosts knowledge; refer&ground adds the capability but slightly hurts other averages—hence α=2.0 chosen later (Figure 6d).
  - Ratio selection (Figures 6–7): αscience=0.1, αmath=0.5, αcode=0.2; `wmulti=0.1`, `wtext=0.1`.
  - Continual pre-training (Figure 9): high‑res 1344×1344 gives best MMBase; OCR‑only at high‑res outperforms synthetic caption alternatives at this stage.
  - Pre-training mixture (Figure 10): replacing text-only with HQ‑Text and shifting to 50:10:40 improves text‑rich (+0.85), knowledge (+0.99), and grounding (~+1.4) averages; slight multi‑image dip (−0.05).
  - Dynamic vs static splitting (Section 3.5.1; Table 1; Table 2; Appendix A.6 Tables 15–18):
    - More sub‑images and higher encoder resolution both help text‑rich; e.g., 10 sub‑images at 672² with 144 tokens/sub‑image is best among tested (Table 1, row 7).
    - Increasing `nmax` benefits DocVQA and InfoVQA; training with larger grids beats inference‑only upgrades (Table 2).
    - Despite better performance, dynamic splitting is not necessarily more expensive on average: in a 100k sample, tiles increased only from 500k (static) to 539k (dynamic) (Section 3.5.1).
  - Video (Tables 10–11):
    - Training‑free `MM1.5‑Video‑3B` achieves strong multiple‑choice scores (e.g., NExTQA 72.8; IntentQA 72.7; Table 10).
    - With SFT, `MM1.5‑Video‑1B` surpasses LLaVAOneVision‑0.5B by large margins on EgoSchema (+24.2 points) and NExTQA (+14.6) (Table 10). The 7B video model is near or at SOTA on several benchmarks.
  - UI (Table 12):
    - `MM1.5‑UI‑1B` outperforms Ferret‑UI‑13B on core elementary tasks by a wide gap—e.g., iOS Referring 90.0 vs 80.5; iOS Grounding 86.5 vs 79.4—highlighting the transfer of MM1.5’s general recipe to UI.

- Do the experiments support the claims?
  - Yes, because:
    - Each claimed design choice has an ablation (SFT mixture, continual pre-training resolution/data, pre-training ratios, dynamic splitting variants).
    - The final models win or are competitive across capability groups, especially when compared at similar parameter budgets (Tables 4–9).
  - Caveats:
    - Some test sets overlap with training sources (marked † in Tables 4–9 and noted under Figure 4), so generalization must be assessed with that in mind.
    - Open‑ended video scores use LLM‑judged metrics for some datasets (Section 5.1), which, while common practice, can introduce evaluation variance.

## 6. Limitations and Trade-offs
- Data mixture choices are tuned to specific goals
  - Reducing interleaved pre-training improves language-heavy tasks post‑SFT but slightly lowers multi-image averages (Figure 10). Projects prioritizing multi‑image might prefer higher interleaved ratios.
- Resolution vs token budget
  - Dynamic splitting lifts OCR/text‑rich performance but increases token counts for long documents; inference at higher grids (e.g., 4→16) boosts accuracy but increases latency/memory (Table 2).
- Continual pre-training dependencies
  - Gains rely on 45M curated OCR-style images at high resolution (Section 3.3). Such data volumes may be costly to acquire/host.
- Video modeling simplifications
  - Per-frame encoding without dynamic splitting and using only 24 frames may miss tiny text or long-range temporal dependencies; still, results are strong, but long videos with sparse cues remain challenging (Section 5.2).
- Grounding sensitivity
  - Inference-time changes to the minimum grid disrupt local→global coordinate conversion and can harm grounding (Table 2, row 7).
- Plateau on UI scaling
  - UI performance improvements from 7B→30B are modest, suggesting data diversity or resolution, not just parameters, limit further gains (Section 6.2).

## 7. Implications and Future Directions
- How this work changes the landscape
  - Shifts emphasis from “which model” to “which data recipe and resolution strategy” to unlock new capabilities, particularly for small, on‑device models (1B–3B) that now compete across text‑rich, grounding, and multi-image tasks (Tables 4–9).
  - Provides a concrete, end‑to‑end playbook—pre-training mix, high‑res continual pre-training, and SFT ratios—that others can adapt (Sections 3–4).
- Follow‑up research enabled
  - Unified training of image, video, and UI within a single set of weights leveraging the same dynamic splitting and data-balancing principles (Sections 5–6, Conclusion).
  - Deeper study of synthetic caption quality/style/length and their interaction with OCR‑heavy continual pre-training (Appendix A.1).
  - Smarter, learnable grid selection or token‑budget allocation per image to automate the resolution–compute trade-off.
  - Robust evaluation protocols that reduce reliance on LLM‑judged scoring for video and ensure minimal train‑test overlap.
- Practical applications
  - Document AI: invoices, contracts, scientific figures, charts (Table 6).
  - Grounded assistants and agents: UI automation, region‑aware instruction following (Table 7; Section 6).
  - Multi-image analytics: surveillance change detection, medical imaging series, retail catalog comparisons (Table 9).
  - Edge deployment: 1B/3B dense and MoE variants make on-device multimodal assistants more realistic (Table 4).

Overall, `MM1.5` demonstrates that careful data curation, high‑resolution processing, and dynamic token allocation—not just larger models—are key to mastering OCR‑heavy, grounded, and multi‑image reasoning tasks. The paper’s ablations and consistent gains across scales make its recipe a practical blueprint for building capable generalist and specialized MLLMs.
