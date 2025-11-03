# Video models are zero‑shot learners and reasoners

**ArXiv:** [2509.20328](https://arxiv.org/abs/2509.20328)
**Authors:** Thaddäus Wiedemer, Effie Li, Paul Vicol, Shixiang Shane Gu, Nick Matarese, Kevin Swersky, Been Kim, Priyank Jaini, Robert Geirhos
**Institutions:** Google DeepMind

## 🎯 Pitch

This paper demonstrates that modern video generation models like Google's Veo 3 can act as general-purpose vision foundation models, achieving zero-shot proficiency in diverse tasks such as perception, manipulation, and visual reasoning without task-specific training. The significance of this breakthrough lies in its potential to revolutionize computer vision by unifying tasks traditionally handled by separate models, akin to the transformation seen in NLP with large language models, thus paving the way for more streamlined and cost-effective visual AI applications.

---

## 1. Executive Summary (2-3 sentences)
This paper empirically tests whether modern, large video generation models can act as general-purpose “vision foundation models” without task-specific training. By prompting Google’s `Veo 3` with images plus text instructions, the study shows zero-shot competence across perception (e.g., edge detection, segmentation), modeling (intuitive physics, optics), manipulation (image editing, 3D-aware edits), and early forms of visual reasoning (mazes, symmetry, analogies)—with consistent gains over `Veo 2` (Secs. 3–4; Figs. 1–9).  

## 2. Context and Motivation
- Problem or gap
  - Computer vision still relies heavily on task-specific systems (e.g., separate models for detection, segmentation, editing), whereas NLP has converged on unified large language models that can be prompted for many tasks. The core question here is whether “video-first” generative models—trained at web scale to continue and edit visual sequences—can similarly generalize by prompt, i.e., do zero-shot learning and reasoning (Sec. 1).
- Why this matters
  - Practical: A single prompted model could replace many narrow pipelines (lower integration cost, faster iteration, broader coverage).
  - Scientific: If sequence-generation over frames can support stepwise problem solving (“chain-of-frames”), it parallels chain-of-thought in language and may open new paths to visual reasoning (Sec. 3, “Visual reasoning across time and space”).
- Prior approaches and limitations
  - Strong task specialists exist (e.g., SAMv2 for segmentation [12], YOLO for detection [13,14]) but require prompts like clicks/boxes or retraining to switch tasks. Some unifying image models and visual in‑context approaches exist [15–25, 55–58] yet still fall short of a promptable “do-anything” visual model.
  - Earlier video editors and generators (e.g., Dreamix [24]) focus on editing or style rather than broad, zero-shot problem solving across perception → modeling → manipulation → reasoning.
- Positioning
  - The paper evaluates an off-the-shelf, black-box video generator (`Veo 3`) by prompting it for 62 qualitative and 7 quantitative tasks (18,384 generated videos, plus 744 for qualitative samples; Table 1) without any task-specific training or heads (Secs. 2–4).
  - It introduces the idea of “chain-of-frames (CoF)”—frame-by-frame, temporally extended reasoning analogous to chain-of-thought in LLMs (Sec. 3, “Visual reasoning across time and space”).

## 3. Technical Approach
The paper’s methodology is intentionally minimal: treat the production `Veo 3` model as a black box and “just prompt it” for many tasks, then measure success. Key elements:

- What model is tested and how
  - Models: `Veo 3` (`veo-3.0-generate-preview`) and `Veo 2` (`veo-2.0-generate-001`) via Google Cloud Vertex AI (Sec. 2, “Video generation”). For editing/image baselines and dataset creation, `Nano Banana` (Gemini’s image generation/editing) is used; `Gemini 2.5 Pro` is used for text-based maze solving and as an auto-rater in the analogy evaluation (Secs. 2, 4; B.5–B.7).
  - Each run uses an input image as the first frame plus a natural-language instruction. Video specs: 16:9, 720p, 24 FPS, 8 seconds (Sec. 2).
  - Note: Vertex AI uses an LLM-based “prompt rewriter” before the video model (Sec. 2; [30]); thus, the end-to-end system includes an LLM front-end and the Veo video generator. To check that reasoning isn’t simply “done by an LLM,” the paper verifies that a standalone LLM (Gemini 2.5 Pro) cannot reliably solve key visual tasks (e.g., maze and symmetry) from an image alone (Sec. 2).
- Task suite and evaluation philosophy
  - Four capability tiers (Fig. 1; Sec. 3): Perception → Modeling → Manipulation → Reasoning.
  - Qualitative sweep (Sec. 3; Appendix A): For each of 62 tasks, prompt 12 times and report the fraction of generated videos that solve the task (author judgment). Purpose: breadth discovery and capability mapping.
  - Quantitative evaluation (Sec. 4): Seven tasks with measurable criteria and baselines; 10 attempts per sample; report performance for the “best frame” and the “last frame.”
    - “Best frame” = max performance achieved by any frame in any sample (upper bound if you can pick the right frame).
    - “Last frame” = predetermined end state (more realistic for automation but may be worse because videos often keep animating after “finishing” a task; Sec. 4).
    - “pass@k” = probability of at least one success across k independent attempts (k ∈ [1,10]).
- Datasets, metrics, and automatic graders (details in Appendix B)
  - Edge detection: BIPEDv2 test set (50 images), metric `OIS` (best F1 over thresholds), with non-max suppression and edge-matching tolerance (B.1; Fig. 3). `OIS` stands for “Optimal Image Scale,” commonly used in edge detection to summarize best-threshold F1.
  - Instance segmentation: LVIS subset (50 easy images with 1–3 large objects). Metric `mIoU` (mean Intersection over Union) after automatically extracting flat-color masks from generated frames (B.2; Fig. 4).
  - Object extraction: A small “animal-count” dataset the paper generated with `Nano Banana` (54 images). Metric: count of connected components after a white-background extraction; success if the count matches ground truth (B.3; Fig. 5).
  - Image editing: 30 Emu-Edit samples; 3 human raters judge fidelity (“did the requested edit happen?”) and precision (“did it happen without unintended changes like camera motion?”) (B.4; Fig. 6).
  - Maze solving: 50 randomly generated rectangular mazes per size (5×5, 7×7, 9×9) and 40 irregular mazes drawn by hand. Automatic path validation checks for illegal moves or wall crossings. Baselines: `Nano Banana` (draws a path in a single edit) and `Gemini 2.5 Pro` with either image-to-text (I2T) or ASCII text-to-text (T2T) inputs (B.5; Fig. 7).
  - Visual symmetry: A 10×16 grid with left-half painted; task is to mirror to the right-half. Automatic grading compares per-cell colors in perceptual LAB space with a tolerance threshold (B.6; Fig. 8).
  - Visual analogies (KiVA): 2×2 analogies with four transformations—`color`, `resize`, `reflect`, `rotate`. The model must fill in the missing quadrant. An auto-rater (Gemini 2.5 Pro) compares the generated fill against three candidate choices; author pilot study shows >88% agreement with human judgments (B.7; Figs. 9, 61).
- Why this design
  - The “prompt-only” approach mirrors the LLM transition from fine-tuned task heads to instruction following. It tests if large video models already display “emergent” skills (Sec. 2).
  - The “best vs last frame” split distinguishes capability (can the model get there at any point?) from practical usability (is the final frame right without post-selection? Sec. 4).
  - Multiple attempts and pass@k reflect the widely used inference-time scaling practices in LLMs (e.g., self-consistency), probing whether “try a few times” helps here too (Sec. 4; Figs. 3–9).

## 4. Key Insights and Innovations
- A single video model shows zero-shot breadth across the vision stack (qualitative)
  - The paper demonstrates many classic CV tasks—edge detection (Fig. 10), segmentation (Fig. 11), keypoint localization (Fig. 12), super-resolution (Fig. 13), denoising (Fig. 15), low-light enhancement (Fig. 16)—as well as more conceptual tasks like the Dalmatian illusion (Fig. 18) and visual search/binding (Fig. 17)—all via prompts, no task retraining (Sec. 3).  
  - Significance: This suggests generative video pretraining may internalize versatile visual processing rules beyond its nominal objective (generation).
- “Chain-of-Frames” (CoF) as a visual analog of chain-of-thought
  - By generating frame-by-frame, Veo can effect stepwise visual transformations—useful for problems like mazes, graph traversals, and sequence completions (Sec. 3; Figs. 48–59).  
  - Significance: CoF provides a mechanism for visual reasoning over time and space, not just static recognition (Takeaway 3 in Sec. 3).
- Quantitative evidence of rapid capability scaling from `Veo 2` to `Veo 3`
  - Across tasks (edge detection, segmentation, object extraction, editing, mazes, symmetry, analogies), `Veo 3` consistently outperforms `Veo 2` (Sec. 4; Figs. 3–9).  
  - Significance: The pace of improvement suggests the “LLM-like” progression may also be happening in video models.
- Prompting best practices and sensitivity
  - The paper systematically probes prompt phrasing for symmetry (Table 2, Sec. C) and notes that simple changes (“green screen” background, or adding a “motion outlet” to satisfy the model’s animation prior) materially affect success rates (Secs. 3–4, C).  
  - Significance: As in NLP, instruction design and visual context (the first frame) are powerful levers for performance—important for practical deployment.

## 5. Experimental Analysis
- Evaluation setup and scope
  - Total videos: 17,640 for the quantitative tasks (Table 1) plus 744 for the qualitative gallery. Best-vs-last frame scoring and pass@k measure both capability and practical reliability (Sec. 4).
- Main quantitative results (with specifics)
  - Edge detection (BIPEDv2, OIS; Fig. 3)
    - Quote: “Veo 3 (0.77 pass@10) … Veo 2 (0.57 pass@10) … task-specific SOTA: 0.90” (Fig. 3).  
    - Finding: While below a dedicated edge detector, `Veo 3`’s zero-shot OIS is remarkably close—especially given some “false positives” are actually plausible edges missing from the dataset (Fig. 60).
  - Instance segmentation (LVIS subset, mIoU; Fig. 4)
    - Quote: “Veo 3 achieves mIoU of 0.74 (best-frame pass@10) … Nano Banana: 0.73 … with green background better than white (0.74 vs 0.66)” (Fig. 4; Sec. 4.2).  
    - Finding: Zero-shot `Veo 3` is competitive with a strong image editor on easy scenes and sensitive to background color, plausibly due to “green screen” priors.
  - Object extraction (animal-count extraction; Fig. 5)
    - Quote: “Veo 3 achieves up to 93% pass@10 on last frame; Veo 2 near chance” (Fig. 5; Sec. 4.3).  
    - Method detail: Count connected components after background whitening (B.3).
  - Image editing (Emu-Edit subset; human ratings; Fig. 6)
    - Quote: “Fidelity: Veo 3 ≈ 100%, Veo 2 ≈ 20%; Precision: Veo 3 ≈ 60%, Veo 2 ≈ 10%” (Fig. 6).  
    - Interpretation: Veo 3 usually makes the intended edit and preserves fine detail well; unintended changes (e.g., camera motion, animating people) still occur, lowering precision (Sec. 4.4).
  - Maze solving (rectangular and irregular; Fig. 7)
    - Quote: “On 5×5, Veo 3 pass@10 = 78% vs Veo 2 = 14%” (Sec. 4.5). On irregular mazes, `Veo 3` succeeds while `Nano Banana` “fails entirely” (Fig. 7 caption).  
    - Baselines: `Gemini 2.5 Pro` performs well with ASCII (T2T) on small mazes, but falls off on 9×9 and struggles with image-as-input (I2T), highlighting the benefit of solving visually via CoF (Fig. 7).
  - Visual symmetry (grid reflection; Fig. 8)
    - Quote: “Shapes: Best-frame pass@10 = 88%, Last-frame pass@10 = 44%; Random patterns: Best-frame pass@10 = 100%, Last-frame pass@10 = 72%” for `Veo 3` (Fig. 8).  
    - Prompt sensitivity: Best/worst prompt varied pass@1 by 40–64 percentage points depending on split (Table 2; Sec. C).
  - Visual analogies (KiVA; Fig. 9 and Fig. 61)
    - Quote: “Veo 3 pass@1: color=95%, resize=67%, reflect=29% (<33% chance), rotate=19% (<33% chance); Veo 2: color=68%, resize=40%, reflect=23% (<33%), rotate=22% (≈chance)” (Fig. 9).  
    - Majority-vote across attempts can hurt when biases are systematic: performance decreases with k for reflect/rotate (Fig. 61), implying consistent but wrong transforms (e.g., mirroring along the wrong axis).
- Do the experiments support the claims?
  - Breadth: The qualitative gallery (Appendix A) is extensive, spanning perception to reasoning, with nontrivial tasks like graph traversal (Fig. 48), tree BFS (Fig. 49), and Sudoku (Fig. 55).
  - Zero-shot competence: Quantitative tasks show clear improvements over `Veo 2` and non-video baselines in settings where visual, stepwise manipulation is crucial (mazes—Fig. 7; symmetry—Fig. 8).
  - Caveats addressed:
    - The system includes a prompt rewriter LLM. For tasks like Sudoku, the LLM may contribute; to isolate video reasoning, the paper shows a standalone LLM cannot solve core visual tasks from the same images (Sec. 2).
    - Performance is sensitive to prompt and representation; reported numbers are a lower bound on capability (Sec. 5 “Performance is a lower bound”).
- Failure cases and robustness checks
  - The paper devotes Appendix D to failures: depth & normals (Figs. 62–63), force/motion prompting (Fig. 64), knot tying and collisions (Figs. 65, 73), planning tasks (sofa through doorway—Fig. 77), puzzles (spot-the-difference—Fig. 70).  
  - These reveal limitations in metric understanding (exact geometry), physics fidelity, and adherence to strict constraints when visual cues are subtle or when the scene invites hallucinations.

## 6. Limitations and Trade-offs
- Black-box system with an LLM prompt rewriter
  - The measured capability is of the overall Vertex AI pipeline (prompt rewriter + video generator), not a pure video model in isolation (Sec. 2). Sudoku (Fig. 55) likely benefits from LLM text reasoning. The authors mitigate this by checking that the LLM alone can’t solve certain visual tasks from images (mazes, symmetry), but some attribution ambiguity remains.
- Motion prior and last-frame instability
  - Veo strongly prefers animation, which can degrade last-frame performance after a correct intermediate state (Sec. 4; symmetry Fig. 8, big gap between best and last). Prompts often need “static camera,” “no zoom,” or a “motion outlet” (e.g., spinning indicator) to keep the solution intact (Sec. C).
- Physics and geometry fidelity
  - Failures on depth/normal maps, collisions, knot tying, and motion planning (Figs. 62–73, 75–77) indicate the learned “world model” is not yet precise enough for fine-grained physical reasoning or constrained planning.
- Representation dependence and prompt sensitivity
  - Performance varies with visual framing (e.g., green-screen background helps segmentation; Fig. 4) and phrasing (Table 2). This creates operational overhead and tuning needs.
- Cost and scalability
  - Video generation is currently more expensive than running a narrow, specialized model, though the paper argues inference costs are falling rapidly (Sec. 5 “Video generation is expensive, but costs tend to fall”).
- Evaluation scope
  - While broad, some domains (e.g., precise 3D reconstruction, metric depth, formal geometric proofs) are not included. Modeling evaluations rely mainly on existing literature for context rather than new benchmarks (Sec. 4 preface).

## 7. Implications and Future Directions
- Field impact
  - If a single, prompted video model can cover perception → modeling → manipulation → reasoning, computer vision may undergo the same unification that NLP experienced. The “CoF” lens suggests that time-as-a-computation-axis can enable visual problem solving that static image models struggle with (Sec. 3 Takeaway 3).
- Practical applications
  - Unified visual assistants for editing, design, prototyping, and instruction videos; rapid “what-if” simulation for user interfaces and creative tools; visual planning in simplified environments; multimodal agents that reason by “showing” intermediate steps (Figs. 39–47, 48–59).
- Research directions
  - Attribution and architecture: disentangle and benchmark the contributions of the prompt rewriter vs. the video generator; design native CoF planning modules and verifiers.
  - Reliability upgrades: instruction tuning/RLHF for video (Sec. 5 cites NLP analogs [76,77]); inference-time scaling (self-consistency, self-refine) adapted to video generation (Sec. 5, refs. [72–75]).
  - Physics-grounded training: targeted data or auxiliary losses for depth, normals, rigid/smooth collisions, and tool-use constraints; integrate motion-trajectory/force annotations that current models ignore (failures in Fig. 64).
  - Prompting and representation: formalize “visual prompt engineering” (Sec. C), including standardized initial frames and task-specific camera/scene constraints.
  - Benchmarks for CoF reasoning: rigorous, automatically gradable suites for mazes, symmetry, visual algorithms (sorting, BFS), and analogy transformations with minimal LLM leakage.
  - Cost–accuracy trade-offs: study how many attempts (k) and frame-selection strategies yield the best accuracy per dollar, and develop last-frame stabilization techniques.

> Bottom line (Sec. 5 Summary): Veo 3’s zero-shot performance across diverse tasks, plus rapid gains over Veo 2, suggest video models are on a credible path to becoming general-purpose, promptable vision foundation models—much like LLMs in language—though physics fidelity, prompt sensitivity, and attribution to the video model vs. LLM front-end remain important open areas.
