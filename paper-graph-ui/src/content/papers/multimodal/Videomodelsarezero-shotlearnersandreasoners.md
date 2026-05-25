# Video models are zero-shot learners and reasoners

**ArXiv:** [2509.20328](https://arxiv.org/abs/2509.20328)

## 🎯 Pitch

This paper demonstrates that large, generative video models like Veo 3 are emergent zero-shot learners and reasoning engines for vision: given only an input image and a prompt, these models can solve a wide array of perception, physics, manipulation, and visual reasoning tasks without task-specific training. By revealing 'chain-of-frames' visual reasoning—akin to language models' chain-of-thought—the study signals a transformative shift: video models are poised to unify computer vision under a single, generalist foundation model, with huge implications for both research and real-world applications.

---

## 1. Executive Summary
This paper investigates whether modern text-to-video models can act as general-purpose, zero-shot visual problem solvers. Using Google DeepMind’s Veo 3, the study shows that with only an input image and a natural-language prompt, the model can perform a wide variety of perception, modeling (physics/world knowledge), manipulation (editing/3D-aware changes), and step-by-step visual reasoning tasks, often without task-specific training. The significance is twofold: it suggests a shift in computer vision from many bespoke models to a unified “foundation model” paradigm, and it introduces “chain-of-frames” reasoning—analogous to chain-of-thought in language models—where the model “thinks” by generating video frames over time (Sec. 3; Figs. 1–2).

## 2. Context and Motivation
- Problem/gap addressed
  - Computer vision today remains largely a collection of specialized models (e.g., SAM for segmentation, YOLO for detection). Few approaches can solve arbitrary vision tasks by prompt alone. The paper asks whether large, generative video models—trained with simple objectives on web-scale data—already exhibit the kind of general-purpose, zero-shot abilities that transformed NLP (Intro; Sec. 1).
- Why it matters
  - A generalist visual model would unify tasks (perception → modeling → manipulation → reasoning) and reduce integration overhead in practical systems (editing, robotics, planning). It may also yield new reasoning affordances—solving problems step-by-step in time and space, rather than symbolically (Sec. 3; “Takeaway 3”).
- Prior approaches and limitations
  - Task-specific SOTAs exist for edges, segmentation, detection, etc. [11–14, 36], and there are early attempts at generalist image/video generation or in-context visual learning [15–25], but these typically require adaptation or are not evaluated as universal, promptable problem solvers.
- Positioning
  - The study treats a state-of-the-art video generator (Veo 3) as a black-box foundation model and probes its breadth of zero-shot capabilities through careful prompting, qualitative exploration (62 tasks) and seven quantitative benchmarks (Secs. 3–4; Fig. 1, Table 1 in App. B). It also compares to Veo 2 (prior version), an image editor baseline (“Nano Banana”), and a strong LLM (Gemini 2.5 Pro) for text- and image-based variants of tasks like mazes (Sec. 4; Fig. 7).

Definition: zero-shot learning in this paper means solving a task purely from a natural-language instruction prompt (plus an optional input image) without fine-tuning or adding task-specific heads (Intro; Sec. 2).

## 3. Technical Approach
The core method is deliberately minimal—prompting the video model—and then extracting and scoring the resulting frames (Sec. 2, Sec. 4; App. B).

- System setup
  - Models: Veo 3 (`veo-3.0-generate-preview`) and Veo 2 (`veo-2.0-generate-001`) via Google Cloud Vertex AI (Sec. 2). The API includes a prompt rewriter based on an LLM [30]; the study evaluates the rewriter+video model as a single system. To isolate visual reasoning, separate checks show a standalone LLM (Gemini 2.5 Pro) cannot reliably solve certain visual tasks from the image alone (e.g., robot navigation, maze symmetry; Sec. 2, Secs. 4.5–4.6).
  - Generation: For each task, supply (i) an initial image (first frame) and (ii) a text instruction; the model outputs an 8s, 16:9, 720p, 24 FPS video (Sec. 2).
  - Key evaluation conventions:
    - `pass@k`: best performance achieved over k independent attempts (k ≤ 10), a common inference-time scaling measure (Sec. 4; figures throughout).
    - “Best frame” vs “Last frame”: Many tasks complete early; continuing animation can degrade the final frame. The study reports both the best frame across all frames and the predetermined last frame (Sec. 4).
- Task families and how they’re prompted (Sec. 3; App. A–B)
  1) Perception tasks (edges, segmentation, keypoints, denoising, etc.) are induced by prompts that ask the model to transform the input into diagnostic renderings (e.g., “outline edges in black, fade everything else,” Fig. 10).
  2) Modeling tasks (intuitive physics, optics, color mixing, memory of scene state) are elicited by prompts that request realistic, causally plausible temporal evolutions (e.g., buoyancy: “let go of object in water,” Fig. 24; optics: “roll a glass sphere,” Fig. 27).
  3) Manipulation tasks (background removal, in/outpainting, style transfer, 3D reposing) are framed as edits while constraining camera/scene (“static camera, no zoom/pan,” Figs. 32–41).
  4) Visual reasoning tasks (mazes, symmetry, analogies, BFS/traversal) are set up so that the desired “solution” is a sequence of visible steps over frames—“chain-of-frames” (CoF). For example, in mazes, a red circle must move along white corridors to a green goal without illegal crossings (Sec. 4.5; Fig. 7).
- Quantitative evaluation pipeline (Sec. 4; App. B)
  - Edge detection (BIPEDv2): Convert generated “edge videos” to binarized edge maps; evaluate Optimal Image Scale (OIS) F1 after thinning and tolerant matching (App. B.1; Fig. 3, Fig. 60).
  - Segmentation (LVIS subset): Ask the model to overlay each distinct object with a unique flat color; recover masks by clustering hue; compute mIoU with GT instance masks (App. B.2; Fig. 4).
  - Object extraction: Custom dataset with 1–9 animals in a photo. The prompt asks the model to place all animals in a row on white; count connected components in the last frame and compare to the true count (App. B.3; Fig. 5).
  - Editing (Emu-Edit subset): 3 human raters judge fidelity (correct edit happened) and precision (no unintended changes like camera motion) (App. B.4; Fig. 6).
  - Mazes: 50 mazes per size (5×5, 7×7, 9×9) plus 40 “irregular” mazes. Verify paths frame-by-frame: no wall-crossing, start→goal continuity (App. B.5; Fig. 7). Also evaluate Nano Banana (drawn path) and Gemini 2.5 Pro with image (I2T) or ASCII (T2T) inputs.
  - Visual symmetry: Complete a 10×16 grid to be mirror-symmetric about the vertical axis; score by perceptual color differences in CIELAB (threshold 15) at the cell level (App. B.6; Fig. 8).
  - Visual analogies (KiVA): Prompt the model to fill the missing quadrant; use an LLM-based autorater to compare the generated object to multiple-choice candidates; report pass@1 (App. B.7; Fig. 9, Fig. 61).
- Prompt engineering and control variables
  - The study systematically explores prompt sensitivity for symmetry: pass@1 varied by up to 40 percentage points (shapes) and 64 points (random) across 10 prompt variants (App. C, Table 2).
  - Practical tips: Specify what must not change, provide a “motion outlet” to stop further edits once a solution appears, and explicitly constrain camera motion (App. C).

Definition: `chain-of-frames (CoF)` is the idea that the video generator can “reason” by applying a sequence of visible, intermediate changes across frames—analogous to chain-of-thought in LLMs but grounded in time and space (Sec. 3, “Takeaway 3”).

## 4. Key Insights and Innovations
1) A single video model exhibits broad, zero-shot competence across the vision stack
   - Novelty: Perception, modeling, manipulation, and visual reasoning tasks are induced by prompts alone—no fine-tuning, no task-specific heads (Sec. 3; Figs. 1–2, App. A).
   - Why it matters: This mirrors LLMs’ transition in NLP from many bespoke systems to a unified foundation model (Intro; “Takeaway 1”).
2) Chain-of-frames (CoF) as a mechanism for visual reasoning
   - Novelty: The model solves constrained problems (e.g., mazes, symmetry, graph traversal) by visibly progressing step-by-step in generated frames (Sec. 3; Figs. 7–9, 48–59).
   - Significance: CoF leverages spatial-temporal continuity as an alternative to symbolic reasoning; in several tasks, it outperforms image-only or text-only baselines (Sec. 4.5, Fig. 7).
3) Evaluation methodology that treats generative outputs as task solutions
   - Novelty: For edges and segmentation, the study converts stylized videos back into measurable predictions (e.g., hue clustering for masks); for reasoning tasks, it automatically verifies legal motion/path constraints (App. B).
   - Significance: This creates a reusable blueprint for quantitatively scoring generalist video models without retraining.
4) Rapid capability gains from Veo 2 to Veo 3
   - Evidence: Large jumps across tasks—edges (OIS +0.20 absolute at pass@10 best frame; Fig. 3), mazes (5×5 pass@10: 78% vs 14%; Fig. 7), symmetry (best@10: up to 88–100%; Fig. 8), and analogies (color/resize markedly improved; Fig. 9).
   - Meaning: Suggests that scaling and model improvements can quickly close the gap to task-specific systems, echoing NLP trajectories (Sec. 4; Discussion).

## 5. Experimental Analysis
- Setup summary (App. B, Table 1)
  - 17,640 videos for the seven quantitative tasks (plus 744 more for 62 qualitative tasks). Each quantitative task typically uses 10 attempts per input and reports best- and/or last-frame metrics.
- Datasets, metrics, baselines
  - Perception
    - Edge detection on BIPEDv2 (50 images), metric: OIS F1 with tolerant matching (App. B.1; Fig. 3).
    - Instance segmentation on 50 “easy” LVIS images (1–3 large objects), metric: mIoU across instances (App. B.2; Fig. 4).
  - Manipulation
    - Object extraction with a custom 1–9 animal dataset (54 images), metric: exact count via connected components (App. B.3; Fig. 5).
    - Emu-Edit subset (30 images), metric: 3-rater human judgments of fidelity and precision (App. B.4; Fig. 6).
  - Reasoning
    - Mazes: 50 per grid size + 40 irregular; metric: legal path from start to goal (App. B.5; Fig. 7).
    - Visual symmetry: 25 shapes + 25 random patterns; metric: 0 cell errors in CIELAB-perceived color space (App. B.6; Fig. 8).
    - Visual analogies (KiVA): 4 transformation types (color, resize, reflect, rotate); metric: pass@1 using an autorater (App. B.7; Fig. 9, Fig. 61).
  - Baselines: Veo 2, Nano Banana (image editing model), Gemini 2.5 Pro (I2T image input; T2T ASCII text).
- Main quantitative results (Sec. 4; Figs. 3–9)
  - Edges
    > Fig. 3 (best frame, pass@10): Veo 3 OIS 0.77 vs task-specific SOTA 0.90; Veo 2 0.57; Nano Banana 0.74.  
    Interpretation: Remarkable zero-shot performance; some “false positives” are actually finer detail than GT annotations (Fig. 60).
  - Segmentation
    > Fig. 4 (best frame, pass@10, green background): Veo 3 mIoU 0.74; Nano Banana 0.73; Veo 2 0.52.  
    Prompt sensitivity: green background outperforms white (0.74 vs 0.66), possibly leveraging a “green screen” prior.
  - Object extraction (counting)
    > Fig. 5 (last frame, pass@10): Veo 3 93% vs Veo 2 near chance.  
    Interpretation: The task is simple; 100% should be achievable by a perfect system—headroom remains.
  - Editing (human study)
    > Fig. 6: Fidelity: Veo 3 ~100% vs Veo 2 ~20%; Precision (no unintended changes): Veo 3 ~60% vs Veo 2 ~10%.  
    Interpretation: Veo 3 often makes the correct edit but may introduce side effects (camera motion, animation). Precision is the bottleneck.
  - Mazes
    > Fig. 7 (5×5, pass@10): Veo 3 78% vs Veo 2 14%; Nano Banana ≈ matches/surpasses on rectangular mazes but fails entirely on irregular. Gemini 2.5 Pro does well on small ASCII mazes (T2T) but degrades at 9×9 and struggles with image mazes (I2T).  
    Takeaway: Visual step-by-step CoF provides an advantage, especially on non-grid, irregular layouts.
  - Visual symmetry
    > Fig. 8: Shapes—Veo 3 best@10 up to 88% (last@10 44%); Random—Veo 3 best@10 100% (last@10 72%); both far above Veo 2 and Nano Banana.  
    Also: Prompt wording matters a lot (App. C, Table 2).
  - Visual analogies (KiVA)
    > Fig. 9 (pass@1 on last frame): Veo 3—color 95%, resize 67%, reflect 29%, rotate 19% (last two below chance 33%); Veo 2 generally worse.  
    Fig. 61 shows that majority voting across attempts can reduce accuracy for reflect/rotate, indicating systematic bias (e.g., mirroring across the wrong axis).
- Do the experiments substantiate the claims?
  - Breadth: The 62-task qualitative survey (Fig. 1; App. A) plus seven quantitative tasks cover perception → reasoning. Many are solved in zero-shot form with nontrivial reliability (e.g., maze success scaling with k; segmentation nearing a strong image editor).
  - Improvement over Veo 2: Clear, consistent gains across tasks (Figs. 3–9).
  - CoF advantage: Image-only or text-only baselines underperform on visual reasoning tasks that benefit from step-by-step visual progress (Fig. 7).
- Robustness and failure analysis
  - Failures include depth/normal estimation, following explicit force/trajectory annotations, certain physics (glass breaking), strict combinatorial constraints (Eulerian path), precise word search, and complex motion planning (App. D; Figs. 62–77).
  - Prompt sensitivity is substantial (App. C, Table 2), reinforcing that measured “performance” can be a lower bound on “competence” (Discussion).

## 6. Limitations and Trade-offs
- System composition and attribution
  - The Vertex API employs an LLM prompt rewriter; some successes (e.g., Sudoku in Fig. 55) may partly arise from the LLM subsystem. The paper mitigates this by testing a standalone LLM on image-only variants of key tasks (Sec. 2), but full disentanglement is not possible without internal access.
- Prompt sensitivity and evaluation choice
  - Performance heavily depends on visual/textual prompt design (App. C). Results should be viewed as lower bounds that can improve with better prompting (Discussion).
- Precision vs creativity
  - Veo 3 tends to animate or alter scenes unless explicitly constrained, harming last-frame metrics and precision for editing (Fig. 6; App. C tips).
- Physics and control limitations
  - The model struggles with strict physical control signals (force, trajectory), material realism (glass shattering), and topologically constrained puzzles (Eulerian paths), suggesting incomplete world modeling and control grounding (App. D; Figs. 62–66, 68, 72–73).
- Computational cost and scalability
  - Video generation is currently more expensive than running specialized models, though inference costs are trending down rapidly in LLMs; a similar trajectory is anticipated for video models (Discussion; [71]).
- Evaluation caveats
  - Some autorating relies on Gemini 2.5 Pro (analogies; App. B.7), introducing a dependency on another model. Agreement checks are reported (>88% with human judgments on small samples), but full-scale human evaluation would be stronger.

## 7. Implications and Future Directions
- Field-level impact
  - The results point toward a “GPT-3 moment” for vision: large, generative video models—trained on broad data with simple predictive objectives—can already act as zero-shot, promptable solvers across diverse tasks (Discussion; Sec. 3–4). CoF offers a new path to visual reasoning grounded in spatiotemporal continuity rather than symbols alone.
- Follow-up research
  - Improving controllability to raise precision (e.g., camera/scene locks, termination signals) and integrating verifiers/iterative refinement (self-consistency and test-time scaling [72–75]) could close gaps to SOTA.
  - Disentangling the LLM rewriter from the video model to diagnose where reasoning occurs, and adding explicit tools for physics-consistent control (cf. failures in App. D; force/motion prompting).
  - Formalizing chain-of-frames: benchmarks and methods that explicitly leverage intermediate visual steps, akin to chain-of-thought datasets for language.
  - Prompt-curricula and visual prompt engineering: the symmetry prompt study (App. C) suggests systematic gains from well-designed instructions and “motion outlets.”
- Applications
  - Unified visual assistants for editing, design, and content creation; prototyping and simulation for robotics (e.g., affordance visualization, simple navigation; Figs. 45, 58); education and training tools that “show” solutions step-by-step; vision-centric agents that plan by simulating and inspecting candidate futures in video.

In short, the study demonstrates that a modern video generator (Veo 3) already functions as a generalist, zero-shot visual problem solver across perception-to-reasoning tasks. While precision, physics fidelity, and controllability need work, the rapid gains from Veo 2 to Veo 3, plus the effectiveness of CoF, strongly suggest video models are on a credible path to becoming the foundation models of machine vision.
