# On‑Policy Distillation of Language Models: Learning from Self‑Generated Mistakes

**ArXiv:** [2306.13649](https://arxiv.org/abs/2306.13649)
**Authors:** Rishabh Agarwal, Nino Vieillard, Yongchao Zhou, Piotr Stanczyk, Sabela Ramos, Matthieu Geist, Olivier Bachem
**Institutions:** 

## 🎯 Pitch

Introducing Generalized Knowledge Distillation (GKD), this paper offers an on-policy distillation framework that trains auto-regressive language models on student-generated sequences with real-time teacher feedback, effectively reducing train–inference mismatch. By enabling flexible divergences, GKD consistently enhances performance across tasks like summarization and translation, integrating effortlessly with RL fine-tuning, thus empowering more efficient and accurate deployment of language models in resource-constrained environments.

---

## 1. Executive Summary (2-3 sentences)
This paper introduces Generalized Knowledge Distillation (`GKD`), an on-policy distillation framework for auto-regressive language models that trains a smaller student model on its own generated sequences while leveraging token-level feedback from a larger teacher. GKD reduces the train–inference mismatch that plagues prior distillation methods and allows flexible divergences (e.g., reverse KL, generalized JSD), producing consistent improvements across summarization, translation, reasoning, and instruction tuning and integrating cleanly with RL fine-tuning.

## 2. Context and Motivation
- Problem addressed
  - Distilling large language models into smaller ones often trains the student on a fixed dataset of outputs (ground-truth or teacher outputs). In auto-regressive generation, the student’s inference-time prefixes differ from the training prefixes, causing train–inference distribution mismatch (also termed exposure bias in sequence modeling). This mismatch accumulates errors during decoding and reduces generation quality (Section 3; imitation learning perspective in Ross et al., 2011).
  - Additionally, standard distillation minimizes the forward KL divergence between teacher and student token distributions. When the student has lower capacity, forward KL can force it to allocate probability mass to tokens that the teacher would rarely generate, potentially exacerbating hallucinations (discussion in Section 3.1; Figure A.16 illustrates mode-covering vs. mode-seeking).

- Why it matters
  - Compressing LLMs is crucial for deployment due to memory and inference cost constraints (Introduction).
  - Reducing exposure bias in distillation can markedly improve real-world generation quality in tasks like summarization, translation, and multi-step reasoning.

- Prior approaches and their gaps
  - Supervised fine-tuning (SFT): trains only on ground-truth targets; no teacher signal (Section 3).
  - Supervised KD: matches token-level teacher probabilities on fixed datasets (Eq. 3); still suffers train–inference mismatch.
  - Sequence-level KD (SeqKD; teacher-generated sequences): expensive to pre-generate and still fixed; mismatch persists once the student diverges (Section 3).
  - ImitKD and f-distill: partially mix student- and dataset-generated sequences or change the divergence, but do not fully embrace on-policy training nor integrate with RL (Related Work; Figures 2, 9).

- Positioning
  - GKD reframes distillation as an imitation learning problem with an interactive expert (teacher) and trains on student-generated (on-policy) data. It unifies supervised KD, SeqKD, and mixed-data variants as special cases by exposing two knobs: the data source mixture and the choice of divergence (Algorithm 1; Section 3.1). It also plugs into RL fine-tuning with a single regularized objective (Eq. 5; Section 3.2).

## 3. Technical Approach
Step-by-step overview of GKD

1) Setup and notation (Section 2)
- Auto-regressive generation: an output sequence `y = (y1, y2, …, yL)` is generated token by token with policy `p(yn | y<n, x)`. Teacher `pT` and student `pSθ` are auto-regressive models.
- Token-level divergence for a given input `x` and output `y` averages divergence over positions (Eq. 2):
  - D(pT || pSθ)(y|x) = (1/Ly) Σn D(pT(·|y<n, x) || pSθ(·|y<n, x)).

2) Generalized KD objective (Section 3.1; Algorithm 1)
- Data sources
  - Fixed dataset `(X, Y)` (ground-truth or teacher-generated).
  - On-policy student samples: For inputs `x ~ X`, sample `y ~ pSθ(·|x)` at temperature 1 during training to ensure diversity.
- Mixture
  - A scalar `λ ∈ [0,1]` controls the student data fraction:
    - `λ = 0`: purely supervised on fixed sequences.
    - `λ = 1`: purely on-policy (student-generated).
    - Intermediate values: mixed.
- Choice of divergence `D` between token distributions:
  - Forward KL, reverse KL, and generalized Jensen–Shannon divergence `JSD(β)` (Eq. 1), which interpolates between forward (β→0) and reverse (β→1) KL.
- Objective (LGKD; Section 3.1):
  - LGKD(θ) = (1−λ) E_(x,y)∼(X,Y) [D(pT || pSθ)(y|x)] + λ E_x∼X E_y∼pS(·|x) [D(pT || pSθ)(y|x)].
- Key training detail
  - No backpropagation through the student sampling process `y ~ pS` (“stop-gradient” through sampling), which stabilizes and simplifies optimization (Section 3.1).

> Algorithm 1 (Section 3.1) operationalizes this: at each step, with probability `λ` it generates student samples; otherwise it uses the fixed dataset; then it updates θ by descending the gradient of the selected divergence on those sequences.

3) Why on-policy helps (mechanism)
- Imitation-learning perspective: The student is trained on exactly the prefixes it will encounter at inference (its own), and the teacher provides token-level probabilities on those prefixes (expert feedback). This directly tackles distribution mismatch and limits error compounding during decoding (Section 3.1).

4) Choice of divergence (Section 3.1; Figure A.16)
- Forward KL is mode-covering; with capacity-limited students it can allocate probability to low-teacher-probability tokens, risking hallucinations.
- Reverse KL is mode-seeking; it concentrates probability on high-teacher-probability tokens, improving faithfulness but potentially reducing diversity.
- JSD(β) provides a controllable trade-off between these behaviors (Eq. 1), which can be tuned per task (see Figures 4, 6, 7, 10).

5) Integration with RL fine-tuning (Section 3.2; Eq. 5)
- Joint objective with a scalar reward `r(y)` and a distillation regularizer:
  - E_x [(1−α) E_y∼pSθ [r(y)] − α E_y∼pS [D(pT || pSθ)(y|x)]].
- `α` trades off reward maximization and staying close to the teacher on student-visited prefixes.
- This combines the benefits of reinforcement optimization (e.g., factual consistency rewards in summarization) with on-policy distillation (Figure 5).

6) Practical setup
- Students are initialized from SFT checkpoints so they already generate reasonable outputs that the teacher can usefully label (Section 3.1, Remark).
- Training-time sampling temperature is 1 for the student; evaluation uses greedy, beam, or temperature sampling depending on the task (Sections 2, 4; figure captions).
- Computational overhead of student sampling is reported as ~1.8–2.2× vs. fixed-dataset training for the tested model size ratios (Appendix A.2, “Computational cost of GKD”).

## 4. Key Insights and Innovations
- On-policy distillation for auto-regressive LMs (fundamental)
  - Shift from fixed datasets to student-generated sequences labeled by the teacher at the token level (Section 3.1; Algorithm 1).
  - Significance: directly reduces train–inference mismatch, a central issue in sequence generation (Figures 1, 2, 6, 7, 9).

- Unified, generalized framework (fundamental)
  - Two orthogonal knobs: data source mixture `λ` and divergence `D` (Section 3.1; Eq. LGKD). Supervised KD, SeqKD, ImitKD, and f-distill become special cases.
  - Significance: practitioners can tailor distillation to task needs (e.g., diversity vs. faithfulness) and computational constraints.

- Divergence as a task-dependent control (insightful)
  - Systematic exploration of forward KL, reverse KL, and `JSD(β)` shows quality–diversity and task-dependent trade-offs (Figures 4, 6, 7, 10; Appendix A.12–A.14).
  - Example: For summarization under temperature sampling, mode-seeking divergences (e.g., reverse KL, JSD(0.9)) yield better ROUGE-2 but less diversity (Figure 4).

- Seamless combination with RL fine-tuning (innovative capability)
  - A single on-policy objective (Eq. 5) optimizes a sequence-level reward while distilling from a teacher (Section 3.2).
  - Demonstrated to improve factual consistency via textual entailment rewards while maintaining or improving summarization quality (Figure 5).

- Data efficiency (practical innovation)
  - On-policy GKD with only 5% of XSum training data (no ground-truth summaries) outperforms supervised KD and ImitKD trained on 100% of the data (Figure 3).

## 5. Experimental Analysis
- Evaluation methodology
  - Tasks and metrics
    - Summarization: XSum; ROUGE-2 on validation (Section 4.1; Figure 1 caption; Appendix A.3).
    - Translation: WMT14 en→de; BLEU on validation with beam search (Section 4.2; Figure 1; Figure 6; Appendix A.5).
    - Arithmetic reasoning: GSM8K with 4-shot chain-of-thought (CoT); exact-match accuracy using a calculator (Section 4.3; Figures 1, 7–9; Appendix A.4).
    - Instruction tuning: FLAN2021 (5.36M examples); evaluation on held-out MMLU and BBH via few-shot prompting (Section 4.4; Figure 10; Appendix A.6).
  - Models
    - Teacher: SFT T5-XL (~3B) for task-specific experiments; FLAN T5-XL for instruction tuning (Section 4; Appendix A.2).
    - Students: T5-Small (77M), T5-Base (250M), T5-Large (800M); FLAN variants for GSM8K and instruction tuning (Section 4).
  - Baselines
    - SFT, Supervised KD (Eq. 3), SeqKD (teacher-generated sequences), ImitKD, f-distill (Sections 3, 4).
  - Reporting
    - Greedy/beam/temp sampling as appropriate; multiple seeds for WMT and GSM8K; final checkpoints (Section 4; Appendix A.5).

- Main quantitative results
  - Cross-task summary (Figure 1)
    - On-policy GKD consistently outperforms supervised fine-tuning, supervised KD, and SeqKD across student sizes on XSum (ROUGE-2), WMT (BLEU), and GSM8K (accuracy).
    - The paper reports relative gains vs. baseline KD improvements averaged across student sizes of roughly 2.1× (XSum), 1.7× (WMT), and 1.9× (GSM8K) (Introduction, last paragraph).
  - Summarization (Sections 4.1; Figures 2–5)
    - Against ImitKD and f-distill, on-policy GKD (e.g., JSD(0.9)) yields higher ROUGE-2 under both greedy and temperature sampling (Figure 2).
    - Data efficiency: On-policy GKD with just 5% XSum data surpasses supervised KD and ImitKD trained on 100% (Figure 3).
    - Divergence–diversity trade-off: As temperature increases, mode-seeking divergences (reverse KL, JSD(0.9)) improve quality but reduce diversity measured by Self-BLEU (Figure 4).
    - RL + GKD: With a textual entailment reward (RLAIF-style), varying `α` trades off factual consistency gains vs. ROUGE-2; joint RL+GKD achieves higher ROUGE-2 than RLEF* while also being more factually consistent than the teacher (Figure 5).
  - Translation (Sections 4.2; Figure 6; Appendix A.15)
    - On-policy GKD with JSD(0.1) is best. Absolute BLEU improvements over the original student reach ~0.85 (T5-Small) and ~0.71 (T5-Base) using 100% student-generated data (Figure 6, top-left entries).
    - Mixed/on-policy data outperforms purely supervised variants across divergences (Figure 6).
    - As student size increases, performance gaps between divergences shrink (Figure 6, right panel).
  - Arithmetic reasoning (Sections 4.3; Figures 7–9; Appendix A.14)
    - Teacher (fine-tuned FLAN T5-XL) achieves 27.9% accuracy with greedy sampling; T5-Base student starts at 10.16% (Figure 7 caption).
    - Using on-policy data is crucial: accuracy gains are largest when λ=1; e.g., T5-Base sees up to +8.8 absolute points with forward KL (top-left of Figure 7).
    - Increasing the on-policy fraction above 25% consistently improves accuracy (Figure 8).
    - Across sizes, on-policy GKD clearly beats supervised KD, SeqKD, ImitKD, and f-distill (Figure 9).
  - Instruction tuning (Section 4.4; Figure 10)
    - On-policy GKD with reverse KL yields the largest gains on held-out MMLU and BBH in few-shot prompting.
    - Reported absolute improvements are roughly +1–2% on these suites (main text: “2% and 1% absolute accuracy improvement on … BBH and MMLU,” Figure 10 caption provides base accuracies).

- Ablations and robustness checks
  - Divergence vs. data fraction λ across tasks (Figures 6–8; Appendices A.12–A.14):
    - On-policy (λ=1) or mixed (λ=0.5) generally outperforms supervised (λ=0). 
    - Forward KL often strong under greedy evaluation; mode-seeking divergences help under temperature sampling (XSum, Figure 4; Appendix A.12–A.13).
  - Self-distillation on GSM8K (Appendix A.1; Figure A.11):
    - Training a FLAN T5-Large student from a FLAN T5-Large teacher shows on-policy GKD surpasses supervised KD and even the teacher’s test accuracy.
  - Computational overhead (Appendix A.2):
    - Student sampling adds ~1.8–2.2× cost relative to fixed datasets for the tested model ratios; argued to be acceptable given inference-time savings of smaller students.

- Overall assessment
  - The experimental suite convincingly supports the central claims:
    - On-policy training reduces mismatch and improves downstream metrics.
    - Divergence choice is task- and decoding-regime-dependent, and JSD offers a tunable middle ground.
    - GKD integrates smoothly with RL to improve factuality while preserving quality.

## 6. Limitations and Trade-offs
- Assumptions and preconditions
  - Requires a capable teacher that can return token-level probabilities on arbitrary prefixes (Section 3.1).
  - Starts from a reasonably good student (SFT checkpoint) so that on-policy samples are meaningful (Section 3.1, Remark).

- Computational considerations
  - On-policy sampling increases training-time cost (~1.8–2.2× vs fixed datasets; Appendix A.2), though still far cheaper than training the teacher and potentially amortized by inference savings.

- Diversity vs. faithfulness
  - Mode-seeking divergences (reverse KL, high-β JSD) can reduce generation diversity (Figure 4). Selecting `D` and decoding temperature requires task-specific tuning.

- Scope of evaluation
  - Results are shown on T5-family models and specific tasks. While tasks are diverse, behavior on much larger or different architectures is not directly tested here.
  - The method does not address cases where teacher guidance is unreliable or misaligned; it will faithfully imitate teacher biases on student prefixes.

- Optimization choice
  - Not backpropagating through sampling simplifies training but gives a biased gradient relative to full policy gradient; the paper prioritizes stability and efficiency (Section 3.1).

- Data mixture scheduling
  - The work uses fixed λ settings (0, 0.5, 1). More nuanced curricula (e.g., annealing λ or adaptive schedules) are not explored and might further improve results.

## 7. Implications and Future Directions
- Impact on the field
  - Reframing distillation as on-policy imitation for auto-regressive models provides a principled way to eliminate exposure bias during compression. This general template can subsume prior KD methods and encourage divergence-aware distillation practices.

- Practical applications
  - Deploying smaller, faster students with improved quality on summarization, translation, and reasoning.
  - Enhancing instruction-tuned models’ generalization (Figure 10), enabling broader task coverage with compact models.
  - Production pipelines that already run RLHF can integrate GKD with minimal changes (Eq. 5), potentially lowering the “alignment tax” by retaining general capabilities while optimizing preference rewards (Section 3.2; Figure 5).

- Research directions
  - Adaptive divergence and λ scheduling: Automatically learn when to be mode-covering vs. mode-seeking across tokens, tasks, or training phases (Figures 4, 6–7 suggest potential).
  - Broader modalities and tasks: Apply GKD to auto-regressive audio, video, and text-to-image generators (Conclusion references), and to decoding-speed applications (e.g., speculative decoding).
  - Robustness and safety: Combine GKD with disagreement-aware or uncertainty-aware training to avoid overfitting to teacher errors on off-distribution prefixes.
  - Theoretical analysis: Characterize convergence and generalization properties of on-policy token-level distillation under capacity mismatch and varying divergences.

Overall, GKD operationalizes a simple but powerful idea—distill exactly where the student will operate—while giving practitioners knobs (data source and divergence) to balance quality, diversity, and efficiency, and doing so in a way that integrates naturally with modern RL-based fine-tuning.
