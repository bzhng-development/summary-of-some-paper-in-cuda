# ON-POLICY DISTILLATION OF LANGUAGE MODELS: LEARNING FROM SELF-GENERATED MISTAKES

**ArXiv:** [2306.13649](https://arxiv.org/abs/2306.13649)

## 🎯 Pitch

This paper introduces Generalized Knowledge Distillation (GKD), a novel approach for distilling large language models by training student models directly on their own generated outputs, using teacher feedback to correct mistakes in real time. By addressing the critical train–inference distribution mismatch that limits prior distillation methods, GKD achieves substantial improvements across summarization, translation, and reasoning tasks, and can be seamlessly integrated with reinforcement learning fine-tuning—making it a transformative solution for crafting efficient and high-quality language models ready for real-world deployment.

---

## 1. Executive Summary
This paper introduces Generalized Knowledge Distillation (`GKD`), a simple but powerful way to distill large language models into smaller ones by training on the student model’s own generated outputs (on‑policy) while using the teacher’s token‑level probabilities as feedback. By addressing the core train–inference mismatch that plagues prior distillation methods, `GKD` delivers large, consistent gains across summarization, translation, and reasoning, and it integrates cleanly with reinforcement-learning (RL) fine-tuning.

## 2. Context and Motivation
- The specific gap:
  - Distilling auto-regressive language models typically uses a fixed set of sequences: either ground‑truth outputs (“supervised KD”) or sequences produced by the teacher (“sequence‑level KD”). During inference, however, the student conditions on its own previous tokens, not those from ground truth or the teacher. This causes a train–test distribution mismatch (also known as exposure bias in sequence generation) that cascades errors across tokens. Section 3 explains this mismatch and links it to imitation learning; Figures 1–3 empirically show the limits of existing KD methods.
- Importance:
  - Compressing large models cuts inference cost and memory footprint without sacrificing quality. Addressing the train–test mismatch is key to making distilled models reliable for real deployments (e.g., summarization, translation, reasoning).
- Prior approaches and shortcomings:
  - Supervised KD (Eq. 3) uses teacher token‑level probabilities on a fixed dataset and ignores the student’s own trajectories.
  - Sequence-level KD (Kim & Rush, 2016) trains on teacher-generated sequences but still uses a fixed set of outputs.
  - Mixed approaches like ImitKD and f‑distill partly use student data but do not fully commit to on‑policy training nor explore alternative divergences beyond the standard forward KL. See comparisons in Figures 2, 6, 9, and A.15.
- Positioning:
  - `GKD` reframes distillation for auto-regressive models as on‑policy imitation learning with an interactive expert (teacher). It unifies supervised and on‑policy distillation in one objective, allows different divergences (forward KL, reverse KL, generalized Jensen‑Shannon), and plugs directly into RL fine‑tuning (Section 3.2; Eq. 5).

## 3. Technical Approach
At a high level: train the student on the sequences it actually generates, and use the teacher to provide token‑level guidance on those exact sequences.

- Preliminaries (Section 2)
  - Auto‑regressive generation: for input `x` and token sequence `y = (y1, …, yL)`, the model predicts next‑token distributions `p(· | y<n, x)`.
  - Teacher and student: `p_T` and `p_S`, with student parameters `θ`. The token‑wise discrepancy between teacher and student for a sequence `y` is averaged over positions (Eq. 2).
  - Divergences:
    - `forward KL`: `DKL(p_T || p_S)` encourages covering the teacher’s full support (mode‑covering).
    - `reverse KL`: `DKL(p_S || p_T)` is mode‑seeking (focuses on teacher’s high‑mass tokens).
    - `JSD(β)` (Eq. 1): a bounded divergence that smoothly interpolates between forward (β→0) and reverse KL (β→1). This flexibility matters when the student cannot match the teacher’s full distribution.

- Baseline objectives (Section 3)
  - Supervised fine‑tuning: maximum likelihood on ground truth.
  - Supervised KD (Eq. 3): minimize `DKL(p_T || p_S)` at each token on a fixed dataset of (x, y).
  - Sequence‑level KD: maximize student likelihood of teacher‑generated sequences (a form of supervised FT on the teacher’s outputs).

- On‑policy distillation (Section 3.1; Eq. 4)
  - Key idea: generate outputs from the student itself and then match the teacher’s token‑level probabilities on those exact (potentially erroneous) trajectories.
  - Objective:
    - `LOD(θ) = E_x E_{y ~ p_S(·|x)} [ D(p_T || p_S)(y | x) ]` (Eq. 4),
    - where gradients do not backpropagate through the student sampling process, keeping training stable and efficient (no REINFORCE variance).
  - Why it works: the student receives targeted feedback exactly where it errs (its own partial sequences `y<n`), directly addressing train–inference mismatch.

- Generalized KD (Section 3.1; Algorithm 1)
  - Mixture of supervised and on‑policy data:
    - `LGKD(θ) = (1 − λ) E_{(x,y)∼(X,Y)} [ D(p_T || p_S)(y|x) ] + λ E_x E_{y ~ p_S(·|x)} [ D(p_T || p_S)(y|x) ]`
    - `λ ∈ [0, 1]` controls the student data fraction. `λ=0` recovers supervised KD; `λ=1` is purely on‑policy.
  - Algorithm 1 in practice:
    1) Sample a minibatch of inputs.
    2) With probability `λ`, generate outputs `y` from the student; otherwise, use outputs from the fixed dataset (ground truth or teacher‑generated).
    3) For every token position, compute the chosen divergence `D` between teacher and student distributions.
    4) Take a gradient step on `θ` with respect to this loss.
  - Choice of `D`:
    - Forward KL encourages coverage but can waste capacity on low‑probability teacher tokens (potentially increasing hallucination).
    - Reverse KL and high‑β JSD are more mode‑seeking, improving quality at the cost of diversity.
    - The paper shows the best `D` is task‑ and sampling‑temperature‑dependent (Figures 4, 6, 7, 10).

- RL fine‑tuning + on‑policy GKD (Section 3.2; Eq. 5)
  - When optimizing a sequence‑level reward `r(y)`, combine policy optimization with GKD regularization:
    - `E_x[(1−α) E_{y ~ p_S}[r(y)] − α E_{y ~ p_S} D(p_T || p_S)(y|x)]` (Eq. 5).
  - `α` trades off reward maximization and distillation strength.
  - Practical note: if you already run RLHF/RLAIF, you can add GKD with reverse KL or high‑β JSD with minimal changes (Remark in Section 3.2).

- Implementation essentials
  - Teacher: `T5‑XL` (~3B params) fine‑tuned per task; Students: `T5‑Small/Base/Large` (77M/250M/800M) or `FLAN‑T5` variants for instruction tuning and reasoning (Section 4; Appendix A.2).
  - Training lengths: typically 40K steps (XSum, GSM8K), 100K (WMT), 50K (FLAN instruction tuning), with Adafactor optimizer; details in Tables A.1–A.4.
  - Compute overhead of on‑policy sampling (Appendix A.2): roughly 1.8×–2.2× over using a fixed dataset, but serving-time benefits usually dominate total cost.

## 4. Key Insights and Innovations
- On‑policy distillation for auto‑regressive LMs
  - Novelty: train the student on its own trajectories and ask the teacher for token‑wise guidance at the exact states it will visit at inference (Eq. 4; Algorithm 1). This directly tackles exposure bias—a core failure mode of prior distillation.
  - Significance: consistently better quality across tasks and model sizes (Figures 1–3, 6–9, 10).

- Generalized objective that unifies KD variants
  - Novelty: a single formulation (mixture over data sources `λ` + choice of divergence `D`) that subsumes supervised KD, SeqKD, and prior “mixed” methods (Section 3.1).
  - Significance: flexibility to adapt to capacity limitations (choose `D`) and data availability (choose `λ`).

- Divergence choice as a task‑dependent knob
  - Novelty: systematic comparison of forward KL, reverse KL, and `JSD(β)` (Eq. 1) under on‑policy training.
  - Significance: clear quality–diversity trade‑offs (Figure 4); for some tasks and sampling temperatures, mode‑seeking divergences (reverse KL/JSD with high β) yield better quality; for others or with greedy decoding, the choice matters less (Figures A.12–A.13).

- Seamless integration with RL fine‑tuning
  - Novelty: a simple additive objective (Eq. 5) that improves factuality while preserving or enhancing task performance (Figure 5).
  - Significance: helps mitigate the “alignment tax” by letting RL pursue a reward (e.g., entailment) while GKD transfers broad teacher competence.

## 5. Experimental Analysis
- Evaluation setup (Section 4)
  - Datasets and metrics:
    - XSum summarization: ROUGE‑2 (Figure 1; details in A.3).
    - WMT14 en→de translation: BLEU with beam search (Figure 1; A.5).
    - GSM8K arithmetic reasoning: exact‑match accuracy with few‑shot chain‑of‑thought (CoT) and calculator (Figures 1, 7–9; A.4).
    - Task‑agnostic instruction tuning (FLAN2021): evaluate on held‑out MMLU (57 tasks) and BBH (23 tasks) with few‑shot accuracy (Figure 10; A.6).
  - Baselines:
    - Supervised FT, Supervised KD (Eq. 3), SeqKD, ImitKD, f‑distill; all start from the same supervised FT student checkpoint.

- Main results (Figures 1, 2, 6, 7, 9, 10; text in Section 4)
  - Overall gains:
    > Figure 1 and Section 1 report that averaged over student sizes, on‑policy GKD achieves relative improvements of about 2.1× (XSum), 1.7× (WMT), and 1.9× (GSM8K) over the improvements obtained by baseline KD approaches.
  - Summarization (XSum):
    - On‑policy `GKD(JSD(0.9))` outperforms Supervised KD, SeqKD, ImitKD, and f‑distill under both greedy and temperature sampling (Figure 2).
    - Data efficiency: with only 5% of training inputs and no ground‑truth summaries, on‑policy GKD surpasses Supervised KD and ImitKD trained on the full dataset (Figure 3).
    - Divergence vs diversity: at higher sampling temperatures, mode‑seeking divergences (reverse KL/JSD(0.9)) improve ROUGE‑2 but reduce diversity (Self‑BLEU rises); differences shrink under greedy decoding (Figure 4; A.12–A.13).
    - RL + GKD for factual consistency: using entailment reward and varying `α`, the method traces a Pareto frontier—higher `α` increases ROUGE‑2 but reduces entailment gains; the combined method achieves higher ROUGE‑2 than RLAIF‑style regularization while being more factually consistent than the teacher (Figure 5).
  - Machine translation (WMT14 en→de):
    - Teacher (T5‑XL) BLEU ≈ 28 with temperature 1.0; T5‑Small starts at 25.58, T5‑Base at 26.98 (Section 4.2).
    - On‑policy `GKD(JSD(0.1))` consistently beats supervised and mixed variants (Figure 6). Gains shrink with larger students, but on‑policy remains best. Figure A.15 shows larger improvements over ImitKD and f‑distill (53% and 162% higher BLEU improvement on average).
  - Arithmetic reasoning (GSM8K):
    - Setup: 4‑shot CoT prompting; teacher FLAN‑T5‑XL reaches 27.9% accuracy with greedy decoding (Section 4.3).
    - Results: on‑policy GKD outperforms Supervised KD, SeqKD, ImitKD, and f‑distill across all student sizes (Figure 9). Forward KL and reverse KL both work well; using only student‑generated CoTs beats mixing with fixed CoT datasets (Figure 7; A.14).
    - On‑policy fraction matters: accuracy improves monotonically once the student‑data fraction `λ` exceeds ≈25% (Figure 8).
  - Task‑agnostic instruction tuning (FLAN2021 → evaluate on MMLU, BBH):
    - On‑policy GKD with reverse KL yields the strongest improvements: 
      > Figure 10 shows absolute gains of roughly +2% on MMLU and +1% on BBH over the already instruction‑tuned student baseline (teacher: 52.4% MMLU, 41% BBH; student: 35.6% MMLU, 31.25% BBH).
    - Reverse KL likely helps the student focus on the core behavior specified by instructions (Section 4.4).
  - Self‑distillation (Appendix A.1):
    - Even when teacher and student have the same architecture (FLAN‑T5‑Large on GSM8K), on‑policy GKD improves over the teacher; on‑policy variants outperform Supervised KD (Figure A.11).

- Do the experiments support the claims?
  - Yes: Across four task families with diverse metrics, `GKD` is consistently better than standard KD baselines and prior mixed methods. Extensive ablations show the benefit of on‑policy data (`λ>0`) and clarify when different divergences help (Figures 2–4, 6–8, A.12–A.15). The RL integration shows a clear quality–factuality trade‑off controlled by `α` (Figure 5).

- Notable ablations, robustness, and caveats:
  - Divergence vs sampling temperature: quality–diversity trade‑offs (Figure 4).
  - Student data fraction `λ`: on‑policy or mixed generally beats purely supervised; performance improves with higher `λ`, especially beyond 25% (Figures 6–8).
  - Compute overhead: on‑policy sampling is ≈1.8–2.2× costlier than using a fixed dataset (Appendix A.2), though inference/serving dominates total costs in practice.

## 6. Limitations and Trade-offs
- Assumptions and prerequisites:
  - The student should be reasonably capable before distillation (typically after supervised FT), so that its on‑policy outputs are meaningful for teacher feedback (Remark in Section 3.1).
  - Access to teacher logits (token‑level probabilities) is required for every token of every training sequence; this may be unavailable for proprietary teachers.

- Sensitivity and task dependence:
  - The best divergence (`forward KL`, `reverse KL`, `JSD(β)`) depends on task and decoding temperature (Figures 4, 6, 7, 10). This introduces an extra hyperparameter to tune.
  - Mode‑seeking divergences can reduce output diversity (Figure 4), which may be undesirable in creative generation settings.

- Computational considerations:
  - On‑policy data collection increases training cost (1.8×–2.2×; Appendix A.2). The approach trades training efficiency for improved inference reliability and quality.

- Evaluation scope:
  - Summarization quality and factuality rely on automatic metrics (ROUGE‑2; entailment scores from a T5‑XXL NLI model in Figure 5). Human evaluations are not reported.
  - RL experiments use textual entailment as a proxy reward; optimizing this may not capture all aspects of factuality or faithfulness.

- Open questions:
  - No formal guidance on how to schedule `λ` over training or how to adaptively choose the divergence per task/instance.
  - Theoretical understanding of why particular divergences win under certain temperatures or capacities remains empirical.

## 7. Implications and Future Directions
- How this changes the landscape:
  - `GKD` reframes distillation for auto‑regressive LMs as on‑policy imitation, closing the gap between training and inference. This is a conceptual and practical shift that unifies old KD variants and aligns distillation practices with how the student is actually used.
  - The clean integration with RLHF/RLAIF suggests a path to reduce the “alignment tax” by combining reward optimization with distillation-based capability preservation (Section 3.2; Figure 5).

- Practical applications:
  - Deploy smaller, cheaper models for summarization, translation, and reasoning with quality closer to larger teachers.
  - Improve instruction‑tuned models in a task‑agnostic way (Figure 10), benefiting general‑purpose assistants.
  - Enhance speculative decoding by aligning draft and target models via GKD (related applications noted in Section 5).

- Follow‑up research:
  - Adaptive selection/scheduling of `λ` and divergence `D` during training based on online signals (e.g., disagreement or uncertainty).
  - Sequence‑level divergences and credit assignment that remain stable without backpropagating through sampling.
  - Extending on‑policy GKD to other auto‑regressive domains (audio, video, text‑to‑image) as suggested in the conclusion.
  - Human‑in‑the‑loop evaluation of quality/diversity/factuality trade‑offs and exploration of safety/robustness impacts.

Overall, `GKD` offers a simple training loop (Algorithm 1) that squarely addresses a long‑standing issue—exposure bias in distillation—while providing practical levers (`λ`, divergence choice, RL integration) to adapt to capacity constraints and task goals.
