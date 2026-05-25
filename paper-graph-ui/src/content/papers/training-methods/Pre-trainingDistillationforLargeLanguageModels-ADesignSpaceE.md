# Pre-training Distillation for Large Language Models: A Design Space Exploration

**ArXiv:** [2410.16215](https://arxiv.org/abs/2410.16215)

## 🎯 Pitch

This paper pioneers the study of pre-training distillation (PD) for large language models, transferring knowledge from a larger teacher model to a smaller student during the massive, expensive pre-training phase rather than only after. Through systematic experimentation across key design choices—including processing of teacher logits, loss selection, scaling behavior, and teacher signal delivery—the authors show PD is both feasible and impactful, yielding concrete best practices that improve student models' final performance and efficiency. By advancing distillation into pre-training, this work challenges the status quo and provides pathways to more cost-effective, high-quality LLMs.

---

## 1. Executive Summary
This paper introduces pre-training distillation (PD): transferring knowledge from a larger “teacher” large language model (LLM) to a smaller “student” during the pre-training phase rather than after pre-training. Through a systematic design-space study across logits processing, loss design, scaling trends, and online/offline teacher signals, the paper shows PD is feasible, identifies configurations that substantially improve over standard pre-training, and distills actionable best practices.

## 2. Context and Motivation
- Problem addressed:
  - Knowledge distillation (KD) is widely used for LLMs but almost always during post-training (instruction tuning). This paper asks: can we distill during pre-training (i.e., while learning from raw text), and how do we do it effectively? (§1)
- Why it matters:
  - Pre-training consumes the vast majority of compute and data. If KD can shape learning earlier, it could accelerate optimization, serve as soft supervision (label smoothing), and improve final quality and sample efficiency (§1, §2).
- Prior approaches and gaps:
  - Post-training KD (e.g., Alpaca, Vicuna) uses teacher-generated instruction-response pairs and works well, but it does not guide the massive, earlier pre-training stage (§1).
  - Some works distill either small pre-ChatGPT models or report PD without methodological detail (§4). There is no systematic exploration of PD’s key design choices for modern billion-parameter LLMs.
- Positioning:
  - The paper frames PD as a standardizable objective and explores four critical dimensions that control its success: how to process teacher logits, which losses to use and how to combine them with the usual language modeling loss, how gains scale with model/data size, and whether to use offline versus online teacher logits (§2, §§3.2–3.5).

Terminology (selective):
- `Knowledge distillation (KD)`: training a smaller model to match the outputs (often probabilities/logits) of a larger model (the teacher).
- `Pre-training distillation (PD)`: applying KD during pre-training on raw text, not just during post-training (§1, §2).
- `Logits`: the unnormalized scores a model produces before applying softmax to obtain token probabilities.
- `Top-p` truncation: keep the smallest set of tokens with cumulative probability mass ≥ p.
- `Top-k` truncation: keep only the k highest-probability tokens.
- `Temperature (τ)`: divides logits before softmax; higher τ makes distributions flatter, lower τ makes them sharper.

## 3. Technical Approach
Step-by-step pipeline (formalized in §2 with Equations (1)–(4), implemented in §3):
1. Setup:
   - Have a teacher LLM `θ_T` and a student LLM `θ_S`. Training data are standard pre-training corpora (plain text). Tokens form sequences `x = {x_t}_{t=1}^T`.

2. Objective (Eq. (1)):
   - Minimize a mixture of the usual language modeling loss and a KD loss:
     - `L = (1 - α) * L_lm + α * L_kd`
     - `α ∈ [0,1]` controls how much to trust the teacher.

3. The two loss terms:
   - `L_lm` (Eq. (2)): standard next-token negative log-likelihood on true tokens.
   - `L_kd` (Eq. (3)): matches the student’s next-token distribution to a processed version of the teacher’s distribution:
     - `L_kd = (1/T) * Σ_t L(P_{θ_S}(x_t | x_<t), F(P_{θ_T}(x_t | x_<t)))`
     - `L(·,·)` can be NLL, Kullback–Leibler divergence (KLD), or MSE (§3.3).
     - `F(·)` processes teacher logits via truncation and temperature-normalized softmax (Eq. (4)).

4. Logit processing `F` (§3.2):
   - To avoid storing full-vocab logits (58.6 PB for 100B tokens with ~150k vocab; §3.1), apply a two-stage truncation:
     - First `top-p` (e.g., p = 0.95), then `top-k` (e.g., k = 100). After truncation, re-normalize with temperature `τ` via softmax (Eq. (4)).
   - Rationale: captures the “mass” of the teacher’s distribution while cutting storage 4,000× to ~15 TB for 100B tokens (§3.1). Top-p handles sharp distributions; top-k caps long tails (§3.2).

5. Training mechanics (§3.1, App. A.1):
   - Preliminary config: teacher `GLM-4-9B`; student 1.9B; 100B pre-training tokens; context length 4096; Adam; batch size 2048; cosine LR schedule.
   - Because small students are weak on complex benchmarks, they perform supervised fine-tuning (SFT) after pre-training to stabilize evaluation: a mixture totaling 20B tokens (10B instruction-tuning + 10B extra pre-training text; App. A.1). During instruction data, loss is computed only on responses.

6. Loss scheduling (§3.3):
   - Beyond static `α`, the paper tests dynamic schedules, notably a “Warmup–Stable–Decay” (`WSD`) schedule for both `α` and the learning rate. Intuition: give KD more weight when the LR is high (stable plateau), then reduce later. This synchronizes the optimization “pressure” of KD with high learning rates.

7. Online vs offline teacher logits (§3.5):
   - Offline: run the (pretrained) teacher over the corpus, store truncated logits, then train the student.
   - Online: store teacher logits “on the fly” while pre-training the teacher from scratch, then use those stored logits to pre-train students.

Why these choices?
- Truncation: massive storage savings without losing the “signal” (teacher probability mass). Figures 2–3 and Table 8 show performance is fairly robust to the exact p and k.
- Temperature: balances signal sharpness (too sharp can overfit to the teacher’s top choice, too flat loses guidance). Tables 2 and 9 quantify the sweet spot.
- Loss type and scheduling: empirical testing (Table 5) shows KLD/NLL are viable; MSE underperforms. Scheduling `α` with WSD and aligning it with a WSD LR produces the strongest gains, suggesting timing matters.

Illustrative example:
- Consider the next-token distribution for a prefix. The teacher might spread probability across “cat” (0.55), “dog” (0.25), “car” (0.05), and many small-mass tokens. Top-p=0.95 keeps {cat, dog, car, …} until the cumulative mass exceeds 0.95; top-k=50 caps count. After renormalization with τ (e.g., τ=0.5), the student learns to assign probability mass similarly—not just copying the argmax (“cat”), but also learning the teacher’s uncertainty structure.

## 4. Key Insights and Innovations
- Broad, systematic design-space exploration for PD (§§3.2–3.5):
  - Novelty: prior LLM PD reports usually omit details or focus on post-training KD. Here, PD is decomposed into four practical levers (logit processing, loss, scaling, online/offline), each studied with targeted ablations.

- Storage-efficient teacher signal via two-stage `top-p-k` truncation (§3.1, §3.2):
  - What’s new: a pragmatic combination of top-p then top-k to control both mass and tail length.
  - Why it matters: reduces 58.6 PB → ~15 TB for 100B tokens (4,000× smaller) while keeping PD effective (Table 1; Figures 2–3; Table 8).

- Loss scheduling synergy (WSD-α + WSD-LR) delivers the strongest gains (§3.3, Table 5):
  - What’s new: jointly scheduling the KD mixture weight `α` and the learning rate with warmup–stable–decay yields the best result (“WSD-α+WSD-LR” achieves the highest average score 40.7, +8.0% over baseline; Table 5).
  - Why it matters: it shows PD’s benefit depends not just on “what loss” but “when and at what LR” that loss is emphasized.

- Scaling-law observations that invert common expectations (§3.4, Figure 4, Table 11):
  - Larger students benefit more from PD; using a much larger teacher does not always help (capacity gap issues).
  - Significance: practical guidance—align teacher/student sizes to avoid overwhelming small students (e.g., 3.8B student gains +2.9 points with a 9B teacher vs +0.9 with a 32B teacher; Table 11).

- Online logits are usable but require care (§3.5, Table 6):
  - Using logits from an unconverged teacher hurts; later-stage online logits and lower α help (Table 6).
  - Practical implication: when pre-training a family of models, log teacher logits later in training to reuse for smaller students at minimal extra inference cost.

Fundamental vs incremental:
- Fundamental: establishing PD as a viable, configurable pre-training paradigm for LLMs with clear, generalizable levers and constraints.
- Incremental but important: specific choices—top-p-k truncation settings, τ ranges, KLD/NLL preference, WSD scheduling—translate into repeatable recipes.

## 5. Experimental Analysis
Evaluation setup (§3.1, App. A.1):
- Data and training:
  - Preliminary PD: teacher `GLM-4-9B`, student 1.9B, 100B tokens; top-0.95 then top-100 truncation; τ=1.0; KD uses NLL of student against normalized teacher distribution; Adam; batch=2048; seq len=4096; cosine LR (§3.1).
  - After pre-training, SFT on a 20B mixture (10B instruction + 10B extra pre-training text; App. A.1). For instruction data, loss computed only on responses.
- Datasets and shots:
  - English: HellaSwag, WinoGrande, PIQA, MMLU; Chinese: KBQA, C3, C-Eval; Math: GSM8k (§3.1; App. A.1). Zero-shot for most; 5-shot C3/C-Eval; 6-shot MMLU; 8-shot GSM8k; decoding temperature=0 (App. A.1).
- Baselines:
  - `LLM-LM`: same student trained with standard LM loss only (α=0).
  - Compare multiple PD variants changing truncation, τ, loss type, α schedules, sizes, and online/offline logits.

Main quantitative results (selected):
- Feasibility of PD (Table 1):
  - Average accuracy: `LLM-LM` 37.7 vs `LLM-KD` 38.3
  - Relative improvement: +1.6%
  - Quote:
    > Table 1: “∆ ↑ 1.6%” (average) with PD over LM-only for the 1.9B student; GSM8k improves 24.6% relatively (8.6 → 10.8).

- Logits processing (Figures 2–3; Tables 2, 8–9):
  - Top-p-k:
    - Robust across p and k; smaller p or k can shrink storage with similar performance (Figures 2–3; Table 8).
    - Example: `top-0.95-50` reaches the best average in Table 8 (39.6) and markedly lifts MMLU (33.2).
  - Temperature:
    - Static τ: τ ≤ 2.0 is best; τ ≥ 5.0 degrades (Table 2; Table 9). 
    - Adaptive τ: AdaKDH is best among adaptives (avg 38.8; Table 3), but not meaningfully better than a well-chosen static τ=0.5/2.0 in aggregate (§3.2).

- Loss selection and scheduling (Tables 4–5):
  - Loss type (α=1):
    - `KLD` averages 38.7; `NLL` 38.3; `MSE` drops to 34.9 (Table 5).
    - Quote:
      > Table 5: “LLM-MSE … 34.9 ↓ 7.6%” vs baseline, confirming MSE underperforms for PD.
  - Mixing with LM loss:
    - Best static α ≈ 0.9 (Table 4, average improvement peak at α=0.9).
    - Dynamic schedules:
      - Linear decrease outperforms linear increase (Table 5: “Linear Dec … 39.2 ↑ 4.1%”).
      - WSD-α + WSD-LR is the best overall (Table 5: “WSD-α+WSD-LR … 40.7 ↑ 8.0%”).

- Scaling laws (Figure 4; Table 11):
  - Student size:
    - Gains grow with student size; small students (330M) may not benefit (or regress) from PD.
  - Teacher size:
    - 9B teacher can outperform 32B teacher for mid-size students (e.g., 3.8B student: +2.9 vs +0.9 average; Table 11).
  - Corpus size and training dynamics (Figure 5; Table 12):
    - PD improves across the entire 500B-token training curve for 1.9B and 3.8B students.
    - Quote:
      > Figure 5: both “1.9B-KD” and “3.8B-KD” curves stay above their LM-only counterparts; the gains increase early then partially converge by the end.
    - Final checkpoint averages (Table 12):
      - 1.9B: LM 44.2 vs KD 45.4
      - 3.8B: LM 50.2 vs KD 53.7

- Online vs offline logits (Table 6; §3.5):
  - Early-stage online logits hurt (teacher not converged):
    - `LLM-Online-100B-L` average 29.8 (−20.9% relative to LM baseline).
  - Late-stage online logits help modestly with careful weighting:
    - `LLM-Online-100B*` (α=0.1, `top-0.95-50`) reaches 37.9 (slightly above 37.7 baseline), but still below offline PD (Table 6).

- Best recipe (“PD*”) and end-to-end gains (App. A.6; Table 13; Figure 1):
  - Configuration: `top-0.95-50`, τ=2.0, KLD loss, WSD-α (max α=0.9) + WSD-LR; offline logits.
  - Results (Table 13):
    - 1.9B: 41.2 (vs 37.7 baseline) 
    - 3.8B: 45.7 (vs 42.0)
    - 6.8B: 49.8 (vs 44.9)
  - Quote:
    > Figure 1: PD* improves all three student sizes beyond both LM-only and “vanilla PD,” demonstrating the value of the explored configuration.

Do the experiments support the claims?
- Yes, for feasibility and best practices:
  - Multiple ablations isolate the role of truncation p/k, τ (static vs adaptive), loss type, α scheduling, LR scheduling, and scaling. Improvements persist across datasets and larger token budgets (Tables 1, 5, 8–13; Figures 1–5).
- Robustness:
  - Stronger students benefit more (Table 11, Figure 4), and improvements persist over 500B tokens (Figure 5). Failure modes are acknowledged for online logits from early teacher checkpoints (Table 6).

## 6. Limitations and Trade-offs
- Assumption of teacher access and compatibility:
  - Requires a competent teacher and the ability to run it over large corpora (offline) or store its logits during pre-training (online). Full-vocab logits are prohibitively large—58.6 PB for 100B tokens—necessitating truncation (§3.1).
- Interaction effects not fully explored:
  - The paper studies each factor with controlled variables; it does not exhaustively search interactions among all factors due to compute costs (Limitations).
- Online PD sensitivity:
  - Online logits from an unconverged teacher can harm performance unless weighted down (`α=0.1`) and taken from later training (§3.5, Table 6).
- Small students may not benefit:
  - 330M/670M students show small or negative gains, especially with overly large teachers (Table 11, Figure 4). This reflects a capacity gap issue.
- Compute and environmental cost:
  - Extensive experiments incur significant compute and associated emissions (Limitations; Strubell et al., 2019 cited).
- Evaluation relies on SFT to stabilize measurement:
  - Because small students perform near chance on challenging tasks, the study adds SFT (20B tokens) prior to evaluation (App. A.1). This is practical, but it means pure pre-training-only effects on these tasks are not directly reported.

## 7. Implications and Future Directions
- How it changes the landscape:
  - Establishes PD as a viable and tunable pre-training strategy for LLMs, not merely a post-training trick. It provides a practical blueprint (top-p-k truncation, τ≤2, KLD/NLL, WSD scheduling) and evidence that PD can raise both training efficiency and final quality (Tables 5, 12–13; Figures 1, 5).
- Practical applications:
  - Organizations training LLM families can:
    - Distill larger models into smaller ones during pre-training to reduce compute for downstream models.
    - Log teacher logits later in training (online) to amortize teacher inference for multiple students (§3.5).
    - Use `PD*`-like recipes to get reliable gains with manageable storage.
- Follow-up research:
  - Systematic interaction search:
    - Joint optimization of truncation, τ, loss type, α schedule, and LR schedule (stated as future work in Limitations).
  - Closing the capacity gap:
    - Methods to bridge teacher–student mismatch (e.g., progressive teachers, teacher assistants; see Mirzadeh et al., 2020) within PD.
  - Weak-to-strong setups:
    - Train very large students from smaller teachers to test weak-to-strong generalization (suggested in §3.4).
  - Trillion-token regimes:
    - Validate PD at multi-trillion-token scales typical of frontier models (§3.4 notes compute constraints prevented this).
  - Better online PD:
    - Curriculum-like designs for when and how to capture teacher logits during its training; adaptive α over time tailored to teacher convergence.

In short, the study turns the vague idea of “do KD during pre-training” into a concrete, evidence-backed methodology. With sensible truncation, appropriate temperature, the right losses, and carefully coordinated schedules, PD can deliver consistent gains—especially for mid-to-large students—while remaining practical in storage and compute.
