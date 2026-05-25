## 3. Technical Approach

### 3.1 Reader orientation
The system is an **adaptive inference harness** that decides, for each incoming math prompt, how to spend a fixed budget of test‑time generation tokens — by choosing among search algorithms, beam widths, lookahead depths, and the mix of sequential refinement and parallel sampling — in order to maximise the probability of producing the correct answer.  
It solves the problem of **wasteful uniform allocation**: without adaptation, the same strategy (e.g., best‑of‑N) is used on every prompt, ignoring the fact that easy problems benefit from local refinement while harder problems need broad exploration. The solution shape is a **difficulty‑conditioned policy** that estimates each prompt’s hardness from the base model’s perspective and then selects the best combination of verifier‑guided search and iterative revision for that difficulty and compute budget.

### 3.2 Big‑picture architecture
The system is built around five major components:

1. **Base LLM** (PaLM 2‑S*) – the pretrained language model that produces candidate solutions; it serves as the “proposal distribution” from which all answers originate.
2. **Process Reward Model (PRM)** – a fine‑tuned verifier that scores every intermediate step of a solution with a value (estimated probability that the solution will eventually be correct). It is used to guide search and to select answers.
3. **Revision Model** – a fine‑tuned variant of the base LLM that conditions on its own previous (incorrect) answers and generates an improved answer, thereby altering the proposal distribution at test time.
4. **Search Algorithms** – procedures (best‑of‑N, beam search, lookahead search) that use the PRM’s step‑level scores to navigate the space of possible solutions.
5. **Compute‑Optimal Allocation Policy** – a meta‑strategy that, given an estimate of the prompt’s difficulty and a total generation budget, looks up which algorithm, beam width, and sequential‑to‑parallel ratio to run.

Information flow:  
**Prompt** → **Difficulty estimator** (bins the prompt into one of five quantiles) → **Policy lookup** (best strategy for that bin & budget) → **Generation** (base LLM or revision model, possibly with beam/lookahead search) → **Verification** (PRM scores each step or final answer) → **Answer selection** (best‑of‑N weighted or majority voting).

### 3.3 Roadmap for the deep dive
- **First**, the formal compute‑optimal objective (Equation 1) — what “optimal” means and why difficulty is the key conditioning variable.  
- **Second**, the difficulty estimation mechanism, because it is the linchpin that makes the whole policy adaptive and is shared by both the search and revision pipelines.  
- **Third**, the Process Reward Model (PRM) — how it is trained (Monte Carlo rollouts), how it scores solutions, and how per‑step and inter‑answer scores are aggregated — since all search methods depend on it.  
- **Fourth**, the three search algorithms (best‑of‑N weighted, beam search, lookahead search), their mechanics, their cost model, and how their behaviour differs across difficulty levels.  
- **Fifth**, the Revision Model — how it is fine‑tuned with pairwise data, how it generates sequential revisions, and how sequential and parallel sampling are combined.  
- **Sixth**, the Compute‑Optimal Allocation strategy that wraps everything together, using cross‑validation to select hyperparameters per difficulty bin.  
- **Seventh**, the FLOPs‑matched comparison framework that enables the pretraining‑vs‑inference tradeoff analysis, including the FLOP accounting equations.

### 3.4 Detailed, sentence‑based technical breakdown

This is primarily an **empirical analysis paper** whose core idea is that the optimal way to use test‑time compute is prompt‑dependent, and that a difficulty‑conditioned policy can recover large efficiency gains. We walk through how each piece is built and connected.

#### The Compute‑Optimal Objective

The paper formalises the test‑time compute allocation problem as an optimisation over hyper‑parameters that control how the budget is spent.

$$ \theta_{q, y^*(q)}^*(N) = \arg\max_\theta \left( \mathbb{E}_{y \sim \text{Target}(\theta, N, q)} \left[ \mathbb{1}_{y = y^*(q)} \right] \right) $$

where:
- `$q$` is a prompt,
- `$y^*(q)$` is the ground‑truth correct answer for that prompt,
- `$N$` is the total compute budget measured in “generations” (one generation = one complete sampled solution from the base LLM),
- `$\theta$` is a vector of hyperparameters that determines how that budget is spent — which search algorithm (best‑of‑N, beam search, lookahead), what beam width `$M$`, how many lookahead steps `$k$`, and what ratio of sequential‑to‑parallel sampling to use,
- `$\text{Target}(\theta, N, q)$` is the distribution over output tokens induced by the model under strategy `$\theta$` with budget `$N$` on prompt `$q$`.

**What it computes:** for a specific prompt and a fixed generation budget, the equation picks the strategy hyperparameters `$\theta$` that maximise the expected probability that the final selected answer exactly matches the ground truth. In plain language: “given this question and this compute allowance, among all possible ways to spend it, which one gives the highest chance of being correct?”

**Why this form:** The expectation over `$\text{Target}$` reflects that the system may be stochastic (sampling, search tree). The indicator `$\mathbb{1}_{y = y^*(q)}$` is the binary accuracy reward — only an exact match counts. The argmax captures the idea that for each prompt there exists a best strategy `$\theta^*$`, but that `$\theta^*$` is not universal; it depends on the prompt’s inherent difficulty. This formulation makes explicit that the key challenge is approximating the prompt‑dependent optimum without knowing `$y^*(q)$` at test time.

The paper never solves Equation 1 directly; instead it approximates the optimum by conditioning on a single scalar statistic — the estimated difficulty of the prompt.

#### Difficulty Estimation: Oracle and Predicted

Difficulty is defined in terms of the base LLM’s capability, not the human‑assigned difficulty labels in the MATH dataset. The authors bin prompts into five difficulty quintiles, with quintile 1 being the easiest (highest frequency of the model producing a correct answer) and quintile 5 the hardest (lowest frequency).

**Oracle difficulty.** For each of the 500 test‑set questions, the authors sample 2048 complete solutions from the few‑shot prompted PaLM 2‑S* model. They compute each question’s “pass@1” — the fraction of those 2048 solutions that exactly match the ground‑truth answer. The questions are then sorted by this pass@1 rate and divided into five equal‑sized bins (the quintiles). Because this step requires knowing which answers are correct, it is called “oracle” difficulty.

**Predicted (model‑based) difficulty.** Since ground‑truth labels are not available at deployment, the paper replaces the correctness check with **the PRM’s own confidence**. For each question, they again generate 2048 samples and score each sample’s final answer with the PRM (using the last‑step prediction, described below). They then average the PRM’s final‑answer score over those 2048 samples to obtain a single “predicted easiness” value for the question. Finally, these average scores are binned into five quintiles using the same procedure. This yields difficulty bins that require no ground‑truth labels, but still incur the cost of generating 2048 samples — a cost the paper explicitly flags as a limitation (Section 3.2).  

**Why difficulty matters for the compute‑optimal strategy:** The authors observe that the effectiveness of different test‑time strategies changes dramatically across difficulty bins (easy problems → over‑optimisation under beam search; hard problems → search helps). Therefore, **difficulty acts as a sufficient statistic** for selecting the best strategy: the compute‑optimal policy is simply a look‑up table from (difficulty bin, budget) to the best strategy hyperparameters observed on a validation fold.

**Cross‑validation protocol.** To avoid contamination between strategy selection and evaluation, the paper uses two‑fold cross‑validation within each difficulty bin on the 500‑question test set. The best strategy for a given bin and budget is selected using one fold, and its performance is measured on the other fold; then the folds are swapped and the results averaged. This ensures that the reported compute‑optimal curves do not over‑fit the test set.

#### Process Reward Model (PRM) Training and Usage

A Process Reward Model (PRM) predicts, at each step of a solution, the likelihood that the solution will eventually arrive at the correct answer. Unlike an Outcome Reward Model (ORM) that scores only the final answer, a PRM provides a dense, step‑by‑step signal that can guide search.

**Training data generation — Monte Carlo rollout supervision.** The paper follows the approach of Wang et al. (2023) to avoid expensive human labels. The procedure:

1. For each training question, sample 16 complete solutions from the few‑shot prompted base model.
2. For each step in each solution, perform **16 Monte Carlo rollouts**: continue the solution from that step onward by sampling 16 completions with the same base model.
3. Compute the **soft label** for that step: the fraction of those 16 rollouts that reach the correct final answer. This soft label (a value between 0 and 1) represents an empirical estimate of the probability that a correct answer is attainable from this partial solution.
4. The soft labels serve as target values for training.

Samples that fail to produce a parsable final answer are filtered out. The training set is constructed from 12,000 MATH training questions.

**Loss function.** The PRM is fine‑tuned as a binary classifier that outputs a scalar `$\hat{y} \in [0,1]$` per step, trained with **binary cross‑entropy** against the soft Monte Carlo labels `$y$`:

$$ \mathcal{L} = -\left(y \log(\hat{y}) + (1 - y) \log(1 - \hat{y})\right) $$

where `$y$` is the soft rollout‑derived ground‑truth value and `$\hat{y}$` is the model’s predicted probability that the solution can be completed correctly.

**What it computes:** the standard binary cross‑entropy between the model’s predicted scalar and the empirical rollout‑derived probability. The first term `$-y\log\hat{y}$` penalises under‑confident predictions when the true probability is high; the second term `$-(1-y)\log(1-\hat{y})$` penalises over‑confident predictions when the true probability is low. The result is a single non‑negative scalar per step.

**Why this form:** binary cross‑entropy is the maximum‑likelihood objective for a Bernoulli target, which is appropriate because the soft labels are themselves probabilities (empirical fractions). Mean‑squared error would be less suitable for probability calibration. Using soft labels rather than hard 0/1 labels was found to be critical — it changes the PRM’s behaviour, and earlier work that used hard labels (e.g., Lightman et al., 2023) exhibited different aggregation properties (see Appendix E).

**Training hyperparameters.** AdamW optimizer, learning rate `$3 \times 10^{-5}$`, batch size 128, dropout 0.05, Adam betas `$(0.9, 0.95)$`. Early stopping is performed using validation loss on a random 10% held‑out split of the PRM800k training questions. The base model is initialised from PaLM 2‑S*.

**Why Monte Carlo rollouts over human labels.** The authors found the public PRM800k dataset (GPT‑4‑generated solutions with human step‑level labels) “largely ineffective” for their PaLM 2 models, likely due to distribution shift. Generating supervision with the base model’s own sampling policy avoids this shift and is much cheaper than crowd‑worker annotation.

#### PRM Aggregation Strategies

When the PRM scores a complete solution consisting of multiple steps, those per‑step scores must be combined into a single final score for the solution. The paper then uses that score to select among multiple solutions.

**Step‑wise aggregation (within a single solution).** Three methods are compared:
- **min**: take the minimum PRM score across all steps.
- **prod**: take the product of per‑step scores (treating them as independent probabilities).
- **last**: use only the PRM’s prediction at the final step.

The paper finds that **“last” performs best** (Figure 13), contrary to prior work that found “min” best. The hypothesis is that the soft Monte Carlo labels produce PRM values that are not calibrated as independent step‑correctness probabilities, making the last‑step prediction a more reliable overall signal. Interestingly, using the last step effectively turns the PRM into an ORM at aggregation time, yet the PRM still **outperforms a separately trained ORM** (Figure 14), suggesting that the per‑step training acts as a powerful form of representation learning even when the intermediate predictions are not used directly for final scoring.

**Inter‑answer aggregation (across multiple solutions).** After each candidate solution receives a single score, the system must choose the best answer. The paper uses **“best‑of‑N weighted”** selection following Li et al. (2023). In standard best‑of‑N, one would simply pick the single solution with the highest score. In best‑of‑N weighted, all solutions that produce the same final answer have their scores **summed**, and the answer with the largest total sum is selected. This incorporates a form of consensus — if many solutions agree on an answer, their scores accumulate even if no single one is the top scorer — which makes the selection more robust against idiosyncratic high‑scoring but wrong solutions.

#### Search Methods Against a PRM

The paper studies three search approaches that all start from the same few‑shot prompted base LLM and use the PRM’s per‑step judgments to guide or filter candidates. The overall diagram is shown in the paper’s Figure 2.

**1. Best‑of‑N weighted (simplest baseline).**  
1. Sample `$N$` complete solutions **independently** in parallel from the base LLM.  
2. Score each solution with the PRM using the **last‑step** aggregation to obtain a single scalar per solution.  
3. Apply best‑of‑N weighted selection across the `$N$` candidates to pick the final answer.  

The generation budget equals `$N$`. No step‑by‑step pruning or exploration is performed; the computation is fully parallel.

**2. Beam search.**  
Beam search uses the PRM’s per‑step scores to prune unpromising partial solutions at each step, directing more compute toward the most promising prefixes. The algorithm works as follows:

1. **Initialise:** Sample `$N$` candidate first steps (e.g., the first line of a solution) from the base LLM.
2. **Score:** For each candidate prefix, obtain the PRM’s predicted step‑wise reward‑to‑go (the probability that the final answer will be correct). In this setting rewards are sparse (only the final answer counts), so the PRM’s score at the current step directly estimates the total future reward.
3. **Prune:** Keep only the top `$N/M$` highest‑scoring steps, where `$M$` is the **beam width** (branching factor).
4. **Expand:** From each surviving prefix, sample `$M$` proposals for the next step, resulting in a total of `$(N/M) \times M = N$` new candidate prefixes — maintaining a constant beam population of `$N$` after each round.
5. **Repeat:** Steps 2–4 are repeated until either the solution is complete (an end‑of‑solution token is generated) or a maximum of **40 rounds** of expansion is reached.
6. **Final selection:** The search terminates with `$N$` completed solutions (or as many as pop out). Best‑of‑N weighted selection is applied to these final candidates.

The generation budget is also `$N$`, making it directly comparable to best‑of‑N. Two beam‑width configurations are swept:
- **`$M = \sqrt{N}$`** — the branching factor grows with the budget.
- **`$M = 4$`** — a fixed small beam width.

**3. Lookahead search.**  
Lookahead search modifies beam search by using a more accurate per‑step value estimate. Instead of trusting the PRM’s single‑shot score at the current step, the algorithm **rolls out `$k$` additional steps forward** using the base LLM at temperature 0 (to minimise variance), and then uses the PRM’s prediction at the end of that rollout as the value estimate for the current step.

In other words, when deciding which partial solutions to keep, the algorithm says: “Imagine completing this partial solution with `$k$` more high‑confidence steps; how likely does the full path look then?” This makes the selection less myopic.

Lookahead search can be seen as a special case of Monte Carlo Tree Search (MCTS) with the stochastic exploration components removed — since the PRM is already trained and frozen, the test‑time goal is exploitation, not value learning.

**Cost model for fair comparison:** Because each lookahead rollout incurs extra computation, the paper defines the cost of lookahead search with `$k$` lookahead steps as `$N \times (k+1)$` generations — one for the beam step itself, plus `$k$` for the rollout. Thus, at the same total generation budget, lookahead search explores fewer beams than beam search or best‑of‑N. The specific configurations swept (with a maximum budget of 256 generations):
- Lookahead with `$k = 3$`, applied to both beam‑width settings (`$M = \sqrt{N}$` and `$M = 4$`).
- Lookahead with `$k = 1$`, applied to `$M = \sqrt{N}$`.

**Key difficulty‑dependent findings (referenced but detailed in the paper):** On easy questions, beam search can **degrade** performance at high budgets because it over‑optimises the PRM; on medium questions it substantially outperforms best‑of‑N; on the hardest questions no method helps.

#### Training and Using the Revision Model

While search modifies how outputs are selected, the Revision Model **modifies the proposal distribution itself** by generating improved answers conditioned on its own previous (incorrect) attempts. This moves the model away from simple i.i.d. sampling and towards a sequential refinement process.

**Why fine‑tuning is necessary.** Prompting off‑the‑shelf LLMs to self‑correct on reasoning tasks is largely ineffective (Huang et al., 2023). Therefore, the authors fine‑tune the base model to learn the skill of revising when given a context of incorrect prior attempts.

**Training data generation (offline pairwise construction).** The procedure is inspired by Qu et al. (2024) but modified to be computationally feasible:

1. For each training question, sample **64 responses in parallel** from the base LLM at **elevated temperature** (higher temperature encourages diversity).
2. Identify which responses are correct and which are incorrect (using ground‑truth labels during training).
3. For each correct answer, construct a multi‑turn training sequence: the model will see a context containing **0–4 incorrect answers** followed by the correct answer (which the model is trained to generate). The number of incorrect answers in context is sampled **uniformly from `$\{0, 1, 2, 3, 4\}$`**.
4. The *last* incorrect answer in the context is selected to be the one with the **smallest character‑level edit distance** to the correct answer. This ensures the incorrect answer is structurally similar to the correct one (it contains a mistake that can be fixed by local editing) rather than a completely unrelated approach. The remaining incorrect answers are sampled randomly.
5. Fine‑tune the base model with standard supervised learning (SFT) on these trajectories, training **only on the correct answer tokens**.

**Why edit‑distance pairing?** If incorrect answers were fully random, the model might learn to ignore the context and simply generate the correct answer from scratch — it would not learn to identify and fix mistakes. By pairing a “close” incorrect answer with the correct target, the model is taught to make **targeted edits** (e.g., fixing an arithmetic error or adjusting a sign) while preserving the overall structure. This heuristic was found sufficient to produce a usable revision model.

**Training hyperparameters:** AdamW optimizer, learning rate `$1 \times 10^{-5}$`, batch size 128, dropout 0.0, Adam betas `$(0.9, 0.95)$`. Validation loss on off‑line trajectories is not a reliable early‑stopping signal because the evaluation data becomes off‑policy after fine‑tuning; the authors select a checkpoint **slightly after the point where validation loss begins to increase**.

**Inference with the revision model (sequential sampling).** At test time, the fine‑tuned revision model is used to generate a chain of revisions:

1. It first generates an initial answer (without any previous context).
2. Then, conditioning on that first answer as context, it generates a revision.
3. Then it conditions on the first two answers, generates another revision, and so on.

Although the model was trained with at most 4 incorrect answers in context, longer chains are supported by **truncating the context to the most recent 4 answers** (keeping a sliding window).

**The correct‑to‑incorrect reversion problem.** Because the model was trained exclusively on sequences where all in‑context answers are incorrect (followed by a correct one), it has no training signal for what to do when one of its own revisions turns out to be correct. At test time, if the model accidentally produces a correct answer early in the chain, the next revision step may “fix” it into a wrong answer. The paper reports that roughly **38% of correct answers get converted back to incorrect ones** under naïve sequential sampling.

To mitigate this, the system does not simply take the last revision. Instead, it uses **within‑chain selection**: a verifier (ORM) or majority voting scans all answers in the revision chain and picks the best one. Thus the chain serves as a bank of candidates that are refined over time, but the final output is chosen by looking back across the whole chain.

**Verifier for the revision model.** The PRM trained on base model outputs does not transfer well to revision model outputs due to distribution shift (Figure 15a). Therefore, the authors train a separate **Outcome Reward Model (ORM)** specifically on the revision model’s outputs. This ORM is modified to include the previous revisions in its context, giving it access to the same information the revision model sees. Including revision history improves ORM performance slightly (Figure 15b), but even without history, sequential revisions still outperform parallel sampling — confirming that the benefit comes from the proposal distribution change, not just from the verifier seeing more context.

#### Sequential vs. Parallel Sampling and the Revision Model

The revision model can be used in two modes at test time:

- **Fully parallel**: use the revision model like the base model, generating `$N$` independent answers (each is a first attempt, no conditioning on previous ones) and selecting the best via ORM or majority. This corresponds to standard best‑of‑N but with a slightly improved proposal distribution.
- **Fully sequential**: generate a single chain of `$N$` revisions and select the best answer within that chain.
- **Hybrid (sequential‑parallel combination)**: split the budget into `$\sqrt{N}$` parallel chains, each of length `$\sqrt{N}$` (or other ratios). Within each chain, the best answer is selected via the ORM (or majority); then across chains, a second round of best‑of‑N weighted selection picks the final answer (hierarchical aggregation described in Appendix I).

The ratio of sequential to parallel compute is a hyperparameter that the compute‑optimal policy will tune per difficulty bin. The paper sweeps ratios from purely parallel to purely sequential (with intermediate powers of two). On easy questions, purely sequential revisions perform best (the model just needs local refinement). On harder questions, an intermediate ratio works better — some parallel exploration is needed to discover the right high‑level approach, followed by sequential refinement to correct local mistakes.

#### The Compute‑Optimal Allocation Strategy

Given the difficulty bins and the multi‑modal strategy space (search algorithm, beam width, lookahead steps, sequential‑to‑parallel ratio), the compute‑optimal policy is **selected empirically on a validation split** and then applied to the test set.

**For PRM search:** the strategy hyperparameters `$\theta$` include the choice among best‑of‑N weighted, beam search with `$M = \sqrt{N}$`, beam search with `$M = 4$`, lookahead search with `$k = 1$` or `$k = 3$` (combined with the two beam widths). For each difficulty bin and each budget level `$N$` (powers of two up to 256), the authors identify the single best search configuration on the validation fold.

**For revisions:** `$\theta$` specifies the ratio of sequential to parallel sampling. The procedure is the same: for each difficulty bin and budget, pick the ratio that maximises accuracy on the validation fold.

**Implementation of the policy at test time:**  
1. Estimate the prompt’s difficulty (either using pre‑computed oracle bins for analysis, or using the PRM‑predicted bins for a realistic setting).  
2. For the given difficulty bin and the target budget, look up the pre‑selected `$\theta^*$`.  
3. Execute that strategy.

Because the strategy is selected per bin, the policy is **piecewise constant** over difficulty. The two‑fold cross‑validation ensures that the same test questions are not used both to choose the strategy and to evaluate it.

**Why this works so much better than uniform:** The optimal strategy is not monotonic — in some bins, beam search is best at low budgets but harmful at high budgets; in others, purely sequential revisions beat any parallel sampling. A one‑size‑fits‑all strategy (like standard best‑of‑N) must average these conflicting requirements, leaving performance on the table. The compute‑optimal policy avoids that averaging loss by adapting to each prompt’s difficulty. The paper demonstrates that this yields up to **4× efficiency gains**: compute‑optimal scaling with 16 generations can nearly match best‑of‑N with 64 generations, and compute‑optimal revisions with 64 generations can match parallel best‑of‑N with 256 generations.

#### FLOPs‑Matched Tradeoff: Pretraining vs. Test‑Time Compute

To answer whether test‑time compute can substitute for scaling pretraining, the paper constructs a FLOPs‑matched comparison. The base model is PaLM 2‑S*, and the “larger” model has approximately **14× more parameters** (parameter count not explicitly stated, but the multiplier is 14). Both models are assumed to see the same amount of pretraining data (tokens); only parameters are scaled up, following the LLaMA paradigm rather than Chinchilla‑optimal scaling.

**FLOP accounting.** The standard approximations are used:

Pretraining FLOPs:  
$$ X = 6 N D_{\text{pretrain}} $$

Inference FLOPs (greedy decoding):  
$$ Y = 2 N D_{\text{inference}} $$

where:
- `$N$` is the number of model parameters,
- `$D_{\text{pretrain}}$` is the number of pretraining tokens,
- `$D_{\text{inference}}$` is the total number of tokens generated at inference time (summed over all queries).

**What these compute:** `$X$` approximates the total floating‑point operations for training (forward + backward passes); `$Y$` approximates inference (forward pass only, roughly one‑third the FLOPs per token). These formulas are standard from scaling‑law literature (Hoffmann et al., 2022; Sardana and Frankle, 2023).

**Why this form:** The factor 6 for training arises because each backward pass is about 2× the forward pass and there are two passes (a heuristic that has been validated empirically for transformer models). The factor 2 for inference accounts for both the matrix multiplications and attention computations; it is a simplified but standard proxy.

**Matching FLOPs when scaling parameters.** If the larger model has `$M$` times more parameters (`$M \approx 14$`), then its pretraining FLOPs become `$M X$` and its inference FLOPs (with greedy decoding) become `$M Y$`. The total FLOPs for the larger model over its lifetime is `$M(X + Y)$`.

Now, if we instead keep the smaller model and spend the extra compute on **additional test‑time computation**, the smaller model’s total inference FLOPs can be increased. Let the smaller model’s original inference FLOPs be `$Y$`; we scale it to `$Y'$` while keeping pretraining FLOPs `$X$`. We want total FLOPs to match:

$$ X + Y' = M(X + Y) $$

Solving for `$Y'$`:

$$ Y' = M(X + Y) - X = M X + M Y - X = X(M-1) + M Y $$

Now express the **inference multiplier** — how many times more inference FLOPs the smaller model gets compared to its baseline `$Y$`:

$$ \frac{Y'}{Y} = \frac{X(M-1) + M Y}{Y} = M + \frac{X}{Y}(M-1) $$

Substituting `$X = 6 N D_{\text{pretrain}}$` and `$Y = 2 N D_{\text{inference}}$`, the ratio `$X/Y$` simplifies to `$3 \cdot \frac{D_{\text{pretrain}}}{D_{\text{inference}}}$`. So the inference multiplier becomes:

$$ \text{Multiplier} = M + 3 \cdot \frac{D_{\text{pretrain}}}{D_{\text{inference}}} \cdot (M-1) $$

Thus, the amount of extra test‑time compute available to the smaller model depends critically on the **ratio** `$R = D_{\text{inference}} / D_{\text{pretrain}}$`. The paper uses three values for `$R$`:

- `$R = 0.16$` (`$R \ll 1$`, few inference tokens relative to pretraining) — the smaller model gets a large inference budget because the pretraining savings dominate.
- `$R = 0.79$` (`$R \approx 1$`).
- `$R = 22$` (`$R \gg 1$`, many inference tokens) — the budget is tighter because the larger model’s per‑token inference cost is a larger share of total FLOPs.

**Why this derivation matters:** It quantifies the exchange rate between pretraining and inference compute in a realistic deployment context. The key insight is that when `$R$` is small (e.g., in self‑improvement pipelines where inference tokens are dwarfed by pretraining tokens), the smaller model with test‑time compute can afford many more samples and thus can outperform a much larger model on easy‑to‑medium questions. When `$R$` is large (high‑throughput production), the advantage shrinks or reverses, and pretraining a larger model may be more economical.

In the FLOPs‑matched evaluation (Figure 9), the stars represent the larger model’s greedy‑decoding accuracy at the FLOPs‑equivalent point. If the compute‑optimal scaling curve for the smaller model lies above the star, test‑time compute wins; below, pretraining wins. The paper finds that on easy questions test‑time compute wins across most `$R$` values, but on hard questions pretraining remains necessary, demonstrating that the two are **not 1‑to‑1 exchangeable** — test‑time compute amplifies existing capability but cannot compensate for fundamental gaps that larger pretraining would address.

#### Summary of Key Design Choices

- **PRM training with Monte Carlo rollouts** over human labels: avoids distribution shift and annotation cost, uses the base model’s own sampling policy.
- **Last‑step PRM aggregation** over min or product: matches the soft‑label calibration and yields better performance while effectively acting as representation learning.
- **Best‑of‑N weighted** over plain best‑of‑N: incorporates answer consensus.
- **Edit‑distance‑based pairing** for revision training data: ensures that the incorrect‑to‑correct transition is a local edit, teaching the model to fix mistakes rather than restart from scratch.
- **Five‑quintile difficulty bins**: a coarse but effective discretisation that enables a simple lookup policy while still capturing the main difficulty‑dependent trends.
- **Two‑fold cross‑validation**: prevents overfitting the compute‑optimal policy to the test set.
- **Generation budget as the universal cost unit**: a fair comparison across search methods after accounting for lookahead overhead via `$N\cdot(k+1)$`.
- **Separate ORM for the revision model**: needed because the base‑model PRM suffers from distribution shift on revision‑model outputs.
- **Parameter‑only scaling for the larger model**: chosen to match the LLaMA paradigm, with Chinchilla‑optimal scaling left for future work.