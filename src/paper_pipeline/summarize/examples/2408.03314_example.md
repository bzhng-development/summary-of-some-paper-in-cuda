## 1. Executive Summary

This paper systematically studies how to optimally allocate test-time computation for LLMs, analyzing two primary mechanisms—searching against process-based reward model (PRM) verifiers and iteratively revising the model's own responses—on the MATH benchmark using PaLM 2-S* models. The core contribution is the concept of a **compute-optimal test-time scaling strategy** that adaptively selects how to spend an inference budget (e.g., beam search vs. best-of-N, sequential revisions vs. parallel sampling) based on the estimated difficulty of each prompt, yielding **more than 4× better efficiency** over a standard best-of-N baseline. In a FLOPs-matched comparison, the authors demonstrate that a smaller model augmented with compute-optimal test-time strategies can outperform a **~14× larger** pretrained model on easy-to-medium difficulty problems, establishing that—at least for prompts within a base model's rough capability range—scaling inference compute can be more effective than scaling pretraining compute.

## 2. Context and Motivation

### The Core Problem: We Don't Know How to Spend Inference Compute Wisely

The fundamental question this paper tackles is deceptively simple: **if you give an LLM extra computation at inference time, what is the best way to use it?** This matters because, unlike training—where scaling laws are relatively well-understood thanks to work like Chinchilla (Hoffmann et al., 2022)—the scaling behavior of test-time computation is poorly characterized. Prior to this work, there was no systematic understanding of *which* test-time strategy works best *when*, or how test-time compute scales compare to simply training a bigger model.

This gap is significant for several practical reasons the authors highlight (Section 1):

- **On-device deployment**: If test-time compute can substitute for model size, smaller models could replace datacenter-scale LLMs for certain tasks, running on edge devices with additional inference-time processing.
- **Self-improvement pipelines**: An LLM that can reliably improve its own outputs using extra computation opens the door to automated self-improvement loops that reduce dependence on human supervision.
- **Resource allocation decisions**: Organizations deciding how to split their compute budget between pretraining and inference need principled guidance—this paper provides some of the first empirical evidence for that tradeoff.

### Conflicting Prior Evidence

The paper is motivated by a genuine contradiction in the literature. On one side, several works show that LLMs *can* use test-time compute productively—self-critique and debate approaches (Bai et al., 2022; Du et al., 2023; Madaan et al., 2023; Saunders et al., 2022), verifier-guided sampling (Cobbe et al., 2021), and tree-of-thought style search (Yao et al., 2023). On the other side, other studies paint a much more pessimistic picture: Huang et al. (2023) showed that "large language models cannot self-correct reasoning yet," Stechly et al. (2023) found GPT-4 fails to recognize its own reasoning errors through iterative prompting, and Valmeekam et al. (2023) demonstrated that self-critiquing plans largely doesn't work.

These conflicting findings are not necessarily contradictory—they likely reflect different methods being applied to different difficulty levels under different conditions—but the field lacked a framework for reconciling them. This paper's central insight is that **the effectiveness of any given test-time strategy is highly dependent on prompt difficulty**, which explains why different papers (testing on different distributions of problems) reach opposite conclusions.

### Where Existing Approaches Fall Short

The paper identifies specific limitations in prior work along two axes:

**Best-of-N sampling is the dominant but crude baseline.** The most studied approach to test-time compute scaling is best-of-N: generate N complete solutions, score them with a verifier, and pick the best one (Cobbe et al., 2021). This is simple but treats every token of compute identically regardless of the problem. There's no adaptation to the nature of the prompt—easy problems get the same treatment as hard ones.

**Self-correction via prompting doesn't work for reasoning.** Off-the-shelf LLMs prompted to "check their work" or "revise their answer" show minimal improvement on math reasoning tasks (Huang et al., 2023; Section 6). The paper explicitly acknowledges this: "Simply prompting existing LLMs to correct their own mistakes tends to be largely ineffective for obtaining performance improvements on reasoning problems." This means that to get revisions to work, you need purpose-built fine-tuned models, which the paper develops following the recipe of Qu et al. (2024).

**Process reward models (PRMs) exist but their search-time behavior is unexplored.** Lightman et al. (2023) and Wang et al. (2023) introduced PRMs that score individual solution steps rather than just final answers. However, prior work had not systematically studied *how* to search against these verifiers at test time—which search algorithm to use, how the choice depends on compute budget, or when search over-optimizes the verifier signal.

**No unified analysis framework.** Perhaps most critically, prior work studied these mechanisms (verifiers, revisions, search algorithms) in isolation. There was no framework for comparing them on equal footing, understanding their complementary strengths, or combining them adaptively.

### How This Paper Positions Itself

The paper frames all test-time compute methods through a unifying lens described in Section 2: any approach modifies the LLM's output distribution through either **(1) changes to the proposal distribution** (what the model generates—e.g., by conditioning on previous attempts via revisions) or **(2) changes to how outputs are selected/verified** (scoring and filtering generated candidates—e.g., via PRM search). This is explicitly analogized to MCMC sampling, where a simple proposal distribution is combined with a score function to sample from a more complex target distribution.

Within this framework, the paper's position is not to propose a single new method, but rather to provide the first systematic scaling analysis of representative methods from each axis—revisions for the proposal distribution, PRM-guided search for the verifier—and then show that **adaptive, difficulty-aware allocation** (what they call "compute-optimal" scaling) is the key missing ingredient. The paper draws a direct parallel to compute-optimal pretraining scaling laws (Hoffmann et al., 2022) but applied at inference time, filling a gap that the authors argue is equally important for the future of LLM deployment.

The paper also explicitly connects to the training-inference tradeoff literature (Jones, 2021; Villalobos and Atkinson, 2023; Sardana and Frankle, 2023), but notes that prior FLOPs-matched comparisons in the language modeling domain largely assumed access to ground-truth answers. This paper's FLOPs-matched analysis (Section 7) operates in the realistic setting where the correct answer is unknown, making the comparison more practically relevant.

## 3. Technical Approach

### 3.1 Reader Orientation

The paper builds a system for deciding *how* to spend a fixed budget of inference-time computation on a given math problem so that a language model produces the correct answer as often as possible. The core idea is that no single test-time strategy dominates across all problems; instead, the paper develops a **compute-optimal scaling policy** that estimates each prompt's difficulty and then selects the best combination of search algorithm, revision depth, and parallel sampling accordingly.

### 3.2 Big-Picture Architecture (Diagram in Words)

The system has five major components:

1. **Base LLM (PaLM 2-S\*)** — the pretrained language model that generates candidate solutions. It serves as the "proposal distribution" from which all answers originate.
2. **Process Reward Model (PRM)** — a learned verifier that scores every intermediate step of a solution, producing a value estimate of whether the solution is on track. Used to guide search and select answers.
3. **Revision Model** — a fine-tuned variant of the base LLM that takes its own previous (incorrect) answers as context and produces improved answers sequentially.
4. **Search Algorithms** — procedures (best-of-N, beam search, lookahead search) that use the PRM's step-level scores to navigate the space of possible solutions at test time.
5. **Compute-Optimal Allocation Policy** — a meta-strategy that, given an estimate of the prompt's difficulty and a compute budget, selects which search algorithm and which sequential/parallel sampling ratio to deploy.

Information flows as follows: a prompt enters the system → the difficulty estimator bins it into one of five difficulty levels → the allocation policy selects hyperparameters (search method, beam width, revision chain length, number of parallel chains) → the base LLM or revision model generates candidate solutions under those hyperparameters → the PRM scores the candidates → an aggregation procedure selects the final answer.

### 3.3 Roadmap for the Deep Dive

- **First**, the formal compute-optimal objective (Equation 1), which defines what "optimal" means and why difficulty is the key variable.
- **Second**, the difficulty estimation mechanism, since it is the linchpin that enables adaptive allocation and is shared across both the search and revision pipelines.
- **Third**, the PRM verifier — how it is trained, how it scores solutions, and how scores are aggregated — since all search methods depend on it.
- **Fourth**, the three search algorithms (best-of-N, beam search, lookahead search), their mechanics, their cost model, and their difficulty-dependent behavior.
- **Fifth**, the revision model — how it is trained, how it generates sequential revisions, and how sequential and parallel sampling are combined.
- **Sixth**, the FLOPs-matched comparison framework that enables the pretraining-vs-inference tradeoff analysis.

### 3.4 Detailed, Sentence-Based Technical Breakdown

This is primarily an **empirical analysis paper** whose core idea is that the optimal way to use test-time compute is prompt-dependent, and that a difficulty-conditioned allocation policy can recover large efficiency gains.

---

#### The Compute-Optimal Objective

The paper formalizes the test-time compute allocation problem as an optimization over strategy hyperparameters (Section 3.1). Let $q$ be a prompt, $y^*(q)$ be the ground-truth correct answer, $N$ be the total compute budget (measured in number of generations), and $\theta$ be a vector of hyperparameters that determines how that budget is spent (e.g., which search algorithm, what beam width, what ratio of sequential to parallel sampling). Define $\text{Target}(\theta, N, q)$ as the distribution over output tokens induced by the model under strategy $\theta$ with budget $N$ on prompt $q$. The compute-optimal strategy is:

$$\theta_{q, y^*(q)}^*(N) = \arg\max_\theta \left( \mathbb{E}_{y \sim \text{Target}(\theta, N, q)} \left[ \mathbb{1}_{y = y^*(q)} \right] \right)$$

In plain language: for a given question and compute budget, pick the inference-time strategy that maximizes the probability of producing the correct answer. The key insight is that $\theta^*$ is not a single universal setting — it varies with $q$. A strategy that works best on easy problems (e.g., pure sequential revision) may be suboptimal on hard problems (where broad parallel search is needed).

Because solving this optimization exactly for every prompt is intractable, the paper approximates it by conditioning on a single sufficient statistic: **estimated question difficulty**.

---

#### Difficulty Estimation

The difficulty of a prompt is defined relative to the base LLM's capabilities, not the MATH dataset's hand-labeled difficulty levels (Section 3.2). The paper explicitly states that the model-specific difficulty bins are more predictive of test-time compute efficacy than the dataset's built-in labels.

**Oracle difficulty.** For each question in the test set, the authors sample 2048 complete solutions from the base model and compute the pass@1 rate — the fraction of those 2048 solutions that are correct. They then bin questions into five quintiles based on this pass@1 rate, with quintile 1 being the easiest (highest pass@1) and quintile 5 the hardest (lowest pass@1). This requires knowing which answers are correct, hence "oracle."

**Predicted (model-based) difficulty.** Since oracle difficulty requires ground-truth labels unavailable at deployment, the paper approximates it by replacing the ground-truth correctness check with the PRM's final-answer score. Specifically, they average the PRM's predicted final-answer correctness across the same 2048 samples per question, then bin into five quintiles using the same procedure. This removes the need for ground-truth labels but still requires the computational cost of generating 2048 samples and scoring them. The authors acknowledge this cost (Section 3.2) and frame it as an exploration-exploitation tradeoff — compute spent assessing difficulty versus compute spent solving the problem — flagging it as a key avenue for future work.

**Cross-validation protocol.** To avoid the circularity of selecting the best strategy and evaluating it on the same data, the paper uses two-fold cross-validation within each difficulty bin on the test set. The best-performing strategy is selected on one fold and evaluated on the other, and vice versa, with results averaged.

Once a question is assigned to a difficulty bin, the compute-optimal policy is simply a lookup: for each bin and each budget level $N$, the authors have pre-computed (on the validation fold) which strategy $\theta$ performs best. At test time, the system estimates the question's difficulty, looks up the best strategy for that difficulty-budget pair, and runs it.

---

#### The Process Reward Model (PRM)

The PRM is the central verifier component. Unlike an Outcome Reward Model (ORM) that assigns a single correctness score to a complete solution, a PRM assigns a score to **each intermediate step** (Section 5.1). These per-step scores represent the model's estimate of the probability that a correct final answer can be reached from the current partial solution — essentially a value function or reward-to-go estimate for the base model's sampling policy.

**Training procedure.** The authors do not use human labels as in Lightman et al. (2023). They found the PRM800k dataset (which contains GPT-4 generated solutions with human step-level labels) to be "largely ineffective" for their PaLM 2 models, likely due to distribution shift between GPT-4 and PaLM 2 outputs. Instead, they follow Wang et al. (2023)'s approach of **Monte Carlo rollout supervision**:

1. For each question in the training set, sample 16 solutions from the few-shot prompted base model.
2. For each step in each solution, sample 16 Monte Carlo rollouts (completions from that step onward using the same base model).
3. Compute the fraction of rollouts that reach the correct final answer — this fraction is the soft label for that step.
4. Fine-tune the base model as a binary classifier that predicts a value between 0 and 1 at each step, trained with binary cross-entropy loss against these soft Monte Carlo labels:

$$\mathcal{L} = -\left(y \log(\hat{y}) + (1 - y) \log(1 - \hat{y})\right)$$

where $y$ is the soft rollout-derived ground-truth value and $\hat{y}$ is the model's prediction.

Training hyperparameters (Appendix D): AdamW optimizer, learning rate $3 \times 10^{-5}$, batch size 128, dropout 0.05, Adam betas $(0.9, 0.95)$. Early stopping is based on validation loss on a random 10% held-out split of the PRM800k training questions. Samples that fail to produce a parsable final answer are filtered out.

**Step-wise score aggregation.** Given a complete solution with per-step PRM scores, the system needs a single score for the whole solution. The paper compares three aggregation methods (Appendix E): taking the minimum score across steps ("min"), taking the product of step-level correctness probabilities ("prod"), and using only the PRM's prediction at the final step ("last"). Contrary to prior work (Lightman et al., 2023; Wang et al., 2023) which found "min" to be best, this paper finds **"last" performs best** (Figure 13). The authors hypothesize that the discrepancy arises because their PRM is trained with soft Monte Carlo labels rather than binary correctness labels, which changes how the per-step scores distribute. An interesting consequence: using the last-step prediction effectively makes the PRM behave like an ORM at aggregation time, yet the PRM still outperforms a separately trained ORM (Appendix F, Figure 14), suggesting that the step-level PRM training acts as a form of beneficial **representation learning** even when the intermediate predictions aren't directly used at aggregation time.

**Inter-answer aggregation (best-of-N weighted).** When selecting among $N$ complete candidate solutions, the paper does not simply pick the one with the highest score. Instead, following Li et al. (2023), it uses **best-of-N weighted selection**: all solutions that arrive at the same final answer have their PRM scores summed (marginalized), and the final answer with the greatest total sum is selected. This is more robust than picking a single highest-scoring solution because it incorporates a form of consensus — if many solutions agree on an answer, their scores accumulate even if no single one is the top scorer.

---

#### Search Algorithms Against the PRM

The paper studies three search methods, all illustrated in Figure 2 (Section 5.2):

**Best-of-N weighted.** Sample $N$ complete solutions independently from the few-shot prompted base LLM. Score each with the PRM (using last-step aggregation). Apply best-of-N weighted selection to pick the final answer. The generation budget is simply $N$.

**Beam search.** This is a step-by-step search that prunes unpromising partial solutions at each step. The procedure works as follows:

1. Sample $N$ candidate first steps from the base LLM.
2. Score each first step with the PRM's step-level prediction.
3. Keep only the top $N/M$ highest-scoring steps (where $M$ is the beam width — the branching factor).
4. From each surviving step, sample $M$ candidate next steps, producing $N/M \times M = N$ candidate prefixes again.
5. Repeat steps 2–4 until the solution is complete or a maximum of 40 rounds of expansion are reached.
6. Apply best-of-N weighted selection to the $N$ completed solutions.

The paper sweeps two beam width settings: $M = \sqrt{N}$ (growing with budget) and $M = 4$ (fixed). The generation budget is $N$, same as best-of-N, making the comparison fair.

**Lookahead search.** This modifies beam search by improving the accuracy of step-level scoring. Instead of using the PRM's raw prediction at the current step to decide which beams to keep, lookahead search **simulates $k$ additional steps forward** (at temperature 0 to minimize variance) and uses the PRM's prediction at the end of this lookahead rollout as the score for the current step. This gives the PRM more context to assess whether a partial solution is on a good track.

The paper notes (Section 5.2) that this is a special case of Monte Carlo Tree Search (MCTS) with the stochastic exploration elements removed, since the PRM is already frozen and the goal is exploitation rather than learning. The cost model accounts for the extra computation: a lookahead search with $k$ steps costs $N \times (k + 1)$ generations. This means that for a fair comparison at a given generation budget, lookahead search explores fewer beams than beam search or best-of-N.

Specific configurations swept (Section 5.3, with a maximum budget of 256 generations):
- Beam search with $M = \sqrt{N}$
- Beam search with $M = 4$
- Lookahead search with $k = 3$ applied to both beam width settings
- Lookahead search with $k = 1$ applied to $M = \sqrt{N}$

**Key finding on search algorithms.** The results in Figure 3 (left) reveal a non-obvious pattern: beam search significantly outperforms best-of-N at low generation budgets but its advantage diminishes or reverses at high budgets. Lookahead search generally underperforms all methods at the same budget because its extra cost reduces the effective number of beams explored. The degradation at high budgets is attributed to **over-optimization of the PRM** — search finds solutions that score highly under the PRM but are actually incorrect. Specific failure modes include low-information repetitive steps at the end of solutions (visible in Figure 29) and overly short 1–2 step solutions.

**Difficulty-dependent behavior of search (Figure 3, right).** When results are broken out by difficulty bin, a clear pattern emerges:
- On easy questions (bins 1–2), beam search *degrades* performance with increasing budget — a hallmark of verifier exploitation, since the PRM makes mostly correct assessments on easy problems and aggressive optimization amplifies any residual errors.
- On medium questions (bins 3–4), beam search *consistently outperforms* best-of-N — the PRM's guidance genuinely helps navigate toward correct solutions that the model wouldn't find by random sampling alone.
- On the hardest questions (bin 5), no method makes meaningful progress — the base model simply lacks the capability to produce correct solutions regardless of how the budget is allocated.

This difficulty-dependence is what motivates the compute-optimal allocation: use best-of-N on easy problems and beam search on medium-hard problems, adapting per-question.

**Compute-optimal search results (Figure 4).** By selecting the best search strategy per difficulty bin, compute-optimal scaling nearly outperforms best-of-N using up to $4\times$ less test-time compute (e.g., 16 generations vs. 64). Both oracle and predicted difficulty bins produce similar gains, with the curves largely overlapping (Figure 4), which is encouraging because it means the system works without access to ground-truth labels.

---

#### The Revision Model (Modifying the Proposal Distribution)

Rather than relying solely on the verifier to select among independently generated solutions, the paper also studies modifying the **proposal distribution itself** so that the model generates better candidates in the first place. The mechanism is **iterative revision**: the model conditions on its own previous (incorrect) answers and produces an improved answer (Section 6).

**Why fine-tuning is necessary.** The paper explicitly notes that prompting off-the-shelf LLMs to self-correct on reasoning tasks is largely ineffective (citing Huang et al., 2023). To make revisions work, the authors fine-tune the base model to learn the revision skill.

**Training data generation (Section 6.1).** The procedure builds on Qu et al. (2024) with modifications:

1. For each training question, sample 64 responses in parallel from the base LLM at elevated temperature.
2. Identify which responses are correct and which are incorrect.
3. Construct multi-turn training sequences: a sequence of 0–4 incorrect answers followed by a correct answer. The number of incorrect answers is sampled uniformly from $\{0, 1, 2, 3, 4\}$.
4. The last incorrect answer in the sequence is selected to be the one with the smallest **character-level edit distance** to the correct answer. This ensures the incorrect answer is "close" to the correct one — similar in structure but containing a mistake — so the model learns to make targeted edits rather than ignoring context and starting from scratch. Remaining incorrect answers are sampled randomly.
5. Fine-tune the base model with SFT (supervised fine-tuning) on these trajectories, training only on the correct answer tokens. Hyperparameters (Appendix H): AdamW optimizer, learning rate $1 \times 10^{-5}$, batch size 128, dropout 0.0, Adam betas $(0.9, 0.95)$.

A subtle training detail: the authors note that standard validation loss is not a good signal for early stopping because, after fine-tuning, the validation trajectories become off-policy (they were generated by the base model, not the fine-tuned revision model). They select the checkpoint slightly after the point where validation loss begins increasing.

**A crucial difference from Qu et al. (2024):** The original approach used on-policy multi-turn rollouts — the model generates a revision, gets feedback, generates another revision, etc. This was computationally infeasible for the authors, so they approximated the multi-turn structure by pairing independently sampled correct and incorrect solutions post-hoc, using edit distance as a proxy for trajectory coherence.

**Inference with the revision model (Figure 5).** At test time, the revision model generates a chain of revisions: it produces an initial answer, then conditions on that answer to produce a revision, then conditions on the revision to produce another revision, and so on. Although the model was only trained with up to 4 previous answers in context, longer chains are supported by truncating context to the most recent 4 answers. Figure 6 (left) shows that pass@1 at each step gradually improves throughout the chain, even beyond the 4 steps the model was trained for — evidence that the model has learned a generalizable revision skill.

**The correct-to-incorrect reversion problem.** A significant practical issue: since the model was trained only on sequences where all in-context answers are incorrect (followed by a correct target), at test time the model may encounter correct answers in its context (produced during earlier revisions) and incorrectly "revise" them into wrong answers. The paper reports that approximately **38% of correct answers get converted back to incorrect ones** using a naive approach. To mitigate this, the system uses a selection mechanism (majority voting or verifier-based selection) across the entire chain of revisions, picking the best answer from any point in the chain rather than always taking the last revision.

**Sequential vs. parallel sampling (Figure 5, right panel and Figure 6).** The paper frames the choice of how to spend a budget of $N$ generations as a tradeoff between two extremes:

- **Fully parallel**: Generate $N$ independent solutions and select the best one (standard best-of-N).
- **Fully sequential**: Generate one chain of $N$ revisions and select the best answer in the chain.
- **Hybrid**: Generate $\sqrt{N}$ parallel chains, each of length $\sqrt{N}$, and use the verifier to first select the best answer within each chain, then select the best answer across chains (hierarchical aggregation, described in Appendix I).

Figure 6 (right) shows that, aggregated across all questions, sequential revision marginally outperforms parallel sampling under both verifier-based and majority-based answer selection.

**Difficulty-dependent optimal ratio (Figure 7).** When broken out by difficulty:
- **Easy questions** perform best with purely sequential revisions — the model's initial attempts are roughly correct and just need refinement.
- **Hard questions** perform best with a balanced ratio of sequential and parallel — they need the diversity of parallel sampling to explore different high-level approaches, but also benefit from refinement within each chain.

This mirrors the finding from search: easy problems benefit from exploitation (local refinement), hard problems benefit from exploration (global search).

**Compute-optimal revision results (Figure 8).** Selecting the optimal sequential-to-parallel ratio per difficulty bin yields compute-optimal scaling that outperforms the parallel best-of-N baseline by up to $4\times$ in compute efficiency (e.g., matching best-of-256 performance with only 64 generations). Importantly, at higher generation budgets, parallel sampling appears to plateau while compute-optimal scaling continues to improve.

**Verifier for revisions.** The PRM trained on base model outputs does not transfer well to the revision model's outputs due to distribution shift (Figure 15a). The authors train a separate ORM specifically on the revision model's outputs. This ORM is modified to include previous revisions in context, giving the verifier access to the same information the revision model sees. Including revision history improves verifier performance slightly (Figure 15b), but even without it, sequential revisions still outperform parallel sampling — showing the benefit is not purely an artifact of the verifier seeing more context.

---

#### FLOPs-Matched Comparison: Pretraining vs. Test-Time Compute

Section 7 asks: given a fixed total FLOPs budget, is it better to train a larger model or to keep the smaller model and spend the extra FLOPs on inference-time computation?

**FLOP accounting.** The paper uses standard approximations from the scaling laws literature:

$$X = 6ND_{\text{pretrain}} \quad \text{(pretraining FLOPs)}$$
$$Y = 2ND_{\text{inference}} \quad \text{(inference FLOPs)}$$

where $N$ is the number of model parameters, $D_{\text{pretrain}}$ is pretraining tokens, and $D_{\text{inference}}$ is total inference tokens generated. Scaling model parameters by a factor of $M$ multiplies both $X$ and $Y$ by $M$ (the larger model costs more per token at both training and inference).

To match the total FLOPs of the $M\times$-larger model using the smaller model with additional test-time compute, the smaller model's inference compute must be multiplied by:

$$M + 3 \cdot \frac{D_{\text{pretrain}}}{D_{\text{inference}}} \cdot (M - 1)$$

The critical quantity is the ratio $R = D_{\text{inference}} / D_{\text{pretrain}}$. When $R \ll 1$ (few inference tokens relative to pretraining), the smaller model gets a large inference budget to work with because the pretraining savings dominate. When $R \gg 1$ (many inference tokens), the budget is tighter because the larger model's per-token inference cost is a bigger fraction of total compute.

**Experimental setup.** The comparison is between PaLM 2-S\* with compute-optimal test-time scaling and a model with approximately $14\times$ more parameters (greedy decoding, no extra test-time compute). Three values of $R$ are tested: $0.16$ ($R \ll 1$), $0.79$ ($R \approx 1$), and $22$ ($R \gg 1$).

**Results (Figure 9 and the bar charts in Figure 1).** The findings are nuanced and difficulty-dependent:

- **Easy questions (bins 1–2):** Test-time compute with the smaller model outperforms the $14\times$ larger model across nearly all values of $R$. The improvement from test-time compute is large (e.g., +27.8% relative improvement on easy questions at $R \ll 1$ for revisions).
- **Medium questions (bin 3):** Test-time compute is competitive or better when $R \ll 1$ or $R \approx 1$, but pretraining becomes preferable when $R \gg 1$.
- **Hard questions (bins 4–5):** Pretraining is almost always more effective. Test-time compute provides minimal gains on problems that are fundamentally outside the base model's capability range. For example, at $R \gg 1$ with PRM search, hard questions show a -52.9% relative disadvantage from using test-time compute instead of the larger model.

The key takeaway is that test-time and pretraining compute are **not 1-to-1 exchangeable**: test-time compute is powerful when problems are within the base model's reach (it already produces correct solutions at some non-trivial rate), but it cannot compensate for fundamental capability gaps that larger pretraining would address.

**Design choice: scaling parameters vs. data.** The paper fixes training data and scales only model parameters when increasing pretraining compute, matching the approach of the LLaMA model series (Touvron et al., 2023). They acknowledge that compute-optimal pretraining would scale both data and parameters equally (Hoffmann et al., 2022), and leave that comparison to future work. This is an important caveat: the $14\times$ larger model may not be compute-optimally trained, which could make the comparison somewhat favorable to test-time compute.

---

#### Summary of Design Choices and Their Justifications

- **Monte Carlo rollout PRM training** over human labels: avoids distribution shift between GPT-4-labeled data and PaLM 2 outputs, and eliminates the need for expensive human annotation.
- **Last-step aggregation** over min/product: empirically superior with soft MC labels; effectively leverages the PRM's training as representation learning.
- **Best-of-N weighted** over standard best-of-N: incorporates consensus across solutions sharing the same final answer, providing robustness.
- **Edit-distance-based pairing** for revision training data: ensures incorrect in-context answers are structurally related to the correct target, teaching the model to make targeted corrections rather than restart from scratch.
- **Five difficulty quintiles** rather than continuous difficulty: provides a discrete, interpretable partitioning that enables simple lookup-based policy selection while still capturing the key difficulty-dependent trends.
- **Two-fold cross-validation** for strategy selection: prevents overfitting the compute-optimal policy to the test set.

## 4. Key Insights and Innovations

### Innovation 1: Difficulty-Conditioned Compute-Optimal Test-Time Scaling

The paper's most fundamental contribution is not any single method but rather the **meta-strategy** of adaptively allocating test-time compute based on prompt difficulty. Prior work treated test-time compute as a uniform knob: turn it up (more samples, more search) and performance improves. This paper demonstrates that the relationship between compute and performance is **qualitatively different** depending on problem difficulty, and that ignoring this heterogeneity leaves enormous efficiency on the table.

What makes this genuinely novel — rather than an obvious observation — is that the difficulty-dependent behavior is often *counterintuitive*. Beam search, the strongest optimizer, actually **hurts** performance on easy problems at high budgets due to verifier over-optimization (Figure 3, right), while it **helps** substantially on medium-difficulty problems. Similarly, sequential revisions dominate on easy problems but a balanced sequential-parallel ratio is optimal on hard ones (Figure 7, right). These are not monotonic relationships where "more powerful = better." The compute-optimal policy exploits these non-monotonicities to achieve $4\times$ better efficiency than best-of-N (Figures 4 and 8), which is a significant practical gain.

This contribution is best understood as an **inference-time analog of the Chinchilla scaling laws** for pretraining. Just as Hoffmann et al. (2022) showed that the optimal allocation of pretraining compute between model size and data quantity varies with total budget, this paper shows that the optimal allocation of test-time compute between search strategies varies with problem difficulty. The conceptual parallel is direct, but the underlying mechanism is entirely different — pretraining scaling laws optimize over continuous variables (parameters, tokens), while this paper optimizes over a discrete, combinatorial space of strategy hyperparameters conditioned on a difficulty estimate.

A subtle but important point: the predicted (non-oracle) difficulty bins perform nearly as well as oracle bins (the curves largely overlap in Figures 4 and 8). This is what makes the contribution *practical* rather than merely analytical. If the gains required ground-truth labels to estimate difficulty, the approach would be circular. The fact that the PRM's own score distribution serves as a sufficient proxy means the system is deployable without access to answers.

### Innovation 2: The Proposal Distribution and Verifier as Complementary, Independent Scaling Axes

The unifying framework in Section 2 — decomposing all test-time compute methods into modifications to the **proposal distribution** (what the model generates) versus the **verifier** (how outputs are selected) — is not itself technically novel. It echoes the proposer-scorer decomposition familiar from MCMC and reinforcement learning. What *is* novel is the paper's empirical demonstration that these two axes have **complementary, difficulty-dependent strengths** and that combining them yields gains neither achieves alone.

Concretely: revisions (proposal modification) are most effective on easy problems where the model's initial output is roughly correct and just needs refinement — a local search in answer space. Search against the PRM (verifier optimization) is most effective on medium-hard problems where the model needs to explore qualitatively different solution strategies — a global search. Prior work studied these mechanisms in isolation, often reaching pessimistic conclusions (e.g., "LLMs cannot self-correct reasoning" from Huang et al., 2023). This paper's framework reconciles those findings: self-correction *does* work, but only on the right difficulty tier. Search *does* help, but only with the right algorithm at the right budget. The conflicting prior results were an artifact of testing different methods on different (implicitly difficulty-biased) problem distributions.

This insight is more than taxonomic. It implies that future systems should not choose *between* revisions and search but should deploy both, switching between them per-prompt. The paper doesn't fully realize this vision (Section 8 acknowledges that PRM tree-search was not combined with revisions), but the framework provides the intellectual scaffolding for doing so.

### Innovation 3: Empirical Evidence That Test-Time Compute Can Substitute for Pretraining — With Sharp Boundaries

The FLOPs-matched comparison in Section 7 is, to the authors' knowledge, the first to demonstrate in a realistic setting (no ground-truth access at inference) that a smaller model with additional test-time compute can **outperform a ~14× larger model** on problems within its capability range. This is significant not as a method but as an **empirical finding with direct implications for how compute budgets should be allocated** in production systems.

What distinguishes this from prior work on training-inference tradeoffs (Jones, 2021; Villalobos and Atkinson, 2023) is the specificity of the finding. The paper doesn't claim a universal substitution — it precisely characterizes *where* the substitution works (easy-to-medium problems, low $R$ regimes) and *where* it fails (hard problems, high $R$ regimes). The failure case is equally informative: on the hardest problems (bin 5), test-time compute provides essentially zero benefit regardless of budget, meaning that some capabilities can **only** be acquired through pretraining, not recovered at inference time. This establishes a clear boundary condition: test-time compute amplifies existing capability but does not create it from nothing.

The dependence on $R = D_{\text{inference}} / D_{\text{pretrain}}$ adds practical nuance that prior analyses missed. For self-improvement pipelines where $R \ll 1$, the case for test-time compute is strong. For high-throughput production deployments where $R \gg 1$, the case weakens because the per-query inference cost of the larger model dominates the budget anyway. This is an incremental but practically important refinement of the training-inference tradeoff picture.

### Innovation 4: Verifier Over-Optimization as a First-Class Phenomenon in Test-Time Scaling

While reward hacking / over-optimization is well-documented in the RLHF literature, this paper provides some of the first clear evidence that **the same phenomenon governs test-time search scaling** and is the primary bottleneck preventing unbounded improvements from additional compute. The evidence is concrete: beam search degrades easy-problem performance at high budgets (Figure 3, right); lookahead search — the most powerful optimizer — paradoxically performs *worst* overall (Figure 3, left); and qualitative examples in Appendix M show search producing degenerate outputs (repetitive low-information steps, overly short solutions) that score highly under the PRM.

This finding is significant because it shifts the narrative around test-time compute from "more is better" to "more is better only up to the verifier's reliability frontier." It explains why prior work found negative results for sophisticated search methods: those studies likely pushed past the over-optimization threshold. It also implies that **improving verifier robustness is the key bottleneck** for further scaling test-time compute, not improving search algorithms. The paper's compute-optimal policy can be understood partly as a way to stay *below* the over-optimization threshold per difficulty level — using weaker optimization (best-of-N) where the verifier is reliable (easy problems) and stronger optimization (beam search) only where the verifier signal has more room to provide genuine guidance (medium problems).

## 5. Experimental Analysis

### Evaluation Methodology

**Dataset.** All experiments use the MATH benchmark (Hendrycks et al., 2021), consisting of high-school competition-level math problems. The authors use the specific split from Lightman et al. (2022): 12,000 training questions and 500 test questions. The choice of MATH is deliberate (Section 4): test-time compute is expected to help most when the model already possesses the necessary knowledge and the challenge is drawing complex inferences — mathematical reasoning fits this profile because it requires multi-step logical deduction rather than novel factual recall.

**Base model.** All experiments use PaLM 2-S* (Codey) (Anil et al., 2023). The authors argue this model is "representative of the capabilities of many contemporary LLMs" and sits in a useful regime: non-trivial performance on MATH (roughly 10–19% pass@1 depending on the prompt and sampling configuration) but far from saturation, leaving room for test-time compute to make a difference. For the FLOPs-matched comparison, a second model with approximately $14\times$ more parameters is used as the pretraining-scaled baseline.

**Metrics.** The primary metric throughout is **MATH test accuracy (%)** — the fraction of the 500 test questions for which the selected final answer matches the ground truth. Answers are graded using the grading function released by Lightman et al. (2022) (Appendix G). When analyzing difficulty-dependent behavior, the paper reports accuracy within each of the five difficulty quintiles separately.

**Generation budget** is the universal unit of test-time compute. One "generation" equals one complete sampled answer from the base LLM. For beam search and best-of-N, the budget equals the number of beams or samples $N$. For lookahead search with $k$ lookahead steps, the cost is $N \times (k+1)$ to account for the additional rollout computation (Section 5.3). Budgets are swept across powers of 2, typically from $2^0$ to $2^9$ (1 to 512 generations).

**Baselines.** The paper uses several baselines:
- **Majority voting**: select the most common final answer among $N$ sampled solutions (no learned verifier).
- **ORM best-of-N weighted**: score $N$ solutions with an outcome reward model and apply best-of-N weighted selection.
- **PRM best-of-N weighted**: score $N$ solutions with the process reward model and apply best-of-N weighted selection.
- **Parallel sampling** (for revisions): generate $N$ independent solutions from the revision model and select the best via verifier or majority.

**Cross-validation for compute-optimal strategies.** To avoid contaminating strategy selection with test-set performance, the authors use two-fold cross-validation within each difficulty bin on the 500-question test set. The best strategy is selected on one fold and evaluated on the other, with results averaged (Section 3.2).

---

### Main Quantitative Results

#### Search Against PRM Verifiers (Section 5)

**Aggregate search algorithm comparison (Figure 3, left).** Across all 500 test questions with a maximum budget of 256 generations:

- At low budgets (2–8 generations), beam search with $M = 4$ significantly outperforms best-of-N weighted. For example, at 4 generations beam search ($M = 4$) achieves roughly 27% accuracy versus roughly 16% for best-of-N weighted — a substantial gap.
- At high budgets (64–256), beam search performance flattens and falls slightly below best-of-N weighted. Best-of-N weighted reaches approximately 38% at 512 generations; beam search ($M = 4$) plateaus around 34%.
- Lookahead search (both $k = 1$ and $k = 3$) generally underperforms at the same generation budget due to its higher per-step cost. The 3-step lookahead variants converge to similar performance as other methods at very high budgets but never surpass them.
- Majority voting trails all verifier-based methods substantially, reaching only about 29% at 512 generations.

**Difficulty-bin analysis for search (Figure 3, right).** The per-difficulty breakdown (beam search $M = 4$ vs. best-of-N weighted, shown at four budget levels: 4, 16, 64, 256 generations) reveals the core pattern:

- **Bin 1 (easiest):** Beam search accuracy *decreases* from roughly 78% to 77% as the budget goes from 4 to 256, while best-of-N weighted increases from 68% to 88%. This is the clearest evidence of PRM over-optimization — beam search finds solutions that exploit the verifier signal.
- **Bin 2:** Beam search improves modestly (roughly 14% → 32%) but best-of-N weighted improves faster (roughly 14% → 60%), maintaining a clear advantage at high budgets.
- **Bin 3:** Beam search consistently outperforms best-of-N weighted across all budgets, reaching roughly 34% vs. 23% at 256 generations.
- **Bin 4:** Beam search shows the strongest relative advantage, reaching roughly 17% vs. 10% for best-of-N at 256 generations.
- **Bin 5 (hardest):** Both methods hover near 1–3% regardless of budget. No method makes meaningful progress.

**Compute-optimal search (Figure 4).** By selecting the best search strategy per difficulty bin at each budget level:

- At 16 generations, compute-optimal (oracle bins) achieves approximately 27% accuracy, roughly matching PRM best-of-N weighted at 64 generations — a $4\times$ compute reduction.
- At 256 generations, compute-optimal oracle reaches approximately 39.5%, surpassing PRM best-of-N weighted at the same budget (roughly 37%).
- Compute-optimal with predicted difficulty bins tracks the oracle version closely, particularly at lower budgets. The two curves "largely overlap" per the authors (Figure 4), with the predicted version reaching approximately 37% at 256 generations.
- Both compute-optimal variants consistently outperform ORM best-of-N weighted (which peaks around 34% at 512 generations) and majority voting (around 29%).

**PRM vs. ORM (Figure 14, Appendix F).** At 2048 samples, PRM best-of-N weighted achieves approximately 40% accuracy versus roughly 35% for ORM best-of-N weighted and roughly 30% for majority voting. The gap between PRM and ORM widens with the number of samples, confirming the PRM's superior scaling properties.

---

#### Revision Model Results (Section 6)

**Revision model pass@1 trajectory (Figure 6, left).** Starting from approximately 18.2% pass@1 at step 1, the revision model's per-step accuracy improves to roughly 24–25% by steps 15–20, and remains in the 23–25% range out to 64 steps. The model generalizes beyond its 4-step training horizon.

**Sequential vs. parallel (Figure 6, right).** At 64 generations:
- Sequential + best-of-N weighted: approximately 41.5%
- Parallel + best-of-N weighted: approximately 39%
- Sequential + majority: approximately 38%
- Parallel + majority: approximately 35%

Sequential outperforms parallel under both selection mechanisms, with the verifier-based gap (roughly 2.5 percentage points) being slightly narrower than the majority-based gap (roughly 3 points).

**Sequential-to-parallel ratio sweep (Figure 7, left).** For a fixed generation budget, varying the ratio reveals:
- At 256 generations, the optimal ratio is around $2^1$ to $2^3$ (2:1 to 8:1 sequential-to-parallel), achieving approximately 43–44% accuracy.
- Fully parallel (leftmost point) yields approximately 40%.
- Fully sequential (rightmost point) yields approximately 42%.
- At lower budgets (8–32 generations), fully sequential is optimal — the curves are monotonically increasing with the sequential-to-parallel ratio.

**Difficulty-dependent ratio (Figure 7, right).** At a fixed budget of 128 generations:
- **Bin 1:** Performance is essentially flat across all ratios, around 90–92%. Easy questions are insensitive to the allocation strategy.
- **Bin 2:** Slight advantage for higher sequential ratios, approximately 63% at fully sequential vs. 58% at fully parallel.
- **Bin 3:** A clear optimal ratio emerges at moderate sequential-to-parallel values (around $2^1$ to $2^3$), reaching approximately 42% vs. 35% at the extremes.
- **Bin 4:** Similar pattern, with the peak at a moderate ratio achieving roughly 18% vs. 14% at fully parallel.
- **Bin 5:** All ratios produce roughly 2–3% accuracy. No allocation strategy helps.

**Compute-optimal revisions (Figure 8).** Selecting the optimal sequential-to-parallel ratio per difficulty bin:
- At 64 generations, compute-optimal oracle achieves approximately 40%, matching parallel best-of-N weighted at 256 generations — a $4\times$ improvement.
- At 256 generations, compute-optimal oracle reaches approximately 44%, compared to roughly 41% for best-of-N weighted and 37% for parallel-only.
- Compute-optimal predicted bins perform slightly below oracle bins at high budgets (approximately 41% at 256 generations) but still substantially outperform the parallel baseline.
- Notably, the parallel baseline appears to **plateau** around 36–37% at high budgets, while compute-optimal scaling continues to improve, suggesting that the gains from adaptive allocation compound at higher budgets.

---

#### FLOPs-Matched Comparison: Test-Time vs. Pretraining Compute (Section 7)

**Revisions (Figure 9, left; Figure 1, top-right bar chart).** Comparing PaLM 2-S* with compute-optimal revisions against the $\sim 14\times$ larger model:

| Difficulty | $R \ll 1$ (0.16) | $R \approx 1$ (0.79) | $R \gg 1$ (22) |
|---|---|---|---|
| Easy (bin 1) | +11.8% | +3.5% | −11.9% |
| Medium (bin 2–3) | +27.8% | +16.7% | +5.4% |
| Hard (bins 4–5) | +21.6% | −(implied negative) | −37.2% |

(Numbers from the bar chart in Figure 1, top-right. Note: the "easy/medium/hard" groupings in the bar chart differ slightly from the five difficulty bins, aggregating bins for readability.)

At $R \ll 1$, test-time compute outperforms the larger model across **all** difficulty levels. At $R \gg 1$, it only remains preferable on easy questions, with hard questions showing a −37.2% relative disadvantage.

**PRM search (Figure 9, right; Figure 1, bottom-right bar chart).** The pattern is starker:

| Difficulty | $R \ll 1$ (0.16) | $R \approx 1$ (0.79) | $R \gg 1$ (22) |
|---|---|---|---|
| Easy | +19.1% | +2.2% | +2.0% |
| Medium | 0.0% | −35.3% | −30.8% |
| Hard | −3.6% | −35.3% | −52.9% |

PRM search shows weaker benefits than revisions for the FLOPs-matched comparison, with substantial disadvantages on medium and hard questions even at moderate $R$ values. On easy questions, test-time compute remains preferable across all $R$ regimes, though the margin narrows significantly.

**Figure 9 detail.** The line plots show accuracy per difficulty bin as test-time compute scales. The $14\times$ larger model's greedy performance (stars) is placed at three x-axis positions corresponding to the three $R$ values. Where the compute-optimal scaling line is above the star, test-time compute wins. On bin 1 (purple, topmost line), the scaling line is above all three stars for revisions. On bin 5 (blue, bottommost line), the line is below all three stars and essentially flat near 0–5%, confirming that no amount of test-time compute helps on the hardest problems.

---

### Ablation Studies and Robustness Checks

**PRM aggregation strategy (Appendix E, Figure 13).** Comparing "min," "prod," and "last" step-wise aggregation:
- "Last" achieves roughly 37% at 256 samples.
- "Min" achieves roughly 35%.
- "Prod" achieves roughly 27%.
- ORM achieves roughly 34%.

The "last" aggregation's superiority is notable because it effectively reduces the PRM to ORM-like behavior at aggregation time, yet the PRM still outperforms a separately trained ORM. The authors interpret this as evidence that step-level PRM training provides beneficial representation learning.

**PRM vs. ORM (Appendix F, Figure 14).** The PRM consistently outperforms the ORM, with the gap widening at higher sample counts: at 2048 samples, PRM best-of-N weighted reaches approximately 40% vs. ORM's 35%.

**Revision model verifier choice (Appendix J, Figure 15a).** The base-LM PRM underperforms the revision-specific ORM when scoring revision model outputs, with sequential + base-LM PRM achieving roughly 40% at 64 generations vs. sequential + revision ORM at roughly 42%. This confirms distribution shift as a practical concern.

**Revision history in verifier context (Appendix J, Figure 15b).** Including previous revisions in the ORM's context provides a small improvement over the no-history ablation (approximately 1–2 percentage points at 64 generations), but both variants outperform the parallel baseline, confirming that the sequential sampling benefit is not solely attributable to the verifier seeing more context.

**Oracle vs. predicted difficulty bins (Figures 4, 8, and Appendix C, Figures 11–12).** Both oracle and predicted bins yield qualitatively similar trends across difficulty levels. Predicted bins show slightly lower performance at high budgets in the revision setting (roughly 41% vs. 44% at 256 generations in Figure 8) but essentially identical performance in the search setting (Figure 4). This is the critical robustness check: the compute-optimal strategy works without ground-truth labels.

**Majority voting for revisions (Appendix B, Figure 10).** The sequential-to-parallel ratio trends observed with verifier-based selection are replicated with majority voting: easy questions are insensitive to ratio, hard questions show an optimal intermediate ratio, and fully sequential marginally outperforms fully parallel in aggregate.

**ReST$^{EM}$ revision model (Appendix K, Figure 16).** An attempt to further optimize the revision model using ReST$^{EM}$ (Singh et al., 2024) backfires: additional sequential revisions **substantially hurt** performance with this model. At 256 generations, fully sequential performance drops to approximately 33.5% compared to roughly 38.5% at the optimal ratio. The authors hypothesize that the on-policy data collection in ReST$^{EM}$ exacerbates spurious correlations in revision data, causing the model to fail to learn the revision task properly. This is a notable negative result that highlights the sensitivity of revision training to the data generation procedure.

---

### Assessment: Do the Experiments Support the Claims?

**Claim 1: Compute-optimal scaling improves efficiency by more than $4\times$ over best-of-N.** Supported for both search (Figure 4: 16 generations matching 64) and revisions (Figure 8: 64 generations matching 256). The $4\times$ figure specifically refers to achieving equivalent accuracy with $4\times$ fewer generations, and the evidence is consistent across oracle and predicted difficulty settings. However, at the highest budgets (256–512), the gains narrow somewhat with predicted difficulty bins, suggesting the $4\times$ figure is most reliable in the lower-to-moderate compute regime.

**Claim 2: Test-time compute with a smaller model can outperform a $14\times$ larger model.** Supported with sharp conditions. The claim holds convincingly for easy-to-medium problems at $R \ll 1$ and weakens progressively as difficulty increases or $R$ grows. The paper is transparent about these boundaries, which strengthens credibility. One caveat: the $14\times$ larger model uses greedy decoding with no test-time augmentation of its own, making it a somewhat weak baseline — a fairer comparison might give the larger model some test-time compute budget as well.

**Claim 3: Efficacy depends critically on prompt difficulty.** Very strongly supported. The difficulty-bin analyses (Figures 3 right, 7 right) show qualitatively different — and sometimes opposite — effects of the same strategy at different difficulty levels. This is the most robust finding in the paper, replicated across search methods, revision strategies, and selection mechanisms.

**Potential weaknesses in the experimental design:**

- **Single benchmark, single model family.** All results are on MATH with PaLM 2-S*. The authors acknowledge this but argue the model is "representative" — a claim that cannot be verified without replication on other models and datasets.
- **Difficulty estimation cost is unaccounted for.** Generating 2048 samples per question to estimate difficulty is extremely expensive — comparable to or greater than the test-time compute budget being studied. The authors flag this explicitly (Section 3.2) but do not include this cost in any budget calculation, which makes the reported efficiency gains ($4\times$) somewhat overstated in a deployment context.
- **The $14\times$ larger model may not be compute-optimally trained.** The paper scales parameters only (not data), following the LLaMA paradigm rather than Chinchilla-optimal training. A compute-optimally trained larger model (scaling both parameters and data) would be a stronger baseline.
- **Test set of 500 questions.** The difficulty bins split 500 questions into quintiles of ~100 each. With cross-validation splitting each bin roughly in half, strategy selection is based on ~50 questions per bin — a relatively small sample that could introduce variance in the computed-optimal policy.
- **No combination of PRM search with revisions.** The paper studies search and revisions independently but never combines PRM tree-search with the revision model as the proposal distribution — a natural next step that the authors acknowledge in Section 8. The current results therefore represent a lower bound on what combined approaches might achieve.

## 6. Limitations and Trade-offs

### Assumption: Difficulty Can Be Estimated Cheaply Enough to Be Practical

The entire compute-optimal framework rests on the ability to estimate prompt difficulty *before* deciding how to allocate the inference budget. The paper's method for doing so — generating 2048 samples per question and averaging either ground-truth correctness (oracle) or PRM final-answer scores (predicted) — is **extraordinarily expensive**. At 2048 samples per question, the difficulty estimation step alone consumes more compute than the largest test-time budgets studied (256–512 generations). The authors acknowledge this explicitly in Section 3.2:

> "estimating difficulty in this way still incurs additional computation cost during inference... our experiments do not account for this cost largely for simplicity"

This is a significant gap. The reported $4\times$ efficiency gains over best-of-N are computed *after* difficulty is known, without amortizing the cost of learning it. In a realistic deployment, the total cost would be difficulty estimation + strategy execution, and the former could dominate the latter. The paper suggests future work on training models to predict difficulty directly from the question text, but no such model is developed or evaluated. Until this gap is closed, the $4\times$ figure should be understood as an **upper bound on achievable efficiency** rather than a realized deployment gain.

### Single Benchmark, Single Model Family

All experiments use the MATH benchmark (500 test questions) with PaLM 2-S\* as the base model. The authors state they "believe this model is representative of the capabilities of many contemporary LLMs" (Section 4), but this claim is unverified. Several aspects of the findings could be model-specific:

- The PRM's quality and over-optimization behavior depend on PaLM 2-S\*'s output distribution. A model with different calibration properties or different error patterns might exhibit different difficulty-dependent scaling curves.
- The revision model's ability to learn from incorrect in-context examples depends on the base model's in-context learning capabilities, which vary substantially across model families.
- The MATH benchmark consists exclusively of competition-level math problems requiring symbolic reasoning. It is unclear whether the difficulty-dependent patterns (beam search hurting easy problems, revisions helping easy problems) generalize to other reasoning domains (code generation, logical reasoning, scientific QA) or to tasks requiring factual knowledge rather than inference.

The test set of 500 questions, split into five difficulty quintiles of ~100 each, then further split by two-fold cross-validation, means the compute-optimal policy is **selected based on ~50 questions per fold per bin**. This is a small sample, and the selected strategies may not be robust. The paper does not report confidence intervals on the compute-optimal scaling curves, making it difficult to assess whether the observed gains are statistically reliable at this sample size.

### The $14\times$ Larger Model Baseline Is Not Compute-Optimal

The FLOPs-matched comparison in Section 7 scales model parameters while holding training data fixed, following the LLaMA paradigm (Touvron et al., 2023). The authors acknowledge that this departs from compute-optimal pretraining (Hoffmann et al., 2022), where both data and parameters are scaled equally:

> "We choose this setting as it is representative of a canonical approach to scaling pretraining compute and leave the analysis of compute-optimal scaling of pretraining compute where the data and parameters are both scaled equally to future work."

This matters because a Chinchilla-optimal model trained with $14\times$ more total FLOPs would likely outperform a parameter-only-scaled model, making the pretraining baseline **weaker than it needs to be**. The reported advantages of test-time compute over pretraining (e.g., +27.8% on easy questions at $R \ll 1$) may shrink or reverse against a properly compute-optimal larger model. Additionally, the $14\times$ larger model uses only **greedy decoding** — no majority voting, no best-of-N, no search. Giving the larger model even a modest test-time compute budget (say, best-of-8) would create a much stronger baseline that is never tested.

### Verifier Over-Optimization Is a Hard Ceiling, Not a Solved Problem

The paper documents verifier over-optimization as a central limiting factor: beam search degrades easy-problem performance at high budgets (Figure 3, right), lookahead search — the strongest optimizer — paradoxically performs worst overall (Figure 3, left), and qualitative examples show degenerate outputs (repetitive steps, overly short solutions; Appendix M, Figures 29, etc.). The compute-optimal policy *mitigates* this by routing easy problems away from aggressive search, but it does not *solve* the underlying problem. On medium-difficulty problems where beam search is deployed, over-optimization still limits the scaling ceiling — the beam search curves in Figure 3 flatten and sometimes decline well before the budget is exhausted.

This means the compute-optimal approach is fundamentally bounded by verifier quality. Improving the PRM (e.g., through better training data, adversarial robustness, or ensemble methods) would likely shift the difficulty thresholds and change the optimal policy. The current results are therefore **specific to the verifier quality achievable with the Monte Carlo rollout training procedure** described in Appendix D. The paper does not explore how verifier improvements would alter the scaling landscape.

### Hard Problems Remain Essentially Unsolved

Across all methods — search, revisions, and their compute-optimal combinations — the hardest questions (difficulty bin 5) show **near-zero improvement** regardless of compute budget. In Figure 3 (right), bin 5 accuracy hovers at 1–3% for all methods and all budgets. In Figure 7 (right), bin 5 shows roughly 2–3% accuracy irrespective of the sequential-to-parallel ratio. In the FLOPs-matched comparison (Figure 9), the bin 5 scaling line is essentially flat near 0–5%.

This is a fundamental limitation: test-time compute can amplify existing capability but cannot create it. If the base model's pass@1 is near zero on a problem class, no amount of search or revision will help — there are no correct solutions in the proposal distribution to find or refine. The paper is candid about this (Section 7 takeaway box), but it means the approach offers **no path forward for genuinely novel or out-of-distribution reasoning** that exceeds the base model's training distribution. For such problems, pretraining remains the only viable path.

### Revisions and Search Are Studied Independently, Not Combined

The paper studies two complementary axes — PRM search and iterative revisions — but never combines them. Section 8 explicitly acknowledges this:

> "we did not experiment with PRM tree-search techniques in combination with revisions"

This is a significant gap because the two mechanisms have complementary strengths: revisions improve the proposal distribution (generating better candidates), while PRM search improves candidate selection (finding the best among generated candidates). Applying beam search to revision model outputs — or using the PRM to guide which revisions to pursue — could yield gains beyond either method alone. The current results therefore represent a **lower bound** on what a fully integrated system could achieve.

### The Revision Model Has a 38% Correct-to-Incorrect Reversion Rate

As noted in Section 6.1, approximately 38% of correct answers produced during a revision chain get "revised" back to incorrect answers in the subsequent step. This is a direct consequence of the training data construction: the model only sees incorrect-to-correct trajectories during training, so it has no signal for what to do when the current answer is already correct. The paper mitigates this with majority voting or verifier-based selection across the chain, but these are imperfect patches. A more principled solution — such as training the model to recognize when no revision is needed — is not explored.

The ReST$^{EM}$ experiment (Appendix K, Figure 16) further highlights the fragility of revision training. Attempting to optimize the revision model with RL-style training caused performance to **degrade substantially** with sequential revisions, likely because on-policy data collection amplified spurious correlations in the revision trajectories. This suggests that the revision approach is sensitive to training methodology in ways that are not fully understood, and the positive results depend on specific choices (offline data construction, edit-distance-based pairing) that may not transfer to other settings.

### No Accounting for Latency or Wall-Clock Time

The paper measures compute in "generations" (number of complete solutions sampled), which is a reasonable proxy for total FLOPs but ignores **latency**. Sequential revisions are inherently serial — each revision depends on the previous one — while parallel best-of-N can be executed simultaneously with sufficient hardware. A strategy that allocates 128 generations as 64 sequential × 2 parallel takes roughly $64\times$ longer wall-clock time than one that runs 128 parallel samples simultaneously. For latency-sensitive applications (interactive assistants, real-time decision-making), the sequential-heavy strategies favored by the compute-optimal policy on easy problems may be impractical regardless of their accuracy advantages. The paper does not discuss this tradeoff.

### Difficulty Bins Are Static and Coarse

The five-quintile difficulty binning is a coarse discretization of a continuous space. Within a single bin, there may be substantial heterogeneity — a question at the easy end of bin 3 and one at the hard end of bin 3 would receive the identical strategy, even though different strategies might be optimal. A finer-grained or continuous difficulty estimate could improve allocation, but would also require more data to estimate the optimal policy per bin. The paper does not explore the sensitivity of results to the number of bins.

Additionally, difficulty bins are computed once and treated as fixed. There is no mechanism for **dynamically adjusting** the strategy mid-computation — for instance, starting with a few parallel samples, assessing whether the problem appears easy or hard based on the verifier's scores on those initial samples, and then allocating the remaining budget accordingly. Such an adaptive scheme could subsume the difficulty estimation cost into the solution process, but it is not explored.

### Scope Limited to Closed-Form Answer Problems

The MATH benchmark has ground-truth answers that can be checked with exact string matching (via a grading function). This enables both difficulty estimation (via pass@1) and the PRM training pipeline (via Monte Carlo rollout correctness). Many important real-world applications — open-ended generation, dialogue, creative writing, complex multi-step planning — lack such clean correctness signals. Extending the compute-optimal framework to tasks where correctness is ambiguous, multi-dimensional, or subjective would require fundamentally different verifier training and difficulty estimation approaches that the paper does not address.

## 7. Implications and Future Directions

### How This Work Changes the Landscape

This paper shifts the conversation around LLM scaling from a pretraining-centric view toward one where **inference-time compute is a first-class resource to be optimized**. Before this work, the scaling laws community (Hoffmann et al., 2022; Sardana and Frankle, 2023) had established principled frameworks for allocating pretraining compute between model size and data quantity, but no analogous framework existed for inference-time compute. The field's default approach was best-of-N sampling — uniformly applied regardless of problem characteristics. This paper provides empirical evidence that such uniform allocation is deeply suboptimal and that a difficulty-conditioned policy can recover $4\times$ efficiency gains, establishing the conceptual foundation for **inference-time scaling laws** that parallel pretraining scaling laws.

The practical implication is a reframing of how organizations should think about their total compute budget. Rather than the prevailing paradigm of "train the largest model you can afford, then deploy it with greedy decoding," the results suggest a regime where it is sometimes **more cost-effective to train a smaller model and invest the savings in smarter inference**. This is not a universal prescription — the paper is careful to show it fails on hard problems and at high inference-to-pretraining ratios — but for deployments where the problem distribution skews toward easy-to-medium difficulty (which is likely common in production settings where users ask questions within the model's rough capability range), the economics strongly favor test-time compute.

Perhaps equally important is the paper's reconciliation of conflicting prior findings. The observation that self-correction works on easy problems but fails on hard ones (Section 6), and that search helps on medium problems but over-optimizes on easy ones (Section 5.3), provides a unified explanation for why Huang et al. (2023) found "LLMs cannot self-correct reasoning" while Madaan et al. (2023) found that self-refinement helps. These studies were implicitly testing on different difficulty distributions. This resolution is valuable because it converts a confusing set of contradictory results into a coherent picture with clear boundary conditions, enabling future researchers to design experiments with appropriate difficulty controls.

The identification of **verifier over-optimization as the primary bottleneck** for test-time compute scaling (Sections 5.3, 8) is also a landscape-changing finding. It redirects research attention: rather than developing ever-more-sophisticated search algorithms (which the paper shows can be counterproductive — lookahead search underperforms simpler methods), the priority should be building more robust verifiers that remain reliable under aggressive optimization pressure. This is analogous to how the RLHF community recognized reward hacking as a central challenge, and it opens a parallel research agenda for test-time compute.

### Follow-Up Research This Work Enables or Suggests

**1. Cheap difficulty estimation.** The most immediate bottleneck the paper identifies is the cost of estimating question difficulty. The current method (2048 samples + PRM scoring) is far too expensive for deployment. The paper explicitly calls for future work on "pretraining or finetuning models to directly predict difficulty of a question" (Section 8). A natural approach would be to train a lightweight classifier — possibly distilled from the PRM — that takes only the question text as input and predicts the difficulty bin. If such a classifier could achieve accuracy comparable to the PRM-based method, the compute-optimal framework becomes immediately practical. Another direction is **adaptive difficulty estimation**: start by generating a small number of samples (say, 4–8), use the verifier's score distribution on those samples as a quick difficulty signal, and then allocate the remaining budget accordingly. This amortizes difficulty estimation into the problem-solving process itself.

**2. Combining search and revisions.** The paper studies PRM tree-search and iterative revisions as independent mechanisms but explicitly notes they were never combined (Section 8). The natural next step is to use the revision model as the proposal distribution within beam search — at each step of the search tree, the model conditions on previous rejected branches as context, potentially producing higher-quality candidate steps. Alternatively, the PRM could guide which revisions to pursue: rather than blindly generating a long revision chain, use the PRM's per-step scores to decide when a revision is on track versus when to restart from scratch. This combination could break through the performance ceiling that each method individually hits, particularly on medium-difficulty problems where both mechanisms show complementary strengths.

**3. Robust verifiers resistant to over-optimization.** The over-optimization phenomenon documented in Figure 3 (right) and Appendix M suggests a direct research agenda: how do we train process reward models that remain calibrated under aggressive search? Potential directions include adversarial training (where the PRM is trained on search-generated solutions, not just i.i.d. samples), ensemble verification (aggregating predictions from multiple independently trained PRMs), or constrained search methods that penalize solutions deviating too far from the base model's typical output distribution (a KL-penalty approach analogous to what is used in RLHF). The paper's finding that the PRM trained with Monte Carlo soft labels behaves differently from binary-label PRMs (Appendix E) hints that label quality and calibration are important levers.

**4. Self-improvement loops.** Section 8 envisions "distilling the outputs of applying additional test-time compute back into the base LLM, enabling an iterative self-improvement loop." This is a direct extension: use compute-optimal test-time strategies to generate high-quality solutions on training data, then fine-tune the base model on these solutions, then repeat. The compute-optimal framework provides a principled way to allocate the test-time budget in each iteration of such a loop, potentially making self-improvement pipelines (similar to STaR/ReST$^{EM}$; Zelikman et al., 2022; Singh et al., 2024) significantly more sample-efficient. The paper's finding that the ReST$^{EM}$-trained revision model degraded (Appendix K) suggests that naïve self-improvement can backfire, making careful test-time budget allocation during data generation even more important.

**5. Extension to other domains and modalities.** All results are on MATH with PaLM 2-S\*. Replicating the study on code generation (e.g., HumanEval, MBPP), logical reasoning (e.g., ARC, FOLIO), scientific QA, and open-ended generation tasks would determine which findings are universal and which are domain-specific. Code generation is particularly promising because it has clean correctness signals (unit tests) that can serve as both verifier training data and difficulty estimation oracles. Domains without clean correctness signals (dialogue, creative writing) pose a harder challenge and may require fundamentally different verifier architectures.

**6. Continuous and dynamic allocation policies.** The five-bin discretization is coarse and static. Future work could develop continuous difficulty estimates with smooth policy functions (e.g., using the PRM's average score as a continuous feature that parameterizes the strategy choice via a learned policy network). More ambitiously, **dynamic policies** could adjust strategy mid-computation: begin with a few parallel samples, assess the score distribution, and decide in real-time whether to switch to beam search, revisions, or continue parallel sampling. This is the exploration-exploitation tradeoff the paper flags in Section 3.2, and it connects naturally to the multi-armed bandit and Bayesian optimization literatures.

**7. Compute-optimal pretraining + compute-optimal inference jointly.** The paper studies the pretraining-inference tradeoff with a fixed (parameter-only-scaled) pretraining baseline. A complete picture would jointly optimize the pretraining recipe (model size, data quantity, data mixture) and the inference strategy (search method, revision depth, difficulty allocation) under a total FLOPs constraint. This is a substantially harder optimization problem but is the natural endpoint of the research direction this paper initiates.

### Practical Applications and Downstream Use Cases

**On-device deployment with smaller models.** The paper's most directly actionable finding for practitioners is that on easy-to-medium problems, a small model with compute-optimal test-time strategies can match or exceed a $14\times$ larger model. For applications where the problem distribution is skewed toward routine tasks (customer support, document summarization, standard coding tasks), this suggests a deployment architecture where a small on-device model handles most queries with variable test-time compute, and only genuinely hard queries are routed to a larger cloud-based model. The difficulty estimator serves double duty: it determines how much test-time compute to allocate *and* whether to escalate to the larger model.

**Cost-efficient batch inference pipelines.** For organizations running large-scale batch inference (e.g., evaluating thousands of math problems, generating training data, or scoring candidate solutions), the compute-optimal framework offers a concrete recipe for reducing costs. Rather than applying a uniform best-of-256 to every problem, estimate difficulty first and allocate budgets per-problem: easy problems might need only 4–8 generations with sequential revisions, medium problems might get 32–64 generations of beam search, and hard problems might receive best-of-N with the full budget or be flagged for human review. The $4\times$ efficiency gain translates directly to cost savings at scale.

**Data generation for self-improvement.** When using LLMs to generate training data for themselves (as in STaR, ReST$^{EM}$, or rejection sampling fine-tuning), the quality and diversity of generated solutions matter enormously. The compute-optimal framework provides a principled way to allocate the generation budget: spend more compute on medium-difficulty problems (where search and revisions can push the model to produce correct solutions it wouldn't find by chance) and less on easy problems (where a few samples suffice) or hard problems (where no amount of compute helps). This targeted allocation could make self-improvement pipelines significantly more data-efficient.

**Verifier development as a research investment.** The paper's finding that verifier quality is the primary bottleneck (not search algorithm sophistication) has direct implications for research prioritization. Teams working on test-time compute should invest more heavily in training better PRMs — with more on-policy data, better calibration, and robustness to adversarial optimization — rather than developing more complex search algorithms. The Monte Carlo rollout training procedure described in Section 5.1 and Appendix D provides a concrete, human-label-free recipe for PRM training that practitioners can adopt immediately.

### When to Prefer This Method Over Alternatives

**Prefer compute-optimal test-time scaling when:**
- The problem distribution includes a substantial fraction of easy-to-medium problems (where the base model's pass@1 is non-trivially above zero).
- The inference-to-pretraining token ratio $R$ is low (e.g., self-improvement pipelines, one-time evaluation tasks, low-volume high-stakes applications).
- Deploying a larger model is infeasible due to hardware constraints (on-device, edge deployment).
- A reliable verifier (PRM or ORM) can be trained on the base model's output distribution, ideally using the Monte Carlo rollout approach when human labels are unavailable.

**Prefer scaling pretraining instead when:**
- The problem distribution skews toward genuinely hard problems outside the base model's capability range (difficulty bin 5 in this paper's taxonomy).
- The inference volume is very high ($R \gg 1$), making per-query inference costs dominate the total budget.
- Latency is critical, since sequential revision strategies introduce serial dependencies that increase wall-clock time regardless of total FLOPs.
- The domain lacks clean correctness signals for training verifiers (open-ended generation, subjective evaluation tasks).

**For replication or integration**, the key practical choices are: (1) train a PRM using Monte Carlo rollouts from your own base model (not borrowed from another model family — distribution shift matters, as the paper found with PRM800k); (2) use last-step aggregation and best-of-N weighted selection for answer selection; (3) estimate difficulty using the PRM's average final-answer score over a modest number of samples; (4) use two-fold cross-validation on a held-out set to select the best strategy per difficulty bin; (5) default to beam search ($M = 4$) on medium-difficulty problems and best-of-N weighted on easy problems. If building revision models, use offline data construction with edit-distance-based incorrect-correct pairing rather than on-policy rollouts, and always apply within-chain selection (majority or verifier) rather than taking the final revision output.