# Genesys: Autonomous Multi‑Agent Discovery of Language Model Architectures

**ArXiv:** [2510.12345](https://arxiv.org/abs/2510.12345)
**Authors:** First Author, Second Author, Third Author
**Institutions:** Institution A, Institution B, Institution C

## 🎯 Pitch

Genesys introduces a groundbreaking multi-agent system that leverages large language models (LLMs) to autonomously discover, implement, and verify novel language-model architectures, achieving competitive performance against human-designed baselines. By transforming architecture research into a structured program search, Genesys not only accelerates discovery but also democratizes access, potentially reshaping the landscape of language model innovation and deployment.

---

## 1. Executive Summary (2-3 sentences)
This paper introduces Genesys, a multi‑agent system that uses large language models (LLMs) to autonomously discover, implement, pretrain, and evaluate new language‑model (LM) architectures. Within a controlled discovery environment (LMADE), Genesys proposes and codes architectures as modular “blocks,” verifies them with a compute‑aware “Ladder‑of‑Scales” training strategy, and evolves better designs; across 1,062 fully verified models (14M–350M parameters), the best discovered designs match or surpass strong human baselines on most small‑LM benchmarks (e.g., 6/9 wins at 350M; Table 5).

## 2. Context and Motivation
- Problem addressed
  - Can LLMs be used not merely to write code but to conduct end‑to‑end, verifiable research that discovers better LM architectures? The paper targets the specific, high‑value problem of discovering novel autoregressive LM blocks that outperform standard Transformer‑style blocks (§1, §3).
- Why it matters
  - Architecture decisions strongly affect LM capability, efficiency, and scalability, but exploring the huge design space demands literature expertise, reliable coding, and expensive verification (pretraining and evaluation). Automating this process would accelerate research and lower the barrier to entry (§1, §3.1).
- Prior approaches and gaps
  - LLM‑driven “automated scientific discovery” (ASD) systems such as AI Scientist or AgentLab focus on open‑ended ideas with weak verification loops; discoveries are hard to check and reproduce (§1–2).
  - Neural Architecture Search (NAS) usually assumes a fixed, narrow operation space (e.g., head counts, kernel sizes) and does not model full research workflows, literature grounding, or open‑ended code generation (§2).
- Positioning
  - Genesys combines ASD with NAS‑like search but in a broader “program search” setting: it lets LLM agents propose, implement, and verify arbitrary block programs, not just choose among a fixed set of operators. It builds a discovery environment with a literature engine and a verification pipeline, couples that with a genetic‑programming (GP) backbone over factorized code, and enforces rigorous, automated checks (§3.2, §4).

## 3. Technical Approach
The system consists of a discovery environment (LMADE) and a discovery system (Genesys).

- LMADE: the discovery environment (§3.2)
  - Knowledge Engine (KE)
    - A curated reference library of 297 LM papers stored in a vector database, plus tools to query arXiv, Semantic Scholar, Papers with Code, and the web (Perplexity) (Fig. 2; §B.1.1–B.1.3). Returned items include text chunks and code snippets.
  - Verification Engine (VE)
    - A common code interface for all candidate blocks: the `Generalized Autoregressive Block` (`GABBase`, Fig. 3②, Fig. 18). Each block takes a tensor sequence `X` and an auxiliary dictionary `Z`, and returns updated `(X, Z)`; this normalizes diverse architectures as functions `(X, Z) → (X, Z)` (§3.1; Fig. 3①–②).
    - Symbolic checker that statically and dynamically validates block code for formatting, initialization, forward/backward passes, causality, differentiability, and “effectiveness” (e.g., stable gradients on a tiny training loop) (Table 1; §B.1.2).
    - Automated pretraining and evaluation: filtered SmolLM corpus (“SmolLM‑1/8‑Corpus”, 33.94B train tokens; Table 9; §C.1.1), AdamW training with standard hyperparameters (Table 12), and LM‑Eval on 29 tasks with caching and speedups (§3.2; §B.1.4; §C.1.3).

- Genesys: the multi‑agent discovery system (§4)
  - Code factorization and the evolution tree
    - Each block program is decomposed into a tree of `Generalized Autoregressive Units` (`GAUs`)—fine‑grained components that also implement `(X, Z) → (X, Z)` (Fig. 3③–⑥; Fig. 17). Genesys stores GAU trees plus code, provenance, and fitness in an “evolution tree” (Fig. 4).
    - This supports GP operations: local mutation (replace or edit a unit/subtree) and crossover (merge units from multiple parents), illustrated in Fig. 5.
    - Formal guarantee: any composition of functions of type `Σ → Σ` admits a finite “unit tree” factorization; even non‑`Σ → Σ` modules can be “lifted” into `Σ → Σ` to apply the same scheme (Appendix A.2).
  - Designer agents: proposal and implementation (§4.2)
    - Proposal stage (Fig. 6 left; Fig. 7; Alg. 1)
      - A proposer LLM selects promising parent design(s) from the evolution tree plus references from the KE and writes a research proposal that modifies a specific GAU (or creates a new block). A separate reviewer LLM adversarially reviews it, checks novelty against prior proposals, and assigns a score; only proposals above a threshold proceed (§B.1.5).
    - Implementation stage (Fig. 6 right; Fig. 8–9; Alg. 3)
      - Planner LLM chooses the next GAU to implement and outlines the plan.
      - Coder LLM generates Python for that GAU; it may declare child GAUs to be implemented later. The Symbolic checker validates format/semantics; an observer LLM assesses quality and adherence; only if both pass is the unit “accepted.” Otherwise the system rolls back to the previous valid state and retries.
      - Crucially, implementation is “unit‑by‑unit” with checkpointing—a Viterbi‑style search (Fig. 9B) rather than one-shot code generation (Fig. 9A). Appendix A.1 proves the expected number of model calls grows linearly with the number of units for this strategy, versus exponentially for a single-shot approach under multiple constraints.
  - Verifiers and compute‑aware evolution (§4.3)
    - Distributed workers run designers and verifiers in parallel, communicating through the evolution tree (Fig. 1 right).
    - Design selection balances exploration and exploitation using a 2D quadrant on fitness and verification confidence (number of scales the design was trained at). Different quadrants are prioritized for designers vs verifiers (Fig. 10; Alg. 4).
    - Ladder‑of‑Scales (LoS) verification (Fig. 11; Alg. 5): the system runs many cheap small‑scale trials (e.g., 1,000 models at ~14M params on 0.7B tokens) and progressively fewer at larger scales (e.g., 5 trials at ~350M on 50B tokens), releasing budget to higher scales only when enough evidence accumulates at lower scales. An auto‑tuner adjusts `num_block` and `embed_dim` to hit the target parameter budget and tunes gradient accumulation to avoid OOM (Table 7; §B.1.4).

Analogy for the core mechanism: think of each architecture block as a LEGO model built from smaller, typed bricks (GAUs). The system proposes a new model design, then snaps in or swaps out bricks one at a time, checking stability at every step; a population of such models is continuously tested and the best ones get more “expensive” tests.

## 4. Key Insights and Innovations
- GAU/GAB program representation with theoretical backing
  - What’s new: representing arbitrary LM blocks as compositions of typed units `(X, Z) → (X, Z)` and using that as the search substrate (Fig. 3; §3.1; Appendix A.2).
  - Why it matters: it turns open‑ended code generation into a structured, checkable program search problem and enables principled GP (mutation/crossover) at the unit level instead of ad‑hoc whole‑file edits.
- Viterbi‑style, unit‑by‑unit code generation with rollback
  - What’s new: replacing one‑shot “write the whole block” prompting with incremental unit synthesis, checkpointing, and re‑tries (Fig. 8–9; Alg. 3).
  - Why it matters: analytically reduces expected model calls from exponential (single-shot) to linear (unit‑wise) in the number of constrained sub‑decisions (Appendix A.1). Empirically, the “Full” agent produced valid implementations in 92% of cases vs 6% for direct prompting (Table 4).
- A compute‑aware, end‑to‑end discovery loop
  - What’s new: a closed loop that grounds ideation in literature (KE), automates verification (VE), schedules experiments with a Ladder‑of‑Scales budget (Fig. 11), and uses a fitness‑confidence quadrant for selection (Fig. 10).
  - Why it matters: it couples idea generation to reliable, resource‑bounded pretraining and evaluation, enabling large‑scale, interpretable evolution (>1,000 verified models; §5, §C).
- Evidence that autonomous systems can discover competitive LM blocks
  - What’s new: 1,062 fully pre‑trained designs across 14M–350M parameters, with top discoveries outperforming strong human baselines on most small‑LM benchmarks (Tables 5, 14).
  - Why it matters: it demonstrates feasibility of LLM‑driven, verifiable architecture discovery at non‑trivial scale.

## 5. Experimental Analysis
- Evaluation setup
  - Data and training
    - Pretraining on SmolLM‑1/8‑Corpus (34.78B tokens total; 33.94B for train; Table 9) with standard small‑LM training recipes (§B.1.4; Table 12).
    - Scales: 14M–350M parameters. For headline comparisons: 125M (25B tokens) and 350M (50B tokens) (§5.3; §C.1.3).
    - LoS schedule: many trials at small scales and few at large scales (Fig. 11). Auto‑tuning fits model size and gradient accumulation (Table 7).
  - Benchmarks
    - 29 LM‑Eval tasks are available; nine “small‑LM‑informative” tasks are emphasized in final comparisons (e.g., BLiMP, WNLI, RTE, WinoGrande, CoLA, SST‑2, WSC, Inverse Scaling (IS), MRPC), selected by variance and difficulty (Table 16; §E.3.1).
  - Baselines
    - Human‑designed seeds: `GPT` (Transformer), `Mamba2` (SSM), `RWKV6/7` (modern RNN), `RetNet`, and `TTT` (test‑time training) (§4.3; Fig. 13; §5.3).
- Main quantitative results
  - 350M scale (50B tokens), zero‑shot evaluation on nine tasks (Table 5)
    - A discovered model (`Geogate`) achieves the best average accuracy (61.81%) vs `GPT` (61.78%) and `Mamba2` (61.45%).
    - Task highlights:
      - BLiMP: `VQH` 94.37% vs `GPT` 92.70% (win).
      - WNLI: `SRN` 65.52% (top) vs baselines ≤ 63.38%.
      - WSC: `Geogate` 58.96% (top).
      - MRPC: `Hippovq` 69.88% (top).
      - Some tasks still favor baselines (e.g., RTE: `Mamba2` 63.88% is highest; Table 5).
    - Overall, discovered models match or exceed baselines on 6/9 tasks while remaining competitive on the others (Table 5).
  - 125M scale (25B tokens), nine tasks (Table 14)
    - Discovered models win on 7/9 tasks and have the highest average score across all designs (Table 14).
- Discovery scale and throughput
  - 1,162 designs were proposed; 1,062 passed pretraining verification; >1B tokens, 2.76M lines of code, and 86K agent interactions were involved (p. 4). Training losses decrease consistently with parameters and tokens, reflecting expected scaling behavior (Fig. 46–47).
- Ablations and system diagnostics
  - Evolution quality over time (Table 2; Fig. 12–13)
    - Over the first 300 designs, the “Full” system shows the largest fitness improvement (Δ= +4.10%) and the best risk‑adjusted progress (Sharpe Ratio 0.69) with the smallest maximum drawdown (−0.38%). Without experiment feedback (w/o Exp), improvement drops (Δ= +2.20%, SR 0.263; Table 2) and the evolution graph becomes more random (Fig. 13B).
    - Removing literature access (w/o Lit) reduces stability and overall gain (Δ= +3.37%, SR 0.567; Table 2).
  - Code‑generation components (Table 4)
    - Valid implementation rate: Full 92%; removing unit‑based generation drops to 73%; removing the symbolic checker plunges to 30%; direct single‑shot prompting achieves only 6%.
    - Average lines of function‑body code (a proxy for non‑triviality): Full 181 vs Direct 49 (Table 4).
  - Runtime reliability (Table 3)
    - Verification‑time error rate rises sharply without experiment‑driven selection (w/o Exp 27.31% vs Full 8.61%), showing the value of fitness signals for avoiding fragile designs downstream.
- Do the experiments support the claims?
  - The results substantiate three core claims:
    - The unit‑based, checked implementation process is dramatically more reliable than direct prompting (Table 4).
    - The closed verification loop with LoS and fitness‑aware selection produces steadier evolutionary gains than ablations (Table 2; Fig. 12–13).
    - The best discovered architectures are competitive with well‑implemented human baselines at comparable scale and tokens (Tables 5, 14).
  - Robustness and diagnostics
    - Metric sensitivity analyses show the “Full” system’s advantage persists across population sizes and step sizes (Fig. 19).
    - Reviewer ratings correlate weakly with eventual fitness (ρ≈0.04), suggesting reviewers help filter egregious ideas but are not sufficient for quality prediction (Fig. 36); this underscores the need for verification‑based selection.

## 6. Limitations and Trade-offs
- Assumptions and scope
  - The approach assumes that performance at small scales correlates with larger scales (the premise behind Ladder‑of‑Scales; Fig. 11). While broadly observed in scaling‑law literature, this correlation can be imperfect for some design choices or tasks (§4.3; §6).
  - The block interface constrains designs to `(X, Z) → (X, Z)` units; Appendix A.2 justifies this as general via type‑lifting, but implementing very unconventional paradigms may still require careful lifting (§A.2.2).
- Computational constraints
  - The work focuses on 14M–350M ranges due to compute limits (Discussion §6). Billion‑parameter‑level discovery remains out of scope; efficiency techniques that depend on hardware‑specific kernels (e.g., FlashAttention) are not fully integrated due to evaluation complexity (§6).
- Error modes and overhead
  - Despite strong checks, ~16% of design sessions end invalid (Fig. 20), often due to late‑stage verification failures; functional and format error distributions indicate common pitfalls (Fig. 44–45).
  - The system still spends non‑trivial tokens/time per design (mean ~$28.6 per design across agents; Fig. 25–27). Optimal designer‑to‑verifier ratios require tuning; a theoretical analysis suggests ~0.56 verifiers per designer thread under median assumptions (§E.4.2).
- Search dynamics
  - Evolutionary graphs exhibit “hubness” (few popular parents), which can accelerate exploitation but risks premature convergence (Fig. 12 right; Fig. 39; Table 15).
- Generalization and external validity
  - Benchmarks emphasize small‑LM‑informative tasks (§E.3.1); while diverse, they do not cover long‑context or instruction‑following regimes. Some tasks remain dominated by specific baselines (e.g., RTE by Mamba2 in Table 5).

## 7. Implications and Future Directions
- How this changes the landscape
  - It reframes LM architecture research as a verifiable, end‑to‑end, LLM‑driven program search problem rather than a hand‑crafted or fixed‑space NAS exercise. The GAU/GAB abstraction plus LoS verification provides a reusable blueprint for ASD in other domains requiring code synthesis and iterative experimentation (§3–4; Appendix A).
- Practical applications
  - Rapid prototyping of architecture variants for small/medium LMs (on‑device models, edge inference) and for domain‑specific LMs where training budgets are tight.
  - “Design console” and artifacts (Fig. 15) enable engineers to inspect proposals, GAU trees, and code lineage for interpretability and reuse.
- Research directions
  - Scaling up: better cost models, multi‑fidelity simulation, and integration of hardware‑aware kernels (Discussion §6).
  - Learning‑to‑design: using reinforcement learning or meta‑learning to improve agent decision‑making from verification signals (§6).
  - Task‑targeted design: unit‑to‑performance analyses show that bag‑of‑units features can predict fitness and per‑task outcomes with non‑trivial accuracy (e.g., LOOCV F1 ≈0.658 for fitness, with 19 tasks >0.5; §E.3.3; Tables 17–18), suggesting automated, goal‑conditioned architecture synthesis.
  - Broader scientific discovery: Appendix A.2 shows the Σ→Σ factorization applies beyond LMs via type‑lifting, hinting that the methodology could extend to other scientific program‑search problems (e.g., optimizers, simulators).

> Representative headline results
> - Discovery scale: 1,062 fully verified models trained at 14M–350M (p. 4).
> - Reliability: 92% valid implementations with unit‑wise generation vs 6% for direct prompting (Table 4).
> - Evolution quality: Δ= +4.10% fitness improvement and SR=0.69 for the full system (first 300 designs) vs Δ= +2.20% and SR=0.263 without experiment feedback (Table 2).
> - Competitiveness: At 350M parameters, discovered models match or beat strong baselines on 6/9 tasks, and the best average (Geogate 61.81%) slightly edges GPT (61.78%) and Mamba2 (61.45%) (Table 5).

Overall, the paper delivers a principled, working blueprint for LLM‑driven, verifiable architecture discovery, combines theoretical justifications with large‑scale empirical evidence, and surfaces a set of practical tools (GAU/GAB, symbolic checks, LoS) that others can reuse.
