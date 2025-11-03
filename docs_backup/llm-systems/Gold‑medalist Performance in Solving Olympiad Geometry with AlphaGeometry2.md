# Gold‑medalist Performance in Solving Olympiad Geometry with AlphaGeometry2

**ArXiv:** [2502.03544](https://arxiv.org/abs/2502.03544)
**Authors:** Yuri Chervonyi, Trieu H. Trinh, Miroslav Olšák, Xiaomeng Yang, Hoang Nguyen, Marcelo Menegali, Junehyuk Jung, Vikas Verma, Quoc V. Le, Thang Luong
**Institutions:** Google DeepMind, University of Cambridge, Georgia Tech, Brown University

## 🎯 Pitch

AlphaGeometry2 (AG2) is a groundbreaking neuro-symbolic system that solves complex Olympiad-level Euclidean geometry problems, achieving an 84% solve rate on all IMO geometry problems from 2000–2024. By innovating with a richer formal language, an enhanced symbolic engine, innovative multi-tree knowledge-sharing search algorithms, and extensive training on improved synthetic data, AG2 sets a new standard in mathematical reasoning, exceeding even top human performers and paving the way for advanced scientific automation.

---

## 1. Executive Summary
AlphaGeometry2 (AG2) is a neuro‑symbolic system that solves Olympiad‑level Euclidean geometry problems, achieving 84% solve rate on all IMO geometry problems from 2000–2024 and surpassing the average gold medalist benchmark. It accomplishes this through four coordinated upgrades: a broader formal language, a faster and stronger symbolic engine, a multi‑tree knowledge‑sharing search algorithm, and larger/better‑trained language models with improved synthetic data.

## 2. Context and Motivation
- Problem gap
  - Automating Olympiad geometry requires constructing auxiliary points, chasing angles/ratios, and handling non‑constructive statements—tasks where pure language models hallucinate and pure symbolic systems lack creativity.
  - AlphaGeometry1 (AG1) made a first step but was limited by:
    - Language coverage (its domain language could formalize only 66% of IMO 2000–2024 problems; Section 2).
    - Symbolic engine speed and capabilities (no “double points,” i.e., coincident points with different names; Section 4.1).
    - Modest search (single beam) and smaller models/data.
    - Result: 54% solve rate on all IMO geometry problems when tested retrospectively (Introduction; Figure 8).
- Importance
  - Theoretical: Geometry is a canonical testbed for symbolic reasoning, proof search, and tool‑use by LMs.
  - Practical: Progress in robust mathematical reasoning and auto‑formalization informs broader scientific automation.
- Prior approaches and shortfalls
  - Algebraic “bashing” (Wu’s method, Gröbner bases, Area method) can decide many theorems but often produces opaque proofs and may be brittle on Olympiad constructions (Introduction).
  - Synthetic approaches (Deduction Database, Full‑angle method) mirror human reasoning but historically needed hand‑crafted rules or problem‑specific heuristics.
  - AG1 combined a language model with a deduction database but struggled on non‑constructive/locus problems and suffered from slow closure computations.
- Positioning
  - AG2 remains fully synthetic (no algebraic bashing) but expands the language, strengthens the deductive engine, scales the LM and synthetic data, and introduces a novel multi‑tree search with knowledge sharing (Sections 2, 4–7).

## 3. Technical Approach
AG2 is a pipeline that turns a natural‑language problem into a formal proof, with the following stages and components.

- Expanded domain‑specific language (DSL) for geometry (Section 2)
  - New “compute‑type” queries:
    - `acompute a b c d`: find the angle between AB and CD.
    - `rcompute a b c d`: find the ratio AB/CD.
  - Linear equations of geometric quantities:
    - `distmeq ... t1 t2 ... y`: linear equation over log‑distances (useful for multiplicative constraints).
    - `distseq ... t1 t2 ...`: linear equation over distances.
    - `angeq ... t1 t2 ... y`: linear equation over line directions (undirected angle to horizontal).
    - These capture problems like IMO 2024 P4 with linear relations in angles/lengths (Section 2).
  - Locus problems (moving objects) encoded via unified predicate patterns (Table 2), using a `*` token to denote the moving/fixed placeholder. Eleven cases cover points moving on lines/circles, lines moving through fixed points with fixed direction/angle, fixed distance constraints, etc.
  - Diagram/topology checks as first‑class predicates:
    - `sameclock a b c d e f`: triples have same clockwise orientation.
    - `noverlap a b` and `overlap a b`: points are distinct or coincident.
    - `lessthan a b c d`: AB < CD (used, e.g., SSA triangle case).
    - `cyclic_with_center ... x`: compactly asserts centers coincide for cyclic constraints.
  - Non‑constructive definitions are allowed (points may be defined by 3+ predicates); AG2 therefore adds automated diagram generation (Section 3).

- Automated formalization and diagram generation (Section 3)
  - Auto‑formalization: a few‑shot Gemini prompt translates natural language into the AG DSL. A 5‑way sampling plus consolidation step produces one formalization; this succeeds on 33/44 formalizable IMO problems (Section 3).
  - Diagram generation for non‑constructive specs:
    - Initialization via three strategies: random, construction order, or heuristic order.
    - For a point X, infer candidate line/circle constraints from predicates; sample up to 10 candidate positions; pick the best by a loss combining exact constraints, inequality/equality topological constraints, and non‑degeneracy terms (Section 3, bullet list of five loss components).
    - Two‑stage solver: Adam on aggregated loss, then Gauss‑Newton‑Levenberg to refine solutions (Figure 3 is for a different greedy pruning; the solver is detailed in Section 3).
    - Outcome: diagrams found for 43/44 problems within 1 hour via repeated restarts.

- Stronger, faster symbolic engine `DDAR2` (Section 4)
  - DDAR = Deductive Database Arithmetic Reasoning. It constructs a “deduction closure”: all facts implied by given facts under a rule system.
  - New capability: “double points” (Section 4.1). AG2 can reason when two differently named points coincide, enabling proof reformulations. Example (Figure 1): to show the intersection of lines `a,b` lies on circle `ω`, AG2 constructs X′ at `a∩ω`, proves X′ lies on `b`, deduces X=X′, hence X is on `ω`.
  - Faster algorithm (Section 4.2): instead of exhaustively matching rules, AG2 hard‑codes fast searches for essential patterns:
    - Similar triangles: hash “shape” over all triples; detect repeats.
    - Cyclic quadrilaterals: hash triples (A,B,∠AXB) over all pairs (X, segment AB).
    - A lightweight algebraic sub‑engine maintains normal forms of linear equations over angles, distances, and log‑distances to support hashing and inference.
  - Faster implementation (Section 4.3): core linear algebra (Gaussian elimination) written in C++ and bound to Python via `pybind11`, yielding >300× speedup on a 25‑problem benchmark (from 1179.6±8.1 s to 3.45±0.055 s per run; Section 4.3).

- Better synthetic training data (Section 5)
  - Data is generated from random diagrams only (no human problems to avoid contamination).
  - For each deduced fact, a traceback extracts minimal premises, auxiliary points, and steps proving it.
  - Larger and more balanced:
    - Bigger diagrams (roughly 2× the size), deeper proofs (up to 10× steps), and balanced question types and with/without‑aux distributions (Figure 2a–c).
  - Locus‑type theorems: AG2 tracks movement dependencies `P(.)` during construction and emits “when X moves …, Y lies on …” statements in 17 detection cases (Table 5, Section 5; Table 3 shows examples of how `P(.)` is computed).
  - Faster minimal‑problem extraction: replaces exponential subset search with a greedy pruning algorithm (Figure 3) processed in reverse topological order to preserve constructability.

- Novel search algorithm `SKEST`: Shared Knowledge Ensemble of Search Trees (Section 6; Figure 4)
  - Many differently configured beam searches run in parallel; when any attempt fails, it writes “interesting” proved facts (i.e., not tied to its private aux point) to a shared database.
  - Other trees can then use those facts, effectively collaborating.
  - Tree variants include:
    - “Classic” single‑aux beam search (as in AG1).
    - Multi‑aux per node (LM proposes several constructions at once to jump deeper).
    - Operator‑conditioned prompts to diversify first tokens (`x00 a : cong`, `x00 a : coll`, etc.).
    - Deep‑narrow and shallow‑wide beams to cover different search profiles.
  - Infra: multiple LM replicas on TPUv4, asynchronous LM and DDAR workers sharing a task DB (Section 6, System design details).

- Better language models and inference interface (Section 7)
  - Models: sparse MoE Gemini‑based Transformers trained on ~300M synthetic theorems. Three setups: train from scratch with AG tokenizer; fine‑tune math‑specialized Gemini on AG data; multi‑modal variant that also ingests problem diagrams (Appendix A, B).
  - The “analysis string” (Section 7.2): before proposing aux points, the LM is fed three serialized fact sets:
    - `S1`: what DDAR proves from premises alone.
    - `S2`: what DDAR proves if the goal is assumed true.
    - `S3`: what holds numerically in the current diagram.
    - Input format: `<problem> serialized(S1) serialized(S2−S1) serialized(S3−S2)`.
  - Inference: top‑k sampling with temperature 1.0, k=32 is important for diversity (Figure 6 shows unique‑sample ratio vs temperature). Greedy decoding (t=0) solves only 2/26 aux‑requiring IMOs without search; t=1.0 with 32 samples solves 9/26 even without tree search (Section 7.2).

## 4. Key Insights and Innovations
- Expanded, problem‑covering DSL (Section 2)
  - What’s new: compute‑type queries, linear equations over angles/distances/log‑distances, robust locus predicates, and explicit topology checks.
  - Why it matters: DSL coverage of IMO 2000–2024 rises from 66% (AG1) to 88% (Section 2). This unlocks many previously unformalizable problems, especially involving “moving objects” and “find the angle/ratio” tasks.

- Knowledge‑sharing ensemble search (`SKEST`) with the “analysis string” (Sections 6 and 7.2)
  - What’s new: multiple heterogeneous beams share proved facts; the LM is primed with structured, DDAR‑derived analyses (`S1`, `S2`, `S3`) before proposing constructions.
  - Why it matters: greater robustness and efficiency—trees avoid re‑discovering the same facts, and the LM gets grounded hints, reducing blind exploration. This is a fundamental shift from AG1’s single‑beam, LM‑only‑aux interface.

- Symbolic engine upgrades including “double points” and 300× speed (Sections 4.1–4.3)
  - What’s new: reasoning where two named points coincide enables powerful proof reformulations (Figure 1). The new search and C++ core drastically reduce closure computation time.
  - Why it matters: enables harder proofs within fixed budgets and supports the large synthetic data generation program.

- Higher‑quality synthetic data at scale, including locus theorems (Section 5; Figure 2 and Table 5)
  - What’s new: longer proofs, more complex diagrams, balanced distributions, and formalization of movement dependencies for locus statements.
  - Why it matters: training signal better matches Olympiad reasoning patterns (auxiliary constructions, multi‑step chains), improving the LM’s constructive creativity.

- Automated formalization and non‑constructive diagram generation (Section 3)
  - What’s new: Gemini‑based auto‑formalization (33/44 successes) and a three‑stage optimization pipeline for diagrams (43/44 within 1 hour).
  - Why it matters: reduces manual bottlenecks toward a fully automated system from natural language to proof.

## 5. Experimental Analysis
- Evaluation methodology
  - Benchmarks:
    - IMO‑AG‑50: all IMO 2000–2024 geometry problems formalized into 50 AG tasks (some IMOs split into two; Section 8).
    - IMOSL‑AG‑30: 30 of the hardest shortlist problems that fit the AG2 language (Appendix D; Figure 15).
  - Metrics: number solved by the system (proof found and verified by DDAR).
  - Baselines and comparators: AG1 (Trinh et al., 2024), the algebraic TongGeometry variants, human medalist averages, and DDAR alone (Table 4; Figure 8).

- Main quantitative results
  - Overall performance (Figure 8; Table 4):
    - AG2 solves 42/50 on IMO‑AG‑50 and 30/30 on IMO‑AG‑30 within its full setting.
    - Quote:
      > “AG2 achieves a new state‑of‑the‑art solving rate of 84% on all IMO geometry problems from 2000 to 2024” (Section 8).
    - DDAR alone in AG2 solves 16 problems in IMO‑AG‑50 (Figure 8; Table 4), up from 14 in AG1.
    - Comparisons (Table 4):
      - AG1: 27/50. Average gold medalist: 40.9/50. AG2 full: 42/50.
      - On IMOSL‑AG‑30: AG2 full 30/30; TongGeometry full 30/30; Wu+AG1 27/30; AG1 25/30.
  - Training dynamics (Figure 7):
    - One LM coupled with classic search reaches 27/50 after ~2×10^8 tokens and improves steadily with more tokens.
  - Inference ablations (Figure 9):
    - Single‑tree performance peaks around beam size 128, beam depth 4, 32 samples. Larger beams/samples bring diminishing returns or no gain.
  - Auto‑formalization/diagram generation (Section 3):
    - Formalization: 33/44 successes on formalizable IMOs using a few‑shot Gemini scheme.
    - Diagrams: 43/44 problems obtain diagrams within 1 hour by iterative restarts; three‑stage optimizer ensures convergence.
  - Engine speed (Section 4.3):
    - On 25 hard IMOs, C++ DDAR2 averages 3.45 s per run vs 1179.6 s for DDAR1 (>300×).
  - Proof‑by‑LM alone (Appendix E; Figure 16):
    - Step‑level verification shows few syntax errors; many steps verified or plausibly correct. Small and larger models are similar on this metric, indicating the LM can generate substantial proof fragments even without DDAR.

- Case studies and creativity
  - Appendix C analyzes three notable solutions:
    - IMO 2024 P4: a single auxiliary point E on BI with ∠AEB = 90° unlocks pairs of similar triangles to relate midpoints and the incenter, solving an angle‑sum identity; judged full‑mark quality by an IMO gold medalist (Appendix C; Figure 11).
    - IMO 2013 P3: unconventional choice of the arc midpoint D yields cyclicities with the A‑excenter I_a, essentially characterizing right triangles (Appendix C; Figure 12).
    - IMOSL 2009 G7: constructs a web of circumcenters and reflections to exploit the equilateral condition on triangle XYZ, deriving that ABC is equilateral (Appendix C; Figure 14).
  - These examples underscore the benefit of multi‑aux proposals and “double points” reasoning.

- Do results support the claims?
  - Yes. The aggregate numbers (Table 4, Figure 8) show clear improvement over AG1 and beyond average gold medalist performance. Ablations (Figure 9) and training curves (Figure 5, Figure 7) make the dependence on data/model size and decoding diversity transparent. The engine speedup and the success of auto‑formalization/diagramting further support the end‑to‑end viability.

- Mixed/conditional findings
  - Multi‑modal LM (diagram+text) alone does not improve solve rate, though it adds diversity that helps in the ensemble (Appendix B).
  - Tokenizer and language choice (AG DSL vs natural language) do not materially affect results when model/data scale are matched; both reach similar performance (Appendix A, Figure 10).

## 6. Limitations and Trade-offs
- Language coverage still incomplete (Section 2; Section 8)
  - Not supported: 3D geometry, inequalities, non‑linear equations beyond the introduced linear forms, and problems with a variable number of points. These account for the remaining 12% of IMO problems not covered by the AG2 language.
- Missing higher‑level geometry machinery (Section 8)
  - Techniques like inversion, projective geometry, and radical axis are not explicitly implemented in DDAR. Some unsolved IMOs (e.g., 2018 P6, 2023 P6) are known to benefit from them; AG2 could in principle solve them synthetically but would likely require longer proofs and more aux points within larger compute budgets.
- Reliance on sampling diversity (Section 7.2; Figure 6)
  - AG2 depends on high‑temperature, top‑k sampling (t=1.0, k=32) and multi‑tree search; greedy decoding is insufficient on aux‑heavy problems.
- Auto‑formalization and diagramming are strong but not perfect (Section 3)
  - Formalization succeeds on 33/44 formalizable IMOs; diagram generator needs repeated restarts and careful numerical optimization; certain configurations remain hard (1/44 failures).
- Compute and engineering complexity
  - End‑to‑end system uses TPUv4 serving, many concurrent search trees, and a C++ symbolic core. While DDAR2 is fast per run, the overall pipeline’s throughput depends on orchestration, database sharing, and sampling budgets.
- Proof verification by DDAR rules
  - The system verifies steps with its rule set; if a valid mathematical argument lies outside these rules, it may be marked “unverified” (Appendix E). Thus, the proof notion is tied to the expressiveness of DDAR.

## 7. Implications and Future Directions
- Field impact
  - Demonstrates that a well‑engineered neuro‑symbolic stack can reach and surpass expert human performance on a long‑standing reasoning benchmark without algebraic bashing. The `SKEST` paradigm and the “analysis string” offer a blueprint for tool‑augmented LMs in other proof domains.
- Near‑term research enabled
  - Extend DSL to inequalities, non‑linear relations, and variable‑arity constructs; add projective/inversion/radical‑axis rules to DDAR to reduce proof length on advanced problems.
  - Reinforcement learning and hierarchical search to break problems into sub‑goals (Section 9).
  - Improved auto‑formalization via more supervised fine‑tuning and better type‑checking (Section 3; Section 9), potentially closing the remaining gap to “NL → formal proof” reliability.
  - Hybridization with algebraic provers (e.g., Wu’s method) under a unified search framework to cover the remaining hard cases while keeping synthetic readability.
- Practical applications
  - Educational tools that can not only solve but also explain geometry problems with human‑style steps and diagrams.
  - General scientific reasoning assistants where a language model coordinates specialized tools, shares intermediate results across searches, and constructs auxiliary entities to simplify proofs or derivations.
  - Data generation engines for training reasoning models in other mathematical domains via random‑diagram analogs and greedy minimal‑proof extraction.

> Bottom line (Figure 8; Table 4): “AG2 solves 42/50 IMO geometry problems (84%), surpassing the average gold medalist, with key gains coming from broader language coverage, a much faster and more capable symbolic core, and a knowledge‑sharing ensemble search guided by a stronger LM trained on deeper, more balanced synthetic data.”
