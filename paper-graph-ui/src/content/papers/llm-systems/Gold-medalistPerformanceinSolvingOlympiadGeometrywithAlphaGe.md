# Gold-medalist Performance in Solving Olympiad Geometry with AlphaGeometry2

**ArXiv:** [2502.03544](https://arxiv.org/abs/2502.03544)

## 🎯 Pitch

AlphaGeometry2 sets a new benchmark for automated reasoning by outperforming average IMO gold medalists in solving Olympiad geometry problems, raising the solve rate from 54% to 84% on problems from 2000 to 2024. By substantially expanding its geometric language, automating problem formalization and diagram generation, strengthening its symbolic engine, and leveraging large language models in a novel multi-tree search framework, AlphaGeometry2 demonstrates that AI can achieve creative, human-style mathematical reasoning—paving the way for advanced AI tutors, automated proof assistants, and formal math understanding.

---

## 1. Executive Summary
AlphaGeometry2 (AG2) is a neuro-symbolic system that solves high‑school Olympiad geometry problems at a level exceeding an average IMO gold medalist. It advances the prior AlphaGeometry system by expanding its geometry language, automating problem formalization and diagram generation, speeding up and strengthening the symbolic engine, training larger language models, and introducing a knowledge‑sharing multi‑tree search—raising the solve rate on all IMO 2000–2024 geometry problems to 84% (42 of 50 formalized tasks; Figure 8, Table 4).

## 2. Context and Motivation
- Problem addressed
  - Automated solution of challenging Euclidean geometry problems written in natural language, including problems that require constructing auxiliary points and multi‑step synthetic (human‑style) proofs.
  - Prior AlphaGeometry (AG1) solved 54% of IMO 2000–2024 geometry problems but was limited by its domain language coverage (66%), a slower symbolic engine, a simpler search, and the need for manual formalization/diagramming (Introduction; end of Section 2; Section 4; Section 6).

- Why this matters
  - Geometry requires nontrivial reasoning about relationships among points, lines, and circles; success here is a stringent testbed for broader mathematical and symbolic reasoning.
  - A robust solver can power tutoring, proof assistants, and formalization pipelines that translate human problems into machine-checkable proofs.

- Prior approaches and shortcomings
  - Algebraic “bashing”: Wu’s method, Gröbner bases, area method convert geometry to algebra (Introduction). Powerful but often produce opaque proofs and may struggle to mimic human synthetic reasoning.
  - Synthetic approaches: deduction databases and full-angle methods provide readable, geometric proofs but need strong search and domain knowledge. AG1 used a synthetic, neuro‑symbolic approach but could not express “moving locus” statements, linear equations of angles/distances, or non-constructive point definitions; it also lacked robust handling of “double points” (same geometric point with different names), and it required manual problem encoding (Section 2; Section 4.1; Section 3).

- Positioning
  - AG2 keeps the synthetic, neuro‑symbolic direction but expands the language, automates inputs, and accelerates symbolic search. Relative to contemporaries like TongGeometry, AG2 deliberately generates synthetic training data from random diagrams (not human-made diagrams) to avoid data leakage and broaden theorem distributions (Section 5). In the main benchmark (Table 4), AG2’s full system outperforms prior systems and medalist baselines.

## 3. Technical Approach
AG2 is an end‑to‑end pipeline from natural language problems to verified geometric proofs. It couples a stronger symbolic engine (`DDAR2`) with larger language models and a shared‑knowledge multi‑tree search.

1) Expanded domain-specific language (DSL) for geometry (Section 2)
- Goal-query predicates
  - `acompute a b c d`: “Find the angle between AB and CD.”
  - `rcompute a b c d`: “Find the ratio AB/CD.”
- Linear equation predicates (new expressivity):
  - `distmeq ... t1 t2 ... tn y`: linear equation in logarithms of distances, e.g., t1·log(A1B1)+...+y=0.
  - `distseq ... t1 t2 ... tn`: linear equation of distances, e.g., t1·A1B1+...=0.
  - `angeq ... t1 t2 ... tn y`: linear combination of undirected line angles plus constant equals zero; here `d(AB)` is the direction of line AB vs. horizontal.
- Locus statements (moving objects) via 11 cases (Table 2), using `*` as a moving/fixed placeholder, e.g., “point on fixed circle” or “line through fixed points.” This enables “When X moves on Y, Z moves on T.”
- Topological/non-degeneracy predicates available to proofs:
  - `sameclock a b c d e f` (clockwise ordering),
  - `noverlap a b` (distinct points),
  - `lessthan a b c d` (AB < CD; used in SSA congruence).
- Overlapping points:
  - `overlap a b` marks that A and B are the same geometric point, enabling “name unification.”
  - `cyclic_with_center a1 ... an x` lets proofs assert common circle centers when points coincide.
- Non‑constructive definitions:
  - AG2 can define a point by three or more predicates (AG1 allowed ≤2), supporting problems where points are constrained but not directly constructible in a simple order.
- Effect: DSL coverage of IMO 2000–2024 geometry rises from 66% to 88% (end of Section 2; Figure 8 shows which problems become “attemptable”).

2) Automated formalization and diagram generation (Section 3)
- Auto‑formalization from natural language
  - Uses the Gemini model in a prompted, few‑shot setup: query five times and aggregate into a final AG‑language formalization. This successfully formalizes 33 of 44 formalizable IMO problems; consistent on easier problems.
- Automatic diagram generation for non‑constructive constraints
  - Initialization strategies: random; constructive order; heuristic order.
  - For a point X with predicates involving already-built points, infer lines/circles X must lie on; sample candidate positions (10 per point) and choose the one minimizing a loss.
  - Optimize all point coordinates with two‑stage numeric methods:
    - Stage 1: Adam optimization minimizes a sum of losses: squared residuals of exact constraints f_c(x)=0; softplus penalties for inequality/equality topological constraints; two non‑degeneracy losses (coordinate norm and pairwise 1/(|AB|^2+ε)).
    - Stage 2: Gauss‑Newton‑Levenberg refines to a numerical solution (addresses convergence issues noted in earlier work).
  - Success: diagrams found for 43/44 problems within an hour using repeated random restarts (Section 3).

3) Stronger and faster symbolic engine, `DDAR2` (Section 4)
- What `DDAR` does: computes the “deduction closure,” i.e., all facts logically implied by initial facts under a fixed set of geometry deduction rules.
- New capability: reasoning with “double points” (Section 4.1)
  - If two constructions generate the same geometric point under different names (e.g., X is intersection of lines a and b; X′ is intersection of line a and circle ω), `DDAR2` can prove they coincide and transfer facts across names. This enables “reformulation” strategies (Figure 1).
- Faster algorithm for key pattern searches (Section 4.2)
  - Similar triangles: hash triangle “shapes” over all point triples and detect duplicates (much less than the AG1 worst case O(N^8)).
  - Cyclic quadrilaterals: hash triples (X, segment AB, angle ∠AXB) in a symbolic normal form. When a match repeats, a cyclic relation is found.
  - An AR submodule maintains linear relations among angles, distances, and log‑distances, reducing expressions to normal forms for hashing and deduction.
- Faster implementation (Section 4.3)
  - Core linear algebra (Gaussian elimination) is rewritten in C++ and bound to Python via `pybind11`.
  - Speed benchmark on 25 IMO problems: average engine run time drops from 1179.57 ± 8.06 s (`DDAR1`) to 3.447 ± 0.055 s (`DDAR2`)—a >300× speedup.

4) Better synthetic training data (Section 5)
- Scale and distribution
  - Larger random diagrams, more complex problems/proofs, and balanced question types (Figure 2a–c).
  - Proofs with and without auxiliary points are balanced (~50:50 in AG2 vs. 9:91 in AG1; Figure 2c).
- Locus‑type data
  - Track movement dependencies `P(.)` for each point during random construction (Table 3 shows examples). Use 17 detection cases to generate 11 flavors of “moving locus” theorems (Table 5).
- Greedy minimal‑problem extraction (Figure 3)
  - Replace AG1’s exponential subset search with a linear‑pass, reverse‑topological pruning that preserves constructibility and provability—enabling far larger data generation.

5) Novel search: SKEST—Shared Knowledge Ensemble of Search Trees (Section 6; Figure 4)
- Multiple heterogeneous search trees run in parallel and share “interesting facts” (facts about the original problem that DDAR cannot derive without a specific auxiliary point) via a shared database.
- Tree variants:
  - Classic tree (one auxiliary point per node).
  - Multi‑aux tree (LM can propose several auxiliary points at once; models are trained to output whole proofs, which helps here).
  - Operator‑prompted tree (force diversity across first tokens like `coll`, `cong`, `cyclic`, `perp` to explore varied constructions).
  - Deep‑narrow and shallow‑wide beams to cover different search regimes.
- Systems engineering
  - Asynchronous LM workers and `DDAR2` workers; LM writes candidate nodes to a database; `DDAR2` pulls and evaluates. Worker pools are shared across problems to rebalance compute.

6) Stronger language model and inference interface (Section 7)
- Training (Section 7.1)
  - Sparse Mixture‑of‑Experts Transformer models based on Gemini; trained either:
    1) from scratch with a custom AG tokenizer,
    2) fine‑tuned from math‑specialized Gemini models in natural language (Appendix A),
    3) multimodally from scratch with diagram images (Appendix B).
  - ~300M synthetic proofs used for training; held‑out evaluations on synthetic and on an IMO set with known proofs (imo_eval).
  - Larger models consistently reduce loss (Figure 5).
- Inference (Section 7.2)
  - Use top‑k sampling with temperature 1.0 and k=32; diversity is essential. With greedy decoding, only 2/26 aux‑requiring IMOs are solved; with t=1.0, k=32 (no tree search), 9/26 are solved (Figure 6 visualizes uniqueness vs. temperature).
  - “Analysis string”: before proposing constructions, the LM receives serialized sets:
    - `S1`: facts deducible from premises by `DDAR2`.
    - `S2`: facts deducible when also assuming the goal.
    - `S3`: numerically true facts read from the diagram.
    - The model input is: problem statement + serialized `S1` + `(S2−S1)` + `(S3−S2)` (Section 7.2). This orients the LM toward gaps between current knowledge and the goal.

## 4. Key Insights and Innovations
- Expanded, more expressive geometry DSL (Section 2; Table 2; new predicates)
  - Novelty: supports linear angle/distance equations, “find angle/ratio” queries, locus problems, topological checks, and overlapping points.
  - Significance: raises formalizability coverage of IMO problems from 66% to 88% (end of Section 2), enabling broader problem types (including non‑constructive points).

- Handling “double points” within a synthetic deduction engine (Section 4.1; Figure 1)
  - Novelty: allows reformulating goals by proving coincidence of differently constructed names for the same geometric point and transferring properties.
  - Significance: unlocks powerful equivalence arguments that are common in expert solutions but were previously unavailable to AG1.

- Knowledge‑sharing multi‑tree search (SKEST) (Section 6; Figure 4)
  - Novelty: parallel, heterogeneous beam searches that exchange distilled, DDAR‑verified “interesting facts,” rather than raw generations.
  - Significance: increases robustness and depth of exploration; makes better use of multiple model variants; materially contributes to the global 84% solve rate.

- Orders‑of‑magnitude engine acceleration with targeted pattern hashing and C++ (Sections 4.2–4.3)
  - Novelty: task‑specific symbolic hashing for similar triangles and cyclic quadrilaterals; AR normal forms; C++ implementation.
  - Significance: >300× speedup (Section 4.3) makes aggressive data generation and deep search feasible.

- Fully automated front‑end: formalization and diagram generation (Section 3)
  - Novelty: Gemini‑based auto‑formalization plus a three‑stage numeric diagram construction (loss‑based initialization → Adam → Gauss‑Newton‑Levenberg).
  - Significance: 33/44 successful auto‑formalizations and 43/44 diagrams within 1 hour, removing key human bottlenecks.

- Data generation with movement dependencies and greedy minimality (Section 5; Table 5; Figure 3)
  - Novelty: tracks `P(.)` movement sources to synthesize locus theorems reliably; scalable minimal‑subset pruning.
  - Significance: richer, harder, and more balanced synthetic training corpus (Figure 2), especially for auxiliary‑heavy proofs.

## 5. Experimental Analysis
- Evaluation protocol
  - Benchmarks:
    - IMO‑AG‑50: all IMO 2000–2024 geometry problems formalized into 50 AG tasks (Section 8; Figure 8).
    - IMOSL‑AG‑30: 30 of the hardest formalizable Shortlist problems not used at IMO (Appendix D; Figure 15).
  - Metrics: solve rate (DDAR‑verified proofs); language coverage (formalizability); engine runtime.
  - Baselines and comparisons: AG1, TongGeometry, medalist levels (average bronze/silver/gold), and engine‑only variants (Table 4).

- Main results
  - Overall performance (Table 4; Figure 8)
    > “AG2 full setting” solves 42/50 on IMO‑AG‑50, surpassing the average gold medalist level (40.9) and exceeding AG1 (27/50).
    > “AG2 DDAR” (engine‑only) already solves 16 problems (Figure 8), slightly above AG1 DDAR’s 14.
  - Shortlist robustness (Appendix D; Figure 15)
    > On IMOSL‑AG‑30, AG2 full setting solves 20/30. This indicates improved robustness but leaves a significant tail of very hard problems.
  - Engine speed (Section 4.3)
    > Average runtime drops from 1179.57 s to 3.447 s across a 25‑problem set (50 runs each), enabling much broader search and data generation.
  - DSL coverage (end of Section 2)
    > Coverage rises 66% → 88%, enlarging the solvable set by including locus and linear‑equation problems.
  - Search ablations (Figure 9)
    > For a single tree, best configuration is beam size 128, depth 4, 32 samples; larger beams or more samples do not add solves.
  - Training progression (Figure 7)
    > A single LM coupled with classic tree solves 27/50 after ~200M tokens of training, climbing with more tokens.
  - Decoding temperature (Figure 6)
    > Diversity (unique samples) increases up to T≈1.0; higher T increases syntax errors.
  - Multi‑modal and tokenizer/DSL ablations (Appendix A–B)
    > Training with a large general tokenizer vs. a small custom AG tokenizer yields similar end performance; training directly in natural language instead of AG DSL also yields similar results on IMOs (Appendix A), suggesting robustness to tokenization and language representation.
    > Multimodal models (with diagram images) do not improve standalone solve rates, likely due to crowded diagrams and VLM visual weaknesses; they are still beneficial when ensembled via SKEST (Appendix B).

- Qualitative case studies (Appendix C)
  - IMO 2024 P4: constructs an unexpected point E on BI with ∠AEB = 90° to tie incenter and midpoints via triangle similarities, yielding a proof within 30 seconds verified by an IMO expert (Appendix C; Figure 11).
  - IMO 2013 P3: single auxiliary point D (a nonsymmetric arc midpoint) reveals a key cyclicity (B, A1, D, Ia) that forces right angle, an unconventional but elegant synthetic approach (Appendix C; Figure 12).
  - IMO 2014 P3: proves a stronger result OH ⟂ BD using reflections and cyclic quadrilaterals, avoiding heavy trigonometry or inversion often used by humans (Appendix C; Figure 13).
  - IMOSL 2009 G7: synthesizes a purely synthetic solution to a problem usually solved by algebraic methods (Appendix C; Figure 14).

- Validation of LM‑only proofs (Appendix E; Figure 16)
  > Step‑level verification shows few syntax errors and many steps labeled “verified” or “unverified but numerically correct,” indicating the LM can generate meaningful partial proofs. However, full proofs without `DDAR2` remain unreliable, so the symbolic engine is still essential at inference.

- Overall assessment
  - The experimental suite is diverse (coverage, speed, ablations, qualitative analysis) and supports the central claims: broader coverage, faster and smarter search, and higher solve rates. The remaining failures (e.g., problems needing inversion/projective tools) are explicitly diagnosed (Section 8).

## 6. Limitations and Trade-offs
- Scope limitations of the DSL (end of Section 2; Section 8)
  - Not yet handling inequalities, non‑linear equations, variable numbers of points, or 3D geometry—about 12% of IMO problems remain outside language coverage.
- Missing geometric machinery in `DDAR2` (Section 8)
  - No built‑in inversion, projective geometry, or radical‑axis reasoning. Two unsolved IMOs (2018 P6, 2023 P6) are flagged as needing such tools. While solvable synthetically in principle, proofs would become much longer and search would be harder.
- Auto‑formalization accuracy (Section 3)
  - Formalizes 33/44 formalizable IMOs—nontrivial failures remain, especially on harder, more ambiguous statements.
- Diagram generation reliance on numeric optimization (Section 3)
  - Success rate is high (43/44 within 1 hour), but the approach is heuristic and may struggle with pathological configurations or tight topological constraints.
- Dependence on high compute
  - Training large MoE models and running multi‑tree searches with TPUv4 replicas is resource‑intensive (Section 6 “System design details”).
- LM proof hallucinations (Appendix E)
  - Although step syntax is good, many steps remain “unverified,” and full LM‑only proofs are not yet reliable enough—necessitating the symbolic engine.
- Data distribution
  - Synthetic data come from random diagrams rather than curated human problems (Section 5). This avoids leakage but could introduce distributional shift compared to competition problems.

## 7. Implications and Future Directions
- Field impact
  - Demonstrates that a carefully engineered neuro‑symbolic stack—expanded DSL, very fast symbolic engine, and knowledge‑sharing LM search—can surpass expert human benchmarks on a long‑standing challenge set (Figure 8, Table 4). This strengthens the case for hybrid systems in formal reasoning tasks beyond geometry.

- Enabled research directions
  - Extend DSL to inequalities, non‑linear relations, variable‑size constructions, and 3D geometry to close the remaining 12% coverage.
  - Incorporate projective/inversion/radical‑axis modules and/or hybridize with algebraic methods (e.g., Wu’s method) to attack the hardest problems identified in Section 8.
  - Reinforcement learning over tree‑of‑subproblems: Section 9 suggests breaking problems into subproblems could boost success on complex cases.
  - Improve auto‑formalization with supervised fine‑tuning and larger exemplars, moving toward reliable natural‑language‑to‑proof pipelines (Section 9).
  - Develop faster, more faithful visual tokenization and spatial reasoning if multi‑modal gains are to materialize (Appendix B).

- Applications
  - Intelligent tutoring: step‑by‑step, verifiable geometric proofs and auxiliary constructions.
  - Proof assistants and graders for geometry competitions or coursework.
  - Theorem discovery and diagram synthesis from free‑form text (Sections 3 and 5).
  - Methodological transfer: the SKEST paradigm and “analysis string” interface (Section 7.2) could guide LMs with symbolic context in other domains (algebra, program verification).

> Headline result: “AG2 full setting solves 42/50 of all IMO 2000–2024 geometry problems” (Table 4; Figure 8), with engine runtime reduced by >300× (Section 4.3) and DSL coverage expanded to 88% (end of Section 2). These advances collectively shift the frontier of machine mathematical reasoning toward robust, automated understanding and solution of complex geometric problems.
