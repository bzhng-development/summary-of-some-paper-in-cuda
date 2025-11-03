# Is Chain-of-Thought Reasoning of LLMs a Mirage? A Data Distribution Lens

**ArXiv:** [2508.01191](https://arxiv.org/abs/2508.01191)
**Authors:** Chengshuai Zhao, Zhen Tan, Pingchuan Ma, Dawei Li, Bohan Jiang, Yancheng Wang, Yingzhen Yang, Huan Liu
**Institutions:** Arizona State University

## 🎯 Pitch

This paper reveals that the effectiveness of Chain-of-Thought (CoT) prompting in large language models stems from pattern matching within the training data, rather than genuine reasoning. By using a synthetic testbed, the study demonstrates that CoT performance dramatically declines under distribution shifts, exposing critical limitations in real-world applications where reliable and interpretable model outputs are essential.

---

## 1. Executive Summary
This paper argues that the apparent benefits of Chain‑of‑Thought (CoT) prompting in large language models are largely a product of pattern matching to the training data rather than true reasoning. Using a controlled synthetic environment called DataAlchemy, it shows that CoT performance collapses under out‑of‑distribution (OOD) shifts along three axes—task, length, and format—and formalizes why performance should degrade as the training–test distribution diverges.

## 2. Context and Motivation
- Problem addressed
  - Many practitioners treat CoT prompting—asking a model to write intermediate steps like “Let’s think step by step”—as evidence of deliberate reasoning. Yet coherent steps sometimes end with a wrong conclusion or the right conclusion is produced via faulty steps. The paper seeks to answer “why and when does CoT fail?” using a data‑distribution perspective.
- Importance
  - Real‑world deployments (e.g., in medicine or finance) can be misled by fluent but unfaithful reasoning chains. Understanding CoT’s failure modes is essential for safe, reliable use. The paper explicitly shows such unfaithfulness in an example where an LLM correctly recalls the leap‑year rule for 1776 and then concludes the opposite (Introduction, p. 1).
- Prior approaches and gaps
  - CoT and its extensions (zero‑shot CoT, self‑consistency, tree‑of‑thought) have improved benchmark scores (Sec. 2.1). A growing body of critiques finds CoT brittle to distractions, formatting changes, and logically irrelevant perturbations (Sec. 2.2). What is missing is a clean, systematic assessment that isolates training‑data effects from massive pretraining confounders.
- Positioning
  - This paper reframes CoT through a data‑distribution lens: CoT works when test inputs resemble training inputs. It contributes a controlled testbed that trains models from scratch so results are not due to unknown pretraining artifacts, and it decomposes generalization into task, length, and format (Figure 1 and Sec. 3–4).

## 3. Technical Approach
Step‑by‑step overview of the framework and experiments.

- Core hypothesis (Sec. 3)
  - Treat CoT as conditional text generation constrained by the training distribution. If the test distribution `Dtest` diverges from the training distribution `Dtrain`, performance should deteriorate proportionally to the discrepancy Δ(`Dtrain`, `Dtest`).
  - This is formalized by the bound in Eq. (4):
    - Test risk `Rtest(fθ)` ≤ Training risk `Rtrain(fθ)` + Λ·Δ(`Dtrain`,`Dtest`) + statistical term `O(√(log(1/δ)/n))`.
    - In plain language: even a perfect in‑distribution learner will incur extra error proportional to how far the test data is from what it saw in training; Λ captures model/task sensitivity.

- The DataAlchemy environment (Sec. 4; Figure 2)
  - Goal: A controllable, fully synthetic setup where LLMs are trained from scratch and all distribution shifts are known and tunable.
  - Building blocks
    - “Basic atoms” A–Z (26 symbols). An “element” `e` is a fixed‑length tuple of atoms, e.g., `(A,P,P,L,E)` (Eq. 5). Default length `l=4`, yielding 26^4 = 456,976 unique elements.
    - Two primitive transformations (Sec. 4.2):
      1) `ROT` transformation `frot(e, n)` shifts each character forward by `n` mod 26 (Eq. 6). Default `f1 = frot(·,13)` (ROT13).
      2) `Cyclic position shift` `fpos(e, n)` rotates the positions by `n` (Eq. 7). Default `f2 = fpos(·,1)` (right shift by 1).
    - Compositional transformation `fS` chains these operations: `ek = fk(...f2(f1(e)))` (Eqs. 8–9). The chain yields natural “reasoning steps”: `e → e(1) → ... → ê` (Eq. 10). These intermediate states are written into the dataset to produce explicit CoT traces (`<think>` and `<answer>` examples shown in Appendix C).
  - Models and training (Sec. 4.3; Appendix B)
    - Small GPT‑2‑style decoder: 4 layers, hidden size 32, 4 attention heads; BPE tokenizer; trained for 10 epochs with AdamW, cosine LR schedule, and weight decay.
    - Default inference temperature 1e‑5; temperature and model‑size robustness reported later (Sec. 8).
  - Metrics
    - `Exact match` (string equality), `Edit distance` (Levenshtein; lower is better), and `BLEU` (n‑gram overlap; higher is better). They evaluate the “reasoning steps,” the “final answer,” and the “full chain” separately.

- Three generalization axes (Figure 1; Figure 2; Sec. 3, 5–7)
  - Task generalization (Sec. 5): Can the model handle unseen transformations or novel elements?
    - Two sub‑types:
      - Transformation generalization: train on some compositions (e.g., `f1∘f1`) and test on others (e.g., `f2∘f2`). Four regimes (Figure 2):
        - `ID` (in‑distribution): same composition at test as train.
        - `CMP` (composition): new composition of known transforms.
        - `POOD` (partial OOD): at least one transform unseen in that position.
        - `OOD`: transformation types at test unseen in training.
      - Element generalization: test on new atom combinations or entirely new atoms (Figure 2).
    - A paper‑specific complexity score `TGC(C)` (Eq. 11) counts novelty of atoms, functions, and their order. A failure threshold theorem (Eq. 12; Appendix A.2) claims success probability decays exponentially once `TGC` exceeds a threshold, modeling why performance collapses under sufficient novelty.
  - Length generalization (Sec. 6):
    - Text length: train at element length `l=4`, test at `l∈{2,3,5,6}` with different padding strategies; Figure 7 and Table 3.
    - Reasoning steps: train with `k=2` steps, test at `k=1` or `k=3` (Figure 8).
    - Proposition 6.1 posits a “Gaussian” error vs. length gap curve (Eq. 13), meaning error grows smoothly with distance from training length.
  - Format generalization (Sec. 7):
    - Perturb the prompt surface form by token insertion, deletion, or modification at different probabilities; also vary where noise is applied (elements vs. transformation tokens vs. generic prompt tokens). A “Format Alignment Score” (Def. 7.1) measures closeness between a test prompt and known training prompts via cosine similarity of prompt embeddings.

- Why these design choices?
  - Using two orthogonal, transparent transformations (ROT13 and cyclic shift) makes ground‑truth reasoning steps, compositionality, and failure modes unambiguous (Figure 2).
  - Training small models from scratch ensures that successes and failures are not inherited from opaque pretraining knowledge.
  - Separating evaluation into task, length, and format isolates where generalization fails and what kind of “data shift” causes it.

## 4. Key Insights and Innovations
- A data‑distribution lens for CoT (Sec. 3; Eq. 4; Figure 1)
  - Innovation: CoT is modeled not as an internal “reasoning algorithm” but as conditional generation bounded by distributional similarity between train and test. The generalization bound in Eq. (4) makes this explicit.
  - Significance: This reframes many CoT successes as interpolation within the training manifold and predicts failure when distribution discrepancy grows.

- DataAlchemy: a controlled, compositional testbed (Sec. 4; Figure 2)
  - Innovation: A from‑scratch training environment with precise control over symbols, operations, lengths, and prompts. Intermediate states are naturally interpretable as “reasoning steps.”
  - Significance: Removes pretraining confounds and enables targeted OOD diagnostics across axes.

- Systematic three‑axis dissection of CoT generalization (Sec. 5–7)
  - Innovation: Decomposes “reasoning generalization” into task, length, and format, with measurable sub‑regimes (ID/CMP/POOD/OOD). Introduces `TGC` (Eq. 11) and a length extrapolation model (Eq. 13).
  - Significance: Provides a principled way to test when CoT fails and why—showing brittleness even under moderate shifts.

- Evidence of unfaithful or superficial chains (Sec. 5; Table 2; Appendix D)
  - Novel observation in this controlled setting: models can produce correct‑looking intermediate steps with wrong answers or the right answers with wrong steps due to coincidental properties (e.g., commutativity) rather than correct reasoning. Examples are documented in Appendix D.1 and summarized in Table 2.

## 5. Experimental Analysis
- Evaluation setup
  - Model and data: GPT‑2‑style model trained on elements of length `l=4` unless varying `l` (Sec. 4.3). Default transforms `f1=ROT13`, `f2=position shift by 1`. Three metrics: `Exact match`, `Edit distance`, `BLEU`. They evaluate “reasoning,” “answer,” and “full chain” (Sec. 4.3).
  - Baseline regimes: `ID`, `CMP`, `POOD`, `OOD` for transformations (Figure 2). For elements, `ID`, `CMP` (new orderings of seen atoms), and `OOD` (new atoms).

- Main quantitative findings
  - Transformation generalization collapses outside the training composition.
    - Table 1 shows full‑chain exact match dropping from 100% in `ID` (train `f1∘f1` → test `f1∘f1`) to 0.01% in `CMP`, 0% in `POOD`, and 0% in `OOD`. Edit distance rises from 0 to 0.2997; BLEU falls from 1.0 to 0.2947.
    - Figure 3 plots BLEU vs. edit distance and shows a clear degradation trend as the measured distribution shift increases.
    - Table 2 exposes mismatches between reasoning steps and answers. For example, after training on a set that excludes `f2∘f2`, test on `f2∘f2` yields “Reasoning exact match 100%” but “Answer exact match 0.01%,” indicating the model produces plausible intermediate steps yet the final answer is wrong. Conversely, when testing `f2∘f1` after training on `f1∘f2`, “Answer exact match 100%” but “Reasoning exact match 0%” suggests correct answers via shortcut properties (commutativity), not faithful reasoning. Appendix D.1 provides concrete strings illustrating these phenomena.
  - Small amounts of new data “patch” failures without true generalization.
    - Figure 4 shows that supervised fine‑tuning (SFT) on as little as λ≈1.5×10^-4 of unseen transformation data rapidly raises exact match across `ID`, `CMP`, `POOD`, and `OOD`. This supports the view that CoT success expands with exposure rather than abstract reasoning.
  - Element generalization is also brittle.
    - Figure 5 heatmaps: moving from `ID` to `CMP` or `OOD` drives exact match to 0% across transformations; BLEU often falls to near zero, showing the model cannot handle new combinations or new atoms.
    - Figure 6a: SFT improves generalization; closer combinations (smaller edit‑distance `n`) need less data. Figure 6b reveals a gap between “Reasoning step,” “Answer,” and “Full chain” accuracy during SFT, indicating persistent inconsistency between steps and conclusions.
  - Length generalization fails in both text length and reasoning steps.
    - Text length (Table 3): training at `l=4`, exact match is 100% at `l=4` but 0% at `l∈{2,3,5,6}`. BLEU drops from 1.0 to 0.4214 (l=2), 0.5471 (l=3), 0.6220 (l=5), 0.4763 (l=6). Appendix D.1 shows the model sometimes pads or trims steps to mimic the seen length.
    - Padding strategy matters (Figure 7): simple max‑padding does not help; “group” padding (chunking sequences into segments) improves BLEU and edit distance when testing unseen lengths.
    - Reasoning‑step length (Figure 8): training on `k=2` does not generalize to `k=1` or `k=3`. Mixing in a fraction of target‑`k` data increases performance on that target but simultaneously reduces performance on the original training setup—highlighting a distribution trade‑off.
  - Format generalization is sensitive to surface changes (Figure 9).
    - Insertion, deletion, and modification noise all hurt; insertion is most damaging. Noise on the “element” and “transformation” tokens degrades performance far more than noise on generic prompt tokens (Figure 9b). This indicates that CoT depends heavily on surface form around the structured parts of the prompt.
  - Robustness checks (Sec. 8; Figure 10)
    - Temperature: BLEU and edit distance remain stable from very low to moderate temperatures (1e‑5 to 1.0). At high temperatures (5–10), BLEU collapses and edit distance spikes, across `CMP/POOD/OOD` (Figure 10a).
    - Model size: After pretraining on `f1∘f1` and SFT on `f2∘f2`, accuracy as a function of SFT ratio is similar across sizes from 68K up to 543M parameters (Figure 10b), suggesting the data‑distribution effect dominates capacity in this setup.

- Do the experiments support the claims?
  - Yes for the central claim: across three orthogonal axes, quantitative results repeatedly show sharp degradation outside the training distribution, together with documented step/answer inconsistencies (Table 2; Appendix D.1). The SFT “patching” result (Figures 4 and 6) is consistent with a pattern‑matching explanation.

- Notable qualitative failure cases (Appendix D)
  - “Correct reasoning, wrong answer” under new compositions (D.1.2).
  - “Correct answer, wrong reasoning” due to orthogonality/commutativity (D.1.1).
  - “No response” on unseen elements (D.1.3).
  - “Forcing seen length” in reasoning output (D.1.4).

## 6. Limitations and Trade-offs
- Synthetic scope
  - The world of two transformations (ROT13 and position shift) and 26 symbols is intentionally simple. While this isolates effects cleanly, it does not capture linguistic ambiguity, world knowledge, or tool use found in real tasks.
- Theoretical assumptions
  - The “failure threshold” (Eq. 12) assumes independent contributions to failure from novel atoms, functions, and patterns; real LLMs may have interactions that violate this independence. The generalization bound (Eq. 4) leverages a standard Lipschitz/IPM framing; it is qualitatively informative but likely loose quantitatively.
- Model scale and pretraining
  - Many real LLMs are orders of magnitude larger and heavily pretrained. Although Figure 10b probes sizes up to 543M, the core results are from small models trained from scratch; transfer to frontier models with specialized RL or verification mechanisms may differ.
- Limited variety of formats
  - Format perturbations are token‑level noise; human prompt variations can be more structured (e.g., paraphrase, discourse markers, nested instructions).
- Distribution trade‑offs
  - Mixing more target‑distribution data improves that target but degrades the original one (Figure 8), highlighting a general tension between specialization and broad coverage that is not fully resolved here.
- Computational simplicity vs. ecological validity
  - The clear, interpretable operations that make analysis easy also mean CoT could, in principle, learn the exact algorithms; yet it still fails. This is a strength for the paper’s core argument but a caveat for external validity to complex natural tasks.

## 7. Implications and Future Directions
- What changes in practice
  - Treat CoT outputs as hypotheses, not proofs. Build pipelines that verify steps and final answers (e.g., with symbolic checks or unit tests) before acting on them, especially in high‑stakes contexts. The Discussion (Sec. 9) explicitly warns against over‑reliance on “fluent nonsense.”
  - Evaluate with OOD protocols. Incorporate task, length, and format stress tests, not just random splits that resemble training (Sec. 9).
  - View SFT as a patch, not a cure. Figure 4 and Figure 6 show that small amounts of in‑distribution data can “fix” failures, but only by expanding the training manifold; no evidence of abstract reasoning is gained.

- Research directions
  - Explicit algorithm learning and verification: Combine CoT with symbolic execution, proofs, or program synthesis so steps are checked, not just generated.
  - Data curricula for compositionality: Train on systematically varied compositions and lengths to encourage algorithmic generalization; explore positional encodings and architectures shown to improve length generalization (cf. Sec. 2.3 discussion).
  - Faithfulness metrics and diagnostics: Go beyond final‑answer accuracy; evaluate step‑level correctness and causal necessity of steps (Table 2 highlights why this matters).
  - Distribution‑aware inference: Detect when a query is OOD w.r.t. the model’s “reasoning manifold” and trigger fallback strategies (retrieval, tools, or human review).
  - Broader format robustness: Study paraphrase, multilingual prompts, and domain‑specific jargon; expand the “Format Alignment Score” idea (Def. 7.1) to real prompt spaces.

- Downstream use cases
  - Safety monitors for CoT‑enabled systems: Given a prompt and chain, estimate distribution distance (Δ) or PAS (Def. 7.1) and flag high‑risk generations.
  - Data collection: Targeted synthesis of training data that covers key compositions and lengths identified by `TGC` (Eq. 11) to reduce brittleness where it matters.

> Overall, the paper’s controlled evidence—Figures 3–10, Tables 1–3, and the qualitative cases in Appendix D—supports a clear takeaway: CoT, as currently elicited, behaves like structured pattern matching bounded by the training distribution. Understanding and mitigating this boundary is crucial for building trustworthy reasoning systems.
