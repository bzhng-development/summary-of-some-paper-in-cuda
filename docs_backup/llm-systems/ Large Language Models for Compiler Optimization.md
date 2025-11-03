# Large Language Models for Compiler Optimization

**ArXiv:** [2309.07062](https://arxiv.org/abs/2309.07062)
**Authors:** Chris Cummins, Volker Seeker, Dejan Grubisic, Mostafa Elhoushi, Youwei Liang, Baptiste Roziere, Jonas Gehring, Fabian Gloeckle, Kim Hazelwood, Gabriel Synnaeve, Hugh Leather
**Institutions:** Google DeepMind (likely, per author affiliations)

## 🎯 Pitch

This paper introduces a 7B-parameter LLM that optimizes LLVM intermediate representation for code size by predicting compiler optimization pass sequences, achieving a 3.01% improvement over the `-Oz` baseline on 100,000 unseen functions without additional compilation steps. By leveraging lossless IR text and auxiliary tasks like instruction count prediction, it rivals expensive autotuners and marks a significant advancement in compiler optimization, reducing computational costs and enhancing efficiency for real-world applications.

---

## 1. Executive Summary
This paper trains a 7B-parameter large language model (LLM) from scratch to optimize LLVM intermediate representation (LLVM-IR) for code size by predicting a sequence of compiler optimization passes (“pass ordering”). It achieves a single-compile 3.01% improvement over LLVM’s strong `-Oz` baseline on 100,000 unseen functions (Table III), while requiring zero exploratory compilations at inference time, and it approaches many of the gains of an expensive autotuner.

The approach is significant because prior machine learning methods either require thousands of compilations per program or operate on lossy program representations; here, a text-only LLM consumes full, lossless IR, and auxiliary training tasks (predicting instruction counts and the optimized IR) substantially improve both reasoning and optimization quality (Section V-B, Table VI).

## 2. Context and Motivation
- Problem addressed
  - The paper tackles `compiler pass ordering` for LLVM: given an unoptimized program in LLVM-IR, choose an ordered list of optimization passes that best reduces code size (Section II). A `pass` is a compiler transformation (e.g., `instcombine`, `jump-threading`) and different orderings can dramatically change results. The goal metric is IR instruction count, used as a practical proxy for binary size (Section II).
- Why it matters
  - Better pass orderings can reduce code size and runtime significantly in real systems; however, finding good orders is a combinatorial problem (“around 10^18” possibilities because passes can repeat and sequences vary in length; Section II-A).
  - Traditional iterative or ML-guided search often needs many compilations per program (e.g., tens of thousands), which is too costly for production builds (Section II; Section III-B).
- Prior approaches and their shortcomings
  - Hand-crafted feature models (e.g., MLGO for inlining) and graph neural networks like ProGraML provide partial program views; they typically drop information (e.g., exact constants, some types), which hinders fidelity (Introduction).
  - Search-based and RL baselines (e.g., AutoPhase, Coreset-NVP) can find good cases but generalize poorly and cause regressions, and they still require many compilation trials per input (Table III).
- Positioning relative to existing work
  - This work introduces LLMs for compiler optimization, using full, lossless IR text as both input and auxiliary output. The model predicts a pass list in one shot. During deployment only the pass list is used, so correctness is guaranteed by running the real compiler once with that list (Section II-A, Figure 1).

## 3. Technical Approach
Step-by-step pipeline (Figure 1; Sections II–III):
1. Problem framing
   - Input: normalized, unoptimized LLVM-IR for a single function.
   - Output at training time (“Answer”): 
     - a pass list (sequence of optimization passes, including meta-flags like `-Oz`),
     - the input and output instruction counts,
     - the optimized IR after applying that pass list (Section II-A; Figure 1).
   - Output at inference time (deployment): only the pass list. The compiler then applies this list to produce the optimized code, ensuring correctness (Figure 1, right).
2. Pass space
   - 122 optimization passes plus six meta-level flags (`-O0`, `-O1`, `-O2`, `-O3`, `-Os`, `-Oz`). Passes can repeat; meta-flags can appear at most once. Typical lists are up to nine passes long (Section II-A).
3. Data creation (Section III-B; Table I)
   - 1,000,000 LLVM-IR functions (610k “handwritten,” 389k “synthetic” from compiler test generators). Data is per-function to fit the context window (Section III-A).
   - For each function, an `autotuner` searches pass lists to find the best for instruction-count reduction. Method:
     - Run random search for 780 seconds per function.
     - Minimize the best found sequence by removing non-contributing passes.
     - Broadcast unique best lists across all other functions to try them broadly (Section III-B).
   - Cost/quality of autotuning used for supervision:
     - Average 37,424 compilations per function; 5.8% better than `-Oz`; 9,016 CPU-days total (Section III-B). This becomes the “gold” training target but is not needed at inference.
4. IR normalization (Section II-B)
   - Strip comments, debug metadata, attributes; canonicalize whitespace via a custom lexer (keeps newlines). This reduces token length to fit the context window.
5. Model architecture and training (Section III)
   - Architecture: LLaMA 2–style 7B transformer (32 layers, 32 heads, 4096 hidden dim), trained from scratch; LLaMA 2 BPE tokenizer (Section III-A).
   - Context window: 2,048 tokens. With LLVM-IR averaging ~2.02 characters/token, training pairs are constrained to ~2 KB prompt + ~2 KB answer (Section III-A).
   - Optimization: AdamW, cosine schedule, 15.7B training tokens over 30k steps on 64×V100 GPUs (620 GPU-days). Validation every 250 steps on 1,000 held-out IRs (Section III-C).
   - Decoding: greedy sampling for the output text (Section III).
6. Learning strategy and why it helps
   - Multi-task outputs (pass list + counts + optimized IR) force semantic understanding of IR and the consequences of passes (Section II-A; Figure 1). Ablations (Section V-B, Table VI) show removing the optimized-IR generation task reduces downstream pass-ordering performance by 16%.
7. Inference
   - Given an unseen function’s unoptimized IR, the model emits a pass list in a single forward pass. The compiler then applies those passes once. No iterative search or multiple compilations are needed (Figure 1; Table III “Our Approach” shows 0 additional compilations).

Analogy for clarity
- Think of the autotuner as an expensive teacher that solves many riddles (find best pass orders) by trial-and-error. The LLM is a student that watches millions of worked examples (IR → good pass list + what changes) and learns to answer new riddles directly, often choosing the same or similar solutions without redoing the trial-and-error.

## 4. Key Insights and Innovations
- LLMs can perform pass ordering directly from full IR text
  - Novelty: Prior ML systems relied on handcrafted features or partial graph encodings; this model consumes the complete, lossless IR and outputs a pass list, achieving 3.01% improvement over `-Oz` with zero exploratory compilations on 100k unseen functions (Table III). This captures 60% of an autotuner’s gains (5.03%) without its 2.5B compilations (Table III).
  - Significance: Demonstrates that a general-purpose sequence model can learn a complex compiler heuristic purely from examples, scaling beyond bespoke feature engineering.
- Auxiliary co-training improves optimization ability
  - What’s new: During training, the model must also predict pre/post instruction counts and generate the optimized IR (Figure 1). 
  - Evidence: Removing the optimized-IR generation task drops overall improvement by 16% on held-out validation (4.95% → 4.15%; Table VI, “No Aux”). The model also learns to produce optimized IR that compiles 90.5% of the time and matches the compiler exactly 70% on validation (Figure 2c).
  - Why it matters: These auxiliary tasks force the model to internalize IR semantics and pass effects, improving the core decision (pass ordering).
- One-shot prediction with guaranteed correctness at deploy time
  - Novelty: Only the pass list is used at inference, and the compiler executes it, so the final code is correct by construction (Figure 1). This sidesteps correctness concerns that arise when deploying model-generated code directly.
  - Trade-off: A single compilation per input is still required to apply the predicted passes, but no search is needed.
- Evidence that LLMs learn specific compiler optimizations
  - New capability: A separate “single-pass translation” experiment trains the model to apply individual passes to IR. Average BLEU 0.846, exact match 73.7%, and 82.3% compile rate across 60 passes (Figure 7). This suggests the model learns nontrivial transformations like control-flow simplification and instruction combining (Listing 7).

## 5. Experimental Analysis
- Evaluation setup
  - Datasets: 100,000 deduplicated functions across six sources (Table II): AI-SOCO (coding competition), ExeBench (executable C functions), POJ-104, Transcoder, CSmith (random C), YARPGen (random C/C++), with 1.716M unoptimized instructions total, 645k after `-Oz`.
  - Metrics: 
    - Main: percent improvement in instruction count vs `-Oz` (“Improvement over `-Oz`”). 
    - Secondary: number of functions improved/regressed, total instructions saved/regressed (Table III). 
    - For generated IR quality: BLEU, compile success, exact character match (Figures 2c, 6; Table V).
  - Baselines (Section IV-B):
    - `Autotuner`: expensive random search + minimization + cross-broadcasting; strongest but costly.
    - `AutoPhase`: PPO agent over a 56D feature vector; 45-step action episodes.
    - `Coreset-NVP`: ProGraML-based GNN cost model; tries top sequences within a 45-attempt budget.
- Main results (Table III; Figures 2–5)
  - Pass-ordering quality
    - On the 100k-function test: 
      > Our approach: 3.01% overall improvement over `-Oz`, with 4,136 improved and 526 regressed functions; 21,935 instructions saved and 3,095 regressed; 0 additional compilations (Table III).
    - The autotuner does better but at massive cost:
      > Autotuner: 5.03% with 2,522,253,069 additional compilations (949 CPU-days just for test time; Table III).
    - RL and GNN baselines overfit and regress on average:
      > AutoPhase: −3.85%; Coreset-NVP: −1.88% (Table III).
    - “`-Oz` backup” rescue:
      - If, whenever a model predicts something other than `-Oz`, you also run `-Oz` and pick the better of the two, regressions disappear and net improvements rise. 
      > With backup: our approach improves to 3.52% with only 5,721 extra compilations; AutoPhase and Coreset-NVP become positive (1.02% and 2.55%) but remain behind the LLM (Table IV).
  - Learning dynamics and validation (Figure 2)
    - The model reaches parity with `-Oz` quickly (≈393M tokens) and peaks at ≈10.9B tokens with 4.4% validation improvement (Figure 2a).
    - Count prediction: input-count estimates become near-perfect; output-count MAPE stabilizes around 5.9% (Figure 2b).
    - Generated IR quality on validation: BLEU 0.952; 90.5% compiles; 70% exact match with the compiler’s output (Figure 2c). Copying the input IR would give BLEU 0.531 and 0% exact match, so the model performs real transformations.
  - Generalization and behavior of pass lists (Figure 3)
    - Pass frequencies track the autotuner’s distribution; `-Oz` is most common (model predicts it 94.3% of the time). Excluding `-Oz`, model lists average 3.4 passes (max 10), similar to autotuner (3.1, max 9). Notably, “105 of the pass lists generated by the model never appear in the training data,” showing some extrapolation (Figure 3).
    - In 710 test cases, the model’s pass list beats the autotuner’s (Listing 1 shows an example with control-flow simplification of fewer blocks).
  - Where improvements are larger (Figures 4 and 5)
    - Human-written code (POJ-104, Transcoder) offers more opportunities than random generators like YARPGen (Figure 4).
    - Larger functions yield larger improvements over `-Oz` (Figure 5).
  - Quality and failure modes of generated IR (Section IV-D; Table V; Listings 2–5; Figure 6)
    - On the 100k test functions (when asking the model to also generate IR): 
      > 90.3% compile; 68.4% exact match with compiler output (Section IV-D).
    - Error taxonomy for the 9.7% non-compiling cases: type errors (5,777), forward references (1,521), undefined values (1,113), redefinitions (616), syntax (280), invalid constants (144), etc. (Table V).
    - Illustrative failures:
      - Numeric reasoning error on constant folding (Listing 3).
      - Producing a correct optimized IR but missing a necessary pass (e.g., `-mem2reg`) in the predicted list (Listing 4).
      - Unsafe optimization (removing a loop that may not terminate), reminding why the deployed system uses only pass lists and lets the compiler enforce correctness (Listing 5).
    - Correlation: when the model’s pass list is worse than `-Oz`, its generated IR resembles the ground truth less (lower BLEU), compiles less frequently, and count prediction errors rise (Figure 6).
  - Ablations and single-pass learning (Section V; Figure 8; Table VI; Figure 7)
    - Dataset size matters: training on 50% and 25% of data reduces improvement by 21% and 24% respectively (Table VI; Figure 8 shows earlier overfitting).
    - Auxiliary task matters: removing optimized-IR generation reduces improvement by 16% (Table VI; Figure 8).
    - Single-pass translation: many passes are learned almost perfectly; others (e.g., `-instcombine`) expose data-flow reasoning limits; some failures stem from missing context (e.g., `-name-anon-globals` needs a module name to hash; Listing 6a–b). Still, average BLEU 0.846, 82.3% compile, and 73.7% exact match (Figure 7).

Do the experiments support the claims?
- Yes, in scope. The LLM consistently outperforms `-Oz` and prior ML baselines on large, diverse test data (Table III), while requiring zero compile-time exploration. Validation and ablations justify the design choices (Figure 2; Figure 8; Table VI), and the single-pass study shows the model learns concrete transformations (Figure 7). Limitations are transparently discussed and evidenced by failure cases (Section VI; Listings 2–6).

## 6. Limitations and Trade-offs
- Context window and scope
  - The model uses a 2,048-token context window and operates per function (Section III-A; VI-A). This:
    - prevents whole-program reasoning (e.g., cross-function inlining decisions),
    - constrains the size/complexity of code processed,
    - may miss optimizations needing global context.
- Optimization objective and proxy
  - The target is code size, measured by IR instruction count, not runtime performance (Section II). Instruction count is an imperfect proxy for binary size or speed; extending to performance optimization remains future work.
- Reasoning gaps
  - Arithmetic and data-flow logic sometimes fail (constant folding, precise value propagation), documented in Listings 3 and 6b and discussed in Section VI-B. Output-count prediction error remains ~5.9% MAPE at validation peak (Figure 2b).
- Correctness and safety of generated IR
  - While the model can generate IR, the deployed system does not rely on it; it only uses the pass list and lets the compiler ensure correctness (Figure 1). Generated IR can be wrong or unsafe (Listing 5), which underscores the importance of this deployment choice.
- Computational costs
  - Training is heavy (620 GPU-days; Section III-C). Inference per input is slower than running the chosen pass list itself (Section VI-C), though it is vastly cheaper than autotuning. The approach also depends on a large supervised dataset produced by expensive autotuning (Section III-B).
- Passes requiring external context
  - Some passes depend on information not present in function IR (e.g., module names for hashing in `-name-anon-globals`; Listing 6a), leading to unavoidable hallucinations unless the prompt includes more context.
- Reliance on LLVM version and pass set
  - The system targets LLVM 10 and a specific pass set (Section II-A). Portability to newer compiler versions or different compilers may require re-collection of training data and re-training.

## 7. Implications and Future Directions
- How this changes the landscape
  - The work demonstrates that a single, general-purpose LLM can learn pragmatic compiler decision-making from examples, rivaling specialized ML systems while eliminating costly search at inference. This suggests a path toward LLM-augmented compilers that use rich textual IR context instead of narrow, handcrafted features.
- Follow-up research enabled
  - Long-context and whole-program optimization: incorporate longer context windows or hierarchical representations to capture interprocedural opportunities (Section VI-A).
  - Better reasoning: apply chain-of-thought, tool use (e.g., calling solvers for arithmetic), or verifier-guided decoding to address data-flow and constant-folding errors (Section VI-B).
  - Broader objectives: extend beyond code size to runtime performance, memory, or energy; combine pass ordering with algorithmic transformations.
  - Active and curriculum learning: emphasize difficult passes (e.g., `-instcombine`) and arithmetic curricula; include module-level metadata to support context-dependent passes (Listings 6a–b).
  - Robustness and safeguards: integrate equivalence checking or differential testing for any pipeline that wants to deploy model-generated code directly, and explore structured decoding that respects IR grammar.
  - Practical integration: 
    - “`-Oz` backup” shows a low-cost guardrail that improves net gains with few extra compilations (Table IV).
    - Compiler-side tokenization or phrase-level BPE for common pass sequences may also improve inference speed (Section VI-C).
- Applications
  - Fast, reliable pass-order selection in production builds without search.
  - Offline tuning: quickly evaluate thousands of functions to suggest near-optimal orders before a build system adopts them.
  - Educational and debugging aid: generate optimized IR with explanations to help compiler engineers explore transformations.

Key takeaway in the paper’s own numbers:
> On 100,000 unseen functions, a 7B LLM trained on 1M pass-ordered examples improves over `-Oz` by 3.01% with zero additional compilations, achieving about 60% of a costly autotuner’s gains (5.03%) that required 2.5 billion compilations (Table III), and co-training on instruction counts and optimized IR is crucial to this performance (Table VI).
