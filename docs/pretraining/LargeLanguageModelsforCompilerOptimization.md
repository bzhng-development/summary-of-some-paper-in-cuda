# Large Language Models for Compiler Optimization

**ArXiv:** [2309.07062](https://arxiv.org/abs/2309.07062)

## 🎯 Pitch

This paper introduces the first large language model (LLM) specifically trained for code optimization, leveraging a 7B-parameter transformer to predict LLVM compiler pass sequences that reduce code size. By operating directly on unoptimized LLVM-IR and requiring zero extra compilations at inference, this model achieves a 3% improvement over LLVM's own size-optimized baseline and outperforms state-of-the-art machine learning baselines that demand orders of magnitude more computational effort. This breakthrough demonstrates the feasibility and power of applying LLMs to automate and enhance compiler optimization, opening new possibilities for faster, leaner, and more intelligent software builds across the computing industry.

---

## 1. Executive Summary
This paper trains a 7B-parameter transformer to optimize compiler settings: given unoptimized LLVM intermediate representation (LLVM-IR), the model predicts a sequence of optimization passes that reduces code size. On 100,000 unseen functions, it improves instruction count by 3.01% over LLVM’s size-optimized baseline `-Oz` with zero extra compilations, outperforming two state-of-the-art ML baselines that require thousands of compilations per program (Table III), and achieves high-quality auxiliary behaviors such as generating compilable optimized IR 90–91% of the time (Figure 2c).

## 2. Context and Motivation
- Problem addressed
  - Traditional compilers apply many optimization passes (each pass is a transformation such as constant folding or control-flow simplification). The order of passes greatly influences the final code quality; picking the best order per program is the “pass ordering” or “phase ordering” problem.
  - The paper targets predicting good pass orders for reducing code size, using LLVM’s IR as input and LLVM’s pass flags as output (Section II; Figure 1).
- Why it matters
  - Pass ordering affects runtime performance and binary size; better ordering yields leaner and faster code (Section II; supported by [19, 26] in Related Work).
  - Prior “autotuning” methods can find strong orderings but at prohibitive cost (thousands to billions of compilations), making them impractical for production build pipelines (Section III-B; Table III).
- Prior approaches and gaps
  - Earlier ML methods represent programs with hand-crafted features or graphs (e.g., MLGO, ProGraML; Section I), which lose information and still need many compilations for search or validation.
  - LLMs used for code have focused on generation or translation of source code, not optimization; they also struggle with IR-level reasoning out-of-the-box (Section II).
- Positioning
  - This work treats optimization as text-to-text learning: input is normalized IR text; output is a pass list and (during training) predicted instruction counts and the optimized IR (Figure 1). It aims to replace expensive search with a single prediction and to keep correctness by running the predicted pass list inside LLVM at inference time.

Key terms used here
- `LLVM-IR`: a low-level, language- and target-agnostic assembly-like representation used by LLVM. It exposes the precise operations and control flow compilers reason about.
- `Pass ordering`: selecting and sequencing optimization passes (e.g., `-instcombine`, `-simplifycfg`) to improve code.
- `-Oz`: LLVM’s built-in optimization level that prioritizes minimizing code size.
- `Autotuner`: a search procedure that compiles a program many times with different pass lists to find the best one.

## 3. Technical Approach
Step-by-step methodology (Figure 1; Sections II–III):
1. Data construction with “ground truth” pass lists
   - For each function-level LLVM-IR sample, a custom autotuner searches pass sequences that minimize instruction count (a proxy for code size). The search combines random exploration, minimization of redundant passes, and “all-to-all broadcasting” of promising pass lists across functions (Section III-B).
     - Search budget: 780 seconds per function; average 37,424 compilations per training program; about 9,016 CPU-days to produce labels (Section III-B).
     - Outcome: on training programs, the autotuner achieves 5.8% instruction-count reduction over `-Oz` (Section III-B).

2. Input/output format and auxiliary tasks
   - Input (prompt): a single LLVM-IR function, normalized to remove comments/metadata and standardize whitespace for token efficiency (Section II-B).
   - Output (answer) during training (Figure 1):
     - A pass list (sequence chosen among 122 passes plus 6 meta-level flags: `-O0/-O1/-O2/-O3/-Oz/-Os`; Section II-A).
     - Predicted instruction counts before and after optimization.
     - The optimized LLVM-IR text resulting from applying the pass list.
   - Inference-time behavior: only the pass list is generated; the compiler applies it to ensure correctness (Figure 1, “Inference Phase”). This sidesteps correctness issues in directly generated code (end of Section II-A).

3. Search space and representation choices
   - Order matters; passes can repeat; typical predicted sequences are up to length ~9, yielding an enormous combinatorial space (~10^18, Section II-A).
   - Operating at IR text avoids lossy features/graphs and keeps a “lossless” representation, enabling the model to learn optimization-relevant patterns directly (Introduction).

4. Model and training
   - Architecture: Llama 2–style 7B transformer (32 layers, 32 attention heads, hidden size 4096) with Llama 2 BPE tokenizer, context length 2048 tokens (Section III-A).
   - Training data: 1,000,000 function IRs (610k “handwritten” from public C/C++ sources + 389k synthetic from test generators), totaling 373M tokens and ~1 GB IR (Table I). Functions are used (not whole modules) to fit the context window (Section III-B).
   - Token efficiency: LLVM-IR averages ~2.02 characters per token, and the 2048 token limit effectively allows ~2KB prompt + ~2KB answer (Section III-A).
   - Optimization and schedule: trained from scratch for 30,000 steps on 64 V100 GPUs (620 GPU days), with AdamW, a cosine learning rate, peak LR 1e−5, 15.7B training tokens, and evaluations every 250 steps on a 1,000-function holdout set (Section III-C). Peak validation performance occurs around 10.9B tokens (Figure 2).

5. Why the multi-task setup?
   - Forcing the model to predict before/after instruction counts and the resulting optimized IR aims to internalize the mechanics of optimizations (Section II-A). An ablation (“No Aux”) shows a 16% drop in downstream performance when the model is not trained to generate optimized IR (Figure 8; Table VI).

Analogy for clarity: Think of the model as a “meta-compiler coach.” It reads the raw IR (the “game plan”), explains what the score is now and could be after certain plays (instruction counts), sketches what the field would look like after those plays (optimized IR), and recommends the ordered list of plays (pass sequence). At deployment, only the recommended plays are actually executed by the real team (LLVM), ensuring a valid outcome.

## 4. Key Insights and Innovations
- LLMs can optimize compiler pass ordering with a single prediction
  - Fundamental innovation: text-in/text-out learning on LLVM-IR for optimization decisions, not just code generation. The model achieves a 3.01% instruction-count reduction over `-Oz` on 100k unseen functions with zero extra compilations (Table III), outperforming RL and GNN-based baselines that regress overall due to many bad suggestions.
  - Significance: removes the need for expensive multi-compile search in many cases, enabling “single-compile” optimization.

- Multi-task supervision (predicting counts and optimized IR) materially improves pass list prediction
  - Evidence: removing the auxiliary task of generating optimized IR reduces performance by 16% on the validation set (Figure 8; Table VI). This suggests the model benefits from learning the semantics of transformations, not just mapping IR-to-pass-list.

- Generalization across passes and programs, including novel pass lists
  - The learned distribution of passes closely tracks the autotuner’s usage (Figure 3), yet the model also proposes 105 pass lists not seen in training and beats the autotuner in 710 test cases (albeit by small margins; Listing 1). This indicates pattern learning beyond memorization.

- First pass-level “translation” capability at scale
  - As an auxiliary study, the paper trains a model to emulate single passes (input: IR + pass name; output: IR after that pass). Average BLEU 0.846, exact match 73.7%, and 82.3% compile rate across 60 passes (Figure 7). This shows LLMs can learn many concrete compiler transformations directly from examples.

Incremental vs. fundamental:
- Fundamental: the idea and demonstration that an LLM can make effective optimization decisions directly from IR text with multi-task learning; the single-pass translation capability for dozens of passes.
- Incremental: architectural choices (using Llama 2 7B) and standard training strategies; the autotuning setup for label generation is adapted from prior search techniques.

## 5. Experimental Analysis
- Evaluation setup
  - Datasets: 100,000 deduplicated test functions from six sources (AI-SOCO, ExeBench, POJ-104, Transcoder, CSmith, YARPGen), covering coding competition code, open-source C functions, and random compiler stress tests (Table II). The space varies in structure and difficulty.
  - Metric: instruction count (proxy for code size). Improvements are reported relative to `-Oz` (Sections II and IV).
  - Baselines (Section IV-B):
    - `Autotuner`: strong but very expensive search (as in training label creation).
    - `AutoPhase` (deep RL) and `Coreset-NVP` (graph-based value predictor + search). Both try up to 45 pass sequences per program.
  - The LLM predicts once per program; the predicted pass list is then executed by LLVM.

- Main results
  - Overall performance (Table III):
    > Autotuner: 5.03% improvement over `-Oz`, but requires 2,522,253,069 additional compilations (949 CPU-days).
    > Our Approach (LLM): 3.01% improvement, 0 additional compilations; 4,136 functions improved, 526 regressed.
    > AutoPhase: −3.85% overall (many regressions) with 4.5M compilations.
    > Coreset-NVP: −1.88% overall with 442,747 compilations.
  - Adding a simple safety net (“`-Oz` backup”: also compile with `-Oz` and pick the better of the two) removes regressions and yields (Table IV):
    > LLM: 3.52% improvement with only 5,721 extra compilations (one extra per non-`-Oz` suggestion).
    > AutoPhase: 1.02% with 4.6M extra compilations.
    > Coreset-NVP: 2.55% with 542,747 extra compilations.

- Training dynamics and auxiliary behaviors
  - The validation curve reaches parity with `-Oz` around 393M training tokens and peaks near 10.9B (Figure 2a).
  - The model predicts instruction counts for unoptimized code with near-perfect accuracy and for optimized code with ~5.9% MAPE at peak (Figure 2b).
  - When asked to also emit optimized IR, it compiles 90.5% of the time, with BLEU 0.952 vs. compiler output and exact character match 70% at peak (Figure 2c). These are not used in deployment but indicate strong semantic learning.

- Where gains happen
  - Dataset breakdown (Figure 4): larger improvements on POJ-104 and Transcoder (handwritten code), smaller on YARPGen (random code). This suggests “human-like” structures present more predictable optimization opportunities.
  - Larger functions benefit more (Figure 5): both the autotuner and the LLM find bigger improvements as unoptimized instruction count increases.

- Pass usage and generalization
  - Pass frequency distribution of the LLM broadly matches the autotuner (Figure 3). Excluding `-Oz`, average pass list length is 3.4 for the LLM (max 10) vs. 3.1 for the autotuner (max 9).
  - The LLM sometimes surpasses the autotuner; Listing 1 shows a case where control-flow simplification yields an additional instruction saved vs. the autotuner’s best.

- Failure analysis and robustness
  - Compilation errors for generated IR (only for the auxiliary task; deployment avoids using generated IR): 9.7% fail. The main categories are type errors, forward-referenced instructions, undefined values, and a few syntax issues (Table V; Listing 2).
  - Mathematical reasoning is a weak spot. For instance, constant folding of a 64-bit literal to an 8-bit value is wrong in Listing 3. Output count prediction is also harder than input count prediction (Figure 2b).
  - Sometimes the model generates the correct optimized IR but predicts an incomplete pass list (Listing 4), highlighting a mismatch between “can do” and “says which passes to do.”
  - Unsafe transformations can occur when emitting IR directly (Listing 5). This is why the main deployment path runs LLVM on the predicted pass list, not the model’s IR.
  - Quality correlates with pass-list effectiveness: when the predicted pass list performs worse than `-Oz`, generated IR quality (BLEU, compile rate) also drops and count prediction errors rise (Figure 6).

- Ablations and auxiliary experiments
  - Data size matters: reducing training data to 50% and 25% causes 21% and 24% performance drops, respectively, with overfitting signs around 8B tokens (Figure 8; Table VI).
  - Single-pass translation study: high average quality (Figure 7), but certain passes fail for specific reasons (e.g., `-name-anon-globals` needs module name to compute a hash; Listing 6a). Complex passes like `-instcombine` reveal data-flow mistakes (Listing 6b). A correct complex example is shown in Listing 7.

- Do the experiments support the claims?
  - Yes for the main claim: “single-compile” improvements over `-Oz` that beat strong ML baselines while avoiding their regressions and large search costs (Table III and IV).
  - The auxiliary results convincingly show the model learns substantive IR semantics (Figure 2c; Figure 7), but the paper does not rely on generated IR for correctness at deployment.

## 6. Limitations and Trade-offs
- Context window and granularity
  - Operations occur at the function level due to the 2048-token context limit (Section VI-A). This:
    - Limits interprocedural optimization (cross-function reasoning).
    - Excludes very large functions or modules.
    - Forces reliance on function-level pass effects.
- Objective and proxy metric
  - The work targets code size (instruction count) rather than runtime performance (Section II). Instruction count is an imperfect proxy for binary size and does not capture runtime gains or losses.
- Dependence on expensive labels
  - Training labels come from a heavy autotuning process (9,016 CPU-days; Section III-B). This is a one-time offline cost, but practical adoption would need either reusable labels or cheaper label acquisition strategies.
- Arithmetic and logical reasoning
  - The model struggles with constant folding and data-flow subtleties (Listings 3 and 6b; Section VI-B). Output instruction count prediction is also harder (Figure 2b).
- Inference speed and compute
  - Generating a pass list is two orders of magnitude slower than LLVM executing it, and requires GPU resources (Section VI-C). Though still far cheaper than autotuning, latency may matter in continuous integration pipelines.
- Safety of directly generated IR
  - Direct IR generation can be incorrect or unsafe (Listings 2 and 5). The paper mitigates this by only using the pass list in deployment, but any attempt to rely on generated IR would need verification.

## 7. Implications and Future Directions
- Field impact
  - This work reframes compiler optimization as a text understanding and decision-making task, showing that an LLM can absorb nontrivial IR semantics and deliver practical single-compile gains. It opens a path to replacing or accelerating heuristic and search-heavy optimization strategies.
- Practical applications
  - Compiler toolchains can call an LLM once per function to propose a tailored pass list, optionally with an `-Oz` fallback (Table IV) to guarantee no regressions. This is valuable for:
    - Size-constrained environments (embedded, mobile).
    - Continuous integration systems aiming to cut build artifacts without lengthy autotuning.
    - Rapid design-space exploration in code size sensitive libraries.
- Research directions
  - Longer context: adopt long-context transformers or positional scaling to operate on whole modules and enable interprocedural optimization (Section VI-A; refs [45–47]).
  - Stronger reasoning: incorporate chain-of-thought, tool-use (symbolic math/solvers), and curricula focused on arithmetic and data-flow to improve constant folding and value reasoning (Section VI-B; refs [48–50]).
  - Active learning and pass-aware training: upsample complex or frequently beneficial passes (e.g., `-instcombine`) to reduce specific failure modes (Section V-C).
  - Runtime metrics: extend the approach to runtime performance, not only instruction count, including multi-objective trade-offs (Section II).
  - Fast inference: quantization, batching, and vocabulary specialization (e.g., tokenizing common pass subsequences as single tokens) to reduce compute and latency (Section VI-C).
  - Hybrid strategies: combine the LLM’s first guess with a tiny local search over a few alternatives (including `-Oz`), exploiting the model’s strong prior while preserving robustness (Table IV).

Overall, the paper demonstrates a new capability—LLMs that act as competent “optimization planners” for compilers—backed by large-scale experiments showing consistent gains over a strong default with minimal overhead, and identifies clear paths to lift remaining constraints around context, reasoning, and speed.
