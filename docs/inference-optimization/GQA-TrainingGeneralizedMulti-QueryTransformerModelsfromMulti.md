# GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints

**ArXiv:** [2305.13245](https://arxiv.org/abs/2305.13245)

## 🎯 Pitch

This paper introduces a novel, low-cost uptraining method to convert existing multi-head attention (MHA) language model checkpoints into faster multi-query attention (MQA) and a new grouped-query attention (GQA) scheme—requiring only about 5% of the original pre-training compute. GQA smartly shares keys and values among small groups of query heads, dramatically reducing inference latency and memory overhead while nearly matching the output quality of full MHA models—making it a practical breakthrough for deploying high-performance, efficient large language models.

---

## 1. Executive Summary (2-3 sentences)
This paper tackles slow autoregressive decoding in Transformer-based language models by reducing the memory traffic required to load attention keys and values at each step. It introduces two practical methods: (1) a low-cost “uptraining” recipe that converts existing multi-head attention (MHA) checkpoints into multi-query attention (MQA), and (2) “grouped-query attention” (`GQA`), which shares keys/values across small groups of query heads. With only ~5% of the original pre-training compute, uptrained `GQA` reaches nearly the quality of full `MHA` while being almost as fast as `MQA` (Figure 3; Table 1).

## 2. Context and Motivation
- Problem addressed:
  - Autoregressive decoding is bottlenecked by memory bandwidth: at each generated token, the model must repeatedly load all attention keys and values (“KV cache”) from memory (Introduction; Section 1).
  - `MQA`—using a single shared key head and a single shared value head for all query heads—reduces this KV memory traffic but can hurt quality and be unstable to train (Abstract; Section 1; Appendix A).

- Why this matters:
  - Inference cost dominates production deployment of large language models. Reducing KV memory traffic directly speeds up decoding and lowers serving cost, especially for long outputs (Section 1; Related Work).
  - Many strong public checkpoints (e.g., T5, LLaMA) are trained with `MHA` and therefore inherit the KV-bandwidth bottleneck (Section 1).

- Prior approaches and gaps:
  - `MQA` (Shazeer, 2019) is known to speed up decoding but can degrade quality and be unstable (Section 1; Appendix A).
  - Other efficiency ideas—FlashAttention, quantization, distillation, layer-sparsity, speculative decoding—address different parts of the compute/memory stack; they do not directly trade KV bandwidth for model capacity in the way `MQA/GQA` do (Related Work).

- Positioning of this work:
  - Provides a recipe to “uptrain” existing `MHA` checkpoints into `MQA` or `GQA` using a small compute fraction, avoiding full retraining (Section 2.1).
  - Introduces `GQA`, an interpolation between `MHA` and `MQA`, to recover most of the quality of `MHA` with near-`MQA` speed (Section 2.2; Figure 2).
  - Demonstrates this on T5.1.1 Large/XXL models across summarization, translation, and QA (Section 3; Table 1).

## 3. Technical Approach
Step-by-step overview of what is changed and how it works.

- Background: the KV cache
  - During decoding, each new token attends to all previous tokens. To avoid recomputing attention features, models store keys and values (“KV”) for past tokens. Loading these KVs every step dominates memory bandwidth—especially harmful on accelerators (Section 1).
  - In `MHA` with `H` heads, there are `H` distinct key and value projections; the KV cache and memory traffic scale with `H` (Section 2.2).

- Idea 1: Multi-Query Attention (`MQA`)
  - Mechanism: keep multiple query heads but share a single key head and a single value head across all queries. This shrinks the KV cache by roughly a factor of `H`, because the number of stored key/value tensors per time step goes from `H` to `1` (Section 1; Figure 2 right).
  - Trade-off: reduced KV capacity can hurt quality and training stability (Abstract; Appendix A).

- Idea 2: Grouped-Query Attention (`GQA`)
  - Mechanism: split the `H` query heads into `G` groups. Each group of queries shares one key head and one value head. Special cases: `GQA-1` equals `MQA`; `GQA-H` equals `MHA` (Section 2.2; Figure 2 center).
  - Effect: the KV cache shrinks by `≈ H/G` relative to `MHA`. Larger `G` improves capacity (quality), smaller `G` improves speed (Section 2.2).
  - Why this helps large models: as models scale, `H` typically increases; moving from `MHA` to `MQA` becomes a more aggressive capacity cut. `GQA` keeps the bandwidth reduction proportional to model size while retaining more capacity than `MQA` (Section 2.2). It also mitigates waste in tensor-parallel sharding where a single MQA KV head would be replicated across partitions (Section 2.2).

- Uptraining recipe: converting checkpoints without full retraining
  - Step 1: checkpoint conversion by mean pooling the key and value projection matrices from the original heads to form the new shared head(s). For `MQA`, average all heads into one; for `GQA`, average within each group (Section 2.1; Figure 1; Figure 2).
    - Why mean pooling: preserves information from all heads better than picking one head or random init (Ablation; Figure 4).
  - Step 2: continued pre-training (“uptraining”) for a small fraction `α` of the original pre-training steps, using the same data and recipe (Section 2.1).
    - The paper uses `α = 0.05` (5%) as the main setting; this required “approximately 600 TPUv3 chip-days” for T5-XXL (Section 3.1, Uptraining).
  - Where applied: decoder self-attention and cross-attention are converted to `MQA`/`GQA`; encoder self-attention remains standard since it runs in parallel and is not the bottleneck (Section 3.1, Configurations; Section 2.2 note).

- Implementation and training details (Section 3.1)
  - Models: T5.1.1 Large and XXL (JAX/Flax/Flaxformer).
  - Optimizer and schedule: Adafactor with T5 hyperparameters.
  - Fine-tuning: constant LR 0.001, batch size 128, dropout 0.1; greedy decoding; input/output lengths depend on the task (Fine-tuning subsection).
  - Timing: per-sample time per TPUv4 chip via xprof; measured on 8 TPUs with largest feasible batch size up to 32 per TPU; parallelization tuned per model (Timing subsection).

- Why these design choices:
  - Mean pooling preserves pre-trained structure, making adaptation easier (Figure 4).
  - Partial uptraining adapts the model to its new attention geometry at a fraction of the original cost (Section 2.1; Figure 5 shows performance improves quickly and saturates around 5–10%).
  - Grouping balances KV efficiency with capacity to avoid the quality and stability issues of pure `MQA` (Section 2.2; Appendix A).

## 4. Key Insights and Innovations
- Low-cost checkpoint “uptraining” from `MHA` to `MQA/GQA` (Section 2.1)
  - What’s new: a simple, reproducible recipe—mean-pool K/V projections then continue pre-training briefly.
  - Why it matters: avoids training a separate fast model from scratch; leverages existing high-quality checkpoints. With `α=0.05`, training cost is modest (“~600 TPUv3 chip-days” for XXL; Section 3.1).

- `GQA`: a controllable middle ground between `MHA` and `MQA` (Section 2.2; Figure 2)
  - What’s new: share K/V within groups of query heads, parameterized by group count `G`.
  - Why it matters: preserves most of the quality of `MHA` while approaching the speed of `MQA`. It scales better for large models (where `MQA`’s capacity cut is too severe) and reduces sharding waste (Section 2.2).

- Empirical recipe choices that stabilize and improve performance (Section 3.3)
  - Mean-pooling K/V projections is best among tested conversion methods (Figure 4).
  - `GQA` requires little or no uptraining to be useful, while `MQA` needs uptraining for good performance (Figure 5).
  - Choosing a moderate number of groups (e.g., 8) yields strong trade-offs with modest overhead over `MQA` (Figure 6).

- Practical evidence of stability benefits with `GQA` (Appendix A)
  - Training `MQA` from scratch showed frequent loss spikes and divergence on long-input tasks; uptrained `MQA` is better but high-variance; uptrained `GQA` appears stable.

## 5. Experimental Analysis
- Evaluation methodology (Section 3.1)
  - Datasets:
    - Summarization: CNN/DailyMail, arXiv, PubMed, MediaSum, Multi-News.
    - Translation: WMT14 En→De.
    - Question answering: TriviaQA.
  - Metrics:
    - ROUGE-1 (“R1”) for summarization, BLEU for WMT14, F1 for TriviaQA (Table 1).
  - Models compared:
    - `MHA-Large`, `MHA-XXL` (baseline T5.1.1 checkpoints).
    - Uptrained `MQA-XXL` and `GQA-8-XXL` with `α=0.05` (Sections 3.1–3.2).
  - Inference timing:
    - Per-sample time per TPUv4 chip via xprof; parallelization optimized per model (Timing subsection).
  - Fine-tuning setup:
    - Constant LR 0.001, batch size 128, dropout 0.1; task-specific input/output lengths; greedy decoding (Fine-tuning subsection).

- Main quantitative results
  - Overall quality vs speed (Figure 3; Table 1):
    - Quote:
      > Table 1 shows `MHA-XXL` average score 47.2 with per-sample time 1.51; `MQA-XXL` average 46.6 with time 0.24; `GQA-8-XXL` average 47.1 with time 0.28.
    - Interpretation:
      - `MQA-XXL` is much faster than `MHA-XXL` with a small quality drop.
      - `GQA-8-XXL` recovers nearly all of `MHA-XXL`’s quality (47.1 vs 47.2) while staying close to `MQA-XXL` in speed (0.28 vs 0.24).
      - Compared to `MHA-Large` (46.0, 0.37), uptrained `MQA-XXL` is both faster and higher-quality.
  - Task-level highlights (Table 1):
    - Summarization (ROUGE-1): `GQA-8-XXL` often matches or exceeds `MHA-XXL` (e.g., MediaSum 47.7 vs 47.5; MultiNews 36.3 vs 36.4).
    - Translation (BLEU): `GQA-8-XXL` 28.4 vs `MHA-XXL` 28.4 (parity).
    - TriviaQA (F1): `GQA-8-XXL` 81.6 vs `MHA-XXL` 81.9 (near parity).
  - Speed-quality frontier (Figure 3):
    - Quote:
      > Figure 3 shows `GQA-8-XXL` sits close to `MHA-XXL` in quality at a time per sample close to `MQA-XXL`, improving the Pareto frontier compared to `MHA-Large` and `MHA-XXL`.

- Ablations and robustness (Section 3.3)
  - Conversion methods (Figure 4):
    - Quote:
      > Mean pooling outperforms selecting a single head and random initialization when converting to `MQA`.
    - Reasonable: mean pooling best preserves information from the original heads.
  - Uptraining budget `α` (Figure 5):
    - Quote:
      > Both `MQA` and `GQA` improve up to ~5% uptraining with diminishing returns by 10%; `GQA` is already reasonable immediately after conversion, whereas `MQA` requires uptraining to be useful.
  - Number of groups (Figure 6):
    - Quote:
      > Increasing `G` from 1 (`MQA`) to 8 adds modest inference overhead for XXL; cost grows more steeply as `G` approaches `H` (`MHA`).
    - Practical choice: `G=8` selected as a good middle ground.
  - Stability (Appendix A):
    - Quote:
      > MQA from scratch had “frequent loss spikes” and diverged on long input fine-tuning; uptrained MQA improved but remained high variance; uptrained GQA appeared stable.

- Do the experiments support the claims?
  - Yes, on the tested tasks and hardware:
    - The speed benefits are clear (Table 1).
    - `GQA-8-XXL` achieves near-`MHA-XXL` quality across diverse tasks while remaining close in speed to `MQA` (Figure 3; Table 1).
    - Ablations justify implementation choices (Figures 4–6) and highlight `GQA`’s stability (Appendix A).
  - Scope:
    - Results are on T5 encoder–decoder models and specific datasets; the paper notes broader applicability (decoder-only models) but does not evaluate them here (Limitations).

## 6. Limitations and Trade-offs
- Assumptions and scope:
  - The main bottleneck is KV memory bandwidth during decoding; gains are most pronounced for longer sequences (Section 1, Limitations).
  - The approach is demonstrated on encoder–decoder T5.1.1 models; not evaluated on decoder-only LLMs, where the authors expect even larger benefits (Limitations).

- Quality vs speed trade-off:
  - `MQA` maximizes KV savings but can degrade quality and be unstable—especially on long-input tasks (Appendix A).
  - `GQA` introduces a tunable parameter `G`:
    - Smaller `G` → faster but lower capacity.
    - Larger `G` → slower but higher capacity.
    - Choosing `G` requires task- and model-size–aware tuning (Figure 6).

- Compute and engineering constraints:
  - Uptraining is much cheaper than full pre-training but still non-trivial (e.g., “~600 TPUv3 chip-days” for XXL at `α=0.05`; Section 3.1).
  - Requires modifying attention implementations and checkpoint conversion tooling (Figures 1–2).

- Evaluation gaps:
  - Summarization metrics (ROUGE) have known limitations for long-form quality; thus, exact quality trade-offs are hard to fully assess (Limitations).
  - No direct comparison to training `GQA` from scratch; unclear whether uptraining reaches the same optimum (Limitations).

- Stability caveat:
  - `MQA` may remain high-variance even after uptraining on certain tasks (Appendix A). `GQA` alleviates this but the root cause of `MQA` instability is not analyzed in depth.

## 7. Implications and Future Directions
- Impact on the field:
  - Establishes a practical path to retrofitting existing `MHA` checkpoints for faster inference without sacrificing much quality.
  - Introduces `GQA` as a general knob for KV bandwidth versus capacity, making attention design more flexible for large-scale deployment.

- Practical applications:
  - Production LLM serving where latency/cost are dominated by decoding:
    - Long-form generation (summarization, code generation, multi-turn dialogue).
    - Multilingual translation systems with long outputs.
  - Model distillation or cascades where faster models are desired without retraining from scratch.

- Research directions enabled:
  - Decoder-only models: validate the expected stronger advantage of `GQA` when there is no separate cross-attention (Limitations).
  - Adaptive grouping: dynamically choose `G` per layer, head, or input length to optimize the quality-speed frontier.
  - Training stability: analyze and mitigate `MQA`’s instability (Appendix A)—e.g., via regularizers, initialization schemes beyond mean pooling, or curriculum strategies.
  - Systems co-design:
    - Combine `GQA` with FlashAttention and quantization to compound benefits.
    - Explore partitioning strategies that exploit `GQA` to reduce KV replication across shards (Section 2.2).
  - From-scratch training comparisons: quantify whether uptraining matches or trails models trained with `GQA` from initialization (Limitations).

Key citations to the paper’s content:
- Mechanisms and conversion: Figure 1 (MHA→MQA conversion), Figure 2 (`GQA` overview).
- Main results and speed-quality trade-off: Figure 3; Table 1.
- Ablations: Figure 4 (conversion methods), Figure 5 (uptraining proportion), Figure 6 (groups vs time).
- Stability: Appendix A.
- Setup details: Section 3.1 (Configurations, Uptraining, Data, Fine-tuning, Timing).
