# GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints

**ArXiv:** [2305.13245](https://arxiv.org/abs/2305.13245)

## 🎯 Pitch

This paper introduces a novel uptraining recipe that efficiently converts existing multi-head attention (MHA) Transformer models into either fast multi-query attention (MQA) or the newly proposed grouped-query attention (GQA) models using just 5% of the original pretraining compute. The key innovation, GQA, recovers nearly all the accuracy lost in MQA while retaining its dramatic inference speed advantage, enabling legacy models like T5 or LLaMA to achieve faster, more cost-effective decoding without sacrificing quality—a breakthrough for production-scale language model deployment.

---

## 1. Executive Summary (2-3 sentences)
This paper introduces a practical way to convert existing Transformer language models that use standard multi-head attention (`MHA`) into faster models using either multi-query attention (`MQA`) or a new middle ground called grouped-query attention (`GQA`). With a simple “checkpoint conversion + brief uptraining” recipe (about 5% of original pretraining compute), the converted models achieve nearly the same quality as the originals while cutting decoder inference time dramatically; in particular, `GQA` recovers most of the quality lost by `MQA` while retaining almost all of its speed (Table 1, Figure 3).

## 2. Context and Motivation
- Problem addressed:
  - Autoregressive decoding in large Transformers is bottlenecked by memory bandwidth: at every generated token, the decoder must load large “key” and “value” tensors for attention from the key–value cache (`KV-cache`) and the layer weights from memory (Section 1). This is especially costly for long inputs and long generations.
- Why it matters:
  - Faster decoding directly lowers serving latency and cost for real-world applications like chat assistants, summarization, translation, and QA.
  - Memory bandwidth dominates runtime more than raw FLOPs in many deployed settings (Roofline perspective cited in Section 4; also Section 1).
- Prior approaches and their shortcomings:
  - `MQA` (one key head and one value head shared across all query heads) sharply reduces KV-cache size and memory reads (Shazeer, 2019), but can degrade quality and be unstable in training or fine-tuning (Appendix A).
  - Many strong public models (e.g., T5, LLaMA) ship with `MHA` and thus do not enjoy `MQA` speedups; retraining large models from scratch just to get `MQA` is often impractical (Section 1).
  - Other efficiency methods (FlashAttention, quantization, distillation, speculative decoding) are complementary but do not specifically address the KV-cache head sharing trade-off (Section 4).
- Positioning of this work:
  - Provides a low-cost “uptraining” path from existing `MHA` checkpoints to `MQA` or the proposed `GQA` so users can get fast decoding without full retraining (Sections 2.1–2.2).
  - Introduces `GQA`, an interpolation between `MHA` and `MQA` that shares one key/value head per group of query heads, striking a tunable quality–speed balance (Section 2.2, Figure 2).

## 3. Technical Approach
The paper presents two tightly coupled pieces: a conversion-and-uptraining recipe, and the `GQA` attention variant.

- Terminology (uncommon terms):
  - `KV-cache`: the stored keys and values for past tokens used by the decoder at each new decoding step. Reducing its size reduces memory reads.
  - `MQA`: attention with many query heads but a single shared key head and a single shared value head.
  - `GQA-G`: attention with many query heads split into `G` groups; each group shares its own single key head and single value head. `G=1` equals `MQA`; `G=H` (number of query heads) equals `MHA` (Section 2.2, Figure 2).
  - `Uptraining`: resuming pretraining from a modified checkpoint for a small fraction (`α`) of the original pretraining budget to let the model adapt to the new architecture (Section 2.1).

A) Checkpoint conversion (how to get from MHA to MQA/GQA)
- What changes:
  - Only the key and value projection matrices are restructured; query and output projections remain as in the original model.
- How it works (Section 2.1, Figure 1):
  - For each attention layer with `H` heads, gather the key projection matrices K₁…K_H and value projection matrices V₁…V_H.
  - To produce `MQA`:
    - “Mean-pool” the per-head projections: K_MQA = mean(K₁…K_H); V_MQA = mean(V₁…V_H).
  - To produce `GQA-G`:
    - Partition the `H` heads into `G` equal (or near-equal) groups.
    - For each group, mean-pool the K and V matrices of the heads in that group to form one shared key and one shared value projection for that group (Section 2.2; Figure 2).
  - Rationale for mean-pooling:
    - Preserves information from all heads rather than discarding most (Figure 4 shows it outperforms head selection and random reinit after uptraining).
- Where applied:
  - Decoder self-attention and decoder cross-attention only; encoder self-attention is left as MHA because encoder runs in parallel and is not the bottleneck (Section 2.1, 2.2).

B) Uptraining to adapt after conversion
- Recipe (Section 2.1):
  - Continue pretraining on the same corpus and schedule as the original T5.1.1 models for a small fraction `α` of original steps (e.g., `α=0.05`).
  - For `α=0.05`, total cost was ~600 TPUv3 chip-days (Section 3.1 Uptraining).
- Why needed:
  - Converting the attention parameters changes the inductive structure; few-percent additional pretraining lets the model re-equilibrate. `GQA` looks reasonable immediately after conversion, while `MQA` needs uptraining to be competitive (Figure 5).

C) Grouped-Query Attention (`GQA`) mechanics and why it helps
- Mechanics (Section 2.2, Figure 2):
  - Keep the usual number of query heads to preserve fine-grained attention queries.
  - Share one key and one value projection within each group of query heads, which reduces KV-cache size and memory bandwidth relative to MHA.
- Why not always `MQA`?
  - `MQA` collapses all key/value heads to one, saving the most bandwidth but reducing representational capacity and sometimes hurting stability/quality.
  - `GQA` scales bandwidth and capacity between MHA and MQA. As models get wider with more heads, `GQA` can maintain a proportional cut in KV-cache while preserving multiple K/V “channels” (Section 2.2).
  - Practical system benefit: with standard tensor-parallel sharding, single K/V heads get replicated across partitions, wasting memory; `GQA` reduces that waste since there are several K/V heads to shard (Section 2.2).

D) Implementation and experimental setup essentials
- Architecture and training stack: T5.1.1 in JAX/Flax/Flaxformer, Adafactor with T5 schedules (Section 3.1 Configurations).
- Where evaluated: CNN/DM, arXiv, PubMed, MediaSum, Multi-News (summarization); WMT14 En→De (translation); TriviaQA (QA) (Section 3.1 Data).
- Fine-tuning hyperparameters: LR 0.001, batch 128, dropout 0.1; length ranges: inputs up to 2048, outputs up to 512 for long summarization; greedy decoding (Section 3.1 Fine-tuning).
- Timing methodology: time per sample per TPUv4 chip using xprof; 8 TPUs, largest batch that fits (Section 3.1 Timing).

Analogy for intuition:
- Think of `MHA` as many microphones (query heads) listening to many separate sound channels (key/value heads). `MQA` shrinks all sound channels to just one, which is efficient but loses richness. `GQA` keeps several sound channels—fewer than MHA but more than MQA—giving a good mix of fidelity and efficiency.

## 4. Key Insights and Innovations
- Low-cost “uptraining” conversion path from MHA to MQA/GQA:
  - Innovation: Mean-pool K/V projections per head (or per group) then continue pretraining for only ~5–10% of the original budget to recover quality (Section 2.1; Figure 4; Figure 5).
  - Significance: Enables practitioners to “upgrade” existing MHA checkpoints (e.g., public T5) into fast-decoding variants without full retraining. At `α=0.05`, the XXL conversions required ~600 TPUv3 chip-days (Section 3.1).
- Grouped-Query Attention (`GQA`):
  - Innovation: An architectural interpolation that shares K/V within groups of query heads (`GQA-G`), with `G` controlling the speed–quality trade-off (Section 2.2, Figure 2).
  - Significance: Retains most of MQA’s speed while regaining much of MHA’s quality; particularly attractive for large models and for sharded deployments where single K/V heads are replicated (Section 2.2; Figure 6).
- Empirically validated trade-off curves:
  - Finding: Uptrained `MQA`-XXL is both faster and higher-quality than `MHA`-Large; uptrained `GQA-8`-XXL nearly matches `MHA`-XXL quality at MQA-like speed (Figure 3, Table 1).
  - Significance: Demonstrates a concrete Pareto improvement path in practice for long-sequence tasks and generation workloads.
- Stability observation:
  - Finding: `MQA` training/fine-tuning can be unstable on long-input tasks (loss spikes, divergence), while uptrained `GQA` is stable (Appendix A).
  - Significance: Strengthens the case for `GQA` as the safer deployment target when converting existing models.

## 5. Experimental Analysis
- Evaluation methodology (Section 3.1):
  - Datasets and metrics:
    - Summarization: CNN/DailyMail, arXiv, PubMed, MediaSum, Multi-News; metric: ROUGE-1 (`R1`).
    - Translation: WMT14 En→De; metric: BLEU.
    - QA: TriviaQA; metric: F1.
  - Models compared:
    - `MHA-Large` (T5-Large), `MHA-XXL` (T5-XXL), and 5%-uptrained `MQA-XXL` and `GQA-8-XXL`.
  - Decoding and timing:
    - Greedy decoding.
    - Time per sample per TPUv4 chip via xprof; 8 TPUs, large batch (Section 3.1).
- Main quantitative results (Table 1; Figure 3):
  - Speed:
    - `MHA-XXL`: 1.51 s/sample
    - `MQA-XXL`: 0.24 s/sample
    - `GQA-8-XXL`: 0.28 s/sample
    - `MHA-Large`: 0.37 s/sample
  - Average quality across tasks:
    - `MHA-XXL`: 47.2 (avg)
    - `GQA-8-XXL`: 47.1 (avg)
    - `MQA-XXL`: 46.6 (avg)
    - `MHA-Large`: 46.0 (avg)
  - Concrete per-task snapshots (Table 1):
    - CNN/DM R1: 42.9 (Large), 43.8 (MHA-XXL), 43.0 (MQA-XXL), 43.5 (GQA-8-XXL)
    - WMT14 BLEU: 27.7 (Large), 28.4 (MHA-XXL), 28.5 (MQA-XXL), 28.4 (GQA-8-XXL)
    - TriviaQA F1: 78.2 (Large), 81.9 (MHA-XXL), 81.3 (MQA-XXL), 81.6 (GQA-8-XXL)
  - Trade-off picture (Figure 3):
    - > Uptrained `MQA-XXL` is faster than `MHA-Large` while yielding higher average quality.
    - > `GQA-8-XXL` reaches quality very close to `MHA-XXL` but at a speed near `MQA-XXL`.
- Ablation and diagnostic studies:
  - Checkpoint conversion strategies (Figure 4):
    - > Mean-pooling K/V projections performs best after uptraining, followed by taking a single head, with random initialization worst—consistent with how much pretrained information is preserved.
  - Uptraining proportion `α` (Figure 5):
    - > `GQA` already works reasonably well right after conversion, while `MQA` needs uptraining to be competitive.
    - > Both `MQA` and `GQA` benefit substantially from 5% uptraining, with diminishing returns beyond 10%.
  - Number of `GQA` groups vs. speed (Figure 6):
    - > Moving from `MQA` (1 group) to 8 groups incurs only a modest slowdown, but the cost rises as you approach `MHA`. The paper selects 8 groups as a practical middle ground.
- Stability evidence (Appendix A):
  - > Multiple `MQA` T5-Large runs trained from scratch showed frequent loss spikes and divergence when fine-tuned on long-input tasks; uptrained `MQA` models are better but still high-variance. Uptrained `GQA` models are stable; thus, for unstable tasks, `MQA` results are averaged over three runs.
- Do the results support the claims?
  - Yes. The key claims are about throughput and the quality–speed trade-off:
    - `MQA-XXL` achieves a ~6.3× speedup over `MHA-XXL` (1.51 → 0.24 s/sample) with a modest average quality drop (47.2 → 46.6).
    - `GQA-8-XXL` retains most of the speedup (0.28 s/sample) and is almost indistinguishable in average performance from `MHA-XXL` (47.1 vs. 47.2). These are visible in Table 1 and summarized in Figure 3.
  - The ablations credibly justify the specific design choices (mean-pooling, 5% uptraining, ~8 groups).

## 6. Limitations and Trade-offs
- Quality vs. bandwidth:
  - `MQA` reduces K/V capacity to one head, which can reduce quality and stability; `GQA` reintroduces some capacity but with some bandwidth cost (Sections 2.2, 3.3; Figure 6).
- Evaluation scope:
  - Focused on encoder–decoder models (T5 family). Decoder-only LLMs are not empirically evaluated here, though the authors expect even stronger relative benefits there (Limitations).
- Metric sensitivity for long outputs:
  - Many showcased gains are on long-sequence tasks where automatic metrics like ROUGE-1 or BLEU are imperfect measures of quality (Limitations).
- Training recipe coverage:
  - The study converts and uptrains; it does not compare to an `MQA/GQA` model trained fully from scratch at the XXL scale (Limitations).
- Stability:
  - `MQA` can be unstable on long-input tasks; although uptraining helps, variance remains higher than with `GQA` (Appendix A).
- Compute and implementation constraints:
  - Uptraining still costs nontrivial compute (~600 TPUv3 chip-days for XXL at `α=0.05`), though far less than full pretraining (Section 3.1).
  - Gains depend on memory bandwidth bottlenecks and the serving stack; different hardware or kernels may change the balance.

## 7. Implications and Future Directions
- Impact on the field and practice:
  - Provides a turnkey path to speed up existing `MHA` checkpoints with minimal compute and code changes, directly improving deployability of popular public models (Section 2.1–2.2; Table 1).
  - `GQA` offers a tunable knob (`G`) to match service-level objectives: choose more groups for quality, fewer for speed (Figure 6).
- Potential follow-ups:
  - Apply `GQA` to decoder-only models (the paper anticipates stronger advantages due to the absence of cross-attention splitting; Limitations).
  - Jointly optimize with complementary methods:
    - FlashAttention to reduce on-chip memory movement.
    - Quantization of KV-cache and projections.
    - Speculative decoding to parallelize acceptance checking while still benefiting from smaller KV-cache loads (Section 4).
  - Stability research:
    - Understand and mitigate `MQA` instability on long inputs; explore loss scaling, normalization, or architectural tweaks (Appendix A).
  - Adaptive or layer-wise grouping:
    - Vary `G` by layer or by sequence length; dynamically allocate more K/V groups for long contexts or harder inputs.
  - Systems optimization:
    - Explore tensor-parallel/sharding strategies that most effectively exploit `GQA`’s multi-KV-head structure to avoid replication overhead (Section 2.2).
- Practical applications:
  - Low-latency summarization of long documents, real-time translation, and QA, where memory bandwidth dominates inference time.
  - Cost-effective model serving on bandwidth-limited hardware or at very large batch sizes, especially for long-context generation.

Quoted highlights for quick reference:
- > “Key and value projection matrices from all heads are mean pooled into a single head” during conversion (Section 2.1; Figure 1).
- > “Uptrained GQA achieves quality close to multi-head attention while being almost as fast as multi-query attention” (Abstract; confirmed by Table 1 and Figure 3).
- > “For α = 0.05, training took approximately 600 TPUv3 chip-days” (Section 3.1 Uptraining).
- > “Going from 1 (MQA) to 8 groups adds modest inference overhead, with increasing cost to adding more groups” (Figure 6).
- > “MQA… can lead to training instability… uptrained GQA models… appear to be stable” (Appendix A).
