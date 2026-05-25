## 2. Context and Motivation

### The Core Problem: KV Cache Memory Bottlenecks in Extended Reasoning

Modern LLMs that perform chain-of-thought reasoning can generate sequences spanning tens of thousands of tokens—a trend accelerated by reasoning-focused training and inference-time scaling. Every generated token appends new keys and values to the KV cache, whose size grows linearly with sequence length. For a model with $L$ layers, $h$ heads per layer, and $d$ keys/values per head, storing the KV cache for a single sequence of $N$ tokens costs $2 \times L \times h \times d \times N$ elements (keys and values), easily consuming over 50 GB of GPU memory for 8B-parameter models at 32K tokens. This memory pressure constrains batch sizes, limits the maximum context length that can be practically handled, and prevents deployment on consumer GPUs—all critical barriers for reasoning models that are becoming foundational building blocks in AI systems.

The problem is especially acute for **retrieval heads**: specialized attention heads that attend to tokens far in the past when specific information becomes relevant again. In reasoning chains, the model may need to recall an intermediate result produced thousands of tokens earlier. If that information was evicted from the KV cache because it appeared unimportant at some intermediate step, the reasoning chain breaks—causing hallucinations, logical errors, or outright failure to solve the problem. Prior compression methods that rely on short observation windows of recent queries often fail precisely in these “dormant then critical” scenarios, as we detail below.

### Why This Is Important

The practical value is immediate: efficient KV compression allows long-reasoning LLMs to run with reduced hardware requirements and higher throughput. In this paper, TriAttention enables the OpenClaw agent to complete a multi-turn document processing task on a single RTX 4090 (24 GB) with a 32B-parameter model, where Full Attention runs out of memory (Appendix J). On the AIME25 benchmark, matching full-accuracy throughput improvements reach 2.5× (Table 4, Figure 1), directly translating to cost savings in deployment.

The theoretical significance is equally compelling. RoPE-based attention is the dominant architecture in modern LLMs, but its interaction with cached representations has largely been studied through the lens of observing post-RoPE attention scores—a reactive approach. By discovering that pre-RoPE vectors in most heads concentrate tightly around non-zero centers, the paper reveals that attention patterns are **predictable from stable pre-RoPE statistics**, not just observable from noisy post-RoPE snapshots. This reframes KV importance estimation from an empirical observation problem into a geometric prediction problem rooted in the structure of RoPE itself.

### Prior Approaches and Where They Fall Short

KV cache compression methods fall into three categories, all of which operate on **post-RoPE** (position-rotated) representations. To understand their limitations, recall how Rotary Position Embedding (RoPE) works: a query vector $\mathbf{q}$ at position $p$ is rotated by frequency $\omega_f$ in each 2D band as $\tilde{\mathbf{q}}_f = \mathbf{q}_f \cdot e^{i \omega_f p}$. This rotation entangles positional information with the original vector direction, so the same token's query representation rotates continuously as it advances through positions.

#### Heuristic Methods

StreamingLLM (Xiao et al., 2024) exploits the “attention sink” phenomenon—initial tokens receive disproportionately high attention regardless of their content—by permanently retaining a few initial sink tokens plus a sliding window of recent tokens. While enabling theoretically infinite-length streaming, heuristic rules cannot adapt to content-dependent importance: tokens that are critical but fall outside the sliding window or sink set are irretrievably lost. For reasoning, where important intermediate results can appear far from the beginning or end, this approach is too rigid.

#### Attention-Based Methods: The Observation Window Problem

H2O (Zhang et al., 2023), SnapKV (Li et al., 2024b), R-KV (Cai et al., 2025), and LazyEviction (Zhang et al., 2025) all estimate key importance by accumulating or querying attention scores from recent queries. The core assumption is that high attention from recent queries signals future importance. However, RoPE causes queries to **rotate with position**, so a query at position $p$ and a query at position $p+\delta$ have different angular orientations in each frequency band. Consequently:

> “only the most recent queries retain up-to-date orientations, forming a tiny observation window. With so few representative queries, important keys go undetected—a token receiving low attention during this short window may be permanently evicted, even if it becomes critical later.” (Section 1)

The window is not just small; prior work shows that **extending it does not help**. Zhang et al. (2025) found that performance peaks at around 25 queries and declines thereafter, because older queries have corrupted positional rotations and act as noise. The paper’s own experiments confirm the practical impact: on AIME25, R-KV—a state-of-the-art attention-based method for reasoning models—achieves only 17.5% accuracy compared to Full Attention’s 40.8% at a fixed KV budget of 2048 (Table 1), essentially halving correct answer rates. This instability is not a marginal degradation; it fundamentally undermines reasoning quality.

#### Norm-Based Methods: Ignoring Directional Information

VATP (Guo et al., 2024) improves on pure attention scores by incorporating the norm of value vectors, since attention sinks receive high attention but have near-zero value norms and contribute little to output. While this corrects one blind spot of attention-score-based methods, it **discards all directional information**. In post-RoPE space, the direction between query and key encodes positional alignment—critical for determining whether a key's content actually matches the query's current need. However, because the post-RoPE direction rotates with position, it is difficult to extract a stable directional signal without a fixed reference frame. Norm-based methods therefore provide only a partial importance signal.

#### The Common Source of Failure

All these methods share a structural weakness: they operate in the **post-RoPE** space where positional rotation corrupts the signal used for importance estimation. For attention-based methods, the rotation limits the useful observation window. For norm-based methods, the rotation obscures directional relationships. Neither can exploit the fact that the **pre-RoPE** vectors—before positional encoding is applied—are position-independent and might contain predictable structure.

### How TriAttention Positions Itself

TriAttention breaks from this tradition entirely by **moving to the pre-RoPE space**. The paper’s key empirical discovery is **Q/K concentration**: in the pre-RoPE space, the query and key vectors in a large fraction of attention heads cluster tightly around fixed non-zero centers, and this concentration is stable across positions, input contexts, and even different model architectures (Section 3, Figures 2 and 3). The mean resultant length $R = \| \mathbb{E}[\mathbf{q}] \| / \mathbb{E}[\|\mathbf{q}\|]$ approaches 1.0 for over 84% of heads in GQA models and over 96% in MLA models (Appendix I, Table G).

This concentration has a profound consequence: when $\mathbf{q}$ and $\mathbf{k}$ are approximately constant, substituting their centers into the RoPE attention formula transforms the attention logit into a **trigonometric series that depends only on the relative distance $\Delta = p_q - p_k$**:

$$\logit(\Delta) \approx \sum_f \underbrace{\| \mathbb{E}[\mathbf{q}_f] \| \,\| \mathbb{E}[\mathbf{k}_f] \|}_{\text{amplitude}} \cos\!\big(\omega_f \Delta + \bar{\phi}_f\big)$$

The coefficients (amplitudes, phases) are completely determined by the pre-RoPE centers, which are fixed and can be computed offline from calibration data. This means attention patterns—which keys will receive high attention from future queries—are **predictable from distance alone**, using a formula that requires no observation window at all.

TriAttention’s scoring function therefore avoids the instability of post-RoPE methods entirely: it scores each cached key using the predicted distance-dependent attention curve from the Q center and the trigonometric series, plus a norm-based term weighted by concentration (Equation 10). The method is not an incremental improvement to observation-window-based scoring; it is a **principled shift** from “observe attention to guess importance” to “predict importance from pre-RoPE geometry.”

The paper also explicitly re-contextualizes prior negative results. For instance, the difficulty of retrieval heads—where tokens receive zero attention until suddenly needed—is precisely the scenario where observation windows fail. TriAttention’s distance-based scoring naturally assigns high scores to tokens at the distances where retrieval heads peak, regardless of current attention values, addressing the core failure mode of existing compression methods on long-reasoning tasks.