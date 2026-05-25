# TriAttention: Efficient Long Reasoning with Trigonometric KV Compression

A trigonometric formula derived from fixed Q/K centers can determine key importance so reliably that a model retains full reasoning accuracy with 10.7× less KV memory—while leading compression methods drop to roughly half the accuracy at the same budget. The key is to escape the instability of post-RoPE attention scores by exploiting a previously overlooked clustering phenomenon in the pre-RoPE space, where stable centers let you predict which distances each head will attend to.

---

## 1. Executive Summary

This paper introduces **TriAttention**, a KV cache compression method for long-reasoning LLMs that first identifies a pre-RoPE property called **Q/K concentration**—query and key vectors cluster tightly around fixed non-zero centers across most heads—and then uses these centers to derive a trigonometric series that scores key importance from the predicted distance-dependent attention pattern, combined with a norm-based term. On AIME25 with Qwen3-8B, TriAttention matches Full Attention accuracy (40.8%) while achieving 2.5× higher throughput or 10.7× KV memory reduction, whereas leading baselines like R-KV attain only about half that accuracy at the same efficiency. The approach establishes that pre-RoPE concentration is a model-intrinsic, architecture-general property that yields stable, predictable distance preferences, enabling KV compression that avoids the instability of observation-window-based post-RoPE methods and delivers the largest gains on challenging mathematical reasoning benchmarks where those methods substantially degrade.
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
## 3. Technical Approach

### 3.1 Reader Orientation (What Is Being Built)

TriAttention is a **compression strategy for the key–value (KV) cache** used in transformer language models. During long reasoning, it decides—once every 128 generated tokens—which of the many stored keys to keep (the rest are evicted) so that the cache stays within a fixed memory budget while barely affecting the model’s answer quality. The core idea is to **predict key importance from the per-head geometry of the queries and keys as they exist before positional rotation**, rather than by observing short windows of recent attention scores, thereby avoiding the instability that cripples prior methods.

### 3.2 Big-Picture Architecture (Diagram in Words)

The system has four major stages that alternate between **offline** preparation and **online** execution:

1. **Offline Calibration** – Over a corpus of calibration text, the raw (pre-RoPE) query and key vectors are collected for every attention head. From these, two quantities are extracted per head and per frequency band: the **mean vector** (the centre of the query/key cloud) and the **expected norm**. These are the only statistics that TriAttention needs to carry into inference; they are fixed once calibrated.

2. **Online Scoring** – During generation, every 128 tokens a scoring pass is triggered. For each cached key (and for each shared query head in grouped-query attention), TriAttention computes a single importance score by evaluating (a) a **trigonometric-series score** that captures the key’s expected usefulness at a future query–key distance, and (b) a **norm-based score** that supplements the series on heads where concentration is weaker. The two are blended using the head’s own concentration metric.

3. **Multi-Offset Aggregation** – Because a key may be queried from many future positions, the trigonometric-series score is evaluated at a geometric progression of future offsets (1, 2, 4, … up to large values) and averaged, yielding a single “distance-score” that is robust to future position.

4. **Top-$B$ Pruning** – The combined score is z-normalised per query head (to align scales), then maximised across the query heads that share a KV head in GQA. The $B$ highest-scoring keys are kept; all others are dropped. After the pruning pass, generation resumes with the shrunken cache.

### 3.3 Roadmap for the Deep Dive

To understand how TriAttention works, we will move through the following steps, each building on the previous:

- **The pre-RoPE concentration phenomenon** (what it is, how it is measured, why it is robust) – because it is the foundation that makes the trigonometric-series approach possible.
- **From concentration to a trigonometric series** – showing through the RoPE formula why concentrated Q/K vectors cause attention to follow a function of distance alone, and how that function can be written in closed form.
- **Validation via reconstruction correlation** – confirming that the predicted distance-based attention does indeed match real attention patterns, and across many models.
- **The TriAttention scoring function** – how the trigonometric series is turned into a per-key score $S_\text{trig}$, how a complementary norm score $S_\text{norm}$ is defined, and how the two are combined via the concentration metric $R$ into a final $S(k, \Delta)$.
- **Handling future queries: the offset averaging and GQA aggregation** – why the same key must be scored at multiple future distances, and how scores from multiple query heads are normalised and merged.
- **The window-based pruning schedule** – when pruning happens, why 128-token windows, and how the overall inference loop works.
- **Offline calibration** – exactly what statistics are collected, over how much data, and why cross-domain generalisation holds.

### 3.4 Detailed, Sentence-Based Technical Breakdown

This is primarily an **analysis-and-method paper**. Its core insight is that the pre-RoPE query and key vectors in modern LLMs are highly concentrated around fixed, non-zero centres, and that this concentration lets us replace observation-dependent importance estimation with a **geometry-prediction** approach that is both simpler and more stable. What follows is a walk-through of every piece that makes that insight into a functioning cache compressor.

---

#### The Pre-RoPE Q/K Concentration Phenomenon

Before RoPE applies its position-dependent rotation, each attention head processes raw query and key vectors (the “pre‑RoPE” vectors) that are **content‑only**—no position information has yet been injected. The paper’s key empirical discovery is that in the vast majority of attention heads, these raw vectors are **not** spread uniformly; instead they cluster tightly around a fixed, non-zero centre.

**Quantifying concentration.** The paper uses the **Mean Resultant Length** ($R$), a standard statistic from directional data analysis. For a frequency band $f$, where queries $\mathbf{q}_f$ are 2‑D complex vectors (one complex number per RoPE pair), the Mean Resultant Length is:

$$R_f = \frac{\|\mathbb{E}[\mathbf{q}_f]\|}{\mathbb{E}[\|\mathbf{q}_f\|\]}$$

where $\mathbb{E}[\mathbf{q}_f]$ is the arithmetic mean (centre) of the query vectors in that band, $\|\cdot\|$ denotes the complex modulus, and $\mathbb{E}[\|\mathbf{q}_f\|\]$ is the expected length of those vectors. The ratio lies in $[0,1]$: $R_f = 1$ means every query vector points in exactly the same direction and has the same magnitude (perfect concentration); $R_f = 0$ means the vectors are evenly dispersed in angle or magnitude.

**What it computes:** The numerator captures the “net” vector—if all vectors point in similar directions, their average is a strong vector with large length. The denominator normalises by the typical vector length, so $R_f$ is near 1 only when the vectors are tightly bunched around their mean direction *and* have consistent magnitude. If the vectors vary widely in direction or scale, the numerator shrinks relative to the denominator, and $R_f$ falls.

**Why this form:** Naïvely checking “is $\|\mathbb{E}[\mathbf{q}]\|$ large?” is not enough because it confuses large magnitudes (just long vectors) with real angular concentration. Dividing by the expected norm removes that scale dependence; only genuine directional clustering yields high $R_f$. The statistic is also invariant to uniform rescaling, which is desirable because what matters for the trigonometric series is the shape of the Q/K cloud, not overall strength.

**Evidence of concentration.** Figure 2(C) shows a histogram of $R_f$ across all 1152 attention heads of Qwen3‑8B. The vast majority of heads have $R_f$ approaching 1.0—the distribution is strongly skewed right. This is not an artifact of one model: Appendix I (Table G) reports that 84.7% of heads in GQA architectures (Qwen3‑8B) and 96.6% of heads in MLA architectures (GLM‑4.7‑Flash) have $R > 0.95$. The paper explicitly states: “Q/K concentration is a model-intrinsic property, consistent across domains and architectures.”

**Stability across positions and contexts.** Figure 2(A) overlays the dominant‑band Q/K vectors from three different input sequences (Math, Coding, Chat) on the same 2‑D plane; the points form a tight cluster, not three distinct clusters. Quantitatively, the mean $R$ computed over Math, Coding, and Chat domains yields nearly identical values (0.977–0.980). Therefore, the Q/K centres are not task‑specific; they are a fixed property of the pretrained model.

**Why this matters.** Since the pre‑RoPE vectors are position‑agnostic (position is added *after* this stage), their concentration means that for a given head, the query vector $\mathbf{q}$ can be approximated by its fixed centre $\mathbb{E}[\mathbf{q}]$ regardless of where in the sequence it occurs. That approximation unlocks the trigonometric‑series representation we examine next.

---

#### From Concentration to a Trigonometric Series

RoPE attention computes the logit between a query at position $p_q$ and a key at position $p_k$ by rotating the pre‑RoPE vectors and taking the dot product. In complex form, for frequency band $f$, the contribution from that band is:

$$\langle \mathbf{q}, \mathbf{k} \rangle_f = \operatorname{Re}\!\big( \tilde{\mathbf{q}}_f(p_q) \,\bar{\tilde{\mathbf{k}}}_f(p_k) \big) = \|\mathbf{q}_f\| \|\mathbf{k}_f\| \cos\big(\omega_f \Delta + \phi_f\big)$$

where $\Delta = p_q - p_k$ is the query–key distance, $\omega_f = \theta^{-2f/d}$ is the rotation frequency (with $\theta = 10\,000$), and $\phi_f = \arg(\mathbf{q}_f) - \arg(\mathbf{k}_f)$ is the phase difference between the pre‑RoPE vectors. Summing over all bands $f$ gives the full pre‑softmax logit:

$$\operatorname{logit}(\mathbf{q}, \mathbf{k}) = \sum_f \|\mathbf{q}_f\| \|\mathbf{k}_f\| \cos(\omega_f \Delta + \phi_f) \qquad (2)$$

So far this is exact and per‑token. **Now we apply the concentration insight.** If $\mathbf{q}_f$ and $\mathbf{k}_f$ are each tightly clustered around their respective centres $\bar{\mathbf{q}}_f = \mathbb{E}[\mathbf{q}_f]$, $\bar{\mathbf{k}}_f = \mathbb{E}[\mathbf{k}_f]$, then we can replace the per‑token vectors with these centres and obtain an approximation that depends only on distance $\Delta$:

$$\operatorname{logit}(\Delta) \approx \sum_f \underbrace{\|\bar{\mathbf{q}}_f\| \|\bar{\mathbf{k}}_f\|}_{\text{amplitude } a_f} \cos\!\big(\omega_f \Delta + \underbrace{\bar{\phi}_f}_{\text{phase}}\big)$$

where $\bar{\phi}_f = \arg(\bar{\mathbf{q}}_f) - \arg(\bar{\mathbf{k}}_f)$. Using the cosine angle addition formula, this can be rewritten as a trigonometric series over the frequencies $\omega_f$:

$$\operatorname{logit}(\Delta) = \sum_f \big[ a_f \cos(\omega_f \Delta) + b_f \sin(\omega_f \Delta) \big] \qquad (3)$$

where $a_f = \|\bar{\mathbf{q}}_f\| \|\bar{\mathbf{k}}_f\| \cos(\bar{\phi}_f)$ and $b_f = -\|\bar{\mathbf{q}}_f\| \|\bar{\mathbf{k}}_f\| \sin(\bar{\phi}_f)$ are constants determined entirely by the Q/K centres.

**What it computes:** Given only the pre‑RoPE centres (which are fixed after calibration), the equation produces a scalar that predicts the attention logit a query *at any future position* would assign to a key at a distance $\Delta$ behind it. It is a **distance‑only function**—no per‑token query vector appears because we have approximated it by its centre.

**Why this form:** The original logit (2) couples query‑specific magnitudes $\|\mathbf{q}_f\|$ and phases $\phi_f$ with key content; no simplification is possible without strong assumptions. The concentration assumption makes those magnitudes and phases approximately constant, converting the logit into a spectral synthesis over the fixed RoPE frequency set. Although the RoPE frequencies follow a geometric progression $\omega_f = 10000^{-2f/d}$ rather than the harmonic progression of a Fourier series, the principle is identical: a set of learned amplitudes and phases shapes an **attention‑vs‑distance curve**. Different heads learn different centres, and therefore different curves—some peak at small distances (local attention), others at large distances (attention sinks), etc.

---

#### Validation: Reconstruction Correlation

The paper tests whether this centre‑based trigonometric series actually captures real attention patterns by measuring **reconstruction correlation** $\bar{r}$.

For a given head, the predicted logit at distance $\Delta$ is computed from the centres using:

$$\hat{s}(\Delta) = \sum_f \|\mathbb{E}[\mathbf{q}_f]\| \|\mathbb{E}[\mathbf{k}_f]\| \cos(\omega_f \Delta + \bar{\phi}_f) \qquad (4)$$

where $\bar{\phi}_f = \arg(\mathbb{E}[\mathbf{q}_f]) - \arg(\mathbb{E}[\mathbf{k}_f])$. For each query $i$ in a real sequence, let $\mathbf{a}_i$ be the vector of actual attention logits that query produced against a set of keys at logarithmically‑spaced distances $\Delta \in \{1, 2, 4, 8, \ldots\}$, and let $\hat{\mathbf{s}}$ be the corresponding predicted logits from (4). The per‑query Pearson correlation is:

$$r_i = \rho(\mathbf{a}_i, \hat{\mathbf{s}}) = \frac{\operatorname{Cov}(\mathbf{a}_i, \hat{\mathbf{s}})}{\sigma_{\mathbf{a}_i} \sigma_{\hat{\mathbf{s}}}} \qquad (24)$$

The final reconstruction correlation $\bar{r}$ is the average over all queries in the sequence:

$$\bar{r} = \frac{1}{N} \sum_{i=1}^N r_i \qquad (25)$$

**What it computes:** A single number per head, $\bar{r} \in [-1, 1]$, that quantifies how well the centre‑based curve predicts the attention logits actually observed during a forward pass. Log‑spaced distances are used so that nearby and far‑away keys are equally represented; otherwise, the many nearby positions would dominate the correlation.

**Why this form:** Pearson correlation is scale‑ and location‑invariant, so it focuses purely on the *shape* agreement between predicted and actual logits rather than absolute magnitudes. This is appropriate because the centre approximation may miss some scale factors (the key‑specific norms, which we later handle separately) but should still capture the positional profile. If $\bar{r}$ is high, it confirms that Q/K concentration truly causes predictable distance preferences.

**Results.** Figure 2(D) shows an example head (Layer 0, Head 0 of Qwen3‑8B) with $\bar{r} = 0.72$, visually very close. Across all heads in three model families (Qwen3, Qwen2.5, Llama3), the distribution of $\bar{r}$ peaks between 0.6 and 0.9 with means above 0.5 (Figure 3). Thus, the distance‑based prediction is not just statistically significant; it is **strong** for a large fraction of heads. This validates the chain: concentration $\rightarrow$ constant Q/K $\rightarrow$ distance‑only attention.

---

#### The TriAttention Scoring Function

TriAttention uses the geometric understanding above to assign each cached key an importance score. The overall form is a sum of two terms, weighted by how concentrated the query distribution is:

$$S(k, \Delta) = S_{\text{trig}}(k, \Delta) + S_{\text{norm}}(k) \qquad (10)$$

We unpack each term, then the weighting mechanism.

##### Trigonometric Series Score $S_{\text{trig}}$

The series derived in §3 predicts attention logits when *both* the query and the key are approximated by their centres. In a real KV cache, however, the keys are known exactly—we have the actual pre‑RoPE key vectors of the tokens we are considering keeping. Therefore, we can improve accuracy by using the real key’s vector $\mathbf{k}_f$ in the amplitude, while still approximating the unknown future query by its centre. This gives:

$$S_{\text{trig}}(k, \Delta) = \sum_f \|\mathbb{E}[\mathbf{q}_f]\| \;\| \mathbf{k}_f \| \cos\!\big(\omega_f \Delta + \phi_f\big) \qquad (6)$$

where $\phi_f = \arg(\mathbb{E}[\mathbf{q}_f]) - \arg(\mathbf{k}_f)$.

**What it computes:** For a specific cached key $k$ at a specific offset $\Delta$ from the *expected* future query position, it estimates the attention logit that key would receive, using the Q centre for direction but the key’s true norm and phase. The result is a scalar per frequency band, summed across bands.

**Why this form:** If we used the key centre $\mathbb{E}[\mathbf{k}_f]$ instead of the actual $\mathbf{k}_f$, we would treat all keys at a given distance identically, which is clearly too coarse—two keys at the same distance can differ in content salience. By retaining the real $\|\mathbf{k}_f\|$ and $\arg(\mathbf{k}_f)$, the score distinguishes between equally‑distant keys. This hybrid “one‑side centre, one‑side real” design balances the stability of the centre approximation with sensitivity to per‑key content.

##### Norm-Based Score $S_{\text{norm}}$

Even with real keys, the trigonometric series assumes that every query $\mathbf{q}$ is exactly at its centre, which is not true for all heads. When query vectors vary around the centre, a key with a large norm can still receive high attention regardless of the exact angle alignment, because the dot product magnitude scales with $\|\mathbf{q}\|\|\mathbf{k}\|$. To capture this “norm‑based salience” in a way that complements the distance score, the paper defines a base norm score:

$$S_{\text{norm}}^{(0)}(k) = \sum_f \mathbb{E}[\|\mathbf{q}_f\|\] \; \|\mathbf{k}_f\| \qquad (7)$$

where $\mathbb{E}[\|\mathbf{q}_f\|\]$ is the expected query norm in band $f$, and $\|\mathbf{k}_f\|$ is the actual key norm.

**What it computes:** A maximum‑possible attention amplitude, ignoring angular alignment entirely. It is the product of the expected query strength and the key strength, summed over frequency bands.

**Why this form:** Attention mechanisms fundamentally multiply query and key vectors; the cosine term modulates that product between ‑1 and +1. When the cosine term is unpredictable (because queries are not concentrated), the best a‑priori guess is to focus on keys that are physically “loud,” i.e., have large norms. This is similar in spirit to the “value norm” idea from VATP but applied in the pre‑RoPE space and weighted by expected query activity.

---

#### Adaptive Weighting via Concentration ($R_f$)

The two scores are not equally reliable for every head. The paper therefore weights $S_{\text{norm}}$ by a factor that reflects how much the query distribution actually disperses around its centre. Specifically, the Mean Resultant Length $R_f$ is used to **down‑weight** the norm term for highly concentrated heads, where the trigonometric series is already accurate. The refined norm score is:

$$S_{\text{norm}}(k) = \sum_f (1 - R_f) \; \mathbb{E}[\|\mathbf{q}_f\|\] \; \|\mathbf{k}_f\| \qquad (8)$$

where $R_f = \|\mathbb{E}[\mathbf{q}_f]\| / \mathbb{E}[\|\mathbf{q}_f\|\]$ as before.

**What it computes:** It scales the base norm score by the complement of concentration, so that when $R_f \rightarrow 1$ (tight clustering), the norm term vanishes and $S_{\text{trig}}$ dominates. When $R_f$ is smaller, the full norm contribution is preserved.

**Why this form:** A simple alternative would be to use a single global weight, but different frequency bands can have different concentrations. By applying $(1 - R_f)$ per band, the weighting adapts to the actual per‑band geometry. Moreover, this formulation has a natural interpretation after rewriting the raw norms:

$$S_{\text{norm}}(k) = \sum_f \big( \mathbb{E}[\|\mathbf{q}_f\|\] - \|\mathbb{E}[\mathbf{q}_f]\|\big) \;\|\mathbf{k}_f\| \qquad (9)$$

The term $\mathbb{E}[\|\mathbf{q}\|] - \|\mathbb{E}[\mathbf{q}]\|$ is exactly the “variance in expectation” of the query magnitude—when all queries point the same direction, this difference is zero, and the norm score disappears. Thus the weighting is not arbitrary; it directly encodes how much positional variability remains after factoring out the centre.

---

#### Aggregating Across Future Offsets

A key in the cache may be queried from many possible future positions; its importance is not determined by a single distance $\Delta$. TriAttention therefore evaluates $S(k, \Delta + \delta)$ at multiple future offsets $\delta$ and averages:

$$\tilde{S}(k) = \frac{1}{|D|} \sum_{\delta \in D} S(k, \Delta + \delta) \qquad (11)$$

where $D = \{1, 2, 4, 8, \ldots, 2^{16}\}$ (geometrically spaced up to a maximum near 65 536).

**What it computes:** For each cached key $k$ at its current distance $\Delta$ from the query, the function evaluates the combined score at that distance plus each offset, then averages. This yields a single scalar $\tilde{S}(k)$ that represents the expected importance of the key over a wide range of future positions.

**Why this form:** If we used only the immediate distance $\Delta$, the score would be heavily biased toward queries at that exact position. The trigonometric series can oscillate with distance (it is a sum of cosines), so a key that scores low at the current $\Delta$ might score high at $\Delta + 128$. Averaging over a geometric progression smooths out these oscillations and captures long‑range importance. The geometric spacing is chosen because near distances require finer sampling (the curves vary more rapidly at small $\Delta^1$), while remote distances can be coarser. Ablation in Appendix G (Table E) shows that adding offsets up to 4096 raises accuracy from 41.7% to 48.8% (AIME24), and that geometric spacing dramatically outperforms linear spacing (45.8% vs. 28.7%), confirming the need for both long range and denser near sampling.

---

#### Handling Grouped-Query Attention (GQA)

Most modern LLMs use Grouped-Query Attention, where $G$ query heads share one KV head. Consequently, the same cached key must be scored from $G$ different perspectives because different query heads have different centres $\mathbb{E}[\mathbf{q}_f]$ and norms $\mathbb{E}[\|\mathbf{q}_f\|\]$. This yields $G$ separate scores $\tilde{S}^{(g)}(k)$ for $g = 0, \ldots, G-1$.

These raw scores are not directly comparable across $g$; different heads may have different scale ranges. TriAttention therefore applies **per‑head z‑score normalisation** before aggregation:

$$\hat{S}^{(g)}(k) = \frac{\tilde{S}^{(g)}(k) - \mu_g}{\sigma_g} \qquad (12)$$

where $\mu_g$ and $\sigma_g$ are respectively the mean and standard deviation of $\{\tilde{S}^{(g)}(k)\}$ computed over all keys in the current cache for that head.

**What it computes:** For each query head, it maps the raw scores to a common scale where 0 is the average importance and ± units are standard deviations. This makes it safe to compare the same key’s importance across heads.

The final score for the key is the **maximum** across the sharing query heads:

$$S_{\text{final}}(k) = \max_{g \in \{0,\ldots,G-1\}} \hat{S}^{(g)}(k) \qquad (13)$$

**Why this form:** Using the maximum implements the principle that a key should be retained if *any* query head finds it important; even if only one head needs it, evicting the key would hurt that head’s attention. Normalisation before the max ensures that heads with larger dynamic ranges do not drown out others. The alternative of summing scores would penalise keys that are irrelevant to some heads; maximum is more conservative and empirically effective.

---

#### KV Cache Pruning Schedule and Inference Loop

TriAttention does not re‑score the cache at every decoding step, which would be computationally prohibitive. Instead, the cache is pruned **once every $\beta = 128$ generated tokens** (called a “window”). When the 128‑th token of the interval is generated, if the current cache size exceeds the budget $B$, all keys are scored using the procedure above, the top‑$B$ by $S_{\text{final}}(k)$ are retained, and the rest are evicted. Generation then continues with the reduced cache. This batching amortizes the scoring cost, which is a small overhead (the paper reports throughput improvements over Full Attention despite the periodic scoring).

The budget $B$ is a fixed hyperparameter; typical values in the experiments are 2048 or 512 tokens, depending on sequence length and model. The pruning operation itself is a simple top‑$B$ selection over the scores, requiring one additional sort per window.

**Why 128‑token windows?** This follows the setting of R‑KV (Cai et al., 2025), enabling a fair comparison. Shorter windows would reduce latency spikes but increase average overhead; longer windows would risk evicting too late. 128 is a pragmatic choice that balances both.

**Generalization to different attention architectures.** In models without GQA (e.g., standard multi‑head attention), there is no aggregation step; each head simply has its own KV cache and scores are computed directly. In MLA (Multi‑head Latent Attention), the sharing pattern differs, but the same normalisation‑and‑maximum principle can be applied after mapping latent states to per‑head scores. The paper demonstrates that the underlying Q/K concentration holds across both GQA and MLA (Appendix I), suggesting the method extends naturally.

---

#### Offline Calibration: Extracting Q/K Statistics

TriAttention’s scoring function depends on the following per‑head, per‑band statistics:

- The query centre: $\mathbb{E}[\mathbf{q}_f]$ (a complex vector per frequency band).
- The expected query norm: $\mathbb{E}[\|\mathbf{q}_f\|\]$.
- The Mean Resultant Length $R_f$ (derived from the above).
- The key centre $\mathbb{E}[\mathbf{k}_f]$ is used implicitly through the phase $\bar{\phi}_f$ in the validation, but for scoring, the real key vector $\mathbf{k}_f$ is used directly; however, $\mathbb{E}[\mathbf{k}_f]$ is still needed to compute the amplitudes in the reconstruction correlation analysis, and implicitly in the derivation.

In practice, the calibration process (described in §4.1 and Appendix H) involves:

1. Running the base model on a **calibration dataset** (e.g., 200k–960k tokens of text) in inference mode with no KV pruning, merely recording the pre‑RoPE Q/K vectors before RoPE is applied.
2. For each head and each frequency band, accumulate sum and sum‑of‑norms to obtain the sample means $\mathbb{E}[\mathbf{q}_f]$, $\mathbb{E}[\|\mathbf{q}_f\|\]$, and similarly for keys.
3. Compute $R_f$ from these aggregates.

**The calibration data can be any diverse text.** Appendix H shows that calibration data size (50k–960k tokens) and quality (Google homepage HTML vs. high‑quality chat data) have minimal impact on downstream accuracy: the Q/K centres are model‑intrinsic. The paper also demonstrates that calibrating on **code data** and evaluating on reasoning benchmarks yields comparable accuracy to calibrating directly on reasoning data (Table 3C), confirming that the centres are task‑agnostic. This is a crucial practical property: one calibration step suffices for any downstream use.

---

#### Summary of Design Choices and Their Justifications

- **Pre‑RoPE operation** over post‑RoPE: avoids the rotational instability that limits observation windows, making importance estimation a geometric prediction rather than a statistical guess.
- **Trigonometric series for distance scoring**: exploits the learned Q/K centres to predict attention shape without requiring any online samples of recent attention; stable and deterministic.
- **Hybrid of centre and real key norms**: uses the Q centre for clean distance preference but real key norms for content discrimination—the best of both worlds.
- **Concentration‑weighted norm term**: automatically adapts to per‑band geometry; norm signal only supplements where concentration is weak, preventing it from injecting noise when the series is already reliable.
- **Future offset averaging over geometric progression**: captures long‑term importance while densely sampling the critical near‑distance region, improving accuracy substantially over a single‑offset or linear spacing.
- **Max aggregation across GQA heads with per‑head normalisation**: conservative retention policy that preserves any head’s critical tokens without scale mismatches.
- **Window‑based pruning with 128‑token batch**: practical trade‑off between overhead and memory; follows established reasoning‑model compression settings.
## 4. Key Insights and Innovations

### Innovation 1: Pre‑RoPE Q/K Concentration as an Intrinsic, Model‑Wide Diagnostic Phenomenon

This paper makes the community visible a structural fact about modern RoPE‑based LLMs that had gone entirely unnoticed: in the vector space *before* positional rotation, almost all attention heads exhibit **Q/K concentration**—the raw query and key vectors are tightly clustered around non‑zero, head‑specific centers, and this clustering is stable across tokens, domains, and even architectures. The finding is not just a curiosity; it is a diagnostic property that upends how we think about attention. Previously, the field tacitly assumed that attention patterns are best understood by inspecting the *interaction* of position‑rotated queries and keys—that is, the post‑RoPE space where rotations entangle position with content. That assumption led to a whole family of observation‑based KV‑cache compression methods (H2O, SnapKV, R‑KV) that treat importance as a signal to be estimated from recent, post‑RoPE attention scores. The problem, as the paper demonstrates, is that RoPE rotations cause the set of “representative” queries to be tiny and unstable, since only the very latest queries share the correct angular orientation.

TriAttention’s first innovation is to **reframe the unit of analysis from the post‑rotation interaction to the pre‑rotation geometry**. By quantifying concentration through the Mean Resultant Length $R_f$, the paper establishes this geometry as a **model‑intrinsic**, training‑time product that is essentially frozen after pretraining. Across three distinct model families (Qwen3, Qwen2.5, Llama3) and two attention architectures (GQA and MLA), over 84–96% of heads show $R_f > 0.95$ (Figure 2C, Appendix I). The Q/K centers are so stable that calibrating them on code data versus reasoning benchmarks yields nearly identical downstream accuracy (Table 3C), and even calibration on noise (Google homepage HTML) works as well as high‑quality chat (Appendix H). This is a genuine conceptual advance: it tells us that a large fraction of a head’s “personality”—its distance‑dependent attention bias—is encoded statically in its pre‑RoPE parameters and does not depend on runtime observations. The contribution is fundamental, not incremental, because it opens an entirely new line of analysis (pre‑RoPE geometry) for understanding and engineering attention, whereas prior work was trapped inside the information‑limited post‑RoPE regime.

### Innovation 2: The Trigonometric Series as a Deterministic Predictor of Attention, Grounded in Q/K Centers

While the discovery of concentration is the phenomenological innovation, the second intellectual move is its mathematical consequence: under the high‑concentration regime, the RoPE attention logit collapses to a **trigonometric series whose coefficients are determined fully by the pre‑RoPE centers**. This is not a coarse “distance bias” heuristic; it is an exact functional identity under the approximation, and it provides a **closed‑form, parametric model of attention‑vs‑distance** for any concentrated head. Previous work discussed attention sinks, local attention, and retrieval patterns as qualitative observations with no unifying generative rule. TriAttention shows that all of these patterns can be seen as specific parameter choices in a single Fourier‑like synthesis over the fixed RoPE frequencies, where the centers $\mathbb{E}[\mathbf{q}_f]$ and $\mathbb{E}[\mathbf{k}_f]$ supply the amplitude and phase of each band.

This is a **conceptual reframing of attention**: from an opaque, token‑dependent computation to a predictable, head‑intrinsic function of position distance. It does not merely *describe* attention; it **predicts** it from static head statistics, transforming KV importance estimation from an empirical observation problem into a **deterministic, geometry‑driven prediction**. The practical validation is decisive: the reconstruction correlation $\bar{r}$ (Pearson correlation between the predicted distance curve and actual attention logits) exceeds 0.5 as a mean across all heads in three architectures (Figure 3), and individual heads achieve $\bar{r} = 0.6$–$0.9$. This high predictive power validates that the centers genuinely control the attention shape and that the trigonometric series is not a statistical artifact but the mechanism through which learned pre‑RoPE parameters enforce distance preferences. As a theoretical contribution, it explains *why* heads exhibit specific distance preferences, and it gives future researchers a mathematical language for analyzing, designing, or even steering attention patterns through the pre‑RoPE space—something that was purely descriptive before.

### Innovation 3: The Shift from Observation‑Driven to Geometry‑Predicted KV Compression (The TriAttention Scoring Paradigm)

Prior KV‑cache compression methods—regardless of whether they used attention heuristics (StreamingLLM), accumulated scores (H2O), or local windows (SnapKV, R‑KV)—share a common **reactive philosophy**: importance is estimated by *observing* which keys have recently received high attention. RoPE’s rotation turns this into a fundamentally unstable estimation problem because the useful observation window is minuscule and corrupts with distance. TriAttention’s third, and most consequential, conceptual move is to **abandon reactivity entirely**. It reframes KV compression as a **prioritization problem solvable by static geometry**: the importance of a key is not something you *measure* from recent queries; it is something you *predict* from the pre‑RoPE centers (a model constant) and the key’s own content vector, using a scoring function that has no free reliance on any runtime query sample.

This is a **paradigm shift**, not an incremental improvement to observation‑window methods. The scoring function $S(k,\Delta) = S_\text{trig}(k,\Delta) + S_\text{norm}(k)$ is **observationally zero‑shot**—it requires no recent attention scores at all—so it completely sidesteps the window‑size bottleneck and the decay problem. Moreover, the adaptive weighting by the concentration metric $R_f$ (equations 8–9) is itself a conceptual insight: instead of treating concentration as a binary property (“concentrated” vs. “not”), the paper uses it as a **continous reliability gauge** that organically blends the two signals, making the method robust without any manual tuning per head. The result is a compression method whose performance is not only higher but also qualitatively different from prior methods: on AIME25, TriAttention achieves 32.9% vs. R‑KV’s 17.5% at the same KV budget (Table 1), and it sustains memory‑retention in recursive reasoning where R‑KV suffers catastrophic degradation (Figure 5D). These are not marginal gains but the signature of a method that is structurally immune to the information bottleneck that limits all competitors. The significance extends beyond compression: it demonstrates that the pre‑RoPE space contains enough information to make runtime observation superfluous, suggesting a more general design principle for efficient attention—predict what you can from the static head geometry, and observe only the unpredictable residue.
## 5. Experimental Analysis

### Evaluation Methodology

- **Datasets.** Primary reasoning evaluation uses **AIME 2024** (30 problems), **AIME 2025** (30 problems), and **MATH 500** (500 problems). AIME benchmarks are competition‑level mathematics requiring multi‑step chain‑of‑thought; MATH 500 spans diverse mathematical reasoning tasks (Hendrycks et al., 2021). Longer‑context and memory‑retention analysis additionally employs the **Recursive State Query** benchmark (DFS simulation) and the **LongBench** (16 subtasks, 50 % KV budget) and **RULER** (retrieval, 4 K context) suites (Appendix E, F); these are used to probe generalisation beyond mathematics.

- **Base models.** Four reasoning‑capable LLMs spanning architectures and scales are evaluated: **Qwen3‑8B** (GQA), **DeepSeek‑R1‑Distill‑Llama‑8B**, **DeepSeek‑R1‑Distill‑Qwen‑7B**, and **GPT‑OSS‑20B**. These models are selected for their strong chain‑of‑thought reasoning and diverse pretraining backgrounds; most experiments are performed on Qwen3‑8B, with the others serving as cross‑model validation. Throughput and deployment tests also use **Qwen3‑32B (INT4)** on a consumer GPU (RTX 4090).

- **Metrics.** For reasoning benchmarks, correctness is measured by **pass rate**: on AIME each problem is sampled 8 times and the average fraction of correct answers is reported; on MATH 500 each problem is sampled once. Throughput is measured as the **average tokens generated per second** over a 16 K‑token decoding run on a single NVIDIA A100 80 GB GPU at the maximum batch size that fits in memory. KV‑cache memory reduction is expressed as the ratio of the original sequence length to the fixed budget $B$.

- **Baselines.** Three primary baselines are compared:
  * **Full Attention** – no KV pruning; the performance upper bound.
  * **SnapKV** (Li et al., 2024b) – an attention‑based method that scores keys from a local observation window.
  * **R‑KV** (Cai et al., 2025) – a state‑of‑the‑art KV‑cache compressor for reasoning models that combines attention‑based importance scoring with redundancy detection.
  Additional comparisons with StreamingLLM, PyramidKV, KnormPress, Ada‑KV+SnapKV, and H2O appear on LongBench and RULER (Appendix E–F), but the core analysis contrasts TriAttention with SnapKV and R‑KV.

- **Generation budget / compute accounting.** The key resource is the **KV‑cache budget** $B$, the number of key–value pairs retained per head. Uniform budgets are enforced across all layers and heads. All methods prune once every $\beta = 128$ generated tokens; when the cache exceeds $B$, it is trimmed back to $B$. For throughput comparisons, the measurement includes the overhead of periodic scoring and pruning, and all methods use FlashAttention‑2 (except GPT‑OSS which uses FlashAttention‑3 on H100). Default budgets are $B = 2048$ (AIME, etc.) and $B = 512$ (MATH 500, DS‑Llama to ensure compression is exercised). Generation length is capped at 32 768 tokens, temperature 0.6, top‑p 0.95.

- **Cross‑validation / statistical protocol.** No explicit cross‑validation on the test benchmarks is reported; the method’s hyper‑parameters (e.g., calibration settings) are fixed based on the calibration dataset. TriAttention’s offline statistics are extracted from a calibration corpus (e.g., ShareGPT, 200 K–960 K tokens) completely disjoint from the evaluation tasks; the stability of the statistics is confirmed by experiments varying the calibration data domain (Table 3C) and quantity (Appendix H). For each AIME problem, 8 samples are drawn to reduce variance; for MATH 500 only one sample is used per problem. No confidence intervals or error bars are provided, so the observed differences should be interpreted with that limitation in mind.

---

### Main Quantitative Results

#### Reasoning Task Accuracy (AIME, MATH 500)

**Tables 1 and 2** report accuracy on AIME24, AIME25, and MATH 500 across all models, using a fixed KV budget (2048 for AIME, 512 for MATH 500 except as noted). The headline finding is that **TriAttention consistently achieves the highest accuracy among compression methods**, often closing most of the gap to Full Attention while substantially outperforming SnapKV and R‑KV.

- On **AIME25** (Qwen3‑8B, budget 2048): Full Attention achieves 40.8 %; TriAttention reaches **32.9 %**, SnapKV 20.0 %, R‑KV 17.5 %. TriAttention thus more than doubles the pass rate of the next‑best competitor (Table 1).
- On **AIME24** (Qwen3‑8B, budget 2048): TriAttention 42.1 % vs. R‑KV 25.4 % and SnapKV 34.6 %. Full Attention is at 57.1 %.
- On **MATH 500** (Qwen3‑8B, budget 512): TriAttention obtains 56.0 %, Full Attention 69.6 %, R‑KV 46.4 %, SnapKV 49.2 % (Table 2). With a budget of 1024, TriAttention reaches 68.4 %, nearly matching Full Attention (69.6 %), as shown in the budget sweep of **Figure 5C**.
- Across the other three models (DeepSeek‑Distill‑Llama‑8B, DeepSeek‑Distill‑Qwen‑7B, GPT‑OSS‑20B), TriAttention also leads all compression baselines on AIME24 and AIME25, with the gap being particularly large on GPT‑OSS‑20B (TriAttention 59.2 % vs. R‑KV 49.6 % on AIME24).

**Figure 5** plots accuracy versus KV budget for Qwen3‑8B on the three reasoning benchmarks. TriAttention dominates R‑KV at every budget, and the advantage is most pronounced at low‑to‑mid budgets. On MATH 500, TriAttention **matches Full Attention at budget 1024 and slightly exceeds it** at higher budgets (Figure 5C). On AIME25, at budget 2048 TriAttention reaches 32.9 % vs. R‑KV’s 17.5 %; with budget 4096 it climbs to 43.3 %, surpassing the Full Attention baseline (40.8 %). This demonstrates that TriAttention not only compresses without severe degradation but can even slightly outperform Full Attention when a modest budget is provided, possibly because pruning occasionally removes noise.

#### Memory Retention (Recursive State Query Benchmark)

The DFS‑based Recursive State Query benchmark (Appendix C) is specifically designed to stress the ability of a KV‑cache compressor to retain intermediate states during backtracking. **Figure 5D** reports accuracy (stack‑exact‑match) on Qwen3‑8B for depths from 6 to 20 steps, comparing Full Attention, R‑KV, and TriAttention (both compression methods use budget 2048).

- At low‑to‑moderate depths (6–14), TriAttention performs **comparably to Full Attention** and slightly outperforms it at depths 8 and 12.
- At depth 16 and beyond, only TriAttention maintains high accuracy; **R‑KV suffers catastrophic degradation**, falling from ∼61 % at depth 14 to ∼31 % at depth 16, while TriAttention remains above 60 %.
- This sharp drop indicates that R‑KV’s observation‑window‑based pruning consistently evicts intermediate states that are essential for backtracking, whereas TriAttention’s distance‑based scoring retains them predictably.

#### Throughput and Efficiency

The paper quantifies the practical benefit by measuring throughput at configurations where TriAttention achieves accuracy comparable to Full Attention (**Table 4**). All measurements are taken on a single A100 80 GB GPU at maximum batch size.

- **MATH 500:** TriAttention with budget 1024 achieves 68.4 % accuracy (Full Attention 69.6 %) while delivering **6.3× higher throughput** (1405.2  vs.  222.8  tokens/s). The KV cache memory is reduced by a factor of roughly 32 K / 1024 ≈ 31 × (the paper does not explicitly state this factor for MATH 500, but the throughput gain includes the memory benefit).
- **AIME25:** TriAttention with budget 3072 matches Full Attention at 40.8 % and yields **2.5× higher throughput** (563.5  vs.  222.8  tokens/s). The KV memory reduction factor is 32 768 / 3072 ≈ **10.7×**, as quoted in the abstract and **Figure 1**.
- **AIME24:** with budget 4096, accuracy is 54.6 % (vs. Full 57.1 %) and throughput improves 1.9× (413.9  vs.  222.8  tokens/s). This configuration is slightly below Full Attention accuracy; the 2.5× throughput achievement is reported for AIME25 where equality is strictly met.

**Table 5** compares TriAttention and R‑KV directly under two regimes.  
- **Comparable accuracy:** TriAttention reaches the same performance as R‑KV with **half the KV budget** (1024 vs. 2048 on MATH 500 and AIME24) while achieving **85 % higher throughput** (1405.2  vs.  760.4  tokens/s on MATH 500).  
- **Comparable memory (same budget):** at budget 1024 (the common setting where both methods get the same memory), TriAttention improves MATH 500 accuracy by **+8.0 %** (68.4 % vs. 60.4 %) and AIME24 accuracy by **+15.4 %** (25.8 % vs. 10.4 %), with throughput being nearly equal (1405.2  vs.  1345.5  tokens/s).  

These numbers demonstrate that TriAttention shifts the entire accuracy–efficiency Pareto front upward relative to R‑KV.

#### Results on LongBench and RULER (Generalisation)

The main paper references additional results on **LongBench** (16 diverse subtasks, 50 % KV budget) and **RULER** (retrieval tasks, 4 K context) to verify that the gains are not specific to math reasoning. In **Appendix F (Table B)** TriAttention achieves the highest average LongBench score (48.1) among compression methods, surpassing Ada‑KV+SnapKV (45.6), SnapKV (45.2), and PyramidKV (42.7). It wins on 11 of 16 subtasks. On RULER (**Table C**), TriAttention scores **66.1** vs. SnapKV’s 55.6 and StreamingLLM’s 61.1. A separate comparison with H2O on the 12 LongBench subtasks where H2O can fit in 48 GB (**Table D**) shows TriAttention wins 10 out of 12 subtasks (average 45.4 vs. 41.4). While these results are in the appendix and not as extensively analysed, they corroborate that TriAttention’s pre‑RoPE scoring generalises beyond the AIME/MATH domain.

---

### Ablation Studies and Robustness Checks

**Removal of the trigonometric series score $S_{\text{trig}}$ (Table 3A):** When the trigonometric series term is omitted and only the norm‑based score $S_{\text{norm}}$ is used, AIME24 accuracy collapses from 42.1 % to **18.8 %** and AIME25 from 32.9 % to **21.2 %**. This confirms that the distance‑preference signal captured by the centres is the dominant source of key importance, not just norm magnitude.

**Removal of the norm‑based score:** In text the paper states that removing $S_{\text{norm}}$ and relying solely on $S_{\text{trig}}$ “drops AIME24 accuracy from 45.8 % to 40.4 %” (this 45.8 % corresponds to a configuration with larger offsets; the exact numbers vary but the drop is ∼5.4 %). Thus the norm term provides a complementary, albeit smaller, benefit.

**Concentration‑based weighting (Table 3B):** Replacing the adaptive weighting $(1-R_f)$ with the base norm score $S_{\text{norm}}^{(0)}$ (which does not down‑weight concentrated bands) reduces AIME24 accuracy from 42.1 % to 41.3 % and AIME25 accuracy from 32.9 % to **28.7 %** (a 4.2‑percentage‑point drop on the harder benchmark). The weighting is especially valuable in harder tasks where noisy norm information in concentrated heads would otherwise hurt the combined score.

**Cross‑domain calibration (Table 3C):** Offline statistics collected on coding data (LiveCodeBench; Jain et al., 2025) instead of reasoning data yield AIME24 accuracy of 44.2 % vs. 42.1 % (reasoning calibration) and AIME25 accuracy of 29.2 % vs. 32.9 %. The small differences (sometimes even favouring coding calibration) confirm that the Q/K centres are model‑intrinsic and not overfit to any specific task domain. This is a practically important result because it eliminates the need for per‑task calibration.

**Future offset design (Appendix G, Table E):**  
- Increasing the maximum offset distance from 128 to 4096 (with denser sampling) raises AIME24 accuracy from 41.7 % to **48.8 %** (+7.1 %), proving that long‑range future queries contribute meaningful importance information.  
- Using **geometric spacing** {1, 2, 4, …} instead of **linear spacing** for the offsets causes a dramatic performance difference: 45.8 % vs. 28.7 % (−17.1 %). Near‑distance regions need finer sampling because the trigonometric series changes more rapidly there; linear spacing severely underestimates the importance of nearby future positions.

**Calibration data quantity and quality (Appendix H, Table F):**  
- Performance is stable across calibration sizes from 50 K to 960 K tokens: 45.4 %, 45.8 %, 45.8 % on AIME24.  
- Calibration on low‑quality data (Google homepage HTML) achieves 46.2 %, comparable to high‑quality chat data (46.7 %). This demonstrates that the Q/K centres are robust to the choice of calibration data, consistent with the claim that they are a model‑intrinsic property.

**MLA architecture validation (Appendix I, Table G):** TriAttention’s assumptions are verified on the GLM‑4.7‑Flash model, which uses Multi‑head Latent Attention (MLA). The Q/K concentration (MRL) is even stronger in MLA: 96.6 % of heads have $R > 0.95$ vs. 84.7 % in Qwen3‑8B (GQA). Reconstruction correlation remains comparable, confirming that the distance‑preference mechanism is architecture‑general.

**Comparisons beyond the primary baselines (Appendix E, Table A):** On AIME24 with DeepSeek‑R1‑Distill‑Qwen‑7B, TriAttention is compared at varying budgets against LazyEviction, H2O, TOVA, and RaaS. TriAttention outperforms all methods at every budget, and at 30 % KV budget it **matches Full Attention** (46.7 %). This additional head‑to‑head reinforces the claim that the method surpasses a wider range of competitors.

**OpenClaw deployment on a single consumer GPU (Appendix J):** A qualitative demonstration shows that TriAttention enables the OpenClaw multi‑turn agent (Qwen3‑32B, INT4) to run on an RTX 4090 (24 GB) without out‑of‑memory errors, whereas Full Attention fails. This validates the practical deployment claim, albeit without a formal accuracy metric for the agent.

---

### Critical Assessment

The experiments collectively provide strong evidence that TriAttention compresses the KV cache with substantially less reasoning degradation than prior methods, and that the underlying Q/K‑concentration and trigonometric‑series mechanism is responsible for the gain. However, several limitations in the experimental design temper the generality and precision of the conclusions.

- **Are the headline throughput and memory claims rigorously supported?** The statements “2.5× higher throughput” and “10.7× KV memory reduction” are specific to the AIME25 setting where TriAttention with budget 3072 exactly matches Full Attention’s 40.8 % accuracy (Table 4, Figure 1). This is a single operating point; the throughput advantage varies across benchmarks (6.3× on MATH 500, 1.9× on AIME24). The paper carefully notes the conditions, so the claims are accurate for those conditions, but one should not extrapolate a universal factor. Moreover, the throughput measurement uses a batch size that fills the GPU; in a deployment with varying batch sizes or lower GPU utilisation, the relative speedup may differ.

- **Is the “matches Full Attention” claim adequately tested?** On AIME25, the accuracy match is exact (both 40.8 %), but with only 240 samples (30 problems × 8) the observed equality could be coincidental. Confidence intervals are not reported, so we cannot assess whether TriAttention’s true accuracy might be slightly lower or higher. On MATH 500, TriAttention at budget 1024 achieves 68.4 % vs. Full 69.6 %—the gap is 1.2 percentage points but is presented as “closely matching”; the statistical significance of this small gap is unknown. The robustness would be stronger if multiple seeds or wider error bars were shown.

- **Do the experiments establish that TriAttention solves the instability of observation‑window methods?** The ablation on Strig (Table 3A) and the memory‑retention benchmark (Figure 5D) are the most direct evidence. Removing Strig (leaving only norm‑based scoring) causes a drastic accuracy drop, and R‑KV’s catastrophic failure on DFS recursion contrasts sharply with TriAttention’s stability. These results strongly support the claim that the trigonometric‑series signal is the critical component and that observation‑window methods suffer from state loss that TriAttention avoids. However, the DFS benchmark is a synthetic stress test, not a natural reasoning task. The paper does not isolate whether the failure of R‑KV on AIME is due specifically to retrieval‑head issues—it is plausible but not proven.

- **Generalisation beyond mathematical reasoning:** The LongBench and RULER results (Appendix F) extend the evidence to summarisation, QA, dialogue, retrieval, and code tasks. TriAttention leads compression methods there, but the margins are smaller than on AIME (e.g., LongBench average 48.1 vs. 45.6 for Ada‑KV+SnapKV). These results indicate that TriAttention still benefits general tasks, but the dramatic improvements on reasoning may partly reflect that reasoning has longer and more structured dependencies where distance‑based scoring is especially advantageous. The paper does not evaluate on extremely long contexts (e.g., 128 K tokens) or on multi‑turn agent benchmarks with quantitative success metrics; the OpenClaw demo is promising but lacks a controlled accuracy comparison.

- **Single‑model family for most ablations:** The core ablations (Table 3, Appendix G–H) are performed solely on Qwen3‑8B. While cross‑model validation (Tables 1–2) shows TriAttention consistently outperforming baselines on three other models, the detailed understanding of how the mechanism behaves (e.g., the relative importance of Strig vs. Snorm across architectures) is only characterised for one model. This is a typical limitation but worth noting.

- **Potential confounding in the budget sweep:** In Figure 5, TriAttention occasionally exceeds Full Attention at high budgets (e.g., AIME25 at budget 4096). This could indicate that pruning acts as a regulariser that removes attention noise, but it could also be statistical noise given the small test set. The paper does not discuss this phenomenon, and it remains unclear whether it is a reliable property.

- **Missing baselines and ablations:** The paper does not compare against a simple “distance‑based” baseline that retains keys uniformly at the positions where the trigonometric series peaks (without using key‑specific norms). Such a baseline would isolate whether the centre‑driven distance preference alone, without key‑content discrimination, is sufficient. Additionally, there is no ablation that replaces the Q centre with a random constant to confirm that the actual direction of the centre matters (though the high reconstruction correlation implicitly does). Finally, the scoring function uses a fixed geometric offset set; sensitivity to the precise offset numbers (e.g., using only powers of 2 vs. a different progression) is only partially explored (linear vs. geometric spacing, but not, say, logarithmic spacing with different base).

- **Limited scale of test sets:** AIME24 and AIME25 each contain 30 problems; even with 8 samples, the confidence interval around accuracy estimates is wide. The performance advantage on AIME25 (TriAttention 32.9 % vs. R‑KV 17.5 %) is large enough to be convincing, but the precise margin has high variance. MATH 500 with 500 problems gives more stable estimates, but TriAttention’s advantage there is smaller relative to R‑KV (56.0 % vs. 46.4 %) and is subject to the same lack of error bars.

In summary, the experiments convincingly demonstrate that TriAttention substantially outperforms prior KV‑cache compression methods on mathematical reasoning benchmarks and on a dedicated memory‑retention test, while achieving practical throughput improvements. The ablation studies firmly tie the gains to the trigonometric‑series component and the concentration‑based weighting. However, the evidence for the precise efficiency factors, the exact magnitude of the advantage on broader tasks, and the statistical robustness of the accuracy equality with Full Attention is less definitive due to the small test sets and unreported uncertainties. These are typical limitations of an initial empirical study and do not undermine the core contribution, but they indicate where more extensive evaluation would strengthen the results.
## 6. Limitations and Trade-offs

### 6.1 Small Benchmark Sizes and Absence of Confidence Intervals

**The assumption or constraint.** The primary reasoning benchmarks, AIME24 and AIME25, each contain only 30 problems. Accuracy is estimated as the average pass rate over 8 samples per problem, but no confidence intervals, standard errors, or statistical tests are reported. The paper states that on AIME25 TriAttention matches Full Attention exactly at 40.8 % (Table 4), and on MATH 500 it “closely matches” (68.4 % vs. 69.6 %).

**The consequence.** With a 30‑problem test set, an observed accuracy difference of several percentage points can easily arise from sampling noise, especially when the per‑problem pass rate is estimated from only 8 draws. The claim of “matching Full Attention” is thus statistically unvalidated; the true accuracy of TriAttention at the chosen budget could be meaningfully lower or higher. The same uncertainty affects the ablation experiments (Table 3), where performance drops of a few percentage points are interpreted as meaningful but could be within the test‑set variance. Practitioners cannot reliably determine whether a particular deployment will see the reported gains or whether a more expensive configuration is actually necessary.

**What evidence exists in the paper.** The paper does not report any measure of variability (error bars, confidence intervals, or significance tests) in any benchmark or ablation table. The AIME24/AIME25 results in Tables 1 and 4, the budget sweeps in Figure 5, and the ablation in Table 3 all present single‑point estimates without uncertainty.

**Mitigation status.** No mitigation is attempted; the limitation is not discussed. Future work would benefit from reporting 95% binomial confidence intervals (easily computable from the sample sizes) and, ideally, from testing on a larger evaluation corpus, though AIME’s fixed size is inherent.

---

### 6.2 Deployment Difficulty: Requirement of Pre‑RoPE Vector Access

**The assumption or constraint.** TriAttention’s scoring function requires access to the pre‑RoPE query and key vectors—the raw vectors before the positional rotation is applied—as well as the per‑head statistics $\mathbb{E}[\mathbf{q}_f]$, $R_f$, etc. Standard inference engines (vLLM, TensorRT‑LLM, Hugging Face Transformers with FlashAttention‑2) typically compute, store, and compress the *post‑RoPE* keys. The paper does not detail how the pre‑RoPE vectors are extracted during inference, nor does it discuss the engineering effort needed to expose them without slowing down the core attention kernel.

**The consequence.** Adopting TriAttention in existing production stacks is likely to require non‑trivial modifications to the model code, potentially breaking optimised fused attention kernels that consume post‑RoPE keys internally. The calibration step is straightforward (it can be run offline), but the online scoring pass must intercept Q and K before the RoPE rotation, which may not be readily available after the rotation and caching have already happened. Without a well‑engineered solution, the method may be limited to research prototypes that modify the transformer implementation, reducing its practical impact despite the reported throughput improvements.

**What evidence exists in the paper.** Section 4 mentions “we directly use their representations $\mathbf{k}_f$” but does not describe the inference‑time hook. The throughput measurements (Section 5.2.4) do include the overhead of scoring and pruning, but they are obtained in a setting where pre‑RoPE access is presumably implemented; the difficulty of achieving this in other frameworks is not discussed. Appendix J’s OpenClaw demo runs a 32B‑parameter model on an RTX 4090, but again no details about code modifications are given.

**Mitigation status.** The paper does not acknowledge this as a limitation. The future‑work paragraph (Appendix A) only mentions “a dedicated, hardware‑aware inference kernel” for reducing *latency* of the trigonometric series computation, not for the general problem of pre‑RoPE access. A practical mitigation would be to show a reference integration with a popular inference framework and to quantify the total lines of code change required.

---

### 6.3 Unexplored Overhead Scaling and Per‑Step Latency

**The assumption or constraint.** TriAttention prunes the KV cache once every β = 128 tokens, scoring every remaining key independently and retaining the top‑B. The end‑to‑end throughput is measured using this schedule, and the reported speedups (2.5× on AIME25, 6.3× on MATH 500, Tables 4 and 5) incorporate this overhead. However, the paper does **not** separately analyse how the scoring cost grows with the number of cached keys or the sequence length, nor does it report per‑token latency (only overall tokens per second).

**The consequence.** For sequences much longer than the tested maximum (32 K tokens) or with larger KV budgets, the scoring pass could become a noticeable fraction of inference time, potentially eroding the throughput advantage. Moreover, the periodic scoring introduces a burst of compute every 128 tokens; if this burst creates jitter, it could degrade interactive latencies, which matters for real‑time applications even if average throughput remains high. Since the scaling behavior of the scoring cost is not characterised, practitioners cannot extrapolate the reported efficiency gains to contexts of, say, 128 K tokens or to hardware configurations with different compute–memory ratios.

**What evidence exists in the paper.** All throughput numbers are taken on an A100 80 GB with 16 K‑decoding‑length measurements. The maximum batch size is chosen to fill memory after compression; details of how the pruning overhead varies with sequence length, budget, or batch size are absent. The supplementary material (Appendix G) shows that using more future offsets improves accuracy, but it does not report the corresponding increase in latency.

**Mitigation status.** The paper acknowledges that future work could design a dedicated kernel to “further accelerate the computation of trigonometric series and the subsequent cache pruning process” (Appendix A), but it does not provide any latency profiling of the current Python‑level implementation. Without a microbenchmark, the cost‑benefit of the method at extreme scales remains unquantified.

---

### 6.4 Uniform KV Budget Across Layers and Heads

**The assumption or constraint.** TriAttention applies the same fixed memory budget B to every attention head in every layer. The scoring function is computed identically across all heads, and the same number of keys is retained per head. This is a deliberate simplification to facilitate comparison with prior work that also uses uniform budgets (e.g., R‑KV, SnapKV).

**The consequence.** In reality, different heads exhibit very different attention patterns (local, retrieval, sink, etc.) and vary widely in how much compression they can tolerate. Forcing the same budget onto a head that primarily attends to a sliding window and onto a head that must retain long‑range retrieval tokens is likely suboptimal: some heads are allocated memory they do not need, while others are starved. An adaptive, per‑head budget allocation could yield additional accuracy for the same total memory, or the same accuracy with a smaller total budget.

**What evidence exists in the paper.** The paper does not evaluate any per‑head budget variation. The ablation on concentration‑based weighting (Table 3B) adjusts the relative importance of the two scoring terms per band but does not change the number of retained keys per head. The cross‑architecture analysis (Appendix I) shows that concentration varies across heads, but no experiments test whether redistributing the budget accordingly improves performance.

**Mitigation status.** The paper explicitly mentions “head‑specific budgets” as a direction for future work (Appendix A). The limitation is recognised but not addressed in the present study.

---

### 6.5 Degradation Under Extreme Memory‑Retention Pressure

**The assumption or constraint.** TriAttention’s scoring function predicts key importance from pre‑RoPE centres and approximates future queries as their mean. This approximation holds best when the actual queries are tightly concentrated around their centres; i.e., when $R_f$ is very high. On tasks that require maintaining a large number of diverse intermediate states over very long distances, even a small amount of query variance can accumulate errors, causing some critical keys to receive scores that are slightly too low and, eventually, to be evicted.

**The consequence.** The paper demonstrates this limit with the Recursive State Query benchmark (Figure 5D): beyond a recursion depth of 18, TriAttention begins to lag behind Full Attention, while R‑KV already fails catastrophically at depth 16. Although TriAttention is substantially more robust than prior methods, it still loses some information when the memory pressure becomes extreme. For tasks with exceptionally long reasoning chains (e.g., very deep recursive algorithms, multi‑step planning over thousands of tokens), the gap to Full Attention may widen further, though the paper does not test depths beyond 20.

**What evidence exists in the paper.** Figure 5D clearly shows that at depth 20 (the maximum tested), TriAttention’s accuracy drops to approximately 55 % (estimated from the plot), whereas Full Attention remains above 70 %. The text notes that “only beyond depth 18 does TriAttention begin to lag behind,” but the lag is not quantified with a table.

**Mitigation status.** The paper does not suggest a specific mitigation for this regime, beyond the generic future work on refined compression strategies (Appendix A). One could imagine increasing the budget for particularly hard tasks or using a dynamic budget, but those approaches are not explored.

---

### 6.6 Generalisation Beyond Mathematics‑Oriented Long‑Reasoning

**The assumption or constraint.** The strongest improvements are demonstrated on competition‑level mathematical reasoning benchmarks (AIME, MATH 500). While the appendix includes LongBench and RULER results (Appendix F), the accuracy gains over compression baselines are notably smaller on those broader tasks. For example, on LongBench the average improvement over the next‑best compression method (Ada‑KV+SnapKV) is 2.5 points (48.1 vs. 45.6, Table B), whereas on AIME25 the gap over R‑KV is 15.4 percentage points. The paper does not analyse why TriAttention’s distance‑preference scoring is particularly beneficial for mathematical reasoning or whether it transfers equally to tasks like multi‑turn dialogue, factual news QA, or code summarisation where attention patterns may be less distance‑structured.

**The consequence.** A practitioner deploying TriAttention on a general‑purpose assistant that handles a mix of reasoning, retrieval, summarisation, and conversation might see a less dramatic accuracy retention than the headline AIME numbers suggest. The method may be highly effective for certain long‑reasoning workloads but only moderately beneficial for others. Without a systematic investigation of task characteristics that correlate with TriAttention’s advantage, it is hard to predict its performance on an unseen task.

**What evidence exists in the paper.** The LongBench subtask breakdown (Table B) shows that TriAttention wins on 11 of 16 subtasks but that its lead is often narrow, and on some subtasks (e.g., 2Wiki, PaRe, LCC) it slightly trails a competitor. The RULER retrieval result (66.1) is stronger, but the paper does not report TriAttention’s accuracy relative to Full Attention on these long‑context benchmarks, which makes it impossible to gauge the absolute compression cost.

**Mitigation status.** The limitation is not discussed directly, but Section 8 and Appendix A mention future evaluation on “broader domains such as coding and agentic tasks.” Whether the method’s design will prove equally effective for those domains remains an open question.
## 7. Implications and Future Directions

### How This Work Changes the Landscape

TriAttention introduces a **conceptual reorientation** in how we approach KV cache compression: from reactive observation to geometric prediction. The dominant paradigm in the field—exemplified by H2O, SnapKV, R-KV, and related methods—has been to estimate key importance by observing attention scores from recent queries, effectively treating the attention mechanism as a black box whose output must be monitored at runtime. This paper demonstrates that this paradigm is fundamentally constrained: RoPE's position-dependent rotation means the set of queries with the "correct" orientation is tiny (about 25 queries, after which older queries act as noise), creating an information bottleneck that no amount of window-extension can resolve.

The shift is to **predict importance from the static, pre-RoPE geometry of the model itself**—specifically, from the centers around which query and key vectors cluster in the pre-rotation space. This is not an incremental refinement of observation-window methods; it is a different category of solution entirely. The paper's core discovery—that 84–96% of attention heads exhibit Q/K concentration with Mean Resultant Lengths above 0.95 (Figure 2C, Appendix I Table G)—means that a large fraction of attention behavior is encoded in per-head constants that are fixed after pretraining. This is analogous to the shift in pretraining from "train bigger models" to "follow scaling laws": it transforms what was an empirical tuning problem into one with a principled, analytical foundation.

The reconciliation of prior contradictions is a significant contribution that extends beyond the method itself. Huang et al. (2023) found that "LLMs cannot self-correct reasoning," while others found conditioning on previous outputs to be effective. TriAttention does not directly address self-correction, but it resolves the analogous contradiction in KV compression: why do observation-window methods sometimes work acceptably (e.g., on short-context QA) but catastrophically fail on long reasoning? The answer is that post-RoPE observation is stable only when the needed information falls within the tiny window of valid queries. In long reasoning chains, where retrieval heads must reach back over thousands of tokens to recover intermediate results, those tokens are invisible to recent queries. TriAttention's distance-based scoring naturally assigns high importance to tokens at the distances where retrieval heads peak, regardless of current attention values, making it inherently robust to the failure mode that cripples competitors. The Recursive State Query benchmark (Figure 5D) makes this concrete: R-KV loses intermediate states at depth 16, suffering a catastrophic accuracy drop from ~61% to ~31%, while TriAttention remains stable.

The paper also **redirects research priorities** in KV cache compression. The finding that geometric offset spacing dramatically outperforms linear spacing (45.8% vs. 28.7% on AIME24, Appendix G Table E) and that extending the maximum offset to 4096 yields +7.1 percentage points over 128 establishes that **long-range positional structure is the dominant signal for key importance**, not recent attention patterns. This implies that future work should invest in better modeling of distance-dependent attention preferences (e.g., through higher-resolution series approximations, learned distance embeddings, or per-head budget allocation) rather than in more sophisticated observation-window strategies, which the paper shows are fundamentally bottlenecked. The field should move from "how can we observe more queries?" to "how can we better predict from pre-RoPE geometry?"

A subtler landscape shift concerns **verifier over-optimization analogies in compression**. Prior work in RLHF and test-time compute scaling identified that verifiers can be exploited by aggressive optimization (the reward hacking phenomenon). TriAttention reveals an analogous principle for KV cache scoring: the trigonometric series alone can overfit to distance preferences and miss content-norm signals, while pure norm-based scoring misses distance structure. The concentration-based weighting (equations 8–9) acts as a calibration mechanism that blends the two signals according to head-level uncertainty, preventing either from dominating where it is unreliable. This suggests a general design principle—**uncertainty-weighted fusion of geometry and content signals**—that could apply beyond KV compression to other attention-engineering tasks.

Finally, the paper makes **pre-RoPE analysis a newly viable diagnostic tool** for understanding attention. Prior mechanistic interpretability work on attention heads (e.g., induction heads, retrieval heads, attention sinks) has largely studied post-RoPE patterns qualitatively. TriAttention provides a quantitative language: for any head, the trigonometric series coefficients $a_f$, $b_f$ from the Q/K centers encode the head's distance preference as a spectral signature. This means researchers can now **classify heads by their center-derived curves** without running a single forward pass on data, and can potentially predict how a head will behave under different positional configurations. This is a genuine diagnostic advance that opens head-level analysis to methods from signal processing and directional statistics.

### Follow-Up Research This Work Enables

- **Joint optimization of pre-RoPE centers during pretraining to produce compression-friendly heads.** TriAttention relies on existing pretrained centers, which were never optimized with compression in mind. A natural extension is to add an auxiliary loss during pretraining that encourages heads to have high Mean Resultant Length $R_f$ (stronger concentration, better trigonometric-series approximation) and/or to learn centers that produce distance curves with clear peaks at the distances most important for downstream tasks. Concretely, one could fine-tune a pretrained model with a regularization term $\lambda \sum_f (1 - R_f)$ on the per-head concentration, then measure whether TriAttention achieves higher accuracy at the same KV budget compared to the unregularized baseline. A positive result would establish that models can be deliberately made "compression-friendly" without sacrificing pretraining quality.

- **Per-head adaptive budget allocation using the concentration metric.** The current method applies a uniform KV budget across all heads, but the paper's own data shows that heads vary in concentration (Figure 2C, Appendix I) and in reconstruction quality (Figure 3). Heads with $\bar{r} > 0.7$ (strong distance-preference predictability) can likely tolerate aggressive compression, while heads with lower $\bar{r}$ or lower $R_f$ may need more keys to maintain accuracy. A follow-up could allocate the total budget $B_{\text{total}}$ across heads proportionally to $(1 - \bar{r})$ or $(1 - R_f)$, then evaluate whether this adaptive allocation improves AIME accuracy over the uniform baseline at an equivalent total memory. The paper already provides the per-head statistics needed to parameterize such an allocation; the experiment would simply replace the uniform top-B selection with per-head top-$B_h$ selection where $\sum_h B_h = B_{\text{total}}$ and $B_h$ is allocated by the chosen heuristic. A negative result (no improvement) would suggest that the uniform budget is already near-optimal, but a positive result could add several percentage points at no additional memory cost.

- **Stress-testing TriAttention on extreme-length reasoning (64K–256K tokens) with systematic retrieval-head analysis.** The paper evaluates at a maximum generation length of 32K tokens, but the motivation emphasizes retrieval heads that must recall information over very long distances. A targeted experiment would use a synthetic retrieval benchmark (e.g., passkey retrieval with the key placed at varying distances from 1K to 128K tokens) and measure per-head attention to determine exactly which heads are retrieval heads, what distances they peak at, and whether TriAttention's predicted distance curve correctly assigns high scores to the passkey position. This would directly validate (or falsify) the claim that TriAttention solves the retrieval-head instability problem. If TriAttention consistently retains the passkey while R-KV/SnapKV evict it at long distances, the mechanistic claim is proven. If TriAttention also fails beyond some distance threshold, the failure mode would pinpoint where the center approximation breaks down and motivate refined modeling (e.g., distance-dependent center drift).

- **Online, dynamic scoring with adaptive offset sets instead of fixed geometric progression.** The current method uses a fixed set of future offsets $D = \{1, 2, 4, \ldots\}$ and averages scores over them. This is a static approximation of the true expected attention over all future positions. A more principled approach would maintain a probability distribution over the remaining sequence length (e.g., from an empirical distribution of generation lengths or a learned predictor) and compute the expected score as a weighted integral over that distribution. Concretely, one could fit a Poisson or negative binomial distribution to the remaining token counts observed during calibration, then replace the uniform average in equation (11) with a weighted average $\sum_{\delta \in D} w(\delta) S(k, \Delta + \delta)$ where $w(\delta)$ is the probability of the sequence extending by at least $\delta$ more tokens. The hypothesis is that this would reduce the overhead of scoring at very long offsets when the sequence is likely to terminate, and improve accuracy by focusing the budget where future queries are most probable.

- **Integrating TriAttention scoring into the FlashAttention kernel pipeline for zero-overhead compression.** The current throughput measurements include Python-level scoring overhead every 128 tokens, but the paper does not profile the scoring cost independently. A systems follow-up would implement the trigonometric-series evaluation and top-B selection as a fused CUDA kernel that operates on the pre-RoPE K vectors immediately after they are computed (before RoPE), amortizing the scoring cost into the existing attention computation. The benchmark would measure end-to-end latency with and without the fused kernel, ideally showing that the pruning overhead becomes negligible (<1% of total inference time). Microbenchmarks would report the latency of the scoring pass as a function of KV cache size (1K to 128K keys) and batch size. A negative result (scoring remains expensive even when fused) would indicate that the trigonometric series computation—which requires per-key, per-band complex arithmetic over d/2 frequency bands—is inherently costly for wide models, motivating a low-rank approximation to the series.

- **Extending the trigonometric series framework to non-RoPE positional encodings and to value vectors.** The mathematical derivation in Section 3 and Appendix B depends specifically on RoPE's rotation-by-position property. However, the broader insight—that pre-encoding vectors may exhibit concentration and encode predictable patterns—could apply to other positional encoding schemes (ALiBi, learned absolute positions, NoPE). A diagnostic experiment would measure Q/K concentration (Mean Resultant Length) in non-RoPE models (e.g., models using ALiBi or no positional encoding at all) and check whether distance-preference curves can still be reconstructed from vector centers. A second extension would apply the same center-based prediction to **value vectors**: if V vectors also exhibit concentration around centers, then the attention-weighted output (which depends on V norms) could be predicted before full attention computation, enabling a different form of sparse attention. The paper's norm-based score $S_{\text{norm}}$ already uses key norms; extending to value norms and value-direction centers would be a natural generalization that the current work explicitly leaves unexplored.

### Practical Applications and Downstream Use Cases

- **Deploying large reasoning models on consumer GPUs for interactive agents.** Appendix J demonstrates that TriAttention enables OpenClaw (a multi-turn agent using Qwen3-32B at INT4 precision) to complete a document-processing task on a single RTX 4090 (24 GB) without out-of-memory errors, whereas Full Attention fails. This is not a synthetic benchmark—it is a real agent that reads multiple documents, processes instructions, and generates a report over multiple interaction rounds, each extending the context. The practical implication is that developers can now run 32B-parameter reasoning models locally on hardware that costs under $2,000, where previously a datacenter GPU (A100 80GB) or cloud API was required. The specific benefit is **feasibility**: without TriAttention, the task crashes; with it, the task completes within the GPU memory budget. Quantitatively, the KV cache compression keeps memory usage bounded at the chosen budget (the paper does not specify the exact budget for the OpenClaw demo, but the principle is that the cache size is capped at B tokens per head rather than growing to the full context length). For any application involving long multi-turn interactions—legal document analysis, codebase exploration, research literature review—this enables offline, private, low-cost deployment.

- **Cost-efficient batch inference for mathematical reasoning at scale.** The throughput measurements in Table 4 provide a direct cost model: on MATH 500, TriAttention with a budget of 1024 achieves 6.3× higher throughput than Full Attention (1405.2 vs. 222.8 tokens/s) while maintaining 68.4% accuracy (vs. 69.6%). For a service that evaluates thousands of math problems per day, this translates to running roughly 1/6 the number of GPU-hours for the same accuracy. On AIME25, the 2.5× throughput gain (563.5 vs. 222.8 tokens/s) at matched accuracy (40.8%) similarly reduces compute cost by 60%. Organizations running large-scale reasoning evaluations—educational platforms, competition grading services, or LLM benchmarking pipelines—can adopt TriAttention as a drop-in replacement for Full Attention with no accuracy loss (by selecting the appropriate budget from Figure 5) and immediate cost savings proportional to the throughput multiplier. The calibration is a one-time cost (200K–960K tokens of any text), and the cross-domain robustness (Table 3C) means the same calibration works for all downstream math tasks.

- **Enabling long-context reasoning on memory-constrained edge devices.** The 10.7× KV memory reduction on AIME25 (from 32K tokens to 3072 budget) means that a device with, say, 8 GB of RAM could theoretically handle a model whose full KV cache would require 10.7× more memory—moving from infeasible to feasible. While the paper's experiments use server GPUs (A100), the memory reduction factor is architecture-independent and applies directly to edge hardware (Jetson, Apple Silicon, mobile GPUs) where memory is the binding constraint. The use case is on-device reasoning assistants that need to perform multi-step problem solving without network latency or privacy concerns. The key metric is not throughput but peak memory: by capping the KV cache at B tokens, the maximum memory footprint becomes predictable and bounded, enabling deterministic deployment planning. The paper does not benchmark edge hardware, but the memory reduction is a direct consequence of the budget B, which is a user-configurable parameter.

- **Improving training data generation pipelines through better KV retention during long rollouts.** When LLMs are used to generate training data for self-improvement (e.g., in STaR or ReST$^{\text{EM}}$ pipelines), the generator model must produce long, coherent reasoning chains where every step is logically dependent on earlier steps. If the KV cache evicts an intermediate state, the generated solution may contain logical errors that contaminate the training set. The Recursive State Query benchmark (Figure 5D) shows that R-KV suffers catastrophic degradation on depth-16 DFS recursion (accuracy drops from ~61% to ~31%), while TriAttention remains above 60%. This reliability advantage directly applies to data generation: using TriAttention as the inference engine for the generator model reduces the risk of producing corrupted reasoning traces, improving data quality in automated self-improvement loops. The practical setting is scalable: a training pipeline that previously used Full Attention for quality could switch to TriAttention (with a budget chosen to match Full Attention accuracy, as in Figure 5C for MATH 500) and benefit from the 6.3× throughput gain while maintaining generation quality, reducing the total cost of data generation by the same factor.
