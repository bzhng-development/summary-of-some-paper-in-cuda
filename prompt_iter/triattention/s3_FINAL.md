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