# Muon Outperforms Adam in Tail-End Associative Memory Learning

**ArXiv:** [2509.26030](https://arxiv.org/abs/2509.26030)

## 🎯 Pitch

This paper demystifies why the Muon optimizer surpasses Adam in training transformers by pinpointing its primary advantage to the associative memory components—specifically, the attention value/output matrices and feed-forward networks. Through targeted ablation, spectral analysis, heavy-tailed learning tasks, and theory, the authors show that Muon updates yield more balanced and isotropic representations, enabling more effective learning of rare 'tail' knowledge and reducing biases in large language models. This insight bridges a critical gap in optimizer understanding and paves the way for fairer, stronger, and more reliable AI systems.

---

## 1. Executive Summary

This paper studies why the Muon optimizer consistently outperforms Adam in transformer training by identifying the mechanism through the lens of associative memory. Through component-level ablation on a 160M NanoGPT model trained on FineWeb and a heavy-tailed synthetic knowledge task, the authors reveal that Muon's superiority stems primarily from its effect on the Value-Output (VO) attention weights and Feed-Forward Networks (FFNs)—the transformer components that serve as linear associative memories. The paper establishes a **balanced learning of tail classes in heavy-tailed distributions** as Muon's core advantage: its update rule normalizes gradient singular values to be isotropic, which prevents frequent (head) facts from dominating and enables more effective learning of infrequent (tail) facts compared to Adam—demonstrated by Muon achieving 97.6% first-token accuracy on extreme tail classes versus Adam's 26.4% at 10,000 training steps. The analysis theoretically proves in a one-layer associative memory model that Muon maintains balanced learning across classes regardless of feature embeddings, while Adam's performance is unstable and embedding-dependent, establishing that Muon's alignment with the outer-product structure of linear associative memories enables uniform optimization only when the optimizer respects the matrix structure of the parameters rather than treating them as flat vectors.

## 2. Context and Motivation

### The Core Problem: Muon Works, But Nobody Knows Why

The Muon optimizer, introduced by Jordan et al. (2024), has demonstrated a striking empirical result: it trains transformers **approximately 2× faster** than Adam across a wide range of model sizes and architectures (Liu et al., 2025). This is not a marginal improvement — it represents a potential halving of training compute for large language models, which is enormously consequential given the computational cost of modern LLM training. However, the mechanism underlying Muon's superiority has remained fundamentally unexplained.

The existing theoretical understanding of Muon frames it as steepest gradient descent with respect to the matrix spectral norm (Bernstein & Newhouse, 2024). Concretely, while Adam can be interpreted as steepest descent under the vector ℓ∞ norm (each parameter moves with the same magnitude based on its gradient's sign — see Appendix A), Muon performs steepest descent under the matrix ℓ₂→ℓ₂ operator norm, which yields an update direction equal to the sum of normalized orthogonal factors of the gradient matrix: $O_t = U_t V_t^\top$ where $G_t = U_t S_t V_t^\top$ is the SVD of the momentum-accumulated gradient.

This norm-based interpretation, while mathematically elegant, suffers from a critical gap: **it does not explain why optimizing with respect to the spectral norm should produce better results than optimizing with respect to the infinity norm**. Both are legitimate norms — both provide valid optimization directions — but the framework offers no insight into why the matrix-structured version consistently wins. As the authors put it directly:

> "this perspective alone does not explain why using the matrix operator norm rather than the vector norm leads to better performance"

This gap matters because without understanding *why* Muon works, practitioners cannot predict *when* it will work. They cannot confidently extend Muon to new architectures, new modalities, or new training paradigms. They cannot design principled hybrid optimizers that selectively apply Muon's update rule where it helps most. And they cannot identify the fundamental limitations of the approach. The field is left with an empirically powerful tool whose mechanism is a black box.

### Why This Matters: The Economic and Scientific Stakes

The practical stakes are immediate and substantial. Training frontier LLMs costs tens to hundreds of millions of dollars in compute. A 2× reduction in training time translates directly to millions of dollars saved per training run, or equivalently, the ability to train models twice as large on the same budget. However, deploying an unexplained optimizer in production training pipelines carries significant risk — if the mechanism is not understood, failure modes cannot be anticipated. An optimizer that works brilliantly on one architecture or data distribution might collapse on another, and without mechanistic insight, the collapse would be a surprise.

Beyond practical economics, there is a deeper scientific question at stake. The fact that a matrix-structured update — one that treats weight matrices as *matrices* rather than flattened vectors — consistently outperforms element-wise adaptive methods in transformers suggests that **the matrix structure of transformer parameters carries exploitable information that element-wise optimizers discard**. Understanding exactly what structure matters and why would illuminate fundamental properties of how transformers learn, potentially leading to more principled architecture designs that bake this structure in from the start rather than recovering it through careful optimization.

The paper frames this as a mystery about the relationship between optimizer design and transformer architecture:

> "This paper takes the first step toward understanding the mechanisms underlying Muon's superiority over Adam in training LLMs."

The phrase "first step" is carefully chosen — this is not presented as a complete theory, but as an initial mechanistic explanation that opens up a new direction of inquiry.

### Conflicting Perspectives in Prior Work

The literature on Muon and related structured optimizers has developed along several largely disconnected lines, each providing partial insight but none offering a unified explanation.

**The steepest descent interpretation** (Bernstein & Newhouse, 2024) provides mathematical grounding for Muon's update rule but, as discussed above, cannot explain its empirical advantage. It shows *what* Muon computes, but not *why* that computation helps. Subsequent convergence analyses derived from this perspective (Li & Hong, 2025; Shen et al., 2025) have established that Muon converges in standard settings, but these analyses do not demonstrate any convergence *rate advantage* over Adam — they simply show Muon is not pathological. As the paper notes:

> "convergence analyses of Muon derived from this interpretation fail to account for its observed superiority over Adam"

**The preconditioning perspective** (Lau et al., 2025; Yang et al., 2023) frames Muon as addressing "gradient anisotropy" rather than "curvature anisotropy" — it normalizes the gradient itself rather than attempting to estimate and invert the Hessian. This distinguishes Muon's mechanism from Adam's, but does not identify *which parameters* benefit from gradient anisotropy correction or *why* transformers specifically would have anisotropic gradients that benefit from normalization.

**Empirical scaling studies** (Liu et al., 2025; Shah et al., 2025; Sato et al., 2025) have systematically demonstrated Muon's advantages across model sizes and explored practical considerations like critical batch size, but these works treat the optimizer as a black box — they measure *that* it works without explaining *how*. Liu et al. (2025) in particular conducted spectral analyses of weight matrices but aggregated parameters across all transformer components, obscuring which specific components drive Muon's behavior. The paper directly critiques this aggregation:

> "We decompose the parameters according to associative memories, whereas Liu et al. (2025) aggregates them, obscuring the essential components driving Muon's behavior."

**Concurrent work on shallow vision architectures** (Vasudeva et al., 2025) studies Muon in a linear regression setting for vision transformers and finds generalization benefits on imbalanced data, but connects these results to standard generalization theory rather than to the specific architectural properties of transformer components. The paper positions its contribution as complementary but distinct:

> "In contrast, we investigate Muon in the context of LLMs, focusing on its effects on associative memory in next-token prediction."

### The Overlooked Connection: Associative Memory as a Unifying Lens

Perhaps the most significant gap the paper identifies is that **no prior work connected Muon's optimization behavior to the associative memory structure of transformer components**. This is a striking omission because the associative memory view of transformers is well-established in a separate strand of literature.

Research on knowledge storage in transformers has converged on the insight that both the attention output weights $W_O$ and the feed-forward output weights $W_{\text{out}}$ function as key-value associative memories. Geva et al. (2020) and Dai et al. (2021) demonstrated that FFN modules store factual knowledge in $W_{\text{out}}$, where each column acts as a key for a stored concept and the corresponding output encodes the associated value. Bietti et al. (2023) extended this to show that the attention output matrix $W_O$ similarly encodes associations between query patterns and output representations. Knowledge editing techniques (Meng et al., 2022a,b; Fang et al., 2024) directly leverage this structure, modifying these specific weight matrices to alter stored facts without retraining.

The critical structural observation is that linear associative memories can be expressed as **sums of outer products**:

$$W = \sum_{i=1}^{K} e_{o_i} e_{s_i}^\top$$

where $\{(e_{s_i}, e_{o_i})\}$ are key-value embedding pairs representing $K$ stored facts, with $e_{s_i}$ encoding subject-relation pairs and $e_{o_i}$ encoding objects.

This outer-product structure has a profound connection to Muon's update rule that — remarkably — had gone unnoticed. Muon's update (without momentum) computes the SVD of the gradient $G = \sum_i s_i u_i v_i^\top$ and forms $O = \sum_i u_i v_i^\top$, updating **all orthogonal singular directions with equal magnitude** regardless of their singular values $s_i$. When the gradient itself decomposes as a sum of outer products (as it naturally does for associative memory parameters), the singular values $s_i$ encode the **frequencies** of the corresponding facts in the training data. Muon's normalization strips out this frequency information, preventing frequently-occurring facts from dominating the update magnitude. Adam's element-wise sign operation, by contrast, does not respect this outer-product structure — it treats each matrix entry independently, which can destroy the alignment between the update direction and the stored fact directions when embeddings overlap (a phenomenon the paper proves formally in Theorem 5.3).

The paper makes this connection explicit:

> "Comparing this with the linear associative memory $\sum_{i=1}^K e_{o_i} e_{s_i}^\top$, we see that Muon updates all 'orthogonal' facts at the same rate. ... the singular values $S$ encode the frequencies of knowledge in the training data ... This implies that Muon can learn both frequent and infrequent facts uniformly."

### The Heavy-Tailed Data Problem: Where Prior Work Falls Short

Real-world language data is notoriously heavy-tailed: a small number of linguistic patterns, facts, and entities appear very frequently, while the vast majority appear rarely. This imbalance creates a fundamental challenge for gradient-based optimizers — gradients from frequent patterns dominate the update, causing the model to learn head classes quickly while tail classes languish.

Prior work by Kunstner et al. (2024) established that **Adam handles heavy-tailed distributions better than SGD** precisely because its element-wise adaptive learning rates prevent the gradients from frequent classes from completely overwhelming those from rare classes. However, this analysis stops at the element-wise level and does not consider whether a matrix-structured optimizer could do even better. The paper's key insight is that while Adam already addresses class imbalance to some degree (by normalizing each parameter independently), it is fundamentally limited because it **treats the weight matrix as a flat vector** rather than respecting the outer-product structure that encodes factual associations.

This limitation shows up concretely in the theoretical analysis (Theorem 5.3): when feature embeddings of different facts overlap in their support (non-zero entries), Adam's element-wise sign operation can produce an update whose singular value spectrum is highly skewed — the ratio of smallest to largest singular value can fall below 25%, meaning some directions in the associative memory space receive much smaller updates than others. Muon, by computing the SVD and normalizing singular values, is immune to this embedding-dependent collapse.

### How This Paper Positions Itself

The paper's positioning is distinctive: rather than proposing a new optimizer or a new convergence proof, it offers a **mechanistic explanation** for an empirically observed phenomenon. The research questions are decompositional and diagnostic:

> "Which transformer components benefit most from Muon's matrix-norm–based optimization compared to Adam?"

> "What structural features of the transformer allow Muon to optimize these components more effectively?"

This positions the work at the intersection of three previously disconnected literatures: (1) optimizer design and analysis (Adam, Muon, steepest descent interpretations), (2) transformer interpretability (associative memory models, knowledge storage, editing), and (3) heavy-tailed learning dynamics (class imbalance, head-tail gaps). By bridging these areas, the paper provides a unified explanation for Muon's superiority that is both theoretically grounded (proven in a tractable one-layer model) and empirically validated (demonstrated through component ablations, spectral analyses on the 160M and 0.7B models, and head-vs-tail performance on a synthetic knowledge task).

The paper acknowledges that this is a first step rather than a complete theory — it deliberately scopes the analysis to associative memory components, sets aside QK parameters as not participating in the same outer-product mechanism, and leaves open the question of whether Muon's advantages extend to higher-order tensor product structures:

> "Intuitively, this property of Muon may extend beyond outer products to higher-order tensor products, an exciting direction for future work."

## 3. Technical Approach

### 3.1 Reader Orientation

This paper is a **mechanistic diagnosis** rather than a new optimizer proposal — it constructs a series of targeted experiments and theoretical models to isolate *why* Muon outperforms Adam in transformer training. The problem being solved is the **black-box nature of Muon's empirical superiority**: the existing steepest-descent interpretation explains what Muon computes (a spectrally-normalized update) but cannot explain why that computation produces better models than Adam's element-wise sign normalization. The solution takes the shape of a **two-part mechanistic explanation**: first, through component-level ablations, identify that Muon's benefits concentrate in the associative memory parameters (VO attention weights and FFN output weights); second, through spectral analysis, knowledge acquisition experiments, and a tractable one-layer theoretical model, demonstrate that the mechanism is **balanced learning across frequencies** — Muon's SVD-based normalization prevents head-class dominance by assigning equal update magnitude to all orthogonal gradient directions, directly counteracting the frequency skew in heavy-tailed training data.

### 3.2 Big-Picture Architecture (Diagram in Words)

The paper's investigative architecture has five major components operating in sequence:

1. **Component Ablation Framework** — a controlled experimental setup that selectively applies Muon vs. Adam to individual transformer blocks (W_Q, W_K, W_V, W_O, W_in, W_out, W_gate) on a 160M NanoGPT model trained on FineWeb, measuring which substitutions recover the full-Muon performance trajectory. This identifies the critical components.

2. **Spectral Analysis Pipeline** — for each weight matrix during training, computes normalized SVD entropy, effective rank, Top-k energy fraction, and eigenvalue quantile ratio to quantify the isotropy (evenness) of the learned singular value spectrum under different optimizers. This characterizes *how* Muon shapes the weights differently from Adam.

3. **Heavy-Tailed Knowledge Task** — a synthetic question-answering benchmark derived from 200,000+ biographical entities where class frequencies follow a power-law distribution, enabling direct measurement of head-class vs. tail-class learning dynamics under different optimizers and under hybrid optimizer configurations (Muon on some components, Adam on others).

4. **One-Layer Associative Memory Model** — a mathematically tractable abstraction consisting of a linear associative memory W ∈ ℝ^(d_o × d_s), orthonormal key embeddings E, value embeddings Ẽ, and a population cross-entropy loss with class frequencies p_k. This model enables closed-form analysis of one-step and multi-step update dynamics for GD, SignGD (Adam without momentum), and Muon, under both support-decoupled and support-coupled embedding regimes.

5. **Theoretical Framework (Theorems 5.3 and 5.4)** — proves bounds on ϱ^ϵ_opt (the minimum correct-class probability across all K facts when at least one fact reaches probability ≥ 1-ε). For Muon, proves ϱ^ϵ_Muon ≥ 1 - ε(1 + O(log K/K)) regardless of embedding structure, establishing balanced learning as a provable property. For Adam (SignGD), constructs explicit embedding matrices demonstrating that ϱ^ϵ_SignGD can collapse to O(ε^{-0.7}K^{-0.3}) with singular value ratios below 25%, establishing instability as a provable property.

Information flows through these components as follows: component ablations (Section 4.1) identify VO+FFN as the critical target → spectral analysis (Section 4.2) confirms that Muon produces more isotropic weights in these components → heavy-tailed task (Section 4.3) demonstrates that this isotropy translates to balanced head-tail learning → one-layer model (Section 5) provides the theoretical mechanism: outer-product gradient structure + SVD normalization = frequency-independent updates → theorems prove that this mechanism is provably robust for Muon and provably fragile for Adam.

### 3.3 Roadmap for the Deep Dive

- **First**, the component ablation framework (Section 4.1): how the "Independent Blocks" and "Combined Configurations" experiments isolate VO+FFN as Muon's primary beneficiaries. This establishes *where* to look.
- **Second**, the spectral isotropy metrics and measurement protocol (Section 4.2): how normalized SVD entropy, effective rank, Top-k energy fraction, and eigenvalue quantile ratio quantify the evenness of learned representations, and what the training dynamics reveal about optimizer differences.
- **Third**, the heavy-tailed knowledge task design (Section 4.3): how the power-law class distribution is constructed, how First Token Accuracy is measured, and what the head-vs-tail performance gaps reveal about balanced learning.
- **Fourth**, the one-layer associative memory abstraction (Section 5): how the model reduction preserves the essential structure (outer-product parameterization, cross-entropy loss, frequency-weighted population risk) while enabling closed-form analysis.
- **Fifth**, the theoretical analysis (Theorems 5.3 and 5.4): the definition of ϱ^ϵ_opt, the SVD calculations that expose Muon's isotropic updates, the embedding constructions that expose Adam's instability, and the interpretation of the proved bounds.
- **Sixth**, the unified mechanism: how the empirical and theoretical results converge on the outer-product-alignment explanation for Muon's superiority.

### 3.4 Detailed, Sentence-Based Technical Breakdown

This is primarily a **mechanistic analysis paper** whose core idea is that Muon's SVD-based update rule naturally aligns with the outer-product structure of linear associative memories, producing frequency-independent learning that is provably robust under any embedding geometry, whereas Adam's element-wise sign operation is embedding-dependent and can catastrophically amplify class imbalance in certain regimes.

---

#### Component Ablation Framework (Section 4.1)

The goal of the component ablation experiments is to answer the question: **"Which transformer parameters benefit most from Muon's matrix-norm–based optimization?"** The methodology is to take a baseline model where all parameters use Adam, then selectively switch individual components (or groups of components) to Muon while keeping everything else on Adam, and measure how much of the full-Muon performance is recovered.

**Model architecture.** The experiments use a 160M-parameter NanoGPT model trained on the FineWeb dataset, evaluated by validation loss. Two FFN variants are tested: a standard (non-gated) FFN defined by

$$X^{(\ell)} = H^{(\ell)} + W_{\text{out}}^{(\ell)} \sigma(W_{\text{in}}^{(\ell)} H^{(\ell)})$$

and a gated FFN variant common in modern LLMs (Touvron et al., 2023) defined by

$$X^{(\ell)} = H^{(\ell)} + W_{\text{out}}^{(\ell)} \left(\sigma(W_{\text{in}}^{(\ell)} H^{(\ell)}) \odot (W_{\text{gate}}^{(\ell)} H^{(\ell)})\right)$$

where $\sigma$ is an element-wise activation, $\odot$ is the Hadamard (element-wise) product, and $W_{\text{in}}^{(\ell)} \in \mathbb{R}^{d_f \times d}$, $W_{\text{out}}^{(\ell)} \in \mathbb{R}^{d \times d_f}$, $W_{\text{gate}}^{(\ell)} \in \mathbb{R}^{d_f \times d}$. The attention module for each head $h$ at layer $\ell$ is:

$$H^{(\ell)} = X^{(\ell-1)} + \sum_{h=1}^H W_{O,h}^{(\ell)} W_{V,h}^{(\ell)} X^{(\ell-1)} \text{sm}\left(X^{(\ell-1),\top} W_{K,h}^{(\ell),\top} W_{Q,h}^{(\ell)} X^{(\ell-1)}\right)$$

where $\text{sm}(\cdot)$ is column-wise softmax, $W_{Q,h}^{(\ell)}, W_{K,h}^{(\ell)} \in \mathbb{R}^{d_k \times d}$, and $W_{V,h}^{(\ell)} \in \mathbb{R}^{d_v \times d}$, $W_{O,h}^{(\ell)} \in \mathbb{R}^{d \times d_v}$. Crucially, the paper does not use grouped-query attention, so $W_Q$, $W_K$, $W_V$, and $W_O$ all have the same parameter count per head.

**Optimizer configuration.** Both Adam and Muon are run **without weight decay and without Nesterov acceleration** to isolate the effect of the update rule itself. Adam uses $\beta_1 = 0.8$, $\beta_2 = 0.95$; Muon uses momentum $\mu = 0.95$. Learning rates are tuned via grid search over $\{1\times 10^{-1}, 5\times 10^{-2}, 2\times 10^{-2}, 1\times 10^{-2}, 5\times 10^{-3}, 2\times 10^{-3}, 1\times 10^{-3}, 5\times 10^{-4}, 2\times 10^{-4}\}$. The chosen learning rates are then fixed for all component ablation experiments — no per-configuration retuning is done, which the paper notes means the partial-Muon configurations use the same learning rate as full Muon without further optimization.

**Two-stage experimental protocol.** The ablation proceeds in two stages:

**Stage 1: "Independent Blocks."** Apply Muon to exactly one type of parameter matrix at a time, keeping all others on Adam. The tested configurations are:
- Muon on QK attention ($W_Q$, $W_K$ only) with Adam on VO attention ($W_V$, $W_O$) and FFN — labeled "Muon(QK Attn) & Adam(VO Attn, FFN)"
- Muon on VO attention with Adam on QK attention and FFN — labeled "Muon(VO Attn) & Adam(QK Attn, FFN)"
- Muon on $W_V$ only with Adam on $W_Q$, $W_K$, $W_O$, and FFN — labeled "Muon(V Attn) & Adam(QKO Attn, FFN)"
- Muon on $W_O$ only with Adam on $W_Q$, $W_K$, $W_V$, and FFN — labeled "Muon(O Attn) & Adam(QKV Attn, FFN)"
- Muon on $W_{\text{in}}$ only with Adam on all attention and $W_{\text{out}}$ (and $W_{\text{gate}}$ for gated FFN)
- Muon on $W_{\text{out}}$ only with Adam on all attention and $W_{\text{in}}$ (and $W_{\text{gate}}$ for gated FFN)
- Muon on $W_{\text{gate}}$ only (gated FFN only)

**Stage 2: "Combined Configurations."** Apply Muon to the most impactful groups identified in Stage 1 simultaneously, to test whether a partial application can recover the performance of full Muon. The tested configurations are:
- Muon on VO attention and FFN (all three FFN matrices) with Adam on QK attention — the primary configuration of interest
- Muon on VO attention and $W_{\text{in}}$ only, with Adam on QK and $W_{\text{out}}$
- Muon on VO attention and $W_{\text{out}}$ only, with Adam on QK and $W_{\text{in}}$
- For gated FFN: analogous splits (VO+Win+Wgate, VO+Wout+Wgate, VO+Win+Wout)
- Muon on $W_V$ attention and FFN only, with Adam on QKO attention
- Muon on $W_O$ attention and FFN only, with Adam on QKV attention

**What the results show (Figure 1, Table 1).** The independent-block experiments reveal a clear hierarchy among attention weights: VO (value-output) substantially outperforms QK (query-key). For the non-gated FFN model at 10,000 steps, applying Muon to VO attention and keeping QK on Adam achieves validation loss 3.7644, compared to 3.8925 for Muon on QK with Adam on VO — a gap of approximately 0.13 loss points. The full Adam baseline reaches 3.9242, and full Muon reaches 3.5654. Breaking VO apart: Muon on $W_O$ alone (3.7712) outperforms Muon on $W_V$ alone (3.8301), and both substantially outperform QK (3.8925). For FFN, $W_{\text{out}}$ (3.7023) is more impactful than $W_{\text{in}}$ (3.7170) in the non-gated setting; in the gated setting, all three FFN matrices ($W_{\text{in}}$, $W_{\text{gate}}$, $W_{\text{out}}$) show comparable benefits (range 3.7843–3.7918).

The combined-configuration experiments deliver the paper's central empirical finding: **Muon on VO+FFN nearly recovers the full-Muon trajectory**. In the non-gated setting, Muon(VO Attn, FFN) & Adam(QK Attn) achieves 3.5858, compared to 3.5654 for full Muon — a gap of only 0.0204. In the gated setting, the corresponding numbers are 3.5312 vs. 3.5125 (gap 0.0187). The paper attributes the small remaining gap to the fact that VO+FFN uses the same learning rate as full Muon without further tuning:

> "The small remaining gap between full Muon and VO+FFN may arise because VO+FFN uses the same learning rate as full Muon without further tuning. This gap could likely be reduced by adjusting the learning rate specifically for VO+FFN."

Further ablating within VO+FFN reveals architectural dependence: in the non-gated setting, Muon(VO Attn, W_out) & Adam(QK Attn, W_in) achieves 3.6054, nearly matching full VO+FFN (3.5858), indicating that $W_{\text{out}}$ is the dominant FFN contributor. In the gated setting, the same configuration (with appropriate gate handling) falls short at 3.5833 vs. 3.5312, suggesting that the gating mechanism distributes Muon's benefit more evenly across FFN matrices.

**Why this matters for the rest of the paper.** This component ablation establishes that the associative memory parameters — specifically $W_V$, $W_O$, $W_{\text{in}}$, $W_{\text{out}}$, and $W_{\text{gate}}$ — are the primary beneficiaries of Muon's optimization. The QK parameters, despite having the same parameter count as VO, contribute negligible benefit. This observation is not explainable by parameter counting or by avoiding attention logit explosion (the paper verifies in Appendix C.1 that MaxLogit values remain stable and moderate, ranging from 5.613 to 8.396 across layers, ruling out the hypothesis that Muon simply suppresses an attention instability). The finding directly motivates the associative memory lens: since prior work (Geva et al., 2020; Bietti et al., 2023; Meng et al., 2022a) identifies $W_O$, $W_V$, and FFN weights as the model's primary associative memory stores, the fact that these are precisely the components where Muon shines suggests that Muon's mechanism is specifically well-suited to learning associative memory structures.

---

#### Spectral Isotropy Analysis (Section 4.2)

The spectral analysis addresses the question: **"How does Muon shape the learned weight matrices differently from Adam?"** The hypothesis is that Muon's SVD normalization — which sets all non-zero singular values of the gradient to 1 before forming the update — produces weight matrices whose singular value spectra are more evenly distributed (isotropic) than those learned by Adam, which normalizes individual entries via the sign function without respecting the matrix's spectral structure.

**Four isotropy metrics are defined.** For a weight matrix with $n$ non-zero singular values $\sigma = (\sigma_1, \sigma_2, \ldots, \sigma_n)$ sorted in descending order, the normalized singular energy distribution is $q = (q_1, q_2, \ldots, q_n)$ where

$$q_i = \frac{\sigma_i^2}{\sum_{j=1}^n \sigma_j^2}$$

This distribution represents the fraction of the matrix's total squared Frobenius norm captured by the $i$-th singular direction.

**Metric 1: Normalized SVD Entropy.** Adapted from Alter et al. (2000), this quantifies the uniformity of the energy distribution:

$$H_{\text{norm}}(\sigma) = -\frac{1}{\log n} \sum_{i=1}^n q_i \log q_i$$

where $q_i$ is the $i$-th normalized singular energy, $n$ is the number of non-zero singular values, and $\log n$ is the maximum possible entropy (achieved when $q_i = 1/n$ for all $i$).

**What it computes:** the Shannon entropy of the distribution $q$, divided by the maximum entropy possible for an $n$-dimensional distribution, yielding a value in $[0, 1]$. A value of 1 means all singular directions carry exactly equal energy (perfectly isotropic). A value near 0 means almost all energy is concentrated in a single direction (rank-1 approximation).

**Why this form:** Shannon entropy is the canonical measure of distribution uniformity because it is maximized by the uniform distribution and decreases monotonically as the distribution concentrates. Normalizing by $\log n$ makes the metric scale-invariant with respect to matrix dimension, enabling comparison across layers with different hidden dimensions.

**Metric 2: Effective Rank.** From Roy & Vetterli (2007), this provides a continuous measure of the number of "significant" singular dimensions:

$$\text{eRank}(\sigma) = \exp\left(-\sum_{i=1}^n q_i \log q_i\right)$$

where $q_i$ is the normalized singular energy as defined above.

**What it computes:** the perplexity (exponentiated entropy) of the singular energy distribution. If all $n$ singular values are equal, $\text{eRank} = n$. If the energy concentrates in $k$ directions equally, $\text{eRank} \approx k$. The exponential transformation maps entropy to an interpretable "count" of effective dimensions.

**Why this form:** perplexity is more intuitive than entropy for practitioners — "this weight matrix is using approximately 340 effective dimensions" is more interpretable than "this weight matrix has normalized entropy 0.72." Both capture the same information, but effective rank provides a natural units interpretation.

**Metric 3: Top-k Energy Fraction.** Measures energy concentration in the dominant components:

$$\text{TopE}_k(\sigma) = \frac{\sum_{i=1}^k \sigma_i^2}{\sum_{j=1}^n \sigma_j^2}$$

where the singular values are sorted in descending order ($\sigma_1 \geq \sigma_2 \geq \cdots \geq \sigma_n$). The paper reports $\text{TopE}_{10}$ (Top-10 energy fraction).

**What it computes:** the fraction of the total energy explained by the $k$ largest singular directions. A value close to 1 means the matrix is dominated by a few large components; a small value means energy is distributed.

**Why this form:** this is the standard metric from PCA for dimensionality assessment. It is complementary to entropy — entropy measures overall evenness while Top-k energy specifically characterizes the tail of the distribution relative to the head.

**Metric 4: Eigenvalue Quantile Ratio.** Measures the spread of the singular energy distribution while being robust to extreme outliers:

$$Q_{75/25}(\sigma) = \frac{Q_3(\{\sigma_i^2\}_{i=1}^n)}{Q_1(\{\sigma_i^2\}_{i=1}^n)}$$

where $Q_3$ is the 75th percentile of the squared singular values and $Q_1$ is the 25th percentile.

**What it computes:** the ratio of the upper quartile to the lower quartile of the eigenvalue distribution. A value of 1 means the middle 50% of eigenvalues are identical. Large values indicate heavy skew.

**Why this form:** the quantile ratio is robust to the largest and smallest singular values, which can be extreme outliers. Standard metrics like condition number ($\sigma_1/\sigma_n$) are dominated by the single largest and smallest values, making them sensitive to noise and sensitive to whether the matrix has near-zero singular values. The interquartile ratio captures the bulk of the distribution.

**Measurement protocol.** All four metrics are computed for each weight matrix at 100-step intervals throughout the 10,000-step training run. Each experiment is repeated with 10 different random seeds, and the metrics are averaged. The error bars in Figure 2 represent the standard deviation across seeds, deliberately made small (often invisible) for Muon and large for Adam.

**What the results show (Figure 2).** The analysis focuses on the VO attention weights (aggregated $W_V$ and $W_O$) and $W_{\text{out}}$ FFN weights, as these are the dominant associative memory components identified in Section 4.1. The results are reported for both non-gated and gated FFN variants across four panels.

For **VO weights (non-gated FFN, Figure 2a):** under Muon, the normalized SVD entropy starts high (approximately 0.7–0.8 at step 0) and stays at this level throughout training with negligible variance across seeds. Under Adam, it starts around 0.45 at step 0, then exhibits large oscillations — rising to approximately 0.65 by step 2000, dropping back to 0.40 by step 4000, rising again — with substantial error bars (the shaded regions in the figure span roughly ±0.10 entropy units). The Top-10 energy fraction shows the mirror pattern: Muon consistently around 0.15–0.20 (only 15–20% of energy in the top 10 components), while Adam fluctuates between 0.30 and 0.70. The effective rank under Muon is stable at approximately 300–350 (out of a maximum possible rank), while Adam's effective rank oscillates between 100 and 250. The Q75/Q25 ratio is consistently near 1–3 under Muon and ranges from 5 to over 100 under Adam, with large seed-to-seed variance.

For **VO weights (gated FFN, Figure 2b):** the pattern is similar but with some attenuation. Muon's entropy remains in the 0.75–0.85 range, while Adam's fluctuates between 0.55 and 0.75. The effective rank difference is smaller but still consistent (Muon 350–400, Adam 200–350). The error bars for Adam are again substantially larger.

For **$W_{\text{out}}$ weights (non-gated FFN, Figure 2c):** Muon's entropy starts near 1.0 and stays there — the matrix is almost perfectly isotropic — while Adam's starts near 0.5 and oscillates with large variance. The Top-10E under Muon is near 0.0 (negligible concentration), while Adam's ranges from 0.2 to 0.8 depending on the training step. Effective rank under Muon is approximately 550–600 (near the theoretical maximum for the matrix dimensions), while Adam's ranges from 100–400 with high variance.

For **$W_{\text{out}}$ weights (gated FFN, Figure 2d):** the trends persist, though the absolute entropy under Adam is higher than in the non-gated case.

**Key observation about initialization sensitivity.** Perhaps the most striking result is the difference in seed-to-seed variance. Muon's metrics have **negligible error bars** across all 10 seeds, indicating that the optimizer converges to the same spectral structure regardless of random initialization. Adam's metrics have **large error bars** that persist throughout training, indicating that the optimizer's final spectral structure is highly sensitive to initial conditions. This is the empirical foundation for the theoretical claim in Theorem 5.3 that Adam's performance is "embedding-dependent" — different initializations sample different effective embedding geometries, and Adam's element-wise sign operation amplifies these differences rather than suppressing them.

**Connection to associative memory.** The paper interprets these spectral patterns through the associative memory lens. An isotropic singular spectrum means that all stored fact directions have approximately equal representation strength in the weight matrix. If the weight matrix represents a sum of outer products $W = \sum_i e_{o_i} e_{s_i}^\top$ for $K$ facts, then isotropic singular values mean each fact's contribution has comparable magnitude, regardless of how frequently that fact appeared in training. The concentrated spectrum under Adam means a small number of singular directions dominate the matrix — these correspond to the most frequently occurring facts that received the largest gradient updates. The oscillation in Adam's isotropy over training steps reflects the "competition" between fact frequencies as the optimizer alternates between fitting head classes and catching up on tail classes.

---

#### Heavy-Tailed Knowledge Task Design (Section 4.3)

The heavy-tailed knowledge task is designed to answer: **"Does Muon's more isotropic weight spectrum translate to more balanced knowledge acquisition across class frequencies?"** The task isolates the effect by creating a pure associative memory problem where the training data imbalance is precisely controlled.

**Dataset construction.** The foundation is the synthetic biographical QA dataset from Allen-Zhu & Li (2024), containing structured biographical information for over 200,000 uniquely named individuals. Each individual is assigned a combination of seven attributes: first name (from approximately 400 options), surname (approximately 1000 options), birthdate, birthplace, educational institution (approximately 300 options), major (approximately 100 options), and employer (approximately 300 options). From each biography, six QA pairs are generated using template sentences. For example, for a hypothetical individual "Ashton Hilda Older," the six QA pairs might be:

```
Q: What is the birth date of Ashton Hilda Older?  A: February 01, 2063.
Q: What is the birth city of Ashton Hilda Older?   A: Miami, FL.
Q: Which university did Ashton Hilda Older study?   A: Saddleback College.
Q: What major did Ashton Hilda Older study?         A: General Literature.
Q: Which company did Ashton Hilda Older work for?   A: BlockFi.
Q: Where did Ashton Hilda Older work?               A: Jersey City.
```

**Power-law class distribution.** The class frequency is controlled by an integer parameter $m = 15$. The classes are organized into $m+1 = 16$ groups, indexed $g = 0$ to $g = 15$. Group $g$ contains $N_g$ classes where $N_0 = 1$ and $N_g = 2^{g-1}$ for $g > 0$. Each class in group $g$ is allocated $S_g = 2^{m-g}$ "selections," and each selection generates $n_{\text{qa}} = 6$ unique QA pairs. Thus, the total number of QA samples per class in group $g$ is $S_g \times n_{\text{qa}}$.

Concretely:
- **Group 0** (head): 1 class with $2^{15} = 32,768$ selections × 6 QA = 196,608 total samples for that single class
- **Group 1**: 1 class with $2^{14} = 16,384$ selections × 6 = 98,304 samples
- **Group 15** (extreme tail): $2^{14} = 16,384$ classes, each with $2^0 = 1$ selection × 6 = 6 samples per class

Total number of classes: $1 + 1 + 2 + 4 + \cdots + 2^{14} = 2^{15} = 32,768$ classes. The sample distribution per class is visualized in Figure 3(a), showing the characteristic power-law shape on a log-log plot.

**Model architecture.** A 160M NanoGPT model (the same architecture as the FineWeb experiments) is trained from scratch on this dataset. The model sees QA-formatted text and is trained with the standard next-token prediction objective. The key difference from the FineWeb experiments is that this dataset is **purely knowledge-intensive** — every correct answer requires the model to have stored the corresponding biographical fact in its associative memory weights and to retrieve it given the question as a query.

**Evaluation metric: First Token Accuracy (FTA).** Following Allen-Zhu & Li (2024), accuracy is measured at the first token of the answer. Since each QA pair has a unique correct answer token (the first token of the answer string, e.g., "February" or "Miami" or "Saddleback"), the model's prediction is correct if and only if the highest-probability token matches this ground-truth first token. FTA directly measures whether the model has stored the correct association, without confounding from autoregressive decoding artifacts.

**Optimizer configurations tested.** Six configurations are compared:
1. **Full Muon** on all attention and FFN parameters
2. **Full Adam** on all parameters
3. **SGD + Momentum** on all parameters (baseline for the heavy-tailed learning literature)
4. **Muon(VO, FFN) & Adam(QK)** — the hybrid identified as critical in Section 4.1
5. **Muon(QK) & Adam(VO, FFN)** — the reverse hybrid as a control
6. Each configuration is repeated with multiple random seeds, with error bars representing standard deviation.

**What the results show (Figure 3, Tables 3–5).** The FTA curves are plotted as a function of training steps (0 to 10,000), stratified by class group. Group 0 (head) through Group 15 (extreme tail) are shown as separate lines in Figures 3(b)–3(f), with group index indicated by color (warm colors for head groups, cool colors for tail groups).

For **full Muon (Figure 3b):** head classes (Groups 0–2) reach near-perfect accuracy (>0.95) by approximately 2,000 steps. Mid-frequency classes (Groups 5–8) reach >0.90 by 5,000 steps. The most important result is the tail performance: by 10,000 steps, Group 13 achieves 1.000 ± 0.000 accuracy and Group 15 (the extreme tail with only 6 training examples per class) achieves **0.976 ± 0.006** accuracy. The error bars across seeds are very narrow for all groups.

For **full Adam (Figure 3c):** head classes also reach near-perfect accuracy by 2,000 steps. Mid-frequency classes follow a similar trajectory to Muon through approximately Group 8. However, tail performance diverges dramatically: by 10,000 steps, Group 13 achieves 0.890 ± 0.042 (0.110 lower than Muon) and Group 15 achieves only **0.264 ± 0.048** (0.712 lower than Muon). The error bars are substantially larger, especially for tail groups — Group 15's accuracy at 10,000 steps ranges from approximately 0.216 to 0.312 depending on the seed, versus Muon's 0.970 to 0.982.

For **SGD + Momentum (Figure 3d):** all groups perform worse, including head classes. Group 0 reaches only approximately 0.8 accuracy by 10,000 steps, and tail groups (13–15) plateau at 0.12–0.30. This confirms the well-established finding that SGD struggles with heavy-tailed distributions (Kunstner et al., 2024) and provides a lower bound.

For **Muon(VO, FFN) & Adam(QK) (Figure 3e):** the performance closely tracks full Muon. Group 15 reaches 0.954 ± 0.021, only 0.022 below full Muon. Group 13 reaches 0.998 ± 0.002. This confirms that the associative memory parameters are the causal mechanism — applying Muon only to these parameters is sufficient to achieve tail-class learning nearly as balanced as full Muon.

For **Muon(QK) & Adam(VO, FFN) (Figure 3f):** tail performance degrades substantially. Group 15 reaches only 0.286 ± 0.039, essentially matching full Adam's performance (0.264 ± 0.048). The QK parameters, when optimized with Muon while VO+FFN uses Adam, provide essentially no benefit for tail-class learning. This is a crucial control that rules out the possibility that Muon's benefit comes from some general optimization property — it specifically matters when applied to the associative memory components.

**The training dynamics reveal the mechanism.** The paper's key observation is that Muon's advantage is **not about faster learning of head classes** (Adam matches Muon there) but about **continuing to learn tail classes** after head classes are already saturated. In the Adam training curve, the tail-group lines (Groups 13–15) show a distinctive pattern: they initially rise (as the model begins to learn rare facts), then **plateau or decline** as head-class gradients continue to dominate the update. This is the signature of head-class dominance: once head facts are well-learned, their gradients remain non-zero (the model continues to refine its predictions for frequently seen facts), and these gradients overwhelm the weak gradient signal from rare facts. Muon's SVD normalization prevents this — by equalizing the update magnitude across all orthogonal gradient directions, rare-fact gradient directions contribute just as strongly to the parameter update as frequent-fact directions, allowing tail classes to continue learning even after head classes saturate.

---

#### One-Layer Associative Memory Abstraction (Section 5)

The one-layer model is designed to **abstract away all transformer-specific complexity while preserving the essential structure that matters for the optimizer comparison**: a linear associative memory trained on frequency-weighted cross-entropy loss.

**Setup.** Consider $K$ triplets $\{(s_i, r_i, o_i)\}_{i=1}^K$, where each triplet represents a fact (subject $s_i$, relation $r_i$, object $o_i$). The subject-relation pairs are embedded into the columns of a matrix $E \in \mathbb{R}^{d_s \times K}$, and the objects are embedded into the columns of $\tilde{E} \in \mathbb{R}^{d_o \times K}$. That is, $E_k$ (the $k$-th column of $E$) is the embedding of the $k$-th subject-relation pair, and $\tilde{E}_k$ is the embedding of the $k$-th object.

**The associative memory.** A linear map $W \in \mathbb{R}^{d_o \times d_s}$ maps key embeddings to value embeddings. Given a query $E_k$ (the embedding of the $k$-th subject-relation pair), the model predicts object probabilities via:

$$f_W(E_k) = \text{sm}(\tilde{E}^\top W E_k) \in \mathbb{R}^K$$

where $\text{sm}$ is the softmax function over the $K$ possible objects. The $k'$-th component of this vector is:

$$[f_W(E_k)]_{k'} = \frac{\exp(\tilde{E}_{k'}^\top W E_k)}{\sum_{k''=1}^K \exp(\tilde{E}_{k''}^\top W E_k)}$$

This is the probability assigned to object $k'$ given the key for fact $k$.

**The loss function.** The model is trained to minimize the population cross-entropy loss:

$$\mathcal{L}(W) = -\sum_{k=1}^K p_k \log [f_W(E_k)]_k$$

where $p_k \in [0, 1]$ is the frequency (probability) of the $k$-th triplet in the training distribution, with $\sum_k p_k = 1$.

**What this loss computes:** for each fact $k$, it penalizes the model by $-\log(\text{probability assigned to the correct object})$ weighted by how frequently that fact appears. The weighting $p_k$ is what creates the class imbalance — frequent facts contribute more to the total loss gradient.

**Why this form:** cross-entropy is the standard objective for next-token prediction in LLMs (the full transformer is trained with exactly this loss on its output logits). The population version (expectation over the data distribution) captures the infinite-data limit where the only source of imbalance is the class frequencies $p_k$, not finite-sample noise.

**The three optimizers compared:**

1. **Gradient Descent (GD):**
   $$W_{t+1}^{\text{GD}} = W_t^{\text{GD}} - \eta_{t+1} \nabla_W \mathcal{L}(W_t^{\text{GD}})$$
   where $\nabla_W \mathcal{L}(W_t^{\text{GD}})$ is the gradient of the population loss at the current parameters. This is the baseline optimizer without any normalization.

2. **SignGD (Adam without momentum):**
   $$W_{t+1}^{\text{SignGD}} = W_t^{\text{SignGD}} - \eta_{t+1} \text{sign}(\nabla_W \mathcal{L}(W_t^{\text{SignGD}}))$$
   where $\text{sign}(\cdot)$ is the element-wise sign function, applied independently to each entry of the gradient matrix. This is Adam with $\beta_1 = \beta_2 = 0$ (no exponential moving averages), which reduces Adam to its core operation: element-wise sign normalization.

   **Why this simplification:** the paper follows established theoretical practice (Kunstner et al., 2024; Bernstein & Newhouse, 2024) of disabling momentum for analysis. The momentum terms in Adam and Muon serve to reduce gradient variance across minibatches, but they are not the mechanism that differentiates the two optimizers — the key difference is in how each normalizes the gradient (element-wise sign vs. SVD-based spectral normalization). By analyzing without momentum, the paper isolates this normalization mechanism.

3. **Muon (without momentum):**
   $$W_{t+1}^{\text{Muon}} = W_t^{\text{Muon}} - \eta_{t+1} U_t \text{norm}(\Sigma_t) V_t^\top$$
   where $U_t \Sigma_t V_t^\top$ is the SVD of the gradient $\nabla_W \mathcal{L}(W_t^{\text{Muon}})$, and $\text{norm}(\Sigma_t)$ normalizes all non-zero singular values to 1 (element-wise on the diagonal of $\Sigma_t$). The resulting matrix $U_t \text{norm}(\Sigma_t) V_t^\top = U_t V_t^\top$ is the nearest semi-orthogonal matrix to the gradient.

   **Why SVD normalization matters here:** the gradient of the associative memory loss has a specific structure — it decomposes as a sum of outer products. Under the orthonormal embedding assumption ($E^\top E = \tilde{E}^\top \tilde{E} = I_{K,K}$), Proposition F.1 shows that the gradient at initialization ($W_0 = 0$) is:

   $$-\nabla_W \mathcal{L}(W_0) = \frac{\alpha}{L} \tilde{E}_{1:L} E_{1:L}^\top + \frac{1-\alpha}{K-L} \tilde{E}_{L+1:K} F_{L+1:K}^\top - \frac{\alpha}{LK} \tilde{E} J_{K,L} E_{1:L}^\top - \frac{1-\alpha}{(K-L)K} \tilde{E} J_{K,K-L} E_{L+1:K}^\top$$

   The first two terms are scaled versions of the outer products for the two class groups (head and tail). The last two terms are corrections that ensure the softmax denominator is accounted for. When Muon normalizes the singular values of this gradient matrix, it equalizes the contribution from the head and tail outer products, regardless of the frequency coefficients $\alpha/L$ and $(1-\alpha)/(K-L)$. Adam's element-wise sign does not have this effect — it operates per-entry, which can distort the outer-product structure when embeddings overlap.

**Assumptions for the theory.** Two assumptions make the analysis tractable:

**Assumption 5.1 (Orthonormal embeddings):** $E^\top E = \tilde{E}^\top \tilde{E} = I_{K,K}$. This rules out feature-level imbalance (e.g., some embeddings having larger norms than others), which would otherwise couple with frequency imbalance and complicate the analysis. The paper empirically verifies this assumption on Llama3-8b-instruct (Figure 4a): extracting the key and value embeddings for 3,000 knowledge items from Counterfact across FFN layers 5, 10, 15, 20, 25, the average pairwise angles between embeddings are near 90° (ranging from approximately 75° to 88° depending on layer), confirming approximate orthonormality in real LLMs.

**Assumption 5.2 (Two-class frequency structure):** The first $L$ triplets share the same probability and together contribute a total mass of $\alpha$:

$$p_k = \begin{cases} \alpha/L & \text{for } k \in [L] \\ (1-\alpha)/(K-L) & \text{for } k > L \end{cases}$$

**Why this simplification:** the two-class structure captures the essential head-vs-tail dynamic while enabling closed-form SVD calculations. The degree of imbalance is measured by comparing $\alpha$ (the total probability mass of the head class) to $\beta = L/K$ (the fraction of classes in the head). If $\alpha > \beta$, the head classes are over-represented relative to uniform; if $\alpha < \beta$, they are under-represented. The ratio $\alpha/\beta$ quantifies the balance. The multi-class extension follows directly from the same proof technique by extending the SVD calculation to a block-diagonal structure, so the two-class case is not a fundamental limitation.

**Experimental protocol (Figures 4b, 4c).** The paper tests two optimization protocols and two embedding regimes:

**Embedding regimes:**
- **Support-decoupled:** $E = \tilde{E} = I_{K,K}$. The embeddings of different facts have disjoint support — $E_k$ has a 1 in position $k$ and zeros elsewhere, so different facts use completely separate dimensions. This is the "easy" case where the optimizer can address each fact independently.
- **Support-coupled:** $E$ and $\tilde{E}$ are constructed as block-diagonal rotation matrices (see Theorem 5.3 proof for the explicit construction using 3×3 rotation blocks parameterized by Euler angles). Different facts' embeddings share dimensions, so the optimizer must disentangle overlapping representations.

**Optimization protocols:**
- **One-step (Figure 4b):** Take a single update $W = W_0 - \eta \cdot G_{\text{opt}}(W_0)$ and sweep the step size $\eta$ over a range to produce different values of the population loss $\mathcal{L}(W)$. Plot $\mathcal{L}(W)$ against the maximal probability gap $\Delta(W) = \max_{i,j} [f_W(E_i)]_i - [f_W(E_j)]_j$. Each point on the curve corresponds to a different step size.
- **Multi-step (Figure 4c):** Run multiple update steps to reduce $\mathcal{L}(W)$, varying the number of steps. Each point corresponds to a different optimization horizon.

**Why two protocols:** the one-step analysis is the simplest setting where the SVD calculation can be done exactly in closed form (Theorem 5.3). If the mechanism is already visible at one step, it is not an artifact of complex training dynamics. The multi-step analysis (Theorem 5.4) confirms that the same properties persist throughout training, not just at initialization.

**What the results show (Figures 4b, 4c).** Across both one-step and multi-step protocols, and across both embedding regimes:

- **For all optimizers**, $\Delta(W)$ first increases and then decreases as $\mathcal{L}(W)$ decreases. Early in training (high loss, low accuracy), correct-class probabilities are near $1/K$ (random guessing), so the gap is small. As learning progresses, some classes are learned faster than others, increasing the gap. Eventually, when all classes are well-learned (probabilities ≥ 0.9), the gap narrows again.

- **GD** consistently exhibits a large maximal probability gap (the curves for GD are shifted to the right in the $\Delta(W)$ vs. $\mathcal{L}(W)$ plots). For a given loss value, GD has a substantially higher gap than Muon, meaning GD learns classes more unevenly.

- **Muon** consistently exhibits a small gap. In the one-step plot, the Muon curve (both coupled and decoupled) hugs the leftmost region of the plot — the gap remains small (< $10^{-3}$) even as the loss decreases. In the multi-step plot, Muon maintains this small gap throughout training.

- **SignGD** shows **unstable** behavior. Under decoupled embeddings, SignGD behaves similarly to Muon (small gap). Under coupled embeddings, SignGD behaves similarly to GD (large gap). This is the crucial demonstration of Adam's embedding-dependence: the optimizer's ability to learn balanced representations depends on the particular geometry of the feature embeddings, which is essentially random from the optimizer's perspective.

---

#### The Quantity ϱ^ϵ_opt: Formalizing Balanced Learning (Section 5.2)

The paper defines a quantity that precisely captures the notion of balanced learning:

$$\varrho^\epsilon_{\text{opt}} = \inf_{\eta \geq 0} \left\{ \min_{k \in [K]} [f_{W_\eta}(E_k)]_k \;\middle|\; \max_{k \in [K]} [f_{W_\eta}(E_k)]_k \geq 1 - \epsilon \right\}$$

where $\text{opt} \in \{\text{GD}, \text{SignGD}, \text{Muon}\}$, $W_\eta = W_0 - \eta \cdot G_{\text{opt}}(W_0)$ is the parameter after one step with step size $\eta$, and $[f_{W_\eta}(E_k)]_k$ is the probability assigned to the correct object for the $k$-th fact.

**What this computes:** first, find the smallest step size $\eta$ such that at least one fact $k$ has its correct-class probability at least $1 - \epsilon$. Then, among all such step sizes, look at the fact that is *worst*-learned (minimum correct-class probability). Take the infimum over all valid step sizes. The result $\varrho^\epsilon_{\text{opt}} \in [0, 1-\epsilon]$ is the minimum correct-class probability that any fact can have when the best-learned fact has reached probability $1-\epsilon$.

If $\varrho^\epsilon_{\text{opt}} \approx 1-\epsilon$, then **all facts are learned nearly equally** — when the best fact is at $1-\epsilon$, the worst fact is also at $1-\epsilon$. If $\varrho^\epsilon_{\text{opt}} \approx 0$, then learning is **extremely imbalanced** — some facts have near-zero probability even as others are confidently correct.

**Why this specific definition:** it captures the head-tail gap in a single scalar that is comparable across optimizers and across embedding regimes. The alternative would be to compare learning curves across all $K$ classes, but this collapses the comparison to a single interpretable number. The relationship to $\Delta(W)$ (the maximal probability gap) is $\Delta(W) = 1 - \epsilon - \varrho^\epsilon_{\text{opt}}$ — a larger $\varrho^\epsilon_{\text{opt}}$ means a smaller gap.

---

#### Theoretical Results: Muon's Provable Balance (Theorem 5.3, Muon portion)

The Muon portion of Theorem 5.3 states:

**Under Assumptions 5.1 and 5.2**, with fixed $\alpha, \beta$ such that $\alpha \neq \beta$, as $K \to \infty$:

$$\varrho^\epsilon_{\text{Muon}} \geq 1 - \epsilon\left(1 + O\left(\frac{\log K}{K}\right)\right)$$

and the Muon update at initialization is:

$$G_{\text{Muon}}(W_0) = -\tilde{E} E^\top + O\left(\frac{1}{K} \tilde{E} J_{K,K} E^\top\right)$$

where $J_{K,K} \in \mathbb{R}^{K \times K}$ is the all-ones matrix.

**What the bound says:** for large $K$, when the best-learned fact reaches probability $1-\epsilon$, the worst-learned fact reaches probability at least $1-\epsilon(1 + O(\log K/K))$. Since $\log K/K \to 0$ as $K \to \infty$, this means $\varrho^\epsilon_{\text{Muon}} \approx 1-\epsilon$ in the large-$K$ limit — **nearly perfect balance, regardless of $\alpha, \beta$, and regardless of the embeddings $\tilde{E}, E$ as long as they satisfy orthonormality**.

**How the proof works (outlined in Appendix D, Step 2).** The key step is computing the SVD of the gradient matrix at $W_0 = 0$:

$$-\nabla_W \mathcal{L}(W_0) = \tilde{E} \underbrace{\left[ \text{diag}\left(\frac{\alpha}{L} I_L, \frac{1-\alpha}{K-L} I_{K-L}\right) - \frac{1}{K} I_K \cdot \left(\frac{\alpha}{L} I_L^\top, \frac{1-\alpha}{K-L} I_{K-L}^\top\right)^\top \right]}_{X} E^\top$$

The matrix $X \in \mathbb{R}^{K \times K}$ has a special structure analyzed in Proposition F.3: a diagonal matrix with two constant blocks minus a rank-1 term. Its SVD has:

- $L-1$ singular values equal to $\alpha/L$ (corresponding to the head-class subspace orthogonal to the all-ones direction)
- $K-L-1$ singular values equal to $(1-\alpha)/(K-L)$ (corresponding to the tail-class subspace orthogonal to the all-ones direction)
- One additional non-zero singular value $s_1 = \sqrt{(\alpha^2(K-L) + (1-\alpha)^2 L)/K}$
- One zero singular value

When Muon normalizes all non-zero singular values to 1, the update becomes:

$$G_{\text{Muon}}(W_0) = \tilde{E}_{1:L} E_{1:L}^\top + \tilde{E}_{L+1:K} F_{L+1:K}^\top + \text{(correction terms of order $O(1/K)$)}$$

The dominant term $\tilde{E}_{1:L} E_{1:L}^\top + \tilde{E}_{L+1:K} F_{L+1:K}^\top = \tilde{E} E^\top$ assigns **exactly equal update magnitude to every fact direction**, regardless of the frequency coefficients $\alpha/L$ and $(1-\alpha)/(K-L)$. The $O(1/K)$ correction terms arise from the fact that the singular vectors are slightly perturbed from the ideal identity mapping by the rank-1 all-ones component, but this perturbation vanishes as $K$ grows.

The bound on $\varrho^\epsilon_{\text{Muon}}$ follows from analyzing the softmax scores under this update: for any $k$, the correct-object logit grows at rate $1 + O(1/K)$ (from the $\tilde{E}_k E_k^\top$ term), while incorrect-object logits grow at rate $O(1/K)$ (from the off-diagonal correction terms). The ratio of growth rates tends to infinity as $K \to \infty$, meaning the model can achieve high correct-class probability for all facts simultaneously without any fact being left behind.

**Contrast with GD (Theorem 5.3, GD portion):**

$$\varrho^\epsilon_{\text{GD}} = O(\epsilon^{-r(\alpha,\beta)} K^{r(\alpha,\beta)-1})$$

where

$$r(\alpha, \beta) = \min\left\{\frac{\alpha(1-\beta)}{\beta(1-\alpha)}, \frac{\beta(1-\alpha)}{\alpha(1-\beta)}\right\} < 1$$

**What this means:** when data is imbalanced ($\alpha \neq \beta$), $r(\alpha,\beta) < 1$, and so $K^{r(\alpha,\beta)-1} \to 0$ as $K \to \infty$. The worst-class probability vanishes polynomially in $K$, meaning GD leaves many classes essentially unlearned when one class is learned well. When data is perfectly balanced ($\alpha = \beta$), $r(\alpha,\beta) = 1$, and the bound becomes $O(1)$, consistent with balanced learning under uniform frequencies.

---

#### Theoretical Results: Adam's Embedding-Dependent Instability (Theorem 5.3, Adam portion)

The Adam portion of Theorem 5.3 constructs two **explicit embedding matrices** that demonstrate the optimizer's instability:

**Case 1 (support-decoupled, $E = \tilde{E} = I_{K,K}$):** Under this construction,

$$\varrho^\epsilon_{\text{SignGD}} \geq 1 - \epsilon$$

meaning SignGD achieves **nearly perfect balance**, matching Muon's performance. The intuition: when embeddings are decoupled, the gradient matrix is diagonal, and the element-wise sign operation does not mix different fact directions — each fact's update depends only on its own frequency, and the sign function maps all non-zero gradients to ±1 regardless of magnitude.

**Case 2 (support-coupled, explicit rotation matrix construction):** The paper constructs $\tilde{E}$ and $E$ as block-diagonal matrices where each 3×3 block contains an orthonormal basis encoded via rotation matrices $R(a, b, c)$ parameterized by Euler angles. The specific construction:

$$\tilde{E} = I_{K/3, K/3} \otimes R(3.638, 2.949, 5.218)$$
$$E = I_{K/3, K/3} \otimes R(1.715, 0.876, 3.098)$$

where $\otimes$ is the Kronecker product. The rotation matrix $R(a, b, c)$ is:

$$R(a, b, c) = \begin{bmatrix} \cos a \cos b \cos c - \sin a \sin c & -\cos a \cos b \sin c - \sin a \cos c & \cos a \sin b \\ \sin a \cos b \cos c + \cos a \sin c & -\sin a \cos b \sin c + \cos a \cos c & \sin a \sin b \\ -\sin b \cos c & \sin b \sin c & \cos b \end{bmatrix}$$

**What this construction does:** each 3×3 block is an orthonormal matrix (rotation in 3D), ensuring $R^\top R = I_{3,3}$, so Assumption 5.1 is satisfied. The specific Euler angles are chosen (the paper does not state the selection criterion explicitly, but the values are given to 3 decimal places) so that when the element-wise sign function is applied to the gradient expressed in this basis, the resulting update matrix has a highly skewed singular value spectrum.

**Under this construction,** the paper shows that for sufficiently large $K$:

$$\varrho^\epsilon_{\text{SignGD}} = O(\epsilon^{-0.7} K^{-0.3})$$

and

$$\frac{\sigma_{\min}(G_{\text{SignGD}}(W_0))}{\sigma_{\max}(G_{\text{SignGD}}(W_0))} \leq 25\%$$

**What these results mean:**

- **The bound on $\varrho^\epsilon_{\text{SignGD}}$:** as $K \to \infty$, $K^{-0.3} \to 0$, so the worst-class probability vanishes — SignGD cannot learn all classes when one class is learned well. Crucially, the exponent $-0.7$ on $\epsilon$ is **independent of $\alpha$ and $\beta$** — it arises from the specific embedding geometry, not from the data imbalance. This means that even with perfectly balanced data, SignGD can still produce imbalanced learning under unfavorable embeddings.

- **The singular value ratio bound:** the smallest singular value of the SignGD update is less than 25% of the largest. This means some directions in parameter space receive much smaller updates than others under SignGD. Since singular directions correspond to stored fact patterns in the associative memory, this directly implies that some facts are updated much less than others — the spectral signature of imbalanced learning.

**How the proof works (outlined in Appendix D, Step 3).** The key is computing the element-wise sign of the gradient when expressed in the coupled embedding basis. Under the support-decoupled case, the gradient is diagonal, so $\text{sign}(\nabla L) = \text{sign}(\text{diag}(\gamma_1, \ldots, \gamma_K)) = \text{diag}(\pm 1, \ldots, \pm 1)$ — each entry is independent. Under the coupled case, the gradient matrix after applying the rotation transformations has a block structure where different facts' gradients interact within each 3×3 block. The sign function, applied element-wise, does not respect this block structure — it can produce a matrix where the effective update directions (the singular vectors) are misaligned with the fact directions (the embedding vectors). This misalignment is what causes some facts to receive smaller updates than others.

The specific numbers (0.7, 0.3, 25%) are not claimed to be fundamental constants — they depend on the particular choice of Euler angles. But the existence of *some* embedding for which SignGD collapses is the key theoretical insight: **Adam's performance is contingent on the (essentially random) embedding geometry, while Muon's is provably robust to it**.

---

#### Multi-Step Extension (Theorem 5.4)

Theorem 5.4 extends the one-step analysis to the multi-step case:

$$\varrho^\epsilon_{\text{Muon}} \geq 1 - \epsilon\left(1 + O\left(\frac{\log K}{K}\right)\right)$$

and

$$G_{\text{Muon}}(W_t) = -\tilde{E} E^\top + O\left(\frac{1}{K} \tilde{E} J_{K,K} E^\top\right)$$

for any $t \geq 0$.

**What this adds beyond Theorem 5.3:** the proof (Appendix E) shows by induction that the parameter matrix $W_t$ maintains the same structural decomposition at every step:

$$W_t = \tilde{E} X_t E^\top, \quad X_t = \Lambda_t + C_t$$

where $\Lambda_t = \text{diag}(a_t \cdot I_L, b_t \cdot I_{K-L})$ is a diagonal matrix with two constant blocks, $C_t$ is a block-wise constant matrix, and $a_t = b_t \geq 0$ (the head and tail diagonal entries remain equal throughout training). The off-diagonal correction terms satisfy $c_t^{ij} = O(a_t/K)$ — they remain proportionally small relative to the diagonal entries.

**Why this structure persists:** at each step, the softmax scores depend only on whether the query index $k$ is in the head or tail group, not on the specific index within the group. This symmetry, combined with the orthonormal embedding assumption, means the gradient at every step has the same block structure as the initial gradient. The Muon update, being a function of the SVD of this structured gradient, preserves the structure. The crucial property $a_t = b_t$ — that head and tail classes receive equal diagonal weight — is maintained because Muon's SVD normalization equalizes the singular values, which correspond to these diagonal weights.

**The bound on $\varrho^\epsilon_{\text{Muon}}$ in the multi-step case** follows from the same argument as the one-step case, applied to $W_t$ rather than $W_0$. Since the structural properties (equal diagonal entries, small off-diagonal corrections) are preserved, the analysis of softmax scores at any step $t$ is isomorphic to the analysis at step 0, just with larger diagonal values (representing the accumulated learning).

---

#### Synthesis: The Complete Mechanism

The paper's proposed mechanism, synthesized from the empirical and theoretical results, is:

1. **Transformer associative memory parameters** ($W_V$, $W_O$, FFN weights) can be approximated as sums of outer products $\sum_i e_{o_i} e_{s_i}^\top$ representing stored facts.

2. **The gradient** of the loss with respect to these parameters **also decomposes as a sum of outer products**, with each term weighted by the frequency of the corresponding fact in the training data (Proposition F.1).

3. **Muon's SVD-based normalization** computes the singular value decomposition of this gradient and sets all non-zero singular values to 1 before forming the update. Since the singular values encode fact frequencies, this normalization **equalizes the update magnitude across all facts**, regardless of frequency.

4. **Adam's element-wise sign normalization** treats each matrix entry independently. When fact embeddings are decoupled (disjoint support), this is equivalent to frequency-independent normalization. But when embeddings overlap (coupled support), the sign operation **mixes contributions from different facts** in a way that can amplify frequency differences — frequent facts dominate the update because their contributions appear in more matrix entries.

5. **On heavy-tailed data**, Muon's frequency-independent updates prevent frequent (head) facts from dominating the parameter updates, allowing infrequent (tail) facts to continue receiving meaningful gradient signal even after head classes are well-learned. Adam's frequency-dependent updates cause tail-class learning to stagnate once head classes saturate.

6. **The isotropy of the learned weights** (Observation 2) is the spectral signature of this mechanism: Muon produces weight matrices where all singular directions have comparable energy because all facts are represented with comparable strength. Adam produces concentrated spectra because frequent facts dominate the representation.

7. **The QK attention parameters do not participate in this mechanism** because they do not serve as associative memory stores — they compute query-key similarities rather than storing key-value associations. This explains why applying Muon to QK provides negligible benefit (Figures 1, 3f).

The paper concludes Section 5 by noting that this outer-product alignment mechanism may generalize beyond linear associative memories:

> "Intuitively, this property of Muon may extend beyond outer products to higher-order tensor products, an exciting direction for future work."

This suggests that any parameter matrix whose natural gradient structure involves sums of rank-1 terms could benefit from Muon's spectral normalization, potentially including higher-order interaction terms in attention and FFN computations that are not captured by the linear associative memory approximation.

## 4. Key Insights and Innovations

### Innovation 1: Associative Memory as the Diagnostic Lens for Optimizer Behavior

The paper's most distinctive intellectual move is **diagnostic rather than prescriptive**: it asks *why* an optimizer works, and in doing so, discovers that the answer lies in a previously overlooked connection between two disconnected literatures—optimizer design and associative memory models of transformers.

Prior to this work, the study of Muon followed a predictable pattern shared by most optimizer research: (1) propose a new update rule, (2) demonstrate empirical gains, (3) analyze convergence properties in standard non-convex optimization settings. The steepest-descent interpretation (Bernstein & Newhouse, 2024) provided mathematical grounding by showing that Muon corresponds to descent under the spectral norm, while Adam corresponds to descent under the vector ∞-norm. But this framing **explains what Muon computes without explaining why that computation helps**—it reduces the difference between Adam and Muon to a choice of norm, which is a mathematically valid but mechanistically vacuous distinction. Norms do not explain why one set of transformer parameters benefits dramatically while another sees negligible effect.

The paper's breakthrough is to **replace the norm-centric view with a structure-centric view**. By decomposing the transformer into its functional components and measuring which ones respond to Muon, the authors discover that the answer is not about norms in the abstract—it is about which parameters encode linear associative memories. The VO attention weights and FFN output weights, which prior work (Geva et al., 2020; Bietti et al., 2023; Meng et al., 2022a) had independently identified as the model's knowledge storage substrates, are precisely the components where Muon shines (Figure 1, Table 1). The QK attention weights, despite having identical parameter count and being optimized under the same norm, show minimal benefit.

This diagnostic move is significant because it **reframes the optimizer design problem**. Rather than asking "which norm should we optimize under?"—a question that provides no architectural guidance—the associative memory lens asks "which parameters have outer-product gradient structure, and how can we design optimizers that respect that structure?" This reframing suggests that optimizer design should be **parameter-role-aware**: different components of the same model may benefit from different update rules depending on their functional role. A QK matrix computing attention scores has fundamentally different gradient structure than a VO matrix storing key-value associations, and a good optimizer would recognize this.

The finding is fundamental rather than incremental because it **explains a previously unexplained gap** in the theoretical understanding of Muon. The existing steepest-descent interpretation had an explicit limitation—Li & Hong (2025) and Shen et al. (2025) established convergence for Muon but could not demonstrate any rate advantage over Adam. The associative memory lens provides the missing piece: Muon's advantage is not about faster convergence in general, but about **balanced convergence across frequencies** in a specific parameter regime that dominates transformer training. The fact that this mechanism was hiding in plain sight—the outer-product decomposition of associative memory gradients—but went unnoticed because no one had connected the two literatures, makes this a genuine conceptual contribution rather than a technical refinement.

The empirical basis for this innovation is the component ablation in Section 4.1, particularly the "Combined Configurations" result that Muon(VO, FFN) with Adam on QK nearly recovers full Muon performance (validation loss 3.5858 vs. 3.5654 in the non-gated setting; 3.5312 vs. 3.5125 in the gated setting at 10,000 steps). This finding is not preordained by parameter counting—QK and VO have the same parameter count, yet VO provides essentially all the benefit. It is also not merely a consequence of which parameters are "important"—FFN weights are important and do benefit, but the fact that QK weights are equally important for model function yet do not benefit from Muon requires the associative memory explanation.

---

### Innovation 2: The Concept of Optimizer-Induced Isotropy as a Training Diagnostic

The paper introduces a new way of thinking about what optimizers do to learned representations: **not just whether they minimize loss, but what spectral structure they imprint on the weight matrices they produce**. The four metrics—normalized SVD entropy, effective rank, Top-k energy fraction, and eigenvalue quantile ratio—form a diagnostic toolkit that characterizes the "evenness" of learned knowledge representation.

Prior work on optimizer analysis focuses overwhelmingly on **optimization trajectories**: convergence rates, loss curves, gradient norms, Hessian structure. Even the spectral analyses that exist (Liu et al., 2025) aggregate across all parameters, obscuring the per-component dynamics that reveal the mechanism. The paper's innovation is to treat the **final weight matrix spectrum as an object of study in its own right**, and to connect spectral isotropy to a functional property: balanced knowledge storage.

What makes this concept distinctive is the **stability finding**. Muon's weight matrices not only have higher isotropy than Adam's—they have **negligible variance across random initializations**, while Adam's isotropy oscillates wildly and has large seed-to-seed error bars (Figure 2). This is more than a quantitative difference; it reveals a qualitative property of the optimizer. Muon's SVD normalization acts as a **stabilizing mechanism** that drives the weights toward a consistent spectral structure regardless of initialization. Adam's element-wise sign operation, by contrast, amplifies initialization-dependent differences, producing final weight matrices whose singular value distributions depend sensitively on random initial conditions.

The significance of this finding extends beyond Muon. It suggests that **isotropy is a trainable property** that can be directly influenced by optimizer choice, not just an architectural constraint. Prior work often treats representation diversity as something that emerges from architecture (e.g., attention heads specializing in different patterns) or from explicit regularization (e.g., orthogonality penalties). The paper demonstrates that optimizer choice alone can produce dramatically different spectral structures in the same architecture, with downstream consequences for knowledge acquisition. This opens a new axis for optimizer design: an optimizer can be evaluated not just by the loss it achieves, but by the **spectral health** of the representations it produces—whether knowledge is stored in a distributed, balanced way or concentrated in a dominant few components.

The Top-10 energy fraction metric is particularly revealing as a single-number summary. Under Muon, VO weights in the non-gated FFN model maintain Top-10E around 0.15–0.20 (only 15–20% of total energy in the 10 largest components). Under Adam, the same metric oscillates between 0.30 and 0.70 (Figure 2a). A model where 70% of the representational energy is concentrated in 10 singular vectors is functionally a much lower-capacity associative memory than one where energy is broadly distributed—even if both models have the same parameter count and similar training loss. This reframes the optimizer comparison from "which converges faster?" to "which produces better distributed knowledge representations?"

The empirical basis is the spectral dynamics analysis in Section 4.2 (Figure 2), with the critical evidence being (a) the consistent gap in all four isotropy metrics throughout training, (b) the negligible Muon error bars versus large Adam error bars across 10 random seeds, and (c) the persistence of these patterns in both non-gated and gated FFN architectures and at both 160M and 0.7B model scales (Appendix C.2, Figures 6–7).

This contribution is fundamental rather than incremental because it introduces a new evaluative dimension for optimizers that did not previously exist in the literature. The concept of optimizer-induced isotropy as a diagnostic is novel, and the connection to balanced knowledge acquisition gives it immediate practical relevance beyond spectral aesthetics.

---

### Innovation 3: Proving That Adam's Performance Is Embedding-Geometry-Dependent (A Negative Result with Positive Implications)

One of the paper's most intellectually distinctive contributions is a **negative result stated as a theorem**: that Adam (SignGD) is not a uniformly reliable optimizer for associative memory learning—its ability to learn balanced representations depends on the geometry of the feature embeddings, and there exist perfectly valid embedding configurations where it fails catastrophically.

Prior theoretical analyses of Adam focus on **worst-case convergence guarantees** in non-convex optimization—they prove that Adam converges *somewhere*, not that it converges *well* for any particular problem structure. The existing empirical literature on Adam and heavy-tailed distributions (Kunstner et al., 2024) demonstrates that Adam handles class imbalance better than SGD, but does not probe the conditions under which Adam itself might fail. The default assumption in the field has been that Adam's element-wise adaptivity makes it robust to essentially any data geometry.

The paper challenges this assumption with an explicit construction. In Theorem 5.3 (Adam portion), the authors build orthonormal embedding matrices (satisfying Assumption 5.1 exactly) where Adam's update has:

$$\frac{\sigma_{\min}(G_{\text{SignGD}}(W_0))}{\sigma_{\max}(G_{\text{SignGD}}(W_0))} \leq 25\%$$

and

$$\varrho^\epsilon_{\text{SignGD}} = O(\epsilon^{-0.7} K^{-0.3})$$

The embedding construction is not pathological—it uses 3×3 rotation matrices parameterized by Euler angles, which are standard orthonormal bases. The Euler angles (3.638, 2.949, 5.218 for $\tilde{E}$; 1.715, 0.876, 3.098 for $E$) are specific but not degenerate; they represent generic 3D rotations. The fact that **generic** (non-adversarial) embedding choices can cause Adam's effective rank to collapse to 25% of its maximum means the optimizer has a **structural blind spot** that was not previously recognized.

What makes this contribution significant beyond the specific result is the **conceptual reframing it enables**. The field has treated Adam's robustness as a given—it works well across diverse architectures and tasks, so it is considered "safe." The paper shows that this robustness is conditional: Adam works well when embeddings happen to be approximately decoupled (which they often are, as the orthogonality measurements on Llama3-8b-instruct in Figure 4a confirm), but it has no built-in mechanism to *ensure* balanced learning when embeddings overlap. Muon, by construction, is invariant to the choice of orthonormal embedding basis—its update is $-\tilde{E} E^\top$ up to $O(1/K)$ corrections regardless of the specific $\tilde{E}$ and $E$.

This is not merely a theoretical curiosity. The seed-to-seed variance in Adam's spectral metrics (Figure 2) is the empirical manifestation of this embedding-dependence. Different random initializations effectively sample different "effective embeddings" for the stored facts, and Adam's performance fluctuates accordingly. Muon's invariance to embedding geometry explains why its error bars are negligible—the optimizer converges to the same balanced solution regardless of the particular embedding realization sampled by the random initialization.

The comparison between the two cases in Theorem 5.3—decoupled embeddings (where Adam achieves $\varrho^\epsilon_{\text{SignGD}} \geq 1-\epsilon$, matching Muon) versus coupled embeddings (where Adam collapses to $O(\epsilon^{-0.7}K^{-0.3})$) —is particularly instructive. It shows that the property distinguishing the two regimes is **whether the element-wise sign operation preserves the outer-product structure of the gradient**. Under decoupled embeddings, the gradient is diagonal, and sign preserves the structure (each diagonal entry is individually normalized). Under coupled embeddings, the gradient has off-diagonal structure, and sign destroys the alignment between the update matrix and the fact directions.

The significance of this contribution is that it **explains when and why Adam can fail** in a way that was previously opaque. It also provides a **principled motivation for Muon**: Muon is not just a different norm choice—it is specifically designed to be invariant to embedding basis, which is a property that matters for associative memory learning but is invisible from the norm-based perspective. This is a fundamental theoretical insight, not an incremental refinement, because it identifies a structural limitation in the most widely used optimizer in deep learning and proves that an alternative optimizer (Muon) provably avoids it.

The empirical anchor for this innovation is Figure 4 and the accompanying theoretical results in Section 5.2. The contrast between Adam's behavior under decoupled versus coupled embeddings (Figure 4b: SignGD with decoupled embeddings tracks Muon; SignGD with coupled embeddings tracks GD) provides direct experimental validation of the theoretical claim.

---

### Innovation 4: The Head-Tail Gap as the Primary Consequence of Optimizer-Induced Imbalance

The paper identifies and quantifies a specific **failure mode of element-wise adaptive optimizers** on heavy-tailed data: not that they fail to learn tail classes entirely, but that tail-class learning **stagnates** once head classes are saturated, producing a persistent head-tail accuracy gap that widens with training rather than closing.

This is distinct from the standard narrative about imbalanced learning. The conventional story (grounded in works like Kunstner et al., 2024) is that adaptive optimizers help with heavy-tailed data by preventing head-class gradients from completely overwhelming tail-class gradients early in training. Under this view, Adam's advantage over SGD is about **initial access**—Adam gives tail classes enough gradient signal to begin learning, whereas SGD starves them entirely. The implicit assumption is that once tail classes start learning, they eventually catch up.

The paper's heavy-tailed knowledge task reveals that the dynamics are more subtle. All optimizers—Muon, Adam, and even SGD—learn head classes rapidly (Groups 0–2 reach near-perfect accuracy by 2,000 steps in Figures 3b–3d). The divergence happens in the **mid-to-late training regime** (steps 2,000–10,000), where head classes are already well-learned but tail classes are still climbing. Under Adam, the tail-group learning curves (Groups 13–15, Figure 3c) show a distinctive pattern: initial improvement followed by **plateau or decline**. At 5,000 steps, Adam's Group 15 accuracy is 0.110; at 10,000 steps, it reaches 0.264—an improvement, but dramatically slower than Muon, which reaches 0.976 by 10,000 steps (Table 5 vs. Table 4). The gap between Group 0 (head) and Group 15 (extreme tail) under Adam at 10,000 steps is approximately 0.74; under Muon it is approximately 0.02.

The innovation is in **identifying the mechanism of this stagnation**. Plateaus in tail-class learning under Adam are not caused by tail-class gradients being too small in absolute terms—they are caused by head-class gradients continuing to dominate the update direction even after head-class accuracy is near-perfect. This is because cross-entropy loss never saturates to zero gradient—even at 99% accuracy, the gradient for a head class remains non-zero because the model can always push the probability higher. These persistent head-class gradients, when their frequencies are orders of magnitude larger than tail-class frequencies, overwhelm the tail-class gradient signal in the aggregate parameter update. Adam's element-wise sign normalization does not solve this because it normalizes per-entry, not per-fact-direction—head-class contributions can still dominate if they appear in more matrix entries (which they do under coupled embeddings) or if their per-entry magnitudes are larger (which they are due to frequency weighting).

Muon solves this by normalizing at the level of **singular directions**, which correspond to fact-level contributions rather than parameter-level contributions. By equalizing the singular values of the gradient, Muon ensures that the update magnitude allocated to each fact direction is independent of frequency. This is the mechanism that allows tail classes to continue learning after head classes saturate—they receive equal "update budget" at every step, regardless of how many times the head classes appeared in the training data.

This contribution is significant because it provides a **unified explanation for three previously separate phenomena**: (1) why Muon outperforms Adam on language modeling (Figure 1), (2) why Muon's weights are more isotropic (Figure 2), and (3) why Muon achieves better tail-class accuracy (Figure 3). These are not independent findings—they are manifestations of the same underlying mechanism (frequency-independent updates → balanced representation → head-tail gap closure), viewed through different lenses (loss curves, weight spectra, knowledge task accuracy).

The empirical basis is the heavy-tailed knowledge task in Section 4.3, with Tables 3–5 providing the precise head-tail gap measurements at multiple training checkpoints. The critical evidence is the **trajectory shape**—Muon's tail-class accuracy curves continue to rise through 10,000 steps, while Adam's flatten out—rather than any single snapshot number. The hybrid optimizer experiments (Figures 3e–3f) provide the causal link: applying Muon to VO+FFN recovers Muon-like tail performance (Group 15 reaches 0.954 vs. full Muon's 0.976), while applying Muon to QK only leaves tail performance near Adam's level (Group 15 at 0.286 vs. Adam's 0.264), confirming that the head-tail gap closure is specifically mediated through the associative memory parameters.

This is a fundamental contribution rather than incremental because it **changes the optimization objective** from "minimize aggregate loss" to "minimize loss while ensuring balanced learning across frequencies." The latter is a stricter criterion that is invisible in standard optimization benchmarks but critically important for knowledge-intensive tasks where tail facts represent the majority of unique knowledge. The finding that Adam, the field's default optimizer, fails this criterion under certain embedding geometries—while Muon passes it provably—has immediate implications for how LLMs should be trained.

## 5. Experimental Analysis

### Evaluation Methodology

- **Dataset.** The paper uses two primary datasets. The first is FineWeb (Penedo et al., 2024), used for the component ablation experiments in Section 4.1 to measure validation loss during language model pretraining. The second is a synthetic biographical QA dataset constructed following Allen-Zhu & Li (2024), containing structured biographical information (name, birthdate, birthplace, educational institution, major, employer, workplace) for over 200,000 uniquely named individuals, used for the heavy-tailed knowledge task in Section 4.3. The synthetic dataset's class frequencies are controlled to follow a power-law distribution with parameter m=15, producing 32,768 total classes where the head class has 196,608 QA samples and each of the 16,384 extreme tail classes has only 6 QA samples.

- **Base model(s).** The primary model is a 160M-parameter NanoGPT transformer, evaluated with both non-gated and gated FFN variants. This model size is chosen as representative of the scale where careful optimizer comparisons can be conducted with multiple random seeds and extensive component-level ablations within reasonable compute budgets. For scaling validation, experiments are extended to a 0.7B-parameter NanoGPT model (Appendix C.2). For the embedding orthogonality measurement (Figure 4a), the paper uses the pretrained Llama3-8b-instruct model (Dubey et al., 2024), extracting embeddings from FFN layers 5, 10, 15, 20, 25 across 3,000 knowledge items from the Counterfact dataset (Meng et al., 2022a).

- **Metrics.** Three categories of metrics are used. For language modeling, the metric is **validation loss** on the FineWeb dataset, measured at 100-step intervals throughout 10,000 training steps. For the heavy-tailed knowledge task, the metric is **First Token Accuracy (FTA)**, which checks whether the model's highest-probability first token of the answer matches the ground-truth answer token — this directly measures associative memory retrieval without autoregressive decoding artifacts. For spectral analysis, four isotropy metrics are computed per weight matrix: **normalized SVD entropy** ($H_{\text{norm}}(\sigma) = -\frac{1}{\log n}\sum_i q_i \log q_i$, where $q_i = \sigma_i^2/\sum_j \sigma_j^2$), **effective rank** ($\text{eRank}(\sigma) = \exp(-\sum_i q_i \log q_i)$), **Top-k energy fraction** ($\text{TopE}_k(\sigma) = \sum_{i=1}^k \sigma_i^2 / \sum_{j=1}^n \sigma_j^2$, with k=10), and **eigenvalue quantile ratio** ($Q_{75/25}(\sigma) = Q_3(\{\sigma_i^2\})/Q_1(\{\sigma_i^2\})$). Higher isotropy corresponds to larger entropy and effective rank, and smaller Top-k energy and quantile ratio.

- **Baselines.** The paper compares three optimizers throughout: **(1) Adam** (Kingma & Ba, 2015) with β₁=0.8, β₂=0.95 for the 160M model and β₁=0.9, β₂=0.95 for the 0.7B model, no weight decay or Nesterov acceleration; **(2) Muon** (Jordan et al., 2024) with momentum μ=0.95 and Newton-Schulz iterations for approximate orthogonalization; **(3) SGD with momentum** as an additional baseline in the heavy-tailed knowledge task (Section 4.3). For the theoretical one-layer model in Section 5, the paper analyzes **GD** (standard gradient descent), **SignGD** (Adam with β₁=β₂=0, reducing to element-wise sign normalization), and **Muon** (SVD-based spectral normalization without momentum), all with zero initialization W₀=0.

- **Generation budget / compute accounting.** The paper does not use generation budgets (as in LLM inference-time compute studies). Instead, compute is measured in **training steps** (up to 10,000 for the 160M model on FineWeb and the heavy-tailed task). All optimizer comparisons use the same number of training steps and the same model architecture, so the per-step compute is identical across configurations — the only variable is the optimizer's update computation, which is not separately accounted for (SVD in Muon is more expensive per step than element-wise operations in Adam, but this cost is not included in any efficiency comparison). For the one-layer theoretical model (Section 5), compute is measured in **optimization steps** (one-step or multi-step), with step sizes swept or fixed according to a predetermined schedule.

- **Cross-validation / statistical protocol.** For the FineWeb experiments, each configuration is run with **10 different random seeds**, and results are reported as mean ± standard deviation (shown as error bars in Figures 2–3 and Tables 3–5). Learning rates are tuned via grid search over {1×10⁻¹, 5×10⁻², 2×10⁻², 1×10⁻², 5×10⁻³, 2×10⁻³, 1×10⁻³, 5×10⁻⁴, 2×10⁻⁴} for both Adam and Muon on the full-model setting; the chosen learning rates are then **fixed** for all component ablation experiments without per-configuration retuning. For the heavy-tailed knowledge task, results are reported as mean ± standard deviation across multiple seeds, with per-group First Token Accuracy tabulated at three checkpoints (2,000, 5,000, and 10,000 steps). For the theoretical one-layer model, the explicit parameter settings are K=d=999, α=0.8, β=0.2, and the specific Euler angles for the support-coupled embedding construction are given to 3 decimal places. There is no train/validation/test split for the theoretical model — it operates on the population loss directly.

---

### Main Quantitative Results

#### Component Ablation: VO+FFN as Muon's Primary Beneficiaries (Section 4.1)

The component ablation experiments ask a diagnostic question: if Muon is applied only to specific transformer components while Adam is used for everything else, which components recover the performance of full Muon? The answer, reported in Figure 1 and Table 1, is that **VO attention weights and FFN weights together nearly recover the full-Muon trajectory**, while QK attention weights contribute negligibly.

In the **non-gated FFN** setting at 10,000 training steps (Figure 1a, 1c; Table 1):
- Full Muon achieves validation loss **3.5654**
- Full Adam achieves **3.9242** (gap of 0.3588)
- Muon on VO Attn only (with Adam on QK and FFN): **3.7644** — recovers 44.6% of the gap
- Muon on QK Attn only (with Adam on VO and FFN): **3.8925** — recovers only 8.8% of the gap
- Muon on VO+FFN combined (with Adam on QK): **3.5858** — recovers **94.3%** of the gap, leaving only 0.0204 above full Muon
- Muon on W_out only: **3.7023** — stronger than W_in only (3.7170), consistent with W_out being a more direct associative memory store

Breaking VO into components: Muon on W_O only (3.7712) outperforms Muon on W_V only (3.8301). Breaking FFN in the VO+FFN combined settings: Muon(VO Attn, W_out) with Adam on QK and W_in achieves **3.6054**, nearly matching full VO+FFN (3.5858), indicating W_out is the dominant FFN contributor in the non-gated architecture.

In the **gated FFN** setting (Figure 1b, 1d; Table 1):
- Full Muon: **3.5125**
- Full Adam: **3.8837** (gap of 0.3712)
- Muon on VO Attn only: **3.6874** — recovers 52.9% of the gap
- Muon on QK Attn only: **3.8518** — recovers only 8.6%
- Muon on VO+FFN combined: **3.5312** — recovers **95.0%** of the gap, leaving 0.0187 above full Muon
- Within FFN, W_in, W_gate, and W_out all show comparable independent benefits (range 3.7843–3.7918 at 10k steps), and the combined configurations show nuanced interactions: VO+Win+Wgate+W_out (i.e., full FFN) performs best, but the gap between different FFN sub-combinations is small (3.5312 to 3.5833)

The paper explicitly rules out the possibility that Muon's limited QK benefit is due to suppressing attention logit explosion. Appendix C.1 (Table 2) measures per-layer MaxLogit values (the maximum pre-softmax attention score) across all 12 layers of the 160M model under full Muon with RMSNorm applied to Q and K (following Gemma 3, Kamath et al., 2025). The values range from 5.613 to 8.396, remaining moderate and showing no runaway growth, confirming that QK weights under Muon are not experiencing a fundamentally different attention dynamic that would confound the comparison.

These results are **not a trivial consequence of parameter counting**. QK and VO have the same parameter count per attention head (the model does not use grouped-query attention), yet VO delivers roughly 5× the loss reduction of QK when each is individually switched to Muon. The paper attributes this asymmetry to the functional role: VO and FFN serve as associative memory stores, while QK computes attention patterns that do not have the same outer-product structure.

**Scaling to 0.7B (Appendix C.2, Figure 5).** The same pattern holds at larger scale. For the non-gated FFN 0.7B model at 10,000 steps:
- Full Muon: **2.91**
- Full Adam: **3.07** (gap of 0.16)
- Muon(VO Attn, FFN) with Adam(QK): **~2.92** — recovers nearly the entire gap
- Muon(QK Attn, FFN) with Adam(VO): substantially worse, closer to full Adam

For the gated FFN 0.7B model:
- Full Muon: **2.96**
- Full Adam: **3.15** (gap of 0.19)
- Muon(VO Attn, FFN) with Adam(QK): **~2.98** — again nearly recovers full Muon

These scaling results confirm that the VO+FFN finding is not an artifact of the 160M scale and persists to models large enough to be practically relevant.

---

#### Spectral Dynamics: Muon Produces Consistently More Isotropic Weights (Section 4.2)

The spectral analysis in Figure 2 tracks four isotropy metrics for VO weights and W_out weights throughout the 10,000-step training run, averaged over 10 random seeds, for both non-gated and gated FFN architectures.

**VO weights (non-gated, Figure 2a):**
- **SVD entropy:** Muon maintains approximately 0.75–0.85 throughout training with negligible error bars. Adam starts at ~0.45, oscillates between 0.40 and 0.65, and shows large seed-to-seed variance (error bars spanning ~±0.10).
- **Top-10 energy:** Muon stays at 0.15–0.20. Adam fluctuates between 0.30 and 0.70 — at some checkpoints, 70% of total energy is concentrated in the top 10 singular directions.
- **Effective rank:** Muon maintains ~300–350 (out of maximum possible). Adam oscillates between 100 and 250, meaning the matrix uses fewer than half the effective dimensions under Adam.
- **Q75/Q25 ratio:** Muon stays near 1–3 (the interquartile range of eigenvalues is small). Adam ranges from 5 to over 100, with massive error bars.

**W_out weights (non-gated, Figure 2c):**
- **SVD entropy:** Muon near 1.0 (almost perfectly isotropic). Adam oscillates around 0.5 with large variance.
- **Top-10 energy:** Muon near 0.0 (negligible concentration). Adam ranges from 0.2 to 0.8.
- **Effective rank:** Muon ~550–600 (near theoretical maximum). Adam ranges from 100–400.
- **Q75/Q25 ratio:** Muon near 1. Adam ranges from 10 to over 100.

**Gated FFN (Figures 2b, 2d):** The patterns are qualitatively identical, though the absolute entropy values under Adam are somewhat higher (shifted up by ~0.1–0.2), suggesting the gating mechanism partially mitigates but does not eliminate Adam's spectral collapse.

The **seed-to-seed variance** is perhaps the most diagnostically important aspect of these results. Muon's error bars are consistently negligible — the optimizer converges to essentially the same spectral structure regardless of random initialization. Adam's error bars are large and persistent throughout training, indicating that the final spectral structure is highly initialization-dependent. The paper interprets this as the empirical manifestation of Adam's theoretical embedding-dependence (Theorem 5.3): different initializations sample different effective embedding geometries, and Adam amplifies these differences while Muon suppresses them.

Additional spectral analyses for W_in and W_gate (Appendix C.3, Figure 8) confirm the same pattern, with Muon consistently producing higher entropy, higher effective rank, lower Top-10 energy, and lower Q75/Q25 ratio than Adam. The 0.7B model analysis (Appendix C.2, Figures 6–7) further confirms these trends at larger scale.

---

#### Heavy-Tailed Knowledge Task: Muon Closes the Head-Tail Gap (Section 4.3)

The heavy-tailed knowledge task measures First Token Accuracy (FTA) stratified by class frequency group, where Group 0 has 196,608 QA samples per class and Group 15 has only 6. The results are shown in Figure 3 and quantified in Tables 3–5 (non-gated FFN) and Tables 6–8 (gated FFN).

**Key aggregate finding at 10,000 steps (Table 5, non-gated FFN):**
- Full Muon, Group 15 (extreme tail): **0.976 ± 0.006**
- Full Adam, Group 15: **0.264 ± 0.048**
- SGD+Momentum, Group 15: **0.126 ± 0.021**
- Muon(VO, FFN) & Adam(QK), Group 15: **0.954 ± 0.021**
- Muon(QK) & Adam(VO, FFN), Group 15: **0.286 ± 0.039**

The head-tail gap (Group 0 minus Group 15) under full Muon is approximately 0.024; under full Adam it is approximately 0.736. Muon reduces the gap by **~30×**.

**Training dynamics reveal the mechanism (Figure 3):**
- **Head classes (Groups 0–2):** All optimizers reach near-perfect accuracy (>0.95) by ~2,000 steps. Muon and Adam are statistically indistinguishable on head classes.
- **Mid-frequency classes (Groups 5–10):** Both Muon and Adam perform well, with Muon showing a consistent but modest advantage (e.g., Group 8 at 5,000 steps: Muon ~0.90, Adam ~0.85 from visual inspection of Figure 3).
- **Tail classes (Groups 11–15):** This is where the optimizers diverge dramatically. Adam's tail-class learning curves show a distinctive pattern: initial improvement followed by **plateau or decline**. At 5,000 steps (Table 4, non-gated), Adam's Group 13 accuracy is 0.298 ± 0.074 and Group 15 is 0.110 ± 0.027. By 10,000 steps, Group 13 reaches 0.890 ± 0.042 and Group 15 reaches 0.264 ± 0.048 — Group 13 eventually learns, but Group 15 remains poorly learned. Muon's tail-class learning curves, by contrast, show steady improvement throughout training: Group 13 reaches 0.964 ± 0.023 at 5,000 steps and 1.000 ± 0.000 at 10,000; Group 15 reaches 0.320 ± 0.028 at 5,000 and 0.976 ± 0.006 at 10,000.

The **error bars** for tail-class accuracy under Adam are substantially larger than under Muon. At 10,000 steps, Adam's Group 15 accuracy spans approximately 0.216–0.312 across seeds (±0.048 standard deviation), while Muon spans 0.970–0.982 (±0.006). This is consistent with the spectral analysis finding that Adam's final representations are initialization-dependent.

**Gated FFN results (Figure 9, Tables 6–8)** replicate the pattern:
- Full Muon, Group 15 at 10,000 steps: **0.994 ± 0.006**
- Full Adam, Group 15 at 10,000 steps: **0.244 ± 0.085** (even larger variance than non-gated)
- Muon(VO, FFN) & Adam(QK), Group 15: **0.990 ± 0.010** — nearly identical to full Muon
- Muon(QK) & Adam(VO, FFN), Group 15: **0.274 ± 0.042** — statistically indistinguishable from full Adam

The hybrid optimizer results provide the **causal evidence** that VO+FFN is the mechanism: applying Muon only to VO+FFN with Adam on QK achieves 0.954–0.990 tail accuracy (depending on architecture), closely tracking full Muon. Applying Muon only to QK with Adam on VO+FFN achieves 0.274–0.286 tail accuracy, essentially matching full Adam. The associative memory parameters are both necessary and sufficient for Muon's tail-class benefit.

---

#### One-Layer Model: Provable Balance vs. Provable Instability (Section 5)

The one-layer associative memory model formalizes the balanced learning concept through ϱ^ϵ_opt, the minimum correct-class probability across all K facts when at least one fact reaches probability ≥ 1-ε.

**Experimental results (Figures 4b, 4c).** These are computed for the one-layer model with K=d=999, α=0.8, β=0.2:

- **One-step protocol (Figure 4b):** The plot shows population loss L(W) vs. maximal probability gap Δ(W). For GD, the gap is large regardless of embedding type (the curve is shifted to the right). For Muon, the gap is small (<10⁻³) regardless of embedding type, and the curves for decoupled and coupled embeddings overlap almost perfectly. For SignGD, the behavior is split: under decoupled embeddings, SignGD tracks Muon (small gap); under coupled embeddings, SignGD tracks GD (large gap). This directly visualizes Adam's embedding-dependence.
- **Multi-step protocol (Figure 4c):** The same qualitative patterns persist across multiple optimization steps. Muon (both decoupled and coupled) maintains a small gap throughout training. GD maintains a large gap. SignGD splits by embedding type.

**Theoretical results (Theorem 5.3):**

For a two-class frequency structure where the first L=βK triplets have total probability mass α, and K→∞:

- **GD:** ϱ^ϵ_GD = O(ε^{-r(α,β)} K^{r(α,β)-1}) where r(α,β) = min{α(1-β)/(β(1-α)), β(1-α)/(α(1-β))} < 1 when α≠β. The worst-class probability vanishes polynomially in K — GD leaves tail classes essentially unlearned.

- **Muon:** ϱ^ϵ_Muon ≥ 1 - ε(1 + O(log K/K)). The worst-class probability approaches 1-ε as K grows — near-perfect balance regardless of α, β, or the specific orthonormal embeddings E, Ẽ. The Muon update is G_Muon(W₀) = -ẼE^⊤ + O(K⁻¹ Ẽ J_{K,K} E^⊤), which assigns equal update magnitude to all fact directions.

- **SignGD (Adam without momentum):** Constructed embeddings exist where ϱ^ϵ_SignGD = O(ε^{-0.7} K^{-0.3}), with the singular value ratio of the update satisfying σ_min/σ_max ≤ 25%. The exponents 0.7 and 0.3 are intrinsic to the embedding geometry, not the data imbalance — even with balanced data, Adam can fail under unfavorable embeddings.

**Multi-step extension (Theorem 5.4):** The Muon properties extend to any step t: G_Muon(W_t) = -ẼE^⊤ + O(K⁻¹ Ẽ J_{K,K} E^⊤) and ϱ^ϵ_Muon ≥ 1 - ε(1 + O(log K/K)). The proof shows by induction that W_t maintains the structure Ẽ(Λ_t + C_t)E^⊤ where Λ_t = diag(a_t·I_L, b_t·I_{K-L}) with a_t = b_t (head and tail diagonal entries remain equal) and C_t entries are O(a_t/K).

**Embedding orthogonality validation (Figure 4a):** For Llama3-8b-instruct on 3,000 Counterfact knowledge items, the average pairwise angles between key embeddings E_i and between value embeddings Ẽ_i range from ~75° to ~88° across FFN layers 5–25, with a slight increasing trend toward higher layers. These angles are near 90°, supporting the orthonormality assumption (Assumption 5.1). Appendix C.6 (Figure 10) reports similar findings on the ZsRE dataset (Levy et al., 2017), confirming that approximate orthonormality is not dataset-specific.

---

### Ablation Studies and Robustness Checks

**Non-gated vs. gated FFN architectures (Figures 1a–1d, Table 1):** The VO+FFN finding holds in both architectures. However, the decomposition within FFN differs: in the non-gated setting, W_out alone nearly recovers full VO+FFN performance (3.6054 vs. 3.5858), while in the gated setting, all three FFN matrices (W_in, W_gate, W_out) contribute more evenly, and no single FFN matrix recovers as much of the full VO+FFN performance alone. This is an architectural sensitivity — the gating mechanism distributes Muon's benefit across FFN matrices rather than concentrating it in W_out. The overall VO+FFN benefit remains robust to this architectural variation.

**WO vs. WV within VO (Figures 1a–1d, Table 1):** When VO is broken into individual components, W_O consistently outperforms W_V in the independent-block setting: Muon(O Attn) yields 3.7712 (non-gated) and 3.7604 (gated), while Muon(V Attn) yields 3.8301 (non-gated) and 3.7482 (gated). In the combined configurations, O+FFN (3.6042 non-gated, 3.5634 gated) outperforms V+FFN (3.6702 non-gated, 3.7185 gated). This asymmetry is consistent with W_O being the more direct associative memory store (it maps from value representations to output representations), while W_V maps from input representations to value representations and is one step removed from the final output.

**Hybrid optimizer configurations (Figures 3e–3f, Tables 3–5):** The causal role of VO+FFN is tested by applying Muon to VO+FFN with Adam on QK (Figure 3e), and the reverse (Figure 3f). The results are definitive: Muon(VO, FFN) & Adam(QK) achieves Group 15 accuracy 0.954 ± 0.021 (vs. 0.976 for full Muon), while Muon(QK) & Adam(VO, FFN) achieves 0.286 ± 0.039 (vs. 0.264 for full Adam). VO+FFN is both sufficient (the first hybrid nearly matches full Muon) and necessary (the second hybrid fails to improve over Adam). The gated FFN replicates this (Tables 6–8): Muon(VO, FFN) & Adam(QK) achieves Group 15 accuracy 0.990 ± 0.010; Muon(QK) & Adam(VO, FFN) achieves 0.274 ± 0.042.

**Embedding support-decoupled vs. support-coupled (Figures 4b, 4c):** The one-layer model experiments explicitly test whether the embedding geometry matters. For GD, it does not — the curves for decoupled and coupled embeddings overlap (the gradient structure is the same up to orthogonal transformation, and GD is not basis-dependent). For Muon, it does not — the curves overlap (Muon's SVD normalization is invariant to the orthonormal basis). For SignGD, it matters critically — decoupled embeddings produce Muon-like balance, coupled embeddings produce GD-like imbalance. This ablation isolates embedding-dependence as the specific failure mode of element-wise sign normalization.

**One-step vs. multi-step optimization (Figures 4b vs. 4c):** The qualitative results are consistent across both protocols. This is non-trivial — it means the mechanism (Muon's frequency-independent updates, Adam's embedding-dependent sign collapse) is present from the first step and persists throughout training, rather than being an artifact of initialization or a phenomenon that only emerges after many iterations.

**SGD + Momentum baseline (Figures 3d, Tables 3–5):** SGD+Momentum serves as a lower bound, demonstrating that both Adam and Muon substantially outperform a non-adaptive baseline on heavy-tailed data. SGD's Group 15 accuracy at 10,000 steps is 0.126 ± 0.021, confirming the known weakness of non-adaptive methods on class-imbalanced problems (Kunstner et al., 2024). This establishes that some form of adaptivity is necessary, but the choice of adaptivity mechanism (element-wise sign vs. SVD normalization) determines whether tail classes are eventually learned.

**Logit stability verification (Appendix C.1, Table 2):** To rule out the hypothesis that Muon's limited QK benefit is due to suppressing attention logit explosion (a known issue in some architectures, Team et al., 2025), per-layer MaxLogit values are measured on the 160M model under full Muon with RMSNorm applied to Q and K. The values range from 5.613 to 8.396, remaining moderate and stable. This is important because if Muon were suppressing an attention instability that Adam fails to control, the QK benefit would appear small not because QK is intrinsically less responsive to Muon, but because the baseline Adam configuration is pathologically poor on QK. The MaxLogit measurement rules out this confound.

**Embedding orthogonality verification on multiple datasets (Figure 4a, Appendix C.6 Figure 10):** The orthonormality assumption (Assumption 5.1) is validated on two datasets (Counterfact and ZsRE) using Llama3-8b-instruct. Both show average pairwise embedding angles near 90° across layers, with Counterfact ranging ~75°–88° and ZsRE showing similar values (exact ranges not quoted in the main text). The consistency across datasets strengthens the claim that approximate orthonormality is a general property of LLM associative memory embeddings, not an artifact of one evaluation set.

**Scaling to 0.7B model (Appendix C.2, Figures 5–7):** The component ablation, spectral dynamics, and optimizer comparisons are all replicated at 0.7B scale. The key findings — VO+FFN nearly recovers full Muon, Muon produces more isotropic weights, and Adam shows larger spectral variance — all persist. This is a critical robustness check because it addresses the concern that the 160M findings might be scale-dependent.

---

### Critical Assessment

The central claim of this paper is that **Muon outperforms Adam because its SVD-based update rule aligns with the outer-product structure of linear associative memories, enabling more balanced learning of tail classes in heavy-tailed distributions**. The paper deploys four distinct lines of evidence: component ablations (identifying VO+FFN as Muon's beneficiaries), spectral analysis (showing Muon produces more isotropic weights), heavy-tailed knowledge task results (showing Muon closes head-tail accuracy gaps), and theoretical analysis (proving balanced learning for Muon and embedding-dependence for Adam).

**Does the evidence support the central claim?**

The component ablation evidence (Section 4.1) convincingly establishes that Muon's benefit concentrates in VO and FFN parameters. The magnitude of the effect is large and consistent: VO+FFN recovers ~95% of the full-Muon performance in both the non-gated and gated 160M models and at 0.7B scale. The control experiment — Muon on QK with Adam on VO+FFN — shows that QK provides negligible benefit despite having the same parameter count as VO. This specificity is the strongest single piece of evidence in the paper because it is difficult to explain under any hypothesis that does not involve the functional role of the parameters.

However, the evidence connecting VO+FFN to associative memory structure is **indirect**. The paper cites prior work (Geva et al., 2020; Bietti et al., 2023; Meng et al., 2022a) establishing that these parameters function as associative memories, but does not itself demonstrate that the VO+FFN weights in its trained 160M models actually encode factual associations in a way that QK weights do not. A direct test — for example, measuring whether editing VO+FFN weights changes factual recall while editing QK weights does not — would strengthen the associative memory interpretation. The current evidence shows correlation (Muon helps VO+FFN; VO+FFN is known to be associative memory) but not direct causation (Muon helps *because* VO+FFN is associative memory).

The spectral analysis (Section 4.2) provides strong evidence that Muon and Adam produce qualitatively different weight matrix structures. The four metrics consistently show Muon producing more isotropic spectra with negligible seed-to-seed variance. This is a genuine empirical finding that does not depend on the associative memory interpretation. However, the connection between isotropy and balanced knowledge acquisition is largely **theoretical rather than directly measured**. The paper does not, for example, show that specific singular vectors in the trained weight matrices correspond to specific facts from the heavy-tailed knowledge task — such a demonstration would require probing or decoding the weight matrices, which is not attempted. The isotropy finding is consistent with the proposed mechanism but does not uniquely confirm it. An alternative hypothesis — that Muon simply regularizes the weights toward isotropy through its normalization, and this happens to improve generalization regardless of any frequency-balancing mechanism — is not ruled out.

The heavy-tailed knowledge task (Section 4.3) provides the most direct behavioral evidence. The head-tail gap closure under Muon (Group 15 at 0.976 vs. Adam's 0.264) is dramatic and practically significant. The hybrid optimizer experiments (Muon on VO+FFN vs. Muon on QK) provide causal evidence linking the component-level finding to the behavioral outcome. The training dynamics — Adam's tail-class learning stagnates while Muon's continues to improve — are consistent with the proposed mechanism of frequency-independent updates.

However, this task is **highly artificial**. The biographical QA dataset is synthetic, the class frequencies are exactly power-law with a specific exponent, and the task requires pure memorization of structured facts without the linguistic complexity of natural text. The finding that Muon learns rare facts better on this task does not necessarily imply that the same mechanism explains Muon's advantage on natural language pretraining (FineWeb), where the "facts" are not cleanly separable into discrete classes, the frequency distribution is more complex, and the model must simultaneously learn syntax, semantics, and world knowledge. The paper does not analyze per-token or per-entity prediction accuracy on the FineWeb validation set stratified by frequency — such an analysis would bridge the gap between the synthetic task and real language modeling.

The theoretical analysis (Section 5) provides a clean, provable demonstration of the proposed mechanism in a simplified setting. The key results — that Muon's update is invariant to embedding basis while Adam's is not, and that this leads to provable balance for Muon and provable potential for imbalance for Adam — are mathematically rigorous and well-matched to the empirical findings. The construction of explicit embeddings where Adam fails (Theorem 5.3) is particularly strong because it shows that the failure mode is not pathological — it occurs under generic orthonormal embeddings.

However, the gap between the one-layer model and real transformer training is **substantial**, and the paper does not attempt to close it. The one-layer model makes several assumptions that do not hold in practice: (1) the embeddings are exactly orthonormal rather than approximately so; (2) the associative memory is exactly linear rather than mediated by attention softmax and FFN activations; (3) the loss is population cross-entropy rather than empirical next-token prediction with finite samples; (4) there is no momentum (β₁=β₂=0 for Adam, μ=0 for Muon), which is a significant simplification given that momentum is known to interact with normalization in complex ways; (5) the model is trained from zero initialization, whereas real LLMs are randomly initialized with small weights. The paper's empirical verification of approximate orthonormality (Figure 4a) addresses concern (1), but concerns (2)–(5) remain. The multi-step extension (Theorem 5.4) addresses the one-step limitation but does not address the other simplifications.

**What is and is not tested.**

The paper **tests**: which components benefit from Muon (tested directly via ablation), what spectral structure Muon produces (tested directly via isotropy metrics), how Muon and Adam compare on a heavy-tailed pure-memory task (tested directly via FTA), and what the one-step/multi-step dynamics are in a one-layer associative memory model (tested theoretically).

The paper **does not test**: whether the spectral isotropy finding mediates the head-tail gap closure (there is no causal intervention that varies isotropy independently of optimizer choice), whether the specific singular vectors in trained Muon weights align with stored fact directions, whether the mechanism extends to natural language modeling where facts are not cleanly separable, whether Adam's performance on the heavy-tailed task can be improved by tuning its hyperparameters specifically for that task (β₁, β₂, ε were not swept for the knowledge task — the same values from FineWeb were used), whether Muon's advantage persists when Adam is given equivalent computational budget (Muon's SVD is more expensive per step than Adam's element-wise operations, and this cost is not accounted for in any efficiency comparison), and whether the mechanism holds for models with grouped-query attention or other architectural variants common in production LLMs.

**Missing experiments that would strengthen the paper:**

1. **A frequency-stratified evaluation on the FineWeb validation set.** Measuring per-token prediction accuracy binned by token frequency would directly test whether Muon's advantage on natural language is attributable to better tail-token prediction, extending the heavy-tailed knowledge task finding to real language modeling.

2. **A direct manipulation of isotropy.** For example, adding an explicit spectral regularization term to Adam that encourages isotropic weight spectra, and measuring whether this closes the head-tail gap on the knowledge task independently of Muon's update rule, would test whether isotropy is a causal mechanism or a correlated outcome.

3. **A comparison against Adam with the same per-step computational budget.** Muon's SVD (even with Newton-Schulz approximation) is more expensive than Adam's element-wise operations. A fair comparison would give Adam additional training steps to equalize total FLOPs. If Muon still wins under FLOPs-matched comparison, the mechanistic argument is strengthened; if Adam catches up, the advantage is about per-step efficiency rather than a fundamental learning mechanism.

4. **A probe of whether trained VO weights encode facts in an interpretable way under Muon vs. Adam.** For the heavy-tailed knowledge task, one could attempt to extract stored biographical facts from the weight matrices (e.g., via the technique of Meng et al., 2022a) and compare the quality of extraction for head vs. tail facts. This would directly test the claim that Muon stores tail facts more robustly.

5. **Sensitivity analysis on the power-law exponent.** The heavy-tailed task uses a fixed exponent (m=15). Varying the exponent would test whether Muon's advantage is robust to different degrees of imbalance or whether there is a threshold beyond which even Muon cannot learn tail classes.

6. **An experiment testing the orthogonality assumption's necessity.** Training the one-layer model with deliberately non-orthogonal embeddings (e.g., by adding Gaussian noise to the rotation matrices) and measuring whether Muon's balanced learning degrades gracefully or collapses would characterize the robustness of the mechanism.

**Where claims hold conditionally:**

- The claim that "Muon's update rule aligns with the outer-product structure of linear associative memories" holds **under the condition that embeddings are approximately orthonormal**. The theoretical proof assumes exact orthonormality (Assumption 5.1); the empirical validation (Figure 4a) shows approximate orthonormality (~75°–88° angles rather than exactly 90°). The paper does not analyze sensitivity to violations of orthonormality.

- The claim that "Muon enables more balanced and effective learning of tail classes in heavy-tailed distributions" holds **for the specific power-law distribution and model architecture tested**. The paper does not explore whether the mechanism generalizes to other heavy-tailed distributions (e.g., log-normal, Zipfian with different exponents).

- The claim that "Adam's performance is unstable and strongly dependent on the embedding structure" holds **in the sense that there exist embeddings where Adam fails, not that Adam always fails**. The paper explicitly demonstrates both a case where Adam succeeds (decoupled embeddings, ϱ^ϵ_SignGD ≥ 1-ε) and a case where Adam fails (coupled embeddings, ϱ^ϵ_SignGD = O(ε^{-0.7}K^{-0.3})). The practical question — how often real LLM embeddings fall into the failure regime — is not addressed beyond the orthogonality measurement, which is consistent with the decoupled regime where Adam performs well empirically.

- The finding that VO+FFN nearly recovers full-Muon performance holds **at the tested model scales (160M, 0.7B) and learning rates**. The paper acknowledges that the residual gap between VO+FFN and full Muon may be reducible with learning rate tuning, but does not perform this tuning. At larger scales, the relative importance of QK vs. VO may shift, particularly if attention patterns become more structured or if the model develops specialized heads.

**Overall assessment.** The paper provides a coherent, multi-faceted case for its central mechanism, with each piece of evidence addressing a different aspect of the claim (where the benefit concentrates, how the weights are structured, what the behavioral consequence is, why it happens theoretically). The component ablation and heavy-tailed knowledge task results are particularly strong, providing specific, quantitative, and causally-tested evidence. The theoretical analysis, while simplified, provides a clean mechanism that matches the empirical patterns. The main weaknesses are the gap between the synthetic knowledge task and natural language modeling, the lack of FLOPs-matched or isotropy-manipulation experiments to establish causality, and the limited exploration of conditions under which the mechanism might break down. The paper makes a convincing case that associative memory structure is a key part of the explanation for Muon's superiority, while leaving open the question of whether it is the complete explanation.

## 6. Limitations and Trade-offs

### VO+FFN Focus Leaves QK Contributions Unresolved

The paper's central empirical finding is that applying Muon to VO+FFN nearly recovers full-Muon performance, while applying Muon to QK provides negligible benefit. The authors treat this as evidence that VO+FFN *are* Muon's targets and QK *are not*. But the paper also reports that Muon produces more isotropic QK weights than Adam (Appendix C.3), which creates an unresolved tension: **Muon does change QK parameter structure, but this change does not translate to improved validation loss or tail-class accuracy**.

> "Empirically, we also find that Muon learns more isotropic QK weights than Adam. However, as discussed in Section 4.1, QK weights are not part of the linear associative memory mechanism and are therefore not expected to benefit from the isotropic property of the weight matrices."

This is an assertion, not a demonstration. It assumes that isotropy is only beneficial when combined with associative memory structure. But it is equally possible that the spectral benefits to QK *would* matter on tasks where attention pattern quality is the bottleneck—tasks requiring long-range dependencies, precise token-level disambiguation, or multi-hop reasoning—and that the FineWeb validation loss used in Section 4.1 is simply not the right metric to reveal this. The paper provides no experiment that isolates a QK-dependent task and measures whether Muon on QK outperforms Adam on QK for that task.

The practical consequence: a practitioner deciding which parameters to optimize with Muon vs. Adam in a production training run cannot determine from this paper whether QK parameters should be included. The recommendation to focus on VO+FFN is based on validation loss on a single dataset (FineWeb) and a single knowledge-intensive task. On a different task distribution—one where attention pattern quality matters more than knowledge storage capacity—the optimal assignment of Muon to QK versus Adam might differ. The paper offers no guidance for making this determination in new settings.

The hybrid optimizer experiments (Figures 3e–3f) partially address this by testing Muon(QK) with Adam(VO,FFN) on the heavy-tailed knowledge task, but this task is explicitly designed to be limited by associative memory capacity, not attention quality. That Muon(QK) fails to help on a memory-bottlenecked task is not evidence that it would not help on an attention-bottlenecked task. No such task is tested.

The paper does not attempt to mitigate this limitation—it acknowledges the asymmetry descriptively but does not explore whether QK's spectral improvement could be exploited in a different task regime. This is listed as future work only indirectly, through the suggestion that Muon's properties "may extend beyond outer products to higher-order tensor products" (Section 6), which would presumably include attention computations.

---

### Difficulty Estimation Cost Is Not Accounted for in the Associative Memory Mechanism

The paper's mechanistic explanation relies on the claim that Muon's SVD normalization equalizes update magnitudes across fact-specific gradient directions, which requires that those directions are identifiable in the gradient's singular value decomposition. However, **the SVD must be computed at every optimizer step** for every weight matrix to which Muon is applied. This is substantially more expensive than Adam's element-wise operations.

In practice, Muon implementations approximate the SVD using Newton-Schulz iterations (typically 5 iterations, as mentioned in Section 3), which is cheaper than a full SVD but still more expensive than element-wise sign and scaling. The paper **does not account for this per-step compute difference anywhere**. All comparisons use the same number of training steps, not the same wall-clock time or total FLOPs. The 2× speedup reported in prior work (Jordan et al., 2024; Liu et al., 2025) is measured in training steps to reach a target loss, not in FLOPs or GPU-hours. If Muon's per-step cost is higher than Adam's, the step-count advantage overstates the true computational advantage.

Even if Newton-Schulz iterations amortize to negligible overhead at scale (which is plausible given that the SVD is only computed for matrix-shaped parameters, not the full model, and matrix operations are hardware-efficient), the paper provides no measurement to support this. The 160M model training cost in GPU-hours is not reported. The 0.7B model scaling experiment (Appendix C.2) only reports validation loss curves, not training time. The one-layer theoretical model (Section 5) assumes Muon computes an exact SVD at every step—no approximation cost is modeled.

The practical consequence is that a practitioner choosing between Adam and Muon cannot determine from this paper whether the optimizer's per-step overhead is justified by the convergence speedup in wall-clock terms. If Newton-Schulz iterations add even 10% overhead per step, the effective 2× speedup in steps becomes a ~1.8× speedup in time—still substantial, but the margin matters for cost calculations in large training runs. If the overhead varies with model architecture or hardware (e.g., the SVD approximation becomes relatively more expensive when matrix dimensions are small or when the batch size is small), the paper provides no guidance.

This limitation is not acknowledged in the paper. The only mention of computational cost is in the context of the difficulty estimation framework from a different paper (which is not used here). The steepest-descent interpretation (Appendix A) treats Muon's update as a mathematical operation without discussing its computational realization.

---

### The One-Layer Theoretical Model Makes Assumptions That Do Not Hold in Real LLM Training

Section 5 provides the paper's theoretical foundation, but the gap between the one-layer model and actual transformer training is large in ways that directly affect whether the proved mechanism operates in practice.

**Assumption 5.1 (exact orthonormality):** $E^\top E = \tilde{E}^\top \tilde{E} = I_{K,K}$. The paper validates *approximate* orthonormality on Llama3-8b-instruct (Figure 4a), showing average pairwise embedding angles of ~75°–88° rather than exactly 90°. The theory provides no bound on how the balanced learning guarantee degrades when embeddings deviate from orthonormality. If embeddings have correlations of 0.1–0.3 (corresponding to angles of 75°–85°), the gradient's outer-product structure is no longer exactly diagonalizable in the embedding basis, and the SVD normalization no longer cleanly separates fact-specific directions. The paper does not analyze whether Muon's balanced learning property is robust to small embedding correlations or whether it requires near-exact orthogonality.

**Momentum is disabled ($\beta_1 = \beta_2 = 0$ for Adam, $\mu = 0$ for Muon):** Both theoretical analyses (Theorems 5.3, 5.4) analyze the optimizers without their momentum accumulators. However, the empirical experiments (Sections 4.1–4.3) use momentum ($\beta_1=0.8$ for Adam, $\mu=0.95$ for Muon). Momentum fundamentally changes the effective gradient direction—Adam with momentum applies element-wise sign normalization to an exponentially weighted moving average of past gradients, not to the instantaneous gradient. If past head-class gradients accumulate disproportionately in the momentum buffer (because they appear more frequently), the sign-normalized update may remain head-dominated even if the instantaneous gradient has become more balanced later in training. The theoretical analysis provides no insight into how momentum and SVD normalization interact—whether momentum amplifies or mitigates Adam's embedding-dependence, or whether Muon's momentum accumulator preserves the isotropic structure of the normalized updates across steps.

**Zero initialization ($W_0 = 0$):** Real transformers are randomly initialized with small weights. Under random initialization, the softmax scores at $t=0$ are not uniform $1/K$—they have random variation that depends on the initialization scale and the embedding geometry. This variation creates an initial head start for some facts over others, independent of their training frequencies. The theory's clean separation between "frequency determines which facts are head vs. tail" and "Muon's normalization equalizes frequencies" does not account for this initialization-induced imbalance. If the random initialization happens to align some fact directions with the gradient's dominant singular vectors, those facts may receive disproportionately large updates from Muon's SVD normalization as well (since the SVD is of the gradient, not of the weight matrix), potentially undermining the balanced learning guarantee.

**Population loss with exact frequencies:** The theory uses the population cross-entropy loss where class probabilities $p_k$ are exactly the specified frequencies. In real training, minibatch sampling introduces gradient noise—the empirical gradient in any given batch does not have the clean block-diagonal structure assumed in Proposition F.3. The paper does not analyze how minibatch stochasticity interacts with Muon's SVD normalization. If the gradient's singular value spectrum is noisy (fluctuating from batch to batch), Muon's normalization will amplify different noise directions at each step, potentially introducing variance that counteracts the balanced learning benefit.

The multi-step extension (Theorem 5.4) partially addresses the one-step simplification, showing that the structural properties ($a_t = b_t$, $c_t^{ij} = O(a_t/K)$) are preserved by induction. But the induction argument depends on the symmetry of the softmax scores (facts within the same frequency group have identical scores), which holds under exact Assumptions 5.1–5.2 but would break under embedding non-orthogonality, momentum, random initialization, or minibatch noise.

The practical consequence is that the theoretical mechanism, while clean and suggestive, does not provide quantitative guidance for practical Muon deployment. A practitioner cannot use Theorems 5.3–5.4 to predict how much tail-class improvement Muon will provide on their specific dataset, architecture, and training configuration. The theory demonstrates *that* balanced learning is possible, not *when* or *how much* it occurs under realistic conditions. The paper acknowledges none of these gaps explicitly—the assumptions are stated in Section 5 but their practical consequences are not discussed in Section 6.

---

### The Heavy-Tailed Knowledge Task Is Too Synthetic to Validate the Mechanism for Natural Language

The paper's strongest behavioral evidence for the associative memory mechanism comes from the heavy-tailed knowledge task (Section 4.3), where Muon achieves 0.976 FTA on extreme tail classes versus Adam's 0.264 at 10,000 steps. However, this task strips away essentially all complexity of natural language modeling except associative memory capacity.

**What the task removes:** (1) All factual associations are independent—knowing Ashton Hilda Older's birthplace provides no information about their employer, unlike natural text where facts are correlated and contextual. (2) The input format is rigid template-based QA—the model never sees facts embedded in narrative text, never needs to disambiguate entities from context, and never encounters the same fact expressed in different ways. (3) The "tail" is defined by entity frequency, not by linguistic pattern frequency—rare entities are queried using the same templates as common entities, so the only thing varying is how often the model sees a particular name-date-location tuple. (4) The vocabulary is constrained to the biographical attributes (names, dates, locations, institutions), which are presumably well-represented in the tokenizer and do not pose the out-of-vocabulary challenges that real tail tokens face. (5) The answer is always a single token (the first token of the answer string), so there is no multi-token decoding and no exposure bias from autoregressive generation.

Natural language pretraining on FineWeb involves none of these simplifications. Facts are correlated, expressed in diverse ways, embedded in complex linguistic context, and involve tokens with wildly varying frequencies. The "tail" in natural language includes not just rare facts but rare syntactic constructions, rare word senses, rare discourse patterns, and rare combinations of common elements. Muon's mechanism—equalizing gradient updates across orthogonal fact directions—assumes that these fact directions exist as identifiable components of the gradient's SVD. In natural language, it is unclear whether "facts" correspond to clean singular directions or whether the gradient spectrum is more continuously distributed, with no sharp separation between head and tail components.

The paper provides no evidence that the heavy-tailed task finding translates to natural language. The FineWeb validation loss improvement (Figure 1) could be driven entirely by better learning of frequent patterns (where Muon's isotropy might provide a different benefit, such as better generalization rather than tail-class learning) or by interactions between the associative memory mechanism and other aspects of language modeling that the synthetic task does not test. The paper does not stratify FineWeb validation loss by token frequency, entity frequency, or any other frequency-based metric that would bridge the synthetic and natural settings.

The causal evidence from hybrid optimizers (Muon on VO+FFN vs. Muon on QK) is only provided for the synthetic task (Figures 3e–3f), not for FineWeb. The component ablation on FineWeb (Figure 1) shows that VO+FFN recovers full-Muon validation loss, but this does not isolate whether the benefit is specifically from better tail-fact learning or from some other property of Muon's updates that improves VO+FFN optimization generally.

The paper does not acknowledge this gap between the synthetic and natural settings. The conclusion states that Muon "enables more balanced and effective learning of tail classes in heavy-tailed distributions" without qualifying that this has only been demonstrated for a synthetic factual recall task, not for natural language modeling.

---

### The Headline Efficiency Claims Are Not Cost-Adjusted

Prior work establishes that Muon trains transformers "nearly 2 times faster" than Adam (Liu et al., 2025; Jordan et al., 2024), measured in training steps to reach a target loss. This paper does not replicate or adjust this claim—it focuses on mechanism rather than efficiency—but it also does not provide the cost context that a practitioner would need to evaluate whether the mechanism's benefits justify switching optimizers.

**Per-step compute cost:** Muon computes (an approximation of) the SVD of the momentum-accumulated gradient for every matrix-shaped parameter at every step. For a transformer with $n$ matrix parameters, this requires $n$ Newton-Schulz iterations per step, each involving matrix multiplications. Adam requires only element-wise operations (sign, square, sqrt, addition) on the same parameters. While matrix multiplications are hardware-efficient, they are not free. The paper does not report wall-clock time per step, total training time, or FLOPs for any experiment. This makes it impossible to determine whether Muon's 2× advantage in steps translates to a 2× advantage in GPU-hours, or whether it is closer to 1.5× or 1.2×.

**Learning rate tuning cost:** The paper tunes learning rates via grid search for the full-model configurations and then fixes those rates for all component ablation experiments. This is methodologically sound for the ablation (it isolates the effect of switching optimizer assignment), but it means the reported performance of partial-Muon configurations may be suboptimal relative to what could be achieved with dedicated learning rate tuning. The paper acknowledges this:

> "The small remaining gap between full Muon and VO+FFN may arise because VO+FFN uses the same learning rate as full Muon without further tuning. This gap could likely be reduced by adjusting the learning rate specifically for VO+FFN."

If the residual gap is indeed due to suboptimal learning rates, then the conclusion that full Muon provides a small additional benefit beyond VO+FFN may be an artifact of the tuning protocol. A practitioner deploying VO+FFN-only Muon would tune the learning rate, potentially closing the gap entirely. The cost of this additional tuning (which is a real practical consideration) is not discussed.

**Memory cost:** Muon must store the momentum buffer $B_t$ for each matrix parameter, which is the same size as the parameter itself. This is comparable to Adam's storage of first and second moment estimates ($m_t$ and $v_t$), so the memory footprint is similar. However, the Newton-Schulz iteration may require temporary storage for intermediate matrices, potentially increasing peak memory usage during the optimizer step. This is not measured or discussed.

**No FLOPs-matched comparison:** The paper compares Muon and Adam at the same number of training steps, not at the same total computational budget. If Muon's per-step cost is, say, 20% higher than Adam's, then a fair comparison would give Adam 20% more steps. The paper provides no basis for estimating what this correction factor should be, so a practitioner cannot determine the true efficiency advantage of switching to Muon.

These cost considerations are not limitations of the mechanistic findings per se—the paper's contribution is understanding *why* Muon works, not benchmarking its efficiency—but they are critical for anyone deciding whether to deploy Muon based on this understanding. A practitioner convinced by the associative memory explanation still needs to know: is the optimizer switch worth the engineering effort and potential per-step overhead? The paper provides none of the data needed to answer this.

## 7. Implications and Future Directions
- How this changes the landscape
  - It reframes Muon’s edge as architectural: matrix‑norm, SVD‑based updates match the outer‑product structure of transformer memories. This motivates component‑aware optimizer design rather than one‑size‑fits‑all.
  - For practitioners: a strong default is hybrid optimization—use Muon for VO and FFN, Adam for QK and embeddings—capturing most gains at likely lower overhead (Section 4.1, Figure 1c–d; Table 1).
- Follow‑up research it enables
  - Optimizer design
    - Extend Muon‑like ideas to higher‑order tensor memories (Section 6) and to other memory‑bearing components (e.g., MoE experts, KV caches).
    - Adaptive variants (e.g., Adamuon, PolarGrad) targeted to associative memory spectra; schedules that switch optimizers per component or per training phase.
  - Theory
    - Beyond one‑layer to multi‑layer nonlinear settings; incorporate EMA effects for Adam; relax orthogonality assumptions and analyze richer heavy‑tail distributions.
    - Study generalization: how isotropic spectra in memory weights relate to factual robustness and editing stability.
  - Evaluation
    - Real‑world heavy‑tail benchmarks (Wikipedia entities, long‑tail slot filling), beyond synthetic QA; measure recall fairness across entities and attributes.
- Practical applications
  - Pretraining/fine‑tuning regimes where tail coverage matters: enterprise knowledge bases, legal/medical recall, safety‑critical FAQ systems.
  - Knowledge editing and maintenance: isotropic memory spectra may yield more predictable edits in `W_out` and `W_O` (Section 3, related work).
  - Fairness audits: spectral isotropy metrics (Section 4.2) as training diagnostics for balanced learning across rare classes.

Overall, the paper provides a cohesive empirical‑theoretical narrative: Muon’s spectral normalization aligns with the outer‑product structure of transformer memories, producing isotropic updates that improve tail learning. The ablations (Figure 1), spectra (Figure 2), heavy‑tail QA (Figure 3; Appendix C.4–C.5), and theory (Theorems 5.3–5.4) collectively make the case and offer a concrete, actionable recipe for deploying Muon where it counts most.
