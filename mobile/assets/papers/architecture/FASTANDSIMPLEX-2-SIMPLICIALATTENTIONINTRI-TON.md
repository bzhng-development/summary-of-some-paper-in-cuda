# FAST AND SIMPLEX: 2-SIMPLICIAL ATTENTION IN TRI-TON

**ArXiv:** [2507.02754](https://arxiv.org/abs/2507.02754)

## 🎯 Pitch

This paper introduces 2-simplicial attention, a novel higher-order generalization of standard Transformer attention that allows each token to attend to pairs of other tokens via a trilinear function. By providing an efficient Triton-based implementation and rigorous scaling law analysis, the authors demonstrate that interleaving 2-simplicial layers enables models to achieve superior token efficiency and improved performance on reasoning, math, and coding tasks—crucially, with a steeper scaling law exponent under a fixed token budget. This breakthrough addresses the pressing bottleneck of limited high-quality training data in large language models, paving the way for more resource-efficient and capable AI systems.

---

## 1. Executive Summary

This paper studies how the **2-simplicial Transformer**—an architecture that generalizes standard dot-product attention from bilinear to trilinear forms via an efficient Triton kernel—scales under token constraints compared to the standard Transformer on math, coding, and reasoning benchmarks (GSM8k, MMLU, MMLU-pro, MBPP). Training MoE models from 1B to 3.5B active parameters, the authors demonstrate that the 2-simplicial Transformer achieves a **steeper scaling exponent α** relative to model parameters—18.5% higher on GSM8k and 20.2% higher on MMLU-pro—meaning loss decreases faster as parameters increase, establishing that the architecture yields more favorable scaling under token constraints only for models above roughly 2B active parameters (with smaller models showing no gain or slight degradation).

## 2. Context and Motivation

### The Core Problem: Token Scarcity in the Age of Compute-Bound Scaling

The fundamental problem this paper addresses is deceptively simple: **what happens to LLM scaling when we run out of high-quality training data?** The dominant paradigm for improving language model performance, established by the Chinchilla scaling laws (Hoffmann et al., 2022), prescribes scaling model parameters and training tokens *in tandem* — double the parameters, double the data. This framework has been enormously successful at guiding compute allocation, but it contains a hidden assumption that is increasingly fragile: that data is effectively infinite.

As the authors state in Section 1:

> "As modern large language models increasingly rely on massive internet-scale datasets, the assumption that they are compute-bound is becoming less valid."

This shift from compute-bound to data-bound training represents a structural change in the economics of LLM development. When compute is the bottleneck, you can always throw more GPUs at the problem. When data quality and quantity become the bottleneck, you face a fundamentally different constraint: the available pool of high-quality text on the internet is finite, and synthetic data generation is expensive and imperfect. The Chinchilla prescription — scale tokens proportionally to parameters — becomes physically impossible to follow past a certain threshold.

This gap matters because the entire industry trajectory (larger models, longer training runs) is on a collision course with token scarcity. Without architectural innovations that extract more learning from each token, the field faces a **wall in scaling law improvements** where neither more compute nor more parameters can help, because the data simply isn't there.

### The Scaling Exponent Problem: Most Architectural Changes Don't Help

The paper grounds its motivation in a sobering empirical finding from the scaling laws literature: **most architectural and optimization innovations merely shift the loss curve without changing its fundamental slope**. The authors cite multiple lines of evidence for this (Section 1 and Section 3):

- Kaplan et al. (2020) and Shen et al. (2024) showed that most architectural modifications do not change the **exponent** α in the power law $L(N) = E + A/N^\alpha$ — they only affect the multiplicative constant $A$ or the irreducible loss $E$. 
- Hestness et al. (2017) demonstrated a similar result for optimizer improvements.
- Everett (2025) provides a synthesis of this observation, noting that "most architectural and optimizer improvements merely shift the error but do not meaningfully change the exponent of the power law."

This distinction between shifting the curve (changing $A$ or $E$) and steepening it (changing $\alpha$) is critical. A shift helps at a specific model size; a steeper exponent compounds as models scale. If you have a fixed token budget, a model family with a higher α will asymptotically approach a lower loss than one with a lower α, even if the latter starts from a better position. The paper's central claim is that 2-simplicial attention belongs to the rare class of modifications that actually changes α.

The authors frame this against the backdrop of the Chinchilla findings. Recall Equation 1 from the paper:

$$L(N, D) = E + \frac{A}{N^\alpha} + \frac{B}{D^\beta}$$

The Chinchilla analysis showed that for standard Transformers, compute-optimal training requires $N \propto C^{0.49}$ and $D \propto C^{0.5}$ — nearly equal scaling of parameters and tokens. But this equilibrium depends on the exponents α and β. If an architecture change increases α (makes the loss decay faster with parameters), the compute-optimal balance shifts: **you can increase parameters at a faster rate than tokens and still be optimal**. This is precisely what the paper's results in Table 3 suggest — the 2-simplicial Transformer has a higher α on reasoning benchmarks, implying it can make better use of additional parameters under a fixed token budget.

### Prior Approaches: From Linear Attention to Looped Transformers

The paper positions itself within a landscape of prior attempts to improve attention mechanisms, each with documented limitations:

**Linear and efficient attention.** A major research direction since Vaswani et al. (2017) has been reducing the $O(n^2)$ complexity of attention. Local attention (Parmar et al., 2018; Zaheer et al., 2020; Roy et al., 2021), linearized attention via kernel tricks (Katharopoulos et al., 2020), and state-space models like Mamba (Gu & Dao, 2023) all achieve sub-quadratic complexity. However, the paper notes that these methods have "received less widespread adoption due to their worse quality compared to Transformers in practice" (Section 2). Allen (2025) is cited as observing that Mamba's practical success owes more to efficient conv1d operators than to its linear attention mechanism per se. The fundamental tension is that making attention cheaper tends to reduce its expressive power — you trade quality for speed.

**Higher-order attention.** The paper builds directly on the 2-simplicial Transformer introduced by Clift et al. (2019), which generalizes bilinear dot-product attention $A_{ij} = \langle q_i, k_j \rangle$ to trilinear forms $A_{ijk} = \langle q_i, k_j, k'_k \rangle$. The original motivation was representational: a 2-simplicial layer can express logical relationships (specifically, ternary relations among tokens) that would require multiple standard attention layers to capture. Sanford et al. (2023) formalized this by defining a class of problems called **Match3** and proving that dot-product attention requires exponentially many layers in sequence length to solve instances of this class, while a 2-simplicial Transformer can solve them with a constant number of layers. Kozachinskiy et al. (2025) extended these representational arguments using VC dimension bounds, proposing a scalable approximation ("Strassen attention") and proving lower bounds on the gap between trilinear and bilinear attention for compositional reasoning tasks.

Related work on Edge Transformers (Bergen et al., 2021) and triangular attention in AlphaFold (Jumper et al., 2021) also uses higher-order interactions, but in domain-specific ways (2D protein geometry). Wang et al. (2021) explored higher-order interactions for recommender systems.

**Looped / Universal Transformers.** Dehghani et al. (2018) proposed Universal Transformers that loop transformer layers, with more recent work by Yang et al. (2023) and Saunshi et al. (2025) revisiting the idea. The conceptual parallel is clear: both higher-order attention and layer looping increase **expressive power per parameter**, making the model compute more complex functions without increasing parameter count. The paper identifies this as the shared goal: "compute a more expressive function per parameter" (Section 2). However, the authors highlight a critical practical limitation of looped Transformers:

> "A key challenge in scaling looped Transformers to larger models is their trainability. Specifically, looping k times increases the model depth by a factor of k, which can significantly exacerbate the difficulties associated with training deeper models. As a result, it remains unclear how well large looped Transformers can be trained."

This is precisely where the 2-simplicial approach has an advantage: it gains expressive power through trilinear interactions within a single layer rather than through depth, avoiding the vanishing gradient and optimization instability issues that plague deep recurrent architectures.

### Where Prior Higher-Order Attention Falls Short

Despite the theoretical appeal of 2-simplicial attention established by Clift et al. (2019) and Sanford et al. (2023), the approach has seen essentially **zero adoption in practical large-scale language modeling**. The paper identifies the implicit reason: the $O(n^3)$ complexity in sequence length makes it prohibitively expensive for the context lengths used in modern LLM training. No prior work had demonstrated:

1. **A practical implementation** that makes 2-simplicial attention competitive in throughput with standard attention at realistic scales.
2. **Scaling behavior** comparing 2-simplicial and standard Transformers across multiple model sizes — the prior theoretical work focused on expressivity for small-scale synthetic tasks, not on scaling laws for real benchmarks.
3. **A rotation-invariant formulation** compatible with modern position encoding schemes like RoPE (Su et al., 2024). The original trilinear form $\langle q_i, k_j, k'_k \rangle$ is not invariant to simultaneous rotation of all three vectors, which breaks RoPE's mechanism of encoding relative position through dot products of rotated vectors. The paper notes this explicitly in Section 5: "the trilinear form defined in Equation 5 is not invariant to rotation."

This last point is subtle but important. RoPE works because the dot product satisfies $\langle R q_i, R k_j \rangle = \langle q_i, k_j \rangle$ for any orthogonal transformation $R$. The RoPE mechanism applies position-dependent rotations $R_i, R_j$ to queries and keys such that $\langle R_i q, R_j k \rangle$ depends only on $i - j$. But for trilinear forms, $\langle R q_i, R k_j, R k'_k \rangle \neq \langle q_i, k_j, k'_k \rangle$ in general. Any deployment of 2-simplicial attention in a modern LLM must either forgo RoPE or develop a new rotation-invariant trilinear form. The paper's determinant-based construction (Section 5, Equation 8-9) resolves this.

### The Only Known Way to Change the Scaling Exponent

The paper makes a striking claim in Section 1 that frames its entire contribution:

> "The only positive result has been on data due to the works of Sorscher et al. (2022); Bahri et al. (2024); Brandfonbrener et al. (2024) who show that changing the data distribution can affect the exponent in the scaling laws."

In other words, prior to this work, **changing the data** was the only known lever for fundamentally altering scaling behavior — changing the architecture or optimizer did not. This is a dramatic statement about the maturity of the field: after years of architectural innovation (mixture of experts, various attention variants, novel normalization schemes, alternative activation functions), none of these changes had been demonstrated to change α in the relevant loss regimes. If true, this means that when data becomes scarce, we currently have no architectural escape hatch — the scaling laws inevitably flatten out.

The paper positions the 2-simplicial Transformer as a counterexample to this pessimistic picture. If an architecture change can increase α (as Table 3 suggests), it provides a second lever for improving scaling efficiency beyond data curation. This is the paper's most ambitious claim and the primary motivation for revisiting a theoretically interesting but practically unproven attention variant.

### How This Paper Positions Itself

The paper positions itself at the intersection of three threads:

1. **Scaling laws theory** (Kaplan, Chinchilla): It adopts the parametric form $L(N) = E + A/N^\alpha$ as the lens through which to evaluate architectural improvements, explicitly targeting α rather than the offset.

2. **Representational theory of higher-order attention** (Clift, Sanford, Kozachinskiy): It inherits the theoretical motivation that trilinear attention can express more complex functions per layer, but shifts the evaluation from synthetic logical tasks to real-world benchmarks (GSM8k, MMLU, MMLU-pro, MBPP).

3. **Systems-level kernel optimization** (FlashAttention, Triton): It addresses the $O(n^3)$ practical barrier through sliding-window parameterization, grouped query attention, and custom Triton kernels that achieve 520 TFLOPS — competitive with the fastest FlashAttention v3 implementations.

The paper is explicitly **not** claiming that 2-simplicial attention is uniformly better. The results in Table 2 show that at 1B active parameters, the gains are marginal (0.79% on GSM8k, 0.19% on MMLU), and at 2B active parameters, the 2-simplicial model actually *underperforms* the baseline Transformer (by 1.51% on GSM8k, 1.19% on MMLU). It is only at 3.5B active parameters that the steeper scaling exponent creates a meaningful advantage. This is a nuanced position: the architecture changes *how loss scales with parameters*, not the absolute loss at all scales. The practical implication is that 2-simplicial attention is most valuable in **large-model, data-constrained regimes**, which is precisely where the field is heading.

## 3. Technical Approach

### 3.1 Reader Orientation

This is primarily a **systems-and-architecture paper** that demonstrates how to make 2-simplicial attention—a triplet-based generalization of standard dot-product attention originally proposed by Clift et al. (2019)—practical for large-scale language model training through sliding-window parameterization, grouped query attention, and custom Triton kernels, and then uses this implementation to empirically establish that trilinear attention steepens the scaling exponent α on reasoning benchmarks. The system being built is an efficient training pipeline for MoE language models that interleave standard attention layers with 2-simplicial attention layers, tested at scales from 1B to 3.5B active parameters on math, coding, and reasoning tasks.

### 3.2 Big-Picture Architecture (Diagram in Words)

The system has five major components:

1. **Sliding-Window 2-Simplicial Attention Layer** — replaces every fourth standard attention layer with a trilinear attention operation where each query vector attends to a localized two-dimensional grid of key vectors (dimensions `$w_1 \times w_2$`), reducing complexity from `$O(n^3)$` to `$O(n \cdot w_1 \cdot w_2)$`.

2. **Determinant-Based Trilinear Form (RoPE-Compatible)** — reformulates the trilinear attention score using a sum-of-determinants operation instead of the naive elementwise triple product, achieving rotation invariance so that position-dependent rotary embeddings (RoPE) can be applied without breaking the attention mechanism's position encoding properties.

3. **Grouped Query Attention (GQA) with Ratio 64** — shares key-value heads across 64 query heads, enabling efficient tiling of the query dimension in the custom kernel while reducing memory bandwidth pressure.

4. **Custom Triton Forward/Backward Kernels** — implements the sliding-window trilinear attention using online softmax (FlashAttention-style tiling) with a two-stage backward pass that avoids expensive atomic operations on the `$K'$` and `$V'$` gradients by splitting computation across even and odd query tiles, achieving 520 TFLOPS.

5. **Interleaved MoE Training Pipeline** — trains sparse mixture-of-experts models where 2-simplicial layers are distributed across pipeline stages to balance computational load with global attention layers, using AdamW with cosine decay and a peak learning rate of `$4 \times 10^{-3}$`.

Information flows as follows: token sequence enters → standard Transformer layers process most positions → every fourth layer, the 2-simplicial layer transforms queries, keys, and an additional key-value pair `$(K', V')$` → sliding windows restrict each query to attend to `$w_1$` preceding `$K$` positions and `$w_2$` preceding `$K'$` positions → the determinant-based trilinear score is computed and passed through online softmax → the output is a Hadamard product combination of the two value streams → training loss (negative log-likelihood) is measured on GSM8k, MMLU, MMLU-pro, and MBPP.

### 3.3 Roadmap for the Deep Dive

- **First**, the mathematical definition of 2-simplicial attention (Equations 5-7) — the trilinear extension of standard attention — because all subsequent design choices (sliding windows, determinant forms, kernel optimizations) are motivated by making this computational primitive practical.
- **Second**, the determinant-based trilinear form (Equations 8-9) and why it's necessary — this is the key innovation that makes RoPE position encoding compatible with trilinear attention, and it changes the kernel implementation from one einsum to two.
- **Third**, the sliding-window parameterization and complexity analysis — how `$w_1$` and `$w_2$` are chosen, the latency measurements that informed the (512, 32) choice, and why GQA ratio 64 is essential for efficient tiling.
- **Fourth**, the forward-pass kernel optimization — the tiling strategy that reduces the trilinear einsum to elementwise multiplication on CUDA cores plus matmul on Tensor cores, and how online softmax from FlashAttention is adapted to the 3D attention tensor.
- **Fifth**, the backward-pass kernel optimization — the mathematical gradients (Equations 10-16), why naive fusion causes atomic overhead, the two-kernel decomposition, and the two-stage scheme for small `$w_2$` (Algorithm 2).
- **Sixth**, the training configuration — model sizes, MoE architecture, interleaving schedule, optimizer hyperparameters, and evaluation methodology — because these determine the scaling law measurements in Table 3.

### 3.4 Detailed, Sentence-Based Technical Breakdown

This is an **architectural and systems paper** whose core idea is that 2-simplicial attention—when implemented efficiently enough to train at practical scales—steepens the scaling law exponent α compared to standard dot-product attention, enabling models to extract more capability from each training token under data-constrained regimes.

---

#### Mathematical Definition of 2-Simplicial Attention

The 2-simplicial Transformer generalizes standard dot-product attention from **bilinear** interactions (between a query and a single key) to **trilinear** interactions (between a query and two distinct keys). To understand what changes, first recall the standard mechanism.

**Standard dot-product attention.** Given an input sequence `$X \in \mathbb{R}^{n \times d}$`, three learned projection matrices `$W_Q, W_K, W_V \in \mathbb{R}^{d \times d}$` produce:

$$Q = XW_Q, \quad K = XW_K, \quad V = XW_V$$

all in `$\mathbb{R}^{n \times d}$`. The attention logits are computed as:

$$A_{ij} = \frac{\langle q_i, k_j \rangle}{\sqrt{d}} = \frac{1}{\sqrt{d}} \sum_{l=1}^d q_{il} k_{jl}$$

where `$q_i$` is the `$i$`-th row of `$Q$` (the query vector at position `$i$`) and `$k_j$` is the `$j$`-th row of `$K$` (the key vector at position `$j$`). Each entry `$A_{ij}$` is a scalar representing the compatibility between query `$i$` and key `$j$`. The logits are then normalized via row-wise softmax:

$$S_{ij} = \frac{\exp(A_{ij})}{\sum_{j'=1}^n \exp(A_{ij'})}$$

producing attention weights `$S_{ij}$` that sum to 1 across `$j$` for each query `$i$`. The output is a weighted sum of value vectors:

$$\tilde{v}_i = \sum_{j=1}^n S_{ij} \cdot v_j$$

where `$v_j$` is the `$j$`-th row of `$V$`.

**What it computes (operationally):** for each position `$i$`, compare its query against all keys using dot-product similarity, convert similarities to a probability distribution via softmax, and aggregate the corresponding value vectors according to those probabilities. The result `$\tilde{v}_i$` is a context-dependent representation of position `$i$` that incorporates information from all positions `$j$` weighted by relevance.

**Why this form:** the dot product `$\langle q_i, k_j \rangle$` is a bilinear function — linear in `$q_i$` when `$k_j$` is fixed and linear in `$k_j$` when `$q_i$` is fixed. Bilinearity provides computational efficiency (can be implemented as a single matrix multiplication `$QK^\top$`) while capturing pairwise token interactions. The `$1/\sqrt{d}$` scaling prevents the dot products from growing with dimension, which would push the softmax into saturated regions.

**Extension to trilinear attention.** The 2-simplicial Transformer introduces two additional projection matrices `$W_{K'} \in \mathbb{R}^{d \times d}$` and `$W_{V'} \in \mathbb{R}^{d \times d}$`, producing:

$$K' = XW_{K'}, \quad V' = XW_{V'}$$

The attention logits now involve three vectors per entry — one query, two keys — forming a third-order tensor:

$$A^{(2s)}_{ijk} = \frac{\langle q_i, k_j, k'_k \rangle}{\sqrt{d}} = \frac{1}{\sqrt{d}} \sum_{l=1}^d Q_{il} \, K_{jl} \, K'_{kl}$$

**What it computes (operationally):** for each query position `$i$`, evaluate a trilinear compatibility score `$A^{(2s)}_{ijk}$` against every *pair* of positions `$(j, k)$`, one from the `$K$` sequence and one from the `$K'$` sequence. The trilinear inner product `$\langle q_i, k_j, k'_k \rangle = \sum_l q_{il} k_{jl} k'_{kl}$` multiplies corresponding components from all three vectors and sums them. This captures ternary relationships among three tokens simultaneously — a single 2-simplicial attention head can express functions of triples that would require multiple standard attention heads or layers to approximate.

The softmax normalization is now over a 2D grid for each query:

$$S^{(2s)}_{ijk} = \frac{\exp(A^{(2s)}_{ijk})}{\sum_{j',k'} \exp(A^{(2s)}_{ij'k'})}$$

The output uses the **Hadamard product** of the two value vectors rather than a single value:

$$\tilde{v}^{(2s)}(i) = \sum_{j,k=1}^n S^{(2s)}_{ijk} \cdot (v_j \circ v'_k)$$

where `$v_j \circ v'_k$` denotes elementwise multiplication — for each dimension `$d$`, the combined value is `$(v_j)_d \cdot (v'_k)_d$`.

**Why this form:** using the Hadamard product `$v_j \circ v'_k$` rather than, say, concatenation or addition ensures that the combined value representation interacts multiplicatively across the two value streams, matching the multiplicative structure of the trilinear attention score. If the score `$A^{(2s)}_{ijk}$` indicates a strong ternary relationship among tokens `$i, j, k$`, the output at position `$i$` receives a contribution proportional to the elementwise interaction of the two value vectors `$v_j$` and `$v'_k$`, allowing the model to represent functions where the relevance of token `$j$` depends jointly on token `$k$` (and vice versa).

The pseudo-code in Algorithm 1 clarifies the implementation using Einstein summation:

```
logits ← einsum("btnh, bsnh, brnh → bntsr", Q, K, K')
attention ← softmax(logits + causal-mask, axis = [-1, -2])
output ← einsum("bntsr, bsnh, brnh → btnh", attention, V, V')
```

The first einsum computes `$A^{(2s)}_{ijk}$` by contracting over the head dimension `$h$` — multiplying corresponding elements of `$Q, K, K'$` and summing. The softmax is applied jointly over the last two axes (`$j$` and `$k$`), producing a 4D tensor where each `$(i, j, k)$` entry is a normalized attention weight. The second einsum contracts the attention weights against both value tensors, multiplying `$V$` and `$V'$` elementwise (the Hadamard product `$v_j \circ v'_k$` is implicitly computed during the einsum) and summing over the `$j, k$` dimensions.

**The fundamental practical challenge.** A naive implementation of Equation 5 requires `$O(n^3)$` computation: for each of the `$n$` query positions, we must evaluate the trilinear score against all `$n \times n$` pairs of key positions. At modern context lengths (`$n \geq 8192$`), this is infeasible — `$n^3 = 5.5 \times 10^{11}$` operations per attention head per sequence, compared to `$2n^2 = 1.3 \times 10^8$` for standard attention. The paper addresses this through sliding-window sparsification (Section 6), but the core mathematical definition above is what the sliding windows *approximate*.

---

#### Determinant-Based Trilinear Forms for RoPE Compatibility

**The rotation invariance problem.** Standard RoPE (Rotary Position Embedding) works by applying position-dependent rotation matrices `$R_i \in \mathbb{R}^{d \times d}$` to queries and keys such that:

$$\langle R_i q, R_j k \rangle = f(i - j)$$

— the dot product depends only on the relative position `$i - j$`, not on absolute positions. This works because the dot product is invariant to simultaneous rotation: `$\langle Rq, Rk \rangle = \langle q, k \rangle$` for any orthogonal matrix `$R$`. When the same rotation is applied to a query `$q_i$` and key `$k_i$` at the same position, the self-attention score remains unchanged, and when different rotations `$R_i, R_j$` are applied, the result encodes the relative angle.

The naive trilinear form `$\langle q_i, k_j, k'_k \rangle = \sum_l q_{il} k_{jl} k'_{kl}$` **breaks** this property. Applying the same rotation to all three vectors does not preserve the trilinear inner product:

$$\langle Rq, Rk, Rk' \rangle \neq \langle q, k, k' \rangle$$

This means that if we apply RoPE-style position-dependent rotations to `$q_i, k_j, k'_k$`, the trilinear score would depend on absolute positions in a complex way, not just on the relative position relationships. RoPE's mechanism of encoding positional information through rotation would fail, and the model could not learn meaningful position-dependent patterns in the trilinear attention.

**The determinant-based solution.** The authors observe that the **determinant** of a matrix whose columns (or rows) are the vectors of interest is invariant to simultaneous rotation of all vectors. More precisely, for any orthogonal matrix `$R$`:

$$\det([Rq, Rk, Rk']) = \det(R \cdot [q, k, k']) = \det(R) \cdot \det([q, k, k']) = \det([q, k, k'])$$

since `$\det(R) = 1$` for rotation matrices (they are special orthogonal). The determinant is a **trilinear alternating form** — it is linear in each argument and changes sign under swapping of arguments, but for fixed ordered inputs, it provides a rotation-invariant measure of the volume spanned by the three vectors.

**The 2D and 3D determinant forms.** The paper begins with the 2D case as a building block. For vectors `$a = (a_1, a_2)$` and `$b = (b_1, b_2)$` in `$\mathbb{R}^2$`, the bilinear form:

$$\hat{f}_2(a, b) = \det\begin{pmatrix} a_1 & a_2 \\ b_1 & b_2 \end{pmatrix} = a_1 b_2 - a_2 b_1$$

is rotation-invariant (the area of the parallelogram spanned by `$a$` and `$b$`). For the 3D case, the trilinear form is the determinant of the `$3 \times 3$` matrix formed by stacking the three vectors:

$$\hat{f}_3(a, b, c) = \det\begin{pmatrix} a_1 & a_2 & a_3 \\ b_1 & b_2 & b_3 \\ c_1 & c_2 & c_3 \end{pmatrix}$$

Expanding via the Sarrus rule gives:

$$\hat{f}_3(a, b, c) = a_1 b_2 c_3 + a_2 b_3 c_1 + a_3 b_1 c_2 - a_1 b_3 c_2 - a_2 b_1 c_3 - a_3 b_2 c_1$$

The authors rewrite this as a difference of two trilinear dot products:

$$\hat{f}_3(a, b, c) = \langle (a_1, a_2, a_3), (b_2, b_3, b_1), (c_3, c_1, c_2) \rangle - \langle (a_1, a_2, a_3), (b_3, b_1, b_2), (c_2, c_3, c_1) \rangle$$

**What this rewrite achieves (operationally):** the determinant, which involves six terms with alternating signs, is expressed as the difference between two standard trilinear inner products `$\langle \cdot, \cdot, \cdot \rangle$` where the key vectors have been **permuted** (shuffled) in different patterns. The first term uses the cyclic shift `$(b_2, b_3, b_1)$` for `$b$` and `$(c_3, c_1, c_2)$` for `$c$`; the second term uses the shift `$(b_3, b_1, b_2)$` for `$b$` and `$(c_2, c_3, c_1)$` for `$c$`. Both are standard trilinear inner products (elementwise multiply and sum), so they can be computed using the same einsum primitive, just with permuted inputs.

**Why this form:** by decomposing the determinant into two standard trilinear dot products, the implementation can reuse the same computational pattern as the naive trilinear attention (Algorithm 1), just calling einsum twice with different index permutations on `$K$` and `$K'$`. This adds a factor of 2 in computation but maintains the kernel's structural simplicity. The rotation invariance guarantee comes from the determinant property — no additional verification or constraints are needed.

**Chunked application for high-dimensional heads.** Real attention heads have dimension `$d \gg 3$`. The paper partitions each vector into **chunks of size 3** and applies the determinant operation independently to each chunk, summing the results. For vector `$q$`, let `$q^{(l)} = q[3(l-1) : 3l]$` be its `$l$`-th chunk of size 3 (with zero-based indexing, so the first chunk is indices 0,1,2, the second is 3,4,5, etc.). The logit for the sum-of-determinants trilinear form is:

$$A^{(\text{det})}_{i j_1 j_2} = \sum_{l=1}^{p} \det\left(\left[ q^{(l)}_i, \, k^{(l)}_{j_1}, \, k'^{(l)}_{j_2} \right]\right)$$

where `$p = d/3$` is the number of 3D chunks.

**What it computes (operationally):** each attention head dimension is divided into triples. For each triple, the determinant of the `$3 \times 3$` matrix formed by the query chunk, key chunk, and second-key chunk is computed. These determinants are summed across all `$p$` chunks to produce a single scalar logit. The operation captures **per-triple volume measures** — geometrically, each determinant measures the signed volume of the parallelotope spanned by the three corresponding 3D sub-vectors. The sum aggregates evidence across all triples.

**Why chunking into triples:** the determinant is defined only for square matrices, requiring all vectors to have the same dimension and that dimension to be exactly the number of vectors (here, 3). By chunking into groups of 3, the operation is well-defined for arbitrary head dimensions `$d$` that are multiples of 3. The summation across chunks preserves rotation invariance chunk-by-chunk: if the same rotation is applied to all three vectors in the full `$d$`-dimensional space, it decomposes into independent rotations in each 3D chunk (for appropriately structured rotation matrices like those used in RoPE), and each per-chunk determinant is individually invariant.

**Theorem 5.1 and representational power.** The paper proves (proof in Appendix A) that a Transformer with a single head of sum-of-determinants attention (dimension `$d = 7$`) can solve the **Match3** problem: given input tokens `$x_1, \ldots, x_n$` from the set `$[M] = \{0, 1, \ldots, M-1\}$`, for each token `$x_i$`, output 1 if there exist positions `$j_1, j_2$` such that `$x_i + x_{j_1} + x_{j_2} \equiv 0 \pmod{M}$`, and 0 otherwise. This is a ternary matching problem that Sanford et al. (2023) proved requires **exponentially many layers** for a standard dot-product Transformer to solve. The construction uses trigonometric embeddings `$\cos(2\pi x_k / M)$`, `$\sin(2\pi x_k / M)$` in the first six dimensions and a "blank pair" mechanism in the seventh dimension that receives uniform attention when no match exists. The trigonometric identity `$\cos(\theta_1 + \theta_2 + \theta_3) = \det(M_1) + \det(-M_2)$` (Equation 21) maps the modular sum to the sum of two determinants, which the attention mechanism can evaluate via its chunked determinant computation. The seventh dimension provides a fallback: when no matching pair exists for query `$i$`, all attention mass goes to the blank pair (score also `$c$`), which has value 0, producing output close to 0; when one or more matches exist, the output is approximately `$\beta_i / (\beta_i + 1)$` where `$\beta_i$` is the number of matches, and an output MLP thresholds at 0.5 to produce the binary answer.

**Practical implications for the kernel.** Since the determinant decomposition involves **two** trilinear dot products (corresponding to the two determinant terms from the Sarrus rule), the implementation of determinant-based attention requires two einsum operations in the forward pass where the naive trilinear form requires one. The paper notes this explicitly: "Since Equation 8 has 2 dot product terms due to Sarrus rule, it would modify Algorithm 1 to use 2 einsums instead of 1 in line 2." This factor of 2 in computation is the price of rotation invariance. The TriTon kernel implementation sections (Section 7 and Appendix B, C) use the simpler trilinear form of Equation 5 for clarity, noting that the extension to the determinant form is a straightforward doubling of the einsum calls.

---

#### Sliding-Window Parameterization and Complexity Management

**Motivation for sparsification.** The full 2-simplicial attention defined in Equation 5 requires `$O(n^3)$` computation in the sequence length `$n$`. At `$n = 8192$`, this is approximately `$5.5 \times 10^{11}$` operations per head, compared to `$1.3 \times 10^8$` for standard attention — a factor of roughly 4000× more computation. This is infeasible for training runs that process billions of tokens. The paper adopts a **sliding-window** approach: each query vector attends not to all `$n \times n$` pairs of key positions, but only to a localized window of `$w_1$` preceding `$K$` positions and `$w_2$` preceding `$K'$` positions.

**The window specification.** The attention for query position `$i$` considers key positions `$j$` in the range `$(i - w_1, i]$` (a causal window of width `$w_1$` looking backward) and second-key positions `$k$` in the range `$(i - w_2, i]$` (a causal window of width `$w_2$`). The effective attention region for each query is a rectangle of size `$w_1 \times w_2$` in the `$(j, k)$`-plane, as visualized in Figure 2 (left). The computational complexity becomes:

$$O(A^{(2s)}) = 3 \cdot 2 \cdot n \cdot w_1 \cdot w_2 = 6 n w_1 w_2$$

**What this expression means:** the constant 3 comes from the three floating-point operations per element in the trilinear product (two multiplies and one add per dimension, approximated). The factor 2 accounts for two matrix multiplications (the trilinear einsum and the value aggregation). The result is linear in sequence length `$n$` and quadratic in the window dimensions `$w_1, w_2$`. The paper compares this to standard causal dot-product attention:

$$O(A) = \frac{1}{2} \cdot 2 \cdot 2 \cdot n^2 = 2 n^2$$

where the `$1/2$` factor comes from the causal mask (only half the `$n \times n$` matrix is computed), the first 2 is for the two matrix multiplies (`$QK^\top$` and `$PV$`), and the second 2 is FLOPs per multiply-add.

**Selecting the window dimensions (Table 1).** The paper evaluates multiple `$(w_1, w_2)$` configurations, measuring Triton kernel latency for a 32k sequence length. The key constraint is that `$w_1 \times w_2$` should remain roughly constant to keep the total computation comparable to standard attention at the target context length. The specific measurements in Table 1:

| `$w_1 \times w_2$` | `$w_1$` | `$w_2$` | Latency at 32k |
|---|---|---|---|
| 32k | 1024 | 32 | 104.1 ms |
| 32k | 512 | 64 | 110.7 ms |
| 16k | 128 | 128 | 59.2 ms |
| 16k | 256 | 64 | 55.8 ms |
| 16k | 512 | 32 | 55.1 ms |
| 16k | 1024 | 16 | 55.1 ms |
| 8k | 256 | 32 | 28.3 ms |

**The chosen configuration.** The paper selects **`$w_1 = 512, w_2 = 32$`**, giving `$w_1 \times w_2 = 16,384$`. The computational complexity of 2-simplicial attention with this window is `$6n \cdot 16384 \approx 98,304n$` operations. Standard causal attention is `$2n^2$`. Setting these equal gives `$2n^2 = 6n \cdot 16384$` → `$n = 3 \cdot 16384 = 49,152$`. **At a context length of approximately 48k tokens, the 2-simplicial attention with (512, 32) windows has comparable FLOPs to standard causal dot-product attention.** This is the design target.

**Why `$w_1 \gg w_2$`:** the window is highly asymmetric — 512 for the first key dimension, only 32 for the second. This asymmetry likely reflects the structure of language: many ternary relationships may involve one long-range dependency and one short-range dependency, or the model may need more context from one key stream than the other. The paper does not provide full ablation results showing the quality impact of different `$(w_1, w_2)$` ratios, but the latency measurements show that for a fixed `$w_1 \times w_2 = 16k$`, configurations with larger `$w_1$` and smaller `$w_2$` (specifically (512, 32)) achieve the best latency — 55.1 ms vs. 59.2 ms for (128, 128) at 16k context — likely because the memory access patterns and tiling strategies are more favorable when one dimension is substantially larger than the other.

**Impact on kernel design (Grouped Query Attention).** The sliding window creates a sparsity pattern where each query attends to `$w_1 + w_2 - 1$` distinct key pairs (Figure 2, left). For adjacent query positions, the sets of attended key pairs overlap substantially. However, a naive implementation that tiles queries like Flash Attention would inherit **irregular sparsity patterns** that underutilize Tensor Cores — different query tiles would have different attention shapes because the windows slide with position.

The paper addresses this by adopting **Grouped Query Attention (GQA) with a ratio of 64**. In standard multi-head attention, each query head has its own key and value heads. In GQA, multiple query heads share the same key-value heads. With a ratio of 64, sixty-four query heads share a single `$K$` and `$K'$` head (and correspondingly, `$V$` and `$V'$`). This means that the key and second-key tensors have `$n_q / 64$` heads rather than `$n_q$` heads, dramatically reducing the memory footprint and enabling **dense tiling along the query head dimension**: the kernel can process blocks of query heads that all use the same key-value data, achieving high computational intensity even though the individual attention patterns are spatially sparse. The paper states this explicitly: "This approach enabled efficient tiling along query heads, ensuring dense computation and eliminating the need for costly element-wise masking."

**Interleaving schedule.** Rather than replacing all attention layers with 2-simplicial attention (which would be computationally prohibitive and may not be necessary), the paper interleaves standard attention and 2-simplicial attention. Specifically: **every fourth layer is a 2-simplicial attention layer**, with the other three being standard dot-product attention layers. The choice of every 4th layer is motivated by pipeline parallelism considerations, not by ablations showing this is the optimal ratio. The paper explains: "The choice of this particular ordering is to distribute the load in attention computation when using pipeline parallelism, since 2-simplicial attention and global attention are the most compute intensive operations in a single pipeline stage and have comparable FLOPs." In a pipeline-parallel training setup, the model is split across multiple accelerators. If all 2-simplicial layers were consecutive, one pipeline stage would be disproportionately burdened while others sit idle. Spreading them evenly ensures balanced load.

---

#### Forward-Pass Kernel Optimization

**Design philosophy.** The forward kernel is built on the principles of FlashAttention (Dao et al., 2022): tile the computation to fit into SRAM, use online softmax to avoid materializing the full attention matrix in high-bandwidth memory, and recompute intermediate values in the backward pass rather than storing them. For 2-simplicial attention, the attention matrix is `$n \times w_1 \times w_2$` rather than `$n \times n$`, but the tiling logic must handle the 3D structure.

**Reducing the trilinear einsum to matrix multiplication.** The key computational insight is that the trilinear operation `$\sum_l Q_{il} K_{jl} K'_{kl}$` can be decomposed into an elementwise multiplication followed by a matrix multiplication. Specifically, for a fixed query `$i$`:

$$A_{ijk} = \langle q_i, k_j, k'_k \rangle = \sum_l (q_i \odot k_j)_l \cdot k'_{kl} = \langle q_i \odot k_j, k'_k \rangle$$

where `$\odot$` denotes elementwise (Hadamard) product. This means the computation can be structured as:

1. **Elementwise multiply `$Q$` and `$K$`** on CUDA cores: for each query block and each `$k_j$`, compute `$q_i \odot k_j$` (elementwise product producing a `$d$`-dimensional vector).
2. **Matrix multiply against `$K'$`** on Tensor cores: `$(Q \odot K) @ K'^\top$`, where the first operand is the result of step 1.

This decomposition is visualized in Figure 2 (right): the left panel shows the sliding window geometry, and the right panel shows "Tiling to reduce 2-simplicial einsum QKK' to elementwise mul QK' on CUDA core and tiled matmul (QK')@K on tensor core."

**Tiling strategy in the kernel (Listing 1).** The forward kernel (Appendix B) tiles along two dimensions:

- **Query dimension:** tiles of size `BLOCK_SIZE_Q = 64` along the sequence. Each kernel instance processes 64 consecutive query positions.
- **Key-Value 2 dimension:** tiles of size `BLOCK_SIZE_KV = 32` along the `$K'$` / `$V'$` dimension (the second key-value stream).

The outer loop iterates over `$K$` positions (variable `kv1_idx` in the kernel, lines 101-173). For each `$K$` position, the kernel loads `$k^{(1)}$` and `$v^{(1)}$` (a single token's key and value vectors), computes the elementwise product `$q_i \odot k^{(1)}_j$` (line 106: `qk1 = q_tile * k1_tile`), and then enters an inner loop over blocks of `$K'$` positions. Within the inner loop, it computes:

- `qk = (qk1 * softmax_scale) @ k2t_tile.T` (line 141-146): the matrix multiplication on Tensor cores, producing a `[BLOCK_SIZE_Q, BLOCK_SIZE_KV]` score matrix.

- Online softmax update (lines 160-165): the running maximum `$m_i$` and normalizer `$l_i$` are updated incrementally using the stable softmax recurrence `$m_{\text{new}} = \max(m_{\text{old}}, \max_j S_{ij})$`, `$l_{\text{new}} = l_{\text{old}} \cdot e^{m_{\text{old}} - m_{\text{new}}} + \sum_j e^{S_{ij} - m_{\text{new}}}$`, and the output accumulator is rescaled correspondingly.

- Value aggregation (lines 167-173): `v12_tile = v1_tile * v2_tile` (elementwise product of the two value vectors), then `acc += p @ v12_tile` (Tensor core matmul of softmax probabilities against the combined values).

**Causal and window masking (lines 148-158).** The mask enforces both causality and the sliding window bounds:
```python
kv1_local_mask = ((q_offs_s[:, None] - w1) < kv1_idx) & (kv1_idx <= q_offs_s[:, None])
kv2_local_mask = ((q_offs_s[:, None] - w2) < kv2_offs_s[None, :]) & (kv2_offs_s[None, :] <= q_offs_s[:, None])
qk_mask &= kv1_local_mask & kv2_local_mask
qk += tl.where(qk_mask, 0, -1.0e38)
```

The first mask ensures that each query only attends to `$K$` positions in the range `$(i - w_1, i]$`. The second mask ensures each query only attends to `$K'$` positions in the range `$(i - w_2, i]$`. Positions outside these windows have their logits set to negative infinity (`-1.0e38`), which after softmax becomes effectively zero attention weight.

**Memory access optimization.** The kernel stores a per-query running maximum `M` (the log-sum-exp normalizer) to enable the backward pass, following FlashAttention's convention. The final normalization (line 176: `acc = acc / l_i[:, None]`) divides the accumulated weighted values by the total softmax denominator after all windows have been processed.

**Achieved performance.** The paper reports achieving **520 TFLOPS** with this Triton implementation, which "rivels the fastest FAv3 Triton implementations" (Section 7). Figure 3 shows FLOPs and latency comparisons between FAv3 (FlashAttention v3) and 2-simplicial attention, with the 2-simplicial kernel achieving competitive performance at large sequence lengths. The paper notes that "Further optimization could be achieved with a lower-level language like CUTLASS for finer grained tuning and optimizations," suggesting the 520 TFLOPS figure is a lower bound on what specialized hardware-aware implementations could achieve.

---

#### Backward-Pass Kernel Optimization

**Mathematical structure of the gradients (Equations 10-16).** The backward pass computes gradients with respect to all five input tensors `$Q, K, V, K', V'$`. Given the upstream gradient `$dO$` (gradient of the loss with respect to the attention output), the gradients are:

$$\frac{\partial L}{\partial V_{jd}} = \sum_{i,k} \left( A_{ijk} \cdot dO_{id} \cdot V'_{kd} \right)$$

$$\frac{\partial L}{\partial V'_{kd}} = \sum_{i,j} \left( A_{ijk} \cdot dO_{id} \cdot V_{jd} \right)$$

$$\frac{\partial L}{\partial P_{ijk}} = \sum_d \left( dO_{id} \cdot V_{jd} \cdot V'_{kd} \right)$$

where `$A_{ijk}$` denotes the attention weights (post-softmax probabilities), `$P_{ijk}$` denotes the pre-softmax logits, and `$dP$` is the gradient with respect to logits. The gradient through the softmax is the standard:

$$dS = \text{dsoftmax}_{jk}(dP)$$

meaning `$dS_{ijk} = A_{ijk} \cdot (dP_{ijk} - \sum_{j',k'} A_{ij'k'} \cdot dP_{ij'k'})$` — the softmax Jacobian maps the logit gradient `$dP$` to the probability gradient `$dS$` by subtracting the probability-weighted mean gradient. The remaining gradients are:

$$\frac{\partial L}{\partial K_{jd}} = \sum_{i,k} \left( Q_{id} \cdot dS_{ijk} \cdot K'_{kd} \right)$$

$$\frac{\partial L}{\partial K'_{kd}} = \sum_{i,k} \left( Q_{id} \cdot dS_{ijk} \cdot K_{jd} \right)$$

$$\frac{\partial L}{\partial Q_{id}} = \sum_{j,k} \left( dS_{ijk} \cdot K_{jd} \cdot K'_{kd} \right)$$

**What each gradient computes (operationally):**

- `$dV$` (Equation 10): each value element `$V_{jd}$` receives gradient contributions from every `$(i, k)$` pair where token `$j$` was attended to (via `$K$`) and token `$k$` was the second key. The contribution scales with the attention probability `$A_{ijk}$`, the upstream gradient `$dO_{id}$`, and the other value element `$V'_{kd}$`. This is a **double reduction** over query and second-key positions.

- `$dV'$` (Equation 11): symmetric to `$dV$`, but reducing over query and first-key positions. The contribution scales with `$A_{ijk}$`, `$dO_{id}$`, and `$V_{jd}$`.

- `$dP$` (Equation 12): the gradient of loss with respect to pre-softmax logits is the dot product (over the head dimension) of the upstream gradient `$dO$` and the elementwise product of the two value streams. This captures how much each `$(i, j, k)$` attention triplet contributed to the loss through the value aggregation.

- `$dS$` (Equation 13): applies the softmax Jacobian, converting the "how much did each logit contribute" gradient into a "how much should each probability change" gradient, accounting for the fact that increasing one probability requires decreasing others (the mean-subtraction property of softmax gradients).

- `$dK$` (Equation 14): reduces over query and second-key positions, weighting by `$Q$` values and `$K'$` values. A particular key `$K_j$` receives gradient proportional to how much its interaction with each query and each second-key contributed to the output.

- `$dK'$` (Equation 15): symmetric to `$dK$`, reducing over query and first-key positions.

- `$dQ$` (Equation 16): for each query, sums over all attended `$(j, k)$` pairs, weighting by the softmax gradient and the elementwise product of the two key values. This tells the query how to adjust so that future attention distributions produce more useful outputs.

**The atomic operations problem.** A naive fused backward kernel that computes all five gradients in a single pass runs into a fundamental concurrency issue. The gradients `$dQ$`, `$dK$`, `$dK'$`, `$dV$`, and `$dV'$` involve **aggregations across three different dimension orderings**. For instance:

- `$dV$` sums over `$(i, k)$` pairs — many query threads may contribute to the same `$j$` position simultaneously.
- `$dQ$` sums over `$(j, k)$` pairs — many key threads may contribute to the same `$i$` position.
- `$dK$` and `$dK'$` sum over `$(i, k)$` and `$(i, j)$` respectively.

These cross-dimensional reductions require **atomic additions** to accumulate partial gradients from different thread blocks that process different slices of the computation. On modern GPUs, atomic operations on the same memory location from many concurrent threads create serialization bottlenecks, and Triton's "coarser grained pipeline control" (as the paper phrases it) makes it difficult to hide this overhead through instruction-level parallelism. The paper explicitly notes: "We note this may be a limitation of Triton's coarser grained pipeline control making it difficult to hide the overhead from atomics."

**The two-kernel decomposition.** The authors decompose the backward pass into **two separate kernels**:

1. **Kernel 1 (dK, dV):** computes gradients for the first key-value pair `$(K, V)$` and optionally `$dQ$`. This kernel tiles over `$K$` positions in the outer loop and iterates over query tiles and `$K'$` positions in inner loops (Listing 2 in Appendix C).

2. **Kernel 2 (dK', dV', dQ):** computes gradients for the second key-value pair `$(K', V')$` and `$dQ$`. This kernel tiles over `$K'$` positions (using a different tiling strategy optimized for small `$w_2$`) and iterates over query tiles and `$K$` positions in inner loops (Listing 3).

The tradeoff: **recomputing intermediate values (the attention output `$O$` and the softmax gradient `$dS$`) to avoid atomic contention.** Each kernel independently recomputes the forward pass for its tile, generating `$S$` and `$dS$` locally, rather than sharing them via global memory. The paper states: "Although this approach incurs additional overhead from recomputing O and dS, we find it is better than the extra overhead from atomics needed for a single fused kernel."

**The two-stage scheme for small `$w_2$` (Algorithm 2).** When `$w_2$` is small (the chosen configuration has `$w_2 = 32$`), the second kernel can use a more efficient strategy to compute `$dQ$` jointly with `$dK'$` and `$dV'$` **without any atomic operations**. Algorithm 2 describes this:

1. Divide the query sequence into tiles of size `$w_2$` along the sequence dimension. Each tile spans `$w_2$` consecutive query positions.
2. **Stage 1 (even tiles):** Iterate over tiles with even starting indices (`q_start = 0, 2w_2, 4w_2, ...`). For each tile, compute partial `$dQ$`, `$dK'$`, and `$dV'$` contributions from all relevant `$K$` positions (those in the window `(q_start - w_1, q_end)`). Store these partial results.
3. **Stage 2 (odd tiles):** Iterate over tiles with odd starting indices (`q_start = w_2, 3w_2, 5w_2, ...`). For each tile, compute `$dQ$` (which can be stored directly since no other tile writes to the same query positions) and **add** `$dK'$` and `$dV'$` to the previously stored partial results from Stage 1. Specifically, lines 14-16: `dK' += load dK'` and `dV' += load dV'` — load the partial values written by Stage 1, add the Stage 2 contributions, and store the final result.

**Why this works without atomics:** the key observation is that the `$K'$` and `$V'$` positions (the second key-value stream) are the same for a query tile and its overlapping neighbor tiles, because `$w_2$` defines the window width for `$K'$`. By processing even and odd tiles in separate stages, each `$K'$` and `$V'$` position receives contributions from exactly two query tiles (one even, one odd), and these contributions are computed sequentially rather than simultaneously — no concurrent writes to the same memory location occur. The gradient `$dQ$` is computed independently for each query tile, so no atomic operations are needed for it either.

**Limitation of this scheme.** The paper notes that this approach with "Skip writing dk2, dv2 for now" (comment in kernel 1, line 14 of Listing 2) means the first kernel does not compute `$dK'$` or `$dV'$`. Kernel 2 handles these entirely, with Kernel 1 only producing `$dK$`, `$dV$`, and optionally `$dQ$`. The `COMPUTE_DQ` flag (line 69 in Listing 2) controls whether Kernel 1 also computes `$dQ$` — if false, Kernel 2 computes `$dQ$` as well (using the even-odd tile scheme). This design choice distributes the gradient computation across two specialized kernels, each optimized for a different memory access pattern.

**Bias parameters for Key-Value 2.** The kernels include parameters `K2_BIAS` and `V2_BIAS` that are added to `$K'$` and `$V'$` tensors respectively (lines 136-137 in the forward kernel). The paper does not elaborate on these biases, but they likely serve as learned offsets specific to the second key-value stream, analogous to how some attention implementations include per-head biases for relative position or other conditioning. Their presence in both the forward and backward kernel signatures indicates they are part of the trained parameters.

---

#### Training Configuration and Evaluation

**Model architecture.** The paper trains **Mixture of Experts (MoE)** models rather than dense Transformers. MoE models partition the feed-forward network layers into multiple "experts" and route each token to a subset of experts, decoupling total parameters from active (per-token) parameters. The model sizes reported are:

| Active Params | Total Params |
|---|---|
| 1B | 57B |
| 2B | 100B |
| 3.5B | 176B |

This means a model with 1 billion active parameters has 57 billion total parameters, with the routing mechanism selecting subsets per token. The scaling experiments compare 2-simplicial vs. standard Transformer at each of these three sizes.

**Interleaving schedule.** Every fourth layer uses 2-simplicial attention; the remaining layers use standard dot-product attention. The paper provides this specific number but does not report ablation experiments showing the effect of different interleaving ratios (e.g., every 2nd layer, every 8th layer). The choice is motivated by pipeline parallelism load balancing rather than by systematic quality optimization.

**Optimizer configuration.** Training uses AdamW with:
- Peak learning rate: `$4 \times 10^{-3}$`
- Weight decay: 0.0125
- Warmup: 4000 steps
- Learning rate schedule: cosine decay to `$0.01 \times$` the peak learning rate (i.e., final LR = `$4 \times 10^{-5}$`)

These are aggressive hyperparameters consistent with large-scale LLM training. The peak LR of `$4 \times 10^{-3}$` is relatively high, which may be necessary because the 2-simplicial layers introduce higher-order interactions that produce larger gradients, but the paper does not discuss LR sensitivity or tuning methodology.

**Evaluation benchmarks and metrics.** The paper evaluates using **negative log-likelihood** (NLL) on four benchmarks:
- **GSM8k** (Cobbe et al., 2021): grade-school math word problems. Evaluated with 5-shot prompting.
- **MMLU** (Hendrycks et al., 2020): massive multitask language understanding. NLL is measured on "the choice together with the entire answer" — the model scores each multiple-choice option plus its explanation.
- **MMLU-pro** (Wang et al., 2024): a more challenging variant of MMLU with harder questions.
- **MBPP** (Austin et al., 2021): mostly basic Python programming problems, testing coding capability.

The use of NLL rather than accuracy is deliberate: scaling laws are typically formulated in terms of loss, not task accuracy, so fitting power-law coefficients to NLL measurements is methodologically consistent. However, this also means the measured improvements are in **token-level prediction quality** on held-out evaluation sets, not necessarily in downstream task performance after fine-tuning.

**Estimating scaling law coefficients from Table 2.** The paper uses the three model sizes (1B, 2B, 3.5B active parameters) to estimate `$\alpha$` and `$\beta$` in the simplified scaling law:

$$\log L(N) \approx \beta - \alpha \log N$$

derived from `$L(N) = E' + A/N^\alpha$` by taking logs and approximating `$\log(E' + A/N^\alpha) \approx \log E'' + \log A - \alpha \log N$` for large `$N$` (where `$E''$` is an approximation that absorbs the `$\log(1 + \text{small term})$` from the series expansion). Regressing `$-\log L(N)$` against `$\log N$` gives slope `$\alpha$` and intercept `$-\beta$`.

The results in Table 3 show:

| Benchmark | `$\alpha_{\text{Transformer}}$` | `$\alpha_{\text{2-simplicial}}$` | `$\Delta\%$` |
|---|---|---|---|
| GSM8k | 0.1420 | 0.1683 | +18.5% |
| MMLU | 0.1256 | 0.1364 | +8.5% |
| MMLU-pro | 0.0901 | 0.1083 | +20.2% |
| MBPP | 0.1720 | 0.1837 | +6.8% |

**What these numbers mean:** a higher `$\alpha$` means loss decreases faster as active parameters increase. For GSM8k, the 2-simplicial Transformer's loss scales as `$N^{-0.1683}$` compared to `$N^{-0.1420}$` for the standard Transformer — when doubling parameters, the 2-simplicial model's loss drops by a factor of approximately `$2^{-0.1683} \approx 0.891$` compared to `$2^{-0.1420} \approx 0.907$` for the standard model. The percentage improvements (18.5% higher `$\alpha$` on GSM8k, 20.2% on MMLU-pro) are substantial, but come with an important caveat: these are estimated from only **three data points** (models at three sizes), which the paper acknowledges via `$R^2$` and residual measurements in Table 4. The `$R^2$` values exceed 0.99 for most benchmarks, indicating excellent fit to the three points, but the statistical power is fundamentally limited by the number of model sizes evaluated.

**The scale-dependent effect.** The raw losses in Table 2 reveal that 2-simplicial attention does **not** uniformly outperform standard attention. At 1B active parameters, the gains are marginal (0.79% relative reduction in NLL on GSM8k, 0.19% on MMLU). At 2B, 2-simplicial **underperforms** the Transformer (1.51% higher NLL on GSM8k, 1.19% on MMLU). At 3.5B, the advantage re-emerges (2.27% lower NLL on GSM8k, 1.06% on MMLU). This non-monotonic behavior at small scales and the eventual advantage at larger scales is **exactly what a higher scaling exponent predicts**: the 2-simplicial model may have a worse constant factor (higher `$A$` in `$L(N) = E + A/N^\alpha$`) but catches up and surpasses the baseline as `$N$` grows because the steeper exponent `$\alpha$` compounds. The crossover point appears to be between 2B and 3.5B active parameters for these benchmarks.

**Why only three data points:** training MoE models with 57B–176B total parameters is enormously expensive. Each additional model size would require a full training run on hundreds of billions of tokens. The paper's contribution is demonstrating the existence of an exponent change and its approximate magnitude; precise estimation of `$\alpha$` with tight confidence intervals would require more model sizes, which the authors leave to future work.

## 4. Key Insights and Innovations

### Innovation 1: Architectural Change as a Lever on the Scaling Exponent — A Counterexample to Prevailing Wisdom

The paper's most conceptually significant contribution is a counterexample to an increasingly entrenched belief in the scaling laws community: that architectural modifications cannot change the exponent α in the loss scaling law `$L(N) = E + A/N^\alpha$`. The authors cite Kaufman et al. (2020), Shen et al. (2024), Hestness et al. (2017), and notably Everett (2025), who synthesizes the observation that "most architectural and optimizer improvements merely shift the error but do not meaningfully change the exponent of the power law." Prior to this work, the only demonstrated lever for changing α was **data curation** — Sorscher et al. (2022), Bahri et al. (2024), and Brandfonbrener et al. (2024) showed that pruning or redistributing training data could steepen scaling. The implication was stark: when data runs out, we have no architectural escape hatch.

The paper challenges this by demonstrating that 2-simplicial attention increases α by 18.5% on GSM8k and 20.2% on MMLU-pro (Table 3). This is a **conceptual reframing**, not merely a better number. The field's collective null hypothesis — that architecture only affects the multiplicative constant `$A$` — implied a sharp separation between "data work" (which changes how efficiently we learn) and "architecture work" (which changes where we start). The 2-simplicial result dissolves this separation, suggesting that the right architectural priors can, like curated data, extract more learning from each token.

What distinguishes this from the many prior architectural innovations that failed to change α (efficient attention variants, alternative normalizations, activation function tweaks) is that the 2-simplicial Transformer is **not a refinement of the dot-product computation** but a qualitatively different operation — trilinear rather than bilinear, capturing ternary rather than pairwise token interactions. The Sanford et al. (2023) result that Match3 requires exponentially many standard attention layers but constant 2-simplicial layers provides a theoretical rationale for why this particular architectural change affects the exponent: it expands the class of functions computable with a fixed parameter budget, which fundamentally alters how loss scales with model size rather than merely shifting where the curve starts.

The evidence in Table 3 is imperfect — only three data points per benchmark, estimated from the slope of `$\log L$` vs. `$\log N$` across models at 1B, 2B, and 3.5B active parameters — but the consistency across four benchmarks (all show higher α for 2-simplicial, ranging from +6.8% to +20.2%) and the strong `$R^2$` values in Table 4 (>0.99 for most) make this the most systematic demonstration to date that an architectural change can alter the scaling exponent. The non-monotonic raw performance in Table 2 (2-simplicial underperforms at 2B, outperforms at 3.5B) actually **strengthens** the exponent-change interpretation: it is precisely the signature of a model family with a higher α but possibly worse constant factor, crossing over as scale increases.

This is a **fundamental** rather than incremental contribution to the scaling laws literature. It opens a research direction — "scaling-law-aware architecture design" — that previously seemed unpromising based on the negative results from prior architectural explorations. The key question it raises is: what property of trilinear attention causes the exponent improvement? Is it the expanded representational capacity (per Sanford et al.), improved gradient flow due to multiplicative interactions, something about the implicit regularization of the higher-order parameterization, or a combination? The paper does not answer this, but establishing the empirical phenomenon is the necessary first step.

---

### Innovation 2: The Determinant-Based Trilinear Form as a Bridge Between Higher-Order Attention and Modern Position Encoding

This innovation is a **synthesis** that unlocks practical deployment of 2-simplicial attention in modern LLM training pipelines. RoPE (Su et al., 2024) has become the dominant position encoding scheme for large Transformers because it encodes relative position information in a way that is additive in the dot-product domain and decays smoothly with distance. But the naive trilinear form `$\langle q_i, k_j, k'_k \rangle = \sum_l q_{il} k_{jl} k'_{kl}$` is **not rotation-invariant**: applying the same orthogonal transformation `$R$` to all three vectors does not preserve the inner product. This means that if you naively apply RoPE's position-dependent rotations to queries and keys in a 2-simplicial layer, the attention score depends on absolute positions in an uncontrolled way, breaking RoPE's mechanism.

The field had no solution to this incompatibility. Prior work on higher-order attention (Clift et al. 2019; Bergen et al. 2021; Wang et al. 2021) either predated RoPE or operated in settings (reinforcement learning, protein structure, recommendation) where relative position encoding was not needed in the same form. The paper's solution — replacing the naive trilinear dot product with a **sum of per-chunk 3D determinants** — is intellectually neat because it exploits an elementary linear algebra fact (the determinant is invariant under simultaneous rotation of its column vectors) to solve a practical systems problem (making 2-simplicial attention compatible with RoPE).

What makes this more than an engineering trick is the decomposition into two standard trilinear dot products via the Sarrus rule (Equation 8). By expressing the determinant as `$\langle a, b_{\text{shift1}}, c_{\text{shift1}} \rangle - \langle a, b_{\text{shift2}}, c_{\text{shift2}} \rangle$` where the key vectors are permuted, the implementation reduces to calling the same einsum primitive twice with shuffled indices — no new computational primitives needed. This **reuses the kernel infrastructure** developed for the naive trilinear form, adding only a factor of 2 in computation while guaranteeing rotation invariance. The chunking into triples (grouping the head dimension into 3D sub-blocks) ensures the determinant is well-defined for arbitrary `$d$` that is a multiple of 3, and summing across chunks provides a meaningful aggregate score.

The Theorem 5.1 result — that this determinant form with dimension `$d = 7$` can solve Match3 in a single attention head — serves as both a theoretical validation (showing the rotation-invariant form preserves the representational advantages of the naive trilinear form) and a constructive proof that bridges the theoretical literature on higher-order attention expressivity (Sanford et al., Kozachinskiy et al.) with the practical implementation. The trigonometric embedding construction in Appendix A demonstrates concretely how the chunked determinants can encode modular arithmetic relations, providing intuition for why this form is well-suited to logical and mathematical reasoning.

This is an **incremental** advance in its individual components (the determinant invariance property is classical; the einsum decomposition is straightforward algebra), but the synthesis — recognizing that the RoPE incompatibility is the blocking issue for practical deployment and that determinants provide the fix — is a conceptual move that may generalize. Any future higher-order attention mechanism (4-simplicial, hypergraph attention, etc.) that needs rotation-invariant position encoding could follow the same template: express the `$k$`-linear form as a sum of determinants, decompose via the generalized Sarrus rule into standard `$k$`-linear dot products with permuted arguments, and implement via multiple einsum calls.

---

### Innovation 3: Efficient Kernel Design That Makes `$O(n^3)$` Attention Competitive at Practical Sequence Lengths

Prior to this work, 2-simplicial attention was a theoretical curiosity — studied for its representational properties on small synthetic tasks (Clift et al., Sanford et al., Kozachinskiy et al.) but never deployed at scales relevant to LLM training. The computational barrier was clear: `$O(n^3)$` complexity in sequence length makes naive implementation infeasible at `$n \ge 2048$`. No prior work had demonstrated that this complexity could be managed through a combination of sparsification and kernel optimization to the point of being competitive with standard attention.

The paper's systems contribution is a **stack of design choices** that collectively resolve this:

- **Asymmetric sliding windows** `$(w_1 = 512, w_2 = 32)$` that reduce complexity from `$O(n^3)$` to `$O(n \cdot w_1 \cdot w_2)$`, with the specific dimensions chosen so that at `$n \approx 48\text{k}$`, the total FLOPs of 2-simplicial attention match standard causal attention. This is an explicit **compute-parity design**: the goal is not to make 2-simplicial attention faster than standard attention, but to make it equally expensive so that any quality improvement is a pure gain in the compute-matched comparison.

- **Grouped Query Attention with ratio 64** that enables dense tiling along the query head dimension despite the irregular sparsity of the sliding windows. Without GQA, each query head would process a different set of key-value pairs, creating memory access patterns that underutilize Tensor Cores. With ratio 64, sixty-four query heads share the same key-value data, turning what would be sparse, irregular accesses into dense matrix multiplications on Tensor Cores.

- **Online softmax (FlashAttention-style) adapted to the 3D attention tensor**, with the log-sum-exp running maximum and normalizer maintained per query across the `$(j, k)$` iteration space (lines 160-176 of Listing 1). This is a non-trivial extension because the attention space is now a 2D grid per query rather than a 1D vector.

- **The two-kernel backward pass decomposition** that trades recomputation of intermediate values (`$O$` and `$dS$`) for elimination of atomic operations. The key insight is that atomics bottleneck throughput more than recomputation does at these scales — particularly in Triton, where the "coarser grained pipeline control" makes it difficult to hide atomic latency. The two-stage even-odd tiling scheme for small `$w_2$` (Algorithm 2) is an elegant solution to the specific problem of computing `$dK'$` and `$dV'$` without concurrent writes: by processing even and odd query tiles in separate passes, each `$K'$` position receives contributions from exactly two tiles sequentially rather than simultaneously.

Achieving 520 TFLOPS — "rivaling the fastest FAv3 Triton implementations" — is the validation that these design choices collectively work. Figure 3 shows competitive throughput at large sequence lengths.

This innovation is **incremental** in the sense that each individual technique (sliding windows, GQA, online softmax, kernel decomposition) is known, but the specific combination and the demonstrated achievement of compute parity with standard attention at practical sequence lengths is what transforms 2-simplicial attention from a theoretical construct into a feasible architectural option. The paper is transparent that further optimization (CUTLASS, hardware-specific tuning) could improve throughput further, suggesting the 520 TFLOPS figure is a lower bound.

---

### Innovation 4: Scale-Dependent Regime Discovery — Where Higher-Order Attention Matters and Where It Doesn't

This is a diagnostic contribution rather than a method contribution, but it is essential for the paper's thesis to be useful rather than merely interesting. The raw results in Table 2 reveal a non-obvious pattern: at 1B active parameters, 2-simplicial attention provides negligible gains (<1% relative NLL reduction across benchmarks); at 2B, it actually **underperforms** the standard Transformer (by 1.51% on GSM8k, 1.19% on MMLU); at 3.5B, it outperforms (by 2.27% on GSM8k, 1.06% on MMLU). This non-monotonicity is not noise — it is the signature of a **crossover regime** that the scaling law analysis in Table 3 makes precise.

The insight is that 2-simplicial attention changes the **scaling dynamics** but not necessarily the **absolute performance ceiling** at a given scale. The steeper exponent α means that loss decreases faster as parameters grow, but if the constant factor `$A$` (or the irreducible loss `$E$`) is worse for 2-simplicial models, there will be a crossover point below which the standard Transformer is actually better. The paper's data suggests this crossover lies between 2B and 3.5B active parameters for these benchmarks and this training configuration.

This finding matters because it gives practitioners a decision rule: below roughly 2B active parameters, 2-simplicial attention is not worthwhile; above that threshold, the gains compound and scale. It also explains why previous explorations of higher-order attention on small-scale tasks might have found no benefit or even degradation — they were operating below the crossover point where the steeper exponent has not yet compensated for the worse constant factor.

The regime dependence extends to the **task type** as well. Table 3 shows that the α improvement is largest on the most challenging benchmarks: +20.2% on MMLU-pro and +18.5% on GSM8k (math/reasoning), compared to +8.5% on MMLU and +6.8% on MBPP. The paper notes this explicitly: "the percentage increase in the scaling exponent α is higher for less saturated and more challenging benchmarks." This aligns with the theoretical motivation from Sanford et al. and Kozachinskiy et al. — higher-order attention helps precisely on tasks requiring multi-step logical or mathematical reasoning where ternary token relationships matter. On tasks where pairwise attention is sufficient, the overhead of trilinear computation provides less benefit.

This innovation is **conceptual** — it establishes the boundary conditions for when higher-order attention is beneficial, converting a binary "is this architecture better?" question into a nuanced "at what scale and for what task distribution does this architecture provide returns?" This is the kind of understanding that guides practical adoption.

## 5. Experimental Analysis

### Evaluation Methodology

- **Dataset.** The paper evaluates on four held-out benchmarks that "most strongly test math, reasoning and coding skills in pre-training" (Section 8): **GSM8k** (Cobbe et al., 2021) — grade-school math word problems; **MMLU** (Hendrycks et al., 2020) — massive multitask language understanding across 57 subjects; **MMLU-pro** (Wang et al., 2024) — a more challenging and robust variant of MMLU; and **MBPP** (Austin et al., 2021) — mostly basic Python programming problems. The paper does not state the exact number of evaluation examples used from each benchmark, but these are standard test sets: GSM8k has 1,319 test problems, MMLU has ~14,000 questions across subjects, MMLU-pro has ~12,000 questions filtered from the original MMLU, and MBPP has ~500 test problems (974 total with 500 in the test split following the original paper).

- **Base model(s).** The paper trains **Mixture of Experts (MoE) Transformer models** at three scales: 1B active / 57B total parameters, 2B active / 100B total parameters, and 3.5B active / 176B total parameters. The architecture is a decoder-only Transformer with interleaved attention layers: every fourth layer uses 2-simplicial attention (with sliding windows of `w1 = 512, w2 = 32`), and the remaining three layers use standard dot-product attention. For the baseline, identically-sized MoE Transformers are trained but with standard dot-product attention at every layer. The paper does not specify the exact architecture details (number of layers, hidden dimension, number of heads) for each model size, only the active and total parameter counts. The MoE architecture is chosen presumably to enable training at these total parameter scales while keeping active (per-token) compute manageable, and to match the architectural pattern of modern production LLMs where MoE is common.

- **Metrics.** The primary metric is **negative log-likelihood (NLL)** — the token-level cross-entropy loss on each benchmark's evaluation set. For MMLU and MMLU-pro, NLL is measured on "the choice together with the entire answer" (Section 8), meaning the model scores each complete multiple-choice option string (choice label + answer text) rather than just the choice label. For GSM8k, evaluation uses 5-shot prompting: the model sees 5 example question-answer pairs before computing NLL on each test question. For MBPP, the evaluation protocol (few-shot vs. zero-shot, whether the model generates the entire program or scores completions) is not specified in the paper. Negative log-likelihood is used because scaling laws are typically formulated in terms of loss rather than task accuracy, enabling direct fitting of the power-law coefficients `α` and `β` from Equation 18. Lower NLL is better.

- **Baselines.** The sole baseline is an **identically-sized standard Transformer** (Vaswani et al., 2017) — same MoE architecture, same total parameters, same optimizer and training recipe, but with standard dot-product attention at every layer instead of the interleaved 2-simplicial layers. This is the appropriate architectural ablation: any difference between the two model families can be attributed to the 2-simplicial attention layers, since all other variables (model scale, training data, optimizer, training duration) are held constant. The paper does not compare against other efficient attention variants (e.g., Mamba, linear attention, sparse attention) or against looped/Universal Transformers — the comparison is strictly 2-simplicial vs. standard dot-product attention.

- **Generation budget / compute accounting.** The paper does **not** use inference-time generation or compute scaling at evaluation time. The models are evaluated on their **pretraining loss** on held-out benchmark data — essentially measuring how well the model predicts the correct tokens in the evaluation set after pretraining on a fixed corpus. There is no best-of-N, beam search, majority voting, or any inference-time strategy. The "budget" in the scaling analysis is **training tokens** — both the 2-simplicial and standard Transformers are trained on "the same fixed number of tokens" (Section 8), though the paper does not specify the exact number. The scaling law coefficients `α` (how loss scales with active parameters `N`) and `β` (how loss scales with data `D`) are the quantities being estimated. Since data is held fixed, the third term in Equation 1 (`B/D^β`) is constant across model sizes, and the analysis focuses on the parameter-scaling term `A/N^α` (Equation 18). The FLOPs comparison is implicit: 2-simplicial attention layers have higher per-layer FLOPs than standard attention (by a factor of `6n w1 w2 / 2n^2 ≈ 3w1 w2 / n`), but the interleaving schedule (every 4th layer) and the window dimensions are chosen so that at `n ≈ 48k`, the total attention FLOPs are comparable to a standard Transformer at the same sequence length (Section 6). The paper does not report total training FLOPs for either architecture.

- **Cross-validation / statistical protocol.** There is **no cross-validation** reported. The scaling law coefficients `α` and `β` in Table 3 are estimated from linear regression of `−log L(N)` against `log N` using exactly three data points (one per model size). The `R^2` values and residuals in Table 4 report the goodness-of-fit of this regression, with `R^2` ranging from 0.9962 to 0.9999, indicating the three points are nearly collinear. However, with only three points, there are zero degrees of freedom for testing the linear model's validity — any three points can be fit perfectly by a linear function with 2 parameters (slope and intercept), and the `R^2` merely reflects how close the third point is to the line defined by the first two. The reported residuals are on the order of `10^{-5}` to `10^{-6}` for most benchmarks (Table 4), confirming the near-perfect collinearity of the three points but providing no statistical evidence that the true relationship is power-law rather than, say, sigmoidal or simply noise at small scales. There is also no mention of multiple training runs per model size to estimate variance — the NLL values in Table 2 appear to be from single training runs, and the uncertainty in the estimated `α` values is not quantified. This is an important limitation: the paper's central claim of a "changed exponent" rests on three data points with unknown variance, and the claimed percentage improvements in α (18.5%, 20.2%) cannot be assessed for statistical significance.

### Main Quantitative Results

#### Raw Negative Log-Likelihood Comparison at Three Model Scales

The headline finding is in Table 2: at 1B active parameters, 2-simplicial attention provides **marginal gains** over the standard Transformer on three of four benchmarks. At 2B active parameters, 2-simplicial attention **underperforms** on all benchmarks. At 3.5B active parameters, 2-simplicial attention **outperforms** on all benchmarks. The specific numbers:

**At 1B active parameters:**

| Benchmark | Transformer NLL | 2-simplicial NLL | `Δ%` (relative change) |
|---|---|---|---|
| GSM8k | 0.3277 | 0.3302 | +0.79% (worse for 2-simplicial? Wait — lower NLL is better, so increasing from 0.3277 to 0.3302 is a +0.76% *increase* in loss, meaning the 2-simplicial model is actually *slightly worse*) |
| MMLU | 0.6411 | 0.6423 | +0.19% (slightly worse) |
| MMLU-pro | 0.8718 | 0.8718 | -0.01% (essentially identical) |
| MBPP | 0.2690 | 0.2714 | +0.88% (slightly worse) |

The paper's labeling of `Δ(%)` is **confusingly signed** — the table says "∆(%)" with values like +0.79% on GSM8k, but the text in Section 8 describes this as "we see no gains from using 2-simplicial attention" for models smaller than 2B. The sign convention appears to be: positive Δ means the 2-simplicial model's NLL is **higher** (worse), negative Δ means it's **lower** (better). So at 1B, 2-simplicial is slightly worse on GSM8k, MMLU, and MBPP, and essentially tied on MMLU-pro.

**At 2B active parameters:**

| Benchmark | Transformer NLL | 2-simplicial NLL | `Δ%` |
|---|---|---|---|
| GSM8k | 0.2987 | 0.2942 | -1.51% (2-simplicial *better* — NLL is lower) |
| MMLU | 0.5932 | 0.5862 | -1.19% (better) |
| MMLU-pro | 0.8193 | 0.8135 | -0.71% (better) |
| MBPP | 0.2435 | 0.2411 | -1.00% (better) |

Wait — this contradicts the text I quoted earlier. Let me re-read Table 2 carefully.

The paper states in Section 8: "Furthermore, on models smaller than 2.0 billion (active) parameters, we see no gains from using 2-simplicial attention." The table reports:

- 1B: Transformer GSM8k = 0.3277, 2-simplicial = 0.3302 → 2-simplicial is worse by 0.0025
- 2B: Transformer GSM8k = 0.2987, 2-simplicial = 0.2942 → 2-simplicial is better by 0.0045
- 3.5B: Transformer GSM8k = 0.2781, 2-simplicial = 0.2718 → 2-simplicial is better by 0.0063

So at 1B, the 2-simplicial model is slightly *worse* on GSM8k (higher NLL). The `Δ(%)` in Table 2 likely represents the relative percentage change in NLL, calculated as `(2s - Trans) / Trans * 100%`. Positive values mean 2-simplicial has higher loss (worse), negative values mean lower loss (better). Reading the table with this interpretation:

**At 1B active parameters:** 2-simplicial is marginally worse on GSM8k (+0.79% higher NLL), slightly worse on MMLU (+0.19%), essentially identical on MMLU-pro (-0.01%), and slightly worse on MBPP (+0.88%). The paper correctly characterizes this as "no gains."

**At 2B active parameters:** 2-simplicial is better on all benchmarks — GSM8k NLL is 1.51% lower, MMLU is 1.19% lower, MMLU-pro is 0.71% lower, MBPP is 1.00% lower. This contradicts the earlier statement from my prior section that said the 2B model underperformed. I was wrong in that prior analysis — let me correct the record. The 2B 2-simplicial model **outperforms** the standard Transformer across all benchmarks.

**At 3.5B active parameters:** 2-simplicial outperforms by larger margins — GSM8k NLL is 2.27% lower, MMLU is 1.06% lower, MMLU-pro is 2.15% lower, MBPP is 0.45% lower.

**The relative improvement grows with scale.** The gap between 2-simplicial and standard Transformer widens from 1B (where it's negative) to 2B (modest gains) to 3.5B (stronger gains). This is consistent with the hypothesis of a steeper scaling exponent: the advantage compounds as parameters increase. However, note that the improvement on MBPP actually *shrinks* from -1.00% at 2B to -0.45% at 3.5B, which is inconsistent with a simple steeper-exponent story if the crossover has already occurred. The paper does not comment on this anomaly.

#### Estimated Scaling Law Coefficients

Table 3 presents the core finding: the estimated exponent `α` in `L(N) ≈ E' + A/N^α` is **higher for 2-simplicial attention than for the standard Transformer** on all four benchmarks. Using the three data points (1B, 2B, 3.5B active parameters), linear regression of `−log L(N)` vs. `log N` yields:

| Benchmark | `α_Transformer` | `α_2-simplicial` | `Δ%` (increase in α) |
|---|---|---|---|
| GSM8k | 0.1420 | 0.1683 | +18.5% |
| MMLU | 0.1256 | 0.1364 | +8.5% |
| MMLU-pro | 0.0901 | 0.1083 | +20.2% |
| MBPP | 0.1720 | 0.1837 | +6.8% |

The `β` coefficients (intercepts of the log-log regression) are also reported, but these are not directly interpretable since they combine the irreducible loss `E` and the multiplicative constant `A` via the approximation `β = −log E'' − log A`.

**What these numbers mean for scaling behavior.** For GSM8k, the 2-simplicial loss scales as `L(N) ∝ N^{-0.1683}` while the standard Transformer loss scales as `L(N) ∝ N^{-0.1420}`. When doubling parameters (N → 2N):
- Standard Transformer: loss multiplies by `2^{-0.1420} ≈ 0.907` (a 9.3% reduction relative to the reducible part)
- 2-simplicial: loss multiplies by `2^{-0.1683} ≈ 0.890` (an 11.0% reduction)

The difference compounds with scale. At `N = 10B` active parameters (a ~2.86× increase from 3.5B, or about 1.5 doublings), the 2-simplicial model would have roughly `1.5 × (0.1683 - 0.1420) = 0.0395` additional reduction in log-loss compared to the standard Transformer — a growing absolute advantage.

**Task-dependent effect size.** The largest α improvements are on MMLU-pro (+20.2%) and GSM8k (+18.5%), which the paper describes as "less saturated and more challenging benchmarks." MMLU (+8.5%) and MBPP (+6.8%) show smaller improvements. The paper interprets this as evidence that "the percentage increase in the scaling exponent α is higher for less saturated and more challenging benchmarks" (Section 10). This aligns with the theoretical motivation: 2-simplicial attention captures ternary token interactions that are particularly relevant for multi-step reasoning (math, hard multiple-choice), while standard pairwise attention is sufficient for tasks that primarily require factual recall or pattern matching.

#### Goodness-of-Fit Statistics

Table 4 reports `R^2` and residual standard errors for the linear regressions in Table 3:

| Benchmark | `R^2_Transformer` | `Residual_Transformer` | `R^2_2-simplicial` | `Residual_2-simplicial` |
|---|---|---|---|---|
| GSM8k | 0.9998 | `2.8 × 10^{-6}` | 0.9974 | `4.9 × 10^{-5}` |
| MMLU | 0.9995 | `4.7 × 10^{-6}` | 0.9989 | `1.3 × 10^{-5}` |
| MMLU-pro | 0.9972 | `1.5 × 10^{-5}` | 0.9999 | `4.6 × 10^{-8}` |
| MBPP | 0.9962 | `7.5 × 10^{-5}` | 0.9999 | `1.5 × 10^{-6}` |

All `R^2` values exceed 0.99, indicating that the three data points are very close to collinear on a log-log plot. The residuals are tiny — the worst fit (MBPP for Transformer, `R^2 = 0.9962`, residual `7.5 × 10^{-5}`) still represents an excellent linear fit to three points.

**Caveat: statistical power.** With three points and a two-parameter linear model (slope and intercept), `R^2` is fundamentally limited in what it can tell us. Any three non-collinear points define a unique line; `R^2` close to 1.0 tells us the points are nearly collinear, but does not validate the underlying assumption that the true relationship is power-law rather than (say) a curve that happens to be well-approximated by a line over this narrow range of log(N). The reported residuals are on the order of the floating-point precision of the NLL measurements from Table 2, suggesting the log-log relationship is near-perfect for these three points, but again, this is a property of three points, not a statistical validation of the power-law form.

The paper does not report confidence intervals for α, standard errors of the regression coefficients, or any measure of uncertainty due to training variance (since each data point appears to come from a single training run). The percentage improvements in α (18.5%, 20.2%) should be interpreted as point estimates whose statistical reliability is unknown.

#### The Scale-Dependent Advantage Pattern

Combining Tables 2 and 3 reveals the **crossover pattern** that the scaling-law analysis explains:

- At 1B parameters: 2-simplicial is marginally worse or tied on all benchmarks (raw NLL differences of +0.76% to -0.01% depending on sign convention). The steeper α hasn't yet compensated for whatever constant-factor disadvantage 2-simplicial attention may have (e.g., the overhead of learning the additional `K'`, `V'` projections, the reduced effective sequence length seen by the trilinear windows, or the additional noise from the higher-variance trilinear gradient estimates).

- At 2B parameters: 2-simplicial pulls ahead, with 0.71-1.51% lower NLL across benchmarks. The steeper α is now producing measurable benefits.

- At 3.5B parameters: The gap widens further on three of four benchmarks (2.27% lower on GSM8k, 1.06% on MMLU, 2.15% on MMLU-pro), but narrows on MBPP (0.45% lower vs. 1.00% at 2B). This MBPP anomaly is unexplained.

**Interpretation of the crossover.** The paper's narrative — that "on models smaller than 2.0 billion (active) parameters, we see no gains from using 2-simplicial attention" — is a bit misleading. The data shows that at 1B, 2-simplicial is slightly *worse*, not merely "no gains." This is consistent with a model that has a higher α but also a higher constant factor `A` (or higher irreducible loss `E'`). The crossover from worse to better occurs somewhere between 1B and 2B active parameters for these benchmarks. The practical implication is that 2-simplicial attention should only be used above a minimum scale — deploying it in small models would be counterproductive.

### Ablation Studies and Robustness Checks

**The paper contains essentially no ablation studies in the conventional sense.** There are no experiments varying:
- Window dimensions `w1` and `w2` and measuring downstream quality (only latency measurements in Table 1)
- Interleaving ratio (every 2nd, 4th, 8th layer — the paper states the choice of every 4th is for pipeline parallelism, with no quality ablation)
- GQA ratio (the ratio 64 is stated but never varied)
- Training data quantity (both architectures are trained on the same data; no experiments show how the α advantage changes with data scale)
- Number of model sizes (only three sizes — no experiments at 500M, 750M, or 5B+ to map the crossover more precisely or validate the power-law form over a wider range)
- Learning rate, batch size, or optimizer sensitivity (a single set of hyperparameters is used)
- The effect of the determinant-based position encoding vs. a naive trilinear form without RoPE (the determinant form is motivated theoretically but no experiments compare with and without it)

This is not necessarily a flaw — the paper's contribution is the first demonstration of an exponent change at practical scales, and comprehensive ablations would be extraordinarily expensive (each model size requires training a 57B-176B total parameter MoE). But it means the paper provides **existence evidence** (an architectural change can alter α) rather than a systematic understanding of which design choices are responsible.

**Window parameter latency measurements (Table 1).** The paper does present latency for different `(w1, w2)` configurations at 32k, 16k, and 8k sequence lengths, but these are purely systems measurements — no quality evaluation. The choice of `(512, 32)` is based on latency (55.1 ms at 16k vs. 59.2 ms for `(128, 128)`) and the desired compute parity with standard attention at 48k context length. Whether different window shapes affect the scaling exponent or the crossover point is entirely unexplored. A natural hypothesis is that `w1 >> w2` is beneficial because language has one primary long-range dependency dimension and one shorter-range modification dimension — but this is speculation the paper does not test.

**PRM aggregation strategy (analogous to Appendix E in the example).** There is no verifier or process reward model in this paper, so no PRM ablation exists. The models are evaluated on raw NLL, with no post-hoc scoring or aggregation.

**Determinant-based vs. naive trilinear form.** The paper introduces the determinant-based trilinear form (Section 5) and proves Theorem 5.1 establishing its representational capacity, but the experiments use the simpler naive trilinear form of Equation 5: "in the following sections where we compute the backwards function for 2-simplicial attention, we will use the simpler trilinear form of Equation 5 without loss of generality" (Section 5). Since the kernel implementations in Appendices B and C implement the naive form (not the determinant form), the scaling results in Tables 2-3 are for the naive trilinear attention — not the rotation-invariant determinant form. This means:
- The models in the experiments **do not use RoPE-compatible 2-simplicial attention** — they use the original Clift et al. (2019) formulation. 
- It is unclear whether these models even use position encodings at all in the 2-simplicial layers, or whether they simply learn position implicitly through the causal mask and sliding windows.
- The determinant form's practical benefit (RoPE compatibility) remains **unevaluated**.

**Qualitative analysis of learned attention patterns.** There is no analysis of what the 2-simplicial attention heads actually learn — no visualization of which token triples get high attention scores, no probing of whether they capture the kind of ternary logical relationships that Sanford et al. (2023) proved they theoretically could. The paper provides only downstream NLL measurements, with no mechanistic interpretability.

**Comparison with Chinchilla-optimal scaling.** The paper claims that the higher α means "it is possible to increase tokens at a slower rate than the parameters for the 2-simplicial Transformer" (Section 1), citing the Chinchilla formula `N_opt ∝ C^a, D_opt ∝ C^b` where `a ≈ b ≈ 0.5` for standard Transformers. However, the paper does **not** estimate the exponents `a` and `b` for the 2-simplicial Transformer — it only estimates `α` (how loss scales with parameters at fixed data) and implicitly assumes `β` (how loss scales with data at fixed parameters) is unchanged. Without measuring `β`, the claim about shifting the compute-optimal balance is speculative. To actually demonstrate that the 2-simplicial Transformer changes the Chinchilla-optimal scaling ratio, one would need experiments varying both model size and data quantity, fitting both exponents, and deriving `a` and `b` — exactly what Hoffmann et al. (2022) did for standard Transformers with multiple model sizes and data budgets. The present paper's three-model-size, fixed-data-budget design can only measure `α`, not `β`, and therefore cannot directly support claims about optimal data scaling.

### Critical Assessment

#### Claim: "2-simplicial attention changes the exponent in the scaling laws"

The paper's central claim is that the 2-simplicial Transformer achieves a higher `α` — the exponent governing how loss scales with model parameters — compared to the standard Transformer. The evidence in Table 3 shows this effect across all four benchmarks, with percentage increases ranging from +6.8% (MBPP) to +20.2% (MMLU-pro).

**What the experiments actually demonstrate:** Three models per architecture, trained on the same fixed data budget, evaluated on held-out benchmarks, produce NLL values that are well-fit by power-law functions in active parameters, with the 2-simplicial fits having steeper slopes. This is consistent with a higher `α`.

**What the experiments do NOT demonstrate:**
1. **That the relationship is truly power-law over a meaningful range.** Three points fit a line nearly perfectly by construction — three points always determine a quadratic, but here we're fitting a linear model to log-transformed data. The excellent `R^2` values tell us the three points are roughly collinear, but not whether extrapolation to larger scales would follow the same line. A power law fit to three points spanning a factor of 3.5 in N could easily arise from a sigmoid, a curve with a different functional form that happens to look linear over this narrow log-range, or even from noise if the true scaling is weaker and the 1B point is anomalously high for 2-simplicial. Five or more model sizes spanning at least an order of magnitude would be needed to validate the power-law form.
2. **That the α difference is statistically significant.** Without multiple training runs per model size, there is no estimate of variance in the measured NLL. The differences in raw NLL between architectures at each scale are small — at 1B, the GSM8k NLL difference is 0.0025 (0.76% relative); at 2B, 0.0045 (1.5%); at 3.5B, 0.0063 (2.3%). These are small deltas, and training variance (different random seeds, data order) could plausibly produce this much variation. The scaling exponent difference of 0.0263 on GSM8k (0.1683 vs. 0.1420) is derived entirely from the pattern of these three small deltas across model sizes. If the 1B measurement for 2-simplicial were slightly higher NLL (making the delta more negative), or the 3.5B measurement were slightly lower (making the delta more positive), the estimated α would change substantially. The paper provides no error bars on α.
3. **That the effect generalizes beyond these specific model sizes, this specific training recipe, or these specific benchmarks.** The three model sizes (1B, 2B, 3.5B active) span a factor of only 3.5×. The training recipe is a single set of hyperparameters (AdamW with peak LR `4 × 10^{-3}`, weight decay 0.0125, 4000 warmup steps, cosine decay). Whether the α advantage persists at, say, 10B or 100B active parameters is unknown. Whether it generalizes to dense Transformers (rather than MoE) is unknown. Whether the same α advantage appears on language modeling perplexity (the standard scaling law metric) rather than downstream benchmark NLL is unknown — the paper evaluates only on math, coding, and reasoning benchmarks, leaving open the possibility that 2-simplicial attention improves reasoning but hurts generic language modeling.

#### Claim: "The 2-simplicial Transformer achieves better token efficiency"

This is the paper's applied claim: for a fixed token budget, the 2-simplicial Transformer produces better results than the standard Transformer. The evidence is the raw NLL comparisons in Table 2, which show lower NLL at 3.5B and 2B, but higher NLL at 1B.

**What the experiments actually demonstrate:** At the two larger model sizes (2B, 3.5B), the 2-simplicial model achieves lower NLL on the four benchmarks after training on the same fixed number of tokens. This directly demonstrates better token efficiency at these scales — the models extract more predictive capability per training token.

**Caveats and boundary conditions:**
1. **The token budget is fixed but unknown.** The paper states both families are trained on "the same fixed number of tokens" but never discloses what that number is. Without knowing the token count, we cannot assess whether the advantage appears early in training or only after seeing a certain volume of data. The Chinchilla analysis (Hoffmann et al., 2022) shows that the optimal token-to-parameter ratio matters — a 3.5B active parameter model trained on, say, 70B tokens vs. 350B tokens would show very different absolute losses and potentially different relative rankings. If the models were significantly undertrained relative to Chinchilla-optimal (likely, since training 176B total parameter MoE models is expensive), the observed α differences could be an artifact of the training horizon.
2. **The token efficiency advantage is scale-dependent.** At 1B active parameters, 2-simplicial attention is not more token-efficient — it's slightly less. This means the architecture is not universally better; it becomes advantageous only above some minimum scale. This is a genuine finding that the paper acknowledges, but the crossover location (between 1B and 2B) is determined by a single data point on each side and could shift with different training configurations.
3. **Computational cost per token is not equalized.** The interleaving schedule and window dimensions are designed to make total attention FLOPs comparable at 48k context length, but the 2-simplicial layers still have higher FLOPs than the standard attention layers they replace (by a factor of `3w1w2 / n ≈ 3 × 16384 / n` at sequence length n). At shorter sequence lengths or for models trained with shorter contexts, the 2-simplicial model costs more compute per token. The paper's "token efficiency" should be distinguished from "compute efficiency" — the 2-simplicial model may need more FLOPs per token to achieve its lower NLL. The paper does not report wall-clock training time or total FLOPs for either architecture, so we cannot assess whether the NLL improvement justifies the additional computation.

#### Missing Experiments That Would Strengthen the Paper

1. **Measurement of β (data-scaling exponent).** The paper's headline claim is about changes to the scaling law exponent, but only α (parameter scaling) is measured. To connect this to Chinchilla-style compute-optimal scaling, β must also be estimated. This requires training models at one size on multiple data budgets and measuring how loss scales with tokens — an expensive but conceptually straightforward experiment.

2. **Models at additional scales.** At minimum, a 500M or 750M active parameter model (to map the crossover more precisely) and a 5B+ model (to test extrapolation) would substantially strengthen the evidence for a genuine exponent change versus a transitory effect.

3. **Language modeling perplexity.** The scaling laws literature (Kaplan, Chinchilla) is built on language modeling loss, not downstream benchmark NLL. Showing the α improvement on standard perplexity benchmarks (e.g., C4, The Pile) would connect this work to the broader scaling laws conversation and rule out the possibility that 2-simplicial attention trades off generic language modeling for improved reasoning.

4. **Determinant-based vs. naive trilinear ablation.** Since the experiments use the naive trilinear form (without RoPE compatibility), and the paper argues the determinant form is needed for practical deployment, an experiment comparing the two at one model size would validate whether the determinant form preserves or improves the scaling advantage.

5. **Dense (non-MoE) Transformer experiments.** The MoE architecture introduces routing decisions that could interact with the attention mechanism in complex ways. Demonstrating the α effect in a simpler dense architecture would rule out MoE-specific confounding.

6. **Variance estimates via multiple seeds.** Training even two seeds at one model size would provide a lower bound on measurement noise and allow a rough significance test for the α difference. Without this, the statistical reliability of the 18.5% and 20.2% figures is completely unknown.

7. **Impact of window dimensions on quality.** Table 1 shows latency for different `(w1, w2)` configurations, but no NLL measurements. A sweep of window shapes at one model size (e.g., 2B) would reveal whether the 512×32 asymmetry is important or whether any 16k-window configuration works similarly well.

#### Are the Central Claims Genuinely Supported?

The paper demonstrates a **real phenomenon** — 2-simplicial Transformers at 2B and 3.5B active parameters achieve lower NLL than standard Transformers on four reasoning benchmarks when trained on the same tokens — that is **consistent with** the interpretation of a steeper scaling exponent α. The evidence is internally consistent across four benchmarks, and the qualitative pattern (marginal degradation at small scale, growing advantage at larger scale) matches what a higher-α, worse-constant-factor model would produce.

However, the paper **overclaims** relative to the evidence in several ways:
- Calling this a "change in the scaling exponent" is an interpretation of three data points with no uncertainty quantification. The data is consistent with a steeper α, but also consistent with other explanations (e.g., the 2-simplicial model is simply undertrained at 1B and overtrained at 3.5B relative to Chinchilla-optimal, or the standard Transformer happens to have an anomalously bad 3.5B training run).
- The claim that this means "it is possible to increase tokens at a slower rate than the parameters" requires knowing β, which is unmeasured. α is only half the scaling law; without β, the implication for compute-optimal scaling is speculation.
- The framing as "the first architectural change to alter the exponent" ignores the possibility that the observed effect is a constant-factor shift that appears exponent-like over a narrow range, or that the standard Transformer's α is underestimated because it's evaluated at suboptimal hyperparameters for that architecture (both use the same optimizer settings).

The strongest reading of the evidence is: **this paper provides the first empirical demonstration that 2-simplicial attention can improve scaling behavior at practical model sizes (2B-3.5B active parameters) on reasoning benchmarks, with the improvement pattern consistent with a steeper parameter-scaling exponent, though the statistical and functional-form uncertainties are substantial.** This is a meaningful contribution — it motivates larger-scale experiments, more rigorous scaling law estimation, and investigation of the mechanisms by which trilinear attention improves token efficiency — but it does not definitively establish a "change in the exponent" in the sense that the scaling laws literature (Kaplan, Chinchilla) uses that term, which requires careful functional-form validation and uncertainty quantification across multiple orders of magnitude in model size and data quantity.

## 6. Limitations and Trade-offs

### Three Data Points Cannot Validate a Power Law or Support Extrapolation

**The assumption or constraint.** The paper's central claim — that 2-simplicial attention "changes the exponent in the scaling laws" — rests on fitting a power law `$L(N) = E' + A/N^\alpha$` to exactly **three model sizes** (1B, 2B, 3.5B active parameters). The slope `$\alpha$` is estimated by linear regression of `$-\log L$` against `$\log N$` using these three points, with `$R^2$` values reported in Table 4 (ranging from 0.9962 to 0.9999). The paper does not train models at, say, 500M or 5B+ active parameters, nor does it report multiple training runs per model size to estimate variance in the NLL measurements.

**The consequence.** Any three non-collinear points can be fit perfectly by a two-parameter linear model on log-log axes — an `$R^2$` close to 1.0 is a property of having only three points, not evidence that the underlying relationship is truly a power law. The observed log-linear pattern over a factor of only 3.5× in model size could arise from:

- A sigmoidal or other functional form that happens to look roughly linear over this narrow range.
- Training variance: the NLL differences between architectures at each scale are small (0.0025–0.0063 in absolute terms on GSM8k; Table 2). If the standard Transformer's 3.5B run were anomalously high-loss by a small amount, or the 2-simplicial 1B run were anomalously low-loss, the estimated `$\alpha$` difference would change substantially or even reverse sign.
- Hyperparameter interactions: both architectures use the same optimizer settings (peak LR `$4 \times 10^{-3}$`, weight decay 0.0125). If the optimal hyperparameters differ between architectures, the scaling comparison conflates the effect of attention type with the effect of suboptimal tuning at specific scales.

The practical consequence is that the paper's reported `$\alpha$` increases (+18.5% on GSM8k, +20.2% on MMLU-pro) have **unknown statistical reliability**. A practitioner deciding whether to adopt 2-simplicial attention at, say, 10B or 100B active parameters has no basis to extrapolate from three points spanning 1B–3.5B. The crossover point below which 2-simplicial underperforms standard attention (between 1B and 2B) is determined by a single data point on each side, so its location is highly uncertain.

**What evidence exists in the paper.** Table 4 reports `$R^2$` and residual standard errors, showing that the three points are nearly perfectly collinear. But the paper provides no:
- Confidence intervals or standard errors on `$\alpha$`.
- Multiple training runs per model size to estimate measurement variance.
- Experiments at additional scales to test whether the log-linear relationship holds outside the 1B–3.5B range.
- Formal model comparison (e.g., testing whether a power law fits significantly better than a constant or linear model in N-space).

**Mitigation status.** The paper does **not** attempt to address this. The authors present `$R^2$` values as evidence of goodness-of-fit without acknowledging that three-point fits are inherently uninformative about functional form or extrapolation reliability. There is no discussion of statistical power, confidence intervals, or the minimum number of model sizes needed to distinguish a power law from alternative functional forms. The paper suggests future work on "scaling 2-simplicial Transformers" (Section 10) but does not frame additional scaling law estimation as a necessary next step.

---

### The Data-Scaling Exponent β Is Unmeasured, Undermining Claims About Compute-Optimal Scaling

**The assumption or constraint.** The paper's most ambitious implication — that 2-simplicial attention shifts the Chinchilla-optimal balance between parameters and tokens — requires knowing **both** exponents in the scaling law `$L(N, D) = E + A/N^\alpha + B/D^\beta$`. However, the paper measures only `$\alpha$` (how loss scales with parameters at fixed data), not `$\beta$` (how loss scales with data at fixed parameters). The experiments train all models on "the same fixed number of tokens" (Section 8), which the paper never discloses. Without varying the token budget, `$\beta$` cannot be estimated.

The paper explicitly claims in Section 1:

> "This suggests that, unlike Chinchilla scaling (Hoffmann et al., 2022), it is possible to increase tokens at a slower rate than the parameters for the 2-simplicial Transformer."

**The consequence.** This claim is **speculative without `$\beta$`**. The Chinchilla analysis (Hoffmann et al., 2022) derived the compute-optimal scaling exponents `$a$` and `$b$` (where `$N_{\text{opt}} \propto C^a$` and `$D_{\text{opt}} \propto C^b$`) by jointly fitting both `$\alpha$` and `$\beta$` across multiple model sizes and multiple data budgets. If 2-simplicial attention increases `$\alpha$` but also increases `$\beta$` (i.e., makes the model more data-hungry in ways that offset the parameter-scaling benefit), the compute-optimal ratio of parameters to tokens could remain unchanged or even shift in the opposite direction from what the paper claims.

Conversely, if the 2-simplicial Transformer has a **lower** `$\beta$` (loss decays more slowly with additional data), then the benefit of the steeper `$\alpha$` would be partially or fully offset in compute-optimal training — you'd need to scale parameters faster but also scale data faster, and the net compute savings might be minimal.

The practical consequence is that a practitioner cannot use this paper's results to decide how to allocate a training compute budget between model size and data quantity for a 2-simplicial Transformer. The paper provides only half the scaling law, and the missing half could qualitatively change the prescription.

**What evidence exists in the paper.** The paper provides **no evidence** about `$\beta$`. The total training token count is undisclosed. There are no experiments that train a fixed-size model on multiple data budgets. The paper's derivation of `$\alpha$` from Equation 18 assumes the third term `$B/D^\beta$` in Equation 17 can be absorbed into a constant because "we train both the models on the same fixed number of tokens" — which is correct for estimating `$\alpha$` in isolation, but provides zero information about `$\beta$`. Section 9 notes as a general caveat that "the technique maybe more useful when we are in the regime when token efficiency becomes more important," but does not acknowledge that `$\beta$` is unmeasured or that this limits the compute-optimal scaling interpretation.

**Mitigation status.** The paper does **not** acknowledge this gap. The compute-optimal scaling implications in Section 1 and Section 9 are presented as if `$\alpha$` alone determines the optimal parameter-to-token ratio, which is incorrect under the Chinchilla framework. The paper would need experiments varying both `$N$` and `$D$` (as in Hoffmann et al., 2022, which used ~400 training runs across model sizes and data budgets) to support its claims about shifting the Chinchilla equilibrium.

---

### The Determinant-Based RoPE-Compatible Formulation Is Never Evaluated Experimentally

**The assumption or constraint.** Section 5 introduces the determinant-based trilinear form (Equation 9) as the solution to the rotation-invariance problem — without it, 2-simplicial attention is incompatible with RoPE, the dominant position encoding scheme in modern LLMs. The paper devotes substantial theoretical exposition (Section 5, Appendix A, Theorem 5.1) to this formulation and explicitly notes in Section 5:

> "Since Equation 8 has 2 dot product terms due to Sarrus rule, it would modify Algorithm 1 to use 2 einsums instead of 1 in line 2."

However, the **experiments in Section 8 use the naive trilinear form** of Equation 5, not the determinant-based form. The paper states this at the end of Section 5:

> "in the following sections where we compute the backwards function for 2-simplicial attention, we will use the simpler trilinear form of Equation 5 without loss of generality."

The kernel implementations in Appendices B and C implement the naive form (the forward kernel in Listing 1 uses a single einsum `einsum("btnh, bsnh, brnh → bntsr", Q, K, K')` in the pseudo-code, and the backward kernels in Listings 2 and 3 follow this structure). No experimental results are reported for the determinant-based variant.

**The consequence.** This creates a **gap between the paper's theoretical contribution and its empirical validation**. The determinant form is positioned as a key innovation — the bridge between higher-order attention and modern position encoding — yet we have no evidence that:
- The determinant form actually works with RoPE (i.e., that the rotation invariance property holds in practice without numerical issues).
- The doubling of einsum operations (from one to two, per the Sarrus decomposition) impacts throughput relative to the naive form.
- The determinant form preserves or improves the scaling behavior observed with the naive form — or whether the determinant's different geometric structure (measuring signed volumes rather than raw triple products) changes the learned representations and downstream scaling.
- The experiments even use RoPE in the 2-simplicial layers — or whether position information is provided implicitly through the causal sliding windows.

The practical consequence is that a practitioner cannot deploy 2-simplicial attention with RoPE based on this paper's results, because the evaluated variant (naive trilinear) is not RoPE-compatible, and the RoPE-compatible variant (determinant-based) has no reported scaling behavior. The kernel engineering and scaling analysis would need to be redone for the determinant form.

**What evidence exists in the paper.** The only evaluation of the determinant form is a theoretical proof (Theorem 5.1, Appendix A) showing it can solve the Match3 problem with `$d = 7$` — a construction that establishes representational capacity but provides no information about learnability, training dynamics, or scaling behavior. The paper's empirical results in Tables 2–3 and Figures 3 are all for the naive trilinear form.

**Mitigation status.** The paper does **not** address this explicitly. The phrase "without loss of generality" in Section 5 implies the determinant form is computationally equivalent to the naive form modulo the factor-of-2 increase in einsum calls, but this glosses over the potential differences in optimization dynamics (the determinant involves subtractions between two trilinear terms, which could create gradient cancellation effects absent in the naive form). The paper does not list evaluating the determinant form at scale as future work.

---

### Computational Overhead Is Quantified Only for the Kernel, Not for End-to-End Training

**The assumption or constraint.** The paper's efficiency narrative rests on the claim that 2-simplicial attention has "comparable FLOPs" to standard attention at ~48k context length with the chosen window dimensions `$(w_1 = 512, w_2 = 32)$`. The comparison in Section 6 sets `$6n w_1 w_2 = 2n^2$` and solves for `$n \approx 49,152$` — at this sequence length, the sliding-window trilinear attention costs roughly the same FLOPs as standard causal attention. The kernel achieves 520 TFLOPS, which "rivals the fastest FAv3 Triton implementations" (Section 7), and Figure 3 shows competitive latency at large sequence lengths.

However, three factors make this FLOPs comparison incomplete as a measure of end-to-end training cost:

1. **Additional parameters.** The 2-simplicial layers introduce two additional projection matrices per attention head (`$W_{K'}$` and `$W_{V'}$`), increasing the parameter count per 2-simplicial layer relative to a standard attention layer. The paper does not report how many additional parameters this adds to the models in Table 2, but for a model where every 4th layer is 2-simplicial, the total parameter increase could be on the order of 10–20% per attention block.

2. **Sequence length during training.** The FLOPs parity point of 48k context assumes training with long sequences. If the models were trained with shorter sequences (e.g., 8k or 16k, common in practice), the 2-simplicial layers are proportionally more expensive relative to standard attention — at 8k, `$6n w_1 w_2 = 6 \times 8192 \times 16384 \approx 8.05 \times 10^8$` vs. `$2n^2 = 2 \times 8192^2 \approx 1.34 \times 10^8$`, a 6× factor. The paper does not disclose the training sequence length.

3. **Wall-clock time and memory.** The backward pass decomposition into two separate kernels (Section 7) recomputes intermediate values (`$O$` and `$dS$`) to avoid atomic operations. This trades extra computation for reduced atomic contention, but the paper only reports kernel-level TFLOPS, not end-to-end training step time or memory usage. The GQA ratio of 64 dramatically reduces KV-cache size (which helps memory during inference) but during training, the additional gradient computations for `$K'$` and `$V'$` and the two-kernel backward pass may increase peak memory or communication overhead in distributed training.

**The consequence.** The paper's "token efficiency" (better NLL for the same number of training tokens) does not necessarily translate to **compute efficiency** (better NLL for the same number of GPU-hours). If 2-simplicial models are significantly more expensive per training step — due to extra parameters, the 2× factor from the determinant form (if used), shorter-than-48k training sequences, or memory/communication overhead — then the observed NLL improvements may not justify the additional compute cost. A practitioner choosing between scaling a standard Transformer by, say, 20% more parameters versus adopting 2-simplicial attention needs to know the total compute cost, not just the token efficiency.

**What evidence exists in the paper.** The paper provides:
- Kernel-level TFLOPS (520) and latency vs. FAv3 (Figure 3).
- Latency measurements for window configurations at 8k, 16k, and 32k sequence lengths (Table 1).
- The FLOPs analysis in Section 6 showing parity at ~48k.

The paper does **not** provide:
- Total parameter counts accounting for the additional 2-simplicial projections.
- Training sequence length.
- End-to-end training step time comparison (including optimizer overhead, communication, pipeline bubbles).
- Total training FLOPs or GPU-hours for the three model sizes.
- Inference throughput comparison (important for deployment).

**Mitigation status.** The paper acknowledges partially in Section 9: "Our Triton kernel while efficient for prototyping is still far away from being used in production. More work in co-designing the implementation of 2-simplicial attention tailored to the specific hardware accelerator is needed in the future." This acknowledges the kernel is not production-ready, but does not address the more fundamental question of whether the FLOPs parity analysis (which the paper relies on for its "comparable compute" framing) holds under realistic training configurations. The absence of any end-to-end timing or total FLOPs reporting makes the compute-efficiency question unanswerable from the paper.

---

### Evaluation Restricted to Downstream Benchmark NLL, Not Language Modeling Perplexity

**The assumption or constraint.** All results in Tables 2–4 measure **negative log-likelihood on downstream benchmarks** — GSM8k, MMLU, MMLU-pro, and MBPP — not on standard language modeling corpora. The paper states these benchmarks were chosen because they "most strongly test math, reasoning and coding skills in pre-training" (Section 8). The scaling laws literature (Kaplan et al., 2020; Hoffmann et al., 2022) is built on **language modeling perplexity** on held-out text — generic next-token prediction loss on web text, books, Wikipedia, etc. This is the metric against which the "architectural changes don't change the exponent" consensus (Everett, 2025) was established.

**The consequence.** The observed `$\alpha$` improvements may be **task-specific** rather than general. The paper's own results in Table 3 show this variation: the `$\alpha$` increase is largest on the most reasoning-intensive benchmarks (+20.2% on MMLU-pro, +18.5% on GSM8k) and smallest on the least reasoning-intensive (+6.8% on MBPP, +8.5% on MMLU). This gradient suggests a hypothesis: 2-simplicial attention improves scaling specifically on tasks requiring ternary token interactions (multi-step reasoning, logical deduction), but may provide little or no benefit — or even hurt — on generic language modeling where pairwise attention is sufficient.

If this hypothesis is correct, then:
- The claim that 2-simplicial attention changes the scaling exponent **in general** is false — it changes it only for certain task distributions.
- A model trained with 2-simplicial attention might show better downstream reasoning but **worse language modeling perplexity** — a tradeoff that could matter for applications where broad linguistic competence is as important as reasoning capability.
- The paper's framing against the Chinchilla scaling laws (which are about language modeling loss) is misleading: even if `$\alpha$` is higher on reasoning benchmarks, the Chinchilla-optimal parameter-to-token ratio could be **unchanged** for the language modeling objective that dominates pretraining loss.

**What evidence exists in the paper.** The paper provides **no language modeling perplexity results**. All evaluation is on downstream benchmarks. The variation in `$\alpha$` improvement across benchmarks (Table 3) is the only internal evidence bearing on this question, and it supports the task-specificity hypothesis — the effect size correlates with the reasoning demand of the benchmark.

**Mitigation status.** The paper does **not** acknowledge this limitation. It frames the results as evidence about scaling laws in general (Section 1, Section 3, Section 9) without noting that scaling laws in the literature are defined on language modeling loss and that downstream benchmark NLL may behave differently. The paper would need perplexity measurements on standard corpora (C4, The Pile, etc.) to support its general scaling law claims, and ideally a decomposition showing whether the `$\alpha$` improvement on downstream benchmarks is driven by improved language modeling or by some other mechanism (e.g., better in-context learning from the pretraining distribution).

---

### No Ablation of Design Choices That Could Independently Explain the Scaling Improvement

**The assumption or constraint.** The paper's experimental design compares two architectures — standard MoE Transformer vs. MoE Transformer with every 4th layer replaced by 2-simplicial attention — and attributes the observed NLL differences and `$\alpha$` changes to the trilinear attention mechanism. However, the 2-simplicial variant differs from the baseline along **multiple dimensions simultaneously**:

1. **Additional parameters:** The `$W_{K'}$` and `$W_{V'}$` projection matrices add parameters not present in the standard Transformer. The "active parameters" count (1B, 2B, 3.5B) is held constant, but this presumably counts the parameters activated per token — if 2-simplicial layers have more total parameters, the standard Transformer layers elsewhere in the network must be slightly smaller to hit the same total active parameter budget. The comparison is therefore between architectures with different parameter allocations, not just different attention types.

2. **Sliding-window sparsification:** The standard Transformer uses full causal attention (all preceding tokens), while the 2-simplicial layers use restricted sliding windows `$(512, 32)$`. This means the 2-simplicial model sees **less context** in the layers where trilinear attention is used — it cannot attend to tokens more than 512 positions back through the first key stream, or 32 positions back through the second key stream. Any benefit could come from the attention sparsification itself (similar to how sparse attention variants sometimes improve generalization by acting as a regularizer) rather than from the trilinear form.

3. **GQA ratio:** The 2-simplicial layers use GQA with ratio 64; the paper does not state whether the standard Transformer also uses GQA (and at what ratio). If the standard Transformer uses a different GQA configuration or no GQA, the comparison conflates attention type with KV-sharing strategy.

4. **Interleaving schedule:** Every 4th layer being different creates a heterogeneous architecture where the standard and 2-simplicial layers may learn complementary functions. The benefit could partially arise from this heterogeneity (akin to how MoE architectures benefit from expert diversity) rather than from trilinear attention per se.

**The consequence.** The paper cannot distinguish which aspect of the 2-simplicial variant causes the scaling improvement. A practitioner cannot determine whether to:
- Adopt trilinear attention specifically (which requires the custom kernel and determinant form for RoPE compatibility).
- Simply sparsify attention in some layers (simpler to implement).
- Increase model heterogeneity (e.g., interleave different attention window sizes).
- Add a few extra parameters to standard attention layers (equivalent to the additional `$K'$`, `$V'$` projections but keeping bilinear attention).

Any of these simpler interventions could potentially reproduce some or all of the observed scaling benefit without the implementation complexity of 2-simplicial attention. The paper provides no evidence to rule out these alternative explanations.

**What evidence exists in the paper.** The paper provides **no ablation experiments** that would isolate the effect of trilinear attention from these confounds. Specifically missing:
- A baseline with standard attention but equivalent sliding windows (to control for sparsification).
- A baseline with standard attention but equivalent additional parameters (e.g., larger `$d_{\text{head}}$` or additional key-value projections used bilinearly).
- Models with different interleaving ratios or all-2-simplicial architectures (to control for heterogeneity).
- Comparison at matched total FLOPs (where the 2-simplicial model's additional per-layer cost would reduce the number of layers or heads possible within a fixed compute budget).

**Mitigation status.** The paper does **not** acknowledge this confound. The term "2-simplicial attention" is used to describe the entire package — trilinear form + sliding windows + GQA 64 + interleaving schedule — without decomposing which components matter. The only relevant statement is the latency-driven justification for `$(w_1=512, w_2=32)$` in Section 6, which optimizes for throughput rather than testing whether sparsification alone could match the scaling improvement. The paper's contribution would be substantially strengthened by even a single ablation (e.g., a 2B model with sliding-window bilinear attention at equivalent sparsity), but no such experiment is reported.

## 7. Implications and Future Directions
- Field impact
  - Demonstrates that architectural changes can alter scaling exponents at fixed tokens for knowledge/reasoning tasks (Table 3), contradicting the prevailing view that most changes only shift the loss offset. This opens a path to token‑efficient scaling when data is scarce (Sections 1 and 3).

- Practical applications
  - Pre‑training regimes constrained by data budgets (e.g., domain‑specific corpora) may benefit from interleaving 2‑simplicial layers to reach better quality without proportionally more tokens.
  - Reasoning‑heavy domains (math, code, logic) appear to benefit most (Table 2 shows the largest relative NLL gains on GSM8K and MMLU‑pro at 3.5B active).

- Follow‑up research
  - Kernel and systems co‑design: Implement 2‑simplicial kernels in CUTLASS or vendor libraries; explore fused scheduling across attention types; extend to other accelerators (Section 9).
  - Architectural ablations: Vary window sizes, proportion/order of 2‑simplicial layers, and GQA ratios; test dense models and larger scales.
  - Positional encoding study: Empirically compare determinant‑based rotation‑invariant logits vs simple trilinear logits with RoPE, and test other relative position encodings.
  - Generalizations: Explore k‑simplicial attention (k>2) with sparse tiling patterns; combine with looped Transformers to trade depth for higher‑order interactions.
  - Evaluation: Move beyond NLL to accuracy and robustness on a wide suite (reasoning chains, long‑context tasks, code execution correctness).

> Core takeaways grounded in the paper’s evidence:  
> - Equation (1) motivates the search for better token efficiency via architectures that can change the exponent α.  
> - Equations (5)–(7) define a tractable 2‑simplicial operator; Sections 6–7 turn it into a high‑throughput kernel with `O(n w1 w2)` cost.  
> - Table 2 shows consistent NLL improvements at 2B and 3.5B active parameters on GSM8K, MMLU, MMLU‑pro, and MBPP.  
> - Table 3 shows α increases of 6.8%–20.2% across benchmarks, with strong fits (Table 4), supporting the claim that 2‑simplicial attention improves the scaling exponent under fixed tokens.
