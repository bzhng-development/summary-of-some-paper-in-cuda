# Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality

**ArXiv:** [2405.21060](https://arxiv.org/abs/2405.21060)

## 🎯 Pitch

This paper unveils a deep mathematical connection between state space models (SSMs) and attention mechanisms by showing they are two perspectives on the same class of structured matrices—semiseparable matrices. Leveraging this duality, the authors introduce the SSD (Structured State Space Dual) framework and a new Mamba-2 architecture, achieving unprecedented efficiency: up to 8× faster than prior SSMs and on par with optimized attention for long sequences. This unified theory bridges two major modeling paradigms, enabling more scalable, hardware-friendly sequence models without sacrificing accuracy, and promises new directions for both algorithmic research and practical deployment of long-context neural networks.

---

## 1. Executive Summary

This paper introduces a theoretical framework connecting state space models (SSMs) and attention through the lens of structured matrices, revealing that these model families are closely related via decompositions of semiseparable matrices. Using this **Structured State Space Duality** (SSD) framework, the authors design **Mamba-2**, a refined architecture whose core layer is a scalar-identity structured SSM that admits both a linear recurrent form and a quadratic attention-like dual form—enabling a new block-decomposition algorithm that combines the hardware efficiency of matrix multiplications with the linear scaling of recurrences. On language modeling, Mamba-2 achieves 2–8× faster training than Mamba's optimized selective scan while maintaining competitive perplexity, and in FLOPs-matched scaling law comparisons it Pareto-dominates both Mamba and a strong Transformer++ baseline, establishing that the SSD formulation permits much larger state sizes (up to 8× or more) with minimal slowdown—but only when the underlying structured matrix admits the specific 1-semiseparable form that enables the dual quadratic computation.

## 2. Context and Motivation

### The Core Problem: SSMs and Transformers Occupy Separate Research Ecosystems

The fundamental problem this paper addresses is not simply that SSMs and Transformers are different architectures—it is that **they have developed in disjoint research communities with incompatible tools, optimizations, and theoretical understanding**. This fragmentation creates concrete practical problems: researchers must choose one lineage or the other, improvements on one side do not readily transfer to the other, and there is no shared language for reasoning about the tradeoffs between them.

Transformers, particularly decoder-only autoregressive models like GPT and Llama, have driven the deep learning revolution through their softmax attention mechanism. The community has accumulated a vast ecosystem around them: hardware-efficient implementations like FlashAttention (Dao 2024) that leverage GPU tensor cores through matrix multiplications, model parallelism strategies like tensor parallelism (Shoeybi et al. 2019) that enable scaling to hundreds of billions of parameters, sequence parallelism techniques for extremely long contexts, and a rich body of theoretical work interpreting attention as performing gradient descent or implementing specific algorithms. These optimizations are the product of years of collective engineering effort and are not straightforward to adapt to other architectures.

Structured state space models (SSMs)—exemplified by S4 (Gu, Goel, and Ré 2022), S5 (J. T. Smith, Warrington, and Linderman 2023), and Mamba (Gu and Dao 2023)—have emerged from a completely separate intellectual tradition. They are inspired by continuous-time dynamical systems, signal processing, and control theory rather than the pairwise token interaction mechanism of attention. Their primary appeal is asymptotic scaling: SSMs scale linearly in sequence length during training (versus quadratic for attention) and maintain a constant-size state during autoregressive generation (versus a cache scaling linearly with sequence length for attention). Recent work showed that selective SSMs like Mamba can match or exceed Transformers on language modeling at small to medium scale, demonstrating that the alternative approach is not just theoretically elegant but practically competitive.

However, the paper explicitly identifies multiple ways this fragmentation is costly:

> "the development of SSMs have appeared disjoint from the community's collective effort to improve Transformers, such as understanding them theoretically as well as optimizing them on modern hardware. As a result, it is more difficult to understand and experiment with SSMs compared to Transformers, and it remains challenging to train SSMs as efficiently as Transformers from both an algorithmic and systems perspective."

This is not just an academic concern. The practical consequence is that even when SSMs match Transformers on quality metrics, they can be **harder to train at scale** because they cannot benefit from the systems optimizations that have been engineered for Transformers over the past seven years. Tensor parallelism, for instance, was designed around the structure of attention and MLP blocks; adapting it to Mamba turns out to add synchronization overhead (Section 8.1). The hardware-aware selective scan implementation in Mamba, while faster than a naive recurrence, **does not use matrix multiplication units (tensor cores) on modern GPUs**, which are the most powerful computational resource available (Section 6, Figure 10). This means that even the optimized Mamba is leaving significant hardware performance on the table compared to attention implementations that are built around matrix multiplications.

### The Specific Gap: No Theoretical Bridge Between Linear Recurrences and Quadratic Attention

Beyond the practical fragmentation, the paper identifies a precise theoretical gap. Prior to this work, there was no unified framework connecting the linear-time SSM recurrence and the quadratic-time attention computation. The relationship was understood only in a limited special case:

The Linear Attention framework (Katharopoulos et al. 2020) established that **kernelized attention without softmax** could be computed either in quadratic form (materializing the attention matrix) or in linear form (via a recurrence using associativity of matrix multiplication). The paper's title—"Transformers are SSMs"—is explicitly an homage to Katharopoulos et al.'s seminal paper "Transformers are RNNs," which first showed a duality between quadratic kernel attention and a linear recurrence. However, Linear Attention establishes this duality only for the specific case where:

1. The softmax is dropped and replaced by a kernel feature map.
2. The causal mask is a lower-triangular matrix of 1's (a cumulative sum operator in the recurrent form).

The paper asks a broader question: **what class of mask matrices enables both efficient quadratic and linear computation?** And conversely, **what class of SSMs admits a dual quadratic form that looks like attention?** The answer, developed in Sections 3–5, turns out to involve semiseparable matrices—a class of rank-structured matrices that had not previously been connected to either SSMs or attention in the deep learning literature.

### Prior SSM Approaches and Their Limitations

The paper distinguishes several phases of structured SSM development, each with specific limitations:

**Time-Invariant SSMs (S4, S4D, DSS).** These models use fixed parameters throughout the sequence, making them equivalent to convolutions. Their key advantage is that they can use the **convolutional mode** for efficient parallelizable training (computing the entire output from the input in one pass) and the **recurrent mode** for efficient autoregressive inference. However, they lack **selectivity**—the ability to adapt their dynamics based on the input content. The Mamba paper (Gu and Dao 2023) showed that this selectivity is crucial for information-dense modalities like language, where the model needs to decide when to remember and when to ignore inputs.

**Selective SSMs (Mamba/S6).** Mamba introduced time-varying parameters where the recurrent matrices $(A_t, B_t, C_t)$ are functions of the input at each timestep. This makes the model much more powerful—it learns to selectively propagate or forget information—but at a computational cost:

> "However, it can only be computed in recurrent instead of convolutional mode, and requires a careful hardware-aware implementation to be efficient. Even so, it is still less efficient than hardware-friendly models such as CNNs and Transformers because it does not leverage matrix multiplication units, which modern accelerators such as GPUs and TPUs are specialized for."

In other words, Mamba trades away the convolutional training mode (which was already a secondary concern since convolutional SSMs had been largely superseded) but more critically, it cannot be computed using the matrix multiplication primitives that make Transformers fast. The selective scan requires a specialized fused CUDA kernel that still underperforms optimized attention on modern hardware.

**Diagonal vs. Scalar SSM Structure.** Within selective SSMs, the structure of the $A$ matrix matters enormously for computation. Mamba uses a diagonal $A_t$ (each of the $N$ state dimensions has its own independent scalar recurrence). This means the SSM computation factorizes into $N$ independent scalar recurrences, each of the form $h_t = a_t h_{t-1} + b_t x_t$, which can be computed efficiently with a parallel associative scan. However, the diagonal structure does not admit the attention-like dual form that this paper exploits. The key innovation of SSD is to further restrict $A_t$ to be a **scalar multiple of the identity**: all $N$ state dimensions share the same scalar dynamics. This restriction enables the dual quadratic form (Section 5.1) at the cost of expressivity, which is then compensated by increasing the state dimension $N$ and using a larger head dimension $P$ (these tradeoffs are precisely what the SSD algorithm makes practical).

### The Efficiency Frontier: Why "Just Scale Up SSMs" Is Not Enough

A reasonable reader might ask: if Mamba already matches Transformers on perplexity at moderate scales, why not simply scale up Mamba to larger models and larger state sizes? Why is a theoretical reframing necessary?

The paper's efficiency benchmark (Figure 10) makes the answer concrete. Mamba's fused scan implementation has runtime that scales **linearly** with the state dimension $N$. Doubling $N$ doubles the scan time. FlashAttention-2, by contrast, uses matrix multiplications that are **sublinear** in the effective state size due to the efficiency of tensor cores. The SSD algorithm introduced in Section 6 circumvents this bottleneck by reformulating the SSM computation as a block decomposition that uses matrix multiplications for the bulk of the work. The result (Figure 10, left): SSD is 2–8× faster than Mamba's scan at $N=64$, and can increase to $N=256$ with minimal slowdown (Figure 10, right)—opening the door to state sizes that were previously impractical.

This matters because prior work on SSMs had identified state expansion factor $N$ as a critical parameter for performance on information-dense tasks. Mamba showed that increasing $N$ helps, particularly for language modeling. But Mamba's practical state sizes were limited to $N=16$ (as used in the original Mamba paper at larger scales) because of the scan bottleneck. SSD enables $N=64$ or $N=128$ with better speed than Mamba at $N=16$. The theoretical framework thus **directly enables a capability improvement**—larger state sizes—that was theoretically desirable but practically infeasible before.

### The Organization of Prior Knowledge and How This Paper Reorganizes It

The paper's motivation can be understood as reorganizing a fragmented landscape. Before this work, the map of sequence models looked like:

- **Attention**: defined by quadratic pairwise interactions; hardware-efficient via matrix multiplications; KV cache grows with sequence length.
- **Linear Attention**: drops softmax for kernel function; has dual quadratic/linear forms; loss of expressivity relative to softmax attention.
- **SSMs**: defined by linear recurrences; linear scaling in sequence length; constant-size state; hardware-inefficient compared to attention.
- **Structured Matrices**: a classical topic in numerical linear algebra; semiseparable matrices specifically had never been connected to deep learning sequence models.

The paper's central reorganization (Figure 1) places **semiseparable matrices** as the bridge:

1. SSMs are **exactly** N-semiseparable matrix transformations (Section 3, Theorem 3.5).
2. Attention variants are **masked kernel attention** that can be abstracted as structured masked attention (SMA), where the mask $L$ can be any structured matrix (Section 4, Definition 4.2).
3. When $L$ is a **1-semiseparable matrix** and the SSM has **scalar-identity structure** on $A$, these two model classes **coincide** (Section 5, Corollary 5.1). This is the SSD model.
4. Any semiseparable matrix can be decomposed into blocks; the diagonal blocks can use the quadratic (attention-like) computation while off-diagonal blocks use the linear (recurrence-like) computation (Section 6).

This reorganization is powerful because it provides a **vocabulary for reasoning about tradeoffs** that was previously absent. For example, the Mamba architecture's choice to share $B$ and $C$ across input channels is revealed (Proposition 7.2) to be analogous to **multi-value attention** (MVA)—a head pattern that, from the attention perspective, is non-standard. The paper can then systematically ablate head patterns (multi-head, multi-query, multi-value) and discover that the MVA pattern indeed performs best for SSMs (Section 9.4.2, Table 5). This is the kind of insight that emerges when two previously separate research traditions are connected.

### How This Paper Positions Itself Relative to Existing Work

The paper positions itself not as proposing a single new method but as **building infrastructure**: a theoretical framework that enables transferring algorithmic, systems, and architectural innovations between the Transformer and SSM lineages. The abstract states this ambition clearly:

> "Our state space duality (SSD) framework allows us to design a new architecture (Mamba-2) whose core layer is an a refinement of Mamba's selective SSM that is 2-8× faster, while continuing to be competitive with Transformers on language modeling."

Note the framing: Mamba-2 is a **demonstration** of the framework's utility, not the framework itself. The framework is the contribution; the architecture and algorithm are instantiations that validate it.

The paper draws a deliberate parallel to the Chinchilla scaling laws (Hoffmann et al. 2022), which provided a principled framework for allocating pretraining compute. Just as Chinchilla reoriented the field from "make models bigger" to "scale model size and data together optimally," SSD aims to reorient the conversation around sequence models from "choose attention or SSMs" to "use the right computation pattern for the right setting." The block-decomposition algorithm is the practical embodiment of this philosophy: use attention-like matrix multiplications where they are fast (within chunks), use recurrence where state compression is efficient (between chunks), and allow the framework to determine the optimal boundary.

The paper is also explicitly refreshing the Linear Attention narrative. While Katharopoulos et al. (2020) showed that "Transformers are RNNs," their result applied only to a specific form of masked attention (causal mask of 1's). Subsequent work like RetNet (Y. Sun et al. 2023) and TransNormer (Qin, Dong Li, et al. 2023) extended this to decay masks (where the mask matrix is $L_{ij} = \gamma^{i-j}$), and GateLoop (Katsch 2023) independently proposed input-dependent decay factors. The SSD framework subsumes all of these as special cases of **structured masked attention** with different choices of the structured matrix $L$, and generalizes the duality to any semiseparable mask—importantly including **input-dependent** masks that enable the selectivity which Mamba showed to be critical.

### The Title's Significance

The title "Transformers are SSMs" functions at multiple levels. At face value, it claims that attention mechanisms can be viewed as state space models—a direct extension of "Transformers are RNNs." More subtly, it asserts a strong equivalence: not just that Transformers have a recurrent form (which was already known for linear attention), but that the **full theoretical machinery of SSMs**—including state expansion, selectivity, and structured matrix algorithms—applies to understanding and improving attention-like models. The duality goes both ways: SSMs are also revealed to have an attention-like dual form, which enables the hardware-efficient algorithm that makes Mamba-2 practical. The title is thus a statement about **bidirectional knowledge transfer** between two previously separate research paradigms.

## 3. Technical Approach

### 3.1 Reader Orientation

This is primarily a **theoretical unification paper** whose core idea is that state space models and attention are not separate model classes but rather dual computational perspectives on the same underlying mathematical object—structured semiseparable matrices. By identifying this equivalence, the paper derives a new matrix-multiplication algorithm (the SSD algorithm) that combines the linear scaling of recurrences with the hardware efficiency of matrix multiplications, enabling a refined architecture (Mamba-2) that can be trained faster and with larger state sizes than its predecessor while remaining competitive with Transformers on language modeling quality. The "shape" of the solution is not a single new module but a **framework** that reinterprets existing sequence models as structured matrix transformations, then leverages classical numerical linear algebra to design more efficient computations.

### 3.2 Big-Picture Architecture (Diagram in Words)

The SSD framework has three conceptual layers, each built on the previous one:

1. **SSMs as Semiseparable Matrices (Section 3).** The sequence-to-sequence transformation performed by any state space model with state size $N$ is **exactly** a multiplication by an $N$-semiseparable matrix—a matrix where every submatrix contained strictly below the diagonal has rank at most $N$. This equivalence is exact and bidirectional: the SSM's recurrence parameters $(A_t, B_t, C_t)$ define the compressed SSS (sequentially semiseparable) representation of the matrix, and every semiseparable matrix can be written in this form. This layer establishes the **linear (recurrent) vs. quadratic (naive matrix multiplication) duality** for SSMs: the standard recurrence computes the matrix-vector product in $O(TN)$ time by exploiting the compressed representation, while naive matrix multiplication materializes the full $T \times T$ matrix and costs $O(T^2)$.

2. **Structured Masked Attention (SMA) as a Contraction (Section 4).** Masked kernel attention $(L \circ QK^\top) \cdot V$ is reframed as a single 4-way tensor contraction `contract(TN, SN, SP, TS → TP)(Q, K, V, L)`. The standard attention computation orders this contraction as $(L \circ (QK^\top)) \cdot V$, which is quadratic. Linear attention orders it as $Q \cdot (L \cdot (K^\top V))$, where the inner multiplication by the mask $L$ becomes efficient when $L$ is a structured matrix. This layer generalizes linear attention from the specific case where $L$ is a causal mask of 1's (a cumulative sum operator) to any structured matrix $L$ that admits fast multiplication—including decay matrices, Toeplitz matrices, Fourier matrices, and crucially, semiseparable matrices.

3. **State Space Duality (Section 5).** These two frameworks intersect when the SSM uses **scalar-identity structure** on $A$ (i.e., $A_t = a_t \cdot I$, where $a_t$ is a scalar) and the SMA mask $L$ is **1-semiseparable** (generated by exactly those same scalars $a_t$). At this intersection, the SSM's quadratic naive computation is **identical** to the SMA quadratic attention computation, and the SSM's linear recurrent computation is **identical** to the SMA linear form. This is the SSD model: a single mathematical function that can be legitimately viewed either as a selective SSM or as input-dependent structured attention.

Information flows through the **SSD computation** (Section 6) as follows: the input sequence $X \in \mathbb{R}^{(T, P)}$ is partitioned into chunks of size $Q$. For each chunk's **diagonal block** (intra-chunk computation), the quadratic attention-like form is used—efficiently implemented via matrix multiplications. For the **off-diagonal blocks** (inter-chunk computation), the low-rank structure of semiseparable matrices is exploited: the output from earlier chunks is compressed into a state and passed forward, then expanded back to full dimension. The chunk boundary is the algorithmic decision point: within a chunk, use hardware-friendly matrix multiply; across chunks, use the compressed state recurrence to maintain linear scaling. Finally, the **Mamba-2 architecture** (Section 7) wraps this SSD layer in a Transformer-style block with parallel projections, multi-head structure, and normalization layers.

### 3.3 Roadmap for the Deep Dive

- **First, the SSM-to-semiseparable equivalence (Theorem 3.5).** This is the foundation: I need to show exactly how unrolling an SSM recurrence produces a matrix, why that matrix has the semiseparable property, and how the SSS representation encodes it. Understanding this reveals why SSMs have both a linear and quadratic form.

- **Second, the 1-semiseparable special case.** The simplest non-trivial case where $N=1$ produces the scalar SSM recurrence $y_t = a_t y_{t-1} + x_t$, which is the primitive operation that everything else builds on. I'll explain why this is a "cumprodsum" operator and how it connects to the structural decompositions used in Section 6.

- **Third, the tensor contraction reframing of masked attention (Section 4).** Instead of treating attention as the standard three-matrix formula, I'll show how it can be written as a single 4-way contraction, and how the choice of contraction order determines whether the computation is quadratic (standard attention) or linear (linear attention). This generalizes immediately to structured masked attention by replacing the causal mask with any structured matrix.

- **Fourth, the scalar-identity SSM and its attention dual (Section 5.1).** When the SSM's $A$ matrix is a scalar times identity, the SSM's quadratic naive form becomes exactly a kernel attention computation. I'll show the algebraic manipulation that reveals this and explain why this specific restriction is necessary for the duality to hold.

- **Fifth, 1-semiseparable SMA and the general equivalence theorem (Sections 5.2–5.3).** I'll show the converse direction: when SMA uses a 1-semiseparable mask, its linear form reduces to a scalar SSM recurrence. Theorem 5.2 then characterizes **all** SMA instances with efficient autoregression as semiseparable SMA. The intersection is the SSD model.

- **Sixth, the block-decomposition SSD algorithm (Section 6).** This is the practical payoff: how to decompose a semiseparable matrix into diagonal blocks (computed via attention-like matrix multiplies) and low-rank off-diagonal blocks (computed via state passing). I'll walk through the three-step computation (right factors, center factors, left factors), their interpretations as state space operations, and the computational cost analysis.

- **Seventh, the Mamba-2 architecture (Section 7).** I'll explain how the theoretical framework informs architectural decisions: parallel vs. sequential projections, head patterns (multi-value vs. multi-head vs. multi-query), normalization, and kernel feature maps.

### 3.4 Detailed, Sentence-Based Technical Breakdown

#### The SSM as a Matrix Transformation

The paper begins by writing down what happens when you unroll an SSM recurrence step by step. This is simple algebra but reveals the matrix structure that everything else depends on.

Start with the SSM definition from equation (2):

$$h_t = A_t h_{t-1} + B_t x_t$$
$$y_t = C_t^\top h_t$$

where $h_t \in \mathbb{R}^N$ is the hidden state at time $t$, $A_t \in \mathbb{R}^{(N,N)}$ is the state transition matrix, $B_t \in \mathbb{R}^{(N,1)}$ is the input projection vector, $C_t \in \mathbb{R}^{(N,1)}$ is the output projection vector, and $x_t, y_t \in \mathbb{R}$ are scalar input and output at time $t$.

By definition, $h_0 = B_0 x_0$ (assuming zero initial state). Unrolling the recurrence for subsequent timesteps:

$$h_1 = A_1 h_0 + B_1 x_1 = A_1 B_0 x_0 + B_1 x_1$$
$$h_2 = A_2 h_1 + B_2 x_2 = A_2 A_1 B_0 x_0 + A_2 B_1 x_1 + B_2 x_2$$

The general formula for $h_t$ is a sum over all previous inputs $x_s$ for $s \leq t$, where each $x_s$ is multiplied by the cumulative product of $A$ matrices from time $s+1$ to $t$:

$$h_t = \sum_{s=0}^t A_{t:s}^\times B_s x_s$$

where $A_{t:s}^\times = A_t A_{t-1} \cdots A_{s+1}$ denotes the ordered product of $A$ matrices from index $s+1$ up to $t$ (with $A_{t:t}^\times = I$ by convention).

Multiplying by $C_t^\top$ to get the output at time $t$:

$$y_t = \sum_{s=0}^t C_t^\top A_{t:s}^\times B_s x_s$$

Now comes the key observation. If we collect all inputs $x_0, \ldots, x_{T-1}$ into a vector $x \in \mathbb{R}^T$ and all outputs $y_0, \ldots, y_{T-1}$ into a vector $y \in \mathbb{R}^T$, the entire sequence transformation can be written as a matrix-vector multiplication $y = M x$ where:

$$M_{ji} = C_j^\top A_j A_{j-1} \cdots A_{i+1} B_i \quad \text{for } j \geq i$$

and $M_{ji} = 0$ for $j < i$ (since the SSM is causal—output at time $j$ only depends on inputs at times $i \leq j$).

**What this computes:** the matrix $M \in \mathbb{R}^{(T,T)}$ is the full linear transformation implemented by the SSM. Each entry $M_{ji}$ tells you how much input $x_i$ contributes to output $y_j$ after passing through the state dynamics from time $i$ to time $j$. The SSR expresses this as $C_j^\top$ (readout at time $j$) times the cumulative state transition $A_j \cdots A_{i+1}$ (propagation from $i$ to $j$) times $B_i$ (input encoding at time $i$).

**Why this form matters:** this explicit matrix representation reveals that an SSM is not just a recurrence—it is a **linear transformation on the sequence** whose structure is completely determined by the $(A_t, B_t, C_t)$ parameters. The recurrence is one algorithm for computing this transformation; the matrix is the transformation itself. The distinction between what a model computes and how it computes it is the conceptual breakthrough that drives the entire paper.

#### Equivalence to Semiseparable Matrices

The matrix $M$ defined above has a specific algebraic form. The paper identifies this as the **sequentially semiseparable (SSS) representation** of a semiseparable matrix.

**Definition 3.2** (SSS representation): A lower triangular matrix $M \in \mathbb{R}^{(T,T)}$ has an $N$-SSS representation if it can be written as:

$$M_{ji} = C_j^\top A_j \cdots A_{i+1} B_i$$

for vectors $B_0, \ldots, B_{T-1}, C_0, \ldots, C_{T-1} \in \mathbb{R}^N$ and matrices $A_0, \ldots, A_{T-1} \in \mathbb{R}^{(N,N)}$. The paper defines the operator $M = \text{SSS}(A_{0:T}, B_{0:T}, C_{0:T})$ as the construction of this matrix from these parameters.

**What this defines:** a compressed parameterization of an otherwise-dense $T \times T$ matrix. Instead of $O(T^2)$ entries, the SSS representation uses $O(N^2 T)$ parameters (for the $A$ matrices) plus $O(NT)$ for $B$ and $C$. When $N \ll T$, this is a dramatic compression.

Now, what makes these matrices "semiseparable"? **Definition 3.1:** A lower triangular matrix $M$ is $N$-semiseparable if every submatrix contained strictly in the lower triangular portion (on or below the diagonal) has **rank at most $N$**. That is, for any $i' < i \leq j < j'$, the block

$$\begin{bmatrix} M_{j,i'} & \cdots & M_{j,i-1} \\ \vdots & & \vdots \\ M_{j'-1,i'} & \cdots & M_{j'-1,i-1} \end{bmatrix}$$

has rank $\leq N$.

**Lemma 3.3** proves one direction: every $N$-SSS matrix is $N$-semiseparable. The proof is constructive: for any off-diagonal block $M_{j:j', i':i}$ with $j' > j \geq i > i'$, the paper exhibits an explicit rank-$N$ factorization. The $(a, b)$ entry of this block (with $a$ indexing rows $j$ to $j'-1$ and $b$ indexing columns $i'$ to $i-1$) is $C_{j+a}^\top A_{j+a:j}^\times \cdot A_{j:i-1}^\times \cdot A_{i-1:i+b}^\times B_{i+b}$. The factorization separates this into:

- A left factor $[C_j^\top A_{j:j}^\times; \cdots; C_{j'-1}^\top A_{j'-1:j}^\times] \in \mathbb{R}^{(j'-j, N)}$
- A center factor $A_{j:i-1}^\times \in \mathbb{R}^{(N,N)}$
- A right factor $[A_{i-1:i'}^\times B_{i'}, \cdots, A_{i-1:i-1}^\times B_{i-1}]^\top \in \mathbb{R}^{(i-i', N)}$

The product is a $(j'-j) \times (i-i')$ matrix of rank at most $N$. **Proposition 3.4** states the converse (without proof in the paper, citing the semiseparable matrix literature): every $N$-semiseparable matrix has an $N$-SSS representation.

**Theorem 3.5** then states the main result of this section:

> "The state space model transformation $y = \text{SSM}(A, B, C)(x)$ with state size $N$ is identical to matrix multiplication by an $N$-SS matrix in sequentially semiseparable representation $y = \text{SSS}(A, B, C) \cdot x$."

**What this means operationally:** the sequence transformation that an SSM performs IS the matrix $M$, and the state size $N$ of the SSM IS the semiseparable order of the matrix. The recurrence parameters $(A_t, B_t, C_t)$ ARE the SSS representation. There is no approximation, no asymptotic equivalence—this is an exact identity.

**Why this is not just a trivial restatement:** the semiseparable characterization reveals properties of SSMs that are not obvious from the recurrence formulation. Semiseparable matrices have known compressed representations (the SSS form is one of several), fast multiplication algorithms, closure properties under addition, multiplication, and inversion (Proposition C.1), and connections to other structured matrix families (banded matrices, butterfly matrices). All of these become available as tools for analyzing and improving SSMs once the equivalence is established.

#### Computing SSMs: The Linear (Recurrent) Mode

Given the matrix equivalence, how do we actually compute $y = Mx$ efficiently? The paper identifies two modes.

**The linear mode** is the standard SSM recurrence computation. For diagonal SSMs (like S4D and Mamba, where each $A_t$ is a diagonal matrix), the computation factorizes: each of the $N$ state dimensions follows an independent scalar recurrence. The paper formalizes this as a tensor contraction sequence (equation 8):

$$Z = \text{contract}(\text{SP, SN} \to \text{SPN})(X, B) \quad \text{(S, P, N)}$$
$$H = \text{contract}(\text{TSN, SPN} \to \text{TPN})(L, Z) \quad \text{(T, P, N)}$$
$$Y = \text{contract}(\text{TN, TPN} \to \text{TP})(C, H) \quad \text{(T, P)}$$

**Step 1** (equation 8a) performs the input expansion: each of the $P$ input channels is multiplied by the corresponding $B_t$ (of shape $(P, N)$ when $P > 1$, or just a scalar $N$-vector when $P=1$), expanding the dimension from $P$ to $P \times N$. **Step 2** (equation 8b) runs the scalar SSM recurrence on each of the $P \times N$ channels independently. Here $L \in \mathbb{R}^{(T,T)}$ is defined as $\text{1SS}(A)$, meaning $L_{ji} = a_{j:i}^\times$ if $j \geq i$ and 0 otherwise, where $a_t$ is the scalar on the diagonal of $A_t$ (for the $t$-th state dimension). **Step 3** (equation 8c) contracts the expanded dimension back down using $C_t$ (of shape $(T, N)$).

**What this computes:** the full SSM output in three stages—expand the input into a higher-dimensional state space, evolve each dimension independently according to a scalar recurrence, then contract back to the output dimension. The total FLOPs are $O(TPN)$.

**Why this is "linear":** the bottleneck step is the scalar SSM recurrence (or equivalently, multiplication by the 1-SS matrix $L$). Each of these $P \times N$ recurrences costs $O(T)$ time, for a total of $O(TPN)$. The computation scales **linearly** in sequence length $T$, which is the defining efficiency advantage of SSMs over attention (which costs $O(T^2 P)$ in its standard form).

**Why Mamba's implementation required hardware-aware kernels:** the intermediate tensor $H$ has shape $(T, P, N)$, which for typical dimensions (e.g., $T=8192$, $P=64$, $N=16$) requires substantial memory. Mamba's selective scan fused the three steps to avoid materializing these expanded tensors. However, even fused, this computation does not use matrix multiplications and therefore cannot leverage tensor cores.

#### Computing SSMs: The Quadratic (Naive) Mode

The second mode is to simply **materialize the matrix $M$** and multiply by $x$. Since $M \in \mathbb{R}^{(T,T)}$, this costs $O(T^2)$ time and $O(T^2)$ memory.

**Why this would ever be considered:** for short sequence lengths, the overhead of the recurrent computation (which involves many small operations) can exceed the cost of a single dense matrix multiplication, which is highly optimized on modern hardware. Moreover, for specific structures on $A$, this quadratic computation turns out to look exactly like attention, enabling the use of optimized attention kernels.

**The significance of Theorem 3.7.** The paper states:

> "Any state space model of state size $N$ on sequence length $T$ can be computed in time $O(TN)$ (not accounting for potential preprocessing)."

This follows from Proposition 3.6 (from the semiseparable matrix literature): an $N$-SS matrix of size $T$ can be represented in $O(NT)$ parameters and has matrix-vector multiplication in time and space $O(NT)$. The theorem is non-trivial because the SSS representation appears to have $O(N^2 T)$ parameters (from the $A$ matrices). The result says that even with dense unstructured $A_t$ matrices, an SSM can theoretically be compressed to $O(NT)$ parameters and computed in $O(TN)$ time.

**The practical caveat (Remark 4):** this theoretical compression requires a preprocessing step involving operations like singular value decompositions that are both hardware-inefficient and add extra FLOPs. In practice, efficiently computable SSMs still need additional structure on $A$—which is why past work (S4, S4D, Mamba) imposed diagonal structure, and why this paper's SSD algorithm imposes the even simpler scalar-times-identity structure.

#### The 1-Semiseparable Matrix: Building Block for Everything

The paper singles out the special case of 1-semiseparable matrices because they are the fundamental primitive from which all the efficient algorithms are built. When $N=1$, the $B_i$ and $C_j$ are scalars that can be factored out of the matrix:

$$M = \text{diag}(c) \cdot \tilde{M} \cdot \text{diag}(b)$$

where $\tilde{M}_{ji} = a_{j:i}^\times = a_j \cdot a_{j-1} \cdots a_{i+1}$ (the cumulative product of $a$ values from $i+1$ to $j$).

The paper uses the notation $\text{1SS}(a_{0:T})$ to denote this matrix (equation 6), explicitly writing it out as:

$$M = \begin{bmatrix} 1 & & & & \\ a_1 & 1 & & & \\ a_2 a_1 & a_2 & 1 & & \\ \vdots & \vdots & \vdots & \ddots & \\ a_{T-1} \cdots a_1 & a_{T-1} \cdots a_2 & \cdots & a_{T-1} & 1 \end{bmatrix}$$

**What multiplication by this matrix does:** given input vector $x$, the output $y = M x$ satisfies:

$$y_t = a_{t:0} x_0 + a_{t:1} x_1 + \cdots + a_{t:t} x_t$$

This can be computed by the scalar recurrence (equation 7):

$$y_t = a_t y_{t-1} + x_t$$

**Why it is called "cumprodsum":** it combines cumulative product (the $a_{j:i}^\times$ factors) and cumulative sum (the summation over $s$). When all $a_t = 1$, it reduces to a cumulative sum $y_t = y_{t-1} + x_t$. When all $x_t = 0$ except $x_0$, it reduces to a cumulative product $y_t = a_t \cdots a_1 \cdot y_0$.

**The crucial insight for the SSD algorithm (Appendix B):** there are **many different ways** to compute 1-SS matrix multiplication, each corresponding to a different structured matrix decomposition. These include:
- **Sequential recurrence:** $O(T)$ time, $O(1)$ space per step, sequential (not parallelizable within a sequence).
- **Associative scan:** $O(T \log T)$ work or $O(T)$ work with $O(\log T)$ depth (parallelizable).
- **Dilated mode:** factors the matrix into $\log_2 T$ factors with increasing strides, $O(T \log T)$ work.
- **State-passing (chunkwise) mode:** divide the sequence into chunks; compute within each chunk using any method; pass the final state to the next chunk.
- **Block decomposition mode:** divide the matrix into hierarchical blocks; recurse on diagonal blocks in parallel; combine with low-rank off-diagonal correction.

The **SSD algorithm (Section 6)** is essentially the state-passing mode applied to the full $N$-semiseparable matrix, where the within-chunk computation uses the quadratic (attention-like) form, the between-chunk state passing uses 1-SS multiplication, and the block size $Q$ is chosen to balance the two. Appendix B's comprehensive catalog of 1-SS multiplication algorithms is therefore the toolbox from which the SSD algorithm is assembled.

#### The Tensor Contraction Reframing of Masked Attention

The paper's second major building block is a reformulation of attention as a tensor contraction, which reveals how to make attention efficient by changing the contraction order.

Standard single-head attention operates on three sequences: queries $Q \in \mathbb{R}^{(T, N)}$, keys $K \in \mathbb{R}^{(S, N)}$, and values $V \in \mathbb{R}^{(S, P)}$, where $T$ is the target sequence length, $S$ is the source sequence length, $N$ is the feature dimension, and $P$ is the head (output) dimension. In self-attention, $S = T$ and typically $N = P$.

The standard masked attention computation (equation 11) proceeds in three steps:

$$G = QK^\top \quad \text{(T, S)}$$
$$M = G \circ L \quad \text{(T, S)}$$
$$Y = M V \quad \text{(T, P)}$$

where $L \in \mathbb{R}^{(T, S)}$ is a mask matrix (for causal attention, $L$ is lower-triangular 1's).

**The key observation (equation 12):** this entire computation can be written as a single 4-way tensor contraction:

$$Y = \text{contract}(\text{TN, SN, SP, TS} \to \text{TP})(Q, K, V, L)$$

This notation means: take the four input tensors $Q$ (indices T, N), $K$ (indices S, N), $V$ (indices S, P), and $L$ (indices T, S); sum over the indices N and S; produce the output tensor $Y$ with indices T, P. The indices T, S, N, P are **not** multiplied together—they are the dimensions over which the contraction aligns and sums.

**What this contraction abstractly computes:** for each target position $t$ and output channel $p$, it computes $\sum_{s, n} Q_{t,n} \cdot K_{s,n} \cdot V_{s,p} \cdot L_{t,s}$. Because of the summation convention in einsum notation, indices that appear in the inputs but not the output (here S and N) are summed over.

**The computation is the same regardless of order, but the cost differs dramatically.** The standard attention algorithm (equation 13) implements this by:

1. First contracting $Q$ and $K$ over the feature dimension $N$: $G_{t,s} = \sum_n Q_{t,n} K_{s,n}$ (cost $O(TSN)$)
2. Then elementwise-multiplying by $L$: $M_{t,s} = G_{t,s} \cdot L_{t,s}$ (cost $O(TS)$)
3. Then contracting $M$ and $V$ over the source dimension $S$: $Y_{t,p} = \sum_s M_{t,s} V_{s,p}$ (cost $O(TSP)$)

The total is $O(TSN + TSP)$, which for self-attention ($S=T$, $N=P$) is $O(T^2 N)$.

But this is only one of many possible contraction orders. The **linear attention algorithm (equation 15)** implements the same contraction in a different order:

$$Z = \text{contract}(\text{SP, SN} \to \text{SPN})(V, K) \quad \text{(S, P, N)}$$
$$H = \text{contract}(\text{TS, SPN} \to \text{TPN})(L, Z) \quad \text{(T, P, N)}$$
$$Y = \text{contract}(\text{TN, TPN} \to \text{TP})(Q, H) \quad \text{(T, P)}$$

**Step 1** (equation 15a) contracts $V$ and $K$ over the source dimension $S$, creating a tensor $Z_{s,p,n} = V_{s,p} \cdot K_{s,n}$ (expanding the feature dimension). **Step 2** (equation 15b) is the bottleneck: multiplying by the mask $L$ along the $S$ and $T$ dimensions, producing $H_{t,p,n} = \sum_s L_{t,s} Z_{s,p,n}$. For each $(p,n)$ pair, this is a matrix-vector multiplication by $L$. **Step 3** (equation 15c) contracts $Q$ and $H$ over the feature dimension $N$ to produce the output.

**Why this can be linear:** if $L$ is a structured matrix with fast multiplication, step 2 costs $O(T \cdot \text{cost}(L))$ rather than $O(T^2)$. For the causal mask (all 1's below diagonal), multiplication by $L$ is simply a cumulative sum, costing $O(T)$ per channel. The total is $O(TPN)$, which is linear in sequence length.

**This reframing is the paper's second major conceptual contribution.** It separates the ATTENTION FUNCTION (the tensor contraction) from the ATTENTION ALGORITHM (the contraction order). The standard quadratic attention and linear attention are simply two different algorithms for computing the same function. The distinction was obscured in the traditional matrix notation $(L \circ (QK^\top)) \cdot V = Q \cdot \text{cumsum}(K^\top V)$.

#### Structured Masked Attention (SMA)

The paper generalizes the above by observing that **any** structured matrix $L$ with efficient multiplication enables a fast linear form. This yields Definition 4.2:

> "Structured masked attention (SMA) is defined as a function on queries/keys/values $Q, K, V$ as well as any structured matrix $L$ (i.e., has sub-quadratic matrix multiplication), through the 4-way tensor contraction $Y = \text{contract}(\text{TN, SN, SP, TS} \to \text{TP})(Q, K, V, L)$."

The SMA quadratic mode implements this via the standard attention contraction order (equation 13). The SMA linear mode implements it via the alternate contraction order (equation 15), where step 15b is computed using the structured matrix multiplication algorithm for $L$.

**What this abstraction accomplishes:** it creates a vocabulary for describing different efficient attention variants by their choice of $L$:
- **Linear Attention:** $L$ is the causal mask (lower triangular 1's), corresponding to a cumulative sum operator.
- **RetNet:** $L_{ij} = \gamma^{i-j} \cdot \mathbb{1}[j \geq i]$ for some decay factor $\gamma \in [0, 1]$. In the recurrent form, this is $y_t = \gamma y_{t-1} + x_t$.
- **Toeplitz SMA:** $L$ is a Toeplitz matrix (constant along diagonals), implementing a form of relative positional encoding multiplicative instead of additive (like AliBi).
- **Fourier SMA:** $L_{ij} = \omega^{ij/T}$, encoding positional structure through Fourier transforms.
- **Semiseparable SMA:** $L$ is a semiseparable matrix—the focus of this paper.

**Why semiseparable SMA is special:** among all these choices, semiseparable matrices are exactly the ones that have **efficient autoregressive generation**—a point formalized in Theorem 5.2 (proved in Appendix C.2). An autoregressive transformation of order $k$ is defined (Definition C.2) as one where each output $y_t$ depends only on the current input $x_t$ and the previous $k$ outputs: $y_t = \mu_t x_t + \ell_{t1} y_{t-1} + \cdots + \ell_{tk} y_{t-k}$. The theorem shows that any such transformation can be inverted to produce a $(k+1)$-semiseparable mask matrix. Conversely, any semiseparable mask enables an efficient autoregressive step: compute the new state and output in $O(N)$ time, independent of sequence length. This property is essential for autoregressive generation, distinguishing semiseparable SMA from other structured masks where generating the next token would require recomputation over the full history.

#### The Scalar-Identity SSM: Where the Duality Emerges

Now the paper brings the two threads together. Consider a structured SSM where the $A_t$ matrices have **scalar-identity structure**: $A_t = a_t \cdot I$ for scalar $a_t$ and identity matrix $I \in \mathbb{R}^{(N,N)}$. This is a restriction from the diagonal structure used in Mamba: instead of each state dimension having its own independent scalar $a_t^{(n)}$, all $N$ dimensions share the same scalar $a_t$.

Under this restriction, the matrix $M$ simplifies dramatically. Recall the general SSS form:

$$M_{ji} = C_j^\top A_{j:i}^\times B_i$$

With $A_t = a_t I$, the cumulative product becomes $A_{j:i}^\times = (a_{j:i}^\times) \cdot I$, a scalar times the identity. Therefore:

$$M_{ji} = a_{j:i}^\times \cdot (C_j^\top B_i)$$

**What happened algebraically:** the matrix $A$ matrices, being scalar multiples of identity, commute with everything and factor out as scalars. What remains is the inner product $C_j^\top B_i$, which measures the similarity between the output encoding at time $j$ and the input encoding at time $i$.

This can be vectorized across all $(j, i)$ pairs. Define $L \in \mathbb{R}^{(T,T)}$ as $L = \text{1SS}(a)$, the 1-semiseparable matrix generated by the scalars $a_t$:

$$L_{ji} = \begin{cases} a_{j:i}^\times & j \geq i \\ 0 & j < i \end{cases}$$

Define the kernel matrix $G = C B^\top \in \mathbb{R}^{(T,T)}$ where $G_{ji} = C_j^\top B_i$. Then:

$$M = L \circ G = L \circ (C B^\top)$$

**What this means:** the scalar-identity SSM's matrix transformation is exactly an elementwise product of a 1-semiseparable mask $L$ (determined purely by the $a_t$ scalars) and a kernel Gram matrix $C B^\top$ (determined purely by the $B_t$ and $C_t$ vectors). This is **identical in form** to masked kernel attention, with $C$ playing the role of queries, $B$ playing the role of keys, and the mask $L$ replacing both the softmax and the positional encoding of standard attention.

**The quadratic (naive) computation of this SSM** is exactly equation (16):

$$G = \text{contract}(\text{TN, SN} \to \text{TS})(C, B)$$
$$M = \text{contract}(\text{TS, TS} \to \text{TS})(G, L)$$
$$Y = \text{contract}(\text{TS, SP} \to \text{TP})(M, X)$$

This is the SMA quadratic mode with $C$ as queries and $B$ as keys. The $X$ input to the SSM corresponds to $V$ (values) in attention.

**Why this is the crucial junction point:** the scalar-identity restriction on $A$ is what makes the quadratic form look like attention. With diagonal but not scalar-identity $A$, the $C_j^\top A_{j:i}^\times B_i$ expression does not factor cleanly into a scalar times an inner product—the different state dimensions mix differently over time. The scalar-identity restriction is therefore a deliberate tradeoff: lose some expressivity in the dynamics (all dimensions decay/evolve at the same rate $a_t$) but gain the ability to compute the SSM using hardware-efficient attention-like matrix multiplications.

#### 1-Semiseparable SMA and the Complete Duality

The converse direction is equally important. Consider structured masked attention where the mask $L$ is **1-semiseparable**. In the SMA linear form (equation 15), the bottleneck step is $H = \text{contract}(\text{TS, SPN} \to \text{TPN})(L, Z)$, which for each $(p,n)$ channel is a multiplication by the 1-SS matrix $L$. But multiplication by a 1-SS matrix is exactly the **scalar SSM recurrence** (equation 7):

$$h_t^{(p,n)} = a_t \cdot h_{t-1}^{(p,n)} + z_t^{(p,n)}$$

This is precisely step 8b of the diagonal SSM computation (equation 8), but with the crucial simplification: **all $N$ feature dimensions share the same scalar $a_t$** rather than having $N$ different scalars. This is Corollary 5.1:

> "1-SS SMA is a special case of a diagonal SSM where the diagonal matrix is a scalar multiple of the identity."

**The full duality loop is now closed:**
- A scalar-identity SSM has a quadratic dual form that is identical to SMA with a 1-SS mask.
- 1-SS SMA has a linear dual form that is identical to a scalar SSM recurrence.
- Therefore, the **SSD model**—defined as either a scalar-identity SSM or equivalently as 1-semiseparable SMA—has both a linear recurrent form and a quadratic attention-like form, and these two forms compute **exactly the same function**.

**Figure 4** summarizes this with a Venn diagram: SSMs and SMA intersect at the SSD region. S4D, S5, and Mamba are diagonal SSMs (where $A$ is diagonal but not necessarily scalar-identity) that lie outside the SMA intersection. Linear attention and RetNet are SMA instances where the mask is not input-dependent (not selective) that lie outside the SSM intersection. SSD is the intersection: input-dependent (selective) 1-SS mask, scalar-identity SSM structure.

**The theoretical significance:** this duality is exact, not approximate. The linear and quadratic forms compute the same matrix transformation $M = L \circ (C B^\top)$ up to numerical precision. The choice between them is purely an **algorithmic decision** based on computational efficiency, not a modeling decision that affects the function being computed. This is fundamentally different from approaches like Performer or Reformer, which approximate the attention matrix and therefore change the model's behavior.

#### Theorem 5.2: Efficient Autoregressive Attention Must Be Semiseparable

The paper goes further and characterizes the **necessary** condition for efficient autoregression in the SMA framework. Theorem 5.2 states:

> "For any instantiation of structured masked attention that is an autoregressive process with bounded order, the structured mask $L$ must be a semiseparable matrix."

The proof (Appendix C.2) is elegant. An autoregressive process of order $k$ means $y_t = \mu_t x_t + \ell_{t1} y_{t-1} + \cdots + \ell_{tk} y_{t-k}$, which can be rearranged to $y_t - \ell_{t1} y_{t-1} - \cdots - \ell_{tk} y_{t-k} = \mu_t x_t$. In matrix form, this is $\tilde{L}^{-1} y = \text{diag}(\mu) x$ where $\tilde{L}^{-1}$ is a $(k+1)$-banded lower triangular matrix (only the diagonal and $k$ subdiagonals are non-zero). Therefore $y = \tilde{L} \cdot \text{diag}(\mu) x$, and $L = \tilde{L}$ is the INVERSE of a banded matrix. By closure properties of semiseparable matrices (Proposition C.1), the inverse of a $(k+1)$-banded matrix is $(k+1)$-semiseparable.

**What this means operationally:** if you want a masked attention model that can generate tokens one at a time in constant time per step (independent of sequence length), the mask $L$ **must** be semiseparable. Any other structured matrix—Toeplitz, Fourier, sparse—either requires recomputation over the full history at each generation step or must be converted to a semiseparable form. This is a strong theoretical justification for focusing on semiseparable SMA as the "right" class for autoregressive sequence models.

#### The SSD Layer: Formal Definition

At this point, the SSD layer can be defined precisely. It is the sequence transformation:

$$Y = \text{SSD}(A, B, C)(X)$$

where:
- $A \in \mathbb{R}^{(T)}$: a vector of scalar parameters, one per timestep (input-dependent in practice).
- $B \in \mathbb{R}^{(T, N)}$: the "expansion" matrix (analogous to keys in attention).
- $C \in \mathbb{R}^{(T, N)}$: the "contraction" matrix (analogous to queries in attention).
- $X \in \mathbb{R}^{(T, P)}$: the input sequence (analogous to values in attention).

The computation is defined via the matrix $M \in \mathbb{R}^{(T,T)}$:

$$M_{ji} = a_{j:i}^\times \cdot C_j^\top B_i$$

and $Y = M X$ (broadcasting over the $P$ dimension of $X$). Equivalently, in SMA form:

$$L = \text{1SS}(A) \quad \text{(mask matrix)}$$
$$G = C B^\top \quad \text{(kernel matrix)}$$
$$M = L \circ G$$
$$Y = M X$$

**The key computational insight:** this can be computed EITHER via the SSM recurrence (linear time, but uses many small scalar operations) OR via the quadratic form (matrix multiplications, but quadratic in $T$). Neither is optimal for all sequence lengths and hardware. The SSD algorithm (Section 6) provides a third way that combines the best of both.

#### The Block-Decomposition SSD Algorithm

The SSD algorithm is built on the rank-structured nature of the semiseparable matrix $M$. The central idea is to partition the $T \times T$ matrix into blocks of size $Q \times Q$, compute the diagonal blocks using the fast quadratic (attention) form, and handle the off-diagonal blocks using the compressed state recurrence.

**Step 0: Block Partitioning.** Divide the time indices $[0, T)$ into chunks of size $Q$, indexed by $c = 0, 1, \ldots, \lceil T/Q \rceil - 1$. Chunk $c$ covers indices $[cQ, (c+1)Q)$. The matrix $M$ is partitioned into blocks $M^{(c,c')}$ where the block at (chunk row $c$, chunk column $c'$) contains the interactions from inputs in chunk $c'$ to outputs in chunk $c$. Because $M$ is lower triangular, only blocks with $c \geq c'$ are non-zero.

**The structure of a diagonal block** $M^{(c,c)}$ (same chunk row and column): this block contains the interactions among positions within the same chunk. Since it is a principal submatrix of the semiseparable matrix, it is itself an $N$-semiseparable matrix, generated by the chunk's slice of the SSS parameters:

$$M^{(c,c)} = \text{SSS}(A_{cQ:(c+1)Q}, B_{cQ:(c+1)Q}, C_{cQ:(c+1)Q})$$

For the SSD model (scalar-identity $A$), this diagonal block can be computed using the quadratic form:

$$M^{(c,c)} = \text{1SS}(A_{cQ:(c+1)Q}) \circ (C_{cQ:(c+1)Q} B_{cQ:(c+1)Q}^\top)$$

This is a $Q \times Q$ matrix that can be computed with batched matrix multiplications.

**The structure of an off-diagonal block** $M^{(c,c')}$ with $c > c'$: this block has a low-rank factorization (the defining property of semiseparable matrices). The SSS representation yields the explicit factorization from equation (5):

$$M^{(c,c')} = \underbrace{\begin{bmatrix} C_{cQ}^\top A_{cQ:cQ}^\times \\ \vdots \\ C_{(c+1)Q-1}^\top A_{(c+1)Q-1:cQ}^\times \end{bmatrix}}_{\text{Left factor } L_c \in \mathbb{R}^{(Q, N)}} \cdot \underbrace{A_{cQ-1:(c'+1)Q-1}^\times}_{\text{Center factor } \in \mathbb{R}^{(N,N)}} \cdot \underbrace{\begin{bmatrix} A_{(c'+1)Q-1:c'Q}^\times B_{c'Q} \\ \vdots \\ A_{(c'+1)Q-1:(c'+1)Q-1}^\times B_{(c'+1)Q-1} \end{bmatrix}^{\top}}_{\text{Right factor}^\top R_{c'} \in \mathbb{R}^{(N, Q)}}$$

**What this factorization represents.** The right factor $R_{c'}^\top$ (of shape $N \times Q$) maps the $Q$ inputs in chunk $c'$ to an $N$-dimensional state at the right boundary of chunk $c'$—essentially, it computes the **final state contribution of chunk $c'$** assuming it started from state zero. The center factor $A_{cQ-1:(c'+1)Q-1}^\times$ (of shape $N \times N$) propagates this state from the end of chunk $c'$ to the beginning of chunk $c$—it is the cumulative product of all $A$ matrices between the chunks. The left factor $L_c$ (of shape $Q \times N$) maps the state at the beginning of chunk $c$ to the $Q$ outputs in chunk $c$—it computes the **output contribution of the initial state** assuming zero inputs within the chunk.

**The algorithm then proceeds in three stages, matching this factorization:**

**Stage 1: Compute intra-chunk outputs (diagonal blocks).** For each chunk $c$, compute the contribution of inputs within the chunk to outputs within the same chunk. This uses the quadratic form:

$$Y_{\text{diag}}^{(c)} = (\text{1SS}(A_{cQ:(c+1)Q}) \circ (C_{cQ:(c+1)Q} B_{cQ:(c+1)Q}^\top)) \cdot X_{cQ:(c+1)Q}$$

This involves a batched matrix multiply of size `BMM(T/Q, Q, P, N)` to compute the kernel matrix, an elementwise mask multiplication, and a batched matrix multiply of size `BMM(T/Q, Q, P, N)` to multiply by $X$. The chunk size $Q$ is chosen small enough that these $O(Q^2)$ operations are fast (dominated by matrix multiplication units). The interpretation: **what would the outputs be if each chunk started from state zero?**

**Stage 2: Compute chunk-final states (right factors).** For each chunk $c$, compute the hidden state at the end of the chunk, assuming it started from state zero. This is the right factor multiplied by the inputs:

$$S_c = \sum_{i=0}^{Q-1} A_{(c+1)Q-1:cQ+i}^\times \cdot B_{cQ+i} \cdot X_{cQ+i}$$

where $S_c \in \mathbb{R}^{(N, P)}$. This is computed as a batched matrix multiply of size `BMM(T/Q, N, P, Q)`. The interpretation: **compress each chunk's inputs into a state vector that captures all information needed for future chunks.**

**Stage 3: Propagate states between chunks (center factors).** The chunk-final states $S_0, S_1, \ldots, S_{\lceil T/Q \rceil -1}$ now need to be propagated forward. The true state at the end of chunk $c$ (accounting for all previous chunks) is:

$$\tilde{S}_c = \sum_{c'=0}^c A_{cQ-1:(c'+1)Q-1}^\times \cdot S_{c'}$$

This is a **1-SS matrix multiplication** on the state sequence: $\tilde{S}_{0:\lceil T/Q \rceil} = \text{1SS}(\tilde{A}) \cdot S_{0:\lceil T/Q \rceil}$, where $\tilde{a}_c = A_{cQ-1:(c-1)Q-1}^\times$ (the cumulative product of $A$ across one chunk gap). This step costs $O((T/Q) \cdot N \cdot P)$ FLOPs and uses the scalar SSM scan (either sequential or via parallel scan). The interpretation: **the compressed states evolve forward using the SSM recurrence, but with sequence length $T/Q$ instead of $T$—a factor of $Q$ shorter than the original recurrence.**

**Stage 4: Compute inter-chunk outputs (left factors).** For each chunk $c$, compute the output contribution from the initial state $\tilde{S}_{c-1}$ (the state arriving from all previous chunks):

$$Y_{\text{off}}^{(c)} = L_c \cdot \tilde{S}_{c-1} = \begin{bmatrix} C_{cQ}^\top \\ C_{cQ+1}^\top A_{cQ+1:cQ}^\times \\ \vdots \\ C_{(c+1)Q-1}^\top A_{(c+1)Q-1:cQ}^\times \end{bmatrix} \cdot \tilde{S}_{c-1}$$

This is a batched matrix multiply of size `BMM(T/Q, Q, P, N)`. The interpretation: **expand the incoming state back into per-position outputs, as if the chunk had no inputs of its own.**

**Stage 5: Combine.** The final output for chunk $c$ is $Y^{(c)} = Y_{\text{diag}}^{(c)} + Y_{\text{off}}^{(c)}$—the sum of the within-chunk contribution and the contribution from all previous chunks via the state. Linearity of the SSM ensures this decomposition is exact.

**Computational cost analysis.** Setting $N = P = Q$ (the state dimension, head dimension, and chunk size are equal), the major operations become:
- Two batched matrix multiplies for diagonal blocks: `BMM(T/N, N, N, N)` each → $O(T N^2)$ FLOPs.
- One batched matrix multiply for right factors: `BMM(T/N, N, N, N)` → $O(T N^2)$ FLOPs.
- One 1-SS scan on $N^2$ channels of length $T/N$: $O(T N)$ FLOPs.
- One batched matrix multiply for left factors: `BMM(T/N, N, N, N)` → $O(T N^2)$ FLOPs.

Total FLOPs: $O(T N^2)$. All the batched matrix multiplies are on matrices of shape $(N, N)$, which is ideal for tensor core utilization. The 1-SS scan is the only non-matmul operation, and its cost is lower-order ($O(TN)$ vs. $O(TN^2)$) and can be amortized.

**Comparison to alternatives.** Pure quadratic attention has cost $O(T^2 N)$—a factor of $T/N$ more expensive, corresponding to the fact that attention caches the entire history (state size grows with $T$) while SSD compresses it (fixed state size $N$). Pure linear SSM recurrence has the same $O(TN^2)$ FLOPs, but a naive implementation materializes expanded tensors of size $(T, P, N)$ and does not use matrix multiplications—SSD avoids the memory expansion and uses tensor cores.

**The block size $Q$ as a tunable parameter.** When $Q=1$, SSD reduces to the pure recurrent form (one timestep per chunk). When $Q=T$, SSD reduces to the pure quadratic form (one chunk containing the entire sequence). The optimal $Q$ balances: larger $Q$ makes the matrix multiplies more efficient (larger matrices better utilize tensor cores) but makes the 1-SS scan longer and increases the $Q^2$ cost of the diagonal blocks. The paper chooses $Q$ equal to the head dimension $P$ (typically 64 or 128) as a sweet spot.

#### The Mamba-2 Architecture

The SSD layer is embedded in a neural network block that refines the Mamba block design using insights from the Transformer-connection framework.

**Parallel vs. sequential projections (Section 7.1, Figure 6).** The original Mamba block generates the SSM parameters $(A, B, C)$ from the SSM input $x$, creating a sequential dependency: first project the input to get $x$, then project $x$ to get the parameters. In Mamba-2, all projections are done in parallel at the beginning of the block. The input $u \in \mathbb{R}^{L \times d}$ is simultaneously projected to produce $x$, $z$, $B$, $C$, and $A$ (the latter is produced via a small network or linear projection with activation). This mirrors how Transformer blocks project $Q, K, V$ in parallel, and critically **reduces the number of synchronization points for tensor parallelism** (Section 8.1): Mamba's sequential design requires an all-reduce between the $x$ projection and the parameter projections, while Mamba-2's parallel design needs only one all-reduce at the output.

**Gating.** As in Mamba, the output of the SSM is gated: $y_g = y \cdot \phi(z)$ where $\phi$ is an activation (Swish by default) and $z$ is a parallel projection of the input. This is the "gated MLP" branch that originated in the gated attention unit and was adopted by Mamba.

**Extra normalization.** An additional normalization layer (GroupNorm by default) is inserted after the gating and before the final output projection. This follows NormFormer (Shleifer, Weston, and Ott 2021) and was found to stabilize training at larger scales. The number of groups in GroupNorm is chosen to be divisible by the tensor parallelism degree to avoid cross-GPU communication within the block (Section 8.1).

**Head patterns (Section 7.2).** The SSD layer (and SSMs more generally) processes input $X \in \mathbb{R}^{(T, P)}$ independently along the $P$ dimension—each of the $P$ channels gets its own sequence transformation. The paper develops a vocabulary for how the $B$ and $C$ parameters are shared across these channels, drawing explicit analogies to attention head patterns:

- **Multi-head SSM (MHS) / Multi-head attention (MHA):** $B$ and $C$ each have $H$ separate heads (where $H = \text{n\_heads}$), independent per channel. Total parameters: $H$ copies of $B$ and $C$, each of state size $N$.

- **Multi-contract SSM (MCS) / Multi-query attention (MQA):** $B$ is shared across all $H$ heads (1 head), while $C$ has $H$ heads. Analogous to sharing $K$ but not $Q$ in attention. The "contract" refers to $C$ contracting the state dimension.

- **Multi-input SSM (MIS) / Multi-value attention (MVA):** $B$ and $C$ are each single-headed (shared across all $P$ channels), but different heads have different $A$ parameters (different decay rates). This is the pattern used in Mamba: all input channels share the same input/output projections but can have different selective dynamics.

- **Multi-state SSM (MSS):** $A$, $B$, $C$ are all single-headed—one copy shared across all channels. Maximally parameter-efficient, but the $H$ copies of $A$ in other patterns allow different channels to have different selectivity.

**Why Mamba's MIS/MVA pattern is not the obvious choice from the attention perspective.** In Transformers, MQA (sharing $K$ and $V$) was motivated by reducing the KV cache for inference. In SSMs, $B$ and $C$ are not cached (the hidden state $h$ is the cache), so the motivation is different. Proposition 7.2 characterizes Mamba: it uses head dimension $P=1$ (every channel has independent dynamics) and MIS head structure ($B$ and $C$ shared across channels). In Mamba-2, the head dimension is increased to $P=64$ or $P=128$ (matching Transformer conventions), and the MIS pattern is preserved. Ablations (Section 9.4.2, Table 5) confirm that MIS/MVA outperforms MCS/MQA and MES/MKA at matched parameter counts, an empirical finding that the framework made possible to systematically test.

**Grouped-value attention (GVA).** Between single-head sharing and full multi-head independence, the paper introduces grouped-input SSM (GIS) or grouped-value attention (GVA): $B$ and $C$ have $G$ independent groups (where $1 < G < H$ and $G$ divides $H$). This is analogous to grouped-query attention (Ainslie et al. 2023) and is motivated by tensor parallelism: setting $G$ to be a multiple of the number of TP shards avoids additional communication.

**Kernel feature maps (Section 7.3).** The SSD dual form reveals that $C$ and $B$ play the role of queries and keys in a kernel attention computation. The paper optionally applies a kernel feature map $\psi$ to both: $\tilde{B}_t = \psi(B_t)$, $\tilde{C}_t = \psi(C_t)$. By default, $\psi$ is the Swish/SiLU activation. The paper ablates alternatives from the linear attention literature (cosFormer, Random Feature Attention, Performer's positive random features, Taylor expansion features from Based and ReBased) in Tables 6-7, finding that simple pointwise activations perform best—the negative result is informative: techniques designed to approximate softmax attention may be unnecessary when the model already includes the 1-SS mask $L$, which provides a different form of regularization. A normalization (denominator) term can be incorporated by augmenting $X$ with an extra channel of 1's, computing the sum of the attention weights, and normalizing—but this introduced instabilities except for the ReLU activation.

## 4. Key Insights and Innovations

### Innovation 1: The Semiseparable Matrix as a Unifying Computational Abstraction

The paper's deepest conceptual move is not the duality itself—it is identifying **semiseparable matrices** as the fundamental object that both SSMs and efficient attention are computing. Before this work, the field understood sequence models through their computational mechanisms: attention as pairwise token interactions with a softmax, SSMs as linear recurrences inspired by continuous-time dynamical systems. These mechanisms looked so different that the research communities developed entirely separate vocabularies, optimization strategies, and theoretical tools.

By reframing both as matrix transformations—specifically, as multiplication by a semiseparable matrix—the paper makes a move analogous to what category theory does in mathematics: it identifies a common abstract structure underlying superficially different objects, then uses the properties of that structure to prove theorems and design algorithms that transfer between domains. The "what is computed" (the matrix transformation) is separated from "how it is computed" (recurrence vs. attention), exactly the distinction between function and algorithm that is central to computer science but had been missing from sequence model discourse.

**What makes this distinctive.** The semiseparable characterization is **exact**, not approximate. This is not another paper showing that SSMs can approximate attention or vice versa under asymptotic limits. Theorem 3.5 states an identity: the SSM operator IS the SSS matrix constructor. The state size N IS the semiseparable order. There is no approximation error, no asymptotic regime, no "in the limit of large N" qualifier. This exactness matters because it means the theoretical machinery of semiseparable matrices—their closure properties under addition, multiplication, and inversion (Proposition C.1), their multiple compressed representations, their fast structured multiplication algorithms—transfers to SSMs without any "up to approximation" caveats.

**Comparison to prior work.** The dominant way of understanding SSMs came from their continuous-time origins (Gu, Goel, and Ré 2022) or signal processing connections (Gu, Gupta, et al. 2022). The dominant way of understanding efficient attention came from kernel methods (Katharopoulos et al. 2020) or low-rank matrix approximation (Choromanski et al. 2021; Sinong Wang et al. 2020). Neither lineage had connected to the classical numerical linear algebra of rank-structured matrices. The paper's citation to Pernet and Storjohann (2018) and the broader semiseparable matrix literature (Vandebril et al. 2005) imports a mature mathematical toolchain that had never been applied to deep learning sequence models. The significance is not that semiseparable matrices are new—they date back decades—but that someone recognized they were the right abstraction sitting unnoticed at the intersection of two active research areas.

**Fundamental vs. incremental.** This is a fundamental reframing. It does not propose a new SSM variant or attention approximation; it reorganizes the conceptual landscape so that both become instances of a single framework. The downstream consequences—the SSD algorithm, the Mamba-2 architecture, the tensor parallelism design—are demonstrations that the reframing pays operational dividends, not the core contribution itself. The paper could have stopped after Section 5 (the duality proof) and still constituted a significant theoretical advance; the remainder is validation that the abstraction is productive.

**Evidence anchor.** The theoretical edifice rests on Theorem 3.5 (SSM = SSS) and Corollary 5.1 (1-SS SMA = scalar-identity SSM). The practical payoff is measured in Figure 10 (2–8× speedup over Mamba scan, competitive with FlashAttention-2 beyond sequence length 2K) and Figure 9 (Pareto dominance over Transformer++ and Mamba in scaling laws). But the innovation is the framework, not the metric.

---

### Innovation 2: The Contraction-Order Duality as a Design Principle

The paper reinterprets masked attention not as a three-step matrix multiplication pipeline but as a single 4-way tensor contraction, revealing that the choice between "quadratic attention" and "linear attention" is simply a choice of **contraction order**. This is a specific, elegant formalization of a notion that had been floating in the literature implicitly—Katharopoulos et al. (2020) observed that $(QK^\top)V = Q(K^\top V)$ by associativity, but the mask complicated the picture. Previous works like RetNet (Y. Sun et al. 2023) and TransNormer (Qin, Dong Li, et al. 2023) reproduced the linear attention formula for specific non-standard masks by algebraic manipulation, but each was derived as a special case.

The contraction notation `contract(TN, SN, SP, TS → TP)(Q, K, V, L)` makes explicit that there are **four inputs** (Q, K, V, L) being combined, and that the output is the result of summing over the S and N dimensions. The order of pairwise contractions determines complexity: contracting T,N with S,N first costs O(TSN), contracting S,P with S,N first costs O(SPN). When S = T and N = P, these are O(T²N) vs. O(TN²). The mask L is just another tensor in the contraction—it doesn't change the function, only the algorithm by which we compute it.

**What makes this distinctive.** The "pairwise contraction ordering" language abstracts away all the mechanism-specific details of attention and speaks directly to computational complexity. It reveals that the key question is not "what mask should we use?" but rather **"what masks admit an efficient contraction order?"** The answer—structured matrices—is Definition 4.2 (Structured Masked Attention), which generalizes beyond any specific mask choice. This unifies linear attention (causal mask), RetNet (decay mask), and SSD (1-semiseparable mask) into a single framework parameterized by the choice of L, with the linear/quadratic duality holding for any structured L.

**Comparison to prior work.** Prior to this paper, the relationship between attention and recurrence was framed through specific algebraic tricks: "drop the softmax, use a kernel, and apply associativity." This paper's contraction framing shows that the associativity argument is just one instance of a much broader principle—changing contraction orders in a tensor network changes computational complexity. The connection to tensor network theory (common in physics and quantum computing but rare in deep learning architecture design) is implicit but powerful.

**Why this is more than a notation change.** The contraction framing immediately generates new research questions: what other structured matrices L admit efficient multiplication? What other contraction orders are possible beyond the two considered? Could there be orders that are partially quadratic and partially linear, yielding a continuum of algorithms between the two extremes? The SSD algorithm (Section 6) is exactly one answer to this last question—a hybrid contraction order implemented through block decomposition. Without the contraction abstraction, it would be much harder to even formulate the idea of a hybrid algorithm, let alone analyze its complexity.

**Evidence anchor.** Proposition 4.1 states that autoregressive kernel attention with a causal mask can be computed in O(T) time—a known result that the paper re-derives transparently in three lines of contraction notation (equation 15). Definition 4.2 and the SMA framework generalize this transparent derivation to any structured L. The conceptual leap from "here is a specific formula that works for one mask" to "here is the general condition on L that makes this work" is the innovation.

---

### Innovation 3: Verifier-Free Scalability Through Exact Duality (A Negative Result with Positive Implications)

One of the paper's most understated but consequential insights is a **negative result**: Theorem 5.2 proves that efficient autoregressive attention **must** use a semiseparable mask L. Any attempt to design an autoregressive attention variant with a different structured mask—Toeplitz, Fourier, sparse random—either cannot generate tokens one at a time in constant time per step, or can be reduced to a semiseparable form.

This is significant because it **narrows the search space** for future research. The efficient Transformer literature (surveyed in Tay et al. 2022) contains dozens of proposed attention approximations with different induced matrix structures. Theorem 5.2 says: if you want autoregressive generation with constant per-step cost, your mask matrix L had better be semiseparable. This is not an empirical finding that happens to hold on current benchmarks—it is a mathematical necessity that follows from the definition of autoregressive processes (Definition C.2).

**What makes this distinctive.** The paper does not just show that SSD is _a_ good model; it proves that semiseparable structure is _the_ structure for autoregressive masked attention. This is the kind of "no-free-lunch" result that provides genuine theoretical guidance: it tells researchers what not to try, which is often more valuable than suggesting what to try. The proof (Appendix C.2) is elegant—an autoregressive process of order k has an inverse that is (k+1)-banded; banded matrices are semiseparable; the inverse of a semiseparable matrix is semiseparable (Proposition C.1); therefore the mask must be semiseparable. It leverages the closure properties of semiseparable matrices that were established as part of the theoretical framework but are applied here to prove a constraint rather than enable a capability.

**Why this matters beyond SSD.** This result provides theoretical justification for why linear attention (Katharopoulos et al. 2020) and RetNet (Y. Sun et al. 2023) worked—their mask matrices L happen to be semiseparable (the causal mask is 1-SS with a_t = 1; the decay mask is 1-SS with a_t = γ). It also explains why more exotic proposals like Fourier attention or random sparse attention struggle with autoregressive generation: their masks are not semiseparable, so they cannot maintain a constant-size recurrent state. The result thus **retroactively organizes** the prior literature into "things that could have worked for autoregressive generation" and "things that were structurally unsuited from the start."

**Fundamental vs. incremental.** This is a fundamental theoretical contribution. It does not improve any metric; it constrains the design space. Such results are rare in deep learning and valuable precisely because they prevent wasted effort on structural dead ends.

**Evidence anchor.** The proof is in Theorem C.3 (Appendix C.2), which formalizes Theorem 5.2. The empirical consequence is indirect but powerful: Mamba-2's strong performance on autoregressive language modeling (Table 1, Figure 9) is not serendipitous but structurally necessary—it occupies the intersection of "models that can be computed efficiently" and "models that have efficient autoregressive generation." The SSD framework identifies this intersection explicitly.

---

### Innovation 4: The Hardware-Aware Algorithm as an Instance of Structured Matrix Decomposition

The SSD algorithm is typically presented as engineering—making SSMs fast on GPUs. But the paper reveals a deeper insight: **the algorithm is not an ad hoc optimization but a principled block decomposition of the semiseparable matrix**, and the choice of which computation to use for which blocks is a **first-class design parameter** (the block size Q) that navigates a continuum between fully quadratic and fully linear computation.

This is conceptually distinct from the "chunkwise" algorithms proposed in concurrent work (Y. Sun et al. 2023; Yang et al. 2024), which were presented as practical tricks to trade off between parallel and sequential computation. The paper frames chunking not as a heuristic but as an instance of a general algorithm design pattern for semiseparable matrices: partition into blocks, use different structured matrix multiplication algorithms for different blocks based on their size and structure, and combine. The diagonal blocks are small enough to use the quadratic form (attention-like matmuls); the off-diagonal blocks are factored through their low-rank structure (state passing). The block boundary Q is the algorithmic degree of freedom that controls the quadratic-linear tradeoff.

**What makes this distinctive.** The paper provides Appendix B, a compendium of alternative algorithms for computing 1-SS matrix multiplication, each corresponding to a different matrix decomposition (dilated, state-passing, block decomposition, associative scan). The SSD algorithm is simply the state-passing decomposition applied to the full N-SS matrix, with the quadratic form used within chunks. This catalog makes explicit that there are many possible algorithms for the same matrix operation, each with different parallelization, memory, and hardware compatibility profiles. The "right" algorithm depends on the specific hardware and sequence length regime—the paper is not claiming SSD is universally optimal but rather that it occupies a sweet spot for current GPU architectures.

**Comparison to prior work.** Mamba's selective scan is a specific implementation of the associative scan algorithm for 1-SS matrices. FlashAttention is a specific implementation of the quadratic form optimized for GPU memory hierarchy. SSD combines both: the quadratic form within chunks (leveraging FlashAttention-like optimizations for small matrices) and the state-passing between chunks (leveraging the linear scaling of SSM recurrences). The innovation is not either component individually but the **systematic engineering of the boundary between them** guided by the structured matrix perspective.

**Why this is more than an implementation detail.** The block decomposition approach scales with problem parameters in a way that pure quadratic or pure linear approaches do not. The total FLOPs are O(TN²), matching the lower bound for an SSM with state size N and head dimension N. The memory is O(TN), matching the size of the input and output. All major operations are batched matrix multiplications on matrices of shape (N, N), ideal for tensor cores. The only non-matmul operation (the 1-SS scan) has lower-order cost O(TN²/Q) and can use highly optimized scan implementations or even be implemented as a matrix multiply since the sequence length T/Q is small. The asymptotic analysis (Theorem 6.1) guarantees that this algorithm architecture will remain efficient as N and T scale.

**Evidence anchor.** Figure 10: SSD is 2–8× faster than Mamba's fused scan at N=64, crosses over with FlashAttention-2 at sequence length 2K, and is 6× faster at sequence length 16K. The right panel of Figure 10 shows that increasing state dimension from N=16 to N=256 causes minimal slowdown for SSD while the Mamba scan slows down linearly. Listing 1 provides a complete implementation in a few lines of PyTorch, demonstrating that the algorithmic complexity is manageable.

---

### Innovation 5: Multi-Value Attention as the Natural Head Pattern for SSMs

The paper uses its attention-SSM duality to systematically analyze head patterns for sequence transformations—an architectural axis that had been explored extensively for Transformers (Shazeer 2019; Ainslie et al. 2023) but never formalized for SSMs. The result is a **design space taxonomy** that both characterizes existing models and enables targeted ablations.

The core finding (Proposition 7.2 and Table 5) is that the head pattern used in Mamba—which the paper names **multi-input SSM (MIS)** or equivalently **multi-value attention (MVA)**—is not an arbitrary choice but the pattern that empirically performs best for SSMs, outperforming the multi-query (MQA) and multi-head (MHA) patterns that are standard in Transformers. This is a case where transferring design patterns directly from Transformers to SSMs would be a mistake: the optimal head structure is different because the computational roles of B, C, and X are different from those of K, Q, and V.

**What makes this distinctive.** The paper does not treat "heads" as a fixed Transformer convention but as a formal concept (Definition 7.1) parameterized by which tensor dimensions are shared across heads. This yields a taxonomy: multi-head (all independent), multi-query (K/V shared), multi-key (Q/V shared), multi-value (Q/K shared), and grouped variants. The taxonomy applies to **any** sequence transformation, not just attention. The paper then uses this taxonomy to characterize Mamba precisely (MIS/MVA pattern with P=1) and to design Mamba-2 (MIS/MVA pattern with P=64 or 128, and grouped variants for tensor parallelism).

**Comparison to prior work.** Mamba's original paper did not discuss head patterns as a design dimension—it used P=1 by default and shared B and C across channels because that was the natural SSM-centric view (B and C project into and out of the state space, and it makes sense to share them across input channels). From the attention-centric view, sharing B and C across channels is highly unusual (it would be like sharing Q and K across all value heads). The paper's contribution is recognizing that this "unusual" choice is actually optimal for SSMs and providing the vocabulary to articulate why. The ablation in Table 5 confirms: MIS/MVA substantially outperforms MCS/MQA and MES/MKA at matched parameter counts (perplexity 11.66 vs. 12.62 vs. 12.59 for 125M models; 8.73 vs. 9.33 vs. 9.36 for 360M models).

**Fundamental vs. incremental.** The taxonomy itself is an incremental formalization of well-known ideas. The empirical finding that MVA outperforms MQA for SSMs is a moderate contribution—important for practitioners but not conceptually transformative. However, the **methodological contribution** of using the duality to systematically ablate design choices that would not have been obvious from either the SSM or attention perspective alone is significant: it demonstrates the operational value of the unified framework. The paper is essentially arguing "because we now have a shared language, we can ask and answer questions that neither the SSM nor the attention community would have thought to ask."

**Evidence anchor.** Table 5 (125M and 360M scales), Table 4 (block design ablations for parallel vs. sequential projections and extra normalization), and Tables 6-7 (kernel activation ablations). The negative results in Tables 6-7—where sophisticated kernel approximations from the linear attention literature do not improve over simple Swish activations—are themselves informative: they suggest that the 1-SS mask L already provides the regularization that kernel approximations were designed to supply, a hypothesis that would be difficult to formulate without the SSD framework.

## 5. Experimental Analysis

### Evaluation Methodology

- **Dataset.** All experiments use the **Pile** dataset (L. Gao, Biderman, et al. 2020), an 800GB corpus of diverse text, for pretraining. Scaling law experiments use the GPT2 tokenizer; downstream evaluations use the GPT-NeoX tokenizer (Black et al. 2022). Zero-shot evaluations cover LAMBADA (Paperno et al. 2016), HellaSwag (Zellers et al. 2019), PIQA (Bisk et al. 2020), ARC-challenge and ARC-easy (P. Clark et al. 2018), WinoGrande (Sakaguchi et al. 2021), and OpenBookQA (Mihaylov et al. 2018). The synthetic associative recall experiments use the multi-query associative recall (MQAR) task from Arora, Eyuboglu, Zhang, et al. (2024), with a harder variant replacing non-query/key/value tokens with random tokens, using more key-value pairs and longer sequences to stress-test memory capacity.

- **Base model(s).** Scaling law experiments use models from approximately 125M to 1.3B parameters following GPT3 specifications (Brown et al. 2020), with depths and widths specified in Table 9. Full training downstream evaluations use models up to 2.7B parameters trained on 300B tokens. The baseline comparisons include Pythia (Biderman et al. 2023), which was trained with the same tokenizer, dataset, and training length. For the MQAR synthetic task, models with dimension D ∈ {32, 64, 128, 256} and 2 layers are used, which is deliberately smaller than typical language model scales to test fundamental recall capability rather than large-scale learned behaviors.

- **Metrics.** Pretraining quality is measured by **validation perplexity** on the Pile's held-out validation split. Downstream evaluation uses **accuracy** (exact match) for LAMBADA, HellaSwag, PIQA, ARC-easy, ARC-challenge, WinoGrande, and OpenBookQA, following the standard LM evaluation harness protocol (L. Gao, Tow, et al. 2021). For MQAR, accuracy is measured as the fraction of test examples where the model correctly recalls the value associated with a queried key. Training efficiency is measured in **wall-clock time** (milliseconds per forward pass) on A100 80GB PCIe GPUs, benchmarked against FlashAttention-2 (Dao 2024) and Mamba's optimized selective scan implementation. Scaling laws report perplexity against **theoretical FLOPs** (total floating point operations) following the Chinchilla protocol (Hoffmann et al. 2022).

- **Baselines.** The paper compares against:
    - **Transformer++:** a strong Transformer baseline with rotary embeddings, SwiGLU MLP, RMSNorm (not LayerNorm), no linear bias, and higher learning rates, following the recipe from the Mamba paper (Gu and Dao 2023). This recipe was itself shown to outperform the standard GPT3 architecture.
    - **Mamba-1** (Gu and Dao 2023): the original selective SSM architecture with diagonal structured $A_t$ and hardware-aware selective scan implementation, at matched model sizes.
    - **Pythia** (Biderman et al. 2023): open-source Transformer models trained with the same tokenizer (GPT-NeoX), dataset (the Pile), and training length (300B tokens) as Mamba-2, providing the most controlled pretraining comparison. Pythia-160M through Pythia-6.9B are compared at corresponding model sizes.
    - **Additional open-source models** in Table 1: Hybrid H3 (Dao, D. Y. Fu, et al. 2023), OPT (Zhang et al. 2022, cited indirectly), RWKV4 (B. Peng, Alcaide, et al. 2023), and GPT-Neo (Black et al. 2022) at comparable sizes.
    - For MQAR: standard multi-head softmax attention and the Based architecture (Arora, Eyuboglu, Zhang, et al. 2024), which combines convolutions, local attention, and a linear attention variant.

- **Generation budget / compute accounting.** For scaling law experiments (Figure 9), compute is measured in **theoretical FLOPs** following the Chinchilla formula (Hoffmann et al. 2022), with total FLOPs = 6 × parameters × training tokens. The models are trained at varying compute budgets from approximately 10^19 to 10^20 FLOPs for the 125M–1.3B range, with the training tokens per model size specified in Table 9 to roughly match the Chinchilla-optimal ratio (tokens proportional to parameters). For speed benchmarks (Figure 10), compute is measured in **wall-clock time** (milliseconds) for a forward pass at specified sequence lengths and state dimensions, with all methods run on the same A100 80GB PCIe GPU hardware. SSD, Mamba scan, convolution, and FlashAttention-2 timings are reported as single forward pass latencies. For downstream evaluations (Table 1), all models were trained for 300B tokens (a fixed data budget independent of model size, which is below Chinchilla-optimal for the largest models), allowing comparison against Pythia which used the identical training length.

- **Cross-validation / statistical protocol.** No cross-validation or statistical significance testing is reported for the main language modeling results—perplexity and downstream accuracy are reported as single-run values on the fixed validation splits. For the MQAR synthetic task, each configuration is swept over model dimensions and learning rates, with the best configuration selected. Model sizes and training hyperparameters are specified in Appendix D (Tables 9 and associated text). The training recipe uses AdamW optimizer with gradient clipping at 1.0, weight decay 0.1, no dropout, linear warmup with cosine decay, and the "improved recipe" modifications (RMSNorm, no linear bias, β = (0.9, 0.95)). The lack of error bars or multiple seeds is a standard limitation of large-scale language model experiments where computational constraints typically prohibit statistical replication.

### Main Quantitative Results

#### Scaling Laws: Mamba-2 Pareto Dominates Mamba and Transformer++

The headline result of the Chinchilla scaling law experiments (Figure 9) is that **Mamba-2 Pareto-dominates both Mamba-1 and the Transformer++ baseline** across the measured compute range of approximately 10^19 to 10^20 FLOPs. "Pareto-dominates" here means that at every measured compute budget, Mamba-2 achieves lower perplexity than both competing architectures, with the gap widening at higher compute budgets.

The figure shows log-scale perplexity vs. log-scale FLOPs for models from approximately 125M to 1.3B parameters, trained on the Pile with sequence length 8192. All three architectures (Transformer++, Mamba, Mamba-2) show the expected power-law improvement in perplexity with increased compute. However, Mamba-2's curve is consistently below both alternatives. At the highest measured budget (approximately 10^20 FLOPs, corresponding to the 1.3B parameter models), the gap between Mamba-2 and Transformer++ is visually estimated at roughly 0.05–0.10 nat in log perplexity (a meaningful gap in this regime where total log-perplexity values are in the range of 6–7 × 10^0).

The paper does not report exact perplexity numbers in the main text for the scaling law experiments, presenting results only through the Figure 9 log-scale plot. However, the downstream evaluation results (Table 1) provide concrete numbers at the fixed 300B token training budget for each model size. For instance, at the 1.3B–1.4B scale (300B tokens):

> "Mamba-2-1.3B: Pile ppl 6.66, Mamba-1.4B: Pile ppl 6.80" (Table 1)

The Mamba-2 model achieves perplexity 6.66 vs. Mamba's 6.80 despite having slightly fewer parameters (1.3B vs. 1.4B). This is consistent with the scaling law trend showing Mamba-2's advantage emerging at larger model sizes.

**Why Pareto dominance matters more than a single-point comparison.** The paper emphasizes that Mamba-2 is Pareto-optimal not just on model quality (perplexity) but also on **wall-clock time**:

> "Compared to our Transformer baseline, Mamba-2 is Pareto dominant on performance (perplexity), theoretical FLOPs, and actual wall-clock time." (Figure 9 caption)

This claim in the caption goes beyond what Figure 9 directly visualizes (which shows only perplexity vs. theoretical FLOPs). The wall-clock time comparison is supported by the separate speed benchmarks (Figure 10, discussed below) and the architecture analysis in Section 9.2.3 showing that hybrid Mamba-2-MLP models can match pure Mamba-2 quality while being faster to train at short sequence lengths due to the hardware efficiency of MLP layers. The "Pareto dominance on wall-clock time" claim thus requires synthesizing Figure 9 with Figure 10 and the Section 9.2.3 hybrid results—Figure 9 alone shows only the perplexity-FLOPs relationship.

#### Zero-Shot Downstream Evaluations: Competitive with Transformers at Matched Size

Table 1 reports zero-shot evaluation results for models trained on 300B tokens on the Pile using the GPT-NeoX tokenizer. The main findings are:

**At every model size, Mamba-2 matches or outperforms Mamba-1 on average accuracy.** For the 780M/790M comparison: Mamba-2 achieves 53.5% average vs. Mamba's 53.0% (a small but consistent edge). At 1.3B/1.4B: both achieve 56.4% average, suggesting the architectures converge at this scale under fixed token budgets. However, the paper emphasizes that Mamba-2 achieves this parity with a faster and more hardware-efficient implementation—the quality is matched while training speed is substantially improved (per Figure 10).

**Mamba-2 at size X roughly matches Pythia at size 2X.** The paper states:

> "For each model size, Mamba-2 outperforms Mamba, and generally matches Pythia at twice the model size." (Table 1 caption)

Concretely:
- Mamba-2-780M (53.5% average) outperforms Pythia-1B (49.0%) and is competitive with Pythia-1.4B (51.7%).
- Mamba-2-1.3B (56.4%) substantially outperforms Pythia-1.4B (51.7%) and approaches Pythia-2.8B (55.7%).
- Mamba-2-2.7B (60.2%) outperforms Pythia-2.8B (55.7%) and even Pythia-6.9B (58.3%).

The most striking head-to-head is the 2.7B comparison: Mamba-2-2.7B at 60.2% average accuracy exceeds Pythia-6.9B at 58.3%, despite having 2.5× fewer parameters. This is the empirical evidence for the paper's claim that Mamba-2 provides better quality per parameter than strong Transformer baselines. However, it is important to note that the 2.7B Mamba-2 model was trained for only 300B tokens, which is substantially below the Chinchilla-optimal token count for a model of that size—the Chinchilla laws suggest roughly 50B tokens per 1B parameters, implying a 2.7B model should be trained on ~135B tokens, which is far less than 300B. This means the comparison is at a fixed (and generous for the smaller models) data budget, potentially favoring architectures that are more sample-efficient at this training horizon.

**Task-specific patterns reveal Mamba-2's strengths.** On WinoGrande (commonsense reasoning), Mamba-2 shows a pronounced advantage: Mamba-2-780M achieves 60.2% vs. Mamba-790M's 56.1% and Pythia-1.4B's 57.2%—a 4-point improvement over Mamba at matched size. On OpenBookQA (knowledge-intensive reasoning), Mamba-2-780M achieves 36.2% vs. Mamba's 34.2% and Pythia-1.4B's 30.8%. On PIQA (physical commonsense), performance is similar across architectures at matched sizes, suggesting this task saturates with model scale and is less sensitive to architectural differences. The pattern suggests Mamba-2's advantages are most pronounced on tasks requiring the integration of context with world knowledge (WinoGrande, OpenBookQA) rather than pure pattern matching.

#### Speed Benchmarks: SSD is 2–8× Faster Than Mamba Scan

Figure 10 reports wall-clock forward pass times on an A100 80GB PCIe GPU for the core sequence mixing operation (SSD, Mamba scan, convolution, and FlashAttention-2), not the full model including MLP layers.

**Left panel (varying sequence length, fixed N=64):**
- SSD is 2–8× faster than Mamba's fused scan across the range of sequence lengths from 512 to 512K tokens. The exact speedup varies because Mamba's scan has different scaling properties from SSD's block-decomposed matmuls.
- At sequence length 2K, SSD crosses over with FlashAttention-2—for shorter sequences, attention is faster (due to small T making the O(T²) cost manageable and attention's highly optimized matmul kernels), while for longer sequences, SSD's linear scaling dominates.
- At sequence length 16K, SSD is approximately 6× faster than FlashAttention-2 (estimated from the log-scale figure: FlashAttention-2 is at roughly 100ms while SSD is at roughly 16ms).
- Convolution (representing time-invariant SSM training) is competitive at short lengths but scales worse than SSD at long lengths.

**Right panel (varying state dimension N, fixed sequence length 4K):**
- FlashAttention-2 is constant at approximately 30ms regardless of N (since attention's state is the KV cache, not a compressed state dimension).
- Mamba's optimized scan scales **linearly** with N: approximately 1ms at N=16, 2ms at N=32, 4ms at N=64, 8ms at N=128. This linear scaling is the core bottleneck that prevented Mamba from using large state sizes.
- SSD scales much more gracefully: approximately 2ms at N=16, 2.5ms at N=32, 3ms at N=64, 3.5ms at N=128, 5ms at N=256. The sublinear scaling comes from the dominance of matrix multiplications, whose cost grows as O(N²) in the matmul dimension but benefits from tensor core parallelism that makes the constant factor very small for the N range tested.
- At N=256, SSD is approximately 3× faster than the extrapolated Mamba scan time (roughly 5ms vs. 16ms if the scan scaled linearly to 256).

**What these numbers mean architecturally.** The key consequence is that Mamba-2 can use **larger state dimensions (N=64, N=128, N=256) with minimal slowdown** compared to Mamba's typical N=16, while Mamba's scan cost would be prohibitive at those sizes. Since state dimension N controls information capacity—how much the model can memorize in its compressed state—this directly enables better performance on memory-intensive tasks. The MQAR results (Figure 8) validate this: Mamba-2 with N=256 dramatically outperforms N=16 on associative recall tasks that require holding many key-value pairs in memory.

#### Synthetic Associative Recall: Mamba-2 Surpasses Attention on Hard Memory Tasks

The MQAR experiments (Figure 8) test a fundamental capability that has been challenging for recurrent models: memorizing multiple key-value associations and correctly recalling the value when queried with a previously seen key. The task configuration is deliberately hard—tokens that are not query/key/value pairs are replaced with random tokens (increasing distractor noise), more key-value pairs are used (T/4 pairs for sequence length T), and models are small (2 layers, dimension D ∈ {32, 64, 128, 256}).

The results in Figure 8 show:

- **Mamba-1 (N=16) performs poorly** across all sequence lengths and model dimensions, often near 0% accuracy at the hardest settings (sequence length 1024, small model dimensions). This replicates the known weakness of SSMs on associative recall identified by prior work (De et al. 2024; Jelassi et al. 2024).

- **Mamba-2 (N=16)—with the same state size as Mamba-1—substantially outperforms Mamba-1.** This is striking because the state size is controlled: the improvement does not come from increased memory capacity. The paper notes:

> "Surprisingly, it is significantly better than Mamba-1 even when the state sizes are controlled (N = 16). (We are not sure which aspect of the architecture is the predominant factor, which remains a question to explore in future work.)"

This is an important open question. Possible explanations include: the scalar-identity structure on $A$ (all state dimensions sharing the same decay) might provide better learning dynamics for this task; the parallel projections in the Mamba-2 block design might improve optimization; or the larger head dimension P=64 (vs. Mamba's P=1) might help the model learn better key-value representations through the $B$ and $C$ projections. The paper does not ablate these to isolate the causal factor.

- **Mamba-2 with increased state size (N=64, N=256) achieves strong performance, surpassing vanilla attention at many settings.** For example, at sequence length 1024 and model dimension 256, Mamba-2 (N=256) reaches approximately 95% accuracy while attention reaches approximately 85%. The performance improves monotonically with N, directly validating the hypothesis that larger state size enables better memorization.

- **Attention-based models perform well at shorter sequences but degrade at longer ones** relative to high-N Mamba-2, consistent with attention's difficulty scaling to long sequences where the number of key-value pairs to track grows.

The importance of these results for the paper's claims is twofold. First, they validate that the SSD reformulation enables larger state sizes that were previously impractical in Mamba, and that these larger states translate to meaningful capability improvements. Second, they demonstrate that Mamba-2 is not merely matching Mamba-1 with better efficiency—it is genuinely improving on capability tasks that were previously weak points for the SSM family.

#### Hybrid Models: SSD Complements Attention and MLP Layers

Section 9.2.3 explores combining SSD layers with attention and MLP layers in hybrid architectures. The experiments use a 350M parameter model with 48 layers trained for 7B tokens on the Pile with the GPT-2 tokenizer, varying the number of attention layers while maintaining total layers and parameters. Results are in Table 2.

**SSD and attention are complementary.** A pure Mamba-2 model (0 attention layers) achieves perplexity 8.60. Adding just 1 attention layer (2% of the total) reduces perplexity to 8.38—a meaningful improvement. The optimal configuration uses 6 attention layers (12.5% of total), achieving perplexity 8.26. Adding more attention layers beyond this point degrades performance: with 24 attention layers (50%), perplexity rises to 8.50, and the pure Transformer++ (presumably all 48 layers use attention) achieves 8.68.

The paper interprets this inverted-U shape as evidence that:

> "the SSM layers function well as a general sequence-to-sequence mapping, and attention layers act as a retrieval mechanism to quickly refer to previous tokens in the sequence instead of forcing the model to compress all the context to its memory (SSM states)."

Too few attention layers provide insufficient retrieval capability; too many attention layers lose the benefits of SSM compression and global context modeling. The optimal tradeoff around 10% attention layers is consistent across the tested configurations.

**At the 2.7B scale with 300B training tokens,** the hybrid results (Table 3) are more nuanced:

- **Mamba-2-Attention** (58 SSD layers + 6 attention layers): achieves the best overall average accuracy at 61.0%, outperforming both pure Mamba-2 (60.2%) and Transformer++ (60.2%). The Pile perplexity is 5.95 vs. Mamba-2's 6.09 and Transformer++'s 6.13.
- **Mamba-2-MLP-Attention** (28 SSD + 4 attention + 32 gated MLP, interleaved): achieves 60.7% average, slightly below Mamba-2-Attention but still competitive. The MLP layers reduce quality slightly but may offer training speed advantages.
- **Mamba-2-MLP** (32 SSD + 32 gated MLP, interleaved): achieves 59.6% average, below pure Mamba-2 (60.2%). This suggests MLP layers alone (without attention) degrade quality compared to pure SSD, but may be useful for training efficiency since MLP layers are simpler and more hardware-efficient than SSD layers.
- **Pure Mamba-2 and Transformer++ achieve nearly identical average accuracy** (both 60.2%), with different task-level strengths: Mamba-2 performs better on PIQA (76.4 vs. 75.2) and ARC-easy (69.6 vs. 67.7), while Transformer++ performs better on ARC-challenge (37.8 vs. 36.4) and OpenBookQA (40.4 vs. 38.8).

The hybrid experiments support the paper's claim that the SSD framework is not just about building better pure SSMs, but about understanding how to combine sequence modeling primitives (SSM compression, attention retrieval, MLP transformation) in principled ways. The finding that ~10% attention layers is optimal provides concrete design guidance.

### Ablation Studies and Robustness Checks

- **Block design (parallel vs. sequential projections, extra normalization):** Table 4 ablates the Mamba-2 block design choices independent of the core SSD layer. Using Mamba-1's sequential projections (where $A$, $B$, $C$ are generated from the SSM input $x$) yields perplexity 11.76 at 129.3M parameters. Switching to parallel projections (all generated simultaneously at the block input) reduces perplexity to 11.66 at 126.5M parameters—a perplexity improvement with 2.2% fewer parameters. Adding the extra normalization layer (GroupNorm before the final output projection) further reduces perplexity: sequential + normalization achieves 11.54 at 129.3M parameters; parallel + normalization (the full Mamba-2 block) achieves 11.49 at 126.5M parameters. The normalization thus contributes roughly 0.05–0.17 perplexity improvement across configurations. The paper notes that the normalization was primarily motivated by training stability at larger scales, making the small perplexity improvement a secondary benefit.

- **Multi-head structure:** Table 5 compares head patterns at two scales (125M parameters trained on 2.5B tokens; 360M parameters trained on 7B tokens), controlling for total state dimension (HPN is constant across configurations). At 125M scale: MIS/MVA (multi-input SSM / multi-value attention—the pattern used in Mamba and Mamba-2) achieves perplexity 11.66. MCS/MQA (multi-contract SSM / multi-query attention) achieves 12.62. MES/MKA (multi-expand SSM / multi-key attention) achieves 12.59. MHS/MHA (multi-head) achieves 12.06 at 24 heads, and a variant with all heads shared (multi-state SSM) achieves 12.00. At 360M scale, the pattern is consistent: MIS/MVA achieves 8.73 vs. MCS/MQA at 9.33, MES/MKA at 9.36, and MHS/MHA at 9.01–9.04. The perplexity gap of approximately 0.6–1.0 between MIS/MVA and the alternatives is substantial—larger than the gap between different architectures in the scaling law experiments. The paper attributes this to the fact that sharing $B$ and $C$ across channels while allowing independent $A$ per head (the MIS/MVA pattern) provides effective state capacity while avoiding over-parameterization of the input/output projections. The result validates Proposition 7.2's characterization of Mamba's architecture and confirms that the MVA pattern is not an arbitrary historical choice but genuinely outperforms the attention-inspired alternatives.

- **Kernel activation functions:** Table 6 tests various feature maps $\psi$ for the $B$ and $C$ projections (Section 7.3). No activation (none) achieves perplexity 11.58. Swish (the default) achieves 11.66—slightly worse than no activation. Exp achieves 11.62. ReLU achieves 11.73, and ReLU with normalization (adding a denominator term analogous to softmax normalization) achieves 11.64. cosFormer's feature map (Qin, Weixuan Sun, et al. 2022) degrades performance to 11.97. Random Feature Attention (H. Peng et al. 2021) achieves 11.57, nearly matching no activation. Positive Random Features (Performer, Choromanski et al. 2021) significantly degrades performance to 12.21. The key finding is that sophisticated kernel approximations from the linear attention literature do not improve over simple choices, and some actively hurt. The paper interprets this as evidence that the 1-SS mask $L$ already provides the regularization (input-dependent gating and positional structure) that kernel approximations were designed to supply in softmax-free attention.

- **Based and ReBased methods:** Table 7 tests feature maps from the Based (Arora, Eyuboglu, Zhang, et al. 2024) and ReBased (Aksenov et al. 2024) models, which use Taylor expansion approximations to the exponential kernel. For 130M models with N=64: Swish (default) achieves 11.67. Swish + Taylor (Based, concatenating $[1, x, 1/\sqrt{2} \cdot x \otimes x]$) degrades to 12.19. LayerNorm (applied before the SSM, analogous to QK-Norm in Transformers) achieves 11.50—an improvement over Swish. LayerNorm + Square (ReBased, using $x \otimes x$ as the kernel) degrades to 11.84. For 360M models with N=256, the pattern is similar but the gaps narrow: Swish 8.58, Swish + Taylor 8.71, LayerNorm 8.61, LayerNorm + Square 8.63. The LayerNorm result is promising (11.50 vs. 11.67 for Swish at 130M scale) but the paper does not adopt it as default, noting it requires further investigation. The negative results for Taylor/Squared features suggest that expanding the feature dimension through outer products does not help when the model already has a dedicated state expansion mechanism through $N$.

### Critical Assessment

The paper's experiments provide strong support for its central efficiency claim—that the SSD algorithm is substantially faster than Mamba's selective scan—while the quality claims (competitive or superior language modeling) are supported with appropriate controls but leave some questions unresolved.

**The speed improvement claim (2–8× faster than Mamba scan) is robustly supported.** Figure 10 provides clear, well-controlled benchmarks: same hardware, same sequence lengths, varying state dimensions. The mechanism is transparent: Mamba's scan does not use tensor cores; SSD's block decomposition converts most computation to matrix multiplications that do. The right panel of Figure 10 is particularly informative because it demonstrates the practical consequence—Mamba-2 can use 8–16× larger state sizes (N=64 to N=256 vs. Mamba's typical N=16) with better or equal speed, which directly enables better performance on memory-intensive tasks as validated by the MQAR experiments (Figure 8).

However, the speed comparison has a subtle limitation: it benchmarks the **SSM/SSD layer in isolation**, not the full model. A real Transformer model at short sequence lengths (e.g., 2K) has roughly half its layers as MLP (which are extremely fast matmuls) and half as attention. A Mamba-2 model at the same parameter count has all layers as SSD. The paper acknowledges this in Section 9.3:

> "the Mamba-2 model as a whole might not be as efficient to train as Transformer at short sequence length (e.g. at 2K), since a Transformer with L layers would have L/2 MLP layers and L/2 attention layers, while a Mamba-2 model would have L SSD layers for the same number of parameters."

The hybrid results (Section 9.2.3) partially address this: Mamba-2-MLP (half SSD, half MLP) achieves competitive quality and would train faster at short sequences by replacing expensive SSD layers with cheap MLP layers. But a direct full-model wall-clock training comparison (total time to reach a given perplexity) across architectures is not reported. The scaling law experiments (Figure 9) are plotted against **theoretical FLOPs**, not wall-clock time—the Pareto dominance on wall-clock time is asserted but not directly measured for these experiments.

**The language modeling quality claims are well-supported but the comparison is imperfect.** Table 1 shows Mamba-2 matching or exceeding Pythia at 2× the parameter count, which is the paper's primary quality claim. The controlled comparison is strong: same dataset (the Pile), same tokenizer (GPT-NeoX), same training length (300B tokens). However, the training length is fixed at 300B tokens for all model sizes, which means that smaller models (130M, 370M) are trained far beyond Chinchilla-optimal while larger models (2.7B) are trained below optimal. This fixed budget favors architectures that are more compute-efficient (or sample-efficient) at the specific training horizon tested. Chinchilla-optimal training would scale tokens proportionally to parameters, which might yield different relative rankings.

The scaling law experiments (Figure 9) partially address this by using Chinchilla-like token budgets (Table 9), but these are only reported up to 1.3B parameters and 26B tokens—well below the scale of the downstream evaluations. The downstream evaluations at 2.7B and 300B tokens are thus operating in a different (and less Chinchilla-optimal) regime than the scaling laws. The paper does not discuss whether the relative ordering of architectures might change if all models were trained at their Chinchilla-optimal token budgets.

**The MQAR results support the state size hypothesis but leave the mechanism unexplained.** Figure 8 clearly shows that Mamba-2 with larger N performs better on associative recall, and that Mamba-2 with N=16 already outperforms Mamba-1 with N=16. The second finding is important because it suggests that factors other than state size contribute to the improvement. The paper explicitly acknowledges this as an open question:

> "We are not sure which aspect of the architecture is the predominant factor, which remains a question to explore in future work."

Possible explanations include: (1) the scalar-identity structure on A (all state dimensions share the same input-dependent gate) might provide better learning dynamics for selective memorization than independent per-dimension gates; (2) the larger head dimension P=64 (vs. Mamba's P=1) might provide richer key-value representations through the B and C projections; (3) the parallel projection design might improve optimization; (4) the extra normalization might stabilize training on this task. The paper does not ablate these factors in the MQAR setting, so the causal attribution remains unclear.

**The hybrid model experiments are suggestive but limited in scale.** Table 2 (350M, 7B tokens) and Table 3 (2.7B, 300B tokens) show that adding a small number of attention layers improves over pure Mamba-2. However, the optimal ratio (approximately 10% attention layers) is identified at a single scale (350M) and then applied at 2.7B without further optimization. The paper does not explore whether this ratio is scale-dependent—it is plausible that larger models benefit from different ratios, or that the optimal ratio depends on the specific training data distribution. Additionally, the hybrid experiments in Table 3 compare architectures at fixed total layer count but do not control for total FLOPs per forward pass, making it difficult to determine whether the hybrid benefits come from increased expressivity or simply increased computation.

**Ablation scope is limited by computational constraints.** The kernel activation ablations (Tables 6–7), head pattern ablations (Table 5), and block design ablations (Table 4) are performed at the 125M–380M parameter scale with 2.5B–7B training tokens—substantially below the scales where architectural differences often matter most. It is possible that some negative results (e.g., Taylor features in Table 7 hurting performance) would reverse at larger scales where the model has more capacity to leverage expanded feature representations. The paper acknowledges this implicitly by labeling these as "ablations" rather than definitive conclusions, but the computational constraints of large-scale training mean several design choices (MVA head pattern, Swish activation, no normalization denominator) are adopted without verification at the 2.7B scale.

**Missing experiments that would strengthen the paper.** Several experiments would help solidify the claims:
- A full-model wall-clock training comparison (time to reach a given perplexity) across Transformer++, Mamba-1, and Mamba-2 at matched parameter counts and at least two sequence lengths (short: 2K, long: 8K or 16K) would directly validate the "Pareto dominant on wall-clock time" claim.
- Chinchilla-optimal training of all architectures at a single model size (e.g., 1.3B parameters trained on ~26B tokens, 2.7B on ~50B tokens, etc.) would test whether Mamba-2's advantage persists when token budgets are matched to model capacity rather than fixed.
- Ablation of the scalar-identity vs. diagonal structure on A in the MQAR setting would isolate how much of the Mamba-2 improvement over Mamba-1 comes from the structural restriction (scalar-identity) vs. other architectural changes (head dimension, parallel projections, normalization).
- Scaling the hybrid ratio experiment (Table 2) to 1.3B or 2.7B parameters would test whether the ~10% optimal ratio is robust to scale.

**Summary assessment.** The experiments provide strong support for the paper's central thesis—that the SSD framework enables faster and more capable SSMs—but the support is strongest for the efficiency claims (Figure 10, Figure 9) and most compelling at moderate scales (up to 1.3B for scaling laws, 2.7B for downstream). The quality claims (matching Transformers, outperforming at 2× parameter count) are supported with appropriate controls but operate in a fixed-data-budget regime that may favor certain architectures. The MQAR results compellingly demonstrate that larger state sizes improve memory-intensive capabilities, directly validating the practical benefit of SSD's efficiency. The paper's open acknowledgment of unexplained findings (Mamba-2 outperforming Mamba-1 at matched N on MQAR, the specific mechanism undetermined) strengthens credibility by not overclaiming causal understanding where it is lacking.

## 6. Limitations and Trade-offs

### 6.1 The Scalar-Identity Restriction on A Reduces Expressivity Relative to Diagonal SSMs

**The assumption or constraint.** The SSD model requires the state transition matrix to have scalar-identity structure: $A_t = a_t \cdot I$ for scalar $a_t$, rather than the more general diagonal structure $A_t = \text{diag}(a_t^{(1)}, \ldots, a_t^{(N)})$ used in Mamba (S6). The paper explicitly acknowledges this tradeoff in Section 5:

> "The main restriction of SSD is on the expressivity of the state transitions $A_t$. We note that more general SSMs, such as the case of diagonal $A_t$, have the same theoretical efficiency as SSD, but are less hardware-friendly. This is because the dual quadratic form loses its attention-like interpretation and becomes more difficult to compute."

The scalar-identity restriction is what enables the factorization $M = L \circ (C B^\top)$ in Section 5.1 that makes the SSD quadratic form identical to masked kernel attention. Without it, the matrix $C_j^\top A_{j:i}^\times B_i$ does not cleanly separate into a scalar mask times an inner product, and the block-decomposition algorithm's diagonal blocks cannot be computed via the efficient attention-like form.

**The consequence.** The scalar-identity structure means all $N$ state dimensions share the same input-dependent gating scalar $a_t$ at each timestep. In a diagonal SSM like Mamba, different state dimensions can have different gating values—some dimensions might choose to remember while others forget, creating a richer set of dynamical behaviors. The scalar-identity restriction collapses this diversity: all dimensions must decay or retain information at the same rate. The paper compensates by increasing the state dimension $N$ (using $N=64$ or $N=128$ rather than Mamba's $N=16$) and increasing the head dimension $P$, but this trades one form of expressivity (independent per-dimension dynamics) for another (more total state capacity through larger $N$). Whether this is a net win or loss likely depends on the task. For language modeling, the scaling laws (Figure 9) suggest the tradeoff is favorable—Mamba-2 Pareto-dominates Mamba at matched compute. But for tasks where fine-grained selective gating across different representational subspaces is critical, the scalar-identity restriction could be limiting. The paper does not characterize which tasks or data distributions would expose this limitation.

**What evidence exists in the paper.** The paper provides no direct ablation comparing scalar-identity SSD against diagonal SSD at matched state dimension $N$ and matched compute. The MQAR experiments (Figure 8) show Mamba-2 (scalar-identity) outperforming Mamba-1 (diagonal) even at matched $N=16$, which the paper explicitly flags as unexplained:

> "Surprisingly, it is significantly better than Mamba-1 even when the state sizes are controlled ($N = 16$). (We are not sure which aspect of the architecture is the predominant factor, which remains a question to explore in future work.)"

This means the MQAR results do not isolate the effect of the scalar-identity restriction—the improvement could come from other factors (head dimension, parallel projections, normalization), and the scalar-identity restriction might actually be hurting performance relative to a hypothetical diagonal SSD with all other architectural improvements held constant.

**Mitigation status.** The paper acknowledges this limitation in Section 10.1:

> "We hypothesize that it may be possible to refine our structured matrix algorithms to improve to the general diagonal SSM case as well."

This is flagged as future work. No algorithmic extension to diagonal SSMs is provided. The block-decomposition approach (Section 6) relies on the diagonal blocks being computable via the quadratic attention form, which requires the scalar-identity structure to produce the factorized $M = L \circ (C B^\top)$. Extending to diagonal $A_t$ would require either a different within-chunk computation (losing the matmul efficiency advantage) or a more complex factorization that the paper does not develop. Until such an extension exists, practitioners choosing between Mamba-1 and Mamba-2 face a genuine tradeoff: Mamba-2 offers much faster training and larger state sizes, but with potentially less flexible state dynamics per dimension.

### 6.2 Full-Model Training Efficiency at Short Sequence Lengths Is Unresolved

**The assumption or constraint.** The speed benchmarks in Figure 10 measure the core sequence mixing operation (SSD, Mamba scan, attention) in isolation, not the full model including MLP layers, normalization, residual connections, and output projections. The paper explicitly notes this gap in Section 9.3:

> "However, we note that the Mamba-2 model as a whole might not be as efficient to train as Transformer at short sequence length (e.g. at 2K), since a Transformer with $L$ layers would have $L/2$ MLP layers and $L/2$ attention layers, while a Mamba-2 model would have $L$ SSD layers for the same number of parameters. Generally the MLP layers are very hardware efficient since they consist of simple matrix multiplication and pointwise linearity."

At short sequence lengths (e.g., 2K tokens), the attention operation is already fast because the quadratic cost $O(T^2)$ is manageable when $T$ is small, and FlashAttention-2 is highly optimized. Meanwhile, a pure Mamba-2 model replaces the cheap MLP layers (present in roughly half of Transformer layers) with SSD layers that, while faster than Mamba's scan, are still more expensive than simple matmul-based MLP operations. The isolated layer speed advantage of SSD over attention (Figure 10) therefore may not translate directly to full-model training speed at short sequence lengths.

**The consequence.** For practitioners training at typical sequence lengths (2K–4K tokens, which covers most current LLM training regimes), the decision between Transformer and Mamba-2 is not settled by the paper's efficiency numbers. A Transformer with FlashAttention-2 at sequence length 2K might train faster in wall-clock time than a pure Mamba-2 model at matched parameter count, even if Mamba-2 would be faster at sequence length 8K or 16K. The paper's scaling law experiments (Figure 9) are plotted against theoretical FLOPs, not wall-clock time—the "Pareto dominant on wall-clock time" claim in the Figure 9 caption is not directly measured for these experiments. The hybrid results (Section 9.2.3) suggest a partial mitigation: Mamba-2-MLP (half SSD, half MLP layers) achieves competitive perplexity while replacing expensive SSD layers with cheap MLP layers, which would improve training speed at short sequence lengths. But the paper does not report wall-clock training times for hybrid configurations.

**What evidence exists in the paper.** The paper provides no full-model wall-clock training time comparison across Transformer++, Mamba-1, and Mamba-2 at any sequence length. The speed benchmarks (Figure 10) isolate the sequence mixing operation. The hybrid experiments (Tables 2–3) report only perplexity and downstream accuracy, not training speed. The scaling laws (Figure 9) use theoretical FLOPs as the x-axis. The paper acknowledges the issue (Section 9.3) but does not resolve it empirically.

**Mitigation status.** The paper proposes the hybrid Mamba-2-MLP architecture as a practical solution:

> "As shown in Section 9.2.3, one can also combine $L/2$ SSD layers and $L/2$ MLP layers to speed up training at short sequence length."

This is a partial mitigation: it recognizes that pure SSD may be suboptimal at short lengths and provides an architectural alternative. However, the training speed of Mamba-2-MLP relative to Transformer++ at short sequence lengths is not benchmarked. The quality of Mamba-2-MLP is slightly below pure Mamba-2 (Table 3: 59.6% vs. 60.2% average downstream accuracy at 2.7B), so the tradeoff between training speed and model quality is not characterized. A practitioner deciding whether to adopt Mamba-2 for training at sequence length 2K would need to run their own benchmarks.

### 6.3 The Theoretical Framework Does Not Extend to Softmax Attention

**The assumption or constraint.** The SSD duality (Section 5) establishes an exact equivalence between scalar-identity SSMs and 1-semiseparable SMA, where the mask $L$ replaces the softmax nonlinearity of standard attention. The paper explicitly acknowledges this boundary in Section 10.3:

> "We emphasize that SSD does not generalize standard softmax attention, or any other transformation on the attention kernel matrix that does not have a finite feature map $\psi$."

Standard softmax attention applies the nonlinear function $f(G) = \text{softmax}(G)$ row-wise to the Gram matrix $G = QK^\top$, then multiplies by $V$. The softmax function is not a finite-dimensional kernel feature map—it requires infinite-dimensional feature space (the exponential can be represented via an infinite series, but the row-wise normalization $\text{softmax}(G) = \exp(G) / \exp(G) \cdot \mathbf{1}$ couples all positions in a row and cannot be expressed as a finite-rank matrix factorization). Therefore, standard softmax attention lies strictly outside the SSD framework. The SMA abstraction (Definition 4.2) applies only to masked kernel attention with a finite feature map, which is a specific subclass of attention variants. Any property of softmax attention that depends crucially on the row-wise normalization—such as the "attention sink" phenomenon, sharp selectivity through exponentiation, or the dynamic range compression from softmax—does not carry over to SSD.

**The consequence.** The paper's framework cannot be used to analyze or improve standard softmax attention directly. While the paper hypothesizes that the 1-SS mask $L$ provides similar benefits to the softmax (input-dependent gating, selectivity), this is an empirical claim, not a theoretical consequence of the duality. The negative results in Tables 6–7—where kernel approximations designed to mimic softmax attention (Performer's positive random features, Based's Taylor expansion, cosFormer's cosine reweighting) do not improve over simple Swish activations—suggest that the 1-SS mask provides a different form of regularization than softmax, not an approximation of it. Models that rely on softmax-specific properties (e.g., the ability to attend sharply to a single position through near-one softmax weights while suppressing all others) may not be replicable with the SSD mechanism, where the $a_t$ scalars provide a smoother, multiplicative gating.

**What evidence exists in the paper.** Table 6 and Table 7 provide evidence that kernel approximation techniques from the linear attention literature do not transfer beneficially to SSD. The paper notes in Section 7.3:

> "We emphasize however that SSD and vanilla linear attention differ in the inclusion of the 1-semiseparable mask $L$, while the various linear attention methods in the literature were derived to approximate softmax attention without this term; thus, our negative results may be not unexpected."

The MQAR results (Figure 8) compare SSD against standard softmax attention, showing Mamba-2 outperforming attention at longer sequence lengths. But this comparison is on a specific synthetic task, and the relative performance on tasks where softmax's sharp selectivity is important (e.g., precise copying of a single token from a long context) is not characterized.

**Mitigation status.** The paper does not attempt to bridge SSD to softmax attention. Section 10.3 suggests this as a future direction:

> "We suggest that... finding ways to characterize and bridge the gap between softmax attention and sub-quadratic models through analyzing their matrix transformation structure" is a promising avenue.

This is a fundamental limitation of the theoretical framework, not a temporary gap. The duality is exact for kernel attention with finite feature maps; softmax attention requires an infinite-dimensional feature map. Extending the framework to encompass softmax would require either approximating softmax with finite feature maps (introducing approximation error that breaks the exact duality) or developing a different theoretical apparatus for row-wise normalized attention matrices. The paper provides neither.

### 6.4 Headline Speedup Claims Exclude the Cost of Full-Model Components

**The assumption or constraint.** The 2–8× speedup claim in the abstract and Section 9.3 refers specifically to the SSD layer compared to Mamba's selective scan implementation. Both measurements are for the core sequence mixing operation in isolation. A full Mamba-2 model includes additional components that are identical in both architectures (or nearly so): input and output projections, the gating branch $z$, the convolution, normalization layers, and residual connections. The relative speedup of the full model is therefore strictly less than the layer-level speedup, by a factor that depends on the ratio of time spent in the SSM/SSD layer vs. other components.

The paper acknowledges this framing but does not quantify the dilution:

> "our SSD algorithm is much more efficient than Mamba-1" (Section 9, referring to Figure 10)

The abstract states "2-8× faster" without the qualification that this is layer-level, not model-level. A careful reader must consult Section 9.3 to understand the scope of the measurement.

The additional components in a Mamba-2 block include: projection matrices $W^{(x)}$ and $W^{(z)}$ (each of size $d \times ed$ where $e$ is the expansion factor, typically 2, costing $2 \cdot 2d^2$ FLOPs per token), the output projection $W^{(o)}$ (size $ed \times d$, costing $2d^2$ FLOPs per token), the convolution (a depthwise 1D convolution with kernel size typically 3–4), the Swish gating activation, and the GroupNorm layer. For typical model dimensions ($d=2048$ or $d=2560$ at 1.3B–2.7B scale), the projection FLOPs are substantial. For a 2.7B model with $d=2560$ and expansion factor 2: input projections cost $2 \times 2560 \times 5120 \times 2 = 52.4$M FLOPs per token (for $W^{(x)}$ and $W^{(z)}$ combined), output projection costs $2560 \times 5120 = 13.1$M FLOPs per token, for a total of roughly 65M FLOPs per token in linear projections alone. The SSD layer with $N=64$ costs roughly $T N^2 = 8192 \times 4096 ≈ 33.6$M FLOPs per token for the dominant matrix multiplications. The projections thus constitute roughly 2/3 of the total FLOPs in this regime, which means a 4× speedup in the SSD layer translates to perhaps a 1.5–2× speedup in the full model.

**The consequence.** A practitioner reading the abstract's "2-8× faster" claim and expecting that training Mamba-2 will be 2-8× faster than training Mamba-1 at matched model sizes would be overestimating the practical benefit. The actual full-model speedup depends on the sequence length (longer sequences → SSD dominates more of the total FLOPs → speedup closer to the layer-level figure) and the model dimension (wider models → projections cost more → speedup further diluted). Without a full-model breakdown, the paper's efficiency claims are upper bounds that may not be realized in typical training configurations.

**What evidence exists in the paper.** The paper provides layer-level benchmarks (Figure 10) and notes the caveat about full-model efficiency (Section 9.3), but does not provide a full-model timing breakdown or measure the fraction of total FLOPs spent in the SSD layer vs. other components at typical model sizes and sequence lengths. The scaling laws (Figure 9) use theoretical total FLOPs, which would correctly account for all components, but the speedup factor in the abstract is not tied to these FLOPs measurements—it comes from Figure 10, which is layer-level.

**Mitigation status.** The paper is transparent about the scope of the measurement (Section 9.3 identifies the full-model caveat), but the headline numbers in the abstract and introduction do not carry this qualification. The hybrid model experiments (Section 9.2.3) provide an indirect mitigation: by replacing some SSD layers with MLP layers (which are cheaper), the effective speedup of the hybrid model relative to pure Mamba-2 at short sequence lengths can be substantial, though not quantified. A practitioner can infer from the discussion that for short-sequence training, the hybrid Mamba-2-MLP or Mamba-2-MLP-Attention architectures likely provide better throughput than pure Mamba-2, even without exact numbers.

### 6.5 Exploration of the Head Pattern and Kernel Design Space Is Limited to Small Scale

**The assumption or constraint.** All of the paper's architectural ablations—head patterns (Table 5), kernel activation functions (Tables 6–7), block design choices (Table 4)—are performed at small model scales (125M–380M parameters) with limited training tokens (2.5B–7B). The downstream evaluations that demonstrate Mamba-2's competitiveness (Table 1) use the architectural choices derived from these small-scale ablations at the 780M–2.7B scale (300B tokens). The paper implicitly assumes that the relative ordering of architectural choices is preserved across scale. This assumption is common in the LLM literature due to computational constraints but is known to be violated in some cases—for instance, the optimal learning rate, batch size, and sometimes even architectural choices (like activation functions or normalization placement) can shift with model scale.

**The consequence.** Several architectural decisions in Mamba-2 are based on small-scale ablations that could reverse at larger scales:
- The MVA (multi-value attention) head pattern outperforms MQA and MHA by ~0.6–1.0 perplexity points at 125M–360M scale (Table 5). If this gap narrows at larger scales, the choice might be less consequential than it appears, or an alternative pattern might overtake MVA.
- The negative results for kernel approximations (Tables 6–7) showing that Performer features, Taylor expansions, and cosFormer all underperform Swish or no activation could be an artifact of small scale—at larger scales with more capacity, expanded feature maps might become beneficial.
- The LayerNorm activation (Table 7) achieves 11.50 vs. Swish's 11.67 at 130M scale, a promising result that suggests LayerNorm might be superior. But it is not tested at larger scales, and Mamba-2 defaults to Swish.
- The extra normalization layer (GroupNorm after gating) was primarily adopted for training stability at larger scales (Section 7.1):

> "In preliminary experiments, we found that instabilities were prone to arising in larger models."

This suggests the normalization choice was driven by large-scale behavior not visible at the ablation scale, making the small-scale perplexity results (Table 4, ~0.05–0.17 improvement) potentially understating its importance. Conversely, some design choices that look neutral or negative at small scale might become important at large scale.

**What evidence exists in the paper.** All ablation tables (4, 5, 6, 7) explicitly report model sizes (125M or 360M) and training tokens (2.5B or 7B). The paper does not claim these results transfer to larger scales, but neither does it verify the transfer. The downstream evaluations (Table 1) use the chosen configuration without reporting ablations at the 780M–2.7B scale, so there is no direct evidence that the small-scale optimal choices remain optimal at deployment scale.

**Mitigation status.** No mitigation is attempted. The paper treats the ablation results as sufficient justification for the architectural choices, following standard practice in the LLM literature where full-scale architectural sweeps are prohibitively expensive. The acknowledgment is implicit in the fact that the paper labels these as "ablations" (suggesting preliminary investigation) rather than "scaling studies." Section 10.3 notes the unexplained improvement of Mamba-2 over Mamba-1 at matched N on MQAR as an open question, which indirectly acknowledges that the architectural factors are not fully understood. A practitioner adopting Mamba-2 should treat the specific design choices (MVA pattern, Swish activation, parallel projections with GroupNorm) as the authors' best estimate based on available evidence, not as definitively validated at scale.

### 6.6 The Model's Effectiveness on Hard Reasoning Tasks Remains Uncharacterized

**The assumption or constraint.** The downstream evaluation suite in Table 1 covers standard zero-shot benchmarks: LAMBADA (language modeling / next-word prediction), HellaSwag (commonsense reasoning), PIQA (physical commonsense), ARC-easy and ARC-challenge (science reasoning), WinoGrande (commonsense reasoning), and OpenBookQA (knowledge-intensive reasoning). These tasks all involve relatively short-range reasoning over single passages or questions. The paper does not evaluate on tasks requiring: (1) long-range multi-step reasoning (e.g., multi-hop QA, mathematical reasoning, code execution), (2) few-shot or in-context learning over long contexts, or (3) information retrieval from very long documents. These capabilities are where the distinction between SSMs' compressed state (fixed size N, independent of sequence length) and attention's cache (size scaling with T) should matter most.

The paper's synthetic evaluations partially address this gap: the MQAR experiments (Figure 8) test memory over long sequences, and the SSD model with large N performs well. But MQAR tests a specific form of associative recall (mapping a key to a memorized value) that may not capture the complexity of real-world long-range reasoning tasks. More challenging evaluations like multi-hop QA (requiring chaining multiple facts across a long document), code execution (requiring precise variable tracking over many lines), or mathematical proof generation (requiring maintaining intermediate lemmas) would stress-test the fixed-size state bottleneck more directly.

**The consequence.** For practitioners considering Mamba-2 as a replacement for Transformers in applications requiring long-range reasoning, the paper provides insufficient evidence. While Mamba-2 matches or exceeds Pythia on the tested benchmarks (Table 1), these benchmarks primarily evaluate knowledge and short-range reasoning, not the memory-compression vs. caching tradeoff that is the core architectural difference between SSMs and attention. An application that requires retrieving a specific fact from a 100K-token document or reasoning over a complex multi-step procedure might find that Mamba-2's compressed state loses critical information, even with large N (256+). The paper's MQAR results suggest that increasing N helps with associative recall, but MQAR is a simpler task than real-world long-range reasoning. The hybrid architecture results (Table 3, where adding attention layers to Mamba-2 improves performance) suggest that attention's non-compressed cache provides information that the SSM state does not capture, even in the tested evaluation suite. The paper hypothesizes:

> "the SSM layers function well as a general sequence-to-sequence mapping, and attention layers act as a retrieval mechanism to quickly refer to previous tokens in the sequence instead of forcing the model to compress all the context to its memory (SSM states)."

If this hypothesis is correct, then pure Mamba-2 (without attention layers) should be expected to underperform on tasks where retrieval of specific previous tokens is critical—which includes many real-world long-context applications.

**What evidence exists in the paper.** The downstream evaluation suite (Table 1) does not include long-range reasoning tasks. The MQAR results (Figure 8) cover synthetic associative recall. The hybrid experiments (Table 3) show that adding attention layers improves performance, indirectly supporting the hypothesis that pure SSM compression loses information. The paper acknowledges the gap in Section 10.3, citing concurrent work:

> "Concurrent work has starting studying the tradeoffs of these representations, for example on copying and in-context learning tasks (Akyürek et al. 2024; Grazzi et al. 2024; Jelassi et al. 2024; Park et al. 2024). We note that Mamba-2 significantly improves on Mamba on some of these capabilities (e.g. as demonstrated by MQAR results in Section 9.1), but more remains to be understood."

**Mitigation status.** The paper does not characterize the failure modes. The hybrid architecture results (Section 9.2.3) provide a practical mitigation: combine SSD layers with a small number of attention layers (around 10% of total layers) to cover the retrieval capability that pure SSM compression misses. This hybrid approach achieves the best results in Table 3 (Mamba-2-Attention at 61.0% average vs. pure Mamba-2 at 60.2%). However, the optimal configuration for specific long-range reasoning tasks is not explored, and the paper does not provide guidance on when hybrid architectures are necessary vs. when pure Mamba-2 suffices. Until evaluations on long-range reasoning benchmarks are available, a practitioner deploying Mamba-2 in a retrieval-heavy application should either adopt the hybrid architecture (sacrificing some training speed for the attention layers) or conduct their own task-specific evaluations.

## 7. Implications and Future Directions
- How this changes the landscape
  - Provides a single, principled lens—semiseparable matrices—for both SSMs and efficient attention, clarifying when and why linear-time forms exist (Theorems 3.5, 5.2; Figures 2–4). This is a foundation for designing long-context models that are both linear-time and systems-friendly.

- Research directions enabled
  - Explore other `SMA` masks (Toeplitz, Fourier) to encode richer positional or frequency structure with efficient recurrent forms (Figure 3).
  - Extend the `SSD` algorithm to broader `A_t` structures (full diagonal) while keeping matmul dominance (Section 10.1).
  - Non-causal and bidirectional variants using the matrix-mixer view; inverse/closure properties (Appendix C.1) suggest principled constructions (e.g., invertible semiseparable mixers).
  - Theoretical analysis of in-context learning and copying for SSD vs attention, leveraging the autoregressive characterization (Theorem 5.2) and recent benchmarks (Section 9.1).

- Practical applications
  - Long-context language models with lower memory and compute, especially for streaming or on-device settings where constant-state inference is valuable (Sections 1, 8.2).
  - Training at scale with standard parallelism stacks (TP/SP) thanks to the Mamba-2 block design (Section 8; Figure 7).
  - Hybrid stacks combining SSD with a small fraction of attention layers to balance retrieval and compression (Tables 2–3).

> In the authors’ code release, `Mamba-2` and the `SSD` layer target production use (Section 9), with Listing 1 offering an end-to-end reference implementation and the system sections detailing how to integrate with tensor/sequence parallelism.

Overall, this paper contributes a unifying theory, a practical algorithm, and an architecture that together make SSMs a first-class, hardware-efficient alternative (and complement) to Transformers for long-context modeling.
