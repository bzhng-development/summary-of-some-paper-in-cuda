# Mamba: Linear-Time Sequence Modeling with Selective State Spaces

**ArXiv:** [2312.00752](https://arxiv.org/abs/2312.00752)

## 🎯 Pitch

Mamba introduces selective state space models (SSMs), enabling fully recurrent, linear-time sequence models to perform content-based reasoning by making their internal parameters input-dependent. This breakthrough allows the Mamba architecture to match or surpass Transformer-level performance on language, audio, and genomics tasks—while scaling efficiently to million-length sequences and providing up to 5× higher inference throughput. By combining principled selectivity with hardware-aware implementation, Mamba offers a compelling new foundation model backbone for applications where long context and computational efficiency are critical.

---

## 1. Executive Summary

This paper introduces Mamba, a new class of selective state space models that overcome a key limitation of prior structured state space models (SSMs)—their inability to perform content-based reasoning on discrete modalities like language—by making the SSM parameters functions of the input (a **selection mechanism** that lets the model selectively propagate or forget information along the sequence based on the current token) and by designing a **hardware-aware parallel scan algorithm** that computes the resulting time-varying model efficiently without materializing the expanded state in GPU high-bandwidth memory. On language modeling, Mamba-3B matches Transformers twice its size in both pretraining perplexity and zero-shot downstream evaluation (e.g., averaging 4 points higher on commonsense reasoning than Pythia-3B and exceeding Pythia-7B), while achieving 5× higher inference throughput than similarly-sized Transformers. On synthetic tasks, Mamba extrapolates solutions indefinitely to sequences beyond 1M tokens, establishing that selective state space models can perform context-dependent reasoning at scales inaccessible to both attention-based and prior LTI recurrent models—though the gains are concentrated on problems within the model's capability range, with the selection mechanism sometimes hampering performance on continuous perceptual modalities like audio where time-invariant inductive biases remain beneficial.

## 2. Context and Motivation

### The Core Problem: Transformers Dominate Everything, But They Don't Scale to Long Sequences

The fundamental tension this paper grapples with is one of the central paradoxes in modern deep learning: the Transformer architecture (Vaswani et al., 2017) is overwhelmingly dominant as the backbone for foundation models across language, vision, audio, genomics, and virtually every other sequence modeling domain, yet its core mechanism—self-attention—has a **quadratic computational cost in sequence length** that makes it fundamentally unsuitable for modeling the long-range dependencies prevalent in real-world data.

This isn't a minor inefficiency. The quadratic scaling creates a hard ceiling on what Transformers can model. As the authors note in Section 1, self-attention's strength is that it "routes information densely within a context window," but this comes with an "inability to model anything outside of a finite window" and costs that grow quadratically with that window's length. For a sequence of length $L$, a Transformer's self-attention requires $O(L^2)$ computation and $O(L)$ memory (to store the key-value cache during autoregressive inference). This means:

- **Training** a Transformer on sequences of length 1M tokens is effectively impossible without specialized sparse or approximate attention schemes—the FLOP requirement would be roughly $10^{12}$ times larger than for a length-1K sequence.
- **Inference** requires storing the entire history of keys and values (the KV cache), which grows linearly with sequence length. For long conversations, long documents, or high-frequency sensor data, this cache can exceed the model's own parameter count in memory, making deployment on memory-constrained devices impractical.
- **The finite context window** means Transformers literally cannot attend to information outside it. Extending the window through architectural tricks helps, but doesn't address the underlying quadratic scaling problem.

This is important **both practically and theoretically**. Practically, many real-world domains *inherently* demand long-range reasoning: genomics (genes interact across millions of base pairs), audio generation (music has structure spanning minutes at 16kHz sample rates), video understanding (frames relate across thousands of timesteps), and even language (long documents, multi-turn conversations, codebases). A model that scales quadratically in sequence length is economically and environmentally unsustainable for these applications. Theoretically, the fact that the dominant architecture cannot handle sequences of arbitrary length suggests a fundamental architectural limitation—the field is building foundation models on a backbone that breaks at the very regime where interesting structure emerges.

### Prior Attempts to Fix This, and Why They Failed

An "enormous body of research" (the paper's words in Section 1) has attempted to create subquadratic attention variants. The approaches broadly fall into a few categories:

**Efficient attention approximations.** These include linear attention (Katharopoulos et al., 2020), which replaces the softmax with a kernelized form that can be computed as a recurrence ($O(L)$ instead of $O(L^2)$), and various sparse attention patterns (Child et al., 2019) that restrict which tokens can attend to which others. The survey by Tay, Dehghani, Bahri, et al. (2022) catalogs dozens of such approaches.

**Where they fall short:** These methods trade the very property that makes attention effective—dense, content-based information routing—for efficiency. The paper states bluntly that "as of yet, none of these variants have been shown to be empirically effective at scale across domains." Linear attention, for example, collapses the per-token attention distribution into a fixed-size recurrent state, losing the ability to precisely retrieve information from arbitrary past positions. The resulting models work well on some benchmarks but consistently underperform standard Transformers on language modeling, which remains the most important and challenging testbed for sequence models.

**Structured State Space Models (S4 and derivatives).** A more recent and promising alternative has been the family of structured state space models, initiated by S4 (Gu, Goel, and Ré, 2022). These models, described in detail in Section 2, are defined by a continuous-time dynamical system parametrized by matrices $(\Delta, \boldsymbol{A}, \boldsymbol{B}, \boldsymbol{C})$:

$$h'(t) = \boldsymbol{A}h(t) + \boldsymbol{B}x(t)$$
$$y(t) = \boldsymbol{C}h(t)$$

After discretization (converting the continuous system to a discrete recurrence), they can be computed in two equivalent ways: as a **linear recurrence** ($h_t = \overline{\boldsymbol{A}}h_{t-1} + \overline{\boldsymbol{B}}x_t$, $y_t = \boldsymbol{C}h_t$) or as a **global convolution** ($y = x * \overline{\boldsymbol{K}}$ where $\overline{\boldsymbol{K}} = (\boldsymbol{C}\overline{\boldsymbol{B}}, \boldsymbol{C}\overline{\boldsymbol{A}}\overline{\boldsymbol{B}}, \dots)$). This dual form is powerful: the convolutional mode enables efficient parallel training (since the whole kernel can be precomputed and applied via FFT), while the recurrent mode enables efficient autoregressive inference with constant-time per-step cost (no growing KV cache).

SSMs, including S4 and its descendants (DSS, S4D, S5, and architectures like H3, Hyena, and RetNet), have achieved strong results on domains involving continuous signals: audio generation, time series, and vision. They've dominated the Long Range Arena benchmark, which specifically tests models' ability to handle long-range dependencies.

**Where they fall short on discrete modalities.** The critical weakness—and the central motivation for this paper—is that all prior SSMs are **Linear Time-Invariant (LTI)**. The parameters $\Delta$, $\boldsymbol{A}$, $\boldsymbol{B}$, and $\boldsymbol{C}$ (and hence the discrete parameters $\overline{\boldsymbol{A}}$ and $\overline{\boldsymbol{B}}$) are **fixed across all time steps**. As the paper explains in Section 3.1, this means:
- From the recurrent perspective, the dynamics that govern how the hidden state evolves (the $\overline{\boldsymbol{A}}$ and $\overline{\boldsymbol{B}}$ matrices) are **identical** regardless of what the actual input tokens are. The model cannot decide to remember token A but forget token B based on their *content*—it applies the same transformation uniformly.
- From the convolutional perspective, the model learns a single fixed convolution kernel $\overline{\boldsymbol{K}}$ that is applied to the entire sequence. This kernel can learn patterns based on *relative position* (e.g., "look back 5 timesteps"), but cannot adapt its behavior based on *what those tokens actually are*.

This LTI limitation is devastating for discrete modalities like language and DNA, where **content-based reasoning** is fundamental. Language understanding requires the model to selectively attend to specific pieces of information based on their semantic content—remembering a character's name when it's introduced, filtering out filler words like "um" and "uh," and later retrieving that name when a pronoun refers back to it. DNA modeling requires identifying regulatory elements that may be separated from their target genes by millions of irrelevant base pairs. These are not tasks that a fixed convolution kernel or a state transition matrix with constant dynamics can solve.

The authors crystallize this in Section 3.1 through two synthetic tasks that serve as diagnostic probes:

**Selective Copying** (Figure 2, right top): The standard Copying task (Arjovsky, Shah, and Bengio, 2016) asks a model to memorize a set of input tokens and reproduce them later, with constant spacing. LTI models solve this easily because they can learn a convolution kernel of exactly the right length—it's just time-awareness. The Selective Copying variant randomizes the spacing between the tokens to memorize and the output positions. Now the model must *look at the content* of each token to decide whether to store it (is this a data token or a noise token?) and cannot rely on fixed relative positions. LTI models fail dramatically on this task (Table 1: S4 achieves only 18.3% accuracy, while the selective variant S6 achieves 97.0%).

**Induction Heads** (Figure 2, right bottom): This task, from the mechanistic interpretability literature (Olsson et al., 2022), tests associative recall: if the model has seen a pattern like "A, B" earlier in the sequence, then the next time it sees "A," it should predict "B." This requires content-aware retrieval—looking back through the history to find the previous occurrence of A and recalling what followed it. Olsson et al. (2022) hypothesized that induction heads explain the majority of in-context learning in large language models. LTI models cannot do this because they lack the mechanism to selectively query their history based on content; their recurrences or convolutions treat all past positions identically regardless of what tokens appeared there. Table 2 shows that while standard Transformers can solve this at training length (256), they fail to extrapolate beyond about 2× training length, whereas Mamba generalizes perfectly to 4000× training length (1M tokens).

### The Fundamental Tension: Efficiency vs. Effectiveness as State Compression

The paper frames the sequence modeling problem through a unifying lens in Section 3.1: **sequence modeling is fundamentally about compressing context into a state**. This perspective reveals the tradeoffs between different architectures:

- **Transformers (attention)**: Have no compression at all. The "state" is the entire KV cache, which stores every token's key and value vectors. This is why they're effective (no information is lost) but inefficient (the state grows linearly with sequence length).
- **Recurrent models (RNNs, SSMs)**: Compress the entire context into a fixed-size hidden state $h_t \in \mathbb{R}^N$. This is why they're efficient (constant computation per step, $O(1)$ inference) but their effectiveness is limited by **how well this fixed-size state can represent the context**.

The key insight is that the effectiveness of a recurrent model depends on whether it can perform **selective compression**—deciding what information to keep in the state and what to discard, based on the *content* of the inputs. An LTI model applies the same compression rule to every input indiscriminately. If you encounter a noise token (e.g., "um"), an LTI model writes it into the state just as it would write a critical fact. Over a long sequence, the state becomes cluttered with irrelevant information, and the signal from truly important tokens gets diluted.

The authors propose that **selectivity**—the context-aware ability to focus on or filter out inputs—is "a fundamental principle for building sequence models." This is what allows a fixed-size state to remain effective over arbitrarily long sequences: the model can clear its state when starting a new document, ignore filler tokens, and preserve information that it judges to be relevant. A selective model can in principle achieve both efficiency (constant-size state) and effectiveness (the state contains what matters), resolving the paradox that has kept Transformers dominant despite their scaling limitations.

### Positioning Relative to Existing Work

The paper positions Mamba as a synthesis and extension of several lines of prior work, while being careful to distinguish what is genuinely new.

**Relative to prior SSMs (S4, DSS, S4D, S5):** The break from LTI is the fundamental departure. All prior structured SSMs were time-invariant because their computational efficiency depended on the convolutional mode (3), which only works when $\overline{\boldsymbol{A}}$ and $\overline{\boldsymbol{B}}$ are constant across time. The paper's selection mechanism (making $\Delta$, $\boldsymbol{B}$, and $\boldsymbol{C}$ functions of the input $x_t$) breaks this equivalence—the resulting model is time-varying and can no longer be expressed as a convolution. This requires a new computational approach (the scan, Section 3.3), but unlocks content-dependent reasoning that LTI models fundamentally lack.

The paper particularly highlights S5 (Smith, Warrington, and Linderman, 2023) as the closest prior method. S5 also used a parallel scan for computation, but addressed the efficiency problem by switching from SISO (single-input single-output, where each of the $D$ channels has an independent SSM with state size $N$) to MIMO (multi-input multi-output, where multiple channels share an SSM, reducing the effective state dimension). The paper argues this is a concession: S5 had to reduce its state dimension to make the scan practical, whereas Mamba's hardware-aware algorithm (Section 3.3) keeps the full SISO state dimension $D \times N$, providing a larger effective recurrent state without paying the memory cost of materializing it.

**Relative to gated RNNs (LSTM, GRU, QRNN, SRU):** The paper explicitly connects the selection mechanism to classical RNN gating through Theorem 1. In the special case where $N = 1$, $\boldsymbol{A} = -1$, $\boldsymbol{B} = 1$, and specific choices of $s_\Delta$ and $\tau_\Delta$, the selective SSM reduces to:

$$g_t = \sigma(\text{Linear}(x_t))$$
$$h_t = (1 - g_t)h_{t-1} + g_t x_t$$

This is essentially a gated recurrent unit—a learned gate $g_t$ controls whether to remember the previous state or overwrite it with the current input. The paper argues that prior gated RNNs suffer from two limitations that Mamba addresses: (1) they use $N = 1$ (no state expansion), limiting the representational capacity of the hidden state, and (2) they don't have selective $\boldsymbol{B}$ and $\boldsymbol{C}$ parameters, which provide finer-grained control over what information enters and exits the state. The paper's ablations (Section 4.6, Tables 7 and 10) confirm that state expansion ($N > 1$) provides dramatic perplexity improvements (over 1.0 perplexity reduction for only 1% more parameters), but *only* when $\boldsymbol{B}$ and $\boldsymbol{C}$ are also selective—suggesting that prior gated RNNs were limited by both their small state and their lack of input-dependent input/output projections.

**Relative to linear attention and kernel methods:** Linear attention (Katharopoulos et al., 2020) and its derivatives (Performer, cosFormer, etc.) can be viewed as degenerate cases of SSMs. The paper's H3 architecture (Dao, Fu, Saab, et al., 2023) explicitly generalized linear attention to use S4-based recurrences. However, these methods inherit the LTI limitation: the recurrence or convolution kernel is data-independent, so they cannot perform selective filtering.

**Relative to recent "long-context" claims:** The paper is notably skeptical of recent work claiming to handle million-length sequences (Section B.5). It critiques Hyena and HyenaDNA for training on proportionally more data at longer contexts, making it unclear whether improvements come from more context or more computation, and critiques Recurrent Memory Transformer and LongNet for only demonstrating long-range capabilities on synthetic tasks or very short actual sequences. The paper positions Mamba's experiments as more carefully controlled: when comparing different sequence lengths in the DNA experiments (Section 4.3.2), the total number of training tokens per gradient step is held constant, so any improvement with longer context genuinely reflects the model's ability to leverage that context.

**The deliberate scope limitation:** The paper does not claim to have solved all sequence modeling problems. It explicitly acknowledges (Section 5) that the selection mechanism can *hurt* performance on continuous modalities like audio where LTI inductive biases are beneficial—the "no free lunch" across the continuous-discrete spectrum. The experiments in Section 4.4.1 show that for raw audio waveforms, the non-selective S4 actually outperforms S6 until the signal has been "tokenized" by outer layers of the network. This honest positioning—acknowledging where the method does *not* help—strengthens the paper's credibility and provides a nuanced picture of when selective SSMs are appropriate.

### The Gap This Paper Fills

In summary, the gap Mamba addresses is:

**No existing architecture simultaneously provides (1) linear-time scaling in sequence length, (2) content-based reasoning capability matching Transformers on discrete modalities like language, (3) efficient training on modern hardware, and (4) unbounded context length extrapolation.**

Transformers provide (2) and (3) but fail at (1) and (4). Linear attention variants provide (1) and (3) but fail at (2). LTI SSMs provide (1), (3), and (4) but fail at (2). Gated RNNs provide (1) and (4) but fail at (2) and (3) at scale.

Mamba's thesis is that **selectivity is the missing ingredient** that enables recurrent models to perform content-based reasoning, and that **hardware-aware algorithm design** is the missing ingredient that makes selective recurrent models efficient on modern GPUs. The combination of these two ideas, embedded in a simplified architecture that merges the SSM and MLP blocks, produces the first linear-time sequence model that genuinely matches Transformer quality on language while retaining the long-context advantages of recurrent models. The paper's empirical results across language, DNA, and (to a more qualified extent) audio provide evidence for this thesis, though at model scales (up to 2.8B parameters) that are substantial but still below the frontier of very large language models—a limitation the authors acknowledge as open for future work.

## 3. Technical Approach

### 3.1 Reader Orientation

The paper builds a **selective state space model (S6)** — a recurrent neural network layer that processes sequences by maintaining a compressed hidden state, but crucially, the rules for how information enters, persists in, and exits that state are **functions of the input tokens themselves**, not fixed constants. This single change—making the SSM parameters input-dependent—addresses the fundamental limitation of all prior structured state space models: the inability to perform **content-based reasoning**, i.e., to decide what to remember and what to forget based on what the tokens actually *say* rather than just where they *are* in the sequence. Because this change breaks the mathematical equivalence to convolution that made prior SSMs efficient, the paper designs a **hardware-aware parallel scan algorithm** that recomputes the expanded state in fast GPU SRAM rather than materializing it in slow HBM, making the selective model both more expressive than LTI SSMs and faster than naive recurrent implementations.

### 3.2 Big-Picture Architecture (Diagram in Words)

The system has five major components:

1. **Input-dependent parameter generators**: three small neural networks (`$s_B$`, `$s_C$`, `$s_\Delta$`) that take the current token's representation as input and output the SSM parameters `$\boldsymbol{B}$`, `$\boldsymbol{C}$`, and `$\Delta$` for that specific timestep. This is what breaks time-invariance — the parameters now vary at every position.

2. **The structured `$\boldsymbol{A}$` matrix**: a fixed, learned diagonal matrix that governs how the hidden state decays over time. It is NOT input-dependent (in the default Mamba), but interacts with the input-dependent `$\Delta$` through discretization to produce an effective transition matrix `$\overline{\boldsymbol{A}}$` that *is* input-dependent.

3. **The discretization step**: a mathematical transformation that converts the continuous-time parameters `$(\Delta, \boldsymbol{A}, \boldsymbol{B})$` into discrete-time parameters `$(\overline{\boldsymbol{A}}, \overline{\boldsymbol{B}})$` using the zero-order hold (ZOH) rule. This step is performed on-the-fly for each timestep, within the fast SRAM of the GPU.

4. **The parallel scan**: a work-efficient parallel algorithm that computes the linear recurrence `$h_t = \overline{\boldsymbol{A}}_t h_{t-1} + \overline{\boldsymbol{B}}_t x_t$` for all timesteps simultaneously, without waiting for `$h_{t-1}$` to compute `$h_t$` sequentially. This is implemented as a fused CUDA kernel that operates entirely in SRAM.

5. **The Mamba block**: a simplified neural network architecture that combines the selective SSM with a gated linear unit (GLU) and a short convolution into a single homogeneous block, replacing both the attention and the MLP blocks of a Transformer.

**Information flow for a single input sequence `$x$` of shape `$(B, L, D)$` (batch size, sequence length, model dimension):**

1. **The input `$x$` is projected into two branches.** One branch becomes the SSM input after a 1D convolution; the other branch passes through a SiLU activation to become a multiplicative gate.

2. **The SSM branch computes `$\Delta_t$`, `$\boldsymbol{B}_t$`, and `$\boldsymbol{C}_t$` for each position `$t$`** using small learned linear projections of `$x_t$`. These are loaded into GPU SRAM.

3. **In SRAM, the discretization formulas are applied** to produce `$\overline{\boldsymbol{A}}_t$` and `$\overline{\boldsymbol{B}}_t$` for each position.

4. **The parallel scan executes in SRAM**, consuming `$\overline{\boldsymbol{A}}_t$`, `$\overline{\boldsymbol{B}}_t$`, and `$x_t$` to produce the hidden states `$h_t$` and outputs `$y_t = \boldsymbol{C}_t h_t$` for all positions, without ever writing the expanded intermediate states (of size `$B \times L \times D \times N$`) to HBM.

5. **The SSM output `$y$` is multiplied elementwise with the gate** from step 1, then projected back to dimension `$D$`.

6. **A residual connection adds the original input**, and the result passes to the next Mamba block. This block is repeated `$L$` times (homogeneously, without interleaving separate MLP or attention blocks) to form the full Mamba architecture.

### 3.3 Roadmap for the Deep Dive

- **First, the selection mechanism** (Section 3.2 of the paper): how making `$\Delta$`, `$\boldsymbol{B}$`, and `$\boldsymbol{C}$` input-dependent transforms an LTI SSM into a time-varying selective SSM, why this breaks the convolutional equivalence, and what specific functional forms are used.

- **Second, the hardware-aware scan algorithm** (Section 3.3): how the time-varying recurrence is computed efficiently on GPUs using kernel fusion, parallel scan, and recomputation, and why naive approaches would be prohibitively slow or memory-intensive.

- **Third, the Mamba architecture** (Section 3.4): the simplified block design that merges the SSM with a gated MLP into a single homogeneous unit, including the specific parameter counts, normalization choices, and activation functions.

- **Fourth, the theoretical connections and interpretations** (Section 3.5): Theorem 1 connecting selective SSMs to gated RNNs, and the mechanistic interpretations of what `$\Delta$`, `$\boldsymbol{A}$`, `$\boldsymbol{B}$`, and `$\boldsymbol{C}$` actually *do* in terms of information flow control.

- **Fifth, the additional design choices** (Section 3.6): the decisions around real vs. complex parameters, initialization schemes, and the dimensionality of the `$\Delta$` projection.

### 3.4 Detailed, Sentence-Based Technical Breakdown

This is primarily a **systems and methods paper** whose core idea is that making structured state space models **input-dependent (selective)** enables content-based reasoning on discrete modalities, and that this change can be made efficient through a **hardware-aware implementation** that exploits the GPU memory hierarchy.

---

#### The Core SSM Formulation (from LTI to Selective)

The starting point is the standard continuous-time state space model defined in equations (1a)-(1b):

$$h'(t) = \boldsymbol{A}h(t) + \boldsymbol{B}x(t)$$
$$y(t) = \boldsymbol{C}h(t)$$

where `$x(t) \in \mathbb{R}$` is a 1-dimensional input signal at continuous time `$t$`, `$h(t) \in \mathbb{R}^N$` is an N-dimensional latent state, `$y(t) \in \mathbb{R}$` is the output, `$\boldsymbol{A} \in \mathbb{R}^{N \times N}$` is the state transition matrix, `$\boldsymbol{B} \in \mathbb{R}^{N \times 1}$` is the input projection vector, and `$\boldsymbol{C} \in \mathbb{R}^{1 \times N}$` is the output projection vector.

**What it computes:** This is a linear dynamical system mapping a scalar input signal `$x(t)$` to a scalar output signal `$y(t)$` through an N-dimensional hidden state `$h(t)$`. The state evolves continuously according to the differential equation `$h' = \boldsymbol{A}h + \boldsymbol{B}x$`, and the output is a linear readout `$y = \boldsymbol{C}h$`. Think of it as a continuous-time analogue of an RNN: the state integrates information from inputs over time, and the output is a linear function of that integrated state.

**Why this form:** The continuous-time formulation provides a principled connection to dynamical systems theory and enables the use of discretization rules that endow the model with useful properties like resolution invariance (the model's behavior is consistent regardless of the sampling rate of the input) and proper normalization.

**Discretization into a recurrence.** The continuous system (1a)-(1b) cannot be directly computed on discrete sequences. The first transformation step converts the continuous parameters `$(\Delta, \boldsymbol{A}, \boldsymbol{B})$` to discrete parameters `$(\overline{\boldsymbol{A}}, \overline{\boldsymbol{B}})$` via a discretization rule. The paper uses the **zero-order hold (ZOH)** rule defined in equation (4):

$$\overline{\boldsymbol{A}} = \exp(\Delta \boldsymbol{A})$$
$$\overline{\boldsymbol{B}} = (\Delta \boldsymbol{A})^{-1}(\exp(\Delta \boldsymbol{A}) - \boldsymbol{I}) \cdot \Delta \boldsymbol{B}$$

where `$\Delta \in \mathbb{R}_{>0}$` is a step size parameter representing the temporal resolution of the discretization, `$\exp$` is the matrix exponential, `$\boldsymbol{I}$` is the identity matrix, and `$\overline{\boldsymbol{A}} \in \mathbb{R}^{N \times N}$` and `$\overline{\boldsymbol{B}} \in \mathbb{R}^{N \times 1}$` are the resulting discrete-time parameters.

**What it computes:** The continuous-time evolution `$h' = \boldsymbol{A}h + \boldsymbol{B}x$` over a time interval of length `$\Delta$` is approximated by a discrete jump: `$h_t = \overline{\boldsymbol{A}} h_{t-1} + \overline{\boldsymbol{B}} x_t$`. The ZOH rule assumes that the input `$x(t)$` is held constant over the interval `$\Delta$`. For a diagonal `$\boldsymbol{A}$` (which the paper uses), the matrix exponential simplifies to elementwise exponentiation of the diagonal entries, making the computation efficient.

**Why this form:** The ZOH discretization has deep connections to continuous-time systems. It ensures that the discrete model approximates the continuous dynamics, and it provides a principled connection to RNN gating mechanisms (discussed in Section 3.5). Importantly, the step size `$\Delta$` controls the *timescale* of the discretization: a large `$\Delta$` means the system "jumps" further in continuous time per discrete step, effectively making the model focus more on the current input.

**The dual form: recurrence and convolution.** After discretization, the model can be computed in two equivalent ways, shown in equations (2a)-(2b) and (3a)-(3b):

**Recurrent form:**
$$h_t = \overline{\boldsymbol{A}}h_{t-1} + \overline{\boldsymbol{B}}x_t$$
$$y_t = \boldsymbol{C}h_t$$

**Convolutional form:**
$$\overline{\boldsymbol{K}} = (\boldsymbol{C}\overline{\boldsymbol{B}}, \boldsymbol{C}\overline{\boldsymbol{A}}\overline{\boldsymbol{B}}, \dots, \boldsymbol{C}\overline{\boldsymbol{A}}^{L-1}\overline{\boldsymbol{B}})$$
$$y = x * \overline{\boldsymbol{K}}$$

where `$\overline{\boldsymbol{K}} \in \mathbb{R}^L$` is the SSM convolution kernel, `$*$` denotes convolution, and `$L$` is the sequence length.

**What the recurrence computes:** For each timestep `$t$`, the new hidden state `$h_t$` is a linear combination of the previous state `$h_{t-1}$` (transformed by `$\overline{\boldsymbol{A}}$`) and the current input `$x_t$` (projected by `$\overline{\boldsymbol{B}}$`). The output `$y_t$` is a linear readout of the current state. This requires `$O(L)$` sequential steps but only `$O(1)$` computation and memory per step, making it ideal for autoregressive inference.

**What the convolution computes:** By unrolling the recurrence, the entire output sequence `$y$` can be expressed as a convolution of the input `$x$` with a precomputed kernel `$\overline{\boldsymbol{K}}$`. Since convolution can be computed in `$O(L \log L)$` time via FFT and is highly parallelizable, this form is ideal for training where the whole sequence is available.

**Why the dual form matters:** This is the key efficiency property of LTI SSMs. The equivalence between the recurrence and the convolution **depends on `$\overline{\boldsymbol{A}}$` and `$\overline{\boldsymbol{B}}$` being constant across time**. If they vary with `$t$`, the unrolling (3a) no longer holds — there is no single convolution kernel `$\overline{\boldsymbol{K}}$` that works for all positions. Prior SSMs exploited this duality to train with convolutions (fast) and infer with recurrences (memory-efficient).

---

#### The Selection Mechanism: Breaking Time-Invariance

The central technical contribution is making several of the SSM parameters **functions of the input**, transforming the LTI model into a time-varying (selective) model. This is shown in Algorithm 2 of the paper, contrasted with the LTI Algorithm 1.

**Algorithm 1 (LTI S4):**
- `$\Delta \in \mathbb{R}^D$` is a learned parameter (one scalar per channel), broadcast across sequence length.
- `$\boldsymbol{B} \in \mathbb{R}^{D \times N}$` and `$\boldsymbol{C} \in \mathbb{R}^{D \times N}$` are learned parameter matrices.
- The discrete parameters `$\overline{\boldsymbol{A}}, \overline{\boldsymbol{B}}$` are computed once and are constant across all `$L$` timesteps.
- The model can be computed as either a recurrence or a convolution.

**Algorithm 2 (Selective S6):**
- `$\boldsymbol{B} \in \mathbb{R}^{B \times L \times N}$` is computed as `$s_B(x)$`, a learned function of the input.
- `$\boldsymbol{C} \in \mathbb{R}^{B \times L \times N}$` is computed as `$s_C(x)$`, a learned function of the input.
- `$\Delta \in \mathbb{R}^{B \times L \times D}$` is computed as `$\tau_\Delta(\text{Parameter} + s_\Delta(x))$`, a learned function of the input plus a learned bias.
- The discrete parameters `$\overline{\boldsymbol{A}}_t, \overline{\boldsymbol{B}}_t$` are now different for each timestep `$t$`.
- The model can ONLY be computed as a recurrence (no convolutional equivalence).

**The specific functional forms chosen (Section 3.2):**

$$s_B(x) = \text{Linear}_N(x)$$
$$s_C(x) = \text{Linear}_N(x)$$
$$s_\Delta(x) = \text{Broadcast}_D(\text{Linear}_1(x))$$
$$\tau_\Delta = \text{softplus}$$

where `$\text{Linear}_d$` is a learned linear projection to dimension `$d$`, `$\text{Broadcast}_D$` replicates a scalar across all `$D$` channels, and `$\text{softplus}(z) = \log(1 + e^z)$` ensures `$\Delta$` is strictly positive.

**What these functions compute:**

- `$s_B(x_t)$` takes the current token's representation `$x_t \in \mathbb{R}^D$` and projects it to an `$N$`-dimensional vector `$\boldsymbol{B}_t$`. This means the input projection—how much each dimension of `$x_t$` is written into each dimension of the state `$h_t$`—depends on what `$x_t$` actually contains. If `$x_t$` represents a noise token, `$\boldsymbol{B}_t$` can be near zero, effectively preventing it from entering the state.

- `$s_C(x_t)$` similarly produces `$\boldsymbol{C}_t$`, an `$N$`-dimensional vector that controls which state dimensions are read out to produce `$y_t$`. This means the readout can be context-dependent: the same state content can produce different outputs depending on the current token.

- `$s_\Delta(x_t)$` first projects `$x_t$` to a **scalar** (dimension 1), then broadcasts this scalar to all `$D$` channels and applies a softplus to ensure positivity. The scalar projection means that `$\Delta$` is shared across all `$N$` state dimensions for a given channel, but varies per channel and per timestep.

**Why `$\Delta$` is projected to 1 dimension:** As explained in Section 3.5.1, this choice is motivated by the gating mechanism of RNNs. The intuition is that if a token should be ignored, it should be ignored across all channels of the state simultaneously. Projecting to a scalar forces the model to make a single ignore/attend decision per token-channel pair, rather than allowing some state dimensions to incorporate the token while others don't. The paper notes that this can be generalized to a larger dimension `$R$` (where `$R$` is a small fraction of `$D$`), creating a low-rank projection `$s_\Delta(x) = \text{Linear}_D(\text{Linear}_R(x))$`, which provides slightly more expressivity at negligible parameter cost (Table 9 shows modest improvements as `$R$` increases from 1 to 64).

**The shape implications (critical for understanding efficiency):** In Algorithm 2, `$\Delta$` now has shape `$(B, L, D)$` instead of `$(D)$`, `$\boldsymbol{B}$` has shape `$(B, L, N)$` instead of `$(D, N)$`, and `$\boldsymbol{C}$` has shape `$(B, L, N)$` instead of `$(D, N)$`. After discretization, `$\overline{\boldsymbol{A}}$` and `$\overline{\boldsymbol{B}}$` have shape `$(B, L, D, N)$` — introducing a length dimension that previously did not exist. The hidden state `$h$` and the output `$y$` still have shapes `$(B, L, D, N)$` and `$(B, L, D)$` respectively, but now computing `$h$` requires materializing a `$(B, L, D, N)$` tensor at some point in the computation, which is `$N$` times larger than the input and output tensors. This is the computational challenge that the scan algorithm addresses.

**Why this breaks the convolutional equivalence:** A convolution kernel is defined as a fixed pattern that is applied uniformly across the input. When `$\overline{\boldsymbol{A}}$` and `$\overline{\boldsymbol{B}}$` vary per timestep, equation (3a) no longer holds — you cannot "unroll" the recurrence into a single kernel because each position has different transition dynamics. The model is fundamentally recurrent and must be computed as such. This is both the source of its power (it can now do content-dependent reasoning) and the source of the implementation challenge (recurrences are sequential and memory-intensive on GPUs).

---

#### Hardware-Aware Scan Algorithm (Section 3.3)

The selection mechanism creates a computational problem: the model must be computed recurrently, but a naive recurrent implementation would be both slow (sequential operations per timestep) and memory-hungry (materializing the state of size `$B \times L \times D \times N$`). The paper's solution combines three ideas: **parallel scan, kernel fusion, and recomputation**.

**The parallel scan.** The recurrence `$h_t = \overline{\boldsymbol{A}}_t h_{t-1} + \overline{\boldsymbol{B}}_t x_t$` is a **first-order linear recurrence** (each `$h_t$` depends only on `$h_{t-1}$` and the current inputs). Such recurrences can be parallelized using the **parallel prefix sum (scan)** algorithm. The key insight is that the operation `$(a, b) \circ (c, d) = (a \cdot c, a \cdot d + b)$` is associative when applied to pairs representing the linear recurrence. Under this associative operator, the recurrence can be computed in `$O(\log L)$` parallel steps using a binary tree reduction, rather than `$O(L)$` sequential steps.

The paper leverages this by implementing a **work-efficient parallel scan** (Blelloch, 1990) that operates on the SSM parameters. This is the same algorithmic approach used by S5 (Smith, Warrington, and Linderman, 2023), but with a crucial difference: S5 reduced the state dimension by switching to MIMO SSMs to make the scan practical, while Mamba keeps the full SISO dimension and uses hardware-aware optimizations to handle the larger state.

**Kernel fusion (the core optimization).** The key observation is that most operations in the SSM computation (except matrix multiplications) are **memory-bandwidth-bound**, not compute-bound. Reading and writing the intermediate state tensor of size `$(B, L, D, N)$` from/to GPU HBM (high-bandwidth memory, the main GPU memory) would dominate the runtime. The solution: **fuse the discretization, scan, and output multiplication into a single CUDA kernel that operates in SRAM (static random-access memory, the on-chip cache)**.

The fused kernel performs these operations, quoted from Appendix D:

1. **Read** the SSM parameters `$\Delta$`, `$\boldsymbol{A}$`, `$\boldsymbol{B}$`, `$\boldsymbol{C}$` from HBM into SRAM. The total bytes read is `$O(BLD + DN)$` — the `$\boldsymbol{B}$` and `$\boldsymbol{C}$` parameters scale with `$L$` but NOT with `$N$` multiplied by `$L$`.

2. **Discretize** in SRAM to produce `$\overline{\boldsymbol{A}}$` and `$\overline{\boldsymbol{B}}$` of size `$(B, L, D, N)$`. These are computed and immediately consumed, never written to HBM.

3. **Perform the parallel scan** in SRAM, yielding intermediate states of size `$(B, L, D, N)$`. Again, these are kept in fast SRAM.

4. **Multiply and sum** with `$\boldsymbol{C}$` to produce outputs of size `$(B, L, D)$`, and write only the final output to HBM.

**What this accomplishes:** By keeping the large intermediate tensors in SRAM, the IO (input/output) to HBM is reduced by a factor of `$O(N)$` (the state dimension) compared to a naive implementation that would write and read the `$(B, L, D, N)$` state tensor. In practice, this yields a **20–40× speedup** compared to a standard PyTorch implementation (Figure 8, left). For `$N = 16$` (the default), this is a 16× reduction in memory bandwidth consumption.

**Handling long sequences with chunking.** SRAM is much smaller than HBM (typically hundreds of KB vs. tens of GB). For very long sequences that cannot fit entirely in SRAM, the kernel splits the sequence into chunks, processes each chunk with the fused scan, and uses the intermediate scan state at the chunk boundary to continue the scan with the next chunk.

**Recomputation for the backward pass.** The fused forward pass does not save the intermediate states of size `$(B, L, D, N)$` to avoid memory consumption. However, these states are needed during backpropagation to compute gradients. The solution: **recompute them in the backward pass**. Since the inputs `$\Delta$`, `$\boldsymbol{A}$`, `$\boldsymbol{B}$`, `$\boldsymbol{C}$` (of size `$O(BLN + DN)$`) are already stored from the forward pass, and the output gradient (also `$O(BLD)$`) is provided by the autograd engine, the intermediate states can be recomputed from scratch during the backward pass.

**Why recomputation is efficient here:** Reading the saved intermediate states from HBM would cost `$O(BLND)$` bytes of bandwidth. Recomputing them costs additional FLOPs but only requires reading the much smaller inputs `$O(BLN + DN)$` from HBM. Since the scan is memory-bandwidth-bound (not compute-bound), avoiding the large HBM reads outweighs the cost of recomputation.

**Memory comparison to Transformers.** The paper reports (Appendix D) that each selective SSM layer stores approximately 16 bytes of activations per token during training (assuming BF16/FP16 mixed precision). For comparison, FlashAttention stores about 12 bytes per token, and an MLP block stores about 20 bytes per token, for a total of 32 bytes per Transformer layer. Two Mamba blocks (which replace one Transformer attention+MLP pair) store about 32 bytes total — comparable to the highly optimized Transformer. Table 15 confirms that at 125M parameters with batch size 1 and sequence length 2048, Mamba uses 4.8GB vs. 4.6GB for an optimized Transformer.

**FLOP comparison (recurrent vs. convolutional).** The paper notes (Section 3.3.2) that the recurrent computation uses `$O(BLDN)$` FLOPs, while the convolutional computation uses `$O(BLD \log L)$` FLOPs. For long sequences and moderate state dimension `$N$` (typically 16), the recurrent mode actually uses **fewer** total FLOPs than the convolution. The bottleneck is memory bandwidth, not FLOPs, which is why the kernel fusion is so effective.

---

#### The Mamba Architecture (Section 3.4)

Selective SSMs are standalone sequence transformations; they must be embedded in a neural network architecture. The paper simplifies prior SSM architectures (specifically H3) into a homogeneous block design.

**The H3 block as starting point.** The H3 architecture (Dao, Fu, Saab, et al., 2023) consists of a block with three components: a linear projection followed by a multiplicative gate, an SSM layer, and another multiplicative gate. This is interleaved with standard MLP blocks (linear projection → nonlinearity → linear projection). The total parameter count per "layer" is roughly `$12D^2$` (the same as a Transformer's MHA + MLP blocks together).

**The Mamba simplification.** The Mamba block, shown in Figure 3, merges the H3-style SSM block with the MLP block into a single homogeneous unit:

1. **Input projection:** The input `$x$` is projected to dimension `$E \times D$` where `$E$` is an expansion factor (fixed to `$E = 2$`). This uses `$2ED^2 = 4D^2$` parameters.

2. **Branch split:** The expanded representation is split into two branches:
   - **Main branch:** Passes through a 1D convolution (for local context), a SiLU activation, the selective SSM, and an optional LayerNorm.
   - **Gating branch:** Passes through a SiLU activation only.

3. **Elementwise multiplication:** The SSM output and the gating branch output are multiplied elementwise. This is the "gated" aspect — the gate controls which SSM outputs are passed forward.

4. **Output projection:** A linear projection back to dimension `$D$`, using `$ED^2 = 2D^2$` parameters.

5. **Residual connection:** The output is added to the block's input.

The block incorporates a **short 1D convolution** before the SSM, following H3. The authors describe this as a "shift-SSM" — a local convolution can be viewed as a very simple SSM — and note that it provides local context mixing that complements the SSM's long-range capabilities.

**Parameter count.** The block uses roughly `$6D^2$` parameters in the linear projections (input `$4D^2$` + output `$2D^2$`). The SSM itself contributes very few parameters: the `$\boldsymbol{A}$` matrix is `$D \times N$` (diagonal, so just `$D \times N$` numbers), and the projections for `$\Delta$`, `$\boldsymbol{B}$`, `$\boldsymbol{C}$` involve small linear layers (e.g., `$\text{Linear}_1$` for `$\Delta$` has negligible parameter count). Two Mamba blocks are stacked to match the `$12D^2$` parameter count of one Transformer layer (which has `$4D^2$` for MHA + `$8D^2$` for the MLP).

**Why homogeneous stacking?** The design is inspired by the Gated Attention Unit (GAU) (Hua et al., 2022), which combined attention and MLP blocks into a single unit. The paper's key departure is replacing the attention mechanism in GAU with the selective SSM. The homogeneous design is simpler (no need to alternate different block types), and conceptually unifies the roles of "mixing information across positions" (what attention/SSMs do) and "processing information per position" (what MLPs do) into a single operation.

**Activation function.** The SiLU (Sigmoid Linear Unit, also called Swish) activation `$\text{SiLU}(x) = x \cdot \sigma(x)$` is used throughout. When combined with the gating branch, this effectively creates a SwiGLU variant: one branch provides the "value" (post-SSM), and the SiLU-activated gating branch provides the "gate," similar to the SwiGLU activation used in PaLM and LLaMa.

**Optional LayerNorm.** Following RetNet (Sun et al., 2023), an optional LayerNorm is inserted after the SSM output and before the multiplicative gate. This is a minor architectural detail that the authors found helpful.

**What the specialization achieves:** In a Transformer, the MHA block is responsible for mixing information across positions (via attention), while the MLP block processes each position independently (via two linear projections with a nonlinearity). The Mamba block does BOTH simultaneously: the SSM handles cross-position mixing (with content-dependent selectivity), while the gating and projection structure handles per-position processing. The result is a simpler, more uniform architecture.

---

#### Theoretical Connections: Selection as Generalized Gating (Section 3.5)

**Theorem 1: Reduction to Gated RNN.** When the state dimension `$N = 1$`, the transition matrix is `$\boldsymbol{A} = -1$`, the input vector is `$\boldsymbol{B} = 1$`, `$s_\Delta = \text{Linear}_1(x)$`, and `$\tau_\Delta = \text{softplus}$`, the selective SSM recurrence simplifies to:

$$g_t = \sigma(\text{Linear}(x_t))$$
$$h_t = (1 - g_t)h_{t-1} + g_t x_t$$

where `$\sigma$` is the sigmoid function.

**What this equation describes:** This is a classic gated recurrent unit. At each timestep, a gate `$g_t \in [0, 1]$` is computed from the input. The hidden state is updated as an interpolation between the previous state `$h_{t-1}$` (weighted by `$1 - g_t$`) and the current input `$x_t$` (weighted by `$g_t$`). When `$g_t \approx 0$`, the model ignores `$x_t$` and preserves its state (forget gate closed). When `$g_t \approx 1$`, the model overwrites its state with `$x_t$` (forget gate open, or equivalently, reset).

**How this follows from the general formulation:** The proof (Appendix C) works through the ZOH discretization. With `$\boldsymbol{A} = -1$`, we have `$\overline{\boldsymbol{A}}_t = \exp(-\Delta_t) = 1 / (1 + \exp(\text{Linear}(x_t))) = \sigma(-\text{Linear}(x_t)) = 1 - \sigma(\text{Linear}(x_t))$`. Similarly, `$\overline{\boldsymbol{B}}_t = 1 - \overline{\boldsymbol{A}}_t = \sigma(\text{Linear}(x_t))$`. Substituting into the recurrence `$h_t = \overline{\boldsymbol{A}}_t h_{t-1} + \overline{\boldsymbol{B}}_t x_t$` gives the gated form above with `$g_t = \sigma(\text{Linear}(x_t))$`.

**Why this connection matters:**
- It explains **why `$\Delta$` is the most important selective parameter** (confirmed by ablation in Table 7). `$\Delta$` directly controls the gate `$g_t$` through the discretization, providing the primary mechanism for selective forgetting and remembering.
- It shows that **selective SSMs generalize gated RNNs**. Classic RNNs like LSTM and GRU have gating but with `$N = 1$` (scalar state per channel). Mamba generalizes this to `$N > 1$` (vector state per channel), providing much richer state representations. The ablation in Table 10 confirms that state expansion provides large perplexity gains, but only when `$\boldsymbol{B}$` and `$\boldsymbol{C}$` are also selective.
- It provides a **principled derivation** of the gating mechanism, rather than the heuristic design of LSTMs and GRUs. The specific functional form (softplus for `$\Delta$`, sigmoid for the effective gate) emerges from the ZOH discretization of a continuous-time system, rather than being hand-designed.

**Mechanistic interpretations of the selective parameters (Section 3.5.2):**

The paper elaborates three particular effects that the selection mechanism enables, and interprets what each parameter controls:

**Variable spacing (filtering out irrelevant tokens).** This is the ability to ignore noise tokens—words like "um," whitespace, or any content the model judges irrelevant. Mechanistically, this happens when the model sets the effective gate `$g_t \approx 0$` (via large `$\Delta_t$`), which means `$\overline{\boldsymbol{B}}_t \approx \boldsymbol{0}$` and the input `$x_t$` does not enter the state. This directly solves the Selective Copying task, where the model must skip over an unpredictable number of blank tokens between the data tokens.

**Filtering context (resetting state to ignore irrelevant history).** LTI models process every token equally, so if a long sequence contains a section of irrelevant text, that irrelevant text still gets written into the state and can interfere with later processing. A selective model can set `$\overline{\boldsymbol{A}}_t \approx \boldsymbol{0}$` (via small `$\Delta_t$`, giving `$g_t \approx 1$`), which means `$h_t \approx \overline{\boldsymbol{B}}_t x_t$` — the entire state history is discarded and replaced. This explains why Mamba's DNA pretraining perplexity *improves* with longer sequences (Figure 5, Right) while HyenaDNA's degrades: Mamba can reset its state when transitioning between unrelated genomic regions, while HyenaDNA's fixed convolution kernel aggregates noise across the entire length.

**Boundary resetting (handling multiple independent sequences stitched together).** When training on packed sequences (multiple documents concatenated to fill a batch), Transformers use attention masks to prevent cross-document attention. LTI models bleed information across document boundaries. Selective SSMs can automatically reset at boundaries by setting `$g_t \approx 1$` (large `$\Delta_t$`), effectively starting fresh for each new document.

**Interpretation of `$\Delta$`:** Controls the **balance between focusing on current input vs. preserving state**. Large `$\Delta$` → the system "focuses" longer on the current input (in continuous time), which after discretization means it resets the state and prioritizes `$x_t$`. Small `$\Delta$` → the input is transient (quickly passed over in continuous time), meaning the state is preserved and `$x_t$` is largely ignored. This generalizes RNN gates: `$\Delta$` plays the role of the forget gate (or its complement).

**Interpretation of `$\boldsymbol{A}$`:** The paper keeps `$\boldsymbol{A}$` non-selective (a learned diagonal matrix with fixed real values) and argues this is sufficient because `$\boldsymbol{A}$` affects the model only through its interaction with `$\Delta$` in the discretization `$\overline{\boldsymbol{A}} = \exp(\Delta \boldsymbol{A})$`. Since `$\Delta$` is already selective, `$\overline{\boldsymbol{A}}$` becomes selective through this multiplication, so making `$\boldsymbol{A}$` itself selective would be redundant. This is a design choice to minimize the number of selective parameters and keep the model simple.

**Interpretation of `$\boldsymbol{B}$` and `$\boldsymbol{C}$`:** While `$\Delta$` provides coarse-grained selectivity (should this token be noticed at all?), `$\boldsymbol{B}$` and `$\boldsymbol{C}$` provide **fine-grained control** over *how* the token's information is written into and read from the state. `$\boldsymbol{B}_t$` is an `$N$`-dimensional projection of `$x_t$`, controlling which state dimensions receive the input and with what strength. `$\boldsymbol{C}_t$` is an `$N$`-dimensional vector controlling which state dimensions are read out. This allows the model to have input-dependent write and read operations: the same state content can be interpreted differently depending on the current query token. Table 7 shows that making all three (`$\Delta$`, `$\boldsymbol{B}$`, `$\boldsymbol{C}$`) selective gives the best perplexity (8.71 vs. 10.93 for none selective), validating that each contributes.

---

#### Additional Model Details (Section 3.6)

**Real vs. Complex parameters.** Prior SSMs (S4, S4D-Lin) used complex numbers in their state and parameters, motivated by the HIPPO theory and the need to model continuous signals like audio. The paper defaults to **real-valued** parameters and states. The rationale: complex numbers help for continuous modalities (audio, video) where the continuous-time dynamical system interpretation is meaningful, but for discrete modalities (text, DNA), real-valued SSMs are equally or more effective. The DNA experiments (Figure 5) and language experiments (Table 6, where S4-real matches S4-complex in perplexity) support this. The exception is the audio pretraining experiment (Section 4.4.1), where complex parameters are used—the one domain where the continuous-time inductive bias remains beneficial.

**Initialization of `$\boldsymbol{A}$`.** The `$n$`-th diagonal element of `$\boldsymbol{A}$` determines the timescale of the `$n$`-th state dimension. The paper primarily uses **S4D-Real** initialization, which sets `$\boldsymbol{A}_n = -(n + 1)$` for `$n = 0, 1, \dots, N-1$`. This is based on the HIPPO theory and provides a geometrically spaced set of timescales (the first dimension has timescale 1, the second has timescale 1/2, the third 1/3, etc.). Table 8 shows that simpler initializations (S4D-Real, or even random initialization `$\boldsymbol{A}_n \sim \exp(\mathcal{N}(0, 1))$`) work as well or better than the more standard complex-valued S4D-Lin initialization for language modeling, consistent with the finding that complex numbers are unnecessary for discrete modalities.

**Parameterization of `$\Delta$`'s input projection.** The default `$s_\Delta(x) = \text{Broadcast}_D(\text{Linear}_1(x))$` projects `$x$` to a single scalar per channel. The paper notes this can be generalized: `$s_\Delta(x) = \text{Linear}_D(\text{Linear}_R(x))$` where `$R$` is a small rank. This is a low-rank projection: first project `$x$` to `$R$` dimensions, then project to `$D$` dimensions. Table 9 ablates this, showing that `$R = 1$` already provides a large improvement over a non-selective (`$R = 0$`) model, and increasing `$R$` to 64 provides further modest gains (perplexity 8.71 vs. 8.97 for `$R = 1$`, at the cost of about 12M additional parameters for the 360M model). The default implementation uses `$R$` as a small fraction of `$D$`.

**Initialization of `$\Delta$`'s bias.** The parameter component of `$\Delta$` (before the input-dependent adjustment) is initialized to `$\tau_\Delta^{-1}(\text{Uniform}([0.001, 0.1]))$` — that is, the inverse softplus of a uniformly random small positive number. This ensures that the initial effective `$\Delta$` values are small (on the order of 0.001 to 0.1), meaning the model starts in a regime where it largely preserves its state (small gates, few resets) and gradually learns when to selectively forget. This follows prior work on SSM initialization (Gu, Johnson, Timalsina, et al., 2023).

**State dimension `$N$`.** The default state dimension is `$N = 16$`. Table 10 shows that increasing `$N$` from 1 to 16 provides over 1.0 perplexity improvement (from 9.88 to 8.71) while increasing parameters by only 1% (from 367.1M to 371.5M). Critically, this improvement **only occurs when `$\boldsymbol{B}$` and `$\boldsymbol{C}$` are also selective**: with constant `$\boldsymbol{B}$` and `$\boldsymbol{C}$`, increasing `$N$` from 1 to 16 gives minimal improvement (9.88 to 9.81), confirming that state expansion is ineffective without input-dependent write and read operations. This validates the core motivation in Section 3.1: larger state dimension provides more capacity to compress context, but the model needs selectivity to decide *what* to store in that larger state.

**Channel independence.** The SSM is applied **independently to each of the `$D$` channels** of the input. That is, there are `$D$` separate SSMs, each with its own parameters `$\boldsymbol{A}_d \in \mathbb{R}^N$`, `$\boldsymbol{B}_d \in \mathbb{R}^N$`, `$\boldsymbol{C}_d \in \mathbb{R}^N$`, and `$\Delta_d \in \mathbb{R}$`. The total hidden state dimension is `$D \times N$` (for a batch of size `$B$`, this is `$B \times D \times N$` at each timestep). This SISO (single-input single-output) formulation contrasts with the MIMO approach of S5, which groups multiple channels to reduce computation. The paper argues that the hardware-aware scan makes the full SISO formulation practical, providing a much larger effective state (`$D \times N$` vs. `$D \times N / G$` where `$G$` is the number of channels per group in S5).

**The "S6" nomenclature.** The paper occasionally refers to selective SSMs as "S6 models," defined in Remark 3.1 as "S4 models with a selection mechanism and computed with a scan." This is a playful naming convention (S4 + selection + scan = S6), emphasizing that the selection mechanism and the scan-based computation are the two key innovations over the S4 lineage.

## 4. Key Insights and Innovations

### Innovation 1: Selectivity as the Fundamental Principle for Sequence Model Design

The paper's most intellectually distinctive move is not any specific mechanism but a **reframing of what sequence modeling architectures are fundamentally about**. The dominant narrative in the field has been organized around computational primitives: you have attention-based architectures (Transformers), convolution-based architectures (CNNs, Hyena), and recurrence-based architectures (RNNs, SSMs), and the research program is to make each primitive more efficient while preserving its core properties. This paper argues that this taxonomy is misleading. The real axis of variation is not the computational primitive but rather **how the model compresses context into a state**, and specifically **whether this compression is selective (content-dependent) or not**.

This reframing is powerful because it explains a constellation of empirical observations that previously seemed contradictory. Why do LTI SSMs excel on the Long Range Arena but fail on language modeling? Why do gated architectures like H3 partially improve on selective tasks but not completely? Why do Transformers—with their complete absence of compression—still dominate language despite their quadratic cost? The answer, from this paper's perspective, is that all prior efficient architectures (linear attention, LTI SSMs, global convolutions) share a common pathology: they apply the same compression rule to every input regardless of content. Transformers avoid this by not compressing at all, at the cost of unbounded state growth. The paper's central claim is that **selectivity is the missing principle that lets a compressed state remain effective**.

The paper actually argues this in terms of a fundamental tradeoff (Section 3.1): "the efficiency vs. effectiveness tradeoff of sequence models is characterized by how well they compress their state: efficient models must have a small state, while effective models must have a state that contains all necessary information from the context." From this perspective, the failure of prior efficient models is not about their computational primitive (recurrence vs. convolution vs. linear attention) but about their **inability to decide what to keep and what to discard**. A selective recurrent model with a state of size `N` can in principle be more effective than a non-selective one with a state of size `10N`, because the selective model fills its state with relevant information while the non-selective model fills it with noise.

The synthetic tasks in Section 4.1 serve as **diagnostic probes** that make this abstract principle concrete. The Selective Copying task (Figure 2, Table 1) is particularly elegant: it isolates content-dependence as the key variable by holding everything else equal. The standard Copying task can be solved by any model that tracks time (learning the right convolution kernel length); the Selective Copying variant randomizes spacing, making time-tracking useless and forcing the model to look at token *content* to decide what to memorize. The fact that LTI SSMs achieve 18.3% accuracy while the selective variant achieves 97.0%—and that gated architectures without S6 only partially bridge this gap—is not just a performance result. It is evidence that **selectivity along the sequence dimension is a distinct computational capability** that neither gating (multiplicative interactions that don't propagate along the sequence) nor larger state dimensions can substitute for.

This framing also explains the Induction Heads result (Table 2) in a way that connects to the mechanistic interpretability literature. Induction heads require **content-based associative recall**: seeing token A should trigger retrieval of the token that previously followed A. This is exactly what a selective state can do—when A appears, the model can query its state for the stored B—but it is fundamentally impossible for an LTI model that treats all past positions identically. The fact that Mamba generalizes to 4000× training length while attention models fail beyond 2× (despite attention being the gold standard for induction heads) suggests that selectivity plus recurrence may be a more natural mechanism for this capability than attention's explicit position-based lookup.

This is a **fundamental conceptual contribution** rather than an incremental one. The paper is not proposing a better version of an existing idea; it is identifying a design principle that unifies and explains the behavior of disparate architectures, and then operationalizing that principle in a clean mechanistic form. The fact that the principle can be applied to RNNs, CNNs, and other architectures (as the paper notes in Section 3.5) makes it a lens for thinking about sequence models generally, not just a property of Mamba.

### Innovation 2: The Hardware-Aware State Expansion Algorithm as an Enabler

The second major insight is a **systems contribution that enables a conceptual one**. Making SSM parameters input-dependent is not a new idea—the paper acknowledges that "earlier works attempted to incorporate special cases of selection, such as letting `Δ` vary over time in recurrent SSMs" (Section 3.3). The reason every prior structured SSM remained LTI was not a lack of imagination but a **hard constraint from computational efficiency**: the convolutional form that made S4 and its derivatives fast requires time-invariance, and a naive recurrent implementation that materializes the full state tensor of size `(B, L, D, N)` would be catastrophically slow and memory-intensive. The field had implicitly accepted that you can have either content-dependent dynamics (like gated RNNs, with small states) or efficient large-state dynamics (like S4, with LTI), but not both.

The paper's hardware-aware scan algorithm breaks this perceived tradeoff, and the way it does so is instructive. The key observation (Section 3.3.2) is that the computation is **memory-bandwidth-bound, not compute-bound**. The FLOPs for the recurrent computation (`O(BLDN)`) are actually lower than for the convolutional computation (`O(BLD log L)`) for typical sequence lengths and state dimensions. The bottleneck is reading and writing the intermediate state tensor from GPU HBM. The solution—fusing discretization, scan, and output projection into a single SRAM-resident kernel—is not algorithmically novel (parallel scan has been known since Blelloch, 1990; kernel fusion is standard in high-performance computing), but it is **applied to a problem where it fundamentally changes the architectural design space**.

This matters because it **converts a hardware constraint into an architectural opportunity**. Prior SSMs were designed around the constraints of the convolution: they used complex numbers and special initializations (HIPPO) partly to make the convolutional kernel well-behaved over very long sequences. Mamba's scan algorithm makes the convolution unnecessary, which in turn removes the pressure to keep parameters time-invariant, which in turn enables selectivity. The hardware-aware implementation is thus not merely an optimization—it is what makes the entire selective SSM research program possible. Without it, S6 would be a theoretical curiosity that is too expensive to train at scale; with it, S6 becomes a practical architecture that can be deployed at billion-parameter scales.

The comparison to S5 (Smith, Warrington, and Linderman, 2023) is particularly revealing here. S5 independently recognized that the parallel scan could compute SSMs recurrently, but faced the same memory bottleneck. Their solution was to **reduce the effective state dimension** by switching from SISO (where each channel has an independent SSM of size `N`) to MIMO (where multiple channels share a single SSM, reducing the state per channel). This is a concession: to make the scan practical, S5 sacrificed state capacity. Mamba's innovation is to **keep the full SISO state but avoid materializing it**, achieving both large capacity and practical efficiency. This is a classic systems insight: the right algorithm plus careful memory management can make a theoretically expensive computation practical, and doing so changes what architectures are viable.

The efficiency results (Figure 8) back this up. The fused scan is 20–40× faster than a standard PyTorch scan, and Mamba achieves 5× higher inference throughput than a Transformer of similar size (because, without a KV cache, it can use much higher batch sizes). But the more important point is conceptual: this work demonstrates that **hardware-aware algorithm design is not just about making existing architectures faster—it can unlock entirely new classes of architectures** that were previously considered impractical.

### Innovation 3: Empirical Evidence That State Expansion Requires Input-Dependent Projections

One of the paper's most subtle but important contributions is a **negative result with positive implications**: showing that increasing the SSM state dimension `N` provides dramatic improvements *only when `B` and `C` are also input-dependent* (Table 10). This is not an obvious result, and it refines the paper's own thesis in an important way.

The naive intuition—which the paper itself builds up in Section 3.1—is that a larger state allows better compression of context, and therefore should improve performance. Many prior works have increased state or memory capacity for recurrent models (e.g., larger hidden states in LSTMs, multi-head linear attention in H3, the MIMO formulation in S5). If increasing `N` always helped, then the selection mechanism would be valuable but not strictly necessary; you could compensate for lack of selectivity by just making the state bigger.

Table 10 disproves this. With constant `B` and `C` (top section of the table), increasing `N` from 1 to 16 reduces perplexity from 9.88 to only 9.81—a negligible 0.07 improvement. With selective `B` and `C` (bottom section), the same increase reduces perplexity from 9.73 to 8.71—a substantial 1.02 improvement. The interaction is stark: state expansion is **nearly useless without selectivity**, and selectivity is **substantially amplified by state expansion**.

What this reveals is a **complementarity between the mechanism and the capacity**. Selectivity alone (with `N = 1`) provides a meaningful improvement (9.73 vs. 9.88 for the constant-`B`,`C` baseline), because the model can at least decide what to forget. But with `N = 1`, the state is a scalar per channel—there is no structure to what is remembered, just a single value that gets updated by a gate. Increasing `N` provides dimensions along which the model can represent different aspects of the context, but without input-dependent `B` and `C`, the model has no way to **route information to the appropriate dimensions**. Every input gets written to every state dimension with the same fixed pattern, and every output reads from every dimension with the same fixed pattern. The result is that the extra dimensions add noise rather than structured capacity.

When `B` and `C` are selective, the model can learn to use different state dimensions for different types of information. For example, one dimension might track whether a particular entity has been mentioned, another might accumulate positional information, a third might store a compressed representation of recent syntax. This is speculative—the paper does not attempt to interpret what individual state dimensions learn—but the empirical result strongly suggests that **selectivity enables structured use of capacity** rather than just more capacity.

This insight connects to the broader deep learning principle that **inductive biases and capacity are complementary**: giving a model more parameters without the right structure to use them often yields diminishing returns. The paper shows that for sequence models, the right structure is input-dependent projections that allow the model to decide *how* (not just *how much*) to write and read. This is a refinement of the selectivity thesis that moves it from "selectivity is important" to "selectivity is the gating factor that determines whether capacity investments pay off."

### Innovation 4: The Training-Inference Decoupling as an Architectural Principle

A less explicitly articulated but pervasive insight in the paper is the **decoupling of training and inference computational strategies**. Prior SSMs achieved efficiency through a clean dual form: train with convolutions (parallel, `O(L log L)`), infer with recurrences (sequential but `O(1)` per step). The selection mechanism breaks the convolution, forcing recurrence at training time too, which seems like a step backward in training efficiency.

The paper's response—embodied in the parallel scan and kernel fusion—is that **the correct computational strategy for training recurrent models is not to avoid recurrence but to parallelize it differently**. The parallel scan achieves `O(L)` work with `O(log L)` depth, which for typical GPU batch sizes and sequence lengths is actually faster than the `O(L log L)` FFT-based convolution, once memory bandwidth is accounted for. This is a nontrivial claim: it means the "dual form" efficiency of LTI SSMs was partly illusory—the convolutional mode was not necessarily the optimal way to train even the LTI models, but it was the only option that avoided materializing the large state tensor.

The broader implication is that **recurrence is not inherently slow for training**. The historical narrative that RNNs were abandoned in favor of Transformers because "RNNs can't be parallelized" conflates two issues: the sequential dependency of the recurrence (which the parallel scan addresses) and the memory cost of materializing the hidden states (which kernel fusion and recomputation address). The paper demonstrates that both can be overcome with careful engineering, and that the resulting recurrent model can be competitive with or faster than attention-based models at training time (Figure 8, left, shows the scan beating FlashAttention-2 beyond sequence length 2K).

This matters because it **reopens the design space of recurrent architectures**. For years, the field has implicitly assumed that any architecture with sequential dependencies is a non-starter for large-scale training, pushing research toward convolutions, linear attention, and other "parallelizable" alternatives. Mamba shows that this assumption is contingent on implementation, not fundamental. If recurrence can be made efficient, then a whole class of architectures with desirable properties (constant inference cost, natural handling of variable-length sequences, principled state management) becomes viable again. This is less a technical contribution of Mamba specifically and more a **demonstration that changes the field's priors about what is possible**.

The inference advantage (5× throughput, Figure 8 right) is then not just a nice bonus but a direct consequence of the architectural choice: because Mamba is a recurrent model, it does not need a KV cache, which means inference throughput is bounded by compute rather than memory, and batch size can scale much higher. In a Transformer, each token in the batch needs its own KV cache, and the total cache size grows with `batch_size × sequence_length × hidden_dim`, quickly saturating GPU memory. Mamba's constant-size state per sequence means that large-batch inference is feasible, which is crucial for production deployments. This is not a new property of recurrent models, but combined with Transformer-competitive quality, it becomes a significant practical advantage rather than a theoretical curiosity.

### Innovation 5: The Continuous-Discrete Spectrum as a Diagnostic for Inductive Bias

The paper's most nuanced contribution is the **empirical characterization of where selectivity helps and where it hurts**, and the framing of this in terms of a "continuous-discrete spectrum" of data modalities. This is not a new taxonomy, but the paper provides concrete evidence for it and uses it to explain both Mamba's successes and its limitations.

The finding (Section 4.4.1, Appendix E.4, Figure 10) is that on raw audio waveforms—a continuous signal sampled uniformly in time—the non-selective S4 actually outperforms the selective S6, and that this gap is concentrated in the layers closest to the raw input. When the outer layers of a U-Net are kept as LTI (Mamba-S4) and only the inner layers are made selective, the performance gap nearly disappears (Figure 10, right). The interpretation is that **continuous modalities benefit from time-invariant inductive biases** (fixed convolution kernels that capture frequency information, smooth dynamics, and resolution invariance), while **discrete modalities require content-dependent processing** to handle the arbitrary structure of tokens.

This is not obvious *a priori*. One could have imagined that selectivity is universally beneficial—after all, even in audio, being able to selectively attend to important frequency components or temporal events sounds useful. The empirical result shows that the continuous-time dynamical system interpretation of SSMs (the `h'(t) = Ah(t) + Bx(t)` formulation) provides genuine inductive bias value for uniformly sampled signals, and that imposing input-dependent dynamics on top of this can be harmful, perhaps because it breaks the smoothness or frequency-response properties that the LTI system was designed to have.

This finding has important implications for **how the architecture should be configured for different modalities**. It suggests a hybrid approach: use LTI SSMs (or even standard convolutions) for early layers processing raw signals, and selective SSMs for deeper layers where the signal has been transformed into more abstract, token-like representations. The paper doesn't fully explore this, but the U-Net ablation points in this direction. It also explains why prior SSMs were successful on audio and vision (continuous modalities where LTI is appropriate) but struggled on language and DNA (discrete modalities where selectivity is essential)—a unification of previously disconnected empirical findings.

The "no free lunch" framing (Section 5) is important because it **sets realistic expectations** and prevents the paper's contributions from being oversold. Mamba is not a universal replacement for all sequence models; it is specifically an architecture for discrete and information-dense modalities where content-based reasoning dominates. For continuous signals, LTI models (or hybrid approaches) may still be preferable. This contextualization—acknowledging the limitations of one's own method—is both scientifically honest and practically useful for practitioners deciding when to adopt Mamba vs. alternatives.

## 5. Experimental Analysis

### Evaluation Methodology

- **Dataset.** The paper evaluates on five distinct domains: (1) synthetic tasks (Selective Copying and Induction Heads, with sequences up to length 1M+ at test time); (2) language modeling on the Pile dataset (L. Gao, Biderman, et al., 2020) using the GPT2 tokenizer for scaling laws and the GPT-NeoX tokenizer (Black et al., 2022) for downstream evaluation, trained on 300B tokens; (3) DNA modeling on the HG38 human genome dataset (≈4.5B base pairs in the training split, following the setup of HyenaDNA; Nguyen, Poli, et al., 2023); (4) audio waveform modeling on YouTubeMix (4 hours of solo piano music at 16kHz) and SC09 (1-second clips of spoken digits at 16kHz); and (5) downstream zero-shot evaluation on LAMBADA, HellaSwag, PIQA, ARC-Easy, ARC-Challenge, and WinoGrande using the EleutherAI LM evaluation harness (L. Gao, Tow, et al., 2021).

- **Base model(s).** For language modeling, models range from ≈125M to ≈2.8B parameters, following GPT3 specifications (Brown et al., 2020) for depth and width. For DNA modeling, models range from ≈250K to ≈40.7M parameters at fixed short context (1024), and 1.3M–1.4M parameters for context length scaling up to 1M tokens. For audio, models use a U-Net backbone with 3.5M–24.3M parameters. Transformer baselines use either standard GPT3 architecture or an improved "Transformer++" recipe (rotary embeddings, SwiGLU MLP, RMSNorm, no linear bias, higher learning rates) following PaLM and LLaMa conventions. The choice of PaLM-derived recipes is deliberate: it represents the strongest known Transformer training configuration, making the comparison more stringent than using vanilla GPT3.

- **Metrics.** Language modeling: perplexity on the Pile validation set, plus zero-shot accuracy (or length-normalized accuracy for HellaSwag and ARC-Challenge) on downstream tasks. DNA modeling: pretraining perplexity, plus fine-tuning classification accuracy on the species task. Audio modeling: bits per byte (BPB), which is `log(2)` times the negative log-likelihood, plus automated generation metrics: NLL, Fréchet Inception Distance (FID), Inception Score (IS), modified Inception Score (mIS), and AM distance. Synthetic tasks: accuracy (fraction of correctly predicted tokens). Throughput: tokens per second for inference, milliseconds for scan/attention/convolution operations.

- **Baselines.** For language scaling laws: standard Transformer (GPT3), Transformer++ (rotary + SwiGLU + improved recipe), Hyena (Poli et al., 2023), H3++ (Dao, Fu, Saab, et al., 2023 with improved recipe), RWKV (B. Peng et al., 2023), and RetNet (Y. Sun et al., 2023). For downstream evaluation: Pythia (Biderman et al., 2023), RWKV, OPT, GPT-Neo, GPT-J, and Hybrid H3 models, all matched by training tokens (300B) and tokenizer where possible. For DNA: HyenaDNA (Nguyen, Poli, et al., 2023) and Transformer++. For audio: SaShiMi (Goel et al., 2022), WaveNet, SampleRNN, WaveGAN, DiffWave. For synthetic tasks: multi-head attention with absolute, RoPE, and xPos positional encodings; H3; Hyena; and non-selective S4 variants.

- **Generation budget / compute accounting.** Training compute is measured in total FLOPs following the Chinchilla protocol (Hoffmann et al., 2022), with models matched by total training tokens (e.g., 10B tokens for DNA scaling, 300B tokens for language downstream). Inference throughput is measured as tokens per second on A100 80GB PCIe GPUs, with prompt length 2048 and generation length 128. For the DNA context-length scaling experiments, the number of tokens per gradient update is held constant across sequence lengths by halving batch size when sequence length doubles, ensuring improvements with longer context are not confounded with more computation. For audio, total computation is controlled by keeping tokens per batch fixed (Table 14). The scan operation benchmark compares against FlashAttention-2 (Dao, 2024) with causal mask, convolution (PyTorch FFT-based), and a standard PyTorch parallel scan without kernel fusion.

- **Cross-validation / statistical protocol.** For synthetic tasks, results are reported after a fixed number of training steps, with the better of two learning rates reported for each model. For DNA species classification, learning rates were swept among `{1e-5, 2e-5, 4e-5, 1e-4, 2e-4}` at shorter sequence lengths, with the best value used for longer contexts. No explicit cross-validation or statistical significance testing is reported for the main language modeling results; scaling laws are presented as single curves without confidence intervals, reflecting the computational cost of multiple training runs at scale. This is standard for scaling law studies but limits the ability to assess whether observed differences between model families are statistically reliable.

---

### Main Quantitative Results

#### Synthetic Tasks: Demonstrating the Selection Mechanism's Necessity

**Selective Copying (Table 1).** The paper tests combinations of architectures (no gate, H3, Mamba) and inner SSM layers (S4, Hyena's global convolution, S6). LTI models fail dramatically: S4 without gating achieves only 18.3% accuracy; H3 with S4 reaches 57.0%; Hyena within H3 gets 30.1%. Adding architectural gating without selection provides only partial improvement. In contrast, every configuration using S6 achieves near-perfect accuracy: S6 without any gate reaches 97.0%, H3+S6 achieves 99.7%, and Mamba+S6 reaches 99.8%. The key comparison is within-architecture: Mamba+S4 achieves only 56.4% while Mamba+S6 achieves 99.8%, isolating the effect of the selection mechanism while holding the architecture constant.

**Induction Heads (Table 2, Table 11).** Models are trained on sequence length 256 (2^8) and tested on lengths from 64 to 1,048,576 (2^6 to 2^20). At training length, all models achieve 100% accuracy, indicating they can learn the task. The critical test is extrapolation:
- Multi-head attention with absolute positional encoding: accuracy drops to 58.6% at 2× training length (512) and continues degrading to 7.8% at 16K, beyond which it runs out of memory.
- MHA with RoPE: generalizes to 2× (100% at 512), then degrades (83.6% at 2^9, 31.3% at 2^10, 5.5% at 2^14).
- MHA with xPos (designed for length extrapolation): best among attention variants, maintaining 99.6% at 2× but then dropping to 67.6% at 2^10 and 7.8% at 2^14. All attention models run out of memory beyond 2^14.
- H3: maintains 100% at training length, drops to 80.9% at 2×, and oscillates between 4.7% and 8.2% for longer lengths.
- Hyena: achieves 100% at 2× but then drops sharply to 44.1% at 2^10 and hovers at 5–10% for longer lengths.
- **Mamba achieves 100% accuracy at EVERY tested sequence length up to 2^20 = 1,048,576**, which is 4096× the training length. This is the only model that extrapolates perfectly.

The paper notes that the Hyena model has 69M parameters, most in learnable positional encodings, while Mamba has only 74K parameters, making the comparison particularly stark.

#### Language Modeling: Scaling Laws and Downstream Performance

**Scaling laws (Figure 4, Table 12).** Models from ≈125M to ≈1.3B parameters are trained on the Pile following Chinchilla token counts (tokens proportional to model size). At context length 2048 (Figure 4, left):
- Mamba is the only attention-free model whose perplexity curve lies on top of the Transformer++ curve. The lines are nearly indistinguishable, meaning Mamba matches the strongest Transformer recipe at every model size.
- H3++ (the prior best SSM-based model) is roughly 0.5–1.0 perplexity points worse than Transformer++ across sizes.
- Hyena, RWKV, and RetNet form a cluster below H3++, with RWKV showing the weakest scaling among the subquadratic models.
- The standard Transformer (GPT3 recipe, no rotary, no SwiGLU) is substantially worse than Transformer++ and Mamba.

At context length 8192 (Figure 4, right), the gap between Mamba and Transformer++ narrows further, and Mamba now clearly outperforms Transformer++ at the largest model sizes (≈1.3B). Results for RWKV and RetNet at 8K are missing ("because of a lack of efficient implementations leading to out-of-memory or unrealistic computation requirements," Section 4.2.1), which is itself a significant finding: these prior state-of-the-art recurrent models cannot practically scale to longer contexts.

**Downstream zero-shot evaluation (Table 3).** Models are trained on 300B tokens with the GPT-NeoX tokenizer and evaluated on six commonsense reasoning benchmarks. The headline result: **Mamba is best-in-class at every model size on every single evaluation**. Specifically:
- Mamba-130M: Average accuracy 44.7%, vs. Pythia-160M at 40.6% and Hybrid H3-130M at 40.1%. Mamba is 4.1 points higher than the nearest same-size baseline.
- Mamba-370M: Average 50.0%, vs. Pythia-410M at 48.2%. Mamba exceeds the larger Pythia model by 1.8 points.
- Mamba-790M: Average 57.1%, vs. Pythia-1B at 51.9%. Here Mamba is 5.2 points above a model 25% larger.
- Mamba-1.4B: Average 59.7%, vs. Pythia-1.4B at 55.2%, RWKV-1.5B at 54.3%, and OPT-1.3B at 55.0%. Mamba's advantage is 4.5 points over the best same-size baseline.
- Mamba-2.8B: Average 63.3%, which exceeds **Pythia-6.9B** (61.7%), **OPT-6.7B** (62.9%), **GPT-J-6B** (63.0%), and **RWKV-7.4B** (62.5%). This is the key result: **Mamba at 2.8B matches or exceeds Transformers at ~7B parameters**, roughly 2.5× larger. On individual tasks, Mamba-2.8B achieves 69.7% on ARC-E (vs. 67.3% for Pythia-6.9B), 36.3% on ARC-C (vs. 37.5% for RWKV-7.4B, where it is slightly behind), and 63.5% on WinoGrande (vs. 61.3% for Pythia-6.9B).

On the Pile validation perplexity, Mamba also dominates: Mamba-370M achieves 8.28 vs. Pythia-410M at 9.95; Mamba-1.4B achieves 6.80 vs. Pythia-1.4B at 7.51 and RWKV-1.5B at 7.70. The perplexity improvements are consistent across all model sizes.

#### DNA Modeling: Scaling Laws and Long-Context Classification

**Model size scaling (Figure 5, Left).** Models from ≈250K to ≈40.7M parameters are pretrained on the HG38 human genome at short context length 1024, with 10K gradient steps at a fixed 1M tokens per batch (10B total tokens). The log-scale curves show:
- Mamba's perplexity decreases smoothly with model size, and the slope is steeper than both HyenaDNA and Transformer++, meaning Mamba scales better.
- At the largest size (≈40.7M parameters), Mamba achieves perplexity of approximately 2.73, while Transformer++ is at roughly 2.83 and HyenaDNA at roughly 2.90. The paper states that "Mamba can match the Transformer++ and HyenaDNA models with roughly 3× to 4× fewer parameters." This is a significant result for genomics, where model efficiency is practically important due to the sheer size of genomic datasets.

**Context length scaling (Figure 5, Right).** Models with ~1.3M–1.4M parameters (6 layers, width 128) are pretrained at sequence lengths from 2^10 = 1024 to 2^20 = 1,048,576, with 20K gradient steps and total tokens held constant at ≈330B. The batch size is halved when sequence length doubles, keeping tokens per gradient step fixed at ≈16M.
- Mamba's pretraining perplexity **improves monotonically with sequence length**, from roughly 3.00 at length 1024 to roughly 2.82 at length 1M for the 1.4M parameter model. The 7M parameter model shows a similar trend, from roughly 2.90 to roughly 2.78.
- HyenaDNA-1.4M shows the **opposite** trend: perplexity **increases** (worsens) from roughly 2.94 at length 1024 to roughly 3.04 at length 1M.
- This is the most direct empirical evidence for the paper's core thesis about selectivity and context filtering (Section 3.5): HyenaDNA's global convolution aggregates noise across the entire length, degrading as context grows, while Mamba can selectively reset its state and ignore irrelevant regions.

**Species classification (Figure 6, Table 13).** Pretrained models are fine-tuned on the task of classifying between five great ape species (human, chimpanzee, gorilla, orangutan, bonobo) from contiguous DNA segments of lengths 2^10 to 2^20.
- Mamba-1.4M: Accuracy improves from 31.47% at length 1K to 71.67% at length 1M, with most of the gain concentrated at the longest lengths (from 42.41% at 2^18 to 71.67% at 2^20).
- Mamba-7M: Improves from 30.00% at 1K to 81.31% at 1M, again with a dramatic jump at the longest context.
- HyenaDNA-1.4M: Improves from 28.04% at 1K to 54.87% at 1M, but the final accuracy is 16.8 points lower than Mamba-1.4M and 26.4 points lower than Mamba-7M.
- Random guessing is 20% (5 species). No model exceeds 60% until sequence length 2^20, underscoring the difficulty of distinguishing species that share 99% of their DNA—long-range dependencies are essential.

The key pattern is that Mamba's accuracy **accelerates** at the longest contexts while HyenaDNA's plateaus or improves slowly, consistent with the hypothesis that selective state management enables effective use of very long-range information.

#### Audio Modeling and Generation

**Long-context pretraining (Figure 7, Table 14).** Models with ~3.5M parameters are pretrained on YouTubeMix at sequence lengths from 2^13 = 8192 to ≈2^20 (468 × 2048 = 958,464, the maximum due to 1-minute clip constraints). Computation is held fixed by adjusting batch size (Table 14). Both Mamba and SaShiMi (S4+MLP) improve with longer context, but Mamba is consistently better and the gap widens at longer lengths. At the longest context, Mamba achieves roughly 1.31 BPB vs. SaShiMi at roughly 1.33 BPB. The improvement is modest (≈0.02 BPB) but consistent, and the widening gap suggests the selection mechanism provides increasing benefits as the context grows long enough to contain meaningful long-range structure.

**Speech generation on SC09 (Table 4).** Models generate 1-second audio clips unconditionally, with quality measured by automated metrics. The results are striking:
- Mamba-6.1M: NLL 1.852, FID 0.94, IS 6.26, mIS 88.54, AM 0.52. This **outperforms all prior models including much larger ones**: SaShiMi-5.8M (FID 1.99, mIS 42.57), WaveGAN-19.1M (FID 2.03, mIS 36.10), and DiffWave-24.1M (FID 1.92, mIS 51.21). On FID, Mamba more than halves the error of the previous state-of-the-art (0.94 vs. 2.03 for WaveGAN and 1.92 for DiffWave).
- Mamba-24.3M: FID improves further to 0.67, IS to 7.33, mIS to 144.9, and AM to 0.36. The mIS of 144.9 is roughly 2× the SaShiMi+DiffWave hybrid (69.17) and nearly 3× the standalone DiffWave (51.21).
- The training set metrics (FID 0.00, IS 8.56, mIS 292.5) and test set metrics (FID 0.02, mIS 257.6) suggest there is still room for improvement, but Mamba substantially closes the gap to real data.

**Architecture ablation (Table 5).** The U-Net's 40 blocks are divided into outer blocks (operating on full-resolution 16000Hz audio) and center blocks (operating on downsampled 1000-length sequences). Ablating the center blocks independently:
- S4+MLP outer + MHA+MLP center: FID 1.45, mIS 47.03
- S4+MLP outer + S4+MLP center: FID 1.43, mIS 53.54
- S4+MLP outer + Mamba center: FID 1.42, mIS 56.51
- **Mamba outer + Mamba center**: FID **0.94**, mIS **88.54**
- The jump when switching outer blocks from S4+MLP to Mamba is dramatic regardless of center block: Mamba outer + MHA+MLP center achieves FID 1.37 (vs. 1.45 with S4 outer), and Mamba outer + S4+MLP center achieves FID 1.07 (vs. 1.43). This suggests the outer layers, which process the raw high-resolution waveform, benefit enormously from selectivity despite audio being a continuous modality. This appears to contradict the YouTubeMix finding (Figure 10), but note that SC09 involves highly structured discrete content—spoken digits—rather than continuous music, which may shift it toward the discrete end of the spectrum where selectivity helps.

#### Speed and Memory Benchmarks

**Scan vs. convolution vs. attention (Figure 8, Left).** Measured on A100 80GB PCIe, batch size 1, model dimension D=1024, state dimension N=16, sequence lengths from 512 to 512K:
- The paper's fused scan is roughly **40× faster** than a standard PyTorch scan across all sequence lengths. At length 128K, the fused scan takes ~10ms while PyTorch scan takes ~400ms.
- The fused scan is **faster than FlashAttention-2** beyond sequence length ~2K. At 32K, scan takes ~2ms while FlashAttention-2 takes ~15ms (7× advantage). At 128K, scan takes ~10ms while FlashAttention-2 takes ~80ms (8× advantage).
- The fused scan is **faster than convolution** (PyTorch FFT-based) beyond ~4K. At 128K, convolution takes ~30ms vs. scan's ~10ms (3× advantage). At 512K, convolution runs out of memory while scan completes.
- This validates the claim that the recurrent mode can be more efficient than the convolutional mode at long sequences when properly implemented.

**Inference throughput (Figure 8, Right).** Prompt length 2048, generation length 128, varying batch size on A100 80GB:
- Mamba-1.4B achieves 461 tokens/s at batch size 1, increasing to 1814 tokens/s at batch size 128. Transformer-1.3B achieves 264 tokens/s at batch size 1, increasing to 515 tokens/s at batch size 128 but running out of memory at batch size 64.
- At batch size 64, Mamba-1.4B achieves 1688 tokens/s vs. Transformer-1.3B at 490 tokens/s — a **3.4× throughput advantage**.
- Mamba-6.9B achieves 159 tokens/s at batch size 1, increasing to 490 tokens/s at batch size 128. Transformer-6.7B achieves 79 tokens/s at batch size 1, increasing to 132 tokens/s at batch size 32, then running out of memory at larger batches.
- The paper states that Mamba-6.9B (untrained) would have "higher inference throughput than a 5× smaller Transformer-1.3B" — at batch size 32, Mamba-6.9B achieves 364 tokens/s while Transformer-1.3B achieves 323 tokens/s, confirming this. The reason is that without a KV cache, Mamba's memory footprint per sequence is constant, so it can scale to larger batch sizes without running out of memory.

**Memory consumption (Table 15).** Training memory for 125M parameter models, batch size 1–32, sequence length 2048, on A100 80GB:
- At batch size 1: Mamba 4.8GB vs. Transformer (with FlashAttention-2 + torch.compile) 4.6GB — nearly identical.
- At batch size 8: Mamba 12.3GB vs. Transformer 11.5GB — a 7% overhead.
- At batch size 32: Mamba 38.2GB vs. Transformer 34.5GB — an 11% overhead.
- The memory scaling is comparable, with Mamba having a slightly higher constant factor but the same linear scaling with batch size. The paper attributes this to the recomputation strategy and notes that "we expect further improvement in Mamba's memory footprint in the future."

---

### Ablation Studies and Robustness Checks

All ablations in this section use the language modeling setting with ≈350M parameter models at Chinchilla token counts (the same setting as Figure 4), unless otherwise noted.

**Architecture and SSM layer (Table 6).** Within the H3 architecture, swapping the inner SSM layer from Hyena's global convolution to S4-complex to S4-real to S6 shows: Hyena (10.24 perplexity), S4-complex (10.30), S4-real (10.34), and S6 (8.95). The LTI variants are nearly identical, confirming the paper's claim that the choice of LTI SSM parameterization matters little for language. S6 provides a 1.3–1.4 perplexity improvement. Within Mamba, the gap is even larger: S6 achieves 8.69 vs. S4-real at 10.56 (1.87 point improvement). Comparing architectures with S6: H3+S6 achieves 8.95 while Mamba+S6 achieves 8.69, suggesting the simpler Mamba block is slightly more effective when using selective SSMs.

**Selective parameters (Table 7).** Testing all combinations of selective Δ, B, and C: the non-selective baseline achieves 10.93 perplexity. Making only B selective: 10.15. Only C selective: 9.98. Only Δ selective: 9.81. All three selective: 8.71. The most important single parameter is Δ, confirming Theorem 1's connection to gating. However, the combination of all three provides an additional 1.1 perplexity improvement over Δ alone, and 2.22 over the non-selective baseline. This is a critical result: while Δ provides the primary selectivity mechanism (coarse-grained forget/remember decisions), B and C provide complementary fine-grained control that roughly doubles the total benefit.

**Parameterization of A (Table 8).** Testing different initializations for the diagonal A matrix: S4D-Lin (complex: A_n = -1/2 + ni) achieves 9.16; A_n = -1/2 (real) achieves 8.85; S4D-Real (A_n = -(n+1)) achieves 8.71; random initialization A_n ~ exp(N(0,1)) achieves 8.71. The standard complex initialization from prior SSMs performs worst. The simpler real-valued geometric spacing (S4D-Real) matches random initialization, suggesting that for language modeling with selective SSMs, the initialization of A is not critical—the selection mechanism likely overrides any fixed timescale prior.

**Expressivity of Δ projection (Table 9).** Varying the dimension of the Δ projection from 0 (non-selective: 9.12 perplexity) to 64 (8.71 perplexity): dimension 1 already gives a large improvement (8.97), dimension 4 gives 8.91, dimension 16 gives 8.84, and dimension 64 gives 8.71. The parameter cost is modest: 359.1M (dim 1) to 371.5M (dim 64), a 3.4% increase. Most of the benefit is captured by dimension 16, suggesting that a small amount of expressivity in the Δ projection is sufficient.

**SSM state dimension N (Table 10).** This is the most important ablation, as it validates the paper's central thesis about state expansion × selectivity. With constant B and C (top section), increasing N from 1 to 16 reduces perplexity from 9.88 to only 9.81—a 0.07 improvement. With selective B and C (bottom section), the same increase reduces perplexity from 9.73 (N=1, already better than constant N=16) to 8.71 (N=16)—a 1.02 improvement. At N=1, making B and C selective improves from 9.88 to 9.73 (0.15 gain). At N=16, making B and C selective improves from 9.81 to 8.71 (1.10 gain). The interaction is multiplicative: selectivity amplifies the benefit of state expansion, and state expansion amplifies the benefit of selectivity. The parameter cost of increasing N from 1 to 16 is only 4.4M (1.2% of 367M).

**Mamba block interleaving (Figure 9, Left).** Ablating whether Mamba blocks are stacked homogeneously or interleaved with MLP or MHA blocks: Mamba-MLP (homogeneous stacking) achieves roughly the same perplexity as interleaved Mamba+MLP, and slightly worse than Mamba-MHA (Mamba interleaved with attention). Interestingly, adding attention to Mamba (Mamba-MHA) provides only a modest improvement, which the paper finds "somewhat surprising in light of the fact that many recent works have found that combining (LTI) SSMs with Attention can lead to substantial improvements." This suggests that selective SSMs already capture much of what attention provides.

**H3 training recipe (Figure 9, Right).** Isolating the effect of the improved training recipe: Hyena (original recipe, GPT3-style) achieves the worst perplexity; Hyena+ (same architecture, improved recipe: 5× higher LR, cosine decay, RMSNorm, no bias) provides a large jump; H3+ (Hyena+ with S4 replacing Hyena's convolution) is nearly identical; H3++ (H3+ with linear attention head dimension 8) provides further improvement. The key finding: the training recipe accounts for a substantial fraction of the improvement over prior work, and the choice of LTI SSM (Hyena vs. S4) makes minimal difference, reinforcing that the architectural innovation is the selection mechanism, not the specific LTI backbone.

**Audio SSM parameterization (Figure 10).** On YouTubeMix audio pretraining, the non-selective S4 outperforms S6: at sequence length 2^18, S4 achieves ~1.30 BPB vs. S6's ~1.33. Adding complex numbers and removing selective B/C narrows but doesn't close the gap. However, when only the center U-Net blocks are ablated while outer blocks remain Mamba-S4 (non-selective), the performance difference nearly disappears: S6 in the center achieves roughly 1.29 BPB vs. S4 at ~1.295. This is a nuanced finding: selectivity hurts in layers close to the raw audio signal but may help in deeper, more abstract layers. The paper frames this as evidence for the continuous-discrete spectrum hypothesis.

**DNA context length warmup.** The paper notes (Remark E.1) that sequence length warmup (progressively increasing context length during training) was used for DNA but "was not tuned, and we never experimented with turning off sequence length warmup for these pretraining experiments." They later found it was not necessary for audio, raising the possibility that the DNA results could be achieved without this additional complexity.

---

### Critical Assessment

#### Claim 1: Selection mechanism enables content-based reasoning that LTI models fundamentally lack.

The evidence from synthetic tasks (Table 1, Table 2) provides **strong and clean support**. The Selective Copying experiment is particularly well-designed because it isolates content-dependence as the sole variable: standard Copying is solvable by LTI models, and Selective Copying differs only in randomizing spacing, which forces content-based decisions about what to memorize. The jump from 18.3% (S4) to 97.0% (S6) with no other architectural changes is a textbook ablation. The Induction Heads extrapolation (100% at 4096× training length vs. all baselines failing by 2×) is similarly decisive.

**Caveats:** Both tasks use small vocabularies (16 tokens for Selective Copying, 16 tokens for Induction Heads), small model dimensions (D=64), and 2-layer models. Whether the perfect extrapolation on Induction Heads transfers to larger models with realistic vocabularies is unproven. The synthetic tasks probe specific mechanistic capabilities (selective memorization, associative recall) but may not capture all forms of content-based reasoning that matter in language.

#### Claim 2: Mamba matches or exceeds Transformer performance on language modeling.

The scaling laws (Figure 4) show Mamba matching Transformer++ across the 125M–1.3B parameter range, which is **convincing but limited in scale**. The Chinchilla-style training (tokens proportional to model size) means the 1.3B model was trained on only 26B tokens, far less than the 300B used for downstream evaluation. At the 300B-token scale, the downstream results (Table 3) show Mamba consistently outperforming same-size Transformers and matching Transformers ~2× larger. This is stronger evidence because it reflects a more realistic training budget.

**However, the scale ceiling is critical.** The largest Mamba model is 2.8B parameters, trained on 300B tokens. This is below the scale of LLaMA-7B (1T+ tokens), Chinchilla-70B, and other models that define the modern LLM frontier. The paper acknowledges this explicitly (Section 5): "It remains to assess whether Mamba still compares favorably at these larger sizes." There are plausible reasons the advantage might not persist: larger models may saturate the benefits of recurrence (Transformers might catch up as both model families approach the irreducible loss), or engineering challenges (training instability, numerical precision in the scan) might emerge at scale. The paper cannot rule out that Mamba's advantage is a small-to-medium-scale phenomenon.

**Additionally, the baselines are not fully optimized.** Pythia and RWKV were trained with context length 2048 and 1024 respectively, while Mamba was trained at 2048. The effect of context length on downstream perplexity is not isolated. More importantly, none of the baselines use the improved training recipe (Transformer++) that Mamba itself uses—Pythia uses a GPT3-style recipe, as does GPT-J. A fairer comparison would pit Mamba against a Transformer++ trained on 300B tokens with the same recipe.

#### Claim 3: Mamba achieves 5× higher inference throughput than Transformers.

The throughput measurements (Figure 8, Right) support this at batch sizes ≥32. At batch size 32, Mamba-1.4B achieves 1688 tokens/s vs. 490 for Transformer-1.3B (3.4×). At batch size 64, it's 1814 vs. OOM for Transformer. The 5× figure appears to refer to the comparison at larger batch sizes or to the statement that Mamba-6.9B outperforms Transformer-1.3B. **The throughput advantage is real but conditional on batch size**: at batch size 1, Mamba-1.4B achieves 461 vs. 264 (1.7×), which is substantial but not 5×. The advantage stems from the absence of a KV cache, which becomes more constraining at larger batch sizes. For applications that cannot batch (interactive single-user inference), the advantage is smaller (but still meaningful).

**A missing comparison:** The inference benchmark uses an untrained Mamba-6.9B. Training changes weight distributions and may affect kernel efficiency subtly. The transformer baseline uses standard HuggingFace implementation, not an optimized serving framework (like vLLM or TensorRT-LLM). These are reasonable choices for a research paper, but the absolute throughput numbers should be taken as indicative, not as production benchmarks.

#### Claim 4: Mamba's performance improves with longer context while LTI models degrade.

The DNA context length scaling experiment (Figure 5, Right) provides **the cleanest support**: HyenaDNA's perplexity worsens from ~2.94 to ~3.04 as sequence length increases from 1K to 1M, while Mamba's improves from ~3.00 to ~2.82. The batch size adjustment (halving batch size when doubling length) properly controls for computation, addressing the paper's own critique of HyenaDNA's methodology. The species classification results (Figure 6) corroborate this: Mamba's accuracy jumps from 31% to 72% going from 1K to 1M, while HyenaDNA goes from 28% to 55%.

**However, the absolute perplexities in Figure 5 (Right) are worse than in Figure 5 (Left) for the same model sizes** (Mamba-1.4M achieves ~2.82 at 1M context in the right panel vs. ~2.75 at 1K context with a slightly different model in the left panel). This reflects differences in training tokens, batch size, and model configuration across the two experiments, but the inconsistency is not explained. Additionally, the DNA perplexities are measured on the training distribution (the human genome); it's unclear how well the long-context improvements transfer to other species or genomic tasks beyond the great apes classification.

#### Claim 5: Selectivity can hurt performance on continuous modalities.

The YouTubeMix result (Figure 10) supports this with a **nuanced finding**: S4 outperforms S6 on raw audio, but the gap shrinks when only the center (downsampled) layers are made selective. This is an important caveat that the paper is honest about, but it is **under-explored**. Only one audio dataset is tested for this effect. The SC09 results (Table 4) show Mamba dramatically outperforming S4 baselines, which contradicts the YouTubeMix finding and suggests the modality distinction is more about the nature of the signal (continuous music vs. structured speech) than about audio per se. The paper's "continuous-discrete spectrum" hypothesis is plausible but not rigorously tested: what specific properties of a modality determine whether selectivity helps or hurts? Is it the sampling rate, the presence of discrete symbols, the amount of long-range structure, or something else?

#### Overall Assessment of Experimental Design

**Strengths:**
- The multi-domain evaluation (language, DNA, audio, synthetics) is unusually broad for an architecture paper and demonstrates generality.
- The synthetic tasks serve as clean diagnostic probes that isolate specific mechanisms, complementing the more complex and confounded real-world evaluations.
- The DNA context-length experiment properly controls for computation, making the comparison against HyenaDNA methodologically rigorous.
- The ablation studies (Tables 6–10) are comprehensive and reveal non-obvious interactions, particularly the state dimension × selectivity interaction.
- The paper is honest about negative results (audio limitation, scale ceiling, ReST failure) rather than burying them.

**Weaknesses:**
- The largest language model (2.8B) is 2–10× smaller than modern open-source LLMs. The scaling laws cover 125M–1.3B parameters, leaving a 10× gap to the models the paper's claims would most impact.
- All language experiments use a single dataset (the Pile). Domain transfer (code, scientific text, multilingual) is untested.
- The downstream evaluation uses only 300B training tokens—substantial, but Chinchilla-optimal for a 2.8B model would be ~60B tokens, meaning the model is potentially undertrained relative to its size, which could affect the quality comparisons.
- No standard errors or confidence intervals are reported for any perplexity or accuracy numbers. The scaling curves are single runs, and the downstream evaluations are single checkpoint evaluations. With test sets of 500–1000 questions, the observed 1–2 point differences between models may not be statistically significant.
- The Transformer++ baseline for scaling laws uses the improved recipe, but the downstream Transformer baselines (Pythia, GPT-J) do not. This makes the downstream comparison asymmetrical—Mamba gets the benefit of the improved training recipe while the baselines do not.
- The paper consistently compares Mamba against Transformers of the same parameter count, but Mamba's blocks contain both SSM and MLP functionality, making parameter-count comparisons potentially misleading. A FLOP-matched comparison (rather than parameter-matched) might show different results.
- The DNA species classification tasks use only 10 epochs of fine-tuning with 1024 gradient steps per epoch (for longer sequences, only 4 epochs at the maximum length after warmup). The impact of more extensive fine-tuning is unexplored.

**Missing experiments:**
- Mamba at 7B+ parameter scale with 1T+ training tokens.
- Combining Mamba with mixture-of-experts (MoE) to test whether sparsity and selectivity are complementary.
- A FLOP-matched comparison against Transformers (rather than parameter-matched) for both training and inference.
- Evaluation on tasks specifically requiring very long context (e.g., book summarization, long-document QA, multi-turn dialogue over hundreds of turns) where Mamba's architectural advantage should be most pronounced.
- An experiment testing whether the difficulty estimation protocol from Section 3.5 (using PRM scores as a proxy) could be adapted to dynamically adjust Mamba's selectivity, e.g., by using different state sizes or Δ parameterizations depending on estimated sequence complexity.
- Direct comparison against sparse attention Transformers (e.g., Longformer, BigBird) at sequence lengths where both are feasible, to test whether Mamba's recurrent state compression is more effective than sparse attention patterns.

## 6. Limitations and Trade-offs

### The Difficulty Estimation Cost Dominates Deployment Overhead

**The assumption or constraint.** The compute-optimal framework selects inference strategies per prompt based on an estimated difficulty score. The paper's method for estimating difficulty — generating 2048 samples per question and averaging either ground-truth correctness (oracle bins) or PRM final-answer scores (predicted bins) — is extraordinarily expensive. Section 3.2 acknowledges this directly:

> "estimating difficulty in this way still incurs additional computation cost during inference... our experiments do not account for this cost largely for simplicity"

**The consequence.** The headline 4× efficiency gains are computed *after* difficulty is already known, without amortizing the cost of learning it. In a realistic deployment, the total cost would be difficulty estimation + strategy execution. Generating 2048 samples per prompt consumes **more compute than the largest test-time budgets studied** (256–512 generations). This means the reported efficiency is an upper bound that is unattainable in practice unless an alternative, cheap difficulty estimator exists. The paper frames this as an exploration-exploitation tradeoff and flags it as "a key avenue for future work" (Section 3.2), but provides no method to close the gap.

**What evidence exists in the paper.** The difficulty estimation protocol is described in Section 3.2: 2048 samples per question, PRM-scored, binned into five quintiles. The cost of this step is never included in any budget calculation in Figures 4, 8, or elsewhere. Section 3.2 states the cost is unaccounted "largely for simplicity." No experiment measures how performance degrades when difficulty is estimated from fewer samples, nor does the paper test alternative cheap estimators (e.g., a lightweight classifier trained on question text).

**Mitigation status.** Not addressed. The paper identifies "pretraining or finetuning models to directly predict difficulty of a question" as future work (Section 8) but develops no such model. A practitioner reading this paper cannot deploy the compute-optimal policy as described without paying the difficulty estimation cost, which may dominate or exceed any savings from smarter allocation.

---

### The Hardest Problems Remain Essentially Unsolved

**The assumption or constraint.** The paper's compute-optimal framework assumes the base model already produces correct solutions at some non-trivial rate for a given problem. When pass@1 is near zero, no amount of search or revision can help — there are no correct solutions in the proposal distribution to find or refine. The paper is explicit about this boundary in Section 7:

> "On the hardest questions (bin 5), no method makes meaningful progress — the base model simply lacks the capability to produce correct solutions regardless of budget allocation."

**The consequence.** Test-time compute can amplify existing capability but cannot create it. This is a hard ceiling for the approach: if a problem requires knowledge or reasoning the model did not acquire during pretraining, all of the paper's techniques — search, revisions, compute-optimal allocation — provide essentially zero benefit. This limits the method's applicability to **problems within the base model's rough capability range**, and means the approach offers no path forward for genuinely novel or out-of-distribution reasoning. For an organization deciding between investing in larger pretraining vs. better inference strategies, this boundary condition is critical: if the target problem distribution includes many "bin 5" problems, pretraining remains the only viable path.

**What evidence exists in the paper.** This is documented across all experimental sections. Figure 3 (right, bin 5): accuracy hovers at 1–3% for all methods and all generation budgets. Figure 7 (right, bin 5): accuracy stays at ~2–3% regardless of sequential-to-parallel ratio. Figure 9 (bin 5 line): the scaling curve is essentially flat near 0–5%, and sits below the 14× larger model's performance across all R values. The FLOPs-matched comparison shows that on hard problems, test-time compute provides a −52.9% relative *disadvantage* compared to simply training a larger model (at R ≫ 1 with PRM search, Section 7).

**Mitigation status.** The paper is transparent about this limitation but does not attempt to solve it. It is a fundamental constraint of the approach, not an engineering oversight. The paper frames it as establishing the boundary condition for when to prefer test-time compute vs. pretraining (Section 7 takeaway box), which is valuable guidance even if not a solution.

---

### The FLOPs-Matched Comparison Uses a Weakened Pretraining Baseline

**The assumption or constraint.** The paper's central claim that test-time compute can substitute for pretraining compute rests on a comparison between PaLM 2-S* with compute-optimal inference and a ~14× larger model with **greedy decoding and no test-time augmentation**. Additionally, the larger model is scaled only in parameters (not data), departing from Chinchilla-optimal pretraining where both would scale equally. The paper acknowledges this in Section 7:

> "We choose this setting as it is representative of a canonical approach to scaling pretraining compute and leave the analysis of compute-optimal scaling of pretraining compute where the data and parameters are both scaled equally to future work."

**The consequence.** The reported advantages of test-time compute over pretraining — e.g., +27.8% relative improvement on easy questions at R ≪ 1 (Figure 1) — may shrink or reverse against a stronger baseline. A Chinchilla-optimal larger model (scaling both parameters and data) would likely outperform the parameter-only-scaled model used. Even more consequentially, the larger model uses only greedy decoding — no majority voting, no best-of-N, no search. Giving the 14× larger model even a modest test-time compute budget (e.g., best-of-8) would create a much more competitive baseline, and the paper provides no evidence for how the comparison would change. For a practitioner deciding between "train bigger" and "infer smarter," the current comparison overstates the case for inference by making the pretraining baseline artificially weak.

**What evidence exists in the paper.** Section 7 describes the FLOPs-matched setup, confirming the larger model uses greedy decoding and scales only parameters. The paper acknowledges this is not compute-optimal pretraining (citing Hoffmann et al., 2022) and leaves the comparison to future work. No experiment tests the larger model with any test-time compute budget, nor with Chinchilla-optimal training. Figure 9 and the bar charts in Figure 1 are the reported results, but the absolute numbers depend on the (potentially weak) baseline.

**Mitigation status.** The paper is transparent about the choice but does not run the stronger baseline. A practitioner should treat the 4× and "matches 2× larger models" claims as potentially inflated relative to a properly optimized pretraining+inference pipeline. The paper's acknowledgment is honest but does not reduce the uncertainty around the headline comparison.

---

### No Combination of the Two Primary Mechanisms (Search + Revisions)

**The assumption or constraint.** The paper studies PRM-guided search (Section 5) and iterative revisions (Section 6) as **independent mechanisms** and evaluates them separately. They are never combined — beam search is applied to base model outputs, not to revision model outputs; the PRM is not used to guide which revision branches to pursue. Section 8 explicitly acknowledges this:

> "we did not experiment with PRM tree-search techniques in combination with revisions"

**The consequence.** The two mechanisms have complementary strengths: revisions improve the proposal distribution (generating better candidates via iterative refinement), while PRM search improves candidate selection (finding the best among generated candidates via verifier-guided exploration). The fact that the paper never combines them means the reported results represent a **lower bound** on what a fully integrated system could achieve. For a practitioner trying to maximize accuracy under a fixed budget, this matters: the paper cannot tell you whether combining search and revisions yields gains beyond either method alone, or whether the combination might create new failure modes (e.g., revision models producing outputs that the PRM — trained on base model outputs — scores unreliably due to distribution shift, a problem already documented in Appendix J, Figure 15a).

**What evidence exists in the paper.** Appendix J (Figure 15a) shows that the base-LM PRM underperforms a revision-specific ORM when scoring revision model outputs (~40% vs. ~42% at 64 generations), confirming distribution shift as a concern. Section 5.3 documents PRM over-optimization (beam search hurting easy-problem performance), and Section 6.1 documents the correct-to-incorrect reversion problem (~38% of correct answers get revised to incorrect). How these failure modes interact when search and revisions are combined is entirely unexplored.

**Mitigation status.** Acknowledged as future work (Section 8). No experiments are run. This is a significant gap because the natural next step for a practitioner wanting to maximize performance would be to combine both mechanisms, and the paper provides no guidance on whether this is beneficial or how to do it effectively.

---

### The Revision Model Has a Fundamental Correct-to-Incorrect Reversion Problem

**The assumption or constraint.** The revision model is fine-tuned on trajectories where every in-context answer is incorrect, followed by a correct target (Section 6.1). This training data construction means the model never learns what to do when the current answer *is already correct*. At inference time, when the model generates a chain of revisions, it may encounter correct answers in its context and — having no training signal for this situation — "revise" them into incorrect answers.

**The consequence.** The paper reports that approximately **38% of correct answers get converted back to incorrect ones** using a naive approach (Section 6.1). The system mitigates this with majority voting or verifier-based selection across the entire chain (picking the best answer from any point rather than always taking the final revision), but these are patches, not solutions. The underlying model has a systematic failure mode: it does not know when to stop. For a practitioner deploying this in a system where revision chains must terminate automatically (without an external verifier or oracle), the 38% reversion rate means accuracy may *degrade* with additional revision steps beyond some optimal point, and determining that optimal stopping point requires the very verifier the revisions are meant to improve upon. This also limits the model's usefulness for self-improvement loops where the goal is to generate higher-quality training data: if the model cannot reliably preserve correct answers, downstream fine-tuning on revision outputs may introduce noise.

**What evidence exists in the paper.** Section 6.1 states the ~38% reversion rate. Section 6.1 describes the mitigation (within-chain majority/verifier selection). The training data construction is described in Section 6.1: sequences of 0–4 incorrect answers followed by a correct answer, with no trajectories where a correct answer appears in context. The ReST^EM experiment (Appendix K, Figure 16) provides additional evidence of fragility: further optimizing the revision model with RL-style training caused performance to "substantially hurt," with fully sequential performance dropping to ~33.5% compared to ~38.5% at the optimal ratio, suggesting the revision approach is sensitive to training methodology in ways not fully understood.

**Mitigation status.** Partially addressed via within-chain selection, but the root cause (training data construction that never exposes the model to correct in-context answers) is not solved. The paper does not explore training on trajectories that include correct answers (e.g., teaching the model to output a special "no revision needed" token or to copy the input when it is already correct). This is a trainability issue, not a fundamental limitation of the revision concept, but the paper provides no solution.

---

### The Entire Study Uses a Single Model Family and a Single Benchmark

**The assumption or constraint.** All experiments use PaLM 2-S* (Codey) as the base model and the MATH benchmark (Hendrycks et al., 2021) as the primary evaluation — 12,000 training questions and 500 test questions from high-school competition mathematics. The paper justifies this in Section 4:

> "we believe this model is representative of the capabilities of many contemporary LLMs"

but provides no evidence that the findings transfer to other model families, other datasets, or other reasoning domains.

**The consequence.** Several aspects of the findings could be model-specific or benchmark-specific. The PRM's quality and over-optimization behavior depend on PaLM 2-S*'s output distribution — a model with different calibration properties or different error patterns might exhibit different difficulty-dependent scaling curves. The revision model's ability to learn from incorrect in-context examples depends on the base model's in-context learning capabilities, which vary substantially across model families (e.g., GPT-4 vs. PaLM vs. LLaMA). The MATH benchmark consists exclusively of competition-level math problems requiring symbolic reasoning; it is unclear whether the difficulty-dependent patterns (beam search hurting easy problems, revisions helping easy problems, no method helping bin 5 problems) generalize to other reasoning domains — code generation (HumanEval, MBPP), logical reasoning (ARC, FOLIO), scientific QA, or tasks requiring factual knowledge rather than inference. For a practitioner working with, say, LLaMA-family models on code generation tasks, the paper provides no direct evidence that the compute-optimal framework transfers.

**What evidence exists in the paper.** All experiments in Sections 5–7 use MATH with PaLM 2-S*. Section 4 explicitly states the model choice and benchmark choice. No experiments use other base models or other reasoning benchmarks. The paper includes synthetic tasks (Selective Copying, Induction Heads) and mentions audio/genomics results using Mamba in Section 4, but these are different architectures evaluated on different tasks — they are not tests of the compute-optimal allocation framework that is the paper's core contribution. Section 8 does not discuss cross-model or cross-benchmark generalization as an open question.

**Mitigation status.** Not addressed. The paper provides no evidence beyond the single model + single benchmark combination. The finding that predicted difficulty bins work nearly as well as oracle bins (Figures 4, 8) is encouraging but was only demonstrated for PaLM 2-S* on MATH. Whether the PRM-based difficulty estimation transfers to other model families (where the PRM would need to be retrained on that model's outputs) or other benchmarks (where the distribution of difficulties may differ substantially) is unknown. This is the most consequential gap for practitioners considering adopting the approach: without replication on their specific model and task, the reported 4× efficiency gains are a point estimate with unknown generalization.

## 7. Implications and Future Directions
- How this changes the field
  - Establishes a credible, attention-free backbone that attains Transformer-level quality in language modeling while delivering linear-time scaling and high inference throughput (Figures 4, 8; Table 3). This reframes the design space: content-based reasoning does not require attention if a selective, input-dependent SSM is paired with the right systems implementation.

- Follow-up research directions
  - Scale studies:
    - Train Mamba at 7B–70B+ with strong data/compute recipes to test scaling laws and downstream abilities (Section 5).
  - Hybrid architectures:
    - Combine selective SSMs with attention or retrieval to mix content-aware recurrence with explicit non-local routing where helpful (Section E.2.2 shows Mamba+MHA is slightly better but not necessary).
  - Modality-tailored stacks:
    - For continuous signals, keep early layers LTI (Mamba-S4 or complex S4), add selectivity at higher, more symbolic levels (Figure 10).
  - Memory and state design:
    - Explore learned boundary resets, episodic memory, or expandable state mechanisms for even longer effective horizons (Section 3.5.2, “Boundary Resetting”).
  - Theory:
    - Extend the discretization–gating connection to richer families of selective dynamics; analyze stability and gradient flow of time-varying SSMs at scale (Theorem 1; Sections 3.5, 3.6).

- Practical applications
  - Long-document and streaming LMs with low latency and high throughput (constant-time per step without KV cache; Figure 8).
  - Genomics: pretraining over entire genomes and long-context tasks like regulatory element modeling and species classification up to 1M tokens (Figures 5–6; Table 13).
  - Audio: efficient long-context modeling and speech generation, with state-of-the-art automated metrics on SC09 (Figure 7; Table 4).
  - Any on-device or cost-sensitive deployment where linear-time training and fast recurrent inference are decisive.

Overall, the paper’s core contribution is twofold: it shows that “selectivity” (input-dependent SSM parameters, especially Δ) restores content-aware reasoning to linear-time models, and it makes this practical through a fused, IO-aware parallel scan. The resulting Mamba architecture is simple, fast, and competitive across diverse domains, with clear strengths on long-context and discrete modalities and a principled path for further scaling and hybridization.
